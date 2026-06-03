from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cayu._validation import copy_json_object, copy_json_value, require_nonblank
from cayu.core.events import Event, copy_event
from cayu.core.messages import Message
from cayu.runtime.sessions import (
    EventQuery,
    EventRecord,
    RunRequest,
    Session,
    SessionIdentity,
    SessionOrder,
    SessionQuery,
    SessionStatus,
    SessionStore,
    _validate_status_set,
    copy_event_query,
    copy_run_request,
    copy_session_identity,
    copy_session_query,
    copy_transcript_messages,
)
from cayu.runtime.tasks import (
    Task,
    TaskCreate,
    TaskOrder,
    TaskQuery,
    TaskStatus,
    TaskStore,
    _ensure_can_transition,
    _task_from_create,
    copy_task_create,
    copy_task_query,
)

_SCHEMA_VERSION = 4


class SQLiteSessionStore(SessionStore):
    """SQLite-backed session store for durable local runtime state."""

    def __init__(self, path: str | Path) -> None:
        if isinstance(path, Path):
            db_path = path
        elif type(path) is str:
            db_path = Path(require_nonblank(path, "path"))
        else:
            raise TypeError("SQLiteSessionStore path must be a string or Path.")

        self.path = db_path
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
            session = _session_from_request(request, identity=identity)
            try:
                with self._connection:
                    self._connection.execute(
                        """
                        INSERT INTO sessions (
                            id,
                            agent_name,
                            provider_name,
                            model,
                            runtime_name,
                            runtime_version,
                            environment_name,
                            status,
                            created_at,
                            updated_at,
                            metadata_json
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            session.id,
                            session.agent_name,
                            session.provider_name,
                            session.model,
                            session.runtime_name,
                            session.runtime_version,
                            session.environment_name,
                            str(session.status),
                            _format_datetime(session.created_at),
                            _format_datetime(session.updated_at),
                            _json_dumps(session.metadata),
                        ),
                    )
            except sqlite3.IntegrityError as exc:
                if self._session_exists_unlocked(session.id):
                    raise ValueError(f"Session already exists: {session.id}") from exc
                raise
            return session.model_copy(deep=True)

    async def load(self, session_id: str) -> Session | None:
        session_id = require_nonblank(session_id, "session_id")
        async with self._lock:
            row = self._connection.execute(
                """
                SELECT id, agent_name, provider_name, model, runtime_name,
                       runtime_version, environment_name, status, created_at,
                       updated_at, metadata_json
                FROM sessions
                WHERE id = ?
                """,
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            return _session_from_row(row)

    async def update_status(self, session_id: str, status: SessionStatus) -> Session:
        session_id = require_nonblank(session_id, "session_id")
        if not isinstance(status, SessionStatus):
            raise ValueError("Session status must be a SessionStatus.")

        updated_at = datetime.now(UTC)
        async with self._lock:
            with self._connection:
                cursor = self._connection.execute(
                    """
                    UPDATE sessions
                    SET status = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (str(status), _format_datetime(updated_at), session_id),
                )
            if cursor.rowcount != 1:
                raise KeyError(f"Session not found: {session_id}")

            loaded = self._load_unlocked(session_id)
            if loaded is None:
                raise KeyError(f"Session not found: {session_id}")
            return loaded

    async def update_model(self, session_id: str, model: str) -> Session:
        session_id = require_nonblank(session_id, "session_id")
        model = require_nonblank(model, "model")
        updated_at = datetime.now(UTC)
        async with self._lock:
            with self._connection:
                cursor = self._connection.execute(
                    """
                    UPDATE sessions
                    SET model = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (model, _format_datetime(updated_at), session_id),
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
        session_id = require_nonblank(session_id, "session_id")
        allowed_statuses = _validate_status_set(from_statuses, "from_statuses")
        if not isinstance(to_status, SessionStatus):
            raise ValueError("to_status must be a SessionStatus.")

        updated_at = datetime.now(UTC)
        async with self._lock:
            placeholders = ", ".join("?" for _ in allowed_statuses)
            params: list[object] = [
                str(to_status),
                _format_datetime(updated_at),
                session_id,
                *[str(status) for status in allowed_statuses],
            ]
            with self._connection:
                cursor = self._connection.execute(
                    f"""
                    UPDATE sessions
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

    async def append_event(self, session_id: str, event: Event) -> None:
        await self.append_events(session_id, [event])

    async def append_events(self, session_id: str, events: list[Event]) -> None:
        session_id = require_nonblank(session_id, "session_id")
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
                        INSERT INTO events (
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
                                _format_datetime(event.timestamp),
                                event.agent_name,
                                event.environment_name,
                                event.workflow_name,
                                event.tool_name,
                                _json_dumps(event.payload),
                                _json_dumps(event.model_dump(mode="json")),
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
        session_id = require_nonblank(session_id, "session_id")
        async with self._lock:
            if not self._session_exists_unlocked(session_id):
                raise KeyError(f"Session not found: {session_id}")
            rows = self._connection.execute(
                """
                SELECT event_json
                FROM events
                WHERE session_id = ?
                ORDER BY sequence ASC
                """,
                (session_id,),
            ).fetchall()
            return [Event(**json.loads(row["event_json"])) for row in rows]

    async def query_events(self, query: EventQuery | None = None) -> list[EventRecord]:
        query = copy_event_query(query)
        clauses: list[str] = []
        params: list[object] = []

        if query.after_sequence is not None:
            clauses.append("sequence > ?")
            params.append(query.after_sequence)
        if query.session_id is not None:
            clauses.append("session_id = ?")
            params.append(query.session_id)
        if query.event_type is not None:
            clauses.append("event_type = ?")
            params.append(str(query.event_type))
        if query.agent_name is not None:
            clauses.append("agent_name = ?")
            params.append(query.agent_name)
        if query.environment_name is not None:
            clauses.append("environment_name = ?")
            params.append(query.environment_name)
        if query.workflow_name is not None:
            clauses.append("workflow_name = ?")
            params.append(query.workflow_name)
        if query.tool_name is not None:
            clauses.append("tool_name = ?")
            params.append(query.tool_name)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(query.limit)

        async with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT sequence, event_json
                FROM events
                {where_sql}
                ORDER BY sequence ASC
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

    async def list_sessions(self, query: SessionQuery | None = None) -> list[Session]:
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

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        order_sql = _session_order_sql(query.order_by)
        params.extend([query.limit, query.offset])

        async with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT id, agent_name, provider_name, model, runtime_name,
                       runtime_version, environment_name, status, created_at,
                       updated_at, metadata_json
                FROM sessions
                {where_sql}
                ORDER BY {order_sql}, id ASC
                LIMIT ? OFFSET ?
                """,
                params,
            ).fetchall()
            return [_session_from_row(row) for row in rows]

    async def append_transcript_messages(
        self,
        session_id: str,
        messages: list[Message],
    ) -> None:
        session_id = require_nonblank(session_id, "session_id")
        copied_messages = copy_transcript_messages(messages)

        async with self._lock:
            if not self._session_exists_unlocked(session_id):
                raise KeyError(f"Session not found: {session_id}")
            if not copied_messages:
                return
            with self._connection:
                self._connection.executemany(
                    """
                    INSERT INTO transcript_messages (
                        session_id,
                        message_json
                    )
                    VALUES (?, ?)
                    """,
                    [
                        (
                            session_id,
                            _json_dumps(message.model_dump(mode="json")),
                        )
                        for message in copied_messages
                    ],
                )

    async def load_transcript(self, session_id: str) -> list[Message]:
        session_id = require_nonblank(session_id, "session_id")
        async with self._lock:
            if not self._session_exists_unlocked(session_id):
                raise KeyError(f"Session not found: {session_id}")
            rows = self._connection.execute(
                """
                SELECT message_json
                FROM transcript_messages
                WHERE session_id = ?
                ORDER BY sequence ASC
                """,
                (session_id,),
            ).fetchall()
            return [Message(**json.loads(row["message_json"])) for row in rows]

    async def checkpoint(self, session_id: str, state: dict[str, Any]) -> None:
        session_id = require_nonblank(session_id, "session_id")
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
                    INSERT INTO checkpoints (session_id, state_json, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        state_json = excluded.state_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        session_id,
                        _json_dumps(checkpoint),
                        _format_datetime(updated_at),
                    ),
                )

    async def load_checkpoint(self, session_id: str) -> dict[str, Any] | None:
        session_id = require_nonblank(session_id, "session_id")
        async with self._lock:
            row = self._connection.execute(
                """
                SELECT state_json
                FROM checkpoints
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
        return _connect(path)

    def _initialize_schema(self) -> None:
        _initialize_schema(self._connection)

    def _load_unlocked(self, session_id: str) -> Session | None:
        row = self._connection.execute(
            """
            SELECT id, agent_name, provider_name, model, runtime_name,
                   runtime_version, environment_name, status, created_at,
                   updated_at, metadata_json
            FROM sessions
            WHERE id = ?
            """,
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return _session_from_row(row)

    def _session_exists_unlocked(self, session_id: str) -> bool:
        row = self._connection.execute(
            "SELECT 1 FROM sessions WHERE id = ?",
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
                "SELECT 1 FROM events WHERE session_id = ? AND event_id = ?",
                (session_id, event_id),
            ).fetchone()
            if row is not None:
                return event_id
        return None


class SQLiteTaskStore(TaskStore):
    """SQLite-backed task store for durable local work items."""

    def __init__(self, path: str | Path) -> None:
        if isinstance(path, Path):
            db_path = path
        elif type(path) is str:
            db_path = Path(require_nonblank(path, "path"))
        else:
            raise TypeError("SQLiteTaskStore path must be a string or Path.")

        self.path = db_path
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
                        INSERT INTO tasks (
                            id,
                            type,
                            title,
                            description,
                            status,
                            session_id,
                            parent_task_id,
                            assigned_agent_name,
                            input_json,
                            result_json,
                            error_json,
                            metadata_json,
                            created_at,
                            updated_at,
                            started_at,
                            completed_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        _task_to_row_values(task),
                    )
            except sqlite3.IntegrityError as exc:
                if self._task_exists_unlocked(task.id):
                    raise ValueError(f"Task already exists: {task.id}") from exc
                raise
            return task.model_copy(deep=True)

    async def load_task(self, task_id: str) -> Task | None:
        task_id = require_nonblank(task_id, "task_id")
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
        order_sql = _task_order_sql(query.order_by)
        params.extend([query.limit, query.offset])

        async with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT *
                FROM tasks
                {where_sql}
                ORDER BY {order_sql}, id ASC
                LIMIT ? OFFSET ?
                """,
                params,
            ).fetchall()
            return [_task_from_row(row) for row in rows]

    async def start_task(
        self,
        task_id: str,
        *,
        session_id: str | None = None,
    ) -> Task:
        task_id = require_nonblank(task_id, "task_id")
        if session_id is not None:
            session_id = require_nonblank(session_id, "session_id")
        async with self._lock:
            now = datetime.now(UTC)
            with self._connection:
                cursor = self._connection.execute(
                    """
                    UPDATE tasks
                    SET status = ?,
                        session_id = COALESCE(?, session_id),
                        started_at = COALESCE(started_at, ?),
                        updated_at = ?
                    WHERE id = ? AND status = ?
                    """,
                    (
                        str(TaskStatus.RUNNING),
                        session_id,
                        _format_datetime(now),
                        _format_datetime(now),
                        task_id,
                        str(TaskStatus.PENDING),
                    ),
                )
            if cursor.rowcount != 1:
                task = self._require_task_unlocked(task_id)
                _ensure_can_transition(task, TaskStatus.RUNNING)
                raise ValueError(f"Task {task.id} cannot transition to running from {task.status}")
            updated = self._require_task_unlocked(task_id)
            return updated.model_copy(deep=True)

    async def complete_task(self, task_id: str, result: dict[str, Any]) -> Task:
        task_id = require_nonblank(task_id, "task_id")
        result = copy_json_object(result, "result")
        async with self._lock:
            return self._finish_task_unlocked(
                task_id,
                TaskStatus.COMPLETED,
                result=result,
                error=None,
            )

    async def fail_task(self, task_id: str, error: dict[str, Any]) -> Task:
        task_id = require_nonblank(task_id, "task_id")
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
        task_id = require_nonblank(task_id, "task_id")
        copied_error = None if error is None else copy_json_object(error, "error")
        async with self._lock:
            return self._finish_task_unlocked(
                task_id,
                TaskStatus.CANCELLED,
                result=None,
                error=copied_error,
            )

    async def close(self) -> None:
        async with self._lock:
            self._connection.close()

    def _connect(self, path: Path) -> sqlite3.Connection:
        return _connect(path)

    def _initialize_schema(self) -> None:
        _initialize_schema(self._connection)

    def _load_task_unlocked(self, task_id: str) -> Task | None:
        row = self._connection.execute(
            "SELECT * FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if row is None:
            return None
        return _task_from_row(row)

    def _require_task_unlocked(self, task_id: str) -> Task:
        task = self._load_task_unlocked(task_id)
        if task is None:
            raise KeyError(f"Task not found: {task_id}")
        return task

    def _task_exists_unlocked(self, task_id: str) -> bool:
        row = self._connection.execute(
            "SELECT 1 FROM tasks WHERE id = ?",
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
                UPDATE tasks
                SET status = ?,
                    result_json = ?,
                    error_json = ?,
                    started_at = COALESCE(started_at, ?),
                    completed_at = ?,
                    updated_at = ?
                WHERE id = ?
                  AND status NOT IN (?, ?, ?)
                """,
                (
                    str(status),
                    None if result is None else _json_dumps(result),
                    None if error is None else _json_dumps(error),
                    _format_datetime(now),
                    _format_datetime(now),
                    _format_datetime(now),
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


def _connect(path: Path) -> sqlite3.Connection:
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    if str(path) != ":memory:":
        connection.execute("PRAGMA journal_mode = WAL")
    return connection


def _initialize_schema(connection: sqlite3.Connection) -> None:
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version > _SCHEMA_VERSION:
        raise RuntimeError(
            f"SQLite store database was created by a newer Cayu schema version: {version}"
        )
    if version == 0 and _has_user_tables(connection):
        raise RuntimeError(
            "SQLite store database has existing tables but no Cayu schema version. "
            "Recreate the database with the current Cayu version."
        )
    if version not in (0, _SCHEMA_VERSION):
        raise RuntimeError(
            "SQLite store database uses an unsupported pre-public Cayu schema "
            f"version: {version}. Recreate the database with the current Cayu version."
        )

    with connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                agent_name TEXT NOT NULL,
                provider_name TEXT NOT NULL,
                model TEXT NOT NULL,
                runtime_name TEXT NOT NULL,
                runtime_version TEXT,
                environment_name TEXT,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                event_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                agent_name TEXT,
                environment_name TEXT,
                workflow_name TEXT,
                tool_name TEXT,
                payload_json TEXT NOT NULL,
                event_json TEXT NOT NULL,
                UNIQUE(session_id, event_id)
            );

            CREATE TABLE IF NOT EXISTS checkpoints (
                session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
                state_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS transcript_messages (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                message_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                title TEXT,
                description TEXT,
                status TEXT NOT NULL,
                session_id TEXT,
                parent_task_id TEXT,
                assigned_agent_name TEXT,
                input_json TEXT NOT NULL,
                result_json TEXT,
                error_json TEXT,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_status
                ON sessions(status);
            CREATE INDEX IF NOT EXISTS idx_sessions_agent_name
                ON sessions(agent_name);
            CREATE INDEX IF NOT EXISTS idx_sessions_environment_name
                ON sessions(environment_name);
            CREATE INDEX IF NOT EXISTS idx_events_session_sequence
                ON events(session_id, sequence);
            CREATE INDEX IF NOT EXISTS idx_events_type_timestamp
                ON events(event_type, timestamp);
            CREATE INDEX IF NOT EXISTS idx_events_agent_name
                ON events(agent_name);
            CREATE INDEX IF NOT EXISTS idx_events_environment_name
                ON events(environment_name);
            CREATE INDEX IF NOT EXISTS idx_events_workflow_name
                ON events(workflow_name);
            CREATE INDEX IF NOT EXISTS idx_events_tool_name
                ON events(tool_name);
            CREATE INDEX IF NOT EXISTS idx_transcript_messages_session_sequence
                ON transcript_messages(session_id, sequence);
            CREATE INDEX IF NOT EXISTS idx_tasks_status
                ON tasks(status);
            CREATE INDEX IF NOT EXISTS idx_tasks_type
                ON tasks(type);
            CREATE INDEX IF NOT EXISTS idx_tasks_session_id
                ON tasks(session_id);
            CREATE INDEX IF NOT EXISTS idx_tasks_parent_task_id
                ON tasks(parent_task_id);
            CREATE INDEX IF NOT EXISTS idx_tasks_assigned_agent_name
                ON tasks(assigned_agent_name);
            """
        )
        connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")


def _has_user_tables(connection: sqlite3.Connection) -> bool:
    row = connection.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table'
          AND name NOT LIKE 'sqlite_%'
        LIMIT 1
        """
    ).fetchone()
    return row is not None


def _session_from_request(request: RunRequest, *, identity: SessionIdentity) -> Session:
    now = datetime.now(UTC)
    values = {
        "agent_name": request.agent_name,
        "provider_name": identity.provider_name,
        "model": identity.model,
        "runtime_name": identity.runtime_name,
        "runtime_version": identity.runtime_version,
        "environment_name": request.environment_name,
        "status": SessionStatus.PENDING,
        "created_at": now,
        "updated_at": now,
        "metadata": copy_json_value(request.metadata, "metadata"),
    }
    if request.session_id is not None:
        values["id"] = request.session_id
    return Session(**values)


def _task_to_row_values(task: Task) -> tuple[object, ...]:
    return (
        task.id,
        task.type,
        task.title,
        task.description,
        str(task.status),
        task.session_id,
        task.parent_task_id,
        task.assigned_agent_name,
        _json_dumps(task.input),
        None if task.result is None else _json_dumps(task.result),
        None if task.error is None else _json_dumps(task.error),
        _json_dumps(task.metadata),
        _format_datetime(task.created_at),
        _format_datetime(task.updated_at),
        _format_optional_datetime(task.started_at),
        _format_optional_datetime(task.completed_at),
    )


def _task_from_row(row: sqlite3.Row) -> Task:
    result_json = row["result_json"]
    error_json = row["error_json"]
    return Task(
        id=row["id"],
        type=row["type"],
        title=row["title"],
        description=row["description"],
        status=TaskStatus(row["status"]),
        session_id=row["session_id"],
        parent_task_id=row["parent_task_id"],
        assigned_agent_name=row["assigned_agent_name"],
        input=json.loads(row["input_json"]),
        result=None if result_json is None else json.loads(result_json),
        error=None if error_json is None else json.loads(error_json),
        metadata=json.loads(row["metadata_json"]),
        created_at=_parse_datetime(row["created_at"]),
        updated_at=_parse_datetime(row["updated_at"]),
        started_at=_parse_optional_datetime(row["started_at"]),
        completed_at=_parse_optional_datetime(row["completed_at"]),
    )


def _session_from_row(row: sqlite3.Row) -> Session:
    return Session(
        id=row["id"],
        agent_name=row["agent_name"],
        provider_name=row["provider_name"],
        model=row["model"],
        runtime_name=row["runtime_name"],
        runtime_version=row["runtime_version"],
        environment_name=row["environment_name"],
        status=SessionStatus(row["status"]),
        created_at=_parse_datetime(row["created_at"]),
        updated_at=_parse_datetime(row["updated_at"]),
        metadata=json.loads(row["metadata_json"]),
    )


def _format_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def _format_optional_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    return _format_datetime(value)


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_optional_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    return _parse_datetime(value)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _session_order_sql(order_by: SessionOrder) -> str:
    if order_by == SessionOrder.CREATED_AT_ASC:
        return "created_at ASC"
    if order_by == SessionOrder.CREATED_AT_DESC:
        return "created_at DESC"
    if order_by == SessionOrder.UPDATED_AT_ASC:
        return "updated_at ASC"
    return "updated_at DESC"


def _task_order_sql(order_by: TaskOrder) -> str:
    if order_by == TaskOrder.CREATED_AT_ASC:
        return "created_at ASC"
    if order_by == TaskOrder.CREATED_AT_DESC:
        return "created_at DESC"
    if order_by == TaskOrder.UPDATED_AT_ASC:
        return "updated_at ASC"
    return "updated_at DESC"
