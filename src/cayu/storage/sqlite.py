from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cayu._validation import copy_json_value, require_nonblank
from cayu.core.events import Event, copy_event
from cayu.runtime.sessions import (
    EventQuery,
    EventRecord,
    RunRequest,
    Session,
    SessionOrder,
    SessionQuery,
    SessionStatus,
    SessionStore,
    copy_event_query,
    copy_run_request,
    copy_session_query,
)


_SCHEMA_VERSION = 1


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

    async def create(self, request: RunRequest) -> Session:
        request = copy_run_request(request)
        async with self._lock:
            session = _session_from_request(request)
            try:
                with self._connection:
                    self._connection.execute(
                        """
                        INSERT INTO sessions (
                            id,
                            agent_name,
                            environment_name,
                            status,
                            created_at,
                            updated_at,
                            metadata_json
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            session.id,
                            session.agent_name,
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
                SELECT id, agent_name, environment_name, status, created_at,
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

        updated_at = datetime.now(timezone.utc)
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
                        "Event already exists for session "
                        f"{session_id}: {existing_event_id}"
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
                SELECT id, agent_name, environment_name, status, created_at,
                       updated_at, metadata_json
                FROM sessions
                {where_sql}
                ORDER BY {order_sql}, id ASC
                LIMIT ? OFFSET ?
                """,
                params,
            ).fetchall()
            return [_session_from_row(row) for row in rows]

    async def checkpoint(self, session_id: str, state: dict[str, Any]) -> None:
        session_id = require_nonblank(session_id, "session_id")
        if not isinstance(state, dict):
            raise ValueError("Checkpoint state must be a dictionary.")
        checkpoint = copy_json_value(state, "checkpoint")
        updated_at = datetime.now(timezone.utc)

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
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        if str(path) != ":memory:":
            connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize_schema(self) -> None:
        version = self._connection.execute("PRAGMA user_version").fetchone()[0]
        if version > _SCHEMA_VERSION:
            raise RuntimeError(
                "SQLiteSessionStore database was created by a newer Cayu schema "
                f"version: {version}"
            )

        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    agent_name TEXT NOT NULL,
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
                """
            )
            self._connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")

    def _load_unlocked(self, session_id: str) -> Session | None:
        row = self._connection.execute(
            """
            SELECT id, agent_name, environment_name, status, created_at,
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


def _session_from_request(request: RunRequest) -> Session:
    now = datetime.now(timezone.utc)
    values = {
        "agent_name": request.agent_name,
        "environment_name": request.environment_name,
        "status": SessionStatus.PENDING,
        "created_at": now,
        "updated_at": now,
        "metadata": copy_json_value(request.metadata, "metadata"),
    }
    if request.session_id is not None:
        values["id"] = request.session_id
    return Session(**values)


def _session_from_row(row: sqlite3.Row) -> Session:
    return Session(
        id=row["id"],
        agent_name=row["agent_name"],
        environment_name=row["environment_name"],
        status=SessionStatus(row["status"]),
        created_at=_parse_datetime(row["created_at"]),
        updated_at=_parse_datetime(row["updated_at"]),
        metadata=json.loads(row["metadata_json"]),
    )


def _format_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


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
