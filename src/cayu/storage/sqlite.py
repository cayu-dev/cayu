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
    RunRequest,
    Session,
    SessionStatus,
    SessionStore,
    copy_run_request,
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
        session_id = require_nonblank(session_id, "session_id")
        if type(event) is not Event:
            raise TypeError("Session events must be Event instances.")
        event = copy_event(event)
        if event.session_id != session_id:
            raise ValueError("Event session_id does not match target session.")

        async with self._lock:
            if not self._session_exists_unlocked(session_id):
                raise KeyError(f"Session not found: {session_id}")

            try:
                with self._connection:
                    self._connection.execute(
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
                        ),
                    )
            except sqlite3.IntegrityError as exc:
                if self._event_exists_unlocked(session_id, event.id):
                    raise ValueError(
                        f"Event already exists for session {session_id}: {event.id}"
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
                CREATE INDEX IF NOT EXISTS idx_events_session_sequence
                    ON events(session_id, sequence);
                CREATE INDEX IF NOT EXISTS idx_events_type_timestamp
                    ON events(event_type, timestamp);
                CREATE INDEX IF NOT EXISTS idx_events_agent_name
                    ON events(agent_name);
                CREATE INDEX IF NOT EXISTS idx_events_environment_name
                    ON events(environment_name);
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

    def _event_exists_unlocked(self, session_id: str, event_id: str) -> bool:
        row = self._connection.execute(
            "SELECT 1 FROM events WHERE session_id = ? AND event_id = ?",
            (session_id, event_id),
        ).fetchone()
        return row is not None


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
