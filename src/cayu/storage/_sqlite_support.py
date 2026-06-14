from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from cayu._validation import copy_json_value
from cayu.runtime.sessions import (
    RunRequest,
    Session,
    SessionIdentity,
    SessionOrder,
    SessionStatus,
)
from cayu.runtime.tasks import Task, TaskOrder, TaskStatus

SCHEMA_VERSION = 6


def connect(path: Path) -> sqlite3.Connection:
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    if str(path) != ":memory:":
        connection.execute("PRAGMA journal_mode = WAL")
    return connection


def initialize_schema(connection: sqlite3.Connection) -> None:
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version > SCHEMA_VERSION:
        raise RuntimeError(
            f"SQLite store database was created by a newer Cayu schema version: {version}"
        )
    if version == 0 and _has_user_tables(connection):
        raise RuntimeError(
            "SQLite store database has existing tables but no Cayu schema version. "
            "Recreate the database with the current Cayu version."
        )
    if version not in (0, SCHEMA_VERSION):
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
                parent_session_id TEXT REFERENCES sessions(id) ON DELETE SET NULL,
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
                role TEXT NOT NULL,
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
            CREATE INDEX IF NOT EXISTS idx_transcript_messages_session_role_sequence
                ON transcript_messages(session_id, role, sequence);
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
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


def session_from_request(request: RunRequest, *, identity: SessionIdentity) -> Session:
    now = datetime.now(UTC)
    return Session(
        id=request.session_id if request.session_id is not None else str(uuid4()),
        agent_name=request.agent_name,
        provider_name=identity.provider_name,
        model=identity.model,
        runtime_name=identity.runtime_name,
        runtime_version=identity.runtime_version,
        environment_name=request.environment_name,
        status=SessionStatus.PENDING,
        created_at=now,
        updated_at=now,
        metadata=copy_json_value(request.metadata, "metadata"),
    )


def session_to_row_values(session: Session) -> tuple[object, ...]:
    return (
        session.id,
        session.agent_name,
        session.provider_name,
        session.model,
        session.parent_session_id,
        session.runtime_name,
        session.runtime_version,
        session.environment_name,
        str(session.status),
        format_datetime(session.created_at),
        format_datetime(session.updated_at),
        json_dumps(session.metadata),
    )


def task_to_row_values(task: Task) -> tuple[object, ...]:
    return (
        task.id,
        task.type,
        task.title,
        task.description,
        str(task.status),
        task.session_id,
        task.parent_task_id,
        task.assigned_agent_name,
        json_dumps(task.input),
        None if task.result is None else json_dumps(task.result),
        None if task.error is None else json_dumps(task.error),
        json_dumps(task.metadata),
        format_datetime(task.created_at),
        format_datetime(task.updated_at),
        format_optional_datetime(task.started_at),
        format_optional_datetime(task.completed_at),
    )


def task_from_row(row: sqlite3.Row) -> Task:
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
        created_at=parse_datetime(row["created_at"]),
        updated_at=parse_datetime(row["updated_at"]),
        started_at=parse_optional_datetime(row["started_at"]),
        completed_at=parse_optional_datetime(row["completed_at"]),
    )


def session_from_row(row: sqlite3.Row) -> Session:
    return Session(
        id=row["id"],
        agent_name=row["agent_name"],
        provider_name=row["provider_name"],
        model=row["model"],
        parent_session_id=row["parent_session_id"],
        runtime_name=row["runtime_name"],
        runtime_version=row["runtime_version"],
        environment_name=row["environment_name"],
        status=SessionStatus(row["status"]),
        created_at=parse_datetime(row["created_at"]),
        updated_at=parse_datetime(row["updated_at"]),
        metadata=json.loads(row["metadata_json"]),
    )


def format_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def format_optional_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    return format_datetime(value)


def parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def parse_optional_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    return parse_datetime(value)


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def session_order_sql(order_by: SessionOrder) -> str:
    if order_by == SessionOrder.CREATED_AT_ASC:
        return "created_at ASC"
    if order_by == SessionOrder.CREATED_AT_DESC:
        return "created_at DESC"
    if order_by == SessionOrder.UPDATED_AT_ASC:
        return "updated_at ASC"
    return "updated_at DESC"


def task_order_sql(order_by: TaskOrder) -> str:
    if order_by == TaskOrder.CREATED_AT_ASC:
        return "created_at ASC"
    if order_by == TaskOrder.CREATED_AT_DESC:
        return "created_at DESC"
    if order_by == TaskOrder.UPDATED_AT_ASC:
        return "updated_at ASC"
    return "updated_at DESC"


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
