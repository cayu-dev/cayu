from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from cayu.runtime.sessions import (
    Session,
    SessionOrder,
    SessionStatus,
)
from cayu.runtime.tasks import Task, TaskOrder, TaskStatus

# Postgres schema mirrors the SQLite store (both at ADR 0001 baseline revision 1)
# but uses Postgres-native types: TEXT ids, JSONB payloads, TIMESTAMPTZ times,
# a global BIGINT identity event cursor, and a per-session monotonic order column.
# All tables carry the cayu_ prefix (ADR 0001 Decision 5) so Cayu state never
# collides with an application's own tables in a shared database. This tuple is the
# baseline-revision DDL (ADR 0001 revision 1); the cayu_schema_migrations
# bookkeeping table is created separately by the migrator.
SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS cayu_sessions (
        id TEXT PRIMARY KEY,
        agent_name TEXT NOT NULL,
        provider_name TEXT NOT NULL,
        model TEXT NOT NULL,
        parent_session_id TEXT REFERENCES cayu_sessions(id) ON DELETE SET NULL,
        causal_budget_id TEXT NOT NULL,
        runtime_name TEXT NOT NULL,
        runtime_version TEXT,
        environment_name TEXT,
        status TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL,
        metadata JSONB NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cayu_events (
        sequence BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES cayu_sessions(id) ON DELETE CASCADE,
        session_order BIGINT NOT NULL,
        event_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        timestamp TIMESTAMPTZ NOT NULL,
        agent_name TEXT,
        environment_name TEXT,
        workflow_name TEXT,
        tool_name TEXT,
        payload JSONB NOT NULL,
        event JSONB NOT NULL,
        UNIQUE (session_id, event_id),
        UNIQUE (session_id, session_order)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cayu_checkpoints (
        session_id TEXT PRIMARY KEY REFERENCES cayu_sessions(id) ON DELETE CASCADE,
        state JSONB NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cayu_transcript_messages (
        sequence BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES cayu_sessions(id) ON DELETE CASCADE,
        message JSONB NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cayu_tasks (
        id TEXT PRIMARY KEY,
        type TEXT NOT NULL,
        title TEXT,
        description TEXT,
        status TEXT NOT NULL,
        session_id TEXT,
        parent_task_id TEXT,
        assigned_agent_name TEXT,
        input JSONB NOT NULL,
        result JSONB,
        error JSONB,
        metadata JSONB NOT NULL,
        created_at TIMESTAMPTZ NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL,
        started_at TIMESTAMPTZ,
        completed_at TIMESTAMPTZ
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_cayu_sessions_status ON cayu_sessions(status)",
    "CREATE INDEX IF NOT EXISTS idx_cayu_sessions_agent_name ON cayu_sessions(agent_name)",
    "CREATE INDEX IF NOT EXISTS idx_cayu_sessions_environment_name "
    "ON cayu_sessions(environment_name)",
    "CREATE INDEX IF NOT EXISTS idx_cayu_sessions_causal_budget_id "
    "ON cayu_sessions(causal_budget_id)",
    "CREATE INDEX IF NOT EXISTS idx_cayu_events_session_order "
    "ON cayu_events(session_id, session_order)",
    "CREATE INDEX IF NOT EXISTS idx_cayu_events_type_timestamp ON cayu_events(event_type, timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_cayu_events_agent_name ON cayu_events(agent_name)",
    "CREATE INDEX IF NOT EXISTS idx_cayu_events_environment_name ON cayu_events(environment_name)",
    "CREATE INDEX IF NOT EXISTS idx_cayu_events_workflow_name ON cayu_events(workflow_name)",
    "CREATE INDEX IF NOT EXISTS idx_cayu_events_tool_name ON cayu_events(tool_name)",
    "CREATE INDEX IF NOT EXISTS idx_cayu_transcript_messages_session_sequence "
    "ON cayu_transcript_messages(session_id, sequence)",
    "CREATE INDEX IF NOT EXISTS idx_cayu_tasks_status ON cayu_tasks(status)",
    "CREATE INDEX IF NOT EXISTS idx_cayu_tasks_type ON cayu_tasks(type)",
    "CREATE INDEX IF NOT EXISTS idx_cayu_tasks_session_id ON cayu_tasks(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_cayu_tasks_parent_task_id ON cayu_tasks(parent_task_id)",
    "CREATE INDEX IF NOT EXISTS idx_cayu_tasks_assigned_agent_name "
    "ON cayu_tasks(assigned_agent_name)",
)

# Bookkeeping table created/owned by the migrator (separate from a revision's DDL).
MIGRATIONS_TABLE_DDL = """
    CREATE TABLE IF NOT EXISTS cayu_schema_migrations (
        revision INTEGER PRIMARY KEY,
        kind TEXT NOT NULL,
        compatible_from INTEGER NOT NULL,
        checksum TEXT,
        applied_at TIMESTAMPTZ NOT NULL
    )
"""


def to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def to_utc_optional(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return to_utc(value)


def session_insert_values(session: Session) -> tuple[object, ...]:
    return (
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
        to_utc(session.created_at),
        to_utc(session.updated_at),
        _dumps(session.metadata),
    )


def session_from_row(row: tuple[Any, ...]) -> Session:
    return Session(
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
        created_at=to_utc(row[10]),
        updated_at=to_utc(row[11]),
        metadata=_loads(row[12]),
    )


SESSION_COLUMNS = (
    "id, agent_name, provider_name, model, parent_session_id, causal_budget_id, "
    "runtime_name, runtime_version, environment_name, status, created_at, updated_at, "
    "metadata"
)


def task_insert_values(task: Task) -> tuple[object, ...]:
    return (
        task.id,
        task.type,
        task.title,
        task.description,
        str(task.status),
        task.session_id,
        task.parent_task_id,
        task.assigned_agent_name,
        _dumps(task.input),
        None if task.result is None else _dumps(task.result),
        None if task.error is None else _dumps(task.error),
        _dumps(task.metadata),
        to_utc(task.created_at),
        to_utc(task.updated_at),
        to_utc_optional(task.started_at),
        to_utc_optional(task.completed_at),
    )


TASK_COLUMNS = (
    "id, type, title, description, status, session_id, parent_task_id, "
    "assigned_agent_name, input, result, error, metadata, created_at, "
    "updated_at, started_at, completed_at"
)


def task_from_row(row: tuple[Any, ...]) -> Task:
    return Task(
        id=row[0],
        type=row[1],
        title=row[2],
        description=row[3],
        status=TaskStatus(row[4]),
        session_id=row[5],
        parent_task_id=row[6],
        assigned_agent_name=row[7],
        input=_loads(row[8]),
        result=None if row[9] is None else _loads(row[9]),
        error=None if row[10] is None else _loads(row[10]),
        metadata=_loads(row[11]),
        created_at=to_utc(row[12]),
        updated_at=to_utc(row[13]),
        started_at=to_utc_optional(row[14]),
        completed_at=to_utc_optional(row[15]),
    )


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


def _dumps(value: Any) -> str:
    # JSONB columns accept a JSON-text string; we serialize explicitly so the
    # same json round-trip semantics as the SQLite store are preserved.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _loads(value: Any) -> Any:
    # psycopg returns JSONB as already-decoded Python objects, but we accept a
    # JSON string too for robustness across configurations.
    if isinstance(value, str):
        return json.loads(value)
    return value
