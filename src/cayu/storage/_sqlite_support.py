from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote
from uuid import uuid4

from cayu._validation import copy_json_value, copy_label_map
from cayu.runtime.sessions import (
    RunRequest,
    Session,
    SessionIdentity,
    SessionOrder,
    SessionStatus,
    session_order_is_descending,
    session_sort_column,
)
from cayu.runtime.tasks import Task, TaskOrder, TaskStatus
from cayu.storage import migrations as schema


def connect(path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    if str(path) == ":memory:":
        if read_only:
            raise ValueError("Read-only connections require a file-backed SQLite database.")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    if read_only:
        # A dedicated read-only connection lets queries run in worker threads
        # without contending with the writer connection's transactions (WAL
        # readers never block on the writer). query_only guards against any
        # accidental write slipping onto the read path.
        uri = f"file:{quote(str(path.resolve()), safe='/')}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA query_only = ON")
        return connection
    connection = sqlite3.connect(path, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    if str(path) != ":memory:":
        connection.execute("PRAGMA journal_mode = WAL")
    return connection


# Baseline-revision (ADR 0001 revision 1) DDL. Every table carries the cayu_ prefix
# (Decision 5) so Cayu state never collides with an app's own tables. The
# cayu_schema_migrations bookkeeping table is created separately by the migrator.
_BASELINE_DDL = """
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
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        metadata_json TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS cayu_events (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL REFERENCES cayu_sessions(id) ON DELETE CASCADE,
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

    CREATE TABLE IF NOT EXISTS cayu_session_labels (
        session_id TEXT NOT NULL REFERENCES cayu_sessions(id) ON DELETE CASCADE,
        key TEXT NOT NULL,
        value TEXT NOT NULL,
        PRIMARY KEY (session_id, key)
    );

    CREATE TABLE IF NOT EXISTS cayu_checkpoints (
        session_id TEXT PRIMARY KEY REFERENCES cayu_sessions(id) ON DELETE CASCADE,
        state_json TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS cayu_transcript_messages (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL REFERENCES cayu_sessions(id) ON DELETE CASCADE,
        role TEXT NOT NULL,
        message_json TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS cayu_tasks (
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

    CREATE TABLE IF NOT EXISTS cayu_event_watcher_state (
        watcher_name TEXT PRIMARY KEY,
        cursor_sequence INTEGER NOT NULL,
        pending_event_id TEXT,
        pending_event_sequence INTEGER,
        pending_attempt INTEGER NOT NULL,
        pending_claim_id TEXT,
        delivery_status TEXT,
        lease_expires_at TEXT,
        last_error TEXT,
        dead_lettered_count INTEGER NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_cayu_sessions_status
        ON cayu_sessions(status);
    CREATE INDEX IF NOT EXISTS idx_cayu_sessions_agent_name
        ON cayu_sessions(agent_name);
    CREATE INDEX IF NOT EXISTS idx_cayu_sessions_environment_name
        ON cayu_sessions(environment_name);
    CREATE INDEX IF NOT EXISTS idx_cayu_sessions_causal_budget_id
        ON cayu_sessions(causal_budget_id);
    CREATE INDEX IF NOT EXISTS idx_cayu_session_labels_key_value_session
        ON cayu_session_labels(key, value, session_id);
    CREATE INDEX IF NOT EXISTS idx_cayu_events_session_sequence
        ON cayu_events(session_id, sequence);
    CREATE INDEX IF NOT EXISTS idx_cayu_events_type_timestamp
        ON cayu_events(event_type, timestamp);
    CREATE INDEX IF NOT EXISTS idx_cayu_events_agent_name
        ON cayu_events(agent_name);
    CREATE INDEX IF NOT EXISTS idx_cayu_events_environment_name
        ON cayu_events(environment_name);
    CREATE INDEX IF NOT EXISTS idx_cayu_events_workflow_name
        ON cayu_events(workflow_name);
    CREATE INDEX IF NOT EXISTS idx_cayu_events_tool_name
        ON cayu_events(tool_name);
    CREATE INDEX IF NOT EXISTS idx_cayu_transcript_messages_session_sequence
        ON cayu_transcript_messages(session_id, sequence);
    CREATE INDEX IF NOT EXISTS idx_cayu_transcript_messages_session_role_sequence
        ON cayu_transcript_messages(session_id, role, sequence);
    CREATE INDEX IF NOT EXISTS idx_cayu_tasks_status
        ON cayu_tasks(status);
    CREATE INDEX IF NOT EXISTS idx_cayu_tasks_type
        ON cayu_tasks(type);
    CREATE INDEX IF NOT EXISTS idx_cayu_tasks_session_id
        ON cayu_tasks(session_id);
    CREATE INDEX IF NOT EXISTS idx_cayu_tasks_parent_task_id
        ON cayu_tasks(parent_task_id);
    CREATE INDEX IF NOT EXISTS idx_cayu_tasks_assigned_agent_name
        ON cayu_tasks(assigned_agent_name);
    CREATE INDEX IF NOT EXISTS idx_cayu_event_watcher_state_delivery
        ON cayu_event_watcher_state(delivery_status, lease_expires_at);
"""

# Bookkeeping table created/owned by the migrator (separate from a revision's DDL).
_MIGRATIONS_TABLE_DDL = """
    CREATE TABLE IF NOT EXISTS cayu_schema_migrations (
        revision INTEGER PRIMARY KEY,
        kind TEXT NOT NULL,
        compatible_from INTEGER NOT NULL,
        checksum TEXT,
        applied_at TEXT NOT NULL
    )
"""

# Per-revision forward-migration DDL, keyed by revision number. The baseline
# (revision 1) is applied from _BASELINE_DDL, so it is not listed here; future
# additive/breaking revisions append their ALTER/CREATE scripts.
_MIGRATION_STEPS: dict[int, str] = {
    2: """
        CREATE TABLE IF NOT EXISTS cayu_session_labels (
            session_id TEXT NOT NULL REFERENCES cayu_sessions(id) ON DELETE CASCADE,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            PRIMARY KEY (session_id, key)
        );

        CREATE INDEX IF NOT EXISTS idx_cayu_session_labels_key_value_session
            ON cayu_session_labels(key, value, session_id);
    """,
    3: """
        CREATE TABLE IF NOT EXISTS cayu_event_watcher_state (
            watcher_name TEXT PRIMARY KEY,
            cursor_sequence INTEGER NOT NULL,
            pending_event_id TEXT,
            pending_event_sequence INTEGER,
            pending_attempt INTEGER NOT NULL,
            pending_claim_id TEXT,
            delivery_status TEXT,
            lease_expires_at TEXT,
            last_error TEXT,
            dead_lettered_count INTEGER NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_cayu_event_watcher_state_delivery
            ON cayu_event_watcher_state(delivery_status, lease_expires_at);
    """,
    4: """
        ALTER TABLE cayu_tasks ADD COLUMN worker_id TEXT;
        ALTER TABLE cayu_tasks ADD COLUMN lease_expires_at TEXT;

        CREATE INDEX IF NOT EXISTS idx_cayu_tasks_worker_id
            ON cayu_tasks(worker_id);
        CREATE INDEX IF NOT EXISTS idx_cayu_tasks_status_lease
            ON cayu_tasks(status, lease_expires_at);
    """,
    5: """
        ALTER TABLE cayu_tasks ADD COLUMN status_reason TEXT;
        ALTER TABLE cayu_tasks ADD COLUMN status_payload_json TEXT;
    """,
    6: """
        CREATE TABLE IF NOT EXISTS cayu_knowledge_entries (
            id TEXT PRIMARY KEY,
            namespace TEXT NOT NULL,
            text TEXT NOT NULL,
            kind TEXT NOT NULL,
            visibility TEXT NOT NULL,
            status TEXT NOT NULL,
            created_by_type TEXT NOT NULL,
            created_by TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            source_type TEXT,
            source_uri TEXT,
            source_id TEXT,
            source_hash TEXT,
            importance REAL,
            importance_source TEXT,
            confidence REAL,
            last_used_at TEXT,
            expires_at TEXT,
            title TEXT,
            metadata_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS cayu_knowledge_labels (
            entry_id TEXT NOT NULL REFERENCES cayu_knowledge_entries(id) ON DELETE CASCADE,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            PRIMARY KEY (entry_id, key)
        );

        CREATE TABLE IF NOT EXISTS cayu_knowledge_aspects (
            entry_id TEXT NOT NULL REFERENCES cayu_knowledge_entries(id) ON DELETE CASCADE,
            aspect TEXT NOT NULL,
            PRIMARY KEY (entry_id, aspect)
        );

        CREATE TABLE IF NOT EXISTS cayu_knowledge_impact_targets (
            entry_id TEXT NOT NULL REFERENCES cayu_knowledge_entries(id) ON DELETE CASCADE,
            impact_target TEXT NOT NULL,
            PRIMARY KEY (entry_id, impact_target)
        );

        CREATE TABLE IF NOT EXISTS cayu_knowledge_chunks (
            id TEXT PRIMARY KEY,
            entry_id TEXT NOT NULL REFERENCES cayu_knowledge_entries(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            content_hash TEXT,
            source_uri TEXT,
            metadata_json TEXT NOT NULL,
            UNIQUE (entry_id, chunk_index)
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS cayu_knowledge_chunks_fts
        USING fts5(entry_id UNINDEXED, chunk_id UNINDEXED, title, text);

        CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_entries_namespace_status
            ON cayu_knowledge_entries(namespace, status);
        CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_entries_kind
            ON cayu_knowledge_entries(kind);
        CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_entries_visibility
            ON cayu_knowledge_entries(visibility);
        CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_entries_source
            ON cayu_knowledge_entries(source_type, source_id);
        CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_entries_expires_at
            ON cayu_knowledge_entries(expires_at);
        CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_labels_key_value_entry
            ON cayu_knowledge_labels(key, value, entry_id);
        CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_aspects_aspect_entry
            ON cayu_knowledge_aspects(aspect, entry_id);
        CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_impact_targets_target_entry
            ON cayu_knowledge_impact_targets(impact_target, entry_id);
        CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_chunks_entry_index
            ON cayu_knowledge_chunks(entry_id, chunk_index);
    """,
}


def reconcile_schema(
    connection: sqlite3.Connection,
    schema_mode: schema.SchemaMode = schema.SchemaMode.CREATE,
) -> None:
    """Reconcile the SQLite schema with this binary per ``schema_mode`` (ADR 0001).

    SQLite's single writer plus ``PRAGMA busy_timeout`` provides the cross-process
    coordination that the Postgres backend gets from an advisory lock.

    - ``validate``: read the recorded revision and fail fast unless this binary can
      operate against it. Never runs DDL.
    - ``create``: initialize the baseline schema on an empty database; otherwise
      validate. The default for SQLite (dev / test / local durability).
    - ``migrate``: apply pending forward revisions, then validate.
    """
    if schema_mode is not schema.SchemaMode.VALIDATE:
        connection.execute(_MIGRATIONS_TABLE_DDL)
        connection.commit()
    state = read_schema_state(connection)
    if schema_mode is schema.SchemaMode.VALIDATE:
        schema.validate(state)
    elif schema_mode is schema.SchemaMode.CREATE:
        if state.revision == schema.UNINITIALIZED:
            _apply_pending(connection, state)
        else:
            schema.validate(state)
    else:  # MIGRATE
        _apply_pending(connection, state)
        schema.validate(read_schema_state(connection))


def initialize_schema(connection: sqlite3.Connection) -> None:
    reconcile_schema(connection, schema.SchemaMode.CREATE)


def read_schema_state(connection: sqlite3.Connection) -> schema.SchemaState:
    """Read the recorded schema state without applying DDL or failing fast."""
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'cayu_schema_migrations'"
    ).fetchone()
    if exists is None:
        return schema.SchemaState(revision=schema.UNINITIALIZED, compatible_from=0)
    row = connection.execute(
        "SELECT revision, compatible_from FROM cayu_schema_migrations "
        "ORDER BY revision DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return schema.SchemaState(revision=schema.UNINITIALIZED, compatible_from=0)
    return schema.SchemaState(revision=row[0], compatible_from=row[1])


def _apply_baseline(connection: sqlite3.Connection) -> None:
    connection.executescript(_BASELINE_DDL)
    _record_revision(connection, schema.revision(schema.BASELINE_REVISION))
    # user_version mirrors the revision as a cheap SQLite-native marker; the
    # cayu_schema_migrations table remains the cross-backend source of truth.
    connection.execute(f"PRAGMA user_version = {schema.BASELINE_REVISION}")
    connection.commit()


def _apply_pending(connection: sqlite3.Connection, state: schema.SchemaState) -> None:
    current = state.revision
    if current == schema.UNINITIALIZED:
        _apply_baseline(connection)
        current = schema.BASELINE_REVISION
    for rev in schema.pending(current):
        ddl = _MIGRATION_STEPS.get(rev.revision)
        if ddl:
            connection.executescript(ddl)
        _record_revision(connection, rev)
        connection.execute(f"PRAGMA user_version = {rev.revision}")
        connection.commit()


def _record_revision(connection: sqlite3.Connection, rev: schema.Revision) -> None:
    connection.execute(
        "INSERT OR IGNORE INTO cayu_schema_migrations "
        "(revision, kind, compatible_from, checksum, applied_at) VALUES (?, ?, ?, ?, ?)",
        (
            rev.revision,
            str(rev.kind),
            rev.compatible_from,
            None,
            format_datetime(datetime.now(UTC)),
        ),
    )


def session_from_request(request: RunRequest, *, identity: SessionIdentity) -> Session:
    now = datetime.now(UTC)
    session_id = request.session_id if request.session_id is not None else str(uuid4())
    return Session(
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
        labels=copy_label_map(request.labels, "labels"),
    )


def session_to_row_values(session: Session) -> tuple[object, ...]:
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
        format_datetime(session.created_at),
        format_datetime(session.updated_at),
        json_dumps(session.metadata),
    )


def session_label_row_values(session: Session) -> list[tuple[str, str, str]]:
    return [(session.id, key, value) for key, value in sorted(session.labels.items())]


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
        task.worker_id,
        format_optional_datetime(task.lease_expires_at),
        task.status_reason,
        None if task.status_payload is None else json_dumps(task.status_payload),
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
    status_payload_json = row["status_payload_json"]
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
        worker_id=row["worker_id"],
        lease_expires_at=parse_optional_datetime(row["lease_expires_at"]),
        status_reason=row["status_reason"],
        status_payload=(None if status_payload_json is None else json.loads(status_payload_json)),
        input=json.loads(row["input_json"]),
        result=None if result_json is None else json.loads(result_json),
        error=None if error_json is None else json.loads(error_json),
        metadata=json.loads(row["metadata_json"]),
        created_at=parse_datetime(row["created_at"]),
        updated_at=parse_datetime(row["updated_at"]),
        started_at=parse_optional_datetime(row["started_at"]),
        completed_at=parse_optional_datetime(row["completed_at"]),
    )


def session_from_row(row: sqlite3.Row, labels: dict[str, str] | None = None) -> Session:
    return Session(
        id=row["id"],
        agent_name=row["agent_name"],
        provider_name=row["provider_name"],
        model=row["model"],
        parent_session_id=row["parent_session_id"],
        causal_budget_id=row["causal_budget_id"],
        runtime_name=row["runtime_name"],
        runtime_version=row["runtime_version"],
        environment_name=row["environment_name"],
        status=SessionStatus(row["status"]),
        created_at=parse_datetime(row["created_at"]),
        updated_at=parse_datetime(row["updated_at"]),
        metadata=json.loads(row["metadata_json"]),
        labels=copy_label_map(labels, "labels"),
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
    # Derived from the same helpers the keyset cursor uses, so the ORDER BY direction
    # can never drift from the cursor comparison.
    direction = "DESC" if session_order_is_descending(order_by) else "ASC"
    return f"{session_sort_column(order_by)} {direction}"


def task_order_sql(order_by: TaskOrder) -> str:
    if order_by == TaskOrder.CREATED_AT_ASC:
        return "created_at ASC"
    if order_by == TaskOrder.CREATED_AT_DESC:
        return "created_at DESC"
    if order_by == TaskOrder.UPDATED_AT_ASC:
        return "updated_at ASC"
    return "updated_at DESC"
