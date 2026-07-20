from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote
from uuid import uuid4

from cayu._validation import copy_json_value, copy_label_map
from cayu.core.events import Event
from cayu.runtime.sessions import (
    PENDING_ACTION_EVENT_TYPE_VALUES,
    PendingActionSession,
    RunRequest,
    Session,
    SessionIdentity,
    SessionOrder,
    SessionStatus,
)
from cayu.runtime.tasks import Task, TaskOrder, TaskStatus
from cayu.storage import _session_store_sql as session_store_sql
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
        _register_sqlite_functions(connection)
        return connection
    connection = sqlite3.connect(path, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    if str(path) != ":memory:":
        connection.execute("PRAGMA journal_mode = WAL")
    _register_sqlite_functions(connection)
    return connection


def _register_sqlite_functions(connection: sqlite3.Connection) -> None:
    from cayu.runtime.pending_actions import pending_action_lookup_key

    def lookup_key(value: object) -> str | None:
        return pending_action_lookup_key(value) if type(value) is str else None

    connection.create_function(
        "cayu_pending_action_lookup_key",
        1,
        lookup_key,
        deterministic=True,
    )
    connection.create_aggregate(
        "cayu_exact_usage_sum",
        11,
        cast("Any", _ExactUsageSum),
    )


class _ExactUsageSum:
    """Sum all normalized usage counters in one SQLite aggregate callback."""

    # A SQLite table has at most 2**63 - 1 rows and each accepted JSON integer is
    # at most 2**63 - 1, so every possible sum fits in 38 decimal digits.
    _DECIMAL_WIDTH = 38

    def __init__(self) -> None:
        self._totals = [0] * 11

    def step(self, *values: object) -> None:
        for index, value in enumerate(values):
            if type(value) is int and value >= 0:
                self._totals[index] += value

    def finalize(self) -> str:
        return json.dumps(
            [str(total).zfill(self._DECIMAL_WIDTH) for total in self._totals],
            separators=(",", ":"),
        )


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
        last_activity_at TEXT NOT NULL,
        run_epoch INTEGER NOT NULL DEFAULT 0,
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
        pending_action_lookup_key TEXT,
        pending_action_projection_json TEXT,
        pending_action_projection_bytes INTEGER,
        UNIQUE(session_id, event_id)
    );

    CREATE TABLE IF NOT EXISTS cayu_persisted_event_side_effects (
        session_id TEXT NOT NULL,
        event_id TEXT NOT NULL,
        event_sequence INTEGER NOT NULL,
        status TEXT NOT NULL,
        attempts INTEGER NOT NULL DEFAULT 0,
        claim_id TEXT,
        lease_expires_at TEXT,
        next_attempt_at TEXT,
        last_error TEXT,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (session_id, event_id),
        FOREIGN KEY (session_id, event_id)
            REFERENCES cayu_events(session_id, event_id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_cayu_persisted_event_side_effects_delivery
        ON cayu_persisted_event_side_effects(
            status, next_attempt_at, lease_expires_at, event_sequence
        );

    CREATE TRIGGER IF NOT EXISTS cayu_protect_undelivered_event_side_effects
    BEFORE DELETE ON cayu_events
    FOR EACH ROW
    WHEN EXISTS (
        SELECT 1
        FROM cayu_persisted_event_side_effects AS delivery
        WHERE delivery.session_id = OLD.session_id
          AND delivery.event_id = OLD.event_id
          AND delivery.status <> 'delivered'
    ) AND EXISTS (
        SELECT 1 FROM cayu_sessions WHERE id = OLD.session_id
    )
    BEGIN
        SELECT RAISE(IGNORE);
    END;

    CREATE TABLE IF NOT EXISTS cayu_session_labels (
        session_id TEXT NOT NULL REFERENCES cayu_sessions(id) ON DELETE CASCADE,
        key TEXT NOT NULL,
        value TEXT NOT NULL,
        PRIMARY KEY (session_id, key)
    );

    CREATE TABLE IF NOT EXISTS cayu_checkpoints (
        session_id TEXT PRIMARY KEY REFERENCES cayu_sessions(id) ON DELETE CASCADE,
        state_json TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        pending_action_source_bytes INTEGER,
        pending_action_tool_call_count INTEGER NOT NULL DEFAULT 0,
        pending_action_flags INTEGER NOT NULL DEFAULT 0,
        pending_action_metrics_ready INTEGER NOT NULL DEFAULT 1
    );

    CREATE TABLE IF NOT EXISTS cayu_session_operations (
        session_id TEXT NOT NULL REFERENCES cayu_sessions(id) ON DELETE CASCADE,
        idempotency_key TEXT NOT NULL,
        record_json TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (session_id, idempotency_key)
    );

    CREATE INDEX IF NOT EXISTS idx_cayu_checkpoints_pending_interruption_cascade
        ON cayu_checkpoints(session_id)
        WHERE json_type(state_json, '$.pending_interruption_cascade') IS NOT NULL;

    CREATE INDEX IF NOT EXISTS idx_cayu_checkpoints_pending_control_action
        ON cayu_checkpoints(session_id)
        WHERE pending_action_flags <> 0;

    CREATE TABLE IF NOT EXISTS cayu_transcript_messages (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL REFERENCES cayu_sessions(id) ON DELETE CASCADE,
        role TEXT NOT NULL,
        message_json TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS cayu_session_message_queue (
        ordering_key INTEGER PRIMARY KEY AUTOINCREMENT,
        queue_id TEXT NOT NULL UNIQUE,
        session_id TEXT NOT NULL REFERENCES cayu_sessions(id) ON DELETE CASCADE,
        idempotency_key TEXT NOT NULL,
        content TEXT NOT NULL,
        delivery_mode TEXT NOT NULL,
        status TEXT NOT NULL,
        requested_by_json TEXT,
        accepted_run_epoch INTEGER NOT NULL,
        accepted_transcript_cursor INTEGER NOT NULL,
        accepted_event_id TEXT NOT NULL,
        accepted_at TEXT NOT NULL,
        delivered_run_epoch INTEGER,
        delivered_transcript_cursor INTEGER,
        delivered_event_id TEXT,
        delivered_at TEXT,
        UNIQUE (session_id, idempotency_key)
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

    CREATE TABLE IF NOT EXISTS cayu_event_watcher_dead_letters (
        watcher_name TEXT NOT NULL,
        event_sequence INTEGER NOT NULL,
        event_id TEXT NOT NULL,
        attempts INTEGER NOT NULL,
        error TEXT NOT NULL,
        dead_lettered_at TEXT NOT NULL,
        resolved_at TEXT,
        PRIMARY KEY (watcher_name, event_sequence)
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
    CREATE INDEX IF NOT EXISTS idx_cayu_events_pending_action_barrier
        ON cayu_events(session_id, sequence)
        WHERE event_type = 'session.resumed'
           OR event_type = 'session.completed'
           OR event_type = 'session.failed';
    CREATE INDEX IF NOT EXISTS idx_cayu_events_pending_action_lookup
        ON cayu_events(
            session_id,
            pending_action_lookup_key,
            event_type,
            sequence
        )
        WHERE event_type IN (
            'tool.call.approval_requested',
            'session.awaiting_user_input',
            'session.interrupted',
            'tool.call.started',
            'tool.call.completed',
            'tool.call.failed',
            'tool.call.blocked',
            'tool.call.approval_denied'
        )
          AND pending_action_lookup_key IS NOT NULL;
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
    CREATE INDEX IF NOT EXISTS idx_cayu_session_message_queue_delivery
        ON cayu_session_message_queue(session_id, status, delivery_mode, ordering_key);
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
    CREATE INDEX IF NOT EXISTS idx_cayu_event_watcher_dead_letters_unresolved
        ON cayu_event_watcher_dead_letters(watcher_name, resolved_at, event_sequence);
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
    # The ADD COLUMN steps for revisions 4 and 5 live in _MIGRATION_ADD_COLUMNS
    # (applied idempotently before this DDL) because SQLite's ALTER TABLE ADD
    # COLUMN is not IF-NOT-EXISTS-guarded and would fail a re-run after a crash.
    4: """
        CREATE INDEX IF NOT EXISTS idx_cayu_tasks_worker_id
            ON cayu_tasks(worker_id);
        CREATE INDEX IF NOT EXISTS idx_cayu_tasks_status_lease
            ON cayu_tasks(status, lease_expires_at);
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
    8: """
        CREATE TABLE IF NOT EXISTS cayu_budget_reservations (
            reservation_id TEXT PRIMARY KEY,
            scope TEXT NOT NULL,
            budget_key TEXT,
            budget_window TEXT NOT NULL,
            currency TEXT NOT NULL,
            session_id TEXT NOT NULL,
            agent_name TEXT NOT NULL,
            provider_name TEXT NOT NULL,
            model TEXT NOT NULL,
            reserved_amount TEXT NOT NULL,
            actual_amount TEXT,
            status TEXT NOT NULL,
            reason TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_cayu_budget_reservations_scope
            ON cayu_budget_reservations(scope, budget_key, budget_window, currency, status);
    """,
    11: """
        CREATE TABLE IF NOT EXISTS cayu_event_watcher_dead_letters (
            watcher_name TEXT NOT NULL,
            event_sequence INTEGER NOT NULL,
            event_id TEXT NOT NULL,
            attempts INTEGER NOT NULL,
            error TEXT NOT NULL,
            dead_lettered_at TEXT NOT NULL,
            resolved_at TEXT,
            PRIMARY KEY (watcher_name, event_sequence)
        );

        CREATE INDEX IF NOT EXISTS idx_cayu_event_watcher_dead_letters_unresolved
            ON cayu_event_watcher_dead_letters(watcher_name, resolved_at, event_sequence);
    """,
    15: """
        CREATE INDEX IF NOT EXISTS idx_cayu_checkpoints_pending_interruption_cascade
            ON cayu_checkpoints(session_id)
            WHERE json_type(state_json, '$.pending_interruption_cascade') IS NOT NULL;
    """,
    17: """
        CREATE INDEX IF NOT EXISTS idx_cayu_checkpoints_pending_control_action
            ON cayu_checkpoints(session_id)
            WHERE pending_action_flags <> 0;

        CREATE INDEX IF NOT EXISTS idx_cayu_events_pending_action_barrier
            ON cayu_events(session_id, sequence)
            WHERE event_type = 'session.resumed'
               OR event_type = 'session.completed'
               OR event_type = 'session.failed';

        CREATE INDEX IF NOT EXISTS idx_cayu_events_pending_action_lookup
            ON cayu_events(
                session_id,
                pending_action_lookup_key,
                event_type,
                sequence
            )
            WHERE event_type IN (
                'tool.call.approval_requested',
                'session.awaiting_user_input',
                'session.interrupted',
                'tool.call.started',
                'tool.call.completed',
                'tool.call.failed',
                'tool.call.blocked',
                'tool.call.approval_denied'
            )
              AND pending_action_lookup_key IS NOT NULL;
    """,
    18: """
        CREATE TABLE IF NOT EXISTS cayu_session_operations (
            session_id TEXT NOT NULL REFERENCES cayu_sessions(id) ON DELETE CASCADE,
            idempotency_key TEXT NOT NULL,
            record_json TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (session_id, idempotency_key)
        );
    """,
    19: """
        CREATE TABLE IF NOT EXISTS cayu_session_message_queue (
            ordering_key INTEGER PRIMARY KEY AUTOINCREMENT,
            queue_id TEXT NOT NULL UNIQUE,
            session_id TEXT NOT NULL REFERENCES cayu_sessions(id) ON DELETE CASCADE,
            idempotency_key TEXT NOT NULL,
            content TEXT NOT NULL,
            delivery_mode TEXT NOT NULL,
            status TEXT NOT NULL,
            requested_by_json TEXT,
            accepted_run_epoch INTEGER NOT NULL,
            accepted_transcript_cursor INTEGER NOT NULL,
            accepted_event_id TEXT NOT NULL,
            accepted_at TEXT NOT NULL,
            delivered_run_epoch INTEGER,
            delivered_transcript_cursor INTEGER,
            delivered_event_id TEXT,
            delivered_at TEXT,
            UNIQUE (session_id, idempotency_key)
        );

        CREATE INDEX IF NOT EXISTS idx_cayu_session_message_queue_delivery
            ON cayu_session_message_queue(session_id, status, delivery_mode, ordering_key);
    """,
    20: """
        CREATE TABLE IF NOT EXISTS cayu_persisted_event_side_effects (
            session_id TEXT NOT NULL,
            event_id TEXT NOT NULL,
            event_sequence INTEGER NOT NULL,
            status TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            claim_id TEXT,
            lease_expires_at TEXT,
            next_attempt_at TEXT,
            last_error TEXT,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (session_id, event_id),
            FOREIGN KEY (session_id, event_id)
                REFERENCES cayu_events(session_id, event_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_cayu_persisted_event_side_effects_delivery
            ON cayu_persisted_event_side_effects(
                status, next_attempt_at, lease_expires_at, event_sequence
            );

        CREATE TRIGGER IF NOT EXISTS cayu_protect_undelivered_event_side_effects
        BEFORE DELETE ON cayu_events
        FOR EACH ROW
        WHEN EXISTS (
            SELECT 1
            FROM cayu_persisted_event_side_effects AS delivery
            WHERE delivery.session_id = OLD.session_id
              AND delivery.event_id = OLD.event_id
              AND delivery.status <> 'delivered'
        ) AND EXISTS (
            SELECT 1 FROM cayu_sessions WHERE id = OLD.session_id
        )
        BEGIN
            SELECT RAISE(IGNORE);
        END;

    """,
    21: "",
}

# Per-revision ``ALTER TABLE ADD COLUMN`` steps, keyed by revision. SQLite has no
# ``ADD COLUMN IF NOT EXISTS``, so these are applied via _add_column_if_missing
# (a table_info existence check) rather than raw DDL, making a re-run after a
# crash a no-op instead of a "duplicate column name" error that wedges migrate.
# They run before the revision's _MIGRATION_STEPS DDL so indexes on the new
# columns are created only after the columns exist.
_MIGRATION_ADD_COLUMNS: dict[int, tuple[tuple[str, str, str], ...]] = {
    4: (
        ("cayu_tasks", "worker_id", "TEXT"),
        ("cayu_tasks", "lease_expires_at", "TEXT"),
    ),
    5: (
        ("cayu_tasks", "status_reason", "TEXT"),
        ("cayu_tasks", "status_payload_json", "TEXT"),
    ),
    14: (
        (
            "cayu_sessions",
            "last_activity_at",
            "TEXT NOT NULL DEFAULT '1970-01-01T00:00:00+00:00'",
        ),
        ("cayu_sessions", "run_epoch", "INTEGER NOT NULL DEFAULT 0"),
    ),
    17: (
        ("cayu_events", "pending_action_lookup_key", "TEXT"),
        ("cayu_events", "pending_action_projection_json", "TEXT"),
        ("cayu_events", "pending_action_projection_bytes", "INTEGER"),
        ("cayu_checkpoints", "pending_action_source_bytes", "INTEGER"),
        (
            "cayu_checkpoints",
            "pending_action_tool_call_count",
            "INTEGER NOT NULL DEFAULT 0",
        ),
        ("cayu_checkpoints", "pending_action_flags", "INTEGER NOT NULL DEFAULT 0"),
        (
            "cayu_checkpoints",
            "pending_action_metrics_ready",
            "INTEGER NOT NULL DEFAULT 0",
        ),
    ),
}

# Per-revision ``ALTER TABLE DROP COLUMN`` steps, keyed by revision. Like the ADD
# steps, these are applied conditionally (via _drop_column_if_present) so that a
# fresh baseline (which never created the column) and a re-run after a crash are
# both no-ops rather than an "no such column" error that would wedge migrate.
# Revision 9 drops cayu_events.event_json: the full serialized Event duplicated
# what the individual indexed columns plus payload_json already carry, so it was
# pure write amplification and unbounded storage growth. The store now
# reconstructs Events from those columns.
_MIGRATION_DROP_COLUMNS: dict[int, tuple[tuple[str, str], ...]] = {
    9: (("cayu_events", "event_json"),),
}


def _migrate_legacy_budget_reservations(connection: sqlite3.Connection) -> None:
    """Carry rows from the pre-revision-8 ad-hoc ``budget_reservations`` table.

    Before revision 8 the SQLite budget ledger created an unprefixed
    ``budget_reservations`` table outside the migration machinery. When such a
    legacy table exists, copy its rows into ``cayu_budget_reservations`` and drop
    it so active reservations survive the rename.
    """
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'budget_reservations'"
    ).fetchone()
    if exists is None:
        return
    connection.execute(
        """
        INSERT OR IGNORE INTO cayu_budget_reservations (
            reservation_id, scope, budget_key, budget_window, currency, session_id,
            agent_name, provider_name, model, reserved_amount, actual_amount,
            status, reason, created_at, updated_at
        )
        SELECT reservation_id, scope, budget_key, window, currency, session_id,
               agent_name, provider_name, model, reserved_amount, actual_amount,
               status, reason, created_at, updated_at
        FROM budget_reservations
        """
    )
    connection.execute("DROP TABLE budget_reservations")


def _backfill_session_activity(connection: sqlite3.Connection) -> None:
    connection.execute("UPDATE cayu_sessions SET last_activity_at = updated_at")


def _backfill_pending_action_checkpoint_batch(
    connection: sqlite3.Connection,
    after_session_id: str | None,
) -> str | None:
    from cayu.runtime.pending_actions import (
        pending_action_checkpoint_metrics,
    )

    rows = connection.execute(
        "SELECT session_id FROM cayu_checkpoints "
        "WHERE pending_action_metrics_ready = 0 AND (? IS NULL OR session_id > ?) "
        "ORDER BY session_id LIMIT 100",
        (after_session_id, after_session_id),
    ).fetchall()
    if not rows:
        return None
    for row in rows:
        checkpoint_row = connection.execute(
            "SELECT state_json FROM cayu_checkpoints WHERE session_id = ?",
            (row["session_id"],),
        ).fetchone()
        if checkpoint_row is None:  # pragma: no cover - this transaction holds the writer lock.
            continue
        source_bytes, tool_call_count, flags = pending_action_checkpoint_metrics(
            json.loads(checkpoint_row["state_json"])
        )
        connection.execute(
            "UPDATE cayu_checkpoints SET pending_action_source_bytes = ?, "
            "pending_action_tool_call_count = ?, pending_action_flags = ?, "
            "pending_action_metrics_ready = 1 WHERE session_id = ?",
            (source_bytes, tool_call_count, flags, row["session_id"]),
        )
        del checkpoint_row
    return str(rows[-1]["session_id"])


def _backfill_pending_action_event_batch(
    connection: sqlite3.Connection,
    after_sequence: int,
) -> int | None:
    from cayu.runtime.pending_actions import (
        PENDING_ACTION_EVENT_TYPE_VALUES,
        pending_action_event_storage_values,
    )

    event_types = sorted(PENDING_ACTION_EVENT_TYPE_VALUES)
    placeholders = ", ".join("?" for _ in event_types)
    sequence_rows = connection.execute(
        f"""
        SELECT sequence
        FROM cayu_events
        WHERE pending_action_projection_bytes IS NULL
          AND sequence > ?
          AND event_type IN ({placeholders})
        ORDER BY sequence
        LIMIT 25
        """,
        (after_sequence, *event_types),
    ).fetchall()
    if not sequence_rows:
        return None
    for sequence_row in sequence_rows:
        row = connection.execute(
            """
            SELECT sequence, session_id, event_id, event_type, timestamp,
                   agent_name, environment_name, workflow_name, tool_name, payload_json
            FROM cayu_events
            WHERE sequence = ?
            """,
            (sequence_row["sequence"],),
        ).fetchone()
        if row is None:  # pragma: no cover - this transaction holds the writer lock.
            continue
        event = Event(
            session_id=row["session_id"],
            id=row["event_id"],
            type=row["event_type"],
            timestamp=parse_datetime(row["timestamp"]),
            agent_name=row["agent_name"],
            environment_name=row["environment_name"],
            workflow_name=row["workflow_name"],
            tool_name=row["tool_name"],
            payload=json.loads(row["payload_json"]),
        )
        lookup_key, projection, projection_bytes = pending_action_event_storage_values(event)
        connection.execute(
            "UPDATE cayu_events SET pending_action_lookup_key = ?, "
            "pending_action_projection_json = ?, pending_action_projection_bytes = ? "
            "WHERE sequence = ?",
            (
                lookup_key,
                projection,
                projection_bytes,
                row["sequence"],
            ),
        )
        # Do not retain one arbitrary-size legacy payload while loading the next.
        del event, lookup_key, projection, projection_bytes, row
    return int(sequence_rows[-1]["sequence"])


def _add_budget_billing_identity_if_present(connection: sqlite3.Connection) -> None:
    """Add revision-21 evidence when this database owns a budget ledger table."""

    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'cayu_budget_reservations'"
    ).fetchone()
    if exists is not None:
        _add_column_if_missing(
            connection,
            "cayu_budget_reservations",
            "billing_identity_json",
            "TEXT",
        )


# Per-revision Python follow-ups that cannot be expressed as unconditional DDL
# (e.g. conditionally carrying data out of a legacy ad-hoc table). Each hook runs
# after its revision's DDL and before the revision is recorded.
_MIGRATION_HOOKS: dict[int, Callable[[sqlite3.Connection], None]] = {
    8: _migrate_legacy_budget_reservations,
    14: _backfill_session_activity,
    21: _add_budget_billing_identity_if_present,
}

_REVISION_17_INDEX_NAMES = frozenset(
    {
        "idx_cayu_checkpoints_pending_control_action",
        "idx_cayu_events_pending_action_barrier",
        "idx_cayu_events_pending_action_lookup",
    }
)


def _normalize_sqlite_schema_definition(definition: str) -> str:
    """Normalize formatting, while preserving every structural SQL token."""
    normalized = re.sub(r"\s+", "", definition.casefold())
    normalized = normalized.replace('"', "").replace("`", "").replace("[", "").replace("]", "")
    return normalized.replace("ifnotexists", "")


def _revision_17_index_definitions() -> dict[str, str]:
    definitions: dict[str, str] = {}
    for statement in _iter_statements(_MIGRATION_STEPS[17]):
        match = re.match(
            r"CREATE\s+INDEX\s+(?:IF\s+NOT\s+EXISTS\s+)?([^\s(]+)",
            statement,
            flags=re.IGNORECASE,
        )
        if match is not None and match.group(1) in _REVISION_17_INDEX_NAMES:
            definitions[match.group(1)] = statement
    if definitions.keys() != _REVISION_17_INDEX_NAMES:
        raise RuntimeError("Cayu revision 17 index definitions are incomplete.")
    return definitions


def _validate_revision_17_indexes(
    connection: sqlite3.Connection,
    *,
    require_all: bool,
) -> None:
    """Reject same-name SQLite indexes whose structure is not Cayu's contract."""
    for index_name, expected in _revision_17_index_definitions().items():
        row = connection.execute(
            "SELECT type, tbl_name, sql FROM sqlite_master WHERE name = ?",
            (index_name,),
        ).fetchone()
        if row is None:
            if require_all:
                raise RuntimeError(
                    f"Required Cayu SQLite index is missing: {index_name}. "
                    "Run with schema_mode='migrate' to repair the schema."
                )
            continue
        actual_type, _table_name, actual_definition = row
        if (
            actual_type != "index"
            or actual_definition is None
            or (
                _normalize_sqlite_schema_definition(actual_definition)
                != _normalize_sqlite_schema_definition(expected)
            )
        ):
            raise RuntimeError(
                f"SQLite schema object {index_name!r} conflicts with Cayu revision 17. "
                "Rename or remove the conflicting object, then run with "
                "schema_mode='migrate' to create the required index."
            )


def _repair_missing_revision_17_indexes(connection: sqlite3.Connection) -> None:
    """Recreate missing required indexes even when revision 17 is already recorded."""
    with _transaction(connection):
        _validate_revision_17_indexes(connection, require_all=False)
        existing_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            ).fetchall()
        }
        for index_name, definition in _revision_17_index_definitions().items():
            if index_name not in existing_names:
                connection.execute(definition)
        _validate_revision_17_indexes(connection, require_all=True)


def reconcile_schema(
    connection: sqlite3.Connection,
    schema_mode: schema.SchemaMode = schema.SchemaMode.CREATE,
    *,
    app_min_supported: int = schema.MIN_SUPPORTED_REVISION,
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
        schema.validate(state, app_min_supported=app_min_supported)
    elif schema_mode is schema.SchemaMode.CREATE:
        if state.revision == schema.UNINITIALIZED:
            _apply_pending(connection, state)
        else:
            schema.validate(state, app_min_supported=app_min_supported)
    else:  # MIGRATE
        _apply_pending(connection, state)
        schema.validate(
            read_schema_state(connection),
            app_min_supported=app_min_supported,
        )
    current = read_schema_state(connection)
    if current.revision >= 17:
        if schema_mode is schema.SchemaMode.MIGRATE:
            _repair_missing_revision_17_indexes(connection)
        else:
            _validate_revision_17_indexes(connection, require_all=True)


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


@contextmanager
def _transaction(connection: sqlite3.Connection) -> Iterator[None]:
    """Run a block inside one explicit SQLite transaction (BEGIN/COMMIT/ROLLBACK).

    Most revisions apply DDL, their data hook, and their revision marker
    atomically. Large revision-17 backfills instead use this helper for short,
    independently committed batches and explicit ready markers, making a crash
    resumable without holding one write lock for the entire data set.

    ``executescript`` cannot be used here: it force-commits any open transaction,
    so revision DDL is executed statement-by-statement.
    """
    connection.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        connection.rollback()
        raise
    else:
        connection.commit()


def _iter_statements(script: str) -> Iterator[str]:
    """Yield complete statements while preserving trigger bodies and literals."""
    pending: list[str] = []
    for line in script.splitlines(keepends=True):
        pending.append(line)
        statement = "".join(pending).strip()
        if statement and sqlite3.complete_statement(statement):
            yield statement.removesuffix(";").rstrip()
            pending.clear()
    trailing = "".join(pending).strip()
    if trailing:
        raise ValueError("SQLite migration DDL ended with an incomplete statement")


def _add_column_if_missing(
    connection: sqlite3.Connection, table: str, column: str, decl: str
) -> None:
    """Idempotently ``ALTER TABLE ... ADD COLUMN`` (SQLite lacks IF NOT EXISTS)."""
    existing = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


def _drop_column_if_present(connection: sqlite3.Connection, table: str, column: str) -> None:
    """Idempotently ``ALTER TABLE ... DROP COLUMN`` (SQLite lacks IF EXISTS)."""
    existing = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
    if column in existing:
        connection.execute(f"ALTER TABLE {table} DROP COLUMN {column}")


def _apply_baseline(connection: sqlite3.Connection) -> None:
    with _transaction(connection):
        for statement in _iter_statements(_BASELINE_DDL):
            connection.execute(statement)
        _record_revision(connection, schema.revision(schema.BASELINE_REVISION))
        # user_version mirrors the revision as a cheap SQLite-native marker; the
        # cayu_schema_migrations table remains the cross-backend source of truth.
        connection.execute(f"PRAGMA user_version = {schema.BASELINE_REVISION}")


def _apply_pending(connection: sqlite3.Connection, state: schema.SchemaState) -> None:
    current = state.revision
    if current == schema.UNINITIALIZED:
        _apply_baseline(connection)
        current = schema.BASELINE_REVISION
    for rev in schema.pending(current):
        _apply_revision(connection, rev)


def _apply_revision(connection: sqlite3.Connection, rev: schema.Revision) -> None:
    if rev.revision == 17:
        _apply_revision_seventeen(connection, rev)
        return
    with _transaction(connection):
        for table, column, decl in _MIGRATION_ADD_COLUMNS.get(rev.revision, ()):
            _add_column_if_missing(connection, table, column, decl)
        for table, column in _MIGRATION_DROP_COLUMNS.get(rev.revision, ()):
            _drop_column_if_present(connection, table, column)
        ddl = _MIGRATION_STEPS.get(rev.revision)
        if ddl:
            for statement in _iter_statements(ddl):
                connection.execute(statement)
        hook = _MIGRATION_HOOKS.get(rev.revision)
        if hook is not None:
            hook(connection)
        _record_revision(connection, rev)
        connection.execute(f"PRAGMA user_version = {rev.revision}")


def _apply_revision_seventeen(
    connection: sqlite3.Connection,
    rev: schema.Revision,
) -> None:
    # CREATE INDEX IF NOT EXISTS silently accepts a wrong same-name index.
    # Validate before any staged work so a conflict cannot be followed by a
    # falsely recorded successful migration.
    with _transaction(connection):
        _validate_revision_17_indexes(connection, require_all=False)
        for table, column, decl in _MIGRATION_ADD_COLUMNS[17]:
            _add_column_if_missing(connection, table, column, decl)
        for statement in _iter_statements(_MIGRATION_STEPS[17]):
            connection.execute(statement)

    after_session_id: str | None = None
    while True:
        with _transaction(connection):
            next_session_id = _backfill_pending_action_checkpoint_batch(
                connection,
                after_session_id,
            )
            checkpoint_remaining = (
                next_session_id is None
                and connection.execute(
                    "SELECT EXISTS(SELECT 1 FROM cayu_checkpoints "
                    "WHERE pending_action_metrics_ready = 0)"
                ).fetchone()[0]
                == 1
            )
        if next_session_id is not None:
            after_session_id = next_session_id
            continue
        if not checkpoint_remaining:
            break
        after_session_id = None

    after_sequence = 0
    event_types = sorted(PENDING_ACTION_EVENT_TYPE_VALUES)
    event_type_placeholders = ", ".join("?" for _ in event_types)
    while True:
        with _transaction(connection):
            next_sequence = _backfill_pending_action_event_batch(connection, after_sequence)
            event_remaining = (
                next_sequence is None
                and connection.execute(
                    "SELECT EXISTS(SELECT 1 FROM cayu_events "
                    "WHERE pending_action_projection_bytes IS NULL "
                    f"AND event_type IN ({event_type_placeholders}))",
                    event_types,
                ).fetchone()[0]
                == 1
            )
        if next_sequence is not None:
            after_sequence = next_sequence
            continue
        if not event_remaining:
            break
        after_sequence = 0

    with _transaction(connection):
        _validate_revision_17_indexes(connection, require_all=True)
        _record_revision(connection, rev)
        connection.execute(f"PRAGMA user_version = {rev.revision}")


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
        last_activity_at=now,
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
        format_datetime(session.last_activity_at),
        session.run_epoch,
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
        last_activity_at=parse_datetime(row["last_activity_at"]),
        run_epoch=row["run_epoch"],
        metadata=json.loads(row["metadata_json"]),
        labels=copy_label_map(labels, "labels"),
    )


def pending_action_session_from_row(
    row: sqlite3.Row,
    labels: dict[str, str] | None = None,
) -> PendingActionSession:
    return PendingActionSession(
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


def checkpoint_row_values(
    session_id: str,
    checkpoint: dict[str, Any],
    updated_at: datetime,
) -> tuple[object, ...]:
    from cayu.runtime.pending_actions import pending_action_checkpoint_metrics

    source_bytes, tool_call_count, flags = pending_action_checkpoint_metrics(checkpoint)
    return (
        session_id,
        json_dumps(checkpoint),
        format_datetime(updated_at),
        source_bytes,
        tool_call_count,
        flags,
        1,
    )


def session_order_sql(order_by: SessionOrder) -> str:
    return session_store_sql.session_order_sql(order_by)


def task_order_sql(order_by: TaskOrder) -> str:
    if order_by == TaskOrder.CREATED_AT_ASC:
        return "created_at ASC"
    if order_by == TaskOrder.CREATED_AT_DESC:
        return "created_at DESC"
    if order_by == TaskOrder.UPDATED_AT_ASC:
        return "updated_at ASC"
    return "updated_at DESC"
