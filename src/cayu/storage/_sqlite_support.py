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

from cayu._validation import (
    copy_durable_json_object,
    copy_label_map,
    require_clean_nonblank,
    require_execution_unit_id,
)
from cayu.core.events import Event
from cayu.runtime.invocation import SessionInvocation, TaskInvocation
from cayu.runtime.sessions import (
    PENDING_ACTION_EVENT_TYPE_VALUES,
    PendingActionSession,
    RunRequest,
    Session,
    SessionIdentity,
    SessionOrder,
    SessionStatus,
    session_invocation_for_run_request,
    session_metadata_for_creation,
)
from cayu.runtime.tasks import (
    TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES,
    TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES,
    Task,
    TaskOrder,
    TaskStatus,
    TaskTopologyInconsistent,
    TaskTopologyNode,
)
from cayu.storage import _session_store_sql as session_store_sql
from cayu.storage import migrations as schema


def connect(path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    if str(path) == ":memory:":
        if read_only:
            raise ValueError("Read-only connections require a file-backed SQLite database.")
    elif not read_only:
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

    def is_clean_nonblank_text(value: object) -> int:
        if type(value) is not str:
            return 0
        try:
            require_clean_nonblank(value, "value")
        except ValueError:
            return 0
        return 1

    def is_execution_unit_id(value: object, field_name: object) -> int:
        if type(value) is not str or type(field_name) is not str:
            return 0
        try:
            require_execution_unit_id(value, field_name)
        except (TypeError, ValueError):
            return 0
        return 1

    connection.create_function(
        "cayu_pending_action_lookup_key",
        1,
        lookup_key,
        deterministic=True,
    )
    connection.create_function(
        "cayu_is_clean_nonblank_text",
        1,
        is_clean_nonblank_text,
        deterministic=True,
    )
    connection.create_function(
        "cayu_is_execution_unit_id",
        2,
        is_execution_unit_id,
        deterministic=True,
    )
    connection.create_aggregate(
        "cayu_exact_usage_sum",
        11,
        cast("Any", _ExactUsageSum),
    )


class _ExactUsageSum:
    """Sum normalized counters and canonical outputs from a prior exact sum."""

    # A SQLite table has at most 2**63 - 1 rows and each accepted JSON integer is
    # at most 2**63 - 1, so every possible sum fits in 38 decimal digits.
    _DECIMAL_WIDTH = 38

    def __init__(self) -> None:
        self._totals = [0] * 11

    def step(self, *values: object) -> None:
        for index, value in enumerate(values):
            if type(value) is int and value >= 0:
                self._totals[index] += value
            elif (
                type(value) is str
                and len(value) == self._DECIMAL_WIDTH
                and value.isascii()
                and value.isdecimal()
            ):
                self._totals[index] += int(value)

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
        transcript_seq INTEGER NOT NULL DEFAULT 0,
        invocation_json TEXT NOT NULL,
        metadata_json TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS cayu_events (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL REFERENCES cayu_sessions(id) ON DELETE CASCADE,
        event_id TEXT NOT NULL,
        interaction_id TEXT,
        event_type TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        agent_name TEXT,
        environment_name TEXT,
        workflow_name TEXT,
        tool_name TEXT,
        payload_json TEXT NOT NULL,
        input_contract_runtime_owned INTEGER NOT NULL DEFAULT 0
            CHECK (input_contract_runtime_owned IN (0, 1)),
        pending_action_lookup_key TEXT,
        pending_action_projection_json TEXT,
        pending_action_projection_bytes INTEGER,
        UNIQUE(session_id, event_id)
    );

    CREATE TABLE IF NOT EXISTS cayu_budget_reservation_identities (
        reservation_id TEXT PRIMARY KEY,
        publication_session_id TEXT NOT NULL,
        publication_id TEXT NOT NULL,
        published INTEGER NOT NULL CHECK (published IN (0, 1))
    );

    CREATE TABLE IF NOT EXISTS cayu_mcp_manifest_baselines (
        history_key TEXT PRIMARY KEY,
        generation INTEGER NOT NULL CHECK (generation >= 1),
        baseline_json TEXT NOT NULL,
        updated_at TEXT NOT NULL
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

    CREATE TABLE IF NOT EXISTS cayu_public_authority_aliases (
        field_name TEXT NOT NULL,
        scope_session_id TEXT NOT NULL,
        public_alias TEXT NOT NULL,
        private_value TEXT NOT NULL,
        PRIMARY KEY (field_name, scope_session_id, public_alias)
    );

    CREATE INDEX IF NOT EXISTS idx_cayu_public_authority_private_value
        ON cayu_public_authority_aliases(field_name, scope_session_id, private_value);

    CREATE TABLE IF NOT EXISTS cayu_public_authority_alias_keys (
        key_id TEXT PRIMARY KEY,
        fingerprint TEXT NOT NULL,
        backfill_completed INTEGER NOT NULL CHECK (backfill_completed IN (0, 1))
    );

    CREATE TABLE IF NOT EXISTS cayu_public_authority_alias_config (
        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
        active_key_id TEXT NOT NULL REFERENCES cayu_public_authority_alias_keys(key_id),
        keyring_fingerprint TEXT NOT NULL,
        generation INTEGER NOT NULL CHECK (generation >= 1),
        retired_key_ids_json TEXT NOT NULL CHECK (json_valid(retired_key_ids_json))
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
        interaction_id TEXT,
        session_order INTEGER,
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

    CREATE TABLE IF NOT EXISTS cayu_session_message_deliveries (
        delivery_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES cayu_sessions(id) ON DELETE CASCADE,
        interaction_id TEXT,
        include_on_idle INTEGER NOT NULL,
        requested_eligible_through INTEGER,
        eligible_through INTEGER NOT NULL,
        batch_limit INTEGER NOT NULL,
        has_more INTEGER NOT NULL,
        interaction_started_event_json TEXT,
        queue_ids_json TEXT NOT NULL,
        events_json TEXT NOT NULL,
        created_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_cayu_session_message_deliveries_session
        ON cayu_session_message_deliveries(session_id, created_at);

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
        completed_at TEXT,
        invocation_json TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS cayu_task_terminalization_receipts (
        task_id TEXT NOT NULL,
        idempotency_key TEXT NOT NULL,
        request_sha256 TEXT NOT NULL,
        worker_id TEXT NOT NULL,
        terminal_kind TEXT NOT NULL,
        task_json TEXT NOT NULL,
        committed_at TEXT NOT NULL,
        PRIMARY KEY (task_id, idempotency_key)
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
    CREATE INDEX IF NOT EXISTS idx_cayu_sessions_parent_created_id
        ON cayu_sessions(parent_session_id, created_at, id);
    CREATE INDEX IF NOT EXISTS idx_cayu_session_labels_key_value_session
        ON cayu_session_labels(key, value, session_id);
    CREATE INDEX IF NOT EXISTS idx_cayu_events_session_sequence
        ON cayu_events(session_id, sequence);
    CREATE UNIQUE INDEX IF NOT EXISTS idx_cayu_events_budget_reservation_identity
        ON cayu_events(json_extract(payload_json, '$.reservation_id'))
        WHERE event_type = 'budget.reserved'
          AND json_type(payload_json, '$.reservation_id') = 'text';
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
    CREATE INDEX IF NOT EXISTS idx_cayu_events_pending_action_round_scope
        ON cayu_events(
            session_id,
            json_extract(
                pending_action_projection_json,
                '$.payload.tool_round_id'
            ),
            sequence
        )
        WHERE event_type IN (
            'tool.call.started',
            'tool.call.completed',
            'tool.call.failed',
            'tool.call.blocked',
            'tool.call.approval_denied'
        )
          AND json_type(
              pending_action_projection_json,
              '$.payload.tool_round_id'
          ) = 'text'
          AND length(json_extract(
              pending_action_projection_json,
              '$.payload.tool_round_id'
          )) = 39
          AND substr(json_extract(
              pending_action_projection_json,
              '$.payload.tool_round_id'
          ), 1, 7) = 'tround_'
          AND substr(json_extract(
              pending_action_projection_json,
              '$.payload.tool_round_id'
          ), 8) NOT GLOB '*[^0-9a-f]*';
    CREATE INDEX IF NOT EXISTS idx_cayu_events_pending_action_attempt_scope
        ON cayu_events(
            session_id,
            json_extract(
                pending_action_projection_json,
                '$.payload.model_step_id'
            ),
            json_extract(
                pending_action_projection_json,
                '$.payload.model_attempt_id'
            ),
            sequence
        )
        WHERE event_type IN (
            'tool.call.started',
            'tool.call.completed',
            'tool.call.failed',
            'tool.call.blocked',
            'tool.call.approval_denied'
        )
          AND json_type(
              pending_action_projection_json,
              '$.payload.model_step_id'
          ) = 'text'
          AND json_type(
              pending_action_projection_json,
              '$.payload.model_attempt_id'
          ) = 'text'
          AND length(json_extract(
              pending_action_projection_json,
              '$.payload.model_step_id'
          )) = 38
          AND substr(json_extract(
              pending_action_projection_json,
              '$.payload.model_step_id'
          ), 1, 6) = 'mstep_'
          AND substr(json_extract(
              pending_action_projection_json,
              '$.payload.model_step_id'
          ), 7) NOT GLOB '*[^0-9a-f]*'
          AND length(json_extract(
              pending_action_projection_json,
              '$.payload.model_attempt_id'
          )) = 37
          AND substr(json_extract(
              pending_action_projection_json,
              '$.payload.model_attempt_id'
          ), 1, 5) = 'matt_'
          AND substr(json_extract(
              pending_action_projection_json,
              '$.payload.model_attempt_id'
          ), 6) NOT GLOB '*[^0-9a-f]*';
    CREATE INDEX IF NOT EXISTS idx_cayu_events_type_timestamp
        ON cayu_events(event_type, timestamp);
    CREATE INDEX IF NOT EXISTS idx_cayu_events_agent_name
        ON cayu_events(agent_name);
    CREATE INDEX IF NOT EXISTS idx_cayu_events_environment_name
        ON cayu_events(environment_name);
    CREATE INDEX IF NOT EXISTS idx_cayu_events_workflow_name
        ON cayu_events(workflow_name);
    CREATE INDEX IF NOT EXISTS idx_cayu_events_workflow_step_replay
        ON cayu_events(
            session_id,
            workflow_name,
            json_extract(payload_json, '$.step_id'),
            event_type,
            sequence DESC
        )
        WHERE json_valid(payload_json);
    CREATE INDEX IF NOT EXISTS idx_cayu_events_workflow_step_attempt
        ON cayu_events(
            session_id,
            workflow_name,
            json_extract(payload_json, '$.attempt_id'),
            json_extract(payload_json, '$.step_id'),
            event_type,
            sequence DESC
        )
        WHERE json_valid(payload_json);
    CREATE INDEX IF NOT EXISTS idx_cayu_events_workflow_attempt_marker
        ON cayu_events(session_id, workflow_name, sequence DESC)
        WHERE event_type = 'custom.cayu.workflow.attempt';
    CREATE INDEX IF NOT EXISTS idx_cayu_events_tool_name
        ON cayu_events(tool_name);
    CREATE INDEX IF NOT EXISTS idx_cayu_transcript_messages_session_sequence
        ON cayu_transcript_messages(session_id, sequence);
    CREATE INDEX IF NOT EXISTS idx_cayu_transcript_messages_session_role_sequence
        ON cayu_transcript_messages(session_id, role, sequence);
    CREATE INDEX IF NOT EXISTS idx_cayu_events_session_interaction_sequence
        ON cayu_events(session_id, interaction_id, sequence);
    CREATE INDEX IF NOT EXISTS idx_cayu_transcript_messages_session_interaction_sequence
        ON cayu_transcript_messages(session_id, interaction_id, sequence);
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
    CREATE INDEX IF NOT EXISTS idx_cayu_tasks_session_created_id
        ON cayu_tasks(session_id, created_at, id);
    CREATE INDEX IF NOT EXISTS idx_cayu_tasks_parent_created_id
        ON cayu_tasks(parent_task_id, created_at, id);
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
            fts_rowid INTEGER PRIMARY KEY,
            id TEXT NOT NULL UNIQUE,
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
    22: """
        CREATE TABLE IF NOT EXISTS cayu_mcp_manifest_baselines (
            history_key TEXT PRIMARY KEY,
            generation INTEGER NOT NULL CHECK (generation >= 1),
            baseline_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

    """,
    23: """
        CREATE TABLE IF NOT EXISTS cayu_budget_reservation_identities (
            reservation_id TEXT PRIMARY KEY,
            publication_session_id TEXT NOT NULL,
            publication_id TEXT NOT NULL,
            published INTEGER NOT NULL CHECK (published IN (0, 1))
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_cayu_events_budget_reservation_identity
            ON cayu_events(json_extract(payload_json, '$.reservation_id'))
            WHERE event_type = 'budget.reserved'
              AND json_type(payload_json, '$.reservation_id') = 'text';

        CREATE INDEX IF NOT EXISTS idx_cayu_events_pending_action_round_scope
            ON cayu_events(
                session_id,
                json_extract(
                    pending_action_projection_json,
                    '$.payload.tool_round_id'
                ),
                sequence
            )
            WHERE event_type IN (
                'tool.call.started',
                'tool.call.completed',
                'tool.call.failed',
                'tool.call.blocked',
                'tool.call.approval_denied'
            )
              AND json_type(
                  pending_action_projection_json,
                  '$.payload.tool_round_id'
              ) = 'text'
              AND length(json_extract(
                  pending_action_projection_json,
                  '$.payload.tool_round_id'
              )) = 39
              AND substr(json_extract(
                  pending_action_projection_json,
                  '$.payload.tool_round_id'
              ), 1, 7) = 'tround_'
              AND substr(json_extract(
                  pending_action_projection_json,
                  '$.payload.tool_round_id'
              ), 8) NOT GLOB '*[^0-9a-f]*';

        CREATE INDEX IF NOT EXISTS idx_cayu_events_pending_action_attempt_scope
            ON cayu_events(
                session_id,
                json_extract(
                    pending_action_projection_json,
                    '$.payload.model_step_id'
                ),
                json_extract(
                    pending_action_projection_json,
                    '$.payload.model_attempt_id'
                ),
                sequence
            )
            WHERE event_type IN (
                'tool.call.started',
                'tool.call.completed',
                'tool.call.failed',
                'tool.call.blocked',
                'tool.call.approval_denied'
            )
              AND json_type(
                  pending_action_projection_json,
                  '$.payload.model_step_id'
              ) = 'text'
              AND json_type(
                  pending_action_projection_json,
                  '$.payload.model_attempt_id'
              ) = 'text'
              AND length(json_extract(
                  pending_action_projection_json,
                  '$.payload.model_step_id'
              )) = 38
              AND substr(json_extract(
                  pending_action_projection_json,
                  '$.payload.model_step_id'
              ), 1, 6) = 'mstep_'
              AND substr(json_extract(
                  pending_action_projection_json,
                  '$.payload.model_step_id'
              ), 7) NOT GLOB '*[^0-9a-f]*'
              AND length(json_extract(
                  pending_action_projection_json,
                  '$.payload.model_attempt_id'
              )) = 37
              AND substr(json_extract(
                  pending_action_projection_json,
                  '$.payload.model_attempt_id'
              ), 1, 5) = 'matt_'
              AND substr(json_extract(
                  pending_action_projection_json,
                  '$.payload.model_attempt_id'
              ), 6) NOT GLOB '*[^0-9a-f]*';
    """,
    24: """
        CREATE INDEX IF NOT EXISTS idx_cayu_sessions_parent_created_id
            ON cayu_sessions(parent_session_id, created_at, id);
    """,
    25: """
        CREATE TABLE IF NOT EXISTS cayu_budget_settlements (
            settlement_id TEXT PRIMARY KEY,
            reservation_id TEXT NOT NULL UNIQUE
                REFERENCES cayu_budget_reservations(reservation_id),
            session_id TEXT NOT NULL,
            settled_at TEXT NOT NULL,
            settlement_json TEXT NOT NULL,
            event_published INTEGER NOT NULL CHECK (event_published IN (0, 1))
        );

        CREATE INDEX IF NOT EXISTS idx_cayu_budget_settlements_pending
            ON cayu_budget_settlements(session_id, event_published, settled_at, settlement_id);

        CREATE INDEX IF NOT EXISTS idx_cayu_budget_settlements_pending_global
            ON cayu_budget_settlements(event_published, settled_at, settlement_id);

        CREATE INDEX IF NOT EXISTS idx_cayu_budget_reservation_identities_session
            ON cayu_budget_reservation_identities(
                publication_session_id,
                reservation_id
            );
    """,
    26: """
        CREATE INDEX IF NOT EXISTS idx_cayu_events_session_interaction_sequence
            ON cayu_events(session_id, interaction_id, sequence);
        CREATE INDEX IF NOT EXISTS idx_cayu_transcript_messages_session_interaction_sequence
            ON cayu_transcript_messages(session_id, interaction_id, sequence);

        CREATE UNIQUE INDEX IF NOT EXISTS idx_cayu_transcript_session_order
            ON cayu_transcript_messages(session_id, session_order);
        CREATE INDEX IF NOT EXISTS idx_cayu_transcript_interaction_order
            ON cayu_transcript_messages(session_id, interaction_id, session_order);

        CREATE TRIGGER IF NOT EXISTS cayu_reject_explicit_transcript_order
        BEFORE INSERT ON cayu_transcript_messages
        FOR EACH ROW
        WHEN NEW.session_order IS NOT NULL
        BEGIN
            SELECT RAISE(ABORT, 'cayu_transcript_messages.session_order is runtime-owned');
        END;

        CREATE TRIGGER IF NOT EXISTS cayu_assign_transcript_order
        AFTER INSERT ON cayu_transcript_messages
        FOR EACH ROW
        WHEN NEW.session_order IS NULL
        BEGIN
            UPDATE cayu_sessions
            SET transcript_seq = transcript_seq + 1
            WHERE id = NEW.session_id;
            UPDATE cayu_transcript_messages
            SET session_order = (
                SELECT transcript_seq FROM cayu_sessions WHERE id = NEW.session_id
            )
            WHERE sequence = NEW.sequence;
        END;

        CREATE TABLE IF NOT EXISTS cayu_interaction_latest_events (
            session_id TEXT NOT NULL,
            interaction_id TEXT NOT NULL,
            latest_event_sequence INTEGER NOT NULL,
            PRIMARY KEY (session_id, interaction_id),
            FOREIGN KEY (session_id) REFERENCES cayu_sessions(id) ON DELETE CASCADE,
            FOREIGN KEY (latest_event_sequence)
                REFERENCES cayu_events(sequence) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_cayu_interaction_latest_events_page
            ON cayu_interaction_latest_events(session_id, latest_event_sequence DESC);
        CREATE TRIGGER IF NOT EXISTS cayu_track_interaction_latest_event
        AFTER INSERT ON cayu_events
        FOR EACH ROW
        WHEN NEW.interaction_id IS NOT NULL
         AND NEW.event_type IN (
              'interaction.started', 'interaction.resumed', 'interaction.paused',
              'interaction.completed', 'interaction.failed', 'interaction.interrupted'
         )
        BEGIN
            INSERT INTO cayu_interaction_latest_events (
                session_id, interaction_id, latest_event_sequence
            ) VALUES (NEW.session_id, NEW.interaction_id, NEW.sequence)
            ON CONFLICT(session_id, interaction_id) DO UPDATE SET
                latest_event_sequence = excluded.latest_event_sequence
            WHERE excluded.latest_event_sequence > latest_event_sequence;
        END;

        CREATE TABLE IF NOT EXISTS cayu_deferred_interaction_inputs (
            session_id TEXT PRIMARY KEY
                REFERENCES cayu_sessions(id) ON DELETE CASCADE,
            interaction_id TEXT NOT NULL,
            source_messages_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS cayu_session_message_deliveries (
            delivery_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES cayu_sessions(id) ON DELETE CASCADE,
            interaction_id TEXT,
            include_on_idle INTEGER NOT NULL,
            requested_eligible_through INTEGER,
            eligible_through INTEGER NOT NULL,
            batch_limit INTEGER NOT NULL,
            has_more INTEGER NOT NULL,
            interaction_started_event_json TEXT,
            queue_ids_json TEXT NOT NULL,
            events_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_cayu_session_message_deliveries_session
            ON cayu_session_message_deliveries(session_id, created_at);
    """,
    27: """
        CREATE INDEX IF NOT EXISTS idx_cayu_tasks_session_created_id
            ON cayu_tasks(session_id, created_at, id);
        CREATE INDEX IF NOT EXISTS idx_cayu_tasks_parent_created_id
            ON cayu_tasks(parent_task_id, created_at, id);
    """,
    28: """
        CREATE TABLE IF NOT EXISTS cayu_public_authority_aliases (
            field_name TEXT NOT NULL,
            scope_session_id TEXT NOT NULL,
            public_alias TEXT NOT NULL,
            private_value TEXT NOT NULL,
            PRIMARY KEY (field_name, scope_session_id, public_alias)
        );

        CREATE INDEX IF NOT EXISTS idx_cayu_public_authority_private_value
            ON cayu_public_authority_aliases(field_name, scope_session_id, private_value);

        CREATE TABLE IF NOT EXISTS cayu_public_authority_alias_keys (
            key_id TEXT PRIMARY KEY,
            fingerprint TEXT NOT NULL,
            backfill_completed INTEGER NOT NULL CHECK (backfill_completed IN (0, 1))
        );

        CREATE TABLE IF NOT EXISTS cayu_public_authority_alias_config (
            singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
            active_key_id TEXT NOT NULL REFERENCES cayu_public_authority_alias_keys(key_id),
            keyring_fingerprint TEXT NOT NULL,
            generation INTEGER NOT NULL CHECK (generation >= 1),
            retired_key_ids_json TEXT NOT NULL CHECK (json_valid(retired_key_ids_json))
        );

        CREATE TRIGGER IF NOT EXISTS cayu_fence_stale_alias_session_writer
        BEFORE INSERT ON cayu_sessions
        FOR EACH ROW
        WHEN (SELECT active_key_id FROM cayu_public_authority_alias_config WHERE singleton = 1)
             IS NOT cayu_public_authority_active_key_id()
          OR (SELECT keyring_fingerprint FROM cayu_public_authority_alias_config WHERE singleton = 1)
             IS NOT cayu_public_authority_keyring_fingerprint()
        BEGIN
            SELECT RAISE(ABORT, 'stale public authority alias key configuration');
        END;

        CREATE TRIGGER IF NOT EXISTS cayu_fence_stale_alias_event_writer
        BEFORE INSERT ON cayu_events
        FOR EACH ROW
        WHEN (SELECT active_key_id FROM cayu_public_authority_alias_config WHERE singleton = 1)
             IS NOT cayu_public_authority_active_key_id()
          OR (SELECT keyring_fingerprint FROM cayu_public_authority_alias_config WHERE singleton = 1)
             IS NOT cayu_public_authority_keyring_fingerprint()
        BEGIN
            SELECT RAISE(ABORT, 'stale public authority alias key configuration');
        END;

        CREATE TRIGGER IF NOT EXISTS cayu_fence_stale_alias_transcript_writer
        BEFORE INSERT ON cayu_transcript_messages
        FOR EACH ROW
        WHEN (SELECT active_key_id FROM cayu_public_authority_alias_config WHERE singleton = 1)
             IS NOT cayu_public_authority_active_key_id()
          OR (SELECT keyring_fingerprint FROM cayu_public_authority_alias_config WHERE singleton = 1)
             IS NOT cayu_public_authority_keyring_fingerprint()
        BEGIN
            SELECT RAISE(ABORT, 'stale public authority alias key configuration');
        END;

        CREATE TRIGGER IF NOT EXISTS cayu_reject_public_authority_alias_conflict
        BEFORE INSERT ON cayu_public_authority_aliases
        FOR EACH ROW
        WHEN EXISTS (
            SELECT 1
            FROM cayu_public_authority_aliases AS existing
            WHERE existing.field_name = NEW.field_name
              AND existing.scope_session_id = NEW.scope_session_id
              AND existing.public_alias = NEW.public_alias
              AND existing.private_value <> NEW.private_value
        )
        BEGIN
            SELECT RAISE(ABORT, 'public authority alias conflicts with existing authority');
        END;

        CREATE TRIGGER IF NOT EXISTS cayu_require_session_public_authority_codec
        BEFORE INSERT ON cayu_sessions
        FOR EACH ROW
        WHEN EXISTS (
            SELECT 1 FROM cayu_public_authority_alias_keys WHERE backfill_completed = 1
        )
         AND cayu_public_authority_alias(NEW.id, 'session_id', NULL) IS NULL
        BEGIN
            SELECT RAISE(ABORT, 'public authority alias codec is required');
        END;

        CREATE TRIGGER IF NOT EXISTS cayu_require_event_public_authority_codec
        BEFORE INSERT ON cayu_events
        FOR EACH ROW
        WHEN EXISTS (
            SELECT 1 FROM cayu_public_authority_alias_keys WHERE backfill_completed = 1
        )
         AND cayu_public_authority_alias(NEW.session_id, 'session_id', NULL) IS NULL
        BEGIN
            SELECT RAISE(ABORT, 'public authority alias codec is required');
        END;

        CREATE TRIGGER IF NOT EXISTS cayu_require_transcript_public_authority_codec
        BEFORE INSERT ON cayu_transcript_messages
        FOR EACH ROW
        WHEN EXISTS (
            SELECT 1 FROM cayu_public_authority_alias_keys WHERE backfill_completed = 1
        )
         AND cayu_public_authority_alias(NEW.session_id, 'session_id', NULL) IS NULL
        BEGIN
            SELECT RAISE(ABORT, 'public authority alias codec is required');
        END;

        CREATE TRIGGER IF NOT EXISTS cayu_register_session_public_authority_alias
        AFTER INSERT ON cayu_sessions
        FOR EACH ROW
        WHEN cayu_public_authority_alias(NEW.id, 'session_id', NULL) IS NOT NULL
        BEGIN
            INSERT OR IGNORE INTO cayu_public_authority_aliases (
                field_name, scope_session_id, public_alias, private_value
            )
            SELECT 'session_id', '', alias.value, NEW.id
            FROM json_each(
                cayu_public_authority_aliases(NEW.id, 'session_id', NULL)
            ) AS alias;
        END;

        CREATE TRIGGER IF NOT EXISTS cayu_register_event_interaction_public_authority_alias
        AFTER INSERT ON cayu_events
        FOR EACH ROW
        WHEN NEW.interaction_id IS NOT NULL
         AND cayu_public_authority_alias(
             NEW.interaction_id, 'interaction_id', NEW.session_id
         ) IS NOT NULL
        BEGIN
            INSERT OR IGNORE INTO cayu_public_authority_aliases (
                field_name, scope_session_id, public_alias, private_value
            )
            SELECT 'interaction_id', NEW.session_id, alias.value, NEW.interaction_id
            FROM json_each(cayu_public_authority_aliases(
                NEW.interaction_id, 'interaction_id', NEW.session_id
            )) AS alias;
        END;

        CREATE TRIGGER IF NOT EXISTS cayu_register_transcript_interaction_public_authority_alias
        AFTER INSERT ON cayu_transcript_messages
        FOR EACH ROW
        WHEN NEW.interaction_id IS NOT NULL
         AND cayu_public_authority_alias(
             NEW.interaction_id, 'interaction_id', NEW.session_id
         ) IS NOT NULL
        BEGIN
            INSERT OR IGNORE INTO cayu_public_authority_aliases (
                field_name, scope_session_id, public_alias, private_value
            )
            SELECT 'interaction_id', NEW.session_id, alias.value, NEW.interaction_id
            FROM json_each(cayu_public_authority_aliases(
                NEW.interaction_id, 'interaction_id', NEW.session_id
            )) AS alias;
        END;

        CREATE TRIGGER IF NOT EXISTS cayu_register_turn_interaction_public_authority_aliases
        AFTER INSERT ON cayu_events
        FOR EACH ROW
        WHEN NEW.event_type = 'turn.completed'
         AND json_valid(NEW.payload_json)
         AND json_type(NEW.payload_json, '$.interaction_ids') = 'array'
        BEGIN
            INSERT OR IGNORE INTO cayu_public_authority_aliases (
                field_name, scope_session_id, public_alias, private_value
            )
            SELECT
                'interaction_id', NEW.session_id,
                alias.value,
                interaction.value
            FROM json_each(NEW.payload_json, '$.interaction_ids') AS interaction,
                 json_each(cayu_public_authority_aliases(
                     interaction.value, 'interaction_id', NEW.session_id
                 )) AS alias
            WHERE interaction.type = 'text'
              AND trim(interaction.value) <> '';
        END;
    """,
    29: """
        CREATE INDEX IF NOT EXISTS idx_cayu_events_workflow_step_replay
            ON cayu_events(
                session_id,
                workflow_name,
                json_extract(payload_json, '$.step_id'),
                event_type,
                sequence DESC
            )
            WHERE json_valid(payload_json);
        CREATE INDEX IF NOT EXISTS idx_cayu_events_workflow_step_attempt
            ON cayu_events(
                session_id,
                workflow_name,
                json_extract(payload_json, '$.attempt_id'),
                json_extract(payload_json, '$.step_id'),
                event_type,
                sequence DESC
            )
            WHERE json_valid(payload_json);
        CREATE INDEX IF NOT EXISTS idx_cayu_events_workflow_attempt_marker
            ON cayu_events(session_id, workflow_name, sequence DESC)
            WHERE event_type = 'custom.cayu.workflow.attempt';
    """,
    32: """
        CREATE TABLE IF NOT EXISTS cayu_eval_corpora (
            revision TEXT PRIMARY KEY,
            target_key TEXT NOT NULL,
            evidence_policy_revision TEXT NOT NULL,
            pricing_profile_fingerprint TEXT,
            suite_count INTEGER NOT NULL CHECK (suite_count >= 1 AND suite_count <= 64),
            case_count INTEGER NOT NULL CHECK (case_count >= 1 AND case_count <= 1000),
            assertion_count INTEGER NOT NULL
                CHECK (assertion_count >= case_count AND assertion_count <= case_count * 64),
            expanded_assertion_result_count INTEGER NOT NULL
                CHECK (expanded_assertion_result_count >= assertion_count
                    AND expanded_assertion_result_count <= 640000),
            document_json TEXT NOT NULL,
            document_bytes INTEGER NOT NULL
                CHECK (document_bytes >= 1 AND document_bytes <= 8388608)
                CHECK (document_bytes = length(CAST(document_json AS BLOB))),
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_cayu_eval_corpora_catalog
            ON cayu_eval_corpora(created_at DESC, revision ASC);
        CREATE INDEX IF NOT EXISTS idx_cayu_eval_corpora_target_catalog
            ON cayu_eval_corpora(target_key, created_at DESC, revision ASC);

        CREATE TABLE IF NOT EXISTS cayu_eval_suites (
            corpus_revision TEXT NOT NULL
                REFERENCES cayu_eval_corpora(revision) ON DELETE CASCADE,
            suite_id TEXT COLLATE BINARY NOT NULL,
            suite_revision TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            case_count INTEGER NOT NULL CHECK (case_count >= 1 AND case_count <= 1000),
            assertion_count INTEGER NOT NULL
                CHECK (assertion_count >= case_count AND assertion_count <= case_count * 64),
            trials INTEGER NOT NULL CHECK (trials >= 1 AND trials <= 100),
            timeout_seconds INTEGER NOT NULL
                CHECK (timeout_seconds >= 1 AND timeout_seconds <= 3600),
            CHECK (assertion_count * trials <= 10000),
            PRIMARY KEY (corpus_revision, suite_id)
        );

        CREATE TABLE IF NOT EXISTS cayu_eval_cases (
            corpus_revision TEXT NOT NULL,
            case_id TEXT COLLATE BINARY NOT NULL,
            case_revision TEXT NOT NULL,
            suite_id TEXT COLLATE BINARY NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            message_count INTEGER NOT NULL
                CHECK (message_count >= 1 AND message_count <= 16),
            assertion_count INTEGER NOT NULL
                CHECK (assertion_count >= 1 AND assertion_count <= 64),
            PRIMARY KEY (corpus_revision, case_id),
            FOREIGN KEY (corpus_revision, suite_id)
                REFERENCES cayu_eval_suites(corpus_revision, suite_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_cayu_eval_cases_suite
            ON cayu_eval_cases(corpus_revision, suite_id, case_id ASC);

        CREATE TABLE IF NOT EXISTS cayu_eval_runs (
            run_id TEXT COLLATE BINARY PRIMARY KEY,
            idempotency_key TEXT NOT NULL UNIQUE,
            corpus_revision TEXT NOT NULL
                REFERENCES cayu_eval_corpora(revision),
            target_key TEXT NOT NULL,
            suite_id TEXT COLLATE BINARY NOT NULL,
            suite_revision TEXT NOT NULL,
            max_concurrency INTEGER NOT NULL
                CHECK (max_concurrency >= 1 AND max_concurrency <= 32),
            status TEXT NOT NULL CHECK (
                status IN ('queued', 'running', 'cancelling', 'completed', 'failed', 'cancelled')
            ),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT,
            cancel_requested_at TEXT,
            claim_id TEXT,
            ownership_epoch INTEGER NOT NULL DEFAULT 0
                CHECK (ownership_epoch >= 0 AND ownership_epoch <= 9223372036854775807),
            lease_expires_at TEXT,
            result_revision TEXT,
            result_status TEXT CHECK (
                result_status IS NULL
                OR result_status IN ('passed', 'failed', 'unavailable', 'error')
            ),
            result_score REAL CHECK (
                result_score IS NULL OR (result_score >= 0.0 AND result_score <= 1.0)
            ),
            result_duration_ms INTEGER CHECK (
                result_duration_ms IS NULL OR result_duration_ms >= 0
            ),
            failure_code TEXT CHECK (
                failure_code IS NULL OR failure_code IN (
                    'target_unavailable', 'corpus_unavailable', 'execution_failed',
                    'worker_interrupted'
                )
            ),
            CHECK (
                (status IN ('completed', 'failed', 'cancelled') AND finished_at IS NOT NULL)
                OR (status NOT IN ('completed', 'failed', 'cancelled') AND finished_at IS NULL)
            ),
            CHECK (
                (status IN ('cancelling', 'cancelled') AND cancel_requested_at IS NOT NULL)
                OR (status NOT IN ('cancelling', 'cancelled') AND cancel_requested_at IS NULL)
            ),
            CHECK (
                (status IN ('running', 'cancelling') AND started_at IS NOT NULL
                    AND claim_id IS NOT NULL
                    AND lease_expires_at IS NOT NULL AND lease_expires_at > updated_at)
                OR (status NOT IN ('running', 'cancelling') AND lease_expires_at IS NULL)
            ),
            CHECK (status NOT IN ('completed', 'failed') OR started_at IS NOT NULL),
            CHECK (
                (status = 'completed' AND result_revision IS NOT NULL
                    AND result_status IS NOT NULL AND result_duration_ms IS NOT NULL)
                OR (status != 'completed' AND result_revision IS NULL
                    AND result_status IS NULL AND result_score IS NULL
                    AND result_duration_ms IS NULL)
            ),
            CHECK (
                (result_status IN ('passed', 'failed') AND result_score IS NOT NULL)
                OR (result_status NOT IN ('passed', 'failed') AND result_score IS NULL)
                OR result_status IS NULL
            ),
            CHECK (
                (status = 'failed' AND failure_code IS NOT NULL)
                OR (status != 'failed' AND failure_code IS NULL)
            ),
            FOREIGN KEY (corpus_revision, suite_id)
                REFERENCES cayu_eval_suites(corpus_revision, suite_id)
        );
        CREATE INDEX IF NOT EXISTS idx_cayu_eval_runs_catalog
            ON cayu_eval_runs(created_at DESC, run_id ASC);
        CREATE INDEX IF NOT EXISTS idx_cayu_eval_runs_status_claim
            ON cayu_eval_runs(status, lease_expires_at, created_at ASC, run_id ASC);
        CREATE INDEX IF NOT EXISTS idx_cayu_eval_runs_corpus_catalog
            ON cayu_eval_runs(corpus_revision, created_at DESC, run_id ASC);

        CREATE TABLE IF NOT EXISTS cayu_eval_results (
            run_id TEXT PRIMARY KEY
                REFERENCES cayu_eval_runs(run_id) ON DELETE RESTRICT,
            revision TEXT NOT NULL,
            result_json TEXT NOT NULL,
            result_bytes INTEGER NOT NULL
                CHECK (result_bytes >= 1 AND result_bytes <= 41943040)
                CHECK (result_bytes = length(CAST(result_json AS BLOB))),
            created_at TEXT NOT NULL
        );
    """,
    33: """
        CREATE INDEX IF NOT EXISTS idx_cayu_eval_runs_target_catalog
            ON cayu_eval_runs(target_key, created_at DESC, run_id ASC);
        CREATE INDEX IF NOT EXISTS idx_cayu_eval_runs_target_status_claim
            ON cayu_eval_runs(
                target_key, status, lease_expires_at, created_at ASC, run_id ASC
            );
    """,
    34: """
        CREATE INDEX IF NOT EXISTS idx_cayu_tasks_claim_availability
            ON cayu_tasks(status, session_id, created_at, id, available_at);
    """,
    35: """
        CREATE TABLE IF NOT EXISTS cayu_knowledge_publication_receipts (
            operation_id TEXT PRIMARY KEY,
            entry_id TEXT NOT NULL,
            request_sha256 TEXT NOT NULL,
            entry_created_at TEXT NOT NULL,
            entry_updated_at TEXT NOT NULL,
            committed_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_publication_receipts_entry
            ON cayu_knowledge_publication_receipts(entry_id);
    """,
    38: """
        CREATE TABLE IF NOT EXISTS cayu_task_terminalization_receipts (
            task_id TEXT NOT NULL,
            idempotency_key TEXT NOT NULL,
            request_sha256 TEXT NOT NULL,
            worker_id TEXT NOT NULL,
            terminal_kind TEXT NOT NULL,
            task_json TEXT NOT NULL,
            committed_at TEXT NOT NULL,
            PRIMARY KEY (task_id, idempotency_key)
        );
    """,
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
    26: (
        ("cayu_events", "interaction_id", "TEXT"),
        ("cayu_transcript_messages", "interaction_id", "TEXT"),
        ("cayu_sessions", "transcript_seq", "INTEGER NOT NULL DEFAULT 0"),
        ("cayu_transcript_messages", "session_order", "INTEGER"),
    ),
    31: (
        (
            "cayu_events",
            "input_contract_runtime_owned",
            "INTEGER NOT NULL DEFAULT 0 CHECK (input_contract_runtime_owned IN (0, 1))",
        ),
    ),
    34: (("cayu_tasks", "available_at", "TEXT"),),
    36: (("cayu_sessions", "invocation_json", "TEXT NOT NULL"),),
    39: (("cayu_tasks", "invocation_json", "TEXT NOT NULL"),),
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


def _reject_populated_pre_interaction_database(connection: sqlite3.Connection) -> None:
    if connection.execute("SELECT EXISTS(SELECT 1 FROM cayu_sessions)").fetchone()[0]:
        raise schema.SchemaTooOld(
            "Storage revision 26 is a clean prerelease break and cannot migrate a "
            "populated Cayu session database. Recreate the Cayu database before "
            "starting this build."
        )


def _reject_populated_pre_invocation_database(connection: sqlite3.Connection) -> None:
    if connection.execute("SELECT EXISTS(SELECT 1 FROM cayu_sessions)").fetchone()[0]:
        raise schema.SchemaTooOld(
            "Storage revision 36 requires invocation provenance for every session and "
            "cannot migrate a populated Cayu session database. Recreate the Cayu "
            "database before starting this build."
        )


def _reject_populated_pre_task_invocation_database(
    connection: sqlite3.Connection,
) -> None:
    if connection.execute("SELECT EXISTS(SELECT 1 FROM cayu_tasks)").fetchone()[0]:
        raise schema.SchemaTooOld(
            "Storage revision 39 requires invocation provenance for every task and "
            "cannot migrate a populated Cayu task database. Recreate the Cayu "
            "database before starting this build."
        )


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


def _add_budget_execution_identity_if_present(connection: sqlite3.Connection) -> None:
    """Add revision-23 identity without fabricating attribution for old rows."""

    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'cayu_budget_reservations'"
    ).fetchone()
    if exists is not None:
        _add_column_if_missing(
            connection,
            "cayu_budget_reservations",
            "budget_limit_id",
            "TEXT",
        )
        _add_column_if_missing(
            connection,
            "cayu_budget_reservations",
            "model_step_id",
            "TEXT",
        )
        _add_column_if_missing(
            connection,
            "cayu_budget_reservations",
            "model_attempt_id",
            "TEXT",
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_cayu_budget_reservations_limit "
            "ON cayu_budget_reservations(budget_limit_id, status, updated_at)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_cayu_budget_reservations_model_attempt "
            "ON cayu_budget_reservations(model_attempt_id, budget_limit_id, status)"
        )


def _prepare_revision_twenty_three(connection: sqlite3.Connection) -> None:
    """Install execution columns and preserve exact historical reservation ownership."""

    _add_budget_execution_identity_if_present(connection)
    connection.execute(
        """
        INSERT OR IGNORE INTO cayu_budget_reservation_identities (
            reservation_id,
            publication_session_id,
            publication_id,
            published
        )
        SELECT
            json_extract(payload_json, '$.reservation_id'),
            session_id,
            event_id,
            1
        FROM cayu_events
        WHERE event_type = 'budget.reserved'
          AND json_type(payload_json, '$.reservation_id') = 'text'
        """
    )


def _prepare_revision_twenty_five(connection: sqlite3.Connection) -> None:
    """Install crash-safe budget dispatch and audit-outbox columns."""

    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'cayu_budget_reservations'"
    ).fetchone()
    if exists is None:
        return
    active = connection.execute(
        "SELECT 1 FROM cayu_budget_reservations WHERE status = 'active' LIMIT 1"
    ).fetchone()
    if active is not None:
        raise RuntimeError(
            "Schema revision 25 cannot migrate active budget reservations because "
            "their dispatch state is unknown. Drain or explicitly settle every active "
            "reservation, then retry the migration."
        )
    for column, definition in (
        ("environment_name", "TEXT"),
        ("settlement_event_payload_json", "TEXT NOT NULL DEFAULT '{}'"),
        ("settlement_fallback_json", "TEXT"),
        ("dispatch_id", "TEXT"),
        ("dispatched_at", "TEXT"),
    ):
        _add_column_if_missing(
            connection,
            "cayu_budget_reservations",
            column,
            definition,
        )
    rows = connection.execute(
        """
        SELECT reservation_id, created_at
        FROM cayu_budget_reservations
        WHERE settlement_fallback_json IS NULL
        """
    ).fetchall()
    for reservation_id, created_at in rows:
        connection.execute(
            """
            UPDATE cayu_budget_reservations
            SET settlement_fallback_json = ?
            WHERE reservation_id = ?
            """,
            (
                json.dumps(
                    {
                        "settled_at": created_at,
                        "reconciliation_reason": (
                            "model completion settlement evidence was not publishable; "
                            "charged reserved amount"
                        ),
                        "release_reason": "reservation released before provider dispatch",
                        "expiration_reason": None,
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                reservation_id,
            ),
        )


_KNOWLEDGE_CHUNK_LEGACY_COLUMNS = (
    "id",
    "entry_id",
    "chunk_index",
    "text",
    "content_hash",
    "source_uri",
    "metadata_json",
)
_KNOWLEDGE_CHUNK_KEYED_COLUMNS = ("fts_rowid", *_KNOWLEDGE_CHUNK_LEGACY_COLUMNS)


def _sqlite_table_columns(connection: sqlite3.Connection, table: str) -> tuple[str, ...]:
    return tuple(str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})"))


def _sqlite_has_unique_index(
    connection: sqlite3.Connection,
    table: str,
    columns: tuple[str, ...],
) -> bool:
    for row in connection.execute(f"PRAGMA index_list({table})"):
        if not bool(row[2]):
            continue
        index_columns = tuple(
            str(column[2]) for column in connection.execute(f"PRAGMA index_info({row[1]})")
        )
        if index_columns == columns:
            return True
    return False


def _validate_revision_37_knowledge_fts_schema(connection: sqlite3.Connection) -> None:
    columns = connection.execute("PRAGMA table_info(cayu_knowledge_chunks)").fetchall()
    if tuple(str(row[1]) for row in columns) != _KNOWLEDGE_CHUNK_KEYED_COLUMNS:
        raise RuntimeError(
            "SQLite knowledge chunks do not provide the revision-37 stable FTS key. "
            "Restore the required schema from a known-good backup."
        )
    fts_rowid = columns[0]
    if str(fts_rowid[2]).upper() != "INTEGER" or int(fts_rowid[5]) != 1:
        raise RuntimeError(
            "SQLite knowledge chunks have an invalid revision-37 FTS key. "
            "Restore the required schema from a known-good backup."
        )
    if not _sqlite_has_unique_index(connection, "cayu_knowledge_chunks", ("id",)):
        raise RuntimeError("SQLite knowledge chunks are missing their unique public id constraint.")
    if not _sqlite_has_unique_index(
        connection,
        "cayu_knowledge_chunks",
        ("entry_id", "chunk_index"),
    ):
        raise RuntimeError(
            "SQLite knowledge chunks are missing their entry/chunk identity constraint."
        )
    entry_index = connection.execute(
        "SELECT tbl_name, sql FROM sqlite_master WHERE type = 'index' "
        "AND name = 'idx_cayu_knowledge_chunks_entry_index'"
    ).fetchone()
    entry_index_columns = (
        tuple(
            str(column[2])
            for column in connection.execute(
                "PRAGMA index_info(idx_cayu_knowledge_chunks_entry_index)"
            )
        )
        if entry_index is not None
        else ()
    )
    if (
        entry_index is None
        or entry_index[0] != "cayu_knowledge_chunks"
        or entry_index_columns != ("entry_id", "chunk_index")
    ):
        raise RuntimeError("Required Cayu SQLite knowledge chunk index is missing.")
    fts = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'cayu_knowledge_chunks_fts'"
    ).fetchone()
    normalized_fts = " ".join(str(fts[0]).lower().split()) if fts is not None else ""
    required_fts = "using fts5(entry_id unindexed, chunk_id unindexed, title, text)"
    if required_fts not in normalized_fts:
        raise RuntimeError("SQLite knowledge FTS does not match the revision-37 search contract.")


def _validate_revision_37_knowledge_fts_data(connection: sqlite3.Connection) -> None:
    mismatch = connection.execute(
        """
        SELECT 1
        FROM cayu_knowledge_chunks AS chunk
        JOIN cayu_knowledge_entries AS entry ON entry.id = chunk.entry_id
        LEFT JOIN cayu_knowledge_chunks_fts AS fts ON fts.rowid = chunk.fts_rowid
        WHERE fts.rowid IS NULL
           OR fts.entry_id IS NOT chunk.entry_id
           OR fts.chunk_id IS NOT chunk.id
           OR fts.title IS NOT COALESCE(entry.title, '')
           OR fts.text IS NOT CASE
                WHEN chunk.text = entry.text THEN chunk.text
                ELSE entry.text || char(10) || chunk.text
              END
        LIMIT 1
        """
    ).fetchone()
    extra = connection.execute(
        """
        SELECT 1
        FROM cayu_knowledge_chunks_fts AS fts
        LEFT JOIN cayu_knowledge_chunks AS chunk ON chunk.fts_rowid = fts.rowid
        WHERE chunk.fts_rowid IS NULL
        LIMIT 1
        """
    ).fetchone()
    if mismatch is not None or extra is not None:
        raise RuntimeError(
            "SQLite revision-37 knowledge FTS rebuild did not preserve an exact "
            "source-to-index relationship."
        )


def _migrate_revision_thirty_seven_knowledge_fts(connection: sqlite3.Connection) -> None:
    columns = _sqlite_table_columns(connection, "cayu_knowledge_chunks")
    if columns == _KNOWLEDGE_CHUNK_KEYED_COLUMNS:
        # Greenfield baseline databases already have the current layout. This also
        # makes an explicitly retried migration safe if the schema was prepared by
        # a compatible deployment before its ledger marker was restored.
        _validate_revision_37_knowledge_fts_schema(connection)
        _validate_revision_37_knowledge_fts_data(connection)
        return
    if columns != _KNOWLEDGE_CHUNK_LEGACY_COLUMNS:
        raise RuntimeError(
            "SQLite knowledge chunks conflict with both the legacy and revision-37 "
            "schemas. Restore the database from a known-good backup."
        )

    connection.execute("DROP TABLE cayu_knowledge_chunks_fts")
    connection.execute(
        """
        CREATE TABLE cayu_knowledge_chunks_revision_37 (
            fts_rowid INTEGER PRIMARY KEY,
            id TEXT NOT NULL UNIQUE,
            entry_id TEXT NOT NULL
                REFERENCES cayu_knowledge_entries(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            content_hash TEXT,
            source_uri TEXT,
            metadata_json TEXT NOT NULL,
            UNIQUE (entry_id, chunk_index)
        )
        """
    )
    connection.execute(
        """
        INSERT INTO cayu_knowledge_chunks_revision_37 (
            fts_rowid, id, entry_id, chunk_index, text,
            content_hash, source_uri, metadata_json
        )
        SELECT
            rowid, id, entry_id, chunk_index, text,
            content_hash, source_uri, metadata_json
        FROM cayu_knowledge_chunks
        ORDER BY rowid
        """
    )
    connection.execute("DROP TABLE cayu_knowledge_chunks")
    connection.execute(
        "ALTER TABLE cayu_knowledge_chunks_revision_37 RENAME TO cayu_knowledge_chunks"
    )
    connection.execute(
        "CREATE INDEX idx_cayu_knowledge_chunks_entry_index "
        "ON cayu_knowledge_chunks(entry_id, chunk_index)"
    )
    connection.execute(
        """
        CREATE VIRTUAL TABLE cayu_knowledge_chunks_fts
        USING fts5(entry_id UNINDEXED, chunk_id UNINDEXED, title, text)
        """
    )
    connection.execute(
        """
        INSERT INTO cayu_knowledge_chunks_fts (
            rowid, entry_id, chunk_id, title, text
        )
        SELECT
            chunk.fts_rowid,
            chunk.entry_id,
            chunk.id,
            COALESCE(entry.title, ''),
            CASE
                WHEN chunk.text = entry.text THEN chunk.text
                ELSE entry.text || char(10) || chunk.text
            END
        FROM cayu_knowledge_chunks AS chunk
        JOIN cayu_knowledge_entries AS entry ON entry.id = chunk.entry_id
        ORDER BY chunk.fts_rowid
        """
    )
    _validate_revision_37_knowledge_fts_schema(connection)
    _validate_revision_37_knowledge_fts_data(connection)


# Per-revision Python follow-ups that cannot be expressed as unconditional DDL
# (e.g. conditionally carrying data out of a legacy ad-hoc table). Each hook runs
# after its revision's DDL and before the revision is recorded.
_MIGRATION_HOOKS: dict[int, Callable[[sqlite3.Connection], None]] = {
    8: _migrate_legacy_budget_reservations,
    14: _backfill_session_activity,
    21: _add_budget_billing_identity_if_present,
    23: _prepare_revision_twenty_three,
    25: _prepare_revision_twenty_five,
    37: _migrate_revision_thirty_seven_knowledge_fts,
}

_REVISION_17_INDEX_NAMES = frozenset(
    {
        "idx_cayu_checkpoints_pending_control_action",
        "idx_cayu_events_pending_action_barrier",
        "idx_cayu_events_pending_action_lookup",
    }
)
_RESERVATION_EVENT_INDEX_NAME = "idx_cayu_events_budget_reservation_identity"
_RESERVATION_IDENTITY_TABLE_NAME = "cayu_budget_reservation_identities"
_PENDING_ACTION_SCOPE_INDEX_NAMES = frozenset(
    {
        "idx_cayu_events_pending_action_round_scope",
        "idx_cayu_events_pending_action_attempt_scope",
    }
)
_WORKFLOW_REPLAY_INDEX_NAMES = frozenset(
    {
        "idx_cayu_events_workflow_step_replay",
        "idx_cayu_events_workflow_step_attempt",
        "idx_cayu_events_workflow_attempt_marker",
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


def _workflow_replay_index_definitions() -> dict[str, str]:
    definitions: dict[str, str] = {}
    for statement in _iter_statements(_MIGRATION_STEPS[29]):
        match = re.match(
            r"CREATE\s+INDEX\s+(?:IF\s+NOT\s+EXISTS\s+)?([^\s(]+)",
            statement,
            flags=re.IGNORECASE,
        )
        if match is not None and match.group(1) in _WORKFLOW_REPLAY_INDEX_NAMES:
            definitions[match.group(1)] = statement
    if definitions.keys() != _WORKFLOW_REPLAY_INDEX_NAMES:
        raise RuntimeError("Cayu workflow replay index definitions are incomplete.")
    return definitions


def _validate_workflow_replay_indexes(
    connection: sqlite3.Connection,
    *,
    require_all: bool,
) -> None:
    for index_name, expected in _workflow_replay_index_definitions().items():
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
        actual_type, table_name, actual_definition = row
        if (
            actual_type != "index"
            or table_name != "cayu_events"
            or actual_definition is None
            or _normalize_sqlite_schema_definition(actual_definition)
            != _normalize_sqlite_schema_definition(expected)
        ):
            raise RuntimeError(
                f"SQLite schema object {index_name!r} conflicts with Cayu's "
                "workflow replay contract. Rename or remove the conflicting "
                "object, then run with schema_mode='migrate'."
            )


def _repair_missing_workflow_replay_indexes(connection: sqlite3.Connection) -> None:
    with _transaction(connection):
        _validate_workflow_replay_indexes(connection, require_all=False)
        existing_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            ).fetchall()
        }
        for index_name, definition in _workflow_replay_index_definitions().items():
            if index_name not in existing_names:
                connection.execute(definition)
        _validate_workflow_replay_indexes(connection, require_all=True)


def _reservation_event_index_definition() -> str:
    statements = tuple(
        statement
        for statement in _iter_statements(_MIGRATION_STEPS[23])
        if _RESERVATION_EVENT_INDEX_NAME in statement
    )
    if len(statements) != 1:
        raise RuntimeError("Cayu reservation event index definition is incomplete.")
    return statements[0]


def _validate_reservation_event_index(
    connection: sqlite3.Connection,
    *,
    require: bool,
) -> None:
    row = connection.execute(
        "SELECT type, tbl_name, sql FROM sqlite_master WHERE name = ?",
        (_RESERVATION_EVENT_INDEX_NAME,),
    ).fetchone()
    if row is None:
        if require:
            raise RuntimeError(
                f"Required Cayu SQLite index is missing: {_RESERVATION_EVENT_INDEX_NAME}. "
                "Run with schema_mode='migrate' to repair the schema."
            )
        return
    actual_type, table_name, actual_definition = row
    if (
        actual_type != "index"
        or table_name != "cayu_events"
        or actual_definition is None
        or (
            _normalize_sqlite_schema_definition(actual_definition)
            != _normalize_sqlite_schema_definition(_reservation_event_index_definition())
        )
    ):
        raise RuntimeError(
            f"SQLite schema object {_RESERVATION_EVENT_INDEX_NAME!r} conflicts with "
            "Cayu's reservation identity contract. Rename or remove the conflicting object, then run "
            "with schema_mode='migrate' to create the required unique index."
        )


def _repair_missing_reservation_event_index(connection: sqlite3.Connection) -> None:
    with _transaction(connection):
        _validate_reservation_event_index(connection, require=False)
        row = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = ?",
            (_RESERVATION_EVENT_INDEX_NAME,),
        ).fetchone()
        if row is None:
            connection.execute(_reservation_event_index_definition())
        _validate_reservation_event_index(connection, require=True)


def _pending_action_scope_index_definitions() -> dict[str, str]:
    definitions: dict[str, str] = {}
    for statement in _iter_statements(_MIGRATION_STEPS[23]):
        for index_name in _PENDING_ACTION_SCOPE_INDEX_NAMES:
            if index_name in statement:
                definitions[index_name] = statement
    if definitions.keys() != _PENDING_ACTION_SCOPE_INDEX_NAMES:
        raise RuntimeError("Cayu pending-action scope index definitions are incomplete.")
    return definitions


def _validate_pending_action_scope_indexes(
    connection: sqlite3.Connection,
    *,
    require_all: bool,
) -> None:
    for index_name, expected in _pending_action_scope_index_definitions().items():
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
        actual_type, table_name, actual_definition = row
        if (
            actual_type != "index"
            or table_name != "cayu_events"
            or actual_definition is None
            or _normalize_sqlite_schema_definition(actual_definition)
            != _normalize_sqlite_schema_definition(expected)
        ):
            raise RuntimeError(
                f"SQLite schema object {index_name!r} conflicts with Cayu's "
                "pending-action scope contract. Rename or remove the conflicting "
                "object, then run with schema_mode='migrate'."
            )


def _repair_missing_pending_action_scope_indexes(connection: sqlite3.Connection) -> None:
    with _transaction(connection):
        _validate_pending_action_scope_indexes(connection, require_all=False)
        existing_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            ).fetchall()
        }
        for index_name, definition in _pending_action_scope_index_definitions().items():
            if index_name not in existing_names:
                connection.execute(definition)
        _validate_pending_action_scope_indexes(connection, require_all=True)


def _validate_reservation_identity_registry(
    connection: sqlite3.Connection,
    *,
    require: bool,
    verify_event_ownership: bool = False,
) -> None:
    row = connection.execute(
        "SELECT type FROM sqlite_master WHERE name = ?",
        (_RESERVATION_IDENTITY_TABLE_NAME,),
    ).fetchone()
    if row is None:
        if require:
            raise RuntimeError(
                f"Required Cayu SQLite table is missing: "
                f"{_RESERVATION_IDENTITY_TABLE_NAME}. Restore the permanent "
                "reservation ownership registry from a known-good backup."
            )
        return
    columns = connection.execute(
        f"PRAGMA table_info({_RESERVATION_IDENTITY_TABLE_NAME})"
    ).fetchall()
    actual = tuple(
        (column[1], column[2].upper(), bool(column[3]), int(column[5])) for column in columns
    )
    expected = (
        ("reservation_id", "TEXT", False, 1),
        ("publication_session_id", "TEXT", True, 0),
        ("publication_id", "TEXT", True, 0),
        ("published", "INTEGER", True, 0),
    )
    foreign_keys = connection.execute(
        f"PRAGMA foreign_key_list({_RESERVATION_IDENTITY_TABLE_NAME})"
    ).fetchall()
    if row[0] != "table" or actual != expected or foreign_keys:
        raise RuntimeError(
            f"SQLite schema object {_RESERVATION_IDENTITY_TABLE_NAME!r} conflicts "
            "with Cayu's reservation identity contract. Restore the required "
            "ownership registry from a known-good backup."
        )
    if not verify_event_ownership:
        return
    unmatched_event = connection.execute(
        """
        SELECT 1
        FROM cayu_events AS event
        LEFT JOIN cayu_budget_reservation_identities AS identity
          ON identity.reservation_id = json_extract(
              event.payload_json,
              '$.reservation_id'
          )
        WHERE event.event_type = 'budget.reserved'
          AND json_type(event.payload_json, '$.reservation_id') = 'text'
          AND (
              identity.reservation_id IS NULL
              OR identity.publication_session_id != event.session_id
              OR identity.publication_id != event.event_id
              OR identity.published != 1
          )
        LIMIT 1
        """
    ).fetchone()
    if unmatched_event is not None:
        raise RuntimeError(
            "SQLite budget reservation events disagree with the permanent "
            "reservation ownership registry."
        )


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
    if current.revision >= 23:
        if schema_mode is schema.SchemaMode.MIGRATE:
            _repair_missing_reservation_event_index(connection)
            _repair_missing_pending_action_scope_indexes(connection)
        else:
            _validate_reservation_event_index(connection, require=True)
            _validate_pending_action_scope_indexes(connection, require_all=True)
        _validate_reservation_identity_registry(connection, require=True)
    if current.revision >= 29:
        if schema_mode is schema.SchemaMode.MIGRATE:
            _repair_missing_workflow_replay_indexes(connection)
        else:
            _validate_workflow_replay_indexes(connection, require_all=True)
    if app_min_supported >= 36:
        _validate_session_invocation_column(connection)
    if current.revision >= 37:
        # Structural validation is intentionally constant-size. The full source/
        # FTS census belongs to the one-time revision hook, never ordinary startup.
        _validate_revision_37_knowledge_fts_schema(connection)
    if app_min_supported >= 38:
        _validate_task_terminalization_receipt_table(connection)
    if app_min_supported >= 39:
        _validate_task_invocation_column(connection)


def _validate_session_invocation_column(connection: sqlite3.Connection) -> None:
    columns = {
        str(row[1]): (str(row[2]).upper(), int(row[3]))
        for row in connection.execute("PRAGMA table_info(cayu_sessions)")
    }
    if columns.get("invocation_json") != ("TEXT", 1):
        raise RuntimeError(
            "SQLite schema object 'cayu_sessions.invocation_json' conflicts with "
            "Cayu's required invocation-provenance contract. Recreate the Cayu "
            "database from a known-good revision-36 schema."
        )


def _validate_task_terminalization_receipt_table(
    connection: sqlite3.Connection,
) -> None:
    columns = tuple(
        (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
        for row in connection.execute("PRAGMA table_info(cayu_task_terminalization_receipts)")
    )
    expected = (
        ("task_id", "TEXT", 1, 1),
        ("idempotency_key", "TEXT", 1, 2),
        ("request_sha256", "TEXT", 1, 0),
        ("worker_id", "TEXT", 1, 0),
        ("terminal_kind", "TEXT", 1, 0),
        ("task_json", "TEXT", 1, 0),
        ("committed_at", "TEXT", 1, 0),
    )
    if columns != expected:
        raise RuntimeError(
            "SQLite task terminalization receipt table conflicts with Cayu's "
            "revision-38 durability contract. Run `cayu storage migrate` or restore "
            "the database from a known-good backup."
        )


def _validate_task_invocation_column(connection: sqlite3.Connection) -> None:
    columns = {
        str(row[1]): (str(row[2]).upper(), int(row[3]))
        for row in connection.execute("PRAGMA table_info(cayu_tasks)")
    }
    if columns.get("invocation_json") != ("TEXT", 1):
        raise RuntimeError(
            "SQLite schema object 'cayu_tasks.invocation_json' conflicts with "
            "Cayu's required task invocation-provenance contract. Recreate the "
            "Cayu database from a known-good revision-39 schema."
        )


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
def _transaction(
    connection: sqlite3.Connection,
    *,
    begin_immediate: bool = True,
) -> Iterator[None]:
    """Run a block inside one owned SQLite transaction (BEGIN/COMMIT/ROLLBACK).

    Most revisions apply DDL, their data hook, and their revision marker
    atomically. Large revision-17 backfills instead use this helper for short,
    independently committed batches and explicit ready markers, making a crash
    resumable without holding one write lock for the entire data set.

    ``executescript`` cannot be used here: it force-commits any open transaction,
    so revision DDL is executed statement-by-statement.

    ``begin_immediate=False`` preserves ``sqlite3``'s implicit writer admission
    for knowledge operations that previously used the connection context manager.
    """
    failure: BaseException | None = None
    suppress_implicit_context = False
    try:
        if begin_immediate:
            connection.execute("BEGIN IMMEDIATE")
        yield
        connection.commit()
    except BaseException as primary:
        failure = _settle_failed_transaction(connection, primary)
        suppress_implicit_context = failure is not primary
    if failure is not None:
        # A context-manager body exception remains Python's active implicit
        # context while __exit__ runs. Suppress that incidental chain because the
        # primary is already a cleanup aggregate's first explicit member. Preserve
        # an unchanged primary's own causal evidence when rollback succeeds.
        if suppress_implicit_context:
            raise failure from None
        raise failure


def _settle_failed_transaction(
    connection: sqlite3.Connection,
    primary: BaseException,
) -> BaseException:
    """Roll back ``primary`` or fence a connection whose cleanup is uncertain."""
    try:
        in_transaction = connection.in_transaction
    except BaseException as state_error:
        state_error.__context__ = None
        return _fence_failed_transaction(connection, primary, state_error)
    if not in_transaction:
        # A commit may have completed before its caller observed the failure.
        return primary
    try:
        connection.rollback()
    except BaseException as rollback_error:
        # The primary is already an explicit member of the aggregate. Remove only
        # Python's incidental handler context so it is represented exactly once.
        rollback_error.__context__ = None
        return _fence_failed_transaction(connection, primary, rollback_error)
    return primary


def _fence_failed_transaction(
    connection: sqlite3.Connection,
    primary: BaseException,
    cleanup_error: BaseException,
) -> BaseException:
    failures = [primary, cleanup_error]
    try:
        # Closing a SQLite connection abandons its active transaction and releases
        # writer ownership. The closed connection then fails closed on later use.
        connection.close()
    except BaseException as close_error:
        close_error.__context__ = None
        failures.append(close_error)
    return BaseExceptionGroup(
        "SQLite transaction failed and cleanup could not prove rollback",
        failures,
    )


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
    if (
        current != schema.UNINITIALIZED
        and current < 26
        and any(revision.revision == 26 for revision in schema.pending(current))
    ):
        # Refuse the clean break before applying any earlier pending revision.
        # A failed migration must not leave an old populated database advanced
        # partway to revision 25.
        _reject_populated_pre_interaction_database(connection)
    if (
        current != schema.UNINITIALIZED
        and current < 36
        and any(revision.revision == 36 for revision in schema.pending(current))
    ):
        _reject_populated_pre_invocation_database(connection)
    if (
        current != schema.UNINITIALIZED
        and current < 39
        and any(revision.revision == 39 for revision in schema.pending(current))
    ):
        _reject_populated_pre_task_invocation_database(connection)
    if current == schema.UNINITIALIZED:
        _apply_baseline(connection)
        current = schema.BASELINE_REVISION
    for rev in schema.pending(current):
        _apply_revision(connection, rev)


def _apply_revision(connection: sqlite3.Connection, rev: schema.Revision) -> None:
    if rev.revision == 17:
        _apply_revision_seventeen(connection, rev)
        return
    if rev.revision == 23:
        with _transaction(connection):
            _validate_reservation_event_index(connection, require=False)
            _validate_pending_action_scope_indexes(connection, require_all=False)
            for statement in _iter_statements(_MIGRATION_STEPS[23]):
                connection.execute(statement)
            hook = _MIGRATION_HOOKS[23]
            hook(connection)
            _validate_reservation_event_index(connection, require=True)
            _validate_pending_action_scope_indexes(connection, require_all=True)
            _validate_reservation_identity_registry(
                connection,
                require=True,
                verify_event_ownership=True,
            )
            _record_revision(connection, rev)
            connection.execute(f"PRAGMA user_version = {rev.revision}")
        return
    if rev.revision == 29:
        with _transaction(connection):
            _validate_workflow_replay_indexes(connection, require_all=False)
            for statement in _iter_statements(_MIGRATION_STEPS[29]):
                connection.execute(statement)
            _validate_workflow_replay_indexes(connection, require_all=True)
            _record_revision(connection, rev)
            connection.execute(f"PRAGMA user_version = {rev.revision}")
        return
    if rev.revision == 38:
        with _transaction(connection):
            for statement in _iter_statements(_MIGRATION_STEPS[38]):
                connection.execute(statement)
            _validate_task_terminalization_receipt_table(connection)
            _record_revision(connection, rev)
            connection.execute(f"PRAGMA user_version = {rev.revision}")
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


def session_from_request(
    request: RunRequest,
    *,
    identity: SessionIdentity,
    parent_session: Session | None,
) -> Session:
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
        invocation=session_invocation_for_run_request(
            request,
            session_id=session_id,
            parent_session=parent_session,
        ),
        metadata=session_metadata_for_creation(request.metadata, identity=identity),
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
        json_dumps(session.invocation.model_dump(mode="json")),
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
        format_optional_datetime(task.available_at),
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
        json_dumps(task.invocation.model_dump(mode="json")),
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
        available_at=parse_optional_datetime(row["available_at"]),
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
        invocation=TaskInvocation.model_validate(json.loads(row["invocation_json"])),
    )


_TASK_TOPOLOGY_MAX_TIMESTAMP_BYTES = 128

TASK_TOPOLOGY_COLUMNS = f"""
    CASE
        WHEN length(CAST(id AS BLOB)) <= {TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES}
        THEN id
    END AS topology_id,
    length(CAST(id AS BLOB)) > {TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES}
        AS topology_id_oversized,
    CASE
        WHEN length(CAST(type AS BLOB)) <= {TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES}
        THEN type
    END AS topology_type,
    length(CAST(type AS BLOB)) > {TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES}
        AS topology_type_truncated,
    CASE
        WHEN title IS NULL
          OR length(CAST(title AS BLOB)) <= {TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES}
        THEN title
    END AS topology_title,
    title IS NOT NULL
      AND length(CAST(title AS BLOB)) > {TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES}
        AS topology_title_truncated,
    CASE
        WHEN length(CAST(status AS BLOB)) <= {TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES}
        THEN status
    END AS topology_status,
    CASE
        WHEN status_reason IS NULL
          OR length(CAST(status_reason AS BLOB)) <= {TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES}
        THEN status_reason
    END AS topology_status_reason,
    status_reason IS NOT NULL
      AND length(CAST(status_reason AS BLOB)) > {TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES}
        AS topology_status_reason_truncated,
    CASE
        WHEN session_id IS NULL
          OR length(CAST(session_id AS BLOB)) <= {TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES}
        THEN session_id
    END AS topology_session_id,
    session_id IS NOT NULL
      AND length(CAST(session_id AS BLOB)) > {TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES}
        AS topology_session_id_oversized,
    CASE
        WHEN parent_task_id IS NULL
          OR length(CAST(parent_task_id AS BLOB)) <= {TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES}
        THEN parent_task_id
    END AS topology_parent_task_id,
    parent_task_id IS NOT NULL
      AND length(CAST(parent_task_id AS BLOB)) > {TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES}
        AS topology_parent_task_id_oversized,
    CASE
        WHEN assigned_agent_name IS NULL
          OR length(CAST(assigned_agent_name AS BLOB))
             <= {TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES}
        THEN assigned_agent_name
    END AS topology_assigned_agent_name,
    assigned_agent_name IS NOT NULL
      AND length(CAST(assigned_agent_name AS BLOB))
          > {TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES}
        AS topology_assigned_agent_name_truncated,
    CASE
        WHEN length(CAST(created_at AS BLOB)) <= {_TASK_TOPOLOGY_MAX_TIMESTAMP_BYTES}
        THEN created_at
    END AS topology_created_at,
    CASE
        WHEN length(CAST(updated_at AS BLOB)) <= {_TASK_TOPOLOGY_MAX_TIMESTAMP_BYTES}
        THEN updated_at
    END AS topology_updated_at
"""


def task_topology_node_from_row(row: sqlite3.Row) -> TaskTopologyNode:
    if (
        row["topology_id_oversized"]
        or row["topology_session_id_oversized"]
        or row["topology_parent_task_id_oversized"]
    ):
        raise TaskTopologyInconsistent(
            "A task topology record contains an oversized structural identifier."
        )
    truncated_fields = tuple(
        field_name
        for field_name, column_name in (
            ("type", "topology_type_truncated"),
            ("title", "topology_title_truncated"),
            ("assigned_agent_name", "topology_assigned_agent_name_truncated"),
            ("status_reason", "topology_status_reason_truncated"),
        )
        if row[column_name]
    )
    try:
        return TaskTopologyNode(
            id=row["topology_id"],
            type=row["topology_type"],
            title=row["topology_title"],
            status=TaskStatus(row["topology_status"]),
            status_reason=row["topology_status_reason"],
            session_id=row["topology_session_id"],
            parent_task_id=row["topology_parent_task_id"],
            assigned_agent_name=row["topology_assigned_agent_name"],
            created_at=parse_datetime(row["topology_created_at"]),
            updated_at=parse_datetime(row["topology_updated_at"]),
            truncated_fields=truncated_fields,
        )
    except (TypeError, ValueError) as exc:
        raise TaskTopologyInconsistent(
            "A task record cannot be represented by the bounded topology contract."
        ) from exc


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
        invocation=SessionInvocation.model_validate(json.loads(row["invocation_json"])),
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
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def checkpoint_row_values(
    session_id: str,
    checkpoint: dict[str, Any],
    updated_at: datetime,
) -> tuple[object, ...]:
    from cayu.runtime.pending_actions import pending_action_checkpoint_metrics

    checkpoint = copy_durable_json_object(checkpoint, "checkpoint")
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
