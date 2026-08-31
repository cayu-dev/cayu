from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn, cast
from urllib.parse import quote
from uuid import uuid4

from cayu._validation import (
    copy_durable_json_object,
    copy_label_map,
    require_clean_nonblank,
    require_execution_unit_id,
)
from cayu.core.events import Event
from cayu.core.messages import Message
from cayu.runtime.invocation import SessionInvocation, TaskInvocation
from cayu.runtime.sessions import (
    PENDING_ACTION_EVENT_TYPE_VALUES,
    RUNTIME_BUILD_PROVENANCE_METADATA_KEY,
    TRANSCRIPT_SEARCH_TOKENIZER_VERSION,
    PendingActionSession,
    RunRequest,
    Session,
    SessionIdentity,
    SessionOrder,
    SessionStatus,
    deferred_interaction_input_from_storage_payload,
    deferred_interaction_input_storage_payload,
    runtime_build_provenance_from_session_metadata,
    session_instance_id_for_run_request,
    session_invocation_for_run_request,
    session_metadata_for_creation,
    transcript_search_document,
    transcript_search_session_token,
)
from cayu.runtime.tasks import (
    TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES,
    TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES,
    Task,
    TaskOrder,
    TaskRetrySeriesSnapshot,
    TaskStatus,
    TaskTopologyInconsistent,
    TaskTopologyNode,
)
from cayu.runtime.work_contracts import WorkContractRef
from cayu.storage import _session_store_sql as session_store_sql
from cayu.storage import migrations as schema
from cayu.storage.knowledge_transition import require_empty_knowledge_revision_transition
from cayu.storage.memory import (
    MAX_KNOWLEDGE_CHUNK_ID_BYTES,
    MAX_KNOWLEDGE_ENTRY_ID_BYTES,
)


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

    def transcript_text(message_json: object) -> str:
        if type(message_json) is not str:
            raise ValueError("Transcript message JSON must be text.")
        try:
            payload = json.loads(message_json)
            message = Message.model_validate(payload)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("Transcript message JSON is invalid.") from exc
        return transcript_search_document(message)

    def transcript_session_token(session_id: object) -> str:
        if type(session_id) is not str:
            raise ValueError("Transcript session id must be text.")
        return transcript_search_session_token(session_id)

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
    connection.create_function(
        "cayu_transcript_search_document",
        1,
        transcript_text,
        deterministic=True,
    )
    connection.create_function(
        "cayu_transcript_session_token",
        1,
        transcript_session_token,
        deterministic=True,
    )
    connection.create_function(
        "cayu_transcript_search_tokenizer_version",
        0,
        lambda: TRANSCRIPT_SEARCH_TOKENIZER_VERSION,
        deterministic=True,
    )
    connection.create_aggregate(
        "cayu_exact_usage_sum",
        13,
        cast("Any", _ExactUsageSum),
    )


class _ExactUsageSum:
    """Sum normalized counters and canonical outputs from a prior exact sum."""

    # A SQLite table has at most 2**63 - 1 rows and each accepted JSON integer is
    # at most 2**63 - 1, so every possible sum fits in 38 decimal digits.
    _DECIMAL_WIDTH = 38

    def __init__(self) -> None:
        self._totals = [0] * 13

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
        instance_id TEXT NOT NULL UNIQUE,
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
        file_attachment_attestations_runtime_owned INTEGER NOT NULL DEFAULT 0
            CHECK (file_attachment_attestations_runtime_owned IN (0, 1)),
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

    CREATE INDEX IF NOT EXISTS idx_cayu_public_authority_public_alias
        ON cayu_public_authority_aliases(field_name, public_alias);

    CREATE TABLE IF NOT EXISTS cayu_targeted_tool_grants (
        grant_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES cayu_sessions(id) ON DELETE CASCADE,
        interaction_id TEXT NOT NULL,
        request_id TEXT NOT NULL,
        tool_ref TEXT NOT NULL,
        generation_id TEXT NOT NULL,
        tool_id TEXT NOT NULL,
        tool_name TEXT NOT NULL,
        catalogue_revision TEXT NOT NULL,
        descriptor_version TEXT NOT NULL,
        issued_at TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        max_calls INTEGER NOT NULL CHECK (max_calls >= 1 AND max_calls <= 32),
        used_calls INTEGER NOT NULL DEFAULT 0
            CHECK (used_calls >= 0 AND used_calls <= max_calls),
        revoked_at TEXT,
        record_json TEXT NOT NULL CHECK (json_valid(record_json)),
        UNIQUE (session_id, interaction_id, request_id),
        UNIQUE (session_id, interaction_id, tool_id)
    );

    CREATE INDEX IF NOT EXISTS idx_cayu_targeted_tool_grants_interaction
        ON cayu_targeted_tool_grants(session_id, interaction_id, issued_at, grant_id);

    CREATE TABLE IF NOT EXISTS cayu_targeted_tool_grant_uses (
        use_id TEXT PRIMARY KEY,
        grant_id TEXT NOT NULL
            REFERENCES cayu_targeted_tool_grants(grant_id) ON DELETE CASCADE,
        session_id TEXT NOT NULL REFERENCES cayu_sessions(id) ON DELETE CASCADE,
        interaction_id TEXT NOT NULL,
        model_step_id TEXT NOT NULL,
        outer_tool_call_id TEXT NOT NULL,
        arguments_sha256 TEXT NOT NULL,
        invocation_id TEXT NOT NULL,
        bound_at TEXT NOT NULL,
        record_json TEXT NOT NULL CHECK (json_valid(record_json)),
        UNIQUE (session_id, interaction_id, invocation_id),
        UNIQUE (session_id, interaction_id, outer_tool_call_id)
    );

    CREATE INDEX IF NOT EXISTS idx_cayu_targeted_tool_grant_uses_grant
        ON cayu_targeted_tool_grant_uses(grant_id, bound_at, use_id);

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

    CREATE INDEX IF NOT EXISTS idx_cayu_checkpoints_queued_dispatch_run
        ON cayu_checkpoints(session_id)
        WHERE json_type(
            state_json,
            '$.session_run_operation.queue_task_id'
        ) IS NOT NULL;

    CREATE INDEX IF NOT EXISTS idx_cayu_checkpoints_queued_dispatch_receipts
        ON cayu_checkpoints(session_id)
        WHERE json_type(
            state_json,
            '$.queued_dispatch_terminal_receipts.receipts'
        ) IS NOT NULL;

    CREATE TABLE IF NOT EXISTS cayu_transcript_messages (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL REFERENCES cayu_sessions(id) ON DELETE CASCADE,
        role TEXT NOT NULL,
        interaction_id TEXT,
        session_order INTEGER,
        message_json TEXT NOT NULL,
        transcript_search_document TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS cayu_session_message_queue (
        ordering_key INTEGER PRIMARY KEY AUTOINCREMENT,
        queue_id TEXT NOT NULL UNIQUE,
        session_id TEXT NOT NULL REFERENCES cayu_sessions(id) ON DELETE CASCADE,
        idempotency_key TEXT NOT NULL,
        content TEXT NOT NULL,
        message_json TEXT CHECK (message_json IS NULL OR json_valid(message_json)),
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
        session_instance_id TEXT,
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
        invocation_json TEXT NOT NULL,
        retry_series_json TEXT
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

    CREATE TABLE IF NOT EXISTS cayu_task_interrupted_handoff_receipts (
        task_id TEXT NOT NULL,
        handoff_id TEXT NOT NULL,
        request_sha256 TEXT NOT NULL,
        request_json TEXT NOT NULL CHECK (json_valid(request_json)),
        task_json TEXT NOT NULL CHECK (json_valid(task_json)),
        committed_at TEXT NOT NULL,
        PRIMARY KEY (task_id, handoff_id)
    );

    CREATE TABLE IF NOT EXISTS cayu_task_retry_settlements (
        task_id TEXT NOT NULL,
        idempotency_key TEXT NOT NULL,
        request_sha256 TEXT NOT NULL,
        receipt_json TEXT NOT NULL,
        committed_at TEXT NOT NULL,
        PRIMARY KEY (task_id, idempotency_key)
    );

    CREATE TABLE IF NOT EXISTS cayu_task_retry_reconciliation_rejections (
        task_id TEXT NOT NULL,
        reconciliation_idempotency_key TEXT NOT NULL,
        request_sha256 TEXT NOT NULL,
        record_json TEXT NOT NULL CHECK (json_valid(record_json)),
        recorded_at TEXT NOT NULL,
        PRIMARY KEY (task_id, reconciliation_idempotency_key)
    );

    CREATE TABLE IF NOT EXISTS cayu_recall_receipts (
        receipt_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES cayu_sessions(id) ON DELETE CASCADE,
        interaction_id TEXT NOT NULL,
        model_step_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        receipt_json TEXT NOT NULL CHECK (json_valid(receipt_json)),
        document_bytes INTEGER NOT NULL CHECK (
            document_bytes >= 1 AND document_bytes <= 256000
        )
    );

    CREATE TABLE IF NOT EXISTS cayu_context_exposures (
        exposure_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES cayu_sessions(id) ON DELETE CASCADE,
        interaction_id TEXT NOT NULL,
        model_step_id TEXT NOT NULL,
        model_attempt_id TEXT NOT NULL,
        provider_attempt_id TEXT NOT NULL,
        state TEXT NOT NULL CHECK (state IN (
            'planned', 'prepared', 'dispatch_started', 'acknowledged',
            'completed', 'failed', 'cancelled', 'indeterminate'
        )),
        state_revision INTEGER NOT NULL CHECK (
            state_revision >= 0 AND state_revision < 16
        ),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        exposure_json TEXT NOT NULL CHECK (json_valid(exposure_json)),
        document_bytes INTEGER NOT NULL CHECK (
            document_bytes >= 1 AND document_bytes <= 128000
        ),
        UNIQUE (session_id, model_attempt_id),
        UNIQUE (session_id, provider_attempt_id)
    );

    CREATE TABLE IF NOT EXISTS cayu_recall_item_exposures (
        exposure_id TEXT NOT NULL
            REFERENCES cayu_context_exposures(exposure_id) ON DELETE CASCADE,
        ordinal INTEGER NOT NULL CHECK (ordinal >= 0 AND ordinal < 64),
        receipt_id TEXT NOT NULL
            REFERENCES cayu_recall_receipts(receipt_id) ON DELETE CASCADE,
        receipt_item_ordinal INTEGER NOT NULL CHECK (
            receipt_item_ordinal >= 0 AND receipt_item_ordinal < 64
        ),
        item_json TEXT NOT NULL CHECK (json_valid(item_json)),
        document_bytes INTEGER NOT NULL CHECK (
            document_bytes >= 1 AND document_bytes <= 16384
        ),
        PRIMARY KEY (exposure_id, ordinal),
        UNIQUE (exposure_id, receipt_id, receipt_item_ordinal)
    );

    CREATE INDEX IF NOT EXISTS idx_cayu_recall_receipts_session_page
        ON cayu_recall_receipts(session_id, created_at, receipt_id);
    CREATE INDEX IF NOT EXISTS idx_cayu_recall_receipts_interaction_page
        ON cayu_recall_receipts(session_id, interaction_id, created_at, receipt_id);
    CREATE INDEX IF NOT EXISTS idx_cayu_recall_receipts_step_page
        ON cayu_recall_receipts(session_id, model_step_id, created_at, receipt_id);
    CREATE INDEX IF NOT EXISTS idx_cayu_recall_receipts_interaction_step_page
        ON cayu_recall_receipts(
            session_id, interaction_id, model_step_id, created_at, receipt_id
        );
    CREATE INDEX IF NOT EXISTS idx_cayu_context_exposures_session_page
        ON cayu_context_exposures(session_id, created_at, exposure_id);
    CREATE INDEX IF NOT EXISTS idx_cayu_context_exposures_interaction_page
        ON cayu_context_exposures(session_id, interaction_id, created_at, exposure_id);
    CREATE INDEX IF NOT EXISTS idx_cayu_context_exposures_step_page
        ON cayu_context_exposures(session_id, model_step_id, created_at, exposure_id);
    CREATE INDEX IF NOT EXISTS idx_cayu_context_exposures_interaction_step_page
        ON cayu_context_exposures(
            session_id, interaction_id, model_step_id, created_at, exposure_id
        );
    CREATE INDEX IF NOT EXISTS idx_cayu_recall_item_exposures_receipt
        ON cayu_recall_item_exposures(receipt_id, exposure_id, ordinal);

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
    45: """
        CREATE TABLE IF NOT EXISTS cayu_task_retry_settlements (
            task_id TEXT NOT NULL,
            idempotency_key TEXT NOT NULL,
            request_sha256 TEXT NOT NULL,
            receipt_json TEXT NOT NULL,
            committed_at TEXT NOT NULL,
            PRIMARY KEY (task_id, idempotency_key)
        );
    """,
    46: """
        CREATE TABLE IF NOT EXISTS cayu_transcript_search_configuration (
            singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
            tokenizer_version TEXT NOT NULL
        );

        INSERT OR IGNORE INTO cayu_transcript_search_configuration (
            singleton, tokenizer_version
        ) VALUES (1, cayu_transcript_search_tokenizer_version());

        CREATE VIRTUAL TABLE IF NOT EXISTS cayu_transcript_messages_fts
        USING fts5(session_token, message_text, content='');

        CREATE TRIGGER IF NOT EXISTS cayu_transcript_messages_fts_insert
        AFTER INSERT ON cayu_transcript_messages
        WHEN new.role IN ('user', 'assistant')
        BEGIN
            INSERT INTO cayu_transcript_messages_fts(
                rowid, session_token, message_text
            ) VALUES (
                new.sequence,
                cayu_transcript_session_token(new.session_id),
                new.transcript_search_document
            );
        END;

        CREATE TRIGGER IF NOT EXISTS cayu_transcript_messages_fts_delete
        AFTER DELETE ON cayu_transcript_messages
        WHEN old.role IN ('user', 'assistant')
        BEGIN
            INSERT INTO cayu_transcript_messages_fts(
                cayu_transcript_messages_fts,
                rowid,
                session_token,
                message_text
            ) VALUES (
                'delete',
                old.sequence,
                cayu_transcript_session_token(old.session_id),
                old.transcript_search_document
            );
        END;

        CREATE TRIGGER IF NOT EXISTS cayu_transcript_messages_fts_update
        AFTER UPDATE OF session_id, role, message_json
        ON cayu_transcript_messages
        BEGIN
            INSERT INTO cayu_transcript_messages_fts(
                cayu_transcript_messages_fts,
                rowid,
                session_token,
                message_text
            )
            SELECT
                'delete',
                old.sequence,
                cayu_transcript_session_token(old.session_id),
                old.transcript_search_document
            WHERE old.role IN ('user', 'assistant');

            UPDATE cayu_transcript_messages
            SET transcript_search_document =
                cayu_transcript_search_document(new.message_json)
            WHERE sequence = new.sequence;

            INSERT INTO cayu_transcript_messages_fts(
                rowid, session_token, message_text
            )
            SELECT
                new.sequence,
                cayu_transcript_session_token(new.session_id),
                cayu_transcript_search_document(new.message_json)
            WHERE new.role IN ('user', 'assistant');
        END;

        CREATE TRIGGER IF NOT EXISTS cayu_transcript_messages_search_document_insert
        BEFORE INSERT ON cayu_transcript_messages
        WHEN new.transcript_search_document IS NULL
             OR new.transcript_search_document
                <> cayu_transcript_search_document(new.message_json)
        BEGIN
            SELECT RAISE(ABORT, 'invalid transcript search document');
        END;

        CREATE TRIGGER IF NOT EXISTS cayu_transcript_messages_search_document_update
        BEFORE UPDATE OF transcript_search_document
        ON cayu_transcript_messages
        WHEN new.transcript_search_document IS NULL
             OR new.transcript_search_document
                <> cayu_transcript_search_document(new.message_json)
        BEGIN
            SELECT RAISE(ABORT, 'invalid transcript search document');
        END;
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
            message_json TEXT CHECK (message_json IS NULL OR json_valid(message_json)),
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
            invocation_json TEXT NOT NULL,
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
    40: """
        CREATE INDEX IF NOT EXISTS idx_cayu_checkpoints_queued_dispatch_run
            ON cayu_checkpoints(session_id)
            WHERE json_type(
                state_json,
                '$.session_run_operation.queue_task_id'
            ) IS NOT NULL;

        CREATE INDEX IF NOT EXISTS idx_cayu_checkpoints_queued_dispatch_receipts
            ON cayu_checkpoints(session_id)
            WHERE json_type(
                state_json,
                '$.queued_dispatch_terminal_receipts.receipts'
            ) IS NOT NULL;
    """,
    42: """
        DROP TABLE IF EXISTS cayu_knowledge_change_acknowledgements;
        DROP TABLE IF EXISTS cayu_knowledge_change_consumers;
        DROP TABLE IF EXISTS cayu_knowledge_change_labels;
        DROP TABLE IF EXISTS cayu_knowledge_change_audiences;
        DROP TABLE IF EXISTS cayu_knowledge_changes;
        DROP TABLE IF EXISTS cayu_knowledge_evidence;
        DROP VIEW IF EXISTS cayu_knowledge_current_entries;
        DROP TABLE IF EXISTS cayu_knowledge_chunks_fts;
        DROP TABLE IF EXISTS cayu_knowledge_publication_receipts;
        DROP TABLE IF EXISTS cayu_knowledge_chunks;
        DROP TABLE IF EXISTS cayu_knowledge_impact_targets;
        DROP TABLE IF EXISTS cayu_knowledge_aspects;
        DROP TABLE IF EXISTS cayu_knowledge_labels;
        DROP TABLE IF EXISTS cayu_knowledge_revisions;
        DROP TABLE IF EXISTS cayu_knowledge_entries;

        CREATE TABLE cayu_knowledge_entries (
            id TEXT PRIMARY KEY,
            namespace TEXT NOT NULL,
            current_revision INTEGER NOT NULL
                CHECK (current_revision > 0 AND current_revision <= 2147483647),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (id, current_revision)
                REFERENCES cayu_knowledge_revisions(entry_id, revision)
                DEFERRABLE INITIALLY DEFERRED
        );

        CREATE TABLE cayu_knowledge_revisions (
            entry_id TEXT NOT NULL REFERENCES cayu_knowledge_entries(id) ON DELETE CASCADE,
            revision INTEGER NOT NULL CHECK (revision > 0 AND revision <= 2147483647),
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
            metadata_json TEXT NOT NULL,
            PRIMARY KEY (entry_id, revision)
        );

        CREATE TABLE cayu_knowledge_labels (
            entry_id TEXT NOT NULL,
            entry_revision INTEGER NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            PRIMARY KEY (entry_id, entry_revision, key),
            FOREIGN KEY (entry_id, entry_revision)
                REFERENCES cayu_knowledge_revisions(entry_id, revision) ON DELETE CASCADE
        );

        CREATE TABLE cayu_knowledge_aspects (
            entry_id TEXT NOT NULL,
            entry_revision INTEGER NOT NULL,
            aspect TEXT NOT NULL,
            PRIMARY KEY (entry_id, entry_revision, aspect),
            FOREIGN KEY (entry_id, entry_revision)
                REFERENCES cayu_knowledge_revisions(entry_id, revision) ON DELETE CASCADE
        );

        CREATE TABLE cayu_knowledge_impact_targets (
            entry_id TEXT NOT NULL,
            entry_revision INTEGER NOT NULL,
            impact_target TEXT NOT NULL,
            PRIMARY KEY (entry_id, entry_revision, impact_target),
            FOREIGN KEY (entry_id, entry_revision)
                REFERENCES cayu_knowledge_revisions(entry_id, revision) ON DELETE CASCADE
        );

        CREATE TABLE cayu_knowledge_chunks (
            fts_rowid INTEGER PRIMARY KEY,
            id TEXT NOT NULL UNIQUE,
            entry_id TEXT NOT NULL,
            entry_revision INTEGER NOT NULL
                CHECK (entry_revision > 0 AND entry_revision <= 2147483647),
            chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
            text TEXT NOT NULL,
            content_hash TEXT,
            source_uri TEXT,
            metadata_json TEXT NOT NULL,
            UNIQUE (entry_id, entry_revision, chunk_index),
            FOREIGN KEY (entry_id, entry_revision)
                REFERENCES cayu_knowledge_revisions(entry_id, revision) ON DELETE CASCADE
        );

        CREATE VIRTUAL TABLE cayu_knowledge_chunks_fts
        USING fts5(
            entry_id UNINDEXED,
            entry_revision UNINDEXED,
            chunk_id UNINDEXED,
            title,
            text
        );

        CREATE TABLE cayu_knowledge_publication_receipts (
            operation_id TEXT PRIMARY KEY,
            entry_id TEXT NOT NULL,
            entry_revision INTEGER NOT NULL
                CHECK (entry_revision > 0 AND entry_revision <= 2147483647),
            expected_revision INTEGER
                CHECK (expected_revision > 0 AND expected_revision <= 2147483647),
            request_sha256 TEXT NOT NULL,
            entry_created_at TEXT NOT NULL,
            entry_updated_at TEXT NOT NULL,
            committed_at TEXT NOT NULL,
            access_snapshot_json TEXT NOT NULL,
            CHECK (
                (expected_revision IS NULL AND entry_revision = 1)
                OR entry_revision = expected_revision + 1
            )
        );

        CREATE VIEW cayu_knowledge_current_entries AS
        SELECT
            logical.id AS id,
            revision.revision AS revision,
            logical.namespace AS namespace,
            revision.text AS text,
            revision.kind AS kind,
            revision.visibility AS visibility,
            revision.status AS status,
            revision.created_by_type AS created_by_type,
            revision.created_by AS created_by,
            revision.created_at AS created_at,
            revision.updated_at AS updated_at,
            revision.source_type AS source_type,
            revision.source_uri AS source_uri,
            revision.source_id AS source_id,
            revision.source_hash AS source_hash,
            revision.importance AS importance,
            revision.importance_source AS importance_source,
            revision.confidence AS confidence,
            revision.last_used_at AS last_used_at,
            revision.expires_at AS expires_at,
            revision.title AS title,
            revision.metadata_json AS metadata_json
        FROM cayu_knowledge_entries AS logical
        JOIN cayu_knowledge_revisions AS revision
          ON revision.entry_id = logical.id
         AND revision.revision = logical.current_revision;

        CREATE INDEX idx_cayu_knowledge_entries_namespace_current
            ON cayu_knowledge_entries(namespace, current_revision, id);
        CREATE INDEX idx_cayu_knowledge_revisions_status
            ON cayu_knowledge_revisions(status, entry_id, revision);
        CREATE INDEX idx_cayu_knowledge_revisions_kind
            ON cayu_knowledge_revisions(kind, entry_id, revision);
        CREATE INDEX idx_cayu_knowledge_revisions_visibility
            ON cayu_knowledge_revisions(visibility, entry_id, revision);
        CREATE INDEX idx_cayu_knowledge_revisions_source
            ON cayu_knowledge_revisions(source_type, source_id, entry_id, revision);
        CREATE INDEX idx_cayu_knowledge_revisions_expires_at
            ON cayu_knowledge_revisions(expires_at, entry_id, revision);
        CREATE INDEX idx_cayu_knowledge_labels_key_value_entry
            ON cayu_knowledge_labels(key, value, entry_id, entry_revision);
        CREATE INDEX idx_cayu_knowledge_aspects_aspect_entry
            ON cayu_knowledge_aspects(aspect, entry_id, entry_revision);
        CREATE INDEX idx_cayu_knowledge_impact_targets_target_entry
            ON cayu_knowledge_impact_targets(impact_target, entry_id, entry_revision);
        CREATE INDEX idx_cayu_knowledge_chunks_entry_revision_index
            ON cayu_knowledge_chunks(entry_id, entry_revision, chunk_index);
        CREATE INDEX idx_cayu_knowledge_publication_receipts_entry_revision
            ON cayu_knowledge_publication_receipts(entry_id, entry_revision);
    """,
    43: """
        CREATE UNIQUE INDEX idx_cayu_knowledge_chunks_identity_owner
            ON cayu_knowledge_chunks(id, entry_id, entry_revision);

        CREATE TABLE cayu_knowledge_evidence (
            id TEXT PRIMARY KEY,
            entry_id TEXT NOT NULL,
            entry_revision INTEGER NOT NULL
                CHECK (entry_revision > 0 AND entry_revision <= 2147483647),
            chunk_id TEXT,
            role TEXT NOT NULL CHECK (role IN ('origin', 'supporting')),
            source_type TEXT NOT NULL,
            source_id TEXT,
            source_uri TEXT,
            source_revision TEXT,
            source_hash TEXT,
            locator_json TEXT NOT NULL,
            disposition TEXT NOT NULL
                CHECK (disposition IN ('live', 'detached', 'retained')),
            created_at TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            CHECK (source_id IS NOT NULL OR source_uri IS NOT NULL),
            CHECK (source_revision IS NOT NULL OR source_hash IS NOT NULL),
            FOREIGN KEY (entry_id, entry_revision)
                REFERENCES cayu_knowledge_revisions(entry_id, revision) ON DELETE CASCADE,
            FOREIGN KEY (chunk_id, entry_id, entry_revision)
                REFERENCES cayu_knowledge_chunks(id, entry_id, entry_revision)
                ON DELETE CASCADE
        );

        CREATE INDEX idx_cayu_knowledge_evidence_entry_revision
            ON cayu_knowledge_evidence(entry_id, entry_revision, id COLLATE BINARY);
        CREATE INDEX idx_cayu_knowledge_evidence_source
            ON cayu_knowledge_evidence(source_type, source_id, entry_id, entry_revision);

        CREATE TABLE cayu_knowledge_changes (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT
                CHECK (sequence > 0 AND sequence <= 9223372036854775807),
            id TEXT NOT NULL UNIQUE,
            kind TEXT NOT NULL CHECK (
                kind IN (
                    'created',
                    'revision_appended',
                    'status_transitioned',
                    'tombstoned',
                    'hard_deleted',
                    'expired'
                )
            ),
            entry_id TEXT NOT NULL,
            entry_revision INTEGER NOT NULL
                CHECK (entry_revision > 0 AND entry_revision <= 2147483647),
            committed_at TEXT NOT NULL,
            operation_id TEXT
        );

        CREATE TABLE cayu_knowledge_change_audiences (
            change_sequence INTEGER NOT NULL,
            audience_kind TEXT NOT NULL CHECK (audience_kind IN ('before', 'after')),
            namespace TEXT NOT NULL,
            visibility TEXT NOT NULL,
            source_type TEXT,
            source_id TEXT,
            status TEXT NOT NULL,
            requires_include_expired INTEGER NOT NULL CHECK (
                requires_include_expired IN (0, 1)
            ),
            PRIMARY KEY (change_sequence, audience_kind),
            FOREIGN KEY (change_sequence)
                REFERENCES cayu_knowledge_changes(sequence) ON DELETE CASCADE
        );

        CREATE INDEX idx_cayu_knowledge_changes_entry_revision
            ON cayu_knowledge_changes(entry_id, entry_revision, sequence);
        CREATE UNIQUE INDEX idx_cayu_knowledge_changes_operation
            ON cayu_knowledge_changes(operation_id)
            WHERE operation_id IS NOT NULL;

        CREATE INDEX idx_cayu_knowledge_change_audiences_namespace
            ON cayu_knowledge_change_audiences(namespace, change_sequence, audience_kind);
        CREATE INDEX idx_cayu_knowledge_change_audiences_status
            ON cayu_knowledge_change_audiences(status, change_sequence, audience_kind);
        CREATE INDEX idx_cayu_knowledge_change_audiences_source
            ON cayu_knowledge_change_audiences(
                source_type, source_id, change_sequence, audience_kind
            );

        CREATE TABLE cayu_knowledge_change_labels (
            change_sequence INTEGER NOT NULL,
            audience_kind TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            PRIMARY KEY (change_sequence, audience_kind, key),
            FOREIGN KEY (change_sequence, audience_kind)
                REFERENCES cayu_knowledge_change_audiences(
                    change_sequence, audience_kind
                ) ON DELETE CASCADE
        );

        CREATE INDEX idx_cayu_knowledge_change_labels_lookup
            ON cayu_knowledge_change_labels(
                key, value, change_sequence, audience_kind
            );

        CREATE TABLE cayu_knowledge_change_consumers (
            consumer_id TEXT PRIMARY KEY,
            access_scope_sha256 TEXT NOT NULL,
            cursor_sequence INTEGER NOT NULL DEFAULT 0
                CHECK (cursor_sequence >= 0),
            pending_change_sequence INTEGER,
            pending_claim_id TEXT,
            pending_worker_id TEXT,
            pending_attempt INTEGER NOT NULL DEFAULT 0
                CHECK (pending_attempt >= 0),
            claimed_at TEXT,
            lease_expires_at TEXT,
            last_acknowledged_claim_id TEXT,
            updated_at TEXT NOT NULL,
            CHECK (
                (pending_change_sequence IS NULL
                    AND pending_claim_id IS NULL
                    AND pending_worker_id IS NULL
                    AND claimed_at IS NULL
                    AND lease_expires_at IS NULL)
                OR
                (pending_change_sequence IS NOT NULL
                    AND pending_change_sequence > cursor_sequence
                    AND pending_claim_id IS NOT NULL
                    AND pending_worker_id IS NOT NULL
                    AND pending_attempt > 0
                    AND claimed_at IS NOT NULL
                    AND lease_expires_at IS NOT NULL
                    AND lease_expires_at > claimed_at)
            ),
            FOREIGN KEY (pending_change_sequence)
                REFERENCES cayu_knowledge_changes(sequence)
        );

        CREATE INDEX idx_cayu_knowledge_change_consumers_lease
            ON cayu_knowledge_change_consumers(lease_expires_at)
            WHERE pending_change_sequence IS NOT NULL;

        CREATE TABLE cayu_knowledge_change_acknowledgements (
            consumer_id TEXT NOT NULL,
            claim_id TEXT NOT NULL,
            claim_sha256 TEXT NOT NULL CHECK (
                length(claim_sha256) = 64
                AND claim_sha256 NOT GLOB '*[^0-9a-f]*'
            ),
            change_sequence INTEGER NOT NULL,
            acknowledged_at TEXT NOT NULL,
            PRIMARY KEY (consumer_id, claim_id),
            FOREIGN KEY (consumer_id)
                REFERENCES cayu_knowledge_change_consumers(consumer_id) ON DELETE CASCADE,
            FOREIGN KEY (change_sequence)
                REFERENCES cayu_knowledge_changes(sequence)
        );
    """,
    44: """
        CREATE TABLE IF NOT EXISTS cayu_knowledge_index_readiness_events (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT
                CHECK (sequence > 0 AND sequence <= 9223372036854775807),
            identity_sha256 TEXT NOT NULL CHECK (
                length(identity_sha256) = 64
                AND identity_sha256 NOT GLOB '*[^0-9a-f]*'
            ),
            entry_id TEXT NOT NULL,
            entry_revision INTEGER NOT NULL
                CHECK (entry_revision > 0 AND entry_revision <= 2147483647),
            chunk_id TEXT,
            projection_type TEXT NOT NULL,
            projection_content_hash TEXT NOT NULL,
            embedding_model TEXT NOT NULL,
            dimensions INTEGER NOT NULL CHECK (dimensions > 0),
            preprocessing_version TEXT NOT NULL,
            generator TEXT NOT NULL,
            generator_version TEXT NOT NULL,
            index_representation_version TEXT NOT NULL,
            state TEXT NOT NULL CHECK (state IN ('pending', 'ready', 'failed')),
            attempt_id TEXT NOT NULL,
            failure_code TEXT,
            operation_id TEXT NOT NULL UNIQUE,
            update_sha256 TEXT NOT NULL CHECK (
                length(update_sha256) = 64
                AND update_sha256 NOT GLOB '*[^0-9a-f]*'
            ),
            published_at TEXT NOT NULL,
            CHECK (
                (state = 'failed' AND failure_code IS NOT NULL)
                OR (state <> 'failed' AND failure_code IS NULL)
            ),
            UNIQUE (identity_sha256, sequence)
        );

        CREATE TABLE IF NOT EXISTS cayu_knowledge_index_readiness_current (
            identity_sha256 TEXT PRIMARY KEY,
            sequence INTEGER NOT NULL UNIQUE,
            FOREIGN KEY (identity_sha256, sequence)
                REFERENCES cayu_knowledge_index_readiness_events(
                    identity_sha256, sequence
                )
                ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_index_readiness_identity_sequence
            ON cayu_knowledge_index_readiness_events(identity_sha256, sequence);
        CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_index_readiness_entry_revision
            ON cayu_knowledge_index_readiness_events(
                entry_id, entry_revision, projection_type, sequence
            );
        CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_index_readiness_projection_lookup
            ON cayu_knowledge_index_readiness_events(
                entry_id, entry_revision, chunk_id, projection_type,
                embedding_model, dimensions, sequence
            );
    """,
    47: """
        CREATE TABLE IF NOT EXISTS cayu_eval_result_records (
            revision TEXT PRIMARY KEY,
            origin TEXT NOT NULL CHECK (origin IN ('captured_session', 'fresh_execution')),
            target_key TEXT NOT NULL,
            corpus_revision TEXT NOT NULL,
            suite_id TEXT COLLATE BINARY NOT NULL,
            suite_revision TEXT NOT NULL,
            application_release_id TEXT NOT NULL,
            app_manifest_schema_version TEXT NOT NULL,
            app_manifest_fingerprint TEXT NOT NULL CHECK (
                length(app_manifest_fingerprint) = 64
                AND app_manifest_fingerprint NOT GLOB '*[^0-9a-f]*'
            ),
            result_status TEXT NOT NULL CHECK (
                result_status IN ('passed', 'failed', 'unavailable', 'error')
            ),
            result_score REAL CHECK (
                result_score IS NULL OR (result_score >= 0.0 AND result_score <= 1.0)
            ),
            fresh_run_id TEXT UNIQUE REFERENCES cayu_eval_results(run_id) ON DELETE RESTRICT,
            captured_result_json TEXT,
            document_bytes INTEGER NOT NULL
                CHECK (document_bytes >= 1 AND document_bytes <= 41943040),
            created_at TEXT NOT NULL,
            CHECK (
                (result_status IN ('passed', 'failed') AND result_score IS NOT NULL)
                OR (result_status IN ('unavailable', 'error') AND result_score IS NULL)
            ),
            CHECK (
                (origin = 'fresh_execution' AND fresh_run_id IS NOT NULL
                    AND captured_result_json IS NULL)
                OR (origin = 'captured_session' AND fresh_run_id IS NULL
                    AND captured_result_json IS NOT NULL
                    AND document_bytes = length(CAST(captured_result_json AS BLOB)))
            ),
            FOREIGN KEY (corpus_revision, suite_id)
                REFERENCES cayu_eval_suites(corpus_revision, suite_id)
        );
        CREATE INDEX IF NOT EXISTS idx_cayu_eval_result_records_target_catalog
            ON cayu_eval_result_records(target_key, created_at DESC, revision ASC);
        CREATE INDEX IF NOT EXISTS idx_cayu_eval_result_records_contract
            ON cayu_eval_result_records(
                target_key, corpus_revision, suite_id, created_at DESC, revision ASC
            );

        INSERT OR IGNORE INTO cayu_eval_result_records (
            revision, origin, target_key, corpus_revision, suite_id, suite_revision,
            application_release_id, app_manifest_schema_version,
            app_manifest_fingerprint, result_status, result_score, fresh_run_id,
            captured_result_json, document_bytes, created_at
        )
        SELECT
            result.revision, 'fresh_execution', run.target_key, run.corpus_revision,
            run.suite_id, run.suite_revision,
            json_extract(result.result_json, '$.target.application_release_id'),
            json_extract(result.result_json, '$.target.app_manifest.schema_version'),
            json_extract(result.result_json, '$.target.app_manifest.fingerprint'),
            run.result_status, run.result_score, result.run_id, NULL,
            result.result_bytes, result.created_at
        FROM cayu_eval_results AS result
        JOIN cayu_eval_runs AS run ON run.run_id = result.run_id;

        CREATE TABLE IF NOT EXISTS cayu_eval_baselines (
            target_key TEXT NOT NULL,
            corpus_revision TEXT NOT NULL,
            suite_id TEXT COLLATE BINARY NOT NULL,
            result_revision TEXT NOT NULL
                REFERENCES cayu_eval_result_records(revision) ON DELETE RESTRICT,
            generation INTEGER NOT NULL
                CHECK (generation >= 1 AND generation <= 9223372036854775807),
            updated_by TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (target_key, corpus_revision, suite_id),
            FOREIGN KEY (corpus_revision, suite_id)
                REFERENCES cayu_eval_suites(corpus_revision, suite_id)
        );

        CREATE TABLE IF NOT EXISTS cayu_eval_baseline_mutations (
            operation_id TEXT PRIMARY KEY,
            target_key TEXT NOT NULL,
            corpus_revision TEXT NOT NULL,
            suite_id TEXT COLLATE BINARY NOT NULL,
            expected_generation INTEGER NOT NULL
                CHECK (expected_generation >= 0
                    AND expected_generation < 9223372036854775807),
            previous_result_revision TEXT,
            selected_result_revision TEXT NOT NULL
                REFERENCES cayu_eval_result_records(revision) ON DELETE RESTRICT,
            resulting_generation INTEGER NOT NULL
                CHECK (resulting_generation >= 1
                    AND resulting_generation <= 9223372036854775807),
            actor_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            CHECK (resulting_generation = expected_generation + 1),
            CHECK (
                (expected_generation = 0 AND previous_result_revision IS NULL)
                OR (expected_generation > 0 AND previous_result_revision IS NOT NULL)
            ),
            FOREIGN KEY (previous_result_revision)
                REFERENCES cayu_eval_result_records(revision) ON DELETE RESTRICT,
            FOREIGN KEY (corpus_revision, suite_id)
                REFERENCES cayu_eval_suites(corpus_revision, suite_id)
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_cayu_eval_baseline_mutations_scope
            ON cayu_eval_baseline_mutations(
                target_key, corpus_revision, suite_id, resulting_generation
            );
    """,
    48: """
        ALTER TABLE cayu_eval_cases RENAME TO cayu_eval_cases_revision_47;
        DROP INDEX idx_cayu_eval_cases_suite;
        CREATE TABLE cayu_eval_cases (
            corpus_revision TEXT NOT NULL,
            case_id TEXT COLLATE BINARY NOT NULL,
            case_revision TEXT NOT NULL,
            suite_id TEXT COLLATE BINARY NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            message_count INTEGER NOT NULL
                CHECK (message_count >= 0 AND message_count <= 16),
            assertion_count INTEGER NOT NULL
                CHECK (assertion_count >= 1 AND assertion_count <= 64),
            PRIMARY KEY (corpus_revision, case_id),
            FOREIGN KEY (corpus_revision, suite_id)
                REFERENCES cayu_eval_suites(corpus_revision, suite_id) ON DELETE CASCADE
        );
        INSERT INTO cayu_eval_cases (
            corpus_revision, case_id, case_revision, suite_id, name,
            description, message_count, assertion_count
        )
        SELECT
            corpus_revision, case_id, case_revision, suite_id, name,
            description, message_count, assertion_count
        FROM cayu_eval_cases_revision_47;
        DROP TABLE cayu_eval_cases_revision_47;
        CREATE INDEX idx_cayu_eval_cases_suite
            ON cayu_eval_cases(corpus_revision, suite_id, case_id ASC);
    """,
    49: """
        CREATE TABLE IF NOT EXISTS cayu_work_contracts (
            contract_id TEXT NOT NULL,
            version INTEGER NOT NULL CHECK (version >= 1),
            fingerprint TEXT NOT NULL CHECK (
                length(fingerprint) = 64
                AND fingerprint NOT GLOB '*[^0-9a-f]*'
            ),
            contract_json TEXT NOT NULL CHECK (json_valid(contract_json)),
            PRIMARY KEY (contract_id, version)
        );

        CREATE TABLE IF NOT EXISTS cayu_task_session_execution_authority (
            session_id TEXT NOT NULL PRIMARY KEY,
            authority_kind TEXT NOT NULL CHECK (
                authority_kind IN ('ordinary', 'contracted')
            ),
            committed_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS cayu_work_attempts (
            attempt_id TEXT NOT NULL PRIMARY KEY,
            task_id TEXT NOT NULL REFERENCES cayu_tasks(id) ON DELETE RESTRICT,
            ordinal INTEGER NOT NULL CHECK (ordinal >= 1),
            request_sha256 TEXT NOT NULL CHECK (
                length(request_sha256) = 64
                AND request_sha256 NOT GLOB '*[^0-9a-f]*'
            ),
            started_at TEXT NOT NULL,
            attempt_json TEXT NOT NULL CHECK (json_valid(attempt_json)),
            UNIQUE (task_id, ordinal)
        );

        CREATE TABLE IF NOT EXISTS cayu_completion_proposals (
            proposal_id TEXT NOT NULL PRIMARY KEY,
            attempt_id TEXT NOT NULL UNIQUE
                REFERENCES cayu_work_attempts(attempt_id) ON DELETE RESTRICT,
            task_id TEXT NOT NULL REFERENCES cayu_tasks(id) ON DELETE RESTRICT,
            request_sha256 TEXT NOT NULL CHECK (
                length(request_sha256) = 64
                AND request_sha256 NOT GLOB '*[^0-9a-f]*'
            ),
            proposed_at TEXT NOT NULL,
            proposal_json TEXT NOT NULL CHECK (json_valid(proposal_json))
        );

        CREATE TABLE IF NOT EXISTS cayu_completion_verification_claims (
            claim_id TEXT NOT NULL PRIMARY KEY,
            proposal_id TEXT NOT NULL
                REFERENCES cayu_completion_proposals(proposal_id) ON DELETE RESTRICT,
            attempt_number INTEGER NOT NULL CHECK (attempt_number >= 1),
            request_sha256 TEXT NOT NULL CHECK (
                length(request_sha256) = 64
                AND request_sha256 NOT GLOB '*[^0-9a-f]*'
            ),
            lease_expires_at TEXT NOT NULL,
            is_current INTEGER NOT NULL CHECK (is_current IN (0, 1)),
            claim_json TEXT NOT NULL CHECK (json_valid(claim_json)),
            UNIQUE (proposal_id, attempt_number)
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_cayu_completion_claim_current
            ON cayu_completion_verification_claims(proposal_id)
            WHERE is_current = 1;

        CREATE TABLE IF NOT EXISTS cayu_completion_decisions (
            decision_id TEXT NOT NULL PRIMARY KEY,
            proposal_id TEXT NOT NULL UNIQUE
                REFERENCES cayu_completion_proposals(proposal_id) ON DELETE RESTRICT,
            task_id TEXT NOT NULL REFERENCES cayu_tasks(id) ON DELETE RESTRICT,
            attempt_id TEXT NOT NULL
                REFERENCES cayu_work_attempts(attempt_id) ON DELETE RESTRICT,
            claim_id TEXT NOT NULL
                REFERENCES cayu_completion_verification_claims(claim_id) ON DELETE RESTRICT,
            verdict TEXT NOT NULL CHECK (
                verdict IN ('accepted', 'rejected', 'blocked', 'needs_review')
            ),
            gap_fingerprint TEXT NOT NULL CHECK (
                length(gap_fingerprint) = 64
                AND gap_fingerprint NOT GLOB '*[^0-9a-f]*'
            ),
            request_sha256 TEXT NOT NULL CHECK (
                length(request_sha256) = 64
                AND request_sha256 NOT GLOB '*[^0-9a-f]*'
            ),
            decided_at TEXT NOT NULL,
            decision_json TEXT NOT NULL CHECK (json_valid(decision_json))
        );
        CREATE INDEX IF NOT EXISTS idx_cayu_completion_decisions_task_gap
            ON cayu_completion_decisions(task_id, verdict, gap_fingerprint);

        CREATE TABLE IF NOT EXISTS cayu_completion_decision_application_receipts (
            task_id TEXT NOT NULL REFERENCES cayu_tasks(id) ON DELETE RESTRICT,
            idempotency_key TEXT NOT NULL,
            decision_id TEXT NOT NULL UNIQUE
                REFERENCES cayu_completion_decisions(decision_id) ON DELETE RESTRICT,
            request_sha256 TEXT NOT NULL CHECK (
                length(request_sha256) = 64
                AND request_sha256 NOT GLOB '*[^0-9a-f]*'
            ),
            applied_at TEXT NOT NULL,
            receipt_json TEXT NOT NULL CHECK (json_valid(receipt_json)),
            PRIMARY KEY (task_id, idempotency_key)
        );

        CREATE INDEX IF NOT EXISTS idx_cayu_tasks_contracted_session
            ON cayu_tasks(session_id, created_at, id)
            WHERE work_contract_json IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_cayu_work_attempts_task_latest
            ON cayu_work_attempts(task_id, ordinal DESC);
    """,
    51: """
        CREATE TABLE IF NOT EXISTS cayu_recall_receipts (
            receipt_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES cayu_sessions(id) ON DELETE CASCADE,
            interaction_id TEXT NOT NULL,
            model_step_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            receipt_json TEXT NOT NULL CHECK (json_valid(receipt_json)),
            document_bytes INTEGER NOT NULL CHECK (
                document_bytes >= 1 AND document_bytes <= 256000
            )
        );
        CREATE TABLE IF NOT EXISTS cayu_context_exposures (
            exposure_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES cayu_sessions(id) ON DELETE CASCADE,
            interaction_id TEXT NOT NULL,
            model_step_id TEXT NOT NULL,
            model_attempt_id TEXT NOT NULL,
            provider_attempt_id TEXT NOT NULL,
            state TEXT NOT NULL CHECK (state IN (
                'planned', 'prepared', 'dispatch_started', 'acknowledged',
                'completed', 'failed', 'cancelled', 'indeterminate'
            )),
            state_revision INTEGER NOT NULL CHECK (
                state_revision >= 0 AND state_revision < 16
            ),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            exposure_json TEXT NOT NULL CHECK (json_valid(exposure_json)),
            document_bytes INTEGER NOT NULL CHECK (
                document_bytes >= 1 AND document_bytes <= 128000
            ),
            UNIQUE (session_id, model_attempt_id),
            UNIQUE (session_id, provider_attempt_id)
        );
        CREATE TABLE IF NOT EXISTS cayu_recall_item_exposures (
            exposure_id TEXT NOT NULL
                REFERENCES cayu_context_exposures(exposure_id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL CHECK (ordinal >= 0 AND ordinal < 64),
            receipt_id TEXT NOT NULL
                REFERENCES cayu_recall_receipts(receipt_id) ON DELETE CASCADE,
            receipt_item_ordinal INTEGER NOT NULL CHECK (
                receipt_item_ordinal >= 0 AND receipt_item_ordinal < 64
            ),
            item_json TEXT NOT NULL CHECK (json_valid(item_json)),
            document_bytes INTEGER NOT NULL CHECK (
                document_bytes >= 1 AND document_bytes <= 16384
            ),
            PRIMARY KEY (exposure_id, ordinal),
            UNIQUE (exposure_id, receipt_id, receipt_item_ordinal)
        );
        CREATE INDEX IF NOT EXISTS idx_cayu_recall_receipts_session_page
            ON cayu_recall_receipts(session_id, created_at, receipt_id);
        CREATE INDEX IF NOT EXISTS idx_cayu_recall_receipts_interaction_page
            ON cayu_recall_receipts(session_id, interaction_id, created_at, receipt_id);
        CREATE INDEX IF NOT EXISTS idx_cayu_recall_receipts_step_page
            ON cayu_recall_receipts(session_id, model_step_id, created_at, receipt_id);
        CREATE INDEX IF NOT EXISTS idx_cayu_recall_receipts_interaction_step_page
            ON cayu_recall_receipts(
                session_id, interaction_id, model_step_id, created_at, receipt_id
            );
        CREATE INDEX IF NOT EXISTS idx_cayu_context_exposures_session_page
            ON cayu_context_exposures(session_id, created_at, exposure_id);
        CREATE INDEX IF NOT EXISTS idx_cayu_context_exposures_interaction_page
            ON cayu_context_exposures(session_id, interaction_id, created_at, exposure_id);
        CREATE INDEX IF NOT EXISTS idx_cayu_context_exposures_step_page
            ON cayu_context_exposures(session_id, model_step_id, created_at, exposure_id);
        CREATE INDEX IF NOT EXISTS idx_cayu_context_exposures_interaction_step_page
            ON cayu_context_exposures(
                session_id, interaction_id, model_step_id, created_at, exposure_id
            );
        CREATE INDEX IF NOT EXISTS idx_cayu_recall_item_exposures_receipt
            ON cayu_recall_item_exposures(receipt_id, exposure_id, ordinal);
    """,
    52: """
        CREATE INDEX IF NOT EXISTS idx_cayu_public_authority_public_alias
            ON cayu_public_authority_aliases(field_name, public_alias);
        CREATE TABLE IF NOT EXISTS cayu_targeted_tool_grants (
            grant_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES cayu_sessions(id) ON DELETE CASCADE,
            interaction_id TEXT NOT NULL,
            request_id TEXT NOT NULL,
            tool_ref TEXT NOT NULL,
            generation_id TEXT NOT NULL,
            tool_id TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            catalogue_revision TEXT NOT NULL,
            descriptor_version TEXT NOT NULL,
            issued_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            max_calls INTEGER NOT NULL CHECK (max_calls >= 1 AND max_calls <= 32),
            used_calls INTEGER NOT NULL DEFAULT 0
                CHECK (used_calls >= 0 AND used_calls <= max_calls),
            revoked_at TEXT,
            record_json TEXT NOT NULL CHECK (json_valid(record_json)),
            UNIQUE (session_id, interaction_id, request_id),
            UNIQUE (session_id, interaction_id, tool_id)
        );
        CREATE INDEX IF NOT EXISTS idx_cayu_targeted_tool_grants_interaction
            ON cayu_targeted_tool_grants(session_id, interaction_id, issued_at, grant_id);
        CREATE TABLE IF NOT EXISTS cayu_targeted_tool_grant_uses (
            use_id TEXT PRIMARY KEY,
            grant_id TEXT NOT NULL
                REFERENCES cayu_targeted_tool_grants(grant_id) ON DELETE CASCADE,
            session_id TEXT NOT NULL REFERENCES cayu_sessions(id) ON DELETE CASCADE,
            interaction_id TEXT NOT NULL,
            model_step_id TEXT NOT NULL,
            outer_tool_call_id TEXT NOT NULL,
            arguments_sha256 TEXT NOT NULL,
            invocation_id TEXT NOT NULL,
            bound_at TEXT NOT NULL,
            record_json TEXT NOT NULL CHECK (json_valid(record_json)),
            UNIQUE (session_id, interaction_id, invocation_id),
            UNIQUE (session_id, interaction_id, outer_tool_call_id)
        );
        CREATE INDEX IF NOT EXISTS idx_cayu_targeted_tool_grant_uses_grant
            ON cayu_targeted_tool_grant_uses(grant_id, bound_at, use_id);
    """,
    53: """
        CREATE TABLE IF NOT EXISTS cayu_eval_scenarios (
            revision TEXT PRIMARY KEY,
            scenario_id TEXT COLLATE BINARY NOT NULL,
            target_key TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            event_count INTEGER NOT NULL
                CONSTRAINT cayu_eval_scenarios_event_count_check
                CHECK (event_count >= 1 AND event_count <= 1024),
            input_event_count INTEGER NOT NULL
                CONSTRAINT cayu_eval_scenarios_input_event_count_check
                CHECK (input_event_count >= 1 AND input_event_count <= 1024),
            approval_checkpoint_count INTEGER NOT NULL
                CONSTRAINT cayu_eval_scenarios_approval_checkpoint_count_check
                CHECK (approval_checkpoint_count >= 0
                    AND approval_checkpoint_count <= 1024),
            message_count INTEGER NOT NULL
                CONSTRAINT cayu_eval_scenarios_message_count_check
                CHECK (message_count >= input_event_count AND message_count <= 32768),
            part_count INTEGER NOT NULL
                CONSTRAINT cayu_eval_scenarios_part_count_check
                CHECK (part_count >= message_count AND part_count <= 1048576),
            artifact_requirement_count INTEGER NOT NULL
                CONSTRAINT cayu_eval_scenarios_artifact_requirement_count_check
                CHECK (artifact_requirement_count >= 0
                    AND artifact_requirement_count <= 128),
            secret_requirement_count INTEGER NOT NULL
                CONSTRAINT cayu_eval_scenarios_secret_requirement_count_check
                CHECK (secret_requirement_count >= 0
                    AND secret_requirement_count <= 128),
            document_json TEXT NOT NULL
                CONSTRAINT cayu_eval_scenarios_document_json_check
                CHECK (json_valid(document_json)),
            document_bytes INTEGER NOT NULL
                CONSTRAINT cayu_eval_scenarios_document_bytes_check
                CHECK (document_bytes >= 1 AND document_bytes <= 8388608)
                CONSTRAINT cayu_eval_scenarios_document_size_check
                CHECK (document_bytes = length(CAST(document_json AS BLOB))),
            created_at TEXT NOT NULL,
            CONSTRAINT cayu_eval_scenarios_event_partition_check
                CHECK (input_event_count + approval_checkpoint_count = event_count)
        );
        CREATE INDEX IF NOT EXISTS idx_cayu_eval_scenarios_catalog
            ON cayu_eval_scenarios(created_at DESC, revision ASC);
        CREATE INDEX IF NOT EXISTS idx_cayu_eval_scenarios_target_catalog
            ON cayu_eval_scenarios(target_key, created_at DESC, revision ASC);
        CREATE INDEX IF NOT EXISTS idx_cayu_eval_scenarios_id_catalog
            ON cayu_eval_scenarios(scenario_id, created_at DESC, revision ASC);
    """,
    55: """
        CREATE TABLE IF NOT EXISTS cayu_task_retry_reconciliation_rejections (
            task_id TEXT NOT NULL,
            reconciliation_idempotency_key TEXT NOT NULL,
            request_sha256 TEXT NOT NULL,
            record_json TEXT NOT NULL CHECK (json_valid(record_json)),
            recorded_at TEXT NOT NULL,
            PRIMARY KEY (task_id, reconciliation_idempotency_key)
        );
    """,
    56: "",
    57: "",
    58: """
        CREATE TABLE IF NOT EXISTS cayu_completion_verifier_profiles (
            proposal_id TEXT NOT NULL PRIMARY KEY
                REFERENCES cayu_completion_proposals(proposal_id) ON DELETE RESTRICT,
            task_id TEXT NOT NULL REFERENCES cayu_tasks(id) ON DELETE RESTRICT,
            attempt_id TEXT NOT NULL UNIQUE
                REFERENCES cayu_work_attempts(attempt_id) ON DELETE RESTRICT,
            profile_fingerprint TEXT NOT NULL CHECK (
                length(profile_fingerprint) = 64
                AND profile_fingerprint NOT GLOB '*[^0-9a-f]*'
            ),
            request_sha256 TEXT NOT NULL CHECK (
                length(request_sha256) = 64
                AND request_sha256 NOT GLOB '*[^0-9a-f]*'
            ),
            prepared_at TEXT NOT NULL,
            profile_json TEXT NOT NULL CHECK (json_valid(profile_json))
        );
        CREATE INDEX IF NOT EXISTS idx_cayu_completion_verifier_profiles_task
            ON cayu_completion_verifier_profiles(task_id, attempt_id);
    """,
    59: """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_cayu_sessions_instance_id
            ON cayu_sessions(instance_id);
    """,
    60: """
        DROP TABLE IF EXISTS cayu_knowledge_relation_publication_receipts;
        DROP TABLE IF EXISTS cayu_knowledge_relations;
        DROP TABLE IF EXISTS cayu_knowledge_change_acknowledgements;
        DROP TABLE IF EXISTS cayu_knowledge_change_consumers;
        DROP TABLE IF EXISTS cayu_knowledge_change_labels;
        DROP TABLE IF EXISTS cayu_knowledge_change_audiences;
        DROP TABLE IF EXISTS cayu_knowledge_changes;

        CREATE TABLE cayu_knowledge_relations (
            id TEXT PRIMARY KEY,
            subject_entry_id TEXT NOT NULL,
            subject_revision INTEGER NOT NULL
                CHECK (subject_revision > 0 AND subject_revision <= 2147483647),
            object_entry_id TEXT NOT NULL,
            object_revision INTEGER NOT NULL
                CHECK (object_revision > 0 AND object_revision <= 2147483647),
            kind TEXT NOT NULL CHECK (
                kind IN ('supersedes', 'derived_from', 'contradicts')
            ),
            created_by_type TEXT NOT NULL,
            created_by TEXT NOT NULL,
            policy_id TEXT,
            created_at TEXT NOT NULL,
            metadata_json TEXT NOT NULL CHECK (json_valid(metadata_json)),
            CHECK (subject_entry_id <> object_entry_id),
            CHECK (
                kind <> 'contradicts'
                OR subject_entry_id COLLATE BINARY < object_entry_id COLLATE BINARY
            ),
            UNIQUE (
                kind,
                subject_entry_id,
                subject_revision,
                object_entry_id,
                object_revision
            ),
            FOREIGN KEY (subject_entry_id, subject_revision)
                REFERENCES cayu_knowledge_revisions(entry_id, revision) ON DELETE CASCADE,
            FOREIGN KEY (object_entry_id, object_revision)
                REFERENCES cayu_knowledge_revisions(entry_id, revision) ON DELETE CASCADE
        );

        CREATE INDEX idx_cayu_knowledge_relations_subject
            ON cayu_knowledge_relations(
                subject_entry_id, subject_revision, created_at, id COLLATE BINARY
            );
        CREATE INDEX idx_cayu_knowledge_relations_object
            ON cayu_knowledge_relations(
                object_entry_id, object_revision, created_at, id COLLATE BINARY
            );
        CREATE INDEX idx_cayu_knowledge_relations_subject_kind
            ON cayu_knowledge_relations(
                subject_entry_id, subject_revision, kind, created_at, id COLLATE BINARY
            );
        CREATE INDEX idx_cayu_knowledge_relations_object_kind
            ON cayu_knowledge_relations(
                object_entry_id, object_revision, kind, created_at, id COLLATE BINARY
            );

        CREATE TABLE cayu_knowledge_relation_publication_receipts (
            operation_id TEXT PRIMARY KEY,
            relation_ids_json TEXT NOT NULL CHECK (json_valid(relation_ids_json)),
            request_sha256 TEXT NOT NULL CHECK (
                length(request_sha256) = 64
                AND request_sha256 NOT GLOB '*[^0-9a-f]*'
            ),
            committed_at TEXT NOT NULL,
            access_snapshots_json TEXT NOT NULL CHECK (json_valid(access_snapshots_json))
        );

        CREATE TABLE cayu_knowledge_changes (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT
                CHECK (sequence > 0 AND sequence <= 9223372036854775807),
            id TEXT NOT NULL UNIQUE,
            kind TEXT NOT NULL CHECK (
                kind IN (
                    'created',
                    'revision_appended',
                    'status_transitioned',
                    'tombstoned',
                    'hard_deleted',
                    'expired',
                    'relation_published'
                )
            ),
            entry_id TEXT NOT NULL,
            entry_revision INTEGER NOT NULL
                CHECK (entry_revision > 0 AND entry_revision <= 2147483647),
            committed_at TEXT NOT NULL,
            operation_id TEXT,
            relation_id TEXT,
            CHECK (
                (kind = 'relation_published' AND relation_id IS NOT NULL)
                OR (kind <> 'relation_published' AND relation_id IS NULL)
            )
        );

        CREATE TABLE cayu_knowledge_change_audiences (
            change_sequence INTEGER NOT NULL,
            audience_kind TEXT NOT NULL CHECK (
                audience_kind IN (
                    'before',
                    'after',
                    'subject_exact',
                    'subject_current',
                    'object_exact',
                    'object_current'
                )
            ),
            namespace TEXT NOT NULL,
            visibility TEXT NOT NULL,
            source_type TEXT,
            source_id TEXT,
            status TEXT NOT NULL,
            requires_include_expired INTEGER NOT NULL CHECK (
                requires_include_expired IN (0, 1)
            ),
            PRIMARY KEY (change_sequence, audience_kind),
            FOREIGN KEY (change_sequence)
                REFERENCES cayu_knowledge_changes(sequence) ON DELETE CASCADE
        );

        CREATE INDEX idx_cayu_knowledge_changes_entry_revision
            ON cayu_knowledge_changes(entry_id, entry_revision, sequence);
        CREATE INDEX idx_cayu_knowledge_changes_operation
            ON cayu_knowledge_changes(operation_id, sequence)
            WHERE operation_id IS NOT NULL;
        CREATE UNIQUE INDEX idx_cayu_knowledge_changes_relation
            ON cayu_knowledge_changes(relation_id)
            WHERE relation_id IS NOT NULL;
        CREATE INDEX idx_cayu_knowledge_change_audiences_namespace
            ON cayu_knowledge_change_audiences(
                namespace, change_sequence, audience_kind
            );
        CREATE INDEX idx_cayu_knowledge_change_audiences_status
            ON cayu_knowledge_change_audiences(status, change_sequence, audience_kind);
        CREATE INDEX idx_cayu_knowledge_change_audiences_source
            ON cayu_knowledge_change_audiences(
                source_type, source_id, change_sequence, audience_kind
            );

        CREATE TABLE cayu_knowledge_change_labels (
            change_sequence INTEGER NOT NULL,
            audience_kind TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            PRIMARY KEY (change_sequence, audience_kind, key),
            FOREIGN KEY (change_sequence, audience_kind)
                REFERENCES cayu_knowledge_change_audiences(
                    change_sequence, audience_kind
                ) ON DELETE CASCADE
        );

        CREATE INDEX idx_cayu_knowledge_change_labels_lookup
            ON cayu_knowledge_change_labels(
                key, value, change_sequence, audience_kind
            );

        CREATE TABLE cayu_knowledge_change_consumers (
            consumer_id TEXT PRIMARY KEY,
            access_scope_sha256 TEXT NOT NULL,
            cursor_sequence INTEGER NOT NULL DEFAULT 0 CHECK (cursor_sequence >= 0),
            pending_change_sequence INTEGER,
            pending_claim_id TEXT,
            pending_worker_id TEXT,
            pending_attempt INTEGER NOT NULL DEFAULT 0 CHECK (pending_attempt >= 0),
            claimed_at TEXT,
            lease_expires_at TEXT,
            last_acknowledged_claim_id TEXT,
            updated_at TEXT NOT NULL,
            CHECK (
                (pending_change_sequence IS NULL
                    AND pending_claim_id IS NULL
                    AND pending_worker_id IS NULL
                    AND claimed_at IS NULL
                    AND lease_expires_at IS NULL)
                OR
                (pending_change_sequence IS NOT NULL
                    AND pending_change_sequence > cursor_sequence
                    AND pending_claim_id IS NOT NULL
                    AND pending_worker_id IS NOT NULL
                    AND pending_attempt > 0
                    AND claimed_at IS NOT NULL
                    AND lease_expires_at IS NOT NULL
                    AND lease_expires_at > claimed_at)
            ),
            FOREIGN KEY (pending_change_sequence)
                REFERENCES cayu_knowledge_changes(sequence)
        );
        CREATE INDEX idx_cayu_knowledge_change_consumers_lease
            ON cayu_knowledge_change_consumers(lease_expires_at)
            WHERE pending_change_sequence IS NOT NULL;

        CREATE TABLE cayu_knowledge_change_acknowledgements (
            consumer_id TEXT NOT NULL,
            claim_id TEXT NOT NULL,
            claim_sha256 TEXT NOT NULL CHECK (
                length(claim_sha256) = 64
                AND claim_sha256 NOT GLOB '*[^0-9a-f]*'
            ),
            change_sequence INTEGER NOT NULL,
            acknowledged_at TEXT NOT NULL,
            PRIMARY KEY (consumer_id, claim_id),
            FOREIGN KEY (consumer_id)
                REFERENCES cayu_knowledge_change_consumers(consumer_id) ON DELETE CASCADE,
            FOREIGN KEY (change_sequence)
                REFERENCES cayu_knowledge_changes(sequence)
        );
    """,
    61: """
        CREATE TABLE IF NOT EXISTS cayu_work_attempt_admissions (
            admission_id TEXT PRIMARY KEY,
            attempt_id TEXT NOT NULL UNIQUE,
            task_id TEXT NOT NULL REFERENCES cayu_tasks(id) ON DELETE RESTRICT,
            session_id TEXT NOT NULL,
            interaction_id TEXT NOT NULL,
            state TEXT NOT NULL CHECK (state IN (
                'preparing', 'active', 'recovering', 'released'
            )),
            prepare_request_sha256 TEXT NOT NULL CHECK (
                length(prepare_request_sha256) = 64
                AND prepare_request_sha256 NOT GLOB '*[^0-9a-f]*'
            ),
            current_claim_id TEXT NOT NULL,
            current_generation INTEGER NOT NULL CHECK (
                current_generation >= 1 AND current_generation <= 64
            ),
            lease_expires_at TEXT NOT NULL,
            admission_json TEXT NOT NULL CHECK (json_valid(admission_json))
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_cayu_work_attempt_admission_interaction
            ON cayu_work_attempt_admissions(session_id, interaction_id);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_cayu_work_attempt_admission_session_current
            ON cayu_work_attempt_admissions(session_id)
            WHERE state != 'released';
        CREATE INDEX IF NOT EXISTS idx_cayu_work_attempt_admission_task
            ON cayu_work_attempt_admissions(task_id, current_generation DESC);

        CREATE TABLE IF NOT EXISTS cayu_work_attempt_execution_claims (
            claim_id TEXT PRIMARY KEY,
            admission_id TEXT NOT NULL
                REFERENCES cayu_work_attempt_admissions(admission_id) ON DELETE RESTRICT,
            generation INTEGER NOT NULL CHECK (generation >= 1 AND generation <= 64),
            request_sha256 TEXT NOT NULL CHECK (
                length(request_sha256) = 64
                AND request_sha256 NOT GLOB '*[^0-9a-f]*'
            ),
            lease_expires_at TEXT NOT NULL,
            is_current INTEGER NOT NULL CHECK (is_current IN (0, 1)),
            claim_json TEXT NOT NULL CHECK (json_valid(claim_json)),
            UNIQUE (admission_id, generation)
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_cayu_work_attempt_claim_current
            ON cayu_work_attempt_execution_claims(admission_id)
            WHERE is_current = 1;
    """,
    63: """
        DROP TABLE IF EXISTS cayu_knowledge_maintenance_decisions;
        CREATE TABLE cayu_knowledge_maintenance_decisions (
            operation_id TEXT PRIMARY KEY,
            proposal_id TEXT NOT NULL UNIQUE,
            proposal_fingerprint TEXT NOT NULL CHECK (
                length(proposal_fingerprint) = 64
                AND proposal_fingerprint NOT GLOB '*[^0-9a-f]*'
            ),
            request_sha256 TEXT NOT NULL CHECK (
                length(request_sha256) = 64
                AND request_sha256 NOT GLOB '*[^0-9a-f]*'
            ),
            committed_at TEXT NOT NULL,
            proposal_json TEXT NOT NULL CHECK (
                json_valid(proposal_json) AND json_type(proposal_json) = 'object'
            ),
            decision_json TEXT NOT NULL CHECK (
                json_valid(decision_json) AND json_type(decision_json) = 'object'
            ),
            receipt_json TEXT NOT NULL CHECK (
                json_valid(receipt_json) AND json_type(receipt_json) = 'object'
            ),
            access_snapshot_json TEXT NOT NULL CHECK (
                json_valid(access_snapshot_json)
                AND json_type(access_snapshot_json) = 'object'
            )
        );
    """,
    64: """
        CREATE TABLE IF NOT EXISTS cayu_eval_authored_suites (
            revision TEXT COLLATE BINARY PRIMARY KEY,
            suite_id TEXT COLLATE BINARY NOT NULL,
            suite_revision TEXT NOT NULL,
            target_key TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            case_count INTEGER NOT NULL
                CONSTRAINT cayu_eval_authored_suites_case_count_check
                CHECK (case_count >= 1 AND case_count <= 1000),
            assertion_count INTEGER NOT NULL
                CONSTRAINT cayu_eval_authored_suites_assertion_count_check
                CHECK (assertion_count >= case_count AND assertion_count <= 64000),
            simple_input_count INTEGER NOT NULL
                CONSTRAINT cayu_eval_authored_suites_simple_input_count_check
                CHECK (simple_input_count >= 0 AND simple_input_count <= case_count),
            scenario_count INTEGER NOT NULL
                CONSTRAINT cayu_eval_authored_suites_scenario_count_check
                CHECK (scenario_count >= 0 AND scenario_count <= case_count),
            trials INTEGER NOT NULL
                CONSTRAINT cayu_eval_authored_suites_trials_check
                CHECK (trials >= 1 AND trials <= 100),
            timeout_seconds INTEGER NOT NULL
                CONSTRAINT cayu_eval_authored_suites_timeout_check
                CHECK (timeout_seconds >= 1 AND timeout_seconds <= 3600),
            document_json TEXT NOT NULL
                CONSTRAINT cayu_eval_authored_suites_document_json_check
                CHECK (json_valid(document_json)),
            document_bytes INTEGER NOT NULL
                CONSTRAINT cayu_eval_authored_suites_document_bytes_check
                CHECK (document_bytes >= 1 AND document_bytes <= 8388608)
                CONSTRAINT cayu_eval_authored_suites_document_size_check
                CHECK (document_bytes = length(CAST(document_json AS BLOB))),
            created_at TEXT NOT NULL,
            CONSTRAINT cayu_eval_authored_suites_stimulus_partition_check
                CHECK (simple_input_count + scenario_count = case_count),
            CONSTRAINT cayu_eval_authored_suites_expansion_check
                CHECK (assertion_count * trials <= 10000)
        );
        CREATE INDEX IF NOT EXISTS idx_cayu_eval_authored_suites_catalog
            ON cayu_eval_authored_suites(created_at DESC, revision ASC);
        CREATE INDEX IF NOT EXISTS idx_cayu_eval_authored_suites_target_catalog
            ON cayu_eval_authored_suites(target_key, created_at DESC, revision ASC);
        CREATE INDEX IF NOT EXISTS idx_cayu_eval_authored_suites_id_catalog
            ON cayu_eval_authored_suites(suite_id, created_at DESC, revision ASC);
    """,
    68: """
        CREATE TABLE IF NOT EXISTS cayu_eval_judge_calibrations (
            revision TEXT COLLATE BINARY PRIMARY KEY,
            run_id TEXT COLLATE BINARY NOT NULL UNIQUE,
            definition_revision TEXT NOT NULL,
            target_key TEXT NOT NULL,
            trial_count INTEGER NOT NULL
                CONSTRAINT cayu_eval_judge_calibrations_trial_count_check
                CHECK (trial_count >= 1 AND trial_count <= 10),
            report_json TEXT NOT NULL
                CONSTRAINT cayu_eval_judge_calibrations_report_json_check
                CHECK (json_valid(report_json) AND json_type(report_json) = 'object'),
            document_bytes INTEGER NOT NULL
                CONSTRAINT cayu_eval_judge_calibrations_document_bytes_check
                CHECK (document_bytes >= 1 AND document_bytes <= 2097152)
                CONSTRAINT cayu_eval_judge_calibrations_document_size_check
                CHECK (document_bytes = length(CAST(report_json AS BLOB))),
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_cayu_eval_judge_calibrations_target
            ON cayu_eval_judge_calibrations(target_key, created_at DESC, revision ASC);
        CREATE INDEX IF NOT EXISTS idx_cayu_eval_judge_calibrations_definition
            ON cayu_eval_judge_calibrations(
                definition_revision, created_at DESC, revision ASC
            );
    """,
    65: """
        DROP VIEW IF EXISTS cayu_knowledge_current_entries;
        CREATE VIEW cayu_knowledge_current_entries AS
        SELECT
            logical.id AS id,
            revision.revision AS revision,
            logical.namespace AS namespace,
            revision.text AS text,
            revision.kind AS kind,
            revision.visibility AS visibility,
            revision.status AS status,
            revision.created_by_type AS created_by_type,
            revision.created_by AS created_by,
            revision.created_at AS created_at,
            revision.updated_at AS updated_at,
            revision.source_type AS source_type,
            revision.source_uri AS source_uri,
            revision.source_id AS source_id,
            revision.source_hash AS source_hash,
            revision.importance AS importance,
            revision.importance_source AS importance_source,
            revision.confidence AS confidence,
            revision.last_used_at AS last_used_at,
            revision.expires_at AS expires_at,
            revision.title AS title,
            revision.metadata_json AS metadata_json,
            revision.payload_bytes AS payload_bytes
        FROM cayu_knowledge_entries AS logical
        JOIN cayu_knowledge_revisions AS revision
          ON revision.entry_id = logical.id
         AND revision.revision = logical.current_revision;
    """,
    66: """
        CREATE TABLE IF NOT EXISTS cayu_local_execution_attempts (
            attempt_id TEXT COLLATE BINARY NOT NULL PRIMARY KEY,
            task_id TEXT NOT NULL REFERENCES cayu_tasks(id) ON DELETE RESTRICT,
            retry_series_id TEXT COLLATE BINARY,
            effect_lineage_id TEXT COLLATE BINARY NOT NULL,
            request_sha256 TEXT COLLATE BINARY NOT NULL,
            phase TEXT NOT NULL CHECK (
                phase IN ('prepared', 'starting', 'running', 'terminal')
            ),
            quiescence TEXT NOT NULL CHECK (
                quiescence IN (
                    'not_dispatched', 'terminal_not_quiescent', 'quiescent',
                    'unavailable', 'persistent_detached'
                )
            ),
            retry_admissible INTEGER NOT NULL CHECK (retry_admissible IN (0, 1)),
            recovery_generation INTEGER NOT NULL CHECK (recovery_generation >= 0),
            recovery_owner_id TEXT,
            recovery_owner_expires_at TEXT,
            record_json TEXT NOT NULL CHECK (
                json_valid(record_json) AND json_type(record_json) = 'object'
            ),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE (task_id, effect_lineage_id, attempt_id)
        );
        CREATE INDEX IF NOT EXISTS idx_cayu_local_execution_attempts_task_fence
            ON cayu_local_execution_attempts(task_id, retry_admissible, created_at, attempt_id);
        CREATE INDEX IF NOT EXISTS idx_cayu_local_execution_attempts_lineage
            ON cayu_local_execution_attempts(
                retry_series_id, task_id, effect_lineage_id,
                created_at DESC, attempt_id DESC
            );
        CREATE INDEX IF NOT EXISTS idx_cayu_local_execution_attempts_recovery
            ON cayu_local_execution_attempts(
                retry_admissible, phase, updated_at, attempt_id
            );
        CREATE INDEX IF NOT EXISTS idx_cayu_local_execution_attempts_discovery
            ON cayu_local_execution_attempts(created_at, attempt_id);
    """,
    67: """
        CREATE TABLE IF NOT EXISTS cayu_knowledge_maintenance_proposals (
            operation_id TEXT PRIMARY KEY,
            proposal_id TEXT NOT NULL UNIQUE,
            replacement_entry_id TEXT NOT NULL UNIQUE,
            replacement_revision INTEGER NOT NULL CHECK (
                replacement_revision > 0 AND replacement_revision <= 2147483647
            ),
            proposal_fingerprint TEXT NOT NULL CHECK (
                length(proposal_fingerprint) = 64
                AND proposal_fingerprint NOT GLOB '*[^0-9a-f]*'
            ),
            accepted_plan_fingerprint TEXT NOT NULL CHECK (
                length(accepted_plan_fingerprint) = 64
                AND accepted_plan_fingerprint NOT GLOB '*[^0-9a-f]*'
            ),
            request_sha256 TEXT NOT NULL CHECK (
                length(request_sha256) = 64
                AND request_sha256 NOT GLOB '*[^0-9a-f]*'
            ),
            committed_at TEXT NOT NULL,
            proposal_json TEXT NOT NULL CHECK (
                json_valid(proposal_json) AND json_type(proposal_json) = 'object'
            ),
            accepted_plan_json TEXT NOT NULL CHECK (
                json_valid(accepted_plan_json)
                AND json_type(accepted_plan_json) = 'object'
            ),
            receipt_json TEXT NOT NULL CHECK (
                json_valid(receipt_json) AND json_type(receipt_json) = 'object'
            ),
            access_snapshot_json TEXT NOT NULL CHECK (
                json_valid(access_snapshot_json)
                AND json_type(access_snapshot_json) = 'object'
            )
        );
    """,
    69: """
        CREATE TABLE IF NOT EXISTS cayu_agent_work_context_revisions (
            task_id TEXT COLLATE BINARY NOT NULL,
            revision INTEGER NOT NULL CHECK (
                revision > 0 AND revision <= 2147483647
            ),
            content_sha256 TEXT COLLATE BINARY NOT NULL CHECK (
                length(content_sha256) = 64
                AND content_sha256 NOT GLOB '*[^0-9a-f]*'
            ),
            operation_id TEXT COLLATE BINARY NOT NULL UNIQUE,
            record_json TEXT NOT NULL CHECK (
                json_valid(record_json) AND json_type(record_json) = 'object'
            ),
            published_at TEXT NOT NULL,
            PRIMARY KEY (task_id, revision)
        );
        CREATE TABLE IF NOT EXISTS cayu_agent_work_context_heads (
            task_id TEXT COLLATE BINARY NOT NULL PRIMARY KEY,
            current_revision INTEGER NOT NULL CHECK (
                current_revision > 0 AND current_revision <= 2147483647
            ),
            FOREIGN KEY (task_id, current_revision)
                REFERENCES cayu_agent_work_context_revisions(task_id, revision)
                ON DELETE RESTRICT
        );
        CREATE TABLE IF NOT EXISTS cayu_agent_work_context_publications (
            operation_id TEXT COLLATE BINARY NOT NULL PRIMARY KEY,
            task_id TEXT COLLATE BINARY NOT NULL,
            request_sha256 TEXT COLLATE BINARY NOT NULL CHECK (
                length(request_sha256) = 64
                AND request_sha256 NOT GLOB '*[^0-9a-f]*'
            ),
            context_revision INTEGER NOT NULL CHECK (
                context_revision > 0 AND context_revision <= 2147483647
            ),
            changed INTEGER NOT NULL CHECK (changed IN (0, 1)),
            receipt_json TEXT NOT NULL CHECK (
                json_valid(receipt_json) AND json_type(receipt_json) = 'object'
            ),
            committed_at TEXT NOT NULL,
            FOREIGN KEY (task_id, context_revision)
                REFERENCES cayu_agent_work_context_revisions(task_id, revision)
                ON DELETE RESTRICT
        );
        CREATE TABLE IF NOT EXISTS cayu_agent_recall_checkpoints (
            agent_id TEXT COLLATE BINARY NOT NULL,
            task_id TEXT COLLATE BINARY NOT NULL,
            knowledge_namespace TEXT COLLATE BINARY NOT NULL,
            access_policy_sha256 TEXT COLLATE BINARY NOT NULL CHECK (
                length(access_policy_sha256) = 64
                AND access_policy_sha256 NOT GLOB '*[^0-9a-f]*'
            ),
            checkpoint_stream_id TEXT COLLATE BINARY NOT NULL,
            revision INTEGER NOT NULL CHECK (
                revision > 0 AND revision <= 2147483647
            ),
            work_context_revision INTEGER NOT NULL CHECK (
                work_context_revision > 0 AND work_context_revision <= 2147483647
            ),
            work_context_sha256 TEXT COLLATE BINARY NOT NULL CHECK (
                length(work_context_sha256) = 64
                AND work_context_sha256 NOT GLOB '*[^0-9a-f]*'
            ),
            knowledge_sequence INTEGER NOT NULL CHECK (
                knowledge_sequence >= 0
                AND knowledge_sequence <= 9223372036854775807
            ),
            index_readiness_sequence INTEGER NOT NULL CHECK (
                index_readiness_sequence >= 0
                AND index_readiness_sequence <= 9223372036854775807
            ),
            knowledge_high_water_sequence INTEGER NOT NULL CHECK (
                knowledge_high_water_sequence >= 0
                AND knowledge_high_water_sequence <= 9223372036854775807
            ),
            index_readiness_high_water_sequence INTEGER NOT NULL CHECK (
                index_readiness_high_water_sequence >= 0
                AND index_readiness_high_water_sequence <= 9223372036854775807
            ),
            processing_mode TEXT COLLATE BINARY NOT NULL CHECK (
                processing_mode IN ('full_index', 'delta')
            ),
            processing_id TEXT COLLATE BINARY NOT NULL,
            operation_id TEXT COLLATE BINARY NOT NULL UNIQUE,
            record_json TEXT NOT NULL CHECK (
                json_valid(record_json) AND json_type(record_json) = 'object'
            ),
            updated_at TEXT NOT NULL,
            CHECK (knowledge_sequence <= knowledge_high_water_sequence),
            CHECK (
                index_readiness_sequence <= index_readiness_high_water_sequence
            ),
            PRIMARY KEY (
                agent_id, task_id, knowledge_namespace,
                access_policy_sha256, checkpoint_stream_id, revision
            ),
            FOREIGN KEY (task_id, work_context_revision)
                REFERENCES cayu_agent_work_context_revisions(task_id, revision)
                ON DELETE RESTRICT
        );
        CREATE TABLE IF NOT EXISTS cayu_agent_recall_checkpoint_heads (
            agent_id TEXT COLLATE BINARY NOT NULL,
            task_id TEXT COLLATE BINARY NOT NULL,
            knowledge_namespace TEXT COLLATE BINARY NOT NULL,
            access_policy_sha256 TEXT COLLATE BINARY NOT NULL,
            checkpoint_stream_id TEXT COLLATE BINARY NOT NULL,
            current_revision INTEGER NOT NULL CHECK (
                current_revision > 0 AND current_revision <= 2147483647
            ),
            PRIMARY KEY (
                agent_id, task_id, knowledge_namespace,
                access_policy_sha256, checkpoint_stream_id
            ),
            FOREIGN KEY (
                agent_id, task_id, knowledge_namespace,
                access_policy_sha256, checkpoint_stream_id, current_revision
            ) REFERENCES cayu_agent_recall_checkpoints(
                agent_id, task_id, knowledge_namespace,
                access_policy_sha256, checkpoint_stream_id, revision
            ) ON DELETE RESTRICT
        );
    """,
    70: """
        CREATE TABLE IF NOT EXISTS cayu_task_interrupted_handoff_receipts (
            task_id TEXT NOT NULL,
            handoff_id TEXT NOT NULL,
            request_sha256 TEXT NOT NULL,
            request_json TEXT NOT NULL CHECK (json_valid(request_json)),
            task_json TEXT NOT NULL CHECK (json_valid(task_json)),
            committed_at TEXT NOT NULL,
            PRIMARY KEY (task_id, handoff_id)
        );

        CREATE INDEX IF NOT EXISTS idx_cayu_tasks_interrupted_handoff_recovery
            ON cayu_tasks(status, lease_expires_at, id);
    """,
    71: """
        CREATE TABLE IF NOT EXISTS cayu_agent_recall_deliveries (
            delivery_id TEXT COLLATE BINARY NOT NULL PRIMARY KEY,
            operation_id TEXT COLLATE BINARY NOT NULL UNIQUE,
            agent_id TEXT COLLATE BINARY NOT NULL,
            task_id TEXT COLLATE BINARY NOT NULL,
            knowledge_namespace TEXT COLLATE BINARY NOT NULL,
            access_policy_sha256 TEXT COLLATE BINARY NOT NULL CHECK (
                length(access_policy_sha256) = 64
                AND access_policy_sha256 NOT GLOB '*[^0-9a-f]*'
            ),
            checkpoint_stream_id TEXT COLLATE BINARY NOT NULL,
            checkpoint_revision INTEGER NOT NULL CHECK (
                checkpoint_revision > 0 AND checkpoint_revision <= 2147483647
            ),
            processing_result_sha256 TEXT COLLATE BINARY NOT NULL CHECK (
                length(processing_result_sha256) = 64
                AND processing_result_sha256 NOT GLOB '*[^0-9a-f]*'
            ),
            delivery_json TEXT NOT NULL CHECK (
                json_valid(delivery_json) AND json_type(delivery_json) = 'object'
            ),
            staged_at TEXT NOT NULL,
            UNIQUE (
                agent_id, task_id, knowledge_namespace,
                access_policy_sha256, checkpoint_stream_id, checkpoint_revision
            ),
            FOREIGN KEY (
                agent_id, task_id, knowledge_namespace,
                access_policy_sha256, checkpoint_stream_id, checkpoint_revision
            ) REFERENCES cayu_agent_recall_checkpoints(
                agent_id, task_id, knowledge_namespace,
                access_policy_sha256, checkpoint_stream_id, revision
            ) ON DELETE RESTRICT,
            FOREIGN KEY (operation_id)
                REFERENCES cayu_agent_recall_checkpoints(operation_id)
                ON DELETE RESTRICT
        );
        CREATE TABLE IF NOT EXISTS cayu_agent_recall_delivery_states (
            delivery_id TEXT COLLATE BINARY NOT NULL PRIMARY KEY,
            agent_id TEXT COLLATE BINARY NOT NULL,
            task_id TEXT COLLATE BINARY NOT NULL,
            knowledge_namespace TEXT COLLATE BINARY NOT NULL,
            access_policy_sha256 TEXT COLLATE BINARY NOT NULL,
            checkpoint_stream_id TEXT COLLATE BINARY NOT NULL,
            checkpoint_revision INTEGER NOT NULL CHECK (
                checkpoint_revision > 0 AND checkpoint_revision <= 2147483647
            ),
            state TEXT COLLATE BINARY NOT NULL CHECK (
                state IN ('pending', 'claimed', 'acknowledged')
            ),
            attempt INTEGER NOT NULL CHECK (
                attempt >= 0 AND attempt <= 9223372036854775807
            ),
            state_revision INTEGER NOT NULL CHECK (
                state_revision >= 0 AND state_revision <= 9223372036854775807
            ),
            lease_expires_at TEXT,
            release_id TEXT COLLATE BINARY UNIQUE,
            acknowledgement_id TEXT COLLATE BINARY UNIQUE,
            state_json TEXT NOT NULL CHECK (
                json_valid(state_json) AND json_type(state_json) = 'object'
            ),
            updated_at TEXT NOT NULL,
            CHECK (
                (state = 'pending' AND lease_expires_at IS NULL
                    AND acknowledgement_id IS NULL
                    AND (
                        (attempt = 0 AND state_revision = 0 AND release_id IS NULL)
                        OR (attempt > 0 AND state_revision > 0
                            AND release_id IS NOT NULL)
                    ))
                OR (state = 'claimed' AND lease_expires_at IS NOT NULL
                    AND attempt > 0 AND state_revision > 0
                    AND release_id IS NULL AND acknowledgement_id IS NULL)
                OR (state = 'acknowledged' AND lease_expires_at IS NULL
                    AND attempt > 0 AND state_revision > 0
                    AND release_id IS NULL AND acknowledgement_id IS NOT NULL)
            ),
            FOREIGN KEY (delivery_id)
                REFERENCES cayu_agent_recall_deliveries(delivery_id)
                ON DELETE RESTRICT,
            FOREIGN KEY (release_id)
                REFERENCES cayu_agent_recall_delivery_releases(release_id)
                ON DELETE RESTRICT,
            FOREIGN KEY (
                agent_id, task_id, knowledge_namespace,
                access_policy_sha256, checkpoint_stream_id, checkpoint_revision
            ) REFERENCES cayu_agent_recall_deliveries(
                agent_id, task_id, knowledge_namespace,
                access_policy_sha256, checkpoint_stream_id, checkpoint_revision
            ) ON DELETE RESTRICT
        );
        CREATE TABLE IF NOT EXISTS cayu_agent_recall_delivery_claims (
            claim_id TEXT COLLATE BINARY NOT NULL PRIMARY KEY,
            delivery_id TEXT COLLATE BINARY NOT NULL,
            worker_id TEXT COLLATE BINARY NOT NULL,
            request_sha256 TEXT COLLATE BINARY NOT NULL CHECK (
                length(request_sha256) = 64
                AND request_sha256 NOT GLOB '*[^0-9a-f]*'
            ),
            attempt INTEGER NOT NULL CHECK (
                attempt > 0 AND attempt <= 9223372036854775807
            ),
            claimed_at TEXT NOT NULL,
            UNIQUE (delivery_id, attempt),
            FOREIGN KEY (delivery_id)
                REFERENCES cayu_agent_recall_deliveries(delivery_id)
                ON DELETE RESTRICT
        );
        CREATE TABLE IF NOT EXISTS cayu_agent_recall_delivery_releases (
            release_id TEXT COLLATE BINARY NOT NULL PRIMARY KEY,
            delivery_id TEXT COLLATE BINARY NOT NULL,
            claim_id TEXT COLLATE BINARY NOT NULL,
            request_sha256 TEXT COLLATE BINARY NOT NULL CHECK (
                length(request_sha256) = 64
                AND request_sha256 NOT GLOB '*[^0-9a-f]*'
            ),
            release_json TEXT NOT NULL CHECK (
                json_valid(release_json) AND json_type(release_json) = 'object'
            ),
            released_at TEXT NOT NULL,
            FOREIGN KEY (delivery_id)
                REFERENCES cayu_agent_recall_deliveries(delivery_id)
                ON DELETE RESTRICT,
            FOREIGN KEY (claim_id)
                REFERENCES cayu_agent_recall_delivery_claims(claim_id)
                ON DELETE RESTRICT
        );
        CREATE INDEX IF NOT EXISTS idx_cayu_agent_recall_delivery_pending
            ON cayu_agent_recall_delivery_states(
                agent_id, task_id, knowledge_namespace,
                access_policy_sha256, checkpoint_stream_id,
                checkpoint_revision, delivery_id
            ) WHERE state != 'acknowledged';
    """,
    73: """
        CREATE TABLE IF NOT EXISTS cayu_agent_recall_subscription_revisions (
            subscription_id TEXT COLLATE BINARY NOT NULL,
            revision INTEGER NOT NULL CHECK (
                revision > 0 AND revision <= 2147483647
            ),
            operation_id TEXT COLLATE BINARY NOT NULL UNIQUE,
            agent_id TEXT COLLATE BINARY NOT NULL,
            task_id TEXT COLLATE BINARY NOT NULL,
            knowledge_namespace TEXT COLLATE BINARY NOT NULL,
            access_policy_sha256 TEXT COLLATE BINARY NOT NULL CHECK (
                length(access_policy_sha256) = 64
                AND access_policy_sha256 NOT GLOB '*[^0-9a-f]*'
            ),
            work_context_revision INTEGER NOT NULL CHECK (
                work_context_revision > 0 AND work_context_revision <= 2147483647
            ),
            work_context_sha256 TEXT COLLATE BINARY NOT NULL CHECK (
                length(work_context_sha256) = 64
                AND work_context_sha256 NOT GLOB '*[^0-9a-f]*'
            ),
            status TEXT COLLATE BINARY NOT NULL CHECK (
                status IN ('active', 'paused', 'cancelled')
            ),
            priority INTEGER NOT NULL CHECK (
                priority >= 0 AND priority <= 1000
            ),
            subscription_json TEXT NOT NULL CHECK (
                json_valid(subscription_json)
                AND json_type(subscription_json) = 'object'
            ),
            expires_at TEXT NOT NULL,
            published_at TEXT NOT NULL,
            PRIMARY KEY (subscription_id, revision),
            FOREIGN KEY (task_id, work_context_revision)
                REFERENCES cayu_agent_work_context_revisions(task_id, revision)
                ON DELETE RESTRICT
        );
        CREATE TABLE IF NOT EXISTS cayu_agent_recall_subscription_heads (
            subscription_id TEXT COLLATE BINARY NOT NULL PRIMARY KEY,
            current_revision INTEGER NOT NULL CHECK (
                current_revision > 0 AND current_revision <= 2147483647
            ),
            FOREIGN KEY (subscription_id, current_revision)
                REFERENCES cayu_agent_recall_subscription_revisions(
                    subscription_id, revision
                ) ON DELETE RESTRICT
        );
        CREATE TABLE IF NOT EXISTS cayu_agent_recall_subscription_publications (
            operation_id TEXT COLLATE BINARY NOT NULL PRIMARY KEY,
            subscription_id TEXT COLLATE BINARY NOT NULL,
            subscription_revision INTEGER NOT NULL CHECK (
                subscription_revision > 0 AND subscription_revision <= 2147483647
            ),
            request_sha256 TEXT COLLATE BINARY NOT NULL CHECK (
                length(request_sha256) = 64
                AND request_sha256 NOT GLOB '*[^0-9a-f]*'
            ),
            receipt_json TEXT NOT NULL CHECK (
                json_valid(receipt_json) AND json_type(receipt_json) = 'object'
            ),
            committed_at TEXT NOT NULL,
            FOREIGN KEY (subscription_id, subscription_revision)
                REFERENCES cayu_agent_recall_subscription_revisions(
                    subscription_id, revision
                ) ON DELETE RESTRICT
        );
        CREATE TABLE IF NOT EXISTS cayu_agent_recall_subscription_states (
            subscription_id TEXT COLLATE BINARY NOT NULL PRIMARY KEY,
            current_revision INTEGER NOT NULL CHECK (
                current_revision > 0 AND current_revision <= 2147483647
            ),
            agent_id TEXT COLLATE BINARY NOT NULL,
            task_id TEXT COLLATE BINARY NOT NULL,
            knowledge_namespace TEXT COLLATE BINARY NOT NULL,
            access_policy_sha256 TEXT COLLATE BINARY NOT NULL,
            run_state TEXT COLLATE BINARY NOT NULL CHECK (
                run_state IN ('due', 'claimed')
            ),
            attempt INTEGER NOT NULL CHECK (
                attempt >= 0 AND attempt <= 9223372036854775807
            ),
            state_revision INTEGER NOT NULL CHECK (
                state_revision >= 0 AND state_revision <= 9223372036854775807
            ),
            lease_expires_at TEXT,
            release_id TEXT COLLATE BINARY UNIQUE,
            next_evaluation_at TEXT NOT NULL,
            last_evaluation_id TEXT COLLATE BINARY,
            state_json TEXT NOT NULL CHECK (
                json_valid(state_json) AND json_type(state_json) = 'object'
            ),
            updated_at TEXT NOT NULL,
            CHECK (
                (run_state = 'due' AND lease_expires_at IS NULL)
                OR (run_state = 'claimed' AND lease_expires_at IS NOT NULL
                    AND release_id IS NULL AND attempt > 0 AND state_revision > 0)
            ),
            FOREIGN KEY (subscription_id, current_revision)
                REFERENCES cayu_agent_recall_subscription_revisions(
                    subscription_id, revision
                ) ON DELETE RESTRICT,
            FOREIGN KEY (release_id)
                REFERENCES cayu_agent_recall_subscription_releases(release_id)
                ON DELETE RESTRICT
        );
        CREATE TABLE IF NOT EXISTS cayu_agent_recall_subscription_claims (
            claim_id TEXT COLLATE BINARY NOT NULL PRIMARY KEY,
            subscription_id TEXT COLLATE BINARY NOT NULL,
            subscription_revision INTEGER NOT NULL CHECK (
                subscription_revision > 0 AND subscription_revision <= 2147483647
            ),
            runner_id TEXT COLLATE BINARY NOT NULL,
            request_sha256 TEXT COLLATE BINARY NOT NULL CHECK (
                length(request_sha256) = 64
                AND request_sha256 NOT GLOB '*[^0-9a-f]*'
            ),
            attempt INTEGER NOT NULL CHECK (
                attempt > 0 AND attempt <= 9223372036854775807
            ),
            claimed_at TEXT NOT NULL,
            UNIQUE (subscription_id, attempt),
            FOREIGN KEY (subscription_id, subscription_revision)
                REFERENCES cayu_agent_recall_subscription_revisions(
                    subscription_id, revision
                ) ON DELETE RESTRICT
        );
        CREATE TABLE IF NOT EXISTS cayu_agent_recall_subscription_releases (
            release_id TEXT COLLATE BINARY NOT NULL PRIMARY KEY,
            subscription_id TEXT COLLATE BINARY NOT NULL,
            claim_id TEXT COLLATE BINARY NOT NULL,
            request_sha256 TEXT COLLATE BINARY NOT NULL CHECK (
                length(request_sha256) = 64
                AND request_sha256 NOT GLOB '*[^0-9a-f]*'
            ),
            release_json TEXT NOT NULL CHECK (
                json_valid(release_json) AND json_type(release_json) = 'object'
            ),
            released_at TEXT NOT NULL,
            FOREIGN KEY (subscription_id)
                REFERENCES cayu_agent_recall_subscription_heads(subscription_id)
                ON DELETE RESTRICT,
            FOREIGN KEY (claim_id)
                REFERENCES cayu_agent_recall_subscription_claims(claim_id)
                ON DELETE RESTRICT
        );
        CREATE TABLE IF NOT EXISTS cayu_agent_recall_subscription_evaluations (
            evaluation_id TEXT COLLATE BINARY NOT NULL PRIMARY KEY,
            subscription_id TEXT COLLATE BINARY NOT NULL,
            subscription_revision INTEGER NOT NULL CHECK (
                subscription_revision > 0 AND subscription_revision <= 2147483647
            ),
            agent_id TEXT COLLATE BINARY NOT NULL,
            task_id TEXT COLLATE BINARY NOT NULL,
            knowledge_namespace TEXT COLLATE BINARY NOT NULL,
            access_policy_sha256 TEXT COLLATE BINARY NOT NULL,
            claim_id TEXT COLLATE BINARY NOT NULL UNIQUE,
            processing_operation_id TEXT COLLATE BINARY NOT NULL UNIQUE,
            request_sha256 TEXT COLLATE BINARY NOT NULL CHECK (
                length(request_sha256) = 64
                AND request_sha256 NOT GLOB '*[^0-9a-f]*'
            ),
            outcome TEXT COLLATE BINARY NOT NULL CHECK (
                outcome IN ('no_work', 'silent', 'wake')
            ),
            delivery_id TEXT COLLATE BINARY UNIQUE,
            evaluation_json TEXT NOT NULL CHECK (
                json_valid(evaluation_json) AND json_type(evaluation_json) = 'object'
            ),
            committed_at TEXT NOT NULL,
            CHECK (
                (outcome = 'wake' AND delivery_id IS NOT NULL)
                OR (outcome != 'wake' AND delivery_id IS NULL)
            ),
            FOREIGN KEY (subscription_id, subscription_revision)
                REFERENCES cayu_agent_recall_subscription_revisions(
                    subscription_id, revision
                ) ON DELETE RESTRICT,
            FOREIGN KEY (claim_id)
                REFERENCES cayu_agent_recall_subscription_claims(claim_id)
                ON DELETE RESTRICT,
            FOREIGN KEY (delivery_id)
                REFERENCES cayu_agent_recall_deliveries(delivery_id)
                ON DELETE RESTRICT
        );
        CREATE TABLE IF NOT EXISTS cayu_agent_recall_subscription_wake_claims (
            claim_id TEXT COLLATE BINARY NOT NULL PRIMARY KEY,
            wake_id TEXT COLLATE BINARY NOT NULL,
            delivery_id TEXT COLLATE BINARY NOT NULL,
            runner_id TEXT COLLATE BINARY NOT NULL,
            request_sha256 TEXT COLLATE BINARY NOT NULL CHECK (
                length(request_sha256) = 64
                AND request_sha256 NOT GLOB '*[^0-9a-f]*'
            ),
            attempt INTEGER NOT NULL CHECK (
                attempt > 0 AND attempt <= 9223372036854775807
            ),
            claimed_at TEXT NOT NULL,
            UNIQUE (wake_id, attempt),
            FOREIGN KEY (wake_id)
                REFERENCES cayu_agent_recall_subscription_evaluations(evaluation_id)
                ON DELETE RESTRICT,
            FOREIGN KEY (delivery_id)
                REFERENCES cayu_agent_recall_deliveries(delivery_id)
                ON DELETE RESTRICT
        );
        CREATE TABLE IF NOT EXISTS cayu_agent_recall_subscription_wake_releases (
            release_id TEXT COLLATE BINARY NOT NULL PRIMARY KEY,
            wake_id TEXT COLLATE BINARY NOT NULL,
            claim_id TEXT COLLATE BINARY NOT NULL,
            request_sha256 TEXT COLLATE BINARY NOT NULL CHECK (
                length(request_sha256) = 64
                AND request_sha256 NOT GLOB '*[^0-9a-f]*'
            ),
            release_json TEXT NOT NULL CHECK (
                json_valid(release_json) AND json_type(release_json) = 'object'
            ),
            released_at TEXT NOT NULL,
            FOREIGN KEY (wake_id)
                REFERENCES cayu_agent_recall_subscription_evaluations(evaluation_id)
                ON DELETE RESTRICT,
            FOREIGN KEY (claim_id)
                REFERENCES cayu_agent_recall_subscription_wake_claims(claim_id)
                ON DELETE RESTRICT
        );
        CREATE TABLE IF NOT EXISTS cayu_agent_recall_subscription_wake_states (
            wake_id TEXT COLLATE BINARY NOT NULL PRIMARY KEY,
            delivery_id TEXT COLLATE BINARY NOT NULL UNIQUE,
            agent_id TEXT COLLATE BINARY NOT NULL,
            task_id TEXT COLLATE BINARY NOT NULL,
            knowledge_namespace TEXT COLLATE BINARY NOT NULL,
            access_policy_sha256 TEXT COLLATE BINARY NOT NULL,
            state TEXT COLLATE BINARY NOT NULL CHECK (
                state IN ('pending', 'claimed', 'acknowledged')
            ),
            attempt INTEGER NOT NULL CHECK (
                attempt >= 0 AND attempt <= 9223372036854775807
            ),
            state_revision INTEGER NOT NULL CHECK (
                state_revision >= 0 AND state_revision <= 9223372036854775807
            ),
            claim_id TEXT COLLATE BINARY,
            lease_expires_at TEXT,
            release_id TEXT COLLATE BINARY UNIQUE,
            acknowledgement_id TEXT COLLATE BINARY UNIQUE,
            state_json TEXT NOT NULL CHECK (
                json_valid(state_json) AND json_type(state_json) = 'object'
            ),
            committed_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            CHECK (
                (state = 'pending' AND (
                    (attempt = 0 AND state_revision = 0 AND claim_id IS NULL
                     AND lease_expires_at IS NULL AND release_id IS NULL)
                    OR (attempt > 0 AND state_revision > 0 AND claim_id IS NOT NULL
                        AND lease_expires_at IS NULL AND release_id IS NOT NULL)
                ) AND acknowledgement_id IS NULL)
                OR (state = 'claimed' AND attempt > 0 AND state_revision > 0
                    AND claim_id IS NOT NULL AND lease_expires_at IS NOT NULL
                    AND release_id IS NULL AND acknowledgement_id IS NULL)
                OR (state = 'acknowledged' AND attempt > 0 AND state_revision > 0
                    AND claim_id IS NOT NULL AND lease_expires_at IS NULL
                    AND release_id IS NULL AND acknowledgement_id IS NOT NULL)
            ),
            FOREIGN KEY (wake_id)
                REFERENCES cayu_agent_recall_subscription_evaluations(evaluation_id)
                ON DELETE RESTRICT,
            FOREIGN KEY (delivery_id)
                REFERENCES cayu_agent_recall_deliveries(delivery_id)
                ON DELETE RESTRICT,
            FOREIGN KEY (claim_id)
                REFERENCES cayu_agent_recall_subscription_wake_claims(claim_id)
                ON DELETE RESTRICT,
            FOREIGN KEY (release_id)
                REFERENCES cayu_agent_recall_subscription_wake_releases(release_id)
                ON DELETE RESTRICT
        );
        CREATE INDEX IF NOT EXISTS idx_cayu_agent_recall_subscription_due
            ON cayu_agent_recall_subscription_states(
                agent_id, task_id, knowledge_namespace, access_policy_sha256,
                next_evaluation_at, subscription_id
            );
        CREATE INDEX IF NOT EXISTS idx_cayu_agent_recall_subscription_evaluations
            ON cayu_agent_recall_subscription_evaluations(
                subscription_id, evaluation_id
            );
        CREATE INDEX IF NOT EXISTS idx_cayu_agent_recall_subscription_wakes
            ON cayu_agent_recall_subscription_wake_states(
                agent_id, task_id, knowledge_namespace, access_policy_sha256,
                committed_at, wake_id
            ) WHERE state != 'acknowledged';
    """,
    72: """
        ALTER TABLE cayu_eval_runs RENAME TO cayu_eval_runs_revision_71;

        CREATE TABLE cayu_eval_runs (
            run_id TEXT COLLATE BINARY PRIMARY KEY,
            idempotency_key TEXT NOT NULL UNIQUE,
            corpus_revision TEXT NOT NULL
                REFERENCES cayu_eval_corpora(revision),
            target_key TEXT NOT NULL,
            suite_id TEXT COLLATE BINARY NOT NULL,
            suite_revision TEXT NOT NULL,
            max_concurrency INTEGER NOT NULL
                CHECK (max_concurrency >= 1 AND max_concurrency <= 2147483647),
            invocation_json TEXT NOT NULL,
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
            scenario_progress_json TEXT CHECK (
                scenario_progress_json IS NULL OR (
                    json_valid(scenario_progress_json)
                    AND length(CAST(scenario_progress_json AS BLOB)) BETWEEN 1 AND 262144
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

        INSERT INTO cayu_eval_runs (
            run_id, idempotency_key, corpus_revision, target_key, suite_id,
            suite_revision, max_concurrency, invocation_json, status, created_at,
            updated_at, started_at, finished_at, cancel_requested_at, claim_id,
            ownership_epoch, lease_expires_at, result_revision, result_status,
            result_score, result_duration_ms, failure_code, scenario_progress_json
        )
        SELECT
            run_id, idempotency_key, corpus_revision, target_key, suite_id,
            suite_revision, max_concurrency, invocation_json, status, created_at,
            updated_at, started_at, finished_at, cancel_requested_at, claim_id,
            ownership_epoch, lease_expires_at, result_revision, result_status,
            result_score, result_duration_ms, failure_code, scenario_progress_json
        FROM cayu_eval_runs_revision_71;

        DROP TABLE cayu_eval_runs_revision_71;

        CREATE INDEX idx_cayu_eval_runs_catalog
            ON cayu_eval_runs(created_at DESC, run_id ASC);
        CREATE INDEX idx_cayu_eval_runs_status_claim
            ON cayu_eval_runs(status, lease_expires_at, created_at ASC, run_id ASC);
        CREATE INDEX idx_cayu_eval_runs_corpus_catalog
            ON cayu_eval_runs(corpus_revision, created_at DESC, run_id ASC);
        CREATE INDEX idx_cayu_eval_runs_target_catalog
            ON cayu_eval_runs(target_key, created_at DESC, run_id ASC);
        CREATE INDEX idx_cayu_eval_runs_target_status_claim
            ON cayu_eval_runs(
                target_key, status, lease_expires_at, created_at ASC, run_id ASC
            );

    """,
    74: """
        CREATE TABLE IF NOT EXISTS cayu_eval_run_trial_checkpoints (
            run_id TEXT COLLATE BINARY NOT NULL
                REFERENCES cayu_eval_runs(run_id) ON DELETE CASCADE,
            case_id TEXT COLLATE BINARY NOT NULL,
            trial_number INTEGER NOT NULL CHECK (trial_number BETWEEN 1 AND 100),
            checkpoint_json TEXT NOT NULL CHECK (
                json_valid(checkpoint_json)
                AND json_type(checkpoint_json) = 'object'
                AND length(CAST(checkpoint_json AS BLOB)) BETWEEN 1 AND 41943040
            ),
            document_bytes INTEGER NOT NULL CHECK (
                document_bytes BETWEEN 1 AND 41943040
                AND document_bytes = length(CAST(checkpoint_json AS BLOB))
            ),
            PRIMARY KEY (run_id, case_id, trial_number)
        );
        CREATE INDEX IF NOT EXISTS idx_cayu_eval_runs_authored_suite_launch_claim
            ON cayu_eval_runs(
                authored_suite_launch_revision, authored_suite_launch_lane,
                created_at ASC, run_id ASC, status
            )
            WHERE authored_suite_launch_revision IS NOT NULL;
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
    45: (("cayu_tasks", "retry_series_json", "TEXT"),),
    49: (
        (
            "cayu_tasks",
            "work_contract_json",
            "TEXT CHECK (work_contract_json IS NULL OR json_valid(work_contract_json))",
        ),
    ),
    46: (("cayu_transcript_messages", "transcript_search_document", "TEXT NOT NULL"),),
    58: (
        (
            "cayu_completion_verification_claims",
            "verifier_profile_fingerprint",
            "TEXT CHECK (verifier_profile_fingerprint IS NOT NULL AND "
            "(length(verifier_profile_fingerprint) = 64 AND "
            "verifier_profile_fingerprint NOT GLOB '*[^0-9a-f]*'))",
        ),
        (
            "cayu_completion_decisions",
            "verifier_profile_fingerprint",
            "TEXT CHECK (verifier_profile_fingerprint IS NOT NULL AND "
            "(length(verifier_profile_fingerprint) = 64 AND "
            "verifier_profile_fingerprint NOT GLOB '*[^0-9a-f]*'))",
        ),
    ),
    59: (
        ("cayu_sessions", "instance_id", "TEXT"),
        ("cayu_tasks", "session_instance_id", "TEXT"),
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
    41: (
        (
            "cayu_knowledge_publication_receipts",
            "access_snapshot_json",
            "TEXT NOT NULL",
        ),
    ),
    50: (
        (
            "cayu_eval_runs",
            "invocation_json",
            "TEXT NOT NULL DEFAULT "
            '\'{"schema_version":1,"source":"sdk_run","origin":null,'
            '"max_steps":null,"limits":null,"cost_budget":null}\'',
        ),
    ),
    54: (
        (
            "cayu_events",
            "file_attachment_attestations_runtime_owned",
            "INTEGER NOT NULL DEFAULT 0 CHECK "
            "(file_attachment_attestations_runtime_owned IN (0, 1))",
        ),
    ),
    56: (
        (
            "cayu_eval_runs",
            "scenario_progress_json",
            "TEXT CHECK (scenario_progress_json IS NULL OR "
            "(json_valid(scenario_progress_json) AND "
            "length(CAST(scenario_progress_json AS BLOB)) BETWEEN 1 AND 262144))",
        ),
    ),
    57: (
        (
            "cayu_session_message_queue",
            "message_json",
            "TEXT CHECK (message_json IS NULL OR json_valid(message_json))",
        ),
    ),
    65: (
        (
            "cayu_knowledge_revisions",
            "payload_bytes",
            "INTEGER NOT NULL DEFAULT 1 CHECK (payload_bytes > 0 AND payload_bytes <= 2147483647)",
        ),
    ),
    73: (
        (
            "cayu_agent_recall_deliveries",
            "processing_schema_version",
            "TEXT COLLATE BINARY NOT NULL CHECK (processing_schema_version = "
            "'cayu.agent_recall_processing.v3')",
        ),
    ),
    74: (
        (
            "cayu_eval_runs",
            "trial_checkpoint_count",
            "INTEGER NOT NULL DEFAULT 0 CHECK (trial_checkpoint_count BETWEEN 0 AND 100000)",
        ),
        (
            "cayu_eval_runs",
            "trial_checkpoint_bytes",
            "INTEGER NOT NULL DEFAULT 0 CHECK (trial_checkpoint_bytes BETWEEN 0 AND 41943040)",
        ),
        (
            "cayu_eval_runs",
            "authored_suite_launch_revision",
            "TEXT COLLATE BINARY CHECK (authored_suite_launch_revision IS NULL OR "
            "(length(authored_suite_launch_revision) = 71 AND "
            "substr(authored_suite_launch_revision, 1, 7) = 'sha256:' AND "
            "substr(authored_suite_launch_revision, 8) NOT GLOB '*[^0-9a-f]*'))",
        ),
        (
            "cayu_eval_runs",
            "authored_suite_launch_lane",
            "INTEGER CHECK (authored_suite_launch_lane IS NULL OR "
            "authored_suite_launch_lane BETWEEN 0 AND 63)",
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


def _reject_populated_pre_targeted_tool_grant_database(
    connection: sqlite3.Connection,
) -> None:
    if connection.execute("SELECT EXISTS(SELECT 1 FROM cayu_sessions)").fetchone()[0]:
        raise schema.SchemaTooOld(
            "Storage revision 52 is a clean prerelease break and cannot migrate a "
            "populated Cayu session database. Recreate the Cayu database before "
            "starting this build."
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


def _reject_populated_pre_recall_subscription_database(
    connection: sqlite3.Connection,
) -> None:
    checkpoint_exists = connection.execute(
        "SELECT 1 FROM sqlite_master "
        "WHERE type = 'table' AND name = 'cayu_agent_recall_checkpoints'"
    ).fetchone()
    if checkpoint_exists is not None:
        checkpoint_columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(cayu_agent_recall_checkpoints)")
        }
        if "checkpoint_stream_id" not in checkpoint_columns:
            raise schema.SchemaTooOld(
                "Storage revision 73 introduces independent recall checkpoint streams and "
                "does not migrate the prerelease checkpoint schema. Recreate the Cayu "
                "database before starting this build."
            )
    delivery_exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'cayu_agent_recall_deliveries'"
    ).fetchone()
    if delivery_exists is None:
        return
    if connection.execute("SELECT EXISTS(SELECT 1 FROM cayu_agent_recall_deliveries)").fetchone()[
        0
    ]:
        raise schema.SchemaTooOld(
            "Storage revision 73 binds recall results to exact subscription input "
            "and cannot migrate a populated recall-delivery database without "
            "inventing missing retrieval authority. Recreate the Cayu database before "
            "starting this build."
        )


def _reject_populated_pre_knowledge_access_snapshot_database(
    connection: sqlite3.Connection,
) -> None:
    if (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' "
            "AND name = 'cayu_knowledge_publication_receipts'"
        ).fetchone()
        is None
    ):
        return
    if connection.execute(
        "SELECT EXISTS(SELECT 1 FROM cayu_knowledge_publication_receipts)"
    ).fetchone()[0]:
        raise schema.SchemaTooOld(
            "Storage revision 41 requires an authorization snapshot for every "
            "knowledge publication receipt and cannot infer one for existing "
            "receipts. Recreate the Cayu database before starting this build."
        )


def _reject_populated_pre_knowledge_revision_database(
    connection: sqlite3.Connection,
) -> None:
    candidates = (
        "cayu_knowledge_entries",
        "cayu_knowledge_labels",
        "cayu_knowledge_aspects",
        "cayu_knowledge_impact_targets",
        "cayu_knowledge_chunks",
        "cayu_knowledge_publication_receipts",
        "cayu_knowledge_embeddings",
    )
    existing = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'cayu_knowledge_%'"
        )
    }
    inspected = [table for table in candidates if table in existing]
    if not inspected:
        return
    counts = {
        table: int(
            connection.execute(f"SELECT EXISTS(SELECT 1 FROM {table} LIMIT 1)").fetchone()[0]
        )
        for table in inspected
    }
    require_empty_knowledge_revision_transition(
        counts,
        required_tables=inspected,
    )


_KNOWLEDGE_RELATION_CLEAN_BREAK_TABLES = (
    "cayu_knowledge_entries",
    "cayu_knowledge_revisions",
    "cayu_knowledge_chunks",
    "cayu_knowledge_chunks_fts",
    "cayu_knowledge_labels",
    "cayu_knowledge_aspects",
    "cayu_knowledge_impact_targets",
    "cayu_knowledge_evidence",
    "cayu_knowledge_publication_receipts",
    "cayu_knowledge_relations",
    "cayu_knowledge_relation_publication_receipts",
    "cayu_knowledge_changes",
    "cayu_knowledge_change_audiences",
    "cayu_knowledge_change_labels",
    "cayu_knowledge_change_consumers",
    "cayu_knowledge_change_acknowledgements",
    "cayu_knowledge_index_readiness_events",
    "cayu_knowledge_index_readiness_current",
    "cayu_knowledge_embeddings",
)
_KNOWLEDGE_MAINTENANCE_CLEAN_BREAK_TABLES = (
    *_KNOWLEDGE_RELATION_CLEAN_BREAK_TABLES,
    "cayu_knowledge_maintenance_decisions",
    "cayu_knowledge_maintenance_proposals",
)


def _reject_populated_pre_knowledge_relation_database(
    connection: sqlite3.Connection,
) -> None:
    _reject_populated_pre_knowledge_contract_database(
        connection,
        candidates=_KNOWLEDGE_RELATION_CLEAN_BREAK_TABLES,
        revision=60,
        contract="knowledge-lineage",
    )


def _reject_populated_pre_knowledge_maintenance_database(
    connection: sqlite3.Connection,
) -> None:
    _reject_populated_pre_knowledge_contract_database(
        connection,
        candidates=_KNOWLEDGE_MAINTENANCE_CLEAN_BREAK_TABLES,
        revision=63,
        contract="reviewed-maintenance",
    )


def _reject_populated_pre_bounded_knowledge_entry_database(
    connection: sqlite3.Connection,
) -> None:
    _reject_populated_pre_knowledge_contract_database(
        connection,
        candidates=_KNOWLEDGE_MAINTENANCE_CLEAN_BREAK_TABLES,
        revision=65,
        contract="bounded-entry-read",
    )


def _reject_populated_pre_knowledge_contract_database(
    connection: sqlite3.Connection,
    *,
    candidates: tuple[str, ...],
    revision: int,
    contract: str,
) -> None:
    existing = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'cayu_knowledge_%'"
        )
    }
    for table in candidates:
        if table not in existing:
            continue
        if connection.execute(f"SELECT EXISTS(SELECT 1 FROM {table} LIMIT 1)").fetchone()[0]:
            raise schema.SchemaTooOld(
                f"Storage revision {revision} is a clean prerelease {contract} break "
                "and cannot migrate a populated Cayu knowledge database. Recreate "
                "the Cayu knowledge database before starting this build."
            )


def _reject_populated_pre_transcript_search_database(
    connection: sqlite3.Connection,
) -> None:
    table = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'cayu_transcript_messages'"
    ).fetchone()
    if table is None:
        return
    if connection.execute("SELECT EXISTS(SELECT 1 FROM cayu_transcript_messages)").fetchone()[0]:
        raise schema.SchemaTooOld(
            "Storage revision 46 requires the final transcript-search projection "
            "on every transcript row and deliberately does not backfill earlier "
            "data. Recreate the Cayu database before starting this build."
        )


def _reject_populated_pre_result_resolver_database(
    connection: sqlite3.Connection,
) -> None:
    table = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'cayu_work_contracts'"
    ).fetchone()
    if table is None:
        return
    if connection.execute("SELECT EXISTS(SELECT 1 FROM cayu_work_contracts)").fetchone()[0]:
        raise schema.SchemaTooOld(
            "Storage revision 59 requires an exact result-resolver identity for every "
            "verified-work contract and cannot infer one for existing contracts. "
            "Recreate the Cayu task database before starting this build."
        )


def _backfill_session_instance_ids(connection: sqlite3.Connection) -> None:
    rows = connection.execute(
        "SELECT id FROM cayu_sessions WHERE instance_id IS NULL ORDER BY id"
    ).fetchall()
    for row in rows:
        connection.execute(
            "UPDATE cayu_sessions SET instance_id = ? WHERE id = ? AND instance_id IS NULL",
            (str(uuid4()), row[0]),
        )


def _validate_session_instance_schema(connection: sqlite3.Connection) -> None:
    session_columns = {row[1] for row in connection.execute("PRAGMA table_info(cayu_sessions)")}
    task_columns = {row[1] for row in connection.execute("PRAGMA table_info(cayu_tasks)")}
    if "instance_id" not in session_columns or "session_instance_id" not in task_columns:
        raise RuntimeError("SQLite session-instance authority columns are missing.")
    invalid = connection.execute(
        "SELECT EXISTS(SELECT 1 FROM cayu_sessions WHERE instance_id IS NULL)"
    ).fetchone()
    if invalid is None or invalid[0]:
        raise RuntimeError("SQLite session-instance authority is incomplete.")


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
_KNOWLEDGE_CHUNK_REVISION_COLUMNS = (
    "fts_rowid",
    "id",
    "entry_id",
    "entry_revision",
    "chunk_index",
    "text",
    "content_hash",
    "source_uri",
    "metadata_json",
)


def _sqlite_table_columns(connection: sqlite3.Connection, table: str) -> tuple[str, ...]:
    return tuple(str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})"))


def _sqlite_has_unique_index(
    connection: sqlite3.Connection,
    table: str,
    columns: tuple[str, ...],
) -> bool:
    primary_key_columns = tuple(
        str(row[1])
        for row in sorted(
            (row for row in connection.execute(f"PRAGMA table_info({table})") if row[5]),
            key=lambda row: int(row[5]),
        )
    )
    if primary_key_columns == columns:
        return True
    for row in connection.execute(f"PRAGMA index_list({table})"):
        if not bool(row[2]):
            continue
        index_columns = tuple(
            str(column[2]) for column in connection.execute(f"PRAGMA index_info({row[1]})")
        )
        if index_columns == columns:
            return True
    return False


def _sqlite_foreign_key_groups(
    connection: sqlite3.Connection,
    table: str,
) -> set[tuple[str, tuple[str, ...], tuple[str, ...], str]]:
    grouped: dict[int, list[sqlite3.Row]] = {}
    for row in connection.execute(f"PRAGMA foreign_key_list({table})"):
        grouped.setdefault(int(row[0]), []).append(row)
    result: set[tuple[str, tuple[str, ...], tuple[str, ...], str]] = set()
    for rows in grouped.values():
        ordered = sorted(rows, key=lambda row: int(row[1]))
        result.add(
            (
                str(ordered[0][2]),
                tuple(str(row[3]) for row in ordered),
                tuple(str(row[4]) for row in ordered),
                str(ordered[0][6]).upper(),
            )
        )
    return result


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
    if columns == _KNOWLEDGE_CHUNK_REVISION_COLUMNS:
        # A current binary may be recovering a revision-42 schema whose ledger
        # was restored or rewound independently. Revision 42 will validate the
        # revision-bound layout later in the same migration sequence; revision
        # 37 must not reject that known-newer shape first.
        return
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


def _migrate_deferred_interaction_input_payloads(connection: sqlite3.Connection) -> None:
    rows = connection.execute(
        "SELECT session_id, interaction_id, source_messages_json "
        "FROM cayu_deferred_interaction_inputs ORDER BY session_id"
    ).fetchall()
    for row in rows:
        try:
            payload = json.loads(row["source_messages_json"])
            if type(payload) is list:
                payload = {
                    "source_messages": payload,
                    "initial_transcript_messages": None,
                }
            stable = deferred_interaction_input_from_storage_payload(
                row["interaction_id"],
                payload,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "SQLite deferred interaction input cannot be migrated to revision 62."
            ) from exc
        connection.execute(
            "UPDATE cayu_deferred_interaction_inputs SET source_messages_json = ? "
            "WHERE session_id = ?",
            (
                json_dumps(deferred_interaction_input_storage_payload(stable)),
                row["session_id"],
            ),
        )


def _work_attempt_continuation_authority(
    connection: sqlite3.Connection,
    admission_json: object,
) -> tuple[dict[str, Any], dict[str, Any] | None, str | None]:
    if type(admission_json) is not dict:
        raise ValueError("Work-attempt admission payload must be an object.")
    stable_admission_json = cast("dict[str, Any]", admission_json)
    continuation = stable_admission_json.get("continuation")
    if continuation is None:
        return stable_admission_json, None, None
    if type(continuation) is not dict:
        raise ValueError("Work-attempt continuation payload must be an object.")
    stable_continuation = cast("dict[str, Any]", continuation)
    prior_attempt_id = stable_continuation.get("prior_attempt_id")
    if type(prior_attempt_id) is not str or not prior_attempt_id.strip():
        raise ValueError("Work-attempt continuation has no prior attempt identity.")
    row = connection.execute(
        "SELECT admission_id FROM cayu_work_attempt_admissions WHERE attempt_id = ?",
        (prior_attempt_id,),
    ).fetchone()
    if row is None:
        raise ValueError("Work-attempt continuation predecessor is missing.")
    return stable_admission_json, stable_continuation, str(row["admission_id"])


def _migrate_work_attempt_continuation_authority(connection: sqlite3.Connection) -> None:
    rows = connection.execute(
        "SELECT admission_id, admission_json FROM cayu_work_attempt_admissions "
        "ORDER BY admission_id"
    ).fetchall()
    for row in rows:
        try:
            admission_json, continuation, prior_admission_id = _work_attempt_continuation_authority(
                connection,
                json.loads(row["admission_json"]),
            )
            if continuation is None:
                continue
            if "prior_admission_id" in continuation:
                stored_prior_admission_id = continuation["prior_admission_id"]
                if stored_prior_admission_id != prior_admission_id:
                    raise ValueError("Work-attempt continuation predecessor authority conflicts.")
                continue
            migrated_continuation = dict(continuation)
            migrated_continuation["prior_admission_id"] = prior_admission_id
            migrated_admission = dict(admission_json)
            migrated_admission["continuation"] = migrated_continuation
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "SQLite work-attempt continuation cannot be migrated to revision 62."
            ) from exc
        connection.execute(
            "UPDATE cayu_work_attempt_admissions SET admission_json = ? WHERE admission_id = ?",
            (json_dumps(migrated_admission), row["admission_id"]),
        )


def _validate_revision_sixty_two_payload_schema(connection: sqlite3.Connection) -> None:
    """Validate revision-62 storage shape without scanning durable history."""

    deferred_columns = tuple(
        (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
        for row in connection.execute("PRAGMA table_info(cayu_deferred_interaction_inputs)")
    )
    if deferred_columns != (
        ("session_id", "TEXT", 0, 1),
        ("interaction_id", "TEXT", 1, 0),
        ("source_messages_json", "TEXT", 1, 0),
    ):
        raise RuntimeError(
            "SQLite deferred interaction input schema conflicts with Cayu's "
            "revision-62 durable payload contract. Run `cayu storage migrate` "
            "or restore the database from a known-good backup."
        )


def _migrate_revision_sixty_two_payloads(connection: sqlite3.Connection) -> None:
    _migrate_deferred_interaction_input_payloads(connection)
    _migrate_work_attempt_continuation_authority(connection)


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
    59: _backfill_session_instance_ids,
    62: _migrate_revision_sixty_two_payloads,
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


def _validate_local_execution_attempt_schema(connection: sqlite3.Connection) -> None:
    expected_columns = (
        ("attempt_id", "TEXT", 1),
        ("task_id", "TEXT", 1),
        ("retry_series_id", "TEXT", 0),
        ("effect_lineage_id", "TEXT", 1),
        ("request_sha256", "TEXT", 1),
        ("phase", "TEXT", 1),
        ("quiescence", "TEXT", 1),
        ("retry_admissible", "INTEGER", 1),
        ("recovery_generation", "INTEGER", 1),
        ("recovery_owner_id", "TEXT", 0),
        ("recovery_owner_expires_at", "TEXT", 0),
        ("record_json", "TEXT", 1),
        ("created_at", "TEXT", 1),
        ("updated_at", "TEXT", 1),
    )
    actual_columns = tuple(
        (str(row[1]), str(row[2]).upper(), int(row[3]))
        for row in connection.execute("PRAGMA table_info(cayu_local_execution_attempts)")
    )
    if actual_columns != expected_columns:
        raise RuntimeError("SQLite local execution-attempt storage conflicts with revision 66.")
    table_row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' "
        "AND name = 'cayu_local_execution_attempts'"
    ).fetchone()
    definition = (
        ""
        if table_row is None or table_row[0] is None
        else " ".join(str(table_row[0]).lower().split())
    )
    required_fragments = (
        "references cayu_tasks(id) on delete restrict",
        "json_valid(record_json)",
        "retry_admissible in (0, 1)",
        "unique (task_id, effect_lineage_id, attempt_id)",
    )
    if any(fragment not in definition for fragment in required_fragments):
        raise RuntimeError("SQLite local execution-attempt constraints conflict with revision 66.")
    expected_indexes = {
        "idx_cayu_local_execution_attempts_task_fence": (
            "task_id",
            "retry_admissible",
            "created_at",
            "attempt_id",
        ),
        "idx_cayu_local_execution_attempts_lineage": (
            "retry_series_id",
            "task_id",
            "effect_lineage_id",
            "created_at",
            "attempt_id",
        ),
        "idx_cayu_local_execution_attempts_recovery": (
            "retry_admissible",
            "phase",
            "updated_at",
            "attempt_id",
        ),
        "idx_cayu_local_execution_attempts_discovery": (
            "created_at",
            "attempt_id",
        ),
    }
    for index_name, expected in expected_indexes.items():
        row = connection.execute(
            "SELECT tbl_name FROM sqlite_master WHERE type = 'index' AND name = ?",
            (index_name,),
        ).fetchone()
        columns = tuple(
            str(index_row[2])
            for index_row in connection.execute(f"PRAGMA index_info({index_name})")
        )
        if row is None or row[0] != "cayu_local_execution_attempts" or columns != expected:
            raise RuntimeError(f"SQLite schema object {index_name!r} conflicts with revision 66.")


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
    state = read_schema_state(connection)
    if (
        schema_mode is not schema.SchemaMode.VALIDATE
        and state.revision < 42
        and any(revision.revision == 42 for revision in schema.pending(state.revision))
    ):
        # This check intentionally precedes even bookkeeping-table DDL. A
        # populated unversioned/partially versioned knowledge schema is not a
        # fresh database and must remain recoverable for an explicit reset.
        _reject_populated_pre_knowledge_revision_database(connection)
    if (
        schema_mode is not schema.SchemaMode.VALIDATE
        and state.revision < 60
        and any(revision.revision == 60 for revision in schema.pending(state.revision))
    ):
        _reject_populated_pre_knowledge_relation_database(connection)
    if (
        schema_mode is not schema.SchemaMode.VALIDATE
        and state.revision < 63
        and any(revision.revision == 63 for revision in schema.pending(state.revision))
    ):
        _reject_populated_pre_knowledge_maintenance_database(connection)
    if (
        schema_mode is not schema.SchemaMode.VALIDATE
        and state.revision < 65
        and any(revision.revision == 65 for revision in schema.pending(state.revision))
    ):
        _reject_populated_pre_bounded_knowledge_entry_database(connection)
    if (
        schema_mode is not schema.SchemaMode.VALIDATE
        and state.revision == schema.UNINITIALIZED
        and any(revision.revision == 46 for revision in schema.pending(state.revision))
    ):
        # Refuse before even creating migration bookkeeping in an unversioned
        # database. The old transcript remains untouched for an explicit reset.
        _reject_populated_pre_transcript_search_database(connection)
    if (
        schema_mode is not schema.SchemaMode.VALIDATE
        and state.revision < 59
        and any(revision.revision == 59 for revision in schema.pending(state.revision))
    ):
        _reject_populated_pre_result_resolver_database(connection)
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
    if 37 <= current.revision < 42:
        # Structural validation is intentionally constant-size. The full source/
        # FTS census belongs to the one-time revision hook, never ordinary startup.
        _validate_revision_37_knowledge_fts_schema(connection)
    if current.revision >= 42:
        _validate_revision_42_knowledge_schema(
            connection,
            require_payload_bytes=current.revision >= 65,
        )
    if current.revision >= 43:
        _validate_revision_43_knowledge_schema(
            connection,
            relation_aware=current.revision >= 60,
        )
    if current.revision >= 44:
        _validate_revision_44_knowledge_schema(connection)
    if current.revision >= 60:
        _validate_revision_60_knowledge_schema(connection)
    if current.revision >= 63:
        _validate_revision_63_knowledge_schema(connection)
    if current.revision >= 67:
        _validate_revision_67_knowledge_schema(connection)
    if current.revision >= 69:
        _validate_revision_69_work_context_schema(connection)
    if current.revision >= 71:
        _validate_revision_71_recall_delivery_schema(
            connection,
            require_processing_schema_version=current.revision >= 73,
        )
    if current.revision >= 73:
        _validate_revision_73_recall_subscription_schema(connection)
    if app_min_supported >= 38:
        _validate_task_terminalization_receipt_table(connection)
    if app_min_supported >= 70:
        _validate_interrupted_task_handoff_schema(connection)
    if app_min_supported >= 39:
        _validate_task_invocation_column(connection)
    if app_min_supported >= 41:
        _validate_knowledge_publication_access_snapshot_column(connection)
    if app_min_supported >= 45:
        _validate_task_retry_series_schema(connection)
    if app_min_supported >= 46:
        _validate_revision_46_transcript_search_schema(connection)
    if app_min_supported >= 47:
        _validate_eval_result_baseline_schema(connection)
    if app_min_supported >= 48:
        _validate_captured_eval_case_schema(connection)
    if app_min_supported >= 49:
        _validate_verified_work_schema(
            connection,
            require_verifier_profiles=current.revision >= 58,
        )
    if app_min_supported >= 50:
        _validate_eval_run_invocation_column(connection)
    if app_min_supported >= 51:
        _validate_memory_evidence_schema(connection)
    if app_min_supported >= 52:
        _validate_targeted_tool_grant_schema(connection)
    if app_min_supported >= 53:
        _validate_eval_scenario_schema(connection)
    if app_min_supported >= 55:
        _validate_task_retry_reconciliation_schema(connection)
    if app_min_supported >= 56:
        _validate_eval_run_scenario_progress_column(connection)
    if app_min_supported >= 57:
        _validate_session_message_queue_typed_message_column(connection)
    if app_min_supported >= 59:
        _validate_session_instance_schema(connection)
    if app_min_supported >= 61:
        _validate_work_attempt_admission_schema(connection)
    if app_min_supported >= 62:
        # The revision hook performs the one-time complete payload census.
        # Ordinary startup remains independent of durable history size; each
        # payload is validated again at its indexed read boundary.
        _validate_revision_sixty_two_payload_schema(connection)
    if app_min_supported >= 64:
        _validate_eval_authored_suite_schema(connection)
    if app_min_supported >= 66:
        _validate_local_execution_attempt_schema(connection)
    if app_min_supported >= 68:
        _validate_eval_judge_calibration_schema(connection)
    if app_min_supported >= 72:
        _validate_eval_run_max_concurrency_schema(connection)
    if app_min_supported >= 74:
        _validate_eval_run_trial_checkpoint_schema(connection)


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


def _validate_targeted_tool_grant_schema(connection: sqlite3.Connection) -> None:
    expected_columns = {
        "cayu_targeted_tool_grants": (
            ("grant_id", "TEXT", 0, 1),
            ("session_id", "TEXT", 1, 0),
            ("interaction_id", "TEXT", 1, 0),
            ("request_id", "TEXT", 1, 0),
            ("tool_ref", "TEXT", 1, 0),
            ("generation_id", "TEXT", 1, 0),
            ("tool_id", "TEXT", 1, 0),
            ("tool_name", "TEXT", 1, 0),
            ("catalogue_revision", "TEXT", 1, 0),
            ("descriptor_version", "TEXT", 1, 0),
            ("issued_at", "TEXT", 1, 0),
            ("expires_at", "TEXT", 1, 0),
            ("max_calls", "INTEGER", 1, 0),
            ("used_calls", "INTEGER", 1, 0),
            ("revoked_at", "TEXT", 0, 0),
            ("record_json", "TEXT", 1, 0),
        ),
        "cayu_targeted_tool_grant_uses": (
            ("use_id", "TEXT", 0, 1),
            ("grant_id", "TEXT", 1, 0),
            ("session_id", "TEXT", 1, 0),
            ("interaction_id", "TEXT", 1, 0),
            ("model_step_id", "TEXT", 1, 0),
            ("outer_tool_call_id", "TEXT", 1, 0),
            ("arguments_sha256", "TEXT", 1, 0),
            ("invocation_id", "TEXT", 1, 0),
            ("bound_at", "TEXT", 1, 0),
            ("record_json", "TEXT", 1, 0),
        ),
    }
    for table, expected in expected_columns.items():
        actual = tuple(
            (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
            for row in connection.execute(f"PRAGMA table_info({table})")
        )
        if actual != expected:
            raise RuntimeError(
                f"SQLite schema object {table!r} conflicts with Cayu's revision-52 "
                "targeted-grant durability contract. Run `cayu storage migrate` or "
                "restore the database from a known-good backup."
            )

    required_indexes = {
        "cayu_public_authority_aliases": {
            (False, ("field_name", "public_alias")),
        },
        "cayu_targeted_tool_grants": {
            (False, ("session_id", "interaction_id", "issued_at", "grant_id")),
            (True, ("session_id", "interaction_id", "request_id")),
            (True, ("session_id", "interaction_id", "tool_id")),
        },
        "cayu_targeted_tool_grant_uses": {
            (False, ("grant_id", "bound_at", "use_id")),
            (True, ("session_id", "interaction_id", "invocation_id")),
            (True, ("session_id", "interaction_id", "outer_tool_call_id")),
        },
    }
    for table, required in required_indexes.items():
        actual: set[tuple[bool, tuple[str, ...]]] = set()
        for index_row in connection.execute(f"PRAGMA index_list({table})"):
            columns = tuple(
                str(column_row[0])
                for column_row in connection.execute(
                    "SELECT name FROM pragma_index_info(?) ORDER BY seqno",
                    (str(index_row[1]),),
                )
            )
            actual.add((bool(index_row[2]), columns))
        if not required <= actual:
            raise RuntimeError(
                f"SQLite indexes for {table!r} conflict with Cayu's revision-52 "
                "targeted-grant contention contract."
            )

    expected_foreign_keys = {
        "cayu_targeted_tool_grants": {
            ("cayu_sessions", "session_id", "id", "CASCADE"),
        },
        "cayu_targeted_tool_grant_uses": {
            ("cayu_targeted_tool_grants", "grant_id", "grant_id", "CASCADE"),
            ("cayu_sessions", "session_id", "id", "CASCADE"),
        },
    }
    for table, expected in expected_foreign_keys.items():
        actual_foreign_keys: set[tuple[str, str, str, str]] = {
            (str(row[2]), str(row[3]), str(row[4]), str(row[6]).upper())
            for row in connection.execute(f"PRAGMA foreign_key_list({table})")
        }
        if actual_foreign_keys != expected:
            raise RuntimeError(
                f"SQLite foreign keys for {table!r} conflict with Cayu's revision-52 "
                "targeted-grant scope contract."
            )

    required_table_fragments = {
        "cayu_targeted_tool_grants": {
            "check(max_calls>=1andmax_calls<=32)",
            "check(used_calls>=0andused_calls<=max_calls)",
            "check(json_valid(record_json))",
        },
        "cayu_targeted_tool_grant_uses": {"check(json_valid(record_json))"},
    }
    for table, required in required_table_fragments.items():
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        normalized = "" if row is None or row[0] is None else "".join(str(row[0]).lower().split())
        if any(fragment not in normalized for fragment in required):
            raise RuntimeError(
                f"SQLite checks for {table!r} conflict with Cayu's revision-52 "
                "targeted-grant budget contract."
            )


def _validate_revision_42_knowledge_schema(
    connection: sqlite3.Connection,
    *,
    require_payload_bytes: bool = False,
) -> None:
    required_tables = {
        "cayu_knowledge_entries": (
            ("id", "TEXT", 0, 1),
            ("namespace", "TEXT", 1, 0),
            ("current_revision", "INTEGER", 1, 0),
            ("created_at", "TEXT", 1, 0),
            ("updated_at", "TEXT", 1, 0),
        ),
        "cayu_knowledge_revisions": (
            ("entry_id", "TEXT", 1, 1),
            ("revision", "INTEGER", 1, 2),
            ("text", "TEXT", 1, 0),
            ("kind", "TEXT", 1, 0),
            ("visibility", "TEXT", 1, 0),
            ("status", "TEXT", 1, 0),
            ("created_by_type", "TEXT", 1, 0),
            ("created_by", "TEXT", 1, 0),
            ("created_at", "TEXT", 1, 0),
            ("updated_at", "TEXT", 1, 0),
            ("source_type", "TEXT", 0, 0),
            ("source_uri", "TEXT", 0, 0),
            ("source_id", "TEXT", 0, 0),
            ("source_hash", "TEXT", 0, 0),
            ("importance", "REAL", 0, 0),
            ("importance_source", "TEXT", 0, 0),
            ("confidence", "REAL", 0, 0),
            ("last_used_at", "TEXT", 0, 0),
            ("expires_at", "TEXT", 0, 0),
            ("title", "TEXT", 0, 0),
            ("metadata_json", "TEXT", 1, 0),
            *((("payload_bytes", "INTEGER", 1, 0),) if require_payload_bytes else ()),
        ),
        "cayu_knowledge_labels": (
            ("entry_id", "TEXT", 1, 1),
            ("entry_revision", "INTEGER", 1, 2),
            ("key", "TEXT", 1, 3),
            ("value", "TEXT", 1, 0),
        ),
        "cayu_knowledge_aspects": (
            ("entry_id", "TEXT", 1, 1),
            ("entry_revision", "INTEGER", 1, 2),
            ("aspect", "TEXT", 1, 3),
        ),
        "cayu_knowledge_impact_targets": (
            ("entry_id", "TEXT", 1, 1),
            ("entry_revision", "INTEGER", 1, 2),
            ("impact_target", "TEXT", 1, 3),
        ),
        "cayu_knowledge_chunks": (
            ("fts_rowid", "INTEGER", 0, 1),
            ("id", "TEXT", 1, 0),
            ("entry_id", "TEXT", 1, 0),
            ("entry_revision", "INTEGER", 1, 0),
            ("chunk_index", "INTEGER", 1, 0),
            ("text", "TEXT", 1, 0),
            ("content_hash", "TEXT", 0, 0),
            ("source_uri", "TEXT", 0, 0),
            ("metadata_json", "TEXT", 1, 0),
        ),
        "cayu_knowledge_publication_receipts": (
            ("operation_id", "TEXT", 0, 1),
            ("entry_id", "TEXT", 1, 0),
            ("entry_revision", "INTEGER", 1, 0),
            ("expected_revision", "INTEGER", 0, 0),
            ("request_sha256", "TEXT", 1, 0),
            ("entry_created_at", "TEXT", 1, 0),
            ("entry_updated_at", "TEXT", 1, 0),
            ("committed_at", "TEXT", 1, 0),
            ("access_snapshot_json", "TEXT", 1, 0),
        ),
    }
    for table, expected in required_tables.items():
        rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
        actual = tuple((str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5])) for row in rows)
        if actual != expected:
            _raise_revision_42_sqlite_schema_error(table)

    expected_foreign_keys = {
        "cayu_knowledge_entries": (
            ("cayu_knowledge_revisions", "current_revision", "revision", "NO ACTION"),
            ("cayu_knowledge_revisions", "id", "entry_id", "NO ACTION"),
        ),
        "cayu_knowledge_revisions": (("cayu_knowledge_entries", "entry_id", "id", "CASCADE"),),
        "cayu_knowledge_labels": (
            ("cayu_knowledge_revisions", "entry_id", "entry_id", "CASCADE"),
            ("cayu_knowledge_revisions", "entry_revision", "revision", "CASCADE"),
        ),
        "cayu_knowledge_aspects": (
            ("cayu_knowledge_revisions", "entry_id", "entry_id", "CASCADE"),
            ("cayu_knowledge_revisions", "entry_revision", "revision", "CASCADE"),
        ),
        "cayu_knowledge_impact_targets": (
            ("cayu_knowledge_revisions", "entry_id", "entry_id", "CASCADE"),
            ("cayu_knowledge_revisions", "entry_revision", "revision", "CASCADE"),
        ),
        "cayu_knowledge_chunks": (
            ("cayu_knowledge_revisions", "entry_id", "entry_id", "CASCADE"),
            ("cayu_knowledge_revisions", "entry_revision", "revision", "CASCADE"),
        ),
        "cayu_knowledge_publication_receipts": (),
    }
    for table, expected in expected_foreign_keys.items():
        actual = tuple(
            sorted(
                (
                    str(row[2]),
                    str(row[3]),
                    str(row[4]),
                    str(row[6]).upper(),
                )
                for row in connection.execute(f"PRAGMA foreign_key_list({table})")
            )
        )
        if actual != expected:
            _raise_revision_42_sqlite_schema_error(table)

    required_unique_keys = {
        "cayu_knowledge_entries": (("id",),),
        "cayu_knowledge_revisions": (("entry_id", "revision"),),
        "cayu_knowledge_labels": (("entry_id", "entry_revision", "key"),),
        "cayu_knowledge_aspects": (("entry_id", "entry_revision", "aspect"),),
        "cayu_knowledge_impact_targets": (("entry_id", "entry_revision", "impact_target"),),
        "cayu_knowledge_chunks": (
            ("id",),
            ("entry_id", "entry_revision", "chunk_index"),
        ),
        "cayu_knowledge_publication_receipts": (("operation_id",),),
    }
    for table, keys in required_unique_keys.items():
        if any(not _sqlite_has_unique_index(connection, table, key) for key in keys):
            _raise_revision_42_sqlite_schema_error(table)

    required_indexes = {
        "idx_cayu_knowledge_entries_namespace_current": (
            "cayu_knowledge_entries",
            ("namespace", "current_revision", "id"),
        ),
        "idx_cayu_knowledge_revisions_status": (
            "cayu_knowledge_revisions",
            ("status", "entry_id", "revision"),
        ),
        "idx_cayu_knowledge_revisions_kind": (
            "cayu_knowledge_revisions",
            ("kind", "entry_id", "revision"),
        ),
        "idx_cayu_knowledge_revisions_visibility": (
            "cayu_knowledge_revisions",
            ("visibility", "entry_id", "revision"),
        ),
        "idx_cayu_knowledge_revisions_source": (
            "cayu_knowledge_revisions",
            ("source_type", "source_id", "entry_id", "revision"),
        ),
        "idx_cayu_knowledge_revisions_expires_at": (
            "cayu_knowledge_revisions",
            ("expires_at", "entry_id", "revision"),
        ),
        "idx_cayu_knowledge_labels_key_value_entry": (
            "cayu_knowledge_labels",
            ("key", "value", "entry_id", "entry_revision"),
        ),
        "idx_cayu_knowledge_aspects_aspect_entry": (
            "cayu_knowledge_aspects",
            ("aspect", "entry_id", "entry_revision"),
        ),
        "idx_cayu_knowledge_impact_targets_target_entry": (
            "cayu_knowledge_impact_targets",
            ("impact_target", "entry_id", "entry_revision"),
        ),
        "idx_cayu_knowledge_chunks_entry_revision_index": (
            "cayu_knowledge_chunks",
            ("entry_id", "entry_revision", "chunk_index"),
        ),
        "idx_cayu_knowledge_publication_receipts_entry_revision": (
            "cayu_knowledge_publication_receipts",
            ("entry_id", "entry_revision"),
        ),
    }
    for index, (table, columns) in required_indexes.items():
        row = connection.execute(
            "SELECT tbl_name FROM sqlite_master WHERE type = 'index' AND name = ?",
            (index,),
        ).fetchone()
        actual_columns = tuple(
            str(column[2]) for column in connection.execute(f"PRAGMA index_info({index})")
        )
        if row is None or str(row[0]) != table or actual_columns != columns:
            _raise_revision_42_sqlite_schema_error(index)

    current_view = connection.execute(
        "SELECT type, sql FROM sqlite_master WHERE name = 'cayu_knowledge_current_entries'"
    ).fetchone()
    current_view_sql = _normalize_sqlite_schema_sql(
        None if current_view is None else current_view[1]
    )
    current_view_columns = _sqlite_table_columns(
        connection,
        "cayu_knowledge_current_entries",
    )
    if (
        current_view is None
        or str(current_view[0]) != "view"
        or current_view_columns
        != (
            "id",
            "revision",
            "namespace",
            "text",
            "kind",
            "visibility",
            "status",
            "created_by_type",
            "created_by",
            "created_at",
            "updated_at",
            "source_type",
            "source_uri",
            "source_id",
            "source_hash",
            "importance",
            "importance_source",
            "confidence",
            "last_used_at",
            "expires_at",
            "title",
            "metadata_json",
            *(("payload_bytes",) if require_payload_bytes else ()),
        )
        or "from cayu_knowledge_entries as logical" not in current_view_sql
        or "join cayu_knowledge_revisions as revision" not in current_view_sql
        or "revision.revision = logical.current_revision" not in current_view_sql
    ):
        _raise_revision_42_sqlite_schema_error("cayu_knowledge_current_entries")

    fts = connection.execute(
        "SELECT type, sql FROM sqlite_master WHERE name = 'cayu_knowledge_chunks_fts'"
    ).fetchone()
    fts_sql = _normalize_sqlite_schema_sql(None if fts is None else fts[1])
    if (
        fts is None
        or str(fts[0]) != "table"
        or _sqlite_table_columns(connection, "cayu_knowledge_chunks_fts")
        != ("entry_id", "entry_revision", "chunk_id", "title", "text")
        or "using fts5" not in fts_sql
        or any(
            fragment not in fts_sql
            for fragment in (
                "entry_id unindexed",
                "entry_revision unindexed",
                "chunk_id unindexed",
            )
        )
    ):
        _raise_revision_42_sqlite_schema_error("cayu_knowledge_chunks_fts")

    required_table_sql = {
        "cayu_knowledge_entries": (
            "check (current_revision > 0 and current_revision <= 2147483647)",
            "deferrable initially deferred",
        ),
        "cayu_knowledge_revisions": (
            "check (revision > 0 and revision <= 2147483647)",
            *(
                (
                    "payload_bytes > 0",
                    "payload_bytes <= 2147483647",
                )
                if require_payload_bytes
                else ()
            ),
        ),
        "cayu_knowledge_chunks": (
            "check (entry_revision > 0 and entry_revision <= 2147483647)",
            "check (chunk_index >= 0)",
        ),
        "cayu_knowledge_publication_receipts": (
            "check (entry_revision > 0 and entry_revision <= 2147483647)",
            "check (expected_revision > 0 and expected_revision <= 2147483647)",
            "entry_revision = expected_revision + 1",
        ),
    }
    for table, fragments in required_table_sql.items():
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        normalized = _normalize_sqlite_schema_sql(None if row is None else row[0])
        if any(fragment not in normalized for fragment in fragments):
            _raise_revision_42_sqlite_schema_error(table)


def _validate_revision_43_knowledge_schema(
    connection: sqlite3.Connection,
    *,
    relation_aware: bool = False,
) -> None:
    expected_columns = {
        "cayu_knowledge_evidence": (
            "id",
            "entry_id",
            "entry_revision",
            "chunk_id",
            "role",
            "source_type",
            "source_id",
            "source_uri",
            "source_revision",
            "source_hash",
            "locator_json",
            "disposition",
            "created_at",
            "metadata_json",
        ),
        "cayu_knowledge_changes": (
            "sequence",
            "id",
            "kind",
            "entry_id",
            "entry_revision",
            "committed_at",
            "operation_id",
            *(("relation_id",) if relation_aware else ()),
        ),
        "cayu_knowledge_change_audiences": (
            "change_sequence",
            "audience_kind",
            "namespace",
            "visibility",
            "source_type",
            "source_id",
            "status",
            "requires_include_expired",
        ),
        "cayu_knowledge_change_consumers": (
            "consumer_id",
            "access_scope_sha256",
            "cursor_sequence",
            "pending_change_sequence",
            "pending_claim_id",
            "pending_worker_id",
            "pending_attempt",
            "claimed_at",
            "lease_expires_at",
            "last_acknowledged_claim_id",
            "updated_at",
        ),
        "cayu_knowledge_change_acknowledgements": (
            "consumer_id",
            "claim_id",
            "claim_sha256",
            "change_sequence",
            "acknowledged_at",
        ),
        "cayu_knowledge_change_labels": (
            "change_sequence",
            "audience_kind",
            "key",
            "value",
        ),
    }
    for table, columns in expected_columns.items():
        if _sqlite_table_columns(connection, table) != columns:
            _raise_revision_43_sqlite_schema_error(table)

    evidence_foreign_keys = tuple(
        sorted(
            (
                str(row[2]),
                str(row[3]),
                str(row[4]),
                str(row[6]).upper(),
            )
            for row in connection.execute("PRAGMA foreign_key_list(cayu_knowledge_evidence)")
        )
    )
    if evidence_foreign_keys != tuple(
        sorted(
            (
                ("cayu_knowledge_chunks", "chunk_id", "id", "CASCADE"),
                ("cayu_knowledge_chunks", "entry_id", "entry_id", "CASCADE"),
                (
                    "cayu_knowledge_chunks",
                    "entry_revision",
                    "entry_revision",
                    "CASCADE",
                ),
                ("cayu_knowledge_revisions", "entry_id", "entry_id", "CASCADE"),
                (
                    "cayu_knowledge_revisions",
                    "entry_revision",
                    "revision",
                    "CASCADE",
                ),
            )
        )
    ):
        _raise_revision_43_sqlite_schema_error("cayu_knowledge_evidence")

    change_audience_foreign_keys = tuple(
        (
            str(row[2]),
            str(row[3]),
            str(row[4]),
            str(row[6]).upper(),
        )
        for row in connection.execute("PRAGMA foreign_key_list(cayu_knowledge_change_audiences)")
    )
    if change_audience_foreign_keys != (
        ("cayu_knowledge_changes", "change_sequence", "sequence", "CASCADE"),
    ):
        _raise_revision_43_sqlite_schema_error("cayu_knowledge_change_audiences")

    change_label_foreign_keys = tuple(
        sorted(
            (
                str(row[2]),
                str(row[3]),
                str(row[4]),
                str(row[6]).upper(),
            )
            for row in connection.execute("PRAGMA foreign_key_list(cayu_knowledge_change_labels)")
        )
    )
    if change_label_foreign_keys != tuple(
        sorted(
            (
                (
                    "cayu_knowledge_change_audiences",
                    "change_sequence",
                    "change_sequence",
                    "CASCADE",
                ),
                (
                    "cayu_knowledge_change_audiences",
                    "audience_kind",
                    "audience_kind",
                    "CASCADE",
                ),
            )
        )
    ):
        _raise_revision_43_sqlite_schema_error("cayu_knowledge_change_labels")

    consumer_foreign_keys = tuple(
        (
            str(row[2]),
            str(row[3]),
            str(row[4]),
            str(row[6]).upper(),
        )
        for row in connection.execute("PRAGMA foreign_key_list(cayu_knowledge_change_consumers)")
    )
    if consumer_foreign_keys != (
        ("cayu_knowledge_changes", "pending_change_sequence", "sequence", "NO ACTION"),
    ):
        _raise_revision_43_sqlite_schema_error("cayu_knowledge_change_consumers")

    acknowledgement_foreign_keys = tuple(
        sorted(
            (
                str(row[2]),
                str(row[3]),
                str(row[4]),
                str(row[6]).upper(),
            )
            for row in connection.execute(
                "PRAGMA foreign_key_list(cayu_knowledge_change_acknowledgements)"
            )
        )
    )
    if acknowledgement_foreign_keys != tuple(
        sorted(
            (
                (
                    "cayu_knowledge_change_consumers",
                    "consumer_id",
                    "consumer_id",
                    "CASCADE",
                ),
                (
                    "cayu_knowledge_changes",
                    "change_sequence",
                    "sequence",
                    "NO ACTION",
                ),
            )
        )
    ):
        _raise_revision_43_sqlite_schema_error("cayu_knowledge_change_acknowledgements")

    for table, key in (
        ("cayu_knowledge_evidence", ("id",)),
        ("cayu_knowledge_changes", ("sequence",)),
        ("cayu_knowledge_changes", ("id",)),
        (
            "cayu_knowledge_change_audiences",
            ("change_sequence", "audience_kind"),
        ),
        ("cayu_knowledge_change_consumers", ("consumer_id",)),
        (
            "cayu_knowledge_change_acknowledgements",
            ("consumer_id", "claim_id"),
        ),
        (
            "cayu_knowledge_change_labels",
            ("change_sequence", "audience_kind", "key"),
        ),
        (
            "cayu_knowledge_chunks",
            ("id", "entry_id", "entry_revision"),
        ),
    ):
        if not _sqlite_has_unique_index(connection, table, key):
            _raise_revision_43_sqlite_schema_error(table)

    required_indexes = {
        "idx_cayu_knowledge_evidence_entry_revision": (
            "cayu_knowledge_evidence",
            ("entry_id", "entry_revision", "id"),
        ),
        "idx_cayu_knowledge_evidence_source": (
            "cayu_knowledge_evidence",
            ("source_type", "source_id", "entry_id", "entry_revision"),
        ),
        "idx_cayu_knowledge_changes_entry_revision": (
            "cayu_knowledge_changes",
            ("entry_id", "entry_revision", "sequence"),
        ),
        "idx_cayu_knowledge_change_audiences_namespace": (
            "cayu_knowledge_change_audiences",
            ("namespace", "change_sequence", "audience_kind"),
        ),
        "idx_cayu_knowledge_change_audiences_status": (
            "cayu_knowledge_change_audiences",
            ("status", "change_sequence", "audience_kind"),
        ),
        "idx_cayu_knowledge_change_audiences_source": (
            "cayu_knowledge_change_audiences",
            ("source_type", "source_id", "change_sequence", "audience_kind"),
        ),
        "idx_cayu_knowledge_changes_operation": (
            "cayu_knowledge_changes",
            (("operation_id", "sequence") if relation_aware else ("operation_id",)),
        ),
        "idx_cayu_knowledge_change_consumers_lease": (
            "cayu_knowledge_change_consumers",
            ("lease_expires_at",),
        ),
        "idx_cayu_knowledge_change_labels_lookup": (
            "cayu_knowledge_change_labels",
            ("key", "value", "change_sequence", "audience_kind"),
        ),
        "idx_cayu_knowledge_chunks_identity_owner": (
            "cayu_knowledge_chunks",
            ("id", "entry_id", "entry_revision"),
        ),
    }
    for index, (table, columns) in required_indexes.items():
        row = connection.execute(
            "SELECT tbl_name, sql FROM sqlite_master WHERE type = 'index' AND name = ?",
            (index,),
        ).fetchone()
        actual_columns = tuple(
            str(column[2]) for column in connection.execute(f"PRAGMA index_info({index})")
        )
        if row is None or str(row[0]) != table or actual_columns != columns:
            _raise_revision_43_sqlite_schema_error(index)
        normalized = _normalize_sqlite_schema_sql(row[1])
        if index == "idx_cayu_knowledge_evidence_entry_revision" and (
            "id collate binary" not in normalized
        ):
            _raise_revision_43_sqlite_schema_error(index)
        if index == "idx_cayu_knowledge_changes_operation" and (
            "where operation_id is not null" not in normalized
        ):
            _raise_revision_43_sqlite_schema_error(index)
        if index == "idx_cayu_knowledge_change_consumers_lease" and (
            "where pending_change_sequence is not null" not in normalized
        ):
            _raise_revision_43_sqlite_schema_error(index)

    required_sql = {
        "cayu_knowledge_evidence": (
            "check (entry_revision > 0 and entry_revision <= 2147483647)",
            "check (role in ('origin', 'supporting'))",
            "check (source_id is not null or source_uri is not null)",
            "check (source_revision is not null or source_hash is not null)",
            "check (disposition in ('live', 'detached', 'retained'))",
        ),
        "cayu_knowledge_changes": (
            "check (sequence > 0 and sequence <= 9223372036854775807)",
            "check (entry_revision > 0 and entry_revision <= 2147483647)",
            (
                "kind in ( 'created', 'revision_appended', 'status_transitioned', "
                "'tombstoned', 'hard_deleted', 'expired', 'relation_published' )"
                if relation_aware
                else "kind in ( 'created', 'revision_appended', 'status_transitioned', "
                "'tombstoned', 'hard_deleted', 'expired' )"
            ),
        ),
        "cayu_knowledge_change_audiences": (
            (
                "check ( audience_kind in ( 'before', 'after', 'subject_exact', "
                "'subject_current', 'object_exact', 'object_current' ) )"
                if relation_aware
                else "check (audience_kind in ('before', 'after'))"
            ),
            "check ( requires_include_expired in (0, 1) )",
        ),
        "cayu_knowledge_change_consumers": (
            "check (cursor_sequence >= 0)",
            "check (pending_attempt >= 0)",
            "pending_change_sequence > cursor_sequence",
            "lease_expires_at > claimed_at",
        ),
        "cayu_knowledge_change_acknowledgements": (
            "length(claim_sha256) = 64",
            "claim_sha256 not glob '*[^0-9a-f]*'",
        ),
    }
    for table, fragments in required_sql.items():
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        normalized = _normalize_sqlite_schema_sql(None if row is None else row[0])
        if any(fragment not in normalized for fragment in fragments):
            _raise_revision_43_sqlite_schema_error(table)


def _normalize_sqlite_schema_sql(value: object | None) -> str:
    return " ".join(str(value or "").lower().split())


def _validate_revision_60_knowledge_schema(connection: sqlite3.Connection) -> None:
    expected_columns = {
        "cayu_knowledge_relations": (
            "id",
            "subject_entry_id",
            "subject_revision",
            "object_entry_id",
            "object_revision",
            "kind",
            "created_by_type",
            "created_by",
            "policy_id",
            "created_at",
            "metadata_json",
        ),
        "cayu_knowledge_relation_publication_receipts": (
            "operation_id",
            "relation_ids_json",
            "request_sha256",
            "committed_at",
            "access_snapshots_json",
        ),
    }
    for table, columns in expected_columns.items():
        if _sqlite_table_columns(connection, table) != columns:
            _raise_revision_60_sqlite_schema_error(table)

    relation_foreign_keys = tuple(
        sorted(
            (
                str(row[2]),
                str(row[3]),
                str(row[4]),
                str(row[6]).upper(),
            )
            for row in connection.execute("PRAGMA foreign_key_list(cayu_knowledge_relations)")
        )
    )
    expected_foreign_keys = tuple(
        sorted(
            (
                (
                    "cayu_knowledge_revisions",
                    "subject_entry_id",
                    "entry_id",
                    "CASCADE",
                ),
                (
                    "cayu_knowledge_revisions",
                    "subject_revision",
                    "revision",
                    "CASCADE",
                ),
                (
                    "cayu_knowledge_revisions",
                    "object_entry_id",
                    "entry_id",
                    "CASCADE",
                ),
                (
                    "cayu_knowledge_revisions",
                    "object_revision",
                    "revision",
                    "CASCADE",
                ),
            )
        )
    )
    if relation_foreign_keys != expected_foreign_keys:
        _raise_revision_60_sqlite_schema_error("cayu_knowledge_relations")
    if tuple(
        connection.execute("PRAGMA foreign_key_list(cayu_knowledge_relation_publication_receipts)")
    ):
        _raise_revision_60_sqlite_schema_error("cayu_knowledge_relation_publication_receipts")

    for table, key in (
        ("cayu_knowledge_relations", ("id",)),
        (
            "cayu_knowledge_relations",
            (
                "kind",
                "subject_entry_id",
                "subject_revision",
                "object_entry_id",
                "object_revision",
            ),
        ),
        ("cayu_knowledge_relation_publication_receipts", ("operation_id",)),
        ("cayu_knowledge_changes", ("relation_id",)),
    ):
        if not _sqlite_has_unique_index(connection, table, key):
            _raise_revision_60_sqlite_schema_error(table)

    required_indexes = {
        "idx_cayu_knowledge_relations_subject": (
            "cayu_knowledge_relations",
            ("subject_entry_id", "subject_revision", "created_at", "id"),
        ),
        "idx_cayu_knowledge_relations_object": (
            "cayu_knowledge_relations",
            ("object_entry_id", "object_revision", "created_at", "id"),
        ),
        "idx_cayu_knowledge_relations_subject_kind": (
            "cayu_knowledge_relations",
            ("subject_entry_id", "subject_revision", "kind", "created_at", "id"),
        ),
        "idx_cayu_knowledge_relations_object_kind": (
            "cayu_knowledge_relations",
            ("object_entry_id", "object_revision", "kind", "created_at", "id"),
        ),
        "idx_cayu_knowledge_changes_relation": (
            "cayu_knowledge_changes",
            ("relation_id",),
        ),
    }
    for index, (table, columns) in required_indexes.items():
        row = connection.execute(
            "SELECT tbl_name, sql FROM sqlite_master WHERE type = 'index' AND name = ?",
            (index,),
        ).fetchone()
        actual_columns = tuple(
            str(column[2]) for column in connection.execute(f"PRAGMA index_info({index})")
        )
        if row is None or str(row[0]) != table or actual_columns != columns:
            _raise_revision_60_sqlite_schema_error(index)
        normalized = _normalize_sqlite_schema_sql(row[1])
        if index.startswith("idx_cayu_knowledge_relations_") and (
            "id collate binary" not in normalized
        ):
            _raise_revision_60_sqlite_schema_error(index)
        if index == "idx_cayu_knowledge_changes_relation" and (
            "where relation_id is not null" not in normalized
        ):
            _raise_revision_60_sqlite_schema_error(index)

    required_sql = {
        "cayu_knowledge_relations": (
            "kind in ('supersedes', 'derived_from', 'contradicts')",
            "subject_entry_id <> object_entry_id",
            "kind <> 'contradicts' or subject_entry_id collate binary < object_entry_id collate binary",
            "json_valid(metadata_json)",
        ),
        "cayu_knowledge_relation_publication_receipts": (
            "json_valid(relation_ids_json)",
            "json_valid(access_snapshots_json)",
            "length(request_sha256) = 64",
        ),
        "cayu_knowledge_changes": (
            "kind = 'relation_published' and relation_id is not null",
            "kind <> 'relation_published' and relation_id is null",
        ),
    }
    for table, fragments in required_sql.items():
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        normalized = _normalize_sqlite_schema_sql(None if row is None else row[0])
        if any(fragment not in normalized for fragment in fragments):
            _raise_revision_60_sqlite_schema_error(table)


def _raise_revision_60_sqlite_schema_error(name: str) -> NoReturn:
    raise RuntimeError(
        "SQLite schema object "
        f"{name!r} conflicts with Cayu's revision-bound knowledge relation contract. "
        "Recreate the prerelease knowledge database with schema_mode=CREATE or MIGRATE."
    )


def _validate_revision_63_knowledge_schema(connection: sqlite3.Connection) -> None:
    table = "cayu_knowledge_maintenance_decisions"
    expected_columns = (
        ("operation_id", "TEXT", 0, 1),
        ("proposal_id", "TEXT", 1, 0),
        ("proposal_fingerprint", "TEXT", 1, 0),
        ("request_sha256", "TEXT", 1, 0),
        ("committed_at", "TEXT", 1, 0),
        ("proposal_json", "TEXT", 1, 0),
        ("decision_json", "TEXT", 1, 0),
        ("receipt_json", "TEXT", 1, 0),
        ("access_snapshot_json", "TEXT", 1, 0),
    )
    actual_columns = tuple(
        (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
        for row in connection.execute(f"PRAGMA table_info({table})")
    )
    if actual_columns != expected_columns:
        _raise_revision_63_sqlite_schema_error(table)
    if tuple(connection.execute(f"PRAGMA foreign_key_list({table})")):
        _raise_revision_63_sqlite_schema_error(table)
    for key in (("operation_id",), ("proposal_id",)):
        if not _sqlite_has_unique_index(connection, table, key):
            _raise_revision_63_sqlite_schema_error(table)
    row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    normalized = _normalize_sqlite_schema_sql(None if row is None else row[0])
    required = (
        "length(proposal_fingerprint) = 64",
        "proposal_fingerprint not glob '*[^0-9a-f]*'",
        "length(request_sha256) = 64",
        "request_sha256 not glob '*[^0-9a-f]*'",
        "json_valid(proposal_json)",
        "json_type(proposal_json) = 'object'",
        "json_valid(decision_json)",
        "json_type(decision_json) = 'object'",
        "json_valid(receipt_json)",
        "json_type(receipt_json) = 'object'",
        "json_valid(access_snapshot_json)",
        "json_type(access_snapshot_json) = 'object'",
    )
    if any(fragment not in normalized for fragment in required):
        _raise_revision_63_sqlite_schema_error(table)


def _raise_revision_63_sqlite_schema_error(name: str) -> NoReturn:
    raise RuntimeError(
        "SQLite schema object "
        f"{name!r} conflicts with Cayu's reviewed knowledge maintenance contract. "
        "Recreate the prerelease knowledge database with schema_mode=CREATE or MIGRATE."
    )


def _validate_revision_67_knowledge_schema(connection: sqlite3.Connection) -> None:
    table = "cayu_knowledge_maintenance_proposals"
    expected_columns = (
        ("operation_id", "TEXT", 0, 1),
        ("proposal_id", "TEXT", 1, 0),
        ("replacement_entry_id", "TEXT", 1, 0),
        ("replacement_revision", "INTEGER", 1, 0),
        ("proposal_fingerprint", "TEXT", 1, 0),
        ("accepted_plan_fingerprint", "TEXT", 1, 0),
        ("request_sha256", "TEXT", 1, 0),
        ("committed_at", "TEXT", 1, 0),
        ("proposal_json", "TEXT", 1, 0),
        ("accepted_plan_json", "TEXT", 1, 0),
        ("receipt_json", "TEXT", 1, 0),
        ("access_snapshot_json", "TEXT", 1, 0),
    )
    actual_columns = tuple(
        (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
        for row in connection.execute(f"PRAGMA table_info({table})")
    )
    if actual_columns != expected_columns:
        _raise_revision_67_sqlite_schema_error(table)
    if tuple(connection.execute(f"PRAGMA foreign_key_list({table})")):
        _raise_revision_67_sqlite_schema_error(table)
    for key in (("operation_id",), ("proposal_id",), ("replacement_entry_id",)):
        if not _sqlite_has_unique_index(connection, table, key):
            _raise_revision_67_sqlite_schema_error(table)
    row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    normalized = _normalize_sqlite_schema_sql(None if row is None else row[0])
    required = (
        "replacement_revision > 0",
        "replacement_revision <= 2147483647",
        "length(proposal_fingerprint) = 64",
        "proposal_fingerprint not glob '*[^0-9a-f]*'",
        "length(accepted_plan_fingerprint) = 64",
        "accepted_plan_fingerprint not glob '*[^0-9a-f]*'",
        "length(request_sha256) = 64",
        "request_sha256 not glob '*[^0-9a-f]*'",
        "json_valid(proposal_json)",
        "json_type(proposal_json) = 'object'",
        "json_valid(accepted_plan_json)",
        "json_type(accepted_plan_json) = 'object'",
        "json_valid(receipt_json)",
        "json_type(receipt_json) = 'object'",
        "json_valid(access_snapshot_json)",
        "json_type(access_snapshot_json) = 'object'",
    )
    if any(fragment not in normalized for fragment in required):
        _raise_revision_67_sqlite_schema_error(table)


def _raise_revision_67_sqlite_schema_error(name: str) -> NoReturn:
    raise RuntimeError(
        "SQLite schema object "
        f"{name!r} conflicts with Cayu's pending maintenance proposal contract. "
        "Recreate the prerelease knowledge database with schema_mode=CREATE or MIGRATE."
    )


def _validate_revision_69_work_context_schema(connection: sqlite3.Connection) -> None:
    expected_columns = {
        "cayu_agent_work_context_revisions": (
            ("task_id", "TEXT", 1, 1),
            ("revision", "INTEGER", 1, 2),
            ("content_sha256", "TEXT", 1, 0),
            ("operation_id", "TEXT", 1, 0),
            ("record_json", "TEXT", 1, 0),
            ("published_at", "TEXT", 1, 0),
        ),
        "cayu_agent_work_context_heads": (
            ("task_id", "TEXT", 1, 1),
            ("current_revision", "INTEGER", 1, 0),
        ),
        "cayu_agent_work_context_publications": (
            ("operation_id", "TEXT", 1, 1),
            ("task_id", "TEXT", 1, 0),
            ("request_sha256", "TEXT", 1, 0),
            ("context_revision", "INTEGER", 1, 0),
            ("changed", "INTEGER", 1, 0),
            ("receipt_json", "TEXT", 1, 0),
            ("committed_at", "TEXT", 1, 0),
        ),
        "cayu_agent_recall_checkpoints": (
            ("agent_id", "TEXT", 1, 1),
            ("task_id", "TEXT", 1, 2),
            ("knowledge_namespace", "TEXT", 1, 3),
            ("access_policy_sha256", "TEXT", 1, 4),
            ("checkpoint_stream_id", "TEXT", 1, 5),
            ("revision", "INTEGER", 1, 6),
            ("work_context_revision", "INTEGER", 1, 0),
            ("work_context_sha256", "TEXT", 1, 0),
            ("knowledge_sequence", "INTEGER", 1, 0),
            ("index_readiness_sequence", "INTEGER", 1, 0),
            ("knowledge_high_water_sequence", "INTEGER", 1, 0),
            ("index_readiness_high_water_sequence", "INTEGER", 1, 0),
            ("processing_mode", "TEXT", 1, 0),
            ("processing_id", "TEXT", 1, 0),
            ("operation_id", "TEXT", 1, 0),
            ("record_json", "TEXT", 1, 0),
            ("updated_at", "TEXT", 1, 0),
        ),
        "cayu_agent_recall_checkpoint_heads": (
            ("agent_id", "TEXT", 1, 1),
            ("task_id", "TEXT", 1, 2),
            ("knowledge_namespace", "TEXT", 1, 3),
            ("access_policy_sha256", "TEXT", 1, 4),
            ("checkpoint_stream_id", "TEXT", 1, 5),
            ("current_revision", "INTEGER", 1, 0),
        ),
    }
    for table, expected in expected_columns.items():
        actual = tuple(
            (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
            for row in connection.execute(f"PRAGMA table_info({table})")
        )
        if actual != expected:
            _raise_revision_69_sqlite_schema_error(table)

    unique_keys = (
        ("cayu_agent_work_context_revisions", ("task_id", "revision")),
        ("cayu_agent_work_context_revisions", ("operation_id",)),
        ("cayu_agent_work_context_heads", ("task_id",)),
        ("cayu_agent_work_context_publications", ("operation_id",)),
        (
            "cayu_agent_recall_checkpoints",
            (
                "agent_id",
                "task_id",
                "knowledge_namespace",
                "access_policy_sha256",
                "checkpoint_stream_id",
                "revision",
            ),
        ),
        ("cayu_agent_recall_checkpoints", ("operation_id",)),
        (
            "cayu_agent_recall_checkpoint_heads",
            (
                "agent_id",
                "task_id",
                "knowledge_namespace",
                "access_policy_sha256",
                "checkpoint_stream_id",
            ),
        ),
    )
    for table, columns in unique_keys:
        if not _sqlite_has_unique_index(connection, table, columns):
            _raise_revision_69_sqlite_schema_error(table)

    required_foreign_keys = {
        "cayu_agent_work_context_heads": {
            (
                "cayu_agent_work_context_revisions",
                ("task_id", "current_revision"),
                ("task_id", "revision"),
                "RESTRICT",
            ),
        },
        "cayu_agent_work_context_publications": {
            (
                "cayu_agent_work_context_revisions",
                ("task_id", "context_revision"),
                ("task_id", "revision"),
                "RESTRICT",
            ),
        },
        "cayu_agent_recall_checkpoints": {
            (
                "cayu_agent_work_context_revisions",
                ("task_id", "work_context_revision"),
                ("task_id", "revision"),
                "RESTRICT",
            ),
        },
        "cayu_agent_recall_checkpoint_heads": {
            (
                "cayu_agent_recall_checkpoints",
                (
                    "agent_id",
                    "task_id",
                    "knowledge_namespace",
                    "access_policy_sha256",
                    "checkpoint_stream_id",
                    "current_revision",
                ),
                (
                    "agent_id",
                    "task_id",
                    "knowledge_namespace",
                    "access_policy_sha256",
                    "checkpoint_stream_id",
                    "revision",
                ),
                "RESTRICT",
            ),
        },
    }
    for table, expected in required_foreign_keys.items():
        actual = _sqlite_foreign_key_groups(connection, table)
        if actual != expected:
            _raise_revision_69_sqlite_schema_error(table)

    required_sql = {
        "cayu_agent_work_context_revisions": (
            "task_id text collate binary not null",
            "revision integer not null check ( revision > 0 and revision <= 2147483647 )",
            "content_sha256 text collate binary not null",
            "operation_id text collate binary not null unique",
            "length(content_sha256) = 64",
            "content_sha256 not glob '*[^0-9a-f]*'",
            "json_valid(record_json)",
            "json_type(record_json) = 'object'",
        ),
        "cayu_agent_work_context_heads": (
            "task_id text collate binary not null primary key",
            "current_revision > 0",
            "current_revision <= 2147483647",
        ),
        "cayu_agent_work_context_publications": (
            "operation_id text collate binary not null primary key",
            "task_id text collate binary not null",
            "request_sha256 text collate binary not null",
            "length(request_sha256) = 64",
            "request_sha256 not glob '*[^0-9a-f]*'",
            "context_revision > 0",
            "context_revision <= 2147483647",
            "changed in (0, 1)",
            "json_valid(receipt_json)",
            "json_type(receipt_json) = 'object'",
        ),
        "cayu_agent_recall_checkpoints": (
            "agent_id text collate binary not null",
            "task_id text collate binary not null",
            "knowledge_namespace text collate binary not null",
            "access_policy_sha256 text collate binary not null",
            "checkpoint_stream_id text collate binary not null",
            "revision integer not null check ( revision > 0 and revision <= 2147483647 )",
            "work_context_sha256 text collate binary not null",
            "processing_mode text collate binary not null",
            "processing_id text collate binary not null",
            "operation_id text collate binary not null unique",
            "length(access_policy_sha256) = 64",
            "access_policy_sha256 not glob '*[^0-9a-f]*'",
            "work_context_revision > 0",
            "work_context_revision <= 2147483647",
            "length(work_context_sha256) = 64",
            "work_context_sha256 not glob '*[^0-9a-f]*'",
            "knowledge_sequence >= 0",
            "knowledge_sequence <= 9223372036854775807",
            "index_readiness_sequence >= 0",
            "index_readiness_sequence <= 9223372036854775807",
            "knowledge_high_water_sequence >= 0",
            "knowledge_high_water_sequence <= 9223372036854775807",
            "index_readiness_high_water_sequence >= 0",
            "index_readiness_high_water_sequence <= 9223372036854775807",
            "knowledge_sequence <= knowledge_high_water_sequence",
            "index_readiness_sequence <= index_readiness_high_water_sequence",
            "processing_mode in ('full_index', 'delta')",
            "json_valid(record_json)",
            "json_type(record_json) = 'object'",
        ),
        "cayu_agent_recall_checkpoint_heads": (
            "agent_id text collate binary not null",
            "task_id text collate binary not null",
            "knowledge_namespace text collate binary not null",
            "access_policy_sha256 text collate binary not null",
            "checkpoint_stream_id text collate binary not null",
            "current_revision > 0",
            "current_revision <= 2147483647",
        ),
    }
    for table, fragments in required_sql.items():
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        normalized = _normalize_sqlite_schema_sql(None if row is None else row[0])
        if any(fragment not in normalized for fragment in fragments):
            _raise_revision_69_sqlite_schema_error(table)


def _raise_revision_69_sqlite_schema_error(name: str) -> NoReturn:
    raise RuntimeError(
        "SQLite schema object "
        f"{name!r} conflicts with Cayu's agent work-context/checkpoint contract. "
        "Run schema_mode=MIGRATE to install the additive revision or recreate the database."
    )


def _validate_revision_71_recall_delivery_schema(
    connection: sqlite3.Connection,
    *,
    require_processing_schema_version: bool = False,
) -> None:
    delivery_columns = (
        ("delivery_id", "TEXT", 1, 1),
        ("operation_id", "TEXT", 1, 0),
        ("agent_id", "TEXT", 1, 0),
        ("task_id", "TEXT", 1, 0),
        ("knowledge_namespace", "TEXT", 1, 0),
        ("access_policy_sha256", "TEXT", 1, 0),
        ("checkpoint_stream_id", "TEXT", 1, 0),
        ("checkpoint_revision", "INTEGER", 1, 0),
        ("processing_result_sha256", "TEXT", 1, 0),
        ("delivery_json", "TEXT", 1, 0),
        ("staged_at", "TEXT", 1, 0),
    )
    delivery_columns_with_processing_schema = (
        *delivery_columns,
        ("processing_schema_version", "TEXT", 1, 0),
    )
    expected_columns = {
        "cayu_agent_recall_deliveries": (
            delivery_columns_with_processing_schema
            if require_processing_schema_version
            else delivery_columns
        ),
        "cayu_agent_recall_delivery_states": (
            ("delivery_id", "TEXT", 1, 1),
            ("agent_id", "TEXT", 1, 0),
            ("task_id", "TEXT", 1, 0),
            ("knowledge_namespace", "TEXT", 1, 0),
            ("access_policy_sha256", "TEXT", 1, 0),
            ("checkpoint_stream_id", "TEXT", 1, 0),
            ("checkpoint_revision", "INTEGER", 1, 0),
            ("state", "TEXT", 1, 0),
            ("attempt", "INTEGER", 1, 0),
            ("state_revision", "INTEGER", 1, 0),
            ("lease_expires_at", "TEXT", 0, 0),
            ("release_id", "TEXT", 0, 0),
            ("acknowledgement_id", "TEXT", 0, 0),
            ("state_json", "TEXT", 1, 0),
            ("updated_at", "TEXT", 1, 0),
        ),
        "cayu_agent_recall_delivery_claims": (
            ("claim_id", "TEXT", 1, 1),
            ("delivery_id", "TEXT", 1, 0),
            ("worker_id", "TEXT", 1, 0),
            ("request_sha256", "TEXT", 1, 0),
            ("attempt", "INTEGER", 1, 0),
            ("claimed_at", "TEXT", 1, 0),
        ),
        "cayu_agent_recall_delivery_releases": (
            ("release_id", "TEXT", 1, 1),
            ("delivery_id", "TEXT", 1, 0),
            ("claim_id", "TEXT", 1, 0),
            ("request_sha256", "TEXT", 1, 0),
            ("release_json", "TEXT", 1, 0),
            ("released_at", "TEXT", 1, 0),
        ),
    }
    validate_processing_schema_version = require_processing_schema_version
    for table, expected in expected_columns.items():
        actual = tuple(
            (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
            for row in connection.execute(f"PRAGMA table_info({table})")
        )
        if (
            table == "cayu_agent_recall_deliveries"
            and not require_processing_schema_version
            and actual == delivery_columns_with_processing_schema
        ):
            validate_processing_schema_version = True
            continue
        if actual != expected:
            _raise_revision_71_sqlite_schema_error(table)

    unique_keys = (
        ("cayu_agent_recall_deliveries", ("delivery_id",)),
        ("cayu_agent_recall_deliveries", ("operation_id",)),
        (
            "cayu_agent_recall_deliveries",
            (
                "agent_id",
                "task_id",
                "knowledge_namespace",
                "access_policy_sha256",
                "checkpoint_stream_id",
                "checkpoint_revision",
            ),
        ),
        ("cayu_agent_recall_delivery_states", ("delivery_id",)),
        ("cayu_agent_recall_delivery_states", ("release_id",)),
        ("cayu_agent_recall_delivery_states", ("acknowledgement_id",)),
        ("cayu_agent_recall_delivery_claims", ("claim_id",)),
        ("cayu_agent_recall_delivery_claims", ("delivery_id", "attempt")),
        ("cayu_agent_recall_delivery_releases", ("release_id",)),
    )
    for table, columns in unique_keys:
        if not _sqlite_has_unique_index(connection, table, columns):
            _raise_revision_71_sqlite_schema_error(table)

    required_foreign_keys = {
        "cayu_agent_recall_deliveries": {
            (
                "cayu_agent_recall_checkpoints",
                (
                    "agent_id",
                    "task_id",
                    "knowledge_namespace",
                    "access_policy_sha256",
                    "checkpoint_stream_id",
                    "checkpoint_revision",
                ),
                (
                    "agent_id",
                    "task_id",
                    "knowledge_namespace",
                    "access_policy_sha256",
                    "checkpoint_stream_id",
                    "revision",
                ),
                "RESTRICT",
            ),
            (
                "cayu_agent_recall_checkpoints",
                ("operation_id",),
                ("operation_id",),
                "RESTRICT",
            ),
        },
        "cayu_agent_recall_delivery_states": {
            (
                "cayu_agent_recall_deliveries",
                ("delivery_id",),
                ("delivery_id",),
                "RESTRICT",
            ),
            (
                "cayu_agent_recall_delivery_releases",
                ("release_id",),
                ("release_id",),
                "RESTRICT",
            ),
            (
                "cayu_agent_recall_deliveries",
                (
                    "agent_id",
                    "task_id",
                    "knowledge_namespace",
                    "access_policy_sha256",
                    "checkpoint_stream_id",
                    "checkpoint_revision",
                ),
                (
                    "agent_id",
                    "task_id",
                    "knowledge_namespace",
                    "access_policy_sha256",
                    "checkpoint_stream_id",
                    "checkpoint_revision",
                ),
                "RESTRICT",
            ),
        },
        "cayu_agent_recall_delivery_claims": {
            (
                "cayu_agent_recall_deliveries",
                ("delivery_id",),
                ("delivery_id",),
                "RESTRICT",
            ),
        },
        "cayu_agent_recall_delivery_releases": {
            (
                "cayu_agent_recall_deliveries",
                ("delivery_id",),
                ("delivery_id",),
                "RESTRICT",
            ),
            (
                "cayu_agent_recall_delivery_claims",
                ("claim_id",),
                ("claim_id",),
                "RESTRICT",
            ),
        },
    }
    for table, expected in required_foreign_keys.items():
        if _sqlite_foreign_key_groups(connection, table) != expected:
            _raise_revision_71_sqlite_schema_error(table)

    required_sql = {
        "cayu_agent_recall_deliveries": (
            "delivery_id text collate binary not null primary key",
            "operation_id text collate binary not null unique",
            "checkpoint_revision > 0",
            "checkpoint_stream_id text collate binary not null",
            "checkpoint_revision <= 2147483647",
            "length(access_policy_sha256) = 64",
            "access_policy_sha256 not glob '*[^0-9a-f]*'",
            "length(processing_result_sha256) = 64",
            "processing_result_sha256 not glob '*[^0-9a-f]*'",
            "json_valid(delivery_json)",
            "json_type(delivery_json) = 'object'",
        ),
        "cayu_agent_recall_delivery_states": (
            "delivery_id text collate binary not null primary key",
            "checkpoint_stream_id text collate binary not null",
            "state in ('pending', 'claimed', 'acknowledged')",
            "attempt >= 0",
            "attempt <= 9223372036854775807",
            "state_revision >= 0",
            "state_revision <= 9223372036854775807",
            "state = 'pending'",
            "state = 'claimed'",
            "state = 'acknowledged'",
            "json_valid(state_json)",
            "json_type(state_json) = 'object'",
        ),
        "cayu_agent_recall_delivery_claims": (
            "claim_id text collate binary not null primary key",
            "attempt > 0",
            "attempt <= 9223372036854775807",
            "length(request_sha256) = 64",
            "request_sha256 not glob '*[^0-9a-f]*'",
        ),
        "cayu_agent_recall_delivery_releases": (
            "release_id text collate binary not null primary key",
            "length(request_sha256) = 64",
            "request_sha256 not glob '*[^0-9a-f]*'",
            "json_valid(release_json)",
            "json_type(release_json) = 'object'",
        ),
    }
    if validate_processing_schema_version:
        required_sql["cayu_agent_recall_deliveries"] += (
            "processing_schema_version text collate binary not null",
            "processing_schema_version = 'cayu.agent_recall_processing.v3'",
        )
    for table, fragments in required_sql.items():
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        normalized = _normalize_sqlite_schema_sql(None if row is None else row[0])
        if any(fragment not in normalized for fragment in fragments):
            _raise_revision_71_sqlite_schema_error(table)

    index = "idx_cayu_agent_recall_delivery_pending"
    row = connection.execute(
        "SELECT tbl_name, sql FROM sqlite_master WHERE type = 'index' AND name = ?",
        (index,),
    ).fetchone()
    columns = tuple(
        str(index_row[2]) for index_row in connection.execute(f"PRAGMA index_info({index})")
    )
    if (
        row is None
        or row[0] != "cayu_agent_recall_delivery_states"
        or columns
        != (
            "agent_id",
            "task_id",
            "knowledge_namespace",
            "access_policy_sha256",
            "checkpoint_stream_id",
            "checkpoint_revision",
            "delivery_id",
        )
        or "where state != 'acknowledged'" not in _normalize_sqlite_schema_sql(row[1])
    ):
        _raise_revision_71_sqlite_schema_error(index)


def _raise_revision_71_sqlite_schema_error(name: str) -> NoReturn:
    raise RuntimeError(
        "SQLite schema object "
        f"{name!r} conflicts with Cayu's staged recall-delivery contract. "
        "Run schema_mode=MIGRATE to install the breaking revision or recreate the database."
    )


def _validate_revision_73_recall_subscription_schema(
    connection: sqlite3.Connection,
) -> None:
    expected_columns = {
        "cayu_agent_recall_subscription_revisions": (
            ("subscription_id", "TEXT", 1, 1),
            ("revision", "INTEGER", 1, 2),
            ("operation_id", "TEXT", 1, 0),
            ("agent_id", "TEXT", 1, 0),
            ("task_id", "TEXT", 1, 0),
            ("knowledge_namespace", "TEXT", 1, 0),
            ("access_policy_sha256", "TEXT", 1, 0),
            ("work_context_revision", "INTEGER", 1, 0),
            ("work_context_sha256", "TEXT", 1, 0),
            ("status", "TEXT", 1, 0),
            ("priority", "INTEGER", 1, 0),
            ("subscription_json", "TEXT", 1, 0),
            ("expires_at", "TEXT", 1, 0),
            ("published_at", "TEXT", 1, 0),
        ),
        "cayu_agent_recall_subscription_heads": (
            ("subscription_id", "TEXT", 1, 1),
            ("current_revision", "INTEGER", 1, 0),
        ),
        "cayu_agent_recall_subscription_publications": (
            ("operation_id", "TEXT", 1, 1),
            ("subscription_id", "TEXT", 1, 0),
            ("subscription_revision", "INTEGER", 1, 0),
            ("request_sha256", "TEXT", 1, 0),
            ("receipt_json", "TEXT", 1, 0),
            ("committed_at", "TEXT", 1, 0),
        ),
        "cayu_agent_recall_subscription_states": (
            ("subscription_id", "TEXT", 1, 1),
            ("current_revision", "INTEGER", 1, 0),
            ("agent_id", "TEXT", 1, 0),
            ("task_id", "TEXT", 1, 0),
            ("knowledge_namespace", "TEXT", 1, 0),
            ("access_policy_sha256", "TEXT", 1, 0),
            ("run_state", "TEXT", 1, 0),
            ("attempt", "INTEGER", 1, 0),
            ("state_revision", "INTEGER", 1, 0),
            ("lease_expires_at", "TEXT", 0, 0),
            ("release_id", "TEXT", 0, 0),
            ("next_evaluation_at", "TEXT", 1, 0),
            ("last_evaluation_id", "TEXT", 0, 0),
            ("state_json", "TEXT", 1, 0),
            ("updated_at", "TEXT", 1, 0),
        ),
        "cayu_agent_recall_subscription_claims": (
            ("claim_id", "TEXT", 1, 1),
            ("subscription_id", "TEXT", 1, 0),
            ("subscription_revision", "INTEGER", 1, 0),
            ("runner_id", "TEXT", 1, 0),
            ("request_sha256", "TEXT", 1, 0),
            ("attempt", "INTEGER", 1, 0),
            ("claimed_at", "TEXT", 1, 0),
        ),
        "cayu_agent_recall_subscription_releases": (
            ("release_id", "TEXT", 1, 1),
            ("subscription_id", "TEXT", 1, 0),
            ("claim_id", "TEXT", 1, 0),
            ("request_sha256", "TEXT", 1, 0),
            ("release_json", "TEXT", 1, 0),
            ("released_at", "TEXT", 1, 0),
        ),
        "cayu_agent_recall_subscription_evaluations": (
            ("evaluation_id", "TEXT", 1, 1),
            ("subscription_id", "TEXT", 1, 0),
            ("subscription_revision", "INTEGER", 1, 0),
            ("agent_id", "TEXT", 1, 0),
            ("task_id", "TEXT", 1, 0),
            ("knowledge_namespace", "TEXT", 1, 0),
            ("access_policy_sha256", "TEXT", 1, 0),
            ("claim_id", "TEXT", 1, 0),
            ("processing_operation_id", "TEXT", 1, 0),
            ("request_sha256", "TEXT", 1, 0),
            ("outcome", "TEXT", 1, 0),
            ("delivery_id", "TEXT", 0, 0),
            ("evaluation_json", "TEXT", 1, 0),
            ("committed_at", "TEXT", 1, 0),
        ),
        "cayu_agent_recall_subscription_wake_claims": (
            ("claim_id", "TEXT", 1, 1),
            ("wake_id", "TEXT", 1, 0),
            ("delivery_id", "TEXT", 1, 0),
            ("runner_id", "TEXT", 1, 0),
            ("request_sha256", "TEXT", 1, 0),
            ("attempt", "INTEGER", 1, 0),
            ("claimed_at", "TEXT", 1, 0),
        ),
        "cayu_agent_recall_subscription_wake_releases": (
            ("release_id", "TEXT", 1, 1),
            ("wake_id", "TEXT", 1, 0),
            ("claim_id", "TEXT", 1, 0),
            ("request_sha256", "TEXT", 1, 0),
            ("release_json", "TEXT", 1, 0),
            ("released_at", "TEXT", 1, 0),
        ),
        "cayu_agent_recall_subscription_wake_states": (
            ("wake_id", "TEXT", 1, 1),
            ("delivery_id", "TEXT", 1, 0),
            ("agent_id", "TEXT", 1, 0),
            ("task_id", "TEXT", 1, 0),
            ("knowledge_namespace", "TEXT", 1, 0),
            ("access_policy_sha256", "TEXT", 1, 0),
            ("state", "TEXT", 1, 0),
            ("attempt", "INTEGER", 1, 0),
            ("state_revision", "INTEGER", 1, 0),
            ("claim_id", "TEXT", 0, 0),
            ("lease_expires_at", "TEXT", 0, 0),
            ("release_id", "TEXT", 0, 0),
            ("acknowledgement_id", "TEXT", 0, 0),
            ("state_json", "TEXT", 1, 0),
            ("committed_at", "TEXT", 1, 0),
            ("updated_at", "TEXT", 1, 0),
        ),
    }
    for table, expected in expected_columns.items():
        actual = tuple(
            (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
            for row in connection.execute(f"PRAGMA table_info({table})")
        )
        if actual != expected:
            _raise_revision_73_sqlite_schema_error(table)

    unique_keys = (
        ("cayu_agent_recall_subscription_revisions", ("subscription_id", "revision")),
        ("cayu_agent_recall_subscription_revisions", ("operation_id",)),
        ("cayu_agent_recall_subscription_heads", ("subscription_id",)),
        ("cayu_agent_recall_subscription_publications", ("operation_id",)),
        ("cayu_agent_recall_subscription_states", ("subscription_id",)),
        ("cayu_agent_recall_subscription_states", ("release_id",)),
        ("cayu_agent_recall_subscription_claims", ("claim_id",)),
        ("cayu_agent_recall_subscription_claims", ("subscription_id", "attempt")),
        ("cayu_agent_recall_subscription_releases", ("release_id",)),
        ("cayu_agent_recall_subscription_evaluations", ("evaluation_id",)),
        ("cayu_agent_recall_subscription_evaluations", ("claim_id",)),
        (
            "cayu_agent_recall_subscription_evaluations",
            ("processing_operation_id",),
        ),
        ("cayu_agent_recall_subscription_evaluations", ("delivery_id",)),
        ("cayu_agent_recall_subscription_wake_claims", ("claim_id",)),
        ("cayu_agent_recall_subscription_wake_claims", ("wake_id", "attempt")),
        ("cayu_agent_recall_subscription_wake_releases", ("release_id",)),
        ("cayu_agent_recall_subscription_wake_states", ("wake_id",)),
        ("cayu_agent_recall_subscription_wake_states", ("delivery_id",)),
        ("cayu_agent_recall_subscription_wake_states", ("release_id",)),
        ("cayu_agent_recall_subscription_wake_states", ("acknowledgement_id",)),
    )
    for table, columns in unique_keys:
        if not _sqlite_has_unique_index(connection, table, columns):
            _raise_revision_73_sqlite_schema_error(table)

    required_foreign_keys = {
        "cayu_agent_recall_subscription_revisions": {
            (
                "cayu_agent_work_context_revisions",
                ("task_id", "work_context_revision"),
                ("task_id", "revision"),
                "RESTRICT",
            ),
        },
        "cayu_agent_recall_subscription_heads": {
            (
                "cayu_agent_recall_subscription_revisions",
                ("subscription_id", "current_revision"),
                ("subscription_id", "revision"),
                "RESTRICT",
            ),
        },
        "cayu_agent_recall_subscription_publications": {
            (
                "cayu_agent_recall_subscription_revisions",
                ("subscription_id", "subscription_revision"),
                ("subscription_id", "revision"),
                "RESTRICT",
            ),
        },
        "cayu_agent_recall_subscription_states": {
            (
                "cayu_agent_recall_subscription_revisions",
                ("subscription_id", "current_revision"),
                ("subscription_id", "revision"),
                "RESTRICT",
            ),
            (
                "cayu_agent_recall_subscription_releases",
                ("release_id",),
                ("release_id",),
                "RESTRICT",
            ),
        },
        "cayu_agent_recall_subscription_claims": {
            (
                "cayu_agent_recall_subscription_revisions",
                ("subscription_id", "subscription_revision"),
                ("subscription_id", "revision"),
                "RESTRICT",
            ),
        },
        "cayu_agent_recall_subscription_releases": {
            (
                "cayu_agent_recall_subscription_heads",
                ("subscription_id",),
                ("subscription_id",),
                "RESTRICT",
            ),
            (
                "cayu_agent_recall_subscription_claims",
                ("claim_id",),
                ("claim_id",),
                "RESTRICT",
            ),
        },
        "cayu_agent_recall_subscription_evaluations": {
            (
                "cayu_agent_recall_subscription_revisions",
                ("subscription_id", "subscription_revision"),
                ("subscription_id", "revision"),
                "RESTRICT",
            ),
            (
                "cayu_agent_recall_subscription_claims",
                ("claim_id",),
                ("claim_id",),
                "RESTRICT",
            ),
            (
                "cayu_agent_recall_deliveries",
                ("delivery_id",),
                ("delivery_id",),
                "RESTRICT",
            ),
        },
        "cayu_agent_recall_subscription_wake_claims": {
            (
                "cayu_agent_recall_subscription_evaluations",
                ("wake_id",),
                ("evaluation_id",),
                "RESTRICT",
            ),
            (
                "cayu_agent_recall_deliveries",
                ("delivery_id",),
                ("delivery_id",),
                "RESTRICT",
            ),
        },
        "cayu_agent_recall_subscription_wake_releases": {
            (
                "cayu_agent_recall_subscription_evaluations",
                ("wake_id",),
                ("evaluation_id",),
                "RESTRICT",
            ),
            (
                "cayu_agent_recall_subscription_wake_claims",
                ("claim_id",),
                ("claim_id",),
                "RESTRICT",
            ),
        },
        "cayu_agent_recall_subscription_wake_states": {
            (
                "cayu_agent_recall_subscription_evaluations",
                ("wake_id",),
                ("evaluation_id",),
                "RESTRICT",
            ),
            (
                "cayu_agent_recall_deliveries",
                ("delivery_id",),
                ("delivery_id",),
                "RESTRICT",
            ),
            (
                "cayu_agent_recall_subscription_wake_claims",
                ("claim_id",),
                ("claim_id",),
                "RESTRICT",
            ),
            (
                "cayu_agent_recall_subscription_wake_releases",
                ("release_id",),
                ("release_id",),
                "RESTRICT",
            ),
        },
    }
    for table, expected in required_foreign_keys.items():
        if _sqlite_foreign_key_groups(connection, table) != expected:
            _raise_revision_73_sqlite_schema_error(table)

    required_sql = {
        "cayu_agent_recall_subscription_revisions": (
            "status in ('active', 'paused', 'cancelled')",
            "priority >= 0",
            "priority <= 1000",
            "json_valid(subscription_json)",
            "json_type(subscription_json) = 'object'",
            "length(access_policy_sha256) = 64",
            "length(work_context_sha256) = 64",
        ),
        "cayu_agent_recall_subscription_states": (
            "run_state in ('due', 'claimed')",
            "run_state = 'due' and lease_expires_at is null",
            "run_state = 'claimed' and lease_expires_at is not null",
            "json_valid(state_json)",
        ),
        "cayu_agent_recall_subscription_evaluations": (
            "outcome in ('no_work', 'silent', 'wake')",
            "outcome = 'wake' and delivery_id is not null",
            "outcome != 'wake' and delivery_id is null",
            "json_valid(evaluation_json)",
        ),
        "cayu_agent_recall_subscription_wake_states": (
            "state in ('pending', 'claimed', 'acknowledged')",
            "state = 'claimed' and attempt > 0",
            "state = 'acknowledged' and attempt > 0",
            "json_valid(state_json)",
        ),
    }
    for table, fragments in required_sql.items():
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        normalized = _normalize_sqlite_schema_sql(None if row is None else row[0])
        if any(fragment not in normalized for fragment in fragments):
            _raise_revision_73_sqlite_schema_error(table)

    expected_indexes = {
        "idx_cayu_agent_recall_subscription_due": (
            "cayu_agent_recall_subscription_states",
            (
                "agent_id",
                "task_id",
                "knowledge_namespace",
                "access_policy_sha256",
                "next_evaluation_at",
                "subscription_id",
            ),
            None,
        ),
        "idx_cayu_agent_recall_subscription_wakes": (
            "cayu_agent_recall_subscription_wake_states",
            (
                "agent_id",
                "task_id",
                "knowledge_namespace",
                "access_policy_sha256",
                "committed_at",
                "wake_id",
            ),
            "where state != 'acknowledged'",
        ),
        "idx_cayu_agent_recall_subscription_evaluations": (
            "cayu_agent_recall_subscription_evaluations",
            ("subscription_id", "evaluation_id"),
            None,
        ),
    }
    for index, (table, expected, predicate) in expected_indexes.items():
        row = connection.execute(
            "SELECT tbl_name, sql FROM sqlite_master WHERE type = 'index' AND name = ?",
            (index,),
        ).fetchone()
        columns = tuple(
            str(index_row[2]) for index_row in connection.execute(f"PRAGMA index_info({index})")
        )
        normalized = "" if row is None else _normalize_sqlite_schema_sql(row[1])
        if (
            row is None
            or row[0] != table
            or columns != expected
            or (predicate is not None and predicate not in normalized)
        ):
            _raise_revision_73_sqlite_schema_error(index)


def _raise_revision_73_sqlite_schema_error(name: str) -> NoReturn:
    raise RuntimeError(
        "SQLite schema object "
        f"{name!r} conflicts with Cayu's idle recall-subscription contract. "
        "Run schema_mode=MIGRATE to install revision 73 or recreate the database."
    )


def _validate_revision_44_knowledge_schema(connection: sqlite3.Connection) -> None:
    expected_columns = {
        "cayu_knowledge_index_readiness_events": (
            "sequence",
            "identity_sha256",
            "entry_id",
            "entry_revision",
            "chunk_id",
            "projection_type",
            "projection_content_hash",
            "embedding_model",
            "dimensions",
            "preprocessing_version",
            "generator",
            "generator_version",
            "index_representation_version",
            "state",
            "attempt_id",
            "failure_code",
            "operation_id",
            "update_sha256",
            "published_at",
        ),
        "cayu_knowledge_index_readiness_current": (
            "identity_sha256",
            "sequence",
        ),
    }
    for table, columns in expected_columns.items():
        if _sqlite_table_columns(connection, table) != columns:
            _raise_revision_44_sqlite_schema_error(table)

    foreign_keys = tuple(
        (
            str(row[2]),
            str(row[3]),
            str(row[4]),
            str(row[6]).upper(),
        )
        for row in connection.execute(
            "PRAGMA foreign_key_list(cayu_knowledge_index_readiness_current)"
        )
    )
    if foreign_keys != (
        (
            "cayu_knowledge_index_readiness_events",
            "identity_sha256",
            "identity_sha256",
            "CASCADE",
        ),
        ("cayu_knowledge_index_readiness_events", "sequence", "sequence", "CASCADE"),
    ):
        _raise_revision_44_sqlite_schema_error("cayu_knowledge_index_readiness_current")

    for table, key in (
        ("cayu_knowledge_index_readiness_events", ("sequence",)),
        ("cayu_knowledge_index_readiness_events", ("operation_id",)),
        ("cayu_knowledge_index_readiness_events", ("identity_sha256", "sequence")),
        ("cayu_knowledge_index_readiness_current", ("identity_sha256",)),
        ("cayu_knowledge_index_readiness_current", ("sequence",)),
    ):
        if not _sqlite_has_unique_index(connection, table, key):
            _raise_revision_44_sqlite_schema_error(table)

    required_indexes = {
        "idx_cayu_knowledge_index_readiness_identity_sequence": (
            "cayu_knowledge_index_readiness_events",
            ("identity_sha256", "sequence"),
        ),
        "idx_cayu_knowledge_index_readiness_entry_revision": (
            "cayu_knowledge_index_readiness_events",
            ("entry_id", "entry_revision", "projection_type", "sequence"),
        ),
        "idx_cayu_knowledge_index_readiness_projection_lookup": (
            "cayu_knowledge_index_readiness_events",
            (
                "entry_id",
                "entry_revision",
                "chunk_id",
                "projection_type",
                "embedding_model",
                "dimensions",
                "sequence",
            ),
        ),
    }
    for index, (table, columns) in required_indexes.items():
        row = connection.execute(
            "SELECT tbl_name FROM sqlite_master WHERE type = 'index' AND name = ?",
            (index,),
        ).fetchone()
        actual_columns = tuple(
            str(column[2]) for column in connection.execute(f"PRAGMA index_info({index})")
        )
        if row is None or str(row[0]) != table or actual_columns != columns:
            _raise_revision_44_sqlite_schema_error(index)

    row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' "
        "AND name = 'cayu_knowledge_index_readiness_events'"
    ).fetchone()
    normalized = _normalize_sqlite_schema_sql(None if row is None else row[0])
    required_fragments = (
        "check (sequence > 0 and sequence <= 9223372036854775807)",
        "length(identity_sha256) = 64",
        "identity_sha256 not glob '*[^0-9a-f]*'",
        "check (entry_revision > 0 and entry_revision <= 2147483647)",
        "check (dimensions > 0)",
        "check (state in ('pending', 'ready', 'failed'))",
        "length(update_sha256) = 64",
        "update_sha256 not glob '*[^0-9a-f]*'",
        "state = 'failed' and failure_code is not null",
        "state <> 'failed' and failure_code is null",
    )
    if any(fragment not in normalized for fragment in required_fragments):
        _raise_revision_44_sqlite_schema_error("cayu_knowledge_index_readiness_events")


def _raise_revision_42_sqlite_schema_error(name: str) -> NoReturn:
    raise RuntimeError(
        f"SQLite schema object {name!r} conflicts with Cayu's revision-first "
        "knowledge contract. Recreate the Cayu database from a known-good "
        "revision-42 schema."
    )


def _raise_revision_43_sqlite_schema_error(name: str) -> NoReturn:
    raise RuntimeError(
        f"SQLite schema object {name!r} conflicts with Cayu's knowledge evidence "
        "and atomic change contract. Recreate or migrate the Cayu database from "
        "a known-good revision-43 schema."
    )


def _raise_revision_44_sqlite_schema_error(name: str) -> NoReturn:
    raise RuntimeError(
        f"SQLite schema object {name!r} conflicts with Cayu's derived-index "
        "identity and readiness contract. Recreate or migrate the Cayu database "
        "from a known-good revision-44 schema."
    )


def _reject_revision_43_knowledge_identity_overflow(
    connection: sqlite3.Connection,
) -> None:
    row = connection.execute(
        """
        SELECT 1
        FROM cayu_knowledge_entries
        WHERE length(CAST(id AS BLOB)) > ?
        UNION ALL
        SELECT 1
        FROM cayu_knowledge_chunks
        WHERE length(CAST(id AS BLOB)) > ?
        LIMIT 1
        """,
        (MAX_KNOWLEDGE_ENTRY_ID_BYTES, MAX_KNOWLEDGE_CHUNK_ID_BYTES),
    ).fetchone()
    if row is not None:
        raise schema.SchemaTooOld(
            "Storage revision 43 bounds knowledge entry and chunk identities for "
            "portable indexed storage. Shorten out-of-contract revision-42 identities "
            "or recreate the Cayu database before migration."
        )


def _validate_knowledge_publication_access_snapshot_column(
    connection: sqlite3.Connection,
) -> None:
    columns = {
        str(row[1]): (str(row[2]).upper(), int(row[3]))
        for row in connection.execute("PRAGMA table_info(cayu_knowledge_publication_receipts)")
    }
    if columns.get("access_snapshot_json") != ("TEXT", 1):
        raise RuntimeError(
            "SQLite schema object "
            "'cayu_knowledge_publication_receipts.access_snapshot_json' conflicts "
            "with Cayu's knowledge authorization contract. Recreate the Cayu "
            "database from a known-good revision-41 schema."
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


def _validate_interrupted_task_handoff_schema(connection: sqlite3.Connection) -> None:
    columns = tuple(
        (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
        for row in connection.execute("PRAGMA table_info(cayu_task_interrupted_handoff_receipts)")
    )
    expected = (
        ("task_id", "TEXT", 1, 1),
        ("handoff_id", "TEXT", 1, 2),
        ("request_sha256", "TEXT", 1, 0),
        ("request_json", "TEXT", 1, 0),
        ("task_json", "TEXT", 1, 0),
        ("committed_at", "TEXT", 1, 0),
    )
    index_columns = tuple(
        str(row[2])
        for row in connection.execute(
            "PRAGMA index_info(idx_cayu_tasks_interrupted_handoff_recovery)"
        )
    )
    if columns != expected or index_columns != (
        "status",
        "lease_expires_at",
        "id",
    ):
        raise RuntimeError(
            "SQLite interrupted-task handoff storage conflicts with Cayu's "
            "revision-70 durability contract. Run `cayu storage migrate` or restore "
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


def _validate_task_retry_series_schema(connection: sqlite3.Connection) -> None:
    task_columns = {
        str(row[1]): (str(row[2]).upper(), int(row[3]))
        for row in connection.execute("PRAGMA table_info(cayu_tasks)")
    }
    receipt_columns = tuple(
        (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
        for row in connection.execute("PRAGMA table_info(cayu_task_retry_settlements)")
    )
    expected_receipt_columns = (
        ("task_id", "TEXT", 1, 1),
        ("idempotency_key", "TEXT", 1, 2),
        ("request_sha256", "TEXT", 1, 0),
        ("receipt_json", "TEXT", 1, 0),
        ("committed_at", "TEXT", 1, 0),
    )
    if (
        task_columns.get("retry_series_json") != ("TEXT", 0)
        or receipt_columns != expected_receipt_columns
    ):
        raise RuntimeError(
            "SQLite task retry-series schema conflicts with Cayu's revision-45 "
            "durability contract. Run `cayu storage migrate` or restore the "
            "database from a known-good backup."
        )


def _validate_task_retry_reconciliation_schema(connection: sqlite3.Connection) -> None:
    rejection_columns = tuple(
        (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
        for row in connection.execute(
            "PRAGMA table_info(cayu_task_retry_reconciliation_rejections)"
        )
    )
    if rejection_columns != (
        ("task_id", "TEXT", 1, 1),
        ("reconciliation_idempotency_key", "TEXT", 1, 2),
        ("request_sha256", "TEXT", 1, 0),
        ("record_json", "TEXT", 1, 0),
        ("recorded_at", "TEXT", 1, 0),
    ):
        raise RuntimeError(
            "SQLite task retry-reconciliation schema conflicts with Cayu's "
            "revision-55 durability contract. Run `cayu storage migrate` or "
            "restore the database from a known-good backup."
        )


def _validate_work_attempt_admission_schema(connection: sqlite3.Connection) -> None:
    admission_columns = tuple(
        (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
        for row in connection.execute("PRAGMA table_info(cayu_work_attempt_admissions)")
    )
    claim_columns = tuple(
        (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
        for row in connection.execute("PRAGMA table_info(cayu_work_attempt_execution_claims)")
    )
    expected_admission_columns = (
        ("admission_id", "TEXT", 0, 1),
        ("attempt_id", "TEXT", 1, 0),
        ("task_id", "TEXT", 1, 0),
        ("session_id", "TEXT", 1, 0),
        ("interaction_id", "TEXT", 1, 0),
        ("state", "TEXT", 1, 0),
        ("prepare_request_sha256", "TEXT", 1, 0),
        ("current_claim_id", "TEXT", 1, 0),
        ("current_generation", "INTEGER", 1, 0),
        ("lease_expires_at", "TEXT", 1, 0),
        ("admission_json", "TEXT", 1, 0),
    )
    expected_claim_columns = (
        ("claim_id", "TEXT", 0, 1),
        ("admission_id", "TEXT", 1, 0),
        ("generation", "INTEGER", 1, 0),
        ("request_sha256", "TEXT", 1, 0),
        ("lease_expires_at", "TEXT", 1, 0),
        ("is_current", "INTEGER", 1, 0),
        ("claim_json", "TEXT", 1, 0),
    )
    named_index_properties: dict[str, tuple[int, int]] = {}
    for table_name in (
        "cayu_work_attempt_admissions",
        "cayu_work_attempt_execution_claims",
    ):
        for row in connection.execute(f"PRAGMA index_list({table_name})"):
            index_name = str(row[1])
            if index_name.startswith("sqlite_autoindex_"):
                continue
            named_index_properties[index_name] = (int(row[2]), int(row[4]))
    named_indexes = {
        "idx_cayu_work_attempt_admission_interaction": (
            *named_index_properties.get("idx_cayu_work_attempt_admission_interaction", (-1, -1)),
            tuple(
                str(column[2])
                for column in connection.execute(
                    "PRAGMA index_info(idx_cayu_work_attempt_admission_interaction)"
                )
            ),
        ),
        "idx_cayu_work_attempt_admission_task": (
            *named_index_properties.get("idx_cayu_work_attempt_admission_task", (-1, -1)),
            tuple(
                str(column[2])
                for column in connection.execute(
                    "PRAGMA index_info(idx_cayu_work_attempt_admission_task)"
                )
            ),
        ),
        "idx_cayu_work_attempt_admission_session_current": (
            *named_index_properties.get(
                "idx_cayu_work_attempt_admission_session_current",
                (-1, -1),
            ),
            tuple(
                str(column[2])
                for column in connection.execute(
                    "PRAGMA index_info(idx_cayu_work_attempt_admission_session_current)"
                )
            ),
        ),
        "idx_cayu_work_attempt_claim_current": (
            *named_index_properties.get("idx_cayu_work_attempt_claim_current", (-1, -1)),
            tuple(
                str(column[2])
                for column in connection.execute(
                    "PRAGMA index_info(idx_cayu_work_attempt_claim_current)"
                )
            ),
        ),
    }
    table_sql = {
        str(row[0]): " ".join(str(row[1]).lower().split())
        for row in connection.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'table' AND name IN (?, ?)",
            (
                "cayu_work_attempt_admissions",
                "cayu_work_attempt_execution_claims",
            ),
        )
    }
    session_index_row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'index' "
        "AND name = 'idx_cayu_work_attempt_admission_session_current'"
    ).fetchone()
    session_index_sql = (
        "" if session_index_row is None else " ".join(str(session_index_row[0]).lower().split())
    )
    admission_sql = table_sql.get("cayu_work_attempt_admissions", "")
    claim_sql = table_sql.get("cayu_work_attempt_execution_claims", "")
    if (
        admission_columns != expected_admission_columns
        or claim_columns != expected_claim_columns
        or named_indexes.get("idx_cayu_work_attempt_admission_interaction")
        != (1, 0, ("session_id", "interaction_id"))
        or named_indexes.get("idx_cayu_work_attempt_admission_task")
        != (0, 0, ("task_id", "current_generation"))
        or named_indexes.get("idx_cayu_work_attempt_admission_session_current")
        != (1, 1, ("session_id",))
        or "where state != 'released'" not in session_index_sql
        or named_indexes.get("idx_cayu_work_attempt_claim_current") != (1, 1, ("admission_id",))
        or "references cayu_tasks(id) on delete restrict" not in admission_sql
        or any(
            state not in admission_sql
            for state in ("'preparing'", "'active'", "'recovering'", "'released'")
        )
        or "'draining'" in admission_sql
        or "unique (admission_id, generation)" not in claim_sql
        or "references cayu_work_attempt_admissions(admission_id) on delete restrict"
        not in claim_sql
    ):
        raise RuntimeError(
            "SQLite work-attempt admission schema conflicts with Cayu's revision-61 "
            "durability contract. Run `cayu storage migrate` or restore the "
            "database from a known-good backup."
        )


def _validate_revision_46_transcript_search_schema(
    connection: sqlite3.Connection,
) -> None:
    transcript_columns = {
        str(row[1]): (str(row[2]).upper(), int(row[3]))
        for row in connection.execute("PRAGMA table_info(cayu_transcript_messages)")
    }
    if transcript_columns.get("transcript_search_document") != ("TEXT", 1):
        raise RuntimeError(
            "SQLite transcript search document column is missing or nullable. "
            "Recreate or restore a known-good revision-46 Cayu database."
        )
    configuration_columns = tuple(
        (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
        for row in connection.execute("PRAGMA table_info(cayu_transcript_search_configuration)")
    )
    if configuration_columns != (
        ("singleton", "INTEGER", 0, 1),
        ("tokenizer_version", "TEXT", 1, 0),
    ):
        raise RuntimeError(
            "SQLite transcript search tokenizer configuration is missing or malformed. "
            "Recreate a revision-46 Cayu database with this runtime."
        )
    configuration_table = connection.execute(
        "SELECT sql FROM sqlite_master "
        "WHERE type = 'table' AND name = 'cayu_transcript_search_configuration'"
    ).fetchone()
    configuration_sql = (
        ""
        if configuration_table is None
        else "".join(str(configuration_table[0] or "").lower().split())
    )
    if "check(singleton=1)" not in configuration_sql:
        raise RuntimeError(
            "SQLite transcript search tokenizer configuration lacks its singleton "
            "constraint. Recreate a revision-46 Cayu database."
        )
    configuration = connection.execute(
        "SELECT singleton, tokenizer_version "
        "FROM cayu_transcript_search_configuration ORDER BY singleton"
    ).fetchall()
    if len(configuration) != 1 or tuple(configuration[0]) != (
        1,
        TRANSCRIPT_SEARCH_TOKENIZER_VERSION,
    ):
        raise RuntimeError(
            "SQLite transcript search tokenizer identity conflicts with this runtime. "
            "Recreate a revision-46 Cayu database with this runtime."
        )
    expected = {
        "cayu_transcript_messages_fts": "table",
        "cayu_transcript_messages_fts_insert": "trigger",
        "cayu_transcript_messages_fts_delete": "trigger",
        "cayu_transcript_messages_fts_update": "trigger",
        "cayu_transcript_messages_search_document_insert": "trigger",
        "cayu_transcript_messages_search_document_update": "trigger",
    }
    rows = connection.execute(
        "SELECT name, type, sql FROM sqlite_master WHERE name IN (?, ?, ?, ?, ?, ?)",
        tuple(expected),
    ).fetchall()
    found = {str(row[0]): (str(row[1]), str(row[2] or "")) for row in rows}
    if set(found) != set(expected) or any(
        found[name][0] != object_type for name, object_type in expected.items()
    ):
        raise RuntimeError(
            "SQLite transcript search schema is incomplete. Recreate or restore "
            "a known-good revision-46 Cayu database."
        )
    fts_sql = "".join(found["cayu_transcript_messages_fts"][1].lower().split())
    if "usingfts5(session_token,message_text,content='')" not in fts_sql:
        raise RuntimeError(
            "SQLite transcript search index conflicts with Cayu's contentless FTS contract."
        )
    insert_sql = "".join(found["cayu_transcript_messages_fts_insert"][1].lower().split())
    delete_sql = "".join(found["cayu_transcript_messages_fts_delete"][1].lower().split())
    update_sql = "".join(found["cayu_transcript_messages_fts_update"][1].lower().split())
    document_insert_sql = "".join(
        found["cayu_transcript_messages_search_document_insert"][1].lower().split()
    )
    document_update_sql = "".join(
        found["cayu_transcript_messages_search_document_update"][1].lower().split()
    )
    if (
        not all(
            fragment in insert_sql
            for fragment in (
                "afterinsertoncayu_transcript_messages",
                "whennew.rolein('user','assistant')",
                "cayu_transcript_session_token(new.session_id)",
                "new.transcript_search_document",
            )
        )
        or not all(
            fragment in delete_sql
            for fragment in (
                "afterdeleteoncayu_transcript_messages",
                "whenold.rolein('user','assistant')",
                "'delete',old.sequence",
                "cayu_transcript_session_token(old.session_id)",
                "old.transcript_search_document",
            )
        )
        or not all(
            fragment in update_sql
            for fragment in (
                "afterupdateofsession_id,role,message_json",
                "'delete',old.sequence",
                "cayu_transcript_search_document(new.message_json)",
                "wherenew.rolein('user','assistant')",
            )
        )
        or not all(
            fragment in document_insert_sql
            for fragment in (
                "beforeinsertoncayu_transcript_messages",
                "new.transcript_search_documentisnullor",
                "cayu_transcript_search_document(new.message_json)",
                "raise(abort,'invalidtranscriptsearchdocument')",
            )
        )
        or not all(
            fragment in document_update_sql
            for fragment in (
                "beforeupdateoftranscript_search_document",
                "new.transcript_search_documentisnullor",
                "cayu_transcript_search_document(new.message_json)",
                "raise(abort,'invalidtranscriptsearchdocument')",
            )
        )
    ):
        raise RuntimeError(
            "SQLite transcript search maintenance triggers conflict with Cayu's contract."
        )
    fixture = json.dumps(
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "visible"},
                {"type": "thinking", "text": "hidden"},
            ],
        }
    )
    projected = connection.execute(
        "SELECT cayu_transcript_search_document(?)",
        (fixture,),
    ).fetchone()
    if projected is None or projected[0] != "x76697369626c65":
        raise RuntimeError(
            "SQLite transcript search projection does not preserve the narrative-only boundary."
        )


def _validate_eval_result_baseline_schema(connection: sqlite3.Connection) -> None:
    expected_columns = {
        "cayu_eval_result_records": (
            ("revision", "TEXT", 0, 1),
            ("origin", "TEXT", 1, 0),
            ("target_key", "TEXT", 1, 0),
            ("corpus_revision", "TEXT", 1, 0),
            ("suite_id", "TEXT", 1, 0),
            ("suite_revision", "TEXT", 1, 0),
            ("application_release_id", "TEXT", 1, 0),
            ("app_manifest_schema_version", "TEXT", 1, 0),
            ("app_manifest_fingerprint", "TEXT", 1, 0),
            ("result_status", "TEXT", 1, 0),
            ("result_score", "REAL", 0, 0),
            ("fresh_run_id", "TEXT", 0, 0),
            ("captured_result_json", "TEXT", 0, 0),
            ("document_bytes", "INTEGER", 1, 0),
            ("created_at", "TEXT", 1, 0),
        ),
        "cayu_eval_baselines": (
            ("target_key", "TEXT", 1, 1),
            ("corpus_revision", "TEXT", 1, 2),
            ("suite_id", "TEXT", 1, 3),
            ("result_revision", "TEXT", 1, 0),
            ("generation", "INTEGER", 1, 0),
            ("updated_by", "TEXT", 1, 0),
            ("updated_at", "TEXT", 1, 0),
        ),
        "cayu_eval_baseline_mutations": (
            ("operation_id", "TEXT", 0, 1),
            ("target_key", "TEXT", 1, 0),
            ("corpus_revision", "TEXT", 1, 0),
            ("suite_id", "TEXT", 1, 0),
            ("expected_generation", "INTEGER", 1, 0),
            ("previous_result_revision", "TEXT", 0, 0),
            ("selected_result_revision", "TEXT", 1, 0),
            ("resulting_generation", "INTEGER", 1, 0),
            ("actor_id", "TEXT", 1, 0),
            ("created_at", "TEXT", 1, 0),
        ),
    }
    for table, expected in expected_columns.items():
        actual = tuple(
            (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
            for row in connection.execute(f"PRAGMA table_info({table})")
        )
        if actual != expected:
            raise RuntimeError(
                f"SQLite schema object {table!r} conflicts with Cayu's revision-47 "
                "Evals result and baseline contract. Run `cayu storage migrate` or "
                "restore the database from a known-good backup."
            )
    expected_indexes = {
        "idx_cayu_eval_result_records_target_catalog": (
            "cayu_eval_result_records",
            False,
            ("target_key", "created_at", "revision"),
        ),
        "idx_cayu_eval_result_records_contract": (
            "cayu_eval_result_records",
            False,
            ("target_key", "corpus_revision", "suite_id", "created_at", "revision"),
        ),
        "idx_cayu_eval_baseline_mutations_scope": (
            "cayu_eval_baseline_mutations",
            True,
            ("target_key", "corpus_revision", "suite_id", "resulting_generation"),
        ),
    }
    for index_name, (table_name, expected_unique, expected_columns) in expected_indexes.items():
        actual_columns = tuple(
            str(row[2]) for row in connection.execute(f"PRAGMA index_info({index_name})")
        )
        index_row = connection.execute(
            "SELECT tbl_name FROM sqlite_master WHERE type = 'index' AND name = ?",
            (index_name,),
        ).fetchone()
        unique = next(
            (
                bool(row[2])
                for row in connection.execute(f"PRAGMA index_list({table_name})")
                if str(row[1]) == index_name
            ),
            None,
        )
        if (
            index_row is None
            or str(index_row[0]) != table_name
            or unique is not expected_unique
            or actual_columns != expected_columns
        ):
            raise RuntimeError(
                f"SQLite schema object {index_name!r} conflicts with Cayu's revision-47 "
                "Evals query contract. Run `cayu storage migrate` or restore the "
                "database from a known-good backup."
            )


def _validate_captured_eval_case_schema(connection: sqlite3.Connection) -> None:
    row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'cayu_eval_cases'"
    ).fetchone()
    normalized = "" if row is None or row[0] is None else "".join(str(row[0]).lower().split())
    if "check(message_count>=0andmessage_count<=16)" not in normalized:
        raise RuntimeError(
            "SQLite schema object 'cayu_eval_cases.message_count' conflicts with Cayu's "
            "revision-48 captured-evaluation contract. Run `cayu storage migrate` or "
            "restore the database from a known-good backup."
        )


def _validate_eval_run_max_concurrency_schema(connection: sqlite3.Connection) -> None:
    row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'cayu_eval_runs'"
    ).fetchone()
    normalized = "" if row is None or row[0] is None else "".join(str(row[0]).lower().split())
    if "check(max_concurrency>=1andmax_concurrency<=2147483647)" not in normalized:
        raise RuntimeError(
            "SQLite schema object 'cayu_eval_runs.max_concurrency' conflicts with "
            "Cayu's revision-72 portable concurrency contract. Run `cayu storage "
            "migrate` or restore the database from a known-good backup."
        )


def _validate_eval_scenario_schema(connection: sqlite3.Connection) -> None:
    expected_columns = (
        ("revision", "TEXT", 0, 1),
        ("scenario_id", "TEXT", 1, 0),
        ("target_key", "TEXT", 1, 0),
        ("name", "TEXT", 1, 0),
        ("description", "TEXT", 0, 0),
        ("event_count", "INTEGER", 1, 0),
        ("input_event_count", "INTEGER", 1, 0),
        ("approval_checkpoint_count", "INTEGER", 1, 0),
        ("message_count", "INTEGER", 1, 0),
        ("part_count", "INTEGER", 1, 0),
        ("artifact_requirement_count", "INTEGER", 1, 0),
        ("secret_requirement_count", "INTEGER", 1, 0),
        ("document_json", "TEXT", 1, 0),
        ("document_bytes", "INTEGER", 1, 0),
        ("created_at", "TEXT", 1, 0),
    )
    actual_columns = tuple(
        (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
        for row in connection.execute("PRAGMA table_info(cayu_eval_scenarios)")
    )
    if actual_columns != expected_columns:
        raise RuntimeError(
            "SQLite schema object 'cayu_eval_scenarios' conflicts with Cayu's "
            "revision-53 portable scenario contract. Run `cayu storage migrate` "
            "or restore the database from a known-good backup."
        )
    table_row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'cayu_eval_scenarios'"
    ).fetchone()
    normalized_table = (
        ""
        if table_row is None or table_row[0] is None
        else "".join(str(table_row[0]).lower().split())
    )
    required_constraints = (
        "constraintcayu_eval_scenarios_event_count_checkcheck(event_count>=1andevent_count<=1024)",
        "constraintcayu_eval_scenarios_input_event_count_checkcheck(input_event_count>=1andinput_event_count<=1024)",
        "constraintcayu_eval_scenarios_approval_checkpoint_count_checkcheck(approval_checkpoint_count>=0andapproval_checkpoint_count<=1024)",
        "constraintcayu_eval_scenarios_message_count_checkcheck(message_count>=input_event_countandmessage_count<=32768)",
        "constraintcayu_eval_scenarios_part_count_checkcheck(part_count>=message_countandpart_count<=1048576)",
        "constraintcayu_eval_scenarios_artifact_requirement_count_checkcheck(artifact_requirement_count>=0andartifact_requirement_count<=128)",
        "constraintcayu_eval_scenarios_secret_requirement_count_checkcheck(secret_requirement_count>=0andsecret_requirement_count<=128)",
        "constraintcayu_eval_scenarios_document_json_checkcheck(json_valid(document_json))",
        "constraintcayu_eval_scenarios_document_bytes_checkcheck(document_bytes>=1anddocument_bytes<=8388608)",
        "constraintcayu_eval_scenarios_document_size_checkcheck(document_bytes=length(cast(document_jsonasblob)))",
        "constraintcayu_eval_scenarios_event_partition_checkcheck(input_event_count+approval_checkpoint_count=event_count)",
    )
    if "collatenocase" in normalized_table or any(
        fragment not in normalized_table for fragment in required_constraints
    ):
        raise RuntimeError(
            "SQLite schema object 'cayu_eval_scenarios' is missing Cayu's "
            "revision-53 scenario safety constraints. Run `cayu storage migrate` "
            "or restore the database from a known-good backup."
        )
    index_rows = tuple(connection.execute("PRAGMA index_list(cayu_eval_scenarios)"))
    if any(bool(row[2]) and str(row[3]) in {"c", "u"} for row in index_rows):
        raise RuntimeError(
            "SQLite schema object 'cayu_eval_scenarios' has an unexpected unique "
            "constraint or index under Cayu's revision-53 scenario contract. Run "
            "`cayu storage migrate` or restore the database from a known-good backup."
        )
    expected_indexes = {
        "idx_cayu_eval_scenarios_catalog": (
            ("created_at", 1, "BINARY"),
            ("revision", 0, "BINARY"),
        ),
        "idx_cayu_eval_scenarios_id_catalog": (
            ("scenario_id", 0, "BINARY"),
            ("created_at", 1, "BINARY"),
            ("revision", 0, "BINARY"),
        ),
        "idx_cayu_eval_scenarios_target_catalog": (
            ("target_key", 0, "BINARY"),
            ("created_at", 1, "BINARY"),
            ("revision", 0, "BINARY"),
        ),
    }
    for index_name, expected_index_columns in expected_indexes.items():
        index_row = connection.execute(
            "SELECT tbl_name FROM sqlite_master WHERE type = 'index' AND name = ?",
            (index_name,),
        ).fetchone()
        index_metadata = next(
            (row for row in index_rows if str(row[1]) == index_name),
            None,
        )
        actual_index_columns = tuple(
            (str(row[2]), int(row[3]), str(row[4]).upper())
            for row in connection.execute(f"PRAGMA index_xinfo({index_name})")
            if bool(row[5])
        )
        if (
            index_row is None
            or str(index_row[0]) != "cayu_eval_scenarios"
            or index_metadata is None
            or bool(index_metadata[2])
            or str(index_metadata[3]) != "c"
            or bool(index_metadata[4])
            or actual_index_columns != expected_index_columns
        ):
            raise RuntimeError(
                f"SQLite schema object {index_name!r} conflicts with Cayu's "
                "revision-53 scenario catalog contract. Run `cayu storage migrate` "
                "or restore the database from a known-good backup."
            )


def _validate_eval_authored_suite_schema(connection: sqlite3.Connection) -> None:
    expected_columns = (
        ("revision", "TEXT", 0, 1),
        ("suite_id", "TEXT", 1, 0),
        ("suite_revision", "TEXT", 1, 0),
        ("target_key", "TEXT", 1, 0),
        ("name", "TEXT", 1, 0),
        ("description", "TEXT", 0, 0),
        ("case_count", "INTEGER", 1, 0),
        ("assertion_count", "INTEGER", 1, 0),
        ("simple_input_count", "INTEGER", 1, 0),
        ("scenario_count", "INTEGER", 1, 0),
        ("trials", "INTEGER", 1, 0),
        ("timeout_seconds", "INTEGER", 1, 0),
        ("document_json", "TEXT", 1, 0),
        ("document_bytes", "INTEGER", 1, 0),
        ("created_at", "TEXT", 1, 0),
    )
    actual_columns = tuple(
        (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
        for row in connection.execute("PRAGMA table_info(cayu_eval_authored_suites)")
    )
    if actual_columns != expected_columns:
        _raise_eval_authored_suite_schema_error("cayu_eval_authored_suites")
    table_row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'cayu_eval_authored_suites'"
    ).fetchone()
    normalized = (
        ""
        if table_row is None or table_row[0] is None
        else "".join(str(table_row[0]).lower().split())
    )
    required_fragments = (
        "check(case_count>=1andcase_count<=1000)",
        "check(assertion_count>=case_countandassertion_count<=64000)",
        "check(simple_input_count>=0andsimple_input_count<=case_count)",
        "check(scenario_count>=0andscenario_count<=case_count)",
        "check(trials>=1andtrials<=100)",
        "check(timeout_seconds>=1andtimeout_seconds<=3600)",
        "check(json_valid(document_json))",
        "check(document_bytes>=1anddocument_bytes<=8388608)",
        "check(document_bytes=length(cast(document_jsonasblob)))",
        "check(simple_input_count+scenario_count=case_count)",
        "check(assertion_count*trials<=10000)",
    )
    if "collatenocase" in normalized or any(
        fragment not in normalized for fragment in required_fragments
    ):
        _raise_eval_authored_suite_schema_error("authored suite safety constraints")
    index_rows = tuple(connection.execute("PRAGMA index_list(cayu_eval_authored_suites)"))
    if any(bool(row[2]) and str(row[3]) in {"c", "u"} for row in index_rows):
        _raise_eval_authored_suite_schema_error("unexpected unique index")
    expected_indexes = {
        "idx_cayu_eval_authored_suites_catalog": (
            ("created_at", 1, "BINARY"),
            ("revision", 0, "BINARY"),
        ),
        "idx_cayu_eval_authored_suites_id_catalog": (
            ("suite_id", 0, "BINARY"),
            ("created_at", 1, "BINARY"),
            ("revision", 0, "BINARY"),
        ),
        "idx_cayu_eval_authored_suites_target_catalog": (
            ("target_key", 0, "BINARY"),
            ("created_at", 1, "BINARY"),
            ("revision", 0, "BINARY"),
        ),
    }
    for index_name, expected in expected_indexes.items():
        metadata = next((row for row in index_rows if str(row[1]) == index_name), None)
        actual = tuple(
            (str(row[2]), int(row[3]), str(row[4]).upper())
            for row in connection.execute(f"PRAGMA index_xinfo({index_name})")
            if bool(row[5])
        )
        if (
            metadata is None
            or bool(metadata[2])
            or str(metadata[3]) != "c"
            or bool(metadata[4])
            or actual != expected
        ):
            _raise_eval_authored_suite_schema_error(index_name)


def _raise_eval_authored_suite_schema_error(name: str) -> NoReturn:
    raise RuntimeError(
        f"SQLite schema object {name!r} conflicts with Cayu's revision-64 "
        "authored-suite contract. Run `cayu storage migrate` or restore the "
        "database from a known-good backup."
    )


def _validate_eval_judge_calibration_schema(connection: sqlite3.Connection) -> None:
    expected_columns = (
        ("revision", "TEXT", 0, 1),
        ("run_id", "TEXT", 1, 0),
        ("definition_revision", "TEXT", 1, 0),
        ("target_key", "TEXT", 1, 0),
        ("trial_count", "INTEGER", 1, 0),
        ("report_json", "TEXT", 1, 0),
        ("document_bytes", "INTEGER", 1, 0),
        ("created_at", "TEXT", 1, 0),
    )
    actual_columns = tuple(
        (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
        for row in connection.execute("PRAGMA table_info(cayu_eval_judge_calibrations)")
    )
    if actual_columns != expected_columns:
        _raise_eval_judge_calibration_schema_error("cayu_eval_judge_calibrations")
    table_row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' "
        "AND name = 'cayu_eval_judge_calibrations'"
    ).fetchone()
    normalized = (
        ""
        if table_row is None or table_row[0] is None
        else "".join(str(table_row[0]).lower().split())
    )
    required_fragments = (
        "run_idtextcollatebinarynotnullunique",
        "check(trial_count>=1andtrial_count<=10)",
        "check(json_valid(report_json)andjson_type(report_json)='object')",
        "check(document_bytes>=1anddocument_bytes<=2097152)",
        "check(document_bytes=length(cast(report_jsonasblob)))",
    )
    if "revisiontextcollatebinaryprimarykey" not in normalized or any(
        fragment not in normalized for fragment in required_fragments
    ):
        _raise_eval_judge_calibration_schema_error("calibration safety constraints")
    index_rows = tuple(connection.execute("PRAGMA index_list(cayu_eval_judge_calibrations)"))
    unique_rows = tuple(row for row in index_rows if bool(row[2]) and str(row[3]) == "u")
    if len(unique_rows) != 1:
        _raise_eval_judge_calibration_schema_error("run_id uniqueness")
    unique_columns = tuple(
        str(row[2])
        for row in connection.execute(f"PRAGMA index_xinfo({unique_rows[0][1]})")
        if bool(row[5])
    )
    if unique_columns != ("run_id",) or any(
        bool(row[2]) and str(row[3]) == "c" for row in index_rows
    ):
        _raise_eval_judge_calibration_schema_error("run_id uniqueness")
    expected_indexes = {
        "idx_cayu_eval_judge_calibrations_target": (
            ("target_key", 0, "BINARY"),
            ("created_at", 1, "BINARY"),
            ("revision", 0, "BINARY"),
        ),
        "idx_cayu_eval_judge_calibrations_definition": (
            ("definition_revision", 0, "BINARY"),
            ("created_at", 1, "BINARY"),
            ("revision", 0, "BINARY"),
        ),
    }
    for index_name, expected in expected_indexes.items():
        metadata = next((row for row in index_rows if str(row[1]) == index_name), None)
        actual = tuple(
            (str(row[2]), int(row[3]), str(row[4]).upper())
            for row in connection.execute(f"PRAGMA index_xinfo({index_name})")
            if bool(row[5])
        )
        if (
            metadata is None
            or bool(metadata[2])
            or str(metadata[3]) != "c"
            or bool(metadata[4])
            or actual != expected
        ):
            _raise_eval_judge_calibration_schema_error(index_name)


def _raise_eval_judge_calibration_schema_error(name: str) -> NoReturn:
    raise RuntimeError(
        f"SQLite schema object {name!r} conflicts with Cayu's revision-68 "
        "judge-calibration contract. Run `cayu storage migrate` or restore the "
        "database from a known-good backup."
    )


def _validate_verified_work_schema(
    connection: sqlite3.Connection,
    *,
    require_verifier_profiles: bool,
) -> None:
    task_columns = {
        str(row[1]): (str(row[2]).upper(), int(row[3]))
        for row in connection.execute("PRAGMA table_info(cayu_tasks)")
    }
    if task_columns.get("work_contract_json") != ("TEXT", 0):
        raise RuntimeError(
            "SQLite task work-contract storage conflicts with Cayu's revision-49 contract."
        )
    required_columns = {
        "cayu_work_contracts": (
            ("contract_id", "TEXT", 1, 1),
            ("version", "INTEGER", 1, 2),
            ("fingerprint", "TEXT", 1, 0),
            ("contract_json", "TEXT", 1, 0),
        ),
        "cayu_task_session_execution_authority": (
            ("session_id", "TEXT", 1, 1),
            ("authority_kind", "TEXT", 1, 0),
            ("committed_at", "TEXT", 1, 0),
        ),
        "cayu_work_attempts": (
            ("attempt_id", "TEXT", 1, 1),
            ("task_id", "TEXT", 1, 0),
            ("ordinal", "INTEGER", 1, 0),
            ("request_sha256", "TEXT", 1, 0),
            ("started_at", "TEXT", 1, 0),
            ("attempt_json", "TEXT", 1, 0),
        ),
        "cayu_completion_proposals": (
            ("proposal_id", "TEXT", 1, 1),
            ("attempt_id", "TEXT", 1, 0),
            ("task_id", "TEXT", 1, 0),
            ("request_sha256", "TEXT", 1, 0),
            ("proposed_at", "TEXT", 1, 0),
            ("proposal_json", "TEXT", 1, 0),
        ),
        "cayu_completion_verification_claims": (
            ("claim_id", "TEXT", 1, 1),
            ("proposal_id", "TEXT", 1, 0),
            ("attempt_number", "INTEGER", 1, 0),
            ("request_sha256", "TEXT", 1, 0),
            ("lease_expires_at", "TEXT", 1, 0),
            ("is_current", "INTEGER", 1, 0),
            ("claim_json", "TEXT", 1, 0),
        ),
        "cayu_completion_decisions": (
            ("decision_id", "TEXT", 1, 1),
            ("proposal_id", "TEXT", 1, 0),
            ("task_id", "TEXT", 1, 0),
            ("attempt_id", "TEXT", 1, 0),
            ("claim_id", "TEXT", 1, 0),
            ("verdict", "TEXT", 1, 0),
            ("gap_fingerprint", "TEXT", 1, 0),
            ("request_sha256", "TEXT", 1, 0),
            ("decided_at", "TEXT", 1, 0),
            ("decision_json", "TEXT", 1, 0),
        ),
        "cayu_completion_decision_application_receipts": (
            ("task_id", "TEXT", 1, 1),
            ("idempotency_key", "TEXT", 1, 2),
            ("decision_id", "TEXT", 1, 0),
            ("request_sha256", "TEXT", 1, 0),
            ("applied_at", "TEXT", 1, 0),
            ("receipt_json", "TEXT", 1, 0),
        ),
    }
    if require_verifier_profiles:
        required_columns["cayu_completion_verification_claims"] += (
            ("verifier_profile_fingerprint", "TEXT", 0, 0),
        )
        required_columns["cayu_completion_decisions"] += (
            ("verifier_profile_fingerprint", "TEXT", 0, 0),
        )
        required_columns["cayu_completion_verifier_profiles"] = (
            ("proposal_id", "TEXT", 1, 1),
            ("task_id", "TEXT", 1, 0),
            ("attempt_id", "TEXT", 1, 0),
            ("profile_fingerprint", "TEXT", 1, 0),
            ("request_sha256", "TEXT", 1, 0),
            ("prepared_at", "TEXT", 1, 0),
            ("profile_json", "TEXT", 1, 0),
        )
    contract_revision = 58 if require_verifier_profiles else 49
    for table, expected in required_columns.items():
        actual = tuple(
            (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
            for row in connection.execute(f"PRAGMA table_info({table})")
        )
        if actual != expected:
            raise RuntimeError(
                f"SQLite schema object {table!r} conflicts with Cayu's "
                f"revision-{contract_revision} verified-work contract."
            )

    def normalized_sql(name: str, object_type: str) -> str:
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = ? AND name = ?",
            (object_type, name),
        ).fetchone()
        return "" if row is None or row[0] is None else " ".join(str(row[0]).lower().split())

    if (
        "work_contract_json text check (work_contract_json is null or "
        "json_valid(work_contract_json))" not in normalized_sql("cayu_tasks", "table")
    ):
        raise RuntimeError(
            "SQLite task work-contract storage conflicts with Cayu's revision-49 contract."
        )

    required_table_fragments = {
        "cayu_work_contracts": (
            "primary key (contract_id, version)",
            "check (version >= 1)",
            "length(fingerprint) = 64",
            "fingerprint not glob '*[^0-9a-f]*'",
            "check (json_valid(contract_json))",
        ),
        "cayu_task_session_execution_authority": (
            "session_id text not null primary key",
            "authority_kind in ('ordinary', 'contracted')",
        ),
        "cayu_work_attempts": (
            "references cayu_tasks(id) on delete restrict",
            "unique (task_id, ordinal)",
            "check (ordinal >= 1)",
            "length(request_sha256) = 64",
            "request_sha256 not glob '*[^0-9a-f]*'",
            "check (json_valid(attempt_json))",
        ),
        "cayu_completion_proposals": (
            "attempt_id text not null unique references cayu_work_attempts(attempt_id) on delete restrict",
            "task_id text not null references cayu_tasks(id) on delete restrict",
            "length(request_sha256) = 64",
            "request_sha256 not glob '*[^0-9a-f]*'",
            "check (json_valid(proposal_json))",
        ),
        "cayu_completion_verification_claims": (
            "references cayu_completion_proposals(proposal_id) on delete restrict",
            "unique (proposal_id, attempt_number)",
            "check (attempt_number >= 1)",
            "length(request_sha256) = 64",
            "request_sha256 not glob '*[^0-9a-f]*'",
            "check (is_current in (0, 1))",
            "check (json_valid(claim_json))",
        ),
        "cayu_completion_decisions": (
            "proposal_id text not null unique references cayu_completion_proposals(proposal_id) on delete restrict",
            "task_id text not null references cayu_tasks(id) on delete restrict",
            "references cayu_work_attempts(attempt_id) on delete restrict",
            "references cayu_completion_verification_claims(claim_id) on delete restrict",
            "verdict in ('accepted', 'rejected', 'blocked', 'needs_review')",
            "length(gap_fingerprint) = 64",
            "gap_fingerprint not glob '*[^0-9a-f]*'",
            "length(request_sha256) = 64",
            "request_sha256 not glob '*[^0-9a-f]*'",
            "check (json_valid(decision_json))",
        ),
        "cayu_completion_decision_application_receipts": (
            "primary key (task_id, idempotency_key)",
            "task_id text not null references cayu_tasks(id) on delete restrict",
            "decision_id text not null unique references cayu_completion_decisions(decision_id) on delete restrict",
            "length(request_sha256) = 64",
            "request_sha256 not glob '*[^0-9a-f]*'",
            "check (json_valid(receipt_json))",
        ),
    }
    if require_verifier_profiles:
        required_table_fragments["cayu_completion_verification_claims"] += (
            "verifier_profile_fingerprint is not null",
            "length(verifier_profile_fingerprint) = 64",
            "verifier_profile_fingerprint not glob '*[^0-9a-f]*'",
        )
        required_table_fragments["cayu_completion_decisions"] += (
            "verifier_profile_fingerprint is not null",
            "length(verifier_profile_fingerprint) = 64",
            "verifier_profile_fingerprint not glob '*[^0-9a-f]*'",
        )
        required_table_fragments["cayu_completion_verifier_profiles"] = (
            "proposal_id text not null primary key",
            "references cayu_completion_proposals(proposal_id) on delete restrict",
            "task_id text not null references cayu_tasks(id) on delete restrict",
            "attempt_id text not null unique references cayu_work_attempts(attempt_id) on delete restrict",
            "length(profile_fingerprint) = 64",
            "profile_fingerprint not glob '*[^0-9a-f]*'",
            "length(request_sha256) = 64",
            "request_sha256 not glob '*[^0-9a-f]*'",
            "check (json_valid(profile_json))",
        )
    for table, fragments in required_table_fragments.items():
        definition = normalized_sql(table, "table")
        if any(fragment not in definition for fragment in fragments):
            raise RuntimeError(
                f"SQLite schema object {table!r} conflicts with Cayu's "
                f"revision-{contract_revision} verified-work contract."
            )
    required_indexes = {
        "idx_cayu_completion_claim_current": (
            "cayu_completion_verification_claims",
            ("proposal_id",),
            True,
            "where is_current = 1",
        ),
        "idx_cayu_completion_decisions_task_gap": (
            "cayu_completion_decisions",
            ("task_id", "verdict", "gap_fingerprint"),
            False,
            None,
        ),
        "idx_cayu_tasks_contracted_session": (
            "cayu_tasks",
            ("session_id", "created_at", "id"),
            False,
            "where work_contract_json is not null",
        ),
        "idx_cayu_work_attempts_task_latest": (
            "cayu_work_attempts",
            ("task_id", "ordinal"),
            False,
            None,
        ),
    }
    if require_verifier_profiles:
        required_indexes["idx_cayu_completion_verifier_profiles_task"] = (
            "cayu_completion_verifier_profiles",
            ("task_id", "attempt_id"),
            False,
            None,
        )
    for index, (table, columns, unique, predicate) in required_indexes.items():
        index_rows = {str(row[1]): row for row in connection.execute(f"PRAGMA index_list({table})")}
        row = index_rows.get(index)
        actual_columns = tuple(
            str(column[2]) for column in connection.execute(f"PRAGMA index_info({index})")
        )
        definition = normalized_sql(index, "index")
        if (
            row is None
            or bool(row[2]) is not unique
            or actual_columns != columns
            or (predicate is None and int(row[4]) != 0)
            or (predicate is not None and (int(row[4]) != 1 or predicate not in definition))
        ):
            raise RuntimeError(
                f"SQLite schema object {index!r} conflicts with Cayu's "
                "revision-49 verified-work contract."
            )


def _validate_eval_run_invocation_column(connection: sqlite3.Connection) -> None:
    columns = {
        str(row[1]): (str(row[2]).upper(), int(row[3]))
        for row in connection.execute("PRAGMA table_info(cayu_eval_runs)")
    }
    if columns.get("invocation_json") != ("TEXT", 1):
        raise RuntimeError(
            "SQLite schema object 'cayu_eval_runs.invocation_json' conflicts with "
            "Cayu's revision-50 durable eval invocation contract. Run "
            "`cayu storage migrate` or restore the database from a known-good backup."
        )


def _validate_eval_run_scenario_progress_column(connection: sqlite3.Connection) -> None:
    columns = {
        str(row[1]): (str(row[2]).upper(), int(row[3]))
        for row in connection.execute("PRAGMA table_info(cayu_eval_runs)")
    }
    if columns.get("scenario_progress_json") != ("TEXT", 0):
        raise RuntimeError(
            "SQLite schema object 'cayu_eval_runs.scenario_progress_json' conflicts "
            "with Cayu's revision-56 controlled-scenario execution contract. Run "
            "`cayu storage migrate` or restore the database from a known-good backup."
        )


def _validate_eval_run_trial_checkpoint_schema(connection: sqlite3.Connection) -> None:
    run_columns = {
        str(row[1]): (str(row[2]).upper(), int(row[3]), row[4])
        for row in connection.execute("PRAGMA table_info(cayu_eval_runs)")
    }
    if run_columns.get("trial_checkpoint_count") != ("INTEGER", 1, "0"):
        raise RuntimeError(
            "SQLite schema object 'cayu_eval_runs.trial_checkpoint_count' conflicts "
            "with Cayu's revision-74 eval trial recovery contract. Run `cayu storage "
            "migrate` or restore the database from a known-good backup."
        )
    if run_columns.get("trial_checkpoint_bytes") != ("INTEGER", 1, "0"):
        raise RuntimeError(
            "SQLite schema object 'cayu_eval_runs.trial_checkpoint_bytes' conflicts "
            "with Cayu's revision-74 eval trial recovery contract. Run `cayu storage "
            "migrate` or restore the database from a known-good backup."
        )
    checkpoint_columns = tuple(
        (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
        for row in connection.execute("PRAGMA table_info(cayu_eval_run_trial_checkpoints)")
    )
    if checkpoint_columns != (
        ("run_id", "TEXT", 1, 1),
        ("case_id", "TEXT", 1, 2),
        ("trial_number", "INTEGER", 1, 3),
        ("checkpoint_json", "TEXT", 1, 0),
        ("document_bytes", "INTEGER", 1, 0),
    ):
        raise RuntimeError(
            "SQLite schema object 'cayu_eval_run_trial_checkpoints' conflicts with "
            "Cayu's revision-74 eval trial recovery contract. Run `cayu storage migrate` "
            "or restore the database from a known-good backup."
        )
    checkpoint_foreign_keys = tuple(
        (str(row[2]), str(row[3]), str(row[4]), str(row[6]).upper())
        for row in connection.execute("PRAGMA foreign_key_list(cayu_eval_run_trial_checkpoints)")
    )
    if checkpoint_foreign_keys != (("cayu_eval_runs", "run_id", "run_id", "CASCADE"),):
        raise RuntimeError(
            "SQLite schema object 'cayu_eval_run_trial_checkpoints' has an invalid "
            "run ownership constraint. Run `cayu storage migrate` or restore the "
            "database from a known-good backup."
        )
    if run_columns.get("authored_suite_launch_revision") != ("TEXT", 0, None):
        raise RuntimeError(
            "SQLite schema object 'cayu_eval_runs.authored_suite_launch_revision' "
            "conflicts with Cayu's revision-74 authored-suite concurrency contract. "
            "Run `cayu storage migrate` or restore the database from a known-good backup."
        )
    if run_columns.get("authored_suite_launch_lane") != ("INTEGER", 0, None):
        raise RuntimeError(
            "SQLite schema object 'cayu_eval_runs.authored_suite_launch_lane' "
            "conflicts with Cayu's revision-74 authored-suite concurrency contract. "
            "Run `cayu storage migrate` or restore the database from a known-good backup."
        )
    index_row = next(
        (
            row
            for row in connection.execute("PRAGMA index_list(cayu_eval_runs)")
            if str(row[1]) == "idx_cayu_eval_runs_authored_suite_launch_claim"
        ),
        None,
    )
    index = (
        None
        if index_row is None
        else (
            int(index_row[2]),
            int(index_row[4]),
            tuple(
                str(column[2])
                for column in connection.execute(
                    "PRAGMA index_info(idx_cayu_eval_runs_authored_suite_launch_claim)"
                )
            ),
        )
    )
    expected = (
        0,
        1,
        (
            "authored_suite_launch_revision",
            "authored_suite_launch_lane",
            "created_at",
            "run_id",
            "status",
        ),
    )
    if index != expected:
        raise RuntimeError(
            "SQLite schema object 'idx_cayu_eval_runs_authored_suite_launch_claim' "
            "conflicts with Cayu's revision-74 authored-suite concurrency contract. "
            "Run `cayu storage migrate` or restore the database from a known-good backup."
        )


def _validate_session_message_queue_typed_message_column(
    connection: sqlite3.Connection,
) -> None:
    columns = {
        str(row[1]): (str(row[2]).upper(), int(row[3]))
        for row in connection.execute("PRAGMA table_info(cayu_session_message_queue)")
    }
    if columns.get("message_json") != ("TEXT", 0):
        raise RuntimeError(
            "SQLite schema object 'cayu_session_message_queue.message_json' conflicts "
            "with Cayu's revision-57 typed queued-message contract. Run "
            "`cayu storage migrate` or restore the database from a known-good backup."
        )


def _validate_memory_evidence_schema(connection: sqlite3.Connection) -> None:
    expected_columns = {
        "cayu_recall_receipts": (
            ("receipt_id", "TEXT", 0, 1),
            ("session_id", "TEXT", 1, 0),
            ("interaction_id", "TEXT", 1, 0),
            ("model_step_id", "TEXT", 1, 0),
            ("created_at", "TEXT", 1, 0),
            ("receipt_json", "TEXT", 1, 0),
            ("document_bytes", "INTEGER", 1, 0),
        ),
        "cayu_context_exposures": (
            ("exposure_id", "TEXT", 0, 1),
            ("session_id", "TEXT", 1, 0),
            ("interaction_id", "TEXT", 1, 0),
            ("model_step_id", "TEXT", 1, 0),
            ("model_attempt_id", "TEXT", 1, 0),
            ("provider_attempt_id", "TEXT", 1, 0),
            ("state", "TEXT", 1, 0),
            ("state_revision", "INTEGER", 1, 0),
            ("created_at", "TEXT", 1, 0),
            ("updated_at", "TEXT", 1, 0),
            ("exposure_json", "TEXT", 1, 0),
            ("document_bytes", "INTEGER", 1, 0),
        ),
        "cayu_recall_item_exposures": (
            ("exposure_id", "TEXT", 1, 1),
            ("ordinal", "INTEGER", 1, 2),
            ("receipt_id", "TEXT", 1, 0),
            ("receipt_item_ordinal", "INTEGER", 1, 0),
            ("item_json", "TEXT", 1, 0),
            ("document_bytes", "INTEGER", 1, 0),
        ),
    }
    for table, expected in expected_columns.items():
        actual = tuple(
            (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
            for row in connection.execute(f"PRAGMA table_info({table})")
        )
        if actual != expected:
            _raise_memory_evidence_schema_error(table)

    required_table_sql = {
        "cayu_recall_receipts": (
            "check(json_valid(receipt_json))",
            "document_bytes>=1anddocument_bytes<=256000",
        ),
        "cayu_context_exposures": (
            "statein('planned','prepared','dispatch_started','acknowledged',"
            "'completed','failed','cancelled','indeterminate')",
            "state_revision>=0andstate_revision<16",
            "check(json_valid(exposure_json))",
            "document_bytes>=1anddocument_bytes<=128000",
        ),
        "cayu_recall_item_exposures": (
            "ordinal>=0andordinal<64",
            "receipt_item_ordinal>=0andreceipt_item_ordinal<64",
            "check(json_valid(item_json))",
            "document_bytes>=1anddocument_bytes<=16384",
        ),
    }
    for table, fragments in required_table_sql.items():
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        normalized = (
            ""
            if row is None or row[0] is None
            else "".join(str(row[0]).lower().split()).replace("(", "").replace(")", "")
        )
        if "collatenocase" in normalized or any(
            fragment.replace("(", "").replace(")", "") not in normalized for fragment in fragments
        ):
            _raise_memory_evidence_schema_error(table)

    expected_foreign_keys = {
        "cayu_recall_receipts": {
            ("session_id", "cayu_sessions", "id", "CASCADE"),
        },
        "cayu_context_exposures": {
            ("session_id", "cayu_sessions", "id", "CASCADE"),
        },
        "cayu_recall_item_exposures": {
            ("exposure_id", "cayu_context_exposures", "exposure_id", "CASCADE"),
            ("receipt_id", "cayu_recall_receipts", "receipt_id", "CASCADE"),
        },
    }
    for table, expected in expected_foreign_keys.items():
        actual = {
            (str(row[3]), str(row[2]), str(row[4]), str(row[6]).upper())
            for row in connection.execute(f"PRAGMA foreign_key_list({table})")
        }
        if actual != expected:
            _raise_memory_evidence_schema_error(f"{table} foreign keys")

    expected_unique_columns = {
        "cayu_recall_receipts": set(),
        "cayu_context_exposures": {
            ("session_id", "model_attempt_id"),
            ("session_id", "provider_attempt_id"),
        },
        "cayu_recall_item_exposures": {
            ("exposure_id", "receipt_id", "receipt_item_ordinal"),
        },
    }
    for table, expected in expected_unique_columns.items():
        index_rows = tuple(connection.execute(f"PRAGMA index_list({table})"))
        standalone_unique_indexes = tuple(
            str(index[1]) for index in index_rows if bool(index[2]) and str(index[3]) == "c"
        )
        if standalone_unique_indexes:
            _raise_memory_evidence_schema_error(standalone_unique_indexes[0])
        actual = {
            tuple(str(column[2]) for column in connection.execute(f"PRAGMA index_info({index[1]})"))
            for index in index_rows
            if bool(index[2]) and str(index[3]) == "u"
        }
        if actual != expected:
            _raise_memory_evidence_schema_error(f"{table} uniqueness")

    expected_indexes = {
        "idx_cayu_recall_receipts_session_page": (
            "cayu_recall_receipts",
            ("session_id", "created_at", "receipt_id"),
        ),
        "idx_cayu_recall_receipts_interaction_page": (
            "cayu_recall_receipts",
            ("session_id", "interaction_id", "created_at", "receipt_id"),
        ),
        "idx_cayu_recall_receipts_step_page": (
            "cayu_recall_receipts",
            ("session_id", "model_step_id", "created_at", "receipt_id"),
        ),
        "idx_cayu_recall_receipts_interaction_step_page": (
            "cayu_recall_receipts",
            (
                "session_id",
                "interaction_id",
                "model_step_id",
                "created_at",
                "receipt_id",
            ),
        ),
        "idx_cayu_context_exposures_session_page": (
            "cayu_context_exposures",
            ("session_id", "created_at", "exposure_id"),
        ),
        "idx_cayu_context_exposures_interaction_page": (
            "cayu_context_exposures",
            ("session_id", "interaction_id", "created_at", "exposure_id"),
        ),
        "idx_cayu_context_exposures_step_page": (
            "cayu_context_exposures",
            ("session_id", "model_step_id", "created_at", "exposure_id"),
        ),
        "idx_cayu_context_exposures_interaction_step_page": (
            "cayu_context_exposures",
            (
                "session_id",
                "interaction_id",
                "model_step_id",
                "created_at",
                "exposure_id",
            ),
        ),
        "idx_cayu_recall_item_exposures_receipt": (
            "cayu_recall_item_exposures",
            ("receipt_id", "exposure_id", "ordinal"),
        ),
    }
    for index_name, (table, expected_columns_for_index) in expected_indexes.items():
        index_row = connection.execute(
            "SELECT tbl_name FROM sqlite_master WHERE type = 'index' AND name = ?",
            (index_name,),
        ).fetchone()
        index_metadata = next(
            (
                row
                for row in connection.execute(f"PRAGMA index_list({table})")
                if str(row[1]) == index_name
            ),
            None,
        )
        indexed_columns = tuple(
            (str(row[2]), int(row[3]), str(row[4]).upper())
            for row in connection.execute(f"PRAGMA index_xinfo({index_name})")
            if bool(row[5])
        )
        if (
            index_row is None
            or str(index_row[0]) != table
            or index_metadata is None
            or bool(index_metadata[2])
            or str(index_metadata[3]) != "c"
            or bool(index_metadata[4])
            or indexed_columns
            != tuple((column, 0, "BINARY") for column in expected_columns_for_index)
        ):
            _raise_memory_evidence_schema_error(index_name)


def _raise_memory_evidence_schema_error(name: str) -> NoReturn:
    raise RuntimeError(
        f"SQLite schema object {name!r} conflicts with Cayu's revision-51 memory "
        "evidence contract. Recreate the database or restore a known-good "
        "revision-51 backup."
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

    ``begin_immediate=False`` starts a deferred transaction. Read paths use it
    to hydrate one authorization-checked result from a stable WAL snapshot;
    writes still take the immediate writer reservation before inspecting state.
    """
    failure: BaseException | None = None
    suppress_implicit_context = False
    transaction_started = False
    try:
        if begin_immediate:
            connection.execute("BEGIN IMMEDIATE")
        else:
            connection.execute("BEGIN")
        transaction_started = True
        yield
        connection.commit()
    except BaseException as primary:
        if transaction_started:
            failure = _settle_failed_transaction(connection, primary)
            suppress_implicit_context = failure is not primary
        else:
            # A boundary wrapper can raise after SQLite accepted BEGIN. Probe
            # conservatively so that owned work is rolled back, while a closed
            # or otherwise unreadable connection preserves the original BEGIN
            # failure instead of manufacturing a cleanup aggregate.
            try:
                began_despite_error = connection.in_transaction
            except BaseException:
                failure = primary
            else:
                failure = (
                    _settle_failed_transaction(connection, primary)
                    if began_despite_error
                    else primary
                )
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


def _reject_unprofiled_verified_work_records(connection: sqlite3.Connection) -> None:
    tables = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN "
            "('cayu_completion_verification_claims', 'cayu_completion_decisions')"
        )
    }
    for table in sorted(tables):
        row = connection.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone()
        if row is not None:
            raise RuntimeError(
                "SQLite migration revision 58 cannot attribute existing completion-"
                "verification records to immutable verifier profiles. Recreate the pre-release "
                "database before migrating."
            )


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
    if (
        current != schema.UNINITIALIZED
        and current < 41
        and any(revision.revision == 41 for revision in schema.pending(current))
    ):
        _reject_populated_pre_knowledge_access_snapshot_database(connection)
    if current < 42 and any(revision.revision == 42 for revision in schema.pending(current)):
        _reject_populated_pre_knowledge_revision_database(connection)
    if current < 46 and any(revision.revision == 46 for revision in schema.pending(current)):
        _reject_populated_pre_transcript_search_database(connection)
    if (
        current != schema.UNINITIALIZED
        and current < 52
        and any(revision.revision == 52 for revision in schema.pending(current))
    ):
        _reject_populated_pre_targeted_tool_grant_database(connection)
    if current < 58 and any(revision.revision == 58 for revision in schema.pending(current)):
        _reject_unprofiled_verified_work_records(connection)
    if current < 59 and any(revision.revision == 59 for revision in schema.pending(current)):
        _reject_populated_pre_result_resolver_database(connection)
    if current < 65 and any(revision.revision == 65 for revision in schema.pending(current)):
        _reject_populated_pre_bounded_knowledge_entry_database(connection)
    if (
        current != schema.UNINITIALIZED
        and current < 73
        and any(revision.revision == 73 for revision in schema.pending(current))
    ):
        _reject_populated_pre_recall_subscription_database(connection)
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
    if rev.revision == 72:
        _apply_revision_seventy_two(connection, rev)
        return
    with _transaction(connection):
        if rev.revision == 42:
            # Recheck under the same immediate writer transaction that owns the
            # destructive reset DDL. A legacy writer cannot populate an empty
            # table between the refusal check and the schema replacement.
            _reject_populated_pre_knowledge_revision_database(connection)
        if rev.revision == 43:
            _reject_revision_43_knowledge_identity_overflow(connection)
        if rev.revision == 46:
            # BEGIN IMMEDIATE fences transcript writers between the clean-break
            # check and installation of the final non-null projection.
            _reject_populated_pre_transcript_search_database(connection)
        if rev.revision == 52:
            # BEGIN IMMEDIATE fences session writers between the clean-break
            # check and installation of targeted-grant durability.
            _reject_populated_pre_targeted_tool_grant_database(connection)
        if rev.revision == 59:
            _reject_populated_pre_result_resolver_database(connection)
        if rev.revision == 60:
            # Recheck while BEGIN IMMEDIATE excludes legacy writers. The earlier
            # preflight preserves a populated database before any schema DDL;
            # this fence closes the race between that check and replacement of
            # the empty prerelease knowledge outbox/relation tables.
            _reject_populated_pre_knowledge_relation_database(connection)
        if rev.revision == 63:
            # Recheck while BEGIN IMMEDIATE excludes pre-63 writers. Populated
            # prerelease knowledge is never inferred into reviewed decisions.
            _reject_populated_pre_knowledge_maintenance_database(connection)
        if rev.revision == 65:
            # The stored size is authoritative for pre-content read rejection;
            # deriving it for existing rows would be a forbidden backfill.
            _reject_populated_pre_bounded_knowledge_entry_database(connection)
        if rev.revision == 73:
            # BEGIN IMMEDIATE fences revision-71 delivery writers between the
            # clean-break check and installation of input-bound subscriptions.
            _reject_populated_pre_recall_subscription_database(connection)
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
        if rev.revision == 41:
            _validate_knowledge_publication_access_snapshot_column(connection)
        if rev.revision == 42:
            _validate_revision_42_knowledge_schema(connection)
        if rev.revision == 43:
            _validate_revision_43_knowledge_schema(connection)
        if rev.revision == 44:
            _validate_revision_44_knowledge_schema(connection)
        if rev.revision == 45:
            _validate_task_retry_series_schema(connection)
        if rev.revision == 46:
            _validate_revision_46_transcript_search_schema(connection)
        if rev.revision == 47:
            _validate_eval_result_baseline_schema(connection)
        if rev.revision == 48:
            _validate_captured_eval_case_schema(connection)
        if rev.revision == 50:
            _validate_eval_run_invocation_column(connection)
        if rev.revision == 51:
            _validate_memory_evidence_schema(connection)
        if rev.revision == 52:
            _validate_targeted_tool_grant_schema(connection)
        if rev.revision == 53:
            _validate_eval_scenario_schema(connection)
        if rev.revision == 55:
            _validate_task_retry_reconciliation_schema(connection)
        if rev.revision == 56:
            _validate_eval_run_scenario_progress_column(connection)
        if rev.revision == 57:
            _validate_session_message_queue_typed_message_column(connection)
        if rev.revision == 58:
            _validate_verified_work_schema(
                connection,
                require_verifier_profiles=True,
            )
        if rev.revision == 59:
            _validate_session_instance_schema(connection)
        if rev.revision == 61:
            _validate_work_attempt_admission_schema(connection)
        if rev.revision == 62:
            _validate_revision_sixty_two_payload_schema(connection)
        if rev.revision == 60:
            _validate_revision_60_knowledge_schema(connection)
        if rev.revision == 63:
            _validate_revision_63_knowledge_schema(connection)
        if rev.revision == 64:
            _validate_eval_authored_suite_schema(connection)
        if rev.revision == 65:
            _validate_revision_42_knowledge_schema(
                connection,
                require_payload_bytes=True,
            )
        if rev.revision == 66:
            _validate_local_execution_attempt_schema(connection)
        if rev.revision == 67:
            _validate_revision_67_knowledge_schema(connection)
        if rev.revision == 68:
            _validate_eval_judge_calibration_schema(connection)
        if rev.revision == 69:
            _validate_revision_69_work_context_schema(connection)
        if rev.revision == 70:
            _validate_interrupted_task_handoff_schema(connection)
        if rev.revision == 71:
            _validate_revision_71_recall_delivery_schema(connection)
        if rev.revision == 73:
            _validate_revision_71_recall_delivery_schema(
                connection,
                require_processing_schema_version=True,
            )
            _validate_revision_73_recall_subscription_schema(connection)
        if rev.revision == 74:
            _validate_eval_run_trial_checkpoint_schema(connection)
        _record_revision(connection, rev)
        connection.execute(f"PRAGMA user_version = {rev.revision}")


def _apply_revision_seventy_two(
    connection: sqlite3.Connection,
    rev: schema.Revision,
) -> None:
    """Rebuild the eval-run check without retargeting dependent foreign keys."""

    connection.execute("PRAGMA foreign_keys = OFF")
    connection.execute("PRAGMA legacy_alter_table = ON")
    try:
        with _transaction(connection):
            for statement in _iter_statements(_MIGRATION_STEPS[72]):
                connection.execute(statement)
            _validate_eval_run_max_concurrency_schema(connection)
            violation = connection.execute("PRAGMA foreign_key_check").fetchone()
            if violation is not None:
                raise RuntimeError("SQLite migration revision 72 broke an eval-store foreign key.")
            _record_revision(connection, rev)
            connection.execute(f"PRAGMA user_version = {rev.revision}")
    finally:
        connection.execute("PRAGMA legacy_alter_table = OFF")
        connection.execute("PRAGMA foreign_keys = ON")


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
        instance_id=session_instance_id_for_run_request(
            request,
            session_id=session_id,
        ),
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
        metadata=session_metadata_for_creation(
            request.metadata,
            identity=identity,
            tool_capability_ceiling=request.tool_capability_ceiling,
        ),
        labels=copy_label_map(request.labels, "labels"),
    )


def session_to_row_values(session: Session) -> tuple[object, ...]:
    return (
        session.id,
        session.instance_id,
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
        task.session_instance_id,
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
        (
            None
            if task.retry_series is None
            else json_dumps(task.retry_series.model_dump(mode="json"))
        ),
        (
            None
            if task.work_contract is None
            else json_dumps(task.work_contract.model_dump(mode="json", warnings=False))
        ),
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
        session_instance_id=row["session_instance_id"],
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
        retry_series=(
            None
            if row["retry_series_json"] is None
            else TaskRetrySeriesSnapshot.model_validate(json.loads(row["retry_series_json"]))
        ),
        work_contract=(
            None
            if row["work_contract_json"] is None
            else WorkContractRef.model_validate(json.loads(row["work_contract_json"]))
        ),
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
        instance_id=row["instance_id"],
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
        runtime_build_provenance=runtime_build_provenance_from_session_metadata(
            {}
            if row["runtime_build_provenance_json"] is None
            else {
                RUNTIME_BUILD_PROVENANCE_METADATA_KEY: json.loads(
                    row["runtime_build_provenance_json"]
                )
            }
        ),
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
