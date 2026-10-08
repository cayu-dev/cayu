from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from typing import Any, cast
from uuid import uuid4

from cayu.events import Event
from cayu.knowledge.records import (
    MAX_KNOWLEDGE_CHUNK_ID_BYTES,
    MAX_KNOWLEDGE_ENTRY_ID_BYTES,
)
from cayu.sessions.base import (
    PENDING_ACTION_EVENT_TYPE_VALUES,
    TRANSCRIPT_SEARCH_TOKENIZER_VERSION,
    deferred_interaction_input_from_storage_payload,
    deferred_interaction_input_storage_payload,
)
from cayu.storage import _sqlite_catalog as sqlite_catalog
from cayu.storage import _sqlite_connection as sqlite_connection
from cayu.storage import _sqlite_eval_schema as sqlite_eval_schema
from cayu.storage import _sqlite_knowledge_schema as sqlite_knowledge_schema
from cayu.storage import _sqlite_memory_evidence_schema as sqlite_memory_evidence_schema
from cayu.storage import _sqlite_records as sqlite_records
from cayu.storage import _sqlite_session_schema as sqlite_session_schema
from cayu.storage import _sqlite_task_schema as sqlite_task_schema
from cayu.storage import _sqlite_verified_work_schema as sqlite_verified_work_schema
from cayu.storage import _sqlite_work_context_schema as sqlite_work_context_schema
from cayu.storage import migrations as schema
from cayu.storage._accounting_schema import SQLITE_ACCOUNTING_DDL, SQLITE_AUXILIARY_ACCOUNTING_DDL
from cayu.storage._collaboration_schema import (
    SQLITE_COLLABORATION_CLARIFICATION_DDL,
    SQLITE_COLLABORATION_DDL,
    SQLITE_COLLABORATION_LIFECYCLE_DDL,
    SQLITE_COLLABORATION_PLANNING_DDL,
    SQLITE_COLLABORATION_REQUEST_DDL,
    validate_sqlite_collaboration_schema,
)
from cayu.storage._collaboration_wait_schema import (
    SQLITE_COLLABORATION_WAIT_DDL,
    validate_sqlite_wait_discovery,
)
from cayu.storage._completion_evaluation_schema import SQLITE_COMPLETION_EVALUATION_DDL
from cayu.storage._completion_verifier_dispatch_schema import (
    SQLITE_COMPLETION_VERIFIER_DISPATCH_DDL,
)
from cayu.storage._context_selection_schema import validate_sqlite_context_selection_schema
from cayu.storage._diagnostic_inspection import current_diagnostic_store_inspection
from cayu.storage._external_wait_schema import SQLITE_EXTERNAL_WAIT_DDL
from cayu.storage._model_policy_schema import SQLITE_MODEL_POLICY_DDL
from cayu.storage._participant_bindings_schema import (
    SQLITE_PARTICIPANT_BINDINGS_DDL,
    validate_sqlite_participant_bindings,
)
from cayu.storage._product_operation_schema import (
    SQLITE_PRODUCT_OPERATION_DDL,
    validate_sqlite_product_operation_schema,
)
from cayu.storage._session_execution import SQLITE_EXECUTION_DDL
from cayu.storage._task_graph_schema import SQLITE_TASK_GRAPH_DDL
from cayu.storage._task_group_schema import (
    SQLITE_TASK_GROUP_DDL,
    SQLITE_TASK_GROUP_QUIESCENCE_DDL,
)
from cayu.storage._task_scheduling_schema import SQLITE_SCHEDULING_DDL
from cayu.storage.knowledge_transition import require_empty_knowledge_revision_transition
from cayu.tasks.base import (
    TaskInterruptedHandoffRequest,
    prepare_interrupted_task_handoff,
)
from cayu.tasks.records import Task, TaskStatus

_INTERRUPTED_HANDOFF_MIGRATION_BATCH_SIZE = 256


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
        conditions_json TEXT,
        terminal_json TEXT,
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
        reject_only INTEGER NOT NULL DEFAULT 0 CHECK (reject_only IN (0, 1)),
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
            'session.delegated_action.updated',
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
    CREATE INDEX IF NOT EXISTS idx_cayu_events_queue_acceptance
        ON cayu_events(session_id, json_extract(payload_json, '$.queue_id'))
        WHERE event_type = 'session.message.queued';
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

_BASELINE_DDL += SQLITE_ACCOUNTING_DDL

# Per-revision forward-migration DDL, keyed by revision number. The baseline
# (revision 1) is applied from _BASELINE_DDL, so it is not listed here; future
# additive/breaking revisions append their ALTER/CREATE scripts.
_MIGRATION_STEPS: dict[int, str] = {
    108: """
        CREATE TABLE IF NOT EXISTS cayu_context_selection_exclusions (
            selection_key TEXT PRIMARY KEY,
            owner_scope TEXT NOT NULL,
            owner_id TEXT NOT NULL,
            owner_incarnation TEXT NOT NULL,
            source_session_id TEXT NOT NULL,
            source_session_instance_id TEXT NOT NULL,
            request_commitment TEXT NOT NULL,
            decision_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_context_selection_exclusions_owner
            ON cayu_context_selection_exclusions(owner_scope, owner_id, owner_incarnation);
    """,
    103: """
        CREATE TABLE IF NOT EXISTS cayu_session_creation_decisions (
            operation_key TEXT PRIMARY KEY,
            owner_key TEXT NOT NULL,
            state TEXT NOT NULL,
            recovery_pending INTEGER NOT NULL,
            decision_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_creation_decisions_pending
            ON cayu_session_creation_decisions(owner_key, recovery_pending, operation_key);
    """,
    102: SQLITE_PARTICIPANT_BINDINGS_DDL,
    101: """
        CREATE TABLE IF NOT EXISTS cayu_budget_binding_consumptions (
            binding_id TEXT NOT NULL,
            consumption_id TEXT NOT NULL,
            consumed_at TEXT NOT NULL,
            PRIMARY KEY (binding_id, consumption_id)
        );
    """,
    100: """
        CREATE TABLE IF NOT EXISTS cayu_budget_bindings (
            binding_id TEXT PRIMARY KEY,
            authority_digest TEXT NOT NULL,
            registered_at TEXT NOT NULL
        );
    """,
    107: SQLITE_COLLABORATION_PLANNING_DDL,
    111: SQLITE_COLLABORATION_WAIT_DDL,
    112: SQLITE_PRODUCT_OPERATION_DDL,
    113: SQLITE_EXECUTION_DDL,
    114: SQLITE_MODEL_POLICY_DDL,
    115: SQLITE_EXTERNAL_WAIT_DDL,
    116: SQLITE_COMPLETION_VERIFIER_DISPATCH_DDL,
    117: SQLITE_COMPLETION_EVALUATION_DDL,
    110: """
        CREATE TABLE IF NOT EXISTS cayu_producer_cleanup_receipts (
            operation_key TEXT PRIMARY KEY NOT NULL,
            namespace_key TEXT NOT NULL,
            generation INTEGER NOT NULL CHECK (generation BETWEEN 1 AND 9007199254740991),
            receipt_json TEXT NOT NULL CHECK (
                json_valid(receipt_json) AND json_type(receipt_json) = 'object'
                AND length(CAST(receipt_json AS BLOB)) BETWEEN 1 AND 65536
            )
        );
        CREATE INDEX IF NOT EXISTS idx_cayu_producer_cleanup_namespace
        ON cayu_producer_cleanup_receipts(namespace_key, generation, operation_key);
        CREATE TABLE IF NOT EXISTS cayu_producer_cleanup_retirements (
            namespace_key TEXT PRIMARY KEY NOT NULL,
            through_generation INTEGER NOT NULL CHECK (through_generation BETWEEN 1 AND 9007199254740991)
        );
    """,
    109: """
        CREATE INDEX IF NOT EXISTS idx_cayu_budget_reservations_session_identity
        ON cayu_budget_reservations(session_id, reservation_id);
    """,
    106: "",  # Contract-only writer fence; existing typed request records own storage.
    105: SQLITE_COLLABORATION_CLARIFICATION_DDL,
    104: """
        CREATE TABLE IF NOT EXISTS cayu_peer_content_attempts (
            operation_key TEXT PRIMARY KEY, request_json TEXT NOT NULL, receipt_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS cayu_peer_content_receipts (
            append_key_json TEXT PRIMARY KEY,
            operation_key TEXT NOT NULL UNIQUE,
            commitment_json TEXT NOT NULL,
            receipt_json TEXT NOT NULL CHECK (json_valid(receipt_json)),
            request_json TEXT,
            target_deleted INTEGER NOT NULL DEFAULT 0 CHECK (target_deleted IN (0, 1))
        );
        CREATE TABLE IF NOT EXISTS cayu_peer_content_exposures (
            exposure_id TEXT PRIMARY KEY,
            operation_key TEXT NOT NULL UNIQUE,
            append_key_json TEXT NOT NULL,
            commitment_json TEXT NOT NULL,
            receipt_json TEXT NOT NULL CHECK (json_valid(receipt_json))
        );
        CREATE INDEX IF NOT EXISTS idx_cayu_peer_content_exposures_append
            ON cayu_peer_content_exposures(append_key_json, exposure_id);
    """,
    99: """
        CREATE TABLE IF NOT EXISTS cayu_context_view_lifecycle_events (
            event_id TEXT PRIMARY KEY,
            operation_key TEXT NOT NULL UNIQUE,
            selection_key TEXT NOT NULL,
            view_id TEXT NOT NULL,
            state TEXT NOT NULL CHECK (state IN ('adopted', 'transferred', 'released', 'expired')),
            owner_scope TEXT NOT NULL,
            owner_id TEXT NOT NULL,
            owner_incarnation TEXT NOT NULL,
            pin_commitment TEXT NOT NULL,
            ownership_revision INTEGER NOT NULL CHECK (ownership_revision >= 1),
            event_json TEXT NOT NULL CHECK (json_valid(event_json))
        );
        CREATE INDEX IF NOT EXISTS idx_cayu_context_view_lifecycle_events_view
            ON cayu_context_view_lifecycle_events(view_id, ownership_revision, event_id);
    """,
    98: """
        CREATE TABLE IF NOT EXISTS cayu_context_view_ownership_operations (
            operation_key TEXT PRIMARY KEY,
            selection_key TEXT NOT NULL,
            request_commitment TEXT NOT NULL,
            receipt_json TEXT NOT NULL CHECK (json_valid(receipt_json))
        );
        CREATE INDEX IF NOT EXISTS idx_cayu_context_view_ownership_operations_selection
            ON cayu_context_view_ownership_operations(selection_key);
    """,
    97: """
        CREATE TABLE IF NOT EXISTS cayu_context_views (
            view_id TEXT PRIMARY KEY,
            publication_key TEXT NOT NULL UNIQUE,
            source_owner_scope TEXT NOT NULL,
            source_owner_id TEXT NOT NULL,
            source_owner_incarnation TEXT NOT NULL,
            source_session_id TEXT NOT NULL,
            source_session_instance_id TEXT NOT NULL,
            transcript_cursor INTEGER NOT NULL CHECK (transcript_cursor >= 0),
            projection_schema TEXT NOT NULL,
            extension_set_commitment TEXT NOT NULL,
            manifest_json TEXT NOT NULL CHECK (json_valid(manifest_json))
        );
        CREATE INDEX IF NOT EXISTS idx_cayu_context_views_source
            ON cayu_context_views(
                source_owner_scope, source_owner_id, source_owner_incarnation,
                source_session_id, source_session_instance_id, transcript_cursor, view_id
            );
        CREATE TABLE IF NOT EXISTS cayu_context_view_selections (
            selection_key TEXT PRIMARY KEY,
            request_commitment TEXT NOT NULL,
            view_id TEXT NOT NULL,
            owner_scope TEXT NOT NULL,
            owner_id TEXT NOT NULL,
            owner_incarnation TEXT NOT NULL,
            state TEXT NOT NULL CHECK (state IN ('selected', 'adopted', 'transferred', 'released', 'expired')),
            pin_commitment TEXT NOT NULL,
            expires_at_ms INTEGER NOT NULL CHECK (expires_at_ms >= 0),
            receipt_json TEXT NOT NULL CHECK (json_valid(receipt_json))
        );
        CREATE INDEX IF NOT EXISTS idx_cayu_context_view_selections_view
            ON cayu_context_view_selections(view_id, state);
    """,
    96: SQLITE_TASK_GROUP_QUIESCENCE_DDL + SQLITE_PARTICIPANT_BINDINGS_DDL,
    92: SQLITE_TASK_GROUP_DDL,
    93: SQLITE_COLLABORATION_DDL,
    94: SQLITE_COLLABORATION_LIFECYCLE_DDL,
    95: SQLITE_COLLABORATION_REQUEST_DDL,
    91: SQLITE_TASK_GRAPH_DDL,
    90: SQLITE_SCHEDULING_DDL,
    81: """
        CREATE TABLE IF NOT EXISTS cayu_event_watcher_settlements (
            watcher_name TEXT NOT NULL,
            claim_id TEXT NOT NULL,
            receipt_json TEXT NOT NULL,
            PRIMARY KEY (watcher_name, claim_id)
        );
    """,
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
                'session.delegated_action.updated',
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
    75: """
        CREATE TABLE IF NOT EXISTS cayu_knowledge_activation_receipts (
            operation_id TEXT COLLATE BINARY PRIMARY KEY,
            entry_id TEXT COLLATE BINARY NOT NULL,
            entry_revision INTEGER NOT NULL
                CHECK (entry_revision > 0 AND entry_revision <= 2147483647),
            expected_revision INTEGER
                CHECK (expected_revision > 0 AND expected_revision <= 2147483647),
            publication_request_sha256 TEXT COLLATE BINARY NOT NULL CHECK (
                length(publication_request_sha256) = 64
                AND publication_request_sha256 NOT GLOB '*[^0-9a-f]*'
            ),
            committed_at TEXT NOT NULL,
            receipt_json TEXT NOT NULL CHECK (
                json_valid(receipt_json)
                AND json_type(receipt_json) = 'object'
                AND length(CAST(receipt_json AS BLOB)) BETWEEN 1 AND 1114112
            ),
            access_snapshot_json TEXT NOT NULL CHECK (
                json_valid(access_snapshot_json)
                AND json_type(access_snapshot_json) = 'object'
            ),
            CHECK (
                (expected_revision IS NULL AND entry_revision = 1)
                OR entry_revision = expected_revision + 1
            )
        );
        CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_activation_receipts_entry_revision
            ON cayu_knowledge_activation_receipts(entry_id, entry_revision);
        CREATE TABLE IF NOT EXISTS cayu_knowledge_activation_retirements (
            entry_id TEXT COLLATE BINARY PRIMARY KEY,
            entry_revision INTEGER NOT NULL
                CHECK (entry_revision > 0 AND entry_revision <= 2147483647),
            retired_at TEXT NOT NULL,
            retirement_json TEXT NOT NULL CHECK (
                json_valid(retirement_json)
                AND json_type(retirement_json) = 'object'
                AND length(CAST(retirement_json AS BLOB)) BETWEEN 1 AND 1048576
            )
        );
    """,
    76: """
        CREATE TABLE IF NOT EXISTS cayu_task_interrupted_continuation_claims (
            handoff_id_sha256 TEXT COLLATE BINARY NOT NULL PRIMARY KEY CHECK (
                length(handoff_id_sha256) = 64
                AND handoff_id_sha256 NOT GLOB '*[^0-9a-f]*'
            ),
            task_id TEXT COLLATE BINARY NOT NULL,
            worker_id TEXT COLLATE BINARY NOT NULL,
            claimed_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_cayu_tasks_interrupted_handoff_continuation
            ON cayu_tasks(status, created_at, id)
            WHERE worker_id IS NULL
              AND lease_expires_at IS NULL
              AND interrupted_handoff_id IS NOT NULL
              AND session_id IS NOT NULL
              AND session_instance_id IS NOT NULL
              AND status_reason IS NULL;
        CREATE UNIQUE INDEX IF NOT EXISTS idx_cayu_tasks_interrupted_handoff_generation
            ON cayu_tasks(interrupted_handoff_id)
            WHERE interrupted_handoff_id IS NOT NULL;
    """,
    77: """
        CREATE TABLE IF NOT EXISTS cayu_knowledge_maintenance_governance_routes (
            operation_id TEXT COLLATE BINARY PRIMARY KEY,
            proposal_id TEXT COLLATE BINARY NOT NULL UNIQUE,
            proposal_fingerprint TEXT COLLATE BINARY NOT NULL CHECK (
                length(proposal_fingerprint) = 64
                AND proposal_fingerprint NOT GLOB '*[^0-9a-f]*'
            ),
            request_sha256 TEXT COLLATE BINARY NOT NULL CHECK (
                length(request_sha256) = 64
                AND request_sha256 NOT GLOB '*[^0-9a-f]*'
            ),
            committed_at TEXT NOT NULL,
            receipt_json TEXT NOT NULL CHECK (
                json_valid(receipt_json)
                AND json_type(receipt_json) = 'object'
                AND length(CAST(receipt_json AS BLOB)) BETWEEN 1 AND 640000
            ),
            access_snapshot_json TEXT NOT NULL CHECK (
                json_valid(access_snapshot_json)
                AND json_type(access_snapshot_json) = 'object'
            )
        );
    """,
    78: """
        CREATE TABLE IF NOT EXISTS cayu_knowledge_semantic_watch_receipts (
            operation_id TEXT COLLATE BINARY PRIMARY KEY,
            invocation_sha256 TEXT COLLATE BINARY NOT NULL CHECK (
                length(invocation_sha256) = 64
                AND invocation_sha256 NOT GLOB '*[^0-9a-f]*'
            ),
            request_sha256 TEXT COLLATE BINARY NOT NULL CHECK (
                length(request_sha256) = 64
                AND request_sha256 NOT GLOB '*[^0-9a-f]*'
            ),
            committed_at TEXT NOT NULL,
            receipt_json TEXT NOT NULL CHECK (
                json_valid(receipt_json)
                AND json_type(receipt_json) = 'object'
                AND length(CAST(receipt_json AS BLOB)) BETWEEN 1 AND 384000
            ),
            access_scope_json TEXT NOT NULL CHECK (
                json_valid(access_scope_json)
                AND json_type(access_scope_json) = 'object'
                AND length(CAST(access_scope_json AS BLOB)) BETWEEN 1 AND 384000
            )
        );
    """,
    80: "ALTER TABLE cayu_eval_runs ADD COLUMN failure_diagnostic_json TEXT;",
    82: SQLITE_ACCOUNTING_DDL,
    89: SQLITE_AUXILIARY_ACCOUNTING_DDL,
    83: """
        CREATE INDEX IF NOT EXISTS idx_cayu_events_queue_acceptance
        ON cayu_events(session_id, json_extract(payload_json, '$.queue_id'))
        WHERE event_type = 'session.message.queued';
    """,
    84: """
        CREATE TABLE IF NOT EXISTS cayu_work_attempt_preparation_holds (
            hold_id TEXT PRIMARY KEY NOT NULL,
            task_id TEXT NOT NULL REFERENCES cayu_tasks(id) ON DELETE RESTRICT,
            request_sha256 TEXT NOT NULL CHECK (
                length(request_sha256) = 64 AND request_sha256 NOT GLOB '*[^0-9a-f]*'
            ),
            receipt_json TEXT NOT NULL CHECK (
                json_valid(receipt_json) AND json_type(receipt_json) = 'object'
                AND length(CAST(receipt_json AS BLOB)) BETWEEN 1 AND 1097728
            )
        );
        CREATE TABLE IF NOT EXISTS cayu_work_attempt_lifecycle_receipts (
            admission_id TEXT PRIMARY KEY NOT NULL
                REFERENCES cayu_work_attempt_admissions(admission_id) ON DELETE RESTRICT,
            settlement_id TEXT NOT NULL UNIQUE,
            task_id TEXT NOT NULL REFERENCES cayu_tasks(id) ON DELETE RESTRICT,
            request_sha256 TEXT NOT NULL CHECK (
                length(request_sha256) = 64 AND request_sha256 NOT GLOB '*[^0-9a-f]*'
            ),
            retired_contract_binding INTEGER NOT NULL CHECK (retired_contract_binding IN (0, 1)),
            settled_at TEXT NOT NULL,
            receipt_json TEXT NOT NULL CHECK (
                json_valid(receipt_json) AND json_type(receipt_json) = 'object'
                AND length(CAST(receipt_json AS BLOB)) BETWEEN 1 AND 1097728
            )
        );
        CREATE INDEX IF NOT EXISTS idx_cayu_work_attempt_lifecycle_task
            ON cayu_work_attempt_lifecycle_receipts(task_id, retired_contract_binding);
    """,
    86: """CREATE INDEX IF NOT EXISTS idx_cayu_side_effect_health
 ON cayu_persisted_event_side_effects(status, next_attempt_at, lease_expires_at, updated_at, attempts);
CREATE INDEX IF NOT EXISTS idx_cayu_side_effect_outstanding
 ON cayu_persisted_event_side_effects(session_id, event_id) WHERE status <> 'delivered';""",
    85: """
        CREATE TABLE IF NOT EXISTS cayu_session_closure_receipts (
            session_id TEXT COLLATE BINARY NOT NULL,
            plan_id TEXT COLLATE BINARY NOT NULL CHECK (
                length(plan_id) = 64 AND plan_id NOT GLOB '*[^0-9a-f]*'
            ),
            committed_at TEXT NOT NULL,
            receipt_json TEXT NOT NULL CHECK (
                json_valid(receipt_json)
                AND json_type(receipt_json) = 'object'
                AND length(CAST(receipt_json AS BLOB)) BETWEEN 1 AND 384000
            ),
            PRIMARY KEY (session_id, plan_id)
        );
    """,
    87: """
        CREATE TABLE IF NOT EXISTS cayu_session_closure_tombstones (
            root_session_id TEXT COLLATE BINARY NOT NULL,
            plan_id TEXT COLLATE BINARY NOT NULL CHECK (
                length(plan_id) = 64 AND plan_id NOT GLOB '*[^0-9a-f]*'
            ),
            child_session_id TEXT COLLATE BINARY NOT NULL,
            original_parent_session_id TEXT COLLATE BINARY NOT NULL,
            detached_at TEXT NOT NULL,
            tombstone_json TEXT NOT NULL CHECK (
                json_valid(tombstone_json)
                AND json_type(tombstone_json) = 'object'
                AND length(CAST(tombstone_json AS BLOB)) BETWEEN 1 AND 32768
            ),
            PRIMARY KEY (root_session_id, plan_id, child_session_id)
        );
    """,
    88: """
        CREATE TABLE IF NOT EXISTS cayu_task_session_closure_claims (
            session_id TEXT COLLATE BINARY PRIMARY KEY,
            plan_id TEXT COLLATE BINARY NOT NULL CHECK (
                length(plan_id) = 64 AND plan_id NOT GLOB '*[^0-9a-f]*'
            ),
            claim_json TEXT NOT NULL CHECK (
                json_valid(claim_json) AND json_type(claim_json) = 'object'
                AND length(CAST(claim_json AS BLOB)) BETWEEN 1 AND 16777216
            )
        );
        CREATE TRIGGER IF NOT EXISTS cayu_task_closure_insert_guard
        BEFORE INSERT ON cayu_tasks
        WHEN EXISTS (
            SELECT 1 FROM cayu_task_session_closure_claims WHERE session_id = NEW.session_id
        )
        BEGIN
            SELECT RAISE(ABORT, 'Task session is owned by closure.');
        END;
        CREATE TRIGGER IF NOT EXISTS cayu_task_closure_update_guard
        BEFORE UPDATE ON cayu_tasks
        WHEN EXISTS (
            SELECT 1 FROM cayu_task_session_closure_claims
            WHERE session_id IN (OLD.session_id, NEW.session_id)
        )
        BEGIN
            SELECT RAISE(ABORT, 'Task session is owned by closure.');
        END;
        CREATE TABLE IF NOT EXISTS cayu_session_closure_progress (
            root_session_id TEXT COLLATE BINARY NOT NULL,
            plan_id TEXT COLLATE BINARY NOT NULL CHECK (
                length(plan_id) = 64 AND plan_id NOT GLOB '*[^0-9a-f]*'
            ),
            progress_json TEXT NOT NULL CHECK (
                json_valid(progress_json) AND json_type(progress_json) = 'object'
                AND length(CAST(progress_json AS BLOB)) BETWEEN 1 AND 384000
            ),
            PRIMARY KEY (root_session_id, plan_id)
        );
    """,
    79: """
        CREATE TABLE IF NOT EXISTS cayu_child_session_lifecycle_candidates (
            child_session_id TEXT COLLATE BINARY PRIMARY KEY
                REFERENCES cayu_sessions(id) ON DELETE CASCADE,
            parent_session_id TEXT COLLATE BINARY NOT NULL
                REFERENCES cayu_sessions(id) ON DELETE CASCADE,
            priority INTEGER NOT NULL CHECK (priority IN (0, 1, 2)),
            sort_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_cayu_child_lifecycle_candidates_page
            ON cayu_child_session_lifecycle_candidates(
                parent_session_id, priority, sort_at, child_session_id
            );
        CREATE INDEX IF NOT EXISTS idx_cayu_events_child_lifecycle
            ON cayu_events(session_id, event_type, sequence DESC)
            WHERE event_type IN (
                'session.started', 'session.resumed', 'session.forked',
                'session.completed', 'session.failed', 'session.interrupted'
            );
        CREATE INDEX IF NOT EXISTS idx_cayu_transcript_messages_session_role_order
            ON cayu_transcript_messages(session_id, role, session_order DESC);

        CREATE VIEW IF NOT EXISTS cayu_child_session_lifecycle_canonical AS
        WITH boundaries AS (
            SELECT
                child.id AS child_session_id,
                child.parent_session_id,
                child.instance_id,
                child.status,
                child.created_at,
                (
                    SELECT event.sequence
                    FROM cayu_events AS event
                    WHERE event.session_id = child.id
                      AND event.event_type IN (
                          'session.started', 'session.resumed', 'session.forked',
                          'session.completed', 'session.failed', 'session.interrupted'
                      )
                    ORDER BY event.sequence DESC
                    LIMIT 1
                ) AS latest_lifecycle_sequence,
                EXISTS (
                    SELECT 1
                    FROM cayu_events AS event
                    WHERE event.session_id = child.id
                      AND event.event_type IN (
                          'session.started', 'session.resumed',
                          'session.completed', 'session.failed', 'session.interrupted'
                      )
                ) AS pending_has_forbidden_event
            FROM cayu_sessions AS child
            WHERE child.parent_session_id IS NOT NULL
        ), canonical AS (
            SELECT
                boundary.*,
                latest.event_id AS latest_event_id,
                latest.event_type AS latest_event_type,
                latest.timestamp AS latest_event_at,
                CASE
                    WHEN boundary.status = 'pending' THEN
                        NOT boundary.pending_has_forbidden_event
                    WHEN boundary.status IN ('running', 'interrupting') THEN
                        latest.event_type IN (
                            'session.started', 'session.resumed', 'session.forked'
                        )
                    WHEN boundary.status = 'completed' THEN
                        latest.event_type = 'session.completed'
                    WHEN boundary.status = 'failed' THEN
                        latest.event_type = 'session.failed'
                    WHEN boundary.status = 'interrupted' THEN
                        latest.event_type = 'session.interrupted'
                    ELSE 0
                END AS is_available
            FROM boundaries AS boundary
            LEFT JOIN cayu_events AS latest
              ON latest.sequence = boundary.latest_lifecycle_sequence
        )
        SELECT
            canonical.child_session_id,
            canonical.parent_session_id,
            CASE
                WHEN canonical.is_available
                 AND canonical.status IN ('completed', 'failed', 'interrupted')
                 AND NOT EXISTS (
                     SELECT 1
                     FROM cayu_session_operations AS consumption
                     WHERE consumption.session_id = canonical.parent_session_id
                       AND consumption.idempotency_key =
                           '__cayu_child_session_notification_v1__:' ||
                           length(canonical.instance_id) || ':' ||
                           canonical.instance_id || canonical.latest_event_id
                 ) THEN 0
                WHEN canonical.is_available
                 AND canonical.status IN ('completed', 'failed', 'interrupted') THEN 2
                ELSE 1
            END AS priority,
            CASE
                WHEN canonical.is_available
                 AND canonical.latest_event_at IS NOT NULL
                    THEN canonical.latest_event_at
                ELSE canonical.created_at
            END AS sort_at
        FROM canonical;

        CREATE TRIGGER IF NOT EXISTS cayu_index_child_lifecycle_session_insert
        AFTER INSERT ON cayu_sessions
        FOR EACH ROW
        WHEN NEW.parent_session_id IS NOT NULL
        BEGIN
            INSERT INTO cayu_child_session_lifecycle_candidates (
                child_session_id, parent_session_id, priority, sort_at
            )
            SELECT child_session_id, parent_session_id, priority, sort_at
            FROM cayu_child_session_lifecycle_canonical
            WHERE child_session_id = NEW.id;
        END;

        CREATE TRIGGER IF NOT EXISTS cayu_index_child_lifecycle_session_update
        AFTER UPDATE OF parent_session_id, instance_id, status, created_at ON cayu_sessions
        FOR EACH ROW
        BEGIN
            DELETE FROM cayu_child_session_lifecycle_candidates
            WHERE child_session_id = NEW.id;
            INSERT INTO cayu_child_session_lifecycle_candidates (
                child_session_id, parent_session_id, priority, sort_at
            )
            SELECT child_session_id, parent_session_id, priority, sort_at
            FROM cayu_child_session_lifecycle_canonical
            WHERE child_session_id = NEW.id;
        END;

        CREATE TRIGGER IF NOT EXISTS cayu_index_child_lifecycle_event_insert
        AFTER INSERT ON cayu_events
        FOR EACH ROW
        WHEN NEW.event_type IN (
            'session.started', 'session.resumed', 'session.forked',
            'session.completed', 'session.failed', 'session.interrupted'
        )
        BEGIN
            DELETE FROM cayu_child_session_lifecycle_candidates
            WHERE child_session_id = NEW.session_id;
            INSERT INTO cayu_child_session_lifecycle_candidates (
                child_session_id, parent_session_id, priority, sort_at
            )
            SELECT child_session_id, parent_session_id, priority, sort_at
            FROM cayu_child_session_lifecycle_canonical
            WHERE child_session_id = NEW.session_id;
        END;

        CREATE TRIGGER IF NOT EXISTS cayu_index_child_lifecycle_event_delete
        AFTER DELETE ON cayu_events
        FOR EACH ROW
        WHEN OLD.event_type IN (
            'session.started', 'session.resumed', 'session.forked',
            'session.completed', 'session.failed', 'session.interrupted'
        )
        BEGIN
            DELETE FROM cayu_child_session_lifecycle_candidates
            WHERE child_session_id = OLD.session_id;
            INSERT INTO cayu_child_session_lifecycle_candidates (
                child_session_id, parent_session_id, priority, sort_at
            )
            SELECT child_session_id, parent_session_id, priority, sort_at
            FROM cayu_child_session_lifecycle_canonical
            WHERE child_session_id = OLD.session_id;
        END;

        CREATE TRIGGER IF NOT EXISTS cayu_index_child_lifecycle_event_update
        AFTER UPDATE OF session_id, event_id, event_type, sequence, timestamp ON cayu_events
        FOR EACH ROW
        WHEN OLD.event_type IN (
            'session.started', 'session.resumed', 'session.forked',
            'session.completed', 'session.failed', 'session.interrupted'
        ) OR NEW.event_type IN (
            'session.started', 'session.resumed', 'session.forked',
            'session.completed', 'session.failed', 'session.interrupted'
        )
        BEGIN
            DELETE FROM cayu_child_session_lifecycle_candidates
            WHERE child_session_id IN (OLD.session_id, NEW.session_id);
            INSERT INTO cayu_child_session_lifecycle_candidates (
                child_session_id, parent_session_id, priority, sort_at
            )
            SELECT child_session_id, parent_session_id, priority, sort_at
            FROM cayu_child_session_lifecycle_canonical
            WHERE child_session_id IN (OLD.session_id, NEW.session_id);
        END;

        CREATE TRIGGER IF NOT EXISTS cayu_index_child_lifecycle_consumption
        AFTER INSERT ON cayu_session_operations
        FOR EACH ROW
        WHEN json_extract(NEW.record_json, '$.record_type') =
             'cayu.child-session-notification-consumption'
        BEGIN
            DELETE FROM cayu_child_session_lifecycle_candidates
            WHERE child_session_id = json_extract(NEW.record_json, '$.child_session_id');
            INSERT INTO cayu_child_session_lifecycle_candidates (
                child_session_id, parent_session_id, priority, sort_at
            )
            SELECT child_session_id, parent_session_id, priority, sort_at
            FROM cayu_child_session_lifecycle_canonical
            WHERE child_session_id = json_extract(
                NEW.record_json, '$.child_session_id'
            );
        END;

        CREATE TRIGGER IF NOT EXISTS cayu_index_child_lifecycle_consumption_delete
        AFTER DELETE ON cayu_session_operations
        FOR EACH ROW
        WHEN json_extract(OLD.record_json, '$.record_type') =
             'cayu.child-session-notification-consumption'
        BEGIN
            DELETE FROM cayu_child_session_lifecycle_candidates
            WHERE child_session_id = json_extract(OLD.record_json, '$.child_session_id');
            INSERT INTO cayu_child_session_lifecycle_candidates (
                child_session_id, parent_session_id, priority, sort_at
            )
            SELECT child_session_id, parent_session_id, priority, sort_at
            FROM cayu_child_session_lifecycle_canonical
            WHERE child_session_id = json_extract(
                OLD.record_json, '$.child_session_id'
            );
        END;

        CREATE TRIGGER IF NOT EXISTS cayu_index_child_lifecycle_consumption_update
        AFTER UPDATE OF session_id, idempotency_key, record_json
        ON cayu_session_operations
        FOR EACH ROW
        WHEN json_extract(OLD.record_json, '$.record_type') =
                 'cayu.child-session-notification-consumption'
          OR json_extract(NEW.record_json, '$.record_type') =
                 'cayu.child-session-notification-consumption'
        BEGIN
            DELETE FROM cayu_child_session_lifecycle_candidates
            WHERE child_session_id IN (
                json_extract(OLD.record_json, '$.child_session_id'),
                json_extract(NEW.record_json, '$.child_session_id')
            );
            INSERT INTO cayu_child_session_lifecycle_candidates (
                child_session_id, parent_session_id, priority, sort_at
            )
            SELECT child_session_id, parent_session_id, priority, sort_at
            FROM cayu_child_session_lifecycle_canonical
            WHERE child_session_id IN (
                json_extract(OLD.record_json, '$.child_session_id'),
                json_extract(NEW.record_json, '$.child_session_id')
            );
        END;

        INSERT INTO cayu_child_session_lifecycle_candidates (
            child_session_id, parent_session_id, priority, sort_at
        )
        SELECT child_session_id, parent_session_id, priority, sort_at
        FROM cayu_child_session_lifecycle_canonical
        WHERE 1
        ON CONFLICT(child_session_id) DO UPDATE SET
            parent_session_id = excluded.parent_session_id,
            priority = excluded.priority,
            sort_at = excluded.sort_at;
    """,
}

# Per-revision ``ALTER TABLE ADD COLUMN`` steps, keyed by revision. SQLite has no
# ``ADD COLUMN IF NOT EXISTS``, so these are applied via _add_column_if_missing
# (a table_info existence check) rather than raw DDL, making a re-run after a
# crash a no-op instead of a "duplicate column name" error that wedges migrate.
# They run before the revision's _MIGRATION_STEPS DDL so indexes on the new
# columns are created only after the columns exist.
_MIGRATION_ADD_COLUMNS: dict[int, tuple[tuple[str, str, str], ...]] = {
    101: (("cayu_budget_bindings", "allowance", "INTEGER"),),
    96: (
        ("cayu_task_groups", "barrier_status", "TEXT NOT NULL DEFAULT 'not_requested'"),
        ("cayu_task_groups", "barrier_deadline", "TEXT"),
    ),
    98: (("cayu_context_view_selections", "ownership_revision", "INTEGER NOT NULL DEFAULT 1"),),
    91: (
        ("cayu_tasks", "graph_id", "TEXT"),
        ("cayu_tasks", "prerequisite_task_ids_json", "TEXT NOT NULL DEFAULT '[]'"),
    ),
    90: (
        (
            "cayu_tasks",
            "schedule_json",
            "TEXT CHECK (schedule_json IS NULL OR "
            "(json_valid(schedule_json) AND json_type(schedule_json) = 'object' "
            "AND length(CAST(schedule_json AS BLOB)) BETWEEN 1 AND 32768))",
        ),
    ),
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
    83: (
        ("cayu_session_message_queue", "conditions_json", "TEXT"),
        ("cayu_session_message_queue", "terminal_json", "TEXT"),
        (
            "cayu_session_message_deliveries",
            "reject_only",
            "INTEGER NOT NULL DEFAULT 0 CHECK (reject_only IN (0, 1))",
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
    76: (("cayu_tasks", "interrupted_handoff_id", "TEXT"),),
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


_EMPTY_RECALL_RESET_TABLES = (
    "cayu_agent_recall_delivery_states",
    "cayu_agent_recall_delivery_releases",
    "cayu_agent_recall_delivery_claims",
    "cayu_agent_recall_deliveries",
    "cayu_agent_recall_checkpoint_heads",
    "cayu_agent_recall_checkpoints",
)


def preflight_empty_recall_state_reset(connection: sqlite3.Connection) -> None:
    """Prove the prerelease recall tables can be rebuilt without losing rows."""

    for table in _EMPTY_RECALL_RESET_TABLES:
        registered = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        if registered is None:
            continue
        if connection.execute(f"SELECT EXISTS(SELECT 1 FROM {table})").fetchone()[0]:
            raise schema.SchemaTooOld(
                "Storage revision 73 can rebuild prerelease recall state only when all "
                f"six checkpoint/delivery tables are empty; {table!r} is populated."
            )


def reset_empty_recall_state(connection: sqlite3.Connection) -> None:
    """Rebuild empty revision-69/71 recall tables from this Runtime's DDL."""

    preflight_empty_recall_state_reset(connection)
    with sqlite_connection._transaction(connection):
        for table in _EMPTY_RECALL_RESET_TABLES:
            connection.execute(f"DROP TABLE IF EXISTS {table}")
        for revision in (69, 71):
            for statement in _iter_statements(_MIGRATION_STEPS[revision]):
                connection.execute(statement)
        sqlite_work_context_schema._validate_revision_69_work_context_schema(connection)
        sqlite_work_context_schema._validate_revision_71_recall_delivery_schema(connection)


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
_KNOWLEDGE_ACTIVATION_CLEAN_BREAK_TABLES = (
    *_KNOWLEDGE_MAINTENANCE_CLEAN_BREAK_TABLES,
    "cayu_knowledge_activation_receipts",
    "cayu_knowledge_activation_retirements",
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


def _reject_populated_pre_knowledge_activation_database(
    connection: sqlite3.Connection,
) -> None:
    _reject_populated_pre_knowledge_contract_database(
        connection,
        candidates=_KNOWLEDGE_ACTIVATION_CLEAN_BREAK_TABLES,
        revision=75,
        contract="knowledge-activation-authority",
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


def _backfill_session_activity(connection: sqlite3.Connection) -> None:
    connection.execute("UPDATE cayu_sessions SET last_activity_at = updated_at")


def _backfill_pending_action_checkpoint_batch(
    connection: sqlite3.Connection,
    after_session_id: str | None,
) -> str | None:
    from cayu.sessions.pending_actions import pending_action_checkpoint_metrics

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
    from cayu.sessions.pending_actions import (
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
            timestamp=sqlite_records.parse_datetime(row["timestamp"]),
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
    if not sqlite_catalog._sqlite_has_unique_index(connection, "cayu_knowledge_chunks", ("id",)):
        raise RuntimeError("SQLite knowledge chunks are missing their unique public id constraint.")
    if not sqlite_catalog._sqlite_has_unique_index(
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
    columns = sqlite_catalog._sqlite_table_columns(connection, "cayu_knowledge_chunks")
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
                sqlite_records.json_dumps(deferred_interaction_input_storage_payload(stable)),
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
            (sqlite_records.json_dumps(migrated_admission), row["admission_id"]),
        )


def _migrate_revision_sixty_two_payloads(connection: sqlite3.Connection) -> None:
    _migrate_deferred_interaction_input_payloads(connection)
    _migrate_work_attempt_continuation_authority(connection)


def _backfill_interrupted_handoff_generations(connection: sqlite3.Connection) -> None:
    """Carry unambiguous revision-70 handoff authority into revision 76."""

    cursor: tuple[str, str, str] | None = None
    active_task_id: str | None = None
    active_current_task: Task | None = None
    active_matching_generations: list[str] = []

    def finalize_active_task() -> None:
        if active_task_id is None or not active_matching_generations:
            return
        if len(active_matching_generations) != 1:
            raise RuntimeError(
                "SQLite revision-76 migration cannot determine one interrupted-task "
                f"handoff generation for task {active_task_id!r}. Resolve the ambiguous "
                "recovery receipts before migrating."
            )
        connection.execute(
            "UPDATE cayu_tasks SET interrupted_handoff_id = ? WHERE id = ?",
            (active_matching_generations[0], active_task_id),
        )

    while True:
        after_sql = ""
        after_params: tuple[object, ...] = ()
        if cursor is not None:
            after_sql = "WHERE (task_id, committed_at, handoff_id) > (?, ?, ?)"
            after_params = cursor
        receipts = connection.execute(
            f"""
            SELECT task_id, handoff_id, request_sha256, request_json, task_json,
                   committed_at
            FROM cayu_task_interrupted_handoff_receipts
            {after_sql}
            ORDER BY task_id, committed_at, handoff_id
            LIMIT ?
            """,
            (*after_params, _INTERRUPTED_HANDOFF_MIGRATION_BATCH_SIZE),
        ).fetchall()
        if not receipts:
            break
        task_ids = list(dict.fromkeys(str(row["task_id"]) for row in receipts))
        placeholders = ", ".join("?" for _ in task_ids)
        current_tasks = {
            task.id: task
            for task in (
                sqlite_records.task_from_row(row)
                for row in connection.execute(
                    f"SELECT * FROM cayu_tasks WHERE id IN ({placeholders})",
                    task_ids,
                )
            )
        }
        receipt_updates: list[tuple[str, str, str]] = []
        for receipt_row in receipts:
            task_id = str(receipt_row["task_id"])
            if task_id != active_task_id:
                finalize_active_task()
                active_task_id = task_id
                active_current_task = current_tasks.get(task_id)
                active_matching_generations = []
            try:
                request = TaskInterruptedHandoffRequest.model_validate(
                    json.loads(receipt_row["request_json"])
                )
                request, request_sha256 = prepare_interrupted_task_handoff(request)
                receipt_task = Task.model_validate(json.loads(receipt_row["task_json"]))
                if (
                    request.task_id != task_id
                    or request.handoff_id != receipt_row["handoff_id"]
                    or request_sha256 != receipt_row["request_sha256"]
                    or receipt_task.id != request.task_id
                    or receipt_task.status is not TaskStatus.RUNNING
                    or receipt_task.session_id != request.session_id
                    or receipt_task.session_instance_id != request.session_instance_id
                    or receipt_task.worker_id is not None
                    or receipt_task.lease_expires_at is not None
                    or receipt_task.interrupted_handoff_id is not None
                ):
                    raise ValueError("receipt conflicts with its pre-76 handoff authority")
            except Exception as exc:
                raise RuntimeError(
                    "SQLite revision-76 migration found malformed interrupted-task "
                    f"handoff authority for task {task_id!r}. Restore the database "
                    "from known-good recovery evidence."
                ) from exc
            upgraded_task = receipt_task.model_copy(
                update={"interrupted_handoff_id": request.handoff_id},
                deep=True,
            )
            upgraded_task = Task.model_validate(upgraded_task.model_dump(mode="python"))
            receipt_updates.append(
                (
                    sqlite_records.json_dumps(
                        upgraded_task.model_dump(mode="json", warnings=False)
                    ),
                    task_id,
                    request.handoff_id,
                )
            )
            if active_current_task == receipt_task:
                active_matching_generations.append(request.handoff_id)
        connection.executemany(
            """
            UPDATE cayu_task_interrupted_handoff_receipts
            SET task_json = ?
            WHERE task_id = ? AND handoff_id = ?
            """,
            receipt_updates,
        )
        last = receipts[-1]
        cursor = (
            str(last["task_id"]),
            str(last["committed_at"]),
            str(last["handoff_id"]),
        )
    finalize_active_task()


def _upgrade_continuation_indexes(connection: sqlite3.Connection) -> None:
    from cayu.storage._continuation_index_migration import migrate_sqlite_continuation_indexes

    migrate_sqlite_continuation_indexes(connection)


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
    76: _backfill_interrupted_handoff_generations,
    115: _upgrade_continuation_indexes,
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
    with sqlite_connection._transaction(connection):
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
    with sqlite_connection._transaction(connection):
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


def _validate_producer_cleanup_receipts(connection: sqlite3.Connection) -> None:
    names = (
        ("cayu_producer_cleanup_receipts", "table"),
        ("idx_cayu_producer_cleanup_namespace", "index"),
        ("cayu_producer_cleanup_retirements", "table"),
    )
    for (name, kind), expected in zip(names, _iter_statements(_MIGRATION_STEPS[110]), strict=True):
        row = connection.execute(
            "SELECT type, sql FROM sqlite_master WHERE name = ?", (name,)
        ).fetchone()
        if (
            row is None
            or row[0] != kind
            or row[1] is None
            or _normalize_sqlite_schema_definition(row[1])
            != _normalize_sqlite_schema_definition(expected)
        ):
            raise RuntimeError(
                "Required Cayu producer cleanup receipt table or fence is missing or conflicting."
            )


def _validate_reservation_inventory_index(connection: sqlite3.Connection) -> None:
    name = "idx_cayu_budget_reservations_session_identity"
    row = connection.execute(
        "SELECT type, tbl_name, sql FROM sqlite_master WHERE name = ?", (name,)
    ).fetchone()
    expected = next(_iter_statements(_MIGRATION_STEPS[109]))
    if (
        row is None
        or row[0] != "index"
        or row[1] != "cayu_budget_reservations"
        or row[2] is None
        or _normalize_sqlite_schema_definition(row[2])
        != _normalize_sqlite_schema_definition(expected)
    ):
        raise RuntimeError("Required Cayu reservation inventory index is missing or conflicting.")


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
    with sqlite_connection._transaction(connection):
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
    with sqlite_connection._transaction(connection):
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
    if current_diagnostic_store_inspection() is not None and not sqlite_connection._is_in_memory(
        connection
    ):
        schema_mode = schema.SchemaMode.VALIDATE
    state = read_schema_state(connection)
    if schema_mode is schema.SchemaMode.MIGRATE:
        schema.validate_migration_input(state)
        # Freeze every clean-break decision before even migration-bookkeeping
        # DDL. Individual revision transactions repeat the relevant checks to
        # close races with legacy writers.
        preflight_migration(connection, state)
    elif schema_mode is schema.SchemaMode.CREATE:
        _preflight_creation(connection, state)
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
    if current.revision >= 108:
        validate_sqlite_context_selection_schema(connection)
    if current.revision >= 96:
        validate_sqlite_participant_bindings(connection)
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
        sqlite_session_schema._validate_session_invocation_column(connection)
    if 37 <= current.revision < 42:
        # Structural validation is intentionally constant-size. The full source/
        # FTS census belongs to the one-time revision hook, never ordinary startup.
        _validate_revision_37_knowledge_fts_schema(connection)
    if current.revision >= 42:
        sqlite_knowledge_schema._validate_revision_42_knowledge_schema(
            connection,
            require_payload_bytes=current.revision >= 65,
        )
    if current.revision >= 43:
        sqlite_knowledge_schema._validate_revision_43_knowledge_schema(
            connection,
            relation_aware=current.revision >= 60,
        )
    if current.revision >= 44:
        sqlite_knowledge_schema._validate_revision_44_knowledge_schema(connection)
    if current.revision >= 60:
        sqlite_knowledge_schema._validate_revision_60_knowledge_schema(connection)
    if current.revision >= 63:
        sqlite_knowledge_schema._validate_revision_63_knowledge_schema(connection)
    if current.revision >= 67:
        sqlite_knowledge_schema._validate_revision_67_knowledge_schema(connection)
    if current.revision >= 69:
        sqlite_work_context_schema._validate_revision_69_work_context_schema(connection)
    if current.revision >= 71:
        sqlite_work_context_schema._validate_revision_71_recall_delivery_schema(
            connection,
            require_processing_schema_version=current.revision >= 73,
        )
    if current.revision >= 73:
        sqlite_work_context_schema._validate_revision_73_recall_subscription_schema(connection)
    if current.revision >= 75:
        sqlite_knowledge_schema._validate_revision_75_knowledge_activation_schema(connection)
    if current.revision >= 77:
        sqlite_knowledge_schema._validate_revision_77_knowledge_maintenance_governance_schema(
            connection
        )
    if current.revision >= 78:
        sqlite_knowledge_schema._validate_revision_78_knowledge_semantic_watch_schema(connection)
    if current.revision >= 79:
        sqlite_session_schema._validate_revision_79_child_lifecycle_schema(connection)
    if current.revision >= 88:
        _validate_revision_88_closure_schema(connection)
    if current.revision >= 93:
        validate_sqlite_collaboration_schema(
            connection,
            lifecycle=current.revision >= 94,
            requests=current.revision >= 95,
            clarifications=current.revision >= 105,
            planning=current.revision >= 107,
        )
    if app_min_supported >= 38:
        sqlite_task_schema._validate_task_terminalization_receipt_table(connection)
    if app_min_supported >= 70:
        sqlite_task_schema._validate_interrupted_task_handoff_schema(connection)
    if current.revision >= 76:
        sqlite_task_schema._validate_interrupted_handoff_generation_column(connection)
    if current.revision >= 109:
        _validate_reservation_inventory_index(connection)
    if current.revision >= 110:
        _validate_producer_cleanup_receipts(connection)
    if current.revision >= 111:
        validate_sqlite_wait_discovery(connection)
    if current.revision >= 112:
        validate_sqlite_product_operation_schema(connection)
    if app_min_supported >= 39:
        sqlite_task_schema._validate_task_invocation_column(connection)
    if app_min_supported >= 41:
        sqlite_knowledge_schema._validate_knowledge_publication_access_snapshot_column(connection)
    if app_min_supported >= 45:
        sqlite_task_schema._validate_task_retry_series_schema(connection)
    if app_min_supported >= 46:
        _validate_revision_46_transcript_search_schema(connection)
    if app_min_supported >= 47:
        sqlite_eval_schema._validate_eval_result_baseline_schema(connection)
    if app_min_supported >= 48:
        sqlite_eval_schema._validate_captured_eval_case_schema(connection)
    if app_min_supported >= 49:
        sqlite_verified_work_schema._validate_verified_work_schema(
            connection,
            require_verifier_profiles=current.revision >= 58,
        )
    if app_min_supported >= 50:
        sqlite_eval_schema._validate_eval_run_invocation_column(connection)
    if app_min_supported >= 51:
        sqlite_memory_evidence_schema._validate_memory_evidence_schema(connection)
    if app_min_supported >= 52:
        sqlite_session_schema._validate_targeted_tool_grant_schema(connection)
    if app_min_supported >= 53:
        sqlite_eval_schema._validate_eval_scenario_schema(connection)
    if app_min_supported >= 55:
        sqlite_task_schema._validate_task_retry_reconciliation_schema(connection)
    if app_min_supported >= 56:
        sqlite_eval_schema._validate_eval_run_scenario_progress_column(connection)
    if app_min_supported >= 57:
        sqlite_session_schema._validate_session_message_queue_typed_message_column(connection)
    if app_min_supported >= 83:
        sqlite_session_schema._validate_session_message_lifecycle_columns(connection)
    if app_min_supported >= 59:
        sqlite_session_schema._validate_session_instance_schema(connection)
    if app_min_supported >= 61:
        sqlite_verified_work_schema._validate_work_attempt_admission_schema(connection)
    if app_min_supported >= 84:
        sqlite_verified_work_schema._validate_work_attempt_lifecycle_schema(connection)
    if app_min_supported >= 62:
        # The revision hook performs the one-time complete payload census.
        # Ordinary startup remains independent of durable history size; each
        # payload is validated again at its indexed read boundary.
        sqlite_session_schema._validate_revision_sixty_two_payload_schema(connection)
    if app_min_supported >= 64:
        sqlite_eval_schema._validate_eval_authored_suite_schema(connection)
    if app_min_supported >= 66:
        _validate_local_execution_attempt_schema(connection)
    if app_min_supported >= 68:
        sqlite_eval_schema._validate_eval_judge_calibration_schema(connection)
    if app_min_supported >= 72:
        sqlite_eval_schema._validate_eval_run_max_concurrency_schema(connection)
    if app_min_supported >= 74:
        sqlite_eval_schema._validate_eval_run_trial_checkpoint_schema(connection)


def _validate_revision_88_closure_schema(connection: sqlite3.Connection) -> None:
    for statement in _iter_statements(_MIGRATION_STEPS[88]):
        match = re.match(r"CREATE (TABLE|TRIGGER) IF NOT EXISTS (\w+)", statement)
        if match is None:
            raise RuntimeError("Unrecognized closure schema definition.")
        object_type, name = match.groups()
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = ? AND name = ?",
            (object_type.lower(), name),
        ).fetchone()
        if row is None or _normalize_sqlite_schema_definition(row[0]) != (
            _normalize_sqlite_schema_definition(statement)
        ):
            raise RuntimeError("SQLite closure schema is missing or conflicts with its contract.")


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
    with sqlite_connection._transaction(connection):
        for statement in _iter_statements(_BASELINE_DDL):
            connection.execute(statement)
        _record_revision(connection, schema.revision(schema.BASELINE_REVISION))
        # user_version mirrors the revision as a cheap SQLite-native marker; the
        # cayu_schema_migrations table remains the cross-backend source of truth.
        connection.execute(f"PRAGMA user_version = {schema.BASELINE_REVISION}")


def _apply_pending(connection: sqlite3.Connection, state: schema.SchemaState) -> None:
    preflight_migration(connection, state)
    _apply_pending_after_preflight(connection, state)


def preflight_migration(
    connection: sqlite3.Connection,
    state: schema.SchemaState | None = None,
    *,
    allow_empty_recall_reset: bool = False,
) -> schema.SchemaState:
    """Validate every SQLite clean break without performing schema DDL.

    The CLI invokes this on a read-only connection before it prepares or
    publishes a migration. The migration engine repeats it so direct store
    users retain the same fail-before-bookkeeping contract.
    """

    if state is None:
        state = read_schema_state(connection)
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
    if current < 84 and any(revision.revision == 84 for revision in schema.pending(current)):
        _reject_pre_worker_admission_history(connection)
    if current < 60 and any(revision.revision == 60 for revision in schema.pending(current)):
        _reject_populated_pre_knowledge_relation_database(connection)
    if current < 63 and any(revision.revision == 63 for revision in schema.pending(current)):
        _reject_populated_pre_knowledge_maintenance_database(connection)
    if current < 65 and any(revision.revision == 65 for revision in schema.pending(current)):
        _reject_populated_pre_bounded_knowledge_entry_database(connection)
    if (
        current != schema.UNINITIALIZED
        and current < 73
        and any(revision.revision == 73 for revision in schema.pending(current))
    ):
        if allow_empty_recall_reset:
            preflight_empty_recall_state_reset(connection)
        else:
            _reject_populated_pre_recall_subscription_database(connection)
    if (
        current != schema.UNINITIALIZED
        and current < 75
        and any(revision.revision == 75 for revision in schema.pending(current))
    ):
        _reject_populated_pre_knowledge_activation_database(connection)
    return state


def _preflight_creation(
    connection: sqlite3.Connection,
    state: schema.SchemaState,
) -> None:
    """Preserve create-mode's narrower legacy checks without planning a migration."""

    current = state.revision
    planned = schema.pending(current)
    if current < 42 and any(revision.revision == 42 for revision in planned):
        _reject_populated_pre_knowledge_revision_database(connection)
    if current < 60 and any(revision.revision == 60 for revision in planned):
        _reject_populated_pre_knowledge_relation_database(connection)
    if current < 63 and any(revision.revision == 63 for revision in planned):
        _reject_populated_pre_knowledge_maintenance_database(connection)
    if current < 65 and any(revision.revision == 65 for revision in planned):
        _reject_populated_pre_bounded_knowledge_entry_database(connection)
    if current == schema.UNINITIALIZED and any(revision.revision == 46 for revision in planned):
        _reject_populated_pre_transcript_search_database(connection)
    if current < 59 and any(revision.revision == 59 for revision in planned):
        _reject_populated_pre_result_resolver_database(connection)
    if current < 84 and any(revision.revision == 84 for revision in planned):
        _reject_pre_worker_admission_history(connection)


def _reject_pre_worker_admission_history(connection: sqlite3.Connection) -> None:
    for table, authority in (
        ("cayu_work_attempt_admissions", "work-attempt admissions"),
        ("cayu_completion_verification_claims", "verification claims"),
    ):
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
        ).fetchone()
        if (
            exists is not None
            and connection.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone() is not None
        ):
            raise RuntimeError(
                "SQLite revision 84 cannot reconstruct executable settings for existing "
                f"{authority}. Recreate the pre-release database before migrating."
            )


def _apply_pending_after_preflight(
    connection: sqlite3.Connection,
    state: schema.SchemaState,
) -> None:
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
    if rev.revision == 23:
        with sqlite_connection._transaction(connection):
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
        with sqlite_connection._transaction(connection):
            _validate_workflow_replay_indexes(connection, require_all=False)
            for statement in _iter_statements(_MIGRATION_STEPS[29]):
                connection.execute(statement)
            _validate_workflow_replay_indexes(connection, require_all=True)
            _record_revision(connection, rev)
            connection.execute(f"PRAGMA user_version = {rev.revision}")
        return
    if rev.revision == 38:
        with sqlite_connection._transaction(connection):
            for statement in _iter_statements(_MIGRATION_STEPS[38]):
                connection.execute(statement)
            sqlite_task_schema._validate_task_terminalization_receipt_table(connection)
            _record_revision(connection, rev)
            connection.execute(f"PRAGMA user_version = {rev.revision}")
        return
    if rev.revision == 72:
        _apply_revision_seventy_two(connection, rev)
        return
    with sqlite_connection._transaction(connection):
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
        if rev.revision == 75:
            # BEGIN IMMEDIATE fences pre-75 writers between the clean-break
            # check and installation of exact activation authority.
            _reject_populated_pre_knowledge_activation_database(connection)
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
            sqlite_knowledge_schema._validate_knowledge_publication_access_snapshot_column(
                connection
            )
        if rev.revision == 42:
            sqlite_knowledge_schema._validate_revision_42_knowledge_schema(connection)
        if rev.revision == 43:
            sqlite_knowledge_schema._validate_revision_43_knowledge_schema(connection)
        if rev.revision == 44:
            sqlite_knowledge_schema._validate_revision_44_knowledge_schema(connection)
        if rev.revision == 45:
            sqlite_task_schema._validate_task_retry_series_schema(connection)
        if rev.revision == 46:
            _validate_revision_46_transcript_search_schema(connection)
        if rev.revision == 47:
            sqlite_eval_schema._validate_eval_result_baseline_schema(connection)
        if rev.revision == 48:
            sqlite_eval_schema._validate_captured_eval_case_schema(connection)
        if rev.revision == 50:
            sqlite_eval_schema._validate_eval_run_invocation_column(connection)
        if rev.revision == 51:
            sqlite_memory_evidence_schema._validate_memory_evidence_schema(connection)
        if rev.revision == 52:
            sqlite_session_schema._validate_targeted_tool_grant_schema(connection)
        if rev.revision == 53:
            sqlite_eval_schema._validate_eval_scenario_schema(connection)
        if rev.revision == 55:
            sqlite_task_schema._validate_task_retry_reconciliation_schema(connection)
        if rev.revision == 56:
            sqlite_eval_schema._validate_eval_run_scenario_progress_column(connection)
        if rev.revision == 57:
            sqlite_session_schema._validate_session_message_queue_typed_message_column(connection)
        if rev.revision == 83:
            sqlite_session_schema._validate_session_message_lifecycle_columns(connection)
        if rev.revision == 58:
            sqlite_verified_work_schema._validate_verified_work_schema(
                connection,
                require_verifier_profiles=True,
            )
        if rev.revision == 59:
            sqlite_session_schema._validate_session_instance_schema(connection)
        if rev.revision == 61:
            sqlite_verified_work_schema._validate_work_attempt_admission_schema(connection)
        if rev.revision == 84:
            sqlite_verified_work_schema._validate_work_attempt_lifecycle_schema(connection)
        if rev.revision == 62:
            sqlite_session_schema._validate_revision_sixty_two_payload_schema(connection)
        if rev.revision == 60:
            sqlite_knowledge_schema._validate_revision_60_knowledge_schema(connection)
        if rev.revision == 63:
            sqlite_knowledge_schema._validate_revision_63_knowledge_schema(connection)
        if rev.revision == 64:
            sqlite_eval_schema._validate_eval_authored_suite_schema(connection)
        if rev.revision == 65:
            sqlite_knowledge_schema._validate_revision_42_knowledge_schema(
                connection,
                require_payload_bytes=True,
            )
        if rev.revision == 66:
            _validate_local_execution_attempt_schema(connection)
        if rev.revision == 67:
            sqlite_knowledge_schema._validate_revision_67_knowledge_schema(connection)
        if rev.revision == 68:
            sqlite_eval_schema._validate_eval_judge_calibration_schema(connection)
        if rev.revision == 69:
            sqlite_work_context_schema._validate_revision_69_work_context_schema(connection)
        if rev.revision == 70:
            sqlite_task_schema._validate_interrupted_task_handoff_schema(connection)
        if rev.revision == 71:
            sqlite_work_context_schema._validate_revision_71_recall_delivery_schema(connection)
        if rev.revision == 73:
            sqlite_work_context_schema._validate_revision_71_recall_delivery_schema(
                connection,
                require_processing_schema_version=True,
            )
            sqlite_work_context_schema._validate_revision_73_recall_subscription_schema(connection)
        if rev.revision == 74:
            sqlite_eval_schema._validate_eval_run_trial_checkpoint_schema(connection)
        if rev.revision == 75:
            sqlite_knowledge_schema._validate_revision_75_knowledge_activation_schema(connection)
        if rev.revision == 76:
            sqlite_task_schema._validate_interrupted_handoff_generation_column(connection)
        if rev.revision == 77:
            sqlite_knowledge_schema._validate_revision_77_knowledge_maintenance_governance_schema(
                connection
            )
        if rev.revision == 78:
            sqlite_knowledge_schema._validate_revision_78_knowledge_semantic_watch_schema(
                connection
            )
        if rev.revision == 79:
            sqlite_session_schema._validate_revision_79_child_lifecycle_schema(connection)
        if rev.revision == 88:
            _validate_revision_88_closure_schema(connection)
        if rev.revision == 93:
            validate_sqlite_collaboration_schema(connection)
        if rev.revision == 94:
            validate_sqlite_collaboration_schema(connection, lifecycle=True)
        if rev.revision == 102:
            validate_sqlite_participant_bindings(connection)
        if rev.revision == 108:
            validate_sqlite_context_selection_schema(connection)
        if rev.revision == 111:
            validate_sqlite_wait_discovery(connection)
        if rev.revision == 112:
            validate_sqlite_product_operation_schema(connection)
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
        with sqlite_connection._transaction(connection):
            for statement in _iter_statements(_MIGRATION_STEPS[72]):
                connection.execute(statement)
            sqlite_eval_schema._validate_eval_run_max_concurrency_schema(connection)
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
    with sqlite_connection._transaction(connection):
        _validate_revision_17_indexes(connection, require_all=False)
        for table, column, decl in _MIGRATION_ADD_COLUMNS[17]:
            _add_column_if_missing(connection, table, column, decl)
        for statement in _iter_statements(_MIGRATION_STEPS[17]):
            connection.execute(statement)

    after_session_id: str | None = None
    while True:
        with sqlite_connection._transaction(connection):
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
        with sqlite_connection._transaction(connection):
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

    with sqlite_connection._transaction(connection):
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
            sqlite_records.format_datetime(datetime.now(UTC)),
        ),
    )
