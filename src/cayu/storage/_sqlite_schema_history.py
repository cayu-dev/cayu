"""SQLite baseline and forward-revision schema declarations.

Migration execution, hooks and conditional column changes live in
``_sqlite_support``. Domain schema modules supply their canonical DDL here.
"""

from __future__ import annotations

from cayu.storage._accounting_schema import SQLITE_ACCOUNTING_DDL, SQLITE_AUXILIARY_ACCOUNTING_DDL
from cayu.storage._collaboration_schema import (
    SQLITE_COLLABORATION_CLARIFICATION_DDL,
    SQLITE_COLLABORATION_DDL,
    SQLITE_COLLABORATION_LIFECYCLE_DDL,
    SQLITE_COLLABORATION_PLANNING_DDL,
    SQLITE_COLLABORATION_REQUEST_DDL,
)
from cayu.storage._collaboration_wait_schema import SQLITE_COLLABORATION_WAIT_DDL
from cayu.storage._completion_evaluation_schema import SQLITE_COMPLETION_EVALUATION_DDL
from cayu.storage._completion_verifier_dispatch_schema import (
    SQLITE_COMPLETION_VERIFIER_DISPATCH_DDL,
)
from cayu.storage._external_wait_schema import SQLITE_EXTERNAL_WAIT_DDL
from cayu.storage._model_policy_schema import SQLITE_MODEL_POLICY_DDL
from cayu.storage._participant_bindings_schema import SQLITE_PARTICIPANT_BINDINGS_DDL
from cayu.storage._product_operation_schema import SQLITE_PRODUCT_OPERATION_DDL
from cayu.storage._retention_schema import SQLITE_RETENTION_AUDIT_DDL
from cayu.storage._session_execution import SQLITE_EXECUTION_DDL
from cayu.storage._sqlite_closure_schema import SQLITE_CLOSURE_DDL
from cayu.storage._task_graph_schema import SQLITE_TASK_GRAPH_DDL
from cayu.storage._task_group_schema import SQLITE_TASK_GROUP_DDL, SQLITE_TASK_GROUP_QUIESCENCE_DDL
from cayu.storage._task_scheduling_schema import SQLITE_SCHEDULING_DDL

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
    118: SQLITE_RETENTION_AUDIT_DDL,
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
    88: SQLITE_CLOSURE_DDL,
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
