"""PostgreSQL baseline, migration SQL and concurrent-index history."""

from __future__ import annotations

from dataclasses import dataclass

from cayu._validation import EXECUTION_UNIT_ID_MAX_CHARS
from cayu.approvals.tools import _PENDING_TOOL_APPROVAL_EVENT_PROJECTION_KEYS
from cayu.sessions.base import MAX_PENDING_ACTION_RESULT_BYTES, MAX_PENDING_ACTION_TOOL_CALLS
from cayu.sessions.transcript_queries import TRANSCRIPT_SEARCH_TOKENIZER_VERSION
from cayu.storage._accounting_schema import (
    POSTGRES_ACCOUNTING_DDL,
    POSTGRES_AUXILIARY_ACCOUNTING_DDL,
)
from cayu.storage._collaboration_schema import (
    POSTGRES_COLLABORATION_CLARIFICATION_DDL,
    POSTGRES_COLLABORATION_DDL,
    POSTGRES_COLLABORATION_LIFECYCLE_DDL,
    POSTGRES_COLLABORATION_PLANNING_DDL,
    POSTGRES_COLLABORATION_REQUEST_DDL,
)
from cayu.storage._collaboration_wait_schema import POSTGRES_COLLABORATION_WAIT_DDL
from cayu.storage._completion_evaluation_schema import POSTGRES_COMPLETION_EVALUATION_DDL
from cayu.storage._completion_verifier_dispatch_schema import (
    POSTGRES_COMPLETION_VERIFIER_DISPATCH_DDL,
)
from cayu.storage._external_wait_schema import POSTGRES_EXTERNAL_WAIT_DDL
from cayu.storage._model_policy_schema import POSTGRES_MODEL_POLICY_DDL
from cayu.storage._participant_bindings_schema import POSTGRES_PARTICIPANT_BINDINGS_DDL
from cayu.storage._product_operation_schema import POSTGRES_PRODUCT_OPERATION_DDL
from cayu.storage._session_closure_sql import POSTGRES_TASK_CLOSURE_GUARD_DDL
from cayu.storage._session_execution import POSTGRES_EXECUTION_DDL
from cayu.storage._task_graph_schema import POSTGRES_TASK_GRAPH_DDL
from cayu.storage._task_group_schema import (
    POSTGRES_TASK_GROUP_DDL,
    POSTGRES_TASK_GROUP_QUIESCENCE_DDL,
)
from cayu.storage._task_scheduling_schema import POSTGRES_SCHEDULING_DDL

# Postgres schema mirrors the SQLite store (both at ADR 0001 baseline revision 1)
# but uses Postgres-native types: TEXT ids, JSONB payloads, TIMESTAMPTZ times,
# a global BIGINT identity event cursor, and a per-session monotonic order column.
# All tables carry the cayu_ prefix (ADR 0001 Decision 5) so Cayu state never
# collides with an application's own tables in a shared database. This tuple is the
# baseline-revision DDL (ADR 0001 revision 1); the cayu_schema_migrations
# bookkeeping table is created separately by the migrator.
SESSION_MESSAGE_ACCEPTANCE_INDEX_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_cayu_events_queue_acceptance "
    "ON cayu_events (session_id, (event #>> '{payload,queue_id}')) "
    "WHERE event_type = 'session.message.queued'"
)

SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
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
        created_at TIMESTAMPTZ NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL,
        last_activity_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        run_epoch BIGINT NOT NULL DEFAULT 0,
        event_seq BIGINT NOT NULL DEFAULT 0,
        transcript_seq BIGINT NOT NULL DEFAULT 0,
        invocation JSONB NOT NULL,
        metadata JSONB NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cayu_events (
        sequence BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        insert_xid xid8 NOT NULL DEFAULT pg_current_xact_id(),
        session_id TEXT NOT NULL REFERENCES cayu_sessions(id) ON DELETE CASCADE,
        session_order BIGINT NOT NULL,
        event_id TEXT NOT NULL,
        interaction_id TEXT,
        event_type TEXT NOT NULL,
        timestamp TIMESTAMPTZ NOT NULL,
        agent_name TEXT,
        environment_name TEXT,
        workflow_name TEXT,
        tool_name TEXT,
        payload JSONB NOT NULL,
        event JSONB NOT NULL,
        input_contract_runtime_owned BOOLEAN NOT NULL DEFAULT FALSE,
        file_attachment_attestations_runtime_owned BOOLEAN NOT NULL DEFAULT FALSE,
        pending_action_lookup_key TEXT,
        pending_action_projection JSONB,
        pending_action_projection_bytes BIGINT,
        UNIQUE (session_id, event_id),
        UNIQUE (session_id, session_order)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cayu_budget_reservation_identities (
        reservation_id TEXT PRIMARY KEY,
        publication_session_id TEXT NOT NULL,
        publication_id TEXT NOT NULL,
        published BOOLEAN NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cayu_budget_bindings (
        binding_id TEXT PRIMARY KEY,
        authority_digest TEXT NOT NULL,
        registered_at TIMESTAMPTZ NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cayu_budget_binding_consumptions (
        binding_id TEXT NOT NULL,
        consumption_id TEXT NOT NULL,
        consumed_at TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (binding_id, consumption_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cayu_persisted_event_side_effects (
        session_id TEXT NOT NULL,
        event_id TEXT NOT NULL,
        event_sequence BIGINT NOT NULL,
        status TEXT NOT NULL,
        attempts INTEGER NOT NULL DEFAULT 0,
        claim_id TEXT,
        lease_expires_at TIMESTAMPTZ,
        next_attempt_at TIMESTAMPTZ,
        last_error TEXT,
        updated_at TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (session_id, event_id),
        FOREIGN KEY (session_id, event_id)
            REFERENCES cayu_events(session_id, event_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cayu_session_labels (
        session_id TEXT NOT NULL REFERENCES cayu_sessions(id) ON DELETE CASCADE,
        key TEXT NOT NULL,
        value TEXT NOT NULL,
        PRIMARY KEY (session_id, key)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cayu_public_authority_aliases (
        field_name TEXT NOT NULL,
        scope_session_id TEXT NOT NULL,
        public_alias TEXT NOT NULL,
        private_value TEXT NOT NULL,
        PRIMARY KEY (field_name, scope_session_id, public_alias)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_cayu_public_authority_private_value
        ON cayu_public_authority_aliases(field_name, scope_session_id, private_value)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_cayu_public_authority_public_alias
        ON cayu_public_authority_aliases(field_name, public_alias)
    """,
    """
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
        issued_at TIMESTAMPTZ NOT NULL,
        expires_at TIMESTAMPTZ NOT NULL,
        max_calls BIGINT NOT NULL CHECK (max_calls >= 1 AND max_calls <= 32),
        used_calls BIGINT NOT NULL DEFAULT 0
            CHECK (used_calls >= 0 AND used_calls <= max_calls),
        revoked_at TIMESTAMPTZ,
        record JSONB NOT NULL,
        UNIQUE (session_id, interaction_id, request_id),
        UNIQUE (session_id, interaction_id, tool_id)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_cayu_targeted_tool_grants_interaction
        ON cayu_targeted_tool_grants(session_id, interaction_id, issued_at, grant_id)
    """,
    """
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
        bound_at TIMESTAMPTZ NOT NULL,
        record JSONB NOT NULL,
        UNIQUE (session_id, interaction_id, invocation_id),
        UNIQUE (session_id, interaction_id, outer_tool_call_id)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_cayu_targeted_tool_grant_uses_grant
        ON cayu_targeted_tool_grant_uses(grant_id, bound_at, use_id)
    """,
    """
    CREATE TABLE IF NOT EXISTS cayu_public_authority_alias_keys (
        key_id TEXT PRIMARY KEY,
        fingerprint TEXT NOT NULL,
        backfill_completed BOOLEAN NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cayu_public_authority_alias_config (
        singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
        active_key_id TEXT NOT NULL REFERENCES cayu_public_authority_alias_keys(key_id),
        keyring_fingerprint TEXT NOT NULL,
        generation BIGINT NOT NULL CHECK (generation >= 1),
        retired_key_ids JSONB NOT NULL CHECK (jsonb_typeof(retired_key_ids) = 'array')
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cayu_checkpoints (
        session_id TEXT PRIMARY KEY REFERENCES cayu_sessions(id) ON DELETE CASCADE,
        state JSONB NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL,
        pending_action_source_bytes BIGINT,
        pending_action_tool_call_count INTEGER NOT NULL DEFAULT 0,
        pending_action_flags INTEGER NOT NULL DEFAULT 0,
        pending_action_metrics_ready BOOLEAN NOT NULL DEFAULT TRUE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cayu_session_operations (
        session_id TEXT NOT NULL REFERENCES cayu_sessions(id) ON DELETE CASCADE,
        idempotency_key TEXT NOT NULL,
        record JSONB NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (session_id, idempotency_key)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_cayu_checkpoints_pending_interruption_cascade
        ON cayu_checkpoints(session_id)
        WHERE state ? 'pending_interruption_cascade'
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_cayu_checkpoints_pending_control_action
        ON cayu_checkpoints(session_id)
        WHERE pending_action_flags <> 0
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_cayu_checkpoints_queued_dispatch_run
        ON cayu_checkpoints(session_id COLLATE "C")
        WHERE state #> '{session_run_operation,queue_task_id}' IS NOT NULL
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_cayu_checkpoints_queued_dispatch_receipts
        ON cayu_checkpoints(session_id COLLATE "C")
        WHERE state #> '{queued_dispatch_terminal_receipts,receipts}' IS NOT NULL
    """,
    """
    CREATE TABLE IF NOT EXISTS cayu_transcript_messages (
        sequence BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES cayu_sessions(id) ON DELETE CASCADE,
        interaction_id TEXT,
        session_order BIGINT,
        message JSONB NOT NULL,
        transcript_search_document TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cayu_transcript_search_configuration (
        singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
        tokenizer_version TEXT NOT NULL
    )
    """,
    f"""
    INSERT INTO cayu_transcript_search_configuration (singleton, tokenizer_version)
    VALUES (TRUE, '{TRANSCRIPT_SEARCH_TOKENIZER_VERSION}')
    ON CONFLICT (singleton) DO NOTHING
    """,
    """
    CREATE TABLE IF NOT EXISTS cayu_mcp_manifest_baselines (
        history_key TEXT PRIMARY KEY,
        generation BIGINT NOT NULL CHECK (generation >= 1),
        baseline JSONB NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cayu_session_message_queue (
        ordering_key BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        queue_id TEXT NOT NULL UNIQUE,
        session_id TEXT NOT NULL REFERENCES cayu_sessions(id) ON DELETE CASCADE,
        idempotency_key TEXT NOT NULL,
        content TEXT NOT NULL,
        message_json JSONB,
        conditions_json JSONB,
        terminal_json JSONB,
        delivery_mode TEXT NOT NULL,
        status TEXT NOT NULL,
        requested_by JSONB,
        accepted_run_epoch BIGINT NOT NULL,
        accepted_transcript_cursor BIGINT NOT NULL,
        accepted_event_id TEXT NOT NULL,
        accepted_at TIMESTAMPTZ NOT NULL,
        delivered_run_epoch BIGINT,
        delivered_transcript_cursor BIGINT,
        delivered_event_id TEXT,
        delivered_at TIMESTAMPTZ,
        UNIQUE (session_id, idempotency_key)
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
        session_instance_id TEXT,
        parent_task_id TEXT,
        assigned_agent_name TEXT,
        input JSONB NOT NULL,
        result JSONB,
        error JSONB,
        metadata JSONB NOT NULL,
        created_at TIMESTAMPTZ NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL,
        started_at TIMESTAMPTZ,
        completed_at TIMESTAMPTZ,
        invocation JSONB NOT NULL,
        retry_series JSONB
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cayu_task_terminalization_receipts (
        task_id TEXT NOT NULL,
        idempotency_key TEXT NOT NULL,
        request_sha256 TEXT NOT NULL,
        worker_id TEXT NOT NULL,
        terminal_kind TEXT NOT NULL,
        task_json JSONB NOT NULL,
        committed_at TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (task_id, idempotency_key)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cayu_task_interrupted_handoff_receipts (
        task_id TEXT NOT NULL,
        handoff_id TEXT NOT NULL,
        request_sha256 TEXT NOT NULL,
        request_json JSONB NOT NULL,
        task_json JSONB NOT NULL,
        committed_at TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (task_id, handoff_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cayu_task_retry_settlements (
        task_id TEXT NOT NULL,
        idempotency_key TEXT NOT NULL,
        request_sha256 TEXT NOT NULL,
        receipt_json JSONB NOT NULL,
        committed_at TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (task_id, idempotency_key)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cayu_task_retry_reconciliation_rejections (
        task_id TEXT NOT NULL,
        reconciliation_idempotency_key TEXT NOT NULL,
        request_sha256 TEXT NOT NULL,
        record_json JSONB NOT NULL,
        recorded_at TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (task_id, reconciliation_idempotency_key)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cayu_recall_receipts (
        receipt_id TEXT COLLATE "C" PRIMARY KEY,
        session_id TEXT COLLATE "C" NOT NULL
            REFERENCES cayu_sessions(id) ON DELETE CASCADE,
        interaction_id TEXT COLLATE "C" NOT NULL,
        model_step_id TEXT COLLATE "C" NOT NULL,
        created_at TIMESTAMPTZ NOT NULL,
        receipt_json JSONB NOT NULL,
        document_bytes BIGINT NOT NULL CHECK (
            document_bytes >= 1 AND document_bytes <= 256000
        )
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cayu_context_exposures (
        exposure_id TEXT COLLATE "C" PRIMARY KEY,
        session_id TEXT COLLATE "C" NOT NULL
            REFERENCES cayu_sessions(id) ON DELETE CASCADE,
        interaction_id TEXT COLLATE "C" NOT NULL,
        model_step_id TEXT COLLATE "C" NOT NULL,
        model_attempt_id TEXT COLLATE "C" NOT NULL,
        provider_attempt_id TEXT COLLATE "C" NOT NULL,
        state TEXT NOT NULL CHECK (state IN (
            'planned', 'prepared', 'dispatch_started', 'acknowledged',
            'completed', 'failed', 'cancelled', 'indeterminate'
        )),
        state_revision INTEGER NOT NULL CHECK (
            state_revision >= 0 AND state_revision < 16
        ),
        created_at TIMESTAMPTZ NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL,
        exposure_json JSONB NOT NULL,
        document_bytes BIGINT NOT NULL CHECK (
            document_bytes >= 1 AND document_bytes <= 128000
        ),
        UNIQUE (session_id, model_attempt_id),
        UNIQUE (session_id, provider_attempt_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cayu_recall_item_exposures (
        exposure_id TEXT COLLATE "C" NOT NULL
            REFERENCES cayu_context_exposures(exposure_id) ON DELETE CASCADE,
        ordinal INTEGER NOT NULL CHECK (ordinal >= 0 AND ordinal < 64),
        receipt_id TEXT COLLATE "C" NOT NULL
            REFERENCES cayu_recall_receipts(receipt_id) ON DELETE CASCADE,
        receipt_item_ordinal INTEGER NOT NULL CHECK (
            receipt_item_ordinal >= 0 AND receipt_item_ordinal < 64
        ),
        item_json JSONB NOT NULL,
        document_bytes BIGINT NOT NULL CHECK (
            document_bytes >= 1 AND document_bytes <= 16384
        ),
        PRIMARY KEY (exposure_id, ordinal),
        UNIQUE (exposure_id, receipt_id, receipt_item_ordinal)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_cayu_recall_receipts_session_page "
    'ON cayu_recall_receipts(session_id, created_at, receipt_id COLLATE "C")',
    "CREATE INDEX IF NOT EXISTS idx_cayu_recall_receipts_interaction_page "
    'ON cayu_recall_receipts(session_id, interaction_id, created_at, receipt_id COLLATE "C")',
    "CREATE INDEX IF NOT EXISTS idx_cayu_recall_receipts_step_page "
    'ON cayu_recall_receipts(session_id, model_step_id, created_at, receipt_id COLLATE "C")',
    "CREATE INDEX IF NOT EXISTS idx_cayu_recall_receipts_interaction_step_page "
    "ON cayu_recall_receipts(session_id, interaction_id, model_step_id, created_at, "
    'receipt_id COLLATE "C")',
    "CREATE INDEX IF NOT EXISTS idx_cayu_context_exposures_session_page "
    'ON cayu_context_exposures(session_id, created_at, exposure_id COLLATE "C")',
    "CREATE INDEX IF NOT EXISTS idx_cayu_context_exposures_interaction_page "
    'ON cayu_context_exposures(session_id, interaction_id, created_at, exposure_id COLLATE "C")',
    "CREATE INDEX IF NOT EXISTS idx_cayu_context_exposures_step_page "
    'ON cayu_context_exposures(session_id, model_step_id, created_at, exposure_id COLLATE "C")',
    "CREATE INDEX IF NOT EXISTS idx_cayu_context_exposures_interaction_step_page "
    "ON cayu_context_exposures(session_id, interaction_id, model_step_id, created_at, "
    'exposure_id COLLATE "C")',
    "CREATE INDEX IF NOT EXISTS idx_cayu_recall_item_exposures_receipt "
    "ON cayu_recall_item_exposures(receipt_id, exposure_id, ordinal)",
    """
    CREATE TABLE IF NOT EXISTS cayu_event_watcher_state (
        watcher_name TEXT PRIMARY KEY,
        cursor_sequence BIGINT NOT NULL,
        pending_event_id TEXT,
        pending_event_sequence BIGINT,
        pending_attempt INTEGER NOT NULL,
        pending_claim_id TEXT,
        delivery_status TEXT,
        lease_expires_at TIMESTAMPTZ,
        last_error TEXT,
        dead_lettered_count INTEGER NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cayu_event_watcher_dead_letters (
        watcher_name TEXT NOT NULL,
        event_sequence BIGINT NOT NULL,
        event_id TEXT NOT NULL,
        attempts INTEGER NOT NULL,
        error TEXT NOT NULL,
        dead_lettered_at TIMESTAMPTZ NOT NULL,
        resolved_at TIMESTAMPTZ,
        PRIMARY KEY (watcher_name, event_sequence)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_cayu_sessions_status ON cayu_sessions(status)",
    "CREATE INDEX IF NOT EXISTS idx_cayu_sessions_agent_name ON cayu_sessions(agent_name)",
    "CREATE INDEX IF NOT EXISTS idx_cayu_sessions_environment_name "
    "ON cayu_sessions(environment_name)",
    "CREATE INDEX IF NOT EXISTS idx_cayu_sessions_causal_budget_id "
    "ON cayu_sessions(causal_budget_id)",
    "CREATE INDEX IF NOT EXISTS idx_cayu_sessions_parent_created_id "
    'ON cayu_sessions(parent_session_id, created_at, id COLLATE "C")',
    "CREATE INDEX IF NOT EXISTS idx_cayu_session_labels_key_value_session "
    "ON cayu_session_labels(key, value, session_id)",
    "CREATE INDEX IF NOT EXISTS idx_cayu_events_session_order "
    "ON cayu_events(session_id, session_order)",
    SESSION_MESSAGE_ACCEPTANCE_INDEX_DDL,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_cayu_events_budget_reservation_identity
    ON cayu_events ((payload ->> 'reservation_id'))
    WHERE event_type = 'budget.reserved'
      AND jsonb_typeof(payload -> 'reservation_id') = 'string'
    """,
    *POSTGRES_ACCOUNTING_DDL,
    "CREATE INDEX IF NOT EXISTS idx_cayu_events_session_sequence "
    "ON cayu_events(session_id, sequence)",
    """
    CREATE INDEX IF NOT EXISTS idx_cayu_events_pending_action_barrier
    ON cayu_events(session_id, sequence)
    WHERE event_type = 'session.resumed'
       OR event_type = 'session.completed'
       OR event_type = 'session.failed'
    """,
    """
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
      AND pending_action_lookup_key IS NOT NULL
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_cayu_events_pending_action_round_scope
    ON cayu_events(
        session_id,
        (pending_action_projection #>> '{payload,tool_round_id}'),
        sequence
    )
    WHERE event_type IN (
        'tool.call.started',
        'tool.call.completed',
        'tool.call.failed',
        'tool.call.blocked',
        'tool.call.approval_denied'
    )
      AND jsonb_typeof(
          pending_action_projection #> '{payload,tool_round_id}'
      ) = 'string'
      AND pending_action_projection #>> '{payload,tool_round_id}'
          ~ '^tround_[0-9a-f]{32}$'
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_cayu_events_pending_action_attempt_scope
    ON cayu_events(
        session_id,
        (pending_action_projection #>> '{payload,model_step_id}'),
        (pending_action_projection #>> '{payload,model_attempt_id}'),
        sequence
    )
    WHERE event_type IN (
        'tool.call.started',
        'tool.call.completed',
        'tool.call.failed',
        'tool.call.blocked',
        'tool.call.approval_denied'
    )
      AND jsonb_typeof(
          pending_action_projection #> '{payload,model_step_id}'
      ) = 'string'
      AND jsonb_typeof(
          pending_action_projection #> '{payload,model_attempt_id}'
      ) = 'string'
      AND pending_action_projection #>> '{payload,model_step_id}'
          ~ '^mstep_[0-9a-f]{32}$'
      AND pending_action_projection #>> '{payload,model_attempt_id}'
          ~ '^matt_[0-9a-f]{32}$'
    """,
    "CREATE INDEX IF NOT EXISTS idx_cayu_events_insert_xid ON cayu_events(insert_xid)",
    "CREATE INDEX IF NOT EXISTS idx_cayu_persisted_event_side_effects_delivery "
    "ON cayu_persisted_event_side_effects"
    "(status, next_attempt_at, lease_expires_at, event_sequence)",
    "CREATE INDEX IF NOT EXISTS idx_cayu_events_type_timestamp ON cayu_events(event_type, timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_cayu_events_agent_name ON cayu_events(agent_name)",
    "CREATE INDEX IF NOT EXISTS idx_cayu_events_environment_name ON cayu_events(environment_name)",
    """
    CREATE INDEX IF NOT EXISTS idx_cayu_events_workflow_step_replay
    ON cayu_events(
        session_id,
        workflow_name,
        (event -> 'payload' ->> 'step_id'),
        event_type,
        sequence DESC
    )
    WHERE event_type IN (
        'workflow.step.started',
        'workflow.step.completed'
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_cayu_events_workflow_step_attempt
    ON cayu_events(
        session_id,
        workflow_name,
        (event -> 'payload' ->> 'attempt_id'),
        (event -> 'payload' ->> 'step_id'),
        event_type,
        sequence DESC
    )
    WHERE event_type IN (
        'workflow.step.started',
        'workflow.step.completed'
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_cayu_events_workflow_attempt_marker
    ON cayu_events(session_id, workflow_name, sequence DESC)
    WHERE event_type = 'custom.cayu.workflow.attempt'
    """,
    "CREATE INDEX IF NOT EXISTS idx_cayu_events_workflow_name ON cayu_events(workflow_name)",
    "CREATE INDEX IF NOT EXISTS idx_cayu_events_tool_name ON cayu_events(tool_name)",
    "CREATE INDEX IF NOT EXISTS idx_cayu_transcript_messages_session_sequence "
    "ON cayu_transcript_messages(session_id, sequence)",
    "CREATE INDEX IF NOT EXISTS idx_cayu_events_session_interaction_sequence "
    "ON cayu_events(session_id, interaction_id, sequence)",
    "CREATE INDEX IF NOT EXISTS idx_cayu_transcript_messages_session_interaction_sequence "
    "ON cayu_transcript_messages(session_id, interaction_id, sequence)",
    "CREATE INDEX IF NOT EXISTS idx_cayu_session_message_queue_delivery "
    "ON cayu_session_message_queue(session_id, status, delivery_mode, ordering_key)",
    "CREATE INDEX IF NOT EXISTS idx_cayu_tasks_status ON cayu_tasks(status)",
    "CREATE INDEX IF NOT EXISTS idx_cayu_tasks_type ON cayu_tasks(type)",
    "CREATE INDEX IF NOT EXISTS idx_cayu_tasks_session_id ON cayu_tasks(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_cayu_tasks_parent_task_id ON cayu_tasks(parent_task_id)",
    "CREATE INDEX IF NOT EXISTS idx_cayu_tasks_session_created_id "
    'ON cayu_tasks(session_id, created_at, id COLLATE "C")',
    "CREATE INDEX IF NOT EXISTS idx_cayu_tasks_parent_created_id "
    'ON cayu_tasks(parent_task_id, created_at, id COLLATE "C")',
    "CREATE INDEX IF NOT EXISTS idx_cayu_tasks_assigned_agent_name "
    "ON cayu_tasks(assigned_agent_name)",
    "CREATE INDEX IF NOT EXISTS idx_cayu_event_watcher_state_delivery "
    "ON cayu_event_watcher_state(delivery_status, lease_expires_at)",
    "CREATE INDEX IF NOT EXISTS idx_cayu_event_watcher_dead_letters_unresolved "
    "ON cayu_event_watcher_dead_letters(watcher_name, resolved_at, event_sequence)",
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

# One active CLI migration receipt is retained beside the revision ledger until
# its final rendering succeeds.  The singleton row makes a different invocation
# fail closed instead of overwriting evidence for resumable committed progress.
MIGRATION_RECEIPTS_TABLE_DDL = """
    CREATE TABLE IF NOT EXISTS cayu_schema_migration_receipts (
        singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
        operation_sha256 TEXT NOT NULL UNIQUE
            CHECK (operation_sha256 ~ '^[0-9a-f]{64}$'),
        receipt_json JSONB NOT NULL
            CHECK (jsonb_typeof(receipt_json) = 'object')
    )
"""


# Per-revision forward-migration DDL, keyed by revision number. The baseline
# (revision 1) is applied from SCHEMA_STATEMENTS, so it is not listed
# here; future additive/breaking revisions append their ALTER/CREATE statements.
_MIGRATION_STEPS: dict[int, tuple[str, ...]] = {
    108: (
        """CREATE TABLE IF NOT EXISTS cayu_context_selection_exclusions (
            selection_key TEXT PRIMARY KEY,
            owner_scope TEXT NOT NULL,
            owner_id TEXT NOT NULL,
            owner_incarnation TEXT NOT NULL,
            source_session_id TEXT NOT NULL,
            source_session_instance_id TEXT NOT NULL,
            request_commitment TEXT NOT NULL,
            decision_json TEXT NOT NULL
        )""",
        """CREATE INDEX IF NOT EXISTS idx_context_selection_exclusions_owner
            ON cayu_context_selection_exclusions(owner_scope, owner_id, owner_incarnation)""",
    ),
    110: (
        """CREATE TABLE IF NOT EXISTS cayu_producer_cleanup_receipts (
            operation_key TEXT PRIMARY KEY,
            namespace_key TEXT NOT NULL,
            generation BIGINT NOT NULL CHECK (generation BETWEEN 1 AND 9007199254740991),
            receipt_json JSONB NOT NULL CHECK (
                jsonb_typeof(receipt_json) = 'object'
                AND octet_length(receipt_json::text) BETWEEN 1 AND 65536
            )
        )""",
        """CREATE INDEX IF NOT EXISTS idx_cayu_producer_cleanup_namespace
        ON cayu_producer_cleanup_receipts(namespace_key, generation, operation_key)""",
        """CREATE TABLE IF NOT EXISTS cayu_producer_cleanup_retirements (
            namespace_key TEXT PRIMARY KEY,
            through_generation BIGINT NOT NULL CHECK (through_generation BETWEEN 1 AND 9007199254740991)
        )""",
    ),
    103: (
        """CREATE TABLE IF NOT EXISTS cayu_session_creation_decisions (
            operation_key TEXT PRIMARY KEY,
            owner_key TEXT NOT NULL,
            state TEXT NOT NULL,
            recovery_pending INTEGER NOT NULL,
            decision_json TEXT NOT NULL
        )""",
        """CREATE INDEX IF NOT EXISTS idx_creation_decisions_pending
            ON cayu_session_creation_decisions(owner_key, recovery_pending, operation_key)""",
    ),
    101: (
        """
        ALTER TABLE cayu_budget_bindings ADD COLUMN IF NOT EXISTS allowance BIGINT
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_budget_binding_consumptions (
            binding_id TEXT NOT NULL,
            consumption_id TEXT NOT NULL,
            consumed_at TIMESTAMPTZ NOT NULL,
            PRIMARY KEY (binding_id, consumption_id)
        )
        """,
    ),
    100: (
        """
        CREATE TABLE IF NOT EXISTS cayu_budget_bindings (
            binding_id TEXT PRIMARY KEY,
            authority_digest TEXT NOT NULL,
            registered_at TIMESTAMPTZ NOT NULL
        )
        """,
    ),
    107: POSTGRES_COLLABORATION_PLANNING_DDL,
    111: POSTGRES_COLLABORATION_WAIT_DDL,
    112: POSTGRES_PRODUCT_OPERATION_DDL,
    113: POSTGRES_EXECUTION_DDL,
    114: POSTGRES_MODEL_POLICY_DDL,
    115: POSTGRES_EXTERNAL_WAIT_DDL,
    116: POSTGRES_COMPLETION_VERIFIER_DISPATCH_DDL,
    117: POSTGRES_COMPLETION_EVALUATION_DDL,
    106: (),  # Contract-only writer fence; existing typed request records own storage.
    105: POSTGRES_COLLABORATION_CLARIFICATION_DDL,
    104: (
        """CREATE TABLE IF NOT EXISTS cayu_peer_content_attempts (
            operation_key TEXT PRIMARY KEY, request_json JSONB NOT NULL, receipt_json JSONB NOT NULL
        )""",
        """
        CREATE TABLE IF NOT EXISTS cayu_peer_content_receipts (
            append_key_json JSONB PRIMARY KEY,
            operation_key TEXT UNIQUE NOT NULL,
            commitment_json JSONB NOT NULL,
            receipt_json JSONB NOT NULL,
            request_json JSONB,
            target_deleted BOOLEAN NOT NULL DEFAULT FALSE
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_peer_content_exposures (
            exposure_id TEXT PRIMARY KEY,
            operation_key TEXT UNIQUE NOT NULL,
            append_key_json JSONB NOT NULL,
            commitment_json JSONB NOT NULL,
            receipt_json JSONB NOT NULL
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_cayu_peer_content_exposures_append
            ON cayu_peer_content_exposures(append_key_json, exposure_id)
        """,
    ),
    99: (
        """
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
            ownership_revision BIGINT NOT NULL CHECK (ownership_revision >= 1),
            event_json JSONB NOT NULL
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_cayu_context_view_lifecycle_events_view
            ON cayu_context_view_lifecycle_events(view_id, ownership_revision, event_id)
        """,
    ),
    98: (
        """
        ALTER TABLE cayu_context_view_selections
            ADD COLUMN IF NOT EXISTS ownership_revision BIGINT NOT NULL DEFAULT 1
            CHECK (ownership_revision >= 1)
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_context_view_ownership_operations (
            operation_key TEXT PRIMARY KEY,
            selection_key TEXT NOT NULL,
            request_commitment TEXT NOT NULL,
            receipt_json JSONB NOT NULL
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_cayu_context_view_ownership_operations_selection
            ON cayu_context_view_ownership_operations(selection_key)
        """,
    ),
    97: (
        """
        CREATE TABLE IF NOT EXISTS cayu_context_views (
            view_id TEXT PRIMARY KEY,
            publication_key TEXT NOT NULL UNIQUE,
            source_owner_scope TEXT NOT NULL,
            source_owner_id TEXT NOT NULL,
            source_owner_incarnation TEXT NOT NULL,
            source_session_id TEXT NOT NULL,
            source_session_instance_id TEXT NOT NULL,
            transcript_cursor BIGINT NOT NULL CHECK (transcript_cursor >= 0),
            projection_schema TEXT NOT NULL,
            extension_set_commitment TEXT NOT NULL,
            manifest_json JSONB NOT NULL
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_cayu_context_views_source
            ON cayu_context_views(
                source_owner_scope, source_owner_id, source_owner_incarnation,
                source_session_id, source_session_instance_id, transcript_cursor, view_id
            )
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_context_view_selections (
            selection_key TEXT PRIMARY KEY,
            request_commitment TEXT NOT NULL,
            view_id TEXT NOT NULL,
            owner_scope TEXT NOT NULL,
            owner_id TEXT NOT NULL,
            owner_incarnation TEXT NOT NULL,
            state TEXT NOT NULL CHECK (state IN ('selected', 'adopted', 'transferred', 'released', 'expired')),
            pin_commitment TEXT NOT NULL,
            expires_at_ms BIGINT NOT NULL CHECK (expires_at_ms >= 0),
            receipt_json JSONB NOT NULL
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_cayu_context_view_selections_view
            ON cayu_context_view_selections(view_id, state)
        """,
    ),
    96: (*POSTGRES_TASK_GROUP_QUIESCENCE_DDL, *POSTGRES_PARTICIPANT_BINDINGS_DDL),
    102: POSTGRES_PARTICIPANT_BINDINGS_DDL,
    90: POSTGRES_SCHEDULING_DDL,
    92: POSTGRES_TASK_GROUP_DDL,
    93: POSTGRES_COLLABORATION_DDL,
    94: POSTGRES_COLLABORATION_LIFECYCLE_DDL,
    95: POSTGRES_COLLABORATION_REQUEST_DDL,
    91: POSTGRES_TASK_GRAPH_DDL,
    88: (
        """
        CREATE TABLE IF NOT EXISTS cayu_task_session_closure_claims (
            session_id TEXT PRIMARY KEY,
            plan_id TEXT NOT NULL CHECK (plan_id ~ '^[0-9a-f]{64}$'),
            claim_json JSONB NOT NULL CHECK (
                octet_length(claim_json::text) BETWEEN 1 AND 16777216
                AND jsonb_typeof(claim_json) = 'object'
            )
        )
        """,
        POSTGRES_TASK_CLOSURE_GUARD_DDL,
        """
        DO $migration$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_trigger
                WHERE tgrelid = 'cayu_tasks'::regclass
                  AND tgname = 'cayu_task_closure_admission_guard'
            ) THEN
                CREATE TRIGGER cayu_task_closure_admission_guard
                BEFORE INSERT OR UPDATE ON cayu_tasks
                FOR EACH ROW EXECUTE FUNCTION cayu_task_closure_admission_guard();
            END IF;
        END
        $migration$
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_session_closure_progress (
            root_session_id TEXT NOT NULL,
            plan_id TEXT NOT NULL CHECK (plan_id ~ '^[0-9a-f]{64}$'),
            progress_json JSONB NOT NULL CHECK (
                octet_length(progress_json::text) BETWEEN 1 AND 384000
                AND jsonb_typeof(progress_json) = 'object'
            ),
            PRIMARY KEY (root_session_id, plan_id)
        )
        """,
    ),
    87: (
        """
        CREATE TABLE IF NOT EXISTS cayu_session_closure_tombstones (
            root_session_id TEXT NOT NULL,
            plan_id TEXT NOT NULL CHECK (plan_id ~ '^[0-9a-f]{64}$'),
            child_session_id TEXT NOT NULL,
            original_parent_session_id TEXT NOT NULL,
            detached_at TIMESTAMPTZ NOT NULL,
            tombstone_json JSONB NOT NULL CHECK (
                octet_length(tombstone_json::text) BETWEEN 1 AND 32768
                AND jsonb_typeof(tombstone_json) = 'object'
            ),
            PRIMARY KEY (root_session_id, plan_id, child_session_id)
        )
        """,
    ),
    81: (
        """
        CREATE TABLE IF NOT EXISTS cayu_event_watcher_settlements (
            watcher_name TEXT NOT NULL,
            claim_id TEXT NOT NULL,
            receipt_json JSONB NOT NULL,
            PRIMARY KEY (watcher_name, claim_id)
        )
        """,
    ),
    2: (
        """
        CREATE TABLE IF NOT EXISTS cayu_session_labels (
            session_id TEXT NOT NULL REFERENCES cayu_sessions(id) ON DELETE CASCADE,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            PRIMARY KEY (session_id, key)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_cayu_session_labels_key_value_session "
        "ON cayu_session_labels(key, value, session_id)",
    ),
    3: (
        """
        CREATE TABLE IF NOT EXISTS cayu_event_watcher_state (
            watcher_name TEXT PRIMARY KEY,
            cursor_sequence BIGINT NOT NULL,
            pending_event_id TEXT,
            pending_event_sequence BIGINT,
            pending_attempt INTEGER NOT NULL,
            pending_claim_id TEXT,
            delivery_status TEXT,
            lease_expires_at TIMESTAMPTZ,
            last_error TEXT,
            dead_lettered_count INTEGER NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_cayu_event_watcher_state_delivery "
        "ON cayu_event_watcher_state(delivery_status, lease_expires_at)",
    ),
    4: (
        "ALTER TABLE cayu_tasks ADD COLUMN worker_id TEXT",
        "ALTER TABLE cayu_tasks ADD COLUMN lease_expires_at TIMESTAMPTZ",
        "CREATE INDEX IF NOT EXISTS idx_cayu_tasks_worker_id ON cayu_tasks(worker_id)",
        "CREATE INDEX IF NOT EXISTS idx_cayu_tasks_status_lease "
        "ON cayu_tasks(status, lease_expires_at)",
    ),
    5: (
        "ALTER TABLE cayu_tasks ADD COLUMN status_reason TEXT",
        "ALTER TABLE cayu_tasks ADD COLUMN status_payload JSONB",
    ),
    6: (
        """
        CREATE TABLE IF NOT EXISTS cayu_knowledge_entries (
            id TEXT PRIMARY KEY,
            namespace TEXT NOT NULL,
            text TEXT NOT NULL,
            kind TEXT NOT NULL,
            visibility TEXT NOT NULL,
            status TEXT NOT NULL,
            created_by_type TEXT NOT NULL,
            created_by TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            source_type TEXT,
            source_uri TEXT,
            source_id TEXT,
            source_hash TEXT,
            importance DOUBLE PRECISION,
            importance_source TEXT,
            confidence DOUBLE PRECISION,
            last_used_at TIMESTAMPTZ,
            expires_at TIMESTAMPTZ,
            title TEXT,
            metadata JSONB NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_knowledge_labels (
            entry_id TEXT NOT NULL REFERENCES cayu_knowledge_entries(id) ON DELETE CASCADE,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            PRIMARY KEY (entry_id, key)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_knowledge_aspects (
            entry_id TEXT NOT NULL REFERENCES cayu_knowledge_entries(id) ON DELETE CASCADE,
            aspect TEXT NOT NULL,
            PRIMARY KEY (entry_id, aspect)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_knowledge_impact_targets (
            entry_id TEXT NOT NULL REFERENCES cayu_knowledge_entries(id) ON DELETE CASCADE,
            impact_target TEXT NOT NULL,
            PRIMARY KEY (entry_id, impact_target)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_knowledge_chunks (
            id TEXT PRIMARY KEY,
            entry_id TEXT NOT NULL REFERENCES cayu_knowledge_entries(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            content_hash TEXT,
            source_uri TEXT,
            metadata JSONB NOT NULL,
            UNIQUE (entry_id, chunk_index)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_entries_namespace_status "
        "ON cayu_knowledge_entries(namespace, status)",
        "CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_entries_kind "
        "ON cayu_knowledge_entries(kind)",
        "CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_entries_visibility "
        "ON cayu_knowledge_entries(visibility)",
        "CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_entries_source "
        "ON cayu_knowledge_entries(source_type, source_id)",
        "CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_entries_expires_at "
        "ON cayu_knowledge_entries(expires_at)",
        "CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_labels_key_value_entry "
        "ON cayu_knowledge_labels(key, value, entry_id)",
        "CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_aspects_aspect_entry "
        "ON cayu_knowledge_aspects(aspect, entry_id)",
        "CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_impact_targets_target_entry "
        "ON cayu_knowledge_impact_targets(impact_target, entry_id)",
        "CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_chunks_entry_index "
        "ON cayu_knowledge_chunks(entry_id, chunk_index)",
    ),
    7: (
        "CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_entries_title_fts "
        "ON cayu_knowledge_entries USING GIN (to_tsvector('simple', COALESCE(title, '')))",
        "CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_entries_text_fts "
        "ON cayu_knowledge_entries USING GIN (to_tsvector('simple', text))",
        "CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_chunks_text_fts "
        "ON cayu_knowledge_chunks USING GIN (to_tsvector('simple', text))",
    ),
    8: (
        """
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
            reserved_amount NUMERIC NOT NULL,
            actual_amount NUMERIC,
            status TEXT NOT NULL,
            reason TEXT,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_cayu_budget_reservations_scope "
        "ON cayu_budget_reservations(scope, budget_key, budget_window, currency, status)",
    ),
    10: (
        # Per-session monotonic counter that append_events advances with a single
        # UPDATE ... RETURNING (replacing the row-lock + COALESCE(MAX()) scan).
        # IF NOT EXISTS keeps the greenfield-through-migrations path a no-op, since
        # the baseline schema already declares the column.
        "ALTER TABLE cayu_sessions ADD COLUMN IF NOT EXISTS event_seq BIGINT NOT NULL DEFAULT 0",
        # Seed the counter from the highest existing session_order so the first
        # post-migration append continues the sequence instead of colliding with
        # already-stored rows.
        """
        UPDATE cayu_sessions AS s
        SET event_seq = COALESCE(
            (SELECT MAX(e.session_order) FROM cayu_events AS e WHERE e.session_id = s.id),
            0
        )
        """,
    ),
    11: (
        """
        CREATE TABLE IF NOT EXISTS cayu_event_watcher_dead_letters (
            watcher_name TEXT NOT NULL,
            event_sequence BIGINT NOT NULL,
            event_id TEXT NOT NULL,
            attempts INTEGER NOT NULL,
            error TEXT NOT NULL,
            dead_lettered_at TIMESTAMPTZ NOT NULL,
            resolved_at TIMESTAMPTZ,
            PRIMARY KEY (watcher_name, event_sequence)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_cayu_event_watcher_dead_letters_unresolved "
        "ON cayu_event_watcher_dead_letters(watcher_name, resolved_at, event_sequence)",
    ),
    # Add the embedding-space version column so the standard `cayu storage migrate` deploy step (which
    # runs this table via PostgresSessionStore) reaches an existing cayu_knowledge_embeddings table.
    # `IF EXISTS` makes it a no-op when the embeddings table was never created (embedding store unused).
    12: (
        "ALTER TABLE IF EXISTS cayu_knowledge_embeddings "
        "ADD COLUMN IF NOT EXISTS embedding_space_version INTEGER NOT NULL DEFAULT 1",
    ),
    13: (
        "ALTER TABLE cayu_events "
        "ADD COLUMN IF NOT EXISTS insert_xid xid8 NOT NULL DEFAULT pg_current_xact_id()",
        "CREATE INDEX IF NOT EXISTS idx_cayu_events_insert_xid ON cayu_events(insert_xid)",
    ),
    14: (
        "ALTER TABLE cayu_sessions ADD COLUMN IF NOT EXISTS "
        "last_activity_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP",
        "ALTER TABLE cayu_sessions ADD COLUMN IF NOT EXISTS run_epoch BIGINT NOT NULL DEFAULT 0",
    ),
    15: (
        "CREATE INDEX IF NOT EXISTS idx_cayu_checkpoints_pending_interruption_cascade "
        "ON cayu_checkpoints(session_id) "
        "WHERE state ? 'pending_interruption_cascade'",
    ),
    17: (
        "ALTER TABLE cayu_events ADD COLUMN IF NOT EXISTS pending_action_lookup_key TEXT",
        "ALTER TABLE cayu_events ADD COLUMN IF NOT EXISTS pending_action_projection JSONB",
        "ALTER TABLE cayu_events ADD COLUMN IF NOT EXISTS pending_action_projection_bytes BIGINT",
        "ALTER TABLE cayu_checkpoints ADD COLUMN IF NOT EXISTS pending_action_source_bytes BIGINT",
        "ALTER TABLE cayu_checkpoints ADD COLUMN IF NOT EXISTS "
        "pending_action_tool_call_count INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE cayu_checkpoints ADD COLUMN IF NOT EXISTS "
        "pending_action_flags INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE cayu_checkpoints ADD COLUMN IF NOT EXISTS "
        "pending_action_metrics_ready BOOLEAN NOT NULL DEFAULT FALSE",
    ),
    18: (
        """
        CREATE TABLE IF NOT EXISTS cayu_session_operations (
            session_id TEXT NOT NULL REFERENCES cayu_sessions(id) ON DELETE CASCADE,
            idempotency_key TEXT NOT NULL,
            record JSONB NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            PRIMARY KEY (session_id, idempotency_key)
        )
        """,
    ),
    19: (
        """
        CREATE TABLE IF NOT EXISTS cayu_session_message_queue (
            ordering_key BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            queue_id TEXT NOT NULL UNIQUE,
            session_id TEXT NOT NULL REFERENCES cayu_sessions(id) ON DELETE CASCADE,
            idempotency_key TEXT NOT NULL,
            content TEXT NOT NULL,
            message_json JSONB,
            delivery_mode TEXT NOT NULL,
            status TEXT NOT NULL,
            requested_by JSONB,
            accepted_run_epoch BIGINT NOT NULL,
            accepted_transcript_cursor BIGINT NOT NULL,
            accepted_event_id TEXT NOT NULL,
            accepted_at TIMESTAMPTZ NOT NULL,
            delivered_run_epoch BIGINT,
            delivered_transcript_cursor BIGINT,
            delivered_event_id TEXT,
            delivered_at TIMESTAMPTZ,
            UNIQUE (session_id, idempotency_key)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_cayu_session_message_queue_delivery "
        "ON cayu_session_message_queue(session_id, status, delivery_mode, ordering_key)",
    ),
    20: (
        """
        CREATE TABLE IF NOT EXISTS cayu_persisted_event_side_effects (
            session_id TEXT NOT NULL,
            event_id TEXT NOT NULL,
            event_sequence BIGINT NOT NULL,
            status TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            claim_id TEXT,
            lease_expires_at TIMESTAMPTZ,
            next_attempt_at TIMESTAMPTZ,
            last_error TEXT,
            updated_at TIMESTAMPTZ NOT NULL,
            PRIMARY KEY (session_id, event_id),
            FOREIGN KEY (session_id, event_id)
                REFERENCES cayu_events(session_id, event_id) ON DELETE CASCADE
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_cayu_persisted_event_side_effects_delivery "
        "ON cayu_persisted_event_side_effects"
        "(status, next_attempt_at, lease_expires_at, event_sequence)",
    ),
    21: (
        "ALTER TABLE IF EXISTS cayu_budget_reservations "
        "ADD COLUMN IF NOT EXISTS billing_identity JSONB",
    ),
    22: (
        """
        CREATE TABLE IF NOT EXISTS cayu_mcp_manifest_baselines (
            history_key TEXT PRIMARY KEY,
            generation BIGINT NOT NULL CHECK (generation >= 1),
            baseline JSONB NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL
        )
        """,
    ),
    26: (
        "ALTER TABLE cayu_events ADD COLUMN IF NOT EXISTS interaction_id TEXT",
        "ALTER TABLE cayu_transcript_messages ADD COLUMN IF NOT EXISTS interaction_id TEXT",
        "ALTER TABLE cayu_sessions ADD COLUMN IF NOT EXISTS "
        "transcript_seq BIGINT NOT NULL DEFAULT 0",
        "ALTER TABLE cayu_transcript_messages ADD COLUMN IF NOT EXISTS session_order BIGINT",
        "ALTER TABLE cayu_transcript_messages ALTER COLUMN session_order SET NOT NULL",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_cayu_transcript_session_order "
        "ON cayu_transcript_messages(session_id, session_order)",
        "CREATE INDEX IF NOT EXISTS idx_cayu_transcript_interaction_order "
        "ON cayu_transcript_messages(session_id, interaction_id, session_order)",
        """
        CREATE OR REPLACE FUNCTION cayu_assign_transcript_order()
        RETURNS TRIGGER AS $$
        BEGIN
            IF NEW.session_order IS NOT NULL THEN
                RAISE EXCEPTION
                    'cayu_transcript_messages.session_order is runtime-owned';
            END IF;
            UPDATE cayu_sessions
            SET transcript_seq = transcript_seq + 1
            WHERE id = NEW.session_id
            RETURNING transcript_seq INTO NEW.session_order;
            IF NEW.session_order IS NULL THEN
                RAISE EXCEPTION 'transcript session does not exist: %', NEW.session_id;
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
        """,
        "DROP TRIGGER IF EXISTS cayu_assign_transcript_order ON cayu_transcript_messages",
        """
        CREATE TRIGGER cayu_assign_transcript_order
        BEFORE INSERT ON cayu_transcript_messages
        FOR EACH ROW EXECUTE FUNCTION cayu_assign_transcript_order()
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_interaction_latest_events (
            session_id TEXT NOT NULL REFERENCES cayu_sessions(id) ON DELETE CASCADE,
            interaction_id TEXT NOT NULL,
            latest_event_sequence BIGINT NOT NULL
                REFERENCES cayu_events(sequence) ON DELETE CASCADE,
            PRIMARY KEY (session_id, interaction_id)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_cayu_interaction_latest_events_page "
        "ON cayu_interaction_latest_events(session_id, latest_event_sequence DESC)",
        """
        CREATE OR REPLACE FUNCTION cayu_track_interaction_latest_event()
        RETURNS TRIGGER AS $$
        BEGIN
            IF NEW.interaction_id IS NOT NULL AND NEW.event_type = ANY(ARRAY[
                'interaction.started', 'interaction.resumed', 'interaction.paused',
                'interaction.completed', 'interaction.failed', 'interaction.interrupted'
            ]) THEN
                INSERT INTO cayu_interaction_latest_events (
                    session_id, interaction_id, latest_event_sequence
                ) VALUES (NEW.session_id, NEW.interaction_id, NEW.sequence)
                ON CONFLICT (session_id, interaction_id) DO UPDATE SET
                    latest_event_sequence = EXCLUDED.latest_event_sequence
                WHERE EXCLUDED.latest_event_sequence
                    > cayu_interaction_latest_events.latest_event_sequence;
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
        """,
        "DROP TRIGGER IF EXISTS cayu_track_interaction_latest_event ON cayu_events",
        """
        CREATE TRIGGER cayu_track_interaction_latest_event
        AFTER INSERT ON cayu_events
        FOR EACH ROW EXECUTE FUNCTION cayu_track_interaction_latest_event()
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_deferred_interaction_inputs (
            session_id TEXT PRIMARY KEY REFERENCES cayu_sessions(id) ON DELETE CASCADE,
            interaction_id TEXT NOT NULL,
            source_messages JSONB NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_session_message_deliveries (
            delivery_id TEXT PRIMARY KEY,
            reject_only BOOLEAN NOT NULL DEFAULT FALSE,
            session_id TEXT NOT NULL REFERENCES cayu_sessions(id) ON DELETE CASCADE,
            interaction_id TEXT,
            include_on_idle BOOLEAN NOT NULL,
            requested_eligible_through BIGINT,
            eligible_through BIGINT NOT NULL,
            batch_limit INTEGER NOT NULL,
            has_more BOOLEAN NOT NULL,
            interaction_started_event JSONB,
            queue_ids JSONB NOT NULL,
            events JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_cayu_session_message_deliveries_session "
        "ON cayu_session_message_deliveries(session_id, created_at)",
    ),
    23: (
        "ALTER TABLE IF EXISTS cayu_budget_reservations "
        "ADD COLUMN IF NOT EXISTS budget_limit_id TEXT",
        "CREATE INDEX IF NOT EXISTS idx_cayu_budget_reservations_limit "
        "ON cayu_budget_reservations(budget_limit_id, status, updated_at)",
        "ALTER TABLE IF EXISTS cayu_budget_reservations "
        "ADD COLUMN IF NOT EXISTS model_step_id TEXT",
        "ALTER TABLE IF EXISTS cayu_budget_reservations "
        "ADD COLUMN IF NOT EXISTS model_attempt_id TEXT",
        "CREATE INDEX IF NOT EXISTS idx_cayu_budget_reservations_model_attempt "
        "ON cayu_budget_reservations(model_attempt_id, budget_limit_id, status)",
        """
        CREATE TABLE IF NOT EXISTS cayu_budget_reservation_identities (
            reservation_id TEXT PRIMARY KEY,
            publication_session_id TEXT NOT NULL,
            publication_id TEXT NOT NULL,
            published BOOLEAN NOT NULL
        )
        """,
        """
        INSERT INTO cayu_budget_reservation_identities (
            reservation_id,
            publication_session_id,
            publication_id,
            published
        )
        SELECT
            payload ->> 'reservation_id',
            session_id,
            event_id,
            TRUE
        FROM cayu_events
        WHERE event_type = 'budget.reserved'
          AND jsonb_typeof(payload -> 'reservation_id') = 'string'
        ON CONFLICT (reservation_id) DO NOTHING
        """,
    ),
    25: (
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM cayu_budget_reservations
                WHERE status = 'active'
            ) THEN
                RAISE EXCEPTION USING MESSAGE =
                    'Schema revision 25 cannot migrate active budget reservations because '
                    'their dispatch state is unknown. Drain or explicitly settle every active '
                    'reservation, then retry the migration.';
            END IF;
        END
        $$
        """,
        "ALTER TABLE IF EXISTS cayu_budget_reservations "
        "ADD COLUMN IF NOT EXISTS environment_name TEXT",
        "ALTER TABLE IF EXISTS cayu_budget_reservations "
        "ADD COLUMN IF NOT EXISTS settlement_event_payload JSONB NOT NULL "
        "DEFAULT '{}'::jsonb",
        "ALTER TABLE IF EXISTS cayu_budget_reservations "
        "ADD COLUMN IF NOT EXISTS settlement_fallback JSONB",
        """
        UPDATE cayu_budget_reservations
        SET settlement_fallback = jsonb_build_object(
            'settled_at', to_jsonb(created_at),
            'reconciliation_reason',
                'model completion settlement evidence was not publishable; '
                'charged reserved amount',
            'release_reason', 'reservation released before provider dispatch',
            'expiration_reason', NULL
        )
        WHERE settlement_fallback IS NULL
        """,
        "ALTER TABLE IF EXISTS cayu_budget_reservations "
        "ALTER COLUMN settlement_fallback SET NOT NULL",
        "ALTER TABLE IF EXISTS cayu_budget_reservations ADD COLUMN IF NOT EXISTS dispatch_id TEXT",
        "ALTER TABLE IF EXISTS cayu_budget_reservations "
        "ADD COLUMN IF NOT EXISTS dispatched_at TIMESTAMPTZ",
        """
        CREATE TABLE IF NOT EXISTS cayu_budget_settlements (
            settlement_id TEXT PRIMARY KEY,
            reservation_id TEXT NOT NULL UNIQUE
                REFERENCES cayu_budget_reservations(reservation_id),
            session_id TEXT NOT NULL,
            settled_at TIMESTAMPTZ NOT NULL,
            settlement_json JSONB NOT NULL,
            event_published BOOLEAN NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_cayu_budget_settlements_pending "
        "ON cayu_budget_settlements"
        "(session_id, event_published, settled_at, settlement_id)",
        "CREATE INDEX IF NOT EXISTS idx_cayu_budget_settlements_pending_global "
        "ON cayu_budget_settlements"
        "(event_published, settled_at, settlement_id)",
        "CREATE INDEX IF NOT EXISTS idx_cayu_budget_reservation_identities_session "
        "ON cayu_budget_reservation_identities(publication_session_id, reservation_id)",
    ),
    28: (
        """
        CREATE TABLE IF NOT EXISTS cayu_public_authority_aliases (
            field_name TEXT NOT NULL,
            scope_session_id TEXT NOT NULL,
            public_alias TEXT NOT NULL,
            private_value TEXT NOT NULL,
            PRIMARY KEY (field_name, scope_session_id, public_alias)
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_cayu_public_authority_private_value
            ON cayu_public_authority_aliases(field_name, scope_session_id, private_value)
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_public_authority_alias_keys (
            key_id TEXT PRIMARY KEY,
            fingerprint TEXT NOT NULL,
            backfill_completed BOOLEAN NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_public_authority_alias_config (
            singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
            active_key_id TEXT NOT NULL REFERENCES cayu_public_authority_alias_keys(key_id),
            keyring_fingerprint TEXT NOT NULL,
            generation BIGINT NOT NULL CHECK (generation >= 1),
            retired_key_ids JSONB NOT NULL CHECK (jsonb_typeof(retired_key_ids) = 'array')
        )
        """,
    ),
    31: (
        "ALTER TABLE cayu_events ADD COLUMN IF NOT EXISTS "
        "input_contract_runtime_owned BOOLEAN NOT NULL DEFAULT FALSE",
    ),
    54: (
        "ALTER TABLE cayu_events ADD COLUMN IF NOT EXISTS "
        "file_attachment_attestations_runtime_owned BOOLEAN NOT NULL DEFAULT FALSE",
    ),
    32: (
        """
        CREATE TABLE IF NOT EXISTS cayu_eval_corpora (
            revision TEXT COLLATE "C" PRIMARY KEY,
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
            document TEXT NOT NULL,
            document_bytes BIGINT NOT NULL
                CHECK (document_bytes >= 1 AND document_bytes <= 8388608)
                CHECK (document_bytes = octet_length(document)),
            created_at TIMESTAMPTZ NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_cayu_eval_corpora_catalog "
        "ON cayu_eval_corpora(created_at DESC, revision ASC)",
        "CREATE INDEX IF NOT EXISTS idx_cayu_eval_corpora_target_catalog "
        "ON cayu_eval_corpora(target_key, created_at DESC, revision ASC)",
        """
        CREATE TABLE IF NOT EXISTS cayu_eval_suites (
            corpus_revision TEXT NOT NULL
                REFERENCES cayu_eval_corpora(revision) ON DELETE CASCADE,
            suite_id TEXT COLLATE "C" NOT NULL,
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
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_eval_cases (
            corpus_revision TEXT NOT NULL,
            case_id TEXT COLLATE "C" NOT NULL,
            case_revision TEXT NOT NULL,
            suite_id TEXT COLLATE "C" NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            message_count INTEGER NOT NULL
                CHECK (message_count >= 1 AND message_count <= 16),
            assertion_count INTEGER NOT NULL
                CHECK (assertion_count >= 1 AND assertion_count <= 64),
            PRIMARY KEY (corpus_revision, case_id),
            FOREIGN KEY (corpus_revision, suite_id)
                REFERENCES cayu_eval_suites(corpus_revision, suite_id) ON DELETE CASCADE
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_cayu_eval_cases_suite "
        "ON cayu_eval_cases(corpus_revision, suite_id, case_id ASC)",
        """
        CREATE TABLE IF NOT EXISTS cayu_eval_runs (
            run_id TEXT COLLATE "C" PRIMARY KEY,
            idempotency_key TEXT NOT NULL UNIQUE,
            corpus_revision TEXT NOT NULL
                REFERENCES cayu_eval_corpora(revision),
            target_key TEXT NOT NULL,
            suite_id TEXT COLLATE "C" NOT NULL,
            suite_revision TEXT NOT NULL,
            max_concurrency INTEGER NOT NULL
                CHECK (max_concurrency >= 1 AND max_concurrency <= 32),
            invocation_json TEXT NOT NULL,
            status TEXT NOT NULL CHECK (
                status IN ('queued', 'running', 'cancelling', 'completed', 'failed', 'cancelled')
            ),
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            started_at TIMESTAMPTZ,
            finished_at TIMESTAMPTZ,
            cancel_requested_at TIMESTAMPTZ,
            claim_id TEXT,
            ownership_epoch BIGINT NOT NULL DEFAULT 0
                CHECK (ownership_epoch >= 0 AND ownership_epoch <= 9223372036854775807),
            lease_expires_at TIMESTAMPTZ,
            result_revision TEXT,
            result_status TEXT CHECK (
                result_status IS NULL
                OR result_status IN ('passed', 'failed', 'unavailable', 'error')
            ),
            result_score DOUBLE PRECISION CHECK (
                result_score IS NULL OR (result_score >= 0.0 AND result_score <= 1.0)
            ),
            result_duration_ms BIGINT CHECK (
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
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_cayu_eval_runs_catalog "
        "ON cayu_eval_runs(created_at DESC, run_id ASC)",
        "CREATE INDEX IF NOT EXISTS idx_cayu_eval_runs_status_claim "
        "ON cayu_eval_runs(status, lease_expires_at, created_at ASC, run_id ASC)",
        "CREATE INDEX IF NOT EXISTS idx_cayu_eval_runs_corpus_catalog "
        "ON cayu_eval_runs(corpus_revision, created_at DESC, run_id ASC)",
        """
        CREATE TABLE IF NOT EXISTS cayu_eval_results (
            run_id TEXT PRIMARY KEY
                REFERENCES cayu_eval_runs(run_id) ON DELETE RESTRICT,
            revision TEXT NOT NULL,
            result TEXT NOT NULL,
            result_bytes BIGINT NOT NULL
                CHECK (result_bytes >= 1 AND result_bytes <= 41943040)
                CHECK (result_bytes = octet_length(result)),
            created_at TIMESTAMPTZ NOT NULL
        )
        """,
    ),
    33: (
        "CREATE INDEX IF NOT EXISTS idx_cayu_eval_runs_target_catalog "
        "ON cayu_eval_runs(target_key, created_at DESC, run_id ASC)",
        "CREATE INDEX IF NOT EXISTS idx_cayu_eval_runs_target_status_claim "
        "ON cayu_eval_runs("
        "target_key, status, lease_expires_at, created_at ASC, run_id ASC)",
    ),
    34: ("ALTER TABLE cayu_tasks ADD COLUMN IF NOT EXISTS available_at TIMESTAMPTZ",),
    35: (
        """
        CREATE TABLE IF NOT EXISTS cayu_knowledge_publication_receipts (
            operation_id TEXT PRIMARY KEY,
            entry_id TEXT NOT NULL,
            request_sha256 TEXT NOT NULL,
            entry_created_at TIMESTAMPTZ NOT NULL,
            entry_updated_at TIMESTAMPTZ NOT NULL,
            committed_at TIMESTAMPTZ NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_publication_receipts_entry "
        "ON cayu_knowledge_publication_receipts(entry_id)",
    ),
    36: ("ALTER TABLE cayu_sessions ADD COLUMN IF NOT EXISTS invocation JSONB NOT NULL",),
    38: (
        """
        CREATE TABLE IF NOT EXISTS cayu_task_terminalization_receipts (
            task_id TEXT NOT NULL,
            idempotency_key TEXT NOT NULL,
            request_sha256 TEXT NOT NULL,
            worker_id TEXT NOT NULL,
            terminal_kind TEXT NOT NULL,
            task_json JSONB NOT NULL,
            committed_at TIMESTAMPTZ NOT NULL,
            PRIMARY KEY (task_id, idempotency_key)
        )
        """,
    ),
    39: ("ALTER TABLE cayu_tasks ADD COLUMN IF NOT EXISTS invocation JSONB NOT NULL",),
    41: (
        "ALTER TABLE cayu_knowledge_publication_receipts "
        "ADD COLUMN IF NOT EXISTS access_snapshot JSONB NOT NULL",
    ),
    42: (
        # Embeddings are derived and the preflight rejects any populated table.
        # Preserve an existing empty vector(N) table because its dimension is an
        # application-owned schema choice that the generic storage migration
        # cannot reconstruct. Detach it while the authoritative tables are
        # rebuilt, then restore its canonical foreign keys below.
        "ALTER TABLE IF EXISTS cayu_knowledge_embeddings "
        "DROP CONSTRAINT IF EXISTS cayu_knowledge_embeddings_chunk_id_fkey",
        "ALTER TABLE IF EXISTS cayu_knowledge_embeddings "
        "DROP CONSTRAINT IF EXISTS cayu_knowledge_embeddings_entry_id_fkey",
        "DROP TABLE IF EXISTS cayu_knowledge_change_acknowledgements",
        "DROP TABLE IF EXISTS cayu_knowledge_change_consumers",
        "DROP TABLE IF EXISTS cayu_knowledge_change_labels",
        "DROP TABLE IF EXISTS cayu_knowledge_change_audiences",
        "DROP TABLE IF EXISTS cayu_knowledge_changes",
        "DROP TABLE IF EXISTS cayu_knowledge_evidence",
        "DROP VIEW IF EXISTS cayu_knowledge_current_entries",
        "DROP TABLE IF EXISTS cayu_knowledge_publication_receipts",
        "DROP TABLE IF EXISTS cayu_knowledge_chunks",
        "DROP TABLE IF EXISTS cayu_knowledge_impact_targets",
        "DROP TABLE IF EXISTS cayu_knowledge_aspects",
        "DROP TABLE IF EXISTS cayu_knowledge_labels",
        "ALTER TABLE IF EXISTS cayu_knowledge_entries "
        "DROP CONSTRAINT IF EXISTS cayu_knowledge_entries_current_revision_fk",
        "DROP TABLE IF EXISTS cayu_knowledge_revisions",
        "DROP TABLE IF EXISTS cayu_knowledge_entries",
        """
        CREATE TABLE cayu_knowledge_entries (
            id TEXT PRIMARY KEY,
            namespace TEXT NOT NULL,
            current_revision INTEGER NOT NULL
                CHECK (current_revision > 0 AND current_revision <= 2147483647),
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL
        )
        """,
        """
        CREATE TABLE cayu_knowledge_revisions (
            entry_id TEXT NOT NULL REFERENCES cayu_knowledge_entries(id) ON DELETE CASCADE,
            revision INTEGER NOT NULL CHECK (revision > 0 AND revision <= 2147483647),
            text TEXT NOT NULL,
            kind TEXT NOT NULL,
            visibility TEXT NOT NULL,
            status TEXT NOT NULL,
            created_by_type TEXT NOT NULL,
            created_by TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            source_type TEXT,
            source_uri TEXT,
            source_id TEXT,
            source_hash TEXT,
            importance DOUBLE PRECISION,
            importance_source TEXT,
            confidence DOUBLE PRECISION,
            last_used_at TIMESTAMPTZ,
            expires_at TIMESTAMPTZ,
            title TEXT,
            metadata JSONB NOT NULL,
            PRIMARY KEY (entry_id, revision)
        )
        """,
        """
        ALTER TABLE cayu_knowledge_entries
        ADD CONSTRAINT cayu_knowledge_entries_current_revision_fk
        FOREIGN KEY (id, current_revision)
        REFERENCES cayu_knowledge_revisions(entry_id, revision)
        DEFERRABLE INITIALLY DEFERRED
        """,
        """
        CREATE TABLE cayu_knowledge_labels (
            entry_id TEXT NOT NULL,
            entry_revision INTEGER NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            PRIMARY KEY (entry_id, entry_revision, key),
            FOREIGN KEY (entry_id, entry_revision)
                REFERENCES cayu_knowledge_revisions(entry_id, revision) ON DELETE CASCADE
        )
        """,
        """
        CREATE TABLE cayu_knowledge_aspects (
            entry_id TEXT NOT NULL,
            entry_revision INTEGER NOT NULL,
            aspect TEXT NOT NULL,
            PRIMARY KEY (entry_id, entry_revision, aspect),
            FOREIGN KEY (entry_id, entry_revision)
                REFERENCES cayu_knowledge_revisions(entry_id, revision) ON DELETE CASCADE
        )
        """,
        """
        CREATE TABLE cayu_knowledge_impact_targets (
            entry_id TEXT NOT NULL,
            entry_revision INTEGER NOT NULL,
            impact_target TEXT NOT NULL,
            PRIMARY KEY (entry_id, entry_revision, impact_target),
            FOREIGN KEY (entry_id, entry_revision)
                REFERENCES cayu_knowledge_revisions(entry_id, revision) ON DELETE CASCADE
        )
        """,
        """
        CREATE TABLE cayu_knowledge_chunks (
            id TEXT PRIMARY KEY,
            entry_id TEXT NOT NULL,
            entry_revision INTEGER NOT NULL
                CHECK (entry_revision > 0 AND entry_revision <= 2147483647),
            chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
            text TEXT NOT NULL,
            content_hash TEXT,
            source_uri TEXT,
            metadata JSONB NOT NULL,
            UNIQUE (entry_id, entry_revision, chunk_index),
            FOREIGN KEY (entry_id, entry_revision)
                REFERENCES cayu_knowledge_revisions(entry_id, revision) ON DELETE CASCADE
        )
        """,
        "ALTER TABLE IF EXISTS cayu_knowledge_embeddings "
        "ADD CONSTRAINT cayu_knowledge_embeddings_chunk_id_fkey "
        "FOREIGN KEY (chunk_id) REFERENCES cayu_knowledge_chunks(id) ON DELETE CASCADE",
        "ALTER TABLE IF EXISTS cayu_knowledge_embeddings "
        "ADD CONSTRAINT cayu_knowledge_embeddings_entry_id_fkey "
        "FOREIGN KEY (entry_id) REFERENCES cayu_knowledge_entries(id) ON DELETE CASCADE",
        """
        CREATE TABLE cayu_knowledge_publication_receipts (
            operation_id TEXT PRIMARY KEY,
            entry_id TEXT NOT NULL,
            entry_revision INTEGER NOT NULL
                CHECK (entry_revision > 0 AND entry_revision <= 2147483647),
            expected_revision INTEGER
                CHECK (expected_revision > 0 AND expected_revision <= 2147483647),
            request_sha256 TEXT NOT NULL,
            entry_created_at TIMESTAMPTZ NOT NULL,
            entry_updated_at TIMESTAMPTZ NOT NULL,
            committed_at TIMESTAMPTZ NOT NULL,
            access_snapshot JSONB NOT NULL,
            CHECK (
                (expected_revision IS NULL AND entry_revision = 1)
                OR entry_revision = expected_revision + 1
            )
        )
        """,
        """
        CREATE VIEW cayu_knowledge_current_entries AS
        SELECT
            logical.id AS id,
            revision.revision AS revision,
            logical.namespace AS namespace,
            revision.text,
            revision.kind,
            revision.visibility,
            revision.status,
            revision.created_by_type,
            revision.created_by,
            revision.created_at,
            revision.updated_at,
            revision.source_type,
            revision.source_uri,
            revision.source_id,
            revision.source_hash,
            revision.importance,
            revision.importance_source,
            revision.confidence,
            revision.last_used_at,
            revision.expires_at,
            revision.title,
            revision.metadata
        FROM cayu_knowledge_entries AS logical
        JOIN cayu_knowledge_revisions AS revision
          ON revision.entry_id = logical.id
         AND revision.revision = logical.current_revision
        """,
        "CREATE INDEX idx_cayu_knowledge_entries_namespace_current "
        "ON cayu_knowledge_entries(namespace, current_revision, id)",
        "CREATE INDEX idx_cayu_knowledge_revisions_status "
        "ON cayu_knowledge_revisions(status, entry_id, revision)",
        "CREATE INDEX idx_cayu_knowledge_revisions_kind "
        "ON cayu_knowledge_revisions(kind, entry_id, revision)",
        "CREATE INDEX idx_cayu_knowledge_revisions_visibility "
        "ON cayu_knowledge_revisions(visibility, entry_id, revision)",
        "CREATE INDEX idx_cayu_knowledge_revisions_source "
        "ON cayu_knowledge_revisions(source_type, source_id, entry_id, revision)",
        "CREATE INDEX idx_cayu_knowledge_revisions_expires_at "
        "ON cayu_knowledge_revisions(expires_at, entry_id, revision)",
        "CREATE INDEX idx_cayu_knowledge_revisions_title_fts "
        "ON cayu_knowledge_revisions USING GIN "
        "(to_tsvector('simple', COALESCE(title, '')))",
        "CREATE INDEX idx_cayu_knowledge_revisions_text_fts "
        "ON cayu_knowledge_revisions USING GIN (to_tsvector('simple', text))",
        "CREATE INDEX idx_cayu_knowledge_labels_key_value_entry "
        "ON cayu_knowledge_labels(key, value, entry_id, entry_revision)",
        "CREATE INDEX idx_cayu_knowledge_aspects_aspect_entry "
        "ON cayu_knowledge_aspects(aspect, entry_id, entry_revision)",
        "CREATE INDEX idx_cayu_knowledge_impact_targets_target_entry "
        "ON cayu_knowledge_impact_targets(impact_target, entry_id, entry_revision)",
        "CREATE INDEX idx_cayu_knowledge_chunks_entry_revision_index "
        "ON cayu_knowledge_chunks(entry_id, entry_revision, chunk_index)",
        "CREATE INDEX idx_cayu_knowledge_chunks_text_fts "
        "ON cayu_knowledge_chunks USING GIN (to_tsvector('simple', text))",
        "CREATE INDEX idx_cayu_knowledge_publication_receipts_entry_revision "
        "ON cayu_knowledge_publication_receipts(entry_id, entry_revision)",
    ),
    43: (
        "ALTER TABLE cayu_knowledge_chunks "
        "ADD CONSTRAINT cayu_knowledge_chunks_identity_owner_key "
        "UNIQUE (id, entry_id, entry_revision)",
        """
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
            locator JSONB NOT NULL,
            disposition TEXT NOT NULL
                CHECK (disposition IN ('live', 'detached', 'retained')),
            created_at TIMESTAMPTZ NOT NULL,
            metadata JSONB NOT NULL,
            CHECK (source_id IS NOT NULL OR source_uri IS NOT NULL),
            CHECK (source_revision IS NOT NULL OR source_hash IS NOT NULL),
            FOREIGN KEY (entry_id, entry_revision)
                REFERENCES cayu_knowledge_revisions(entry_id, revision) ON DELETE CASCADE,
            FOREIGN KEY (chunk_id, entry_id, entry_revision)
                REFERENCES cayu_knowledge_chunks(id, entry_id, entry_revision)
                ON DELETE CASCADE
        )
        """,
        "CREATE INDEX idx_cayu_knowledge_evidence_entry_revision "
        'ON cayu_knowledge_evidence(entry_id, entry_revision, id COLLATE "C")',
        "CREATE INDEX idx_cayu_knowledge_evidence_source "
        "ON cayu_knowledge_evidence(source_type, source_id, entry_id, entry_revision)",
        """
        CREATE TABLE cayu_knowledge_changes (
            sequence BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY
                CHECK (sequence > 0),
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
            committed_at TIMESTAMPTZ NOT NULL,
            operation_id TEXT
        )
        """,
        """
        CREATE TABLE cayu_knowledge_change_audiences (
            change_sequence BIGINT NOT NULL,
            audience_kind TEXT NOT NULL CHECK (audience_kind IN ('before', 'after')),
            namespace TEXT NOT NULL,
            visibility TEXT NOT NULL,
            source_type TEXT,
            source_id TEXT,
            status TEXT NOT NULL,
            requires_include_expired BOOLEAN NOT NULL,
            PRIMARY KEY (change_sequence, audience_kind),
            FOREIGN KEY (change_sequence)
                REFERENCES cayu_knowledge_changes(sequence) ON DELETE CASCADE
        )
        """,
        "CREATE INDEX idx_cayu_knowledge_changes_entry_revision "
        "ON cayu_knowledge_changes(entry_id, entry_revision, sequence)",
        "CREATE UNIQUE INDEX idx_cayu_knowledge_changes_operation "
        "ON cayu_knowledge_changes(operation_id) WHERE operation_id IS NOT NULL",
        "CREATE INDEX idx_cayu_knowledge_change_audiences_namespace "
        "ON cayu_knowledge_change_audiences(namespace, change_sequence, audience_kind)",
        "CREATE INDEX idx_cayu_knowledge_change_audiences_status "
        "ON cayu_knowledge_change_audiences(status, change_sequence, audience_kind)",
        "CREATE INDEX idx_cayu_knowledge_change_audiences_source "
        "ON cayu_knowledge_change_audiences("
        "source_type, source_id, change_sequence, audience_kind)",
        """
        CREATE TABLE cayu_knowledge_change_labels (
            change_sequence BIGINT NOT NULL,
            audience_kind TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            PRIMARY KEY (change_sequence, audience_kind, key),
            FOREIGN KEY (change_sequence, audience_kind)
                REFERENCES cayu_knowledge_change_audiences(
                    change_sequence, audience_kind
                ) ON DELETE CASCADE
        )
        """,
        "CREATE INDEX idx_cayu_knowledge_change_labels_lookup "
        "ON cayu_knowledge_change_labels("
        "key, value, change_sequence, audience_kind)",
        """
        CREATE TABLE cayu_knowledge_change_consumers (
            consumer_id TEXT PRIMARY KEY,
            access_scope_sha256 TEXT NOT NULL,
            cursor_sequence BIGINT NOT NULL DEFAULT 0
                CHECK (cursor_sequence >= 0),
            pending_change_sequence BIGINT,
            pending_claim_id TEXT,
            pending_worker_id TEXT,
            pending_attempt INTEGER NOT NULL DEFAULT 0
                CHECK (pending_attempt >= 0),
            claimed_at TIMESTAMPTZ,
            lease_expires_at TIMESTAMPTZ,
            last_acknowledged_claim_id TEXT,
            updated_at TIMESTAMPTZ NOT NULL,
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
        )
        """,
        "CREATE INDEX idx_cayu_knowledge_change_consumers_lease "
        "ON cayu_knowledge_change_consumers(lease_expires_at) "
        "WHERE pending_change_sequence IS NOT NULL",
        """
        CREATE TABLE cayu_knowledge_change_acknowledgements (
            consumer_id TEXT NOT NULL,
            claim_id TEXT NOT NULL,
            claim_sha256 TEXT NOT NULL CHECK (claim_sha256 ~ '^[0-9a-f]{64}$'),
            change_sequence BIGINT NOT NULL,
            acknowledged_at TIMESTAMPTZ NOT NULL,
            PRIMARY KEY (consumer_id, claim_id),
            FOREIGN KEY (consumer_id)
                REFERENCES cayu_knowledge_change_consumers(consumer_id) ON DELETE CASCADE,
            FOREIGN KEY (change_sequence)
                REFERENCES cayu_knowledge_changes(sequence)
        )
        """,
    ),
    44: (
        "DROP TABLE IF EXISTS cayu_knowledge_embeddings",
        """
        CREATE TABLE IF NOT EXISTS cayu_knowledge_index_readiness_events (
            sequence BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY
                CHECK (sequence > 0),
            identity_sha256 TEXT NOT NULL
                CHECK (identity_sha256 ~ '^[0-9a-f]{64}$'),
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
            update_sha256 TEXT NOT NULL
                CHECK (update_sha256 ~ '^[0-9a-f]{64}$'),
            published_at TIMESTAMPTZ NOT NULL,
            CHECK (
                (state = 'failed' AND failure_code IS NOT NULL)
                OR (state <> 'failed' AND failure_code IS NULL)
            ),
            UNIQUE (identity_sha256, sequence)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_knowledge_index_readiness_current (
            identity_sha256 TEXT PRIMARY KEY,
            sequence BIGINT NOT NULL UNIQUE,
            FOREIGN KEY (identity_sha256, sequence)
                REFERENCES cayu_knowledge_index_readiness_events(
                    identity_sha256, sequence
                )
                ON DELETE CASCADE
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_index_readiness_identity_sequence "
        "ON cayu_knowledge_index_readiness_events(identity_sha256, sequence)",
        "CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_index_readiness_entry_revision "
        "ON cayu_knowledge_index_readiness_events("
        "entry_id, entry_revision, projection_type, sequence)",
        "CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_index_readiness_projection_lookup "
        "ON cayu_knowledge_index_readiness_events("
        "entry_id, entry_revision, chunk_id, projection_type, "
        "embedding_model, dimensions, sequence)",
    ),
    45: (
        "ALTER TABLE cayu_tasks ADD COLUMN IF NOT EXISTS retry_series JSONB",
        """
        CREATE TABLE IF NOT EXISTS cayu_task_retry_settlements (
            task_id TEXT NOT NULL,
            idempotency_key TEXT NOT NULL,
            request_sha256 TEXT NOT NULL,
            receipt_json JSONB NOT NULL,
            committed_at TIMESTAMPTZ NOT NULL,
            PRIMARY KEY (task_id, idempotency_key)
        )
        """,
    ),
    46: (
        "ALTER TABLE cayu_transcript_messages ADD COLUMN IF NOT EXISTS "
        "transcript_search_document TEXT NOT NULL",
        """
        CREATE TABLE IF NOT EXISTS cayu_transcript_search_configuration (
            singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
            tokenizer_version TEXT NOT NULL
        )
        """,
        f"""
        INSERT INTO cayu_transcript_search_configuration (singleton, tokenizer_version)
        VALUES (TRUE, '{TRANSCRIPT_SEARCH_TOKENIZER_VERSION}')
        ON CONFLICT (singleton) DO NOTHING
        """,
    ),
    47: (
        """
        CREATE TABLE IF NOT EXISTS cayu_eval_result_records (
            revision TEXT COLLATE "C" PRIMARY KEY,
            origin TEXT NOT NULL CHECK (origin IN ('captured_session', 'fresh_execution')),
            target_key TEXT NOT NULL,
            corpus_revision TEXT NOT NULL,
            suite_id TEXT COLLATE "C" NOT NULL,
            suite_revision TEXT NOT NULL,
            application_release_id TEXT NOT NULL,
            app_manifest_schema_version TEXT NOT NULL,
            app_manifest_fingerprint TEXT NOT NULL
                CHECK (app_manifest_fingerprint ~ '^[0-9a-f]{64}$'),
            result_status TEXT NOT NULL CHECK (
                result_status IN ('passed', 'failed', 'unavailable', 'error')
            ),
            result_score DOUBLE PRECISION CHECK (
                result_score IS NULL OR (result_score >= 0.0 AND result_score <= 1.0)
            ),
            fresh_run_id TEXT UNIQUE REFERENCES cayu_eval_results(run_id) ON DELETE RESTRICT,
            captured_result TEXT,
            document_bytes BIGINT NOT NULL
                CHECK (document_bytes >= 1 AND document_bytes <= 41943040),
            created_at TIMESTAMPTZ NOT NULL,
            CHECK (
                (result_status IN ('passed', 'failed') AND result_score IS NOT NULL)
                OR (result_status IN ('unavailable', 'error') AND result_score IS NULL)
            ),
            CHECK (
                (origin = 'fresh_execution' AND fresh_run_id IS NOT NULL
                    AND captured_result IS NULL)
                OR (origin = 'captured_session' AND fresh_run_id IS NULL
                    AND captured_result IS NOT NULL
                    AND document_bytes = octet_length(captured_result))
            ),
            FOREIGN KEY (corpus_revision, suite_id)
                REFERENCES cayu_eval_suites(corpus_revision, suite_id)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_cayu_eval_result_records_target_catalog "
        "ON cayu_eval_result_records(target_key, created_at DESC, revision ASC)",
        "CREATE INDEX IF NOT EXISTS idx_cayu_eval_result_records_contract "
        "ON cayu_eval_result_records("
        "target_key, corpus_revision, suite_id, created_at DESC, revision ASC)",
        """
        INSERT INTO cayu_eval_result_records (
            revision, origin, target_key, corpus_revision, suite_id, suite_revision,
            application_release_id, app_manifest_schema_version,
            app_manifest_fingerprint, result_status, result_score, fresh_run_id,
            captured_result, document_bytes, created_at
        )
        SELECT
            result.revision, 'fresh_execution', run.target_key, run.corpus_revision,
            run.suite_id, run.suite_revision,
            result.result::jsonb #>> '{target,application_release_id}',
            result.result::jsonb #>> '{target,app_manifest,schema_version}',
            result.result::jsonb #>> '{target,app_manifest,fingerprint}',
            run.result_status, run.result_score, result.run_id, NULL,
            result.result_bytes, result.created_at
        FROM cayu_eval_results AS result
        JOIN cayu_eval_runs AS run ON run.run_id = result.run_id
        ON CONFLICT (revision) DO NOTHING
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_eval_baselines (
            target_key TEXT NOT NULL,
            corpus_revision TEXT NOT NULL,
            suite_id TEXT COLLATE "C" NOT NULL,
            result_revision TEXT NOT NULL
                REFERENCES cayu_eval_result_records(revision) ON DELETE RESTRICT,
            generation BIGINT NOT NULL
                CHECK (generation >= 1 AND generation <= 9223372036854775807),
            updated_by TEXT NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            PRIMARY KEY (target_key, corpus_revision, suite_id),
            FOREIGN KEY (corpus_revision, suite_id)
                REFERENCES cayu_eval_suites(corpus_revision, suite_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_eval_baseline_mutations (
            operation_id TEXT PRIMARY KEY,
            target_key TEXT NOT NULL,
            corpus_revision TEXT NOT NULL,
            suite_id TEXT COLLATE "C" NOT NULL,
            expected_generation BIGINT NOT NULL
                CHECK (expected_generation >= 0
                    AND expected_generation < 9223372036854775807),
            previous_result_revision TEXT,
            selected_result_revision TEXT NOT NULL
                REFERENCES cayu_eval_result_records(revision) ON DELETE RESTRICT,
            resulting_generation BIGINT NOT NULL
                CHECK (resulting_generation >= 1
                    AND resulting_generation <= 9223372036854775807),
            actor_id TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            CHECK (resulting_generation = expected_generation + 1),
            CHECK (
                (expected_generation = 0 AND previous_result_revision IS NULL)
                OR (expected_generation > 0 AND previous_result_revision IS NOT NULL)
            ),
            FOREIGN KEY (previous_result_revision)
                REFERENCES cayu_eval_result_records(revision) ON DELETE RESTRICT,
            FOREIGN KEY (corpus_revision, suite_id)
                REFERENCES cayu_eval_suites(corpus_revision, suite_id)
        )
        """,
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_cayu_eval_baseline_mutations_scope "
        "ON cayu_eval_baseline_mutations("
        "target_key, corpus_revision, suite_id, resulting_generation)",
    ),
    48: (
        "ALTER TABLE cayu_eval_cases DROP CONSTRAINT IF EXISTS cayu_eval_cases_message_count_check",
        "ALTER TABLE cayu_eval_cases ADD CONSTRAINT "
        "cayu_eval_cases_message_count_check "
        "CHECK (message_count >= 0 AND message_count <= 16)",
    ),
    49: (
        "ALTER TABLE cayu_tasks ADD COLUMN IF NOT EXISTS work_contract JSONB",
        """
        CREATE TABLE IF NOT EXISTS cayu_work_contracts (
            contract_id TEXT NOT NULL,
            version BIGINT NOT NULL CHECK (version >= 1),
            fingerprint TEXT NOT NULL CHECK (fingerprint ~ '^[0-9a-f]{64}$'),
            contract_json JSONB NOT NULL,
            PRIMARY KEY (contract_id, version)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_task_session_execution_authority (
            session_id TEXT PRIMARY KEY,
            authority_kind TEXT NOT NULL CHECK (
                authority_kind IN ('ordinary', 'contracted')
            ),
            committed_at TIMESTAMPTZ NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_work_attempts (
            attempt_id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL REFERENCES cayu_tasks(id) ON DELETE RESTRICT,
            ordinal BIGINT NOT NULL CHECK (ordinal >= 1),
            request_sha256 TEXT NOT NULL CHECK (request_sha256 ~ '^[0-9a-f]{64}$'),
            started_at TIMESTAMPTZ NOT NULL,
            attempt_json JSONB NOT NULL,
            UNIQUE (task_id, ordinal)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_completion_proposals (
            proposal_id TEXT PRIMARY KEY,
            attempt_id TEXT NOT NULL UNIQUE
                REFERENCES cayu_work_attempts(attempt_id) ON DELETE RESTRICT,
            task_id TEXT NOT NULL REFERENCES cayu_tasks(id) ON DELETE RESTRICT,
            request_sha256 TEXT NOT NULL CHECK (request_sha256 ~ '^[0-9a-f]{64}$'),
            proposed_at TIMESTAMPTZ NOT NULL,
            proposal_json JSONB NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_completion_verification_claims (
            claim_id TEXT PRIMARY KEY,
            proposal_id TEXT NOT NULL
                REFERENCES cayu_completion_proposals(proposal_id) ON DELETE RESTRICT,
            attempt_number BIGINT NOT NULL CHECK (attempt_number >= 1),
            request_sha256 TEXT NOT NULL CHECK (request_sha256 ~ '^[0-9a-f]{64}$'),
            lease_expires_at TIMESTAMPTZ NOT NULL,
            is_current BOOLEAN NOT NULL,
            claim_json JSONB NOT NULL,
            UNIQUE (proposal_id, attempt_number)
        )
        """,
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_cayu_completion_claim_current "
        "ON cayu_completion_verification_claims(proposal_id) WHERE is_current",
        """
        CREATE TABLE IF NOT EXISTS cayu_completion_decisions (
            decision_id TEXT PRIMARY KEY,
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
            gap_fingerprint TEXT NOT NULL CHECK (gap_fingerprint ~ '^[0-9a-f]{64}$'),
            request_sha256 TEXT NOT NULL CHECK (request_sha256 ~ '^[0-9a-f]{64}$'),
            decided_at TIMESTAMPTZ NOT NULL,
            decision_json JSONB NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_cayu_completion_decisions_task_gap "
        "ON cayu_completion_decisions(task_id, verdict, gap_fingerprint)",
        """
        CREATE TABLE IF NOT EXISTS cayu_completion_decision_application_receipts (
            task_id TEXT NOT NULL REFERENCES cayu_tasks(id) ON DELETE RESTRICT,
            idempotency_key TEXT NOT NULL,
            decision_id TEXT NOT NULL UNIQUE
                REFERENCES cayu_completion_decisions(decision_id) ON DELETE RESTRICT,
            request_sha256 TEXT NOT NULL CHECK (request_sha256 ~ '^[0-9a-f]{64}$'),
            applied_at TIMESTAMPTZ NOT NULL,
            receipt_json JSONB NOT NULL,
            PRIMARY KEY (task_id, idempotency_key)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_cayu_tasks_contracted_session "
        "ON cayu_tasks(session_id, created_at, id) WHERE work_contract IS NOT NULL",
        "CREATE INDEX IF NOT EXISTS idx_cayu_work_attempts_task_latest "
        "ON cayu_work_attempts(task_id, ordinal DESC)",
    ),
    50: (
        "ALTER TABLE cayu_eval_runs ADD COLUMN IF NOT EXISTS invocation_json TEXT "
        "NOT NULL DEFAULT "
        '\'{"schema_version":1,"source":"sdk_run","origin":null,'
        '"max_steps":null,"limits":null,"cost_budget":null}\'',
        "ALTER TABLE cayu_eval_runs ALTER COLUMN invocation_json DROP DEFAULT",
    ),
    51: (
        """
        CREATE TABLE IF NOT EXISTS cayu_recall_receipts (
            receipt_id TEXT COLLATE "C" PRIMARY KEY,
            session_id TEXT COLLATE "C" NOT NULL
                REFERENCES cayu_sessions(id) ON DELETE CASCADE,
            interaction_id TEXT COLLATE "C" NOT NULL,
            model_step_id TEXT COLLATE "C" NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            receipt_json JSONB NOT NULL,
            document_bytes BIGINT NOT NULL CHECK (
                document_bytes >= 1 AND document_bytes <= 256000
            )
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_context_exposures (
            exposure_id TEXT COLLATE "C" PRIMARY KEY,
            session_id TEXT COLLATE "C" NOT NULL
                REFERENCES cayu_sessions(id) ON DELETE CASCADE,
            interaction_id TEXT COLLATE "C" NOT NULL,
            model_step_id TEXT COLLATE "C" NOT NULL,
            model_attempt_id TEXT COLLATE "C" NOT NULL,
            provider_attempt_id TEXT COLLATE "C" NOT NULL,
            state TEXT NOT NULL CHECK (state IN (
                'planned', 'prepared', 'dispatch_started', 'acknowledged',
                'completed', 'failed', 'cancelled', 'indeterminate'
            )),
            state_revision INTEGER NOT NULL CHECK (
                state_revision >= 0 AND state_revision < 16
            ),
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            exposure_json JSONB NOT NULL,
            document_bytes BIGINT NOT NULL CHECK (
                document_bytes >= 1 AND document_bytes <= 128000
            ),
            UNIQUE (session_id, model_attempt_id),
            UNIQUE (session_id, provider_attempt_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_recall_item_exposures (
            exposure_id TEXT COLLATE "C" NOT NULL
                REFERENCES cayu_context_exposures(exposure_id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL CHECK (ordinal >= 0 AND ordinal < 64),
            receipt_id TEXT COLLATE "C" NOT NULL
                REFERENCES cayu_recall_receipts(receipt_id) ON DELETE CASCADE,
            receipt_item_ordinal INTEGER NOT NULL CHECK (
                receipt_item_ordinal >= 0 AND receipt_item_ordinal < 64
            ),
            item_json JSONB NOT NULL,
            document_bytes BIGINT NOT NULL CHECK (
                document_bytes >= 1 AND document_bytes <= 16384
            ),
            PRIMARY KEY (exposure_id, ordinal),
            UNIQUE (exposure_id, receipt_id, receipt_item_ordinal)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_cayu_recall_receipts_session_page "
        'ON cayu_recall_receipts(session_id, created_at, receipt_id COLLATE "C")',
        "CREATE INDEX IF NOT EXISTS idx_cayu_recall_receipts_interaction_page "
        "ON cayu_recall_receipts("
        'session_id, interaction_id, created_at, receipt_id COLLATE "C")',
        "CREATE INDEX IF NOT EXISTS idx_cayu_recall_receipts_step_page "
        "ON cayu_recall_receipts("
        'session_id, model_step_id, created_at, receipt_id COLLATE "C")',
        "CREATE INDEX IF NOT EXISTS idx_cayu_recall_receipts_interaction_step_page "
        "ON cayu_recall_receipts(session_id, interaction_id, model_step_id, created_at, "
        'receipt_id COLLATE "C")',
        "CREATE INDEX IF NOT EXISTS idx_cayu_context_exposures_session_page "
        'ON cayu_context_exposures(session_id, created_at, exposure_id COLLATE "C")',
        "CREATE INDEX IF NOT EXISTS idx_cayu_context_exposures_interaction_page "
        "ON cayu_context_exposures("
        'session_id, interaction_id, created_at, exposure_id COLLATE "C")',
        "CREATE INDEX IF NOT EXISTS idx_cayu_context_exposures_step_page "
        "ON cayu_context_exposures("
        'session_id, model_step_id, created_at, exposure_id COLLATE "C")',
        "CREATE INDEX IF NOT EXISTS idx_cayu_context_exposures_interaction_step_page "
        "ON cayu_context_exposures(session_id, interaction_id, model_step_id, created_at, "
        'exposure_id COLLATE "C")',
        "CREATE INDEX IF NOT EXISTS idx_cayu_recall_item_exposures_receipt "
        "ON cayu_recall_item_exposures(receipt_id, exposure_id, ordinal)",
    ),
    52: (
        """
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
            issued_at TIMESTAMPTZ NOT NULL,
            expires_at TIMESTAMPTZ NOT NULL,
            max_calls BIGINT NOT NULL CHECK (max_calls >= 1 AND max_calls <= 32),
            used_calls BIGINT NOT NULL DEFAULT 0
                CHECK (used_calls >= 0 AND used_calls <= max_calls),
            revoked_at TIMESTAMPTZ,
            record JSONB NOT NULL,
            UNIQUE (session_id, interaction_id, request_id),
            UNIQUE (session_id, interaction_id, tool_id)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_cayu_targeted_tool_grants_interaction "
        "ON cayu_targeted_tool_grants(session_id, interaction_id, issued_at, grant_id)",
        """
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
            bound_at TIMESTAMPTZ NOT NULL,
            record JSONB NOT NULL,
            UNIQUE (session_id, interaction_id, invocation_id),
            UNIQUE (session_id, interaction_id, outer_tool_call_id)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_cayu_targeted_tool_grant_uses_grant "
        "ON cayu_targeted_tool_grant_uses(grant_id, bound_at, use_id)",
    ),
    53: (
        """
        CREATE TABLE IF NOT EXISTS cayu_eval_scenarios (
            revision TEXT COLLATE "C" PRIMARY KEY,
            scenario_id TEXT COLLATE "C" NOT NULL,
            target_key TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            event_count BIGINT NOT NULL
                CONSTRAINT cayu_eval_scenarios_event_count_check
                CHECK (event_count BETWEEN 1 AND 1024),
            input_event_count BIGINT NOT NULL
                CONSTRAINT cayu_eval_scenarios_input_event_count_check
                CHECK (input_event_count BETWEEN 1 AND 1024),
            approval_checkpoint_count BIGINT NOT NULL
                CONSTRAINT cayu_eval_scenarios_approval_checkpoint_count_check
                CHECK (approval_checkpoint_count BETWEEN 0 AND 1024),
            message_count BIGINT NOT NULL
                CONSTRAINT cayu_eval_scenarios_message_count_check
                CHECK (message_count >= input_event_count AND message_count <= 32768),
            part_count BIGINT NOT NULL
                CONSTRAINT cayu_eval_scenarios_part_count_check
                CHECK (part_count >= message_count AND part_count <= 1048576),
            artifact_requirement_count BIGINT NOT NULL
                CONSTRAINT cayu_eval_scenarios_artifact_requirement_count_check
                CHECK (artifact_requirement_count BETWEEN 0 AND 128),
            secret_requirement_count BIGINT NOT NULL
                CONSTRAINT cayu_eval_scenarios_secret_requirement_count_check
                CHECK (secret_requirement_count BETWEEN 0 AND 128),
            document_json TEXT NOT NULL,
            document_bytes BIGINT NOT NULL
                CONSTRAINT cayu_eval_scenarios_document_bytes_check
                CHECK (document_bytes BETWEEN 1 AND 8388608)
                CONSTRAINT cayu_eval_scenarios_document_size_check
                CHECK (document_bytes = octet_length(document_json)),
            created_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT cayu_eval_scenarios_event_partition_check
                CHECK (input_event_count + approval_checkpoint_count = event_count),
            CONSTRAINT cayu_eval_scenarios_document_json_check
                CHECK (document_json::jsonb IS NOT NULL)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_cayu_eval_scenarios_catalog "
        "ON cayu_eval_scenarios(created_at DESC, revision ASC)",
        "CREATE INDEX IF NOT EXISTS idx_cayu_eval_scenarios_target_catalog "
        "ON cayu_eval_scenarios(target_key, created_at DESC, revision ASC)",
        "CREATE INDEX IF NOT EXISTS idx_cayu_eval_scenarios_id_catalog "
        "ON cayu_eval_scenarios(scenario_id, created_at DESC, revision ASC)",
    ),
    55: (
        """
        CREATE TABLE IF NOT EXISTS cayu_task_retry_reconciliation_rejections (
            task_id TEXT NOT NULL,
            reconciliation_idempotency_key TEXT NOT NULL,
            request_sha256 TEXT NOT NULL,
            record_json JSONB NOT NULL,
            recorded_at TIMESTAMPTZ NOT NULL,
            PRIMARY KEY (task_id, reconciliation_idempotency_key)
        )
        """,
    ),
    56: (
        "ALTER TABLE cayu_eval_runs ADD COLUMN IF NOT EXISTS "
        "scenario_progress_json TEXT CHECK (scenario_progress_json IS NULL OR "
        "(octet_length(scenario_progress_json) BETWEEN 1 AND 262144 AND "
        "scenario_progress_json::jsonb IS NOT NULL))",
    ),
    57: ("ALTER TABLE cayu_session_message_queue ADD COLUMN IF NOT EXISTS message_json JSONB",),
    74: (
        "ALTER TABLE cayu_eval_runs ADD COLUMN IF NOT EXISTS "
        "trial_checkpoint_count INTEGER NOT NULL DEFAULT 0 CHECK "
        "(trial_checkpoint_count BETWEEN 0 AND 100000)",
        "ALTER TABLE cayu_eval_runs ADD COLUMN IF NOT EXISTS "
        "trial_checkpoint_bytes BIGINT NOT NULL DEFAULT 0 CHECK "
        "(trial_checkpoint_bytes BETWEEN 0 AND 41943040)",
        "ALTER TABLE cayu_eval_runs ADD COLUMN IF NOT EXISTS "
        "authored_suite_launch_revision TEXT CHECK "
        "(authored_suite_launch_revision IS NULL OR "
        "authored_suite_launch_revision ~ '^sha256:[0-9a-f]{64}$')",
        "ALTER TABLE cayu_eval_runs ADD COLUMN IF NOT EXISTS "
        "authored_suite_launch_lane INTEGER CHECK "
        "(authored_suite_launch_lane IS NULL OR "
        "authored_suite_launch_lane BETWEEN 0 AND 63)",
        "CREATE INDEX IF NOT EXISTS idx_cayu_eval_runs_authored_suite_launch_claim "
        "ON cayu_eval_runs(authored_suite_launch_revision, authored_suite_launch_lane, "
        "created_at ASC, run_id ASC, status) "
        "WHERE authored_suite_launch_revision IS NOT NULL",
        """
        CREATE TABLE IF NOT EXISTS cayu_eval_run_trial_checkpoints (
            run_id TEXT COLLATE "C" NOT NULL
                REFERENCES cayu_eval_runs(run_id) ON DELETE CASCADE,
            case_id TEXT COLLATE "C" NOT NULL,
            trial_number INTEGER NOT NULL CHECK (trial_number BETWEEN 1 AND 100),
            checkpoint_json TEXT NOT NULL CHECK (
                octet_length(checkpoint_json) BETWEEN 1 AND 41943040
                AND checkpoint_json::jsonb IS NOT NULL
                AND jsonb_typeof(checkpoint_json::jsonb) = 'object'
            ),
            document_bytes BIGINT NOT NULL CHECK (
                document_bytes BETWEEN 1 AND 41943040
                AND document_bytes = octet_length(checkpoint_json)
            ),
            PRIMARY KEY (run_id, case_id, trial_number)
        )
        """,
    ),
    75: (
        """
        CREATE TABLE IF NOT EXISTS cayu_knowledge_activation_receipts (
            operation_id TEXT COLLATE "C" PRIMARY KEY,
            entry_id TEXT COLLATE "C" NOT NULL,
            entry_revision INTEGER NOT NULL
                CHECK (entry_revision > 0 AND entry_revision <= 2147483647),
            expected_revision INTEGER
                CHECK (expected_revision > 0 AND expected_revision <= 2147483647),
            publication_request_sha256 TEXT COLLATE "C" NOT NULL
                CHECK (publication_request_sha256 ~ '^[0-9a-f]{64}$'),
            committed_at TIMESTAMPTZ NOT NULL,
            receipt_json TEXT NOT NULL CHECK (
                octet_length(receipt_json) BETWEEN 1 AND 1114112
                AND jsonb_typeof(receipt_json::jsonb) = 'object'
            ),
            access_snapshot JSONB NOT NULL CHECK (jsonb_typeof(access_snapshot) = 'object'),
            CHECK (
                (expected_revision IS NULL AND entry_revision = 1)
                OR entry_revision = expected_revision + 1
            )
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_activation_receipts_entry_revision "
        "ON cayu_knowledge_activation_receipts(entry_id, entry_revision)",
        """
        CREATE TABLE IF NOT EXISTS cayu_knowledge_activation_retirements (
            entry_id TEXT COLLATE "C" PRIMARY KEY,
            entry_revision INTEGER NOT NULL
                CHECK (entry_revision > 0 AND entry_revision <= 2147483647),
            retired_at TIMESTAMPTZ NOT NULL,
            retirement_json TEXT NOT NULL CHECK (
                octet_length(retirement_json) BETWEEN 1 AND 1048576
                AND jsonb_typeof(retirement_json::jsonb) = 'object'
            )
        )
        """,
    ),
    77: (
        """
        CREATE TABLE IF NOT EXISTS cayu_knowledge_maintenance_governance_routes (
            operation_id TEXT COLLATE "C" PRIMARY KEY,
            proposal_id TEXT COLLATE "C" NOT NULL UNIQUE,
            proposal_fingerprint TEXT COLLATE "C" NOT NULL
                CHECK (proposal_fingerprint ~ '^[0-9a-f]{64}$'),
            request_sha256 TEXT COLLATE "C" NOT NULL
                CHECK (request_sha256 ~ '^[0-9a-f]{64}$'),
            committed_at TIMESTAMPTZ NOT NULL,
            receipt_json TEXT NOT NULL CHECK (
                octet_length(receipt_json) BETWEEN 1 AND 640000
                AND jsonb_typeof(receipt_json::jsonb) = 'object'
            ),
            access_snapshot JSONB NOT NULL CHECK (jsonb_typeof(access_snapshot) = 'object')
        )
        """,
    ),
    78: (
        """
        CREATE TABLE IF NOT EXISTS cayu_knowledge_semantic_watch_receipts (
            operation_id TEXT COLLATE "C" PRIMARY KEY,
            invocation_sha256 TEXT COLLATE "C" NOT NULL
                CHECK (invocation_sha256 ~ '^[0-9a-f]{64}$'),
            request_sha256 TEXT COLLATE "C" NOT NULL
                CHECK (request_sha256 ~ '^[0-9a-f]{64}$'),
            committed_at TIMESTAMPTZ NOT NULL,
            receipt_json TEXT NOT NULL CHECK (
                octet_length(receipt_json) BETWEEN 1 AND 384000
                AND jsonb_typeof(receipt_json::jsonb) = 'object'
            ),
            access_scope JSONB NOT NULL CHECK (
                octet_length(access_scope::text) BETWEEN 1 AND 384000
                AND jsonb_typeof(access_scope) = 'object'
            )
        )
        """,
    ),
    80: ("ALTER TABLE cayu_eval_runs ADD COLUMN IF NOT EXISTS failure_diagnostic_json TEXT",),
    82: POSTGRES_ACCOUNTING_DDL,
    89: POSTGRES_AUXILIARY_ACCOUNTING_DDL,
    83: (
        SESSION_MESSAGE_ACCEPTANCE_INDEX_DDL,
        "ALTER TABLE cayu_session_message_queue ADD COLUMN IF NOT EXISTS conditions_json JSONB",
        "ALTER TABLE cayu_session_message_queue ADD COLUMN IF NOT EXISTS terminal_json JSONB",
        "ALTER TABLE cayu_session_message_deliveries ADD COLUMN IF NOT EXISTS reject_only BOOLEAN NOT NULL DEFAULT FALSE",
    ),
    84: (
        """
        CREATE TABLE IF NOT EXISTS cayu_work_attempt_preparation_holds (
            hold_id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL REFERENCES cayu_tasks(id) ON DELETE RESTRICT,
            request_sha256 TEXT NOT NULL CHECK (request_sha256 ~ '^[0-9a-f]{64}$'),
            receipt_json TEXT NOT NULL CHECK (
                octet_length(receipt_json) BETWEEN 1 AND 1097728
                AND jsonb_typeof(receipt_json::jsonb) = 'object'
            )
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_work_attempt_lifecycle_receipts (
            admission_id TEXT PRIMARY KEY
                REFERENCES cayu_work_attempt_admissions(admission_id) ON DELETE RESTRICT,
            settlement_id TEXT NOT NULL UNIQUE,
            task_id TEXT NOT NULL REFERENCES cayu_tasks(id) ON DELETE RESTRICT,
            request_sha256 TEXT NOT NULL CHECK (request_sha256 ~ '^[0-9a-f]{64}$'),
            retired_contract_binding BOOLEAN NOT NULL,
            settled_at TIMESTAMPTZ NOT NULL,
            receipt_json TEXT NOT NULL CHECK (
                octet_length(receipt_json) BETWEEN 1 AND 1097728
                AND jsonb_typeof(receipt_json::jsonb) = 'object'
            )
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_cayu_work_attempt_lifecycle_task "
        "ON cayu_work_attempt_lifecycle_receipts(task_id, retired_contract_binding)",
    ),
    85: (
        """
        CREATE TABLE IF NOT EXISTS cayu_session_closure_receipts (
            session_id TEXT NOT NULL,
            plan_id TEXT NOT NULL CHECK (plan_id ~ '^[0-9a-f]{64}$'),
            committed_at TIMESTAMPTZ NOT NULL,
            receipt_json JSONB NOT NULL CHECK (
                octet_length(receipt_json::text) BETWEEN 1 AND 384000
                AND jsonb_typeof(receipt_json) = 'object'
            ),
            PRIMARY KEY (session_id, plan_id)
        )
        """,
    ),
    79: (
        """
        CREATE TABLE IF NOT EXISTS cayu_child_session_lifecycle_candidates (
            child_session_id TEXT COLLATE "C" PRIMARY KEY
                REFERENCES cayu_sessions(id) ON DELETE CASCADE,
            parent_session_id TEXT COLLATE "C" NOT NULL
                REFERENCES cayu_sessions(id) ON DELETE CASCADE,
            priority INTEGER NOT NULL CHECK (priority IN (0, 1, 2)),
            sort_at TIMESTAMPTZ NOT NULL
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_cayu_child_lifecycle_candidates_page
            ON cayu_child_session_lifecycle_candidates(
                parent_session_id, priority, sort_at, child_session_id COLLATE "C"
            )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_cayu_events_child_lifecycle
            ON cayu_events(session_id, event_type, sequence DESC)
            WHERE event_type IN (
                'session.started', 'session.resumed', 'session.forked',
                'session.completed', 'session.failed', 'session.interrupted'
            )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_cayu_transcript_messages_session_role_order
            ON cayu_transcript_messages(
                session_id, (message ->> 'role'), session_order DESC
            )
        """,
        """
        CREATE OR REPLACE FUNCTION cayu_refresh_child_session_lifecycle(
            target_child_session_id TEXT
        ) RETURNS VOID AS $$
        BEGIN
            DELETE FROM cayu_child_session_lifecycle_candidates
            WHERE child_session_id = target_child_session_id;
            INSERT INTO cayu_child_session_lifecycle_candidates (
                child_session_id, parent_session_id, priority, sort_at
            )
            WITH canonical AS (
                SELECT
                    child.id AS child_session_id,
                    child.parent_session_id,
                    child.instance_id,
                    child.status,
                    child.created_at,
                    latest.event_id AS latest_event_id,
                    latest.event_type AS latest_event_type,
                    latest.timestamp AS latest_event_at,
                    CASE
                        WHEN child.status = 'pending' THEN NOT EXISTS (
                            SELECT 1
                            FROM cayu_events AS pending_event
                            WHERE pending_event.session_id = child.id
                              AND pending_event.event_type = ANY(ARRAY[
                                  'session.started', 'session.resumed',
                                  'session.completed', 'session.failed',
                                  'session.interrupted'
                              ])
                        )
                        WHEN child.status IN ('running', 'interrupting') THEN
                            latest.event_type = ANY(ARRAY[
                                'session.started', 'session.resumed', 'session.forked'
                            ])
                        WHEN child.status = 'completed' THEN
                            latest.event_type = 'session.completed'
                        WHEN child.status = 'failed' THEN
                            latest.event_type = 'session.failed'
                        WHEN child.status = 'interrupted' THEN
                            latest.event_type = 'session.interrupted'
                        ELSE FALSE
                    END AS is_available
                FROM cayu_sessions AS child
                LEFT JOIN LATERAL (
                    SELECT event.event_id, event.event_type, event.timestamp
                    FROM cayu_events AS event
                    WHERE event.session_id = child.id
                      AND event.event_type = ANY(ARRAY[
                          'session.started', 'session.resumed', 'session.forked',
                          'session.completed', 'session.failed', 'session.interrupted'
                      ])
                    ORDER BY event.sequence DESC
                    LIMIT 1
                ) AS latest ON TRUE
                WHERE child.id = target_child_session_id
                  AND child.parent_session_id IS NOT NULL
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
                               char_length(canonical.instance_id)::text || ':' ||
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
        END;
        $$ LANGUAGE plpgsql
        """,
        """
        CREATE OR REPLACE FUNCTION cayu_index_child_lifecycle_session()
        RETURNS TRIGGER AS $$
        BEGIN
            PERFORM cayu_refresh_child_session_lifecycle(NEW.id);
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
        """,
        "DROP TRIGGER IF EXISTS cayu_index_child_lifecycle_session ON cayu_sessions",
        """
        CREATE TRIGGER cayu_index_child_lifecycle_session
        AFTER INSERT OR UPDATE OF parent_session_id, instance_id, status, created_at
        ON cayu_sessions
        FOR EACH ROW EXECUTE FUNCTION cayu_index_child_lifecycle_session()
        """,
        """
        CREATE OR REPLACE FUNCTION cayu_index_child_lifecycle_event()
        RETURNS TRIGGER AS $$
        BEGIN
            IF TG_OP = 'DELETE' THEN
                PERFORM cayu_refresh_child_session_lifecycle(OLD.session_id);
                RETURN OLD;
            END IF;
            IF TG_OP = 'UPDATE' AND OLD.session_id <> NEW.session_id THEN
                PERFORM cayu_refresh_child_session_lifecycle(OLD.session_id);
            END IF;
            PERFORM cayu_refresh_child_session_lifecycle(NEW.session_id);
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
        """,
        "DROP TRIGGER IF EXISTS cayu_index_child_lifecycle_event_insert ON cayu_events",
        """
        CREATE TRIGGER cayu_index_child_lifecycle_event_insert
        AFTER INSERT ON cayu_events
        FOR EACH ROW
        WHEN (NEW.event_type IN (
            'session.started', 'session.resumed', 'session.forked',
            'session.completed', 'session.failed', 'session.interrupted'
        ))
        EXECUTE FUNCTION cayu_index_child_lifecycle_event()
        """,
        "DROP TRIGGER IF EXISTS cayu_index_child_lifecycle_event_delete ON cayu_events",
        """
        CREATE TRIGGER cayu_index_child_lifecycle_event_delete
        AFTER DELETE ON cayu_events
        FOR EACH ROW
        WHEN (OLD.event_type IN (
            'session.started', 'session.resumed', 'session.forked',
            'session.completed', 'session.failed', 'session.interrupted'
        ))
        EXECUTE FUNCTION cayu_index_child_lifecycle_event()
        """,
        "DROP TRIGGER IF EXISTS cayu_index_child_lifecycle_event_update ON cayu_events",
        """
        CREATE TRIGGER cayu_index_child_lifecycle_event_update
        AFTER UPDATE OF session_id, event_id, event_type, sequence, timestamp ON cayu_events
        FOR EACH ROW
        WHEN (
            OLD.event_type IN (
                'session.started', 'session.resumed', 'session.forked',
                'session.completed', 'session.failed', 'session.interrupted'
            ) OR NEW.event_type IN (
                'session.started', 'session.resumed', 'session.forked',
                'session.completed', 'session.failed', 'session.interrupted'
            )
        )
        EXECUTE FUNCTION cayu_index_child_lifecycle_event()
        """,
        """
        CREATE OR REPLACE FUNCTION cayu_index_child_lifecycle_consumption()
        RETURNS TRIGGER AS $$
        BEGIN
            IF TG_OP = 'DELETE' THEN
                IF OLD.record ->> 'record_type' =
                   'cayu.child-session-notification-consumption' THEN
                    PERFORM cayu_refresh_child_session_lifecycle(
                        OLD.record ->> 'child_session_id'
                    );
                END IF;
                RETURN OLD;
            END IF;
            IF TG_OP = 'UPDATE'
               AND OLD.record ->> 'record_type' =
                   'cayu.child-session-notification-consumption'
               AND (
                   NEW.record ->> 'record_type' IS DISTINCT FROM
                       'cayu.child-session-notification-consumption'
                   OR OLD.record ->> 'child_session_id' IS DISTINCT FROM
                       NEW.record ->> 'child_session_id'
               ) THEN
                PERFORM cayu_refresh_child_session_lifecycle(
                    OLD.record ->> 'child_session_id'
                );
            END IF;
            IF NEW.record ->> 'record_type' =
               'cayu.child-session-notification-consumption' THEN
                PERFORM cayu_refresh_child_session_lifecycle(
                    NEW.record ->> 'child_session_id'
                );
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
        """,
        "DROP TRIGGER IF EXISTS cayu_index_child_lifecycle_consumption_insert "
        "ON cayu_session_operations",
        """
        CREATE TRIGGER cayu_index_child_lifecycle_consumption_insert
        AFTER INSERT ON cayu_session_operations
        FOR EACH ROW EXECUTE FUNCTION cayu_index_child_lifecycle_consumption()
        """,
        "DROP TRIGGER IF EXISTS cayu_index_child_lifecycle_consumption_delete "
        "ON cayu_session_operations",
        """
        CREATE TRIGGER cayu_index_child_lifecycle_consumption_delete
        AFTER DELETE ON cayu_session_operations
        FOR EACH ROW EXECUTE FUNCTION cayu_index_child_lifecycle_consumption()
        """,
        "DROP TRIGGER IF EXISTS cayu_index_child_lifecycle_consumption_update "
        "ON cayu_session_operations",
        """
        CREATE TRIGGER cayu_index_child_lifecycle_consumption_update
        AFTER UPDATE OF session_id, idempotency_key, record
        ON cayu_session_operations
        FOR EACH ROW EXECUTE FUNCTION cayu_index_child_lifecycle_consumption()
        """,
        """
        SELECT cayu_refresh_child_session_lifecycle(child.id)
        FROM cayu_sessions AS child
        WHERE child.parent_session_id IS NOT NULL
        ORDER BY child.id COLLATE "C"
        """,
    ),
    58: (
        "ALTER TABLE cayu_completion_verification_claims "
        "ADD COLUMN IF NOT EXISTS verifier_profile_fingerprint TEXT "
        "CHECK (verifier_profile_fingerprint IS NOT NULL AND "
        "verifier_profile_fingerprint ~ '^[0-9a-f]{64}$')",
        "ALTER TABLE cayu_completion_decisions "
        "ADD COLUMN IF NOT EXISTS verifier_profile_fingerprint TEXT "
        "CHECK (verifier_profile_fingerprint IS NOT NULL AND "
        "verifier_profile_fingerprint ~ '^[0-9a-f]{64}$')",
        "ALTER TABLE cayu_completion_verification_claims "
        "ALTER COLUMN verifier_profile_fingerprint SET NOT NULL",
        "ALTER TABLE cayu_completion_decisions "
        "ALTER COLUMN verifier_profile_fingerprint SET NOT NULL",
        """
        CREATE TABLE IF NOT EXISTS cayu_completion_verifier_profiles (
            proposal_id TEXT PRIMARY KEY
                REFERENCES cayu_completion_proposals(proposal_id) ON DELETE RESTRICT,
            task_id TEXT NOT NULL REFERENCES cayu_tasks(id) ON DELETE RESTRICT,
            attempt_id TEXT NOT NULL UNIQUE
                REFERENCES cayu_work_attempts(attempt_id) ON DELETE RESTRICT,
            profile_fingerprint TEXT NOT NULL
                CHECK (profile_fingerprint ~ '^[0-9a-f]{64}$'),
            request_sha256 TEXT NOT NULL CHECK (request_sha256 ~ '^[0-9a-f]{64}$'),
            prepared_at TIMESTAMPTZ NOT NULL,
            profile_json JSONB NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_cayu_completion_verifier_profiles_task "
        "ON cayu_completion_verifier_profiles(task_id, attempt_id)",
    ),
    59: (
        "ALTER TABLE cayu_sessions ADD COLUMN IF NOT EXISTS instance_id TEXT",
        "ALTER TABLE cayu_tasks ADD COLUMN IF NOT EXISTS session_instance_id TEXT",
        """
        WITH generated AS (
            SELECT id,
                   md5(
                       id || ':' || clock_timestamp()::text || ':' || random()::text
                   ) AS digest
            FROM cayu_sessions
            WHERE instance_id IS NULL
        )
        UPDATE cayu_sessions AS session
        SET instance_id = lower(
            substr(generated.digest, 1, 8) || '-' ||
            substr(generated.digest, 9, 4) || '-4' ||
            substr(generated.digest, 14, 3) || '-a' ||
            substr(generated.digest, 18, 3) || '-' ||
            substr(generated.digest, 21, 12)
        )
        FROM generated
        WHERE session.id = generated.id
          AND session.instance_id IS NULL
        """,
        "ALTER TABLE cayu_sessions ALTER COLUMN instance_id SET NOT NULL",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_cayu_sessions_instance_id "
        "ON cayu_sessions(instance_id)",
    ),
    60: (
        "DROP TABLE IF EXISTS cayu_knowledge_relation_publication_receipts",
        "DROP TABLE IF EXISTS cayu_knowledge_relations",
        "DROP TABLE IF EXISTS cayu_knowledge_change_acknowledgements",
        "DROP TABLE IF EXISTS cayu_knowledge_change_consumers",
        "DROP TABLE IF EXISTS cayu_knowledge_change_labels",
        "DROP TABLE IF EXISTS cayu_knowledge_change_audiences",
        "DROP TABLE IF EXISTS cayu_knowledge_changes",
        """
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
            created_at TIMESTAMPTZ NOT NULL,
            metadata JSONB NOT NULL,
            CHECK (subject_entry_id <> object_entry_id),
            CHECK (
                kind <> 'contradicts'
                OR subject_entry_id COLLATE "C" < object_entry_id COLLATE "C"
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
        )
        """,
        "CREATE INDEX idx_cayu_knowledge_relations_subject "
        'ON cayu_knowledge_relations(subject_entry_id, subject_revision, created_at, id COLLATE "C")',
        "CREATE INDEX idx_cayu_knowledge_relations_object "
        'ON cayu_knowledge_relations(object_entry_id, object_revision, created_at, id COLLATE "C")',
        "CREATE INDEX idx_cayu_knowledge_relations_subject_kind "
        "ON cayu_knowledge_relations("
        'subject_entry_id, subject_revision, kind, created_at, id COLLATE "C")',
        "CREATE INDEX idx_cayu_knowledge_relations_object_kind "
        "ON cayu_knowledge_relations("
        'object_entry_id, object_revision, kind, created_at, id COLLATE "C")',
        """
        CREATE TABLE cayu_knowledge_relation_publication_receipts (
            operation_id TEXT PRIMARY KEY,
            relation_ids JSONB NOT NULL,
            request_sha256 TEXT NOT NULL CHECK (request_sha256 ~ '^[0-9a-f]{64}$'),
            committed_at TIMESTAMPTZ NOT NULL,
            access_snapshots JSONB NOT NULL,
            CHECK (jsonb_typeof(relation_ids) = 'array'),
            CHECK (jsonb_typeof(access_snapshots) = 'array')
        )
        """,
        """
        CREATE TABLE cayu_knowledge_changes (
            sequence BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY
                CHECK (sequence > 0),
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
            committed_at TIMESTAMPTZ NOT NULL,
            operation_id TEXT,
            relation_id TEXT,
            CHECK (
                (kind = 'relation_published' AND relation_id IS NOT NULL)
                OR (kind <> 'relation_published' AND relation_id IS NULL)
            )
        )
        """,
        """
        CREATE TABLE cayu_knowledge_change_audiences (
            change_sequence BIGINT NOT NULL,
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
            requires_include_expired BOOLEAN NOT NULL,
            PRIMARY KEY (change_sequence, audience_kind),
            FOREIGN KEY (change_sequence)
                REFERENCES cayu_knowledge_changes(sequence) ON DELETE CASCADE
        )
        """,
        "CREATE INDEX idx_cayu_knowledge_changes_entry_revision "
        "ON cayu_knowledge_changes(entry_id, entry_revision, sequence)",
        "CREATE INDEX idx_cayu_knowledge_changes_operation "
        "ON cayu_knowledge_changes(operation_id, sequence) WHERE operation_id IS NOT NULL",
        "CREATE UNIQUE INDEX idx_cayu_knowledge_changes_relation "
        "ON cayu_knowledge_changes(relation_id) WHERE relation_id IS NOT NULL",
        "CREATE INDEX idx_cayu_knowledge_change_audiences_namespace "
        "ON cayu_knowledge_change_audiences(namespace, change_sequence, audience_kind)",
        "CREATE INDEX idx_cayu_knowledge_change_audiences_status "
        "ON cayu_knowledge_change_audiences(status, change_sequence, audience_kind)",
        "CREATE INDEX idx_cayu_knowledge_change_audiences_source "
        "ON cayu_knowledge_change_audiences("
        "source_type, source_id, change_sequence, audience_kind)",
        """
        CREATE TABLE cayu_knowledge_change_labels (
            change_sequence BIGINT NOT NULL,
            audience_kind TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            PRIMARY KEY (change_sequence, audience_kind, key),
            FOREIGN KEY (change_sequence, audience_kind)
                REFERENCES cayu_knowledge_change_audiences(
                    change_sequence, audience_kind
                ) ON DELETE CASCADE
        )
        """,
        "CREATE INDEX idx_cayu_knowledge_change_labels_lookup "
        "ON cayu_knowledge_change_labels("
        "key, value, change_sequence, audience_kind)",
        """
        CREATE TABLE cayu_knowledge_change_consumers (
            consumer_id TEXT PRIMARY KEY,
            access_scope_sha256 TEXT NOT NULL,
            cursor_sequence BIGINT NOT NULL DEFAULT 0 CHECK (cursor_sequence >= 0),
            pending_change_sequence BIGINT,
            pending_claim_id TEXT,
            pending_worker_id TEXT,
            pending_attempt INTEGER NOT NULL DEFAULT 0 CHECK (pending_attempt >= 0),
            claimed_at TIMESTAMPTZ,
            lease_expires_at TIMESTAMPTZ,
            last_acknowledged_claim_id TEXT,
            updated_at TIMESTAMPTZ NOT NULL,
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
        )
        """,
        "CREATE INDEX idx_cayu_knowledge_change_consumers_lease "
        "ON cayu_knowledge_change_consumers(lease_expires_at) "
        "WHERE pending_change_sequence IS NOT NULL",
        """
        CREATE TABLE cayu_knowledge_change_acknowledgements (
            consumer_id TEXT NOT NULL,
            claim_id TEXT NOT NULL,
            claim_sha256 TEXT NOT NULL CHECK (claim_sha256 ~ '^[0-9a-f]{64}$'),
            change_sequence BIGINT NOT NULL,
            acknowledged_at TIMESTAMPTZ NOT NULL,
            PRIMARY KEY (consumer_id, claim_id),
            FOREIGN KEY (consumer_id)
                REFERENCES cayu_knowledge_change_consumers(consumer_id) ON DELETE CASCADE,
            FOREIGN KEY (change_sequence)
                REFERENCES cayu_knowledge_changes(sequence)
        )
        """,
    ),
    61: (
        """
        CREATE TABLE IF NOT EXISTS cayu_work_attempt_admissions (
            admission_id TEXT PRIMARY KEY,
            attempt_id TEXT NOT NULL UNIQUE,
            task_id TEXT NOT NULL REFERENCES cayu_tasks(id) ON DELETE RESTRICT,
            session_id TEXT NOT NULL,
            interaction_id TEXT NOT NULL,
            state TEXT NOT NULL CHECK (state IN (
                'preparing', 'active', 'recovering', 'released'
            )),
            prepare_request_sha256 TEXT NOT NULL CHECK (prepare_request_sha256 ~ '^[0-9a-f]{64}$'),
            current_claim_id TEXT NOT NULL,
            current_generation BIGINT NOT NULL CHECK (
                current_generation >= 1 AND current_generation <= 64
            ),
            lease_expires_at TIMESTAMPTZ NOT NULL,
            admission_json JSONB NOT NULL
        )
        """,
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_cayu_work_attempt_admission_interaction "
        "ON cayu_work_attempt_admissions(session_id, interaction_id)",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_cayu_work_attempt_admission_session_current "
        "ON cayu_work_attempt_admissions(session_id) WHERE state != 'released'",
        "CREATE INDEX IF NOT EXISTS idx_cayu_work_attempt_admission_task "
        "ON cayu_work_attempt_admissions(task_id, current_generation DESC)",
        """
        CREATE TABLE IF NOT EXISTS cayu_work_attempt_execution_claims (
            claim_id TEXT PRIMARY KEY,
            admission_id TEXT NOT NULL
                REFERENCES cayu_work_attempt_admissions(admission_id) ON DELETE RESTRICT,
            generation BIGINT NOT NULL CHECK (generation >= 1 AND generation <= 64),
            request_sha256 TEXT NOT NULL CHECK (request_sha256 ~ '^[0-9a-f]{64}$'),
            lease_expires_at TIMESTAMPTZ NOT NULL,
            is_current BOOLEAN NOT NULL,
            claim_json JSONB NOT NULL,
            UNIQUE (admission_id, generation)
        )
        """,
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_cayu_work_attempt_claim_current "
        "ON cayu_work_attempt_execution_claims(admission_id) WHERE is_current",
    ),
    62: (
        """
        UPDATE cayu_deferred_interaction_inputs
        SET source_messages = jsonb_build_object(
            'source_messages', source_messages,
            'initial_transcript_messages', NULL
        )
        WHERE jsonb_typeof(source_messages) = 'array'
        """,
        """
        UPDATE cayu_work_attempt_admissions AS admission
        SET admission_json = jsonb_set(
            admission.admission_json,
            '{continuation,prior_admission_id}',
            to_jsonb(predecessor.admission_id),
            true
        )
        FROM cayu_work_attempt_admissions AS predecessor
        WHERE jsonb_typeof(admission.admission_json -> 'continuation') = 'object'
          AND NOT (admission.admission_json -> 'continuation' ? 'prior_admission_id')
          AND predecessor.attempt_id = (
              admission.admission_json #>> '{continuation,prior_attempt_id}'
          )
        """,
    ),
    63: (
        "DROP TABLE IF EXISTS cayu_knowledge_maintenance_decisions",
        """
        CREATE TABLE cayu_knowledge_maintenance_decisions (
            operation_id TEXT PRIMARY KEY,
            proposal_id TEXT NOT NULL UNIQUE,
            proposal_fingerprint TEXT NOT NULL
                CHECK (proposal_fingerprint ~ '^[0-9a-f]{64}$'),
            request_sha256 TEXT NOT NULL
                CHECK (request_sha256 ~ '^[0-9a-f]{64}$'),
            committed_at TIMESTAMPTZ NOT NULL,
            proposal JSONB NOT NULL,
            decision JSONB NOT NULL,
            receipt JSONB NOT NULL,
            access_snapshot JSONB NOT NULL,
            CHECK (jsonb_typeof(proposal) = 'object'),
            CHECK (jsonb_typeof(decision) = 'object'),
            CHECK (jsonb_typeof(receipt) = 'object'),
            CHECK (jsonb_typeof(access_snapshot) = 'object')
        )
        """,
    ),
    64: (
        """
        CREATE TABLE IF NOT EXISTS cayu_eval_authored_suites (
            revision TEXT COLLATE "C" PRIMARY KEY,
            suite_id TEXT COLLATE "C" NOT NULL,
            suite_revision TEXT NOT NULL,
            target_key TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            case_count BIGINT NOT NULL
                CONSTRAINT cayu_eval_authored_suites_case_count_check
                CHECK (case_count BETWEEN 1 AND 1000),
            assertion_count BIGINT NOT NULL
                CONSTRAINT cayu_eval_authored_suites_assertion_count_check
                CHECK (assertion_count >= case_count AND assertion_count <= 64000),
            simple_input_count BIGINT NOT NULL
                CONSTRAINT cayu_eval_authored_suites_simple_input_count_check
                CHECK (simple_input_count BETWEEN 0 AND case_count),
            scenario_count BIGINT NOT NULL
                CONSTRAINT cayu_eval_authored_suites_scenario_count_check
                CHECK (scenario_count BETWEEN 0 AND case_count),
            trials BIGINT NOT NULL
                CONSTRAINT cayu_eval_authored_suites_trials_check
                CHECK (trials BETWEEN 1 AND 100),
            timeout_seconds BIGINT NOT NULL
                CONSTRAINT cayu_eval_authored_suites_timeout_check
                CHECK (timeout_seconds BETWEEN 1 AND 3600),
            document_json TEXT NOT NULL,
            document_bytes BIGINT NOT NULL
                CONSTRAINT cayu_eval_authored_suites_document_bytes_check
                CHECK (document_bytes BETWEEN 1 AND 8388608)
                CONSTRAINT cayu_eval_authored_suites_document_size_check
                CHECK (document_bytes = octet_length(document_json)),
            created_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT cayu_eval_authored_suites_stimulus_partition_check
                CHECK (simple_input_count + scenario_count = case_count),
            CONSTRAINT cayu_eval_authored_suites_expansion_check
                CHECK (assertion_count * trials <= 10000),
            CONSTRAINT cayu_eval_authored_suites_document_json_check
                CHECK (document_json::jsonb IS NOT NULL)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_cayu_eval_authored_suites_catalog "
        "ON cayu_eval_authored_suites(created_at DESC, revision ASC)",
        "CREATE INDEX IF NOT EXISTS idx_cayu_eval_authored_suites_target_catalog "
        "ON cayu_eval_authored_suites(target_key, created_at DESC, revision ASC)",
        "CREATE INDEX IF NOT EXISTS idx_cayu_eval_authored_suites_id_catalog "
        "ON cayu_eval_authored_suites(suite_id, created_at DESC, revision ASC)",
    ),
    68: (
        """
        CREATE TABLE IF NOT EXISTS cayu_eval_judge_calibrations (
            revision TEXT COLLATE "C" PRIMARY KEY,
            run_id TEXT COLLATE "C" NOT NULL UNIQUE,
            definition_revision TEXT NOT NULL,
            target_key TEXT NOT NULL,
            trial_count BIGINT NOT NULL
                CONSTRAINT cayu_eval_judge_calibrations_trial_count_check
                CHECK (trial_count BETWEEN 1 AND 10),
            report_json TEXT NOT NULL,
            document_bytes BIGINT NOT NULL
                CONSTRAINT cayu_eval_judge_calibrations_document_bytes_check
                CHECK (document_bytes BETWEEN 1 AND 2097152)
                CONSTRAINT cayu_eval_judge_calibrations_document_size_check
                CHECK (document_bytes = octet_length(report_json)),
            created_at TIMESTAMPTZ NOT NULL,
            CONSTRAINT cayu_eval_judge_calibrations_report_json_check
                CHECK (jsonb_typeof(report_json::jsonb) = 'object')
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_cayu_eval_judge_calibrations_target "
        "ON cayu_eval_judge_calibrations(target_key, created_at DESC, revision ASC)",
        "CREATE INDEX IF NOT EXISTS idx_cayu_eval_judge_calibrations_definition "
        "ON cayu_eval_judge_calibrations("
        "definition_revision, created_at DESC, revision ASC)",
    ),
    65: (
        "DROP VIEW IF EXISTS cayu_knowledge_current_entries",
        "ALTER TABLE cayu_knowledge_revisions ADD COLUMN IF NOT EXISTS "
        "payload_bytes BIGINT NOT NULL DEFAULT 1 CHECK "
        "(payload_bytes > 0 AND payload_bytes <= 2147483647)",
        "ALTER TABLE cayu_knowledge_revisions ALTER COLUMN payload_bytes DROP DEFAULT",
        """
        CREATE VIEW cayu_knowledge_current_entries AS
        SELECT
            logical.id AS id,
            revision.revision AS revision,
            logical.namespace AS namespace,
            revision.text,
            revision.kind,
            revision.visibility,
            revision.status,
            revision.created_by_type,
            revision.created_by,
            revision.created_at,
            revision.updated_at,
            revision.source_type,
            revision.source_uri,
            revision.source_id,
            revision.source_hash,
            revision.importance,
            revision.importance_source,
            revision.confidence,
            revision.last_used_at,
            revision.expires_at,
            revision.title,
            revision.metadata,
            revision.payload_bytes
        FROM cayu_knowledge_entries AS logical
        JOIN cayu_knowledge_revisions AS revision
          ON revision.entry_id = logical.id
         AND revision.revision = logical.current_revision
        """,
    ),
    66: (
        """
        CREATE TABLE IF NOT EXISTS cayu_local_execution_attempts (
            attempt_id TEXT COLLATE "C" PRIMARY KEY,
            task_id TEXT NOT NULL REFERENCES cayu_tasks(id) ON DELETE RESTRICT,
            retry_series_id TEXT COLLATE "C",
            effect_lineage_id TEXT COLLATE "C" NOT NULL,
            request_sha256 TEXT COLLATE "C" NOT NULL,
            phase TEXT NOT NULL CHECK (
                phase IN ('prepared', 'starting', 'running', 'terminal')
            ),
            quiescence TEXT NOT NULL CHECK (
                quiescence IN (
                    'not_dispatched', 'terminal_not_quiescent', 'quiescent',
                    'unavailable', 'persistent_detached'
                )
            ),
            retry_admissible BOOLEAN NOT NULL,
            recovery_generation BIGINT NOT NULL CHECK (recovery_generation >= 0),
            recovery_owner_id TEXT,
            recovery_owner_expires_at TIMESTAMPTZ,
            record_json JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            UNIQUE (task_id, effect_lineage_id, attempt_id)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_cayu_local_execution_attempts_task_fence "
        "ON cayu_local_execution_attempts(task_id, retry_admissible, created_at, attempt_id)",
        "CREATE INDEX IF NOT EXISTS idx_cayu_local_execution_attempts_lineage "
        "ON cayu_local_execution_attempts("
        "retry_series_id, task_id, effect_lineage_id, created_at DESC, attempt_id DESC)",
        "CREATE INDEX IF NOT EXISTS idx_cayu_local_execution_attempts_recovery "
        "ON cayu_local_execution_attempts("
        "retry_admissible, phase, updated_at, attempt_id)",
        "CREATE INDEX IF NOT EXISTS idx_cayu_local_execution_attempts_discovery "
        "ON cayu_local_execution_attempts(created_at, attempt_id)",
    ),
    67: (
        """
        CREATE TABLE IF NOT EXISTS cayu_knowledge_maintenance_proposals (
            operation_id TEXT PRIMARY KEY,
            proposal_id TEXT NOT NULL UNIQUE,
            replacement_entry_id TEXT NOT NULL UNIQUE,
            replacement_revision INTEGER NOT NULL
                CHECK (replacement_revision > 0 AND replacement_revision <= 2147483647),
            proposal_fingerprint TEXT NOT NULL
                CHECK (proposal_fingerprint ~ '^[0-9a-f]{64}$'),
            accepted_plan_fingerprint TEXT NOT NULL
                CHECK (accepted_plan_fingerprint ~ '^[0-9a-f]{64}$'),
            request_sha256 TEXT NOT NULL
                CHECK (request_sha256 ~ '^[0-9a-f]{64}$'),
            committed_at TIMESTAMPTZ NOT NULL,
            proposal JSONB NOT NULL,
            accepted_plan JSONB NOT NULL,
            receipt JSONB NOT NULL,
            access_snapshot JSONB NOT NULL,
            CHECK (jsonb_typeof(proposal) = 'object'),
            CHECK (jsonb_typeof(accepted_plan) = 'object'),
            CHECK (jsonb_typeof(receipt) = 'object'),
            CHECK (jsonb_typeof(access_snapshot) = 'object')
        )
        """,
    ),
    69: (
        """
        CREATE TABLE IF NOT EXISTS cayu_agent_work_context_revisions (
            task_id TEXT COLLATE "C" NOT NULL,
            revision INTEGER NOT NULL CHECK (
                revision > 0 AND revision <= 2147483647
            ),
            content_sha256 TEXT COLLATE "C" NOT NULL CHECK (
                content_sha256 ~ '^[0-9a-f]{64}$'
            ),
            operation_id TEXT COLLATE "C" NOT NULL UNIQUE,
            record_json JSONB NOT NULL CHECK (jsonb_typeof(record_json) = 'object'),
            published_at TIMESTAMPTZ NOT NULL,
            PRIMARY KEY (task_id, revision)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_agent_work_context_heads (
            task_id TEXT COLLATE "C" PRIMARY KEY,
            current_revision INTEGER NOT NULL CHECK (
                current_revision > 0 AND current_revision <= 2147483647
            ),
            FOREIGN KEY (task_id, current_revision)
                REFERENCES cayu_agent_work_context_revisions(task_id, revision)
                ON DELETE RESTRICT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_agent_work_context_publications (
            operation_id TEXT COLLATE "C" PRIMARY KEY,
            task_id TEXT COLLATE "C" NOT NULL,
            request_sha256 TEXT COLLATE "C" NOT NULL CHECK (
                request_sha256 ~ '^[0-9a-f]{64}$'
            ),
            context_revision INTEGER NOT NULL CHECK (
                context_revision > 0 AND context_revision <= 2147483647
            ),
            changed BOOLEAN NOT NULL,
            receipt_json JSONB NOT NULL CHECK (jsonb_typeof(receipt_json) = 'object'),
            committed_at TIMESTAMPTZ NOT NULL,
            FOREIGN KEY (task_id, context_revision)
                REFERENCES cayu_agent_work_context_revisions(task_id, revision)
                ON DELETE RESTRICT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_agent_recall_checkpoints (
            agent_id TEXT COLLATE "C" NOT NULL,
            task_id TEXT COLLATE "C" NOT NULL,
            knowledge_namespace TEXT COLLATE "C" NOT NULL,
            access_policy_sha256 TEXT COLLATE "C" NOT NULL CHECK (
                access_policy_sha256 ~ '^[0-9a-f]{64}$'
            ),
            checkpoint_stream_id TEXT COLLATE "C" NOT NULL,
            revision INTEGER NOT NULL CHECK (
                revision > 0 AND revision <= 2147483647
            ),
            work_context_revision INTEGER NOT NULL CHECK (
                work_context_revision > 0 AND work_context_revision <= 2147483647
            ),
            work_context_sha256 TEXT COLLATE "C" NOT NULL CHECK (
                work_context_sha256 ~ '^[0-9a-f]{64}$'
            ),
            knowledge_sequence BIGINT NOT NULL CHECK (
                knowledge_sequence >= 0
                AND knowledge_sequence <= 9223372036854775807
            ),
            index_readiness_sequence BIGINT NOT NULL CHECK (
                index_readiness_sequence >= 0
                AND index_readiness_sequence <= 9223372036854775807
            ),
            knowledge_high_water_sequence BIGINT NOT NULL CHECK (
                knowledge_high_water_sequence >= 0
                AND knowledge_high_water_sequence <= 9223372036854775807
            ),
            index_readiness_high_water_sequence BIGINT NOT NULL CHECK (
                index_readiness_high_water_sequence >= 0
                AND index_readiness_high_water_sequence <= 9223372036854775807
            ),
            processing_mode TEXT COLLATE "C" NOT NULL CHECK (
                processing_mode IN ('full_index', 'delta')
            ),
            processing_id TEXT COLLATE "C" NOT NULL,
            operation_id TEXT COLLATE "C" NOT NULL UNIQUE,
            record_json JSONB NOT NULL CHECK (jsonb_typeof(record_json) = 'object'),
            updated_at TIMESTAMPTZ NOT NULL,
            CHECK (knowledge_sequence <= knowledge_high_water_sequence),
            CHECK (index_readiness_sequence <= index_readiness_high_water_sequence),
            PRIMARY KEY (
                agent_id, task_id, knowledge_namespace,
                access_policy_sha256, checkpoint_stream_id, revision
            ),
            FOREIGN KEY (task_id, work_context_revision)
                REFERENCES cayu_agent_work_context_revisions(task_id, revision)
                ON DELETE RESTRICT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_agent_recall_checkpoint_heads (
            agent_id TEXT COLLATE "C" NOT NULL,
            task_id TEXT COLLATE "C" NOT NULL,
            knowledge_namespace TEXT COLLATE "C" NOT NULL,
            access_policy_sha256 TEXT COLLATE "C" NOT NULL,
            checkpoint_stream_id TEXT COLLATE "C" NOT NULL,
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
        )
        """,
    ),
    70: (
        """
        CREATE TABLE IF NOT EXISTS cayu_task_interrupted_handoff_receipts (
            task_id TEXT NOT NULL,
            handoff_id TEXT NOT NULL,
            request_sha256 TEXT NOT NULL,
            request_json JSONB NOT NULL,
            task_json JSONB NOT NULL,
            committed_at TIMESTAMPTZ NOT NULL,
            PRIMARY KEY (task_id, handoff_id)
        )
        """,
    ),
    71: (
        """
        CREATE TABLE IF NOT EXISTS cayu_agent_recall_deliveries (
            delivery_id TEXT COLLATE "C" PRIMARY KEY,
            operation_id TEXT COLLATE "C" NOT NULL UNIQUE,
            agent_id TEXT COLLATE "C" NOT NULL,
            task_id TEXT COLLATE "C" NOT NULL,
            knowledge_namespace TEXT COLLATE "C" NOT NULL,
            access_policy_sha256 TEXT COLLATE "C" NOT NULL CHECK (
                access_policy_sha256 ~ '^[0-9a-f]{64}$'
            ),
            checkpoint_stream_id TEXT COLLATE "C" NOT NULL,
            checkpoint_revision INTEGER NOT NULL CHECK (
                checkpoint_revision > 0 AND checkpoint_revision <= 2147483647
            ),
            processing_result_sha256 TEXT COLLATE "C" NOT NULL CHECK (
                processing_result_sha256 ~ '^[0-9a-f]{64}$'
            ),
            delivery_json JSONB NOT NULL CHECK (jsonb_typeof(delivery_json) = 'object'),
            staged_at TIMESTAMPTZ NOT NULL,
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
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_agent_recall_delivery_claims (
            claim_id TEXT COLLATE "C" PRIMARY KEY,
            delivery_id TEXT COLLATE "C" NOT NULL,
            worker_id TEXT COLLATE "C" NOT NULL,
            request_sha256 TEXT COLLATE "C" NOT NULL CHECK (
                request_sha256 ~ '^[0-9a-f]{64}$'
            ),
            attempt BIGINT NOT NULL CHECK (
                attempt > 0 AND attempt <= 9223372036854775807
            ),
            claimed_at TIMESTAMPTZ NOT NULL,
            UNIQUE (delivery_id, attempt),
            FOREIGN KEY (delivery_id)
                REFERENCES cayu_agent_recall_deliveries(delivery_id)
                ON DELETE RESTRICT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_agent_recall_delivery_releases (
            release_id TEXT COLLATE "C" PRIMARY KEY,
            delivery_id TEXT COLLATE "C" NOT NULL,
            claim_id TEXT COLLATE "C" NOT NULL,
            request_sha256 TEXT COLLATE "C" NOT NULL CHECK (
                request_sha256 ~ '^[0-9a-f]{64}$'
            ),
            release_json JSONB NOT NULL CHECK (jsonb_typeof(release_json) = 'object'),
            released_at TIMESTAMPTZ NOT NULL,
            FOREIGN KEY (delivery_id)
                REFERENCES cayu_agent_recall_deliveries(delivery_id)
                ON DELETE RESTRICT,
            FOREIGN KEY (claim_id)
                REFERENCES cayu_agent_recall_delivery_claims(claim_id)
                ON DELETE RESTRICT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_agent_recall_delivery_states (
            delivery_id TEXT COLLATE "C" PRIMARY KEY,
            agent_id TEXT COLLATE "C" NOT NULL,
            task_id TEXT COLLATE "C" NOT NULL,
            knowledge_namespace TEXT COLLATE "C" NOT NULL,
            access_policy_sha256 TEXT COLLATE "C" NOT NULL,
            checkpoint_stream_id TEXT COLLATE "C" NOT NULL,
            checkpoint_revision INTEGER NOT NULL CHECK (
                checkpoint_revision > 0 AND checkpoint_revision <= 2147483647
            ),
            state TEXT COLLATE "C" NOT NULL CHECK (
                state IN ('pending', 'claimed', 'acknowledged')
            ),
            attempt BIGINT NOT NULL CHECK (
                attempt >= 0 AND attempt <= 9223372036854775807
            ),
            state_revision BIGINT NOT NULL CHECK (
                state_revision >= 0 AND state_revision <= 9223372036854775807
            ),
            lease_expires_at TIMESTAMPTZ,
            release_id TEXT COLLATE "C" UNIQUE,
            acknowledgement_id TEXT COLLATE "C" UNIQUE,
            state_json JSONB NOT NULL CHECK (jsonb_typeof(state_json) = 'object'),
            updated_at TIMESTAMPTZ NOT NULL,
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
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_cayu_agent_recall_delivery_pending
            ON cayu_agent_recall_delivery_states(
                agent_id, task_id, knowledge_namespace,
                access_policy_sha256, checkpoint_stream_id,
                checkpoint_revision, delivery_id
            ) WHERE state != 'acknowledged'
        """,
    ),
    73: (
        """
        ALTER TABLE cayu_agent_recall_deliveries
            ADD COLUMN IF NOT EXISTS processing_schema_version TEXT COLLATE "C"
            NOT NULL CHECK (
                processing_schema_version = 'cayu.agent_recall_processing.v3'
            )
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_agent_recall_subscription_revisions (
            subscription_id TEXT COLLATE "C" NOT NULL,
            revision INTEGER NOT NULL CHECK (
                revision > 0 AND revision <= 2147483647
            ),
            operation_id TEXT COLLATE "C" NOT NULL UNIQUE,
            agent_id TEXT COLLATE "C" NOT NULL,
            task_id TEXT COLLATE "C" NOT NULL,
            knowledge_namespace TEXT COLLATE "C" NOT NULL,
            access_policy_sha256 TEXT COLLATE "C" NOT NULL CHECK (
                access_policy_sha256 ~ '^[0-9a-f]{64}$'
            ),
            work_context_revision INTEGER NOT NULL CHECK (
                work_context_revision > 0 AND work_context_revision <= 2147483647
            ),
            work_context_sha256 TEXT COLLATE "C" NOT NULL CHECK (
                work_context_sha256 ~ '^[0-9a-f]{64}$'
            ),
            status TEXT COLLATE "C" NOT NULL CHECK (
                status IN ('active', 'paused', 'cancelled')
            ),
            priority INTEGER NOT NULL CHECK (
                priority >= 0 AND priority <= 1000
            ),
            subscription_json JSONB NOT NULL CHECK (
                jsonb_typeof(subscription_json) = 'object'
            ),
            expires_at TIMESTAMPTZ NOT NULL,
            published_at TIMESTAMPTZ NOT NULL,
            PRIMARY KEY (subscription_id, revision),
            FOREIGN KEY (task_id, work_context_revision)
                REFERENCES cayu_agent_work_context_revisions(task_id, revision)
                ON DELETE RESTRICT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_agent_recall_subscription_heads (
            subscription_id TEXT COLLATE "C" PRIMARY KEY,
            current_revision INTEGER NOT NULL CHECK (
                current_revision > 0 AND current_revision <= 2147483647
            ),
            FOREIGN KEY (subscription_id, current_revision)
                REFERENCES cayu_agent_recall_subscription_revisions(
                    subscription_id, revision
                ) ON DELETE RESTRICT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_agent_recall_subscription_publications (
            operation_id TEXT COLLATE "C" PRIMARY KEY,
            subscription_id TEXT COLLATE "C" NOT NULL,
            subscription_revision INTEGER NOT NULL CHECK (
                subscription_revision > 0 AND subscription_revision <= 2147483647
            ),
            request_sha256 TEXT COLLATE "C" NOT NULL CHECK (
                request_sha256 ~ '^[0-9a-f]{64}$'
            ),
            receipt_json JSONB NOT NULL CHECK (jsonb_typeof(receipt_json) = 'object'),
            committed_at TIMESTAMPTZ NOT NULL,
            FOREIGN KEY (subscription_id, subscription_revision)
                REFERENCES cayu_agent_recall_subscription_revisions(
                    subscription_id, revision
                ) ON DELETE RESTRICT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_agent_recall_subscription_claims (
            claim_id TEXT COLLATE "C" PRIMARY KEY,
            subscription_id TEXT COLLATE "C" NOT NULL,
            subscription_revision INTEGER NOT NULL CHECK (
                subscription_revision > 0 AND subscription_revision <= 2147483647
            ),
            runner_id TEXT COLLATE "C" NOT NULL,
            request_sha256 TEXT COLLATE "C" NOT NULL CHECK (
                request_sha256 ~ '^[0-9a-f]{64}$'
            ),
            attempt BIGINT NOT NULL CHECK (
                attempt > 0 AND attempt <= 9223372036854775807
            ),
            claimed_at TIMESTAMPTZ NOT NULL,
            UNIQUE (subscription_id, attempt),
            FOREIGN KEY (subscription_id, subscription_revision)
                REFERENCES cayu_agent_recall_subscription_revisions(
                    subscription_id, revision
                ) ON DELETE RESTRICT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_agent_recall_subscription_releases (
            release_id TEXT COLLATE "C" PRIMARY KEY,
            subscription_id TEXT COLLATE "C" NOT NULL,
            claim_id TEXT COLLATE "C" NOT NULL,
            request_sha256 TEXT COLLATE "C" NOT NULL CHECK (
                request_sha256 ~ '^[0-9a-f]{64}$'
            ),
            release_json JSONB NOT NULL CHECK (jsonb_typeof(release_json) = 'object'),
            released_at TIMESTAMPTZ NOT NULL,
            FOREIGN KEY (subscription_id)
                REFERENCES cayu_agent_recall_subscription_heads(subscription_id)
                ON DELETE RESTRICT,
            FOREIGN KEY (claim_id)
                REFERENCES cayu_agent_recall_subscription_claims(claim_id)
                ON DELETE RESTRICT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_agent_recall_subscription_states (
            subscription_id TEXT COLLATE "C" PRIMARY KEY,
            current_revision INTEGER NOT NULL CHECK (
                current_revision > 0 AND current_revision <= 2147483647
            ),
            agent_id TEXT COLLATE "C" NOT NULL,
            task_id TEXT COLLATE "C" NOT NULL,
            knowledge_namespace TEXT COLLATE "C" NOT NULL,
            access_policy_sha256 TEXT COLLATE "C" NOT NULL,
            run_state TEXT COLLATE "C" NOT NULL CHECK (
                run_state IN ('due', 'claimed')
            ),
            attempt BIGINT NOT NULL CHECK (
                attempt >= 0 AND attempt <= 9223372036854775807
            ),
            state_revision BIGINT NOT NULL CHECK (
                state_revision >= 0 AND state_revision <= 9223372036854775807
            ),
            lease_expires_at TIMESTAMPTZ,
            release_id TEXT COLLATE "C" UNIQUE,
            next_evaluation_at TIMESTAMPTZ NOT NULL,
            last_evaluation_id TEXT COLLATE "C",
            state_json JSONB NOT NULL CHECK (jsonb_typeof(state_json) = 'object'),
            updated_at TIMESTAMPTZ NOT NULL,
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
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_agent_recall_subscription_evaluations (
            evaluation_id TEXT COLLATE "C" PRIMARY KEY,
            subscription_id TEXT COLLATE "C" NOT NULL,
            subscription_revision INTEGER NOT NULL CHECK (
                subscription_revision > 0 AND subscription_revision <= 2147483647
            ),
            agent_id TEXT COLLATE "C" NOT NULL,
            task_id TEXT COLLATE "C" NOT NULL,
            knowledge_namespace TEXT COLLATE "C" NOT NULL,
            access_policy_sha256 TEXT COLLATE "C" NOT NULL,
            claim_id TEXT COLLATE "C" NOT NULL UNIQUE,
            processing_operation_id TEXT COLLATE "C" NOT NULL UNIQUE,
            request_sha256 TEXT COLLATE "C" NOT NULL CHECK (
                request_sha256 ~ '^[0-9a-f]{64}$'
            ),
            outcome TEXT COLLATE "C" NOT NULL CHECK (
                outcome IN ('no_work', 'silent', 'wake')
            ),
            delivery_id TEXT COLLATE "C" UNIQUE,
            evaluation_json JSONB NOT NULL CHECK (
                jsonb_typeof(evaluation_json) = 'object'
            ),
            committed_at TIMESTAMPTZ NOT NULL,
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
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_agent_recall_subscription_wake_claims (
            claim_id TEXT COLLATE "C" PRIMARY KEY,
            wake_id TEXT COLLATE "C" NOT NULL,
            delivery_id TEXT COLLATE "C" NOT NULL,
            runner_id TEXT COLLATE "C" NOT NULL,
            request_sha256 TEXT COLLATE "C" NOT NULL CHECK (
                request_sha256 ~ '^[0-9a-f]{64}$'
            ),
            attempt BIGINT NOT NULL CHECK (
                attempt > 0 AND attempt <= 9223372036854775807
            ),
            claimed_at TIMESTAMPTZ NOT NULL,
            UNIQUE (wake_id, attempt),
            FOREIGN KEY (wake_id)
                REFERENCES cayu_agent_recall_subscription_evaluations(evaluation_id)
                ON DELETE RESTRICT,
            FOREIGN KEY (delivery_id)
                REFERENCES cayu_agent_recall_deliveries(delivery_id)
                ON DELETE RESTRICT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_agent_recall_subscription_wake_releases (
            release_id TEXT COLLATE "C" PRIMARY KEY,
            wake_id TEXT COLLATE "C" NOT NULL,
            claim_id TEXT COLLATE "C" NOT NULL,
            request_sha256 TEXT COLLATE "C" NOT NULL CHECK (
                request_sha256 ~ '^[0-9a-f]{64}$'
            ),
            release_json JSONB NOT NULL CHECK (jsonb_typeof(release_json) = 'object'),
            released_at TIMESTAMPTZ NOT NULL,
            FOREIGN KEY (wake_id)
                REFERENCES cayu_agent_recall_subscription_evaluations(evaluation_id)
                ON DELETE RESTRICT,
            FOREIGN KEY (claim_id)
                REFERENCES cayu_agent_recall_subscription_wake_claims(claim_id)
                ON DELETE RESTRICT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cayu_agent_recall_subscription_wake_states (
            wake_id TEXT COLLATE "C" PRIMARY KEY,
            delivery_id TEXT COLLATE "C" NOT NULL UNIQUE,
            agent_id TEXT COLLATE "C" NOT NULL,
            task_id TEXT COLLATE "C" NOT NULL,
            knowledge_namespace TEXT COLLATE "C" NOT NULL,
            access_policy_sha256 TEXT COLLATE "C" NOT NULL,
            state TEXT COLLATE "C" NOT NULL CHECK (
                state IN ('pending', 'claimed', 'acknowledged')
            ),
            attempt BIGINT NOT NULL CHECK (
                attempt >= 0 AND attempt <= 9223372036854775807
            ),
            state_revision BIGINT NOT NULL CHECK (
                state_revision >= 0 AND state_revision <= 9223372036854775807
            ),
            claim_id TEXT COLLATE "C",
            lease_expires_at TIMESTAMPTZ,
            release_id TEXT COLLATE "C" UNIQUE,
            acknowledgement_id TEXT COLLATE "C" UNIQUE,
            state_json JSONB NOT NULL CHECK (jsonb_typeof(state_json) = 'object'),
            committed_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
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
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_cayu_agent_recall_subscription_due
            ON cayu_agent_recall_subscription_states(
                agent_id, task_id, knowledge_namespace, access_policy_sha256,
                next_evaluation_at, subscription_id
            )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_cayu_agent_recall_subscription_evaluations
            ON cayu_agent_recall_subscription_evaluations(
                subscription_id, evaluation_id
            )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_cayu_agent_recall_subscription_wakes
            ON cayu_agent_recall_subscription_wake_states(
                agent_id, task_id, knowledge_namespace, access_policy_sha256,
                committed_at, wake_id
            ) WHERE state != 'acknowledged'
        """,
    ),
    72: (
        "ALTER TABLE cayu_eval_runs DROP CONSTRAINT IF EXISTS cayu_eval_runs_max_concurrency_check",
        "ALTER TABLE cayu_eval_runs ADD CONSTRAINT "
        "cayu_eval_runs_max_concurrency_check "
        "CHECK (max_concurrency >= 1 AND max_concurrency <= 2147483647)",
    ),
    76: (
        "ALTER TABLE cayu_tasks ADD COLUMN IF NOT EXISTS interrupted_handoff_id TEXT",
        """
        CREATE TABLE IF NOT EXISTS cayu_task_interrupted_continuation_claims (
            handoff_id_sha256 TEXT COLLATE "C" PRIMARY KEY
                CHECK (handoff_id_sha256 ~ '^[0-9a-f]{64}$'),
            task_id TEXT COLLATE "C" NOT NULL,
            worker_id TEXT COLLATE "C" NOT NULL,
            claimed_at TIMESTAMPTZ NOT NULL
        )
        """,
    ),
    86: (
        "CREATE INDEX IF NOT EXISTS idx_cayu_side_effect_health\n ON cayu_persisted_event_side_effects(status, next_attempt_at, lease_expires_at, updated_at, attempts)",
        'CREATE INDEX IF NOT EXISTS idx_cayu_side_effect_outstanding\n ON cayu_persisted_event_side_effects(session_id COLLATE "C", event_id COLLATE "C") WHERE status <> \'delivered\'',
    ),
}

_REVISION_17_PENDING_TOOL_CALL_COUNT_SQL = """
    GREATEST(
        CASE
            WHEN jsonb_typeof(
                target.state #> '{pending_tool_approval,tool_calls}'
            ) = 'array'
            THEN jsonb_array_length(
                target.state #> '{pending_tool_approval,tool_calls}'
            )
            ELSE 0
        END,
        CASE
            WHEN jsonb_typeof(
                target.state #> '{pending_user_input,tool_calls}'
            ) = 'array'
            THEN jsonb_array_length(
                target.state #> '{pending_user_input,tool_calls}'
            )
            ELSE 0
        END,
        CASE
            WHEN jsonb_typeof(
                target.state #> '{pending_tool_round,tool_calls}'
            ) = 'array'
            THEN jsonb_array_length(
                target.state #> '{pending_tool_round,tool_calls}'
            )
            ELSE 0
        END
    )
"""

_REVISION_17_CHECKPOINT_BACKFILL_SQL = f"""
    WITH batch AS MATERIALIZED (
        SELECT session_id
        FROM cayu_checkpoints
        WHERE NOT pending_action_metrics_ready
          AND (%s::text IS NULL OR session_id > %s)
        ORDER BY session_id
        LIMIT 100
        FOR UPDATE SKIP LOCKED
    )
    UPDATE cayu_checkpoints AS target
    SET pending_action_flags =
            CASE WHEN target.state -> 'pending_tool_approval' IS NOT NULL
                  AND target.state -> 'pending_tool_approval' <> 'null'::jsonb
                THEN 1 ELSE 0 END
            + CASE WHEN target.state -> 'pending_user_input' IS NOT NULL
                  AND target.state -> 'pending_user_input' <> 'null'::jsonb
                THEN 2 ELSE 0 END
            + CASE WHEN target.state -> 'pending_tool_round' IS NOT NULL
                  AND target.state -> 'pending_tool_round' <> 'null'::jsonb
                THEN 4 ELSE 0 END,
        pending_action_source_bytes = CASE
            WHEN ({_REVISION_17_PENDING_TOOL_CALL_COUNT_SQL})
                > {MAX_PENDING_ACTION_TOOL_CALLS}
            THEN 0
            WHEN (target.state -> 'pending_tool_approval' IS NOT NULL
                  AND target.state -> 'pending_tool_approval' <> 'null'::jsonb)
              OR (target.state -> 'pending_user_input' IS NOT NULL
                  AND target.state -> 'pending_user_input' <> 'null'::jsonb)
              OR (target.state -> 'pending_tool_round' IS NOT NULL
                  AND target.state -> 'pending_tool_round' <> 'null'::jsonb)
            THEN octet_length(jsonb_strip_nulls(jsonb_build_object(
                'pending_tool_approval', target.state -> 'pending_tool_approval',
                'pending_user_input', target.state -> 'pending_user_input',
                'pending_tool_round', target.state -> 'pending_tool_round'
            ))::text)
            ELSE NULL
        END,
        pending_action_tool_call_count = ({_REVISION_17_PENDING_TOOL_CALL_COUNT_SQL}),
        pending_action_metrics_ready = TRUE
    FROM batch
    WHERE target.session_id = batch.session_id
    RETURNING target.session_id
"""

_REVISION_17_EVENT_BACKFILL_SMALL_EVENT_BYTES = 1024 * 1024
_REVISION_17_APPROVAL_PROJECTION_KEYS_SQL = ", ".join(
    f"'{key}'" for key in _PENDING_TOOL_APPROVAL_EVENT_PROJECTION_KEYS
)
_REVISION_17_APPROVAL_PROJECTION_SQL = f"""
    (
        SELECT COALESCE(
            jsonb_object_agg(approval_field.key, approval_field.value),
            '{{}}'::jsonb
        )
        FROM jsonb_each(payload -> 'approval') AS approval_field
        WHERE approval_field.key IN ({_REVISION_17_APPROVAL_PROJECTION_KEYS_SQL})
    )
"""


def _revision_17_event_backfill_sql(*, source_predicate: str, batch_limit: int) -> str:
    return f"""
    WITH batch AS MATERIALIZED (
        SELECT sequence, event_type, payload, event
        FROM cayu_events
        WHERE pending_action_projection_bytes IS NULL
          AND sequence > %s
          AND ({source_predicate})
          AND event_type IN (
              'tool.call.approval_requested',
              'session.awaiting_user_input',
              'session.interrupted',
              'session.delegated_action.updated',
              'session.resumed',
              'session.completed',
              'session.failed',
              'tool.call.started',
              'tool.call.completed',
              'tool.call.failed',
              'tool.call.blocked',
              'tool.call.approval_denied'
        )
        ORDER BY sequence
        LIMIT {batch_limit}
        FOR UPDATE SKIP LOCKED
    ),
    projected AS MATERIALIZED (
        SELECT
            sequence,
            CASE
                WHEN event_type IN (
                    'tool.call.started',
                    'tool.call.completed',
                    'tool.call.failed',
                    'tool.call.blocked',
                    'tool.call.approval_denied'
                )
                  AND jsonb_typeof(payload -> 'tool_call_id') = 'string'
                  AND payload ->> 'tool_call_id' !~ '^[[:space:]]*$'
                THEN payload ->> 'tool_call_id'
                WHEN jsonb_typeof(payload -> 'approval_id') = 'string'
                  AND payload ->> 'approval_id' !~ '^[[:space:]]*$'
                THEN payload ->> 'approval_id'
                WHEN jsonb_typeof(payload #> '{{approval,approval_id}}') = 'string'
                  AND payload #>> '{{approval,approval_id}}' !~ '^[[:space:]]*$'
                THEN payload #>> '{{approval,approval_id}}'
                WHEN jsonb_typeof(payload -> 'input_id') = 'string'
                  AND payload ->> 'input_id' !~ '^[[:space:]]*$'
                THEN payload ->> 'input_id'
                WHEN jsonb_typeof(payload #> '{{user_input,input_id}}') = 'string'
                  AND payload #>> '{{user_input,input_id}}' !~ '^[[:space:]]*$'
                THEN payload #>> '{{user_input,input_id}}'
                WHEN jsonb_typeof(payload -> 'tool_call_id') = 'string'
                  AND payload ->> 'tool_call_id' !~ '^[[:space:]]*$'
                THEN payload ->> 'tool_call_id'
                WHEN jsonb_typeof(payload -> 'tool_round_id') = 'string'
                  AND payload ->> 'tool_round_id' !~ '^[[:space:]]*$'
                THEN payload ->> 'tool_round_id'
                ELSE NULL
            END AS lookup_id,
            jsonb_set(
                event,
                '{{payload}}',
                CASE
                    WHEN event_type = 'tool.call.approval_requested' THEN
                        jsonb_strip_nulls(jsonb_build_object(
                            'approval_id', payload -> 'approval_id',
                            'tool_call_id', payload -> 'tool_call_id',
                            'model_step_id', payload -> 'model_step_id',
                            'model_attempt_id', payload -> 'model_attempt_id',
                            'tool_round_id', payload -> 'tool_round_id',
                            'approval', CASE
                                WHEN jsonb_typeof(payload -> 'approval') = 'object'
                                THEN {_REVISION_17_APPROVAL_PROJECTION_SQL}
                                ELSE NULL
                            END
                        ))
                    WHEN event_type = 'session.awaiting_user_input' THEN
                        jsonb_strip_nulls(jsonb_build_object(
                            'input_id', payload -> 'input_id',
                            'tool_call_id', payload -> 'tool_call_id',
                            'question', payload -> 'question',
                            'options', payload -> 'options',
                            'model_step_id', payload -> 'model_step_id',
                            'model_attempt_id', payload -> 'model_attempt_id',
                            'tool_round_id', payload -> 'tool_round_id'
                        ))
                    WHEN event_type = 'session.interrupted' THEN
                        jsonb_strip_nulls(jsonb_build_object(
                            'interruption_type', payload -> 'interruption_type',
                            'child_session_id', payload -> 'child_session_id',
                            'action_kind', payload -> 'action_kind',
                            'action_id', payload -> 'action_id',
                            'status', payload -> 'status',
                            'manual_recovery_required', payload -> 'manual_recovery_required',
                            'approval_id', payload -> 'approval_id',
                            'tool_call_id', payload -> 'tool_call_id',
                            'model_step_id', payload -> 'model_step_id',
                            'model_attempt_id', payload -> 'model_attempt_id',
                            'tool_round_id', payload -> 'tool_round_id',
                            'error', payload -> 'error',
                            'message', payload -> 'message',
                            'tool_name', payload -> 'tool_name',
                            'tool_evidence_conflict', payload -> 'tool_evidence_conflict',
                            'approval', CASE
                                WHEN jsonb_typeof(payload -> 'approval') = 'object'
                                THEN {_REVISION_17_APPROVAL_PROJECTION_SQL}
                                ELSE NULL
                            END,
                            'user_input', CASE
                                WHEN jsonb_typeof(payload -> 'user_input') = 'object'
                                THEN jsonb_strip_nulls(jsonb_build_object(
                                    'input_id', payload #> '{{user_input,input_id}}',
                                    'tool_call_id', payload #> '{{user_input,tool_call_id}}',
                                    'question', payload #> '{{user_input,question}}',
                                    'options', payload #> '{{user_input,options}}'
                                ))
                                ELSE NULL
                            END
                        ))
                    WHEN event_type = 'session.delegated_action.updated' THEN
                        jsonb_strip_nulls(jsonb_build_object(
                            'interruption_type', payload -> 'interruption_type',
                            'child_session_id', payload -> 'child_session_id',
                            'action_kind', payload -> 'action_kind',
                            'action_id', payload -> 'action_id',
                            'status', payload -> 'status',
                            'tool_call_id', payload -> 'tool_call_id',
                            'model_step_id', payload -> 'model_step_id',
                            'model_attempt_id', payload -> 'model_attempt_id',
                            'tool_round_id', payload -> 'tool_round_id'
                        ))
                    WHEN event_type IN (
                        'tool.call.started',
                        'tool.call.completed',
                        'tool.call.failed',
                        'tool.call.blocked',
                        'tool.call.approval_denied'
                    ) THEN jsonb_strip_nulls(jsonb_build_object(
                        'tool_call_id', payload -> 'tool_call_id',
                        'model_step_id', payload -> 'model_step_id',
                        'model_attempt_id', payload -> 'model_attempt_id',
                        'tool_round_id', payload -> 'tool_round_id',
                        'manual_recovery', payload -> 'manual_recovery',
                        '__cayu_terminal_result_valid__',
                        CASE WHEN event_type = 'tool.call.started' THEN NULL ELSE COALESCE(
                        jsonb_typeof(payload -> 'result') = 'object'
                        AND (payload -> 'result')
                            - ARRAY['content', 'structured', 'artifacts', 'is_error']
                            = '{{}}'::jsonb
                        AND (
                            NOT ((payload -> 'result') ? 'content')
                            OR jsonb_typeof(payload #> '{{result,content}}') = 'string'
                        )
                        AND (
                            NOT ((payload -> 'result') ? 'structured')
                            OR jsonb_typeof(payload #> '{{result,structured}}')
                                IN ('object', 'null')
                        )
                        AND (
                            NOT ((payload -> 'result') ? 'artifacts')
                            OR (
                                jsonb_typeof(payload #> '{{result,artifacts}}') = 'array'
                                AND NOT EXISTS (
                                    SELECT 1
                                    FROM jsonb_array_elements(
                                        CASE
                                            WHEN jsonb_typeof(
                                                payload #> '{{result,artifacts}}'
                                            ) = 'array'
                                            THEN payload #> '{{result,artifacts}}'
                                            ELSE '[]'::jsonb
                                        END
                                    ) AS artifact
                                    WHERE jsonb_typeof(artifact) <> 'object'
                                )
                            )
                        )
                        AND (
                            NOT ((payload -> 'result') ? 'is_error')
                            OR jsonb_typeof(payload #> '{{result,is_error}}') = 'boolean'
                        ), FALSE) END
                    ))
                    WHEN event_type = 'session.resumed' THEN
                        jsonb_strip_nulls(jsonb_build_object(
                            'model_step_id', payload -> 'model_step_id',
                            'model_attempt_id', payload -> 'model_attempt_id',
                            'tool_round_id', payload -> 'tool_round_id'
                        ))
                    WHEN event_type = 'session.failed' THEN
                        jsonb_strip_nulls(jsonb_build_object(
                            'tool_evidence_conflict', payload -> 'tool_evidence_conflict'
                        ))
                    ELSE '{{}}'::jsonb
                END,
                true
            ) AS projection
        FROM batch
    ),
    measured AS MATERIALIZED (
        SELECT sequence, lookup_id, projection, octet_length(projection::text) AS bytes
        FROM projected
    )
    UPDATE cayu_events AS target
    SET pending_action_lookup_key = CASE
            WHEN measured.lookup_id IS NULL THEN NULL
            ELSE encode(sha256(convert_to(measured.lookup_id, 'UTF8')), 'hex')
        END,
        pending_action_projection = CASE
            WHEN measured.bytes <= {MAX_PENDING_ACTION_RESULT_BYTES}
            THEN measured.projection
            ELSE jsonb_build_object(
                'type', measured.projection -> 'type',
                'session_id', 'cayu_oversized_pending_action_projection',
                'interaction_id', measured.projection -> 'interaction_id',
                'id', 'cayu_oversized_pending_action_projection',
                'timestamp', measured.projection -> 'timestamp',
                'agent_name', NULL,
                'environment_name', NULL,
                'workflow_name', NULL,
                'tool_name', NULL,
                'payload', jsonb_strip_nulls(jsonb_build_object(
                    'model_step_id', CASE
                        WHEN jsonb_typeof(
                            measured.projection #> '{{payload,model_step_id}}'
                        ) = 'string'
                          AND char_length(
                              measured.projection #>> '{{payload,model_step_id}}'
                          ) <= {EXECUTION_UNIT_ID_MAX_CHARS}
                        THEN measured.projection #> '{{payload,model_step_id}}'
                        ELSE NULL
                    END,
                    'model_attempt_id', CASE
                        WHEN jsonb_typeof(
                            measured.projection #> '{{payload,model_attempt_id}}'
                        ) = 'string'
                          AND char_length(
                              measured.projection #>> '{{payload,model_attempt_id}}'
                          ) <= {EXECUTION_UNIT_ID_MAX_CHARS}
                        THEN measured.projection #> '{{payload,model_attempt_id}}'
                        ELSE NULL
                    END,
                    'tool_round_id', CASE
                        WHEN jsonb_typeof(
                            measured.projection #> '{{payload,tool_round_id}}'
                        ) = 'string'
                          AND char_length(
                              measured.projection #>> '{{payload,tool_round_id}}'
                          ) <= {EXECUTION_UNIT_ID_MAX_CHARS}
                        THEN measured.projection #> '{{payload,tool_round_id}}'
                        ELSE NULL
                    END,
                    '__cayu_pending_action_projection_bytes__',
                    {MAX_PENDING_ACTION_RESULT_BYTES + 1}
                ))
            )
        END,
        pending_action_projection_bytes = CASE
            WHEN measured.bytes <= {MAX_PENDING_ACTION_RESULT_BYTES}
            THEN measured.bytes
            ELSE {MAX_PENDING_ACTION_RESULT_BYTES + 1}
        END
    FROM measured
    WHERE target.sequence = measured.sequence
    RETURNING target.sequence
"""


_REVISION_17_EVENT_BACKFILL_SMALL_SQL = _revision_17_event_backfill_sql(
    source_predicate=(
        f"octet_length(event::text) <= {_REVISION_17_EVENT_BACKFILL_SMALL_EVENT_BYTES}"
    ),
    batch_limit=25,
)
_REVISION_17_EVENT_BACKFILL_LARGE_SQL = _revision_17_event_backfill_sql(
    source_predicate=(
        f"octet_length(event::text) > {_REVISION_17_EVENT_BACKFILL_SMALL_EVENT_BYTES}"
    ),
    batch_limit=1,
)


def _revision_17_event_backfill_remaining_sql(source_predicate: str) -> str:
    return f"""
        SELECT EXISTS(
            SELECT 1
            FROM cayu_events
            WHERE pending_action_projection_bytes IS NULL
              AND ({source_predicate})
              AND event_type IN (
                  'tool.call.approval_requested',
                  'session.awaiting_user_input',
                  'session.interrupted',
                  'session.delegated_action.updated',
                  'session.resumed',
                  'session.completed',
                  'session.failed',
                  'tool.call.started',
                  'tool.call.completed',
                  'tool.call.failed',
                  'tool.call.blocked',
                  'tool.call.approval_denied'
              )
        )
    """


_REVISION_17_EVENT_BACKFILL_SMALL_REMAINING_SQL = _revision_17_event_backfill_remaining_sql(
    f"octet_length(event::text) <= {_REVISION_17_EVENT_BACKFILL_SMALL_EVENT_BYTES}"
)
_REVISION_17_EVENT_BACKFILL_LARGE_REMAINING_SQL = _revision_17_event_backfill_remaining_sql(
    f"octet_length(event::text) > {_REVISION_17_EVENT_BACKFILL_SMALL_EVENT_BYTES}"
)


@dataclass(frozen=True)
class _ConcurrentIndexMigration:
    index_name: str
    table_name: str
    key_definitions: tuple[str, ...]
    predicate_definition: str | None
    create_statement: str
    drop_statement: str
    access_method: str = "btree"
    required_key_collations: tuple[str | None, ...] = ()
    unique: bool = False
    replace_existing: bool = False
    replacement_predicates: tuple[str, ...] = ()

    def transactional_create_statement(self) -> str:
        """Return the equivalent index DDL for an empty, locked schema."""

        parts = self.create_statement.split("CONCURRENTLY")
        if len(parts) != 2:
            raise RuntimeError(
                f"Concurrent index {self.index_name} must have exactly one CONCURRENTLY clause."
            )
        return "".join(parts)


_CONCURRENT_INDEX_MIGRATIONS: dict[int, tuple[_ConcurrentIndexMigration, ...]] = {
    109: (
        _ConcurrentIndexMigration(
            index_name="idx_cayu_budget_reservations_session_identity",
            table_name="cayu_budget_reservations",
            key_definitions=("session_id", "reservation_id"),
            predicate_definition=None,
            create_statement=(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
                "idx_cayu_budget_reservations_session_identity "
                "ON cayu_budget_reservations(session_id, reservation_id)"
            ),
            drop_statement=(
                "DROP INDEX CONCURRENTLY IF EXISTS idx_cayu_budget_reservations_session_identity"
            ),
        ),
    ),
    16: (
        _ConcurrentIndexMigration(
            index_name="idx_cayu_events_session_sequence",
            table_name="cayu_events",
            key_definitions=("session_id", "sequence"),
            predicate_definition=None,
            create_statement=(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_cayu_events_session_sequence "
                "ON cayu_events(session_id, sequence)"
            ),
            drop_statement=("DROP INDEX CONCURRENTLY IF EXISTS idx_cayu_events_session_sequence"),
        ),
    ),
    17: (
        _ConcurrentIndexMigration(
            index_name="idx_cayu_checkpoints_pending_control_action",
            table_name="cayu_checkpoints",
            key_definitions=("session_id",),
            predicate_definition=("pending_action_flags <> 0"),
            create_statement=(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
                "idx_cayu_checkpoints_pending_control_action "
                "ON cayu_checkpoints(session_id) WHERE pending_action_flags <> 0"
            ),
            drop_statement=(
                "DROP INDEX CONCURRENTLY IF EXISTS idx_cayu_checkpoints_pending_control_action"
            ),
        ),
        _ConcurrentIndexMigration(
            index_name="idx_cayu_events_pending_action_barrier",
            table_name="cayu_events",
            key_definitions=("session_id", "sequence"),
            predicate_definition="""
                event_type = 'session.resumed'
                OR event_type = 'session.completed'
                OR event_type = 'session.failed'
            """,
            create_statement="""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS
                    idx_cayu_events_pending_action_barrier
                ON cayu_events(session_id, sequence)
                WHERE event_type = 'session.resumed'
                   OR event_type = 'session.completed'
                   OR event_type = 'session.failed'
            """,
            drop_statement=(
                "DROP INDEX CONCURRENTLY IF EXISTS idx_cayu_events_pending_action_barrier"
            ),
        ),
        _ConcurrentIndexMigration(
            index_name="idx_cayu_events_pending_action_lookup",
            table_name="cayu_events",
            key_definitions=(
                "session_id",
                "pending_action_lookup_key",
                "event_type",
                "sequence",
            ),
            predicate_definition="""
                event_type = ANY (ARRAY[
                    'tool.call.approval_requested',
                    'session.awaiting_user_input',
                    'session.interrupted',
                    'tool.call.started',
                    'tool.call.completed',
                    'tool.call.failed',
                    'tool.call.blocked',
                    'tool.call.approval_denied'
                ])
                AND pending_action_lookup_key IS NOT NULL
            """,
            create_statement="""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS
                    idx_cayu_events_pending_action_lookup
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
                  AND pending_action_lookup_key IS NOT NULL
            """,
            drop_statement=(
                "DROP INDEX CONCURRENTLY IF EXISTS idx_cayu_events_pending_action_lookup"
            ),
        ),
    ),
    26: (
        _ConcurrentIndexMigration(
            index_name="idx_cayu_events_session_interaction_sequence",
            table_name="cayu_events",
            key_definitions=("session_id", "interaction_id", "sequence"),
            predicate_definition=None,
            create_statement=(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
                "idx_cayu_events_session_interaction_sequence "
                "ON cayu_events(session_id, interaction_id, sequence)"
            ),
            drop_statement=(
                "DROP INDEX CONCURRENTLY IF EXISTS idx_cayu_events_session_interaction_sequence"
            ),
        ),
        _ConcurrentIndexMigration(
            index_name="idx_cayu_transcript_messages_session_interaction_sequence",
            table_name="cayu_transcript_messages",
            key_definitions=("session_id", "interaction_id", "sequence"),
            predicate_definition=None,
            create_statement=(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
                "idx_cayu_transcript_messages_session_interaction_sequence "
                "ON cayu_transcript_messages(session_id, interaction_id, sequence)"
            ),
            drop_statement=(
                "DROP INDEX CONCURRENTLY IF EXISTS "
                "idx_cayu_transcript_messages_session_interaction_sequence"
            ),
        ),
    ),
    23: (
        _ConcurrentIndexMigration(
            index_name="idx_cayu_events_budget_reservation_identity",
            table_name="cayu_events",
            key_definitions=("payload ->> 'reservation_id'",),
            predicate_definition="""
                event_type = 'budget.reserved'
                AND jsonb_typeof(payload -> 'reservation_id') = 'string'
            """,
            create_statement="""
                CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS
                    idx_cayu_events_budget_reservation_identity
                ON cayu_events ((payload ->> 'reservation_id'))
                WHERE event_type = 'budget.reserved'
                  AND jsonb_typeof(payload -> 'reservation_id') = 'string'
            """,
            drop_statement=(
                "DROP INDEX CONCURRENTLY IF EXISTS idx_cayu_events_budget_reservation_identity"
            ),
            unique=True,
        ),
        _ConcurrentIndexMigration(
            index_name="idx_cayu_events_pending_action_round_scope",
            table_name="cayu_events",
            key_definitions=(
                "session_id",
                "pending_action_projection #>> '{payload,tool_round_id}'",
                "sequence",
            ),
            predicate_definition="""
                event_type = ANY (ARRAY[
                    'tool.call.started',
                    'tool.call.completed',
                    'tool.call.failed',
                    'tool.call.blocked',
                    'tool.call.approval_denied'
                ])
                AND jsonb_typeof(
                    pending_action_projection #> '{payload,tool_round_id}'
                ) = 'string'
                AND pending_action_projection #>> '{payload,tool_round_id}'
                    ~ '^tround_[0-9a-f]{32}$'
            """,
            create_statement="""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS
                    idx_cayu_events_pending_action_round_scope
                ON cayu_events(
                    session_id,
                    (pending_action_projection #>> '{payload,tool_round_id}'),
                    sequence
                )
                WHERE event_type IN (
                    'tool.call.started',
                    'tool.call.completed',
                    'tool.call.failed',
                    'tool.call.blocked',
                    'tool.call.approval_denied'
                )
                  AND jsonb_typeof(
                      pending_action_projection #> '{payload,tool_round_id}'
                  ) = 'string'
                  AND pending_action_projection #>> '{payload,tool_round_id}'
                      ~ '^tround_[0-9a-f]{32}$'
            """,
            drop_statement=(
                "DROP INDEX CONCURRENTLY IF EXISTS idx_cayu_events_pending_action_round_scope"
            ),
        ),
        _ConcurrentIndexMigration(
            index_name="idx_cayu_events_pending_action_attempt_scope",
            table_name="cayu_events",
            key_definitions=(
                "session_id",
                "pending_action_projection #>> '{payload,model_step_id}'",
                "pending_action_projection #>> '{payload,model_attempt_id}'",
                "sequence",
            ),
            predicate_definition="""
                event_type = ANY (ARRAY[
                    'tool.call.started',
                    'tool.call.completed',
                    'tool.call.failed',
                    'tool.call.blocked',
                    'tool.call.approval_denied'
                ])
                AND jsonb_typeof(
                    pending_action_projection #> '{payload,model_step_id}'
                ) = 'string'
                AND jsonb_typeof(
                    pending_action_projection #> '{payload,model_attempt_id}'
                ) = 'string'
                AND pending_action_projection #>> '{payload,model_step_id}'
                    ~ '^mstep_[0-9a-f]{32}$'
                AND pending_action_projection #>> '{payload,model_attempt_id}'
                    ~ '^matt_[0-9a-f]{32}$'
            """,
            create_statement="""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS
                    idx_cayu_events_pending_action_attempt_scope
                ON cayu_events(
                    session_id,
                    (pending_action_projection #>> '{payload,model_step_id}'),
                    (pending_action_projection #>> '{payload,model_attempt_id}'),
                    sequence
                )
                WHERE event_type IN (
                    'tool.call.started',
                    'tool.call.completed',
                    'tool.call.failed',
                    'tool.call.blocked',
                    'tool.call.approval_denied'
                )
                  AND jsonb_typeof(
                      pending_action_projection #> '{payload,model_step_id}'
                  ) = 'string'
                  AND jsonb_typeof(
                      pending_action_projection #> '{payload,model_attempt_id}'
                  ) = 'string'
                  AND pending_action_projection #>> '{payload,model_step_id}'
                      ~ '^mstep_[0-9a-f]{32}$'
                  AND pending_action_projection #>> '{payload,model_attempt_id}'
                      ~ '^matt_[0-9a-f]{32}$'
            """,
            drop_statement=(
                "DROP INDEX CONCURRENTLY IF EXISTS idx_cayu_events_pending_action_attempt_scope"
            ),
        ),
    ),
    24: (
        _ConcurrentIndexMigration(
            index_name="idx_cayu_sessions_parent_created_id",
            table_name="cayu_sessions",
            key_definitions=("parent_session_id", "created_at", "id"),
            predicate_definition=None,
            create_statement=(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
                "idx_cayu_sessions_parent_created_id "
                "ON cayu_sessions(parent_session_id, created_at, id)"
            ),
            drop_statement=(
                "DROP INDEX CONCURRENTLY IF EXISTS idx_cayu_sessions_parent_created_id"
            ),
        ),
    ),
    27: (
        _ConcurrentIndexMigration(
            index_name="idx_cayu_tasks_session_created_id",
            table_name="cayu_tasks",
            key_definitions=("session_id", "created_at", "id"),
            predicate_definition=None,
            create_statement=(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
                "idx_cayu_tasks_session_created_id "
                'ON cayu_tasks(session_id, created_at, id COLLATE "C")'
            ),
            drop_statement=("DROP INDEX CONCURRENTLY IF EXISTS idx_cayu_tasks_session_created_id"),
            required_key_collations=(None, None, "C"),
        ),
        _ConcurrentIndexMigration(
            index_name="idx_cayu_tasks_parent_created_id",
            table_name="cayu_tasks",
            key_definitions=("parent_task_id", "created_at", "id"),
            predicate_definition=None,
            create_statement=(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
                "idx_cayu_tasks_parent_created_id "
                'ON cayu_tasks(parent_task_id, created_at, id COLLATE "C")'
            ),
            drop_statement=("DROP INDEX CONCURRENTLY IF EXISTS idx_cayu_tasks_parent_created_id"),
            required_key_collations=(None, None, "C"),
        ),
    ),
    29: (
        _ConcurrentIndexMigration(
            index_name="idx_cayu_events_workflow_step_replay",
            table_name="cayu_events",
            key_definitions=(
                "session_id",
                "workflow_name",
                "event -> 'payload' ->> 'step_id'",
                "event_type",
                "sequence",
            ),
            predicate_definition="""
                event_type = ANY (ARRAY[
                    'workflow.step.started',
                    'workflow.step.completed'
                ])
            """,
            create_statement="""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS
                    idx_cayu_events_workflow_step_replay
                ON cayu_events(
                    session_id,
                    workflow_name,
                    (event -> 'payload' ->> 'step_id'),
                    event_type,
                    sequence DESC
                )
                WHERE event_type IN (
                    'workflow.step.started',
                    'workflow.step.completed'
                )
            """,
            drop_statement=(
                "DROP INDEX CONCURRENTLY IF EXISTS idx_cayu_events_workflow_step_replay"
            ),
        ),
        _ConcurrentIndexMigration(
            index_name="idx_cayu_events_workflow_attempt_marker",
            table_name="cayu_events",
            key_definitions=("session_id", "workflow_name", "sequence"),
            predicate_definition=("event_type = 'custom.cayu.workflow.attempt'"),
            create_statement="""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS
                    idx_cayu_events_workflow_attempt_marker
                ON cayu_events(session_id, workflow_name, sequence DESC)
                WHERE event_type = 'custom.cayu.workflow.attempt'
            """,
            drop_statement=(
                "DROP INDEX CONCURRENTLY IF EXISTS idx_cayu_events_workflow_attempt_marker"
            ),
        ),
        _ConcurrentIndexMigration(
            index_name="idx_cayu_events_workflow_step_attempt",
            table_name="cayu_events",
            key_definitions=(
                "session_id",
                "workflow_name",
                "event -> 'payload' ->> 'attempt_id'",
                "event -> 'payload' ->> 'step_id'",
                "event_type",
                "sequence",
            ),
            predicate_definition="""
                event_type = ANY (ARRAY[
                    'workflow.step.started',
                    'workflow.step.completed'
                ])
            """,
            create_statement="""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS
                    idx_cayu_events_workflow_step_attempt
                ON cayu_events(
                    session_id,
                    workflow_name,
                    (event -> 'payload' ->> 'attempt_id'),
                    (event -> 'payload' ->> 'step_id'),
                    event_type,
                    sequence DESC
                )
                WHERE event_type IN (
                    'workflow.step.started',
                    'workflow.step.completed'
                )
            """,
            drop_statement=(
                "DROP INDEX CONCURRENTLY IF EXISTS idx_cayu_events_workflow_step_attempt"
            ),
        ),
    ),
    30: (
        _ConcurrentIndexMigration(
            index_name="idx_cayu_sessions_parent_created_id",
            table_name="cayu_sessions",
            key_definitions=("parent_session_id", "created_at", "id"),
            predicate_definition=None,
            create_statement=(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
                "idx_cayu_sessions_parent_created_id "
                'ON cayu_sessions(parent_session_id, created_at, id COLLATE "C")'
            ),
            drop_statement=(
                "DROP INDEX CONCURRENTLY IF EXISTS idx_cayu_sessions_parent_created_id"
            ),
            required_key_collations=(None, None, "C"),
            replace_existing=True,
        ),
    ),
    34: (
        _ConcurrentIndexMigration(
            index_name="idx_cayu_tasks_claim_availability",
            table_name="cayu_tasks",
            key_definitions=("created_at", "id", "available_at"),
            predicate_definition=("status = 'pending' AND session_id IS NULL"),
            create_statement=(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
                "idx_cayu_tasks_claim_availability "
                "ON cayu_tasks(created_at, id, available_at) "
                "WHERE status = 'pending' AND session_id IS NULL"
            ),
            drop_statement=("DROP INDEX CONCURRENTLY IF EXISTS idx_cayu_tasks_claim_availability"),
        ),
    ),
    40: (
        _ConcurrentIndexMigration(
            index_name="idx_cayu_checkpoints_queued_dispatch_run",
            table_name="cayu_checkpoints",
            key_definitions=("session_id",),
            predicate_definition=("state #> '{session_run_operation,queue_task_id}' IS NOT NULL"),
            create_statement=(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
                "idx_cayu_checkpoints_queued_dispatch_run "
                'ON cayu_checkpoints(session_id COLLATE "C") '
                "WHERE state #> '{session_run_operation,queue_task_id}' IS NOT NULL"
            ),
            drop_statement=(
                "DROP INDEX CONCURRENTLY IF EXISTS idx_cayu_checkpoints_queued_dispatch_run"
            ),
            required_key_collations=("C",),
        ),
        _ConcurrentIndexMigration(
            index_name="idx_cayu_checkpoints_queued_dispatch_receipts",
            table_name="cayu_checkpoints",
            key_definitions=("session_id",),
            predicate_definition=(
                "state #> '{queued_dispatch_terminal_receipts,receipts}' IS NOT NULL"
            ),
            create_statement=(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
                "idx_cayu_checkpoints_queued_dispatch_receipts "
                'ON cayu_checkpoints(session_id COLLATE "C") '
                "WHERE state #> "
                "'{queued_dispatch_terminal_receipts,receipts}' IS NOT NULL"
            ),
            drop_statement=(
                "DROP INDEX CONCURRENTLY IF EXISTS idx_cayu_checkpoints_queued_dispatch_receipts"
            ),
            required_key_collations=("C",),
        ),
    ),
    46: (
        _ConcurrentIndexMigration(
            index_name="idx_cayu_transcript_messages_narrative_fts",
            table_name="cayu_transcript_messages",
            key_definitions=("to_tsvector('simple'::regconfig, transcript_search_document)",),
            predicate_definition="message ->> 'role' = ANY (ARRAY['user', 'assistant'])",
            create_statement="""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS
                    idx_cayu_transcript_messages_narrative_fts
                ON cayu_transcript_messages USING GIN (
                    to_tsvector(
                        'simple'::regconfig,
                        transcript_search_document
                    )
                )
                WHERE message ->> 'role' IN ('user', 'assistant')
            """,
            drop_statement=(
                "DROP INDEX CONCURRENTLY IF EXISTS idx_cayu_transcript_messages_narrative_fts"
            ),
            access_method="gin",
        ),
    ),
    52: (
        _ConcurrentIndexMigration(
            index_name="idx_cayu_public_authority_public_alias",
            table_name="cayu_public_authority_aliases",
            key_definitions=("field_name", "public_alias"),
            predicate_definition=None,
            create_statement=(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
                "idx_cayu_public_authority_public_alias "
                "ON cayu_public_authority_aliases(field_name, public_alias)"
            ),
            drop_statement=(
                "DROP INDEX CONCURRENTLY IF EXISTS idx_cayu_public_authority_public_alias"
            ),
        ),
    ),
    70: (
        _ConcurrentIndexMigration(
            index_name="idx_cayu_tasks_interrupted_handoff_recovery",
            table_name="cayu_tasks",
            key_definitions=("status", "lease_expires_at", "id"),
            predicate_definition=None,
            create_statement=(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
                "idx_cayu_tasks_interrupted_handoff_recovery "
                "ON cayu_tasks(status, lease_expires_at, id)"
            ),
            drop_statement=(
                "DROP INDEX CONCURRENTLY IF EXISTS idx_cayu_tasks_interrupted_handoff_recovery"
            ),
        ),
    ),
    76: (
        _ConcurrentIndexMigration(
            index_name="idx_cayu_tasks_interrupted_handoff_continuation",
            table_name="cayu_tasks",
            key_definitions=("status", "created_at", "id"),
            predicate_definition=(
                "worker_id IS NULL AND lease_expires_at IS NULL "
                "AND interrupted_handoff_id IS NOT NULL AND session_id IS NOT NULL "
                "AND session_instance_id IS NOT NULL AND status_reason IS NULL"
            ),
            create_statement=(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
                "idx_cayu_tasks_interrupted_handoff_continuation "
                "ON cayu_tasks(status, created_at, id) "
                "WHERE worker_id IS NULL AND lease_expires_at IS NULL "
                "AND interrupted_handoff_id IS NOT NULL AND session_id IS NOT NULL "
                "AND session_instance_id IS NOT NULL AND status_reason IS NULL"
            ),
            drop_statement=(
                "DROP INDEX CONCURRENTLY IF EXISTS idx_cayu_tasks_interrupted_handoff_continuation"
            ),
        ),
        _ConcurrentIndexMigration(
            index_name="idx_cayu_tasks_interrupted_handoff_generation",
            table_name="cayu_tasks",
            key_definitions=("interrupted_handoff_id",),
            predicate_definition="interrupted_handoff_id IS NOT NULL",
            unique=True,
            create_statement=(
                "CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS "
                "idx_cayu_tasks_interrupted_handoff_generation "
                "ON cayu_tasks(interrupted_handoff_id) "
                "WHERE interrupted_handoff_id IS NOT NULL"
            ),
            drop_statement=(
                "DROP INDEX CONCURRENTLY IF EXISTS idx_cayu_tasks_interrupted_handoff_generation"
            ),
        ),
    ),
    # Include delegated-action updates in the revision-97 pending-action index.
    97: (
        _ConcurrentIndexMigration(
            index_name="idx_cayu_events_pending_action_lookup",
            table_name="cayu_events",
            key_definitions=("session_id", "pending_action_lookup_key", "event_type", "sequence"),
            predicate_definition="""
                event_type = ANY (ARRAY[
                    'tool.call.approval_requested', 'session.awaiting_user_input',
                    'session.interrupted', 'session.delegated_action.updated',
                    'tool.call.started', 'tool.call.completed', 'tool.call.failed',
                    'tool.call.blocked', 'tool.call.approval_denied'
                ]) AND pending_action_lookup_key IS NOT NULL
            """,
            create_statement="""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_cayu_events_pending_action_lookup
                ON cayu_events(session_id, pending_action_lookup_key, event_type, sequence)
                WHERE event_type IN (
                    'tool.call.approval_requested', 'session.awaiting_user_input',
                    'session.interrupted', 'session.delegated_action.updated',
                    'tool.call.started', 'tool.call.completed', 'tool.call.failed',
                    'tool.call.blocked', 'tool.call.approval_denied'
                ) AND pending_action_lookup_key IS NOT NULL
            """,
            drop_statement=(
                "DROP INDEX CONCURRENTLY IF EXISTS idx_cayu_events_pending_action_lookup"
            ),
            replace_existing=True,
            replacement_predicates=(
                """
                event_type = ANY (ARRAY[
                    'tool.call.approval_requested', 'session.awaiting_user_input',
                    'session.interrupted', 'tool.call.started', 'tool.call.completed',
                    'tool.call.failed', 'tool.call.blocked', 'tool.call.approval_denied'
                ]) AND pending_action_lookup_key IS NOT NULL
                """,
            ),
        ),
    ),
}


def _required_concurrent_indexes(revision: int) -> tuple[_ConcurrentIndexMigration, ...]:
    latest_by_name: dict[str, _ConcurrentIndexMigration] = {}
    for index_revision, indexes in sorted(_CONCURRENT_INDEX_MIGRATIONS.items()):
        if index_revision > revision:
            break
        for index in indexes:
            latest_by_name[index.index_name] = index
    return tuple(latest_by_name.values())
