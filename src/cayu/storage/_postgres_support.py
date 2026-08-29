from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from cayu._validation import copy_label_map
from cayu.runtime.invocation import SessionInvocation, TaskInvocation
from cayu.runtime.sessions import (
    TRANSCRIPT_SEARCH_TOKENIZER_VERSION,
    PendingActionSession,
    Session,
    SessionOrder,
    SessionStatus,
    SessionTopologyNode,
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
    """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_cayu_events_budget_reservation_identity
    ON cayu_events ((payload ->> 'reservation_id'))
    WHERE event_type = 'budget.reserved'
      AND jsonb_typeof(payload -> 'reservation_id') = 'string'
    """,
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
        to_utc(session.created_at),
        to_utc(session.updated_at),
        to_utc(session.last_activity_at),
        session.run_epoch,
        _dumps(session.invocation.model_dump(mode="json")),
        _dumps(session.metadata),
    )


def session_label_insert_values(session: Session) -> list[tuple[str, str, str]]:
    return [(session.id, key, value) for key, value in sorted(session.labels.items())]


def session_from_row(row: tuple[Any, ...], labels: dict[str, str] | None = None) -> Session:
    return Session(
        id=row[0],
        instance_id=row[1],
        agent_name=row[2],
        provider_name=row[3],
        model=row[4],
        parent_session_id=row[5],
        causal_budget_id=row[6],
        runtime_name=row[7],
        runtime_version=row[8],
        environment_name=row[9],
        status=SessionStatus(row[10]),
        created_at=to_utc(row[11]),
        updated_at=to_utc(row[12]),
        last_activity_at=to_utc(row[13]),
        run_epoch=row[14],
        invocation=SessionInvocation.model_validate(_loads(row[15])),
        metadata=_loads(row[16]),
        labels=copy_label_map(labels, "labels"),
    )


SESSION_COLUMNS = (
    "id, instance_id, agent_name, provider_name, model, parent_session_id, causal_budget_id, "
    "runtime_name, runtime_version, environment_name, status, created_at, updated_at, "
    "last_activity_at, run_epoch, invocation, metadata"
)

SESSION_TOPOLOGY_COLUMNS = (
    "id, agent_name, provider_name, model, parent_session_id, causal_budget_id, "
    "runtime_name, runtime_version, environment_name, status, created_at, updated_at, "
    "last_activity_at"
)


def session_topology_node_from_row(row: tuple[Any, ...]) -> SessionTopologyNode:
    return SessionTopologyNode(
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
        last_activity_at=to_utc(row[12]),
    )


PENDING_ACTION_SESSION_COLUMNS = (
    "id, agent_name, provider_name, model, parent_session_id, causal_budget_id, "
    "runtime_name, runtime_version, environment_name, status, created_at, updated_at"
)


def pending_action_session_from_row(
    row: tuple[Any, ...],
    labels: dict[str, str] | None = None,
) -> PendingActionSession:
    return PendingActionSession(
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
        labels=copy_label_map(labels, "labels"),
    )


def task_insert_values(task: Task) -> tuple[object, ...]:
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
        to_utc_optional(task.available_at),
        task.worker_id,
        to_utc_optional(task.lease_expires_at),
        task.status_reason,
        None if task.status_payload is None else _dumps(task.status_payload),
        _dumps(task.input),
        None if task.result is None else _dumps(task.result),
        None if task.error is None else _dumps(task.error),
        _dumps(task.metadata),
        to_utc(task.created_at),
        to_utc(task.updated_at),
        to_utc_optional(task.started_at),
        to_utc_optional(task.completed_at),
        _dumps(task.invocation.model_dump(mode="json")),
        (None if task.retry_series is None else _dumps(task.retry_series.model_dump(mode="json"))),
        (
            None
            if task.work_contract is None
            else _dumps(task.work_contract.model_dump(mode="json", warnings=False))
        ),
    )


TASK_COLUMNS = (
    "id, type, title, description, status, session_id, session_instance_id, parent_task_id, "
    "assigned_agent_name, available_at, worker_id, lease_expires_at, status_reason, "
    "status_payload, input, result, error, metadata, created_at, updated_at, started_at, "
    "completed_at, invocation, retry_series, work_contract"
)


def task_from_row(row: tuple[Any, ...]) -> Task:
    return Task(
        id=row[0],
        type=row[1],
        title=row[2],
        description=row[3],
        status=TaskStatus(row[4]),
        session_id=row[5],
        session_instance_id=row[6],
        parent_task_id=row[7],
        assigned_agent_name=row[8],
        available_at=to_utc_optional(row[9]),
        worker_id=row[10],
        lease_expires_at=to_utc_optional(row[11]),
        status_reason=row[12],
        status_payload=None if row[13] is None else _loads(row[13]),
        input=_loads(row[14]),
        result=None if row[15] is None else _loads(row[15]),
        error=None if row[16] is None else _loads(row[16]),
        metadata=_loads(row[17]),
        created_at=to_utc(row[18]),
        updated_at=to_utc(row[19]),
        started_at=to_utc_optional(row[20]),
        completed_at=to_utc_optional(row[21]),
        invocation=TaskInvocation.model_validate(_loads(row[22])),
        retry_series=(
            None if row[23] is None else TaskRetrySeriesSnapshot.model_validate(_loads(row[23]))
        ),
        work_contract=(
            None if row[24] is None else WorkContractRef.model_validate(_loads(row[24]))
        ),
    )


TASK_TOPOLOGY_COLUMNS = f"""
    CASE
        WHEN octet_length(id) <= {TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES}
        THEN id
    END AS topology_id,
    octet_length(id) > {TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES}
        AS topology_id_oversized,
    CASE
        WHEN octet_length(type) <= {TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES}
        THEN type
    END AS topology_type,
    octet_length(type) > {TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES}
        AS topology_type_truncated,
    CASE
        WHEN title IS NULL
          OR octet_length(title) <= {TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES}
        THEN title
    END AS topology_title,
    title IS NOT NULL
      AND octet_length(title) > {TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES}
        AS topology_title_truncated,
    CASE
        WHEN octet_length(status) <= {TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES}
        THEN status
    END AS topology_status,
    CASE
        WHEN status_reason IS NULL
          OR octet_length(status_reason) <= {TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES}
        THEN status_reason
    END AS topology_status_reason,
    status_reason IS NOT NULL
      AND octet_length(status_reason) > {TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES}
        AS topology_status_reason_truncated,
    CASE
        WHEN session_id IS NULL
          OR octet_length(session_id) <= {TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES}
        THEN session_id
    END AS topology_session_id,
    session_id IS NOT NULL
      AND octet_length(session_id) > {TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES}
        AS topology_session_id_oversized,
    CASE
        WHEN parent_task_id IS NULL
          OR octet_length(parent_task_id) <= {TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES}
        THEN parent_task_id
    END AS topology_parent_task_id,
    parent_task_id IS NOT NULL
      AND octet_length(parent_task_id) > {TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES}
        AS topology_parent_task_id_oversized,
    CASE
        WHEN assigned_agent_name IS NULL
          OR octet_length(assigned_agent_name)
             <= {TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES}
        THEN assigned_agent_name
    END AS topology_assigned_agent_name,
    assigned_agent_name IS NOT NULL
      AND octet_length(assigned_agent_name)
          > {TASK_TOPOLOGY_MAX_DISPLAY_TEXT_BYTES}
        AS topology_assigned_agent_name_truncated,
    created_at AS topology_created_at,
    updated_at AS topology_updated_at
"""


def task_topology_node_from_row(row: tuple[Any, ...]) -> TaskTopologyNode:
    if row[1] or row[10] or row[12]:
        raise TaskTopologyInconsistent(
            "A task topology record contains an oversized structural identifier."
        )
    truncated_fields = tuple(
        field_name
        for field_name, truncated in (
            ("type", row[3]),
            ("title", row[5]),
            ("assigned_agent_name", row[14]),
            ("status_reason", row[8]),
        )
        if truncated
    )
    try:
        return TaskTopologyNode(
            id=row[0],
            type=row[2],
            title=row[4],
            status=TaskStatus(row[6]),
            status_reason=row[7],
            session_id=row[9],
            parent_task_id=row[11],
            assigned_agent_name=row[13],
            created_at=to_utc(row[15]),
            updated_at=to_utc(row[16]),
            truncated_fields=truncated_fields,
        )
    except (TypeError, ValueError) as exc:
        raise TaskTopologyInconsistent(
            "A task record cannot be represented by the bounded topology contract."
        ) from exc


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


def _dumps(value: Any) -> str:
    # JSONB columns accept a JSON-text string; we serialize explicitly so the
    # same json round-trip semantics as the SQLite store are preserved.
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _loads(value: Any) -> Any:
    # psycopg returns JSONB as already-decoded Python objects, but we accept a
    # JSON string too for robustness across configurations.
    if isinstance(value, str):
        return json.loads(value)
    return value
