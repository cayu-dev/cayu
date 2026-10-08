from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable
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
from cayu.storage import _sqlite_closure_schema as sqlite_closure_schema
from cayu.storage import _sqlite_connection as sqlite_connection
from cayu.storage import _sqlite_eval_schema as sqlite_eval_schema
from cayu.storage import _sqlite_knowledge_schema as sqlite_knowledge_schema
from cayu.storage import _sqlite_memory_evidence_schema as sqlite_memory_evidence_schema
from cayu.storage import _sqlite_records as sqlite_records
from cayu.storage import _sqlite_schema_history as sqlite_schema_history
from cayu.storage import _sqlite_session_schema as sqlite_session_schema
from cayu.storage import _sqlite_task_schema as sqlite_task_schema
from cayu.storage import _sqlite_transcript_schema as sqlite_transcript_schema
from cayu.storage import _sqlite_verified_work_schema as sqlite_verified_work_schema
from cayu.storage import _sqlite_work_context_schema as sqlite_work_context_schema
from cayu.storage import migrations as schema
from cayu.storage._collaboration_schema import validate_sqlite_collaboration_schema
from cayu.storage._collaboration_wait_schema import validate_sqlite_wait_discovery
from cayu.storage._context_selection_schema import validate_sqlite_context_selection_schema
from cayu.storage._diagnostic_inspection import current_diagnostic_store_inspection
from cayu.storage._participant_bindings_schema import validate_sqlite_participant_bindings
from cayu.storage._product_operation_schema import validate_sqlite_product_operation_schema
from cayu.storage.knowledge_transition import require_empty_knowledge_revision_transition
from cayu.tasks.handoff import TaskInterruptedHandoffRequest, prepare_interrupted_task_handoff
from cayu.tasks.records import Task, TaskStatus

_INTERRUPTED_HANDOFF_MIGRATION_BATCH_SIZE = 256


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
            for statement in sqlite_catalog._iter_statements(
                sqlite_schema_history._MIGRATION_STEPS[revision]
            ):
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


def _revision_17_index_definitions() -> dict[str, str]:
    definitions: dict[str, str] = {}
    for statement in sqlite_catalog._iter_statements(sqlite_schema_history._MIGRATION_STEPS[17]):
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
                sqlite_catalog._normalize_sqlite_schema_definition(actual_definition)
                != sqlite_catalog._normalize_sqlite_schema_definition(expected)
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
    for statement in sqlite_catalog._iter_statements(sqlite_schema_history._MIGRATION_STEPS[29]):
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
            or sqlite_catalog._normalize_sqlite_schema_definition(actual_definition)
            != sqlite_catalog._normalize_sqlite_schema_definition(expected)
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
        for statement in sqlite_catalog._iter_statements(sqlite_schema_history._MIGRATION_STEPS[23])
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
    for (name, kind), expected in zip(
        names,
        sqlite_catalog._iter_statements(sqlite_schema_history._MIGRATION_STEPS[110]),
        strict=True,
    ):
        row = connection.execute(
            "SELECT type, sql FROM sqlite_master WHERE name = ?", (name,)
        ).fetchone()
        if (
            row is None
            or row[0] != kind
            or row[1] is None
            or sqlite_catalog._normalize_sqlite_schema_definition(row[1])
            != sqlite_catalog._normalize_sqlite_schema_definition(expected)
        ):
            raise RuntimeError(
                "Required Cayu producer cleanup receipt table or fence is missing or conflicting."
            )


def _validate_reservation_inventory_index(connection: sqlite3.Connection) -> None:
    name = "idx_cayu_budget_reservations_session_identity"
    row = connection.execute(
        "SELECT type, tbl_name, sql FROM sqlite_master WHERE name = ?", (name,)
    ).fetchone()
    expected = next(sqlite_catalog._iter_statements(sqlite_schema_history._MIGRATION_STEPS[109]))
    if (
        row is None
        or row[0] != "index"
        or row[1] != "cayu_budget_reservations"
        or row[2] is None
        or sqlite_catalog._normalize_sqlite_schema_definition(row[2])
        != sqlite_catalog._normalize_sqlite_schema_definition(expected)
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
            sqlite_catalog._normalize_sqlite_schema_definition(actual_definition)
            != sqlite_catalog._normalize_sqlite_schema_definition(
                _reservation_event_index_definition()
            )
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
    for statement in sqlite_catalog._iter_statements(sqlite_schema_history._MIGRATION_STEPS[23]):
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
            or sqlite_catalog._normalize_sqlite_schema_definition(actual_definition)
            != sqlite_catalog._normalize_sqlite_schema_definition(expected)
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
        connection.execute(sqlite_schema_history._MIGRATIONS_TABLE_DDL)
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
        sqlite_closure_schema._validate_revision_88_closure_schema(connection)
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
        sqlite_transcript_schema._validate_revision_46_transcript_search_schema(
            connection, expected_tokenizer_version=TRANSCRIPT_SEARCH_TOKENIZER_VERSION
        )
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
        for statement in sqlite_catalog._iter_statements(sqlite_schema_history._BASELINE_DDL):
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
            for statement in sqlite_catalog._iter_statements(
                sqlite_schema_history._MIGRATION_STEPS[23]
            ):
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
            for statement in sqlite_catalog._iter_statements(
                sqlite_schema_history._MIGRATION_STEPS[29]
            ):
                connection.execute(statement)
            _validate_workflow_replay_indexes(connection, require_all=True)
            _record_revision(connection, rev)
            connection.execute(f"PRAGMA user_version = {rev.revision}")
        return
    if rev.revision == 38:
        with sqlite_connection._transaction(connection):
            for statement in sqlite_catalog._iter_statements(
                sqlite_schema_history._MIGRATION_STEPS[38]
            ):
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
        for table, column, decl in sqlite_schema_history._MIGRATION_ADD_COLUMNS.get(
            rev.revision, ()
        ):
            _add_column_if_missing(connection, table, column, decl)
        for table, column in sqlite_schema_history._MIGRATION_DROP_COLUMNS.get(rev.revision, ()):
            _drop_column_if_present(connection, table, column)
        ddl = sqlite_schema_history._MIGRATION_STEPS.get(rev.revision)
        if ddl:
            for statement in sqlite_catalog._iter_statements(ddl):
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
            sqlite_transcript_schema._validate_revision_46_transcript_search_schema(
                connection, expected_tokenizer_version=TRANSCRIPT_SEARCH_TOKENIZER_VERSION
            )
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
            sqlite_closure_schema._validate_revision_88_closure_schema(connection)
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
            for statement in sqlite_catalog._iter_statements(
                sqlite_schema_history._MIGRATION_STEPS[72]
            ):
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
        for table, column, decl in sqlite_schema_history._MIGRATION_ADD_COLUMNS[17]:
            _add_column_if_missing(connection, table, column, decl)
        for statement in sqlite_catalog._iter_statements(
            sqlite_schema_history._MIGRATION_STEPS[17]
        ):
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
