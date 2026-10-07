"""SQLite schema checks for task provenance, receipts, retries and handoffs.

Reconciliation and migration callers retain revision gates, validation order and
native transaction ownership. These checks only inspect existing schema.
"""

from __future__ import annotations

import sqlite3

from cayu.storage import _sqlite_catalog as sqlite_catalog


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


def _validate_interrupted_handoff_generation_column(connection: sqlite3.Connection) -> None:
    task_columns = {
        str(row[1]): (str(row[2]).upper(), int(row[3]))
        for row in connection.execute("PRAGMA table_info(cayu_tasks)")
    }
    index_name = "idx_cayu_tasks_interrupted_handoff_continuation"
    index = connection.execute(
        "SELECT tbl_name, sql FROM sqlite_master WHERE type = 'index' AND name = ?",
        (index_name,),
    ).fetchone()
    index_columns = tuple(
        str(row[2]) for row in connection.execute(f"PRAGMA index_info({index_name})")
    )
    normalized_index_sql = sqlite_catalog._normalize_sqlite_schema_sql(
        None if index is None else index[1]
    )
    generation_index_name = "idx_cayu_tasks_interrupted_handoff_generation"
    generation_index = connection.execute(
        "SELECT tbl_name, sql FROM sqlite_master WHERE type = 'index' AND name = ?",
        (generation_index_name,),
    ).fetchone()
    generation_index_columns = tuple(
        str(row[2]) for row in connection.execute(f"PRAGMA index_info({generation_index_name})")
    )
    normalized_generation_index_sql = sqlite_catalog._normalize_sqlite_schema_sql(
        None if generation_index is None else generation_index[1]
    )
    claim_columns = tuple(
        (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
        for row in connection.execute(
            "PRAGMA table_info(cayu_task_interrupted_continuation_claims)"
        )
    )
    expected_claim_columns = (
        ("handoff_id_sha256", "TEXT", 1, 1),
        ("task_id", "TEXT", 1, 0),
        ("worker_id", "TEXT", 1, 0),
        ("claimed_at", "TEXT", 1, 0),
    )
    claim_foreign_keys = connection.execute(
        "PRAGMA foreign_key_list(cayu_task_interrupted_continuation_claims)"
    ).fetchall()
    required_predicate_fragments = (
        "where worker_id is null",
        "lease_expires_at is null",
        "interrupted_handoff_id is not null",
        "session_id is not null",
        "session_instance_id is not null",
        "status_reason is null",
    )
    if (
        task_columns.get("interrupted_handoff_id") != ("TEXT", 0)
        or index is None
        or index[0] != "cayu_tasks"
        or index_columns != ("status", "created_at", "id")
        or any(fragment not in normalized_index_sql for fragment in required_predicate_fragments)
        or generation_index is None
        or generation_index[0] != "cayu_tasks"
        or generation_index_columns != ("interrupted_handoff_id",)
        or "create unique index" not in normalized_generation_index_sql
        or "where interrupted_handoff_id is not null" not in normalized_generation_index_sql
        or claim_columns != expected_claim_columns
        or claim_foreign_keys
    ):
        raise RuntimeError(
            "SQLite task handoff generation or bounded continuation index conflicts with Cayu's "
            "revision-76 durability contract. Run `cayu storage migrate` or restore "
            "the database from a known-good backup."
        )
