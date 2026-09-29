"""Revision-112 schema for runtime-owned product operation stores."""

from __future__ import annotations

import sqlite3
from typing import Any

from cayu.storage.migrations import SchemaError

PRODUCT_OPERATIONS_TABLE = "cayu_product_operations"
PRODUCT_OPERATIONS_REVISION = 112

_COLUMNS = (
    "work_id",
    "public_id",
    "tenant_id",
    "subject_id",
    "idempotency_key",
    "request_fingerprint",
    "session_id",
    "task_id",
    "request_text",
    "status",
    "result",
    "result_receipt",
    "recovery_status",
    "execution_claim_id",
    "execution_claim_expires_at",
)
_REQUIRED = frozenset(_COLUMNS[:10])
_UNIQUE = ("idempotency_key", "public_id", "session_id", "task_id")
_POSTGRES_TYPES = {
    "result_receipt": "jsonb",
    "execution_claim_expires_at": "timestamp with time zone",
}
_SQLITE_TYPES = {"execution_claim_expires_at": "INTEGER"}

# Each authority identity is individually unique, so public reads, idempotent
# reservation, and trusted continuation lookup use exact indexed rows.
POSTGRES_PRODUCT_OPERATION_DDL: tuple[str, ...] = (
    f"""CREATE TABLE IF NOT EXISTS {PRODUCT_OPERATIONS_TABLE} (
        work_id TEXT PRIMARY KEY,
        public_id TEXT NOT NULL UNIQUE,
        tenant_id TEXT NOT NULL,
        subject_id TEXT NOT NULL,
        idempotency_key TEXT NOT NULL UNIQUE,
        request_fingerprint TEXT NOT NULL,
        session_id TEXT NOT NULL UNIQUE,
        task_id TEXT NOT NULL UNIQUE,
        request_text TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('pending', 'completed', 'failed')),
        result TEXT,
        result_receipt JSONB,
        recovery_status TEXT,
        execution_claim_id TEXT,
        execution_claim_expires_at TIMESTAMPTZ
    )""",
)

SQLITE_PRODUCT_OPERATION_DDL = f"""
    CREATE TABLE IF NOT EXISTS {PRODUCT_OPERATIONS_TABLE} (
        work_id TEXT PRIMARY KEY NOT NULL,
        public_id TEXT NOT NULL UNIQUE,
        tenant_id TEXT NOT NULL,
        subject_id TEXT NOT NULL,
        idempotency_key TEXT NOT NULL UNIQUE,
        request_fingerprint TEXT NOT NULL,
        session_id TEXT NOT NULL UNIQUE,
        task_id TEXT NOT NULL UNIQUE,
        request_text TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('pending', 'completed', 'failed')),
        result TEXT,
        result_receipt TEXT,
        recovery_status TEXT,
        execution_claim_id TEXT,
        execution_claim_expires_at INTEGER
    );
"""


def validate_sqlite_product_operation_schema(connection: sqlite3.Connection) -> None:
    """Refuse a missing or conflicting product-operation authority table."""

    rows = connection.execute(f"PRAGMA table_info({PRODUCT_OPERATIONS_TABLE})").fetchall()
    expected = [
        (
            name,
            _SQLITE_TYPES.get(name, "TEXT"),
            1 if name in _REQUIRED else 0,
            1 if name == "work_id" else 0,
        )
        for name in _COLUMNS
    ]
    if [(row[1], str(row[2]).upper(), row[3], row[5]) for row in rows] != expected:
        raise SchemaError("Cayu product operation table is missing or conflicting.")
    if connection.execute(f"PRAGMA foreign_key_list({PRODUCT_OPERATIONS_TABLE})").fetchall():
        raise SchemaError("Cayu product operations must not depend on other Cayu records.")
    unique_columns: list[str] = []
    for index in connection.execute(f"PRAGMA index_list({PRODUCT_OPERATIONS_TABLE})"):
        if not index[2] or index[4]:
            continue
        columns = [row[2] for row in connection.execute(f"PRAGMA index_info('{index[1]}')")]
        if len(columns) == 1:
            unique_columns.append(columns[0])
    if sorted(unique_columns) != sorted(("work_id", *_UNIQUE)):
        raise SchemaError("Cayu product operation identity indexes are missing or conflicting.")


async def validate_postgres_product_operation_schema(cursor: Any) -> None:
    """Refuse a missing or conflicting product-operation authority table."""

    await cursor.execute(
        "SELECT a.attname, format_type(a.atttypid, a.atttypmod), a.attnotnull "
        "FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid "
        "WHERE c.oid = to_regclass(%s) AND c.relkind = 'r' "
        "AND a.attnum > 0 AND NOT a.attisdropped ORDER BY a.attnum",
        (PRODUCT_OPERATIONS_TABLE,),
    )
    expected = [(name, _POSTGRES_TYPES.get(name, "text"), name in _REQUIRED) for name in _COLUMNS]
    if list(await cursor.fetchall()) != expected:
        raise SchemaError("Cayu product operation table is missing or conflicting.")
    await cursor.execute(
        "SELECT contype, pg_get_constraintdef(oid) FROM pg_constraint "
        "WHERE conrelid = to_regclass(%s) AND contype IN ('p', 'u', 'f') "
        "ORDER BY contype, pg_get_constraintdef(oid)",
        (PRODUCT_OPERATIONS_TABLE,),
    )
    if list(await cursor.fetchall()) != [
        ("p", "PRIMARY KEY (work_id)"),
        *(("u", f"UNIQUE ({column})") for column in _UNIQUE),
    ]:
        raise SchemaError("Cayu product operation identity constraints are missing or conflicting.")
    await cursor.execute(
        "SELECT count(*) FROM pg_constraint WHERE conrelid = to_regclass(%s) "
        "AND contype = 'c' AND convalidated "
        "AND pg_get_constraintdef(oid) LIKE '%%status%%pending%%completed%%failed%%'",
        (PRODUCT_OPERATIONS_TABLE,),
    )
    row = await cursor.fetchone()
    if row is None or row[0] != 1:
        raise SchemaError("Cayu product operation status constraint is missing or conflicting.")


__all__ = [
    "POSTGRES_PRODUCT_OPERATION_DDL",
    "PRODUCT_OPERATIONS_REVISION",
    "PRODUCT_OPERATIONS_TABLE",
    "SQLITE_PRODUCT_OPERATION_DDL",
    "validate_postgres_product_operation_schema",
    "validate_sqlite_product_operation_schema",
]
