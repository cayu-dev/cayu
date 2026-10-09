"""PostgreSQL schema checks for producer cleanup receipts and retirement fences.

Callers retain revision gates, validation order and the existing cursor/transaction.
These functions inspect the schema without acquiring connections or applying DDL.
"""

from __future__ import annotations

from typing import Any


async def _validate_producer_cleanup_receipts(cur: Any) -> None:
    await cur.execute(
        "SELECT a.attname, format_type(a.atttypid, a.atttypmod), a.attnotnull "
        "FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid "
        "WHERE c.oid = to_regclass('cayu_producer_cleanup_receipts') "
        "AND c.relkind = 'r' AND a.attnum > 0 AND NOT a.attisdropped ORDER BY a.attname"
    )
    if list(await cur.fetchall()) != [
        ("generation", "bigint", True),
        ("namespace_key", "text", True),
        ("operation_key", "text", True),
        ("receipt_json", "jsonb", True),
    ]:
        raise RuntimeError(
            "Required Cayu producer cleanup receipt table is missing or conflicting."
        )
    await cur.execute(
        "SELECT contype, pg_get_constraintdef(oid) FROM pg_constraint "
        "WHERE conrelid = to_regclass('cayu_producer_cleanup_receipts') AND contype IN ('p', 'f')"
    )
    if list(await cur.fetchall()) != [("p", "PRIMARY KEY (operation_key)")]:
        raise RuntimeError("Producer cleanup receipts require an independent exact primary key.")
    await cur.execute(
        "SELECT a.attname, format_type(a.atttypid, a.atttypmod), a.attnotnull "
        "FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid "
        "WHERE c.oid = to_regclass('cayu_producer_cleanup_retirements') "
        "AND c.relkind = 'r' AND a.attnum > 0 AND NOT a.attisdropped ORDER BY a.attname"
    )
    if list(await cur.fetchall()) != [
        ("namespace_key", "text", True),
        ("through_generation", "bigint", True),
    ]:
        raise RuntimeError("Producer cleanup retirement fence is missing or conflicting.")
    await cur.execute(
        "SELECT contype, pg_get_constraintdef(oid) FROM pg_constraint "
        "WHERE conrelid = to_regclass('cayu_producer_cleanup_retirements') AND contype IN ('p', 'f')"
    )
    if list(await cur.fetchall()) != [("p", "PRIMARY KEY (namespace_key)")]:
        raise RuntimeError("Producer cleanup retirement fence requires an independent key.")
    await cur.execute(
        "SELECT i.indisvalid, i.indisready, i.indisunique, i.indpred IS NULL, i.indexprs IS NULL, "
        "ARRAY(SELECT a.attname FROM unnest(i.indkey) WITH ORDINALITY AS k(attnum, ordinal) "
        "JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = k.attnum ORDER BY k.ordinal) "
        "FROM pg_index i WHERE i.indexrelid = to_regclass('idx_cayu_producer_cleanup_namespace') "
        "AND i.indrelid = to_regclass('cayu_producer_cleanup_receipts')"
    )
    if await cur.fetchone() != (
        True,
        True,
        False,
        True,
        True,
        ["namespace_key", "generation", "operation_key"],
    ):
        raise RuntimeError("Producer cleanup receipt namespace index is missing or conflicting.")
