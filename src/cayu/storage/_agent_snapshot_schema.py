"""Revision-120 PostgreSQL schema for ``PostgresAgentSnapshotStore``.

The tables and columns mirror the ones ``SQLiteAgentSnapshotStore`` creates when
it opens. Documents are canonical JSON text, not JSONB, because both stores
compare stored documents byte for byte. ``released`` stays an integer flag so
the dialect-neutral storage-retention SQL reads pins and protections the same
way on both backends. On PostgreSQL the tables join Cayu's shared revision
history: ``cayu storage migrate`` creates them and stores only validate them.
"""

from __future__ import annotations

from typing import Any

from cayu.storage.migrations import SchemaError

AGENT_SNAPSHOT_REVISION = 120

AGENT_SNAPSHOT_TABLES = (
    "cayu_agent_snapshot_records",
    "cayu_agent_snapshot_materialization_progress",
    "cayu_agent_snapshot_nodes",
    "cayu_agent_snapshot_roots",
    "cayu_agent_snapshot_root_nodes",
    "cayu_agent_snapshot_bindings",
    "cayu_agent_snapshot_pins",
    "cayu_agent_snapshot_protections",
    "cayu_agent_snapshot_lifecycle_operations",
    "cayu_agent_snapshot_gc_plans",
    "cayu_agent_snapshot_gc_receipts",
)

# (column, PostgreSQL type, NOT NULL) per table, in SQLite's column order.
_COLUMNS: dict[str, tuple[tuple[str, str, bool], ...]] = {
    "cayu_agent_snapshot_records": (
        ("record_kind", "text", True),
        ("fingerprint", "text", True),
        ("document", "text", True),
    ),
    "cayu_agent_snapshot_materialization_progress": (
        ("snapshot_fingerprint", "text", True),
        ("candidate_id", "text", True),
        ("state_scope_id", "text", True),
        ("state_mode", "text", True),
        ("progress_id", "text", True),
        ("revision", "bigint", True),
        ("document", "text", True),
        ("materialization_fingerprint", "text", False),
    ),
    "cayu_agent_snapshot_nodes": (
        ("digest", "text", True),
        ("node_kind", "text", True),
        ("schema_id", "text", True),
        ("byte_count", "bigint", True),
        ("document", "text", True),
    ),
    "cayu_agent_snapshot_roots": (
        ("snapshot_root", "text", True),
        ("manifest_document", "text", True),
        ("manifest_bytes", "bigint", True),
        ("closure_bytes", "bigint", True),
        ("object_count", "bigint", True),
    ),
    "cayu_agent_snapshot_root_nodes": (
        ("snapshot_root", "text", True),
        ("node_digest", "text", True),
        ("ordinal", "bigint", True),
    ),
    "cayu_agent_snapshot_bindings": (
        ("binding_id", "text", True),
        ("snapshot_root", "text", True),
        ("authority_scope_fingerprint", "text", True),
        ("binding_document", "text", True),
        ("snapshot_document", "text", True),
        ("put_receipt_document", "text", True),
    ),
    "cayu_agent_snapshot_pins": (
        ("pin_id", "text", True),
        ("snapshot_root", "text", True),
        ("binding_id", "text", True),
        ("document", "text", True),
        ("released", "integer", True),
    ),
    "cayu_agent_snapshot_protections": (
        ("protection_id", "text", True),
        ("snapshot_root", "text", True),
        ("binding_id", "text", True),
        ("kind", "text", True),
        ("document", "text", True),
        ("released", "integer", True),
    ),
    "cayu_agent_snapshot_lifecycle_operations": (
        ("operation_kind", "text", True),
        ("operation_id", "text", True),
        ("request_digest", "text", True),
        ("response_model", "text", True),
        ("response_document", "text", True),
    ),
    "cayu_agent_snapshot_gc_plans": (
        ("operation_id", "text", True),
        ("plan_id", "text", True),
        ("request_digest", "text", True),
        ("request_document", "text", True),
        ("plan_document", "text", True),
    ),
    "cayu_agent_snapshot_gc_receipts": (
        ("plan_id", "text", True),
        ("receipt_document", "text", True),
    ),
}

# Identity constraints per table, as PostgreSQL's pg_get_constraintdef renders them.
_CONSTRAINTS: dict[str, tuple[tuple[str, str], ...]] = {
    "cayu_agent_snapshot_records": (("p", "PRIMARY KEY (record_kind, fingerprint)"),),
    "cayu_agent_snapshot_materialization_progress": (
        (
            "p",
            "PRIMARY KEY (snapshot_fingerprint, candidate_id, state_scope_id, state_mode)",
        ),
        ("u", "UNIQUE (progress_id)"),
    ),
    "cayu_agent_snapshot_nodes": (("p", "PRIMARY KEY (digest)"),),
    "cayu_agent_snapshot_roots": (("p", "PRIMARY KEY (snapshot_root)"),),
    "cayu_agent_snapshot_root_nodes": (
        ("p", "PRIMARY KEY (snapshot_root, node_digest)"),
        ("u", "UNIQUE (snapshot_root, ordinal)"),
    ),
    "cayu_agent_snapshot_bindings": (("p", "PRIMARY KEY (binding_id)"),),
    "cayu_agent_snapshot_pins": (("p", "PRIMARY KEY (pin_id)"),),
    "cayu_agent_snapshot_protections": (("p", "PRIMARY KEY (protection_id)"),),
    "cayu_agent_snapshot_lifecycle_operations": (
        ("p", "PRIMARY KEY (operation_kind, operation_id)"),
    ),
    "cayu_agent_snapshot_gc_plans": (
        ("p", "PRIMARY KEY (operation_id)"),
        ("u", "UNIQUE (plan_id)"),
    ),
    "cayu_agent_snapshot_gc_receipts": (("p", "PRIMARY KEY (plan_id)"),),
}

_INDEXES = (
    (
        "cayu_agent_snapshot_root_nodes_by_node",
        "cayu_agent_snapshot_root_nodes",
        "node_digest, snapshot_root",
    ),
    ("cayu_agent_snapshot_bindings_by_root", "cayu_agent_snapshot_bindings", "snapshot_root"),
    (
        "cayu_agent_snapshot_active_pins_by_root",
        "cayu_agent_snapshot_pins",
        "snapshot_root, released",
    ),
    (
        "cayu_agent_snapshot_active_protections_by_root",
        "cayu_agent_snapshot_protections",
        "snapshot_root, released",
    ),
)

_RELEASED_CHECK = "CHECK (released IN (0, 1))"


def _table_ddl(table: str) -> str:
    lines = []
    for name, kind, not_null in _COLUMNS[table]:
        line = f"{name} {kind.upper()}" + (" NOT NULL" if not_null else "")
        if name == "released":
            line += f" DEFAULT 0 {_RELEASED_CHECK}"
        lines.append(line)
    lines.extend(definition for _, definition in _CONSTRAINTS[table])
    body = ",\n    ".join(lines)
    return f"CREATE TABLE IF NOT EXISTS {table} (\n    {body}\n)"


#: Revision-120 statements ``cayu storage migrate`` runs on PostgreSQL.
POSTGRES_AGENT_SNAPSHOT_DDL: tuple[str, ...] = (
    *(_table_ddl(table) for table in AGENT_SNAPSHOT_TABLES),
    *(
        f"CREATE INDEX IF NOT EXISTS {name} ON {table} ({columns})"
        for name, table, columns in _INDEXES
    ),
)


async def validate_postgres_agent_snapshot_schema(cursor: Any) -> None:
    """Refuse missing or conflicting agent snapshot tables, keys or indexes."""

    for table in AGENT_SNAPSHOT_TABLES:
        await cursor.execute(
            "SELECT a.attname, format_type(a.atttypid, a.atttypmod), a.attnotnull "
            "FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid "
            "WHERE c.oid = to_regclass(%s) AND c.relkind = 'r' "
            "AND a.attnum > 0 AND NOT a.attisdropped ORDER BY a.attnum",
            (table,),
        )
        expected = list(_COLUMNS[table])
        if [tuple(row) for row in await cursor.fetchall()] != expected:
            raise SchemaError(f"Cayu agent snapshot table {table} is missing or conflicting.")
        await cursor.execute(
            "SELECT contype, pg_get_constraintdef(oid) FROM pg_constraint "
            "WHERE conrelid = to_regclass(%s) AND contype IN ('p', 'u', 'f') "
            "ORDER BY contype, pg_get_constraintdef(oid)",
            (table,),
        )
        if [tuple(row) for row in await cursor.fetchall()] != sorted(_CONSTRAINTS[table]):
            raise SchemaError(
                f"Cayu agent snapshot table {table} identity constraints are missing "
                "or conflicting."
            )
    await cursor.execute(
        "SELECT count(*) FROM pg_constraint WHERE contype = 'c' AND convalidated "
        "AND conrelid IN (to_regclass(%s), to_regclass(%s)) "
        "AND pg_get_constraintdef(oid) = %s",
        (
            "cayu_agent_snapshot_pins",
            "cayu_agent_snapshot_protections",
            "CHECK ((released = ANY (ARRAY[0, 1])))",
        ),
    )
    row = await cursor.fetchone()
    if row is None or row[0] != 2:
        raise SchemaError("Cayu agent snapshot release flags are missing or conflicting.")
    for name, table, columns in _INDEXES:
        await cursor.execute(
            "SELECT pg_get_indexdef(i.indexrelid) FROM pg_index i "
            "WHERE i.indexrelid = to_regclass(%s) AND i.indrelid = to_regclass(%s) "
            "AND i.indisvalid AND i.indisready",
            (name, table),
        )
        index_row = await cursor.fetchone()
        if index_row is None or not str(index_row[0]).endswith(f"({columns})"):
            raise SchemaError(f"Cayu agent snapshot index {name} is missing or not ready.")


__all__ = [
    "AGENT_SNAPSHOT_REVISION",
    "AGENT_SNAPSHOT_TABLES",
    "POSTGRES_AGENT_SNAPSHOT_DDL",
    "validate_postgres_agent_snapshot_schema",
]
