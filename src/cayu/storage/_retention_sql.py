"""Dialect-neutral SQL execution and audit records for storage retention.

Retention cores are written once against :class:`RetentionSql`. SQL uses ``?``
placeholders; the Postgres executor rewrites them. Literal ``%`` must never
appear in retention SQL: pass LIKE patterns as parameters instead.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import Any

from cayu.storage import _sqlite_records as sqlite_records
from cayu.storage.retention import (
    MAX_RETENTION_AUDIT_LIST_LIMIT,
    RetentionAuditEntry,
    RetentionAuditRecord,
    RetentionAuditState,
    RetentionAuditUnavailable,
    RetentionMode,
)

AUDIT_TABLES = ("cayu_storage_retention_runs", "cayu_storage_retention_entries")
#: Agent-snapshot tables a SQLiteAgentSnapshotStore creates when it shares a database.
SNAPSHOT_TABLES = (
    "cayu_agent_snapshot_pins",
    "cayu_agent_snapshot_protections",
    "cayu_agent_snapshot_bindings",
    "cayu_agent_snapshot_roots",
    "cayu_agent_snapshot_root_nodes",
    "cayu_agent_snapshot_nodes",
)
SQL_CHUNK = 500


class RetentionSql:
    """One connection (and transaction) that a retention core reads or writes."""

    postgres: bool = False

    async def all(self, sql: str, parameters: Sequence[Any] = ()) -> list[tuple[Any, ...]]:
        raise NotImplementedError

    async def one(self, sql: str, parameters: Sequence[Any] = ()) -> tuple[Any, ...] | None:
        rows = await self.all(sql, parameters)
        return rows[0] if rows else None

    async def run(self, sql: str, parameters: Sequence[Any] = ()) -> int:
        raise NotImplementedError

    async def exists(self, sql: str, parameters: Sequence[Any] = ()) -> bool:
        return await self.one(sql, parameters) is not None

    # -- dialect fragments -------------------------------------------------

    def size(self, expression: str) -> str:
        """Logical stored bytes of a text or JSON expression."""

        if self.postgres:
            return f"octet_length(({expression})::text)"
        return f"length(CAST({expression} AS BLOB))"

    def false(self, column: str) -> str:
        return f"NOT {column}" if self.postgres else f"{column} = 0"

    def timestamp(self, value: datetime) -> Any:
        return value if self.postgres else sqlite_records.format_datetime(value)

    def json(self, value: Any) -> Any:
        if self.postgres:
            from psycopg.types.json import Jsonb

            return Jsonb(value)
        return sqlite_records.json_dumps(value)

    @staticmethod
    def read_timestamp(value: Any) -> datetime:
        if isinstance(value, datetime):
            from cayu.storage._postgres_support import to_utc

            return to_utc(value)
        return sqlite_records.parse_datetime(value)

    @staticmethod
    def read_json(value: Any) -> Any:
        if isinstance(value, str | bytes):
            return json.loads(value)
        return value

    async def existing_tables(self, names: Iterable[str]) -> set[str]:
        names = tuple(names)
        if not names:
            return set()
        marks = ", ".join("?" for _ in names)
        if self.postgres:
            rows = await self.all(
                "SELECT table_name FROM information_schema.tables "
                f"WHERE table_schema = current_schema() AND table_name IN ({marks})",
                names,
            )
        else:
            rows = await self.all(
                f"SELECT name FROM sqlite_master WHERE type = 'table' AND name IN ({marks})",
                names,
            )
        return {row[0] for row in rows}


def marks(values: Sequence[Any]) -> str:
    return ", ".join("?" for _ in values)


def chunks(values: Sequence[Any], size: int = SQL_CHUNK) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


class SQLiteRetentionSql(RetentionSql):
    """Synchronous SQLite connection used from the event loop under its owner's lock."""

    postgres = False

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    async def all(self, sql: str, parameters: Sequence[Any] = ()) -> list[tuple[Any, ...]]:
        return [tuple(row) for row in self.connection.execute(sql, tuple(parameters)).fetchall()]

    async def run(self, sql: str, parameters: Sequence[Any] = ()) -> int:
        return self.connection.execute(sql, tuple(parameters)).rowcount


class PostgresRetentionSql(RetentionSql):
    postgres = True

    def __init__(self, cursor: Any) -> None:
        self.cursor = cursor

    async def all(self, sql: str, parameters: Sequence[Any] = ()) -> list[tuple[Any, ...]]:
        await self.cursor.execute(sql.replace("?", "%s"), tuple(parameters))
        if self.cursor.description is None:
            return []
        return [tuple(row) for row in await self.cursor.fetchall()]

    async def run(self, sql: str, parameters: Sequence[Any] = ()) -> int:
        await self.cursor.execute(sql.replace("?", "%s"), tuple(parameters))
        return self.cursor.rowcount


async def snapshot_pin_text(sql: RetentionSql) -> str:
    """Every document of agent snapshots held by an unreleased pin or protection."""

    documents: list[str] = []
    for row in await sql.all(
        "WITH held AS ("
        "SELECT snapshot_root, binding_id FROM cayu_agent_snapshot_pins WHERE released = 0 "
        "UNION SELECT snapshot_root, binding_id FROM cayu_agent_snapshot_protections "
        "WHERE released = 0) "
        "SELECT b.binding_document, b.snapshot_document, b.put_receipt_document "
        "FROM cayu_agent_snapshot_bindings b JOIN held h ON h.binding_id = b.binding_id "
        "UNION ALL SELECT r.manifest_document, '', '' FROM cayu_agent_snapshot_roots r "
        "WHERE r.snapshot_root IN (SELECT snapshot_root FROM held) "
        "UNION ALL SELECT n.document, '', '' FROM cayu_agent_snapshot_nodes n "
        "JOIN cayu_agent_snapshot_root_nodes rn ON rn.node_digest = n.digest "
        "WHERE rn.snapshot_root IN (SELECT snapshot_root FROM held)"
    ):
        documents.extend(str(value) for value in row if value)
    return "\0".join(documents)


MAX_REFERENCE_ATOM_CHARS = 512
_DIGEST = re.compile(r"[0-9a-f]{64}")


def held_snapshot_identifiers(documents: Iterable[str]) -> frozenset[str]:
    """Every short string value and SHA-256 hex digest in snapshot documents."""

    found: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, str):
            if len(value) <= MAX_REFERENCE_ATOM_CHARS:
                found.add(value)
        elif isinstance(value, dict):
            for nested in value.values():
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)

    for document in documents:
        found.update(_DIGEST.findall(document))
        try:
            walk(json.loads(document))
        except ValueError:
            continue
    return frozenset(found)


async def json_reference_atoms(
    sql: RetentionSql,
    table: str,
    column: str,
    *,
    key_suffix: str,
    id_parent: str | None = None,
) -> set[str]:
    """Short strings stored under keys ending in ``key_suffix`` in a JSON column.

    Array elements under a plural key (``key_suffix + "s"``) count too, and so
    does ``id`` inside an object keyed ``id_parent``. Matching is deliberately
    broad: a false positive only keeps more data.
    """

    single, plural = f"%{key_suffix}", f"%{key_suffix}s"
    if sql.postgres:
        id_clause = " OR (key = 'id' AND parent_key = ?)" if id_parent else ""
        rows = await sql.all(
            "WITH RECURSIVE nodes(parent_key, key, value) AS ("
            f"SELECT NULL::text, NULL::text, ({column})::jsonb FROM {table} "
            f"WHERE {column} IS NOT NULL "
            "UNION ALL "
            "SELECT n.key, child.key, child.value FROM nodes n CROSS JOIN LATERAL ("
            "SELECT e.key, e.value FROM jsonb_each(CASE WHEN jsonb_typeof(n.value) = 'object' "
            "THEN n.value ELSE '{}'::jsonb END) AS e "
            "UNION ALL SELECT n.key, a.value FROM jsonb_array_elements("
            "CASE WHEN jsonb_typeof(n.value) = 'array' THEN n.value ELSE '[]'::jsonb END) AS a"
            ") AS child) "
            "SELECT DISTINCT value #>> '{}' FROM nodes WHERE jsonb_typeof(value) = 'string' "
            f"AND length(value #>> '{{}}') <= ? AND (key LIKE ? OR key LIKE ?{id_clause})",
            (MAX_REFERENCE_ATOM_CHARS, single, plural, *((id_parent,) if id_parent else ())),
        )
    else:
        id_clause = " OR (t.key = 'id' AND t.path LIKE ?)" if id_parent else ""
        # json_tree quotes keys containing underscores in paths (for example
        # $."artifact_ids"). Accept both path spellings for array elements.
        rows = await sql.all(
            f"SELECT DISTINCT t.atom FROM {table} AS r, json_tree(r.{column}) AS t "
            f"WHERE r.{column} IS NOT NULL AND t.type = 'text' AND length(t.atom) <= ? "
            f"AND (CAST(t.key AS TEXT) LIKE ? OR t.path LIKE ? OR t.path LIKE ?{id_clause})",
            (
                MAX_REFERENCE_ATOM_CHARS,
                single,
                plural,
                plural + '"',
                *((f"%.{id_parent}",) if id_parent else ()),
            ),
        )
    return {row[0] for row in rows if isinstance(row[0], str)}


# -- audit records -------------------------------------------------------------


async def require_audit_tables(sql: RetentionSql) -> None:
    if await sql.existing_tables(AUDIT_TABLES) != set(AUDIT_TABLES):
        raise RetentionAuditUnavailable(
            "Retention apply requires storage revision 118 for its audit records; "
            "run `cayu storage migrate` first. Dry runs work without it."
        )


async def begin_audit(
    sql: RetentionSql,
    *,
    audit_id: str,
    store_kind: str,
    mode: RetentionMode,
    started_at: datetime,
    policy: Mapping[str, Any],
) -> None:
    await sql.run(
        "INSERT INTO cayu_storage_retention_runs "
        "(audit_id, store_kind, mode, state, started_at, policy_json) "
        "VALUES (?, ?, ?, 'started', ?, ?)",
        (
            audit_id,
            store_kind,
            mode.value,
            sql.timestamp(started_at),
            sql.json(dict(policy)),
        ),
    )


async def record_audit_entry(
    sql: RetentionSql,
    *,
    audit_id: str,
    item_id: str,
    mode: RetentionMode,
    counts: Mapping[str, int],
    size: int,
    recorded_at: datetime,
) -> None:
    await sql.run(
        "INSERT INTO cayu_storage_retention_entries "
        "(audit_id, item_id, action, bytes, counts_json, recorded_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            audit_id,
            item_id,
            mode.value,
            size,
            sql.json(dict(sorted(counts.items()))),
            sql.timestamp(recorded_at),
        ),
    )


async def complete_audit(
    sql: RetentionSql,
    *,
    audit_id: str,
    completed_at: datetime,
    summary: Mapping[str, Any],
) -> None:
    await sql.run(
        "UPDATE cayu_storage_retention_runs SET state = 'completed', completed_at = ?, "
        "summary_json = ? WHERE audit_id = ?",
        (sql.timestamp(completed_at), sql.json(dict(summary)), audit_id),
    )


_RUN_COLUMNS = (
    "audit_id, store_kind, mode, state, started_at, completed_at, policy_json, summary_json"
)


async def list_audits(
    sql: RetentionSql,
    *,
    limit: int,
    store_kind: str | None,
    item_id: str | None,
) -> tuple[RetentionAuditRecord, ...]:
    if type(limit) is not int or not 1 <= limit <= MAX_RETENTION_AUDIT_LIST_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_RETENTION_AUDIT_LIST_LIMIT}.")
    await require_audit_tables(sql)
    filters: list[str] = []
    parameters: list[Any] = []
    if store_kind is not None:
        filters.append("store_kind = ?")
        parameters.append(store_kind)
    if item_id is not None:
        filters.append(
            "audit_id IN (SELECT audit_id FROM cayu_storage_retention_entries WHERE item_id = ?)"
        )
        parameters.append(item_id)
    where = f"WHERE {' AND '.join(filters)} " if filters else ""
    rows = await sql.all(
        f"SELECT {_RUN_COLUMNS} FROM cayu_storage_retention_runs {where}"
        "ORDER BY started_at DESC, audit_id DESC LIMIT ?",
        (*parameters, limit),
    )
    return tuple(_audit_record(sql, row, ()) for row in rows)


async def load_audit(sql: RetentionSql, audit_id: str) -> RetentionAuditRecord | None:
    await require_audit_tables(sql)
    row = await sql.one(
        f"SELECT {_RUN_COLUMNS} FROM cayu_storage_retention_runs WHERE audit_id = ?",
        (audit_id,),
    )
    if row is None:
        return None
    entries = tuple(
        RetentionAuditEntry(
            item_id=entry[0],
            action=RetentionMode(entry[1]),
            counts=sql.read_json(entry[2]),
            bytes=entry[3],
            recorded_at=sql.read_timestamp(entry[4]),
        )
        for entry in await sql.all(
            "SELECT item_id, action, counts_json, bytes, recorded_at "
            "FROM cayu_storage_retention_entries WHERE audit_id = ? "
            "ORDER BY recorded_at, item_id",
            (audit_id,),
        )
    )
    return _audit_record(sql, row, entries)


def _audit_record(
    sql: RetentionSql, row: tuple[Any, ...], entries: tuple[RetentionAuditEntry, ...]
) -> RetentionAuditRecord:
    return RetentionAuditRecord(
        audit_id=row[0],
        store_kind=row[1],
        mode=RetentionMode(row[2]),
        state=RetentionAuditState(row[3]),
        started_at=sql.read_timestamp(row[4]),
        completed_at=None if row[5] is None else sql.read_timestamp(row[5]),
        policy=sql.read_json(row[6]),
        summary={} if row[7] is None else sql.read_json(row[7]),
        entries=entries,
    )
