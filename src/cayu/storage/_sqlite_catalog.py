"""SQLite catalog inspection and schema SQL utilities."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator


def _sqlite_table_columns(connection: sqlite3.Connection, table: str) -> tuple[str, ...]:
    return tuple(str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})"))


def _sqlite_has_unique_index(
    connection: sqlite3.Connection,
    table: str,
    columns: tuple[str, ...],
) -> bool:
    primary_key_columns = tuple(
        str(row[1])
        for row in sorted(
            (row for row in connection.execute(f"PRAGMA table_info({table})") if row[5]),
            key=lambda row: int(row[5]),
        )
    )
    if primary_key_columns == columns:
        return True
    for row in connection.execute(f"PRAGMA index_list({table})"):
        if not bool(row[2]):
            continue
        index_columns = tuple(
            str(column[2]) for column in connection.execute(f"PRAGMA index_info({row[1]})")
        )
        if index_columns == columns:
            return True
    return False


def _sqlite_foreign_key_groups(
    connection: sqlite3.Connection,
    table: str,
) -> set[tuple[str, tuple[str, ...], tuple[str, ...], str]]:
    grouped: dict[int, list[sqlite3.Row]] = {}
    for row in connection.execute(f"PRAGMA foreign_key_list({table})"):
        grouped.setdefault(int(row[0]), []).append(row)
    result: set[tuple[str, tuple[str, ...], tuple[str, ...], str]] = set()
    for rows in grouped.values():
        ordered = sorted(rows, key=lambda row: int(row[1]))
        result.add(
            (
                str(ordered[0][2]),
                tuple(str(row[3]) for row in ordered),
                tuple(str(row[4]) for row in ordered),
                str(ordered[0][6]).upper(),
            )
        )
    return result


def _normalize_sqlite_schema_sql(value: object | None) -> str:
    return " ".join(str(value or "").lower().split())


def _normalize_sqlite_schema_definition(definition: str) -> str:
    """Normalize formatting, while preserving every structural SQL token."""
    normalized = re.sub(r"\s+", "", definition.casefold())
    normalized = normalized.replace('"', "").replace("`", "").replace("[", "").replace("]", "")
    return normalized.replace("ifnotexists", "")


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
