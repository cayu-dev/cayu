"""SQLite producer cleanup receipt and retirement schema checks.

Callers supply canonical revision SQL and retain migration/repair ownership.
These checks only read existing schema and rows.
"""

from __future__ import annotations

import sqlite3

from cayu.storage import _sqlite_catalog as sqlite_catalog


def _validate_producer_cleanup_receipts(
    connection: sqlite3.Connection, *, revision_sql: str
) -> None:
    names = (
        ("cayu_producer_cleanup_receipts", "table"),
        ("idx_cayu_producer_cleanup_namespace", "index"),
        ("cayu_producer_cleanup_retirements", "table"),
    )
    for (name, kind), expected in zip(
        names,
        sqlite_catalog._iter_statements(revision_sql),
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
