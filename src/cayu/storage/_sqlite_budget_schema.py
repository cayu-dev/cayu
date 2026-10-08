"""SQLite budget reservation indexes and durable ownership checks.

Callers supply canonical revision SQL and retain migration/repair ownership.
These checks only read existing schema and rows.
"""

from __future__ import annotations

import sqlite3

from cayu.storage import _sqlite_catalog as sqlite_catalog

_RESERVATION_EVENT_INDEX_NAME = "idx_cayu_events_budget_reservation_identity"


_RESERVATION_IDENTITY_TABLE_NAME = "cayu_budget_reservation_identities"


def _reservation_event_index_definition(revision_sql: str) -> str:
    statements = tuple(
        statement
        for statement in sqlite_catalog._iter_statements(revision_sql)
        if _RESERVATION_EVENT_INDEX_NAME in statement
    )
    if len(statements) != 1:
        raise RuntimeError("Cayu reservation event index definition is incomplete.")
    return statements[0]


def _validate_reservation_inventory_index(
    connection: sqlite3.Connection, *, revision_sql: str
) -> None:
    name = "idx_cayu_budget_reservations_session_identity"
    row = connection.execute(
        "SELECT type, tbl_name, sql FROM sqlite_master WHERE name = ?", (name,)
    ).fetchone()
    expected = next(sqlite_catalog._iter_statements(revision_sql))
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
    revision_sql: str,
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
                _reservation_event_index_definition(revision_sql=revision_sql)
            )
        )
    ):
        raise RuntimeError(
            f"SQLite schema object {_RESERVATION_EVENT_INDEX_NAME!r} conflicts with "
            "Cayu's reservation identity contract. Rename or remove the conflicting object, then run "
            "with schema_mode='migrate' to create the required unique index."
        )


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
