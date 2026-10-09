"""PostgreSQL schema checks for permanent budget reservation ownership.

Callers retain revision gates, validation order and the existing cursor/transaction.
These functions inspect the schema without acquiring connections or applying DDL.
"""

from __future__ import annotations

from typing import Any


async def _validate_budget_reservation_identity_registry(
    cur: Any,
    *,
    require: bool,
    verify_event_ownership: bool = False,
) -> bool:
    table_name = "cayu_budget_reservation_identities"
    await cur.execute("SELECT to_regclass(%s)", (table_name,))
    registered = await cur.fetchone()
    if registered is None or registered[0] is None:
        if require:
            raise RuntimeError(
                f"Required Cayu Postgres table is missing: {table_name}. "
                "Restore the permanent reservation ownership registry from "
                "a known-good backup."
            )
        return False
    await cur.execute(
        """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = %s
            ORDER BY ordinal_position
            """,
        (table_name,),
    )
    columns = tuple(await cur.fetchall())
    await cur.execute(
        """
            SELECT pg_get_constraintdef(constraint_record.oid)
            FROM pg_catalog.pg_constraint AS constraint_record
            JOIN pg_catalog.pg_class AS table_record
              ON table_record.oid = constraint_record.conrelid
            JOIN pg_catalog.pg_namespace AS namespace
              ON namespace.oid = table_record.relnamespace
            WHERE namespace.nspname = current_schema()
              AND table_record.relname = %s
              AND constraint_record.contype = 'p'
            """,
        (table_name,),
    )
    primary_keys = tuple(row[0] for row in await cur.fetchall())
    expected_columns = (
        ("reservation_id", "text", "NO"),
        ("publication_session_id", "text", "NO"),
        ("publication_id", "text", "NO"),
        ("published", "boolean", "NO"),
    )
    if columns != expected_columns or primary_keys != ("PRIMARY KEY (reservation_id)",):
        raise RuntimeError(
            f"Postgres schema object {table_name!r} conflicts with Cayu's "
            "reservation identity contract. Restore the required ownership "
            "registry from a known-good backup."
        )
    if not verify_event_ownership:
        return True
    await cur.execute(
        """
            SELECT 1
            FROM cayu_events AS event
            LEFT JOIN cayu_budget_reservation_identities AS identity
              ON identity.reservation_id = event.payload ->> 'reservation_id'
            WHERE event.event_type = 'budget.reserved'
              AND jsonb_typeof(event.payload -> 'reservation_id') = 'string'
              AND (
                  identity.reservation_id IS NULL
                  OR identity.publication_session_id <> event.session_id
                  OR identity.publication_id <> event.event_id
                  OR NOT identity.published
              )
            LIMIT 1
            """
    )
    if await cur.fetchone() is not None:
        raise RuntimeError(
            "Postgres budget reservation events disagree with the permanent "
            "reservation ownership registry."
        )
    return True
