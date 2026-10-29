"""PostgreSQL transcript search schema and tokenizer identity checks.

Callers retain revision gates, validation order and the existing cursor/transaction.
These functions inspect the schema without acquiring connections or applying DDL.
"""

from __future__ import annotations

from typing import Any


async def _validate_transcript_search_document_column(cur: Any, *, tokenizer_version: str) -> None:
    await cur.execute(
        """
            SELECT data_type, is_nullable, is_generated
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'cayu_transcript_messages'
              AND column_name = 'transcript_search_document'
            """
    )
    if await cur.fetchone() != ("text", "NO", "NEVER"):
        raise RuntimeError(
            "Postgres transcript search document column is missing or nullable. "
            "Recreate or restore a known-good revision-46 Cayu database."
        )
    await cur.execute(
        """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'cayu_transcript_search_configuration'
            ORDER BY ordinal_position
            """
    )
    if tuple(await cur.fetchall()) != (
        ("singleton", "boolean", "NO"),
        ("tokenizer_version", "text", "NO"),
    ):
        raise RuntimeError(
            "Postgres transcript search tokenizer configuration is missing or "
            "malformed. Recreate a revision-46 Cayu database with this runtime."
        )
    await cur.execute(
        """
            SELECT array_agg(attribute.attname ORDER BY key.ordinality)
            FROM pg_constraint AS constraint_definition
            JOIN pg_class AS relation
              ON relation.oid = constraint_definition.conrelid
            JOIN pg_namespace AS namespace
              ON namespace.oid = relation.relnamespace
            CROSS JOIN LATERAL unnest(constraint_definition.conkey)
                WITH ORDINALITY AS key(attribute_number, ordinality)
            JOIN pg_attribute AS attribute
              ON attribute.attrelid = relation.oid
             AND attribute.attnum = key.attribute_number
            WHERE namespace.nspname = current_schema()
              AND relation.relname = 'cayu_transcript_search_configuration'
              AND constraint_definition.contype = 'p'
            """
    )
    if await cur.fetchone() != (["singleton"],):
        raise RuntimeError(
            "Postgres transcript search tokenizer configuration lacks its "
            "singleton primary key. Recreate a revision-46 Cayu database."
        )
    await cur.execute(
        """
            SELECT pg_get_constraintdef(constraint_definition.oid, TRUE)
            FROM pg_constraint AS constraint_definition
            JOIN pg_class AS relation
              ON relation.oid = constraint_definition.conrelid
            JOIN pg_namespace AS namespace
              ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = current_schema()
              AND relation.relname = 'cayu_transcript_search_configuration'
              AND constraint_definition.contype = 'c'
            ORDER BY constraint_definition.conname
            """
    )
    check_definitions = {"".join(str(row[0]).lower().split()) for row in await cur.fetchall()}
    if "check(singleton)" not in check_definitions:
        raise RuntimeError(
            "Postgres transcript search tokenizer configuration lacks its "
            "singleton constraint. Recreate a revision-46 Cayu database."
        )
    await cur.execute(
        "SELECT singleton, tokenizer_version "
        "FROM cayu_transcript_search_configuration ORDER BY singleton"
    )
    if tuple(await cur.fetchall()) != ((True, tokenizer_version),):
        raise RuntimeError(
            "Postgres transcript search tokenizer identity conflicts with this runtime. "
            "Recreate a revision-46 Cayu database with this runtime."
        )


async def _validate_transcript_message_bytes(
    cur: Any, *, event_content_references: bool = False
) -> None:
    """Check the size column and the trigger that fills it.

    From revision 134 the trigger keeps a writer-supplied size for a row whose
    tool results are stored in their tool event.
    """

    await cur.execute(
        """
            SELECT data_type, is_nullable, is_generated
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'cayu_transcript_messages'
              AND column_name = 'message_bytes'
            """
    )
    if await cur.fetchone() != ("bigint", "YES", "NEVER"):
        raise RuntimeError("Postgres transcript message size column is missing or conflicting.")
    await cur.execute(
        """
            SELECT trigger_definition.tgenabled,
                   pg_get_triggerdef(trigger_definition.oid),
                   pg_get_functiondef(trigger_definition.tgfoid)
            FROM pg_trigger AS trigger_definition
            JOIN pg_class AS relation ON relation.oid = trigger_definition.tgrelid
            JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = current_schema()
              AND relation.relname = 'cayu_transcript_messages'
              AND trigger_definition.tgname = 'cayu_record_transcript_message_bytes'
              AND NOT trigger_definition.tgisinternal
            """
    )
    row = await cur.fetchone()
    trigger = "" if row is None else " ".join(str(row[1]).lower().split())
    function = "" if row is None else " ".join(str(row[2]).lower().split())
    if (
        row is None
        or row[0] != "O"
        or "before insert or update of message on" not in trigger
        or "for each row" not in trigger
        or (
            "new.message_bytes := octet_length((new.message)::text)" not in function
            and "new.message_bytes := octet_length(new.message::text)" not in function
        )
        or (
            event_content_references
            and "or not jsonb_path_exists(new.message, '$.content[*].cayu_event_content')"
            not in function
        )
    ):
        raise RuntimeError("Postgres transcript message size trigger is missing or conflicting.")
