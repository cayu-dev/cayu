"""PostgreSQL schema checks for session identity, messages, grants and public aliases.

Callers retain revision gates, validation order and the existing cursor/transaction.
These functions inspect the schema without acquiring connections or applying DDL.
"""

from __future__ import annotations

import re
from typing import Any, NoReturn


async def _validate_session_invocation_column(cur: Any) -> None:
    await cur.execute(
        """
            SELECT data_type, is_nullable, is_generated
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'cayu_sessions'
              AND column_name = 'invocation'
            """
    )
    if await cur.fetchone() != ("jsonb", "NO", "NEVER"):
        raise RuntimeError(
            "Postgres schema object 'cayu_sessions.invocation' conflicts with "
            "Cayu's required invocation-provenance contract. Recreate the Cayu "
            "database from a known-good revision-36 schema."
        )


async def _validate_child_session_lifecycle_schema(cur: Any) -> None:
    table = "cayu_child_session_lifecycle_candidates"
    await cur.execute(
        """
            SELECT column_name, data_type, is_nullable, collation_name
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'cayu_child_session_lifecycle_candidates'
            ORDER BY ordinal_position
            """
    )
    if tuple(await cur.fetchall()) != (
        ("child_session_id", "text", "NO", "C"),
        ("parent_session_id", "text", "NO", "C"),
        ("priority", "integer", "NO", None),
        ("sort_at", "timestamp with time zone", "NO", None),
    ):
        _raise_child_session_lifecycle_schema_error(table)
    await cur.execute(
        """
            SELECT constraint_record.contype,
                   pg_get_constraintdef(constraint_record.oid)
            FROM pg_catalog.pg_constraint AS constraint_record
            JOIN pg_catalog.pg_class AS table_record
              ON table_record.oid = constraint_record.conrelid
            JOIN pg_catalog.pg_namespace AS namespace
              ON namespace.oid = table_record.relnamespace
            WHERE namespace.nspname = current_schema()
              AND table_record.relname = 'cayu_child_session_lifecycle_candidates'
            """
    )
    constraints = tuple(
        (str(kind), " ".join(str(definition).lower().split()))
        for kind, definition in await cur.fetchall()
    )
    required_constraints = (
        ("p", ("primary key (child_session_id)",)),
        ("c", ("priority", "0", "1", "2")),
        ("f", ("foreign key (child_session_id)", "cayu_sessions(id)", "on delete cascade")),
        (
            "f",
            ("foreign key (parent_session_id)", "cayu_sessions(id)", "on delete cascade"),
        ),
    )
    if any(
        not any(
            actual_kind == expected_kind and all(fragment in definition for fragment in fragments)
            for actual_kind, definition in constraints
        )
        for expected_kind, fragments in required_constraints
    ):
        _raise_child_session_lifecycle_schema_error(table)
    await cur.execute(
        """
            SELECT indexname, indexdef
            FROM pg_catalog.pg_indexes
            WHERE schemaname = current_schema()
              AND indexname = ANY(%s)
            """,
        (
            [
                "idx_cayu_child_lifecycle_candidates_page",
                "idx_cayu_events_child_lifecycle",
                "idx_cayu_transcript_messages_session_role_order",
            ],
        ),
    )
    indexes = {
        str(name): " ".join(str(definition).lower().split())
        for name, definition in await cur.fetchall()
    }
    if (
        "parent_session_id, priority, sort_at, child_session_id"
        not in indexes.get(
            "idx_cayu_child_lifecycle_candidates_page",
            "",
        )
        or "session_id, event_type, sequence desc"
        not in indexes.get(
            "idx_cayu_events_child_lifecycle",
            "",
        )
        or "where" not in indexes.get("idx_cayu_events_child_lifecycle", "")
        or not all(
            fragment in indexes.get("idx_cayu_transcript_messages_session_role_order", "")
            for fragment in (
                "session_id",
                "message ->> 'role'::text",
                "session_order desc",
            )
        )
    ):
        _raise_child_session_lifecycle_schema_error("candidate indexes")
    expected_triggers = {
        "cayu_index_child_lifecycle_session",
        "cayu_index_child_lifecycle_event_insert",
        "cayu_index_child_lifecycle_event_delete",
        "cayu_index_child_lifecycle_event_update",
        "cayu_index_child_lifecycle_consumption_insert",
        "cayu_index_child_lifecycle_consumption_delete",
        "cayu_index_child_lifecycle_consumption_update",
    }
    await cur.execute(
        """
            SELECT trigger_record.tgname
            FROM pg_catalog.pg_trigger AS trigger_record
            JOIN pg_catalog.pg_class AS table_record
              ON table_record.oid = trigger_record.tgrelid
            JOIN pg_catalog.pg_namespace AS namespace
              ON namespace.oid = table_record.relnamespace
            WHERE namespace.nspname = current_schema()
              AND NOT trigger_record.tgisinternal
              AND trigger_record.tgname = ANY(%s)
            """,
        (list(expected_triggers),),
    )
    if {str(row[0]) for row in await cur.fetchall()} != expected_triggers:
        _raise_child_session_lifecycle_schema_error("maintenance triggers")


def _raise_child_session_lifecycle_schema_error(name: str) -> NoReturn:
    raise RuntimeError(
        "Postgres schema object "
        f"{name!r} conflicts with Cayu's bounded child-lifecycle projection. "
        "Run `cayu storage migrate` to install revision 79 or recreate the database."
    )


async def _validate_session_instance_schema(cur: Any) -> None:
    await cur.execute(
        """
            SELECT table_name, column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND (
                (table_name = 'cayu_sessions' AND column_name = 'instance_id')
                OR (table_name = 'cayu_tasks' AND column_name = 'session_instance_id')
              )
            ORDER BY table_name, column_name
            """
    )
    if tuple(await cur.fetchall()) != (
        ("cayu_sessions", "instance_id", "text", "NO"),
        ("cayu_tasks", "session_instance_id", "text", "YES"),
    ):
        raise RuntimeError("Postgres session-instance authority columns are malformed.")
    await cur.execute("SELECT EXISTS(SELECT 1 FROM cayu_sessions WHERE instance_id IS NULL)")
    row = await cur.fetchone()
    if row is None or row[0] is True:
        raise RuntimeError("Postgres session-instance authority is incomplete.")


async def _validate_session_message_lifecycle_columns(cur: Any) -> None:
    await cur.execute(
        "SELECT indisvalid, indisready, pg_get_indexdef(indexrelid) "
        "FROM pg_index WHERE indexrelid = to_regclass('idx_cayu_events_queue_acceptance')"
    )
    index = await cur.fetchone()
    definition = "" if index is None else " ".join(index[2].lower().split())
    if (
        index is None
        or not index[0]
        or not index[1]
        or "using btree (session_id, ((event #>> '{payload,queue_id}'::text[])))" not in definition
        or "where (event_type = 'session.message.queued'::text)" not in definition
    ):
        raise RuntimeError("Postgres queue acceptance lookup index conflicts with revision 83.")
    await cur.execute(
        "SELECT column_name, data_type, is_nullable, column_default "
        "FROM information_schema.columns WHERE table_schema = current_schema() "
        "AND table_name = 'cayu_session_message_queue' "
        "AND column_name IN ('conditions_json', 'terminal_json')"
    )
    columns = {row[0]: tuple(row[1:]) for row in await cur.fetchall()}
    if any(
        columns.get(name) != ("jsonb", "YES", None) for name in ("conditions_json", "terminal_json")
    ):
        raise RuntimeError("Postgres session-message lifecycle columns conflict with revision 83.")
    await cur.execute(
        "SELECT data_type, is_nullable, column_default FROM information_schema.columns "
        "WHERE table_schema = current_schema() AND table_name = 'cayu_session_message_deliveries' "
        "AND column_name = 'reject_only'"
    )
    if await cur.fetchone() != ("boolean", "NO", "false"):
        raise RuntimeError("Postgres queue rejection receipt conflicts with revision 83.")


async def _validate_session_message_queue_typed_message_column(cur: Any) -> None:
    await cur.execute(
        """
            SELECT data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'cayu_session_message_queue'
              AND column_name = 'message_json'
            """
    )
    if await cur.fetchone() != ("jsonb", "YES", None):
        raise RuntimeError(
            "Postgres schema object 'cayu_session_message_queue.message_json' "
            "conflicts with Cayu's revision-57 typed queued-message contract. "
            "Run `cayu storage migrate` or restore the database from a known-good backup."
        )


async def _validate_targeted_tool_grant_schema(cur: Any) -> None:
    expected_columns = {
        "cayu_targeted_tool_grants": (
            ("grant_id", "text", "NO"),
            ("session_id", "text", "NO"),
            ("interaction_id", "text", "NO"),
            ("request_id", "text", "NO"),
            ("tool_ref", "text", "NO"),
            ("generation_id", "text", "NO"),
            ("tool_id", "text", "NO"),
            ("tool_name", "text", "NO"),
            ("catalogue_revision", "text", "NO"),
            ("descriptor_version", "text", "NO"),
            ("issued_at", "timestamp with time zone", "NO"),
            ("expires_at", "timestamp with time zone", "NO"),
            ("max_calls", "bigint", "NO"),
            ("used_calls", "bigint", "NO"),
            ("revoked_at", "timestamp with time zone", "YES"),
            ("record", "jsonb", "NO"),
        ),
        "cayu_targeted_tool_grant_uses": (
            ("use_id", "text", "NO"),
            ("grant_id", "text", "NO"),
            ("session_id", "text", "NO"),
            ("interaction_id", "text", "NO"),
            ("model_step_id", "text", "NO"),
            ("outer_tool_call_id", "text", "NO"),
            ("arguments_sha256", "text", "NO"),
            ("invocation_id", "text", "NO"),
            ("bound_at", "timestamp with time zone", "NO"),
            ("record", "jsonb", "NO"),
        ),
    }
    for table_name, expected in expected_columns.items():
        await cur.execute(
            """
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = current_schema() AND table_name = %s
                ORDER BY ordinal_position
                """,
            (table_name,),
        )
        if tuple(await cur.fetchall()) != expected:
            _raise_targeted_tool_grant_schema_error(table_name)

    await cur.execute(
        """
            SELECT column_default
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'cayu_targeted_tool_grants'
              AND column_name = 'used_calls'
            """
    )
    default_row = await cur.fetchone()
    if default_row is None or re.sub(r"\s|::bigint", "", str(default_row[0])) != "0":
        _raise_targeted_tool_grant_schema_error("cayu_targeted_tool_grants.used_calls")

    expected_constraints = {
        "cayu_targeted_tool_grants": {
            ("p", "primarykeygrant_id"),
            ("u", "uniquesession_id,interaction_id,request_id"),
            ("u", "uniquesession_id,interaction_id,tool_id"),
            (
                "f",
                "foreignkeysession_idreferencescayu_sessionsidondeletecascade",
            ),
            ("c", "checkmax_calls>=1andmax_calls<=32"),
            ("c", "checkused_calls>=0andused_calls<=max_calls"),
        },
        "cayu_targeted_tool_grant_uses": {
            ("p", "primarykeyuse_id"),
            ("u", "uniquesession_id,interaction_id,invocation_id"),
            ("u", "uniquesession_id,interaction_id,outer_tool_call_id"),
            (
                "f",
                "foreignkeygrant_idreferencescayu_targeted_tool_grantsgrant_idondeletecascade",
            ),
            (
                "f",
                "foreignkeysession_idreferencescayu_sessionsidondeletecascade",
            ),
        },
    }
    for table_name, expected in expected_constraints.items():
        await cur.execute(
            """
                SELECT constraint_record.contype,
                       pg_get_constraintdef(constraint_record.oid)
                FROM pg_catalog.pg_constraint AS constraint_record
                JOIN pg_catalog.pg_class AS table_record
                  ON table_record.oid = constraint_record.conrelid
                JOIN pg_catalog.pg_namespace AS namespace
                  ON namespace.oid = table_record.relnamespace
                WHERE namespace.nspname = current_schema()
                  AND table_record.relname = %s
                """,
            (table_name,),
        )
        actual = {
            (
                str(row[0]),
                re.sub(r'[\s()"]', "", str(row[1]).lower()),
            )
            for row in await cur.fetchall()
        }
        if actual != expected:
            _raise_targeted_tool_grant_schema_error(f"{table_name} constraints")

    required_indexes = {
        "idx_cayu_targeted_tool_grants_interaction": (
            "cayu_targeted_tool_grantsusingbtreesession_id,interaction_id,issued_at,grant_id"
        ),
        "idx_cayu_targeted_tool_grant_uses_grant": (
            "cayu_targeted_tool_grant_usesusingbtreegrant_id,bound_at,use_id"
        ),
    }
    for index_name, required_definition in required_indexes.items():
        await cur.execute(
            """
                SELECT indexdef
                FROM pg_indexes
                WHERE schemaname = current_schema() AND indexname = %s
                """,
            (index_name,),
        )
        index_rows = tuple(await cur.fetchall())
        if len(index_rows) != 1:
            _raise_targeted_tool_grant_schema_error(index_name)
        index_definition = re.sub(r'[\s()"]', "", str(index_rows[0][0]).lower())
        if (
            "createindex" not in index_definition
            or "createuniqueindex" in index_definition
            or required_definition not in index_definition
        ):
            _raise_targeted_tool_grant_schema_error(index_name)


def _raise_targeted_tool_grant_schema_error(name: str) -> NoReturn:
    raise RuntimeError(
        f"Postgres schema object {name!r} conflicts with Cayu's revision-52 "
        "targeted-grant durability contract. Run `cayu storage migrate` or "
        "restore the database from a known-good backup."
    )


async def _validate_public_authority_alias_registry(cur: Any) -> None:
    table_name = "cayu_public_authority_aliases"
    await cur.execute("SELECT to_regclass(%s)", (table_name,))
    registered = await cur.fetchone()
    if registered is None or registered[0] is None:
        raise RuntimeError(
            f"Required Cayu Postgres table is missing: {table_name}. "
            "Run `cayu storage migrate` to restore the public authority index."
        )
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
        ("field_name", "text", "NO"),
        ("scope_session_id", "text", "NO"),
        ("public_alias", "text", "NO"),
        ("private_value", "text", "NO"),
    )
    if columns != expected_columns or primary_keys != (
        "PRIMARY KEY (field_name, scope_session_id, public_alias)",
    ):
        raise RuntimeError(
            f"Postgres schema object {table_name!r} conflicts with Cayu's "
            "public authority alias contract. Run `cayu storage migrate` "
            "after repairing the conflicting object."
        )

    key_table_name = "cayu_public_authority_alias_keys"
    await cur.execute("SELECT to_regclass(%s)", (key_table_name,))
    registered = await cur.fetchone()
    if registered is None or registered[0] is None:
        raise RuntimeError(
            f"Required Cayu Postgres table is missing: {key_table_name}. "
            "Run `cayu storage migrate` to restore the alias key registry."
        )
    await cur.execute(
        """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = %s
            ORDER BY ordinal_position
            """,
        (key_table_name,),
    )
    key_columns = tuple(await cur.fetchall())
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
        (key_table_name,),
    )
    key_primary_keys = tuple(row[0] for row in await cur.fetchall())
    if key_columns != (
        ("key_id", "text", "NO"),
        ("fingerprint", "text", "NO"),
        ("backfill_completed", "boolean", "NO"),
    ) or key_primary_keys != ("PRIMARY KEY (key_id)",):
        raise RuntimeError(
            f"Postgres schema object {key_table_name!r} conflicts with Cayu's "
            "public authority key-state contract. Run `cayu storage migrate` "
            "after repairing the conflicting object."
        )

    config_table_name = "cayu_public_authority_alias_config"
    await cur.execute("SELECT to_regclass(%s)", (config_table_name,))
    registered = await cur.fetchone()
    if registered is None or registered[0] is None:
        raise RuntimeError(
            f"Required Cayu Postgres table is missing: {config_table_name}. "
            "Run `cayu storage migrate` to restore the alias deployment registry."
        )
    await cur.execute(
        """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = %s
            ORDER BY ordinal_position
            """,
        (config_table_name,),
    )
    config_columns = tuple(await cur.fetchall())
    if config_columns != (
        ("singleton", "boolean", "NO"),
        ("active_key_id", "text", "NO"),
        ("keyring_fingerprint", "text", "NO"),
        ("generation", "bigint", "NO"),
        ("retired_key_ids", "jsonb", "NO"),
    ):
        raise RuntimeError(
            f"Postgres schema object {config_table_name!r} conflicts with Cayu's "
            "public authority deployment contract. Run `cayu storage migrate` "
            "after repairing the conflicting object."
        )
