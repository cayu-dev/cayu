"""PostgreSQL schema checks for task provenance, retries, receipts and handoffs.

Callers retain revision gates, validation order and the existing cursor/transaction.
These functions inspect the schema without acquiring connections or applying DDL.
"""

from __future__ import annotations

from typing import Any


async def _validate_task_invocation_column(cur: Any) -> None:
    await cur.execute(
        """
            SELECT data_type, is_nullable, is_generated
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'cayu_tasks'
              AND column_name = 'invocation'
            """
    )
    if await cur.fetchone() != ("jsonb", "NO", "NEVER"):
        raise RuntimeError(
            "Postgres schema object 'cayu_tasks.invocation' conflicts with "
            "Cayu's required task invocation-provenance contract. Recreate the "
            "Cayu database from a known-good revision-39 schema."
        )


async def _validate_task_closure_guard(cur: Any, *, expected_guard_sql: str) -> None:
    await cur.execute(
        """
            SELECT t.tgtype, t.tgenabled, t.tgqual, t.tgattr::text, t.tgnargs,
                   p.prosrc, p.prosecdef, pn.nspname
            FROM pg_trigger t
            JOIN pg_class c ON c.oid = t.tgrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            JOIN pg_proc p ON p.oid = t.tgfoid
            JOIN pg_namespace pn ON pn.oid = p.pronamespace
            WHERE n.nspname = current_schema() AND c.relname = 'cayu_tasks'
              AND t.tgname = 'cayu_task_closure_admission_guard'
              AND NOT t.tgisinternal
            """
    )
    row = await cur.fetchone()
    expected_source = expected_guard_sql.split("$$")[1]
    await cur.execute("SELECT current_schema()")
    schema_row = await cur.fetchone()
    if (
        row is None
        or row[:5] != (23, "O", None, "", 0)
        or " ".join(row[5].split()) != " ".join(expected_source.split())
        or row[6] is not False
        or schema_row is None
        or row[7] != schema_row[0]
    ):
        raise RuntimeError("Postgres task closure admission guard is missing or conflicting.")


async def _validate_local_execution_attempt_schema(cur: Any) -> None:
    await cur.execute(
        """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'cayu_local_execution_attempts'
            ORDER BY ordinal_position
            """
    )
    if tuple(await cur.fetchall()) != (
        ("attempt_id", "text", "NO"),
        ("task_id", "text", "NO"),
        ("retry_series_id", "text", "YES"),
        ("effect_lineage_id", "text", "NO"),
        ("request_sha256", "text", "NO"),
        ("phase", "text", "NO"),
        ("quiescence", "text", "NO"),
        ("retry_admissible", "boolean", "NO"),
        ("recovery_generation", "bigint", "NO"),
        ("recovery_owner_id", "text", "YES"),
        ("recovery_owner_expires_at", "timestamp with time zone", "YES"),
        ("record_json", "jsonb", "NO"),
        ("created_at", "timestamp with time zone", "NO"),
        ("updated_at", "timestamp with time zone", "NO"),
    ):
        raise RuntimeError("Postgres local execution-attempt storage conflicts with revision 66.")
    await cur.execute(
        """
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE schemaname = current_schema()
              AND indexname = ANY(%s)
            ORDER BY indexname
            """,
        (
            [
                "idx_cayu_local_execution_attempts_discovery",
                "idx_cayu_local_execution_attempts_lineage",
                "idx_cayu_local_execution_attempts_recovery",
                "idx_cayu_local_execution_attempts_task_fence",
            ],
        ),
    )
    indexes = tuple(await cur.fetchall())
    expected_indexes = {
        "idx_cayu_local_execution_attempts_discovery": (
            "created_at",
            "attempt_id",
        ),
        "idx_cayu_local_execution_attempts_lineage": (
            "retry_series_id",
            "task_id",
            "effect_lineage_id",
            "created_at desc",
            "attempt_id desc",
        ),
        "idx_cayu_local_execution_attempts_recovery": (
            "retry_admissible",
            "phase",
            "updated_at",
            "attempt_id",
        ),
        "idx_cayu_local_execution_attempts_task_fence": (
            "task_id",
            "retry_admissible",
            "created_at",
            "attempt_id",
        ),
    }
    if tuple(row[0] for row in indexes) != (
        "idx_cayu_local_execution_attempts_discovery",
        "idx_cayu_local_execution_attempts_lineage",
        "idx_cayu_local_execution_attempts_recovery",
        "idx_cayu_local_execution_attempts_task_fence",
    ) or any(
        not all(
            expected_fragment in " ".join(str(index_definition).lower().split())
            for expected_fragment in expected_indexes[str(index_name)]
        )
        for index_name, index_definition in indexes
    ):
        raise RuntimeError("Postgres local execution-attempt indexes conflict with revision 66.")
    await cur.execute(
        """
            SELECT constraint_definition.contype,
                   pg_get_constraintdef(constraint_definition.oid, TRUE)
            FROM pg_constraint AS constraint_definition
            JOIN pg_class AS relation
              ON relation.oid = constraint_definition.conrelid
            JOIN pg_namespace AS namespace
              ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = current_schema()
              AND relation.relname = 'cayu_local_execution_attempts'
            """
    )
    constraints = tuple(await cur.fetchall())
    normalized_constraints = tuple(
        (str(kind), " ".join(str(definition).lower().split())) for kind, definition in constraints
    )
    required_constraints = (
        ("p", "primary key (attempt_id)"),
        ("f", "foreign key (task_id) references cayu_tasks(id) on delete restrict"),
        ("u", "unique (task_id, effect_lineage_id, attempt_id)"),
        ("c", "recovery_generation >= 0"),
        ("c", "phase"),
        ("c", "quiescence"),
    )
    if any(
        not any(
            kind == expected_kind and fragment in definition
            for kind, definition in normalized_constraints
        )
        for expected_kind, fragment in required_constraints
    ):
        raise RuntimeError(
            "Postgres local execution-attempt constraints conflict with revision 66."
        )


async def _validate_task_retry_series_schema(cur: Any) -> None:
    await cur.execute(
        """
            SELECT data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'cayu_tasks'
              AND column_name = 'retry_series'
            """
    )
    task_column = await cur.fetchone()
    await cur.execute(
        """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'cayu_task_retry_settlements'
            ORDER BY ordinal_position
            """
    )
    receipt_columns = tuple(await cur.fetchall())
    await cur.execute(
        """
            SELECT pg_get_constraintdef(constraint_record.oid)
            FROM pg_catalog.pg_constraint AS constraint_record
            JOIN pg_catalog.pg_class AS table_record
              ON table_record.oid = constraint_record.conrelid
            JOIN pg_catalog.pg_namespace AS namespace
              ON namespace.oid = table_record.relnamespace
            WHERE namespace.nspname = current_schema()
              AND table_record.relname = 'cayu_task_retry_settlements'
              AND constraint_record.contype = 'p'
            """
    )
    primary_keys = tuple(row[0] for row in await cur.fetchall())
    expected = (
        ("task_id", "text", "NO"),
        ("idempotency_key", "text", "NO"),
        ("request_sha256", "text", "NO"),
        ("receipt_json", "jsonb", "NO"),
        ("committed_at", "timestamp with time zone", "NO"),
    )
    if (
        task_column != ("jsonb", "YES")
        or receipt_columns != expected
        or primary_keys != ("PRIMARY KEY (task_id, idempotency_key)",)
    ):
        raise RuntimeError(
            "Postgres task retry-series schema conflicts with Cayu's revision-45 "
            "durability contract. Run `cayu storage migrate` or restore the "
            "database from a known-good backup."
        )


async def _validate_task_retry_reconciliation_schema(cur: Any) -> None:
    await cur.execute(
        """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'cayu_task_retry_reconciliation_rejections'
            ORDER BY ordinal_position
            """
    )
    rejection_columns = tuple(await cur.fetchall())
    await cur.execute(
        """
            SELECT pg_get_constraintdef(constraint_record.oid)
            FROM pg_catalog.pg_constraint AS constraint_record
            JOIN pg_catalog.pg_class AS table_record
              ON table_record.oid = constraint_record.conrelid
            JOIN pg_catalog.pg_namespace AS namespace
              ON namespace.oid = table_record.relnamespace
            WHERE namespace.nspname = current_schema()
              AND table_record.relname = 'cayu_task_retry_reconciliation_rejections'
              AND constraint_record.contype = 'p'
            """
    )
    primary_keys = tuple(row[0] for row in await cur.fetchall())
    if rejection_columns != (
        ("task_id", "text", "NO"),
        ("reconciliation_idempotency_key", "text", "NO"),
        ("request_sha256", "text", "NO"),
        ("record_json", "jsonb", "NO"),
        ("recorded_at", "timestamp with time zone", "NO"),
    ) or primary_keys != ("PRIMARY KEY (task_id, reconciliation_idempotency_key)",):
        raise RuntimeError(
            "Postgres task retry-reconciliation schema conflicts with Cayu's "
            "revision-55 durability contract. Run `cayu storage migrate` or "
            "restore the database from a known-good backup."
        )


async def _validate_task_terminalization_receipt_table(cur: Any) -> None:
    await cur.execute(
        """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'cayu_task_terminalization_receipts'
            ORDER BY ordinal_position
            """
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
              AND table_record.relname = 'cayu_task_terminalization_receipts'
              AND constraint_record.contype = 'p'
            """
    )
    primary_keys = tuple(row[0] for row in await cur.fetchall())
    expected = (
        ("task_id", "text", "NO"),
        ("idempotency_key", "text", "NO"),
        ("request_sha256", "text", "NO"),
        ("worker_id", "text", "NO"),
        ("terminal_kind", "text", "NO"),
        ("task_json", "jsonb", "NO"),
        ("committed_at", "timestamp with time zone", "NO"),
    )
    if columns != expected or primary_keys != ("PRIMARY KEY (task_id, idempotency_key)",):
        raise RuntimeError(
            "Postgres task terminalization receipt table conflicts with Cayu's "
            "revision-38 durability contract. Run `cayu storage migrate` or restore "
            "the database from a known-good backup."
        )


async def _validate_interrupted_task_handoff_schema(cur: Any) -> None:
    await cur.execute(
        """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'cayu_task_interrupted_handoff_receipts'
            ORDER BY ordinal_position
            """
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
              AND table_record.relname = 'cayu_task_interrupted_handoff_receipts'
              AND constraint_record.contype = 'p'
            """
    )
    primary_keys = tuple(row[0] for row in await cur.fetchall())
    await cur.execute(
        """
            SELECT ARRAY(
                SELECT pg_get_indexdef(index_record.indexrelid, key_position, FALSE)
                FROM generate_series(1, index_record.indnkeyatts) AS key_position
                ORDER BY key_position
            )
            FROM pg_catalog.pg_index AS index_record
            JOIN pg_catalog.pg_class AS index_class
              ON index_class.oid = index_record.indexrelid
            JOIN pg_catalog.pg_namespace AS namespace
              ON namespace.oid = index_class.relnamespace
            WHERE namespace.nspname = current_schema()
              AND index_class.relname = 'idx_cayu_tasks_interrupted_handoff_recovery'
              AND index_record.indisvalid
            """
    )
    recovery_indexes = tuple(tuple(row[0]) for row in await cur.fetchall())
    expected = (
        ("task_id", "text", "NO"),
        ("handoff_id", "text", "NO"),
        ("request_sha256", "text", "NO"),
        ("request_json", "jsonb", "NO"),
        ("task_json", "jsonb", "NO"),
        ("committed_at", "timestamp with time zone", "NO"),
    )
    if (
        columns != expected
        or primary_keys != ("PRIMARY KEY (task_id, handoff_id)",)
        or recovery_indexes != (("status", "lease_expires_at", "id"),)
    ):
        raise RuntimeError(
            "Postgres interrupted-task handoff receipt table conflicts with "
            "Cayu's revision-70 durability contract. Run `cayu storage migrate` "
            "or restore the database from a known-good backup."
        )


async def _validate_interrupted_handoff_generation_column(cur: Any) -> None:
    await cur.execute(
        """
            SELECT data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'cayu_tasks'
              AND column_name = 'interrupted_handoff_id'
            """
    )
    row = await cur.fetchone()
    if row != ("text", "YES"):
        raise RuntimeError(
            "Postgres task handoff generation storage conflicts with Cayu's "
            "revision-76 durability contract. Run `cayu storage migrate` or "
            "restore the database from a known-good backup."
        )
    claim_table_name = "cayu_task_interrupted_continuation_claims"
    await cur.execute(
        """
            SELECT column_name, data_type, is_nullable, collation_name
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = %s
            ORDER BY ordinal_position
            """,
        (claim_table_name,),
    )
    claim_columns = tuple(await cur.fetchall())
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
              AND constraint_record.contype IN ('p', 'f')
            ORDER BY constraint_record.contype,
                     pg_get_constraintdef(constraint_record.oid)
            """,
        (claim_table_name,),
    )
    claim_ownership_constraints = tuple(await cur.fetchall())
    expected_claim_columns = (
        ("handoff_id_sha256", "text", "NO", "C"),
        ("task_id", "text", "NO", "C"),
        ("worker_id", "text", "NO", "C"),
        ("claimed_at", "timestamp with time zone", "NO", None),
    )
    if claim_columns != expected_claim_columns or claim_ownership_constraints != (
        ("p", "PRIMARY KEY (handoff_id_sha256)"),
    ):
        raise RuntimeError(
            "Postgres continuation-claim generation registry conflicts with Cayu's "
            "revision-76 durability contract. Run `cayu storage migrate` or restore "
            "the database from a known-good backup."
        )
