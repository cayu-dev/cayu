"""PostgreSQL schema checks for memory evidence and receipt constraints.

Callers retain revision gates, validation order and the existing cursor/transaction.
These functions inspect the schema without acquiring connections or applying DDL.
"""

from __future__ import annotations

from typing import Any, NoReturn

from cayu.storage import _postgres_catalog as postgres_catalog


async def _validate_memory_evidence_schema(cur: Any) -> None:
    expected_columns = {
        "cayu_recall_receipts": (
            ("receipt_id", "text", "NO"),
            ("session_id", "text", "NO"),
            ("interaction_id", "text", "NO"),
            ("model_step_id", "text", "NO"),
            ("created_at", "timestamp with time zone", "NO"),
            ("receipt_json", "jsonb", "NO"),
            ("document_bytes", "bigint", "NO"),
        ),
        "cayu_context_exposures": (
            ("exposure_id", "text", "NO"),
            ("session_id", "text", "NO"),
            ("interaction_id", "text", "NO"),
            ("model_step_id", "text", "NO"),
            ("model_attempt_id", "text", "NO"),
            ("provider_attempt_id", "text", "NO"),
            ("state", "text", "NO"),
            ("state_revision", "integer", "NO"),
            ("created_at", "timestamp with time zone", "NO"),
            ("updated_at", "timestamp with time zone", "NO"),
            ("exposure_json", "jsonb", "NO"),
            ("document_bytes", "bigint", "NO"),
        ),
        "cayu_recall_item_exposures": (
            ("exposure_id", "text", "NO"),
            ("ordinal", "integer", "NO"),
            ("receipt_id", "text", "NO"),
            ("receipt_item_ordinal", "integer", "NO"),
            ("item_json", "jsonb", "NO"),
            ("document_bytes", "bigint", "NO"),
        ),
    }
    for table, expected in expected_columns.items():
        await cur.execute(
            """
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = current_schema() AND table_name = %s
                ORDER BY ordinal_position
                """,
            (table,),
        )
        if tuple(await cur.fetchall()) != expected:
            _raise_memory_evidence_schema_error(table)

    await cur.execute(
        """
            SELECT table_name, column_name, collation_name
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND (table_name, column_name) IN (
                  ('cayu_recall_receipts', 'receipt_id'),
                  ('cayu_recall_receipts', 'session_id'),
                  ('cayu_recall_receipts', 'interaction_id'),
                  ('cayu_recall_receipts', 'model_step_id'),
                  ('cayu_context_exposures', 'exposure_id'),
                  ('cayu_context_exposures', 'session_id'),
                  ('cayu_context_exposures', 'interaction_id'),
                  ('cayu_context_exposures', 'model_step_id'),
                  ('cayu_context_exposures', 'model_attempt_id'),
                  ('cayu_context_exposures', 'provider_attempt_id'),
                  ('cayu_recall_item_exposures', 'exposure_id'),
                  ('cayu_recall_item_exposures', 'receipt_id')
              )
            """
    )
    if set(await cur.fetchall()) != {
        ("cayu_recall_receipts", "receipt_id", "C"),
        ("cayu_recall_receipts", "session_id", "C"),
        ("cayu_recall_receipts", "interaction_id", "C"),
        ("cayu_recall_receipts", "model_step_id", "C"),
        ("cayu_context_exposures", "exposure_id", "C"),
        ("cayu_context_exposures", "session_id", "C"),
        ("cayu_context_exposures", "interaction_id", "C"),
        ("cayu_context_exposures", "model_step_id", "C"),
        ("cayu_context_exposures", "model_attempt_id", "C"),
        ("cayu_context_exposures", "provider_attempt_id", "C"),
        ("cayu_recall_item_exposures", "exposure_id", "C"),
        ("cayu_recall_item_exposures", "receipt_id", "C"),
    }:
        _raise_memory_evidence_schema_error("memory evidence identity collation")

    required_constraints = {
        "cayu_recall_receipts": (
            ("p", ("primary key (receipt_id)",)),
            (
                "f",
                (
                    "foreign key (session_id)",
                    "references cayu_sessions(id)",
                    "on delete cascade",
                ),
            ),
            ("c", ("document_bytes >= 1", "document_bytes <= 256000")),
        ),
        "cayu_context_exposures": (
            ("p", ("primary key (exposure_id)",)),
            ("u", ("unique (session_id, model_attempt_id)",)),
            ("u", ("unique (session_id, provider_attempt_id)",)),
            (
                "f",
                (
                    "foreign key (session_id)",
                    "references cayu_sessions(id)",
                    "on delete cascade",
                ),
            ),
            (
                "c",
                (
                    "planned",
                    "prepared",
                    "dispatch_started",
                    "acknowledged",
                    "completed",
                    "failed",
                    "cancelled",
                    "indeterminate",
                ),
            ),
            ("c", ("state_revision >= 0", "state_revision < 16")),
            ("c", ("document_bytes >= 1", "document_bytes <= 128000")),
        ),
        "cayu_recall_item_exposures": (
            ("p", ("primary key (exposure_id, ordinal)",)),
            (
                "u",
                ("unique (exposure_id, receipt_id, receipt_item_ordinal)",),
            ),
            (
                "f",
                (
                    "foreign key (exposure_id)",
                    "references cayu_context_exposures(exposure_id)",
                    "on delete cascade",
                ),
            ),
            (
                "f",
                (
                    "foreign key (receipt_id)",
                    "references cayu_recall_receipts(receipt_id)",
                    "on delete cascade",
                ),
            ),
            ("c", ("ordinal >= 0", "ordinal < 64")),
            (
                "c",
                ("receipt_item_ordinal >= 0", "receipt_item_ordinal < 64"),
            ),
            ("c", ("document_bytes >= 1", "document_bytes <= 16384")),
        ),
    }
    for table, required in required_constraints.items():
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
                  AND constraint_record.contype IN ('p', 'u', 'f', 'c')
                """,
            (table,),
        )
        candidates = [
            (str(kind), " ".join(str(definition).lower().split()))
            for kind, definition in await cur.fetchall()
        ]
        if not postgres_catalog._constraint_fragments_match_exactly(candidates, required):
            _raise_memory_evidence_schema_error(table)

    await cur.execute(
        """
            SELECT table_record.relname, index_record.relname
            FROM pg_catalog.pg_index AS index_state
            JOIN pg_catalog.pg_class AS index_record
              ON index_record.oid = index_state.indexrelid
            JOIN pg_catalog.pg_class AS table_record
              ON table_record.oid = index_state.indrelid
            JOIN pg_catalog.pg_namespace AS namespace
              ON namespace.oid = table_record.relnamespace
            LEFT JOIN pg_catalog.pg_constraint AS constraint_record
              ON constraint_record.conindid = index_state.indexrelid
             AND constraint_record.contype IN ('p', 'u')
            WHERE namespace.nspname = current_schema()
              AND table_record.relname = ANY(%s)
              AND index_state.indisunique
              AND constraint_record.oid IS NULL
            ORDER BY table_record.relname, index_record.relname
            LIMIT 1
            """,
        (list(expected_columns),),
    )
    unexpected_unique_index = await cur.fetchone()
    if unexpected_unique_index is not None:
        _raise_memory_evidence_schema_error(str(unexpected_unique_index[1]))

    expected_indexes = {
        "idx_cayu_recall_receipts_session_page": (
            "cayu_recall_receipts",
            "using btree (session_id, created_at, receipt_id)",
        ),
        "idx_cayu_recall_receipts_interaction_page": (
            "cayu_recall_receipts",
            "using btree (session_id, interaction_id, created_at, receipt_id)",
        ),
        "idx_cayu_recall_receipts_step_page": (
            "cayu_recall_receipts",
            "using btree (session_id, model_step_id, created_at, receipt_id)",
        ),
        "idx_cayu_recall_receipts_interaction_step_page": (
            "cayu_recall_receipts",
            "using btree (session_id, interaction_id, model_step_id, created_at, receipt_id)",
        ),
        "idx_cayu_context_exposures_session_page": (
            "cayu_context_exposures",
            "using btree (session_id, created_at, exposure_id)",
        ),
        "idx_cayu_context_exposures_interaction_page": (
            "cayu_context_exposures",
            "using btree (session_id, interaction_id, created_at, exposure_id)",
        ),
        "idx_cayu_context_exposures_step_page": (
            "cayu_context_exposures",
            "using btree (session_id, model_step_id, created_at, exposure_id)",
        ),
        "idx_cayu_context_exposures_interaction_step_page": (
            "cayu_context_exposures",
            "using btree (session_id, interaction_id, model_step_id, created_at, exposure_id)",
        ),
        "idx_cayu_recall_item_exposures_receipt": (
            "cayu_recall_item_exposures",
            "using btree (receipt_id, exposure_id, ordinal)",
        ),
    }
    await cur.execute(
        """
            SELECT table_record.relname, index_record.relname,
                   index_state.indisvalid, index_state.indisready,
                   index_state.indisunique, index_state.indpred IS NULL,
                   index_state.indexprs IS NULL,
                   index_state.indnatts = index_state.indnkeyatts,
                   pg_get_indexdef(index_record.oid)
            FROM pg_catalog.pg_index AS index_state
            JOIN pg_catalog.pg_class AS index_record
              ON index_record.oid = index_state.indexrelid
            JOIN pg_catalog.pg_class AS table_record
              ON table_record.oid = index_state.indrelid
            JOIN pg_catalog.pg_namespace AS namespace
              ON namespace.oid = table_record.relnamespace
            WHERE namespace.nspname = current_schema()
              AND index_record.relname = ANY(%s)
            """,
        (list(expected_indexes),),
    )
    indexes = {
        str(index): (
            str(table),
            bool(valid),
            bool(ready),
            bool(unique),
            bool(unconditional),
            bool(plain_columns),
            bool(key_columns_only),
            " ".join(str(definition).lower().split()),
        )
        for (
            table,
            index,
            valid,
            ready,
            unique,
            unconditional,
            plain_columns,
            key_columns_only,
            definition,
        ) in await cur.fetchall()
    }
    for name, (table, definition_fragment) in expected_indexes.items():
        value = indexes.get(name)
        if (
            value is None
            or value[0] != table
            or not value[1]
            or not value[2]
            or value[3]
            or not value[4]
            or not value[5]
            or not value[6]
            or definition_fragment not in value[7]
        ):
            _raise_memory_evidence_schema_error(name)


def _raise_memory_evidence_schema_error(name: str) -> NoReturn:
    raise RuntimeError(
        f"Postgres schema object {name!r} conflicts with Cayu's revision-51 "
        "memory evidence contract. Recreate the database or restore a known-good "
        "revision-51 backup."
    )
