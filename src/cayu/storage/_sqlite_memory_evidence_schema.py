"""Validate SQLite recall and model-context evidence independently of migrations."""

from __future__ import annotations

import sqlite3
from typing import NoReturn


def _validate_memory_evidence_schema(connection: sqlite3.Connection) -> None:
    expected_columns = {
        "cayu_recall_receipts": (
            ("receipt_id", "TEXT", 0, 1),
            ("session_id", "TEXT", 1, 0),
            ("interaction_id", "TEXT", 1, 0),
            ("model_step_id", "TEXT", 1, 0),
            ("created_at", "TEXT", 1, 0),
            ("receipt_json", "TEXT", 1, 0),
            ("document_bytes", "INTEGER", 1, 0),
        ),
        "cayu_context_exposures": (
            ("exposure_id", "TEXT", 0, 1),
            ("session_id", "TEXT", 1, 0),
            ("interaction_id", "TEXT", 1, 0),
            ("model_step_id", "TEXT", 1, 0),
            ("model_attempt_id", "TEXT", 1, 0),
            ("provider_attempt_id", "TEXT", 1, 0),
            ("state", "TEXT", 1, 0),
            ("state_revision", "INTEGER", 1, 0),
            ("created_at", "TEXT", 1, 0),
            ("updated_at", "TEXT", 1, 0),
            ("exposure_json", "TEXT", 1, 0),
            ("document_bytes", "INTEGER", 1, 0),
        ),
        "cayu_recall_item_exposures": (
            ("exposure_id", "TEXT", 1, 1),
            ("ordinal", "INTEGER", 1, 2),
            ("receipt_id", "TEXT", 1, 0),
            ("receipt_item_ordinal", "INTEGER", 1, 0),
            ("item_json", "TEXT", 1, 0),
            ("document_bytes", "INTEGER", 1, 0),
        ),
    }
    for table, expected in expected_columns.items():
        actual = tuple(
            (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
            for row in connection.execute(f"PRAGMA table_info({table})")
        )
        if actual != expected:
            _raise_memory_evidence_schema_error(table)

    required_table_sql = {
        "cayu_recall_receipts": (
            "check(json_valid(receipt_json))",
            "document_bytes>=1anddocument_bytes<=256000",
        ),
        "cayu_context_exposures": (
            "statein('planned','prepared','dispatch_started','acknowledged',"
            "'completed','failed','cancelled','indeterminate')",
            "state_revision>=0andstate_revision<16",
            "check(json_valid(exposure_json))",
            "document_bytes>=1anddocument_bytes<=128000",
        ),
        "cayu_recall_item_exposures": (
            "ordinal>=0andordinal<64",
            "receipt_item_ordinal>=0andreceipt_item_ordinal<64",
            "check(json_valid(item_json))",
            "document_bytes>=1anddocument_bytes<=16384",
        ),
    }
    for table, fragments in required_table_sql.items():
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        normalized = (
            ""
            if row is None or row[0] is None
            else "".join(str(row[0]).lower().split()).replace("(", "").replace(")", "")
        )
        if "collatenocase" in normalized or any(
            fragment.replace("(", "").replace(")", "") not in normalized for fragment in fragments
        ):
            _raise_memory_evidence_schema_error(table)

    expected_foreign_keys = {
        "cayu_recall_receipts": {
            ("session_id", "cayu_sessions", "id", "CASCADE"),
        },
        "cayu_context_exposures": {
            ("session_id", "cayu_sessions", "id", "CASCADE"),
        },
        "cayu_recall_item_exposures": {
            ("exposure_id", "cayu_context_exposures", "exposure_id", "CASCADE"),
            ("receipt_id", "cayu_recall_receipts", "receipt_id", "CASCADE"),
        },
    }
    for table, expected in expected_foreign_keys.items():
        actual = {
            (str(row[3]), str(row[2]), str(row[4]), str(row[6]).upper())
            for row in connection.execute(f"PRAGMA foreign_key_list({table})")
        }
        if actual != expected:
            _raise_memory_evidence_schema_error(f"{table} foreign keys")

    expected_unique_columns = {
        "cayu_recall_receipts": set(),
        "cayu_context_exposures": {
            ("session_id", "model_attempt_id"),
            ("session_id", "provider_attempt_id"),
        },
        "cayu_recall_item_exposures": {
            ("exposure_id", "receipt_id", "receipt_item_ordinal"),
        },
    }
    for table, expected in expected_unique_columns.items():
        index_rows = tuple(connection.execute(f"PRAGMA index_list({table})"))
        standalone_unique_indexes = tuple(
            str(index[1]) for index in index_rows if bool(index[2]) and str(index[3]) == "c"
        )
        if standalone_unique_indexes:
            _raise_memory_evidence_schema_error(standalone_unique_indexes[0])
        actual = {
            tuple(str(column[2]) for column in connection.execute(f"PRAGMA index_info({index[1]})"))
            for index in index_rows
            if bool(index[2]) and str(index[3]) == "u"
        }
        if actual != expected:
            _raise_memory_evidence_schema_error(f"{table} uniqueness")

    expected_indexes = {
        "idx_cayu_recall_receipts_session_page": (
            "cayu_recall_receipts",
            ("session_id", "created_at", "receipt_id"),
        ),
        "idx_cayu_recall_receipts_interaction_page": (
            "cayu_recall_receipts",
            ("session_id", "interaction_id", "created_at", "receipt_id"),
        ),
        "idx_cayu_recall_receipts_step_page": (
            "cayu_recall_receipts",
            ("session_id", "model_step_id", "created_at", "receipt_id"),
        ),
        "idx_cayu_recall_receipts_interaction_step_page": (
            "cayu_recall_receipts",
            (
                "session_id",
                "interaction_id",
                "model_step_id",
                "created_at",
                "receipt_id",
            ),
        ),
        "idx_cayu_context_exposures_session_page": (
            "cayu_context_exposures",
            ("session_id", "created_at", "exposure_id"),
        ),
        "idx_cayu_context_exposures_interaction_page": (
            "cayu_context_exposures",
            ("session_id", "interaction_id", "created_at", "exposure_id"),
        ),
        "idx_cayu_context_exposures_step_page": (
            "cayu_context_exposures",
            ("session_id", "model_step_id", "created_at", "exposure_id"),
        ),
        "idx_cayu_context_exposures_interaction_step_page": (
            "cayu_context_exposures",
            (
                "session_id",
                "interaction_id",
                "model_step_id",
                "created_at",
                "exposure_id",
            ),
        ),
        "idx_cayu_recall_item_exposures_receipt": (
            "cayu_recall_item_exposures",
            ("receipt_id", "exposure_id", "ordinal"),
        ),
    }
    for index_name, (table, expected_columns_for_index) in expected_indexes.items():
        index_row = connection.execute(
            "SELECT tbl_name FROM sqlite_master WHERE type = 'index' AND name = ?",
            (index_name,),
        ).fetchone()
        index_metadata = next(
            (
                row
                for row in connection.execute(f"PRAGMA index_list({table})")
                if str(row[1]) == index_name
            ),
            None,
        )
        indexed_columns = tuple(
            (str(row[2]), int(row[3]), str(row[4]).upper())
            for row in connection.execute(f"PRAGMA index_xinfo({index_name})")
            if bool(row[5])
        )
        if (
            index_row is None
            or str(index_row[0]) != table
            or index_metadata is None
            or bool(index_metadata[2])
            or str(index_metadata[3]) != "c"
            or bool(index_metadata[4])
            or indexed_columns
            != tuple((column, 0, "BINARY") for column in expected_columns_for_index)
        ):
            _raise_memory_evidence_schema_error(index_name)


def _raise_memory_evidence_schema_error(name: str) -> NoReturn:
    raise RuntimeError(
        f"SQLite schema object {name!r} conflicts with Cayu's revision-51 memory "
        "evidence contract. Recreate the database or restore a known-good "
        "revision-51 backup."
    )
