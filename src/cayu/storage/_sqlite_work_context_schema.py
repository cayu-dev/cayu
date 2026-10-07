"""SQLite schema integrity for agent work context, recall delivery and subscriptions.

Revision gates and migration/reset sequencing remain with schema support.
These checks inspect existing schema using the shared SQLite catalog readers.
"""

from __future__ import annotations

import sqlite3
from typing import NoReturn

from cayu.storage import _sqlite_catalog as sqlite_catalog


def _validate_revision_69_work_context_schema(connection: sqlite3.Connection) -> None:
    expected_columns = {
        "cayu_agent_work_context_revisions": (
            ("task_id", "TEXT", 1, 1),
            ("revision", "INTEGER", 1, 2),
            ("content_sha256", "TEXT", 1, 0),
            ("operation_id", "TEXT", 1, 0),
            ("record_json", "TEXT", 1, 0),
            ("published_at", "TEXT", 1, 0),
        ),
        "cayu_agent_work_context_heads": (
            ("task_id", "TEXT", 1, 1),
            ("current_revision", "INTEGER", 1, 0),
        ),
        "cayu_agent_work_context_publications": (
            ("operation_id", "TEXT", 1, 1),
            ("task_id", "TEXT", 1, 0),
            ("request_sha256", "TEXT", 1, 0),
            ("context_revision", "INTEGER", 1, 0),
            ("changed", "INTEGER", 1, 0),
            ("receipt_json", "TEXT", 1, 0),
            ("committed_at", "TEXT", 1, 0),
        ),
        "cayu_agent_recall_checkpoints": (
            ("agent_id", "TEXT", 1, 1),
            ("task_id", "TEXT", 1, 2),
            ("knowledge_namespace", "TEXT", 1, 3),
            ("access_policy_sha256", "TEXT", 1, 4),
            ("checkpoint_stream_id", "TEXT", 1, 5),
            ("revision", "INTEGER", 1, 6),
            ("work_context_revision", "INTEGER", 1, 0),
            ("work_context_sha256", "TEXT", 1, 0),
            ("knowledge_sequence", "INTEGER", 1, 0),
            ("index_readiness_sequence", "INTEGER", 1, 0),
            ("knowledge_high_water_sequence", "INTEGER", 1, 0),
            ("index_readiness_high_water_sequence", "INTEGER", 1, 0),
            ("processing_mode", "TEXT", 1, 0),
            ("processing_id", "TEXT", 1, 0),
            ("operation_id", "TEXT", 1, 0),
            ("record_json", "TEXT", 1, 0),
            ("updated_at", "TEXT", 1, 0),
        ),
        "cayu_agent_recall_checkpoint_heads": (
            ("agent_id", "TEXT", 1, 1),
            ("task_id", "TEXT", 1, 2),
            ("knowledge_namespace", "TEXT", 1, 3),
            ("access_policy_sha256", "TEXT", 1, 4),
            ("checkpoint_stream_id", "TEXT", 1, 5),
            ("current_revision", "INTEGER", 1, 0),
        ),
    }
    for table, expected in expected_columns.items():
        actual = tuple(
            (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
            for row in connection.execute(f"PRAGMA table_info({table})")
        )
        if actual != expected:
            _raise_revision_69_sqlite_schema_error(table)

    unique_keys = (
        ("cayu_agent_work_context_revisions", ("task_id", "revision")),
        ("cayu_agent_work_context_revisions", ("operation_id",)),
        ("cayu_agent_work_context_heads", ("task_id",)),
        ("cayu_agent_work_context_publications", ("operation_id",)),
        (
            "cayu_agent_recall_checkpoints",
            (
                "agent_id",
                "task_id",
                "knowledge_namespace",
                "access_policy_sha256",
                "checkpoint_stream_id",
                "revision",
            ),
        ),
        ("cayu_agent_recall_checkpoints", ("operation_id",)),
        (
            "cayu_agent_recall_checkpoint_heads",
            (
                "agent_id",
                "task_id",
                "knowledge_namespace",
                "access_policy_sha256",
                "checkpoint_stream_id",
            ),
        ),
    )
    for table, columns in unique_keys:
        if not sqlite_catalog._sqlite_has_unique_index(connection, table, columns):
            _raise_revision_69_sqlite_schema_error(table)

    required_foreign_keys = {
        "cayu_agent_work_context_heads": {
            (
                "cayu_agent_work_context_revisions",
                ("task_id", "current_revision"),
                ("task_id", "revision"),
                "RESTRICT",
            ),
        },
        "cayu_agent_work_context_publications": {
            (
                "cayu_agent_work_context_revisions",
                ("task_id", "context_revision"),
                ("task_id", "revision"),
                "RESTRICT",
            ),
        },
        "cayu_agent_recall_checkpoints": {
            (
                "cayu_agent_work_context_revisions",
                ("task_id", "work_context_revision"),
                ("task_id", "revision"),
                "RESTRICT",
            ),
        },
        "cayu_agent_recall_checkpoint_heads": {
            (
                "cayu_agent_recall_checkpoints",
                (
                    "agent_id",
                    "task_id",
                    "knowledge_namespace",
                    "access_policy_sha256",
                    "checkpoint_stream_id",
                    "current_revision",
                ),
                (
                    "agent_id",
                    "task_id",
                    "knowledge_namespace",
                    "access_policy_sha256",
                    "checkpoint_stream_id",
                    "revision",
                ),
                "RESTRICT",
            ),
        },
    }
    for table, expected in required_foreign_keys.items():
        actual = sqlite_catalog._sqlite_foreign_key_groups(connection, table)
        if actual != expected:
            _raise_revision_69_sqlite_schema_error(table)

    required_sql = {
        "cayu_agent_work_context_revisions": (
            "task_id text collate binary not null",
            "revision integer not null check ( revision > 0 and revision <= 2147483647 )",
            "content_sha256 text collate binary not null",
            "operation_id text collate binary not null unique",
            "length(content_sha256) = 64",
            "content_sha256 not glob '*[^0-9a-f]*'",
            "json_valid(record_json)",
            "json_type(record_json) = 'object'",
        ),
        "cayu_agent_work_context_heads": (
            "task_id text collate binary not null primary key",
            "current_revision > 0",
            "current_revision <= 2147483647",
        ),
        "cayu_agent_work_context_publications": (
            "operation_id text collate binary not null primary key",
            "task_id text collate binary not null",
            "request_sha256 text collate binary not null",
            "length(request_sha256) = 64",
            "request_sha256 not glob '*[^0-9a-f]*'",
            "context_revision > 0",
            "context_revision <= 2147483647",
            "changed in (0, 1)",
            "json_valid(receipt_json)",
            "json_type(receipt_json) = 'object'",
        ),
        "cayu_agent_recall_checkpoints": (
            "agent_id text collate binary not null",
            "task_id text collate binary not null",
            "knowledge_namespace text collate binary not null",
            "access_policy_sha256 text collate binary not null",
            "checkpoint_stream_id text collate binary not null",
            "revision integer not null check ( revision > 0 and revision <= 2147483647 )",
            "work_context_sha256 text collate binary not null",
            "processing_mode text collate binary not null",
            "processing_id text collate binary not null",
            "operation_id text collate binary not null unique",
            "length(access_policy_sha256) = 64",
            "access_policy_sha256 not glob '*[^0-9a-f]*'",
            "work_context_revision > 0",
            "work_context_revision <= 2147483647",
            "length(work_context_sha256) = 64",
            "work_context_sha256 not glob '*[^0-9a-f]*'",
            "knowledge_sequence >= 0",
            "knowledge_sequence <= 9223372036854775807",
            "index_readiness_sequence >= 0",
            "index_readiness_sequence <= 9223372036854775807",
            "knowledge_high_water_sequence >= 0",
            "knowledge_high_water_sequence <= 9223372036854775807",
            "index_readiness_high_water_sequence >= 0",
            "index_readiness_high_water_sequence <= 9223372036854775807",
            "knowledge_sequence <= knowledge_high_water_sequence",
            "index_readiness_sequence <= index_readiness_high_water_sequence",
            "processing_mode in ('full_index', 'delta')",
            "json_valid(record_json)",
            "json_type(record_json) = 'object'",
        ),
        "cayu_agent_recall_checkpoint_heads": (
            "agent_id text collate binary not null",
            "task_id text collate binary not null",
            "knowledge_namespace text collate binary not null",
            "access_policy_sha256 text collate binary not null",
            "checkpoint_stream_id text collate binary not null",
            "current_revision > 0",
            "current_revision <= 2147483647",
        ),
    }
    for table, fragments in required_sql.items():
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        normalized = sqlite_catalog._normalize_sqlite_schema_sql(None if row is None else row[0])
        if any(fragment not in normalized for fragment in fragments):
            _raise_revision_69_sqlite_schema_error(table)


def _raise_revision_69_sqlite_schema_error(name: str) -> NoReturn:
    raise RuntimeError(
        "SQLite schema object "
        f"{name!r} conflicts with Cayu's agent work-context/checkpoint contract. "
        "Run schema_mode=MIGRATE to install the additive revision or recreate the database."
    )


def _validate_revision_71_recall_delivery_schema(
    connection: sqlite3.Connection,
    *,
    require_processing_schema_version: bool = False,
) -> None:
    delivery_columns = (
        ("delivery_id", "TEXT", 1, 1),
        ("operation_id", "TEXT", 1, 0),
        ("agent_id", "TEXT", 1, 0),
        ("task_id", "TEXT", 1, 0),
        ("knowledge_namespace", "TEXT", 1, 0),
        ("access_policy_sha256", "TEXT", 1, 0),
        ("checkpoint_stream_id", "TEXT", 1, 0),
        ("checkpoint_revision", "INTEGER", 1, 0),
        ("processing_result_sha256", "TEXT", 1, 0),
        ("delivery_json", "TEXT", 1, 0),
        ("staged_at", "TEXT", 1, 0),
    )
    delivery_columns_with_processing_schema = (
        *delivery_columns,
        ("processing_schema_version", "TEXT", 1, 0),
    )
    expected_columns = {
        "cayu_agent_recall_deliveries": (
            delivery_columns_with_processing_schema
            if require_processing_schema_version
            else delivery_columns
        ),
        "cayu_agent_recall_delivery_states": (
            ("delivery_id", "TEXT", 1, 1),
            ("agent_id", "TEXT", 1, 0),
            ("task_id", "TEXT", 1, 0),
            ("knowledge_namespace", "TEXT", 1, 0),
            ("access_policy_sha256", "TEXT", 1, 0),
            ("checkpoint_stream_id", "TEXT", 1, 0),
            ("checkpoint_revision", "INTEGER", 1, 0),
            ("state", "TEXT", 1, 0),
            ("attempt", "INTEGER", 1, 0),
            ("state_revision", "INTEGER", 1, 0),
            ("lease_expires_at", "TEXT", 0, 0),
            ("release_id", "TEXT", 0, 0),
            ("acknowledgement_id", "TEXT", 0, 0),
            ("state_json", "TEXT", 1, 0),
            ("updated_at", "TEXT", 1, 0),
        ),
        "cayu_agent_recall_delivery_claims": (
            ("claim_id", "TEXT", 1, 1),
            ("delivery_id", "TEXT", 1, 0),
            ("worker_id", "TEXT", 1, 0),
            ("request_sha256", "TEXT", 1, 0),
            ("attempt", "INTEGER", 1, 0),
            ("claimed_at", "TEXT", 1, 0),
        ),
        "cayu_agent_recall_delivery_releases": (
            ("release_id", "TEXT", 1, 1),
            ("delivery_id", "TEXT", 1, 0),
            ("claim_id", "TEXT", 1, 0),
            ("request_sha256", "TEXT", 1, 0),
            ("release_json", "TEXT", 1, 0),
            ("released_at", "TEXT", 1, 0),
        ),
    }
    validate_processing_schema_version = require_processing_schema_version
    for table, expected in expected_columns.items():
        actual = tuple(
            (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
            for row in connection.execute(f"PRAGMA table_info({table})")
        )
        if (
            table == "cayu_agent_recall_deliveries"
            and not require_processing_schema_version
            and actual == delivery_columns_with_processing_schema
        ):
            validate_processing_schema_version = True
            continue
        if actual != expected:
            _raise_revision_71_sqlite_schema_error(table)

    unique_keys = (
        ("cayu_agent_recall_deliveries", ("delivery_id",)),
        ("cayu_agent_recall_deliveries", ("operation_id",)),
        (
            "cayu_agent_recall_deliveries",
            (
                "agent_id",
                "task_id",
                "knowledge_namespace",
                "access_policy_sha256",
                "checkpoint_stream_id",
                "checkpoint_revision",
            ),
        ),
        ("cayu_agent_recall_delivery_states", ("delivery_id",)),
        ("cayu_agent_recall_delivery_states", ("release_id",)),
        ("cayu_agent_recall_delivery_states", ("acknowledgement_id",)),
        ("cayu_agent_recall_delivery_claims", ("claim_id",)),
        ("cayu_agent_recall_delivery_claims", ("delivery_id", "attempt")),
        ("cayu_agent_recall_delivery_releases", ("release_id",)),
    )
    for table, columns in unique_keys:
        if not sqlite_catalog._sqlite_has_unique_index(connection, table, columns):
            _raise_revision_71_sqlite_schema_error(table)

    required_foreign_keys = {
        "cayu_agent_recall_deliveries": {
            (
                "cayu_agent_recall_checkpoints",
                (
                    "agent_id",
                    "task_id",
                    "knowledge_namespace",
                    "access_policy_sha256",
                    "checkpoint_stream_id",
                    "checkpoint_revision",
                ),
                (
                    "agent_id",
                    "task_id",
                    "knowledge_namespace",
                    "access_policy_sha256",
                    "checkpoint_stream_id",
                    "revision",
                ),
                "RESTRICT",
            ),
            (
                "cayu_agent_recall_checkpoints",
                ("operation_id",),
                ("operation_id",),
                "RESTRICT",
            ),
        },
        "cayu_agent_recall_delivery_states": {
            (
                "cayu_agent_recall_deliveries",
                ("delivery_id",),
                ("delivery_id",),
                "RESTRICT",
            ),
            (
                "cayu_agent_recall_delivery_releases",
                ("release_id",),
                ("release_id",),
                "RESTRICT",
            ),
            (
                "cayu_agent_recall_deliveries",
                (
                    "agent_id",
                    "task_id",
                    "knowledge_namespace",
                    "access_policy_sha256",
                    "checkpoint_stream_id",
                    "checkpoint_revision",
                ),
                (
                    "agent_id",
                    "task_id",
                    "knowledge_namespace",
                    "access_policy_sha256",
                    "checkpoint_stream_id",
                    "checkpoint_revision",
                ),
                "RESTRICT",
            ),
        },
        "cayu_agent_recall_delivery_claims": {
            (
                "cayu_agent_recall_deliveries",
                ("delivery_id",),
                ("delivery_id",),
                "RESTRICT",
            ),
        },
        "cayu_agent_recall_delivery_releases": {
            (
                "cayu_agent_recall_deliveries",
                ("delivery_id",),
                ("delivery_id",),
                "RESTRICT",
            ),
            (
                "cayu_agent_recall_delivery_claims",
                ("claim_id",),
                ("claim_id",),
                "RESTRICT",
            ),
        },
    }
    for table, expected in required_foreign_keys.items():
        if sqlite_catalog._sqlite_foreign_key_groups(connection, table) != expected:
            _raise_revision_71_sqlite_schema_error(table)

    required_sql = {
        "cayu_agent_recall_deliveries": (
            "delivery_id text collate binary not null primary key",
            "operation_id text collate binary not null unique",
            "checkpoint_revision > 0",
            "checkpoint_stream_id text collate binary not null",
            "checkpoint_revision <= 2147483647",
            "length(access_policy_sha256) = 64",
            "access_policy_sha256 not glob '*[^0-9a-f]*'",
            "length(processing_result_sha256) = 64",
            "processing_result_sha256 not glob '*[^0-9a-f]*'",
            "json_valid(delivery_json)",
            "json_type(delivery_json) = 'object'",
        ),
        "cayu_agent_recall_delivery_states": (
            "delivery_id text collate binary not null primary key",
            "checkpoint_stream_id text collate binary not null",
            "state in ('pending', 'claimed', 'acknowledged')",
            "attempt >= 0",
            "attempt <= 9223372036854775807",
            "state_revision >= 0",
            "state_revision <= 9223372036854775807",
            "state = 'pending'",
            "state = 'claimed'",
            "state = 'acknowledged'",
            "json_valid(state_json)",
            "json_type(state_json) = 'object'",
        ),
        "cayu_agent_recall_delivery_claims": (
            "claim_id text collate binary not null primary key",
            "attempt > 0",
            "attempt <= 9223372036854775807",
            "length(request_sha256) = 64",
            "request_sha256 not glob '*[^0-9a-f]*'",
        ),
        "cayu_agent_recall_delivery_releases": (
            "release_id text collate binary not null primary key",
            "length(request_sha256) = 64",
            "request_sha256 not glob '*[^0-9a-f]*'",
            "json_valid(release_json)",
            "json_type(release_json) = 'object'",
        ),
    }
    if validate_processing_schema_version:
        required_sql["cayu_agent_recall_deliveries"] += (
            "processing_schema_version text collate binary not null",
            "processing_schema_version = 'cayu.agent_recall_processing.v3'",
        )
    for table, fragments in required_sql.items():
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        normalized = sqlite_catalog._normalize_sqlite_schema_sql(None if row is None else row[0])
        if any(fragment not in normalized for fragment in fragments):
            _raise_revision_71_sqlite_schema_error(table)

    index = "idx_cayu_agent_recall_delivery_pending"
    row = connection.execute(
        "SELECT tbl_name, sql FROM sqlite_master WHERE type = 'index' AND name = ?",
        (index,),
    ).fetchone()
    columns = tuple(
        str(index_row[2]) for index_row in connection.execute(f"PRAGMA index_info({index})")
    )
    if (
        row is None
        or row[0] != "cayu_agent_recall_delivery_states"
        or columns
        != (
            "agent_id",
            "task_id",
            "knowledge_namespace",
            "access_policy_sha256",
            "checkpoint_stream_id",
            "checkpoint_revision",
            "delivery_id",
        )
        or "where state != 'acknowledged'"
        not in sqlite_catalog._normalize_sqlite_schema_sql(row[1])
    ):
        _raise_revision_71_sqlite_schema_error(index)


def _raise_revision_71_sqlite_schema_error(name: str) -> NoReturn:
    raise RuntimeError(
        "SQLite schema object "
        f"{name!r} conflicts with Cayu's staged recall-delivery contract. "
        "Run schema_mode=MIGRATE to install the breaking revision or recreate the database."
    )


def _validate_revision_73_recall_subscription_schema(
    connection: sqlite3.Connection,
) -> None:
    expected_columns = {
        "cayu_agent_recall_subscription_revisions": (
            ("subscription_id", "TEXT", 1, 1),
            ("revision", "INTEGER", 1, 2),
            ("operation_id", "TEXT", 1, 0),
            ("agent_id", "TEXT", 1, 0),
            ("task_id", "TEXT", 1, 0),
            ("knowledge_namespace", "TEXT", 1, 0),
            ("access_policy_sha256", "TEXT", 1, 0),
            ("work_context_revision", "INTEGER", 1, 0),
            ("work_context_sha256", "TEXT", 1, 0),
            ("status", "TEXT", 1, 0),
            ("priority", "INTEGER", 1, 0),
            ("subscription_json", "TEXT", 1, 0),
            ("expires_at", "TEXT", 1, 0),
            ("published_at", "TEXT", 1, 0),
        ),
        "cayu_agent_recall_subscription_heads": (
            ("subscription_id", "TEXT", 1, 1),
            ("current_revision", "INTEGER", 1, 0),
        ),
        "cayu_agent_recall_subscription_publications": (
            ("operation_id", "TEXT", 1, 1),
            ("subscription_id", "TEXT", 1, 0),
            ("subscription_revision", "INTEGER", 1, 0),
            ("request_sha256", "TEXT", 1, 0),
            ("receipt_json", "TEXT", 1, 0),
            ("committed_at", "TEXT", 1, 0),
        ),
        "cayu_agent_recall_subscription_states": (
            ("subscription_id", "TEXT", 1, 1),
            ("current_revision", "INTEGER", 1, 0),
            ("agent_id", "TEXT", 1, 0),
            ("task_id", "TEXT", 1, 0),
            ("knowledge_namespace", "TEXT", 1, 0),
            ("access_policy_sha256", "TEXT", 1, 0),
            ("run_state", "TEXT", 1, 0),
            ("attempt", "INTEGER", 1, 0),
            ("state_revision", "INTEGER", 1, 0),
            ("lease_expires_at", "TEXT", 0, 0),
            ("release_id", "TEXT", 0, 0),
            ("next_evaluation_at", "TEXT", 1, 0),
            ("last_evaluation_id", "TEXT", 0, 0),
            ("state_json", "TEXT", 1, 0),
            ("updated_at", "TEXT", 1, 0),
        ),
        "cayu_agent_recall_subscription_claims": (
            ("claim_id", "TEXT", 1, 1),
            ("subscription_id", "TEXT", 1, 0),
            ("subscription_revision", "INTEGER", 1, 0),
            ("runner_id", "TEXT", 1, 0),
            ("request_sha256", "TEXT", 1, 0),
            ("attempt", "INTEGER", 1, 0),
            ("claimed_at", "TEXT", 1, 0),
        ),
        "cayu_agent_recall_subscription_releases": (
            ("release_id", "TEXT", 1, 1),
            ("subscription_id", "TEXT", 1, 0),
            ("claim_id", "TEXT", 1, 0),
            ("request_sha256", "TEXT", 1, 0),
            ("release_json", "TEXT", 1, 0),
            ("released_at", "TEXT", 1, 0),
        ),
        "cayu_agent_recall_subscription_evaluations": (
            ("evaluation_id", "TEXT", 1, 1),
            ("subscription_id", "TEXT", 1, 0),
            ("subscription_revision", "INTEGER", 1, 0),
            ("agent_id", "TEXT", 1, 0),
            ("task_id", "TEXT", 1, 0),
            ("knowledge_namespace", "TEXT", 1, 0),
            ("access_policy_sha256", "TEXT", 1, 0),
            ("claim_id", "TEXT", 1, 0),
            ("processing_operation_id", "TEXT", 1, 0),
            ("request_sha256", "TEXT", 1, 0),
            ("outcome", "TEXT", 1, 0),
            ("delivery_id", "TEXT", 0, 0),
            ("evaluation_json", "TEXT", 1, 0),
            ("committed_at", "TEXT", 1, 0),
        ),
        "cayu_agent_recall_subscription_wake_claims": (
            ("claim_id", "TEXT", 1, 1),
            ("wake_id", "TEXT", 1, 0),
            ("delivery_id", "TEXT", 1, 0),
            ("runner_id", "TEXT", 1, 0),
            ("request_sha256", "TEXT", 1, 0),
            ("attempt", "INTEGER", 1, 0),
            ("claimed_at", "TEXT", 1, 0),
        ),
        "cayu_agent_recall_subscription_wake_releases": (
            ("release_id", "TEXT", 1, 1),
            ("wake_id", "TEXT", 1, 0),
            ("claim_id", "TEXT", 1, 0),
            ("request_sha256", "TEXT", 1, 0),
            ("release_json", "TEXT", 1, 0),
            ("released_at", "TEXT", 1, 0),
        ),
        "cayu_agent_recall_subscription_wake_states": (
            ("wake_id", "TEXT", 1, 1),
            ("delivery_id", "TEXT", 1, 0),
            ("agent_id", "TEXT", 1, 0),
            ("task_id", "TEXT", 1, 0),
            ("knowledge_namespace", "TEXT", 1, 0),
            ("access_policy_sha256", "TEXT", 1, 0),
            ("state", "TEXT", 1, 0),
            ("attempt", "INTEGER", 1, 0),
            ("state_revision", "INTEGER", 1, 0),
            ("claim_id", "TEXT", 0, 0),
            ("lease_expires_at", "TEXT", 0, 0),
            ("release_id", "TEXT", 0, 0),
            ("acknowledgement_id", "TEXT", 0, 0),
            ("state_json", "TEXT", 1, 0),
            ("committed_at", "TEXT", 1, 0),
            ("updated_at", "TEXT", 1, 0),
        ),
    }
    for table, expected in expected_columns.items():
        actual = tuple(
            (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
            for row in connection.execute(f"PRAGMA table_info({table})")
        )
        if actual != expected:
            _raise_revision_73_sqlite_schema_error(table)

    unique_keys = (
        ("cayu_agent_recall_subscription_revisions", ("subscription_id", "revision")),
        ("cayu_agent_recall_subscription_revisions", ("operation_id",)),
        ("cayu_agent_recall_subscription_heads", ("subscription_id",)),
        ("cayu_agent_recall_subscription_publications", ("operation_id",)),
        ("cayu_agent_recall_subscription_states", ("subscription_id",)),
        ("cayu_agent_recall_subscription_states", ("release_id",)),
        ("cayu_agent_recall_subscription_claims", ("claim_id",)),
        ("cayu_agent_recall_subscription_claims", ("subscription_id", "attempt")),
        ("cayu_agent_recall_subscription_releases", ("release_id",)),
        ("cayu_agent_recall_subscription_evaluations", ("evaluation_id",)),
        ("cayu_agent_recall_subscription_evaluations", ("claim_id",)),
        (
            "cayu_agent_recall_subscription_evaluations",
            ("processing_operation_id",),
        ),
        ("cayu_agent_recall_subscription_evaluations", ("delivery_id",)),
        ("cayu_agent_recall_subscription_wake_claims", ("claim_id",)),
        ("cayu_agent_recall_subscription_wake_claims", ("wake_id", "attempt")),
        ("cayu_agent_recall_subscription_wake_releases", ("release_id",)),
        ("cayu_agent_recall_subscription_wake_states", ("wake_id",)),
        ("cayu_agent_recall_subscription_wake_states", ("delivery_id",)),
        ("cayu_agent_recall_subscription_wake_states", ("release_id",)),
        ("cayu_agent_recall_subscription_wake_states", ("acknowledgement_id",)),
    )
    for table, columns in unique_keys:
        if not sqlite_catalog._sqlite_has_unique_index(connection, table, columns):
            _raise_revision_73_sqlite_schema_error(table)

    required_foreign_keys = {
        "cayu_agent_recall_subscription_revisions": {
            (
                "cayu_agent_work_context_revisions",
                ("task_id", "work_context_revision"),
                ("task_id", "revision"),
                "RESTRICT",
            ),
        },
        "cayu_agent_recall_subscription_heads": {
            (
                "cayu_agent_recall_subscription_revisions",
                ("subscription_id", "current_revision"),
                ("subscription_id", "revision"),
                "RESTRICT",
            ),
        },
        "cayu_agent_recall_subscription_publications": {
            (
                "cayu_agent_recall_subscription_revisions",
                ("subscription_id", "subscription_revision"),
                ("subscription_id", "revision"),
                "RESTRICT",
            ),
        },
        "cayu_agent_recall_subscription_states": {
            (
                "cayu_agent_recall_subscription_revisions",
                ("subscription_id", "current_revision"),
                ("subscription_id", "revision"),
                "RESTRICT",
            ),
            (
                "cayu_agent_recall_subscription_releases",
                ("release_id",),
                ("release_id",),
                "RESTRICT",
            ),
        },
        "cayu_agent_recall_subscription_claims": {
            (
                "cayu_agent_recall_subscription_revisions",
                ("subscription_id", "subscription_revision"),
                ("subscription_id", "revision"),
                "RESTRICT",
            ),
        },
        "cayu_agent_recall_subscription_releases": {
            (
                "cayu_agent_recall_subscription_heads",
                ("subscription_id",),
                ("subscription_id",),
                "RESTRICT",
            ),
            (
                "cayu_agent_recall_subscription_claims",
                ("claim_id",),
                ("claim_id",),
                "RESTRICT",
            ),
        },
        "cayu_agent_recall_subscription_evaluations": {
            (
                "cayu_agent_recall_subscription_revisions",
                ("subscription_id", "subscription_revision"),
                ("subscription_id", "revision"),
                "RESTRICT",
            ),
            (
                "cayu_agent_recall_subscription_claims",
                ("claim_id",),
                ("claim_id",),
                "RESTRICT",
            ),
            (
                "cayu_agent_recall_deliveries",
                ("delivery_id",),
                ("delivery_id",),
                "RESTRICT",
            ),
        },
        "cayu_agent_recall_subscription_wake_claims": {
            (
                "cayu_agent_recall_subscription_evaluations",
                ("wake_id",),
                ("evaluation_id",),
                "RESTRICT",
            ),
            (
                "cayu_agent_recall_deliveries",
                ("delivery_id",),
                ("delivery_id",),
                "RESTRICT",
            ),
        },
        "cayu_agent_recall_subscription_wake_releases": {
            (
                "cayu_agent_recall_subscription_evaluations",
                ("wake_id",),
                ("evaluation_id",),
                "RESTRICT",
            ),
            (
                "cayu_agent_recall_subscription_wake_claims",
                ("claim_id",),
                ("claim_id",),
                "RESTRICT",
            ),
        },
        "cayu_agent_recall_subscription_wake_states": {
            (
                "cayu_agent_recall_subscription_evaluations",
                ("wake_id",),
                ("evaluation_id",),
                "RESTRICT",
            ),
            (
                "cayu_agent_recall_deliveries",
                ("delivery_id",),
                ("delivery_id",),
                "RESTRICT",
            ),
            (
                "cayu_agent_recall_subscription_wake_claims",
                ("claim_id",),
                ("claim_id",),
                "RESTRICT",
            ),
            (
                "cayu_agent_recall_subscription_wake_releases",
                ("release_id",),
                ("release_id",),
                "RESTRICT",
            ),
        },
    }
    for table, expected in required_foreign_keys.items():
        if sqlite_catalog._sqlite_foreign_key_groups(connection, table) != expected:
            _raise_revision_73_sqlite_schema_error(table)

    required_sql = {
        "cayu_agent_recall_subscription_revisions": (
            "status in ('active', 'paused', 'cancelled')",
            "priority >= 0",
            "priority <= 1000",
            "json_valid(subscription_json)",
            "json_type(subscription_json) = 'object'",
            "length(access_policy_sha256) = 64",
            "length(work_context_sha256) = 64",
        ),
        "cayu_agent_recall_subscription_states": (
            "run_state in ('due', 'claimed')",
            "run_state = 'due' and lease_expires_at is null",
            "run_state = 'claimed' and lease_expires_at is not null",
            "json_valid(state_json)",
        ),
        "cayu_agent_recall_subscription_evaluations": (
            "outcome in ('no_work', 'silent', 'wake')",
            "outcome = 'wake' and delivery_id is not null",
            "outcome != 'wake' and delivery_id is null",
            "json_valid(evaluation_json)",
        ),
        "cayu_agent_recall_subscription_wake_states": (
            "state in ('pending', 'claimed', 'acknowledged')",
            "state = 'claimed' and attempt > 0",
            "state = 'acknowledged' and attempt > 0",
            "json_valid(state_json)",
        ),
    }
    for table, fragments in required_sql.items():
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        normalized = sqlite_catalog._normalize_sqlite_schema_sql(None if row is None else row[0])
        if any(fragment not in normalized for fragment in fragments):
            _raise_revision_73_sqlite_schema_error(table)

    expected_indexes = {
        "idx_cayu_agent_recall_subscription_due": (
            "cayu_agent_recall_subscription_states",
            (
                "agent_id",
                "task_id",
                "knowledge_namespace",
                "access_policy_sha256",
                "next_evaluation_at",
                "subscription_id",
            ),
            None,
        ),
        "idx_cayu_agent_recall_subscription_wakes": (
            "cayu_agent_recall_subscription_wake_states",
            (
                "agent_id",
                "task_id",
                "knowledge_namespace",
                "access_policy_sha256",
                "committed_at",
                "wake_id",
            ),
            "where state != 'acknowledged'",
        ),
        "idx_cayu_agent_recall_subscription_evaluations": (
            "cayu_agent_recall_subscription_evaluations",
            ("subscription_id", "evaluation_id"),
            None,
        ),
    }
    for index, (table, expected, predicate) in expected_indexes.items():
        row = connection.execute(
            "SELECT tbl_name, sql FROM sqlite_master WHERE type = 'index' AND name = ?",
            (index,),
        ).fetchone()
        columns = tuple(
            str(index_row[2]) for index_row in connection.execute(f"PRAGMA index_info({index})")
        )
        normalized = "" if row is None else sqlite_catalog._normalize_sqlite_schema_sql(row[1])
        if (
            row is None
            or row[0] != table
            or columns != expected
            or (predicate is not None and predicate not in normalized)
        ):
            _raise_revision_73_sqlite_schema_error(index)


def _raise_revision_73_sqlite_schema_error(name: str) -> NoReturn:
    raise RuntimeError(
        "SQLite schema object "
        f"{name!r} conflicts with Cayu's idle recall-subscription contract. "
        "Run schema_mode=MIGRATE to install revision 73 or recreate the database."
    )
