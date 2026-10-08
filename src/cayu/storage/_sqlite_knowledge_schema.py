"""SQLite knowledge-schema integrity checks, independent of migration execution.

Revision gates remain with schema reconciliation. These validators inspect an
existing database and retain the structural and semantic checks for each gate.
"""

from __future__ import annotations

import sqlite3
from typing import NoReturn

from cayu.storage import _sqlite_catalog as sqlite_catalog

_KNOWLEDGE_CHUNK_LEGACY_COLUMNS = (
    "id",
    "entry_id",
    "chunk_index",
    "text",
    "content_hash",
    "source_uri",
    "metadata_json",
)


_KNOWLEDGE_CHUNK_KEYED_COLUMNS = ("fts_rowid", *_KNOWLEDGE_CHUNK_LEGACY_COLUMNS)


def _validate_revision_37_knowledge_fts_schema(connection: sqlite3.Connection) -> None:
    columns = connection.execute("PRAGMA table_info(cayu_knowledge_chunks)").fetchall()
    if tuple(str(row[1]) for row in columns) != _KNOWLEDGE_CHUNK_KEYED_COLUMNS:
        raise RuntimeError(
            "SQLite knowledge chunks do not provide the revision-37 stable FTS key. "
            "Restore the required schema from a known-good backup."
        )
    fts_rowid = columns[0]
    if str(fts_rowid[2]).upper() != "INTEGER" or int(fts_rowid[5]) != 1:
        raise RuntimeError(
            "SQLite knowledge chunks have an invalid revision-37 FTS key. "
            "Restore the required schema from a known-good backup."
        )
    if not sqlite_catalog._sqlite_has_unique_index(connection, "cayu_knowledge_chunks", ("id",)):
        raise RuntimeError("SQLite knowledge chunks are missing their unique public id constraint.")
    if not sqlite_catalog._sqlite_has_unique_index(
        connection,
        "cayu_knowledge_chunks",
        ("entry_id", "chunk_index"),
    ):
        raise RuntimeError(
            "SQLite knowledge chunks are missing their entry/chunk identity constraint."
        )
    entry_index = connection.execute(
        "SELECT tbl_name, sql FROM sqlite_master WHERE type = 'index' "
        "AND name = 'idx_cayu_knowledge_chunks_entry_index'"
    ).fetchone()
    entry_index_columns = (
        tuple(
            str(column[2])
            for column in connection.execute(
                "PRAGMA index_info(idx_cayu_knowledge_chunks_entry_index)"
            )
        )
        if entry_index is not None
        else ()
    )
    if (
        entry_index is None
        or entry_index[0] != "cayu_knowledge_chunks"
        or entry_index_columns != ("entry_id", "chunk_index")
    ):
        raise RuntimeError("Required Cayu SQLite knowledge chunk index is missing.")
    fts = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'cayu_knowledge_chunks_fts'"
    ).fetchone()
    normalized_fts = " ".join(str(fts[0]).lower().split()) if fts is not None else ""
    required_fts = "using fts5(entry_id unindexed, chunk_id unindexed, title, text)"
    if required_fts not in normalized_fts:
        raise RuntimeError("SQLite knowledge FTS does not match the revision-37 search contract.")


def _validate_revision_42_knowledge_schema(
    connection: sqlite3.Connection,
    *,
    require_payload_bytes: bool = False,
) -> None:
    required_tables = {
        "cayu_knowledge_entries": (
            ("id", "TEXT", 0, 1),
            ("namespace", "TEXT", 1, 0),
            ("current_revision", "INTEGER", 1, 0),
            ("created_at", "TEXT", 1, 0),
            ("updated_at", "TEXT", 1, 0),
        ),
        "cayu_knowledge_revisions": (
            ("entry_id", "TEXT", 1, 1),
            ("revision", "INTEGER", 1, 2),
            ("text", "TEXT", 1, 0),
            ("kind", "TEXT", 1, 0),
            ("visibility", "TEXT", 1, 0),
            ("status", "TEXT", 1, 0),
            ("created_by_type", "TEXT", 1, 0),
            ("created_by", "TEXT", 1, 0),
            ("created_at", "TEXT", 1, 0),
            ("updated_at", "TEXT", 1, 0),
            ("source_type", "TEXT", 0, 0),
            ("source_uri", "TEXT", 0, 0),
            ("source_id", "TEXT", 0, 0),
            ("source_hash", "TEXT", 0, 0),
            ("importance", "REAL", 0, 0),
            ("importance_source", "TEXT", 0, 0),
            ("confidence", "REAL", 0, 0),
            ("last_used_at", "TEXT", 0, 0),
            ("expires_at", "TEXT", 0, 0),
            ("title", "TEXT", 0, 0),
            ("metadata_json", "TEXT", 1, 0),
            *((("payload_bytes", "INTEGER", 1, 0),) if require_payload_bytes else ()),
        ),
        "cayu_knowledge_labels": (
            ("entry_id", "TEXT", 1, 1),
            ("entry_revision", "INTEGER", 1, 2),
            ("key", "TEXT", 1, 3),
            ("value", "TEXT", 1, 0),
        ),
        "cayu_knowledge_aspects": (
            ("entry_id", "TEXT", 1, 1),
            ("entry_revision", "INTEGER", 1, 2),
            ("aspect", "TEXT", 1, 3),
        ),
        "cayu_knowledge_impact_targets": (
            ("entry_id", "TEXT", 1, 1),
            ("entry_revision", "INTEGER", 1, 2),
            ("impact_target", "TEXT", 1, 3),
        ),
        "cayu_knowledge_chunks": (
            ("fts_rowid", "INTEGER", 0, 1),
            ("id", "TEXT", 1, 0),
            ("entry_id", "TEXT", 1, 0),
            ("entry_revision", "INTEGER", 1, 0),
            ("chunk_index", "INTEGER", 1, 0),
            ("text", "TEXT", 1, 0),
            ("content_hash", "TEXT", 0, 0),
            ("source_uri", "TEXT", 0, 0),
            ("metadata_json", "TEXT", 1, 0),
        ),
        "cayu_knowledge_publication_receipts": (
            ("operation_id", "TEXT", 0, 1),
            ("entry_id", "TEXT", 1, 0),
            ("entry_revision", "INTEGER", 1, 0),
            ("expected_revision", "INTEGER", 0, 0),
            ("request_sha256", "TEXT", 1, 0),
            ("entry_created_at", "TEXT", 1, 0),
            ("entry_updated_at", "TEXT", 1, 0),
            ("committed_at", "TEXT", 1, 0),
            ("access_snapshot_json", "TEXT", 1, 0),
        ),
    }
    for table, expected in required_tables.items():
        rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
        actual = tuple((str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5])) for row in rows)
        if actual != expected:
            _raise_revision_42_sqlite_schema_error(table)

    expected_foreign_keys = {
        "cayu_knowledge_entries": (
            ("cayu_knowledge_revisions", "current_revision", "revision", "NO ACTION"),
            ("cayu_knowledge_revisions", "id", "entry_id", "NO ACTION"),
        ),
        "cayu_knowledge_revisions": (("cayu_knowledge_entries", "entry_id", "id", "CASCADE"),),
        "cayu_knowledge_labels": (
            ("cayu_knowledge_revisions", "entry_id", "entry_id", "CASCADE"),
            ("cayu_knowledge_revisions", "entry_revision", "revision", "CASCADE"),
        ),
        "cayu_knowledge_aspects": (
            ("cayu_knowledge_revisions", "entry_id", "entry_id", "CASCADE"),
            ("cayu_knowledge_revisions", "entry_revision", "revision", "CASCADE"),
        ),
        "cayu_knowledge_impact_targets": (
            ("cayu_knowledge_revisions", "entry_id", "entry_id", "CASCADE"),
            ("cayu_knowledge_revisions", "entry_revision", "revision", "CASCADE"),
        ),
        "cayu_knowledge_chunks": (
            ("cayu_knowledge_revisions", "entry_id", "entry_id", "CASCADE"),
            ("cayu_knowledge_revisions", "entry_revision", "revision", "CASCADE"),
        ),
        "cayu_knowledge_publication_receipts": (),
    }
    for table, expected in expected_foreign_keys.items():
        actual = tuple(
            sorted(
                (
                    str(row[2]),
                    str(row[3]),
                    str(row[4]),
                    str(row[6]).upper(),
                )
                for row in connection.execute(f"PRAGMA foreign_key_list({table})")
            )
        )
        if actual != expected:
            _raise_revision_42_sqlite_schema_error(table)

    required_unique_keys = {
        "cayu_knowledge_entries": (("id",),),
        "cayu_knowledge_revisions": (("entry_id", "revision"),),
        "cayu_knowledge_labels": (("entry_id", "entry_revision", "key"),),
        "cayu_knowledge_aspects": (("entry_id", "entry_revision", "aspect"),),
        "cayu_knowledge_impact_targets": (("entry_id", "entry_revision", "impact_target"),),
        "cayu_knowledge_chunks": (
            ("id",),
            ("entry_id", "entry_revision", "chunk_index"),
        ),
        "cayu_knowledge_publication_receipts": (("operation_id",),),
    }
    for table, keys in required_unique_keys.items():
        if any(not sqlite_catalog._sqlite_has_unique_index(connection, table, key) for key in keys):
            _raise_revision_42_sqlite_schema_error(table)

    required_indexes = {
        "idx_cayu_knowledge_entries_namespace_current": (
            "cayu_knowledge_entries",
            ("namespace", "current_revision", "id"),
        ),
        "idx_cayu_knowledge_revisions_status": (
            "cayu_knowledge_revisions",
            ("status", "entry_id", "revision"),
        ),
        "idx_cayu_knowledge_revisions_kind": (
            "cayu_knowledge_revisions",
            ("kind", "entry_id", "revision"),
        ),
        "idx_cayu_knowledge_revisions_visibility": (
            "cayu_knowledge_revisions",
            ("visibility", "entry_id", "revision"),
        ),
        "idx_cayu_knowledge_revisions_source": (
            "cayu_knowledge_revisions",
            ("source_type", "source_id", "entry_id", "revision"),
        ),
        "idx_cayu_knowledge_revisions_expires_at": (
            "cayu_knowledge_revisions",
            ("expires_at", "entry_id", "revision"),
        ),
        "idx_cayu_knowledge_labels_key_value_entry": (
            "cayu_knowledge_labels",
            ("key", "value", "entry_id", "entry_revision"),
        ),
        "idx_cayu_knowledge_aspects_aspect_entry": (
            "cayu_knowledge_aspects",
            ("aspect", "entry_id", "entry_revision"),
        ),
        "idx_cayu_knowledge_impact_targets_target_entry": (
            "cayu_knowledge_impact_targets",
            ("impact_target", "entry_id", "entry_revision"),
        ),
        "idx_cayu_knowledge_chunks_entry_revision_index": (
            "cayu_knowledge_chunks",
            ("entry_id", "entry_revision", "chunk_index"),
        ),
        "idx_cayu_knowledge_publication_receipts_entry_revision": (
            "cayu_knowledge_publication_receipts",
            ("entry_id", "entry_revision"),
        ),
    }
    for index, (table, columns) in required_indexes.items():
        row = connection.execute(
            "SELECT tbl_name FROM sqlite_master WHERE type = 'index' AND name = ?",
            (index,),
        ).fetchone()
        actual_columns = tuple(
            str(column[2]) for column in connection.execute(f"PRAGMA index_info({index})")
        )
        if row is None or str(row[0]) != table or actual_columns != columns:
            _raise_revision_42_sqlite_schema_error(index)

    current_view = connection.execute(
        "SELECT type, sql FROM sqlite_master WHERE name = 'cayu_knowledge_current_entries'"
    ).fetchone()
    current_view_sql = sqlite_catalog._normalize_sqlite_schema_sql(
        None if current_view is None else current_view[1]
    )
    current_view_columns = sqlite_catalog._sqlite_table_columns(
        connection,
        "cayu_knowledge_current_entries",
    )
    if (
        current_view is None
        or str(current_view[0]) != "view"
        or current_view_columns
        != (
            "id",
            "revision",
            "namespace",
            "text",
            "kind",
            "visibility",
            "status",
            "created_by_type",
            "created_by",
            "created_at",
            "updated_at",
            "source_type",
            "source_uri",
            "source_id",
            "source_hash",
            "importance",
            "importance_source",
            "confidence",
            "last_used_at",
            "expires_at",
            "title",
            "metadata_json",
            *(("payload_bytes",) if require_payload_bytes else ()),
        )
        or "from cayu_knowledge_entries as logical" not in current_view_sql
        or "join cayu_knowledge_revisions as revision" not in current_view_sql
        or "revision.revision = logical.current_revision" not in current_view_sql
    ):
        _raise_revision_42_sqlite_schema_error("cayu_knowledge_current_entries")

    fts = connection.execute(
        "SELECT type, sql FROM sqlite_master WHERE name = 'cayu_knowledge_chunks_fts'"
    ).fetchone()
    fts_sql = sqlite_catalog._normalize_sqlite_schema_sql(None if fts is None else fts[1])
    if (
        fts is None
        or str(fts[0]) != "table"
        or sqlite_catalog._sqlite_table_columns(connection, "cayu_knowledge_chunks_fts")
        != ("entry_id", "entry_revision", "chunk_id", "title", "text")
        or "using fts5" not in fts_sql
        or any(
            fragment not in fts_sql
            for fragment in (
                "entry_id unindexed",
                "entry_revision unindexed",
                "chunk_id unindexed",
            )
        )
    ):
        _raise_revision_42_sqlite_schema_error("cayu_knowledge_chunks_fts")

    required_table_sql = {
        "cayu_knowledge_entries": (
            "check (current_revision > 0 and current_revision <= 2147483647)",
            "deferrable initially deferred",
        ),
        "cayu_knowledge_revisions": (
            "check (revision > 0 and revision <= 2147483647)",
            *(
                (
                    "payload_bytes > 0",
                    "payload_bytes <= 2147483647",
                )
                if require_payload_bytes
                else ()
            ),
        ),
        "cayu_knowledge_chunks": (
            "check (entry_revision > 0 and entry_revision <= 2147483647)",
            "check (chunk_index >= 0)",
        ),
        "cayu_knowledge_publication_receipts": (
            "check (entry_revision > 0 and entry_revision <= 2147483647)",
            "check (expected_revision > 0 and expected_revision <= 2147483647)",
            "entry_revision = expected_revision + 1",
        ),
    }
    for table, fragments in required_table_sql.items():
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        normalized = sqlite_catalog._normalize_sqlite_schema_sql(None if row is None else row[0])
        if any(fragment not in normalized for fragment in fragments):
            _raise_revision_42_sqlite_schema_error(table)


def _validate_revision_43_knowledge_schema(
    connection: sqlite3.Connection,
    *,
    relation_aware: bool = False,
) -> None:
    expected_columns = {
        "cayu_knowledge_evidence": (
            "id",
            "entry_id",
            "entry_revision",
            "chunk_id",
            "role",
            "source_type",
            "source_id",
            "source_uri",
            "source_revision",
            "source_hash",
            "locator_json",
            "disposition",
            "created_at",
            "metadata_json",
        ),
        "cayu_knowledge_changes": (
            "sequence",
            "id",
            "kind",
            "entry_id",
            "entry_revision",
            "committed_at",
            "operation_id",
            *(("relation_id",) if relation_aware else ()),
        ),
        "cayu_knowledge_change_audiences": (
            "change_sequence",
            "audience_kind",
            "namespace",
            "visibility",
            "source_type",
            "source_id",
            "status",
            "requires_include_expired",
        ),
        "cayu_knowledge_change_consumers": (
            "consumer_id",
            "access_scope_sha256",
            "cursor_sequence",
            "pending_change_sequence",
            "pending_claim_id",
            "pending_worker_id",
            "pending_attempt",
            "claimed_at",
            "lease_expires_at",
            "last_acknowledged_claim_id",
            "updated_at",
        ),
        "cayu_knowledge_change_acknowledgements": (
            "consumer_id",
            "claim_id",
            "claim_sha256",
            "change_sequence",
            "acknowledged_at",
        ),
        "cayu_knowledge_change_labels": (
            "change_sequence",
            "audience_kind",
            "key",
            "value",
        ),
    }
    for table, columns in expected_columns.items():
        if sqlite_catalog._sqlite_table_columns(connection, table) != columns:
            _raise_revision_43_sqlite_schema_error(table)

    evidence_foreign_keys = tuple(
        sorted(
            (
                str(row[2]),
                str(row[3]),
                str(row[4]),
                str(row[6]).upper(),
            )
            for row in connection.execute("PRAGMA foreign_key_list(cayu_knowledge_evidence)")
        )
    )
    if evidence_foreign_keys != tuple(
        sorted(
            (
                ("cayu_knowledge_chunks", "chunk_id", "id", "CASCADE"),
                ("cayu_knowledge_chunks", "entry_id", "entry_id", "CASCADE"),
                (
                    "cayu_knowledge_chunks",
                    "entry_revision",
                    "entry_revision",
                    "CASCADE",
                ),
                ("cayu_knowledge_revisions", "entry_id", "entry_id", "CASCADE"),
                (
                    "cayu_knowledge_revisions",
                    "entry_revision",
                    "revision",
                    "CASCADE",
                ),
            )
        )
    ):
        _raise_revision_43_sqlite_schema_error("cayu_knowledge_evidence")

    change_audience_foreign_keys = tuple(
        (
            str(row[2]),
            str(row[3]),
            str(row[4]),
            str(row[6]).upper(),
        )
        for row in connection.execute("PRAGMA foreign_key_list(cayu_knowledge_change_audiences)")
    )
    if change_audience_foreign_keys != (
        ("cayu_knowledge_changes", "change_sequence", "sequence", "CASCADE"),
    ):
        _raise_revision_43_sqlite_schema_error("cayu_knowledge_change_audiences")

    change_label_foreign_keys = tuple(
        sorted(
            (
                str(row[2]),
                str(row[3]),
                str(row[4]),
                str(row[6]).upper(),
            )
            for row in connection.execute("PRAGMA foreign_key_list(cayu_knowledge_change_labels)")
        )
    )
    if change_label_foreign_keys != tuple(
        sorted(
            (
                (
                    "cayu_knowledge_change_audiences",
                    "change_sequence",
                    "change_sequence",
                    "CASCADE",
                ),
                (
                    "cayu_knowledge_change_audiences",
                    "audience_kind",
                    "audience_kind",
                    "CASCADE",
                ),
            )
        )
    ):
        _raise_revision_43_sqlite_schema_error("cayu_knowledge_change_labels")

    consumer_foreign_keys = tuple(
        (
            str(row[2]),
            str(row[3]),
            str(row[4]),
            str(row[6]).upper(),
        )
        for row in connection.execute("PRAGMA foreign_key_list(cayu_knowledge_change_consumers)")
    )
    if consumer_foreign_keys != (
        ("cayu_knowledge_changes", "pending_change_sequence", "sequence", "NO ACTION"),
    ):
        _raise_revision_43_sqlite_schema_error("cayu_knowledge_change_consumers")

    acknowledgement_foreign_keys = tuple(
        sorted(
            (
                str(row[2]),
                str(row[3]),
                str(row[4]),
                str(row[6]).upper(),
            )
            for row in connection.execute(
                "PRAGMA foreign_key_list(cayu_knowledge_change_acknowledgements)"
            )
        )
    )
    if acknowledgement_foreign_keys != tuple(
        sorted(
            (
                (
                    "cayu_knowledge_change_consumers",
                    "consumer_id",
                    "consumer_id",
                    "CASCADE",
                ),
                (
                    "cayu_knowledge_changes",
                    "change_sequence",
                    "sequence",
                    "NO ACTION",
                ),
            )
        )
    ):
        _raise_revision_43_sqlite_schema_error("cayu_knowledge_change_acknowledgements")

    for table, key in (
        ("cayu_knowledge_evidence", ("id",)),
        ("cayu_knowledge_changes", ("sequence",)),
        ("cayu_knowledge_changes", ("id",)),
        (
            "cayu_knowledge_change_audiences",
            ("change_sequence", "audience_kind"),
        ),
        ("cayu_knowledge_change_consumers", ("consumer_id",)),
        (
            "cayu_knowledge_change_acknowledgements",
            ("consumer_id", "claim_id"),
        ),
        (
            "cayu_knowledge_change_labels",
            ("change_sequence", "audience_kind", "key"),
        ),
        (
            "cayu_knowledge_chunks",
            ("id", "entry_id", "entry_revision"),
        ),
    ):
        if not sqlite_catalog._sqlite_has_unique_index(connection, table, key):
            _raise_revision_43_sqlite_schema_error(table)

    required_indexes = {
        "idx_cayu_knowledge_evidence_entry_revision": (
            "cayu_knowledge_evidence",
            ("entry_id", "entry_revision", "id"),
        ),
        "idx_cayu_knowledge_evidence_source": (
            "cayu_knowledge_evidence",
            ("source_type", "source_id", "entry_id", "entry_revision"),
        ),
        "idx_cayu_knowledge_changes_entry_revision": (
            "cayu_knowledge_changes",
            ("entry_id", "entry_revision", "sequence"),
        ),
        "idx_cayu_knowledge_change_audiences_namespace": (
            "cayu_knowledge_change_audiences",
            ("namespace", "change_sequence", "audience_kind"),
        ),
        "idx_cayu_knowledge_change_audiences_status": (
            "cayu_knowledge_change_audiences",
            ("status", "change_sequence", "audience_kind"),
        ),
        "idx_cayu_knowledge_change_audiences_source": (
            "cayu_knowledge_change_audiences",
            ("source_type", "source_id", "change_sequence", "audience_kind"),
        ),
        "idx_cayu_knowledge_changes_operation": (
            "cayu_knowledge_changes",
            (("operation_id", "sequence") if relation_aware else ("operation_id",)),
        ),
        "idx_cayu_knowledge_change_consumers_lease": (
            "cayu_knowledge_change_consumers",
            ("lease_expires_at",),
        ),
        "idx_cayu_knowledge_change_labels_lookup": (
            "cayu_knowledge_change_labels",
            ("key", "value", "change_sequence", "audience_kind"),
        ),
        "idx_cayu_knowledge_chunks_identity_owner": (
            "cayu_knowledge_chunks",
            ("id", "entry_id", "entry_revision"),
        ),
    }
    for index, (table, columns) in required_indexes.items():
        row = connection.execute(
            "SELECT tbl_name, sql FROM sqlite_master WHERE type = 'index' AND name = ?",
            (index,),
        ).fetchone()
        actual_columns = tuple(
            str(column[2]) for column in connection.execute(f"PRAGMA index_info({index})")
        )
        if row is None or str(row[0]) != table or actual_columns != columns:
            _raise_revision_43_sqlite_schema_error(index)
        normalized = sqlite_catalog._normalize_sqlite_schema_sql(row[1])
        if index == "idx_cayu_knowledge_evidence_entry_revision" and (
            "id collate binary" not in normalized
        ):
            _raise_revision_43_sqlite_schema_error(index)
        if index == "idx_cayu_knowledge_changes_operation" and (
            "where operation_id is not null" not in normalized
        ):
            _raise_revision_43_sqlite_schema_error(index)
        if index == "idx_cayu_knowledge_change_consumers_lease" and (
            "where pending_change_sequence is not null" not in normalized
        ):
            _raise_revision_43_sqlite_schema_error(index)

    required_sql = {
        "cayu_knowledge_evidence": (
            "check (entry_revision > 0 and entry_revision <= 2147483647)",
            "check (role in ('origin', 'supporting'))",
            "check (source_id is not null or source_uri is not null)",
            "check (source_revision is not null or source_hash is not null)",
            "check (disposition in ('live', 'detached', 'retained'))",
        ),
        "cayu_knowledge_changes": (
            "check (sequence > 0 and sequence <= 9223372036854775807)",
            "check (entry_revision > 0 and entry_revision <= 2147483647)",
            (
                "kind in ( 'created', 'revision_appended', 'status_transitioned', "
                "'tombstoned', 'hard_deleted', 'expired', 'relation_published' )"
                if relation_aware
                else "kind in ( 'created', 'revision_appended', 'status_transitioned', "
                "'tombstoned', 'hard_deleted', 'expired' )"
            ),
        ),
        "cayu_knowledge_change_audiences": (
            (
                "check ( audience_kind in ( 'before', 'after', 'subject_exact', "
                "'subject_current', 'object_exact', 'object_current' ) )"
                if relation_aware
                else "check (audience_kind in ('before', 'after'))"
            ),
            "check ( requires_include_expired in (0, 1) )",
        ),
        "cayu_knowledge_change_consumers": (
            "check (cursor_sequence >= 0)",
            "check (pending_attempt >= 0)",
            "pending_change_sequence > cursor_sequence",
            "lease_expires_at > claimed_at",
        ),
        "cayu_knowledge_change_acknowledgements": (
            "length(claim_sha256) = 64",
            "claim_sha256 not glob '*[^0-9a-f]*'",
        ),
    }
    for table, fragments in required_sql.items():
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        normalized = sqlite_catalog._normalize_sqlite_schema_sql(None if row is None else row[0])
        if any(fragment not in normalized for fragment in fragments):
            _raise_revision_43_sqlite_schema_error(table)


def _validate_revision_60_knowledge_schema(connection: sqlite3.Connection) -> None:
    expected_columns = {
        "cayu_knowledge_relations": (
            "id",
            "subject_entry_id",
            "subject_revision",
            "object_entry_id",
            "object_revision",
            "kind",
            "created_by_type",
            "created_by",
            "policy_id",
            "created_at",
            "metadata_json",
        ),
        "cayu_knowledge_relation_publication_receipts": (
            "operation_id",
            "relation_ids_json",
            "request_sha256",
            "committed_at",
            "access_snapshots_json",
        ),
    }
    for table, columns in expected_columns.items():
        if sqlite_catalog._sqlite_table_columns(connection, table) != columns:
            _raise_revision_60_sqlite_schema_error(table)

    relation_foreign_keys = tuple(
        sorted(
            (
                str(row[2]),
                str(row[3]),
                str(row[4]),
                str(row[6]).upper(),
            )
            for row in connection.execute("PRAGMA foreign_key_list(cayu_knowledge_relations)")
        )
    )
    expected_foreign_keys = tuple(
        sorted(
            (
                (
                    "cayu_knowledge_revisions",
                    "subject_entry_id",
                    "entry_id",
                    "CASCADE",
                ),
                (
                    "cayu_knowledge_revisions",
                    "subject_revision",
                    "revision",
                    "CASCADE",
                ),
                (
                    "cayu_knowledge_revisions",
                    "object_entry_id",
                    "entry_id",
                    "CASCADE",
                ),
                (
                    "cayu_knowledge_revisions",
                    "object_revision",
                    "revision",
                    "CASCADE",
                ),
            )
        )
    )
    if relation_foreign_keys != expected_foreign_keys:
        _raise_revision_60_sqlite_schema_error("cayu_knowledge_relations")
    if tuple(
        connection.execute("PRAGMA foreign_key_list(cayu_knowledge_relation_publication_receipts)")
    ):
        _raise_revision_60_sqlite_schema_error("cayu_knowledge_relation_publication_receipts")

    for table, key in (
        ("cayu_knowledge_relations", ("id",)),
        (
            "cayu_knowledge_relations",
            (
                "kind",
                "subject_entry_id",
                "subject_revision",
                "object_entry_id",
                "object_revision",
            ),
        ),
        ("cayu_knowledge_relation_publication_receipts", ("operation_id",)),
        ("cayu_knowledge_changes", ("relation_id",)),
    ):
        if not sqlite_catalog._sqlite_has_unique_index(connection, table, key):
            _raise_revision_60_sqlite_schema_error(table)

    required_indexes = {
        "idx_cayu_knowledge_relations_subject": (
            "cayu_knowledge_relations",
            ("subject_entry_id", "subject_revision", "created_at", "id"),
        ),
        "idx_cayu_knowledge_relations_object": (
            "cayu_knowledge_relations",
            ("object_entry_id", "object_revision", "created_at", "id"),
        ),
        "idx_cayu_knowledge_relations_subject_kind": (
            "cayu_knowledge_relations",
            ("subject_entry_id", "subject_revision", "kind", "created_at", "id"),
        ),
        "idx_cayu_knowledge_relations_object_kind": (
            "cayu_knowledge_relations",
            ("object_entry_id", "object_revision", "kind", "created_at", "id"),
        ),
        "idx_cayu_knowledge_changes_relation": (
            "cayu_knowledge_changes",
            ("relation_id",),
        ),
    }
    for index, (table, columns) in required_indexes.items():
        row = connection.execute(
            "SELECT tbl_name, sql FROM sqlite_master WHERE type = 'index' AND name = ?",
            (index,),
        ).fetchone()
        actual_columns = tuple(
            str(column[2]) for column in connection.execute(f"PRAGMA index_info({index})")
        )
        if row is None or str(row[0]) != table or actual_columns != columns:
            _raise_revision_60_sqlite_schema_error(index)
        normalized = sqlite_catalog._normalize_sqlite_schema_sql(row[1])
        if index.startswith("idx_cayu_knowledge_relations_") and (
            "id collate binary" not in normalized
        ):
            _raise_revision_60_sqlite_schema_error(index)
        if index == "idx_cayu_knowledge_changes_relation" and (
            "where relation_id is not null" not in normalized
        ):
            _raise_revision_60_sqlite_schema_error(index)

    required_sql = {
        "cayu_knowledge_relations": (
            "kind in ('supersedes', 'derived_from', 'contradicts')",
            "subject_entry_id <> object_entry_id",
            "kind <> 'contradicts' or subject_entry_id collate binary < object_entry_id collate binary",
            "json_valid(metadata_json)",
        ),
        "cayu_knowledge_relation_publication_receipts": (
            "json_valid(relation_ids_json)",
            "json_valid(access_snapshots_json)",
            "length(request_sha256) = 64",
        ),
        "cayu_knowledge_changes": (
            "kind = 'relation_published' and relation_id is not null",
            "kind <> 'relation_published' and relation_id is null",
        ),
    }
    for table, fragments in required_sql.items():
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        normalized = sqlite_catalog._normalize_sqlite_schema_sql(None if row is None else row[0])
        if any(fragment not in normalized for fragment in fragments):
            _raise_revision_60_sqlite_schema_error(table)


def _raise_revision_60_sqlite_schema_error(name: str) -> NoReturn:
    raise RuntimeError(
        "SQLite schema object "
        f"{name!r} conflicts with Cayu's revision-bound knowledge relation contract. "
        "Recreate the prerelease knowledge database with schema_mode=CREATE or MIGRATE."
    )


def _validate_revision_63_knowledge_schema(connection: sqlite3.Connection) -> None:
    table = "cayu_knowledge_maintenance_decisions"
    expected_columns = (
        ("operation_id", "TEXT", 0, 1),
        ("proposal_id", "TEXT", 1, 0),
        ("proposal_fingerprint", "TEXT", 1, 0),
        ("request_sha256", "TEXT", 1, 0),
        ("committed_at", "TEXT", 1, 0),
        ("proposal_json", "TEXT", 1, 0),
        ("decision_json", "TEXT", 1, 0),
        ("receipt_json", "TEXT", 1, 0),
        ("access_snapshot_json", "TEXT", 1, 0),
    )
    actual_columns = tuple(
        (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
        for row in connection.execute(f"PRAGMA table_info({table})")
    )
    if actual_columns != expected_columns:
        _raise_revision_63_sqlite_schema_error(table)
    if tuple(connection.execute(f"PRAGMA foreign_key_list({table})")):
        _raise_revision_63_sqlite_schema_error(table)
    for key in (("operation_id",), ("proposal_id",)):
        if not sqlite_catalog._sqlite_has_unique_index(connection, table, key):
            _raise_revision_63_sqlite_schema_error(table)
    row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    normalized = sqlite_catalog._normalize_sqlite_schema_sql(None if row is None else row[0])
    required = (
        "length(proposal_fingerprint) = 64",
        "proposal_fingerprint not glob '*[^0-9a-f]*'",
        "length(request_sha256) = 64",
        "request_sha256 not glob '*[^0-9a-f]*'",
        "json_valid(proposal_json)",
        "json_type(proposal_json) = 'object'",
        "json_valid(decision_json)",
        "json_type(decision_json) = 'object'",
        "json_valid(receipt_json)",
        "json_type(receipt_json) = 'object'",
        "json_valid(access_snapshot_json)",
        "json_type(access_snapshot_json) = 'object'",
    )
    if any(fragment not in normalized for fragment in required):
        _raise_revision_63_sqlite_schema_error(table)


def _raise_revision_63_sqlite_schema_error(name: str) -> NoReturn:
    raise RuntimeError(
        "SQLite schema object "
        f"{name!r} conflicts with Cayu's reviewed knowledge maintenance contract. "
        "Recreate the prerelease knowledge database with schema_mode=CREATE or MIGRATE."
    )


def _validate_revision_67_knowledge_schema(connection: sqlite3.Connection) -> None:
    table = "cayu_knowledge_maintenance_proposals"
    expected_columns = (
        ("operation_id", "TEXT", 0, 1),
        ("proposal_id", "TEXT", 1, 0),
        ("replacement_entry_id", "TEXT", 1, 0),
        ("replacement_revision", "INTEGER", 1, 0),
        ("proposal_fingerprint", "TEXT", 1, 0),
        ("accepted_plan_fingerprint", "TEXT", 1, 0),
        ("request_sha256", "TEXT", 1, 0),
        ("committed_at", "TEXT", 1, 0),
        ("proposal_json", "TEXT", 1, 0),
        ("accepted_plan_json", "TEXT", 1, 0),
        ("receipt_json", "TEXT", 1, 0),
        ("access_snapshot_json", "TEXT", 1, 0),
    )
    actual_columns = tuple(
        (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
        for row in connection.execute(f"PRAGMA table_info({table})")
    )
    if actual_columns != expected_columns:
        _raise_revision_67_sqlite_schema_error(table)
    if tuple(connection.execute(f"PRAGMA foreign_key_list({table})")):
        _raise_revision_67_sqlite_schema_error(table)
    for key in (("operation_id",), ("proposal_id",), ("replacement_entry_id",)):
        if not sqlite_catalog._sqlite_has_unique_index(connection, table, key):
            _raise_revision_67_sqlite_schema_error(table)
    row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    normalized = sqlite_catalog._normalize_sqlite_schema_sql(None if row is None else row[0])
    required = (
        "replacement_revision > 0",
        "replacement_revision <= 2147483647",
        "length(proposal_fingerprint) = 64",
        "proposal_fingerprint not glob '*[^0-9a-f]*'",
        "length(accepted_plan_fingerprint) = 64",
        "accepted_plan_fingerprint not glob '*[^0-9a-f]*'",
        "length(request_sha256) = 64",
        "request_sha256 not glob '*[^0-9a-f]*'",
        "json_valid(proposal_json)",
        "json_type(proposal_json) = 'object'",
        "json_valid(accepted_plan_json)",
        "json_type(accepted_plan_json) = 'object'",
        "json_valid(receipt_json)",
        "json_type(receipt_json) = 'object'",
        "json_valid(access_snapshot_json)",
        "json_type(access_snapshot_json) = 'object'",
    )
    if any(fragment not in normalized for fragment in required):
        _raise_revision_67_sqlite_schema_error(table)


def _raise_revision_67_sqlite_schema_error(name: str) -> NoReturn:
    raise RuntimeError(
        "SQLite schema object "
        f"{name!r} conflicts with Cayu's pending maintenance proposal contract. "
        "Recreate the prerelease knowledge database with schema_mode=CREATE or MIGRATE."
    )


def _validate_revision_75_knowledge_activation_schema(
    connection: sqlite3.Connection,
) -> None:
    table = "cayu_knowledge_activation_receipts"
    columns = tuple(
        (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
        for row in connection.execute(f"PRAGMA table_info({table})")
    )
    expected_columns = (
        ("operation_id", "TEXT", 0, 1),
        ("entry_id", "TEXT", 1, 0),
        ("entry_revision", "INTEGER", 1, 0),
        ("expected_revision", "INTEGER", 0, 0),
        ("publication_request_sha256", "TEXT", 1, 0),
        ("committed_at", "TEXT", 1, 0),
        ("receipt_json", "TEXT", 1, 0),
        ("access_snapshot_json", "TEXT", 1, 0),
    )
    table_row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    normalized = sqlite_catalog._normalize_sqlite_schema_sql(
        None if table_row is None else table_row[0]
    )
    required_fragments = (
        "operation_id text collate binary primary key",
        "entry_id text collate binary not null",
        "publication_request_sha256 text collate binary not null",
        "entry_revision > 0",
        "entry_revision <= 2147483647",
        "expected_revision > 0",
        "expected_revision <= 2147483647",
        "length(publication_request_sha256) = 64",
        "publication_request_sha256 not glob '*[^0-9a-f]*'",
        "json_valid(receipt_json)",
        "json_type(receipt_json) = 'object'",
        "length(cast(receipt_json as blob)) between 1 and 1114112",
        "json_valid(access_snapshot_json)",
        "json_type(access_snapshot_json) = 'object'",
        "expected_revision is null and entry_revision = 1",
        "entry_revision = expected_revision + 1",
    )
    if (
        columns != expected_columns
        or sqlite_catalog._sqlite_foreign_key_groups(connection, table)
        or any(fragment not in normalized for fragment in required_fragments)
    ):
        _raise_revision_75_sqlite_schema_error(table)

    index = "idx_cayu_knowledge_activation_receipts_entry_revision"
    index_row = connection.execute(
        "SELECT tbl_name FROM sqlite_master WHERE type = 'index' AND name = ?",
        (index,),
    ).fetchone()
    index_columns = tuple(str(row[2]) for row in connection.execute(f"PRAGMA index_info({index})"))
    index_collations = tuple(
        str(row[4]).upper()
        for row in connection.execute(f"PRAGMA index_xinfo({index})")
        if int(row[5]) == 1
    )
    if (
        index_row is None
        or index_row[0] != table
        or index_columns
        != (
            "entry_id",
            "entry_revision",
        )
        or index_collations != ("BINARY", "BINARY")
    ):
        _raise_revision_75_sqlite_schema_error(index)

    retirement_table = "cayu_knowledge_activation_retirements"
    retirement_columns = tuple(
        (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
        for row in connection.execute(f"PRAGMA table_info({retirement_table})")
    )
    expected_retirement_columns = (
        ("entry_id", "TEXT", 0, 1),
        ("entry_revision", "INTEGER", 1, 0),
        ("retired_at", "TEXT", 1, 0),
        ("retirement_json", "TEXT", 1, 0),
    )
    retirement_row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
        (retirement_table,),
    ).fetchone()
    retirement_sql = sqlite_catalog._normalize_sqlite_schema_sql(
        None if retirement_row is None else retirement_row[0]
    )
    required_retirement_fragments = (
        "entry_id text collate binary primary key",
        "entry_revision > 0",
        "entry_revision <= 2147483647",
        "json_valid(retirement_json)",
        "json_type(retirement_json) = 'object'",
        "length(cast(retirement_json as blob)) between 1 and 1048576",
    )
    if (
        retirement_columns != expected_retirement_columns
        or sqlite_catalog._sqlite_foreign_key_groups(connection, retirement_table)
        or any(fragment not in retirement_sql for fragment in required_retirement_fragments)
    ):
        _raise_revision_75_sqlite_schema_error(retirement_table)


def _raise_revision_75_sqlite_schema_error(name: str) -> NoReturn:
    raise RuntimeError(
        "SQLite schema object "
        f"{name!r} conflicts with Cayu's knowledge-activation authority contract. "
        "Run schema_mode=MIGRATE to install revision 75 or recreate the database."
    )


def _validate_revision_77_knowledge_maintenance_governance_schema(
    connection: sqlite3.Connection,
) -> None:
    table = "cayu_knowledge_maintenance_governance_routes"
    columns = tuple(
        (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
        for row in connection.execute(f"PRAGMA table_info({table})")
    )
    expected_columns = (
        ("operation_id", "TEXT", 0, 1),
        ("proposal_id", "TEXT", 1, 0),
        ("proposal_fingerprint", "TEXT", 1, 0),
        ("request_sha256", "TEXT", 1, 0),
        ("committed_at", "TEXT", 1, 0),
        ("receipt_json", "TEXT", 1, 0),
        ("access_snapshot_json", "TEXT", 1, 0),
    )
    row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    normalized = sqlite_catalog._normalize_sqlite_schema_sql(None if row is None else row[0])
    required_fragments = (
        "operation_id text collate binary primary key",
        "proposal_id text collate binary not null unique",
        "proposal_fingerprint text collate binary not null",
        "request_sha256 text collate binary not null",
        "length(proposal_fingerprint) = 64",
        "proposal_fingerprint not glob '*[^0-9a-f]*'",
        "length(request_sha256) = 64",
        "request_sha256 not glob '*[^0-9a-f]*'",
        "json_valid(receipt_json)",
        "json_type(receipt_json) = 'object'",
        "length(cast(receipt_json as blob)) between 1 and 640000",
        "json_valid(access_snapshot_json)",
        "json_type(access_snapshot_json) = 'object'",
    )
    unique_proposal = any(
        int(index_row[2]) == 1
        and tuple(
            str(column_row[0])
            for column_row in connection.execute(
                "SELECT name FROM pragma_index_info(?) ORDER BY seqno",
                (str(index_row[1]),),
            )
        )
        == ("proposal_id",)
        for index_row in connection.execute(f"PRAGMA index_list({table})")
    )
    if (
        columns != expected_columns
        or sqlite_catalog._sqlite_foreign_key_groups(connection, table)
        or not unique_proposal
        or any(fragment not in normalized for fragment in required_fragments)
    ):
        raise RuntimeError(
            "SQLite schema object "
            f"{table!r} conflicts with Cayu's maintenance-governance contract. "
            "Run schema_mode=MIGRATE to install revision 77 or recreate the database."
        )


def _validate_revision_78_knowledge_semantic_watch_schema(
    connection: sqlite3.Connection,
) -> None:
    table = "cayu_knowledge_semantic_watch_receipts"
    columns = tuple(
        (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
        for row in connection.execute(f"PRAGMA table_info({table})")
    )
    expected_columns = (
        ("operation_id", "TEXT", 0, 1),
        ("invocation_sha256", "TEXT", 1, 0),
        ("request_sha256", "TEXT", 1, 0),
        ("committed_at", "TEXT", 1, 0),
        ("receipt_json", "TEXT", 1, 0),
        ("access_scope_json", "TEXT", 1, 0),
    )
    row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    normalized = sqlite_catalog._normalize_sqlite_schema_sql(None if row is None else row[0])
    required_fragments = (
        "operation_id text collate binary primary key",
        "invocation_sha256 text collate binary not null",
        "request_sha256 text collate binary not null",
        "length(invocation_sha256) = 64",
        "invocation_sha256 not glob '*[^0-9a-f]*'",
        "length(request_sha256) = 64",
        "request_sha256 not glob '*[^0-9a-f]*'",
        "json_valid(receipt_json)",
        "json_type(receipt_json) = 'object'",
        "length(cast(receipt_json as blob)) between 1 and 384000",
        "json_valid(access_scope_json)",
        "json_type(access_scope_json) = 'object'",
        "length(cast(access_scope_json as blob)) between 1 and 384000",
    )
    if (
        columns != expected_columns
        or sqlite_catalog._sqlite_foreign_key_groups(connection, table)
        or any(fragment not in normalized for fragment in required_fragments)
    ):
        raise RuntimeError(
            "SQLite schema object "
            f"{table!r} conflicts with Cayu's semantic-watch receipt contract. "
            "Run schema_mode=MIGRATE to install revision 78 or recreate the database."
        )


def _validate_revision_44_knowledge_schema(connection: sqlite3.Connection) -> None:
    expected_columns = {
        "cayu_knowledge_index_readiness_events": (
            "sequence",
            "identity_sha256",
            "entry_id",
            "entry_revision",
            "chunk_id",
            "projection_type",
            "projection_content_hash",
            "embedding_model",
            "dimensions",
            "preprocessing_version",
            "generator",
            "generator_version",
            "index_representation_version",
            "state",
            "attempt_id",
            "failure_code",
            "operation_id",
            "update_sha256",
            "published_at",
        ),
        "cayu_knowledge_index_readiness_current": (
            "identity_sha256",
            "sequence",
        ),
    }
    for table, columns in expected_columns.items():
        if sqlite_catalog._sqlite_table_columns(connection, table) != columns:
            _raise_revision_44_sqlite_schema_error(table)

    foreign_keys = tuple(
        (
            str(row[2]),
            str(row[3]),
            str(row[4]),
            str(row[6]).upper(),
        )
        for row in connection.execute(
            "PRAGMA foreign_key_list(cayu_knowledge_index_readiness_current)"
        )
    )
    if foreign_keys != (
        (
            "cayu_knowledge_index_readiness_events",
            "identity_sha256",
            "identity_sha256",
            "CASCADE",
        ),
        ("cayu_knowledge_index_readiness_events", "sequence", "sequence", "CASCADE"),
    ):
        _raise_revision_44_sqlite_schema_error("cayu_knowledge_index_readiness_current")

    for table, key in (
        ("cayu_knowledge_index_readiness_events", ("sequence",)),
        ("cayu_knowledge_index_readiness_events", ("operation_id",)),
        ("cayu_knowledge_index_readiness_events", ("identity_sha256", "sequence")),
        ("cayu_knowledge_index_readiness_current", ("identity_sha256",)),
        ("cayu_knowledge_index_readiness_current", ("sequence",)),
    ):
        if not sqlite_catalog._sqlite_has_unique_index(connection, table, key):
            _raise_revision_44_sqlite_schema_error(table)

    required_indexes = {
        "idx_cayu_knowledge_index_readiness_identity_sequence": (
            "cayu_knowledge_index_readiness_events",
            ("identity_sha256", "sequence"),
        ),
        "idx_cayu_knowledge_index_readiness_entry_revision": (
            "cayu_knowledge_index_readiness_events",
            ("entry_id", "entry_revision", "projection_type", "sequence"),
        ),
        "idx_cayu_knowledge_index_readiness_projection_lookup": (
            "cayu_knowledge_index_readiness_events",
            (
                "entry_id",
                "entry_revision",
                "chunk_id",
                "projection_type",
                "embedding_model",
                "dimensions",
                "sequence",
            ),
        ),
    }
    for index, (table, columns) in required_indexes.items():
        row = connection.execute(
            "SELECT tbl_name FROM sqlite_master WHERE type = 'index' AND name = ?",
            (index,),
        ).fetchone()
        actual_columns = tuple(
            str(column[2]) for column in connection.execute(f"PRAGMA index_info({index})")
        )
        if row is None or str(row[0]) != table or actual_columns != columns:
            _raise_revision_44_sqlite_schema_error(index)

    row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' "
        "AND name = 'cayu_knowledge_index_readiness_events'"
    ).fetchone()
    normalized = sqlite_catalog._normalize_sqlite_schema_sql(None if row is None else row[0])
    required_fragments = (
        "check (sequence > 0 and sequence <= 9223372036854775807)",
        "length(identity_sha256) = 64",
        "identity_sha256 not glob '*[^0-9a-f]*'",
        "check (entry_revision > 0 and entry_revision <= 2147483647)",
        "check (dimensions > 0)",
        "check (state in ('pending', 'ready', 'failed'))",
        "length(update_sha256) = 64",
        "update_sha256 not glob '*[^0-9a-f]*'",
        "state = 'failed' and failure_code is not null",
        "state <> 'failed' and failure_code is null",
    )
    if any(fragment not in normalized for fragment in required_fragments):
        _raise_revision_44_sqlite_schema_error("cayu_knowledge_index_readiness_events")


def _raise_revision_42_sqlite_schema_error(name: str) -> NoReturn:
    raise RuntimeError(
        f"SQLite schema object {name!r} conflicts with Cayu's revision-first "
        "knowledge contract. Recreate the Cayu database from a known-good "
        "revision-42 schema."
    )


def _raise_revision_43_sqlite_schema_error(name: str) -> NoReturn:
    raise RuntimeError(
        f"SQLite schema object {name!r} conflicts with Cayu's knowledge evidence "
        "and atomic change contract. Recreate or migrate the Cayu database from "
        "a known-good revision-43 schema."
    )


def _raise_revision_44_sqlite_schema_error(name: str) -> NoReturn:
    raise RuntimeError(
        f"SQLite schema object {name!r} conflicts with Cayu's derived-index "
        "identity and readiness contract. Recreate or migrate the Cayu database "
        "from a known-good revision-44 schema."
    )


def _validate_knowledge_publication_access_snapshot_column(
    connection: sqlite3.Connection,
) -> None:
    columns = {
        str(row[1]): (str(row[2]).upper(), int(row[3]))
        for row in connection.execute("PRAGMA table_info(cayu_knowledge_publication_receipts)")
    }
    if columns.get("access_snapshot_json") != ("TEXT", 1):
        raise RuntimeError(
            "SQLite schema object "
            "'cayu_knowledge_publication_receipts.access_snapshot_json' conflicts "
            "with Cayu's knowledge authorization contract. Recreate the Cayu "
            "database from a known-good revision-41 schema."
        )
