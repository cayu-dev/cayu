"""PostgreSQL schema checks for knowledge publication, relations and maintenance.

Callers retain revision gates, validation order and the existing cursor/transaction.
These functions inspect the schema without acquiring connections or applying DDL.
"""

from __future__ import annotations

from typing import Any, NoReturn


async def _validate_knowledge_activation_schema(cur: Any) -> None:
    table = "cayu_knowledge_activation_receipts"
    await cur.execute(
        """
            SELECT column_name, data_type, is_nullable, collation_name
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'cayu_knowledge_activation_receipts'
            ORDER BY ordinal_position
            """
    )
    if tuple(await cur.fetchall()) != (
        ("operation_id", "text", "NO", "C"),
        ("entry_id", "text", "NO", "C"),
        ("entry_revision", "integer", "NO", None),
        ("expected_revision", "integer", "YES", None),
        ("publication_request_sha256", "text", "NO", "C"),
        ("committed_at", "timestamp with time zone", "NO", None),
        ("receipt_json", "text", "NO", None),
        ("access_snapshot", "jsonb", "NO", None),
    ):
        _raise_knowledge_activation_schema_error(table)

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
              AND table_record.relname = 'cayu_knowledge_activation_receipts'
            """
    )
    constraints = tuple(
        (str(kind), " ".join(str(definition).lower().split()))
        for kind, definition in await cur.fetchall()
    )
    required = (
        ("p", ("primary key (operation_id)",)),
        ("c", ("entry_revision > 0", "entry_revision <= 2147483647")),
        ("c", ("expected_revision > 0", "expected_revision <= 2147483647")),
        ("c", ("publication_request_sha256", "[0-9a-f]{64}")),
        (
            "c",
            (
                "octet_length(receipt_json)",
                "jsonb_typeof((receipt_json)::jsonb)",
                "object",
                "1114112",
            ),
        ),
        ("c", ("jsonb_typeof(access_snapshot)", "object")),
        (
            "c",
            (
                "expected_revision is null",
                "entry_revision = 1",
                "entry_revision = (expected_revision + 1)",
            ),
        ),
    )
    if any(kind == "f" for kind, _definition in constraints) or any(
        not any(
            actual_kind == expected_kind and all(fragment in definition for fragment in fragments)
            for actual_kind, definition in constraints
        )
        for expected_kind, fragments in required
    ):
        _raise_knowledge_activation_schema_error(table)

    index = "idx_cayu_knowledge_activation_receipts_entry_revision"
    await cur.execute(
        "SELECT indexdef FROM pg_indexes WHERE schemaname = current_schema() AND indexname = %s",
        (index,),
    )
    row = await cur.fetchone()
    definition = "" if row is None else " ".join(str(row[0]).lower().split())
    if (
        "cayu_knowledge_activation_receipts using btree (entry_id, entry_revision)"
        not in definition
    ):
        _raise_knowledge_activation_schema_error(index)

    retirement_table = "cayu_knowledge_activation_retirements"
    await cur.execute(
        """
            SELECT column_name, data_type, is_nullable, collation_name
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'cayu_knowledge_activation_retirements'
            ORDER BY ordinal_position
            """
    )
    if tuple(await cur.fetchall()) != (
        ("entry_id", "text", "NO", "C"),
        ("entry_revision", "integer", "NO", None),
        ("retired_at", "timestamp with time zone", "NO", None),
        ("retirement_json", "text", "NO", None),
    ):
        _raise_knowledge_activation_schema_error(retirement_table)
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
              AND table_record.relname = 'cayu_knowledge_activation_retirements'
            """
    )
    retirement_constraints = tuple(
        (str(kind), " ".join(str(definition).lower().split()))
        for kind, definition in await cur.fetchall()
    )
    retirement_required = (
        ("p", ("primary key (entry_id)",)),
        ("c", ("entry_revision > 0", "entry_revision <= 2147483647")),
        (
            "c",
            (
                "octet_length(retirement_json)",
                "jsonb_typeof((retirement_json)::jsonb)",
                "object",
                "1048576",
            ),
        ),
    )
    if any(kind == "f" for kind, _definition in retirement_constraints) or any(
        not any(
            actual_kind == expected_kind and all(fragment in definition for fragment in fragments)
            for actual_kind, definition in retirement_constraints
        )
        for expected_kind, fragments in retirement_required
    ):
        _raise_knowledge_activation_schema_error(retirement_table)


def _raise_knowledge_activation_schema_error(name: str) -> NoReturn:
    raise RuntimeError(
        "Postgres schema object "
        f"{name!r} conflicts with Cayu's knowledge-activation authority contract. "
        "Run `cayu storage migrate` to install revision 75 or recreate the database."
    )


async def _validate_knowledge_maintenance_governance_schema(cur: Any) -> None:
    table = "cayu_knowledge_maintenance_governance_routes"
    await cur.execute(
        """
            SELECT column_name, data_type, is_nullable, collation_name
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'cayu_knowledge_maintenance_governance_routes'
            ORDER BY ordinal_position
            """
    )
    if tuple(await cur.fetchall()) != (
        ("operation_id", "text", "NO", "C"),
        ("proposal_id", "text", "NO", "C"),
        ("proposal_fingerprint", "text", "NO", "C"),
        ("request_sha256", "text", "NO", "C"),
        ("committed_at", "timestamp with time zone", "NO", None),
        ("receipt_json", "text", "NO", None),
        ("access_snapshot", "jsonb", "NO", None),
    ):
        _raise_knowledge_maintenance_governance_schema_error(table)
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
              AND table_record.relname = 'cayu_knowledge_maintenance_governance_routes'
            """
    )
    constraints = tuple(
        (str(kind), " ".join(str(definition).lower().split()))
        for kind, definition in await cur.fetchall()
    )
    required = (
        ("p", ("primary key (operation_id)",)),
        ("u", ("unique (proposal_id)",)),
        ("c", ("proposal_fingerprint", "^[0-9a-f]{64}$")),
        ("c", ("request_sha256", "^[0-9a-f]{64}$")),
        (
            "c",
            (
                "octet_length(receipt_json)",
                "jsonb_typeof((receipt_json)::jsonb)",
                "object",
                "640000",
            ),
        ),
        ("c", ("jsonb_typeof(access_snapshot)", "object")),
    )
    if any(kind == "f" for kind, _definition in constraints) or any(
        not any(
            actual_kind == expected_kind and all(fragment in definition for fragment in fragments)
            for actual_kind, definition in constraints
        )
        for expected_kind, fragments in required
    ):
        _raise_knowledge_maintenance_governance_schema_error(table)


def _raise_knowledge_maintenance_governance_schema_error(name: str) -> NoReturn:
    raise RuntimeError(
        "Postgres schema object "
        f"{name!r} conflicts with Cayu's maintenance-governance contract. "
        "Run `cayu storage migrate` to install revision 77 or recreate the database."
    )


async def _validate_knowledge_semantic_watch_schema(cur: Any) -> None:
    table = "cayu_knowledge_semantic_watch_receipts"
    await cur.execute(
        """
            SELECT column_name, data_type, is_nullable, collation_name
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'cayu_knowledge_semantic_watch_receipts'
            ORDER BY ordinal_position
            """
    )
    if tuple(await cur.fetchall()) != (
        ("operation_id", "text", "NO", "C"),
        ("invocation_sha256", "text", "NO", "C"),
        ("request_sha256", "text", "NO", "C"),
        ("committed_at", "timestamp with time zone", "NO", None),
        ("receipt_json", "text", "NO", None),
        ("access_scope", "jsonb", "NO", None),
    ):
        _raise_knowledge_semantic_watch_schema_error(table)
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
              AND table_record.relname = 'cayu_knowledge_semantic_watch_receipts'
            """
    )
    constraints = tuple(
        (str(kind), " ".join(str(definition).lower().split()))
        for kind, definition in await cur.fetchall()
    )
    required = (
        ("p", ("primary key (operation_id)",)),
        ("c", ("invocation_sha256", "^[0-9a-f]{64}$")),
        ("c", ("request_sha256", "^[0-9a-f]{64}$")),
        (
            "c",
            (
                "octet_length(receipt_json)",
                "jsonb_typeof((receipt_json)::jsonb)",
                "object",
                "384000",
            ),
        ),
        (
            "c",
            (
                "octet_length((access_scope)::text)",
                "jsonb_typeof(access_scope)",
                "object",
                "384000",
            ),
        ),
    )
    if any(kind == "f" for kind, _definition in constraints) or any(
        not any(
            actual_kind == expected_kind and all(fragment in definition for fragment in fragments)
            for actual_kind, definition in constraints
        )
        for expected_kind, fragments in required
    ):
        _raise_knowledge_semantic_watch_schema_error(table)


def _raise_knowledge_semantic_watch_schema_error(name: str) -> NoReturn:
    raise RuntimeError(
        "Postgres schema object "
        f"{name!r} conflicts with Cayu's semantic-watch receipt contract. "
        "Run `cayu storage migrate` to install revision 78 or recreate the database."
    )


async def _validate_knowledge_publication_access_snapshot_column(
    cur: Any,
) -> None:
    await cur.execute(
        """
            SELECT data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'cayu_knowledge_publication_receipts'
              AND column_name = 'access_snapshot'
            """
    )
    if await cur.fetchone() != ("jsonb", "NO"):
        raise RuntimeError(
            "Postgres schema object "
            "'cayu_knowledge_publication_receipts.access_snapshot' conflicts "
            "with Cayu's knowledge authorization contract. Recreate the Cayu "
            "database from a known-good revision-41 schema."
        )


async def _validate_knowledge_revision_schema(
    cur: Any,
    *,
    allow_revision_43: bool = False,
    require_payload_bytes: bool = False,
) -> None:
    expected_columns = {
        "cayu_knowledge_entries": (
            ("id", "text", "NO"),
            ("namespace", "text", "NO"),
            ("current_revision", "integer", "NO"),
            ("created_at", "timestamp with time zone", "NO"),
            ("updated_at", "timestamp with time zone", "NO"),
        ),
        "cayu_knowledge_revisions": (
            ("entry_id", "text", "NO"),
            ("revision", "integer", "NO"),
            ("text", "text", "NO"),
            ("kind", "text", "NO"),
            ("visibility", "text", "NO"),
            ("status", "text", "NO"),
            ("created_by_type", "text", "NO"),
            ("created_by", "text", "NO"),
            ("created_at", "timestamp with time zone", "NO"),
            ("updated_at", "timestamp with time zone", "NO"),
            ("source_type", "text", "YES"),
            ("source_uri", "text", "YES"),
            ("source_id", "text", "YES"),
            ("source_hash", "text", "YES"),
            ("importance", "double precision", "YES"),
            ("importance_source", "text", "YES"),
            ("confidence", "double precision", "YES"),
            ("last_used_at", "timestamp with time zone", "YES"),
            ("expires_at", "timestamp with time zone", "YES"),
            ("title", "text", "YES"),
            ("metadata", "jsonb", "NO"),
            *((("payload_bytes", "bigint", "NO"),) if require_payload_bytes else ()),
        ),
        "cayu_knowledge_labels": (
            ("entry_id", "text", "NO"),
            ("entry_revision", "integer", "NO"),
            ("key", "text", "NO"),
            ("value", "text", "NO"),
        ),
        "cayu_knowledge_aspects": (
            ("entry_id", "text", "NO"),
            ("entry_revision", "integer", "NO"),
            ("aspect", "text", "NO"),
        ),
        "cayu_knowledge_impact_targets": (
            ("entry_id", "text", "NO"),
            ("entry_revision", "integer", "NO"),
            ("impact_target", "text", "NO"),
        ),
        "cayu_knowledge_chunks": (
            ("id", "text", "NO"),
            ("entry_id", "text", "NO"),
            ("entry_revision", "integer", "NO"),
            ("chunk_index", "integer", "NO"),
            ("text", "text", "NO"),
            ("content_hash", "text", "YES"),
            ("source_uri", "text", "YES"),
            ("metadata", "jsonb", "NO"),
        ),
        "cayu_knowledge_publication_receipts": (
            ("operation_id", "text", "NO"),
            ("entry_id", "text", "NO"),
            ("entry_revision", "integer", "NO"),
            ("expected_revision", "integer", "YES"),
            ("request_sha256", "text", "NO"),
            ("entry_created_at", "timestamp with time zone", "NO"),
            ("entry_updated_at", "timestamp with time zone", "NO"),
            ("committed_at", "timestamp with time zone", "NO"),
            ("access_snapshot", "jsonb", "NO"),
        ),
    }
    await cur.execute(
        """
            SELECT table_name, column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = ANY(%s)
            ORDER BY table_name, ordinal_position
            """,
        (list(expected_columns),),
    )
    actual_columns: dict[str, list[tuple[str, str, str]]] = {}
    for table, column, data_type, nullable in await cur.fetchall():
        actual_columns.setdefault(str(table), []).append(
            (str(column), str(data_type), str(nullable))
        )
    for table, columns in expected_columns.items():
        if tuple(actual_columns.get(table, ())) != columns:
            _raise_knowledge_revision_schema_error(table)

    expected_view_columns = (
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
        "metadata",
        *(("payload_bytes",) if require_payload_bytes else ()),
    )
    await cur.execute(
        """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'cayu_knowledge_current_entries'
            ORDER BY ordinal_position
            """
    )
    if tuple(str(row[0]) for row in await cur.fetchall()) != expected_view_columns:
        _raise_knowledge_revision_schema_error("cayu_knowledge_current_entries")
    await cur.execute(
        """
            SELECT pg_get_viewdef(view_record.oid, TRUE)
            FROM pg_catalog.pg_class AS view_record
            JOIN pg_catalog.pg_namespace AS namespace
              ON namespace.oid = view_record.relnamespace
            WHERE namespace.nspname = current_schema()
              AND view_record.relname = 'cayu_knowledge_current_entries'
              AND view_record.relkind = 'v'
            """
    )
    view_row = await cur.fetchone()
    view_sql = " ".join(str(view_row[0]).lower().split()) if view_row is not None else ""
    if (
        "cayu_knowledge_entries logical" not in view_sql
        or "cayu_knowledge_revisions revision" not in view_sql
        or "revision.revision = logical.current_revision" not in view_sql
    ):
        _raise_knowledge_revision_schema_error("cayu_knowledge_current_entries")

    expected_constraints: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
        "cayu_knowledge_entries": (
            ("p", ("primary key (id)",)),
            (
                "f",
                (
                    "foreign key (id, current_revision)",
                    "references cayu_knowledge_revisions(entry_id, revision)",
                    "deferrable initially deferred",
                ),
            ),
            ("c", ("current_revision", "> 0", "2147483647")),
        ),
        "cayu_knowledge_revisions": (
            ("p", ("primary key (entry_id, revision)",)),
            (
                "f",
                (
                    "foreign key (entry_id)",
                    "references cayu_knowledge_entries(id)",
                    "on delete cascade",
                ),
            ),
            ("c", ("revision", "> 0", "2147483647")),
            *((("c", ("payload_bytes", "> 0", "2147483647")),) if require_payload_bytes else ()),
        ),
        "cayu_knowledge_labels": (
            ("p", ("primary key (entry_id, entry_revision, key)",)),
            (
                "f",
                (
                    "foreign key (entry_id, entry_revision)",
                    "references cayu_knowledge_revisions(entry_id, revision)",
                    "on delete cascade",
                ),
            ),
        ),
        "cayu_knowledge_aspects": (
            ("p", ("primary key (entry_id, entry_revision, aspect)",)),
            (
                "f",
                (
                    "foreign key (entry_id, entry_revision)",
                    "references cayu_knowledge_revisions(entry_id, revision)",
                    "on delete cascade",
                ),
            ),
        ),
        "cayu_knowledge_impact_targets": (
            ("p", ("primary key (entry_id, entry_revision, impact_target)",)),
            (
                "f",
                (
                    "foreign key (entry_id, entry_revision)",
                    "references cayu_knowledge_revisions(entry_id, revision)",
                    "on delete cascade",
                ),
            ),
        ),
        "cayu_knowledge_chunks": (
            ("p", ("primary key (id)",)),
            ("u", ("unique (entry_id, entry_revision, chunk_index)",)),
            (
                "f",
                (
                    "foreign key (entry_id, entry_revision)",
                    "references cayu_knowledge_revisions(entry_id, revision)",
                    "on delete cascade",
                ),
            ),
            ("c", ("entry_revision", "> 0", "2147483647")),
            ("c", ("chunk_index", ">= 0")),
        ),
        "cayu_knowledge_publication_receipts": (
            ("p", ("primary key (operation_id)",)),
            ("c", ("entry_revision", "> 0", "2147483647")),
            ("c", ("expected_revision", "> 0", "2147483647")),
            (
                "c",
                (
                    "expected_revision is null",
                    "entry_revision = 1",
                    "entry_revision = (expected_revision + 1)",
                ),
            ),
        ),
    }
    await cur.execute(
        """
            SELECT table_record.relname,
                   constraint_record.contype,
                   pg_get_constraintdef(constraint_record.oid)
            FROM pg_catalog.pg_constraint AS constraint_record
            JOIN pg_catalog.pg_class AS table_record
              ON table_record.oid = constraint_record.conrelid
            JOIN pg_catalog.pg_namespace AS namespace
              ON namespace.oid = table_record.relnamespace
            WHERE namespace.nspname = current_schema()
              AND table_record.relname = ANY(%s)
            ORDER BY table_record.relname, constraint_record.contype,
                     constraint_record.conname
            """,
        (list(expected_constraints),),
    )
    actual_constraints: dict[str, list[tuple[str, str]]] = {}
    for table, constraint_type, definition in await cur.fetchall():
        actual_constraints.setdefault(str(table), []).append(
            (
                str(constraint_type),
                " ".join(str(definition).lower().split()),
            )
        )

    for table, expected in expected_constraints.items():
        remaining = list(actual_constraints.get(table, ()))
        for constraint_type, fragments in expected:
            match = next(
                (
                    candidate
                    for candidate in remaining
                    if candidate[0] == constraint_type
                    and all(fragment in candidate[1] for fragment in fragments)
                ),
                None,
            )
            if match is None:
                _raise_knowledge_revision_schema_error(table)
            remaining.remove(match)
        if allow_revision_43 and table == "cayu_knowledge_chunks":
            revision_43_constraint = next(
                (
                    candidate
                    for candidate in remaining
                    if candidate[0] == "u"
                    and "unique (id, entry_id, entry_revision)" in candidate[1]
                ),
                None,
            )
            if revision_43_constraint is not None:
                remaining.remove(revision_43_constraint)
        if remaining:
            _raise_knowledge_revision_schema_error(table)

    expected_indexes = {
        "idx_cayu_knowledge_entries_namespace_current": (
            "cayu_knowledge_entries",
            "using btree (namespace, current_revision, id)",
        ),
        "idx_cayu_knowledge_revisions_status": (
            "cayu_knowledge_revisions",
            "using btree (status, entry_id, revision)",
        ),
        "idx_cayu_knowledge_revisions_kind": (
            "cayu_knowledge_revisions",
            "using btree (kind, entry_id, revision)",
        ),
        "idx_cayu_knowledge_revisions_visibility": (
            "cayu_knowledge_revisions",
            "using btree (visibility, entry_id, revision)",
        ),
        "idx_cayu_knowledge_revisions_source": (
            "cayu_knowledge_revisions",
            "using btree (source_type, source_id, entry_id, revision)",
        ),
        "idx_cayu_knowledge_revisions_expires_at": (
            "cayu_knowledge_revisions",
            "using btree (expires_at, entry_id, revision)",
        ),
        "idx_cayu_knowledge_revisions_title_fts": (
            "cayu_knowledge_revisions",
            "using gin",
            "to_tsvector",
            "title",
        ),
        "idx_cayu_knowledge_revisions_text_fts": (
            "cayu_knowledge_revisions",
            "using gin",
            "to_tsvector",
            "text",
        ),
        "idx_cayu_knowledge_labels_key_value_entry": (
            "cayu_knowledge_labels",
            "using btree (key, value, entry_id, entry_revision)",
        ),
        "idx_cayu_knowledge_aspects_aspect_entry": (
            "cayu_knowledge_aspects",
            "using btree (aspect, entry_id, entry_revision)",
        ),
        "idx_cayu_knowledge_impact_targets_target_entry": (
            "cayu_knowledge_impact_targets",
            "using btree (impact_target, entry_id, entry_revision)",
        ),
        "idx_cayu_knowledge_chunks_entry_revision_index": (
            "cayu_knowledge_chunks",
            "using btree (entry_id, entry_revision, chunk_index)",
        ),
        "idx_cayu_knowledge_chunks_text_fts": (
            "cayu_knowledge_chunks",
            "using gin",
            "to_tsvector",
            "text",
        ),
        "idx_cayu_knowledge_publication_receipts_entry_revision": (
            "cayu_knowledge_publication_receipts",
            "using btree (entry_id, entry_revision)",
        ),
    }
    await cur.execute(
        """
            SELECT table_record.relname,
                   index_record.relname,
                   index_state.indisvalid,
                   index_state.indisready,
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
    actual_indexes = {
        str(index): (
            str(table),
            bool(valid),
            bool(ready),
            " ".join(str(definition).lower().split()),
        )
        for table, index, valid, ready, definition in await cur.fetchall()
    }
    for index, fragments in expected_indexes.items():
        actual = actual_indexes.get(index)
        if (
            actual is None
            or actual[0] != fragments[0]
            or not actual[1]
            or not actual[2]
            or any(fragment not in actual[3] for fragment in fragments[1:])
        ):
            _raise_knowledge_revision_schema_error(index)


async def _validate_knowledge_change_schema(
    cur: Any,
    *,
    relation_aware: bool = False,
) -> None:
    expected_columns = {
        "cayu_knowledge_evidence": (
            ("id", "text", "NO", "NO"),
            ("entry_id", "text", "NO", "NO"),
            ("entry_revision", "integer", "NO", "NO"),
            ("chunk_id", "text", "YES", "NO"),
            ("role", "text", "NO", "NO"),
            ("source_type", "text", "NO", "NO"),
            ("source_id", "text", "YES", "NO"),
            ("source_uri", "text", "YES", "NO"),
            ("source_revision", "text", "YES", "NO"),
            ("source_hash", "text", "YES", "NO"),
            ("locator", "jsonb", "NO", "NO"),
            ("disposition", "text", "NO", "NO"),
            ("created_at", "timestamp with time zone", "NO", "NO"),
            ("metadata", "jsonb", "NO", "NO"),
        ),
        "cayu_knowledge_changes": (
            ("sequence", "bigint", "NO", "YES"),
            ("id", "text", "NO", "NO"),
            ("kind", "text", "NO", "NO"),
            ("entry_id", "text", "NO", "NO"),
            ("entry_revision", "integer", "NO", "NO"),
            ("committed_at", "timestamp with time zone", "NO", "NO"),
            ("operation_id", "text", "YES", "NO"),
            *(((("relation_id", "text", "YES", "NO"),)) if relation_aware else ()),
        ),
        "cayu_knowledge_change_audiences": (
            ("change_sequence", "bigint", "NO", "NO"),
            ("audience_kind", "text", "NO", "NO"),
            ("namespace", "text", "NO", "NO"),
            ("visibility", "text", "NO", "NO"),
            ("source_type", "text", "YES", "NO"),
            ("source_id", "text", "YES", "NO"),
            ("status", "text", "NO", "NO"),
            ("requires_include_expired", "boolean", "NO", "NO"),
        ),
        "cayu_knowledge_change_consumers": (
            ("consumer_id", "text", "NO", "NO"),
            ("access_scope_sha256", "text", "NO", "NO"),
            ("cursor_sequence", "bigint", "NO", "NO"),
            ("pending_change_sequence", "bigint", "YES", "NO"),
            ("pending_claim_id", "text", "YES", "NO"),
            ("pending_worker_id", "text", "YES", "NO"),
            ("pending_attempt", "integer", "NO", "NO"),
            ("claimed_at", "timestamp with time zone", "YES", "NO"),
            ("lease_expires_at", "timestamp with time zone", "YES", "NO"),
            ("last_acknowledged_claim_id", "text", "YES", "NO"),
            ("updated_at", "timestamp with time zone", "NO", "NO"),
        ),
        "cayu_knowledge_change_acknowledgements": (
            ("consumer_id", "text", "NO", "NO"),
            ("claim_id", "text", "NO", "NO"),
            ("claim_sha256", "text", "NO", "NO"),
            ("change_sequence", "bigint", "NO", "NO"),
            ("acknowledged_at", "timestamp with time zone", "NO", "NO"),
        ),
        "cayu_knowledge_change_labels": (
            ("change_sequence", "bigint", "NO", "NO"),
            ("audience_kind", "text", "NO", "NO"),
            ("key", "text", "NO", "NO"),
            ("value", "text", "NO", "NO"),
        ),
    }
    await cur.execute(
        """
            SELECT table_name, column_name, data_type, is_nullable, is_identity
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = ANY(%s)
            ORDER BY table_name, ordinal_position
            """,
        (list(expected_columns),),
    )
    actual: dict[str, list[tuple[str, str, str, str]]] = {}
    for table, column, data_type, nullable, identity in await cur.fetchall():
        actual.setdefault(str(table), []).append(
            (str(column), str(data_type), str(nullable), str(identity))
        )
    for table, columns in expected_columns.items():
        if tuple(actual.get(table, ())) != columns:
            _raise_knowledge_change_schema_error(table)

    await cur.execute(
        """
            SELECT table_record.relname, constraint_record.contype,
                   pg_get_constraintdef(constraint_record.oid)
            FROM pg_catalog.pg_constraint AS constraint_record
            JOIN pg_catalog.pg_class AS table_record
              ON table_record.oid = constraint_record.conrelid
            JOIN pg_catalog.pg_namespace AS namespace
              ON namespace.oid = table_record.relnamespace
            WHERE namespace.nspname = current_schema()
              AND table_record.relname = ANY(%s)
            """,
        (
            [
                "cayu_knowledge_chunks",
                "cayu_knowledge_evidence",
                "cayu_knowledge_changes",
                "cayu_knowledge_change_audiences",
                "cayu_knowledge_change_consumers",
                "cayu_knowledge_change_acknowledgements",
                "cayu_knowledge_change_labels",
            ],
        ),
    )
    constraints: dict[str, list[tuple[str, str]]] = {}
    for table, kind, definition in await cur.fetchall():
        constraints.setdefault(str(table), []).append(
            (str(kind), " ".join(str(definition).lower().split()))
        )
    required_constraints = {
        "cayu_knowledge_chunks": (("u", ("unique (id, entry_id, entry_revision)",)),),
        "cayu_knowledge_evidence": (
            ("p", ("primary key (id)",)),
            (
                "f",
                (
                    "foreign key (entry_id, entry_revision)",
                    "references cayu_knowledge_revisions(entry_id, revision)",
                    "on delete cascade",
                ),
            ),
            (
                "f",
                (
                    "foreign key (chunk_id, entry_id, entry_revision)",
                    "references cayu_knowledge_chunks(id, entry_id, entry_revision)",
                    "on delete cascade",
                ),
            ),
            ("c", ("entry_revision > 0", "2147483647")),
            ("c", ("source_id is not null", "source_uri is not null")),
            ("c", ("source_revision is not null", "source_hash is not null")),
            ("c", ("role", "origin", "supporting")),
            ("c", ("disposition", "live", "detached", "retained")),
        ),
        "cayu_knowledge_changes": (
            ("p", ("primary key (sequence)",)),
            ("u", ("unique (id)",)),
            ("c", ("sequence > 0",)),
            ("c", ("entry_revision > 0", "2147483647")),
            (
                "c",
                (
                    "kind",
                    "created",
                    "revision_appended",
                    "status_transitioned",
                    "tombstoned",
                    "hard_deleted",
                    "expired",
                    *(("relation_published",) if relation_aware else ()),
                ),
            ),
            *(
                (
                    (
                        "c",
                        (
                            "kind = 'relation_published'",
                            "relation_id is not null",
                            "kind <> 'relation_published'",
                            "relation_id is null",
                        ),
                    ),
                )
                if relation_aware
                else ()
            ),
        ),
        "cayu_knowledge_change_audiences": (
            ("p", ("primary key (change_sequence, audience_kind)",)),
            (
                "f",
                (
                    "foreign key (change_sequence)",
                    "references cayu_knowledge_changes(sequence)",
                    "on delete cascade",
                ),
            ),
            (
                "c",
                (
                    "audience_kind",
                    "before",
                    "after",
                    *(
                        (
                            "subject_exact",
                            "subject_current",
                            "object_exact",
                            "object_current",
                        )
                        if relation_aware
                        else ()
                    ),
                ),
            ),
        ),
        "cayu_knowledge_change_consumers": (
            ("p", ("primary key (consumer_id)",)),
            (
                "f",
                (
                    "foreign key (pending_change_sequence)",
                    "references cayu_knowledge_changes(sequence)",
                ),
            ),
            ("c", ("cursor_sequence >= 0",)),
            ("c", ("pending_attempt >= 0",)),
            (
                "c",
                (
                    "pending_change_sequence > cursor_sequence",
                    "lease_expires_at > claimed_at",
                ),
            ),
        ),
        "cayu_knowledge_change_acknowledgements": (
            ("p", ("primary key (consumer_id, claim_id)",)),
            (
                "f",
                (
                    "foreign key (consumer_id)",
                    "references cayu_knowledge_change_consumers(consumer_id)",
                    "on delete cascade",
                ),
            ),
            (
                "f",
                (
                    "foreign key (change_sequence)",
                    "references cayu_knowledge_changes(sequence)",
                ),
            ),
            ("c", ("claim_sha256", "[0-9a-f]{64}")),
        ),
        "cayu_knowledge_change_labels": (
            ("p", ("primary key (change_sequence, audience_kind, key)",)),
            (
                "f",
                (
                    "foreign key (change_sequence, audience_kind)",
                    "references cayu_knowledge_change_audiences(change_sequence, audience_kind)",
                    "on delete cascade",
                ),
            ),
        ),
    }
    for table, required in required_constraints.items():
        candidates = list(constraints.get(table, []))
        for kind, fragments in required:
            match = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate[0] == kind
                    and all(fragment in candidate[1] for fragment in fragments)
                ),
                None,
            )
            if match is None:
                _raise_knowledge_change_schema_error(table)
            candidates.remove(match)
        if table != "cayu_knowledge_chunks" and candidates:
            _raise_knowledge_change_schema_error(table)

    expected_indexes = {
        "idx_cayu_knowledge_evidence_entry_revision": (
            "cayu_knowledge_evidence",
            'using btree (entry_id, entry_revision, id collate "c")',
        ),
        "idx_cayu_knowledge_evidence_source": (
            "cayu_knowledge_evidence",
            "using btree (source_type, source_id, entry_id, entry_revision)",
        ),
        "idx_cayu_knowledge_changes_entry_revision": (
            "cayu_knowledge_changes",
            "using btree (entry_id, entry_revision, sequence)",
        ),
        "idx_cayu_knowledge_change_audiences_namespace": (
            "cayu_knowledge_change_audiences",
            "using btree (namespace, change_sequence, audience_kind)",
        ),
        "idx_cayu_knowledge_change_audiences_status": (
            "cayu_knowledge_change_audiences",
            "using btree (status, change_sequence, audience_kind)",
        ),
        "idx_cayu_knowledge_change_audiences_source": (
            "cayu_knowledge_change_audiences",
            "using btree (source_type, source_id, change_sequence, audience_kind)",
        ),
        "idx_cayu_knowledge_changes_operation": (
            "cayu_knowledge_changes",
            (
                "using btree (operation_id, sequence)"
                if relation_aware
                else "using btree (operation_id)"
            ),
            "where (operation_id is not null)",
        ),
        "idx_cayu_knowledge_change_consumers_lease": (
            "cayu_knowledge_change_consumers",
            "using btree (lease_expires_at)",
            "where (pending_change_sequence is not null)",
        ),
        "idx_cayu_knowledge_change_labels_lookup": (
            "cayu_knowledge_change_labels",
            "using btree (key, value, change_sequence, audience_kind)",
        ),
    }
    await cur.execute(
        """
            SELECT table_record.relname, index_record.relname,
                   index_state.indisvalid, index_state.indisready,
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
            " ".join(str(definition).lower().split()),
        )
        for table, index, valid, ready, definition in await cur.fetchall()
    }
    for name, fragments in expected_indexes.items():
        value = indexes.get(name)
        if (
            value is None
            or value[0] != fragments[0]
            or not value[1]
            or not value[2]
            or any(fragment not in value[3] for fragment in fragments[1:])
        ):
            _raise_knowledge_change_schema_error(name)


async def _validate_knowledge_relation_schema(cur: Any) -> None:
    expected_columns = {
        "cayu_knowledge_relations": (
            ("id", "text", "NO"),
            ("subject_entry_id", "text", "NO"),
            ("subject_revision", "integer", "NO"),
            ("object_entry_id", "text", "NO"),
            ("object_revision", "integer", "NO"),
            ("kind", "text", "NO"),
            ("created_by_type", "text", "NO"),
            ("created_by", "text", "NO"),
            ("policy_id", "text", "YES"),
            ("created_at", "timestamp with time zone", "NO"),
            ("metadata", "jsonb", "NO"),
        ),
        "cayu_knowledge_relation_publication_receipts": (
            ("operation_id", "text", "NO"),
            ("relation_ids", "jsonb", "NO"),
            ("request_sha256", "text", "NO"),
            ("committed_at", "timestamp with time zone", "NO"),
            ("access_snapshots", "jsonb", "NO"),
        ),
    }
    await cur.execute(
        """
            SELECT table_name, column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = ANY(%s)
            ORDER BY table_name, ordinal_position
            """,
        (list(expected_columns),),
    )
    actual: dict[str, list[tuple[str, str, str]]] = {}
    for table, column, data_type, nullable in await cur.fetchall():
        actual.setdefault(str(table), []).append((str(column), str(data_type), str(nullable)))
    for table, columns in expected_columns.items():
        if tuple(actual.get(table, ())) != columns:
            _raise_knowledge_relation_schema_error(table)

    await cur.execute(
        """
            SELECT table_record.relname, constraint_record.contype,
                   pg_get_constraintdef(constraint_record.oid)
            FROM pg_catalog.pg_constraint AS constraint_record
            JOIN pg_catalog.pg_class AS table_record
              ON table_record.oid = constraint_record.conrelid
            JOIN pg_catalog.pg_namespace AS namespace
              ON namespace.oid = table_record.relnamespace
            WHERE namespace.nspname = current_schema()
              AND table_record.relname = ANY(%s)
            """,
        (list(expected_columns),),
    )
    constraints: dict[str, list[tuple[str, str]]] = {}
    for table, kind, definition in await cur.fetchall():
        constraints.setdefault(str(table), []).append(
            (str(kind), " ".join(str(definition).lower().split()))
        )
    required_constraints = {
        "cayu_knowledge_relations": (
            ("p", ("primary key (id)",)),
            (
                "u",
                (
                    "unique (kind, subject_entry_id, subject_revision, "
                    "object_entry_id, object_revision)",
                ),
            ),
            (
                "f",
                (
                    "foreign key (subject_entry_id, subject_revision)",
                    "references cayu_knowledge_revisions(entry_id, revision)",
                    "on delete cascade",
                ),
            ),
            (
                "f",
                (
                    "foreign key (object_entry_id, object_revision)",
                    "references cayu_knowledge_revisions(entry_id, revision)",
                    "on delete cascade",
                ),
            ),
            ("c", ("subject_revision > 0", "2147483647")),
            ("c", ("object_revision > 0", "2147483647")),
            ("c", ("subject_entry_id <> object_entry_id",)),
            (
                "c",
                (
                    "kind <> 'contradicts'",
                    'subject_entry_id collate "c"',
                    'object_entry_id collate "c"',
                ),
            ),
            ("c", ("kind", "supersedes", "derived_from", "contradicts")),
        ),
        "cayu_knowledge_relation_publication_receipts": (
            ("p", ("primary key (operation_id)",)),
            ("c", ("request_sha256", "[0-9a-f]{64}")),
            ("c", ("jsonb_typeof(relation_ids)", "array")),
            ("c", ("jsonb_typeof(access_snapshots)", "array")),
        ),
    }
    for table, required in required_constraints.items():
        candidates = constraints.get(table, [])
        for kind, fragments in required:
            if not any(
                candidate_kind == kind and all(fragment in definition for fragment in fragments)
                for candidate_kind, definition in candidates
            ):
                _raise_knowledge_relation_schema_error(table)

    expected_indexes = {
        "idx_cayu_knowledge_relations_subject": (
            "cayu_knowledge_relations",
            'using btree (subject_entry_id, subject_revision, created_at, id collate "c")',
        ),
        "idx_cayu_knowledge_relations_object": (
            "cayu_knowledge_relations",
            'using btree (object_entry_id, object_revision, created_at, id collate "c")',
        ),
        "idx_cayu_knowledge_relations_subject_kind": (
            "cayu_knowledge_relations",
            'using btree (subject_entry_id, subject_revision, kind, created_at, id collate "c")',
        ),
        "idx_cayu_knowledge_relations_object_kind": (
            "cayu_knowledge_relations",
            'using btree (object_entry_id, object_revision, kind, created_at, id collate "c")',
        ),
        "idx_cayu_knowledge_changes_relation": (
            "cayu_knowledge_changes",
            "using btree (relation_id)",
            "where (relation_id is not null)",
        ),
    }
    await cur.execute(
        """
            SELECT table_record.relname, index_record.relname,
                   index_state.indisvalid, index_state.indisready,
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
            " ".join(str(definition).lower().split()),
        )
        for table, index, valid, ready, definition in await cur.fetchall()
    }
    for name, fragments in expected_indexes.items():
        value = indexes.get(name)
        if (
            value is None
            or value[0] != fragments[0]
            or not value[1]
            or not value[2]
            or any(fragment not in value[3] for fragment in fragments[1:])
        ):
            _raise_knowledge_relation_schema_error(name)


def _raise_knowledge_relation_schema_error(name: str) -> NoReturn:
    raise RuntimeError(
        "Postgres schema object "
        f"{name!r} conflicts with Cayu's revision-bound knowledge relation "
        "contract. Recreate the prerelease knowledge schema with "
        "schema_mode=CREATE or MIGRATE."
    )


async def _validate_knowledge_maintenance_schema(cur: Any) -> None:
    table = "cayu_knowledge_maintenance_decisions"
    await cur.execute(
        """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = current_schema() AND table_name = %s
            ORDER BY ordinal_position
            """,
        (table,),
    )
    expected = (
        ("operation_id", "text", "NO"),
        ("proposal_id", "text", "NO"),
        ("proposal_fingerprint", "text", "NO"),
        ("request_sha256", "text", "NO"),
        ("committed_at", "timestamp with time zone", "NO"),
        ("proposal", "jsonb", "NO"),
        ("decision", "jsonb", "NO"),
        ("receipt", "jsonb", "NO"),
        ("access_snapshot", "jsonb", "NO"),
    )
    if tuple(await cur.fetchall()) != expected:
        _raise_knowledge_maintenance_schema_error(table)
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
        (table,),
    )
    constraints = [
        (str(kind), " ".join(str(definition).lower().split()))
        for kind, definition in await cur.fetchall()
    ]
    required = (
        ("p", ("primary key (operation_id)",)),
        ("u", ("unique (proposal_id)",)),
        ("c", ("proposal_fingerprint", "[0-9a-f]{64}")),
        ("c", ("request_sha256", "[0-9a-f]{64}")),
        ("c", ("jsonb_typeof(proposal)", "object")),
        ("c", ("jsonb_typeof(decision)", "object")),
        ("c", ("jsonb_typeof(receipt)", "object")),
        ("c", ("jsonb_typeof(access_snapshot)", "object")),
    )
    for kind, fragments in required:
        if not any(
            candidate_kind == kind and all(fragment in definition for fragment in fragments)
            for candidate_kind, definition in constraints
        ):
            _raise_knowledge_maintenance_schema_error(table)


def _raise_knowledge_maintenance_schema_error(name: str) -> NoReturn:
    raise RuntimeError(
        "Postgres schema object "
        f"{name!r} conflicts with Cayu's reviewed knowledge maintenance "
        "contract. Recreate the prerelease knowledge schema with "
        "schema_mode=CREATE or MIGRATE."
    )


async def _validate_knowledge_maintenance_proposal_schema(cur: Any) -> None:
    table = "cayu_knowledge_maintenance_proposals"
    await cur.execute(
        """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = current_schema() AND table_name = %s
            ORDER BY ordinal_position
            """,
        (table,),
    )
    expected = (
        ("operation_id", "text", "NO"),
        ("proposal_id", "text", "NO"),
        ("replacement_entry_id", "text", "NO"),
        ("replacement_revision", "integer", "NO"),
        ("proposal_fingerprint", "text", "NO"),
        ("accepted_plan_fingerprint", "text", "NO"),
        ("request_sha256", "text", "NO"),
        ("committed_at", "timestamp with time zone", "NO"),
        ("proposal", "jsonb", "NO"),
        ("accepted_plan", "jsonb", "NO"),
        ("receipt", "jsonb", "NO"),
        ("access_snapshot", "jsonb", "NO"),
    )
    if tuple(await cur.fetchall()) != expected:
        _raise_knowledge_maintenance_proposal_schema_error(table)
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
        (table,),
    )
    constraints = [
        (str(kind), " ".join(str(definition).lower().split()))
        for kind, definition in await cur.fetchall()
    ]
    required = (
        ("p", ("primary key (operation_id)",)),
        ("u", ("unique (proposal_id)",)),
        ("u", ("unique (replacement_entry_id)",)),
        ("c", ("replacement_revision", "> 0", "2147483647")),
        ("c", ("proposal_fingerprint", "[0-9a-f]{64}")),
        ("c", ("accepted_plan_fingerprint", "[0-9a-f]{64}")),
        ("c", ("request_sha256", "[0-9a-f]{64}")),
        ("c", ("jsonb_typeof(proposal)", "object")),
        ("c", ("jsonb_typeof(accepted_plan)", "object")),
        ("c", ("jsonb_typeof(receipt)", "object")),
        ("c", ("jsonb_typeof(access_snapshot)", "object")),
    )
    for kind, fragments in required:
        if not any(
            candidate_kind == kind and all(fragment in definition for fragment in fragments)
            for candidate_kind, definition in constraints
        ):
            _raise_knowledge_maintenance_proposal_schema_error(table)


def _raise_knowledge_maintenance_proposal_schema_error(name: str) -> NoReturn:
    raise RuntimeError(
        "Postgres schema object "
        f"{name!r} conflicts with Cayu's pending maintenance proposal "
        "contract. Recreate the prerelease knowledge schema with "
        "schema_mode=CREATE or MIGRATE."
    )


async def _validate_knowledge_index_readiness_schema(cur: Any) -> None:
    expected_columns = {
        "cayu_knowledge_index_readiness_events": (
            ("sequence", "bigint", "NO", "ALWAYS"),
            ("identity_sha256", "text", "NO", None),
            ("entry_id", "text", "NO", None),
            ("entry_revision", "integer", "NO", None),
            ("chunk_id", "text", "YES", None),
            ("projection_type", "text", "NO", None),
            ("projection_content_hash", "text", "NO", None),
            ("embedding_model", "text", "NO", None),
            ("dimensions", "integer", "NO", None),
            ("preprocessing_version", "text", "NO", None),
            ("generator", "text", "NO", None),
            ("generator_version", "text", "NO", None),
            ("index_representation_version", "text", "NO", None),
            ("state", "text", "NO", None),
            ("attempt_id", "text", "NO", None),
            ("failure_code", "text", "YES", None),
            ("operation_id", "text", "NO", None),
            ("update_sha256", "text", "NO", None),
            ("published_at", "timestamp with time zone", "NO", None),
        ),
        "cayu_knowledge_index_readiness_current": (
            ("identity_sha256", "text", "NO", None),
            ("sequence", "bigint", "NO", None),
        ),
    }
    await cur.execute(
        """
            SELECT table_name, column_name, data_type, is_nullable,
                   identity_generation
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = ANY(%s)
            ORDER BY table_name, ordinal_position
            """,
        (list(expected_columns),),
    )
    actual_columns: dict[str, list[tuple[str, str, str, str | None]]] = {
        table: [] for table in expected_columns
    }
    for table, column, data_type, nullable, identity_generation in await cur.fetchall():
        actual_columns[str(table)].append(
            (
                str(column),
                str(data_type),
                str(nullable),
                None if identity_generation is None else str(identity_generation),
            )
        )
    for table, expected in expected_columns.items():
        if tuple(actual_columns[table]) != expected:
            _raise_knowledge_index_readiness_schema_error(table)

    await cur.execute(
        """
            SELECT table_record.relname, constraint_record.contype,
                   pg_get_constraintdef(constraint_record.oid)
            FROM pg_catalog.pg_constraint AS constraint_record
            JOIN pg_catalog.pg_class AS table_record
              ON table_record.oid = constraint_record.conrelid
            JOIN pg_catalog.pg_namespace AS namespace
              ON namespace.oid = table_record.relnamespace
            WHERE namespace.nspname = current_schema()
              AND table_record.relname = ANY(%s)
            """,
        (list(expected_columns),),
    )
    constraints: dict[str, list[tuple[str, str]]] = {table: [] for table in expected_columns}
    for table, kind, definition in await cur.fetchall():
        constraints[str(table)].append((str(kind), " ".join(str(definition).lower().split())))
    required_constraints = {
        "cayu_knowledge_index_readiness_events": (
            ("p", ("primary key (sequence)",)),
            ("u", ("unique (operation_id)",)),
            ("u", ("unique (identity_sha256, sequence)",)),
            ("c", ("sequence > 0",)),
            ("c", ("identity_sha256", "[0-9a-f]{64}")),
            ("c", ("entry_revision > 0", "entry_revision <= 2147483647")),
            ("c", ("dimensions > 0",)),
            ("c", ("state", "pending", "ready", "failed")),
            ("c", ("update_sha256", "[0-9a-f]{64}")),
            (
                "c",
                (
                    "state = 'failed'",
                    "failure_code is not null",
                    "state <> 'failed'",
                    "failure_code is null",
                ),
            ),
        ),
        "cayu_knowledge_index_readiness_current": (
            ("p", ("primary key (identity_sha256)",)),
            ("u", ("unique (sequence)",)),
            (
                "f",
                (
                    "foreign key (identity_sha256, sequence)",
                    "references cayu_knowledge_index_readiness_events(identity_sha256, sequence)",
                    "on delete cascade",
                ),
            ),
        ),
    }
    for table, required in required_constraints.items():
        candidates = constraints[table]
        for kind, fragments in required:
            if not any(
                candidate_kind == kind and all(fragment in definition for fragment in fragments)
                for candidate_kind, definition in candidates
            ):
                _raise_knowledge_index_readiness_schema_error(table)

    expected_indexes = {
        "idx_cayu_knowledge_index_readiness_identity_sequence": (
            "cayu_knowledge_index_readiness_events",
            "using btree (identity_sha256, sequence)",
        ),
        "idx_cayu_knowledge_index_readiness_entry_revision": (
            "cayu_knowledge_index_readiness_events",
            "using btree (entry_id, entry_revision, projection_type, sequence)",
        ),
        "idx_cayu_knowledge_index_readiness_projection_lookup": (
            "cayu_knowledge_index_readiness_events",
            "using btree (entry_id, entry_revision, chunk_id, projection_type, "
            "embedding_model, dimensions, sequence)",
        ),
    }
    await cur.execute(
        """
            SELECT table_record.relname, index_record.relname,
                   index_state.indisvalid, index_state.indisready,
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
            " ".join(str(definition).lower().split()),
        )
        for table, index, valid, ready, definition in await cur.fetchall()
    }
    for name, fragments in expected_indexes.items():
        value = indexes.get(name)
        if (
            value is None
            or value[0] != fragments[0]
            or not value[1]
            or not value[2]
            or fragments[1] not in value[3]
        ):
            _raise_knowledge_index_readiness_schema_error(name)


def _raise_knowledge_index_readiness_schema_error(name: str) -> NoReturn:
    raise RuntimeError(
        f"Postgres schema object {name!r} conflicts with Cayu's derived-index "
        "identity and readiness contract. Recreate or migrate the Cayu "
        "database from a known-good revision-44 schema."
    )


def _raise_knowledge_change_schema_error(name: str) -> NoReturn:
    raise RuntimeError(
        f"Postgres schema object {name!r} conflicts with Cayu's knowledge "
        "evidence and atomic change contract. Recreate or migrate the Cayu "
        "database from a known-good revision-43 schema."
    )


def _raise_knowledge_revision_schema_error(name: str) -> NoReturn:
    raise RuntimeError(
        f"Postgres schema object {name!r} conflicts with Cayu's "
        "revision-first knowledge contract. Recreate the Cayu database "
        "from a known-good revision-42 schema."
    )
