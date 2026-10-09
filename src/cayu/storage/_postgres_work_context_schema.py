"""PostgreSQL schema checks for agent work contexts and recall delivery.

Callers retain revision gates, validation order and the existing cursor/transaction.
These functions inspect the schema without acquiring connections or applying DDL.
"""

from __future__ import annotations

from typing import Any, NoReturn


async def _validate_agent_work_context_schema(cur: Any) -> None:
    expected_columns = {
        "cayu_agent_work_context_revisions": (
            ("task_id", "text", "NO", "C"),
            ("revision", "integer", "NO", None),
            ("content_sha256", "text", "NO", "C"),
            ("operation_id", "text", "NO", "C"),
            ("record_json", "jsonb", "NO", None),
            ("published_at", "timestamp with time zone", "NO", None),
        ),
        "cayu_agent_work_context_heads": (
            ("task_id", "text", "NO", "C"),
            ("current_revision", "integer", "NO", None),
        ),
        "cayu_agent_work_context_publications": (
            ("operation_id", "text", "NO", "C"),
            ("task_id", "text", "NO", "C"),
            ("request_sha256", "text", "NO", "C"),
            ("context_revision", "integer", "NO", None),
            ("changed", "boolean", "NO", None),
            ("receipt_json", "jsonb", "NO", None),
            ("committed_at", "timestamp with time zone", "NO", None),
        ),
        "cayu_agent_recall_checkpoints": (
            ("agent_id", "text", "NO", "C"),
            ("task_id", "text", "NO", "C"),
            ("knowledge_namespace", "text", "NO", "C"),
            ("access_policy_sha256", "text", "NO", "C"),
            ("checkpoint_stream_id", "text", "NO", "C"),
            ("revision", "integer", "NO", None),
            ("work_context_revision", "integer", "NO", None),
            ("work_context_sha256", "text", "NO", "C"),
            ("knowledge_sequence", "bigint", "NO", None),
            ("index_readiness_sequence", "bigint", "NO", None),
            ("knowledge_high_water_sequence", "bigint", "NO", None),
            ("index_readiness_high_water_sequence", "bigint", "NO", None),
            ("processing_mode", "text", "NO", "C"),
            ("processing_id", "text", "NO", "C"),
            ("operation_id", "text", "NO", "C"),
            ("record_json", "jsonb", "NO", None),
            ("updated_at", "timestamp with time zone", "NO", None),
        ),
        "cayu_agent_recall_checkpoint_heads": (
            ("agent_id", "text", "NO", "C"),
            ("task_id", "text", "NO", "C"),
            ("knowledge_namespace", "text", "NO", "C"),
            ("access_policy_sha256", "text", "NO", "C"),
            ("checkpoint_stream_id", "text", "NO", "C"),
            ("current_revision", "integer", "NO", None),
        ),
    }
    for table, expected in expected_columns.items():
        await cur.execute(
            """
                SELECT column_name, data_type, is_nullable, collation_name
                FROM information_schema.columns
                WHERE table_schema = current_schema() AND table_name = %s
                ORDER BY ordinal_position
                """,
            (table,),
        )
        if tuple(await cur.fetchall()) != expected:
            _raise_agent_work_context_schema_error(table)

    required_constraints = {
        "cayu_agent_work_context_revisions": (
            ("p", ("primary key (task_id, revision)",), ("task_id", "revision")),
            ("u", ("unique (operation_id)",), ("operation_id",)),
            ("c", ("revision", "> 0", "2147483647"), ("revision",)),
            ("c", ("content_sha256", "[0-9a-f]{64}"), ("content_sha256",)),
            ("c", ("jsonb_typeof(record_json)", "object"), ("record_json",)),
        ),
        "cayu_agent_work_context_heads": (
            ("p", ("primary key (task_id)",), ("task_id",)),
            (
                "c",
                ("current_revision", "> 0", "2147483647"),
                ("current_revision",),
            ),
            (
                "f",
                (
                    "foreign key (task_id, current_revision) references ",
                    "cayu_agent_work_context_revisions(task_id, revision)",
                    "on delete restrict",
                ),
                ("task_id", "current_revision"),
            ),
        ),
        "cayu_agent_work_context_publications": (
            ("p", ("primary key (operation_id)",), ("operation_id",)),
            (
                "f",
                (
                    "foreign key (task_id, context_revision) references ",
                    "cayu_agent_work_context_revisions(task_id, revision)",
                    "on delete restrict",
                ),
                ("task_id", "context_revision"),
            ),
            ("c", ("request_sha256", "[0-9a-f]{64}"), ("request_sha256",)),
            (
                "c",
                ("context_revision", "> 0", "2147483647"),
                ("context_revision",),
            ),
            ("c", ("jsonb_typeof(receipt_json)", "object"), ("receipt_json",)),
        ),
        "cayu_agent_recall_checkpoints": (
            (
                "p",
                (
                    "primary key (agent_id, task_id, knowledge_namespace, ",
                    "access_policy_sha256, checkpoint_stream_id, revision)",
                ),
                (
                    "agent_id",
                    "task_id",
                    "knowledge_namespace",
                    "access_policy_sha256",
                    "checkpoint_stream_id",
                    "revision",
                ),
            ),
            ("u", ("unique (operation_id)",), ("operation_id",)),
            (
                "f",
                (
                    "foreign key (task_id, work_context_revision) references ",
                    "cayu_agent_work_context_revisions(task_id, revision)",
                    "on delete restrict",
                ),
                ("task_id", "work_context_revision"),
            ),
            ("c", ("revision", "> 0", "2147483647"), ("revision",)),
            (
                "c",
                ("work_context_revision", "> 0", "2147483647"),
                ("work_context_revision",),
            ),
            (
                "c",
                ("access_policy_sha256", "[0-9a-f]{64}"),
                ("access_policy_sha256",),
            ),
            (
                "c",
                ("work_context_sha256", "[0-9a-f]{64}"),
                ("work_context_sha256",),
            ),
            (
                "c",
                ("knowledge_sequence", ">= 0", "9223372036854775807"),
                ("knowledge_sequence",),
            ),
            (
                "c",
                ("index_readiness_sequence", ">= 0", "9223372036854775807"),
                ("index_readiness_sequence",),
            ),
            (
                "c",
                ("knowledge_high_water_sequence", ">= 0", "9223372036854775807"),
                ("knowledge_high_water_sequence",),
            ),
            (
                "c",
                (
                    "index_readiness_high_water_sequence",
                    ">= 0",
                    "9223372036854775807",
                ),
                ("index_readiness_high_water_sequence",),
            ),
            (
                "c",
                ("knowledge_sequence", "<=", "knowledge_high_water_sequence"),
                ("knowledge_sequence", "knowledge_high_water_sequence"),
            ),
            (
                "c",
                (
                    "index_readiness_sequence",
                    "<=",
                    "index_readiness_high_water_sequence",
                ),
                ("index_readiness_sequence", "index_readiness_high_water_sequence"),
            ),
            ("c", ("processing_mode", "full_index", "delta"), ("processing_mode",)),
            ("c", ("jsonb_typeof(record_json)", "object"), ("record_json",)),
        ),
        "cayu_agent_recall_checkpoint_heads": (
            (
                "p",
                (
                    "primary key (agent_id, task_id, knowledge_namespace, ",
                    "access_policy_sha256, checkpoint_stream_id)",
                ),
                (
                    "agent_id",
                    "task_id",
                    "knowledge_namespace",
                    "access_policy_sha256",
                    "checkpoint_stream_id",
                ),
            ),
            (
                "c",
                ("current_revision", "> 0", "2147483647"),
                ("current_revision",),
            ),
            (
                "f",
                (
                    "foreign key (agent_id, task_id, knowledge_namespace, ",
                    "access_policy_sha256, checkpoint_stream_id, ",
                    "current_revision) references ",
                    "cayu_agent_recall_checkpoints(agent_id, task_id, ",
                    "knowledge_namespace, access_policy_sha256, ",
                    "checkpoint_stream_id, revision)",
                    "on delete restrict",
                ),
                (
                    "agent_id",
                    "task_id",
                    "knowledge_namespace",
                    "access_policy_sha256",
                    "checkpoint_stream_id",
                    "current_revision",
                ),
            ),
        ),
    }
    for table, required in required_constraints.items():
        await cur.execute(
            """
                SELECT constraint_record.contype,
                       pg_get_constraintdef(constraint_record.oid),
                       constraint_record.conkey
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
        actual = tuple(
            (
                str(kind),
                " ".join(str(definition).lower().split()),
                tuple(int(column) for column in (constrained_columns or ())),
            )
            for kind, definition, constrained_columns in await cur.fetchall()
        )
        column_numbers = {
            column[0]: index for index, column in enumerate(expected_columns[table], start=1)
        }
        for expected_kind, fragments, constrained_column_names in required:
            expected_conkey = tuple(
                column_numbers[column_name] for column_name in constrained_column_names
            )
            if not any(
                kind == expected_kind
                and conkey == expected_conkey
                and all(fragment in definition for fragment in fragments)
                for kind, definition, conkey in actual
            ):
                _raise_agent_work_context_schema_error(table)


def _raise_agent_work_context_schema_error(name: str) -> NoReturn:
    raise RuntimeError(
        "Postgres schema object "
        f"{name!r} conflicts with Cayu's agent work-context/checkpoint contract. "
        "Run schema_mode=MIGRATE to install the additive revision or recreate "
        "the database."
    )


async def _validate_agent_recall_delivery_schema(
    cur: Any,
    *,
    require_processing_schema_version: bool = False,
) -> None:
    delivery_columns = (
        ("delivery_id", "text", "NO", "C"),
        ("operation_id", "text", "NO", "C"),
        ("agent_id", "text", "NO", "C"),
        ("task_id", "text", "NO", "C"),
        ("knowledge_namespace", "text", "NO", "C"),
        ("access_policy_sha256", "text", "NO", "C"),
        ("checkpoint_stream_id", "text", "NO", "C"),
        ("checkpoint_revision", "integer", "NO", None),
        ("processing_result_sha256", "text", "NO", "C"),
        ("delivery_json", "jsonb", "NO", None),
        ("staged_at", "timestamp with time zone", "NO", None),
    )
    delivery_columns_with_processing_schema = (
        *delivery_columns,
        ("processing_schema_version", "text", "NO", "C"),
    )
    expected_columns = {
        "cayu_agent_recall_deliveries": (
            delivery_columns_with_processing_schema
            if require_processing_schema_version
            else delivery_columns
        ),
        "cayu_agent_recall_delivery_claims": (
            ("claim_id", "text", "NO", "C"),
            ("delivery_id", "text", "NO", "C"),
            ("worker_id", "text", "NO", "C"),
            ("request_sha256", "text", "NO", "C"),
            ("attempt", "bigint", "NO", None),
            ("claimed_at", "timestamp with time zone", "NO", None),
        ),
        "cayu_agent_recall_delivery_releases": (
            ("release_id", "text", "NO", "C"),
            ("delivery_id", "text", "NO", "C"),
            ("claim_id", "text", "NO", "C"),
            ("request_sha256", "text", "NO", "C"),
            ("release_json", "jsonb", "NO", None),
            ("released_at", "timestamp with time zone", "NO", None),
        ),
        "cayu_agent_recall_delivery_states": (
            ("delivery_id", "text", "NO", "C"),
            ("agent_id", "text", "NO", "C"),
            ("task_id", "text", "NO", "C"),
            ("knowledge_namespace", "text", "NO", "C"),
            ("access_policy_sha256", "text", "NO", "C"),
            ("checkpoint_stream_id", "text", "NO", "C"),
            ("checkpoint_revision", "integer", "NO", None),
            ("state", "text", "NO", "C"),
            ("attempt", "bigint", "NO", None),
            ("state_revision", "bigint", "NO", None),
            ("lease_expires_at", "timestamp with time zone", "YES", None),
            ("release_id", "text", "YES", "C"),
            ("acknowledgement_id", "text", "YES", "C"),
            ("state_json", "jsonb", "NO", None),
            ("updated_at", "timestamp with time zone", "NO", None),
        ),
    }
    validate_processing_schema_version = require_processing_schema_version
    for table, expected in expected_columns.items():
        await cur.execute(
            """
                SELECT column_name, data_type, is_nullable, collation_name
                FROM information_schema.columns
                WHERE table_schema = current_schema() AND table_name = %s
                ORDER BY ordinal_position
                """,
            (table,),
        )
        actual = tuple(await cur.fetchall())
        if (
            table == "cayu_agent_recall_deliveries"
            and not require_processing_schema_version
            and actual == delivery_columns_with_processing_schema
        ):
            validate_processing_schema_version = True
            continue
        if actual != expected:
            _raise_agent_recall_delivery_schema_error(table)

    required_constraints = {
        "cayu_agent_recall_deliveries": (
            ("p", ("primary key (delivery_id)",)),
            ("u", ("unique (operation_id)",)),
            (
                "u",
                (
                    "unique (agent_id, task_id, knowledge_namespace, ",
                    "access_policy_sha256, checkpoint_stream_id, ",
                    "checkpoint_revision)",
                ),
            ),
            ("c", ("access_policy_sha256", "[0-9a-f]{64}")),
            ("c", ("checkpoint_revision", "> 0", "2147483647")),
            ("c", ("processing_result_sha256", "[0-9a-f]{64}")),
            ("c", ("jsonb_typeof(delivery_json)", "object")),
            (
                "f",
                (
                    "foreign key (agent_id, task_id, knowledge_namespace, ",
                    "access_policy_sha256, checkpoint_stream_id, ",
                    "checkpoint_revision) references ",
                    "cayu_agent_recall_checkpoints(agent_id, task_id, ",
                    "knowledge_namespace, access_policy_sha256, ",
                    "checkpoint_stream_id, revision)",
                    "on delete restrict",
                ),
            ),
            (
                "f",
                (
                    "foreign key (operation_id) references ",
                    "cayu_agent_recall_checkpoints(operation_id)",
                    "on delete restrict",
                ),
            ),
        ),
        "cayu_agent_recall_delivery_claims": (
            ("p", ("primary key (claim_id)",)),
            ("u", ("unique (delivery_id, attempt)",)),
            ("c", ("request_sha256", "[0-9a-f]{64}")),
            ("c", ("attempt", "> 0", "9223372036854775807")),
            (
                "f",
                (
                    "foreign key (delivery_id) references ",
                    "cayu_agent_recall_deliveries(delivery_id)",
                    "on delete restrict",
                ),
            ),
        ),
        "cayu_agent_recall_delivery_releases": (
            ("p", ("primary key (release_id)",)),
            ("c", ("request_sha256", "[0-9a-f]{64}")),
            ("c", ("jsonb_typeof(release_json)", "object")),
            (
                "f",
                (
                    "foreign key (delivery_id) references ",
                    "cayu_agent_recall_deliveries(delivery_id)",
                    "on delete restrict",
                ),
            ),
            (
                "f",
                (
                    "foreign key (claim_id) references ",
                    "cayu_agent_recall_delivery_claims(claim_id)",
                    "on delete restrict",
                ),
            ),
        ),
        "cayu_agent_recall_delivery_states": (
            ("p", ("primary key (delivery_id)",)),
            ("u", ("unique (release_id)",)),
            ("u", ("unique (acknowledgement_id)",)),
            ("c", ("checkpoint_revision", "> 0", "2147483647")),
            ("c", ("state", "pending", "claimed", "acknowledged")),
            ("c", ("attempt", ">= 0", "9223372036854775807")),
            ("c", ("state_revision", ">= 0", "9223372036854775807")),
            ("c", ("state = 'pending'", "state = 'claimed'", "state = 'acknowledged'")),
            ("c", ("jsonb_typeof(state_json)", "object")),
            (
                "f",
                (
                    "foreign key (delivery_id) references ",
                    "cayu_agent_recall_deliveries(delivery_id)",
                    "on delete restrict",
                ),
            ),
            (
                "f",
                (
                    "foreign key (release_id) references ",
                    "cayu_agent_recall_delivery_releases(release_id)",
                    "on delete restrict",
                ),
            ),
            (
                "f",
                (
                    "foreign key (agent_id, task_id, knowledge_namespace, ",
                    "access_policy_sha256, checkpoint_stream_id, ",
                    "checkpoint_revision) references ",
                    "cayu_agent_recall_deliveries(agent_id, task_id, ",
                    "knowledge_namespace, access_policy_sha256, ",
                    "checkpoint_stream_id, checkpoint_revision)",
                    "on delete restrict",
                ),
            ),
        ),
    }
    if validate_processing_schema_version:
        required_constraints["cayu_agent_recall_deliveries"] += (
            (
                "c",
                (
                    "processing_schema_version",
                    "cayu.agent_recall_processing.v3",
                ),
            ),
        )
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
                """,
            (table,),
        )
        actual = tuple(
            (str(kind), " ".join(str(definition).lower().split()))
            for kind, definition in await cur.fetchall()
        )
        for expected_kind, fragments in required:
            if not any(
                kind == expected_kind and all(fragment in definition for fragment in fragments)
                for kind, definition in actual
            ):
                _raise_agent_recall_delivery_schema_error(table)

    index = "idx_cayu_agent_recall_delivery_pending"
    await cur.execute(
        """
            SELECT indexdef
            FROM pg_indexes
            WHERE schemaname = current_schema() AND indexname = %s
            """,
        (index,),
    )
    row = await cur.fetchone()
    normalized = "" if row is None else " ".join(str(row[0]).lower().split())
    required_index_fragments = (
        "cayu_agent_recall_delivery_states using btree ",
        "(agent_id, task_id, knowledge_namespace, access_policy_sha256, ",
        "checkpoint_stream_id, checkpoint_revision, delivery_id)",
        "where (state <> 'acknowledged'",
    )
    if any(fragment not in normalized for fragment in required_index_fragments):
        _raise_agent_recall_delivery_schema_error(index)


def _raise_agent_recall_delivery_schema_error(name: str) -> NoReturn:
    raise RuntimeError(
        "Postgres schema object "
        f"{name!r} conflicts with Cayu's staged recall-delivery contract. "
        "Run schema_mode=MIGRATE to install the breaking revision or recreate "
        "the database."
    )


async def _validate_agent_recall_subscription_schema(cur: Any) -> None:
    expected_columns = {
        "cayu_agent_recall_subscription_revisions": (
            ("subscription_id", "text", "NO", "C"),
            ("revision", "integer", "NO", None),
            ("operation_id", "text", "NO", "C"),
            ("agent_id", "text", "NO", "C"),
            ("task_id", "text", "NO", "C"),
            ("knowledge_namespace", "text", "NO", "C"),
            ("access_policy_sha256", "text", "NO", "C"),
            ("work_context_revision", "integer", "NO", None),
            ("work_context_sha256", "text", "NO", "C"),
            ("status", "text", "NO", "C"),
            ("priority", "integer", "NO", None),
            ("subscription_json", "jsonb", "NO", None),
            ("expires_at", "timestamp with time zone", "NO", None),
            ("published_at", "timestamp with time zone", "NO", None),
        ),
        "cayu_agent_recall_subscription_heads": (
            ("subscription_id", "text", "NO", "C"),
            ("current_revision", "integer", "NO", None),
        ),
        "cayu_agent_recall_subscription_publications": (
            ("operation_id", "text", "NO", "C"),
            ("subscription_id", "text", "NO", "C"),
            ("subscription_revision", "integer", "NO", None),
            ("request_sha256", "text", "NO", "C"),
            ("receipt_json", "jsonb", "NO", None),
            ("committed_at", "timestamp with time zone", "NO", None),
        ),
        "cayu_agent_recall_subscription_claims": (
            ("claim_id", "text", "NO", "C"),
            ("subscription_id", "text", "NO", "C"),
            ("subscription_revision", "integer", "NO", None),
            ("runner_id", "text", "NO", "C"),
            ("request_sha256", "text", "NO", "C"),
            ("attempt", "bigint", "NO", None),
            ("claimed_at", "timestamp with time zone", "NO", None),
        ),
        "cayu_agent_recall_subscription_releases": (
            ("release_id", "text", "NO", "C"),
            ("subscription_id", "text", "NO", "C"),
            ("claim_id", "text", "NO", "C"),
            ("request_sha256", "text", "NO", "C"),
            ("release_json", "jsonb", "NO", None),
            ("released_at", "timestamp with time zone", "NO", None),
        ),
        "cayu_agent_recall_subscription_states": (
            ("subscription_id", "text", "NO", "C"),
            ("current_revision", "integer", "NO", None),
            ("agent_id", "text", "NO", "C"),
            ("task_id", "text", "NO", "C"),
            ("knowledge_namespace", "text", "NO", "C"),
            ("access_policy_sha256", "text", "NO", "C"),
            ("run_state", "text", "NO", "C"),
            ("attempt", "bigint", "NO", None),
            ("state_revision", "bigint", "NO", None),
            ("lease_expires_at", "timestamp with time zone", "YES", None),
            ("release_id", "text", "YES", "C"),
            ("next_evaluation_at", "timestamp with time zone", "NO", None),
            ("last_evaluation_id", "text", "YES", "C"),
            ("state_json", "jsonb", "NO", None),
            ("updated_at", "timestamp with time zone", "NO", None),
        ),
        "cayu_agent_recall_subscription_evaluations": (
            ("evaluation_id", "text", "NO", "C"),
            ("subscription_id", "text", "NO", "C"),
            ("subscription_revision", "integer", "NO", None),
            ("agent_id", "text", "NO", "C"),
            ("task_id", "text", "NO", "C"),
            ("knowledge_namespace", "text", "NO", "C"),
            ("access_policy_sha256", "text", "NO", "C"),
            ("claim_id", "text", "NO", "C"),
            ("processing_operation_id", "text", "NO", "C"),
            ("request_sha256", "text", "NO", "C"),
            ("outcome", "text", "NO", "C"),
            ("delivery_id", "text", "YES", "C"),
            ("evaluation_json", "jsonb", "NO", None),
            ("committed_at", "timestamp with time zone", "NO", None),
        ),
        "cayu_agent_recall_subscription_wake_claims": (
            ("claim_id", "text", "NO", "C"),
            ("wake_id", "text", "NO", "C"),
            ("delivery_id", "text", "NO", "C"),
            ("runner_id", "text", "NO", "C"),
            ("request_sha256", "text", "NO", "C"),
            ("attempt", "bigint", "NO", None),
            ("claimed_at", "timestamp with time zone", "NO", None),
        ),
        "cayu_agent_recall_subscription_wake_releases": (
            ("release_id", "text", "NO", "C"),
            ("wake_id", "text", "NO", "C"),
            ("claim_id", "text", "NO", "C"),
            ("request_sha256", "text", "NO", "C"),
            ("release_json", "jsonb", "NO", None),
            ("released_at", "timestamp with time zone", "NO", None),
        ),
        "cayu_agent_recall_subscription_wake_states": (
            ("wake_id", "text", "NO", "C"),
            ("delivery_id", "text", "NO", "C"),
            ("agent_id", "text", "NO", "C"),
            ("task_id", "text", "NO", "C"),
            ("knowledge_namespace", "text", "NO", "C"),
            ("access_policy_sha256", "text", "NO", "C"),
            ("state", "text", "NO", "C"),
            ("attempt", "bigint", "NO", None),
            ("state_revision", "bigint", "NO", None),
            ("claim_id", "text", "YES", "C"),
            ("lease_expires_at", "timestamp with time zone", "YES", None),
            ("release_id", "text", "YES", "C"),
            ("acknowledgement_id", "text", "YES", "C"),
            ("state_json", "jsonb", "NO", None),
            ("committed_at", "timestamp with time zone", "NO", None),
            ("updated_at", "timestamp with time zone", "NO", None),
        ),
    }
    for table, expected in expected_columns.items():
        await cur.execute(
            """
                SELECT column_name, data_type, is_nullable, collation_name
                FROM information_schema.columns
                WHERE table_schema = current_schema() AND table_name = %s
                ORDER BY ordinal_position
                """,
            (table,),
        )
        if tuple(await cur.fetchall()) != expected:
            _raise_agent_recall_subscription_schema_error(table)

    required_constraints = {
        "cayu_agent_recall_subscription_revisions": (
            ("p", ("primary key (subscription_id, revision)",)),
            ("u", ("unique (operation_id)",)),
            ("c", ("revision > 0", "2147483647")),
            ("c", ("access_policy_sha256", "[0-9a-f]{64}")),
            ("c", ("work_context_sha256", "[0-9a-f]{64}")),
            ("c", ("status", "active", "paused", "cancelled")),
            ("c", ("priority >= 0", "priority <= 1000")),
            ("c", ("jsonb_typeof(subscription_json)", "object")),
            (
                "f",
                (
                    "foreign key (task_id, work_context_revision) references ",
                    "cayu_agent_work_context_revisions(task_id, revision)",
                    "on delete restrict",
                ),
            ),
        ),
        "cayu_agent_recall_subscription_wake_claims": (
            ("p", ("primary key (claim_id)",)),
            ("u", ("unique (wake_id, attempt)",)),
            ("c", ("attempt > 0", "9223372036854775807")),
            ("c", ("request_sha256", "[0-9a-f]{64}")),
            (
                "f",
                (
                    "foreign key (wake_id) references ",
                    "cayu_agent_recall_subscription_evaluations(evaluation_id)",
                    "on delete restrict",
                ),
            ),
            (
                "f",
                (
                    "foreign key (delivery_id) references ",
                    "cayu_agent_recall_deliveries(delivery_id)",
                    "on delete restrict",
                ),
            ),
        ),
        "cayu_agent_recall_subscription_wake_releases": (
            ("p", ("primary key (release_id)",)),
            ("c", ("request_sha256", "[0-9a-f]{64}")),
            ("c", ("jsonb_typeof(release_json)", "object")),
            (
                "f",
                (
                    "foreign key (wake_id) references ",
                    "cayu_agent_recall_subscription_evaluations(evaluation_id)",
                    "on delete restrict",
                ),
            ),
            (
                "f",
                (
                    "foreign key (claim_id) references ",
                    "cayu_agent_recall_subscription_wake_claims(claim_id)",
                    "on delete restrict",
                ),
            ),
        ),
        "cayu_agent_recall_subscription_wake_states": (
            ("p", ("primary key (wake_id)",)),
            ("u", ("unique (delivery_id)",)),
            ("u", ("unique (release_id)",)),
            ("u", ("unique (acknowledgement_id)",)),
            ("c", ("state", "pending", "claimed", "acknowledged")),
            ("c", ("state_revision >= 0", "9223372036854775807")),
            ("c", ("jsonb_typeof(state_json)", "object")),
            (
                "f",
                (
                    "foreign key (wake_id) references ",
                    "cayu_agent_recall_subscription_evaluations(evaluation_id)",
                    "on delete restrict",
                ),
            ),
            (
                "f",
                (
                    "foreign key (delivery_id) references ",
                    "cayu_agent_recall_deliveries(delivery_id)",
                    "on delete restrict",
                ),
            ),
            (
                "f",
                (
                    "foreign key (claim_id) references ",
                    "cayu_agent_recall_subscription_wake_claims(claim_id)",
                    "on delete restrict",
                ),
            ),
            (
                "f",
                (
                    "foreign key (release_id) references ",
                    "cayu_agent_recall_subscription_wake_releases(release_id)",
                    "on delete restrict",
                ),
            ),
        ),
        "cayu_agent_recall_subscription_heads": (
            ("p", ("primary key (subscription_id)",)),
            (
                "f",
                (
                    "foreign key (subscription_id, current_revision) references ",
                    "cayu_agent_recall_subscription_revisions(subscription_id, revision)",
                    "on delete restrict",
                ),
            ),
        ),
        "cayu_agent_recall_subscription_publications": (
            ("p", ("primary key (operation_id)",)),
            ("c", ("request_sha256", "[0-9a-f]{64}")),
            ("c", ("jsonb_typeof(receipt_json)", "object")),
            (
                "f",
                (
                    "foreign key (subscription_id, subscription_revision) references ",
                    "cayu_agent_recall_subscription_revisions(subscription_id, revision)",
                    "on delete restrict",
                ),
            ),
        ),
        "cayu_agent_recall_subscription_claims": (
            ("p", ("primary key (claim_id)",)),
            ("u", ("unique (subscription_id, attempt)",)),
            ("c", ("attempt > 0", "9223372036854775807")),
            ("c", ("request_sha256", "[0-9a-f]{64}")),
            (
                "f",
                (
                    "foreign key (subscription_id, subscription_revision) references ",
                    "cayu_agent_recall_subscription_revisions(subscription_id, revision)",
                    "on delete restrict",
                ),
            ),
        ),
        "cayu_agent_recall_subscription_releases": (
            ("p", ("primary key (release_id)",)),
            ("c", ("request_sha256", "[0-9a-f]{64}")),
            ("c", ("jsonb_typeof(release_json)", "object")),
            (
                "f",
                (
                    "foreign key (subscription_id) references ",
                    "cayu_agent_recall_subscription_heads(subscription_id)",
                    "on delete restrict",
                ),
            ),
            (
                "f",
                (
                    "foreign key (claim_id) references ",
                    "cayu_agent_recall_subscription_claims(claim_id)",
                    "on delete restrict",
                ),
            ),
        ),
        "cayu_agent_recall_subscription_states": (
            ("p", ("primary key (subscription_id)",)),
            ("u", ("unique (release_id)",)),
            ("c", ("run_state", "due", "claimed")),
            ("c", ("state_revision >= 0", "9223372036854775807")),
            ("c", ("jsonb_typeof(state_json)", "object")),
            (
                "f",
                (
                    "foreign key (subscription_id, current_revision) references ",
                    "cayu_agent_recall_subscription_revisions(subscription_id, revision)",
                    "on delete restrict",
                ),
            ),
            (
                "f",
                (
                    "foreign key (release_id) references ",
                    "cayu_agent_recall_subscription_releases(release_id)",
                    "on delete restrict",
                ),
            ),
        ),
        "cayu_agent_recall_subscription_evaluations": (
            ("p", ("primary key (evaluation_id)",)),
            ("u", ("unique (claim_id)",)),
            ("u", ("unique (processing_operation_id)",)),
            ("u", ("unique (delivery_id)",)),
            ("c", ("request_sha256", "[0-9a-f]{64}")),
            ("c", ("outcome", "no_work", "silent", "wake")),
            ("c", ("jsonb_typeof(evaluation_json)", "object")),
            (
                "f",
                (
                    "foreign key (subscription_id, subscription_revision) references ",
                    "cayu_agent_recall_subscription_revisions(subscription_id, revision)",
                    "on delete restrict",
                ),
            ),
            (
                "f",
                (
                    "foreign key (claim_id) references ",
                    "cayu_agent_recall_subscription_claims(claim_id)",
                    "on delete restrict",
                ),
            ),
            (
                "f",
                (
                    "foreign key (delivery_id) references ",
                    "cayu_agent_recall_deliveries(delivery_id)",
                    "on delete restrict",
                ),
            ),
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
                """,
            (table,),
        )
        actual = tuple(
            (str(kind), " ".join(str(definition).lower().split()))
            for kind, definition in await cur.fetchall()
        )
        for expected_kind, fragments in required:
            if not any(
                kind == expected_kind and all(fragment in definition for fragment in fragments)
                for kind, definition in actual
            ):
                _raise_agent_recall_subscription_schema_error(table)

    indexes = {
        "idx_cayu_agent_recall_subscription_due": (
            "cayu_agent_recall_subscription_states using btree ",
            "(agent_id, task_id, knowledge_namespace, access_policy_sha256, ",
            "next_evaluation_at, subscription_id)",
        ),
        "idx_cayu_agent_recall_subscription_evaluations": (
            "cayu_agent_recall_subscription_evaluations using btree ",
            "(subscription_id, evaluation_id)",
        ),
        "idx_cayu_agent_recall_subscription_wakes": (
            "cayu_agent_recall_subscription_wake_states using btree ",
            "(agent_id, task_id, knowledge_namespace, access_policy_sha256, ",
            "committed_at, wake_id)",
            "where (state <> 'acknowledged'::text)",
        ),
    }
    for index, fragments in indexes.items():
        await cur.execute(
            "SELECT indexdef FROM pg_indexes "
            "WHERE schemaname = current_schema() AND indexname = %s",
            (index,),
        )
        row = await cur.fetchone()
        normalized = "" if row is None else " ".join(str(row[0]).lower().split())
        if any(fragment not in normalized for fragment in fragments):
            _raise_agent_recall_subscription_schema_error(index)


def _raise_agent_recall_subscription_schema_error(name: str) -> NoReturn:
    raise RuntimeError(
        "Postgres schema object "
        f"{name!r} conflicts with Cayu's idle recall-subscription contract. "
        "Run schema_mode=MIGRATE to install revision 73 or recreate the database."
    )
