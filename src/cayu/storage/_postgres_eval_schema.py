"""PostgreSQL schema checks for evaluation cases, suites, runs and results.

Callers retain revision gates, validation order and the existing cursor/transaction.
These functions inspect the schema without acquiring connections or applying DDL.
"""

from __future__ import annotations

from typing import Any, NoReturn

from cayu.storage import _postgres_catalog as postgres_catalog


async def _validate_eval_result_baseline_schema(cur: Any) -> None:
    expected_columns = {
        "cayu_eval_result_records": (
            ("revision", "text", "NO"),
            ("origin", "text", "NO"),
            ("target_key", "text", "NO"),
            ("corpus_revision", "text", "NO"),
            ("suite_id", "text", "NO"),
            ("suite_revision", "text", "NO"),
            ("application_release_id", "text", "NO"),
            ("app_manifest_schema_version", "text", "NO"),
            ("app_manifest_fingerprint", "text", "NO"),
            ("result_status", "text", "NO"),
            ("result_score", "double precision", "YES"),
            ("fresh_run_id", "text", "YES"),
            ("captured_result", "text", "YES"),
            ("document_bytes", "bigint", "NO"),
            ("created_at", "timestamp with time zone", "NO"),
        ),
        "cayu_eval_baselines": (
            ("target_key", "text", "NO"),
            ("corpus_revision", "text", "NO"),
            ("suite_id", "text", "NO"),
            ("result_revision", "text", "NO"),
            ("generation", "bigint", "NO"),
            ("updated_by", "text", "NO"),
            ("updated_at", "timestamp with time zone", "NO"),
        ),
        "cayu_eval_baseline_mutations": (
            ("operation_id", "text", "NO"),
            ("target_key", "text", "NO"),
            ("corpus_revision", "text", "NO"),
            ("suite_id", "text", "NO"),
            ("expected_generation", "bigint", "NO"),
            ("previous_result_revision", "text", "YES"),
            ("selected_result_revision", "text", "NO"),
            ("resulting_generation", "bigint", "NO"),
            ("actor_id", "text", "NO"),
            ("created_at", "timestamp with time zone", "NO"),
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
            _raise_eval_result_baseline_schema_error(table_name)

    expected_keys = {
        "cayu_eval_result_records": {
            "PRIMARY KEY (revision)",
            "UNIQUE (fresh_run_id)",
        },
        "cayu_eval_baselines": {
            "PRIMARY KEY (target_key, corpus_revision, suite_id)",
        },
        "cayu_eval_baseline_mutations": {"PRIMARY KEY (operation_id)"},
    }
    for table_name, expected in expected_keys.items():
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
                  AND constraint_record.contype IN ('p', 'u')
                """,
            (table_name,),
        )
        if {str(row[0]) for row in await cur.fetchall()} != expected:
            _raise_eval_result_baseline_schema_error(table_name)

    expected_indexes = {
        "idx_cayu_eval_result_records_target_catalog": (
            False,
            "target_key, created_at DESC, revision",
        ),
        "idx_cayu_eval_result_records_contract": (
            False,
            "target_key, corpus_revision, suite_id, created_at DESC, revision",
        ),
        "idx_cayu_eval_baseline_mutations_scope": (
            True,
            "target_key, corpus_revision, suite_id, resulting_generation",
        ),
    }
    await cur.execute(
        """
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE schemaname = current_schema() AND indexname = ANY(%s)
            """,
        (list(expected_indexes),),
    )
    actual_indexes = {str(row[0]): str(row[1]) for row in await cur.fetchall()}
    if set(actual_indexes) != set(expected_indexes) or any(
        f"({columns})" not in actual_indexes[index_name]
        or ("CREATE UNIQUE INDEX" in actual_indexes[index_name]) is not unique
        for index_name, (unique, columns) in expected_indexes.items()
    ):
        _raise_eval_result_baseline_schema_error("eval result indexes")


async def _validate_captured_eval_case_schema(cur: Any) -> None:
    await cur.execute(
        """
            SELECT pg_get_constraintdef(constraint_record.oid)
            FROM pg_catalog.pg_constraint AS constraint_record
            JOIN pg_catalog.pg_class AS table_record
              ON table_record.oid = constraint_record.conrelid
            JOIN pg_catalog.pg_namespace AS namespace
              ON namespace.oid = table_record.relnamespace
            WHERE namespace.nspname = current_schema()
              AND table_record.relname = 'cayu_eval_cases'
              AND constraint_record.conname = 'cayu_eval_cases_message_count_check'
            """
    )
    row = await cur.fetchone()
    definition = "" if row is None else "".join(str(row[0]).lower().split())
    if "message_count>=0" not in definition or "message_count<=16" not in definition:
        raise RuntimeError(
            "Postgres schema object 'cayu_eval_cases.message_count' conflicts with "
            "Cayu's revision-48 captured-evaluation contract. Run `cayu storage "
            "migrate` or restore the database from a known-good backup."
        )


async def _validate_eval_run_invocation_column(cur: Any) -> None:
    await cur.execute(
        """
            SELECT data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'cayu_eval_runs'
              AND column_name = 'invocation_json'
            """
    )
    if await cur.fetchone() != ("text", "NO", None):
        raise RuntimeError(
            "Postgres schema object 'cayu_eval_runs.invocation_json' conflicts "
            "with Cayu's revision-50 durable eval invocation contract. Run "
            "`cayu storage migrate` or restore the database from a known-good backup."
        )


async def _validate_eval_run_max_concurrency_schema(cur: Any) -> None:
    await cur.execute(
        """
            SELECT column_record.data_type,
                   pg_get_constraintdef(constraint_record.oid)
            FROM information_schema.columns AS column_record
            JOIN pg_catalog.pg_namespace AS namespace
              ON namespace.nspname = column_record.table_schema
            JOIN pg_catalog.pg_class AS table_record
              ON table_record.relnamespace = namespace.oid
             AND table_record.relname = column_record.table_name
            LEFT JOIN pg_catalog.pg_constraint AS constraint_record
              ON constraint_record.conrelid = table_record.oid
             AND constraint_record.conname = 'cayu_eval_runs_max_concurrency_check'
            WHERE column_record.table_schema = current_schema()
              AND column_record.table_name = 'cayu_eval_runs'
              AND column_record.column_name = 'max_concurrency'
            """
    )
    row = await cur.fetchone()
    definition = "" if row is None or row[1] is None else "".join(str(row[1]).lower().split())
    if (
        row is None
        or row[0] != "integer"
        or "max_concurrency>=1" not in definition
        or "max_concurrency<=2147483647" not in definition
    ):
        raise RuntimeError(
            "Postgres schema object 'cayu_eval_runs.max_concurrency' conflicts "
            "with Cayu's revision-72 portable concurrency contract. Run `cayu "
            "storage migrate` or restore the database from a known-good backup."
        )


async def _validate_eval_run_scenario_progress_column(cur: Any) -> None:
    await cur.execute(
        """
            SELECT data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'cayu_eval_runs'
              AND column_name = 'scenario_progress_json'
            """
    )
    if await cur.fetchone() != ("text", "YES", None):
        raise RuntimeError(
            "Postgres schema object 'cayu_eval_runs.scenario_progress_json' "
            "conflicts with Cayu's revision-56 controlled-scenario execution "
            "contract. Run `cayu storage migrate` or restore the database from "
            "a known-good backup."
        )


async def _validate_eval_run_trial_checkpoint_schema(cur: Any) -> None:
    await cur.execute(
        """
            SELECT data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'cayu_eval_runs'
              AND column_name = 'trial_checkpoint_count'
            """
    )
    count_column = await cur.fetchone()
    if count_column is None or count_column[:2] != ("integer", "NO") or count_column[2] is None:
        raise RuntimeError(
            "Postgres schema object 'cayu_eval_runs.trial_checkpoint_count' "
            "conflicts with Cayu's revision-74 eval trial recovery contract. "
            "Run `cayu storage migrate` or restore the database from a known-good backup."
        )
    await cur.execute(
        """
            SELECT data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'cayu_eval_runs'
              AND column_name = 'trial_checkpoint_bytes'
            """
    )
    bytes_column = await cur.fetchone()
    if bytes_column is None or bytes_column[:2] != ("bigint", "NO") or bytes_column[2] is None:
        raise RuntimeError(
            "Postgres schema object 'cayu_eval_runs.trial_checkpoint_bytes' "
            "conflicts with Cayu's revision-74 eval trial recovery contract. "
            "Run `cayu storage migrate` or restore the database from a known-good backup."
        )
    await cur.execute(
        """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'cayu_eval_run_trial_checkpoints'
            ORDER BY ordinal_position
            """
    )
    if tuple(await cur.fetchall()) != (
        ("run_id", "text", "NO"),
        ("case_id", "text", "NO"),
        ("trial_number", "integer", "NO"),
        ("checkpoint_json", "text", "NO"),
        ("document_bytes", "bigint", "NO"),
    ):
        raise RuntimeError(
            "Postgres schema object 'cayu_eval_run_trial_checkpoints' conflicts "
            "with Cayu's revision-74 eval trial recovery contract. Run `cayu storage "
            "migrate` or restore the database from a known-good backup."
        )
    await cur.execute(
        """
            SELECT constraint_state.contype,
                   pg_get_constraintdef(constraint_state.oid)
            FROM pg_catalog.pg_constraint AS constraint_state
            JOIN pg_catalog.pg_class AS table_record
              ON table_record.oid = constraint_state.conrelid
            JOIN pg_catalog.pg_namespace AS namespace
              ON namespace.oid = table_record.relnamespace
            WHERE namespace.nspname = current_schema()
              AND table_record.relname = 'cayu_eval_run_trial_checkpoints'
              AND constraint_state.contype IN ('p', 'f')
            ORDER BY constraint_state.contype
            """
    )
    ownership_constraints = {
        str(kind): " ".join(str(definition).lower().split())
        for kind, definition in await cur.fetchall()
    }
    if ownership_constraints != {
        "f": ("foreign key (run_id) references cayu_eval_runs(run_id) on delete cascade"),
        "p": "primary key (run_id, case_id, trial_number)",
    }:
        raise RuntimeError(
            "Postgres schema object 'cayu_eval_run_trial_checkpoints' has invalid "
            "slot identity or run ownership constraints. Run `cayu storage migrate` "
            "or restore the database from a known-good backup."
        )
    await cur.execute(
        """
            SELECT data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'cayu_eval_runs'
              AND column_name = 'authored_suite_launch_lane'
            """
    )
    if await cur.fetchone() != ("integer", "YES", None):
        raise RuntimeError(
            "Postgres schema object 'cayu_eval_runs.authored_suite_launch_lane' "
            "conflicts with Cayu's revision-74 authored-suite concurrency contract. "
            "Run `cayu storage migrate` or restore the database from a known-good backup."
        )
    await cur.execute(
        """
            SELECT data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'cayu_eval_runs'
              AND column_name = 'authored_suite_launch_revision'
            """
    )
    if await cur.fetchone() != ("text", "YES", None):
        raise RuntimeError(
            "Postgres schema object 'cayu_eval_runs.authored_suite_launch_revision' "
            "conflicts with Cayu's revision-74 authored-suite concurrency contract. "
            "Run `cayu storage migrate` or restore the database from a known-good backup."
        )
    await cur.execute(
        """
            SELECT index_state.indisvalid,
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
              AND table_record.relname = 'cayu_eval_runs'
              AND index_record.relname =
                  'idx_cayu_eval_runs_authored_suite_launch_claim'
            """
    )
    index_row = await cur.fetchone()
    normalized_definition = "" if index_row is None else " ".join(str(index_row[2]).lower().split())
    if (
        index_row is None
        or not bool(index_row[0])
        or not bool(index_row[1])
        or "using btree (authored_suite_launch_revision, authored_suite_launch_lane, "
        "created_at, run_id, status)"
        not in normalized_definition
        or "where (authored_suite_launch_revision is not null)" not in normalized_definition
    ):
        raise RuntimeError(
            "Postgres schema object 'idx_cayu_eval_runs_authored_suite_launch_claim' "
            "conflicts with Cayu's revision-74 authored-suite concurrency contract. "
            "Run `cayu storage migrate` or restore the database from a known-good backup."
        )


async def _validate_eval_scenario_schema(cur: Any) -> None:
    await cur.execute(
        """
            SELECT column_name, data_type, is_nullable, collation_name
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'cayu_eval_scenarios'
            ORDER BY ordinal_position
            """
    )
    expected_columns = (
        ("revision", "text", "NO", "C"),
        ("scenario_id", "text", "NO", "C"),
        ("target_key", "text", "NO", None),
        ("name", "text", "NO", None),
        ("description", "text", "YES", None),
        ("event_count", "bigint", "NO", None),
        ("input_event_count", "bigint", "NO", None),
        ("approval_checkpoint_count", "bigint", "NO", None),
        ("message_count", "bigint", "NO", None),
        ("part_count", "bigint", "NO", None),
        ("artifact_requirement_count", "bigint", "NO", None),
        ("secret_requirement_count", "bigint", "NO", None),
        ("document_json", "text", "NO", None),
        ("document_bytes", "bigint", "NO", None),
        ("created_at", "timestamp with time zone", "NO", None),
    )
    if tuple(await cur.fetchall()) != expected_columns:
        _raise_eval_scenario_schema_error("cayu_eval_scenarios")
    expected_constraints = (
        ("p", ("primary key (revision)",)),
        ("c", ("event_count >= 1", "event_count <= 1024")),
        ("c", ("input_event_count >= 1", "input_event_count <= 1024")),
        (
            "c",
            ("approval_checkpoint_count >= 0", "approval_checkpoint_count <= 1024"),
        ),
        ("c", ("message_count >= input_event_count", "message_count <= 32768")),
        ("c", ("part_count >= message_count", "part_count <= 1048576")),
        (
            "c",
            ("artifact_requirement_count >= 0", "artifact_requirement_count <= 128"),
        ),
        (
            "c",
            ("secret_requirement_count >= 0", "secret_requirement_count <= 128"),
        ),
        ("c", ("document_bytes >= 1", "document_bytes <= 8388608")),
        ("c", ("document_bytes = octet_length(document_json)",)),
        (
            "c",
            ("input_event_count + approval_checkpoint_count", "= event_count"),
        ),
        ("c", ("document_json", "::jsonb is not null")),
    )
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
              AND table_record.relname = 'cayu_eval_scenarios'
              AND constraint_record.contype IN ('p', 'u', 'f', 'c')
            """
    )
    candidates = [
        (str(kind), " ".join(str(definition).lower().split()))
        for kind, definition in await cur.fetchall()
    ]
    if not postgres_catalog._constraint_fragments_match_exactly(candidates, expected_constraints):
        _raise_eval_scenario_schema_error("eval scenario constraints")
    expected_indexes = {
        "idx_cayu_eval_scenarios_catalog": (
            "cayu_eval_scenarios",
            "using btree (created_at desc, revision)",
        ),
        "idx_cayu_eval_scenarios_target_catalog": (
            "cayu_eval_scenarios",
            "using btree (target_key, created_at desc, revision)",
        ),
        "idx_cayu_eval_scenarios_id_catalog": (
            "cayu_eval_scenarios",
            "using btree (scenario_id, created_at desc, revision)",
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
            _raise_eval_scenario_schema_error(name)

    await cur.execute(
        """
            SELECT index_record.relname
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
              AND table_record.relname = 'cayu_eval_scenarios'
              AND index_state.indisunique
              AND constraint_record.oid IS NULL
            LIMIT 1
            """
    )
    unexpected_unique_index = await cur.fetchone()
    if unexpected_unique_index is not None:
        _raise_eval_scenario_schema_error(str(unexpected_unique_index[0]))


async def _validate_eval_authored_suite_schema(cur: Any) -> None:
    await cur.execute(
        """
            SELECT column_name, data_type, is_nullable, collation_name
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'cayu_eval_authored_suites'
            ORDER BY ordinal_position
            """
    )
    expected_columns = (
        ("revision", "text", "NO", "C"),
        ("suite_id", "text", "NO", "C"),
        ("suite_revision", "text", "NO", None),
        ("target_key", "text", "NO", None),
        ("name", "text", "NO", None),
        ("description", "text", "YES", None),
        ("case_count", "bigint", "NO", None),
        ("assertion_count", "bigint", "NO", None),
        ("simple_input_count", "bigint", "NO", None),
        ("scenario_count", "bigint", "NO", None),
        ("trials", "bigint", "NO", None),
        ("timeout_seconds", "bigint", "NO", None),
        ("document_json", "text", "NO", None),
        ("document_bytes", "bigint", "NO", None),
        ("created_at", "timestamp with time zone", "NO", None),
    )
    if tuple(await cur.fetchall()) != expected_columns:
        _raise_eval_authored_suite_schema_error("cayu_eval_authored_suites")
    expected_constraints = (
        ("p", ("primary key (revision)",)),
        ("c", ("case_count >= 1", "case_count <= 1000")),
        ("c", ("assertion_count >= case_count", "assertion_count <= 64000")),
        ("c", ("simple_input_count >= 0", "simple_input_count <= case_count")),
        ("c", ("scenario_count >= 0", "scenario_count <= case_count")),
        ("c", ("trials >= 1", "trials <= 100")),
        ("c", ("timeout_seconds >= 1", "timeout_seconds <= 3600")),
        ("c", ("document_bytes >= 1", "document_bytes <= 8388608")),
        ("c", ("document_bytes = octet_length(document_json)",)),
        ("c", ("simple_input_count + scenario_count", "= case_count")),
        ("c", ("assertion_count * trials", "<= 10000")),
        ("c", ("document_json", "::jsonb is not null")),
    )
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
              AND table_record.relname = 'cayu_eval_authored_suites'
              AND constraint_record.contype IN ('p', 'u', 'f', 'c')
            """
    )
    candidates = [
        (str(kind), " ".join(str(definition).lower().split()))
        for kind, definition in await cur.fetchall()
    ]
    if not postgres_catalog._constraint_fragments_match_exactly(candidates, expected_constraints):
        _raise_eval_authored_suite_schema_error("authored suite constraints")
    expected_indexes = {
        "idx_cayu_eval_authored_suites_catalog": (
            "cayu_eval_authored_suites",
            "using btree (created_at desc, revision)",
        ),
        "idx_cayu_eval_authored_suites_target_catalog": (
            "cayu_eval_authored_suites",
            "using btree (target_key, created_at desc, revision)",
        ),
        "idx_cayu_eval_authored_suites_id_catalog": (
            "cayu_eval_authored_suites",
            "using btree (suite_id, created_at desc, revision)",
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
            _raise_eval_authored_suite_schema_error(name)


def _raise_eval_authored_suite_schema_error(name: str) -> NoReturn:
    raise RuntimeError(
        f"Postgres schema object {name!r} conflicts with Cayu's revision-64 "
        "authored-suite contract. Run `cayu storage migrate` or restore the "
        "database from a known-good backup."
    )


async def _validate_eval_judge_calibration_schema(cur: Any) -> None:
    await cur.execute(
        """
            SELECT column_name, data_type, is_nullable, collation_name
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'cayu_eval_judge_calibrations'
            ORDER BY ordinal_position
            """
    )
    expected_columns = (
        ("revision", "text", "NO", "C"),
        ("run_id", "text", "NO", "C"),
        ("definition_revision", "text", "NO", None),
        ("target_key", "text", "NO", None),
        ("trial_count", "bigint", "NO", None),
        ("report_json", "text", "NO", None),
        ("document_bytes", "bigint", "NO", None),
        ("created_at", "timestamp with time zone", "NO", None),
    )
    if tuple(await cur.fetchall()) != expected_columns:
        _raise_eval_judge_calibration_schema_error("cayu_eval_judge_calibrations")
    expected_constraints = (
        ("p", ("primary key (revision)",)),
        ("u", ("unique (run_id)",)),
        ("c", ("trial_count >= 1", "trial_count <= 10")),
        ("c", ("document_bytes >= 1", "document_bytes <= 2097152")),
        ("c", ("document_bytes = octet_length(report_json)",)),
        ("c", ("jsonb_typeof", "report_json", "= 'object'")),
    )
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
              AND table_record.relname = 'cayu_eval_judge_calibrations'
              AND constraint_record.contype IN ('p', 'u', 'f', 'c')
            """
    )
    candidates = [
        (str(kind), " ".join(str(definition).lower().split()))
        for kind, definition in await cur.fetchall()
    ]
    if not postgres_catalog._constraint_fragments_match_exactly(candidates, expected_constraints):
        _raise_eval_judge_calibration_schema_error("judge calibration constraints")
    expected_indexes = {
        "idx_cayu_eval_judge_calibrations_target": (
            "cayu_eval_judge_calibrations",
            "using btree (target_key, created_at desc, revision)",
        ),
        "idx_cayu_eval_judge_calibrations_definition": (
            "cayu_eval_judge_calibrations",
            "using btree (definition_revision, created_at desc, revision)",
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
            _raise_eval_judge_calibration_schema_error(name)


def _raise_eval_judge_calibration_schema_error(name: str) -> NoReturn:
    raise RuntimeError(
        f"Postgres schema object {name!r} conflicts with Cayu's revision-68 "
        "judge-calibration contract. Run `cayu storage migrate` or restore "
        "the database from a known-good backup."
    )


def _raise_eval_scenario_schema_error(name: str) -> NoReturn:
    raise RuntimeError(
        f"Postgres schema object {name!r} conflicts with Cayu's revision-53 "
        "portable scenario contract. Run `cayu storage migrate` or restore "
        "the database from a known-good backup."
    )


def _raise_eval_result_baseline_schema_error(name: str) -> NoReturn:
    raise RuntimeError(
        f"Postgres schema object {name!r} conflicts with Cayu's revision-47 "
        "Evals result and baseline contract. Run `cayu storage migrate` or "
        "restore the database from a known-good backup."
    )
