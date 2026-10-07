"""SQLite schema integrity for stored evaluations and their execution records.

Revision gates and migration sequencing remain with schema support.
These checks inspect existing schema without loading evaluation or store owners.
"""

from __future__ import annotations

import sqlite3
from typing import NoReturn


def _validate_eval_result_baseline_schema(connection: sqlite3.Connection) -> None:
    expected_columns = {
        "cayu_eval_result_records": (
            ("revision", "TEXT", 0, 1),
            ("origin", "TEXT", 1, 0),
            ("target_key", "TEXT", 1, 0),
            ("corpus_revision", "TEXT", 1, 0),
            ("suite_id", "TEXT", 1, 0),
            ("suite_revision", "TEXT", 1, 0),
            ("application_release_id", "TEXT", 1, 0),
            ("app_manifest_schema_version", "TEXT", 1, 0),
            ("app_manifest_fingerprint", "TEXT", 1, 0),
            ("result_status", "TEXT", 1, 0),
            ("result_score", "REAL", 0, 0),
            ("fresh_run_id", "TEXT", 0, 0),
            ("captured_result_json", "TEXT", 0, 0),
            ("document_bytes", "INTEGER", 1, 0),
            ("created_at", "TEXT", 1, 0),
        ),
        "cayu_eval_baselines": (
            ("target_key", "TEXT", 1, 1),
            ("corpus_revision", "TEXT", 1, 2),
            ("suite_id", "TEXT", 1, 3),
            ("result_revision", "TEXT", 1, 0),
            ("generation", "INTEGER", 1, 0),
            ("updated_by", "TEXT", 1, 0),
            ("updated_at", "TEXT", 1, 0),
        ),
        "cayu_eval_baseline_mutations": (
            ("operation_id", "TEXT", 0, 1),
            ("target_key", "TEXT", 1, 0),
            ("corpus_revision", "TEXT", 1, 0),
            ("suite_id", "TEXT", 1, 0),
            ("expected_generation", "INTEGER", 1, 0),
            ("previous_result_revision", "TEXT", 0, 0),
            ("selected_result_revision", "TEXT", 1, 0),
            ("resulting_generation", "INTEGER", 1, 0),
            ("actor_id", "TEXT", 1, 0),
            ("created_at", "TEXT", 1, 0),
        ),
    }
    for table, expected in expected_columns.items():
        actual = tuple(
            (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
            for row in connection.execute(f"PRAGMA table_info({table})")
        )
        if actual != expected:
            raise RuntimeError(
                f"SQLite schema object {table!r} conflicts with Cayu's revision-47 "
                "Evals result and baseline contract. Run `cayu storage migrate` or "
                "restore the database from a known-good backup."
            )
    expected_indexes = {
        "idx_cayu_eval_result_records_target_catalog": (
            "cayu_eval_result_records",
            False,
            ("target_key", "created_at", "revision"),
        ),
        "idx_cayu_eval_result_records_contract": (
            "cayu_eval_result_records",
            False,
            ("target_key", "corpus_revision", "suite_id", "created_at", "revision"),
        ),
        "idx_cayu_eval_baseline_mutations_scope": (
            "cayu_eval_baseline_mutations",
            True,
            ("target_key", "corpus_revision", "suite_id", "resulting_generation"),
        ),
    }
    for index_name, (table_name, expected_unique, expected_columns) in expected_indexes.items():
        actual_columns = tuple(
            str(row[2]) for row in connection.execute(f"PRAGMA index_info({index_name})")
        )
        index_row = connection.execute(
            "SELECT tbl_name FROM sqlite_master WHERE type = 'index' AND name = ?",
            (index_name,),
        ).fetchone()
        unique = next(
            (
                bool(row[2])
                for row in connection.execute(f"PRAGMA index_list({table_name})")
                if str(row[1]) == index_name
            ),
            None,
        )
        if (
            index_row is None
            or str(index_row[0]) != table_name
            or unique is not expected_unique
            or actual_columns != expected_columns
        ):
            raise RuntimeError(
                f"SQLite schema object {index_name!r} conflicts with Cayu's revision-47 "
                "Evals query contract. Run `cayu storage migrate` or restore the "
                "database from a known-good backup."
            )


def _validate_captured_eval_case_schema(connection: sqlite3.Connection) -> None:
    row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'cayu_eval_cases'"
    ).fetchone()
    normalized = "" if row is None or row[0] is None else "".join(str(row[0]).lower().split())
    if "check(message_count>=0andmessage_count<=16)" not in normalized:
        raise RuntimeError(
            "SQLite schema object 'cayu_eval_cases.message_count' conflicts with Cayu's "
            "revision-48 captured-evaluation contract. Run `cayu storage migrate` or "
            "restore the database from a known-good backup."
        )


def _validate_eval_run_max_concurrency_schema(connection: sqlite3.Connection) -> None:
    row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'cayu_eval_runs'"
    ).fetchone()
    normalized = "" if row is None or row[0] is None else "".join(str(row[0]).lower().split())
    if "check(max_concurrency>=1andmax_concurrency<=2147483647)" not in normalized:
        raise RuntimeError(
            "SQLite schema object 'cayu_eval_runs.max_concurrency' conflicts with "
            "Cayu's revision-72 portable concurrency contract. Run `cayu storage "
            "migrate` or restore the database from a known-good backup."
        )


def _validate_eval_scenario_schema(connection: sqlite3.Connection) -> None:
    expected_columns = (
        ("revision", "TEXT", 0, 1),
        ("scenario_id", "TEXT", 1, 0),
        ("target_key", "TEXT", 1, 0),
        ("name", "TEXT", 1, 0),
        ("description", "TEXT", 0, 0),
        ("event_count", "INTEGER", 1, 0),
        ("input_event_count", "INTEGER", 1, 0),
        ("approval_checkpoint_count", "INTEGER", 1, 0),
        ("message_count", "INTEGER", 1, 0),
        ("part_count", "INTEGER", 1, 0),
        ("artifact_requirement_count", "INTEGER", 1, 0),
        ("secret_requirement_count", "INTEGER", 1, 0),
        ("document_json", "TEXT", 1, 0),
        ("document_bytes", "INTEGER", 1, 0),
        ("created_at", "TEXT", 1, 0),
    )
    actual_columns = tuple(
        (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
        for row in connection.execute("PRAGMA table_info(cayu_eval_scenarios)")
    )
    if actual_columns != expected_columns:
        raise RuntimeError(
            "SQLite schema object 'cayu_eval_scenarios' conflicts with Cayu's "
            "revision-53 portable scenario contract. Run `cayu storage migrate` "
            "or restore the database from a known-good backup."
        )
    table_row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'cayu_eval_scenarios'"
    ).fetchone()
    normalized_table = (
        ""
        if table_row is None or table_row[0] is None
        else "".join(str(table_row[0]).lower().split())
    )
    required_constraints = (
        "constraintcayu_eval_scenarios_event_count_checkcheck(event_count>=1andevent_count<=1024)",
        "constraintcayu_eval_scenarios_input_event_count_checkcheck(input_event_count>=1andinput_event_count<=1024)",
        "constraintcayu_eval_scenarios_approval_checkpoint_count_checkcheck(approval_checkpoint_count>=0andapproval_checkpoint_count<=1024)",
        "constraintcayu_eval_scenarios_message_count_checkcheck(message_count>=input_event_countandmessage_count<=32768)",
        "constraintcayu_eval_scenarios_part_count_checkcheck(part_count>=message_countandpart_count<=1048576)",
        "constraintcayu_eval_scenarios_artifact_requirement_count_checkcheck(artifact_requirement_count>=0andartifact_requirement_count<=128)",
        "constraintcayu_eval_scenarios_secret_requirement_count_checkcheck(secret_requirement_count>=0andsecret_requirement_count<=128)",
        "constraintcayu_eval_scenarios_document_json_checkcheck(json_valid(document_json))",
        "constraintcayu_eval_scenarios_document_bytes_checkcheck(document_bytes>=1anddocument_bytes<=8388608)",
        "constraintcayu_eval_scenarios_document_size_checkcheck(document_bytes=length(cast(document_jsonasblob)))",
        "constraintcayu_eval_scenarios_event_partition_checkcheck(input_event_count+approval_checkpoint_count=event_count)",
    )
    if "collatenocase" in normalized_table or any(
        fragment not in normalized_table for fragment in required_constraints
    ):
        raise RuntimeError(
            "SQLite schema object 'cayu_eval_scenarios' is missing Cayu's "
            "revision-53 scenario safety constraints. Run `cayu storage migrate` "
            "or restore the database from a known-good backup."
        )
    index_rows = tuple(connection.execute("PRAGMA index_list(cayu_eval_scenarios)"))
    if any(bool(row[2]) and str(row[3]) in {"c", "u"} for row in index_rows):
        raise RuntimeError(
            "SQLite schema object 'cayu_eval_scenarios' has an unexpected unique "
            "constraint or index under Cayu's revision-53 scenario contract. Run "
            "`cayu storage migrate` or restore the database from a known-good backup."
        )
    expected_indexes = {
        "idx_cayu_eval_scenarios_catalog": (
            ("created_at", 1, "BINARY"),
            ("revision", 0, "BINARY"),
        ),
        "idx_cayu_eval_scenarios_id_catalog": (
            ("scenario_id", 0, "BINARY"),
            ("created_at", 1, "BINARY"),
            ("revision", 0, "BINARY"),
        ),
        "idx_cayu_eval_scenarios_target_catalog": (
            ("target_key", 0, "BINARY"),
            ("created_at", 1, "BINARY"),
            ("revision", 0, "BINARY"),
        ),
    }
    for index_name, expected_index_columns in expected_indexes.items():
        index_row = connection.execute(
            "SELECT tbl_name FROM sqlite_master WHERE type = 'index' AND name = ?",
            (index_name,),
        ).fetchone()
        index_metadata = next(
            (row for row in index_rows if str(row[1]) == index_name),
            None,
        )
        actual_index_columns = tuple(
            (str(row[2]), int(row[3]), str(row[4]).upper())
            for row in connection.execute(f"PRAGMA index_xinfo({index_name})")
            if bool(row[5])
        )
        if (
            index_row is None
            or str(index_row[0]) != "cayu_eval_scenarios"
            or index_metadata is None
            or bool(index_metadata[2])
            or str(index_metadata[3]) != "c"
            or bool(index_metadata[4])
            or actual_index_columns != expected_index_columns
        ):
            raise RuntimeError(
                f"SQLite schema object {index_name!r} conflicts with Cayu's "
                "revision-53 scenario catalog contract. Run `cayu storage migrate` "
                "or restore the database from a known-good backup."
            )


def _validate_eval_authored_suite_schema(connection: sqlite3.Connection) -> None:
    expected_columns = (
        ("revision", "TEXT", 0, 1),
        ("suite_id", "TEXT", 1, 0),
        ("suite_revision", "TEXT", 1, 0),
        ("target_key", "TEXT", 1, 0),
        ("name", "TEXT", 1, 0),
        ("description", "TEXT", 0, 0),
        ("case_count", "INTEGER", 1, 0),
        ("assertion_count", "INTEGER", 1, 0),
        ("simple_input_count", "INTEGER", 1, 0),
        ("scenario_count", "INTEGER", 1, 0),
        ("trials", "INTEGER", 1, 0),
        ("timeout_seconds", "INTEGER", 1, 0),
        ("document_json", "TEXT", 1, 0),
        ("document_bytes", "INTEGER", 1, 0),
        ("created_at", "TEXT", 1, 0),
    )
    actual_columns = tuple(
        (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
        for row in connection.execute("PRAGMA table_info(cayu_eval_authored_suites)")
    )
    if actual_columns != expected_columns:
        _raise_eval_authored_suite_schema_error("cayu_eval_authored_suites")
    table_row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'cayu_eval_authored_suites'"
    ).fetchone()
    normalized = (
        ""
        if table_row is None or table_row[0] is None
        else "".join(str(table_row[0]).lower().split())
    )
    required_fragments = (
        "check(case_count>=1andcase_count<=1000)",
        "check(assertion_count>=case_countandassertion_count<=64000)",
        "check(simple_input_count>=0andsimple_input_count<=case_count)",
        "check(scenario_count>=0andscenario_count<=case_count)",
        "check(trials>=1andtrials<=100)",
        "check(timeout_seconds>=1andtimeout_seconds<=3600)",
        "check(json_valid(document_json))",
        "check(document_bytes>=1anddocument_bytes<=8388608)",
        "check(document_bytes=length(cast(document_jsonasblob)))",
        "check(simple_input_count+scenario_count=case_count)",
        "check(assertion_count*trials<=10000)",
    )
    if "collatenocase" in normalized or any(
        fragment not in normalized for fragment in required_fragments
    ):
        _raise_eval_authored_suite_schema_error("authored suite safety constraints")
    index_rows = tuple(connection.execute("PRAGMA index_list(cayu_eval_authored_suites)"))
    if any(bool(row[2]) and str(row[3]) in {"c", "u"} for row in index_rows):
        _raise_eval_authored_suite_schema_error("unexpected unique index")
    expected_indexes = {
        "idx_cayu_eval_authored_suites_catalog": (
            ("created_at", 1, "BINARY"),
            ("revision", 0, "BINARY"),
        ),
        "idx_cayu_eval_authored_suites_id_catalog": (
            ("suite_id", 0, "BINARY"),
            ("created_at", 1, "BINARY"),
            ("revision", 0, "BINARY"),
        ),
        "idx_cayu_eval_authored_suites_target_catalog": (
            ("target_key", 0, "BINARY"),
            ("created_at", 1, "BINARY"),
            ("revision", 0, "BINARY"),
        ),
    }
    for index_name, expected in expected_indexes.items():
        metadata = next((row for row in index_rows if str(row[1]) == index_name), None)
        actual = tuple(
            (str(row[2]), int(row[3]), str(row[4]).upper())
            for row in connection.execute(f"PRAGMA index_xinfo({index_name})")
            if bool(row[5])
        )
        if (
            metadata is None
            or bool(metadata[2])
            or str(metadata[3]) != "c"
            or bool(metadata[4])
            or actual != expected
        ):
            _raise_eval_authored_suite_schema_error(index_name)


def _raise_eval_authored_suite_schema_error(name: str) -> NoReturn:
    raise RuntimeError(
        f"SQLite schema object {name!r} conflicts with Cayu's revision-64 "
        "authored-suite contract. Run `cayu storage migrate` or restore the "
        "database from a known-good backup."
    )


def _validate_eval_judge_calibration_schema(connection: sqlite3.Connection) -> None:
    expected_columns = (
        ("revision", "TEXT", 0, 1),
        ("run_id", "TEXT", 1, 0),
        ("definition_revision", "TEXT", 1, 0),
        ("target_key", "TEXT", 1, 0),
        ("trial_count", "INTEGER", 1, 0),
        ("report_json", "TEXT", 1, 0),
        ("document_bytes", "INTEGER", 1, 0),
        ("created_at", "TEXT", 1, 0),
    )
    actual_columns = tuple(
        (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
        for row in connection.execute("PRAGMA table_info(cayu_eval_judge_calibrations)")
    )
    if actual_columns != expected_columns:
        _raise_eval_judge_calibration_schema_error("cayu_eval_judge_calibrations")
    table_row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' "
        "AND name = 'cayu_eval_judge_calibrations'"
    ).fetchone()
    normalized = (
        ""
        if table_row is None or table_row[0] is None
        else "".join(str(table_row[0]).lower().split())
    )
    required_fragments = (
        "run_idtextcollatebinarynotnullunique",
        "check(trial_count>=1andtrial_count<=10)",
        "check(json_valid(report_json)andjson_type(report_json)='object')",
        "check(document_bytes>=1anddocument_bytes<=2097152)",
        "check(document_bytes=length(cast(report_jsonasblob)))",
    )
    if "revisiontextcollatebinaryprimarykey" not in normalized or any(
        fragment not in normalized for fragment in required_fragments
    ):
        _raise_eval_judge_calibration_schema_error("calibration safety constraints")
    index_rows = tuple(connection.execute("PRAGMA index_list(cayu_eval_judge_calibrations)"))
    unique_rows = tuple(row for row in index_rows if bool(row[2]) and str(row[3]) == "u")
    if len(unique_rows) != 1:
        _raise_eval_judge_calibration_schema_error("run_id uniqueness")
    unique_columns = tuple(
        str(row[2])
        for row in connection.execute(f"PRAGMA index_xinfo({unique_rows[0][1]})")
        if bool(row[5])
    )
    if unique_columns != ("run_id",) or any(
        bool(row[2]) and str(row[3]) == "c" for row in index_rows
    ):
        _raise_eval_judge_calibration_schema_error("run_id uniqueness")
    expected_indexes = {
        "idx_cayu_eval_judge_calibrations_target": (
            ("target_key", 0, "BINARY"),
            ("created_at", 1, "BINARY"),
            ("revision", 0, "BINARY"),
        ),
        "idx_cayu_eval_judge_calibrations_definition": (
            ("definition_revision", 0, "BINARY"),
            ("created_at", 1, "BINARY"),
            ("revision", 0, "BINARY"),
        ),
    }
    for index_name, expected in expected_indexes.items():
        metadata = next((row for row in index_rows if str(row[1]) == index_name), None)
        actual = tuple(
            (str(row[2]), int(row[3]), str(row[4]).upper())
            for row in connection.execute(f"PRAGMA index_xinfo({index_name})")
            if bool(row[5])
        )
        if (
            metadata is None
            or bool(metadata[2])
            or str(metadata[3]) != "c"
            or bool(metadata[4])
            or actual != expected
        ):
            _raise_eval_judge_calibration_schema_error(index_name)


def _raise_eval_judge_calibration_schema_error(name: str) -> NoReturn:
    raise RuntimeError(
        f"SQLite schema object {name!r} conflicts with Cayu's revision-68 "
        "judge-calibration contract. Run `cayu storage migrate` or restore the "
        "database from a known-good backup."
    )


def _validate_eval_run_invocation_column(connection: sqlite3.Connection) -> None:
    columns = {
        str(row[1]): (str(row[2]).upper(), int(row[3]))
        for row in connection.execute("PRAGMA table_info(cayu_eval_runs)")
    }
    if columns.get("invocation_json") != ("TEXT", 1):
        raise RuntimeError(
            "SQLite schema object 'cayu_eval_runs.invocation_json' conflicts with "
            "Cayu's revision-50 durable eval invocation contract. Run "
            "`cayu storage migrate` or restore the database from a known-good backup."
        )


def _validate_eval_run_scenario_progress_column(connection: sqlite3.Connection) -> None:
    columns = {
        str(row[1]): (str(row[2]).upper(), int(row[3]))
        for row in connection.execute("PRAGMA table_info(cayu_eval_runs)")
    }
    if columns.get("scenario_progress_json") != ("TEXT", 0):
        raise RuntimeError(
            "SQLite schema object 'cayu_eval_runs.scenario_progress_json' conflicts "
            "with Cayu's revision-56 controlled-scenario execution contract. Run "
            "`cayu storage migrate` or restore the database from a known-good backup."
        )


def _validate_eval_run_trial_checkpoint_schema(connection: sqlite3.Connection) -> None:
    run_columns = {
        str(row[1]): (str(row[2]).upper(), int(row[3]), row[4])
        for row in connection.execute("PRAGMA table_info(cayu_eval_runs)")
    }
    if run_columns.get("trial_checkpoint_count") != ("INTEGER", 1, "0"):
        raise RuntimeError(
            "SQLite schema object 'cayu_eval_runs.trial_checkpoint_count' conflicts "
            "with Cayu's revision-74 eval trial recovery contract. Run `cayu storage "
            "migrate` or restore the database from a known-good backup."
        )
    if run_columns.get("trial_checkpoint_bytes") != ("INTEGER", 1, "0"):
        raise RuntimeError(
            "SQLite schema object 'cayu_eval_runs.trial_checkpoint_bytes' conflicts "
            "with Cayu's revision-74 eval trial recovery contract. Run `cayu storage "
            "migrate` or restore the database from a known-good backup."
        )
    checkpoint_columns = tuple(
        (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
        for row in connection.execute("PRAGMA table_info(cayu_eval_run_trial_checkpoints)")
    )
    if checkpoint_columns != (
        ("run_id", "TEXT", 1, 1),
        ("case_id", "TEXT", 1, 2),
        ("trial_number", "INTEGER", 1, 3),
        ("checkpoint_json", "TEXT", 1, 0),
        ("document_bytes", "INTEGER", 1, 0),
    ):
        raise RuntimeError(
            "SQLite schema object 'cayu_eval_run_trial_checkpoints' conflicts with "
            "Cayu's revision-74 eval trial recovery contract. Run `cayu storage migrate` "
            "or restore the database from a known-good backup."
        )
    checkpoint_foreign_keys = tuple(
        (str(row[2]), str(row[3]), str(row[4]), str(row[6]).upper())
        for row in connection.execute("PRAGMA foreign_key_list(cayu_eval_run_trial_checkpoints)")
    )
    if checkpoint_foreign_keys != (("cayu_eval_runs", "run_id", "run_id", "CASCADE"),):
        raise RuntimeError(
            "SQLite schema object 'cayu_eval_run_trial_checkpoints' has an invalid "
            "run ownership constraint. Run `cayu storage migrate` or restore the "
            "database from a known-good backup."
        )
    if run_columns.get("authored_suite_launch_revision") != ("TEXT", 0, None):
        raise RuntimeError(
            "SQLite schema object 'cayu_eval_runs.authored_suite_launch_revision' "
            "conflicts with Cayu's revision-74 authored-suite concurrency contract. "
            "Run `cayu storage migrate` or restore the database from a known-good backup."
        )
    if run_columns.get("authored_suite_launch_lane") != ("INTEGER", 0, None):
        raise RuntimeError(
            "SQLite schema object 'cayu_eval_runs.authored_suite_launch_lane' "
            "conflicts with Cayu's revision-74 authored-suite concurrency contract. "
            "Run `cayu storage migrate` or restore the database from a known-good backup."
        )
    index_row = next(
        (
            row
            for row in connection.execute("PRAGMA index_list(cayu_eval_runs)")
            if str(row[1]) == "idx_cayu_eval_runs_authored_suite_launch_claim"
        ),
        None,
    )
    index = (
        None
        if index_row is None
        else (
            int(index_row[2]),
            int(index_row[4]),
            tuple(
                str(column[2])
                for column in connection.execute(
                    "PRAGMA index_info(idx_cayu_eval_runs_authored_suite_launch_claim)"
                )
            ),
        )
    )
    expected = (
        0,
        1,
        (
            "authored_suite_launch_revision",
            "authored_suite_launch_lane",
            "created_at",
            "run_id",
            "status",
        ),
    )
    if index != expected:
        raise RuntimeError(
            "SQLite schema object 'idx_cayu_eval_runs_authored_suite_launch_claim' "
            "conflicts with Cayu's revision-74 authored-suite concurrency contract. "
            "Run `cayu storage migrate` or restore the database from a known-good backup."
        )
