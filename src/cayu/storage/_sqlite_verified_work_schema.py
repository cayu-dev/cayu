"""SQLite schema checks for verified work, admission and lifecycle receipts.

These checks inspect existing schema through a supplied connection. Reconciliation
and migration callers retain revision selection, ordering and transaction ownership.
"""

from __future__ import annotations

import sqlite3


def _validate_verified_work_schema(
    connection: sqlite3.Connection,
    *,
    require_verifier_profiles: bool,
) -> None:
    task_columns = {
        str(row[1]): (str(row[2]).upper(), int(row[3]))
        for row in connection.execute("PRAGMA table_info(cayu_tasks)")
    }
    if task_columns.get("work_contract_json") != ("TEXT", 0):
        raise RuntimeError(
            "SQLite task work-contract storage conflicts with Cayu's revision-49 contract."
        )
    required_columns = {
        "cayu_work_contracts": (
            ("contract_id", "TEXT", 1, 1),
            ("version", "INTEGER", 1, 2),
            ("fingerprint", "TEXT", 1, 0),
            ("contract_json", "TEXT", 1, 0),
        ),
        "cayu_task_session_execution_authority": (
            ("session_id", "TEXT", 1, 1),
            ("authority_kind", "TEXT", 1, 0),
            ("committed_at", "TEXT", 1, 0),
        ),
        "cayu_work_attempts": (
            ("attempt_id", "TEXT", 1, 1),
            ("task_id", "TEXT", 1, 0),
            ("ordinal", "INTEGER", 1, 0),
            ("request_sha256", "TEXT", 1, 0),
            ("started_at", "TEXT", 1, 0),
            ("attempt_json", "TEXT", 1, 0),
        ),
        "cayu_completion_proposals": (
            ("proposal_id", "TEXT", 1, 1),
            ("attempt_id", "TEXT", 1, 0),
            ("task_id", "TEXT", 1, 0),
            ("request_sha256", "TEXT", 1, 0),
            ("proposed_at", "TEXT", 1, 0),
            ("proposal_json", "TEXT", 1, 0),
        ),
        "cayu_completion_verification_claims": (
            ("claim_id", "TEXT", 1, 1),
            ("proposal_id", "TEXT", 1, 0),
            ("attempt_number", "INTEGER", 1, 0),
            ("request_sha256", "TEXT", 1, 0),
            ("lease_expires_at", "TEXT", 1, 0),
            ("is_current", "INTEGER", 1, 0),
            ("claim_json", "TEXT", 1, 0),
        ),
        "cayu_completion_decisions": (
            ("decision_id", "TEXT", 1, 1),
            ("proposal_id", "TEXT", 1, 0),
            ("task_id", "TEXT", 1, 0),
            ("attempt_id", "TEXT", 1, 0),
            ("claim_id", "TEXT", 1, 0),
            ("verdict", "TEXT", 1, 0),
            ("gap_fingerprint", "TEXT", 1, 0),
            ("request_sha256", "TEXT", 1, 0),
            ("decided_at", "TEXT", 1, 0),
            ("decision_json", "TEXT", 1, 0),
        ),
        "cayu_completion_decision_application_receipts": (
            ("task_id", "TEXT", 1, 1),
            ("idempotency_key", "TEXT", 1, 2),
            ("decision_id", "TEXT", 1, 0),
            ("request_sha256", "TEXT", 1, 0),
            ("applied_at", "TEXT", 1, 0),
            ("receipt_json", "TEXT", 1, 0),
        ),
    }
    if require_verifier_profiles:
        required_columns["cayu_completion_verification_claims"] += (
            ("verifier_profile_fingerprint", "TEXT", 0, 0),
        )
        required_columns["cayu_completion_decisions"] += (
            ("verifier_profile_fingerprint", "TEXT", 0, 0),
        )
        required_columns["cayu_completion_verifier_profiles"] = (
            ("proposal_id", "TEXT", 1, 1),
            ("task_id", "TEXT", 1, 0),
            ("attempt_id", "TEXT", 1, 0),
            ("profile_fingerprint", "TEXT", 1, 0),
            ("request_sha256", "TEXT", 1, 0),
            ("prepared_at", "TEXT", 1, 0),
            ("profile_json", "TEXT", 1, 0),
        )
    contract_revision = 58 if require_verifier_profiles else 49
    for table, expected in required_columns.items():
        actual = tuple(
            (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
            for row in connection.execute(f"PRAGMA table_info({table})")
        )
        if actual != expected:
            raise RuntimeError(
                f"SQLite schema object {table!r} conflicts with Cayu's "
                f"revision-{contract_revision} verified-work contract."
            )

    def normalized_sql(name: str, object_type: str) -> str:
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = ? AND name = ?",
            (object_type, name),
        ).fetchone()
        return "" if row is None or row[0] is None else " ".join(str(row[0]).lower().split())

    if (
        "work_contract_json text check (work_contract_json is null or "
        "json_valid(work_contract_json))" not in normalized_sql("cayu_tasks", "table")
    ):
        raise RuntimeError(
            "SQLite task work-contract storage conflicts with Cayu's revision-49 contract."
        )

    required_table_fragments = {
        "cayu_work_contracts": (
            "primary key (contract_id, version)",
            "check (version >= 1)",
            "length(fingerprint) = 64",
            "fingerprint not glob '*[^0-9a-f]*'",
            "check (json_valid(contract_json))",
        ),
        "cayu_task_session_execution_authority": (
            "session_id text not null primary key",
            "authority_kind in ('ordinary', 'contracted')",
        ),
        "cayu_work_attempts": (
            "references cayu_tasks(id) on delete restrict",
            "unique (task_id, ordinal)",
            "check (ordinal >= 1)",
            "length(request_sha256) = 64",
            "request_sha256 not glob '*[^0-9a-f]*'",
            "check (json_valid(attempt_json))",
        ),
        "cayu_completion_proposals": (
            "attempt_id text not null unique references cayu_work_attempts(attempt_id) on delete restrict",
            "task_id text not null references cayu_tasks(id) on delete restrict",
            "length(request_sha256) = 64",
            "request_sha256 not glob '*[^0-9a-f]*'",
            "check (json_valid(proposal_json))",
        ),
        "cayu_completion_verification_claims": (
            "references cayu_completion_proposals(proposal_id) on delete restrict",
            "unique (proposal_id, attempt_number)",
            "check (attempt_number >= 1)",
            "length(request_sha256) = 64",
            "request_sha256 not glob '*[^0-9a-f]*'",
            "check (is_current in (0, 1))",
            "check (json_valid(claim_json))",
        ),
        "cayu_completion_decisions": (
            "proposal_id text not null unique references cayu_completion_proposals(proposal_id) on delete restrict",
            "task_id text not null references cayu_tasks(id) on delete restrict",
            "references cayu_work_attempts(attempt_id) on delete restrict",
            "references cayu_completion_verification_claims(claim_id) on delete restrict",
            "verdict in ('accepted', 'rejected', 'blocked', 'needs_review')",
            "length(gap_fingerprint) = 64",
            "gap_fingerprint not glob '*[^0-9a-f]*'",
            "length(request_sha256) = 64",
            "request_sha256 not glob '*[^0-9a-f]*'",
            "check (json_valid(decision_json))",
        ),
        "cayu_completion_decision_application_receipts": (
            "primary key (task_id, idempotency_key)",
            "task_id text not null references cayu_tasks(id) on delete restrict",
            "decision_id text not null unique references cayu_completion_decisions(decision_id) on delete restrict",
            "length(request_sha256) = 64",
            "request_sha256 not glob '*[^0-9a-f]*'",
            "check (json_valid(receipt_json))",
        ),
    }
    if require_verifier_profiles:
        required_table_fragments["cayu_completion_verification_claims"] += (
            "verifier_profile_fingerprint is not null",
            "length(verifier_profile_fingerprint) = 64",
            "verifier_profile_fingerprint not glob '*[^0-9a-f]*'",
        )
        required_table_fragments["cayu_completion_decisions"] += (
            "verifier_profile_fingerprint is not null",
            "length(verifier_profile_fingerprint) = 64",
            "verifier_profile_fingerprint not glob '*[^0-9a-f]*'",
        )
        required_table_fragments["cayu_completion_verifier_profiles"] = (
            "proposal_id text not null primary key",
            "references cayu_completion_proposals(proposal_id) on delete restrict",
            "task_id text not null references cayu_tasks(id) on delete restrict",
            "attempt_id text not null unique references cayu_work_attempts(attempt_id) on delete restrict",
            "length(profile_fingerprint) = 64",
            "profile_fingerprint not glob '*[^0-9a-f]*'",
            "length(request_sha256) = 64",
            "request_sha256 not glob '*[^0-9a-f]*'",
            "check (json_valid(profile_json))",
        )
    for table, fragments in required_table_fragments.items():
        definition = normalized_sql(table, "table")
        if any(fragment not in definition for fragment in fragments):
            raise RuntimeError(
                f"SQLite schema object {table!r} conflicts with Cayu's "
                f"revision-{contract_revision} verified-work contract."
            )
    required_indexes = {
        "idx_cayu_completion_claim_current": (
            "cayu_completion_verification_claims",
            ("proposal_id",),
            True,
            "where is_current = 1",
        ),
        "idx_cayu_completion_decisions_task_gap": (
            "cayu_completion_decisions",
            ("task_id", "verdict", "gap_fingerprint"),
            False,
            None,
        ),
        "idx_cayu_tasks_contracted_session": (
            "cayu_tasks",
            ("session_id", "created_at", "id"),
            False,
            "where work_contract_json is not null",
        ),
        "idx_cayu_work_attempts_task_latest": (
            "cayu_work_attempts",
            ("task_id", "ordinal"),
            False,
            None,
        ),
    }
    if require_verifier_profiles:
        required_indexes["idx_cayu_completion_verifier_profiles_task"] = (
            "cayu_completion_verifier_profiles",
            ("task_id", "attempt_id"),
            False,
            None,
        )
    for index, (table, columns, unique, predicate) in required_indexes.items():
        index_rows = {str(row[1]): row for row in connection.execute(f"PRAGMA index_list({table})")}
        row = index_rows.get(index)
        actual_columns = tuple(
            str(column[2]) for column in connection.execute(f"PRAGMA index_info({index})")
        )
        definition = normalized_sql(index, "index")
        if (
            row is None
            or bool(row[2]) is not unique
            or actual_columns != columns
            or (predicate is None and int(row[4]) != 0)
            or (predicate is not None and (int(row[4]) != 1 or predicate not in definition))
        ):
            raise RuntimeError(
                f"SQLite schema object {index!r} conflicts with Cayu's "
                "revision-49 verified-work contract."
            )


def _validate_work_attempt_admission_schema(connection: sqlite3.Connection) -> None:
    admission_columns = tuple(
        (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
        for row in connection.execute("PRAGMA table_info(cayu_work_attempt_admissions)")
    )
    claim_columns = tuple(
        (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
        for row in connection.execute("PRAGMA table_info(cayu_work_attempt_execution_claims)")
    )
    expected_admission_columns = (
        ("admission_id", "TEXT", 0, 1),
        ("attempt_id", "TEXT", 1, 0),
        ("task_id", "TEXT", 1, 0),
        ("session_id", "TEXT", 1, 0),
        ("interaction_id", "TEXT", 1, 0),
        ("state", "TEXT", 1, 0),
        ("prepare_request_sha256", "TEXT", 1, 0),
        ("current_claim_id", "TEXT", 1, 0),
        ("current_generation", "INTEGER", 1, 0),
        ("lease_expires_at", "TEXT", 1, 0),
        ("admission_json", "TEXT", 1, 0),
    )
    expected_claim_columns = (
        ("claim_id", "TEXT", 0, 1),
        ("admission_id", "TEXT", 1, 0),
        ("generation", "INTEGER", 1, 0),
        ("request_sha256", "TEXT", 1, 0),
        ("lease_expires_at", "TEXT", 1, 0),
        ("is_current", "INTEGER", 1, 0),
        ("claim_json", "TEXT", 1, 0),
    )
    named_index_properties: dict[str, tuple[int, int]] = {}
    for table_name in (
        "cayu_work_attempt_admissions",
        "cayu_work_attempt_execution_claims",
    ):
        for row in connection.execute(f"PRAGMA index_list({table_name})"):
            index_name = str(row[1])
            if index_name.startswith("sqlite_autoindex_"):
                continue
            named_index_properties[index_name] = (int(row[2]), int(row[4]))
    named_indexes = {
        "idx_cayu_work_attempt_admission_interaction": (
            *named_index_properties.get("idx_cayu_work_attempt_admission_interaction", (-1, -1)),
            tuple(
                str(column[2])
                for column in connection.execute(
                    "PRAGMA index_info(idx_cayu_work_attempt_admission_interaction)"
                )
            ),
        ),
        "idx_cayu_work_attempt_admission_task": (
            *named_index_properties.get("idx_cayu_work_attempt_admission_task", (-1, -1)),
            tuple(
                str(column[2])
                for column in connection.execute(
                    "PRAGMA index_info(idx_cayu_work_attempt_admission_task)"
                )
            ),
        ),
        "idx_cayu_work_attempt_admission_session_current": (
            *named_index_properties.get(
                "idx_cayu_work_attempt_admission_session_current",
                (-1, -1),
            ),
            tuple(
                str(column[2])
                for column in connection.execute(
                    "PRAGMA index_info(idx_cayu_work_attempt_admission_session_current)"
                )
            ),
        ),
        "idx_cayu_work_attempt_claim_current": (
            *named_index_properties.get("idx_cayu_work_attempt_claim_current", (-1, -1)),
            tuple(
                str(column[2])
                for column in connection.execute(
                    "PRAGMA index_info(idx_cayu_work_attempt_claim_current)"
                )
            ),
        ),
    }
    table_sql = {
        str(row[0]): " ".join(str(row[1]).lower().split())
        for row in connection.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'table' AND name IN (?, ?)",
            (
                "cayu_work_attempt_admissions",
                "cayu_work_attempt_execution_claims",
            ),
        )
    }
    session_index_row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'index' "
        "AND name = 'idx_cayu_work_attempt_admission_session_current'"
    ).fetchone()
    session_index_sql = (
        "" if session_index_row is None else " ".join(str(session_index_row[0]).lower().split())
    )
    admission_sql = table_sql.get("cayu_work_attempt_admissions", "")
    claim_sql = table_sql.get("cayu_work_attempt_execution_claims", "")
    if (
        admission_columns != expected_admission_columns
        or claim_columns != expected_claim_columns
        or named_indexes.get("idx_cayu_work_attempt_admission_interaction")
        != (1, 0, ("session_id", "interaction_id"))
        or named_indexes.get("idx_cayu_work_attempt_admission_task")
        != (0, 0, ("task_id", "current_generation"))
        or named_indexes.get("idx_cayu_work_attempt_admission_session_current")
        != (1, 1, ("session_id",))
        or "where state != 'released'" not in session_index_sql
        or named_indexes.get("idx_cayu_work_attempt_claim_current") != (1, 1, ("admission_id",))
        or "references cayu_tasks(id) on delete restrict" not in admission_sql
        or any(
            state not in admission_sql
            for state in ("'preparing'", "'active'", "'recovering'", "'released'")
        )
        or "'draining'" in admission_sql
        or "unique (admission_id, generation)" not in claim_sql
        or "references cayu_work_attempt_admissions(admission_id) on delete restrict"
        not in claim_sql
    ):
        raise RuntimeError(
            "SQLite work-attempt admission schema conflicts with Cayu's revision-61 "
            "durability contract. Run `cayu storage migrate` or restore the "
            "database from a known-good backup."
        )


def _validate_work_attempt_lifecycle_schema(connection: sqlite3.Connection) -> None:
    hold_columns = tuple(
        (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
        for row in connection.execute("PRAGMA table_info(cayu_work_attempt_preparation_holds)")
    )
    hold_row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' "
        "AND name = 'cayu_work_attempt_preparation_holds'"
    ).fetchone()
    hold_definition = "" if hold_row is None else " ".join(str(hold_row[0]).lower().split())
    if hold_columns != (
        ("hold_id", "TEXT", 1, 1),
        ("task_id", "TEXT", 1, 0),
        ("request_sha256", "TEXT", 1, 0),
        ("receipt_json", "TEXT", 1, 0),
    ) or any(
        fragment not in hold_definition
        for fragment in (
            "references cayu_tasks(id) on delete restrict",
            "length(request_sha256) = 64",
            "request_sha256 not glob '*[^0-9a-f]*'",
            "json_valid(receipt_json)",
            "json_type(receipt_json) = 'object'",
            "length(cast(receipt_json as blob)) between 1 and 1097728",
        )
    ):
        raise RuntimeError(
            "SQLite work-attempt preparation hold schema conflicts with revision 84."
        )
    columns = tuple(
        (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5]))
        for row in connection.execute("PRAGMA table_info(cayu_work_attempt_lifecycle_receipts)")
    )
    expected = (
        ("admission_id", "TEXT", 1, 1),
        ("settlement_id", "TEXT", 1, 0),
        ("task_id", "TEXT", 1, 0),
        ("request_sha256", "TEXT", 1, 0),
        ("retired_contract_binding", "INTEGER", 1, 0),
        ("settled_at", "TEXT", 1, 0),
        ("receipt_json", "TEXT", 1, 0),
    )
    row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' "
        "AND name = 'cayu_work_attempt_lifecycle_receipts'"
    ).fetchone()
    definition = "" if row is None else " ".join(str(row[0]).lower().split())
    index_columns = tuple(
        str(row[2])
        for row in connection.execute("PRAGMA index_info(idx_cayu_work_attempt_lifecycle_task)")
    )
    required = (
        "settlement_id text not null unique",
        "references cayu_work_attempt_admissions(admission_id) on delete restrict",
        "references cayu_tasks(id) on delete restrict",
        "length(request_sha256) = 64",
        "request_sha256 not glob '*[^0-9a-f]*'",
        "retired_contract_binding in (0, 1)",
        "json_valid(receipt_json)",
        "json_type(receipt_json) = 'object'",
        "length(cast(receipt_json as blob)) between 1 and 1097728",
    )
    if (
        columns != expected
        or index_columns != ("task_id", "retired_contract_binding")
        or any(fragment not in definition for fragment in required)
    ):
        raise RuntimeError("SQLite work-attempt lifecycle schema conflicts with revision 84.")
