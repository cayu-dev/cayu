"""PostgreSQL schema checks for verified work and execution-attempt authority.

Callers retain revision gates, validation order and the existing cursor/transaction.
These functions inspect the schema without acquiring connections or applying DDL.
"""

from __future__ import annotations

from typing import Any, NoReturn


async def _validate_work_attempt_lifecycle_schema(cur: Any) -> None:
    await cur.execute(
        "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
        "WHERE table_schema = current_schema() "
        "AND table_name = 'cayu_work_attempt_preparation_holds' ORDER BY ordinal_position"
    )
    if tuple(await cur.fetchall()) != (
        ("hold_id", "text", "NO"),
        ("task_id", "text", "NO"),
        ("request_sha256", "text", "NO"),
        ("receipt_json", "text", "NO"),
    ):
        raise RuntimeError("Postgres preparation hold columns conflict with revision 84.")
    await cur.execute(
        "SELECT c.contype, pg_get_constraintdef(c.oid, TRUE) "
        "FROM pg_constraint AS c JOIN pg_class AS r ON r.oid = c.conrelid "
        "JOIN pg_namespace AS n ON n.oid = r.relnamespace "
        "WHERE n.nspname = current_schema() AND r.relname = 'cayu_work_attempt_preparation_holds'"
    )
    hold_constraints = tuple(
        (str(kind), " ".join(str(definition).lower().split()))
        for kind, definition in await cur.fetchall()
    )
    if any(
        not any(
            kind == expected_kind and fragment in definition
            for kind, definition in hold_constraints
        )
        for expected_kind, fragment in (
            ("p", "primary key (hold_id)"),
            ("f", "foreign key (task_id) references cayu_tasks(id) on delete restrict"),
            ("c", "request_sha256 ~ '^[0-9a-f]{64}$'"),
            ("c", "octet_length(receipt_json) >= 1"),
            ("c", "octet_length(receipt_json) <= 1097728"),
            ("c", "jsonb_typeof(receipt_json::jsonb) = 'object'"),
        )
    ):
        raise RuntimeError("Postgres preparation hold constraints conflict with revision 84.")
    await cur.execute(
        "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
        "WHERE table_schema = current_schema() "
        "AND table_name = 'cayu_work_attempt_lifecycle_receipts' ORDER BY ordinal_position"
    )
    expected = (
        ("admission_id", "text", "NO"),
        ("settlement_id", "text", "NO"),
        ("task_id", "text", "NO"),
        ("request_sha256", "text", "NO"),
        ("retired_contract_binding", "boolean", "NO"),
        ("settled_at", "timestamp with time zone", "NO"),
        ("receipt_json", "text", "NO"),
    )
    if tuple(await cur.fetchall()) != expected:
        raise RuntimeError("Postgres work-attempt lifecycle columns conflict with revision 84.")
    await cur.execute(
        "SELECT c.contype, pg_get_constraintdef(c.oid, TRUE) "
        "FROM pg_constraint AS c JOIN pg_class AS r ON r.oid = c.conrelid "
        "JOIN pg_namespace AS n ON n.oid = r.relnamespace "
        "WHERE n.nspname = current_schema() AND r.relname = 'cayu_work_attempt_lifecycle_receipts'"
    )
    constraints = tuple(
        (str(kind), " ".join(str(definition).lower().split()))
        for kind, definition in await cur.fetchall()
    )
    required = (
        ("p", "primary key (admission_id)"),
        ("u", "unique (settlement_id)"),
        (
            "f",
            "foreign key (admission_id) references cayu_work_attempt_admissions(admission_id) on delete restrict",
        ),
        ("f", "foreign key (task_id) references cayu_tasks(id) on delete restrict"),
        ("c", "request_sha256 ~ '^[0-9a-f]{64}$'"),
        ("c", "octet_length(receipt_json) >= 1"),
        ("c", "octet_length(receipt_json) <= 1097728"),
        ("c", "jsonb_typeof(receipt_json::jsonb) = 'object'"),
    )
    if any(
        not any(
            kind == required_kind and fragment in definition for kind, definition in constraints
        )
        for required_kind, fragment in required
    ):
        raise RuntimeError("Postgres work-attempt lifecycle constraints conflict with revision 84.")
    await cur.execute(
        "SELECT indexdef FROM pg_indexes WHERE schemaname = current_schema() "
        "AND indexname = 'idx_cayu_work_attempt_lifecycle_task'"
    )
    row = await cur.fetchone()
    if row is None or "(task_id, retired_contract_binding)" not in str(row[0]):
        raise RuntimeError("Postgres work-attempt lifecycle index conflicts with revision 84.")


async def _validate_work_attempt_admission_schema(cur: Any) -> None:
    await cur.execute(
        """
            SELECT table_name, column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name IN (
                  'cayu_work_attempt_admissions',
                  'cayu_work_attempt_execution_claims'
              )
            ORDER BY table_name, ordinal_position
            """
    )
    columns = tuple(await cur.fetchall())
    expected_columns = (
        ("cayu_work_attempt_admissions", "admission_id", "text", "NO"),
        ("cayu_work_attempt_admissions", "attempt_id", "text", "NO"),
        ("cayu_work_attempt_admissions", "task_id", "text", "NO"),
        ("cayu_work_attempt_admissions", "session_id", "text", "NO"),
        ("cayu_work_attempt_admissions", "interaction_id", "text", "NO"),
        ("cayu_work_attempt_admissions", "state", "text", "NO"),
        (
            "cayu_work_attempt_admissions",
            "prepare_request_sha256",
            "text",
            "NO",
        ),
        ("cayu_work_attempt_admissions", "current_claim_id", "text", "NO"),
        (
            "cayu_work_attempt_admissions",
            "current_generation",
            "bigint",
            "NO",
        ),
        (
            "cayu_work_attempt_admissions",
            "lease_expires_at",
            "timestamp with time zone",
            "NO",
        ),
        ("cayu_work_attempt_admissions", "admission_json", "jsonb", "NO"),
        ("cayu_work_attempt_execution_claims", "claim_id", "text", "NO"),
        ("cayu_work_attempt_execution_claims", "admission_id", "text", "NO"),
        ("cayu_work_attempt_execution_claims", "generation", "bigint", "NO"),
        (
            "cayu_work_attempt_execution_claims",
            "request_sha256",
            "text",
            "NO",
        ),
        (
            "cayu_work_attempt_execution_claims",
            "lease_expires_at",
            "timestamp with time zone",
            "NO",
        ),
        ("cayu_work_attempt_execution_claims", "is_current", "boolean", "NO"),
        ("cayu_work_attempt_execution_claims", "claim_json", "jsonb", "NO"),
    )
    await cur.execute(
        """
            SELECT indexname, regexp_replace(indexdef, '\\s+', ' ', 'g')
            FROM pg_indexes
            WHERE schemaname = current_schema()
              AND indexname IN (
                  'idx_cayu_work_attempt_admission_interaction',
                  'idx_cayu_work_attempt_admission_session_current',
                  'idx_cayu_work_attempt_admission_task',
                  'idx_cayu_work_attempt_claim_current'
              )
            ORDER BY indexname
            """
    )
    indexes = {str(row[0]): str(row[1]).lower() for row in await cur.fetchall()}
    await cur.execute(
        """
            SELECT table_record.relname, constraint_record.contype,
                   lower(pg_get_constraintdef(constraint_record.oid))
            FROM pg_catalog.pg_constraint AS constraint_record
            JOIN pg_catalog.pg_class AS table_record
              ON table_record.oid = constraint_record.conrelid
            JOIN pg_catalog.pg_namespace AS namespace
              ON namespace.oid = table_record.relnamespace
            WHERE namespace.nspname = current_schema()
              AND table_record.relname IN (
                  'cayu_work_attempt_admissions',
                  'cayu_work_attempt_execution_claims'
              )
            """
    )
    constraints = tuple(await cur.fetchall())
    admission_constraints = tuple(
        (str(kind), str(definition))
        for table, kind, definition in constraints
        if table == "cayu_work_attempt_admissions"
    )
    claim_constraints = tuple(
        (str(kind), str(definition))
        for table, kind, definition in constraints
        if table == "cayu_work_attempt_execution_claims"
    )
    interaction_index = indexes.get("idx_cayu_work_attempt_admission_interaction", "")
    session_index = indexes.get("idx_cayu_work_attempt_admission_session_current", "")
    task_index = indexes.get("idx_cayu_work_attempt_admission_task", "")
    current_index = indexes.get("idx_cayu_work_attempt_claim_current", "")
    state_constraints = tuple(
        definition
        for kind, definition in admission_constraints
        if kind == "c" and "state" in definition
    )
    constraints_valid = (
        any(kind == "p" for kind, _definition in admission_constraints)
        and any(
            kind == "u" and "unique (attempt_id)" in definition
            for kind, definition in admission_constraints
        )
        and any(
            kind == "f"
            and "foreign key (task_id) references cayu_tasks(id) on delete restrict" in definition
            for kind, definition in admission_constraints
        )
        and any(
            all(
                state in definition
                for state in ("'preparing'", "'active'", "'recovering'", "'released'")
            )
            and "'draining'" not in definition
            for definition in state_constraints
        )
        and any(kind == "p" for kind, _definition in claim_constraints)
        and any(
            kind == "u" and "unique (admission_id, generation)" in definition
            for kind, definition in claim_constraints
        )
        and any(
            kind == "f"
            and "foreign key (admission_id) references cayu_work_attempt_admissions(admission_id) on delete restrict"
            in definition
            for kind, definition in claim_constraints
        )
    )
    if (
        columns != expected_columns
        or "unique" not in interaction_index
        or "(session_id, interaction_id)" not in interaction_index
        or "unique" not in session_index
        or "(session_id) where (state <> 'released'::text)" not in session_index
        or "(task_id, current_generation desc)" not in task_index
        or "unique" not in current_index
        or "(admission_id) where is_current" not in current_index
        or not constraints_valid
    ):
        raise RuntimeError(
            "Postgres work-attempt admission schema conflicts with Cayu's "
            "revision-61 durability contract. Run `cayu storage migrate` or "
            "restore the database from a known-good backup."
        )


async def _validate_deferred_interaction_input_payloads(cur: Any) -> None:
    await cur.execute(
        """
            SELECT EXISTS (
                SELECT 1
                FROM cayu_deferred_interaction_inputs
                WHERE jsonb_typeof(source_messages) IS DISTINCT FROM 'object'
                   OR source_messages IS DISTINCT FROM jsonb_build_object(
                       'source_messages', source_messages -> 'source_messages',
                       'initial_transcript_messages',
                       source_messages -> 'initial_transcript_messages'
                   )
                   OR jsonb_typeof(source_messages -> 'source_messages')
                      IS DISTINCT FROM 'array'
                   OR (
                       jsonb_typeof(source_messages -> 'initial_transcript_messages')
                       IS DISTINCT FROM 'array'
                       AND jsonb_typeof(source_messages -> 'initial_transcript_messages')
                           IS DISTINCT FROM 'null'
                   )
            )
            """
    )
    row = await cur.fetchone()
    if row is None or row[0] is True:
        raise RuntimeError(
            "Postgres deferred interaction input conflicts with Cayu's revision-62 "
            "durable payload contract."
        )


async def _validate_work_attempt_continuation_authority(cur: Any) -> None:
    await cur.execute(
        """
            SELECT EXISTS (
                SELECT 1
                FROM cayu_work_attempt_admissions AS admission
                LEFT JOIN cayu_work_attempt_admissions AS predecessor
                  ON predecessor.attempt_id = (
                      admission.admission_json #>> '{continuation,prior_attempt_id}'
                  )
                WHERE admission.admission_json -> 'continuation' IS NOT NULL
                  AND jsonb_typeof(admission.admission_json -> 'continuation')
                      IS DISTINCT FROM 'null'
                  AND (
                      jsonb_typeof(admission.admission_json -> 'continuation')
                          IS DISTINCT FROM 'object'
                      OR jsonb_typeof(
                          admission.admission_json
                              #> '{continuation,prior_attempt_id}'
                      ) IS DISTINCT FROM 'string'
                      OR jsonb_typeof(
                          admission.admission_json
                              #> '{continuation,prior_admission_id}'
                      ) IS DISTINCT FROM 'string'
                      OR predecessor.admission_id IS NULL
                      OR admission.admission_json
                          #>> '{continuation,prior_admission_id}'
                          IS DISTINCT FROM predecessor.admission_id
                  )
            )
            """
    )
    row = await cur.fetchone()
    if row is None or row[0] is True:
        raise RuntimeError(
            "Postgres work-attempt continuation conflicts with Cayu's revision-62 "
            "durable authority contract."
        )


async def _validate_verified_work_schema(
    cur: Any,
    *,
    require_verifier_profiles: bool,
) -> None:
    await cur.execute(
        """
            SELECT data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'cayu_tasks'
              AND column_name = 'work_contract'
            """
    )
    if await cur.fetchone() != ("jsonb", "YES"):
        _raise_verified_work_schema_error("cayu_tasks.work_contract")
    required_columns = {
        "cayu_work_contracts": (
            ("contract_id", "text", "NO"),
            ("version", "bigint", "NO"),
            ("fingerprint", "text", "NO"),
            ("contract_json", "jsonb", "NO"),
        ),
        "cayu_task_session_execution_authority": (
            ("session_id", "text", "NO"),
            ("authority_kind", "text", "NO"),
            ("committed_at", "timestamp with time zone", "NO"),
        ),
        "cayu_work_attempts": (
            ("attempt_id", "text", "NO"),
            ("task_id", "text", "NO"),
            ("ordinal", "bigint", "NO"),
            ("request_sha256", "text", "NO"),
            ("started_at", "timestamp with time zone", "NO"),
            ("attempt_json", "jsonb", "NO"),
        ),
        "cayu_completion_proposals": (
            ("proposal_id", "text", "NO"),
            ("attempt_id", "text", "NO"),
            ("task_id", "text", "NO"),
            ("request_sha256", "text", "NO"),
            ("proposed_at", "timestamp with time zone", "NO"),
            ("proposal_json", "jsonb", "NO"),
        ),
        "cayu_completion_verification_claims": (
            ("claim_id", "text", "NO"),
            ("proposal_id", "text", "NO"),
            ("attempt_number", "bigint", "NO"),
            ("request_sha256", "text", "NO"),
            ("lease_expires_at", "timestamp with time zone", "NO"),
            ("is_current", "boolean", "NO"),
            ("claim_json", "jsonb", "NO"),
        ),
        "cayu_completion_decisions": (
            ("decision_id", "text", "NO"),
            ("proposal_id", "text", "NO"),
            ("task_id", "text", "NO"),
            ("attempt_id", "text", "NO"),
            ("claim_id", "text", "NO"),
            ("verdict", "text", "NO"),
            ("gap_fingerprint", "text", "NO"),
            ("request_sha256", "text", "NO"),
            ("decided_at", "timestamp with time zone", "NO"),
            ("decision_json", "jsonb", "NO"),
        ),
        "cayu_completion_decision_application_receipts": (
            ("task_id", "text", "NO"),
            ("idempotency_key", "text", "NO"),
            ("decision_id", "text", "NO"),
            ("request_sha256", "text", "NO"),
            ("applied_at", "timestamp with time zone", "NO"),
            ("receipt_json", "jsonb", "NO"),
        ),
    }
    if require_verifier_profiles:
        required_columns["cayu_completion_verification_claims"] += (
            ("verifier_profile_fingerprint", "text", "NO"),
        )
        required_columns["cayu_completion_decisions"] += (
            ("verifier_profile_fingerprint", "text", "NO"),
        )
        required_columns["cayu_completion_verifier_profiles"] = (
            ("proposal_id", "text", "NO"),
            ("task_id", "text", "NO"),
            ("attempt_id", "text", "NO"),
            ("profile_fingerprint", "text", "NO"),
            ("request_sha256", "text", "NO"),
            ("prepared_at", "timestamp with time zone", "NO"),
            ("profile_json", "jsonb", "NO"),
        )
    contract_revision = 58 if require_verifier_profiles else 49
    for table, expected in required_columns.items():
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
            _raise_verified_work_schema_error(
                table,
                contract_revision=contract_revision,
            )

    required_constraints: dict[
        str,
        tuple[tuple[str, tuple[str, ...]], ...],
    ] = {
        "cayu_work_contracts": (
            ("p", ("primary key (contract_id, version)",)),
            ("c", ("version >= 1",)),
            ("c", ("fingerprint", "[0-9a-f]{64}")),
        ),
        "cayu_task_session_execution_authority": (
            ("p", ("primary key (session_id)",)),
            ("c", ("authority_kind", "ordinary", "contracted")),
        ),
        "cayu_work_attempts": (
            ("p", ("primary key (attempt_id)",)),
            ("u", ("unique (task_id, ordinal)",)),
            ("f", ("foreign key (task_id)", "references cayu_tasks(id)")),
            ("c", ("ordinal >= 1",)),
            ("c", ("request_sha256", "[0-9a-f]{64}")),
        ),
        "cayu_completion_proposals": (
            ("p", ("primary key (proposal_id)",)),
            ("u", ("unique (attempt_id)",)),
            (
                "f",
                (
                    "foreign key (attempt_id)",
                    "references cayu_work_attempts(attempt_id)",
                ),
            ),
            ("f", ("foreign key (task_id)", "references cayu_tasks(id)")),
            ("c", ("request_sha256", "[0-9a-f]{64}")),
        ),
        "cayu_completion_verification_claims": (
            ("p", ("primary key (claim_id)",)),
            ("u", ("unique (proposal_id, attempt_number)",)),
            (
                "f",
                (
                    "foreign key (proposal_id)",
                    "references cayu_completion_proposals(proposal_id)",
                ),
            ),
            ("c", ("attempt_number >= 1",)),
            ("c", ("request_sha256", "[0-9a-f]{64}")),
        ),
        "cayu_completion_decisions": (
            ("p", ("primary key (decision_id)",)),
            ("u", ("unique (proposal_id)",)),
            (
                "f",
                (
                    "foreign key (proposal_id)",
                    "references cayu_completion_proposals(proposal_id)",
                ),
            ),
            ("f", ("foreign key (task_id)", "references cayu_tasks(id)")),
            (
                "f",
                (
                    "foreign key (attempt_id)",
                    "references cayu_work_attempts(attempt_id)",
                ),
            ),
            (
                "f",
                (
                    "foreign key (claim_id)",
                    "references cayu_completion_verification_claims(claim_id)",
                ),
            ),
            ("c", ("verdict", "accepted", "rejected", "blocked", "needs_review")),
            ("c", ("gap_fingerprint", "[0-9a-f]{64}")),
            ("c", ("request_sha256", "[0-9a-f]{64}")),
        ),
        "cayu_completion_decision_application_receipts": (
            ("p", ("primary key (task_id, idempotency_key)",)),
            ("u", ("unique (decision_id)",)),
            ("f", ("foreign key (task_id)", "references cayu_tasks(id)")),
            (
                "f",
                (
                    "foreign key (decision_id)",
                    "references cayu_completion_decisions(decision_id)",
                ),
            ),
            ("c", ("request_sha256", "[0-9a-f]{64}")),
        ),
    }
    if require_verifier_profiles:
        required_constraints["cayu_completion_verification_claims"] += (
            ("c", ("verifier_profile_fingerprint", "[0-9a-f]{64}")),
        )
        required_constraints["cayu_completion_decisions"] += (
            ("c", ("verifier_profile_fingerprint", "[0-9a-f]{64}")),
        )
        required_constraints["cayu_completion_verifier_profiles"] = (
            ("p", ("primary key (proposal_id)",)),
            ("u", ("unique (attempt_id)",)),
            (
                "f",
                (
                    "foreign key (proposal_id)",
                    "references cayu_completion_proposals(proposal_id)",
                ),
            ),
            ("f", ("foreign key (task_id)", "references cayu_tasks(id)")),
            (
                "f",
                (
                    "foreign key (attempt_id)",
                    "references cayu_work_attempts(attempt_id)",
                ),
            ),
            ("c", ("profile_fingerprint", "[0-9a-f]{64}")),
            ("c", ("request_sha256", "[0-9a-f]{64}")),
        )
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
            """,
        (list(required_constraints),),
    )
    actual_constraints: dict[str, list[tuple[str, str]]] = {}
    for table, constraint_type, definition in await cur.fetchall():
        actual_constraints.setdefault(str(table), []).append(
            (str(constraint_type), " ".join(str(definition).lower().split()))
        )
    for table, expected_constraints in required_constraints.items():
        actual = actual_constraints.get(table, [])
        if any(
            not any(
                candidate_type == constraint_type
                and all(fragment in definition for fragment in fragments)
                for candidate_type, definition in actual
            )
            for constraint_type, fragments in expected_constraints
        ):
            _raise_verified_work_schema_error(
                table,
                contract_revision=contract_revision,
            )

    required_indexes = {
        "idx_cayu_completion_claim_current": (
            "cayu_completion_verification_claims",
            True,
            "(proposal_id)",
            "is_current",
        ),
        "idx_cayu_completion_decisions_task_gap": (
            "cayu_completion_decisions",
            False,
            "(task_id, verdict, gap_fingerprint)",
            None,
        ),
        "idx_cayu_tasks_contracted_session": (
            "cayu_tasks",
            False,
            "(session_id, created_at, id)",
            "work_contract is not null",
        ),
        "idx_cayu_work_attempts_task_latest": (
            "cayu_work_attempts",
            False,
            "(task_id, ordinal desc)",
            None,
        ),
    }
    if require_verifier_profiles:
        required_indexes["idx_cayu_completion_verifier_profiles_task"] = (
            "cayu_completion_verifier_profiles",
            False,
            "(task_id, attempt_id)",
            None,
        )
    await cur.execute(
        """
            SELECT index_record.relname,
                   table_record.relname,
                   index_state.indisunique,
                   index_state.indisvalid,
                   index_state.indisready,
                   pg_get_indexdef(index_record.oid),
                   pg_get_expr(index_state.indpred, index_state.indrelid)
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
        (list(required_indexes),),
    )
    actual_indexes = {
        str(row[0]): (
            str(row[1]),
            bool(row[2]),
            bool(row[3]),
            bool(row[4]),
            " ".join(str(row[5]).lower().split()),
            None if row[6] is None else " ".join(str(row[6]).lower().split()),
        )
        for row in await cur.fetchall()
    }
    for index, (table, unique, columns, predicate) in required_indexes.items():
        actual = actual_indexes.get(index)
        if (
            actual is None
            or actual[0] != table
            or actual[1] is not unique
            or not actual[2]
            or not actual[3]
            or columns not in actual[4]
            or (predicate is None and actual[5] is not None)
            or (predicate is not None and predicate not in (actual[5] or ""))
        ):
            _raise_verified_work_schema_error(
                index,
                contract_revision=contract_revision,
            )


def _raise_verified_work_schema_error(
    name: str,
    *,
    contract_revision: int = 49,
) -> NoReturn:
    raise RuntimeError(
        f"Postgres schema object {name!r} conflicts with Cayu's "
        f"revision-{contract_revision} "
        "verified-work contract. Run `cayu storage migrate` or restore the "
        "database from a known-good backup."
    )
