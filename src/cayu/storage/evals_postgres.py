from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from time import monotonic
from typing import Any, ClassVar, LiteralString, cast
from uuid import uuid4

from cayu.evals.calibration import (
    EVAL_JUDGE_CALIBRATION_MAX_BYTES,
    EvalJudgeCalibrationReportV1,
    eval_judge_calibration_report_from_json,
)
from cayu.evals.corpus import (
    EVAL_CORPUS_MAX_BYTES,
    EvalCorpusDocument,
    EvalCorpusInspectionV1,
    _portable_id,
    _sha256_revision,
    eval_corpus_from_json,
)
from cayu.evals.execution import CORPUS_EXECUTION_RESULT_MAX_BYTES, CorpusExecutionResult
from cayu.evals.execution_reporting import corpus_execution_result_from_json
from cayu.evals.results import (
    CapturedEvaluationResultV1,
    EvalResultOrigin,
    EvalResultTargetIdentityV1,
    captured_evaluation_result_from_json,
)
from cayu.evals.scenario import (
    EVAL_SCENARIO_MAX_BYTES,
    EvalScenarioDocumentV2,
    eval_scenario_from_json,
)
from cayu.evals.store import (
    EVAL_RUN_TRIAL_CHECKPOINTS_MAX_BYTES,
    EVAL_RUN_TRIAL_CHECKPOINTS_MAX_ITEMS,
    TERMINAL_EVAL_RUN_STATUSES,
    EvalAuthoredSuiteCatalogEntry,
    EvalAuthoredSuiteCatalogPage,
    EvalAuthoredSuiteCatalogQuery,
    EvalAuthoredSuiteConflict,
    EvalBaselineConflict,
    EvalBaselineKey,
    EvalBaselineMutationRecord,
    EvalBaselineRecord,
    EvalBaselineUpdate,
    EvalCaseCatalogEntry,
    EvalCaseCatalogPage,
    EvalCaseCatalogQuery,
    EvalCatalogQuery,
    EvalCorpusCatalogEntry,
    EvalCorpusCatalogPage,
    EvalCorpusConflict,
    EvalJudgeCalibrationConflict,
    EvalResultConflict,
    EvalResultPage,
    EvalResultQuery,
    EvalResultRecord,
    EvalRunAdmissionConflict,
    EvalRunClaim,
    EvalRunClaimLost,
    EvalRunFailureCode,
    EvalRunLease,
    EvalRunObservation,
    EvalRunOwnership,
    EvalRunPage,
    EvalRunQuery,
    EvalRunRecord,
    EvalRunRequest,
    EvalRunResultSummary,
    EvalRunSpec,
    EvalRunStateConflict,
    EvalRunStatus,
    EvalRunTrialCheckpoint,
    EvalScenarioApprovalDecisionRecord,
    EvalScenarioApprovalSubmission,
    EvalScenarioCatalogEntry,
    EvalScenarioCatalogPage,
    EvalScenarioCatalogQuery,
    EvalScenarioConflict,
    EvalScenarioRunProgress,
    EvalScenarioTrialPhase,
    EvalScenarioTrialProgress,
    EvalStore,
    EvalStoreResultTooLarge,
    EvalSuiteCatalogEntry,
    EvalSuiteCatalogPage,
    EvalSuiteCatalogQuery,
    _bounded_authored_suite_page,
    _bounded_case_page,
    _bounded_corpus_page,
    _bounded_result_page,
    _bounded_run_page,
    _bounded_scenario_page,
    _bounded_suite_page,
    _claim_target_keys,
    _copy_query,
    _exact_model,
    _idempotency_key,
    _lease_seconds,
    _prepare_authored_suite_for_store,
    _prepare_baseline_update_for_store,
    _prepare_captured_result_for_store,
    _prepare_corpus_catalog_for_store,
    _prepare_judge_calibration_for_store,
    _prepare_result_for_store,
    _prepare_run_request_for_store,
    _prepare_scenario_catalog_for_store,
    _prepare_trial_checkpoint_for_store,
    _read_limit,
    _scenario_progress_for_claim,
    _store_identifier,
    _validate_baseline_result,
    _validate_trial_checkpoints_for_result,
    _validated_trial_checkpoints,
    authored_suite_catalog_entry,
    authored_suite_scenario_cases,
    decode_authored_suite_cursor,
    decode_case_cursor,
    decode_corpus_cursor,
    decode_result_cursor,
    decode_run_cursor,
    decode_scenario_cursor,
    decode_suite_cursor,
    eval_result_record,
    eval_run_invocation_from_json,
    eval_run_trial_checkpoint_from_json,
    result_summary,
    validate_authored_suite_scenario,
    validate_result_for_run,
)
from cayu.evals.suite_authoring import (
    EVAL_SUITE_AUTHORING_MAX_BYTES,
    EvalSuiteDocument,
    eval_suite_document_from_json,
)
from cayu.storage.postgres import _PostgresStoreBase

_POSTGRES_EVAL_MIN_REQUIRED_REVISION = 74

logger = logging.getLogger(__name__)

_RUN_COLUMNS = """
    run_id,
    idempotency_key,
    corpus_revision,
    target_key,
    suite_id,
    suite_revision,
    max_concurrency,
    invocation_json,
    status,
    created_at,
    updated_at,
    started_at,
    finished_at,
    cancel_requested_at,
    claim_id,
    ownership_epoch,
    lease_expires_at,
    result_revision,
    result_status,
    result_score,
    result_duration_ms,
    failure_code,
    scenario_progress_json,
    trial_checkpoint_count,
    trial_checkpoint_bytes,
    authored_suite_launch_revision,
    authored_suite_launch_lane
"""

_RUN_OBSERVATION_COLUMNS = """
    run_id,
    status,
    updated_at,
    ownership_epoch,
    lease_expires_at
"""

_RESULT_RECORD_COLUMNS = """
    revision,
    origin,
    target_key,
    corpus_revision,
    suite_id,
    suite_revision,
    application_release_id,
    app_manifest_schema_version,
    app_manifest_fingerprint,
    result_status,
    result_score,
    document_bytes,
    created_at
"""


async def _database_now(cur: Any) -> datetime:
    """Read the shared PostgreSQL clock for persisted lifecycle decisions."""

    await cur.execute("SELECT clock_timestamp()")
    row = await cur.fetchone()
    if row is None or not isinstance(row[0], datetime):
        raise RuntimeError("PostgreSQL did not return its current timestamp.")
    value = row[0]
    if value.tzinfo is None or value.utcoffset() is None:
        raise RuntimeError("PostgreSQL returned a timezone-naive timestamp.")
    return value.astimezone(UTC)


def _request_from_row(row: Any) -> EvalRunRequest:
    invocation = eval_run_invocation_from_json(row[7])
    if row[25] != invocation.authored_suite_launch_revision:
        raise RuntimeError(
            "Stored authored-suite launch identity conflicts with durable invocation."
        )
    if row[26] != invocation.authored_suite_launch_lane:
        raise RuntimeError("Stored authored-suite launch lane conflicts with durable invocation.")
    return EvalRunRequest(
        run_id=row[0],
        idempotency_key=row[1],
        corpus_revision=row[2],
        target_key=row[3],
        suite_id=row[4],
        suite_revision=row[5],
        max_concurrency=row[6],
        invocation=invocation,
    )


def _run_observation_from_row(row: Any) -> EvalRunObservation:
    status = EvalRunStatus(row[1])
    ownership = None
    if status in {EvalRunStatus.RUNNING, EvalRunStatus.CANCELLING}:
        ownership = EvalRunOwnership(epoch=row[3], lease_expires_at=row[4])
    return EvalRunObservation(
        run_id=row[0],
        status=status,
        attempt_count=row[3],
        updated_at=row[2],
        ownership=ownership,
    )


async def _load_trial_checkpoints(
    cur: Any,
    row: Any,
) -> tuple[EvalRunTrialCheckpoint, ...]:
    await cur.execute(
        """
        SELECT case_id, trial_number, checkpoint_json, document_bytes
        FROM cayu_eval_run_trial_checkpoints
        WHERE run_id = %s
        ORDER BY case_id ASC, trial_number ASC
        """,
        (row[0],),
    )
    checkpoint_rows = await cur.fetchall()
    checkpoints: list[EvalRunTrialCheckpoint] = []
    document_bytes = 0
    for checkpoint_row in checkpoint_rows:
        checkpoint = eval_run_trial_checkpoint_from_json(checkpoint_row[2])
        actual_bytes = len(checkpoint_row[2].encode("utf-8"))
        if (
            checkpoint.case_id != checkpoint_row[0]
            or checkpoint.trial_number != checkpoint_row[1]
            or checkpoint_row[3] != actual_bytes
        ):
            raise RuntimeError("Stored eval trial checkpoint contradicts its indexed slot.")
        checkpoints.append(checkpoint)
        document_bytes += actual_bytes
    if len(checkpoints) != row[23]:
        raise RuntimeError("Stored eval trial checkpoint count is inconsistent.")
    if document_bytes != row[24]:
        raise RuntimeError("Stored eval trial checkpoint byte total is inconsistent.")
    return _validated_trial_checkpoints(
        tuple(checkpoints),
        expected_document_bytes=document_bytes,
    )


async def _delete_trial_checkpoints(cur: Any, run_id: str) -> None:
    await cur.execute(
        "DELETE FROM cayu_eval_run_trial_checkpoints WHERE run_id = %s",
        (run_id,),
    )


def _run_record_from_row(row: Any) -> EvalRunRecord:
    status = EvalRunStatus(row[8])
    ownership = None
    if status in {EvalRunStatus.RUNNING, EvalRunStatus.CANCELLING}:
        ownership = EvalRunOwnership(
            epoch=row[15],
            lease_expires_at=row[16],
        )
    result = None
    if row[17] is not None:
        result = EvalRunResultSummary(
            revision=row[17],
            status=row[18],
            score=row[19],
            duration_ms=row[20],
        )
    return EvalRunRecord(
        spec=EvalRunSpec(
            run_id=row[0],
            corpus_revision=row[2],
            target_key=row[3],
            suite_id=row[4],
            suite_revision=row[5],
            max_concurrency=row[6],
            invocation=eval_run_invocation_from_json(row[7]),
        ),
        status=status,
        attempt_count=row[15],
        created_at=row[9],
        updated_at=row[10],
        started_at=row[11],
        finished_at=row[12],
        cancel_requested_at=row[13],
        ownership=ownership,
        result=result,
        failure_code=None if row[21] is None else EvalRunFailureCode(row[21]),
        scenario_progress=(
            None if row[22] is None else EvalScenarioRunProgress.model_validate_json(row[22])
        ),
    )


def _result_record_from_row(row: Any) -> EvalResultRecord:
    return EvalResultRecord(
        revision=row[0],
        origin=EvalResultOrigin(row[1]),
        target=EvalResultTargetIdentityV1(
            target_key=row[2],
            application_release_id=row[6],
            app_manifest_schema_version=row[7],
            app_manifest_fingerprint=row[8],
        ),
        corpus_revision=row[3],
        suite_id=row[4],
        suite_revision=row[5],
        status=row[9],
        score=row[10],
        document_bytes=row[11],
        created_at=row[12],
    )


def _baseline_key_from_row(row: Any) -> EvalBaselineKey:
    return EvalBaselineKey(
        target_key=row[0],
        corpus_revision=row[1],
        suite_id=row[2],
    )


def _baseline_record_from_row(row: Any) -> EvalBaselineRecord:
    return EvalBaselineRecord(
        key=_baseline_key_from_row(row),
        result_revision=row[3],
        generation=row[4],
        updated_by=row[5],
        updated_at=row[6],
    )


def _baseline_mutation_from_row(row: Any) -> EvalBaselineMutationRecord:
    return EvalBaselineMutationRecord(
        operation_id=row[0],
        key=EvalBaselineKey(
            target_key=row[1],
            corpus_revision=row[2],
            suite_id=row[3],
        ),
        expected_generation=row[4],
        previous_result_revision=row[5],
        selected_result_revision=row[6],
        resulting_generation=row[7],
        actor_id=row[8],
        created_at=row[9],
    )


class PostgresEvalStore(_PostgresStoreBase, EvalStore):
    """Restart-durable, multi-worker eval persistence for PostgreSQL."""

    durable: ClassVar[bool] = True
    captured_results: ClassVar[bool] = True
    scenarios: ClassVar[bool] = True
    scenario_execution: ClassVar[bool] = True
    trial_checkpointing: ClassVar[bool] = True
    suite_authoring: ClassVar[bool] = True
    judge_calibrations: ClassVar[bool] = True
    _min_required_revision = _POSTGRES_EVAL_MIN_REQUIRED_REVISION

    async def save_corpus(
        self,
        corpus: EvalCorpusDocument,
        *,
        redact_json: Callable[[Any], Any],
    ) -> EvalCorpusCatalogEntry:
        corpus, document, inspection, suites, cases = await asyncio.to_thread(
            _prepare_corpus_catalog_for_store,
            corpus,
            redact_json=redact_json,
        )
        document_text = document.decode("utf-8")
        document_bytes = len(document)
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    created_at = await _database_now(cur)
                    entry = await self._save_prepared_corpus_in_transaction(
                        cur,
                        corpus=corpus,
                        document_text=document_text,
                        document_bytes=document_bytes,
                        inspection=inspection,
                        suites=suites,
                        cases=cases,
                        created_at=created_at,
                    )
                await conn.commit()
                assert entry is not None
                return entry
            except BaseException:
                await conn.rollback()
                raise

    async def load_corpus(
        self,
        revision: str,
        *,
        max_bytes: int = EVAL_CORPUS_MAX_BYTES,
    ) -> EvalCorpusDocument | None:
        revision = _sha256_revision(revision, "revision")
        max_bytes = _read_limit(max_bytes, hard_max=EVAL_CORPUS_MAX_BYTES)
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT document_bytes FROM cayu_eval_corpora WHERE revision = %s",
                (revision,),
            )
            size = await cur.fetchone()
            if size is None:
                return None
            if size[0] > max_bytes:
                raise EvalStoreResultTooLarge(max_bytes)
            await cur.execute(
                "SELECT document FROM cayu_eval_corpora WHERE revision = %s",
                (revision,),
            )
            row = await cur.fetchone()
            if row is None:
                raise RuntimeError("Immutable eval corpus disappeared during a read.")
            document = row[0]
        return await asyncio.to_thread(eval_corpus_from_json, document)

    async def list_corpora(
        self,
        query: EvalCatalogQuery | None = None,
    ) -> EvalCorpusCatalogPage:
        query = _copy_query(query, EvalCatalogQuery)
        boundary = (
            decode_corpus_cursor(query.cursor, query.target_key)
            if query.cursor is not None
            else None
        )
        clauses: list[str] = []
        params: list[object] = []
        if query.target_key is not None:
            clauses.append("target_key = %s")
            params.append(query.target_key)
        if boundary is not None:
            clauses.append("(created_at < %s OR (created_at = %s AND revision > %s))")
            params.extend((boundary[0], boundary[0], boundary[1]))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                cast(
                    "LiteralString",
                    f"""
                    SELECT revision, target_key, evidence_policy_revision,
                           pricing_profile_fingerprint, suite_count, case_count,
                           assertion_count, expanded_assertion_result_count,
                           document_bytes, created_at
                    FROM cayu_eval_corpora
                    {where}
                    ORDER BY created_at DESC, revision ASC
                    LIMIT %s
                    """,
                ),
                (*params, query.limit + 1),
            )
            rows = await cur.fetchall()
        return _bounded_corpus_page([self._corpus_entry_from_row(row) for row in rows], query)

    async def save_scenario(
        self,
        scenario: EvalScenarioDocumentV2,
        *,
        redact_json: Callable[[Any], Any],
    ) -> EvalScenarioCatalogEntry:
        scenario, document, inspection = await asyncio.to_thread(
            _prepare_scenario_catalog_for_store,
            scenario,
            redact_json=redact_json,
        )
        document_text = document.decode("utf-8")
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    created_at = await _database_now(cur)
                    await cur.execute(
                        """
                        INSERT INTO cayu_eval_scenarios (
                            revision, scenario_id, target_key, name, description,
                            event_count, input_event_count, approval_checkpoint_count,
                            message_count, part_count, artifact_requirement_count,
                            secret_requirement_count, document_json, document_bytes,
                            created_at
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s
                        )
                        ON CONFLICT (revision) DO NOTHING
                        RETURNING revision
                        """,
                        (
                            scenario.revision,
                            scenario.id,
                            scenario.target_key,
                            scenario.name,
                            scenario.description,
                            inspection.event_count,
                            inspection.input_event_count,
                            inspection.approval_checkpoint_count,
                            inspection.message_count,
                            inspection.part_count,
                            inspection.artifact_requirement_count,
                            inspection.secret_requirement_count,
                            document_text,
                            len(document),
                            created_at,
                        ),
                    )
                    if await cur.fetchone() is None:
                        await cur.execute(
                            "SELECT document_json FROM cayu_eval_scenarios WHERE revision = %s",
                            (scenario.revision,),
                        )
                        existing = await cur.fetchone()
                        if existing is None or existing[0] != document_text:
                            raise EvalScenarioConflict(
                                f"Eval scenario revision {scenario.revision} has "
                                "conflicting content."
                            )
                    entry = await self._load_scenario_entry(cur, scenario.revision)
                    assert entry is not None
                await conn.commit()
                return entry
            except BaseException:
                await conn.rollback()
                raise

    async def load_scenario(
        self,
        revision: str,
        *,
        max_bytes: int = EVAL_SCENARIO_MAX_BYTES,
    ) -> EvalScenarioDocumentV2 | None:
        revision = _sha256_revision(revision, "revision")
        max_bytes = _read_limit(max_bytes, hard_max=EVAL_SCENARIO_MAX_BYTES)
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT document_json, document_bytes
                FROM cayu_eval_scenarios
                WHERE revision = %s
                """,
                (revision,),
            )
            row = await cur.fetchone()
            if row is None:
                return None
            if row[1] > max_bytes:
                raise EvalStoreResultTooLarge(max_bytes)
            document = row[0]
        return await asyncio.to_thread(eval_scenario_from_json, document)

    async def list_scenarios(
        self,
        query: EvalScenarioCatalogQuery | None = None,
    ) -> EvalScenarioCatalogPage:
        query = _copy_query(query, EvalScenarioCatalogQuery)
        boundary = (
            decode_scenario_cursor(query.cursor, query.target_key, query.scenario_id)
            if query.cursor is not None
            else None
        )
        clauses: list[str] = []
        params: list[object] = []
        if query.target_key is not None:
            clauses.append("target_key = %s")
            params.append(query.target_key)
        if query.scenario_id is not None:
            clauses.append("scenario_id = %s")
            params.append(query.scenario_id)
        if boundary is not None:
            clauses.append("(created_at < %s OR (created_at = %s AND revision > %s))")
            params.extend((boundary[0], boundary[0], boundary[1]))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                cast(
                    "LiteralString",
                    f"""
                    SELECT revision, scenario_id, target_key, name, description,
                           event_count, input_event_count, approval_checkpoint_count,
                           message_count, part_count, artifact_requirement_count,
                           secret_requirement_count, document_bytes, created_at
                    FROM cayu_eval_scenarios
                    {where}
                    ORDER BY created_at DESC, revision ASC
                    LIMIT %s
                    """,
                ),
                (*params, query.limit + 1),
            )
            rows = await cur.fetchall()
        return _bounded_scenario_page(
            [self._scenario_entry_from_row(row) for row in rows],
            query,
        )

    async def save_authored_suite(
        self,
        document: EvalSuiteDocument,
        *,
        redact_json: Callable[[Any], Any],
    ) -> EvalAuthoredSuiteCatalogEntry:
        validated, payload = await asyncio.to_thread(
            _prepare_authored_suite_for_store,
            document,
            redact_json=redact_json,
        )
        document_text = payload.decode("utf-8")
        scenario_cases = authored_suite_scenario_cases(validated)
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    scenario_by_revision: dict[str, EvalScenarioDocumentV2] = {}
                    if scenario_cases:
                        await cur.execute(
                            "SELECT revision, document_json FROM cayu_eval_scenarios "
                            "WHERE revision = ANY(%s)",
                            (
                                list(
                                    {reference.scenario_revision for _, reference in scenario_cases}
                                ),
                            ),
                        )
                        scenario_rows = await cur.fetchall()
                        scenario_by_revision = await asyncio.to_thread(
                            lambda: {
                                str(revision): eval_scenario_from_json(scenario_json)
                                for revision, scenario_json in scenario_rows
                            }
                        )
                    for case, reference in scenario_cases:
                        validate_authored_suite_scenario(
                            validated,
                            case,
                            scenario_by_revision.get(reference.scenario_revision),
                        )
                    created_at = await _database_now(cur)
                    entry = authored_suite_catalog_entry(
                        validated,
                        created_at=created_at,
                        document_bytes=len(payload),
                    )
                    await cur.execute(
                        """
                        INSERT INTO cayu_eval_authored_suites (
                            revision, suite_id, suite_revision, target_key, name,
                            description, case_count, assertion_count,
                            simple_input_count, scenario_count, trials,
                            timeout_seconds, document_json, document_bytes, created_at
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s
                        )
                        ON CONFLICT (revision) DO NOTHING
                        RETURNING revision
                        """,
                        (
                            entry.revision,
                            entry.id,
                            entry.suite_revision,
                            entry.target_key,
                            entry.name,
                            entry.description,
                            entry.case_count,
                            entry.assertion_count,
                            entry.simple_input_count,
                            entry.scenario_count,
                            entry.trials,
                            entry.timeout_seconds,
                            document_text,
                            entry.document_bytes,
                            entry.created_at,
                        ),
                    )
                    if await cur.fetchone() is None:
                        await cur.execute(
                            "SELECT document_json FROM cayu_eval_authored_suites "
                            "WHERE revision = %s",
                            (validated.revision,),
                        )
                        existing = await cur.fetchone()
                        if existing is None or existing[0] != document_text:
                            raise EvalAuthoredSuiteConflict(
                                f"Authored eval suite revision {validated.revision} has "
                                "conflicting content."
                            )
                        loaded = await self._load_authored_suite_entry(
                            cur,
                            validated.revision,
                        )
                        assert loaded is not None
                        entry = loaded
                await conn.commit()
                return entry
            except BaseException:
                await conn.rollback()
                raise

    async def load_authored_suite(
        self,
        revision: str,
        *,
        max_bytes: int = EVAL_SUITE_AUTHORING_MAX_BYTES,
    ) -> EvalSuiteDocument | None:
        revision = _sha256_revision(revision, "revision")
        max_bytes = _read_limit(max_bytes, hard_max=EVAL_SUITE_AUTHORING_MAX_BYTES)
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT document_json, document_bytes
                FROM cayu_eval_authored_suites
                WHERE revision = %s
                """,
                (revision,),
            )
            row = await cur.fetchone()
            if row is None:
                return None
            if row[1] > max_bytes:
                raise EvalStoreResultTooLarge(max_bytes)
            payload = row[0]
        return await asyncio.to_thread(eval_suite_document_from_json, payload)

    async def list_authored_suites(
        self,
        query: EvalAuthoredSuiteCatalogQuery | None = None,
    ) -> EvalAuthoredSuiteCatalogPage:
        query = _copy_query(query, EvalAuthoredSuiteCatalogQuery)
        boundary = (
            decode_authored_suite_cursor(
                query.cursor,
                query.target_key,
                query.suite_id,
            )
            if query.cursor is not None
            else None
        )
        clauses: list[str] = []
        params: list[object] = []
        if query.target_key is not None:
            clauses.append("target_key = %s")
            params.append(query.target_key)
        if query.suite_id is not None:
            clauses.append("suite_id = %s")
            params.append(query.suite_id)
        if boundary is not None:
            clauses.append("(created_at < %s OR (created_at = %s AND revision > %s))")
            params.extend((boundary[0], boundary[0], boundary[1]))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                cast(
                    "LiteralString",
                    f"""
                    SELECT revision, suite_id, suite_revision, target_key, name,
                           description, case_count, assertion_count,
                           simple_input_count, scenario_count, trials,
                           timeout_seconds, document_bytes, created_at
                    FROM cayu_eval_authored_suites
                    {where}
                    ORDER BY created_at DESC, revision ASC
                    LIMIT %s
                    """,
                ),
                (*params, query.limit + 1),
            )
            rows = await cur.fetchall()
        return _bounded_authored_suite_page(
            [self._authored_suite_entry_from_row(row) for row in rows],
            query,
        )

    async def save_judge_calibration(
        self,
        report: EvalJudgeCalibrationReportV1,
        *,
        redact_json: Callable[[Any], Any],
    ) -> EvalJudgeCalibrationReportV1:
        validated, payload = await asyncio.to_thread(
            _prepare_judge_calibration_for_store,
            report,
            redact_json=redact_json,
        )
        document_text = payload.decode("utf-8")
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    for lock_key in sorted((validated.revision, validated.run_id)):
                        await cur.execute(
                            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                            (lock_key,),
                        )
                    await cur.execute(
                        """
                        SELECT revision, run_id, report_json
                        FROM cayu_eval_judge_calibrations
                        WHERE revision = %s OR run_id = %s
                        """,
                        (validated.revision, validated.run_id),
                    )
                    rows = await cur.fetchall()
                    if rows:
                        if len(rows) != 1 or tuple(rows[0]) != (
                            validated.revision,
                            validated.run_id,
                            document_text,
                        ):
                            raise EvalJudgeCalibrationConflict(
                                "Judge calibration revision or run ID has conflicting content."
                            )
                        stored = rows[0][2]
                    else:
                        created_at = await _database_now(cur)
                        await cur.execute(
                            """
                            INSERT INTO cayu_eval_judge_calibrations (
                                revision, run_id, definition_revision, target_key,
                                trial_count, report_json, document_bytes, created_at
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                validated.revision,
                                validated.run_id,
                                validated.definition.revision,
                                validated.definition.target_key,
                                len(validated.trials),
                                document_text,
                                len(payload),
                                created_at,
                            ),
                        )
                        stored = document_text
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise
        return await asyncio.to_thread(eval_judge_calibration_report_from_json, stored)

    async def load_judge_calibration(
        self,
        revision: str,
        *,
        max_bytes: int = EVAL_JUDGE_CALIBRATION_MAX_BYTES,
    ) -> EvalJudgeCalibrationReportV1 | None:
        revision = _sha256_revision(revision, "revision")
        max_bytes = _read_limit(max_bytes, hard_max=EVAL_JUDGE_CALIBRATION_MAX_BYTES)
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT report_json, document_bytes
                FROM cayu_eval_judge_calibrations
                WHERE revision = %s
                """,
                (revision,),
            )
            row = await cur.fetchone()
            if row is None:
                return None
            if row[1] > max_bytes:
                raise EvalStoreResultTooLarge(max_bytes)
            stored = row[0]
        return await asyncio.to_thread(eval_judge_calibration_report_from_json, stored)

    async def load_judge_calibration_by_run_id(
        self,
        run_id: str,
        *,
        max_bytes: int = EVAL_JUDGE_CALIBRATION_MAX_BYTES,
    ) -> EvalJudgeCalibrationReportV1 | None:
        run_id = _portable_id(run_id, "run_id")
        max_bytes = _read_limit(max_bytes, hard_max=EVAL_JUDGE_CALIBRATION_MAX_BYTES)
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT report_json, document_bytes
                FROM cayu_eval_judge_calibrations
                WHERE run_id = %s
                """,
                (run_id,),
            )
            row = await cur.fetchone()
            if row is None:
                return None
            if row[1] > max_bytes:
                raise EvalStoreResultTooLarge(max_bytes)
            stored = row[0]
        return await asyncio.to_thread(eval_judge_calibration_report_from_json, stored)

    async def list_suites(self, query: EvalSuiteCatalogQuery) -> EvalSuiteCatalogPage:
        query = _exact_model(query, EvalSuiteCatalogQuery, "query")
        boundary = (
            decode_suite_cursor(query.cursor, query.corpus_revision)
            if query.cursor is not None
            else None
        )
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            if not await self._corpus_exists(cur, query.corpus_revision):
                raise KeyError(f"Eval corpus not found: {query.corpus_revision}")
            await cur.execute(
                """
                SELECT corpus_revision, suite_id, suite_revision, name, description,
                       case_count, assertion_count, trials, timeout_seconds
                FROM cayu_eval_suites
                WHERE corpus_revision = %s AND suite_id > %s
                ORDER BY suite_id ASC
                LIMIT %s
                """,
                (query.corpus_revision, boundary or "", query.limit + 1),
            )
            rows = await cur.fetchall()
        return _bounded_suite_page([self._suite_entry_from_row(row) for row in rows], query)

    async def list_cases(self, query: EvalCaseCatalogQuery) -> EvalCaseCatalogPage:
        query = _copy_query(query, EvalCaseCatalogQuery)
        boundary = (
            decode_case_cursor(query.cursor, query.corpus_revision, query.suite_id)
            if query.cursor is not None
            else None
        )
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT 1 FROM cayu_eval_suites
                WHERE corpus_revision = %s AND suite_id = %s
                """,
                (query.corpus_revision, query.suite_id),
            )
            if await cur.fetchone() is None:
                if not await self._corpus_exists(cur, query.corpus_revision):
                    raise KeyError(f"Eval corpus not found: {query.corpus_revision}")
                raise KeyError(f"Eval suite not found: {query.suite_id}")
            await cur.execute(
                """
                SELECT corpus_revision, case_id, case_revision, suite_id, name,
                       description, message_count, assertion_count
                FROM cayu_eval_cases
                WHERE corpus_revision = %s AND suite_id = %s AND case_id > %s
                ORDER BY case_id ASC
                LIMIT %s
                """,
                (query.corpus_revision, query.suite_id, boundary or "", query.limit + 1),
            )
            rows = await cur.fetchall()
        return _bounded_case_page([self._case_entry_from_row(row) for row in rows], query)

    async def admit_run(
        self,
        request: EvalRunRequest,
        *,
        redact_json: Callable[[Any], Any],
    ) -> EvalRunRecord:
        request = _prepare_run_request_for_store(
            request,
            redact_json=redact_json,
        )
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    duplicate = await self._load_run_by_idempotency_key(
                        cur, request.idempotency_key
                    )
                    if duplicate is not None:
                        record = _run_record_from_row(duplicate)
                        if not _request_from_row(duplicate).same_logical_request(request):
                            raise EvalRunAdmissionConflict(
                                "Eval run idempotency key is already bound to another request."
                            )
                        await conn.commit()
                        return record
                    await cur.execute(
                        """
                        SELECT corpus.target_key, suite.suite_revision
                        FROM cayu_eval_corpora AS corpus
                        JOIN cayu_eval_suites AS suite
                          ON suite.corpus_revision = corpus.revision
                        WHERE corpus.revision = %s AND suite.suite_id = %s
                        """,
                        (request.corpus_revision, request.suite_id),
                    )
                    contract = await cur.fetchone()
                    if contract is None:
                        if not await self._corpus_exists(cur, request.corpus_revision):
                            raise KeyError(f"Eval corpus not found: {request.corpus_revision}")
                        raise EvalRunAdmissionConflict(f"Eval suite not found: {request.suite_id}")
                    if contract[0] != request.target_key:
                        raise EvalRunAdmissionConflict(
                            "Eval run target key does not match its corpus."
                        )
                    if contract[1] != request.suite_revision:
                        raise EvalRunAdmissionConflict(
                            "Eval run suite revision does not match its corpus."
                        )
                    now = await _database_now(cur)
                    await cur.execute(
                        """
                        INSERT INTO cayu_eval_runs (
                            run_id, idempotency_key, corpus_revision, target_key,
                            suite_id, suite_revision, max_concurrency, invocation_json, status,
                            created_at, updated_at, authored_suite_launch_revision,
                            authored_suite_launch_lane
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                        RETURNING run_id
                        """,
                        (
                            request.run_id,
                            request.idempotency_key,
                            request.corpus_revision,
                            request.target_key,
                            request.suite_id,
                            request.suite_revision,
                            request.max_concurrency,
                            request.invocation.model_dump_json(),
                            str(EvalRunStatus.QUEUED),
                            now,
                            now,
                            request.invocation.authored_suite_launch_revision,
                            request.invocation.authored_suite_launch_lane,
                        ),
                    )
                    inserted = await cur.fetchone()
                    if inserted is None:
                        duplicate = await self._load_run_by_idempotency_key(
                            cur, request.idempotency_key
                        )
                        if duplicate is not None:
                            record = _run_record_from_row(duplicate)
                            if _request_from_row(duplicate).same_logical_request(request):
                                await conn.commit()
                                return record
                            raise EvalRunAdmissionConflict(
                                "Eval run idempotency key is already bound to another request."
                            )
                        raise EvalRunAdmissionConflict(
                            f"Eval run id is already bound to another request: {request.run_id}"
                        )
                    row = await self._load_run_row(cur, request.run_id)
                await conn.commit()
                assert row is not None
                return _run_record_from_row(row)
            except BaseException:
                await conn.rollback()
                raise

    async def load_run_by_idempotency_key(
        self,
        idempotency_key: str,
    ) -> EvalRunRecord | None:
        idempotency_key = _idempotency_key(idempotency_key, "idempotency_key")
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            row = await self._load_run_by_idempotency_key(cur, idempotency_key)
            return None if row is None else _run_record_from_row(row)

    async def load_run(self, run_id: str) -> EvalRunRecord | None:
        run_id = _store_identifier(run_id, "run_id")
        started_at = monotonic()
        await self._ensure_ready()
        try:
            async with self._connection() as conn, conn.cursor() as cur:
                row = await self._load_run_row(cur, run_id)
                return None if row is None else _run_record_from_row(row)
        finally:
            logger.debug(
                "PostgreSQL eval run fully rehydrated.",
                extra={
                    "cayu_eval_store_event": "full_run_rehydration",
                    "eval_store_kind": "postgres",
                    "eval_run_id": run_id,
                    "duration_seconds": monotonic() - started_at,
                },
            )

    async def load_run_observation(self, run_id: str) -> EvalRunObservation | None:
        run_id = _store_identifier(run_id, "run_id")
        started_at = monotonic()
        await self._ensure_ready()
        try:
            async with self._connection() as conn, conn.cursor() as cur:
                await cur.execute(
                    f"SELECT {_RUN_OBSERVATION_COLUMNS} FROM cayu_eval_runs WHERE run_id = %s",
                    (run_id,),
                )
                row = await cur.fetchone()
                return None if row is None else _run_observation_from_row(row)
        finally:
            logger.debug(
                "PostgreSQL eval run status observed.",
                extra={
                    "cayu_eval_store_event": "run_status_read",
                    "eval_store_kind": "postgres",
                    "eval_run_id": run_id,
                    "duration_seconds": monotonic() - started_at,
                },
            )

    async def list_runs(self, query: EvalRunQuery | None = None) -> EvalRunPage:
        query = _copy_query(query, EvalRunQuery)
        boundary = (
            decode_run_cursor(
                query.cursor,
                query.target_key,
                query.status,
                query.corpus_revision,
            )
            if query.cursor is not None
            else None
        )
        clauses: list[str] = []
        params: list[object] = []
        if query.target_key is not None:
            clauses.append("target_key = %s")
            params.append(query.target_key)
        if query.status is not None:
            clauses.append("status = %s")
            params.append(str(query.status))
        if query.corpus_revision is not None:
            clauses.append("corpus_revision = %s")
            params.append(query.corpus_revision)
        if boundary is not None:
            clauses.append("(created_at < %s OR (created_at = %s AND run_id > %s))")
            params.extend((boundary[0], boundary[0], boundary[1]))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                cast(
                    "LiteralString",
                    f"""
                    SELECT {_RUN_COLUMNS}
                    FROM cayu_eval_runs
                    {where}
                    ORDER BY created_at DESC, run_id ASC
                    LIMIT %s
                    """,
                ),
                (*params, query.limit + 1),
            )
            rows = await cur.fetchall()
        return _bounded_run_page([_run_record_from_row(row) for row in rows], query)

    async def claim_run(
        self,
        *,
        target_key: str | None = None,
        lease_seconds: int = 300,
    ) -> EvalRunLease | None:
        if target_key is not None:
            target_key = _portable_id(target_key, "target_key")
        lease_seconds = _lease_seconds(lease_seconds)
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    now = await _database_now(cur)
                    target_clause = "" if target_key is None else "AND candidate.target_key = %s"
                    target_params: tuple[str, ...] = () if target_key is None else (target_key,)
                    await cur.execute(
                        f"""
                        SELECT {_RUN_COLUMNS}
                        FROM cayu_eval_runs AS candidate
                        WHERE candidate.ownership_epoch < 9223372036854775807
                          {target_clause}
                          AND (candidate.status = %s
                           OR (candidate.status IN (%s, %s)
                               AND candidate.lease_expires_at <= %s))
                          AND (
                              candidate.authored_suite_launch_revision IS NULL
                              OR NOT EXISTS (
                                  SELECT 1
                                  FROM cayu_eval_runs AS predecessor
                                  WHERE predecessor.authored_suite_launch_revision =
                                        candidate.authored_suite_launch_revision
                                    AND predecessor.authored_suite_launch_lane =
                                        candidate.authored_suite_launch_lane
                                    AND predecessor.status NOT IN (%s, %s, %s)
                                    AND (
                                        predecessor.created_at < candidate.created_at
                                        OR (
                                            predecessor.created_at = candidate.created_at
                                            AND predecessor.run_id < candidate.run_id
                                        )
                                    )
                              )
                          )
                        ORDER BY candidate.created_at ASC, candidate.run_id ASC
                        FOR UPDATE SKIP LOCKED
                        LIMIT 1
                        """,
                        (
                            *target_params,
                            str(EvalRunStatus.QUEUED),
                            str(EvalRunStatus.RUNNING),
                            str(EvalRunStatus.CANCELLING),
                            now,
                            str(EvalRunStatus.COMPLETED),
                            str(EvalRunStatus.FAILED),
                            str(EvalRunStatus.CANCELLED),
                        ),
                    )
                    row = await cur.fetchone()
                    if row is None:
                        await conn.commit()
                        return None
                    status = (
                        EvalRunStatus.CANCELLING if row[13] is not None else EvalRunStatus.RUNNING
                    )
                    next_epoch = row[15] + 1
                    checkpoints = await _load_trial_checkpoints(cur, row)
                    progress = _scenario_progress_for_claim(
                        (
                            None
                            if row[22] is None
                            else EvalScenarioRunProgress.model_validate_json(row[22])
                        ),
                        scenario=_request_from_row(row).invocation.scenario,
                        attempt=next_epoch,
                        terminal_trial_numbers=frozenset(item.trial_number for item in checkpoints),
                    )
                    await cur.execute(
                        """
                        UPDATE cayu_eval_runs
                        SET status = %s, updated_at = %s,
                            started_at = COALESCE(started_at, %s),
                            claim_id = %s,
                            ownership_epoch = ownership_epoch + 1,
                            lease_expires_at = %s,
                            scenario_progress_json = %s
                        WHERE run_id = %s
                        """,
                        (
                            str(status),
                            now,
                            now,
                            str(uuid4()),
                            now + timedelta(seconds=lease_seconds),
                            None if progress is None else progress.model_dump_json(),
                            row[0],
                        ),
                    )
                    claimed = await self._load_run_row(cur, row[0])
                await conn.commit()
                assert claimed is not None
                record = _run_record_from_row(claimed)
                return EvalRunLease(
                    run=record,
                    claim=EvalRunClaim(
                        run_id=record.id,
                        claim_id=claimed[14],
                        epoch=claimed[15],
                    ),
                )
            except BaseException:
                await conn.rollback()
                raise

    async def claim_run_for_targets(
        self,
        target_keys: tuple[str, ...],
        *,
        lease_seconds: int = 300,
    ) -> EvalRunLease | None:
        target_keys = _claim_target_keys(target_keys)
        lease_seconds = _lease_seconds(lease_seconds)
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    now = await _database_now(cur)
                    placeholders = ", ".join("%s" for _ in target_keys)
                    await cur.execute(
                        f"""
                        SELECT {_RUN_COLUMNS}
                        FROM cayu_eval_runs AS candidate
                        WHERE candidate.ownership_epoch < 9223372036854775807
                          AND candidate.target_key IN ({placeholders})
                          AND (candidate.status = %s
                           OR (candidate.status IN (%s, %s)
                               AND candidate.lease_expires_at <= %s))
                          AND (
                              candidate.authored_suite_launch_revision IS NULL
                              OR NOT EXISTS (
                                  SELECT 1
                                  FROM cayu_eval_runs AS predecessor
                                  WHERE predecessor.authored_suite_launch_revision =
                                        candidate.authored_suite_launch_revision
                                    AND predecessor.authored_suite_launch_lane =
                                        candidate.authored_suite_launch_lane
                                    AND predecessor.status NOT IN (%s, %s, %s)
                                    AND (
                                        predecessor.created_at < candidate.created_at
                                        OR (
                                            predecessor.created_at = candidate.created_at
                                            AND predecessor.run_id < candidate.run_id
                                        )
                                    )
                              )
                          )
                        ORDER BY candidate.created_at ASC, candidate.run_id ASC
                        FOR UPDATE SKIP LOCKED
                        LIMIT 1
                        """,
                        (
                            *target_keys,
                            str(EvalRunStatus.QUEUED),
                            str(EvalRunStatus.RUNNING),
                            str(EvalRunStatus.CANCELLING),
                            now,
                            str(EvalRunStatus.COMPLETED),
                            str(EvalRunStatus.FAILED),
                            str(EvalRunStatus.CANCELLED),
                        ),
                    )
                    row = await cur.fetchone()
                    if row is None:
                        await conn.commit()
                        return None
                    status = (
                        EvalRunStatus.CANCELLING if row[13] is not None else EvalRunStatus.RUNNING
                    )
                    claim_id = str(uuid4())
                    next_epoch = row[15] + 1
                    checkpoints = await _load_trial_checkpoints(cur, row)
                    progress = _scenario_progress_for_claim(
                        (
                            None
                            if row[22] is None
                            else EvalScenarioRunProgress.model_validate_json(row[22])
                        ),
                        scenario=_request_from_row(row).invocation.scenario,
                        attempt=next_epoch,
                        terminal_trial_numbers=frozenset(item.trial_number for item in checkpoints),
                    )
                    await cur.execute(
                        """
                        UPDATE cayu_eval_runs
                        SET status = %s, updated_at = %s,
                            started_at = COALESCE(started_at, %s),
                            claim_id = %s,
                            ownership_epoch = ownership_epoch + 1,
                            lease_expires_at = %s,
                            scenario_progress_json = %s
                        WHERE run_id = %s
                        """,
                        (
                            str(status),
                            now,
                            now,
                            claim_id,
                            now + timedelta(seconds=lease_seconds),
                            None if progress is None else progress.model_dump_json(),
                            row[0],
                        ),
                    )
                    claimed = await self._load_run_row(cur, row[0])
                await conn.commit()
                assert claimed is not None
                record = _run_record_from_row(claimed)
                return EvalRunLease(
                    run=record,
                    claim=EvalRunClaim(
                        run_id=record.id,
                        claim_id=claimed[14],
                        epoch=claimed[15],
                    ),
                )
            except BaseException:
                await conn.rollback()
                raise

    async def heartbeat_run(
        self,
        claim: EvalRunClaim,
        *,
        extend_seconds: int = 300,
    ) -> EvalRunRecord:
        claim = _exact_model(claim, EvalRunClaim, "claim")
        extend_seconds = _lease_seconds(extend_seconds)
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    row = await self._require_run_row(cur, claim.run_id, for_update=True)
                    # Capture time only after the row lock is held. A blocked
                    # heartbeat must not renew a lease that expired while waiting.
                    now = await _database_now(cur)
                    self._require_live_claim(row, claim, now)
                    await cur.execute(
                        """
                        UPDATE cayu_eval_runs
                        SET lease_expires_at = %s, updated_at = %s
                        WHERE run_id = %s
                        """,
                        (
                            now + timedelta(seconds=extend_seconds),
                            now,
                            claim.run_id,
                        ),
                    )
                    updated = await self._require_run_row(cur, claim.run_id)
                await conn.commit()
                return _run_record_from_row(updated)
            except BaseException:
                await conn.rollback()
                raise

    async def heartbeat_run_observation(
        self,
        claim: EvalRunClaim,
        *,
        extend_seconds: int = 300,
    ) -> EvalRunObservation:
        claim = _exact_model(claim, EvalRunClaim, "claim")
        extend_seconds = _lease_seconds(extend_seconds)
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        SELECT run_id, status, claim_id, ownership_epoch, lease_expires_at
                        FROM cayu_eval_runs
                        WHERE run_id = %s
                        FOR UPDATE
                        """,
                        (claim.run_id,),
                    )
                    row = await cur.fetchone()
                    if row is None:
                        raise EvalRunClaimLost("Eval run claim is no longer live.")
                    now = await _database_now(cur)
                    if row[2] != claim.claim_id or row[3] != claim.epoch:
                        raise EvalRunClaimLost("Eval run claim is no longer owned by this worker.")
                    if EvalRunStatus(row[1]) not in {
                        EvalRunStatus.RUNNING,
                        EvalRunStatus.CANCELLING,
                    }:
                        raise EvalRunClaimLost("Eval run is no longer active.")
                    if row[4] is None or row[4] <= now:
                        raise EvalRunClaimLost("Eval run claim lease has expired.")
                    await cur.execute(
                        """
                        UPDATE cayu_eval_runs
                        SET lease_expires_at = %s, updated_at = %s
                        WHERE run_id = %s
                        """,
                        (
                            now + timedelta(seconds=extend_seconds),
                            now,
                            claim.run_id,
                        ),
                    )
                    await cur.execute(
                        f"SELECT {_RUN_OBSERVATION_COLUMNS} FROM cayu_eval_runs WHERE run_id = %s",
                        (claim.run_id,),
                    )
                    updated = await cur.fetchone()
                await conn.commit()
                assert updated is not None
                return _run_observation_from_row(updated)
            except BaseException:
                await conn.rollback()
                raise

    async def request_cancel(self, run_id: str) -> EvalRunRecord:
        run_id = _store_identifier(run_id, "run_id")
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    row = await self._require_run_row(cur, run_id, for_update=True)
                    status = EvalRunStatus(row[8])
                    if status in TERMINAL_EVAL_RUN_STATUSES:
                        await conn.commit()
                        return _run_record_from_row(row)
                    now = await _database_now(cur)
                    claim_expired = (
                        status in {EvalRunStatus.RUNNING, EvalRunStatus.CANCELLING}
                        and row[16] is not None
                        and row[16] <= now
                    )
                    next_status = (
                        EvalRunStatus.CANCELLED
                        if status is EvalRunStatus.QUEUED or claim_expired
                        else EvalRunStatus.CANCELLING
                    )
                    await cur.execute(
                        """
                        UPDATE cayu_eval_runs
                        SET status = %s, updated_at = %s,
                            cancel_requested_at = COALESCE(cancel_requested_at, %s),
                            finished_at = CASE WHEN %s = %s THEN %s ELSE finished_at END,
                            claim_id = CASE WHEN %s = %s THEN NULL ELSE claim_id END,
                            lease_expires_at = CASE
                                WHEN %s = %s THEN NULL ELSE lease_expires_at END,
                            trial_checkpoint_count = CASE WHEN %s = %s THEN 0
                                ELSE trial_checkpoint_count END,
                            trial_checkpoint_bytes = CASE WHEN %s = %s THEN 0
                                ELSE trial_checkpoint_bytes END
                        WHERE run_id = %s
                        """,
                        (
                            str(next_status),
                            now,
                            now,
                            str(next_status),
                            str(EvalRunStatus.CANCELLED),
                            now,
                            str(next_status),
                            str(EvalRunStatus.CANCELLED),
                            str(next_status),
                            str(EvalRunStatus.CANCELLED),
                            str(next_status),
                            str(EvalRunStatus.CANCELLED),
                            str(next_status),
                            str(EvalRunStatus.CANCELLED),
                            run_id,
                        ),
                    )
                    if next_status is EvalRunStatus.CANCELLED:
                        await _delete_trial_checkpoints(cur, run_id)
                    updated = await self._require_run_row(cur, run_id)
                await conn.commit()
                return _run_record_from_row(updated)
            except BaseException:
                await conn.rollback()
                raise

    async def initialize_scenario_progress(
        self,
        claim: EvalRunClaim,
        progress: EvalScenarioRunProgress,
    ) -> EvalRunRecord:
        claim = _exact_model(claim, EvalRunClaim, "claim")
        progress = _exact_model(progress, EvalScenarioRunProgress, "progress")
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    row = await self._require_run_row(cur, claim.run_id, for_update=True)
                    now = await _database_now(cur)
                    self._require_live_claim(row, claim, now)
                    scenario = _request_from_row(row).invocation.scenario
                    if scenario is None:
                        raise EvalRunStateConflict(
                            "Only scenario runs may initialize scenario progress."
                        )
                    if (
                        progress.attempt != claim.epoch
                        or progress.scenario_revision != scenario.scenario_revision
                        or progress.binding_revision != scenario.binding_revision
                        or len(progress.trials) != scenario.trials
                    ):
                        raise EvalRunStateConflict(
                            "Scenario progress does not match the claimed run."
                        )
                    await cur.execute(
                        """
                        UPDATE cayu_eval_runs
                        SET scenario_progress_json = %s, updated_at = %s
                        WHERE run_id = %s
                        """,
                        (progress.model_dump_json(), now, claim.run_id),
                    )
                    updated = await self._require_run_row(cur, claim.run_id)
                await conn.commit()
                return _run_record_from_row(updated)
            except BaseException:
                await conn.rollback()
                raise

    async def update_scenario_trial(
        self,
        claim: EvalRunClaim,
        trial: EvalScenarioTrialProgress,
    ) -> EvalRunRecord:
        claim = _exact_model(claim, EvalRunClaim, "claim")
        trial = _exact_model(trial, EvalScenarioTrialProgress, "trial")
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    row = await self._require_run_row(cur, claim.run_id, for_update=True)
                    now = await _database_now(cur)
                    self._require_live_claim(row, claim, now)
                    if row[22] is None:
                        raise EvalRunStateConflict("Scenario progress is absent for this claim.")
                    progress = EvalScenarioRunProgress.model_validate_json(row[22])
                    if progress.attempt != claim.epoch:
                        raise EvalRunStateConflict("Scenario progress belongs to another claim.")
                    updated_progress = progress.replace_trial(trial)
                    await cur.execute(
                        """
                        UPDATE cayu_eval_runs
                        SET scenario_progress_json = %s, updated_at = %s
                        WHERE run_id = %s
                        """,
                        (updated_progress.model_dump_json(), now, claim.run_id),
                    )
                    updated = await self._require_run_row(cur, claim.run_id)
                await conn.commit()
                return _run_record_from_row(updated)
            except BaseException:
                await conn.rollback()
                raise

    async def load_trial_checkpoints(
        self,
        claim: EvalRunClaim,
    ) -> tuple[EvalRunTrialCheckpoint, ...]:
        claim = _exact_model(claim, EvalRunClaim, "claim")
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            row = await self._require_run_row(cur, claim.run_id, for_update=True)
            now = await _database_now(cur)
            self._require_live_claim(row, claim, now)
            return await _load_trial_checkpoints(cur, row)

    async def save_trial_checkpoint(
        self,
        claim: EvalRunClaim,
        checkpoint: EvalRunTrialCheckpoint,
        *,
        redact_json: Callable[[Any], Any],
    ) -> None:
        claim = _exact_model(claim, EvalRunClaim, "claim")
        prepared = await asyncio.to_thread(
            _prepare_trial_checkpoint_for_store,
            checkpoint,
            redact_json=redact_json,
        )
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    row = await self._require_run_row(cur, claim.run_id, for_update=True)
                    now = await _database_now(cur)
                    self._require_live_claim(row, claim, now)
                    request = _request_from_row(row)
                    await cur.execute(
                        """
                        SELECT suites.trials
                        FROM cayu_eval_suites AS suites
                        JOIN cayu_eval_cases AS cases
                          ON cases.corpus_revision = suites.corpus_revision
                         AND cases.suite_id = suites.suite_id
                        WHERE suites.corpus_revision = %s AND suites.suite_id = %s
                          AND cases.case_id = %s
                        """,
                        (
                            request.corpus_revision,
                            request.suite_id,
                            prepared.checkpoint.case_id,
                        ),
                    )
                    suite_row = await cur.fetchone()
                    if suite_row is None:
                        raise EvalRunStateConflict(
                            "Eval trial checkpoint case does not belong to its run."
                        )
                    validated = prepared.checkpoint
                    if validated.trial_number > suite_row[0]:
                        raise EvalRunStateConflict(
                            "Eval trial checkpoint number exceeds its immutable policy."
                        )
                    key = (validated.case_id, validated.trial_number)
                    await cur.execute(
                        """
                        SELECT checkpoint_json, document_bytes
                        FROM cayu_eval_run_trial_checkpoints
                        WHERE run_id = %s AND case_id = %s AND trial_number = %s
                        """,
                        (claim.run_id, *key),
                    )
                    existing = await cur.fetchone()
                    if existing is not None:
                        if existing[1] != len(existing[0].encode("utf-8")):
                            raise RuntimeError(
                                "Stored eval trial checkpoint byte accounting is inconsistent."
                            )
                        current = eval_run_trial_checkpoint_from_json(existing[0])
                        if current == validated:
                            await conn.commit()
                            return
                        raise EvalRunStateConflict(
                            "Eval trial slot already has another terminal result."
                        )
                    if row[23] >= EVAL_RUN_TRIAL_CHECKPOINTS_MAX_ITEMS:
                        raise ValueError("Eval run trial checkpoints exceed their item limit.")
                    if row[24] + prepared.document_bytes > EVAL_RUN_TRIAL_CHECKPOINTS_MAX_BYTES:
                        raise ValueError("Eval run trial checkpoints exceed their byte limit.")
                    await cur.execute(
                        """
                        INSERT INTO cayu_eval_run_trial_checkpoints (
                            run_id, case_id, trial_number, checkpoint_json, document_bytes
                        ) VALUES (%s, %s, %s, %s, %s)
                        """,
                        (
                            claim.run_id,
                            *key,
                            prepared.document,
                            prepared.document_bytes,
                        ),
                    )
                    await cur.execute(
                        """
                        UPDATE cayu_eval_runs
                        SET trial_checkpoint_count = trial_checkpoint_count + 1,
                            trial_checkpoint_bytes = trial_checkpoint_bytes + %s,
                            updated_at = %s
                        WHERE run_id = %s
                        """,
                        (prepared.document_bytes, now, claim.run_id),
                    )
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise

    async def submit_scenario_approval(
        self,
        run_id: str,
        submission: EvalScenarioApprovalSubmission,
    ) -> EvalRunRecord:
        run_id = _store_identifier(run_id, "run_id")
        submission = _exact_model(submission, EvalScenarioApprovalSubmission, "submission")
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    row = await self._require_run_row(cur, run_id, for_update=True)
                    if EvalRunStatus(row[8]) is not EvalRunStatus.RUNNING:
                        raise EvalRunStateConflict("Scenario approval requires an active run.")
                    if row[22] is None:
                        raise EvalRunStateConflict("Scenario progress is unavailable.")
                    progress = EvalScenarioRunProgress.model_validate_json(row[22])
                    if progress.revision != submission.expected_progress_revision:
                        raise EvalRunStateConflict(
                            "Scenario progress changed before approval submission."
                        )
                    if submission.trial_number > len(progress.trials):
                        raise EvalRunStateConflict("Scenario trial does not exist.")
                    trial = progress.trials[submission.trial_number - 1]
                    if (
                        trial.phase is not EvalScenarioTrialPhase.AWAITING_APPROVAL
                        or trial.pending_event_id != submission.event_id
                        or trial.approval is not None
                    ):
                        raise EvalRunStateConflict(
                            "Scenario approval checkpoint is no longer pending."
                        )
                    now = await _database_now(cur)
                    updated_trial = trial.model_copy(
                        update={
                            "approval": EvalScenarioApprovalDecisionRecord(
                                decision=submission.decision,
                                reason=submission.reason,
                                actor_id=submission.actor_id,
                                submitted_at=now,
                            )
                        },
                        deep=True,
                    )
                    updated_progress = progress.replace_trial(updated_trial)
                    await cur.execute(
                        """
                        UPDATE cayu_eval_runs
                        SET scenario_progress_json = %s, updated_at = %s
                        WHERE run_id = %s
                        """,
                        (updated_progress.model_dump_json(), now, run_id),
                    )
                    updated = await self._require_run_row(cur, run_id)
                await conn.commit()
                return _run_record_from_row(updated)
            except BaseException:
                await conn.rollback()
                raise

    async def publish_result(
        self,
        claim: EvalRunClaim,
        result: CorpusExecutionResult,
        *,
        redact_json: Callable[[Any], Any],
    ) -> EvalRunRecord:
        claim = _exact_model(claim, EvalRunClaim, "claim")
        result, result_document = await asyncio.to_thread(
            _prepare_result_for_store,
            result,
            redact_json=redact_json,
        )
        result_text = result_document.decode("utf-8")
        result_bytes = len(result_document)
        request = await self._load_run_request(claim.run_id)
        corpus = await self.load_corpus(request.corpus_revision)
        if corpus is None:
            raise RuntimeError("Eval run references a missing immutable corpus.")
        validated = await asyncio.to_thread(
            validate_result_for_run,
            request,
            result,
            corpus,
        )
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    row = await self._require_run_row(cur, claim.run_id, for_update=True)
                    if _request_from_row(row) != request:
                        raise EvalRunStateConflict(
                            "Eval run request changed during result publication."
                        )
                    status = EvalRunStatus(row[8])
                    if status is EvalRunStatus.COMPLETED:
                        await cur.execute(
                            "SELECT revision, result FROM cayu_eval_results WHERE run_id = %s",
                            (claim.run_id,),
                        )
                        existing = await cur.fetchone()
                        if (
                            self._claim_matches(row, claim)
                            and existing is not None
                            and existing[0] == validated.revision
                            and existing[1] == result_text
                        ):
                            await conn.commit()
                            return _run_record_from_row(row)
                        raise EvalRunStateConflict("Eval run already has another terminal result.")
                    now = await _database_now(cur)
                    self._require_live_claim(row, claim, now)
                    if status is not EvalRunStatus.RUNNING:
                        raise EvalRunStateConflict("Only a running eval may publish a result.")
                    _validate_trial_checkpoints_for_result(
                        await _load_trial_checkpoints(cur, row),
                        validated,
                    )
                    summary = result_summary(validated)
                    await cur.execute(
                        """
                        INSERT INTO cayu_eval_results (
                            run_id, revision, result, result_bytes, created_at
                        ) VALUES (%s, %s, %s, %s, %s)
                        """,
                        (
                            claim.run_id,
                            validated.revision,
                            result_text,
                            result_bytes,
                            now,
                        ),
                    )
                    await self._save_result_in_transaction(
                        cur,
                        result=validated,
                        document_text=result_text,
                        document_bytes=result_bytes,
                        created_at=now,
                        fresh_run_id=claim.run_id,
                    )
                    await cur.execute(
                        """
                        UPDATE cayu_eval_runs
                        SET status = %s, updated_at = %s, finished_at = %s,
                            lease_expires_at = NULL, result_revision = %s,
                            result_status = %s, result_score = %s,
                            result_duration_ms = %s, trial_checkpoint_count = 0,
                            trial_checkpoint_bytes = 0
                        WHERE run_id = %s
                        """,
                        (
                            str(EvalRunStatus.COMPLETED),
                            now,
                            now,
                            summary.revision,
                            summary.status,
                            summary.score,
                            summary.duration_ms,
                            claim.run_id,
                        ),
                    )
                    await _delete_trial_checkpoints(cur, claim.run_id)
                    updated = await self._require_run_row(cur, claim.run_id)
                await conn.commit()
                return _run_record_from_row(updated)
            except BaseException:
                await conn.rollback()
                raise

    async def fail_run(
        self,
        claim: EvalRunClaim,
        code: EvalRunFailureCode,
    ) -> EvalRunRecord:
        claim = _exact_model(claim, EvalRunClaim, "claim")
        if not isinstance(code, EvalRunFailureCode):
            raise TypeError("code must be an EvalRunFailureCode.")
        return await self._terminalize_without_result(
            claim,
            required_status=EvalRunStatus.RUNNING,
            terminal_status=EvalRunStatus.FAILED,
            failure_code=code,
        )

    async def finish_cancel(self, claim: EvalRunClaim) -> EvalRunRecord:
        claim = _exact_model(claim, EvalRunClaim, "claim")
        return await self._terminalize_without_result(
            claim,
            required_status=EvalRunStatus.CANCELLING,
            terminal_status=EvalRunStatus.CANCELLED,
            failure_code=None,
        )

    async def release_run(self, claim: EvalRunClaim) -> EvalRunRecord:
        claim = _exact_model(claim, EvalRunClaim, "claim")
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    row = await self._require_run_row(cur, claim.run_id, for_update=True)
                    now = await _database_now(cur)
                    self._require_live_claim(row, claim, now)
                    status = EvalRunStatus(row[8])
                    if status is EvalRunStatus.CANCELLING:
                        next_status = EvalRunStatus.CANCELLED
                        finished_at = now
                        cancel_requested_at = row[13] or now
                    elif status is EvalRunStatus.RUNNING:
                        next_status = EvalRunStatus.QUEUED
                        finished_at = None
                        cancel_requested_at = None
                    else:
                        raise EvalRunStateConflict("Only active eval work may be released.")
                    await cur.execute(
                        """
                        UPDATE cayu_eval_runs
                        SET status = %s, updated_at = %s, finished_at = %s,
                            cancel_requested_at = %s, claim_id = NULL,
                            lease_expires_at = NULL,
                            trial_checkpoint_count = CASE WHEN %s = %s THEN 0
                                ELSE trial_checkpoint_count END,
                            trial_checkpoint_bytes = CASE WHEN %s = %s THEN 0
                                ELSE trial_checkpoint_bytes END
                        WHERE run_id = %s
                        """,
                        (
                            str(next_status),
                            now,
                            finished_at,
                            cancel_requested_at,
                            str(next_status),
                            str(EvalRunStatus.CANCELLED),
                            str(next_status),
                            str(EvalRunStatus.CANCELLED),
                            claim.run_id,
                        ),
                    )
                    if next_status is EvalRunStatus.CANCELLED:
                        await _delete_trial_checkpoints(cur, claim.run_id)
                    updated = await self._require_run_row(cur, claim.run_id)
                await conn.commit()
                return _run_record_from_row(updated)
            except BaseException:
                await conn.rollback()
                raise

    async def load_result(
        self,
        run_id: str,
        *,
        max_bytes: int = CORPUS_EXECUTION_RESULT_MAX_BYTES,
    ) -> CorpusExecutionResult | None:
        run_id = _store_identifier(run_id, "run_id")
        max_bytes = _read_limit(max_bytes, hard_max=CORPUS_EXECUTION_RESULT_MAX_BYTES)
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT result_bytes FROM cayu_eval_results WHERE run_id = %s",
                (run_id,),
            )
            size = await cur.fetchone()
            if size is None:
                return None
            if size[0] > max_bytes:
                raise EvalStoreResultTooLarge(max_bytes)
            await cur.execute(
                "SELECT result FROM cayu_eval_results WHERE run_id = %s",
                (run_id,),
            )
            row = await cur.fetchone()
            if row is None:
                raise RuntimeError("Immutable eval result disappeared during a read.")
            document = row[0]
        return await asyncio.to_thread(corpus_execution_result_from_json, document)

    async def save_captured_result(
        self,
        corpus: EvalCorpusDocument,
        result: CapturedEvaluationResultV1,
        *,
        redact_json: Callable[[Any], Any],
    ) -> EvalResultRecord:
        corpus, corpus_document, inspection, suites, cases = await asyncio.to_thread(
            _prepare_corpus_catalog_for_store,
            corpus,
            redact_json=redact_json,
        )
        result, result_document = await asyncio.to_thread(
            _prepare_captured_result_for_store,
            result,
            corpus,
            redact_json=redact_json,
        )
        corpus_text = corpus_document.decode("utf-8")
        result_text = result_document.decode("utf-8")
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    now = await _database_now(cur)
                    await self._save_prepared_corpus_in_transaction(
                        cur,
                        corpus=corpus,
                        document_text=corpus_text,
                        document_bytes=len(corpus_document),
                        inspection=inspection,
                        suites=suites,
                        cases=cases,
                        created_at=now,
                    )
                    record = await self._save_result_in_transaction(
                        cur,
                        result=result,
                        document_text=result_text,
                        document_bytes=len(result_document),
                        created_at=now,
                        fresh_run_id=None,
                    )
                await conn.commit()
                return record
            except BaseException:
                await conn.rollback()
                raise

    async def load_result_by_revision(
        self,
        revision: str,
        *,
        max_bytes: int = CORPUS_EXECUTION_RESULT_MAX_BYTES,
    ) -> CorpusExecutionResult | CapturedEvaluationResultV1 | None:
        revision = _sha256_revision(revision, "revision")
        max_bytes = _read_limit(max_bytes, hard_max=CORPUS_EXECUTION_RESULT_MAX_BYTES)
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT origin, document_bytes, fresh_run_id
                FROM cayu_eval_result_records
                WHERE revision = %s
                """,
                (revision,),
            )
            row = await cur.fetchone()
            if row is None:
                return None
            if row[1] > max_bytes:
                raise EvalStoreResultTooLarge(max_bytes)
            origin = EvalResultOrigin(row[0])
            if origin is EvalResultOrigin.CAPTURED_SESSION:
                await cur.execute(
                    "SELECT captured_result FROM cayu_eval_result_records WHERE revision = %s",
                    (revision,),
                )
            else:
                await cur.execute(
                    "SELECT result FROM cayu_eval_results WHERE run_id = %s",
                    (row[2],),
                )
            document_row = await cur.fetchone()
            document = None if document_row is None else document_row[0]
            if document is None:
                raise RuntimeError("Immutable eval result document is unavailable.")
        if origin is EvalResultOrigin.CAPTURED_SESSION:
            return await asyncio.to_thread(captured_evaluation_result_from_json, document)
        return await asyncio.to_thread(corpus_execution_result_from_json, document)

    async def load_result_record(self, revision: str) -> EvalResultRecord | None:
        revision = _sha256_revision(revision, "revision")
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                f"SELECT {_RESULT_RECORD_COLUMNS} FROM cayu_eval_result_records "
                "WHERE revision = %s",
                (revision,),
            )
            row = await cur.fetchone()
            return None if row is None else _result_record_from_row(row)

    async def list_results(self, query: EvalResultQuery) -> EvalResultPage:
        query = _exact_model(query, EvalResultQuery, "query")
        boundary = (
            decode_result_cursor(query.cursor, query.target_key, query.origin)
            if query.cursor is not None
            else None
        )
        clauses = ["target_key = %s"]
        params: list[object] = [query.target_key]
        if query.origin is not None:
            clauses.append("origin = %s")
            params.append(query.origin.value)
        if boundary is not None:
            clauses.append("(created_at < %s OR (created_at = %s AND revision > %s))")
            params.extend((boundary[0], boundary[0], boundary[1]))
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                cast(
                    "LiteralString",
                    f"""
                    SELECT {_RESULT_RECORD_COLUMNS}
                    FROM cayu_eval_result_records
                    WHERE {" AND ".join(clauses)}
                    ORDER BY created_at DESC, revision ASC
                    LIMIT %s
                    """,
                ),
                (*params, query.limit + 1),
            )
            rows = await cur.fetchall()
        return _bounded_result_page([_result_record_from_row(row) for row in rows], query)

    async def set_baseline(
        self,
        update: EvalBaselineUpdate,
        *,
        redact_json: Callable[[Any], Any],
    ) -> EvalBaselineMutationRecord:
        update = _prepare_baseline_update_for_store(update, redact_json=redact_json)
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                        (update.operation_id,),
                    )
                    await cur.execute(
                        """
                        SELECT operation_id, target_key, corpus_revision, suite_id,
                               expected_generation, previous_result_revision,
                               selected_result_revision, resulting_generation,
                               actor_id, created_at
                        FROM cayu_eval_baseline_mutations
                        WHERE operation_id = %s
                        """,
                        (update.operation_id,),
                    )
                    replay_row = await cur.fetchone()
                    if replay_row is not None:
                        replay = _baseline_mutation_from_row(replay_row)
                        if not self._baseline_mutation_matches(replay, update):
                            raise EvalBaselineConflict(
                                "Baseline operation id is already bound to another mutation."
                            )
                        await conn.commit()
                        return replay
                    await cur.execute(
                        f"SELECT {_RESULT_RECORD_COLUMNS} FROM cayu_eval_result_records "
                        "WHERE revision = %s",
                        (update.result_revision,),
                    )
                    result_row = await cur.fetchone()
                    if result_row is None:
                        raise KeyError(f"Eval result not found: {update.result_revision}")
                    _validate_baseline_result(update, _result_record_from_row(result_row))
                    await cur.execute(
                        """
                        SELECT 1 FROM cayu_eval_suites
                        WHERE corpus_revision = %s AND suite_id = %s
                        FOR UPDATE
                        """,
                        (update.key.corpus_revision, update.key.suite_id),
                    )
                    if await cur.fetchone() is None:
                        raise KeyError(f"Eval corpus not found: {update.key.corpus_revision}")
                    await cur.execute(
                        """
                        SELECT target_key, corpus_revision, suite_id, result_revision,
                               generation, updated_by, updated_at
                        FROM cayu_eval_baselines
                        WHERE target_key = %s AND corpus_revision = %s AND suite_id = %s
                        FOR UPDATE
                        """,
                        (
                            update.key.target_key,
                            update.key.corpus_revision,
                            update.key.suite_id,
                        ),
                    )
                    current_row = await cur.fetchone()
                    current = (
                        None if current_row is None else _baseline_record_from_row(current_row)
                    )
                    generation = 0 if current is None else current.generation
                    if generation != update.expected_generation:
                        raise EvalBaselineConflict("Eval baseline generation changed.")
                    if generation >= 9223372036854775807:
                        raise EvalBaselineConflict("Eval baseline generation is exhausted.")
                    now = await _database_now(cur)
                    next_generation = generation + 1
                    await cur.execute(
                        """
                        INSERT INTO cayu_eval_baselines (
                            target_key, corpus_revision, suite_id, result_revision,
                            generation, updated_by, updated_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (target_key, corpus_revision, suite_id) DO UPDATE SET
                            result_revision = EXCLUDED.result_revision,
                            generation = EXCLUDED.generation,
                            updated_by = EXCLUDED.updated_by,
                            updated_at = EXCLUDED.updated_at
                        """,
                        (
                            update.key.target_key,
                            update.key.corpus_revision,
                            update.key.suite_id,
                            update.result_revision,
                            next_generation,
                            update.actor_id,
                            now,
                        ),
                    )
                    await cur.execute(
                        """
                        INSERT INTO cayu_eval_baseline_mutations (
                            operation_id, target_key, corpus_revision, suite_id,
                            expected_generation, previous_result_revision,
                            selected_result_revision, resulting_generation, actor_id, created_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING operation_id, target_key, corpus_revision, suite_id,
                                  expected_generation, previous_result_revision,
                                  selected_result_revision, resulting_generation,
                                  actor_id, created_at
                        """,
                        (
                            update.operation_id,
                            update.key.target_key,
                            update.key.corpus_revision,
                            update.key.suite_id,
                            update.expected_generation,
                            None if current is None else current.result_revision,
                            update.result_revision,
                            next_generation,
                            update.actor_id,
                            now,
                        ),
                    )
                    mutation_row = await cur.fetchone()
                    assert mutation_row is not None
                await conn.commit()
                return _baseline_mutation_from_row(mutation_row)
            except BaseException:
                await conn.rollback()
                raise

    async def load_baseline(self, key: EvalBaselineKey) -> EvalBaselineRecord | None:
        key = _exact_model(key, EvalBaselineKey, "key")
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT target_key, corpus_revision, suite_id, result_revision,
                       generation, updated_by, updated_at
                FROM cayu_eval_baselines
                WHERE target_key = %s AND corpus_revision = %s AND suite_id = %s
                """,
                (key.target_key, key.corpus_revision, key.suite_id),
            )
            row = await cur.fetchone()
            return None if row is None else _baseline_record_from_row(row)

    async def load_baseline_mutation(
        self,
        operation_id: str,
    ) -> EvalBaselineMutationRecord | None:
        operation_id = _sha256_revision(operation_id, "operation_id")
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT operation_id, target_key, corpus_revision, suite_id,
                       expected_generation, previous_result_revision,
                       selected_result_revision, resulting_generation,
                       actor_id, created_at
                FROM cayu_eval_baseline_mutations
                WHERE operation_id = %s
                """,
                (operation_id,),
            )
            row = await cur.fetchone()
            return None if row is None else _baseline_mutation_from_row(row)

    async def _terminalize_without_result(
        self,
        claim: EvalRunClaim,
        *,
        required_status: EvalRunStatus,
        terminal_status: EvalRunStatus,
        failure_code: EvalRunFailureCode | None,
    ) -> EvalRunRecord:
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    row = await self._require_run_row(cur, claim.run_id, for_update=True)
                    status = EvalRunStatus(row[8])
                    if status is terminal_status:
                        stored_code = None if row[21] is None else EvalRunFailureCode(row[21])
                        if self._claim_matches(row, claim) and stored_code is failure_code:
                            await conn.commit()
                            return _run_record_from_row(row)
                        raise EvalRunStateConflict("Eval run already has another terminal outcome.")
                    now = await _database_now(cur)
                    self._require_live_claim(row, claim, now)
                    if status is not required_status:
                        raise EvalRunStateConflict(
                            f"Eval run must be {required_status} before becoming {terminal_status}."
                        )
                    cancel_requested_at = row[13]
                    if terminal_status is EvalRunStatus.CANCELLED and cancel_requested_at is None:
                        cancel_requested_at = now
                    await cur.execute(
                        """
                        UPDATE cayu_eval_runs
                        SET status = %s, updated_at = %s, finished_at = %s,
                            cancel_requested_at = %s, lease_expires_at = NULL,
                            failure_code = %s, trial_checkpoint_count = 0,
                            trial_checkpoint_bytes = 0
                        WHERE run_id = %s
                        """,
                        (
                            str(terminal_status),
                            now,
                            now,
                            cancel_requested_at,
                            None if failure_code is None else str(failure_code),
                            claim.run_id,
                        ),
                    )
                    await _delete_trial_checkpoints(cur, claim.run_id)
                    updated = await self._require_run_row(cur, claim.run_id)
                await conn.commit()
                return _run_record_from_row(updated)
            except BaseException:
                await conn.rollback()
                raise

    @staticmethod
    def _claim_matches(row: Any, claim: EvalRunClaim) -> bool:
        return row[14] == claim.claim_id and row[15] == claim.epoch

    @classmethod
    def _require_live_claim(cls, row: Any, claim: EvalRunClaim, now: datetime) -> None:
        if not cls._claim_matches(row, claim):
            raise EvalRunClaimLost("Eval run claim is no longer owned by this worker.")
        if EvalRunStatus(row[8]) not in {EvalRunStatus.RUNNING, EvalRunStatus.CANCELLING}:
            raise EvalRunClaimLost("Eval run is no longer active.")
        if row[16] is None or row[16] <= now:
            raise EvalRunClaimLost("Eval run claim lease has expired.")

    @staticmethod
    async def _load_run_row(cur: Any, run_id: str, *, for_update: bool = False) -> Any:
        suffix = " FOR UPDATE" if for_update else ""
        await cur.execute(
            f"SELECT {_RUN_COLUMNS} FROM cayu_eval_runs WHERE run_id = %s{suffix}",
            (run_id,),
        )
        return await cur.fetchone()

    @classmethod
    async def _require_run_row(
        cls,
        cur: Any,
        run_id: str,
        *,
        for_update: bool = False,
    ) -> Any:
        row = await cls._load_run_row(cur, run_id, for_update=for_update)
        if row is None:
            raise KeyError(f"Eval run not found: {run_id}")
        return row

    @staticmethod
    async def _load_run_by_idempotency_key(cur: Any, key: str) -> Any:
        await cur.execute(
            f"SELECT {_RUN_COLUMNS} FROM cayu_eval_runs WHERE idempotency_key = %s",
            (key,),
        )
        return await cur.fetchone()

    async def _load_run_request(self, run_id: str) -> EvalRunRequest:
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            row = await self._require_run_row(cur, run_id)
            return _request_from_row(row)

    @classmethod
    async def _save_result_in_transaction(
        cls,
        cur: Any,
        *,
        result: CorpusExecutionResult | CapturedEvaluationResultV1,
        document_text: str,
        document_bytes: int,
        created_at: datetime,
        fresh_run_id: str | None,
    ) -> EvalResultRecord:
        record = eval_result_record(
            result,
            document_bytes=document_bytes,
            created_at=created_at,
        )
        await cur.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 1))",
            (record.revision,),
        )
        await cur.execute(
            f"SELECT {_RESULT_RECORD_COLUMNS}, fresh_run_id, captured_result "
            "FROM cayu_eval_result_records WHERE revision = %s FOR UPDATE",
            (record.revision,),
        )
        existing_row = await cur.fetchone()
        if existing_row is not None:
            existing = _result_record_from_row(existing_row)
            if existing != record.model_copy(update={"created_at": existing.created_at}):
                raise EvalResultConflict(
                    f"Eval result revision {record.revision} has conflicting metadata."
                )
            if existing.origin is EvalResultOrigin.CAPTURED_SESSION:
                existing_document = existing_row[14]
            else:
                await cur.execute(
                    "SELECT result FROM cayu_eval_results WHERE run_id = %s",
                    (existing_row[13],),
                )
                fresh = await cur.fetchone()
                existing_document = None if fresh is None else fresh[0]
            if existing_document != document_text:
                raise EvalResultConflict(
                    f"Eval result revision {record.revision} has conflicting content."
                )
            return existing
        captured_document = (
            document_text if record.origin is EvalResultOrigin.CAPTURED_SESSION else None
        )
        if (record.origin is EvalResultOrigin.FRESH_EXECUTION) != (fresh_run_id is not None):
            raise ValueError("Fresh eval result records require their durable run mapping.")
        await cur.execute(
            """
            INSERT INTO cayu_eval_result_records (
                revision, origin, target_key, corpus_revision, suite_id, suite_revision,
                application_release_id, app_manifest_schema_version,
                app_manifest_fingerprint, result_status, result_score, fresh_run_id,
                captured_result, document_bytes, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                record.revision,
                record.origin.value,
                record.target.target_key,
                record.corpus_revision,
                record.suite_id,
                record.suite_revision,
                record.target.application_release_id,
                record.target.app_manifest_schema_version,
                record.target.app_manifest_fingerprint,
                record.status,
                record.score,
                fresh_run_id,
                captured_document,
                document_bytes,
                created_at,
            ),
        )
        return record

    @staticmethod
    def _baseline_mutation_matches(
        mutation: EvalBaselineMutationRecord,
        update: EvalBaselineUpdate,
    ) -> bool:
        return (
            mutation.operation_id == update.operation_id
            and mutation.key == update.key
            and mutation.expected_generation == update.expected_generation
            and mutation.selected_result_revision == update.result_revision
            and mutation.actor_id == update.actor_id
        )

    @classmethod
    async def _save_prepared_corpus_in_transaction(
        cls,
        cur: Any,
        *,
        corpus: EvalCorpusDocument,
        document_text: str,
        document_bytes: int,
        inspection: EvalCorpusInspectionV1,
        suites: tuple[EvalSuiteCatalogEntry, ...],
        cases: tuple[EvalCaseCatalogEntry, ...],
        created_at: datetime,
    ) -> EvalCorpusCatalogEntry:
        await cur.execute(
            """
            INSERT INTO cayu_eval_corpora (
                revision, target_key, evidence_policy_revision,
                pricing_profile_fingerprint, suite_count, case_count,
                assertion_count, expanded_assertion_result_count,
                document, document_bytes, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (revision) DO NOTHING
            RETURNING revision
            """,
            (
                corpus.revision,
                inspection.target_key,
                inspection.evidence_policy_revision,
                inspection.pricing_profile_fingerprint,
                inspection.suite_count,
                inspection.case_count,
                inspection.assertion_count,
                inspection.expanded_assertion_result_count,
                document_text,
                document_bytes,
                created_at,
            ),
        )
        inserted = await cur.fetchone()
        if inserted is None:
            await cur.execute(
                "SELECT document FROM cayu_eval_corpora WHERE revision = %s",
                (corpus.revision,),
            )
            existing = await cur.fetchone()
            if existing is None or existing[0] != document_text:
                raise EvalCorpusConflict(
                    f"Eval corpus revision {corpus.revision} has conflicting content."
                )
            entry = await cls._load_corpus_entry(cur, corpus.revision)
            assert entry is not None
            return entry
        await cur.executemany(
            """
            INSERT INTO cayu_eval_suites (
                corpus_revision, suite_id, suite_revision, name, description,
                case_count, assertion_count, trials, timeout_seconds
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    item.corpus_revision,
                    item.id,
                    item.revision,
                    item.name,
                    item.description,
                    item.case_count,
                    item.assertion_count,
                    item.trials,
                    item.timeout_seconds,
                )
                for item in suites
            ],
        )
        await cur.executemany(
            """
            INSERT INTO cayu_eval_cases (
                corpus_revision, case_id, case_revision, suite_id, name,
                description, message_count, assertion_count
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    item.corpus_revision,
                    item.id,
                    item.revision,
                    item.suite_id,
                    item.name,
                    item.description,
                    item.message_count,
                    item.assertion_count,
                )
                for item in cases
            ],
        )
        entry = await cls._load_corpus_entry(cur, corpus.revision)
        assert entry is not None
        return entry

    @staticmethod
    async def _corpus_exists(cur: Any, revision: str) -> bool:
        await cur.execute(
            "SELECT 1 FROM cayu_eval_corpora WHERE revision = %s",
            (revision,),
        )
        return await cur.fetchone() is not None

    @classmethod
    async def _load_corpus_entry(
        cls,
        cur: Any,
        revision: str,
    ) -> EvalCorpusCatalogEntry | None:
        await cur.execute(
            """
            SELECT revision, target_key, evidence_policy_revision,
                   pricing_profile_fingerprint, suite_count, case_count,
                   assertion_count, expanded_assertion_result_count,
                   document_bytes, created_at
            FROM cayu_eval_corpora
            WHERE revision = %s
            """,
            (revision,),
        )
        row = await cur.fetchone()
        return None if row is None else cls._corpus_entry_from_row(row)

    @staticmethod
    def _corpus_entry_from_row(row: Any) -> EvalCorpusCatalogEntry:
        return EvalCorpusCatalogEntry(
            revision=row[0],
            target_key=row[1],
            evidence_policy_revision=row[2],
            pricing_profile_fingerprint=row[3],
            suite_count=row[4],
            case_count=row[5],
            assertion_count=row[6],
            expanded_assertion_result_count=row[7],
            document_bytes=row[8],
            created_at=row[9],
        )

    @classmethod
    async def _load_scenario_entry(
        cls,
        cur: Any,
        revision: str,
    ) -> EvalScenarioCatalogEntry | None:
        await cur.execute(
            """
            SELECT revision, scenario_id, target_key, name, description,
                   event_count, input_event_count, approval_checkpoint_count,
                   message_count, part_count, artifact_requirement_count,
                   secret_requirement_count, document_bytes, created_at
            FROM cayu_eval_scenarios
            WHERE revision = %s
            """,
            (revision,),
        )
        row = await cur.fetchone()
        return None if row is None else cls._scenario_entry_from_row(row)

    @staticmethod
    def _scenario_entry_from_row(row: Any) -> EvalScenarioCatalogEntry:
        return EvalScenarioCatalogEntry(
            revision=row[0],
            id=row[1],
            target_key=row[2],
            name=row[3],
            description=row[4],
            event_count=row[5],
            input_event_count=row[6],
            approval_checkpoint_count=row[7],
            message_count=row[8],
            part_count=row[9],
            artifact_requirement_count=row[10],
            secret_requirement_count=row[11],
            document_bytes=row[12],
            created_at=row[13],
        )

    @classmethod
    async def _load_authored_suite_entry(
        cls,
        cur: Any,
        revision: str,
    ) -> EvalAuthoredSuiteCatalogEntry | None:
        await cur.execute(
            """
            SELECT revision, suite_id, suite_revision, target_key, name,
                   description, case_count, assertion_count, simple_input_count,
                   scenario_count, trials, timeout_seconds, document_bytes, created_at
            FROM cayu_eval_authored_suites
            WHERE revision = %s
            """,
            (revision,),
        )
        row = await cur.fetchone()
        return None if row is None else cls._authored_suite_entry_from_row(row)

    @staticmethod
    def _authored_suite_entry_from_row(row: Any) -> EvalAuthoredSuiteCatalogEntry:
        return EvalAuthoredSuiteCatalogEntry(
            revision=row[0],
            id=row[1],
            suite_revision=row[2],
            target_key=row[3],
            name=row[4],
            description=row[5],
            case_count=row[6],
            assertion_count=row[7],
            simple_input_count=row[8],
            scenario_count=row[9],
            trials=row[10],
            timeout_seconds=row[11],
            document_bytes=row[12],
            created_at=row[13],
        )

    @staticmethod
    def _suite_entry_from_row(row: Any) -> EvalSuiteCatalogEntry:
        return EvalSuiteCatalogEntry(
            corpus_revision=row[0],
            id=row[1],
            revision=row[2],
            name=row[3],
            description=row[4],
            case_count=row[5],
            assertion_count=row[6],
            trials=row[7],
            timeout_seconds=row[8],
        )

    @staticmethod
    def _case_entry_from_row(row: Any) -> EvalCaseCatalogEntry:
        return EvalCaseCatalogEntry(
            corpus_revision=row[0],
            id=row[1],
            revision=row[2],
            suite_id=row[3],
            name=row[4],
            description=row[5],
            message_count=row[6],
            assertion_count=row[7],
        )


__all__ = ["PostgresEvalStore"]
