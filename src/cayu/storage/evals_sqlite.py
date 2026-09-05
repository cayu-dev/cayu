from __future__ import annotations

import asyncio
import logging
import math
import sqlite3
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import monotonic
from typing import Any, ClassVar
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
    EvalRunFailureDiagnostic,
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
    EvalStoreTransientContention,
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
from cayu.storage import _sqlite_support as sqlite_support
from cayu.storage import migrations as schema
from cayu.storage.sqlite import _run_off_thread_with_connection_ownership

_SQLITE_EVAL_MIN_REQUIRED_REVISION = 80

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SQLiteEvalWriterContentionPolicy:
    """Bounded SQLite lock acquisition and retry policy for paid-effect evidence."""

    max_wait_seconds: float = 60.0
    lock_attempt_seconds: float = 0.25
    initial_backoff_seconds: float = 0.01
    max_backoff_seconds: float = 0.25

    def __post_init__(self) -> None:
        for field_name in (
            "max_wait_seconds",
            "lock_attempt_seconds",
            "initial_backoff_seconds",
            "max_backoff_seconds",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"{field_name} must be a number.")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field_name} must be finite and greater than zero.")
        if self.lock_attempt_seconds > self.max_wait_seconds:
            raise ValueError("lock_attempt_seconds cannot exceed max_wait_seconds.")
        if self.initial_backoff_seconds > self.max_backoff_seconds:
            raise ValueError("initial_backoff_seconds cannot exceed max_backoff_seconds.")


DEFAULT_SQLITE_EVAL_WRITER_CONTENTION_POLICY = SQLiteEvalWriterContentionPolicy()

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
    authored_suite_launch_lane,
    failure_diagnostic_json
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


def _format_datetime(value: datetime) -> str:
    return sqlite_support.format_datetime(value)


def _parse_optional_datetime(value: str | None) -> datetime | None:
    return None if value is None else sqlite_support.parse_datetime(value)


def _is_sqlite_writer_contention(error: sqlite3.Error) -> bool:
    code = getattr(error, "sqlite_errorcode", None)
    return type(code) is int and code & 0xFF in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}


def _request_from_row(row: sqlite3.Row) -> EvalRunRequest:
    invocation = eval_run_invocation_from_json(row["invocation_json"])
    if row["authored_suite_launch_revision"] != invocation.authored_suite_launch_revision:
        raise RuntimeError(
            "Stored authored-suite launch identity conflicts with durable invocation."
        )
    if row["authored_suite_launch_lane"] != invocation.authored_suite_launch_lane:
        raise RuntimeError("Stored authored-suite launch lane conflicts with durable invocation.")
    return EvalRunRequest(
        run_id=row["run_id"],
        idempotency_key=row["idempotency_key"],
        corpus_revision=row["corpus_revision"],
        target_key=row["target_key"],
        suite_id=row["suite_id"],
        suite_revision=row["suite_revision"],
        max_concurrency=row["max_concurrency"],
        invocation=invocation,
    )


def _run_observation_from_row(row: sqlite3.Row) -> EvalRunObservation:
    status = EvalRunStatus(row["status"])
    ownership = None
    if status in {EvalRunStatus.RUNNING, EvalRunStatus.CANCELLING}:
        lease_expires_at = _parse_optional_datetime(row["lease_expires_at"])
        if lease_expires_at is None:
            raise RuntimeError("Stored active eval run has no lease expiry.")
        ownership = EvalRunOwnership(
            epoch=row["ownership_epoch"],
            lease_expires_at=lease_expires_at,
        )
    return EvalRunObservation(
        run_id=row["run_id"],
        status=status,
        attempt_count=row["ownership_epoch"],
        updated_at=sqlite_support.parse_datetime(row["updated_at"]),
        ownership=ownership,
    )


def _load_trial_checkpoints(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
) -> tuple[EvalRunTrialCheckpoint, ...]:
    checkpoint_rows = connection.execute(
        """
        SELECT case_id, trial_number, checkpoint_json, document_bytes
        FROM cayu_eval_run_trial_checkpoints
        WHERE run_id = ?
        ORDER BY case_id ASC, trial_number ASC
        """,
        (row["run_id"],),
    ).fetchall()
    checkpoints: list[EvalRunTrialCheckpoint] = []
    document_bytes = 0
    for checkpoint_row in checkpoint_rows:
        checkpoint = eval_run_trial_checkpoint_from_json(checkpoint_row["checkpoint_json"])
        actual_bytes = len(checkpoint_row["checkpoint_json"].encode("utf-8"))
        if (
            checkpoint.case_id != checkpoint_row["case_id"]
            or checkpoint.trial_number != checkpoint_row["trial_number"]
            or checkpoint_row["document_bytes"] != actual_bytes
        ):
            raise RuntimeError("Stored eval trial checkpoint contradicts its indexed slot.")
        checkpoints.append(checkpoint)
        document_bytes += actual_bytes
    if len(checkpoints) != row["trial_checkpoint_count"]:
        raise RuntimeError("Stored eval trial checkpoint count is inconsistent.")
    if document_bytes != row["trial_checkpoint_bytes"]:
        raise RuntimeError("Stored eval trial checkpoint byte total is inconsistent.")
    return _validated_trial_checkpoints(
        tuple(checkpoints),
        expected_document_bytes=document_bytes,
    )


def _delete_trial_checkpoints(connection: sqlite3.Connection, run_id: str) -> None:
    connection.execute(
        "DELETE FROM cayu_eval_run_trial_checkpoints WHERE run_id = ?",
        (run_id,),
    )


def _run_record_from_row(row: sqlite3.Row) -> EvalRunRecord:
    status = EvalRunStatus(row["status"])
    ownership = None
    if status in {EvalRunStatus.RUNNING, EvalRunStatus.CANCELLING}:
        ownership = EvalRunOwnership(
            epoch=row["ownership_epoch"],
            lease_expires_at=sqlite_support.parse_datetime(row["lease_expires_at"]),
        )
    result = None
    if row["result_revision"] is not None:
        result = EvalRunResultSummary(
            revision=row["result_revision"],
            status=row["result_status"],
            score=row["result_score"],
            duration_ms=row["result_duration_ms"],
        )
    return EvalRunRecord(
        spec=EvalRunSpec(
            run_id=row["run_id"],
            corpus_revision=row["corpus_revision"],
            target_key=row["target_key"],
            suite_id=row["suite_id"],
            suite_revision=row["suite_revision"],
            max_concurrency=row["max_concurrency"],
            invocation=eval_run_invocation_from_json(row["invocation_json"]),
        ),
        status=status,
        attempt_count=row["ownership_epoch"],
        created_at=sqlite_support.parse_datetime(row["created_at"]),
        updated_at=sqlite_support.parse_datetime(row["updated_at"]),
        started_at=_parse_optional_datetime(row["started_at"]),
        finished_at=_parse_optional_datetime(row["finished_at"]),
        cancel_requested_at=_parse_optional_datetime(row["cancel_requested_at"]),
        ownership=ownership,
        result=result,
        failure_diagnostic=(
            None
            if row["failure_diagnostic_json"] is None
            else EvalRunFailureDiagnostic.model_validate_json(row["failure_diagnostic_json"])
        ),
        failure_code=(
            None if row["failure_code"] is None else EvalRunFailureCode(row["failure_code"])
        ),
        scenario_progress=(
            None
            if row["scenario_progress_json"] is None
            else EvalScenarioRunProgress.model_validate_json(row["scenario_progress_json"])
        ),
    )


def _result_record_from_row(row: sqlite3.Row) -> EvalResultRecord:
    return EvalResultRecord(
        revision=row["revision"],
        origin=EvalResultOrigin(row["origin"]),
        target=EvalResultTargetIdentityV1(
            target_key=row["target_key"],
            application_release_id=row["application_release_id"],
            app_manifest_schema_version=row["app_manifest_schema_version"],
            app_manifest_fingerprint=row["app_manifest_fingerprint"],
        ),
        corpus_revision=row["corpus_revision"],
        suite_id=row["suite_id"],
        suite_revision=row["suite_revision"],
        status=row["result_status"],
        score=row["result_score"],
        document_bytes=row["document_bytes"],
        created_at=sqlite_support.parse_datetime(row["created_at"]),
    )


def _baseline_key_from_row(row: sqlite3.Row) -> EvalBaselineKey:
    return EvalBaselineKey(
        target_key=row["target_key"],
        corpus_revision=row["corpus_revision"],
        suite_id=row["suite_id"],
    )


def _baseline_record_from_row(row: sqlite3.Row) -> EvalBaselineRecord:
    return EvalBaselineRecord(
        key=_baseline_key_from_row(row),
        result_revision=row["result_revision"],
        generation=row["generation"],
        updated_by=row["updated_by"],
        updated_at=sqlite_support.parse_datetime(row["updated_at"]),
    )


def _baseline_mutation_from_row(row: sqlite3.Row) -> EvalBaselineMutationRecord:
    return EvalBaselineMutationRecord(
        operation_id=row["operation_id"],
        key=_baseline_key_from_row(row),
        expected_generation=row["expected_generation"],
        previous_result_revision=row["previous_result_revision"],
        selected_result_revision=row["selected_result_revision"],
        resulting_generation=row["resulting_generation"],
        actor_id=row["actor_id"],
        created_at=sqlite_support.parse_datetime(row["created_at"]),
    )


class SQLiteEvalStore(EvalStore):
    """Restart-durable embedded eval persistence for a single SQLite database."""

    durable: ClassVar[bool] = True
    captured_results: ClassVar[bool] = True
    scenarios: ClassVar[bool] = True
    scenario_execution: ClassVar[bool] = True
    trial_checkpointing: ClassVar[bool] = True
    suite_authoring: ClassVar[bool] = True
    judge_calibrations: ClassVar[bool] = True

    def __init__(
        self,
        path: str | Path,
        *,
        schema_mode: schema.SchemaMode = schema.SchemaMode.CREATE,
        read_only: bool = False,
        writer_contention_policy: SQLiteEvalWriterContentionPolicy = (
            DEFAULT_SQLITE_EVAL_WRITER_CONTENTION_POLICY
        ),
    ) -> None:
        if isinstance(path, Path):
            db_path = path
        elif type(path) is str and path.strip():
            db_path = Path(path)
        else:
            raise TypeError("SQLiteEvalStore path must be a nonblank string or Path.")
        if not isinstance(schema_mode, schema.SchemaMode):
            raise TypeError("schema_mode must be a SchemaMode.")
        if type(read_only) is not bool:
            raise TypeError("read_only must be a bool.")
        if type(writer_contention_policy) is not SQLiteEvalWriterContentionPolicy:
            raise TypeError(
                "writer_contention_policy must be an exact SQLiteEvalWriterContentionPolicy."
            )
        diagnostic_inspection = sqlite_support.current_diagnostic_store_inspection() is not None
        diagnostic_source_missing = sqlite_support.diagnostic_sqlite_source_missing(db_path)
        if diagnostic_inspection:
            if str(db_path) == ":memory:" or diagnostic_source_missing:
                schema_mode = schema.SchemaMode.CREATE
                read_only = False
            else:
                schema_mode = schema.SchemaMode.VALIDATE
                read_only = True
        if read_only and schema_mode is not schema.SchemaMode.VALIDATE:
            raise ValueError("Read-only SQLite EvalStores require schema_mode=VALIDATE.")
        self.path = db_path
        self._diagnostic_source_missing = diagnostic_source_missing
        self.writer_contention_policy = writer_contention_policy
        self._lock = asyncio.Lock()
        effective_db_path = Path(":memory:") if diagnostic_source_missing else db_path
        self._connection = (
            sqlite_support.connect_read_only_inspection(effective_db_path)
            if read_only
            else sqlite_support.connect(effective_db_path)
        )
        try:
            sqlite_support.reconcile_schema(
                self._connection,
                schema_mode,
                app_min_supported=_SQLITE_EVAL_MIN_REQUIRED_REVISION,
            )
            if diagnostic_source_missing:
                self._connection.execute("PRAGMA query_only = ON")
        except BaseException:
            self._connection.close()
            raise
        try:
            self._executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="cayu-evals-sqlite",
            )
        except BaseException:
            self._connection.close()
            raise
        if str(effective_db_path) == ":memory:":
            self._read_connection = self._connection
            self._read_lock = self._lock
            self._read_executor = self._executor
        else:
            try:
                self._read_connection = (
                    sqlite_support.connect_read_only_inspection(effective_db_path)
                    if read_only
                    else sqlite_support.connect(effective_db_path, read_only=True)
                )
                self._read_lock = asyncio.Lock()
                self._read_executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="cayu-evals-sqlite-read",
                )
            except BaseException:
                self._executor.shutdown(wait=True, cancel_futures=True)
                self._connection.close()
                raise

    async def _run(
        self,
        operation,
        *,
        worker_started: asyncio.Event | None = None,
    ):
        return await _run_off_thread_with_connection_ownership(
            self._lock,
            self._connection,
            operation,
            executor=self._executor,
            worker_started=worker_started,
        )

    async def _run_read(self, operation):
        if self._read_connection is self._connection:
            return await self._run(operation)
        return await _run_off_thread_with_connection_ownership(
            self._read_lock,
            self._read_connection,
            operation,
            executor=self._read_executor,
        )

    async def _run_eval_writer(
        self,
        operation,
        *,
        operation_name: str,
        run_id: str,
    ):
        """Run an eval authority/publication write with bounded contention retry."""

        policy = self.writer_contention_policy
        started_at = monotonic()
        deadline = started_at + policy.max_wait_seconds
        attempt = 0
        backoff_seconds = policy.initial_backoff_seconds
        while True:
            attempt += 1
            remaining_seconds = deadline - monotonic()
            if remaining_seconds <= 0:
                self._log_writer_event(
                    "contention_exhausted",
                    operation_name=operation_name,
                    run_id=run_id,
                    attempt=attempt - 1,
                    waited_seconds=monotonic() - started_at,
                )
                raise EvalStoreTransientContention(
                    "SQLite eval publication exhausted its bounded writer-contention budget."
                )

            def configured_operation(connection: sqlite3.Connection):
                operation_remaining_seconds = deadline - monotonic()
                if operation_remaining_seconds <= 0:
                    raise EvalStoreTransientContention(
                        "SQLite eval publication exhausted its bounded writer-contention budget."
                    )
                lock_attempt_milliseconds = max(
                    1,
                    math.ceil(
                        min(policy.lock_attempt_seconds, operation_remaining_seconds) * 1_000
                    ),
                )
                previous_timeout = connection.execute("PRAGMA busy_timeout").fetchone()[0]
                connection.execute(f"PRAGMA busy_timeout = {lock_attempt_milliseconds}")
                try:
                    return operation(connection)
                finally:
                    connection.execute(f"PRAGMA busy_timeout = {previous_timeout}")

            worker_started = asyncio.Event()
            write_attempt = asyncio.create_task(
                self._run(
                    configured_operation,
                    worker_started=worker_started,
                )
            )
            write_attempt.add_done_callback(lambda _task, event=worker_started: event.set())
            try:
                await asyncio.wait_for(
                    worker_started.wait(),
                    timeout=remaining_seconds,
                )
                result = await write_attempt
                completion_event = {
                    "save_trial_checkpoint": "checkpoint_write",
                    "publish_result": "result_publication",
                }.get(operation_name)
                if completion_event is not None:
                    logger.debug(
                        "SQLite eval publication operation completed.",
                        extra={
                            "cayu_eval_store_event": completion_event,
                            "eval_store_kind": "sqlite",
                            "eval_operation": operation_name,
                            "eval_run_id": run_id,
                            "duration_seconds": monotonic() - started_at,
                            "attempt": attempt,
                        },
                    )
                return result
            except asyncio.CancelledError:
                write_attempt.cancel()
                await asyncio.gather(write_attempt, return_exceptions=True)
                raise
            except TimeoutError as exc:
                write_attempt.cancel()
                await asyncio.gather(write_attempt, return_exceptions=True)
                self._log_writer_event(
                    "contention_exhausted",
                    operation_name=operation_name,
                    run_id=run_id,
                    attempt=attempt,
                    waited_seconds=monotonic() - started_at,
                )
                raise EvalStoreTransientContention(
                    "SQLite eval publication exhausted its bounded writer-contention budget."
                ) from exc
            except EvalStoreTransientContention as exc:
                self._log_writer_event(
                    "contention_exhausted",
                    operation_name=operation_name,
                    run_id=run_id,
                    attempt=attempt,
                    waited_seconds=monotonic() - started_at,
                )
                raise EvalStoreTransientContention(
                    "SQLite eval publication exhausted its bounded writer-contention budget."
                ) from exc
            except EvalRunClaimLost:
                self._log_writer_event(
                    "claim_lost",
                    operation_name=operation_name,
                    run_id=run_id,
                    attempt=attempt,
                    waited_seconds=monotonic() - started_at,
                )
                raise
            except sqlite3.Error as exc:
                if not _is_sqlite_writer_contention(exc):
                    self._log_writer_event(
                        "permanent_storage_failure",
                        operation_name=operation_name,
                        run_id=run_id,
                        attempt=attempt,
                        waited_seconds=monotonic() - started_at,
                    )
                    raise
                waited_seconds = monotonic() - started_at
                self._log_writer_event(
                    "lock_wait",
                    operation_name=operation_name,
                    run_id=run_id,
                    attempt=attempt,
                    waited_seconds=waited_seconds,
                )
                remaining_seconds = deadline - monotonic()
                if remaining_seconds <= 0:
                    self._log_writer_event(
                        "contention_exhausted",
                        operation_name=operation_name,
                        run_id=run_id,
                        attempt=attempt,
                        waited_seconds=waited_seconds,
                    )
                    raise EvalStoreTransientContention(
                        "SQLite eval publication exhausted its bounded writer-contention budget."
                    ) from exc
                delay_seconds = min(backoff_seconds, remaining_seconds)
                self._log_writer_event(
                    "retry",
                    operation_name=operation_name,
                    run_id=run_id,
                    attempt=attempt,
                    waited_seconds=waited_seconds,
                    retry_in_seconds=delay_seconds,
                )
                await asyncio.sleep(delay_seconds)
                backoff_seconds = min(backoff_seconds * 2, policy.max_backoff_seconds)

    @staticmethod
    def _log_writer_event(
        event: str,
        *,
        operation_name: str,
        run_id: str,
        attempt: int,
        waited_seconds: float,
        retry_in_seconds: float | None = None,
    ) -> None:
        level = logging.ERROR if event == "permanent_storage_failure" else logging.INFO
        logger.log(
            level,
            "SQLite eval writer %s.",
            event.replace("_", " "),
            extra={
                "cayu_eval_store_event": f"sqlite_writer.{event}",
                "eval_operation": operation_name,
                "eval_run_id": run_id,
                "attempt": attempt,
                "waited_seconds": waited_seconds,
                "retry_in_seconds": retry_in_seconds,
            },
        )

    async def close(self) -> None:
        try:
            if self._read_connection is not self._connection:
                await self._run_read(lambda connection: connection.close())
        finally:
            if self._read_executor is not self._executor:
                self._read_executor.shutdown(wait=True, cancel_futures=True)
            try:
                await self._run(lambda connection: connection.close())
            finally:
                self._executor.shutdown(wait=True, cancel_futures=True)

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

        def operation(connection: sqlite3.Connection) -> EvalCorpusCatalogEntry:
            try:
                connection.execute("BEGIN IMMEDIATE")
                now = datetime.now(UTC)
                entry = self._save_prepared_corpus_in_transaction(
                    connection,
                    corpus=corpus,
                    document_text=document_text,
                    document_bytes=len(document),
                    inspection=inspection,
                    suites=suites,
                    cases=cases,
                    created_at=now,
                )
                connection.commit()
                return entry
            except BaseException:
                connection.rollback()
                raise

        return await self._run(operation)

    async def load_corpus(
        self,
        revision: str,
        *,
        max_bytes: int = EVAL_CORPUS_MAX_BYTES,
    ) -> EvalCorpusDocument | None:
        revision = _sha256_revision(revision, "revision")
        max_bytes = _read_limit(max_bytes, hard_max=EVAL_CORPUS_MAX_BYTES)

        def operation(connection: sqlite3.Connection) -> str | None:
            size_row = connection.execute(
                "SELECT document_bytes FROM cayu_eval_corpora WHERE revision = ?",
                (revision,),
            ).fetchone()
            if size_row is None:
                return None
            if size_row["document_bytes"] > max_bytes:
                raise EvalStoreResultTooLarge(max_bytes)
            row = connection.execute(
                "SELECT document_json FROM cayu_eval_corpora WHERE revision = ?",
                (revision,),
            ).fetchone()
            if row is None:
                raise RuntimeError("Immutable eval corpus disappeared during a read.")
            return row["document_json"]

        document = await self._run(operation)
        if document is None:
            return None
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

        def operation(connection: sqlite3.Connection) -> EvalCorpusCatalogPage:
            clauses: list[str] = []
            params: list[object] = []
            if query.target_key is not None:
                clauses.append("target_key = ?")
                params.append(query.target_key)
            if boundary is not None:
                clauses.append("(created_at < ? OR (created_at = ? AND revision > ?))")
                timestamp = _format_datetime(boundary[0])
                params.extend((timestamp, timestamp, boundary[1]))
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            rows = connection.execute(
                f"""
                SELECT revision, target_key, evidence_policy_revision,
                       pricing_profile_fingerprint, suite_count, case_count,
                       assertion_count, expanded_assertion_result_count,
                       document_bytes, created_at
                FROM cayu_eval_corpora
                {where}
                ORDER BY created_at DESC, revision ASC
                LIMIT ?
                """,
                (*params, query.limit + 1),
            ).fetchall()
            items = [self._corpus_entry_from_row(row) for row in rows]
            return _bounded_corpus_page(items, query)

        return await self._run(operation)

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

        def operation(connection: sqlite3.Connection) -> EvalScenarioCatalogEntry:
            try:
                connection.execute("BEGIN IMMEDIATE")
                existing = connection.execute(
                    "SELECT document_json FROM cayu_eval_scenarios WHERE revision = ?",
                    (scenario.revision,),
                ).fetchone()
                if existing is not None:
                    if existing["document_json"] != document_text:
                        raise EvalScenarioConflict(
                            f"Eval scenario revision {scenario.revision} has conflicting content."
                        )
                    entry = self._load_scenario_entry(connection, scenario.revision)
                    assert entry is not None
                    connection.commit()
                    return entry
                connection.execute(
                    """
                    INSERT INTO cayu_eval_scenarios (
                        revision, scenario_id, target_key, name, description,
                        event_count, input_event_count, approval_checkpoint_count,
                        message_count, part_count, artifact_requirement_count,
                        secret_requirement_count, document_json, document_bytes, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        _format_datetime(datetime.now(UTC)),
                    ),
                )
                entry = self._load_scenario_entry(connection, scenario.revision)
                assert entry is not None
                connection.commit()
                return entry
            except BaseException:
                connection.rollback()
                raise

        return await self._run(operation)

    async def load_scenario(
        self,
        revision: str,
        *,
        max_bytes: int = EVAL_SCENARIO_MAX_BYTES,
    ) -> EvalScenarioDocumentV2 | None:
        revision = _sha256_revision(revision, "revision")
        max_bytes = _read_limit(max_bytes, hard_max=EVAL_SCENARIO_MAX_BYTES)

        def operation(connection: sqlite3.Connection) -> str | None:
            row = connection.execute(
                """
                SELECT document_json, document_bytes
                FROM cayu_eval_scenarios
                WHERE revision = ?
                """,
                (revision,),
            ).fetchone()
            if row is None:
                return None
            if row["document_bytes"] > max_bytes:
                raise EvalStoreResultTooLarge(max_bytes)
            return row["document_json"]

        document = await self._run(operation)
        if document is None:
            return None
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

        def operation(connection: sqlite3.Connection) -> EvalScenarioCatalogPage:
            clauses: list[str] = []
            params: list[object] = []
            if query.target_key is not None:
                clauses.append("target_key = ?")
                params.append(query.target_key)
            if query.scenario_id is not None:
                clauses.append("scenario_id = ?")
                params.append(query.scenario_id)
            if boundary is not None:
                clauses.append("(created_at < ? OR (created_at = ? AND revision > ?))")
                timestamp = _format_datetime(boundary[0])
                params.extend((timestamp, timestamp, boundary[1]))
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            rows = connection.execute(
                f"""
                SELECT revision, scenario_id, target_key, name, description,
                       event_count, input_event_count, approval_checkpoint_count,
                       message_count, part_count, artifact_requirement_count,
                       secret_requirement_count, document_bytes, created_at
                FROM cayu_eval_scenarios
                {where}
                ORDER BY created_at DESC, revision ASC
                LIMIT ?
                """,
                (*params, query.limit + 1),
            ).fetchall()
            return _bounded_scenario_page(
                [self._scenario_entry_from_row(row) for row in rows],
                query,
            )

        return await self._run(operation)

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

        def operation(connection: sqlite3.Connection) -> EvalAuthoredSuiteCatalogEntry:
            try:
                connection.execute("BEGIN IMMEDIATE")
                scenario_cases = authored_suite_scenario_cases(validated)
                scenario_revisions = tuple(
                    sorted({reference.scenario_revision for _, reference in scenario_cases})
                )
                scenario_by_revision: dict[str, EvalScenarioDocumentV2] = {}
                for start in range(0, len(scenario_revisions), 500):
                    chunk = scenario_revisions[start : start + 500]
                    placeholders = ",".join("?" for _ in chunk)
                    rows = connection.execute(
                        "SELECT revision, document_json FROM cayu_eval_scenarios "
                        f"WHERE revision IN ({placeholders})",
                        chunk,
                    ).fetchall()
                    scenario_by_revision.update(
                        {
                            row["revision"]: eval_scenario_from_json(row["document_json"])
                            for row in rows
                        }
                    )
                for case, reference in scenario_cases:
                    validate_authored_suite_scenario(
                        validated,
                        case,
                        scenario_by_revision.get(reference.scenario_revision),
                    )
                existing = connection.execute(
                    "SELECT document_json FROM cayu_eval_authored_suites WHERE revision = ?",
                    (validated.revision,),
                ).fetchone()
                if existing is not None:
                    if existing["document_json"] != document_text:
                        raise EvalAuthoredSuiteConflict(
                            f"Authored eval suite revision {validated.revision} has "
                            "conflicting content."
                        )
                    entry = self._load_authored_suite_entry(
                        connection,
                        validated.revision,
                    )
                    assert entry is not None
                    connection.commit()
                    return entry
                entry = authored_suite_catalog_entry(
                    validated,
                    created_at=datetime.now(UTC),
                    document_bytes=len(payload),
                )
                connection.execute(
                    """
                    INSERT INTO cayu_eval_authored_suites (
                        revision, suite_id, suite_revision, target_key, name,
                        description, case_count, assertion_count,
                        simple_input_count, scenario_count, trials,
                        timeout_seconds, document_json, document_bytes, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        _format_datetime(entry.created_at),
                    ),
                )
                connection.commit()
                return entry
            except BaseException:
                connection.rollback()
                raise

        return await self._run(operation)

    async def load_authored_suite(
        self,
        revision: str,
        *,
        max_bytes: int = EVAL_SUITE_AUTHORING_MAX_BYTES,
    ) -> EvalSuiteDocument | None:
        revision = _sha256_revision(revision, "revision")
        max_bytes = _read_limit(max_bytes, hard_max=EVAL_SUITE_AUTHORING_MAX_BYTES)

        def operation(connection: sqlite3.Connection) -> str | None:
            row = connection.execute(
                """
                SELECT document_json, document_bytes
                FROM cayu_eval_authored_suites
                WHERE revision = ?
                """,
                (revision,),
            ).fetchone()
            if row is None:
                return None
            if row["document_bytes"] > max_bytes:
                raise EvalStoreResultTooLarge(max_bytes)
            return row["document_json"]

        payload = await self._run(operation)
        if payload is None:
            return None
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

        def operation(connection: sqlite3.Connection) -> EvalAuthoredSuiteCatalogPage:
            clauses: list[str] = []
            params: list[object] = []
            if query.target_key is not None:
                clauses.append("target_key = ?")
                params.append(query.target_key)
            if query.suite_id is not None:
                clauses.append("suite_id = ?")
                params.append(query.suite_id)
            if boundary is not None:
                clauses.append("(created_at < ? OR (created_at = ? AND revision > ?))")
                timestamp = _format_datetime(boundary[0])
                params.extend((timestamp, timestamp, boundary[1]))
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            rows = connection.execute(
                f"""
                SELECT revision, suite_id, suite_revision, target_key, name,
                       description, case_count, assertion_count, simple_input_count,
                       scenario_count, trials, timeout_seconds, document_bytes, created_at
                FROM cayu_eval_authored_suites
                {where}
                ORDER BY created_at DESC, revision ASC
                LIMIT ?
                """,
                (*params, query.limit + 1),
            ).fetchall()
            return _bounded_authored_suite_page(
                [self._authored_suite_entry_from_row(row) for row in rows],
                query,
            )

        return await self._run(operation)

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

        def operation(connection: sqlite3.Connection) -> str:
            try:
                connection.execute("BEGIN IMMEDIATE")
                rows = connection.execute(
                    """
                    SELECT revision, run_id, report_json
                    FROM cayu_eval_judge_calibrations
                    WHERE revision = ? OR run_id = ?
                    """,
                    (validated.revision, validated.run_id),
                ).fetchall()
                if rows:
                    if len(rows) != 1 or (
                        rows[0]["revision"],
                        rows[0]["run_id"],
                        rows[0]["report_json"],
                    ) != (validated.revision, validated.run_id, document_text):
                        raise EvalJudgeCalibrationConflict(
                            "Judge calibration revision or run ID has conflicting content."
                        )
                    connection.commit()
                    return rows[0]["report_json"]
                connection.execute(
                    """
                    INSERT INTO cayu_eval_judge_calibrations (
                        revision, run_id, definition_revision, target_key,
                        trial_count, report_json, document_bytes, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        validated.revision,
                        validated.run_id,
                        validated.definition.revision,
                        validated.definition.target_key,
                        len(validated.trials),
                        document_text,
                        len(payload),
                        _format_datetime(datetime.now(UTC)),
                    ),
                )
                connection.commit()
                return document_text
            except BaseException:
                connection.rollback()
                raise

        stored = await self._run(operation)
        return await asyncio.to_thread(eval_judge_calibration_report_from_json, stored)

    async def load_judge_calibration(
        self,
        revision: str,
        *,
        max_bytes: int = EVAL_JUDGE_CALIBRATION_MAX_BYTES,
    ) -> EvalJudgeCalibrationReportV1 | None:
        revision = _sha256_revision(revision, "revision")
        max_bytes = _read_limit(max_bytes, hard_max=EVAL_JUDGE_CALIBRATION_MAX_BYTES)

        def operation(connection: sqlite3.Connection) -> str | None:
            row = connection.execute(
                """
                SELECT report_json, document_bytes
                FROM cayu_eval_judge_calibrations
                WHERE revision = ?
                """,
                (revision,),
            ).fetchone()
            if row is None:
                return None
            if row["document_bytes"] > max_bytes:
                raise EvalStoreResultTooLarge(max_bytes)
            return row["report_json"]

        stored = await self._run(operation)
        if stored is None:
            return None
        return await asyncio.to_thread(eval_judge_calibration_report_from_json, stored)

    async def load_judge_calibration_by_run_id(
        self,
        run_id: str,
        *,
        max_bytes: int = EVAL_JUDGE_CALIBRATION_MAX_BYTES,
    ) -> EvalJudgeCalibrationReportV1 | None:
        run_id = _portable_id(run_id, "run_id")
        max_bytes = _read_limit(max_bytes, hard_max=EVAL_JUDGE_CALIBRATION_MAX_BYTES)

        def operation(connection: sqlite3.Connection) -> str | None:
            row = connection.execute(
                """
                SELECT report_json, document_bytes
                FROM cayu_eval_judge_calibrations
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
            if row is None:
                return None
            if row["document_bytes"] > max_bytes:
                raise EvalStoreResultTooLarge(max_bytes)
            return row["report_json"]

        stored = await self._run(operation)
        if stored is None:
            return None
        return await asyncio.to_thread(eval_judge_calibration_report_from_json, stored)

    async def list_suites(self, query: EvalSuiteCatalogQuery) -> EvalSuiteCatalogPage:
        query = _exact_model(query, EvalSuiteCatalogQuery, "query")
        boundary = (
            decode_suite_cursor(query.cursor, query.corpus_revision)
            if query.cursor is not None
            else None
        )

        def operation(connection: sqlite3.Connection) -> EvalSuiteCatalogPage:
            if not self._corpus_exists(connection, query.corpus_revision):
                raise KeyError(f"Eval corpus not found: {query.corpus_revision}")
            rows = connection.execute(
                """
                SELECT corpus_revision, suite_id, suite_revision, name, description,
                       case_count, assertion_count, trials, timeout_seconds
                FROM cayu_eval_suites
                WHERE corpus_revision = ? AND suite_id > ?
                ORDER BY suite_id ASC
                LIMIT ?
                """,
                (query.corpus_revision, boundary or "", query.limit + 1),
            ).fetchall()
            return _bounded_suite_page(
                [self._suite_entry_from_row(row) for row in rows],
                query,
            )

        return await self._run(operation)

    async def list_cases(self, query: EvalCaseCatalogQuery) -> EvalCaseCatalogPage:
        query = _copy_query(query, EvalCaseCatalogQuery)
        boundary = (
            decode_case_cursor(query.cursor, query.corpus_revision, query.suite_id)
            if query.cursor is not None
            else None
        )

        def operation(connection: sqlite3.Connection) -> EvalCaseCatalogPage:
            suite = connection.execute(
                """
                SELECT 1 FROM cayu_eval_suites
                WHERE corpus_revision = ? AND suite_id = ?
                """,
                (query.corpus_revision, query.suite_id),
            ).fetchone()
            if suite is None:
                if not self._corpus_exists(connection, query.corpus_revision):
                    raise KeyError(f"Eval corpus not found: {query.corpus_revision}")
                raise KeyError(f"Eval suite not found: {query.suite_id}")
            rows = connection.execute(
                """
                SELECT corpus_revision, case_id, case_revision, suite_id, name,
                       description, message_count, assertion_count
                FROM cayu_eval_cases
                WHERE corpus_revision = ? AND suite_id = ? AND case_id > ?
                ORDER BY case_id ASC
                LIMIT ?
                """,
                (query.corpus_revision, query.suite_id, boundary or "", query.limit + 1),
            ).fetchall()
            return _bounded_case_page(
                [self._case_entry_from_row(row) for row in rows],
                query,
            )

        return await self._run(operation)

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

        def operation(connection: sqlite3.Connection) -> EvalRunRecord:
            try:
                connection.execute("BEGIN IMMEDIATE")
                now = datetime.now(UTC)
                duplicate = connection.execute(
                    f"SELECT {_RUN_COLUMNS} FROM cayu_eval_runs WHERE idempotency_key = ?",
                    (request.idempotency_key,),
                ).fetchone()
                if duplicate is not None:
                    record = _run_record_from_row(duplicate)
                    if not _request_from_row(duplicate).same_logical_request(request):
                        raise EvalRunAdmissionConflict(
                            "Eval run idempotency key is already bound to another request."
                        )
                    connection.commit()
                    return record
                if self._load_run_row(connection, request.run_id) is not None:
                    raise EvalRunAdmissionConflict(
                        f"Eval run id is already bound to another request: {request.run_id}"
                    )
                contract = connection.execute(
                    """
                    SELECT corpus.target_key, suite.suite_revision
                    FROM cayu_eval_corpora AS corpus
                    JOIN cayu_eval_suites AS suite
                      ON suite.corpus_revision = corpus.revision
                    WHERE corpus.revision = ? AND suite.suite_id = ?
                    """,
                    (request.corpus_revision, request.suite_id),
                ).fetchone()
                if contract is None:
                    if not self._corpus_exists(connection, request.corpus_revision):
                        raise KeyError(f"Eval corpus not found: {request.corpus_revision}")
                    raise EvalRunAdmissionConflict(f"Eval suite not found: {request.suite_id}")
                if contract["target_key"] != request.target_key:
                    raise EvalRunAdmissionConflict("Eval run target key does not match its corpus.")
                if contract["suite_revision"] != request.suite_revision:
                    raise EvalRunAdmissionConflict(
                        "Eval run suite revision does not match its corpus."
                    )
                timestamp = _format_datetime(now)
                connection.execute(
                    """
                    INSERT INTO cayu_eval_runs (
                        run_id, idempotency_key, corpus_revision, target_key,
                        suite_id, suite_revision, max_concurrency, invocation_json, status,
                        created_at, updated_at, authored_suite_launch_revision,
                        authored_suite_launch_lane
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        timestamp,
                        timestamp,
                        request.invocation.authored_suite_launch_revision,
                        request.invocation.authored_suite_launch_lane,
                    ),
                )
                row = self._load_run_row(connection, request.run_id)
                connection.commit()
                assert row is not None
                return _run_record_from_row(row)
            except BaseException:
                connection.rollback()
                raise

        return await self._run(operation)

    async def load_run_by_idempotency_key(
        self,
        idempotency_key: str,
    ) -> EvalRunRecord | None:
        idempotency_key = _idempotency_key(idempotency_key, "idempotency_key")

        def operation(connection: sqlite3.Connection) -> EvalRunRecord | None:
            row = connection.execute(
                f"SELECT {_RUN_COLUMNS} FROM cayu_eval_runs WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            return None if row is None else _run_record_from_row(row)

        return await self._run(operation)

    async def load_run(self, run_id: str) -> EvalRunRecord | None:
        run_id = _store_identifier(run_id, "run_id")
        started_at = monotonic()

        def operation(connection: sqlite3.Connection) -> EvalRunRecord | None:
            row = self._load_run_row(connection, run_id)
            return None if row is None else _run_record_from_row(row)

        try:
            return await self._run(operation)
        finally:
            logger.debug(
                "SQLite eval run fully rehydrated.",
                extra={
                    "cayu_eval_store_event": "full_run_rehydration",
                    "eval_store_kind": "sqlite",
                    "eval_run_id": run_id,
                    "duration_seconds": monotonic() - started_at,
                },
            )

    async def load_run_observation(self, run_id: str) -> EvalRunObservation | None:
        run_id = _store_identifier(run_id, "run_id")
        started_at = monotonic()

        def operation(connection: sqlite3.Connection) -> EvalRunObservation | None:
            row = connection.execute(
                f"SELECT {_RUN_OBSERVATION_COLUMNS} FROM cayu_eval_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            return None if row is None else _run_observation_from_row(row)

        try:
            return await self._run_read(operation)
        finally:
            logger.debug(
                "SQLite eval run status observed.",
                extra={
                    "cayu_eval_store_event": "run_status_read",
                    "eval_store_kind": "sqlite",
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

        def operation(connection: sqlite3.Connection) -> EvalRunPage:
            clauses: list[str] = []
            params: list[object] = []
            if query.target_key is not None:
                clauses.append("target_key = ?")
                params.append(query.target_key)
            if query.status is not None:
                clauses.append("status = ?")
                params.append(str(query.status))
            if query.corpus_revision is not None:
                clauses.append("corpus_revision = ?")
                params.append(query.corpus_revision)
            if boundary is not None:
                clauses.append("(created_at < ? OR (created_at = ? AND run_id > ?))")
                timestamp = _format_datetime(boundary[0])
                params.extend((timestamp, timestamp, boundary[1]))
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            rows = connection.execute(
                f"""
                SELECT {_RUN_COLUMNS}
                FROM cayu_eval_runs
                {where}
                ORDER BY created_at DESC, run_id ASC
                LIMIT ?
                """,
                (*params, query.limit + 1),
            ).fetchall()
            return _bounded_run_page([_run_record_from_row(row) for row in rows], query)

        return await self._run(operation)

    async def claim_run(
        self,
        *,
        target_key: str | None = None,
        lease_seconds: int = 300,
    ) -> EvalRunLease | None:
        if target_key is not None:
            target_key = _portable_id(target_key, "target_key")
        lease_seconds = _lease_seconds(lease_seconds)

        def operation(connection: sqlite3.Connection) -> EvalRunLease | None:
            try:
                connection.execute("BEGIN IMMEDIATE")
                now = datetime.now(UTC)
                claim_id = str(uuid4())
                target_clause = "" if target_key is None else "AND candidate.target_key = ?"
                target_params: tuple[str, ...] = () if target_key is None else (target_key,)
                row = connection.execute(
                    f"""
                    SELECT {_RUN_COLUMNS}
                    FROM cayu_eval_runs AS candidate
                    WHERE candidate.ownership_epoch < 9223372036854775807
                      {target_clause}
                      AND (candidate.status = ?
                       OR (candidate.status IN (?, ?) AND candidate.lease_expires_at <= ?))
                      AND (
                          candidate.authored_suite_launch_revision IS NULL
                          OR NOT EXISTS (
                              SELECT 1
                              FROM cayu_eval_runs AS predecessor
                              WHERE predecessor.authored_suite_launch_revision =
                                    candidate.authored_suite_launch_revision
                                AND predecessor.authored_suite_launch_lane =
                                    candidate.authored_suite_launch_lane
                                AND predecessor.status NOT IN (?, ?, ?)
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
                    LIMIT 1
                    """,
                    (
                        *target_params,
                        str(EvalRunStatus.QUEUED),
                        str(EvalRunStatus.RUNNING),
                        str(EvalRunStatus.CANCELLING),
                        _format_datetime(now),
                        str(EvalRunStatus.COMPLETED),
                        str(EvalRunStatus.FAILED),
                        str(EvalRunStatus.CANCELLED),
                    ),
                ).fetchone()
                if row is None:
                    connection.commit()
                    return None
                status = (
                    EvalRunStatus.CANCELLING
                    if row["cancel_requested_at"] is not None
                    else EvalRunStatus.RUNNING
                )
                next_epoch = row["ownership_epoch"] + 1
                progress = _scenario_progress_for_claim(
                    (
                        None
                        if row["scenario_progress_json"] is None
                        else EvalScenarioRunProgress.model_validate_json(
                            row["scenario_progress_json"]
                        )
                    ),
                    scenario=_request_from_row(row).invocation.scenario,
                    attempt=next_epoch,
                    terminal_trial_numbers=frozenset(
                        item.trial_number for item in _load_trial_checkpoints(connection, row)
                    ),
                )
                connection.execute(
                    """
                    UPDATE cayu_eval_runs
                    SET status = ?, updated_at = ?,
                        started_at = COALESCE(started_at, ?),
                        claim_id = ?,
                        ownership_epoch = ownership_epoch + 1,
                        lease_expires_at = ?,
                        scenario_progress_json = ?
                    WHERE run_id = ?
                    """,
                    (
                        str(status),
                        _format_datetime(now),
                        _format_datetime(now),
                        claim_id,
                        _format_datetime(now + timedelta(seconds=lease_seconds)),
                        None if progress is None else progress.model_dump_json(),
                        row["run_id"],
                    ),
                )
                claimed = self._load_run_row(connection, row["run_id"])
                connection.commit()
                assert claimed is not None
                record = _run_record_from_row(claimed)
                return EvalRunLease(
                    run=record,
                    claim=EvalRunClaim(
                        run_id=record.id,
                        claim_id=claimed["claim_id"],
                        epoch=claimed["ownership_epoch"],
                    ),
                )
            except BaseException:
                connection.rollback()
                raise

        return await self._run(operation)

    async def claim_run_for_targets(
        self,
        target_keys: tuple[str, ...],
        *,
        lease_seconds: int = 300,
    ) -> EvalRunLease | None:
        target_keys = _claim_target_keys(target_keys)
        lease_seconds = _lease_seconds(lease_seconds)

        def operation(connection: sqlite3.Connection) -> EvalRunLease | None:
            try:
                connection.execute("BEGIN IMMEDIATE")
                now = datetime.now(UTC)
                claim_id = str(uuid4())
                placeholders = ", ".join("?" for _ in target_keys)
                row = connection.execute(
                    f"""
                    SELECT {_RUN_COLUMNS}
                    FROM cayu_eval_runs AS candidate
                    WHERE candidate.ownership_epoch < 9223372036854775807
                      AND candidate.target_key IN ({placeholders})
                      AND (candidate.status = ?
                       OR (candidate.status IN (?, ?) AND candidate.lease_expires_at <= ?))
                      AND (
                          candidate.authored_suite_launch_revision IS NULL
                          OR NOT EXISTS (
                              SELECT 1
                              FROM cayu_eval_runs AS predecessor
                              WHERE predecessor.authored_suite_launch_revision =
                                    candidate.authored_suite_launch_revision
                                AND predecessor.authored_suite_launch_lane =
                                    candidate.authored_suite_launch_lane
                                AND predecessor.status NOT IN (?, ?, ?)
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
                    LIMIT 1
                    """,
                    (
                        *target_keys,
                        str(EvalRunStatus.QUEUED),
                        str(EvalRunStatus.RUNNING),
                        str(EvalRunStatus.CANCELLING),
                        _format_datetime(now),
                        str(EvalRunStatus.COMPLETED),
                        str(EvalRunStatus.FAILED),
                        str(EvalRunStatus.CANCELLED),
                    ),
                ).fetchone()
                if row is None:
                    connection.commit()
                    return None
                status = (
                    EvalRunStatus.CANCELLING
                    if row["cancel_requested_at"] is not None
                    else EvalRunStatus.RUNNING
                )
                next_epoch = row["ownership_epoch"] + 1
                progress = _scenario_progress_for_claim(
                    (
                        None
                        if row["scenario_progress_json"] is None
                        else EvalScenarioRunProgress.model_validate_json(
                            row["scenario_progress_json"]
                        )
                    ),
                    scenario=_request_from_row(row).invocation.scenario,
                    attempt=next_epoch,
                    terminal_trial_numbers=frozenset(
                        item.trial_number for item in _load_trial_checkpoints(connection, row)
                    ),
                )
                connection.execute(
                    """
                    UPDATE cayu_eval_runs
                    SET status = ?, updated_at = ?,
                        started_at = COALESCE(started_at, ?),
                        claim_id = ?,
                        ownership_epoch = ownership_epoch + 1,
                        lease_expires_at = ?,
                        scenario_progress_json = ?
                    WHERE run_id = ?
                    """,
                    (
                        str(status),
                        _format_datetime(now),
                        _format_datetime(now),
                        claim_id,
                        _format_datetime(now + timedelta(seconds=lease_seconds)),
                        None if progress is None else progress.model_dump_json(),
                        row["run_id"],
                    ),
                )
                claimed = self._load_run_row(connection, row["run_id"])
                connection.commit()
                assert claimed is not None
                record = _run_record_from_row(claimed)
                return EvalRunLease(
                    run=record,
                    claim=EvalRunClaim(
                        run_id=record.id,
                        claim_id=claimed["claim_id"],
                        epoch=claimed["ownership_epoch"],
                    ),
                )
            except BaseException:
                connection.rollback()
                raise

        return await self._run(operation)

    async def heartbeat_run(
        self,
        claim: EvalRunClaim,
        *,
        extend_seconds: int = 300,
    ) -> EvalRunRecord:
        claim = _exact_model(claim, EvalRunClaim, "claim")
        extend_seconds = _lease_seconds(extend_seconds)

        def operation(connection: sqlite3.Connection) -> EvalRunRecord:
            try:
                connection.execute("BEGIN IMMEDIATE")
                now = datetime.now(UTC)
                updated = connection.execute(
                    """
                    UPDATE cayu_eval_runs
                    SET lease_expires_at = ?, updated_at = ?
                    WHERE run_id = ? AND claim_id = ? AND ownership_epoch = ?
                      AND status IN (?, ?) AND lease_expires_at > ?
                    """,
                    (
                        _format_datetime(now + timedelta(seconds=extend_seconds)),
                        _format_datetime(now),
                        claim.run_id,
                        claim.claim_id,
                        claim.epoch,
                        str(EvalRunStatus.RUNNING),
                        str(EvalRunStatus.CANCELLING),
                        _format_datetime(now),
                    ),
                )
                if updated.rowcount != 1:
                    raise EvalRunClaimLost("Eval run claim is no longer live.")
                row = self._load_run_row(connection, claim.run_id)
                connection.commit()
                assert row is not None
                return _run_record_from_row(row)
            except BaseException:
                connection.rollback()
                raise

        return await self._run_eval_writer(
            operation,
            operation_name="heartbeat_run",
            run_id=claim.run_id,
        )

    async def heartbeat_run_observation(
        self,
        claim: EvalRunClaim,
        *,
        extend_seconds: int = 300,
    ) -> EvalRunObservation:
        claim = _exact_model(claim, EvalRunClaim, "claim")
        extend_seconds = _lease_seconds(extend_seconds)

        def operation(connection: sqlite3.Connection) -> EvalRunObservation:
            try:
                connection.execute("BEGIN IMMEDIATE")
                now = datetime.now(UTC)
                updated = connection.execute(
                    """
                    UPDATE cayu_eval_runs
                    SET lease_expires_at = ?, updated_at = ?
                    WHERE run_id = ? AND claim_id = ? AND ownership_epoch = ?
                      AND status IN (?, ?) AND lease_expires_at > ?
                    """,
                    (
                        _format_datetime(now + timedelta(seconds=extend_seconds)),
                        _format_datetime(now),
                        claim.run_id,
                        claim.claim_id,
                        claim.epoch,
                        str(EvalRunStatus.RUNNING),
                        str(EvalRunStatus.CANCELLING),
                        _format_datetime(now),
                    ),
                )
                if updated.rowcount != 1:
                    raise EvalRunClaimLost("Eval run claim is no longer live.")
                row = connection.execute(
                    f"SELECT {_RUN_OBSERVATION_COLUMNS} FROM cayu_eval_runs WHERE run_id = ?",
                    (claim.run_id,),
                ).fetchone()
                connection.commit()
                assert row is not None
                return _run_observation_from_row(row)
            except BaseException:
                connection.rollback()
                raise

        return await self._run_eval_writer(
            operation,
            operation_name="heartbeat_run",
            run_id=claim.run_id,
        )

    async def request_cancel(self, run_id: str) -> EvalRunRecord:
        run_id = _store_identifier(run_id, "run_id")

        def operation(connection: sqlite3.Connection) -> EvalRunRecord:
            try:
                connection.execute("BEGIN IMMEDIATE")
                now = datetime.now(UTC)
                row = self._require_run_row(connection, run_id)
                status = EvalRunStatus(row["status"])
                if status in TERMINAL_EVAL_RUN_STATUSES:
                    connection.commit()
                    return _run_record_from_row(row)
                lease_expires_at = _parse_optional_datetime(row["lease_expires_at"])
                claim_expired = (
                    status in {EvalRunStatus.RUNNING, EvalRunStatus.CANCELLING}
                    and lease_expires_at is not None
                    and lease_expires_at <= now
                )
                next_status = (
                    EvalRunStatus.CANCELLED
                    if status is EvalRunStatus.QUEUED or claim_expired
                    else EvalRunStatus.CANCELLING
                )
                connection.execute(
                    """
                    UPDATE cayu_eval_runs
                    SET status = ?, updated_at = ?,
                        cancel_requested_at = COALESCE(cancel_requested_at, ?),
                        finished_at = CASE WHEN ? = ? THEN ? ELSE finished_at END,
                        claim_id = CASE WHEN ? = ? THEN NULL ELSE claim_id END,
                        lease_expires_at = CASE WHEN ? = ? THEN NULL ELSE lease_expires_at END,
                        trial_checkpoint_count = CASE WHEN ? = ? THEN 0
                            ELSE trial_checkpoint_count END,
                        trial_checkpoint_bytes = CASE WHEN ? = ? THEN 0
                            ELSE trial_checkpoint_bytes END
                    WHERE run_id = ?
                    """,
                    (
                        str(next_status),
                        _format_datetime(now),
                        _format_datetime(now),
                        str(next_status),
                        str(EvalRunStatus.CANCELLED),
                        _format_datetime(now),
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
                    _delete_trial_checkpoints(connection, run_id)
                updated = self._require_run_row(connection, run_id)
                connection.commit()
                return _run_record_from_row(updated)
            except BaseException:
                connection.rollback()
                raise

        return await self._run(operation)

    async def initialize_scenario_progress(
        self,
        claim: EvalRunClaim,
        progress: EvalScenarioRunProgress,
    ) -> EvalRunRecord:
        claim = _exact_model(claim, EvalRunClaim, "claim")
        progress = _exact_model(progress, EvalScenarioRunProgress, "progress")

        def operation(connection: sqlite3.Connection) -> EvalRunRecord:
            try:
                connection.execute("BEGIN IMMEDIATE")
                now = datetime.now(UTC)
                row = self._require_run_row(connection, claim.run_id)
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
                    raise EvalRunStateConflict("Scenario progress does not match the claimed run.")
                connection.execute(
                    """
                    UPDATE cayu_eval_runs
                    SET scenario_progress_json = ?, updated_at = ?
                    WHERE run_id = ?
                    """,
                    (
                        progress.model_dump_json(),
                        _format_datetime(now),
                        claim.run_id,
                    ),
                )
                updated = self._require_run_row(connection, claim.run_id)
                connection.commit()
                return _run_record_from_row(updated)
            except BaseException:
                connection.rollback()
                raise

        return await self._run(operation)

    async def update_scenario_trial(
        self,
        claim: EvalRunClaim,
        trial: EvalScenarioTrialProgress,
    ) -> EvalRunRecord:
        claim = _exact_model(claim, EvalRunClaim, "claim")
        trial = _exact_model(trial, EvalScenarioTrialProgress, "trial")

        def operation(connection: sqlite3.Connection) -> EvalRunRecord:
            try:
                connection.execute("BEGIN IMMEDIATE")
                now = datetime.now(UTC)
                row = self._require_run_row(connection, claim.run_id)
                self._require_live_claim(row, claim, now)
                raw_progress = row["scenario_progress_json"]
                if raw_progress is None:
                    raise EvalRunStateConflict("Scenario progress is absent for this claim.")
                progress = EvalScenarioRunProgress.model_validate_json(raw_progress)
                if progress.attempt != claim.epoch:
                    raise EvalRunStateConflict("Scenario progress belongs to another claim.")
                updated_progress = progress.replace_trial(trial)
                connection.execute(
                    """
                    UPDATE cayu_eval_runs
                    SET scenario_progress_json = ?, updated_at = ?
                    WHERE run_id = ?
                    """,
                    (
                        updated_progress.model_dump_json(),
                        _format_datetime(now),
                        claim.run_id,
                    ),
                )
                updated = self._require_run_row(connection, claim.run_id)
                connection.commit()
                return _run_record_from_row(updated)
            except BaseException:
                connection.rollback()
                raise

        return await self._run(operation)

    async def load_trial_checkpoints(
        self,
        claim: EvalRunClaim,
    ) -> tuple[EvalRunTrialCheckpoint, ...]:
        claim = _exact_model(claim, EvalRunClaim, "claim")

        def operation(connection: sqlite3.Connection) -> tuple[EvalRunTrialCheckpoint, ...]:
            try:
                connection.execute("BEGIN")
                now = datetime.now(UTC)
                row = self._require_run_row(connection, claim.run_id)
                self._require_live_claim(row, claim, now)
                checkpoints = _load_trial_checkpoints(connection, row)
                connection.commit()
                return checkpoints
            except BaseException:
                connection.rollback()
                raise

        return await self._run(operation)

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

        def operation(connection: sqlite3.Connection) -> None:
            try:
                connection.execute("BEGIN IMMEDIATE")
                now = datetime.now(UTC)
                row = self._require_run_row(connection, claim.run_id)
                self._require_live_claim(row, claim, now)
                request = _request_from_row(row)
                suite_row = connection.execute(
                    """
                    SELECT suites.trials
                    FROM cayu_eval_suites AS suites
                    JOIN cayu_eval_cases AS cases
                      ON cases.corpus_revision = suites.corpus_revision
                     AND cases.suite_id = suites.suite_id
                    WHERE suites.corpus_revision = ? AND suites.suite_id = ?
                      AND cases.case_id = ?
                    """,
                    (
                        request.corpus_revision,
                        request.suite_id,
                        prepared.checkpoint.case_id,
                    ),
                ).fetchone()
                if suite_row is None:
                    raise EvalRunStateConflict(
                        "Eval trial checkpoint case does not belong to its run."
                    )
                validated = prepared.checkpoint
                if validated.trial_number > suite_row["trials"]:
                    raise EvalRunStateConflict(
                        "Eval trial checkpoint number exceeds its immutable policy."
                    )
                key = (validated.case_id, validated.trial_number)
                existing = connection.execute(
                    """
                    SELECT checkpoint_json, document_bytes
                    FROM cayu_eval_run_trial_checkpoints
                    WHERE run_id = ? AND case_id = ? AND trial_number = ?
                    """,
                    (claim.run_id, *key),
                ).fetchone()
                if existing is not None:
                    if existing["document_bytes"] != len(
                        existing["checkpoint_json"].encode("utf-8")
                    ):
                        raise RuntimeError(
                            "Stored eval trial checkpoint byte accounting is inconsistent."
                        )
                    current = eval_run_trial_checkpoint_from_json(existing["checkpoint_json"])
                    if current == validated:
                        connection.commit()
                        return
                    raise EvalRunStateConflict(
                        "Eval trial slot already has another terminal result."
                    )
                if row["trial_checkpoint_count"] >= EVAL_RUN_TRIAL_CHECKPOINTS_MAX_ITEMS:
                    raise ValueError("Eval run trial checkpoints exceed their item limit.")
                if (
                    row["trial_checkpoint_bytes"] + prepared.document_bytes
                    > EVAL_RUN_TRIAL_CHECKPOINTS_MAX_BYTES
                ):
                    raise ValueError("Eval run trial checkpoints exceed their byte limit.")
                connection.execute(
                    """
                    INSERT INTO cayu_eval_run_trial_checkpoints (
                        run_id, case_id, trial_number, checkpoint_json, document_bytes
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        claim.run_id,
                        *key,
                        prepared.document,
                        prepared.document_bytes,
                    ),
                )
                connection.execute(
                    """
                    UPDATE cayu_eval_runs
                    SET trial_checkpoint_count = trial_checkpoint_count + 1,
                        trial_checkpoint_bytes = trial_checkpoint_bytes + ?,
                        updated_at = ?
                    WHERE run_id = ?
                    """,
                    (prepared.document_bytes, _format_datetime(now), claim.run_id),
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

        await self._run_eval_writer(
            operation,
            operation_name="save_trial_checkpoint",
            run_id=claim.run_id,
        )

    async def submit_scenario_approval(
        self,
        run_id: str,
        submission: EvalScenarioApprovalSubmission,
    ) -> EvalRunRecord:
        run_id = _store_identifier(run_id, "run_id")
        submission = _exact_model(submission, EvalScenarioApprovalSubmission, "submission")

        def operation(connection: sqlite3.Connection) -> EvalRunRecord:
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = self._require_run_row(connection, run_id)
                if EvalRunStatus(row["status"]) is not EvalRunStatus.RUNNING:
                    raise EvalRunStateConflict("Scenario approval requires an active run.")
                raw_progress = row["scenario_progress_json"]
                if raw_progress is None:
                    raise EvalRunStateConflict("Scenario progress is unavailable.")
                progress = EvalScenarioRunProgress.model_validate_json(raw_progress)
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
                    raise EvalRunStateConflict("Scenario approval checkpoint is no longer pending.")
                now = datetime.now(UTC)
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
                connection.execute(
                    """
                    UPDATE cayu_eval_runs
                    SET scenario_progress_json = ?, updated_at = ?
                    WHERE run_id = ?
                    """,
                    (
                        updated_progress.model_dump_json(),
                        _format_datetime(now),
                        run_id,
                    ),
                )
                updated = self._require_run_row(connection, run_id)
                connection.commit()
                return _run_record_from_row(updated)
            except BaseException:
                connection.rollback()
                raise

        return await self._run(operation)

    async def publish_result(
        self,
        claim: EvalRunClaim,
        result: CorpusExecutionResult,
        *,
        redact_json: Callable[[Any], Any],
    ) -> EvalRunRecord:
        claim = _exact_model(claim, EvalRunClaim, "claim")
        result, document = await asyncio.to_thread(
            _prepare_result_for_store,
            result,
            redact_json=redact_json,
        )
        document_text = document.decode("utf-8")
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

        def operation(connection: sqlite3.Connection) -> EvalRunRecord:
            try:
                connection.execute("BEGIN IMMEDIATE")
                now = datetime.now(UTC)
                row = self._require_run_row(connection, claim.run_id)
                if _request_from_row(row) != request:
                    raise EvalRunStateConflict(
                        "Eval run request changed during result publication."
                    )
                status = EvalRunStatus(row["status"])
                if status is EvalRunStatus.COMPLETED:
                    existing = connection.execute(
                        "SELECT revision, result_json FROM cayu_eval_results WHERE run_id = ?",
                        (claim.run_id,),
                    ).fetchone()
                    if (
                        self._claim_matches(row, claim)
                        and existing is not None
                        and existing["revision"] == validated.revision
                        and existing["result_json"] == document_text
                    ):
                        connection.commit()
                        return _run_record_from_row(row)
                    raise EvalRunStateConflict("Eval run already has another terminal result.")
                self._require_live_claim(row, claim, now)
                if status is not EvalRunStatus.RUNNING:
                    raise EvalRunStateConflict("Only a running eval may publish a result.")
                _validate_trial_checkpoints_for_result(
                    _load_trial_checkpoints(connection, row),
                    validated,
                )
                summary = result_summary(validated)
                connection.execute(
                    """
                    INSERT INTO cayu_eval_results (
                        run_id, revision, result_json, result_bytes, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        claim.run_id,
                        validated.revision,
                        document_text,
                        len(document),
                        _format_datetime(now),
                    ),
                )
                self._save_result_in_transaction(
                    connection,
                    result=validated,
                    document_text=document_text,
                    document_bytes=len(document),
                    created_at=now,
                    fresh_run_id=claim.run_id,
                )
                connection.execute(
                    """
                    UPDATE cayu_eval_runs
                    SET status = ?, updated_at = ?, finished_at = ?, lease_expires_at = NULL,
                        result_revision = ?, result_status = ?, result_score = ?,
                        result_duration_ms = ?, trial_checkpoint_count = 0,
                        trial_checkpoint_bytes = 0
                    WHERE run_id = ?
                    """,
                    (
                        str(EvalRunStatus.COMPLETED),
                        _format_datetime(now),
                        _format_datetime(now),
                        summary.revision,
                        summary.status,
                        summary.score,
                        summary.duration_ms,
                        claim.run_id,
                    ),
                )
                _delete_trial_checkpoints(connection, claim.run_id)
                updated = self._require_run_row(connection, claim.run_id)
                connection.commit()
                return _run_record_from_row(updated)
            except BaseException:
                connection.rollback()
                raise

        return await self._run_eval_writer(
            operation,
            operation_name="publish_result",
            run_id=claim.run_id,
        )

    async def fail_run(
        self,
        claim: EvalRunClaim,
        code: EvalRunFailureCode,
        *,
        diagnostic: EvalRunFailureDiagnostic | None = None,
    ) -> EvalRunRecord:
        claim = _exact_model(claim, EvalRunClaim, "claim")
        if not isinstance(code, EvalRunFailureCode):
            raise TypeError("code must be an EvalRunFailureCode.")
        if diagnostic is not None:
            diagnostic = _exact_model(diagnostic, EvalRunFailureDiagnostic, "diagnostic")
        return await self._terminalize_without_result(
            claim,
            required_status=EvalRunStatus.RUNNING,
            terminal_status=EvalRunStatus.FAILED,
            failure_code=code,
            diagnostic=diagnostic,
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

        def operation(connection: sqlite3.Connection) -> EvalRunRecord:
            try:
                connection.execute("BEGIN IMMEDIATE")
                now = datetime.now(UTC)
                row = self._require_run_row(connection, claim.run_id)
                self._require_live_claim(row, claim, now)
                status = EvalRunStatus(row["status"])
                if status is EvalRunStatus.CANCELLING:
                    next_status = EvalRunStatus.CANCELLED
                    finished_at = _format_datetime(now)
                    cancel_requested_at = row["cancel_requested_at"] or _format_datetime(now)
                elif status is EvalRunStatus.RUNNING:
                    next_status = EvalRunStatus.QUEUED
                    finished_at = None
                    cancel_requested_at = None
                else:
                    raise EvalRunStateConflict("Only active eval work may be released.")
                connection.execute(
                    """
                    UPDATE cayu_eval_runs
                    SET status = ?, updated_at = ?, finished_at = ?,
                        cancel_requested_at = ?, claim_id = NULL,
                        lease_expires_at = NULL,
                        trial_checkpoint_count = CASE WHEN ? = ? THEN 0
                            ELSE trial_checkpoint_count END,
                        trial_checkpoint_bytes = CASE WHEN ? = ? THEN 0
                            ELSE trial_checkpoint_bytes END
                    WHERE run_id = ?
                    """,
                    (
                        str(next_status),
                        _format_datetime(now),
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
                    _delete_trial_checkpoints(connection, claim.run_id)
                updated = self._require_run_row(connection, claim.run_id)
                connection.commit()
                return _run_record_from_row(updated)
            except BaseException:
                connection.rollback()
                raise

        return await self._run(operation)

    async def load_result(
        self,
        run_id: str,
        *,
        max_bytes: int = CORPUS_EXECUTION_RESULT_MAX_BYTES,
    ) -> CorpusExecutionResult | None:
        run_id = _store_identifier(run_id, "run_id")
        max_bytes = _read_limit(max_bytes, hard_max=CORPUS_EXECUTION_RESULT_MAX_BYTES)

        def operation(connection: sqlite3.Connection) -> str | None:
            size = connection.execute(
                "SELECT result_bytes FROM cayu_eval_results WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if size is None:
                return None
            if size["result_bytes"] > max_bytes:
                raise EvalStoreResultTooLarge(max_bytes)
            row = connection.execute(
                "SELECT result_json FROM cayu_eval_results WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError("Immutable eval result disappeared during a read.")
            return row["result_json"]

        document = await self._run(operation)
        if document is None:
            return None
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

        def operation(connection: sqlite3.Connection) -> EvalResultRecord:
            try:
                connection.execute("BEGIN IMMEDIATE")
                now = datetime.now(UTC)
                self._save_prepared_corpus_in_transaction(
                    connection,
                    corpus=corpus,
                    document_text=corpus_text,
                    document_bytes=len(corpus_document),
                    inspection=inspection,
                    suites=suites,
                    cases=cases,
                    created_at=now,
                )
                record = self._save_result_in_transaction(
                    connection,
                    result=result,
                    document_text=result_text,
                    document_bytes=len(result_document),
                    created_at=now,
                    fresh_run_id=None,
                )
                connection.commit()
                return record
            except BaseException:
                connection.rollback()
                raise

        return await self._run(operation)

    async def load_result_by_revision(
        self,
        revision: str,
        *,
        max_bytes: int = CORPUS_EXECUTION_RESULT_MAX_BYTES,
    ) -> CorpusExecutionResult | CapturedEvaluationResultV1 | None:
        revision = _sha256_revision(revision, "revision")
        max_bytes = _read_limit(max_bytes, hard_max=CORPUS_EXECUTION_RESULT_MAX_BYTES)

        def operation(connection: sqlite3.Connection) -> tuple[EvalResultOrigin, str] | None:
            row = connection.execute(
                """
                SELECT origin, document_bytes, fresh_run_id
                FROM cayu_eval_result_records
                WHERE revision = ?
                """,
                (revision,),
            ).fetchone()
            if row is None:
                return None
            if row["document_bytes"] > max_bytes:
                raise EvalStoreResultTooLarge(max_bytes)
            origin = EvalResultOrigin(row["origin"])
            if origin is EvalResultOrigin.CAPTURED_SESSION:
                document_row = connection.execute(
                    "SELECT captured_result_json FROM cayu_eval_result_records WHERE revision = ?",
                    (revision,),
                ).fetchone()
                document = None if document_row is None else document_row[0]
            else:
                document_row = connection.execute(
                    "SELECT result_json FROM cayu_eval_results WHERE run_id = ?",
                    (row["fresh_run_id"],),
                ).fetchone()
                document = None if document_row is None else document_row[0]
            if document is None:
                raise RuntimeError("Immutable eval result document is unavailable.")
            return origin, document

        loaded = await self._run(operation)
        if loaded is None:
            return None
        origin, document = loaded
        if origin is EvalResultOrigin.CAPTURED_SESSION:
            return await asyncio.to_thread(captured_evaluation_result_from_json, document)
        return await asyncio.to_thread(corpus_execution_result_from_json, document)

    async def load_result_record(self, revision: str) -> EvalResultRecord | None:
        revision = _sha256_revision(revision, "revision")

        def operation(connection: sqlite3.Connection) -> EvalResultRecord | None:
            row = connection.execute(
                f"SELECT {_RESULT_RECORD_COLUMNS} FROM cayu_eval_result_records WHERE revision = ?",
                (revision,),
            ).fetchone()
            return None if row is None else _result_record_from_row(row)

        return await self._run(operation)

    async def list_results(self, query: EvalResultQuery) -> EvalResultPage:
        query = _exact_model(query, EvalResultQuery, "query")
        boundary = (
            decode_result_cursor(query.cursor, query.target_key, query.origin)
            if query.cursor is not None
            else None
        )

        def operation(connection: sqlite3.Connection) -> EvalResultPage:
            clauses = ["target_key = ?"]
            params: list[object] = [query.target_key]
            if query.origin is not None:
                clauses.append("origin = ?")
                params.append(query.origin.value)
            if boundary is not None:
                clauses.append("(created_at < ? OR (created_at = ? AND revision > ?))")
                timestamp = _format_datetime(boundary[0])
                params.extend((timestamp, timestamp, boundary[1]))
            rows = connection.execute(
                f"""
                SELECT {_RESULT_RECORD_COLUMNS}
                FROM cayu_eval_result_records
                WHERE {" AND ".join(clauses)}
                ORDER BY created_at DESC, revision ASC
                LIMIT ?
                """,
                (*params, query.limit + 1),
            ).fetchall()
            return _bounded_result_page(
                [_result_record_from_row(row) for row in rows],
                query,
            )

        return await self._run(operation)

    async def set_baseline(
        self,
        update: EvalBaselineUpdate,
        *,
        redact_json: Callable[[Any], Any],
    ) -> EvalBaselineMutationRecord:
        update = _prepare_baseline_update_for_store(update, redact_json=redact_json)

        def operation(connection: sqlite3.Connection) -> EvalBaselineMutationRecord:
            try:
                connection.execute("BEGIN IMMEDIATE")
                replay_row = connection.execute(
                    "SELECT * FROM cayu_eval_baseline_mutations WHERE operation_id = ?",
                    (update.operation_id,),
                ).fetchone()
                if replay_row is not None:
                    replay = _baseline_mutation_from_row(replay_row)
                    if not self._baseline_mutation_matches(replay, update):
                        raise EvalBaselineConflict(
                            "Baseline operation id is already bound to another mutation."
                        )
                    connection.commit()
                    return replay
                result_row = connection.execute(
                    f"SELECT {_RESULT_RECORD_COLUMNS} FROM cayu_eval_result_records "
                    "WHERE revision = ?",
                    (update.result_revision,),
                ).fetchone()
                if result_row is None:
                    raise KeyError(f"Eval result not found: {update.result_revision}")
                _validate_baseline_result(update, _result_record_from_row(result_row))
                current_row = connection.execute(
                    """
                    SELECT target_key, corpus_revision, suite_id, result_revision,
                           generation, updated_by, updated_at
                    FROM cayu_eval_baselines
                    WHERE target_key = ? AND corpus_revision = ? AND suite_id = ?
                    """,
                    (
                        update.key.target_key,
                        update.key.corpus_revision,
                        update.key.suite_id,
                    ),
                ).fetchone()
                current = None if current_row is None else _baseline_record_from_row(current_row)
                generation = 0 if current is None else current.generation
                if generation != update.expected_generation:
                    raise EvalBaselineConflict("Eval baseline generation changed.")
                if generation >= 9223372036854775807:
                    raise EvalBaselineConflict("Eval baseline generation is exhausted.")
                now = datetime.now(UTC)
                next_generation = generation + 1
                connection.execute(
                    """
                    INSERT INTO cayu_eval_baselines (
                        target_key, corpus_revision, suite_id, result_revision,
                        generation, updated_by, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(target_key, corpus_revision, suite_id) DO UPDATE SET
                        result_revision = excluded.result_revision,
                        generation = excluded.generation,
                        updated_by = excluded.updated_by,
                        updated_at = excluded.updated_at
                    """,
                    (
                        update.key.target_key,
                        update.key.corpus_revision,
                        update.key.suite_id,
                        update.result_revision,
                        next_generation,
                        update.actor_id,
                        _format_datetime(now),
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO cayu_eval_baseline_mutations (
                        operation_id, target_key, corpus_revision, suite_id,
                        expected_generation, previous_result_revision,
                        selected_result_revision, resulting_generation, actor_id, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        _format_datetime(now),
                    ),
                )
                row = connection.execute(
                    "SELECT * FROM cayu_eval_baseline_mutations WHERE operation_id = ?",
                    (update.operation_id,),
                ).fetchone()
                assert row is not None
                connection.commit()
                return _baseline_mutation_from_row(row)
            except BaseException:
                connection.rollback()
                raise

        return await self._run(operation)

    async def load_baseline(self, key: EvalBaselineKey) -> EvalBaselineRecord | None:
        key = _exact_model(key, EvalBaselineKey, "key")

        def operation(connection: sqlite3.Connection) -> EvalBaselineRecord | None:
            row = connection.execute(
                """
                SELECT target_key, corpus_revision, suite_id, result_revision,
                       generation, updated_by, updated_at
                FROM cayu_eval_baselines
                WHERE target_key = ? AND corpus_revision = ? AND suite_id = ?
                """,
                (key.target_key, key.corpus_revision, key.suite_id),
            ).fetchone()
            return None if row is None else _baseline_record_from_row(row)

        return await self._run(operation)

    async def load_baseline_mutation(
        self,
        operation_id: str,
    ) -> EvalBaselineMutationRecord | None:
        operation_id = _sha256_revision(operation_id, "operation_id")

        def operation(connection: sqlite3.Connection) -> EvalBaselineMutationRecord | None:
            row = connection.execute(
                "SELECT * FROM cayu_eval_baseline_mutations WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
            return None if row is None else _baseline_mutation_from_row(row)

        return await self._run(operation)

    async def _terminalize_without_result(
        self,
        claim: EvalRunClaim,
        *,
        required_status: EvalRunStatus,
        terminal_status: EvalRunStatus,
        failure_code: EvalRunFailureCode | None,
        diagnostic: EvalRunFailureDiagnostic | None = None,
    ) -> EvalRunRecord:

        def operation(connection: sqlite3.Connection) -> EvalRunRecord:
            try:
                connection.execute("BEGIN IMMEDIATE")
                now = datetime.now(UTC)
                row = self._require_run_row(connection, claim.run_id)
                status = EvalRunStatus(row["status"])
                if status is terminal_status:
                    stored_code = (
                        None
                        if row["failure_code"] is None
                        else EvalRunFailureCode(row["failure_code"])
                    )
                    if (
                        self._claim_matches(row, claim)
                        and stored_code is failure_code
                        and _run_record_from_row(row).failure_diagnostic == diagnostic
                    ):
                        connection.commit()
                        return _run_record_from_row(row)
                    raise EvalRunStateConflict("Eval run already has another terminal outcome.")
                self._require_live_claim(row, claim, now)
                if status is not required_status:
                    raise EvalRunStateConflict(
                        f"Eval run must be {required_status} before becoming {terminal_status}."
                    )
                cancel_requested_at = row["cancel_requested_at"]
                if terminal_status is EvalRunStatus.CANCELLED and cancel_requested_at is None:
                    cancel_requested_at = _format_datetime(now)
                connection.execute(
                    """
                    UPDATE cayu_eval_runs
                    SET status = ?, updated_at = ?, finished_at = ?,
                        cancel_requested_at = ?, lease_expires_at = NULL, failure_code = ?,
                        trial_checkpoint_count = 0, trial_checkpoint_bytes = 0, failure_diagnostic_json = ?
                    WHERE run_id = ?
                    """,
                    (
                        str(terminal_status),
                        _format_datetime(now),
                        _format_datetime(now),
                        cancel_requested_at,
                        None if failure_code is None else str(failure_code),
                        None if diagnostic is None else diagnostic.model_dump_json(),
                        claim.run_id,
                    ),
                )
                _delete_trial_checkpoints(connection, claim.run_id)
                updated = self._require_run_row(connection, claim.run_id)
                connection.commit()
                return _run_record_from_row(updated)
            except BaseException:
                connection.rollback()
                raise

        return await self._run(operation)

    @staticmethod
    def _claim_matches(row: sqlite3.Row, claim: EvalRunClaim) -> bool:
        return row["claim_id"] == claim.claim_id and row["ownership_epoch"] == claim.epoch

    @classmethod
    def _require_live_claim(
        cls,
        row: sqlite3.Row,
        claim: EvalRunClaim,
        now: datetime,
    ) -> None:
        if not cls._claim_matches(row, claim):
            raise EvalRunClaimLost("Eval run claim is no longer owned by this worker.")
        if EvalRunStatus(row["status"]) not in {
            EvalRunStatus.RUNNING,
            EvalRunStatus.CANCELLING,
        }:
            raise EvalRunClaimLost("Eval run is no longer active.")
        expires_at = _parse_optional_datetime(row["lease_expires_at"])
        if expires_at is None or expires_at <= now:
            raise EvalRunClaimLost("Eval run claim lease has expired.")

    @staticmethod
    def _load_run_row(connection: sqlite3.Connection, run_id: str) -> sqlite3.Row | None:
        return connection.execute(
            f"SELECT {_RUN_COLUMNS} FROM cayu_eval_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()

    @classmethod
    def _require_run_row(cls, connection: sqlite3.Connection, run_id: str) -> sqlite3.Row:
        row = cls._load_run_row(connection, run_id)
        if row is None:
            raise KeyError(f"Eval run not found: {run_id}")
        return row

    async def _load_run_request(self, run_id: str) -> EvalRunRequest:
        def operation(connection: sqlite3.Connection) -> EvalRunRequest:
            return _request_from_row(self._require_run_row(connection, run_id))

        return await self._run(operation)

    @classmethod
    def _save_result_in_transaction(
        cls,
        connection: sqlite3.Connection,
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
        existing_row = connection.execute(
            f"SELECT {_RESULT_RECORD_COLUMNS}, fresh_run_id, captured_result_json "
            "FROM cayu_eval_result_records WHERE revision = ?",
            (record.revision,),
        ).fetchone()
        if existing_row is not None:
            existing = _result_record_from_row(existing_row)
            if existing != record.model_copy(update={"created_at": existing.created_at}):
                raise EvalResultConflict(
                    f"Eval result revision {record.revision} has conflicting metadata."
                )
            if existing.origin is EvalResultOrigin.CAPTURED_SESSION:
                existing_document = existing_row["captured_result_json"]
            else:
                fresh = connection.execute(
                    "SELECT result_json FROM cayu_eval_results WHERE run_id = ?",
                    (existing_row["fresh_run_id"],),
                ).fetchone()
                existing_document = None if fresh is None else fresh["result_json"]
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
        connection.execute(
            """
            INSERT INTO cayu_eval_result_records (
                revision, origin, target_key, corpus_revision, suite_id, suite_revision,
                application_release_id, app_manifest_schema_version,
                app_manifest_fingerprint, result_status, result_score, fresh_run_id,
                captured_result_json, document_bytes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                _format_datetime(created_at),
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
    def _save_prepared_corpus_in_transaction(
        cls,
        connection: sqlite3.Connection,
        *,
        corpus: EvalCorpusDocument,
        document_text: str,
        document_bytes: int,
        inspection: EvalCorpusInspectionV1,
        suites: tuple[EvalSuiteCatalogEntry, ...],
        cases: tuple[EvalCaseCatalogEntry, ...],
        created_at: datetime,
    ) -> EvalCorpusCatalogEntry:
        existing = connection.execute(
            "SELECT document_json FROM cayu_eval_corpora WHERE revision = ?",
            (corpus.revision,),
        ).fetchone()
        if existing is not None:
            if existing["document_json"] != document_text:
                raise EvalCorpusConflict(
                    f"Eval corpus revision {corpus.revision} has conflicting content."
                )
            entry = cls._load_corpus_entry(connection, corpus.revision)
            assert entry is not None
            return entry
        connection.execute(
            """
            INSERT INTO cayu_eval_corpora (
                revision, target_key, evidence_policy_revision,
                pricing_profile_fingerprint, suite_count, case_count,
                assertion_count, expanded_assertion_result_count,
                document_json, document_bytes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                _format_datetime(created_at),
            ),
        )
        connection.executemany(
            """
            INSERT INTO cayu_eval_suites (
                corpus_revision, suite_id, suite_revision, name, description,
                case_count, assertion_count, trials, timeout_seconds
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        connection.executemany(
            """
            INSERT INTO cayu_eval_cases (
                corpus_revision, case_id, case_revision, suite_id, name,
                description, message_count, assertion_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
        entry = cls._load_corpus_entry(connection, corpus.revision)
        assert entry is not None
        return entry

    @staticmethod
    def _corpus_exists(connection: sqlite3.Connection, revision: str) -> bool:
        return (
            connection.execute(
                "SELECT 1 FROM cayu_eval_corpora WHERE revision = ?",
                (revision,),
            ).fetchone()
            is not None
        )

    @classmethod
    def _load_corpus_entry(
        cls,
        connection: sqlite3.Connection,
        revision: str,
    ) -> EvalCorpusCatalogEntry | None:
        row = connection.execute(
            """
            SELECT revision, target_key, evidence_policy_revision,
                   pricing_profile_fingerprint, suite_count, case_count,
                   assertion_count, expanded_assertion_result_count,
                   document_bytes, created_at
            FROM cayu_eval_corpora
            WHERE revision = ?
            """,
            (revision,),
        ).fetchone()
        return None if row is None else cls._corpus_entry_from_row(row)

    @staticmethod
    def _corpus_entry_from_row(row: sqlite3.Row) -> EvalCorpusCatalogEntry:
        return EvalCorpusCatalogEntry(
            revision=row["revision"],
            target_key=row["target_key"],
            evidence_policy_revision=row["evidence_policy_revision"],
            pricing_profile_fingerprint=row["pricing_profile_fingerprint"],
            suite_count=row["suite_count"],
            case_count=row["case_count"],
            assertion_count=row["assertion_count"],
            expanded_assertion_result_count=row["expanded_assertion_result_count"],
            document_bytes=row["document_bytes"],
            created_at=sqlite_support.parse_datetime(row["created_at"]),
        )

    @classmethod
    def _load_scenario_entry(
        cls,
        connection: sqlite3.Connection,
        revision: str,
    ) -> EvalScenarioCatalogEntry | None:
        row = connection.execute(
            """
            SELECT revision, scenario_id, target_key, name, description,
                   event_count, input_event_count, approval_checkpoint_count,
                   message_count, part_count, artifact_requirement_count,
                   secret_requirement_count, document_bytes, created_at
            FROM cayu_eval_scenarios
            WHERE revision = ?
            """,
            (revision,),
        ).fetchone()
        return None if row is None else cls._scenario_entry_from_row(row)

    @staticmethod
    def _scenario_entry_from_row(row: sqlite3.Row) -> EvalScenarioCatalogEntry:
        return EvalScenarioCatalogEntry(
            revision=row["revision"],
            id=row["scenario_id"],
            target_key=row["target_key"],
            name=row["name"],
            description=row["description"],
            event_count=row["event_count"],
            input_event_count=row["input_event_count"],
            approval_checkpoint_count=row["approval_checkpoint_count"],
            message_count=row["message_count"],
            part_count=row["part_count"],
            artifact_requirement_count=row["artifact_requirement_count"],
            secret_requirement_count=row["secret_requirement_count"],
            document_bytes=row["document_bytes"],
            created_at=sqlite_support.parse_datetime(row["created_at"]),
        )

    @classmethod
    def _load_authored_suite_entry(
        cls,
        connection: sqlite3.Connection,
        revision: str,
    ) -> EvalAuthoredSuiteCatalogEntry | None:
        row = connection.execute(
            """
            SELECT revision, suite_id, suite_revision, target_key, name,
                   description, case_count, assertion_count, simple_input_count,
                   scenario_count, trials, timeout_seconds, document_bytes, created_at
            FROM cayu_eval_authored_suites
            WHERE revision = ?
            """,
            (revision,),
        ).fetchone()
        return None if row is None else cls._authored_suite_entry_from_row(row)

    @staticmethod
    def _authored_suite_entry_from_row(
        row: sqlite3.Row,
    ) -> EvalAuthoredSuiteCatalogEntry:
        return EvalAuthoredSuiteCatalogEntry(
            revision=row["revision"],
            id=row["suite_id"],
            suite_revision=row["suite_revision"],
            target_key=row["target_key"],
            name=row["name"],
            description=row["description"],
            case_count=row["case_count"],
            assertion_count=row["assertion_count"],
            simple_input_count=row["simple_input_count"],
            scenario_count=row["scenario_count"],
            trials=row["trials"],
            timeout_seconds=row["timeout_seconds"],
            document_bytes=row["document_bytes"],
            created_at=sqlite_support.parse_datetime(row["created_at"]),
        )

    @staticmethod
    def _suite_entry_from_row(row: sqlite3.Row) -> EvalSuiteCatalogEntry:
        return EvalSuiteCatalogEntry(
            corpus_revision=row["corpus_revision"],
            id=row["suite_id"],
            revision=row["suite_revision"],
            name=row["name"],
            description=row["description"],
            case_count=row["case_count"],
            assertion_count=row["assertion_count"],
            trials=row["trials"],
            timeout_seconds=row["timeout_seconds"],
        )

    @staticmethod
    def _case_entry_from_row(row: sqlite3.Row) -> EvalCaseCatalogEntry:
        return EvalCaseCatalogEntry(
            corpus_revision=row["corpus_revision"],
            id=row["case_id"],
            revision=row["case_revision"],
            suite_id=row["suite_id"],
            name=row["name"],
            description=row["description"],
            message_count=row["message_count"],
            assertion_count=row["assertion_count"],
        )


__all__ = [
    "DEFAULT_SQLITE_EVAL_WRITER_CONTENTION_POLICY",
    "SQLiteEvalStore",
    "SQLiteEvalWriterContentionPolicy",
]
