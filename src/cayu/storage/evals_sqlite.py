from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar
from uuid import uuid4

from cayu.evals.corpus import (
    EVAL_CORPUS_MAX_BYTES,
    EvalCorpusDocument,
    _sha256_revision,
    inspect_eval_corpus,
)
from cayu.evals.execution import CORPUS_EXECUTION_RESULT_MAX_BYTES, CorpusExecutionResult
from cayu.evals.execution_reporting import corpus_execution_result_from_json
from cayu.evals.store import (
    TERMINAL_EVAL_RUN_STATUSES,
    EvalCaseCatalogEntry,
    EvalCaseCatalogPage,
    EvalCaseCatalogQuery,
    EvalCatalogQuery,
    EvalCorpusCatalogEntry,
    EvalCorpusCatalogPage,
    EvalCorpusConflict,
    EvalRunAdmissionConflict,
    EvalRunClaim,
    EvalRunClaimLost,
    EvalRunFailureCode,
    EvalRunLease,
    EvalRunOwnership,
    EvalRunPage,
    EvalRunQuery,
    EvalRunRecord,
    EvalRunRequest,
    EvalRunResultSummary,
    EvalRunSpec,
    EvalRunStateConflict,
    EvalRunStatus,
    EvalStore,
    EvalStoreResultTooLarge,
    EvalSuiteCatalogEntry,
    EvalSuiteCatalogPage,
    EvalSuiteCatalogQuery,
    _bounded_case_page,
    _bounded_corpus_page,
    _bounded_run_page,
    _bounded_suite_page,
    _copy_query,
    _exact_model,
    _lease_seconds,
    _prepare_corpus_for_store,
    _prepare_result_for_store,
    _prepare_run_request_for_store,
    _read_limit,
    _store_identifier,
    case_catalog_entries,
    decode_case_cursor,
    decode_corpus_cursor,
    decode_run_cursor,
    decode_suite_cursor,
    result_summary,
    suite_catalog_entries,
    validate_result_for_run,
)
from cayu.storage import _sqlite_support as sqlite_support
from cayu.storage import migrations as schema
from cayu.storage.sqlite import _run_off_thread_with_connection_ownership

_SQLITE_EVAL_MIN_REQUIRED_REVISION = 32

_RUN_COLUMNS = """
    run_id,
    idempotency_key,
    corpus_revision,
    target_key,
    suite_id,
    suite_revision,
    max_concurrency,
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
    failure_code
"""


def _format_datetime(value: datetime) -> str:
    return sqlite_support.format_datetime(value)


def _parse_optional_datetime(value: str | None) -> datetime | None:
    return None if value is None else sqlite_support.parse_datetime(value)


def _request_from_row(row: sqlite3.Row) -> EvalRunRequest:
    return EvalRunRequest(
        run_id=row["run_id"],
        idempotency_key=row["idempotency_key"],
        corpus_revision=row["corpus_revision"],
        target_key=row["target_key"],
        suite_id=row["suite_id"],
        suite_revision=row["suite_revision"],
        max_concurrency=row["max_concurrency"],
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
        ),
        status=status,
        created_at=sqlite_support.parse_datetime(row["created_at"]),
        updated_at=sqlite_support.parse_datetime(row["updated_at"]),
        started_at=_parse_optional_datetime(row["started_at"]),
        finished_at=_parse_optional_datetime(row["finished_at"]),
        cancel_requested_at=_parse_optional_datetime(row["cancel_requested_at"]),
        ownership=ownership,
        result=result,
        failure_code=(
            None if row["failure_code"] is None else EvalRunFailureCode(row["failure_code"])
        ),
    )


class SQLiteEvalStore(EvalStore):
    """Restart-durable embedded eval persistence for a single SQLite database."""

    durable: ClassVar[bool] = True

    def __init__(
        self,
        path: str | Path,
        *,
        schema_mode: schema.SchemaMode = schema.SchemaMode.CREATE,
    ) -> None:
        if isinstance(path, Path):
            db_path = path
        elif type(path) is str and path.strip():
            db_path = Path(path)
        else:
            raise TypeError("SQLiteEvalStore path must be a nonblank string or Path.")
        if not isinstance(schema_mode, schema.SchemaMode):
            raise TypeError("schema_mode must be a SchemaMode.")
        self.path = db_path
        self._lock = asyncio.Lock()
        self._connection = sqlite_support.connect(db_path)
        try:
            sqlite_support.reconcile_schema(
                self._connection,
                schema_mode,
                app_min_supported=_SQLITE_EVAL_MIN_REQUIRED_REVISION,
            )
        except BaseException:
            self._connection.close()
            raise

    async def _run(self, operation):
        return await _run_off_thread_with_connection_ownership(
            self._lock,
            self._connection,
            operation,
        )

    async def close(self) -> None:
        await self._run(lambda connection: connection.close())

    async def save_corpus(
        self,
        corpus: EvalCorpusDocument,
        *,
        redact_json_values: Callable[[Any], Any],
    ) -> EvalCorpusCatalogEntry:
        corpus, document = _prepare_corpus_for_store(
            corpus,
            redact_json_values=redact_json_values,
        )
        document_text = document.decode("utf-8")
        suites = suite_catalog_entries(corpus)
        cases = case_catalog_entries(corpus)
        inspection = inspect_eval_corpus(corpus)

        def operation(connection: sqlite3.Connection) -> EvalCorpusCatalogEntry:
            try:
                connection.execute("BEGIN IMMEDIATE")
                now = datetime.now(UTC)
                existing = connection.execute(
                    "SELECT document_json FROM cayu_eval_corpora WHERE revision = ?",
                    (corpus.revision,),
                ).fetchone()
                if existing is not None:
                    if existing["document_json"] != document_text:
                        raise EvalCorpusConflict(
                            f"Eval corpus revision {corpus.revision} has conflicting content."
                        )
                    entry = self._load_corpus_entry(connection, corpus.revision)
                    connection.commit()
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
                        len(document),
                        _format_datetime(now),
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
                connection.commit()
                entry = self._load_corpus_entry(connection, corpus.revision)
                assert entry is not None
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

        def operation(connection: sqlite3.Connection) -> EvalCorpusDocument | None:
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
            return EvalCorpusDocument.model_validate(json.loads(row["document_json"]))

        return await self._run(operation)

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
        redact_json_values: Callable[[Any], Any],
    ) -> EvalRunRecord:
        request = _prepare_run_request_for_store(
            request,
            redact_json_values=redact_json_values,
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
                        suite_id, suite_revision, max_concurrency, status,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        request.run_id,
                        request.idempotency_key,
                        request.corpus_revision,
                        request.target_key,
                        request.suite_id,
                        request.suite_revision,
                        request.max_concurrency,
                        str(EvalRunStatus.QUEUED),
                        timestamp,
                        timestamp,
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

    async def load_run(self, run_id: str) -> EvalRunRecord | None:
        run_id = _store_identifier(run_id, "run_id")

        def operation(connection: sqlite3.Connection) -> EvalRunRecord | None:
            row = self._load_run_row(connection, run_id)
            return None if row is None else _run_record_from_row(row)

        return await self._run(operation)

    async def list_runs(self, query: EvalRunQuery | None = None) -> EvalRunPage:
        query = _copy_query(query, EvalRunQuery)
        boundary = (
            decode_run_cursor(query.cursor, query.status, query.corpus_revision)
            if query.cursor is not None
            else None
        )

        def operation(connection: sqlite3.Connection) -> EvalRunPage:
            clauses: list[str] = []
            params: list[object] = []
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
        lease_seconds: int = 300,
    ) -> EvalRunLease | None:
        lease_seconds = _lease_seconds(lease_seconds)

        def operation(connection: sqlite3.Connection) -> EvalRunLease | None:
            try:
                connection.execute("BEGIN IMMEDIATE")
                now = datetime.now(UTC)
                claim_id = str(uuid4())
                row = connection.execute(
                    f"""
                    SELECT {_RUN_COLUMNS}
                    FROM cayu_eval_runs
                    WHERE ownership_epoch < 9223372036854775807
                      AND (status = ?
                       OR (status IN (?, ?) AND lease_expires_at <= ?))
                    ORDER BY created_at ASC, run_id ASC
                    LIMIT 1
                    """,
                    (
                        str(EvalRunStatus.QUEUED),
                        str(EvalRunStatus.RUNNING),
                        str(EvalRunStatus.CANCELLING),
                        _format_datetime(now),
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
                connection.execute(
                    """
                    UPDATE cayu_eval_runs
                    SET status = ?, updated_at = ?,
                        started_at = COALESCE(started_at, ?),
                        claim_id = ?,
                        ownership_epoch = ownership_epoch + 1,
                        lease_expires_at = ?
                    WHERE run_id = ?
                    """,
                    (
                        str(status),
                        _format_datetime(now),
                        _format_datetime(now),
                        claim_id,
                        _format_datetime(now + timedelta(seconds=lease_seconds)),
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

        return await self._run(operation)

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
                        lease_expires_at = CASE WHEN ? = ? THEN NULL ELSE lease_expires_at END
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
        redact_json_values: Callable[[Any], Any],
    ) -> EvalRunRecord:
        claim = _exact_model(claim, EvalRunClaim, "claim")
        result, document = _prepare_result_for_store(
            result,
            redact_json_values=redact_json_values,
        )
        document_text = document.decode("utf-8")

        def operation(connection: sqlite3.Connection) -> EvalRunRecord:
            try:
                connection.execute("BEGIN IMMEDIATE")
                now = datetime.now(UTC)
                row = self._require_run_row(connection, claim.run_id)
                request = _request_from_row(row)
                validated = validate_result_for_run(request, result)
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
                connection.execute(
                    """
                    UPDATE cayu_eval_runs
                    SET status = ?, updated_at = ?, finished_at = ?, lease_expires_at = NULL,
                        result_revision = ?, result_status = ?, result_score = ?,
                        result_duration_ms = ?
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
                updated = self._require_run_row(connection, claim.run_id)
                connection.commit()
                return _run_record_from_row(updated)
            except BaseException:
                connection.rollback()
                raise

        return await self._run(operation)

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
                        lease_expires_at = NULL
                    WHERE run_id = ?
                    """,
                    (
                        str(next_status),
                        _format_datetime(now),
                        finished_at,
                        cancel_requested_at,
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

    async def load_result(
        self,
        run_id: str,
        *,
        max_bytes: int = CORPUS_EXECUTION_RESULT_MAX_BYTES,
    ) -> CorpusExecutionResult | None:
        run_id = _store_identifier(run_id, "run_id")
        max_bytes = _read_limit(max_bytes, hard_max=CORPUS_EXECUTION_RESULT_MAX_BYTES)

        def operation(connection: sqlite3.Connection) -> CorpusExecutionResult | None:
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
            return corpus_execution_result_from_json(row["result_json"])

        return await self._run(operation)

    async def _terminalize_without_result(
        self,
        claim: EvalRunClaim,
        *,
        required_status: EvalRunStatus,
        terminal_status: EvalRunStatus,
        failure_code: EvalRunFailureCode | None,
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
                    if self._claim_matches(row, claim) and stored_code is failure_code:
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
                        cancel_requested_at = ?, lease_expires_at = NULL, failure_code = ?
                    WHERE run_id = ?
                    """,
                    (
                        str(terminal_status),
                        _format_datetime(now),
                        _format_datetime(now),
                        cancel_requested_at,
                        None if failure_code is None else str(failure_code),
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


__all__ = ["SQLiteEvalStore"]
