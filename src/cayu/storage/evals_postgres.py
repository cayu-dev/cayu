from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, LiteralString, cast
from uuid import uuid4

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
from cayu.evals.store import (
    TERMINAL_EVAL_RUN_STATUSES,
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
    EvalResultConflict,
    EvalResultPage,
    EvalResultQuery,
    EvalResultRecord,
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
    _bounded_result_page,
    _bounded_run_page,
    _bounded_suite_page,
    _claim_target_keys,
    _copy_query,
    _exact_model,
    _lease_seconds,
    _prepare_baseline_update_for_store,
    _prepare_captured_result_for_store,
    _prepare_corpus_catalog_for_store,
    _prepare_result_for_store,
    _prepare_run_request_for_store,
    _read_limit,
    _store_identifier,
    _validate_baseline_result,
    decode_case_cursor,
    decode_corpus_cursor,
    decode_result_cursor,
    decode_run_cursor,
    decode_suite_cursor,
    eval_result_record,
    result_summary,
    validate_result_for_run,
)
from cayu.storage.postgres import _PostgresStoreBase

_POSTGRES_EVAL_MIN_REQUIRED_REVISION = 48

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
    return EvalRunRequest(
        run_id=row[0],
        idempotency_key=row[1],
        corpus_revision=row[2],
        target_key=row[3],
        suite_id=row[4],
        suite_revision=row[5],
        max_concurrency=row[6],
    )


def _run_record_from_row(row: Any) -> EvalRunRecord:
    status = EvalRunStatus(row[7])
    ownership = None
    if status in {EvalRunStatus.RUNNING, EvalRunStatus.CANCELLING}:
        ownership = EvalRunOwnership(
            epoch=row[14],
            lease_expires_at=row[15],
        )
    result = None
    if row[16] is not None:
        result = EvalRunResultSummary(
            revision=row[16],
            status=row[17],
            score=row[18],
            duration_ms=row[19],
        )
    return EvalRunRecord(
        spec=EvalRunSpec(
            run_id=row[0],
            corpus_revision=row[2],
            target_key=row[3],
            suite_id=row[4],
            suite_revision=row[5],
            max_concurrency=row[6],
        ),
        status=status,
        attempt_count=row[14],
        created_at=row[8],
        updated_at=row[9],
        started_at=row[10],
        finished_at=row[11],
        cancel_requested_at=row[12],
        ownership=ownership,
        result=result,
        failure_code=None if row[20] is None else EvalRunFailureCode(row[20]),
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
                            suite_id, suite_revision, max_concurrency, status,
                            created_at, updated_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                            str(EvalRunStatus.QUEUED),
                            now,
                            now,
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

    async def load_run(self, run_id: str) -> EvalRunRecord | None:
        run_id = _store_identifier(run_id, "run_id")
        await self._ensure_ready()
        async with self._connection() as conn, conn.cursor() as cur:
            row = await self._load_run_row(cur, run_id)
            return None if row is None else _run_record_from_row(row)

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
                    target_clause = "" if target_key is None else "AND target_key = %s"
                    target_params: tuple[str, ...] = () if target_key is None else (target_key,)
                    await cur.execute(
                        f"""
                        SELECT {_RUN_COLUMNS}
                        FROM cayu_eval_runs
                        WHERE ownership_epoch < 9223372036854775807
                          {target_clause}
                          AND (status = %s
                           OR (status IN (%s, %s) AND lease_expires_at <= %s))
                        ORDER BY created_at ASC, run_id ASC
                        FOR UPDATE SKIP LOCKED
                        LIMIT 1
                        """,
                        (
                            *target_params,
                            str(EvalRunStatus.QUEUED),
                            str(EvalRunStatus.RUNNING),
                            str(EvalRunStatus.CANCELLING),
                            now,
                        ),
                    )
                    row = await cur.fetchone()
                    if row is None:
                        await conn.commit()
                        return None
                    status = (
                        EvalRunStatus.CANCELLING if row[12] is not None else EvalRunStatus.RUNNING
                    )
                    await cur.execute(
                        """
                        UPDATE cayu_eval_runs
                        SET status = %s, updated_at = %s,
                            started_at = COALESCE(started_at, %s),
                            claim_id = %s,
                            ownership_epoch = ownership_epoch + 1,
                            lease_expires_at = %s
                        WHERE run_id = %s
                        """,
                        (
                            str(status),
                            now,
                            now,
                            str(uuid4()),
                            now + timedelta(seconds=lease_seconds),
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
                        claim_id=claimed[13],
                        epoch=claimed[14],
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
                        FROM cayu_eval_runs
                        WHERE ownership_epoch < 9223372036854775807
                          AND target_key IN ({placeholders})
                          AND (status = %s
                           OR (status IN (%s, %s) AND lease_expires_at <= %s))
                        ORDER BY created_at ASC, run_id ASC
                        FOR UPDATE SKIP LOCKED
                        LIMIT 1
                        """,
                        (
                            *target_keys,
                            str(EvalRunStatus.QUEUED),
                            str(EvalRunStatus.RUNNING),
                            str(EvalRunStatus.CANCELLING),
                            now,
                        ),
                    )
                    row = await cur.fetchone()
                    if row is None:
                        await conn.commit()
                        return None
                    status = (
                        EvalRunStatus.CANCELLING if row[12] is not None else EvalRunStatus.RUNNING
                    )
                    claim_id = str(uuid4())
                    await cur.execute(
                        """
                        UPDATE cayu_eval_runs
                        SET status = %s, updated_at = %s,
                            started_at = COALESCE(started_at, %s),
                            claim_id = %s,
                            ownership_epoch = ownership_epoch + 1,
                            lease_expires_at = %s
                        WHERE run_id = %s
                        """,
                        (
                            str(status),
                            now,
                            now,
                            claim_id,
                            now + timedelta(seconds=lease_seconds),
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
                        claim_id=claimed[13],
                        epoch=claimed[14],
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

    async def request_cancel(self, run_id: str) -> EvalRunRecord:
        run_id = _store_identifier(run_id, "run_id")
        await self._ensure_ready()
        async with self._connection() as conn:
            try:
                async with conn.cursor() as cur:
                    row = await self._require_run_row(cur, run_id, for_update=True)
                    status = EvalRunStatus(row[7])
                    if status in TERMINAL_EVAL_RUN_STATUSES:
                        await conn.commit()
                        return _run_record_from_row(row)
                    now = await _database_now(cur)
                    claim_expired = (
                        status in {EvalRunStatus.RUNNING, EvalRunStatus.CANCELLING}
                        and row[15] is not None
                        and row[15] <= now
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
                                WHEN %s = %s THEN NULL ELSE lease_expires_at END
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
                            run_id,
                        ),
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
                    status = EvalRunStatus(row[7])
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
                            result_duration_ms = %s
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
                    status = EvalRunStatus(row[7])
                    if status is EvalRunStatus.CANCELLING:
                        next_status = EvalRunStatus.CANCELLED
                        finished_at = now
                        cancel_requested_at = row[12] or now
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
                            lease_expires_at = NULL
                        WHERE run_id = %s
                        """,
                        (
                            str(next_status),
                            now,
                            finished_at,
                            cancel_requested_at,
                            claim.run_id,
                        ),
                    )
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
                    status = EvalRunStatus(row[7])
                    if status is terminal_status:
                        stored_code = None if row[20] is None else EvalRunFailureCode(row[20])
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
                    cancel_requested_at = row[12]
                    if terminal_status is EvalRunStatus.CANCELLED and cancel_requested_at is None:
                        cancel_requested_at = now
                    await cur.execute(
                        """
                        UPDATE cayu_eval_runs
                        SET status = %s, updated_at = %s, finished_at = %s,
                            cancel_requested_at = %s, lease_expires_at = NULL,
                            failure_code = %s
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
                    updated = await self._require_run_row(cur, claim.run_id)
                await conn.commit()
                return _run_record_from_row(updated)
            except BaseException:
                await conn.rollback()
                raise

    @staticmethod
    def _claim_matches(row: Any, claim: EvalRunClaim) -> bool:
        return row[13] == claim.claim_id and row[14] == claim.epoch

    @classmethod
    def _require_live_claim(cls, row: Any, claim: EvalRunClaim, now: datetime) -> None:
        if not cls._claim_matches(row, claim):
            raise EvalRunClaimLost("Eval run claim is no longer owned by this worker.")
        if EvalRunStatus(row[7]) not in {EvalRunStatus.RUNNING, EvalRunStatus.CANCELLING}:
            raise EvalRunClaimLost("Eval run is no longer active.")
        if row[15] is None or row[15] <= now:
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
