from __future__ import annotations

import asyncio

import pytest
from tests.evals.eval_store_conformance import assert_eval_store_conformance
from tests.evals.test_corpus_execution import _corpus, _provider, _target

from cayu.evals.execution import run_corpus_suite
from cayu.evals.store import EvalRunRequest, EvalRunStatus
from cayu.vaults.redaction import SecretRedactor

pytestmark = pytest.mark.usefixtures("postgres_dsn")
_NO_SECRETS = SecretRedactor()


async def _save_corpus(store, corpus):
    return await store.save_corpus(
        corpus,
        redact_json_values=_NO_SECRETS.redact_json_values,
    )


async def _admit_run(store, request):
    return await store.admit_run(
        request,
        redact_json_values=_NO_SECRETS.redact_json_values,
    )


async def _publish_result(store, claim, result):
    return await store.publish_result(
        claim,
        result,
        redact_json_values=_NO_SECRETS.redact_json_values,
    )


def _request(corpus, *, run_id: str = "run-1") -> EvalRunRequest:
    suite = corpus.suites[0]
    return EvalRunRequest(
        run_id=run_id,
        idempotency_key="sha256:" + "1" * 64,
        corpus_revision=corpus.revision,
        target_key=corpus.target_key,
        suite_id=suite.id,
        suite_revision=suite.revision,
        max_concurrency=1,
    )


async def _drop_eval_tables(dsn: str) -> None:
    import psycopg
    from psycopg import sql

    async with await psycopg.AsyncConnection.connect(dsn) as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT tablename FROM pg_tables "
            "WHERE schemaname = current_schema() AND tablename LIKE 'cayu_%'"
        )
        for (table,) in await cur.fetchall():
            await cur.execute(sql.SQL("DROP TABLE {} CASCADE").format(sql.Identifier(table)))
        await conn.commit()


def test_postgres_eval_store_shared_conformance(postgres_dsn) -> None:
    async def exercise() -> None:
        from cayu.storage.evals_postgres import PostgresEvalStore
        from cayu.storage.migrations import SchemaMode

        await _drop_eval_tables(postgres_dsn)
        corpus = _corpus(trials=1)
        result = await run_corpus_suite(
            _target(_provider(trials=1)),
            corpus,
            corpus.suites[0].id,
            max_concurrency=1,
        )
        store = PostgresEvalStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.MIGRATE,
        )
        try:
            await assert_eval_store_conformance(store, corpus=corpus, result=result)
        finally:
            await store.close()

    asyncio.run(exercise())


def test_postgres_eval_store_creates_revision_thirty_two_schema(postgres_dsn) -> None:
    async def exercise() -> None:
        import psycopg

        from cayu.storage.evals_postgres import PostgresEvalStore
        from cayu.storage.migrations import SchemaMode

        await _drop_eval_tables(postgres_dsn)
        store = PostgresEvalStore(postgres_dsn, schema_mode=SchemaMode.MIGRATE)
        try:
            await store.list_corpora()
        finally:
            await store.close()

        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as conn,
            conn.cursor() as cur,
        ):
            await cur.execute(
                "SELECT kind, compatible_from FROM cayu_schema_migrations WHERE revision = 32"
            )
            assert await cur.fetchone() == ("additive", 31)
            await cur.execute(
                "SELECT tablename FROM pg_tables "
                "WHERE schemaname = current_schema() AND tablename LIKE 'cayu_eval_%'"
            )
            assert {row[0] for row in await cur.fetchall()} == {
                "cayu_eval_cases",
                "cayu_eval_corpora",
                "cayu_eval_results",
                "cayu_eval_runs",
                "cayu_eval_suites",
            }
            await cur.execute(
                "SELECT table_name, column_name, data_type FROM information_schema.columns "
                "WHERE table_schema = current_schema() AND "
                "((table_name = 'cayu_eval_corpora' AND column_name = 'document') OR "
                "(table_name = 'cayu_eval_results' AND column_name = 'result')) "
                "ORDER BY table_name"
            )
            assert await cur.fetchall() == [
                ("cayu_eval_corpora", "document", "text"),
                ("cayu_eval_results", "result", "text"),
            ]
            await cur.execute(
                "SELECT table_name, column_name, collation_name "
                "FROM information_schema.columns "
                "WHERE table_schema = current_schema() AND "
                "((table_name = 'cayu_eval_corpora' AND column_name = 'revision') OR "
                "(table_name = 'cayu_eval_suites' AND column_name = 'suite_id') OR "
                "(table_name = 'cayu_eval_cases' AND column_name = 'case_id') OR "
                "(table_name = 'cayu_eval_runs' AND column_name = 'run_id')) "
                "ORDER BY table_name"
            )
            assert await cur.fetchall() == [
                ("cayu_eval_cases", "case_id", "C"),
                ("cayu_eval_corpora", "revision", "C"),
                ("cayu_eval_runs", "run_id", "C"),
                ("cayu_eval_suites", "suite_id", "C"),
            ]

    asyncio.run(exercise())


def test_postgres_eval_store_is_restart_durable_and_idempotent(postgres_dsn) -> None:
    async def exercise() -> None:
        from cayu.storage.evals_postgres import PostgresEvalStore
        from cayu.storage.migrations import SchemaMode

        await _drop_eval_tables(postgres_dsn)
        corpus = _corpus(trials=1)
        result = await run_corpus_suite(
            _target(_provider(trials=1)),
            corpus,
            corpus.suites[0].id,
            max_concurrency=1,
        )
        first = PostgresEvalStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.MIGRATE,
        )
        try:
            await _save_corpus(first, corpus)
            admitted = await _admit_run(first, _request(corpus))
            assert admitted.status is EvalRunStatus.QUEUED
            claimed = await first.claim_run()
            assert claimed is not None
            completed = await _publish_result(first, claimed.claim, result)
            assert completed.status is EvalRunStatus.COMPLETED
        finally:
            await first.close()

        reopened = PostgresEvalStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.VALIDATE,
        )
        try:
            assert await reopened.load_corpus(corpus.revision) == corpus
            assert await reopened.load_run(completed.id) == completed
            assert await reopened.load_result(completed.id) == result
            assert await _admit_run(reopened, _request(corpus, run_id="retry-id")) == completed
        finally:
            await reopened.close()

    asyncio.run(exercise())


def test_postgres_eval_store_serializes_concurrent_logical_admission(postgres_dsn) -> None:
    async def exercise() -> None:
        from cayu.storage.evals_postgres import PostgresEvalStore
        from cayu.storage.migrations import SchemaMode

        await _drop_eval_tables(postgres_dsn)
        corpus = _corpus()
        setup = PostgresEvalStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.MIGRATE,
        )
        await _save_corpus(setup, corpus)
        await setup.close()

        left = PostgresEvalStore(postgres_dsn, schema_mode=SchemaMode.VALIDATE)
        right = PostgresEvalStore(postgres_dsn, schema_mode=SchemaMode.VALIDATE)
        try:
            first, second = await asyncio.gather(
                _admit_run(left, _request(corpus, run_id="run-left")),
                _admit_run(right, _request(corpus, run_id="run-right")),
            )
            assert first == second
            assert first.id in {"run-left", "run-right"}
            claims = await asyncio.gather(
                left.claim_run(),
                right.claim_run(),
            )
            assert sum(claim is not None for claim in claims) == 1
        finally:
            await left.close()
            await right.close()

    asyncio.run(exercise())


def test_postgres_eval_store_cancels_expired_claim_without_requeue(postgres_dsn) -> None:
    async def exercise() -> None:
        from cayu.storage.evals_postgres import PostgresEvalStore
        from cayu.storage.migrations import SchemaMode

        await _drop_eval_tables(postgres_dsn)
        store = PostgresEvalStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.MIGRATE,
        )
        corpus = _corpus()
        try:
            await _save_corpus(store, corpus)
            await _admit_run(store, _request(corpus))
            lease = await store.claim_run(lease_seconds=1)
            assert lease is not None
            await asyncio.sleep(1.05)

            cancelled = await store.request_cancel(lease.run.id)
            assert cancelled.status is EvalRunStatus.CANCELLED
            assert cancelled.ownership is None
            assert await store.claim_run() is None
        finally:
            await store.close()

    asyncio.run(exercise())
