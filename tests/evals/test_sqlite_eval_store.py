from __future__ import annotations

import asyncio
import sqlite3

import pytest
from tests.evals.eval_store_conformance import assert_eval_store_conformance
from tests.evals.test_corpus_execution import _corpus, _provider, _target

from cayu.evals.execution import run_corpus_suite
from cayu.evals.store import EvalRunRequest, EvalRunStatus
from cayu.storage.evals_sqlite import SQLiteEvalStore
from cayu.storage.migrations import SchemaMode
from cayu.vaults.redaction import SecretRedactor

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


def test_sqlite_eval_store_shared_conformance(tmp_path) -> None:
    async def exercise() -> None:
        corpus = _corpus(trials=1)
        result = await run_corpus_suite(
            _target(_provider(trials=1)),
            corpus,
            corpus.suites[0].id,
            max_concurrency=1,
        )
        store = SQLiteEvalStore(tmp_path / "evals.db")
        try:
            await assert_eval_store_conformance(store, corpus=corpus, result=result)
        finally:
            await store.close()

    asyncio.run(exercise())


def test_sqlite_eval_store_creates_revision_thirty_two_schema(tmp_path) -> None:
    path = tmp_path / "evals.db"

    async def initialize() -> None:
        store = SQLiteEvalStore(path)
        await store.close()

    asyncio.run(initialize())
    connection = sqlite3.connect(path)
    try:
        revision = connection.execute(
            "SELECT kind, compatible_from FROM cayu_schema_migrations WHERE revision = 32"
        ).fetchone()
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'cayu_eval_%'"
            ).fetchall()
        }
    finally:
        connection.close()
    assert revision == ("additive", 31)
    assert tables == {
        "cayu_eval_cases",
        "cayu_eval_corpora",
        "cayu_eval_results",
        "cayu_eval_runs",
        "cayu_eval_suites",
    }


def test_sqlite_eval_store_is_restart_durable_and_idempotent(tmp_path) -> None:
    async def exercise() -> None:
        path = tmp_path / "evals.db"
        corpus = _corpus(trials=1)
        result = await run_corpus_suite(
            _target(_provider(trials=1)),
            corpus,
            corpus.suites[0].id,
            max_concurrency=1,
        )
        first = SQLiteEvalStore(path)
        await _save_corpus(first, corpus)
        admitted = await _admit_run(first, _request(corpus))
        assert admitted.status is EvalRunStatus.QUEUED
        claimed = await first.claim_run()
        assert claimed is not None
        completed = await _publish_result(first, claimed.claim, result)
        assert completed.status is EvalRunStatus.COMPLETED
        await first.close()

        reopened = SQLiteEvalStore(path, schema_mode=SchemaMode.VALIDATE)
        assert await reopened.load_corpus(corpus.revision) == corpus
        assert await reopened.load_run(completed.id) == completed
        assert await reopened.load_result(completed.id) == result
        assert await _admit_run(reopened, _request(corpus, run_id="retry-id")) == completed
        await reopened.close()

    asyncio.run(exercise())


def test_sqlite_eval_store_serializes_concurrent_logical_admission(tmp_path) -> None:
    async def exercise() -> None:
        path = tmp_path / "evals.db"
        corpus = _corpus()
        setup = SQLiteEvalStore(path)
        await _save_corpus(setup, corpus)
        await setup.close()

        left = SQLiteEvalStore(path, schema_mode=SchemaMode.VALIDATE)
        right = SQLiteEvalStore(path, schema_mode=SchemaMode.VALIDATE)
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


def test_sqlite_eval_store_cancels_expired_claim_without_requeue(tmp_path) -> None:
    async def exercise() -> None:
        store = SQLiteEvalStore(tmp_path / "evals.db")
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


def test_sqlite_eval_store_rolls_back_interrupted_corpus_projection(tmp_path) -> None:
    async def exercise() -> None:
        path = tmp_path / "evals.db"
        store = SQLiteEvalStore(path)
        first = _corpus(input_text="preserved")
        interrupted = _corpus(input_text="must roll back")
        try:
            await _save_corpus(store, first)
            connection = sqlite3.connect(path)
            try:
                connection.execute(
                    """
                    CREATE TRIGGER abort_eval_suite_projection
                    BEFORE INSERT ON cayu_eval_suites
                    BEGIN
                        SELECT RAISE(ABORT, 'simulated projection interruption');
                    END
                    """
                )
                connection.commit()
            finally:
                connection.close()

            with pytest.raises(sqlite3.IntegrityError, match="simulated projection interruption"):
                await _save_corpus(store, interrupted)
            assert await store.load_corpus(first.revision) == first
            assert await store.load_corpus(interrupted.revision) is None
        finally:
            await store.close()

    asyncio.run(exercise())


def test_sqlite_eval_store_rolls_back_interrupted_result_publication(tmp_path) -> None:
    async def exercise() -> None:
        path = tmp_path / "evals.db"
        corpus = _corpus(trials=1)
        result = await run_corpus_suite(
            _target(_provider(trials=1)),
            corpus,
            corpus.suites[0].id,
            max_concurrency=1,
        )
        store = SQLiteEvalStore(path)
        try:
            await _save_corpus(store, corpus)
            await _admit_run(store, _request(corpus))
            lease = await store.claim_run()
            assert lease is not None

            connection = sqlite3.connect(path)
            try:
                connection.execute(
                    """
                    CREATE TRIGGER abort_eval_run_completion
                    BEFORE UPDATE OF status ON cayu_eval_runs
                    WHEN NEW.status = 'completed'
                    BEGIN
                        SELECT RAISE(ABORT, 'simulated publication interruption');
                    END
                    """
                )
                connection.commit()
            finally:
                connection.close()

            with pytest.raises(sqlite3.IntegrityError, match="simulated publication interruption"):
                await _publish_result(store, lease.claim, result)
            assert await store.load_result(lease.run.id) is None
            still_running = await store.load_run(lease.run.id)
            assert still_running is not None
            assert still_running.status is EvalRunStatus.RUNNING

            connection = sqlite3.connect(path)
            try:
                connection.execute("DROP TRIGGER abort_eval_run_completion")
                connection.commit()
            finally:
                connection.close()
            completed = await _publish_result(store, lease.claim, result)
            assert completed.status is EvalRunStatus.COMPLETED
            assert await store.load_result(lease.run.id) == result
        finally:
            await store.close()

    asyncio.run(exercise())
