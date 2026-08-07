from __future__ import annotations

import asyncio
import sqlite3
import threading
from contextlib import suppress

import pytest
from tests.evals.eval_store_conformance import (
    assert_eval_store_conformance,
    assert_eval_store_reconstruction_releases_heartbeat_capacity,
)
from tests.evals.test_corpus_execution import _corpus, _provider, _target

import cayu.storage.evals_sqlite as evals_sqlite_module
from cayu.evals.execution import run_corpus_suite
from cayu.evals.store import EvalRunClaim, EvalRunRecord, EvalRunRequest, EvalRunStatus
from cayu.storage.evals_sqlite import SQLiteEvalStore
from cayu.storage.migrations import SchemaMode
from cayu.vaults.redaction import SecretRedactor

_NO_SECRETS = SecretRedactor()


async def _save_corpus(store, corpus):
    return await store.save_corpus(
        corpus,
        redact_json=_NO_SECRETS.redact_json,
    )


async def _admit_run(store, request):
    return await store.admit_run(
        request,
        redact_json=_NO_SECRETS.redact_json,
    )


async def _publish_result(store, claim, result):
    return await store.publish_result(
        claim,
        result,
        redact_json=_NO_SECRETS.redact_json,
    )


def _request(
    corpus,
    *,
    run_id: str = "run-1",
    idempotency_digit: str = "1",
) -> EvalRunRequest:
    suite = corpus.suites[0]
    return EvalRunRequest(
        run_id=run_id,
        idempotency_key="sha256:" + idempotency_digit * 64,
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


def test_sqlite_eval_store_creates_revision_thirty_three_schema(tmp_path) -> None:
    path = tmp_path / "evals.db"

    async def initialize() -> None:
        store = SQLiteEvalStore(path)
        await store.close()

    asyncio.run(initialize())
    connection = sqlite3.connect(path)
    try:
        revision = connection.execute(
            "SELECT kind, compatible_from FROM cayu_schema_migrations WHERE revision = 33"
        ).fetchone()
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'cayu_eval_%'"
            ).fetchall()
        }
        indexes = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index' "
                "AND name LIKE 'idx_cayu_eval_runs_target_%'"
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
    assert indexes == {
        "idx_cayu_eval_runs_target_catalog",
        "idx_cayu_eval_runs_target_status_claim",
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


@pytest.mark.parametrize(
    ("read_kind", "parser_name"),
    [
        ("corpus", "eval_corpus_from_json"),
        ("result", "corpus_execution_result_from_json"),
    ],
)
def test_sqlite_eval_reconstruction_does_not_occupy_heartbeat_capacity(
    tmp_path,
    monkeypatch,
    read_kind: str,
    parser_name: str,
) -> None:
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
            await assert_eval_store_reconstruction_releases_heartbeat_capacity(
                store,
                corpus=corpus,
                result=result,
                read_kind=read_kind,
                parser_owner=evals_sqlite_module,
                parser_name=parser_name,
                monkeypatch=monkeypatch,
            )
        finally:
            await store.close()

    asyncio.run(exercise())


def test_sqlite_result_validation_does_not_block_live_claim_heartbeats(
    tmp_path,
    monkeypatch,
) -> None:
    async def exercise() -> None:
        corpus = _corpus(trials=1)
        result = await run_corpus_suite(
            _target(_provider(trials=1)),
            corpus,
            corpus.suites[0].id,
            max_concurrency=1,
        )
        path = tmp_path / "evals.db"
        publishing_store = SQLiteEvalStore(path)
        unrelated_store = SQLiteEvalStore(path, schema_mode=SchemaMode.VALIDATE)
        validation_started = threading.Event()
        release_validation = threading.Event()
        stop_heartbeats = asyncio.Event()
        original_validate = evals_sqlite_module.validate_result_for_run
        heartbeat_counts = {"publishing": 0, "unrelated": 0}
        publication: asyncio.Task[EvalRunRecord] | None = None
        heartbeats: list[asyncio.Task[None]] = []

        def blocking_validate(*args, **kwargs):
            validation_started.set()
            if not release_validation.wait(timeout=5):
                raise AssertionError("Timed out releasing SQLite eval result validation.")
            return original_validate(*args, **kwargs)

        async def maintain_claim(
            store: SQLiteEvalStore,
            claim: EvalRunClaim,
            counter: str,
        ) -> None:
            while not stop_heartbeats.is_set():
                await store.heartbeat_run(claim, extend_seconds=1)
                heartbeat_counts[counter] += 1
                with suppress(TimeoutError):
                    await asyncio.wait_for(stop_heartbeats.wait(), timeout=0.1)

        try:
            await _save_corpus(publishing_store, corpus)
            await _admit_run(
                publishing_store,
                _request(corpus, run_id="publishing-run", idempotency_digit="1"),
            )
            publishing_lease = await publishing_store.claim_run(lease_seconds=1)
            assert publishing_lease is not None
            await _admit_run(
                unrelated_store,
                _request(corpus, run_id="unrelated-run", idempotency_digit="2"),
            )
            unrelated_lease = await unrelated_store.claim_run(lease_seconds=1)
            assert unrelated_lease is not None

            monkeypatch.setattr(
                evals_sqlite_module,
                "validate_result_for_run",
                blocking_validate,
            )
            publication = asyncio.create_task(
                _publish_result(publishing_store, publishing_lease.claim, result)
            )
            assert await asyncio.to_thread(validation_started.wait, 2)
            heartbeats = [
                asyncio.create_task(
                    maintain_claim(
                        publishing_store,
                        publishing_lease.claim,
                        "publishing",
                    )
                ),
                asyncio.create_task(
                    maintain_claim(unrelated_store, unrelated_lease.claim, "unrelated")
                ),
            ]

            await asyncio.sleep(1.2)
            assert heartbeat_counts["publishing"] >= 4
            assert heartbeat_counts["unrelated"] >= 4

            stop_heartbeats.set()
            await asyncio.wait_for(asyncio.gather(*heartbeats), timeout=2)
            release_validation.set()
            completed = await asyncio.wait_for(publication, timeout=2)
            assert completed.status is EvalRunStatus.COMPLETED
            await unrelated_store.release_run(unrelated_lease.claim)
        finally:
            stop_heartbeats.set()
            release_validation.set()
            tasks = tuple(task for task in (publication, *heartbeats) if task is not None)
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            await unrelated_store.close()
            await publishing_store.close()

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
