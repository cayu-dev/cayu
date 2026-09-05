from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
from contextlib import suppress

import pytest
from pydantic import ValidationError
from tests.evals.eval_store_conformance import (
    _scenario,
    _terminal_trial_checkpoint,
    assert_captured_eval_store_conformance,
    assert_eval_store_conformance,
    assert_eval_store_reconstruction_releases_heartbeat_capacity,
    assert_judge_calibration_store_conformance,
    assert_scenario_progress_conformance,
    captured_result_for_corpus,
)
from tests.evals.test_artifact_assertions import structural_corpus, structural_target
from tests.evals.test_corpus_execution import (
    _corpus,
    _provider,
    _target,
    _tool_json_corpus,
    _tool_json_target,
)
from tests.evals.test_judge_calibration import _calibration_report

import cayu.storage.evals_sqlite as evals_sqlite_module
from cayu.evals.capacity import EVAL_MAX_CONCURRENCY
from cayu.evals.corpus import EvalCorpusDocument
from cayu.evals.execution import (
    CorpusExecutionResult,
    _run_compiled_corpus_suite,
    compile_corpus_suite,
    run_corpus_suite,
)
from cayu.evals.store import (
    EvalBaselineKey,
    EvalBaselineUpdate,
    EvalRunClaim,
    EvalRunClaimLost,
    EvalRunFailureCode,
    EvalRunInvocation,
    EvalRunRecord,
    EvalRunRequest,
    EvalRunStatus,
    EvalRunTrialCheckpoint,
    EvalStoreTransientContention,
)
from cayu.storage import _sqlite_support as sqlite_support
from cayu.storage import migrations as schema_migrations
from cayu.storage.evals_sqlite import SQLiteEvalStore, SQLiteEvalWriterContentionPolicy
from cayu.storage.migrations import SchemaMode, SchemaTooOld
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
    max_concurrency: int = 1,
) -> EvalRunRequest:
    suite = corpus.suites[0]
    return EvalRunRequest(
        run_id=run_id,
        idempotency_key="sha256:" + idempotency_digit * 64,
        corpus_revision=corpus.revision,
        target_key=corpus.target_key,
        suite_id=suite.id,
        suite_revision=suite.revision,
        max_concurrency=max_concurrency,
    )


def test_sqlite_eval_store_durably_admits_operator_selected_concurrency(tmp_path) -> None:
    async def exercise() -> None:
        corpus = _corpus(trials=1)
        store = SQLiteEvalStore(tmp_path / "evals.db")
        try:
            await _save_corpus(store, corpus)
            admitted = await _admit_run(
                store,
                _request(corpus, max_concurrency=EVAL_MAX_CONCURRENCY),
            )
            assert admitted.spec.max_concurrency == EVAL_MAX_CONCURRENCY
        finally:
            await store.close()

    asyncio.run(exercise())
    with pytest.raises(ValidationError, match="less than or equal"):
        _request(_corpus(trials=1), max_concurrency=EVAL_MAX_CONCURRENCY + 1)


def _fast_contention_policy(*, max_wait_seconds: float = 2.0):
    return SQLiteEvalWriterContentionPolicy(
        max_wait_seconds=max_wait_seconds,
        lock_attempt_seconds=0.02,
        initial_backoff_seconds=0.005,
        max_backoff_seconds=0.02,
    )


async def _result_with_checkpoint(corpus):
    target = _target(_provider(trials=1))
    retained = []

    async def capture(case_id, result, public_data) -> None:
        retained.append(
            EvalRunTrialCheckpoint(
                case_id=case_id,
                result=result,
                public_data=public_data,
            )
        )

    result = await _run_compiled_corpus_suite(
        target,
        compile_corpus_suite(corpus, target, corpus.suites[0].id),
        max_concurrency=1,
        trial_completed=capture,
    )
    assert len(retained) == 1
    return result, retained[0]


def test_sqlite_run_observation_bypasses_full_rehydration_and_writer_queue(
    tmp_path,
    monkeypatch,
) -> None:
    async def exercise() -> None:
        corpus = _corpus(trials=1)
        store = SQLiteEvalStore(tmp_path / "evals.db")
        writer_started = threading.Event()
        release_writer = threading.Event()
        writer = None

        def occupy_writer(_connection) -> None:
            writer_started.set()
            if not release_writer.wait(timeout=5):
                raise AssertionError("Timed out releasing the SQLite eval writer.")

        def reject_invocation_rehydration(_source: str):
            raise AssertionError("immutable invocation was rehydrated")

        try:
            await _save_corpus(store, corpus)
            await _admit_run(store, _request(corpus))
            lease = await store.claim_run()
            assert lease is not None
            monkeypatch.setattr(
                evals_sqlite_module,
                "eval_run_invocation_from_json",
                reject_invocation_rehydration,
            )

            writer = asyncio.create_task(store._run(occupy_writer))
            assert await asyncio.to_thread(writer_started.wait, 2)
            observation = await asyncio.wait_for(
                store.load_run_observation(lease.run.id),
                timeout=0.25,
            )
            assert observation is not None
            assert observation.status is EvalRunStatus.RUNNING
            assert observation.ownership == lease.run.ownership

            release_writer.set()
            await writer
            with pytest.raises(AssertionError, match="immutable invocation"):
                await store.load_run(lease.run.id)
        finally:
            release_writer.set()
            if writer is not None:
                await asyncio.gather(writer, return_exceptions=True)
            await store.close()

    asyncio.run(exercise())


def test_sqlite_terminal_watchers_do_not_starve_checkpoint_or_result_publication(
    tmp_path,
    caplog,
) -> None:
    async def exercise() -> None:
        corpus = _corpus(trials=1)
        result, checkpoint = await _result_with_checkpoint(corpus)
        path = tmp_path / "evals.db"
        owner = SQLiteEvalStore(path)
        observers = []
        waiters = []
        try:
            await _save_corpus(owner, corpus)
            await _admit_run(owner, _request(corpus, run_id="watched-run"))
            lease = await owner.claim_run()
            assert lease is not None
            assert await owner.load_run(lease.run.id) is not None
            observers = [SQLiteEvalStore(path, schema_mode=SchemaMode.VALIDATE) for _ in range(3)]
            stores = [owner, *observers]
            waiters = [
                asyncio.create_task(
                    store.wait_for_run_terminal(
                        lease.run.id,
                        timeout_seconds=2,
                        poll_interval_seconds=0.001,
                        max_poll_interval_seconds=0.05,
                    )
                )
                for store in stores
                for _ in range(8)
            ]
            await asyncio.sleep(0.03)

            checkpoint_started_at = asyncio.get_running_loop().time()
            await owner.save_trial_checkpoint(
                lease.claim,
                checkpoint,
                redact_json=_NO_SECRETS.redact_json,
            )
            assert asyncio.get_running_loop().time() - checkpoint_started_at < 1.0

            publication_started_at = asyncio.get_running_loop().time()
            completed = await _publish_result(owner, lease.claim, result)
            assert asyncio.get_running_loop().time() - publication_started_at < 1.0
            assert completed.status is EvalRunStatus.COMPLETED
            observations = await asyncio.wait_for(asyncio.gather(*waiters), timeout=2)
            assert all(
                observation is not None and observation.status is EvalRunStatus.COMPLETED
                for observation in observations
            )
        finally:
            if waiters:
                await asyncio.gather(*waiters, return_exceptions=True)
            await asyncio.gather(*(store.close() for store in observers))
            await owner.close()

    caplog.set_level(logging.DEBUG)
    asyncio.run(exercise())
    events = {getattr(record, "cayu_eval_store_event", None) for record in caplog.records}
    assert {
        "run_status_read",
        "full_run_rehydration",
        "checkpoint_write",
        "result_publication",
    } <= events


def test_sqlite_eval_writers_retry_across_store_instances(tmp_path, caplog) -> None:
    async def exercise() -> None:
        corpus = _corpus(trials=1)
        result, checkpoint = await _result_with_checkpoint(corpus)
        path = tmp_path / "evals.db"
        stores = [SQLiteEvalStore(path, writer_contention_policy=_fast_contention_policy())]
        blocker = sqlite3.connect(path)
        checkpoint_tasks = []
        publication_tasks = []
        try:
            await _save_corpus(stores[0], corpus)
            stores.extend(
                SQLiteEvalStore(
                    path,
                    schema_mode=SchemaMode.VALIDATE,
                    writer_contention_policy=_fast_contention_policy(),
                )
                for _ in range(3)
            )
            leases = []
            for index, store in enumerate(stores, start=1):
                await _admit_run(
                    store,
                    _request(
                        corpus,
                        run_id=f"contended-run-{index}",
                        idempotency_digit=str(index),
                    ),
                )
                lease = await store.claim_run(target_key=corpus.target_key)
                assert lease is not None
                leases.append(lease)

            blocker.execute("BEGIN IMMEDIATE")
            checkpoint_tasks = [
                asyncio.create_task(
                    store.save_trial_checkpoint(
                        lease.claim,
                        checkpoint,
                        redact_json=_NO_SECRETS.redact_json,
                    )
                )
                for store, lease in zip(stores, leases, strict=True)
            ]
            await asyncio.sleep(0.1)
            blocker.commit()
            await asyncio.wait_for(asyncio.gather(*checkpoint_tasks), timeout=2)

            blocker.execute("BEGIN IMMEDIATE")
            publication_tasks = [
                asyncio.create_task(_publish_result(store, lease.claim, result))
                for store, lease in zip(stores, leases, strict=True)
            ]
            await asyncio.sleep(0.1)
            blocker.commit()
            completed = await asyncio.wait_for(asyncio.gather(*publication_tasks), timeout=2)
            assert all(record.status is EvalRunStatus.COMPLETED for record in completed)
            for store, lease in zip(stores, leases, strict=True):
                assert await store.load_result(lease.claim.run_id) == result
        finally:
            blocker.rollback()
            blocker.close()
            tasks = (*checkpoint_tasks, *publication_tasks)
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.gather(*(store.close() for store in reversed(stores)))

    caplog.set_level(logging.INFO, logger=evals_sqlite_module.__name__)
    asyncio.run(exercise())
    events = {getattr(record, "cayu_eval_store_event", None) for record in caplog.records}
    assert "sqlite_writer.lock_wait" in events
    assert "sqlite_writer.retry" in events
    operations = {
        getattr(record, "eval_operation", None)
        for record in caplog.records
        if hasattr(record, "eval_operation")
    }
    assert operations == {"save_trial_checkpoint", "publish_result"}


def test_sqlite_eval_writer_contention_is_bounded_and_retryable(tmp_path, caplog) -> None:
    async def exercise() -> None:
        corpus = _corpus(trials=1)
        result, checkpoint = await _result_with_checkpoint(corpus)
        path = tmp_path / "evals.db"
        store = SQLiteEvalStore(
            path,
            writer_contention_policy=_fast_contention_policy(max_wait_seconds=0.08),
        )
        blocker = sqlite3.connect(path)
        try:
            await _save_corpus(store, corpus)
            await _admit_run(store, _request(corpus))
            lease = await store.claim_run()
            assert lease is not None
            await store.save_trial_checkpoint(
                lease.claim,
                checkpoint,
                redact_json=_NO_SECRETS.redact_json,
            )

            blocker.execute("BEGIN IMMEDIATE")
            with pytest.raises(EvalStoreTransientContention, match="bounded"):
                await asyncio.wait_for(
                    _publish_result(store, lease.claim, result),
                    timeout=0.5,
                )
            blocker.commit()

            running = await store.load_run(lease.claim.run_id)
            assert running is not None
            assert running.status is EvalRunStatus.RUNNING
            assert await store.load_trial_checkpoints(lease.claim) == (checkpoint,)
            completed = await _publish_result(store, lease.claim, result)
            assert completed.status is EvalRunStatus.COMPLETED
        finally:
            blocker.rollback()
            blocker.close()
            await store.close()

    caplog.set_level(logging.INFO, logger=evals_sqlite_module.__name__)
    asyncio.run(exercise())
    assert "sqlite_writer.contention_exhausted" in {
        getattr(record, "cayu_eval_store_event", None) for record in caplog.records
    }


def test_sqlite_eval_writer_contention_budget_includes_store_queue(tmp_path, caplog) -> None:
    async def exercise() -> None:
        corpus = _corpus(trials=1)
        store = SQLiteEvalStore(
            tmp_path / "evals.db",
            writer_contention_policy=_fast_contention_policy(max_wait_seconds=0.08),
        )
        writer_started = threading.Event()
        release_writer = threading.Event()
        writer = None

        def occupy_writer(_connection) -> None:
            writer_started.set()
            if not release_writer.wait(timeout=5):
                raise AssertionError("Timed out releasing the SQLite eval writer.")

        try:
            await _save_corpus(store, corpus)
            await _admit_run(store, _request(corpus))
            lease = await store.claim_run()
            assert lease is not None
            writer = asyncio.create_task(store._run(occupy_writer))
            assert await asyncio.to_thread(writer_started.wait, 2)

            started_at = asyncio.get_running_loop().time()
            with pytest.raises(EvalStoreTransientContention, match="bounded"):
                await asyncio.wait_for(
                    store.save_trial_checkpoint(
                        lease.claim,
                        _terminal_trial_checkpoint(corpus),
                        redact_json=_NO_SECRETS.redact_json,
                    ),
                    timeout=0.4,
                )
            assert asyncio.get_running_loop().time() - started_at < 0.3
            assert not writer.done()
        finally:
            release_writer.set()
            if writer is not None:
                await asyncio.gather(writer, return_exceptions=True)
            await store.close()

    caplog.set_level(logging.INFO, logger=evals_sqlite_module.__name__)
    asyncio.run(exercise())
    assert "sqlite_writer.contention_exhausted" in {
        getattr(record, "cayu_eval_store_event", None) for record in caplog.records
    }


def test_sqlite_eval_heartbeat_retries_writer_contention(tmp_path, caplog) -> None:
    async def exercise() -> None:
        corpus = _corpus(trials=1)
        path = tmp_path / "evals.db"
        store = SQLiteEvalStore(path, writer_contention_policy=_fast_contention_policy())
        blocker = sqlite3.connect(path)
        heartbeat = None
        try:
            await _save_corpus(store, corpus)
            await _admit_run(store, _request(corpus))
            lease = await store.claim_run(lease_seconds=30)
            assert lease is not None
            assert lease.run.ownership is not None

            blocker.execute("BEGIN IMMEDIATE")
            heartbeat = asyncio.create_task(store.heartbeat_run(lease.claim, extend_seconds=30))
            await asyncio.sleep(0.1)
            blocker.commit()
            renewed = await asyncio.wait_for(heartbeat, timeout=1)
            assert renewed.ownership is not None
            assert renewed.ownership.epoch == lease.claim.epoch
            assert renewed.ownership.lease_expires_at > lease.run.ownership.lease_expires_at
        finally:
            blocker.rollback()
            blocker.close()
            if heartbeat is not None:
                await asyncio.gather(heartbeat, return_exceptions=True)
            await store.close()

    caplog.set_level(logging.INFO, logger=evals_sqlite_module.__name__)
    asyncio.run(exercise())
    events = {
        getattr(record, "cayu_eval_store_event", None)
        for record in caplog.records
        if getattr(record, "eval_operation", None) == "heartbeat_run"
    }
    assert {"sqlite_writer.lock_wait", "sqlite_writer.retry"} <= events


def test_sqlite_eval_writer_contention_wait_is_cancellation_aware(tmp_path) -> None:
    async def exercise() -> None:
        corpus = _corpus(trials=1)
        path = tmp_path / "evals.db"
        store = SQLiteEvalStore(
            path,
            writer_contention_policy=_fast_contention_policy(),
        )
        blocker = sqlite3.connect(path)
        publication = None
        try:
            await _save_corpus(store, corpus)
            await _admit_run(store, _request(corpus))
            lease = await store.claim_run()
            assert lease is not None
            blocker.execute("BEGIN IMMEDIATE")
            publication = asyncio.create_task(
                store.save_trial_checkpoint(
                    lease.claim,
                    _terminal_trial_checkpoint(corpus),
                    redact_json=_NO_SECRETS.redact_json,
                )
            )
            await asyncio.sleep(0.05)
            publication.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(publication, timeout=0.2)
            blocker.commit()
            running = await store.load_run(lease.claim.run_id)
            assert running is not None
            assert running.status is EvalRunStatus.RUNNING
            assert await store.load_trial_checkpoints(lease.claim) == ()
        finally:
            blocker.rollback()
            blocker.close()
            if publication is not None:
                await asyncio.gather(publication, return_exceptions=True)
            await store.close()

    asyncio.run(exercise())


def test_sqlite_eval_writer_revalidates_claim_after_lock_wait(tmp_path, caplog) -> None:
    async def exercise() -> None:
        corpus = _corpus(trials=1)
        path = tmp_path / "evals.db"
        store = SQLiteEvalStore(
            path,
            writer_contention_policy=_fast_contention_policy(),
        )
        blocker = sqlite3.connect(path)
        publication = None
        try:
            await _save_corpus(store, corpus)
            await _admit_run(store, _request(corpus))
            lease = await store.claim_run(lease_seconds=1)
            assert lease is not None
            blocker.execute("BEGIN IMMEDIATE")
            publication = asyncio.create_task(
                store.save_trial_checkpoint(
                    lease.claim,
                    _terminal_trial_checkpoint(corpus),
                    redact_json=_NO_SECRETS.redact_json,
                )
            )
            await asyncio.sleep(1.05)
            blocker.commit()
            with pytest.raises(EvalRunClaimLost, match="expired"):
                await asyncio.wait_for(publication, timeout=0.5)
        finally:
            blocker.rollback()
            blocker.close()
            if publication is not None:
                await asyncio.gather(publication, return_exceptions=True)
            await store.close()

    caplog.set_level(logging.INFO, logger=evals_sqlite_module.__name__)
    asyncio.run(exercise())
    assert "sqlite_writer.claim_lost" in {
        getattr(record, "cayu_eval_store_event", None) for record in caplog.records
    }


def test_sqlite_eval_store_shared_conformance(tmp_path) -> None:
    async def exercise() -> None:
        corpus = _corpus(trials=1)
        result = await run_corpus_suite(
            _target(_provider(trials=1)),
            corpus,
            corpus.suites[0].id,
            max_concurrency=1,
        )
        path = tmp_path / "evals.db"
        store = SQLiteEvalStore(path)
        tool_corpus = _tool_json_corpus(target_key="tool-json-agent")
        tool_result = await run_corpus_suite(
            _tool_json_target(
                query="cayu",
                limit=5,
                count=2,
                application_release_id="release-tool-store",
                target_key=tool_corpus.target_key,
            ),
            tool_corpus,
            tool_corpus.suites[0].id,
        )
        structure_corpus = structural_corpus(target_key="structural-store-agent")
        structure_result = await run_corpus_suite(
            structural_target(
                tmp_path / "structural-runtime",
                target_key=structure_corpus.target_key,
            ),
            structure_corpus,
            structure_corpus.suites[0].id,
        )
        try:
            await assert_eval_store_conformance(store, corpus=corpus, result=result)
            await assert_captured_eval_store_conformance(
                store,
                corpus=corpus,
                result=result,
            )
            await assert_scenario_progress_conformance(store, corpus=corpus)
            await assert_judge_calibration_store_conformance(
                store,
                report=await _calibration_report(),
            )
            await _save_corpus(store, tool_corpus)
            tool_request = _request(
                tool_corpus,
                run_id="tool-json-restart",
                idempotency_digit="0",
            )
            await _admit_run(store, tool_request)
            tool_lease = await store.claim_run(target_key=tool_corpus.target_key)
            assert tool_lease is not None
            await _publish_result(store, tool_lease.claim, tool_result)
            await _save_corpus(store, structure_corpus)
            structure_request = _request(
                structure_corpus,
                run_id="structural-restart",
                idempotency_digit="2",
            )
            await _admit_run(store, structure_request)
            structure_lease = await store.claim_run(target_key=structure_corpus.target_key)
            assert structure_lease is not None
            await _publish_result(store, structure_lease.claim, structure_result)
        finally:
            await store.close()

        restarted = SQLiteEvalStore(path)
        try:
            assert await restarted.load_corpus(tool_corpus.revision) == tool_corpus
            assert await restarted.load_result("tool-json-restart") == tool_result
            assert await restarted.load_corpus(structure_corpus.revision) == structure_corpus
            assert await restarted.load_result("structural-restart") == structure_result
        finally:
            await restarted.close()

    asyncio.run(exercise())


def test_sqlite_trial_checkpoints_are_normalized_and_cleared_atomically(tmp_path) -> None:
    path = tmp_path / "evals.db"

    async def exercise() -> None:
        corpus = _corpus(trials=1)
        store = SQLiteEvalStore(path)
        try:
            await _save_corpus(store, corpus)
            await _admit_run(store, _request(corpus))
            lease = await store.claim_run()
            assert lease is not None
            checkpoint = _terminal_trial_checkpoint(corpus)
            await store.save_trial_checkpoint(
                lease.claim,
                checkpoint,
                redact_json=_NO_SECRETS.redact_json,
            )

            connection = sqlite3.connect(path)
            try:
                count, total_bytes = connection.execute(
                    "SELECT trial_checkpoint_count, trial_checkpoint_bytes "
                    "FROM cayu_eval_runs WHERE run_id = ?",
                    (lease.claim.run_id,),
                ).fetchone()
                rows = connection.execute(
                    "SELECT case_id, trial_number, checkpoint_json, document_bytes "
                    "FROM cayu_eval_run_trial_checkpoints WHERE run_id = ?",
                    (lease.claim.run_id,),
                ).fetchall()
            finally:
                connection.close()
            assert count == 1
            assert len(rows) == 1
            assert rows[0][0:2] == (checkpoint.case_id, checkpoint.trial_number)
            assert rows[0][2].startswith("{")
            assert rows[0][3] == total_bytes == len(rows[0][2].encode("utf-8"))

            await store.fail_run(lease.claim, EvalRunFailureCode.EXECUTION_FAILED)
            connection = sqlite3.connect(path)
            try:
                assert connection.execute(
                    "SELECT trial_checkpoint_count, trial_checkpoint_bytes "
                    "FROM cayu_eval_runs WHERE run_id = ?",
                    (lease.claim.run_id,),
                ).fetchone() == (0, 0)
                assert connection.execute(
                    "SELECT COUNT(*) FROM cayu_eval_run_trial_checkpoints WHERE run_id = ?",
                    (lease.claim.run_id,),
                ).fetchone() == (0,)
            finally:
                connection.close()
        finally:
            await store.close()

    asyncio.run(exercise())


def test_sqlite_eval_store_creates_revision_sixty_eight_schema(tmp_path) -> None:
    path = tmp_path / "evals.db"

    async def initialize() -> None:
        store = SQLiteEvalStore(path)
        await store.close()

    asyncio.run(initialize())
    connection = sqlite3.connect(path)
    try:
        revisions = connection.execute(
            "SELECT revision, kind, compatible_from FROM cayu_schema_migrations "
            "WHERE revision BETWEEN 47 AND 68 "
            "ORDER BY revision"
        ).fetchall()
        invocation_column = connection.execute("PRAGMA table_info(cayu_eval_runs)").fetchall()
        case_table = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'cayu_eval_cases'"
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
                "AND (name LIKE 'idx_cayu_eval_runs_target_%' "
                "OR name LIKE 'idx_cayu_eval_result_records_%' "
                "OR name LIKE 'idx_cayu_eval_scenarios_%' "
                "OR name LIKE 'idx_cayu_eval_authored_suites_%' "
                "OR name LIKE 'idx_cayu_eval_judge_calibrations_%' "
                "OR name = 'idx_cayu_eval_baseline_mutations_scope')"
            ).fetchall()
        }
    finally:
        connection.close()
    assert revisions == [
        (47, "breaking", 47),
        (48, "breaking", 48),
        (49, "breaking", 49),
        (50, "breaking", 50),
        (51, "additive", 50),
        (52, "breaking", 52),
        (53, "additive", 52),
        (54, "breaking", 54),
        (55, "breaking", 55),
        (56, "additive", 55),
        (57, "breaking", 57),
        (58, "breaking", 58),
        (59, "breaking", 59),
        (60, "breaking", 60),
        (61, "breaking", 61),
        (62, "breaking", 62),
        (63, "breaking", 63),
        (64, "additive", 63),
        (65, "breaking", 65),
        (66, "breaking", 66),
        (67, "breaking", 67),
        (68, "additive", 67),
    ]
    assert next(row for row in invocation_column if row[1] == "invocation_json")[2:4] == (
        "TEXT",
        1,
    )
    assert next(row for row in invocation_column if row[1] == "scenario_progress_json")[2:4] == (
        "TEXT",
        0,
    )
    assert case_table is not None
    normalized_case_table = "".join(case_table[0].lower().split())
    assert "check(message_count>=0andmessage_count<=16)" in normalized_case_table
    assert tables == {
        "cayu_eval_baseline_mutations",
        "cayu_eval_baselines",
        "cayu_eval_authored_suites",
        "cayu_eval_judge_calibrations",
        "cayu_eval_cases",
        "cayu_eval_corpora",
        "cayu_eval_result_records",
        "cayu_eval_results",
        "cayu_eval_run_trial_checkpoints",
        "cayu_eval_runs",
        "cayu_eval_scenarios",
        "cayu_eval_suites",
    }
    assert indexes == {
        "idx_cayu_eval_baseline_mutations_scope",
        "idx_cayu_eval_authored_suites_catalog",
        "idx_cayu_eval_authored_suites_id_catalog",
        "idx_cayu_eval_authored_suites_target_catalog",
        "idx_cayu_eval_judge_calibrations_definition",
        "idx_cayu_eval_judge_calibrations_target",
        "idx_cayu_eval_result_records_contract",
        "idx_cayu_eval_result_records_target_catalog",
        "idx_cayu_eval_runs_target_catalog",
        "idx_cayu_eval_runs_target_status_claim",
        "idx_cayu_eval_scenarios_catalog",
        "idx_cayu_eval_scenarios_id_catalog",
        "idx_cayu_eval_scenarios_target_catalog",
    }


def test_sqlite_eval_store_migrates_empty_revision_fifty_six_without_verifier_profiles(
    tmp_path,
) -> None:
    path = tmp_path / "evals-revision-56.db"
    connection = sqlite_support.connect(path)
    revisions = schema_migrations.REVISIONS
    try:
        schema_migrations.REVISIONS = tuple(
            revision for revision in revisions if revision.revision <= 56
        )
        sqlite_support.reconcile_schema(
            connection,
            SchemaMode.MIGRATE,
            app_min_supported=56,
        )
    finally:
        schema_migrations.REVISIONS = revisions
        connection.close()

    async def validate() -> None:
        store = SQLiteEvalStore(path, schema_mode=SchemaMode.MIGRATE)
        await store.close()

    asyncio.run(validate())
    connection = sqlite3.connect(path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone() == (
            schema_migrations.LATEST_REVISION,
        )
        assert connection.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type = 'table' AND name = 'cayu_completion_verifier_profiles'"
        ).fetchone() == (1,)
    finally:
        connection.close()


def test_sqlite_eval_store_requires_current_schema(
    tmp_path,
) -> None:
    path = tmp_path / "evals-revision-56-validate.db"
    connection = sqlite_support.connect(path)
    revisions = schema_migrations.REVISIONS
    try:
        schema_migrations.REVISIONS = tuple(
            revision for revision in revisions if revision.revision <= 56
        )
        sqlite_support.reconcile_schema(
            connection,
            SchemaMode.MIGRATE,
            app_min_supported=56,
        )
    finally:
        schema_migrations.REVISIONS = revisions
        connection.close()

    with pytest.raises(SchemaTooOld, match="requires >= 80"):
        SQLiteEvalStore(path, schema_mode=SchemaMode.VALIDATE)


def test_sqlite_eval_store_requires_and_migrates_revision_seventy_four(
    tmp_path,
    monkeypatch,
) -> None:
    async def exercise() -> None:
        path = tmp_path / "evals-revision-73.db"
        corpus = _corpus(trials=1)
        result = await run_corpus_suite(
            _target(_provider(trials=1)),
            corpus,
            corpus.suites[0].id,
            max_concurrency=1,
        )

        with monkeypatch.context() as legacy:
            legacy.setattr(
                schema_migrations,
                "REVISIONS",
                tuple(
                    revision for revision in schema_migrations.REVISIONS if revision.revision <= 73
                ),
            )
            legacy.setattr(
                evals_sqlite_module,
                "_SQLITE_EVAL_MIN_REQUIRED_REVISION",
                72,
            )
            old_store = SQLiteEvalStore(path, schema_mode=SchemaMode.MIGRATE)
            try:
                await _save_corpus(old_store, corpus)
            finally:
                await old_store.close()

        old_request = _request(corpus, run_id="old-run")
        result_document = json.dumps(
            result.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        timestamp = "2026-08-30T00:00:00+00:00"
        connection = sqlite3.connect(path)
        try:
            connection.execute(
                """
                INSERT INTO cayu_eval_runs (
                    run_id, idempotency_key, corpus_revision, target_key,
                    suite_id, suite_revision, max_concurrency, invocation_json,
                    status, created_at, updated_at, started_at, finished_at,
                    result_revision, result_status, result_score, result_duration_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    old_request.run_id,
                    old_request.idempotency_key,
                    old_request.corpus_revision,
                    old_request.target_key,
                    old_request.suite_id,
                    old_request.suite_revision,
                    old_request.max_concurrency,
                    old_request.invocation.model_dump_json(),
                    "completed",
                    timestamp,
                    timestamp,
                    timestamp,
                    timestamp,
                    result.revision,
                    result.run.status,
                    result.run.score,
                    result.run.duration_ms,
                ),
            )
            connection.execute(
                """
                INSERT INTO cayu_eval_results (
                    run_id, revision, result_json, result_bytes, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    old_request.run_id,
                    result.revision,
                    result_document,
                    len(result_document.encode("utf-8")),
                    timestamp,
                ),
            )
            connection.commit()
        finally:
            connection.close()

        with pytest.raises(SchemaTooOld, match="requires >= 80"):
            SQLiteEvalStore(path, schema_mode=SchemaMode.VALIDATE)

        migrated = SQLiteEvalStore(path, schema_mode=SchemaMode.MIGRATE)
        try:
            admitted = await _admit_run(
                migrated,
                _request(
                    corpus,
                    run_id="wide-run",
                    idempotency_digit="2",
                    max_concurrency=EVAL_MAX_CONCURRENCY,
                ),
            )
            assert admitted.spec.max_concurrency == EVAL_MAX_CONCURRENCY
            assert await migrated.load_result("old-run") == result
        finally:
            await migrated.close()

        connection = sqlite3.connect(path)
        try:
            assert connection.execute("PRAGMA user_version").fetchone() == (
                schema_migrations.LATEST_REVISION,
            )
            assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
            foreign_tables = {
                row[2]
                for row in connection.execute(
                    "PRAGMA foreign_key_list(cayu_eval_results)"
                ).fetchall()
            }
            assert "cayu_eval_runs" in foreign_tables
        finally:
            connection.close()

    asyncio.run(exercise())


def test_sqlite_revision_sixty_four_rejects_conflicting_authored_suite_table(
    tmp_path,
) -> None:
    path = tmp_path / "evals.db"

    async def initialize() -> None:
        store = SQLiteEvalStore(path)
        await store.close()

    asyncio.run(initialize())
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            DROP TABLE cayu_eval_authored_suites;
            CREATE TABLE cayu_eval_authored_suites (
                revision TEXT COLLATE BINARY PRIMARY KEY,
                suite_id TEXT COLLATE BINARY NOT NULL,
                suite_revision TEXT NOT NULL,
                target_key TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                case_count INTEGER NOT NULL,
                assertion_count INTEGER NOT NULL,
                simple_input_count INTEGER NOT NULL,
                scenario_count INTEGER NOT NULL,
                trials INTEGER NOT NULL,
                timeout_seconds INTEGER NOT NULL,
                document_json TEXT NOT NULL,
                document_bytes INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            DELETE FROM cayu_schema_migrations WHERE revision = 64;
            PRAGMA user_version = 63;
            """
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="authored suite safety constraints"):
        SQLiteEvalStore(path, schema_mode=SchemaMode.MIGRATE)
    connection = sqlite3.connect(path)
    try:
        assert (
            connection.execute(
                "SELECT 1 FROM cayu_schema_migrations WHERE revision = 64"
            ).fetchone()
            is None
        )
    finally:
        connection.close()


def test_sqlite_revision_sixty_eight_rejects_conflicting_calibration_table(
    tmp_path,
) -> None:
    path = tmp_path / "evals.db"

    async def initialize() -> None:
        store = SQLiteEvalStore(path)
        await store.close()

    asyncio.run(initialize())
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            DROP TABLE cayu_eval_judge_calibrations;
            CREATE TABLE cayu_eval_judge_calibrations (
                revision TEXT COLLATE BINARY PRIMARY KEY,
                run_id TEXT COLLATE BINARY NOT NULL UNIQUE,
                definition_revision TEXT NOT NULL,
                target_key TEXT NOT NULL,
                trial_count INTEGER NOT NULL,
                report_json TEXT NOT NULL,
                document_bytes INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            DELETE FROM cayu_schema_migrations WHERE revision >= 68;
            PRAGMA user_version = 67;
            """
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="calibration safety constraints"):
        SQLiteEvalStore(path, schema_mode=SchemaMode.MIGRATE)
    connection = sqlite3.connect(path)
    try:
        assert (
            connection.execute(
                "SELECT 1 FROM cayu_schema_migrations WHERE revision = 68"
            ).fetchone()
            is None
        )
    finally:
        connection.close()


def test_sqlite_revision_fifty_three_adds_scenarios_without_rewriting_corpora(
    tmp_path,
) -> None:
    path = tmp_path / "evals.db"
    corpus = _corpus(trials=1)

    async def initialize() -> None:
        store = SQLiteEvalStore(path)
        try:
            await _save_corpus(store, corpus)
        finally:
            await store.close()

    asyncio.run(initialize())
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            DROP TABLE cayu_eval_scenarios;
            DELETE FROM cayu_schema_migrations WHERE revision >= 53;
            PRAGMA user_version = 52;
            """
        )
        connection.commit()
    finally:
        connection.close()

    async def migrate() -> None:
        store = SQLiteEvalStore(path, schema_mode=SchemaMode.MIGRATE)
        try:
            assert await store.load_corpus(corpus.revision) == corpus
            assert (await store.list_scenarios()).items == ()
        finally:
            await store.close()

    asyncio.run(migrate())


def test_sqlite_revision_fifty_three_rejects_conflicting_scenario_table(tmp_path) -> None:
    path = tmp_path / "evals.db"

    async def initialize() -> None:
        store = SQLiteEvalStore(path)
        await store.close()

    asyncio.run(initialize())
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            DROP TABLE cayu_eval_scenarios;
            CREATE TABLE cayu_eval_scenarios (
                revision TEXT PRIMARY KEY,
                scenario_id TEXT COLLATE BINARY NOT NULL,
                target_key TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                event_count INTEGER NOT NULL,
                input_event_count INTEGER NOT NULL,
                approval_checkpoint_count INTEGER NOT NULL,
                message_count INTEGER NOT NULL,
                part_count INTEGER NOT NULL,
                artifact_requirement_count INTEGER NOT NULL,
                secret_requirement_count INTEGER NOT NULL,
                document_json TEXT NOT NULL,
                document_bytes INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            DELETE FROM cayu_schema_migrations WHERE revision >= 53;
            PRAGMA user_version = 52;
            """
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="scenario safety constraints"):
        SQLiteEvalStore(path, schema_mode=SchemaMode.MIGRATE)
    connection = sqlite3.connect(path)
    try:
        assert (
            connection.execute(
                "SELECT 1 FROM cayu_schema_migrations WHERE revision = 53"
            ).fetchone()
            is None
        )
    finally:
        connection.close()


def test_sqlite_revision_fifty_three_rejects_unique_scenario_catalog_index(tmp_path) -> None:
    path = tmp_path / "evals.db"

    async def initialize() -> None:
        store = SQLiteEvalStore(path)
        await store.close()

    asyncio.run(initialize())
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            DROP INDEX idx_cayu_eval_scenarios_catalog;
            CREATE UNIQUE INDEX idx_cayu_eval_scenarios_catalog
                ON cayu_eval_scenarios(created_at DESC, revision ASC);
            DELETE FROM cayu_schema_migrations WHERE revision >= 53;
            PRAGMA user_version = 52;
            """
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="unexpected unique"):
        SQLiteEvalStore(path, schema_mode=SchemaMode.MIGRATE)
    connection = sqlite3.connect(path)
    try:
        assert (
            connection.execute(
                "SELECT 1 FROM cayu_schema_migrations WHERE revision = 53"
            ).fetchone()
            is None
        )
    finally:
        connection.close()


def test_sqlite_revision_forty_eight_preserves_cases_and_admits_zero_messages(
    tmp_path,
) -> None:
    path = tmp_path / "evals.db"
    corpus = _corpus(trials=1)

    async def initialize_revision_forty_eight() -> None:
        store = SQLiteEvalStore(path)
        try:
            await _save_corpus(store, corpus)
        finally:
            await store.close()

    asyncio.run(initialize_revision_forty_eight())
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            ALTER TABLE cayu_eval_cases RENAME TO cayu_eval_cases_revision_48;
            DROP INDEX idx_cayu_eval_cases_suite;
            CREATE TABLE cayu_eval_cases (
                corpus_revision TEXT NOT NULL,
                case_id TEXT COLLATE BINARY NOT NULL,
                case_revision TEXT NOT NULL,
                suite_id TEXT COLLATE BINARY NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                message_count INTEGER NOT NULL
                    CHECK (message_count >= 1 AND message_count <= 16),
                assertion_count INTEGER NOT NULL
                    CHECK (assertion_count >= 1 AND assertion_count <= 64),
                PRIMARY KEY (corpus_revision, case_id),
                FOREIGN KEY (corpus_revision, suite_id)
                    REFERENCES cayu_eval_suites(corpus_revision, suite_id) ON DELETE CASCADE
            );
            INSERT INTO cayu_eval_cases
            SELECT * FROM cayu_eval_cases_revision_48;
            DROP TABLE cayu_eval_cases_revision_48;
            CREATE INDEX idx_cayu_eval_cases_suite
                ON cayu_eval_cases(corpus_revision, suite_id, case_id ASC);
            DELETE FROM cayu_schema_migrations WHERE revision >= 48;
            PRAGMA user_version = 47;
            """
        )
        connection.commit()
    finally:
        connection.close()

    async def migrate() -> None:
        store = SQLiteEvalStore(path, schema_mode=SchemaMode.MIGRATE)
        try:
            assert await store.load_corpus(corpus.revision) == corpus
        finally:
            await store.close()

    asyncio.run(migrate())
    connection = sqlite3.connect(path)
    try:
        assert connection.execute(
            "SELECT message_count FROM cayu_eval_cases WHERE corpus_revision = ? AND case_id = ?",
            (corpus.revision, corpus.cases[0].id),
        ).fetchone() == (len(corpus.cases[0].input.messages),)
        connection.execute(
            """
            INSERT INTO cayu_eval_cases (
                corpus_revision, case_id, case_revision, suite_id, name,
                description, message_count, assertion_count
            ) VALUES (?, ?, ?, ?, ?, NULL, 0, 1)
            """,
            (
                corpus.revision,
                "captured-contract-check",
                "sha256:" + "f" * 64,
                corpus.suites[0].id,
                "Captured contract check",
            ),
        )
    finally:
        connection.close()


def test_sqlite_revision_fifty_backfills_existing_eval_run_invocation(
    tmp_path,
) -> None:
    path = tmp_path / "evals.db"
    corpus = _corpus(trials=1)

    async def initialize_revision_forty_nine() -> None:
        store = SQLiteEvalStore(path)
        try:
            await _save_corpus(store, corpus)
            await _admit_run(store, _request(corpus))
        finally:
            await store.close()

    asyncio.run(initialize_revision_forty_nine())
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            ALTER TABLE cayu_eval_runs DROP COLUMN invocation_json;
            DELETE FROM cayu_schema_migrations WHERE revision >= 50;
            PRAGMA user_version = 49;
            """
        )
        connection.commit()
    finally:
        connection.close()

    async def migrate() -> None:
        store = SQLiteEvalStore(path, schema_mode=SchemaMode.MIGRATE)
        try:
            run = await store.load_run("run-1")
            assert run.spec.invocation == EvalRunInvocation()
        finally:
            await store.close()

    asyncio.run(migrate())


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
        scenario = _scenario(corpus, text="Persist this scenario.")
        await first.save_scenario(scenario, redact_json=_NO_SECRETS.redact_json)
        admitted = await _admit_run(first, _request(corpus))
        assert admitted.status is EvalRunStatus.QUEUED
        claimed = await first.claim_run()
        assert claimed is not None
        completed = await _publish_result(first, claimed.claim, result)
        assert completed.status is EvalRunStatus.COMPLETED
        captured = captured_result_for_corpus(corpus, result)
        await first.save_captured_result(
            corpus,
            captured,
            redact_json=_NO_SECRETS.redact_json,
        )
        baseline_key = EvalBaselineKey(
            target_key=corpus.target_key,
            corpus_revision=corpus.revision,
            suite_id=corpus.suites[0].id,
        )
        baseline_mutation = await first.set_baseline(
            EvalBaselineUpdate(
                key=baseline_key,
                result_revision=captured.revision,
                expected_generation=0,
                operation_id="sha256:" + "9" * 64,
                actor_id="restart-operator",
            ),
            redact_json=_NO_SECRETS.redact_json,
        )
        await first.close()

        reopened = SQLiteEvalStore(path, schema_mode=SchemaMode.VALIDATE)
        assert await reopened.load_corpus(corpus.revision) == corpus
        assert await reopened.load_scenario(scenario.revision) == scenario
        assert await reopened.load_run(completed.id) == completed
        assert await reopened.load_result(completed.id) == result
        assert await reopened.load_result_by_revision(result.revision) == result
        assert await reopened.load_result_by_revision(captured.revision) == captured
        baseline = await reopened.load_baseline(baseline_key)
        assert baseline is not None
        assert baseline.result_revision == captured.revision
        assert (
            await reopened.load_baseline_mutation(baseline_mutation.operation_id)
            == baseline_mutation
        )
        assert await _admit_run(reopened, _request(corpus, run_id="retry-id")) == completed
        await reopened.close()

    asyncio.run(exercise())


def test_sqlite_revision_forty_seven_indexes_existing_fresh_results(tmp_path) -> None:
    path = tmp_path / "evals.db"

    async def prepare_revision_forty_six() -> tuple[EvalCorpusDocument, CorpusExecutionResult]:
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
            await _admit_run(store, _request(corpus, run_id="pre-revision-47"))
            lease = await store.claim_run()
            assert lease is not None
            await _publish_result(store, lease.claim, result)
        finally:
            await store.close()
        return corpus, result

    corpus, result = asyncio.run(prepare_revision_forty_six())
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            DROP TABLE cayu_eval_baseline_mutations;
            DROP TABLE cayu_eval_baselines;
            DROP TABLE cayu_eval_result_records;
            DELETE FROM cayu_schema_migrations WHERE revision >= 47;
            PRAGMA user_version = 46;
            """
        )
        connection.commit()
    finally:
        connection.close()

    async def migrate_and_read() -> None:
        store = SQLiteEvalStore(path, schema_mode=SchemaMode.MIGRATE)
        try:
            record = await store.load_result_record(result.revision)
            assert record is not None
            assert record.target.target_key == corpus.target_key
            assert await store.load_result_by_revision(result.revision) == result
        finally:
            await store.close()

    asyncio.run(migrate_and_read())


def test_sqlite_revision_forty_seven_rejects_a_nonunique_baseline_audit_index(
    tmp_path,
) -> None:
    path = tmp_path / "evals.db"

    async def initialize() -> None:
        store = SQLiteEvalStore(path)
        await store.close()

    asyncio.run(initialize())
    connection = sqlite3.connect(path)
    try:
        connection.execute("DROP INDEX idx_cayu_eval_baseline_mutations_scope")
        connection.execute(
            "CREATE INDEX idx_cayu_eval_baseline_mutations_scope "
            "ON cayu_eval_baseline_mutations("
            "target_key, corpus_revision, suite_id, resulting_generation)"
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="revision-47 Evals query contract"):
        SQLiteEvalStore(path, schema_mode=SchemaMode.VALIDATE)


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


def test_sqlite_eval_store_rolls_back_interrupted_result_publication(tmp_path, caplog) -> None:
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

    caplog.set_level(logging.ERROR, logger=evals_sqlite_module.__name__)
    asyncio.run(exercise())
    assert "sqlite_writer.permanent_storage_failure" in {
        getattr(record, "cayu_eval_store_event", None) for record in caplog.records
    }
