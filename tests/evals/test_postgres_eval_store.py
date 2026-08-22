from __future__ import annotations

import asyncio
import threading
from contextlib import suppress

import pytest
from tests.evals.eval_store_conformance import (
    assert_captured_eval_store_conformance,
    assert_eval_store_conformance,
    assert_eval_store_reconstruction_releases_heartbeat_capacity,
    captured_result_for_corpus,
)
from tests.evals.test_corpus_execution import _corpus, _provider, _target

from cayu.evals.execution import run_corpus_suite
from cayu.evals.store import (
    EvalBaselineKey,
    EvalBaselineUpdate,
    EvalRunClaim,
    EvalRunInvocation,
    EvalRunRecord,
    EvalRunRequest,
    EvalRunStatus,
)
from cayu.vaults.redaction import SecretRedactor

pytestmark = pytest.mark.usefixtures("postgres_dsn")
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
            await assert_captured_eval_store_conformance(
                store,
                corpus=corpus,
                result=result,
            )
        finally:
            await store.close()

    asyncio.run(exercise())


def test_postgres_eval_store_creates_revision_fifty_schema(postgres_dsn) -> None:
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
                "SELECT revision, kind, compatible_from FROM cayu_schema_migrations "
                "WHERE revision IN (47, 48, 49, 50) ORDER BY revision"
            )
            assert await cur.fetchall() == [
                (47, "breaking", 47),
                (48, "breaking", 48),
                (49, "breaking", 49),
                (50, "breaking", 50),
            ]
            await cur.execute(
                "SELECT data_type, is_nullable, column_default "
                "FROM information_schema.columns WHERE table_schema = current_schema() "
                "AND table_name = 'cayu_eval_runs' AND column_name = 'invocation_json'"
            )
            assert await cur.fetchone() == ("text", "NO", None)
            await cur.execute(
                """
                SELECT pg_get_constraintdef(constraint_record.oid)
                FROM pg_catalog.pg_constraint AS constraint_record
                JOIN pg_catalog.pg_class AS table_record
                  ON table_record.oid = constraint_record.conrelid
                JOIN pg_catalog.pg_namespace AS namespace
                  ON namespace.oid = table_record.relnamespace
                WHERE namespace.nspname = current_schema()
                  AND table_record.relname = 'cayu_eval_cases'
                  AND constraint_record.conname = 'cayu_eval_cases_message_count_check'
                """
            )
            constraint = await cur.fetchone()
            assert constraint is not None
            normalized_constraint = "".join(constraint[0].lower().split())
            assert "message_count>=0" in normalized_constraint
            assert "message_count<=16" in normalized_constraint
            await cur.execute(
                "SELECT indexname FROM pg_indexes WHERE schemaname = current_schema() "
                "AND (indexname LIKE 'idx_cayu_eval_runs_target_%' "
                "OR indexname LIKE 'idx_cayu_eval_result_records_%' "
                "OR indexname = 'idx_cayu_eval_baseline_mutations_scope') "
                "ORDER BY indexname"
            )
            assert [row[0] for row in await cur.fetchall()] == [
                "idx_cayu_eval_baseline_mutations_scope",
                "idx_cayu_eval_result_records_contract",
                "idx_cayu_eval_result_records_target_catalog",
                "idx_cayu_eval_runs_target_catalog",
                "idx_cayu_eval_runs_target_status_claim",
            ]
            await cur.execute(
                "SELECT tablename FROM pg_tables "
                "WHERE schemaname = current_schema() AND tablename LIKE 'cayu_eval_%'"
            )
            assert {row[0] for row in await cur.fetchall()} == {
                "cayu_eval_baseline_mutations",
                "cayu_eval_baselines",
                "cayu_eval_cases",
                "cayu_eval_corpora",
                "cayu_eval_result_records",
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


def test_postgres_revision_forty_eight_preserves_cases_and_admits_zero_messages(
    postgres_dsn,
) -> None:
    async def exercise() -> None:
        import psycopg

        from cayu.storage.evals_postgres import PostgresEvalStore
        from cayu.storage.migrations import SchemaMode

        await _drop_eval_tables(postgres_dsn)
        corpus = _corpus(trials=1)
        initialized = PostgresEvalStore(postgres_dsn, schema_mode=SchemaMode.MIGRATE)
        try:
            await _save_corpus(initialized, corpus)
        finally:
            await initialized.close()

        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as conn,
            conn.cursor() as cur,
        ):
            await cur.execute(
                "ALTER TABLE cayu_eval_cases DROP CONSTRAINT cayu_eval_cases_message_count_check"
            )
            await cur.execute(
                "ALTER TABLE cayu_eval_cases ADD CONSTRAINT "
                "cayu_eval_cases_message_count_check "
                "CHECK (message_count >= 1 AND message_count <= 16)"
            )
            await cur.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 48")

        migrated = PostgresEvalStore(postgres_dsn, schema_mode=SchemaMode.MIGRATE)
        try:
            assert await migrated.load_corpus(corpus.revision) == corpus
        finally:
            await migrated.close()

        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as conn,
            conn.cursor() as cur,
        ):
            await cur.execute(
                "SELECT message_count FROM cayu_eval_cases "
                "WHERE corpus_revision = %s AND case_id = %s",
                (corpus.revision, corpus.cases[0].id),
            )
            assert await cur.fetchone() == (len(corpus.cases[0].input.messages),)
            await cur.execute(
                """
                INSERT INTO cayu_eval_cases (
                    corpus_revision, case_id, case_revision, suite_id, name,
                    description, message_count, assertion_count
                ) VALUES (%s, %s, %s, %s, %s, NULL, 0, 1)
                """,
                (
                    corpus.revision,
                    "captured-contract-check",
                    "sha256:" + "f" * 64,
                    corpus.suites[0].id,
                    "Captured contract check",
                ),
            )

    asyncio.run(exercise())


def test_postgres_revision_fifty_backfills_existing_eval_run_invocation(
    postgres_dsn,
) -> None:
    async def exercise() -> None:
        import psycopg

        from cayu.storage.evals_postgres import PostgresEvalStore
        from cayu.storage.migrations import SchemaMode

        await _drop_eval_tables(postgres_dsn)
        corpus = _corpus(trials=1)
        initialized = PostgresEvalStore(postgres_dsn, schema_mode=SchemaMode.MIGRATE)
        try:
            await _save_corpus(initialized, corpus)
            await _admit_run(initialized, _request(corpus))
        finally:
            await initialized.close()

        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as conn,
            conn.cursor() as cur,
        ):
            await cur.execute("ALTER TABLE cayu_eval_runs DROP COLUMN invocation_json")
            await cur.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 50")

        migrated = PostgresEvalStore(postgres_dsn, schema_mode=SchemaMode.MIGRATE)
        try:
            run = await migrated.load_run("run-1")
            assert run.spec.invocation == EvalRunInvocation()
        finally:
            await migrated.close()

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
        finally:
            await reopened.close()

    asyncio.run(exercise())


def test_postgres_revision_forty_seven_indexes_existing_fresh_results(postgres_dsn) -> None:
    async def exercise() -> None:
        import psycopg

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
        first = PostgresEvalStore(postgres_dsn, schema_mode=SchemaMode.MIGRATE)
        try:
            await _save_corpus(first, corpus)
            await _admit_run(first, _request(corpus, run_id="pre-revision-47"))
            lease = await first.claim_run()
            assert lease is not None
            await _publish_result(first, lease.claim, result)
        finally:
            await first.close()

        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as conn,
            conn.cursor() as cur,
        ):
            await cur.execute("DROP TABLE cayu_eval_baseline_mutations")
            await cur.execute("DROP TABLE cayu_eval_baselines")
            await cur.execute("DROP TABLE cayu_eval_result_records")
            await cur.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 47")
            await conn.commit()

        migrated = PostgresEvalStore(postgres_dsn, schema_mode=SchemaMode.MIGRATE)
        try:
            record = await migrated.load_result_record(result.revision)
            assert record is not None
            assert record.target.target_key == corpus.target_key
            assert await migrated.load_result_by_revision(result.revision) == result
        finally:
            await migrated.close()

    asyncio.run(exercise())


def test_postgres_revision_forty_seven_rejects_a_nonunique_baseline_audit_index(
    postgres_dsn,
) -> None:
    async def exercise() -> None:
        import psycopg

        from cayu.storage.evals_postgres import PostgresEvalStore
        from cayu.storage.migrations import SchemaMode

        await _drop_eval_tables(postgres_dsn)
        initialized = PostgresEvalStore(postgres_dsn, schema_mode=SchemaMode.MIGRATE)
        try:
            await initialized.list_corpora()
        finally:
            await initialized.close()
        async with (
            await psycopg.AsyncConnection.connect(postgres_dsn) as conn,
            conn.cursor() as cur,
        ):
            await cur.execute("DROP INDEX idx_cayu_eval_baseline_mutations_scope")
            await cur.execute(
                "CREATE INDEX idx_cayu_eval_baseline_mutations_scope "
                "ON cayu_eval_baseline_mutations("
                "target_key, corpus_revision, suite_id, resulting_generation)"
            )
            await conn.commit()

        validator = PostgresEvalStore(postgres_dsn, schema_mode=SchemaMode.VALIDATE)
        try:
            with pytest.raises(RuntimeError, match="revision-47 Evals result"):
                await validator.list_corpora()
        finally:
            await validator.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("read_kind", "parser_name"),
    [
        ("corpus", "eval_corpus_from_json"),
        ("result", "corpus_execution_result_from_json"),
    ],
)
def test_postgres_eval_reconstruction_releases_its_pool_connection(
    postgres_dsn,
    monkeypatch,
    read_kind: str,
    parser_name: str,
) -> None:
    async def exercise() -> None:
        from cayu.storage import evals_postgres as evals_postgres_module
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
            max_size=1,
            schema_mode=SchemaMode.MIGRATE,
        )
        try:
            await assert_eval_store_reconstruction_releases_heartbeat_capacity(
                store,
                corpus=corpus,
                result=result,
                read_kind=read_kind,
                parser_owner=evals_postgres_module,
                parser_name=parser_name,
                monkeypatch=monkeypatch,
            )
        finally:
            await store.close()

    asyncio.run(exercise())


def test_postgres_result_validation_releases_heartbeat_capacity(
    postgres_dsn,
    monkeypatch,
) -> None:
    async def exercise() -> None:
        from cayu.storage import evals_postgres as evals_postgres_module
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
            max_size=1,
            schema_mode=SchemaMode.MIGRATE,
        )
        validation_started = threading.Event()
        release_validation = threading.Event()
        stop_heartbeats = asyncio.Event()
        original_validate = evals_postgres_module.validate_result_for_run
        heartbeat_count = 0
        publication: asyncio.Task[EvalRunRecord] | None = None
        heartbeats: asyncio.Task[None] | None = None

        def blocking_validate(*args, **kwargs):
            validation_started.set()
            if not release_validation.wait(timeout=5):
                raise AssertionError("Timed out releasing eval result validation.")
            return original_validate(*args, **kwargs)

        async def maintain_claim(claim: EvalRunClaim) -> None:
            nonlocal heartbeat_count
            while not stop_heartbeats.is_set():
                await store.heartbeat_run(claim, extend_seconds=1)
                heartbeat_count += 1
                with suppress(TimeoutError):
                    await asyncio.wait_for(stop_heartbeats.wait(), timeout=0.1)

        try:
            await _save_corpus(store, corpus)
            await _admit_run(store, _request(corpus))
            lease = await store.claim_run(lease_seconds=1)
            assert lease is not None
            monkeypatch.setattr(
                evals_postgres_module,
                "validate_result_for_run",
                blocking_validate,
            )
            publication = asyncio.create_task(_publish_result(store, lease.claim, result))
            assert await asyncio.to_thread(validation_started.wait, 2)
            heartbeats = asyncio.create_task(maintain_claim(lease.claim))

            await asyncio.sleep(1.2)
            assert heartbeat_count >= 4

            stop_heartbeats.set()
            await asyncio.wait_for(heartbeats, timeout=2)
            release_validation.set()
            completed = await asyncio.wait_for(publication, timeout=2)
            assert completed.status is EvalRunStatus.COMPLETED
        finally:
            stop_heartbeats.set()
            release_validation.set()
            tasks = tuple(task for task in (publication, heartbeats) if task is not None)
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            await store.close()

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
