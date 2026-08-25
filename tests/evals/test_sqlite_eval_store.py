from __future__ import annotations

import asyncio
import sqlite3
import threading
from contextlib import suppress

import pytest
from tests.evals.eval_store_conformance import (
    _scenario,
    assert_captured_eval_store_conformance,
    assert_eval_store_conformance,
    assert_eval_store_reconstruction_releases_heartbeat_capacity,
    assert_scenario_progress_conformance,
    captured_result_for_corpus,
)
from tests.evals.test_corpus_execution import _corpus, _provider, _target

import cayu.storage.evals_sqlite as evals_sqlite_module
from cayu.evals.corpus import EvalCorpusDocument
from cayu.evals.execution import CorpusExecutionResult, run_corpus_suite
from cayu.evals.store import (
    EvalBaselineKey,
    EvalBaselineUpdate,
    EvalRunClaim,
    EvalRunInvocation,
    EvalRunRecord,
    EvalRunRequest,
    EvalRunStatus,
)
from cayu.storage import _sqlite_support as sqlite_support
from cayu.storage import migrations as schema_migrations
from cayu.storage.evals_sqlite import SQLiteEvalStore
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
            await assert_captured_eval_store_conformance(
                store,
                corpus=corpus,
                result=result,
            )
            await assert_scenario_progress_conformance(store, corpus=corpus)
        finally:
            await store.close()

    asyncio.run(exercise())


def test_sqlite_eval_store_creates_revision_sixty_four_schema(tmp_path) -> None:
    path = tmp_path / "evals.db"

    async def initialize() -> None:
        store = SQLiteEvalStore(path)
        await store.close()

    asyncio.run(initialize())
    connection = sqlite3.connect(path)
    try:
        revisions = connection.execute(
            "SELECT revision, kind, compatible_from FROM cayu_schema_migrations "
            "WHERE revision IN (47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64) "
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
        "cayu_eval_cases",
        "cayu_eval_corpora",
        "cayu_eval_result_records",
        "cayu_eval_results",
        "cayu_eval_runs",
        "cayu_eval_scenarios",
        "cayu_eval_suites",
    }
    assert indexes == {
        "idx_cayu_eval_baseline_mutations_scope",
        "idx_cayu_eval_authored_suites_catalog",
        "idx_cayu_eval_authored_suites_id_catalog",
        "idx_cayu_eval_authored_suites_target_catalog",
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


def test_sqlite_eval_store_requires_revision_sixty_four_authoring_schema(
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

    with pytest.raises(SchemaTooOld, match="requires >= 64"):
        SQLiteEvalStore(path, schema_mode=SchemaMode.VALIDATE)


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
