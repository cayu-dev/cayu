from __future__ import annotations

import asyncio

import pytest
from tests.evals.eval_store_conformance import _terminal_trial_checkpoint
from tests.evals.test_corpus_execution import _corpus
from tests.evals.test_eval_store import _NO_SECRETS, _request
from tests.server.test_server_eval_suite_authoring import _draft

from cayu.evals.store import InMemoryEvalStore
from cayu.evals.suite_authoring import compile_eval_suite_authoring_draft
from cayu.storage.evals_sqlite import SQLiteEvalStore


async def _exercise_suite_read_store(store):
    from cayu.evals.read_views import EvalSuiteViewIdentity, EvalViewQuery

    corpus = _corpus(trials=2)
    await store.save_corpus(corpus, redact_json=_NO_SECRETS.redact_json)
    draft = _draft().model_copy(update={"target_key": corpus.target_key})
    document = compile_eval_suite_authoring_draft(draft)
    await store.save_authored_suite(document, redact_json=_NO_SECRETS.redact_json)
    first = await store.list_view_index(EvalViewQuery(target_key=corpus.target_key, limit=1))
    assert len(first.items) == 1 and first.has_more
    second = await store.list_view_index(
        EvalViewQuery(target_key=corpus.target_key, limit=1, cursor=first.next_cursor)
    )
    assert len(second.items) == 1 and not second.has_more
    assert {item.source for item in (*first.items, *second.items)} == {"authored", "corpus"}
    with pytest.raises(ValueError, match="scope"):
        await store.list_view_index(
            EvalViewQuery(target_key="wrong-target", cursor=first.next_cursor)
        )
    other = _corpus(target_key="second-target", trials=1)
    await store.save_corpus(other, redact_json=_NO_SECRETS.redact_json)
    global_page = await store.list_view_index(
        EvalViewQuery(target_keys=tuple(sorted((corpus.target_key, other.target_key))))
    )
    assert len(global_page.items) == 3
    assert {item.target_key for item in global_page.items} == {corpus.target_key, other.target_key}
    scoped_page = await store.list_view_index(EvalViewQuery(target_keys=(corpus.target_key,)))
    assert len(scoped_page.items) == 2
    scope = EvalSuiteViewIdentity(
        source="corpus",
        target_key=corpus.target_key,
        revision=corpus.revision,
        suite_id=corpus.suites[0].id,
    )
    request = _request(corpus, run_id="view-run-a", key="view-a")
    await store.admit_run(request, redact_json=_NO_SECRETS.redact_json)
    await store.admit_run(
        _request(corpus, run_id="view-run-b", key="view-b"), redact_json=_NO_SECRETS.redact_json
    )
    query = EvalViewQuery(target_key=corpus.target_key, suite=scope, limit=1)
    history = await store.list_view_index(query)
    assert len(history.items) == 1 and history.has_more
    rest = await store.list_view_index(query.model_copy(update={"cursor": history.next_cursor}))
    assert not rest.has_more
    assert {item.revision for item in (*history.items, *rest.items)} == {"view-run-a", "view-run-b"}
    from cayu.evals.read_views import EvalCaseHistoryQuery

    assert (
        await store.load_latest_case_results(
            EvalCaseHistoryQuery(suite=scope, case_ids=(corpus.cases[0].id,))
        )
        == ()
    )
    queued = await store.load_run_view_snapshot("view-run-a")
    assert queued.run.started_at is None
    assert queued.completed_trials_by_case == {}
    lease = await store.claim_run(lease_seconds=300)
    assert lease is not None
    checkpoint = _terminal_trial_checkpoint(corpus)
    await store.save_trial_checkpoint(lease.claim, checkpoint, redact_json=_NO_SECRETS.redact_json)
    await store.save_trial_checkpoint(lease.claim, checkpoint, redact_json=_NO_SECRETS.redact_json)
    active = await store.load_run_view_snapshot(lease.run.id)
    assert active.run.status == "running"
    assert active.completed_trials_by_case == {corpus.cases[0].id: 1}
    assert "claim_id" not in active.model_dump_json()
    assert "checkpoint" not in active.model_dump_json()
    await store.request_cancel(lease.run.id)
    cancelling = await store.load_run_view_snapshot(lease.run.id)
    assert cancelling.run.status == "cancelling"
    assert cancelling.completed_trials_by_case == active.completed_trials_by_case


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_suite_read_store_unifies_catalog_scopes_history_and_observes_checkpoints(
    tmp_path, backend
):
    async def exercise():
        store = (
            InMemoryEvalStore() if backend == "memory" else SQLiteEvalStore(tmp_path / "views.db")
        )
        try:
            await _exercise_suite_read_store(store)
        finally:
            await store.close()

    asyncio.run(exercise())


def test_postgres_suite_read_store_conformance(postgres_dsn):
    from cayu.storage.evals_postgres import PostgresEvalStore
    from cayu.storage.migrations import SchemaMode

    async def exercise():
        store = PostgresEvalStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            await _exercise_suite_read_store(store)
        finally:
            await store.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("view", ["catalog", "history", "latest"])
def test_postgres_read_plans_preserve_binary_order_under_a_linguistic_default(view):
    """Execute the PG plans with a deterministic locale fixture, without a PG service.

    SQLite drops a column's collation through concatenation, unlike PostgreSQL;
    the alias annotation below models PG's inherited collation. Parameter/cast
    syntax is the only other adaptation. Real PG HTTP coverage lives alongside
    the server read-view tests and runs in its normal CI lane.
    """
    import sqlite3
    from datetime import UTC, datetime

    from cayu.evals.read_views import (
        EvalCaseHistoryQuery,
        EvalSuiteViewIdentity,
        EvalViewQuery,
        encode_view_token,
    )
    from cayu.storage._eval_view_queries import latest_case_results_sql, view_index_sql

    def linguistic(left, right):
        left = "".join(char for char in left.lower() if char.isalnum())
        right = "".join(char for char in right.lower() if char.isalnum())
        return (left > right) - (left < right)

    with sqlite3.connect(":memory:") as db:
        db.create_collation("linguistic", linguistic)
        db.create_collation("C", lambda a, b: (a > b) - (a < b))
        db.executescript("""
            CREATE TABLE cayu_eval_authored_suites (revision TEXT, created_at TEXT, target_key TEXT);
            CREATE TABLE cayu_eval_corpora (revision TEXT, created_at TEXT, target_key TEXT);
            CREATE TABLE cayu_eval_suites (corpus_revision TEXT, suite_id TEXT COLLATE linguistic);
            CREATE TABLE cayu_eval_runs (run_id TEXT COLLATE linguistic, created_at TEXT,
                target_key TEXT, corpus_revision TEXT, suite_id TEXT, result_revision TEXT);
            CREATE TABLE cayu_eval_cases (corpus_revision TEXT, suite_id TEXT, case_id TEXT COLLATE linguistic);
            CREATE TABLE cayu_eval_result_records (revision TEXT, created_at TEXT, target_key TEXT,
                corpus_revision TEXT, suite_id TEXT, origin TEXT, captured_result TEXT);
            CREATE TABLE cayu_eval_session_additions (destination_selector TEXT, result_revision TEXT);
        """)
        revision = "sha256:" + "a" * 64
        at = datetime(2026, 1, 1, tzinfo=UTC)
        db.execute(
            "INSERT INTO cayu_eval_corpora VALUES (?, ?, ?)", (revision, at.isoformat(), "target")
        )
        suite_ids = ("aa", "a-b", "a.b")
        db.executemany(
            "INSERT INTO cayu_eval_suites VALUES (?, ?)", [(revision, value) for value in suite_ids]
        )
        run_ids = ("aa", "a-b", "Zed", "zed")
        db.executemany(
            "INSERT INTO cayu_eval_runs VALUES (?, ?, ?, ?, ?, ?)",
            [(value, at.isoformat(), "target", revision, "a-b", revision) for value in run_ids],
        )
        db.executemany(
            "INSERT INTO cayu_eval_cases VALUES (?, ?, ?)",
            [(revision, "a-b", value) for value in suite_ids],
        )
        scope = EvalSuiteViewIdentity(
            source="corpus", target_key="target", revision=revision, suite_id="a-b"
        )

        def execute(sql, params):
            sql = sql.replace("%s", "?").replace("::jsonb", "")
            sql = sql.replace(" AS sort_key", " COLLATE linguistic AS sort_key")
            params = tuple(
                value.isoformat() if isinstance(value, datetime) else value for value in params
            )
            return db.execute(sql, params).fetchall()

        if view == "latest":
            rows = execute(
                *latest_case_results_sql(
                    EvalCaseHistoryQuery(suite=scope, case_ids=tuple(sorted(suite_ids))),
                    postgres=True,
                )
            )
            assert rows == [(case_id, "run", min(run_ids)) for case_id in sorted(suite_ids)]
            return
        query = EvalViewQuery(
            target_key="target", suite=scope if view == "history" else None, limit=1
        )
        expected = sorted(run_ids if view == "history" else suite_ids)
        observed = []
        for _ in range(len(expected) + 1):
            rows = execute(*view_index_sql(query, postgres=True))
            if not rows:
                break
            source, identity, suite_id, _, _ = rows[0]
            observed.append(identity if view == "history" else suite_id)
            key = f"{source}:{identity}:{suite_id or ''}"
            query = query.model_copy(
                update={
                    "cursor": encode_view_token(
                        ["page", query.scope(), at.isoformat() if view == "history" else None, key]
                    )
                }
            )
        assert observed == expected


async def _exercise_result_run_linkage(store):
    from tests.evals.eval_store_conformance import captured_result_for_corpus
    from tests.server.test_server_evals import _provider, _target

    from cayu.evals.execution import run_corpus_suite

    target = _target(_provider(trials=1), key="view-result-linkage")
    corpus = _corpus(trials=1, target_key=target.key)
    await store.save_corpus(corpus, redact_json=target.app.redact_json)
    result = await run_corpus_suite(target, corpus, corpus.suites[0].id, max_concurrency=1)
    assert await store.load_result_run_view_snapshot(result.revision) is None
    # Identical content can be published by two native runs; retain the catalog's
    # original positive link, rather than guessing from whichever history wins.
    for run_id in ("original", "duplicate"):
        await store.admit_run(
            _request(corpus, run_id=run_id, key=run_id), redact_json=target.app.redact_json
        )
        lease = await store.claim_run(target_key=target.key, lease_seconds=300)
        assert lease is not None and lease.run.id == run_id
        await store.publish_result(lease.claim, result, redact_json=target.app.redact_json)
        snapshot = await store.load_result_run_view_snapshot(result.revision)
        assert snapshot.run.id == "original"
        assert snapshot.run.result.revision == result.revision
        assert snapshot.run.started_at is not None and snapshot.run.finished_at is not None
    captured = captured_result_for_corpus(corpus, result)
    await store.save_captured_result(corpus, captured, redact_json=target.app.redact_json)
    assert await store.load_result_run_view_snapshot(captured.revision) is None
    assert await store.load_result_run_view_snapshot("sha256:" + "f" * 64) is None


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_result_run_linkage_uses_recorded_identity_and_excludes_captured(tmp_path, backend):
    async def exercise():
        store = (
            InMemoryEvalStore() if backend == "memory" else SQLiteEvalStore(tmp_path / "links.db")
        )
        try:
            # Match persistent CI reuse: another scope has a queued native run.
            await _exercise_suite_read_store(store)
            await _exercise_result_run_linkage(store)
        finally:
            await store.close()

    asyncio.run(exercise())


def test_postgres_result_run_linkage_uses_recorded_identity(postgres_dsn):
    from cayu.storage.evals_postgres import PostgresEvalStore
    from cayu.storage.migrations import SchemaMode

    async def exercise():
        store = PostgresEvalStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            await _exercise_result_run_linkage(store)
        finally:
            await store.close()

    asyncio.run(exercise())
