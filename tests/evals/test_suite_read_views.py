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
            CREATE TABLE cayu_eval_authored_suites (revision TEXT, created_at TEXT, target_key TEXT, suite_id TEXT);
            CREATE TABLE cayu_eval_corpora (revision TEXT, created_at TEXT, target_key TEXT);
            CREATE TABLE cayu_eval_suites (corpus_revision TEXT, suite_id TEXT COLLATE linguistic);
            CREATE TABLE cayu_eval_runs (run_id TEXT COLLATE linguistic, created_at TEXT,
                target_key TEXT, corpus_revision TEXT, suite_id TEXT, result_revision TEXT, invocation_json TEXT);
            CREATE TABLE cayu_eval_cases (corpus_revision TEXT, suite_id TEXT, case_id TEXT COLLATE linguistic);
            CREATE TABLE cayu_eval_result_records (revision TEXT, created_at TEXT, target_key TEXT,
                corpus_revision TEXT, suite_id TEXT, origin TEXT, captured_result TEXT);
            CREATE TABLE cayu_eval_session_additions (destination_selector TEXT, result_revision TEXT, record_json TEXT);
            CREATE TABLE cayu_eval_report_series (selector_json TEXT, created_at TEXT, target_key TEXT);
            CREATE TABLE cayu_eval_reports (report_key TEXT, created_at TEXT, target_key TEXT,
                series_id TEXT, association_key TEXT, record_json TEXT);
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
            "INSERT INTO cayu_eval_runs VALUES (?, ?, ?, ?, ?, ?, '{}')",
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


async def _exercise_authored_catalog_dedup(store, independent="native"):
    from cayu.evals.read_views import EvalSuiteViewIdentity, EvalViewQuery
    from cayu.evals.store import EvalRunInvocation

    corpus = _corpus(target_key=f"catalog-dedup-{independent}", trials=1)
    authored_run = f"{corpus.target_key}-authored"
    document = compile_eval_suite_authoring_draft(
        _draft().model_copy(update={"target_key": corpus.target_key, "id": corpus.suites[0].id})
    )
    await store.save_authored_suite(document, redact_json=_NO_SECRETS.redact_json)
    await store.save_corpus(corpus, redact_json=_NO_SECRETS.redact_json)
    # Matching names are not evidence. The persisted admitted invocation is.
    before = await store.list_view_index(EvalViewQuery(target_key=corpus.target_key))
    assert {item.source for item in before.items} == {"authored", "corpus"}
    request = _request(corpus, run_id=authored_run, key=authored_run).model_copy(
        update={
            "invocation": EvalRunInvocation(
                authored_suite_revision=document.revision,
                authored_suite_selection_revision="sha256:" + "a" * 64,
            ),
        }
    )
    await store.admit_run(request, redact_json=_NO_SECRETS.redact_json)
    page = await store.list_view_index(EvalViewQuery(target_key=corpus.target_key, limit=1))
    assert [item.source for item in page.items] == ["authored"]
    assert page.has_more is False
    assert page.next_cursor is None
    scope = EvalSuiteViewIdentity(
        source="corpus",
        target_key=corpus.target_key,
        revision=corpus.revision,
        suite_id=corpus.suites[0].id,
    )
    history = await store.list_view_index(EvalViewQuery(target_key=corpus.target_key, suite=scope))
    assert [item.revision for item in history.items] == [authored_run]
    if independent == "result":
        # A directly saved result has no native run or placement association.
        from tests.evals.eval_store_conformance import captured_result_for_corpus
        from tests.evals.test_corpus_execution import _provider, _target

        from cayu.evals.execution import run_corpus_suite

        result = await run_corpus_suite(
            _target(_provider(trials=1), key=corpus.target_key), corpus, corpus.suites[0].id
        )
        captured = captured_result_for_corpus(corpus, result)
        await store.save_captured_result(corpus, captured, redact_json=_NO_SECRETS.redact_json)
    else:
        await store.admit_run(
            _request(
                corpus,
                run_id=f"{corpus.target_key}-standalone",
                key=f"{corpus.target_key}-standalone",
            ),
            redact_json=_NO_SECRETS.redact_json,
        )
    mixed = await store.list_view_index(EvalViewQuery(target_key=corpus.target_key))
    assert {item.source for item in mixed.items} == {"authored", "corpus"}


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("independent", ["native", "result"])
def test_authored_catalog_deduplicates_only_linked_corpora_before_pagination(
    tmp_path, backend, independent
):
    async def exercise():
        store = (
            InMemoryEvalStore() if backend == "memory" else SQLiteEvalStore(tmp_path / "dedup.db")
        )
        try:
            # PostgreSQL fixtures retain earlier module cases. Exercise the same
            # populated-store boundary even in isolated local backends.
            await _exercise_suite_read_store(store)
            await _exercise_authored_catalog_dedup(store, independent)
        finally:
            await store.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("independent", ["native", "result"])
def test_postgres_authored_catalog_deduplicates_only_linked_corpora(postgres_dsn, independent):
    from cayu.storage.evals_postgres import PostgresEvalStore
    from cayu.storage.migrations import SchemaMode

    async def exercise():
        store = PostgresEvalStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            await _exercise_authored_catalog_dedup(store, independent)
        finally:
            await store.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("mismatch", ["missing_document", "target", "suite"])
def test_corpus_catalog_keeps_ambiguous_authored_links(tmp_path, backend, mismatch):
    from cayu.evals.read_views import EvalViewQuery
    from cayu.evals.store import EvalRunInvocation

    async def exercise():
        store = (
            InMemoryEvalStore()
            if backend == "memory"
            else SQLiteEvalStore(tmp_path / "ambiguous.db")
        )
        try:
            corpus = _corpus(trials=1)
            document = compile_eval_suite_authoring_draft(
                _draft().model_copy(
                    update={
                        "target_key": "other-target" if mismatch == "target" else corpus.target_key,
                        "id": "other-suite" if mismatch == "suite" else corpus.suites[0].id,
                    }
                )
            )
            await store.save_authored_suite(document, redact_json=_NO_SECRETS.redact_json)
            await store.save_corpus(corpus, redact_json=_NO_SECRETS.redact_json)
            request = _request(corpus).model_copy(
                update={
                    "invocation": EvalRunInvocation(
                        authored_suite_revision="sha256:" + "f" * 64
                        if mismatch == "missing_document"
                        else document.revision,
                        authored_suite_selection_revision="sha256:" + "a" * 64,
                    )
                }
            )
            await store.admit_run(request, redact_json=_NO_SECRETS.redact_json)
            page = await store.list_view_index(EvalViewQuery(target_key=corpus.target_key))
            assert any(
                item.source == "corpus" and item.revision == corpus.revision for item in page.items
            )
        finally:
            await store.close()

    asyncio.run(exercise())


async def _exercise_report_catalog_dedup(store):
    from tests.evals.test_corpus_execution import _provider, _target

    from cayu.evals.execution import run_corpus_suite
    from cayu.evals.read_views import EvalSuiteViewIdentity, EvalViewQuery
    from cayu.evals.reported import (
        EvalCorpusReportEvidence,
        EvalReport,
        EvalReportCorpusAssociation,
    )
    from cayu.evals.store import EvalRunInvocation

    corpus = _corpus(target_key="catalog-report-association", trials=1)
    document = compile_eval_suite_authoring_draft(
        _draft().model_copy(update={"target_key": corpus.target_key, "id": corpus.suites[0].id})
    )
    await store.save_corpus(corpus, redact_json=_NO_SECRETS.redact_json)
    await store.save_authored_suite(document, redact_json=_NO_SECRETS.redact_json)
    admitted = _request(
        corpus, run_id="catalog-report-authored", key="catalog-report-authored"
    ).model_copy(
        update={
            "invocation": EvalRunInvocation(
                authored_suite_revision=document.revision,
                authored_suite_selection_revision="sha256:" + "a" * 64,
            )
        }
    )
    await store.admit_run(admitted, redact_json=_NO_SECRETS.redact_json)
    scope = EvalSuiteViewIdentity(
        source="corpus",
        target_key=corpus.target_key,
        revision=corpus.revision,
        suite_id=corpus.suites[0].id,
    )
    query = EvalViewQuery(target_key=corpus.target_key, limit=1)

    async def catalog():
        items, cursor = [], None
        while True:
            page = await store.list_view_index(query.model_copy(update={"cursor": cursor}))
            items.extend(page.items)
            if not page.has_more:
                return items
            assert page.next_cursor != cursor
            cursor = page.next_cursor

    async def hidden():
        assert not any(
            item.source == "corpus" and item.revision == corpus.revision for item in await catalog()
        )

    await hidden()
    result = await run_corpus_suite(
        _target(_provider(trials=1), key=corpus.target_key), corpus, corpus.suites[0].id
    )
    report = EvalReport(
        report_id="unassociated",
        target_key=corpus.target_key,
        program_key="catalog-ci",
        evidence=EvalCorpusReportEvidence(result=result),
    )
    await store.publish_report(report, "catalog-publisher", redact_json=_NO_SECRETS.redact_json)
    await hidden()  # Matching evidence alone is not an explicit association.
    association = EvalReportCorpusAssociation(
        corpus_revision=corpus.revision, suite_id=corpus.suites[0].id
    )
    linked = report.model_copy(update={"report_id": "associated", "association": association})
    for changed in (
        linked.model_copy(update={"target_key": "different-target"}),
        linked.model_copy(
            update={
                "association": association.model_copy(
                    update={"corpus_revision": "sha256:" + "f" * 64}
                )
            }
        ),
        linked.model_copy(
            update={"association": association.model_copy(update={"suite_id": "other-suite"})}
        ),
    ):
        with pytest.raises(ValueError, match="association"):
            await store.publish_report(
                changed, "catalog-publisher", redact_json=_NO_SECRETS.redact_json
            )
        await hidden()
    # Another revision of the same target/suite must not revive this corpus.
    other = _corpus(target_key=corpus.target_key, trials=2)
    await store.save_corpus(other, redact_json=_NO_SECRETS.redact_json)
    other_result = await run_corpus_suite(
        _target(_provider(trials=2), key=corpus.target_key), other, other.suites[0].id
    )
    await store.publish_report(
        EvalReport(
            report_id="other-revision",
            target_key=corpus.target_key,
            program_key="catalog-ci",
            evidence=EvalCorpusReportEvidence(result=other_result),
            association=EvalReportCorpusAssociation(
                corpus_revision=other.revision, suite_id=other.suites[0].id
            ),
        ),
        "catalog-publisher",
        redact_json=_NO_SECRETS.redact_json,
    )
    await hidden()
    accepted = await store.publish_report(
        linked, "catalog-publisher", redact_json=_NO_SECRETS.redact_json
    )
    assert (
        await store.publish_report(linked, "catalog-publisher", redact_json=_NO_SECRETS.redact_json)
        == accepted
    )
    assert await store.load_result_record(result.revision) is None
    items = await catalog()
    assert len({item.key for item in items}) == len(items)
    assert any(item.source == "corpus" and item.revision == corpus.revision for item in items)
    assert {item.source for item in items} == {"authored", "corpus", "reported_suite"}
    history = await store.list_view_index(EvalViewQuery(target_key=corpus.target_key, suite=scope))
    assert {(item.source, item.revision) for item in history.items} == {
        ("run", admitted.run_id),
        ("report", accepted.receipt.id),
    }
    authored = EvalSuiteViewIdentity(
        source="authored", target_key=corpus.target_key, revision=document.revision
    )
    history = await store.list_view_index(
        EvalViewQuery(target_key=corpus.target_key, suite=authored)
    )
    assert [(item.source, item.revision) for item in history.items] == [("run", admitted.run_id)]
    with pytest.raises(ValueError, match="outside the permitted query scope"):
        EvalViewQuery(target_key=corpus.target_key, target_keys=("different-target",))
    excluded = await store.list_view_index(EvalViewQuery(target_keys=("different-target",)))
    assert excluded.items == ()


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_explicit_reports_keep_authored_corpus_catalog_history(tmp_path, backend, request):
    if backend == "postgres":
        from cayu.storage.evals_postgres import PostgresEvalStore
        from cayu.storage.migrations import SchemaMode

        dsn = request.getfixturevalue("postgres_dsn")

    async def exercise():
        store = (
            InMemoryEvalStore()
            if backend == "memory"
            else SQLiteEvalStore(tmp_path / "report-catalog.db")
            if backend == "sqlite"
            else PostgresEvalStore(dsn, schema_mode=SchemaMode.CREATE)
        )
        try:
            await _exercise_report_catalog_dedup(store)
            if isinstance(store, SQLiteEvalStore):
                from cayu.evals.read_views import EvalViewQuery

                fresh = SQLiteEvalStore(store.path)
                try:
                    page = await fresh.list_view_index(
                        EvalViewQuery(target_key="catalog-report-association")
                    )
                    assert sum(item.source == "corpus" for item in page.items) == 2
                finally:
                    await fresh.close()
        finally:
            await store.close()

    asyncio.run(exercise())


def test_sqlite_catalog_uses_json_extract_for_all_nested_identity_fields(tmp_path, monkeypatch):
    """Older supported SQLite builds have JSON1 but cannot parse JSON arrows."""
    import cayu.storage.evals_sqlite as sqlite_store

    original = sqlite_store.view_index_sql
    observed_queries = []

    def compatible_query(query, *, postgres):
        sql, params = original(query, postgres=postgres)
        assert not postgres
        assert "->" not in sql, "SQLite catalog must not require SQLite 3.38 JSON operators"
        observed_queries.append(sql)
        return sql, params

    monkeypatch.setattr(sqlite_store, "view_index_sql", compatible_query)

    async def exercise():
        from cayu.evals.read_views import EvalViewQuery

        store = SQLiteEvalStore(tmp_path / "json1-catalog.db")
        try:
            assert (await store.list_view_index(EvalViewQuery())).items == ()
            await _exercise_authored_catalog_dedup(store, "native")
            await _exercise_authored_catalog_dedup(store, "result")
            await _exercise_report_catalog_dedup(store)
        finally:
            await store.close()

    asyncio.run(exercise())
    assert observed_queries
