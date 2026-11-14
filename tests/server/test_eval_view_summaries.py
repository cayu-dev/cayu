"""The Evals summary tables against a recompute from the documents they summarize.

Every page the dashboard reads from the summary tables (revision 154) must equal
the same page derived from the whole result, corpus and checkpoint documents.
Each history is random: runs with random outcomes, a baseline that moves, a live
run with checkpoints, retention deleting runs, and summaries an older binary
never wrote.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import httpx
import pytest
from tests.evals.eval_store_conformance import _trial_checkpoint
from tests.server.test_server_eval_runs import _bound_eval_invocation
from tests.server.test_server_evals import _AUTH_HEADERS, _server, _target

from cayu.evals.corpus import EvalCaseSpec, EvalCorpusDocument
from cayu.evals.execution import run_corpus_suite
from cayu.evals.read_views import run_view_id
from cayu.evals.store import EvalBaselineKey, EvalBaselineUpdate, EvalRunRequest
from cayu.evals.testing import ScriptedModelProvider
from cayu.evals.view_summary_checks import (
    check_result_view_summary,
    rebuild_result_view_summaries,
)
from cayu.providers.base import ModelStreamEvent
from cayu.storage.evals_sqlite import SQLiteEvalStore
from cayu.storage.retention import EvalRetentionPolicy


class _ReferenceSQLiteEvalStore(SQLiteEvalStore):
    """The same database, read as a store without summaries: every value from documents."""

    view_summaries = False


def _provider(rng: random.Random) -> ScriptedModelProvider:
    def respond(_request):
        output = "Approved" if rng.random() < 0.6 else "Denied"
        return (
            ModelStreamEvent.text_delta(output),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
                }
            ),
        )

    return ScriptedModelProvider(response_factory=respond)


def _corpus(target_key: str, *, cases: int, trials: int) -> EvalCorpusDocument:
    from tests.evals.test_corpus_execution import _corpus as base_corpus

    original = base_corpus(target_key=target_key, trials=trials)
    template = original.cases[0]
    return EvalCorpusDocument.create(
        target_key=target_key,
        evidence_policy=original.evidence_policy,
        pricing_profile=original.pricing_profile,
        suites=original.suites,
        cases=tuple(
            EvalCaseSpec.create(
                id=f"case-{index}",
                suite_id=template.suite_id,
                name=f"Case {index}",
                input=template.input,
                assertions=template.assertions,
                source=template.source,
            )
            for index in range(cases)
        ),
    )


async def _request(target, corpus, run_id: str) -> EvalRunRequest:
    return EvalRunRequest(
        run_id=run_id,
        idempotency_key="sha256:" + hashlib.sha256(run_id.encode()).hexdigest(),
        corpus_revision=corpus.revision,
        target_key=corpus.target_key,
        suite_id=corpus.suites[0].id,
        suite_revision=corpus.suites[0].revision,
        max_concurrency=1,
        invocation=await _bound_eval_invocation(target),
    )


async def _history(store, target, rng: random.Random) -> tuple[EvalCorpusDocument, list[str]]:
    """A random history; returns its corpus and every published result revision."""
    corpus = _corpus(target.key, cases=rng.randint(2, 4), trials=rng.randint(1, 3))
    redact = target.app.redact_json
    await store.save_corpus(corpus, redact_json=redact)
    revisions = []
    generation = 0
    for index in range(rng.randint(3, 5)):
        await store.admit_run(await _request(target, corpus, f"run-{index}"), redact_json=redact)
        lease = await store.claim_run(lease_seconds=300)
        assert lease is not None
        result = await run_corpus_suite(target, corpus, corpus.suites[0].id, max_concurrency=1)
        record = await store.publish_result(lease.claim, result, redact_json=redact)
        assert record.result is not None
        revisions.append(record.result.revision)
        if rng.random() < 0.5:
            await store.set_baseline(
                EvalBaselineUpdate(
                    key=EvalBaselineKey(
                        target_key=target.key,
                        corpus_revision=corpus.revision,
                        suite_id=corpus.suites[0].id,
                    ),
                    result_revision=rng.choice(revisions),
                    expected_generation=generation,
                    operation_id="sha256:" + hashlib.sha256(f"b{index}".encode()).hexdigest(),
                    actor_id="reviewer",
                ),
                redact_json=redact,
            )
            generation += 1
    # A live run with some of its trials checkpointed.
    await store.admit_run(await _request(target, corpus, "live"), redact_json=redact)
    lease = await store.claim_run(lease_seconds=300)
    assert lease is not None
    slots = [
        (case.id, trial)
        for case in corpus.cases
        for trial in range(1, corpus.suites[0].trial_request.trials + 1)
    ]
    for case_id, trial in rng.sample(slots, rng.randint(0, len(slots) - 1)):
        await store.save_trial_checkpoint(
            lease.claim, _trial_checkpoint(case_id, trial), redact_json=redact
        )
    return corpus, revisions


async def _pages(client: httpx.AsyncClient, target_key: str, run_ids: list[str]) -> dict:
    """Every page a dashboard reads for this history, as JSON."""

    async def get(path, **params):
        response = await client.get(f"/api/evals/{path}", params=params, headers=_AUTH_HEADERS)
        return response.status_code, response.json()

    pages: dict = {"catalog": await get("suite-views", target_key=target_key)}
    suite_id = pages["catalog"][1]["items"][0]["id"]
    pages["suite"] = await get(f"suite-views/{suite_id}", cases_limit=2, runs_limit=3)
    cursor = pages["suite"][1]["runs"]["next_cursor"]
    if cursor is not None:
        pages["suite-next"] = await get(f"suite-views/{suite_id}", runs_cursor=cursor)
    views = [run_view_id("run", run_id) for run_id in run_ids]
    for view in views:
        pages[view] = await get(f"run-views/{view}", cases_limit=2)
        status, body = pages[view]
        if status == 200 and body["results"]["next_cursor"] is not None:
            pages[view + ":next"] = await get(
                f"run-views/{view}", cases_limit=2, cases_cursor=body["results"]["next_cursor"]
            )
        if status == 200 and body["run"]["result_revision"] is not None:
            result_view = run_view_id("result", body["run"]["result_revision"])
            pages[result_view] = await get(f"run-views/{result_view}")
        for case_filter in ("regressed", "improved", "failing", "flaky"):
            pages[f"{view}:{case_filter}"] = await get(f"run-views/{view}", case_filter=case_filter)
        pages[f"{view}:compared"] = await get(f"run-views/{view}", compared_with=views[0])
    return pages


@dataclass(frozen=True)
class _Backend:
    """One store under test: with summaries, without them, and plain SQL on it."""

    open: Callable[[], Any]
    open_reference: Callable[[], Any]
    sql: Callable[..., list[tuple[Any, ...]]]
    rerun_migration: Callable[[], None]


def _sqlite_backend(path) -> _Backend:
    from cayu.storage._eval_view_summary_schema import SQLITE_EVAL_VIEW_SUMMARIES_DDL

    def sql(statement: str, *params: object) -> list[tuple[Any, ...]]:
        with sqlite3.connect(path) as raw:
            raw.execute("PRAGMA foreign_keys = ON")
            return raw.execute(statement, params).fetchall()

    def rerun_migration() -> None:
        with sqlite3.connect(path) as raw:
            raw.executescript(SQLITE_EVAL_VIEW_SUMMARIES_DDL)

    return _Backend(
        lambda: SQLiteEvalStore(path), lambda: _ReferenceSQLiteEvalStore(path), sql, rerun_migration
    )


def _postgres_backend(dsn: str) -> _Backend:
    import psycopg

    from cayu.storage._eval_view_summary_schema import POSTGRES_EVAL_VIEW_SUMMARIES_DDL
    from cayu.storage.evals_postgres import PostgresEvalStore
    from cayu.storage.migrations import SchemaMode

    class _ReferencePostgresEvalStore(PostgresEvalStore):
        view_summaries = False

    def sql(statement: str, *params: object) -> list[tuple[Any, ...]]:
        with psycopg.connect(dsn) as raw:
            cursor = raw.execute(statement.replace("?", "%s"), params or None)
            return cursor.fetchall() if cursor.description else []

    def rerun_migration() -> None:
        with psycopg.connect(dsn) as raw:
            for statement in POSTGRES_EVAL_VIEW_SUMMARIES_DDL:
                raw.execute(statement)

    tables = sql(
        "SELECT tablename FROM pg_tables "
        "WHERE schemaname = current_schema() AND tablename LIKE 'cayu_eval_%'"
    )
    if tables:
        sql("TRUNCATE " + ", ".join(row[0] for row in tables) + " CASCADE")
    return _Backend(
        lambda: PostgresEvalStore(dsn, schema_mode=SchemaMode.CREATE),
        lambda: _ReferencePostgresEvalStore(dsn, schema_mode=SchemaMode.CREATE),
        sql,
        rerun_migration,
    )


@pytest.fixture(params=["sqlite", "postgres"])
def backend(request, tmp_path) -> _Backend:
    if request.param == "sqlite":
        return _sqlite_backend(tmp_path / "evals.db")
    return _postgres_backend(request.getfixturevalue("postgres_dsn"))


async def _compare(backend: _Backend, target, run_ids: list[str]) -> None:
    summaries, reference = backend.open(), backend.open_reference()
    try:
        async with (
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=_server(target, summaries)), base_url="http://t"
            ) as fast,
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=_server(target, reference)), base_url="http://t"
            ) as slow,
        ):
            expected = await _pages(slow, target.key, run_ids)
            actual = await _pages(fast, target.key, run_ids)
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            assert actual[key] == value, key
    finally:
        await summaries.close()
        await reference.close()


@pytest.mark.parametrize("seed", range(3))
def test_summary_pages_equal_a_recompute_from_documents(backend, seed):
    rng = random.Random(seed)

    async def exercise():
        target = _target(_provider(rng))
        store = backend.open()
        try:
            _, revisions = await _history(store, target, rng)
            for revision in revisions:
                check = await check_result_view_summary(store, revision)
                assert check.matches, check
        finally:
            await store.close()
        run_ids = [f"run-{index}" for index in range(len(revisions))] + ["live"]
        await _compare(backend, target, run_ids)

        # A result an older binary published has no summary: pages derive it,
        # record it, and still equal the reference.
        backend.sql(
            "DELETE FROM cayu_eval_result_summaries WHERE revision = ?", rng.choice(revisions)
        )
        backend.sql("DELETE FROM cayu_eval_result_comparisons")
        await _compare(backend, target, run_ids)
        store = backend.open()
        try:
            assert await store.list_results_without_view_summaries(10) == ()
            for revision in revisions:
                assert (await check_result_view_summary(store, revision)).matches

            # Retention deletes runs with their results; their summaries go too.
            await store.apply_retention_policy(
                EvalRetentionPolicy(older_than=timedelta(microseconds=1), dry_run=False)
            )
        finally:
            await store.close()
        (orphans,) = backend.sql(
            """SELECT COUNT(*) FROM cayu_eval_result_summaries AS summaries
            WHERE NOT EXISTS (SELECT 1 FROM cayu_eval_result_records AS records
                              WHERE records.revision = summaries.revision)"""
        )
        assert orphans[0] == 0
        kept = [
            row[0]
            for row in backend.sql(
                "SELECT run_id FROM cayu_eval_runs WHERE run_id LIKE 'run-%' ORDER BY run_id"
            )
        ]
        await _compare(backend, target, [*kept, "live"])

    asyncio.run(exercise())


def test_suite_history_refreshes_a_run_published_during_the_read(backend, monkeypatch):
    from cayu.evals.read_views import EvalSuiteViewIdentity

    async def exercise():
        target = _target(_provider(random.Random(9)))
        corpus = _corpus(target.key, cases=2, trials=1)
        store = backend.open()
        try:
            await store.save_corpus(corpus, redact_json=target.app.redact_json)
            await store.admit_run(
                await _request(target, corpus, "finishing"), redact_json=target.app.redact_json
            )
            lease = await store.claim_run(lease_seconds=300)
            assert lease is not None
            result = await run_corpus_suite(target, corpus, corpus.suites[0].id, max_concurrency=1)
            load_latest = store.load_latest_case_results
            published = []

            async def publish_after_history(query):
                # The history already cached this run while it was still running.
                published.append(
                    await store.publish_result(
                        lease.claim, result, redact_json=target.app.redact_json
                    )
                )
                return await load_latest(query)

            monkeypatch.setattr(store, "load_latest_case_results", publish_after_history)
            identity = EvalSuiteViewIdentity(
                source="corpus",
                target_key=target.key,
                revision=corpus.revision,
                suite_id=corpus.suites[0].id,
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=_server(target, store)), base_url="http://t"
            ) as client:
                response = await client.get(
                    f"/api/evals/suite-views/{identity.id}", headers=_AUTH_HEADERS
                )
            assert len(published) == 1
            assert response.status_code == 200, response.text
            cases = response.json()["cases"]["items"]
            assert len(cases) == 2
            assert all(case["last"] is not None for case in cases)
        finally:
            await store.close()

    asyncio.run(exercise())


def test_backfill_and_rebuild_for_results_without_summaries(backend):
    rng = random.Random(7)

    async def exercise():
        target = _target(_provider(rng))
        store = backend.open()
        try:
            _, revisions = await _history(store, target, rng)
        finally:
            await store.close()
        backend.sql("DELETE FROM cayu_eval_result_summaries")
        backend.sql("DELETE FROM cayu_eval_suite_catalog")
        backend.sql("DELETE FROM cayu_eval_trial_checkpoint_summaries")
        # The revision-154 migration, applied again, backfills what triggers keep.
        backend.rerun_migration()
        ((catalog,),) = backend.sql("SELECT COUNT(*) FROM cayu_eval_suite_catalog")
        ((checkpoints,),) = backend.sql("SELECT COUNT(*) FROM cayu_eval_run_trial_checkpoints")
        ((summaries,),) = backend.sql("SELECT COUNT(*) FROM cayu_eval_trial_checkpoint_summaries")
        assert catalog == 1 and summaries == checkpoints
        store = backend.open()
        try:
            assert len(await store.list_results_without_view_summaries(100)) == len(revisions)
            report = await rebuild_result_view_summaries(store, limit=2)
            assert report.written == min(2, len(revisions))
            assert report.remaining == (len(revisions) > 2)
            while (await rebuild_result_view_summaries(store)).remaining:
                pass
            assert await store.list_results_without_view_summaries(100) == ()
            for revision in revisions:
                assert (await check_result_view_summary(store, revision)).matches
        finally:
            await store.close()
        run_ids = [f"run-{index}" for index in range(len(revisions))] + ["live"]
        await _compare(backend, target, run_ids)

    asyncio.run(exercise())


def test_run_view_over_the_byte_budget_pages_the_cases_that_fit(backend):
    rng = random.Random(5)

    async def exercise():
        target = _target(_provider(rng))
        store = backend.open()
        try:
            corpus = _corpus(target.key, cases=4, trials=2)
            redact = target.app.redact_json
            await store.save_corpus(corpus, redact_json=redact)
            await store.admit_run(await _request(target, corpus, "big"), redact_json=redact)
            lease = await store.claim_run(lease_seconds=300)
            assert lease is not None
            result = await run_corpus_suite(target, corpus, corpus.suites[0].id, max_concurrency=1)
            await store.publish_result(lease.claim, result, redact_json=redact)
            path = f"/api/evals/run-views/{run_view_id('run', 'big')}"
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=_server(target, store)), base_url="http://t"
            ) as client:
                whole = await client.get(path, headers=_AUTH_HEADERS)
                assert whole.status_code == 200, whole.text
                cases = [item["case"]["id"] for item in whole.json()["results"]["items"]]
                assert len(cases) == 4
                budget = len(whole.content) * 3 // 4
                seen: list[str] = []
                pages = 0
                cursor = None
                while True:
                    params = {"max_result_bytes": budget}
                    if cursor is not None:
                        params["cases_cursor"] = cursor
                    page = await client.get(path, params=params, headers=_AUTH_HEADERS)
                    assert page.status_code == 200, page.text
                    assert len(page.content) <= budget
                    body = page.json()["results"]
                    assert body["total"] == 4
                    seen += [item["case"]["id"] for item in body["items"]]
                    pages += 1
                    cursor = body["next_cursor"]
                    if cursor is None:
                        break
                assert seen == cases and pages > 1
                tiny = await client.get(
                    path, params={"max_result_bytes": 1024}, headers=_AUTH_HEADERS
                )
                assert tiny.status_code == 413
        finally:
            await store.close()

    asyncio.run(exercise())


def test_live_run_status_answers_304_until_the_run_changes(backend):
    rng = random.Random(3)

    async def exercise():
        target = _target(_provider(rng))
        store = backend.open()
        try:
            corpus = _corpus(target.key, cases=2, trials=2)
            redact = target.app.redact_json
            await store.save_corpus(corpus, redact_json=redact)
            await store.admit_run(await _request(target, corpus, "live"), redact_json=redact)
            view = run_view_id("run", "live")
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=_server(target, store)), base_url="http://t"
            ) as client:
                first = await client.get(
                    f"/api/evals/run-views/{view}/status", headers=_AUTH_HEADERS
                )
                assert first.status_code == 200, first.text
                assert first.json()["status"] == "queued"
                etag = first.headers["etag"]
                again = await client.get(
                    f"/api/evals/run-views/{view}/status",
                    headers={**_AUTH_HEADERS, "If-None-Match": etag},
                )
                assert again.status_code == 304 and again.content == b""
                lease = await store.claim_run(lease_seconds=300)
                assert lease is not None
                claimed = await client.get(
                    f"/api/evals/run-views/{view}/status",
                    headers={**_AUTH_HEADERS, "If-None-Match": etag},
                )
                assert claimed.status_code == 200 and claimed.json()["status"] == "running"
                etag = claimed.headers["etag"]
                await store.heartbeat_run(lease.claim, extend_seconds=300)
                beat = await client.get(
                    f"/api/evals/run-views/{view}/status",
                    headers={**_AUTH_HEADERS, "If-None-Match": etag},
                )
                assert beat.status_code == 304
                await store.save_trial_checkpoint(
                    lease.claim, _trial_checkpoint(corpus.cases[0].id, 1), redact_json=redact
                )
                checkpointed = await client.get(
                    f"/api/evals/run-views/{view}/status",
                    headers={**_AUTH_HEADERS, "If-None-Match": etag},
                )
                assert checkpointed.status_code == 200
                assert checkpointed.json()["completed_trials"] == 1
                missing = await client.get(
                    f"/api/evals/run-views/{run_view_id('run', 'missing')}/status",
                    headers=_AUTH_HEADERS,
                )
                assert missing.status_code == 404
                result_view = await client.get(
                    f"/api/evals/run-views/{run_view_id('result', 'sha256:' + '0' * 64)}/status",
                    headers=_AUTH_HEADERS,
                )
                assert result_view.status_code == 422
        finally:
            await store.close()

    asyncio.run(exercise())


def test_cli_checks_and_rebuilds_a_local_store(tmp_path, capsys):
    import json

    from cayu.cli import main

    rng = random.Random(11)
    path = tmp_path / "evals.db"

    async def seed() -> list[str]:
        store = SQLiteEvalStore(path)
        try:
            return (await _history(store, _target(_provider(rng)), rng))[1]
        finally:
            await store.close()

    revisions = asyncio.run(seed())
    capsys.readouterr()
    assert main(["eval", "runs", "check", "run-0", "--sqlite", str(path), "--json"]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["check"]["matches"] and body["summary"]["revision"] == revisions[0]
    assert {item["case_id"] for item in body["cases"]} == set(body["summary"]["case_ids"])
    assert main(["eval", "runs", "check", "live", "--sqlite", str(path)]) == 0
    assert "no result yet" in capsys.readouterr().out

    _sqlite_backend(path).sql("DELETE FROM cayu_eval_result_summaries")
    assert main(["eval", "runs", "check", "run-0", "--sqlite", str(path)]) == 1
    assert "rebuild-summaries --write" in capsys.readouterr().out
    assert main(["eval", "runs", "rebuild-summaries", "--sqlite", str(path), "--json"]) == 0
    assert set(json.loads(capsys.readouterr().out)["missing"]) == set(revisions)
    assert main(["eval", "runs", "rebuild-summaries", "--write", "--sqlite", str(path)]) == 0
    assert f"Wrote {len(revisions)} result summaries." in capsys.readouterr().out
    assert main(["eval", "runs", "check", "run-0", "--sqlite", str(path)]) == 0
    assert "matches a recompute" in capsys.readouterr().out
