"""Time the Developer dashboard's Evals reads on a store of production size.

Run with ``uv run python scripts/benchmark_eval_view_reads.py``. It builds an eval
store (by default 200 suites in 20 corpora, 25 finished runs per suite, so 5,000,
100 trials per run, and three live runs, one with 40 MB of trial checkpoints),
then times, through the real server:

- ``GET /api/evals/suite-views`` (the suite list),
- ``GET /api/evals/suite-views/{suite}`` (suite detail),
- ``GET /api/evals/run-views/{run}`` for a finished run, a page of its failing
  cases and a live run,
- ``GET /api/evals/runs/{run}/result`` (a run's whole result), and
- ``GET /api/evals/run-views/{run}/status``, the live poll, changed and unchanged.

Every number is the p95 of ``--repeat`` requests, each against a freshly opened
store and server, so no in-process cache is warm. Pass ``--store PATH`` to keep or
reuse a SQLite store, or ``--postgres DSN`` to build and read a Postgres one.

Each corpus executes one suite for real against a scripted provider; the result is
rewritten for the corpus's other suites, which have the same shape, and every
suite's run is published, made its baseline and compared with it through the
store API. The suite's other runs are SQL copies of it, an hour apart, which share
its result document under revisions of their own. The copies are for measurement
only; they are not a supported way to write a store.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from cayu import AgentSpec, CayuApp, Message, RunRequest
from cayu.evals import ScriptedModelProvider
from cayu.evals.corpus import (
    CorpusUserMessageSpec,
    EvalCaseSpec,
    EvalCorpusDocument,
    EvalSuiteSpec,
    EvaluationEvidencePolicySpec,
    EvaluationSourceIdentityV1,
    FinalOutputEqualsAssertionSpec,
    RootStatusAssertionSpec,
    RunInputSpec,
    TrialRequestSpec,
    _content_revision,
)
from cayu.evals.execution import CorpusExecutionResult, CorpusTarget, run_corpus_suite
from cayu.evals.models import EvalStatus, EvalTrialResult
from cayu.evals.read_views import run_view_id
from cayu.evals.result_contract import (
    EvalTrialDiagnosticCode,
    EvalTrialOutputPreviewV1,
    _EvalTrialPublicData,
)
from cayu.evals.store import EvalRunInvocation, EvalRunRequest, EvalRunTrialCheckpoint
from cayu.providers import ModelStreamEvent
from cayu.server import ServerConfig, create_server
from cayu.server.config import DashboardConfig, EvalsConfig

TARGET_KEY = "bench-agent"
_AUTH = {"Authorization": "Bearer bench"}
BUDGET_SECONDS = 0.3


def _provider(seed: int) -> ScriptedModelProvider:
    rng = random.Random(seed)

    def respond(_request):
        output = "Approved" if rng.random() < 0.8 else "Denied"
        return (
            ModelStreamEvent.text_delta(output),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {"input_tokens": 120, "output_tokens": 4, "total_tokens": 124},
                }
            ),
        )

    return ScriptedModelProvider(response_factory=respond)


def _target(seed: int = 0) -> CorpusTarget:
    app = CayuApp(enable_logging=False)
    app.register_provider(_provider(seed), default=True)
    app.register_agent(AgentSpec(name="agent", model="fixture-model"))
    return CorpusTarget(
        key=TARGET_KEY,
        app=app,
        request_base=RunRequest(agent_name="agent", messages=[], max_steps=1),
        bootstrap_messages=(Message.text("system", "Follow the refund policy."),),
        application_release_id="bench-release",
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
    )


def _source() -> EvaluationSourceIdentityV1:
    return EvaluationSourceIdentityV1(
        application_release_id="captured-release",
        app_manifest_schema_version="7",
        app_manifest_fingerprint="a" * 64,
        evidence_revision="sha256:" + "e" * 64,
    )


def _corpus(index: int, *, suites: int, cases: int, trials: int) -> EvalCorpusDocument:
    suite_specs = []
    case_specs = []
    for s in range(suites):
        suite = EvalSuiteSpec.create(
            id=f"suite-{index:03d}-{s:02d}",
            name=f"Refund regressions {index}.{s}",
            trial_request=TrialRequestSpec(trials=trials, timeout_seconds=30),
        )
        suite_specs.append(suite)
        for c in range(cases):
            case_specs.append(
                EvalCaseSpec.create(
                    id=f"case-{s:02d}-{c:03d}",
                    suite_id=suite.id,
                    name=f"Refund case {c}",
                    source=_source(),
                    input=RunInputSpec(
                        messages=(CorpusUserMessageSpec(text=f"Refund order {index}-{s}-{c}."),)
                    ),
                    assertions=(
                        RootStatusAssertionSpec(id="completed", expected="completed"),
                        FinalOutputEqualsAssertionSpec(id="answer", expected="Approved"),
                    ),
                )
            )
    return EvalCorpusDocument.create(
        target_key=TARGET_KEY,
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
        suites=tuple(suite_specs),
        cases=tuple(case_specs),
    )


def _request(corpus: EvalCorpusDocument, suite: EvalSuiteSpec, run_id: str) -> EvalRunRequest:
    digest = _content_revision({"run": run_id}, "benchmark run")
    return EvalRunRequest(
        run_id=run_id,
        idempotency_key=digest,
        corpus_revision=corpus.revision,
        target_key=TARGET_KEY,
        suite_id=suite.id,
        suite_revision=suite.revision,
        max_concurrency=1,
        invocation=EvalRunInvocation(),
    )


def _for_suite(
    template: CorpusExecutionResult,
    corpus: EvalCorpusDocument,
    source: EvalSuiteSpec,
    suite: EvalSuiteSpec,
) -> CorpusExecutionResult:
    """``template``'s run of ``source`` as a run of ``suite``, a suite of the same shape."""
    if suite.id == source.id:
        return template
    mapping = {source.id: suite.id, source.revision: suite.revision}
    cases = {case.id: case for case in corpus.cases}
    for case in corpus.cases:
        if case.suite_id == source.id:
            other = cases[case.id.replace(source.id[-2:] + "-", suite.id[-2:] + "-", 1)]
            mapping[case.id] = other.id
            mapping[case.revision] = other.revision
    text = json.dumps(template.run.model_dump(mode="json"))
    for old, new in mapping.items():
        text = text.replace(old, new)
    document = json.loads(text)
    document.pop("revision")
    document["revision"] = _content_revision(document, "published eval run")
    run = type(template.run).model_validate_json(json.dumps(document))
    return CorpusExecutionResult.create(target=template.target, run=run)


def _checkpoint(case_id: str, trial_number: int, padding: int) -> EvalRunTrialCheckpoint:
    now = datetime.now(UTC)
    return EvalRunTrialCheckpoint(
        case_id=case_id,
        result=EvalTrialResult(
            trial_number=trial_number,
            status=EvalStatus.ERROR,
            error="candidate execution failed " + "x" * padding,
            started_at=now,
            completed_at=now,
        ),
        public_data=_EvalTrialPublicData(
            diagnostic_code=EvalTrialDiagnosticCode.EXECUTION_FAILED,
            output=EvalTrialOutputPreviewV1.unavailable(),
        ),
    )


async def build(store, args: argparse.Namespace) -> None:
    """One real run per suite, through the store API; `copy_history` adds the rest."""
    from cayu.evals.store import EvalBaselineKey, EvalBaselineUpdate
    from cayu.evals.view_comparison import record_suite_baseline_comparisons

    target = _target()
    redact = target.app.redact_json
    per_corpus = args.suites // args.corpora
    trials_per_case = max(1, args.trials // args.cases)
    started = time.perf_counter()
    for c in range(args.corpora):
        corpus = _corpus(c, suites=per_corpus, cases=args.cases, trials=trials_per_case)
        await store.save_corpus(corpus, redact_json=redact)
        source = corpus.suites[0]
        template = await run_corpus_suite(target, corpus, source.id, max_concurrency=1)
        for suite in corpus.suites:
            result = _for_suite(template, corpus, source, suite)
            await store.admit_run(
                _request(corpus, suite, f"run-{suite.id}-000"), redact_json=redact
            )
            lease = await store.claim_run(lease_seconds=3600)
            assert lease is not None
            await store.publish_result(lease.claim, result, redact_json=redact)
            key = EvalBaselineKey(
                target_key=TARGET_KEY, corpus_revision=corpus.revision, suite_id=suite.id
            )
            await store.set_baseline(
                EvalBaselineUpdate(
                    key=key,
                    result_revision=result.revision,
                    expected_generation=0,
                    operation_id=_content_revision({"baseline": suite.id}, "benchmark"),
                    actor_id="benchmark",
                ),
                redact_json=redact,
            )
            await record_suite_baseline_comparisons(store, key)
        print(f"corpus {c + 1}/{args.corpora} in {time.perf_counter() - started:.0f}s", flush=True)
    # Live runs: their trial checkpoints are what a live run view polls.
    corpus = _corpus(0, suites=per_corpus, cases=args.cases, trials=trials_per_case)
    suite = corpus.suites[0]
    slots = [
        (case.id, trial)
        for case in corpus.cases
        if case.suite_id == suite.id
        for trial in range(1, trials_per_case + 1)
    ][:-1]  # The last trial stays open, so the run stays live.
    for index in range(args.live):
        await store.admit_run(_request(corpus, suite, f"live-{index:02d}"), redact_json=redact)
        lease = await store.claim_run(lease_seconds=3600)
        assert lease is not None
        # The first live runs carry 40 MB of checkpoints, the store's limit.
        padding = ((40 << 20) - 2048 * len(slots)) // len(slots) if index < args.large_live else 0
        for case_id, trial_number in slots:
            await store.save_trial_checkpoint(
                lease.claim, _checkpoint(case_id, trial_number, padding), redact_json=redact
            )


_COPIED = (
    "cayu_eval_runs",
    "cayu_eval_results",
    "cayu_eval_result_records",
    "cayu_eval_result_summaries",
    "cayu_eval_case_outcomes",
    "cayu_eval_result_comparisons",
)


def copy_history(connect: Callable[[], Any], *, runs_per_suite: int, postgres: bool) -> int:
    """Copy each suite's real run ``runs_per_suite - 1`` times, an hour apart, with SQL.

    The copies share their source's result document under a revision of their own,
    which no read of a summary page decodes. They are for measurement only; this is
    not a supported way to write a store.
    """
    mark = "%s" if postgres else "?"
    connection = connect()
    copied = 0
    try:
        cursor = connection.cursor()

        def rows(table: str, where: str, value: object) -> list[dict[str, Any]]:
            cursor.execute(f"SELECT * FROM {table} WHERE {where} = {mark}", (value,))
            names = [item[0] for item in cursor.description]
            return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]

        def insert(table: str, items: list[dict[str, Any]]) -> None:
            if not items:
                return
            names = list(items[0])
            cast = {
                "summary_json": "::jsonb",
                "outcome_json": "::jsonb",
                "comparison_json": "::jsonb",
            }
            values = ", ".join(mark + (cast.get(name, "") if postgres else "") for name in names)
            cursor.executemany(
                f"INSERT INTO {table} ({', '.join(names)}) VALUES ({values})",
                [
                    tuple(
                        json.dumps(item[name])
                        if postgres and name in cast and not isinstance(item[name], str)
                        else item[name]
                        for name in names
                    )
                    for item in items
                ],
            )

        def shifted(value: Any, hours: int) -> Any:
            if value is None:
                return None
            if isinstance(value, str):
                return (datetime.fromisoformat(value) - timedelta(hours=hours)).isoformat()
            return value - timedelta(hours=hours)

        cursor.execute("SELECT run_id FROM cayu_eval_runs WHERE run_id LIKE 'run-%-000'")
        sources = [row[0] for row in cursor.fetchall()]
        for source_id in sources:
            (run,) = rows("cayu_eval_runs", "run_id", source_id)
            revision = run["result_revision"]
            (result,) = rows("cayu_eval_results", "run_id", source_id)
            (record,) = rows("cayu_eval_result_records", "revision", revision)
            (summary,) = rows("cayu_eval_result_summaries", "revision", revision)
            outcomes = rows("cayu_eval_case_outcomes", "result_revision", revision)
            comparisons = rows("cayu_eval_result_comparisons", "current_revision", revision)
            batch: dict[str, list[dict[str, Any]]] = {table: [] for table in _COPIED}
            for k in range(1, runs_per_suite):
                run_id = source_id[: -len("000")] + f"{k:03d}"
                copy = _content_revision({"run": run_id}, "benchmark copy")
                batch["cayu_eval_runs"].append(
                    {
                        **run,
                        "run_id": run_id,
                        "idempotency_key": _content_revision({"key": run_id}, "benchmark copy"),
                        "result_revision": copy,
                        **{
                            name: shifted(run[name], k)
                            for name in ("created_at", "updated_at", "started_at", "finished_at")
                        },
                    }
                )
                batch["cayu_eval_results"].append(
                    {
                        **result,
                        "run_id": run_id,
                        "revision": copy,
                        "created_at": shifted(result["created_at"], k),
                    }
                )
                batch["cayu_eval_result_records"].append(
                    {
                        **record,
                        "revision": copy,
                        "fresh_run_id": run_id,
                        "created_at": shifted(record["created_at"], k),
                    }
                )
                text = summary["summary_json"]
                text = text if isinstance(text, str) else json.dumps(text)
                batch["cayu_eval_result_summaries"].append(
                    {"revision": copy, "summary_json": text.replace(revision, copy)}
                )
                batch["cayu_eval_case_outcomes"].extend(
                    {**item, "result_revision": copy} for item in outcomes
                )
                batch["cayu_eval_result_comparisons"].extend(
                    {**item, "current_revision": copy} for item in comparisons
                )
            for table in _COPIED:
                insert(table, batch[table])
            connection.commit()
            copied += runs_per_suite - 1
    finally:
        connection.close()
    return copied


@asynccontextmanager
async def _client(open_store: Callable[[], Any]) -> AsyncIterator[httpx.AsyncClient]:
    store = open_store()
    target = _target()
    server = create_server(
        target.app,
        config=ServerConfig.protected(
            lambda request: _auth(request),
            dashboard=DashboardConfig(enabled=False),
            evals=EvalsConfig(target=target, store=store),
        ),
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server), base_url="http://bench", timeout=600
        ) as client:
            yield client
    finally:
        await store.close()


def _auth(request):
    from fastapi import HTTPException

    from cayu.server.auth import AuthContext

    if request.headers.get("Authorization") != _AUTH["Authorization"]:
        raise HTTPException(status_code=401)
    return AuthContext(subject="bench")


async def _timed(
    open_store, path: str, params: dict[str, Any], headers: dict[str, str], repeat: int
) -> tuple[float, int, int]:
    samples = []
    status = size = 0
    for _ in range(repeat):
        async with _client(open_store) as client:
            begin = time.perf_counter()
            response = await client.get(f"/api{path}", params=params, headers={**_AUTH, **headers})
            samples.append(time.perf_counter() - begin)
            status, size = response.status_code, len(response.content)
            if status not in {200, 304}:
                print(path, status, response.text[:300])
    samples.sort()
    p95 = samples[min(len(samples) - 1, round(0.95 * (len(samples) - 1)))]
    return p95, status, size


async def measure(open_store, repeat: int) -> dict[str, float]:
    async with _client(open_store) as client:
        catalog = (
            await client.get("/api/evals/suite-views", params={"limit": 25}, headers=_AUTH)
        ).json()
        suite = catalog["items"][0]
    finished = suite["last_run"]["native_run_id"]
    requests: list[tuple[str, str, dict[str, Any], dict[str, str]]] = [
        ("suite list", "/evals/suite-views", {"limit": 25}, {}),
        (
            "suite detail",
            f"/evals/suite-views/{suite['id']}",
            {"cases_limit": 25, "runs_limit": 20},
            {},
        ),
        (
            "finished run view",
            f"/evals/run-views/{run_view_id('run', finished)}",
            {"cases_limit": 25},
            {},
        ),
        (
            "case page, failing",
            f"/evals/run-views/{run_view_id('run', finished)}",
            {"cases_limit": 25, "case_filter": "failing"},
            {},
        ),
        ("run result", f"/evals/runs/{finished}/result", {}, {}),
        (
            "live run view",
            f"/evals/run-views/{run_view_id('run', 'live-00')}",
            {"cases_limit": 25},
            {},
        ),
        ("live poll", f"/evals/run-views/{run_view_id('run', 'live-00')}/status", {}, {}),
    ]
    async with _client(open_store) as client:
        status = await client.get(f"/api{requests[-1][1]}", headers=_AUTH)
    if status.status_code == 200:
        requests.append(
            (
                "live poll, unchanged",
                requests[-1][1],
                {},
                {"If-None-Match": status.headers["etag"]},
            )
        )
    timings = {}
    for name, path, params, headers in requests:
        p95, code, size = await _timed(open_store, path, params, headers, repeat)
        timings[name] = p95
        verdict = "ok" if p95 <= BUDGET_SECONDS else "OVER"
        print(
            f"{name:22s} p95 {p95 * 1000:8.1f} ms  {code}  {size / 1024:9.1f} KB  {verdict}",
            flush=True,
        )
    return timings


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--suites", type=int, default=200)
    parser.add_argument("--corpora", type=int, default=20)
    parser.add_argument("--runs", type=int, default=5000)
    parser.add_argument("--cases", type=int, default=10)
    parser.add_argument("--trials", type=int, default=100, help="Trials per run.")
    parser.add_argument("--live", type=int, default=3)
    parser.add_argument(
        "--large-live", type=int, default=1, help="Live runs with 40 MB of checkpoints."
    )
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument(
        "--store", type=Path, help="SQLite store to build, or reuse when it exists."
    )
    parser.add_argument("--postgres", help="Postgres DSN to build and read instead of SQLite.")
    parser.add_argument("--skip-build", action="store_true", help="Reuse the store as it is.")
    args = parser.parse_args()
    runs_per_suite = max(1, args.runs // args.suites)

    if args.postgres:
        import psycopg

        from cayu.storage.evals_postgres import PostgresEvalStore
        from cayu.storage.migrations import SchemaMode

        def open_store():
            return PostgresEvalStore(args.postgres, schema_mode=SchemaMode.CREATE)

        def connect():
            return psycopg.connect(args.postgres)
    else:
        import sqlite3
        import tempfile

        from cayu.storage.evals_sqlite import SQLiteEvalStore

        path = args.store or Path(tempfile.mkdtemp(prefix="cayu-eval-bench-")) / "evals.db"
        args.skip_build = args.skip_build or path.exists()

        def open_store():
            return SQLiteEvalStore(path)

        def connect():
            return sqlite3.connect(path)

    async def run() -> None:
        if not args.skip_build:
            store = open_store()
            try:
                await build(store, args)
            finally:
                await store.close()
            copied = copy_history(
                connect, runs_per_suite=runs_per_suite, postgres=bool(args.postgres)
            )
            print(f"copied {copied} runs", flush=True)
        await measure(open_store, args.repeat)

    asyncio.run(run())


if __name__ == "__main__":
    main()
