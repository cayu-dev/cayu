from __future__ import annotations

import asyncio

import httpx
import pytest
from tests.server.test_server_eval_runs import _admit_run
from tests.server.test_server_eval_suite_authoring import _draft
from tests.server.test_server_evals import _AUTH_HEADERS, _provider, _server, _target

from cayu.evals.store import InMemoryEvalStore
from cayu.evals.suite_authoring import compile_eval_suite_authoring_draft
from cayu.storage.evals_sqlite import SQLiteEvalStore


@pytest.mark.parametrize("backend", ["sqlite"])
def test_suite_views_unify_authored_and_corpus_and_report_exact_queued_progress(tmp_path, backend):
    async def exercise():
        target = _target(_provider())
        store = (
            InMemoryEvalStore() if backend == "memory" else SQLiteEvalStore(tmp_path / "evals.db")
        )
        try:
            admitted = await _admit_run(store, target, "queued-run")
            authored = compile_eval_suite_authoring_draft(
                _draft().model_copy(update={"target_key": target.key})
            )
            await store.save_authored_suite(authored, redact_json=target.app.redact_json)
            server = _server(target, store)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                response = await client.get(
                    "/api/evals/suite-views",
                    params={"target_key": target.key, "limit": 1},
                    headers=_AUTH_HEADERS,
                )
                assert response.status_code == 200, response.text
                first = response.json()
                assert len(first["items"]) == 1 and first["has_more"]
                response = await client.get(
                    "/api/evals/suite-views",
                    params={"target_key": target.key, "limit": 1, "cursor": first["next_cursor"]},
                    headers=_AUTH_HEADERS,
                )
                assert response.status_code == 200, response.text
                second = response.json()
                assert not second["has_more"]
                suites = first["items"] + second["items"]
                assert {item["source"] for item in suites} == {"authored", "corpus"}
                corpus_suite = next(item for item in suites if item["source"] == "corpus")
                detail = await client.get(
                    f"/api/evals/suite-views/{corpus_suite['id']}", headers=_AUTH_HEADERS
                )
                assert detail.status_code == 200, detail.text
                body = detail.json()
                assert body["cases"]["total"] == corpus_suite["case_count"]
                run = body["runs"]["items"][0]
                assert run["native_run_id"] == admitted.id
                assert run["started_at"] is None
                run_response = await client.get(
                    f"/api/evals/run-views/{run['id']}", headers=_AUTH_HEADERS
                )
                assert run_response.status_code == 200, run_response.text
                run_body = run_response.json()
                assert run_body["progress"]["completed_trials"] == 0
                assert run_body["progress"]["total_trials"] == corpus_suite["case_count"]
                assert run_body["comparison"]["state"] == "pending"
                assert run_body["comparison"]["verdict"] is None
                import base64
                import json

                missing = (
                    base64.urlsafe_b64encode(
                        json.dumps(["run", "missing-run"], separators=(",", ":")).encode()
                    )
                    .rstrip(b"=")
                    .decode()
                )
                rejected = await client.get(
                    f"/api/evals/run-views/{run['id']}",
                    params={"compared_with": missing},
                    headers=_AUTH_HEADERS,
                )
                assert rejected.status_code == 404
                # The global list filters stale/unpublished targets before pagination.
                await _admit_run(store, target, "unpublished-run", target_key="unpublished")
                global_catalog = await client.get("/api/evals/suite-views", headers=_AUTH_HEADERS)
                assert global_catalog.status_code == 200, global_catalog.text
                assert len(global_catalog.json()["items"]) == 2
                assert {
                    item["identity"]["target_key"] for item in global_catalog.json()["items"]
                } == {target.key}

        finally:
            await store.close()

    asyncio.run(exercise())


def test_run_views_use_authoritative_comparison_and_preserve_trial_evidence(tmp_path, monkeypatch):
    from tests.evals.test_corpus_execution import _corpus
    from tests.evals.test_eval_store import _request

    from cayu.evals.corpus import EvalCaseSpec, EvalCorpusDocument
    from cayu.evals.execution import run_corpus_suite
    from cayu.evals.store import EvalBaselineKey, EvalBaselineUpdate
    from cayu.evals.testing import ScriptedModelProvider
    from cayu.providers.base import ModelStreamEvent

    async def exercise():
        outputs = ["Approved", "Denied", "Denied", "Approved", "Approved", "Approved"]
        provider = ScriptedModelProvider(
            [
                (
                    ModelStreamEvent.text_delta(output),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                )
                for output in outputs
            ]
        )
        target = _target(provider)
        original = _corpus(trials=1)
        first_case = original.cases[0]
        second_case = EvalCaseSpec.create(
            id="second-case",
            suite_id=first_case.suite_id,
            name="Second case",
            input=first_case.input,
            assertions=first_case.assertions,
            source=first_case.source,
        )
        corpus = EvalCorpusDocument.create(
            target_key=target.key,
            evidence_policy=original.evidence_policy,
            pricing_profile=original.pricing_profile,
            suites=original.suites,
            cases=(first_case, second_case),
        )
        store = SQLiteEvalStore(tmp_path / "comparisons.db")
        try:
            await store.save_corpus(corpus, redact_json=target.app.redact_json)
            for run_id in ("left", "right", "good"):
                await store.admit_run(
                    _request(corpus, run_id=run_id, key=run_id), redact_json=target.app.redact_json
                )
                lease = await store.claim_run(lease_seconds=300)
                result = await run_corpus_suite(
                    target, corpus, corpus.suites[0].id, max_concurrency=1
                )
                await store.publish_result(lease.claim, result, redact_json=target.app.redact_json)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=_server(target, store)), base_url="http://test"
            ) as client:
                catalog = await client.get(
                    "/api/evals/suite-views",
                    params={"target_key": target.key},
                    headers=_AUTH_HEADERS,
                )
                assert catalog.status_code == 200, catalog.text
                suite_id = catalog.json()["items"][0]["id"]
                suite = await client.get(
                    f"/api/evals/suite-views/{suite_id}", headers=_AUTH_HEADERS
                )
                assert suite.status_code == 200, suite.text
                runs = {item["native_run_id"]: item for item in suite.json()["runs"]["items"]}
                no_baseline = await client.get(
                    f"/api/evals/run-views/{runs['good']['id']}", headers=_AUTH_HEADERS
                )
                assert no_baseline.status_code == 200, no_baseline.text
                assert no_baseline.json()["comparison"]["state"] == "no_baseline"
                for baseline, current, verdict, improved, regressed in (
                    ("left", "right", "tradeoff", 1, 1),
                    ("left", "good", "improvement", 1, 0),
                    ("good", "left", "regression", 0, 1),
                    ("good", "good", "tie", 0, 0),
                ):
                    response = await client.get(
                        f"/api/evals/run-views/{runs[current]['id']}",
                        params={"compared_with": runs[baseline]["id"], "cases_limit": 1},
                        headers=_AUTH_HEADERS,
                    )
                    assert response.status_code == 200, response.text
                    body = response.json()
                    assert body["comparison"]["verdict"] == verdict
                    assert body["comparison"]["improved_cases"] == improved
                    assert body["comparison"]["regressed_cases"] == regressed
                    assert body["compared_run"]["native_run_id"] == baseline
                    assert body["compared_run"]["cases"]["total"] == 2
                    assert body["compared_run"]["suite_name"] == corpus.suites[0].name
                    assert body["run"]["suite_name"] == corpus.suites[0].name
                    assert body["run"]["specialist"] == target.request_base.agent_name
                    assert all(
                        type(check["judge"]) is bool
                        for check in body["results"]["items"][0]["case"]["checks"]
                    )
                    assert body["results"]["total"] == 2
                    assert len(body["results"]["items"]) == 1 and body["results"]["has_more"]
                    assert body["progress"]["completed_trials"] == 2
                    assert body["progress"]["total_trials"] == 2
                    assert body["results"]["items"][0]["evidence"]["trials"][0]["output"]
                    next_response = await client.get(
                        f"/api/evals/run-views/{runs[current]['id']}",
                        params={
                            "compared_with": runs[baseline]["id"],
                            "cases_limit": 1,
                            "cases_cursor": body["results"]["next_cursor"],
                        },
                        headers=_AUTH_HEADERS,
                    )
                    assert next_response.status_code == 200, next_response.text
                    assert not next_response.json()["results"]["has_more"]
                await store.set_baseline(
                    EvalBaselineUpdate(
                        key=EvalBaselineKey(
                            target_key=target.key,
                            corpus_revision=corpus.revision,
                            suite_id=corpus.suites[0].id,
                        ),
                        result_revision=runs["left"]["result_revision"],
                        expected_generation=0,
                        operation_id="sha256:" + "3" * 64,
                        actor_id="reviewer",
                    ),
                    redact_json=target.app.redact_json,
                )
                response = await client.get(
                    f"/api/evals/run-views/{runs['good']['id']}", headers=_AUTH_HEADERS
                )
                assert response.status_code == 200, response.text
                assert response.json()["comparison"]["verdict"] == "improvement"
                assert response.json()["run"]["baseline"]["updated_by"] == "reviewer"
                baseline_reads = 0
                load_baseline = store.load_baseline

                async def observe_baseline(key):
                    nonlocal baseline_reads
                    baseline_reads += 1
                    return await load_baseline(key)

                monkeypatch.setattr(store, "load_baseline", observe_baseline)
                response = await client.get(
                    f"/api/evals/suite-views/{suite_id}", headers=_AUTH_HEADERS
                )
                assert response.status_code == 200, response.text
                assert baseline_reads == 1
                assert all(
                    item["baseline"]["generation"] == 1 for item in response.json()["runs"]["items"]
                )

        finally:
            await store.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("custom_roles", [False, True])
def test_member_views_are_read_only_and_auth_precedes_query_validation(tmp_path, custom_roles):
    from tests.server.test_server_evals import _authenticate, _evals_config

    from cayu.server import (
        AuthContext,
        ControlPlaneRoles,
        DashboardConfig,
        ServerConfig,
        create_server,
    )

    class MemberAuth:
        control_plane_roles = ControlPlaneRoles(
            role_claim="access" if custom_roles else "role",
            full_access=("owner",) if custom_roles else ("admin", "developer"),
            read_and_decide=("reader",) if custom_roles else ("member",),
        )
        calls = 0

        def __call__(self, request):
            self.calls += 1
            _authenticate(request)
            context = AuthContext(
                subject="viewer",
                claims={
                    "access" if custom_roles else "role": "reader" if custom_roles else "member"
                },
            )
            self.control_plane_roles.require(request, context)
            return context

    async def exercise():
        target = _target(_provider())
        store = SQLiteEvalStore(tmp_path / "roles.db")
        auth = MemberAuth()
        try:
            await _admit_run(store, target, "protected-run")
            server = create_server(
                target.app,
                config=ServerConfig.protected(
                    auth,
                    dashboard=DashboardConfig(enabled=False),
                    evals=_evals_config(target, store),
                ),
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                denied = await client.get("/api/evals/suite-views?limit=invalid")
                assert denied.status_code == 401
                auth.calls = 0
                response = await client.get(
                    "/api/evals/suite-views",
                    params={"target_key": target.key},
                    headers=_AUTH_HEADERS,
                )
                assert response.status_code == 200, response.text
                assert auth.calls == 1
                assert response.headers["cache-control"] == "private, no-store"
                suite = response.json()["items"][0]
                resolved = await client.get(
                    "/api/evals/suite-views/resolve-corpus",
                    params={
                        "corpus": suite["identity"]["revision"],
                        "suite": suite["identity"]["suite_id"],
                    },
                    headers=_AUTH_HEADERS,
                )
                assert resolved.status_code == 200, resolved.text
                assert resolved.json() == suite["identity"]
                raw = await client.get(
                    f"/api/evals/corpora/{suite['identity']['revision']}", headers=_AUTH_HEADERS
                )
                assert raw.status_code == 403
                denied_resolution = await client.get(
                    "/api/evals/suite-views/resolve-corpus?corpus=invalid&suite=unknown"
                )
                assert denied_resolution.status_code == 401
                store.suite_views = False
                unavailable = await client.get(
                    "/api/evals/suite-views/resolve-corpus",
                    params={
                        "corpus": suite["identity"]["revision"],
                        "suite": suite["identity"]["suite_id"],
                    },
                    headers=_AUTH_HEADERS,
                )
                assert unavailable.status_code == 409
                store.suite_views = True
                for path in (
                    f"/api/evals/suite-views/{suite['id']}",
                    f"/api/evals/run-views/{suite['last_run']['id']}",
                ):
                    response = await client.get(path, headers=_AUTH_HEADERS)
                    assert response.status_code == 200, response.text
                for path in (
                    "/api/evals/runs",
                    "/api/evals/runs/protected-run/cancel",
                    "/api/evals/results/missing/baseline",
                ):
                    response = await client.post(path, json={}, headers=_AUTH_HEADERS)
                    assert response.status_code == 403, response.text
                invalid = await client.get(
                    "/api/evals/run-views/not-a-selector", headers=_AUTH_HEADERS
                )
                assert invalid.status_code == 422
                unknown = await client.get(
                    "/api/evals/suite-views",
                    params={"target_key": "unknown"},
                    headers=_AUTH_HEADERS,
                )
                assert unknown.status_code == 404
                bounded = await client.get(
                    f"/api/evals/suite-views/{suite['id']}",
                    params={"max_result_bytes": 1024},
                    headers=_AUTH_HEADERS,
                )
                assert bounded.status_code == 413
                capability = await client.get("/api/contract", headers=_AUTH_HEADERS)
                surface = capability.json()["capabilities"]["surfaces"]["eval_suite_views"]
                assert surface["read"]["enabled"] is True
                assert surface["mutate"]["enabled"] is False
        finally:
            await store.close()

    asyncio.run(exercise())


def test_captured_results_have_no_invented_run_identity_or_progress(tmp_path):
    from tests.evals.eval_store_conformance import captured_result_for_corpus
    from tests.evals.test_corpus_execution import _corpus

    from cayu.evals.execution import run_corpus_suite

    async def exercise():
        target = _target(_provider(trials=1))
        corpus = _corpus(trials=1)
        fresh = await run_corpus_suite(target, corpus, corpus.suites[0].id, max_concurrency=1)
        captured = captured_result_for_corpus(corpus, fresh)
        store = SQLiteEvalStore(tmp_path / "captured.db")
        try:
            await store.save_captured_result(corpus, captured, redact_json=target.app.redact_json)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=_server(target, store)), base_url="http://test"
            ) as client:
                response = await client.get(
                    "/api/evals/suite-views",
                    params={"target_key": target.key},
                    headers=_AUTH_HEADERS,
                )
                assert response.status_code == 200, response.text
                summary = response.json()["items"][0]
                run = summary["last_run"]
                assert run["native_run_id"] is None
                assert run["result_revision"] == captured.revision
                assert run["origin"] == "captured_session" and run["actor"] is None
                response = await client.get(
                    f"/api/evals/run-views/{run['id']}", headers=_AUTH_HEADERS
                )
                assert response.status_code == 200, response.text
                detail = response.json()
                assert (
                    detail["progress"]["total_trials"]
                    == detail["progress"]["completed_trials"]
                    == 1
                )
                assert detail["run"]["usage"]["tokens"] is None
                assert detail["run"]["candidate"] is None
                assert detail["results"]["items"][0]["evidence"] is None
                response = await client.get(
                    f"/api/evals/suite-views/{summary['id']}", headers=_AUTH_HEADERS
                )
                assert response.status_code == 200, response.text
                assert response.json()["cases"]["items"][0]["last"]["status"] == "passed"
        finally:
            await store.close()

    asyncio.run(exercise())


def test_run_progress_uses_terminal_checkpoint_slots_and_unknown_terminal_counts(tmp_path):
    from tests.evals.eval_store_conformance import _terminal_trial_checkpoint
    from tests.evals.test_corpus_execution import _corpus

    async def exercise():
        target = _target(_provider())
        store = SQLiteEvalStore(tmp_path / "progress.db")
        try:
            await _admit_run(store, target, "progress-run")
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=_server(target, store)), base_url="http://test"
            ) as client:
                catalog = await client.get(
                    "/api/evals/suite-views",
                    params={"target_key": target.key},
                    headers=_AUTH_HEADERS,
                )
                assert catalog.status_code == 200, catalog.text
                run_id = catalog.json()["items"][0]["last_run"]["id"]
                lease = await store.claim_run(lease_seconds=300)
                checkpoint = _terminal_trial_checkpoint(_corpus(trials=1))
                await store.save_trial_checkpoint(
                    lease.claim, checkpoint, redact_json=target.app.redact_json
                )
                for status in ("running", "cancelling"):
                    if status == "cancelling":
                        await store.request_cancel("progress-run")
                    response = await client.get(
                        f"/api/evals/run-views/{run_id}", headers=_AUTH_HEADERS
                    )
                    assert response.status_code == 200, response.text
                    assert response.json()["run"]["status"] == status
                    assert response.json()["progress"]["completed_trials"] == 1
                    assert response.json()["progress"]["completed_cases"] == 1
                    assert response.json()["run"]["cases"] == {
                        "total": 1,
                        "passed": 0,
                        "failed": 0,
                        "flaky": 0,
                        "error": 1,
                        "unavailable": 0,
                    }
                    assert "claim_id" not in response.text
                await store.finish_cancel(lease.claim)
                response = await client.get(f"/api/evals/run-views/{run_id}", headers=_AUTH_HEADERS)
                assert response.status_code == 200, response.text
                assert response.json()["run"]["status"] == "cancelled"
                assert response.json()["progress"]["completed_trials"] is None
                assert response.json()["progress"]["state"] == "unavailable"
                assert response.json()["comparison"]["state"] == "unavailable"
                assert response.json()["comparison"]["verdict"] is None
                assert response.json()["comparison"]["unavailable_reasons"] == [
                    "current_run_has_no_result"
                ]
        finally:
            await store.close()

    asyncio.run(exercise())


def test_unavailable_views_are_registered_and_capability_is_explicit():
    from fastapi.routing import APIRoute
    from fastapi.testclient import TestClient
    from tests.server.test_server_evals import _authenticate

    from cayu.applications import CayuApp
    from cayu.server import ServerConfig, create_server
    from cayu.server.proxy_identity import _READ_AND_DECIDE_ROUTES

    server = create_server(CayuApp(), config=ServerConfig.protected(_authenticate))
    routes = {
        (method, route.path.removeprefix("/api"))
        for route in server.routes
        if isinstance(route, APIRoute)
        for method in route.methods
    }
    assert _READ_AND_DECIDE_ROUTES - routes == set()
    client = TestClient(server)
    response = client.get("/api/evals/suite-views?target_key=unavailable", headers=_AUTH_HEADERS)
    assert response.status_code == 409
    capability = client.get("/api/contract", headers=_AUTH_HEADERS)
    surface = capability.json()["capabilities"]["surfaces"]["eval_suite_views"]
    assert surface["configured"] is False
    assert surface["read"]["enabled"] is False


def test_authored_multi_part_launch_history_retains_exact_execution_groups(sqlite_resources):
    from tests.server.test_server_eval_suite_launch import _reviewed_body, _saved_suite
    from tests.server.test_server_eval_suite_launch import _server as suite_server

    async def exercise():
        async with sqlite_resources as resources:
            target, store, authored, _provider = await _saved_suite(resources, scenario_count=2)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=suite_server(target, store)),
                base_url="http://test",
            ) as client:
                root = f"/api/evals/suites/{authored.revision}/runs"
                body = await _reviewed_body(client, root)
                launched = await client.post(
                    root,
                    headers={**_AUTH_HEADERS, "Idempotency-Key": "read-view-launch"},
                    json=body,
                )
                assert launched.status_code == 202, launched.text
                catalog = await client.get(
                    "/api/evals/suite-views",
                    params={"target_key": target.key},
                    headers=_AUTH_HEADERS,
                )
                assert catalog.status_code == 200, catalog.text
                summary = next(
                    item for item in catalog.json()["items"] if item["source"] == "authored"
                )
                response = await client.get(
                    f"/api/evals/suite-views/{summary['id']}",
                    params={"runs_limit": 1},
                    headers=_AUTH_HEADERS,
                )
                assert response.status_code == 200, response.text
                detail = response.json()
                assert detail["cases"]["total"] == 3
                seen = []
                while True:
                    seen.extend(detail["runs"]["items"])
                    if not detail["runs"]["has_more"]:
                        break
                    response = await client.get(
                        f"/api/evals/suite-views/{summary['id']}",
                        params={"runs_limit": 1, "runs_cursor": detail["runs"]["next_cursor"]},
                        headers=_AUTH_HEADERS,
                    )
                    assert response.status_code == 200, response.text
                    detail = response.json()
                assert len(seen) == 3
                assert len({run["suite"]["revision"] for run in seen}) == 3
                assert {run["authored_suite_revision"] for run in seen} == {authored.revision}
                assert summary["baseline_scope"] == "execution_group"
                assert summary["baseline"] is None
                assert all(
                    run["cases"]["passed"] == run["cases"]["failed"] == run["cases"]["error"] == 0
                    and run["baseline"] is None
                    for run in seen
                )
                from cayu.evals.execution import run_corpus_suite
                from cayu.evals.store import EvalBaselineKey, EvalBaselineUpdate

                leases = [await store.claim_run(lease_seconds=300) for _ in range(3)]
                simple = next(
                    lease for lease in leases if lease.run.spec.invocation.scenario is None
                )
                corpus = await store.load_corpus(simple.run.spec.corpus_revision)
                result = await run_corpus_suite(
                    target, corpus, simple.run.spec.suite_id, max_concurrency=1
                )
                await store.publish_result(simple.claim, result, redact_json=target.app.redact_json)
                await store.set_baseline(
                    EvalBaselineUpdate(
                        key=EvalBaselineKey(
                            target_key=target.key,
                            corpus_revision=corpus.revision,
                            suite_id=simple.run.spec.suite_id,
                        ),
                        result_revision=result.revision,
                        expected_generation=0,
                        operation_id="sha256:" + "8" * 64,
                        actor_id="group-reviewer",
                    ),
                    redact_json=target.app.redact_json,
                )
                grouped = await client.get(
                    f"/api/evals/suite-views/{summary['id']}", headers=_AUTH_HEADERS
                )
                assert grouped.status_code == 200, grouped.text
                body = grouped.json()
                assert body["suite"]["baseline"] is None
                assert body["suite"]["baseline_scope"] == "execution_group"
                for run in body["runs"]["items"]:
                    if run["suite"]["revision"] == corpus.revision:
                        assert run["baseline"]["result_revision"] == result.revision
                        assert run["baseline"]["key"]["corpus_revision"] == corpus.revision
                        assert run["baseline"]["updated_by"] == "group-reviewer"
                        assert run["baseline_result"]["result_revision"] == result.revision
                        assert run["baseline_result"]["cases"]["total"] == 1
                    else:
                        assert run["baseline"] is None

    asyncio.run(exercise())


def test_postgres_suite_views_public_http(postgres_dsn):
    from cayu.storage.evals_postgres import PostgresEvalStore
    from cayu.storage.migrations import SchemaMode

    async def exercise():
        target = _target(_provider())
        store = PostgresEvalStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            await _admit_run(store, target, "postgres-view")
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=_server(target, store)), base_url="http://test"
            ) as client:
                response = await client.get(
                    "/api/evals/suite-views",
                    params={"target_key": target.key},
                    headers=_AUTH_HEADERS,
                )
                assert response.status_code == 200, response.text
                suite = response.json()["items"][0]
                resolved = await client.get(
                    "/api/evals/suite-views/resolve-corpus",
                    params={
                        "corpus": suite["identity"]["revision"],
                        "suite": suite["identity"]["suite_id"],
                    },
                    headers=_AUTH_HEADERS,
                )
                assert resolved.status_code == 200, resolved.text
                assert resolved.json() == suite["identity"]
                response = await client.get(
                    f"/api/evals/suite-views/{suite['id']}", headers=_AUTH_HEADERS
                )
                assert response.status_code == 200, response.text
                response = await client.get(
                    f"/api/evals/run-views/{suite['last_run']['id']}", headers=_AUTH_HEADERS
                )
                assert response.status_code == 200, response.text
                assert response.json()["progress"]["total_trials"] == 1
                assert response.json()["progress"]["completed_trials"] == 0
                from tests.evals.test_corpus_execution import _corpus

                from cayu.evals.execution import run_corpus_suite
                from cayu.evals.store import EvalBaselineKey, EvalBaselineUpdate

                corpus = _corpus(trials=1)
                published = await run_corpus_suite(
                    target, corpus, corpus.suites[0].id, max_concurrency=1
                )
                lease = await store.claim_run(lease_seconds=300)
                await store.publish_result(
                    lease.claim, published, redact_json=target.app.redact_json
                )
                await store.set_baseline(
                    EvalBaselineUpdate(
                        key=EvalBaselineKey(
                            target_key=target.key,
                            corpus_revision=corpus.revision,
                            suite_id=corpus.suites[0].id,
                        ),
                        result_revision=published.revision,
                        expected_generation=0,
                        operation_id="sha256:" + "7" * 64,
                        actor_id="postgres-reviewer",
                    ),
                    redact_json=target.app.redact_json,
                )
                detail = await client.get(
                    f"/api/evals/suite-views/{suite['id']}", headers=_AUTH_HEADERS
                )
                assert detail.status_code == 200, detail.text
                assert detail.json()["cases"]["items"][0]["last"]["status"] == "passed"
                assert detail.json()["runs"]["items"][0]["comparison"]["verdict"] == "tie"
                assert detail.json()["suite"]["baseline_result"]["native_run_id"] == "postgres-view"
                assert (
                    detail.json()["runs"]["items"][0]["baseline_result"]["native_run_id"]
                    == "postgres-view"
                )
        finally:
            await store.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "field,value",
    [
        ("revision", "sha256:" + "f" * 64),
        ("corpus_revision", "sha256:" + "f" * 64),
        ("suite_id", "unrelated"),
        ("status", "failed"),
        ("score", 0.0),
        ("score", True),
        ("document_bytes", 1),
    ],
)
def test_run_views_fail_closed_on_catalog_document_mismatch(tmp_path, monkeypatch, field, value):
    from tests.evals.test_corpus_execution import _corpus

    from cayu.evals.execution import run_corpus_suite

    async def exercise():
        target = _target(_provider(trials=1))
        store = SQLiteEvalStore(tmp_path / "mismatch.db")
        try:
            await _admit_run(store, target, "exact-run")
            corpus = _corpus(trials=1)
            result = await run_corpus_suite(target, corpus, corpus.suites[0].id, max_concurrency=1)
            lease = await store.claim_run(lease_seconds=300)
            await store.publish_result(lease.claim, result, redact_json=target.app.redact_json)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=_server(target, store)), base_url="http://test"
            ) as client:
                response = await client.get(
                    "/api/evals/suite-views",
                    params={"target_key": target.key},
                    headers=_AUTH_HEADERS,
                )
                assert response.status_code == 200, response.text
                selector = response.json()["items"][0]["last_run"]["id"]
                load = store.load_result_record

                async def wrong_record(revision):
                    record = await load(revision)
                    return record.model_copy(update={field: value})

                monkeypatch.setattr(store, "load_result_record", wrong_record)
                rejected = await client.get(
                    f"/api/evals/run-views/{selector}", headers=_AUTH_HEADERS
                )
                assert rejected.status_code == 409, rejected.text
                assert rejected.json() == {
                    "detail": "Eval result catalog metadata does not match its stored document."
                }
        finally:
            await store.close()

    asyncio.run(exercise())


def test_named_incompatible_run_returns_reasons_not_a_tie(tmp_path):
    from tests.evals.test_corpus_execution import _corpus
    from tests.evals.test_eval_store import _request

    from cayu.evals.execution import run_corpus_suite

    async def exercise():
        target = _target(_provider(trials=2))
        store = SQLiteEvalStore(tmp_path / "incompatible.db")
        try:
            for index, text in enumerate(("Refund order 42.", "Refund order 43.")):
                corpus = _corpus(trials=1, input_text=text)
                await store.save_corpus(corpus, redact_json=target.app.redact_json)
                await store.admit_run(
                    _request(corpus, run_id=f"run-{index}", key=f"request-{index}"),
                    redact_json=target.app.redact_json,
                )
                lease = await store.claim_run(lease_seconds=300)
                result = await run_corpus_suite(
                    target, corpus, corpus.suites[0].id, max_concurrency=1
                )
                await store.publish_result(lease.claim, result, redact_json=target.app.redact_json)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=_server(target, store)), base_url="http://test"
            ) as client:
                response = await client.get(
                    "/api/evals/suite-views",
                    params={"target_key": target.key},
                    headers=_AUTH_HEADERS,
                )
                assert response.status_code == 200, response.text
                runs = [suite["last_run"]["id"] for suite in response.json()["items"]]
                response = await client.get(
                    f"/api/evals/run-views/{runs[0]}",
                    params={"compared_with": runs[1]},
                    headers=_AUTH_HEADERS,
                )
                assert response.status_code == 200, response.text
                comparison = response.json()["comparison"]
                assert comparison["state"] == "incompatible" and comparison["verdict"] is None
                assert "corpus_revision_mismatch" in comparison["compatibility"]["reasons"]
                assert (
                    comparison["improved_cases"] is None and comparison["regressed_cases"] is None
                )
                assert all(
                    case["change"] == "uncompared" for case in response.json()["results"]["items"]
                )
        finally:
            await store.close()

    asyncio.run(exercise())


def test_live_outcomes_wait_for_trial_policy_and_survive_store_restart(tmp_path):
    from tests.evals.eval_store_conformance import _terminal_trial_checkpoint
    from tests.evals.test_corpus_execution import _corpus
    from tests.evals.test_eval_store import _request

    from cayu.evals.read_views import run_view_id

    async def exercise():
        target = _target(_provider())
        path = tmp_path / "live-policy.db"
        store = SQLiteEvalStore(path)
        corpus = _corpus(target_key=target.key, trials=2)
        await store.save_corpus(corpus, redact_json=target.app.redact_json)
        await store.admit_run(
            _request(corpus, run_id="live-policy", key="live-policy"),
            redact_json=target.app.redact_json,
        )
        lease = await store.claim_run(lease_seconds=300)
        checkpoint = _terminal_trial_checkpoint(corpus)
        await store.save_trial_checkpoint(
            lease.claim, checkpoint, redact_json=target.app.redact_json
        )

        async def read(current):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=_server(target, current)), base_url="http://test"
            ) as client:
                response = await client.get(
                    f"/api/evals/run-views/{run_view_id('run', 'live-policy')}",
                    headers=_AUTH_HEADERS,
                )
                assert response.status_code == 200, response.text
                assert "candidate execution failed" not in response.text
                return response.json()

        try:
            partial = await read(store)
            assert partial["progress"]["completed_trials"] == 1
            assert partial["progress"]["completed_cases"] == 0
            assert partial["run"]["cases"] == {
                "total": 1,
                "passed": 0,
                "failed": 0,
                "flaky": 0,
                "error": 0,
                "unavailable": 0,
            }
            second = checkpoint.model_copy(
                update={"result": checkpoint.result.model_copy(update={"trial_number": 2})}
            )
            await store.save_trial_checkpoint(
                lease.claim, second, redact_json=target.app.redact_json
            )
            settled = await read(store)
            assert settled["progress"]["completed_cases"] == 1
            assert settled["run"]["cases"]["error"] == 1
            assert settled["run"]["cases"]["passed"] == 0
            await store.close()
            store = SQLiteEvalStore(path)
            assert await read(store) == settled
        finally:
            await store.close()

    asyncio.run(exercise())


def test_run_read_rejects_mutated_result_without_diagnostic_leaks(
    tmp_path, monkeypatch, caplog, capsys
):
    import warnings

    from tests.evals.test_corpus_execution import _corpus

    from cayu.evals.execution import run_corpus_suite
    from cayu.evals.read_views import run_view_id

    canary = "eval-view-secret-canary"

    class Secret:
        def __repr__(self):
            return canary

        def __str__(self):
            return canary

    async def exercise():
        target = _target(_provider(trials=1))
        store = SQLiteEvalStore(tmp_path / "forged-result.db")
        try:
            await _admit_run(store, target, "forged-result")
            corpus = _corpus(trials=1)
            result = await run_corpus_suite(target, corpus, corpus.suites[0].id, max_concurrency=1)
            lease = await store.claim_run(lease_seconds=300)
            await store.publish_result(lease.claim, result, redact_json=target.app.redact_json)
            load = store.load_result_by_revision

            async def forged(revision):
                value = await load(revision)
                object.__setattr__(value.run, "score", Secret())
                return value

            monkeypatch.setattr(store, "load_result_by_revision", forged)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=_server(target, store)), base_url="http://test"
            ) as client:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    response = await client.get(
                        f"/api/evals/run-views/{run_view_id('run', 'forged-result')}",
                        headers=_AUTH_HEADERS,
                    )
                assert response.status_code == 409, response.text
                assert canary not in response.text
                assert canary not in str([item.message for item in caught])
        finally:
            await store.close()

    asyncio.run(exercise())
    captured = capsys.readouterr()
    assert canary not in captured.out + captured.err + caplog.text


def test_cancelled_http_read_preserves_cancellation_and_durable_run(tmp_path, monkeypatch):
    from cayu.evals.read_views import run_view_id

    async def exercise():
        target = _target(_provider())
        store = SQLiteEvalStore(tmp_path / "cancel-read.db")
        try:
            await _admit_run(store, target, "cancel-read")
            load = store.load_run_view_snapshot
            entered = asyncio.Event()
            never = asyncio.Event()

            async def blocked(run_id):
                snapshot = await load(run_id)
                entered.set()
                await never.wait()
                return snapshot

            monkeypatch.setattr(store, "load_run_view_snapshot", blocked)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=_server(target, store)), base_url="http://test"
            ) as client:
                task = asyncio.create_task(
                    client.get(
                        f"/api/evals/run-views/{run_view_id('run', 'cancel-read')}",
                        headers=_AUTH_HEADERS,
                    )
                )
                await asyncio.wait_for(entered.wait(), timeout=10)
                task.cancel()
                assert task.cancelling() == 1
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert task.cancelled()
                monkeypatch.setattr(store, "load_run_view_snapshot", load)
                response = await client.get(
                    f"/api/evals/run-views/{run_view_id('run', 'cancel-read')}",
                    headers=_AUTH_HEADERS,
                )
                assert response.status_code == 200, response.text
                assert response.json()["run"]["status"] == "queued"
                assert (await store.load_run("cancel-read")).status == "queued"
        finally:
            await store.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("minimum_passed,expected", [(1, "passed"), (2, "failed")])
def test_flaky_variability_overlaps_policy_outcomes(tmp_path, minimum_passed, expected):
    from tests.evals.test_corpus_execution import _corpus
    from tests.evals.test_eval_store import _request

    from cayu.evals.corpus import EvalCorpusDocument, EvalSuiteSpec, TrialRequestSpec
    from cayu.evals.execution import run_corpus_suite
    from cayu.evals.testing import ScriptedModelProvider
    from cayu.evals.trial_policy import EvalSuiteTrialPolicyV1
    from cayu.providers.base import ModelStreamEvent

    async def exercise():
        provider = ScriptedModelProvider(
            [
                (
                    ModelStreamEvent.text_delta(output),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                )
                for output in ("Approved", "Denied")
            ]
        )
        target = _target(provider)
        original = _corpus(trials=2)
        suite = EvalSuiteSpec.create(
            id=original.suites[0].id,
            name=original.suites[0].name,
            trial_request=TrialRequestSpec(
                trials=2,
                timeout_seconds=30,
                trial_policy=EvalSuiteTrialPolicyV1.create(
                    trial_count=2, minimum_passed_trials=minimum_passed, max_concurrency=1
                ),
            ),
        )
        corpus = EvalCorpusDocument.create(
            target_key=target.key,
            evidence_policy=original.evidence_policy,
            pricing_profile=original.pricing_profile,
            suites=(suite,),
            cases=original.cases,
        )
        store = SQLiteEvalStore(tmp_path / "variability.db")
        try:
            await store.save_corpus(corpus, redact_json=target.app.redact_json)
            await store.admit_run(
                _request(corpus, run_id="variable", key="variable"),
                redact_json=target.app.redact_json,
            )
            lease = await store.claim_run(lease_seconds=300)
            result = await run_corpus_suite(target, corpus, suite.id, max_concurrency=1)
            await store.publish_result(lease.claim, result, redact_json=target.app.redact_json)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=_server(target, store)), base_url="http://test"
            ) as client:
                catalog = await client.get("/api/evals/suite-views", headers=_AUTH_HEADERS)
                assert catalog.status_code == 200, catalog.text
                run = catalog.json()["items"][0]["last_run"]
                detail = await client.get(
                    f"/api/evals/run-views/{run['id']}", headers=_AUTH_HEADERS
                )
                assert detail.status_code == 200, detail.text
                counts = detail.json()["run"]["cases"]
                assert counts[expected] == counts["flaky"] == counts["total"] == 1
                assert (
                    counts["passed"] + counts["failed"] + counts["error"] + counts["unavailable"]
                    == 1
                )
                assert (
                    detail.json()["results"]["items"][0]["presentation"]["reliability"][
                        "passed_trials"
                    ]
                    == 1
                )
        finally:
            await store.close()

    asyncio.run(exercise())


async def _exercise_binary_view_ordering(store, *, configure_locale=None):
    from tests.evals.test_corpus_execution import _corpus
    from tests.evals.test_eval_store import _request

    from cayu.evals.corpus import EvalCaseSpec, EvalCorpusDocument, EvalSuiteSpec
    from cayu.evals.execution import run_corpus_suite
    from cayu.evals.read_views import EvalCaseHistoryQuery, EvalSuiteViewIdentity, EvalViewQuery

    target = _target(_provider(), key="ordering-agent")
    original = _corpus(target_key=target.key, trials=1)
    suites = tuple(
        EvalSuiteSpec.create(id=value, name=value, trial_request=original.suites[0].trial_request)
        for value in ("a-b", "aa")
    )
    template = original.cases[0]
    cases = tuple(
        EvalCaseSpec.create(
            id=f"case-{suite.id}",
            suite_id=suite.id,
            name=suite.name,
            source=template.source,
            input=template.input,
            assertions=template.assertions,
        )
        for suite in suites
    )
    corpus = EvalCorpusDocument.create(
        target_key=target.key, evidence_policy=original.evidence_policy, suites=suites, cases=cases
    )
    await store.save_corpus(corpus, redact_json=target.app.redact_json)
    if configure_locale is not None:
        configure_locale()
    published = await run_corpus_suite(target, corpus, suites[0].id, max_concurrency=1)
    for run_id in ("aa", "a-b", "zed", "Zed"):
        await store.admit_run(
            _request(corpus, run_id=run_id, key=f"ordering-{run_id}"),
            redact_json=target.app.redact_json,
        )
        lease = await store.claim_run(lease_seconds=300)
        assert lease is not None
        await store.publish_result(lease.claim, published, redact_json=target.app.redact_json)
    scope = EvalSuiteViewIdentity(
        source="corpus", target_key=target.key, revision=corpus.revision, suite_id=suites[0].id
    )
    history = await store.list_view_index(EvalViewQuery(target_key=target.key, suite=scope))
    assert [item.revision for item in history.items] == ["Zed", "a-b", "aa", "zed"]
    assert len({item.created_at for item in history.items}) == 1
    latest = await store.load_latest_case_results(
        EvalCaseHistoryQuery(suite=scope, case_ids=(cases[0].id,))
    )
    assert [(item.source, item.revision) for item in latest] == [("run", "Zed")]
    if not store.durable:
        return  # Server configuration intentionally requires a durable eval store.
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_server(target, store)), base_url="http://test"
    ) as client:
        catalog = []
        params = {"target_key": target.key, "limit": 1}
        while True:
            response = await client.get(
                "/api/evals/suite-views", params=params, headers=_AUTH_HEADERS
            )
            assert response.status_code == 200, response.text
            page = response.json()
            catalog.extend(page["items"])
            if not page["has_more"]:
                break
            assert len(catalog) <= 2
            params["cursor"] = page["next_cursor"]
        assert [item["identity"]["suite_id"] for item in catalog] == ["a-b", "aa"]
        response = await client.get(
            "/api/evals/suite-views", params={"target_key": target.key}, headers=_AUTH_HEADERS
        )
        assert response.status_code == 200, response.text
        assert [item["identity"]["suite_id"] for item in response.json()["items"]] == ["a-b", "aa"]
        route = f"/api/evals/suite-views/{catalog[0]['id']}"
        observed = []
        params = {"runs_limit": 1}
        while True:
            response = await client.get(route, params=params, headers=_AUTH_HEADERS)
            assert response.status_code == 200, response.text
            page = response.json()["runs"]
            observed.extend(item["native_run_id"] for item in page["items"])
            if not page["has_more"]:
                break
            assert len(observed) <= 4
            params["runs_cursor"] = page["next_cursor"]
        assert observed == ["Zed", "a-b", "aa", "zed"]
        response = await client.get(route, headers=_AUTH_HEADERS)
        assert response.status_code == 200, response.text
        assert [item["native_run_id"] for item in response.json()["runs"]["items"]] == observed


def _freeze_eval_store_time(monkeypatch):
    from datetime import UTC, datetime

    import cayu.storage.evals_postgres as postgres_module
    import cayu.storage.evals_sqlite as sqlite_module

    observed_at = datetime(2026, 1, 1, tzinfo=UTC)

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return observed_at

    async def database_now(cursor):
        return observed_at

    monkeypatch.setattr(sqlite_module, "datetime", FrozenDatetime)
    monkeypatch.setattr(postgres_module, "_database_now", database_now)
    return observed_at


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_view_ordering_and_latest_tie_are_identical_across_stores(tmp_path, monkeypatch, backend):
    observed_at = _freeze_eval_store_time(monkeypatch)

    async def exercise():
        store = (
            InMemoryEvalStore(clock=lambda: observed_at)
            if backend == "memory"
            else SQLiteEvalStore(tmp_path / "ordered.db")
        )
        try:
            await _exercise_binary_view_ordering(store)
        finally:
            await store.close()

    asyncio.run(exercise())


def test_postgres_view_http_cursors_and_latest_tie_ignore_non_c_collation(
    postgres_dsn, monkeypatch
):
    import psycopg

    from cayu.storage.evals_postgres import PostgresEvalStore
    from cayu.storage.migrations import SchemaMode

    _freeze_eval_store_time(monkeypatch)

    def configure_locale():
        with psycopg.connect(postgres_dsn) as connection:
            connection.execute(
                "CREATE COLLATION cayu_eval_non_c (provider = icu, locale = 'en-US')"
            )
            rows = connection.execute(
                "SELECT value FROM (VALUES ('aa'), ('a-b'), ('Zed'), ('zed')) AS ids(value) ORDER BY value COLLATE cayu_eval_non_c"
            ).fetchall()
            assert [row[0] for row in rows] != ["Zed", "a-b", "aa", "zed"]
            connection.execute(
                "ALTER TABLE cayu_eval_suites ALTER COLUMN suite_id TYPE TEXT COLLATE cayu_eval_non_c"
            )
            connection.execute(
                "ALTER TABLE cayu_eval_runs ALTER COLUMN run_id TYPE TEXT COLLATE cayu_eval_non_c"
            )

    async def exercise():
        store = PostgresEvalStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            await _exercise_binary_view_ordering(store, configure_locale=configure_locale)
        finally:
            await store.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "check", ["direction", "provenance", "linkage_revision", "linkage_scope", "absent_link"]
)
def test_result_views_preserve_comparison_direction_and_recorded_provenance(
    tmp_path, check, monkeypatch
):
    from tests.evals.test_corpus_execution import _corpus
    from tests.evals.test_eval_store import _request

    from cayu.evals.execution import run_corpus_suite
    from cayu.evals.execution_profiles import (
        EvalExecutionProfilePolicyV1,
        prepare_eval_execution_profile,
    )
    from cayu.evals.read_views import run_view_id
    from cayu.evals.store import EvalBaselineKey, EvalBaselineUpdate, EvalRunInvocation
    from cayu.evals.testing import ScriptedModelProvider
    from cayu.providers.base import ModelStreamEvent
    from cayu.sessions.invocation import InvocationOrigin

    async def exercise():
        target = _target(
            ScriptedModelProvider(
                [
                    (
                        ModelStreamEvent.text_delta(output),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    )
                    for output in ("Denied", "Approved")
                ]
            )
        )
        corpus = _corpus(trials=1)
        path = tmp_path / "direction-provenance.db"
        store = SQLiteEvalStore(path)
        await store.save_corpus(corpus, redact_json=target.app.redact_json)
        records = {}
        for name in ("old", "new"):
            result = await run_corpus_suite(target, corpus, corpus.suites[0].id, max_concurrency=1)
            profile = await prepare_eval_execution_profile(
                target,
                profile_id="candidate",
                label="Candidate",
                source="explicit",
                app_manifest_fingerprint=result.target.app_manifest_fingerprint,
                policy=EvalExecutionProfilePolicyV1(),
            )
            request = _request(corpus, run_id=name, key=name).model_copy(
                update={
                    "invocation": EvalRunInvocation(
                        source="http_run",
                        origin=InvocationOrigin(trust="server_verified", subject="operator"),
                        execution_profile=profile.binding,
                        execution_profile_snapshot=profile.snapshot,
                    ),
                }
            )
            await store.admit_run(request, redact_json=target.app.redact_json)
            lease = await store.claim_run(lease_seconds=300)
            records[name] = await store.publish_result(
                lease.claim, result, redact_json=target.app.redact_json
            )
        await store.set_baseline(
            EvalBaselineUpdate(
                key=EvalBaselineKey(
                    target_key=target.key,
                    corpus_revision=corpus.revision,
                    suite_id=corpus.suites[0].id,
                ),
                result_revision=records["old"].result.revision,
                expected_generation=0,
                operation_id="sha256:" + "b" * 64,
                actor_id="reviewer",
            ),
            redact_json=target.app.redact_json,
        )
        # Reconstruct native provenance from the durable catalog after restart.
        await store.close()
        store = SQLiteEvalStore(path)
        if check in {"linkage_revision", "linkage_scope", "absent_link"}:
            snapshot = await store.load_run_view_snapshot(
                "old" if check != "linkage_revision" else "new"
            )
            if check == "linkage_scope":
                snapshot = snapshot.model_copy(
                    update={
                        "run": snapshot.run.model_copy(
                            update={
                                "spec": snapshot.run.spec.model_copy(
                                    update={"suite_revision": "sha256:" + "f" * 64}
                                ),
                            }
                        )
                    }
                )

            async def wrong_link(revision):
                return None if check == "absent_link" else snapshot

            monkeypatch.setattr(store, "load_result_run_view_snapshot", wrong_link, raising=False)
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=_server(target, store)), base_url="http://test"
            ) as client:
                response = await client.get(
                    f"/api/evals/run-views/{run_view_id('run', 'new')}", headers=_AUTH_HEADERS
                )
                if check in {"linkage_revision", "linkage_scope"}:
                    assert response.status_code == 409, response.text
                    return
                assert response.status_code == 200, response.text
                body = response.json()
                if check == "absent_link":
                    assert body["compared_run"]["native_run_id"] is None
                    assert body["compared_run"]["candidate"] is None
                    assert body["compared_run"]["actor"] is None
                    return
                assert body["comparison"]["verdict"] == "improvement"
                if check == "direction":
                    run_changes = body["comparison"]["improvements"]
                    case_changes = body["results"]["items"][0]["improvements"]
                    assert {item["kind"] for item in run_changes} == {"status", "score"}
                    assert {item["kind"] for item in case_changes} == {
                        "status",
                        "score",
                        "reliability",
                    }
                    for item in run_changes + case_changes:
                        if item["kind"] == "status":
                            assert (item["baseline_status"], item["current_status"]) == (
                                "failed",
                                "passed",
                            )
                        elif item["kind"] == "score":
                            assert item["baseline_score"] < item["current_score"]
                        else:
                            assert item["baseline_reliability"]["passed_trials"] == 0
                            assert item["current_reliability"]["passed_trials"] == 1
                else:
                    for metadata in (body["compared_run"], body["run"]["baseline_result"]):
                        assert metadata["native_run_id"] == "old"
                        assert metadata["actor"] == "operator"
                        assert metadata["origin"] == "http_run"
                        assert metadata["candidate"] == profile.snapshot.candidate.model_dump(
                            mode="json"
                        )
                        assert metadata["candidate_unavailable_reason"] is None
                        assert (
                            metadata["started_at"] is not None
                            and metadata["finished_at"] is not None
                        )
                        assert metadata["id"] == run_view_id(
                            "result", records["old"].result.revision
                        )
        finally:
            await store.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "states,expected_state,expected_tokens,expected_observed",
    [
        (("complete",), "complete", "6", "6"),
        (("zero",), "complete", "0", "0"),
        (("unavailable",), "unavailable", None, None),
        (("partial",), "partial", None, "3"),
        (("complete", "unavailable"), "partial", None, "6"),
        (("partial", "unavailable"), "partial", None, "3"),
        (("missing",), "unavailable", None, None),
        (("complete", "missing"), "partial", None, "6"),
    ],
)
def test_run_usage_requires_positive_token_evidence(
    tmp_path, states, expected_state, expected_tokens, expected_observed
):
    from tests.evals.test_corpus_execution import _corpus, _SearchEvidenceTool
    from tests.evals.test_eval_store import _request

    from cayu import AgentSpec, RunRequest
    from cayu.evals.execution import CorpusExecutionResult, run_corpus_suite
    from cayu.evals.read_views import run_view_id
    from cayu.evals.store import EvalBaselineKey, EvalBaselineUpdate
    from cayu.evals.testing import ScriptedModelProvider
    from cayu.providers import ModelStreamEvent

    async def exercise():
        script = []
        for index, state in enumerate(states):
            for step in range(2):
                reports_usage = state != "unavailable" and not (state == "partial" and step == 1)
                total = 0 if state == "zero" else 3
                metadata = {"finish_reason": "tool_calls" if step == 0 else "stop"}
                if reports_usage:
                    metadata["usage"] = {
                        "input_tokens": total,
                        "output_tokens": 0,
                        "total_tokens": total,
                    }
                script.append(
                    (
                        ModelStreamEvent.tool_call(
                            id=f"call-{index}", name="search", arguments={"query": "test"}
                        )
                        if step == 0
                        else ModelStreamEvent.text_delta("Approved"),
                        ModelStreamEvent.completed(metadata),
                    )
                )
        target = _target(ScriptedModelProvider(script))
        target.app.register_agent(
            AgentSpec(name="with-tool", model="fixture-model"), tools=[_SearchEvidenceTool(count=1)]
        )
        target = target.model_copy(
            update={"request_base": RunRequest(agent_name="with-tool", messages=[], max_steps=2)}
        )
        corpus = _corpus(trials=len(states))
        result = await run_corpus_suite(target, corpus, corpus.suites[0].id, max_concurrency=1)
        trials = result.run.cases[0].trials
        assert [trial.usage_evidence_state for trial in trials] == [
            "complete" if state == "missing" else "unavailable" if state == "zero" else state
            for state in states
        ]
        assert all(trial.usage is not None and trial.evidence_complete for trial in trials)
        if "missing" in states or "zero" in states:
            import json

            from cayu.evals.corpus import _content_revision

            # Cover valid stored optional-field absence and explicitly measured
            # zero independently of the runtime's nonzero-usage classification.
            published = result.run.model_dump(mode="json")
            for trial, state in zip(published["cases"][0]["trials"], states, strict=True):
                if state == "missing":
                    trial.pop("usage_evidence_state")
                elif state == "zero":
                    trial["usage_evidence_state"] = "complete"
            published["revision"] = _content_revision(published, "published eval run")
            result = CorpusExecutionResult.create(
                target=result.target,
                run=type(result.run).model_validate_json(json.dumps(published)),
            )
        store = SQLiteEvalStore(tmp_path / "usage.db")
        try:
            await store.save_corpus(corpus, redact_json=target.app.redact_json)
            await store.admit_run(
                _request(corpus, run_id="usage", key="usage"), redact_json=target.app.redact_json
            )
            lease = await store.claim_run(lease_seconds=300)
            await store.publish_result(lease.claim, result, redact_json=target.app.redact_json)
            await store.set_baseline(
                EvalBaselineUpdate(
                    key=EvalBaselineKey(
                        target_key=target.key,
                        corpus_revision=corpus.revision,
                        suite_id=corpus.suites[0].id,
                    ),
                    result_revision=result.revision,
                    expected_generation=0,
                    operation_id="sha256:" + "9" * 64,
                    actor_id="reviewer",
                ),
                redact_json=target.app.redact_json,
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=_server(target, store)), base_url="http://test"
            ) as client:
                response = await client.get(
                    f"/api/evals/run-views/{run_view_id('run', 'usage')}", headers=_AUTH_HEADERS
                )
                assert response.status_code == 200, response.text
                body = response.json()
                for metadata in (body["run"], body["run"]["baseline_result"], body["compared_run"]):
                    assert metadata["usage"]["state"] == expected_state
                    assert metadata["usage"]["tokens"] == expected_tokens
                    assert metadata["usage"]["observed_tokens"] == expected_observed
        finally:
            await store.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "minimum_passed,check",
    [
        (1, "selection"),
        (2, "selection"),
        (1, "cursor_scope"),
        (1, "baseline_generation"),
        (1, "resolved_comparison"),
        (1, "invalid"),
    ],
)
def test_run_case_filters_cover_full_results_and_bind_cursor_scope(tmp_path, minimum_passed, check):
    from tests.evals.test_corpus_execution import _corpus
    from tests.evals.test_eval_store import _request

    from cayu.evals.corpus import EvalCaseSpec, EvalCorpusDocument, EvalSuiteSpec, TrialRequestSpec
    from cayu.evals.execution import run_corpus_suite
    from cayu.evals.read_views import run_view_id
    from cayu.evals.store import EvalBaselineKey, EvalBaselineUpdate
    from cayu.evals.testing import ScriptedModelProvider
    from cayu.evals.trial_policy import EvalSuiteTrialPolicyV1
    from cayu.providers import ModelStreamEvent

    async def exercise():
        old = (
            "Approved",
            "Approved",
            "Denied",
            "Denied",
            "Approved",
            "Approved",
            "Approved",
            "Approved",
        )
        new = (
            "Approved",
            "Approved",
            "Approved",
            "Approved",
            "Approved",
            "Denied",
            "Denied",
            "Denied",
        )
        provider = ScriptedModelProvider(
            [
                (
                    ModelStreamEvent.text_delta(output),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                )
                for output in old + new
            ]
        )
        target = _target(provider)
        source = _corpus(trials=2)
        suite = EvalSuiteSpec.create(
            id=source.suites[0].id,
            name=source.suites[0].name,
            trial_request=TrialRequestSpec(
                trials=2,
                timeout_seconds=30,
                trial_policy=EvalSuiteTrialPolicyV1.create(
                    trial_count=2, minimum_passed_trials=minimum_passed, max_concurrency=1
                ),
            ),
        )
        template = source.cases[0]
        corpus = EvalCorpusDocument.create(
            target_key=target.key,
            evidence_policy=source.evidence_policy,
            pricing_profile=source.pricing_profile,
            suites=(suite,),
            cases=tuple(
                EvalCaseSpec.create(
                    id=case_id,
                    name=case_id,
                    suite_id=suite.id,
                    input=template.input,
                    assertions=template.assertions,
                    source=template.source,
                )
                for case_id in ("a-clean", "b-improved", "c-flaky", "z-regressed")
            ),
        )
        store = SQLiteEvalStore(tmp_path / "filters.db")
        try:
            await store.save_corpus(corpus, redact_json=target.app.redact_json)
            published = {}
            for run_id in ("before", "after"):
                await store.admit_run(
                    _request(corpus, run_id=run_id, key=run_id), redact_json=target.app.redact_json
                )
                lease = await store.claim_run(lease_seconds=300)
                result = await run_corpus_suite(target, corpus, suite.id, max_concurrency=1)
                await store.publish_result(lease.claim, result, redact_json=target.app.redact_json)
                published[run_id] = result
            baseline_key = EvalBaselineKey(
                target_key=target.key, corpus_revision=corpus.revision, suite_id=suite.id
            )
            await store.set_baseline(
                EvalBaselineUpdate(
                    key=baseline_key,
                    result_revision=published["before"].revision,
                    expected_generation=0,
                    operation_id="sha256:" + "1" * 64,
                    actor_id="reviewer",
                ),
                redact_json=target.app.redact_json,
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=_server(target, store)), base_url="http://test"
            ) as client:
                url = f"/api/evals/run-views/{run_view_id('run', 'after')}"
                response = await client.get(url, params={"cases_limit": 1}, headers=_AUTH_HEADERS)
                assert response.status_code == 200, response.text
                original = response.json()
                assert original["results"]["items"][0]["case"]["id"] == "a-clean"
                assert original["results"]["total"] == original["run"]["cases"]["total"] == 4
                if check == "invalid":
                    response = await client.get(
                        url, params={"case_filter": "unknown"}, headers=_AUTH_HEADERS
                    )
                    assert response.status_code == 422
                elif check == "selection":
                    expected = {
                        "regressed": ["c-flaky", "z-regressed"],
                        "improved": ["b-improved"],
                        "failing": ["z-regressed"]
                        if minimum_passed == 1
                        else ["c-flaky", "z-regressed"],
                        "flaky": ["c-flaky"],
                    }
                    for selected, expected_ids in expected.items():
                        params = {"case_filter": selected, "cases_limit": 1}
                        observed = []
                        while True:
                            response = await client.get(url, params=params, headers=_AUTH_HEADERS)
                            assert response.status_code == 200, response.text
                            body = response.json()
                            assert body["run"]["cases"] == original["run"]["cases"]
                            assert body["comparison"] == original["comparison"]
                            assert body["results"]["total"] == len(expected_ids)
                            observed += [item["case"]["id"] for item in body["results"]["items"]]
                            if not body["results"]["has_more"]:
                                break
                            params["cases_cursor"] = body["results"]["next_cursor"]
                        assert observed == expected_ids
                    empty = await client.get(
                        url,
                        params={
                            "case_filter": "regressed",
                            "compared_with": run_view_id("run", "after"),
                        },
                        headers=_AUTH_HEADERS,
                    )
                    assert empty.status_code == 200
                    assert empty.json()["results"] == {
                        "items": [],
                        "total": 0,
                        "has_more": False,
                        "next_cursor": None,
                    }
                elif check in {"cursor_scope", "baseline_generation"}:
                    if check == "cursor_scope":
                        # Identical ordering cannot authorize a different filter.
                        rejected = await client.get(
                            url,
                            params={
                                "case_filter": "regressed",
                                "cases_cursor": original["results"]["next_cursor"],
                            },
                            headers=_AUTH_HEADERS,
                        )
                        assert rejected.status_code == 422
                        return
                    # Even re-approval of identical result content changes the
                    # baseline generation and invalidates prior response scope.
                    await store.set_baseline(
                        EvalBaselineUpdate(
                            key=baseline_key,
                            result_revision=published["before"].revision,
                            expected_generation=1,
                            operation_id="sha256:" + "2" * 64,
                            actor_id="second-reviewer",
                        ),
                        redact_json=target.app.redact_json,
                    )
                    rejected = await client.get(
                        url,
                        params={"cases_cursor": original["results"]["next_cursor"]},
                        headers=_AUTH_HEADERS,
                    )
                    assert rejected.status_code == 422
                else:
                    await store.admit_run(
                        _request(corpus, run_id="pending", key="pending"),
                        redact_json=target.app.redact_json,
                    )
                    params = {"compared_with": run_view_id("run", "pending"), "cases_limit": 1}
                    pending = await client.get(url, params=params, headers=_AUTH_HEADERS)
                    assert pending.status_code == 200, pending.text
                    assert pending.json()["comparison"]["state"] == "pending"
                    lease = await store.claim_run(lease_seconds=300)
                    await store.publish_result(
                        lease.claim, published["before"], redact_json=target.app.redact_json
                    )
                    params["cases_cursor"] = pending.json()["results"]["next_cursor"]
                    rejected = await client.get(url, params=params, headers=_AUTH_HEADERS)
                    assert rejected.status_code == 422
        finally:
            await store.close()

    asyncio.run(exercise())
