from __future__ import annotations

import asyncio
import hashlib

import httpx
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi import HTTPException, Request
from fastapi.routing import APIRoute
from tests.server.test_server_evals import (
    _AUTH_HEADERS,
    _authenticate,
    _bound_eval_invocation,
    _corpus,
    _evals_config,
    _provider,
    _server,
    _target,
)

from cayu.evals.execution import run_corpus_suite
from cayu.evals.execution_comparison import compare_corpus_execution_results
from cayu.evals.execution_reporting import eval_result_report_to_json, render_corpus_execution_html
from cayu.evals.result_presentation import present_eval_result
from cayu.evals.store import (
    EvalBaselineKey,
    EvalBaselineUpdate,
    EvalRunInvocation,
    EvalRunRequest,
    EvalStoreResultTooLarge,
)
from cayu.server import DashboardConfig, ServerApiConfig, ServerConfig, create_server
from cayu.server import routes as routes_module
from cayu.server.contracts import EvalComparisonResponse, EvalResultResponse
from cayu.storage.evals_sqlite import SQLiteEvalStore


async def _admit_run(store, target, run_id, *, target_key=None):
    corpus = _corpus(target_key=target.key if target_key is None else target_key, trials=1)
    await store.save_corpus(corpus, redact_json=target.app.redact_json)
    request = EvalRunRequest(
        run_id=run_id,
        idempotency_key="sha256:" + hashlib.sha256(run_id.encode()).hexdigest(),
        corpus_revision=corpus.revision,
        target_key=corpus.target_key,
        suite_id=corpus.suites[0].id,
        suite_revision=corpus.suites[0].revision,
        max_concurrency=1,
        invocation=(
            await _bound_eval_invocation(target)
            if corpus.target_key == target.key
            else EvalRunInvocation()
        ),
    )
    return await store.admit_run(request, redact_json=target.app.redact_json)


def _run_requests(root, run_id):
    detail = f"{root}/runs/{run_id}"
    return [
        ("GET", detail, None),
        (
            "POST",
            f"{detail}/scenario-approval",
            {
                "expected_progress_revision": "sha256:" + "0" * 64,
                "trial_number": 1,
                "event_id": "approval",
                "decision": "approve",
            },
        ),
        ("POST", f"{detail}/cancel", None),
        ("GET", f"{detail}/result", None),
        ("GET", f"{detail}/report.json", None),
        ("GET", f"{detail}/report.html", None),
        (
            "POST",
            f"{root}/comparisons",
            {"baseline_run_id": run_id, "current_run_id": run_id},
        ),
    ]


@pytest.mark.parametrize("prefix", ["/api", "/custom/v2"])
def test_run_routes_preserve_auth_bounds_pagination_and_cancellation(sqlite_resources, prefix):
    async def exercise():
        async with sqlite_resources as resources:
            provider = _provider()
            target = _target(provider)
            store = resources.own(SQLiteEvalStore(resources.path()))
            first = await _admit_run(store, target, "first-run")
            second = await _admit_run(store, target, "second-run")
            await _admit_run(store, target, "hidden-run", target_key="unpublished")
            calls = []

            def authenticate(request: Request):
                calls.append(request.url.path)
                return _authenticate(request)

            server = create_server(
                target.app,
                config=ServerConfig.protected(
                    authenticate,
                    api=ServerApiConfig(path=prefix),
                    dashboard=DashboardConfig(enabled=False),
                    evals=_evals_config(target, store),
                ),
            )
            root = f"{prefix}/evals"
            requests = [("GET", f"{root}/runs", None), *_run_requests(root, "first-run")]
            route_keys = [
                (method, path.replace("first-run", "{run_id}")) for method, path, _ in requests
            ]
            routes = [route for route in server.routes if isinstance(route, APIRoute)]
            run_routes = [
                route
                for route in routes
                if any((method, route.path) in route_keys for method in route.methods)
            ]
            assert [(next(iter(route.methods)), route.path) for route in run_routes] == route_keys
            dependency = (
                next(route for route in routes if route.path == f"{prefix}/sessions")
                .dependencies[0]
                .dependency
            )
            for route in run_routes:
                assert isinstance(route, routes_module._BoundedEvalsRoute)
                assert route.dependencies[0].dependency is dependency
                assert type(route).preparse_auth is authenticate

            # Exercise the HTTP adapter without starting the execution worker:
            # queued records stay stable while pagination and cancellation run.
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                for method, path, _ in requests:
                    calls.clear()
                    denied = await client.request(
                        method,
                        path,
                        content=b"private invalid input",
                        headers={"Content-Length": str(run_routes[0].max_request_bytes + 1)},
                    )
                    assert denied.status_code == 401
                    assert denied.json() == {"detail": "unauthorized"}
                    assert denied.headers["cache-control"] == "private, no-store"
                    assert calls == [path]

                query = {
                    "limit": 1,
                    "status": "queued",
                    "corpus_revision": first.spec.corpus_revision,
                }
                page = await client.get(f"{root}/runs", headers=_AUTH_HEADERS, params=query)
                assert page.status_code == 200
                assert len(page.json()["items"]) == 1
                assert page.json()["next_cursor"] is not None
                next_page = await client.get(
                    f"{root}/runs",
                    headers=_AUTH_HEADERS,
                    params={**query, "cursor": page.json()["next_cursor"]},
                )
                assert next_page.status_code == 200
                assert next_page.json()["next_cursor"] is None
                assert {
                    item["spec"]["run_id"]
                    for item in page.json()["items"] + next_page.json()["items"]
                } == {first.spec.run_id, second.spec.run_id}
                for params, detail in (
                    ({"limit": 0}, "Invalid Evals request."),
                    ({"max_result_bytes": 1023}, "Invalid Evals request."),
                    ({"cursor": "invalid"}, "Invalid Evals query."),
                ):
                    invalid = await client.get(f"{root}/runs", headers=_AUTH_HEADERS, params=params)
                    assert invalid.status_code == 422
                    assert invalid.json() == {"detail": detail}
                hidden = await client.get(
                    f"{root}/runs", headers=_AUTH_HEADERS, params={"target_key": "unpublished"}
                )
                assert hidden.status_code == 404
                assert hidden.json() == {"detail": "Eval target not found."}

                for (method, path, body), expected in zip(
                    requests, [200, 200, 409, 202, 409, 409, 409, 409], strict=True
                ):
                    calls.clear()
                    response = await client.request(method, path, headers=_AUTH_HEADERS, json=body)
                    assert response.status_code == expected
                    assert response.headers["cache-control"] == "private, no-store"
                    assert calls == [path]
                    if path.endswith("/cancel"):
                        assert response.json()["status"] == "cancelled"
                    elif path.endswith("/scenario-approval"):
                        assert response.json() == {"detail": "Eval run is not a scenario run."}

                def deny_access():
                    raise HTTPException(status_code=403, detail="run access revoked")

                server.dependency_overrides[dependency] = deny_access
                for method, path, body in requests:
                    response = await client.request(method, path, headers=_AUTH_HEADERS, json=body)
                    assert response.status_code == 403
                    assert response.json() == {"detail": "run access revoked"}
            assert provider.requests == []

    asyncio.run(exercise())


@pytest.mark.parametrize("failure", ["hidden", "missing", "invalid"])
def test_run_visibility_precedes_result_reads_and_mutations(sqlite_resources, monkeypatch, failure):
    async def exercise():
        async with sqlite_resources as resources:
            provider = _provider()
            target = _target(provider)
            store = resources.own(SQLiteEvalStore(resources.path()))
            if failure == "hidden":
                await _admit_run(store, target, "unreadable", target_key="unpublished")
            original = store.load_run
            reads = []

            async def load_run(run_id):
                reads.append(run_id)
                if failure == "invalid":
                    raise ValueError("private store input")
                return await original(run_id)

            async def unexpected(*args, **kwargs):
                pytest.fail("An unreadable run must not load results or mutate state.")

            monkeypatch.setattr(store, "load_run", load_run)
            for name in (
                "load_result",
                "request_cancel",
                "submit_scenario_approval",
                "load_trial_evidence_links",
                "load_baseline",
            ):
                monkeypatch.setattr(store, name, unexpected)
            server = _server(target, store)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                for method, path, body in _run_requests("/api/evals", "unreadable"):
                    response = await client.request(method, path, headers=_AUTH_HEADERS, json=body)
                    assert response.status_code == (422 if failure == "invalid" else 404)
                    assert response.json() == {
                        "detail": "Invalid Evals query."
                        if failure == "invalid"
                        else "Eval run not found."
                    }
                    assert response.headers["cache-control"] == "private, no-store"
            assert reads == ["unreadable"] * 7
            assert provider.requests == []

    asyncio.run(exercise())


@pytest.mark.parametrize("failure", ["missing", "oversized"])
def test_run_result_routes_preserve_private_storage_errors(sqlite_resources, monkeypatch, failure):
    async def exercise():
        async with sqlite_resources as resources:
            target = _target(_provider())
            store = resources.own(SQLiteEvalStore(resources.path()))
            await _admit_run(store, target, "pending-run")
            reads = []

            async def load_result(run_id):
                reads.append(run_id)
                if failure == "oversized":
                    raise EvalStoreResultTooLarge(1024)
                return None

            monkeypatch.setattr(store, "load_result", load_result)
            server = _server(target, store)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                for method, path, body in _run_requests("/api/evals", "pending-run")[3:]:
                    response = await client.request(method, path, headers=_AUTH_HEADERS, json=body)
                    assert response.status_code == (413 if failure == "oversized" else 409)
                    assert response.json() == {
                        "detail": "Eval result exceeds the server byte limit."
                        if failure == "oversized"
                        else "Eval run has no completed result."
                    }
                    assert response.headers["cache-control"] == "private, no-store"
            assert reads == ["pending-run"] * 4

    asyncio.run(exercise())


@pytest.mark.parametrize("catalog_available", [False, True])
def test_run_results_reports_and_same_run_comparison_preserve_store_work(
    sqlite_resources, monkeypatch, catalog_available
):
    async def exercise():
        async with sqlite_resources as resources:
            target = _target(_provider(trials=1))
            store = resources.own(SQLiteEvalStore(resources.path()))
            await _admit_run(store, target, "completed-run")
            corpus = _corpus(trials=1)
            result = await run_corpus_suite(target, corpus, corpus.suites[0].id, max_concurrency=1)
            lease = await store.claim_run(target_key=target.key, lease_seconds=300)
            assert lease is not None
            completed = await store.publish_result(
                lease.claim, result, redact_json=target.app.redact_json
            )
            key = EvalBaselineKey(
                target_key=target.key, corpus_revision=corpus.revision, suite_id=corpus.suites[0].id
            )
            await store.set_baseline(
                EvalBaselineUpdate(
                    key=key,
                    result_revision=result.revision,
                    expected_generation=0,
                    operation_id="sha256:" + "1" * 64,
                    actor_id="reviewer",
                ),
                redact_json=target.app.redact_json,
            )
            baseline = await store.load_baseline(key) if catalog_available else None
            links = await store.load_trial_evidence_links("completed-run")
            monkeypatch.setattr(store, "captured_results", catalog_available)
            calls = []

            def observe(name):
                original = getattr(store, name)

                async def wrapped(*args, **kwargs):
                    calls.append(name)
                    if name == "load_baseline":
                        assert catalog_available
                        assert args == (key,)
                    return await original(*args, **kwargs)

                monkeypatch.setattr(store, name, wrapped)

            for name in ("load_run", "load_result", "load_trial_evidence_links", "load_baseline"):
                observe(name)
            server = _server(target, store)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                root = "/api/evals/runs/completed-run"
                response = await client.get(f"{root}/result", headers=_AUTH_HEADERS)
                assert response.status_code == 200
                expected = EvalResultResponse(
                    run=completed,
                    result=result,
                    presentation=present_eval_result(result),
                    baseline=baseline,
                    trial_evidence_links=links,
                )
                assert response.json() == expected.model_dump(mode="json")
                assert calls == [
                    "load_run",
                    "load_result",
                    "load_run",
                    "load_trial_evidence_links",
                ] + (["load_baseline"] if catalog_available else [])
                for suffix, render, media_type, filename in (
                    (
                        "report.json",
                        eval_result_report_to_json,
                        "application/json",
                        "completed-run.eval-result.json",
                    ),
                    (
                        "report.html",
                        render_corpus_execution_html,
                        "text/html; charset=utf-8",
                        "completed-run.eval-report.html",
                    ),
                ):
                    calls.clear()
                    report = await client.get(f"{root}/{suffix}", headers=_AUTH_HEADERS)
                    assert report.status_code == 200
                    assert report.content == render(result).encode("utf-8")
                    assert report.headers["content-type"] == media_type
                    assert (
                        report.headers["content-disposition"]
                        == f'attachment; filename="{filename}"'
                    )
                    assert report.headers["cache-control"] == "private, no-store"
                    assert calls == ["load_run", "load_result", "load_run"]
                calls.clear()
                comparison = await client.post(
                    "/api/evals/comparisons",
                    headers=_AUTH_HEADERS,
                    json={"baseline_run_id": "completed-run", "current_run_id": "completed-run"},
                )
                assert comparison.status_code == 200
                expected_comparison = EvalComparisonResponse(
                    baseline=completed,
                    current=completed,
                    comparison=compare_corpus_execution_results(result, result),
                )
                assert comparison.json() == expected_comparison.model_dump(mode="json")
                assert calls == ["load_run", "load_result", "load_run"]

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "operation,failure",
    [
        ("list_runs", "invalid"),
        ("list_runs", "oversized"),
        ("request_cancel", "invalid"),
        ("request_cancel", "missing"),
    ],
)
def test_run_queries_and_cancellation_preserve_private_errors(
    sqlite_resources, monkeypatch, operation, failure
):
    async def exercise():
        async with sqlite_resources as resources:
            target = _target(_provider())
            store = resources.own(SQLiteEvalStore(resources.path()))
            await _admit_run(store, target, "queued-run")
            calls = []

            async def fail(argument):
                calls.append(argument)
                if failure == "oversized":
                    raise EvalStoreResultTooLarge(1024)
                if failure == "missing":
                    raise KeyError("private run identity")
                raise ValueError("private store input")

            monkeypatch.setattr(store, operation, fail)
            server = _server(target, store)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                if operation == "list_runs":
                    response = await client.get(
                        "/api/evals/runs",
                        headers=_AUTH_HEADERS,
                        params={"limit": 1, "max_result_bytes": 1024, "status": "queued"},
                    )
                    assert len(calls) == 1
                    assert calls[0].target_key == target.key
                    assert calls[0].limit == 1
                    assert calls[0].max_result_bytes == 1024
                    assert calls[0].status == "queued"
                else:
                    response = await client.post(
                        "/api/evals/runs/queued-run/cancel", headers=_AUTH_HEADERS
                    )
                    assert calls == ["queued-run"]
                status, detail = {
                    "oversized": (413, "Eval run page exceeds the requested byte limit."),
                    "missing": (404, "Eval run not found."),
                    "invalid": (422, "Invalid Evals query."),
                }[failure]
                assert response.status_code == status
                assert response.json() == {"detail": detail}
                assert response.headers["cache-control"] == "private, no-store"

    asyncio.run(exercise())
