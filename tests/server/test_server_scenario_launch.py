from __future__ import annotations

import asyncio
import hashlib
import inspect

import httpx
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi import HTTPException, Request
from fastapi.routing import APIRoute
from tests.server.test_server_eval_scenarios import _AUTH_HEADERS, _authenticate, _scenario, _target

from cayu.evals.store import (
    EvalCorpusConflict,
    EvalRunQuery,
    EvalStorePublicationRejected,
    EvalStoreResultTooLarge,
)
from cayu.server import (
    AuthContext,
    DashboardConfig,
    EvalsConfig,
    ServerApiConfig,
    ServerConfig,
    create_server,
)
from cayu.server import routes as routes_module
from cayu.server.evals_registry import EvalTargetRegistry
from cayu.storage.evals_sqlite import SQLiteEvalStore


def _server(resources, *, prefix="/api", authenticate=_authenticate):
    target, _, provider = _target(resources.path("target"))
    store = resources.own(SQLiteEvalStore(resources.path()))
    server = create_server(
        target.app,
        config=ServerConfig.protected(
            authenticate,
            dashboard=DashboardConfig(enabled=False),
            api=ServerApiConfig(path=prefix),
            evals=EvalsConfig(target=target, store=store),
        ),
    )
    return target, store, provider, server


async def _reviewed_request(client, *, prefix="/api"):
    scenario = _scenario()
    settings = {
        "environment_name": "files",
        "timeout_seconds": 30,
        "max_steps": 3,
        "limits": {"max_total_tokens": 100, "scope": "run"},
    }
    response = await client.post(
        f"{prefix}/evals/scenarios",
        headers=_AUTH_HEADERS,
        json={
            "scenario": scenario.model_dump(mode="json"),
            "expected_scenario_revision": scenario.revision,
            "settings": settings,
        },
    )
    assert response.status_code == 201
    reviewed = response.json()
    assert reviewed["preflight"]["ready"] is True
    assert reviewed["execution_profile_revision"] is not None
    return f"{prefix}/evals/scenarios/{scenario.revision}/runs", {
        "expected_binding_revision": reviewed["preflight"]["binding"]["revision"],
        "expected_execution_profile_revision": reviewed["execution_profile_revision"],
        "settings": settings,
    }


@pytest.mark.parametrize("prefix", ["/api", "/custom/v2"])
def test_scenario_launch_preserves_shared_auth_body_handling_and_bound_invocation(
    sqlite_resources, prefix
):
    async def exercise():
        async with sqlite_resources as resources:
            calls = []

            async def authenticate(request: Request):
                calls.append(request.url.path)
                _authenticate(request)
                await request.body()
                return AuthContext(subject="scenario-reviewer", tenant="scenario-tenant")

            _, _, provider, server = _server(resources, prefix=prefix, authenticate=authenticate)
            routes = [route for route in server.routes if isinstance(route, APIRoute)]
            launch = next(
                route
                for route in routes
                if route.path == f"{prefix}/evals/scenarios/{{scenario_revision}}/runs"
            )
            session = next(route for route in routes if route.path == f"{prefix}/sessions")
            run = next(
                route
                for route in routes
                if route.path == f"{prefix}/evals/runs" and "POST" in route.methods
            )
            assert launch.dependencies[0].dependency is session.dependencies[0].dependency
            assert (
                inspect.signature(launch.endpoint).parameters["auth_context"].default
                is inspect.signature(run.endpoint).parameters["auth_context"].default
            )
            assert isinstance(launch, routes_module._BoundedEvalsRoute)
            assert type(launch).preparse_auth is authenticate
            # Keep the admitted run queued so assertions observe admission alone.
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                path, body = await _reviewed_request(client, prefix=prefix)
                calls.clear()
                denied = await client.post(
                    path,
                    content=b"private invalid input",
                    headers={"Content-Length": str(launch.max_request_bytes + 1)},
                )
                assert denied.status_code == 401
                assert denied.json() == {"detail": "unauthorized"}
                assert denied.headers["cache-control"] == "private, no-store"
                assert calls == [path]
                headers = {**_AUTH_HEADERS, "Idempotency-Key": "scenario-boundary"}
                calls.clear()
                accepted = await client.post(path, headers=headers, json=body)
                assert accepted.status_code == 202
                assert accepted.headers["cache-control"] == "private, no-store"
                assert calls == [path]
                invocation = accepted.json()["spec"]["invocation"]
                assert invocation["origin"] == {
                    "trust": "server_verified",
                    "subject": "scenario-reviewer",
                    "tenant": "scenario-tenant",
                }
                assert invocation["max_steps"] == 3
                assert invocation["limits"]["max_total_tokens"] == 100
                assert invocation["scenario"]["scenario_revision"] == _scenario().revision
                assert (
                    invocation["scenario"]["binding_revision"] == body["expected_binding_revision"]
                )
                assert invocation["scenario"]["environment_name"] == "files"
                assert invocation["scenario"]["timeout_seconds"] == 30
                assert (
                    invocation["execution_profile"]["profile_revision"]
                    == body["expected_execution_profile_revision"]
                )
                for content, extra_headers, status, detail in (
                    (
                        b"{}",
                        {"Content-Length": str(launch.max_request_bytes + 1)},
                        413,
                        "Evals request exceeds the server byte limit.",
                    ),
                    (b'{"x":1,"x":2}', {}, 422, "Invalid Evals request."),
                    (b"NaN", {}, 422, "Invalid Evals request."),
                ):
                    calls.clear()
                    rejected = await client.post(
                        path,
                        content=content,
                        headers={**headers, "Content-Type": "application/json", **extra_headers},
                    )
                    assert rejected.status_code == status
                    assert rejected.json() == {"detail": detail}
                    assert rejected.headers["cache-control"] == "private, no-store"
                    assert calls == [path]

                def deny_access():
                    raise HTTPException(status_code=403, detail="scenario launch access revoked")

                server.dependency_overrides[session.dependencies[0].dependency] = deny_access
                rejected = await client.post(path, headers=headers, json=body)
                assert rejected.status_code == 403
                assert rejected.json() == {"detail": "scenario launch access revoked"}
            assert provider.requests == []

    asyncio.run(exercise())


@pytest.mark.parametrize("failure", ["binding", "profile", "unsupported"])
def test_scenario_launch_rejects_unreviewed_work_before_publication(
    sqlite_resources, monkeypatch, failure
):
    async def exercise():
        async with sqlite_resources as resources:
            _, store, provider, server = _server(resources)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                path, body = await _reviewed_request(client)
                if failure == "unsupported":
                    monkeypatch.setattr(store, "scenario_execution", False)
                    detail = "Durable scenario execution is not available."
                else:
                    field = (
                        "expected_binding_revision"
                        if failure == "binding"
                        else "expected_execution_profile_revision"
                    )
                    body[field] = "sha256:" + "0" * 64
                    detail = (
                        "Eval scenario launch binding changed after review."
                        if failure == "binding"
                        else "The exact eval execution profile changed after readiness. Check launch readiness again."
                    )

                async def unexpected_publication(*args, **kwargs):
                    pytest.fail("Unreviewed work must not publish a corpus or admit a run.")

                monkeypatch.setattr(store, "save_corpus", unexpected_publication)
                monkeypatch.setattr(store, "admit_run", unexpected_publication)
                rejected = await client.post(
                    path,
                    headers={**_AUTH_HEADERS, "Idempotency-Key": "unreviewed-scenario"},
                    json=body,
                )
                assert rejected.status_code == 409
                assert rejected.json() == {"detail": detail}
                assert rejected.headers["cache-control"] == "private, no-store"
            assert provider.requests == []

    asyncio.run(exercise())


@pytest.mark.parametrize("failure", ["conflict", "unsafe", "oversized"])
def test_scenario_launch_publication_errors_remain_private_and_precede_admission(
    sqlite_resources, monkeypatch, failure
):
    async def exercise():
        async with sqlite_resources as resources:
            target, store, provider, server = _server(resources)
            calls = []
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                path, body = await _reviewed_request(client)

                async def fail_publication(corpus, *, redact_json):
                    assert redact_json == target.app.redact_json
                    calls.append(corpus.revision)
                    if failure == "oversized":
                        raise EvalStoreResultTooLarge(1024)
                    error = (
                        EvalCorpusConflict
                        if failure == "conflict"
                        else EvalStorePublicationRejected
                    )
                    raise error("private scenario corpus detail")

                async def unexpected_admission(*args, **kwargs):
                    pytest.fail("Derived corpus publication must precede admission.")

                monkeypatch.setattr(store, "save_corpus", fail_publication)
                monkeypatch.setattr(store, "admit_run", unexpected_admission)
                rejected = await client.post(
                    path,
                    headers={**_AUTH_HEADERS, "Idempotency-Key": "scenario-publication"},
                    json=body,
                )
                status, detail = {
                    "conflict": (
                        409,
                        "Derived scenario result contract conflicts with stored content.",
                    ),
                    "unsafe": (
                        422,
                        "Derived scenario result contract contains unsafe public data.",
                    ),
                    "oversized": (
                        413,
                        "Derived scenario result contract exceeds the server byte limit.",
                    ),
                }[failure]
                assert rejected.status_code == status
                assert rejected.json() == {"detail": detail}
                assert rejected.headers["cache-control"] == "private, no-store"
                assert len(calls) == 1
                assert await store.load_corpus(calls[0]) is None
                assert (await store.list_runs(EvalRunQuery(target_key=target.key))).items == ()
            assert provider.requests == []

    asyncio.run(exercise())


def test_scenario_launch_recovers_after_publication_and_replays_before_readiness(
    sqlite_resources, monkeypatch
):
    async def exercise():
        async with sqlite_resources as resources:
            target, store, provider, server = _server(resources)
            events = []
            corpora = []
            admissions = []
            save = store.save_corpus
            admit = store.admit_run
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                path, body = await _reviewed_request(client)
                headers = {**_AUTH_HEADERS, "Idempotency-Key": "scenario-recovery"}

                async def observe_publication(corpus, *, redact_json):
                    assert redact_json == target.app.redact_json
                    result = await save(corpus, redact_json=redact_json)
                    corpora.append(corpus)
                    events.append("published")
                    return result

                async def fail_first_admission(request, *, redact_json):
                    assert redact_json == target.app.redact_json
                    assert request.corpus_revision == corpora[-1].revision
                    admissions.append(request)
                    events.append("admission")
                    if len(admissions) == 1:
                        raise EvalStorePublicationRejected("private admission detail")
                    return await admit(request, redact_json=redact_json)

                monkeypatch.setattr(store, "save_corpus", observe_publication)
                monkeypatch.setattr(store, "admit_run", fail_first_admission)
                failed = await client.post(path, headers=headers, json=body)
                assert failed.status_code == 422
                assert failed.json() == {"detail": "Eval run request contains unsafe public data."}
                assert events == ["published", "admission"]
                assert await store.load_corpus(corpora[0].revision) == corpora[0]
                assert (await store.list_runs(EvalRunQuery(target_key=target.key))).items == ()
                restored = await client.post(path, headers=headers, json=body)
                assert restored.status_code == 202
                assert events == ["published", "admission", "published", "admission"]
                assert corpora[0] == corpora[1]
                digest = hashlib.sha256(
                    b"cayu-server-eval-idempotency-v1\0assistant.default\0scenario-recovery"
                ).hexdigest()
                assert (
                    admissions[0].idempotency_key
                    == admissions[1].idempotency_key
                    == "sha256:" + digest
                )

                async def unavailable_profile(*args, **kwargs):
                    pytest.fail("Accepted retries must not prepare a new execution profile.")

                monkeypatch.setattr(
                    EvalTargetRegistry, "prepare_execution_profile", unavailable_profile
                )
                replayed = await client.post(path, headers=headers, json=body)
                assert replayed.status_code == 202
                assert replayed.json() == restored.json()
                conflict = await client.post(
                    path,
                    headers=headers,
                    json={**body, "settings": {**body["settings"], "max_steps": 2}},
                )
                assert conflict.status_code == 409
                assert conflict.json() == {
                    "detail": "Idempotency-Key is already bound to another eval run request."
                }
                assert len(corpora) == len(admissions) == 2
                assert len((await store.list_runs(EvalRunQuery(target_key=target.key))).items) == 1
            assert provider.requests == []

    asyncio.run(exercise())


def test_scenario_launch_cancellation_before_publication_does_not_admit_work(
    sqlite_resources, monkeypatch
):
    async def exercise():
        async with sqlite_resources as resources:
            target, store, provider, server = _server(resources)
            entered = asyncio.Event()
            release = asyncio.Event()
            exited = asyncio.Event()
            corpora = []
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                path, body = await _reviewed_request(client)

                async def blocked_publication(corpus, *, redact_json):
                    corpora.append(corpus.revision)
                    entered.set()
                    try:
                        await release.wait()
                    finally:
                        exited.set()

                async def unexpected_admission(*args, **kwargs):
                    pytest.fail("Cancelled publication must not admit a run.")

                monkeypatch.setattr(store, "save_corpus", blocked_publication)
                monkeypatch.setattr(store, "admit_run", unexpected_admission)
                request = resources.task(
                    client.post(
                        path,
                        headers={**_AUTH_HEADERS, "Idempotency-Key": "cancelled-scenario"},
                        json=body,
                    )
                )
                try:
                    await asyncio.wait_for(entered.wait(), 5)
                    request.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await request
                    assert exited.is_set()
                finally:
                    release.set()
                    await asyncio.gather(request, return_exceptions=True)
                assert len(corpora) == 1
                assert await store.load_corpus(corpora[0]) is None
                assert (await store.list_runs(EvalRunQuery(target_key=target.key))).items == ()
            assert provider.requests == []

    asyncio.run(exercise())
