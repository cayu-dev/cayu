from __future__ import annotations

import asyncio
import inspect
import json

import httpx
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi import HTTPException, Request
from fastapi.routing import APIRoute
from tests.server.test_server_eval_admission import _launch_body
from tests.server.test_server_evals import (
    _AUTH_HEADERS,
    _authenticate,
    _corpus,
    _evals_config,
    _provider,
    _server,
    _target,
)

from cayu.evals.store import EvalRunQuery
from cayu.server import DashboardConfig, ServerApiConfig, ServerConfig, create_server
from cayu.server import routes as routes_module
from cayu.server.evals_registry import EvalTargetRegistry
from cayu.storage.evals_sqlite import SQLiteEvalStore


@pytest.mark.parametrize("prefix", ["/api", "/custom/v2"])
def test_corpus_launch_preserves_shared_auth_and_private_json_boundary(
    sqlite_resources, monkeypatch, prefix
):
    async def exercise():
        async with sqlite_resources as resources:
            provider = _provider(trials=1)
            target = _target(provider)
            store = resources.own(SQLiteEvalStore(resources.path()))
            corpus = _corpus(trials=1)
            calls = []

            async def authenticate(request: Request):
                calls.append(request.url.path)
                identity = _authenticate(request)
                await request.body()
                return identity

            server = create_server(
                target.app,
                config=ServerConfig.protected(
                    authenticate,
                    dashboard=DashboardConfig(enabled=False),
                    api=ServerApiConfig(path=prefix),
                    evals=_evals_config(target, store),
                ),
            )
            path = f"{prefix}/evals/runs"
            routes = [route for route in server.routes if isinstance(route, APIRoute)]
            launch = next(
                route for route in routes if route.path == path and "POST" in route.methods
            )
            contract = next(route for route in routes if route.path == f"{prefix}/contract")
            auth_default = inspect.signature(launch.endpoint).parameters["auth_context"].default
            assert (
                auth_default
                is inspect.signature(contract.endpoint).parameters["auth_context"].default
            )
            assert launch.dependencies == []
            assert isinstance(launch, routes_module._BoundedEvalsRoute)
            assert type(launch).preparse_auth is authenticate

            # Observe admission without starting the worker that consumes queued runs.
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                imported = await client.post(
                    f"{prefix}/evals/corpora",
                    headers=_AUTH_HEADERS,
                    json=corpus.model_dump(mode="json"),
                )
                assert imported.status_code == 201
                body = await _launch_body(client, corpus, prefix=prefix)
                headers = {**_AUTH_HEADERS, "Idempotency-Key": "corpus-boundary"}
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

                calls.clear()
                accepted = await client.post(path, headers=headers, json=body)
                assert accepted.status_code == 202
                assert accepted.headers["cache-control"] == "private, no-store"
                assert calls == [path]
                invocation = accepted.json()["spec"]["invocation"]
                assert invocation["origin"]["subject"] == "eval-operator"
                assert invocation["max_steps"] == body["max_steps"]

                async def unexpected_read(*args, **kwargs):
                    pytest.fail("Rejected HTTP input must not read the evaluation corpus.")

                monkeypatch.setattr(store, "load_corpus", unexpected_read)
                for content, extra_headers, status, detail in (
                    (
                        b"{}",
                        {"Content-Length": str(launch.max_request_bytes + 1)},
                        413,
                        "Evals request exceeds the server byte limit.",
                    ),
                    (b'{"x":1,"x":2}', {}, 422, "Invalid Evals request."),
                    (b"NaN", {}, 422, "Invalid Evals request."),
                    (b'{"x":"\xff"}', {}, 422, "Invalid Evals request."),
                    (
                        b'{"max_steps":"private invalid bound"}',
                        {},
                        422,
                        "Invalid Evals request.",
                    ),
                    (
                        json.dumps(body).encode(),
                        {"Content-Type": "text/plain"},
                        422,
                        "Invalid Evals request.",
                    ),
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
                    raise HTTPException(status_code=403, detail="corpus launch access revoked")

                server.dependency_overrides[auth_default.dependency] = deny_access
                rejected = await client.post(path, headers=headers, json=body)
                assert rejected.status_code == 403
                assert rejected.json() == {"detail": "corpus launch access revoked"}
                assert rejected.headers["cache-control"] == "private, no-store"
            assert len((await store.list_runs(EvalRunQuery(target_key=target.key))).items) == 1
            assert provider.requests == []

    asyncio.run(exercise())


@pytest.mark.parametrize("phase", ["corpus_read", "profile_preparation"])
def test_corpus_launch_cancellation_before_admission_does_not_persist_work(
    sqlite_resources, monkeypatch, phase
):
    async def exercise():
        async with sqlite_resources as resources:
            provider = _provider(trials=1)
            target = _target(provider)
            store = resources.own(SQLiteEvalStore(resources.path()))
            corpus = _corpus(trials=1)
            await store.save_corpus(corpus, redact_json=target.app.redact_json)
            load_corpus = store.load_corpus
            server = _server(target, store)
            entered = asyncio.Event()
            release = asyncio.Event()
            exited = asyncio.Event()
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                body = await _launch_body(client, corpus)

                async def blocked_preparation(*args, **kwargs):
                    entered.set()
                    try:
                        await release.wait()
                    finally:
                        exited.set()

                async def unexpected_admission(*args, **kwargs):
                    pytest.fail("Cancelled preparation must not reach durable admission.")

                if phase == "corpus_read":
                    monkeypatch.setattr(store, "load_corpus", blocked_preparation)
                else:
                    monkeypatch.setattr(
                        EvalTargetRegistry, "prepare_execution_profile", blocked_preparation
                    )
                monkeypatch.setattr(store, "admit_run", unexpected_admission)
                request = resources.task(
                    client.post(
                        "/api/evals/runs",
                        headers={**_AUTH_HEADERS, "Idempotency-Key": "cancelled-corpus"},
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
                assert await load_corpus(corpus.revision) == corpus
                assert (await store.list_runs(EvalRunQuery(target_key=target.key))).items == ()
            assert provider.requests == []

    asyncio.run(exercise())
