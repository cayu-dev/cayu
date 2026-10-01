from __future__ import annotations

import asyncio
import hashlib
import inspect
from pathlib import Path

import httpx
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi import HTTPException, Request
from fastapi.routing import APIRoute
from tests.server.test_server_evaluation_promotion import (
    _AUTH_HEADERS,
    _SESSION_ID,
    _authenticate,
    _seed_app,
)

from cayu.evals.store import (
    EvalCorpusConflict,
    EvalResultConflict,
    EvalRunQuery,
    EvalStorePublicationRejected,
    EvalStoreResultTooLarge,
)
from cayu.project_control_plane import (
    ProjectControlPlaneAccess,
    _create_project_control_plane_context,
)
from cayu.server import AuthContext, DashboardConfig, ServerApiConfig, ServerConfig, create_server
from cayu.server import routes as routes_module
from cayu.server.evals_registry import EvalTargetRegistry
from cayu.storage.evals_sqlite import SQLiteEvalStore


async def _server(resources, monkeypatch, *, prefix="/api", authenticate=_authenticate):
    app = await _seed_app()

    async def unexpected_model_call(request):
        yield pytest.fail("Launch admission must not dispatch model work without a worker.")

    monkeypatch.setattr(app.get_provider(), "stream", unexpected_model_call)
    store = SQLiteEvalStore(resources.path())
    context = resources.own(
        _create_project_control_plane_context(
            project_root=Path(__file__).resolve().parents[2],
            project_id="captured-launch-contract",
            configured_release_id="release-current",
            eval_store=store,
            store_backend="sqlite",
            store_source="project",
            access=ProjectControlPlaneAccess.AUTHENTICATED_PRODUCTION,
        ),
        kind="registry",
    )
    server = create_server(
        app,
        config=ServerConfig.protected(
            authenticate,
            api=ServerApiConfig(path=prefix),
            dashboard=DashboardConfig(enabled=False),
        ),
        project_context=context,
    )
    return app, store, server


async def _reviewed_request(client, *, prefix="/api"):
    root = f"{prefix}/evals/sessions/{_SESSION_ID}/evaluation"
    preview = await client.post(f"{root}/preview", headers=_AUTH_HEADERS, json={})
    assert preview.status_code == 200
    assert preview.json()["runnable_conversion"]["available"] is True
    candidate = preview.json()["candidate"]
    catalog = await client.get(f"{prefix}/evals/targets", headers=_AUTH_HEADERS)
    assert catalog.status_code == 200
    target = next(
        item for item in catalog.json()["items"] if item["target_key"] == candidate["target_key"]
    )
    assert target["execution_profile_ready"] is True
    return f"{root}/launch", {
        "candidate": candidate,
        "expected_candidate_revision": candidate["revision"],
        "expected_execution_profile_revision": target["execution_profile"]["revision"],
        "trial_request": {"trials": 1, "timeout_seconds": 30},
        "max_concurrency": 1,
        "max_steps": 3,
    }


@pytest.mark.parametrize("prefix", ["/api", "/custom/v2"])
def test_captured_launch_preserves_shared_auth_body_handling_and_provenance(
    sqlite_resources, monkeypatch, prefix
):
    async def exercise():
        async with sqlite_resources as resources:
            calls = []

            async def authenticate(request: Request):
                calls.append(request.url.path)
                _authenticate(request)
                await request.body()
                return AuthContext(subject="captured-operator", tenant="captured-tenant")

            _, _, server = await _server(
                resources, monkeypatch, prefix=prefix, authenticate=authenticate
            )
            routes = [route for route in server.routes if isinstance(route, APIRoute)]
            launch = next(
                route
                for route in routes
                if route.path == f"{prefix}/evals/sessions/{{session_id}}/evaluation/launch"
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
            assert isinstance(launch, routes_module._BoundedCapturedEvaluationRoute)
            # Keep admissions queued to inspect durable provenance without starting a worker.
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                path, body = await _reviewed_request(client, prefix=prefix)
                calls.clear()
                denied = await client.post(path, json=body)
                assert denied.status_code == 401
                assert denied.json() == {"detail": "unauthorized"}
                assert denied.headers["cache-control"] == "private, no-store"
                assert calls == [path]
                headers = {**_AUTH_HEADERS, "Idempotency-Key": "captured-boundary"}
                calls.clear()
                response = await client.post(path, headers=headers, json=body)
                assert response.status_code == 202
                assert response.headers["cache-control"] == "private, no-store"
                assert calls == [path]
                assert response.json()["run"]["spec"]["invocation"]["origin"] == {
                    "trust": "server_verified",
                    "subject": "captured-operator",
                    "tenant": "captured-tenant",
                }
                for content, extra_headers, status, detail in (
                    (
                        b"{}",
                        {"Content-Length": str(launch.max_request_bytes + 1)},
                        413,
                        "Captured evaluation request exceeds the server byte limit.",
                    ),
                    (b'{"x":1,"x":2}', {}, 422, "Invalid captured evaluation request."),
                    (b"NaN", {}, 422, "Invalid captured evaluation request."),
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
                    # Captured routes reject invalid bodies before endpoint authentication.
                    assert calls == []

                def deny_access():
                    raise HTTPException(status_code=403, detail="launch access revoked")

                server.dependency_overrides[session.dependencies[0].dependency] = deny_access
                rejected = await client.post(path, headers=headers, json=body)
                assert rejected.status_code == 403
                assert rejected.json() == {"detail": "launch access revoked"}

    asyncio.run(exercise())


@pytest.mark.parametrize("failure", ["corpus-conflict", "result-conflict", "unsafe", "oversized"])
def test_captured_launch_publication_errors_remain_private_and_precede_admission(
    sqlite_resources, monkeypatch, failure
):
    async def exercise():
        async with sqlite_resources as resources:
            app, store, server = await _server(resources, monkeypatch)
            calls = []
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                path, body = await _reviewed_request(client)

                async def fail_publication(corpus, result, *, redact_json):
                    assert redact_json == app.redact_json
                    assert result.corpus_revision == corpus.revision
                    calls.append(result.revision)
                    if failure == "oversized":
                        raise EvalStoreResultTooLarge(1024)
                    error = {
                        "corpus-conflict": EvalCorpusConflict,
                        "result-conflict": EvalResultConflict,
                        "unsafe": EvalStorePublicationRejected,
                    }[failure]
                    raise error("private captured storage detail")

                async def unexpected_admission(*args, **kwargs):
                    pytest.fail("Captured evidence must be published before admission.")

                monkeypatch.setattr(store, "save_captured_result", fail_publication)
                monkeypatch.setattr(store, "admit_run", unexpected_admission)
                response = await client.post(
                    path,
                    headers={**_AUTH_HEADERS, "Idempotency-Key": "publication-failure"},
                    json=body,
                )
                if failure.endswith("conflict"):
                    status, detail = (
                        409,
                        "The immutable runnable evaluation conflicts with stored content.",
                    )
                elif failure == "unsafe":
                    status, detail = 422, "The runnable evaluation contains unsafe public data."
                else:
                    status, detail = 413, "The runnable evaluation exceeds the server byte limit."
                assert response.status_code == status
                assert response.json() == {"detail": detail}
                assert response.headers["cache-control"] == "private, no-store"
                assert len(calls) == 1
                for resource in ("corpora", "results", "runs"):
                    catalog = await client.get(
                        f"/api/evals/{resource}",
                        headers=_AUTH_HEADERS,
                        params={"target_key": body["candidate"]["target_key"]},
                    )
                    assert catalog.status_code == 200
                    assert catalog.json()["items"] == []

    asyncio.run(exercise())


def test_captured_launch_retries_after_publication_and_replays_before_readiness(
    sqlite_resources, monkeypatch
):
    async def exercise():
        async with sqlite_resources as resources:
            app, store, server = await _server(resources, monkeypatch)
            events = []
            records = []
            admissions = []
            save = store.save_captured_result
            admit = store.admit_run
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                path, body = await _reviewed_request(client)
                headers = {**_AUTH_HEADERS, "Idempotency-Key": "captured-retry"}

                async def observe_publication(corpus, result, *, redact_json):
                    assert redact_json == app.redact_json
                    record = await save(corpus, result, redact_json=redact_json)
                    records.append(record)
                    events.append("published")
                    return record

                async def fail_first_admission(request, *, redact_json):
                    assert redact_json == app.redact_json
                    assert request.corpus_revision == records[-1].corpus_revision
                    admissions.append(request)
                    events.append("admission")
                    if len(admissions) == 1:
                        raise EvalStorePublicationRejected("private admission detail")
                    return await admit(request, redact_json=redact_json)

                monkeypatch.setattr(store, "save_captured_result", observe_publication)
                monkeypatch.setattr(store, "admit_run", fail_first_admission)
                failed = await client.post(path, headers=headers, json=body)
                assert failed.status_code == 422
                assert failed.json() == {"detail": "Eval run request contains unsafe public data."}
                target_key = body["candidate"]["target_key"]
                assert events == ["published", "admission"]
                assert await store.load_result_record(records[0].revision) == records[0]
                assert (await store.list_runs(EvalRunQuery(target_key=target_key))).items == ()
                restored = await client.post(path, headers=headers, json=body)
                assert restored.status_code == 202
                assert events == ["published", "admission", "published", "admission"]
                assert records[0] == records[1]
                assert admissions[0].idempotency_key == admissions[1].idempotency_key
                digest = hashlib.sha256(
                    b"cayu-server-eval-idempotency-v1\0"
                    + target_key.encode("utf-8")
                    + b"\0captured-retry"
                ).hexdigest()
                assert admissions[1].idempotency_key == "sha256:" + digest

                async def unavailable_profile(*args, **kwargs):
                    pytest.fail("Accepted retries must not prepare a new execution profile.")

                monkeypatch.setattr(
                    EvalTargetRegistry, "prepare_execution_profile", unavailable_profile
                )
                replayed = await client.post(path, headers=headers, json=body)
                assert replayed.status_code == 202
                assert replayed.json() == restored.json()
                conflict = await client.post(
                    path, headers=headers, json={**body, "max_steps": body["max_steps"] + 1}
                )
                assert conflict.status_code == 409
                assert conflict.json() == {
                    "detail": "Idempotency-Key is already bound to another eval run request."
                }
                assert len(records) == len(admissions) == 2
                assert len((await store.list_runs(EvalRunQuery(target_key=target_key))).items) == 1

    asyncio.run(exercise())
