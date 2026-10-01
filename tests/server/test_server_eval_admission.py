from __future__ import annotations

import asyncio
import hashlib
import json

import httpx
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from tests.server.test_server_evals import (
    _AUTH_HEADERS,
    _authenticate,
    _corpus,
    _evals_config,
    _provider,
    _server,
    _target,
)

from cayu.evals._execution_profile_errors import EvalExecutionProfileChangedError
from cayu.evals.corpus import EvalCaseSpec, EvalCorpusDocument
from cayu.evals.store import EvalRunAdmissionConflict, EvalRunQuery, EvalStorePublicationRejected
from cayu.server import (
    AuthContext,
    DashboardConfig,
    OpenAccess,
    ServerApiConfig,
    ServerConfig,
    create_server,
)
from cayu.server.contracts import EvalRunCreateRequest
from cayu.server.evals_registry import EvalTargetRegistry
from cayu.storage.evals_sqlite import SQLiteEvalStore


async def _launch_body(client, corpus, *, prefix="/api"):
    response = await client.get(f"{prefix}/evals/targets", headers=_AUTH_HEADERS)
    assert response.status_code == 200
    target = next(
        item for item in response.json()["items"] if item["target_key"] == corpus.target_key
    )
    assert target["execution_profile_ready"] is True
    return {
        "corpus_revision": corpus.revision,
        "suite_id": corpus.suites[0].id,
        "expected_execution_profile_revision": target["execution_profile"]["revision"],
        "max_steps": 1,
        "limits": {"max_total_tokens": 100, "scope": "run"},
    }


@pytest.mark.parametrize("trusted_local", [False, True])
def test_admission_preserves_durable_identity_provenance_and_replay_order(
    sqlite_resources, monkeypatch, trusted_local
):
    async def exercise():
        async with sqlite_resources as resources:
            provider = _provider(trials=1)
            target = _target(provider)
            store = resources.own(SQLiteEvalStore(resources.path()))
            corpus = _corpus(trials=1)
            await store.save_corpus(corpus, redact_json=target.app.redact_json)
            identity = [AuthContext(subject="operator", tenant="tenant-one")]
            auth_calls = []

            def authenticate(request):
                auth_calls.append(request.url.path)
                _authenticate(request)
                return identity[0]

            prefix = "/api" if trusted_local else "/custom/v2"
            arguments = {
                "dashboard": DashboardConfig(enabled=False),
                "api": ServerApiConfig(path=prefix),
                "evals": _evals_config(target, store),
            }
            config = (
                ServerConfig(access=OpenAccess(trusted_local_development=True), **arguments)
                if trusted_local
                else ServerConfig.protected(authenticate, **arguments)
            )
            server = create_server(target.app, config=config)
            events = []
            admitted_requests = []

            def observe_store(name):
                original = getattr(store, name)

                async def observed(*args, **kwargs):
                    events.append(name)
                    if name == "admit_run":
                        assert kwargs["redact_json"] == target.app.redact_json
                        admitted_requests.append(args[0])
                    return await original(*args, **kwargs)

                monkeypatch.setattr(store, name, observed)

            # No worker is started: retries must read the same durable queued run.
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://localhost"
            ) as client:
                body = await _launch_body(client, corpus, prefix=prefix)
                for name in ("load_corpus", "load_run_by_idempotency_key", "admit_run"):
                    observe_store(name)
                prepare = EvalTargetRegistry.prepare_execution_profile

                async def observed_prepare(registry, target_key, *, effective_target=None):
                    events.append(
                        "published-profile" if effective_target is None else "effective-profile"
                    )
                    return await prepare(registry, target_key, effective_target=effective_target)

                monkeypatch.setattr(
                    EvalTargetRegistry, "prepare_execution_profile", observed_prepare
                )
                auth_calls.clear()
                headers = {**_AUTH_HEADERS, "Idempotency-Key": "run-admission"}
                path = f"{prefix}/evals/runs"
                admitted = await client.post(path, headers=headers, json=body)
                assert admitted.status_code == 202
                assert admitted.headers["cache-control"] == "private, no-store"
                assert auth_calls == ([] if trusted_local else [path])
                assert events == [
                    "load_corpus",
                    "load_run_by_idempotency_key",
                    "published-profile",
                    "effective-profile",
                    "published-profile",
                    "admit_run",
                ]
                assert len(admitted_requests) == 1
                request = admitted_requests[0]
                expected_origin = (
                    {"trust": "server_verified", "subject": "local-developer", "tenant": None}
                    if trusted_local
                    else {"trust": "server_verified", "subject": "operator", "tenant": "tenant-one"}
                )
                invocation = admitted.json()["spec"]["invocation"]
                assert invocation["source"] == "http_run"
                assert invocation["origin"] == expected_origin
                assert invocation["max_steps"] == 1
                assert invocation["limits"]["max_total_tokens"] == 100
                assert (
                    invocation["execution_profile"]["profile_revision"]
                    == invocation["execution_profile_snapshot"]["revision"]
                )
                # These bytes are durable retry contracts, including the domains.
                expected_digest = (
                    "sha256:"
                    + hashlib.sha256(
                        b"cayu-server-eval-idempotency-v1\0refund-agent\0run-admission"
                    ).hexdigest()
                )
                assert request.idempotency_key == expected_digest
                material = {
                    "kind": "corpus",
                    "target_key": target.key,
                    "resource_identity": {"corpus_revision": corpus.revision},
                    "request": EvalRunCreateRequest.model_validate(body).model_dump(mode="json"),
                    "invocation_provenance": {"source": "http_run", "origin": expected_origin},
                }
                expected_revision = (
                    "sha256:"
                    + hashlib.sha256(
                        b"cayu-eval-admission-request-v1\0"
                        + json.dumps(
                            material, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                        ).encode("utf-8")
                    ).hexdigest()
                )
                assert invocation["admission_request_revision"] == expected_revision

                async def unavailable_profile(*args, **kwargs):
                    pytest.fail("An accepted retry must not prepare an execution profile.")

                monkeypatch.setattr(
                    EvalTargetRegistry, "prepare_execution_profile", unavailable_profile
                )
                events.clear()
                replayed = await client.post(path, headers=headers, json=body)
                assert replayed.status_code == 202
                assert replayed.json() == admitted.json()
                assert events == ["load_corpus", "load_run_by_idempotency_key"]
                conflicts = [{**body, "max_steps": 2}, {**body, "max_concurrency": 2}]
                for changed in conflicts:
                    events.clear()
                    rejected = await client.post(path, headers=headers, json=changed)
                    assert rejected.status_code == 409
                    assert rejected.json() == {
                        "detail": "Idempotency-Key is already bound to another eval run request."
                    }
                    assert events == ["load_corpus", "load_run_by_idempotency_key"]
                if not trusted_local:
                    identity[0] = AuthContext(subject="another-operator", tenant="tenant-one")
                    rejected = await client.post(path, headers=headers, json=body)
                    assert rejected.status_code == 409
                    identity[0] = AuthContext(subject="operator", tenant="another-tenant")
                    rejected = await client.post(path, headers=headers, json=body)
                    assert rejected.status_code == 409
                assert len(admitted_requests) == 1
                assert len((await store.list_runs(EvalRunQuery(target_key=target.key))).items) == 1
                assert provider.requests == []

    asyncio.run(exercise())


@pytest.mark.parametrize("phase", ["published", "effective", "recheck"])
@pytest.mark.parametrize("failure", ["unavailable", "identity_changed"])
def test_admission_profile_failures_remain_private_and_precede_persistence(
    sqlite_resources, monkeypatch, phase, failure
):
    async def exercise():
        async with sqlite_resources as resources:
            provider = _provider(trials=1)
            target = _target(provider)
            store = resources.own(SQLiteEvalStore(resources.path()))
            corpus = _corpus(trials=1)
            await store.save_corpus(corpus, redact_json=target.app.redact_json)
            server = _server(target, store)
            calls = []
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                body = await _launch_body(client, corpus)
                prepare = EvalTargetRegistry.prepare_execution_profile
                fail_at = {"published": 1, "effective": 2, "recheck": 3}[phase]

                async def fail_profile(registry, target_key, *, effective_target=None):
                    calls.append(effective_target is not None)
                    if len(calls) == fail_at:
                        error = (
                            EvalExecutionProfileChangedError
                            if failure == "identity_changed"
                            else RuntimeError
                        )
                        raise error("private provider and deployment details")
                    return await prepare(registry, target_key, effective_target=effective_target)

                async def unexpected_admission(*args, **kwargs):
                    pytest.fail("Failed preparation must not persist a run.")

                monkeypatch.setattr(EvalTargetRegistry, "prepare_execution_profile", fail_profile)
                monkeypatch.setattr(store, "admit_run", unexpected_admission)
                response = await client.post(
                    "/api/evals/runs",
                    headers={**_AUTH_HEADERS, "Idempotency-Key": "profile-failure"},
                    json=body,
                )
                assert response.status_code == 409
                if failure == "identity_changed":
                    detail = "The application identity for this eval execution profile changed after the target was published. Refresh the deployment before launching."
                elif phase == "published":
                    detail = "The current eval execution profile is unavailable."
                else:
                    detail = "The exact current eval execution profile is unavailable."
                assert response.json() == {"detail": detail}
                assert response.headers["cache-control"] == "private, no-store"
                assert calls == [False, True, False][:fail_at]
                assert (await store.list_runs(EvalRunQuery(target_key=target.key))).items == ()
                assert provider.requests == []

    asyncio.run(exercise())


@pytest.mark.parametrize("failure", ["conflict", "unsafe", "checkpointing"])
def test_admission_storage_errors_preserve_private_responses_and_redaction(
    sqlite_resources, monkeypatch, failure
):
    async def exercise():
        async with sqlite_resources as resources:
            provider = _provider(trials=1)
            target = _target(provider)
            store = resources.own(SQLiteEvalStore(resources.path()))
            corpus = _corpus(trials=1)
            await store.save_corpus(corpus, redact_json=target.app.redact_json)
            server = _server(target, store)
            calls = []
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                body = await _launch_body(client, corpus)

                async def fail_admission(request, *, redact_json):
                    calls.append(request)
                    assert redact_json == target.app.redact_json
                    assert request.invocation.execution_profile is not None
                    assert request.invocation.admission_request_revision is not None
                    assert request.target_key == target.key
                    if failure == "checkpointing":
                        pytest.fail("An unavailable checkpoint store must reject before admission.")
                    error = (
                        EvalRunAdmissionConflict
                        if failure == "conflict"
                        else EvalStorePublicationRejected
                    )
                    raise error("private durable request data")

                monkeypatch.setattr(store, "admit_run", fail_admission)
                if failure == "checkpointing":
                    monkeypatch.setattr(store, "trial_checkpointing", False)
                response = await client.post(
                    "/api/evals/runs",
                    headers={**_AUTH_HEADERS, "Idempotency-Key": "admission-failure"},
                    json=body,
                )
                status, detail = {
                    "conflict": (
                        409,
                        "Idempotency-Key is already bound to another eval run request.",
                    ),
                    "unsafe": (422, "Eval run request contains unsafe public data."),
                    "checkpointing": (
                        409,
                        "Restart-safe eval trial checkpointing is not available.",
                    ),
                }[failure]
                assert response.status_code == status
                assert response.json() == {"detail": detail}
                assert response.headers["cache-control"] == "private, no-store"
                assert len(calls) == (0 if failure == "checkpointing" else 1)
                assert (await store.list_runs(EvalRunQuery(target_key=target.key))).items == ()
                assert provider.requests == []

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "failure", ["trials", "suite_concurrency", "unrunnable", "blank_key", "cost_budget"]
)
def test_admission_rejects_invalid_work_without_persisting_a_run(
    sqlite_resources, monkeypatch, failure
):
    async def exercise():
        async with sqlite_resources as resources:
            provider = _provider(trials=1)
            target = _target(provider)
            store = resources.own(SQLiteEvalStore(resources.path()))
            corpus = _corpus(trials=2 if failure == "trials" else 1)
            if failure == "unrunnable":
                case = corpus.cases[0]
                corpus = EvalCorpusDocument.create(
                    target_key=corpus.target_key,
                    evidence_policy=corpus.evidence_policy,
                    suites=corpus.suites,
                    cases=(
                        EvalCaseSpec.create(
                            id=case.id,
                            suite_id=case.suite_id,
                            name=case.name,
                            source=case.source,
                            input=None,
                            assertions=case.assertions,
                        ),
                    ),
                )
            await store.save_corpus(corpus, redact_json=target.app.redact_json)
            server = _server(target, store)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                body = await _launch_body(client, corpus)
                if failure == "suite_concurrency":
                    body["max_concurrency"] = 2
                elif failure == "cost_budget":
                    body["cost_budget"] = {"max_estimated_cost": "1", "currency": "USD"}

                async def unexpected_admission(*args, **kwargs):
                    pytest.fail("Invalid work must not reach durable admission.")

                monkeypatch.setattr(store, "admit_run", unexpected_admission)
                response = await client.post(
                    "/api/evals/runs",
                    headers={
                        **_AUTH_HEADERS,
                        "Idempotency-Key": "   " if failure == "blank_key" else "invalid-work",
                    },
                    json=body,
                )
                status, detail = {
                    "trials": (
                        400,
                        "Eval run exceeds the published execution-profile trial limit.",
                    ),
                    "suite_concurrency": (
                        400,
                        "Eval run exceeds the immutable suite concurrency policy.",
                    ),
                    "unrunnable": (
                        409,
                        "This captured evaluation has no runnable input. Author runnable input or a scenario before launching fresh work.",
                    ),
                    "blank_key": (400, "Invalid Idempotency-Key."),
                    "cost_budget": (
                        400,
                        "Eval run is incompatible with the attached target or bounds.",
                    ),
                }[failure]
                assert response.status_code == status
                assert response.json() == {"detail": detail}
                assert response.headers["cache-control"] == "private, no-store"
                assert (await store.list_runs(EvalRunQuery(target_key=target.key))).items == ()
                assert provider.requests == []

    asyncio.run(exercise())
