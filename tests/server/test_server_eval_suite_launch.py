from __future__ import annotations

import asyncio
import hashlib

import httpx
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi import HTTPException, Request
from fastapi.routing import APIRoute
from tests.server.test_server_eval_scenarios import _AUTH_HEADERS, _authenticate, _scenario, _target
from tests.server.test_server_eval_suite_authoring import _simple_case

from cayu.evals.corpus import RootStatusAssertionSpec
from cayu.evals.execution_profiles import EvalExecutionProfilePolicyV1
from cayu.evals.scenario import EvalScenarioDocumentV2
from cayu.evals.store import (
    EvalCorpusConflict,
    EvalRunQuery,
    EvalStorePublicationRejected,
    EvalStoreResultTooLarge,
)
from cayu.evals.suite_authoring import (
    EvalCaseDraftV2,
    EvalScenarioStimulusV1,
    EvalSuiteDraftV3,
    EvalSuiteTrialRequestDraftV3,
    compile_eval_suite_authoring_draft,
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


async def _saved_suite(resources, *, scenario_count=1, include_simple=True):
    target, _, provider = _target(resources.path("target"))
    store = resources.own(SQLiteEvalStore(resources.path()))
    cases = (
        [EvalCaseDraftV2.model_validate(_simple_case().model_dump(mode="python"))]
        if include_simple
        else []
    )
    template = _scenario()
    for index in range(scenario_count):
        scenario = EvalScenarioDocumentV2.create(
            id=f"scenario-{index:02d}",
            target_key=target.key,
            name=f"Scenario {index}",
            events=template.events,
        )
        await store.save_scenario(scenario, redact_json=target.app.redact_json)
        cases.append(
            EvalCaseDraftV2(
                id=f"scenario-{index:02d}",
                name=f"Scenario {index}",
                stimulus=EvalScenarioStimulusV1(
                    scenario_id=scenario.id, scenario_revision=scenario.revision
                ),
                assertions=(RootStatusAssertionSpec(id="completed", expected="completed"),),
            )
        )
    suite = compile_eval_suite_authoring_draft(
        EvalSuiteDraftV3(
            id="launch-regressions",
            target_key=target.key,
            name="Launch regressions",
            trial_request=EvalSuiteTrialRequestDraftV3(
                trials=1, minimum_passed_trials=1, max_concurrency=2, timeout_seconds=300
            ),
            cases=tuple(cases),
        )
    )
    await store.save_authored_suite(suite, redact_json=target.app.redact_json)
    return target, store, suite, provider


def _server(target, store, *, prefix="/api", authenticate=_authenticate):
    return create_server(
        target.app,
        config=ServerConfig.protected(
            authenticate,
            dashboard=DashboardConfig(enabled=False),
            api=ServerApiConfig(path=prefix),
            evals=EvalsConfig(
                target=target,
                store=store,
                execution_profile_policy=EvalExecutionProfilePolicyV1(
                    reset_strategy="application_managed",
                    isolation_revision="sha256:" + "a" * 64,
                    max_concurrency=2,
                ),
            ),
        ),
    )


async def _reviewed_body(client, root):
    response = await client.post(f"{root}/preview", headers=_AUTH_HEADERS, json={})
    assert response.status_code == 200
    preview = response.json()
    assert preview["ready"] is True
    assert preview["diagnostics"] == []
    return {
        "expected_exposure_revision": preview["exposure"]["revision"],
        "expected_execution_profiles": [
            {
                "case_ids": item["case_ids"],
                "execution_profile_revision": item["execution_profile_revision"],
            }
            for item in preview["launches"]
        ],
    }


@pytest.mark.parametrize("prefix", ["/api", "/custom/v2"])
def test_suite_launch_preserves_shared_auth_body_handling_and_provenance(sqlite_resources, prefix):
    async def exercise():
        async with sqlite_resources as resources:
            target, store, suite, provider = await _saved_suite(resources)
            calls = []

            async def authenticate(request: Request):
                calls.append(request.url.path)
                _authenticate(request)
                await request.body()
                return AuthContext(subject="suite-operator", tenant="suite-tenant")

            server = _server(target, store, prefix=prefix, authenticate=authenticate)
            root = f"{prefix}/evals/suites/{suite.revision}/runs"
            routes = [r for r in server.routes if isinstance(r, APIRoute)]
            endpoints = [
                r
                for r in routes
                if r.path
                in (
                    f"{prefix}/evals/suites/{{suite_revision}}/runs/preview",
                    f"{prefix}/evals/suites/{{suite_revision}}/runs",
                )
            ]
            assert [r.name for r in endpoints] == [
                "preview_eval_authored_suite_run",
                "launch_eval_authored_suite_run",
            ]
            dependency = (
                next(r for r in routes if r.path == f"{prefix}/sessions").dependencies[0].dependency
            )
            for route in endpoints:
                assert route.dependencies[0].dependency is dependency
                assert isinstance(route, routes_module._BoundedEvalsRoute)
                assert type(route).preparse_auth is authenticate
            # Admission is exercised without starting a worker, so replay observes
            # unchanged queued records and cannot consume scripted model responses.
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                body = await _reviewed_body(client, root)
                assert calls == [f"{root}/preview"]
                for path, payload, status in ((f"{root}/preview", {}, 200), (root, body, 202)):
                    calls.clear()
                    denied = await client.post(
                        path,
                        content=b"private invalid input",
                        headers={"Content-Length": str(endpoints[0].max_request_bytes + 1)},
                    )
                    assert denied.status_code == 401
                    assert denied.json() == {"detail": "unauthorized"}
                    assert denied.headers["cache-control"] == "private, no-store"
                    assert calls == [path]
                    calls.clear()
                    response = await client.post(
                        path,
                        headers={**_AUTH_HEADERS, "Idempotency-Key": "suite-boundary"},
                        json=payload,
                    )
                    assert response.status_code == status
                    assert response.headers["cache-control"] == "private, no-store"
                    assert calls == [path]
                    if status == 202:
                        for item in response.json()["runs"]:
                            assert item["run"]["spec"]["invocation"]["origin"] == {
                                "trust": "server_verified",
                                "subject": "suite-operator",
                                "tenant": "suite-tenant",
                            }
                    for content, extra_headers, expected_status, detail in (
                        (
                            b"{}",
                            {"Content-Length": str(endpoints[0].max_request_bytes + 1)},
                            413,
                            "Evals request exceeds the server byte limit.",
                        ),
                        (
                            b'{"private":"secret","private":"other"}',
                            {},
                            422,
                            "Invalid Evals request.",
                        ),
                        (b"NaN", {}, 422, "Invalid Evals request."),
                    ):
                        calls.clear()
                        rejected = await client.post(
                            path,
                            content=content,
                            headers={
                                **_AUTH_HEADERS,
                                "Idempotency-Key": "suite-boundary",
                                "Content-Type": "application/json",
                                **extra_headers,
                            },
                        )
                        assert rejected.status_code == expected_status
                        assert rejected.json() == {"detail": detail}
                        assert rejected.headers["cache-control"] == "private, no-store"
                        assert calls == [path]

                def deny_access():
                    raise HTTPException(status_code=403, detail="suite launch access revoked")

                server.dependency_overrides[dependency] = deny_access
                for path, payload in ((f"{root}/preview", {}), (root, body)):
                    response = await client.post(
                        path,
                        headers={**_AUTH_HEADERS, "Idempotency-Key": "suite-boundary"},
                        json=payload,
                    )
                    assert response.status_code == 403
                    assert response.json() == {"detail": "suite launch access revoked"}
                    assert response.headers["cache-control"] == "private, no-store"
            assert provider.requests == []

    asyncio.run(exercise())


def test_suite_launch_prepares_and_publishes_all_parts_before_ordered_admission(
    sqlite_resources, monkeypatch
):
    async def exercise():
        async with sqlite_resources as resources:
            target, store, suite, provider = await _saved_suite(resources, scenario_count=2)
            server = _server(target, store)
            root = f"/api/evals/suites/{suite.revision}/runs"
            events = []
            requests = []
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                body = await _reviewed_body(client, root)
                prepare = EvalTargetRegistry.prepare_execution_profile
                save = store.save_corpus
                admit = store.admit_run

                async def observe_prepare(registry, *args, **kwargs):
                    result = await prepare(registry, *args, **kwargs)
                    events.append("prepared")
                    return result

                async def observe_save(corpus, *, redact_json):
                    assert redact_json == target.app.redact_json
                    result = await save(corpus, redact_json=redact_json)
                    events.append("published")
                    return result

                async def observe_admit(request, *, redact_json):
                    assert redact_json == target.app.redact_json
                    events.append("admitted")
                    requests.append(request)
                    return await admit(request, redact_json=redact_json)

                monkeypatch.setattr(
                    EvalTargetRegistry, "prepare_execution_profile", observe_prepare
                )
                monkeypatch.setattr(store, "save_corpus", observe_save)
                monkeypatch.setattr(store, "admit_run", observe_admit)
                response = await client.post(
                    root,
                    headers={**_AUTH_HEADERS, "Idempotency-Key": "multi-launch"},
                    json=body,
                )
                assert response.status_code == 202
                assert events.count("published") == events.count("admitted") == 3
                assert (
                    events
                    == ["prepared"] * events.count("prepared")
                    + ["published"] * 3
                    + ["admitted"] * 3
                )
                assert events.count("prepared") > 0
                runs = response.json()["runs"]
                assert [item["case_ids"] for item in runs] == [
                    ["refund-request"],
                    ["scenario-00"],
                    ["scenario-01"],
                ]
                assert [request.invocation.authored_suite_launch_lane for request in requests] == [
                    0,
                    1,
                    0,
                ]
                assert [request.max_concurrency for request in requests] == [1, 1, 1]
                for index, request in enumerate(requests):
                    namespace = f"authored-suite-part-{index + 1}".encode("ascii")
                    digest = hashlib.sha256(
                        b"cayu-server-eval-idempotency-v1\0internal\0"
                        + namespace
                        + b"\0assistant.default\0multi-launch"
                    ).hexdigest()
                    assert request.idempotency_key == "sha256:" + digest
                    assert request.invocation.authored_suite_revision == suite.revision
                    assert (
                        request.invocation.authored_suite_exposure.revision
                        == body["expected_exposure_revision"]
                    )
                assert len((await store.list_runs(EvalRunQuery(target_key=target.key))).items) == 3
                assert provider.requests == []

    asyncio.run(exercise())


@pytest.mark.parametrize("failure", ["conflict", "unsafe", "oversized"])
def test_suite_launch_rejects_corpus_publication_before_admitting_any_part(
    sqlite_resources, monkeypatch, failure
):
    async def exercise():
        async with sqlite_resources as resources:
            target, store, suite, provider = await _saved_suite(resources)
            server = _server(target, store)
            root = f"/api/evals/suites/{suite.revision}/runs"
            calls = []
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                body = await _reviewed_body(client, root)
                save = store.save_corpus

                async def fail_second_save(corpus, *, redact_json):
                    calls.append(corpus.revision)
                    assert redact_json == target.app.redact_json
                    if len(calls) == 2:
                        if failure == "oversized":
                            raise EvalStoreResultTooLarge(1024)
                        error = (
                            EvalCorpusConflict
                            if failure == "conflict"
                            else EvalStorePublicationRejected
                        )
                        raise error("private corpus storage detail")
                    return await save(corpus, redact_json=redact_json)

                async def unexpected_admission(*args, **kwargs):
                    pytest.fail("Every corpus must be published before admitting any suite part.")

                monkeypatch.setattr(store, "save_corpus", fail_second_save)
                monkeypatch.setattr(store, "admit_run", unexpected_admission)
                response = await client.post(
                    root,
                    headers={**_AUTH_HEADERS, "Idempotency-Key": "failed-publication"},
                    json=body,
                )
                status, detail = {
                    "conflict": (
                        409,
                        "Derived authored-suite corpus conflicts with stored content.",
                    ),
                    "unsafe": (422, "Derived authored-suite corpus contains unsafe public data."),
                    "oversized": (
                        413,
                        "Derived authored-suite corpus exceeds the server byte limit.",
                    ),
                }[failure]
                assert response.status_code == status
                assert response.json() == {"detail": detail}
                assert response.headers["cache-control"] == "private, no-store"
                assert len(calls) == 2
                assert (await store.list_runs(EvalRunQuery(target_key=target.key))).items == ()
                assert provider.requests == []

    asyncio.run(exercise())


def test_suite_launch_recovers_partial_admission_and_replays_completed_parts(
    sqlite_resources, monkeypatch
):
    async def exercise():
        async with sqlite_resources as resources:
            target, store, suite, provider = await _saved_suite(resources)
            server = _server(target, store)
            root = f"/api/evals/suites/{suite.revision}/runs"
            headers = {**_AUTH_HEADERS, "Idempotency-Key": "partial-launch"}
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                body = await _reviewed_body(client, root)
                admit = store.admit_run
                calls = []

                async def fail_second_admission(request, *, redact_json):
                    calls.append(request)
                    if len(calls) == 2:
                        raise EvalStorePublicationRejected("private second-part detail")
                    return await admit(request, redact_json=redact_json)

                monkeypatch.setattr(store, "admit_run", fail_second_admission)
                response = await client.post(root, headers=headers, json=body)
                assert response.status_code == 422
                assert response.json() == {
                    "detail": "Eval run request contains unsafe public data."
                }
                first = (await store.list_runs(EvalRunQuery(target_key=target.key))).items
                assert len(first) == 1
                first_run_id = first[0].spec.run_id

                async def unavailable_profile(*args, **kwargs):
                    raise RuntimeError("private current provider configuration")

                with monkeypatch.context() as unavailable:
                    unavailable.setattr(
                        EvalTargetRegistry, "prepare_execution_profile", unavailable_profile
                    )
                    response = await client.post(root, headers=headers, json=body)
                    assert response.status_code == 503
                    assert response.json() == {
                        "detail": "An earlier authored eval launch attempt admitted only part of this request. Restore current launch readiness and retry with the same Idempotency-Key."
                    }
                    assert response.headers["cache-control"] == "private, no-store"
                    conflict = await client.post(
                        root, headers=headers, json={**body, "case_ids": ["refund-request"]}
                    )
                    assert conflict.status_code == 409
                    assert conflict.json() == {
                        "detail": "Idempotency-Key is already bound to another eval run request."
                    }
                    assert len(calls) == 2
                restored = await client.post(root, headers=headers, json=body)
                assert restored.status_code == 202
                restored_ids = [item["run"]["spec"]["run_id"] for item in restored.json()["runs"]]
                assert len(set(restored_ids)) == 2
                assert restored_ids[0] == first_run_id
                assert len(calls) == 4
                monkeypatch.setattr(
                    EvalTargetRegistry, "prepare_execution_profile", unavailable_profile
                )
                replayed = await client.post(root, headers=headers, json=body)
                assert replayed.status_code == 202
                assert replayed.json() == restored.json()
                assert len(calls) == 4
                assert len((await store.list_runs(EvalRunQuery(target_key=target.key))).items) == 2
                assert provider.requests == []

    asyncio.run(exercise())


@pytest.mark.parametrize("cancel", [False, True])
def test_suite_launch_preview_bounds_scenario_preparation_and_cleans_up_cancellation(
    sqlite_resources, monkeypatch, cancel
):
    async def exercise():
        async with sqlite_resources as resources:
            target, store, suite, provider = await _saved_suite(
                resources, scenario_count=17, include_simple=False
            )
            server = _server(target, store)
            root = f"/api/evals/suites/{suite.revision}/runs"
            entered = asyncio.Event()
            release = asyncio.Event()
            reads = []
            active = 0
            peak = 0
            load = store.load_scenario

            async def blocked_load(revision):
                nonlocal active, peak
                reads.append(revision)
                active += 1
                peak = max(peak, active)
                if active == 16:
                    entered.set()
                try:
                    await release.wait()
                    return await load(revision)
                finally:
                    active -= 1

            monkeypatch.setattr(store, "load_scenario", blocked_load)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                request = resources.task(
                    client.post(f"{root}/preview", headers=_AUTH_HEADERS, json={})
                )
                try:
                    await asyncio.wait_for(entered.wait(), 5)
                    assert len(reads) == active == 16
                    assert not request.done()
                    if cancel:
                        request.cancel()
                        with pytest.raises(asyncio.CancelledError):
                            await request
                        assert len(reads) == 16
                    else:
                        release.set()
                        response = await asyncio.wait_for(request, 15)
                        assert response.status_code == 200
                        assert response.json()["ready"] is True
                        assert [item["case_ids"] for item in response.json()["launches"]] == [
                            [case.id] for case in suite.cases
                        ]
                        assert len(reads) == len(set(reads)) == 17
                    assert peak == 16
                    assert active == 0
                finally:
                    release.set()
                    await asyncio.gather(request, return_exceptions=True)
            assert (await store.list_runs(EvalRunQuery(target_key=target.key))).items == ()
            assert provider.requests == []

    asyncio.run(exercise())
