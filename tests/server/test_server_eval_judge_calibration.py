from __future__ import annotations

import asyncio

import httpx
import pytest
from tests.evals.test_structured_model_judge import _judgment, _rubric, _target
from tests.server.test_server_eval_scenarios import _AUTH_HEADERS, _authenticate, _server

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi import HTTPException, Request
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from cayu import (
    AgentSpec,
    CayuApp,
    CorpusUserMessageSpec,
    EvalCaseDraftV2,
    EvalJudgeCalibrationCriterionLabelV1,
    EvalJudgeCalibrationDraftV1,
    EvalJudgeEvidenceSelectionV1,
    EvalSimpleInputStimulusV1,
    EvalSuiteDraftV2,
    ModelJudgeTarget,
    ModelStreamEvent,
    PublicJudgeReferenceDraftV1,
    RunInputSpec,
    ScriptedModelProvider,
    StructuredModelJudgeAssertionDraftV1,
    StructuredModelJudgeAssertionSpec,
    StructuredRubricDraftV1,
    model_judge_profile,
)
from cayu.evals.calibration import compile_eval_judge_calibration_draft
from cayu.evals.store import EvalStoreResultTooLarge
from cayu.server import (
    AuthContext,
    DashboardConfig,
    EvalsConfig,
    ServerApiConfig,
    ServerConfig,
    create_server,
)
from cayu.server import routes as routes_module
from cayu.storage.evals_sqlite import SQLiteEvalStore


def _judge_with_trials(count: int) -> tuple[ModelJudgeTarget, ScriptedModelProvider]:
    script = (
        ModelStreamEvent.text_delta(_judgment()),
        ModelStreamEvent.completed(
            {
                "finish_reason": "stop",
                "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
            }
        ),
    )
    provider = ScriptedModelProvider([script for _ in range(count)])
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="judge", model="judge-model"))
    return (
        ModelJudgeTarget(
            key="quality-judge",
            label="Quality judge",
            app=app,
            agent_name="judge",
        ),
        provider,
    )


def _draft(judge: ModelJudgeTarget, *, trials: int = 2) -> EvalJudgeCalibrationDraftV1:
    profile = model_judge_profile(judge)
    return EvalJudgeCalibrationDraftV1(
        id="known-refund-answer",
        target_key="refund-agent",
        assertion=StructuredModelJudgeAssertionSpec(
            id="answer-quality",
            judge_profile_key=profile.key,
            judge_profile_revision=profile.revision,
            rubric=_rubric(),
            threshold="0.6",
            evidence=EvalJudgeEvidenceSelectionV1(),
        ),
        evidence_source_id="reviewed-refund-fixture",
        task="Can I get a refund?",
        final_output="Refunds are available within 30 days.",
        human_criteria=(
            EvalJudgeCalibrationCriterionLabelV1(
                criterion_id="correctness",
                score="1",
            ),
            EvalJudgeCalibrationCriterionLabelV1(
                criterion_id="usefulness",
                score="0.5",
            ),
        ),
        trials=trials,
    )


def test_preview_and_run_calibrate_fixed_evidence_without_candidate_execution(tmp_path) -> None:
    judge, judge_provider = _judge_with_trials(2)
    target, candidate_provider = _target(judge)
    path = tmp_path / "evals.db"
    store = SQLiteEvalStore(path)
    draft = _draft(judge)
    try:
        with TestClient(_server(target, store)) as client:
            assert (
                client.post(
                    "/api/evals/judge-calibrations/preview",
                    json={"draft": draft.model_dump(mode="json")},
                ).status_code
                == 401
            )
            preview = client.post(
                "/api/evals/judge-calibrations/preview",
                headers=_AUTH_HEADERS,
                json={"draft": draft.model_dump(mode="json")},
            )
            assert preview.status_code == 200
            previewed = preview.json()
            assert previewed["ready"] is True
            assert previewed["diagnostics"] == []
            assert previewed["work"]["judge_calls"] == 2
            assert previewed["candidate_route_relation"] == "independent_model"
            assert previewed["definition"]["evidence"]["provenance"] == {
                "schema_version": 1,
                "kind": "operator_supplied",
                "source_id": "reviewed-refund-fixture",
            }
            assert judge_provider.requests == []
            assert candidate_provider.requests == []

            run = client.post(
                "/api/evals/judge-calibrations",
                headers=_AUTH_HEADERS,
                json={
                    "run_id": "calibration-fixed-answer",
                    "expected_definition_revision": previewed["definition"]["revision"],
                    "definition": previewed["definition"],
                },
            )
            assert run.status_code == 201
            report = run.json()["report"]
            assert len(report["trials"]) == 2
            assert all(item["pass_agreement"] is True for item in report["trials"])
            assert len(judge_provider.requests) == 2
            assert candidate_provider.requests == []

            loaded = client.get(
                f"/api/evals/judge-calibrations/{report['revision']}",
                headers=_AUTH_HEADERS,
            )
            assert loaded.status_code == 200
            assert loaded.json() == report

            changed_draft = draft.model_copy(
                update={"final_output": "Refunds are never available."}
            )
            changed_preview = client.post(
                "/api/evals/judge-calibrations/preview",
                headers=_AUTH_HEADERS,
                json={"draft": changed_draft.model_dump(mode="json")},
            )
            assert changed_preview.status_code == 200
            changed_definition = changed_preview.json()["definition"]
            conflict = client.post(
                "/api/evals/judge-calibrations",
                headers=_AUTH_HEADERS,
                json={
                    "run_id": "calibration-fixed-answer",
                    "expected_definition_revision": changed_definition["revision"],
                    "definition": changed_definition,
                },
            )
            assert conflict.status_code == 409
            assert conflict.json() == {
                "detail": "Judge calibration run ID is bound to different reviewed input."
            }
            assert len(judge_provider.requests) == 2
            assert candidate_provider.requests == []

        reopened = SQLiteEvalStore(path)
        try:
            with TestClient(_server(target, reopened)) as restarted_client:
                recovered = restarted_client.post(
                    "/api/evals/judge-calibrations",
                    headers=_AUTH_HEADERS,
                    json={
                        "run_id": "calibration-fixed-answer",
                        "expected_definition_revision": previewed["definition"]["revision"],
                        "definition": previewed["definition"],
                    },
                )
                assert recovered.status_code == 201
                assert recovered.json()["report"] == report
                assert len(judge_provider.requests) == 2
                assert candidate_provider.requests == []
        finally:
            asyncio.run(reopened.close())
    finally:
        asyncio.run(store.close())


def test_structured_suite_preview_save_and_reload_share_the_server_compiled_contract(
    tmp_path,
) -> None:
    judge, judge_provider = _judge_with_trials(1)
    target, candidate_provider = _target(judge)
    profile = model_judge_profile(judge)
    assertion = StructuredModelJudgeAssertionDraftV1(
        id="answer-quality",
        judge_profile_key=profile.key,
        judge_profile_revision=profile.revision,
        rubric=StructuredRubricDraftV1.from_rubric(_rubric()),
        reference=PublicJudgeReferenceDraftV1(
            id="refund-policy",
            expected_answer="Refunds are available within 30 days.",
        ),
        threshold="0.6",
    )
    draft = EvalSuiteDraftV2(
        id="refund-quality",
        target_key=target.key,
        name="Refund quality",
        cases=(
            EvalCaseDraftV2(
                id="refund-answer",
                name="Refund answer",
                stimulus=EvalSimpleInputStimulusV1(
                    input=RunInputSpec(
                        messages=(CorpusUserMessageSpec(text="Can I get a refund?"),)
                    )
                ),
                assertions=(assertion,),
            ),
        ),
    )
    store = SQLiteEvalStore(tmp_path / "evals.db")
    try:
        with TestClient(_server(target, store)) as client:
            preview = client.post(
                "/api/evals/suites/preview",
                headers=_AUTH_HEADERS,
                json={"draft": draft.model_dump(mode="json")},
            )
            assert preview.status_code == 200
            body = preview.json()
            assert body["ready"] is True
            compiled = body["suite"]["cases"][0]["assertions"][0]
            assert compiled["rubric"]["revision"].startswith("sha256:")
            assert compiled["reference"]["revision"].startswith("sha256:")

            saved = client.post(
                "/api/evals/suites",
                headers=_AUTH_HEADERS,
                json={
                    "expected_suite_revision": body["suite"]["revision"],
                    "suite": body["suite"],
                },
            )
            assert saved.status_code == 201
            assert saved.json()["suite"] == body["suite"]

            loaded = client.get(
                f"/api/evals/suites/{body['suite']['revision']}",
                headers=_AUTH_HEADERS,
            )
            assert loaded.status_code == 200
            assert loaded.json() == body["suite"]
        assert judge_provider.requests == []
        assert candidate_provider.requests == []
    finally:
        asyncio.run(store.close())


def _calibration_run_body(draft: EvalJudgeCalibrationDraftV1) -> dict:
    definition = compile_eval_judge_calibration_draft(draft)
    return {
        "run_id": "calibration-concurrent",
        "expected_definition_revision": definition.revision,
        "definition": definition.model_dump(mode="json"),
    }


@pytest.mark.parametrize("prefix", ["/api", "/custom/v2"])
def test_calibration_routes_preserve_auth_and_private_http_boundary(sqlite_resources, prefix):
    async def scenario():
        async with sqlite_resources as resources:
            judge, judge_provider = _judge_with_trials(1)
            target, candidate_provider = _target(judge)
            store = resources.own(SQLiteEvalStore(resources.path()))
            calls = []

            def authenticate(request: Request) -> AuthContext:
                calls.append(request.url.path)
                return _authenticate(request)

            server = create_server(
                target.app,
                config=ServerConfig.protected(
                    authenticate,
                    api=ServerApiConfig(path=prefix),
                    dashboard=DashboardConfig(enabled=False),
                    evals=EvalsConfig(target=target, store=store),
                ),
            )
            routes = [route for route in server.routes if isinstance(route, APIRoute)]
            calibration = [route for route in routes if "/judge-calibrations" in route.path]
            root = f"{prefix}/evals/judge-calibrations"
            assert [route.path for route in calibration] == [
                f"{root}/preview",
                root,
                f"{root}/{{calibration_revision}}",
            ]
            dependency = (
                next(route for route in routes if route.path == f"{prefix}/sessions")
                .dependencies[0]
                .dependency
            )
            for route in calibration:
                assert route.dependencies[0].dependency is dependency
                assert isinstance(route, routes_module._BoundedEvalsRoute)
                assert type(route).preparse_auth is authenticate

            async with (
                server.router.lifespan_context(server),
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=server), base_url="http://test"
                ) as client,
            ):
                for path in (f"{root}/preview", root):
                    calls.clear()
                    denied = await client.post(
                        path,
                        content=b"private invalid input",
                        headers={"Content-Length": str(calibration[0].max_request_bytes + 1)},
                    )
                    assert denied.status_code == 401
                    assert denied.json() == {"detail": "unauthorized"}
                    assert denied.headers["cache-control"] == "private, no-store"
                    assert calls == [path]
                    for body, extra_headers, status, detail in (
                        (
                            b"{}",
                            {"Content-Length": str(calibration[0].max_request_bytes + 1)},
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
                            content=body,
                            headers={
                                **_AUTH_HEADERS,
                                "Content-Type": "application/json",
                                **extra_headers,
                            },
                        )
                        assert rejected.status_code == status
                        assert rejected.json() == {"detail": detail}
                        assert rejected.headers["cache-control"] == "private, no-store"
                        assert calls == [path]

                calls.clear()
                created = await client.post(
                    root, headers=_AUTH_HEADERS, json=_calibration_run_body(_draft(judge, trials=1))
                )
                assert created.status_code == 201
                assert calls == [root]
                report = created.json()["report"]
                loaded = await client.get(f"{root}/{report['revision']}", headers=_AUTH_HEADERS)
                assert loaded.status_code == 200
                assert loaded.json() == report
                assert loaded.headers["cache-control"] == "private, no-store"

                def deny_access():
                    raise HTTPException(status_code=403, detail="calibration access revoked")

                server.dependency_overrides[dependency] = deny_access
                for method, path, payload in (
                    ("POST", f"{root}/preview", {"draft": _draft(judge).model_dump(mode="json")}),
                    ("POST", root, _calibration_run_body(_draft(judge, trials=1))),
                    ("GET", f"{root}/{report['revision']}", None),
                ):
                    denied = await client.request(method, path, headers=_AUTH_HEADERS, json=payload)
                    assert denied.status_code == 403
                    assert denied.json() == {"detail": "calibration access revoked"}
                    assert denied.headers["cache-control"] == "private, no-store"
            assert len(judge_provider.requests) == 1
            assert candidate_provider.requests == []

    asyncio.run(scenario())


@pytest.mark.parametrize("second_request", ["replay", "conflict", "cancel"])
def test_calibration_run_serializes_replay_conflict_and_cancellation(
    sqlite_resources, monkeypatch, second_request
):
    async def scenario():
        async with sqlite_resources as resources:
            judge, judge_provider = _judge_with_trials(1)
            target, candidate_provider = _target(judge)
            store = resources.own(SQLiteEvalStore(resources.path()))
            server = _server(target, store)
            path = "/api/evals/judge-calibrations"
            route = next(r for r in server.routes if isinstance(r, APIRoute) and r.path == path)
            first_read = asyncio.Event()
            release_first = asyncio.Event()
            second_admitted = asyncio.Event()
            reads = []
            original_load = store.load_judge_calibration_by_run_id

            async def gated_load(run_id, *, max_bytes):
                reads.append(run_id)
                if len(reads) == 1:
                    first_read.set()
                    await release_first.wait()
                return await original_load(run_id, max_bytes=max_bytes)

            async def authenticated(request: Request):
                # This dependency runs after private parsing, immediately before
                # synchronous body validation and entry into the async endpoint.
                if request.headers.get("X-Second-Request"):
                    second_admitted.set()
                return _authenticate(request)

            monkeypatch.setattr(store, "load_judge_calibration_by_run_id", gated_load)
            server.dependency_overrides[route.dependencies[0].dependency] = authenticated
            draft = _draft(judge, trials=1)
            first_body = _calibration_run_body(draft)
            second_body = _calibration_run_body(
                draft.model_copy(update={"final_output": "A different reviewed answer."})
                if second_request == "conflict"
                else draft
            )
            async with (
                server.router.lifespan_context(server),
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=server), base_url="http://test"
                ) as client,
            ):
                first = resources.task(client.post(path, headers=_AUTH_HEADERS, json=first_body))
                second = None
                try:
                    await asyncio.wait_for(first_read.wait(), 5)
                    second = resources.task(
                        client.post(
                            path,
                            headers={**_AUTH_HEADERS, "X-Second-Request": "yes"},
                            json=second_body,
                        )
                    )
                    await asyncio.wait_for(second_admitted.wait(), 5)
                    assert reads == [first_body["run_id"]]
                    assert not second.done()
                    if second_request == "cancel":
                        first.cancel()
                        with pytest.raises(asyncio.CancelledError):
                            await first
                    else:
                        release_first.set()
                        result = await asyncio.wait_for(first, 10)
                        assert result.status_code == 201
                    response = await asyncio.wait_for(second, 10)
                    if second_request == "conflict":
                        assert response.status_code == 409
                        assert response.json() == {
                            "detail": "Judge calibration run ID is bound to different reviewed input."
                        }
                    else:
                        assert response.status_code == 201
                        if second_request == "replay":
                            assert response.json() == result.json()
                    assert reads == [first_body["run_id"], first_body["run_id"]]
                    assert len(judge_provider.requests) == 1
                    assert candidate_provider.requests == []
                finally:
                    release_first.set()
                    await asyncio.gather(
                        first, *([second] if second is not None else []), return_exceptions=True
                    )

    asyncio.run(scenario())


def test_calibration_locks_belong_to_each_router(sqlite_resources, monkeypatch):
    async def scenario():
        async with sqlite_resources as resources:
            judge, judge_provider = _judge_with_trials(2)
            target, candidate_provider = _target(judge)
            stores = [
                resources.own(SQLiteEvalStore(resources.path(f"server-{index}.db")))
                for index in range(2)
            ]
            servers = [_server(target, store) for store in stores]
            entered = asyncio.Event()
            release = asyncio.Event()
            original_load = stores[0].load_judge_calibration_by_run_id

            async def gated_load(run_id, *, max_bytes):
                entered.set()
                await release.wait()
                return await original_load(run_id, max_bytes=max_bytes)

            monkeypatch.setattr(stores[0], "load_judge_calibration_by_run_id", gated_load)
            body = _calibration_run_body(_draft(judge, trials=1))
            async with (
                servers[0].router.lifespan_context(servers[0]),
                servers[1].router.lifespan_context(servers[1]),
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=servers[0]), base_url="http://first"
                ) as first_client,
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=servers[1]), base_url="http://second"
                ) as second_client,
            ):
                first = resources.task(
                    first_client.post(
                        "/api/evals/judge-calibrations", headers=_AUTH_HEADERS, json=body
                    )
                )
                try:
                    await asyncio.wait_for(entered.wait(), 5)
                    independent = await asyncio.wait_for(
                        second_client.post(
                            "/api/evals/judge-calibrations", headers=_AUTH_HEADERS, json=body
                        ),
                        10,
                    )
                    assert independent.status_code == 201
                    assert not first.done()
                finally:
                    release.set()
                    await first
                assert first.result().status_code == 201
            assert len(judge_provider.requests) == 2
            assert candidate_provider.requests == []

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["invalid", "oversized", "unsupported"])
def test_calibration_report_preserves_private_storage_errors(
    sqlite_resources, monkeypatch, failure
):
    async def scenario():
        async with sqlite_resources as resources:
            judge, judge_provider = _judge_with_trials(0)
            target, candidate_provider = _target(judge)
            store = resources.own(SQLiteEvalStore(resources.path()))
            reads = []

            async def rejected_load(revision, *, max_bytes):
                reads.append((revision, max_bytes))
                if failure == "invalid":
                    raise ValueError("private store input")
                raise EvalStoreResultTooLarge(max_bytes)

            monkeypatch.setattr(store, "load_judge_calibration", rejected_load)
            if failure == "unsupported":
                monkeypatch.setattr(store, "judge_calibrations", False)
            server = _server(target, store)
            async with (
                server.router.lifespan_context(server),
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=server), base_url="http://test"
                ) as client,
            ):
                response = await client.get(
                    "/api/evals/judge-calibrations/invalid", headers=_AUTH_HEADERS
                )
                status, detail = {
                    "invalid": (422, "Invalid Evals query."),
                    "oversized": (413, "Judge calibration exceeds the server byte limit."),
                    "unsupported": (409, "Durable judge calibration persistence is not available."),
                }[failure]
                assert response.status_code == status
                assert response.json() == {"detail": detail}
                assert response.headers["cache-control"] == "private, no-store"
                if failure == "unsupported":
                    response = await client.post(
                        "/api/evals/judge-calibrations",
                        headers=_AUTH_HEADERS,
                        json=_calibration_run_body(_draft(judge, trials=1)),
                    )
                    assert response.status_code == 409
                    assert response.json() == {"detail": detail}
                    assert reads == []
                else:
                    assert len(reads) == 1
            assert judge_provider.requests == []
            assert candidate_provider.requests == []

    asyncio.run(scenario())
