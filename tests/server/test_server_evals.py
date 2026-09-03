from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from tests.evals.test_corpus_execution import (
    _corpus,
    _model_judge_corpus,
    _model_judge_target,
    _provider,
    _target,
)
from tests.evals.test_memory_reporting import _report_fixture

import cayu.evals.execution as execution_module
import cayu.server.evals_worker as evals_worker_module
import cayu.server.routes as routes_module
import cayu.storage.evals_sqlite as evals_sqlite_module
from cayu import (
    AgentSpec,
    CayuApp,
    CorpusTarget,
    EvalExecutionCapacity,
    EvalExecutionProfilePolicyV1,
    Message,
    ModelJudgeTarget,
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
)
from cayu.evals.corpus import (
    CorpusUserMessageSpec,
    EvalCaseSpec,
    EvalCorpusDocument,
    ModelJudgeAssertionSpec,
    RunInputSpec,
)
from cayu.evals.execution import run_corpus_suite
from cayu.evals.memory_reporting import (
    MemoryExperimentReportRequest,
    build_memory_experiment_report,
)
from cayu.evals.result_presentation import present_eval_result
from cayu.evals.store import (
    EvalRunInvocation,
    EvalRunRequest,
    EvalRunStatus,
    EvalStoreTransientContention,
    InMemoryEvalStore,
)
from cayu.evals.suite_authoring import (
    EvalCaseDraftV1,
    EvalSimpleInputStimulusV1,
    EvalSuiteDraftV1,
)
from cayu.project_control_plane import (
    ProjectControlPlaneAccess,
    _create_project_control_plane_context,
)
from cayu.runtime.invocation import (
    InvocationOrigin,
    InvocationOriginTrust,
    SessionExecutionSource,
)
from cayu.runtime.stop_policy import RunLimits
from cayu.server import (
    AuthContext,
    AuthenticatedAccess,
    DashboardConfig,
    EvalsConfig,
    OpenAccess,
    ServerApiConfig,
    ServerConfig,
    create_server,
    mount_cayu,
)
from cayu.server.contracts import (
    MAX_EVALS_REQUEST_BYTES,
    EvalComparisonRequest,
    EvalResultResponse,
    EvalRunCreateRequest,
)
from cayu.server.evals_registry import explicit_eval_target_registry, target_for_eval_invocation
from cayu.storage.evals_sqlite import SQLiteEvalStore, SQLiteEvalWriterContentionPolicy
from cayu.storage.migrations import SchemaMode

_AUTH_HEADERS = {"Authorization": "Bearer valid"}


def _authenticate(request: Request) -> AuthContext:
    if request.headers.get("Authorization") != "Bearer valid":
        raise HTTPException(status_code=401, detail="unauthorized")
    return AuthContext(subject="eval-operator")


def _evals_config(target, store, **updates) -> EvalsConfig:
    return EvalsConfig(
        target=target,
        store=store,
        poll_interval_seconds=updates.pop("poll_interval_seconds", 0.02),
        lease_seconds=updates.pop("lease_seconds", 5),
        shutdown_grace_seconds=updates.pop("shutdown_grace_seconds", 2.0),
        **updates,
    )


def _repeatable_execution_policy(target) -> EvalExecutionProfilePolicyV1:
    return EvalExecutionProfilePolicyV1(
        fixture_strategy="application_managed",
        reset_strategy="application_managed",
        effect_posture="isolated_application_authority",
        isolation_revision="sha256:" + "1" * 64,
        max_trials=target.limits.max_trials,
        max_concurrency=target.limits.max_concurrency,
    )


async def _bound_eval_invocation(
    target,
    invocation: EvalRunInvocation | None = None,
) -> EvalRunInvocation:
    invocation = EvalRunInvocation() if invocation is None else invocation
    registry = explicit_eval_target_registry(target)
    effective_target = target_for_eval_invocation(target, invocation)
    prepared = await registry.prepare_execution_profile(
        target.key,
        effective_target=effective_target,
    )
    return invocation.model_copy(
        update={
            "execution_profile": prepared.binding,
            "execution_profile_snapshot": prepared.snapshot,
        },
        deep=True,
    )


def _server(target, store, *, execution_profile_policy=None):
    return create_server(
        target.app,
        config=ServerConfig.protected(
            _authenticate,
            dashboard=DashboardConfig(enabled=False),
            evals=_evals_config(
                target,
                store,
                **(
                    {}
                    if execution_profile_policy is None
                    else {"execution_profile_policy": execution_profile_policy}
                ),
            ),
        ),
    )


def _execution_profile_revision(
    client: TestClient,
    target_key: str | None = None,
    *,
    path_prefix: str = "/api",
) -> str:
    response = client.get(f"{path_prefix}/evals/targets", headers=_AUTH_HEADERS)
    assert response.status_code == 200
    catalog = response.json()
    selected_key = target_key or catalog["default_target_key"]
    target = next(item for item in catalog["items"] if item["target_key"] == selected_key)
    assert target["execution_profile_ready"] is True
    assert target["execution_profile_diagnostics"] == []
    return target["execution_profile"]["revision"]


def _wait_for_terminal(
    client: TestClient,
    run_id: str,
    *,
    timeout: float = 5.0,
    path_prefix: str = "/api",
) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"{path_prefix}/evals/runs/{run_id}", headers=_AUTH_HEADERS)
        assert response.status_code == 200
        body = response.json()
        if body["status"] in {"completed", "failed", "cancelled"}:
            return body
        time.sleep(0.01)
    raise AssertionError("Eval run did not terminalize before the test deadline.")


def test_evals_configuration_is_default_off_durable_and_private(tmp_path) -> None:
    target = _target(_provider())
    store = SQLiteEvalStore(tmp_path / "evals.db")
    try:
        assert ServerConfig.protected(_authenticate).evals is None
        config = _evals_config(target, store)
        resolved = ServerConfig.protected(
            _authenticate,
            dashboard=DashboardConfig(enabled=False),
            evals=config,
        )

        assert resolved.evals == config
        assert resolved.safe_summary()["evals"] == {"configured": True}
        assert "target" not in config.model_dump()
        assert "store" not in config.model_dump()
        serialized = resolved.model_dump_json()
        assert target.key not in serialized
        assert str(store.path) not in serialized

        with pytest.raises(ValidationError, match="authenticated API access"):
            ServerConfig(
                access=OpenAccess(),
                dashboard=DashboardConfig(enabled=False),
                evals=config,
            )
        with pytest.raises(ValidationError, match="store must be durable"):
            EvalsConfig(target=target, store=InMemoryEvalStore())
        with pytest.raises(ValidationError, match="evals requires api.enabled"):
            ServerConfig.protected(
                _authenticate,
                api=ServerApiConfig(enabled=False),
                dashboard=DashboardConfig(enabled=False),
                evals=config,
            )
        forged = EvalsConfig.model_construct(
            target=target,
            store=InMemoryEvalStore(),
            lease_seconds=300,
            poll_interval_seconds=1.0,
            shutdown_grace_seconds=30.0,
        )
        with pytest.raises(ValidationError, match="store must be durable"):
            ServerConfig.protected(
                _authenticate,
                dashboard=DashboardConfig(enabled=False),
                evals=forged,
            )
    finally:
        asyncio.run(store.close())


def test_evals_rejects_a_target_from_another_application(tmp_path) -> None:
    target = _target(_provider())
    other = _target(_provider()).app
    store = SQLiteEvalStore(tmp_path / "evals.db")
    try:
        config = ServerConfig.protected(
            _authenticate,
            dashboard=DashboardConfig(enabled=False),
            evals=_evals_config(target, store),
        )
        with pytest.raises(ValueError, match="attached CayuApp"):
            create_server(other, config=config)
    finally:
        asyncio.run(store.close())


def test_server_preserves_explicit_shared_eval_execution_capacity(
    tmp_path,
    monkeypatch,
) -> None:
    target = _target(_provider())
    store = SQLiteEvalStore(tmp_path / "evals.db")
    capacity = EvalExecutionCapacity(max_active_trials=10_000)
    observed = []

    class RecordingCoordinator:
        def __init__(self, runtime) -> None:
            observed.append(runtime.execution_capacity)

        def start(self) -> None:
            pass

        async def stop(self) -> None:
            pass

    monkeypatch.setattr(routes_module, "EvalRunCoordinator", RecordingCoordinator)
    config = ServerConfig.protected(
        _authenticate,
        dashboard=DashboardConfig(enabled=False),
        evals=_evals_config(target, store, execution_capacity=capacity),
    )
    try:
        with TestClient(create_server(target.app, config=config)):
            pass
        assert observed == [capacity]
    finally:
        asyncio.run(store.close())


def test_evals_routes_are_absent_without_complete_configuration() -> None:
    server = create_server(
        _target(_provider()).app,
        config=ServerConfig.protected(
            _authenticate,
            dashboard=DashboardConfig(enabled=False),
        ),
    )
    with TestClient(server) as client:
        assert client.get("/api/evals/corpora", headers=_AUTH_HEADERS).status_code == 404


def test_framework_owned_local_context_allows_loopback_cli_evals_without_relaxing_v1(
    tmp_path,
) -> None:
    target = _target(_provider())
    automatic_path = tmp_path / "automatic.db"
    automatic_store = SQLiteEvalStore(automatic_path)
    explicit_store = SQLiteEvalStore(tmp_path / "explicit.db")
    context = _create_project_control_plane_context(
        project_root=tmp_path.resolve(),
        project_id="local-project",
        configured_release_id="local-release",
        eval_store=automatic_store,
        store_backend="sqlite",
        store_source="project",
        access=ProjectControlPlaneAccess.TRUSTED_LOCAL_DEVELOPMENT,
    )
    server = FastAPI()
    mount_cayu(
        server,
        target.app,
        dashboard=False,
        access=OpenAccess(),
        evals=_evals_config(target, explicit_store),
        _project_context=context,
    )
    try:
        corpus = _corpus()
        with TestClient(server) as client:
            imported = client.post(
                "/cayu/api/evals/corpora",
                json=corpus.model_dump(mode="json"),
            )
            assert imported.status_code == 201

        assert [item.revision for item in asyncio.run(explicit_store.list_corpora()).items] == [
            corpus.revision
        ]
        automatic_reader = SQLiteEvalStore(automatic_path)
        try:
            assert asyncio.run(automatic_reader.list_corpora()).items == ()
        finally:
            asyncio.run(automatic_reader.close())
    finally:
        asyncio.run(context.close())
        asyncio.run(explicit_store.close())


def test_evals_openapi_has_no_dangling_manual_request_schema_references(tmp_path) -> None:
    target = _target(_provider())
    store = SQLiteEvalStore(tmp_path / "evals.db")
    try:
        schema = _server(target, store).openapi()
        components = schema["components"]["schemas"]
        request_schemas = {
            path: schema["paths"][path]["post"]["requestBody"]["content"]["application/json"][
                "schema"
            ]
            for path in ("/api/evals/corpora", "/api/evals/runs", "/api/evals/comparisons")
        }

        def without_null_defaults(value):
            if isinstance(value, dict):
                return {
                    key: without_null_defaults(item)
                    for key, item in value.items()
                    if not (key == "default" and item is None)
                }
            if isinstance(value, list):
                return [without_null_defaults(item) for item in value]
            return value

        expected_run_schema = EvalRunCreateRequest.model_json_schema()
        definitions = expected_run_schema.pop("$defs")

        def inline_definitions(value):
            if isinstance(value, dict):
                reference = value.get("$ref")
                if isinstance(reference, str) and reference.startswith("#/$defs/"):
                    return inline_definitions(definitions[reference.rsplit("/", 1)[-1]])
                return {key: inline_definitions(item) for key, item in value.items()}
            if isinstance(value, list):
                return [inline_definitions(item) for item in value]
            return value

        assert request_schemas["/api/evals/runs"] == without_null_defaults(
            inline_definitions(expected_run_schema)
        )
        assert (
            request_schemas["/api/evals/comparisons"] == EvalComparisonRequest.model_json_schema()
        )

        def references(value):
            if isinstance(value, dict):
                if "$ref" in value:
                    yield value["$ref"]
                for item in value.values():
                    yield from references(item)
            elif isinstance(value, list):
                for item in value:
                    yield from references(item)

        for reference in references(schema):
            if reference.startswith("#/components/schemas/"):
                assert reference.rsplit("/", 1)[-1] in components
    finally:
        asyncio.run(store.close())


def test_mounted_evals_routes_run_under_the_host_lifespan(tmp_path) -> None:
    target = _target(_provider(trials=1))
    corpus = _corpus(trials=1)
    store = SQLiteEvalStore(tmp_path / "evals.db")
    server = FastAPI()
    try:
        mount_cayu(
            server,
            target.app,
            path="/cayu",
            dashboard=False,
            access=AuthenticatedAccess(dependency=_authenticate),
            evals=_evals_config(target, store),
        )
        with TestClient(server) as client:
            imported = client.post(
                "/cayu/api/evals/corpora",
                headers=_AUTH_HEADERS,
                json=corpus.model_dump(mode="json"),
            )
            assert imported.status_code == 201
            admitted = client.post(
                "/cayu/api/evals/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "mounted-run"},
                json={
                    "corpus_revision": corpus.revision,
                    "suite_id": corpus.suites[0].id,
                    "expected_execution_profile_revision": _execution_profile_revision(
                        client,
                        path_prefix="/cayu/api",
                    ),
                },
            )
            assert admitted.status_code == 202
            terminal = _wait_for_terminal(
                client,
                admitted.json()["spec"]["run_id"],
                path_prefix="/cayu/api",
            )
            assert terminal["status"] == "completed"
    finally:
        asyncio.run(store.close())


def test_evals_authenticates_and_bounds_before_parsing(tmp_path) -> None:
    target = _target(_provider())
    store = SQLiteEvalStore(tmp_path / "evals.db")
    calls = 0

    def authenticate(request: Request) -> AuthContext:
        nonlocal calls
        calls += 1
        return _authenticate(request)

    config = ServerConfig.protected(
        authenticate,
        dashboard=DashboardConfig(enabled=False),
        evals=_evals_config(target, store),
    )
    try:
        with TestClient(create_server(target.app, config=config)) as client:
            unauthorized = client.post(
                "/api/evals/corpora",
                content=b'{"not valid"',
                headers={"Content-Type": "application/json"},
            )
            assert unauthorized.status_code == 401
            assert calls == 1

            invalid = client.post(
                "/api/evals/corpora",
                content=b'{"not valid"',
                headers={**_AUTH_HEADERS, "Content-Type": "application/json"},
            )
            assert invalid.status_code == 422
            assert invalid.json() == {"detail": "Invalid Evals request."}
            assert calls == 2

            duplicate = client.post(
                "/api/evals/corpora",
                content=b'{"schema_version":1,"schema_version":1}',
                headers={**_AUTH_HEADERS, "Content-Type": "application/json"},
            )
            assert duplicate.status_code == 422
            assert duplicate.json() == {"detail": "Invalid Evals request."}
            assert calls == 3

            invalid_utf8 = client.post(
                "/api/evals/corpora",
                content=b'{"description":"\xff"}',
                headers={**_AUTH_HEADERS, "Content-Type": "application/json"},
            )
            assert invalid_utf8.status_code == 422
            assert invalid_utf8.json() == {"detail": "Invalid Evals request."}
            assert calls == 4

            wrong_media_type = client.post(
                "/api/evals/corpora",
                content=json.dumps(_corpus().model_dump(mode="json")),
                headers={**_AUTH_HEADERS, "Content-Type": "text/plain"},
            )
            assert wrong_media_type.status_code == 422
            assert wrong_media_type.json() == {"detail": "Invalid Evals request."}
            assert calls == 5

            oversized = client.post(
                "/api/evals/corpora",
                content=b"x" * (MAX_EVALS_REQUEST_BYTES + 1),
                headers={**_AUTH_HEADERS, "Content-Type": "application/json"},
            )
            assert oversized.status_code == 413
            assert calls == 6
    finally:
        asyncio.run(store.close())


def test_evals_replays_a_body_consumed_by_auth_and_bounds_auth_reads(tmp_path) -> None:
    target = _target(_provider(trials=1))
    corpus = _corpus(trials=1)
    store = SQLiteEvalStore(tmp_path / "evals.db")
    completed_auth_body_sizes: list[int] = []

    async def authenticate(request: Request) -> AuthContext:
        body = await request.body()
        completed_auth_body_sizes.append(len(body))
        return AuthContext(subject="body-signing-operator")

    config = ServerConfig.protected(
        authenticate,
        dashboard=DashboardConfig(enabled=False),
        evals=_evals_config(target, store),
    )
    try:
        with TestClient(create_server(target.app, config=config)) as client:
            imported = client.post(
                "/api/evals/corpora",
                json=corpus.model_dump(mode="json"),
            )
            assert imported.status_code == 201
            assert imported.json()["revision"] == corpus.revision
            assert len(completed_auth_body_sizes) == 1
            assert completed_auth_body_sizes[0] > 0

            oversized = client.post(
                "/api/evals/corpora",
                content=b"x" * (MAX_EVALS_REQUEST_BYTES + 1),
                headers={"Content-Type": "application/json"},
            )
            assert oversized.status_code == 413
            assert len(completed_auth_body_sizes) == 1
    finally:
        asyncio.run(store.close())


def test_evals_api_imports_executes_compares_and_exports_deterministically(
    tmp_path,
    monkeypatch,
) -> None:
    provider = _provider(trials=2)
    target = _target(provider)
    corpus = _corpus(trials=2)
    store = SQLiteEvalStore(tmp_path / "evals.db")
    try:
        with TestClient(
            _server(
                target,
                store,
                execution_profile_policy=_repeatable_execution_policy(target),
            )
        ) as client:
            imported = client.post(
                "/api/evals/corpora",
                headers=_AUTH_HEADERS,
                json=corpus.model_dump(mode="json"),
            )
            assert imported.status_code == 201
            assert imported.json()["revision"] == corpus.revision

            catalog = client.get("/api/evals/corpora", headers=_AUTH_HEADERS)
            assert catalog.status_code == 200
            assert [item["revision"] for item in catalog.json()["items"]] == [corpus.revision]
            suites = client.get(
                f"/api/evals/corpora/{corpus.revision}/suites",
                headers=_AUTH_HEADERS,
            )
            assert suites.status_code == 200
            assert suites.json()["items"][0]["id"] == corpus.suites[0].id
            cases = client.get(
                f"/api/evals/corpora/{corpus.revision}/suites/{corpus.suites[0].id}/cases",
                headers=_AUTH_HEADERS,
            )
            assert cases.status_code == 200
            assert cases.json()["items"][0]["id"] == corpus.cases[0].id

            request = {
                "corpus_revision": corpus.revision,
                "suite_id": corpus.suites[0].id,
                "expected_execution_profile_revision": _execution_profile_revision(client),
                "max_concurrency": 1,
                "max_steps": 1,
                "limits": {"max_total_tokens": 100, "scope": "run"},
            }
            admitted = client.post(
                "/api/evals/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "eval-run-one"},
                json=request,
            )
            assert admitted.status_code == 202
            invocation = admitted.json()["spec"]["invocation"]
            execution_profile = invocation.pop("execution_profile")
            execution_profile_snapshot = invocation.pop("execution_profile_snapshot")
            admission_request_revision = invocation.pop("admission_request_revision")
            assert invocation == {
                "schema_version": 1,
                "source": "http_run",
                "origin": {
                    "trust": "server_verified",
                    "subject": "eval-operator",
                    "tenant": None,
                },
                "max_steps": 1,
                "limits": {
                    "max_input_tokens": None,
                    "max_output_tokens": None,
                    "max_total_tokens": 100,
                    "max_tool_calls": None,
                    "max_elapsed_seconds": None,
                    "scope": "run",
                },
                "cost_budget": None,
            }
            assert execution_profile["schema_version"] == 1
            assert execution_profile["profile_revision"].startswith("sha256:")
            assert execution_profile["runtime_execution_profile"]["fingerprint"]
            assert execution_profile_snapshot["revision"] == execution_profile["profile_revision"]
            assert admission_request_revision.startswith("sha256:")
            run_id = admitted.json()["spec"]["run_id"]
            replayed = client.post(
                "/api/evals/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "eval-run-one"},
                json=request,
            )
            assert replayed.status_code == 202
            assert replayed.json()["spec"]["run_id"] == run_id

            terminal = _wait_for_terminal(client, run_id)
            assert terminal["status"] == "completed"
            assert terminal["result"]["status"] == "passed"

            def unavailable_profile(*, model: str) -> None:
                del model
                raise RuntimeError("provider temporarily unavailable")

            monkeypatch.setattr(provider, "preflight_model_target", unavailable_profile)
            replayed_while_unavailable = client.post(
                "/api/evals/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "eval-run-one"},
                json=request,
            )
            assert replayed_while_unavailable.status_code == 202
            assert replayed_while_unavailable.json()["spec"]["run_id"] == run_id
            conflicting_retry = client.post(
                "/api/evals/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "eval-run-one"},
                json={**request, "max_concurrency": 2},
            )
            assert conflicting_retry.status_code == 409
            assert conflicting_retry.json() == {
                "detail": "Idempotency-Key is already bound to another eval run request."
            }
            sessions = asyncio.run(target.app.session_store.list_sessions()).sessions
            assert len(sessions) == 2
            assert all(
                session.invocation.source is SessionExecutionSource.HTTP_RUN
                and session.invocation.origin.trust is InvocationOriginTrust.SERVER_VERIFIED
                and session.invocation.origin.subject == "eval-operator"
                for session in sessions
            )

            result = client.get(f"/api/evals/runs/{run_id}/result", headers=_AUTH_HEADERS)
            assert result.status_code == 200
            result_body = result.json()
            assert result_body["result"]["run"]["status"] == "passed"
            assert (
                result_body["presentation"]["result_revision"] == result_body["result"]["revision"]
            )
            assert result_body["presentation"]["dimensions"] == {
                "candidate": "passed",
                "deterministic_assertions": "passed",
                "semantic_quality": "not_used",
                "evaluator_health": "not_used",
                "runtime": "completed",
                "evidence": "complete",
            }
            comparison = client.post(
                "/api/evals/comparisons",
                headers=_AUTH_HEADERS,
                json={"baseline_run_id": run_id, "current_run_id": run_id},
            )
            assert comparison.status_code == 200
            assert comparison.json()["comparison"]["compatibility"]["comparable"] is True
            assert comparison.json()["comparison"]["regressions"] == []

            json_report = client.get(
                f"/api/evals/runs/{run_id}/report.json",
                headers=_AUTH_HEADERS,
            )
            assert json_report.status_code == 200
            assert json_report.content.endswith(b"\n")
            json_report_body = json.loads(json_report.content)
            assert json_report_body["record_type"] == "cayu.eval-result-report"
            assert json_report_body["result"]["revision"] == terminal["result"]["revision"]
            assert (
                json_report_body["presentation"]["result_revision"]
                == terminal["result"]["revision"]
            )
            assert (
                client.get(
                    f"/api/evals/runs/{run_id}/report.json",
                    headers=_AUTH_HEADERS,
                ).content
                == json_report.content
            )
            html_report = client.get(
                f"/api/evals/runs/{run_id}/report.html",
                headers=_AUTH_HEADERS,
            )
            assert html_report.status_code == 200
            assert b"Cayu Eval Report" in html_report.content

            download = client.get(
                f"/api/evals/corpora/{corpus.revision}/download",
                headers=_AUTH_HEADERS,
            )
            assert download.status_code == 200
            assert download.content.endswith(b"\n")
            assert json.loads(download.content)["revision"] == corpus.revision

            capabilities = client.get("/api/contract", headers=_AUTH_HEADERS).json()["capabilities"]
            capability = capabilities["surfaces"]["evals"]
            assert capability == {
                "configured": True,
                "read": {"enabled": True, "unavailable_reason": None},
                "mutate": {"enabled": True, "unavailable_reason": None},
            }
            assert capabilities["evals_readiness"] == {
                "captured_evaluation": {
                    "state": "ready",
                    "reason_code": None,
                },
                "catalog_read": {"state": "ready", "reason_code": None},
                "catalog_write": {"state": "ready", "reason_code": None},
                "captured_result_persistence": {
                    "state": "ready",
                    "reason_code": None,
                },
                "scenario_conversion": {
                    "state": "ready",
                    "reason_code": None,
                },
                "fresh_launch": {"state": "ready", "reason_code": None},
                "cancellation": {"state": "ready", "reason_code": None},
                "comparison": {"state": "ready", "reason_code": None},
                "reports": {"state": "ready", "reason_code": None},
            }
    finally:
        asyncio.run(store.close())


def test_evals_api_compares_compatible_releases_and_returns_typed_regressions(tmp_path) -> None:
    corpus = _corpus(trials=1)
    database = tmp_path / "evals.db"
    request = {
        "corpus_revision": corpus.revision,
        "suite_id": corpus.suites[0].id,
        "max_concurrency": 1,
    }
    baseline_store = SQLiteEvalStore(database)
    try:
        baseline_target = _target(
            _provider(trials=1),
            application_release_id="baseline-release",
        )
        with TestClient(_server(baseline_target, baseline_store)) as client:
            request["expected_execution_profile_revision"] = _execution_profile_revision(client)
            assert (
                client.post(
                    "/api/evals/corpora",
                    headers=_AUTH_HEADERS,
                    json=corpus.model_dump(mode="json"),
                ).status_code
                == 201
            )
            baseline_admission = client.post(
                "/api/evals/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "baseline-run"},
                json=request,
            )
            assert baseline_admission.status_code == 202
            baseline_run_id = baseline_admission.json()["spec"]["run_id"]
            assert _wait_for_terminal(client, baseline_run_id)["result"]["status"] == "passed"

    finally:
        asyncio.run(baseline_store.close())

    current_store = SQLiteEvalStore(database)
    try:
        current_target = _target(
            _provider(trials=1, output="Denied"),
            application_release_id="current-release",
        )
        with TestClient(_server(current_target, current_store)) as client:
            request["expected_execution_profile_revision"] = _execution_profile_revision(client)
            current_admission = client.post(
                "/api/evals/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "current-run"},
                json=request,
            )
            assert current_admission.status_code == 202
            current_run_id = current_admission.json()["spec"]["run_id"]
            assert _wait_for_terminal(client, current_run_id)["result"]["status"] == "failed"

            response = client.post(
                "/api/evals/comparisons",
                headers=_AUTH_HEADERS,
                json={
                    "baseline_run_id": baseline_run_id,
                    "current_run_id": current_run_id,
                },
            )
            assert response.status_code == 200, response.text
            comparison = response.json()["comparison"]
            assert comparison["compatibility"]["comparable"] is True
            assert comparison["baseline"]["application_release_id"] == "baseline-release"
            assert comparison["current"]["application_release_id"] == "current-release"
            assert [
                (regression["scope"], regression["kind"], regression["case_id"])
                for regression in comparison["regressions"]
            ] == [
                ("run", "status", None),
                ("run", "score", None),
                ("case", "status", "refund-approval"),
                ("case", "score", "refund-approval"),
                ("case", "reliability", "refund-approval"),
            ]
    finally:
        asyncio.run(current_store.close())


def test_eval_result_refreshes_run_after_concurrent_publication(tmp_path, monkeypatch) -> None:
    target = _target(_provider(trials=1))
    corpus = _corpus(trials=1)
    store = SQLiteEvalStore(tmp_path / "evals.db")

    async def seed_result():
        result = await run_corpus_suite(
            target,
            corpus,
            corpus.suites[0].id,
            max_concurrency=1,
        )
        await store.save_corpus(corpus, redact_json=target.app.redact_json)
        request = EvalRunRequest(
            run_id="eval-result-read-race",
            idempotency_key="sha256:" + ("1" * 64),
            corpus_revision=corpus.revision,
            target_key=target.key,
            suite_id=corpus.suites[0].id,
            suite_revision=corpus.suites[0].revision,
            max_concurrency=1,
            invocation=await _bound_eval_invocation(target),
        )
        await store.admit_run(request, redact_json=target.app.redact_json)
        lease = await store.claim_run(target_key=target.key, lease_seconds=300)
        assert lease is not None
        active_record = lease.run
        completed_record = await store.publish_result(
            lease.claim,
            result,
            redact_json=target.app.redact_json,
        )
        return active_record, completed_record, result

    active_record, completed_record, result = asyncio.run(seed_result())
    with pytest.raises(ValidationError, match="requires a completed run"):
        EvalResultResponse(
            run=active_record,
            result=result,
            presentation=present_eval_result(result),
        )
    canonical_presentation = present_eval_result(result)
    forged_presentation_payload = canonical_presentation.model_dump(
        mode="python",
        round_trip=True,
        warnings="none",
    )
    forged_presentation_payload["cases"][0]["trials"][0]["assertions"][0]["kind"] = (
        "forged_deterministic_kind"
    )
    forged_presentation = type(canonical_presentation).model_validate(forged_presentation_payload)
    with pytest.raises(ValidationError, match="presentation does not match"):
        EvalResultResponse(
            run=completed_record,
            result=result,
            presentation=forged_presentation,
        )

    original_load_run = store.load_run
    load_run_calls = 0

    async def staged_load_run(run_id: str):
        nonlocal load_run_calls
        load_run_calls += 1
        if load_run_calls == 1:
            return active_record
        return await original_load_run(run_id)

    monkeypatch.setattr(store, "load_run", staged_load_run)
    try:
        with TestClient(_server(target, store)) as client:
            response = client.get(
                "/api/evals/runs/eval-result-read-race/result",
                headers=_AUTH_HEADERS,
            )
        assert response.status_code == 200
        assert load_run_calls == 2
        body = response.json()
        assert body["run"] == completed_record.model_dump(mode="json")
        assert body["run"]["status"] == "completed"
        assert body["result"]["run"]["status"] == "passed"
    finally:
        asyncio.run(store.close())


def test_memory_report_routes_require_exact_stored_results_and_render_both_formats(
    tmp_path,
) -> None:
    report_request, target, corpus = asyncio.run(_report_fixture())
    store = SQLiteEvalStore(tmp_path / "memory-reports.db")

    async def seed_results() -> None:
        await store.save_corpus(corpus, redact_json=target.app.redact_json)
        variants = {item.variant_id: item for item in report_request.variants}
        for index, evidence in enumerate(report_request.published_results, start=1):
            trial = next(
                item
                for item in report_request.trials
                if item.published_result_revision == evidence.result.revision
            )
            variant = variants[trial.variant_id]
            published = evidence.result.run
            request = EvalRunRequest(
                run_id=evidence.run_id,
                idempotency_key="sha256:" + str(index) * 64,
                corpus_revision=published.corpus_revision,
                target_key=evidence.result.target.target_key,
                suite_id=published.suite_id,
                suite_revision=published.suite_revision,
                max_concurrency=1,
                invocation=EvalRunInvocation(
                    execution_profile=variant.execution_profile_binding,
                    execution_profile_snapshot=variant.execution_profile,
                ),
            )
            await store.admit_run(request, redact_json=target.app.redact_json)
            lease = await store.claim_run(target_key=target.key, lease_seconds=300)
            assert lease is not None
            assert lease.run.id == evidence.run_id
            await store.publish_result(
                lease.claim,
                evidence.result,
                redact_json=target.app.redact_json,
            )

    asyncio.run(seed_results())
    try:
        server = _server(target, store)
        schema = server.openapi()
        expected_request_schema = {"$ref": "#/components/schemas/MemoryExperimentReportRequest"}
        for path in (
            "/api/evals/memory-reports",
            "/api/evals/memory-reports/report.html",
        ):
            assert (
                schema["paths"][path]["post"]["requestBody"]["content"]["application/json"][
                    "schema"
                ]
                == expected_request_schema
            )
        assert "MemoryExperimentReportRequest" in schema["components"]["schemas"]

        with TestClient(server) as client:
            payload = report_request.model_dump(mode="json")
            assert (
                build_memory_experiment_report(
                    MemoryExperimentReportRequest.model_validate(payload)
                ).selected_variant_id
                == "candidate"
            )
            unauthorized = client.post("/api/evals/memory-reports", json=payload)
            assert unauthorized.status_code == 401

            for malformed_result in ("invalid", [], 1, True):
                malformed = json.loads(json.dumps(payload))
                malformed["published_results"][0]["result"] = malformed_result
                rejected_malformed = client.post(
                    "/api/evals/memory-reports",
                    headers=_AUTH_HEADERS,
                    json=malformed,
                )
                assert rejected_malformed.status_code == 422
                assert rejected_malformed.json() == {
                    "detail": "Invalid memory experiment report request."
                }

            aliased_metrics = json.loads(json.dumps(payload))
            task_quality = next(
                item
                for item in aliased_metrics["metric_bindings"]
                if item["role"] == "task_quality"
            )
            safety = next(
                item for item in aliased_metrics["metric_bindings"] if item["role"] == "safety"
            )
            safety["assertion_id"] = task_quality["assertion_id"]
            safety["assertion_revision"] = task_quality["assertion_revision"]
            rejected_alias = client.post(
                "/api/evals/memory-reports",
                headers=_AUTH_HEADERS,
                json=aliased_metrics,
            )
            assert rejected_alias.status_code == 422
            assert rejected_alias.json() == {"detail": "Invalid memory experiment report request."}

            response = client.post(
                "/api/evals/memory-reports",
                headers=_AUTH_HEADERS,
                json=payload,
            )
            assert response.status_code == 200, response.text
            assert response.json()["selected_variant_id"] == "candidate"
            assert len(response.json()["rows"]) == 4

            rendered = client.post(
                "/api/evals/memory-reports/report.html",
                headers=_AUTH_HEADERS,
                json=payload,
            )
            assert rendered.status_code == 200
            assert rendered.headers["content-type"].startswith("text/html")
            assert "Complete trial matrix" in rendered.text

            resultless_trials = tuple(
                trial.model_copy(update={"published_result_revision": None})
                if trial.variant_id == "candidate"
                else trial
                for trial in report_request.trials
            )
            referenced_revisions = {
                trial.published_result_revision
                for trial in resultless_trials
                if trial.published_result_revision is not None
            }
            resultless_request = MemoryExperimentReportRequest.model_validate(
                report_request.model_copy(
                    update={
                        "published_results": tuple(
                            evidence
                            for evidence in report_request.published_results
                            if evidence.result.revision in referenced_revisions
                        ),
                        "trials": resultless_trials,
                    }
                ).model_dump(mode="python")
            )
            resultless = client.post(
                "/api/evals/memory-reports",
                headers=_AUTH_HEADERS,
                json=resultless_request.model_dump(mode="json"),
            )
            assert resultless.status_code == 200, resultless.text
            assert {
                row["availability"]
                for row in resultless.json()["rows"]
                if row["variant_id"] == "candidate"
            } == {"unmatched"}

            forged = json.loads(json.dumps(payload))
            first_run_id = forged["published_results"][0]["run_id"]
            forged["published_results"][0]["run_id"] = forged["published_results"][1]["run_id"]
            forged["published_results"][1]["run_id"] = first_run_id
            rejected = client.post(
                "/api/evals/memory-reports",
                headers=_AUTH_HEADERS,
                json=forged,
            )
            assert rejected.status_code == 409
            assert rejected.json()["detail"] == (
                "Memory report eval run evidence changed during readback."
            )
    finally:
        asyncio.run(store.close())


def test_evals_rejects_incompatible_import_before_provider_dispatch(tmp_path) -> None:
    provider = _provider()
    target = _target(provider)
    corpus = _corpus(target_key="another-target")
    store = SQLiteEvalStore(tmp_path / "evals.db")
    try:
        with TestClient(_server(target, store)) as client:
            response = client.post(
                "/api/evals/corpora",
                headers=_AUTH_HEADERS,
                json=corpus.model_dump(mode="json"),
            )
            assert response.status_code == 400
            assert response.json() == {
                "detail": "Eval corpus is incompatible with the attached target."
            }
            assert provider.requests == []
            assert client.get("/api/evals/corpora", headers=_AUTH_HEADERS).json()["items"] == []
    finally:
        asyncio.run(store.close())


def test_evals_rejects_stale_profile_and_undeclared_repetition_before_admission(
    tmp_path,
) -> None:
    provider = _provider(trials=2)
    target = _target(provider)
    corpus = _corpus(trials=2)
    store = SQLiteEvalStore(tmp_path / "evals.db")
    try:
        with TestClient(_server(target, store)) as client:
            imported = client.post(
                "/api/evals/corpora",
                headers=_AUTH_HEADERS,
                json=corpus.model_dump(mode="json"),
            )
            assert imported.status_code == 201

            stale = client.post(
                "/api/evals/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "stale-profile"},
                json={
                    "corpus_revision": corpus.revision,
                    "suite_id": corpus.suites[0].id,
                    "expected_execution_profile_revision": "sha256:" + "0" * 64,
                },
            )
            assert stale.status_code == 400
            assert stale.json() == {
                "detail": "Eval run exceeds the published execution-profile trial limit."
            }

            single_trial = _corpus(trials=1)
            assert (
                client.post(
                    "/api/evals/corpora",
                    headers=_AUTH_HEADERS,
                    json=single_trial.model_dump(mode="json"),
                ).status_code
                == 201
            )
            stale = client.post(
                "/api/evals/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "stale-profile"},
                json={
                    "corpus_revision": single_trial.revision,
                    "suite_id": single_trial.suites[0].id,
                    "expected_execution_profile_revision": "sha256:" + "0" * 64,
                },
            )
            assert stale.status_code == 409
            assert stale.json() == {
                "detail": (
                    "The selected eval execution profile changed after it was reviewed. "
                    "Refresh readiness before launching."
                )
            }
            assert client.get("/api/evals/runs", headers=_AUTH_HEADERS).json()["items"] == []
            assert provider.requests == []
    finally:
        asyncio.run(store.close())


def test_evals_reports_published_application_identity_drift_before_admission(tmp_path) -> None:
    target = _target(_provider(trials=1))
    corpus = _corpus(trials=1)
    store = SQLiteEvalStore(tmp_path / "evals.db")
    try:
        with TestClient(_server(target, store)) as client:
            profile_revision = _execution_profile_revision(client)
            assert (
                client.post(
                    "/api/evals/corpora",
                    headers=_AUTH_HEADERS,
                    json=corpus.model_dump(mode="json"),
                ).status_code
                == 201
            )
            target.app.register_agent(AgentSpec(name="late-agent", model="fixture-model"))

            catalog = client.get("/api/evals/targets", headers=_AUTH_HEADERS)
            assert catalog.status_code == 200
            target_entry = catalog.json()["items"][0]
            assert target_entry["execution_profile_ready"] is False
            assert target_entry["execution_profile_diagnostics"][0]["code"] == (
                "application_identity_changed"
            )
            launched = client.post(
                "/api/evals/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "changed-application"},
                json={
                    "corpus_revision": corpus.revision,
                    "suite_id": corpus.suites[0].id,
                    "expected_execution_profile_revision": profile_revision,
                },
            )
            assert launched.status_code == 409
            assert launched.json() == {
                "detail": (
                    "The application identity for this eval execution profile changed after "
                    "the target was published. Refresh the deployment before launching."
                )
            }
            assert client.get("/api/evals/runs", headers=_AUTH_HEADERS).json()["items"] == []
    finally:
        asyncio.run(store.close())


def test_evals_server_never_lists_reads_or_claims_another_target_run(tmp_path) -> None:
    target = _target(_provider())
    other_corpus = _corpus(target_key="other-target", trials=1)
    other_suite = other_corpus.suites[0]
    other_request = EvalRunRequest(
        run_id="other-target-run",
        idempotency_key="sha256:" + "f" * 64,
        corpus_revision=other_corpus.revision,
        target_key=other_corpus.target_key,
        suite_id=other_suite.id,
        suite_revision=other_suite.revision,
        max_concurrency=1,
    )
    store = SQLiteEvalStore(tmp_path / "evals.db")

    async def seed_other_target() -> None:
        await store.save_corpus(other_corpus, redact_json=target.app.redact_json)
        await store.admit_run(other_request, redact_json=target.app.redact_json)

    asyncio.run(seed_other_target())
    try:
        with TestClient(_server(target, store)) as client:
            time.sleep(0.1)
            assert client.get("/api/evals/corpora", headers=_AUTH_HEADERS).json()["items"] == []
            assert client.get("/api/evals/runs", headers=_AUTH_HEADERS).json()["items"] == []
            assert (
                client.get(
                    f"/api/evals/corpora/{other_corpus.revision}",
                    headers=_AUTH_HEADERS,
                ).status_code
                == 404
            )
            assert (
                client.get(
                    f"/api/evals/runs/{other_request.run_id}",
                    headers=_AUTH_HEADERS,
                ).status_code
                == 404
            )

        other_record = asyncio.run(store.load_run(other_request.run_id))
        assert other_record is not None
        assert other_record.status is EvalRunStatus.QUEUED
        assert other_record.ownership is None
    finally:
        asyncio.run(store.request_cancel(other_request.run_id))
        asyncio.run(store.close())


class _BlockingProvider(ModelProvider):
    name = "blocking-eval-provider"

    def __init__(self) -> None:
        self.started = threading.Event()
        self.cancelled = threading.Event()

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        self.started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        if False:  # pragma: no cover - keeps this an async generator
            yield ModelStreamEvent.text_delta("")


class _RecoveringJudgeProvider(ModelProvider):
    name = "recovering-judge-provider"

    def __init__(self) -> None:
        self.started = threading.Event()
        self.cancelled = threading.Event()
        self._lock = threading.Lock()
        self.request_count = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        with self._lock:
            self.request_count += 1
            attempt = self.request_count
        if attempt == 1:
            self.started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
        yield ModelStreamEvent.text_delta('{"score": 0.9, "rationale": "recovered judgment"}')
        yield ModelStreamEvent.completed(
            {
                "finish_reason": "stop",
                "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
            }
        )


class _FailingProvider(ModelProvider):
    name = "failing-eval-provider"

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        raise RuntimeError("private provider failure must not be published")
        if False:  # pragma: no cover - keeps this an async generator
            yield ModelStreamEvent.text_delta("")


def test_eval_provider_failure_is_contained_without_publishing_exception_text(tmp_path) -> None:
    target = _target(_FailingProvider())
    corpus = _corpus(trials=1)
    store = SQLiteEvalStore(tmp_path / "evals.db")
    try:
        with TestClient(_server(target, store)) as client:
            imported = client.post(
                "/api/evals/corpora",
                headers=_AUTH_HEADERS,
                json=corpus.model_dump(mode="json"),
            )
            assert imported.status_code == 201
            admitted = client.post(
                "/api/evals/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "provider-failure"},
                json={
                    "corpus_revision": corpus.revision,
                    "suite_id": corpus.suites[0].id,
                    "expected_execution_profile_revision": _execution_profile_revision(client),
                },
            )
            assert admitted.status_code == 202

            run_id = admitted.json()["spec"]["run_id"]
            terminal = _wait_for_terminal(client, run_id)
            assert terminal["status"] == "completed"
            assert terminal["result"]["status"] == "failed"
            result = client.get(f"/api/evals/runs/{run_id}/result", headers=_AUTH_HEADERS)
            assert result.status_code == 200
            assert result.json()["result"]["run"]["status"] == "failed"
            assert "private provider failure" not in json.dumps(terminal)
            assert "private provider failure" not in result.text
    finally:
        asyncio.run(store.close())


def test_running_eval_cancellation_stops_execution_before_terminalizing(tmp_path) -> None:
    provider = _BlockingProvider()
    target = _target(provider)
    corpus = _corpus(trials=1)
    store = SQLiteEvalStore(tmp_path / "evals.db")
    try:
        with TestClient(_server(target, store)) as client:
            assert (
                client.post(
                    "/api/evals/corpora",
                    headers=_AUTH_HEADERS,
                    json=corpus.model_dump(mode="json"),
                ).status_code
                == 201
            )
            admitted = client.post(
                "/api/evals/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "cancel-running"},
                json={
                    "corpus_revision": corpus.revision,
                    "suite_id": corpus.suites[0].id,
                    "expected_execution_profile_revision": _execution_profile_revision(client),
                    "max_concurrency": 1,
                },
            )
            run_id = admitted.json()["spec"]["run_id"]
            assert provider.started.wait(timeout=2)

            cancellation = client.post(
                f"/api/evals/runs/{run_id}/cancel",
                headers=_AUTH_HEADERS,
            )
            assert cancellation.status_code == 202
            assert cancellation.json()["status"] in {"cancelling", "cancelled"}
            terminal = _wait_for_terminal(client, run_id)
            assert terminal["status"] == "cancelled"
            assert provider.cancelled.wait(timeout=2)
            assert (
                client.get(
                    f"/api/evals/runs/{run_id}/result",
                    headers=_AUTH_HEADERS,
                ).status_code
                == 409
            )
    finally:
        asyncio.run(store.close())


def test_attached_worker_runs_the_same_trusted_model_judge_contract(tmp_path) -> None:
    judge, judge_provider = _model_judge_target()
    candidate_provider = _provider(trials=1)
    target = _target(candidate_provider, model_judges=(judge,))
    corpus = _model_judge_corpus(judge)
    store = SQLiteEvalStore(tmp_path / "evals.db")
    try:
        with TestClient(_server(target, store)) as client:
            imported = client.post(
                "/api/evals/corpora",
                headers=_AUTH_HEADERS,
                json=corpus.model_dump(mode="json"),
            )
            assert imported.status_code == 201
            admitted = client.post(
                "/api/evals/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "portable-model-judge"},
                json={
                    "corpus_revision": corpus.revision,
                    "suite_id": corpus.suites[0].id,
                    "expected_execution_profile_revision": _execution_profile_revision(client),
                },
            )
            assert admitted.status_code == 202

            terminal = _wait_for_terminal(client, admitted.json()["spec"]["run_id"])

            assert terminal["status"] == "completed"
            assert terminal["attempt_count"] == 1
            assertion = terminal["result"]
            assert assertion["status"] == "passed"
            result_response = client.get(
                f"/api/evals/runs/{terminal['spec']['run_id']}/result",
                headers=_AUTH_HEADERS,
            )
            assert result_response.status_code == 200
            detail = result_response.json()["result"]["run"]["cases"][0]["trials"][0]["assertions"][
                0
            ]["detail"]
            assert detail["kind"] == "model_judge"
            assert detail["diagnostic"] == "judgment_recorded"
            assert len(candidate_provider.requests) == 1
            assert len(judge_provider.requests) == 1
    finally:
        asyncio.run(store.close())


def test_authored_suite_exposes_and_executes_rubric_string_model_judge(tmp_path) -> None:
    judge, judge_provider = _model_judge_target()
    candidate_provider = _provider(trials=1)
    target = _target(candidate_provider, model_judges=(judge,))
    store = SQLiteEvalStore(tmp_path / "evals.db")
    draft = EvalSuiteDraftV1(
        id="authored-model-judge",
        target_key=target.key,
        name="Authored model judge",
        cases=(
            EvalCaseDraftV1(
                id="case-one",
                name="Case one",
                stimulus=EvalSimpleInputStimulusV1(
                    input=RunInputSpec(
                        messages=(CorpusUserMessageSpec(text="Can I get a refund?"),)
                    )
                ),
                assertions=(
                    ModelJudgeAssertionSpec(
                        id="quality",
                        evaluator_key=judge.key,
                        rubric="The answer correctly explains the refund policy.",
                        rubric_version="v1",
                    ),
                ),
            ),
        ),
    )
    try:
        with TestClient(_server(target, store)) as client:
            preview = client.post(
                "/api/evals/suites/preview",
                headers=_AUTH_HEADERS,
                json={"draft": draft.model_dump(mode="json")},
            )
            assert preview.status_code == 200
            suite = preview.json()["suite"]
            saved = client.post(
                "/api/evals/suites",
                headers=_AUTH_HEADERS,
                json={
                    "expected_suite_revision": suite["revision"],
                    "suite": suite,
                },
            )
            assert saved.status_code == 201
            launch_preview = client.post(
                f"/api/evals/suites/{suite['revision']}/runs/preview",
                headers=_AUTH_HEADERS,
                json={},
            )
            assert launch_preview.status_code == 200
            reviewed = launch_preview.json()
            assert reviewed["ready"] is True
            assert reviewed["exposure"]["judge_evaluations"] == 1
            assert reviewed["exposure"]["judge_profiles"][0]["profile_key"] == judge.key

            launched = client.post(
                f"/api/evals/suites/{suite['revision']}/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "authored-model-judge"},
                json={
                    "expected_exposure_revision": reviewed["exposure"]["revision"],
                    "expected_execution_profiles": [
                        {
                            "case_ids": item["case_ids"],
                            "execution_profile_revision": item["execution_profile_revision"],
                        }
                        for item in reviewed["launches"]
                    ],
                },
            )
            assert launched.status_code == 202
            terminal = _wait_for_terminal(
                client,
                launched.json()["runs"][0]["run"]["spec"]["run_id"],
            )
            assert terminal["status"] == "completed"
            assert len(candidate_provider.requests) == 1
            assert len(judge_provider.requests) == 1
    finally:
        asyncio.run(store.close())


def test_authored_lane_accepts_its_suite_wide_frozen_judge_profiles() -> None:
    revision_a = "sha256:" + "a" * 64
    revision_b = "sha256:" + "b" * 64
    comparison_a = "sha256:" + "c" * 64
    comparison_b = "sha256:" + "d" * 64
    exposure = SimpleNamespace(
        execution_profiles=(
            SimpleNamespace(
                case_ids=("case-one",),
                execution_profile_revision=revision_a,
                execution_profile_comparison_revision=comparison_a,
                candidate_cost_budget=None,
            ),
        ),
        judge_profiles=(
            SimpleNamespace(
                profile_key="judge-one",
                judge_profile_revision=revision_a,
                judge_profile_comparison_revision=comparison_a,
            ),
            SimpleNamespace(
                profile_key="judge-two",
                judge_profile_revision=revision_b,
                judge_profile_comparison_revision=comparison_b,
            ),
        ),
    )
    invocation = SimpleNamespace(
        authored_suite_exposure=exposure,
        execution_profile_snapshot=SimpleNamespace(
            revision=revision_a,
            comparison_revision=comparison_a,
        ),
        cost_budget=None,
    )
    lease = SimpleNamespace(run=SimpleNamespace(spec=SimpleNamespace(invocation=invocation)))
    compiled = SimpleNamespace(
        run_contract=SimpleNamespace(
            suite_id="suite-one",
            cases=(SimpleNamespace(case_id="case-one"),),
        ),
        corpus=SimpleNamespace(
            cases=(
                SimpleNamespace(
                    suite_id="suite-one",
                    assertions=(
                        ModelJudgeAssertionSpec(
                            id="quality",
                            evaluator_key="judge-one",
                            rubric="The answer is correct.",
                            rubric_version="v1",
                        ),
                    ),
                ),
            ),
        ),
    )
    current_profiles = (
        SimpleNamespace(
            key="judge-one",
            revision=revision_a,
            comparison_revision=comparison_a,
        ),
        SimpleNamespace(
            key="judge-two",
            revision=revision_b,
            comparison_revision=comparison_b,
        ),
    )
    registration = SimpleNamespace(catalog_entry=SimpleNamespace(judge_profiles=current_profiles))

    assert evals_worker_module._accepted_authored_work_matches_runtime(
        lease,
        compiled,
        registration,
    )

    registration.catalog_entry.judge_profiles = (
        current_profiles[0],
        SimpleNamespace(
            key="judge-two",
            revision="sha256:" + "e" * 64,
            comparison_revision=comparison_b,
        ),
    )
    assert not evals_worker_module._accepted_authored_work_matches_runtime(
        lease,
        compiled,
        registration,
    )


def test_attached_worker_recovers_interrupted_model_judge_under_a_new_fence(tmp_path) -> None:
    judge_provider = _RecoveringJudgeProvider()
    judge_app = CayuApp(enable_logging=False)
    judge_app.register_provider(judge_provider, default=True)
    judge_app.register_agent(AgentSpec(name="judge", model="judge-model"))
    judge = ModelJudgeTarget(
        key="quality-judge",
        app=judge_app,
        agent_name="judge",
    )
    candidate_provider = _provider(trials=2)
    target = _target(candidate_provider, model_judges=(judge,))
    corpus = _model_judge_corpus(judge)
    database = tmp_path / "evals.db"
    store = SQLiteEvalStore(database)
    recovery_store = None
    run_id = None
    try:
        with TestClient(_server(target, store)) as client:
            assert (
                client.post(
                    "/api/evals/corpora",
                    headers=_AUTH_HEADERS,
                    json=corpus.model_dump(mode="json"),
                ).status_code
                == 201
            )
            admitted = client.post(
                "/api/evals/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "model-judge-recovery"},
                json={
                    "corpus_revision": corpus.revision,
                    "suite_id": corpus.suites[0].id,
                    "expected_execution_profile_revision": _execution_profile_revision(client),
                },
            )
            assert admitted.status_code == 202
            run_id = admitted.json()["spec"]["run_id"]
            assert judge_provider.started.wait(timeout=2)

        assert run_id is not None
        assert judge_provider.cancelled.wait(timeout=2)
        asyncio.run(store.close())
        recovery_store = SQLiteEvalStore(database, schema_mode=SchemaMode.VALIDATE)
        released = asyncio.run(recovery_store.load_run(run_id))
        assert released is not None
        assert released.status is EvalRunStatus.QUEUED
        assert released.attempt_count == 1
        assert released.ownership is None

        with TestClient(_server(target, recovery_store)) as client:
            terminal = _wait_for_terminal(client, run_id)
            assert terminal["status"] == "completed"
            assert terminal["attempt_count"] == 2
            assert terminal["result"]["status"] == "passed"
            published = client.get(
                f"/api/evals/runs/{run_id}/result",
                headers=_AUTH_HEADERS,
            )
            assert published.status_code == 200
            judge_detail = published.json()["result"]["run"]["cases"][0]["trials"][0]["assertions"][
                0
            ]["detail"]
            assert judge_detail["usage"] == {
                "model_steps": 1,
                "input_tokens": "2",
                "output_tokens": "1",
                "total_tokens": "3",
            }
            assert judge_detail["cost"] == {
                "availability": "unavailable",
                "currency": None,
                "estimated_cost": None,
                "priced_model_steps": None,
                "unpriced_model_steps": None,
            }

        assert len(candidate_provider.requests) == 2
        assert judge_provider.request_count == 2
    finally:
        if recovery_store is not None:
            asyncio.run(recovery_store.close())
        else:
            asyncio.run(store.close())


def test_complex_corpus_import_cannot_starve_an_active_eval_lease(
    tmp_path,
    monkeypatch,
) -> None:
    provider = _BlockingProvider()
    target = _target(provider)
    active_corpus = _corpus(trials=1)
    template = active_corpus.cases[0]
    imported_corpus = EvalCorpusDocument.create(
        target_key=active_corpus.target_key,
        evidence_policy=active_corpus.evidence_policy,
        pricing_profile=active_corpus.pricing_profile,
        suites=active_corpus.suites,
        cases=tuple(
            EvalCaseSpec.create(
                id=f"concurrent-case-{index:04d}",
                suite_id=template.suite_id,
                name=f"Concurrent case {index}",
                source=template.source,
                input=template.input,
                assertions=template.assertions,
            )
            for index in range(192)
        ),
    )
    database = tmp_path / "evals.db"
    store = SQLiteEvalStore(database)
    competing_store = SQLiteEvalStore(database)
    preparation_started = threading.Event()
    release_preparation = threading.Event()
    original_prepare = evals_sqlite_module._prepare_corpus_catalog_for_store
    import_outcome: dict[str, object] = {}

    def blocking_prepare(*args, **kwargs):
        preparation_started.set()
        if not release_preparation.wait(timeout=15):
            raise AssertionError("Timed out waiting to release corpus preparation.")
        return original_prepare(*args, **kwargs)

    config = ServerConfig.protected(
        _authenticate,
        dashboard=DashboardConfig(enabled=False),
        evals=_evals_config(
            target,
            store,
            lease_seconds=1,
            poll_interval_seconds=0.02,
        ),
    )
    import_thread = None
    try:
        with TestClient(create_server(target.app, config=config)) as client:
            assert (
                client.post(
                    "/api/evals/corpora",
                    headers=_AUTH_HEADERS,
                    json=active_corpus.model_dump(mode="json"),
                ).status_code
                == 201
            )
            admitted = client.post(
                "/api/evals/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "active-during-import"},
                json={
                    "corpus_revision": active_corpus.revision,
                    "suite_id": active_corpus.suites[0].id,
                    "expected_execution_profile_revision": _execution_profile_revision(client),
                },
            )
            assert admitted.status_code == 202
            assert provider.started.wait(timeout=10)

            monkeypatch.setattr(
                evals_sqlite_module,
                "_prepare_corpus_catalog_for_store",
                blocking_prepare,
            )

            def import_complex_corpus() -> None:
                response = client.post(
                    "/api/evals/corpora",
                    headers=_AUTH_HEADERS,
                    json=imported_corpus.model_dump(mode="json"),
                )
                import_outcome["status_code"] = response.status_code

            import_thread = threading.Thread(target=import_complex_corpus)
            import_thread.start()
            assert preparation_started.wait(timeout=10)

            time.sleep(1.1)
            competing_lease = asyncio.run(
                competing_store.claim_run(target_key=target.key, lease_seconds=5)
            )
            assert competing_lease is None

            release_preparation.set()
            import_thread.join(timeout=15)
            assert not import_thread.is_alive()
            assert import_outcome == {"status_code": 201}
    finally:
        release_preparation.set()
        if import_thread is not None:
            import_thread.join(timeout=15)
        asyncio.run(competing_store.close())
        asyncio.run(store.close())


def test_sqlite_eval_heartbeats_have_capacity_when_default_executor_is_saturated(
    tmp_path,
) -> None:
    provider = _BlockingProvider()
    target = _target(provider)
    corpus = _corpus(trials=1)
    database = tmp_path / "evals.db"
    store = SQLiteEvalStore(database)
    competing_store = SQLiteEvalStore(database)

    async def exercise() -> None:
        await store.save_corpus(corpus, redact_json=target.app.redact_json)
        request = EvalRunRequest(
            run_id="eval-reserved-executor",
            idempotency_key="sha256:" + ("1" * 64),
            corpus_revision=corpus.revision,
            target_key=target.key,
            suite_id=corpus.suites[0].id,
            suite_revision=corpus.suites[0].revision,
            max_concurrency=1,
            invocation=await _bound_eval_invocation(target),
        )
        await store.admit_run(request, redact_json=target.app.redact_json)

        loop = asyncio.get_running_loop()
        worker_count = 2
        loop.set_default_executor(
            ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="cayu-evals-saturation-test",
            )
        )
        coordinator = evals_worker_module.EvalRunCoordinator(
            _evals_config(
                target,
                store,
                lease_seconds=1,
                poll_interval_seconds=0.02,
            )
        )
        release_workers = threading.Event()
        workers_started = 0
        workers_started_lock = threading.Lock()
        all_workers_started = threading.Event()
        blockers: list[asyncio.Task] = []
        competing_lease = None

        def occupy_default_executor() -> None:
            nonlocal workers_started
            with workers_started_lock:
                workers_started += 1
                if workers_started == worker_count:
                    all_workers_started.set()
            if not release_workers.wait(timeout=5):
                raise AssertionError("Timed out releasing saturated executor workers.")

        coordinator.start()
        try:
            deadline = loop.time() + 2
            while not provider.started.is_set() and loop.time() < deadline:
                await asyncio.sleep(0.01)
            assert provider.started.is_set()

            blockers = [
                asyncio.create_task(asyncio.to_thread(occupy_default_executor))
                for _ in range(worker_count)
            ]
            deadline = loop.time() + 2
            while not all_workers_started.is_set() and loop.time() < deadline:
                await asyncio.sleep(0.01)
            assert all_workers_started.is_set()

            await asyncio.sleep(1.1)
            competing_lease = await competing_store.claim_run(
                target_key=target.key,
                lease_seconds=5,
            )
            assert competing_lease is None
        finally:
            release_workers.set()
            await asyncio.gather(*blockers, return_exceptions=True)
            if competing_lease is not None:
                await competing_store.release_run(competing_lease.claim)
            await coordinator.stop()
            await competing_store.close()
            await store.close()

    asyncio.run(exercise())


def test_result_publication_heartbeats_until_the_terminal_commit(
    tmp_path,
    monkeypatch,
) -> None:
    target = _target(_provider(trials=1))
    corpus = _corpus(trials=1)
    database = tmp_path / "evals.db"
    store = SQLiteEvalStore(database)
    competing_store = SQLiteEvalStore(database)
    preparation_started = threading.Event()
    release_preparation = threading.Event()
    original_prepare = evals_sqlite_module._prepare_result_for_store

    def blocking_prepare(*args, **kwargs):
        preparation_started.set()
        if not release_preparation.wait(timeout=5):
            raise AssertionError("Timed out releasing eval result preparation.")
        return original_prepare(*args, **kwargs)

    config = ServerConfig.protected(
        _authenticate,
        dashboard=DashboardConfig(enabled=False),
        evals=_evals_config(
            target,
            store,
            lease_seconds=1,
            poll_interval_seconds=0.02,
        ),
    )
    monkeypatch.setattr(
        evals_sqlite_module,
        "_prepare_result_for_store",
        blocking_prepare,
    )
    try:
        with TestClient(create_server(target.app, config=config)) as client:
            assert (
                client.post(
                    "/api/evals/corpora",
                    headers=_AUTH_HEADERS,
                    json=corpus.model_dump(mode="json"),
                ).status_code
                == 201
            )
            admitted = client.post(
                "/api/evals/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "publication-heartbeat"},
                json={
                    "corpus_revision": corpus.revision,
                    "suite_id": corpus.suites[0].id,
                    "expected_execution_profile_revision": _execution_profile_revision(client),
                },
            )
            assert admitted.status_code == 202
            run_id = admitted.json()["spec"]["run_id"]
            assert preparation_started.wait(timeout=2)

            time.sleep(1.1)
            competing_lease = asyncio.run(
                competing_store.claim_run(target_key=target.key, lease_seconds=5)
            )
            assert competing_lease is None

            release_preparation.set()
            terminal = _wait_for_terminal(client, run_id)
            assert terminal["status"] == "completed"
    finally:
        release_preparation.set()
        asyncio.run(competing_store.close())
        asyncio.run(store.close())


def test_transient_result_store_contention_requeues_without_terminal_failure(
    tmp_path,
    monkeypatch,
) -> None:
    async def exercise() -> None:
        target = _target(_provider(trials=1))
        corpus = _corpus(trials=1)
        result = await run_corpus_suite(
            target,
            corpus,
            corpus.suites[0].id,
            max_concurrency=1,
        )
        store = SQLiteEvalStore(tmp_path / "result.db")
        try:
            await store.save_corpus(corpus, redact_json=target.app.redact_json)
            await store.admit_run(
                EvalRunRequest(
                    run_id="result-contention",
                    idempotency_key="sha256:" + "9" * 64,
                    corpus_revision=corpus.revision,
                    target_key=corpus.target_key,
                    suite_id=corpus.suites[0].id,
                    suite_revision=corpus.suites[0].revision,
                    max_concurrency=1,
                ),
                redact_json=target.app.redact_json,
            )
            lease = await store.claim_run(target_key=target.key)
            assert lease is not None
            coordinator = evals_worker_module.EvalRunCoordinator(_evals_config(target, store))

            async def execution_outcome():
                return result

            async def contended_result_publication(*_args, **_kwargs):
                raise EvalStoreTransientContention("result publication deferred")

            monkeypatch.setattr(store, "publish_result", contended_result_publication)
            execution = asyncio.create_task(execution_outcome())
            await asyncio.wait({execution})
            await coordinator._publish_execution_outcome(lease.claim, target, execution)

            requeued = await store.load_run(lease.claim.run_id)
            assert requeued is not None
            assert requeued.status is EvalRunStatus.QUEUED
            assert requeued.failure_code is None
        finally:
            await store.close()

    asyncio.run(exercise())


def test_checkpoint_contention_exhaustion_retries_without_trial_redispatch(
    tmp_path,
    monkeypatch,
    caplog,
) -> None:
    async def exercise() -> None:
        provider = _provider(trials=1)
        target = _target(provider)
        corpus = _corpus(trials=1)
        path = tmp_path / "checkpoint-retry.db"
        store = SQLiteEvalStore(
            path,
            writer_contention_policy=SQLiteEvalWriterContentionPolicy(
                max_wait_seconds=0.08,
                lock_attempt_seconds=0.02,
                initial_backoff_seconds=0.005,
                max_backoff_seconds=0.02,
            ),
        )
        blocker = sqlite3.connect(path)
        release_blocker = None
        checkpoint_attempts = 0
        original_save = store.save_trial_checkpoint

        async def release_after_exhaustion() -> None:
            await asyncio.sleep(0.15)
            blocker.commit()

        async def contend_first_checkpoint(*args, **kwargs):
            nonlocal checkpoint_attempts, release_blocker
            checkpoint_attempts += 1
            if checkpoint_attempts == 1:
                blocker.execute("BEGIN IMMEDIATE")
                release_blocker = asyncio.create_task(release_after_exhaustion())
            return await original_save(*args, **kwargs)

        monkeypatch.setattr(store, "save_trial_checkpoint", contend_first_checkpoint)
        try:
            await store.save_corpus(corpus, redact_json=target.app.redact_json)
            await store.admit_run(
                EvalRunRequest(
                    run_id="checkpoint-contention-retry",
                    idempotency_key="sha256:" + "7" * 64,
                    corpus_revision=corpus.revision,
                    target_key=target.key,
                    suite_id=corpus.suites[0].id,
                    suite_revision=corpus.suites[0].revision,
                    max_concurrency=1,
                    invocation=await _bound_eval_invocation(target),
                ),
                redact_json=target.app.redact_json,
            )
            lease = await store.claim_run(target_key=target.key, lease_seconds=5)
            assert lease is not None
            coordinator = evals_worker_module.EvalRunCoordinator(_evals_config(target, store))

            await coordinator._run_lease(lease)

            completed = await store.load_run(lease.run.id)
            assert completed is not None
            assert completed.status is EvalRunStatus.COMPLETED
            assert completed.attempt_count == 1
            assert checkpoint_attempts >= 2
            assert len(provider.requests) == 1
        finally:
            blocker.rollback()
            if release_blocker is not None:
                await asyncio.gather(release_blocker, return_exceptions=True)
            blocker.close()
            await store.close()

    caplog.set_level(logging.INFO, logger=evals_sqlite_module.__name__)
    asyncio.run(exercise())
    assert "sqlite_writer.contention_exhausted" in {
        getattr(record, "cayu_eval_store_event", None) for record in caplog.records
    }


def test_claim_monitor_retries_transient_heartbeat_contention(tmp_path) -> None:
    async def exercise() -> None:
        target = _target(_provider(trials=1))
        corpus = _corpus(trials=1)
        store = SQLiteEvalStore(
            tmp_path / "heartbeat-contention.db",
            writer_contention_policy=SQLiteEvalWriterContentionPolicy(
                max_wait_seconds=0.08,
                lock_attempt_seconds=0.02,
                initial_backoff_seconds=0.005,
                max_backoff_seconds=0.02,
            ),
        )
        blocker = sqlite3.connect(store.path)
        monitor = None
        try:
            await store.save_corpus(corpus, redact_json=target.app.redact_json)
            await store.admit_run(
                EvalRunRequest(
                    run_id="heartbeat-contention",
                    idempotency_key="sha256:" + "8" * 64,
                    corpus_revision=corpus.revision,
                    target_key=corpus.target_key,
                    suite_id=corpus.suites[0].id,
                    suite_revision=corpus.suites[0].revision,
                    max_concurrency=1,
                ),
                redact_json=target.app.redact_json,
            )
            lease = await store.claim_run(lease_seconds=2)
            assert lease is not None
            await store._run(lambda connection: connection.execute("PRAGMA busy_timeout = 50"))
            coordinator = evals_worker_module.EvalRunCoordinator(
                _evals_config(
                    target,
                    store,
                    lease_seconds=2,
                    poll_interval_seconds=0.01,
                )
            )
            blocker.execute("BEGIN IMMEDIATE")
            monitor = asyncio.create_task(coordinator._monitor_claim(lease.claim))
            await asyncio.sleep(0.75)
            assert not monitor.done()

            blocker.commit()
            await store.request_cancel(lease.run.id)
            outcome = await asyncio.wait_for(monitor, timeout=1.5)
            assert outcome is evals_worker_module._ClaimMonitorOutcome.CANCELLING
            await store.finish_cancel(lease.claim)
        finally:
            if monitor is not None and not monitor.done():
                monitor.cancel()
                await asyncio.gather(monitor, return_exceptions=True)
            blocker.rollback()
            blocker.close()
            await store.close()

    asyncio.run(exercise())


def test_result_projection_cannot_expire_a_completed_provider_lease(
    tmp_path,
    monkeypatch,
) -> None:
    provider = _provider(trials=1)
    target = _target(provider)
    corpus = _corpus(trials=1)
    database = tmp_path / "evals.db"
    store = SQLiteEvalStore(database)
    competing_store = SQLiteEvalStore(database, schema_mode=SchemaMode.VALIDATE)
    projection_started = threading.Event()
    release_projection = threading.Event()
    original_publish = execution_module._publish_eval_run_with_trial_public_data
    competing_lease = None

    def blocking_publish(*args, **kwargs):
        projection_started.set()
        if not release_projection.wait(timeout=5):
            raise AssertionError("Timed out releasing eval result projection.")
        return original_publish(*args, **kwargs)

    monkeypatch.setattr(
        execution_module,
        "_publish_eval_run_with_trial_public_data",
        blocking_publish,
    )
    config = ServerConfig.protected(
        _authenticate,
        dashboard=DashboardConfig(enabled=False),
        evals=_evals_config(
            target,
            store,
            lease_seconds=1,
            poll_interval_seconds=0.02,
        ),
    )
    try:
        with TestClient(create_server(target.app, config=config)) as client:
            assert (
                client.post(
                    "/api/evals/corpora",
                    headers=_AUTH_HEADERS,
                    json=corpus.model_dump(mode="json"),
                ).status_code
                == 201
            )
            admitted = client.post(
                "/api/evals/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "projection-heartbeat"},
                json={
                    "corpus_revision": corpus.revision,
                    "suite_id": corpus.suites[0].id,
                    "expected_execution_profile_revision": _execution_profile_revision(client),
                },
            )
            assert admitted.status_code == 202
            run_id = admitted.json()["spec"]["run_id"]
            assert projection_started.wait(timeout=2)

            time.sleep(1.1)
            competing_lease = asyncio.run(
                competing_store.claim_run(target_key=target.key, lease_seconds=5)
            )
            assert competing_lease is None

            release_projection.set()
            terminal = _wait_for_terminal(client, run_id)
            assert terminal["status"] == "completed"
            assert len(provider.requests) == 1
    finally:
        release_projection.set()
        if competing_lease is not None:
            asyncio.run(competing_store.release_run(competing_lease.claim))
        asyncio.run(competing_store.close())
        asyncio.run(store.close())


def test_eval_preflight_heartbeats_ownership_before_provider_dispatch(
    tmp_path,
    monkeypatch,
) -> None:
    provider = _provider(trials=1)
    target = _target(provider)
    corpus = _corpus(trials=1)
    database = tmp_path / "evals.db"
    store = SQLiteEvalStore(database)
    competing_store = SQLiteEvalStore(database, schema_mode=SchemaMode.VALIDATE)
    compile_started = threading.Event()
    release_compile = threading.Event()
    original_compile = evals_worker_module.compile_corpus_suite

    def blocking_compile(*args, **kwargs):
        compile_started.set()
        if not release_compile.wait(timeout=5):
            raise AssertionError("Timed out waiting to release eval preflight compilation.")
        return original_compile(*args, **kwargs)

    monkeypatch.setattr(evals_worker_module, "compile_corpus_suite", blocking_compile)
    config = ServerConfig.protected(
        _authenticate,
        dashboard=DashboardConfig(enabled=False),
        evals=_evals_config(
            target,
            store,
            lease_seconds=1,
            poll_interval_seconds=0.02,
        ),
    )
    try:
        with TestClient(create_server(target.app, config=config)) as client:
            assert (
                client.post(
                    "/api/evals/corpora",
                    headers=_AUTH_HEADERS,
                    json=corpus.model_dump(mode="json"),
                ).status_code
                == 201
            )
            admitted = client.post(
                "/api/evals/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "preflight-heartbeat"},
                json={
                    "corpus_revision": corpus.revision,
                    "suite_id": corpus.suites[0].id,
                    "expected_execution_profile_revision": _execution_profile_revision(client),
                },
            )
            run_id = admitted.json()["spec"]["run_id"]
            assert compile_started.wait(timeout=2)

            time.sleep(1.1)
            competing_lease = asyncio.run(
                competing_store.claim_run(target_key=target.key, lease_seconds=5)
            )
            assert competing_lease is None
            assert provider.requests == []

            release_compile.set()
            terminal = _wait_for_terminal(client, run_id)
            assert terminal["status"] == "completed"
            assert len(provider.requests) == 1
    finally:
        release_compile.set()
        asyncio.run(competing_store.close())
        asyncio.run(store.close())


def test_eval_preflight_rechecks_ownership_before_provider_dispatch(
    tmp_path,
    monkeypatch,
) -> None:
    provider = _provider(trials=1)
    target = _target(provider)
    corpus = _corpus(trials=1)
    database = tmp_path / "evals.db"
    store = SQLiteEvalStore(database)
    competing_store = SQLiteEvalStore(database, schema_mode=SchemaMode.VALIDATE)
    compile_started = threading.Event()
    release_compile = threading.Event()
    refresh_attempted = threading.Event()
    original_compile = evals_worker_module.compile_corpus_suite
    original_heartbeat = store.heartbeat_run
    competing_lease = None

    def blocking_compile(*args, **kwargs):
        compile_started.set()
        if not release_compile.wait(timeout=5):
            raise AssertionError("Timed out waiting to release eval preflight compilation.")
        return original_compile(*args, **kwargs)

    async def stalled_monitor(_self, _claim):
        await asyncio.Future()

    async def observed_heartbeat(claim, *, extend_seconds=300):
        try:
            return await original_heartbeat(claim, extend_seconds=extend_seconds)
        finally:
            refresh_attempted.set()

    monkeypatch.setattr(evals_worker_module, "compile_corpus_suite", blocking_compile)
    monkeypatch.setattr(
        evals_worker_module.EvalRunCoordinator,
        "_monitor_claim",
        stalled_monitor,
    )
    monkeypatch.setattr(store, "heartbeat_run", observed_heartbeat)
    config = ServerConfig.protected(
        _authenticate,
        dashboard=DashboardConfig(enabled=False),
        evals=_evals_config(
            target,
            store,
            lease_seconds=1,
            poll_interval_seconds=0.02,
        ),
    )
    try:
        with TestClient(create_server(target.app, config=config)) as client:
            assert (
                client.post(
                    "/api/evals/corpora",
                    headers=_AUTH_HEADERS,
                    json=corpus.model_dump(mode="json"),
                ).status_code
                == 201
            )
            admitted = client.post(
                "/api/evals/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "preflight-fence"},
                json={
                    "corpus_revision": corpus.revision,
                    "suite_id": corpus.suites[0].id,
                    "expected_execution_profile_revision": _execution_profile_revision(client),
                },
            )
            assert admitted.status_code == 202
            assert compile_started.wait(timeout=2)

            time.sleep(1.1)
            competing_lease = asyncio.run(
                competing_store.claim_run(target_key=target.key, lease_seconds=5)
            )
            assert competing_lease is not None
            assert competing_lease.claim.epoch == 2

            release_compile.set()
            assert refresh_attempted.wait(timeout=2)
            time.sleep(0.05)
            assert provider.requests == []
    finally:
        release_compile.set()
        if competing_lease is not None:
            asyncio.run(competing_store.release_run(competing_lease.claim))
        asyncio.run(competing_store.close())
        asyncio.run(store.close())


def test_claim_monitor_uses_adaptive_lightweight_observations(
    tmp_path,
    monkeypatch,
) -> None:
    async def exercise() -> None:
        target = _target(_provider(trials=1))
        corpus = _corpus(trials=1)
        suite = corpus.suites[0]
        path = tmp_path / "evals.db"
        store = SQLiteEvalStore(path)
        competing_store = SQLiteEvalStore(path, schema_mode=SchemaMode.VALIDATE)
        calls = {
            "full_load": 0,
            "observation_load": 0,
            "full_heartbeat": 0,
            "observation_heartbeat": 0,
        }
        original_load_run = store.load_run
        original_load_observation = store.load_run_observation
        original_heartbeat = store.heartbeat_run
        original_heartbeat_observation = store.heartbeat_run_observation
        monitor = None

        async def counted_load_run(run_id):
            calls["full_load"] += 1
            return await original_load_run(run_id)

        async def counted_load_observation(run_id):
            calls["observation_load"] += 1
            return await original_load_observation(run_id)

        async def counted_heartbeat(claim, *, extend_seconds=300):
            calls["full_heartbeat"] += 1
            return await original_heartbeat(claim, extend_seconds=extend_seconds)

        async def counted_heartbeat_observation(claim, *, extend_seconds=300):
            calls["observation_heartbeat"] += 1
            return await original_heartbeat_observation(
                claim,
                extend_seconds=extend_seconds,
            )

        monkeypatch.setattr(store, "load_run", counted_load_run)
        monkeypatch.setattr(store, "load_run_observation", counted_load_observation)
        monkeypatch.setattr(store, "heartbeat_run", counted_heartbeat)
        monkeypatch.setattr(
            store,
            "heartbeat_run_observation",
            counted_heartbeat_observation,
        )
        try:
            await store.save_corpus(corpus, redact_json=target.app.redact_json)
            await store.admit_run(
                EvalRunRequest(
                    run_id="lightweight-monitor",
                    idempotency_key="sha256:" + "1" * 64,
                    corpus_revision=corpus.revision,
                    target_key=corpus.target_key,
                    suite_id=suite.id,
                    suite_revision=suite.revision,
                    max_concurrency=1,
                ),
                redact_json=target.app.redact_json,
            )
            lease = await store.claim_run(lease_seconds=1)
            assert lease is not None
            coordinator = evals_worker_module.EvalRunCoordinator(
                _evals_config(
                    target,
                    store,
                    lease_seconds=1,
                    poll_interval_seconds=0.01,
                )
            )
            monitor = asyncio.create_task(coordinator._monitor_claim(lease.claim))
            await asyncio.sleep(0.8)
            assert calls["full_load"] == 0
            assert calls["full_heartbeat"] == 0
            assert 2 <= calls["observation_load"] <= 5
            assert calls["observation_heartbeat"] >= 2

            cancellation_started_at = asyncio.get_running_loop().time()
            await competing_store.request_cancel(lease.run.id)
            outcome = await asyncio.wait_for(monitor, timeout=0.4)
            assert asyncio.get_running_loop().time() - cancellation_started_at < 0.4
            assert outcome is evals_worker_module._ClaimMonitorOutcome.CANCELLING
            await store.finish_cancel(lease.claim)
        finally:
            if monitor is not None and not monitor.done():
                monitor.cancel()
                await asyncio.gather(monitor, return_exceptions=True)
            await competing_store.close()
            await store.close()

    asyncio.run(exercise())


def test_shutdown_releases_owned_eval_for_restart_recovery(tmp_path) -> None:
    provider = _BlockingProvider()
    target = _target(provider)
    corpus = _corpus(trials=1)
    store = SQLiteEvalStore(tmp_path / "evals.db")
    try:
        run_id = None
        with TestClient(_server(target, store)) as client:
            assert (
                client.post(
                    "/api/evals/corpora",
                    headers=_AUTH_HEADERS,
                    json=corpus.model_dump(mode="json"),
                ).status_code
                == 201
            )
            admitted = client.post(
                "/api/evals/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "restart-release"},
                json={
                    "corpus_revision": corpus.revision,
                    "suite_id": corpus.suites[0].id,
                    "expected_execution_profile_revision": _execution_profile_revision(client),
                },
            )
            run_id = admitted.json()["spec"]["run_id"]
            assert provider.started.wait(timeout=2)

        assert run_id is not None
        record = asyncio.run(store.load_run(run_id))
        assert record is not None
        assert record.status is EvalRunStatus.QUEUED
        assert record.ownership is None
        assert provider.cancelled.wait(timeout=2)
    finally:
        asyncio.run(store.close())


def test_restarted_worker_recreates_persisted_http_provenance_and_run_bounds(tmp_path) -> None:
    provider = _provider(trials=1)
    target = _target(provider)
    corpus = _corpus(trials=1)
    database = tmp_path / "evals.db"

    async def exercise() -> None:
        first = SQLiteEvalStore(database)
        invocation = EvalRunInvocation(
            source=SessionExecutionSource.HTTP_RUN,
            origin=InvocationOrigin(
                trust=InvocationOriginTrust.SERVER_VERIFIED,
                subject="restart-operator",
                tenant="restart-tenant",
            ),
            max_steps=1,
            limits=RunLimits(max_total_tokens=100, scope="run"),
        )
        invocation = await _bound_eval_invocation(target, invocation)
        request = EvalRunRequest(
            run_id="eval-restart-provenance",
            idempotency_key="sha256:" + "9" * 64,
            corpus_revision=corpus.revision,
            target_key=target.key,
            suite_id=corpus.suites[0].id,
            suite_revision=corpus.suites[0].revision,
            max_concurrency=1,
            invocation=invocation,
        )
        await first.save_corpus(corpus, redact_json=target.app.redact_json)
        await first.admit_run(request, redact_json=target.app.redact_json)
        await first.close()

        reopened = SQLiteEvalStore(database, schema_mode=SchemaMode.VALIDATE)
        coordinator = evals_worker_module.EvalRunCoordinator(_evals_config(target, reopened))
        coordinator.start()
        try:
            deadline = asyncio.get_running_loop().time() + 5
            while asyncio.get_running_loop().time() < deadline:
                record = await reopened.load_run(request.run_id)
                assert record is not None
                if record.status in {
                    EvalRunStatus.COMPLETED,
                    EvalRunStatus.FAILED,
                    EvalRunStatus.CANCELLED,
                }:
                    break
                await asyncio.sleep(0.01)
            assert record.status is EvalRunStatus.COMPLETED
            sessions = (await target.app.session_store.list_sessions()).sessions
            assert len(sessions) == 1
            session = sessions[0]
            assert session.invocation.source is SessionExecutionSource.HTTP_RUN
            assert session.invocation.origin == InvocationOrigin(
                trust=InvocationOriginTrust.SERVER_VERIFIED,
                subject="restart-operator",
                tenant="restart-tenant",
            )
        finally:
            await coordinator.stop()
            await reopened.close()

    asyncio.run(exercise())


def test_restarted_worker_rejects_changed_execution_profile_before_provider_dispatch(
    tmp_path,
) -> None:
    provider = _provider(trials=1)
    target = _target(provider)
    corpus = _corpus(trials=1)
    store = SQLiteEvalStore(tmp_path / "profile-drift.db")

    async def exercise() -> None:
        invocation = await _bound_eval_invocation(target)
        request = EvalRunRequest(
            run_id="eval-profile-drift",
            idempotency_key="sha256:" + "4" * 64,
            corpus_revision=corpus.revision,
            target_key=target.key,
            suite_id=corpus.suites[0].id,
            suite_revision=corpus.suites[0].revision,
            max_concurrency=1,
            invocation=invocation,
        )
        await store.save_corpus(corpus, redact_json=target.app.redact_json)
        await store.admit_run(request, redact_json=target.app.redact_json)

        target.app.register_agent(AgentSpec(name="late-agent", model="fixture-model"))
        coordinator = evals_worker_module.EvalRunCoordinator(_evals_config(target, store))
        coordinator.start()
        try:
            deadline = asyncio.get_running_loop().time() + 5
            while asyncio.get_running_loop().time() < deadline:
                record = await store.load_run(request.run_id)
                assert record is not None
                if record.status in {
                    EvalRunStatus.COMPLETED,
                    EvalRunStatus.FAILED,
                    EvalRunStatus.CANCELLED,
                }:
                    break
                await asyncio.sleep(0.01)
            assert record.status is EvalRunStatus.FAILED
            assert record.failure_code == "target_unavailable"
            assert provider.requests == []
        finally:
            await coordinator.stop()
            await store.close()

    asyncio.run(exercise())


def test_restarted_worker_rejects_changed_target_bootstrap_before_provider_dispatch(
    tmp_path,
) -> None:
    provider = _provider(trials=1)
    target = _target(provider)
    corpus = _corpus(trials=1)
    store = SQLiteEvalStore(tmp_path / "target-material-drift.db")
    changed_target = CorpusTarget(
        key=target.key,
        app=target.app,
        request_base=target.request_base,
        bootstrap_messages=(Message.text("system", "A changed candidate bootstrap."),),
        application_release_id=target.application_release_id,
        evidence_policy=target.evidence_policy,
        price_book=target.price_book,
        model_judges=target.model_judges,
        limits=target.limits,
    )

    async def exercise() -> None:
        invocation = await _bound_eval_invocation(target)
        request = EvalRunRequest(
            run_id="eval-target-material-drift",
            idempotency_key="sha256:" + "7" * 64,
            corpus_revision=corpus.revision,
            target_key=target.key,
            suite_id=corpus.suites[0].id,
            suite_revision=corpus.suites[0].revision,
            max_concurrency=1,
            invocation=invocation,
        )
        await store.save_corpus(corpus, redact_json=target.app.redact_json)
        await store.admit_run(request, redact_json=target.app.redact_json)

        coordinator = evals_worker_module.EvalRunCoordinator(_evals_config(changed_target, store))
        coordinator.start()
        try:
            deadline = asyncio.get_running_loop().time() + 5
            while asyncio.get_running_loop().time() < deadline:
                record = await store.load_run(request.run_id)
                assert record is not None
                if record.status in {
                    EvalRunStatus.COMPLETED,
                    EvalRunStatus.FAILED,
                    EvalRunStatus.CANCELLED,
                }:
                    break
                await asyncio.sleep(0.01)
            assert record.status is EvalRunStatus.FAILED
            assert record.failure_code == "target_unavailable"
            assert provider.requests == []
        finally:
            await coordinator.stop()
            await store.close()

    asyncio.run(exercise())


def test_worker_classifies_manifest_drift_after_preflight_as_target_unavailable(
    tmp_path,
    monkeypatch,
) -> None:
    provider = _provider(trials=1)
    target = _target(provider)
    corpus = _corpus(trials=1)
    store = SQLiteEvalStore(tmp_path / "manifest-drift-after-preflight.db")
    original_preflight = evals_worker_module.EvalRunCoordinator._preflight_lease

    async def drift_after_preflight(coordinator, lease, registration):
        prepared = await original_preflight(coordinator, lease, registration)
        if isinstance(prepared, evals_worker_module._PreparedEvalRun):
            target.app.register_agent(AgentSpec(name="late-agent", model="fixture-model"))
        return prepared

    monkeypatch.setattr(
        evals_worker_module.EvalRunCoordinator,
        "_preflight_lease",
        drift_after_preflight,
    )

    async def exercise() -> None:
        invocation = await _bound_eval_invocation(target)
        request = EvalRunRequest(
            run_id="eval-manifest-drift-after-preflight",
            idempotency_key="sha256:" + "5" * 64,
            corpus_revision=corpus.revision,
            target_key=target.key,
            suite_id=corpus.suites[0].id,
            suite_revision=corpus.suites[0].revision,
            max_concurrency=1,
            invocation=invocation,
        )
        await store.save_corpus(corpus, redact_json=target.app.redact_json)
        await store.admit_run(request, redact_json=target.app.redact_json)

        coordinator = evals_worker_module.EvalRunCoordinator(_evals_config(target, store))
        coordinator.start()
        try:
            deadline = asyncio.get_running_loop().time() + 5
            while asyncio.get_running_loop().time() < deadline:
                record = await store.load_run(request.run_id)
                assert record is not None
                if record.status in {
                    EvalRunStatus.COMPLETED,
                    EvalRunStatus.FAILED,
                    EvalRunStatus.CANCELLED,
                }:
                    break
                await asyncio.sleep(0.01)
            assert record.status is EvalRunStatus.FAILED
            assert record.failure_code == "target_unavailable"
            assert provider.requests == []
        finally:
            await coordinator.stop()
            await store.close()

    asyncio.run(exercise())


def test_worker_rechecks_manifest_before_each_fresh_trial(tmp_path, monkeypatch) -> None:
    provider = _provider(trials=2)
    target = _target(provider)
    corpus = _corpus(trials=2)
    store = SQLiteEvalStore(tmp_path / "manifest-drift-between-trials.db")
    original_stream = provider.stream

    async def drift_after_first_trial(request):
        async for event in original_stream(request):
            yield event
        if len(provider.requests) == 1:
            target.app.register_agent(AgentSpec(name="late-agent", model="fixture-model"))

    monkeypatch.setattr(provider, "stream", drift_after_first_trial)
    try:
        with TestClient(
            _server(
                target,
                store,
                execution_profile_policy=_repeatable_execution_policy(target),
            )
        ) as client:
            imported = client.post(
                "/api/evals/corpora",
                headers=_AUTH_HEADERS,
                json=corpus.model_dump(mode="json"),
            )
            assert imported.status_code == 201
            admitted = client.post(
                "/api/evals/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "manifest-drift-between-trials"},
                json={
                    "corpus_revision": corpus.revision,
                    "suite_id": corpus.suites[0].id,
                    "expected_execution_profile_revision": _execution_profile_revision(client),
                    "max_concurrency": 1,
                },
            )
            assert admitted.status_code == 202

            terminal = _wait_for_terminal(client, admitted.json()["spec"]["run_id"])
            assert terminal["status"] == "failed"
            assert terminal["failure_code"] == "target_unavailable"
            assert len(provider.requests) == 1
    finally:
        asyncio.run(store.close())


def test_shutdown_grace_bounds_a_stalled_durable_release(tmp_path, monkeypatch) -> None:
    provider = _BlockingProvider()
    target = _target(provider)
    corpus = _corpus(trials=1)
    database = tmp_path / "evals.db"
    store = SQLiteEvalStore(database)
    recovery_store = SQLiteEvalStore(database, schema_mode=SchemaMode.VALIDATE)

    async def exercise() -> None:
        await store.save_corpus(corpus, redact_json=target.app.redact_json)
        request = EvalRunRequest(
            run_id="bounded-shutdown-run",
            idempotency_key="sha256:" + "3" * 64,
            corpus_revision=corpus.revision,
            target_key=target.key,
            suite_id=corpus.suites[0].id,
            suite_revision=corpus.suites[0].revision,
            max_concurrency=1,
            invocation=await _bound_eval_invocation(target),
        )
        await store.admit_run(request, redact_json=target.app.redact_json)
        release_started = asyncio.Event()
        release_gate = asyncio.Event()
        original_release = store.release_run

        async def stalled_release(claim):
            release_started.set()
            await release_gate.wait()
            return await original_release(claim)

        monkeypatch.setattr(store, "release_run", stalled_release)
        coordinator = evals_worker_module.EvalRunCoordinator(
            _evals_config(
                target,
                store,
                lease_seconds=1,
                poll_interval_seconds=0.02,
                # Invocation cancellation now proves provider and lifecycle
                # quiescence before the eval lease can be released. Leave
                # enough room for that bounded ownership handoff, then stall
                # the release itself to exercise the shutdown deadline.
                shutdown_grace_seconds=0.5,
            )
        )
        recovery_lease = None
        coordinator.start()
        try:
            assert await asyncio.to_thread(provider.started.wait, 2)
            started_at = asyncio.get_running_loop().time()
            await asyncio.wait_for(coordinator.stop(), timeout=0.9)
            elapsed = asyncio.get_running_loop().time() - started_at
            assert release_started.is_set()
            assert elapsed < 0.8
            assert provider.cancelled.is_set()

            await asyncio.sleep(1.05)
            recovery_lease = await recovery_store.claim_run(
                target_key=target.key,
                lease_seconds=5,
            )
            assert recovery_lease is not None
            assert recovery_lease.claim.epoch == 2
        finally:
            release_gate.set()
            if recovery_lease is not None:
                await recovery_store.release_run(recovery_lease.claim)
            await asyncio.sleep(0)
            await recovery_store.close()
            await store.close()

    asyncio.run(exercise())
