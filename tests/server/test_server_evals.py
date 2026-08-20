from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor

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

import cayu.evals.execution as execution_module
import cayu.server.evals_worker as evals_worker_module
import cayu.storage.evals_sqlite as evals_sqlite_module
from cayu import AgentSpec, CayuApp, ModelJudgeTarget, ModelProvider, ModelRequest, ModelStreamEvent
from cayu.evals.corpus import EvalCaseSpec, EvalCorpusDocument
from cayu.evals.execution import run_corpus_suite
from cayu.evals.store import EvalRunRequest, EvalRunStatus, InMemoryEvalStore
from cayu.project_control_plane import (
    ProjectControlPlaneAccess,
    _create_project_control_plane_context,
)
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
from cayu.storage.evals_sqlite import SQLiteEvalStore
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


def _server(target, store):
    return create_server(
        target.app,
        config=ServerConfig.protected(
            _authenticate,
            dashboard=DashboardConfig(enabled=False),
            evals=_evals_config(target, store),
        ),
    )


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
        assert request_schemas["/api/evals/runs"] == EvalRunCreateRequest.model_json_schema()
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


def test_evals_api_imports_executes_compares_and_exports_deterministically(tmp_path) -> None:
    target = _target(_provider(trials=2))
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
                "max_concurrency": 1,
            }
            admitted = client.post(
                "/api/evals/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "eval-run-one"},
                json=request,
            )
            assert admitted.status_code == 202
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

            result = client.get(f"/api/evals/runs/{run_id}/result", headers=_AUTH_HEADERS)
            assert result.status_code == 200
            assert result.json()["result"]["run"]["status"] == "passed"
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
            assert json.loads(json_report.content)["revision"] == terminal["result"]["revision"]
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
                    "state": "gated",
                    "reason_code": "evaluation_promotion_not_configured",
                },
                "catalog_read": {"state": "ready", "reason_code": None},
                "catalog_write": {"state": "ready", "reason_code": None},
                "captured_result_persistence": {
                    "state": "unsupported",
                    "reason_code": "captured_result_persistence_not_available",
                },
                "scenario_conversion": {
                    "state": "unsupported",
                    "reason_code": "scenario_v2_not_available",
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
            assert response.status_code == 200
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
        EvalResultResponse(run=active_record, result=result)

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
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


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
    store = SQLiteEvalStore(tmp_path / "evals.db")
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
                },
            )
            run_id = admitted.json()["spec"]["run_id"]
            assert compile_started.wait(timeout=2)

            time.sleep(1.1)
            competing_lease = asyncio.run(store.claim_run(target_key=target.key, lease_seconds=5))
            assert competing_lease is None
            assert provider.requests == []

            release_compile.set()
            terminal = _wait_for_terminal(client, run_id)
            assert terminal["status"] == "completed"
            assert len(provider.requests) == 1
    finally:
        release_compile.set()
        asyncio.run(store.close())


def test_eval_preflight_rechecks_ownership_before_provider_dispatch(
    tmp_path,
    monkeypatch,
) -> None:
    provider = _provider(trials=1)
    target = _target(provider)
    corpus = _corpus(trials=1)
    store = SQLiteEvalStore(tmp_path / "evals.db")
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
                },
            )
            assert admitted.status_code == 202
            assert compile_started.wait(timeout=2)

            time.sleep(1.1)
            competing_lease = asyncio.run(store.claim_run(target_key=target.key, lease_seconds=5))
            assert competing_lease is not None
            assert competing_lease.claim.epoch == 2

            release_compile.set()
            assert refresh_attempted.wait(timeout=2)
            time.sleep(0.05)
            assert provider.requests == []
    finally:
        release_compile.set()
        if competing_lease is not None:
            asyncio.run(store.release_run(competing_lease.claim))
        asyncio.run(store.close())


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
                shutdown_grace_seconds=0.1,
            )
        )
        recovery_lease = None
        coordinator.start()
        try:
            assert await asyncio.to_thread(provider.started.wait, 2)
            started_at = asyncio.get_running_loop().time()
            await asyncio.wait_for(coordinator.stop(), timeout=0.5)
            elapsed = asyncio.get_running_loop().time() - started_at
            assert release_started.is_set()
            assert elapsed < 0.4
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
