from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi.testclient import TestClient
from tests.evals.test_corpus_execution import _corpus, _provider, _target
from tests.server.test_server_evals import _AUTH_HEADERS, _authenticate

from cayu import AgentSpec
from cayu.evals.store import EvalRunRequest, EvalRunStatus
from cayu.project_control_plane import (
    ProjectControlPlaneAccess,
    _create_project_control_plane_context,
)
from cayu.server import DashboardConfig, EvalsConfig, OpenAccess, ServerConfig, create_server
from cayu.storage.evals_sqlite import SQLiteEvalStore


def _wait_for_terminal(client: TestClient, run_id: str) -> dict:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get(f"/api/evals/runs/{run_id}", headers=_AUTH_HEADERS)
        assert response.status_code == 200
        run = response.json()
        if run["status"] in {"completed", "failed", "cancelled"}:
            return run
        time.sleep(0.01)
    raise AssertionError("Generated-target eval run did not terminalize.")


def test_project_context_generates_multi_agent_targets_and_keeps_all_work_target_scoped(
    tmp_path,
) -> None:
    provider = _provider(trials=2)
    app = _target(provider).app
    app.register_agent(AgentSpec(name="beta", model="fixture-model"))
    project_root = Path(__file__).resolve().parents[2]
    rooted_manifest = app.describe(project_root=project_root)
    assert rooted_manifest.fingerprint != app.describe().fingerprint
    store = SQLiteEvalStore(tmp_path / "evals.db")
    context = _create_project_control_plane_context(
        project_root=project_root,
        project_id="generated-project",
        configured_release_id="release-current",
        eval_store=store,
        store_backend="sqlite",
        store_source="project",
        access=ProjectControlPlaneAccess.AUTHENTICATED_PRODUCTION,
    )
    foreign_corpus = _corpus(target_key="foreign-target", trials=1)
    foreign_suite = foreign_corpus.suites[0]
    foreign_run = EvalRunRequest(
        run_id="foreign-run",
        idempotency_key="sha256:" + "f" * 64,
        corpus_revision=foreign_corpus.revision,
        target_key=foreign_corpus.target_key,
        suite_id=foreign_suite.id,
        suite_revision=foreign_suite.revision,
        max_concurrency=1,
    )

    async def seed_foreign_target() -> None:
        await store.save_corpus(foreign_corpus, redact_json=app.redact_json)
        await store.admit_run(foreign_run, redact_json=app.redact_json)

    asyncio.run(seed_foreign_target())
    server = create_server(
        app,
        config=ServerConfig.protected(
            _authenticate,
            dashboard=DashboardConfig(enabled=False),
        ),
        project_context=context,
    )
    with TestClient(server) as client:
        target_response = client.get("/api/evals/targets", headers=_AUTH_HEADERS)
        assert target_response.status_code == 200
        target_catalog = target_response.json()
        assert [item["agent_name"] for item in target_catalog["items"]] == ["agent", "beta"]
        assert {item["profile_id"] for item in target_catalog["items"]} == {"default"}
        assert {item["source"] for item in target_catalog["items"]} == {"generated"}
        assert {item["app_manifest_fingerprint"] for item in target_catalog["items"]} == {
            rooted_manifest.fingerprint
        }
        catalog_by_agent = {item["agent_name"]: item for item in target_catalog["items"]}
        target_keys = {item["agent_name"]: item["target_key"] for item in target_catalog["items"]}

        corpora = {
            agent_name: _corpus(target_key=target_key, trials=1)
            for agent_name, target_key in target_keys.items()
        }
        for corpus in corpora.values():
            imported = client.post(
                "/api/evals/corpora",
                headers=_AUTH_HEADERS,
                json=corpus.model_dump(mode="json"),
            )
            assert imported.status_code == 201

        default_catalog = client.get("/api/evals/corpora", headers=_AUTH_HEADERS)
        assert default_catalog.status_code == 200
        assert {item["target_key"] for item in default_catalog.json()["items"]} == {
            target_catalog["default_target_key"]
        }
        beta_catalog = client.get(
            "/api/evals/corpora",
            headers=_AUTH_HEADERS,
            params={"target_key": target_keys["beta"]},
        )
        assert beta_catalog.status_code == 200
        assert {item["target_key"] for item in beta_catalog.json()["items"]} == {
            target_keys["beta"]
        }
        assert (
            client.get(
                "/api/evals/corpora",
                headers=_AUTH_HEADERS,
                params={"target_key": "unknown-target"},
            ).status_code
            == 404
        )

        run_ids: dict[str, str] = {}
        for agent_name, corpus in corpora.items():
            admitted = client.post(
                "/api/evals/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": f"run-{agent_name}"},
                json={
                    "corpus_revision": corpus.revision,
                    "suite_id": corpus.suites[0].id,
                    "max_concurrency": 1,
                },
            )
            assert admitted.status_code == 202
            run_ids[agent_name] = admitted.json()["spec"]["run_id"]

        for agent_name, run_id in run_ids.items():
            assert _wait_for_terminal(client, run_id)["status"] == "completed"
            result_response = client.get(
                f"/api/evals/runs/{run_id}/result",
                headers=_AUTH_HEADERS,
            )
            assert result_response.status_code == 200
            result = result_response.json()["result"]
            assert (
                result["target"]["app_manifest"]["fingerprint"]
                == catalog_by_agent[agent_name]["app_manifest_fingerprint"]
            )
        assert len(provider.requests) == 2

        beta_runs = client.get(
            "/api/evals/runs",
            headers=_AUTH_HEADERS,
            params={"target_key": target_keys["beta"]},
        )
        assert beta_runs.status_code == 200
        assert [item["spec"]["run_id"] for item in beta_runs.json()["items"]] == [run_ids["beta"]]

        foreign_record = asyncio.run(store.load_run(foreign_run.run_id))
        assert foreign_record is not None
        assert foreign_record.status is EvalRunStatus.QUEUED
        assert foreign_record.ownership is None

        capabilities = client.get("/api/contract", headers=_AUTH_HEADERS).json()["capabilities"]
        assert capabilities["surfaces"]["evals"]["configured"] is True
        assert capabilities["evals_readiness"]["catalog_read"] == {
            "state": "ready",
            "reason_code": None,
        }
        assert capabilities["evals_readiness"]["fresh_launch"] == {
            "state": "ready",
            "reason_code": None,
        }
    asyncio.run(context.close())


def test_production_project_context_does_not_enable_evals_under_open_access(tmp_path) -> None:
    app = _target(_provider()).app
    store = SQLiteEvalStore(tmp_path / "evals.db")
    context = _create_project_control_plane_context(
        project_root=tmp_path.resolve(),
        project_id="production-project",
        configured_release_id="release-current",
        eval_store=store,
        store_backend="sqlite",
        store_source="project",
        access=ProjectControlPlaneAccess.AUTHENTICATED_PRODUCTION,
    )
    server = create_server(
        app,
        config=ServerConfig(
            access=OpenAccess(),
            dashboard=DashboardConfig(enabled=False),
        ),
        project_context=context,
    )

    with TestClient(server) as client:
        assert client.get("/api/evals/targets").status_code == 404
        capabilities = client.get("/api/contract").json()["capabilities"]
        assert capabilities["surfaces"]["evals"]["configured"] is False
        assert capabilities["evals_readiness"]["fresh_launch"] == {
            "state": "gated",
            "reason_code": "eval_target_not_configured",
        }
    asyncio.run(context.close())


def test_generated_eval_worker_rejects_manifest_drift_after_catalog_publication(tmp_path) -> None:
    provider = _provider()
    app = _target(provider).app
    store = SQLiteEvalStore(tmp_path / "evals.db")
    context = _create_project_control_plane_context(
        project_root=Path(__file__).resolve().parents[2],
        project_id="drift-project",
        configured_release_id="release-current",
        eval_store=store,
        store_backend="sqlite",
        store_source="project",
        access=ProjectControlPlaneAccess.AUTHENTICATED_PRODUCTION,
    )
    server = create_server(
        app,
        config=ServerConfig.protected(
            _authenticate,
            dashboard=DashboardConfig(enabled=False),
        ),
        project_context=context,
    )

    with TestClient(server) as client:
        catalog = client.get("/api/evals/targets", headers=_AUTH_HEADERS).json()
        target_key = catalog["default_target_key"]
        corpus = _corpus(target_key=target_key, trials=1)
        imported = client.post(
            "/api/evals/corpora",
            headers=_AUTH_HEADERS,
            json=corpus.model_dump(mode="json"),
        )
        assert imported.status_code == 201

        app.register_agent(AgentSpec(name="late-agent", model="fixture-model"))
        admitted = client.post(
            "/api/evals/runs",
            headers={**_AUTH_HEADERS, "Idempotency-Key": "manifest-drift"},
            json={
                "corpus_revision": corpus.revision,
                "suite_id": corpus.suites[0].id,
                "max_concurrency": 1,
            },
        )
        assert admitted.status_code == 202
        terminal = _wait_for_terminal(client, admitted.json()["spec"]["run_id"])

        assert terminal["status"] == "failed"
        assert terminal["failure_code"] == "target_unavailable"
        assert provider.requests == []
    asyncio.run(context.close())


def test_explicit_v1_configuration_publishes_one_compatible_target(tmp_path) -> None:
    target = _target(_provider())
    store = SQLiteEvalStore(tmp_path / "explicit.db")
    server = create_server(
        target.app,
        config=ServerConfig.protected(
            _authenticate,
            dashboard=DashboardConfig(enabled=False),
            evals=EvalsConfig(target=target, store=store),
        ),
    )

    try:
        with TestClient(server) as client:
            response = client.get("/api/evals/targets", headers=_AUTH_HEADERS)
            assert response.status_code == 200
            body = response.json()
            assert body["default_target_key"] == target.key
            assert body["items"] == [
                {
                    "target_key": target.key,
                    "project_id": None,
                    "agent_name": target.request_base.agent_name,
                    "profile_id": "explicit",
                    "label": f"{target.request_base.agent_name} · Explicit",
                    "source": "explicit",
                    "application_release_id": target.application_release_id,
                    "app_manifest_fingerprint": body["items"][0]["app_manifest_fingerprint"],
                }
            ]
    finally:
        asyncio.run(store.close())
