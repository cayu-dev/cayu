"""A project's declared eval targets, served by ``cayu serve`` next to the generated ones."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi.testclient import TestClient
from tests.evals.test_corpus_execution import _corpus, _provider, _target
from tests.evals.test_structured_output_field_assertions import (
    _agent_target,
    _spec,
    _verdict_schema,
)
from tests.evals.test_structured_output_field_assertions import (
    _corpus as _structured_output_corpus,
)
from tests.server.test_generated_evals import _wait_for_terminal
from tests.server.test_server_evals import (
    _AUTH_HEADERS,
    _authenticate,
    _repeatable_execution_policy,
)

from cayu.evals.execution_profiles import EvalExecutionProfilePolicyV1
from cayu.project_control_plane import (
    ProjectControlPlaneAccess,
    ProjectEvalTarget,
    _create_project_control_plane_context,
)
from cayu.server import EvalsConfig, ServerConfig, create_server
from cayu.storage.evals_sqlite import SQLiteEvalStore

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _declared(key: str = "refund-agent", *, trials: int = 2) -> ProjectEvalTarget:
    target = _target(_provider(trials=trials), key=key)
    return ProjectEvalTarget(
        reference=f"evals.{key.replace('-', '_')}:build",
        target=target,
        execution_profile_policy=_repeatable_execution_policy(target).model_copy(
            update={"max_trials": trials, "max_concurrency": 1}
        ),
    )


def _context(tmp_path: Path, *declared: ProjectEvalTarget, project_id: str | None):
    return _create_project_control_plane_context(
        project_root=_PROJECT_ROOT,
        project_id=project_id,
        eval_store=SQLiteEvalStore(tmp_path / "evals.db"),
        store_backend="sqlite",
        store_source="project",
        access=ProjectControlPlaneAccess.AUTHENTICATED_PRODUCTION,
        eval_targets=declared,
    )


def test_a_declared_target_runs_a_repeated_corpus_next_to_the_generated_targets(
    tmp_path: Path,
) -> None:
    served = _target(_provider(trials=1)).app
    declared = _declared()
    context = _context(tmp_path, declared, project_id="declared-project")
    try:
        server = create_server(
            served, config=ServerConfig.protected(_authenticate), project_context=context
        )
        with TestClient(server) as client:
            catalog = client.get("/api/evals/targets", headers=_AUTH_HEADERS).json()
            by_source = {item["source"]: item for item in catalog["items"]}
            assert set(by_source) == {"generated", "explicit"}
            assert by_source["generated"]["agent_name"] == "agent"
            item = by_source["explicit"]
            assert item["target_key"] == "refund-agent"
            assert item["profile_id"] == "refund-agent"
            assert item["label"] == "agent · refund-agent"
            assert item["project_id"] is None
            assert item["max_trials"] == 2
            assert item["execution_profile_ready"] is True
            assert item["app_manifest_fingerprint"] == (
                declared.target.app.describe(project_root=_PROJECT_ROOT).fingerprint
            )

            corpus = _corpus(target_key="refund-agent", trials=2)
            imported = client.post(
                "/api/evals/corpora", headers=_AUTH_HEADERS, json=corpus.model_dump(mode="json")
            )
            assert imported.status_code == 201
            admitted = client.post(
                "/api/evals/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "declared-run"},
                json={
                    "corpus_revision": corpus.revision,
                    "suite_id": corpus.suites[0].id,
                    "expected_execution_profile_revision": item["execution_profile"]["revision"],
                },
            )
            assert admitted.status_code == 202
            run = _wait_for_terminal(client, admitted.json()["spec"]["run_id"])
            assert run["status"] == "completed"
            result = client.get(
                f"/api/evals/runs/{run['spec']['run_id']}/result", headers=_AUTH_HEADERS
            ).json()["result"]
            (case,) = result["run"]["cases"]
            assert [trial["status"] for trial in case["trials"]] == ["passed", "passed"]
        # The declared target's own application stops with the server.
        assert declared.target.app.lifecycle_state == "closed"
    finally:
        asyncio.run(context.close())


def test_declared_targets_are_served_without_a_project_identity(tmp_path: Path) -> None:
    served = _target(_provider(trials=1)).app
    context = _context(
        tmp_path,
        _declared("refund-agent"),
        _declared("refund-agent-strict", trials=1),
        project_id=None,
    )
    try:
        server = create_server(
            served, config=ServerConfig.protected(_authenticate), project_context=context
        )
        with TestClient(server) as client:
            catalog = client.get("/api/evals/targets", headers=_AUTH_HEADERS).json()
            assert sorted(item["target_key"] for item in catalog["items"]) == [
                "refund-agent",
                "refund-agent-strict",
            ]
            assert {item["source"] for item in catalog["items"]} == {"explicit"}
    finally:
        asyncio.run(context.close())


def test_an_explicit_server_eval_target_replaces_the_declared_ones(tmp_path: Path) -> None:
    explicit = _target(_provider(trials=1), key="explicit-agent")
    context = _context(tmp_path, _declared(), project_id="declared-project")
    try:
        server = create_server(
            explicit.app,
            config=ServerConfig.protected(
                _authenticate,
                evals=EvalsConfig(
                    target=explicit,
                    store=SQLiteEvalStore(tmp_path / "explicit.db"),
                    execution_profile_policy=EvalExecutionProfilePolicyV1.safe_default(),
                ),
            ),
            project_context=context,
        )
        with TestClient(server) as client:
            catalog = client.get("/api/evals/targets", headers=_AUTH_HEADERS).json()
            assert [item["target_key"] for item in catalog["items"]] == ["explicit-agent"]
    finally:
        asyncio.run(context.close())


def test_a_declared_agent_target_asserts_on_the_structured_output_it_requests(
    tmp_path: Path,
) -> None:
    target = _agent_target(_verdict_schema())
    declared = ProjectEvalTarget(
        reference="evals.verifier:build",
        target=target,
        execution_profile_policy=_repeatable_execution_policy(target).model_copy(
            update={"max_trials": 1, "max_concurrency": 1}
        ),
    )
    context = _context(tmp_path, declared, project_id="verifier-project")
    try:
        server = create_server(
            _target(_provider(trials=1)).app,
            config=ServerConfig.protected(_authenticate),
            project_context=context,
        )
        with TestClient(server) as client:
            catalog = client.get("/api/evals/targets", headers=_AUTH_HEADERS).json()
            (item,) = [entry for entry in catalog["items"] if entry["target_key"] == "verifier"]
            corpus = _structured_output_corpus(
                _spec(
                    id="verdict-confirmed", pointer="/verdict", operator="equals", value="confirmed"
                )
            )
            imported = client.post(
                "/api/evals/corpora", headers=_AUTH_HEADERS, json=corpus.model_dump(mode="json")
            )
            assert imported.status_code == 201
            admitted = client.post(
                "/api/evals/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "verifier-run"},
                json={
                    "corpus_revision": corpus.revision,
                    "suite_id": corpus.suites[0].id,
                    "expected_execution_profile_revision": item["execution_profile"]["revision"],
                },
            )
            assert admitted.status_code == 202
            run = _wait_for_terminal(client, admitted.json()["spec"]["run_id"])
            assert run["status"] == "completed"
            result = client.get(
                f"/api/evals/runs/{run['spec']['run_id']}/result", headers=_AUTH_HEADERS
            ).json()["result"]
            (case,) = result["run"]["cases"]
            (trial,) = case["trials"]
            (assertion,) = trial["assertions"]
            assert trial["status"] == "passed"
            assert assertion["detail"]["observed"] == {"value": "confirmed"}
            assert assertion["detail"]["structured_output_revision"].startswith("sha256:")
    finally:
        asyncio.run(context.close())
