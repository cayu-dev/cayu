from __future__ import annotations

import asyncio
import hashlib

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

from cayu import (
    AgentSpec,
    ArtifactScope,
    CayuApp,
    CorpusTarget,
    Environment,
    EnvironmentSpec,
    EvalScenarioDocumentV2,
    EvalScenarioDraftV2,
    LocalArtifactStore,
    RunRequest,
    ScenarioArtifactRequirementV2,
    ScenarioFilePartV2,
    ScenarioInitialInputEventV2,
    ScenarioInputV2,
    ScenarioTextPartV2,
    ScenarioUserMessageV2,
    ScriptedModelProvider,
)
from cayu.server import AuthContext, DashboardConfig, EvalsConfig, ServerConfig, create_server
from cayu.storage.evals_sqlite import SQLiteEvalStore

_AUTH_HEADERS = {"Authorization": "Bearer valid"}


def _authenticate(request: Request) -> AuthContext:
    if request.headers.get("Authorization") != "Bearer valid":
        raise HTTPException(status_code=401, detail="unauthorized")
    return AuthContext(subject="scenario-operator")


def _target(tmp_path) -> tuple[CorpusTarget, LocalArtifactStore, ScriptedModelProvider]:
    provider = ScriptedModelProvider([])
    artifact_store = LocalArtifactStore(
        tmp_path / "artifacts",
        store_id="scenario-server-artifacts",
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="scenario-model"))
    app.register_environment(
        Environment(
            EnvironmentSpec(name="files"),
            artifact_store=artifact_store,
        ),
        default=True,
    )
    target = CorpusTarget(
        key="assistant.default",
        app=app,
        request_base=RunRequest(
            agent_name="assistant",
            messages=[],
            environment_name="files",
            max_steps=8,
        ),
        application_release_id="release-current",
    )
    return target, artifact_store, provider


def _server(target: CorpusTarget, store: SQLiteEvalStore):
    return create_server(
        target.app,
        config=ServerConfig.protected(
            _authenticate,
            dashboard=DashboardConfig(enabled=False),
            evals=EvalsConfig(
                target=target,
                store=store,
                poll_interval_seconds=0.02,
                lease_seconds=5,
                shutdown_grace_seconds=2,
            ),
        ),
    )


def _scenario(
    requirement: ScenarioArtifactRequirementV2 | None = None,
) -> EvalScenarioDocumentV2:
    content = [ScenarioTextPartV2(text="Review this retained request.")]
    if requirement is not None:
        content.append(ScenarioFilePartV2(artifact_requirement_id=requirement.id))
    return EvalScenarioDocumentV2.create(
        id="retained-request",
        target_key="assistant.default",
        name="Retained request",
        events=(
            ScenarioInitialInputEventV2(
                sequence=0,
                id="initial",
                input=ScenarioInputV2.create((ScenarioUserMessageV2.create(content),)),
            ),
        ),
        artifact_requirements=(() if requirement is None else (requirement,)),
    )


def test_scenario_editor_preview_save_catalog_and_download_are_target_scoped(
    tmp_path,
) -> None:
    target, _, provider = _target(tmp_path)
    store = SQLiteEvalStore(tmp_path / "evals.db")
    scenario = _scenario()
    draft = EvalScenarioDraftV2.from_scenario(scenario)
    try:
        with TestClient(_server(target, store)) as client:
            assert (
                client.post(
                    "/api/evals/scenarios/preview",
                    json={"draft": draft.model_dump(mode="json")},
                ).status_code
                == 401
            )
            preview = client.post(
                "/api/evals/scenarios/preview",
                headers=_AUTH_HEADERS,
                json={
                    "draft": draft.model_dump(mode="json"),
                    "settings": {"trials": 1, "max_concurrency": 1},
                },
            )
            assert preview.status_code == 200
            assert preview.json()["scenario"] == scenario.model_dump(mode="json")
            assert preview.json()["preflight"]["ready"] is True

            stale = client.post(
                "/api/evals/scenarios",
                headers=_AUTH_HEADERS,
                json={
                    "expected_scenario_revision": "sha256:" + "0" * 64,
                    "scenario": scenario.model_dump(mode="json"),
                },
            )
            assert stale.status_code == 409
            assert stale.json() == {"detail": "Eval scenario changed after the reviewed revision."}

            saved = client.post(
                "/api/evals/scenarios",
                headers=_AUTH_HEADERS,
                json={
                    "expected_scenario_revision": scenario.revision,
                    "scenario": scenario.model_dump(mode="json"),
                },
            )
            assert saved.status_code == 201
            assert saved.json()["entry"]["revision"] == scenario.revision
            assert saved.json()["preflight"]["ready"] is True

            catalog = client.get(
                "/api/evals/scenarios",
                headers=_AUTH_HEADERS,
            )
            assert catalog.status_code == 200
            assert [item["revision"] for item in catalog.json()["items"]] == [scenario.revision]
            loaded = client.get(
                f"/api/evals/scenarios/{scenario.revision}",
                headers=_AUTH_HEADERS,
            )
            assert loaded.status_code == 200
            assert loaded.json() == scenario.model_dump(mode="json")
            downloaded = client.get(
                f"/api/evals/scenarios/{scenario.revision}/download",
                headers=_AUTH_HEADERS,
            )
            assert downloaded.status_code == 200
            assert downloaded.content.endswith(b"\n")
            assert EvalScenarioDocumentV2.model_validate_json(downloaded.content) == scenario
        assert provider.requests == []
    finally:
        asyncio.run(store.close())


def test_scenario_artifact_preparation_returns_a_ready_unsaved_revision(tmp_path) -> None:
    target, artifact_store, provider = _target(tmp_path)
    store = SQLiteEvalStore(tmp_path / "evals.db")

    async def seed():
        content = b"retained production attachment"
        artifact = await artifact_store.put_bytes(
            content,
            filename="request.txt",
            content_type="text/plain",
            scope=ArtifactScope.SESSION,
            session_id="production-session",
            environment_name="files",
        )
        return content, artifact

    content, artifact = asyncio.run(seed())
    requirement = ScenarioArtifactRequirementV2(
        id="request-file",
        source="artifact_reference",
        reference=artifact.id,
        content_sha256=hashlib.sha256(content).hexdigest(),
        filename=artifact.filename,
        content_type=artifact.content_type,
        size_bytes=artifact.size_bytes,
    )
    scenario = _scenario(requirement)
    try:
        with TestClient(_server(target, store)) as client:
            preview = client.post(
                "/api/evals/scenarios/preview",
                headers=_AUTH_HEADERS,
                json={"draft": EvalScenarioDraftV2.from_scenario(scenario).model_dump(mode="json")},
            )
            assert preview.status_code == 200
            assert [item["code"] for item in preview.json()["preflight"]["diagnostics"]] == [
                "artifact_binding_required"
            ]

            prepared = client.post(
                f"/api/evals/scenarios/artifacts/{requirement.id}/materialize",
                headers=_AUTH_HEADERS,
                json={
                    "expected_scenario_revision": scenario.revision,
                    "scenario": scenario.model_dump(mode="json"),
                },
            )
            assert prepared.status_code == 200
            body = prepared.json()
            assert body["preflight"]["ready"] is True
            updated = EvalScenarioDocumentV2.model_validate(body["materialization"]["scenario"])
            assert updated.revision != scenario.revision
            fixture_id = body["materialization"]["artifact_id"]
            assert updated.artifact_requirements[0].reference == fixture_id
            fixture = asyncio.run(artifact_store.read_bytes(fixture_id))
            assert fixture.content == content
            assert fixture.metadata.scope is ArtifactScope.ENVIRONMENT

            repeated = client.post(
                f"/api/evals/scenarios/artifacts/{requirement.id}/materialize",
                headers=_AUTH_HEADERS,
                json={
                    "expected_scenario_revision": scenario.revision,
                    "scenario": scenario.model_dump(mode="json"),
                },
            )
            assert repeated.status_code == 200
            assert repeated.json()["materialization"] == body["materialization"]

            repeated_updated = client.post(
                f"/api/evals/scenarios/artifacts/{requirement.id}/materialize",
                headers=_AUTH_HEADERS,
                json={
                    "expected_scenario_revision": updated.revision,
                    "scenario": updated.model_dump(mode="json"),
                },
            )
            assert repeated_updated.status_code == 200
            assert repeated_updated.json()["materialization"] == body["materialization"]
        assert provider.requests == []
    finally:
        asyncio.run(store.close())
