"""Recovered foreground results use artifact stores supplied by environment factories."""

import asyncio

import pytest
from tests.core import test_foreground_gate_recovery_tool_hooks as hook_tests

from cayu import CayuApp, EventType
from cayu.artifacts.local import LocalArtifactStore
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.environments.factory import EnvironmentFactory, EnvironmentFactoryResult
from cayu.providers.base import ModelStreamEvent
from cayu.tools.result_projection import ArtifactExternalizingToolResultPolicy


class _ArtifactEnvironmentFactory(EnvironmentFactory):
    def __init__(self, store):
        self.store = store

    @property
    def execution_profile_identity(self):
        return hook_tests._identity("artifact-environment")

    async def create(self, request):
        return EnvironmentFactoryResult(
            environment=Environment(
                EnvironmentSpec(name=request.environment_name), artifact_store=self.store
            )
        )


@pytest.mark.parametrize("gate", ["approval", "input"])
@pytest.mark.parametrize("child_pauses", [False, True])
def test_foreground_gate_recovery_projects_with_factory_artifact_store(
    tmp_path, monkeypatch, gate, child_pauses
):
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="gate-results")
    child_answer = "child done " + "x" * 2000
    applications = []
    original_provider = hook_tests._Provider

    def provider(batches):
        batches[-2] = [
            ModelStreamEvent.text_delta(child_answer),
            ModelStreamEvent.completed(),
        ]
        return original_provider(batches)

    def build(**kwargs):
        app = CayuApp(
            tool_result_projection_policy=ArtifactExternalizingToolResultPolicy(
                max_inline_bytes=1000
            ),
            **kwargs,
        )
        # Registration deliberately has no store: only materialization supplies it.
        app.register_environment_factory(
            EnvironmentSpec(name="factory"),
            _ArtifactEnvironmentFactory(artifact_store),
            default=True,
        )
        applications.append(app)
        return app

    monkeypatch.setattr(hook_tests, "CayuApp", build)
    monkeypatch.setattr(hook_tests, "_Provider", provider)

    async def scenario():
        await hook_tests._scenario(gate, child_pauses)
        terminals = [
            event
            for event in await applications[0].session_store.load_events("parent")
            if event.type is EventType.TOOL_CALL_COMPLETED
            and event.payload.get("tool_call_id") == "delegate"
        ]
        [terminal] = terminals
        projection = terminal.payload["tool_result_projection"]
        assert projection["status"] == "externalized"
        artifact = await artifact_store.read_bytes(projection["artifact_id"])
        assert child_answer in artifact.content.decode()
        assert "artifact_store_missing" not in terminal.payload["result"]["content"]

    asyncio.run(scenario())
