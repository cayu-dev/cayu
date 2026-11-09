"""Directly registered Docker bindings must still reach their disposal boundary."""

import asyncio

import pytest
from tests.core.test_environment_lifecycle import _collect_events, _FakeProvider
from tests.environments.test_docker_coding import _LocalDockerRunner, _no_git_binding

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.environments import Environment, EnvironmentSpec
from cayu.providers import ModelStreamEvent
from cayu.workspaces import LocalWorkspace


@pytest.mark.parametrize("through_app", [False, True], ids=["binding", "registered-environment"])
def test_direct_docker_binding_finalizes_without_a_factory(tmp_path, through_app):
    async def scenario():
        source_root = tmp_path / "source"
        target_root = tmp_path / "target"
        source_root.mkdir()
        target_root.mkdir()
        (source_root / "input.txt").write_text("original")
        source = LocalWorkspace(source_root, workspace_id="direct-source")
        runner = _LocalDockerRunner(target_root)
        binding = _no_git_binding(runner)
        try:
            if not through_app:
                bound = await binding.bind(source, runner, session_id="direct-docker")
                await binding.finalize(bound, outcome="completed")
            else:
                app = CayuApp(enable_logging=False)
                app.register_provider(
                    _FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})]),
                    default=True,
                )
                app.register_environment(
                    Environment(
                        EnvironmentSpec(name="docker"),
                        workspace=source,
                        runner=runner,
                        binding=binding,
                    ),
                    default=True,
                )
                app.register_agent(AgentSpec(name="assistant", model="fake-model"))
                events = await _collect_events(
                    app,
                    RunRequest(
                        agent_name="assistant",
                        session_id="direct-docker",
                        messages=[Message.text("user", "Complete without edits.")],
                    ),
                )
                assert events[-1].type is EventType.SESSION_COMPLETED, (
                    "Registering a binding directly must not invent a factory disposal requirement",
                    events[-1].payload,
                )
                assert await app.drain_environment_cleanups(timeout_s=2)
            assert runner.is_closed
            assert not binding._states
        finally:
            await runner.close()

    asyncio.run(scenario())
