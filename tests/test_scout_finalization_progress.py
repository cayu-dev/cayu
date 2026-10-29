"""A diagnostic failure after successful copy-back must not replay finalization."""

import asyncio

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.environments import Environment, EnvironmentSpec, SyncBinding
from cayu.environments.lifecycle import EnvironmentLifecyclePolicy
from cayu.providers import ModelStreamEvent
from cayu.workspaces import LocalWorkspace
from tests.core.test_environment_lifecycle import _collect_events, _FakeProvider


def test_finished_binding_is_not_retried_after_progress_publication_failure(tmp_path, monkeypatch):
    asyncio.run(_scenario(tmp_path, monkeypatch))


async def _scenario(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "input.txt").write_text("original")

    class CountingBinding(SyncBinding):
        finalize_calls = 0
        finalize_succeeded = False

        async def finalize(self, bound, **kwargs):
            self.finalize_calls += 1
            result = await super().finalize(bound, **kwargs)
            self.finalize_succeeded = True
            return result

    binding = CountingBinding(target_workspace=LocalWorkspace(target_root, workspace_id="target"))
    app = CayuApp(enable_logging=False)
    app.register_provider(
        _FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})]), default=True
    )
    app.register_environment(
        Environment(
            EnvironmentSpec(name="sync", lifecycle_policy=EnvironmentLifecyclePolicy()),
            workspace=LocalWorkspace(source_root, workspace_id="source"),
            binding=binding,
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    emit = app._event_writer.emit
    injected = False

    async def fail_completed_progress_once(event):
        nonlocal injected
        if (
            not injected
            and event.type == EventType.ENVIRONMENT_LIFECYCLE_PROGRESS
            and event.payload["operation"] == "finalization"
            and event.payload["status"] == "completed"
            and event.payload["operation_terminal"]
        ):
            assert binding.finalize_succeeded
            injected = True
            raise ConnectionError("lost finalization progress write")
        return await emit(event)

    monkeypatch.setattr(app._event_writer, "emit", fail_completed_progress_once)
    try:
        await _collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="scout-finalize-progress",
                messages=[Message.text("user", "run")],
            ),
        )
    except ConnectionError as error:
        assert str(error) == "lost finalization progress write"
    assert injected
    drained = await app.drain_environment_cleanups(timeout_s=1)
    assert binding.finalize_calls == 1, (
        "Only the failed diagnostic may be retried, not a released binding generation",
        binding.finalize_calls,
    )
    assert drained, "A completed binding is retained forever retrying its released generation"
