"""Diagnostic publication failure must not strand a successfully settled marker."""

import asyncio

import pytest

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.environments import Environment, EnvironmentSpec, SyncBinding
from cayu.providers import ModelStreamEvent
from cayu.sessions._completion_finalization import pending_completion_finalization_from_checkpoint
from cayu.sessions.recovery import IncompleteSessionRecoveryRequest
from cayu.workspaces import LocalWorkspace
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_session_support import collect_events


@pytest.mark.parametrize("lose_diagnostic", [False, True])
def test_failed_run_clears_successful_finalization_marker_despite_diagnostic_loss(
    tmp_path, monkeypatch, lose_diagnostic
):
    async def scenario():
        source = tmp_path / "source"
        target = tmp_path / "target"
        source.mkdir()
        target.mkdir()
        (source / "input.txt").write_text("original")

        class CountingBinding(SyncBinding):
            finalize_calls = 0
            finalized = False

            async def finalize(self, bound, **kwargs):
                self.finalize_calls += 1
                result = await super().finalize(bound, **kwargs)
                self.finalized = True
                return result

        class FailingProvider(FakeProvider):
            async def stream(self, request):
                raise RuntimeError("model unavailable")
                yield ModelStreamEvent.completed({"finish_reason": "stop"})

        binding = CountingBinding(target_workspace=LocalWorkspace(target, workspace_id="target"))
        app = CayuApp(enable_logging=False)
        app.register_provider(FailingProvider([]), default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="sync"),
                workspace=LocalWorkspace(source, workspace_id="source"),
                binding=binding,
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        emit = app._event_writer.emit
        injected = False

        async def lose_diagnostic_once(event):
            nonlocal injected
            if (
                lose_diagnostic
                and not injected
                and event.type == EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED
            ):
                assert binding.finalized
                injected = True
                raise ConnectionError("lost finalization diagnostic")
            return await emit(event)

        monkeypatch.setattr(app._event_writer, "emit", lose_diagnostic_once)
        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="scout_terminal_marker",
                messages=[Message.text("user", "run")],
            ),
        )
        assert events[-1].type == EventType.SESSION_FAILED
        assert injected is lose_diagnostic
        assert binding.finalized and binding.finalize_calls == 1
        assert await app.drain_environment_cleanups(timeout_s=1)
        # A settled SyncBinding releases its target for the next session.
        successor = tmp_path / "successor"
        successor.mkdir()
        (successor / "input.txt").write_text("successor session data")
        bound = await binding.bind(
            LocalWorkspace(successor, workspace_id="successor"),
            None,
            session_id="successor-session",
        )
        await binding.finalize(bound, outcome="completed")
        assert (target / "input.txt").read_text() == "successor session data"
        recovery = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id="scout_terminal_marker")
        )
        assert (source / "input.txt").read_text() == "original", (
            "Recovery replayed an already-successful finalization using a successor's target",
            recovery.actions,
        )
        checkpoint = await app.session_store.load_checkpoint("scout_terminal_marker")
        assert pending_completion_finalization_from_checkpoint(checkpoint) is None

    asyncio.run(scenario())
