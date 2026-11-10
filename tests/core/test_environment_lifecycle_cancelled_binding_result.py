"""A successful bind returned during cancellation must retain a cleanup owner."""

import asyncio

import pytest
from tests.core._workload_secret_support import FakeProvider, collect_events

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.environments import (
    Environment,
    EnvironmentFactory,
    EnvironmentFactoryResult,
    EnvironmentSpec,
)
from cayu.environments.bindings import SyncBinding
from cayu.providers.base import ModelStreamEvent
from cayu.workspaces.local import LocalWorkspace


@pytest.mark.parametrize("factory_backed", [False, True], ids=["registered", "factory"])
@pytest.mark.parametrize("cancelled", [False, True], ids=["ordinary-return", "cancelled-return"])
def test_cancelled_static_binding_result_releases_workspace_reservations(
    tmp_path, cancelled, factory_backed
):
    async def scenario():
        source_root = tmp_path / "source"
        target_root = tmp_path / "target"
        source_root.mkdir()
        target_root.mkdir()
        (source_root / "input.txt").write_text("source input")
        source = LocalWorkspace(source_root, workspace_id="source")
        target = LocalWorkspace(target_root, workspace_id="target")
        bound_ready = asyncio.Event()
        allow_return = asyncio.Event()

        class SettlingSyncBinding(SyncBinding):
            returned_bound = None

            async def bind(self, *args, **kwargs):
                bound = await super().bind(*args, **kwargs)
                self.returned_bound = bound
                bound_ready.set()
                try:  # noqa: SIM105
                    await allow_return.wait()
                except asyncio.CancelledError:
                    # An adapter may finish acquiring an opaque mount/transfer
                    # despite cancellation and return its exact cleanup handle.
                    pass
                return bound

        binding = SettlingSyncBinding(target_workspace=target)
        provider = FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})])
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        environment = Environment(EnvironmentSpec(name="local"), workspace=source, binding=binding)
        if factory_backed:

            class Factory(EnvironmentFactory):
                async def create(self, request):
                    async def release(action):
                        pass

                    return EnvironmentFactoryResult(environment, release=release)

            app.register_environment_factory(environment.spec, Factory(), default=True)
        else:
            app.register_environment(environment, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        contender = SyncBinding(target_workspace=target)
        rebound = None
        try:
            task = asyncio.create_task(
                collect_events(
                    app,
                    RunRequest(
                        session_id="cancelled-binding-result",
                        agent_name="assistant",
                        messages=[Message.text("user", "run")],
                    ),
                )
            )
            await asyncio.wait_for(bound_ready.wait(), timeout=10)
            if cancelled:
                task.cancel("stop binding")
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert provider.requests == []
            else:
                allow_return.set()
                events = await task
                assert events[-1].type is EventType.SESSION_COMPLETED
            assert await app.drain_environment_cleanups(timeout_s=2)
            assert not app._environment_lifecycle._active_environment_setups

            # The cancelled run is settled. A subsequent session must be able
            # to acquire the same source/target; no private bind may remain.
            rebound = await contender.bind(source, None, session_id="successor")
            assert rebound.workspace is target
        finally:
            if rebound is not None:
                contender.abandon(rebound)
            if binding.returned_bound is not None:
                binding.abandon(binding.returned_bound)
            await app.aclose()

    asyncio.run(scenario())
