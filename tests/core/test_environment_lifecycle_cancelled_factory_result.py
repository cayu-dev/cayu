"""A factory's late successful result must retain its pre-adoption cleanup owner."""

import asyncio

import pytest
from tests.core._workload_secret_support import FakeProvider, collect_events
from tests.runtime.runtime_environment_support import ProvisionedTestRunner

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.environments import (
    Environment,
    EnvironmentFactory,
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryResult,
    EnvironmentSpec,
)


@pytest.mark.parametrize(
    "finish_mode", ["reject", "cancel"], ids=["ordinary-failure", "cancelled-return"]
)
def test_cancelled_factory_return_still_releases_the_returned_allocation(finish_mode):
    async def scenario():
        factory_entered = asyncio.Event()
        runner = ProvisionedTestRunner()
        releases = []

        class SettlingFactory(EnvironmentFactory):
            async def create(self, request):
                factory_entered.set()
                if finish_mode == "cancel":
                    try:  # noqa: SIM105
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        # Provisioning has settled despite cancellation; transfer
                        # its exact result to the runtime for disposal.
                        pass

                async def release(action):
                    releases.append(action)
                    await runner.close()

                return EnvironmentFactoryResult(
                    Environment(
                        EnvironmentSpec(
                            name=request.environment_name
                            if finish_mode == "cancel"
                            else "wrong-name"
                        ),
                        runner=runner,
                    ),
                    reconnect_metadata={"allocation_id": "late-created"},
                    release=release,
                )

        app = CayuApp(enable_logging=False)
        provider = FakeProvider([])
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="sandbox"), SettlingFactory(), default=True
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        try:
            task = asyncio.create_task(
                collect_events(
                    app,
                    RunRequest(
                        session_id="cancelled-factory-result",
                        agent_name="assistant",
                        messages=[Message.text("user", "run")],
                    ),
                )
            )
            await asyncio.wait_for(factory_entered.wait(), timeout=10)
            if finish_mode == "cancel":
                task.cancel("stop provisioning")
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                events = await task
                assert events[-1].type is EventType.SESSION_FAILED
            assert provider.requests == []
            assert await app._environment_lifecycle.drain_retained_cleanups(timeout_s=2)
            assert releases == [EnvironmentFactoryReleaseAction.DISCARD], (
                "A successfully returned factory result lost its cleanup callback during cancellation",
                releases,
                runner.is_closed,
            )
            assert runner.is_closed
        finally:
            await runner.close()
            await app.aclose()

    asyncio.run(scenario())
