"""Release handoffs must settle identically before and after the wait bound."""

import asyncio

import pytest
from tests.core._workload_secret_support import FakeProvider, collect_events

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.configuration import CayuConfig, OperationsConfig
from cayu.environments import (
    Environment,
    EnvironmentFactory,
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryResult,
    EnvironmentSpec,
)
from cayu.environments.bindings import WorkspaceBinding
from cayu.environments.factory import attach_environment_factory_cleanup_settlement_task
from cayu.providers.base import ModelStreamEvent


@pytest.mark.parametrize("late", [True, False], ids=["after-bound", "before-bound"])
def test_grouped_release_handoff_retires_after_its_cleanup_succeeds(late):
    async def scenario():
        allow_release = asyncio.Event()
        allow_cleanup = asyncio.Event()
        cleanup_started = asyncio.Event()
        cleanup = None
        release_calls = 0

        class FailingBinding(WorkspaceBinding):
            async def bind(self, workspace, runner, **kwargs):
                raise ConnectionError("workspace attachment failed")

            async def finalize(self, bound, **kwargs):
                raise AssertionError("bind never returned a bound workspace")

        class Factory(EnvironmentFactory):
            calls = 0

            async def create(self, request):
                self.calls += 1
                if self.calls > 1:
                    return EnvironmentFactoryResult(
                        Environment(EnvironmentSpec(name=request.environment_name))
                    )

                async def release(action):
                    nonlocal cleanup, release_calls
                    assert action is EnvironmentFactoryReleaseAction.PRESERVE
                    release_calls += 1
                    if late:
                        await allow_release.wait()

                    async def finish_cleanup():
                        cleanup_started.set()
                        await allow_cleanup.wait()

                    cleanup = asyncio.create_task(finish_cleanup())
                    failure = ConnectionError("detach continues in its cleanup owner")
                    attach_environment_factory_cleanup_settlement_task(failure, cleanup)
                    # Composite release implementations report component failures
                    # as groups; the handoff belongs to the failing component.
                    raise ExceptionGroup("release failed", [failure])

                return EnvironmentFactoryResult(
                    Environment(
                        EnvironmentSpec(name=request.environment_name),
                        binding=FailingBinding(),
                    ),
                    release=release,
                    release_timeout_s=0.03,
                )

        factory = Factory()
        app = CayuApp(
            enable_logging=False,
            config=CayuConfig(operations=OperationsConfig(max_environment_lifecycle_owners=1)),
        )
        app.register_provider(
            FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})]),
            default=True,
        )
        app.register_environment_factory(EnvironmentSpec(name="dynamic"), factory, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        def request(session_id):
            return RunRequest(
                session_id=session_id,
                agent_name="assistant",
                messages=[Message.text("user", "run")],
            )

        try:
            events = await collect_events(app, request("release-owner"))
            assert events[-1].type is EventType.SESSION_FAILED
            allow_release.set()
            await asyncio.wait_for(cleanup_started.wait(), timeout=2)
            assert not await app.drain_environment_cleanups(timeout_s=0.05)
            allow_cleanup.set()
            await cleanup
            drained = await app.drain_environment_cleanups(timeout_s=2)
            successor = await collect_events(app, request("successor"))
            assert successor[-1].type is EventType.SESSION_COMPLETED, (
                "Successful delegated cleanup still occupies lifecycle capacity",
                drained,
                successor[-1].payload,
            )
            assert drained
            assert release_calls == 1
        finally:
            allow_release.set()
            allow_cleanup.set()
            if cleanup is not None:
                await cleanup
            await app.aclose()

    asyncio.run(scenario())
