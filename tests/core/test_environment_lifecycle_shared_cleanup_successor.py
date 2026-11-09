"""Converging factory cleanup handoffs are not cyclic ownership chains."""

import asyncio

import pytest
from tests.core.test_environment_lifecycle import _collect_events, _FakeProvider

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.configuration import CayuConfig, OperationsConfig
from cayu.environments import Environment, EnvironmentSpec
from cayu.environments.bindings import WorkspaceBinding
from cayu.environments.factory import (
    EnvironmentFactory,
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryResult,
    attach_environment_factory_cleanup_settlement_task,
)
from cayu.providers import ModelStreamEvent


@pytest.mark.parametrize("shared_successor", [False, True], ids=["separate", "converging"])
def test_release_settles_branches_that_delegate_to_the_same_cleanup(shared_successor):
    async def scenario():
        allow_handoff = asyncio.Event()
        allow_cleanup = asyncio.Event()
        handoff_started = asyncio.Event()
        cleanup_finished = []
        cleanup_tasks = []

        class FailingBinding(WorkspaceBinding):
            async def bind(self, workspace, runner, **kwargs):
                raise RuntimeError("binding failed")

            async def finalize(self, bound, **kwargs):
                raise AssertionError("An unbound workspace cannot be finalized")

        class Factory(EnvironmentFactory):
            calls = 0

            async def create(self, request):
                self.calls += 1
                if self.calls > 1:
                    return EnvironmentFactoryResult(
                        Environment(EnvironmentSpec(name=request.environment_name))
                    )

                async def release(action):
                    assert action is EnvironmentFactoryReleaseAction.PRESERVE
                    await allow_handoff.wait()

                    async def cleanup(name):
                        await allow_cleanup.wait()
                        cleanup_finished.append(name)

                    first = asyncio.create_task(cleanup("container"))
                    second = first if shared_successor else asyncio.create_task(cleanup("binding"))
                    cleanup_tasks.extend(dict.fromkeys((first, second)))

                    async def delegate(successor):
                        error = RuntimeError("component transferred cleanup ownership")
                        attach_environment_factory_cleanup_settlement_task(error, successor)
                        raise error

                    errors = []
                    for successor in (first, second):
                        error = RuntimeError("component release pending")
                        attach_environment_factory_cleanup_settlement_task(
                            error, asyncio.create_task(delegate(successor))
                        )
                        errors.append(error)
                    handoff_started.set()
                    raise ExceptionGroup("composite environment release", errors)

                return EnvironmentFactoryResult(
                    Environment(
                        EnvironmentSpec(name=request.environment_name), binding=FailingBinding()
                    ),
                    release=release,
                    release_timeout_s=0.01,
                )

        factory = Factory()
        app = CayuApp(
            enable_logging=False,
            config=CayuConfig(operations=OperationsConfig(max_environment_lifecycle_owners=1)),
        )
        app.register_provider(
            _FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})]), default=True
        )
        app.register_environment_factory(EnvironmentSpec(name="dynamic"), factory, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        first_events = await _collect_events(
            app,
            RunRequest(
                agent_name="assistant", session_id="first", messages=[Message.text("user", "go")]
            ),
        )
        assert first_events[-1].type is EventType.SESSION_FAILED
        assert "first" in app._environment_lifecycle._deferred_factory_cleanup_tasks
        allow_handoff.set()
        await asyncio.wait_for(handoff_started.wait(), 5)
        allow_cleanup.set()
        await asyncio.gather(*cleanup_tasks)
        assert len(cleanup_finished) == (1 if shared_successor else 2)

        drained = await app.drain_environment_cleanups(timeout_s=2)
        next_events = await _collect_events(
            app,
            RunRequest(
                agent_name="assistant", session_id="second", messages=[Message.text("user", "go")]
            ),
        )
        assert next_events[-1].type is EventType.SESSION_COMPLETED, (
            "Completed cleanup was treated as a cycle and still consumes lifecycle capacity",
            next_events[-1].payload,
        )
        assert drained, "Every distinct cleanup task succeeded, so cleanup must be quiescent"

    asyncio.run(scenario())
