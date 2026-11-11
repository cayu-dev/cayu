"""A cleanup retry shared by two sessions must retire both ownership records."""

import asyncio

import pytest
from tests.core.test_environment_lifecycle import _collect_events, _FakeProvider

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.environments import EnvironmentFactory, EnvironmentSpec
from cayu.environments.factory import (
    attach_environment_factory_cleanup_settlement_task,
    register_environment_factory_cleanup_retry,
)
from cayu.providers import ModelStreamEvent


@pytest.mark.parametrize("shared", [False, True], ids=["separate-owners", "shared-owner"])
def test_successful_shared_cleanup_retry_retires_every_waiting_session(shared):
    async def scenario():
        retries = []
        recovered = []
        tasks = {}
        allow_initial_failure = asyncio.Event()

        def initial_cleanup(key):
            async def fail():
                await allow_initial_failure.wait()
                raise ConnectionError("temporary shared provider cleanup failure")

            def retry():
                retries.append(key)

                async def settle():
                    recovered.append(key)

                return asyncio.create_task(settle())

            task = asyncio.create_task(fail())
            register_environment_factory_cleanup_retry(task, retry)
            return task

        class Factory(EnvironmentFactory):
            async def create(self, request):
                key = "shared" if shared else request.session_id
                if key not in tasks:
                    tasks[key] = initial_cleanup(key)
                error = RuntimeError("factory failed during provisioning")
                attach_environment_factory_cleanup_settlement_task(error, tasks[key])
                raise error

        app = CayuApp(enable_logging=False)
        app.register_provider(
            _FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})]), default=True
        )
        app.register_environment_factory(EnvironmentSpec(name="dynamic"), Factory(), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        for session_id in ("cleanup-one", "cleanup-two"):
            events = await _collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "run")],
                ),
            )
            assert events[-1].type is EventType.SESSION_FAILED
        assert len(app._environment_lifecycle._deferred_factory_cleanup_tasks) == 2
        allow_initial_failure.set()
        await asyncio.gather(*tasks.values(), return_exceptions=True)
        drains = [await app.drain_environment_cleanups(timeout_s=2) for _ in range(2)]
        assert len(recovered) == (1 if shared else 2)
        assert all(drains), (
            "The authenticated cleanup retry succeeded but another session still holds its old task",
            drains,
            retries,
            recovered,
            tuple(app._environment_lifecycle._deferred_factory_cleanup_tasks),
        )
        assert app._environment_lifecycle._deferred_factory_cleanup_tasks == {}
        await app.aclose()

    asyncio.run(scenario())
