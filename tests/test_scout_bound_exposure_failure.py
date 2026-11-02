"""A failed exposure publication must not lose the bound cleanup context."""

import asyncio

import pytest

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.environments import Environment, EnvironmentSpec
from tests.runtime.runtime_environment_support import (
    RecordingEnvironmentFactory,
    RecordingWorkspaceBinding,
)
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_session_support import collect_events


@pytest.mark.parametrize("phase", ["admission", "exposure"])
def test_bound_environment_is_finalized_after_exposure_publication_failure(monkeypatch, phase):
    async def scenario():
        binding = RecordingWorkspaceBinding()
        factory = RecordingEnvironmentFactory(
            Environment(EnvironmentSpec(name="dynamic"), binding=binding)
        )
        app = CayuApp(enable_logging=False)
        provider = FakeProvider([])
        app.register_provider(provider, default=True)
        app.register_environment_factory(EnvironmentSpec(name="dynamic"), factory, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        emit = app._event_writer.emit
        injected = False

        async def lose_exposure_ack(event):
            nonlocal injected
            persisted = await emit(event)
            if (
                not injected
                and event.type == EventType.ENVIRONMENT_LIFECYCLE_TRANSITION
                and event.payload.get("phase") == phase
            ):
                injected = True
                raise ConnectionError("lost exposure publication acknowledgement")
            return persisted

        monkeypatch.setattr(app._event_writer, "emit", lose_exposure_ack)
        events = []
        failure = None
        try:
            events = await collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="scout_bound_exposure_" + phase,
                    messages=[Message.text("user", "run")],
                ),
            )
        except Exception as error:
            failure = error
        drained = await app.drain_environment_cleanups(timeout_s=0.2)
        assert injected and len(binding.bind_calls) == 1
        assert len(binding.finalize_calls) == 1, (
            "The successful bind must still be finalized after publication fails",
            failure,
            drained,
        )
        assert failure is None
        assert events[-1].type == EventType.SESSION_FAILED
        assert provider.requests == []
        assert await app.drain_environment_cleanups(timeout_s=1)

    asyncio.run(scenario())
