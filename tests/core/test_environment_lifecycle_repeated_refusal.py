"""Each rejected allocation must retire its own reconnect authority."""

import asyncio

import pytest
from tests.runtime.test_execution_admission_dispatch import (
    _EvidenceRunner,
    _RecordingProvider,
    _RecoveringFactory,
    _requirements,
    _run,
)

from cayu import AgentSpec, CayuApp, EventType, Message, ResumeRequest
from cayu.environments import EnvironmentSpec
from cayu.environments.factory import (
    EnvironmentFactoryOperation,
    EnvironmentFactoryReleaseAction,
)


@pytest.mark.parametrize("refusals", [1, 2], ids=["one-refusal", "two-refusals"])
def test_each_refused_allocation_retires_before_the_next_resume(refusals):
    async def scenario():
        class RepeatedRefusalFactory(_RecoveringFactory):
            async def create(self, request):
                if request.operation is EnvironmentFactoryOperation.RECONNECT:
                    assert request.reconnect_metadata == {
                        "allocation_id": f"allocation-{len(self.requests)}"
                    }
                    assert self.second_runner.is_closed
                    self.requests.append(request)
                    raise FileNotFoundError(
                        "attempted to reconnect an already discarded allocation"
                    )
                # Each create returns a distinct allocation. The first allocations
                # fail final admission; after those the provider is healthy again.
                self.first_runner = self.second_runner = _EvidenceRunner(
                    "hosted-b" if len(self.requests) < refusals else "hosted-a"
                )
                return await super().create(request)

        factory = RepeatedRefusalFactory()
        provider = _RecordingProvider()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(EnvironmentSpec(name="hosted"), factory, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            execution_requirements=_requirements(),
        )
        session_id = "repeated-admission-refusal"

        async def resume():
            return [
                event
                async for event in app.resume(
                    ResumeRequest(session_id=session_id, messages=[Message.text("user", "retry")])
                )
            ]

        events = await _run(app, session_id)
        for attempt in range(refusals):
            assert events[-1].type is EventType.SESSION_FAILED
            assert factory.release_actions == [EnvironmentFactoryReleaseAction.DISCARD] * (
                attempt + 1
            )
            events = await resume()

        operations = [request.operation.value for request in factory.requests]
        assert operations == ["create"] * (refusals + 1), (
            "Resume reconnected a discarded allocation after repeated refusal",
            operations,
        )
        assert events[-1].type is EventType.SESSION_COMPLETED
        assert len(provider.requests) == 1

    asyncio.run(scenario())
