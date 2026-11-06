"""Agent-level execution requirements must fail closed without an environment."""

import asyncio

import pytest
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_session_support import collect_events

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.environments import Environment, EnvironmentSpec, ExecutionRequirements
from cayu.providers import ModelStreamEvent


@pytest.mark.parametrize("configure_environment", [False, True], ids=["absent", "empty"])
@pytest.mark.parametrize(
    "requirements",
    [
        ExecutionRequirements.untrusted(),
        ExecutionRequirements.trusted(required_executables=("python3",)),
    ],
    ids=["isolation", "required-executable"],
)
def test_no_environment_refuses_unsatisfied_agent_execution_requirements(
    requirements, configure_environment
):
    async def scenario():
        provider = FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})])
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        if configure_environment:
            app.register_environment(Environment(EnvironmentSpec(name="empty")), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            execution_requirements=requirements,
        )
        events = await collect_events(
            app,
            RunRequest(
                session_id="missing-execution-environment",
                agent_name="assistant",
                messages=[Message.text("user", "run")],
            ),
        )
        assert events[-1].type is EventType.SESSION_FAILED
        assert provider.requests == [], (
            "The provider ran despite missing required execution evidence"
        )

    asyncio.run(scenario())


def test_trusted_agent_with_no_requirements_can_run_without_an_environment():
    async def scenario():
        provider = FakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})])
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        events = await collect_events(
            app,
            RunRequest(agent_name="assistant", messages=[Message.text("user", "run")]),
        )
        assert events[-1].type is EventType.SESSION_COMPLETED
        assert len(provider.requests) == 1

    asyncio.run(scenario())
