"""SubagentSpec.environment_name places a child in its own environment (#2288)."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentFactory,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
    EnvironmentSpec,
    EventType,
    InMemorySessionStore,
    Message,
    ModelStreamEvent,
    RunRequest,
    ScriptedModelProvider,
    SessionQuery,
    SessionStatus,
    SubagentSpec,
    SubagentTool,
)

_TASK = "Edit your own checkout."


class _RecordingFactory(EnvironmentFactory):
    def __init__(self) -> None:
        self.requests: list[EnvironmentFactoryRequest] = []

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        self.requests.append(request)
        return EnvironmentFactoryResult(
            environment=Environment(EnvironmentSpec(name=request.environment_name))
        )


def _provider() -> ScriptedModelProvider:
    return ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_builder",
                    name="subagent",
                    arguments={"agent": "builder", "task": _TASK},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("built"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("parent done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )


def _tool(app: CayuApp, environment_name: str | None) -> SubagentTool:
    return SubagentTool(
        app,
        agents={
            "builder": SubagentSpec(
                agent_name="builder",
                description="Builds in its own checkout.",
                environment_name=environment_name,
            )
        },
    )


def test_foreground_child_runs_in_the_spec_environment_and_recovery_matches_it() -> None:
    store = InMemorySessionStore()
    factory = _RecordingFactory()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(_provider(), default=True)
    app.register_environment(Environment(EnvironmentSpec(name="lead")), default=True)
    app.register_environment_factory(EnvironmentSpec(name="builder-checkout"), factory)
    tool = _tool(app, "builder-checkout")
    app.register_agent(AgentSpec(name="lead", model="scripted-model"), tools=[tool])
    app.register_agent(AgentSpec(name="builder", model="scripted-model"))

    async def run() -> list:
        return [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="lead",
                    session_id="sess_lead",
                    messages=[Message.text("user", "delegate")],
                )
            )
        ]

    events = asyncio.run(run())

    assert events[-1].type is EventType.SESSION_COMPLETED
    parent = asyncio.run(store.load("sess_lead"))
    (child,) = asyncio.run(
        store.list_sessions(SessionQuery(parent_session_id="sess_lead"))
    ).sessions
    assert parent is not None
    assert parent.environment_name == "lead"
    assert child.environment_name == "builder-checkout"
    assert child.status is SessionStatus.COMPLETED
    assert [
        (request.session_id, request.parent_session_id, request.agent_name)
        for request in factory.requests
    ] == [(child.id, "sess_lead", "builder")]

    subagent = child.metadata["subagent"]
    match_arguments = {
        "child": child,
        "parent_invocation": parent.invocation,
        "parent_session_id": parent.id,
        "causal_budget_id": parent.causal_budget_id,
        "environment_name": parent.environment_name,
        "tool_call_id": subagent["tool_call_id"],
        "idempotency_key": subagent["idempotency_key"],
        "arguments": {"agent": "builder", "task": _TASK},
        "require_fingerprint": True,
    }
    assert tool.matches_recoverable_child(**match_arguments)
    # A spec that no longer names the child's environment cannot re-attach it.
    for changed in (None, "lead", "another-checkout"):
        assert not _tool(app, changed).matches_recoverable_child(**match_arguments)


def test_child_environment_defaults_to_the_parent_environment() -> None:
    app = CayuApp(enable_logging=False)
    arguments = {"agent": "builder", "task": _TASK}

    inherited = _tool(app, None)
    overridden = _tool(app, "builder-checkout")

    assert (
        inherited.recoverable_child_environment_name(
            arguments=arguments, parent_environment_name="lead"
        )
        == "lead"
    )
    assert (
        overridden.recoverable_child_environment_name(
            arguments=arguments, parent_environment_name="lead"
        )
        == "builder-checkout"
    )
    assert (
        overridden.recoverable_child_environment_name(
            arguments={"agent": "unknown", "task": _TASK}, parent_environment_name="lead"
        )
        == "lead"
    )


@pytest.mark.parametrize("value", ["", "  ", " padded "])
def test_subagent_spec_rejects_blank_or_padded_environment_names(value: str) -> None:
    with pytest.raises(ValidationError):
        SubagentSpec(agent_name="builder", environment_name=value)
