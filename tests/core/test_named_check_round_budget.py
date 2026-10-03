"""Model-step scheduling with real named-check dispatch, not live model quality."""

import asyncio
import sys

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentSpec,
    ExecCommand,
    ExecutionProfileBehaviorIdentity,
    LocalRunner,
    LocalWorkspace,
    Message,
    ModelStreamEvent,
    NamedCheck,
    ProcessCommandPolicy,
    RunCheckTool,
    RunRequest,
    ScriptedModelProvider,
    StaticToolPolicy,
)


@pytest.mark.parametrize("grouped", [False, True])
def test_grouped_named_checks_leave_room_for_final_response(tmp_path, grouped):
    async def scenario():
        names = ("format", "lint", "test", "independent-probe")
        checks = tuple(
            NamedCheck(
                name=name,
                description="Check fixture input through a real process.",
                command=ExecCommand.process(
                    sys.executable, "-c", "assert open('value.txt').read() == 'fixture'"
                ),
                execution_profile_identity=ExecutionProfileBehaviorIdentity(
                    name=f"test.{name}", behavior_version="1", implementation_version="1"
                ),
            )
            for name in names
        )
        (tmp_path / "value.txt").write_text("fixture")
        workspace = LocalWorkspace(tmp_path, workspace_id="round-budget")

        def calls(selected, prefix):
            return [
                *(
                    ModelStreamEvent.tool_call(
                        id=f"{prefix}-{i}", name="run_check", arguments={"check": name}
                    )
                    for i, name in enumerate(selected)
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]

        # Three setup observations, required final checks, then final observation.
        # This isolates scheduling, not a substitute for a real repair campaign.
        responses = [calls(("test",), f"setup-{i}") for i in range(3)]
        responses += (
            [calls(names, "final")]
            if grouped
            else [calls((name,), f"final-{name}") for name in names]
        )
        responses += [
            calls(("test",), "observe"),
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
        app = CayuApp(enable_logging=False)
        app.register_provider(ScriptedModelProvider(responses), default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"), workspace=workspace, runner=LocalRunner(tmp_path)
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="agent", model="fixture"),
            tool_policy=StaticToolPolicy(allow=("run_check",)),
            tools=[
                RunCheckTool(
                    checks=checks,
                    command_policy=ProcessCommandPolicy(
                        allowed_executables=(sys.executable,), allowed_cwds=(str(tmp_path),)
                    ),
                )
            ],
        )
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="agent",
                    max_steps=8,
                    messages=[Message.text("user", "Run checks, then report.")],
                )
            )
        ]
        completed = [e for e in events if e.type.value == "tool.call.completed"]
        assert len(completed) == 8, [
            (e.type.value, e.payload)
            for e in events
            if e.type.value in {"session.failed", "session.interrupted", "tool.call.failed"}
        ]
        assert all(e.payload["result"]["is_error"] is False for e in completed)
        assert events[-1].type.value == ("session.completed" if grouped else "session.interrupted")
        assert sum(e.type.value == "model.started" for e in events) == (6 if grouped else 8)
        assert any(e.type.value == "session.limit_reached" for e in events) is (not grouped)

    asyncio.run(scenario())
