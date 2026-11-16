"""Round-wide secret quarantine also applies to runner completion evidence."""

from __future__ import annotations

import asyncio
import sys

import pytest
from tests.core._workload_secret_support import FakeProvider, collect_events

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.configuration import CayuConfig, ToolExecutionConfig
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.events import EventType
from cayu.messages import Message
from cayu.observability.events import InMemoryEventSink
from cayu.providers.base import ModelStreamEvent
from cayu.runners.base import ExecCommand
from cayu.runners.local import LocalRunner
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.requests import RunRequest
from cayu.tools.base import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.tools.commands import ExecCommandTool
from cayu.vaults.base import SecretRef
from cayu.vaults.redaction import REDACTED_SECRET
from cayu.vaults.static import StaticVault


@pytest.mark.parametrize("max_parallel_tool_calls", [1, 4])
@pytest.mark.parametrize("command_tool_kind", ["builtin", "custom"])
def test_runner_completion_uses_the_final_round_secret_scope(
    tmp_path, max_parallel_tool_calls, command_tool_kind
):
    secret = "runner-sibling-secret-canary-ABCDEFGHIJKLMNOP"

    class EchoCommand(Tool):
        spec = ToolSpec(
            name="echo_command",
            description="Print a supplied value without modifying anything.",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            effect=ToolEffect.NONE,
            parallel_safe=True,
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            assert ctx.runner is not None
            result = await ctx.runner.exec(
                ExecCommand.process(
                    sys.executable, "-c", "import sys; print(sys.argv[1])", args["text"]
                )
            )
            return ToolResult(content=result.stdout)

    class ResolveSiblingSecret(Tool):
        # This is a sequential barrier after the parallel echo segment.
        spec = ToolSpec(
            name="resolve_sibling_secret",
            description="Resolve the shared credential.",
            input_schema={"type": "object"},
            effect=ToolEffect.NONE,
            parallel_safe=False,
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            assert ctx.vault is not None
            await ctx.vault.resolve(SecretRef(name="shared"))
            return ToolResult(content="resolved")

    async def scenario():
        command_tool = ExecCommandTool() if command_tool_kind == "builtin" else EchoCommand()
        command_name = command_tool.spec.name
        command_arguments = (
            {"argv": [sys.executable, "-c", "import sys; print(sys.argv[1])", secret]}
            if command_tool_kind == "builtin"
            else {"text": secret}
        )
        store = InMemorySessionStore()
        sink = InMemoryEventSink()
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="echo-a", name=command_name, arguments=command_arguments
                    ),
                    ModelStreamEvent.tool_call(
                        id="echo-b", name=command_name, arguments=command_arguments
                    ),
                    ModelStreamEvent.tool_call(
                        id="resolve", name="resolve_sibling_secret", arguments={}
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        app = CayuApp(
            session_store=store,
            event_sinks=[sink],
            enable_logging=False,
            config=CayuConfig(
                tool_execution=ToolExecutionConfig(max_parallel_tool_calls=max_parallel_tool_calls)
            ),
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                runner=LocalRunner(tmp_path),
                vault=StaticVault({"shared": secret}),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[command_tool, ResolveSiblingSecret()],
        )
        events = await collect_events(
            app,
            RunRequest(
                session_id="runner-round-secret",
                agent_name="assistant",
                messages=[Message.text("user", "run the tools")],
            ),
        )
        assert events[-1].type is EventType.SESSION_COMPLETED
        terminals = [
            event
            for event in events
            if event.type is EventType.TOOL_CALL_COMPLETED and event.tool_name == command_name
        ]
        assert len(terminals) == 2
        assert all(REDACTED_SECRET in event.payload["result"]["content"] for event in terminals)
        assert secret not in repr(await store.load_transcript("runner-round-secret"))
        assert secret not in repr(provider.requests[1].messages)

        durable = await store.load_events("runner-round-secret")
        completions = [event for event in durable if event.type is EventType.RUNNER_EXEC_COMPLETED]
        assert len(completions) == 2
        # The terminal results are safe; runner events must not bypass the same
        # quarantine and disclose the original, subsequently-redacted arguments.
        assert secret not in repr([event.payload for event in completions])
        assert secret not in repr(sink.events)

    asyncio.run(scenario())
