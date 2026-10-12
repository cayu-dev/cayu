"""Tool hook support."""

from __future__ import annotations

import asyncio

from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_session_support import collect_events

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.messages import Message
from cayu.observability.hooks import BeforeToolCallDecision, BeforeToolCallHookContext, RuntimeHook
from cayu.providers.base import ModelStreamEvent
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.requests import RunRequest
from cayu.tools.base import Tool, ToolContext, ToolResult, ToolSpec


class _RecordingEchoTool(Tool):
    spec = ToolSpec(
        name="echo",
        description="Echo text, recording each invocation.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.calls.append(args)
        return ToolResult(content=args["text"], structured={"echoed": args["text"]})


def _tool_call_provider() -> FakeProvider:
    return FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_echo",
                    name="echo",
                    arguments={"text": "original"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )


class _RewriteTextHook(RuntimeHook):
    def __init__(self, value: str) -> None:
        self._value = value

    async def before_tool_call(self, context: BeforeToolCallHookContext) -> BeforeToolCallDecision:
        modified = dict(context.arguments)
        modified["text"] = self._value
        return BeforeToolCallDecision(action="proceed_modified", modified_arguments=modified)


def _run_reauth_case(session_id, policy, hooks, extra_hooks=()):
    store = InMemorySessionStore()
    tool = _RecordingEchoTool()
    app = CayuApp(session_store=store)
    app.register_provider(_tool_call_provider(), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=policy,
        runtime_hooks=[*hooks, *extra_hooks],
    )
    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "go")],
            ),
        )
    )
    transcript = asyncio.run(store.load_transcript(session_id))
    return tool, events, transcript
