"""Hook changes must be compared using JSON types, not Python numeric equality."""

import asyncio

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest, Tool, ToolResult, ToolSpec
from cayu.evals import ScriptedModelProvider
from cayu.observability.hooks import BeforeToolCallDecision, RuntimeHook
from cayu.providers import ModelStreamEvent
from cayu.sessions import InMemorySessionStore
from cayu.tools.base import ToolEffect
from cayu.tools.policy import ToolPolicy, ToolPolicyDecision, ToolPolicyResult


def test_before_hook_can_change_integer_one_to_boolean_true():
    async def scenario():
        executed = []
        authorized = []

        class Echo(Tool):
            spec = ToolSpec(
                name="echo",
                effect=ToolEffect.NONE,
                input_schema={
                    "type": "object",
                    "properties": {"value": {"type": ["integer", "boolean"]}},
                    "required": ["value"],
                },
            )

            async def run(self, ctx, args):
                executed.append(args["value"])
                return ToolResult(content=type(args["value"]).__name__)

        class Normalize(RuntimeHook):
            async def before_tool_call(self, context):
                return BeforeToolCallDecision(
                    action="proceed_modified", modified_arguments={"value": True}
                )

        class Policy(ToolPolicy):
            async def authorize(self, request):
                authorized.append(request.arguments["value"])
                return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)

        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            ScriptedModelProvider(
                [
                    [
                        ModelStreamEvent.tool_call(
                            id="echo-1", name="echo", arguments={"value": 1}
                        ),
                        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                    ],
                    [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed({})],
                ]
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="worker", model="scripted-model"),
            tools=[Echo()],
            runtime_hooks=[Normalize()],
            tool_policy=Policy(),
        )
        events = [
            event
            async for event in app.run(
                RunRequest(
                    session_id="hook-types",
                    agent_name="worker",
                    messages=[Message.text("user", "go")],
                )
            )
        ]
        assert events[-1].type is EventType.SESSION_COMPLETED
        assert len(executed) == 1
        assert executed[0] is True, (
            "The hook's boolean was silently replaced by the model's integer"
        )
        assert len(authorized) == 2 and authorized[-1] is True
        [terminal] = [event for event in events if event.type is EventType.TOOL_CALL_COMPLETED]
        assert terminal.payload["effective_arguments"]["value"] is True

    asyncio.run(scenario())
