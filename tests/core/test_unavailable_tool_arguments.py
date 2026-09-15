from __future__ import annotations

import asyncio
import io
import json

import pytest
from tests.core._workload_secret_support import FakeProvider, collect_events

from cayu import AgentSpec, CayuApp, Environment, EnvironmentSpec, LocalRunner
from cayu.events import EventType
from cayu.messages import Message, ToolCallPart
from cayu.providers.anthropic import build_anthropic_payload
from cayu.providers.base import ModelRequest, ModelStreamEvent
from cayu.providers.bedrock import build_bedrock_converse_payload
from cayu.providers.chat_completions import build_chat_completions_payload
from cayu.providers.openai import build_openai_payload
from cayu.sessions.base import RunRequest
from cayu.storage.jsonl_export import export_sessions
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.base import Tool, ToolResult, ToolSpec
from cayu.tools.commands import (
    CommandPolicy,
    CommandPolicyDecision,
    CommandPolicyResult,
    ExecCommandTool,
)


@pytest.mark.parametrize(
    "builder",
    [
        build_openai_payload,
        build_chat_completions_payload,
        build_anthropic_payload,
        build_bedrock_converse_payload,
    ],
)
def test_provider_history_distinguishes_unavailable_from_empty(builder):
    def payload(state):
        call = ToolCallPart(tool_call_id="call_1", tool_name="exec_command", arguments_state=state)
        request = ModelRequest(
            model="test-model",
            messages=[
                Message.tool_call(calls=[call]),
                Message.tool_result(
                    tool_call_id="call_1", tool_name="exec_command", content="denied", is_error=True
                ),
            ],
        )
        return json.dumps(builder(request))

    assert "__cayu_arguments_unavailable__" in payload("unavailable")
    assert "operator to authorize" in payload("unavailable")
    assert "__cayu_arguments_unavailable__" not in payload("finalized")


@pytest.mark.parametrize("command_kind", ["process", "shell"])
def test_command_denial_availability_survives_restart_export_and_continuation(
    tmp_path, command_kind
):
    class Deny(CommandPolicy):
        async def evaluate(self, ctx, request):
            return CommandPolicyResult(
                decision=CommandPolicyDecision.DENY,
                reason="Only configured process capabilities may run.",
            )

    class Empty(Tool):
        spec = ToolSpec(
            name="empty", description="Accept an empty object.", input_schema={"type": "object"}
        )

        async def run(self, ctx, args):
            assert args == {}
            return ToolResult(content="empty accepted")

    async def run():
        secret = "unregistered-credential-canary-1737"
        database = tmp_path / "sessions.sqlite"
        store = SQLiteSessionStore(database)
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="denied",
                        name="exec_command",
                        arguments={
                            **(
                                {"argv": ["/bin/echo", secret]}
                                if command_kind == "process"
                                else {"shell": f"echo {secret}"}
                            ),
                            "env": {"TOKEN": secret},
                            "stdin": secret,
                        },
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.tool_call(id="empty_call", name="empty", arguments={}),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), runner=LocalRunner(tmp_path)), default=True
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[ExecCommandTool(policy=Deny()), Empty()],
        )
        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant", session_id="denial", messages=[Message.text("user", "run")]
            ),
        )
        assert events[-1].type is EventType.SESSION_COMPLETED, repr(events)
        terminal = next(e for e in events if e.type is EventType.TOOL_CALL_BLOCKED)
        assert terminal.payload["arguments_state"] == "unavailable"
        assert terminal.payload["arguments_exact"] is False
        assert (
            "application operator"
            in terminal.payload["result"]["structured"]["recovery_instruction"]
        )
        history = provider.requests[1].messages
        call = next(p for m in history for p in m.content if isinstance(p, ToolCallPart))
        assert call.arguments_state == "unavailable"
        assert call.arguments == {}
        assert secret not in repr(events) + repr(history)
        empty_call = next(
            p
            for m in provider.requests[2].messages
            for p in m.content
            if isinstance(p, ToolCallPart) and p.tool_name == "empty"
        )
        assert empty_call.arguments_state == "finalized"
        assert empty_call.arguments == {}
        await store.close()

        reopened = SQLiteSessionStore(database)
        transcript = await reopened.load_transcript("denial")
        assert secret not in repr(transcript)
        output = io.StringIO()
        assert await export_sessions(reopened, stream=output) == 1
        assert secret not in output.getvalue()
        assert '"arguments_state": "unavailable"' in output.getvalue()
        # Round-trip the public message contract independently of the live app.
        restored = Message.model_validate_json(Message.tool_call(calls=[call]).model_dump_json())
        restored_call = restored.content[0]
        assert isinstance(restored_call, ToolCallPart)
        assert restored_call.arguments_state == "unavailable"
        assert "__cayu_arguments_unavailable__" in restored_call.continuation_arguments()
        await reopened.close()

    asyncio.run(run())
