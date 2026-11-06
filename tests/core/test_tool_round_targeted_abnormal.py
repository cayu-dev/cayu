"""Abnormal parallel terminals must retain the resolved gateway's authority."""

from __future__ import annotations

import asyncio
import json

import pytest
from tests.core.test_targeted_tool_grants import _GatewayRememberTool, _Provider

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.messages import ToolResultPart
from cayu.providers.base import ModelStreamEvent
from cayu.tools.base import ToolResult
from cayu.tools.exposure import StaticToolExposurePolicy
from cayu.tools.grants import TargetedToolGrant


@pytest.mark.parametrize("cancel_first", [False, True])
def test_parallel_gateway_child_cancellation_closes_with_linked_terminal(cancel_first):
    async def scenario():
        class CancelOneTool(_GatewayRememberTool):
            async def run(self, ctx, args):
                if cancel_first and args["fact"] == "cancel":
                    raise asyncio.CancelledError()
                return ToolResult(content="sibling completed")

        class GatewayProvider(_Provider):
            async def stream(self, request):
                self.requests.append(request)
                if len(self.requests) == 1:
                    context = next(
                        message.content[0].text
                        for message in request.messages
                        if message.role == "user"
                        and message.content[0].text.startswith("Cayu runtime targeted-tool context")
                    )
                    [descriptor] = json.loads(context.rsplit("\n", 1)[1])["tools"]
                    for call_id, fact in (("cancelled", "cancel"), ("sibling", "ok")):
                        yield ModelStreamEvent.tool_call(
                            id=call_id,
                            name="call_tool",
                            arguments={
                                "tool_ref": descriptor["tool_ref"],
                                "arguments": {"fact": fact},
                            },
                        )
                    yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                else:
                    yield ModelStreamEvent.text_delta("done")
                    yield ModelStreamEvent.completed({"finish_reason": "stop"})

        app = CayuApp(enable_logging=False)
        app.register_provider(GatewayProvider(), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[CancelOneTool()],
            targeted_tool_mode="call_tool",
            tool_exposure_policy=StaticToolExposurePolicy(profile_id="targeted", tools=()),
        )
        session_id = "gateway-parallel-abnormal"
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "Run both")],
                    tool_grants=(
                        TargetedToolGrant(
                            request_id="remember-twice", tool_id="cayu:remember", max_calls=2
                        ),
                    ),
                )
            )
        ]
        assert events[-1].type is EventType.SESSION_COMPLETED, events[-1].payload
        transcript = await app.session_store.load_transcript(session_id)
        results = [
            part
            for message in transcript
            for part in message.content
            if isinstance(part, ToolResultPart)
        ]
        assert [part.tool_call_id for part in results] == ["cancelled", "sibling"]
        assert results[0].is_error is cancel_first
        assert results[1].content == "sibling completed"

    asyncio.run(scenario())
