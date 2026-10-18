"""Finishing keeps signed thinking valid while withdrawing native tool state."""

from __future__ import annotations

import asyncio

import pytest
from tests.core.test_run_limit_finish import ReadTool, answer, spec
from tests.core.test_thinking import _anthropic_state

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.budgets.run_limits import RunLimits
from cayu.context import RecentTurnsContextPolicy, ThinkingConfig
from cayu.evals.testing import ScriptedModelProvider
from cayu.messages import ThinkingPart
from cayu.providers import ModelContextOverflowError, ModelStreamEvent
from cayu.providers.anthropic import build_anthropic_payload


@pytest.mark.parametrize("strategy", ["native", "tool"])
@pytest.mark.parametrize("redacted", [False, True])
@pytest.mark.parametrize("overflow", [False, True])
def test_finishing_preserves_signed_thinking_after_tool_use(strategy, redacted, overflow):
    state = (
        _anthropic_state("redacted_thinking", data="opaque-thinking")
        if redacted
        else _anthropic_state("thinking", signature="signed-thinking")
    )
    expected = (
        {"type": "redacted_thinking", "data": "opaque-thinking"}
        if redacted
        else {
            "type": "thinking",
            "thinking": "Inspect the evidence.",
            "signature": "signed-thinking",
        }
    )

    class Provider(ScriptedModelProvider):
        async def stream(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                yield ModelStreamEvent.thinking(
                    "" if redacted else "Inspect the evidence.", provider_state=state
                )
                yield ModelStreamEvent.tool_call(id="read-1", name="read", arguments={})
                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            elif len(self.requests) == 2 and overflow:
                raise ModelContextOverflowError("too large", provider=self.name)
            else:
                for event in answer(strategy):
                    yield event

    async def scenario():
        provider = Provider([], supports_native_structured_output=True)
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(
                name="agent", model="fake", thinking=ThinkingConfig(enabled=True, max_tokens=2048)
            ),
            tools=[ReadTool()],
            context_overflow_policy=RecentTurnsContextPolicy(max_user_turns=1),
        )
        try:
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="agent",
                        session_id="finish-thinking",
                        messages=[Message.text("user", "Find the answer")],
                        max_steps=2,
                        limits=RunLimits(on_limit="finish"),
                        structured_output=spec(strategy),
                    )
                )
            ]
            assert any(event.type is EventType.STRUCTURED_OUTPUT_VALIDATED for event in events)
            assert len(provider.requests) == (3 if overflow else 2)
            for request in provider.requests[1:]:
                payload = build_anthropic_payload(request)
                assert payload["thinking"]["type"] == "enabled"
                assistant = next(
                    message for message in payload["messages"] if message["role"] == "assistant"
                )
                assert assistant["content"][0] == expected
                assert assistant["content"][1]["type"] == "tool_use"
                assert "read" not in {tool["name"] for tool in request.tools}
            transcript = await app.session_store.load_transcript("finish-thinking")
            thoughts = [
                part
                for message in transcript
                for part in message.content
                if isinstance(part, ThinkingPart)
            ]
            assert len(thoughts) == 1 and thoughts[0].provider_state == state
        finally:
            await app.aclose()

    asyncio.run(scenario())
