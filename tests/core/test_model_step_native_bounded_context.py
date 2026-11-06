from __future__ import annotations

import asyncio

import pytest
from tests.core.test_targeted_tool_grants import (
    _codec,
    _GatewayRememberTool,
    _NativeOpenAITransport,
    _NativeTestOpenAIProvider,
)

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.context import MessageWindowContextPolicy
from cayu.sessions.memory import InMemorySessionStore
from cayu.tools import TargetedToolGrant
from cayu.tools.exposure import StaticToolExposurePolicy


@pytest.mark.parametrize("max_messages", [2, 4])
def test_native_grant_continuation_survives_bounded_context(max_messages):
    async def run():
        store = InMemorySessionStore(public_authority_alias_codec=_codec())
        transport = _NativeOpenAITransport()
        provider = _NativeTestOpenAIProvider(
            api_key="test-key",
            transport=transport,
            additional_tools_models=("fake-model",),
        )
        tool = _GatewayRememberTool()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=(tool,),
            targeted_tool_mode="openai_additional_tools",
            tool_exposure_policy=StaticToolExposurePolicy(profile_id="targeted-only", tools=()),
            context_policy=MessageWindowContextPolicy(max_messages=max_messages),
        )
        try:
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="native-bounded-context",
                        messages=[Message.text("user", "Remember this reviewed fact")],
                        tool_grants=(
                            TargetedToolGrant(
                                request_id="remember-fact",
                                tool_id="cayu:remember",
                                max_calls=1,
                            ),
                        ),
                    )
                )
            ]
            assert tool.calls == [{"fact": "Keep native identity stable."}]
            assert events[-1].type is EventType.SESSION_COMPLETED, events[-1].payload
            assert len(transport.calls) == 2
            assert any(
                item.get("type") == "additional_tools" for item in transport.calls[-1]["input"]
            )
        finally:
            await app.aclose()

    asyncio.run(run())
