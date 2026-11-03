from __future__ import annotations

import asyncio

from tests.core.test_queued_session_messages import BlockingTwoTurnProvider
from tests.core.test_targeted_tool_grants import _RememberTool

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.events import EventType
from cayu.messages import Message
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.messaging import EnqueueSessionMessageRequest, SessionMessageDeliveryMode
from cayu.sessions.requests import RunRequest
from cayu.tools.grants import TargetedToolGrant


def test_queued_interaction_can_continue_after_targeted_grants_expire():
    async def run():
        store = InMemorySessionStore()
        provider = BlockingTwoTurnProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            targeted_tool_mode="call_tool",
            tools=[_RememberTool()],
        )
        events = []

        async def consume():
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="queued-grant-handoff",
                    messages=[Message.text("user", "first request")],
                    tool_grants=(
                        TargetedToolGrant(
                            request_id="remember-fact",
                            tool_id="cayu:remember",
                            max_calls=1,
                        ),
                    ),
                )
            ):
                events.append(event)

        owner = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(provider.first_started.wait(), timeout=10)
            await app.enqueue_session_message(
                EnqueueSessionMessageRequest(
                    session_id="queued-grant-handoff",
                    idempotency_key="follow-up",
                    content="answer this follow-up without tools",
                    delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
                )
            )
            provider.release_first.set()
            await asyncio.wait_for(owner, timeout=10)
            assert any(event.type == EventType.SESSION_MESSAGE_DELIVERED for event in events)
            assert events[-1].type == EventType.SESSION_COMPLETED, [
                (event.type.value, event.payload.get("error"), event.payload.get("error_type"))
                for event in events
                if "error" in event.payload
            ]
            assert len(provider.requests) == 2
        finally:
            provider.release_first.set()
            if not owner.done():
                owner.cancel()
                await asyncio.gather(owner, return_exceptions=True)
            await app.aclose()

    asyncio.run(run())
