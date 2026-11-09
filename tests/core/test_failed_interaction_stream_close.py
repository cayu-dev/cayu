"""Closing after a failed interaction must retain the session terminal publication."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest, SessionStatus
from cayu.providers import ModelProvider, ModelRequest
from cayu.sessions.memory import InMemorySessionStore


class _FailingProvider(ModelProvider):
    name = "failing-close"

    async def stream(self, request: ModelRequest) -> AsyncIterator[object]:
        del request
        raise RuntimeError("provider failed")
        yield  # pragma: no cover


def test_closing_after_interaction_failed_keeps_session_terminal_evidence() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(_FailingProvider(), default=True)
        app.register_agent(AgentSpec(name="assistant", model="failing-model"))

        stream = app.run(
            RunRequest(
                agent_name="assistant",
                session_id="failed-close",
                messages=[Message.text("user", "fail this turn")],
            )
        )
        while True:
            event = await anext(stream)
            if event.type is EventType.INTERACTION_FAILED:
                break
        await stream.aclose()

        assert (await store.load("failed-close")).status is SessionStatus.FAILED
        events = await store.load_events("failed-close")
        assert [event.type for event in events].count(EventType.INTERACTION_FAILED) == 1
        assert [event.type for event in events].count(EventType.TURN_COMPLETED) == 1
        assert [event.type for event in events].count(EventType.SESSION_FAILED) == 1

    asyncio.run(scenario())
