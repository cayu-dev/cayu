"""Runtime discovery calls remain valid siblings when an ordinary round pauses."""

from __future__ import annotations

import asyncio

import pytest

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.evals.testing import ScriptedModelProvider
from cayu.messages import ToolResultPart
from cayu.providers.base import ModelRequest, ModelStreamEvent
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionStatus
from cayu.tools import UserInputTool
from cayu.tools.exposure import StaticToolExposurePolicy


@pytest.mark.parametrize("delivery", ["direct", "discovered"])
@pytest.mark.parametrize("discovery_sibling", [False, True], ids=["question-only", "with-search"])
def test_user_input_pause_retains_authorized_runtime_discovery_sibling(delivery, discovery_sibling):
    def respond(request: ModelRequest):
        assert "search_tools" in {tool["name"] for tool in request.tools}
        arguments = {"question": "Which tool should I use?"}
        question_name = "ask_user"
        if delivery == "discovered":
            results = [
                part
                for message in request.messages
                for part in message.content
                if isinstance(part, ToolResultPart) and part.tool_name == "search_tools"
            ]
            if not results:
                return [
                    ModelStreamEvent.tool_call(
                        id="discover",
                        name="search_tools",
                        arguments={"query": "ask user question"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ]
            [match] = results[-1].structured["matches"]
            assert match["name"] == "ask_user"
            question_name = "call_tool"
            arguments = {"tool_ref": match["tool_ref"], "arguments": arguments}
        calls = []
        if discovery_sibling:
            calls.append(
                ModelStreamEvent.tool_call(
                    id="search", name="search_tools", arguments={"query": "find a tool"}
                )
            )
        calls.append(
            ModelStreamEvent.tool_call(id="question", name=question_name, arguments=arguments)
        )
        return [*calls, ModelStreamEvent.completed({"finish_reason": "tool_calls"})]

    async def scenario():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(ScriptedModelProvider(response_factory=respond), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[UserInputTool()],
            tool_discovery_mode="search_tools",
            tool_exposure_policy=(
                StaticToolExposurePolicy(profile_id="questions", tools=("ask_user",))
                if delivery == "direct"
                else None
            ),
        )
        session_id = f"discovery-pause-{delivery}-{discovery_sibling}"
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "Help me choose.")],
                )
            )
        ]
        failures = [event.payload for event in events if event.type is EventType.SESSION_FAILED]
        assert not failures, failures
        assert any(event.type is EventType.SESSION_AWAITING_USER_INPUT for event in events)
        assert events[-1].type is EventType.SESSION_INTERRUPTED
        assert (await store.load(session_id)).status is SessionStatus.INTERRUPTED
        checkpoint = await store.load_checkpoint(session_id)
        pending = checkpoint["pending_user_input"]
        assert [call["tool_call_id"] for call in pending["tool_calls"]] == (
            ["search", "question"] if discovery_sibling else ["question"]
        )

    asyncio.run(scenario())
