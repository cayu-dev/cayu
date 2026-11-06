"""An explicitly registered MCP source stays auditable after its last tool is removed."""

from __future__ import annotations

import asyncio

import pytest
from tests.core.test_mcp import FakeProvider, _fake_tool_definitions, _fake_toolset

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.providers.base import ModelStreamEvent
from cayu.runtime.mcp_manifest_policy import McpManifestPolicy
from cayu.sessions.memory import InMemorySessionStore


@pytest.mark.parametrize("remaining_names", [("read",), ()], ids=["one-tool", "empty-source"])
def test_mcp_removal_policy_checks_even_an_empty_registered_source(remaining_names):
    async def scenario():
        store = InMemorySessionStore()

        async def run_snapshot(session_id, names):
            provider = FakeProvider(
                [[ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed({})]]
            )
            app = CayuApp(
                session_store=store,
                enable_logging=False,
                mcp_manifest_policy=McpManifestPolicy(),
            )
            app.register_provider(provider, default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                mcp_toolsets=(_fake_toolset(definitions=_fake_tool_definitions(*names)),),
            )
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "go")],
                    )
                )
            ]
            return events, provider.requests

        first, first_requests = await run_snapshot("before-removal", ("read", "safety_check"))
        assert len(first_requests) == 1
        assert any(event.type is EventType.MCP_MANIFEST_CHECKED for event in first)

        # A new connection to the same explicit connection_id advertises a
        # changed catalogue. Retain the source via mcp_toolsets, not tools=[].
        events, requests = await run_snapshot("after-removal", remaining_names)
        assert requests == [], "The model ran despite the configured manifest-change block"
        blocked = [event for event in events if event.type is EventType.MCP_MANIFEST_BLOCKED]
        assert len(blocked) == 1
        assert blocked[0].payload["status"] == "changed"
        assert "tools_removed" in blocked[0].payload["change_classes"]

    asyncio.run(scenario())
