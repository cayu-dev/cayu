"""Explicitly spend-authorized live acceptance for OpenAI hosted Tool Search."""

from __future__ import annotations

import os

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    Message,
    OpenAIProvider,
    RunRequest,
    Tool,
    ToolContext,
    ToolEffect,
    ToolResult,
    ToolSpec,
)


class _HostedContractTool(Tool):
    spec = ToolSpec(
        name="return_hosted_tool_search_contract_nonce",
        description="Return the exact nonce for the hosted Tool Search live contract.",
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "nonce": {
                    "type": "string",
                    "const": "cayu-hosted-tool-search-live",
                }
            },
            "required": ["nonce"],
        },
        effect=ToolEffect.NONE,
    )

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, str]] = []

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del ctx
        self.calls.append(dict(args))
        return ToolResult(content="hosted Tool Search contract passed")


@pytest.mark.anyio
async def test_openai_hosted_tool_search_live() -> None:
    if os.environ.get("CAYU_OPENAI_HOSTED_TOOL_SEARCH_LIVE") != "1":
        pytest.skip("set CAYU_OPENAI_HOSTED_TOOL_SEARCH_LIVE=1 to spend API credits")
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not configured")

    model = os.environ.get(
        "CAYU_OPENAI_HOSTED_TOOL_SEARCH_MODEL",
        "gpt-5.4-mini-2026-03-17",
    )
    tool = _HostedContractTool()
    app = CayuApp(enable_logging=False)
    app.register_provider(
        OpenAIProvider(hosted_tool_search_models=(model,)),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="hosted-tool-search-live", model=model),
        tools=(tool,),
        tool_discovery_mode="openai_tool_search_hosted",
    )

    events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="hosted-tool-search-live",
                messages=[
                    Message.text(
                        "user",
                        "Find and call return_hosted_tool_search_contract_nonce exactly once "
                        "with its required nonce, then answer briefly.",
                    )
                ],
                max_steps=2,
            )
        )
    ]

    assert events[-1].type is EventType.SESSION_COMPLETED
    assert tool.calls == [{"nonce": "cayu-hosted-tool-search-live"}]
    assert sum(event.type is EventType.TOOL_CALL_COMPLETED for event in events) == 1
