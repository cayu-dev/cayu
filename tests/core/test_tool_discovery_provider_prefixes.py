from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any

import pytest

from cayu import Message, ModelRequest
from cayu.providers.anthropic import build_anthropic_payload
from cayu.providers.bedrock import build_bedrock_converse_payload
from cayu.providers.chat_completions import build_chat_completions_payload
from cayu.providers.openai import build_openai_payload
from cayu.providers.vertex import VertexProvider
from cayu.runtime.tool_discovery import search_tools_spec
from cayu.runtime.tool_gateway import call_tool_spec

_HIDDEN_DISCOVERED_TOOL_NAME = "hidden_discovered_tool"
_HIDDEN_DISCOVERED_ARGUMENT_NAME = "secret_argument"
_DIRECT_TOOL = {
    "name": "direct_tool",
    "description": "One deliberately direct capability.",
    "input_schema": {"type": "object", "additionalProperties": False},
}


class _FakeCredentials:
    token = "vertex-token"
    valid = True

    def refresh(self, _request: object) -> None:
        self.token = "vertex-refreshed-token"


class _VertexTransport:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    async def create_message(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        del url, headers, timeout_s
        self.payloads.append(dict(payload))
        return {
            "content": [{"type": "text", "text": "done"}],
            "stop_reason": "end_turn",
            "usage": {},
        }

    async def count_message_tokens(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        del url, headers, payload, timeout_s
        raise AssertionError("Provider projection must not invoke token counting.")

    async def stream_message_events(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
        stream_idle_timeout_s: float,
    ) -> AsyncIterator[Mapping[str, Any]]:
        del url, headers, timeout_s, stream_idle_timeout_s
        self.payloads.append(dict(payload))
        yield {
            "type": "message_start",
            "message": {"id": "vertex-message", "usage": {"input_tokens": 1}},
        }
        yield {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 1},
        }
        yield {"type": "message_stop"}


def _request(*, include_direct_tool: bool) -> ModelRequest:
    return ModelRequest(
        model="provider-model",
        messages=[Message.text("user", "Use the appropriate capability.")],
        tools=[
            search_tools_spec(),
            call_tool_spec(),
            *([_DIRECT_TOOL] if include_direct_tool else []),
        ],
    )


async def _openai_tools(request: ModelRequest) -> list[dict[str, Any]]:
    return build_openai_payload(request)["tools"]


async def _anthropic_tools(request: ModelRequest) -> list[dict[str, Any]]:
    return build_anthropic_payload(request)["tools"]


async def _chat_completions_tools(request: ModelRequest) -> list[dict[str, Any]]:
    return build_chat_completions_payload(request)["tools"]


async def _bedrock_tools(request: ModelRequest) -> list[dict[str, Any]]:
    return build_bedrock_converse_payload(request)["toolConfig"]["tools"]


async def _vertex_tools(request: ModelRequest) -> list[dict[str, Any]]:
    transport = _VertexTransport()
    provider = VertexProvider(
        project_id="test-project",
        region="us-east5",
        credentials=_FakeCredentials(),
        transport=transport,
    )
    _ = [event async for event in provider.stream(request)]
    [payload] = transport.payloads
    return payload["tools"]


_ToolProjector = Callable[[ModelRequest], Awaitable[list[dict[str, Any]]]]


def _projected_tool_name(adapter: str, tool: dict[str, Any]) -> str:
    if adapter in {"openai", "chat_completions"}:
        return tool["name"] if adapter == "openai" else tool["function"]["name"]
    if adapter == "bedrock":
        return tool["toolSpec"]["name"]
    return tool["name"]


@pytest.mark.parametrize(
    ("adapter", "project"),
    [
        ("openai", _openai_tools),
        ("anthropic", _anthropic_tools),
        ("chat_completions", _chat_completions_tools),
        ("bedrock", _bedrock_tools),
        ("vertex", _vertex_tools),
    ],
)
def test_discovery_core_is_a_stable_provider_tool_prefix(
    adapter: str,
    project: _ToolProjector,
) -> None:
    core_only, with_direct = asyncio.run(
        _project_both(project),
    )

    assert [_projected_tool_name(adapter, tool) for tool in core_only] == [
        "search_tools",
        "call_tool",
    ]
    assert [_projected_tool_name(adapter, tool) for tool in with_direct] == [
        "search_tools",
        "call_tool",
        "direct_tool",
    ]
    assert with_direct[:2] == core_only
    projected_json = json.dumps(with_direct, sort_keys=True)
    assert _HIDDEN_DISCOVERED_TOOL_NAME not in projected_json
    assert _HIDDEN_DISCOVERED_ARGUMENT_NAME not in projected_json


async def _project_both(
    project: _ToolProjector,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return (
        await project(_request(include_direct_tool=False)),
        await project(_request(include_direct_tool=True)),
    )
