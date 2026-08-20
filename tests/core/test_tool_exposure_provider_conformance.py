from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from cayu import Message
from cayu.providers import (
    ModelRequest,
    VertexProvider,
    build_anthropic_payload,
    build_bedrock_converse_payload,
    build_chat_completions_payload,
    build_openai_payload,
)
from cayu.providers.anthropic import build_anthropic_token_count_payload
from cayu.providers.openai import build_openai_token_count_payload


def _projected_request() -> ModelRequest:
    """Represent a gamma/alpha exposure selected from alpha/beta/gamma."""

    return ModelRequest(
        model="test-model",
        messages=[Message.text("user", "Use an available tool.")],
        tools=[
            {
                "name": "gamma",
                "description": "Run gamma.",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "alpha",
                "description": "Run alpha.",
                "input_schema": {"type": "object", "properties": {}},
            },
        ],
    )


def test_openai_serializers_preserve_exposure_order_and_omit_hidden_tools() -> None:
    payload = build_openai_payload(_projected_request())
    count_payload = build_openai_token_count_payload(_projected_request())

    assert [tool["name"] for tool in payload["tools"]] == ["gamma", "alpha"]
    assert count_payload["tools"] == payload["tools"]
    assert "beta" not in {tool["name"] for tool in payload["tools"]}


def test_anthropic_serializers_preserve_exposure_order_and_omit_hidden_tools() -> None:
    payload = build_anthropic_payload(_projected_request())
    count_payload = build_anthropic_token_count_payload(_projected_request())

    assert [tool["name"] for tool in payload["tools"]] == ["gamma", "alpha"]
    assert count_payload["tools"] == payload["tools"]
    assert "beta" not in {tool["name"] for tool in payload["tools"]}


def test_chat_completions_preserves_exposure_order_and_omits_hidden_tools() -> None:
    payload = build_chat_completions_payload(_projected_request())
    names = [tool["function"]["name"] for tool in payload["tools"]]

    assert names == ["gamma", "alpha"]
    assert "beta" not in names


def test_bedrock_preserves_exposure_order_and_omits_hidden_tools() -> None:
    payload = build_bedrock_converse_payload(_projected_request())
    names = [tool["toolSpec"]["name"] for tool in payload["toolConfig"]["tools"]]

    assert names == ["gamma", "alpha"]
    assert "beta" not in names


class _VertexCredentials:
    valid = True
    token = "test-token"


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
            "id": "message-1",
            "model": "test-model",
            "role": "assistant",
            "content": [{"type": "text", "text": "done"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }


@pytest.mark.anyio
async def test_vertex_preserves_exposure_order_and_omits_hidden_tools() -> None:
    transport = _VertexTransport()
    provider = VertexProvider(
        project_id="test-project",
        credentials=_VertexCredentials(),
        transport=transport,
    )

    _ = [event async for event in provider.stream(_projected_request())]

    [payload] = transport.payloads
    names = [tool["name"] for tool in payload["tools"]]
    assert names == ["gamma", "alpha"]
    assert "beta" not in names
