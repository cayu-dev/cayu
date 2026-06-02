from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx
import pytest

from cayu import AgentSpec, CayuApp, Message, RunRequest
from cayu.core.messages import TextPart, ToolCallPart
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.providers import (
    AnthropicAPIError,
    AnthropicProtocolError,
    AnthropicProvider,
    HttpxAnthropicTransport,
    ModelRequest,
    ModelStreamEventType,
    anthropic_response_events,
    build_anthropic_payload,
)


class RecordingTransport:
    def __init__(self, responses: list[Mapping[str, Any]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def create_message(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "payload": dict(payload),
                "timeout_s": timeout_s,
            }
        )
        if not self.responses:
            raise AssertionError("No fake Anthropic response queued.")
        return self.responses.pop(0)


class BlankFailingTransport:
    async def create_message(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        raise RuntimeError()


class EchoTool(Tool):
    spec = ToolSpec(
        name="echo",
        description="Echo text.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        return ToolResult(
            content=args["text"],
            structured={"agent": ctx.agent_name, "echoed": args["text"]},
        )


def test_build_anthropic_payload_translates_cayu_messages() -> None:
    request = ModelRequest(
        model="claude-test",
        messages=[
            Message.text("system", "You are a careful assistant."),
            Message.text("user", "Read a file."),
            Message(
                role="assistant",
                content=[
                    TextPart(text="I will inspect it."),
                    ToolCallPart(
                        tool_call_id="toolu_1",
                        tool_name="read_file",
                        arguments={"path": "README.md"},
                    ),
                ],
            ),
            Message.tool_result(
                tool_call_id="toolu_1",
                tool_name="read_file",
                content="README content",
                structured={"ignored_by_provider": True},
            ),
        ],
        tools=[
            {
                "name": "read_file",
                "description": "Read a file.",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            }
        ],
        options={"anthropic": {"temperature": 0.2}},
    )

    payload = build_anthropic_payload(request, default_max_tokens=1234)

    assert payload == {
        "model": "claude-test",
        "max_tokens": 1234,
        "system": "You are a careful assistant.",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Read a file."}],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I will inspect it."},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "read_file",
                        "input": {"path": "README.md"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "README content",
                    }
                ],
            },
        ],
        "tools": [
            {
                "name": "read_file",
                "description": "Read a file.",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            }
        ],
        "temperature": 0.2,
    }


@pytest.mark.anyio
async def test_anthropic_provider_emits_text_and_completed_events() -> None:
    transport = RecordingTransport(
        [
            {
                "id": "msg_1",
                "model": "claude-test",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "hello"}],
                "usage": {"input_tokens": 10, "output_tokens": 2},
            }
        ]
    )
    provider = AnthropicProvider(api_key="test-key", transport=transport)
    request = ModelRequest(
        model="claude-test",
        messages=[Message.text("user", "Say hello.")],
    )

    events = [event async for event in provider.stream(request)]

    assert [event.type for event in events] == [
        ModelStreamEventType.TEXT_DELTA,
        ModelStreamEventType.COMPLETED,
    ]
    assert events[0].delta == "hello"
    assert events[1].payload["stop_reason"] == "end_turn"
    assert transport.calls[0]["url"] == "https://api.anthropic.com/v1/messages"
    assert transport.calls[0]["headers"]["x-api-key"] == "test-key"


@pytest.mark.anyio
async def test_anthropic_provider_emits_tool_call_events() -> None:
    transport = RecordingTransport(
        [
            {
                "id": "msg_1",
                "model": "claude-test",
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "echo",
                        "input": {"text": "hello"},
                    }
                ],
            }
        ]
    )
    provider = AnthropicProvider(api_key="test-key", transport=transport)
    request = ModelRequest(
        model="claude-test",
        messages=[Message.text("user", "Use a tool.")],
    )

    events = [event async for event in provider.stream(request)]

    assert [event.type for event in events] == [
        ModelStreamEventType.TOOL_CALL,
        ModelStreamEventType.COMPLETED,
    ]
    assert events[0].payload == {
        "id": "toolu_1",
        "name": "echo",
        "arguments": {"text": "hello"},
    }
    assert events[1].payload["stop_reason"] == "tool_use"


@pytest.mark.anyio
async def test_anthropic_provider_round_trips_runtime_tool_results() -> None:
    transport = RecordingTransport(
        [
            {
                "id": "msg_1",
                "model": "claude-test",
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "echo",
                        "input": {"text": "hello from claude"},
                    }
                ],
            },
            {
                "id": "msg_2",
                "model": "claude-test",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "final answer"}],
            },
        ]
    )
    provider = AnthropicProvider(api_key="test-key", transport=transport)
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(
            name="assistant",
            model="claude-test",
            system_prompt="Use tools when needed.",
        ),
        tools=[EchoTool()],
    )

    events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "Echo this.")],
            )
        )
    ]

    assert events[-1].type == "session.completed"
    assert len(transport.calls) == 2
    assert transport.calls[0]["payload"]["messages"] == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Echo this."}],
        }
    ]
    assert transport.calls[1]["payload"]["messages"] == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Echo this."}],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "echo",
                    "input": {"text": "hello from claude"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": "hello from claude",
                }
            ],
        },
    ]


def test_anthropic_response_events_rejects_malformed_tool_use() -> None:
    with pytest.raises(AnthropicProtocolError, match="input must be an object"):
        anthropic_response_events(
            {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "echo",
                        "input": [],
                    }
                ]
            }
        )


def test_anthropic_options_must_not_override_reserved_payload_fields() -> None:
    request = ModelRequest(
        model="claude-test",
        messages=[Message.text("user", "hello")],
        options={"anthropic": {"messages": []}},
    )

    with pytest.raises(ValueError, match="reserved"):
        build_anthropic_payload(request)


def test_anthropic_provider_rejects_invalid_max_tokens_option() -> None:
    request = ModelRequest(
        model="claude-test",
        messages=[Message.text("user", "hello")],
        options={"anthropic": {"max_tokens": 0}},
    )

    with pytest.raises(ValueError, match="max_tokens"):
        build_anthropic_payload(request)


def test_anthropic_provider_rejects_empty_message_payload() -> None:
    request = ModelRequest(
        model="claude-test",
        messages=[Message.text("system", "Only system.")],
    )

    with pytest.raises(ValueError, match="non-system message"):
        build_anthropic_payload(request)


def test_anthropic_provider_rejects_invalid_tool_names() -> None:
    request = ModelRequest(
        model="claude-test",
        messages=[Message.text("user", "hello")],
        tools=[
            {
                "name": "invalid tool name",
                "description": "Bad name.",
                "input_schema": {"type": "object"},
            }
        ],
    )

    with pytest.raises(ValueError, match="tool names"):
        build_anthropic_payload(request)


def test_anthropic_provider_rejects_protected_extra_headers() -> None:
    with pytest.raises(ValueError, match="extra_headers cannot override"):
        AnthropicProvider(
            api_key="test-key",
            extra_headers={"x-api-key": "other-key"},
        )


def test_anthropic_provider_requires_https_base_url() -> None:
    with pytest.raises(ValueError, match="https"):
        AnthropicProvider(
            api_key="test-key",
            base_url="http://api.anthropic.com",
        )


@pytest.mark.anyio
async def test_httpx_transport_requires_https_url() -> None:
    with pytest.raises(ValueError, match="https"):
        await HttpxAnthropicTransport().create_message(
            url="http://api.anthropic.com/v1/messages",
            headers={},
            payload={},
            timeout_s=1,
        )


@pytest.mark.anyio
async def test_httpx_transport_includes_url_in_network_errors(monkeypatch) -> None:
    class FailingClient:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> FailingClient:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def post(
            self,
            url: str,
            *,
            headers: dict[str, str],
            json: dict[str, Any],
        ) -> httpx.Response:
            request = httpx.Request("POST", url)
            raise httpx.ConnectError(
                "[Errno 8] nodename nor servname provided, or not known",
                request=request,
            )

    monkeypatch.setattr(
        "cayu.providers.anthropic.httpx.AsyncClient",
        FailingClient,
    )

    with pytest.raises(AnthropicAPIError, match="https://api.anthropic.com/v1/messages"):
        await HttpxAnthropicTransport().create_message(
            url="https://api.anthropic.com/v1/messages",
            headers={},
            payload={},
            timeout_s=1,
        )


@pytest.mark.anyio
async def test_httpx_transport_sanitizes_anthropic_error_body(monkeypatch) -> None:
    class FailingClient:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> FailingClient:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def post(
            self,
            url: str,
            *,
            headers: dict[str, str],
            json: dict[str, Any],
        ) -> httpx.Response:
            request = httpx.Request("POST", url)
            response = httpx.Response(
                400,
                request=request,
                headers={"content-type": "application/json"},
                json={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "bad request",
                        "debug": "not persisted",
                    },
                    "request_id": "req_123",
                    "extra": "not persisted",
                },
            )
            raise httpx.HTTPStatusError(
                "bad request",
                request=request,
                response=response,
            )

    monkeypatch.setattr(
        "cayu.providers.anthropic.httpx.AsyncClient",
        FailingClient,
    )

    with pytest.raises(AnthropicAPIError) as exc_info:
        await HttpxAnthropicTransport().create_message(
            url="https://api.anthropic.com/v1/messages",
            headers={},
            payload={},
            timeout_s=1,
        )

    message = str(exc_info.value)
    assert (
        message == "Anthropic API request failed with HTTP 400: "
        '{"message":"bad request","request_id":"req_123",'
        '"type":"invalid_request_error"}'
    )
    assert "debug" not in message
    assert "not persisted" not in message


@pytest.mark.anyio
async def test_anthropic_provider_emits_nonblank_error_for_blank_exception() -> None:
    provider = AnthropicProvider(api_key="test-key", transport=BlankFailingTransport())
    request = ModelRequest(
        model="claude-test",
        messages=[Message.text("user", "hello")],
    )

    events = [event async for event in provider.stream(request)]

    assert [event.type for event in events] == [ModelStreamEventType.ERROR]
    assert events[0].payload == {"error": "RuntimeError: Anthropic provider failed"}
