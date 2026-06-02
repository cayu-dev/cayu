from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx
import pytest

from cayu import AgentSpec, CayuApp, Message, RunRequest
from cayu.core.messages import ProviderStatePart, TextPart, ToolCallPart
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.providers import (
    HttpxOpenAITransport,
    ModelRequest,
    ModelStreamEventType,
    OpenAIAPIError,
    OpenAIProtocolError,
    OpenAIProvider,
    build_openai_payload,
    openai_response_events,
)


class RecordingTransport:
    def __init__(self, responses: list[Mapping[str, Any]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def create_response(
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
            raise AssertionError("No fake OpenAI response queued.")
        return self.responses.pop(0)


class BlankFailingTransport:
    async def create_response(
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


def test_build_openai_payload_translates_cayu_messages() -> None:
    request = ModelRequest(
        model="gpt-test",
        messages=[
            Message.text("system", "You are a careful assistant."),
            Message.text("user", "Read a file."),
            Message(
                role="assistant",
                content=[
                    TextPart(text="I will inspect it."),
                    ToolCallPart(
                        tool_call_id="call_1",
                        tool_name="read_file",
                        arguments={"path": "README.md"},
                    ),
                ],
            ),
            Message.tool_result(
                tool_call_id="call_1",
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
        options={"openai": {"temperature": 0.2}},
    )

    payload = build_openai_payload(request)

    assert payload == {
        "model": "gpt-test",
        "instructions": "You are a careful assistant.",
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Read a file."}],
            },
            {
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "I will inspect it."},
                ],
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "read_file",
                "arguments": '{"path":"README.md"}',
                "status": "completed",
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "README content",
            },
        ],
        "tools": [
            {
                "type": "function",
                "name": "read_file",
                "description": "Read a file.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
                "strict": False,
            }
        ],
        "store": False,
        "temperature": 0.2,
    }


@pytest.mark.anyio
async def test_openai_provider_emits_text_and_completed_events() -> None:
    transport = RecordingTransport(
        [
            {
                "id": "resp_1",
                "model": "gpt-test",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_1",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "hello",
                                "annotations": [],
                            }
                        ],
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 2},
            }
        ]
    )
    provider = OpenAIProvider(api_key="test-key", transport=transport)
    request = ModelRequest(
        model="gpt-test",
        messages=[Message.text("user", "Say hello.")],
    )

    events = [event async for event in provider.stream(request)]

    assert [event.type for event in events] == [
        ModelStreamEventType.TEXT_DELTA,
        ModelStreamEventType.COMPLETED,
    ]
    assert events[0].delta == "hello"
    assert events[1].payload["status"] == "completed"
    assert transport.calls[0]["url"] == "https://api.openai.com/v1/responses"
    assert transport.calls[0]["headers"]["authorization"] == "Bearer test-key"
    assert transport.calls[0]["payload"]["store"] is False


@pytest.mark.anyio
async def test_openai_provider_emits_tool_call_events() -> None:
    transport = RecordingTransport(
        [
            {
                "id": "resp_1",
                "model": "gpt-test",
                "status": "completed",
                "output": [
                    {
                        "type": "reasoning",
                        "id": "rs_1",
                        "summary": [],
                        "phase": "tool_use",
                    },
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "echo",
                        "arguments": '{"text":"hello"}',
                        "status": "completed",
                    },
                ],
            }
        ]
    )
    provider = OpenAIProvider(api_key="test-key", transport=transport)
    request = ModelRequest(
        model="gpt-test",
        messages=[Message.text("user", "Use a tool.")],
    )

    events = [event async for event in provider.stream(request)]

    assert [event.type for event in events] == [
        ModelStreamEventType.TOOL_CALL,
        ModelStreamEventType.COMPLETED,
    ]
    assert events[0].payload == {
        "id": "call_1",
        "name": "echo",
        "arguments": {"text": "hello"},
    }
    assert events[1].payload["status"] == "completed"


@pytest.mark.anyio
async def test_openai_provider_round_trips_runtime_tool_results() -> None:
    transport = RecordingTransport(
        [
            {
                "id": "resp_1",
                "model": "gpt-test",
                "status": "completed",
                "output": [
                    {
                        "type": "reasoning",
                        "id": "rs_1",
                        "summary": [],
                        "phase": "tool_use",
                    },
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "echo",
                        "arguments": '{"text":"hello from openai"}',
                        "status": "completed",
                    },
                ],
            },
            {
                "id": "resp_2",
                "model": "gpt-test",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_2",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "final answer",
                                "annotations": [],
                            }
                        ],
                    }
                ],
            },
        ]
    )
    provider = OpenAIProvider(api_key="test-key", transport=transport)
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(
            name="assistant",
            model="gpt-test",
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
    model_completed_events = [event for event in events if event.type == "model.completed"]
    assert "provider_state" not in model_completed_events[0].payload
    assert len(transport.calls) == 2
    assert transport.calls[0]["payload"]["input"] == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "Echo this."}],
        }
    ]
    assert transport.calls[1]["payload"]["input"] == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "Echo this."}],
        },
        {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [],
            "phase": "tool_use",
        },
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "echo",
            "arguments": '{"text":"hello from openai"}',
            "status": "completed",
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "hello from openai",
        },
    ]


def test_openai_response_events_rejects_malformed_function_call() -> None:
    with pytest.raises(OpenAIProtocolError, match="arguments were not valid JSON"):
        openai_response_events(
            {
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "echo",
                        "arguments": "{",
                    }
                ]
            }
        )


def test_openai_response_events_sanitizes_response_error() -> None:
    with pytest.raises(OpenAIProtocolError) as exc_info:
        openai_response_events(
            {
                "error": {
                    "type": "invalid_request_error",
                    "code": "bad_request",
                    "message": "bad request",
                    "debug": "not persisted",
                },
                "output": [],
            }
        )

    message = str(exc_info.value)
    assert (
        message == 'OpenAI response error: {"code":"bad_request",'
        '"message":"bad request","type":"invalid_request_error"}'
    )
    assert "debug" not in message
    assert "not persisted" not in message


def test_openai_response_events_rejects_unsupported_output_item() -> None:
    with pytest.raises(OpenAIProtocolError, match="Unsupported OpenAI output item"):
        openai_response_events({"output": [{"type": "web_search_call"}]})


def test_openai_response_events_ignores_reasoning_items() -> None:
    events = openai_response_events(
        {
            "id": "resp_1",
            "model": "gpt-test",
            "status": "completed",
            "output": [
                {"type": "reasoning", "id": "rs_1", "summary": []},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "done"}],
                },
            ],
        }
    )

    assert [event.type for event in events] == [
        ModelStreamEventType.TEXT_DELTA,
        ModelStreamEventType.COMPLETED,
    ]
    assert events[0].delta == "done"


@pytest.mark.parametrize("reserved_option", ["input", "store", "previous_response_id"])
def test_openai_options_must_not_override_reserved_payload_fields(
    reserved_option: str,
) -> None:
    request = ModelRequest(
        model="gpt-test",
        messages=[Message.text("user", "hello")],
        options={"openai": {reserved_option: "bad"}},
    )

    with pytest.raises(ValueError, match="reserved"):
        build_openai_payload(request)


def test_openai_payload_replays_provider_state_items() -> None:
    request = ModelRequest(
        model="gpt-test",
        messages=[
            Message.text("user", "hello"),
            Message(
                role="assistant",
                content=[
                    ProviderStatePart(
                        provider="openai",
                        state={
                            "type": "reasoning",
                            "id": "rs_1",
                            "summary": [],
                            "phase": "tool_use",
                        },
                    ),
                    ProviderStatePart(
                        provider="openai",
                        state={
                            "type": "function_call",
                            "id": "fc_1",
                            "call_id": "call_1",
                            "name": "echo",
                            "arguments": '{"text":"hello"}',
                            "status": "completed",
                        },
                    ),
                    ToolCallPart(
                        tool_call_id="call_1",
                        tool_name="echo",
                        arguments={"text": "hello"},
                    ),
                ],
            ),
            Message.tool_result(
                tool_call_id="call_1",
                tool_name="echo",
                content="hello",
            ),
        ],
    )

    payload = build_openai_payload(request)

    assert payload["input"] == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        },
        {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [],
            "phase": "tool_use",
        },
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "echo",
            "arguments": '{"text":"hello"}',
            "status": "completed",
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "hello",
        },
    ]


def test_openai_payload_ignores_other_provider_state_items() -> None:
    request = ModelRequest(
        model="gpt-test",
        messages=[
            Message.text("user", "hello"),
            Message(
                role="assistant",
                content=[
                    ProviderStatePart(
                        provider="other-provider",
                        state={"type": "opaque", "id": "state_1"},
                    ),
                    TextPart(text="assistant text"),
                ],
            ),
        ],
    )

    payload = build_openai_payload(request)

    assert payload["input"] == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        },
        {
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "assistant text"}],
        },
    ]


def test_openai_provider_rejects_invalid_tool_names() -> None:
    request = ModelRequest(
        model="gpt-test",
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
        build_openai_payload(request)


def test_openai_provider_rejects_protected_extra_headers() -> None:
    with pytest.raises(ValueError, match="extra_headers cannot override"):
        OpenAIProvider(
            api_key="test-key",
            extra_headers={"authorization": "other-key"},
        )


def test_openai_provider_requires_https_base_url() -> None:
    with pytest.raises(ValueError, match="https"):
        OpenAIProvider(
            api_key="test-key",
            base_url="http://api.openai.com",
        )


@pytest.mark.anyio
async def test_httpx_openai_transport_requires_https_url() -> None:
    with pytest.raises(ValueError, match="https"):
        await HttpxOpenAITransport().create_response(
            url="http://api.openai.com/v1/responses",
            headers={},
            payload={},
            timeout_s=1,
        )


@pytest.mark.anyio
async def test_httpx_openai_transport_includes_url_in_network_errors(monkeypatch) -> None:
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
        "cayu.providers.openai.httpx.AsyncClient",
        FailingClient,
    )

    with pytest.raises(OpenAIAPIError, match="https://api.openai.com/v1/responses"):
        await HttpxOpenAITransport().create_response(
            url="https://api.openai.com/v1/responses",
            headers={},
            payload={},
            timeout_s=1,
        )


@pytest.mark.anyio
async def test_httpx_openai_transport_sanitizes_error_body(monkeypatch) -> None:
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
                    "error": {
                        "type": "invalid_request_error",
                        "code": "bad_request",
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
        "cayu.providers.openai.httpx.AsyncClient",
        FailingClient,
    )

    with pytest.raises(OpenAIAPIError) as exc_info:
        await HttpxOpenAITransport().create_response(
            url="https://api.openai.com/v1/responses",
            headers={},
            payload={},
            timeout_s=1,
        )

    message = str(exc_info.value)
    assert (
        message == "OpenAI API request failed with HTTP 400: "
        '{"code":"bad_request","message":"bad request",'
        '"request_id":"req_123","type":"invalid_request_error"}'
    )
    assert "debug" not in message
    assert "not persisted" not in message


@pytest.mark.anyio
async def test_openai_provider_emits_nonblank_error_for_blank_exception() -> None:
    provider = OpenAIProvider(api_key="test-key", transport=BlankFailingTransport())
    request = ModelRequest(
        model="gpt-test",
        messages=[Message.text("user", "hello")],
    )

    events = [event async for event in provider.stream(request)]

    assert [event.type for event in events] == [ModelStreamEventType.ERROR]
    assert events[0].payload == {"error": "RuntimeError: OpenAI provider failed"}
