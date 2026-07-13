from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx
import pytest

from cayu import (
    RESOLVED_FILE_ATTACHMENTS_OPTION,
    AgentSpec,
    AllowlistProxy,
    CacheBreakpoint,
    CachePolicy,
    CayuApp,
    FileAttachmentKind,
    Message,
    RunRequest,
    file_attachment,
)
from cayu.core.messages import FilePart, TextPart, ToolCallPart
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.providers import (
    AnthropicAPIError,
    AnthropicContextOverflowError,
    AnthropicProtocolError,
    AnthropicProvider,
    HttpxAnthropicTransport,
    InputTokenCountConfidence,
    InputTokenCountMethod,
    ModelContextOverflowError,
    ModelRequest,
    ModelStreamEventType,
    anthropic_response_events,
    anthropic_stream_events,
    build_anthropic_payload,
)
from cayu.providers.cache import resolve_cache_policy
from cayu.vaults import SecretRef, StaticVault


class RecordingTransport:
    def __init__(
        self,
        responses: list[Mapping[str, Any]],
        count_responses: list[Mapping[str, Any]] | None = None,
    ) -> None:
        self.responses = list(responses)
        self.count_responses = list(count_responses or [])
        self.calls: list[dict[str, Any]] = []
        self.count_calls: list[dict[str, Any]] = []

    async def count_message_tokens(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        self.count_calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "payload": dict(payload),
                "timeout_s": timeout_s,
            }
        )
        if not self.count_responses:
            raise AssertionError("No fake Anthropic count response queued.")
        return self.count_responses.pop(0)

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
    async def count_message_tokens(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        raise RuntimeError()

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


def test_build_anthropic_payload_passes_provider_cache_options() -> None:
    request = ModelRequest(
        model="claude-test",
        messages=[Message.text("user", "hello")],
        options={"anthropic": {"cache_control": {"type": "ephemeral", "ttl": "1h"}}},
    )

    payload = build_anthropic_payload(request)

    assert payload["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_build_anthropic_payload_translates_file_attachments() -> None:
    attachment = file_attachment(
        artifact_id="art_image",
        kind=FileAttachmentKind.IMAGE,
        filename="invoice.png",
        content_type="image/png",
        size_bytes=5,
    )
    request = ModelRequest(
        model="claude-test",
        messages=[
            Message.text("user", "Read the invoice."),
            Message.tool_call(
                tool_call_id="toolu_1",
                tool_name="read_file",
                arguments={"artifact_id": "art_image"},
            ),
            Message.tool_result(
                tool_call_id="toolu_1",
                tool_name="read_file",
                content="Attached image artifact art_image: invoice.png.",
                artifacts=[attachment],
            ),
        ],
        options={
            RESOLVED_FILE_ATTACHMENTS_OPTION: {
                "art_image": {
                    "artifact_id": "art_image",
                    "kind": "image",
                    "filename": "invoice.png",
                    "content_type": "image/png",
                    "data_base64": "aGVsbG8=",
                    "metadata": {},
                }
            }
        },
    )

    payload = build_anthropic_payload(request)

    tool_result = payload["messages"][2]["content"][0]
    assert tool_result == {
        "type": "tool_result",
        "tool_use_id": "toolu_1",
        "content": [
            {"type": "text", "text": "Attached image artifact art_image: invoice.png."},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "aGVsbG8=",
                },
            },
        ],
    }


def test_build_anthropic_payload_translates_pdf_attachments_without_filename() -> None:
    attachment = file_attachment(
        artifact_id="art_pdf",
        kind=FileAttachmentKind.DOCUMENT,
        filename="blank-page.pdf",
        content_type="application/pdf",
        size_bytes=5,
    )
    request = ModelRequest(
        model="claude-test",
        messages=[
            Message.text("user", "Read the PDF."),
            Message.tool_call(
                tool_call_id="toolu_1",
                tool_name="read_file",
                arguments={"artifact_id": "art_pdf"},
            ),
            Message.tool_result(
                tool_call_id="toolu_1",
                tool_name="read_file",
                content="Attached PDF artifact art_pdf: blank-page.pdf.",
                artifacts=[attachment],
            ),
        ],
        options={
            RESOLVED_FILE_ATTACHMENTS_OPTION: {
                "art_pdf": {
                    "artifact_id": "art_pdf",
                    "kind": "document",
                    "filename": "blank-page.pdf",
                    "content_type": "application/pdf",
                    "data_base64": "JVBERi0xLjQ=",
                    "metadata": {},
                }
            }
        },
    )

    payload = build_anthropic_payload(request)

    document_block = payload["messages"][2]["content"][0]["content"][1]
    assert document_block == {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": "JVBERi0xLjQ=",
        },
    }
    assert "filename" not in document_block


def test_build_anthropic_payload_translates_user_file_parts() -> None:
    image = file_attachment(
        artifact_id="art_image",
        kind=FileAttachmentKind.IMAGE,
        filename="invoice.png",
        content_type="image/png",
        size_bytes=5,
    )
    document = file_attachment(
        artifact_id="art_pdf",
        kind=FileAttachmentKind.DOCUMENT,
        filename="contract.pdf",
        content_type="application/pdf",
        size_bytes=9,
    )
    request = ModelRequest(
        model="claude-test",
        messages=[
            Message(
                role="user",
                content=[
                    TextPart(text="Read the invoice and the contract."),
                    FilePart(attachment=image),
                    FilePart(attachment=document),
                ],
            ),
        ],
        options={
            RESOLVED_FILE_ATTACHMENTS_OPTION: {
                "art_image": {
                    "artifact_id": "art_image",
                    "kind": "image",
                    "filename": "invoice.png",
                    "content_type": "image/png",
                    "data_base64": "aGVsbG8=",
                    "metadata": {},
                },
                "art_pdf": {
                    "artifact_id": "art_pdf",
                    "kind": "document",
                    "filename": "contract.pdf",
                    "content_type": "application/pdf",
                    "data_base64": "JVBERi0xLjQ=",
                    "metadata": {},
                },
            }
        },
    )

    payload = build_anthropic_payload(request)

    assert payload["messages"][0] == {
        "role": "user",
        "content": [
            {"type": "text", "text": "Read the invoice and the contract."},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "aGVsbG8=",
                },
            },
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": "JVBERi0xLjQ=",
                },
            },
        ],
    }


def test_build_anthropic_payload_requires_resolved_user_file_parts() -> None:
    image = file_attachment(
        artifact_id="art_missing",
        kind=FileAttachmentKind.IMAGE,
        filename="invoice.png",
        content_type="image/png",
        size_bytes=5,
    )
    request = ModelRequest(
        model="claude-test",
        messages=[
            Message(
                role="user",
                content=[TextPart(text="Read it."), FilePart(attachment=image)],
            ),
        ],
    )

    with pytest.raises(AnthropicProtocolError, match="Missing resolved file attachment"):
        build_anthropic_payload(request)


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
async def test_anthropic_provider_counts_input_tokens_with_official_endpoint() -> None:
    transport = RecordingTransport([], count_responses=[{"input_tokens": 37}])
    provider = AnthropicProvider(api_key="test-key", transport=transport)
    request = ModelRequest(
        model="claude-test",
        messages=[Message.text("user", "Count this.")],
        options={"anthropic": {"max_tokens": 100}},
    )

    result = await provider.count_input_tokens(request)

    assert result is not None
    assert result.input_tokens == 37
    assert result.method == InputTokenCountMethod.OFFICIAL
    assert result.confidence == InputTokenCountConfidence.HIGH
    assert result.metadata == {
        "endpoint": "messages/count_tokens",
        "provider_billing_status": "documented_free",
        "provider_rate_limit": "separate_rpm_limit",
    }
    assert transport.count_calls[0]["url"] == ("https://api.anthropic.com/v1/messages/count_tokens")
    assert transport.count_calls[0]["headers"]["x-api-key"] == "test-key"
    assert transport.count_calls[0]["payload"] == {
        "model": "claude-test",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Count this."}]}],
    }


@pytest.mark.anyio
async def test_anthropic_provider_rejects_invalid_token_count_response() -> None:
    transport = RecordingTransport([], count_responses=[{"input_tokens": "37"}])
    provider = AnthropicProvider(api_key="test-key", transport=transport)

    with pytest.raises(AnthropicProtocolError, match="input_tokens"):
        await provider.count_input_tokens(
            ModelRequest(model="claude-test", messages=[Message.text("user", "Count this.")])
        )


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


def test_anthropic_response_events_rejects_non_object_usage() -> None:
    with pytest.raises(AnthropicProtocolError, match="usage must be an object"):
        anthropic_response_events({"content": [], "usage": []})


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
            timeout: Any = None,
        ) -> httpx.Response:
            request = httpx.Request("POST", url)
            raise httpx.ConnectError(
                "[Errno 8] nodename nor servname provided, or not known",
                request=request,
            )

    monkeypatch.setattr(
        "cayu.providers._http.httpx.AsyncClient",
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
            timeout: Any = None,
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
        "cayu.providers._http.httpx.AsyncClient",
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
async def test_httpx_anthropic_transport_populates_typed_retry_fields(monkeypatch) -> None:
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
            timeout: Any = None,
        ) -> httpx.Response:
            request = httpx.Request("POST", url)
            response = httpx.Response(
                429,
                request=request,
                headers={
                    "content-type": "application/json",
                    "retry-after": "12",
                    "request-id": "req_rate_limited",
                },
                json={
                    "type": "error",
                    "error": {
                        "type": "rate_limit_error",
                        "message": "rate limit exceeded",
                    },
                },
            )
            raise httpx.HTTPStatusError(
                "rate limited",
                request=request,
                response=response,
            )

    monkeypatch.setattr(
        "cayu.providers._http.httpx.AsyncClient",
        FailingClient,
    )

    with pytest.raises(AnthropicAPIError) as exc_info:
        await HttpxAnthropicTransport().create_message(
            url="https://api.anthropic.com/v1/messages",
            headers={},
            payload={},
            timeout_s=1,
        )

    error = exc_info.value
    assert error.provider == "anthropic"
    assert error.status_code == 429
    assert error.error_type == "rate_limit_error"
    assert error.request_id == "req_rate_limited"
    assert error.retry_after_s == 12.0
    # Typed fields survive into the error-event payload used for retry
    # classification and observability.
    payload = error.error_payload_fields()
    assert payload["status_code"] == 429
    assert payload["retry_after_s"] == 12.0


@pytest.mark.anyio
async def test_httpx_anthropic_transport_classifies_request_too_large(monkeypatch) -> None:
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
            timeout: Any = None,
        ) -> httpx.Response:
            request = httpx.Request("POST", url)
            response = httpx.Response(
                413,
                request=request,
                headers={"content-type": "application/json"},
                json={
                    "type": "error",
                    "error": {
                        "type": "request_too_large",
                        "message": "Request exceeds the maximum allowed number of bytes.",
                    },
                    "request_id": "req_context",
                },
            )
            raise httpx.HTTPStatusError(
                "request too large",
                request=request,
                response=response,
            )

    monkeypatch.setattr(
        "cayu.providers._http.httpx.AsyncClient",
        FailingClient,
    )

    with pytest.raises(AnthropicContextOverflowError) as exc_info:
        await HttpxAnthropicTransport().create_message(
            url="https://api.anthropic.com/v1/messages",
            headers={},
            payload={},
            timeout_s=1,
        )

    assert exc_info.value.provider == "anthropic"
    assert isinstance(exc_info.value, ModelContextOverflowError)
    assert isinstance(exc_info.value, AnthropicAPIError)
    assert exc_info.value.status_code == 413
    assert exc_info.value.error_type == "request_too_large"
    assert exc_info.value.request_id == "req_context"
    assert isinstance(exc_info.value.__cause__, httpx.HTTPStatusError)


@pytest.mark.anyio
async def test_httpx_anthropic_transport_classifies_prompt_too_long(monkeypatch) -> None:
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
            timeout: Any = None,
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
                        "message": "Prompt is too long for this model's context window.",
                    },
                },
            )
            raise httpx.HTTPStatusError(
                "bad request",
                request=request,
                response=response,
            )

    monkeypatch.setattr(
        "cayu.providers._http.httpx.AsyncClient",
        FailingClient,
    )

    with pytest.raises(AnthropicContextOverflowError) as exc_info:
        await HttpxAnthropicTransport().create_message(
            url="https://api.anthropic.com/v1/messages",
            headers={},
            payload={},
            timeout_s=1,
        )

    assert exc_info.value.provider == "anthropic"
    assert isinstance(exc_info.value, ModelContextOverflowError)
    assert isinstance(exc_info.value, AnthropicAPIError)
    assert exc_info.value.status_code == 400
    assert exc_info.value.error_type == "invalid_request_error"
    assert isinstance(exc_info.value.__cause__, httpx.HTTPStatusError)


@pytest.mark.anyio
async def test_anthropic_provider_emits_nonblank_error_for_blank_exception() -> None:
    provider = AnthropicProvider(api_key="test-key", transport=BlankFailingTransport())
    request = ModelRequest(
        model="claude-test",
        messages=[Message.text("user", "hello")],
    )

    events = [event async for event in provider.stream(request)]

    assert [event.type for event in events] == [ModelStreamEventType.ERROR]
    assert events[0].payload == {
        "error": "RuntimeError: Anthropic provider failed",
        "error_type": "RuntimeError",
    }


class ErrorRaisingTransport:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def count_message_tokens(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        raise AssertionError("count_message_tokens should not be called.")

    async def create_message(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        raise self.error


@pytest.mark.anyio
async def test_anthropic_provider_stream_propagates_context_overflow() -> None:
    overflow = AnthropicContextOverflowError(
        "Anthropic model context overflow",
        status_code=413,
        error_type="request_too_large",
        request_id="req_overflow",
    )
    provider = AnthropicProvider(api_key="test-key", transport=ErrorRaisingTransport(overflow))
    request = ModelRequest(
        model="claude-test",
        messages=[Message.text("user", "hello")],
    )

    with pytest.raises(AnthropicContextOverflowError) as exc_info:
        [event async for event in provider.stream(request)]

    # Overflow must escape as a typed exception (not an ERROR event) so
    # runtime context-overflow recovery can shrink context and retry.
    assert exc_info.value is overflow
    assert isinstance(exc_info.value, ModelContextOverflowError)
    assert exc_info.value.retryable is False


@pytest.mark.anyio
async def test_anthropic_provider_stream_emits_typed_api_error_payload() -> None:
    provider = AnthropicProvider(
        api_key="test-key",
        transport=ErrorRaisingTransport(
            AnthropicAPIError(
                "Anthropic API request failed with HTTP 429: rate limited",
                status_code=429,
                error_type="rate_limit_error",
                request_id="req_429",
                retryable=True,
                retry_after_s=1.5,
            )
        ),
    )
    request = ModelRequest(
        model="claude-test",
        messages=[Message.text("user", "hello")],
    )

    events = [event async for event in provider.stream(request)]

    assert [event.type for event in events] == [ModelStreamEventType.ERROR]
    assert events[0].payload == {
        "error": "Anthropic API request failed with HTTP 429: rate limited",
        "error_type": "AnthropicAPIError",
        "provider": "anthropic",
        "status_code": 429,
        "provider_error_type": "rate_limit_error",
        "request_id": "req_429",
        "retryable": True,
        "retry_after_s": 1.5,
    }


_ALL_BREAKPOINTS = (
    CacheBreakpoint.SYSTEM_PROMPT,
    CacheBreakpoint.TOOL_DEFINITIONS,
    CacheBreakpoint.CONVERSATION_PREFIX,
)


def _cache_request() -> ModelRequest:
    return ModelRequest(
        model="claude-test",
        messages=[
            Message.text("system", "You are helpful."),
            Message.text("user", "hi"),
            Message.text("assistant", "hello"),
            Message.text("user", "thanks"),
        ],
        tools=[{"name": "lookup", "description": "d", "input_schema": {"type": "object"}}],
    )


def test_cache_policy_marks_system_and_tools_by_default() -> None:
    payload = build_anthropic_payload(_cache_request(), cache_policy=CachePolicy())
    assert payload["system"] == [
        {"type": "text", "text": "You are helpful.", "cache_control": {"type": "ephemeral"}}
    ]
    assert payload["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    # default policy has no CONVERSATION_PREFIX breakpoint
    assert not any(
        "cache_control" in block for message in payload["messages"] for block in message["content"]
    )


def test_cache_policy_none_is_byte_identical_to_no_caching() -> None:
    request = _cache_request()
    assert build_anthropic_payload(request) == build_anthropic_payload(request, cache_policy=None)
    payload = build_anthropic_payload(request)
    assert isinstance(payload["system"], str)
    assert "cache_control" not in payload["tools"][-1]


def test_cache_policy_conversation_prefix_marks_second_to_last_message() -> None:
    policy = CachePolicy(breakpoints=(CacheBreakpoint.CONVERSATION_PREFIX,))
    payload = build_anthropic_payload(_cache_request(), cache_policy=policy)
    # messages (system stripped): [user, assistant, user]; all_but_last marks index 1.
    marked = [i for i, m in enumerate(payload["messages"]) if "cache_control" in m["content"][-1]]
    assert marked == [1]
    # system/tools untouched (not in breakpoints)
    assert isinstance(payload["system"], str)
    assert "cache_control" not in payload["tools"][-1]


def test_cache_policy_conversation_prefix_all_but_last_n() -> None:
    policy = CachePolicy(
        breakpoints=(CacheBreakpoint.CONVERSATION_PREFIX,),
        conversation_prefix_strategy="all_but_last_n",
        conversation_prefix_n=2,
    )
    payload = build_anthropic_payload(_cache_request(), cache_policy=policy)
    # 3 messages, skip last 2 -> mark index 0
    marked = [i for i, m in enumerate(payload["messages"]) if "cache_control" in m["content"][-1]]
    assert marked == [0]


def test_cache_policy_conversation_prefix_too_few_messages_is_safe() -> None:
    request = ModelRequest(model="claude-test", messages=[Message.text("user", "hi")])
    policy = CachePolicy(breakpoints=(CacheBreakpoint.CONVERSATION_PREFIX,))
    payload = build_anthropic_payload(request, cache_policy=policy)
    assert not any(
        "cache_control" in block for message in payload["messages"] for block in message["content"]
    )


def test_cache_policy_empty_tools_and_system_do_not_crash() -> None:
    request = ModelRequest(model="claude-test", messages=[Message.text("user", "hi")])
    payload = build_anthropic_payload(
        request, cache_policy=CachePolicy(breakpoints=_ALL_BREAKPOINTS)
    )
    assert "system" not in payload
    assert "tools" not in payload


@pytest.mark.anyio
async def test_cache_policy_extended_ttl_sets_1h_marker_without_beta_header() -> None:
    transport = RecordingTransport(
        [
            {
                "id": "msg_1",
                "model": "claude-test",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 5, "output_tokens": 1},
            }
        ]
    )
    provider = AnthropicProvider(
        api_key="test-key",
        transport=transport,
        cache_policy=CachePolicy(ttl="extended"),
    )
    request = ModelRequest(
        model="claude-test",
        messages=[Message.text("system", "You are helpful."), Message.text("user", "hi")],
    )

    _ = [event async for event in provider.stream(request)]

    # The ttl:"1h" marker alone enables 1-hour caching; it is GA and needs no beta header.
    payload = transport.calls[0]["payload"]
    assert payload["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert "anthropic-beta" not in transport.calls[0]["headers"]


@pytest.mark.anyio
async def test_cache_policy_per_request_override_beats_provider_default() -> None:
    transport = RecordingTransport(
        [
            {
                "id": "msg_1",
                "model": "claude-test",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 5, "output_tokens": 1},
            }
        ]
    )
    # Provider default caches nothing relevant; the request overrides with system caching.
    provider = AnthropicProvider(
        api_key="test-key",
        transport=transport,
        cache_policy=CachePolicy(breakpoints=()),
    )
    request = ModelRequest(
        model="claude-test",
        messages=[Message.text("system", "You are helpful."), Message.text("user", "hi")],
        options={"cache_policy": {"breakpoints": ["system_prompt"]}},
    )

    _ = [event async for event in provider.stream(request)]

    system = transport.calls[0]["payload"]["system"]
    assert system[0]["cache_control"] == {"type": "ephemeral"}


def test_cache_policy_conversation_prefix_none_strategy_marks_nothing() -> None:
    policy = CachePolicy(
        breakpoints=(CacheBreakpoint.CONVERSATION_PREFIX,),
        conversation_prefix_strategy="none",
    )
    payload = build_anthropic_payload(_cache_request(), cache_policy=policy)
    assert not any(
        "cache_control" in block for message in payload["messages"] for block in message["content"]
    )


def test_resolve_cache_policy_merges_partial_override_onto_default() -> None:
    default = CachePolicy(breakpoints=(CacheBreakpoint.CONVERSATION_PREFIX,), ttl="extended")
    # Overriding only ttl keeps the provider's breakpoints (no silent reset).
    merged = resolve_cache_policy(default, {"cache_policy": {"ttl": "standard"}})
    assert merged is not None
    assert merged.breakpoints == (CacheBreakpoint.CONVERSATION_PREFIX,)
    assert merged.ttl == "standard"
    # No override returns the default unchanged; no default lets the override stand alone.
    assert resolve_cache_policy(default, {}) is default
    solo = resolve_cache_policy(None, {"cache_policy": {"breakpoints": ["system_prompt"]}})
    assert solo is not None
    assert solo.breakpoints == (CacheBreakpoint.SYSTEM_PROMPT,)


@pytest.mark.anyio
async def test_cache_policy_standard_ttl_has_no_beta_header() -> None:
    transport = RecordingTransport(
        [
            {
                "id": "msg_1",
                "model": "claude-test",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 5, "output_tokens": 1},
            }
        ]
    )
    provider = AnthropicProvider(
        api_key="test-key",
        transport=transport,
        cache_policy=CachePolicy(ttl="standard"),
    )
    request = ModelRequest(
        model="claude-test",
        messages=[Message.text("system", "You are helpful."), Message.text("user", "hi")],
    )

    _ = [event async for event in provider.stream(request)]

    assert "anthropic-beta" not in transport.calls[0]["headers"]
    assert transport.calls[0]["payload"]["system"][0]["cache_control"] == {"type": "ephemeral"}


class StreamingRecordingTransport:
    """Fake transport exposing the SSE streaming seam."""

    def __init__(self, event_batches: list[list[Mapping[str, Any]]]) -> None:
        self.event_batches = list(event_batches)
        self.calls: list[dict[str, Any]] = []

    async def count_message_tokens(self, *, url, headers, payload, timeout_s):
        raise AssertionError("count_message_tokens should not be called.")

    async def create_message(self, *, url, headers, payload, timeout_s):
        raise AssertionError("create_message must not be used when streaming is available.")

    async def stream_message_events(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
        stream_idle_timeout_s: float,
    ):
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "payload": dict(payload),
                "timeout_s": timeout_s,
                "stream_idle_timeout_s": stream_idle_timeout_s,
            }
        )
        if not self.event_batches:
            raise AssertionError("No fake Anthropic stream queued.")
        for event in self.event_batches.pop(0):
            yield event


async def _aiter_events(events: list[Mapping[str, Any]]):
    for event in events:
        yield event


_STREAM_EVENTS: list[Mapping[str, Any]] = [
    {
        "type": "message_start",
        "message": {"id": "msg_s1", "model": "claude-test", "usage": {"input_tokens": 11}},
    },
    {"type": "ping"},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hel"}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo"}},
    {"type": "content_block_stop", "index": 0},
    {
        "type": "content_block_start",
        "index": 1,
        "content_block": {"type": "thinking", "thinking": "", "signature": ""},
    },
    {
        "type": "content_block_delta",
        "index": 1,
        "delta": {"type": "thinking_delta", "thinking": "Let me "},
    },
    {
        "type": "content_block_delta",
        "index": 1,
        "delta": {"type": "thinking_delta", "thinking": "reason."},
    },
    {
        "type": "content_block_delta",
        "index": 1,
        "delta": {"type": "signature_delta", "signature": "sig-abc"},
    },
    {"type": "content_block_stop", "index": 1},
    {
        "type": "content_block_start",
        "index": 2,
        "content_block": {"type": "tool_use", "id": "toolu_9", "name": "echo", "input": {}},
    },
    {
        "type": "content_block_delta",
        "index": 2,
        "delta": {"type": "input_json_delta", "partial_json": '{"text": '},
    },
    {
        "type": "content_block_delta",
        "index": 2,
        "delta": {"type": "input_json_delta", "partial_json": '"hi"}'},
    },
    {"type": "content_block_stop", "index": 2},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "tool_use", "stop_sequence": None},
        "usage": {"output_tokens": 7},
    },
    {"type": "message_stop"},
]


@pytest.mark.anyio
async def test_anthropic_provider_streams_sse_events_incrementally() -> None:
    transport = StreamingRecordingTransport([_STREAM_EVENTS])
    provider = AnthropicProvider(api_key="test-key", transport=transport)
    request = ModelRequest(model="claude-test", messages=[Message.text("user", "Say hello.")])

    events = [event async for event in provider.stream(request)]

    assert [event.type for event in events] == [
        ModelStreamEventType.TEXT_DELTA,
        ModelStreamEventType.TEXT_DELTA,
        ModelStreamEventType.THINKING,
        ModelStreamEventType.TOOL_CALL,
        ModelStreamEventType.COMPLETED,
    ]
    assert [events[0].delta, events[1].delta] == ["hel", "lo"]
    # The thinking block is emitted whole so its signature round-trips over the
    # complete text (a partial block would be rejected on the next turn).
    assert events[2].delta == "Let me reason."
    assert events[2].payload["provider_state"] == {"type": "thinking", "signature": "sig-abc"}
    assert events[3].payload == {"id": "toolu_9", "name": "echo", "arguments": {"text": "hi"}}
    completed = events[4].payload
    assert completed["id"] == "msg_s1"
    assert completed["model"] == "claude-test"
    assert completed["stop_reason"] == "tool_use"
    # Usage merges message_start input counts with message_delta output counts.
    assert completed["usage"] == {"input_tokens": 11, "output_tokens": 7}

    call = transport.calls[0]
    assert call["url"] == "https://api.anthropic.com/v1/messages"
    assert call["payload"]["stream"] is True
    assert call["stream_idle_timeout_s"] == 120.0
    assert call["headers"]["x-api-key"] == "test-key"


@pytest.mark.anyio
async def test_anthropic_stream_events_emits_redacted_thinking_and_empty_tool_input() -> None:
    events = [
        {
            "type": "message_start",
            "message": {"id": "msg_s2", "model": "claude-test", "usage": {"input_tokens": 3}},
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "redacted_thinking", "data": "opaque-blob"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": "ping", "input": {}},
        },
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {}},
        {"type": "message_stop"},
    ]

    translated = [event async for event in anthropic_stream_events(_aiter_events(events))]

    assert [event.type for event in translated] == [
        ModelStreamEventType.THINKING,
        ModelStreamEventType.TOOL_CALL,
        ModelStreamEventType.COMPLETED,
    ]
    assert translated[0].delta == ""
    assert translated[0].payload["provider_state"] == {
        "type": "redacted_thinking",
        "data": "opaque-blob",
    }
    # A tool_use block with no input_json_delta events has empty arguments.
    assert translated[1].payload == {"id": "toolu_1", "name": "ping", "arguments": {}}


@pytest.mark.anyio
async def test_anthropic_provider_stream_error_event_yields_typed_error() -> None:
    events = [
        {
            "type": "message_start",
            "message": {"id": "msg_s3", "model": "claude-test", "usage": {}},
        },
        {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
    ]
    transport = StreamingRecordingTransport([events])
    provider = AnthropicProvider(api_key="test-key", transport=transport)
    request = ModelRequest(model="claude-test", messages=[Message.text("user", "hello")])

    emitted = [event async for event in provider.stream(request)]

    assert [event.type for event in emitted] == [ModelStreamEventType.ERROR]
    assert emitted[0].payload["error"] == "Anthropic streaming error: Overloaded"
    assert emitted[0].payload["error_type"] == "AnthropicAPIError"
    assert emitted[0].payload["provider"] == "anthropic"
    assert emitted[0].payload["provider_error_type"] == "overloaded_error"


@pytest.mark.anyio
async def test_anthropic_provider_stream_error_event_propagates_context_overflow() -> None:
    events = [
        {
            "type": "error",
            "error": {"type": "invalid_request_error", "message": "prompt is too long"},
        },
    ]
    transport = StreamingRecordingTransport([events])
    provider = AnthropicProvider(api_key="test-key", transport=transport)
    request = ModelRequest(model="claude-test", messages=[Message.text("user", "hello")])

    with pytest.raises(AnthropicContextOverflowError) as exc_info:
        [event async for event in provider.stream(request)]

    assert isinstance(exc_info.value, ModelContextOverflowError)
    assert exc_info.value.retryable is False
    assert exc_info.value.error_type == "invalid_request_error"


@pytest.mark.anyio
async def test_anthropic_provider_stream_without_message_stop_is_protocol_error() -> None:
    events = [
        {
            "type": "message_start",
            "message": {"id": "msg_s4", "model": "claude-test", "usage": {}},
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
    ]
    transport = StreamingRecordingTransport([events])
    provider = AnthropicProvider(api_key="test-key", transport=transport)
    request = ModelRequest(model="claude-test", messages=[Message.text("user", "hello")])

    emitted = [event async for event in provider.stream(request)]

    assert [event.type for event in emitted] == [
        ModelStreamEventType.TEXT_DELTA,
        ModelStreamEventType.ERROR,
    ]
    assert "ended before message_stop" in emitted[1].payload["error"]
    assert emitted[1].payload["error_type"] == "AnthropicProtocolError"


@pytest.mark.anyio
async def test_anthropic_stream_events_rejects_unordered_deltas() -> None:
    events = [
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "x"}},
    ]

    with pytest.raises(AnthropicProtocolError, match="before content_block_start"):
        [event async for event in anthropic_stream_events(_aiter_events(events))]


def test_anthropic_provider_rejects_invalid_stream_idle_timeout() -> None:
    with pytest.raises(ValueError, match="stream_idle_timeout_s"):
        AnthropicProvider(api_key="test-key", stream_idle_timeout_s=0)
    with pytest.raises(TypeError, match="stream_idle_timeout_s"):
        AnthropicProvider(api_key="test-key", stream_idle_timeout_s="60")  # type: ignore[arg-type]


@pytest.mark.anyio
async def test_anthropic_provider_resolves_api_key_ref_through_allowlist_proxy() -> None:
    transport = RecordingTransport(
        [
            {
                "id": "msg_1",
                "model": "claude-test",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "hello"}],
                "usage": {"input_tokens": 10, "output_tokens": 2},
            }
        ],
        count_responses=[{"input_tokens": 5}],
    )
    proxy = AllowlistProxy(
        StaticVault({"anthropic_api_key": "sk-vaulted-key"}),
        allowed_destinations=["api.anthropic.com"],
    )
    provider = AnthropicProvider(
        api_key_ref=SecretRef(name="anthropic_api_key"),
        credential_proxy=proxy,
        transport=transport,
    )
    request = ModelRequest(model="claude-test", messages=[Message.text("user", "hi")])

    events = [event async for event in provider.stream(request)]
    count = await provider.count_input_tokens(request)

    # The raw key never lives in provider config; it is resolved per request.
    assert provider.api_key is None
    assert [event.type for event in events] == [
        ModelStreamEventType.TEXT_DELTA,
        ModelStreamEventType.COMPLETED,
    ]
    assert transport.calls[0]["headers"]["x-api-key"] == "sk-vaulted-key"
    assert count is not None
    assert transport.count_calls[0]["headers"]["x-api-key"] == "sk-vaulted-key"


@pytest.mark.anyio
async def test_anthropic_provider_fails_closed_when_proxy_denies_destination() -> None:
    transport = RecordingTransport([])
    proxy = AllowlistProxy(
        StaticVault({"anthropic_api_key": "sk-vaulted-key"}),
        allowed_destinations=["allowed.example.com"],
    )
    provider = AnthropicProvider(
        api_key_ref=SecretRef(name="anthropic_api_key"),
        credential_proxy=proxy,
        transport=transport,
    )
    request = ModelRequest(model="claude-test", messages=[Message.text("user", "hi")])

    events = [event async for event in provider.stream(request)]

    assert [event.type for event in events] == [ModelStreamEventType.ERROR]
    assert "denied" in events[0].payload["error"]
    assert events[0].payload["retryable"] is False
    assert transport.calls == []

    with pytest.raises(AnthropicAPIError, match="denied"):
        await provider.count_input_tokens(request)
    assert transport.count_calls == []


def test_anthropic_provider_validates_credential_source_configuration() -> None:
    proxy = AllowlistProxy(
        StaticVault({"anthropic_api_key": "sk-vaulted-key"}),
        allowed_destinations=["api.anthropic.com"],
    )

    with pytest.raises(ValueError, match="not both"):
        AnthropicProvider(
            api_key="test-key",
            api_key_ref=SecretRef(name="anthropic_api_key"),
            credential_proxy=proxy,
        )

    with pytest.raises(TypeError, match="credential_proxy"):
        AnthropicProvider(api_key_ref=SecretRef(name="anthropic_api_key"))

    with pytest.raises(ValueError, match="requires api_key_ref"):
        AnthropicProvider(api_key="test-key", credential_proxy=proxy)

    with pytest.raises(TypeError, match="SecretRef"):
        AnthropicProvider(api_key_ref="anthropic_api_key", credential_proxy=proxy)  # type: ignore[arg-type]
