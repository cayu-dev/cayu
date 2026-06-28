from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx
import pytest

from cayu import (
    RESOLVED_FILE_ATTACHMENTS_OPTION,
    AgentSpec,
    CacheBreakpoint,
    CachePolicy,
    CayuApp,
    FileAttachmentKind,
    Message,
    RunRequest,
    file_attachment,
)
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
from cayu.providers.cache import resolve_cache_policy


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
