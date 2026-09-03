from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx
import pytest
from tests.provider_traceback_assertions import assert_cayu_traceback_does_not_retain

import cayu.providers.chat_completions as chat_completions_module
from cayu import (
    RESOLVED_FILE_ATTACHMENTS_OPTION,
    AgentSpec,
    CayuApp,
    ChatCompletionsProvider,
    Event,
    EventType,
    ExecutionProfileAdoptionIntent,
    ExecutionProfileAuthorityDecision,
    ExecutionProfilePolicy,
    ExecutionProfilePolicyAction,
    ExecutionProfilePolicyRequest,
    ExecutionProfilePolicyResult,
    FileAttachmentKind,
    InMemorySessionStore,
    Message,
    RecentTurnsContextPolicy,
    ResolutionActor,
    ResolutionActorSource,
    ResumeRequest,
    RetryPolicy,
    RunRequest,
    file_attachment,
)
from cayu.core.messages import FilePart, MessageRole, ProviderStatePart, TextPart, ToolCallPart
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.providers import (
    ChatCompletionsAPIError,
    ChatCompletionsContextOverflowError,
    ChatCompletionsProtocolError,
    HttpxChatCompletionsTransport,
    ModelContextOverflowError,
    ModelFinishReason,
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
    ModelStreamEventType,
    UsageDialect,
    build_chat_completions_payload,
)
from cayu.providers._http import MAX_PROVIDER_ERROR_BODY_CHARS, _TrustedSseJsonEvent
from cayu.providers._sse import aiter_sse_json_events
from cayu.providers.chat_completions import chat_completions_stream_events
from cayu.tools import ListArtifactsTool, ListFilesTool, ReadFileTool


async def _collect_events(app: CayuApp, request: RunRequest) -> list[Event]:
    return [event async for event in app.run(request)]


class RecordingTransport:
    def __init__(
        self,
        stream_events: list[list[Mapping[str, Any]]] | None = None,
    ) -> None:
        self.stream_event_batches = list(stream_events or [])
        self.calls: list[dict[str, Any]] = []

    async def stream_chat_completions(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
        transport_idle_timeout_s: float,
        protocol_idle_timeout_s: float,
        semantic_progress_timeout_s: float,
        absolute_stream_timeout_s: float,
    ):
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "payload": dict(payload),
                "timeout_s": timeout_s,
                "transport_idle_timeout_s": transport_idle_timeout_s,
                "protocol_idle_timeout_s": protocol_idle_timeout_s,
                "semantic_progress_timeout_s": semantic_progress_timeout_s,
                "absolute_stream_timeout_s": absolute_stream_timeout_s,
            }
        )
        if not self.stream_event_batches:
            raise AssertionError("No fake Chat Completions stream queued.")
        for event in self.stream_event_batches.pop(0):
            yield event


class BlankFailingTransport:
    async def stream_chat_completions(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
        transport_idle_timeout_s: float,
        protocol_idle_timeout_s: float,
        semantic_progress_timeout_s: float,
        absolute_stream_timeout_s: float,
    ):
        raise RuntimeError()
        yield {}


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


class _AdoptExecutionProfilePolicy(ExecutionProfilePolicy):
    @property
    def identity(self) -> str:
        return "tests:chat-completions-target-adoption:v1"

    async def decide(
        self,
        request: ExecutionProfilePolicyRequest,
    ) -> ExecutionProfilePolicyResult:
        del request
        return ExecutionProfilePolicyResult(
            action=ExecutionProfilePolicyAction.ADOPT,
            reason="Test-authorized exact Chat Completions target adoption.",
            authority_decision=ExecutionProfileAuthorityDecision.AUTHORIZED,
        )


def _text_chunk(content: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "model": "gemini-test",
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }


def _provider_state_target(
    *,
    provider_name: str = "openrouter",
    endpoint_url: str = "https://openrouter.ai/api/v1/chat/completions",
    model: str = "vendor/reasoning-model",
) -> tuple[str, dict[str, Any]]:
    digest = chat_completions_module._chat_completions_state_target_sha256(
        provider_name=provider_name,
        endpoint_url=endpoint_url,
        model=model,
    )
    return digest, chat_completions_module._provider_state_target(digest)


def _finish_chunk(finish_reason: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "model": "gemini-test",
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    }


def _usage_chunk(usage: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "model": "gemini-test",
        "choices": [],
        "usage": usage,
    }


def _tool_call_chunk(
    tool_calls: list[dict[str, Any]],
    *,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "choices": [
            {
                "index": 0,
                "delta": {"tool_calls": tool_calls},
                "finish_reason": finish_reason,
            }
        ]
    }


def _interleaved_correlated_tool_call_chunks() -> list[Mapping[str, Any]]:
    return [
        _tool_call_chunk(
            [
                {
                    "index": 0,
                    "function": {"name": "alpha", "arguments": '{"value":'},
                }
            ]
        ),
        _tool_call_chunk(
            [
                {
                    "id": "call_b",
                    "function": {"name": "beta", "arguments": '{"value":'},
                }
            ]
        ),
        _tool_call_chunk([{"index": 0, "id": "call_a", "function": {"arguments": "1"}}]),
        _tool_call_chunk([{"index": 1, "id": "call_b", "function": {"arguments": "2"}}]),
        _tool_call_chunk([{"id": "call_a", "function": {"arguments": "}"}}]),
        _tool_call_chunk(
            [{"index": 1, "function": {"arguments": "}"}}],
            finish_reason="tool_calls",
        ),
    ]


def test_build_chat_completions_payload_translates_cayu_messages() -> None:
    request = ModelRequest(
        model="gemini-test",
        messages=[
            Message.text("system", "You are a careful assistant."),
            Message.text("user", "Read a file."),
            Message(
                role=MessageRole.ASSISTANT,
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
                content="file contents",
            ),
        ],
    )

    payload = build_chat_completions_payload(request)

    assert payload["model"] == "gemini-test"
    assert payload["messages"][0] == {"role": "system", "content": "You are a careful assistant."}
    assert payload["messages"][1] == {"role": "user", "content": "Read a file."}
    assert payload["messages"][2] == {
        "role": "assistant",
        "content": "I will inspect it.",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
            }
        ],
    }
    assert payload["messages"][3] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "file contents",
    }


def test_build_chat_completions_payload_round_trips_tool_call_extra_content() -> None:
    target_sha256, target = _provider_state_target(
        provider_name="openai_chat",
        endpoint_url="https://api.openai.com/v1/chat/completions",
        model="gemini-test",
    )
    request = ModelRequest(
        model="gemini-test",
        messages=[
            Message(
                role=MessageRole.ASSISTANT,
                content=[
                    ToolCallPart(
                        tool_call_id="call_1",
                        tool_name="read_file",
                        arguments={"path": "README.md"},
                    ),
                    ProviderStatePart(
                        provider="chat_completions",
                        state={
                            "type": "tool_call_extra_content",
                            "version": 1,
                            "target": target,
                            "tool_call_id": "call_1",
                            "extra_content": {
                                "google": {"thought_signature": "signature-1"},
                            },
                        },
                    ),
                ],
            )
        ],
    )

    payload = build_chat_completions_payload(
        request,
        provider_state_target_sha256=target_sha256,
    )

    assert payload["messages"] == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
                    "extra_content": {
                        "google": {"thought_signature": "signature-1"},
                    },
                }
            ],
        }
    ]


def test_build_chat_completions_payload_round_trips_opaque_reasoning_details() -> None:
    target_sha256, target = _provider_state_target()
    details = [
        {
            "type": "reasoning.encrypted",
            "data": "opaque-encrypted-state",
            "format": "future-provider-v9",
            "unknown": {"signature": "signed-value", "sequence": [2, 1]},
        },
        {"type": "future.reasoning", "opaque": [None, True, 7]},
    ]
    request = ModelRequest(
        model="vendor/reasoning-model",
        messages=[
            Message(
                role=MessageRole.ASSISTANT,
                content=[
                    ToolCallPart(
                        tool_call_id="call_1",
                        tool_name="read_file",
                        arguments={"path": "README.md"},
                    ),
                    ProviderStatePart(
                        provider="chat_completions",
                        state={
                            "type": "reasoning_details",
                            "version": 1,
                            "target": target,
                            "details": details,
                        },
                    ),
                ],
            )
        ],
    )

    payload = build_chat_completions_payload(
        request,
        options_key="openrouter",
        provider_state_target_sha256=target_sha256,
    )

    assert payload["messages"][0]["reasoning_details"] == details
    assert payload["messages"][0]["reasoning_details"] is not details

    different_target_sha256, _ = _provider_state_target(
        endpoint_url="https://different.example.test/v1/chat/completions",
    )
    mismatched = build_chat_completions_payload(
        request,
        options_key="openrouter",
        provider_state_target_sha256=different_target_sha256,
    )
    assert "reasoning_details" not in mismatched["messages"][0]


def test_build_chat_completions_payload_nests_and_cleans_tool_schemas() -> None:
    request = ModelRequest(
        model="gemini-test",
        messages=[Message.text("user", "Hi.")],
        tools=[
            {
                "name": "echo",
                "description": "Echo text.",
                "input_schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "text": {"type": "string"},
                        "meta": {"type": "object", "additionalProperties": True},
                    },
                },
            }
        ],
    )

    payload = build_chat_completions_payload(request, strip_additional_properties=True)

    tool = payload["tools"][0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "echo"
    parameters = tool["function"]["parameters"]
    assert "additionalProperties" not in parameters
    assert "additionalProperties" not in parameters["properties"]["meta"]


def test_build_chat_completions_payload_keeps_property_named_like_a_stripped_keyword() -> None:
    # A parameter literally named "additionalProperties" must survive cleaning: the
    # key is a property name here, not the schema keyword.
    request = ModelRequest(
        model="gemini-test",
        messages=[Message.text("user", "Hi.")],
        tools=[
            {
                "name": "echo",
                "input_schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "additionalProperties": {"type": "boolean"},
                        "text": {"type": "string"},
                    },
                },
            }
        ],
    )

    payload = build_chat_completions_payload(request, strip_additional_properties=True)

    parameters = payload["tools"][0]["function"]["parameters"]
    # The top-level schema keyword is stripped...
    assert "additionalProperties" not in parameters
    # ...but the property named "additionalProperties" is preserved.
    assert parameters["properties"]["additionalProperties"] == {"type": "boolean"}


def test_build_chat_completions_payload_keeps_schema_when_cleaning_disabled() -> None:
    request = ModelRequest(
        model="gemini-test",
        messages=[Message.text("user", "Hi.")],
        tools=[
            {
                "name": "echo",
                "input_schema": {"type": "object", "additionalProperties": False},
            }
        ],
    )

    payload = build_chat_completions_payload(request, clean_schemas=False)

    assert payload["tools"][0]["function"]["parameters"]["additionalProperties"] is False


@pytest.mark.parametrize(
    "tool",
    [ReadFileTool(), ListFilesTool(), ListArtifactsTool()],
    ids=["read", "list-files", "list-artifacts"],
)
def test_build_chat_completions_payload_cleaner_preserves_closed_file_tool_schema(
    tool: Tool,
) -> None:
    request = ModelRequest(
        model="gpt-test",
        messages=[Message.text("user", "Inspect files.")],
        tools=[
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.schema,
            }
        ],
    )

    payload = build_chat_completions_payload(request)

    parameters = payload["tools"][0]["function"]["parameters"]
    assert parameters["additionalProperties"] is False


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("provider_options", "clean_schemas", "preserves_closed_schema"),
    [
        ({}, True, True),
        (
            {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai"},
            True,
            False,
        ),
        ({"strip_additional_properties": True}, True, False),
        (
            {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai"},
            False,
            True,
        ),
    ],
    ids=[
        "generic-cleaned",
        "google-cleaned",
        "explicit-compatibility-cleaned",
        "google-cleaning-disabled",
    ],
)
async def test_provider_projects_closed_schema_for_selected_compatibility_mode(
    provider_options: dict[str, Any],
    clean_schemas: bool,
    preserves_closed_schema: bool,
) -> None:
    transport = RecordingTransport(stream_events=[[_finish_chunk("stop")]])
    provider = ChatCompletionsProvider(
        api_key="test-key",
        clean_schemas=clean_schemas,
        transport=transport,
        **provider_options,
    )
    tool = ReadFileTool()
    request = ModelRequest(
        model="test-model",
        messages=[Message.text("user", "Inspect files.")],
        tools=[
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.schema,
            }
        ],
    )

    events = [event async for event in provider.stream(request)]

    assert [event.type for event in events] == [ModelStreamEventType.COMPLETED]
    parameters = transport.calls[0]["payload"]["tools"][0]["function"]["parameters"]
    assert (parameters.get("additionalProperties") is False) is preserves_closed_schema


def test_build_chat_completions_payload_passes_provider_options() -> None:
    request = ModelRequest(
        model="gemini-test",
        messages=[Message.text("user", "Hi.")],
        options={"openai": {"temperature": 0, "reasoning_effort": "low"}},
    )

    payload = build_chat_completions_payload(request)

    assert payload["temperature"] == 0
    assert payload["reasoning_effort"] == "low"


def test_chat_completions_fingerprint_options_follow_dynamic_payload_namespace() -> None:
    provider = ChatCompletionsProvider(api_key="test-key", name="gemini")
    request = ModelRequest(
        model="gemini-test",
        messages=[Message.text("user", "Hi.")],
        options={
            "gemini": {
                "temperature": 0,
                "metadata": {"route": "active"},
            },
            "openai": {"metadata": {"route": "inactive"}},
            "thinking": {"enabled": True, "effort": "high"},
        },
    )

    assert provider.request_fingerprint_options(request) == {
        "gemini": {
            "temperature": 0,
            "metadata": {"route": "active"},
            "reasoning_effort": "high",
        }
    }


def test_build_chat_completions_payload_rejects_reserved_option() -> None:
    request = ModelRequest(
        model="gemini-test",
        messages=[Message.text("user", "Hi.")],
        options={"openai": {"messages": []}},
    )

    with pytest.raises(ValueError, match="reserved"):
        build_chat_completions_payload(request)


def test_build_chat_completions_payload_translates_image_attachment() -> None:
    attachment = file_attachment(
        artifact_id="art_img",
        kind=FileAttachmentKind.IMAGE,
        filename="chart.png",
        content_type="image/png",
        size_bytes=5,
    )
    request = ModelRequest(
        model="gemini-test",
        messages=[
            Message.text("user", "Inspect the chart."),
            Message.tool_call(
                tool_call_id="call_1",
                tool_name="read_file",
                arguments={"artifact_id": "art_img"},
            ),
            Message.tool_result(
                tool_call_id="call_1",
                tool_name="read_file",
                content="Attached image artifact art_img: chart.png.",
                artifacts=[attachment],
            ),
        ],
        options={
            RESOLVED_FILE_ATTACHMENTS_OPTION: {
                "art_img": {
                    "artifact_id": "art_img",
                    "kind": "image",
                    "filename": "chart.png",
                    "content_type": "image/png",
                    "data_base64": "aGVsbG8=",
                    "metadata": {},
                }
            }
        },
    )

    payload = build_chat_completions_payload(request)

    assert payload["messages"][-2] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "Attached image artifact art_img: chart.png.",
    }
    assert payload["messages"][-1] == {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "The previous tool result returned file content for inspection.",
            },
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,aGVsbG8="},
            },
        ],
    }


def _pdf_document_request() -> ModelRequest:
    attachment = file_attachment(
        artifact_id="art_pdf",
        kind=FileAttachmentKind.DOCUMENT,
        filename="invoice.pdf",
        content_type="application/pdf",
        size_bytes=5,
    )
    return ModelRequest(
        model="gemini-test",
        messages=[
            Message.text("user", "Read the invoice."),
            Message.tool_call(
                tool_call_id="call_1",
                tool_name="read_file",
                arguments={"artifact_id": "art_pdf"},
            ),
            Message.tool_result(
                tool_call_id="call_1",
                tool_name="read_file",
                content="Attached PDF artifact art_pdf: invoice.pdf.",
                artifacts=[attachment],
            ),
        ],
        options={
            RESOLVED_FILE_ATTACHMENTS_OPTION: {
                "art_pdf": {
                    "artifact_id": "art_pdf",
                    "kind": "document",
                    "filename": "invoice.pdf",
                    "content_type": "application/pdf",
                    "data_base64": "aGVsbG8=",
                    "metadata": {},
                }
            }
        },
    )


def test_build_chat_completions_payload_emits_pdf_attachment() -> None:
    payload = build_chat_completions_payload(_pdf_document_request())

    # The tool result is followed by a synthetic user message carrying the PDF as
    # an OpenAI Chat Completions `file` content part (the default encoding).
    assert payload["messages"][-2] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "Attached PDF artifact art_pdf: invoice.pdf.",
    }
    assert payload["messages"][-1] == {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "The previous tool result returned file content for inspection.",
            },
            {
                "type": "file",
                "file": {
                    "filename": "invoice.pdf",
                    "file_data": "data:application/pdf;base64,aGVsbG8=",
                },
            },
        ],
    }


def test_build_chat_completions_payload_emits_pdf_as_image_url_when_configured() -> None:
    payload = build_chat_completions_payload(_pdf_document_request(), document_encoding="image_url")

    # Gemini's compatible endpoint carries PDFs through the image_url content part.
    assert payload["messages"][-1]["content"][-1] == {
        "type": "image_url",
        "image_url": {"url": "data:application/pdf;base64,aGVsbG8="},
    }


def test_build_chat_completions_payload_translates_user_file_parts() -> None:
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
        model="gemini-test",
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

    payload = build_chat_completions_payload(request)

    assert payload["messages"][0] == {
        "role": "user",
        "content": [
            {"type": "text", "text": "Read the invoice and the contract."},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,aGVsbG8="},
            },
            {
                "type": "file",
                "file": {
                    "filename": "contract.pdf",
                    "file_data": "data:application/pdf;base64,JVBERi0xLjQ=",
                },
            },
        ],
    }

    # Text-only user turns keep the plain-string content shape.
    text_only = build_chat_completions_payload(
        ModelRequest(model="gemini-test", messages=[Message.text("user", "Hi.")])
    )
    assert text_only["messages"][0] == {"role": "user", "content": "Hi."}


def test_build_chat_completions_payload_requires_resolved_user_file_parts() -> None:
    image = file_attachment(
        artifact_id="art_missing",
        kind=FileAttachmentKind.IMAGE,
        filename="invoice.png",
        content_type="image/png",
        size_bytes=5,
    )
    request = ModelRequest(
        model="gemini-test",
        messages=[
            Message(
                role="user",
                content=[TextPart(text="Read it."), FilePart(attachment=image)],
            ),
        ],
    )

    with pytest.raises(ChatCompletionsProtocolError, match="Missing resolved file attachment"):
        build_chat_completions_payload(request)


@pytest.mark.parametrize("document_encoding", ["bogus", [], {}, 1, True, None])
def test_provider_rejects_invalid_document_encoding(document_encoding: object) -> None:
    with pytest.raises(
        ValueError,
        match=r"^document_encoding must be one of \['file', 'image_url'\]\.$",
    ):
        ChatCompletionsProvider(
            api_key="test-key",
            name="gemini",
            document_encoding=document_encoding,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("document_encoding", ["bogus", [], {}, 1, True, None])
def test_build_chat_completions_payload_rejects_invalid_document_encoding(
    document_encoding: object,
) -> None:
    request = ModelRequest(model="gemini-test", messages=[Message.text("user", "Hi.")])
    with pytest.raises(
        ValueError,
        match=r"^document_encoding must be one of \['file', 'image_url'\]\.$",
    ):
        build_chat_completions_payload(
            request,
            document_encoding=document_encoding,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("timeout_s", [0, -1, math.nan, math.inf, -math.inf, 10**1000])
def test_provider_rejects_non_positive_or_non_finite_timeout(timeout_s: int | float) -> None:
    with pytest.raises(
        ValueError,
        match=r"^timeout_s must be finite and greater than zero\.$",
    ):
        ChatCompletionsProvider(api_key="test-key", timeout_s=timeout_s)


@pytest.mark.parametrize("timeout_s", [True, "1", None])
def test_provider_rejects_non_numeric_timeout(timeout_s: object) -> None:
    with pytest.raises(TypeError, match=r"^timeout_s must be a number\.$"):
        ChatCompletionsProvider(
            api_key="test-key",
            timeout_s=timeout_s,  # type: ignore[arg-type]
        )


@pytest.mark.anyio
@pytest.mark.parametrize("timeout_s", [math.nan, math.inf, -math.inf])
async def test_http_transport_rejects_non_finite_timeout_before_client_use(
    timeout_s: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = HttpxChatCompletionsTransport()

    def fail_client_use() -> None:
        pytest.fail("HTTP client must not be acquired for an invalid timeout")

    monkeypatch.setattr(transport._client, "get", fail_client_use)
    events = transport.stream_chat_completions(
        url="https://example.test/v1/chat/completions",
        headers={},
        payload={},
        timeout_s=timeout_s,
        transport_idle_timeout_s=1.0,
        protocol_idle_timeout_s=1.0,
        semantic_progress_timeout_s=1.0,
        absolute_stream_timeout_s=1.0,
    )

    with pytest.raises(
        ValueError,
        match=r"^timeout_s must be finite and greater than zero\.$",
    ):
        await anext(events)


@pytest.mark.anyio
async def test_provider_emits_text_and_completed_events() -> None:
    transport = RecordingTransport(
        stream_events=[
            [
                _text_chunk("hello"),
                _finish_chunk("stop"),
                _usage_chunk({"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}),
            ]
        ]
    )
    provider = ChatCompletionsProvider(api_key="test-key", name="gemini", transport=transport)
    request = ModelRequest(
        model="gemini-test",
        messages=[Message.text("user", "Say hello.")],
    )

    events = [event async for event in provider.stream(request)]

    assert [event.type for event in events] == [
        ModelStreamEventType.TEXT_DELTA,
        ModelStreamEventType.COMPLETED,
    ]
    assert events[0].delta == "hello"
    assert events[1].completion is not None
    assert events[1].completion.finish_reason == ModelFinishReason.STOP
    assert events[1].payload["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "total_tokens": 12,
    }
    assert transport.calls[0]["url"] == "https://api.openai.com/v1/chat/completions"
    sent_headers = {key.lower(): value for key, value in transport.calls[0]["headers"].items()}
    assert sent_headers["authorization"] == "Bearer test-key"
    assert transport.calls[0]["payload"]["stream"] is True
    assert transport.calls[0]["payload"]["stream_options"] == {"include_usage": True}
    assert transport.calls[0]["payload"]["messages"] == [{"role": "user", "content": "Say hello."}]


@pytest.mark.anyio
async def test_openrouter_provider_configuration_emits_attribution_and_metadata_headers() -> None:
    referer = "https://example.test/cayu"
    title = "Cayu test"
    transport = RecordingTransport(stream_events=[[_finish_chunk("stop")]])
    provider = ChatCompletionsProvider(
        api_key="test-key",
        name="openrouter",
        api_key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        openrouter_http_referer=referer,
        openrouter_app_title=title,
        openrouter_router_metadata=True,
        transport=transport,
    )

    events = [
        event
        async for event in provider.stream(
            ModelRequest(
                model="vendor/model",
                messages=[Message.text("user", "Hi.")],
                options={
                    "openrouter": {
                        "provider": {
                            "order": ["anthropic"],
                            "allow_fallbacks": False,
                            "require_parameters": True,
                            "zdr": True,
                        }
                    }
                },
            )
        )
    ]

    assert events[-1].type == ModelStreamEventType.COMPLETED
    headers = transport.calls[0]["headers"]
    assert headers["HTTP-Referer"] == referer
    assert headers["X-OpenRouter-Title"] == title
    assert headers["X-OpenRouter-Metadata"] == "enabled"
    assert referer not in repr(events)
    assert title not in repr(events)
    assert transport.calls[0]["payload"]["provider"] == {
        "order": ["anthropic"],
        "allow_fallbacks": False,
        "require_parameters": True,
        "zdr": True,
    }
    default_headers = ChatCompletionsProvider(
        api_key="test-key",
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        transport=RecordingTransport(),
    )._headers()
    assert "HTTP-Referer" not in default_headers
    assert "X-OpenRouter-Title" not in default_headers
    assert "X-OpenRouter-Metadata" not in default_headers


@pytest.mark.anyio
async def test_provider_emits_tool_call_events() -> None:
    transport = RecordingTransport(
        stream_events=[
            [
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {"name": "echo", "arguments": ""},
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [{"index": 0, "function": {"arguments": '{"text":'}}]
                            },
                            "finish_reason": None,
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [{"index": 0, "function": {"arguments": '"hello"}'}}]
                            },
                            "finish_reason": None,
                        }
                    ]
                },
                _finish_chunk("tool_calls"),
            ]
        ]
    )
    provider = ChatCompletionsProvider(api_key="test-key", name="gemini", transport=transport)
    request = ModelRequest(
        model="gemini-test",
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
    assert events[1].completion is not None
    assert events[1].completion.finish_reason == ModelFinishReason.TOOL_CALLS


@pytest.mark.anyio
async def test_provider_correlates_interleaved_tool_call_identity_transitions() -> None:
    transport = RecordingTransport(stream_events=[_interleaved_correlated_tool_call_chunks()])
    provider = ChatCompletionsProvider(
        api_key="test-key",
        name="openai_chat",
        transport=transport,
    )
    request = ModelRequest(
        model="compatible-model",
        messages=[Message.text("user", "Use both tools.")],
    )

    events = [event async for event in provider.stream(request)]

    assert [event.type for event in events] == [
        ModelStreamEventType.TOOL_CALL,
        ModelStreamEventType.TOOL_CALL,
        ModelStreamEventType.COMPLETED,
    ]
    assert events[0].payload == {
        "id": "call_a",
        "name": "alpha",
        "arguments": {"value": 1},
    }
    assert events[1].payload == {
        "id": "call_b",
        "name": "beta",
        "arguments": {"value": 2},
    }


@pytest.mark.anyio
async def test_provider_rejects_multiple_choices_before_tool_call_emission() -> None:
    transport = RecordingTransport(
        stream_events=[
            [
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_a",
                                        "function": {"name": "alpha", "arguments": "{}"},
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        },
                        {
                            "index": 1,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_b",
                                        "function": {"name": "beta", "arguments": "{}"},
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        },
                    ]
                }
            ]
        ]
    )
    provider = ChatCompletionsProvider(
        api_key="test-key",
        name="openai_chat",
        transport=transport,
    )
    request = ModelRequest(
        model="compatible-model",
        messages=[Message.text("user", "Use a tool.")],
    )

    events = [event async for event in provider.stream(request)]

    assert [event.type for event in events] == [ModelStreamEventType.ERROR]
    assert events[0].payload["error_type"] == "ChatCompletionsProtocolError"


@pytest.mark.anyio
async def test_provider_round_trips_runtime_tool_results() -> None:
    transport = RecordingTransport(
        stream_events=[
            [
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "extra_content": {
                                            "google": {"thought_signature": "signature-1"}
                                        },
                                        "function": {
                                            "name": "echo",
                                            "arguments": '{"text":"hello from gemini"}',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ]
                },
                _finish_chunk("stop"),
            ],
            [
                _text_chunk("final answer"),
                _finish_chunk("stop"),
            ],
        ]
    )
    provider = ChatCompletionsProvider(api_key="test-key", name="gemini", transport=transport)
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(
            name="assistant",
            model="gemini-test",
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
        {"role": "system", "content": "Use tools when needed."},
        {"role": "user", "content": "Echo this."},
    ]
    assert transport.calls[1]["payload"]["messages"] == [
        {"role": "system", "content": "Use tools when needed."},
        {"role": "user", "content": "Echo this."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "extra_content": {"google": {"thought_signature": "signature-1"}},
                    "function": {
                        "name": "echo",
                        "arguments": '{"text":"hello from gemini"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "hello from gemini"},
    ]


@pytest.mark.anyio
async def test_openrouter_runtime_replays_reasoning_and_persists_safe_route_evidence() -> None:
    reasoning_details = [
        {
            "type": "reasoning.summary",
            "summary": "Need to use the echo tool.",
            "format": "anthropic-claude-v1",
            "index": 0,
        },
        {
            "type": "reasoning.encrypted",
            "data": "opaque-openrouter-state",
            "format": "future-provider-v9",
            "signature": "signed-state",
            "unknown_future_field": {"preserve": [3, 2, 1]},
        },
    ]
    route_metadata = {
        "requested": "openrouter/auto",
        "strategy": "fallback",
        "region": "iad",
        "attempt": 2,
        "is_byok": False,
        "endpoints": {
            "total": 2,
            "available": [
                {
                    "provider": "Anthropic",
                    "model": "anthropic/claude-sonnet-4.6",
                    "selected": True,
                }
            ],
        },
        "pipeline": [{"data": {"prompt": "must-not-persist"}}],
    }
    usage = {
        "prompt_tokens": 100,
        "prompt_tokens_details": {"cached_tokens": 40, "cache_write_tokens": 10},
        "completion_tokens": 20,
        "completion_tokens_details": {"reasoning_tokens": 7},
        "total_tokens": 120,
        "cost": 0.0123,
        "cost_details": {"upstream_inference_cost": 0.01},
    }
    transport = RecordingTransport(
        stream_events=[
            [
                {
                    "id": "gen-openrouter-tool",
                    "model": "anthropic/claude-sonnet-4.6",
                    "provider": "Anthropic",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"reasoning_details": [reasoning_details[0]]},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "gen-openrouter-tool",
                    "model": "anthropic/claude-sonnet-4.6",
                    "provider": "Anthropic",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "reasoning_details": [reasoning_details[1]],
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "echo",
                                            "arguments": '{"text":"hello from openrouter"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "gen-openrouter-tool",
                    "model": "anthropic/claude-sonnet-4.6",
                    "provider": "Anthropic",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                    "usage": usage,
                    "openrouter_metadata": route_metadata,
                },
            ],
            [
                {
                    "id": "gen-openrouter-retry",
                    "model": "anthropic/claude-sonnet-4.6",
                    "provider": "Anthropic",
                    "error": {
                        "code": 429,
                        "message": "rate limited",
                        "metadata": {"error_type": "rate_limit_exceeded"},
                    },
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": ""},
                            "finish_reason": "error",
                        }
                    ],
                }
            ],
            [
                {
                    "id": "gen-openrouter-final",
                    "model": "anthropic/claude-sonnet-4.6",
                    "provider": "Anthropic",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "final answer"},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "gen-openrouter-final",
                    "model": "anthropic/claude-sonnet-4.6",
                    "provider": "Anthropic",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": 120,
                        "completion_tokens": 3,
                        "total_tokens": 123,
                    },
                    "openrouter_metadata": {
                        **route_metadata,
                        "attempt": 1,
                    },
                },
            ],
        ]
    )
    provider = ChatCompletionsProvider(
        api_key="test-key",
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        transport=transport,
    )
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(
            name="assistant",
            model="openrouter/auto",
            system_prompt="Use tools when needed.",
            provider_options={
                "openrouter": {
                    "provider": {
                        "order": ["anthropic"],
                        "allow_fallbacks": True,
                        "require_parameters": True,
                        "data_collection": "deny",
                        "zdr": True,
                    }
                }
            },
        ),
        tools=[EchoTool()],
    )

    session_id = "openrouter-reasoning-tool-round"
    events = await _collect_events(
        app,
        RunRequest(
            agent_name="assistant",
            session_id=session_id,
            messages=[Message.text("user", "Echo this.")],
            retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
        ),
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert len(transport.calls) == 3
    assert any(event.type == EventType.MODEL_RETRY for event in events)
    assert transport.calls[0]["payload"]["provider"] == {
        "order": ["anthropic"],
        "allow_fallbacks": True,
        "require_parameters": True,
        "data_collection": "deny",
        "zdr": True,
    }
    assert transport.calls[1]["payload"] == transport.calls[2]["payload"]
    assert transport.calls[2]["payload"]["messages"] == [
        {"role": "system", "content": "Use tools when needed."},
        {"role": "user", "content": "Echo this."},
        {
            "role": "assistant",
            "content": None,
            "reasoning_details": reasoning_details,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "echo",
                        "arguments": '{"text":"hello from openrouter"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "hello from openrouter"},
    ]
    transcript = await store.load_transcript(session_id)
    provider_states = [
        part
        for message in transcript
        for part in message.content
        if type(part) is ProviderStatePart and part.state.get("type") == "reasoning_details"
    ]
    assert len(provider_states) == 1
    assert provider_states[0].state["details"] == reasoning_details
    _, expected_target = _provider_state_target(model="openrouter/auto")
    assert provider_states[0].state["target"] == expected_target
    completed = [event for event in events if event.type == EventType.MODEL_COMPLETED]
    first_completion = completed[0].payload
    assert first_completion["model"] == "anthropic/claude-sonnet-4.6"
    assert first_completion["requested_model"] == "openrouter/auto"
    assert first_completion["upstream_provider"] == "Anthropic"
    assert first_completion["openrouter_metadata"] == {
        "requested": "openrouter/auto",
        "strategy": "fallback",
        "attempt": 2,
        "is_byok": False,
        "endpoint_total": 2,
        "selected_endpoint": {
            "provider": "Anthropic",
            "model": "anthropic/claude-sonnet-4.6",
        },
    }
    assert "pipeline" not in first_completion["openrouter_metadata"]
    assert first_completion["usage"]["cost"] == 0.0123
    assert first_completion["usage"]["cost_details"] == {"upstream_inference_cost": 0.01}
    assert first_completion["usage_metrics"]["model"] == "anthropic/claude-sonnet-4.6"
    assert first_completion["usage_metrics"]["requested_model"] == "openrouter/auto"
    assert first_completion["usage_metrics"]["reasoning_output_tokens"] == 7
    assert first_completion["usage_metrics"]["cache"] == {
        "read_tokens": 40,
        "write_tokens": 10,
        "write_unknown_ttl_tokens": 10,
        "cached_input_tokens": 40,
        "uncached_input_tokens": 50,
    }
    assert all("provider_state" not in event.payload for event in events)


@pytest.mark.anyio
async def test_same_name_endpoint_adoption_omits_prior_opaque_reasoning_state() -> None:
    model = "vendor/reasoning-model"
    reasoning_details = [{"type": "reasoning.encrypted", "data": "source-endpoint-only-state"}]
    source_transport = RecordingTransport(
        stream_events=[
            [
                {
                    "id": "source-tool",
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "reasoning_details": reasoning_details,
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "echo",
                                            "arguments": '{"text":"source"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                }
            ],
            [
                {
                    "id": "source-final",
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "source complete"},
                            "finish_reason": "stop",
                        }
                    ],
                }
            ],
        ]
    )
    store = InMemorySessionStore()
    source_app = CayuApp(session_store=store, enable_logging=False)
    source_app.register_provider(
        ChatCompletionsProvider(
            api_key="source-key",
            name="shared-chat",
            base_url="https://source.example.test/v1",
            transport=source_transport,
        ),
        default=True,
    )
    source_app.register_agent(
        AgentSpec(name="assistant", model=model),
        tools=[EchoTool()],
    )
    await _collect_events(
        source_app,
        RunRequest(
            agent_name="assistant",
            session_id="same-name-chat-target-adoption",
            messages=[Message.text("user", "Use the tool.")],
        ),
    )

    target_transport = RecordingTransport(
        stream_events=[
            [
                {
                    "id": "target-final",
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "target complete"},
                            "finish_reason": "stop",
                        }
                    ],
                }
            ]
        ]
    )
    target_app = CayuApp(
        session_store=store,
        execution_profile_policy=_AdoptExecutionProfilePolicy(),
        enable_logging=False,
    )
    target_app.register_provider(
        ChatCompletionsProvider(
            api_key="target-key",
            name="shared-chat",
            base_url="https://target.example.test/v1",
            transport=target_transport,
        ),
        default=True,
    )
    target_app.register_agent(
        AgentSpec(name="assistant", model=model),
        tools=[EchoTool()],
    )
    resumed_events = [
        event
        async for event in target_app.resume(
            ResumeRequest(
                session_id="same-name-chat-target-adoption",
                messages=[Message.text("user", "Continue on the adopted endpoint.")],
                profile_adoption=ExecutionProfileAdoptionIntent(
                    idempotency_key="adopt-same-name-chat-target-v1",
                    reason="Adopt the reviewed endpoint.",
                    requested_by=ResolutionActor(
                        subject="maintainer",
                        source=ResolutionActorSource.REQUEST,
                    ),
                ),
            )
        )
    ]

    assert resumed_events[-1].type == EventType.SESSION_COMPLETED
    target_messages = target_transport.calls[0]["payload"]["messages"]
    assert all("reasoning_details" not in message for message in target_messages)
    assert any(message.get("tool_calls") for message in target_messages)
    transcript = await store.load_transcript("same-name-chat-target-adoption")
    assert any(
        type(part) is ProviderStatePart and part.state.get("details") == reasoning_details
        for message in transcript
        for part in message.content
    )


@pytest.mark.anyio
async def test_chat_completions_stream_events_accumulates_tool_call_fragments() -> None:
    async def raw_events():
        yield {
            "id": "chatcmpl-1",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_9",
                                "function": {"name": "echo", "arguments": '{"te'},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
        yield {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'xt":"hi"}'}}]},
                    "finish_reason": "tool_calls",
                }
            ]
        }

    events = [event async for event in chat_completions_stream_events(raw_events())]

    assert [event.type for event in events] == [
        ModelStreamEventType.TOOL_CALL,
        ModelStreamEventType.COMPLETED,
    ]
    assert events[0].payload == {"id": "call_9", "name": "echo", "arguments": {"text": "hi"}}


@pytest.mark.anyio
async def test_chat_completions_stream_events_correlates_interleaved_aliases() -> None:
    async def raw_events():
        for chunk in _interleaved_correlated_tool_call_chunks():
            yield chunk

    events = [event async for event in chat_completions_stream_events(raw_events())]

    assert [event.type for event in events] == [
        ModelStreamEventType.TOOL_CALL,
        ModelStreamEventType.TOOL_CALL,
        ModelStreamEventType.COMPLETED,
    ]
    assert [event.payload for event in events[:2]] == [
        {"id": "call_a", "name": "alpha", "arguments": {"value": 1}},
        {"id": "call_b", "name": "beta", "arguments": {"value": 2}},
    ]


@pytest.mark.anyio
async def test_chat_completions_stream_events_rejects_multiple_choices_before_deltas() -> None:
    observed: list[ModelStreamEvent] = []

    async def raw_events():
        yield {
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "candidate zero"},
                    "finish_reason": "stop",
                },
                {
                    "index": 1,
                    "delta": {"content": "candidate one"},
                    "finish_reason": "stop",
                },
            ]
        }

    with pytest.raises(ChatCompletionsProtocolError, match="multiple choices"):
        async for event in chat_completions_stream_events(raw_events()):
            observed.append(event)

    assert observed == []


@pytest.mark.anyio
@pytest.mark.parametrize("choice_index", [True, -1, 0.0, "0", [], {}])
async def test_chat_completions_stream_events_rejects_invalid_choice_index(
    choice_index: object,
) -> None:
    async def raw_events():
        yield {
            "choices": [
                {
                    "index": choice_index,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ]
        }

    with pytest.raises(ChatCompletionsProtocolError, match="choice index"):
        [event async for event in chat_completions_stream_events(raw_events())]


@pytest.mark.anyio
async def test_chat_completions_stream_events_rejects_changed_choice_index() -> None:
    observed: list[ModelStreamEvent] = []

    async def raw_events():
        yield _text_chunk("candidate zero")
        yield {
            "choices": [
                {
                    "index": 1,
                    "delta": {"content": "candidate one"},
                    "finish_reason": "stop",
                }
            ]
        }

    with pytest.raises(ChatCompletionsProtocolError, match="conflicting choice indexes"):
        async for event in chat_completions_stream_events(raw_events()):
            observed.append(event)

    assert [event.delta for event in observed] == ["candidate zero"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("tool_call_chunks", "message"),
    [
        (
            [
                [
                    {
                        "index": 0,
                        "id": "call_a",
                        "function": {"name": "alpha", "arguments": "{}"},
                    }
                ],
                [{"index": 0, "id": "call_b"}],
            ],
            "changed ids",
        ),
        (
            [
                [
                    {
                        "index": 0,
                        "id": "call_a",
                        "function": {"name": "alpha", "arguments": "{}"},
                    }
                ],
                [{"index": 1, "id": "call_a"}],
            ],
            "changed indexes",
        ),
        (
            [
                [
                    {
                        "index": 0,
                        "id": "call_a",
                        "function": {"name": "alpha", "arguments": "{}"},
                    },
                    {
                        "index": 1,
                        "id": "call_b",
                        "function": {"name": "beta", "arguments": "{}"},
                    },
                ],
                [{"index": 0, "id": "call_b"}],
            ],
            "identify different calls",
        ),
    ],
)
async def test_chat_completions_stream_events_rejects_tool_call_alias_conflicts(
    tool_call_chunks: list[list[dict[str, Any]]],
    message: str,
) -> None:
    async def raw_events():
        for tool_calls in tool_call_chunks:
            yield _tool_call_chunk(tool_calls)

    with pytest.raises(ChatCompletionsProtocolError, match=message):
        [event async for event in chat_completions_stream_events(raw_events())]


@pytest.mark.anyio
async def test_chat_completions_stream_events_rejects_duplicate_function_names() -> None:
    async def raw_events():
        yield _tool_call_chunk(
            [
                {
                    "index": 0,
                    "id": "call_a",
                    "function": {"name": "alpha", "arguments": ""},
                }
            ]
        )
        yield _tool_call_chunk([{"index": 0, "function": {"name": "alpha", "arguments": "{}"}}])

    with pytest.raises(ChatCompletionsProtocolError, match="more than one function name"):
        [event async for event in chat_completions_stream_events(raw_events())]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("tool_call", "message"),
    [
        (
            {"index": 0, "function": {"name": "alpha", "arguments": "{}"}},
            "missing an id",
        ),
        (
            {
                "index": 0,
                "id": "call_a",
                "function": {"name": "alpha", "arguments": "{"},
            },
            "not valid JSON",
        ),
    ],
)
async def test_chat_completions_stream_events_preserves_terminal_tool_call_validation(
    tool_call: dict[str, Any],
    message: str,
) -> None:
    async def raw_events():
        yield _tool_call_chunk([tool_call], finish_reason="tool_calls")

    with pytest.raises(ChatCompletionsProtocolError, match=message):
        [event async for event in chat_completions_stream_events(raw_events())]


@pytest.mark.anyio
async def test_chat_completions_stream_events_handles_gemini_tool_call_without_index() -> None:
    # Gemini's OpenAI-compatible endpoint omits the per-tool-call ``index`` and
    # delivers the complete call (with an ``id``) in a single delta. The index
    # lives on the choice, not on the tool_call entry.
    async def raw_events():
        yield {
            "id": "gemini-1",
            "model": "gemini-2.5-flash",
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "function-call-123",
                                "type": "function",
                                "function": {"name": "echo", "arguments": '{"text":"hi"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 43, "completion_tokens": 13, "total_tokens": 56},
        }

    events = [event async for event in chat_completions_stream_events(raw_events())]

    assert [event.type for event in events] == [
        ModelStreamEventType.TOOL_CALL,
        ModelStreamEventType.COMPLETED,
    ]
    assert events[0].payload == {
        "id": "function-call-123",
        "name": "echo",
        "arguments": {"text": "hi"},
    }
    assert events[1].completion is not None
    assert events[1].completion.finish_reason == ModelFinishReason.TOOL_CALLS


@pytest.mark.anyio
async def test_chat_completions_stream_events_preserves_gemini_tool_call_extra_content() -> None:
    target_sha256, target = _provider_state_target(
        provider_name="gemini",
        endpoint_url="https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        model="gemini-3.1-flash-lite",
    )

    async def raw_events():
        yield {
            "id": "gemini-1",
            "model": "gemini-3.1-flash-lite",
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "function-call-123",
                                "type": "function",
                                "extra_content": {"google": {"thought_signature": "signature-1"}},
                                "function": {"name": "echo", "arguments": '{"text":"hi"}'},
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        }
        yield {
            "id": "gemini-1",
            "model": "gemini-3.1-flash-lite",
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": "stop",
                }
            ],
        }

    events = [
        event
        async for event in chat_completions_stream_events(
            raw_events(),
            provider_state_target_sha256=target_sha256,
        )
    ]

    assert [event.type for event in events] == [
        ModelStreamEventType.TOOL_CALL,
        ModelStreamEventType.COMPLETED,
    ]
    assert events[0].payload == {
        "id": "function-call-123",
        "name": "echo",
        "arguments": {"text": "hi"},
    }
    assert events[1].completion is not None
    assert events[1].completion.finish_reason == ModelFinishReason.TOOL_CALLS
    assert events[1].payload["provider_state"] == [
        {
            "provider": "chat_completions",
            "state": {
                "type": "tool_call_extra_content",
                "version": 1,
                "target": target,
                "tool_call_id": "function-call-123",
                "extra_content": {"google": {"thought_signature": "signature-1"}},
            },
        }
    ]


@pytest.mark.anyio
async def test_chat_completions_stream_events_rejects_malformed_reasoning_before_tool_call() -> (
    None
):
    async def raw_events():
        yield {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "reasoning_details": {"type": "reasoning.text"},
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "must-not-run",
                                "function": {"name": "echo", "arguments": '{"text":"no"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

    emitted: list[ModelStreamEvent] = []
    with pytest.raises(ChatCompletionsProtocolError, match="reasoning_details must be a list"):
        async for event in chat_completions_stream_events(raw_events()):
            emitted.append(event)

    assert emitted == []


@pytest.mark.anyio
async def test_chat_completions_stream_events_rejects_undurable_reasoning_before_tool_call() -> (
    None
):
    async def raw_events():
        yield {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "reasoning_details": [{"type": "future.reasoning", "counter": 2**63}],
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "must-not-run",
                                "function": {"name": "echo", "arguments": '{"text":"no"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

    emitted: list[ModelStreamEvent] = []
    with pytest.raises(ChatCompletionsProtocolError, match="not durable JSON"):
        async for event in chat_completions_stream_events(raw_events()):
            emitted.append(event)

    assert emitted == []


@pytest.mark.anyio
async def test_chat_completions_stream_events_projects_bounded_openrouter_route_evidence() -> None:
    async def raw_events():
        yield {
            "id": "gen-1",
            "model": "anthropic/claude-sonnet-4.6",
            "provider": "Tencent Cloud",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "openrouter_metadata": {
                "requested": "openrouter/auto",
                "strategy": "direct",
                "region": "iad",
                "summary": "unbounded human description must not persist",
                "attempt": 2,
                "is_byok": False,
                "endpoints": {
                    "total": 2,
                    "available": [
                        {
                            "provider": "Tencent Cloud",
                            "model": "anthropic/claude-sonnet-4.6",
                            "selected": True,
                            "unknown": "ignored",
                        },
                        {
                            "provider": "Other",
                            "model": "other/model",
                            "selected": False,
                        },
                    ],
                },
                "attempts": [{"provider": "secret-upstream-detail"}],
                "pipeline": [{"data": {"prompt": "must-not-persist"}}],
                "future": {"free_form": "must-not-persist"},
            },
        }

    events = [
        event
        async for event in chat_completions_stream_events(
            raw_events(),
            requested_model="openrouter/auto",
        )
    ]

    assert events[-1].payload["model"] == "anthropic/claude-sonnet-4.6"
    assert events[-1].payload["upstream_provider"] == "Tencent Cloud"
    assert events[-1].payload["openrouter_metadata"] == {
        "requested": "openrouter/auto",
        "strategy": "direct",
        "attempt": 2,
        "is_byok": False,
        "endpoint_total": 2,
        "selected_endpoint": {
            "provider": "Tencent Cloud",
            "model": "anthropic/claude-sonnet-4.6",
        },
    }


@pytest.mark.parametrize(
    "strategy",
    [
        "alias",
        "auto",
        "bodybuilder",
        "direct",
        "fallback",
        "free",
        "fusion",
        "latest",
        "pareto",
    ],
)
def test_openrouter_documented_router_strategies_are_retained(strategy: str) -> None:
    assert chat_completions_module._safe_openrouter_metadata(
        {"strategy": strategy},
        requested_model=None,
        effective_model=None,
    ) == {"strategy": strategy}


@pytest.mark.anyio
async def test_chat_completions_stream_events_omits_untrusted_router_strings() -> None:
    arbitrary = "prompt-or-credential-canary"

    async def raw_events():
        yield {
            "id": "gen-unsafe-route-evidence",
            "model": "anthropic/claude-sonnet-4.6",
            "provider": arbitrary,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "openrouter_metadata": {
                "requested": arbitrary,
                "strategy": arbitrary,
                "region": arbitrary,
                "attempt": 1,
                "endpoints": {
                    "available": [
                        {
                            "selected": True,
                            "provider": arbitrary,
                            "model": arbitrary,
                        }
                    ]
                },
            },
        }

    events = [
        event
        async for event in chat_completions_stream_events(
            raw_events(),
            requested_model="openrouter/auto",
        )
    ]

    completion = events[-1].payload
    assert "upstream_provider" not in completion
    assert len(completion["upstream_provider_sha256"]) == 64
    assert completion["openrouter_metadata"]["attempt"] == 1
    assert len(completion["openrouter_metadata"]["strategy_sha256"]) == 64
    selected = completion["openrouter_metadata"]["selected_endpoint"]
    assert len(selected["provider_sha256"]) == 64
    assert arbitrary not in repr(completion)


@pytest.mark.anyio
async def test_chat_completions_stream_events_normalizes_legacy_function_call_finish() -> None:
    async def raw_events():
        yield _finish_chunk("function_call")
        yield _finish_chunk("tool_calls")

    events = [event async for event in chat_completions_stream_events(raw_events())]

    assert [event.type for event in events] == [ModelStreamEventType.COMPLETED]
    assert events[0].completion is not None
    assert events[0].completion.finish_reason == ModelFinishReason.TOOL_CALLS
    assert events[0].completion.raw_finish_reason == "function_call"


@pytest.mark.anyio
async def test_chat_completions_stream_events_requires_finish_reason() -> None:
    async def raw_events():
        yield _text_chunk("partial")

    with pytest.raises(ChatCompletionsProtocolError, match="finish_reason"):
        [event async for event in chat_completions_stream_events(raw_events())]


@pytest.mark.anyio
async def test_chat_completions_stream_events_tolerates_repeated_finish_reason() -> None:
    async def raw_events():
        yield _text_chunk("hello")
        yield _finish_chunk("stop")
        yield _finish_chunk("stop")

    events = [event async for event in chat_completions_stream_events(raw_events())]

    assert [event.type for event in events] == [
        ModelStreamEventType.TEXT_DELTA,
        ModelStreamEventType.COMPLETED,
    ]
    assert events[0].delta == "hello"
    assert events[1].completion is not None
    assert events[1].completion.finish_reason == ModelFinishReason.STOP


@pytest.mark.anyio
async def test_chat_stream_preserves_completion_before_real_cancellation() -> None:
    tail_started = asyncio.Event()
    collected: list[Any] = []

    async def raw_events():
        yield _text_chunk("hello")
        yield _finish_chunk("stop")
        yield _usage_chunk({"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3})
        tail_started.set()
        await asyncio.Event().wait()

    async def collect() -> None:
        async for event in chat_completions_stream_events(raw_events()):
            collected.append(event)

    task = asyncio.create_task(collect())
    await asyncio.wait_for(tail_started.wait(), timeout=1)
    task.cancel("cancel after finish reason")
    assert task.cancelling() == 1

    with pytest.raises(asyncio.CancelledError, match="cancel after finish reason"):
        await task

    assert task.cancelled()
    assert task.cancelling() == 1
    assert [event.type for event in collected] == [
        ModelStreamEventType.TEXT_DELTA,
        ModelStreamEventType.COMPLETED,
    ]
    completion = collected[-1]
    assert completion.payload["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 1,
        "total_tokens": 3,
    }


@pytest.mark.anyio
async def test_chat_completions_stream_events_rejects_conflicting_finish_reasons() -> None:
    async def raw_events():
        yield _finish_chunk("stop")
        yield _finish_chunk("tool_calls")

    with pytest.raises(ChatCompletionsProtocolError, match="conflicting finish_reason"):
        [event async for event in chat_completions_stream_events(raw_events())]


@pytest.mark.anyio
async def test_chat_completions_stream_events_raises_on_mid_stream_error_chunk() -> None:
    # A fault reported after the stream opens arrives as a chunk carrying an
    # ``error`` object with no ``choices``; it must surface as the real API error,
    # not the misleading "ended before a finish_reason" protocol error.
    async def raw_events():
        yield _text_chunk("partial")
        yield {
            "error": {
                "message": "upstream connection reset",
                "type": "server_error",
                "code": "internal_error",
                "request_id": "req_stream_error",
            }
        }

    with pytest.raises(ChatCompletionsAPIError) as exc_info:
        [event async for event in chat_completions_stream_events(raw_events())]

    assert str(exc_info.value) == (
        "Chat Completions stream reported an error: [provider response body omitted]"
    )
    assert exc_info.value.error_type == "server_error"
    assert exc_info.value.error_code == "internal_error"
    assert exc_info.value.request_id == "req_stream_error"
    assert exc_info.value.status_code == 500
    assert exc_info.value.retryable is True
    assert exc_info.value.response_body is None


@pytest.mark.anyio
async def test_openrouter_mid_stream_error_uses_nested_typed_identity_and_numeric_status() -> None:
    async def raw_events():
        yield {
            "request_id": "gen-openrouter-error",
            "error": {
                "code": 429,
                "message": "upstream detail must not surface",
                "metadata": {
                    "error_type": "rate_limit_exceeded",
                    "provider_code": "rate_limited",
                },
            },
            "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
        }

    with pytest.raises(ChatCompletionsAPIError) as exc_info:
        [event async for event in chat_completions_stream_events(raw_events())]

    assert exc_info.value.status_code == 429
    assert exc_info.value.error_type == "rate_limit_exceeded"
    assert exc_info.value.error_code == "rate_limited"
    assert exc_info.value.request_id == "gen-openrouter-error"
    assert exc_info.value.retryable is True
    assert "upstream detail" not in str(exc_info.value)


@pytest.mark.anyio
async def test_openrouter_mid_stream_context_error_preserves_overflow_classification() -> None:
    async def raw_events():
        yield {
            "id": "gen-openrouter-overflow",
            "model": "vendor/reasoning-model",
            "provider": "Upstream",
            "error": {
                "code": 400,
                "message": "request exceeded context window",
                "metadata": {"error_type": "context_length_exceeded"},
            },
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": ""},
                    "finish_reason": "error",
                }
            ],
        }

    with pytest.raises(ChatCompletionsContextOverflowError) as exc_info:
        [event async for event in chat_completions_stream_events(raw_events())]

    assert exc_info.value.status_code == 400
    assert exc_info.value.error_type == "context_length_exceeded"
    assert exc_info.value.retryable is False
    assert "context window" not in str(exc_info.value)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("nested_type", "code", "message"),
    [
        pytest.param("rate_limited", 400, "context too large", id="context-message"),
        pytest.param("rate_limit_error", 429, "denied", id="retryable-type"),
    ],
)
async def test_openrouter_mid_stream_conflicting_type_aliases_fail_closed(
    nested_type: str,
    code: int,
    message: str,
) -> None:
    async def raw_events():
        yield {
            "error": {
                "type": "authentication_error",
                "code": code,
                "message": message,
                "metadata": {"error_type": nested_type},
            }
        }

    with pytest.raises(ChatCompletionsAPIError) as exc_info:
        [event async for event in chat_completions_stream_events(raw_events())]

    assert not isinstance(exc_info.value, ChatCompletionsContextOverflowError)
    assert exc_info.value.error_type is None
    assert exc_info.value.retryable is False


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("error_type", "error_code", "expected_status", "expected_retryable"),
    [
        pytest.param("rate_limit_error", "rate_limit_exceeded", 429, True, id="rate-limit"),
        pytest.param("permission_error", None, 403, False, id="permission"),
        pytest.param("future_error", "future_code", None, None, id="unknown"),
        pytest.param("permission_error", "internal_error", None, False, id="conflict"),
    ],
)
async def test_chat_stream_error_classifies_only_consistent_known_identity(
    error_type: str,
    error_code: str | None,
    expected_status: int | None,
    expected_retryable: bool | None,
) -> None:
    async def raw_events():
        yield {
            "error": {
                "type": error_type,
                "code": error_code,
                "message": "server_error rate_limit_error permission_error",
            }
        }

    with pytest.raises(ChatCompletionsAPIError) as exc_info:
        [event async for event in chat_completions_stream_events(raw_events())]

    assert exc_info.value.status_code == expected_status
    assert exc_info.value.retryable is expected_retryable


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("error_type", "error_code"),
    [
        pytest.param("server_error", None, id="server-type"),
        pytest.param(None, "internal_error", id="internal-code"),
    ],
)
async def test_chat_context_message_does_not_override_transient_identity(
    error_type: str | None,
    error_code: str | None,
) -> None:
    async def raw_events():
        yield {
            "error": {
                "type": error_type,
                "code": error_code,
                "message": "This model's maximum context length was exceeded.",
            }
        }

    with pytest.raises(ChatCompletionsAPIError) as exc_info:
        [event async for event in chat_completions_stream_events(raw_events())]

    assert not isinstance(exc_info.value, ChatCompletionsContextOverflowError)
    assert exc_info.value.status_code == 500
    assert exc_info.value.retryable is True


@pytest.mark.anyio
@pytest.mark.parametrize(
    "secret_start",
    [
        MAX_PROVIDER_ERROR_BODY_CHARS - len("workload-secret-canary-ABCDEFGHIJKLMNOP"),
        MAX_PROVIDER_ERROR_BODY_CHARS - 10,
        MAX_PROVIDER_ERROR_BODY_CHARS,
    ],
    ids=("ends-at-old-bound", "crosses-old-bound", "starts-at-old-bound"),
)
async def test_chat_completions_stream_error_omits_raw_message_at_old_cutoff(
    secret_start: int,
) -> None:
    secret = "workload-secret-canary-ABCDEFGHIJKLMNOP"
    raw_event = {
        "error": {
            "message": "x" * secret_start + secret,
            "type": "server_error",
            "code": "internal_error",
            "request_id": "req_cutoff",
        }
    }

    async def raw_events():
        yield raw_event

    with pytest.raises(ChatCompletionsAPIError) as exc_info:
        [event async for event in chat_completions_stream_events(raw_events())]

    rendered = repr((str(exc_info.value), vars(exc_info.value)))
    assert str(exc_info.value) == (
        "Chat Completions stream reported an error: [provider response body omitted]"
    )
    assert not any(secret[:size] in rendered for size in range(8, len(secret) + 1))
    assert exc_info.value.error_type == "server_error"
    assert exc_info.value.error_code == "internal_error"
    assert exc_info.value.request_id == "req_cutoff"
    assert_cayu_traceback_does_not_retain(exc_info.value, raw_event)


@pytest.mark.anyio
async def test_chat_completions_stream_error_preserves_trusted_retry_after() -> None:
    raw_event = _TrustedSseJsonEvent(
        {
            "error": {
                "type": "server_error",
                "code": "server_error",
                "message": "temporary provider failure",
            }
        },
        retry_after_s=6.0,
    )

    async def raw_events():
        yield raw_event

    with pytest.raises(ChatCompletionsAPIError) as exc_info:
        [event async for event in chat_completions_stream_events(raw_events())]

    assert exc_info.value.status_code == 500
    assert exc_info.value.retryable is True
    assert exc_info.value.retry_after_s == 6.0


@pytest.mark.anyio
async def test_chat_completions_provider_stream_omits_cutoff_secret_fragment() -> None:
    secret = "workload-secret-canary-ABCDEFGHIJKLMNOP"
    transport = RecordingTransport(
        stream_events=[
            [
                {
                    "error": {
                        "message": "x" * (MAX_PROVIDER_ERROR_BODY_CHARS - 10) + secret,
                        "type": "server_error",
                        "code": "internal_error",
                        "request_id": "req_provider_cutoff",
                    }
                }
            ]
        ]
    )
    provider = ChatCompletionsProvider(api_key="test-key", transport=transport)
    request = ModelRequest(model="test-model", messages=[Message.text("user", "hello")])

    events = [event async for event in provider.stream(request)]

    rendered = repr([event.model_dump(mode="json") for event in events])
    assert [event.type for event in events] == [ModelStreamEventType.ERROR]
    assert not any(secret[:size] in rendered for size in range(8, len(secret) + 1))
    assert events[0].payload["provider_error_type"] == "server_error"
    assert events[0].payload["provider_error_code"] == "internal_error"
    assert "request_id" not in events[0].payload


@pytest.mark.anyio
async def test_chat_completions_stream_events_mid_stream_error_context_overflow() -> None:
    secret = "workload-secret-canary-ABCDEFGHIJKLMNOP"

    async def raw_events():
        yield {
            "error": {
                "message": (
                    "x" * (MAX_PROVIDER_ERROR_BODY_CHARS + 1)
                    + " This model's maximum context length is 8192 tokens. "
                    + secret
                ),
                "code": "context_length_exceeded",
            }
        }

    with pytest.raises(ChatCompletionsContextOverflowError) as exc_info:
        [event async for event in chat_completions_stream_events(raw_events())]

    assert str(exc_info.value) == (
        "Chat Completions stream reported an error: [provider response body omitted]"
    )
    assert secret not in repr((str(exc_info.value), vars(exc_info.value)))
    assert exc_info.value.error_code == "context_length_exceeded"
    assert exc_info.value.response_body is None


@pytest.mark.anyio
async def test_chat_completions_provider_rejects_conflicting_context_identity() -> None:
    transport = RecordingTransport(
        stream_events=[
            [
                {
                    "error": {
                        "type": "authentication_error",
                        "code": "context_length_exceeded",
                        "message": "This model's maximum context length was exceeded.",
                    }
                }
            ]
        ]
    )
    provider = ChatCompletionsProvider(api_key="test-key", transport=transport)

    events = [
        event
        async for event in provider.stream(
            ModelRequest(model="test-model", messages=[Message.text("user", "hello")])
        )
    ]

    assert len(transport.calls) == 1
    assert [event.type for event in events] == [ModelStreamEventType.ERROR]
    assert events[0].payload["retryable"] is False
    assert events[0].payload["provider_error_type"] == "authentication_error"
    assert events[0].payload["provider_error_code"] == "context_length_exceeded"


@pytest.mark.anyio
async def test_runtime_retries_chat_transient_identity_instead_of_context_recovery() -> None:
    transport = RecordingTransport(
        stream_events=[
            [
                {
                    "error": {
                        "type": "server_error",
                        "code": "internal_error",
                        "message": "This model's maximum context length was exceeded.",
                    }
                }
            ],
            [_text_chunk("recovered"), _finish_chunk("stop")],
        ]
    )
    provider = ChatCompletionsProvider(api_key="test-key", transport=transport)
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="test-model"),
        context_overflow_policy=RecentTurnsContextPolicy(max_user_turns=1),
    )

    events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "hello")],
                max_steps=1,
                retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
            )
        )
    ]

    event_types = [event.type for event in events]
    assert len(transport.calls) == 2
    assert EventType.MODEL_RETRY in event_types
    retry_event = next(event for event in events if event.type == EventType.MODEL_RETRY)
    assert retry_event.payload["status_code"] == 500
    assert retry_event.payload["reason"] == "http_status"
    assert EventType.CONTEXT_OVERFLOW_DETECTED not in event_types
    assert EventType.CONTEXT_OVERFLOW_RECOVERING not in event_types
    assert events[-1].type == EventType.SESSION_COMPLETED


@pytest.mark.anyio
async def test_chat_completions_stream_events_keyless_named_calls_do_not_merge() -> None:
    # Two keyless fragments (no ``index``, no ``id``) that each name a function
    # are distinct calls and must not collapse into one accumulator slot.
    async def raw_events():
        yield {
            "id": "chatcmpl-1",
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"id": "call_a", "function": {"name": "alpha", "arguments": "{}"}}
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"id": "call_b", "function": {"name": "beta", "arguments": "{}"}}
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

    events = [event async for event in chat_completions_stream_events(raw_events())]

    assert [event.type for event in events] == [
        ModelStreamEventType.TOOL_CALL,
        ModelStreamEventType.TOOL_CALL,
        ModelStreamEventType.COMPLETED,
    ]
    assert events[0].payload["name"] == "alpha"
    assert events[1].payload["name"] == "beta"


@pytest.mark.anyio
async def test_chat_completions_stream_events_keyless_argument_continuation_merges() -> None:
    # A keyless fragment that names no function continues the most recent call,
    # so streamed argument chunks still accumulate into a single tool call.
    async def raw_events():
        yield {
            "id": "chatcmpl-1",
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"id": "call_a", "function": {"name": "echo", "arguments": '{"te'}}
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
        yield {
            "choices": [
                {
                    "delta": {"tool_calls": [{"function": {"arguments": 'xt":"hi"}'}}]},
                    "finish_reason": "tool_calls",
                }
            ]
        }

    events = [event async for event in chat_completions_stream_events(raw_events())]

    assert [event.type for event in events] == [
        ModelStreamEventType.TOOL_CALL,
        ModelStreamEventType.COMPLETED,
    ]
    assert events[0].payload == {"id": "call_a", "name": "echo", "arguments": {"text": "hi"}}


@pytest.mark.anyio
async def test_provider_emits_nonblank_error_for_blank_exception() -> None:
    provider = ChatCompletionsProvider(
        api_key="test-key", name="gemini", transport=BlankFailingTransport()
    )
    request = ModelRequest(model="gemini-test", messages=[Message.text("user", "Hi.")])

    events = [event async for event in provider.stream(request)]

    assert len(events) == 1
    assert events[0].type == ModelStreamEventType.ERROR
    assert events[0].payload["error"].strip()


@pytest.mark.anyio
async def test_provider_stream_propagates_context_overflow() -> None:
    overflow = ChatCompletionsContextOverflowError(
        "Chat Completions model context overflow",
        status_code=400,
        error_code="context_length_exceeded",
    )

    class OverflowTransport:
        async def stream_chat_completions(
            self,
            *,
            url: str,
            headers: Mapping[str, str],
            payload: Mapping[str, Any],
            timeout_s: float,
            transport_idle_timeout_s: float,
            protocol_idle_timeout_s: float,
            semantic_progress_timeout_s: float,
            absolute_stream_timeout_s: float,
        ):
            raise overflow
            yield {}

    provider = ChatCompletionsProvider(api_key="test-key", transport=OverflowTransport())
    request = ModelRequest(model="gpt-test", messages=[Message.text("user", "Hi.")])

    with pytest.raises(ChatCompletionsContextOverflowError) as exc_info:
        [event async for event in provider.stream(request)]

    # Overflow escapes as a fresh detached typed exception so runtime recovery
    # can retry without retaining transport traceback locals containing headers.
    assert exc_info.value is not overflow
    assert isinstance(exc_info.value, ModelContextOverflowError)
    assert exc_info.value.status_code == 400
    assert exc_info.value.error_code == "context_length_exceeded"
    assert exc_info.value.retryable is False


def test_provider_rejects_invalid_tool_names() -> None:
    provider = ChatCompletionsProvider(
        api_key="test-key", name="gemini", transport=RecordingTransport()
    )
    request = ModelRequest(
        model="gemini-test",
        messages=[Message.text("user", "Hi.")],
        tools=[{"name": "bad name!", "input_schema": {"type": "object"}}],
    )

    with pytest.raises(ValueError, match="tool names"):
        build_chat_completions_payload(request, options_key=provider.name)


def test_provider_rejects_protected_extra_headers() -> None:
    with pytest.raises(ValueError, match="authorization"):
        ChatCompletionsProvider(
            api_key="test-key",
            name="gemini",
            extra_headers={"authorization": "Bearer override"},
        )


def test_provider_rejects_extra_header_overriding_custom_auth_header() -> None:
    # The protected set tracks the configured auth header, not just "authorization".
    with pytest.raises(ValueError, match="api-key"):
        ChatCompletionsProvider(
            api_key="test-key",
            name="azure",
            auth_header="api-key",
            auth_value_prefix="",
            extra_headers={"api-key": "override"},
        )


def test_provider_requires_https_base_url() -> None:
    with pytest.raises(ValueError, match="https"):
        ChatCompletionsProvider(
            api_key="test-key",
            name="gemini",
            base_url="http://insecure.example.com",
        )


def test_provider_declares_openai_usage_dialect_for_renamed_gateway() -> None:
    provider = ChatCompletionsProvider(
        api_key="test-key",
        name="gemini-through-gateway",
        base_url="https://gateway.example.com/openai/v1",
    )

    assert provider.usage_dialect is UsageDialect.OPENAI


def test_provider_preserves_subclass_usage_dialect_declaration() -> None:
    class GeminiGatewayProvider(ChatCompletionsProvider):
        usage_dialect = UsageDialect.GEMINI

    provider = GeminiGatewayProvider(
        api_key="test-key",
        name="gemini-through-gateway",
        base_url="https://gateway.example.com/openai/v1",
    )

    assert provider.usage_dialect is UsageDialect.GEMINI


def test_explicit_usage_dialect_overrides_subclass_and_endpoint_detection() -> None:
    class GeminiGatewayProvider(ChatCompletionsProvider):
        usage_dialect = UsageDialect.GEMINI

    provider = GeminiGatewayProvider(
        api_key="test-key",
        name="explicit-openai",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        usage_dialect="openai",
    )

    assert provider.usage_dialect is UsageDialect.OPENAI


def test_explicit_usage_dialect_is_validated() -> None:
    with pytest.raises(ValueError, match="usage_dialect must be one of"):
        ChatCompletionsProvider(
            api_key="test-key",
            usage_dialect="unknown",
        )


def test_provider_declares_gemini_usage_dialect_for_google_endpoint() -> None:
    provider = ChatCompletionsProvider(
        api_key="test-key",
        name="google-ai",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    )

    assert provider.usage_dialect is UsageDialect.GEMINI


@pytest.mark.parametrize(
    "base_url",
    [
        "https://aiplatform.googleapis.com/v1beta1/openapi",
        "https://us-central1-aiplatform.googleapis.com/v1beta1/openapi",
        "https://aiplatform.us.rep.googleapis.com/v1beta1/openapi",
    ],
)
def test_vertex_endpoints_require_explicit_gemini_usage_dialect(
    base_url: str,
) -> None:
    default_provider = ChatCompletionsProvider(
        api_key="test-key",
        name="vertex-gemini",
        base_url=base_url,
    )
    gemini_provider = ChatCompletionsProvider(
        api_key="test-key",
        name="vertex-gemini",
        base_url=base_url,
        usage_dialect=UsageDialect.GEMINI,
    )

    assert default_provider.usage_dialect is UsageDialect.OPENAI
    assert gemini_provider.usage_dialect is UsageDialect.GEMINI


def test_provider_endpoint_includes_api_version() -> None:
    provider = ChatCompletionsProvider(
        api_key="test-key",
        name="azure",
        base_url="https://example.openai.azure.com/openai/deployments/gpt-4o",
        api_version="2024-10-21",
    )

    assert provider._endpoint() == (
        "https://example.openai.azure.com/openai/deployments/gpt-4o"
        "/chat/completions?api-version=2024-10-21"
    )


def test_provider_endpoint_appends_chat_completions_to_base_url() -> None:
    # OpenAI-SDK convention: base_url carries the version path; the provider
    # appends only "/chat/completions" (no extra "/v1").
    openai = ChatCompletionsProvider(api_key="k", name="openai_chat")
    assert openai._endpoint() == "https://api.openai.com/v1/chat/completions"

    gemini = ChatCompletionsProvider(
        api_key="k",
        name="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    )
    assert gemini._endpoint() == (
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    )

    together = ChatCompletionsProvider(
        api_key="k", name="together", base_url="https://api.together.xyz/v1"
    )
    assert together._endpoint() == "https://api.together.xyz/v1/chat/completions"

    # A trailing slash on base_url must not produce a double slash.
    trailing = ChatCompletionsProvider(
        api_key="k", name="together", base_url="https://api.together.xyz/v1/"
    )
    assert trailing._endpoint() == "https://api.together.xyz/v1/chat/completions"


def test_provider_endpoint_preserves_base_query_and_fragment() -> None:
    provider = ChatCompletionsProvider(
        api_key="k",
        name="gateway",
        base_url="https://gateway.example/v1/?tenant=probe&return=%2F#route/",
        api_version="2024-10-21",
    )

    assert provider._endpoint() == (
        "https://gateway.example/v1/chat/completions?tenant=probe&return=%2F&"
        "api-version=2024-10-21#route/"
    )


@pytest.mark.parametrize(
    "endpoint_url",
    [
        "https://gateway.internal/proxy/chat",
        "https://gateway.internal/proxy/chat?",
        "https://gateway.internal/proxy/chat#",
    ],
)
def test_provider_endpoint_url_override_is_used_verbatim(endpoint_url: str) -> None:
    provider = ChatCompletionsProvider(
        api_key="k",
        name="custom",
        endpoint_url=endpoint_url,
    )
    assert provider._endpoint() == endpoint_url


@pytest.mark.anyio
@pytest.mark.parametrize(
    "endpoint_url",
    [
        "https://gateway.internal/proxy/chat?",
        "https://gateway.internal/proxy/chat#",
    ],
)
async def test_provider_passes_endpoint_override_to_transport_verbatim(
    endpoint_url: str,
) -> None:
    transport = RecordingTransport(stream_events=[[_finish_chunk("stop")]])
    provider = ChatCompletionsProvider(
        api_key="k",
        name="custom",
        endpoint_url=endpoint_url,
        transport=transport,
        stream_include_usage=False,
    )
    request = ModelRequest(
        model="m",
        messages=[Message.text("user", "Hi.")],
    )

    events = [event async for event in provider.stream(request)]

    assert [event.type for event in events] == [ModelStreamEventType.COMPLETED]
    assert len(transport.calls) == 1
    assert transport.calls[0]["url"] == endpoint_url


def test_provider_endpoint_url_override_accepts_api_version() -> None:
    versioned = ChatCompletionsProvider(
        api_key="k",
        name="custom",
        endpoint_url="https://gateway.internal/proxy/chat",
        api_version="2024-10-21",
    )
    assert versioned._endpoint() == "https://gateway.internal/proxy/chat?api-version=2024-10-21"


def test_provider_endpoint_url_with_query_uses_ampersand_for_api_version() -> None:
    # endpoint_url already carries a query string, so api-version must be joined
    # with "&", not a second "?".
    provider = ChatCompletionsProvider(
        api_key="k",
        name="custom",
        endpoint_url="https://gateway.internal/proxy/chat?tenant=acme",
        api_version="2024-10-21",
    )
    assert provider._endpoint() == (
        "https://gateway.internal/proxy/chat?tenant=acme&api-version=2024-10-21"
    )


def test_provider_endpoint_api_version_replaces_existing_values_once() -> None:
    provider = ChatCompletionsProvider(
        api_key="k",
        name="custom",
        endpoint_url=(
            "https://gateway.internal/proxy/chat?api%2Dversion=old&tenant=&route=a&"
            "signature=a%20b&api-version=older&route=b#regional"
        ),
        api_version="2024-10-21",
    )

    assert provider._endpoint() == (
        "https://gateway.internal/proxy/chat?tenant=&route=a&signature=a%20b&route=b&"
        "api-version=2024-10-21#regional"
    )


def test_provider_endpoint_preserves_url_api_version_without_override() -> None:
    provider = ChatCompletionsProvider(
        api_key="k",
        name="gateway",
        base_url="https://gateway.internal/v1?tenant=acme&api-version=url-version#regional",
    )

    assert provider._endpoint() == (
        "https://gateway.internal/v1/chat/completions?tenant=acme&api-version=url-version#regional"
    )


def test_provider_endpoint_url_scheme_respects_allow_http() -> None:
    with pytest.raises(ValueError, match="https"):
        ChatCompletionsProvider(
            api_key="k", name="local", endpoint_url="http://localhost:8000/v1/chat"
        )

    provider = ChatCompletionsProvider(
        api_key="k",
        name="local",
        endpoint_url="http://localhost:8000/v1/chat",
        allow_http=True,
    )
    assert provider._endpoint() == "http://localhost:8000/v1/chat"


def test_provider_supports_azure_api_key_auth() -> None:
    provider = ChatCompletionsProvider(
        api_key="secret",
        name="azure",
        base_url="https://example.openai.azure.com/openai/deployments/gpt-4o",
        auth_header="api-key",
        auth_value_prefix="",
        api_version="2024-10-21",
    )
    headers = provider._headers()
    assert headers["api-key"] == "secret"
    assert "authorization" not in {key.lower() for key in headers}


def test_provider_allows_http_only_when_opted_in() -> None:
    with pytest.raises(ValueError, match="https"):
        ChatCompletionsProvider(api_key="k", name="ollama", base_url="http://localhost:11434/v1")

    provider = ChatCompletionsProvider(
        api_key="k",
        name="ollama",
        base_url="http://localhost:11434/v1",
        allow_http=True,
    )
    assert provider._endpoint() == "http://localhost:11434/v1/chat/completions"
    # allow_http must reach the default transport, or the request itself (which
    # re-validates the URL) would reject the http endpoint at send time.
    assert provider.transport.allow_http is True


@pytest.mark.anyio
async def test_transport_rejects_http_unless_opted_in() -> None:
    https_only = HttpxChatCompletionsTransport()
    stream = https_only.stream_chat_completions(
        url="http://localhost:11434/v1/chat/completions",
        headers={},
        payload={},
        timeout_s=1.0,
        transport_idle_timeout_s=1.0,
        protocol_idle_timeout_s=1.0,
        semantic_progress_timeout_s=1.0,
        absolute_stream_timeout_s=1.0,
    )
    with pytest.raises(ValueError, match="https"):
        await stream.__anext__()

    # With allow_http the scheme check passes (the URL is accepted before any
    # network call), so an http endpoint is permitted.
    assert HttpxChatCompletionsTransport(allow_http=True).allow_http is True


@pytest.mark.anyio
async def test_chat_completions_transport_classifies_gemini_context_too_long(
    monkeypatch,
) -> None:
    class ResponseContext:
        async def __aenter__(self) -> httpx.Response:
            request = httpx.Request("POST", "https://example.test/v1/chat/completions")
            return httpx.Response(
                500,
                request=request,
                headers={"content-type": "application/json"},
                json={
                    "error": {
                        "code": 500,
                        "status": "INTERNAL",
                        "message": "The input context is too long.",
                    }
                },
            )

        async def __aexit__(self, *args: Any) -> None:
            return None

    class FailingClient:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> FailingClient:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        def stream(
            self,
            method: str,
            url: str,
            *,
            headers: dict[str, str],
            json: dict[str, Any],
            timeout: Any = None,
        ) -> ResponseContext:
            return ResponseContext()

    monkeypatch.setattr(
        "cayu.providers._http.httpx.AsyncClient",
        FailingClient,
    )

    stream = HttpxChatCompletionsTransport().stream_chat_completions(
        url="https://example.test/v1/chat/completions",
        headers={},
        payload={},
        timeout_s=1,
        transport_idle_timeout_s=1,
        protocol_idle_timeout_s=1,
        semantic_progress_timeout_s=1,
        absolute_stream_timeout_s=1,
    )
    with pytest.raises(ChatCompletionsContextOverflowError) as exc_info:
        await stream.__anext__()

    assert exc_info.value.provider == "chat_completions"
    assert isinstance(exc_info.value, ModelContextOverflowError)
    assert isinstance(exc_info.value, ChatCompletionsAPIError)
    assert exc_info.value.status_code == 500
    assert exc_info.value.error_type == "INTERNAL"


@pytest.mark.parametrize(
    ("status_code", "error_type", "error_code", "expected_retryable"),
    [
        pytest.param(401, "authentication_error", None, False, id="permanent"),
        pytest.param(500, "server_error", "server_error", True, id="transient"),
        pytest.param(500, "future_error", "future_code", None, id="unknown"),
        pytest.param(500, "authentication_error", "server_error", False, id="body-conflict"),
        pytest.param(429, "server_error", "server_error", False, id="status-conflict"),
    ],
)
def test_chat_http_error_uses_stream_identity_classification(
    status_code: int,
    error_type: str,
    error_code: str | None,
    expected_retryable: bool | None,
) -> None:
    response = httpx.Response(
        status_code,
        headers={"content-type": "application/json", "retry-after": "4"},
        json={"error": {"type": error_type, "code": error_code}},
    )

    error = chat_completions_module._chat_api_error_from_response(
        response,
        "Chat Completions failed",
        4.0,
    )

    assert error.status_code == status_code
    assert error.error_type == error_type
    assert error.error_code == error_code
    assert error.retryable is expected_retryable
    assert error.retry_after_s == 4.0


def test_openrouter_http_error_uses_nested_typed_identity_and_provider_code() -> None:
    response = httpx.Response(
        502,
        headers={"content-type": "application/json"},
        json={
            "error": {
                "code": 502,
                "message": "provider disconnected",
                "metadata": {
                    "error_type": "provider_unavailable",
                    "provider_code": "connection_reset",
                },
            }
        },
    )

    error = chat_completions_module._chat_api_error_from_response(
        response,
        "Chat Completions failed",
        None,
    )

    assert error.status_code == 502
    assert error.error_type == "provider_unavailable"
    assert error.error_code == "connection_reset"
    assert error.retryable is True


@pytest.mark.anyio
async def test_chat_transport_rejects_conflicting_http_context_identity(monkeypatch) -> None:
    class ResponseContext:
        async def __aenter__(self) -> httpx.Response:
            request = httpx.Request("POST", "https://example.test/v1/chat/completions")
            return httpx.Response(
                401,
                request=request,
                json={
                    "error": {
                        "type": "authentication_error",
                        "code": "context_length_exceeded",
                        "message": "This model's maximum context length was exceeded.",
                    }
                },
            )

        async def __aexit__(self, *args: Any) -> None:
            return None

    class FailingClient:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs

        def stream(
            self,
            method: str,
            url: str,
            *,
            headers: dict[str, str],
            json: dict[str, Any],
            timeout: Any = None,
        ) -> ResponseContext:
            del method, url, headers, json, timeout
            return ResponseContext()

    monkeypatch.setattr("cayu.providers._http.httpx.AsyncClient", FailingClient)

    stream = HttpxChatCompletionsTransport().stream_chat_completions(
        url="https://example.test/v1/chat/completions",
        headers={},
        payload={},
        timeout_s=1,
        transport_idle_timeout_s=1,
        protocol_idle_timeout_s=1,
        semantic_progress_timeout_s=1,
        absolute_stream_timeout_s=1,
    )
    with pytest.raises(ChatCompletionsAPIError) as exc_info:
        await stream.__anext__()

    assert not isinstance(exc_info.value, ChatCompletionsContextOverflowError)
    assert exc_info.value.status_code == 401
    assert exc_info.value.retryable is False


@pytest.mark.anyio
async def test_chat_completions_transport_does_not_classify_quota_exhausted(
    monkeypatch,
) -> None:
    class ResponseContext:
        async def __aenter__(self) -> httpx.Response:
            request = httpx.Request("POST", "https://example.test/v1/chat/completions")
            return httpx.Response(
                429,
                request=request,
                headers={"content-type": "application/json", "retry-after": "5"},
                json={
                    "error": {
                        "code": 429,
                        "status": "RESOURCE_EXHAUSTED",
                        "message": "You exceeded your current quota.",
                    }
                },
            )

        async def __aexit__(self, *args: Any) -> None:
            return None

    class FailingClient:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> FailingClient:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        def stream(
            self,
            method: str,
            url: str,
            *,
            headers: dict[str, str],
            json: dict[str, Any],
            timeout: Any = None,
        ) -> ResponseContext:
            return ResponseContext()

    monkeypatch.setattr(
        "cayu.providers._http.httpx.AsyncClient",
        FailingClient,
    )

    stream = HttpxChatCompletionsTransport().stream_chat_completions(
        url="https://example.test/v1/chat/completions",
        headers={},
        payload={},
        timeout_s=1,
        transport_idle_timeout_s=1,
        protocol_idle_timeout_s=1,
        semantic_progress_timeout_s=1,
        absolute_stream_timeout_s=1,
    )
    with pytest.raises(ChatCompletionsAPIError) as exc_info:
        await stream.__anext__()

    assert exc_info.value.status_code == 429
    assert exc_info.value.retry_after_s == 5.0


def test_stream_options_is_configurable() -> None:
    request = ModelRequest(model="m", messages=[Message.text("user", "Hi.")])

    with_usage = build_chat_completions_payload(request, stream=True)
    assert with_usage["stream_options"] == {"include_usage": True}

    without_usage = build_chat_completions_payload(request, stream=True, include_usage=False)
    assert "stream_options" not in without_usage


@pytest.mark.anyio
async def test_stream_completes_without_usage_chunk() -> None:
    # When stream_include_usage is off, the server sends no usage chunk. The
    # provider must still emit a COMPLETED event (this is the real consequence of
    # the opt-out, beyond the payload omitting stream_options).
    transport = RecordingTransport(stream_events=[[_text_chunk("hi"), _finish_chunk("stop")]])
    provider = ChatCompletionsProvider(
        api_key="test-key", name="gemini", transport=transport, stream_include_usage=False
    )
    request = ModelRequest(model="gemini-test", messages=[Message.text("user", "Hi.")])

    events = [event async for event in provider.stream(request)]

    assert [event.type for event in events] == [
        ModelStreamEventType.TEXT_DELTA,
        ModelStreamEventType.COMPLETED,
    ]
    assert events[-1].completion is not None
    assert events[-1].completion.finish_reason == ModelFinishReason.STOP
    assert "stream_options" not in transport.calls[0]["payload"]


@pytest.mark.parametrize("n", [2, True, False, 1.5, "1", None, [], {}])
def test_payload_rejects_noncanonical_candidate_count(n: object) -> None:
    request = ModelRequest(
        model="m",
        messages=[Message.text("user", "Hi.")],
        options={"openai": {"n": n}},
    )
    with pytest.raises(ValueError, match="n must be 1"):
        build_chat_completions_payload(request, options_key="openai")


@pytest.mark.parametrize("n", [1, 1.0])
def test_payload_accepts_durable_canonical_single_candidate_count(n: object) -> None:
    request = ModelRequest(
        model="m",
        messages=[Message.text("user", "Hi.")],
        options={"openai": {"n": n}},
    )

    assert type(request.options["openai"]["n"]) is int
    payload = build_chat_completions_payload(request, options_key="openai")
    assert type(payload["n"]) is int
    assert payload["n"] == 1


@pytest.mark.parametrize("include_usage", [1, 0, "true", None, [], {}])
def test_payload_rejects_non_boolean_include_usage(include_usage: object) -> None:
    request = ModelRequest(model="m", messages=[Message.text("user", "Hi.")])

    with pytest.raises(TypeError, match=r"^include_usage must be a bool\.$"):
        build_chat_completions_payload(
            request,
            include_usage=include_usage,  # type: ignore[arg-type]
        )


@pytest.mark.anyio
async def test_sse_comment_heartbeats_do_not_refresh_protocol_progress() -> None:
    async def lines():
        # Each heartbeat arrives within one idle window (0.04 < 0.1), but their
        # cumulative time (0.2s) exceeds it — so the stream survives only if `:`
        # comment lines count as activity and refresh the idle timer.
        for _ in range(5):
            await asyncio.sleep(0.04)
            yield ":"
        yield 'data: {"ok": true}'
        yield ""

    from cayu.providers.deadlines import (
        ProviderDeadlineKind,
        ProviderStreamDeadlineController,
        ProviderStreamDeadlineExceeded,
        ProviderStreamDeadlines,
    )

    controller = ProviderStreamDeadlineController(
        ProviderStreamDeadlines(
            transport_idle_timeout_s=1,
            protocol_idle_timeout_s=0.1,
            semantic_progress_timeout_s=1,
            absolute_stream_timeout_s=1,
        )
    )
    with pytest.raises(ProviderStreamDeadlineExceeded) as captured:
        [
            event
            async for event in aiter_sse_json_events(
                lines(),
                deadline_controller=controller,
                provider_label="Chat Completions",
                protocol_error=ChatCompletionsProtocolError,
            )
        ]
    assert captured.value.evidence.deadline_kind is ProviderDeadlineKind.PROTOCOL_IDLE


def test_cayu_app_bounds_active_http_error_body_before_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StreamingBody(httpx.AsyncByteStream):
        def __init__(self, *, error_body: bool) -> None:
            self.error_body = error_body
            self.closed = False

        async def __aiter__(self):
            if self.error_body:
                while True:
                    await asyncio.sleep(0.001)
                    yield b"x"
            yield (
                b'data: {"id":"chat-1","object":"chat.completion.chunk",'
                b'"model":"fake-model","choices":[{"index":0,'
                b'"delta":{"content":"ok"},"finish_reason":null}]}\n\n'
            )
            yield (
                b'data: {"id":"chat-1","object":"chat.completion.chunk",'
                b'"model":"fake-model","choices":[{"index":0,'
                b'"delta":{},"finish_reason":"stop"}]}\n\n'
            )
            yield b"data: [DONE]\n\n"

        async def aclose(self) -> None:
            self.closed = True

    class StreamContext:
        def __init__(self, response: httpx.Response) -> None:
            self.response = response

        async def __aenter__(self) -> httpx.Response:
            return self.response

        async def __aexit__(self, *args: Any) -> None:
            await self.response.aclose()

    class BoundedErrorHttpClient:
        calls = 0
        bodies: list[StreamingBody] = []

        def __init__(self, **_kwargs: Any) -> None:
            self.is_closed = False

        def stream(self, method: str, url: str, **_kwargs: Any) -> StreamContext:
            type(self).calls += 1
            if type(self).calls == 2:
                assert type(self).bodies[0].closed
            body = StreamingBody(error_body=type(self).calls == 1)
            type(self).bodies.append(body)
            request = httpx.Request(method, url)
            return StreamContext(
                httpx.Response(
                    500 if body.error_body else 200,
                    headers={"content-type": "application/json"}
                    if body.error_body
                    else {"content-type": "text/event-stream"},
                    stream=body,
                    request=request,
                )
            )

        async def aclose(self) -> None:
            self.is_closed = True

    monkeypatch.setattr("cayu.providers._http.httpx.AsyncClient", BoundedErrorHttpClient)
    provider = ChatCompletionsProvider(
        api_key="test-key",
        base_url="https://example.test/v1",
        timeout_s=0.02,
        transport_idle_timeout_s=0.01,
        protocol_idle_timeout_s=1.0,
        semantic_progress_timeout_s=1.0,
        absolute_stream_timeout_s=1.0,
    )
    app = CayuApp(retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0))
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def run() -> list[Event]:
        return await asyncio.wait_for(
            _collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_bounded_active_http_error_body",
                    messages=[Message.text("user", "hi")],
                ),
            ),
            timeout=3.0,
        )

    events = asyncio.run(run())

    assert BoundedErrorHttpClient.calls == 2
    assert all(body.closed for body in BoundedErrorHttpClient.bodies)
    retry = next(event for event in events if event.type == EventType.MODEL_RETRY)
    assert retry.payload["status_code"] == 500
    assert events[-1].type == EventType.SESSION_COMPLETED


@pytest.mark.parametrize(
    "close_failure",
    [None, "delayed_success", "runtime_error", "child_cancellation"],
    ids=[
        "close_succeeds",
        "close_delays_retry",
        "close_runtime_error",
        "close_child_cancellation",
    ],
)
def test_cayu_app_resolves_in_band_error_stream_before_retry_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    close_failure: str | None,
) -> None:
    close_started = asyncio.Event()
    release_close = asyncio.Event()

    class StreamingBody(httpx.AsyncByteStream):
        def __init__(self, *, fail_in_band: bool) -> None:
            self.fail_in_band = fail_in_band
            self.closed = False

        async def __aiter__(self):
            if self.fail_in_band:
                yield (
                    b'data: {"error":{"type":"server_error",'
                    b'"code":"internal_error","message":"temporary failure"}}\n\n'
                )
                # This tail must never retain the connection after the parser
                # classifies the preceding in-band failure.
                await asyncio.Event().wait()
                yield b""  # pragma: no cover
            yield (
                b'data: {"id":"chat-1","object":"chat.completion.chunk",'
                b'"model":"fake-model","choices":[{"index":0,'
                b'"delta":{"content":"ok"},"finish_reason":null}]}\n\n'
            )
            yield (
                b'data: {"id":"chat-1","object":"chat.completion.chunk",'
                b'"model":"fake-model","choices":[{"index":0,'
                b'"delta":{},"finish_reason":"stop"}]}\n\n'
            )
            yield b"data: [DONE]\n\n"

        async def aclose(self) -> None:
            self.closed = True
            if self.fail_in_band and close_failure == "delayed_success":
                close_started.set()
                await release_close.wait()
            if self.fail_in_band and close_failure == "runtime_error":
                raise RuntimeError("provider byte stream close failed")
            if self.fail_in_band and close_failure == "child_cancellation":
                raise asyncio.CancelledError("provider child cleanup cancelled")

    class StreamContext:
        def __init__(self, response: httpx.Response) -> None:
            self.response = response

        async def __aenter__(self) -> httpx.Response:
            return self.response

        async def __aexit__(self, *args: Any) -> None:
            await self.response.aclose()

    class SequencedHttpClient:
        calls = 0
        bodies: list[StreamingBody] = []

        def __init__(self, **_kwargs: Any) -> None:
            self.is_closed = False

        def stream(self, *_args: Any, **_kwargs: Any) -> StreamContext:
            type(self).calls += 1
            if type(self).calls == 2:
                assert close_failure in {None, "delayed_success"}
                assert type(self).bodies[0].closed
            body = StreamingBody(fail_in_band=type(self).calls == 1)
            type(self).bodies.append(body)
            request = httpx.Request("POST", "https://example.test/v1/chat/completions")
            return StreamContext(
                httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    stream=body,
                    request=request,
                )
            )

        async def aclose(self) -> None:
            self.is_closed = True

    class RetainingTransport:
        def __init__(self) -> None:
            self.delegate = HttpxChatCompletionsTransport()
            self.streams: list[AsyncIterator[Mapping[str, Any]]] = []

        def stream_chat_completions(self, **kwargs: Any) -> AsyncIterator[Mapping[str, Any]]:
            stream = self.delegate.stream_chat_completions(**kwargs)
            # Keep an external reference so CPython async-generator finalization
            # cannot make an omitted explicit close accidentally pass this test.
            self.streams.append(stream)
            return stream

        async def aclose(self) -> None:
            await self.delegate.aclose()

    monkeypatch.setattr("cayu.providers._http.httpx.AsyncClient", SequencedHttpClient)
    transport = RetainingTransport()
    provider = ChatCompletionsProvider(
        api_key="test-key",
        base_url="https://example.test/v1",
        transport_idle_timeout_s=1.0,
        protocol_idle_timeout_s=1.0,
        semantic_progress_timeout_s=1.0,
        absolute_stream_timeout_s=1.0,
        transport=transport,
    )
    app = CayuApp(retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0))
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def run() -> tuple[list[Event], int, bool]:
        consumer = asyncio.create_task(
            _collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_in_band_stream_close_before_retry",
                    messages=[Message.text("user", "hi")],
                ),
            )
        )
        if close_failure == "delayed_success":
            await asyncio.wait_for(close_started.wait(), timeout=0.5)
            await asyncio.sleep(0)
            assert SequencedHttpClient.calls == 1
            release_close.set()
        events = await consumer
        return events, consumer.cancelling(), consumer.cancelled()

    events, consumer_cancelling, consumer_cancelled = asyncio.run(run())

    assert SequencedHttpClient.bodies[0].closed
    assert consumer_cancelling == 0
    assert not consumer_cancelled
    if close_failure in {"runtime_error", "child_cancellation"}:
        assert SequencedHttpClient.calls == 1
        assert len(transport.streams) == 1
        assert EventType.MODEL_RETRY not in {event.type for event in events}
        model_errors = [event for event in events if event.type == EventType.MODEL_ERROR]
        assert len(model_errors) == 1
        assert model_errors[0].payload["error_type"] == "ProviderStreamCleanupError"
        assert model_errors[0].payload["status_code"] == 500
        assert model_errors[0].payload["provider_error_type"] == "server_error"
        assert model_errors[0].payload["provider_error_code"] == "internal_error"
        assert model_errors[0].payload["stream_cleanup_failed"] is True
        assert model_errors[0].payload["retryable"] is False
        assert events[-1].type == EventType.SESSION_FAILED
        return

    assert SequencedHttpClient.calls == 2
    assert len(transport.streams) == 2
    retry = next(event for event in events if event.type == EventType.MODEL_RETRY)
    assert retry.payload["status_code"] == 500
    assert events[-1].type == EventType.SESSION_COMPLETED


@pytest.mark.parametrize("termination", ["done", "eof"], ids=["done_marker", "clean_eof"])
def test_cayu_app_preserves_chat_http_completion_before_response_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch,
    termination: str,
) -> None:
    class CompletingBody(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.closed = False
            self.close_started = asyncio.Event()
            self.close_finalized = asyncio.Event()

        async def __aiter__(self):
            yield (
                b'data: {"id":"chat-1","object":"chat.completion.chunk",'
                b'"model":"fake-model","choices":[{"index":0,'
                b'"delta":{"content":"ok"},"finish_reason":null}]}\n\n'
            )
            yield (
                b'data: {"id":"chat-1","object":"chat.completion.chunk",'
                b'"model":"fake-model","choices":[{"index":0,'
                b'"delta":{},"finish_reason":"stop"}]}\n\n'
            )
            yield (
                b'data: {"id":"chat-1","object":"chat.completion.chunk",'
                b'"model":"fake-model","choices":[],"usage":'
                b'{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}\n\n'
            )
            if termination == "done":
                yield b"data: [DONE]\n\n"

        async def aclose(self) -> None:
            self.closed = True
            self.close_started.set()
            try:
                raise httpx.CloseError("completed response close failed")
            finally:
                self.close_finalized.set()

    class StreamContext:
        def __init__(self, response: httpx.Response) -> None:
            self.response = response

        async def __aenter__(self) -> httpx.Response:
            return self.response

        async def __aexit__(self, *args: Any) -> None:
            await self.response.aclose()

    class CompletingHttpClient:
        calls = 0
        bodies: list[CompletingBody] = []

        def __init__(self, **_kwargs: Any) -> None:
            self.is_closed = False

        def stream(self, method: str, url: str, **_kwargs: Any) -> StreamContext:
            type(self).calls += 1
            body = CompletingBody()
            type(self).bodies.append(body)
            return StreamContext(
                httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    stream=body,
                    request=httpx.Request(method, url),
                )
            )

        async def aclose(self) -> None:
            self.is_closed = True

    monkeypatch.setattr("cayu.providers._http.httpx.AsyncClient", CompletingHttpClient)
    store = InMemorySessionStore()
    provider = ChatCompletionsProvider(
        api_key="test-key",
        base_url="https://example.test/v1",
        transport_idle_timeout_s=1.0,
        protocol_idle_timeout_s=1.0,
        semantic_progress_timeout_s=1.0,
        absolute_stream_timeout_s=1.0,
    )
    app = CayuApp(
        session_store=store,
        retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    session_id = f"sess_chat_http_completion_cleanup_{termination}"

    async def run() -> list[Event]:
        events = await _collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hi")],
            ),
        )
        await asyncio.wait_for(
            CompletingHttpClient.bodies[0].close_finalized.wait(),
            timeout=0.5,
        )
        return events

    events = asyncio.run(run())
    transcript = asyncio.run(store.load_transcript(session_id))
    durable_events = asyncio.run(store.load_events(session_id))

    assert CompletingHttpClient.calls == 1
    body = CompletingHttpClient.bodies[0]
    assert body.closed
    assert body.close_started.is_set()
    assert body.close_finalized.is_set()
    assert EventType.MODEL_RETRY not in {event.type for event in events}
    assert EventType.MODEL_ERROR not in {event.type for event in events}
    model_completions = [
        event for event in durable_events if event.type == EventType.MODEL_COMPLETED
    ]
    assert len(model_completions) == 1
    completion = model_completions[0]
    assert completion.payload["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 1,
        "total_tokens": 3,
    }
    assert completion.payload["usage_metrics"]["input_tokens"] == 2
    assert completion.payload["usage_metrics"]["output_tokens"] == 1
    assert completion.payload["usage_metrics"]["total_tokens"] == 3
    assert completion.payload["step_classification"]["type"] == "failed"
    assert events[-1].type == EventType.SESSION_FAILED
    assert [message.role for message in transcript] == ["user"]


@pytest.mark.anyio
@pytest.mark.parametrize("field", ["id", "model"])
async def test_chat_completions_stream_rejects_contradictory_response_identity(
    field: str,
) -> None:
    async def raw_events():
        yield {"id": "response-1", "model": "model-1", "choices": []}
        yield {
            "id": "response-2" if field == "id" else "response-1",
            "model": "model-2" if field == "model" else "model-1",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }

    with pytest.raises(ChatCompletionsProtocolError, match="conflicting"):
        [event async for event in chat_completions_stream_events(raw_events())]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "late_delta",
    [
        {"content": "late"},
        {"reasoning_content": "late"},
        {"tool_calls": [{"index": 0, "id": "late", "function": {"name": "exec"}}]},
    ],
)
async def test_chat_completions_stream_rejects_post_terminal_semantic_output(
    late_delta: dict[str, object],
) -> None:
    async def raw_events():
        yield {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        yield {"choices": [{"index": 0, "delta": late_delta, "finish_reason": None}]}

    with pytest.raises(ChatCompletionsProtocolError, match="after finish_reason"):
        [event async for event in chat_completions_stream_events(raw_events())]


@pytest.mark.anyio
async def test_chat_completions_stream_allows_identity_bound_usage_tail() -> None:
    async def raw_events():
        yield {
            "id": "response-1",
            "model": "model-1",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield {
            "id": "response-1",
            "model": "model-1",
            "choices": [],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }

    events = [event async for event in chat_completions_stream_events(raw_events())]
    assert events[-1].payload["usage"]["total_tokens"] == 3


@pytest.mark.parametrize("wrapped", [False, True], ids=["direct", "transparent-wrapper"])
def test_cayu_app_preserves_chat_http_completion_before_real_tail_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    wrapped: bool,
) -> None:
    body_created = asyncio.Event()

    class BlockingTailBody(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.closed = False
            self.tail_started = asyncio.Event()
            self.tail_cancelled = asyncio.Event()

        async def __aiter__(self):
            yield (
                b'data: {"id":"chat-1","object":"chat.completion.chunk",'
                b'"model":"fake-model","choices":[{"index":0,'
                b'"delta":{"content":"ok"},"finish_reason":null}]}\n\n'
            )
            yield (
                b'data: {"id":"chat-1","object":"chat.completion.chunk",'
                b'"model":"fake-model","choices":[{"index":0,'
                b'"delta":{},"finish_reason":"stop"}]}\n\n'
            )
            yield (
                b'data: {"id":"chat-1","object":"chat.completion.chunk",'
                b'"model":"fake-model","choices":[],"usage":'
                b'{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}\n\n'
            )
            self.tail_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.tail_cancelled.set()
                raise

        async def aclose(self) -> None:
            self.closed = True

    class StreamContext:
        def __init__(self, response: httpx.Response) -> None:
            self.response = response

        async def __aenter__(self) -> httpx.Response:
            return self.response

        async def __aexit__(self, *args: Any) -> None:
            await self.response.aclose()

    class BlockingHttpClient:
        calls = 0
        bodies: list[BlockingTailBody] = []

        def __init__(self, **_kwargs: Any) -> None:
            self.is_closed = False

        def stream(self, method: str, url: str, **_kwargs: Any) -> StreamContext:
            type(self).calls += 1
            body = BlockingTailBody()
            type(self).bodies.append(body)
            body_created.set()
            return StreamContext(
                httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    stream=body,
                    request=httpx.Request(method, url),
                )
            )

        async def aclose(self) -> None:
            self.is_closed = True

    monkeypatch.setattr("cayu.providers._http.httpx.AsyncClient", BlockingHttpClient)
    session_id = "sess_chat_http_completion_tail_cancel"
    store = InMemorySessionStore()
    app = CayuApp(
        session_store=store,
        retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
    )
    delegate = ChatCompletionsProvider(
        api_key="test-key",
        base_url="https://example.test/v1",
        transport=HttpxChatCompletionsTransport(),
        transport_idle_timeout_s=1.0,
        protocol_idle_timeout_s=1.0,
        semantic_progress_timeout_s=1.0,
        absolute_stream_timeout_s=1.0,
    )
    provider: ModelProvider = delegate
    if wrapped:

        class TransparentProvider(ModelProvider):
            def __init__(self, wrapped_provider: ModelProvider) -> None:
                self.delegate = wrapped_provider
                self.name = wrapped_provider.name
                self.billing_provider_name = wrapped_provider.billing_provider_name
                self.usage_dialect = wrapped_provider.usage_dialect
                self.supports_native_structured_output = (
                    wrapped_provider.supports_native_structured_output
                )

            @property
            def stream_deadlines(self):
                return self.delegate.stream_deadlines

            async def runtime_stream(
                self,
                request: ModelRequest,
            ) -> AsyncIterator[ModelStreamEvent]:
                async for event in self.delegate.runtime_stream(request):
                    yield event

            async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
                async for event in self.delegate.stream(request):
                    yield event

        provider = TransparentProvider(delegate)
    app.register_provider(provider, default=True)

    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def run() -> tuple[list[Event], list[Message], asyncio.Task[list[Event]]]:
        task = asyncio.create_task(
            _collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "hi")],
                ),
            )
        )
        await asyncio.wait_for(body_created.wait(), timeout=1)
        body = BlockingHttpClient.bodies[0]
        await asyncio.wait_for(body.tail_started.wait(), timeout=1)
        task.cancel("cancel after Chat Completions finish reason")
        assert task.cancelling() == 1
        with pytest.raises(
            asyncio.CancelledError,
            match="Provider operation cancelled",
        ):
            await task
        return await store.load_events(session_id), await store.load_transcript(session_id), task

    durable_events, transcript, task = asyncio.run(run())

    assert BlockingHttpClient.calls == 1
    body = BlockingHttpClient.bodies[0]
    assert body.tail_cancelled.is_set()
    assert body.closed
    assert task.cancelled()
    assert task.cancelling() == 0
    assert EventType.MODEL_RETRY not in {event.type for event in durable_events}
    completions = [event for event in durable_events if event.type == EventType.MODEL_COMPLETED]
    assert len(completions) == 1
    completion = completions[0]
    assert completion.payload["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 1,
        "total_tokens": 3,
    }
    assert completion.payload["usage_metrics"]["input_tokens"] == 2
    assert completion.payload["usage_metrics"]["output_tokens"] == 1
    assert completion.payload["usage_metrics"]["total_tokens"] == 3
    assert completion.payload["step_classification"]["type"] == "failed"
    assert [message.role for message in transcript] == ["user"]
