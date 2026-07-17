from __future__ import annotations

import asyncio
import base64
import threading
from collections.abc import Iterable
from concurrent.futures import TimeoutError as FutureTimeoutError
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

import cayu.providers.bedrock as bedrock_module
from cayu import (
    RESOLVED_FILE_ATTACHMENTS_OPTION,
    AgentSpec,
    CayuApp,
    EventType,
    FileAttachmentKind,
    Message,
    ModelPrice,
    PriceBook,
    RunRequest,
    StructuredOutputSpec,
    file_attachment,
)
from cayu.core.messages import FilePart, ProviderStatePart, TextPart, ThinkingPart, ToolCallPart
from cayu.providers import (
    BedrockAPIError,
    BedrockContextOverflowError,
    BedrockProvider,
    ModelRequest,
    ModelStreamEvent,
    ModelStreamEventType,
    bedrock_billing_identity,
)
from cayu.runtime.structured_output import STRUCTURED_OUTPUT_TOOL_NAME


class FakeBedrockClient:
    def __init__(self, events: Iterable[dict[str, Any]]) -> None:
        self.events = list(events)
        self.converse_calls: list[dict[str, Any]] = []
        self.count_calls: list[dict[str, Any]] = []
        self.closed = False
        self.meta = SimpleNamespace(region_name="us-east-1")

    def converse_stream(self, **kwargs: Any) -> dict[str, Any]:
        self.converse_calls.append(kwargs)
        return {"stream": iter(self.events)}

    def count_tokens(self, **kwargs: Any) -> dict[str, Any]:
        self.count_calls.append(kwargs)
        return {"inputTokens": 17}

    def close(self) -> None:
        self.closed = True


class FakeClientError(RuntimeError):
    def __init__(
        self,
        *,
        code: str,
        message: str,
        status: int,
        request_id: str,
        retry_after: str | None = None,
    ) -> None:
        super().__init__(message)
        headers = {} if retry_after is None else {"retry-after": retry_after}
        self.response = {
            "Error": {"Code": code, "Message": message},
            "ResponseMetadata": {
                "HTTPStatusCode": status,
                "RequestId": request_id,
                "HTTPHeaders": headers,
            },
        }


class FailingBedrockClient:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def converse_stream(self, **kwargs: Any) -> dict[str, Any]:
        raise self.error

    def count_tokens(self, **kwargs: Any) -> dict[str, Any]:
        raise self.error


class BlockingBedrockStream:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.released = threading.Event()
        self.closed = False

    def __iter__(self) -> BlockingBedrockStream:
        return self

    def __next__(self) -> dict[str, Any]:
        self.started.set()
        self.released.wait(timeout=5)
        raise StopIteration

    def close(self) -> None:
        self.closed = True
        self.released.set()


class BlockingBedrockClient:
    def __init__(self, stream: BlockingBedrockStream) -> None:
        self.stream = stream

    def converse_stream(self, **kwargs: Any) -> dict[str, Any]:
        return {"stream": self.stream}


class BlockingCloseBedrockStream(BlockingBedrockStream):
    def __init__(self) -> None:
        super().__init__()
        self.close_started = threading.Event()
        self.close_released = threading.Event()

    def close(self) -> None:
        self.closed = True
        self.close_started.set()
        self.close_released.wait(timeout=5)


def collect(provider: BedrockProvider, request: ModelRequest) -> list[ModelStreamEvent]:
    async def run() -> list[ModelStreamEvent]:
        return [event async for event in provider.stream(request)]

    return asyncio.run(run())


def test_bedrock_provider_streams_text_and_usage_through_converse() -> None:
    client = FakeBedrockClient(
        [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "hello"}}},
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"messageStop": {"stopReason": "end_turn"}},
            {
                "metadata": {
                    "usage": {
                        "inputTokens": 11,
                        "outputTokens": 2,
                        "totalTokens": 13,
                        "cacheReadInputTokens": 3,
                        "cacheWriteInputTokens": 1,
                        "cacheDetails": [{"ttl": "5m", "inputTokens": 1}],
                    },
                    "metrics": {"latencyMs": 42},
                    "serviceTier": {"type": "default"},
                }
            },
        ]
    )
    provider = BedrockProvider(client=client, region_name="us-west-2")
    request = ModelRequest(
        model="us.anthropic.claude-sonnet-4-6-v1",
        messages=[
            Message.text("system", "Be concise."),
            Message.text("user", "Say hello."),
        ],
    )

    events = collect(provider, request)

    assert [event.type for event in events] == [
        ModelStreamEventType.TEXT_DELTA,
        ModelStreamEventType.COMPLETED,
    ]
    assert events[0].delta == "hello"
    assert events[1].completion is not None
    assert events[1].completion.finish_reason == "stop"
    assert events[1].payload == {
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 11,
            "output_tokens": 2,
            "total_tokens": 13,
            "cache_read_input_tokens": 3,
            "cache_creation_input_tokens": 1,
            "cache_details": [{"ttl": "5m", "input_tokens": 1}],
        },
        "bedrock_usage": {
            "inputTokens": 11,
            "outputTokens": 2,
            "totalTokens": 13,
            "cacheReadInputTokens": 3,
            "cacheWriteInputTokens": 1,
            "cacheDetails": [{"ttl": "5m", "inputTokens": 1}],
        },
        "metrics": {"latencyMs": 42},
        "bedrock_service_tier": "default",
    }
    assert client.converse_calls == [
        {
            "modelId": "us.anthropic.claude-sonnet-4-6-v1",
            "system": [{"text": "Be concise."}],
            "messages": [{"role": "user", "content": [{"text": "Say hello."}]}],
            "inferenceConfig": {"maxTokens": 4096},
        }
    ]


def test_bedrock_billing_identity_uses_actual_client_region_and_service_tier() -> None:
    client = FakeBedrockClient([])
    provider = BedrockProvider(client=client, region_name="eu-west-1")
    request = ModelRequest(
        model="global.anthropic.claude-sonnet-4-6",
        messages=[Message.text("user", "hello")],
        options={"bedrock": {"serviceTier": {"type": "reserved"}}},
    )

    identity = asyncio.run(provider.billing_identity_for_request(request))
    completed = provider.billing_identity_for_completion(
        identity,
        {"bedrock_service_tier": "default"},
    )

    assert identity.provider_name == "bedrock"
    assert identity.resource_id == request.model
    assert identity.request_evidence["source_region"] == "us-east-1"
    assert identity.request_evidence["profile_scope"] == "global"
    assert identity.request_evidence["requested_service_tier"] == "reserved"
    assert completed is not None
    assert completed.completion_evidence["effective_service_tier"] == "default"


@pytest.mark.parametrize(
    ("resource_type", "profile_scope", "message"),
    [
        ("invalid", None, "resource_type"),
        ("inference_profile", "invalid", "profile_scope"),
    ],
)
def test_bedrock_billing_identity_rejects_invalid_classification_literals(
    resource_type: Any,
    profile_scope: Any,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        bedrock_billing_identity(
            invoked_model="global.anthropic.claude-sonnet-4-6",
            source_region="us-east-1",
            resource_type=resource_type,
            profile_scope=profile_scope,
        )


@pytest.mark.parametrize(
    ("model_id", "resource_type", "profile_scope"),
    [
        (
            "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-sonnet-v1:0",
            "foundation_model",
            None,
        ),
        (
            "arn:aws:bedrock:us-east-1:123456789012:"
            "inference-profile/global.anthropic.claude-sonnet-v1:0",
            "inference_profile",
            "global",
        ),
        (
            "arn:aws:bedrock:us-east-1:123456789012:"
            "inference-profile/us.anthropic.claude-sonnet-v1:0",
            "inference_profile",
            "geographic",
        ),
        (
            "arn:aws-us-gov:bedrock:us-gov-west-1:123456789012:"
            "inference-profile/us-gov.anthropic.claude-sonnet-v1:0",
            "inference_profile",
            "geographic",
        ),
        (
            "arn:aws:bedrock:us-east-1:123456789012:prompt/ABCDEFGHIJ:1",
            "prompt",
            None,
        ),
        (
            "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/profile-1",
            "application_inference_profile",
            None,
        ),
        ("arn:aws:bedrock", "unknown", None),
        ("arn:aws:s3:::bucket", "unknown", None),
        (
            "arn:aws:bedrock:us-east-1:123456789012:unsupported-resource/value:1",
            "unknown",
            None,
        ),
    ],
)
def test_bedrock_billing_identity_classifies_complete_arn_resource(
    model_id: str,
    resource_type: str,
    profile_scope: str | None,
) -> None:
    assert bedrock_module._bedrock_resource_identity(model_id) == (
        resource_type,
        profile_scope,
    )


def test_bedrock_provider_round_trips_tools_through_converse() -> None:
    client = FakeBedrockClient(
        [
            {
                "contentBlockStart": {
                    "contentBlockIndex": 0,
                    "start": {"toolUse": {"toolUseId": "tool-1", "name": "read_file"}},
                }
            },
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"toolUse": {"input": '{"path":'}},
                }
            },
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"toolUse": {"input": '"README.md"}'}},
                }
            },
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"messageStop": {"stopReason": "tool_use"}},
        ]
    )
    provider = BedrockProvider(client=client)
    request = ModelRequest(
        model="anthropic.claude-test",
        messages=[
            Message.text("user", "Read a file."),
            Message(
                role="assistant",
                content=[
                    TextPart(text="I'll read it."),
                    ToolCallPart(
                        tool_call_id="previous-tool",
                        tool_name="read_file",
                        arguments={"path": "OLD.md"},
                    ),
                ],
            ),
            Message.tool_result(
                tool_call_id="previous-tool",
                tool_name="read_file",
                content="old contents",
            ),
        ],
        tools=[
            {
                "name": "read_file",
                "description": "Read a file.",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }
        ],
    )

    events = collect(provider, request)

    assert events[0].type == ModelStreamEventType.TOOL_CALL
    assert events[0].payload == {
        "id": "tool-1",
        "name": "read_file",
        "arguments": {"path": "README.md"},
    }
    assert events[1].completion is not None
    assert events[1].completion.finish_reason == "tool_calls"
    assert client.converse_calls[0]["messages"] == [
        {"role": "user", "content": [{"text": "Read a file."}]},
        {
            "role": "assistant",
            "content": [
                {"text": "I'll read it."},
                {
                    "toolUse": {
                        "toolUseId": "previous-tool",
                        "name": "read_file",
                        "input": {"path": "OLD.md"},
                    }
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "previous-tool",
                        "content": [{"text": "old contents"}],
                        "status": "success",
                    }
                }
            ],
        },
    ]


def test_bedrock_provider_round_trips_reasoning_content_through_converse() -> None:
    client = FakeBedrockClient(
        [
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"reasoningContent": {"text": "think "}},
                }
            },
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"reasoningContent": {"text": "hard"}},
                }
            },
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"reasoningContent": {"signature": "signed"}},
                }
            },
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 1,
                    "delta": {"reasoningContent": {"redactedContent": b"\x00\x01"}},
                }
            },
            {"contentBlockStop": {"contentBlockIndex": 1}},
            {"messageStop": {"stopReason": "end_turn"}},
        ]
    )
    provider = BedrockProvider(client=client)
    request = ModelRequest(
        model="anthropic.claude-test",
        messages=[
            Message.text("user", "Reason first."),
            Message(
                role="assistant",
                content=[
                    ThinkingPart(
                        text="earlier thought",
                        provider_state={"type": "reasoning_text", "signature": "prior-sig"},
                    ),
                    ThinkingPart(
                        provider_state={
                            "type": "redacted_content",
                            "data_base64": "AgM=",
                        }
                    ),
                    ProviderStatePart(provider="other", state={"opaque": True}),
                    TextPart(text="Earlier answer."),
                ],
            ),
            Message.text("user", "Continue."),
        ],
    )

    events = collect(provider, request)

    assert [event.type for event in events] == [
        ModelStreamEventType.THINKING,
        ModelStreamEventType.THINKING,
        ModelStreamEventType.COMPLETED,
    ]
    assert events[0].delta == "think hard"
    assert events[0].payload == {
        "provider_state": {"type": "reasoning_text", "signature": "signed"}
    }
    assert events[1].delta == ""
    assert events[1].payload == {
        "provider_state": {"type": "redacted_content", "data_base64": "AAE="}
    }
    assert client.converse_calls[0]["messages"][1] == {
        "role": "assistant",
        "content": [
            {
                "reasoningContent": {
                    "reasoningText": {"text": "earlier thought", "signature": "prior-sig"}
                }
            },
            {"reasoningContent": {"redactedContent": b"\x02\x03"}},
            {"text": "Earlier answer."},
        ],
    }


def test_bedrock_provider_counts_the_projected_conversation() -> None:
    client = FakeBedrockClient([])
    provider = BedrockProvider(client=client)
    request = ModelRequest(
        model="anthropic.claude-test",
        messages=[Message.text("system", "Be concise."), Message.text("user", "Hello")],
        options={
            "bedrock": {
                "additionalModelRequestFields": {"top_k": 12},
                "inferenceConfig": {"temperature": 0.2},
            }
        },
    )

    result = asyncio.run(provider.count_input_tokens(request))

    assert result is not None
    assert result.input_tokens == 17
    assert result.method == "official"
    assert result.confidence == "high"
    assert client.count_calls == [
        {
            "modelId": "anthropic.claude-test",
            "input": {
                "converse": {
                    "messages": [{"role": "user", "content": [{"text": "Hello"}]}],
                    "system": [{"text": "Be concise."}],
                    "additionalModelRequestFields": {"top_k": 12},
                }
            },
        }
    ]


@pytest.mark.parametrize(
    "message",
    [
        "The provided model doesn't support counting tokens.",
        "The provided model does not support CountTokens.",
    ],
)
def test_bedrock_provider_returns_none_when_model_does_not_support_count_tokens(
    message: str,
) -> None:
    provider = BedrockProvider(
        client=FailingBedrockClient(
            FakeClientError(
                code="ValidationException",
                message=message,
                status=400,
                request_id="aws-count-unsupported",
            )
        )
    )
    request = ModelRequest(
        model="us.anthropic.claude-sonnet-4-6",
        messages=[Message.text("user", "Hello")],
    )

    assert asyncio.run(provider.count_input_tokens(request)) is None


def test_bedrock_provider_preserves_other_count_tokens_validation_errors() -> None:
    provider = BedrockProvider(
        client=FailingBedrockClient(
            FakeClientError(
                code="ValidationException",
                message="The tool configuration is invalid.",
                status=400,
                request_id="aws-count-invalid",
            )
        )
    )
    request = ModelRequest(
        model="anthropic.claude-test",
        messages=[Message.text("user", "Hello")],
    )

    with pytest.raises(BedrockAPIError, match="tool configuration is invalid"):
        asyncio.run(provider.count_input_tokens(request))


def test_bedrock_provider_preserves_typed_aws_error_fields() -> None:
    provider = BedrockProvider(
        client=FailingBedrockClient(
            FakeClientError(
                code="ThrottlingException",
                message="slow down",
                status=429,
                request_id="aws-request-1",
                retry_after="2.5",
            )
        )
    )

    events = collect(
        provider,
        ModelRequest(model="anthropic.claude-test", messages=[Message.text("user", "Hello")]),
    )

    assert len(events) == 1
    assert events[0].type == ModelStreamEventType.ERROR
    assert events[0].payload["provider"] == "bedrock"
    assert events[0].payload["status_code"] == 429
    assert events[0].payload["provider_error_code"] == "ThrottlingException"
    assert events[0].payload["request_id"] == "aws-request-1"
    assert events[0].payload["retryable"] is True
    assert events[0].payload["retry_after_s"] == 2.5


def test_bedrock_provider_propagates_context_overflow_for_runtime_recovery() -> None:
    provider = BedrockProvider(
        client=FailingBedrockClient(
            FakeClientError(
                code="ValidationException",
                message="Input exceeds the maximum context window",
                status=400,
                request_id="aws-request-2",
            )
        )
    )

    async def run() -> None:
        request = ModelRequest(
            model="anthropic.claude-test", messages=[Message.text("user", "Hello")]
        )
        async for _event in provider.stream(request):
            pass

    try:
        asyncio.run(run())
    except BedrockContextOverflowError as exc:
        assert exc.request_id == "aws-request-2"
        assert exc.retryable is False
    else:
        raise AssertionError("Expected BedrockContextOverflowError")


def test_bedrock_provider_preserves_original_stream_error_status() -> None:
    provider = BedrockProvider(
        client=FakeBedrockClient(
            [
                {
                    "modelStreamErrorException": {
                        "message": "stream failed",
                        "originalStatusCode": 529,
                        "originalMessage": "upstream overloaded",
                    }
                }
            ]
        )
    )

    events = collect(
        provider,
        ModelRequest(model="anthropic.claude-test", messages=[Message.text("user", "Hello")]),
    )

    assert len(events) == 1
    assert events[0].type == ModelStreamEventType.ERROR
    assert events[0].payload["status_code"] == 529
    assert events[0].payload["provider_error_code"] == "modelStreamErrorException"
    assert events[0].payload["retryable"] is True
    assert "upstream overloaded" in events[0].payload["error"]


def test_bedrock_provider_types_streamed_context_overflow_for_runtime_recovery() -> None:
    provider = BedrockProvider(
        client=FakeBedrockClient(
            [{"validationException": {"message": "Input exceeds the maximum context window"}}]
        )
    )

    async def run() -> None:
        request = ModelRequest(
            model="anthropic.claude-test", messages=[Message.text("user", "Hello")]
        )
        async for _event in provider.stream(request):
            pass

    with pytest.raises(BedrockContextOverflowError):
        asyncio.run(run())


def test_bedrock_provider_rejects_ambiguous_injected_client_configuration() -> None:
    with pytest.raises(ValueError, match="injected client"):
        BedrockProvider(client=FakeBedrockClient([]), profile_name="prod")


def test_bedrock_provider_projects_resolved_image_attachments() -> None:
    client = FakeBedrockClient([{"messageStop": {"stopReason": "end_turn"}}])
    provider = BedrockProvider(client=client)
    attachment = file_attachment(
        artifact_id="image-1",
        kind=FileAttachmentKind.IMAGE,
        filename="screen.png",
        content_type="image/png",
        size_bytes=3,
    )
    request = ModelRequest(
        model="anthropic.claude-test",
        messages=[
            Message(
                role="user",
                content=[TextPart(text="Inspect this."), FilePart(attachment=attachment)],
            )
        ],
        options={
            RESOLVED_FILE_ATTACHMENTS_OPTION: {
                "image-1": {
                    "artifact_id": "image-1",
                    "kind": "image",
                    "filename": "screen.png",
                    "content_type": "image/png",
                    "data_base64": base64.b64encode(b"png").decode("ascii"),
                    "metadata": {},
                }
            }
        },
    )

    collect(provider, request)

    assert client.converse_calls[0]["messages"] == [
        {
            "role": "user",
            "content": [
                {"text": "Inspect this."},
                {"image": {"format": "png", "source": {"bytes": b"png"}}},
            ],
        }
    ]


@pytest.mark.anyio
async def test_bedrock_provider_supports_structured_output_via_tools() -> None:
    client = FakeBedrockClient(
        [
            {
                "contentBlockStart": {
                    "contentBlockIndex": 0,
                    "start": {
                        "toolUse": {
                            "toolUseId": "final-1",
                            "name": STRUCTURED_OUTPUT_TOOL_NAME,
                        }
                    },
                }
            },
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"toolUse": {"input": '{"output":{"answer":"ok"}}'}},
                }
            },
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"messageStop": {"stopReason": "tool_use"}},
            {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 2, "totalTokens": 12}}},
        ]
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(BedrockProvider(client=client), default=True)
    app.register_agent(AgentSpec(name="assistant", model="anthropic.claude-test"))

    events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "Answer with structured output.")],
                structured_output=StructuredOutputSpec(
                    name="answer",
                    json_schema={
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                ),
            )
        )
    ]

    validated = next(
        event for event in events if event.type == EventType.STRUCTURED_OUTPUT_VALIDATED
    )
    completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
    assert validated.payload["output"] == {"answer": "ok"}
    usage_metrics = completed.payload["usage_metrics"]
    assert usage_metrics["provider_name"] == "bedrock"
    assert usage_metrics["input_tokens"] == 10
    assert usage_metrics["output_tokens"] == 2
    assert usage_metrics["total_tokens"] == 12
    cost = await app.get_session_cost(
        completed.session_id,
        PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="bedrock",
                    model="anthropic.claude-test",
                    match="exact",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                    pricing_context={
                        "source_region": ("us-east-1",),
                        "service_tier": ("default",),
                    },
                ),
            )
        ),
    )
    assert cost.priced_model_steps == 1
    assert cost.line_items[0].provider_name == "bedrock"
    assert cost.line_items[0].model == "anthropic.claude-test"
    assert cost.line_items[0].total_cost == Decimal("0.000014")
    sent_tools = client.converse_calls[0]["toolConfig"]["tools"]
    assert any(tool["toolSpec"]["name"] == STRUCTURED_OUTPUT_TOOL_NAME for tool in sent_tools)


@pytest.mark.parametrize(
    ("raw_reason", "normalized"),
    [
        ("stop_sequence", "stop"),
        ("guardrail_intervened", "content_filter"),
        ("content_filtered", "content_filter"),
        ("model_context_window_exceeded", "length"),
        ("malformed_model_output", "error"),
    ],
)
def test_bedrock_provider_normalizes_documented_stop_reasons(
    raw_reason: str, normalized: str
) -> None:
    provider = BedrockProvider(
        client=FakeBedrockClient([{"messageStop": {"stopReason": raw_reason}}])
    )

    events = collect(
        provider,
        ModelRequest(model="anthropic.claude-test", messages=[Message.text("user", "Hello")]),
    )

    assert events[-1].completion is not None
    assert events[-1].completion.raw_finish_reason == raw_reason
    assert events[-1].completion.finish_reason == normalized


@pytest.mark.anyio
async def test_bedrock_provider_closes_blocking_sdk_stream_on_cancellation() -> None:
    stream = BlockingBedrockStream()
    provider = BedrockProvider(client=BlockingBedrockClient(stream), stream_close_timeout_s=1)
    request = ModelRequest(model="anthropic.claude-test", messages=[Message.text("user", "Hello")])

    async def consume() -> None:
        async for _event in provider.stream(request):
            pass

    task = asyncio.create_task(consume())
    started = await asyncio.to_thread(stream.started.wait, 1)
    assert started is True
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert stream.closed is True


@pytest.mark.anyio
async def test_bedrock_provider_bounds_blocking_sdk_stream_close() -> None:
    stream = BlockingCloseBedrockStream()
    provider = BedrockProvider(
        client=BlockingBedrockClient(stream),
        stream_close_timeout_s=0.01,
    )
    request = ModelRequest(model="anthropic.claude-test", messages=[Message.text("user", "Hello")])

    async def consume() -> None:
        async for _event in provider.stream(request):
            pass

    task = asyncio.create_task(consume())
    assert await asyncio.to_thread(stream.started.wait, 1) is True
    task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=0.25)
        assert stream.close_started.is_set()
    finally:
        stream.close_released.set()
        stream.released.set()


@pytest.mark.anyio
async def test_bedrock_provider_does_not_duplicate_events_after_queue_put_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_run_coroutine_threadsafe = asyncio.run_coroutine_threadsafe
    forced_timeout = threading.Event()

    class TimeoutOnceFuture:
        def __init__(self, future: Any) -> None:
            self.future = future

        def result(self, timeout: float | None = None) -> Any:
            result = self.future.result(timeout)
            if not forced_timeout.is_set():
                forced_timeout.set()
                raise FutureTimeoutError
            return result

        def cancel(self) -> bool:
            return self.future.cancel()

    def run_coroutine_threadsafe(coro: Any, loop: asyncio.AbstractEventLoop) -> Any:
        return TimeoutOnceFuture(original_run_coroutine_threadsafe(coro, loop))

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", run_coroutine_threadsafe)
    provider = BedrockProvider(
        client=FakeBedrockClient(
            [
                {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "once"}}},
                {"messageStop": {"stopReason": "end_turn"}},
            ]
        )
    )
    request = ModelRequest(model="anthropic.claude-test", messages=[Message.text("user", "Hello")])

    events = [event async for event in provider.stream(request)]

    assert forced_timeout.is_set()
    assert [event.delta for event in events if event.type == ModelStreamEventType.TEXT_DELTA] == [
        "once"
    ]


@pytest.mark.anyio
async def test_bedrock_provider_creates_lazy_client_once_off_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_loop_thread = threading.get_ident()
    creation_started = threading.Event()
    release_creation = threading.Event()
    creation_threads: list[int] = []
    client = FakeBedrockClient([])

    class FakeSession:
        def __init__(self, **kwargs: Any) -> None:
            creation_threads.append(threading.get_ident())
            creation_started.set()
            release_creation.wait(timeout=1)

        def client(self, service_name: str, **kwargs: Any) -> FakeBedrockClient:
            assert service_name == "bedrock-runtime"
            return client

    class FakeBoto3:
        Session = FakeSession

    monkeypatch.setattr(bedrock_module, "_boto3_module", lambda: FakeBoto3)
    provider = BedrockProvider(region_name="us-west-2")
    tasks = [asyncio.create_task(provider._get_client()) for _ in range(8)]
    assert await asyncio.to_thread(creation_started.wait, 1) is True
    release_creation.set()

    clients = await asyncio.gather(*tasks)

    assert clients == [client] * 8
    assert len(creation_threads) == 1
    assert creation_threads[0] != event_loop_thread


@pytest.mark.anyio
async def test_bedrock_provider_aclose_only_closes_owned_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owned_client = FakeBedrockClient([])
    injected_client = FakeBedrockClient([])

    class FakeSession:
        def client(self, service_name: str, **kwargs: Any) -> FakeBedrockClient:
            assert service_name == "bedrock-runtime"
            return owned_client

    class FakeBoto3:
        Session = FakeSession

    monkeypatch.setattr(bedrock_module, "_boto3_module", lambda: FakeBoto3)
    owned_provider = BedrockProvider()
    injected_provider = BedrockProvider(client=injected_client)
    assert await owned_provider._get_client() is owned_client

    await owned_provider.aclose()
    await injected_provider.aclose()

    assert owned_client.closed is True
    assert owned_provider._client is None
    assert injected_client.closed is False


@pytest.mark.anyio
async def test_bedrock_provider_reports_idle_stream_timeout_and_closes_stream() -> None:
    stream = BlockingBedrockStream()
    provider = BedrockProvider(
        client=BlockingBedrockClient(stream),
        stream_idle_timeout_s=0.01,
        stream_close_timeout_s=1,
    )
    request = ModelRequest(model="anthropic.claude-test", messages=[Message.text("user", "Hello")])

    events = [event async for event in provider.stream(request)]

    assert len(events) == 1
    assert events[0].type == ModelStreamEventType.ERROR
    assert "produced no event for 0.01 seconds" in events[0].payload["error"]
    assert stream.closed is True


def test_bedrock_provider_reports_unfinished_tool_blocks() -> None:
    provider = BedrockProvider(
        client=FakeBedrockClient(
            [
                {
                    "contentBlockStart": {
                        "contentBlockIndex": 0,
                        "start": {"toolUse": {"toolUseId": "tool-1", "name": "read_file"}},
                    }
                },
                {"messageStop": {"stopReason": "tool_use"}},
            ]
        )
    )
    request = ModelRequest(model="anthropic.claude-test", messages=[Message.text("user", "Hello")])

    events = collect(provider, request)

    assert len(events) == 1
    assert events[0].type == ModelStreamEventType.ERROR
    assert "unfinished tool blocks" in events[0].payload["error"]
