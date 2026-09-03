"""Tests for the shared provider transport plumbing (_http + _sse).

The OpenAI, Anthropic, Chat Completions, and Vertex adapters delegate their
HTTP/SSE mechanics to ``cayu.providers._http`` and ``cayu.providers._sse``.
These tests pin the shared behavior: one SSE parser with distinct raw-byte
and decoded-event clocks, provider-labeled errors, bounded framing, and shared
URL validation. Keep-alives prove transport activity only.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import ssl
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import certifi
import httpx
import pytest

from cayu import Message
from cayu.providers import (
    AnthropicAPIError,
    AnthropicError,
    AnthropicProvider,
    BedrockProvider,
    ChatCompletionsAPIError,
    ChatCompletionsError,
    ChatCompletionsProtocolError,
    ChatCompletionsProvider,
    HttpxAnthropicTransport,
    HttpxChatCompletionsTransport,
    HttpxOpenAITransport,
    HttpxVertexTransport,
    ModelProviderError,
    ModelRequest,
    ModelStreamEventType,
    OpenAIAPIError,
    OpenAIError,
    OpenAIProtocolError,
    OpenAIProvider,
    OpenAISubscriptionProvider,
    VertexAPIError,
    VertexError,
    VertexProvider,
)
from cayu.providers._credential_boundary import ProviderStreamCleanupError
from cayu.providers._http import (
    SharedAsyncClient,
    _trusted_sse_retry_after_s,
    credential_safe_error_event,
    new_async_client,
    retry_after_seconds,
    stream_sse_json_events,
    validate_base_url,
    validate_url,
)
from cayu.providers._sse import (
    SseEventLimitError,
    SseEventTimeoutError,
    _aiter_bounded_sse_lines,
    aiter_sse_json_events,
)
from cayu.providers.base import ModelStreamDeadlineError
from cayu.providers.deadlines import (
    ProviderDeadlineKind,
    ProviderStreamDeadlineController,
    ProviderStreamDeadlineExceeded,
    ProviderStreamDeadlines,
)
from cayu.providers.openai_subscription import OpenAISubscriptionCredentials


def _deadline_controller(protocol_idle_timeout_s: float = 1.0) -> ProviderStreamDeadlineController:
    return ProviderStreamDeadlineController(
        ProviderStreamDeadlines(
            transport_idle_timeout_s=10.0,
            protocol_idle_timeout_s=protocol_idle_timeout_s,
            semantic_progress_timeout_s=10.0,
            absolute_stream_timeout_s=10.0,
        )
    )


class _LineByteStream(httpx.AsyncByteStream):
    def __init__(self, lines: list[str], *, heartbeat_sleep_s: float = 0.0) -> None:
        self._lines = lines
        self._heartbeat_sleep_s = heartbeat_sleep_s
        self.closed = False

    async def __aiter__(self):
        for line in self._lines:
            if self._heartbeat_sleep_s:
                await asyncio.sleep(self._heartbeat_sleep_s)
            yield (line + "\n").encode("utf-8")

    async def aclose(self) -> None:
        self.closed = True


class _StreamingResponse(httpx.Response):
    """Real HTTPX response backed by a controlled asynchronous byte stream."""

    def __init__(self, lines: list[str], *, heartbeat_sleep_s: float = 0.0) -> None:
        self.byte_stream = _LineByteStream(
            lines,
            heartbeat_sleep_s=heartbeat_sleep_s,
        )
        super().__init__(
            200,
            headers={"content-type": "text/event-stream"},
            stream=self.byte_stream,
            request=httpx.Request("POST", "https://provider.example/v1/stream"),
        )


class _ChunkedByteStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.yielded_chunks = 0
        self.closed = False

    async def __aiter__(self):
        for chunk in self._chunks:
            self.yielded_chunks += 1
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


class _DelayedChunkStream(_ChunkedByteStream):
    def __init__(self, chunks: list[bytes], *, delay_s: float) -> None:
        super().__init__(chunks)
        self._delay_s = delay_s

    async def __aiter__(self):
        for chunk in self._chunks:
            await asyncio.sleep(self._delay_s)
            self.yielded_chunks += 1
            yield chunk


class _ResetAfterEventByteStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.closed = False

    async def __aiter__(self):
        yield (b'data: {"type":"response.created","response":{"id":"resp-reset"}}\n\n')
        raise httpx.RemoteProtocolError(
            "forced established response-body reset",
            request=httpx.Request("POST", "https://provider.example/v1/stream"),
        )

    async def aclose(self) -> None:
        self.closed = True


class _BlockingByteStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.finalized = asyncio.Event()
        self.closed = False

    async def __aiter__(self):
        try:
            self.started.set()
            await asyncio.Event().wait()
            yield b""  # pragma: no cover
        finally:
            self.finalized.set()

    async def aclose(self) -> None:
        self.closed = True


class _BlockingAfterEventByteStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.closed = False

    async def __aiter__(self):
        yield b'data: {"ok": true}\n\n'
        await asyncio.Event().wait()
        yield b""  # pragma: no cover

    async def aclose(self) -> None:
        self.closed = True


class _ClosableEventStream:
    def __init__(self, event: Mapping[str, Any] | list[Mapping[str, Any]]) -> None:
        self._events = iter(event if isinstance(event, list) else [event])
        self.closed = False

    def __aiter__(self) -> _ClosableEventStream:
        return self

    async def __anext__(self) -> Mapping[str, Any]:
        try:
            return next(self._events)
        except StopIteration:
            pass
        await asyncio.Event().wait()
        raise StopAsyncIteration  # pragma: no cover

    async def aclose(self) -> None:
        self.closed = True


class _FiniteFailingCloseEventStream:
    def __init__(self, events: list[Mapping[str, Any]]) -> None:
        self._events = iter(events)
        self.closed = False

    def __aiter__(self) -> _FiniteFailingCloseEventStream:
        return self

    async def __anext__(self) -> Mapping[str, Any]:
        try:
            return next(self._events)
        except StopIteration:
            raise StopAsyncIteration from None

    async def aclose(self) -> None:
        self.closed = True
        raise RuntimeError("provider stream close failed")


class _EndlessByteStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.yielded_chunks = 0
        self.closed = False

    async def __aiter__(self):
        while True:
            await asyncio.sleep(0.001)
            self.yielded_chunks += 1
            yield b"x"

    async def aclose(self) -> None:
        self.closed = True


class _HeartbeatTailEventStream:
    def __init__(self, completed_event: Mapping[str, Any]) -> None:
        self._completed_event = completed_event
        self._yielded_completion = False
        self.tail_reads = 0
        self.closed = False

    def __aiter__(self) -> _HeartbeatTailEventStream:
        return self

    async def __anext__(self) -> Mapping[str, Any]:
        if not self._yielded_completion:
            self._yielded_completion = True
            return self._completed_event
        self.tail_reads += 1
        await asyncio.sleep(0.01)
        return {"type": "ping"}

    async def aclose(self) -> None:
        self.closed = True


class _ClosableProviderTransport:
    def __init__(self, source: Any) -> None:
        self.source = source

    def stream_response_events(self, **_kwargs: Any) -> Any:
        return self.source

    def stream_chat_completions(self, **_kwargs: Any) -> Any:
        return self.source

    def stream_message_events(self, **_kwargs: Any) -> Any:
        return self.source


class _StaticSubscriptionAuth:
    async def credentials(self) -> OpenAISubscriptionCredentials:
        return OpenAISubscriptionCredentials(
            access_token="subscription-access",
            refresh_token="subscription-refresh",
            expires_at=2_000_000_000,
        )


class _UnusedSubscriptionAuth:
    async def credentials(self) -> Any:  # pragma: no cover - construction-only test seam
        raise AssertionError("credentials must not be resolved during provider construction")


def _successful_raw_stream_events(provider_name: str) -> list[Mapping[str, Any]]:
    if provider_name in {"openai", "openai_subscription"}:
        return [
            {
                "type": "response.completed",
                "response": {
                    "id": "response-1",
                    "model": "test-model",
                    "status": "completed",
                    "output": [],
                    "usage": {"input_tokens": 2, "output_tokens": 1},
                },
            }
        ]
    if provider_name == "chat_completions":
        return [
            {
                "id": "chat-1",
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 1,
                    "total_tokens": 3,
                },
            }
        ]
    if provider_name in {"anthropic", "vertex"}:
        return [
            {
                "type": "message_start",
                "message": {
                    "id": "message-1",
                    "model": "test-model",
                    "usage": {"input_tokens": 2},
                },
            },
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 1},
            },
            {"type": "message_stop"},
        ]
    raise AssertionError(f"Unhandled provider: {provider_name}")


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("transport_factory", "method_name", "patch_target"),
    [
        pytest.param(
            HttpxOpenAITransport,
            "stream_response_events",
            "cayu.providers.openai.stream_sse_json_events",
            id="openai",
        ),
        pytest.param(
            HttpxChatCompletionsTransport,
            "stream_chat_completions",
            "cayu.providers.chat_completions.stream_sse_json_events",
            id="chat-completions",
        ),
        pytest.param(
            HttpxAnthropicTransport,
            "stream_message_events",
            "cayu.providers.anthropic.stream_sse_json_events",
            id="anthropic",
        ),
        pytest.param(
            HttpxVertexTransport,
            "stream_message_events",
            "cayu.providers.vertex.stream_sse_json_events",
            id="vertex",
        ),
    ],
)
async def test_http_transport_pass_through_closes_shared_sse_iterator(
    monkeypatch: pytest.MonkeyPatch,
    transport_factory: Callable[[], Any],
    method_name: str,
    patch_target: str,
) -> None:
    source = _ClosableEventStream({"ok": True})
    monkeypatch.setattr(patch_target, lambda **_kwargs: source)
    transport = transport_factory()
    events = getattr(transport, method_name)(
        url="https://provider.example/v1/stream",
        headers={},
        payload={},
        timeout_s=1.0,
        transport_idle_timeout_s=1.0,
        protocol_idle_timeout_s=1.0,
        semantic_progress_timeout_s=1.0,
        absolute_stream_timeout_s=1.0,
    )

    assert await anext(events) == {"ok": True}
    await events.aclose()

    assert source.closed


@pytest.mark.anyio
@pytest.mark.parametrize(
    "provider_name",
    [
        "openai",
        "chat_completions",
        "anthropic",
        "vertex",
        "openai_subscription",
    ],
)
async def test_provider_closes_translator_and_raw_stream_on_abandonment(
    provider_name: str,
) -> None:
    if provider_name in {"openai", "openai_subscription"}:
        raw_event = {"type": "response.output_text.delta", "delta": "hello"}
    elif provider_name == "chat_completions":
        raw_event = {
            "id": "chat-1",
            "model": "test-model",
            "choices": [{"index": 0, "delta": {"content": "hello"}, "finish_reason": None}],
        }
    elif provider_name in {"anthropic", "vertex"}:
        raw_event = [
            {
                "type": "message_start",
                "message": {"id": "message-1", "model": "test-model"},
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": "hello"},
            },
        ]
    else:  # pragma: no cover - guarded by the parameter list
        raise AssertionError(f"Unhandled provider: {provider_name}")
    source = _ClosableEventStream(raw_event)
    transport = _ClosableProviderTransport(source)

    if provider_name == "openai":
        provider = OpenAIProvider(api_key="test-key", transport=transport)
    elif provider_name == "chat_completions":
        provider = ChatCompletionsProvider(api_key="test-key", transport=transport)
    elif provider_name == "anthropic":
        provider = AnthropicProvider(api_key="test-key", transport=transport)
    elif provider_name == "vertex":
        provider = VertexProvider(
            project_id="test-project",
            region="us-east5",
            credentials=SimpleNamespace(valid=True, token="test-token"),
            transport=transport,
        )
    elif provider_name == "openai_subscription":
        provider = OpenAISubscriptionProvider(
            auth=_StaticSubscriptionAuth(),
            transport=transport,
        )
    else:  # pragma: no cover - guarded by the parameter list
        raise AssertionError(f"Unhandled provider: {provider_name}")
    events = provider.stream(
        ModelRequest(
            model="test-model",
            messages=[Message.text("user", "hello")],
        )
    )
    event = await anext(events)
    assert event.delta == "hello"

    await events.aclose()

    assert source.closed


@pytest.mark.anyio
@pytest.mark.parametrize(
    "provider_name",
    [
        "openai",
        "chat_completions",
        "anthropic",
        "vertex",
        "openai_subscription",
    ],
)
async def test_provider_preserves_completed_before_stream_cleanup_failure(
    provider_name: str,
) -> None:
    source = _FiniteFailingCloseEventStream(_successful_raw_stream_events(provider_name))
    transport = _ClosableProviderTransport(source)

    if provider_name == "openai":
        provider = OpenAIProvider(api_key="test-key", transport=transport)
    elif provider_name == "chat_completions":
        provider = ChatCompletionsProvider(api_key="test-key", transport=transport)
    elif provider_name == "anthropic":
        provider = AnthropicProvider(api_key="test-key", transport=transport)
    elif provider_name == "vertex":
        provider = VertexProvider(
            project_id="test-project",
            region="us-east5",
            credentials=SimpleNamespace(valid=True, token="test-token"),
            transport=transport,
        )
    elif provider_name == "openai_subscription":
        provider = OpenAISubscriptionProvider(
            auth=_StaticSubscriptionAuth(),
            transport=transport,
        )
    else:  # pragma: no cover - guarded by the parameter list
        raise AssertionError(f"Unhandled provider: {provider_name}")

    events = []
    with pytest.raises(ProviderStreamCleanupError) as exc_info:
        async for event in provider.stream(
            ModelRequest(
                model="test-model",
                messages=[Message.text("user", "hello")],
            )
        ):
            events.append(event)

    assert source.closed
    assert [event.type for event in events] == [ModelStreamEventType.COMPLETED]
    assert events[0].payload["usage"] is not None
    expected_provider = "openai" if provider_name == "openai_subscription" else provider_name
    assert exc_info.value.provider == expected_provider
    assert exc_info.value.error_type == "ProviderStreamCleanupError"
    assert exc_info.value.retryable is False


@pytest.mark.anyio
async def test_public_provider_closes_task_affine_custom_stream_in_owner_task() -> None:
    class TaskAffineEventStream:
        def __init__(self) -> None:
            self._events = iter(_successful_raw_stream_events("openai"))
            self.owner: asyncio.Task[Any] | None = None
            self.closed = False

        def __aiter__(self) -> TaskAffineEventStream:
            return self

        async def __anext__(self) -> Mapping[str, Any]:
            task = asyncio.current_task()
            assert task is not None
            if self.owner is None:
                self.owner = task
            assert task is self.owner
            try:
                return next(self._events)
            except StopIteration:
                raise StopAsyncIteration from None

        async def aclose(self) -> None:
            assert asyncio.current_task() is self.owner
            self.closed = True

    source = TaskAffineEventStream()
    provider = OpenAIProvider(
        api_key="test-key",
        transport=_ClosableProviderTransport(source),
    )

    events = [
        event
        async for event in provider.stream(
            ModelRequest(
                model="test-model",
                messages=[Message.text("user", "hello")],
            )
        )
    ]

    assert [event.type for event in events] == [ModelStreamEventType.COMPLETED]
    assert source.closed


@pytest.mark.anyio
async def test_provider_closes_without_reading_post_completion_heartbeat_tail() -> None:
    source = _HeartbeatTailEventStream(_successful_raw_stream_events("openai")[0])
    provider = OpenAIProvider(
        api_key="test-key",
        transport=_ClosableProviderTransport(source),
    )

    async def collect() -> list[Any]:
        return [
            event
            async for event in provider.stream(
                ModelRequest(
                    model="test-model",
                    messages=[Message.text("user", "hello")],
                )
            )
        ]

    events = await asyncio.wait_for(collect(), timeout=0.5)

    assert [event.type for event in events] == [ModelStreamEventType.COMPLETED]
    assert source.closed
    assert source.tail_reads == 0


@pytest.mark.anyio
async def test_provider_close_failure_fails_closed_with_primary_error_identity() -> None:
    class FailingCloseEventStream(_ClosableEventStream):
        async def aclose(self) -> None:
            self.closed = True
            raise RuntimeError("secondary stream close failure")

    source = FailingCloseEventStream(
        {
            "error": {
                "type": "server_error",
                "code": "internal_error",
                "message": "temporary provider failure",
            }
        }
    )
    provider = ChatCompletionsProvider(
        api_key="test-key",
        transport=_ClosableProviderTransport(source),
    )

    events = [
        event
        async for event in provider.stream(
            ModelRequest(
                model="test-model",
                messages=[Message.text("user", "hello")],
            )
        )
    ]

    assert source.closed
    assert len(events) == 1
    assert events[0].payload["error_type"] == "ProviderStreamCleanupError"
    assert events[0].payload["status_code"] == 500
    assert events[0].payload["provider_error_type"] == "server_error"
    assert events[0].payload["provider_error_code"] == "internal_error"
    assert events[0].payload["stream_cleanup_failed"] is True
    assert events[0].payload["retryable"] is False


@pytest.mark.anyio
async def test_subscription_close_failure_uses_the_same_terminal_cleanup_shape() -> None:
    class FailingCloseEventStream(_ClosableEventStream):
        async def aclose(self) -> None:
            self.closed = True
            raise RuntimeError("secondary subscription stream close failure")

    source = FailingCloseEventStream(
        {
            "type": "response.failed",
            "response": {
                "error": {
                    "type": "server_error",
                    "code": "internal_error",
                    "message": "temporary provider failure",
                }
            },
        }
    )
    provider = OpenAISubscriptionProvider(
        auth=_StaticSubscriptionAuth(),
        transport=_ClosableProviderTransport(source),
    )

    events = [
        event
        async for event in provider.stream(
            ModelRequest(
                model="test-model",
                messages=[Message.text("user", "hello")],
            )
        )
    ]

    assert source.closed
    assert len(events) == 1
    assert events[0].payload["error_type"] == "ProviderStreamCleanupError"
    assert events[0].payload["status_code"] == 500
    assert events[0].payload["provider_error_type"] == "server_error"
    assert events[0].payload["provider_error_code"] == "internal_error"
    assert events[0].payload["stream_cleanup_failed"] is True
    assert events[0].payload["retryable"] is False


@pytest.mark.anyio
async def test_provider_preserves_real_cancellation_during_in_band_error_cleanup() -> None:
    close_started = asyncio.Event()
    credential = "provider-close-cancellation-secret-canary"

    class BlockingCloseEventStream(_ClosableEventStream):
        async def aclose(self) -> None:
            close_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise RuntimeError(f"close transformed cancellation near {credential}") from None

    source = BlockingCloseEventStream(
        {
            "error": {
                "type": "server_error",
                "code": "internal_error",
                "message": f"temporary provider failure near {credential}",
            }
        }
    )
    provider = ChatCompletionsProvider(
        api_key=credential,
        transport=_ClosableProviderTransport(source),
    )

    async def consume() -> None:
        async for _ in provider.stream(
            ModelRequest(
                model="test-model",
                messages=[Message.text("user", "hello")],
            )
        ):
            pass

    task = asyncio.create_task(consume())
    await close_started.wait()
    task.cancel()
    assert task.cancelling() == 1

    cancellation: asyncio.CancelledError | None = None
    try:
        await task
    except asyncio.CancelledError as exc:
        cancellation = exc

    assert cancellation is not None
    assert task.cancelled()
    assert cancellation.args == ("Chat Completions provider request cancelled",)
    assert credential not in repr(cancellation)
    assert getattr(cancellation, "__notes__", ()) == [
        "Provider stream cleanup was cancelled after a provider operation failure."
    ]
    assert cancellation.__cause__ is None
    assert cancellation.__context__ is None


async def _stream_mock_sse(
    stream: httpx.AsyncByteStream,
    *,
    headers: Mapping[str, str] | None = None,
    transport_idle_timeout_s: float = 1.0,
    protocol_idle_timeout_s: float = 1.0,
    semantic_progress_timeout_s: float = 1.0,
    absolute_stream_timeout_s: float = 1.0,
) -> list[Mapping[str, Any]]:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream", **dict(headers or {})},
            stream=stream,
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        return [
            event
            async for event in stream_sse_json_events(
                client=client,
                url="https://provider.example/v1/stream",
                headers={},
                payload={},
                timeout_s=1.0,
                transport_idle_timeout_s=transport_idle_timeout_s,
                protocol_idle_timeout_s=protocol_idle_timeout_s,
                semantic_progress_timeout_s=semantic_progress_timeout_s,
                absolute_stream_timeout_s=absolute_stream_timeout_s,
                request_label="OpenAI API",
                response_label="OpenAI",
                api_error=OpenAIAPIError,
                protocol_error=OpenAIProtocolError,
                error_response_text=lambda response: response.text,
            )
        ]


@pytest.mark.parametrize(
    ("provider_name", "provider_label"),
    [
        ("openai", "OpenAI"),
        ("chat_completions", "Chat Completions"),
        ("anthropic", "Anthropic"),
        ("vertex", "Vertex"),
    ],
)
def test_provider_error_projection_omits_arbitrary_identity_strings(
    provider_name: str,
    provider_label: str,
) -> None:
    secret = "provider-identity-secret-canary-ABCDEFGHIJKLMNOP"
    error = ModelProviderError(
        "fixed provider failure",
        provider=provider_name,
        status_code=500,
        error_type=secret,
        error_code=secret[:16],
        request_id=secret,
        retryable=True,
    )

    event = credential_safe_error_event(
        error,
        provider_label=provider_label,
        provider_name=provider_name,
        credential_values=("provider-credential",),
    )

    assert event.payload["status_code"] == 500
    assert event.payload["retryable"] is True
    assert "provider_error_type" not in event.payload
    assert "provider_error_code" not in event.payload
    assert "request_id" not in event.payload
    assert secret not in repr(event.payload)
    assert secret[:16] not in repr(event.payload)


@pytest.mark.parametrize(
    "error_type",
    ["SseEventTimeoutError", "SseEventLimitError"],
)
def test_provider_error_projection_preserves_fixed_sse_classification(error_type: str) -> None:
    error = ModelProviderError(
        "fixed SSE failure",
        provider="openai",
        error_type=error_type,
        retryable=error_type != "SseEventLimitError",
    )

    event = credential_safe_error_event(
        error,
        provider_label="OpenAI",
        provider_name="openai",
        credential_values=("provider-credential",),
    )

    assert event.payload["provider_error_type"] == error_type
    assert event.payload["retryable"] is (error_type != "SseEventLimitError")


@pytest.mark.parametrize("credential_values", [(), ("provider-credential",)])
def test_provider_error_projection_omits_arbitrary_exception_type_names(
    credential_values: tuple[str, ...],
) -> None:
    secret = "provider_identity_secret_canary_ABCDEFGHIJKLMNOP"
    secret_named_error = type(secret, (RuntimeError,), {})

    event = credential_safe_error_event(
        secret_named_error("fixed failure"),
        provider_label="OpenAI",
        provider_name="openai",
        credential_values=credential_values,
    )

    assert event.payload["error_type"] == "Exception"
    assert event.payload["error"] == "Exception: OpenAI provider failed"
    assert secret not in repr(event.payload)


@pytest.mark.parametrize(
    ("error_type", "provider_name", "provider_label"),
    [
        (AnthropicError, "anthropic", "Anthropic"),
        (ChatCompletionsError, "chat_completions", "Chat Completions"),
        (OpenAIError, "openai", "OpenAI"),
        (VertexError, "vertex", "Vertex"),
    ],
)
def test_provider_error_projection_preserves_fixed_cayu_exception_types(
    error_type: type[RuntimeError],
    provider_name: str,
    provider_label: str,
) -> None:
    event = credential_safe_error_event(
        error_type("provider-owned fixed failure"),
        provider_label=provider_label,
        provider_name=provider_name,
        credential_values=(),
    )

    assert event.payload["error_type"] == error_type.__name__
    assert event.payload["error"] == f"{error_type.__name__}: {provider_label} provider failed"


def test_new_async_client_uses_certifi_without_extra_ca(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    created_with: list[str] = []
    context = object()

    def make_client(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    def create_default_context(*, cafile: str) -> object:
        created_with.append(cafile)
        return context

    monkeypatch.delenv("CAYU_PROVIDER_CA_BUNDLE", raising=False)
    monkeypatch.setattr(
        "cayu.providers._http.ssl.create_default_context",
        create_default_context,
    )
    monkeypatch.setattr("cayu.providers._http.httpx.AsyncClient", make_client)

    assert new_async_client() is not None
    assert created_with == [certifi.where()]
    assert captured == {"verify": context}


def test_new_async_client_augments_certifi_with_explicit_extra_ca(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    extra_ca = tmp_path / "session-ca.pem"
    extra_ca.write_text("test certificate", encoding="utf-8")
    loaded: list[str] = []
    created_with: list[str] = []
    captured: dict[str, Any] = {}

    class Context:
        def load_verify_locations(self, *, cafile: str) -> None:
            loaded.append(cafile)

    context = Context()

    def create_default_context(*, cafile: str) -> Context:
        created_with.append(cafile)
        return context

    def make_client(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setenv("CAYU_PROVIDER_CA_BUNDLE", str(extra_ca))
    monkeypatch.setattr(
        "cayu.providers._http.ssl.create_default_context",
        create_default_context,
    )
    monkeypatch.setattr("cayu.providers._http.httpx.AsyncClient", make_client)

    assert new_async_client() is not None
    assert created_with == [certifi.where()]
    assert loaded == [str(extra_ca)]
    assert captured == {"verify": context}


def test_new_async_client_fails_closed_when_extra_ca_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Context:
        def load_verify_locations(self, *, cafile: str) -> None:
            raise ssl.SSLError("invalid CA bundle")

    monkeypatch.setenv("CAYU_PROVIDER_CA_BUNDLE", "/missing/session-ca.pem")
    monkeypatch.setattr(
        "cayu.providers._http.ssl.create_default_context",
        lambda *, cafile: Context(),
    )

    with pytest.raises(ssl.SSLError, match="invalid CA bundle"):
        new_async_client()


def test_new_async_client_fails_closed_when_extra_ca_is_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAYU_PROVIDER_CA_BUNDLE", " \t ")

    with pytest.raises(
        ValueError,
        match="`CAYU_PROVIDER_CA_BUNDLE` cannot be blank",
    ):
        new_async_client()


class _StreamContext:
    def __init__(self, response: httpx.Response) -> None:
        self._response = response

    async def __aenter__(self) -> httpx.Response:
        return self._response

    async def __aexit__(self, *args: Any) -> None:
        await self._response.aclose()


def _client_factory(response: httpx.Response) -> type:
    class FakeClient:
        is_closed = False

        def __init__(self, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> FakeClient:
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
        ) -> _StreamContext:
            return _StreamContext(response)

    return FakeClient


def _request_error_client_factory(error_type: type[httpx.RequestError]) -> type:
    class RequestErrorClient:
        is_closed = False

        def __init__(self, **kwargs: Any) -> None:
            pass

        async def post(
            self,
            url: str,
            *,
            headers: dict[str, str],
            json: dict[str, Any],
            timeout: Any = None,
        ) -> None:
            request = httpx.Request("POST", url)
            raise error_type("forced request failure", request=request)

    return RequestErrorClient


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("12", 12.0),
        ("2.5", 2.5),
        ("Wed, 21 Oct 2015 07:28:10 GMT", 10.0),
        ("Wed, 21 Oct 2015 07:27:40 GMT", 0.0),
        ("", None),
        ("-1", None),
        ("nan", None),
        ("inf", None),
        ("not-a-date", None),
    ],
)
def test_retry_after_seconds_supports_bounded_seconds_and_http_dates(
    value: str,
    expected: float | None,
) -> None:
    response = httpx.Response(429, headers={"retry-after": value})

    assert (
        retry_after_seconds(
            response,
            now=datetime(2015, 10, 21, 7, 28, tzinfo=UTC),
        )
        == expected
    )


def test_retry_after_seconds_returns_none_without_header() -> None:
    assert retry_after_seconds(httpx.Response(429)) is None


@pytest.mark.parametrize(
    ("error_type", "retryable"),
    [
        (httpx.ConnectTimeout, True),
        (httpx.ReadTimeout, True),
        (httpx.WriteTimeout, True),
        (httpx.PoolTimeout, True),
        (httpx.ConnectError, True),
        (httpx.ReadError, True),
        (httpx.WriteError, True),
        (httpx.CloseError, True),
        (httpx.RemoteProtocolError, True),
        (httpx.LocalProtocolError, False),
        (httpx.ProxyError, False),
        (httpx.UnsupportedProtocol, False),
        (httpx.DecodingError, False),
        (httpx.TooManyRedirects, False),
    ],
)
@pytest.mark.anyio
async def test_openai_transport_classifies_only_transient_request_errors_as_retryable(
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[httpx.RequestError],
    retryable: bool,
) -> None:
    monkeypatch.setattr(
        "cayu.providers._http.httpx.AsyncClient",
        _request_error_client_factory(error_type),
    )

    with pytest.raises(OpenAIAPIError) as captured:
        await HttpxOpenAITransport().create_response(
            url="https://api.openai.com/v1/responses",
            headers={},
            payload={},
            timeout_s=1.0,
        )

    assert captured.value.error_type == error_type.__name__
    assert captured.value.retryable is retryable


@pytest.mark.anyio
async def test_openai_transport_omits_arbitrary_request_error_subclass_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "transport-subclass-secret-canary-ABCDEFGHIJKLMNOP"
    secret_error = type(secret, (httpx.RequestError,), {})
    monkeypatch.setattr(
        "cayu.providers._http.httpx.AsyncClient",
        _request_error_client_factory(secret_error),
    )

    with pytest.raises(OpenAIAPIError) as captured:
        await HttpxOpenAITransport().create_response(
            url="https://api.openai.com/v1/responses",
            headers={},
            payload={},
            timeout_s=1.0,
        )

    assert captured.value.error_type == "RequestError"
    assert secret not in captured.value.error_type


_KEEPALIVE_LINES = [
    ": keepalive",
    ": keepalive",
    ": keepalive",
    ": keepalive",
    ": keepalive",
    'data: {"ok": true}',
    "",
    "data: [DONE]",
    "",
]


@pytest.mark.anyio
async def test_openai_transport_classifies_established_sse_reset_separately_from_deadlines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = _ResetAfterEventByteStream()
    response = httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        stream=stream,
        request=httpx.Request("POST", "https://provider.example/v1/stream"),
    )
    monkeypatch.setattr("cayu.providers._http.httpx.AsyncClient", _client_factory(response))
    events = HttpxOpenAITransport().stream_response_events(
        url="https://api.openai.com/v1/responses",
        headers={},
        payload={},
        timeout_s=1.0,
        transport_idle_timeout_s=1.0,
        protocol_idle_timeout_s=1.0,
        semantic_progress_timeout_s=1.0,
        absolute_stream_timeout_s=1.0,
    )

    assert await anext(events) == {
        "type": "response.created",
        "response": {"id": "resp-reset"},
    }
    with pytest.raises(OpenAIAPIError) as captured:
        await anext(events)

    assert type(captured.value) is OpenAIAPIError
    assert captured.value.error_type == "RemoteProtocolError"
    assert captured.value.retryable is False
    assert captured.value.status_code is None
    assert stream.closed is True


@pytest.mark.anyio
async def test_openai_transport_survives_keepalive_heartbeats(monkeypatch) -> None:
    # Five heartbeats, each within one transport-idle window (0.04 < 0.1)
    # but 0.2s in total. Comments prove transport activity without claiming
    # decoded protocol or semantic progress.
    response = _StreamingResponse(_KEEPALIVE_LINES, heartbeat_sleep_s=0.04)
    monkeypatch.setattr("cayu.providers._http.httpx.AsyncClient", _client_factory(response))

    events = [
        event
        async for event in HttpxOpenAITransport().stream_response_events(
            url="https://api.openai.com/v1/responses",
            headers={},
            payload={},
            timeout_s=1.0,
            transport_idle_timeout_s=0.1,
            protocol_idle_timeout_s=1.0,
            semantic_progress_timeout_s=1.0,
            absolute_stream_timeout_s=1.0,
        )
    ]

    assert events == [{"ok": True}]


@pytest.mark.anyio
async def test_byte_active_heartbeat_only_sse_crosses_semantic_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _StreamingResponse(
        [item for _ in range(40) for item in (": keepalive", "")],
        heartbeat_sleep_s=0.005,
    )
    monkeypatch.setattr("cayu.providers._http.httpx.AsyncClient", _client_factory(response))
    provider = ChatCompletionsProvider(
        api_key="test-key",
        transport_idle_timeout_s=0.1,
        protocol_idle_timeout_s=1.0,
        semantic_progress_timeout_s=0.03,
        absolute_stream_timeout_s=1.0,
    )

    with pytest.raises(ModelStreamDeadlineError) as captured:
        async for _ in provider.runtime_stream(
            ModelRequest(model="test-model", messages=[Message.text("user", "hello")])
        ):
            pass

    assert captured.value.deadline_evidence.deadline_kind is ProviderDeadlineKind.SEMANTIC_IDLE
    assert response.byte_stream.closed is True


@pytest.mark.anyio
async def test_semantically_active_real_sse_is_bounded_by_absolute_lifetime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lines = [
        'data: {"type":"response.created","response":{"id":"resp-absolute"}}',
        "",
    ]
    for output_index in range(12):
        lines.extend(
            [
                (
                    'data: {"type":"response.output_text.delta",'
                    f'"output_index":{output_index},"content_index":0,"delta":"x"}}'
                ),
                "",
            ]
        )
    response = _StreamingResponse(lines, heartbeat_sleep_s=0.04)
    monkeypatch.setattr("cayu.providers._http.httpx.AsyncClient", _client_factory(response))
    provider = OpenAIProvider(
        api_key="test-key",
        transport_idle_timeout_s=0.15,
        protocol_idle_timeout_s=0.2,
        semantic_progress_timeout_s=0.2,
        absolute_stream_timeout_s=0.55,
    )

    with pytest.raises(ModelStreamDeadlineError) as captured:
        async for _ in provider.runtime_stream(
            ModelRequest(model="test-model", messages=[Message.text("user", "hello")])
        ):
            pass

    evidence = captured.value.deadline_evidence
    assert evidence.deadline_kind is ProviderDeadlineKind.ABSOLUTE
    assert evidence.elapsed_s >= 0.5
    assert evidence.last_progress_kind is not None
    assert response.byte_stream.closed is True


@pytest.mark.anyio
async def test_chat_completions_transport_survives_keepalive_heartbeats(monkeypatch) -> None:
    response = _StreamingResponse(_KEEPALIVE_LINES, heartbeat_sleep_s=0.04)
    monkeypatch.setattr("cayu.providers._http.httpx.AsyncClient", _client_factory(response))

    events = [
        event
        async for event in HttpxChatCompletionsTransport().stream_chat_completions(
            url="https://api.openai.com/v1/chat/completions",
            headers={},
            payload={},
            timeout_s=1.0,
            transport_idle_timeout_s=0.1,
            protocol_idle_timeout_s=1.0,
            semantic_progress_timeout_s=1.0,
            absolute_stream_timeout_s=1.0,
        )
    ]

    assert events == [{"ok": True}]


@pytest.mark.anyio
@pytest.mark.parametrize("line_ending", [b"\n", b"\r\n", b"\r"])
async def test_bounded_sse_lines_preserve_framing_and_exact_utf8_limit(
    line_ending: bytes,
) -> None:
    encoded_line = 'data: {"value":"é"}'.encode()
    payload = encoded_line + line_ending + line_ending
    chunks = [payload[:7], payload[7:-1], payload[-1:]]

    lines = [
        line
        async for line in _aiter_bounded_sse_lines(
            _iter_byte_chunks(chunks),
            max_line_bytes=len(encoded_line),
            provider_label="OpenAI",
        )
    ]

    assert lines == ['data: {"value":"é"}', ""]


@pytest.mark.anyio
async def test_bounded_sse_lines_retain_replacement_decoding() -> None:
    lines = [
        line
        async for line in _aiter_bounded_sse_lines(
            _iter_byte_chunks([b"data: bad-\xff", b"\n"]),
            max_line_bytes=32,
            provider_label="OpenAI",
        )
    ]

    assert lines == ["data: bad-\ufffd"]


@pytest.mark.anyio
@pytest.mark.parametrize("line_ending", [b"\n", b"\r"])
async def test_bounded_sse_lines_tokenize_dense_delimiters_cooperatively(
    line_ending: bytes,
) -> None:
    progressed = asyncio.Event()

    async def mark_scheduled_progress() -> None:
        progressed.set()

    marker = asyncio.create_task(mark_scheduled_progress())
    line_count = 0
    try:
        async for line in _aiter_bounded_sse_lines(
            _iter_byte_chunks([line_ending * (64 * 1024)]),
            max_line_bytes=16,
            provider_label="OpenAI",
        ):
            assert line == ""
            line_count += 1

        assert line_count == 64 * 1024
        assert progressed.is_set()
    finally:
        await marker


@pytest.mark.anyio
async def test_http_sse_bounds_unterminated_line_before_line_accumulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("cayu.providers._http.DEFAULT_SSE_MAX_EVENT_BYTES", 16)
    stream = _ChunkedByteStream([b"data", b": 12", b"3456", b"7890", b"more", b"unread"])

    with pytest.raises(OpenAIAPIError) as captured:
        await _stream_mock_sse(stream)

    assert captured.value.error_type == "SseEventLimitError"
    assert captured.value.retryable is False
    assert isinstance(captured.value.__cause__, SseEventLimitError)
    assert "SSE line exceeded the 16-byte limit" in str(captured.value.__cause__)
    assert stream.yielded_chunks == 5
    assert stream.closed


@pytest.mark.anyio
async def test_http_sse_rejects_compression_before_reading_expanding_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("cayu.providers._http.DEFAULT_SSE_MAX_EVENT_BYTES", 16)
    compressed = gzip.compress(b"x" * 17)
    stream = _ChunkedByteStream([compressed])

    with pytest.raises(OpenAIProtocolError, match="unsupported content encoding"):
        await _stream_mock_sse(stream, headers={"content-encoding": "gzip"})

    assert stream.yielded_chunks == 0
    assert stream.closed


@pytest.mark.anyio
@pytest.mark.parametrize("status_code", [429, 500])
async def test_http_sse_classifies_compressed_error_without_reading_body(
    status_code: int,
) -> None:
    compressed = gzip.compress(b'{"error":{"message":"temporary failure"}}')
    stream = _ChunkedByteStream([compressed])

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            headers={
                "content-encoding": "gzip",
                "content-type": "application/json",
                "retry-after": "7.5",
            },
            stream=stream,
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        events = stream_sse_json_events(
            client=client,
            url="https://provider.example/v1/stream",
            headers={},
            payload={},
            timeout_s=1.0,
            transport_idle_timeout_s=1.0,
            protocol_idle_timeout_s=1.0,
            semantic_progress_timeout_s=1.0,
            absolute_stream_timeout_s=1.0,
            request_label="OpenAI API",
            response_label="OpenAI",
            api_error=OpenAIAPIError,
            protocol_error=OpenAIProtocolError,
            # Accessing text would require reading and decoding the body. This
            # callback must remain unused for unsupported content encodings.
            error_response_text=lambda response: response.text,
        )
        with pytest.raises(OpenAIAPIError) as captured:
            await events.__anext__()

    error = captured.value
    assert error.status_code == status_code
    assert error.retry_after_s == 7.5
    assert error.response_body is None
    assert stream.yielded_chunks == 0
    assert stream.closed


@pytest.mark.anyio
async def test_http_sse_fails_closed_when_error_response_close_fails() -> None:
    class FailingCloseByteStream(_ChunkedByteStream):
        async def aclose(self) -> None:
            self.closed = True
            raise httpx.CloseError("provider error response close failed")

    stream = FailingCloseByteStream([b"compressed body must remain unread"])

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            headers={
                "content-encoding": "gzip",
                "content-type": "application/json",
            },
            stream=stream,
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        events = stream_sse_json_events(
            client=client,
            url="https://provider.example/v1/stream",
            headers={},
            payload={},
            timeout_s=1.0,
            transport_idle_timeout_s=1.0,
            protocol_idle_timeout_s=1.0,
            semantic_progress_timeout_s=1.0,
            absolute_stream_timeout_s=1.0,
            request_label="OpenAI API",
            response_label="OpenAI",
            api_error=OpenAIAPIError,
            protocol_error=OpenAIProtocolError,
            error_response_text=lambda response: response.text,
        )
        with pytest.raises(ProviderStreamCleanupError) as captured:
            await events.__anext__()

    assert captured.value.status_code == 401
    assert captured.value.retryable is False
    assert stream.yielded_chunks == 0
    assert stream.closed


@pytest.mark.anyio
async def test_http_sse_bounds_identity_error_body_before_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("cayu.providers._http.MAX_PROVIDER_ERROR_BODY_BYTES", 16)
    stream = _ChunkedByteStream([b"x" * 17])
    body_accessed = False

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"content-type": "application/json", "retry-after": "7.5"},
            stream=stream,
            request=request,
        )

    def error_response_text(response: httpx.Response) -> str:
        nonlocal body_accessed
        body_accessed = True
        return response.text

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        events = stream_sse_json_events(
            client=client,
            url="https://provider.example/v1/stream",
            headers={},
            payload={},
            timeout_s=1.0,
            transport_idle_timeout_s=1.0,
            protocol_idle_timeout_s=1.0,
            semantic_progress_timeout_s=1.0,
            absolute_stream_timeout_s=1.0,
            request_label="OpenAI API",
            response_label="OpenAI",
            api_error=OpenAIAPIError,
            protocol_error=OpenAIProtocolError,
            error_response_text=error_response_text,
        )
        with pytest.raises(OpenAIAPIError) as captured:
            await events.__anext__()

    error = captured.value
    assert error.status_code == 429
    assert error.retry_after_s == 7.5
    assert error.response_body is None
    assert not body_accessed
    assert stream.yielded_chunks == 1
    assert stream.closed


@pytest.mark.anyio
async def test_http_sse_bounds_active_never_ending_identity_error_body() -> None:
    stream = _EndlessByteStream()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            headers={"content-type": "application/json", "retry-after": "3"},
            stream=stream,
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        events = stream_sse_json_events(
            client=client,
            url="https://provider.example/v1/stream",
            headers={},
            payload={},
            timeout_s=0.02,
            transport_idle_timeout_s=0.01,
            protocol_idle_timeout_s=0.01,
            semantic_progress_timeout_s=0.01,
            absolute_stream_timeout_s=0.01,
            request_label="OpenAI API",
            response_label="OpenAI",
            api_error=OpenAIAPIError,
            protocol_error=OpenAIProtocolError,
            error_response_text=lambda response: response.text,
        )
        with pytest.raises(OpenAIAPIError) as captured:
            await asyncio.wait_for(events.__anext__(), timeout=0.5)

    error = captured.value
    assert error.status_code == 500
    assert error.retry_after_s == 3.0
    assert error.response_body is None
    assert stream.yielded_chunks > 0
    assert stream.closed


@pytest.mark.anyio
async def test_http_sse_bounds_stalled_identity_error_body_by_idle_timeout() -> None:
    stream = _BlockingByteStream()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            headers={"content-type": "application/json", "retry-after": "2"},
            stream=stream,
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        events = stream_sse_json_events(
            client=client,
            url="https://provider.example/v1/stream",
            headers={},
            payload={},
            timeout_s=1.0,
            transport_idle_timeout_s=0.01,
            protocol_idle_timeout_s=0.01,
            semantic_progress_timeout_s=0.01,
            absolute_stream_timeout_s=0.01,
            request_label="OpenAI API",
            response_label="OpenAI",
            api_error=OpenAIAPIError,
            protocol_error=OpenAIProtocolError,
            error_response_text=lambda response: response.text,
        )
        with pytest.raises(OpenAIAPIError) as captured:
            await asyncio.wait_for(events.__anext__(), timeout=0.5)

    error = captured.value
    assert error.status_code == 503
    assert error.retry_after_s == 2.0
    assert error.response_body is None
    assert stream.started.is_set()
    assert stream.finalized.is_set()
    assert stream.closed


@pytest.mark.anyio
async def test_http_sse_error_body_reader_propagates_real_task_cancellation() -> None:
    stream = _BlockingByteStream()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            headers={"content-type": "application/json"},
            stream=stream,
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        events = stream_sse_json_events(
            client=client,
            url="https://provider.example/v1/stream",
            headers={},
            payload={},
            timeout_s=1.0,
            transport_idle_timeout_s=1.0,
            protocol_idle_timeout_s=1.0,
            semantic_progress_timeout_s=1.0,
            absolute_stream_timeout_s=1.0,
            request_label="OpenAI API",
            response_label="OpenAI",
            api_error=OpenAIAPIError,
            protocol_error=OpenAIProtocolError,
            error_response_text=lambda response: response.text,
        )
        task = asyncio.create_task(events.__anext__())
        await stream.started.wait()
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelling() == 1
        assert task.cancelled()

    assert stream.finalized.is_set()
    assert stream.closed


@pytest.mark.anyio
async def test_http_sse_preserves_error_status_when_identity_body_read_fails() -> None:
    class FailingByteStream(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.closed = False

        async def __aiter__(self):
            raise httpx.ReadError("error response body disconnected")
            yield b""  # pragma: no cover

        async def aclose(self) -> None:
            self.closed = True

    stream = FailingByteStream()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            headers={"content-type": "application/json"},
            stream=stream,
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        events = stream_sse_json_events(
            client=client,
            url="https://provider.example/v1/stream",
            headers={},
            payload={},
            timeout_s=1.0,
            transport_idle_timeout_s=1.0,
            protocol_idle_timeout_s=1.0,
            semantic_progress_timeout_s=1.0,
            absolute_stream_timeout_s=1.0,
            request_label="OpenAI API",
            response_label="OpenAI",
            api_error=OpenAIAPIError,
            protocol_error=OpenAIProtocolError,
            error_response_text=lambda response: response.text,
        )
        with pytest.raises(OpenAIAPIError) as captured:
            await events.__anext__()

    assert captured.value.status_code == 401
    assert captured.value.response_body is None
    assert stream.closed


@pytest.mark.anyio
async def test_http_sse_forces_identity_encoding_case_insensitively() -> None:
    stream = _ChunkedByteStream([b'data: {"ok": true}\n\n'])
    requested_headers: httpx.Headers | None = None

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requested_headers
        requested_headers = request.headers
        return httpx.Response(
            200,
            headers={"content-encoding": "identity"},
            stream=stream,
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        events = [
            event
            async for event in stream_sse_json_events(
                client=client,
                url="https://provider.example/v1/stream",
                headers={"aCcEpT-EnCoDiNg": "gzip", "X-Test": "kept"},
                payload={},
                timeout_s=1.0,
                transport_idle_timeout_s=1.0,
                protocol_idle_timeout_s=1.0,
                semantic_progress_timeout_s=1.0,
                absolute_stream_timeout_s=1.0,
                request_label="OpenAI API",
                response_label="OpenAI",
                api_error=OpenAIAPIError,
                protocol_error=OpenAIProtocolError,
                error_response_text=lambda response: response.text,
            )
        ]

    assert events == [{"ok": True}]
    assert requested_headers is not None
    assert requested_headers.get_list("accept-encoding") == ["identity"]
    assert requested_headers["x-test"] == "kept"
    assert stream.closed


@pytest.mark.anyio
async def test_http_sse_preserves_only_trusted_success_retry_after_metadata() -> None:
    stream = _ChunkedByteStream([b'data: {"retry_after_s": 999, "ok": true}\n\n'])

    events = await _stream_mock_sse(stream, headers={"retry-after": "7.5"})

    assert events == [{"retry_after_s": 999, "ok": True}]
    assert json.loads(json.dumps(events[0])) == {"retry_after_s": 999, "ok": True}
    assert _trusted_sse_retry_after_s(events[0]) == 7.5
    assert _trusted_sse_retry_after_s({"retry_after_s": 999}) is None


@pytest.mark.anyio
async def test_http_sse_counts_fragmented_raw_bytes_as_idle_activity() -> None:
    payload = b'data: {"ok": true}\n\n'
    stream = _DelayedChunkStream(
        [payload[:4], payload[4:8], payload[8:12], payload[12:16], payload[16:]],
        delay_s=0.02,
    )

    events = await _stream_mock_sse(stream, transport_idle_timeout_s=0.05)

    assert events == [{"ok": True}]
    assert stream.closed


@pytest.mark.anyio
async def test_http_sse_bounds_continuously_progressing_partial_line_by_protocol_deadline() -> None:
    stream = _DelayedChunkStream([b"x"] * 12, delay_s=0.02)

    with pytest.raises(ModelStreamDeadlineError) as captured:
        await _stream_mock_sse(
            stream,
            transport_idle_timeout_s=0.05,
            protocol_idle_timeout_s=0.05,
            semantic_progress_timeout_s=1.0,
            absolute_stream_timeout_s=1.0,
        )

    assert captured.value.retryable is False
    assert captured.value.deadline_evidence.deadline_kind is ProviderDeadlineKind.PROTOCOL_IDLE
    assert stream.closed


@pytest.mark.anyio
async def test_http_sse_yields_small_event_without_waiting_for_more_bytes() -> None:
    stream = _BlockingAfterEventByteStream()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        events = stream_sse_json_events(
            client=client,
            url="https://provider.example/v1/stream",
            headers={},
            payload={},
            timeout_s=1.0,
            transport_idle_timeout_s=1.0,
            protocol_idle_timeout_s=1.0,
            semantic_progress_timeout_s=1.0,
            absolute_stream_timeout_s=1.0,
            request_label="OpenAI API",
            response_label="OpenAI",
            api_error=OpenAIAPIError,
            protocol_error=OpenAIProtocolError,
            error_response_text=lambda response: response.text,
        )

        assert await asyncio.wait_for(events.__anext__(), timeout=0.1) == {"ok": True}
        await events.aclose()

    assert stream.closed


@pytest.mark.anyio
async def test_http_sse_byte_reader_propagates_real_task_cancellation() -> None:
    stream = _BlockingByteStream()
    task = asyncio.create_task(_stream_mock_sse(stream))
    await stream.started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelling() == 1
    assert task.cancelled()
    assert stream.finalized.is_set()
    assert stream.closed


@pytest.mark.parametrize(
    ("transport_type", "stream_method", "api_error_type"),
    [
        pytest.param(
            HttpxOpenAITransport,
            "stream_response_events",
            OpenAIAPIError,
            id="openai",
        ),
        pytest.param(
            HttpxAnthropicTransport,
            "stream_message_events",
            AnthropicAPIError,
            id="anthropic",
        ),
        pytest.param(
            HttpxChatCompletionsTransport,
            "stream_chat_completions",
            ChatCompletionsAPIError,
            id="chat-completions",
        ),
        pytest.param(
            HttpxVertexTransport,
            "stream_message_events",
            VertexAPIError,
            id="vertex",
        ),
    ],
)
@pytest.mark.anyio
async def test_http_sse_transport_timeout_is_typed_and_nonretryable(
    monkeypatch: pytest.MonkeyPatch,
    transport_type: type,
    stream_method: str,
    api_error_type: type[Exception],
) -> None:
    response = _StreamingResponse([""], heartbeat_sleep_s=0.05)
    monkeypatch.setattr("cayu.providers._http.httpx.AsyncClient", _client_factory(response))

    transport = transport_type()
    stream = getattr(transport, stream_method)(
        url="https://provider.example/v1/stream",
        headers={},
        payload={},
        timeout_s=1.0,
        transport_idle_timeout_s=0.001,
        protocol_idle_timeout_s=1.0,
        semantic_progress_timeout_s=1.0,
        absolute_stream_timeout_s=1.0,
    )
    with pytest.raises(ModelStreamDeadlineError) as captured:
        await stream.__anext__()

    assert not isinstance(captured.value, api_error_type)
    assert captured.value.retryable is False
    assert captured.value.deadline_evidence.deadline_kind is ProviderDeadlineKind.TRANSPORT_IDLE


@pytest.mark.anyio
async def test_http_sse_protocol_errors_remain_permanent(monkeypatch) -> None:
    response = _StreamingResponse(["data: not-json", ""])
    monkeypatch.setattr("cayu.providers._http.httpx.AsyncClient", _client_factory(response))

    stream = HttpxOpenAITransport().stream_response_events(
        url="https://api.openai.com/v1/responses",
        headers={},
        payload={},
        timeout_s=1.0,
        transport_idle_timeout_s=1.0,
        protocol_idle_timeout_s=1.0,
        semantic_progress_timeout_s=1.0,
        absolute_stream_timeout_s=1.0,
    )
    with pytest.raises(OpenAIProtocolError):
        await stream.__anext__()


@pytest.mark.anyio
async def test_http_sse_event_limit_is_a_typed_nonretryable_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def limited_events(*args: Any, **kwargs: Any):
        raise SseEventLimitError("OpenAI SSE event exceeded its fixed limit.")
        yield  # pragma: no cover

    response = _StreamingResponse([])
    monkeypatch.setattr("cayu.providers._http.httpx.AsyncClient", _client_factory(response))
    monkeypatch.setattr("cayu.providers._http.aiter_sse_json_events", limited_events)

    stream = HttpxOpenAITransport().stream_response_events(
        url="https://api.openai.com/v1/responses",
        headers={},
        payload={},
        timeout_s=1.0,
        transport_idle_timeout_s=1.0,
        protocol_idle_timeout_s=1.0,
        semantic_progress_timeout_s=1.0,
        absolute_stream_timeout_s=1.0,
    )
    with pytest.raises(OpenAIAPIError) as captured:
        await stream.__anext__()

    assert captured.value.retryable is False
    assert captured.value.error_type == "SseEventLimitError"
    assert isinstance(captured.value.__cause__, SseEventLimitError)


@pytest.mark.anyio
async def test_http_sse_event_timeout_is_a_typed_retryable_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def timed_out_events(*args: Any, **kwargs: Any):
        raise SseEventTimeoutError("OpenAI SSE event exceeded its duration limit.")
        yield  # pragma: no cover

    response = _StreamingResponse([])
    monkeypatch.setattr("cayu.providers._http.httpx.AsyncClient", _client_factory(response))
    monkeypatch.setattr("cayu.providers._http.aiter_sse_json_events", timed_out_events)

    stream = HttpxOpenAITransport().stream_response_events(
        url="https://api.openai.com/v1/responses",
        headers={},
        payload={},
        timeout_s=1.0,
        transport_idle_timeout_s=1.0,
        protocol_idle_timeout_s=1.0,
        semantic_progress_timeout_s=1.0,
        absolute_stream_timeout_s=1.0,
    )
    with pytest.raises(OpenAIAPIError) as captured:
        await stream.__anext__()

    assert captured.value.retryable is True
    assert captured.value.error_type == "SseEventTimeoutError"
    assert isinstance(captured.value.__cause__, SseEventTimeoutError)


@pytest.mark.anyio
async def test_sse_parser_rejects_nonpositive_protocol_idle_timeout() -> None:
    async def lines():
        yield ""

    with pytest.raises(ValueError, match="protocol_idle_timeout_s"):
        [
            event
            async for event in aiter_sse_json_events(
                lines(),
                deadline_controller=_deadline_controller(0),
                provider_label="OpenAI",
                protocol_error=OpenAIProtocolError,
            )
        ]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        pytest.param("protocol_idle_timeout_s", float("nan"), id="idle-nan"),
        pytest.param("protocol_idle_timeout_s", float("inf"), id="idle-infinity"),
        pytest.param("max_event_duration_s", float("nan"), id="duration-nan"),
        pytest.param("max_event_duration_s", float("inf"), id="duration-infinity"),
    ],
)
async def test_sse_parser_rejects_non_finite_timeouts(
    field_name: str,
    value: float,
) -> None:
    async def lines():
        yield ""

    protocol_idle_timeout_s = value if field_name == "protocol_idle_timeout_s" else 1.0
    max_event_duration_s = value if field_name == "max_event_duration_s" else None
    with pytest.raises(ValueError, match=field_name):
        [
            event
            async for event in aiter_sse_json_events(
                lines(),
                deadline_controller=_deadline_controller(protocol_idle_timeout_s),
                max_event_duration_s=max_event_duration_s,
                provider_label="OpenAI",
                protocol_error=OpenAIProtocolError,
            )
        ]


@pytest.mark.parametrize(
    ("provider_name", "provider_factory"),
    [
        (
            "openai",
            lambda field, value: OpenAIProvider(api_key="test-key", **{field: value}),
        ),
        (
            "anthropic",
            lambda field, value: AnthropicProvider(api_key="test-key", **{field: value}),
        ),
        (
            "chat_completions",
            lambda field, value: ChatCompletionsProvider(api_key="test-key", **{field: value}),
        ),
        (
            "vertex",
            lambda field, value: VertexProvider(
                project_id="test-project",
                credentials=SimpleNamespace(valid=True, token="test-token"),
                **{field: value},
            ),
        ),
        (
            "openai_subscription",
            lambda field, value: OpenAISubscriptionProvider(
                auth=_UnusedSubscriptionAuth(), **{field: value}
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "field_name",
    [
        "transport_idle_timeout_s",
        "protocol_idle_timeout_s",
        "semantic_progress_timeout_s",
        "absolute_stream_timeout_s",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="infinity"),
    ],
)
def test_public_sse_providers_reject_non_finite_idle_timeout(
    provider_name: str,
    provider_factory: Callable[[str, float], object],
    field_name: str,
    value: float,
) -> None:
    del provider_name
    with pytest.raises(ValueError, match=field_name):
        provider_factory(field_name, value)


@pytest.mark.parametrize(
    "provider_factory",
    [
        pytest.param(
            lambda **kwargs: OpenAIProvider(api_key="test-key", **kwargs),
            id="openai",
        ),
        pytest.param(
            lambda **kwargs: OpenAISubscriptionProvider(
                auth=_UnusedSubscriptionAuth(),
                **kwargs,
            ),
            id="openai-subscription",
        ),
        pytest.param(
            lambda **kwargs: ChatCompletionsProvider(api_key="test-key", **kwargs),
            id="chat-completions",
        ),
        pytest.param(
            lambda **kwargs: AnthropicProvider(api_key="test-key", **kwargs),
            id="anthropic",
        ),
        pytest.param(
            lambda **kwargs: BedrockProvider(client=object(), **kwargs),
            id="bedrock",
        ),
        pytest.param(
            lambda **kwargs: VertexProvider(
                project_id="test-project",
                credentials=SimpleNamespace(valid=True, token="test-token"),
                **kwargs,
            ),
            id="vertex",
        ),
    ],
)
def test_bundled_provider_constructors_preserve_stream_idle_timeout_alias(
    provider_factory: Callable[..., Any],
) -> None:
    provider = provider_factory(
        stream_idle_timeout_s=7.0,
        absolute_stream_timeout_s=11.0,
    )

    assert provider.stream_deadlines == ProviderStreamDeadlines(
        transport_idle_timeout_s=7.0,
        protocol_idle_timeout_s=7.0,
        semantic_progress_timeout_s=7.0,
        absolute_stream_timeout_s=11.0,
    )


def test_stream_idle_timeout_alias_rejects_conflicting_new_idle_clock() -> None:
    with pytest.raises(ValueError, match="stream_idle_timeout_s cannot be combined"):
        OpenAIProvider(
            api_key="test-key",
            stream_idle_timeout_s=7.0,
            semantic_progress_timeout_s=8.0,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.0, -1.0])
def test_stream_idle_timeout_alias_requires_positive_finite_value(value: float) -> None:
    with pytest.raises(ValueError, match="stream_idle_timeout_s"):
        OpenAIProvider(api_key="test-key", stream_idle_timeout_s=value)


@pytest.mark.anyio
async def test_sse_parser_lines_do_not_refresh_protocol_progress() -> None:
    async def lines():
        yield 'data: {"value":'
        await asyncio.sleep(0.04)
        yield "data: 1"
        await asyncio.sleep(0.04)
        yield "data: }"
        await asyncio.sleep(0.04)
        yield ""

    with pytest.raises(ProviderStreamDeadlineExceeded) as captured:
        [
            event
            async for event in aiter_sse_json_events(
                lines(),
                deadline_controller=_deadline_controller(0.1),
                provider_label="OpenAI",
                protocol_error=OpenAIProtocolError,
            )
        ]

    assert captured.value.evidence.deadline_kind is ProviderDeadlineKind.PROTOCOL_IDLE


@pytest.mark.anyio
async def test_sse_parser_does_not_count_downstream_processing_as_idle_time() -> None:
    stream = aiter_sse_json_events(
        _iter_lines(['data: {"value": 1}', "", 'data: {"value": 2}', ""]),
        deadline_controller=_deadline_controller(0.01),
        provider_label="OpenAI",
        protocol_error=OpenAIProtocolError,
    )

    assert await stream.__anext__() == {"value": 1}
    await asyncio.sleep(0.02)
    assert await stream.__anext__() == {"value": 2}


@pytest.mark.anyio
async def test_sse_parser_does_not_relabel_source_timeout_as_its_idle_deadline() -> None:
    source_error = TimeoutError("source-owned timeout")

    async def lines():
        raise source_error
        yield ""  # pragma: no cover

    with pytest.raises(TimeoutError, match="source-owned timeout") as captured:
        [
            event
            async for event in aiter_sse_json_events(
                lines(),
                deadline_controller=_deadline_controller(1.0),
                provider_label="OpenAI",
                protocol_error=OpenAIProtocolError,
            )
        ]

    assert captured.value is source_error


@pytest.mark.anyio
async def test_sse_parser_event_duration_does_not_reset_with_activity() -> None:
    async def lines():
        yield 'data: {"value":'
        for _ in range(10):
            await asyncio.sleep(0.01)
            yield "data: 1"

    with pytest.raises(SseEventTimeoutError, match="did not finish one SSE event"):
        [
            event
            async for event in aiter_sse_json_events(
                lines(),
                deadline_controller=_deadline_controller(1.0),
                max_event_duration_s=0.05,
                provider_label="OpenAI",
                protocol_error=OpenAIProtocolError,
            )
        ]


@pytest.mark.anyio
async def test_sse_parser_enforces_utf8_byte_limit_at_exact_boundary() -> None:
    line = 'data: {"value":"é"}'
    encoded_size = len(line.encode("utf-8"))

    events = [
        event
        async for event in aiter_sse_json_events(
            _iter_lines([line, ""]),
            deadline_controller=_deadline_controller(1.0),
            max_event_bytes=encoded_size,
            provider_label="OpenAI",
            protocol_error=OpenAIProtocolError,
        )
    ]
    assert events == [{"value": "é"}]

    with pytest.raises(SseEventLimitError, match=f"{encoded_size - 1}-byte limit"):
        [
            event
            async for event in aiter_sse_json_events(
                _iter_lines([line, ""]),
                deadline_controller=_deadline_controller(1.0),
                max_event_bytes=encoded_size - 1,
                provider_label="OpenAI",
                protocol_error=OpenAIProtocolError,
            )
        ]


@pytest.mark.anyio
async def test_sse_parser_enforces_event_line_limit_at_exact_boundary() -> None:
    lines = ['data: {"value":', "data: 1}", ""]
    events = [
        event
        async for event in aiter_sse_json_events(
            _iter_lines(lines),
            deadline_controller=_deadline_controller(1.0),
            max_event_lines=2,
            provider_label="OpenAI",
            protocol_error=OpenAIProtocolError,
        )
    ]
    assert events == [{"value": 1}]

    with pytest.raises(SseEventLimitError, match="2-line limit"):
        [
            event
            async for event in aiter_sse_json_events(
                _iter_lines(["event: response", *lines]),
                deadline_controller=_deadline_controller(1.0),
                max_event_lines=2,
                provider_label="OpenAI",
                protocol_error=OpenAIProtocolError,
            )
        ]


@pytest.mark.anyio
async def test_sse_parser_propagates_real_task_cancellation() -> None:
    read_started = asyncio.Event()
    source_finalized = asyncio.Event()

    async def lines():
        try:
            read_started.set()
            await asyncio.Event().wait()
            yield ""  # pragma: no cover
        finally:
            source_finalized.set()

    async def consume() -> list[Mapping[str, Any]]:
        return [
            event
            async for event in aiter_sse_json_events(
                lines(),
                deadline_controller=_deadline_controller(1.0),
                provider_label="OpenAI",
                protocol_error=OpenAIProtocolError,
            )
        ]

    task = asyncio.create_task(consume())
    await read_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelling() == 1
    assert task.cancelled()
    assert source_finalized.is_set()


@pytest.mark.anyio
async def test_sse_parser_raises_provider_labeled_protocol_errors() -> None:
    async def lines():
        yield "data: not-json"
        yield ""

    with pytest.raises(OpenAIProtocolError, match="OpenAI SSE data was not valid JSON"):
        [
            event
            async for event in aiter_sse_json_events(
                lines(),
                deadline_controller=_deadline_controller(1.0),
                provider_label="OpenAI",
                protocol_error=OpenAIProtocolError,
            )
        ]

    with pytest.raises(
        ChatCompletionsProtocolError,
        match="Chat Completions SSE data must decode to a JSON object",
    ):
        [
            event
            async for event in aiter_sse_json_events(
                _iter_lines(['data: ["not-an-object"]', ""]),
                deadline_controller=_deadline_controller(1.0),
                provider_label="Chat Completions",
                protocol_error=ChatCompletionsProtocolError,
            )
        ]


@pytest.mark.anyio
async def test_sse_parser_yields_trailing_data_without_blank_line() -> None:
    events = [
        event
        async for event in aiter_sse_json_events(
            _iter_lines(['data: {"tail": 1}']),
            deadline_controller=_deadline_controller(1.0),
            provider_label="OpenAI",
            protocol_error=OpenAIProtocolError,
        )
    ]
    assert events == [{"tail": 1}]


@pytest.mark.anyio
async def test_sse_parser_stops_at_done_marker() -> None:
    events = [
        event
        async for event in aiter_sse_json_events(
            _iter_lines(['data: {"n": 1}', "", "data: [DONE]", "", 'data: {"n": 2}', ""]),
            deadline_controller=_deadline_controller(1.0),
            provider_label="OpenAI",
            protocol_error=OpenAIProtocolError,
        )
    ]
    assert events == [{"n": 1}]


class _CountingClient:
    """Records how many httpx clients a transport constructs and closes."""

    constructed = 0
    closed = 0

    def __init__(self, **kwargs: Any) -> None:
        type(self).constructed += 1
        self.is_closed = False

    def stream(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
        timeout: Any = None,
    ) -> _StreamContext:
        return _StreamContext(_StreamingResponse(['data: {"ok": true}', "", "data: [DONE]", ""]))

    async def aclose(self) -> None:
        self.is_closed = True
        type(self).closed += 1


async def _drain_stream(transport: HttpxOpenAITransport) -> list[Mapping[str, Any]]:
    return [
        event
        async for event in transport.stream_response_events(
            url="https://api.openai.com/v1/responses",
            headers={},
            payload={},
            timeout_s=1.0,
            transport_idle_timeout_s=1.0,
            protocol_idle_timeout_s=1.0,
            semantic_progress_timeout_s=1.0,
            absolute_stream_timeout_s=1.0,
        )
    ]


@pytest.mark.anyio
async def test_transport_reuses_one_client_across_requests(monkeypatch) -> None:
    class Client(_CountingClient):
        constructed = 0
        closed = 0

    monkeypatch.setattr("cayu.providers._http.httpx.AsyncClient", Client)

    transport = HttpxOpenAITransport()
    assert await _drain_stream(transport) == [{"ok": True}]
    assert await _drain_stream(transport) == [{"ok": True}]

    # Two requests, one shared client: no fresh TLS handshake per request.
    assert Client.constructed == 1
    assert Client.closed == 0


@pytest.mark.anyio
async def test_transport_aclose_closes_shared_client_and_reopens(monkeypatch) -> None:
    class Client(_CountingClient):
        constructed = 0
        closed = 0

    monkeypatch.setattr("cayu.providers._http.httpx.AsyncClient", Client)

    transport = HttpxOpenAITransport()
    await _drain_stream(transport)
    await transport.aclose()
    assert Client.closed == 1

    # A request after aclose transparently recreates the client.
    await _drain_stream(transport)
    assert Client.constructed == 2

    # aclose on an already-closed / never-used transport is a harmless no-op.
    await transport.aclose()
    fresh = HttpxOpenAITransport()
    await fresh.aclose()


@pytest.mark.anyio
async def test_shared_async_client_is_lazy_and_recreates_after_close(monkeypatch) -> None:
    class Client(_CountingClient):
        constructed = 0
        closed = 0

    monkeypatch.setattr("cayu.providers._http.httpx.AsyncClient", Client)

    shared = SharedAsyncClient()
    # Constructing the holder opens no sockets.
    assert Client.constructed == 0

    first = shared.get()
    assert Client.constructed == 1
    # Repeated get() reuses the same live client.
    assert shared.get() is first
    assert Client.constructed == 1

    await shared.aclose()
    assert Client.closed == 1
    # After close, get() builds a fresh client rather than handing back a dead one.
    second = shared.get()
    assert second is not first
    assert Client.constructed == 2


async def _iter_byte_chunks(chunks: list[bytes]):
    for chunk in chunks:
        yield chunk


async def _iter_lines(lines: list[str]):
    for line in lines:
        yield line


def test_validate_url_uses_provider_label_and_https_default() -> None:
    assert (
        validate_url("https://api.openai.com", "url", provider_label="OpenAI")
        == "https://api.openai.com"
    )
    with pytest.raises(ValueError, match="OpenAI url must use https."):
        validate_url("http://api.openai.com", "url", provider_label="OpenAI")
    with pytest.raises(ValueError, match="must include a host"):
        validate_url("https://", "url", provider_label="OpenAI")


def test_validate_url_allow_http_opt_in_and_hint() -> None:
    assert (
        validate_url(
            "http://localhost:11434",
            "url",
            provider_label="Chat Completions",
            allow_http=True,
            allow_http_hint=True,
        )
        == "http://localhost:11434"
    )
    with pytest.raises(ValueError, match=r"set allow_http=True for local http servers"):
        validate_url(
            "http://localhost:11434",
            "url",
            provider_label="Chat Completions",
            allow_http_hint=True,
        )


def test_validate_base_url_strips_trailing_slash() -> None:
    assert (
        validate_base_url("https://api.anthropic.com/", provider_label="Anthropic")
        == "https://api.anthropic.com"
    )
