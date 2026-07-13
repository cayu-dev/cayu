from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from typing import Any
from unittest.mock import patch

from cayu.providers import (
    AnthropicAPIError,
    AnthropicContextOverflowError,
    AnthropicProvider,
    BedrockAPIError,
    BedrockContextOverflowError,
    BedrockProvider,
    ChatCompletionsAPIError,
    ChatCompletionsContextOverflowError,
    ChatCompletionsProvider,
    OpenAIAPIError,
    OpenAIContextOverflowError,
    OpenAIProvider,
    VertexAPIError,
    VertexContextOverflowError,
    VertexProvider,
)
from cayu.providers._sse import aiter_sse_json_events
from tests.providers.conformance import (
    CapabilityClaim,
    ProviderCapabilities,
    ProviderConformanceRegistration,
    ProviderHarness,
    ProviderScenario,
)


class _AsyncTransport:
    def __init__(self) -> None:
        self.closed = False
        self.started = asyncio.Event()
        self.stopped = asyncio.Event()
        self._release = asyncio.Event()

    async def aclose(self) -> None:
        self.closed = True
        self._release.set()

    async def block(self, *, idle_timeout_s: float | None = None) -> None:
        self.started.set()
        try:
            if idle_timeout_s is None:
                await self._release.wait()
                return

            async def silent_lines() -> AsyncIterator[str]:
                await self._release.wait()
                if False:
                    yield ""

            async for _ in aiter_sse_json_events(
                silent_lines(),
                idle_timeout_s=idle_timeout_s,
                provider_label="Conformance",
                protocol_error=ValueError,
            ):
                pass
        finally:
            self.stopped.set()


class _OpenAITransport(_AsyncTransport):
    def __init__(self, scenario: ProviderScenario) -> None:
        super().__init__()
        self.scenario = scenario
        self.calls: list[dict[str, Any]] = []

    async def stream_response_events(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
        stream_idle_timeout_s: float,
    ) -> AsyncIterator[Mapping[str, Any]]:
        del url, headers, timeout_s
        self.calls.append(dict(payload))
        if self.scenario == "cancellation":
            await self.block()
            return
        if self.scenario == "idle_timeout":
            await self.block(idle_timeout_s=stream_idle_timeout_s)
            return
        if self.scenario == "typed_error":
            raise OpenAIAPIError("conformance throttle", **_ERROR_FIELDS)
        if self.scenario == "context_overflow":
            raise OpenAIContextOverflowError("conformance context overflow", **_OVERFLOW_FIELDS)
        if self.scenario == "malformed":
            yield {"type": "response.output_text.delta", "delta": 42}
            return
        if self.scenario == "malformed_terminal":
            yield {
                "type": "response.completed",
                "response": {
                    "id": "resp-malformed-terminal",
                    "model": "gpt-conformance",
                    "status": 42,
                    "output": [],
                },
            }
            return
        if self.scenario == "malformed_usage":
            yield {
                "type": "response.completed",
                "response": {
                    "id": "resp-malformed-usage",
                    "model": "gpt-conformance",
                    "status": "completed",
                    "output": [],
                    "usage": [],
                },
            }
            return
        if self.scenario == "unfinished":
            for event in _openai_unfinished_events():
                yield event
            return
        if self.scenario == "unfinished_reasoning":
            for event in _openai_unfinished_reasoning_events():
                yield event
            return
        if self.scenario == "attachments":
            _require_attachment_payload(payload, marker="data:image/png;base64,aGVsbG8=")
            for event in _openai_text_events():
                yield event
            return
        if self.scenario == "reasoning":
            for event in _openai_reasoning_events():
                yield event
            return
        if self.scenario == "provider_cache_observation":
            for event in _openai_cache_events():
                yield event
            return
        if self.scenario == "tool_round_trip":
            events = _openai_tool_events() if len(self.calls) == 1 else _openai_final_events()
            if len(self.calls) == 2:
                _require_openai_tool_result(payload, tool_call_id="call-conformance")
            for event in events:
                yield event
            return
        if self.scenario not in {"text", "close"}:
            raise AssertionError(f"OpenAI scenario {self.scenario!r} is not implemented.")
        for event in _openai_text_events():
            yield event

    async def create_response(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        del url, headers, payload, timeout_s
        if self.scenario != "token_counting":
            raise AssertionError(f"OpenAI scenario {self.scenario!r} cannot count tokens.")
        return {"object": "response.input_tokens", "input_tokens": 13}


class _AnthropicShapeTransport(_AsyncTransport):
    def __init__(
        self,
        scenario: ProviderScenario,
        *,
        provider_label: str,
        api_error: Callable[..., Exception],
        context_overflow_error: Callable[..., Exception],
    ) -> None:
        super().__init__()
        self.scenario = scenario
        self.provider_label = provider_label
        self.api_error = api_error
        self.context_overflow_error = context_overflow_error
        self.calls: list[dict[str, Any]] = []

    async def stream_message_events(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
        stream_idle_timeout_s: float,
    ) -> AsyncIterator[Mapping[str, Any]]:
        del url, headers, timeout_s
        self.calls.append(dict(payload))
        if self.scenario == "cancellation":
            await self.block()
            return
        if self.scenario == "idle_timeout":
            await self.block(idle_timeout_s=stream_idle_timeout_s)
            return
        if self.scenario == "typed_error":
            raise self.api_error("conformance throttle", **_ERROR_FIELDS)
        if self.scenario == "context_overflow":
            raise self.context_overflow_error("conformance context overflow", **_OVERFLOW_FIELDS)
        if self.scenario == "malformed":
            yield {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "out-of-order"},
            }
            return
        if self.scenario == "malformed_terminal":
            yield {
                "type": "message_delta",
                "delta": {"stop_reason": 42},
                "usage": {"output_tokens": 2},
            }
            return
        if self.scenario == "malformed_usage":
            yield {
                "type": "message_start",
                "message": {
                    "id": "msg-malformed-usage",
                    "model": "claude-conformance",
                    "usage": [],
                },
            }
            yield {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 2},
            }
            yield {"type": "message_stop"}
            return
        if self.scenario == "unfinished":
            for event in _anthropic_unfinished_events(model="claude-conformance"):
                yield event
            return
        if self.scenario == "unfinished_reasoning":
            for event in _anthropic_unfinished_reasoning_events(model="claude-conformance"):
                yield event
            return
        if self.scenario == "attachments":
            _require_attachment_payload(payload, marker="aGVsbG8=")
            for event in _anthropic_text_events(model="claude-conformance"):
                yield event
            return
        if self.scenario == "reasoning":
            for event in _anthropic_reasoning_events(model="claude-conformance"):
                yield event
            return
        if self.scenario == "provider_cache_observation":
            for event in _anthropic_cache_events(model="claude-conformance"):
                yield event
            return
        if self.scenario == "tool_round_trip":
            events = (
                _anthropic_tool_events(model="claude-conformance")
                if len(self.calls) == 1
                else _anthropic_text_events(model="claude-conformance", deltas=("tool-observed",))
            )
            if len(self.calls) == 2:
                _require_anthropic_tool_result(payload, tool_call_id="call-conformance")
            for event in events:
                yield event
            return
        if self.scenario not in {"text", "close"}:
            raise AssertionError(
                f"{self.provider_label} scenario {self.scenario!r} is not implemented."
            )
        for event in _anthropic_text_events(model="claude-conformance"):
            yield event

    async def count_message_tokens(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        del url, headers, payload, timeout_s
        if self.scenario != "token_counting":
            raise AssertionError(
                f"{self.provider_label} scenario {self.scenario!r} cannot count tokens."
            )
        return {"input_tokens": 13}


class _AnthropicTransport(_AnthropicShapeTransport):
    def __init__(self, scenario: ProviderScenario) -> None:
        super().__init__(
            scenario,
            provider_label="Anthropic",
            api_error=AnthropicAPIError,
            context_overflow_error=AnthropicContextOverflowError,
        )


class _ChatCompletionsTransport(_AsyncTransport):
    def __init__(self, scenario: ProviderScenario) -> None:
        super().__init__()
        self.scenario = scenario
        self.calls: list[dict[str, Any]] = []

    async def stream_chat_completions(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
        stream_idle_timeout_s: float,
    ) -> AsyncIterator[Mapping[str, Any]]:
        del url, headers, timeout_s
        self.calls.append(dict(payload))
        if self.scenario == "cancellation":
            await self.block()
            return
        if self.scenario == "idle_timeout":
            await self.block(idle_timeout_s=stream_idle_timeout_s)
            return
        if self.scenario == "typed_error":
            raise ChatCompletionsAPIError("conformance throttle", **_ERROR_FIELDS)
        if self.scenario == "context_overflow":
            raise ChatCompletionsContextOverflowError(
                "conformance context overflow", **_OVERFLOW_FIELDS
            )
        if self.scenario == "malformed":
            yield {
                "id": "chat-malformed",
                "object": "chat.completion.chunk",
                "model": "chat-conformance",
                "choices": [{"index": 0, "delta": {}, "finish_reason": 42}],
            }
            return
        if self.scenario == "malformed_terminal":
            for finish_reason in ("stop", "tool_calls"):
                yield {
                    "id": "chat-malformed-terminal",
                    "object": "chat.completion.chunk",
                    "model": "chat-conformance",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": finish_reason,
                        }
                    ],
                }
            return
        if self.scenario == "malformed_usage":
            yield {
                "id": "chat-malformed-usage",
                "object": "chat.completion.chunk",
                "model": "chat-conformance",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": [],
            }
            return
        if self.scenario == "unfinished":
            for event in _chat_unfinished_events():
                yield event
            return
        if self.scenario == "unfinished_reasoning":
            for event in _chat_unfinished_reasoning_events():
                yield event
            return
        if self.scenario == "attachments":
            _require_attachment_payload(payload, marker="data:image/png;base64,aGVsbG8=")
            for event in _chat_text_events():
                yield event
            return
        if self.scenario == "provider_cache_observation":
            for event in _chat_cache_events():
                yield event
            return
        if self.scenario == "reasoning":
            for event in _chat_reasoning_events():
                yield event
            return
        if self.scenario == "tool_round_trip":
            events = (
                _chat_tool_events()
                if len(self.calls) == 1
                else _chat_text_events(deltas=("tool-observed",))
            )
            if len(self.calls) == 2:
                _require_chat_completions_tool_result(payload, tool_call_id="call-conformance")
            for event in events:
                yield event
            return
        if self.scenario not in {"text", "close"}:
            raise AssertionError(f"Chat Completions scenario {self.scenario!r} is not implemented.")
        for event in _chat_text_events():
            yield event


class _VertexCredentials:
    token = "conformance-token"
    valid = True

    def refresh(self, request: Any) -> None:
        del request


class _VertexTransport(_AnthropicShapeTransport):
    def __init__(self, scenario: ProviderScenario) -> None:
        super().__init__(
            scenario,
            provider_label="Vertex",
            api_error=VertexAPIError,
            context_overflow_error=VertexContextOverflowError,
        )


class _BedrockClient:
    def __init__(self, scenario: ProviderScenario) -> None:
        self.scenario = scenario
        self.closed = False
        self.calls: list[dict[str, Any]] = []
        self.blocking_stream: _BlockingBedrockStream | None = None

    def converse_stream(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.scenario in {"cancellation", "idle_timeout"}:
            self.blocking_stream = _BlockingBedrockStream()
            return {"stream": self.blocking_stream}
        if self.scenario == "typed_error":
            raise BedrockAPIError("conformance throttle", **_ERROR_FIELDS)
        if self.scenario == "context_overflow":
            raise BedrockContextOverflowError("conformance context overflow", **_OVERFLOW_FIELDS)
        if self.scenario == "malformed":
            return {
                "stream": iter(
                    [
                        {
                            "contentBlockDelta": {
                                "contentBlockIndex": 0,
                                "delta": {"text": 42},
                            }
                        }
                    ]
                )
            }
        if self.scenario == "malformed_terminal":
            return {"stream": iter([{"messageStop": {"stopReason": 42}}])}
        if self.scenario == "malformed_usage":
            return {
                "stream": iter(
                    [
                        {"messageStop": {"stopReason": "end_turn"}},
                        {"metadata": {"usage": []}},
                    ]
                )
            }
        if self.scenario == "unfinished":
            return {"stream": iter(_bedrock_unfinished_events())}
        if self.scenario == "unfinished_reasoning":
            return {"stream": iter(_bedrock_unfinished_reasoning_events())}
        if self.scenario == "attachments":
            _require_attachment_payload(kwargs, marker="b'hello'")
            return {"stream": iter(_bedrock_text_events())}
        if self.scenario == "reasoning":
            return {"stream": iter(_bedrock_reasoning_events())}
        if self.scenario == "provider_cache_observation":
            return {"stream": iter(_bedrock_cache_events())}
        if self.scenario == "tool_round_trip":
            events = (
                _bedrock_tool_events()
                if len(self.calls) == 1
                else _bedrock_text_events(deltas=("tool-observed",))
            )
            if len(self.calls) == 2:
                _require_bedrock_tool_result(kwargs, tool_call_id="call-conformance")
            return {"stream": iter(events)}
        if self.scenario not in {"text", "close"}:
            raise AssertionError(f"Bedrock scenario {self.scenario!r} is not implemented.")
        return {"stream": iter(_bedrock_text_events())}

    def close(self) -> None:
        self.closed = True

    def count_tokens(self, **kwargs: Any) -> dict[str, int]:
        del kwargs
        if self.scenario != "token_counting":
            raise AssertionError(f"Bedrock scenario {self.scenario!r} cannot count tokens.")
        return {"inputTokens": 13}


class _BlockingBedrockStream:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.stopped = threading.Event()
        self._release = threading.Event()

    def __iter__(self) -> _BlockingBedrockStream:
        return self

    def __next__(self) -> dict[str, Any]:
        self.started.set()
        released = self._release.wait(timeout=1.0)
        self.stopped.set()
        if not released:
            raise RuntimeError("The Bedrock conformance stream was not released.")
        raise StopIteration

    def close(self) -> None:
        self._release.set()


async def _openai_factory(scenario: ProviderScenario) -> ProviderHarness:
    transport = _OpenAITransport(scenario)
    provider = OpenAIProvider(
        api_key="conformance-key",
        transport=transport,
        stream_idle_timeout_s=0.02,
    )
    return _async_transport_harness(provider, "gpt-conformance", transport)


async def _anthropic_factory(scenario: ProviderScenario) -> ProviderHarness:
    transport = _AnthropicTransport(scenario)
    provider = AnthropicProvider(
        api_key="conformance-key",
        transport=transport,
        stream_idle_timeout_s=0.02,
    )
    return _async_transport_harness(provider, "claude-conformance", transport)


async def _chat_completions_factory(scenario: ProviderScenario) -> ProviderHarness:
    transport = _ChatCompletionsTransport(scenario)
    provider = ChatCompletionsProvider(
        api_key="conformance-key",
        name="chat_conformance",
        transport=transport,
        stream_idle_timeout_s=0.02,
    )
    return _async_transport_harness(provider, "chat-conformance", transport)


async def _vertex_factory(scenario: ProviderScenario) -> ProviderHarness:
    transport = _VertexTransport(scenario)
    provider = VertexProvider(
        project_id="conformance-project",
        credentials=_VertexCredentials(),
        transport=transport,
        stream_idle_timeout_s=0.02,
    )
    return _async_transport_harness(provider, "claude-conformance", transport)


async def _bedrock_factory(scenario: ProviderScenario) -> ProviderHarness:
    client = _BedrockClient(scenario)
    patcher = None
    if scenario == "close":
        patcher = patch.object(BedrockProvider, "_create_client", return_value=client)
        patcher.start()
        provider = BedrockProvider(
            region_name="us-east-1",
            stream_idle_timeout_s=0.02,
            stream_close_timeout_s=0.2,
        )
    else:
        provider = BedrockProvider(
            client=client,
            region_name="us-east-1",
            stream_idle_timeout_s=0.02,
            stream_close_timeout_s=0.2,
        )

    async def close() -> None:
        try:
            await provider.aclose()
        finally:
            if patcher is not None:
                patcher.stop()

    async def wait_started() -> None:
        while client.blocking_stream is None:
            await asyncio.sleep(0)
        observed = await asyncio.to_thread(client.blocking_stream.started.wait, 0.5)
        if not observed:
            raise TimeoutError("The Bedrock conformance stream did not start.")

    async def wait_stopped() -> None:
        if client.blocking_stream is None:
            raise AssertionError("The Bedrock conformance stream did not start.")
        observed = await asyncio.to_thread(client.blocking_stream.stopped.wait, 0.5)
        if not observed:
            raise TimeoutError("The Bedrock conformance stream did not stop.")

    return ProviderHarness(
        provider=provider,
        model="us.anthropic.claude-conformance-v1",
        close=close,
        wait_started=wait_started,
        wait_stopped=wait_stopped,
        is_closed=lambda: client.closed,
    )


def _async_transport_harness(
    provider: OpenAIProvider | AnthropicProvider | ChatCompletionsProvider | VertexProvider,
    model: str,
    transport: _AsyncTransport,
) -> ProviderHarness:
    return ProviderHarness(
        provider=provider,
        model=model,
        wait_started=transport.started.wait,
        wait_stopped=transport.stopped.wait,
        is_closed=lambda: transport.closed,
    )


def _openai_text_events() -> Iterable[dict[str, Any]]:
    yield {"type": "response.created", "response": {"id": "resp-conformance"}}
    yield {"type": "response.output_text.delta", "delta": "hel"}
    yield {"type": "response.output_text.delta", "delta": "lo"}
    yield {
        "type": "response.completed",
        "response": {
            "id": "resp-conformance",
            "model": "gpt-conformance",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg-conformance",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hello", "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 9, "output_tokens": 2, "total_tokens": 11},
        },
    }


def _anthropic_text_events(
    *, model: str, deltas: tuple[str, ...] = ("hel", "lo")
) -> Iterable[dict[str, Any]]:
    yield {
        "type": "message_start",
        "message": {"id": "msg-conformance", "model": model, "usage": {"input_tokens": 9}},
    }
    yield {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    }
    for delta in deltas:
        yield {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": delta},
        }
    yield {"type": "content_block_stop", "index": 0}
    yield {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
        "usage": {"output_tokens": 2},
    }
    yield {"type": "message_stop"}


def _chat_text_events(*, deltas: tuple[str, ...] = ("hel", "lo")) -> Iterable[dict[str, Any]]:
    for delta in deltas:
        yield {
            "id": "chat-conformance",
            "object": "chat.completion.chunk",
            "model": "chat-conformance",
            "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
        }
    yield {
        "id": "chat-conformance",
        "object": "chat.completion.chunk",
        "model": "chat-conformance",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield {
        "id": "chat-conformance",
        "object": "chat.completion.chunk",
        "model": "chat-conformance",
        "choices": [],
        "usage": {"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11},
    }


def _bedrock_text_events(*, deltas: tuple[str, ...] = ("hel", "lo")) -> Iterable[dict[str, Any]]:
    yield {"messageStart": {"role": "assistant"}}
    for delta in deltas:
        yield {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": delta}}}
    yield {"contentBlockStop": {"contentBlockIndex": 0}}
    yield {"messageStop": {"stopReason": "end_turn"}}
    yield {
        "metadata": {
            "usage": {"inputTokens": 9, "outputTokens": 2, "totalTokens": 11},
            "metrics": {"latencyMs": 1},
        }
    }


def _openai_tool_events() -> Iterable[dict[str, Any]]:
    yield {
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {
            "type": "function_call",
            "id": "fc-conformance",
            "call_id": "call-conformance",
            "name": "echo",
            "arguments": "",
        },
    }
    yield {
        "type": "response.function_call_arguments.delta",
        "item_id": "fc-conformance",
        "output_index": 0,
        "delta": '{"text":"conformance-',
    }
    yield {
        "type": "response.function_call_arguments.delta",
        "item_id": "fc-conformance",
        "output_index": 0,
        "delta": 'tool"}',
    }
    yield {
        "type": "response.function_call_arguments.done",
        "item_id": "fc-conformance",
        "output_index": 0,
        "name": "echo",
        "arguments": '{"text":"conformance-tool"}',
    }
    yield {
        "type": "response.completed",
        "response": {
            "id": "resp-tool",
            "model": "gpt-conformance",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "id": "fc-conformance",
                    "call_id": "call-conformance",
                    "name": "echo",
                    "arguments": '{"text":"conformance-tool"}',
                    "status": "completed",
                }
            ],
            "usage": {"input_tokens": 9, "output_tokens": 2, "total_tokens": 11},
        },
    }


def _openai_final_events() -> Iterable[dict[str, Any]]:
    yield {"type": "response.output_text.delta", "delta": "tool-observed"}
    yield {
        "type": "response.completed",
        "response": {
            "id": "resp-final",
            "model": "gpt-conformance",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg-final",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "tool-observed",
                            "annotations": [],
                        }
                    ],
                }
            ],
            "usage": {"input_tokens": 12, "output_tokens": 2, "total_tokens": 14},
        },
    }


def _openai_unfinished_events() -> Iterable[dict[str, Any]]:
    yield {
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {
            "type": "function_call",
            "id": "fc-unfinished",
            "call_id": "call-unfinished",
            "name": "echo",
            "arguments": "",
        },
    }
    yield {
        "type": "response.function_call_arguments.delta",
        "item_id": "fc-unfinished",
        "output_index": 0,
        "delta": '{"text":',
    }
    yield {
        "type": "response.completed",
        "response": {
            "id": "resp-unfinished-tool",
            "model": "gpt-conformance",
            "status": "completed",
            "output": [],
        },
    }


def _openai_unfinished_reasoning_events() -> Iterable[dict[str, Any]]:
    yield {
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {"type": "reasoning", "id": "rs-unfinished", "summary": []},
    }
    yield {"type": "response.reasoning_summary_text.delta", "delta": "partial"}
    yield {
        "type": "response.completed",
        "response": {
            "id": "resp-unfinished-reasoning",
            "model": "gpt-conformance",
            "status": "completed",
            "output": [],
        },
    }


def _openai_reasoning_events() -> Iterable[dict[str, Any]]:
    yield {"type": "response.reasoning_summary_text.delta", "delta": "think"}
    yield {
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {
            "type": "reasoning",
            "id": "rs-conformance",
            "summary": [{"type": "summary_text", "text": "think"}],
            "encrypted_content": "opaque-conformance",
        },
    }
    yield {
        "type": "response.completed",
        "response": {
            "id": "resp-reasoning",
            "model": "gpt-conformance",
            "status": "completed",
            "output": [],
            "usage": {"input_tokens": 9, "output_tokens": 2, "total_tokens": 11},
        },
    }


def _anthropic_tool_events(*, model: str) -> Iterable[dict[str, Any]]:
    yield {
        "type": "message_start",
        "message": {"id": "msg-tool", "model": model, "usage": {"input_tokens": 9}},
    }
    yield {
        "type": "content_block_start",
        "index": 0,
        "content_block": {
            "type": "tool_use",
            "id": "call-conformance",
            "name": "echo",
            "input": {},
        },
    }
    for partial in ('{"text":"conformance-', 'tool"}'):
        yield {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": partial},
        }
    yield {"type": "content_block_stop", "index": 0}
    yield {
        "type": "message_delta",
        "delta": {"stop_reason": "tool_use"},
        "usage": {"output_tokens": 2},
    }
    yield {"type": "message_stop"}


def _anthropic_unfinished_events(*, model: str) -> Iterable[dict[str, Any]]:
    yield {
        "type": "message_start",
        "message": {"id": "msg-unfinished", "model": model, "usage": {"input_tokens": 9}},
    }
    yield {
        "type": "content_block_start",
        "index": 0,
        "content_block": {
            "type": "tool_use",
            "id": "call-unfinished",
            "name": "echo",
            "input": {},
        },
    }
    yield {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "input_json_delta", "partial_json": '{"text":'},
    }
    yield {
        "type": "message_delta",
        "delta": {"stop_reason": "tool_use"},
        "usage": {"output_tokens": 2},
    }
    yield {"type": "message_stop"}


def _anthropic_unfinished_reasoning_events(*, model: str) -> Iterable[dict[str, Any]]:
    yield {
        "type": "message_start",
        "message": {"id": "msg-unfinished-reasoning", "model": model},
    }
    yield {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "thinking", "thinking": "", "signature": ""},
    }
    yield {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "thinking_delta", "thinking": "partial"},
    }
    yield {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
        "usage": {"output_tokens": 2},
    }
    yield {"type": "message_stop"}


def _anthropic_reasoning_events(*, model: str) -> Iterable[dict[str, Any]]:
    yield {
        "type": "message_start",
        "message": {"id": "msg-reasoning", "model": model, "usage": {"input_tokens": 9}},
    }
    yield {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "thinking", "thinking": "", "signature": ""},
    }
    yield {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "thinking_delta", "thinking": "think"},
    }
    yield {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "signature_delta", "signature": "sig-conformance"},
    }
    yield {"type": "content_block_stop", "index": 0}
    yield {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
        "usage": {"output_tokens": 2},
    }
    yield {"type": "message_stop"}


def _chat_tool_events() -> Iterable[dict[str, Any]]:
    yield {
        "id": "chat-tool",
        "object": "chat.completion.chunk",
        "model": "chat-conformance",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call-conformance",
                            "type": "function",
                            "function": {"name": "echo", "arguments": ""},
                        }
                    ]
                },
                "finish_reason": None,
            }
        ],
    }
    for partial in ('{"text":"conformance-', 'tool"}'):
        yield {
            "id": "chat-tool",
            "object": "chat.completion.chunk",
            "model": "chat-conformance",
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": partial}}]},
                    "finish_reason": None,
                }
            ],
        }
    yield {
        "id": "chat-tool",
        "object": "chat.completion.chunk",
        "model": "chat-conformance",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
    }
    yield {
        "id": "chat-tool",
        "object": "chat.completion.chunk",
        "model": "chat-conformance",
        "choices": [],
        "usage": {"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11},
    }


def _chat_unfinished_events() -> Iterable[dict[str, Any]]:
    yield {
        "id": "chat-unfinished",
        "object": "chat.completion.chunk",
        "model": "chat-conformance",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call-unfinished",
                            "type": "function",
                            "function": {"name": "echo", "arguments": '{"text":'},
                        }
                    ]
                },
                "finish_reason": None,
            }
        ],
    }
    yield {
        "id": "chat-unfinished",
        "object": "chat.completion.chunk",
        "model": "chat-conformance",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
    }


def _chat_unfinished_reasoning_events() -> Iterable[dict[str, Any]]:
    yield {
        "id": "chat-unfinished-reasoning",
        "object": "chat.completion.chunk",
        "model": "chat-conformance",
        "choices": [
            {
                "index": 0,
                "delta": {"reasoning_content": "partial"},
                "finish_reason": None,
            }
        ],
    }


def _chat_reasoning_events() -> Iterable[dict[str, Any]]:
    yield {
        "id": "chat-reasoning",
        "object": "chat.completion.chunk",
        "model": "chat-conformance",
        "choices": [
            {
                "index": 0,
                "delta": {"reasoning_content": "think"},
                "finish_reason": None,
            }
        ],
    }
    yield {
        "id": "chat-reasoning",
        "object": "chat.completion.chunk",
        "model": "chat-conformance",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }


def _bedrock_tool_events() -> Iterable[dict[str, Any]]:
    yield {
        "contentBlockStart": {
            "contentBlockIndex": 0,
            "start": {
                "toolUse": {
                    "toolUseId": "call-conformance",
                    "name": "echo",
                }
            },
        }
    }
    for partial in ('{"text":"conformance-', 'tool"}'):
        yield {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": partial}},
            }
        }
    yield {"contentBlockStop": {"contentBlockIndex": 0}}
    yield {"messageStop": {"stopReason": "tool_use"}}
    yield {"metadata": {"usage": {"inputTokens": 9, "outputTokens": 2, "totalTokens": 11}}}


def _bedrock_unfinished_events() -> Iterable[dict[str, Any]]:
    yield {
        "contentBlockStart": {
            "contentBlockIndex": 0,
            "start": {"toolUse": {"toolUseId": "call-unfinished", "name": "echo"}},
        }
    }
    yield {
        "contentBlockDelta": {
            "contentBlockIndex": 0,
            "delta": {"toolUse": {"input": '{"text":'}},
        }
    }
    yield {"messageStop": {"stopReason": "tool_use"}}


def _bedrock_unfinished_reasoning_events() -> Iterable[dict[str, Any]]:
    yield {
        "contentBlockDelta": {
            "contentBlockIndex": 0,
            "delta": {"reasoningContent": {"text": "partial"}},
        }
    }
    yield {"messageStop": {"stopReason": "end_turn"}}


def _bedrock_reasoning_events() -> Iterable[dict[str, Any]]:
    yield {
        "contentBlockDelta": {
            "contentBlockIndex": 0,
            "delta": {"reasoningContent": {"text": "think"}},
        }
    }
    yield {
        "contentBlockDelta": {
            "contentBlockIndex": 0,
            "delta": {"reasoningContent": {"signature": "sig-conformance"}},
        }
    }
    yield {"contentBlockStop": {"contentBlockIndex": 0}}
    yield {"messageStop": {"stopReason": "end_turn"}}


def _openai_cache_events() -> Iterable[dict[str, Any]]:
    yield {"type": "response.output_text.delta", "delta": "cached"}
    yield {
        "type": "response.completed",
        "response": {
            "id": "resp-cache",
            "model": "gpt-conformance",
            "status": "completed",
            "output": [],
            "usage": {
                "input_tokens": 9,
                "output_tokens": 2,
                "total_tokens": 11,
                "input_tokens_details": {"cached_tokens": 3},
            },
        },
    }


def _anthropic_cache_events(*, model: str) -> Iterable[dict[str, Any]]:
    yield {
        "type": "message_start",
        "message": {
            "id": "msg-cache",
            "model": model,
            "usage": {
                "input_tokens": 5,
                "cache_read_input_tokens": 3,
                "cache_creation_input_tokens": 1,
            },
        },
    }
    yield {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    }
    yield {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "cached"},
    }
    yield {"type": "content_block_stop", "index": 0}
    yield {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
        "usage": {"output_tokens": 2},
    }
    yield {"type": "message_stop"}


def _chat_cache_events() -> Iterable[dict[str, Any]]:
    yield {
        "id": "chat-cache",
        "object": "chat.completion.chunk",
        "model": "chat-conformance",
        "choices": [{"index": 0, "delta": {"content": "cached"}, "finish_reason": None}],
    }
    yield {
        "id": "chat-cache",
        "object": "chat.completion.chunk",
        "model": "chat-conformance",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield {
        "id": "chat-cache",
        "object": "chat.completion.chunk",
        "model": "chat-conformance",
        "choices": [],
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 2,
            "total_tokens": 11,
            "prompt_tokens_details": {"cached_tokens": 3},
        },
    }


def _bedrock_cache_events() -> Iterable[dict[str, Any]]:
    yield {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "cached"}}}
    yield {"contentBlockStop": {"contentBlockIndex": 0}}
    yield {"messageStop": {"stopReason": "end_turn"}}
    yield {
        "metadata": {
            "usage": {
                "inputTokens": 5,
                "outputTokens": 2,
                "totalTokens": 11,
                "cacheReadInputTokens": 3,
                "cacheWriteInputTokens": 1,
            }
        }
    }


def _require_openai_tool_result(payload: Mapping[str, Any], *, tool_call_id: str) -> None:
    input_items = payload.get("input")
    valid = isinstance(input_items, list) and any(
        isinstance(item, Mapping)
        and item.get("type") == "function_call_output"
        and item.get("call_id") == tool_call_id
        and item.get("output") == "conformance-tool"
        for item in input_items
    )
    _require_completed_tool_result(valid)


def _require_anthropic_tool_result(payload: Mapping[str, Any], *, tool_call_id: str) -> None:
    messages = payload.get("messages")
    valid = isinstance(messages, list) and any(
        isinstance(message, Mapping)
        and message.get("role") == "user"
        and isinstance((content := message.get("content")), list)
        and any(
            isinstance(block, Mapping)
            and block.get("type") == "tool_result"
            and block.get("tool_use_id") == tool_call_id
            and block.get("content") == "conformance-tool"
            for block in content
        )
        for message in messages
    )
    _require_completed_tool_result(valid)


def _require_chat_completions_tool_result(payload: Mapping[str, Any], *, tool_call_id: str) -> None:
    messages = payload.get("messages")
    valid = isinstance(messages, list) and any(
        isinstance(message, Mapping)
        and message.get("role") == "tool"
        and message.get("tool_call_id") == tool_call_id
        and message.get("content") == "conformance-tool"
        for message in messages
    )
    _require_completed_tool_result(valid)


def _require_bedrock_tool_result(payload: Mapping[str, Any], *, tool_call_id: str) -> None:
    messages = payload.get("messages")
    valid = isinstance(messages, list) and any(
        isinstance(message, Mapping)
        and message.get("role") == "user"
        and isinstance((content := message.get("content")), list)
        and any(
            isinstance(block, Mapping)
            and isinstance((result := block.get("toolResult")), Mapping)
            and result.get("toolUseId") == tool_call_id
            and result.get("status") == "success"
            and isinstance((result_content := result.get("content")), list)
            and any(
                isinstance(part, Mapping) and part.get("text") == "conformance-tool"
                for part in result_content
            )
            for block in content
        )
        for message in messages
    )
    _require_completed_tool_result(valid)


def _require_completed_tool_result(valid: bool) -> None:
    if not valid:
        raise AssertionError("The second provider request omitted the completed tool result.")


def _require_attachment_payload(payload: Mapping[str, Any], *, marker: str) -> None:
    rendered = json.dumps(payload, sort_keys=True, default=str)
    if marker not in rendered:
        raise AssertionError("The provider request omitted the resolved image attachment.")


_ERROR_FIELDS = {
    "status_code": 429,
    "error_type": "rate_limit_error",
    "error_code": "rate_limit_exceeded",
    "request_id": "req-conformance",
    "retryable": True,
    "retry_after_s": 0.25,
}
_OVERFLOW_FIELDS = {
    "status_code": 400,
    "error_type": "invalid_request_error",
    "error_code": "context_length_exceeded",
    "request_id": "req-overflow",
}


_NO_NATIVE = CapabilityClaim.not_applicable(
    "The adapter uses Cayu tool-based structured output rather than a native schema mode."
)

OPENAI = ProviderConformanceRegistration(
    name="openai",
    provider_type=OpenAIProvider,
    factory=_openai_factory,
    capabilities=ProviderCapabilities(
        token_counting=CapabilityClaim.supported(),
        native_structured_output=CapabilityClaim.supported(),
        attachments=CapabilityClaim.supported(),
        reasoning=CapabilityClaim.supported(),
        provider_cache_observation=CapabilityClaim.supported(),
    ),
)

ANTHROPIC = ProviderConformanceRegistration(
    name="anthropic",
    provider_type=AnthropicProvider,
    factory=_anthropic_factory,
    capabilities=ProviderCapabilities(
        token_counting=CapabilityClaim.supported(),
        native_structured_output=_NO_NATIVE,
        attachments=CapabilityClaim.supported(),
        reasoning=CapabilityClaim.supported(),
        provider_cache_observation=CapabilityClaim.supported(),
    ),
)

CHAT_COMPLETIONS = ProviderConformanceRegistration(
    name="chat-completions",
    provider_type=ChatCompletionsProvider,
    factory=_chat_completions_factory,
    capabilities=ProviderCapabilities(
        token_counting=CapabilityClaim.unsupported(
            "The OpenAI-compatible chat-completions protocol has no portable counting endpoint."
        ),
        native_structured_output=_NO_NATIVE,
        attachments=CapabilityClaim.supported(),
        reasoning=CapabilityClaim.supported(),
        provider_cache_observation=CapabilityClaim.supported(),
    ),
    error_provider="chat_completions",
)

VERTEX = ProviderConformanceRegistration(
    name="vertex",
    provider_type=VertexProvider,
    factory=_vertex_factory,
    capabilities=ProviderCapabilities(
        token_counting=CapabilityClaim.supported(),
        native_structured_output=_NO_NATIVE,
        attachments=CapabilityClaim.supported(),
        reasoning=CapabilityClaim.supported(),
        provider_cache_observation=CapabilityClaim.supported(),
    ),
)

BEDROCK = ProviderConformanceRegistration(
    name="bedrock",
    provider_type=BedrockProvider,
    factory=_bedrock_factory,
    capabilities=ProviderCapabilities(
        token_counting=CapabilityClaim.supported(),
        native_structured_output=_NO_NATIVE,
        attachments=CapabilityClaim.supported(),
        reasoning=CapabilityClaim.supported(),
        provider_cache_observation=CapabilityClaim.supported(),
    ),
    reports_model_identity=False,
)

REGISTRATIONS = (OPENAI, ANTHROPIC, CHAT_COMPLETIONS, VERTEX, BEDROCK)
