from __future__ import annotations

import json
import os
import re
from collections.abc import AsyncIterator, Mapping
from typing import TYPE_CHECKING, Any, Protocol, cast

from cayu._validation import copy_json_value, require_clean_nonblank, require_nonblank
from cayu.artifacts import (
    FileAttachmentKind,
    file_attachment_from_payload,
    resolved_file_attachments_from_options,
)
from cayu.core.messages import (
    FilePart,
    Message,
    MessageRole,
    ProviderStatePart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolResultPart,
)
from cayu.embeddings import (
    TextEmbedding,
    TextEmbeddingProvider,
    TextEmbeddingRequest,
    TextEmbeddingResult,
    TextEmbeddingUsage,
    copy_text_embedding_request,
)
from cayu.providers._http import (
    SharedAsyncClient,
    aclose_transport,
    copy_headers,
    exception_message,
    optional_error_string,
    post_json,
    response_json_object,
    safe_error_json,
    safe_error_response_text,
    stream_sse_json_events,
    truncate_error_text,
    validate_base_url,
    validate_url,
)
from cayu.providers.base import (
    InputTokenCountConfidence,
    InputTokenCountMethod,
    InputTokenCountResult,
    ModelCompletion,
    ModelContextOverflowError,
    ModelContextPressureProfile,
    ModelFinishReason,
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelStreamEvent,
    ModelStreamEventType,
)

if TYPE_CHECKING:
    import httpx

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com"
DEFAULT_OPENAI_TIMEOUT_SECONDS = 60.0
DEFAULT_OPENAI_STREAM_IDLE_TIMEOUT_SECONDS = 120.0
OPENAI_CONTEXT_PRESSURE_TOOL_SCHEMA_CHARS_PER_TOKEN = 6

_RESERVED_OPENAI_OPTIONS = {
    "model",
    "input",
    "instructions",
    "previous_response_id",
    "store",
    "tools",
    "stream",
}
_OPENAI_TOKEN_COUNT_FIELDS = frozenset(
    {
        "model",
        "input",
        "previous_response_id",
        "tools",
        "text",
        "reasoning",
        "truncation",
        "instructions",
        "conversation",
        "tool_choice",
        "parallel_tool_calls",
    }
)
_PROTECTED_HEADER_NAMES = {
    "authorization",
    "content-type",
}
_OPENAI_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_VALID_REASONING_STATES = {"inline", "server"}


class OpenAIError(RuntimeError):
    """Base error for OpenAI provider failures."""


class OpenAIAPIError(OpenAIError, ModelProviderError):
    """Raised when the OpenAI HTTP API returns an error response.

    ``param`` carries the OpenAI error body's ``param`` field (the request
    field the error refers to, e.g. ``"previous_response_id"``); it is
    OpenAI-specific, so it lives here rather than on `ModelProviderError`.
    """

    param: str | None = None

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_type: str | None = None,
        error_code: str | None = None,
        param: str | None = None,
        request_id: str | None = None,
        retryable: bool | None = None,
        retry_after_s: float | None = None,
        response_body: str | None = None,
    ) -> None:
        ModelProviderError.__init__(
            self,
            message,
            provider="openai",
            status_code=status_code,
            error_type=error_type,
            error_code=error_code,
            request_id=request_id,
            retryable=retryable,
            retry_after_s=retry_after_s,
            response_body=response_body,
        )
        if param is not None:
            param = require_clean_nonblank(param, "param")
        self.param = param


class OpenAIContextOverflowError(OpenAIAPIError, ModelContextOverflowError):
    """Raised when OpenAI reports that the request exceeds context limits."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_type: str | None = None,
        error_code: str | None = None,
        request_id: str | None = None,
        response_body: str | None = None,
    ) -> None:
        ModelContextOverflowError.__init__(
            self,
            message,
            provider="openai",
            status_code=status_code,
            error_type=error_type,
            error_code=error_code,
            request_id=request_id,
            response_body=response_body,
        )


class OpenAIProtocolError(OpenAIError):
    """Raised when OpenAI data does not match the expected Responses shape."""


class OpenAITransport(Protocol):
    async def create_response(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        """POST a non-streaming Responses API payload and return decoded JSON."""

    def stream_response_events(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
        stream_idle_timeout_s: float,
    ) -> AsyncIterator[Mapping[str, Any]]:
        """POST a streaming Responses API payload and yield decoded SSE data objects."""


class HttpxOpenAITransport:
    """HTTP transport with explicit certifi-backed TLS verification.

    Owns one shared httpx.AsyncClient (created lazily) that is reused across
    requests so each model call does not pay for a fresh TLS handshake. Close it
    with :meth:`aclose` when the transport is no longer needed.
    """

    def __init__(self) -> None:
        self._client = SharedAsyncClient()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def create_response(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        url = _validate_url(url, "url")
        return await post_json(
            client=self._client.get(),
            url=url,
            headers=headers,
            payload=payload,
            timeout_s=timeout_s,
            request_label="OpenAI API",
            response_label="OpenAI",
            api_error=OpenAIAPIError,
            protocol_error=OpenAIProtocolError,
            error_response_text=_safe_error_response_text,
            raise_context_overflow=_raise_openai_context_overflow_if_applicable,
            api_error_from_response=_openai_api_error_from_response,
        )

    async def stream_response_events(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
        stream_idle_timeout_s: float,
    ) -> AsyncIterator[Mapping[str, Any]]:
        url = _validate_url(url, "url")
        events = stream_sse_json_events(
            client=self._client.get(),
            url=url,
            headers=headers,
            payload=payload,
            timeout_s=timeout_s,
            stream_idle_timeout_s=stream_idle_timeout_s,
            request_label="OpenAI API",
            response_label="OpenAI",
            api_error=OpenAIAPIError,
            protocol_error=OpenAIProtocolError,
            error_response_text=_safe_error_response_text,
            raise_context_overflow=_raise_openai_context_overflow_if_applicable,
            api_error_from_response=_openai_api_error_from_response,
        )
        async for event in events:
            yield event


class OpenAIProvider(ModelProvider, TextEmbeddingProvider):
    """OpenAI Responses API adapter for Cayu's provider-neutral runtime."""

    name = "openai"
    supports_native_structured_output = True

    @property
    def context_pressure_profile(self) -> ModelContextPressureProfile:
        return ModelContextPressureProfile(
            tool_schema_chars_per_token=OPENAI_CONTEXT_PRESSURE_TOOL_SCHEMA_CHARS_PER_TOKEN,
        )

    def __init__(
        self,
        *,
        api_key: str | None = None,
        name: str = "openai",
        base_url: str = DEFAULT_OPENAI_BASE_URL,
        timeout_s: float = DEFAULT_OPENAI_TIMEOUT_SECONDS,
        stream_idle_timeout_s: float = DEFAULT_OPENAI_STREAM_IDLE_TIMEOUT_SECONDS,
        transport: OpenAITransport | None = None,
        extra_headers: Mapping[str, str] | None = None,
        reasoning_state: str = "inline",
    ) -> None:
        self.name = require_clean_nonblank(name, "name")
        self.api_key = require_nonblank(
            api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", ""),
            "api_key",
        )
        self.base_url = _validate_base_url(base_url)
        if type(timeout_s) not in {int, float}:
            raise TypeError("timeout_s must be a number.")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be greater than zero.")
        self.timeout_s = float(timeout_s)
        if type(stream_idle_timeout_s) not in {int, float}:
            raise TypeError("stream_idle_timeout_s must be a number.")
        if stream_idle_timeout_s <= 0:
            raise ValueError("stream_idle_timeout_s must be greater than zero.")
        self.stream_idle_timeout_s = float(stream_idle_timeout_s)
        self.transport = transport if transport is not None else HttpxOpenAITransport()
        self.extra_headers = _copy_headers(extra_headers)
        self.reasoning_state = _validate_reasoning_state(reasoning_state)

    async def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        try:
            payload = build_openai_payload(
                request, stream=True, reasoning_state=self.reasoning_state
            )
            yielded_any = False
            try:
                async for event in self._consume(payload):
                    yielded_any = True
                    yield event
                return
            except OpenAIAPIError as exc:
                recoverable = (
                    self.reasoning_state == "server"
                    and not yielded_any
                    and _is_stale_chain_error(exc)
                )
                if not recoverable:
                    raise
            # Recovery: one clean full resend rebuilt from neutral parts.
            recovery_payload = build_openai_payload(
                request, stream=True, reasoning_state=self.reasoning_state, chain=False
            )
            async for event in self._consume(recovery_payload):
                yield event
        except ModelContextOverflowError:
            # Overflow must reach runtime recovery as a typed exception; an
            # error event would flatten it into unrecoverable message text.
            raise
        except Exception as exc:
            yield ModelStreamEvent.error(
                exception_message(exc, provider_label="OpenAI"),
                cause=exc,
            )

    async def count_input_tokens(
        self,
        request: ModelRequest,
    ) -> InputTokenCountResult | None:
        payload = build_openai_token_count_payload(
            request,
            reasoning_state=self.reasoning_state,
        )
        response = await self.transport.create_response(
            url=f"{self.base_url}/v1/responses/input_tokens",
            headers=self._headers(),
            payload=payload,
            timeout_s=self.timeout_s,
        )
        return InputTokenCountResult(
            input_tokens=_openai_input_tokens_from_count_response(response),
            method=InputTokenCountMethod.OFFICIAL,
            confidence=InputTokenCountConfidence.HIGH,
            metadata={
                "endpoint": "responses/input_tokens",
                "provider_billing_status": "not_documented",
            },
        )

    async def embed_texts(self, request: TextEmbeddingRequest) -> TextEmbeddingResult:
        embedding_request = copy_text_embedding_request(request)
        payload = build_openai_embedding_payload(embedding_request)
        response = await self.transport.create_response(
            url=f"{self.base_url}/v1/embeddings",
            headers=self._headers(),
            payload=payload,
            timeout_s=self.timeout_s,
        )
        return openai_embedding_result(response, requested_count=len(embedding_request.texts))

    async def _consume(self, payload: dict[str, Any]) -> AsyncIterator[ModelStreamEvent]:
        raw_events = self.transport.stream_response_events(
            url=f"{self.base_url}/v1/responses",
            headers=self._headers(),
            payload=payload,
            timeout_s=self.timeout_s,
            stream_idle_timeout_s=self.stream_idle_timeout_s,
        )
        async for event in openai_stream_events(raw_events, reasoning_state=self.reasoning_state):
            yield event

    async def aclose(self) -> None:
        """Close the transport's shared HTTP client, if it owns one."""
        await aclose_transport(self.transport)

    def _headers(self) -> dict[str, str]:
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {self.api_key}",
        }
        headers.update(self.extra_headers)
        return headers


def build_openai_payload(
    request: ModelRequest,
    *,
    stream: bool = False,
    reasoning_state: str = "inline",
    chain: bool = True,
) -> dict[str, Any]:
    if type(request) is not ModelRequest:
        raise TypeError("request must be a ModelRequest.")
    reasoning_state = _validate_reasoning_state(reasoning_state)
    if type(chain) is not bool:
        raise TypeError("OpenAI payload chain must be a bool.")

    options = _openai_options(request.options)
    structured_output_format = _openai_structured_output_format(request.options)
    if structured_output_format is not None and "text" in options:
        raise ValueError("OpenAI option text cannot be combined with native structured output.")
    payload: dict[str, Any] = {
        "model": request.model,
        "input": [],
        "store": reasoning_state == "server",
    }
    instructions = _system_text(request.messages)
    if instructions:
        payload["instructions"] = instructions

    resolved_attachments = resolved_file_attachments_from_options(request.options)

    previous_response_id: str | None = None
    messages_to_send = request.messages
    use_provider_state = True
    if reasoning_state == "server" and chain:
        previous_response_id, messages_to_send = _server_chain(request.messages)
    elif reasoning_state == "server" and not chain:
        use_provider_state = False  # recovery: rebuild from neutral parts

    input_items: list[dict[str, Any]] = []
    for message in messages_to_send:
        input_items.extend(
            _openai_input_items(
                message,
                resolved_attachments=resolved_attachments,
                reasoning_state=reasoning_state,
                use_provider_state=use_provider_state,
            )
        )
    if not input_items:
        raise ValueError("OpenAI requests require at least one non-system input item.")
    payload["input"] = input_items
    if previous_response_id is not None:
        payload["previous_response_id"] = previous_response_id

    tools = [_openai_tool(tool) for tool in request.tools]
    if tools:
        payload["tools"] = tools
    if structured_output_format is not None:
        payload["text"] = {"format": structured_output_format}
    # Ask for encrypted reasoning content. Under store=false, reasoning output
    # items carry only an rs_ id that the server cannot resolve on the next call
    # (HTTP 404). Requesting reasoning.encrypted_content attaches an opaque blob
    # that round-trips reasoning across stateless calls. Harmless for non-reasoning
    # models. Apps can still override via options.openai.
    if reasoning_state == "inline":
        payload["include"] = ["reasoning.encrypted_content"]
    if stream:
        payload["stream"] = True
    payload.update(options)
    _apply_thinking_options(payload, request.options.get("thinking"))
    return copy_json_value(payload, "openai_payload")


def build_openai_token_count_payload(
    request: ModelRequest,
    *,
    reasoning_state: str = "inline",
    chain: bool = True,
) -> dict[str, Any]:
    payload = build_openai_payload(
        request,
        stream=False,
        reasoning_state=reasoning_state,
        chain=chain,
    )
    count_payload = {
        key: value for key, value in payload.items() if key in _OPENAI_TOKEN_COUNT_FIELDS
    }
    return copy_json_value(count_payload, "openai_token_count_payload")


def build_openai_embedding_payload(request: TextEmbeddingRequest) -> dict[str, Any]:
    if type(request) is not TextEmbeddingRequest:
        raise TypeError("request must be a TextEmbeddingRequest.")
    options = _openai_embedding_options(request.options)
    payload: dict[str, Any] = {
        "model": request.model,
        "input": list(request.texts),
        "encoding_format": "float",
    }
    if request.dimensions is not None:
        payload["dimensions"] = request.dimensions
    payload.update(options)
    return copy_json_value(payload, "openai_embedding_payload")


def openai_embedding_result(
    response: Mapping[str, Any],
    *,
    requested_count: int,
) -> TextEmbeddingResult:
    if not isinstance(response, Mapping):
        raise OpenAIProtocolError("OpenAI embedding response must be a JSON object.")
    object_type = response.get("object")
    if object_type != "list":
        raise OpenAIProtocolError("OpenAI embedding response has unexpected object.")
    model = response.get("model")
    if type(model) is not str:
        raise OpenAIProtocolError("OpenAI embedding response requires model.")
    data = response.get("data")
    if not isinstance(data, list):
        raise OpenAIProtocolError("OpenAI embedding response data must be a list.")
    if len(data) != requested_count:
        raise OpenAIProtocolError("OpenAI embedding response count did not match request.")
    embeddings: list[TextEmbedding] = []
    for position, item in enumerate(data):
        if not isinstance(item, Mapping):
            raise OpenAIProtocolError(f"OpenAI embedding item {position} must be an object.")
        item_data = cast("Mapping[str, Any]", item)
        index = item_data.get("index")
        vector = item_data.get("embedding")
        if type(index) is not int:
            raise OpenAIProtocolError(f"OpenAI embedding item {position} requires index.")
        if not isinstance(vector, list):
            raise OpenAIProtocolError(f"OpenAI embedding item {position} requires vector.")
        vector_numbers: list[float] = []
        for vector_index, vector_item in enumerate(vector):
            if isinstance(vector_item, bool) or not isinstance(vector_item, int | float):
                raise OpenAIProtocolError(
                    f"OpenAI embedding item {position} vector[{vector_index}] must be a number."
                )
            vector_numbers.append(float(vector_item))
        embeddings.append(TextEmbedding(index=index, vector=vector_numbers))
    embeddings.sort(key=lambda embedding: embedding.index)
    if [embedding.index for embedding in embeddings] != list(range(requested_count)):
        raise OpenAIProtocolError("OpenAI embedding response indexes did not match request.")
    usage = _openai_embedding_usage(response.get("usage"))
    return TextEmbeddingResult(
        model=model,
        embeddings=embeddings,
        usage=usage,
        metadata={"provider": "openai", "endpoint": "embeddings"},
    )


def _openai_embedding_usage(value: object) -> TextEmbeddingUsage | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise OpenAIProtocolError("OpenAI embedding usage must be an object.")
    usage_data = cast("Mapping[str, Any]", value)
    prompt_tokens = _optional_openai_embedding_token_count(usage_data, "prompt_tokens")
    total_tokens = _optional_openai_embedding_token_count(usage_data, "total_tokens")
    return TextEmbeddingUsage(
        input_tokens=prompt_tokens,
        total_tokens=total_tokens,
        metadata={"provider_billing_status": "usage_reported"},
    )


def _optional_openai_embedding_token_count(
    value: Mapping[str, Any],
    key: str,
) -> int | None:
    raw = value.get(key)
    if raw is None:
        return None
    if type(raw) is not int or raw < 0:
        raise OpenAIProtocolError(f"OpenAI embedding usage requires nonnegative {key}.")
    return raw


def _openai_embedding_options(options: Mapping[str, Any]) -> dict[str, Any]:
    copied = copy_json_value(dict(options), "openai_embedding_options")
    if type(copied) is not dict:
        raise ValueError("OpenAI embedding options must be a dictionary.")
    reserved = {"dimensions", "encoding_format", "input", "model"}
    conflict = reserved.intersection(copied)
    if conflict:
        names = ", ".join(sorted(conflict))
        raise ValueError(f"OpenAI embedding options cannot override reserved keys: {names}.")
    return copied


def _apply_thinking_options(payload: dict[str, Any], neutral: Any) -> None:
    """Map the neutral ``options["thinking"]`` payload onto OpenAI ``reasoning`` keys.

    OpenAI reasoning models cannot disable reasoning and expose no token budget, so only
    ``effort`` maps (authoritative — overwrites a raw value). ``summary="auto"`` is added
    as a default to surface readable reasoning, so a caller's raw ``reasoning.summary``
    (and any other raw ``reasoning`` sibling) is preserved. ``enabled=False`` is a no-op
    (the model reasons at its default).
    """
    if not isinstance(neutral, Mapping) or not neutral.get("enabled", True):
        return
    existing = payload.get("reasoning")
    reasoning = dict(existing) if isinstance(existing, dict) else {}
    reasoning.setdefault("summary", "auto")
    effort = neutral.get("effort")
    if effort is not None:
        reasoning["effort"] = effort
    payload["reasoning"] = reasoning


def openai_response_events(response: Mapping[str, Any]) -> list[ModelStreamEvent]:
    if not isinstance(response, Mapping):
        raise OpenAIProtocolError("OpenAI response must be a JSON object.")

    error = response.get("error")
    if error is not None:
        raise OpenAIProtocolError(f"OpenAI response error: {_safe_error_value(error)}")

    output = response.get("output")
    if not isinstance(output, list):
        raise OpenAIProtocolError("OpenAI response output must be a list.")

    events: list[ModelStreamEvent] = []
    provider_state_items: list[dict[str, Any]] = []
    for index, item in enumerate(output):
        if not isinstance(item, Mapping):
            raise OpenAIProtocolError(f"OpenAI output item {index} must be an object.")
        item = cast("Mapping[str, Any]", item)
        item_type = item.get("type")
        if item_type == "message":
            events.extend(_message_output_events(item, index))
            provider_state_items.append(
                {"provider": "openai", "state": copy_json_value(item, "output_item")}
            )
        elif item_type == "function_call":
            events.append(_function_call_event(item, index))
            provider_state_items.append(
                {"provider": "openai", "state": copy_json_value(item, "output_item")}
            )
        elif item_type == "reasoning":
            # Surface the readable reasoning summary as display-only thinking, but keep
            # capturing the full reasoning item (incl. encrypted_content) as provider
            # state so the multi-turn round-trip is unaffected.
            events.extend(_reasoning_thinking_events(item))
            provider_state_items.append(
                {"provider": "openai", "state": copy_json_value(item, "output_item")}
            )
            continue
        else:
            raise OpenAIProtocolError(f"Unsupported OpenAI output item type: {item_type!r}.")

    events.append(_completed_event_from_response(response, provider_state_items))
    return events


def _openai_input_tokens_from_count_response(response: Mapping[str, Any]) -> int:
    if not isinstance(response, Mapping):
        raise OpenAIProtocolError("OpenAI input token count response must be a JSON object.")
    object_type = response.get("object")
    if object_type != "response.input_tokens":
        raise OpenAIProtocolError("OpenAI input token count response has unexpected object.")
    input_tokens = response.get("input_tokens")
    if type(input_tokens) is not int or input_tokens < 0:
        raise OpenAIProtocolError("OpenAI input token count response requires input_tokens.")
    return input_tokens


def _reasoning_thinking_events(item: Mapping[str, Any]) -> list[ModelStreamEvent]:
    """Extract readable reasoning summary text from a non-stream reasoning item.

    Emits one display-only thinking event per ``summary_text`` part; the opaque
    encrypted reasoning is preserved separately as provider state for round-tripping.
    """
    summary = item.get("summary")
    if not isinstance(summary, list):
        return []
    texts: list[str] = []
    for part in summary:
        if not isinstance(part, Mapping):
            continue
        if part.get("type") != "summary_text":
            continue
        text = part.get("text")
        if isinstance(text, str) and text:
            texts.append(text)
    if not texts:
        return []
    # Join distinct summary parts with a blank line so they don't run together when the
    # transcript accumulator concatenates consecutive thinking deltas.
    return [ModelStreamEvent.thinking("\n\n".join(texts))]


async def openai_stream_events(
    events: AsyncIterator[Mapping[str, Any]], *, reasoning_state: str = "inline"
) -> AsyncIterator[ModelStreamEvent]:
    pending_function_calls: dict[int, _PendingFunctionCall] = {}
    fallback_output_items: dict[int, dict[str, Any]] = {}
    completed = False
    async for event in events:
        if not isinstance(event, Mapping):
            raise OpenAIProtocolError("OpenAI stream event must be a JSON object.")
        event_type = event.get("type")
        if event_type == "response.output_text.delta":
            delta = event.get("delta")
            if not isinstance(delta, str):
                raise OpenAIProtocolError("OpenAI output_text delta must be a string.")
            if delta:
                yield ModelStreamEvent.text_delta(delta)
            continue
        if event_type == "response.refusal.delta":
            delta = event.get("delta")
            if not isinstance(delta, str):
                raise OpenAIProtocolError("OpenAI refusal delta must be a string.")
            if delta:
                yield ModelStreamEvent.text_delta(delta)
            continue
        if event_type in {
            "response.reasoning_summary_text.delta",
            "response.reasoning_text.delta",
        }:
            # Display-only readable reasoning; the encrypted reasoning item still
            # round-trips via response.output_item.done -> provider state.
            delta = event.get("delta")
            if not isinstance(delta, str):
                raise OpenAIProtocolError("OpenAI reasoning delta must be a string.")
            if delta:
                yield ModelStreamEvent.thinking(delta)
            continue
        if event_type == "response.output_item.added":
            _record_stream_output_item_added(event, pending_function_calls)
            continue
        if event_type == "response.output_item.done":
            _record_stream_output_item_done(event, fallback_output_items)
            continue
        if event_type == "response.function_call_arguments.delta":
            _record_stream_function_call_delta(event, pending_function_calls)
            continue
        if event_type == "response.function_call_arguments.done":
            tool_call_event, output_item = _stream_function_call_event(
                event,
                pending_function_calls,
            )
            fallback_output_items[_stream_output_index(event)] = output_item
            yield tool_call_event
            continue
        if event_type in {"response.completed", "response.incomplete"}:
            yield _stream_completed_event(
                event, fallback_output_items, reasoning_state=reasoning_state
            )
            completed = True
            continue
        if event_type in {"response.failed", "error"}:
            _raise_openai_stream_context_overflow_if_applicable(event)
            raise OpenAIAPIError(f"OpenAI streaming error: {_stream_error_message(event)}")

    if not completed:
        raise OpenAIProtocolError("OpenAI streaming response ended before response.completed.")


class _PendingFunctionCall:
    def __init__(
        self,
        *,
        item_id: str | None,
        call_id: str | None,
        name: str | None,
        arguments: str,
    ) -> None:
        self.item_id = item_id
        self.call_id = call_id
        self.name = name
        self.arguments_parts = [arguments] if arguments else []

    def append_arguments(self, delta: str) -> None:
        self.arguments_parts.append(delta)

    @property
    def arguments(self) -> str:
        return "".join(self.arguments_parts)


def _message_output_events(
    item: Mapping[str, Any],
    item_index: int,
) -> list[ModelStreamEvent]:
    role = item.get("role")
    if role != "assistant":
        raise OpenAIProtocolError(
            f"OpenAI message output item {item_index} must have assistant role."
        )
    content = item.get("content")
    if not isinstance(content, list):
        raise OpenAIProtocolError(
            f"OpenAI message output item {item_index} content must be a list."
        )
    events: list[ModelStreamEvent] = []
    for content_index, part in enumerate(content):
        if not isinstance(part, Mapping):
            raise OpenAIProtocolError(
                f"OpenAI message output content {item_index}.{content_index} must be an object."
            )
        part = cast("Mapping[str, Any]", part)
        part_type = part.get("type")
        if part_type == "output_text":
            text_key = "text"
        elif part_type == "refusal":
            text_key = "refusal"
        else:
            raise OpenAIProtocolError(
                f"Unsupported OpenAI message output content type: {part_type!r}."
            )
        text = part.get(text_key)
        if not isinstance(text, str):
            raise OpenAIProtocolError(f"OpenAI {part_type} content requires string {text_key}.")
        if text:
            events.append(ModelStreamEvent.text_delta(text))
    return events


def _function_call_event(
    item: Mapping[str, Any],
    item_index: int,
) -> ModelStreamEvent:
    call_id = item.get("call_id")
    name = item.get("name")
    arguments = item.get("arguments")
    if not isinstance(call_id, str) or not call_id.strip():
        raise OpenAIProtocolError(
            f"OpenAI function_call item {item_index} requires nonblank call_id."
        )
    if not isinstance(name, str) or not name.strip():
        raise OpenAIProtocolError(f"OpenAI function_call item {item_index} requires nonblank name.")
    if not isinstance(arguments, str):
        raise OpenAIProtocolError(
            f"OpenAI function_call item {item_index} requires string arguments."
        )
    try:
        decoded_arguments = json.loads(arguments)
    except ValueError as exc:
        raise OpenAIProtocolError(
            f"OpenAI function_call item {item_index} arguments were not valid JSON."
        ) from exc
    if type(decoded_arguments) is not dict:
        raise OpenAIProtocolError(
            f"OpenAI function_call item {item_index} arguments must decode to an object."
        )
    return ModelStreamEvent.tool_call(
        id=call_id,
        name=name,
        arguments=copy_json_value(decoded_arguments, "arguments"),
    )


def _completed_event_from_response(
    response: Mapping[str, Any],
    provider_state_items: list[dict[str, Any]] | None = None,
    *,
    completion_output_items: list[Mapping[str, Any]] | None = None,
    reasoning_state: str = "inline",
) -> ModelStreamEvent:
    if provider_state_items is None:
        provider_state_items = _provider_state_items_from_response(response)
    if reasoning_state == "server":
        response_id = _optional_string(response, "id")
        if response_id:
            provider_state_items = [
                *provider_state_items,
                {"provider": "openai", "state": {"type": "response_ref", "id": response_id}},
            ]
    payload = {
        "id": _optional_string(response, "id"),
        "model": _optional_string(response, "model"),
        "status": _optional_string(response, "status"),
        "provider_state": provider_state_items,
        "usage": copy_json_value(response.get("usage"), "usage"),
        "incomplete_details": copy_json_value(
            response.get("incomplete_details"),
            "incomplete_details",
        ),
    }
    return ModelStreamEvent(
        type=ModelStreamEventType.COMPLETED,
        payload=payload,
        completion=_openai_completion_from_response(
            response,
            output_items=completion_output_items,
        ),
    )


def _openai_completion_from_response(
    response: Mapping[str, Any],
    *,
    output_items: list[Mapping[str, Any]] | None = None,
) -> ModelCompletion:
    status = _optional_string(response, "status")
    raw_finish_reason = _openai_raw_finish_reason(response)
    if status == "failed":
        finish_reason = ModelFinishReason.ERROR
    elif status == "incomplete":
        finish_reason = _openai_incomplete_finish_reason(raw_finish_reason)
    elif _output_items_have_function_call(
        output_items if output_items is not None else _openai_output_items(response)
    ):
        finish_reason = ModelFinishReason.TOOL_CALLS
    elif status == "completed":
        finish_reason = ModelFinishReason.STOP
    else:
        finish_reason = ModelFinishReason.UNKNOWN
    return ModelCompletion(
        finish_reason=finish_reason,
        raw_finish_reason=raw_finish_reason,
        status=status,
    )


def _openai_raw_finish_reason(response: Mapping[str, Any]) -> str | None:
    incomplete_details = response.get("incomplete_details")
    if isinstance(incomplete_details, Mapping):
        return _optional_string(incomplete_details, "reason")
    return None


def _openai_incomplete_finish_reason(raw_finish_reason: str | None) -> ModelFinishReason:
    if raw_finish_reason in {"max_output_tokens", "max_tokens", "length"}:
        return ModelFinishReason.LENGTH
    if raw_finish_reason in {"content_filter", "safety", "refusal"}:
        return ModelFinishReason.CONTENT_FILTER
    return ModelFinishReason.UNKNOWN


def _openai_output_items(response: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    output = response.get("output")
    if not isinstance(output, list):
        return []
    return [item for item in output if isinstance(item, Mapping)]


def _output_items_have_function_call(output_items: list[Mapping[str, Any]]) -> bool:
    return any(item.get("type") == "function_call" for item in output_items)


def _stream_completed_event(
    event: Mapping[str, Any],
    fallback_output_items: Mapping[int, Mapping[str, Any]],
    *,
    reasoning_state: str = "inline",
) -> ModelStreamEvent:
    response = _stream_response_object(event)
    if response.get("output") is None:
        provider_state_items = _provider_state_items_from_output_items(fallback_output_items)
        completion_output_items = list(_sorted_output_items(fallback_output_items))
        return _completed_event_from_response(
            response,
            provider_state_items,
            completion_output_items=completion_output_items,
            reasoning_state=reasoning_state,
        )
    return _completed_event_from_response(response, reasoning_state=reasoning_state)


def _provider_state_items_from_response(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = response.get("output")
    if output is None:
        return []
    if not isinstance(output, list):
        raise OpenAIProtocolError("OpenAI response output must be a list.")
    provider_state_items: list[dict[str, Any]] = []
    for index, item in enumerate(output):
        if not isinstance(item, Mapping):
            raise OpenAIProtocolError(f"OpenAI output item {index} must be an object.")
        item = cast("Mapping[str, Any]", item)
        item_type = item.get("type")
        if item_type in {"reasoning", "message", "function_call"}:
            provider_state_items.append(
                {"provider": "openai", "state": copy_json_value(item, "output_item")}
            )
            continue
        raise OpenAIProtocolError(f"Unsupported OpenAI output item type: {item_type!r}.")
    return provider_state_items


def _provider_state_items_from_output_items(
    output_items: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    provider_state_items: list[dict[str, Any]] = []
    for item in _sorted_output_items(output_items):
        provider_state_items.append(
            {"provider": "openai", "state": copy_json_value(item, "output_item")}
        )
    return provider_state_items


def _sorted_output_items(
    output_items: Mapping[int, Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return [output_items[output_index] for output_index in sorted(output_items)]


def _record_stream_output_item_added(
    event: Mapping[str, Any],
    pending_function_calls: dict[int, _PendingFunctionCall],
) -> None:
    output_index = _stream_output_index(event)
    item = event.get("item")
    if not isinstance(item, Mapping):
        raise OpenAIProtocolError("OpenAI output_item.added requires item object.")
    item_type = item.get("type")
    if item_type != "function_call":
        return
    pending_function_calls[output_index] = _PendingFunctionCall(
        item_id=_mapping_optional_string(item, "id"),
        call_id=_mapping_optional_string(item, "call_id"),
        name=_mapping_optional_string(item, "name"),
        arguments=_mapping_string_or_default(item, "arguments", ""),
    )


def _record_stream_output_item_done(
    event: Mapping[str, Any],
    output_items: dict[int, dict[str, Any]],
) -> None:
    output_index = _stream_output_index(event)
    item = event.get("item")
    if not isinstance(item, Mapping):
        raise OpenAIProtocolError("OpenAI output_item.done requires item object.")
    item_type = item.get("type")
    if item_type in {"reasoning", "message", "function_call"}:
        output_items[output_index] = copy_json_value(item, "output_item")


def _record_stream_function_call_delta(
    event: Mapping[str, Any],
    pending_function_calls: dict[int, _PendingFunctionCall],
) -> None:
    output_index = _stream_output_index(event)
    pending = pending_function_calls.get(output_index)
    if pending is None:
        raise OpenAIProtocolError(
            "OpenAI function_call_arguments.delta arrived before output_item.added."
        )
    item_id = _mapping_optional_string(event, "item_id")
    if pending.item_id is not None and item_id is not None and pending.item_id != item_id:
        raise OpenAIProtocolError("OpenAI function_call_arguments.delta item_id mismatch.")
    delta = event.get("delta")
    if not isinstance(delta, str):
        raise OpenAIProtocolError("OpenAI function_call_arguments.delta requires string delta.")
    pending.append_arguments(delta)


def _stream_function_call_event(
    event: Mapping[str, Any],
    pending_function_calls: dict[int, _PendingFunctionCall],
) -> tuple[ModelStreamEvent, dict[str, Any]]:
    output_index = _stream_output_index(event)
    pending = pending_function_calls.pop(output_index, None)
    if pending is None:
        raise OpenAIProtocolError(
            "OpenAI function_call_arguments.done arrived before output_item.added."
        )
    item_id = _mapping_optional_string(event, "item_id")
    if pending.item_id is not None and item_id is not None and pending.item_id != item_id:
        raise OpenAIProtocolError("OpenAI function_call_arguments.done item_id mismatch.")
    call_id = _first_nonblank_string(pending.call_id)
    name = _first_nonblank_string(
        _mapping_optional_string(event, "name"),
        pending.name,
    )
    arguments = _first_string(
        _mapping_optional_string(event, "arguments"),
        pending.arguments if pending.arguments else None,
    )
    output_item = {
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "status": "completed",
    }
    output_item_id = _first_string_or_none(
        item_id,
        pending.item_id,
    )
    if output_item_id is not None:
        output_item["id"] = output_item_id
    return (
        _function_call_event(output_item, output_index),
        output_item,
    )


def _stream_response_object(event: Mapping[str, Any]) -> Mapping[str, Any]:
    response = event.get("response")
    if response is None:
        response = event
    if not isinstance(response, Mapping):
        raise OpenAIProtocolError("OpenAI stream terminal event requires response object.")
    return response


def _stream_error_message(event: Mapping[str, Any]) -> str:
    event_type = event.get("type")
    if event_type == "response.failed":
        response = _stream_response_object(event)
        error = response.get("error")
        if error is not None:
            return _safe_error_value(error)
        return _safe_error_value(response)
    if event_type == "error":
        return _safe_error_json(event)
    return _safe_error_value(event)


def _raise_openai_stream_context_overflow_if_applicable(event: Mapping[str, Any]) -> None:
    event_type = event.get("type")
    error: Mapping[str, Any] | None = None
    request_id = None
    if event_type == "response.failed":
        response = _stream_response_object(event)
        response_error = response.get("error")
        if isinstance(response_error, Mapping):
            error = response_error
        request_id_value = response.get("request_id")
        if isinstance(request_id_value, str):
            request_id = request_id_value
    elif event_type == "error":
        error = event
    if error is None:
        return
    _raise_openai_context_overflow_from_error(
        status_code=None,
        error=error,
        request_id=request_id,
        response_body=_stream_error_message(event),
    )


def _stream_output_index(event: Mapping[str, Any]) -> int:
    output_index = event.get("output_index")
    if type(output_index) is not int:
        raise OpenAIProtocolError("OpenAI stream event requires integer output_index.")
    if output_index < 0:
        raise OpenAIProtocolError("OpenAI stream event output_index must be non-negative.")
    return output_index


def _mapping_optional_string(value: Mapping[str, Any] | None, key: str) -> str | None:
    if value is None:
        return None
    raw_value = value.get(key)
    if raw_value is None:
        return None
    if not isinstance(raw_value, str):
        raise OpenAIProtocolError(f"OpenAI stream field {key} must be a string.")
    stripped = raw_value.strip()
    return stripped or None


def _mapping_string_or_default(value: Mapping[str, Any], key: str, default: str) -> str:
    raw_value = value.get(key, default)
    if not isinstance(raw_value, str):
        raise OpenAIProtocolError(f"OpenAI stream field {key} must be a string.")
    return raw_value


def _first_nonblank_string(*values: str | None) -> str:
    for value in values:
        if value is not None and value.strip():
            return value
    raise OpenAIProtocolError("OpenAI streaming function call is missing required identity.")


def _first_string(*values: str | None) -> str:
    for value in values:
        if value is not None:
            return value
    raise OpenAIProtocolError("OpenAI streaming function call is missing arguments.")


def _first_string_or_none(*values: str | None) -> str | None:
    for value in values:
        if value is not None:
            return value
    return None


def _openai_options(options: Mapping[str, Any]) -> dict[str, Any]:
    raw_options = options.get("openai", {})
    if raw_options is None:
        return {}
    if type(raw_options) is not dict:
        raise ValueError("ModelRequest options.openai must be an object.")
    copied = copy_json_value(raw_options, "options.openai")
    for key in copied:
        if key in _RESERVED_OPENAI_OPTIONS:
            raise ValueError(f"OpenAI option is reserved: {key}")
    return copied


def _openai_structured_output_format(options: Mapping[str, Any]) -> dict[str, Any] | None:
    raw = options.get("structured_output")
    if raw is None:
        return None
    if type(raw) is not dict:
        raise ValueError("ModelRequest options.structured_output must be an object.")
    strategy = raw.get("strategy", "tool")
    if strategy != "native":
        return None
    schema = raw.get("schema")
    if type(schema) is not dict:
        raise ValueError("Native structured output schema must be an object.")
    name = raw.get("name") or "structured_output"
    if not isinstance(name, str):
        raise ValueError("Native structured output name must be a string.")
    return {
        "type": "json_schema",
        "name": require_clean_nonblank(name, "structured_output.name"),
        "schema": copy_json_value(schema, "structured_output.schema"),
        "strict": True,
    }


def _system_text(messages: list[Message]) -> str:
    system_parts: list[str] = []
    for message in messages:
        if message.role != MessageRole.SYSTEM:
            continue
        for part in message.content:
            if type(part) is TextPart:
                system_parts.append(part.text)
    return "\n\n".join(system_parts)


def _openai_input_items(
    message: Message,
    *,
    resolved_attachments: dict[str, dict[str, Any]],
    reasoning_state: str = "inline",
    use_provider_state: bool = True,
) -> list[dict[str, Any]]:
    if message.role == MessageRole.SYSTEM:
        return []
    if message.role == MessageRole.USER:
        return [
            {
                "role": "user",
                "content": [
                    _user_input_part(part, resolved_attachments) for part in message.content
                ],
            }
        ]
    if message.role == MessageRole.ASSISTANT:
        provider_state_items = _openai_provider_state_items(
            message, reasoning_state=reasoning_state, use_provider_state=use_provider_state
        )
        if provider_state_items:
            return provider_state_items

        items: list[dict[str, Any]] = []
        text_parts = [_output_text_part(part) for part in message.content if type(part) is TextPart]
        if text_parts:
            items.append(
                {
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": text_parts,
                }
            )
        for part in message.content:
            if type(part) is ToolCallPart:
                items.append(_function_call_input_item(part))
            elif type(part) not in {TextPart, ProviderStatePart, ThinkingPart}:
                raise OpenAIProtocolError(
                    "Assistant messages can only contain text, tool_call, provider_state, "
                    "and thinking parts."
                )
        # ThinkingPart is display-only here: OpenAI reasoning round-trips through the
        # encrypted reasoning ProviderStatePart, so the readable summary is not re-sent.
        return items
    if message.role == MessageRole.TOOL:
        items: list[dict[str, Any]] = []
        attachment_parts: list[dict[str, Any]] = []
        for part in message.content:
            items.append(_function_call_output_item(part))
            if type(part) is ToolResultPart:
                attachment_parts.extend(_openai_file_attachment_parts(part, resolved_attachments))
        if attachment_parts:
            items.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "The previous tool result returned file content for inspection.",
                        },
                        *attachment_parts,
                    ],
                }
            )
        return items
    raise OpenAIProtocolError(f"Unsupported Cayu message role: {message.role!r}.")


def _server_chain(messages: list[Message]) -> tuple[str | None, list[Message]]:
    """Return (previous_response_id, messages_to_send) for server mode.

    Finds the latest assistant message carrying a response_ref marker. Everything
    at or before it already lives on OpenAI's servers, so only later messages are
    sent. No marker found -> (None, all messages) for a full first send.
    """
    last_index: int | None = None
    last_id: str | None = None
    for index, message in enumerate(messages):
        if message.role != MessageRole.ASSISTANT:
            continue
        for part in message.content:
            if type(part) is not ProviderStatePart or part.provider != "openai":
                continue
            state = part.state
            if isinstance(state, dict) and state.get("type") == "response_ref":
                response_id = state.get("id")
                if isinstance(response_id, str) and response_id:
                    last_index = index
                    last_id = response_id
    if last_index is None:
        return None, messages
    return last_id, messages[last_index + 1 :]


def _openai_provider_state_items(
    message: Message, *, reasoning_state: str = "inline", use_provider_state: bool = True
) -> list[dict[str, Any]]:
    if not use_provider_state:
        return []
    items: list[dict[str, Any]] = []
    for part in message.content:
        if type(part) is not ProviderStatePart:
            continue
        if part.provider != "openai":
            continue
        state = copy_json_value(part.state, "provider_state")
        if type(state) is not dict:
            raise OpenAIProtocolError("OpenAI provider state must be an object.")
        item_type = state.get("type")
        if item_type == "response_ref":
            continue  # synthetic chain marker, never sent as input
        if item_type == "reasoning":
            # Inline mode replays reasoning with its encrypted_content; server mode
            # leaves reasoning on OpenAI's servers, so never replays it.
            if reasoning_state == "server":
                continue
            items.append(state)
            continue
        if item_type not in {"message", "function_call"}:
            raise OpenAIProtocolError(
                f"Unsupported OpenAI provider state item type: {item_type!r}."
            )
        items.append(state)
    return items


def _user_input_part(
    part: TextPart | ToolCallPart | ToolResultPart | ProviderStatePart | ThinkingPart | FilePart,
    resolved_attachments: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if type(part) is TextPart:
        return {"type": "input_text", "text": part.text}
    if type(part) is FilePart:
        return _openai_file_attachment_part(_resolved_user_attachment(part, resolved_attachments))
    raise OpenAIProtocolError("User messages can only contain text and file parts.")


def _resolved_user_attachment(
    part: FilePart,
    resolved_attachments: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    attachment = file_attachment_from_payload(part.attachment)
    if attachment is None:
        raise OpenAIProtocolError("User file parts require a file attachment payload.")
    resolved = resolved_attachments.get(attachment.artifact_id)
    if resolved is None:
        raise OpenAIProtocolError(f"Missing resolved file attachment: {attachment.artifact_id}")
    return resolved


def _output_text_part(
    part: TextPart | ToolCallPart | ToolResultPart | ProviderStatePart | ThinkingPart | FilePart,
) -> dict[str, str]:
    if type(part) is not TextPart:
        raise OpenAIProtocolError("Assistant text output requires a text part.")
    return {"type": "output_text", "text": part.text}


def _function_call_input_item(part: ToolCallPart) -> dict[str, Any]:
    return {
        "type": "function_call",
        "call_id": part.tool_call_id,
        "name": part.tool_name,
        "arguments": _json_arguments(part.arguments),
        "status": "completed",
    }


def _function_call_output_item(
    part: TextPart | ToolCallPart | ToolResultPart | ProviderStatePart | ThinkingPart | FilePart,
) -> dict[str, Any]:
    if type(part) is not ToolResultPart:
        raise OpenAIProtocolError("Tool messages can only contain tool_result parts.")
    return {
        "type": "function_call_output",
        "call_id": part.tool_call_id,
        "output": part.content,
    }


def _openai_file_attachment_parts(
    part: ToolResultPart,
    resolved_attachments: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for payload in part.artifacts:
        attachment = file_attachment_from_payload(payload)
        if attachment is None:
            continue
        resolved = resolved_attachments.get(attachment.artifact_id)
        if resolved is None:
            raise OpenAIProtocolError(f"Missing resolved file attachment: {attachment.artifact_id}")
        parts.append(_openai_file_attachment_part(resolved))
    return parts


def _openai_file_attachment_part(resolved: dict[str, Any]) -> dict[str, Any]:
    kind = FileAttachmentKind(resolved["kind"])
    data_url = f"data:{resolved['content_type']};base64,{resolved['data_base64']}"
    if kind == FileAttachmentKind.IMAGE:
        return {
            "type": "input_image",
            "image_url": data_url,
        }
    if kind == FileAttachmentKind.DOCUMENT:
        return {
            "type": "input_file",
            "filename": resolved["filename"],
            "file_data": data_url,
        }
    raise OpenAIProtocolError(f"Unsupported file attachment kind: {kind!r}")


def _json_arguments(arguments: Mapping[str, Any]) -> str:
    copied = copy_json_value(arguments, "arguments")
    if type(copied) is not dict:
        raise OpenAIProtocolError("Tool call arguments must be an object.")
    return json.dumps(copied, sort_keys=True, separators=(",", ":"))


def _openai_tool(tool: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(tool, Mapping):
        raise ValueError("Tool definitions must be objects.")
    name = _require_mapping_string(tool, "name")
    if not _OPENAI_TOOL_NAME_RE.fullmatch(name):
        raise ValueError(
            "OpenAI tool names must contain 1-64 letters, numbers, underscores, or hyphens."
        )
    description = tool.get("description", "")
    if not isinstance(description, str):
        raise ValueError("Tool description must be a string.")
    input_schema = tool.get("input_schema", {})
    if type(input_schema) is not dict:
        raise ValueError("Tool input_schema must be an object.")
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": copy_json_value(input_schema, "input_schema"),
        "strict": False,
    }


def _require_mapping_string(value: Mapping[str, Any], key: str) -> str:
    raw_value = value.get(key)
    if not isinstance(raw_value, str):
        raise ValueError(f"Tool {key} must be a string.")
    return require_clean_nonblank(raw_value, f"tool.{key}")


def _optional_string(response: Mapping[str, Any], key: str) -> str | None:
    value = response.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise OpenAIProtocolError(f"OpenAI response {key} must be a string.")
    return value


def _copy_headers(headers: Mapping[str, str] | None) -> dict[str, str]:
    return copy_headers(headers, protected=_PROTECTED_HEADER_NAMES)


def _validate_base_url(base_url: str) -> str:
    return validate_base_url(base_url, provider_label="OpenAI")


def _validate_url(url: str, field_name: str) -> str:
    return validate_url(url, field_name, provider_label="OpenAI")


def _safe_error_response_text(response: httpx.Response) -> str:
    return safe_error_response_text(response, format_error_json=_format_error_json)


def _format_error_json(decoded: Any) -> str | None:
    if not isinstance(decoded, Mapping):
        return None
    return _safe_error_json(decoded)


def _openai_api_error_from_response(response: httpx.Response, message: str) -> OpenAIAPIError:
    """Build a structured `OpenAIAPIError` from an HTTP error response.

    Keeps the OpenAI error body's typed identity (status/type/code/param/
    request_id) on the exception so callers classify failures — e.g. the
    stale-chain recovery in `OpenAIProvider.stream` — without re-parsing
    message text.
    """
    decoded = response_json_object(response)
    error: Mapping[str, Any] = {}
    request_id: str | None = None
    if decoded is not None:
        raw_error = decoded.get("error")
        error = raw_error if isinstance(raw_error, Mapping) else decoded
        raw_request_id = decoded.get("request_id")
        request_id = raw_request_id if isinstance(raw_request_id, str) else None
    return OpenAIAPIError(
        message,
        status_code=response.status_code,
        error_type=optional_error_string(error.get("type")),
        error_code=optional_error_string(error.get("code")),
        param=optional_error_string(error.get("param")),
        request_id=optional_error_string(request_id),
        response_body=_safe_error_response_text(response),
    )


def _raise_openai_context_overflow_if_applicable(response: httpx.Response) -> None:
    decoded = response_json_object(response)
    if decoded is None:
        return
    error = decoded.get("error")
    request_id = decoded.get("request_id")
    if not isinstance(error, Mapping):
        error = decoded
    _raise_openai_context_overflow_from_error(
        status_code=response.status_code,
        error=error,
        request_id=request_id if isinstance(request_id, str) else None,
        response_body=_safe_error_response_text(response),
    )


def _raise_openai_context_overflow_from_error(
    *,
    status_code: int | None,
    error: Mapping[str, Any],
    request_id: str | None,
    response_body: str,
) -> None:
    code = optional_error_string(error.get("code"))
    error_type = optional_error_string(error.get("type"))
    message = optional_error_string(error.get("message"))
    if not _is_openai_context_overflow(code=code, message=message):
        return
    raise OpenAIContextOverflowError(
        "OpenAI model context overflow",
        status_code=status_code,
        error_type=error_type,
        error_code=code,
        request_id=request_id,
        response_body=response_body,
    )


def _is_openai_context_overflow(*, code: str | None, message: str | None) -> bool:
    if code == "context_length_exceeded":
        return True
    if message is None:
        return False
    normalized = message.lower()
    return (
        "context_length_exceeded" in normalized
        or "context length exceeded" in normalized
        or "maximum context length" in normalized
        or "exceeds the context window" in normalized
    )


def _safe_error_json(decoded: Mapping[str, Any]) -> str:
    return safe_error_json(decoded, include_request_id=True)


def _safe_error_value(value: Any) -> str:
    if isinstance(value, Mapping):
        return _safe_error_json(value)
    if isinstance(value, str):
        return truncate_error_text(value)
    return truncate_error_text(str(value))


_STALE_CHAIN_ERROR_CODE = "previous_response_not_found"
_STALE_CHAIN_PARAM = "previous_response_id"


def _is_stale_chain_error(exc: Exception) -> bool:
    """Classify a stale server-side chain from typed error identity.

    OpenAI reports a stale/expired ``previous_response_id`` as HTTP 404 with
    ``code: "previous_response_not_found"`` (``param: "previous_response_id"``).
    Classification reads only the structured fields captured on
    `OpenAIAPIError` — never message text — so unrelated errors that merely
    mention the field (e.g. a 400 for combining it with ``conversation``) are
    not misclassified as recoverable.
    """
    if not isinstance(exc, OpenAIAPIError):
        return False
    if exc.error_code == _STALE_CHAIN_ERROR_CODE:
        return True
    return exc.status_code == 404 and exc.param == _STALE_CHAIN_PARAM


def _validate_reasoning_state(value: str) -> str:
    if value not in _VALID_REASONING_STATES:
        raise ValueError(f"reasoning_state must be one of {sorted(_VALID_REASONING_STATES)}.")
    return value
