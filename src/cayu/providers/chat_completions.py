from __future__ import annotations

import asyncio
import json
import math
import re
from collections.abc import AsyncIterator, Mapping
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import unquote_plus, urlencode, urlsplit, urlunsplit

from cayu._validation import copy_json_value, require_clean_nonblank, require_finite
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
from cayu.providers._api_keys import resolve_api_key
from cayu.providers._credential_boundary import (
    aclosing_provider_stream,
    detach_provider_stream_traceback,
)
from cayu.providers._http import (
    OMITTED_PROVIDER_ERROR_BODY,
    SharedAsyncClient,
    _trusted_sse_retry_after_s,
    aclose_transport,
    copy_headers,
    credential_safe_error_event,
    credential_safe_post_completion_failure,
    credential_safe_provider_exception,
    credential_sanitization_values,
    optional_error_string,
    response_json_object,
    safe_error_json,
    safe_error_response_text,
    sanitize_provider_cancellation,
    stream_sse_json_events,
    validate_url,
)
from cayu.providers.base import (
    ModelContextOverflowError,
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelStreamEvent,
    ModelStreamEventType,
    UsageDialect,
    copy_usage_dialect,
    privacy_safe_provider_option_projection,
)

if TYPE_CHECKING:
    import httpx

# base_url follows the OpenAI-SDK convention: it includes the version path, and
# the endpoint appends only "/chat/completions". So OpenAI is ".../v1", Gemini is
# ".../v1beta/openai", Together is ".../v1", Azure is ".../deployments/<dep>".
DEFAULT_CHAT_COMPLETIONS_BASE_URL = "https://api.openai.com/v1"
DEFAULT_CHAT_COMPLETIONS_TIMEOUT_SECONDS = 60.0
DEFAULT_CHAT_COMPLETIONS_STREAM_IDLE_TIMEOUT_SECONDS = 120.0
DEFAULT_CHAT_COMPLETIONS_API_KEY_ENV = "OPENAI_API_KEY"
# OpenAI/Together use `Authorization: Bearer <key>`; Azure uses `api-key: <key>`.
DEFAULT_CHAT_COMPLETIONS_AUTH_HEADER = "Authorization"
DEFAULT_CHAT_COMPLETIONS_AUTH_VALUE_PREFIX = "Bearer "
_USAGE_DIALECT_UNDECLARED = object()

_RESERVED_CHAT_COMPLETIONS_OPTIONS = {
    "model",
    "messages",
    "tools",
    "stream",
    "stream_options",
}
_CHAT_COMPLETIONS_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
# JSON Schema keys rejected by some OpenAI-compatible vendors (notably Google
# Gemini), stripped recursively when clean_schemas is enabled.
_UNSUPPORTED_SCHEMA_KEYS = {"additionalProperties", "$schema"}
# JSON Schema keys whose values are name->subschema maps (arbitrary property
# names, not schema keywords), so their keys must be preserved when cleaning.
_SUBSCHEMA_MAP_KEYS = {"properties", "patternProperties", "$defs", "definitions"}
# How PDF/document attachments are encoded as a content part. OpenAI/Azure expect
# the `file` part; Google Gemini's compatible endpoint rejects `file` and instead
# accepts a PDF data URL through the `image_url` part. There is no single portable
# shape, so this is selectable per provider instance.
DEFAULT_DOCUMENT_ENCODING = "file"
_VALID_DOCUMENT_ENCODINGS = {"file", "image_url"}

_TOOL_RESULT_ATTACHMENT_LEAD_IN = "The previous tool result returned file content for inspection."


class ChatCompletionsError(RuntimeError):
    """Base error for Chat Completions provider failures."""


class ChatCompletionsAPIError(ChatCompletionsError, ModelProviderError):
    """Raised when the Chat Completions HTTP API returns an error response."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_type: str | None = None,
        error_code: str | None = None,
        request_id: str | None = None,
        retryable: bool | None = None,
        retry_after_s: float | None = None,
        response_body: str | None = None,
    ) -> None:
        ModelProviderError.__init__(
            self,
            message,
            provider="chat_completions",
            status_code=status_code,
            error_type=error_type,
            error_code=error_code,
            request_id=request_id,
            retryable=retryable,
            retry_after_s=retry_after_s,
            response_body=response_body,
        )


class ChatCompletionsContextOverflowError(
    ChatCompletionsAPIError,
    ModelContextOverflowError,
):
    """Raised when a Chat Completions provider reports context overflow."""

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
            provider="chat_completions",
            status_code=status_code,
            error_type=error_type,
            error_code=error_code,
            request_id=request_id,
            response_body=response_body,
        )


class ChatCompletionsProtocolError(ChatCompletionsError):
    """Raised when data does not match the expected Chat Completions shape."""


class ChatCompletionsTransport(Protocol):
    def stream_chat_completions(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
        stream_idle_timeout_s: float,
    ) -> AsyncIterator[Mapping[str, Any]]:
        """POST a streaming Chat Completions payload and yield decoded SSE data objects."""


class HttpxChatCompletionsTransport:
    """HTTP transport with explicit certifi-backed TLS verification.

    Owns one shared httpx.AsyncClient (created lazily) that is reused across
    requests so each model call does not pay for a fresh TLS handshake. Close it
    with :meth:`aclose` when the transport is no longer needed.
    """

    def __init__(self, *, allow_http: bool = False) -> None:
        if type(allow_http) is not bool:
            raise TypeError("allow_http must be a bool.")
        self.allow_http = allow_http
        self._client = SharedAsyncClient()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def stream_chat_completions(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
        stream_idle_timeout_s: float,
    ) -> AsyncIterator[Mapping[str, Any]]:
        timeout_s = _validate_timeout_s(timeout_s)
        url = _validate_url(url, "url", allow_http=self.allow_http)
        events = stream_sse_json_events(
            client=self._client.get(),
            url=url,
            headers=headers,
            payload=payload,
            timeout_s=timeout_s,
            stream_idle_timeout_s=stream_idle_timeout_s,
            request_label="Chat Completions API",
            response_label="Chat Completions",
            api_error=ChatCompletionsAPIError,
            protocol_error=ChatCompletionsProtocolError,
            error_response_text=_safe_error_response_text,
            raise_context_overflow=_raise_chat_context_overflow_if_applicable,
            api_error_from_response=_chat_api_error_from_response,
        )
        async with aclosing_provider_stream(events):
            async for event in events:
                yield event


class ChatCompletionsProvider(ModelProvider):
    """Adapter for OpenAI-compatible ``/v1/chat/completions`` services.

    Many providers expose the OpenAI Chat Completions wire format without the
    newer Responses API: Google Gemini (AI Studio), Azure OpenAI, Together,
    Fireworks, Mistral, Ollama, vLLM, and others. This single adapter targets
    that shared format so those providers work through Cayu's provider-neutral
    runtime. ``OpenAIProvider`` remains the adapter for OpenAI's Responses API.

    The model is resolved from the agent's ``AgentSpec`` (and ``ModelRequest``),
    not from this provider, matching ``OpenAIProvider``/``AnthropicProvider``.
    """

    name = "openai_chat"
    usage_dialect = UsageDialect.OPENAI

    def __init__(
        self,
        *,
        api_key: str | None = None,
        name: str = "openai_chat",
        base_url: str = DEFAULT_CHAT_COMPLETIONS_BASE_URL,
        endpoint_url: str | None = None,
        api_key_env: str = DEFAULT_CHAT_COMPLETIONS_API_KEY_ENV,
        auth_header: str = DEFAULT_CHAT_COMPLETIONS_AUTH_HEADER,
        auth_value_prefix: str = DEFAULT_CHAT_COMPLETIONS_AUTH_VALUE_PREFIX,
        allow_http: bool = False,
        stream_include_usage: bool = True,
        timeout_s: float = DEFAULT_CHAT_COMPLETIONS_TIMEOUT_SECONDS,
        stream_idle_timeout_s: float = DEFAULT_CHAT_COMPLETIONS_STREAM_IDLE_TIMEOUT_SECONDS,
        transport: ChatCompletionsTransport | None = None,
        extra_headers: Mapping[str, str] | None = None,
        api_version: str | None = None,
        clean_schemas: bool = True,
        document_encoding: str = DEFAULT_DOCUMENT_ENCODING,
        usage_dialect: UsageDialect | str | None = None,
    ) -> None:
        self.name = require_clean_nonblank(name, "name")
        self.api_key_env = require_clean_nonblank(api_key_env, "api_key_env")
        self.api_key = resolve_api_key(
            api_key=api_key,
            env_var=self.api_key_env,
            provider_name="ChatCompletionsProvider",
            missing_hint=(
                f"set the {self.api_key_env} environment variable or pass api_key=... "
                "to ChatCompletionsProvider(...)."
            ),
        )
        # Auth header is configurable: OpenAI/Together use Authorization: Bearer,
        # Azure uses an `api-key` header (empty prefix).
        self.auth_header = require_clean_nonblank(auth_header, "auth_header")
        if type(auth_value_prefix) is not str:
            raise TypeError("auth_value_prefix must be a string.")
        self.auth_value_prefix = auth_value_prefix
        if type(allow_http) is not bool:
            raise TypeError("allow_http must be a bool.")
        self.allow_http = allow_http
        self.base_url = _validate_base_url(base_url, allow_http=allow_http)
        self.endpoint_url = (
            _validate_url(endpoint_url, "endpoint_url", allow_http=allow_http)
            if endpoint_url is not None
            else None
        )
        effective_url = self.endpoint_url or self.base_url
        if usage_dialect is not None:
            self.usage_dialect = copy_usage_dialect(usage_dialect)
        else:
            declared_usage_dialect = _declared_subclass_usage_dialect(type(self))
            if declared_usage_dialect is _USAGE_DIALECT_UNDECLARED:
                self.usage_dialect = (
                    UsageDialect.GEMINI
                    if urlsplit(effective_url).hostname == "generativelanguage.googleapis.com"
                    else UsageDialect.OPENAI
                )
            else:
                self.usage_dialect = copy_usage_dialect(
                    declared_usage_dialect,
                    f"{type(self).__name__}.usage_dialect",
                )
        self.timeout_s = _validate_timeout_s(timeout_s)
        if type(stream_idle_timeout_s) not in {int, float}:
            raise TypeError("stream_idle_timeout_s must be a number.")
        stream_idle_timeout_s = require_finite(
            float(stream_idle_timeout_s), "stream_idle_timeout_s"
        )
        if stream_idle_timeout_s <= 0:
            raise ValueError("stream_idle_timeout_s must be greater than zero.")
        self.stream_idle_timeout_s = stream_idle_timeout_s
        # A caller-supplied transport manages its own scheme policy; the default
        # transport inherits allow_http so a local http endpoint actually connects.
        self.transport = (
            transport
            if transport is not None
            else HttpxChatCompletionsTransport(allow_http=allow_http)
        )
        # Protect the headers we set (content-type + the chosen auth header) from
        # being clobbered by extra_headers.
        self.extra_headers = copy_headers(
            extra_headers, protected={"content-type", self.auth_header.lower()}
        )
        if api_version is not None and not require_clean_nonblank(api_version, "api_version"):
            raise ValueError("api_version must be a nonblank string.")
        self.api_version = api_version
        if type(stream_include_usage) is not bool:
            raise TypeError("stream_include_usage must be a bool.")
        self.stream_include_usage = stream_include_usage
        if type(clean_schemas) is not bool:
            raise TypeError("clean_schemas must be a bool.")
        self.clean_schemas = clean_schemas
        self.document_encoding = _validate_document_encoding(document_encoding)

    def request_footprint_options(self, request: ModelRequest) -> dict[str, Any]:
        effective_options = _effective_chat_completions_request_options(
            request.options,
            options_key=self.name,
        )
        projected = privacy_safe_provider_option_projection(effective_options)
        return {self.name: projected} if projected else {}

    def request_fingerprint_options(self, request: ModelRequest) -> dict[str, Any]:
        effective = _effective_chat_completions_request_options(
            request.options,
            options_key=self.name,
        )
        return {self.name: effective} if effective else {}

    async def aclose(self) -> None:
        """Close the transport's shared HTTP client, if it owns one."""
        await aclose_transport(self.transport)

    @detach_provider_stream_traceback
    async def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        cancellation: asyncio.CancelledError | None = None
        overflow_failure: ChatCompletionsContextOverflowError | None = None
        post_completion_failure: ModelProviderError | None = None
        error_event: ModelStreamEvent | None = None
        completion_emitted = False
        try:
            payload = build_chat_completions_payload(
                request,
                stream=True,
                clean_schemas=self.clean_schemas,
                options_key=self.name,
                document_encoding=self.document_encoding,
                include_usage=self.stream_include_usage,
            )
            raw_events = self.transport.stream_chat_completions(
                url=self._endpoint(),
                headers=self._headers(),
                payload=payload,
                timeout_s=self.timeout_s,
                stream_idle_timeout_s=self.stream_idle_timeout_s,
            )
            events = chat_completions_stream_events(raw_events)
            async with aclosing_provider_stream(raw_events), aclosing_provider_stream(events):
                # Chat completion is synthesized only after the raw stream
                # terminates. Exhaust the translator so a deferred transport-
                # cleanup failure reaches runtime after the completion event.
                async for event in events:
                    if event.type == ModelStreamEventType.COMPLETED:
                        completion_emitted = True
                    yield event
        except asyncio.CancelledError as exc:
            cancellation = sanitize_provider_cancellation(
                exc,
                provider_label="Chat Completions",
                credential_values=credential_sanitization_values(
                    self.api_key,
                    extra_headers=self.extra_headers,
                ),
            )
        except ModelContextOverflowError as exc:
            credential_values = credential_sanitization_values(
                self.api_key,
                extra_headers=self.extra_headers,
            )
            if completion_emitted:
                post_completion_failure = credential_safe_post_completion_failure(
                    exc,
                    provider_label="Chat Completions",
                    provider_name="chat_completions",
                    credential_values=credential_values,
                )
            else:
                # Overflow must reach runtime recovery as a typed exception; an
                # error event would flatten it into unrecoverable message text.
                safe = credential_safe_provider_exception(
                    exc,
                    provider_label="Chat Completions",
                    provider_name="chat_completions",
                    credential_values=credential_values,
                )
                overflow_failure = ChatCompletionsContextOverflowError(
                    str(safe),
                    status_code=safe.status_code,
                    error_type=safe.error_type,
                    error_code=safe.error_code,
                    request_id=safe.request_id,
                    response_body=None,
                )
        except Exception as exc:
            credential_values = credential_sanitization_values(
                self.api_key,
                extra_headers=self.extra_headers,
            )
            if completion_emitted:
                post_completion_failure = credential_safe_post_completion_failure(
                    exc,
                    provider_label="Chat Completions",
                    provider_name="chat_completions",
                    credential_values=credential_values,
                )
            else:
                error_event = credential_safe_error_event(
                    exc,
                    provider_label="Chat Completions",
                    provider_name="chat_completions",
                    credential_values=credential_values,
                )
        if cancellation is not None:
            raise cancellation from None
        if overflow_failure is not None:
            raise overflow_failure from None
        if post_completion_failure is not None:
            raise post_completion_failure from None
        if error_event is not None:
            yield error_event

    def _endpoint(self) -> str:
        # OpenAI-SDK convention: base_url already carries the version path, so
        # append only "/chat/completions". `endpoint_url` is a full override.
        if self.endpoint_url is not None and self.api_version is None:
            return self.endpoint_url
        endpoint_override = self.endpoint_url is not None
        url = self.endpoint_url if self.endpoint_url is not None else self.base_url
        parts = urlsplit(url)
        path = parts.path
        if not endpoint_override:
            path = f"{path.rstrip('/')}/chat/completions"
        query = parts.query
        if self.api_version is not None:
            query_parts = (
                []
                if not query
                else [
                    part
                    for part in query.split("&")
                    if unquote_plus(part.partition("=")[0]) != "api-version"
                ]
            )
            query_parts.append(urlencode({"api-version": self.api_version}))
            query = "&".join(query_parts)
        return urlunsplit(parts._replace(path=path, query=query))

    def _headers(self) -> dict[str, str]:
        headers = {
            "content-type": "application/json",
            self.auth_header: f"{self.auth_value_prefix}{self.api_key}",
        }
        headers.update(self.extra_headers)
        return headers


def build_chat_completions_payload(
    request: ModelRequest,
    *,
    stream: bool = False,
    clean_schemas: bool = True,
    options_key: str = "openai",
    document_encoding: str = DEFAULT_DOCUMENT_ENCODING,
    include_usage: bool = True,
) -> dict[str, Any]:
    if type(request) is not ModelRequest:
        raise TypeError("request must be a ModelRequest.")
    if type(clean_schemas) is not bool:
        raise TypeError("clean_schemas must be a bool.")
    if type(include_usage) is not bool:
        raise TypeError("include_usage must be a bool.")
    document_encoding = _validate_document_encoding(document_encoding)

    options = _effective_chat_completions_request_options(
        request.options,
        options_key=options_key,
    )
    resolved_attachments = resolved_file_attachments_from_options(request.options)

    messages: list[dict[str, Any]] = []
    system_text = _system_text(request.messages)
    if system_text:
        messages.append({"role": "system", "content": system_text})
    for message in request.messages:
        messages.extend(
            _chat_completions_messages(
                message,
                resolved_attachments=resolved_attachments,
                document_encoding=document_encoding,
            )
        )
    if not messages:
        raise ValueError("Chat Completions requests require at least one message.")

    payload: dict[str, Any] = {
        "model": request.model,
        "messages": messages,
    }
    tools = [_chat_completions_tool(tool, clean_schemas=clean_schemas) for tool in request.tools]
    if tools:
        payload["tools"] = tools
    if stream:
        payload["stream"] = True
        # Some OpenAI-compatible servers reject stream_options; make it opt-out.
        if include_usage:
            payload["stream_options"] = {"include_usage": True}
    payload.update(options)
    return copy_json_value(payload, "chat_completions_payload")


def _chat_completions_reasoning_options(neutral: Mapping[str, Any]) -> dict[str, Any]:
    """Map the neutral ``options["thinking"]`` payload to Chat Completions request keys.

    The portable knob is ``reasoning_effort`` (low/medium/high), which OpenAI-compatible
    reasoning providers accept. There is no portable way to *disable* reasoning here (the
    ``reasoning_effort="none"`` value is backend-specific — Gemini/DeepSeek accept it,
    OpenAI/Azure reject it), and this generic adapter can't know the backend, so
    ``enabled=False`` is a no-op; pass a raw ``reasoning_effort`` via provider_options to
    target a backend that supports it. There is no portable token budget, so ``max_tokens``
    is not mapped.
    """
    if not neutral.get("enabled", True):
        return {}
    effort = neutral.get("effort")
    if effort is not None:
        return {"reasoning_effort": effort}
    return {}


def _apply_thinking_options(payload: dict[str, Any], neutral: Any) -> None:
    """Merge the mapped reasoning config into the payload (typed config wins)."""
    if not isinstance(neutral, Mapping):
        return
    payload.update(_chat_completions_reasoning_options(neutral))


async def chat_completions_stream_events(
    events: AsyncIterator[Mapping[str, Any]],
) -> AsyncIterator[ModelStreamEvent]:
    tool_calls = _ToolCallAccumulator()
    response_id: str | None = None
    model: str | None = None
    finish_reason: str | None = None
    usage: Any = None
    post_terminal_failure: BaseException | None = None

    iterator = events.__aiter__()
    while True:
        try:
            event = await iterator.__anext__()
        except StopAsyncIteration:
            break
        except asyncio.CancelledError as exc:
            # Preserve real task cancellation without losing authoritative
            # completion evidence already carried by a finish reason. Runtime
            # publishes the completion before restoring the same cancellation.
            if finish_reason is None:
                raise
            post_terminal_failure = exc
            break
        except Exception as exc:
            # A finish reason is authoritative completion evidence. Preserve
            # the accumulated response and usage when the real HTTP iterator
            # fails while closing after that terminal chunk; the provider will
            # surface this failure only after runtime has observed COMPLETED.
            if finish_reason is None:
                raise
            post_terminal_failure = exc
            break
        if not isinstance(event, Mapping):
            raise ChatCompletionsProtocolError(
                "Chat Completions stream event must be a JSON object."
            )
        response_id = response_id or _optional_string(event, "id")
        model = model or _optional_string(event, "model")
        chunk_usage = event.get("usage")
        if chunk_usage is not None:
            if not isinstance(chunk_usage, Mapping):
                raise ChatCompletionsProtocolError("Chat Completions usage must be an object.")
            usage = chunk_usage

        # Some OpenAI-compatible servers report a fault after the stream opens by
        # emitting a data chunk carrying an ``error`` object instead of an HTTP
        # error. Such a chunk has no ``choices``; surfacing it here avoids the
        # misleading "ended before a finish_reason" protocol error it would
        # otherwise trigger downstream.
        error = event.get("error")
        if error is not None:
            failure = _stream_error_chunk_exception(
                error,
                retry_after_s=_trusted_sse_retry_after_s(event),
            )
            # The exported parser is itself a public exception boundary. Do
            # not retain the raw provider envelope in traceback frame locals.
            error = None
            event = {}
            del events
            raise failure from None

        choices = event.get("choices")
        if choices is None:
            continue
        if not isinstance(choices, list):
            raise ChatCompletionsProtocolError("Chat Completions choices must be a list.")
        for choice in choices:
            if not isinstance(choice, Mapping):
                raise ChatCompletionsProtocolError("Chat Completions choice must be an object.")
            delta = choice.get("delta")
            if delta is not None:
                if not isinstance(delta, Mapping):
                    raise ChatCompletionsProtocolError("Chat Completions delta must be an object.")
                reasoning = delta.get("reasoning_content")
                if not (isinstance(reasoning, str) and reasoning):
                    # Fall back to `reasoning` unless reasoning_content is a non-empty
                    # string, so an empty/absent reasoning_content can't shadow it.
                    reasoning = delta.get("reasoning")
                if isinstance(reasoning, str) and reasoning:
                    # Display-only reasoning surfaced by OpenAI-compatible reasoning
                    # providers (DeepSeek/OpenRouter); no round-trip state.
                    yield ModelStreamEvent.thinking(reasoning)
                content = delta.get("content")
                if content is not None:
                    if not isinstance(content, str):
                        raise ChatCompletionsProtocolError(
                            "Chat Completions delta content must be a string."
                        )
                    if content:
                        yield ModelStreamEvent.text_delta(content)
                tool_calls.record(delta.get("tool_calls"))
            choice_finish = choice.get("finish_reason")
            if choice_finish is not None:
                if not isinstance(choice_finish, str):
                    raise ChatCompletionsProtocolError(
                        "Chat Completions finish_reason must be a string."
                    )
                if finish_reason is not None and choice_finish != finish_reason:
                    raise ChatCompletionsProtocolError(
                        "Chat Completions stream emitted conflicting finish_reason values."
                    )
                finish_reason = choice_finish

    # Tool calls are emitted once, after the upstream stream, before Cayu's terminal
    # completed event. Deferring normalization lets trailing usage or repeated
    # identical finish metadata arrive without producing multiple terminal events.
    provider_state = tool_calls.provider_state_items()
    if tool_calls.has_pending():
        for tool_call_event in tool_calls.events():
            yield tool_call_event
        # Gemini's OpenAI-compatible endpoint can return finish_reason="stop"
        # on a chunk that also carries tool_calls. Cayu's runtime needs the
        # provider-neutral completion reason to reflect the actual next step.
        if finish_reason in {"stop", "end_turn"}:
            finish_reason = "tool_calls"

    if finish_reason is None:
        raise ChatCompletionsProtocolError(
            "Chat Completions streaming response ended before a finish_reason."
        )

    completed_payload = {
        "id": response_id,
        "model": model,
        "finish_reason": finish_reason,
        "usage": copy_json_value(usage, "usage"),
    }
    if provider_state:
        completed_payload["provider_state"] = provider_state
    yield ModelStreamEvent.completed(completed_payload)
    if post_terminal_failure is not None:
        failure = post_terminal_failure
        post_terminal_failure = None
        event = {}
        del iterator, events
        raise failure from None


def _stream_error_chunk_exception(
    error: Any,
    *,
    retry_after_s: float | None = None,
) -> ChatCompletionsError:
    """Build the typed exception for a mid-stream ``{"error": ...}`` chunk.

    The error is surfaced as a context-overflow error when its code/type/message
    indicate one (so runtime recovery can see it), else as a plain API error.
    """
    error_mapping = error if isinstance(error, Mapping) else {}
    error_type = optional_error_string(error_mapping.get("type"))
    code = optional_error_string(error_mapping.get("code"))
    message = optional_error_string(error_mapping.get("message"))
    request_id = optional_error_string(error_mapping.get("request_id"))
    safe_message = f"Chat Completions stream reported an error: {OMITTED_PROVIDER_ERROR_BODY}"
    if _is_chat_context_overflow(
        status_code=None,
        error_type=error_type,
        code=code,
        message=message,
    ):
        return ChatCompletionsContextOverflowError(
            safe_message,
            error_type=error_type,
            error_code=code,
            request_id=request_id,
            response_body=None,
        )
    status_code, retryable = _chat_retry_metadata(
        transport_status_code=None,
        error_type=error_type,
        error_code=code,
    )
    return ChatCompletionsAPIError(
        safe_message,
        status_code=status_code,
        error_type=error_type,
        error_code=code,
        request_id=request_id,
        retryable=retryable,
        retry_after_s=retry_after_s,
        response_body=None,
    )


_CHAT_ERROR_TYPE_CLASSIFICATION = {
    "authentication_error": (401, False),
    "context_length_exceeded": (400, False),
    "invalid_request_error": (400, False),
    "not_found_error": (404, False),
    "permission_error": (403, False),
    "rate_limit_error": (429, True),
    "server_error": (500, True),
}
_CHAT_ERROR_CODE_CLASSIFICATION = {
    "context_length_exceeded": (400, False),
    "internal_error": (500, True),
    "rate_limit_exceeded": (429, True),
    "server_error": (500, True),
}


def _chat_retry_metadata(
    *,
    transport_status_code: int | None,
    error_type: str | None,
    error_code: str | None,
) -> tuple[int | None, bool | None]:
    """Classify recognized HTTP/stream identities; conflicts fail closed."""
    classifications = {
        classification
        for classification in (
            _CHAT_ERROR_TYPE_CLASSIFICATION.get(error_type or ""),
            _CHAT_ERROR_CODE_CLASSIFICATION.get(error_code or ""),
        )
        if classification is not None
    }
    if not classifications:
        return transport_status_code, None
    if len(classifications) != 1:
        return transport_status_code, False
    canonical_status, retryable = next(iter(classifications))
    if transport_status_code is not None and transport_status_code != canonical_status:
        return transport_status_code, False
    return canonical_status, retryable


def _tool_call_names_function(tool_call: Mapping[str, Any]) -> bool:
    """Whether a streamed tool-call fragment carries a ``function.name``."""
    function = tool_call.get("function")
    if not isinstance(function, Mapping):
        return False
    return _optional_string(function, "name") is not None


class _PendingToolCall:
    def __init__(self) -> None:
        self.call_id: str | None = None
        self.name: str | None = None
        self.arguments_parts: list[str] = []
        self.extra_content: dict[str, Any] | None = None

    @property
    def arguments(self) -> str:
        return "".join(self.arguments_parts)


class _ToolCallAccumulator:
    """Accumulates streamed tool-call fragments into ordered tool-call events.

    Providers correlate fragments differently. OpenAI puts an ``index`` on each
    ``tool_calls[]`` entry; Gemini's OpenAI-compatible endpoint omits it and
    sends the complete call (with an ``id``) in a single delta. We key by the
    per-call ``index`` when present, else by ``id``, else fall back to the most
    recent slot (a continuation fragment), preserving first-seen order.
    """

    def __init__(self) -> None:
        self._pending: dict[Any, _PendingToolCall] = {}
        self._next_sequence = 0
        self._last_key: Any = None

    def record(self, tool_calls: Any) -> None:
        if tool_calls is None:
            return
        if not isinstance(tool_calls, list):
            raise ChatCompletionsProtocolError("Chat Completions delta tool_calls must be a list.")
        for tool_call in tool_calls:
            if not isinstance(tool_call, Mapping):
                raise ChatCompletionsProtocolError("Chat Completions tool_call must be an object.")
            call_id = _optional_string(tool_call, "id")
            key = self._key_for(tool_call, call_id)
            pending = self._pending.setdefault(key, _PendingToolCall())
            if call_id is not None:
                pending.call_id = call_id
            extra_content = tool_call.get("extra_content")
            if extra_content is not None:
                if not isinstance(extra_content, Mapping):
                    raise ChatCompletionsProtocolError(
                        "Chat Completions tool_call extra_content must be an object."
                    )
                pending.extra_content = copy_json_value(extra_content, "tool_call.extra_content")
            function = tool_call.get("function")
            if function is None:
                continue
            if not isinstance(function, Mapping):
                raise ChatCompletionsProtocolError(
                    "Chat Completions tool_call function must be an object."
                )
            name = _optional_string(function, "name")
            if name is not None:
                pending.name = name
            arguments = function.get("arguments")
            if arguments is not None:
                if not isinstance(arguments, str):
                    raise ChatCompletionsProtocolError(
                        "Chat Completions tool_call arguments must be a string."
                    )
                pending.arguments_parts.append(arguments)

    def _key_for(self, tool_call: Mapping[str, Any], call_id: str | None) -> Any:
        index = tool_call.get("index")
        if index is not None:
            if type(index) is not int or index < 0:
                raise ChatCompletionsProtocolError(
                    "Chat Completions tool_call index must be a non-negative integer."
                )
            key: Any = ("index", index)
        elif call_id is not None:
            key = ("id", call_id)
        elif self._last_key is not None and not _tool_call_names_function(tool_call):
            # A keyless fragment that names no function continues the most recent
            # call (providers stream arguments across chunks that carry only the
            # index-less function.arguments). One that *does* name a function is a
            # distinct call, so fall through to a fresh slot instead of merging it
            # into the previous call's arguments.
            return self._last_key
        else:
            key = ("sequence", self._next_sequence)
            self._next_sequence += 1
        self._last_key = key
        return key

    def has_pending(self) -> bool:
        return bool(self._pending)

    def events(self) -> list[ModelStreamEvent]:
        tool_call_events: list[ModelStreamEvent] = []
        for position, pending in enumerate(self._pending.values()):
            if pending.call_id is None or not pending.call_id.strip():
                raise ChatCompletionsProtocolError(
                    f"Chat Completions tool_call {position} is missing an id."
                )
            if pending.name is None or not pending.name.strip():
                raise ChatCompletionsProtocolError(
                    f"Chat Completions tool_call {position} is missing a name."
                )
            raw_arguments = pending.arguments or "{}"
            try:
                decoded_arguments = json.loads(raw_arguments)
            except ValueError as exc:
                raise ChatCompletionsProtocolError(
                    f"Chat Completions tool_call {position} arguments were not valid JSON."
                ) from exc
            if type(decoded_arguments) is not dict:
                raise ChatCompletionsProtocolError(
                    f"Chat Completions tool_call {position} arguments must decode to an object."
                )
            tool_call_events.append(
                ModelStreamEvent.tool_call(
                    id=pending.call_id,
                    name=pending.name,
                    arguments=copy_json_value(decoded_arguments, "arguments"),
                )
            )
        return tool_call_events

    def provider_state_items(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for pending in self._pending.values():
            if pending.extra_content is None:
                continue
            if pending.call_id is None or not pending.call_id.strip():
                raise ChatCompletionsProtocolError(
                    "Chat Completions tool_call with extra_content is missing an id."
                )
            items.append(
                {
                    "provider": "chat_completions",
                    "state": {
                        "type": "tool_call_extra_content",
                        "tool_call_id": pending.call_id,
                        "extra_content": copy_json_value(
                            pending.extra_content, "tool_call.extra_content"
                        ),
                    },
                }
            )
        return items


def _system_text(messages: list[Message]) -> str:
    system_parts: list[str] = []
    for message in messages:
        if message.role != MessageRole.SYSTEM:
            continue
        for part in message.content:
            if type(part) is TextPart:
                system_parts.append(part.text)
    return "\n\n".join(system_parts)


def _chat_completions_messages(
    message: Message,
    *,
    resolved_attachments: dict[str, dict[str, Any]],
    document_encoding: str,
) -> list[dict[str, Any]]:
    if message.role == MessageRole.SYSTEM:
        return []
    if message.role == MessageRole.USER:
        return [_user_message(message.content, resolved_attachments, document_encoding)]
    if message.role == MessageRole.ASSISTANT:
        return [_assistant_message(message)]
    if message.role == MessageRole.TOOL:
        messages: list[dict[str, Any]] = []
        attachment_parts: list[dict[str, Any]] = []
        for part in message.content:
            if type(part) is not ToolResultPart:
                raise ChatCompletionsProtocolError(
                    "Tool messages can only contain tool_result parts."
                )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": part.tool_call_id,
                    "content": part.content,
                }
            )
            attachment_parts.extend(
                _file_attachment_parts(part, resolved_attachments, document_encoding)
            )
        if attachment_parts:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _TOOL_RESULT_ATTACHMENT_LEAD_IN},
                        *attachment_parts,
                    ],
                }
            )
        return messages
    raise ChatCompletionsProtocolError(f"Unsupported Cayu message role: {message.role!r}.")


def _assistant_message(message: Message) -> dict[str, Any]:
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    tool_call_extra_content = _tool_call_extra_content_by_id(message)
    for part in message.content:
        if type(part) is TextPart:
            text_parts.append(part.text)
        elif type(part) is ToolCallPart:
            tool_call = {
                "id": part.tool_call_id,
                "type": "function",
                "function": {
                    "name": part.tool_name,
                    "arguments": _json_arguments(part.arguments),
                },
            }
            extra_content = tool_call_extra_content.get(part.tool_call_id)
            if extra_content is not None:
                tool_call["extra_content"] = extra_content
            tool_calls.append(tool_call)
        elif type(part) not in {ProviderStatePart, ThinkingPart}:
            raise ChatCompletionsProtocolError(
                "Assistant messages can only contain text, tool_call, provider_state, "
                "and thinking parts."
            )
    assistant: dict[str, Any] = {"role": "assistant"}
    # Chat Completions requires a content key; tool-call-only turns use null.
    assistant["content"] = "\n".join(text_parts) or None
    if tool_calls:
        assistant["tool_calls"] = tool_calls
    return assistant


def _tool_call_extra_content_by_id(message: Message) -> dict[str, dict[str, Any]]:
    extra_content_by_id: dict[str, dict[str, Any]] = {}
    for part in message.content:
        if type(part) is not ProviderStatePart or part.provider != "chat_completions":
            continue
        state = part.state
        if state.get("type") != "tool_call_extra_content":
            continue
        tool_call_id = state.get("tool_call_id")
        if not isinstance(tool_call_id, str) or not tool_call_id.strip():
            raise ChatCompletionsProtocolError(
                "Chat Completions provider_state tool_call_id must be a non-empty string."
            )
        extra_content = state.get("extra_content")
        if not isinstance(extra_content, Mapping):
            raise ChatCompletionsProtocolError(
                "Chat Completions provider_state extra_content must be an object."
            )
        extra_content_by_id[tool_call_id] = copy_json_value(
            extra_content, "provider_state.extra_content"
        )
    return extra_content_by_id


def _user_message(
    content: tuple[
        TextPart | ToolCallPart | ToolResultPart | ProviderStatePart | ThinkingPart | FilePart,
        ...,
    ],
    resolved_attachments: dict[str, dict[str, Any]],
    document_encoding: str,
) -> dict[str, Any]:
    # Text-only turns keep the plain-string content shape for maximum vendor
    # compatibility; file parts require the content-part list form.
    if all(type(part) is TextPart for part in content):
        return {
            "role": "user",
            "content": "\n".join(part.text for part in content if type(part) is TextPart),
        }
    parts: list[dict[str, Any]] = []
    for part in content:
        if type(part) is TextPart:
            parts.append({"type": "text", "text": part.text})
            continue
        if type(part) is FilePart:
            parts.append(
                _file_attachment_part(
                    _resolved_user_attachment(part, resolved_attachments),
                    document_encoding,
                )
            )
            continue
        raise ChatCompletionsProtocolError("User messages can only contain text and file parts.")
    return {"role": "user", "content": parts}


def _resolved_user_attachment(
    part: FilePart,
    resolved_attachments: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    attachment = file_attachment_from_payload(part.attachment)
    if attachment is None:
        raise ChatCompletionsProtocolError("User file parts require a file attachment payload.")
    resolved = resolved_attachments.get(attachment.artifact_id)
    if resolved is None:
        raise ChatCompletionsProtocolError(
            f"Missing resolved file attachment: {attachment.artifact_id}"
        )
    return resolved


def _file_attachment_parts(
    part: ToolResultPart,
    resolved_attachments: dict[str, dict[str, Any]],
    document_encoding: str,
) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for payload in part.artifacts:
        attachment = file_attachment_from_payload(payload)
        if attachment is None:
            continue
        resolved = resolved_attachments.get(attachment.artifact_id)
        if resolved is None:
            raise ChatCompletionsProtocolError(
                f"Missing resolved file attachment: {attachment.artifact_id}"
            )
        parts.append(_file_attachment_part(resolved, document_encoding))
    return parts


def _file_attachment_part(resolved: dict[str, Any], document_encoding: str) -> dict[str, Any]:
    kind = FileAttachmentKind(resolved["kind"])
    data_url = f"data:{resolved['content_type']};base64,{resolved['data_base64']}"
    if kind == FileAttachmentKind.IMAGE:
        return {"type": "image_url", "image_url": {"url": data_url}}
    if kind == FileAttachmentKind.DOCUMENT:
        if document_encoding == "image_url":
            # Google Gemini's compatible endpoint carries PDFs through image_url.
            return {"type": "image_url", "image_url": {"url": data_url}}
        # OpenAI/Azure Chat Completions file-input content part. Vendors that do
        # not implement it reject it with a normal API error, like any other
        # unsupported feature.
        return {
            "type": "file",
            "file": {"filename": resolved["filename"], "file_data": data_url},
        }
    raise ChatCompletionsProtocolError(f"Unsupported file attachment kind: {kind!r}")


def _json_arguments(arguments: Mapping[str, Any]) -> str:
    copied = copy_json_value(arguments, "arguments")
    if type(copied) is not dict:
        raise ChatCompletionsProtocolError("Tool call arguments must be an object.")
    return json.dumps(copied, sort_keys=True, separators=(",", ":"))


def _chat_completions_tool(tool: Mapping[str, Any], *, clean_schemas: bool) -> dict[str, Any]:
    if not isinstance(tool, Mapping):
        raise ValueError("Tool definitions must be objects.")
    name = _require_mapping_string(tool, "name")
    if not _CHAT_COMPLETIONS_TOOL_NAME_RE.fullmatch(name):
        raise ValueError(
            "Chat Completions tool names must contain 1-64 letters, numbers, "
            "underscores, or hyphens."
        )
    description = tool.get("description", "")
    if not isinstance(description, str):
        raise ValueError("Tool description must be a string.")
    input_schema = tool.get("input_schema", {})
    if type(input_schema) is not dict:
        raise ValueError("Tool input_schema must be an object.")
    # Both paths produce a fresh structure; the final whole-payload copy_json_value
    # re-validates JSON-safety, so a separate per-schema copy here would be redundant.
    parameters = (
        _clean_schema(input_schema)
        if clean_schemas
        else copy_json_value(input_schema, "input_schema")
    )
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def _clean_schema(schema: Any, *, in_property_map: bool = False) -> Any:
    """Recursively strip JSON Schema keywords some vendors reject (e.g. Gemini).

    Keys in ``_UNSUPPORTED_SCHEMA_KEYS`` are dropped only where they are schema
    *keywords* (direct keys of a schema object). Inside name->subschema maps
    (``properties``, ``$defs``, ...) the keys are arbitrary names, so a property
    literally named ``additionalProperties`` is preserved and only its subschema
    value is cleaned.
    """
    if isinstance(schema, dict):
        if in_property_map:
            return {name: _clean_schema(value) for name, value in schema.items()}
        return {
            key: _clean_schema(value, in_property_map=key in _SUBSCHEMA_MAP_KEYS)
            for key, value in schema.items()
            if key not in _UNSUPPORTED_SCHEMA_KEYS
        }
    if isinstance(schema, list):
        return [_clean_schema(item) for item in schema]
    return schema


def _chat_completions_options(options: Mapping[str, Any], options_key: str) -> dict[str, Any]:
    raw_options = options.get(options_key, {})
    if raw_options is None:
        return {}
    if type(raw_options) is not dict:
        raise ValueError(f"ModelRequest options.{options_key} must be an object.")
    copied = copy_json_value(raw_options, f"options.{options_key}")
    for key in copied:
        if key in _RESERVED_CHAT_COMPLETIONS_OPTIONS:
            raise ValueError(f"Chat Completions option is reserved: {key}")
    return copied


def _effective_chat_completions_request_options(
    options: Mapping[str, Any],
    *,
    options_key: str,
) -> dict[str, Any]:
    effective = _chat_completions_options(options, options_key)
    # Cayu models one provider response as one assistant step; n>1 would return
    # multiple `choices` that the stream loop cannot represent. Reject it.
    if "n" in effective and (type(effective["n"]) is not int or effective["n"] != 1):
        raise ValueError(
            "Chat Completions n must be 1 (multi-candidate responses are unsupported)."
        )
    _apply_thinking_options(effective, options.get("thinking"))
    return effective


def _require_mapping_string(value: Mapping[str, Any], key: str) -> str:
    raw_value = value.get(key)
    if not isinstance(raw_value, str):
        raise ValueError(f"Tool {key} must be a string.")
    return require_clean_nonblank(raw_value, f"tool.{key}")


def _optional_string(value: Mapping[str, Any], key: str) -> str | None:
    raw_value = value.get(key)
    if raw_value is None:
        return None
    if not isinstance(raw_value, str):
        raise ChatCompletionsProtocolError(f"Chat Completions field {key} must be a string.")
    return raw_value


def _validate_document_encoding(value: object) -> str:
    if type(value) is not str or value not in _VALID_DOCUMENT_ENCODINGS:
        raise ValueError(f"document_encoding must be one of {sorted(_VALID_DOCUMENT_ENCODINGS)}.")
    return value


def _validate_timeout_s(value: float) -> float:
    if type(value) not in {int, float}:
        raise TypeError("timeout_s must be a number.")
    try:
        normalized = float(value)
    except OverflowError:
        raise ValueError("timeout_s must be finite and greater than zero.") from None
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError("timeout_s must be finite and greater than zero.")
    return normalized


def _declared_subclass_usage_dialect(
    provider_type: type[ChatCompletionsProvider],
) -> object:
    for candidate in provider_type.__mro__:
        if candidate is ChatCompletionsProvider:
            break
        if "usage_dialect" in candidate.__dict__:
            return candidate.__dict__["usage_dialect"]
    return _USAGE_DIALECT_UNDECLARED


def _validate_base_url(base_url: str, *, allow_http: bool = False) -> str:
    validated = validate_url(
        base_url,
        "base_url",
        provider_label="Chat Completions",
        allow_http=allow_http,
        allow_http_hint=True,
    )
    parts = urlsplit(validated)
    return urlunsplit(parts._replace(path=parts.path.rstrip("/")))


def _validate_url(url: str, field_name: str, *, allow_http: bool = False) -> str:
    return validate_url(
        url,
        field_name,
        provider_label="Chat Completions",
        allow_http=allow_http,
        allow_http_hint=True,
    )


def _safe_error_response_text(response: httpx.Response) -> str:
    return safe_error_response_text(response, format_error_json=_format_error_json)


def _format_error_json(decoded: Any) -> str | None:
    if not isinstance(decoded, Mapping):
        return None
    return safe_error_json(decoded)


def _chat_api_error_from_response(
    response: httpx.Response,
    message: str,
    retry_after_s: float | None,
) -> ChatCompletionsAPIError:
    """Build a buffered error with the same typed identity rules as streaming."""

    decoded = response_json_object(response)
    error: Mapping[str, Any] = {}
    if decoded is not None:
        raw_error = decoded.get("error")
        error = raw_error if isinstance(raw_error, Mapping) else decoded
    error_type = optional_error_string(error.get("type"))
    error_code = optional_error_string(error.get("code"))
    status_code, retryable = _chat_retry_metadata(
        transport_status_code=response.status_code,
        error_type=error_type,
        error_code=error_code,
    )
    request_id = optional_error_string(error.get("request_id"))
    if request_id is None and decoded is not None:
        request_id = optional_error_string(decoded.get("request_id"))
    return ChatCompletionsAPIError(
        message,
        status_code=status_code,
        error_type=error_type,
        error_code=error_code,
        request_id=request_id,
        retryable=retryable,
        retry_after_s=retry_after_s,
        response_body=_safe_error_response_text(response),
    )


def _raise_chat_context_overflow_if_applicable(response: httpx.Response) -> None:
    decoded = response_json_object(response)
    if decoded is None:
        return
    error = decoded.get("error")
    if not isinstance(error, Mapping):
        error = decoded
    error_type = optional_error_string(error.get("type")) or optional_error_string(
        error.get("status")
    )
    code = optional_error_string(error.get("code"))
    message = optional_error_string(error.get("message"))
    if not _is_chat_context_overflow(
        status_code=response.status_code,
        error_type=error_type,
        code=code,
        message=message,
    ):
        return
    raise ChatCompletionsContextOverflowError(
        "Chat Completions model context overflow",
        status_code=response.status_code,
        error_type=error_type,
        error_code=code,
        response_body=_safe_error_response_text(response),
    )


def _is_chat_context_overflow(
    *,
    status_code: int | None,
    error_type: str | None,
    code: str | None,
    message: str | None,
) -> bool:
    structured_statuses = {
        classification[0]
        for classification in (
            _CHAT_ERROR_TYPE_CLASSIFICATION.get(error_type or ""),
            _CHAT_ERROR_CODE_CLASSIFICATION.get(code or ""),
        )
        if classification is not None
    }
    # Structured type/code identities are authoritative. Only a coherent 400
    # identity can support context recovery; message text must not override a
    # recognized transient, authentication, permission, or rate-limit failure.
    if len(structured_statuses) > 1 or (structured_statuses and structured_statuses != {400}):
        return False
    if status_code is not None:
        # Some compatible providers (notably Gemini) report context overflow as
        # an otherwise-unclassified HTTP 500/504. Preserve that compatibility,
        # but reject a transport status that conflicts with a structured 400.
        if status_code not in {400, 500, 504}:
            return False
        if structured_statuses and status_code not in structured_statuses:
            return False
    if code == "context_length_exceeded":
        return True
    if error_type == "context_length_exceeded":
        return True
    if message is None:
        return False
    normalized = message.lower()
    if any(
        phrase in normalized
        for phrase in (
            "context_length_exceeded",
            "context length exceeded",
            "maximum context length",
            "input context is too long",
            "context is too long",
            "context too large",
            "prompt too large",
            "exceeds the context window",
        )
    ):
        return True
    return status_code in {400, 500, 504} and "context" in normalized and "too large" in normalized
