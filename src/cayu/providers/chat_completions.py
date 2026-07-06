from __future__ import annotations

import json
import os
import re
from collections.abc import AsyncIterator, Mapping
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import urlencode

from cayu._validation import copy_json_value, require_clean_nonblank
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
from cayu.providers._http import (
    SharedAsyncClient,
    aclose_transport,
    copy_headers,
    exception_message,
    json_error_text,
    optional_error_string,
    response_json_object,
    safe_error_json,
    safe_error_response_text,
    stream_sse_json_events,
    truncate_error_text,
    validate_base_url,
    validate_url,
)
from cayu.providers.base import (
    ModelContextOverflowError,
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelStreamEvent,
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
        )
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
    ) -> None:
        self.name = require_clean_nonblank(name, "name")
        self.api_key_env = require_clean_nonblank(api_key_env, "api_key_env")
        if api_key is not None and type(api_key) is not str:
            raise TypeError("api_key must be a string.")
        resolved_api_key = api_key if api_key is not None else os.environ.get(self.api_key_env, "")
        if not resolved_api_key.strip():
            raise ValueError(
                f"ChatCompletionsProvider requires an API key: set the {self.api_key_env} "
                "environment variable or pass api_key=... to ChatCompletionsProvider(...)."
            )
        self.api_key = resolved_api_key
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

    async def aclose(self) -> None:
        """Close the transport's shared HTTP client, if it owns one."""
        await aclose_transport(self.transport)

    async def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
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
            async for event in chat_completions_stream_events(raw_events):
                yield event
        except ModelContextOverflowError:
            # Overflow must reach runtime recovery as a typed exception; an
            # error event would flatten it into unrecoverable message text.
            raise
        except Exception as exc:
            yield ModelStreamEvent.error(
                exception_message(exc, provider_label="Chat Completions"),
                cause=exc,
            )

    def _endpoint(self) -> str:
        # OpenAI-SDK convention: base_url already carries the version path, so
        # append only "/chat/completions". `endpoint_url` is a full override.
        url = (
            self.endpoint_url
            if self.endpoint_url is not None
            else f"{self.base_url}/chat/completions"
        )
        if self.api_version is not None:
            # endpoint_url may already carry a query string, so pick the separator.
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}{urlencode({'api-version': self.api_version})}"
        return url

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
    document_encoding = _validate_document_encoding(document_encoding)

    options = _chat_completions_options(request.options, options_key)
    # Cayu models one provider response as one assistant step; n>1 would return
    # multiple `choices` that the stream loop cannot represent. Reject it.
    if "n" in options and options["n"] != 1:
        raise ValueError(
            "Chat Completions n must be 1 (multi-candidate responses are unsupported)."
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
    _apply_thinking_options(payload, request.options.get("thinking"))
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

    async for event in events:
        if not isinstance(event, Mapping):
            raise ChatCompletionsProtocolError(
                "Chat Completions stream event must be a JSON object."
            )
        response_id = response_id or _optional_string(event, "id")
        model = model or _optional_string(event, "model")
        chunk_usage = event.get("usage")
        if chunk_usage is not None:
            usage = chunk_usage

        # Some OpenAI-compatible servers report a fault after the stream opens by
        # emitting a data chunk carrying an ``error`` object instead of an HTTP
        # error. Such a chunk has no ``choices``; surfacing it here avoids the
        # misleading "ended before a finish_reason" protocol error it would
        # otherwise trigger downstream.
        error = event.get("error")
        if error is not None:
            raise _stream_error_chunk_exception(error)

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
                finish_reason = choice_finish

    # Tool calls are emitted once, after the stream, before the terminal completed
    # event. The finish_reason chunk is terminal for these providers, so nothing
    # follows it that would need an earlier flush.
    if tool_calls.has_pending():
        for tool_call_event in tool_calls.events():
            yield tool_call_event

    if finish_reason is None:
        raise ChatCompletionsProtocolError(
            "Chat Completions streaming response ended before a finish_reason."
        )

    yield ModelStreamEvent.completed(
        {
            "id": response_id,
            "model": model,
            "finish_reason": finish_reason,
            "usage": copy_json_value(usage, "usage"),
        }
    )


def _stream_error_chunk_exception(error: Any) -> ChatCompletionsError:
    """Build the typed exception for a mid-stream ``{"error": ...}`` chunk.

    The error is surfaced as a context-overflow error when its code/type/message
    indicate one (so runtime recovery can see it), else as a plain API error.
    """
    error_mapping = error if isinstance(error, Mapping) else {}
    error_type = optional_error_string(error_mapping.get("type"))
    code = optional_error_string(error_mapping.get("code"))
    message = optional_error_string(error_mapping.get("message"))
    detail = message or json_error_text(error)
    full_message = f"Chat Completions stream reported an error: {truncate_error_text(detail)}"
    if _is_chat_context_overflow(status_code=0, error_type=error_type, code=code, message=message):
        return ChatCompletionsContextOverflowError(
            full_message, error_type=error_type, error_code=code
        )
    return ChatCompletionsAPIError(full_message, error_type=error_type, error_code=code)


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
    for part in message.content:
        if type(part) is TextPart:
            text_parts.append(part.text)
        elif type(part) is ToolCallPart:
            tool_calls.append(
                {
                    "id": part.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": part.tool_name,
                        "arguments": _json_arguments(part.arguments),
                    },
                }
            )
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


def _validate_document_encoding(value: str) -> str:
    if value not in _VALID_DOCUMENT_ENCODINGS:
        raise ValueError(f"document_encoding must be one of {sorted(_VALID_DOCUMENT_ENCODINGS)}.")
    return value


def _validate_base_url(base_url: str, *, allow_http: bool = False) -> str:
    return validate_base_url(
        base_url,
        provider_label="Chat Completions",
        allow_http=allow_http,
        allow_http_hint=True,
    )


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
    status_code: int,
    error_type: str | None,
    code: str | None,
    message: str | None,
) -> bool:
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
