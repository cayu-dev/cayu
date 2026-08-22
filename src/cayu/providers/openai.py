from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Mapping
from typing import TYPE_CHECKING, Any, Protocol, cast
from urllib.parse import quote, urlencode

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    copy_json_value,
    escape_json_pointer_segment,
    require_clean_nonblank,
    require_finite,
    unescape_json_pointer_segment,
)
from cayu.artifacts import (
    FileAttachmentKind,
    file_attachment_from_payload,
    resolved_file_attachments_from_options,
)
from cayu.core.messages import (
    CitationPart,
    FilePart,
    HostedToolCallPart,
    Message,
    MessageRole,
    ProviderStatePart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolResultPart,
    WebSearchAction,
    WebSearchSource,
)
from cayu.embeddings import (
    TextEmbedding,
    TextEmbeddingProvider,
    TextEmbeddingRequest,
    TextEmbeddingResult,
    TextEmbeddingUsage,
    copy_text_embedding_request,
)
from cayu.providers._api_keys import resolve_api_key
from cayu.providers._credential_boundary import (
    aclosing_provider_stream,
    detach_provider_call_traceback,
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
    post_json,
    request_json,
    response_json_object,
    safe_error_json,
    safe_error_response_text,
    sanitize_provider_cancellation,
    stream_sse_json_events,
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
    NativeStructuredOutputSchemaInvalid,
    UsageDialect,
    _preflight_provider_portable_messages,
    privacy_safe_provider_option_projection,
)
from cayu.providers.hosted import HostedToolCapabilityError, OpenAIWebSearch
from cayu.providers.operations import (
    ProviderOperationAdapter,
    ProviderOperationCancellationSupport,
    ProviderOperationConnection,
    ProviderOperationMalformedError,
    ProviderOperationMode,
    ProviderOperationRecoveryMetadata,
    ProviderOperationSnapshot,
    ProviderOperationStartRequest,
    ProviderOperationState,
    ProviderOperationStatus,
    copy_provider_operation_state,
)

if TYPE_CHECKING:
    import httpx

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com"
DEFAULT_OPENAI_TIMEOUT_SECONDS = 60.0
DEFAULT_OPENAI_STREAM_IDLE_TIMEOUT_SECONDS = 120.0
OPENAI_CONTEXT_PRESSURE_TOOL_SCHEMA_CHARS_PER_TOKEN = 6

_RESERVED_OPENAI_OPTIONS = {
    "background",
    "model",
    "input",
    "instructions",
    "previous_response_id",
    "store",
    "tools",
    "include",
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
_OPENAI_HOSTED_WEB_SEARCH_MODELS = frozenset(
    {
        "chat-latest",
        "gpt-5.6",
        "gpt-5.6-luna",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
    }
)
_PROTECTED_HEADER_NAMES = {
    "authorization",
    "content-type",
}
_OPENAI_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
# RFC 6901 array index: ASCII digits, no leading zeros. str.isdigit() is too
# loose here — it accepts '01' and non-decimal digits like '²' (which int()
# then rejects with a raw ValueError).
_OPENAI_POINTER_INDEX_RE = re.compile(r"0|[1-9][0-9]*")
# Bounds this module's own recursion in the schema preflight walk, far above
# any schema OpenAI native mode accepts; NOT a model of OpenAI's (drifting)
# nesting limit.
_OPENAI_SCHEMA_PREFLIGHT_MAX_DEPTH = 128
# OpenAI rejects every $ref sibling key except these containers.
_OPENAI_REF_SIBLING_ALLOWLIST = frozenset({"$ref", "$defs", "definitions"})
_VALID_REASONING_STATES = {"inline", "server"}
_OPENAI_BACKGROUND_STREAM_PROTOCOL = "openai-responses-background-v1"


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


class OpenAIBackgroundTransport(OpenAITransport, Protocol):
    """Additional transport operations required by explicit background mode."""

    async def retrieve_response(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        """GET one existing Responses API object."""

    def reconnect_response_events(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        starting_after: int,
        timeout_s: float,
        stream_idle_timeout_s: float,
    ) -> AsyncIterator[Mapping[str, Any]]:
        """Resume one stored background response after an accepted sequence."""

    async def cancel_response(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        """Cancel one existing Responses API object."""


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
        async with aclosing_provider_stream(events):
            async for event in events:
                yield event

    async def retrieve_response(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        return await self._operation_request(
            method="GET",
            url=url,
            headers=headers,
            timeout_s=timeout_s,
        )

    async def reconnect_response_events(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        starting_after: int,
        timeout_s: float,
        stream_idle_timeout_s: float,
    ) -> AsyncIterator[Mapping[str, Any]]:
        reconnect_url = f"{url}?{urlencode({'stream': 'true', 'starting_after': starting_after})}"
        events = stream_sse_json_events(
            client=self._client.get(),
            method="GET",
            url=reconnect_url,
            headers=headers,
            payload={},
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
        async with aclosing_provider_stream(events):
            async for event in events:
                yield event

    async def cancel_response(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        return await self._operation_request(
            method="POST",
            url=url,
            headers=headers,
            timeout_s=timeout_s,
        )

    async def _operation_request(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        return await request_json(
            client=self._client.get(),
            method=method,
            url=_validate_url(url, "url"),
            headers=headers,
            payload=None,
            timeout_s=timeout_s,
            request_label="OpenAI API",
            response_label="OpenAI",
            api_error=OpenAIAPIError,
            protocol_error=OpenAIProtocolError,
            error_response_text=_safe_error_response_text,
            raise_context_overflow=_raise_openai_context_overflow_if_applicable,
            api_error_from_response=_openai_api_error_from_response,
        )


class _OpenAIBackgroundOperationAdapter(ProviderOperationAdapter):
    """Reconnectable Responses background execution for one OpenAI provider."""

    def __init__(self, provider: OpenAIProvider) -> None:
        self._provider = provider
        self._transport = cast("OpenAIBackgroundTransport", provider.transport)

    @property
    def cancellation_support(self) -> ProviderOperationCancellationSupport:
        return ProviderOperationCancellationSupport.SUPPORTED

    async def start(
        self,
        request: ProviderOperationStartRequest,
    ) -> ProviderOperationConnection:
        if type(request) is not ProviderOperationStartRequest:
            raise TypeError("request must be a ProviderOperationStartRequest.")
        payload = build_openai_payload(
            request.request,
            stream=True,
            reasoning_state=self._provider.reasoning_state,
        )
        payload["background"] = True
        payload["store"] = True
        raw_events: AsyncIterator[Mapping[str, Any]] | None = None
        try:
            raw_events = self._transport.stream_response_events(
                url=f"{self._provider.base_url}/v1/responses",
                headers=self._provider._headers(),
                payload=payload,
                timeout_s=self._provider.timeout_s,
                stream_idle_timeout_s=self._provider.stream_idle_timeout_s,
            )
            created = await self._next_raw_event(
                raw_events,
                empty_message="OpenAI background start ended before response.created.",
            )
            state, status = _openai_background_created_state(created)
        except asyncio.CancelledError as exc:
            if raw_events is not None:
                await _close_openai_operation_stream(raw_events)
            raise self._safe_cancellation(exc) from None
        except Exception as exc:
            if raw_events is not None:
                await _close_openai_operation_stream(raw_events)
            raise self._safe_failure(exc) from None
        except BaseException:
            if raw_events is not None:
                await _close_openai_operation_stream(raw_events)
            raise
        assert raw_events is not None
        return ProviderOperationConnection(
            state=state,
            status=status,
            events=self._events(raw_events, state=state),
        )

    async def retrieve(self, state: ProviderOperationState) -> ProviderOperationSnapshot:
        state = _require_openai_background_state(state)
        url = _openai_operation_url(self._provider.base_url, state.operation_id)
        try:
            response = await self._transport.retrieve_response(
                url=url,
                headers=self._provider._headers(),
                timeout_s=self._provider.timeout_s,
            )
        except OpenAIAPIError as exc:
            if exc.status_code == 404:
                return ProviderOperationSnapshot(
                    state=state,
                    status=ProviderOperationStatus.EXPIRED,
                )
            raise self._safe_failure(exc) from None
        except asyncio.CancelledError as exc:
            raise self._safe_cancellation(exc) from None
        except Exception as exc:
            raise self._safe_failure(exc) from None
        return _openai_background_snapshot(
            state,
            response,
            reasoning_state=self._provider.reasoning_state,
        )

    async def reconnect(self, state: ProviderOperationState) -> ProviderOperationConnection:
        state = _require_openai_background_state(state)
        sequence_number = _openai_recovery_sequence_number(state.recovery_metadata)
        url = _openai_operation_url(self._provider.base_url, state.operation_id)
        raw_events: AsyncIterator[Mapping[str, Any]] | None = None
        try:
            raw_events = self._transport.reconnect_response_events(
                url=url,
                headers=self._provider._headers(),
                starting_after=sequence_number,
                timeout_s=self._provider.timeout_s,
                stream_idle_timeout_s=self._provider.stream_idle_timeout_s,
            )
            first = await self._next_raw_event(raw_events, empty_message=None)
        except StopAsyncIteration:
            return ProviderOperationConnection(
                state=state,
                status=ProviderOperationStatus.IN_PROGRESS,
                events=_empty_model_stream(),
            )
        except OpenAIAPIError as exc:
            if raw_events is not None:
                await _close_openai_operation_stream(raw_events)
            return ProviderOperationConnection(
                state=state,
                status=(
                    ProviderOperationStatus.EXPIRED
                    if exc.status_code == 404
                    else ProviderOperationStatus.UNAVAILABLE
                ),
                events=_empty_model_stream(),
            )
        except asyncio.CancelledError as exc:
            if raw_events is not None:
                await _close_openai_operation_stream(raw_events)
            raise self._safe_cancellation(exc) from None
        except Exception as exc:
            if raw_events is not None:
                await _close_openai_operation_stream(raw_events)
            raise self._safe_failure(exc) from None
        assert raw_events is not None
        status = _openai_stream_operation_status(first)
        return ProviderOperationConnection(
            state=state,
            status=status,
            events=self._events(raw_events, state=state, first=first),
        )

    async def cancel(self, state: ProviderOperationState) -> ProviderOperationSnapshot:
        state = _require_openai_background_state(state)
        url = f"{_openai_operation_url(self._provider.base_url, state.operation_id)}/cancel"
        try:
            response = await self._transport.cancel_response(
                url=url,
                headers=self._provider._headers(),
                timeout_s=self._provider.timeout_s,
            )
        except OpenAIAPIError as exc:
            if exc.status_code == 404:
                return ProviderOperationSnapshot(
                    state=state,
                    status=ProviderOperationStatus.EXPIRED,
                )
            raise self._safe_failure(exc) from None
        except asyncio.CancelledError as exc:
            raise self._safe_cancellation(exc) from None
        except Exception as exc:
            raise self._safe_failure(exc) from None
        return _openai_background_snapshot(
            state,
            response,
            reasoning_state=self._provider.reasoning_state,
        )

    async def _next_raw_event(
        self,
        raw_events: AsyncIterator[Mapping[str, Any]],
        *,
        empty_message: str | None,
    ) -> Mapping[str, Any]:
        try:
            return await anext(raw_events)
        except StopAsyncIteration:
            if empty_message is None:
                raise
            raise OpenAIProtocolError(empty_message) from None
        except asyncio.CancelledError as exc:
            raise self._safe_cancellation(exc) from None
        except Exception as exc:
            raise self._safe_failure(exc) from None

    async def _events(
        self,
        raw_events: AsyncIterator[Mapping[str, Any]],
        *,
        state: ProviderOperationState,
        first: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        try:
            events = _openai_background_stream_events(
                raw_events,
                state=state,
                first=first,
                reasoning_state=self._provider.reasoning_state,
            )
            async with aclosing_provider_stream(raw_events), aclosing_provider_stream(events):
                async for event in events:
                    yield event
        except asyncio.CancelledError as exc:
            raise self._safe_cancellation(exc) from None
        except Exception as exc:
            raise self._safe_failure(exc) from None

    def _safe_cancellation(self, exc: asyncio.CancelledError) -> asyncio.CancelledError:
        return sanitize_provider_cancellation(
            exc,
            provider_label="OpenAI",
            credential_values=credential_sanitization_values(
                self._provider.api_key,
                extra_headers=self._provider.extra_headers,
            ),
        )

    def _safe_failure(self, exc: Exception) -> Exception:
        safe = credential_safe_provider_exception(
            exc,
            provider_label="OpenAI",
            provider_name="openai",
            credential_values=credential_sanitization_values(
                self._provider.api_key,
                extra_headers=self._provider.extra_headers,
            ),
        )
        if isinstance(exc, (OpenAIProtocolError, ProviderOperationMalformedError)):
            return ProviderOperationMalformedError(str(safe))
        return OpenAIAPIError(
            str(safe),
            status_code=safe.status_code,
            error_type=safe.error_type,
            error_code=safe.error_code,
            request_id=safe.request_id,
            retryable=safe.retryable,
            retry_after_s=safe.retry_after_s,
            response_body=None,
        )


class OpenAIProvider(ModelProvider, TextEmbeddingProvider):
    """OpenAI Responses API adapter for Cayu's provider-neutral runtime."""

    name = "openai"
    usage_dialect = UsageDialect.OPENAI
    supports_native_structured_output = True

    @property
    def provider_operation_mode(self) -> ProviderOperationMode:
        return (
            ProviderOperationMode.BACKGROUND
            if self.background
            else ProviderOperationMode.SYNCHRONOUS
        )

    @property
    def provider_operations(self) -> ProviderOperationAdapter | None:
        return self._background_operations

    def preflight_portable_messages(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> None:
        _preflight_provider_portable_messages(
            model=model,
            messages=messages,
            tools=tools,
            supports_system_messages=True,
            supports_tool_history=True,
            supports_tool_definitions=True,
            supports_file_attachments=True,
            tool_name_validator=_validate_openai_tool_name,
            tool_definition_validator=_openai_tool,
        )

    def preflight_hosted_tools(
        self,
        *,
        model: str,
        hosted_tools: tuple[OpenAIWebSearch, ...],
        options: dict[str, Any],
    ) -> None:
        _preflight_openai_hosted_tools(
            model=model,
            hosted_tools=hosted_tools,
            options=options,
            endpoint_supported=self.hosted_web_search_supported,
        )

    @property
    def context_pressure_profile(self) -> ModelContextPressureProfile:
        return ModelContextPressureProfile(
            tool_schema_chars_per_token=OPENAI_CONTEXT_PRESSURE_TOOL_SCHEMA_CHARS_PER_TOKEN,
        )

    def preflight_native_structured_output_schema(self, json_schema: dict[str, Any]) -> None:
        preflight_openai_native_structured_output_schema(json_schema)

    def request_footprint_options(self, request: ModelRequest) -> dict[str, Any]:
        projected = privacy_safe_provider_option_projection(
            _effective_openai_request_options(request.options)
        )
        return {"openai": projected} if projected else {}

    def request_fingerprint_options(self, request: ModelRequest) -> dict[str, Any]:
        effective = _effective_openai_request_options(request.options)
        return {"openai": effective} if effective else {}

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
        hosted_web_search_supported: bool | None = None,
        background: bool = False,
    ) -> None:
        self.name = require_clean_nonblank(name, "name")
        self.api_key = resolve_api_key(
            api_key=api_key,
            env_var="OPENAI_API_KEY",
            provider_name="OpenAIProvider",
            missing_hint=(
                "set the OPENAI_API_KEY environment variable or pass api_key=... "
                "to OpenAIProvider(...)."
            ),
        )
        self.base_url = _validate_base_url(base_url)
        if (
            hosted_web_search_supported is not None
            and type(hosted_web_search_supported) is not bool
        ):
            raise TypeError("hosted_web_search_supported must be a boolean or None.")
        self.hosted_web_search_supported = (
            self.base_url == _validate_base_url(DEFAULT_OPENAI_BASE_URL)
            if hosted_web_search_supported is None
            else hosted_web_search_supported
        )
        if type(background) is not bool:
            raise TypeError("background must be a bool.")
        if background and self.base_url != _validate_base_url(DEFAULT_OPENAI_BASE_URL):
            raise ValueError(
                "OpenAI background operations require the official OpenAI API base URL."
            )
        self.background = background
        if type(timeout_s) not in {int, float}:
            raise TypeError("timeout_s must be a number.")
        timeout_s = require_finite(float(timeout_s), "timeout_s")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be greater than zero.")
        self.timeout_s = timeout_s
        if type(stream_idle_timeout_s) not in {int, float}:
            raise TypeError("stream_idle_timeout_s must be a number.")
        stream_idle_timeout_s = require_finite(
            float(stream_idle_timeout_s), "stream_idle_timeout_s"
        )
        if stream_idle_timeout_s <= 0:
            raise ValueError("stream_idle_timeout_s must be greater than zero.")
        self.stream_idle_timeout_s = stream_idle_timeout_s
        self.transport = transport if transport is not None else HttpxOpenAITransport()
        if self.background:
            missing_operations = [
                operation
                for operation in (
                    "retrieve_response",
                    "reconnect_response_events",
                    "cancel_response",
                )
                if not callable(getattr(self.transport, operation, None))
            ]
            if missing_operations:
                raise TypeError(
                    "OpenAI background transport is missing required operations: "
                    + ", ".join(missing_operations)
                    + "."
                )
        self.extra_headers = _copy_headers(extra_headers)
        self.reasoning_state = _validate_reasoning_state(reasoning_state)
        self._background_operations = (
            _OpenAIBackgroundOperationAdapter(self) if self.background else None
        )

    @detach_provider_stream_traceback
    async def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        cancellation: asyncio.CancelledError | None = None
        overflow_failure: OpenAIContextOverflowError | None = None
        post_completion_failure: ModelProviderError | None = None
        error_event: ModelStreamEvent | None = None
        completion_emitted = False
        try:
            payload = build_openai_payload(
                request, stream=True, reasoning_state=self.reasoning_state
            )
            yielded_any = False
            try:
                events = self._consume(payload)
                async with aclosing_provider_stream(events):
                    async for event in events:
                        yielded_any = True
                        completion_emitted = event.type == ModelStreamEventType.COMPLETED
                        yield event
                        if completion_emitted:
                            break
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
            events = self._consume(recovery_payload)
            async with aclosing_provider_stream(events):
                async for event in events:
                    completion_emitted = event.type == ModelStreamEventType.COMPLETED
                    yield event
                    if completion_emitted:
                        break
        except asyncio.CancelledError as exc:
            cancellation = sanitize_provider_cancellation(
                exc,
                provider_label="OpenAI",
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
                    provider_label="OpenAI",
                    provider_name="openai",
                    credential_values=credential_values,
                )
            else:
                # Overflow must reach runtime recovery as a typed exception; an
                # error event would flatten it into unrecoverable message text.
                safe = credential_safe_provider_exception(
                    exc,
                    provider_label="OpenAI",
                    provider_name="openai",
                    credential_values=credential_values,
                )
                overflow_failure = OpenAIContextOverflowError(
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
                    provider_label="OpenAI",
                    provider_name="openai",
                    credential_values=credential_values,
                )
            else:
                error_event = credential_safe_error_event(
                    exc,
                    provider_label="OpenAI",
                    provider_name="openai",
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

    @detach_provider_call_traceback
    async def count_input_tokens(
        self,
        request: ModelRequest,
    ) -> InputTokenCountResult | None:
        payload = build_openai_token_count_payload(
            request,
            reasoning_state=self.reasoning_state,
        )
        response = await self._safe_create_response(
            url=f"{self.base_url}/v1/responses/input_tokens",
            payload=payload,
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

    @detach_provider_call_traceback
    async def embed_texts(self, request: TextEmbeddingRequest) -> TextEmbeddingResult:
        embedding_request = copy_text_embedding_request(request)
        payload = build_openai_embedding_payload(embedding_request)
        response = await self._safe_create_response(
            url=f"{self.base_url}/v1/embeddings",
            payload=payload,
        )
        return openai_embedding_result(
            response,
            requested_count=len(embedding_request.texts),
        )

    async def _safe_create_response(
        self,
        *,
        url: str,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        cancellation: asyncio.CancelledError | None = None
        failure: ModelProviderError | None = None
        try:
            return await self.transport.create_response(
                url=url,
                headers=self._headers(),
                payload=payload,
                timeout_s=self.timeout_s,
            )
        except asyncio.CancelledError as exc:
            cancellation = sanitize_provider_cancellation(
                exc,
                provider_label="OpenAI",
                credential_values=credential_sanitization_values(
                    self.api_key,
                    extra_headers=self.extra_headers,
                ),
            )
        except Exception as exc:
            safe = credential_safe_provider_exception(
                exc,
                provider_label="OpenAI",
                provider_name="openai",
                credential_values=credential_sanitization_values(
                    self.api_key,
                    extra_headers=self.extra_headers,
                ),
            )
            if isinstance(safe, ModelContextOverflowError):
                failure = OpenAIContextOverflowError(
                    str(safe),
                    status_code=safe.status_code,
                    error_type=safe.error_type,
                    error_code=safe.error_code,
                    request_id=safe.request_id,
                    response_body=None,
                )
            else:
                failure = OpenAIAPIError(
                    str(safe),
                    status_code=safe.status_code,
                    error_type=safe.error_type,
                    error_code=safe.error_code,
                    request_id=safe.request_id,
                    retryable=safe.retryable,
                    retry_after_s=safe.retry_after_s,
                    response_body=None,
                )
        if cancellation is not None:
            raise cancellation
        if failure is None:  # pragma: no cover - the try branch returns
            raise AssertionError("OpenAI response failure was not captured")
        raise failure

    async def _consume(self, payload: dict[str, Any]) -> AsyncIterator[ModelStreamEvent]:
        raw_events = self.transport.stream_response_events(
            url=f"{self.base_url}/v1/responses",
            headers=self._headers(),
            payload=payload,
            timeout_s=self.timeout_s,
            stream_idle_timeout_s=self.stream_idle_timeout_s,
        )
        events = openai_stream_events(raw_events, reasoning_state=self.reasoning_state)
        async with aclosing_provider_stream(raw_events), aclosing_provider_stream(events):
            async for event in events:
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

    options = _effective_openai_request_options(request.options)
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
    tools.extend(_openai_hosted_tool(tool) for tool in request.hosted_tools)
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
    if any(tool.include_sources for tool in request.hosted_tools):
        payload["include"] = [
            *payload.get("include", []),
            "web_search_call.action.sources",
        ]
    if stream:
        payload["stream"] = True
    payload.update(options)
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


def openai_response_events(
    response: Mapping[str, Any],
    *,
    reasoning_state: str = "inline",
) -> list[ModelStreamEvent]:
    reasoning_state = _validate_reasoning_state(reasoning_state)
    if not isinstance(response, Mapping):
        raise OpenAIProtocolError("OpenAI response must be a JSON object.")

    error = response.get("error")
    if error is not None:
        failure = _openai_error_value_exception(
            error,
            safe_message=f"OpenAI response error: {OMITTED_PROVIDER_ERROR_BODY}",
            request_id=optional_error_string(response.get("request_id")),
        )
        # This exported parser is a public exception boundary. Do not retain
        # the raw provider envelope in traceback frame locals.
        error = None
        response = {}
        raise failure from None

    output = response.get("output")
    if not isinstance(output, list):
        raise OpenAIProtocolError("OpenAI response output must be a list.")

    events: list[ModelStreamEvent] = []
    provider_state_items: list[dict[str, Any]] = []
    completion_output_items: list[Mapping[str, Any]] = []
    hosted_call_indexes: dict[str, int] = {}
    assistant_text_offset = 0
    response_status = _optional_string(response, "status")
    for index, item in enumerate(output):
        if not isinstance(item, Mapping):
            raise OpenAIProtocolError(f"OpenAI output item {index} must be an object.")
        item = cast("Mapping[str, Any]", item)
        item_type = item.get("type")
        if item_type == "message":
            message_events, message_text_length = _message_output_events(
                item,
                index,
                text_offset=assistant_text_offset,
            )
            events.extend(message_events)
            assistant_text_offset += message_text_length
            completion_output_items.append(item)
            provider_state_items.append(
                {"provider": "openai", "state": copy_json_value(item, "output_item")}
            )
        elif item_type == "function_call":
            events.append(_function_call_event(item, index))
            completion_output_items.append(item)
            provider_state_items.append(
                {"provider": "openai", "state": copy_json_value(item, "output_item")}
            )
        elif item_type == "reasoning":
            # Surface the readable reasoning summary as display-only thinking, but keep
            # capturing the full reasoning item (incl. encrypted_content) as provider
            # state so the multi-turn round-trip is unaffected.
            events.extend(_reasoning_thinking_events(item))
            completion_output_items.append(item)
            provider_state_items.append(
                {"provider": "openai", "state": copy_json_value(item, "output_item")}
            )
            continue
        elif item_type == "web_search_call":
            normalized = _normalized_web_search_call(item, item_index=index)
            _claim_web_search_call_identity(
                hosted_call_indexes,
                call_id=normalized["id"],
                output_index=index,
            )
            if normalized["status"] in {"in_progress", "searching"}:
                if response_status not in {"incomplete", "failed"}:
                    raise OpenAIProtocolError(
                        "OpenAI terminal response contains a nonterminal web search call."
                    )
                normalized = {
                    "type": "web_search_call",
                    "id": normalized["id"],
                    "status": "outcome_unknown",
                }
            events.append(_web_search_call_event(normalized))
            completion_output_items.append(normalized)
            if normalized["status"] == "completed":
                provider_state_items.append({"provider": "openai", "state": normalized})
        else:
            raise OpenAIProtocolError(f"Unsupported OpenAI output item type: {item_type!r}.")

    events.append(
        _completed_event_from_response(
            response,
            provider_state_items,
            completion_output_items=completion_output_items,
            reasoning_state=reasoning_state,
        )
    )
    return events


def _openai_operation_url(base_url: str, operation_id: str) -> str:
    return f"{base_url}/v1/responses/{quote(operation_id, safe='')}"


def _openai_recovery_sequence_number(metadata: ProviderOperationRecoveryMetadata) -> int:
    value = metadata.opaque.get("sequence_number")
    if type(value) is not int or value < 0 or value > MAX_DURABLE_JSON_INTEGER:
        raise OpenAIProtocolError(
            "OpenAI background recovery metadata requires a nonnegative sequence_number."
        )
    return value


def _require_openai_background_state(state: ProviderOperationState) -> ProviderOperationState:
    state = copy_provider_operation_state(state)
    if state.stream_protocol != _OPENAI_BACKGROUND_STREAM_PROTOCOL:
        raise OpenAIProtocolError("OpenAI background operation uses an unknown stream protocol.")
    _openai_recovery_sequence_number(state.recovery_metadata)
    return state


def _openai_response_operation_status(value: object) -> ProviderOperationStatus | None:
    if type(value) is not str:
        return None
    return {
        "queued": ProviderOperationStatus.QUEUED,
        "in_progress": ProviderOperationStatus.IN_PROGRESS,
        "completed": ProviderOperationStatus.COMPLETED,
        "incomplete": ProviderOperationStatus.COMPLETED,
        "failed": ProviderOperationStatus.FAILED,
        "cancelled": ProviderOperationStatus.CANCELLED,
        "expired": ProviderOperationStatus.EXPIRED,
    }.get(value)


def _openai_background_created_state(
    event: Mapping[str, Any],
) -> tuple[ProviderOperationState, ProviderOperationStatus]:
    if not isinstance(event, Mapping) or event.get("type") != "response.created":
        raise OpenAIProtocolError("OpenAI background start must begin with response.created.")
    sequence_number = _openai_stream_sequence_number(event)
    response = _stream_response_object(event)
    response_id = response.get("id")
    if type(response_id) is not str or not response_id.strip():
        raise OpenAIProtocolError("OpenAI response.created requires a nonblank response id.")
    status = _openai_response_operation_status(response.get("status"))
    if status not in {ProviderOperationStatus.QUEUED, ProviderOperationStatus.IN_PROGRESS}:
        raise OpenAIProtocolError("OpenAI response.created requires queued or in_progress status.")
    return (
        ProviderOperationState(
            operation_id=response_id,
            stream_protocol=_OPENAI_BACKGROUND_STREAM_PROTOCOL,
            recovery_metadata=ProviderOperationRecoveryMetadata.model_validate(
                {
                    "cursor": 0,
                    "opaque": {"sequence_number": sequence_number},
                }
            ),
        ),
        status,
    )


def _openai_background_snapshot(
    state: ProviderOperationState,
    response: Mapping[str, Any],
    *,
    reasoning_state: str,
) -> ProviderOperationSnapshot:
    if not isinstance(response, Mapping):
        return ProviderOperationSnapshot(
            state=state,
            status=ProviderOperationStatus.COMPLETED,
        )
    response_id = response.get("id")
    if type(response_id) is not str or response_id != state.operation_id:
        return ProviderOperationSnapshot(
            state=state,
            status=ProviderOperationStatus.COMPLETED,
        )
    status = _openai_response_operation_status(response.get("status"))
    if status is None:
        return ProviderOperationSnapshot(
            state=state,
            status=ProviderOperationStatus.COMPLETED,
        )
    if status is not ProviderOperationStatus.COMPLETED:
        return ProviderOperationSnapshot(state=state, status=status)
    try:
        parsed = openai_response_events(response, reasoning_state=reasoning_state)
    except (OpenAIAPIError, OpenAIProtocolError, TypeError, ValueError):
        return ProviderOperationSnapshot(state=state, status=status)
    cursor = state.recovery_metadata.cursor
    cursor = 0 if cursor is None else cursor
    events: list[ModelStreamEvent] = []
    for event in parsed:
        cursor += 1
        events.append(
            event.model_copy(
                update={
                    "recovery_metadata": ProviderOperationRecoveryMetadata(
                        cursor=cursor,
                        opaque={
                            "sequence_number": _openai_recovery_sequence_number(
                                state.recovery_metadata
                            )
                        },
                    )
                },
                deep=True,
            )
        )
    return ProviderOperationSnapshot(state=state, status=status, events=tuple(events))


def _openai_stream_sequence_number(event: Mapping[str, Any]) -> int:
    value = event.get("sequence_number")
    if type(value) is not int or value < 0 or value > MAX_DURABLE_JSON_INTEGER:
        raise OpenAIProtocolError(
            "OpenAI background stream events require a nonnegative sequence_number."
        )
    return value


def _openai_stream_operation_status(event: Mapping[str, Any]) -> ProviderOperationStatus:
    event_type = event.get("type")
    by_type = {
        "response.completed": ProviderOperationStatus.COMPLETED,
        "response.incomplete": ProviderOperationStatus.COMPLETED,
        "response.failed": ProviderOperationStatus.FAILED,
        "response.cancelled": ProviderOperationStatus.CANCELLED,
        "response.expired": ProviderOperationStatus.EXPIRED,
    }
    if event_type in by_type:
        return by_type[event_type]
    response = event.get("response")
    if isinstance(response, Mapping):
        status = _openai_response_operation_status(response.get("status"))
        if status is not None:
            return status
    return ProviderOperationStatus.IN_PROGRESS


def _openai_background_parser_state(
    metadata: ProviderOperationRecoveryMetadata,
) -> tuple[dict[int, _PendingFunctionCall], set[int]]:
    parser = metadata.opaque.get("parser")
    if parser is None:
        return {}, set()
    if type(parser) is not dict:
        raise OpenAIProtocolError("OpenAI recovery parser state must be an object.")
    parser = cast("dict[str, Any]", parser)
    raw_calls = parser.get("pending_function_calls", [])
    raw_reasoning = parser.get("pending_reasoning_output_indexes", [])
    if type(raw_calls) is not list or type(raw_reasoning) is not list:
        raise OpenAIProtocolError("OpenAI recovery parser state is malformed.")
    calls: dict[int, _PendingFunctionCall] = {}
    for raw in raw_calls:
        if type(raw) is not dict:
            raise OpenAIProtocolError("OpenAI pending function-call state is malformed.")
        raw = cast("dict[str, Any]", raw)
        output_index = raw.get("output_index")
        if type(output_index) is not int or output_index < 0 or output_index in calls:
            raise OpenAIProtocolError("OpenAI pending function-call index is malformed.")
        calls[output_index] = _PendingFunctionCall(
            item_id=_mapping_optional_string(raw, "item_id"),
            call_id=_mapping_optional_string(raw, "call_id"),
            name=_mapping_optional_string(raw, "name"),
            arguments="",
        )
    reasoning: set[int] = set()
    for raw_index in raw_reasoning:
        if type(raw_index) is not int or raw_index < 0:
            raise OpenAIProtocolError("OpenAI pending reasoning index is malformed.")
        reasoning.add(raw_index)
    return calls, reasoning


def _openai_background_recovery_metadata(
    *,
    cursor: int,
    sequence_number: int,
    pending_function_calls: Mapping[int, _PendingFunctionCall],
    pending_reasoning_items: set[int],
) -> ProviderOperationRecoveryMetadata:
    opaque: dict[str, object] = {"sequence_number": sequence_number}
    if pending_function_calls or pending_reasoning_items:
        calls: list[dict[str, object]] = []
        for output_index, pending in sorted(pending_function_calls.items()):
            call: dict[str, object] = {"output_index": output_index}
            if pending.item_id is not None:
                call["item_id"] = pending.item_id
            if pending.call_id is not None:
                call["call_id"] = pending.call_id
            if pending.name is not None:
                call["name"] = pending.name
            calls.append(call)
        opaque["parser"] = {
            "pending_function_calls": calls,
            "pending_reasoning_output_indexes": sorted(pending_reasoning_items),
        }
    return ProviderOperationRecoveryMetadata(cursor=cursor, opaque=opaque)


def _openai_background_event_with_recovery(
    event: ModelStreamEvent,
    *,
    cursor: int,
    sequence_number: int,
    pending_function_calls: Mapping[int, _PendingFunctionCall],
    pending_reasoning_items: set[int],
) -> ModelStreamEvent:
    return event.model_copy(
        update={
            "recovery_metadata": _openai_background_recovery_metadata(
                cursor=cursor,
                sequence_number=sequence_number,
                pending_function_calls=pending_function_calls,
                pending_reasoning_items=pending_reasoning_items,
            )
        },
        deep=True,
    )


async def _openai_background_stream_events(
    raw_events: AsyncIterator[Mapping[str, Any]],
    *,
    state: ProviderOperationState,
    first: Mapping[str, Any] | None,
    reasoning_state: str,
) -> AsyncIterator[ModelStreamEvent]:
    cursor = state.recovery_metadata.cursor
    cursor = 0 if cursor is None else cursor
    last_sequence_number = _openai_recovery_sequence_number(state.recovery_metadata)
    pending_function_calls, pending_reasoning_items = _openai_background_parser_state(
        state.recovery_metadata
    )

    async def ordered_raw_events() -> AsyncIterator[Mapping[str, Any]]:
        if first is not None:
            yield first
        async for raw in raw_events:
            yield raw

    async for event in ordered_raw_events():
        if not isinstance(event, Mapping):
            raise OpenAIProtocolError("OpenAI stream event must be a JSON object.")
        sequence_number = _openai_stream_sequence_number(event)
        if sequence_number <= last_sequence_number:
            raise OpenAIProtocolError("OpenAI background sequence_number did not advance.")
        last_sequence_number = sequence_number
        cursor += 1
        event_type = event.get("type")
        normalized: ModelStreamEvent
        if event_type in {"response.output_text.delta", "response.refusal.delta"}:
            delta = event.get("delta")
            if not isinstance(delta, str):
                raise OpenAIProtocolError("OpenAI text delta must be a string.")
            normalized = (
                ModelStreamEvent.text_delta(delta) if delta else ModelStreamEvent.thinking()
            )
        elif event_type in {
            "response.reasoning_summary_text.delta",
            "response.reasoning_text.delta",
        }:
            delta = event.get("delta")
            if not isinstance(delta, str):
                raise OpenAIProtocolError("OpenAI reasoning delta must be a string.")
            normalized = ModelStreamEvent.thinking(delta)
        elif event_type == "response.output_item.added":
            item = event.get("item")
            if isinstance(item, Mapping) and item.get("type") == "reasoning":
                pending_reasoning_items.add(_stream_output_index(event))
            _record_stream_output_item_added(event, pending_function_calls)
            normalized = ModelStreamEvent.thinking()
        elif event_type == "response.output_item.done":
            item = event.get("item")
            if isinstance(item, Mapping) and item.get("type") == "reasoning":
                pending_reasoning_items.discard(_stream_output_index(event))
            if not isinstance(item, Mapping):
                raise OpenAIProtocolError("OpenAI output_item.done requires item object.")
            normalized = ModelStreamEvent.thinking()
        elif event_type == "response.function_call_arguments.delta":
            _record_stream_function_call_delta(event, pending_function_calls)
            normalized = ModelStreamEvent.thinking()
        elif event_type == "response.function_call_arguments.done":
            normalized, _ = _stream_function_call_event(event, pending_function_calls)
        elif event_type in {"response.completed", "response.incomplete"}:
            unfinished = {*pending_function_calls, *pending_reasoning_items}
            if event_type == "response.completed" and unfinished:
                raise OpenAIProtocolError(
                    "OpenAI background response completed with unfinished output items."
                )
            for terminal_event in _stream_terminal_events(
                event,
                {},
                excluded_output_indexes=unfinished,
                reasoning_state=reasoning_state,
            ):
                yield _openai_background_event_with_recovery(
                    terminal_event,
                    cursor=cursor,
                    sequence_number=sequence_number,
                    pending_function_calls=pending_function_calls,
                    pending_reasoning_items=pending_reasoning_items,
                )
            return
        elif event_type == "response.failed":
            failure = _openai_stream_error_exception(event)
            normalized = ModelStreamEvent.error(
                str(failure),
                cause=failure,
                provider_operation_status=ProviderOperationStatus.FAILED,
            )
        elif event_type == "error":
            failure = _openai_stream_error_exception(event)
            normalized = ModelStreamEvent.error(
                str(failure),
                cause=failure,
                provider_operation_status=ProviderOperationStatus.IN_PROGRESS,
            )
        elif event_type in {"response.cancelled", "response.expired"}:
            terminal_status = event_type.removeprefix("response.")
            failure = OpenAIAPIError(
                f"OpenAI background response reached {terminal_status} status.",
                error_type="background_response_terminal",
                error_code=f"response_{terminal_status}",
                retryable=False,
            )
            normalized = ModelStreamEvent.error(
                str(failure),
                cause=failure,
                provider_operation_status=ProviderOperationStatus(terminal_status),
            )
        else:
            normalized = ModelStreamEvent.thinking()
        yield _openai_background_event_with_recovery(
            normalized,
            cursor=cursor,
            sequence_number=sequence_number,
            pending_function_calls=pending_function_calls,
            pending_reasoning_items=pending_reasoning_items,
        )
        if event_type in {
            "response.completed",
            "response.incomplete",
            "response.failed",
            "response.cancelled",
            "response.expired",
        }:
            return


async def _empty_model_stream() -> AsyncIterator[ModelStreamEvent]:
    if False:  # pragma: no cover - preserves the async-iterator shape
        yield ModelStreamEvent.thinking()


async def _close_openai_operation_stream(
    raw_events: AsyncIterator[Mapping[str, Any]],
) -> None:
    close = getattr(raw_events, "aclose", None)
    if close is not None:
        try:
            await close()
        except BaseException:
            return


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
    """Normalize OpenAI streaming events with exact hosted-search settlement."""

    pending_call_ids: set[str] = set()
    seen_call_ids: set[str] = set()
    parsed_events = _openai_stream_events(events, reasoning_state=reasoning_state)
    try:
        async with aclosing_provider_stream(parsed_events):
            async for event in parsed_events:
                if event.type is ModelStreamEventType.HOSTED_TOOL_CALL:
                    call_id = event.payload.get("call_id")
                    status = event.payload.get("status")
                    if not isinstance(call_id, str) or not call_id:
                        raise OpenAIProtocolError(
                            "OpenAI hosted search event requires a call identity."
                        )
                    if status in {"in_progress", "searching"}:
                        if status == "in_progress":
                            if call_id in seen_call_ids:
                                raise OpenAIProtocolError(
                                    "OpenAI web search call identity was reused."
                                )
                            seen_call_ids.add(call_id)
                            pending_call_ids.add(call_id)
                        elif call_id not in pending_call_ids:
                            raise OpenAIProtocolError(
                                "OpenAI web search progress has no pending call."
                            )
                    elif status in {"completed", "incomplete", "failed", "outcome_unknown"}:
                        pending_call_ids.discard(call_id)
                yield event
    except Exception:
        for call_id in sorted(pending_call_ids):
            yield _web_search_outcome_unknown_event(call_id)
        raise


async def _openai_stream_events(
    events: AsyncIterator[Mapping[str, Any]], *, reasoning_state: str = "inline"
) -> AsyncIterator[ModelStreamEvent]:
    pending_function_calls: dict[int, _PendingFunctionCall] = {}
    pending_reasoning_items: set[int] = set()
    pending_web_search_calls: dict[int, tuple[str, str]] = {}
    streamed_text: dict[tuple[int, int], str] = {}
    streamed_text_offsets: dict[tuple[int, int], int] = {}
    assembled_text_length = 0
    fallback_output_items: dict[int, dict[str, Any]] = {}
    completed = False
    async for event in _stream_events_with_cancellation_marker(events):
        if not isinstance(event, Mapping):
            raise OpenAIProtocolError("OpenAI stream event must be a JSON object.")
        event_type = event.get("type")
        if event_type == "cayu.internal.transport_cancelled":
            for call_id, _status in pending_web_search_calls.values():
                yield _web_search_outcome_unknown_event(call_id)
            pending_web_search_calls.clear()
            continue
        if event_type == "response.output_text.delta":
            delta = event.get("delta")
            if not isinstance(delta, str):
                raise OpenAIProtocolError("OpenAI output_text delta must be a string.")
            if delta:
                output_index = event.get("output_index")
                content_index = event.get("content_index")
                if type(output_index) is int and type(content_index) is int:
                    key = (output_index, content_index)
                    streamed_text_offsets.setdefault(key, assembled_text_length)
                    streamed_text[key] = streamed_text.get(key, "") + delta
                assembled_text_length += len(delta)
                yield ModelStreamEvent.text_delta(delta)
            continue
        if event_type == "response.output_text.annotation.added":
            annotation = event.get("annotation")
            if not isinstance(annotation, Mapping):
                raise OpenAIProtocolError("OpenAI annotation.added requires annotation object.")
            output_index = _stream_output_index(event)
            content_index = event.get("content_index")
            if type(content_index) is not int or content_index < 0:
                raise OpenAIProtocolError(
                    "OpenAI annotation.added requires non-negative content_index."
                )
            if annotation.get("type") == "url_citation":
                yield _url_citation_event(
                    annotation,
                    text=streamed_text.get((output_index, content_index), ""),
                    path=f"stream.{output_index}.{content_index}",
                    text_offset=streamed_text_offsets.get((output_index, content_index), 0),
                )
            continue
        if event_type == "response.refusal.delta":
            delta = event.get("delta")
            if not isinstance(delta, str):
                raise OpenAIProtocolError("OpenAI refusal delta must be a string.")
            if delta:
                assembled_text_length += len(delta)
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
            item = event.get("item")
            if isinstance(item, Mapping) and item.get("type") == "reasoning":
                pending_reasoning_items.add(_stream_output_index(event))
            if isinstance(item, Mapping) and item.get("type") == "web_search_call":
                output_index = _stream_output_index(event)
                if output_index in pending_web_search_calls:
                    raise OpenAIProtocolError(
                        "OpenAI web_search_call output_item.added was repeated."
                    )
                normalized = _normalized_web_search_call(item, item_index=output_index)
                if normalized["status"] != "in_progress":
                    raise OpenAIProtocolError(
                        "OpenAI web_search_call output_item.added must be in progress."
                    )
                pending_web_search_calls[output_index] = (
                    normalized["id"],
                    normalized["status"],
                )
                yield _web_search_call_event(normalized)
            _record_stream_output_item_added(event, pending_function_calls)
            continue
        if event_type in {
            "response.web_search_call.in_progress",
            "response.web_search_call.searching",
            "response.web_search_call.completed",
        }:
            output_index = _stream_output_index(event)
            pending = pending_web_search_calls.get(output_index)
            if pending is None:
                raise OpenAIProtocolError(
                    "OpenAI web search lifecycle arrived before output_item.added."
                )
            item_id = _mapping_optional_string(event, "item_id")
            if item_id is not None and item_id != pending[0]:
                raise OpenAIProtocolError("OpenAI web search lifecycle item_id mismatch.")
            status = event_type.rsplit(".", 1)[-1]
            if status == "searching" and pending[1] != status:
                pending_web_search_calls[output_index] = (pending[0], status)
                yield ModelStreamEvent.hosted_tool_call(
                    {
                        "tool_type": "web_search",
                        "call_id": pending[0],
                        "status": status,
                    }
                )
            # output_item.done is the terminal evidence: it carries the bounded
            # action and sources needed for transcript replay.
            continue
        if event_type == "response.output_item.done":
            item = event.get("item")
            if isinstance(item, Mapping) and item.get("type") == "reasoning":
                pending_reasoning_items.discard(_stream_output_index(event))
            if isinstance(item, Mapping) and item.get("type") == "web_search_call":
                output_index = _stream_output_index(event)
                pending = pending_web_search_calls.pop(output_index, None)
                if pending is None:
                    raise OpenAIProtocolError(
                        "OpenAI web_search_call output_item.done arrived before added."
                    )
                normalized = _normalized_web_search_call(item, item_index=output_index)
                if normalized["id"] != pending[0]:
                    raise OpenAIProtocolError("OpenAI web_search_call output identity mismatch.")
                fallback_output_items[output_index] = normalized
                yield _web_search_call_event(normalized)
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
            unfinished_output_indexes = {
                *pending_function_calls,
                *pending_reasoning_items,
                *pending_web_search_calls,
            }
            # A completed response promises complete output items. An incomplete
            # response may end mid-item, so retain the terminal classification but
            # exclude partial state that cannot be replayed or executed safely.
            if event_type == "response.completed" and pending_function_calls:
                raise OpenAIProtocolError(
                    "OpenAI streaming response completed with unfinished function calls."
                )
            if event_type == "response.completed" and pending_reasoning_items:
                raise OpenAIProtocolError(
                    "OpenAI streaming response completed with unfinished reasoning items."
                )
            if event_type == "response.completed" and pending_web_search_calls:
                for call_id, _status in pending_web_search_calls.values():
                    yield _web_search_outcome_unknown_event(call_id)
                raise OpenAIProtocolError(
                    "OpenAI streaming response completed with unfinished web search calls."
                )
            if event_type == "response.incomplete":
                for call_id, _status in pending_web_search_calls.values():
                    yield _web_search_outcome_unknown_event(call_id)
                pending_web_search_calls.clear()
            for terminal_event in _stream_terminal_events(
                event,
                fallback_output_items,
                excluded_output_indexes=unfinished_output_indexes,
                reasoning_state=reasoning_state,
            ):
                yield terminal_event
            if event_type == "response.completed":
                return
            completed = True
            continue
        if event_type in {"response.failed", "error"}:
            failure = _openai_stream_error_exception(event)
            for call_id, _status in pending_web_search_calls.values():
                yield _web_search_outcome_unknown_event(call_id)
            # Keep raw provider envelopes out of direct parser tracebacks; the
            # outer provider wrapper is not the only supported caller.
            event = {}
            del events
            raise failure from None

    if not completed:
        for call_id, _status in pending_web_search_calls.values():
            yield _web_search_outcome_unknown_event(call_id)
        raise OpenAIProtocolError("OpenAI streaming response ended before response.completed.")


async def _stream_events_with_cancellation_marker(
    events: AsyncIterator[Mapping[str, Any]],
) -> AsyncIterator[Mapping[str, Any]]:
    iterator = events.__aiter__()
    while True:
        try:
            event = await iterator.__anext__()
        except StopAsyncIteration:
            return
        except asyncio.CancelledError:
            yield {"type": "cayu.internal.transport_cancelled"}
            raise
        yield event


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
    *,
    text_offset: int,
) -> tuple[list[ModelStreamEvent], int]:
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
    message_text_length = 0
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
        annotations = part.get("annotations", [])
        if not isinstance(annotations, list):
            raise OpenAIProtocolError(f"OpenAI {part_type} content annotations must be a list.")
        for annotation_index, annotation in enumerate(annotations):
            if not isinstance(annotation, Mapping):
                raise OpenAIProtocolError(
                    "OpenAI output annotation "
                    f"{item_index}.{content_index}.{annotation_index} must be an object."
                )
            annotation = cast("Mapping[str, Any]", annotation)
            if annotation.get("type") != "url_citation":
                continue
            events.append(
                _url_citation_event(
                    annotation,
                    text=text,
                    path=f"{item_index}.{content_index}.{annotation_index}",
                    text_offset=text_offset + message_text_length,
                )
            )
        message_text_length += len(text)
    return events, message_text_length


def _web_search_call_event(item: Mapping[str, Any]) -> ModelStreamEvent:
    action = item.get("action")
    source_count = (
        len(action.get("sources", []))
        if isinstance(action, Mapping) and isinstance(action.get("sources", []), list)
        else 0
    )
    return ModelStreamEvent.hosted_tool_call(
        {
            "tool_type": "web_search",
            "call_id": item["id"],
            "status": item["status"],
            **({"action": action, "source_count": source_count} if action is not None else {}),
        }
    )


def _web_search_outcome_unknown_event(call_id: str) -> ModelStreamEvent:
    return ModelStreamEvent.hosted_tool_call(
        {
            "tool_type": "web_search",
            "call_id": call_id,
            "status": "outcome_unknown",
        }
    )


def _claim_web_search_call_identity(
    call_indexes: dict[str, int],
    *,
    call_id: str,
    output_index: int,
) -> None:
    existing_index = call_indexes.get(call_id)
    if existing_index is not None:
        raise OpenAIProtocolError(
            "OpenAI web search call identity is duplicated across output items."
        )
    call_indexes[call_id] = output_index


def _normalized_web_search_call(
    item: Mapping[str, Any],
    *,
    item_index: int,
) -> dict[str, Any]:
    call_id = item.get("id")
    if not isinstance(call_id, str) or not call_id.strip():
        raise OpenAIProtocolError(f"OpenAI web_search_call item {item_index} requires nonblank id.")
    status = item.get("status")
    if status not in {"in_progress", "searching", "completed", "incomplete", "failed"}:
        raise OpenAIProtocolError(
            f"OpenAI web_search_call item {item_index} has unsupported status."
        )
    normalized: dict[str, Any] = {
        "type": "web_search_call",
        "id": call_id.strip(),
        "status": status,
    }
    action = item.get("action")
    if action is not None:
        normalized["action"] = _normalized_web_search_action(
            action,
            path=f"output[{item_index}].action",
        )
    if status == "completed" and "action" not in normalized:
        raise OpenAIProtocolError(
            f"OpenAI web_search_call item {item_index} completed without action evidence."
        )
    return normalized


def _normalized_web_search_action(action: object, *, path: str) -> dict[str, Any]:
    if not isinstance(action, Mapping):
        raise OpenAIProtocolError(f"OpenAI {path} must be an object.")
    action = cast("Mapping[str, Any]", action)
    action_type = action.get("type")
    if action_type not in {"search", "open_page", "find_in_page"}:
        raise OpenAIProtocolError(f"OpenAI {path}.type is unsupported.")
    normalized: dict[str, Any] = {"type": action_type}
    for key in ("query", "pattern"):
        value = action.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip() or len(value) > 4096:
            raise OpenAIProtocolError(f"OpenAI {path}.{key} must be a bounded string.")
        normalized[key] = value
    raw_url = action.get("url")
    if raw_url is not None:
        normalized["url"] = _normalized_external_web_url(raw_url, path=f"{path}.url")
    raw_queries = action.get("queries")
    if raw_queries is not None:
        if not isinstance(raw_queries, list) or not raw_queries or len(raw_queries) > 100:
            raise OpenAIProtocolError(f"OpenAI {path}.queries must be a bounded list.")
        queries: list[str] = []
        for index, query in enumerate(raw_queries):
            if not isinstance(query, str) or not query.strip() or len(query) > 4096:
                raise OpenAIProtocolError(
                    f"OpenAI {path}.queries[{index}] must be a bounded string."
                )
            queries.append(query)
        normalized["queries"] = queries
    raw_sources = action.get("sources")
    if raw_sources is not None:
        if not isinstance(raw_sources, list) or len(raw_sources) > 100:
            raise OpenAIProtocolError(f"OpenAI {path}.sources must be a bounded list.")
        sources: list[dict[str, str]] = []
        for index, source in enumerate(raw_sources):
            if not isinstance(source, Mapping):
                raise OpenAIProtocolError(f"OpenAI {path}.sources[{index}] must be an object.")
            source = cast("Mapping[str, Any]", source)
            source_type = source.get("type", "url")
            url = source.get("url")
            title = source.get("title")
            if source_type != "url":
                raise OpenAIProtocolError(f"OpenAI {path}.sources[{index}].type is unsupported.")
            url = _normalized_external_web_url(
                url,
                path=f"{path}.sources[{index}].url",
            )
            if title is not None and (
                not isinstance(title, str) or not title.strip() or len(title) > 1024
            ):
                raise OpenAIProtocolError(
                    f"OpenAI {path}.sources[{index}].title must be a bounded string."
                )
            sources.append(
                {"type": "url", "url": url, **({"title": title} if title is not None else {})}
            )
        normalized["sources"] = sources
    try:
        WebSearchAction.model_validate(normalized)
    except ValueError as exc:
        raise OpenAIProtocolError(f"OpenAI {path} is invalid.") from exc
    return normalized


def _normalized_external_web_url(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 4096:
        raise OpenAIProtocolError(f"OpenAI {path} must be a bounded URL.")
    try:
        return WebSearchSource(url=value).url
    except ValueError as exc:
        raise OpenAIProtocolError(f"OpenAI {path} must use http or https.") from exc


def _url_citation_event(
    annotation: Mapping[str, Any],
    *,
    text: str,
    path: str,
    text_offset: int = 0,
) -> ModelStreamEvent:
    url = annotation.get("url")
    title = annotation.get("title")
    start_index = annotation.get("start_index")
    end_index = annotation.get("end_index")
    url = _normalized_external_web_url(url, path=f"citation {path}.url")
    if title is not None and (not isinstance(title, str) or not title.strip() or len(title) > 1024):
        raise OpenAIProtocolError(f"OpenAI citation {path}.title must be a bounded string.")
    if (start_index is None) != (end_index is None):
        raise OpenAIProtocolError(f"OpenAI citation {path} has invalid text offsets.")
    if start_index is not None and (
        type(start_index) is not int
        or type(end_index) is not int
        or start_index < 0
        or end_index <= start_index
        or end_index > len(text)
    ):
        raise OpenAIProtocolError(f"OpenAI citation {path} has invalid text offsets.")
    payload: dict[str, Any] = {
        "citation_type": "url_citation",
        "url": url,
    }
    if title is not None:
        payload["title"] = title
    if start_index is not None:
        payload["start_index"] = text_offset + start_index
        payload["end_index"] = text_offset + cast("int", end_index)
    return ModelStreamEvent.citation(payload)


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
    usage = response.get("usage")
    if usage is not None and not isinstance(usage, Mapping):
        raise OpenAIProtocolError("OpenAI response usage must be an object.")
    payload = {
        "id": _optional_string(response, "id"),
        "model": _optional_string(response, "model"),
        "status": _optional_string(response, "status"),
        "provider_state": provider_state_items,
        "usage": copy_json_value(None if usage is None else dict(usage), "usage"),
        "incomplete_details": copy_json_value(
            response.get("incomplete_details"),
            "incomplete_details",
        ),
    }
    usage_output = completion_output_items
    if usage_output is None:
        raw_output = response.get("output")
        usage_output = raw_output if isinstance(raw_output, list) else []
    web_search_calls = sum(
        1
        for item in usage_output
        if isinstance(item, Mapping)
        and item.get("type") == "web_search_call"
        and item.get("status") in {"completed", "incomplete", "failed"}
    )
    web_search_outcome_unknown = sum(
        1
        for item in usage_output
        if isinstance(item, Mapping)
        and item.get("type") == "web_search_call"
        and item.get("status") == "outcome_unknown"
    )
    if web_search_calls or web_search_outcome_unknown:
        payload["hosted_tool_usage"] = {
            "web_search_calls": web_search_calls,
            "web_search_outcome_unknown": web_search_outcome_unknown,
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
        end_turn=_openai_end_turn(response, status=status),
    )


def _openai_end_turn(
    response: Mapping[str, Any],
    *,
    status: str | None,
) -> bool | None:
    value = response.get("end_turn")
    if value is None:
        return None
    if type(value) is not bool:
        raise OpenAIProtocolError("OpenAI response end_turn must be a boolean or null.")
    return value if status == "completed" else None


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


def _stream_terminal_events(
    event: Mapping[str, Any],
    fallback_output_items: Mapping[int, Mapping[str, Any]],
    *,
    excluded_output_indexes: set[int] | None = None,
    reasoning_state: str = "inline",
) -> list[ModelStreamEvent]:
    response = _stream_response_object(event)
    excluded_output_indexes = excluded_output_indexes or set()
    if response.get("output") is None:
        completed_output_items = {
            index: item
            for index, item in fallback_output_items.items()
            if index not in excluded_output_indexes
        }
        provider_state_items = _provider_state_items_from_output_items(completed_output_items)
        completion_output_items = list(_sorted_output_items(completed_output_items))
        return [
            _completed_event_from_response(
                response,
                provider_state_items,
                completion_output_items=completion_output_items,
                reasoning_state=reasoning_state,
            )
        ]
    output = response.get("output")
    if not isinstance(output, list):
        raise OpenAIProtocolError("OpenAI response output must be a list.")

    terminal_hosted_calls: dict[int, dict[str, Any]] = {}
    hosted_call_indexes: dict[str, int] = {}
    completion_output_items: list[Mapping[str, Any]] = []
    for output_index, item in enumerate(output):
        if output_index in excluded_output_indexes:
            continue
        if not isinstance(item, Mapping):
            raise OpenAIProtocolError(f"OpenAI output item {output_index} must be an object.")
        item = cast("Mapping[str, Any]", item)
        if item.get("type") != "web_search_call":
            completion_output_items.append(item)
            continue
        normalized = _normalized_web_search_call(item, item_index=output_index)
        _claim_web_search_call_identity(
            hosted_call_indexes,
            call_id=normalized["id"],
            output_index=output_index,
        )
        if normalized["status"] in {"in_progress", "searching"}:
            response_status = _optional_string(response, "status")
            if response_status not in {"incomplete", "failed"}:
                raise OpenAIProtocolError(
                    "OpenAI terminal response contains a nonterminal web search call."
                )
            normalized = {
                "type": "web_search_call",
                "id": normalized["id"],
                "status": "outcome_unknown",
            }
        terminal_hosted_calls[output_index] = normalized
        completion_output_items.append(normalized)

    lifecycle_hosted_calls: dict[int, Mapping[str, Any]] = {}
    for output_index, item in fallback_output_items.items():
        if item.get("type") != "web_search_call":
            continue
        lifecycle_hosted_calls[output_index] = item
        existing_index = hosted_call_indexes.get(cast("str", item.get("id")))
        if existing_index is not None and existing_index != output_index:
            raise OpenAIProtocolError(
                "OpenAI terminal web search identity conflicts with lifecycle evidence."
            )

    for output_index, lifecycle_item in lifecycle_hosted_calls.items():
        terminal_item = terminal_hosted_calls.get(output_index)
        if terminal_item is None:
            raise OpenAIProtocolError(
                "OpenAI terminal response omitted completed web search lifecycle evidence."
            )
        if terminal_item != lifecycle_item:
            raise OpenAIProtocolError(
                "OpenAI terminal web search evidence conflicts with lifecycle evidence."
            )

    terminal_events: list[ModelStreamEvent] = []
    for output_index, terminal_item in terminal_hosted_calls.items():
        fallback_item = fallback_output_items.get(output_index)
        if fallback_item is not None and fallback_item.get("type") != "web_search_call":
            raise OpenAIProtocolError(
                "OpenAI terminal web search output conflicts with lifecycle item identity."
            )
        if output_index not in lifecycle_hosted_calls:
            terminal_events.append(_web_search_call_event(terminal_item))

    response_for_completion = (
        {
            **response,
            "output": [
                item
                for output_index, item in enumerate(output)
                if output_index not in excluded_output_indexes
            ],
        }
        if excluded_output_indexes
        else response
    )
    provider_state_items = _provider_state_items_from_response(response_for_completion)
    terminal_events.append(
        _completed_event_from_response(
            response_for_completion,
            provider_state_items,
            completion_output_items=completion_output_items,
            reasoning_state=reasoning_state,
        )
    )
    return terminal_events


def _provider_state_items_from_response(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = response.get("output")
    if output is None:
        return []
    if not isinstance(output, list):
        raise OpenAIProtocolError("OpenAI response output must be a list.")
    provider_state_items: list[dict[str, Any]] = []
    hosted_call_indexes: dict[str, int] = {}
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
        if item_type == "web_search_call":
            normalized = _normalized_web_search_call(item, item_index=index)
            _claim_web_search_call_identity(
                hosted_call_indexes,
                call_id=normalized["id"],
                output_index=index,
            )
            if normalized["status"] == "completed":
                provider_state_items.append(
                    {
                        "provider": "openai",
                        "state": normalized,
                    }
                )
            continue
        raise OpenAIProtocolError(f"Unsupported OpenAI output item type: {item_type!r}.")
    return provider_state_items


def _provider_state_items_from_output_items(
    output_items: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    provider_state_items: list[dict[str, Any]] = []
    for item in _sorted_output_items(output_items):
        if item.get("type") == "web_search_call" and item.get("status") != "completed":
            continue
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


def _openai_stream_error_exception(event: Mapping[str, Any]) -> OpenAIAPIError:
    retry_after_s = _trusted_sse_retry_after_s(event)
    event_type = event.get("type")
    if event_type == "response.failed":
        response = _stream_response_object(event)
        error = response.get("error")
        error_mapping = error if isinstance(error, Mapping) else {}
        status_code, status_conflict = _openai_stream_status_code(
            error_mapping,
            response,
            event,
        )
        return _openai_error_value_exception(
            response if error is None else error,
            safe_message=f"OpenAI streaming error: {OMITTED_PROVIDER_ERROR_BODY}",
            request_id=optional_error_string(response.get("request_id")),
            retry_after_s=retry_after_s,
            transport_status_code=status_code,
            status_conflict=status_conflict,
        )
    status_code, status_conflict = _openai_stream_status_code(event)
    return _openai_error_value_exception(
        event,
        safe_message=f"OpenAI streaming error: {OMITTED_PROVIDER_ERROR_BODY}",
        request_id=optional_error_string(event.get("request_id")),
        retry_after_s=retry_after_s,
        transport_status_code=status_code,
        status_conflict=status_conflict,
    )


def _openai_error_value_exception(
    error: Any,
    *,
    safe_message: str,
    request_id: str | None,
    retry_after_s: float | None = None,
    transport_status_code: int | None = None,
    status_conflict: bool = False,
) -> OpenAIAPIError:
    error_mapping = error if isinstance(error, Mapping) else {}
    error_type = optional_error_string(error_mapping.get("type"))
    error_code = optional_error_string(error_mapping.get("code"))
    param = optional_error_string(error_mapping.get("param"))
    resolved_request_id = request_id or optional_error_string(error_mapping.get("request_id"))
    raw_message = optional_error_string(error_mapping.get("message"))
    if not status_conflict and _is_openai_context_overflow(
        status_code=transport_status_code,
        error_type=error_type,
        code=error_code,
        message=raw_message,
    ):
        return OpenAIContextOverflowError(
            "OpenAI model context overflow",
            status_code=transport_status_code,
            error_type=error_type,
            error_code=error_code,
            request_id=resolved_request_id,
            response_body=None,
        )
    status_code, retryable = _openai_retry_metadata(
        transport_status_code=transport_status_code,
        error_type=error_type,
        error_code=error_code,
    )
    if status_conflict:
        status_code = None
        retryable = False
    return OpenAIAPIError(
        safe_message,
        status_code=status_code,
        error_type=error_type,
        error_code=error_code,
        param=param,
        request_id=resolved_request_id,
        retryable=retryable,
        retry_after_s=retry_after_s,
        response_body=None,
    )


def _openai_stream_status_code(
    *values: Mapping[str, Any],
) -> tuple[int | None, bool]:
    """Read only explicit SSE status fields and fail closed on disagreement."""

    statuses = {
        status
        for value in values
        if type(status := value.get("status_code")) is int and 100 <= status <= 599
    }
    if len(statuses) > 1:
        return None, True
    return (next(iter(statuses)), False) if statuses else (None, False)


_STALE_CHAIN_ERROR_CODE = "previous_response_not_found"
_STALE_CHAIN_PARAM = "previous_response_id"


_OPENAI_ERROR_TYPE_CLASSIFICATION = {
    "authentication_error": (401, False),
    "insufficient_quota": (429, False),
    "invalid_request_error": (400, False),
    "not_found_error": (404, False),
    "permission_error": (403, False),
    "rate_limit_error": (429, True),
    "server_error": (500, True),
}
_OPENAI_ERROR_CODE_CLASSIFICATION = {
    "bad_request": (400, False),
    "context_length_exceeded": (400, False),
    "insufficient_quota": (429, False),
    "internal_error": (500, True),
    "previous_response_not_found": (404, False),
    "rate_limit_exceeded": (429, True),
    "server_error": (500, True),
}
_OPENAI_RETRYABLE_SERVER_STATUS_CODES = frozenset({500, 502, 503, 504})


def _openai_retry_metadata(
    *,
    transport_status_code: int | None,
    error_type: str | None,
    error_code: str | None,
) -> tuple[int | None, bool | None]:
    """Classify recognized HTTP/stream identities; conflicts fail closed."""
    # OpenAI reports an expired previous response as a 404 while retaining the
    # broad ``invalid_request_error`` envelope.  The specific stale-chain code
    # is authoritative for this documented combination; other recognized
    # type/code disagreements remain conflicts.
    if error_type == "invalid_request_error" and error_code == _STALE_CHAIN_ERROR_CODE:
        classification = _OPENAI_ERROR_CODE_CLASSIFICATION[_STALE_CHAIN_ERROR_CODE]
    else:
        classifications = {
            classification
            for classification in (
                _OPENAI_ERROR_TYPE_CLASSIFICATION.get(error_type or ""),
                _OPENAI_ERROR_CODE_CLASSIFICATION.get(error_code or ""),
            )
            if classification is not None
        }
        if not classifications:
            return transport_status_code, None
        if len(classifications) != 1:
            return transport_status_code, False
        classification = next(iter(classifications))
    canonical_status, retryable = classification
    if transport_status_code is not None and transport_status_code != canonical_status:
        if retryable and canonical_status == 500:
            return (
                transport_status_code,
                transport_status_code in _OPENAI_RETRYABLE_SERVER_STATUS_CODES,
            )
        return transport_status_code, False
    return canonical_status, retryable


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


def _effective_openai_request_options(options: Mapping[str, Any]) -> dict[str, Any]:
    effective = _openai_options(options)
    _apply_thinking_options(effective, options.get("thinking"))
    return effective


def preflight_openai_native_structured_output_schema(json_schema: dict[str, Any]) -> None:
    """Reject a NATIVE structured-output schema OpenAI strict mode always refuses.

    Checks only the structural rules that have been invariant since strict
    mode launched: the root must be an object schema, every object schema must
    set ``additionalProperties: false``, every ``properties`` key must appear
    in ``required``, and every ``$ref`` must resolve inside the document,
    carry no sibling keys beyond ``$defs``/``definitions``, and point at a
    schema that itself satisfies these rules. Keyword-level limits are
    deliberately not checked: OpenAI has only ever relaxed them, so a denylist
    would falsely reject schemas the API now accepts. Violations outside this
    core still surface as an OpenAI-side 400.

    Raises ``NativeStructuredOutputSchemaInvalid`` (a ``ValueError``) with the
    offending JSON path for schema-rule violations, and ``TypeError`` when
    ``json_schema`` is not a dict (a caller error, matching the runtime's
    copy/validate conventions).
    """
    if type(json_schema) is not dict:
        raise TypeError("Native structured output schema must be an object.")
    root: Any = json_schema
    if "$ref" in json_schema:
        root = _resolve_openai_schema_ref(json_schema["$ref"], document=json_schema, path="$")
    if type(root) is not dict or not _is_openai_object_schema(root):
        raise NativeStructuredOutputSchemaInvalid(
            "$: OpenAI native structured output requires the root schema to be an object type."
        )
    _walk_openai_strict_schema(json_schema, document=json_schema, path="$", walked_refs=set())


def _walk_openai_strict_schema(
    schema: Any,
    *,
    document: dict[str, Any],
    path: str,
    walked_refs: set[str],
    depth: int = 0,
) -> None:
    # Boolean subschemas (`true`/`false`) are terminal; anything else non-dict
    # is malformed JSON Schema and already rejected by the runtime's generic
    # Draft 2020-12 check.
    if type(schema) is not dict:
        return
    if depth > _OPENAI_SCHEMA_PREFLIGHT_MAX_DEPTH:
        raise NativeStructuredOutputSchemaInvalid(
            f"{path}: schema nesting exceeds the preflight depth limit "
            f"({_OPENAI_SCHEMA_PREFLIGHT_MAX_DEPTH})."
        )
    if "$ref" in schema:
        # OpenAI rejects any $ref sibling key except the $defs/definitions
        # container the root form needs ({"$ref": "#/$defs/x", "$defs": ...}
        # is accepted; a description sibling is a 400 — probed live 2026-07).
        siblings = sorted(key for key in schema if key not in _OPENAI_REF_SIBLING_ALLOWLIST)
        if siblings:
            raise NativeStructuredOutputSchemaInvalid(
                f"{path}: OpenAI native structured output does not allow $ref to "
                f"carry sibling keys (found: {', '.join(siblings)}); only "
                "$defs/definitions may accompany it."
            )
        ref = schema["$ref"]
        target = _resolve_openai_schema_ref(ref, document=document, path=path)
        # Walk each distinct ref target exactly once so referenced schemas are
        # held to the same rules wherever they live in the document, while
        # recursive schemas ("$ref": "#", supported by OpenAI) cannot loop.
        if ref not in walked_refs:
            walked_refs.add(ref)
            _walk_openai_strict_schema(
                target,
                document=document,
                path=f"${ref[1:]}",
                walked_refs=walked_refs,
                depth=depth + 1,
            )
    if _is_openai_object_schema(schema):
        _check_openai_object_schema(schema, path=path)
    for keyword in ("properties", "$defs", "definitions"):
        members = schema.get(keyword)
        if type(members) is dict:
            for key, subschema in members.items():
                _walk_openai_strict_schema(
                    subschema,
                    document=document,
                    path=f"{path}/{keyword}/{escape_json_pointer_segment(str(key))}",
                    walked_refs=walked_refs,
                    depth=depth + 1,
                )
    items = schema.get("items")
    if items is not None:
        _walk_openai_strict_schema(
            items,
            document=document,
            path=f"{path}/items",
            walked_refs=walked_refs,
            depth=depth + 1,
        )
    for keyword in ("prefixItems", "anyOf", "oneOf", "allOf"):
        entries = schema.get(keyword)
        if type(entries) is list:
            for index, subschema in enumerate(entries):
                _walk_openai_strict_schema(
                    subschema,
                    document=document,
                    path=f"{path}/{keyword}[{index}]",
                    walked_refs=walked_refs,
                    depth=depth + 1,
                )
    additional = schema.get("additionalProperties")
    if type(additional) is dict:
        _walk_openai_strict_schema(
            additional,
            document=document,
            path=f"{path}/additionalProperties",
            walked_refs=walked_refs,
            depth=depth + 1,
        )


def _check_openai_object_schema(schema: dict[str, Any], *, path: str) -> None:
    if schema.get("additionalProperties") is not False:
        raise NativeStructuredOutputSchemaInvalid(
            f"{path}: OpenAI native structured output requires every object schema "
            "to set additionalProperties: false."
        )
    properties = schema.get("properties")
    property_names = list(properties) if type(properties) is dict else []
    required = schema.get("required")
    required_names = set(required) if type(required) is list else set()
    missing = [name for name in property_names if name not in required_names]
    if missing:
        raise NativeStructuredOutputSchemaInvalid(
            f"{path}: OpenAI native structured output requires every property to be "
            f"listed in required; missing: {', '.join(sorted(missing))}."
        )


def _is_openai_object_schema(schema: dict[str, Any]) -> bool:
    if "properties" in schema:
        return True
    schema_type = schema.get("type")
    if schema_type == "object":
        return True
    return type(schema_type) is list and "object" in schema_type


def _resolve_openai_schema_ref(ref: Any, *, document: dict[str, Any], path: str) -> Any:
    if type(ref) is not str or not ref.startswith("#"):
        raise NativeStructuredOutputSchemaInvalid(
            f"{path}/$ref: OpenAI native structured output requires internal $refs "
            f"(a JSON pointer starting with '#'); got: {ref!r}."
        )
    pointer = ref[1:]
    if pointer and not pointer.startswith("/"):
        raise NativeStructuredOutputSchemaInvalid(
            f"{path}/$ref: $ref does not resolve within the schema document: {ref!r}."
        )
    target: Any = document
    if pointer:
        for raw_segment in pointer[1:].split("/"):
            # Escape handling is deliberately lenient: ~1/~0 are decoded and
            # everything else is matched literally. OpenAI's own resolver
            # accepts non-RFC-6901 segments the same way (probed live 2026-07:
            # "#/$defs/a~b" resolves against a literal "a~b" key), so strict
            # escape validation here would falsely reject working schemas.
            segment = unescape_json_pointer_segment(raw_segment)
            if type(target) is dict and segment in target:
                target = target[segment]
            elif (
                type(target) is list
                and _OPENAI_POINTER_INDEX_RE.fullmatch(segment)
                and int(segment) < len(target)
            ):
                target = target[int(segment)]
            else:
                raise NativeStructuredOutputSchemaInvalid(
                    f"{path}/$ref: $ref does not resolve within the schema document: {ref!r}."
                )
    return target


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
    # The schema is forwarded verbatim on purpose: rewriting it here would
    # make the provider enforce a different contract than the runtime
    # validates the final JSON against. The stable structural core of strict
    # mode is rejected earlier by preflight_openai_native_structured_output_schema
    # (via the runtime's pre-session preflight); rules outside that core are
    # OpenAI-defined and drift, so violations still surface as an OpenAI-side 400.
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
        if not use_provider_state:
            return _openai_neutral_assistant_items(message)

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
            elif type(part) not in {
                TextPart,
                ProviderStatePart,
                ThinkingPart,
                HostedToolCallPart,
                CitationPart,
            }:
                raise OpenAIProtocolError(
                    "Assistant messages can only contain text, tool_call, provider_state, "
                    "thinking, hosted_tool_call, and citation parts."
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


def _openai_neutral_assistant_items(message: Message) -> list[dict[str, Any]]:
    """Rebuild accepted assistant output without server-owned provider state."""

    items: list[dict[str, Any]] = []
    pending_text: list[str] = []
    pending_citations: list[CitationPart] = []
    assembled_text_length = 0
    pending_text_offset = 0

    def flush_text() -> None:
        nonlocal pending_text_offset
        if not pending_text:
            if pending_citations:
                raise OpenAIProtocolError(
                    "OpenAI neutral replay cannot attach a citation without assistant text."
                )
            return
        text = "".join(pending_text)
        annotations = [
            _openai_neutral_citation(
                citation,
                text_offset=pending_text_offset,
                text_length=len(text),
            )
            for citation in pending_citations
        ]
        output_text: dict[str, Any] = {"type": "output_text", "text": text}
        if annotations:
            output_text["annotations"] = annotations
        items.append(
            {
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [output_text],
            }
        )
        pending_text.clear()
        pending_citations.clear()
        pending_text_offset = assembled_text_length

    for part in message.content:
        if type(part) is TextPart:
            if not pending_text:
                pending_text_offset = assembled_text_length
            pending_text.append(part.text)
            assembled_text_length += len(part.text)
            continue
        if type(part) is CitationPart:
            pending_citations.append(part)
            continue
        if type(part) is HostedToolCallPart:
            if part.status == "completed":
                flush_text()
                if part.action is None:  # pragma: no cover - model validation owns this
                    raise OpenAIProtocolError(
                        "Completed hosted search replay requires action evidence."
                    )
                action = part.action.model_dump(mode="json", exclude_none=True)
                if not action.get("queries"):
                    action.pop("queries", None)
                if not action.get("sources"):
                    action.pop("sources", None)
                items.append(
                    {
                        "type": "web_search_call",
                        "id": part.call_id,
                        "status": "completed",
                        "action": action,
                    }
                )
            continue
        if type(part) is ToolCallPart:
            flush_text()
            items.append(_function_call_input_item(part))
            continue
        if type(part) in {ProviderStatePart, ThinkingPart}:
            continue
        raise OpenAIProtocolError(
            "Assistant messages can only contain text, tool_call, provider_state, "
            "thinking, hosted_tool_call, and citation parts."
        )
    flush_text()
    return items


def _openai_neutral_citation(
    citation: CitationPart,
    *,
    text_offset: int,
    text_length: int,
) -> dict[str, Any]:
    annotation: dict[str, Any] = {
        "type": "url_citation",
        "url": citation.url,
    }
    if citation.title is not None:
        annotation["title"] = citation.title
    if citation.start_index is not None and citation.end_index is not None:
        segment_end = text_offset + text_length
        if citation.start_index < text_offset or citation.end_index > segment_end:
            raise OpenAIProtocolError(
                "OpenAI neutral replay citation does not belong to its assistant text item."
            )
        annotation["start_index"] = citation.start_index - text_offset
        annotation["end_index"] = citation.end_index - text_offset
    return annotation


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
        if item_type not in {"message", "function_call", "web_search_call"}:
            raise OpenAIProtocolError(
                f"Unsupported OpenAI provider state item type: {item_type!r}."
            )
        items.append(state)
    return items


def _user_input_part(
    part: TextPart
    | ToolCallPart
    | ToolResultPart
    | ProviderStatePart
    | ThinkingPart
    | FilePart
    | HostedToolCallPart
    | CitationPart,
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
    part: TextPart
    | ToolCallPart
    | ToolResultPart
    | ProviderStatePart
    | ThinkingPart
    | FilePart
    | HostedToolCallPart
    | CitationPart,
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
    part: TextPart
    | ToolCallPart
    | ToolResultPart
    | ProviderStatePart
    | ThinkingPart
    | FilePart
    | HostedToolCallPart
    | CitationPart,
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
    _validate_openai_tool_name(name)
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


def _openai_hosted_tool(tool: OpenAIWebSearch) -> dict[str, Any]:
    if type(tool) is not OpenAIWebSearch:
        raise TypeError("OpenAI hosted tools must be OpenAIWebSearch instances.")
    projected: dict[str, Any] = {
        "type": "web_search",
        "search_context_size": tool.search_context_size,
        "external_web_access": tool.external_web_access,
    }
    filters: dict[str, list[str]] = {}
    if tool.allowed_domains:
        filters["allowed_domains"] = list(tool.allowed_domains)
    if tool.blocked_domains:
        filters["blocked_domains"] = list(tool.blocked_domains)
    if filters:
        projected["filters"] = filters
    if tool.return_token_budget != "default":
        projected["return_token_budget"] = tool.return_token_budget
    return projected


def _preflight_openai_hosted_tools(
    *,
    model: str,
    hosted_tools: tuple[OpenAIWebSearch, ...],
    options: dict[str, Any],
    endpoint_supported: bool = True,
) -> None:
    require_clean_nonblank(model, "model")
    if type(hosted_tools) is not tuple or any(
        type(tool) is not OpenAIWebSearch for tool in hosted_tools
    ):
        raise TypeError("hosted_tools must contain exact OpenAIWebSearch instances.")
    if type(options) is not dict:
        raise TypeError("Hosted-tool provider options must be a dictionary.")
    if not hosted_tools:
        return
    if not endpoint_supported:
        raise HostedToolCapabilityError(
            "OpenAI hosted web search is not established for this custom endpoint; "
            "set hosted_web_search_supported=True only after verifying its Responses contract."
        )
    if model not in _OPENAI_HOSTED_WEB_SEARCH_MODELS:
        raise HostedToolCapabilityError(
            f"OpenAI hosted web search support is not established for model {model!r}."
        )
    effective = _effective_openai_request_options(options)
    reasoning = effective.get("reasoning")
    if isinstance(reasoning, Mapping) and reasoning.get("effort") == "minimal":
        raise HostedToolCapabilityError(
            "OpenAI hosted web search does not support minimal reasoning effort."
        )
    if any(tool.return_token_budget == "unlimited" for tool in hosted_tools) and not re.match(
        r"^gpt-5(?:[.-]|$)", model
    ):
        raise HostedToolCapabilityError(
            "OpenAI return_token_budget='unlimited' requires a GPT-5+ reasoning model."
        )


def _validate_openai_tool_name(name: str) -> None:
    if not _OPENAI_TOOL_NAME_RE.fullmatch(name):
        raise ValueError(
            "OpenAI tool names must contain 1-64 letters, numbers, underscores, or hyphens."
        )


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


def _openai_api_error_from_response(
    response: httpx.Response,
    message: str,
    retry_after_s: float | None,
) -> OpenAIAPIError:
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
    error_type = optional_error_string(error.get("type"))
    error_code = optional_error_string(error.get("code"))
    status_code, retryable = _openai_retry_metadata(
        transport_status_code=response.status_code,
        error_type=error_type,
        error_code=error_code,
    )
    return OpenAIAPIError(
        message,
        status_code=status_code,
        error_type=error_type,
        error_code=error_code,
        param=optional_error_string(error.get("param")),
        request_id=optional_error_string(request_id),
        retryable=retryable,
        retry_after_s=retry_after_s,
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
    if not _is_openai_context_overflow(
        status_code=status_code,
        error_type=error_type,
        code=code,
        message=message,
    ):
        return
    raise OpenAIContextOverflowError(
        "OpenAI model context overflow",
        status_code=status_code,
        error_type=error_type,
        error_code=code,
        request_id=request_id,
        response_body=response_body,
    )


def _is_openai_context_overflow(
    *,
    status_code: int | None,
    error_type: str | None,
    code: str | None,
    message: str | None,
) -> bool:
    classifications = {
        classification[0]
        for classification in (
            _OPENAI_ERROR_TYPE_CLASSIFICATION.get(error_type or ""),
            _OPENAI_ERROR_CODE_CLASSIFICATION.get(code or ""),
        )
        if classification is not None
    }
    if status_code is not None:
        classifications.add(status_code)
    if classifications and classifications != {400}:
        return False
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


def _is_stale_chain_error(exc: Exception) -> bool:
    """Classify a stale server-side chain from coherent typed error identity.

    OpenAI reports a stale/expired ``previous_response_id`` as HTTP 404 with
    ``code: "previous_response_not_found"`` (``param: "previous_response_id"``).
    Recovery requires that positive 404 evidence and rejects recognized type or
    code identities that conflict with it.  Classification never reads message
    text, so unrelated errors that merely mention the field (e.g. a 400 for
    combining it with ``conversation``) are not misclassified as recoverable.
    """
    if not isinstance(exc, OpenAIAPIError):
        return False
    if exc.status_code != 404:
        return False
    if exc.error_code != _STALE_CHAIN_ERROR_CODE and exc.param != _STALE_CHAIN_PARAM:
        return False
    if exc.param is not None and exc.param != _STALE_CHAIN_PARAM:
        return False

    type_classification = _OPENAI_ERROR_TYPE_CLASSIFICATION.get(exc.error_type or "")
    if (
        type_classification is not None
        and exc.error_type != "invalid_request_error"
        and type_classification[0] != 404
    ):
        return False

    code_classification = _OPENAI_ERROR_CODE_CLASSIFICATION.get(exc.error_code or "")
    return code_classification is None or code_classification[0] == 404


def _validate_reasoning_state(value: str) -> str:
    if value not in _VALID_REASONING_STATES:
        raise ValueError(f"reasoning_state must be one of {sorted(_VALID_REASONING_STATES)}.")
    return value
