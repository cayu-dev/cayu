from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Iterable, Mapping
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Protocol, cast
from urllib.parse import quote, urlencode

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    copy_json_value,
    escape_json_pointer_segment,
    require_clean_nonblank,
    require_durable_clean_nonblank,
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
from cayu.providers._config import positive_finite_seconds
from cayu.providers._credential_boundary import (
    aclosing_provider_stream,
    close_provider_stream_after_deadline,
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
from cayu.providers._openai_protocol import protocol_diagnostic_fields
from cayu.providers.base import (
    EXACT_MODEL_STREAM_RECOVERY_DISPOSITION,
    MANUAL_MODEL_STREAM_RECOVERY_DISPOSITION,
    OPENAI_ADDITIONAL_TOOLS_PROTOCOL,
    OPENAI_CLIENT_TOOL_SEARCH_PROTOCOL,
    OPENAI_HOSTED_TOOL_SEARCH_PROTOCOL,
    TARGETED_TOOL_PROJECTION_MARKER_TYPE,
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
    ModelStreamDeadlineError,
    ModelStreamEvent,
    ModelStreamEventType,
    NativeStructuredOutputSchemaInvalid,
    TargetedToolProjectionRequest,
    ToolDiscoveryProjectionRequest,
    ToolDiscoveryProjectionResult,
    UsageDialect,
    _guard_normalized_provider_stream,
    _preflight_provider_portable_messages,
    _terminal_preserving_provider_stream,
    call_tool_core_callable,
    privacy_safe_provider_option_projection,
    targeted_tool_native_cache_anchor_name,
)
from cayu.providers.deadlines import (
    ProviderDeadlineKind,
    ProviderProgressKind,
    ProviderStreamDeadlineController,
    ProviderStreamDeadlineExceeded,
    ProviderStreamDeadlines,
    _resolve_provider_stream_deadlines,
    bind_provider_deadline_controller,
    observe_provider_semantic_progress,
    reset_provider_deadline_controller,
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
from cayu.vaults import REDACTED_SECRET

if TYPE_CHECKING:
    import httpx

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com"
DEFAULT_OPENAI_TIMEOUT_SECONDS = 60.0
OPENAI_CONTEXT_PRESSURE_TOOL_SCHEMA_CHARS_PER_TOKEN = 6
_CAYU_SEARCH_TOOLS_NAME = "search_tools"
_CAYU_CALL_TOOL_NAME = "call_tool"
_OPENAI_CLIENT_TOOL_SEARCH_MAX_ITEMS = 256
_OPENAI_CLIENT_TOOL_SEARCH_MAX_QUERY_CHARS = 256
_OPENAI_CLIENT_TOOL_SEARCH_MAX_RESULTS = 8
_OPENAI_HOSTED_TOOL_SEARCH_MAX_ARGUMENT_BYTES = 64 * 1024
_OPENAI_CLIENT_TOOL_SEARCH_DESCRIPTION = (
    "Search the current session's registered tool catalogue. Matching registered "
    "functions are loaded for direct use through the client Tool Search protocol. "
    "Already visible tools are omitted."
)
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
_OPENAI_NATIVE_CAPABILITY_MAX_MODELS = 256
_OPENAI_NATIVE_CAPABILITY_MODEL_MAX_BYTES = 1024


def _copy_exact_model_allowlist(
    value: Iterable[str] | None,
    *,
    field_name: str,
) -> frozenset[str]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise TypeError(f"{field_name} must be an iterable of model names.")
    try:
        iterator = iter(() if value is None else value)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an iterable of model names.") from exc
    models: list[str] = []
    for index, configured_model in enumerate(iterator):
        if index >= _OPENAI_NATIVE_CAPABILITY_MAX_MODELS:
            raise ValueError(
                f"{field_name} cannot contain more than "
                f"{_OPENAI_NATIVE_CAPABILITY_MAX_MODELS} names."
            )
        if type(configured_model) is not str:
            raise TypeError(f"{field_name} must contain strings.")
        model = require_durable_clean_nonblank(
            configured_model,
            f"{field_name} item",
        )
        if len(model.encode("utf-8")) > _OPENAI_NATIVE_CAPABILITY_MODEL_MAX_BYTES:
            raise ValueError(
                f"{field_name} items cannot exceed "
                f"{_OPENAI_NATIVE_CAPABILITY_MODEL_MAX_BYTES} UTF-8 bytes."
            )
        models.append(model)
    if len(models) != len(set(models)):
        raise ValueError(f"{field_name} must contain unique model names.")
    return frozenset(models)


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
    """Responses validation failure with an allowlisted diagnostic identity.

    ``reason_code`` is independent of the exception message. Unknown values are
    projected as ``unspecified``; raw messages never supply diagnostic fields.
    """

    def __init__(self, message: str, *, reason_code: str = "unspecified") -> None:
        super().__init__(message)
        self.reason_code = reason_code


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
        transport_idle_timeout_s: float,
        protocol_idle_timeout_s: float,
        semantic_progress_timeout_s: float,
        absolute_stream_timeout_s: float,
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
        transport_idle_timeout_s: float,
        protocol_idle_timeout_s: float,
        semantic_progress_timeout_s: float,
        absolute_stream_timeout_s: float,
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
        transport_idle_timeout_s: float,
        protocol_idle_timeout_s: float,
        semantic_progress_timeout_s: float,
        absolute_stream_timeout_s: float,
    ) -> AsyncIterator[Mapping[str, Any]]:
        url = _validate_url(url, "url")
        events = stream_sse_json_events(
            client=self._client.get(),
            url=url,
            headers=headers,
            payload=payload,
            timeout_s=timeout_s,
            transport_idle_timeout_s=transport_idle_timeout_s,
            protocol_idle_timeout_s=protocol_idle_timeout_s,
            semantic_progress_timeout_s=semantic_progress_timeout_s,
            absolute_stream_timeout_s=absolute_stream_timeout_s,
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
        transport_idle_timeout_s: float,
        protocol_idle_timeout_s: float,
        semantic_progress_timeout_s: float,
        absolute_stream_timeout_s: float,
    ) -> AsyncIterator[Mapping[str, Any]]:
        reconnect_url = f"{url}?{urlencode({'stream': 'true', 'starting_after': starting_after})}"
        events = stream_sse_json_events(
            client=self._client.get(),
            method="GET",
            url=reconnect_url,
            headers=headers,
            payload={},
            timeout_s=timeout_s,
            transport_idle_timeout_s=transport_idle_timeout_s,
            protocol_idle_timeout_s=protocol_idle_timeout_s,
            semantic_progress_timeout_s=semantic_progress_timeout_s,
            absolute_stream_timeout_s=absolute_stream_timeout_s,
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
        self._provider._preflight_dynamic_tool_request(request.request)
        payload = build_openai_payload(
            request.request,
            stream=True,
            reasoning_state=self._provider.reasoning_state,
        )
        payload["background"] = True
        payload["store"] = True
        raw_events: AsyncIterator[Mapping[str, Any]] | None = None
        controller = ProviderStreamDeadlineController(self._provider.stream_deadlines)
        controller_handed_off = False
        try:
            raw_events = self._transport.stream_response_events(
                url=f"{self._provider.base_url}/v1/responses",
                headers=self._provider._headers(),
                payload=payload,
                timeout_s=self._provider.timeout_s,
                transport_idle_timeout_s=self._provider.stream_deadlines.transport_idle_timeout_s,
                protocol_idle_timeout_s=self._provider.stream_deadlines.protocol_idle_timeout_s,
                semantic_progress_timeout_s=(
                    self._provider.stream_deadlines.semantic_progress_timeout_s
                ),
                absolute_stream_timeout_s=self._provider.stream_deadlines.absolute_stream_timeout_s,
            )
            created = await self._next_raw_event(
                raw_events,
                controller=controller,
                empty_message="OpenAI background start ended before response.created.",
            )
            state, status = _openai_background_created_state(
                created,
                targeted_tool_marker_id=(
                    None
                    if request.request.targeted_tool_projection is None
                    else request.request.targeted_tool_projection.marker_id
                ),
                discovery_loaded_tool_names=(
                    _openai_discovery_ownership_tokens(request.request.tool_discovery_projection)
                ),
            )
            controller.observe_semantic(ProviderProgressKind.RESPONSE_IDENTITY)
            semantic_pause_started = controller.idle_pause_started()
            connection = ProviderOperationConnection(
                state=state,
                status=status,
                events=self._events(
                    raw_events,
                    state=state,
                    controller=controller,
                    semantic_pause_started=semantic_pause_started,
                ),
            )
            controller_handed_off = True
            return connection
        except asyncio.CancelledError as exc:
            if raw_events is not None:
                await self._close_raw_events(raw_events)
            raise self._safe_cancellation(exc) from None
        except Exception as exc:
            cleanup_failed = (
                False
                if raw_events is None
                else await self._close_raw_events(
                    raw_events,
                    deadline_failure=type(exc) is ModelStreamDeadlineError,
                )
            )
            raise self._safe_failure(
                _openai_operation_failure_after_close(
                    exc,
                    cleanup_failed=cleanup_failed,
                )
            ) from None
        except BaseException:
            if raw_events is not None:
                await _close_openai_operation_stream(raw_events)
            raise
        finally:
            if not controller_handed_off:
                controller.close()

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
        controller = ProviderStreamDeadlineController(self._provider.stream_deadlines)
        controller_handed_off = False
        controller.observe_semantic(ProviderProgressKind.RESPONSE_IDENTITY)
        try:
            raw_events = self._transport.reconnect_response_events(
                url=url,
                headers=self._provider._headers(),
                starting_after=sequence_number,
                timeout_s=self._provider.timeout_s,
                transport_idle_timeout_s=self._provider.stream_deadlines.transport_idle_timeout_s,
                protocol_idle_timeout_s=self._provider.stream_deadlines.protocol_idle_timeout_s,
                semantic_progress_timeout_s=(
                    self._provider.stream_deadlines.semantic_progress_timeout_s
                ),
                absolute_stream_timeout_s=self._provider.stream_deadlines.absolute_stream_timeout_s,
            )
            first = await self._next_raw_event(
                raw_events,
                controller=controller,
                empty_message=None,
                exact_operation=True,
            )
            status = _openai_stream_operation_status(first)
            semantic_pause_started = controller.idle_pause_started()
            connection = ProviderOperationConnection(
                state=state,
                status=status,
                events=self._events(
                    raw_events,
                    state=state,
                    first=first,
                    controller=controller,
                    semantic_pause_started=semantic_pause_started,
                ),
            )
            controller_handed_off = True
            return connection
        except StopAsyncIteration:
            return ProviderOperationConnection(
                state=state,
                status=ProviderOperationStatus.IN_PROGRESS,
                events=_empty_model_stream(),
            )
        except OpenAIAPIError as exc:
            if raw_events is not None:
                await self._close_raw_events(raw_events)
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
                await self._close_raw_events(raw_events)
            raise self._safe_cancellation(exc) from None
        except Exception as exc:
            cleanup_failed = (
                False
                if raw_events is None
                else await self._close_raw_events(
                    raw_events,
                    deadline_failure=type(exc) is ModelStreamDeadlineError,
                )
            )
            safe_failure = self._safe_failure(
                _openai_operation_failure_after_close(
                    exc,
                    cleanup_failed=cleanup_failed,
                )
            )
            if type(safe_failure) is ModelStreamDeadlineError:
                # Reconnect is entered with a validated exact response identity.
                # Restore runtime exact-operation authority after the generic
                # credential boundary has deliberately stripped any provider-
                # supplied recovery claim.
                safe_failure = ModelStreamDeadlineError(
                    provider=safe_failure.provider,
                    evidence=safe_failure.deadline_evidence,
                    stream_cleanup_failed=safe_failure.stream_cleanup_failed,
                    recovery_disposition=EXACT_MODEL_STREAM_RECOVERY_DISPOSITION,
                )
            raise safe_failure from None
        finally:
            if not controller_handed_off:
                controller.close()

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
        controller: ProviderStreamDeadlineController,
        empty_message: str | None,
        exact_operation: bool = False,
    ) -> Mapping[str, Any]:
        token = bind_provider_deadline_controller(controller)
        try:
            return await controller.wait_for(
                anext(raw_events),
                kinds=(ProviderDeadlineKind.SEMANTIC_IDLE, ProviderDeadlineKind.ABSOLUTE),
            )
        except StopAsyncIteration:
            if empty_message is None:
                raise
            raise OpenAIProtocolError(
                empty_message, reason_code="background_stream_ended_early"
            ) from None
        except asyncio.CancelledError as exc:
            raise self._safe_cancellation(exc) from None
        except ProviderStreamDeadlineExceeded as exc:
            raise ModelStreamDeadlineError(
                provider=self._provider.name,
                evidence=exc.evidence,
                stream_cleanup_failed=exc.stream_cleanup_failed,
                recovery_disposition=(
                    EXACT_MODEL_STREAM_RECOVERY_DISPOSITION
                    if exact_operation
                    else MANUAL_MODEL_STREAM_RECOVERY_DISPOSITION
                ),
            ) from None
        except Exception as exc:
            raise self._safe_failure(exc) from None
        finally:
            reset_provider_deadline_controller(token)

    async def _events(
        self,
        raw_events: AsyncIterator[Mapping[str, Any]],
        *,
        state: ProviderOperationState,
        first: Mapping[str, Any] | None = None,
        controller: ProviderStreamDeadlineController,
        semantic_pause_started: float,
    ) -> AsyncIterator[ModelStreamEvent]:
        controller.exclude_idle_pause(
            semantic_pause_started,
            kinds=(ProviderDeadlineKind.SEMANTIC_IDLE,),
        )
        try:
            events = _openai_background_stream_events(
                raw_events,
                state=state,
                first=first,
                reasoning_state=self._provider.reasoning_state,
            )
            guarded = _guard_normalized_provider_stream(
                events,
                provider=self._provider.name,
                deadlines=self._provider.stream_deadlines,
                controller=controller,
            )
            async with (
                aclosing_provider_stream(raw_events),
                aclosing_provider_stream(events),
                aclosing_provider_stream(guarded),
            ):
                async for event in guarded:
                    yield event
        except ModelStreamDeadlineError as exc:
            raise ModelStreamDeadlineError(
                provider=exc.provider,
                evidence=exc.deadline_evidence,
                stream_cleanup_failed=exc.stream_cleanup_failed,
                recovery_disposition=EXACT_MODEL_STREAM_RECOVERY_DISPOSITION,
            ) from None
        except asyncio.CancelledError as exc:
            raise self._safe_cancellation(exc) from None
        except Exception as exc:
            raise self._safe_failure(exc) from None
        finally:
            controller.close()

    async def _close_raw_events(
        self,
        raw_events: AsyncIterator[Mapping[str, Any]],
        *,
        deadline_failure: bool = False,
    ) -> bool:
        try:
            return await _close_openai_operation_stream(
                raw_events,
                deadline_failure=deadline_failure,
            )
        except asyncio.CancelledError as exc:
            raise self._safe_cancellation(exc) from None

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
        if type(safe) is ModelStreamDeadlineError:
            return safe
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
    def stream_deadlines(self) -> ProviderStreamDeadlines:
        return self._stream_deadlines

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

    def supports_targeted_tool_projection(self, *, model: str, protocol: str) -> bool:
        require_clean_nonblank(model, "model")
        require_clean_nonblank(protocol, "protocol")
        return (
            protocol == OPENAI_ADDITIONAL_TOOLS_PROTOCOL and model in self.additional_tools_models
        )

    def supports_tool_discovery_projection(self, *, model: str, protocol: str) -> bool:
        require_clean_nonblank(model, "model")
        require_clean_nonblank(protocol, "protocol")
        if self.base_url != _validate_base_url(DEFAULT_OPENAI_BASE_URL):
            return False
        if protocol == OPENAI_CLIENT_TOOL_SEARCH_PROTOCOL:
            return model in self.client_tool_search_models
        if protocol == OPENAI_HOSTED_TOOL_SEARCH_PROTOCOL:
            return model in self.hosted_tool_search_models
        return False

    def preflight_tool_discovery_projection(self, *, model: str, protocol: str) -> None:
        model = require_clean_nonblank(model, "model")
        protocol = require_clean_nonblank(protocol, "protocol")
        if protocol not in {
            OPENAI_CLIENT_TOOL_SEARCH_PROTOCOL,
            OPENAI_HOSTED_TOOL_SEARCH_PROTOCOL,
        }:
            raise ValueError(f"OpenAIProvider does not support projection {protocol!r}.")
        if self.base_url != _validate_base_url(DEFAULT_OPENAI_BASE_URL):
            raise ValueError(
                "OpenAI Tool Search is established only for the official OpenAI Responses endpoint."
            )
        allowlist = (
            self.client_tool_search_models
            if protocol == OPENAI_CLIENT_TOOL_SEARCH_PROTOCOL
            else self.hosted_tool_search_models
        )
        if model not in allowlist:
            execution = "client" if protocol == OPENAI_CLIENT_TOOL_SEARCH_PROTOCOL else "hosted"
            field_name = (
                "client_tool_search_models"
                if protocol == OPENAI_CLIENT_TOOL_SEARCH_PROTOCOL
                else "hosted_tool_search_models"
            )
            raise ValueError(
                f"OpenAI {execution} Tool Search support is not established for model "
                f"{model!r}; list the exact verified model in {field_name}."
            )

    def preflight_targeted_tool_projection(self, *, model: str, protocol: str) -> None:
        model = require_clean_nonblank(model, "model")
        protocol = require_clean_nonblank(protocol, "protocol")
        if protocol != OPENAI_ADDITIONAL_TOOLS_PROTOCOL:
            raise ValueError(f"OpenAIProvider does not support projection {protocol!r}.")
        if model not in self.additional_tools_models:
            raise ValueError(
                "OpenAI additional_tools support is not established for model "
                f"{model!r}; list the exact verified model in additional_tools_models."
            )

    def _preflight_dynamic_tool_request(self, request: ModelRequest) -> None:
        if type(request) is not ModelRequest:
            raise TypeError("request must be a ModelRequest.")
        projection = request.targeted_tool_projection
        cache_anchor = targeted_tool_native_cache_anchor_name(request.options)
        if (
            call_tool_core_callable(request.options)
            and cache_anchor is None
            and request.tool_discovery_projection is None
        ):
            raise ValueError("Callable call_tool core requires a stable cache anchor.")
        if cache_anchor is not None:
            self.preflight_targeted_tool_projection(
                model=request.model,
                protocol=OPENAI_ADDITIONAL_TOOLS_PROTOCOL,
            )
        if projection is not None:
            if cache_anchor is None:
                raise ValueError(
                    "OpenAI additional_tools projection requires a stable cache anchor."
                )
            self.preflight_targeted_tool_projection(
                model=request.model,
                protocol=projection.protocol,
            )
        discovery_projection = request.tool_discovery_projection
        if discovery_projection is not None:
            self.preflight_tool_discovery_projection(
                model=request.model,
                protocol=discovery_projection.protocol,
            )
            if projection is not None and (
                set(discovery_projection.candidate_tool_names)
                & {cast("str", tool["name"]) for tool in projection.tools}
            ):
                raise ValueError(
                    "A hosted Tool Search candidate cannot also be an additional_tools function."
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
            _effective_openai_request_options_for_request(request)
        )
        return {"openai": projected} if projected else {}

    def request_fingerprint_options(self, request: ModelRequest) -> dict[str, Any]:
        effective = _effective_openai_request_options_for_request(request)
        return {"openai": effective} if effective else {}

    def __init__(
        self,
        *,
        api_key: str | None = None,
        name: str = "openai",
        base_url: str = DEFAULT_OPENAI_BASE_URL,
        timeout_s: float = DEFAULT_OPENAI_TIMEOUT_SECONDS,
        stream_deadlines: ProviderStreamDeadlines | None = None,
        transport: OpenAITransport | None = None,
        extra_headers: Mapping[str, str] | None = None,
        reasoning_state: str = "inline",
        hosted_web_search_supported: bool | None = None,
        additional_tools_models: Iterable[str] | None = None,
        client_tool_search_models: Iterable[str] | None = None,
        hosted_tool_search_models: Iterable[str] | None = None,
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
        self.additional_tools_models = _copy_exact_model_allowlist(
            additional_tools_models,
            field_name="additional_tools_models",
        )
        self.client_tool_search_models = _copy_exact_model_allowlist(
            client_tool_search_models,
            field_name="client_tool_search_models",
        )
        self.hosted_tool_search_models = _copy_exact_model_allowlist(
            hosted_tool_search_models,
            field_name="hosted_tool_search_models",
        )
        if type(background) is not bool:
            raise TypeError("background must be a bool.")
        if background and self.base_url != _validate_base_url(DEFAULT_OPENAI_BASE_URL):
            raise ValueError(
                "OpenAI background operations require the official OpenAI API base URL."
            )
        self.background = background
        self.timeout_s = positive_finite_seconds(timeout_s, "timeout_s")
        self._stream_deadlines = _resolve_provider_stream_deadlines(
            stream_deadlines=stream_deadlines,
        )
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

    @_terminal_preserving_provider_stream
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
            self._preflight_dynamic_tool_request(request)
            payload = build_openai_payload(
                request, stream=True, reasoning_state=self.reasoning_state
            )
            yielded_any = False
            try:
                events = self._consume(payload)
                async with aclosing_provider_stream(events):
                    async for event in events:
                        yielded_any = True
                        is_completion = event.type == ModelStreamEventType.COMPLETED
                        event = _event_with_server_dynamic_tool_ownership(
                            event,
                            reasoning_state=self.reasoning_state,
                            marker_id=(
                                None
                                if request.targeted_tool_projection is None
                                else request.targeted_tool_projection.marker_id
                            ),
                            discovery_loaded_tool_names=(
                                _openai_discovery_ownership_tokens(
                                    request.tool_discovery_projection
                                )
                            ),
                        )
                        completion_emitted = is_completion
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
                    is_completion = event.type == ModelStreamEventType.COMPLETED
                    event = _event_with_server_dynamic_tool_ownership(
                        event,
                        reasoning_state=self.reasoning_state,
                        marker_id=(
                            None
                            if request.targeted_tool_projection is None
                            else request.targeted_tool_projection.marker_id
                        ),
                        discovery_loaded_tool_names=(
                            _openai_discovery_ownership_tokens(request.tool_discovery_projection)
                        ),
                    )
                    completion_emitted = is_completion
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
        except OpenAIProtocolError as exc:
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
                # A parser/protocol failure is an unknown provider outcome, not
                # an application exception. Preserve that typed identity so the
                # runtime applies max_unknown_attempts rather than stopping at 1.
                protocol_event = credential_safe_error_event(
                    exc,
                    provider_label="OpenAI",
                    provider_name="openai",
                    credential_values=credential_values,
                    unresolved_message="OpenAIProtocolError: OpenAI provider failed",
                )
                protocol_payload = dict(protocol_event.payload)
                # Preserve the public protocol-error type while marking the
                # fixed, credential-free provider classification explicitly.
                # The runtime can then apply its bounded unknown retry policy.
                protocol_payload["provider_error_type"] = "protocol_error"
                protocol_payload.update(
                    protocol_diagnostic_fields(getattr(exc, "reason_code", None))
                )
                error_event = ModelStreamEvent(
                    type=protocol_event.type,
                    payload=protocol_payload,
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
        self._preflight_dynamic_tool_request(request)
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
            transport_idle_timeout_s=self.stream_deadlines.transport_idle_timeout_s,
            protocol_idle_timeout_s=self.stream_deadlines.protocol_idle_timeout_s,
            semantic_progress_timeout_s=self.stream_deadlines.semantic_progress_timeout_s,
            absolute_stream_timeout_s=self.stream_deadlines.absolute_stream_timeout_s,
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

    options = _effective_openai_request_options_for_request(request)
    discovery_projection = request.tool_discovery_projection
    if (
        discovery_projection is not None
        and discovery_projection.protocol == OPENAI_HOSTED_TOOL_SEARCH_PROTOCOL
        and discovery_projection.candidate_tools
    ):
        configured_parallel = options.get("parallel_tool_calls")
        if configured_parallel is not None and configured_parallel is not False:
            raise ValueError("OpenAI hosted Tool Search requires parallel_tool_calls=false.")
        options["parallel_tool_calls"] = False
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
        server_owned_count = len(request.messages) - len(messages_to_send)
        if previous_response_id is not None and _server_prefix_has_unsafe_dynamic_tools(
            request.messages[:server_owned_count],
            targeted_projection=request.targeted_tool_projection,
            discovery_projection=request.tool_discovery_projection,
        ):
            # A previous_response_id would keep an earlier interaction's
            # dynamic tool definitions alive on the provider. Rebuild neutrally
            # so authority absent from the current request cannot remain
            # model-addressable.
            previous_response_id = None
            messages_to_send = request.messages
            use_provider_state = False
    elif reasoning_state == "server" and not chain:
        use_provider_state = False  # recovery: rebuild from neutral parts

    _validate_targeted_tool_projection_marker(
        request.messages,
        request.targeted_tool_projection,
    )
    _validate_tool_search_replay(
        request.messages,
        request.tool_discovery_projection,
    )
    direct_tool_names = {name for tool in request.tools if type(name := tool.get("name")) is str}
    if request.targeted_tool_projection is not None and any(
        tool.get("name") in direct_tool_names for tool in request.targeted_tool_projection.tools
    ):
        raise ValueError(
            "A targeted additional_tools function cannot also be a direct request tool."
        )

    input_items: list[dict[str, Any]] = []
    for message in messages_to_send:
        input_items.extend(
            _openai_input_items(
                message,
                resolved_attachments=resolved_attachments,
                reasoning_state=reasoning_state,
                use_provider_state=use_provider_state,
                targeted_tool_projection=request.targeted_tool_projection,
                tool_discovery_projection=request.tool_discovery_projection,
            )
        )
    if not input_items:
        raise ValueError("OpenAI requests require at least one non-system input item.")
    payload["input"] = input_items
    if previous_response_id is not None:
        payload["previous_response_id"] = previous_response_id

    tools = _openai_request_tools(request)
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
        raise OpenAIProtocolError(
            "OpenAI embedding response must be a JSON object.",
            reason_code="embedding_response_must_be_a_json_object",
        )
    object_type = response.get("object")
    if object_type != "list":
        raise OpenAIProtocolError(
            "OpenAI embedding response has unexpected object.",
            reason_code="embedding_response_has_unexpected_object",
        )
    model = response.get("model")
    if type(model) is not str:
        raise OpenAIProtocolError(
            "OpenAI embedding response requires model.",
            reason_code="embedding_response_requires_model",
        )
    data = response.get("data")
    if not isinstance(data, list):
        raise OpenAIProtocolError(
            "OpenAI embedding response data must be a list.",
            reason_code="embedding_response_data_must_be_a_list",
        )
    if len(data) != requested_count:
        raise OpenAIProtocolError(
            "OpenAI embedding response count did not match request.",
            reason_code="embedding_response_count_did_not_match_request",
        )
    embeddings: list[TextEmbedding] = []
    for position, item in enumerate(data):
        if not isinstance(item, Mapping):
            raise OpenAIProtocolError(
                f"OpenAI embedding item {position} must be an object.",
                reason_code="embedding_item_must_be_an_object",
            )
        item_data = cast("Mapping[str, Any]", item)
        index = item_data.get("index")
        vector = item_data.get("embedding")
        if type(index) is not int:
            raise OpenAIProtocolError(
                f"OpenAI embedding item {position} requires index.",
                reason_code="embedding_item_requires_index",
            )
        if not isinstance(vector, list):
            raise OpenAIProtocolError(
                f"OpenAI embedding item {position} requires vector.",
                reason_code="embedding_item_requires_vector",
            )
        vector_numbers: list[float] = []
        for vector_index, vector_item in enumerate(vector):
            if isinstance(vector_item, bool) or not isinstance(vector_item, int | float):
                raise OpenAIProtocolError(
                    f"OpenAI embedding item {position} vector[{vector_index}] must be a number.",
                    reason_code="embedding_item_vector_must_be_a_number",
                )
            vector_numbers.append(float(vector_item))
        embeddings.append(TextEmbedding(index=index, vector=vector_numbers))
    embeddings.sort(key=lambda embedding: embedding.index)
    if [embedding.index for embedding in embeddings] != list(range(requested_count)):
        raise OpenAIProtocolError(
            "OpenAI embedding response indexes did not match request.",
            reason_code="embedding_response_indexes_did_not_match_request",
        )
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
        raise OpenAIProtocolError(
            "OpenAI embedding usage must be an object.",
            reason_code="embedding_usage_must_be_an_object",
        )
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
        raise OpenAIProtocolError(
            f"OpenAI embedding usage requires nonnegative {key}.",
            reason_code="embedding_usage_requires_nonnegative",
        )
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
        raise OpenAIProtocolError(
            "OpenAI response must be a JSON object.", reason_code="response_must_be_a_json_object"
        )

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
        raise OpenAIProtocolError(
            "OpenAI response output must be a list.", reason_code="response_output_must_be_a_list"
        )

    hosted_items, hosted_result = _normalized_hosted_tool_search_items(output)
    events: list[ModelStreamEvent] = []
    provider_state_items: list[dict[str, Any]] = []
    completion_output_items: list[Mapping[str, Any]] = []
    hosted_call_indexes: dict[str, int] = {}
    tool_search_call_count = 0
    assistant_text_offset = 0
    response_status = _optional_string(response, "status")
    for index, item in enumerate(output):
        if not isinstance(item, Mapping):
            raise OpenAIProtocolError(
                f"OpenAI output item {index} must be an object.",
                reason_code="output_item_must_be_an_object",
            )
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
        elif item_type == "tool_search_call":
            tool_search_call_count += 1
            if tool_search_call_count > _OPENAI_CLIENT_TOOL_SEARCH_MAX_ITEMS:
                raise OpenAIProtocolError(
                    "OpenAI response contains too many tool search calls.",
                    reason_code="response_contains_too_many_tool_search_calls",
                )
            normalized = _normalized_tool_search_call(item, item_index=index)
            if normalized["execution"] == "client":
                events.append(_tool_search_call_event(normalized))
            completion_output_items.append(normalized)
            provider_state_items.append({"provider": "openai", "state": normalized})
        elif item_type == "tool_search_output":
            normalized = hosted_items.get(index)
            if normalized is None:
                raise OpenAIProtocolError(
                    "OpenAI tool_search_output has no adjacent hosted search call.",
                    reason_code="tool_search_output_has_no_adjacent_hosted_search_call",
                )
            completion_output_items.append(normalized)
            provider_state_items.append({"provider": "openai", "state": normalized})
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
                        "OpenAI terminal response contains a nonterminal web search call.",
                        reason_code="terminal_response_contains_a_nonterminal_web_search_call",
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
            raise OpenAIProtocolError(
                f"Unsupported OpenAI output item type: {item_type!r}.",
                reason_code="unsupported_openai_output_item_type",
            )

    events.append(
        _completed_event_from_response(
            response,
            provider_state_items,
            completion_output_items=completion_output_items,
            reasoning_state=reasoning_state,
            tool_discovery_result=hosted_result,
        )
    )
    return events


def _openai_operation_url(base_url: str, operation_id: str) -> str:
    return f"{base_url}/v1/responses/{quote(operation_id, safe='')}"


def _openai_recovery_sequence_number(metadata: ProviderOperationRecoveryMetadata) -> int:
    value = metadata.opaque.get("sequence_number")
    if type(value) is not int or value < 0 or value > MAX_DURABLE_JSON_INTEGER:
        raise OpenAIProtocolError(
            "OpenAI background recovery metadata requires a nonnegative sequence_number.",
            reason_code="background_recovery_metadata_requires_a_nonnegative_sequence_number",
        )
    return value


def _openai_background_targeted_tool_marker_id(
    metadata: ProviderOperationRecoveryMetadata,
) -> str | None:
    value = metadata.opaque.get("targeted_tool_marker_id")
    if value is None:
        return None
    if (
        type(value) is not str
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise OpenAIProtocolError(
            "OpenAI background recovery metadata has an invalid targeted-tool marker id.",
            reason_code="background_recovery_metadata_has_an_invalid_targeted_tool_marker_id",
        )
    return value


def _openai_background_discovery_loaded_tool_names(
    metadata: ProviderOperationRecoveryMetadata,
) -> tuple[str, ...] | None:
    value = metadata.opaque.get("tool_discovery_loaded_tool_names")
    if value is None:
        return None
    if type(value) is not list:
        raise OpenAIProtocolError(
            "OpenAI background recovery metadata has invalid discovery tool names.",
            reason_code="background_recovery_metadata_has_invalid_discovery_tool_names",
        )
    if len(value) > _OPENAI_CLIENT_TOOL_SEARCH_MAX_ITEMS or any(
        type(name) is not str for name in value
    ):
        raise OpenAIProtocolError(
            "OpenAI background recovery metadata has invalid discovery tool names.",
            reason_code="background_recovery_metadata_has_invalid_discovery_tool_names",
        )
    try:
        names = tuple(
            require_clean_nonblank(name, "background discovery tool name")
            for name in cast("list[str]", value)
        )
    except (TypeError, ValueError):
        raise OpenAIProtocolError(
            "OpenAI background recovery metadata has invalid discovery tool names.",
            reason_code="background_recovery_metadata_has_invalid_discovery_tool_names",
        ) from None
    if names != tuple(sorted(set(names))):
        raise OpenAIProtocolError(
            "OpenAI background recovery metadata has invalid discovery tool names.",
            reason_code="background_recovery_metadata_has_invalid_discovery_tool_names",
        )
    return names


def _require_openai_background_state(state: ProviderOperationState) -> ProviderOperationState:
    state = copy_provider_operation_state(state)
    if state.stream_protocol != _OPENAI_BACKGROUND_STREAM_PROTOCOL:
        raise OpenAIProtocolError(
            "OpenAI background operation uses an unknown stream protocol.",
            reason_code="background_operation_uses_an_unknown_stream_protocol",
        )
    _openai_recovery_sequence_number(state.recovery_metadata)
    _openai_background_targeted_tool_marker_id(state.recovery_metadata)
    _openai_background_discovery_loaded_tool_names(state.recovery_metadata)
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
    *,
    targeted_tool_marker_id: str | None,
    discovery_loaded_tool_names: tuple[str, ...] | None,
) -> tuple[ProviderOperationState, ProviderOperationStatus]:
    if not isinstance(event, Mapping) or event.get("type") != "response.created":
        raise OpenAIProtocolError(
            "OpenAI background start must begin with response.created.",
            reason_code="background_start_must_begin_with_response_created",
        )
    sequence_number = _openai_stream_sequence_number(event)
    response = _stream_response_object(event)
    response_id = response.get("id")
    if type(response_id) is not str or not response_id.strip():
        raise OpenAIProtocolError(
            "OpenAI response.created requires a nonblank response id.",
            reason_code="response_created_requires_a_nonblank_response_id",
        )
    status = _openai_response_operation_status(response.get("status"))
    if status not in {ProviderOperationStatus.QUEUED, ProviderOperationStatus.IN_PROGRESS}:
        raise OpenAIProtocolError(
            "OpenAI response.created requires queued or in_progress status.",
            reason_code="response_created_requires_queued_or_in_progress_status",
        )
    return (
        ProviderOperationState(
            operation_id=response_id,
            stream_protocol=_OPENAI_BACKGROUND_STREAM_PROTOCOL,
            recovery_metadata=ProviderOperationRecoveryMetadata.model_validate(
                {
                    "cursor": 0,
                    "opaque": {
                        "sequence_number": sequence_number,
                        "targeted_tool_marker_id": targeted_tool_marker_id,
                        **(
                            {"tool_discovery_loaded_tool_names": list(discovery_loaded_tool_names)}
                            if discovery_loaded_tool_names is not None
                            else {}
                        ),
                    },
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
    targeted_tool_marker_id = _openai_background_targeted_tool_marker_id(state.recovery_metadata)
    discovery_loaded_tool_names = _openai_background_discovery_loaded_tool_names(
        state.recovery_metadata
    )
    events: list[ModelStreamEvent] = []
    for event in parsed:
        cursor += 1
        event = _event_with_server_dynamic_tool_ownership(
            event,
            reasoning_state=reasoning_state,
            marker_id=targeted_tool_marker_id,
            discovery_loaded_tool_names=discovery_loaded_tool_names,
        )
        events.append(
            event.model_copy(
                update={
                    "recovery_metadata": ProviderOperationRecoveryMetadata(
                        cursor=cursor,
                        opaque={
                            "sequence_number": _openai_recovery_sequence_number(
                                state.recovery_metadata
                            ),
                            "targeted_tool_marker_id": targeted_tool_marker_id,
                            **(
                                {
                                    "tool_discovery_loaded_tool_names": list(
                                        discovery_loaded_tool_names
                                    )
                                }
                                if discovery_loaded_tool_names is not None
                                else {}
                            ),
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
            "OpenAI background stream events require a nonnegative sequence_number.",
            reason_code="background_stream_events_require_a_nonnegative_sequence_number",
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
) -> tuple[
    dict[int, _PendingFunctionCall],
    dict[int, tuple[str, str] | None],
    set[int],
    dict[int, dict[str, Any]],
    dict[int, dict[str, Any]],
    dict[int, str],
]:
    parser = metadata.opaque.get("parser")
    if parser is None:
        return {}, {}, set(), {}, {}, {}
    if type(parser) is not dict:
        raise OpenAIProtocolError(
            "OpenAI recovery parser state must be an object.",
            reason_code="recovery_parser_state_must_be_an_object",
        )
    parser = cast("dict[str, Any]", parser)
    raw_calls = parser.get("pending_function_calls", [])
    has_reasoning_indexes = "pending_reasoning_output_indexes" in parser
    raw_reasoning_indexes = parser.get("pending_reasoning_output_indexes", [])
    raw_reasoning_items = parser.get("pending_reasoning_items", [])
    raw_completed_reasoning = parser.get("completed_reasoning_output_indexes", [])
    raw_tool_search = parser.get("pending_tool_search_calls", [])
    raw_completed_tool_search = parser.get("completed_tool_search_items", [])
    raw_completed_function_calls = parser.get("completed_function_call_digests", [])
    if (
        type(raw_calls) is not list
        or type(raw_reasoning_indexes) is not list
        or type(raw_reasoning_items) is not list
        or type(raw_completed_reasoning) is not list
        or type(raw_tool_search) is not list
        or type(raw_completed_tool_search) is not list
        or type(raw_completed_function_calls) is not list
        or len(raw_tool_search) + len(raw_completed_tool_search)
        > _OPENAI_CLIENT_TOOL_SEARCH_MAX_ITEMS
        or len(raw_completed_function_calls) > _OPENAI_CLIENT_TOOL_SEARCH_MAX_ITEMS
    ):
        raise OpenAIProtocolError(
            "OpenAI recovery parser state is malformed.",
            reason_code="recovery_parser_state_is_malformed",
        )
    calls: dict[int, _PendingFunctionCall] = {}
    for raw in raw_calls:
        if type(raw) is not dict:
            raise OpenAIProtocolError(
                "OpenAI pending function-call state is malformed.",
                reason_code="pending_function_call_state_is_malformed",
            )
        raw = cast("dict[str, Any]", raw)
        output_index = raw.get("output_index")
        if type(output_index) is not int or output_index < 0 or output_index in calls:
            raise OpenAIProtocolError(
                "OpenAI pending function-call index is malformed.",
                reason_code="pending_function_call_index_is_malformed",
            )
        calls[output_index] = _PendingFunctionCall(
            item_id=_mapping_optional_string(raw, "item_id"),
            call_id=_mapping_optional_string(raw, "call_id"),
            name=_mapping_optional_string(raw, "name"),
            arguments="",
        )
    reasoning: dict[int, tuple[str, str] | None] = {}
    for raw_index in raw_reasoning_indexes:
        if type(raw_index) is not int or raw_index < 0 or raw_index in reasoning:
            raise OpenAIProtocolError(
                "OpenAI pending reasoning index is malformed.",
                reason_code="pending_reasoning_index_is_malformed",
            )
        reasoning[raw_index] = None
    for raw in raw_reasoning_items:
        if type(raw) is not dict:
            raise OpenAIProtocolError(
                "OpenAI pending reasoning state is malformed.",
                reason_code="pending_reasoning_state_is_malformed",
            )
        raw = cast("dict[str, Any]", raw)
        output_index = raw.get("output_index")
        item_id = _mapping_optional_string(raw, "item_id")
        if (
            type(output_index) is not int
            or output_index < 0
            or item_id is None
            or (has_reasoning_indexes and output_index not in reasoning)
            or reasoning.get(output_index) is not None
        ):
            raise OpenAIProtocolError(
                "OpenAI pending reasoning state is malformed.",
                reason_code="pending_reasoning_state_is_malformed",
            )
        reasoning[output_index] = ("reasoning", item_id)
    completed_reasoning: set[int] = set()
    for raw_index in raw_completed_reasoning:
        if (
            type(raw_index) is not int
            or raw_index < 0
            or raw_index in reasoning
            or raw_index in completed_reasoning
        ):
            raise OpenAIProtocolError(
                "OpenAI completed reasoning index is malformed.",
                reason_code="completed_reasoning_index_is_malformed",
            )
        completed_reasoning.add(raw_index)
    tool_search: dict[int, dict[str, Any]] = {}
    for raw in raw_tool_search:
        if type(raw) is not dict:
            raise OpenAIProtocolError(
                "OpenAI pending tool-search state is malformed.",
                reason_code="pending_tool_search_state_is_malformed",
            )
        output_index = raw.get("output_index")
        execution = raw.get("execution")
        item_id = _mapping_optional_string(raw, "item_id")
        call_id = _mapping_optional_string(raw, "call_id")
        if (
            type(output_index) is not int
            or output_index < 0
            or output_index in tool_search
            or execution not in {"client", "server"}
            or (execution == "client" and (item_id is None or call_id is None))
            or (execution == "server" and call_id is not None)
        ):
            raise OpenAIProtocolError(
                "OpenAI pending tool-search state is malformed.",
                reason_code="pending_tool_search_state_is_malformed",
            )
        tool_search[output_index] = {
            "execution": execution,
            "id": item_id,
            "call_id": call_id,
        }
    completed_tool_search: dict[int, dict[str, Any]] = {}
    for raw in raw_completed_tool_search:
        if type(raw) is not dict:
            raise OpenAIProtocolError(
                "OpenAI completed tool-search state is malformed.",
                reason_code="completed_tool_search_state_is_malformed",
            )
        output_index = raw.get("output_index")
        item = raw.get("item")
        if (
            type(output_index) is not int
            or output_index < 0
            or output_index in completed_tool_search
            or output_index in tool_search
            or not isinstance(item, Mapping)
        ):
            raise OpenAIProtocolError(
                "OpenAI completed tool-search state is malformed.",
                reason_code="completed_tool_search_state_is_malformed",
            )
        item_type = item.get("type")
        if item_type == "tool_search_call":
            completed_tool_search[output_index] = _normalized_tool_search_call(
                item,
                item_index=output_index,
            )
        elif item_type == "tool_search_output":
            completed_tool_search[output_index], _loaded_tools = (
                _normalized_hosted_tool_search_output(
                    item,
                    item_index=output_index,
                )
            )
        else:
            raise OpenAIProtocolError(
                "OpenAI completed tool-search state is malformed.",
                reason_code="completed_tool_search_state_is_malformed",
            )
    hosted_call_indexes = {
        output_index
        for output_index, item in completed_tool_search.items()
        if item.get("type") == "tool_search_call" and item.get("execution") == "server"
    } | {
        output_index for output_index, item in tool_search.items() if item["execution"] == "server"
    }
    hosted_output_indexes = {
        output_index
        for output_index, item in completed_tool_search.items()
        if item.get("type") == "tool_search_output"
    }
    if (
        len(hosted_call_indexes) > 1
        or len(hosted_output_indexes) > 1
        or (
            hosted_output_indexes
            and next(iter(hosted_output_indexes)) - 1 not in hosted_call_indexes
        )
    ):
        raise OpenAIProtocolError(
            "OpenAI hosted Tool Search recovery state is malformed.",
            reason_code="hosted_tool_search_recovery_state_is_malformed",
        )
    completed_function_calls: dict[int, str] = {}
    for raw in raw_completed_function_calls:
        if type(raw) is not dict:
            raise OpenAIProtocolError(
                "OpenAI completed function-call state is malformed.",
                reason_code="completed_function_call_state_is_malformed",
            )
        output_index = raw.get("output_index")
        item_sha256 = raw.get("item_sha256")
        if (
            type(output_index) is not int
            or output_index < 0
            or output_index in completed_function_calls
            or type(item_sha256) is not str
            or len(item_sha256) != 64
            or any(character not in "0123456789abcdef" for character in item_sha256)
        ):
            raise OpenAIProtocolError(
                "OpenAI completed function-call state is malformed.",
                reason_code="completed_function_call_state_is_malformed",
            )
        completed_function_calls[output_index] = item_sha256
    return (
        calls,
        reasoning,
        completed_reasoning,
        tool_search,
        completed_tool_search,
        completed_function_calls,
    )


def _openai_background_recovery_metadata(
    *,
    cursor: int,
    sequence_number: int,
    pending_function_calls: Mapping[int, _PendingFunctionCall],
    pending_reasoning_items: Mapping[int, tuple[str, str] | None],
    completed_reasoning_items: set[int],
    pending_tool_search_calls: Mapping[int, Mapping[str, Any]],
    completed_tool_search_items: Mapping[int, Mapping[str, Any]],
    completed_function_call_digests: Mapping[int, str],
    targeted_tool_marker_id: str | None,
    discovery_loaded_tool_names: tuple[str, ...] | None,
) -> ProviderOperationRecoveryMetadata:
    opaque: dict[str, object] = {
        "sequence_number": sequence_number,
        "targeted_tool_marker_id": targeted_tool_marker_id,
        **(
            {"tool_discovery_loaded_tool_names": list(discovery_loaded_tool_names)}
            if discovery_loaded_tool_names is not None
            else {}
        ),
    }
    if (
        pending_function_calls
        or pending_reasoning_items
        or completed_reasoning_items
        or pending_tool_search_calls
        or completed_tool_search_items
        or completed_function_call_digests
    ):
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
            "pending_reasoning_items": [
                {
                    "output_index": output_index,
                    "item_id": pending[1],
                }
                for output_index, pending in sorted(pending_reasoning_items.items())
                if pending is not None
            ],
            "completed_reasoning_output_indexes": sorted(completed_reasoning_items),
            "pending_tool_search_calls": [
                {
                    "output_index": output_index,
                    "execution": pending["execution"],
                    "item_id": pending["id"],
                    "call_id": pending["call_id"],
                }
                for output_index, pending in sorted(pending_tool_search_calls.items())
            ],
            "completed_tool_search_items": [
                {
                    "output_index": output_index,
                    "item": copy_json_value(item, "completed tool_search item"),
                }
                for output_index, item in sorted(completed_tool_search_items.items())
            ],
            "completed_function_call_digests": [
                {
                    "output_index": output_index,
                    "item_sha256": item_sha256,
                }
                for output_index, item_sha256 in sorted(completed_function_call_digests.items())
            ],
        }
    return ProviderOperationRecoveryMetadata(cursor=cursor, opaque=opaque)


def _openai_background_event_with_recovery(
    event: ModelStreamEvent,
    *,
    cursor: int,
    sequence_number: int,
    pending_function_calls: Mapping[int, _PendingFunctionCall],
    pending_reasoning_items: Mapping[int, tuple[str, str] | None],
    completed_reasoning_items: set[int],
    pending_tool_search_calls: Mapping[int, Mapping[str, Any]],
    completed_tool_search_items: Mapping[int, Mapping[str, Any]],
    completed_function_call_digests: Mapping[int, str],
    targeted_tool_marker_id: str | None,
    discovery_loaded_tool_names: tuple[str, ...] | None,
) -> ModelStreamEvent:
    return event.model_copy(
        update={
            "recovery_metadata": _openai_background_recovery_metadata(
                cursor=cursor,
                sequence_number=sequence_number,
                pending_function_calls=pending_function_calls,
                pending_reasoning_items=pending_reasoning_items,
                completed_reasoning_items=completed_reasoning_items,
                pending_tool_search_calls=pending_tool_search_calls,
                completed_tool_search_items=completed_tool_search_items,
                completed_function_call_digests=completed_function_call_digests,
                targeted_tool_marker_id=targeted_tool_marker_id,
                discovery_loaded_tool_names=discovery_loaded_tool_names,
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
    targeted_tool_marker_id = _openai_background_targeted_tool_marker_id(state.recovery_metadata)
    discovery_loaded_tool_names = _openai_background_discovery_loaded_tool_names(
        state.recovery_metadata
    )
    (
        pending_function_calls,
        pending_reasoning_items,
        completed_reasoning_items,
        pending_tool_search_calls,
        completed_tool_search_items,
        completed_function_call_digests,
    ) = _openai_background_parser_state(state.recovery_metadata)

    async def ordered_raw_events() -> AsyncIterator[Mapping[str, Any]]:
        if first is not None:
            yield first
        async for raw in raw_events:
            yield raw

    async for event in ordered_raw_events():
        if not isinstance(event, Mapping):
            raise OpenAIProtocolError(
                "OpenAI stream event must be a JSON object.",
                reason_code="stream_event_must_be_a_json_object",
            )
        sequence_number = _openai_stream_sequence_number(event)
        if sequence_number <= last_sequence_number:
            raise OpenAIProtocolError(
                "OpenAI background sequence_number did not advance.",
                reason_code="background_sequence_number_did_not_advance",
            )
        last_sequence_number = sequence_number
        cursor += 1
        event_type = event.get("type")
        normalized: ModelStreamEvent
        if event_type in {"response.output_text.delta", "response.refusal.delta"}:
            delta = event.get("delta")
            if not isinstance(delta, str):
                raise OpenAIProtocolError(
                    "OpenAI text delta must be a string.", reason_code="text_delta_must_be_a_string"
                )
            normalized = (
                ModelStreamEvent.text_delta(delta) if delta else ModelStreamEvent.thinking()
            )
        elif event_type in {
            "response.reasoning_summary_text.delta",
            "response.reasoning_text.delta",
        }:
            delta = event.get("delta")
            if not isinstance(delta, str):
                raise OpenAIProtocolError(
                    "OpenAI reasoning delta must be a string.",
                    reason_code="reasoning_delta_must_be_a_string",
                )
            normalized = ModelStreamEvent.thinking(delta)
        elif event_type == "response.output_item.added":
            item = event.get("item")
            reasoning_added = False
            if isinstance(item, Mapping) and item.get("type") == "reasoning":
                output_index = _stream_output_index(event)
                if (
                    output_index in pending_reasoning_items
                    or output_index in completed_reasoning_items
                ):
                    raise OpenAIProtocolError(
                        "OpenAI background reasoning output_item.added was repeated.",
                        reason_code="background_reasoning_output_item_added_was_repeated",
                    )
                item_id = _mapping_optional_string(item, "id")
                if item_id is None:
                    raise OpenAIProtocolError(
                        "OpenAI reasoning output_item.added requires nonblank id.",
                        reason_code="reasoning_output_item_added_requires_nonblank_id",
                    )
                if item.get("status") not in {None, "in_progress", "incomplete"}:
                    raise OpenAIProtocolError(
                        "OpenAI reasoning output_item.added has invalid lifecycle status.",
                        reason_code="reasoning_output_item_added_has_invalid_lifecycle_status",
                    )
                _validate_stream_reasoning_shape(item, output_index)
                pending_reasoning_items[output_index] = ("reasoning", item_id)
                reasoning_added = True
            if isinstance(item, Mapping) and item.get("type") == "function_call":
                output_index = _stream_output_index(event)
                if (
                    output_index in pending_function_calls
                    or output_index in completed_function_call_digests
                    or len(pending_function_calls) + len(completed_function_call_digests)
                    >= _OPENAI_CLIENT_TOOL_SEARCH_MAX_ITEMS
                ):
                    raise OpenAIProtocolError(
                        "OpenAI background function_call added item is malformed.",
                        reason_code="background_function_call_added_item_is_malformed",
                    )
            if isinstance(item, Mapping) and item.get("type") == "tool_search_call":
                output_index = _stream_output_index(event)
                execution = item.get("execution")
                item_id = _mapping_optional_string(item, "id")
                call_id = _mapping_optional_string(item, "call_id")
                if (
                    output_index in pending_tool_search_calls
                    or output_index in completed_tool_search_items
                    or len(pending_tool_search_calls) + len(completed_tool_search_items)
                    >= _OPENAI_CLIENT_TOOL_SEARCH_MAX_ITEMS
                    or execution not in {"client", "server"}
                    or (execution == "client" and (item_id is None or call_id is None))
                    or (execution == "server" and ("call_id" not in item or call_id is not None))
                    or item.get("status") != "in_progress"
                    or (
                        execution == "server"
                        and any(
                            pending["execution"] == "server"
                            for pending in pending_tool_search_calls.values()
                        )
                    )
                    or (
                        execution == "server"
                        and any(
                            completed.get("type") == "tool_search_call"
                            and completed.get("execution") == "server"
                            for completed in completed_tool_search_items.values()
                        )
                    )
                ):
                    raise OpenAIProtocolError(
                        "OpenAI background tool_search_call added item is malformed.",
                        reason_code="background_tool_search_call_added_item_is_malformed",
                    )
                pending_tool_search_calls[output_index] = {
                    "execution": execution,
                    "id": item_id,
                    "call_id": call_id,
                }
            function_call_added = _record_stream_output_item_added(
                event,
                pending_function_calls,
            )
            if reasoning_added:
                observe_provider_semantic_progress(ProviderProgressKind.REASONING)
            if function_call_added:
                observe_provider_semantic_progress(ProviderProgressKind.TOOL_CALL)
            elif isinstance(item, Mapping) and item.get("type") == "tool_search_call":
                observe_provider_semantic_progress(
                    ProviderProgressKind.TOOL_CALL
                    if item.get("execution") == "client"
                    else ProviderProgressKind.HOSTED_TOOL
                )
            normalized = ModelStreamEvent.thinking()
        elif event_type == "response.output_item.done":
            item = event.get("item")
            if not isinstance(item, Mapping):
                raise OpenAIProtocolError(
                    "OpenAI output_item.done requires item object.",
                    reason_code="output_item_done_requires_item_object",
                )
            reasoning_completed = False
            if item.get("type") == "reasoning":
                output_index = _stream_output_index(event)
                if output_index in completed_reasoning_items:
                    raise OpenAIProtocolError(
                        "OpenAI background reasoning output_item.done was repeated.",
                        reason_code="background_reasoning_output_item_done_was_repeated",
                    )
                _validate_completed_stream_reasoning(item, output_index)
                pending = pending_reasoning_items.get(output_index)
                item_id = _mapping_optional_string(item, "id")
                if pending is not None and item_id != pending[1]:
                    raise OpenAIProtocolError(
                        "OpenAI background reasoning output_item.done identity conflicts "
                        "with added item.",
                        reason_code="background_reasoning_output_item_done_identity_conflicts_with_added_item",
                    )
                pending_reasoning_items.pop(output_index, None)
                completed_reasoning_items.add(output_index)
                reasoning_completed = True
            if item.get("type") == "tool_search_call":
                output_index = _stream_output_index(event)
                pending = pending_tool_search_calls.pop(output_index, None)
                tool_search = _normalized_tool_search_call(item, item_index=output_index)
                if pending != {
                    "execution": tool_search["execution"],
                    "id": tool_search.get("id"),
                    "call_id": tool_search.get("call_id"),
                }:
                    raise OpenAIProtocolError(
                        "OpenAI background tool_search_call output identity mismatch.",
                        reason_code="background_tool_search_call_output_identity_mismatch",
                    )
                completed_tool_search_items[output_index] = tool_search
                normalized = (
                    _tool_search_call_event(tool_search)
                    if tool_search["execution"] == "client"
                    else ModelStreamEvent.thinking()
                )
            elif item.get("type") == "tool_search_output":
                output_index = _stream_output_index(event)
                hosted_call = completed_tool_search_items.get(output_index - 1)
                if (
                    output_index in completed_tool_search_items
                    or hosted_call is None
                    or hosted_call.get("type") != "tool_search_call"
                    or hosted_call.get("execution") != "server"
                    or any(
                        completed.get("type") == "tool_search_output"
                        for completed in completed_tool_search_items.values()
                    )
                ):
                    raise OpenAIProtocolError(
                        "OpenAI background tool_search_output has no unique adjacent server call.",
                        reason_code="background_tool_search_output_has_no_unique_adjacent_server_call",
                    )
                tool_search_output, _loaded_tools = _normalized_hosted_tool_search_output(
                    item,
                    item_index=output_index,
                )
                completed_tool_search_items[output_index] = tool_search_output
                observe_provider_semantic_progress(ProviderProgressKind.HOSTED_TOOL)
                normalized = ModelStreamEvent.thinking()
            else:
                normalized = ModelStreamEvent.thinking()
            if reasoning_completed:
                observe_provider_semantic_progress(ProviderProgressKind.REASONING)
        elif event_type == "response.function_call_arguments.delta":
            _record_stream_function_call_delta(event, pending_function_calls)
            if event.get("delta"):
                observe_provider_semantic_progress(ProviderProgressKind.TOOL_CALL)
            normalized = ModelStreamEvent.thinking()
        elif event_type == "response.function_call_arguments.done":
            normalized, output_item = _stream_function_call_event(
                event,
                pending_function_calls,
            )
            completed_function_call_digests[_stream_output_index(event)] = (
                _openai_function_call_recovery_digest(
                    output_item,
                    item_index=_stream_output_index(event),
                )
            )
        elif event_type in {"response.completed", "response.incomplete"}:
            unfinished = {
                *pending_function_calls,
                *pending_reasoning_items,
                *pending_tool_search_calls,
            }
            if event_type == "response.completed" and unfinished:
                raise OpenAIProtocolError(
                    "OpenAI background response completed with unfinished output items.",
                    reason_code="background_response_completed_with_unfinished_output_items",
                )
            for terminal_event in _stream_terminal_events(
                event,
                completed_tool_search_items,
                excluded_output_indexes=unfinished,
                reasoning_state=reasoning_state,
                emitted_function_call_digests=completed_function_call_digests,
            ):
                terminal_event = _event_with_server_dynamic_tool_ownership(
                    terminal_event,
                    reasoning_state=reasoning_state,
                    marker_id=targeted_tool_marker_id,
                    discovery_loaded_tool_names=discovery_loaded_tool_names,
                )
                yield _openai_background_event_with_recovery(
                    terminal_event,
                    cursor=cursor,
                    sequence_number=sequence_number,
                    pending_function_calls=pending_function_calls,
                    pending_reasoning_items=pending_reasoning_items,
                    completed_reasoning_items=completed_reasoning_items,
                    pending_tool_search_calls=pending_tool_search_calls,
                    completed_tool_search_items=completed_tool_search_items,
                    completed_function_call_digests=completed_function_call_digests,
                    targeted_tool_marker_id=targeted_tool_marker_id,
                    discovery_loaded_tool_names=discovery_loaded_tool_names,
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
            completed_reasoning_items=completed_reasoning_items,
            pending_tool_search_calls=pending_tool_search_calls,
            completed_tool_search_items=completed_tool_search_items,
            completed_function_call_digests=completed_function_call_digests,
            targeted_tool_marker_id=targeted_tool_marker_id,
            discovery_loaded_tool_names=discovery_loaded_tool_names,
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
    *,
    deadline_failure: bool = False,
) -> bool:
    """Close one pre-publication operation stream and report bounded failure."""

    if deadline_failure:
        return await close_provider_stream_after_deadline(raw_events)
    close = getattr(raw_events, "aclose", None)
    if close is None:
        return False
    task = asyncio.current_task()
    cancellation_baseline = 0 if task is None else task.cancelling()
    try:
        await close()
    except asyncio.CancelledError:
        if task is not None and task.cancelling() > cancellation_baseline:
            raise
        return True
    except (GeneratorExit, KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        return True
    return False


def _openai_operation_failure_after_close(
    failure: Exception,
    *,
    cleanup_failed: bool,
) -> Exception:
    if type(failure) is not ModelStreamDeadlineError or not cleanup_failed:
        return failure
    return ModelStreamDeadlineError(
        provider=failure.provider,
        evidence=failure.deadline_evidence,
        stream_cleanup_failed=True,
    )


def _openai_input_tokens_from_count_response(response: Mapping[str, Any]) -> int:
    if not isinstance(response, Mapping):
        raise OpenAIProtocolError(
            "OpenAI input token count response must be a JSON object.",
            reason_code="input_token_count_response_must_be_a_json_object",
        )
    object_type = response.get("object")
    if object_type != "response.input_tokens":
        raise OpenAIProtocolError(
            "OpenAI input token count response has unexpected object.",
            reason_code="input_token_count_response_has_unexpected_object",
        )
    input_tokens = response.get("input_tokens")
    if type(input_tokens) is not int or input_tokens < 0:
        raise OpenAIProtocolError(
            "OpenAI input token count response requires input_tokens.",
            reason_code="input_token_count_response_requires_input_tokens",
        )
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
                            "OpenAI hosted search event requires a call identity.",
                            reason_code="hosted_search_event_requires_a_call_identity",
                        )
                    if status in {"in_progress", "searching"}:
                        if status == "in_progress":
                            if call_id in seen_call_ids:
                                raise OpenAIProtocolError(
                                    "OpenAI web search call identity was reused.",
                                    reason_code="web_search_call_identity_was_reused",
                                )
                            seen_call_ids.add(call_id)
                            pending_call_ids.add(call_id)
                        elif call_id not in pending_call_ids:
                            raise OpenAIProtocolError(
                                "OpenAI web search progress has no pending call.",
                                reason_code="web_search_progress_has_no_pending_call",
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
    pending_tool_search_calls: dict[int, dict[str, Any]] = {}
    seen_tool_search_output_indexes: set[int] = set()
    hosted_tool_search_call_index: int | None = None
    hosted_tool_search_output_index: int | None = None
    streamed_text: dict[tuple[int, int], str] = {}
    streamed_text_offsets: dict[tuple[int, int], int] = {}
    streamed_visible_text: list[str] = []
    assembled_text_length = 0
    fallback_output_items: dict[int, dict[str, Any]] = {}
    pending_replay_items: dict[int, tuple[str, str]] = {}
    response_id: str | None = None
    completed = False
    async for event in _stream_events_with_cancellation_marker(events):
        if not isinstance(event, Mapping):
            raise OpenAIProtocolError(
                "OpenAI stream event must be a JSON object.",
                reason_code="stream_event_must_be_a_json_object",
            )
        event_type = event.get("type")
        if event_type == "cayu.internal.transport_cancelled":
            for call_id, _status in pending_web_search_calls.values():
                yield _web_search_outcome_unknown_event(call_id)
            pending_web_search_calls.clear()
            continue
        if event_type == "response.created":
            response = event.get("response")
            candidate_response_id = response.get("id") if isinstance(response, Mapping) else None
            if isinstance(candidate_response_id, str) and candidate_response_id.strip():
                if response_id is None:
                    response_id = candidate_response_id
                    observe_provider_semantic_progress(ProviderProgressKind.RESPONSE_IDENTITY)
                elif candidate_response_id != response_id:
                    raise OpenAIProtocolError(
                        "OpenAI stream emitted conflicting response identities.",
                        reason_code="stream_emitted_conflicting_response_identities",
                    )
            continue
        if event_type == "response.output_text.delta":
            delta = event.get("delta")
            if not isinstance(delta, str):
                raise OpenAIProtocolError(
                    "OpenAI output_text delta must be a string.",
                    reason_code="output_text_delta_must_be_a_string",
                )
            if delta:
                streamed_visible_text.append(delta)
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
                raise OpenAIProtocolError(
                    "OpenAI annotation.added requires annotation object.",
                    reason_code="annotation_added_requires_annotation_object",
                )
            output_index = _stream_output_index(event)
            content_index = event.get("content_index")
            if type(content_index) is not int or content_index < 0:
                raise OpenAIProtocolError(
                    "OpenAI annotation.added requires non-negative content_index.",
                    reason_code="annotation_added_requires_non_negative_content_index",
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
                raise OpenAIProtocolError(
                    "OpenAI refusal delta must be a string.",
                    reason_code="refusal_delta_must_be_a_string",
                )
            if delta:
                streamed_visible_text.append(delta)
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
                raise OpenAIProtocolError(
                    "OpenAI reasoning delta must be a string.",
                    reason_code="reasoning_delta_must_be_a_string",
                )
            if delta:
                yield ModelStreamEvent.thinking(delta)
            continue
        if event_type == "response.output_item.added":
            item = event.get("item")
            reasoning_added = False
            if isinstance(item, Mapping) and item.get("type") == "reasoning":
                output_index = _stream_output_index(event)
                if output_index in pending_reasoning_items or output_index in fallback_output_items:
                    raise OpenAIProtocolError(
                        "OpenAI reasoning output_item.added was repeated.",
                        reason_code="reasoning_output_item_added_was_repeated",
                    )
                pending_reasoning_items.add(output_index)
                reasoning_added = True
            _record_stream_replay_item_added(event, pending_replay_items)
            if reasoning_added:
                observe_provider_semantic_progress(ProviderProgressKind.REASONING)
            if isinstance(item, Mapping) and item.get("type") == "web_search_call":
                output_index = _stream_output_index(event)
                if output_index in pending_web_search_calls:
                    raise OpenAIProtocolError(
                        "OpenAI web_search_call output_item.added was repeated.",
                        reason_code="web_search_call_output_item_added_was_repeated",
                    )
                normalized = _normalized_web_search_call(item, item_index=output_index)
                if normalized["status"] != "in_progress":
                    raise OpenAIProtocolError(
                        "OpenAI web_search_call output_item.added must be in progress.",
                        reason_code="web_search_call_output_item_added_must_be_in_progress",
                    )
                pending_web_search_calls[output_index] = (
                    normalized["id"],
                    normalized["status"],
                )
                yield _web_search_call_event(normalized)
            if isinstance(item, Mapping) and item.get("type") == "tool_search_call":
                output_index = _stream_output_index(event)
                if output_index in seen_tool_search_output_indexes:
                    raise OpenAIProtocolError(
                        "OpenAI tool_search_call output_item.added was repeated.",
                        reason_code="tool_search_call_output_item_added_was_repeated",
                    )
                if len(seen_tool_search_output_indexes) >= _OPENAI_CLIENT_TOOL_SEARCH_MAX_ITEMS:
                    raise OpenAIProtocolError(
                        "OpenAI stream contains too many tool search calls.",
                        reason_code="stream_contains_too_many_tool_search_calls",
                    )
                execution = item.get("execution")
                item_id = _mapping_optional_string(item, "id")
                call_id = _mapping_optional_string(item, "call_id")
                if execution == "client":
                    if item_id is None or call_id is None:
                        raise OpenAIProtocolError(
                            "OpenAI client tool_search_call added requires exact identities.",
                            reason_code="client_tool_search_call_added_requires_exact_identities",
                        )
                elif execution == "server":
                    if "call_id" not in item or item.get("call_id") is not None:
                        raise OpenAIProtocolError(
                            "OpenAI hosted tool_search_call added requires null call_id.",
                            reason_code="hosted_tool_search_call_added_requires_null_call_id",
                        )
                    if hosted_tool_search_call_index is not None:
                        raise OpenAIProtocolError(
                            "OpenAI stream contains multiple hosted tool search calls.",
                            reason_code="stream_contains_multiple_hosted_tool_search_calls",
                        )
                    hosted_tool_search_call_index = output_index
                else:
                    raise OpenAIProtocolError(
                        "OpenAI tool_search_call added has unsupported execution.",
                        reason_code="tool_search_call_added_has_unsupported_execution",
                    )
                if item.get("status") != "in_progress":
                    raise OpenAIProtocolError(
                        "OpenAI tool_search_call output_item.added must be in progress.",
                        reason_code="tool_search_call_output_item_added_must_be_in_progress",
                    )
                seen_tool_search_output_indexes.add(output_index)
                pending_tool_search_calls[output_index] = {
                    "execution": execution,
                    "id": item_id,
                    "call_id": call_id,
                }
            function_call_added = _record_stream_output_item_added(
                event,
                pending_function_calls,
            )
            if function_call_added:
                observe_provider_semantic_progress(ProviderProgressKind.TOOL_CALL)
            elif isinstance(item, Mapping) and item.get("type") == "tool_search_call":
                observe_provider_semantic_progress(
                    ProviderProgressKind.TOOL_CALL
                    if item.get("execution") == "client"
                    else ProviderProgressKind.HOSTED_TOOL
                )
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
                    "OpenAI web search lifecycle arrived before output_item.added.",
                    reason_code="web_search_lifecycle_arrived_before_output_item_added",
                )
            item_id = _mapping_optional_string(event, "item_id")
            if item_id is not None and item_id != pending[0]:
                raise OpenAIProtocolError(
                    "OpenAI web search lifecycle item_id mismatch.",
                    reason_code="web_search_lifecycle_item_id_mismatch",
                )
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
            if isinstance(item, Mapping) and item.get("type") == "web_search_call":
                output_index = _stream_output_index(event)
                pending = pending_web_search_calls.pop(output_index, None)
                if pending is None:
                    raise OpenAIProtocolError(
                        "OpenAI web_search_call output_item.done arrived before added.",
                        reason_code="web_search_call_output_item_done_arrived_before_added",
                    )
                normalized = _normalized_web_search_call(item, item_index=output_index)
                if normalized["id"] != pending[0]:
                    raise OpenAIProtocolError(
                        "OpenAI web_search_call output identity mismatch.",
                        reason_code="web_search_call_output_identity_mismatch",
                    )
                fallback_output_items[output_index] = normalized
                yield _web_search_call_event(normalized)
            if isinstance(item, Mapping) and item.get("type") == "tool_search_call":
                output_index = _stream_output_index(event)
                pending = pending_tool_search_calls.pop(output_index, None)
                if pending is None:
                    raise OpenAIProtocolError(
                        "OpenAI tool_search_call output_item.done arrived before added.",
                        reason_code="tool_search_call_output_item_done_arrived_before_added",
                    )
                normalized_tool_search = _normalized_tool_search_call(
                    item,
                    item_index=output_index,
                )
                if (
                    normalized_tool_search["execution"] != pending["execution"]
                    or normalized_tool_search.get("id") != pending["id"]
                    or normalized_tool_search.get("call_id") != pending["call_id"]
                ):
                    raise OpenAIProtocolError(
                        "OpenAI tool_search_call output identity mismatch.",
                        reason_code="tool_search_call_output_identity_mismatch",
                    )
                fallback_output_items[output_index] = normalized_tool_search
                if normalized_tool_search["execution"] == "client":
                    yield _tool_search_call_event(normalized_tool_search)
            if isinstance(item, Mapping) and item.get("type") == "tool_search_output":
                output_index = _stream_output_index(event)
                if (
                    output_index in fallback_output_items
                    or hosted_tool_search_call_index is None
                    or output_index != hosted_tool_search_call_index + 1
                    or hosted_tool_search_output_index is not None
                ):
                    raise OpenAIProtocolError(
                        "OpenAI tool_search_output has no unique adjacent server call.",
                        reason_code="tool_search_output_has_no_unique_adjacent_server_call",
                    )
                normalized_output, _loaded_tools = _normalized_hosted_tool_search_output(
                    item,
                    item_index=output_index,
                )
                fallback_output_items[output_index] = normalized_output
                hosted_tool_search_output_index = output_index
                observe_provider_semantic_progress(ProviderProgressKind.HOSTED_TOOL)
            _record_stream_output_item_done(
                event,
                fallback_output_items,
                pending_replay_items=pending_replay_items,
                streamed_text=streamed_text,
            )
            if isinstance(item, Mapping) and item.get("type") == "reasoning":
                pending_reasoning_items.discard(_stream_output_index(event))
                observe_provider_semantic_progress(ProviderProgressKind.REASONING)
            continue
        if event_type == "response.function_call_arguments.delta":
            _record_stream_function_call_delta(event, pending_function_calls)
            if event.get("delta"):
                observe_provider_semantic_progress(ProviderProgressKind.TOOL_CALL)
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
                *pending_tool_search_calls,
                *pending_replay_items,
            }
            # A completed response promises complete output items. An incomplete
            # response may end mid-item, so retain the terminal classification but
            # exclude partial state that cannot be replayed or executed safely.
            if event_type == "response.completed" and pending_function_calls:
                raise OpenAIProtocolError(
                    "OpenAI streaming response completed with unfinished function calls.",
                    reason_code="streaming_response_completed_with_unfinished_function_calls",
                )
            if event_type == "response.completed" and pending_reasoning_items:
                raise OpenAIProtocolError(
                    "OpenAI streaming response completed with unfinished reasoning items.",
                    reason_code="streaming_response_completed_with_unfinished_reasoning_items",
                )
            if event_type == "response.completed" and pending_replay_items:
                raise OpenAIProtocolError(
                    "OpenAI streaming response completed with unfinished output items.",
                    reason_code="streaming_response_completed_with_unfinished_output_items",
                )
            if event_type == "response.completed" and pending_web_search_calls:
                for call_id, _status in pending_web_search_calls.values():
                    yield _web_search_outcome_unknown_event(call_id)
                raise OpenAIProtocolError(
                    "OpenAI streaming response completed with unfinished web search calls.",
                    reason_code="streaming_response_completed_with_unfinished_web_search_calls",
                )
            if event_type == "response.completed" and pending_tool_search_calls:
                raise OpenAIProtocolError(
                    "OpenAI streaming response completed with unfinished tool search calls.",
                    reason_code="streaming_response_completed_with_unfinished_tool_search_calls",
                )
            if event_type == "response.incomplete":
                for call_id, _status in pending_web_search_calls.values():
                    yield _web_search_outcome_unknown_event(call_id)
                pending_web_search_calls.clear()
                pending_tool_search_calls.clear()
            observe_provider_semantic_progress(ProviderProgressKind.TERMINAL)
            for terminal_event in _stream_terminal_events(
                event,
                fallback_output_items,
                excluded_output_indexes=unfinished_output_indexes,
                reasoning_state=reasoning_state,
                streamed_visible_text=(
                    "".join(streamed_visible_text) if streamed_visible_text else None
                ),
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
        raise OpenAIProtocolError(
            "OpenAI streaming response ended before response.completed.",
            reason_code="streaming_response_ended_before_response_completed",
        )


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
            f"OpenAI message output item {item_index} must have assistant role.",
            reason_code="message_output_item_must_have_assistant_role",
        )
    content = item.get("content")
    if not isinstance(content, list):
        raise OpenAIProtocolError(
            f"OpenAI message output item {item_index} content must be a list.",
            reason_code="message_output_item_content_must_be_a_list",
        )
    events: list[ModelStreamEvent] = []
    message_text_length = 0
    for content_index, part in enumerate(content):
        if not isinstance(part, Mapping):
            raise OpenAIProtocolError(
                f"OpenAI message output content {item_index}.{content_index} must be an object.",
                reason_code="message_output_content_must_be_an_object",
            )
        part = cast("Mapping[str, Any]", part)
        part_type = part.get("type")
        if part_type == "output_text":
            text_key = "text"
        elif part_type == "refusal":
            text_key = "refusal"
        else:
            raise OpenAIProtocolError(
                f"Unsupported OpenAI message output content type: {part_type!r}.",
                reason_code="unsupported_openai_message_output_content_type",
            )
        text = part.get(text_key)
        if not isinstance(text, str):
            raise OpenAIProtocolError(
                f"OpenAI {part_type} content requires string {text_key}.",
                reason_code="content_requires_string",
            )
        if text:
            events.append(ModelStreamEvent.text_delta(text))
        annotations = part.get("annotations", [])
        if not isinstance(annotations, list):
            raise OpenAIProtocolError(
                f"OpenAI {part_type} content annotations must be a list.",
                reason_code="content_annotations_must_be_a_list",
            )
        for annotation_index, annotation in enumerate(annotations):
            if not isinstance(annotation, Mapping):
                raise OpenAIProtocolError(
                    "OpenAI output annotation "
                    f"{item_index}.{content_index}.{annotation_index} must be an object.",
                    reason_code="output_annotation_must_be_an_object",
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
            "OpenAI web search call identity is duplicated across output items.",
            reason_code="web_search_call_identity_is_duplicated_across_output_items",
        )
    call_indexes[call_id] = output_index


def _normalized_web_search_call(
    item: Mapping[str, Any],
    *,
    item_index: int,
) -> dict[str, Any]:
    call_id = item.get("id")
    if not isinstance(call_id, str) or not call_id.strip():
        raise OpenAIProtocolError(
            f"OpenAI web_search_call item {item_index} requires nonblank id.",
            reason_code="web_search_call_item_requires_nonblank_id",
        )
    status = item.get("status")
    if status not in {"in_progress", "searching", "completed", "incomplete", "failed"}:
        raise OpenAIProtocolError(
            f"OpenAI web_search_call item {item_index} has unsupported status.",
            reason_code="web_search_call_item_has_unsupported_status",
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
            f"OpenAI web_search_call item {item_index} completed without action evidence.",
            reason_code="web_search_call_item_completed_without_action_evidence",
        )
    return normalized


def _normalized_web_search_action(action: object, *, path: str) -> dict[str, Any]:
    if not isinstance(action, Mapping):
        raise OpenAIProtocolError(
            f"OpenAI {path} must be an object.", reason_code="web_search_action_must_be_an_object"
        )
    action = cast("Mapping[str, Any]", action)
    action_type = action.get("type")
    if action_type not in {"search", "open_page", "find_in_page"}:
        raise OpenAIProtocolError(
            f"OpenAI {path}.type is unsupported.",
            reason_code="web_search_action_type_is_unsupported",
        )
    normalized: dict[str, Any] = {"type": action_type}
    for key in ("query", "pattern"):
        value = action.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip() or len(value) > 4096:
            raise OpenAIProtocolError(
                f"OpenAI {path}.{key} must be a bounded string.",
                reason_code="web_search_action_must_be_a_bounded_string",
            )
        normalized[key] = value
    raw_url = action.get("url")
    if raw_url is not None:
        normalized["url"] = _normalized_external_web_url(raw_url, path=f"{path}.url")
    raw_queries = action.get("queries")
    if raw_queries is not None:
        if not isinstance(raw_queries, list) or len(raw_queries) > 100:
            raise OpenAIProtocolError(
                f"OpenAI {path}.queries must be a bounded list.",
                reason_code="web_search_action_queries_must_be_a_bounded_list",
            )
        queries: list[str] = []
        for index, query in enumerate(raw_queries):
            if not isinstance(query, str) or not query.strip() or len(query) > 4096:
                raise OpenAIProtocolError(
                    f"OpenAI {path}.queries[{index}] must be a bounded string.",
                    reason_code="web_search_action_queries_must_be_a_bounded_string",
                )
            queries.append(query)
        normalized["queries"] = queries
    raw_sources = action.get("sources")
    if raw_sources is not None:
        if not isinstance(raw_sources, list) or len(raw_sources) > 100:
            raise OpenAIProtocolError(
                f"OpenAI {path}.sources must be a bounded list.",
                reason_code="web_search_action_sources_must_be_a_bounded_list",
            )
        sources: list[dict[str, str]] = []
        for index, source in enumerate(raw_sources):
            if not isinstance(source, Mapping):
                raise OpenAIProtocolError(
                    f"OpenAI {path}.sources[{index}] must be an object.",
                    reason_code="web_search_action_sources_must_be_an_object",
                )
            source = cast("Mapping[str, Any]", source)
            source_type = source.get("type", "url")
            url = source.get("url")
            title = source.get("title")
            if source_type != "url":
                raise OpenAIProtocolError(
                    f"OpenAI {path}.sources[{index}].type is unsupported.",
                    reason_code="web_search_action_sources_type_is_unsupported",
                )
            url = _normalized_external_web_url(
                url,
                path=f"{path}.sources[{index}].url",
            )
            if title is not None and (
                not isinstance(title, str) or not title.strip() or len(title) > 1024
            ):
                raise OpenAIProtocolError(
                    f"OpenAI {path}.sources[{index}].title must be a bounded string.",
                    reason_code="web_search_action_sources_title_must_be_a_bounded_string",
                )
            sources.append(
                {"type": "url", "url": url, **({"title": title} if title is not None else {})}
            )
        normalized["sources"] = sources
    try:
        WebSearchAction.model_validate(normalized)
    except ValueError as exc:
        raise OpenAIProtocolError(
            f"OpenAI {path} is invalid.", reason_code="web_search_action_is_invalid"
        ) from exc
    return normalized


def _normalized_external_web_url(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 4096:
        raise OpenAIProtocolError(
            f"OpenAI {path} must be a bounded URL.", reason_code="external_url_must_be_bounded"
        )
    try:
        return WebSearchSource(url=value).url
    except ValueError as exc:
        raise OpenAIProtocolError(
            f"OpenAI {path} must use http or https.",
            reason_code="external_url_requires_http_or_https",
        ) from exc


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
        raise OpenAIProtocolError(
            f"OpenAI citation {path}.title must be a bounded string.",
            reason_code="citation_title_must_be_a_bounded_string",
        )
    if (start_index is None) != (end_index is None):
        raise OpenAIProtocolError(
            f"OpenAI citation {path} has invalid text offsets.",
            reason_code="citation_has_invalid_text_offsets",
        )
    if start_index is not None and (
        type(start_index) is not int
        or type(end_index) is not int
        or start_index < 0
        or end_index <= start_index
        or end_index > len(text)
    ):
        raise OpenAIProtocolError(
            f"OpenAI citation {path} has invalid text offsets.",
            reason_code="citation_has_invalid_text_offsets",
        )
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
            f"OpenAI function_call item {item_index} requires nonblank call_id.",
            reason_code="function_call_item_requires_nonblank_call_id",
        )
    if not isinstance(name, str) or not name.strip():
        raise OpenAIProtocolError(
            f"OpenAI function_call item {item_index} requires nonblank name.",
            reason_code="function_call_item_requires_nonblank_name",
        )
    if not isinstance(arguments, str):
        raise OpenAIProtocolError(
            f"OpenAI function_call item {item_index} requires string arguments.",
            reason_code="function_call_item_requires_string_arguments",
        )
    try:
        decoded_arguments = json.loads(arguments)
    except ValueError as exc:
        raise OpenAIProtocolError(
            f"OpenAI function_call item {item_index} arguments were not valid JSON.",
            reason_code="function_call_item_arguments_were_not_valid_json",
        ) from exc
    if type(decoded_arguments) is not dict:
        raise OpenAIProtocolError(
            f"OpenAI function_call item {item_index} arguments must decode to an object.",
            reason_code="function_call_item_arguments_must_decode_to_an_object",
        )
    return ModelStreamEvent.tool_call(
        id=call_id,
        name=name,
        arguments=copy_json_value(decoded_arguments, "arguments"),
    )


def _openai_function_call_recovery_digest(
    item: Mapping[str, Any],
    *,
    item_index: int,
) -> str:
    """Commit to one normalized call without persisting its arguments."""

    _validate_completed_stream_item_status(item, item_index)
    _function_call_event(item, item_index)
    normalized: dict[str, Any] = {
        "type": "function_call",
        "call_id": item.get("call_id"),
        "name": item.get("name"),
        "arguments": item.get("arguments"),
        "status": item.get("status"),
    }
    if item.get("id") is not None:
        normalized["id"] = item.get("id")
    material = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(material).hexdigest()


def _normalized_tool_search_call(
    item: Mapping[str, Any],
    *,
    item_index: int,
) -> dict[str, Any]:
    execution = item.get("execution")
    if execution == "server":
        if not set(item).issubset({"type", "id", "call_id", "execution", "arguments", "status"}):
            raise OpenAIProtocolError(
                f"OpenAI hosted tool_search_call item {item_index} has unsupported fields.",
                reason_code="hosted_tool_search_call_item_has_unsupported_fields",
            )
        if "call_id" not in item or item.get("call_id") is not None:
            raise OpenAIProtocolError(
                f"OpenAI hosted tool_search_call item {item_index} requires null call_id.",
                reason_code="hosted_tool_search_call_item_requires_null_call_id",
            )
        if item.get("status") != "completed":
            raise OpenAIProtocolError(
                f"OpenAI hosted tool_search_call item {item_index} must be completed.",
                reason_code="hosted_tool_search_call_item_must_be_completed",
            )
        arguments = item.get("arguments")
        if type(arguments) is not dict:
            raise OpenAIProtocolError(
                f"OpenAI hosted tool_search_call item {item_index} arguments must be an object.",
                reason_code="hosted_tool_search_call_item_arguments_must_be_an_object",
            )
        copied_arguments = copy_json_value(arguments, "hosted tool_search arguments")
        if (
            len(
                json.dumps(
                    copied_arguments,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            > _OPENAI_HOSTED_TOOL_SEARCH_MAX_ARGUMENT_BYTES
        ):
            raise OpenAIProtocolError(
                f"OpenAI hosted tool_search_call item {item_index} arguments are oversized.",
                reason_code="hosted_tool_search_call_item_arguments_are_oversized",
            )
        normalized: dict[str, Any] = {
            "type": "tool_search_call",
            "execution": "server",
            "call_id": None,
            "status": "completed",
            "arguments": copied_arguments,
        }
        item_id = item.get("id")
        if item_id is not None:
            if not isinstance(item_id, str) or not item_id.strip():
                raise OpenAIProtocolError(
                    f"OpenAI hosted tool_search_call item {item_index} has invalid id.",
                    reason_code="hosted_tool_search_call_item_has_invalid_id",
                )
            normalized["id"] = item_id
        return normalized
    if execution != "client":
        raise OpenAIProtocolError(
            f"OpenAI tool_search_call item {item_index} has unsupported execution.",
            reason_code="tool_search_call_item_has_unsupported_execution",
        )
    item_id = item.get("id")
    call_id = item.get("call_id")
    if not isinstance(item_id, str) or not item_id.strip():
        raise OpenAIProtocolError(
            f"OpenAI tool_search_call item {item_index} requires nonblank id.",
            reason_code="tool_search_call_item_requires_nonblank_id",
        )
    if not isinstance(call_id, str) or not call_id.strip():
        raise OpenAIProtocolError(
            f"OpenAI tool_search_call item {item_index} requires nonblank call_id.",
            reason_code="tool_search_call_item_requires_nonblank_call_id",
        )
    if item.get("status") != "completed":
        raise OpenAIProtocolError(
            f"OpenAI tool_search_call item {item_index} must be completed.",
            reason_code="tool_search_call_item_must_be_completed",
        )
    arguments = item.get("arguments")
    if type(arguments) is not dict:
        raise OpenAIProtocolError(
            f"OpenAI tool_search_call item {item_index} arguments must be an object.",
            reason_code="tool_search_call_item_arguments_must_be_an_object",
        )
    if not set(arguments).issubset({"query", "limit"}):
        raise OpenAIProtocolError(
            f"OpenAI tool_search_call item {item_index} has unsupported arguments.",
            reason_code="tool_search_call_item_has_unsupported_arguments",
        )
    query = arguments.get("query")
    if type(query) is not str:
        raise OpenAIProtocolError(
            f"OpenAI tool_search_call item {item_index} requires a bounded query.",
            reason_code="tool_search_call_item_requires_a_bounded_query",
        )
    try:
        query = require_durable_clean_nonblank(query, "tool_search query")
    except (TypeError, ValueError):
        raise OpenAIProtocolError(
            f"OpenAI tool_search_call item {item_index} requires a bounded query.",
            reason_code="tool_search_call_item_requires_a_bounded_query",
        ) from None
    if len(query) > _OPENAI_CLIENT_TOOL_SEARCH_MAX_QUERY_CHARS:
        raise OpenAIProtocolError(
            f"OpenAI tool_search_call item {item_index} requires a bounded query.",
            reason_code="tool_search_call_item_requires_a_bounded_query",
        )
    limit = arguments.get("limit")
    if limit is not None and (
        type(limit) is not int or not 1 <= limit <= _OPENAI_CLIENT_TOOL_SEARCH_MAX_RESULTS
    ):
        raise OpenAIProtocolError(
            f"OpenAI tool_search_call item {item_index} has an invalid limit.",
            reason_code="tool_search_call_item_has_an_invalid_limit",
        )
    return {
        "type": "tool_search_call",
        "id": item_id,
        "call_id": call_id,
        "execution": "client",
        "arguments": copy_json_value(arguments, "tool_search arguments"),
        "status": "completed",
    }


def _normalized_hosted_tool_search_output(
    item: Mapping[str, Any],
    *,
    item_index: int,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    if not set(item).issubset({"type", "id", "call_id", "execution", "status", "tools"}):
        raise OpenAIProtocolError(
            f"OpenAI hosted tool_search_output item {item_index} has unsupported fields.",
            reason_code="hosted_tool_search_output_item_has_unsupported_fields",
        )
    if item.get("execution") != "server":
        raise OpenAIProtocolError(
            f"OpenAI hosted tool_search_output item {item_index} must use server execution.",
            reason_code="hosted_tool_search_output_item_must_use_server_execution",
        )
    if "call_id" not in item or item.get("call_id") is not None:
        raise OpenAIProtocolError(
            f"OpenAI hosted tool_search_output item {item_index} requires null call_id.",
            reason_code="hosted_tool_search_output_item_requires_null_call_id",
        )
    if item.get("status") != "completed":
        raise OpenAIProtocolError(
            f"OpenAI hosted tool_search_output item {item_index} must be completed.",
            reason_code="hosted_tool_search_output_item_must_be_completed",
        )
    raw_tools = item.get("tools")
    if type(raw_tools) is not list:
        raise OpenAIProtocolError(
            f"OpenAI hosted tool_search_output item {item_index} requires a tools array.",
            reason_code="hosted_tool_search_output_item_requires_a_tools_array",
        )
    loaded_tools: list[dict[str, Any]] = []
    normalized_tools: list[dict[str, Any]] = []
    for tool_index, raw_tool in enumerate(raw_tools):
        if type(raw_tool) is not dict:
            raise OpenAIProtocolError(
                "Cayu's hosted Tool Search projection accepts only loaded functions; "
                f"item {item_index} tool {tool_index} is unsupported.",
                reason_code="cayu_s_hosted_tool_search_projection_accepts_only_loaded_functions_item_tool_is_unsupported",
            )
        raw_tool = cast("dict[str, Any]", raw_tool)
        if not set(raw_tool).issubset(
            {
                "type",
                "name",
                "description",
                "parameters",
                "strict",
                "defer_loading",
                "output_schema",
            }
        ):
            raise OpenAIProtocolError(
                f"OpenAI hosted loaded function {tool_index} has unsupported fields.",
                reason_code="hosted_loaded_function_has_unsupported_fields",
            )
        if raw_tool.get("type") != "function":
            raise OpenAIProtocolError(
                "Cayu's hosted Tool Search projection accepts only loaded functions; "
                f"item {item_index} tool {tool_index} is unsupported.",
                reason_code="cayu_s_hosted_tool_search_projection_accepts_only_loaded_functions_item_tool_is_unsupported",
            )
        name = raw_tool.get("name")
        description = raw_tool.get("description")
        parameters = raw_tool.get("parameters")
        if not isinstance(name, str) or not name.strip():
            raise OpenAIProtocolError(
                f"OpenAI hosted loaded function {tool_index} requires a nonblank name.",
                reason_code="hosted_loaded_function_requires_a_nonblank_name",
            )
        _validate_openai_tool_name(name)
        if not isinstance(description, str) or type(parameters) is not dict:
            raise OpenAIProtocolError(
                f"OpenAI hosted loaded function {tool_index} is malformed.",
                reason_code="hosted_loaded_function_is_malformed",
            )
        if raw_tool.get("defer_loading") is not True:
            raise OpenAIProtocolError(
                f"OpenAI hosted loaded function {tool_index} lost defer_loading authority.",
                reason_code="hosted_loaded_function_lost_defer_loading_authority",
            )
        strict = raw_tool.get("strict", False)
        if strict is not False:
            raise OpenAIProtocolError(
                f"OpenAI hosted loaded function {tool_index} changed strict mode.",
                reason_code="hosted_loaded_function_changed_strict_mode",
            )
        # Cayu has no registered output-schema authority. OpenAI may echo null
        # for the omitted optional field; a contract-bearing schema must fail closed.
        if raw_tool.get("output_schema") is not None:
            raise OpenAIProtocolError(
                f"OpenAI hosted loaded function {tool_index} added an output schema.",
                reason_code="hosted_loaded_function_added_an_output_schema",
            )
        normalized_tool = {
            "type": "function",
            "name": name,
            "description": description,
            "parameters": copy_json_value(parameters, "hosted loaded parameters"),
            "strict": strict,
            "defer_loading": True,
        }
        normalized_tools.append(normalized_tool)
        loaded_tools.append(
            {
                "name": name,
                "description": description,
                "input_schema": copy_json_value(parameters, "hosted loaded input_schema"),
            }
        )
    try:
        result = ToolDiscoveryProjectionResult(loaded_tools=tuple(loaded_tools))
    except (TypeError, ValueError) as exc:
        raise OpenAIProtocolError(
            "OpenAI hosted loaded tools are not bounded and canonical.",
            reason_code="hosted_loaded_tools_are_not_bounded_and_canonical",
        ) from exc
    normalized = {
        "type": "tool_search_output",
        "execution": "server",
        "call_id": None,
        "status": "completed",
        "tools": normalized_tools,
    }
    item_id = item.get("id")
    if item_id is not None:
        if not isinstance(item_id, str) or not item_id.strip():
            raise OpenAIProtocolError(
                f"OpenAI hosted tool_search_output item {item_index} has invalid id.",
                reason_code="hosted_tool_search_output_item_has_invalid_id",
            )
        normalized["id"] = item_id
    return normalized, result.loaded_tools


def _normalized_hosted_tool_search_items(
    output: list[Any],
) -> tuple[dict[int, dict[str, Any]], ToolDiscoveryProjectionResult | None]:
    """Validate the exact adjacent server call/output sequence in one response."""

    normalized_items: dict[int, dict[str, Any]] = {}
    result: ToolDiscoveryProjectionResult | None = None
    pair_start: int | None = None
    for index, item in enumerate(output):
        if not isinstance(item, Mapping):
            continue
        item_type = item.get("type")
        if item_type == "tool_search_call" and item.get("execution") == "server":
            normalized_call = _normalized_tool_search_call(item, item_index=index)
            if result is not None or index + 1 >= len(output):
                raise OpenAIProtocolError(
                    "OpenAI hosted Tool Search must contain one adjacent call/output pair.",
                    reason_code="hosted_tool_search_must_contain_one_adjacent_call_output_pair",
                )
            raw_output = output[index + 1]
            if (
                not isinstance(raw_output, Mapping)
                or raw_output.get("type") != "tool_search_output"
            ):
                raise OpenAIProtocolError(
                    "OpenAI hosted tool_search_call must be followed by tool_search_output.",
                    reason_code="hosted_tool_search_call_must_be_followed_by_tool_search_output",
                )
            normalized_output, loaded_tools = _normalized_hosted_tool_search_output(
                raw_output,
                item_index=index + 1,
            )
            normalized_items[index] = normalized_call
            normalized_items[index + 1] = normalized_output
            result = ToolDiscoveryProjectionResult(loaded_tools=loaded_tools)
            pair_start = index
        elif item_type == "tool_search_output" and index not in normalized_items:
            raise OpenAIProtocolError(
                "OpenAI hosted tool_search_output must follow its server search call.",
                reason_code="hosted_tool_search_output_must_follow_its_server_search_call",
            )
    if pair_start is not None and any(
        isinstance(item, Mapping) and item.get("type") == "function_call"
        for item in output[:pair_start]
    ):
        raise OpenAIProtocolError(
            "OpenAI hosted Tool Search function calls must follow the loaded output.",
            reason_code="hosted_tool_search_function_calls_must_follow_the_loaded_output",
        )
    if pair_start is not None and any(
        isinstance(item, Mapping)
        and item.get("type") == "tool_search_call"
        and item.get("execution") != "server"
        for item in output
    ):
        raise OpenAIProtocolError(
            "OpenAI hosted Tool Search cannot mix client and server search calls.",
            reason_code="hosted_tool_search_cannot_mix_client_and_server_search_calls",
        )
    return normalized_items, result


def _tool_search_call_event(item: Mapping[str, Any]) -> ModelStreamEvent:
    return ModelStreamEvent.tool_call(
        id=cast("str", item["call_id"]),
        name=_CAYU_SEARCH_TOOLS_NAME,
        arguments=copy_json_value(item["arguments"], "tool_search arguments"),
    )


def _completed_event_from_response(
    response: Mapping[str, Any],
    provider_state_items: list[dict[str, Any]] | None = None,
    *,
    completion_output_items: list[Mapping[str, Any]] | None = None,
    reasoning_state: str = "inline",
    tool_discovery_result: ToolDiscoveryProjectionResult | None = None,
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
        raise OpenAIProtocolError(
            "OpenAI response usage must be an object.",
            reason_code="response_usage_must_be_an_object",
        )
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
        tool_discovery_result=tool_discovery_result,
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
        raise OpenAIProtocolError(
            "OpenAI response end_turn must be a boolean or null.",
            reason_code="response_end_turn_must_be_a_boolean_or_null",
        )
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
    return any(item.get("type") in {"function_call", "tool_search_call"} for item in output_items)


def _stream_terminal_events(
    event: Mapping[str, Any],
    fallback_output_items: Mapping[int, Mapping[str, Any]],
    *,
    excluded_output_indexes: set[int] | None = None,
    reasoning_state: str = "inline",
    streamed_visible_text: str | None = None,
    emitted_function_call_digests: Mapping[int, str] | None = None,
) -> list[ModelStreamEvent]:
    response = _stream_response_object(event)
    excluded_output_indexes = excluded_output_indexes or set()
    emitted_function_call_digests = emitted_function_call_digests or {}
    output = response.get("output")
    # Some completed Responses streams deliver every authoritative item through
    # output_item.done and leave the terminal envelope's repeated output empty.
    if output is None or (
        event.get("type") == "response.completed" and isinstance(output, list) and not output
    ):
        completed_output_items = {
            index: item
            for index, item in fallback_output_items.items()
            if index not in excluded_output_indexes
        }
        if streamed_visible_text is not None:
            _reconcile_fallback_visible_text(completed_output_items, streamed_visible_text)
        provider_state_items = _provider_state_items_from_output_items(completed_output_items)
        completion_output_items = list(_sorted_output_items(completed_output_items))
        _hosted_items, hosted_result = _normalized_hosted_tool_search_items(completion_output_items)
        return [
            _completed_event_from_response(
                response,
                provider_state_items,
                completion_output_items=completion_output_items,
                reasoning_state=reasoning_state,
                tool_discovery_result=hosted_result,
            )
        ]
    if not isinstance(output, list):
        raise OpenAIProtocolError(
            "OpenAI response output must be a list.", reason_code="response_output_must_be_a_list"
        )

    hosted_items, hosted_result = _normalized_hosted_tool_search_items(output)
    terminal_hosted_calls: dict[int, dict[str, Any]] = {}
    terminal_tool_search_calls: dict[int, dict[str, Any]] = {}
    terminal_tool_search_outputs: dict[int, dict[str, Any]] = {}
    terminal_function_calls: dict[int, Mapping[str, Any]] = {}
    hosted_call_indexes: dict[str, int] = {}
    completion_output_items: list[Mapping[str, Any]] = []
    for output_index, item in enumerate(output):
        if output_index in excluded_output_indexes:
            continue
        if not isinstance(item, Mapping):
            raise OpenAIProtocolError(
                f"OpenAI output item {output_index} must be an object.",
                reason_code="output_item_must_be_an_object",
            )
        item = cast("Mapping[str, Any]", item)
        if item.get("type") == "tool_search_call":
            if len(terminal_tool_search_calls) >= _OPENAI_CLIENT_TOOL_SEARCH_MAX_ITEMS:
                raise OpenAIProtocolError(
                    "OpenAI terminal response contains too many tool search calls.",
                    reason_code="terminal_response_contains_too_many_tool_search_calls",
                )
            normalized_tool_search = _normalized_tool_search_call(
                item,
                item_index=output_index,
            )
            terminal_tool_search_calls[output_index] = normalized_tool_search
            completion_output_items.append(normalized_tool_search)
            continue
        if item.get("type") == "tool_search_output":
            normalized_tool_search_output = hosted_items.get(output_index)
            if normalized_tool_search_output is None:
                raise OpenAIProtocolError(
                    "OpenAI terminal tool_search_output has no hosted search call.",
                    reason_code="terminal_tool_search_output_has_no_hosted_search_call",
                )
            terminal_tool_search_outputs[output_index] = normalized_tool_search_output
            completion_output_items.append(normalized_tool_search_output)
            continue
        if item.get("type") == "function_call":
            _validate_completed_stream_item_status(item, output_index)
            _function_call_event(item, output_index)
            terminal_function_calls[output_index] = item
            completion_output_items.append(item)
            continue
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
                    "OpenAI terminal response contains a nonterminal web search call.",
                    reason_code="terminal_response_contains_a_nonterminal_web_search_call",
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
                "OpenAI terminal web search identity conflicts with lifecycle evidence.",
                reason_code="terminal_web_search_identity_conflicts_with_lifecycle_evidence",
            )

    for output_index, lifecycle_item in lifecycle_hosted_calls.items():
        terminal_item = terminal_hosted_calls.get(output_index)
        if terminal_item is None:
            raise OpenAIProtocolError(
                "OpenAI terminal response omitted completed web search lifecycle evidence.",
                reason_code="terminal_response_omitted_completed_web_search_lifecycle_evidence",
            )
        if terminal_item != lifecycle_item:
            raise OpenAIProtocolError(
                "OpenAI terminal web search evidence conflicts with lifecycle evidence.",
                reason_code="terminal_web_search_evidence_conflicts_with_lifecycle_evidence",
            )

    terminal_events: list[ModelStreamEvent] = []
    for output_index, terminal_item in terminal_function_calls.items():
        fallback_item = fallback_output_items.get(output_index)
        if fallback_item is not None and any(
            fallback_item.get(key) != terminal_item.get(key)
            for key in ("type", "id", "call_id", "name", "arguments", "status")
        ):
            raise OpenAIProtocolError(
                "OpenAI terminal function-call evidence conflicts with lifecycle evidence.",
                reason_code="terminal_function_call_evidence_conflicts_with_lifecycle_evidence",
            )
        emitted_digest = emitted_function_call_digests.get(output_index)
        if emitted_digest is not None and emitted_digest != _openai_function_call_recovery_digest(
            terminal_item,
            item_index=output_index,
        ):
            raise OpenAIProtocolError(
                "OpenAI terminal function-call evidence conflicts with recovered lifecycle "
                "evidence.",
                reason_code="terminal_function_call_evidence_conflicts_with_recovered_lifecycle_evidence",
            )
        if fallback_item is None and emitted_digest is None:
            terminal_events.append(_function_call_event(terminal_item, output_index))
    for output_index, terminal_item in terminal_tool_search_calls.items():
        fallback_item = fallback_output_items.get(output_index)
        if fallback_item is not None and fallback_item != terminal_item:
            raise OpenAIProtocolError(
                "OpenAI terminal tool search evidence conflicts with lifecycle evidence.",
                reason_code="terminal_tool_search_evidence_conflicts_with_lifecycle_evidence",
            )
        if fallback_item is None and terminal_item["execution"] == "client":
            terminal_events.append(_tool_search_call_event(terminal_item))
    for output_index, terminal_item in terminal_tool_search_outputs.items():
        fallback_item = fallback_output_items.get(output_index)
        if fallback_item is not None and fallback_item != terminal_item:
            raise OpenAIProtocolError(
                "OpenAI terminal tool search output conflicts with lifecycle evidence.",
                reason_code="terminal_tool_search_output_conflicts_with_lifecycle_evidence",
            )
    for output_index, terminal_item in terminal_hosted_calls.items():
        fallback_item = fallback_output_items.get(output_index)
        if fallback_item is not None and fallback_item.get("type") != "web_search_call":
            raise OpenAIProtocolError(
                "OpenAI terminal web search output conflicts with lifecycle item identity.",
                reason_code="terminal_web_search_output_conflicts_with_lifecycle_item_identity",
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
            tool_discovery_result=hosted_result,
        )
    )
    return terminal_events


def _provider_state_items_from_response(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = response.get("output")
    if output is None:
        return []
    if not isinstance(output, list):
        raise OpenAIProtocolError(
            "OpenAI response output must be a list.", reason_code="response_output_must_be_a_list"
        )
    provider_state_items: list[dict[str, Any]] = []
    hosted_call_indexes: dict[str, int] = {}
    for index, item in enumerate(output):
        if not isinstance(item, Mapping):
            raise OpenAIProtocolError(
                f"OpenAI output item {index} must be an object.",
                reason_code="output_item_must_be_an_object",
            )
        item = cast("Mapping[str, Any]", item)
        item_type = item.get("type")
        if item_type in {"reasoning", "message", "function_call"}:
            provider_state_items.append(
                {"provider": "openai", "state": copy_json_value(item, "output_item")}
            )
            continue
        if item_type == "tool_search_call":
            provider_state_items.append(
                {
                    "provider": "openai",
                    "state": _normalized_tool_search_call(item, item_index=index),
                }
            )
            continue
        if item_type == "tool_search_output":
            normalized, _loaded_tools = _normalized_hosted_tool_search_output(
                item,
                item_index=index,
            )
            provider_state_items.append({"provider": "openai", "state": normalized})
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
        raise OpenAIProtocolError(
            f"Unsupported OpenAI output item type: {item_type!r}.",
            reason_code="unsupported_openai_output_item_type",
        )
    return provider_state_items


def _provider_state_items_from_output_items(
    output_items: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    provider_state_items: list[dict[str, Any]] = []
    for output_index in sorted(output_items):
        item = output_items[output_index]
        item_type = item.get("type")
        if item_type == "message":
            _validate_completed_stream_message(item, output_index)
        elif item_type == "function_call":
            _validate_completed_stream_item_status(item, output_index)
            _function_call_event(item, output_index)
        elif item_type == "reasoning":
            _validate_completed_stream_reasoning(item, output_index)
        elif item_type == "web_search_call":
            normalized = _normalized_web_search_call(item, item_index=output_index)
            if normalized["status"] != "completed":
                continue
        elif item_type == "tool_search_call":
            item = _normalized_tool_search_call(item, item_index=output_index)
        elif item_type == "tool_search_output":
            item, _loaded_tools = _normalized_hosted_tool_search_output(
                item,
                item_index=output_index,
            )
        else:
            raise OpenAIProtocolError(
                f"Unsupported OpenAI fallback output item type: {item_type!r}.",
                reason_code="unsupported_openai_fallback_output_item_type",
            )
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
) -> bool:
    output_index = _stream_output_index(event)
    item = event.get("item")
    if not isinstance(item, Mapping):
        raise OpenAIProtocolError(
            "OpenAI output_item.added requires item object.",
            reason_code="output_item_added_requires_item_object",
        )
    item_type = item.get("type")
    if item_type != "function_call":
        return False
    if output_index in pending_function_calls:
        raise OpenAIProtocolError(
            "OpenAI function_call output_item.added was repeated.",
            reason_code="function_call_output_item_added_was_repeated",
        )
    pending_function_calls[output_index] = _PendingFunctionCall(
        item_id=_mapping_optional_string(item, "id"),
        call_id=_mapping_optional_string(item, "call_id"),
        name=_mapping_optional_string(item, "name"),
        arguments=_mapping_string_or_default(item, "arguments", ""),
    )
    return True


def _record_stream_replay_item_added(
    event: Mapping[str, Any],
    pending_replay_items: dict[int, tuple[str, str]],
) -> None:
    output_index = _stream_output_index(event)
    item = event.get("item")
    if not isinstance(item, Mapping):
        raise OpenAIProtocolError(
            "OpenAI output_item.added requires item object.",
            reason_code="output_item_added_requires_item_object",
        )
    item_type = item.get("type")
    if item_type not in {"message", "reasoning"}:
        return
    if output_index in pending_replay_items:
        raise OpenAIProtocolError(
            "OpenAI replayable output_item.added was repeated.",
            reason_code="replayable_output_item_added_was_repeated",
        )
    item_id = _mapping_optional_string(item, "id")
    if item_id is None:
        raise OpenAIProtocolError(
            f"OpenAI {item_type} output_item.added requires nonblank id.",
            reason_code="output_item_added_requires_nonblank_id",
        )
    if item.get("status") not in {None, "in_progress", "incomplete"}:
        raise OpenAIProtocolError(
            f"OpenAI {item_type} output_item.added has invalid lifecycle status.",
            reason_code="output_item_added_has_invalid_lifecycle_status",
        )
    if item_type == "message":
        _message_output_events(item, output_index, text_offset=0)
    else:
        _validate_stream_reasoning_shape(item, output_index)
    pending_replay_items[output_index] = (item_type, item_id)


def _record_stream_output_item_done(
    event: Mapping[str, Any],
    output_items: dict[int, dict[str, Any]],
    *,
    pending_replay_items: dict[int, tuple[str, str]],
    streamed_text: Mapping[tuple[int, int], str],
) -> None:
    output_index = _stream_output_index(event)
    item = event.get("item")
    if not isinstance(item, Mapping):
        raise OpenAIProtocolError(
            "OpenAI output_item.done requires item object.",
            reason_code="output_item_done_requires_item_object",
        )
    item_type = item.get("type")
    if item_type in {"reasoning", "message"}:
        if output_index in output_items:
            raise OpenAIProtocolError(
                f"OpenAI {item_type} output_item.done was repeated.",
                reason_code="output_item_done_was_repeated",
            )
        pending = pending_replay_items.pop(output_index, None)
        item_id = _mapping_optional_string(item, "id")
        if pending is not None and (item_type != pending[0] or item_id != pending[1]):
            raise OpenAIProtocolError(
                f"OpenAI {item_type} output_item.done identity conflicts with added item.",
                reason_code="output_item_done_identity_conflicts_with_added_item",
            )
        if item_type == "message":
            _validate_completed_stream_message(item, output_index)
            _reconcile_streamed_message_text(item, output_index, streamed_text)
        else:
            _validate_completed_stream_reasoning(item, output_index)
        output_items[output_index] = copy_json_value(item, "output_item")
        return
    if item_type == "function_call":
        _validate_completed_stream_item_status(item, output_index)
        _function_call_event(item, output_index)
        existing = output_items.get(output_index)
        if existing is None:
            raise OpenAIProtocolError(
                "OpenAI function_call output_item.done arrived before arguments completion.",
                reason_code="function_call_output_item_done_arrived_before_arguments_completion",
            )
        if existing is not None and any(
            existing.get(key) != item.get(key)
            for key in ("type", "id", "call_id", "name", "arguments", "status")
        ):
            raise OpenAIProtocolError(
                "OpenAI function_call output_item.done conflicts with streamed arguments.",
                reason_code="function_call_output_item_done_conflicts_with_streamed_arguments",
            )
        output_items[output_index] = copy_json_value(item, "output_item")
        return
    if item_type == "tool_search_call":
        normalized = _normalized_tool_search_call(item, item_index=output_index)
        existing = output_items.get(output_index)
        if existing != normalized:
            raise OpenAIProtocolError(
                "OpenAI tool_search_call output_item.done conflicts with lifecycle evidence.",
                reason_code="tool_search_call_output_item_done_conflicts_with_lifecycle_evidence",
            )
        output_items[output_index] = normalized
        return
    if item_type == "tool_search_output":
        normalized, _loaded_tools = _normalized_hosted_tool_search_output(
            item,
            item_index=output_index,
        )
        existing = output_items.get(output_index)
        if existing != normalized:
            raise OpenAIProtocolError(
                "OpenAI tool_search_output output_item.done conflicts with lifecycle evidence.",
                reason_code="tool_search_output_output_item_done_conflicts_with_lifecycle_evidence",
            )
        output_items[output_index] = normalized
        return
    if item_type != "web_search_call":
        raise OpenAIProtocolError(
            f"Unsupported OpenAI output_item.done item type: {item_type!r}.",
            reason_code="unsupported_openai_output_item_done_item_type",
        )


def _validate_completed_stream_item_status(
    item: Mapping[str, Any],
    output_index: int,
) -> None:
    if item.get("status") not in {None, "completed"}:
        raise OpenAIProtocolError(
            f"OpenAI output_item.done item {output_index} must be completed.",
            reason_code="output_item_done_item_must_be_completed",
        )


def _validate_completed_stream_message(item: Mapping[str, Any], output_index: int) -> None:
    _validate_completed_stream_item_status(item, output_index)
    _message_output_events(item, output_index, text_offset=0)


def _validate_stream_reasoning_shape(item: Mapping[str, Any], output_index: int) -> None:
    summary = item.get("summary", [])
    if not isinstance(summary, list):
        raise OpenAIProtocolError(
            f"OpenAI reasoning output item {output_index} summary must be a list.",
            reason_code="reasoning_output_item_summary_must_be_a_list",
        )
    encrypted_content = item.get("encrypted_content")
    if encrypted_content is not None and not isinstance(encrypted_content, str):
        raise OpenAIProtocolError(
            f"OpenAI reasoning output item {output_index} encrypted_content must be a string.",
            reason_code="reasoning_output_item_encrypted_content_must_be_a_string",
        )


def _validate_completed_stream_reasoning(item: Mapping[str, Any], output_index: int) -> None:
    _validate_completed_stream_item_status(item, output_index)
    if _mapping_optional_string(item, "id") is None:
        raise OpenAIProtocolError(
            f"OpenAI reasoning output_item.done {output_index} requires nonblank id.",
            reason_code="reasoning_output_item_done_requires_nonblank_id",
        )
    _validate_stream_reasoning_shape(item, output_index)


def _reconcile_streamed_message_text(
    item: Mapping[str, Any],
    output_index: int,
    streamed_text: Mapping[tuple[int, int], str],
) -> None:
    content = cast("list[Mapping[str, Any]]", item["content"])
    for (streamed_output_index, content_index), text in streamed_text.items():
        if streamed_output_index != output_index:
            continue
        if content_index >= len(content):
            raise OpenAIProtocolError(
                "OpenAI message output_item.done omitted streamed text content.",
                reason_code="message_output_item_done_omitted_streamed_text_content",
            )
        part = content[content_index]
        if part.get("type") != "output_text" or part.get("text") != text:
            raise OpenAIProtocolError(
                "OpenAI message output_item.done conflicts with streamed text.",
                reason_code="message_output_item_done_conflicts_with_streamed_text",
            )


def _reconcile_fallback_visible_text(
    output_items: Mapping[int, Mapping[str, Any]],
    streamed_visible_text: str,
) -> None:
    fallback_visible_text: list[str] = []
    has_message = False
    for output_index in sorted(output_items):
        item = output_items[output_index]
        if item.get("type") != "message":
            continue
        has_message = True
        _validate_completed_stream_message(item, output_index)
        content = cast("list[Mapping[str, Any]]", item["content"])
        for part in content:
            text_key = "text" if part.get("type") == "output_text" else "refusal"
            fallback_visible_text.append(cast("str", part[text_key]))
    if has_message and "".join(fallback_visible_text) != streamed_visible_text:
        raise OpenAIProtocolError(
            "OpenAI fallback message content conflicts with streamed visible text.",
            reason_code="fallback_message_content_conflicts_with_streamed_visible_text",
        )


def _record_stream_function_call_delta(
    event: Mapping[str, Any],
    pending_function_calls: dict[int, _PendingFunctionCall],
) -> None:
    output_index = _stream_output_index(event)
    pending = pending_function_calls.get(output_index)
    if pending is None:
        raise OpenAIProtocolError(
            "OpenAI function_call_arguments.delta arrived before output_item.added.",
            reason_code="function_call_arguments_delta_arrived_before_output_item_added",
        )
    item_id = _mapping_optional_string(event, "item_id")
    if pending.item_id is not None and item_id is not None and pending.item_id != item_id:
        raise OpenAIProtocolError(
            "OpenAI function_call_arguments.delta item_id mismatch.",
            reason_code="function_call_arguments_delta_item_id_mismatch",
        )
    delta = event.get("delta")
    if not isinstance(delta, str):
        raise OpenAIProtocolError(
            "OpenAI function_call_arguments.delta requires string delta.",
            reason_code="function_call_arguments_delta_requires_string_delta",
        )
    pending.append_arguments(delta)


def _stream_function_call_event(
    event: Mapping[str, Any],
    pending_function_calls: dict[int, _PendingFunctionCall],
) -> tuple[ModelStreamEvent, dict[str, Any]]:
    output_index = _stream_output_index(event)
    pending = pending_function_calls.pop(output_index, None)
    if pending is None:
        raise OpenAIProtocolError(
            "OpenAI function_call_arguments.done arrived before output_item.added.",
            reason_code="function_call_arguments_done_arrived_before_output_item_added",
        )
    item_id = _mapping_optional_string(event, "item_id")
    if pending.item_id is not None and item_id is not None and pending.item_id != item_id:
        raise OpenAIProtocolError(
            "OpenAI function_call_arguments.done item_id mismatch.",
            reason_code="function_call_arguments_done_item_id_mismatch",
        )
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
        raise OpenAIProtocolError(
            "OpenAI stream terminal event requires response object.",
            reason_code="stream_terminal_event_requires_response_object",
        )
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
        raise OpenAIProtocolError(
            "OpenAI stream event requires integer output_index.",
            reason_code="stream_event_requires_integer_output_index",
        )
    if output_index < 0:
        raise OpenAIProtocolError(
            "OpenAI stream event output_index must be non-negative.",
            reason_code="stream_event_output_index_must_be_non_negative",
        )
    return output_index


def _mapping_optional_string(value: Mapping[str, Any] | None, key: str) -> str | None:
    if value is None:
        return None
    raw_value = value.get(key)
    if raw_value is None:
        return None
    if not isinstance(raw_value, str):
        raise OpenAIProtocolError(
            f"OpenAI stream field {key} must be a string.",
            reason_code="stream_field_must_be_a_string",
        )
    stripped = raw_value.strip()
    return stripped or None


def _mapping_string_or_default(value: Mapping[str, Any], key: str, default: str) -> str:
    raw_value = value.get(key, default)
    if not isinstance(raw_value, str):
        raise OpenAIProtocolError(
            f"OpenAI stream field {key} must be a string.",
            reason_code="stream_field_must_be_a_string",
        )
    return raw_value


def _first_nonblank_string(*values: str | None) -> str:
    for value in values:
        if value is not None and value.strip():
            return value
    raise OpenAIProtocolError(
        "OpenAI streaming function call is missing required identity.",
        reason_code="streaming_function_call_is_missing_required_identity",
    )


def _first_string(*values: str | None) -> str:
    for value in values:
        if value is not None:
            return value
    raise OpenAIProtocolError(
        "OpenAI streaming function call is missing arguments.",
        reason_code="streaming_function_call_is_missing_arguments",
    )


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


def _effective_openai_request_options_for_request(
    request: ModelRequest,
) -> dict[str, Any]:
    """Apply runtime-owned native-tool callability without changing the stable tool list."""

    if type(request) is not ModelRequest:
        raise TypeError("request must be a ModelRequest.")
    effective = _effective_openai_request_options(request.options)
    anchor_name = targeted_tool_native_cache_anchor_name(request.options)
    if anchor_name is None:
        if request.targeted_tool_projection is not None:
            raise ValueError("OpenAI additional_tools projection requires a stable cache anchor.")
        return effective

    anchor_matches = [tool for tool in request.tools if tool.get("name") == anchor_name]
    if len(anchor_matches) != 1:
        raise ValueError(
            "OpenAI native targeted-tool requests require one exact stable cache anchor."
        )
    available = _openai_native_allowed_tool_selectors(
        request,
        cache_anchor_name=anchor_name,
    )
    configured = effective.pop("tool_choice", None)
    effective["tool_choice"] = _openai_native_tool_choice(
        configured,
        available=available,
    )
    return effective


def _openai_native_allowed_tool_selectors(
    request: ModelRequest,
    *,
    cache_anchor_name: str,
) -> tuple[dict[str, str], ...]:
    selectors: list[dict[str, str]] = []
    callable_anchor = call_tool_core_callable(request.options)
    for tool in request.tools:
        selector = _openai_function_tool_selector(tool)
        if (
            request.tool_discovery_projection is not None
            and selector["name"] == _CAYU_SEARCH_TOOLS_NAME
        ):
            selectors.append({"type": "tool_search"})
            continue
        if selector["name"] != cache_anchor_name or callable_anchor:
            selectors.append(selector)
    for hosted_tool in request.hosted_tools:
        if type(hosted_tool) is not OpenAIWebSearch:
            raise TypeError("OpenAI hosted tools must be OpenAIWebSearch instances.")
        selectors.append({"type": "web_search"})
    projection = request.targeted_tool_projection
    if projection is not None:
        targeted_selectors = tuple(
            _openai_function_tool_selector(tool) for tool in projection.tools
        )
        direct_function_names = {
            selector["name"] for selector in selectors if selector["type"] == "function"
        }
        if any(selector["name"] in direct_function_names for selector in targeted_selectors):
            raise ValueError(
                "A targeted additional_tools function cannot also be a direct request tool."
            )
        selectors.extend(targeted_selectors)
    discovery_projection = request.tool_discovery_projection
    if discovery_projection is not None:
        selectors.extend(
            {"type": "function", "name": name} for name in discovery_projection.loaded_tool_names
        )
    identities = [tuple(sorted(selector.items())) for selector in selectors]
    if len(identities) != len(set(identities)):
        raise ValueError("OpenAI native targeted-tool callability contains duplicate tools.")
    return tuple(selectors)


def _openai_function_tool_selector(tool: Mapping[str, Any]) -> dict[str, str]:
    return {"type": "function", "name": _openai_tool_name(tool)}


def _openai_native_tool_choice(
    configured: object,
    *,
    available: tuple[dict[str, str], ...],
) -> str | dict[str, Any]:
    available_identities = {tuple(sorted(selector.items())) for selector in available}
    if configured is None or configured == "auto":
        if not available:
            return "none"
        return {
            "type": "allowed_tools",
            "mode": "auto",
            "tools": [dict(selector) for selector in available],
        }
    if configured == "none":
        return "none"
    if configured == "required":
        if not available:
            raise ValueError("OpenAI tool_choice='required' has no callable tools.")
        return {
            "type": "allowed_tools",
            "mode": "required",
            "tools": [dict(selector) for selector in available],
        }
    copied = copy_json_value(configured, "options.openai.tool_choice")
    if type(copied) is not dict:
        raise ValueError(
            "OpenAI native dynamic-tool mode supports tool_choice none, auto, required, "
            "a named callable tool, or an allowed_tools subset."
        )
    choice_type = copied.get("type")
    if choice_type == "allowed_tools":
        if set(copied) != {"type", "mode", "tools"}:
            raise ValueError("OpenAI allowed_tools tool_choice is not canonical.")
        if copied.get("mode") not in {"auto", "required"}:
            raise ValueError("OpenAI allowed_tools mode must be auto or required.")
        selected = copied.get("tools")
        if type(selected) is not list or not selected:
            raise ValueError("OpenAI allowed_tools must contain at least one tool selector.")
        selectors = tuple(_openai_allowed_tool_selector(item) for item in selected)
        identities = [tuple(sorted(selector.items())) for selector in selectors]
        if len(identities) != len(set(identities)):
            raise ValueError("OpenAI allowed_tools cannot contain duplicate selectors.")
        if any(identity not in available_identities for identity in identities):
            raise ValueError(
                "OpenAI allowed_tools contains a tool unavailable in the native request."
            )
        return {
            "type": "allowed_tools",
            "mode": copied["mode"],
            "tools": [dict(selector) for selector in selectors],
        }
    selector = _openai_allowed_tool_selector(copied)
    if tuple(sorted(selector.items())) not in available_identities:
        raise ValueError("OpenAI tool_choice selects a tool unavailable in the native request.")
    return selector


def _openai_allowed_tool_selector(value: object) -> dict[str, str]:
    if type(value) is not dict:
        raise ValueError("OpenAI tool selectors must be objects.")
    selector = cast("dict[str, Any]", value)
    selector_type = selector.get("type")
    if selector_type == "function":
        if set(selector) != {"type", "name"}:
            raise ValueError("OpenAI function tool selectors require only type and name.")
        name = selector.get("name")
        if type(name) is not str:
            raise ValueError("OpenAI function tool selectors require a string name.")
        _validate_openai_tool_name(name)
        return {"type": "function", "name": name}
    if selector_type == "web_search" and set(selector) == {"type"}:
        return {"type": "web_search"}
    if selector_type == "tool_search" and set(selector) == {"type"}:
        return {"type": "tool_search"}
    raise ValueError("OpenAI native dynamic-tool mode received an unsupported tool selector.")


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
    targeted_tool_projection: TargetedToolProjectionRequest | None = None,
    tool_discovery_projection: ToolDiscoveryProjectionRequest | None = None,
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
            message,
            reasoning_state=reasoning_state,
            use_provider_state=use_provider_state,
            targeted_tool_projection=targeted_tool_projection,
            tool_discovery_projection=tool_discovery_projection,
        )
        if provider_state_items:
            return provider_state_items
        if not use_provider_state:
            return _openai_neutral_assistant_items(
                message,
                tool_discovery_projection=tool_discovery_projection,
            )

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
                items.append(
                    _assistant_tool_call_input_item(
                        part,
                        tool_discovery_projection=tool_discovery_projection,
                    )
                )
            elif type(part) not in {
                TextPart,
                ProviderStatePart,
                ThinkingPart,
                HostedToolCallPart,
                CitationPart,
            }:
                raise OpenAIProtocolError(
                    "Assistant messages can only contain text, tool_call, provider_state, "
                    "thinking, hosted_tool_call, and citation parts.",
                    reason_code="assistant_messages_can_only_contain_text_tool_call_provider_state_thinking_hosted_tool_call_and_citation_parts",
                )
        # ThinkingPart is display-only here: OpenAI reasoning round-trips through the
        # encrypted reasoning ProviderStatePart, so the readable summary is not re-sent.
        return items
    if message.role == MessageRole.TOOL:
        items: list[dict[str, Any]] = []
        attachment_parts: list[dict[str, Any]] = []
        for part in message.content:
            items.append(
                _tool_result_output_item(
                    part,
                    tool_discovery_projection=tool_discovery_projection,
                )
            )
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
    raise OpenAIProtocolError(
        f"Unsupported Cayu message role: {message.role!r}.",
        reason_code="unsupported_cayu_message_role",
    )


def _openai_neutral_assistant_items(
    message: Message,
    *,
    tool_discovery_projection: ToolDiscoveryProjectionRequest | None = None,
) -> list[dict[str, Any]]:
    """Rebuild accepted assistant output without server-owned provider state."""

    items: list[dict[str, Any]] = []
    pending_text: list[str] = []
    pending_citations: list[CitationPart] = []
    assembled_text_length = 0
    pending_text_offset = 0
    tool_search_items_by_call_id: dict[str, dict[str, Any]] = {}
    used_tool_search_call_ids: set[str] = set()
    if (
        tool_discovery_projection is not None
        and tool_discovery_projection.protocol == OPENAI_CLIENT_TOOL_SEARCH_PROTOCOL
    ):
        for index, part in enumerate(message.content):
            if (
                type(part) is not ProviderStatePart
                or part.provider != "openai"
                or part.state.get("type") != "tool_search_call"
            ):
                continue
            normalized = _normalized_tool_search_call(part.state, item_index=index)
            call_id = cast("str", normalized["call_id"])
            if call_id in tool_search_items_by_call_id:
                raise OpenAIProtocolError(
                    "OpenAI neutral replay contains duplicate tool search call identity.",
                    reason_code="neutral_replay_contains_duplicate_tool_search_call_identity",
                )
            tool_search_items_by_call_id[call_id] = normalized

    def flush_text() -> None:
        nonlocal pending_text_offset
        if not pending_text:
            if pending_citations:
                raise OpenAIProtocolError(
                    "OpenAI neutral replay cannot attach a citation without assistant text.",
                    reason_code="neutral_replay_cannot_attach_a_citation_without_assistant_text",
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
                        "Completed hosted search replay requires action evidence.",
                        reason_code="completed_hosted_search_replay_requires_action_evidence",
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
            tool_search_item = tool_search_items_by_call_id.get(part.tool_call_id)
            if tool_search_item is not None and part.tool_name != _CAYU_SEARCH_TOOLS_NAME:
                raise OpenAIProtocolError(
                    "OpenAI tool search provider state conflicts with neutral tool-call replay.",
                    reason_code="tool_search_provider_state_conflicts_with_neutral_tool_call_replay",
                )
            if tool_search_item is not None:
                used_tool_search_call_ids.add(part.tool_call_id)
            items.append(
                _assistant_tool_call_input_item(
                    part,
                    tool_discovery_projection=tool_discovery_projection,
                    tool_search_item=tool_search_item,
                )
            )
            continue
        if type(part) in {ProviderStatePart, ThinkingPart}:
            continue
        raise OpenAIProtocolError(
            "Assistant messages can only contain text, tool_call, provider_state, "
            "thinking, hosted_tool_call, and citation parts.",
            reason_code="assistant_messages_can_only_contain_text_tool_call_provider_state_thinking_hosted_tool_call_and_citation_parts",
        )
    if used_tool_search_call_ids != set(tool_search_items_by_call_id):
        raise OpenAIProtocolError(
            "OpenAI neutral replay has tool search provider state without terminal tool evidence.",
            reason_code="neutral_replay_has_tool_search_provider_state_without_terminal_tool_evidence",
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
                "OpenAI neutral replay citation does not belong to its assistant text item.",
                reason_code="neutral_replay_citation_does_not_belong_to_its_assistant_text_item",
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


def _openai_discovery_ownership_tokens(
    projection: ToolDiscoveryProjectionRequest | None,
) -> tuple[str, ...] | None:
    if projection is None:
        return None
    if projection.protocol == OPENAI_CLIENT_TOOL_SEARCH_PROTOCOL:
        return projection.loaded_tool_names
    if not projection.candidate_tools:
        return ()
    material = json.dumps(
        {
            "protocol": projection.protocol,
            "generation_id": projection.generation_id,
            "candidate_tools": list(projection.candidate_tools),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return (f"hosted:{sha256(material).hexdigest()}",)


def _server_prefix_has_unsafe_dynamic_tools(
    server_owned_messages: list[Message],
    *,
    targeted_projection: TargetedToolProjectionRequest | None,
    discovery_projection: ToolDiscoveryProjectionRequest | None,
) -> bool:
    """Return whether a server chain retains authority absent from this request."""

    active_marker_id = None if targeted_projection is None else targeted_projection.marker_id
    latest_response_state: dict[str, Any] | None = None
    for message in server_owned_messages:
        if message.role is not MessageRole.ASSISTANT:
            continue
        for part in message.content:
            if type(part) is not ProviderStatePart or part.provider != "openai":
                continue
            state = part.state
            if state.get("type") == "response_ref":
                latest_response_state = state
    if latest_response_state is None or "targeted_tool_marker_id" not in latest_response_state:
        # A chained response without explicit ownership evidence cannot prove
        # that its provider-side prefix is safe to reuse. Rebuild neutrally.
        return True
    owned_marker_id = latest_response_state["targeted_tool_marker_id"]
    if owned_marker_id is None:
        if active_marker_id is not None:
            return True
    elif (
        type(owned_marker_id) is not str
        or len(owned_marker_id) != 71
        or not owned_marker_id.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in owned_marker_id[7:])
        or owned_marker_id != active_marker_id
    ):
        return True

    raw_owned_names = latest_response_state.get("tool_discovery_loaded_tool_names")
    if discovery_projection is None:
        # No current discovery projection means no loaded discovery authority
        # may remain addressable through a server-owned prefix.
        return "tool_discovery_loaded_tool_names" in latest_response_state
    if (
        type(raw_owned_names) is not list
        or len(raw_owned_names) > _OPENAI_CLIENT_TOOL_SEARCH_MAX_ITEMS
        or any(type(name) is not str for name in raw_owned_names)
    ):
        return True
    try:
        owned_names = tuple(
            require_clean_nonblank(name, "server-owned discovery tool name")
            for name in cast("list[str]", raw_owned_names)
        )
    except (TypeError, ValueError):
        return True
    if owned_names != tuple(sorted(set(owned_names))):
        return True
    expected_tokens = _openai_discovery_ownership_tokens(discovery_projection)
    if expected_tokens is None:  # pragma: no cover - projection is non-null above
        return True
    if discovery_projection.protocol == OPENAI_CLIENT_TOOL_SEARCH_PROTOCOL:
        return not set(owned_names).issubset(expected_tokens)
    return owned_names != expected_tokens


def _event_with_server_dynamic_tool_ownership(
    event: ModelStreamEvent,
    *,
    reasoning_state: str,
    marker_id: str | None,
    discovery_loaded_tool_names: tuple[str, ...] | None,
) -> ModelStreamEvent:
    """Bind a response reference to the dynamic tools retained on the server."""

    if reasoning_state != "server" or event.type is not ModelStreamEventType.COMPLETED:
        return event
    payload = copy_json_value(event.payload, "completed event payload")
    if type(payload) is not dict:  # pragma: no cover - ModelStreamEvent invariant
        raise AssertionError("Completed OpenAI event payload must be an object.")
    provider_state = payload.get("provider_state")
    if type(provider_state) is not list:
        raise OpenAIProtocolError(
            "OpenAI server completion lost its provider state.",
            reason_code="server_completion_lost_its_provider_state",
        )
    response_refs = [
        item
        for item in provider_state
        if type(item) is dict
        and item.get("provider") == "openai"
        and type(item.get("state")) is dict
        and item["state"].get("type") == "response_ref"
    ]
    if len(response_refs) != 1:
        raise OpenAIProtocolError(
            "OpenAI server completion requires one exact response reference.",
            reason_code="server_completion_requires_one_exact_response_reference",
        )
    response_refs[0]["state"]["targeted_tool_marker_id"] = marker_id
    if discovery_loaded_tool_names is not None:
        response_refs[0]["state"]["tool_discovery_loaded_tool_names"] = list(
            discovery_loaded_tool_names
        )
    return ModelStreamEvent(
        type=event.type,
        delta=event.delta,
        payload=payload,
        completion=event.completion,
        tool_discovery_result=event.tool_discovery_result,
        recovery_metadata=event.recovery_metadata,
        provider_operation_status=event.provider_operation_status,
    )


def _openai_provider_state_items(
    message: Message,
    *,
    reasoning_state: str = "inline",
    use_provider_state: bool = True,
    targeted_tool_projection: TargetedToolProjectionRequest | None = None,
    tool_discovery_projection: ToolDiscoveryProjectionRequest | None = None,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for part in message.content:
        if type(part) is not ProviderStatePart:
            continue
        if part.provider != "openai":
            continue
        state = copy_json_value(part.state, "provider_state")
        if type(state) is not dict:
            raise OpenAIProtocolError(
                "OpenAI provider state must be an object.",
                reason_code="provider_state_must_be_an_object",
            )
        item_type = state.get("type")
        if item_type == TARGETED_TOOL_PROJECTION_MARKER_TYPE:
            if (
                targeted_tool_projection is not None
                and state.get("protocol") == targeted_tool_projection.protocol
                and state.get("marker_id") == targeted_tool_projection.marker_id
            ):
                items.append(
                    {
                        "type": "additional_tools",
                        "role": "developer",
                        "tools": [_openai_tool(tool) for tool in targeted_tool_projection.tools],
                    }
                )
            continue
        if not use_provider_state:
            continue
        if item_type == "response_ref":
            continue  # synthetic chain marker, never sent as input
        if item_type == "reasoning":
            # Inline mode replays reasoning with its encrypted_content; server mode
            # leaves reasoning on OpenAI's servers, so never replays it.
            if reasoning_state == "server":
                continue
            items.append(state)
            continue
        if item_type == "tool_search_call":
            if tool_discovery_projection is None:
                raise OpenAIProtocolError(
                    "OpenAI tool_search_call state requires an active Tool Search projection.",
                    reason_code="tool_search_call_state_requires_an_active_tool_search_projection",
                )
            items.append(_normalized_tool_search_call(state, item_index=len(items)))
            continue
        if item_type == "tool_search_output":
            if (
                tool_discovery_projection is None
                or tool_discovery_projection.protocol != OPENAI_HOSTED_TOOL_SEARCH_PROTOCOL
            ):
                raise OpenAIProtocolError(
                    "OpenAI hosted tool_search_output state requires hosted Tool Search.",
                    reason_code="hosted_tool_search_output_state_requires_hosted_tool_search",
                )
            normalized, _loaded_tools = _normalized_hosted_tool_search_output(
                state,
                item_index=len(items),
            )
            items.append(normalized)
            continue
        if item_type not in {"message", "function_call", "web_search_call"}:
            raise OpenAIProtocolError(
                f"Unsupported OpenAI provider state item type: {item_type!r}.",
                reason_code="unsupported_openai_provider_state_item_type",
            )
        if item_type == "function_call":
            state = _rematerialize_targeted_provider_history_call(state)
        items.append(state)
    return items


def _validate_targeted_tool_projection_marker(
    messages: list[Message],
    projection: TargetedToolProjectionRequest | None,
) -> None:
    """Require one exact durable acquisition marker for an active projection."""

    if projection is None:
        return
    matching_markers = 0
    for message in messages:
        if message.role is not MessageRole.ASSISTANT:
            continue
        for part in message.content:
            if type(part) is not ProviderStatePart or part.provider != "openai":
                continue
            state = part.state
            if (
                state.get("type") != TARGETED_TOOL_PROJECTION_MARKER_TYPE
                or state.get("protocol") != projection.protocol
                or state.get("marker_id") != projection.marker_id
            ):
                continue
            if len(message.content) != 1 or set(state) != {
                "type",
                "protocol",
                "marker_id",
            }:
                raise OpenAIProtocolError(
                    "The targeted-tool acquisition marker is not canonical.",
                    reason_code="the_targeted_tool_acquisition_marker_is_not_canonical",
                )
            matching_markers += 1
    if matching_markers != 1:
        raise OpenAIProtocolError(
            "An active targeted-tool projection requires one exact acquisition marker.",
            reason_code="an_active_targeted_tool_projection_requires_one_exact_acquisition_marker",
        )


def _validate_tool_search_replay(
    messages: list[Message],
    projection: ToolDiscoveryProjectionRequest | None,
) -> None:
    """Require exact provider call evidence and one matching result per native search."""

    if projection is None:
        return
    if projection.protocol == OPENAI_HOSTED_TOOL_SEARCH_PROTOCOL:
        candidates_by_name = {
            cast("str", tool["name"]): tool for tool in projection.candidate_tools
        }
        replay_loaded_by_name: dict[str, dict[str, Any]] = {}
        for message in messages:
            if message.role is not MessageRole.ASSISTANT:
                continue
            states = [
                part.state
                for part in message.content
                if type(part) is ProviderStatePart and part.provider == "openai"
            ]
            pair_indexes: list[int] = []
            for index, state in enumerate(states):
                item_type = state.get("type")
                if item_type == "tool_search_output":
                    if (
                        index == 0
                        or states[index - 1].get("type") != "tool_search_call"
                        or states[index - 1].get("execution") != "server"
                    ):
                        raise OpenAIProtocolError(
                            "Hosted Tool Search replay contains an orphan loaded output.",
                            reason_code="hosted_tool_search_replay_contains_an_orphan_loaded_output",
                        )
                    continue
                if item_type != "tool_search_call":
                    continue
                if state.get("execution") != "server":
                    raise OpenAIProtocolError(
                        "Hosted Tool Search replay cannot contain a client search call.",
                        reason_code="hosted_tool_search_replay_cannot_contain_a_client_search_call",
                    )
                if (
                    index + 1 >= len(states)
                    or states[index + 1].get("type") != "tool_search_output"
                ):
                    raise OpenAIProtocolError(
                        "Hosted Tool Search replay requires an adjacent call/output pair.",
                        reason_code="hosted_tool_search_replay_requires_an_adjacent_call_output_pair",
                    )
                _normalized_tool_search_call(state, item_index=index)
                _normalized_output, loaded_tools = _normalized_hosted_tool_search_output(
                    states[index + 1],
                    item_index=index + 1,
                )
                for loaded_tool in loaded_tools:
                    name = cast("str", loaded_tool["name"])
                    if candidates_by_name.get(name) != loaded_tool:
                        raise OpenAIProtocolError(
                            "Hosted Tool Search replay loaded a function outside the current "
                            "candidate projection.",
                            reason_code="hosted_tool_search_replay_loaded_a_function_outside_the_current_candidate_projection",
                        )
                    replay_loaded_by_name[name] = loaded_tool
                pair_indexes.append(index)
            if len(pair_indexes) > 1:
                raise OpenAIProtocolError(
                    "Hosted Tool Search replay contains multiple search pairs in one response.",
                    reason_code="hosted_tool_search_replay_contains_multiple_search_pairs_in_one_response",
                )
            if pair_indexes and any(
                state.get("type") == "function_call" for state in states[: pair_indexes[0]]
            ):
                raise OpenAIProtocolError(
                    "Hosted Tool Search replay function calls must follow the loaded output.",
                    reason_code="hosted_tool_search_replay_function_calls_must_follow_the_loaded_output",
                )
            if any(
                state.get("type") == "function_call"
                and state.get("name") in candidates_by_name
                and state.get("name") not in replay_loaded_by_name
                for state in states
            ):
                raise OpenAIProtocolError(
                    "Hosted Tool Search replay called a deferred function outside the loaded "
                    "subset.",
                    reason_code="hosted_tool_search_replay_called_a_deferred_function_outside_the_loaded_subset",
                )
        if any(
            replay_loaded_by_name.get(cast("str", tool["name"])) != tool
            for tool in projection.loaded_tools
        ):
            raise OpenAIProtocolError(
                "Hosted Tool Search replay-loaded authority has no exact retained output.",
                reason_code="hosted_tool_search_replay_loaded_authority_has_no_exact_retained_output",
            )
        return
    calls: dict[str, dict[str, Any]] = {}
    pending_results: set[str] = set()
    for message in messages:
        if message.role is MessageRole.ASSISTANT:
            provider_calls: dict[str, dict[str, Any]] = {}
            assistant_calls: dict[str, ToolCallPart] = {}
            for index, part in enumerate(message.content):
                if (
                    type(part) is ProviderStatePart
                    and part.provider == "openai"
                    and part.state.get("type") == "tool_search_call"
                ):
                    normalized = _normalized_tool_search_call(part.state, item_index=index)
                    call_id = cast("str", normalized["call_id"])
                    if call_id in provider_calls or call_id in calls:
                        raise OpenAIProtocolError(
                            "OpenAI tool search replay repeats a provider call identity.",
                            reason_code="tool_search_replay_repeats_a_provider_call_identity",
                        )
                    provider_calls[call_id] = normalized
                elif type(part) is ToolCallPart and part.tool_name == _CAYU_SEARCH_TOOLS_NAME:
                    if part.tool_call_id in assistant_calls or part.tool_call_id in calls:
                        raise OpenAIProtocolError(
                            "OpenAI tool search replay repeats an assistant call identity.",
                            reason_code="tool_search_replay_repeats_an_assistant_call_identity",
                        )
                    assistant_calls[part.tool_call_id] = part
            if set(provider_calls) != set(assistant_calls):
                raise OpenAIProtocolError(
                    "OpenAI tool search replay requires matching provider and assistant calls.",
                    reason_code="tool_search_replay_requires_matching_provider_and_assistant_calls",
                )
            for call_id in assistant_calls:
                provider_call = provider_calls[call_id]
                calls[call_id] = provider_call
                pending_results.add(call_id)
            continue
        if message.role is not MessageRole.TOOL:
            continue
        for part in message.content:
            if type(part) is not ToolResultPart or part.tool_name != _CAYU_SEARCH_TOOLS_NAME:
                continue
            if part.tool_call_id not in pending_results:
                raise OpenAIProtocolError(
                    "OpenAI tool search output has no matching pending search call.",
                    reason_code="tool_search_output_has_no_matching_pending_search_call",
                )
            pending_results.remove(part.tool_call_id)
    if pending_results:
        raise OpenAIProtocolError(
            "OpenAI tool search replay has a call without its output.",
            reason_code="tool_search_replay_has_a_call_without_its_output",
        )


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
    raise OpenAIProtocolError(
        "User messages can only contain text and file parts.",
        reason_code="user_messages_can_only_contain_text_and_file_parts",
    )


def _resolved_user_attachment(
    part: FilePart,
    resolved_attachments: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    attachment = file_attachment_from_payload(part.attachment)
    if attachment is None:
        raise OpenAIProtocolError(
            "User file parts require a file attachment payload.",
            reason_code="user_file_parts_require_a_file_attachment_payload",
        )
    resolved = resolved_attachments.get(attachment.artifact_id)
    if resolved is None:
        raise OpenAIProtocolError(
            f"Missing resolved file attachment: {attachment.artifact_id}",
            reason_code="missing_resolved_file_attachment",
        )
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
        raise OpenAIProtocolError(
            "Assistant text output requires a text part.",
            reason_code="assistant_text_output_requires_a_text_part",
        )
    return {"type": "output_text", "text": part.text}


_TARGETED_PROVIDER_HISTORY_REFERENCE_PREFIX = "cayu_provider_history_v1."


def _rematerialize_targeted_provider_history_call(
    item: Mapping[str, Any],
) -> dict[str, Any]:
    """Replace a non-executable transcript marker only on the OpenAI wire."""

    copied = copy_json_value(item, "provider_history_function_call")
    if copied.get("name") != _CAYU_CALL_TOOL_NAME:
        return copied
    raw_arguments = copied.get("arguments")
    if type(raw_arguments) is not str:
        return copied
    try:
        arguments = json.loads(raw_arguments)
    except ValueError:
        return copied
    if type(arguments) is not dict or arguments.get("tool_ref") != REDACTED_SECRET:
        return copied
    call_id = require_clean_nonblank(copied.get("call_id"), "provider history call_id")
    arguments["tool_ref"] = (
        _TARGETED_PROVIDER_HISTORY_REFERENCE_PREFIX + sha256(call_id.encode("utf-8")).hexdigest()
    )
    copied["arguments"] = json.dumps(
        arguments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return copied


def _function_call_input_item(part: ToolCallPart) -> dict[str, Any]:
    return _rematerialize_targeted_provider_history_call(
        {
            "type": "function_call",
            "call_id": part.tool_call_id,
            "name": part.tool_name,
            "arguments": _json_arguments(part.arguments),
            "status": "completed",
        }
    )


def _assistant_tool_call_input_item(
    part: ToolCallPart,
    *,
    tool_discovery_projection: ToolDiscoveryProjectionRequest | None,
    tool_search_item: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if tool_discovery_projection is not None and part.tool_name == _CAYU_SEARCH_TOOLS_NAME:
        if tool_search_item is None:
            raise OpenAIProtocolError(
                "OpenAI client Tool Search replay requires the provider-issued item id.",
                reason_code="client_tool_search_replay_requires_the_provider_issued_item_id",
            )
        normalized = _normalized_tool_search_call(tool_search_item, item_index=0)
        if normalized["call_id"] != part.tool_call_id:
            raise OpenAIProtocolError(
                "OpenAI tool search provider state conflicts with neutral tool-call replay.",
                reason_code="tool_search_provider_state_conflicts_with_neutral_tool_call_replay",
            )
        return normalized
    return _function_call_input_item(part)


def _tool_result_output_item(
    part: TextPart
    | ToolCallPart
    | ToolResultPart
    | ProviderStatePart
    | ThinkingPart
    | FilePart
    | HostedToolCallPart
    | CitationPart,
    *,
    tool_discovery_projection: ToolDiscoveryProjectionRequest | None,
) -> dict[str, Any]:
    if (
        type(part) is ToolResultPart
        and tool_discovery_projection is not None
        and part.tool_name == _CAYU_SEARCH_TOOLS_NAME
    ):
        return _tool_search_output_item(
            part,
            loaded_tools=tool_discovery_projection.loaded_tools,
        )
    return _function_call_output_item(part)


def _tool_search_output_item(
    part: ToolResultPart,
    *,
    loaded_tools: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    trusted_tools = {cast("str", tool["name"]): tool for tool in loaded_tools}
    tools: list[dict[str, Any]] = []
    if not part.is_error:
        structured = part.structured
        if type(structured) is not dict or set(structured) != {
            "schema_version",
            "query",
            "matches",
            "view_revision",
            "truncated",
        }:
            raise OpenAIProtocolError(
                "A successful search_tools result requires the canonical structured payload.",
                reason_code="a_successful_search_tools_result_requires_the_canonical_structured_payload",
            )
        matches = structured.get("matches")
        if type(matches) is not list or len(matches) > _OPENAI_CLIENT_TOOL_SEARCH_MAX_RESULTS:
            raise OpenAIProtocolError(
                "search_tools result matches must be a bounded list.",
                reason_code="search_tools_result_matches_must_be_a_bounded_list",
            )
        names: set[str] = set()
        for index, match in enumerate(matches):
            if type(match) is not dict or set(match) != {
                "tool_ref",
                "tool_id",
                "name",
                "description",
                "input_schema",
                "descriptor_version",
                "schema_fingerprint",
                "readiness",
            }:
                raise OpenAIProtocolError(
                    f"search_tools result match {index} is not canonical.",
                    reason_code="search_tools_result_match_is_not_canonical",
                )
            match = cast("dict[str, Any]", match)
            name = match.get("name")
            if not isinstance(name, str) or name in names:
                raise OpenAIProtocolError(
                    "search_tools result tool names must be unique.",
                    reason_code="search_tools_result_tool_names_must_be_unique",
                )
            if match.get("readiness") != "registered":
                raise OpenAIProtocolError(
                    "search_tools result contains an unready tool.",
                    reason_code="search_tools_result_contains_an_unready_tool",
                )
            names.add(name)
            # Search results are durable transcript evidence, but callable
            # authority belongs to the current branch-local discovery view. A
            # fork starts with an empty view, so inherited matches must replay as
            # an empty client Tool Search output instead of reloading parent tools.
            trusted_tool = trusted_tools.get(name)
            if trusted_tool is None:
                continue
            if (
                match.get("description") != trusted_tool["description"]
                or match.get("input_schema") != trusted_tool["input_schema"]
            ):
                raise OpenAIProtocolError(
                    "search_tools result conflicts with the trusted loaded definition.",
                    reason_code="search_tools_result_conflicts_with_the_trusted_loaded_definition",
                )
            tools.append(_openai_tool(trusted_tool))
    return {
        "type": "tool_search_output",
        "call_id": part.tool_call_id,
        "execution": "client",
        "tools": tools,
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
        raise OpenAIProtocolError(
            "Tool messages can only contain tool_result parts.",
            reason_code="tool_messages_can_only_contain_tool_result_parts",
        )
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
            raise OpenAIProtocolError(
                f"Missing resolved file attachment: {attachment.artifact_id}",
                reason_code="missing_resolved_file_attachment",
            )
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
    raise OpenAIProtocolError(
        f"Unsupported file attachment kind: {kind!r}",
        reason_code="unsupported_file_attachment_kind",
    )


def _json_arguments(arguments: Mapping[str, Any]) -> str:
    copied = copy_json_value(arguments, "arguments")
    if type(copied) is not dict:
        raise OpenAIProtocolError(
            "Tool call arguments must be an object.",
            reason_code="tool_call_arguments_must_be_an_object",
        )
    return json.dumps(copied, sort_keys=True, separators=(",", ":"))


def _openai_tool(tool: Mapping[str, Any]) -> dict[str, Any]:
    name = _openai_tool_name(tool)
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


def _openai_request_tools(request: ModelRequest) -> list[dict[str, Any]]:
    projection = request.tool_discovery_projection
    if projection is None:
        return [_openai_tool(tool) for tool in request.tools]
    for name in (*projection.loaded_tool_names, *projection.candidate_tool_names):
        _validate_openai_tool_name(name)
    cache_anchor_name = targeted_tool_native_cache_anchor_name(request.options)
    search_indexes = [
        index
        for index, tool in enumerate(request.tools)
        if tool.get("name") == _CAYU_SEARCH_TOOLS_NAME
    ]
    if len(search_indexes) != 1:
        raise ValueError("OpenAI Tool Search requires one exact search_tools definition.")
    projected: list[dict[str, Any]] = []
    for index, tool in enumerate(request.tools):
        name = tool.get("name")
        if index == search_indexes[0]:
            if projection.protocol == OPENAI_CLIENT_TOOL_SEARCH_PROTOCOL:
                function = _openai_tool(tool)
                projected.append(
                    {
                        "type": "tool_search",
                        "execution": "client",
                        "description": _OPENAI_CLIENT_TOOL_SEARCH_DESCRIPTION,
                        "parameters": function["parameters"],
                    }
                )
            continue
        if (
            name == _CAYU_CALL_TOOL_NAME
            and cache_anchor_name != _CAYU_CALL_TOOL_NAME
            and not call_tool_core_callable(request.options)
        ):
            continue
        projected.append(_openai_tool(tool))
    if projection.protocol == OPENAI_HOSTED_TOOL_SEARCH_PROTOCOL:
        direct_names = {
            cast("str", tool["name"])
            for tool in projected
            if tool.get("type") == "function" and type(tool.get("name")) is str
        }
        overlap = sorted(direct_names & set(projection.candidate_tool_names))
        if overlap:
            raise ValueError(
                "Hosted Tool Search candidates cannot also be direct request tools: "
                + ", ".join(overlap)
            )
        projected.extend(
            {
                **_openai_tool(tool),
                "defer_loading": True,
            }
            for tool in projection.candidate_tools
        )
        if projection.candidate_tools:
            projected.append(
                {
                    "type": "tool_search",
                    "execution": "server",
                }
            )
    return projected


def _openai_tool_name(tool: Mapping[str, Any]) -> str:
    if not isinstance(tool, Mapping):
        raise ValueError("Tool definitions must be objects.")
    name = _require_mapping_string(tool, "name")
    _validate_openai_tool_name(name)
    return name


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
        raise OpenAIProtocolError(
            f"OpenAI response {key} must be a string.",
            reason_code="response_field_must_be_a_string",
        )
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
