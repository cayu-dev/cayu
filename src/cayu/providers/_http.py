"""Shared HTTP transport plumbing for provider adapters.

The provider transports (OpenAI, Anthropic, Chat Completions, Vertex) share
identical httpx POST/stream mechanics: certifi-backed TLS, URL validation,
error-body sanitizing, and SSE decoding. This module holds that plumbing once;
each adapter keeps only its provider-specific error classification and shaping.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import ssl
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, aclosing, suppress
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlparse

import certifi
import httpx

from cayu._exception_state import exception_state_contains
from cayu._validation import require_clean_nonblank, require_nonblank
from cayu.providers._credential_boundary import (
    ProviderStreamCleanupError,
    aclosing_provider_stream,
    credential_safe_provider_cancellation,
    provider_cancellation_failures,
    stream_cleanup_cancelled_after_provider_failure,
)
from cayu.providers._sse import (
    DEFAULT_SSE_MAX_EVENT_BYTES,
    SseEventLimitError,
    SseEventTimeoutError,
    _aiter_bounded_sse_lines,
    aiter_sse_json_events,
)
from cayu.providers.base import (
    ModelContextOverflowError,
    ModelProviderError,
    ModelStreamDeadlineError,
    ModelStreamEvent,
    ModelStreamEventType,
)
from cayu.providers.deadlines import (
    ProviderDeadlineKind,
    ProviderStreamDeadlineController,
    ProviderStreamDeadlineExceeded,
    ProviderStreamDeadlines,
    current_provider_deadline_controller,
)
from cayu.vaults.redaction import SecretRedactor

MAX_PROVIDER_ERROR_BODY_CHARS = 2_000
MAX_PROVIDER_ERROR_BODY_BYTES = 64 * 1024
OMITTED_PROVIDER_ERROR_BODY = "[provider response body omitted]"
_PROVIDER_CA_BUNDLE_ENV = "CAYU_PROVIDER_CA_BUNDLE"
_ApiErrorFromResponse = Callable[[httpx.Response, str, float | None], Exception]
_RaiseContextOverflowFromStatus = Callable[[int], None]


class _TrustedSseJsonEvent(dict[str, Any]):
    """Decoded SSE dict carrying Cayu-owned HTTP response metadata."""

    def __init__(self, event: Mapping[str, Any], *, retry_after_s: float | None) -> None:
        super().__init__(event)
        self._retry_after_s = retry_after_s


def _trusted_sse_retry_after_s(event: Mapping[str, Any]) -> float | None:
    """Return response-header delay only for the exact internal SSE envelope."""
    if type(event) is not _TrustedSseJsonEvent:
        return None
    return event._retry_after_s


def _identity_sse_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Override content negotiation so SSE bounds always see identity bytes."""
    identity_headers = {
        name: value for name, value in headers.items() if name.lower() != "accept-encoding"
    }
    identity_headers["Accept-Encoding"] = "identity"
    return identity_headers


def _response_uses_identity_encoding(response: httpx.Response) -> bool:
    content_encoding = response.headers.get("content-encoding")
    return content_encoding is None or content_encoding.strip().lower() == "identity"


async def _read_bounded_identity_error_response(
    response: httpx.Response,
    *,
    idle_timeout_s: float,
    max_duration_s: float,
) -> httpx.Response | None:
    """Read a small identity body, or leave provider classification status-only."""

    content_length = response.headers.get("content-length")
    if content_length is not None:
        try:
            declared_bytes = int(content_length)
        except ValueError:
            declared_bytes = -1
        if declared_bytes > MAX_PROVIDER_ERROR_BODY_BYTES:
            return None

    # Preserve compatibility with bounded, already-buffered responses supplied
    # by custom clients and test doubles. Bundled ``AsyncClient.stream``
    # responses reach the incremental path below instead.
    if response.is_stream_consumed:
        try:
            buffered_body = response.content
        except httpx.ResponseNotRead:
            return None
        if len(buffered_body) > MAX_PROVIDER_ERROR_BODY_BYTES:
            return None
        return httpx.Response(
            status_code=response.status_code,
            headers=response.headers,
            content=buffered_body,
        )

    body = bytearray()
    iterator = _aiter_unclosed_response_bytes(response)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max_duration_s
    while True:
        remaining = min(idle_timeout_s, deadline - loop.time())
        if remaining <= 0:
            return None
        try:
            async with asyncio.timeout(remaining):
                chunk = await iterator.__anext__()
        except StopAsyncIteration:
            break
        except TimeoutError:
            return None
        except httpx.RequestError:
            # The response status is already authoritative. A body read
            # failure must not replace (for example) HTTP 401 with a retryable
            # connection classification.
            return None
        if len(body) + len(chunk) > MAX_PROVIDER_ERROR_BODY_BYTES:
            return None
        body.extend(chunk)
    return httpx.Response(
        status_code=response.status_code,
        headers=response.headers,
        content=bytes(body),
    )


async def _aiter_unclosed_response_bytes(
    response: httpx.Response,
) -> AsyncGenerator[bytes, None]:
    """Yield raw identity bytes while leaving closure to the owned context."""

    if response.is_stream_consumed:
        raise httpx.StreamConsumed()
    if response.is_closed:
        raise httpx.StreamClosed()
    if not isinstance(response.stream, httpx.AsyncByteStream):
        raise RuntimeError("Attempted to call an async iterator on a sync stream.")
    response.is_stream_consumed = True
    try:
        async for chunk in response.stream:
            yield chunk
    except httpx.RequestError as exc:
        # Match ``Response.aiter_raw()`` by retaining request context on read
        # failures without inheriting its implicit response-close behavior.
        with suppress(RuntimeError):
            exc.request = response.request
        raise


async def _aiter_owned_stream_response(
    response_context: AbstractAsyncContextManager[httpx.Response],
) -> AsyncIterator[httpx.Response]:
    """Yield one HTTP response whose context exit is owned by stream cleanup."""

    async with response_context as response:
        yield response


_TRUSTED_HTTPX_REQUEST_ERROR_TYPES: dict[type[httpx.RequestError], str] = {
    httpx.CloseError: "CloseError",
    httpx.ConnectError: "ConnectError",
    httpx.ConnectTimeout: "ConnectTimeout",
    httpx.DecodingError: "DecodingError",
    httpx.LocalProtocolError: "LocalProtocolError",
    httpx.NetworkError: "NetworkError",
    httpx.PoolTimeout: "PoolTimeout",
    httpx.ProtocolError: "ProtocolError",
    httpx.ProxyError: "ProxyError",
    httpx.ReadError: "ReadError",
    httpx.ReadTimeout: "ReadTimeout",
    httpx.RemoteProtocolError: "RemoteProtocolError",
    httpx.RequestError: "RequestError",
    httpx.TimeoutException: "TimeoutException",
    httpx.TooManyRedirects: "TooManyRedirects",
    httpx.TransportError: "TransportError",
    httpx.UnsupportedProtocol: "UnsupportedProtocol",
    httpx.WriteError: "WriteError",
    httpx.WriteTimeout: "WriteTimeout",
}
_SAFE_INTERNAL_PROVIDER_ERROR_TYPES = frozenset(
    {
        *_TRUSTED_HTTPX_REQUEST_ERROR_TYPES.values(),
        "ProviderStreamCleanupError",
        "SseEventLimitError",
        "SseEventTimeoutError",
        "ModelStreamDeadlineError",
    }
)
_SAFE_PROVIDER_ERROR_TYPES = {
    "anthropic": frozenset(
        {
            "api_error",
            "authentication_error",
            "billing_error",
            "conflict_error",
            "invalid_request_error",
            "not_found_error",
            "overloaded_error",
            "permission_error",
            "rate_limit_error",
            "request_too_large",
            "timeout_error",
        }
    ),
    "chat_completions": frozenset(
        {
            "authentication_error",
            "context_length_exceeded",
            "error",
            "invalid_request_error",
            "not_found_error",
            "permission_error",
            "rate_limit_error",
            "server_error",
        }
    ),
    "openai": frozenset(
        {
            "authentication_error",
            "context_length_exceeded",
            "error",
            "invalid_request_error",
            "not_found_error",
            "permission_error",
            "rate_limit_error",
            "server_error",
        }
    ),
    "vertex": frozenset(
        {
            "DEADLINE_EXCEEDED",
            "INTERNAL",
            "INVALID_ARGUMENT",
            "PERMISSION_DENIED",
            "RESOURCE_EXHAUSTED",
            "UNAUTHENTICATED",
            "UNAVAILABLE",
            "api_error",
            "authentication_error",
            "billing_error",
            "conflict_error",
            "invalid_request_error",
            "not_found_error",
            "overloaded_error",
            "permission_error",
            "rate_limit_error",
            "request_too_large",
            "timeout_error",
        }
    ),
}
_SAFE_PROVIDER_ERROR_CODES = {
    "anthropic": frozenset(
        {
            "context_length_exceeded",
            "rate_limit_exceeded",
        }
    ),
    "chat_completions": frozenset(
        {
            "context_length_exceeded",
            "internal_error",
            "rate_limit_exceeded",
            "server_error",
        }
    ),
    "openai": frozenset(
        {
            "bad_request",
            "context_length_exceeded",
            "internal_error",
            "previous_response_not_found",
            "rate_limit_exceeded",
            "server_error",
        }
    ),
    "vertex": frozenset(
        {
            "context_length_exceeded",
            "rate_limit_exceeded",
        }
    ),
}
_SAFE_PROVIDER_EXCEPTION_TYPE_NAMES = frozenset(
    {
        "AnthropicAPIError",
        "AnthropicContextOverflowError",
        "AnthropicError",
        "AnthropicProtocolError",
        "ChatCompletionsAPIError",
        "ChatCompletionsContextOverflowError",
        "ChatCompletionsError",
        "ChatCompletionsProtocolError",
        "Exception",
        "ModelContextOverflowError",
        "ModelProviderError",
        "ModelStreamDeadlineError",
        "OpenAIAPIError",
        "OpenAIContextOverflowError",
        "OpenAIError",
        "OpenAIProtocolError",
        "OpenAISubscriptionAuthError",
        "ProviderStreamCleanupError",
        "RuntimeError",
        "SseEventLimitError",
        "SseEventTimeoutError",
        "VertexAPIError",
        "VertexContextOverflowError",
        "VertexError",
        "VertexProtocolError",
    }
)


def new_async_client() -> httpx.AsyncClient:
    """Build an httpx.AsyncClient with explicit, optionally augmented CA trust.

    Per-request timeouts are passed at call time (see :func:`post_json` and
    :func:`stream_sse_json_events`), so the client itself carries no fixed
    timeout and can be reused across both blocking and streaming requests.

    Public roots always come from certifi. A trusted runtime may additionally
    set ``CAYU_PROVIDER_CA_BUNDLE`` to a PEM bundle for a private provider or
    virtual-egress broker. That bundle augments certifi rather than replacing
    it, and invalid configuration fails while the lazy client is created.
    """
    context = ssl.create_default_context(cafile=certifi.where())
    extra_ca_bundle = os.environ.get(_PROVIDER_CA_BUNDLE_ENV)
    if extra_ca_bundle is not None:
        context.load_verify_locations(
            cafile=require_nonblank(extra_ca_bundle, _PROVIDER_CA_BUNDLE_ENV)
        )
    return httpx.AsyncClient(verify=context)


class SharedAsyncClient:
    """One lazily-created httpx.AsyncClient reused across a transport's requests.

    Constructing a fresh ``httpx.AsyncClient`` per model request performs a full
    TLS handshake and throws away the connection pool every time. A provider
    transport keeps one ``SharedAsyncClient`` for its lifetime instead, so
    keep-alive connections are reused across requests, and closes it via
    :meth:`aclose`. The client is created lazily on first use, so constructing a
    provider never opens sockets; a client closed out from under the transport
    (e.g. after ``aclose``) is transparently recreated on the next request.
    """

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    def get(self) -> httpx.AsyncClient:
        client = self._client
        if client is None or client.is_closed:
            client = new_async_client()
            self._client = client
        return client

    async def aclose(self) -> None:
        client = self._client
        self._client = None
        if client is not None and not client.is_closed:
            await client.aclose()


async def aclose_transport(transport: object) -> None:
    """Close a provider transport's shared HTTP client if it exposes ``aclose``.

    Injected custom transports need not own an httpx client, so a transport
    without an ``aclose`` method is a no-op rather than an error.
    """
    aclose = getattr(transport, "aclose", None)
    if aclose is not None:
        await aclose()


async def post_json(
    *,
    client: httpx.AsyncClient,
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout_s: float,
    request_label: str,
    response_label: str,
    api_error: Callable[..., Exception],
    protocol_error: type[Exception],
    error_response_text: Callable[[httpx.Response], str],
    raise_context_overflow: Callable[[httpx.Response], None] | None = None,
    api_error_from_response: _ApiErrorFromResponse | None = None,
) -> Mapping[str, Any]:
    """POST a JSON payload and return the decoded JSON object response.

    The caller-owned ``client`` is reused (its connection pool is kept warm)
    rather than opening a fresh TLS connection per request. HTTP failures raise
    ``api_error`` (after ``raise_context_overflow`` gets a chance to classify
    them); non-object response bodies raise ``protocol_error``.
    ``request_label``/``response_label`` prefix messages (e.g. ``"OpenAI API"``
    / ``"OpenAI"``). ``api_error_from_response`` lets an adapter build a
    structured error (typed status/code fields) from the HTTP error response;
    the shared layer supplies its parsed ``Retry-After`` delay.
    """
    return await request_json(
        client=client,
        method="POST",
        url=url,
        headers=headers,
        payload=payload,
        timeout_s=timeout_s,
        request_label=request_label,
        response_label=response_label,
        api_error=api_error,
        protocol_error=protocol_error,
        error_response_text=error_response_text,
        raise_context_overflow=raise_context_overflow,
        api_error_from_response=api_error_from_response,
    )


async def request_json(
    *,
    client: httpx.AsyncClient,
    method: str,
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any] | None,
    timeout_s: float,
    request_label: str,
    response_label: str,
    api_error: Callable[..., Exception],
    protocol_error: type[Exception],
    error_response_text: Callable[[httpx.Response], str],
    raise_context_overflow: Callable[[httpx.Response], None] | None = None,
    api_error_from_response: _ApiErrorFromResponse | None = None,
) -> Mapping[str, Any]:
    """Send a bounded JSON request and return one decoded object response."""

    method = require_clean_nonblank(method, "method").upper()
    try:
        request_kwargs: dict[str, Any] = {
            "headers": dict(headers),
            "timeout": timeout_s,
        }
        if payload is not None:
            request_kwargs["json"] = dict(payload)
        if method == "POST":
            response = await client.post(url, **request_kwargs)
        elif method == "GET":
            response = await client.get(url, **request_kwargs)
        else:
            response = await client.request(method, url, **request_kwargs)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        if raise_context_overflow is not None:
            try:
                raise_context_overflow(exc.response)
            except ModelContextOverflowError as overflow:
                raise overflow from exc
        message = (
            f"{request_label} request failed with HTTP "
            f"{exc.response.status_code}: "
            f"{error_response_text(exc.response)}"
        )
        raise _response_api_error(
            exc.response,
            message,
            api_error=api_error,
            api_error_from_response=api_error_from_response,
        ) from exc
    except httpx.RequestError as exc:
        raise _request_api_error(
            api_error, request_label=request_label, url=url, cause=exc
        ) from exc

    try:
        decoded = response.json()
    except ValueError as exc:
        raise protocol_error(f"{response_label} response was not valid JSON.") from exc
    if not isinstance(decoded, Mapping):
        raise protocol_error(f"{response_label} response must be a JSON object.")
    return decoded


async def stream_sse_json_events(
    *,
    client: httpx.AsyncClient,
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout_s: float,
    transport_idle_timeout_s: float,
    protocol_idle_timeout_s: float,
    semantic_progress_timeout_s: float,
    absolute_stream_timeout_s: float,
    request_label: str,
    response_label: str,
    api_error: Callable[..., Exception],
    protocol_error: type[Exception],
    error_response_text: Callable[[httpx.Response], str],
    raise_context_overflow: Callable[[httpx.Response], None] | None = None,
    raise_context_overflow_from_status: _RaiseContextOverflowFromStatus | None = None,
    api_error_from_response: _ApiErrorFromResponse | None = None,
    method: str = "POST",
) -> AsyncIterator[Mapping[str, Any]]:
    """Send a streaming JSON request and yield decoded SSE data objects.

    The caller-owned ``client`` is reused across requests; only the streaming
    response is opened and closed per call. ``api_error_from_response`` mirrors
    :func:`post_json`: adapters can build a structured error (typed status/code
    fields) while the shared layer supplies the parsed ``Retry-After`` delay.
    ``raise_context_overflow_from_status`` is reserved for provider statuses
    that are authoritative when the response body cannot be read safely.
    """
    method = require_clean_nonblank(method, "method").upper()
    deadlines = ProviderStreamDeadlines(
        transport_idle_timeout_s=transport_idle_timeout_s,
        protocol_idle_timeout_s=protocol_idle_timeout_s,
        semantic_progress_timeout_s=semantic_progress_timeout_s,
        absolute_stream_timeout_s=absolute_stream_timeout_s,
    )
    deadline_controller = current_provider_deadline_controller()
    owns_deadline_controller = deadline_controller is None
    if deadline_controller is None:
        deadline_controller = ProviderStreamDeadlineController(deadlines)
    elif deadline_controller.deadlines != deadlines:
        raise ValueError("Provider transport deadline policy changed after dispatch admission.")
    error_body_idle_timeout_s = deadline_controller.deadlines.transport_idle_timeout_s
    request_timeout_s = float(timeout_s)
    error_body_max_duration_s = (
        min(request_timeout_s, error_body_idle_timeout_s * 2)
        if math.isfinite(request_timeout_s) and request_timeout_s > 0
        else error_body_idle_timeout_s * 2
    )
    timeout = httpx.Timeout(timeout_s, read=None)
    successful_response_established = False
    try:
        request_kwargs: dict[str, Any] = {
            "headers": _identity_sse_headers(headers),
            "timeout": timeout,
        }
        if method != "GET":
            request_kwargs["json"] = dict(payload)
        response_context = client.stream(method, url, **request_kwargs)
        responses = _aiter_owned_stream_response(response_context)
        async with aclosing_provider_stream(responses):
            response = await deadline_controller.wait_for(
                anext(responses),
                kinds=(
                    ProviderDeadlineKind.TRANSPORT_IDLE,
                    ProviderDeadlineKind.PROTOCOL_IDLE,
                    ProviderDeadlineKind.ABSOLUTE,
                ),
            )
            deadline_controller.observe_transport()
            successful_response_established = response.status_code < 400
            if response.status_code >= 400:
                identity_encoding = _response_uses_identity_encoding(response)
                error_response: httpx.Response | None = None
                if identity_encoding:
                    # Preserve structured identity only when the complete body
                    # arrives within fixed byte, idle, and duration ceilings.
                    error_response = await _read_bounded_identity_error_response(
                        response,
                        idle_timeout_s=error_body_idle_timeout_s,
                        max_duration_s=error_body_max_duration_s,
                    )
                    if error_response is not None and raise_context_overflow is not None:
                        raise_context_overflow(error_response)
                if error_response is None and raise_context_overflow_from_status is not None:
                    # Only classifiers explicitly wired for status-only
                    # evidence may run here. Unsupported, oversized, or stalled
                    # bodies remain undecoded, so body-dependent identities fail
                    # closed.
                    raise_context_overflow_from_status(response.status_code)
                message = (
                    f"{request_label} request failed with HTTP "
                    f"{response.status_code}: "
                    f"{error_response_text(error_response) if error_response is not None else OMITTED_PROVIDER_ERROR_BODY}"
                )
                if error_response is None:
                    # The HTTP status and Retry-After header are authoritative
                    # even when a provider or intermediary supplied a body Cayu
                    # could not safely read. Do not decode that body.
                    raise api_error(
                        message,
                        status_code=response.status_code,
                        retry_after_s=retry_after_seconds(response),
                    )
                raise _response_api_error(
                    error_response,
                    message,
                    api_error=api_error,
                    api_error_from_response=api_error_from_response,
                )
            if not _response_uses_identity_encoding(response):
                raise protocol_error(
                    f"{response_label} SSE response used unsupported content encoding."
                )
            retry_after_s = retry_after_seconds(response)
            # ``Response.aiter_raw()`` auto-closes on clean EOF, which would
            # run arbitrary transport cleanup inside the read iterator instead
            # of the response-context owner surrounding this block.
            response_bytes = _aiter_unclosed_response_bytes(response)
            async with aclosing(response_bytes):
                bounded_lines = _aiter_bounded_sse_lines(
                    response_bytes,
                    max_line_bytes=DEFAULT_SSE_MAX_EVENT_BYTES,
                    provider_label=response_label,
                    emit_byte_activity=True,
                    deadline_controller=deadline_controller,
                )
                async for event in aiter_sse_json_events(
                    bounded_lines,
                    deadline_controller=deadline_controller,
                    provider_label=response_label,
                    protocol_error=protocol_error,
                ):
                    yield _TrustedSseJsonEvent(event, retry_after_s=retry_after_s)
    except ProviderStreamDeadlineExceeded as exc:
        raise ModelStreamDeadlineError(
            provider=response_label.lower().replace(" ", "_"),
            evidence=exc.evidence,
            stream_cleanup_failed=exc.stream_cleanup_failed,
        ) from None
    except SseEventTimeoutError as exc:
        raise api_error(
            str(exc),
            error_type=type(exc).__name__,
            retryable=True,
        ) from exc
    except SseEventLimitError as exc:
        raise api_error(
            str(exc),
            error_type=type(exc).__name__,
            retryable=False,
        ) from exc
    except httpx.RequestError as exc:
        raise _request_api_error(
            api_error,
            request_label=request_label,
            url=url,
            cause=exc,
            retryable=False if successful_response_established else None,
        ) from exc
    finally:
        if owns_deadline_controller:
            deadline_controller.close()


def _request_api_error(
    api_error: Callable[..., Exception],
    *,
    request_label: str,
    url: str,
    cause: httpx.RequestError,
    retryable: bool | None = None,
) -> Exception:
    return api_error(
        f"{request_label} request failed for {url}: {cause}",
        error_type=_trusted_httpx_request_error_type(cause),
        retryable=(_is_retryable_transport_error(cause) if retryable is None else retryable),
    )


def _trusted_httpx_request_error_type(error: httpx.RequestError) -> str:
    """Return a fixed classification without exposing arbitrary subclass names."""

    return _TRUSTED_HTTPX_REQUEST_ERROR_TYPES.get(type(error), "RequestError")


def _response_api_error(
    response: httpx.Response,
    message: str,
    *,
    api_error: Callable[..., Exception],
    api_error_from_response: _ApiErrorFromResponse | None,
) -> Exception:
    retry_after_s = retry_after_seconds(response)
    if api_error_from_response is not None:
        return api_error_from_response(response, message, retry_after_s)
    return api_error(
        message,
        status_code=response.status_code,
        retry_after_s=retry_after_s,
    )


def _is_retryable_transport_error(exc: httpx.RequestError) -> bool:
    # Local protocol and proxy failures usually require request/configuration changes;
    # only failures that can plausibly succeed unchanged are retried automatically.
    return isinstance(
        exc,
        (
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.RemoteProtocolError,
        ),
    )


def retry_after_seconds(
    response: httpx.Response,
    *,
    now: datetime | None = None,
) -> float | None:
    """Parse Retry-After delta-seconds or an HTTP-date into a non-negative delay."""
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        try:
            target = parsedate_to_datetime(raw)
        except (TypeError, ValueError):
            return None
        if target is None:
            return None
        if target.tzinfo is None:
            target = target.replace(tzinfo=UTC)
        current = datetime.now(UTC) if now is None else now
        if current.tzinfo is None:
            current = current.replace(tzinfo=UTC)
        return max(0.0, (target - current).total_seconds())
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def validate_url(
    url: str,
    field_name: str,
    *,
    provider_label: str,
    allow_http: bool = False,
    allow_http_hint: bool = False,
) -> str:
    """Require an absolute URL; https-only unless ``allow_http`` opts in.

    ``allow_http_hint`` mentions the ``allow_http=True`` opt-in in the rejection
    message for providers that expose that switch.
    """
    value = require_clean_nonblank(url, field_name)
    parsed = urlparse(value)
    allowed_schemes = {"https", "http"} if allow_http else {"https"}
    if parsed.scheme not in allowed_schemes:
        suffix = (
            " (set allow_http=True for local http servers)"
            if allow_http_hint and not allow_http
            else ""
        )
        raise ValueError(
            f"{provider_label} {field_name} must use "
            f"{' or '.join(sorted(allowed_schemes))}{suffix}."
        )
    if not parsed.netloc:
        raise ValueError(f"{provider_label} {field_name} must include a host.")
    return value


def validate_base_url(
    base_url: str,
    *,
    provider_label: str,
    allow_http: bool = False,
    allow_http_hint: bool = False,
) -> str:
    return validate_url(
        base_url,
        "base_url",
        provider_label=provider_label,
        allow_http=allow_http,
        allow_http_hint=allow_http_hint,
    ).rstrip("/")


def copy_headers(headers: Mapping[str, str] | None, *, protected: set[str]) -> dict[str, str]:
    """Copy caller-supplied extra headers, rejecting the protected names."""
    if headers is None:
        return {}
    copied: dict[str, str] = {}
    for key, value in headers.items():
        header_name = require_clean_nonblank(key, "header name")
        if header_name.lower() in protected:
            raise ValueError(f"extra_headers cannot override {header_name}.")
        copied[header_name] = require_nonblank(value, f"header {key}")
    return copied


def safe_error_response_text(
    response: httpx.Response,
    *,
    format_error_json: Callable[[Any], str | None],
) -> str:
    """Omit an untrusted body when the active workload redactor is unavailable.

    Provider adapters run below the application-owned workload-secret scope.
    Retaining even a parsed or truncated body here could preserve a recoverable
    secret fragment before the application gets a chance to redact it. Typed
    provider exceptions retain authoritative HTTP status/type/code fields.
    """

    del response, format_error_json
    return OMITTED_PROVIDER_ERROR_BODY


def safe_error_json(decoded: Mapping[str, Any], *, include_request_id: bool = False) -> str:
    """Sanitize an OpenAI-shaped ``{"error": {...}}`` body to safe flat fields."""
    error = decoded.get("error")
    request_id = decoded.get("request_id") if include_request_id else None
    if isinstance(error, Mapping):
        safe_error = safe_flat_error_json(error)
        if isinstance(request_id, str):
            safe_error["request_id"] = request_id
        if safe_error:
            return json_error_text(safe_error)
    safe_error = safe_flat_error_json(decoded)
    if safe_error:
        return json_error_text(safe_error)
    return truncate_error_text(json_error_text(dict(decoded)))


def safe_flat_error_json(error: Mapping[str, Any]) -> dict[str, str]:
    error_type = error.get("type")
    message = error.get("message")
    code = error.get("code")
    safe_error: dict[str, str] = {}
    if isinstance(error_type, str):
        safe_error["type"] = error_type
    if isinstance(code, str):
        safe_error["code"] = code
    if isinstance(message, str):
        safe_error["message"] = truncate_error_text(message)
    return safe_error


def response_json_object(response: httpx.Response) -> Mapping[str, Any] | None:
    content_type = response.headers.get("content-type", "")
    if "application/json" not in content_type:
        return None
    try:
        decoded = response.json()
    except ValueError:
        return None
    if not isinstance(decoded, Mapping):
        return None
    return decoded


def optional_error_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def json_error_text(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    except TypeError:
        return str(value)


def truncate_error_text(value: str) -> str:
    if len(value) <= MAX_PROVIDER_ERROR_BODY_CHARS:
        return value
    return value[:MAX_PROVIDER_ERROR_BODY_CHARS] + "... [truncated]"


def exception_message(exc: Exception, *, provider_label: str) -> str:
    message = str(exc).strip()
    if message:
        return message
    return f"{type(exc).__name__}: {provider_label} provider failed"


def credential_sanitization_values(
    *credential_values: str | None,
    extra_headers: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Return every request credential value that must be removed from failures.

    Provider-specific ``extra_headers`` are caller-controlled and may carry an
    authorization token under an arbitrary header name. Treat every non-empty
    value as credential-bearing at the transport error boundary.
    """

    values = [
        value
        for value in (
            *credential_values,
            *((extra_headers or {}).values()),
        )
        if value
    ]
    return tuple(dict.fromkeys(values))


def credential_safe_error_event(
    exc: Exception,
    *,
    provider_label: str,
    provider_name: str,
    credential_values: Sequence[str],
    unresolved_message: str | None = None,
) -> ModelStreamEvent:
    """Build a typed provider error event with known credentials removed.

    Provider transports are an untrusted projection boundary: HTTP clients,
    proxy implementations, and test doubles may include an authorization value
    in their exception message or structured provider-error fields.  Keep the
    useful typed classification produced by ``ModelStreamEvent.error`` while
    redacting every known model-provider credential before the event can enter
    runtime persistence or diagnostics.
    """
    if type(exc) is ModelStreamDeadlineError:
        safe = ModelStreamDeadlineError(
            provider=provider_name,
            evidence=exc.deadline_evidence,
            stream_cleanup_failed=exc.stream_cleanup_failed,
        )
        return ModelStreamEvent.error(str(safe), cause=safe)
    safe_message = (
        require_nonblank(unresolved_message, "unresolved_message")
        if unresolved_message is not None
        else f"{safe_provider_exception_type_name(exc)}: {provider_label} provider failed"
    )
    if not credential_values:
        # Authentication may fail before a deferred credential resolver returns
        # the value needed for comparison.  In that case no untrusted exception
        # text or structured string field is safe to retain.
        payload: dict[str, Any] = {
            "error": safe_message,
            "error_type": safe_provider_exception_type_name(exc),
        }
        if isinstance(exc, ModelProviderError):
            if type(exc.status_code) is int:
                payload["status_code"] = exc.status_code
            if type(exc.retryable) is bool:
                payload["retryable"] = exc.retryable
            if type(exc.retry_after_s) in {int, float}:
                payload["retry_after_s"] = exc.retry_after_s
        if isinstance(exc, ProviderStreamCleanupError):
            payload["stream_cleanup_failed"] = True
        return ModelStreamEvent(type=ModelStreamEventType.ERROR, payload=payload)
    if isinstance(exc, ModelProviderError):
        safe_exception = credential_safe_provider_exception(
            exc,
            provider_label=provider_label,
            provider_name=provider_name,
            credential_values=credential_values,
            safe_message=safe_message,
        )
        event = ModelStreamEvent.error(
            safe_message,
            cause=safe_exception,
        )
        payload = dict(event.payload)
        payload["error_type"] = safe_provider_exception_type_name(exc)
        if isinstance(exc, ProviderStreamCleanupError):
            payload["stream_cleanup_failed"] = True
        event = ModelStreamEvent(type=event.type, payload=payload)
    else:
        event = ModelStreamEvent(
            type=ModelStreamEventType.ERROR,
            payload={
                "error": safe_message,
                "error_type": safe_provider_exception_type_name(exc),
            },
        )
    redacted = SecretRedactor(credential_values).redact_json_values(event.payload)
    if type(redacted) is not dict:  # pragma: no cover - SecretRedactor contract guard
        raise AssertionError("provider error payload redaction returned a non-object")
    return ModelStreamEvent(type=event.type, payload=redacted)


def credential_safe_post_completion_failure(
    exc: Exception,
    *,
    provider_label: str,
    provider_name: str,
    credential_values: Sequence[str],
    safe_message: str | None = None,
) -> ModelProviderError:
    """Detach a terminal failure raised after a provider completion event.

    The runtime owns durable completion publication and accounting once it has
    observed ``completed``. Propagate a credential-safe exception instead of a
    second stream event so that existing post-completion handling can preserve
    usage without authorizing another provider dispatch.
    """

    safe = credential_safe_provider_exception(
        exc,
        provider_label=provider_label,
        provider_name=provider_name,
        credential_values=credential_values,
        safe_message=safe_message,
    )
    if isinstance(safe, ProviderStreamCleanupError):
        return safe
    return ModelProviderError(
        str(safe),
        provider=provider_name,
        status_code=safe.status_code,
        error_type=safe.error_type,
        error_code=safe.error_code,
        request_id=safe.request_id,
        retryable=False,
        retry_after_s=safe.retry_after_s,
        response_body=None,
    )


def credential_safe_provider_exception(
    exc: Exception,
    *,
    provider_label: str,
    provider_name: str,
    credential_values: Sequence[str],
    safe_message: str | None = None,
) -> ModelProviderError:
    """Return a detached provider exception safe for public propagation."""

    provider_label = require_clean_nonblank(provider_label, "provider_label")
    provider_name = require_clean_nonblank(provider_name, "provider_name")
    redactor = SecretRedactor(credential_values)
    if safe_message is not None:
        message = require_nonblank(safe_message, "safe_message")
    elif isinstance(exc, ModelContextOverflowError):
        message = f"{provider_label} model context window exceeded"
    else:
        message = f"{safe_provider_exception_type_name(exc)}: {provider_label} provider failed"

    source = exc if isinstance(exc, ModelProviderError) else None
    string_fields: dict[str, str | None] = {
        "error_type": _safe_provider_identity_field(
            provider_name,
            "error_type",
            source.error_type if source is not None else None,
        ),
        "error_code": _safe_provider_identity_field(
            provider_name,
            "error_code",
            source.error_code if source is not None else None,
        ),
        # Provider request IDs are arbitrary provider-controlled strings. They
        # cannot be compared with a workload secret registry at this boundary.
        "request_id": None,
    }
    if redactor.has_values:
        string_fields = {
            name: redactor.redact_text(value) if value is not None else None
            for name, value in string_fields.items()
        }
    else:
        string_fields = {name: None for name in string_fields}

    common: dict[str, Any] = {
        "provider": provider_name,
        "status_code": source.status_code if source is not None else None,
        "error_type": string_fields["error_type"],
        "error_code": string_fields["error_code"],
        "request_id": string_fields["request_id"],
        "response_body": None,
    }
    if isinstance(exc, ModelContextOverflowError):
        return ModelContextOverflowError(message, **common)
    if type(exc) is ModelStreamDeadlineError:
        return ModelStreamDeadlineError(
            provider=provider_name,
            evidence=exc.deadline_evidence,
            stream_cleanup_failed=exc.stream_cleanup_failed,
        )
    if isinstance(exc, ProviderStreamCleanupError):
        return ProviderStreamCleanupError(
            message,
            **common,
            retryable=False,
            retry_after_s=source.retry_after_s if source is not None else None,
        )
    return ModelProviderError(
        message,
        **common,
        retryable=source.retryable if source is not None else None,
        retry_after_s=source.retry_after_s if source is not None else None,
    )


def _safe_provider_identity_field(
    provider_name: str,
    field_name: str,
    value: object,
) -> str | None:
    if type(value) is not str:
        return None
    if field_name == "error_type":
        allowed = (
            _SAFE_PROVIDER_ERROR_TYPES.get(
                provider_name,
                frozenset(),
            )
            | _SAFE_INTERNAL_PROVIDER_ERROR_TYPES
        )
    else:
        allowed = _SAFE_PROVIDER_ERROR_CODES.get(provider_name, frozenset())
    return value if value in allowed else None


def safe_provider_exception_type_name(error: BaseException) -> str:
    """Return only fixed provider-boundary exception classifications."""

    try:
        name = type.__getattribute__(type(error), "__name__")
    except BaseException:
        return "Exception"
    return (
        name if type(name) is str and name in _SAFE_PROVIDER_EXCEPTION_TYPE_NAMES else "Exception"
    )


def sanitize_provider_cancellation(
    exc: asyncio.CancelledError,
    *,
    provider_label: str,
    credential_values: Sequence[str],
    safe_message: str | None = None,
) -> asyncio.CancelledError:
    """Return a fresh detached cancellation safe for public propagation."""

    provider_label = require_clean_nonblank(provider_label, "provider_label")
    del credential_values
    had_artifacts = exception_state_contains(exc, "artifacts")
    message = (
        require_nonblank(safe_message, "safe_message")
        if safe_message is not None
        else f"{provider_label} provider request cancelled"
    )
    safe = credential_safe_provider_cancellation(
        message,
        preserve_empty_artifacts=had_artifacts,
        stream_cleanup_cancelled_after_failure=(
            stream_cleanup_cancelled_after_provider_failure(exc)
        ),
        provider_cancellation_failures=provider_cancellation_failures(exc),
    )
    safe.__cause__ = None
    safe.__context__ = None
    return safe


__all__ = [
    "MAX_PROVIDER_ERROR_BODY_BYTES",
    "MAX_PROVIDER_ERROR_BODY_CHARS",
    "OMITTED_PROVIDER_ERROR_BODY",
    "SharedAsyncClient",
    "aclose_transport",
    "copy_headers",
    "credential_safe_error_event",
    "credential_safe_post_completion_failure",
    "credential_safe_provider_exception",
    "credential_sanitization_values",
    "exception_message",
    "json_error_text",
    "new_async_client",
    "optional_error_string",
    "post_json",
    "request_json",
    "response_json_object",
    "retry_after_seconds",
    "safe_error_json",
    "safe_error_response_text",
    "safe_flat_error_json",
    "safe_provider_exception_type_name",
    "sanitize_provider_cancellation",
    "stream_sse_json_events",
    "truncate_error_text",
    "validate_base_url",
    "validate_url",
]
