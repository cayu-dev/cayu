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
import re
import ssl
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, aclosing, suppress
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from http.cookies import CookieError, SimpleCookie
from typing import Any
from urllib.parse import parse_qsl, urlparse
from weakref import WeakKeyDictionary

import certifi
import httpx

from cayu._exception_state import exception_state_contains
from cayu._validation import require_clean_nonblank, require_nonblank
from cayu.providers._api_error_diagnostics import api_error_diagnostic_fields
from cayu.providers._credential_boundary import (
    ProviderStreamCleanupError,
    _contains_fatal_signal,
    _provider_stream_cleanup_error,
    _raise_detached_provider_stream_cleanup_error,
    aclosing_provider_stream,
    credential_safe_provider_cancellation,
    provider_cancellation_failures,
    stream_cleanup_cancelled_after_provider_failure,
)
from cayu.providers._openai_search_trace import ResponseStructureDiagnostic, ResponseStructureTrace
from cayu.providers._rejection_diagnostics import (
    project_rejection_response,
    rejection_fields,
    retain_rejection_diagnostic,
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
from cayu.providers.diagnostics import (
    _record_http_error,
    _record_stream_error,
    _record_transport_error,
)
from cayu.vaults.redaction import SecretRedactor

MAX_PROVIDER_ERROR_BODY_CHARS = 2_000
MAX_PROVIDER_ERROR_BODY_BYTES = 64 * 1024
OMITTED_PROVIDER_ERROR_BODY = "[provider response body omitted]"
# Public provider error messages are bounded after redaction, never before.
PUBLIC_PROVIDER_ERROR_MESSAGE_BYTES = 2_048
_PUBLIC_PROVIDER_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_PUBLIC_PROVIDER_REQUEST_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/+=-]{0,255}\Z")
_CREDENTIAL_NAME_PARTS = (
    "auth",
    "cookie",
    "credential",
    "key",
    "secret",
    "session",
    "signature",
    "token",
)
_PROVIDER_ERROR_WORKLOAD_REDACTOR: ContextVar[SecretRedactor | None] = ContextVar(
    "cayu_provider_error_workload_redactor", default=None
)
# Raw provider error text, keyed by the typed failure it explains. Kept outside
# the exception so its message, vars and traceback stay free of provider text.
_PROVIDER_ERROR_TEXT: WeakKeyDictionary[ModelProviderError, str] = WeakKeyDictionary()
_PROVIDER_ERROR_REQUEST_REDACTORS: WeakKeyDictionary[ModelProviderError, SecretRedactor] = (
    WeakKeyDictionary()
)
_PROVIDER_CA_BUNDLE_ENV = "CAYU_PROVIDER_CA_BUNDLE"
_POST_TERMINAL_DRAIN_SECONDS = 0.05
_ApiErrorFromResponse = Callable[[httpx.Response, str, float | None], Exception]
_RaiseContextOverflowFromStatus = Callable[[int], None]


class _TrustedJsonResponse(dict[str, Any]):
    """Decoded provider JSON with Cayu-owned request metadata outside wire fields."""

    def __init__(
        self,
        response: Mapping[str, Any],
        *,
        error_redactor: SecretRedactor | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(response)
        self._error_redactor = error_redactor
        self._request_id = request_id


class _TrustedSseJsonEvent(_TrustedJsonResponse):
    """Decoded SSE dict carrying Cayu-owned HTTP response metadata."""

    def __init__(self, event: Mapping[str, Any], *, retry_after_s: float | None) -> None:
        super().__init__(event)
        self._retry_after_s = retry_after_s
        self._response_structure: ResponseStructureDiagnostic | None = None
        self._terminal_accepted = False


def retain_provider_error_metadata(failure: Exception, event: Mapping[str, Any]) -> None:
    """Transfer trusted HTTP error metadata without retaining the raw JSON envelope."""

    if (type(event) is not _TrustedJsonResponse and type(event) is not _TrustedSseJsonEvent) or (
        not isinstance(failure, ModelProviderError)
    ):
        return
    if event._error_redactor is not None:
        _PROVIDER_ERROR_REQUEST_REDACTORS[failure] = event._error_redactor
    if (
        failure.request_id is None
        and event._request_id is not None
        and _PUBLIC_PROVIDER_REQUEST_ID.fullmatch(event._request_id) is not None
    ):
        failure.request_id = event._request_id


def _accept_sse_terminal(event: Mapping[str, Any]) -> None:
    """Acknowledge a terminal only after the provider parser validates it.

    Wire fields cannot set this flag. Raw transport consumers and custom event
    iterators retain their existing EOF contract.
    """
    if type(event) is _TrustedSseJsonEvent:
        event._terminal_accepted = True


def _trusted_sse_retry_after_s(event: Mapping[str, Any]) -> float | None:
    """Return response-header delay only for the exact internal SSE envelope."""
    if type(event) is not _TrustedSseJsonEvent:
        return None
    return event._retry_after_s


def _trusted_sse_response_structure(event: Mapping[str, Any]) -> object:
    return event._response_structure if type(event) is _TrustedSseJsonEvent else None


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
    *,
    terminal_accepted: Callable[[], bool] | None = None,
) -> AsyncGenerator[bytes, None]:
    """Yield raw identity bytes while leaving closure to the owned context."""

    if response.is_stream_consumed:
        raise httpx.StreamConsumed()
    if response.is_closed:
        raise httpx.StreamClosed()
    if not isinstance(response.stream, httpx.AsyncByteStream):
        raise RuntimeError("Attempted to call an async iterator on a sync stream.")
    response.is_stream_consumed = True
    iterator = response.stream.__aiter__()
    drain_deadline: float | None = None
    loop = asyncio.get_running_loop()
    try:
        while True:
            if drain_deadline is None and terminal_accepted is not None and terminal_accepted():
                drain_deadline = loop.time() + _POST_TERMINAL_DRAIN_SECONDS
            if drain_deadline is not None and loop.time() >= drain_deadline:
                return
            deadline = asyncio.timeout_at(drain_deadline) if drain_deadline is not None else None
            read_cancelled = False
            read_cleanup_failed = False
            try:
                if deadline is None:
                    chunk = await anext(iterator)
                else:
                    async with deadline:
                        try:
                            chunk = await anext(iterator)
                        except asyncio.CancelledError as exc:
                            # Observe the read outcome before asyncio converts its
                            # own cancellation to TimeoutError. A transport's own
                            # TimeoutError must never acquire drain-expiry authority.
                            read_cancelled = True
                            read_cleanup_failed = bool(provider_cancellation_failures(exc)) or (
                                stream_cleanup_cancelled_after_provider_failure(exc)
                            )
                            raise
            except StopAsyncIteration:
                return
            except TimeoutError:
                if deadline is not None and deadline.expired() and read_cancelled:
                    if read_cleanup_failed:
                        _raise_detached_provider_stream_cleanup_error(
                            _provider_stream_cleanup_error()
                        )
                    return
                raise
            yield chunk
            # Drain readily available tails across HTTP chunk boundaries. The
            # fixed deadline never refreshes for heartbeats or additional bytes.
            # Existing stream clocks and ordered response closure still apply.
    except httpx.RequestError as exc:
        # Match ``Response.aiter_raw()`` by retaining request context on read
        # failures without inheriting its implicit response-close behavior.
        with suppress(RuntimeError):
            exc.request = response.request
        raise


class _HttpResponseCloseTrace:
    """Observe HTTP-core closure when cancellation precedes response headers."""

    def __init__(self) -> None:
        self.succeeded: bool | None = None

    async def __call__(self, name: str, info: Mapping[str, Any]) -> None:
        del info
        if name in {"http11.response_closed.started", "http2.response_closed.started"}:
            self.succeeded = None
        elif name in {"http11.response_closed.complete", "http2.response_closed.complete"}:
            self.succeeded = True
        elif name in {"http11.response_closed.failed", "http2.response_closed.failed"}:
            self.succeeded = False


async def _aiter_owned_stream_response(
    response_context: AbstractAsyncContextManager[httpx.Response],
    deadline_controller: ProviderStreamDeadlineController,
    close_trace: _HttpResponseCloseTrace | None = None,
    *,
    terminal_accepted: Callable[[], bool] | None = None,
) -> AsyncIterator[tuple[httpx.Response, AsyncGenerator[bytes, None]]]:
    """Own bytes and response closure together, after the interrupted read joins."""

    opened = False
    succeeded = False
    fatal_failure = False
    try:
        async with response_context as response:
            opened = True
            response_bytes = _aiter_unclosed_response_bytes(
                response, terminal_accepted=terminal_accepted
            )
            async with aclosing(response_bytes):
                with suppress(GeneratorExit):
                    yield response, response_bytes
        succeeded = True
    except BaseException as failure:
        fatal_failure = _contains_fatal_signal(failure)
        raise
    finally:
        observer = deadline_controller._cleanup_observer
        if observer is not None and not fatal_failure:
            if opened:
                await observer.closed(succeeded=succeeded)
            elif close_trace is not None and close_trace.succeeded is not None:
                # A joined request is not itself proof of cleanup. Require the
                # HTTP layer's completed close operation before publishing.
                await observer.closed(succeeded=close_trace.succeeded)


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
        controller = current_provider_deadline_controller()
        if controller is not None:
            response = await _read_owned_json_response(
                client, method, url, request_kwargs, controller
            )
        elif method == "POST":
            response = await client.post(url, **request_kwargs)
        elif method == "GET":
            response = await client.get(url, **request_kwargs)
        else:
            response = await client.request(method, url, **request_kwargs)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        _record_http_error(exc.response, headers=headers, body_response=exc.response)
        if raise_context_overflow is not None:
            try:
                raise_context_overflow(exc.response)
            except ModelContextOverflowError as overflow:
                _PROVIDER_ERROR_REQUEST_REDACTORS[overflow] = request_credential_redactor(
                    exc.response
                )
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
        _record_transport_error(_trusted_httpx_request_error_type(exc), headers=headers)
        raise _request_api_error(
            api_error, request_label=request_label, url=url, cause=exc
        ) from exc

    try:
        decoded = response.json()
    except ValueError as exc:
        raise protocol_error(f"{response_label} response was not valid JSON.") from exc
    if not isinstance(decoded, Mapping):
        raise protocol_error(f"{response_label} response must be a JSON object.")
    return _TrustedJsonResponse(
        decoded,
        error_redactor=request_credential_redactor(response),
        request_id=_response_error_request_id(response),
    )


async def _read_owned_json_response(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    request_kwargs: dict[str, Any],
    controller: ProviderStreamDeadlineController,
) -> httpx.Response:
    """Retain final-JSON HTTP closure under the admitted provider operation.

    ``AsyncClient.request`` consumes and implicitly closes its response before
    returning it. That hides close outcome from the durable deadline observer.
    Reuse the streaming response owner while buffering the complete JSON body;
    the outer provider controller still owns the original wait and any retained
    cancellation. A local close receipt never certifies the remote operation.
    """
    close_trace = _HttpResponseCloseTrace()
    request_kwargs = {**request_kwargs, "extensions": {"trace": close_trace}}
    responses = _aiter_owned_stream_response(
        client.stream(method, url, **request_kwargs), controller, close_trace
    )
    async with aclosing_provider_stream(responses):
        response, chunks = await anext(responses)
        if response.is_stream_consumed:
            # Custom HTTPX transports may supply an already buffered response.
            # Its content is already decoded; do not decode it a second time.
            return response
        content = b"".join([chunk async for chunk in chunks])
        # HTTPX decodes Content-Encoding on the buffered copy, preserving final
        # JSON/error-body semantics without implicitly closing the live stream.
        return httpx.Response(
            response.status_code,
            headers=response.headers,
            content=content,
            request=response.request,
            extensions=response.extensions,
        )


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
    capture_response_structure: bool = False,
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
    response: httpx.Response | None = None
    try:
        request_kwargs: dict[str, Any] = {
            "headers": _identity_sse_headers(headers),
            "timeout": timeout,
        }
        if method != "GET":
            request_kwargs["json"] = dict(payload)
        close_trace = _HttpResponseCloseTrace()
        request_kwargs["extensions"] = {"trace": close_trace}
        response_context = client.stream(method, url, **request_kwargs)
        terminal_accepted = False
        responses = _aiter_owned_stream_response(
            response_context,
            deadline_controller,
            close_trace,
            terminal_accepted=lambda: terminal_accepted,
        )
        interrupted_read: asyncio.Future[Any] | None = None

        def retain_interrupted_read(operation: asyncio.Future[Any]) -> None:
            nonlocal interrupted_read
            interrupted_read = operation

        async with aclosing_provider_stream(
            responses,
            pending_read=lambda: interrupted_read,
            retain_cleanup=deadline_controller.retain_dispatched_operation,
            # Socket cancellation can require several event-loop turns.
            # Retain anything still pending after this local-close grace.
            cancellation_grace_s=0.05,
        ):
            response, response_bytes = await deadline_controller.wait_for(
                anext(responses),
                on_interrupted=retain_interrupted_read,
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
                _record_http_error(
                    response,
                    headers=headers,
                    body_response=error_response,
                )
                if error_response is not None and raise_context_overflow is not None:
                    try:
                        raise_context_overflow(error_response)
                    except ModelContextOverflowError as overflow:
                        _PROVIDER_ERROR_REQUEST_REDACTORS[overflow] = request_credential_redactor(
                            response
                        )
                        raise
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
                    raise _response_api_error(
                        httpx.Response(response.status_code, headers=response.headers),
                        message,
                        api_error=api_error,
                        api_error_from_response=None,
                        request_redactor=request_credential_redactor(response),
                    )
                raise _response_api_error(
                    error_response,
                    message,
                    api_error=api_error,
                    api_error_from_response=api_error_from_response,
                    request_redactor=request_credential_redactor(response),
                )
            if not _response_uses_identity_encoding(response):
                raise protocol_error(
                    f"{response_label} SSE response used unsupported content encoding."
                )
            retry_after_s = retry_after_seconds(response)
            # ``Response.aiter_raw()`` auto-closes on clean EOF, which would
            # run arbitrary transport cleanup inside the read iterator instead
            # of the response-context owner surrounding this block.
            bounded_lines = _aiter_bounded_sse_lines(
                response_bytes,
                max_line_bytes=DEFAULT_SSE_MAX_EVENT_BYTES,
                provider_label=response_label,
                emit_byte_activity=True,
                deadline_controller=deadline_controller,
            )
            structure = ResponseStructureTrace() if capture_response_structure else None
            error_redactor = request_credential_redactor(response)
            async for event in aiter_sse_json_events(
                bounded_lines,
                deadline_controller=deadline_controller,
                provider_label=response_label,
                protocol_error=protocol_error,
                on_interrupted=retain_interrupted_read,
            ):
                envelope = _TrustedSseJsonEvent(event, retry_after_s=retry_after_s)
                envelope._error_redactor = error_redactor
                envelope._request_id = response.headers.get("x-request-id") or response.headers.get(
                    "request-id"
                )
                if structure is not None:
                    structure.record(event)
                    envelope._response_structure = structure.snapshot()
                _record_stream_error(event, headers=headers, response=response)
                yield envelope
                terminal_accepted = terminal_accepted or envelope._terminal_accepted
    except ProviderStreamDeadlineExceeded as exc:
        _record_transport_error(
            "ProviderStreamDeadlineExceeded", headers=headers, response=response
        )
        raise ModelStreamDeadlineError(
            provider=response_label.lower().replace(" ", "_"),
            evidence=exc.evidence,
            stream_cleanup_failed=exc.stream_cleanup_failed,
        ) from None
    except SseEventTimeoutError as exc:
        _record_transport_error("SseEventTimeoutError", headers=headers, response=response)
        raise api_error(
            str(exc),
            error_type=type(exc).__name__,
            retryable=True,
        ) from exc
    except SseEventLimitError as exc:
        _record_transport_error("SseEventLimitError", headers=headers, response=response)
        raise api_error(
            str(exc),
            error_type=type(exc).__name__,
            retryable=False,
        ) from exc
    except httpx.RequestError as exc:
        _record_transport_error(
            _trusted_httpx_request_error_type(exc), headers=headers, response=response
        )
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
    request_redactor: SecretRedactor | None = None,
) -> Exception:
    retry_after_s = retry_after_seconds(response)
    if api_error_from_response is not None:
        error = api_error_from_response(response, message, retry_after_s)
        if isinstance(error, ModelProviderError) and not error.rejection_diagnostic:
            error.rejection_diagnostic = project_rejection_response(response)
    else:
        error = api_error(
            message,
            status_code=response.status_code,
            retry_after_s=retry_after_s,
        )
    if isinstance(error, ModelProviderError):
        _PROVIDER_ERROR_REQUEST_REDACTORS[error] = (
            request_credential_redactor(response) if request_redactor is None else request_redactor
        )
        if error.request_id is None:
            error.request_id = _response_error_request_id(response)
    # Keep the complete text for one pass with the combined request/workload
    # registry; separate passes can expose fragments of overlapping secrets.
    attach_provider_error_text(error, _provider_error_body_text(response))
    return error


def _response_error_request_id(response: httpx.Response) -> str | None:
    """Recover a plain correlation ID when the adapter has not supplied one."""

    candidates: list[object] = [
        response.headers.get("x-request-id"),
        response.headers.get("request-id"),
    ]
    try:
        content = response.content
    except httpx.ResponseNotRead:
        content = b""
    if content and len(content) <= MAX_PROVIDER_ERROR_BODY_BYTES:
        with suppress(ValueError, RecursionError):
            decoded = json.loads(content)
            if isinstance(decoded, list) and decoded:
                decoded = decoded[0]
            if isinstance(decoded, Mapping):
                candidates.append(decoded.get("request_id"))
                nested = decoded.get("error")
                if isinstance(nested, Mapping):
                    candidates.append(nested.get("request_id"))
    for candidate in candidates:
        if type(candidate) is str and _PUBLIC_PROVIDER_REQUEST_ID.fullmatch(candidate) is not None:
            return candidate
    return None


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


def bind_provider_error_workload_redactor(
    redactor: SecretRedactor | None,
) -> Token[SecretRedactor | None]:
    """Let the public provider error boundary also remove workload secrets.

    The runtime binds its application redactor around provider stream steps
    and token counts, so messages can be shown with registered secrets removed.
    """

    if redactor is not None and not isinstance(redactor, SecretRedactor):
        raise TypeError("redactor must be a SecretRedactor or None.")
    return _PROVIDER_ERROR_WORKLOAD_REDACTOR.set(redactor)


def reset_provider_error_workload_redactor(token: Token[SecretRedactor | None]) -> None:
    _PROVIDER_ERROR_WORKLOAD_REDACTOR.reset(token)


def _is_credential_name(name: str) -> bool:
    lowered = name.lower()
    return any(part in lowered for part in _CREDENTIAL_NAME_PARTS)


def request_credential_redactor(response: httpx.Response) -> SecretRedactor:
    """Return a redactor for the credentials the failed request itself carried."""

    try:
        request = response.request
    except RuntimeError:
        return SecretRedactor()
    values: list[str] = []
    for name, value in request.headers.items():
        if not _is_credential_name(name) or not value.strip():
            continue
        values.append(value)
        if name.lower() == "cookie":
            cookies: SimpleCookie[str] = SimpleCookie()
            with suppress(CookieError):
                cookies.load(value)
            values.extend(morsel.value for morsel in cookies.values() if morsel.value.strip())
            # Also retain the wire values, including cookies accepted by a
            # backend but rejected by SimpleCookie's stricter name grammar.
            for cookie in value.split(";"):
                _, separator, cookie_value = cookie.partition("=")
                cookie_value = cookie_value.strip().strip('"')
                if separator and cookie_value.strip():
                    values.append(cookie_value)
        _, separator, credential = value.partition(" ")
        if separator and credential.strip():
            values.append(credential.strip())
    query = request.url.query.decode("ascii", "ignore")
    values.extend(
        value
        for name, value in parse_qsl(query, keep_blank_values=False)
        if _is_credential_name(name) and value.strip()
    )
    return SecretRedactor(values)


def safe_error_response_text(
    response: httpx.Response,
    *,
    format_error_json: Callable[[Any], str | None],
) -> str:
    """Omit an untrusted body when the active workload redactor is unavailable.

    Provider adapters run below the application-owned workload-secret scope.
    Retaining even a parsed or truncated body here could preserve a recoverable
    secret fragment before the application gets a chance to redact it. Typed
    provider exceptions retain authoritative HTTP status/type/code fields. The
    provider's own message travels separately (``attach_provider_error_text``)
    and is shown only by the public provider boundary, after redaction.
    """

    del response, format_error_json
    return OMITTED_PROVIDER_ERROR_BODY


def provider_error_body_text(response: httpx.Response) -> str | None:
    """Extract a provider error body's message, minus the request's credentials.

    Returns ``None`` when the body was not read completely within the byte
    bound or carries no text. Registered workload secrets are removed, and the
    text bounded, only at the public provider boundary.
    """

    text = _provider_error_body_text(response)
    return None if text is None else request_credential_redactor(response).redact_text(text)


def _provider_error_body_text(response: httpx.Response) -> str | None:
    """Read complete error text without changing secrets before public redaction."""

    try:
        content = response.content
    except httpx.ResponseNotRead:
        return None
    if not content or len(content) > MAX_PROVIDER_ERROR_BODY_BYTES:
        return None
    text: str | None = None
    try:
        decoded: Any = json.loads(content)
    except (ValueError, RecursionError):
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            return None
    else:
        text = _provider_error_message_field(decoded)
    if text is None or not text.strip():
        return None
    return text


def _provider_error_message_field(decoded: Any) -> str | None:
    # GCP array-wraps some errors; FastAPI-style backends use {"detail": "..."}.
    if isinstance(decoded, list) and decoded:
        decoded = decoded[0]
    if not isinstance(decoded, Mapping):
        return None
    error = decoded.get("error")
    for container in (error, decoded):
        if isinstance(container, Mapping):
            message = container.get("message")
            if type(message) is str:
                return message
    detail = decoded.get("detail")
    if type(error) is str:
        return error
    return detail if type(detail) is str else None


def attach_provider_error_text(failure: BaseException, text: str | None) -> None:
    """Carry a provider's raw error text beside a typed failure.

    The text never enters the exception's message, attributes or traceback; the
    public provider boundary reads it, redacts known secrets, then bounds it.
    """

    if isinstance(failure, ModelProviderError) and type(text) is str and text.strip():
        _PROVIDER_ERROR_TEXT[failure] = text


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
    except (ValueError, RecursionError, httpx.ResponseNotRead):
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
            payload["model_provider_error"] = True
            if type(exc.status_code) is int:
                payload["status_code"] = exc.status_code
            if type(exc.retryable) is bool:
                payload["retryable"] = exc.retryable
            if type(exc.retry_after_s) in {int, float}:
                payload["retry_after_s"] = exc.retry_after_s
        if isinstance(exc, ProviderStreamCleanupError):
            payload["stream_cleanup_failed"] = True
        payload.update(api_error_diagnostic_fields(exc))
        return ModelStreamEvent(type=ModelStreamEventType.ERROR, payload=payload)
    if isinstance(exc, ModelProviderError):
        safe_exception = credential_safe_provider_exception(
            exc,
            provider_label=provider_label,
            provider_name=provider_name,
            credential_values=credential_values,
            safe_message=unresolved_message,
        )
        event = ModelStreamEvent.error(
            str(safe_exception),
            cause=safe_exception,
        )
        payload = dict(event.payload)
        # Preserve typed origin independently of optional, allowlisted identity.
        payload["model_provider_error"] = True
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
    diagnostic_payload = dict(event.payload)
    diagnostic_payload.update(
        api_error_diagnostic_fields(exc, credential_values=tuple(credential_values))
    )
    redacted = _public_error_redactor(credential_values, failure=exc).redact_json_values(
        diagnostic_payload
    )
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
    return retain_rejection_diagnostic(
        ModelProviderError(
            str(safe),
            provider=provider_name,
            status_code=safe.status_code,
            error_type=safe.error_type,
            error_code=safe.error_code,
            request_id=safe.request_id,
            retryable=False,
            retry_after_s=safe.retry_after_s,
            response_body=None,
        ),
        safe.rejection_diagnostic,
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
    # Without the request's credentials, no provider text can be checked for
    # them, so only the fixed classification survives.
    credentials_known = SecretRedactor(credential_values).has_values
    redactor = _public_error_redactor(credential_values, failure=exc)
    source = exc if isinstance(exc, ModelProviderError) else None
    if safe_message is not None:
        message = require_nonblank(safe_message, "safe_message")
    elif isinstance(exc, ModelContextOverflowError):
        message = f"{provider_label} model context window exceeded"
    else:
        message = (
            _public_provider_message(source, redactor)
            if source is not None and credentials_known
            else None
        ) or f"{safe_provider_exception_type_name(exc)}: {provider_label} provider failed"

    string_fields: dict[str, str | None] = {
        "error_type": None,
        "error_code": None,
        "request_id": None,
    }
    if source is not None and credentials_known:
        string_fields = {
            "error_type": _public_provider_identity(
                source.error_type, _PUBLIC_PROVIDER_IDENTIFIER, redactor
            ),
            "error_code": _public_provider_identity(
                source.error_code, _PUBLIC_PROVIDER_IDENTIFIER, redactor
            ),
            "request_id": _public_provider_identity(
                source.request_id, _PUBLIC_PROVIDER_REQUEST_ID, redactor
            ),
        }

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
        rejection_diagnostic=rejection_fields(
            getattr(exc, "rejection_diagnostic", None),
            credential_values=tuple(credential_values),
        ),
        retryable=source.retryable if source is not None else None,
        retry_after_s=source.retry_after_s if source is not None else None,
    )


def _public_error_redactor(
    credential_values: Sequence[str], *, failure: Exception | None = None
) -> SecretRedactor:
    """Request credentials plus any workload secrets the runtime bound."""

    redactor = SecretRedactor(credential_values)
    # Opaque transport failures may be built-in exceptions without weakrefs.
    # Only typed provider failures carry request metadata across this boundary.
    request_redactor = (
        _PROVIDER_ERROR_REQUEST_REDACTORS.get(failure)
        if isinstance(failure, ModelProviderError)
        else None
    )
    if request_redactor is not None:
        redactor = redactor.merged_with(request_redactor)
    workload = _PROVIDER_ERROR_WORKLOAD_REDACTOR.get()
    return redactor if workload is None else redactor.merged_with(workload)


def _public_provider_message(source: ModelProviderError, redactor: SecretRedactor) -> str | None:
    """Show the provider's own message, redacted first and bounded second.

    Truncation and control-character normalization need the complete registry:
    without it, a later redaction pass could miss a shortened or changed secret.
    Such messages stay omitted until the workload registry is available.
    """

    text = _PROVIDER_ERROR_TEXT.get(source)
    if text is None:
        return None
    try:
        prefix = str(source)
    except Exception:
        return None
    rendered = (
        prefix.replace(OMITTED_PROVIDER_ERROR_BODY, text)
        if OMITTED_PROVIDER_ERROR_BODY in prefix
        else f"{prefix}: {text}"
    )
    if len(rendered) > MAX_PROVIDER_ERROR_BODY_BYTES:
        # Never truncate before redaction: an oversized message stays omitted.
        return None
    try:
        rendered.encode("utf-8")
    except UnicodeEncodeError:
        # Lossy repair after redaction could reconstruct a registered secret.
        return None
    redacted = redactor.redact_text(rendered)
    printable = "".join(
        character if character in "\n\t" or character.isprintable() else " "
        for character in redacted
    )
    if printable != redacted and _PROVIDER_ERROR_WORKLOAD_REDACTOR.get() is None:
        # A later caller may know workload secrets containing these control
        # characters. Formatting here would prevent that caller from matching.
        return None
    # Redact again after normalization in case replacing a control character
    # reconstructed a different registered secret, then apply the byte bound.
    bounded, truncated = redactor.redact_text_bounded_with_marker(
        printable,
        max_bytes=PUBLIC_PROVIDER_ERROR_MESSAGE_BYTES,
        truncation_marker="...[truncated]",
    )
    if truncated and _PROVIDER_ERROR_WORKLOAD_REDACTOR.get() is None:
        return None
    # Keep edge whitespace for a caller that has not yet supplied its registry.
    return bounded if _PROVIDER_ERROR_WORKLOAD_REDACTOR.get() is None else bounded.strip() or None


def _public_provider_identity(
    value: object,
    pattern: re.Pattern[str],
    redactor: SecretRedactor,
) -> str | None:
    """Pass a plain identifier through unless it contains a known secret."""

    if type(value) is not str or pattern.fullmatch(value) is None:
        return None
    # Error parsers normalize the outer whitespace of identity fields. Check
    # that form of the registry too, so normalization cannot expose a secret.
    return value if redactor.redact_stripped_text(value) == value else None


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
    "PUBLIC_PROVIDER_ERROR_MESSAGE_BYTES",
    "SharedAsyncClient",
    "aclose_transport",
    "attach_provider_error_text",
    "bind_provider_error_workload_redactor",
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
    "provider_error_body_text",
    "request_credential_redactor",
    "request_json",
    "reset_provider_error_workload_redactor",
    "response_json_object",
    "retain_provider_error_metadata",
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
