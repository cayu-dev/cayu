"""Shared HTTP transport plumbing for provider adapters.

The provider transports (OpenAI, Anthropic, Chat Completions, Vertex) share
identical httpx POST/stream mechanics: certifi-backed TLS, URL validation,
error-body sanitizing, and SSE decoding. This module holds that plumbing once;
each adapter keeps only its provider-specific error classification and shaping.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Mapping
from typing import Any
from urllib.parse import urlparse

import certifi
import httpx

from cayu._validation import require_clean_nonblank, require_nonblank
from cayu.providers._sse import aiter_sse_json_events
from cayu.providers.base import ModelContextOverflowError

MAX_PROVIDER_ERROR_BODY_CHARS = 2_000


def new_async_client() -> httpx.AsyncClient:
    """Build an httpx.AsyncClient with explicit certifi-backed TLS verification.

    Per-request timeouts are passed at call time (see :func:`post_json` and
    :func:`stream_sse_json_events`), so the client itself carries no fixed
    timeout and can be reused across both blocking and streaming requests.
    """
    return httpx.AsyncClient(verify=certifi.where())


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
    api_error_from_response: Callable[[httpx.Response, str], Exception] | None = None,
) -> Mapping[str, Any]:
    """POST a JSON payload and return the decoded JSON object response.

    The caller-owned ``client`` is reused (its connection pool is kept warm)
    rather than opening a fresh TLS connection per request. HTTP failures raise
    ``api_error`` (after ``raise_context_overflow`` gets a chance to classify
    them); non-object response bodies raise ``protocol_error``.
    ``request_label``/``response_label`` prefix messages (e.g. ``"OpenAI API"``
    / ``"OpenAI"``). ``api_error_from_response`` lets an adapter build a
    structured error (typed status/code fields) from the HTTP error response
    instead of the flat ``api_error(message)``.
    """
    try:
        response = await client.post(
            url,
            headers=dict(headers),
            json=dict(payload),
            timeout=timeout_s,
        )
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
        if api_error_from_response is not None:
            raise api_error_from_response(exc.response, message) from exc
        raise api_error(message) from exc
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
    stream_idle_timeout_s: float,
    request_label: str,
    response_label: str,
    api_error: Callable[..., Exception],
    protocol_error: type[Exception],
    error_response_text: Callable[[httpx.Response], str],
    raise_context_overflow: Callable[[httpx.Response], None] | None = None,
    api_error_from_response: Callable[[httpx.Response, str], Exception] | None = None,
) -> AsyncIterator[Mapping[str, Any]]:
    """POST a streaming JSON payload and yield decoded SSE data objects.

    The caller-owned ``client`` is reused across requests; only the streaming
    response is opened and closed per call. ``api_error_from_response`` mirrors
    :func:`post_json`: adapters can build a structured error (typed status/code
    fields) from the HTTP error response.
    """
    timeout = httpx.Timeout(timeout_s, read=None)
    try:
        async with client.stream(
            "POST",
            url,
            headers=dict(headers),
            json=dict(payload),
            timeout=timeout,
        ) as response:
            if response.status_code >= 400:
                # Read the streamed error body while the response is still open.
                # Otherwise error_response_text touches an unread streaming body
                # and raises httpx.ResponseNotRead, masking the real API error
                # (e.g. HTTP 404 from a wrong endpoint).
                await response.aread()
                if raise_context_overflow is not None:
                    raise_context_overflow(response)
                message = (
                    f"{request_label} request failed with HTTP "
                    f"{response.status_code}: "
                    f"{error_response_text(response)}"
                )
                if api_error_from_response is not None:
                    raise api_error_from_response(response, message)
                raise api_error(message)
            async for event in aiter_sse_json_events(
                response.aiter_lines(),
                idle_timeout_s=stream_idle_timeout_s,
                provider_label=response_label,
                protocol_error=protocol_error,
            ):
                yield event
    except httpx.RequestError as exc:
        raise _request_api_error(
            api_error, request_label=request_label, url=url, cause=exc
        ) from exc


def _request_api_error(
    api_error: Callable[..., Exception],
    *,
    request_label: str,
    url: str,
    cause: httpx.RequestError,
) -> Exception:
    return api_error(
        f"{request_label} request failed for {url}: {cause}",
        error_type=type(cause).__name__,
        retryable=_is_retryable_transport_error(cause),
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
    """Render an error response body safely (sanitized JSON or truncated text).

    ``format_error_json`` receives the decoded JSON body and returns the
    sanitized text, or ``None`` to fall back to the truncated raw body.
    """
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            decoded = response.json()
        except ValueError:
            return truncate_error_text(response.text)
        formatted = format_error_json(decoded)
        if formatted is not None:
            return formatted
    return truncate_error_text(response.text)


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


__all__ = [
    "MAX_PROVIDER_ERROR_BODY_CHARS",
    "SharedAsyncClient",
    "aclose_transport",
    "copy_headers",
    "exception_message",
    "json_error_text",
    "new_async_client",
    "optional_error_string",
    "post_json",
    "response_json_object",
    "safe_error_json",
    "safe_error_response_text",
    "safe_flat_error_json",
    "stream_sse_json_events",
    "truncate_error_text",
    "validate_base_url",
    "validate_url",
]
