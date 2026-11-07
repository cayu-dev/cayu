"""OpenAI Responses over a persistent WebSocket.

`WebSocketOpenAITransport` is an `OpenAITransport`; pass it to
``OpenAIProvider(transport=...)``. A streaming call becomes one
``response.create`` message on ``wss://<base>/v1/responses``. The server answers
with the same Responses events the HTTP stream carries, so the provider parses,
classifies and recovers them as it does for SSE. Calls that do not stream (token
counts, embeddings, ``streaming=False``) go over HTTP.

A connection runs one response at a time and keeps its latest response in
memory, under ``store=false`` too. A request is sent as only its new input items
plus ``previous_response_id`` when all of these hold:

- it goes out on the connection that produced the previous response;
- that response completed and the provider accepted it;
- every field except ``input`` equals the previous request's (model,
  instructions, tools, options, ``store``, ``prompt_cache_key``);
- its input starts with exactly the previous request's input followed by that
  response's output items, and has at least one item after them.

The server then rebuilds the same input the full request carries. Any other
request is sent in full, as over HTTP: the first request on a connection, a
reconnect, a changed setting, history that was compacted, redacted or edited
since the previous response, a previous response that did not complete, and a
request the caller already chained with ``previous_response_id``. When the
server no longer holds the previous response (``previous_response_not_found``),
the request is sent again in full on the same connection before any event
reaches the caller.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from functools import cache
from hashlib import sha256
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from cayu.providers._config import positive_finite_seconds
from cayu.providers._http import (
    _response_api_error,
    _TrustedSseJsonEvent,
    aclose_transport,
    provider_ssl_context,
    request_credential_redactor,
)
from cayu.providers._openai_search_trace import ResponseStructureTrace
from cayu.providers._sse import DEFAULT_SSE_MAX_EVENT_BYTES
from cayu.providers.base import ModelStreamDeadlineError
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
from cayu.providers.openai import (
    HttpxOpenAITransport,
    OpenAIAPIError,
    OpenAIProtocolError,
    OpenAITransport,
    _is_stale_chain_error,
    _openai_api_error_from_response,
    _openai_stream_error_exception,
    _safe_error_response_text,
    _validate_url,
)

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection
    from websockets.exceptions import InvalidStatus

    class _ObservedConnection(ClientConnection):
        on_data: Callable[[], None] | None


DEFAULT_WEBSOCKET_IDLE_TIMEOUT_S = 300.0
"""How long an idle connection stays pooled for the next call of its lineage."""

DEFAULT_WEBSOCKET_MAX_IDLE_CONNECTIONS = 32

_CLOSE_TIMEOUT_S = 2.0
_DEADLINE_KINDS = (
    ProviderDeadlineKind.TRANSPORT_IDLE,
    ProviderDeadlineKind.PROTOCOL_IDLE,
    ProviderDeadlineKind.ABSOLUTE,
)
_IDLE_PAUSE_KINDS = (
    ProviderDeadlineKind.TRANSPORT_IDLE,
    ProviderDeadlineKind.PROTOCOL_IDLE,
)
# After one of these the server sends nothing more for the request, and the
# connection can take the next one.
_TERMINAL_EVENT_TYPES = frozenset(
    {
        "error",
        "response.cancelled",
        "response.completed",
        "response.expired",
        "response.failed",
        "response.incomplete",
    }
)
# HTTP framing headers that do not belong on a WebSocket handshake; websockets
# owns the handshake's own headers.
_HTTP_ONLY_HEADERS = frozenset(
    {"accept-encoding", "connection", "content-length", "content-type", "host", "upgrade"}
)
_MESSAGE_TOO_BIG_CLOSE_CODE = 1009


def _digest(value: object) -> bytes:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).digest()


@dataclass(frozen=True, slots=True)
class _HeldResponse:
    """What the server holds for a connection's latest completed response.

    ``input_items`` digests the full input the server saw for it followed by its
    output items, which is the prefix the next request must start with.
    """

    settings: bytes
    input_items: tuple[bytes, ...]
    response_id: str


class _RequestBody:
    """One request's full ``response.create`` body and how it may chain."""

    def __init__(self, payload: Mapping[str, Any], *, incremental: bool) -> None:
        body = dict(payload)
        # Every WebSocket response streams; the field belongs to HTTP.
        body.pop("stream", None)
        self.full = body
        items = body.get("input")
        cache_key = body.get("prompt_cache_key")
        self.cache_key = cache_key if type(cache_key) is str else None
        # A caller that chained the request itself (server reasoning state) or
        # used a conversation sends it as given: its input is not a full transcript.
        self.chainable = (
            incremental
            and type(items) is list
            and "previous_response_id" not in body
            and "conversation" not in body
        )
        self.settings = b""
        self.input_items: tuple[bytes, ...] = ()
        if self.chainable and isinstance(items, list):
            self.settings = _digest(
                {name: value for name, value in body.items() if name != "input"}
            )
            self.input_items = tuple(_digest(item) for item in items)

    def chains(self, held: _HeldResponse | None) -> bool:
        """Whether this request extends the held response with at least one new item."""

        if not self.chainable or held is None or held.settings != self.settings:
            return False
        held_count = len(held.input_items)
        return (
            len(self.input_items) > held_count and self.input_items[:held_count] == held.input_items
        )

    def incremental(self, held: _HeldResponse | None) -> dict[str, Any] | None:
        """Return the body with only new input, or None when it must go in full."""

        if held is None or not self.chains(held):
            return None
        held_count = len(held.input_items)
        return {
            **self.full,
            "input": self.full["input"][held_count:],
            "previous_response_id": held.response_id,
        }

    def held_after(self, response_id: str, output_items: list[Any]) -> _HeldResponse:
        return _HeldResponse(
            settings=self.settings,
            input_items=self.input_items + tuple(_digest(item) for item in output_items),
            response_id=response_id,
        )


class _ResponseOutput:
    """The output items of the response streaming on a connection."""

    def __init__(self) -> None:
        self._done_items: dict[int, Mapping[str, Any]] = {}
        self._consistent = True

    def observe(self, event: Mapping[str, Any]) -> None:
        if event.get("type") != "response.output_item.done":
            return
        index = event.get("output_index")
        item = event.get("item")
        if type(index) is not int or not isinstance(item, Mapping) or index in self._done_items:
            self._consistent = False
            return
        self._done_items[index] = item

    def held_response(
        self, body: _RequestBody, completed: Mapping[str, Any]
    ) -> _HeldResponse | None:
        if not body.chainable:
            return None
        response = completed.get("response")
        if not isinstance(response, Mapping):
            return None
        response_id = response.get("id")
        if type(response_id) is not str or not response_id:
            return None
        output = response.get("output")
        if isinstance(output, list) and output:
            items: list[Any] = output
        elif (output is None or output == []) and self._consistent and self._done_items:
            # Some servers leave the completed envelope's output empty and send
            # each item only in output_item.done, as the provider parser accepts.
            items = [self._done_items[index] for index in sorted(self._done_items)]
        else:
            return None
        if not all(isinstance(item, Mapping) for item in items):
            return None
        return body.held_after(response_id, items)


class _Connection:
    __slots__ = ("held", "idle_since", "key", "loop", "websocket")

    def __init__(
        self,
        websocket: _ObservedConnection,
        *,
        key: tuple[str, bytes, str | None],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self.websocket = websocket
        self.key = key
        self.loop = loop
        self.held: _HeldResponse | None = None
        self.idle_since = 0.0

    def is_open(self) -> bool:
        from websockets.protocol import State

        return self.websocket.state is State.OPEN

    def abort(self) -> None:
        """Drop the connection without waiting, for any state it is left in."""

        transport = getattr(self.websocket, "transport", None)
        if transport is not None:
            with suppress(Exception):
                transport.abort()


def _abort_unclaimed_handshake(handshake: asyncio.Future[Any]) -> None:
    if handshake.cancelled() or handshake.exception() is not None:
        return
    transport = getattr(handshake.result(), "transport", None)
    if transport is not None:
        with suppress(Exception):
            transport.abort()


@cache
def _websocket_client() -> tuple[Any, Any]:
    """The websockets ``connect`` and connection classes this transport uses.

    ``connect`` never follows a redirect: the HTTP transport does not either,
    and a redirect could carry the credentials and the conversation to another
    origin. A 3xx answer fails like any other refused handshake. The connection
    reports every chunk of received bytes so the transport-idle clock sees
    activity inside a message that is still arriving.
    """

    from websockets.asyncio.client import ClientConnection, connect

    class NoRedirectConnect(connect):
        def process_redirect(self, exc: Exception) -> Exception | str:
            return exc

    class ObservedConnection(ClientConnection):
        on_data: Callable[[], None] | None = None

        def data_received(self, data: bytes) -> None:
            on_data = self.on_data
            if on_data is not None and data:
                on_data()
            super().data_received(data)

    return NoRedirectConnect, ObservedConnection


def _require_websockets() -> None:
    try:
        import websockets.asyncio.client  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "WebSocketOpenAITransport needs the websockets package; install cayu[openai-websocket]."
        ) from exc


def _websocket_url(url: str) -> str:
    parts = urlsplit(url)
    scheme = {"https": "wss", "http": "ws"}.get(parts.scheme.lower())
    if scheme is None:  # pragma: no cover - _validate_url admits only http(s)
        raise ValueError("OpenAI WebSocket URL must use http or https.")
    return urlunsplit((scheme, parts.netloc, parts.path, parts.query, ""))


def _websocket_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {
        name: value
        for name, value in headers.items()
        if name.lower() not in _HTTP_ONLY_HEADERS and not name.lower().startswith("sec-websocket-")
    }


def _decode_event(frame: object) -> dict[str, Any]:
    if not isinstance(frame, str):
        raise OpenAIProtocolError(
            "OpenAI WebSocket sent a binary message.",
            reason_code="websocket_sent_a_binary_message",
        )
    try:
        decoded = json.loads(frame)
    except ValueError as exc:
        raise OpenAIProtocolError(
            "OpenAI WebSocket message was not valid JSON.",
            reason_code="websocket_message_was_not_valid_json",
        ) from exc
    if not isinstance(decoded, dict):
        raise OpenAIProtocolError(
            "OpenAI WebSocket message must decode to a JSON object.",
            reason_code="websocket_message_must_decode_to_a_json_object",
        )
    status = decoded.get("status")
    if (
        decoded.get("type") == "error"
        and type(status) is int
        and 100 <= status <= 599
        and "status_code" not in decoded
    ):
        # WebSocket error events carry the HTTP status the same request would
        # have failed with as ``status``; the Responses parser reads ``status_code``.
        decoded["status_code"] = status
    return decoded


def _is_missing_previous_response(event: Mapping[str, Any]) -> bool:
    return event.get("type") == "error" and _is_stale_chain_error(
        _openai_stream_error_exception(event)
    )


def _handshake_status_error(
    exc: InvalidStatus, *, url: str, headers: Mapping[str, str]
) -> Exception:
    """Classify a refused handshake exactly like the same HTTP error response."""

    raw = exc.response
    response_headers = [
        (name, value)
        for name, value in raw.headers.raw_items()
        if name.lower() != "content-encoding" or value.strip().lower() == "identity"
    ]
    encoded = len(response_headers) != len(list(raw.headers.raw_items()))
    response = httpx.Response(
        raw.status_code,
        headers=response_headers,
        # A compressed body is not decoded here, as the HTTP transport omits it.
        content=b"" if encoded else bytes(raw.body or b""),
        request=httpx.Request("GET", url, headers=headers),
    )
    _record_http_error(response, headers=headers, body_response=response)
    message = (
        f"OpenAI API request failed with HTTP {response.status_code}: "
        f"{_safe_error_response_text(response)}"
    )
    return _response_api_error(
        response,
        message,
        api_error=OpenAIAPIError,
        api_error_from_response=_openai_api_error_from_response,
    )


class WebSocketOpenAITransport:
    """Responses streaming over one persistent WebSocket per conversation lineage.

    Idle connections are pooled by endpoint, credentials and
    ``prompt_cache_key``, so the next call of a lineage finds the connection
    holding its previous response. ``incremental_input=False`` keeps the
    connection reuse but always sends the full input. ``http_transport`` serves
    the calls that do not stream; it defaults to `HttpxOpenAITransport`.
    """

    def __init__(
        self,
        *,
        allow_http: bool = False,
        incremental_input: bool = True,
        idle_timeout_s: float = DEFAULT_WEBSOCKET_IDLE_TIMEOUT_S,
        max_idle_connections: int = DEFAULT_WEBSOCKET_MAX_IDLE_CONNECTIONS,
        http_transport: OpenAITransport | None = None,
    ) -> None:
        if type(allow_http) is not bool:
            raise TypeError("allow_http must be a bool.")
        if type(incremental_input) is not bool:
            raise TypeError("incremental_input must be a bool.")
        if type(max_idle_connections) is not int or max_idle_connections < 0:
            raise ValueError("max_idle_connections must be a non-negative integer.")
        _require_websockets()
        self.allow_http = allow_http
        self.incremental_input = incremental_input
        self.idle_timeout_s = positive_finite_seconds(idle_timeout_s, "idle_timeout_s")
        self.max_idle_connections = max_idle_connections
        self.http_transport: OpenAITransport = (
            http_transport
            if http_transport is not None
            else HttpxOpenAITransport(allow_http=allow_http)
        )
        self._idle: dict[tuple[str, bytes, str | None], list[_Connection]] = {}

    async def aclose(self) -> None:
        """Close pooled connections and the HTTP transport."""

        connections = [connection for pooled in self._idle.values() for connection in pooled]
        self._idle.clear()
        loop = asyncio.get_running_loop()
        try:
            for connection in connections:
                if connection.loop is loop:
                    with suppress(Exception):
                        await connection.websocket.close()
        finally:
            # A cancelled or failed graceful close must not leave the rest open;
            # aborting a connection that already closed does nothing.
            for connection in connections:
                connection.abort()
            await aclose_transport(self.http_transport)

    async def create_response(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        return await self.http_transport.create_response(
            url=url, headers=headers, payload=payload, timeout_s=timeout_s
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
        from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidStatus

        url = _validate_url(url, "url", allow_http=self.allow_http)
        websocket_url = _websocket_url(url)
        websocket_headers = _websocket_headers(headers)
        body = _RequestBody(payload, incremental=self.incremental_input)
        key = (
            websocket_url,
            _digest(sorted((name.lower(), value) for name, value in websocket_headers.items())),
            body.cache_key,
        )
        deadlines = ProviderStreamDeadlines(
            transport_idle_timeout_s=transport_idle_timeout_s,
            protocol_idle_timeout_s=protocol_idle_timeout_s,
            semantic_progress_timeout_s=semantic_progress_timeout_s,
            absolute_stream_timeout_s=absolute_stream_timeout_s,
        )
        controller = current_provider_deadline_controller()
        owns_controller = controller is None
        if controller is None:
            controller = ProviderStreamDeadlineController(deadlines)
        elif controller.deadlines != deadlines:
            raise ValueError("Provider transport deadline policy changed after dispatch admission.")
        error_redactor = request_credential_redactor(
            httpx.Response(101, request=httpx.Request("GET", url, headers=websocket_headers))
        )
        connection: _Connection | None = None
        terminal: _TrustedSseJsonEvent | None = None
        output = _ResponseOutput()
        received = False
        try:
            send_in_full = False
            reopened = False
            while True:
                reused = False
                if connection is None:
                    connection = self._take_idle(key, body)
                    reused = connection is not None
                if connection is None:
                    connection = await self._open(
                        websocket_url,
                        websocket_headers,
                        key=key,
                        timeout_s=timeout_s,
                        controller=controller,
                    )
                # Raw bytes, including fragments of a message still arriving,
                # refresh the transport clock as they do for the HTTP stream.
                connection.websocket.on_data = controller.observe_transport
                chained = None if send_in_full else body.incremental(connection.held)
                request = body.full if chained is None else chained
                # Whatever happens next, only a newly completed response is held.
                connection.held = None
                message = json.dumps(
                    {"type": "response.create", **request},
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                try:
                    await controller.wait_for(
                        connection.websocket.send(message), kinds=_DEADLINE_KINDS
                    )
                except ConnectionClosed:
                    if not reused or reopened:
                        raise
                    # The server closed this pooled connection while it was idle,
                    # before the request went out. Open a new one, once.
                    connection.abort()
                    connection = None
                    reopened = True
                    continue
                send_in_full = False
                structure = ResponseStructureTrace()
                while True:
                    frame = await controller.wait_for(
                        connection.websocket.recv(), kinds=_DEADLINE_KINDS
                    )
                    controller.observe_transport()
                    event = _decode_event(frame)
                    controller.observe_protocol()
                    if (
                        chained is not None
                        and not received
                        and _is_missing_previous_response(event)
                    ):
                        send_in_full = True
                        break
                    received = True
                    envelope = _TrustedSseJsonEvent(event, retry_after_s=None)
                    envelope._error_redactor = error_redactor
                    structure.record(event)
                    envelope._response_structure = structure.snapshot()
                    _record_stream_error(event, headers=headers)
                    output.observe(event)
                    if event.get("type") in _TERMINAL_EVENT_TYPES:
                        terminal = envelope
                    # The caller's pause is excluded from the idle clocks as a
                    # whole, so bytes arriving during it must not also move them.
                    connection.websocket.on_data = None
                    pause_started = controller.idle_pause_started()
                    yield envelope
                    controller.exclude_idle_pause(pause_started, kinds=_IDLE_PAUSE_KINDS)
                    connection.websocket.on_data = controller.observe_transport
                    if terminal is not None:
                        return
        except ProviderStreamDeadlineExceeded as exc:
            _record_transport_error("ProviderStreamDeadlineExceeded", headers=headers)
            raise ModelStreamDeadlineError(
                provider="openai",
                evidence=exc.evidence,
                stream_cleanup_failed=exc.stream_cleanup_failed,
            ) from None
        except ConnectionClosed as exc:
            sent = exc.sent
            if exc.rcvd is None and sent is not None and sent.code == _MESSAGE_TOO_BIG_CLOSE_CODE:
                _record_transport_error("SseEventLimitError", headers=headers)
                raise OpenAIAPIError(
                    "OpenAI WebSocket message exceeded the "
                    f"{DEFAULT_SSE_MAX_EVENT_BYTES}-byte limit.",
                    error_type="SseEventLimitError",
                    retryable=False,
                ) from None
            _record_transport_error("RemoteProtocolError", headers=headers)
            # As over HTTP: a connection lost before the response started can be
            # retried; one lost after output arrived is not retried blindly.
            raise OpenAIAPIError(
                f"OpenAI API WebSocket connection closed for {url}.",
                error_type="RemoteProtocolError",
                retryable=not received,
            ) from None
        except InvalidStatus as exc:
            raise _handshake_status_error(exc, url=url, headers=websocket_headers) from None
        except InvalidHandshake:
            _record_transport_error("ProtocolError", headers=headers)
            raise OpenAIAPIError(
                f"OpenAI API WebSocket handshake failed for {url}.",
                error_type="ProtocolError",
                retryable=False,
            ) from None
        except TimeoutError:
            _record_transport_error("ConnectTimeout", headers=headers)
            raise OpenAIAPIError(
                f"OpenAI API WebSocket connection timed out for {url}.",
                error_type="ConnectTimeout",
                retryable=True,
            ) from None
        except OSError:
            _record_transport_error("ConnectError", headers=headers)
            raise OpenAIAPIError(
                f"OpenAI API WebSocket connection failed for {url}.",
                error_type="ConnectError",
                retryable=True,
            ) from None
        finally:
            if connection is not None:
                connection.websocket.on_data = None
                if terminal is None:
                    # The caller stopped reading, or the stream failed, while the
                    # server may still be sending: the connection cannot be reused.
                    connection.abort()
                else:
                    if terminal.get("type") == "response.completed" and terminal._terminal_accepted:
                        connection.held = output.held_response(body, terminal)
                    self._release(connection)
            if owns_controller:
                controller.close()

    async def _open(
        self,
        websocket_url: str,
        headers: Mapping[str, str],
        *,
        key: tuple[str, bytes, str | None],
        timeout_s: float,
        controller: ProviderStreamDeadlineController,
    ) -> _Connection:
        connect, connection_type = _websocket_client()

        async def open_connection() -> _ObservedConnection:
            return await connect(
                websocket_url,
                additional_headers=list(headers.items()),
                open_timeout=timeout_s,
                close_timeout=_CLOSE_TIMEOUT_S,
                max_size=DEFAULT_SSE_MAX_EVENT_BYTES,
                ssl=provider_ssl_context() if websocket_url.startswith("wss:") else None,
                create_connection=connection_type,
            )

        handshake = asyncio.ensure_future(open_connection())
        try:
            websocket = await controller.wait_for(handshake, kinds=_DEADLINE_KINDS)
        except BaseException:
            # A deadline or cancellation can win the race against a handshake
            # that has just succeeded; nobody else would own that connection.
            handshake.add_done_callback(_abort_unclaimed_handshake)
            raise
        controller.observe_transport()
        return _Connection(websocket, key=key, loop=asyncio.get_running_loop())

    def _take_idle(
        self, key: tuple[str, bytes, str | None], body: _RequestBody
    ) -> _Connection | None:
        self._prune()
        pooled = self._idle.get(key)
        if not pooled:
            return None
        chosen = next(
            (connection for connection in reversed(pooled) if body.chains(connection.held)),
            pooled[-1],
        )
        pooled.remove(chosen)
        if not pooled:
            del self._idle[key]
        return chosen

    def _prune(self) -> None:
        loop = asyncio.get_running_loop()
        now = loop.time()
        for key in list(self._idle):
            kept: list[_Connection] = []
            for connection in self._idle[key]:
                if (
                    connection.loop is loop
                    and connection.is_open()
                    and now - connection.idle_since < self.idle_timeout_s
                ):
                    kept.append(connection)
                else:
                    connection.abort()
            if kept:
                self._idle[key] = kept
            else:
                del self._idle[key]

    def _release(self, connection: _Connection) -> None:
        if not connection.is_open():
            connection.abort()
            return
        connection.idle_since = connection.loop.time()
        self._idle.setdefault(connection.key, []).append(connection)
        idle_count = sum(len(pooled) for pooled in self._idle.values())
        while idle_count > self.max_idle_connections:
            oldest_key, oldest = min(
                (
                    (key, pooled_connection)
                    for key, pooled in self._idle.items()
                    for pooled_connection in pooled
                ),
                key=lambda entry: entry[1].idle_since,
            )
            self._idle[oldest_key].remove(oldest)
            if not self._idle[oldest_key]:
                del self._idle[oldest_key]
            oldest.abort()
            idle_count -= 1


__all__ = [
    "DEFAULT_WEBSOCKET_IDLE_TIMEOUT_S",
    "DEFAULT_WEBSOCKET_MAX_IDLE_CONNECTIONS",
    "WebSocketOpenAITransport",
]
