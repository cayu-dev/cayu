"""Tests for the shared provider transport plumbing (_http + _sse).

The OpenAI, Anthropic, Chat Completions, and Vertex adapters delegate their
HTTP/SSE mechanics to ``cayu.providers._http`` and ``cayu.providers._sse``.
These tests pin the shared behavior: one SSE parser with one heartbeat/idle
timer (keep-alives count as activity for every provider), provider-labeled
error messages, and the shared URL validation.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from cayu.providers import (
    ChatCompletionsProtocolError,
    HttpxChatCompletionsTransport,
    HttpxOpenAITransport,
    OpenAIProtocolError,
)
from cayu.providers._http import SharedAsyncClient, validate_base_url, validate_url
from cayu.providers._sse import aiter_sse_json_events


class _StreamingResponse:
    """Minimal streaming-response stub for the shared SSE transport path."""

    status_code = 200

    def __init__(self, lines: list[str], *, heartbeat_sleep_s: float = 0.0) -> None:
        self._lines = lines
        self._heartbeat_sleep_s = heartbeat_sleep_s

    async def aiter_lines(self):
        for line in self._lines:
            if self._heartbeat_sleep_s:
                await asyncio.sleep(self._heartbeat_sleep_s)
            yield line


class _StreamContext:
    def __init__(self, response: _StreamingResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _StreamingResponse:
        return self._response

    async def __aexit__(self, *args: Any) -> None:
        return None


def _client_factory(response: _StreamingResponse) -> type:
    class FakeClient:
        is_closed = False

        def __init__(self, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        def stream(
            self,
            method: str,
            url: str,
            *,
            headers: dict[str, str],
            json: dict[str, Any],
            timeout: Any = None,
        ) -> _StreamContext:
            return _StreamContext(response)

    return FakeClient


_KEEPALIVE_LINES = [
    ": keepalive",
    ": keepalive",
    ": keepalive",
    ": keepalive",
    ": keepalive",
    'data: {"ok": true}',
    "",
    "data: [DONE]",
    "",
]


@pytest.mark.anyio
async def test_openai_transport_survives_keepalive_heartbeats(monkeypatch) -> None:
    # Five heartbeats, each within one idle window (0.04 < 0.1) but 0.2s in
    # total: the stream survives only if `:` comments refresh the idle timer.
    # Before the transports shared one SSE parser, the OpenAI parser ignored
    # heartbeats and killed exactly this stream.
    response = _StreamingResponse(_KEEPALIVE_LINES, heartbeat_sleep_s=0.04)
    monkeypatch.setattr("cayu.providers._http.httpx.AsyncClient", _client_factory(response))

    events = [
        event
        async for event in HttpxOpenAITransport().stream_response_events(
            url="https://api.openai.com/v1/responses",
            headers={},
            payload={},
            timeout_s=1.0,
            stream_idle_timeout_s=0.1,
        )
    ]

    assert events == [{"ok": True}]


@pytest.mark.anyio
async def test_chat_completions_transport_survives_keepalive_heartbeats(monkeypatch) -> None:
    response = _StreamingResponse(_KEEPALIVE_LINES, heartbeat_sleep_s=0.04)
    monkeypatch.setattr("cayu.providers._http.httpx.AsyncClient", _client_factory(response))

    events = [
        event
        async for event in HttpxChatCompletionsTransport().stream_chat_completions(
            url="https://api.openai.com/v1/chat/completions",
            headers={},
            payload={},
            timeout_s=1.0,
            stream_idle_timeout_s=0.1,
        )
    ]

    assert events == [{"ok": True}]


@pytest.mark.anyio
async def test_sse_parser_rejects_nonpositive_idle_timeout() -> None:
    async def lines():
        yield ""

    with pytest.raises(ValueError, match="idle_timeout_s"):
        [
            event
            async for event in aiter_sse_json_events(
                lines(),
                idle_timeout_s=0,
                provider_label="OpenAI",
                protocol_error=OpenAIProtocolError,
            )
        ]


@pytest.mark.anyio
async def test_sse_parser_raises_provider_labeled_protocol_errors() -> None:
    async def lines():
        yield "data: not-json"
        yield ""

    with pytest.raises(OpenAIProtocolError, match="OpenAI SSE data was not valid JSON"):
        [
            event
            async for event in aiter_sse_json_events(
                lines(),
                idle_timeout_s=1.0,
                provider_label="OpenAI",
                protocol_error=OpenAIProtocolError,
            )
        ]

    with pytest.raises(
        ChatCompletionsProtocolError,
        match="Chat Completions SSE data must decode to a JSON object",
    ):
        [
            event
            async for event in aiter_sse_json_events(
                _iter_lines(['data: ["not-an-object"]', ""]),
                idle_timeout_s=1.0,
                provider_label="Chat Completions",
                protocol_error=ChatCompletionsProtocolError,
            )
        ]


@pytest.mark.anyio
async def test_sse_parser_yields_trailing_data_without_blank_line() -> None:
    events = [
        event
        async for event in aiter_sse_json_events(
            _iter_lines(['data: {"tail": 1}']),
            idle_timeout_s=1.0,
            provider_label="OpenAI",
            protocol_error=OpenAIProtocolError,
        )
    ]
    assert events == [{"tail": 1}]


@pytest.mark.anyio
async def test_sse_parser_stops_at_done_marker() -> None:
    events = [
        event
        async for event in aiter_sse_json_events(
            _iter_lines(['data: {"n": 1}', "", "data: [DONE]", "", 'data: {"n": 2}', ""]),
            idle_timeout_s=1.0,
            provider_label="OpenAI",
            protocol_error=OpenAIProtocolError,
        )
    ]
    assert events == [{"n": 1}]


class _CountingClient:
    """Records how many httpx clients a transport constructs and closes."""

    constructed = 0
    closed = 0

    def __init__(self, **kwargs: Any) -> None:
        type(self).constructed += 1
        self.is_closed = False

    def stream(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
        timeout: Any = None,
    ) -> _StreamContext:
        return _StreamContext(_StreamingResponse(['data: {"ok": true}', "", "data: [DONE]", ""]))

    async def aclose(self) -> None:
        self.is_closed = True
        type(self).closed += 1


async def _drain_stream(transport: HttpxOpenAITransport) -> list[dict[str, Any]]:
    return [
        event
        async for event in transport.stream_response_events(
            url="https://api.openai.com/v1/responses",
            headers={},
            payload={},
            timeout_s=1.0,
            stream_idle_timeout_s=1.0,
        )
    ]


@pytest.mark.anyio
async def test_transport_reuses_one_client_across_requests(monkeypatch) -> None:
    class Client(_CountingClient):
        constructed = 0
        closed = 0

    monkeypatch.setattr("cayu.providers._http.httpx.AsyncClient", Client)

    transport = HttpxOpenAITransport()
    assert await _drain_stream(transport) == [{"ok": True}]
    assert await _drain_stream(transport) == [{"ok": True}]

    # Two requests, one shared client: no fresh TLS handshake per request.
    assert Client.constructed == 1
    assert Client.closed == 0


@pytest.mark.anyio
async def test_transport_aclose_closes_shared_client_and_reopens(monkeypatch) -> None:
    class Client(_CountingClient):
        constructed = 0
        closed = 0

    monkeypatch.setattr("cayu.providers._http.httpx.AsyncClient", Client)

    transport = HttpxOpenAITransport()
    await _drain_stream(transport)
    await transport.aclose()
    assert Client.closed == 1

    # A request after aclose transparently recreates the client.
    await _drain_stream(transport)
    assert Client.constructed == 2

    # aclose on an already-closed / never-used transport is a harmless no-op.
    await transport.aclose()
    fresh = HttpxOpenAITransport()
    await fresh.aclose()


@pytest.mark.anyio
async def test_shared_async_client_is_lazy_and_recreates_after_close(monkeypatch) -> None:
    class Client(_CountingClient):
        constructed = 0
        closed = 0

    monkeypatch.setattr("cayu.providers._http.httpx.AsyncClient", Client)

    shared = SharedAsyncClient()
    # Constructing the holder opens no sockets.
    assert Client.constructed == 0

    first = shared.get()
    assert Client.constructed == 1
    # Repeated get() reuses the same live client.
    assert shared.get() is first
    assert Client.constructed == 1

    await shared.aclose()
    assert Client.closed == 1
    # After close, get() builds a fresh client rather than handing back a dead one.
    second = shared.get()
    assert second is not first
    assert Client.constructed == 2


async def _iter_lines(lines: list[str]):
    for line in lines:
        yield line


def test_validate_url_uses_provider_label_and_https_default() -> None:
    assert (
        validate_url("https://api.openai.com", "url", provider_label="OpenAI")
        == "https://api.openai.com"
    )
    with pytest.raises(ValueError, match="OpenAI url must use https."):
        validate_url("http://api.openai.com", "url", provider_label="OpenAI")
    with pytest.raises(ValueError, match="must include a host"):
        validate_url("https://", "url", provider_label="OpenAI")


def test_validate_url_allow_http_opt_in_and_hint() -> None:
    assert (
        validate_url(
            "http://localhost:11434",
            "url",
            provider_label="Chat Completions",
            allow_http=True,
            allow_http_hint=True,
        )
        == "http://localhost:11434"
    )
    with pytest.raises(ValueError, match=r"set allow_http=True for local http servers"):
        validate_url(
            "http://localhost:11434",
            "url",
            provider_label="Chat Completions",
            allow_http_hint=True,
        )


def test_validate_base_url_strips_trailing_slash() -> None:
    assert (
        validate_base_url("https://api.anthropic.com/", provider_label="Anthropic")
        == "https://api.anthropic.com"
    )
