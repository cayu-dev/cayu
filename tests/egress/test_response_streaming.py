"""Incremental (streamed) upstream responses through the egress broker and proxy."""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import ssl
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import httpx
import pytest
from tests.egress.test_broker_pipeline import (
    REAL_SECRET,
    _destination_resolver,
    _mint,
    _request,
    _stripe_example_policy,
)

from cayu.egress import (
    CapturedResponse,
    EgressDecision,
    EgressResponseStream,
    HttpxUpstream,
    TransparentEgressBroker,
    VirtualCredentialRegistry,
)
from cayu.egress.broker import _ResponseStreamChannel
from cayu.egress.proxy_server import TransparentEgressProxyServer
from cayu.vaults import REDACTED_SECRET, StaticVault

pytest.importorskip("cryptography")

_SSE = {"Content-Type": "text/event-stream; charset=utf-8"}


class _StreamingOrigin:
    """An HTTPX mock origin whose body is produced by an async generator."""

    def __init__(
        self,
        body: Callable[[], AsyncIterator[bytes]],
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._body = body
        self._headers = _SSE if headers is None else headers
        self.requests: list[httpx.Request] = []

    def upstream(self) -> HttpxUpstream:
        origin = self

        class _Body(httpx.AsyncByteStream):
            async def __aiter__(self) -> AsyncIterator[bytes]:
                async for chunk in origin._body():
                    yield chunk

        async def handler(request: httpx.Request) -> httpx.Response:
            origin.requests.append(request)
            return httpx.Response(200, headers=self._headers, stream=_Body(), request=request)

        return HttpxUpstream(
            transport=httpx.MockTransport(handler),
            destination_resolver=_destination_resolver("93.184.216.34"),
        )


def _broker(
    upstream: Any, **options: Any
) -> tuple[TransparentEgressBroker, VirtualCredentialRegistry, list[EgressDecision]]:
    registry = VirtualCredentialRegistry()
    decisions: list[EgressDecision] = []
    broker = TransparentEgressBroker(
        registry=registry,
        resolver=StaticVault({"stripe_test_key": REAL_SECRET}),
        policies={"stripe-example": _stripe_example_policy()},
        upstream=upstream,
        audit=decisions.append,
        **options,
    )
    return broker, registry, decisions


async def _with_proxy_client(
    broker: TransparentEgressBroker,
    use: Callable[[httpx.AsyncClient], Awaitable[Any]],
) -> Any:
    server = TransparentEgressProxyServer(broker, loop=asyncio.get_running_loop())
    port = await server.start()
    ca_dir = tempfile.mkdtemp(prefix="cayu-egress-stream-test-")
    ca_path = os.path.join(ca_dir, "ca.pem")
    with open(ca_path, "wb") as handle:
        handle.write(server.authority.ca_cert_pem())
    try:
        async with httpx.AsyncClient(
            proxy=f"http://127.0.0.1:{port}",
            verify=ssl.create_default_context(cafile=ca_path),
            timeout=10.0,
        ) as client:
            return await use(client)
    finally:
        await server.close()
        shutil.rmtree(ca_dir, ignore_errors=True)


async def _wait_until(condition: Callable[[], bool], *, what: str) -> None:
    for _ in range(500):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"Timed out waiting for {what}.")


def test_slow_streamed_response_reaches_the_client_before_the_upstream_finishes() -> None:
    async def run() -> None:
        release = asyncio.Event()
        # The secret straddles two later chunks so scrubbing must hold back a
        # possible prefix without delaying the complete first event.
        first = b"data: one\n\n"
        rest = [
            b'data: {"echo":"' + REAL_SECRET[:9].encode(),
            REAL_SECRET[9:].encode() + b'"}\n\n',
            b"data: [DONE]\n\n",
        ]

        async def body() -> AsyncIterator[bytes]:
            yield first
            # Nothing more arrives until the client has seen the first event.
            await release.wait()
            for chunk in rest:
                yield chunk

        origin = _StreamingOrigin(body)
        broker, registry, decisions = _broker(origin.upstream())
        grant = _mint(registry)

        async def use(client: httpx.AsyncClient) -> None:
            async with client.stream(
                "POST",
                "https://api.stripe.com/v1/customers",
                headers={"Authorization": f"Bearer {grant.presented_value}"},
            ) as response:
                assert response.status_code == 200
                assert response.headers["transfer-encoding"] == "chunked"
                assert "content-length" not in response.headers
                chunks = response.aiter_raw()
                received = await asyncio.wait_for(anext(chunks), timeout=10.0)
                assert received == first
                # The broker has not recorded a decision while the body is open.
                assert decisions == []
                release.set()
                async for chunk in chunks:
                    received += chunk
            assert received == (
                first
                + b'data: {"echo":"'
                + REDACTED_SECRET.encode()
                + b'"}\n\n'
                + b"data: [DONE]\n\n"
            )

        await _with_proxy_client(broker, use)
        # Credential handling is unchanged: the real key went only upstream.
        assert origin.requests[0].headers["Authorization"] == f"Bearer {REAL_SECRET}"
        assert [(d.allowed, d.status_code, d.error_code) for d in decisions] == [(True, 200, None)]
        assert not broker._active_upstream_operations
        assert broker._response_buffer_budget.used_bytes == 0
        assert not registry._active_counts

    asyncio.run(run())


@pytest.mark.parametrize("split", range(1, len(REAL_SECRET)))
def test_streamed_scrubbing_redacts_a_secret_split_at_any_chunk_boundary(split: int) -> None:
    payload = b"data: " + REAL_SECRET.encode() + b"\n\n"
    cut = len(b"data: ") + split

    async def body() -> AsyncIterator[bytes]:
        yield payload[:cut]
        yield payload[cut:]

    async def run() -> list[tuple[str, object]]:
        broker, registry, _decisions = _broker(_StreamingOrigin(body).upstream())
        grant = _mint(registry)
        channel = _ResponseStreamChannel(asyncio.get_running_loop())
        response, buffers = await broker._handle_request_holding_buffer(
            _request(grant.presented_value, "/v1/customers"),
            response_stream=channel,
        )
        for response_buffer in buffers:
            response_buffer.release()
        assert response.body == b""
        items = []
        while (item := channel.get(0)) is not None:
            items.append(item)
        return items

    items = asyncio.run(run())

    assert items[0][0] == "head"
    assert items[-1] == ("end", None)
    data = b"".join(payload for kind, payload in items if kind == "data")  # type: ignore[misc]
    assert data == b"data: " + REDACTED_SECRET.encode() + b"\n\n"


def test_streamed_response_over_the_byte_limit_is_truncated_after_its_head() -> None:
    async def run() -> None:
        release = asyncio.Event()

        async def body() -> AsyncIterator[bytes]:
            yield b"data: " + b"a" * 40 + b"\n\n"
            await release.wait()
            yield b"data: " + b"b" * 40 + b"\n\n"

        broker, registry, decisions = _broker(
            _StreamingOrigin(body).upstream(),
            upstream_max_response_bytes=64,
            browser_max_response_bytes=64,
        )
        grant = _mint(registry)

        async def use(client: httpx.AsyncClient) -> None:
            async with client.stream(
                "POST",
                "https://api.stripe.com/v1/customers",
                headers={"Authorization": f"Bearer {grant.presented_value}"},
            ) as response:
                chunks = response.aiter_raw()
                first = await asyncio.wait_for(anext(chunks), timeout=10.0)
                assert first.startswith(b"data: aaa")
                release.set()
                # The body ends without the terminal chunk, so the client
                # cannot mistake the truncated stream for a complete one.
                with pytest.raises(httpx.RemoteProtocolError):
                    async for _chunk in chunks:
                        pass

        await _with_proxy_client(broker, use)
        assert [(d.allowed, d.status_code, d.error_code) for d in decisions] == [
            (False, 502, "oversized_response")
        ]
        assert not broker._active_upstream_operations
        assert broker._response_buffer_budget.used_bytes == 0
        assert not registry._active_counts

    asyncio.run(run())


def test_streamed_response_holds_at_most_one_window_for_a_slow_client() -> None:
    async def body() -> AsyncIterator[bytes]:
        for _ in range(16):
            yield b"x" * (64 * 1024)

    async def run() -> tuple[int, int]:
        broker, registry, _decisions = _broker(_StreamingOrigin(body).upstream())
        grant = _mint(registry)
        channel = _ResponseStreamChannel(asyncio.get_running_loop(), max_pending_bytes=128 * 1024)
        task = asyncio.create_task(
            broker._handle_request_holding_buffer(
                _request(grant.presented_value, "/v1/customers"),
                response_stream=channel,
            )
        )
        # Nobody reads, so the broker stops at the window instead of buffering.
        await _wait_until(lambda: channel.pending_bytes >= 128 * 1024, what="a full window")
        await asyncio.sleep(0.05)
        pending, reserved = channel.pending_bytes, broker._response_buffer_budget.used_bytes
        assert not task.done()
        received = 0
        while not task.done() or channel.pending_bytes:
            item = await asyncio.to_thread(channel.get, 0.05)
            if item is not None and item[0] == "data":
                received += len(item[1])  # type: ignore[arg-type]
        _response, buffers = task.result()
        for response_buffer in buffers:
            response_buffer.release()
        assert received == 16 * 64 * 1024
        return pending, reserved

    pending, reserved = asyncio.run(run())

    assert pending == 128 * 1024
    assert reserved <= 3 * 64 * 1024


def test_denied_streaming_request_never_reaches_the_upstream() -> None:
    async def run() -> None:
        async def body() -> AsyncIterator[bytes]:
            yield b"data: never\n\n"

        origin = _StreamingOrigin(body)
        broker, registry, decisions = _broker(origin.upstream())
        grant = _mint(registry)

        async def use(client: httpx.AsyncClient) -> httpx.Response:
            return await client.post(
                "https://api.stripe.com/v1/payouts",
                headers={"Authorization": f"Bearer {grant.presented_value}"},
            )

        response = await _with_proxy_client(broker, use)
        assert response.status_code == 403
        assert origin.requests == []
        assert [(d.allowed, d.error_code) for d in decisions] == [(False, "destination_denied")]

    asyncio.run(run())


def test_non_streaming_media_type_stays_buffered_with_content_length() -> None:
    async def run() -> None:
        async def body() -> AsyncIterator[bytes]:
            yield b'{"id":"cus_1",'
            yield b'"key":"' + REAL_SECRET.encode() + b'"}'

        broker, registry, decisions = _broker(
            _StreamingOrigin(body, headers={"Content-Type": "application/json"}).upstream()
        )
        grant = _mint(registry)

        async def use(client: httpx.AsyncClient) -> httpx.Response:
            return await client.post(
                "https://api.stripe.com/v1/customers",
                headers={"Authorization": f"Bearer {grant.presented_value}"},
            )

        response = await _with_proxy_client(broker, use)
        assert response.status_code == 200
        assert "transfer-encoding" not in response.headers
        assert response.headers["content-length"] == str(len(response.content))
        assert response.content == b'{"id":"cus_1","key":"' + REDACTED_SECRET.encode() + b'"}'
        assert [(d.allowed, d.status_code) for d in decisions] == [(True, 200)]

    asyncio.run(run())


def test_client_disconnect_mid_stream_stops_the_upstream_and_releases_capacity() -> None:
    async def run() -> int:
        keepalives = 0

        async def body() -> AsyncIterator[bytes]:
            nonlocal keepalives
            yield b"data: one\n\n"
            # An injected HTTPX transport cannot be cancelled, so only the
            # stream refusing the next chunk can stop this origin.
            while True:
                await asyncio.sleep(0.02)
                keepalives += 1
                yield b": keepalive\n\n"

        broker, registry, _decisions = _broker(_StreamingOrigin(body).upstream())
        grant = _mint(registry)

        async def use(client: httpx.AsyncClient) -> None:
            async with client.stream(
                "POST",
                "https://api.stripe.com/v1/customers",
                headers={"Authorization": f"Bearer {grant.presented_value}"},
            ) as response:
                chunks = response.aiter_raw()
                assert await asyncio.wait_for(anext(chunks), timeout=10.0) == b"data: one\n\n"
            # Leaving the block closes the connection mid-body.
            await _wait_until(
                lambda: not broker._active_upstream_operations and not registry._active_counts,
                what="upstream capacity and the grant lease to be released",
            )

        await _with_proxy_client(broker, use)
        assert broker._response_buffer_budget.used_bytes == 0
        stopped_at = keepalives
        await asyncio.sleep(0.1)
        assert keepalives == stopped_at
        return stopped_at

    assert asyncio.run(run()) >= 1


def test_custom_upstream_can_ignore_the_stream_and_stay_buffered() -> None:
    from cayu.egress import EgressUpstreamLimits, EgressUpstreamOperation

    seen: list[EgressResponseStream | None] = []

    class _BufferedUpstream:
        def prepare(self, request: Any, *, limits: EgressUpstreamLimits) -> EgressUpstreamOperation:
            seen.append(limits.response_stream)

            async def send() -> CapturedResponse:
                return CapturedResponse(status_code=200, headers=_SSE, body=b"data: x\n\n")

            return EgressUpstreamOperation(send)

    async def run() -> tuple[CapturedResponse, list[tuple[str, object]]]:
        broker, registry, _decisions = _broker(_BufferedUpstream())
        grant = _mint(registry)
        channel = _ResponseStreamChannel(asyncio.get_running_loop())
        response, buffers = await broker._handle_request_holding_buffer(
            _request(grant.presented_value, "/v1/customers"),
            response_stream=channel,
        )
        for response_buffer in buffers:
            response_buffer.release()
        items = []
        while (item := channel.get(0)) is not None:
            items.append(item)
        return response, items

    response, items = asyncio.run(run())

    assert isinstance(seen[0], EgressResponseStream)
    assert response.body == b"data: x\n\n"
    assert items == []
    with pytest.raises(TypeError):
        EgressResponseStream()


def test_custom_upstream_stream_is_bounded_by_the_broker() -> None:
    from cayu.egress import EgressUpstreamLimits, EgressUpstreamOperation

    raised: list[BaseException] = []

    class _StreamingUpstream:
        def prepare(self, request: Any, *, limits: EgressUpstreamLimits) -> EgressUpstreamOperation:
            stream = limits.response_stream
            assert stream is not None

            async def send() -> CapturedResponse:
                head = CapturedResponse(status_code=200, headers=_SSE)
                assert await stream.start(head)
                await stream.write(b"x" * 40)
                try:
                    # Ignores the upstream read limit; the broker still stops it.
                    await stream.write(b"y" * 40)
                except Exception as error:
                    raised.append(error)
                    raise
                return head

            return EgressUpstreamOperation(send)

    async def run() -> tuple[CapturedResponse, list[tuple[str, object]], list[EgressDecision]]:
        broker, registry, decisions = _broker(
            _StreamingUpstream(),
            upstream_max_response_bytes=64,
            browser_max_response_bytes=64,
        )
        grant = _mint(registry)
        channel = _ResponseStreamChannel(asyncio.get_running_loop())
        response, buffers = await broker._handle_request_holding_buffer(
            _request(grant.presented_value, "/v1/customers"),
            response_stream=channel,
        )
        for response_buffer in buffers:
            response_buffer.release()
        items = []
        while (item := channel.get(0)) is not None:
            items.append(item)
        return response, items, decisions

    _response, items, decisions = asyncio.run(run())

    assert len(raised) == 1
    assert [kind for kind, _payload in items] == ["head", "data", "abort"]
    assert [(d.allowed, d.error_code) for d in decisions] == [(False, "oversized_response")]


def test_revoking_authority_mid_stream_truncates_the_response() -> None:
    async def body() -> AsyncIterator[bytes]:
        yield b"data: one\n\n"
        while True:
            await asyncio.sleep(0.02)
            yield b": keepalive\n\n"

    async def run() -> tuple[list[str], list[EgressDecision], bool]:
        broker, registry, decisions = _broker(_StreamingOrigin(body).upstream())
        grant = _mint(registry)
        channel = _ResponseStreamChannel(asyncio.get_running_loop())
        task = asyncio.create_task(
            broker._handle_request_holding_buffer(
                _request(grant.presented_value, "/v1/customers"),
                response_stream=channel,
            )
        )
        kinds: list[str] = []
        while "data" not in kinds:
            item = await asyncio.to_thread(channel.get, 0.05)
            if item is not None:
                kinds.append(item[0])
        await asyncio.wait_for(broker.revoke_authority_and_wait([grant.presented_value]), 10.0)
        _response, buffers = await asyncio.wait_for(task, 10.0)
        for response_buffer in buffers:
            response_buffer.release()
        while (item := channel.get(0)) is not None:
            kinds.append(item[0])
        return kinds, decisions, not registry._active_counts

    kinds, decisions, released = asyncio.run(run())

    assert kinds[0] == "head"
    assert kinds[-1] == "abort"
    assert "end" not in kinds
    # Revocation settles the upstream call, so the stream records a denial.
    assert [d.allowed for d in decisions] == [False]
    assert released is True


def test_client_disconnect_from_a_cancellable_stream_still_records_a_decision() -> None:
    from cayu.egress import EgressUpstreamLimits, EgressUpstreamOperation

    async def run() -> tuple[list[EgressDecision], bool]:
        cancelled = asyncio.Event()

        class _CancellableStreamingUpstream:
            def prepare(
                self, request: Any, *, limits: EgressUpstreamLimits
            ) -> EgressUpstreamOperation:
                stream = limits.response_stream
                assert stream is not None

                async def send() -> CapturedResponse:
                    assert await stream.start(CapturedResponse(status_code=200, headers=_SSE))
                    await stream.write(b"data: one\n\n")
                    try:
                        # Never writes again, so only cancellation can stop it.
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        cancelled.set()
                        raise
                    raise AssertionError("unreachable")

                async def cancel_and_wait(task: asyncio.Task[CapturedResponse]) -> None:
                    task.cancel()
                    with contextlib.suppress(BaseException):
                        await task

                return EgressUpstreamOperation(send, cancel_and_wait=cancel_and_wait)

        broker, registry, decisions = _broker(_CancellableStreamingUpstream())
        grant = _mint(registry)

        async def use(client: httpx.AsyncClient) -> None:
            async with client.stream(
                "POST",
                "https://api.stripe.com/v1/customers",
                headers={"Authorization": f"Bearer {grant.presented_value}"},
            ) as response:
                chunks = response.aiter_raw()
                assert await asyncio.wait_for(anext(chunks), timeout=10.0) == b"data: one\n\n"
            await _wait_until(
                lambda: not broker._active_upstream_operations and not registry._active_counts,
                what="upstream capacity and the grant lease to be released",
            )

        await _with_proxy_client(broker, use)
        assert broker._response_buffer_budget.used_bytes == 0
        return decisions, cancelled.is_set()

    decisions, cancelled = asyncio.run(run())

    assert cancelled is True
    assert [(d.allowed, d.status_code, d.error_code) for d in decisions] == [
        (False, 502, "client_disconnected")
    ]


def test_stream_that_finishes_between_polls_is_still_relayed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import time

    import cayu.egress.proxy_server as proxy_server_module

    class _StaleStartChannel(_ResponseStreamChannel):
        """Reports "not started" once, after the broker has already finished."""

        stale_reads = 0

        @property
        def started(self) -> bool:
            if _StaleStartChannel.stale_reads == 0:
                _StaleStartChannel.stale_reads += 1
                deadline = time.monotonic() + 5.0
                while not self._closed and time.monotonic() < deadline:
                    time.sleep(0.01)
                # Give the broker future time to complete as well.
                time.sleep(0.2)
                return False
            return self._started

    monkeypatch.setattr(proxy_server_module, "_ResponseStreamChannel", _StaleStartChannel)

    async def body() -> AsyncIterator[bytes]:
        yield b"data: one\n\n"
        yield b"data: two\n\n"

    async def run() -> tuple[httpx.Response, int]:
        broker, registry, _decisions = _broker(_StreamingOrigin(body).upstream())
        grant = _mint(registry)

        async def use(client: httpx.AsyncClient) -> httpx.Response:
            return await client.post(
                "https://api.stripe.com/v1/customers",
                headers={"Authorization": f"Bearer {grant.presented_value}"},
            )

        response = await _with_proxy_client(broker, use)
        return response, broker._response_buffer_budget.used_bytes

    response, held = asyncio.run(run())

    assert _StaleStartChannel.stale_reads == 1
    assert response.status_code == 200
    assert response.content == b"data: one\n\ndata: two\n\n"
    assert held == 0


def test_finished_stream_keeps_its_reservation_until_queued_chunks_are_taken() -> None:
    async def body() -> AsyncIterator[bytes]:
        yield b"x" * (64 * 1024)
        yield b"y" * (64 * 1024)

    async def run() -> tuple[int, int, int]:
        broker, registry, _decisions = _broker(_StreamingOrigin(body).upstream())
        grant = _mint(registry)
        channel = _ResponseStreamChannel(asyncio.get_running_loop())
        _response, buffers = await broker._handle_request_holding_buffer(
            _request(grant.presented_value, "/v1/customers"),
            response_stream=channel,
        )
        # The broker is done, but the whole body still waits for the client.
        for response_buffer in buffers:
            response_buffer.release()
        held_while_queued = broker._response_buffer_budget.used_bytes
        queued = channel.pending_bytes
        while (item := channel.get(0)) is not None:
            if item[0] in {"end", "abort"}:
                break
        return held_while_queued, queued, broker._response_buffer_budget.used_bytes

    held_while_queued, queued, held_after = asyncio.run(run())

    assert queued == 128 * 1024
    assert held_while_queued >= queued
    assert held_after == 0
