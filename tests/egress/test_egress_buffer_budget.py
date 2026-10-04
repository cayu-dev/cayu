"""Broker-wide response buffer budget and broker-owned default upstream limits."""

from __future__ import annotations

import asyncio
import contextlib
import gzip
import os
import ssl
import tempfile
import threading
import zlib

import httpx
import pytest
from tests.egress.test_broker_pipeline import (
    REAL_SECRET,
    _destination_resolver,
    _mint,
    _request,
    _stripe_example_policy,
)

import cayu.egress.proxy_server as proxy_server_module
from cayu.egress import (
    CapturedRequest,
    CapturedResponse,
    EgressResponseBuffer,
    EgressResponseBufferBudget,
    EgressUpstreamLimits,
    EgressUpstreamOperation,
    HttpxUpstream,
    TransparentEgressBroker,
    VirtualCredentialRegistry,
)
from cayu.egress._content_decoding import (
    _DECODE_OUTPUT_STEP,
    DecodedContentTooLargeError,
    content_decoder,
)
from cayu.egress.broker import (
    CAYU_EGRESS_ERROR_HEADER,
    MAX_EGRESS_UPSTREAM_TOTAL_TIMEOUT_S,
)
from cayu.egress.proxy_server import TransparentEgressProxyServer
from cayu.vaults import StaticVault

# Small limits so one test response can approach the whole budget.
_SMALL = {"upstream_max_response_bytes": 1000, "browser_max_response_bytes": 1000}


def _broker(
    upstream, *, secret: str = REAL_SECRET, **options
) -> tuple[TransparentEgressBroker, VirtualCredentialRegistry]:  # type: ignore[no-untyped-def]
    registry = VirtualCredentialRegistry()
    broker = TransparentEgressBroker(
        registry=registry,
        resolver=StaticVault({"stripe_test_key": secret}),
        policies={"stripe-example": _stripe_example_policy()},
        upstream=upstream,
        **options,
    )
    return broker, registry


def _identity_upstream(chunks: list[bytes], *, encoding: str | None = None) -> HttpxUpstream:
    class _Body(httpx.AsyncByteStream):
        async def __aiter__(self):  # type: ignore[no-untyped-def]
            for chunk in chunks:
                yield chunk

    async def handler(request: httpx.Request) -> httpx.Response:
        headers = {} if encoding is None else {"Content-Encoding": encoding}
        return httpx.Response(200, headers=headers, stream=_Body(), request=request)

    return HttpxUpstream(
        transport=httpx.MockTransport(handler),
        destination_resolver=_destination_resolver("93.184.216.34"),
    )


class _HeldUpstream:
    """Reserves part of the budget, then waits until the test releases it."""

    def __init__(self, reserve: int, body: bytes) -> None:
        self.reserve = reserve
        self.body = body
        self.reserved = asyncio.Event()
        self.release = asyncio.Event()
        self.buffers: list[EgressResponseBuffer | None] = []

    def prepare(
        self, request: CapturedRequest, *, limits: EgressUpstreamLimits
    ) -> EgressUpstreamOperation:
        async def send() -> CapturedResponse:
            self.buffers.append(limits.response_buffer)
            assert limits.response_buffer is not None
            limits.response_buffer.reserve_to(self.reserve)
            self.reserved.set()
            await self.release.wait()
            return CapturedResponse(status_code=200, body=self.body)

        async def cancel_and_wait(task: asyncio.Task[CapturedResponse]) -> None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        return EgressUpstreamOperation(send, cancel_and_wait=cancel_and_wait)


class _BodyUpstream:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def prepare(
        self, request: CapturedRequest, *, limits: EgressUpstreamLimits
    ) -> EgressUpstreamOperation:
        async def send() -> CapturedResponse:
            return CapturedResponse(status_code=200, body=self.body)

        return EgressUpstreamOperation(send)


def _held(budget: EgressResponseBufferBudget, size: int) -> EgressResponseBuffer:
    held = EgressResponseBuffer(budget)
    held.reserve_to(size)
    return held


def test_default_upstream_takes_the_broker_limits() -> None:
    broker, _registry = _broker(
        None,
        upstream_max_response_bytes=1024 * 1024 * 1024,
        upstream_total_timeout_s=1200,
    )

    upstream = broker._upstream
    assert type(upstream) is HttpxUpstream
    # Before, the default upstream kept its own 256 MiB / 600 s ceiling and the
    # smaller value silently won.
    assert upstream._max_response_bytes == 1024 * 1024 * 1024
    assert upstream._timeout_s == 1200


def test_upstream_timeout_has_an_upper_bound() -> None:
    with pytest.raises(ValueError, match="at most"):
        _broker(None, upstream_total_timeout_s=MAX_EGRESS_UPSTREAM_TOTAL_TIMEOUT_S + 1)
    with pytest.raises(ValueError, match="at most"):
        HttpxUpstream(timeout_s=MAX_EGRESS_UPSTREAM_TOTAL_TIMEOUT_S + 1)


def test_budget_must_hold_one_largest_response() -> None:
    with pytest.raises(ValueError, match="largest response"):
        _broker(None, response_buffer_budget_bytes=999, **_SMALL)
    with pytest.raises(ValueError, match="largest response"):
        # The browser default (64 MiB) is larger than this budget.
        _broker(None, response_buffer_budget_bytes=1000, upstream_max_response_bytes=1000)
    with pytest.raises(ValueError, match="either"):
        _broker(
            None,
            response_buffer_budget_bytes=1000,
            response_buffer_budget=EgressResponseBufferBudget(1000),
            **_SMALL,
        )
    _broker(None, response_buffer_budget_bytes=1000, **_SMALL)


def test_httpx_response_over_the_remaining_budget_is_refused_and_released() -> None:
    budget = EgressResponseBufferBudget(1000)
    other = _held(budget, 600)
    upstream = _identity_upstream([b"x" * 300, b"y" * 300])
    broker, registry = _broker(upstream, response_buffer_budget=budget, **_SMALL)
    grant = _mint(registry)

    response = asyncio.run(broker.handle_request(_request(grant.presented_value, "/v1/customers")))

    assert response.status_code == 503
    assert response.headers[CAYU_EGRESS_ERROR_HEADER] == "upstream_capacity_exhausted"
    assert budget.used_bytes == 600
    other.release()
    assert budget.used_bytes == 0


def test_httpx_response_within_the_budget_is_delivered_and_released() -> None:
    upstream = _identity_upstream([b"x" * 400, b"y" * 400])
    broker, registry = _broker(upstream, response_buffer_budget_bytes=1000, **_SMALL)
    grant = _mint(registry)

    response = asyncio.run(broker.handle_request(_request(grant.presented_value, "/v1/customers")))

    assert response.status_code == 200
    assert response.body == b"x" * 400 + b"y" * 400
    assert broker._response_buffer_budget.used_bytes == 0


def test_custom_upstream_body_is_reserved_after_return() -> None:
    budget = EgressResponseBufferBudget(1000)
    other = _held(budget, 600)
    broker, registry = _broker(_BodyUpstream(b"z" * 500), response_buffer_budget=budget, **_SMALL)
    grant = _mint(registry)

    response = asyncio.run(broker.handle_request(_request(grant.presented_value, "/v1/customers")))

    assert response.status_code == 503
    other.release()
    assert budget.used_bytes == 0


def test_concurrent_responses_share_one_budget() -> None:
    async def run() -> None:
        held = _HeldUpstream(reserve=600, body=b"ok")
        broker, registry = _broker(held, response_buffer_budget_bytes=1000, **_SMALL)
        grant = _mint(registry)
        first = asyncio.create_task(
            broker.handle_request(_request(grant.presented_value, "/v1/customers"))
        )
        await held.reserved.wait()
        assert broker._response_buffer_budget.used_bytes == 600

        held.reserved.clear()
        refused = await broker.handle_request(_request(grant.presented_value, "/v1/customers"))
        assert refused.status_code == 503
        assert refused.headers[CAYU_EGRESS_ERROR_HEADER] == "upstream_capacity_exhausted"

        held.release.set()
        assert (await first).status_code == 200
        assert broker._response_buffer_budget.used_bytes == 0
        assert held.buffers[0] is not held.buffers[1]

    asyncio.run(run())


def test_cancelled_request_releases_its_reservation() -> None:
    async def run() -> None:
        held = _HeldUpstream(reserve=700, body=b"ok")
        broker, registry = _broker(held, response_buffer_budget_bytes=1000, **_SMALL)
        grant = _mint(registry)
        task = asyncio.create_task(
            broker.handle_request(_request(grant.presented_value, "/v1/customers"))
        )
        await held.reserved.wait()
        assert broker._response_buffer_budget.used_bytes == 700
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert broker._response_buffer_budget.used_bytes == 0
        released = held.buffers[0]
        assert released is not None
        with pytest.raises(RuntimeError):
            released.reserve_to(800)

    asyncio.run(run())


def test_holding_entrance_keeps_the_reservation_until_released() -> None:
    broker, registry = _broker(
        _BodyUpstream(b"z" * 500), response_buffer_budget_bytes=1000, **_SMALL
    )
    grant = _mint(registry)

    response, buffers = asyncio.run(
        broker._handle_request_holding_buffer(_request(grant.presented_value, "/v1/customers"))
    )

    assert response.status_code == 200
    assert broker._response_buffer_budget.used_bytes == 500
    for response_buffer in buffers:
        response_buffer.release()
        response_buffer.release()
    assert broker._response_buffer_budget.used_bytes == 0


def test_abandoned_proxy_call_releases_a_finished_but_uncopied_result() -> None:
    # Force the race: the broker task has finished, but the loop has not yet
    # copied its result into the worker's concurrent future when the worker
    # gives up, so cancel() succeeds and the result itself is dropped.
    broker, registry = _broker(
        _BodyUpstream(b"z" * 500), response_buffer_budget_bytes=1000, **_SMALL
    )
    grant = _mint(registry)
    loop = asyncio.new_event_loop()
    runner = threading.Thread(target=loop.run_forever, daemon=True)
    runner.start()
    gate = threading.Event()
    finished = threading.Event()
    original = broker._handle_request_holding_buffer

    async def holding(request: CapturedRequest):  # type: ignore[no-untyped-def]
        result = await original(request)
        # Runs before the task's done-callbacks, so the copy waits on the gate.
        asyncio.get_running_loop().call_soon(lambda: (finished.set(), gate.wait(5)))
        return result

    broker._handle_request_holding_buffer = holding  # type: ignore[method-assign]
    try:
        future, claim = proxy_server_module._submit_broker_request(
            broker, _request(grant.presented_value, "/v1/customers"), loop
        )
        assert finished.wait(5)
        assert broker._response_buffer_budget.used_bytes == 500
        proxy_server_module._abandon_broker_request(future, claim)
        assert future.cancelled()
        gate.set()
        asyncio.run_coroutine_threadsafe(asyncio.sleep(0), loop).result(5)
        assert broker._response_buffer_budget.used_bytes == 0
    finally:
        gate.set()
        loop.call_soon_threadsafe(loop.stop)
        runner.join(5)
        loop.close()


def test_abandon_before_the_offer_releases_when_the_loop_offers() -> None:
    # The worker abandons while the loop is about to offer the reservations;
    # the offer must then release them itself.
    broker, registry = _broker(
        _BodyUpstream(b"z" * 500), response_buffer_budget_bytes=1000, **_SMALL
    )
    grant = _mint(registry)
    loop = asyncio.new_event_loop()
    runner = threading.Thread(target=loop.run_forever, daemon=True)
    runner.start()
    original = broker._handle_request_holding_buffer
    reserved = threading.Event()
    gate = asyncio.Event()

    async def holding(request: CapturedRequest):  # type: ignore[no-untyped-def]
        result = await original(request)
        reserved.set()
        await gate.wait()
        return result

    broker._handle_request_holding_buffer = holding  # type: ignore[method-assign]
    try:
        future, claim = proxy_server_module._submit_broker_request(
            broker, _request(grant.presented_value, "/v1/customers"), loop
        )
        assert reserved.wait(5)
        assert broker._response_buffer_budget.used_bytes == 500
        claim.abandon()
        assert broker._response_buffer_budget.used_bytes == 500
        loop.call_soon_threadsafe(gate.set)
        future.result(5)
        assert broker._response_buffer_budget.used_bytes == 0
    finally:
        loop.call_soon_threadsafe(gate.set)
        loop.call_soon_threadsafe(loop.stop)
        runner.join(5)
        loop.close()


def test_abandoned_proxy_call_before_the_result_releases_on_the_loop() -> None:
    async def run() -> None:
        held = _HeldUpstream(reserve=600, body=b"ok")
        broker, registry = _broker(held, response_buffer_budget_bytes=1000, **_SMALL)
        grant = _mint(registry)
        loop = asyncio.get_running_loop()
        future, claim = proxy_server_module._submit_broker_request(
            broker, _request(grant.presented_value, "/v1/customers"), loop
        )
        await held.reserved.wait()
        proxy_server_module._abandon_broker_request(future, claim)
        for _ in range(20):
            await asyncio.sleep(0)
        assert broker._response_buffer_budget.used_bytes == 0

    asyncio.run(run())


def _proxy_body_writes(
    monkeypatch: pytest.MonkeyPatch, *, fail: bool
) -> tuple[list[int], TransparentEgressBroker, int]:
    body = b"p" * 700
    broker, registry = _broker(_BodyUpstream(body), response_buffer_budget_bytes=1000, **_SMALL)
    grant = _mint(registry)
    observed: list[int] = []
    original = proxy_server_module._send_all

    def send_all(conn, data, *, stop=None):  # type: ignore[no-untyped-def]
        if data == body:
            observed.append(broker._response_buffer_budget.used_bytes)
            if fail:
                raise OSError("client stopped reading")
        return original(conn, data, stop=stop)

    monkeypatch.setattr(proxy_server_module, "_send_all", send_all)

    async def run() -> None:
        server = TransparentEgressProxyServer(broker, loop=asyncio.get_running_loop())
        port = await server.start()
        ca_dir = tempfile.mkdtemp(prefix="cayu-egress-budget-")
        ca_path = os.path.join(ca_dir, "ca.pem")
        with open(ca_path, "wb") as handle:
            handle.write(server.authority.ca_cert_pem())
        try:
            async with httpx.AsyncClient(
                proxy=f"http://127.0.0.1:{port}",
                verify=ssl.create_default_context(cafile=ca_path),
                timeout=15.0,
            ) as client:
                with contextlib.suppress(httpx.HTTPError):
                    await client.post(
                        "https://api.stripe.com/v1/customers",
                        headers={"Authorization": f"Bearer {grant.presented_value}"},
                    )
        finally:
            await server.close()

    asyncio.run(run())
    return observed, broker, len(body)


def test_proxy_holds_the_reservation_while_writing_the_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed, broker, size = _proxy_body_writes(monkeypatch, fail=False)

    assert observed == [size]
    assert broker._response_buffer_budget.used_bytes == 0


def test_proxy_releases_the_reservation_when_the_write_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed, broker, size = _proxy_body_writes(monkeypatch, fail=True)

    assert observed == [size]
    assert broker._response_buffer_budget.used_bytes == 0


def test_reservations_are_thread_safe() -> None:
    budget = EgressResponseBufferBudget(1_000_000)
    buffers = [EgressResponseBuffer(budget) for _ in range(8)]

    def grow(response_buffer: EgressResponseBuffer) -> None:
        for total in range(1, 5001):
            response_buffer.reserve_to(total)

    threads = [threading.Thread(target=grow, args=(item,)) for item in buffers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert budget.used_bytes == 8 * 5000
    releasers = [threading.Thread(target=item.release) for item in buffers * 2]
    for thread in releasers:
        thread.start()
    for thread in releasers:
        thread.join()
    assert budget.used_bytes == 0


@pytest.mark.parametrize(
    ("coding", "encode"),
    [
        ("gzip", gzip.compress),
        ("deflate", zlib.compress),
        ("deflate", lambda data: zlib.compress(data, wbits=-zlib.MAX_WBITS)),
    ],
)
def test_decoder_reserves_before_each_output_step(coding, encode) -> None:  # type: ignore[no-untyped-def]
    decoded = b"a" * (5 * _DECODE_OUTPUT_STEP + 123)
    encoded = encode(decoded)
    limit = 8 * _DECODE_OUTPUT_STEP
    decoder = content_decoder(coding, max_response_bytes=limit)
    assert decoder is not None
    sink = bytearray()
    reservations: list[tuple[int, int]] = []

    def reserve(total: int) -> None:
        reservations.append((total, len(sink)))

    decoder.feed(encoded, sink, limit, reserve)
    decoder.finish(sink, limit, reserve)

    assert bytes(sink) == decoded
    totals = [total for total, _held_now in reservations]
    assert totals == sorted(totals)
    # Each reservation covers what is held plus at most one step (and, for
    # deflate, the retained encoded input).
    for total, held_now in reservations:
        assert total - held_now <= _DECODE_OUTPUT_STEP + len(encoded)
    assert max(totals) >= len(decoded)


def test_decoder_refuses_a_step_the_budget_cannot_hold() -> None:
    limit = 8 * _DECODE_OUTPUT_STEP
    encoded = gzip.compress(b"b" * (4 * _DECODE_OUTPUT_STEP))
    decoder = content_decoder("gzip", max_response_bytes=limit)
    assert decoder is not None
    sink = bytearray()

    def reserve(total: int) -> None:
        if total > 2 * _DECODE_OUTPUT_STEP:
            raise DecodedContentTooLargeError("budget")

    with pytest.raises(DecodedContentTooLargeError):
        decoder.feed(encoded, sink, limit, reserve)
    # Nothing was decoded past what had been reserved.
    assert len(sink) <= 2 * _DECODE_OUTPUT_STEP


class _RecordingBudget(EgressResponseBufferBudget):
    """Records every reservation and the decoded size when each was taken."""

    def __init__(self, limit_bytes: int, sinks: list[bytearray]) -> None:
        super().__init__(limit_bytes)
        self.sinks = sinks
        self.takes: list[tuple[int, int]] = []

    def _take(self, size: int) -> None:
        decoded = max((len(sink) for sink in self.sinks), default=0)
        self.takes.append((self.used_bytes + size, decoded))
        super()._take(size)


def _record_sinks(monkeypatch: pytest.MonkeyPatch) -> list[bytearray]:
    import cayu.egress._content_decoding as decoding

    sinks: list[bytearray] = []
    original = decoding._append

    def append(sink: bytearray, output: bytes, limit: int) -> None:
        if not any(sink is seen for seen in sinks):
            sinks.append(sink)
        original(sink, output, limit)

    monkeypatch.setattr(decoding, "_append", append)
    return sinks


_STEP_LIMITS = {
    "upstream_max_response_bytes": 8 * _DECODE_OUTPUT_STEP,
    "browser_max_response_bytes": 8 * _DECODE_OUTPUT_STEP,
}


def test_default_upstream_reserves_step_by_step_while_decoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sinks = _record_sinks(monkeypatch)
    budget = _RecordingBudget(8 * _DECODE_OUTPUT_STEP, sinks)
    decoded = b"c" * (3 * _DECODE_OUTPUT_STEP + 512 * 1024)
    encoded = gzip.compress(decoded)
    chunks = [encoded[index : index + 4096] for index in range(0, len(encoded), 4096)]
    broker, registry = _broker(
        _identity_upstream(chunks, encoding="gzip"), response_buffer_budget=budget, **_STEP_LIMITS
    )
    grant = _mint(registry)

    response = asyncio.run(broker.handle_request(_request(grant.presented_value, "/v1/customers")))

    assert response.status_code == 200
    assert response.body == decoded
    # The upstream reserved while decoding, not only once after return.
    assert len(budget.takes) > 3
    for total, decoded_before in budget.takes:
        assert total - decoded_before <= _DECODE_OUTPUT_STEP
    assert budget.used_bytes == 0


def test_gzip_bomb_is_refused_before_inflating_past_the_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sinks = _record_sinks(monkeypatch)
    budget = _RecordingBudget(8 * _DECODE_OUTPUT_STEP, sinks)
    other = _held(budget, 6 * _DECODE_OUTPUT_STEP)
    bomb = gzip.compress(b"\0" * (7 * _DECODE_OUTPUT_STEP))
    broker, registry = _broker(
        _identity_upstream([bomb], encoding="gzip"), response_buffer_budget=budget, **_STEP_LIMITS
    )
    grant = _mint(registry)

    response = asyncio.run(broker.handle_request(_request(grant.presented_value, "/v1/customers")))

    assert response.status_code == 503
    # The bomb fit the per-response limit, so only the budget stopped it, and
    # it never decoded more than the two MiB the budget had left.
    assert sinks
    assert max(len(sink) for sink in sinks) <= 2 * _DECODE_OUTPUT_STEP
    assert budget.used_bytes == 6 * _DECODE_OUTPUT_STEP
    other.release()
    assert budget.used_bytes == 0


def test_budget_option_is_validated() -> None:
    with pytest.raises(ValueError):
        EgressResponseBufferBudget(0)
    with pytest.raises(TypeError):
        EgressResponseBufferBudget(1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        _broker(None, response_buffer_budget=object())
    with pytest.raises(TypeError):
        EgressUpstreamLimits(
            max_response_bytes=1,
            total_timeout_s=1,
            response_buffer=object(),  # type: ignore[arg-type]
        )


# A short secret echoed by the upstream grows when scrubbed: 315 -> 595 bytes.
_SHORT_SECRET = "sk_test_x"
_ECHOED = _SHORT_SECRET.encode() * 35
_SCRUBBED_SIZE = 35 * len(b"[REDACTED_SECRET]")


def test_scrubbing_expansion_is_reserved_and_held() -> None:
    broker, registry = _broker(
        _BodyUpstream(_ECHOED), secret=_SHORT_SECRET, response_buffer_budget_bytes=1000, **_SMALL
    )
    grant = _mint(registry)

    response, buffers = asyncio.run(
        broker._handle_request_holding_buffer(_request(grant.presented_value, "/v1/customers"))
    )

    assert response.status_code == 200
    assert _SHORT_SECRET.encode() not in response.body
    assert len(response.body) == _SCRUBBED_SIZE
    # The held reservation covers the expanded body actually handed on.
    assert broker._response_buffer_budget.used_bytes == len(response.body)
    for response_buffer in buffers:
        response_buffer.release()
    assert broker._response_buffer_budget.used_bytes == 0


def test_held_expanded_responses_never_exceed_the_budget() -> None:
    broker, registry = _broker(
        _BodyUpstream(_ECHOED), secret=_SHORT_SECRET, response_buffer_budget_bytes=1000, **_SMALL
    )
    grant = _mint(registry)
    budget = broker._response_buffer_budget

    async def run() -> list[tuple[CapturedResponse, tuple[EgressResponseBuffer, ...]]]:
        return [
            await broker._handle_request_holding_buffer(
                _request(grant.presented_value, "/v1/customers")
            )
            for _ in range(3)
        ]

    results = asyncio.run(run())

    statuses = [response.status_code for response, _buffers in results]
    assert statuses[0] == 200
    # The second's scrubbed 595 bytes do not fit beside the first's 595: refused.
    assert statuses[1:] == [503, 503]
    assert all(
        response.headers.get(CAYU_EGRESS_ERROR_HEADER) == "upstream_capacity_exhausted"
        for response, _buffers in results[1:]
    )
    held = sum(len(response.body) for response, _buffers in results)
    assert budget.used_bytes <= budget.limit_bytes
    # Never more than the bodies actually held; always the expanded one in full.
    # (A small denial body is not reserved when the budget is already short.)
    assert _SCRUBBED_SIZE <= budget.used_bytes <= held
    for _response, buffers in results:
        for response_buffer in buffers:
            response_buffer.release()
    assert budget.used_bytes == 0


@pytest.mark.parametrize(
    "secret", [_SHORT_SECRET, "sk_test_" + "0123456789abcdef" * 2], ids=["grows", "shrinks"]
)
def test_idle_broker_scrubs_a_response_larger_than_half_the_budget(secret: str) -> None:
    # A budget may equal the response limit, so scrubbing must not need twice
    # the body: only growth is reserved.
    body = b"x" * (600 - len(secret)) + secret.encode()
    broker, registry = _broker(
        _BodyUpstream(body), secret=secret, response_buffer_budget_bytes=1000, **_SMALL
    )
    grant = _mint(registry)

    response = asyncio.run(broker.handle_request(_request(grant.presented_value, "/v1/customers")))

    assert response.status_code == 200
    assert secret.encode() not in response.body
    assert broker._response_buffer_budget.used_bytes == 0


def test_scrubbing_refused_by_the_budget_releases_everything() -> None:
    budget = EgressResponseBufferBudget(1000)
    # 500 held + 315 received leaves too little for the 595-byte scrubbed body.
    other = _held(budget, 500)
    broker, registry = _broker(
        _BodyUpstream(_ECHOED), secret=_SHORT_SECRET, response_buffer_budget=budget, **_SMALL
    )
    grant = _mint(registry)

    response = asyncio.run(broker.handle_request(_request(grant.presented_value, "/v1/customers")))

    assert response.status_code == 503
    assert _SHORT_SECRET.encode() not in response.body
    assert budget.used_bytes == 500
    other.release()
    assert budget.used_bytes == 0


def test_proxy_write_failure_releases_an_expanded_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker, registry = _broker(
        _BodyUpstream(_ECHOED), secret=_SHORT_SECRET, response_buffer_budget_bytes=1000, **_SMALL
    )
    grant = _mint(registry)
    observed: list[int] = []
    original = proxy_server_module._send_all

    def send_all(conn, data, *, stop=None):  # type: ignore[no-untyped-def]
        if len(data) == _SCRUBBED_SIZE and _SHORT_SECRET.encode() not in data:
            observed.append(broker._response_buffer_budget.used_bytes)
            raise OSError("client stopped reading")
        return original(conn, data, stop=stop)

    monkeypatch.setattr(proxy_server_module, "_send_all", send_all)

    async def run() -> None:
        server = TransparentEgressProxyServer(broker, loop=asyncio.get_running_loop())
        port = await server.start()
        ca_dir = tempfile.mkdtemp(prefix="cayu-egress-scrub-")
        ca_path = os.path.join(ca_dir, "ca.pem")
        with open(ca_path, "wb") as handle:
            handle.write(server.authority.ca_cert_pem())
        try:
            async with httpx.AsyncClient(
                proxy=f"http://127.0.0.1:{port}",
                verify=ssl.create_default_context(cafile=ca_path),
                timeout=15.0,
            ) as client:
                with contextlib.suppress(httpx.HTTPError):
                    await client.post(
                        "https://api.stripe.com/v1/customers",
                        headers={"Authorization": f"Bearer {grant.presented_value}"},
                    )
        finally:
            await server.close()

    asyncio.run(run())

    assert observed == [_SCRUBBED_SIZE]
    assert broker._response_buffer_budget.used_bytes == 0
