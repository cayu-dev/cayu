from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from cayu.applications import CayuApp
from cayu.server import (
    AuthContext,
    AuthenticatedAccess,
    OpenAccess,
    RequestTimingConfig,
    ServerConfig,
    create_server,
    mount_cayu,
)
from cayu.server._request_timing import (
    RequestTimingMiddleware,
    RequestTimingRecord,
    RequestTimingRecorder,
)


class _Clock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _recorder(
    config: RequestTimingConfig | None = None,
) -> tuple[RequestTimingRecorder, _Clock, _Clock]:
    clock = _Clock()
    cpu = _Clock()
    recorder = RequestTimingRecorder(
        config or RequestTimingConfig(),
        clock=clock,
        cpu_clock=cpu,
    )
    return recorder, clock, cpu


def _get(
    recorder: RequestTimingRecorder,
    clock: _Clock,
    *,
    route: str = "/api/state",
    wall: float = 0.01,
    body: bytes = b'{"jobs": 3}',
    client: str = "client-a",
    status: int = 200,
) -> RequestTimingRecord | None:
    pending = recorder.begin()
    clock.advance(wall)
    return recorder.finish(
        pending,
        method="GET",
        route=route,
        status=status,
        response_bytes=len(body),
        client_key=client,
        streaming=False,
        body_digest=hashlib.blake2b(body, digest_size=16).digest(),
    )


def _messages(caplog: pytest.LogCaptureFixture, event: str) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == event]


def _operator_auth(request: Request) -> AuthContext:
    if request.headers.get("authorization") != "Bearer operator-token":
        raise HTTPException(status_code=401, detail="Operator authentication required.")
    return AuthContext(subject="operator")


def test_records_route_templates_and_keyed_client_hashes_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    raw_values = (
        "job-8f3a91",
        "session-77c2e0",
        "query-secret-value",
        "cookie-secret-value",
        "token-secret-value",
        "agent-secret-value",
        "testclient",
    )
    host = FastAPI()

    @host.get("/api/jobs/{job_id}")
    async def read_job(job_id: str) -> dict[str, str]:
        del job_id
        return {"status": "ok"}

    mount_cayu(
        host,
        CayuApp(enable_logging=False),
        access=OpenAccess(),
        dashboard=False,
        observe_host_requests=True,
        # Every GET is "slow" so warnings are logged and can be checked too.
        request_timing=RequestTimingConfig(slow_get_threshold_ms=0.001, slow_get_min_requests=1),
    )
    recorder = host.state.cayu_request_timing

    with caplog.at_level(logging.DEBUG, logger="cayu"), TestClient(host) as client:
        client.get(
            "/api/jobs/job-8f3a91",
            params={"token": "query-secret-value"},
            headers={"Cookie": "session=cookie-secret-value", "User-Agent": "agent-secret-value"},
        )
        client.get("/api/jobs/job-8f3a91", headers={"Authorization": "Bearer token-secret-value"})
        client.get("/cayu/api/sessions/session-77c2e0", params={"cursor": "query-secret-value"})
        client.get("/untracked/job-8f3a91", params={"q": "query-secret-value"})
        summary = client.get("/cayu/api/diagnostics/requests").json()

    records = recorder.records()
    assert [(record.method, record.route) for record in records] == [
        ("GET", "/api/jobs/{job_id}"),
        ("GET", "/api/jobs/{job_id}"),
        ("GET", "/cayu/api/sessions/{session_id}"),
        ("GET", "(unmatched)"),
    ]
    assert all(re.fullmatch(r"[0-9a-f]{16}", record.client_key) for record in records)
    # Cookie and Authorization clients hash differently; nothing raw is kept.
    assert records[0].client_key != records[1].client_key
    assert {route["route"] for route in summary["routes"]} == {
        "/api/jobs/{job_id}",
        "/cayu/api/sessions/{session_id}",
        "(unmatched)",
    }
    cayu_logs = [record for record in caplog.records if record.name.startswith("cayu")]
    assert any(record.name == "cayu.server.slow_request" for record in cayu_logs)
    stored = "\n".join(
        [
            repr(records),
            json.dumps(summary),
            *(record.getMessage() for record in cayu_logs),
            *(repr(vars(record)) for record in cayu_logs),
        ]
    )
    for value in raw_values:
        assert value not in stored


def test_mount_records_only_cayu_routes_unless_host_requests_are_observed() -> None:
    host = FastAPI()

    @host.get("/api/state")
    async def state() -> dict[str, int]:
        return {"jobs": 0}

    mount_cayu(host, CayuApp(enable_logging=False), access=OpenAccess(), dashboard=False)
    recorder = host.state.cayu_request_timing

    with TestClient(host) as client:
        client.get("/api/state")
        client.get("/cayu/api/health")
        client.get("/cayu/api/diagnostics/requests")

    # The summary route itself is excluded so measuring adds no cost.
    assert [record.route for record in recorder.records()] == ["/cayu/api/health"]


def test_request_timing_defaults_on_for_local_development_only() -> None:
    local = create_server(CayuApp(enable_logging=False), config=ServerConfig.local_development())
    assert isinstance(local.state.cayu_request_timing, RequestTimingRecorder)
    assert local.state.cayu_server_config_summary["request_timing"] == {"enabled": True}

    protected = create_server(
        CayuApp(enable_logging=False),
        config=ServerConfig.protected(_operator_auth),
    )
    assert getattr(protected.state, "cayu_request_timing", None) is None
    opted_in = create_server(
        CayuApp(enable_logging=False),
        config=ServerConfig.protected(_operator_auth, request_timing=RequestTimingConfig()),
    )
    assert isinstance(opted_in.state.cayu_request_timing, RequestTimingRecorder)

    open_host = FastAPI()
    mount_cayu(open_host, CayuApp(enable_logging=False), access=OpenAccess(), dashboard=False)
    assert isinstance(open_host.state.cayu_request_timing, RequestTimingRecorder)

    authenticated_host = FastAPI()
    mount_cayu(
        authenticated_host,
        CayuApp(enable_logging=False),
        access=AuthenticatedAccess(dependency=_operator_auth),
        dashboard=False,
    )
    assert getattr(authenticated_host.state, "cayu_request_timing", None) is None

    explicit_host = FastAPI()
    mount_cayu(
        explicit_host,
        CayuApp(enable_logging=False),
        access=AuthenticatedAccess(dependency=_operator_auth),
        dashboard=False,
        request_timing=True,
    )
    assert isinstance(explicit_host.state.cayu_request_timing, RequestTimingRecorder)

    with pytest.raises(ValueError, match="observe_host_requests requires request timing"):
        mount_cayu(
            FastAPI(),
            CayuApp(enable_logging=False),
            access=OpenAccess(),
            dashboard=False,
            request_timing=False,
            observe_host_requests=True,
        )


def test_request_summary_route_is_operator_only() -> None:
    server = create_server(
        CayuApp(enable_logging=False),
        config=ServerConfig.protected(_operator_auth, request_timing=RequestTimingConfig()),
    )
    operator = {"Authorization": "Bearer operator-token"}
    with TestClient(server) as client:
        assert client.get("/api/diagnostics/requests").status_code == 401
        client.get("/api/health")
        response = client.get(
            "/api/diagnostics/requests",
            params={"since_seconds": 60, "vcpu": 0.5},
            headers=operator,
        )

    assert response.status_code == 200
    assert response.headers["cache-control"] == "private, no-store"
    body = response.json()
    assert body["enabled"] is True
    assert body["vcpu"] == 0.5
    assert body["requested_window_seconds"] == 60
    assert [route["route"] for route in body["routes"]] == ["/api/health"]

    disabled = create_server(
        CayuApp(enable_logging=False),
        config=ServerConfig.protected(_operator_auth),
    )
    with TestClient(disabled) as client:
        body = client.get("/api/diagnostics/requests", headers=operator).json()
    assert body["enabled"] is False
    assert body["routes"] == []


def test_cpu_time_is_apportioned_across_requests_in_flight() -> None:
    recorder, clock, cpu = _recorder()

    def finish(pending, route: str) -> RequestTimingRecord:
        record = recorder.finish(
            pending,
            method="GET",
            route=route,
            status=200,
            response_bytes=0,
            client_key="client",
            streaming=False,
        )
        assert record is not None
        return record

    first = recorder.begin()
    cpu.advance(0.2)
    second = recorder.begin()
    cpu.advance(0.2)
    first_record = finish(first, "/first")
    cpu.advance(0.1)
    second_record = finish(second, "/second")
    # CPU used while nothing is in flight is not charged to any request.
    cpu.advance(5.0)
    idle_record = finish(recorder.begin(), "/idle")

    assert first_record.cpu_seconds == pytest.approx(0.3)
    assert second_record.cpu_seconds == pytest.approx(0.2)
    assert idle_record.cpu_seconds == 0.0
    del clock


def test_summary_reports_rate_cpu_and_vcpu_share_over_the_covered_window() -> None:
    recorder, clock, cpu = _recorder()
    clock.advance(600)
    for index in range(30):
        pending = recorder.begin()
        cpu.advance(0.01)
        clock.advance(0.05 if index % 2 else 0.35)
        recorder.finish(
            pending,
            method="GET",
            route="/api/state",
            status=200,
            response_bytes=100,
            client_key="client-a",
            streaming=False,
        )
        clock.advance(1.95 if index % 2 else 1.65)

    summary = recorder.summary(since_seconds=60, vcpu=0.5)

    assert summary.enabled is True
    assert summary.window_seconds == 60
    assert summary.truncated is False
    assert summary.requests == 30
    assert summary.requests_per_minute == 30
    assert summary.cpu_seconds == pytest.approx(0.3)
    assert summary.cpu_seconds_per_minute == pytest.approx(0.3)
    assert summary.vcpu_share == pytest.approx(0.01)
    (route,) = summary.routes
    assert (route.method, route.route, route.requests, route.clients) == (
        "GET",
        "/api/state",
        30,
        1,
    )
    assert route.wall_ms_p50 == pytest.approx(50)
    assert route.wall_ms_p95 == pytest.approx(350)
    assert route.response_bytes == 3000

    # A recently started process only covers its own uptime.
    fresh, fresh_clock, _ = _recorder()
    fresh_clock.advance(30)
    _get(fresh, fresh_clock)
    assert fresh.summary(since_seconds=300).window_seconds == pytest.approx(30.01)


def test_summary_flags_windows_the_ring_buffer_no_longer_covers() -> None:
    recorder, clock, _ = _recorder(RequestTimingConfig(buffer_size=100))
    for _ in range(150):
        _get(recorder, clock)
        clock.advance(1)

    summary = recorder.summary(since_seconds=3600)

    assert summary.truncated is True
    assert summary.requests == 100
    assert summary.buffer_capacity == 100
    # Coverage starts where the newest evicted record ended, not an hour ago.
    assert summary.window_seconds == pytest.approx(151.5 - 49.5)


def test_slow_get_warns_once_per_route_and_window(caplog: pytest.LogCaptureFixture) -> None:
    recorder, clock, _ = _recorder(
        RequestTimingConfig(slow_get_window_seconds=60, slow_get_min_requests=3)
    )
    with caplog.at_level(logging.WARNING, logger="cayu.server"):
        for _ in range(10):
            _get(recorder, clock, route="/api/state", wall=0.4)
            _get(recorder, clock, route="/api/health", wall=0.002)
            # Fewer than half of these are slow, so p50 stays under the threshold.
            _get(recorder, clock, route="/api/mixed", wall=0.3 if _ % 3 == 0 else 0.01)
            clock.advance(1)
        assert len(_messages(caplog, "cayu.server.slow_request")) == 1

        clock.value = 60
        for _ in range(10):
            _get(recorder, clock, route="/api/state", wall=0.4)
            clock.advance(1)

    messages = _messages(caplog, "cayu.server.slow_request")
    assert len(messages) == 2
    assert messages[0] == (
        "cayu.server.slow_request: GET /api/state took 400 ms at p50 and 400 ms at p95 "
        "over 3 requests in the current 1 min window (threshold 250 ms). Check what the "
        "handler reads on each call; run `cayu diagnostics requests` for per-route cost "
        "and see cayu guide diagnostics#request-cost."
    )


def test_growing_cost_flags_a_get_whose_latency_keeps_rising(
    caplog: pytest.LogCaptureFixture,
) -> None:
    recorder, clock, _ = _recorder(
        RequestTimingConfig(growing_cost_bucket_requests=5, growing_cost_buckets=4)
    )
    with caplog.at_level(logging.WARNING, logger="cayu.server"):
        for wall in (0.02, 0.03, 0.045, 0.07, 0.07, 0.07):
            for _ in range(5):
                _get(recorder, clock, route="/api/state", wall=wall)
                _get(recorder, clock, route="/api/flat", wall=0.05 if _ % 2 else 0.04)
                clock.advance(2)

    messages = _messages(caplog, "cayu.server.growing_cost")
    assert messages == [
        "cayu.server.growing_cost: GET /api/state keeps getting slower: median 20 ms -> "
        "30 ms -> 45 ms -> 70 ms over the last 20 requests. Cost that grows with every call "
        "usually means the handler reads all stored history (likely O(history)); read only "
        "recent or changed records and see cayu guide diagnostics#request-cost."
    ]


def test_hot_poll_warns_for_one_client_repeating_an_identical_get(
    caplog: pytest.LogCaptureFixture,
) -> None:
    recorder, clock, _ = _recorder()
    with caplog.at_level(logging.WARNING, logger="cayu.server"):
        # A page polling every 2 seconds for 3 minutes.
        for _ in range(90):
            _get(recorder, clock, route="/api/state")
            clock.advance(2)
            if clock.value <= 120:
                assert _messages(caplog, "cayu.server.hot_poll") == []

    assert _messages(caplog, "cayu.server.hot_poll") == [
        "cayu.server.hot_poll: GET /api/state was called 30 times a minute by one client "
        "for 2 min; 98% of responses were identical. Poll less often, answer with 304 Not "
        "Modified, or push changes over a stream; see cayu guide app-ui"
    ]
    (record,) = [item for item in caplog.records if item.name == "cayu.server.hot_poll"]
    assert record.cayu_route == "/api/state"
    assert record.cayu_identical_share == pytest.approx(60 / 61, abs=1e-4)


def test_hot_poll_is_quiet_for_normal_interactive_use(caplog: pytest.LogCaptureFixture) -> None:
    recorder, clock, _ = _recorder()
    with caplog.at_level(logging.WARNING, logger="cayu.server"):
        # A person revisiting the same view every 10 seconds for 10 minutes.
        for _ in range(60):
            _get(recorder, clock, route="/api/state", client="person")
            clock.advance(10)
        # A burst while navigating: 40 quick loads, then the user reads for a while.
        for _ in range(40):
            _get(recorder, clock, route="/api/jobs/{job_id}", client="person")
            clock.advance(1)
        clock.advance(300)
        # Several people each loading the same page every 10 seconds: busy route,
        # but no single client is polling it.
        for index in range(300):
            _get(recorder, clock, route="/api/summary", client=f"person-{index % 5}")
            clock.advance(2)
        # A live view that polls fast but gets new data every time.
        for index in range(150):
            _get(recorder, clock, route="/api/live", client="dashboard", body=str(index).encode())
            clock.advance(2)

    assert _messages(caplog, "cayu.server.hot_poll") == []


def test_event_streams_pass_through_the_middleware_without_buffering() -> None:
    async def scenario() -> None:
        first_delivered = asyncio.Event()
        app = FastAPI()

        async def events():
            yield "data: first\n\n"
            # The second event is produced only after the first reached the
            # client, so a buffering middleware would hang here.
            await asyncio.wait_for(first_delivered.wait(), timeout=5)
            yield "data: second\n\n"

        @app.get("/api/stream")
        async def stream() -> StreamingResponse:
            return StreamingResponse(events(), media_type="text/event-stream")

        recorder = RequestTimingRecorder(RequestTimingConfig())
        app.add_middleware(RequestTimingMiddleware, recorder=recorder)
        sent: list[dict] = []
        never = asyncio.Event()

        async def receive() -> dict:
            await never.wait()
            return {"type": "http.disconnect"}

        async def send(message: dict) -> None:
            sent.append(message)
            if b"first" in message.get("body", b""):
                first_delivered.set()

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/api/stream",
            "raw_path": b"/api/stream",
            "query_string": b"",
            "root_path": "",
            "headers": [(b"host", b"testserver")],
            "client": ("127.0.0.1", 50000),
            "server": ("testserver", 80),
        }
        await asyncio.wait_for(app(scope, receive, send), timeout=10)

        bodies = [message.get("body", b"") for message in sent if message["type"].endswith("body")]
        assert b"".join(bodies) == b"data: first\n\ndata: second\n\n"
        (record,) = recorder.records()
        assert record.streaming is True
        assert record.route == "/api/stream"
        assert record.response_bytes == len(b"data: first\n\ndata: second\n\n")
        # A settled stream no longer counts as in flight for CPU apportioning.
        assert recorder.begin().cpu_share_start >= 0

    asyncio.run(scenario())


def _asgi_get_cpu_seconds(app, *, requests: int) -> float:
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/api/items/42",
        "raw_path": b"/api/items/42",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"testserver"), (b"user-agent", b"benchmark")],
        "client": ("127.0.0.1", 50000),
        "server": ("testserver", 80),
    }

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(_message: dict) -> None:
        return None

    async def run() -> float:
        started = time.process_time()
        for _ in range(requests):
            await app(dict(scope), receive, send)
        return time.process_time() - started

    return asyncio.run(run())


def measure_request_timing_overhead(*, requests: int = 1000, repeats: int = 3) -> float:
    """Return the middleware's added CPU seconds per request (best of ``repeats``)."""

    def build(timed: bool) -> FastAPI:
        app = FastAPI()

        @app.get("/api/items/{item_id}")
        async def item(item_id: str) -> dict[str, str]:
            return {"id": item_id, "state": "ready"}

        if timed:
            app.add_middleware(
                RequestTimingMiddleware,
                recorder=RequestTimingRecorder(RequestTimingConfig()),
            )
        return app

    plain, timed = build(False), build(True)
    _asgi_get_cpu_seconds(plain, requests=50)
    _asgi_get_cpu_seconds(timed, requests=50)
    baseline = min(_asgi_get_cpu_seconds(plain, requests=requests) for _ in range(repeats))
    measured = min(_asgi_get_cpu_seconds(timed, requests=requests) for _ in range(repeats))
    return max(0.0, measured - baseline) / requests


def test_middleware_overhead_stays_under_one_percent_cpu_at_fifty_requests_per_second() -> None:
    overhead = measure_request_timing_overhead()
    print(f"request timing overhead: {overhead * 1e6:.1f} us/request, {overhead * 50:.4%} CPU")

    # 1% of one CPU at 50 requests per second is 200 microseconds per request.
    assert overhead * 50 < 0.01
