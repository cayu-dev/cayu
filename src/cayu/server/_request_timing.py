"""Per-request cost recording, development warnings, and pure ASGI timing middleware.

The middleware is plain ASGI rather than Starlette's ``BaseHTTPMiddleware`` so
it forwards every response message as it arrives: streamed and SSE responses
are neither buffered nor delayed.

CPU time is process CPU (``time.process_time()``, all threads) apportioned
across the requests in flight. At every request start and finish the recorder
reads process CPU and divides the delta since the previous reading equally
among the requests that were running during it. Each request is charged its
accumulated share. This counts work done by synchronous handlers in the thread
pool, which ``time.thread_time()`` on the event-loop thread would miss, and it
does not double-count concurrent async requests the way a per-request
``process_time()`` delta would. CPU used while no request is in flight is not
charged to any request, and background work that overlaps a request (an agent
run, recovery sweeps) is charged to it, so measure idle UI cost while no agent
work is running. An event stream stops accruing CPU once its response starts;
its later chunks are not charged.
"""

from __future__ import annotations

import hashlib
import logging
import math
import secrets
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable, MutableMapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import pairwise
from typing import Any

from starlette.routing import Mount

from cayu.runtime.request_costs import (
    MAX_REQUEST_COST_ROUTES,
    UNMATCHED_ROUTE_TEMPLATE,
    RequestCostRoute,
    RequestCostSummary,
)
from cayu.server.config import RequestTimingConfig

_SLOW_REQUEST_EVENT = "cayu.server.slow_request"
_GROWING_COST_EVENT = "cayu.server.growing_cost"
_HOT_POLL_EVENT = "cayu.server.hot_poll"
_LOGGERS = {
    name: logging.getLogger(name)
    for name in (_SLOW_REQUEST_EVENT, _GROWING_COST_EVENT, _HOT_POLL_EVENT)
}

_BODY_DIGEST_LIMIT_BYTES = 1024 * 1024
_HOT_POLL_RATE_WINDOW_SECONDS = 60.0
_GROWING_COST_DIP_TOLERANCE = 0.9
_MAX_ROUTE_STATES = 1024
_MAX_CLIENT_ROUTE_STATES = 4096
_MAX_SLOW_SAMPLES = 256
_MAX_MOUNT_TEMPLATE_CACHE = 64
_MAX_TEMPLATE_CHARS = 512
_EVENT_STREAM_MEDIA_TYPE = b"text/event-stream"

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class RequestTimingRecord:
    """One completed request. Holds no raw path, query, header, or address."""

    ended_at: float
    method: str
    route: str
    status: int
    wall_seconds: float
    cpu_seconds: float
    response_bytes: int
    client_key: str
    streaming: bool


@dataclass(slots=True)
class PendingRequest:
    started_at: float
    cpu_share_start: float
    cpu_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class _Warning:
    event: str
    message: str
    extra: dict[str, object]


class RequestTimingRecorder:
    """Bounded in-memory request record with apportioned CPU accounting.

    ``clock`` (monotonic seconds), ``cpu_clock`` (process CPU seconds), and
    ``now`` (wall-clock timestamps for summaries) are injectable so behaviour
    can be tested without real time passing.
    """

    def __init__(
        self,
        config: RequestTimingConfig,
        *,
        excluded_routes: frozenset[str] = frozenset(),
        clock: Callable[[], float] = time.monotonic,
        cpu_clock: Callable[[], float] = time.process_time,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if type(config) is not RequestTimingConfig:
            raise TypeError("config must be a RequestTimingConfig.")
        self.config = config
        self._excluded_routes = frozenset(excluded_routes)
        self._clock = clock
        self._cpu_clock = cpu_clock
        self._now = now
        self._lock = threading.Lock()
        self._records: deque[RequestTimingRecord] = deque(maxlen=config.buffer_size)
        self._evicted_until: float | None = None
        self._started_at = clock()
        self._in_flight = 0
        self._cpu_share = 0.0
        self._last_cpu = cpu_clock()
        # Client keys are keyed hashes with a per-process secret, so they
        # cannot be reversed or tested against guessed credentials and do not
        # link clients across restarts.
        self._client_salt = secrets.token_bytes(16)
        self._monitor = _RequestCostMonitor(config) if config.warnings else None

    def client_key(self, identity: bytes) -> str:
        return hashlib.blake2b(identity, key=self._client_salt, digest_size=8).hexdigest()

    def begin(self) -> PendingRequest:
        with self._lock:
            share = self._advance_cpu()
            self._in_flight += 1
            return PendingRequest(started_at=self._clock(), cpu_share_start=share)

    def settle_cpu(self, pending: PendingRequest) -> None:
        """Stop charging CPU to a request that has started a long-lived stream."""

        with self._lock:
            self._settle_cpu(pending)

    def finish(
        self,
        pending: PendingRequest,
        *,
        method: str,
        route: str,
        status: int,
        response_bytes: int,
        client_key: str,
        streaming: bool,
        body_digest: bytes | None = None,
    ) -> RequestTimingRecord | None:
        warnings: list[_Warning] = []
        with self._lock:
            ended_at = self._clock()
            self._settle_cpu(pending)
            if route in self._excluded_routes:
                return None
            record = RequestTimingRecord(
                ended_at=ended_at,
                method=method,
                route=route,
                status=status,
                wall_seconds=max(0.0, ended_at - pending.started_at),
                cpu_seconds=pending.cpu_seconds or 0.0,
                response_bytes=response_bytes,
                client_key=client_key,
                streaming=streaming,
            )
            if len(self._records) == self._records.maxlen:
                self._evicted_until = self._records[0].ended_at
            self._records.append(record)
            if self._monitor is not None:
                self._monitor.observe(record, body_digest, warnings)
        for warning in warnings:
            _LOGGERS[warning.event].warning(warning.message, extra=warning.extra)
        return record

    def records(self) -> tuple[RequestTimingRecord, ...]:
        with self._lock:
            return tuple(self._records)

    def summary(self, *, since_seconds: float, vcpu: float = 1.0) -> RequestCostSummary:
        """Summarize retained records that ended within the last ``since_seconds``."""

        with self._lock:
            now = self._clock()
            observed_at = self._now()
            cutoff = now - since_seconds
            records: list[RequestTimingRecord] = []
            for record in reversed(self._records):
                if record.ended_at <= cutoff:
                    break
                records.append(record)
            evicted_until = self._evicted_until
            started_at = self._started_at
        window_start = max(cutoff, started_at)
        truncated = evicted_until is not None and evicted_until > cutoff
        if truncated and evicted_until is not None:
            window_start = max(window_start, evicted_until)
        window_seconds = max(0.0, now - window_start)
        # Rates over a sub-second span would be meaningless extrapolations.
        rate_seconds = max(window_seconds, 1.0)

        groups: dict[tuple[str, str], _RouteAggregate] = {}
        total_cpu = 0.0
        for record in records:
            key = (record.method, record.route)
            aggregate = groups.get(key)
            if aggregate is None:
                aggregate = groups[key] = _RouteAggregate()
            aggregate.add(record)
            total_cpu += record.cpu_seconds
        routes = sorted(
            (
                aggregate.project(
                    method=method,
                    route=route,
                    rate_seconds=rate_seconds,
                    vcpu=vcpu,
                )
                for (method, route), aggregate in groups.items()
            ),
            key=lambda item: (-item.cpu_seconds, -item.requests, item.method, item.route),
        )
        return RequestCostSummary(
            observed_at=observed_at,
            enabled=True,
            requested_window_seconds=since_seconds,
            window_seconds=round(window_seconds, 3),
            truncated=truncated,
            buffer_capacity=self.config.buffer_size,
            vcpu=vcpu,
            requests=len(records),
            requests_per_minute=_round(len(records) * 60.0 / rate_seconds),
            cpu_seconds=_round(total_cpu),
            cpu_seconds_per_minute=_round(total_cpu * 60.0 / rate_seconds),
            vcpu_share=_round(total_cpu / rate_seconds / vcpu),
            route_count=len(routes),
            routes=tuple(routes[:MAX_REQUEST_COST_ROUTES]),
        )

    def _advance_cpu(self) -> float:
        cpu = self._cpu_clock()
        delta = cpu - self._last_cpu
        self._last_cpu = cpu
        if delta > 0 and self._in_flight > 0:
            self._cpu_share += delta / self._in_flight
        return self._cpu_share

    def _settle_cpu(self, pending: PendingRequest) -> None:
        if pending.cpu_seconds is not None:
            return
        share = self._advance_cpu()
        pending.cpu_seconds = max(0.0, share - pending.cpu_share_start)
        self._in_flight -= 1


@dataclass(slots=True)
class _RouteAggregate:
    requests: int = 0
    streaming: int = 0
    statuses: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    walls: list[float] = field(default_factory=list)
    cpu_seconds: float = 0.0
    response_bytes: int = 0
    clients: set[str] = field(default_factory=set)

    def add(self, record: RequestTimingRecord) -> None:
        self.requests += 1
        status_class = record.status // 100
        if 2 <= status_class <= 5:
            self.statuses[status_class - 2] += 1
        if record.streaming:
            self.streaming += 1
        else:
            self.walls.append(record.wall_seconds)
        self.cpu_seconds += record.cpu_seconds
        self.response_bytes += record.response_bytes
        self.clients.add(record.client_key)

    def project(
        self,
        *,
        method: str,
        route: str,
        rate_seconds: float,
        vcpu: float,
    ) -> RequestCostRoute:
        walls = sorted(self.walls)
        return RequestCostRoute(
            method=method,
            route=route,
            requests=self.requests,
            requests_per_minute=_round(self.requests * 60.0 / rate_seconds),
            streaming_requests=self.streaming,
            status_2xx=self.statuses[0],
            status_3xx=self.statuses[1],
            status_4xx=self.statuses[2],
            status_5xx=self.statuses[3],
            wall_ms_p50=_milliseconds(_percentile(walls, 0.5)),
            wall_ms_p95=_milliseconds(_percentile(walls, 0.95)),
            wall_ms_max=_milliseconds(walls[-1] if walls else None),
            cpu_seconds=_round(self.cpu_seconds),
            cpu_seconds_per_minute=_round(self.cpu_seconds * 60.0 / rate_seconds),
            vcpu_share=_round(self.cpu_seconds / rate_seconds / vcpu),
            response_bytes=self.response_bytes,
            clients=len(self.clients),
        )


@dataclass(slots=True)
class _RouteLatencyState:
    window_index: int = -1
    count: int = 0
    over: int = 0
    samples: list[float] = field(default_factory=list)
    warned: bool = False
    bucket: list[float] = field(default_factory=list)
    bucket_medians: deque[float] = field(default_factory=deque)
    growth_warned_at: float | None = None


@dataclass(slots=True)
class _HotPollState:
    recent: deque[float]
    identical: deque[bool]
    last_digest: bytes | None = None
    run_started: float | None = None
    run_requests: int = 0
    run_identical: int = 0
    warned: bool = False


class _RequestCostMonitor:
    """Slow-GET, growing-cost, and hot-poll detection over completed GETs."""

    def __init__(self, config: RequestTimingConfig) -> None:
        self._config = config
        self._routes: OrderedDict[str, _RouteLatencyState] = OrderedDict()
        self._clients: OrderedDict[tuple[str, str], _HotPollState] = OrderedDict()

    def observe(
        self,
        record: RequestTimingRecord,
        body_digest: bytes | None,
        warnings: list[_Warning],
    ) -> None:
        if record.method != "GET" or record.streaming:
            return
        self._observe_latency(record, warnings)
        if 200 <= record.status < 300:
            self._observe_hot_poll(record, body_digest, warnings)

    def _observe_latency(self, record: RequestTimingRecord, warnings: list[_Warning]) -> None:
        config = self._config
        state = _lru_get(self._routes, record.route, _RouteLatencyState, _MAX_ROUTE_STATES)
        window_index = math.floor(record.ended_at / config.slow_get_window_seconds)
        if window_index != state.window_index:
            state.window_index = window_index
            state.count = 0
            state.over = 0
            state.samples.clear()
            state.warned = False
        state.count += 1
        if record.wall_seconds * 1000.0 > config.slow_get_threshold_ms:
            state.over += 1
        if len(state.samples) < _MAX_SLOW_SAMPLES:
            state.samples.append(record.wall_seconds)
        # p50 exceeds the threshold exactly when more than half the samples do,
        # so the check is constant-time; percentiles are computed only to warn.
        if (
            not state.warned
            and state.count >= config.slow_get_min_requests
            and state.over * 2 > state.count
        ):
            state.warned = True
            samples = sorted(state.samples)
            p50 = _percentile(samples, 0.5) or 0.0
            p95 = _percentile(samples, 0.95) or 0.0
            warnings.append(
                _Warning(
                    event=_SLOW_REQUEST_EVENT,
                    message=(
                        f"{_SLOW_REQUEST_EVENT}: GET {record.route} took "
                        f"{p50 * 1000:.0f} ms at p50 and {p95 * 1000:.0f} ms at p95 over "
                        f"{state.count} requests in the current "
                        f"{_duration_text(config.slow_get_window_seconds)} window "
                        f"(threshold {config.slow_get_threshold_ms:.0f} ms). Check what the "
                        "handler reads on each call; run `cayu diagnostics requests` for "
                        "per-route cost and see cayu guide diagnostics#request-cost."
                    ),
                    extra={
                        "cayu_event": _SLOW_REQUEST_EVENT,
                        "cayu_method": "GET",
                        "cayu_route": record.route,
                        "cayu_p50_ms": round(p50 * 1000, 3),
                        "cayu_p95_ms": round(p95 * 1000, 3),
                        "cayu_requests": state.count,
                    },
                )
            )
        self._observe_growth(record, state, warnings)

    def _observe_growth(
        self,
        record: RequestTimingRecord,
        state: _RouteLatencyState,
        warnings: list[_Warning],
    ) -> None:
        config = self._config
        state.bucket.append(record.wall_seconds)
        if len(state.bucket) < config.growing_cost_bucket_requests:
            return
        state.bucket.sort()
        state.bucket_medians.append(_percentile(state.bucket, 0.5) or 0.0)
        state.bucket.clear()
        while len(state.bucket_medians) > config.growing_cost_buckets:
            state.bucket_medians.popleft()
        if len(state.bucket_medians) < config.growing_cost_buckets:
            return
        medians = tuple(state.bucket_medians)
        first, last = medians[0], medians[-1]
        growing = (
            all(
                later >= earlier * _GROWING_COST_DIP_TOLERANCE
                for earlier, later in pairwise(medians)
            )
            and last >= first * config.growing_cost_ratio
            and last * 1000.0 >= config.growing_cost_min_ms
        )
        if not growing:
            return
        # Warn once per growth episode; warn again only if it doubles again.
        if state.growth_warned_at is not None and last < state.growth_warned_at * 2:
            return
        state.growth_warned_at = last
        path = " -> ".join(f"{value * 1000:.0f} ms" for value in medians)
        warnings.append(
            _Warning(
                event=_GROWING_COST_EVENT,
                message=(
                    f"{_GROWING_COST_EVENT}: GET {record.route} keeps getting slower: "
                    f"median {path} over the last "
                    f"{len(medians) * config.growing_cost_bucket_requests} requests. "
                    "Cost that grows with every call usually means the handler reads "
                    "all stored history (likely O(history)); read only recent or changed "
                    "records and see cayu guide diagnostics#request-cost."
                ),
                extra={
                    "cayu_event": _GROWING_COST_EVENT,
                    "cayu_method": "GET",
                    "cayu_route": record.route,
                    "cayu_median_ms": [round(value * 1000, 3) for value in medians],
                },
            )
        )

    def _observe_hot_poll(
        self,
        record: RequestTimingRecord,
        body_digest: bytes | None,
        warnings: list[_Warning],
    ) -> None:
        config = self._config
        threshold = config.hot_poll_requests_per_minute
        state = _lru_get(
            self._clients,
            (record.client_key, record.route),
            lambda: _HotPollState(
                recent=deque(maxlen=threshold + 1),
                identical=deque(maxlen=threshold + 1),
            ),
            _MAX_CLIENT_ROUTE_STATES,
        )
        identical = body_digest is not None and body_digest == state.last_digest
        state.last_digest = body_digest
        state.recent.append(record.ended_at)
        state.identical.append(identical)
        # More than `threshold` calls inside one sliding minute.
        over_rate = (
            len(state.recent) > threshold
            and record.ended_at - state.recent[0] <= _HOT_POLL_RATE_WINDOW_SECONDS
        )
        if not over_rate:
            state.run_started = None
            state.warned = False
            return
        if state.run_started is None:
            state.run_started = state.recent[0]
            state.run_requests = len(state.recent)
            state.run_identical = sum(state.identical)
        else:
            state.run_requests += 1
            state.run_identical += identical
        duration = record.ended_at - state.run_started
        if state.warned or duration <= config.hot_poll_duration_seconds:
            return
        identical_share = state.run_identical / state.run_requests
        if identical_share < config.hot_poll_identical_share:
            return
        state.warned = True
        per_minute = state.run_requests * 60.0 / duration
        warnings.append(
            _Warning(
                event=_HOT_POLL_EVENT,
                message=(
                    f"{_HOT_POLL_EVENT}: GET {record.route} was called "
                    f"{per_minute:.0f} times a minute by one client for "
                    f"{_duration_text(duration)}; {identical_share:.0%} of responses were "
                    "identical. Poll less often, answer with 304 Not Modified, or push "
                    "changes over a stream; see cayu guide app-ui"
                ),
                extra={
                    "cayu_event": _HOT_POLL_EVENT,
                    "cayu_method": "GET",
                    "cayu_route": record.route,
                    "cayu_requests_per_minute": round(per_minute, 3),
                    "cayu_duration_seconds": round(duration, 3),
                    "cayu_identical_share": round(identical_share, 4),
                },
            )
        )


class RequestTimingMiddleware:
    """Pure ASGI middleware that records one ``RequestTimingRecord`` per HTTP request.

    ``observe_prefix`` limits recording to one mounted path (``None`` records
    every request). The raw path is compared but never stored.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        recorder: RequestTimingRecorder,
        observe_prefix: str | None = None,
    ) -> None:
        self.app = app
        self.recorder = recorder
        self.observe_prefix = None if observe_prefix in {None, "/"} else observe_prefix
        self._mount_templates: dict[str, str | None] = {}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self._observed(scope):
            await self.app(scope, receive, send)
            return
        recorder = self.recorder
        entry_app = scope.get("app")
        entry_root = scope.get("root_path", "")
        method = scope.get("method", "GET")
        pending = recorder.begin()
        status = 500
        response_bytes = 0
        streaming = False
        hasher: Any = None
        hashed_bytes = 0

        async def timed_send(message: Message) -> None:
            nonlocal status, response_bytes, streaming, hasher, hashed_bytes
            message_type = message["type"]
            if message_type == "http.response.start":
                status = message["status"]
                content_type = _header(message.get("headers", ()), b"content-type")
                if content_type is not None and content_type.lower().startswith(
                    _EVENT_STREAM_MEDIA_TYPE
                ):
                    streaming = True
                    recorder.settle_cpu(pending)
                elif method == "GET" and 200 <= status < 300:
                    hasher = hashlib.blake2b(digest_size=16)
            elif message_type == "http.response.body":
                body = message.get("body", b"")
                response_bytes += len(body)
                if hasher is not None and hashed_bytes < _BODY_DIGEST_LIMIT_BYTES:
                    chunk = body[: _BODY_DIGEST_LIMIT_BYTES - hashed_bytes]
                    hasher.update(chunk)
                    hashed_bytes += len(chunk)
            await send(message)

        try:
            await self.app(scope, receive, timed_send)
        finally:
            try:
                body_digest = None
                if hasher is not None:
                    hasher.update(response_bytes.to_bytes(8, "big"))
                    body_digest = hasher.digest()
                recorder.finish(
                    pending,
                    method=method,
                    route=self._route_template(scope, entry_app, entry_root),
                    status=status,
                    response_bytes=response_bytes,
                    client_key=recorder.client_key(_client_identity(scope)),
                    streaming=streaming,
                    body_digest=body_digest,
                )
            except Exception:
                # Diagnostics must never change the response or its error.
                logging.getLogger(__name__).debug("Request timing failed.", exc_info=True)

    def _observed(self, scope: Scope) -> bool:
        prefix = self.observe_prefix
        if prefix is None:
            return True
        path = _route_path(scope)
        return path == prefix or path.startswith(prefix + "/")

    def _route_template(self, scope: Scope, entry_app: object, entry_root: str) -> str:
        route = scope.get("route")
        inner = getattr(route, "path_format", None) if route is not None else None
        if not isinstance(inner, str):
            inner = None
        root_path = scope.get("root_path", "")
        mounted = root_path[len(entry_root) :] if root_path.startswith(entry_root) else ""
        if not mounted:
            template = inner
        else:
            # Inside a mounted sub-application. The matched prefix can contain
            # raw parameter values, so it is replaced by the mount's template.
            prefix = self._mount_template(entry_app, mounted)
            template = None if prefix is None else prefix + (inner or "/{path}")
        if template is None or len(template) > _MAX_TEMPLATE_CHARS:
            return UNMATCHED_ROUTE_TEMPLATE
        return template

    def _mount_template(self, entry_app: object, mounted: str) -> str | None:
        cached = self._mount_templates.get(mounted, "")
        if cached != "":
            return cached
        template = None
        for route in getattr(entry_app, "routes", None) or ():
            if isinstance(route, Mount) and route.path_regex.match(mounted + "/"):
                template = route.path
                break
        if len(self._mount_templates) < _MAX_MOUNT_TEMPLATE_CACHE:
            self._mount_templates[mounted] = template
        return template


def _header(headers: Any, name: bytes) -> bytes | None:
    for key, value in headers:
        if key.lower() == name:
            return value
    return None


def _client_identity(scope: Scope) -> bytes:
    authorization = cookie = user_agent = None
    for name, value in scope.get("headers", ()):
        if name == b"authorization":
            authorization = value
        elif name == b"cookie":
            cookie = value
        elif name == b"user-agent":
            user_agent = value
    if authorization:
        return b"authorization\x00" + authorization
    if cookie:
        return b"cookie\x00" + cookie
    client = scope.get("client")
    host = client[0] if client else ""
    return b"address\x00" + str(host).encode("utf-8", "replace") + b"\x00" + (user_agent or b"")


def _route_path(scope: Scope) -> str:
    path = scope.get("path", "")
    root_path = scope.get("root_path", "")
    if not root_path or not path.startswith(root_path):
        return path
    if path == root_path:
        return ""
    if path[len(root_path)] == "/":
        return path[len(root_path) :]
    return path


def _lru_get(mapping: OrderedDict, key: Any, factory: Callable[[], Any], limit: int) -> Any:
    value = mapping.get(key)
    if value is None:
        value = mapping[key] = factory()
        if len(mapping) > limit:
            mapping.popitem(last=False)
    else:
        mapping.move_to_end(key)
    return value


def _percentile(sorted_values: list[float], quantile: float) -> float | None:
    if not sorted_values:
        return None
    rank = max(1, math.ceil(quantile * len(sorted_values)))
    return sorted_values[rank - 1]


def _milliseconds(value: float | None) -> float | None:
    return None if value is None else round(value * 1000.0, 3)


def _round(value: float) -> float:
    return round(value, 6)


def _duration_text(seconds: float) -> str:
    if seconds >= 60:
        return f"{seconds / 60:.0f} min"
    return f"{seconds:.0f} s"
