"""Shared scheduling and lease-heartbeat mechanics for durable workers.

The dispatcher and generic task worker retain their task-specific claim,
authority, handling, and terminalization rules.  This module owns the repeated
step -> idle wait -> stop/max-work cycle and the common lease-heartbeat clock so
those mechanics cannot drift between adapters.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from random import random
from threading import Lock
from time import monotonic
from typing import Generic, TypeVar

_HeartbeatUpdateT = TypeVar("_HeartbeatUpdateT")
_HeartbeatOutcomeT = TypeVar("_HeartbeatOutcomeT")
_MaintenanceOutcomeT = TypeVar("_MaintenanceOutcomeT")
_ClaimT = TypeVar("_ClaimT")


class DurableWorkerWaitResult(StrEnum):
    """Reason the shared worker's bounded idle wait ended."""

    TIMEOUT = "timeout"
    HINT = "hint"
    STOP = "stop"


WorkerWait = Callable[
    [float, asyncio.Event | None],
    Awaitable[bool | DurableWorkerWaitResult],
]


@dataclass(frozen=True, init=False)
class DurableWorkerDemandPolicy:
    """Bounded idle economics shared by one compatible worker cohort."""

    dispatch_latency_s: float
    minimum_idle_delay_s: float
    maximum_idle_delay_s: float
    backoff_multiplier: float = 2.0
    jitter_ratio: float = 0.1

    def __init__(
        self,
        *,
        dispatch_latency_s: float,
        minimum_idle_delay_s: float | None = None,
        maximum_idle_delay_s: float | None = None,
        backoff_multiplier: float = 2.0,
        jitter_ratio: float = 0.1,
    ) -> None:
        validate_worker_interval(dispatch_latency_s, "dispatch_latency_s")
        minimum = (
            min(0.05, dispatch_latency_s) if minimum_idle_delay_s is None else minimum_idle_delay_s
        )
        maximum = dispatch_latency_s if maximum_idle_delay_s is None else maximum_idle_delay_s
        validate_worker_interval(minimum, "minimum_idle_delay_s")
        validate_worker_interval(maximum, "maximum_idle_delay_s")
        if minimum > maximum:
            raise ValueError("minimum_idle_delay_s must be <= maximum_idle_delay_s.")
        if maximum > dispatch_latency_s:
            raise ValueError("maximum_idle_delay_s must be <= dispatch_latency_s.")
        if (
            isinstance(backoff_multiplier, bool)
            or not isinstance(backoff_multiplier, int | float)
            or not isfinite(backoff_multiplier)
            or backoff_multiplier <= 1
        ):
            raise ValueError("backoff_multiplier must be finite and > 1.")
        if (
            isinstance(jitter_ratio, bool)
            or not isinstance(jitter_ratio, int | float)
            or not isfinite(jitter_ratio)
            or not 0 <= jitter_ratio <= 1
        ):
            raise ValueError("jitter_ratio must be finite and between 0 and 1.")
        object.__setattr__(self, "minimum_idle_delay_s", float(minimum))
        object.__setattr__(self, "maximum_idle_delay_s", float(maximum))
        object.__setattr__(self, "dispatch_latency_s", float(dispatch_latency_s))
        object.__setattr__(self, "backoff_multiplier", float(backoff_multiplier))
        object.__setattr__(self, "jitter_ratio", float(jitter_ratio))


@dataclass(frozen=True)
class DurableWorkerClaim(Generic[_ClaimT]):
    """One cohort-gated authoritative claim attempt."""

    attempted: bool
    value: _ClaimT | None = None


class DurableWorkerPollerGroup:
    """Coordinate one fair active poller for a compatible worker cohort."""

    def __init__(
        self,
        *,
        on_empty: Callable[[DurableWorkerPollerGroup], None] | None = None,
    ) -> None:
        if on_empty is not None and not callable(on_empty):
            raise TypeError("on_empty must be callable.")
        self._lock = Lock()
        self._on_empty = on_empty
        self._policy: DurableWorkerDemandPolicy | None = None
        self._tokens: list[int] = []
        self._next_token = 1
        self._preferred_token: int | None = None
        self._active_token: int | None = None
        self._active_until: float | None = None
        self._next_poll_at = 0.0
        self._next_empty_delay_s = 0.0

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._tokens)

    @property
    def active_poller_count(self) -> int:
        with self._lock:
            return 0 if self._active_token is None else 1

    def subscribe(
        self,
        policy: DurableWorkerDemandPolicy,
        *,
        clock: Callable[[], float],
        random_source: Callable[[], float] = random,
    ) -> DurableWorkerPoller:
        if not isinstance(policy, DurableWorkerDemandPolicy):
            raise TypeError("Durable worker pollers require a demand policy.")
        if not callable(clock) or not callable(random_source):
            raise TypeError("Durable worker poller clocks and random sources must be callable.")
        with self._lock:
            if self._tokens and self._policy != policy:
                raise ValueError("A worker cohort must use one shared demand policy.")
            if not self._tokens:
                self._policy = policy
                self._preferred_token = None
                self._active_token = None
                self._active_until = None
                self._next_poll_at = 0.0
                self._next_empty_delay_s = policy.minimum_idle_delay_s
            token = self._next_token
            self._next_token += 1
            self._tokens.append(token)
            if self._preferred_token is None:
                self._preferred_token = token
        return DurableWorkerPoller(
            self,
            token,
            clock=clock,
            random_source=random_source,
        )

    def _begin(
        self,
        token: int,
        *,
        now: float,
        forced: bool,
        maximum_active_s: float | None,
    ) -> bool:
        _validate_worker_clock(now)
        with self._lock:
            if token not in self._tokens:
                raise RuntimeError("Durable worker poller is closed.")
            if (
                self._active_token is not None
                and self._active_until is not None
                and now >= self._active_until
            ):
                expired_token = self._active_token
                self._active_token = None
                self._active_until = None
                self._preferred_token = self._successor_token(expired_token)
            if self._active_token is not None:
                return False
            if not forced and (token != self._preferred_token or now < self._next_poll_at):
                return False
            self._active_token = token
            self._active_until = None if maximum_active_s is None else now + maximum_active_s
            return True

    def _finish(
        self,
        token: int,
        *,
        now: float,
        claimed: bool,
        random_source: Callable[[], float],
    ) -> None:
        _validate_worker_clock(now)
        with self._lock:
            if self._active_token != token:
                return
            self._active_token = None
            self._active_until = None
            self._preferred_token = self._successor_token(token)
            policy = self._required_policy()
            if claimed:
                self._next_empty_delay_s = policy.minimum_idle_delay_s
                self._next_poll_at = now
                return
            delay = _jittered_idle_delay(
                self._next_empty_delay_s,
                policy=policy,
                random_source=random_source,
            )
            self._next_poll_at = now + delay
            self._next_empty_delay_s = min(
                policy.maximum_idle_delay_s,
                self._next_empty_delay_s * policy.backoff_multiplier,
            )

    def _reset(
        self,
        token: int,
        *,
        now: float,
        delay_next: bool,
        prefer_token: bool,
        random_source: Callable[[], float],
    ) -> None:
        _validate_worker_clock(now)
        with self._lock:
            if token not in self._tokens:
                return
            policy = self._required_policy()
            self._next_empty_delay_s = policy.minimum_idle_delay_s
            if prefer_token and self._active_token is None:
                self._preferred_token = token
            self._next_poll_at = (
                now
                if not delay_next
                else now
                + _jittered_idle_delay(
                    policy.minimum_idle_delay_s,
                    policy=policy,
                    random_source=random_source,
                )
            )

    def _deadline(self, token: int, *, now: float, forced: bool) -> float:
        _validate_worker_clock(now)
        with self._lock:
            policy = self._required_policy()
            if forced and self._active_token is None:
                return now
            if self._active_token is not None or token != self._preferred_token:
                return max(self._next_poll_at, now + policy.minimum_idle_delay_s)
            return max(self._next_poll_at, now)

    def _unsubscribe(self, token: int, *, now: float) -> None:
        _validate_worker_clock(now)
        became_empty = False
        with self._lock:
            if token not in self._tokens:
                return
            index = self._tokens.index(token)
            self._tokens.remove(token)
            if self._active_token == token:
                self._active_token = None
                self._active_until = None
            if self._preferred_token == token:
                self._preferred_token = (
                    None if not self._tokens else self._tokens[index % len(self._tokens)]
                )
            if not self._tokens:
                self._policy = None
                self._active_until = None
                self._next_poll_at = 0.0
                self._next_empty_delay_s = 0.0
                became_empty = True
        if became_empty and self._on_empty is not None:
            self._on_empty(self)

    def _successor_token(self, token: int) -> int | None:
        if not self._tokens:
            return None
        try:
            index = self._tokens.index(token)
        except ValueError:
            return self._tokens[0]
        return self._tokens[(index + 1) % len(self._tokens)]

    def _required_policy(self) -> DurableWorkerDemandPolicy:
        if self._policy is None:
            raise RuntimeError("Durable worker poller group has no active policy.")
        return self._policy


class DurableWorkerPoller:
    """One worker's fair claim entrance into a shared poller group."""

    def __init__(
        self,
        group: DurableWorkerPollerGroup,
        token: int,
        *,
        clock: Callable[[], float],
        random_source: Callable[[], float],
    ) -> None:
        self._group = group
        self._token = token
        self._clock = clock
        self._random_source = random_source
        self._forced = False
        self._closed = False
        self._last_attempted = False
        self._last_claimed = False

    @property
    def last_attempted(self) -> bool:
        return self._last_attempted

    @property
    def last_claimed(self) -> bool:
        return self._last_claimed

    def begin_step(self) -> None:
        self._last_attempted = False
        self._last_claimed = False

    async def claim(
        self,
        action: Callable[[], Awaitable[_ClaimT | None]],
        *,
        maximum_active_s: float | None = None,
    ) -> DurableWorkerClaim[_ClaimT]:
        if not callable(action):
            raise TypeError("Durable worker claim action must be callable.")
        if maximum_active_s is not None:
            validate_worker_interval(maximum_active_s, "maximum_active_s")
        now = self._clock()
        if not self._group._begin(
            self._token,
            now=now,
            forced=self._forced,
            maximum_active_s=maximum_active_s,
        ):
            return DurableWorkerClaim(attempted=False)
        self._forced = False
        self._last_attempted = True
        try:
            value = await action()
        except BaseException:
            self._group._finish(
                self._token,
                now=self._clock(),
                claimed=False,
                random_source=self._random_source,
            )
            raise
        self._last_claimed = value is not None
        self._group._finish(
            self._token,
            now=self._clock(),
            claimed=self._last_claimed,
            random_source=self._random_source,
        )
        return DurableWorkerClaim(attempted=True, value=value)

    def note_hint(self) -> None:
        self._forced = True
        self._group._reset(
            self._token,
            now=self._clock(),
            delay_next=False,
            prefer_token=True,
            random_source=self._random_source,
        )

    def note_activity(self, *, delay_next: bool) -> None:
        self._group._reset(
            self._token,
            now=self._clock(),
            delay_next=delay_next,
            prefer_token=False,
            random_source=self._random_source,
        )

    def next_wake_at(self) -> float:
        now = self._clock()
        return self._group._deadline(self._token, now=now, forced=self._forced)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._group._unsubscribe(self._token, now=self._clock())


@dataclass(frozen=True)
class DurableWorkerStep:
    """One adapter-owned worker step returned to the shared scheduler."""

    handled: int = 0
    idle: bool = False
    continue_immediately: bool = False
    stop: bool = False
    next_wake_at: float | None = None
    activity: bool = False

    def __post_init__(self) -> None:
        if type(self.handled) is not int or self.handled < 0:
            raise ValueError("handled must be a non-negative integer.")
        if self.idle and self.continue_immediately:
            raise ValueError("An idle worker step cannot also continue immediately.")
        if type(self.activity) is not bool:
            raise TypeError("activity must be a bool.")
        if self.next_wake_at is not None:
            if not isfinite(self.next_wake_at) or self.next_wake_at < 0:
                raise ValueError("next_wake_at must be finite and non-negative.")
            if not self.idle:
                raise ValueError("next_wake_at requires an idle worker step.")


@dataclass
class DurableWorkerCadence:
    """Runtime-owned fixed cadence for one adapter maintenance operation.

    ``every_s=None`` represents an operation that runs before every claim cycle.
    Timed operations run immediately on the first cycle and then at most once
    per interval.
    """

    every_s: float | None
    _next_run_at: float = 0.0

    def __post_init__(self) -> None:
        if self.every_s is not None:
            validate_worker_interval(self.every_s, "every_s")

    @property
    def next_run_at(self) -> float | None:
        """Return the next timed deadline, or ``None`` for every-cycle work."""

        return None if self.every_s is None else self._next_run_at

    def expedite(self) -> None:
        """Make a timed operation eligible on the next worker cycle."""

        if self.every_s is not None:
            self._next_run_at = 0.0

    async def run_if_due(
        self,
        action: Callable[[], Awaitable[_MaintenanceOutcomeT]],
        *,
        now: float,
        clock: Callable[[], float],
    ) -> tuple[bool, _MaintenanceOutcomeT | None]:
        """Run ``action`` when due and advance its cadence after the attempt."""

        if not isfinite(now) or now < 0:
            raise ValueError("now must be finite and non-negative.")
        if self.every_s is not None and now < self._next_run_at:
            return False, None
        try:
            return True, await action()
        finally:
            if self.every_s is not None:
                next_run_at = clock() + self.every_s
                if not isfinite(next_run_at) or next_run_at < 0:
                    raise ValueError("The worker cadence clock returned an invalid deadline.")
                self._next_run_at = next_run_at


def worker_stop_requested(stop: asyncio.Event | None) -> bool:
    """Return whether an optional worker stop signal is set."""

    return stop is not None and stop.is_set()


def validate_worker_interval(seconds: float, field_name: str) -> None:
    """Require one finite positive worker-loop interval."""

    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, int | float)
        or not isfinite(seconds)
        or seconds <= 0
    ):
        raise ValueError(f"{field_name} must be finite and positive.")


def _validate_worker_clock(now: float) -> None:
    if isinstance(now, bool) or not isinstance(now, int | float) or not isfinite(now) or now < 0:
        raise ValueError("The durable worker clock must return a finite non-negative value.")


def _jittered_idle_delay(
    delay_s: float,
    *,
    policy: DurableWorkerDemandPolicy,
    random_source: Callable[[], float],
) -> float:
    sample = random_source()
    if (
        isinstance(sample, bool)
        or not isinstance(sample, int | float)
        or not isfinite(sample)
        or not 0 <= sample <= 1
    ):
        raise ValueError("The durable worker random source must return a value from 0 to 1.")
    jitter = delay_s * policy.jitter_ratio * ((2 * sample) - 1)
    return min(
        policy.maximum_idle_delay_s,
        max(policy.minimum_idle_delay_s, delay_s + jitter),
    )


def _normalize_worker_wait_result(
    result: bool | DurableWorkerWaitResult,
) -> DurableWorkerWaitResult:
    if type(result) is bool:
        return DurableWorkerWaitResult.STOP if result else DurableWorkerWaitResult.TIMEOUT
    if not isinstance(result, DurableWorkerWaitResult):
        raise TypeError("A durable worker wait returned an invalid outcome.")
    return result


async def wait_or_stop(seconds: float, stop: asyncio.Event | None) -> bool:
    """Wait for a bounded delay or an optional stop signal."""

    if not isfinite(seconds) or seconds < 0:
        raise ValueError("Worker wait duration must be finite and non-negative.")
    if worker_stop_requested(stop):
        return True
    if seconds == 0:
        await asyncio.sleep(0)
        return worker_stop_requested(stop)
    if stop is None:
        await asyncio.sleep(seconds)
        return False
    try:
        await asyncio.wait_for(stop.wait(), timeout=seconds)
    except TimeoutError:
        return stop.is_set()
    return True


async def run_durable_worker_loop(
    step: Callable[[float, int], Awaitable[DurableWorkerStep]],
    *,
    poll_interval_s: float,
    stop: asyncio.Event | None,
    max_handled: int | None = None,
    wait: WorkerWait = wait_or_stop,
    demand_policy: DurableWorkerDemandPolicy | None = None,
    poller: DurableWorkerPoller | None = None,
    clock: Callable[[], float] | None = None,
) -> int:
    """Run adapter steps with adaptive idle waits and one stop contract."""

    validate_worker_interval(poll_interval_s, "poll_interval_s")
    if demand_policy is None:
        demand_policy = DurableWorkerDemandPolicy(dispatch_latency_s=poll_interval_s)
    elif not isinstance(demand_policy, DurableWorkerDemandPolicy):
        raise TypeError("demand_policy must be a DurableWorkerDemandPolicy.")
    elif demand_policy.dispatch_latency_s != poll_interval_s:
        raise ValueError("poll_interval_s must equal demand_policy.dispatch_latency_s.")
    if poller is not None and not isinstance(poller, DurableWorkerPoller):
        raise TypeError("poller must be a DurableWorkerPoller.")
    if max_handled is not None and max_handled < 0:
        raise ValueError("max_handled must be a non-negative integer.")

    handled = 0
    loop = asyncio.get_running_loop()
    worker_clock = loop.time if clock is None else clock
    if not callable(worker_clock):
        raise TypeError("clock must be callable.")
    local_empty_delay_s = demand_policy.minimum_idle_delay_s
    while (max_handled is None or handled < max_handled) and not worker_stop_requested(stop):
        if poller is not None:
            poller.begin_step()
        step_now = worker_clock()
        _validate_worker_clock(step_now)
        outcome = await step(step_now, handled)
        if type(outcome) is not DurableWorkerStep:
            raise TypeError("A durable worker step returned an invalid outcome.")
        handled += outcome.handled
        if outcome.activity:
            if poller is None:
                local_empty_delay_s = demand_policy.minimum_idle_delay_s
            else:
                poller.note_activity(delay_next=outcome.idle)
        if (
            outcome.stop
            or worker_stop_requested(stop)
            or (max_handled is not None and handled >= max_handled)
        ):
            break
        if outcome.continue_immediately or not outcome.idle:
            continue

        now = worker_clock()
        _validate_worker_clock(now)
        if poller is None:
            idle_wait_s = _jittered_idle_delay(
                local_empty_delay_s,
                policy=demand_policy,
                random_source=random,
            )
            if not outcome.activity:
                local_empty_delay_s = min(
                    demand_policy.maximum_idle_delay_s,
                    local_empty_delay_s * demand_policy.backoff_multiplier,
                )
        else:
            idle_wait_s = max(poller.next_wake_at() - now, 0.0)
        if outcome.next_wake_at is not None:
            idle_wait_s = min(
                idle_wait_s,
                max(outcome.next_wake_at - now, 0.0),
            )
        if idle_wait_s == 0:
            continue
        wait_result = _normalize_worker_wait_result(await wait(idle_wait_s, stop))
        if wait_result is DurableWorkerWaitResult.STOP:
            break
        if wait_result is DurableWorkerWaitResult.HINT:
            if poller is None:
                local_empty_delay_s = demand_policy.minimum_idle_delay_s
            else:
                poller.note_hint()
    return handled


def lease_heartbeat_interval(
    lease_seconds: float,
    *,
    maximum_s: float | None = None,
) -> float:
    """Return the canonical one-third-lease heartbeat interval."""

    if not isfinite(lease_seconds) or lease_seconds <= 0:
        raise ValueError("lease_seconds must be finite and positive.")
    interval = lease_seconds / 3
    if maximum_s is not None:
        if not isfinite(maximum_s) or maximum_s <= 0:
            raise ValueError("maximum_s must be finite and positive.")
        interval = min(interval, maximum_s)
    return interval


async def run_durable_lease_heartbeat(
    heartbeat: Callable[[], Awaitable[_HeartbeatUpdateT]],
    *,
    lease_seconds: float,
    stop: asyncio.Event,
    stopped_outcome: _HeartbeatOutcomeT,
    maximum_interval_s: float | None = None,
    after_heartbeat: Callable[[_HeartbeatUpdateT], Awaitable[_HeartbeatOutcomeT | None]]
    | None = None,
    on_failure: Callable[[Exception], Awaitable[_HeartbeatOutcomeT | None]] | None = None,
    wait: WorkerWait = wait_or_stop,
    lease_deadline: Callable[[], float] | None = None,
    deadline_failure: Callable[[], BaseException] | None = None,
    clock: Callable[[], float] = monotonic,
) -> _HeartbeatOutcomeT:
    """Maintain one lease until stopped or an adapter returns an outcome.

    Adapters supply authority-specific inspection and failure reconciliation.
    Returning ``None`` from either callback keeps the heartbeat alive; raising
    preserves the adapter failure and traceback.
    """

    if (lease_deadline is None) != (deadline_failure is None):
        raise ValueError("lease_deadline and deadline_failure must be provided together.")
    interval = lease_heartbeat_interval(
        lease_seconds,
        maximum_s=maximum_interval_s,
    )
    while not stop.is_set():
        wait_seconds = interval
        if lease_deadline is not None:
            remaining = lease_deadline() - clock()
            if remaining <= 0:
                assert deadline_failure is not None
                raise deadline_failure()
            half_remaining = remaining / 2
            if half_remaining + 1e-5 < wait_seconds:
                wait_seconds = half_remaining
        wait_result = _normalize_worker_wait_result(await wait(wait_seconds, stop))
        if wait_result is DurableWorkerWaitResult.STOP:
            return stopped_outcome
        try:
            update = await heartbeat()
            outcome = None if after_heartbeat is None else await after_heartbeat(update)
        except Exception as exc:
            if on_failure is None:
                raise
            outcome = await on_failure(exc)
        if outcome is not None:
            return outcome
    return stopped_outcome
