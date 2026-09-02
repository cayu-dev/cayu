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
from math import isfinite
from typing import TypeVar

_HeartbeatUpdateT = TypeVar("_HeartbeatUpdateT")
_HeartbeatOutcomeT = TypeVar("_HeartbeatOutcomeT")
_MaintenanceOutcomeT = TypeVar("_MaintenanceOutcomeT")

WorkerWait = Callable[[float, asyncio.Event | None], Awaitable[bool]]


@dataclass(frozen=True)
class DurableWorkerStep:
    """One adapter-owned worker step returned to the shared scheduler."""

    handled: int = 0
    idle: bool = False
    continue_immediately: bool = False
    stop: bool = False
    next_wake_at: float | None = None

    def __post_init__(self) -> None:
        if type(self.handled) is not int or self.handled < 0:
            raise ValueError("handled must be a non-negative integer.")
        if self.idle and self.continue_immediately:
            raise ValueError("An idle worker step cannot also continue immediately.")
        if self.next_wake_at is not None:
            if not isfinite(self.next_wake_at) or self.next_wake_at < 0:
                raise ValueError("next_wake_at must be finite and non-negative.")
            if not self.idle:
                raise ValueError("next_wake_at requires an idle worker step.")


@dataclass
class DurableWorkerCadence:
    """Runtime-owned fixed cadence for one adapter maintenance operation.

    ``every_s=None`` represents an operation, such as the generic task-worker's
    legacy reclaim policy, that runs before every claim cycle. Timed operations
    run immediately on the first cycle and then at most once per interval.
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

    if not isfinite(seconds) or seconds <= 0:
        raise ValueError(f"{field_name} must be finite and positive.")


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
) -> int:
    """Run adapter steps with one canonical idle-wait and stop contract."""

    validate_worker_interval(poll_interval_s, "poll_interval_s")
    if max_handled is not None and max_handled < 0:
        raise ValueError("max_handled must be a non-negative integer.")

    handled = 0
    loop = asyncio.get_running_loop()
    while (max_handled is None or handled < max_handled) and not worker_stop_requested(stop):
        outcome = await step(loop.time(), handled)
        if type(outcome) is not DurableWorkerStep:
            raise TypeError("A durable worker step returned an invalid outcome.")
        handled += outcome.handled
        if (
            outcome.stop
            or worker_stop_requested(stop)
            or (max_handled is not None and handled >= max_handled)
        ):
            break
        if outcome.continue_immediately or not outcome.idle:
            continue

        idle_wait_s = poll_interval_s
        if outcome.next_wake_at is not None:
            idle_wait_s = min(
                idle_wait_s,
                max(outcome.next_wake_at - loop.time(), 0.0),
            )
        if idle_wait_s == 0:
            continue
        if await wait(idle_wait_s, stop):
            break
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
) -> _HeartbeatOutcomeT:
    """Maintain one lease until stopped or an adapter returns an outcome.

    Adapters supply authority-specific inspection and failure reconciliation.
    Returning ``None`` from either callback keeps the heartbeat alive; raising
    preserves the adapter failure and traceback.
    """

    interval = lease_heartbeat_interval(
        lease_seconds,
        maximum_s=maximum_interval_s,
    )
    while not stop.is_set():
        if await wait(interval, stop):
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
