"""Content-free, loss-tolerant wakeups for durable task workers."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from threading import Lock
from typing import Any

from cayu.runtime._durable_worker_loop import DurableWorkerWaitResult

_BROADCAST = object()
_AdmissionMatcher = Callable[[object], bool]


@dataclass(slots=True)
class _TaskAdmissionSubscriber:
    token: int
    loop: asyncio.AbstractEventLoop
    event: asyncio.Event
    matcher: _AdmissionMatcher
    waiting: bool = False
    pending: bool = False
    closed: bool = False


class TaskAdmissionWakeup:
    """One worker's content-free, coalescing admission wake subscription."""

    __slots__ = ("_broker", "_subscriber")

    def __init__(
        self,
        broker: TaskAdmissionWakeupBroker,
        subscriber: _TaskAdmissionSubscriber,
    ) -> None:
        self._broker = broker
        self._subscriber = subscriber

    async def wait(self, seconds: float, stop: asyncio.Event | None) -> bool:
        """Wait for a matching hint, the bounded audit deadline, or shutdown.

        ``True`` means shutdown won. A hint and an ordinary timeout both return
        ``False`` because either outcome merely asks the shared worker loop to
        perform another authoritative store claim.
        """

        result = await self.wait_for_worker(seconds, stop)
        return result is DurableWorkerWaitResult.STOP

    async def wait_for_worker(
        self,
        seconds: float,
        stop: asyncio.Event | None,
    ) -> DurableWorkerWaitResult:
        """Return whether a hint, audit timeout, or shutdown ended the wait."""

        return await self._broker.wait_for_worker(self._subscriber, seconds, stop)

    def close(self) -> None:
        """Unregister this worker without retaining query or task data."""

        self._broker.unsubscribe(self._subscriber)


class TaskAdmissionWakeupBroker:
    """Route one content-free wake to a compatible in-process worker.

    The broker uses task attributes only inside its matcher. Subscribers receive
    an ``asyncio.Event`` edge, never the task, query values, or workload content.
    Events coalesce, and every wait remains bounded by the worker poll interval.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._subscribers: dict[int, _TaskAdmissionSubscriber] = {}
        self._next_token = 1
        self._last_woken_token = 0

    @property
    def subscriber_count(self) -> int:
        """Return the process-local waiter count for bounded lifecycle tests."""

        with self._lock:
            return len(self._subscribers)

    def subscribe(self, matcher: _AdmissionMatcher) -> TaskAdmissionWakeup:
        """Register one worker on its current event loop."""

        if not callable(matcher):
            raise TypeError("Task admission wake matcher must be callable.")
        loop = asyncio.get_running_loop()
        with self._lock:
            token = self._next_token
            self._next_token += 1
            subscriber = _TaskAdmissionSubscriber(
                token=token,
                loop=loop,
                event=asyncio.Event(),
                matcher=matcher,
            )
            self._subscribers[token] = subscriber
        return TaskAdmissionWakeup(self, subscriber)

    def publish(self, admitted: object = _BROADCAST) -> None:
        """Wake at most one compatible worker; duplicate edges may coalesce."""

        with self._lock:
            compatible = [
                subscriber
                for subscriber in self._subscribers.values()
                if not subscriber.closed
                and not subscriber.pending
                and (admitted is _BROADCAST or subscriber.matcher(admitted))
            ]
            waiting = [subscriber for subscriber in compatible if subscriber.waiting]
            candidates = waiting or compatible
            if not candidates:
                return
            candidates.sort(key=lambda subscriber: subscriber.token)
            subscriber = next(
                (candidate for candidate in candidates if candidate.token > self._last_woken_token),
                candidates[0],
            )
            subscriber.pending = True
            self._last_woken_token = subscriber.token
        self._set_event(subscriber)

    async def wait(
        self,
        subscriber: _TaskAdmissionSubscriber,
        seconds: float,
        stop: asyncio.Event | None,
    ) -> bool:
        """Consume one coalesced edge while preserving the legacy bool result."""

        result = await self.wait_for_worker(subscriber, seconds, stop)
        return result is DurableWorkerWaitResult.STOP

    async def wait_for_worker(
        self,
        subscriber: _TaskAdmissionSubscriber,
        seconds: float,
        stop: asyncio.Event | None,
    ) -> DurableWorkerWaitResult:
        """Consume one edge and report what ended the bounded fallback wait."""

        if not isinstance(seconds, int | float) or not isfinite(seconds) or seconds < 0:
            raise ValueError("Task admission wait duration must be finite and non-negative.")
        if asyncio.get_running_loop() is not subscriber.loop:
            raise RuntimeError("Task admission wakeups must be awaited on their subscribing loop.")
        if stop is not None and stop.is_set():
            return DurableWorkerWaitResult.STOP

        with self._lock:
            if subscriber.closed:
                raise RuntimeError("Task admission wakeup is closed.")
            subscriber.waiting = True
            pending = subscriber.pending
            if pending:
                subscriber.pending = False
                subscriber.waiting = False
        if pending:
            subscriber.event.clear()
            return (
                DurableWorkerWaitResult.STOP
                if stop is not None and stop.is_set()
                else DurableWorkerWaitResult.HINT
            )

        try:
            return await _wait_for_hint_or_stop(subscriber.event, float(seconds), stop)
        finally:
            with self._lock:
                subscriber.waiting = False
                subscriber.pending = False
            subscriber.event.clear()

    def unsubscribe(self, subscriber: _TaskAdmissionSubscriber) -> None:
        """Remove one exact subscription idempotently."""

        with self._lock:
            current = self._subscribers.get(subscriber.token)
            if current is not subscriber:
                return
            subscriber.closed = True
            subscriber.waiting = False
            subscriber.pending = False
            del self._subscribers[subscriber.token]
        self._set_event(subscriber)

    @staticmethod
    def _set_event(subscriber: _TaskAdmissionSubscriber) -> None:
        if subscriber.loop.is_closed():
            return
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if current_loop is subscriber.loop:
            subscriber.event.set()
            return
        if subscriber.loop.is_running():
            subscriber.loop.call_soon_threadsafe(subscriber.event.set)


async def _wait_for_hint_or_stop(
    hint: asyncio.Event,
    seconds: float,
    stop: asyncio.Event | None,
) -> DurableWorkerWaitResult:
    if stop is None:
        try:
            await asyncio.wait_for(hint.wait(), timeout=seconds)
        except TimeoutError:
            return DurableWorkerWaitResult.TIMEOUT
        return DurableWorkerWaitResult.HINT

    hint_wait = asyncio.create_task(hint.wait(), name="cayu-task-admission-hint")
    stop_wait = asyncio.create_task(stop.wait(), name="cayu-task-worker-stop")
    waits: tuple[asyncio.Task[Any], ...] = (hint_wait, stop_wait)
    try:
        done, _ = await asyncio.wait(
            waits,
            timeout=seconds,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if stop_wait in done and stop.is_set():
            return DurableWorkerWaitResult.STOP
        if hint_wait in done and hint.is_set():
            return DurableWorkerWaitResult.HINT
        return DurableWorkerWaitResult.TIMEOUT
    finally:
        for wait in waits:
            if not wait.done():
                wait.cancel()
        await asyncio.gather(*waits, return_exceptions=True)
