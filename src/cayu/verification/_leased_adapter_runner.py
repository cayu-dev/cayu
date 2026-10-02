"""Process-local ownership of leased extension work and its exact settlement.

Durable authority and outward diagnostics belong to the consuming coordinator.
An adapter finishing does not release its capacity or its retained drain: those
belong to the complete claim/publication settlement, which may outlive dispatch.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable, Coroutine, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from typing import Generic, TypeVar, cast

from cayu._exception_groups import exception_cause, iter_exception_tree, set_exception_cause
from cayu._task_wait import (
    CapturedAwaitableOutcome,
    ShieldedTaskOutcome,
    await_shielded_task_outcome,
    capture_awaitable_outcome,
    consume_pending_task_cancellation,
    restore_task_cancellation_requests,
)

_T = TypeVar("_T")
_DrainT = TypeVar("_DrainT")

_SETTLEMENT_START_POLL_SECONDS = 0.01


@dataclass(slots=True)
class _SingleFlightLock:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0


@dataclass(frozen=True, slots=True)
class LeasedAdapterLease:
    """Cancel dispatch once on lease loss, using the lease's exact provenance.

    Publication leases omit this policy: their callbacks must settle before
    publication ownership is released, even when the lease becomes unhealthy.
    """

    ownership_lost: asyncio.Future[BaseException] = field(repr=False)
    heartbeat: asyncio.Task[None] = field(repr=False)
    marker: object | None = None


@dataclass(frozen=True, slots=True)
class LeasedAdapterInvocation(Generic[_T]):
    task: asyncio.Task[CapturedAwaitableOutcome[_T]] = field(repr=False)
    outcome: ShieldedTaskOutcome[CapturedAwaitableOutcome[_T]] = field(repr=False)


@dataclass(slots=True)
class LeasedAdapterSettlement(Generic[_T]):
    """Retain one exact settlement and its observed domain failure."""

    settlement_task: asyncio.Task[_T] | None = field(default=None, init=False, repr=False)
    settlement_failure: BaseException | None = field(default=None, init=False, repr=False)
    settlement_processed: bool = field(default=False, init=False)
    failure_reported: bool = field(default=False, init=False)


class LeasedAdapterRunner(Generic[_DrainT]):
    """Own local locks, admission slots, tasks, heartbeats and retained drains.

    Single-flight encloses durable admission and publication, not just the
    extension callback. Capacity is reserved before admission and released by
    its settlement owner. Cancellation/timeout returns the exact invocation;
    the coordinator retains that invocation with its authority-specific drain.
    """

    def __init__(self) -> None:
        self._process_id = os.getpid()
        self._locks: dict[str, _SingleFlightLock] = {}
        self._capacity_reservations: set[object] = set()
        self._adapter_tasks: set[asyncio.Task[object]] = set()
        self._heartbeat_tasks: set[asyncio.Task[object]] = set()
        self._drains: dict[str, _DrainT] = {}

    @property
    def active_adapter_count(self) -> int:
        return len(self._adapter_tasks)

    @property
    def active_heartbeat_count(self) -> int:
        return len(self._heartbeat_tasks)

    def _running(self) -> tuple[set[asyncio.Future[object]], bool]:
        running: set[asyncio.Future[object]] = {*self._adapter_tasks, *self._heartbeat_tasks}
        unsettled = False
        for drain in self._drains.values():
            # Only a settlement records whether its work has finished.
            if not isinstance(drain, LeasedAdapterSettlement):
                continue
            task = drain.settlement_task
            if task is None:
                # The adapter finished but its settlement has not started yet.
                unsettled = True
            elif not task.done():
                running.add(cast("asyncio.Future[object]", task))
        return running, unsettled

    def report_settlement_failures(self) -> list[BaseException | None]:
        """Return, once each, retained settlements that finished without success.

        ``None`` stands for a settlement that ended without an outcome. A reported
        drain stays retained, so the exact retry still receives and acknowledges
        its failure.
        """

        reported: list[BaseException | None] = []
        for drain in self._drains.values():
            if not isinstance(drain, LeasedAdapterSettlement) or drain.failure_reported:
                continue
            task = drain.settlement_task
            if task is None or not task.done():
                continue
            if drain.settlement_processed:
                if drain.settlement_failure is None:
                    continue
                reported.append(drain.settlement_failure)
            elif task.cancelled() or task.exception() is not None:
                reported.append(None)
            else:
                # Its done callback has not recorded the outcome yet.
                continue
            drain.failure_reported = True
        return reported

    @property
    def pending(self) -> bool:
        """Whether adapters, heartbeats or settlements this runner owns still run."""

        running, unsettled = self._running()
        return bool(running) or unsettled

    async def wait_idle(self, timeout_s: float) -> bool:
        """Wait up to ``timeout_s`` for owned work to finish, without cancelling it.

        Adapters may resist cancellation, so shutdown only observes them; work
        still running when the time is up keeps running and is reported.
        """

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while True:
            running, unsettled = self._running()
            if not running and not unsettled:
                return True
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False
            if running:
                await asyncio.wait(running, timeout=remaining, return_when=asyncio.FIRST_COMPLETED)
            else:
                # A finished adapter's settlement starts from its done callback.
                await asyncio.sleep(min(remaining, _SETTLEMENT_START_POLL_SECONDS))

    def ensure_process_local(
        self,
        failure: Callable[[], BaseException],
        *,
        additional_active: bool = False,
    ) -> bool:
        """Refresh a quiescent generation; reject inherited active ownership."""
        process_id = os.getpid()
        if process_id == self._process_id:
            return False
        if (
            self._locks
            or self._capacity_reservations
            or self._adapter_tasks
            or self._heartbeat_tasks
            or self._drains
            or additional_active
        ):
            error = failure()
            del failure
            raise error from None
        self._process_id = process_id
        return True

    @contextmanager
    def single_flight(self, key: str) -> Iterator[asyncio.Lock]:
        """Keep one lock alive through every holder and cancelled waiter.

        The consumer acquires/releases it and sanitizes acquisition failures at
        its existing public boundary. No extension data enters this scope.
        """
        entry = self._locks.get(key)
        if entry is None:
            entry = _SingleFlightLock()
            self._locks[key] = entry
        entry.users += 1
        try:
            yield entry.lock
        finally:
            entry.users -= 1
            if entry.users == 0 and self._locks.get(key) is entry:
                self._locks.pop(key, None)

    def reserve_capacity(self, limit: int, failure: Callable[[], BaseException]) -> object:
        if len(self._capacity_reservations) >= limit:
            error = failure()
            del failure
            raise error from None
        reservation = object()
        self._capacity_reservations.add(reservation)
        return reservation

    def release_capacity(self, reservation: object) -> None:
        self._capacity_reservations.discard(reservation)

    def draining(self, key: str) -> _DrainT | None:
        return self._drains.get(key)

    def retain_drain(self, key: str, drain: _DrainT) -> None:
        self._drains[key] = drain

    def acknowledge_drain(self, key: str, expected: _DrainT) -> None:
        """An old callback or handle must never consume a successor's drain."""
        if self._drains.get(key) is expected:
            self._drains.pop(key, None)

    def start_settlement(
        self,
        key: str,
        drain: LeasedAdapterSettlement[_T],
        operation: Callable[[], Coroutine[object, object, _T]],
        *,
        name: str,
        failure_for: Callable[[_T], BaseException | None],
    ) -> asyncio.Task[_T]:
        """Start the settlement once, including when an observer beats its callback."""
        if drain.settlement_task is None:
            self.adopt_settlement(
                key,
                drain,
                asyncio.create_task(operation(), name=name),
                failure_for=failure_for,
            )
        assert drain.settlement_task is not None
        return drain.settlement_task

    def adopt_settlement(
        self,
        key: str,
        drain: LeasedAdapterSettlement[_T],
        task: asyncio.Task[_T],
        *,
        failure_for: Callable[[_T], BaseException | None],
        on_settled: Callable[[], None] | None = None,
    ) -> None:
        """Own an existing task without restarting work or duplicating completion."""
        if drain.settlement_task is not None:
            if drain.settlement_task is not task:
                raise RuntimeError("A drain already owns a different settlement task.")
            return
        drain.settlement_task = task

        def settled(completed: asyncio.Task[_T]) -> None:
            # A replaced drain still owns its original capacity reservation.
            if on_settled is not None:
                on_settled()
            self.finalize_settlement(key, drain, completed, failure_for=failure_for)

        task.add_done_callback(settled)

    def finalize_settlement(
        self,
        key: str,
        drain: LeasedAdapterSettlement[_T],
        completed: asyncio.Task[_T],
        *,
        failure_for: Callable[[_T], BaseException | None],
    ) -> bool:
        """Observe a terminal result once; retain failure until explicit acknowledgement."""
        if drain.settlement_task is not completed:
            return False
        if drain.settlement_processed:
            return True
        try:
            result = completed.result()
        except BaseException:
            return False
        failure = failure_for(result)
        drain.settlement_failure = failure
        drain.settlement_processed = True
        if failure is None and self._drains.get(key) is drain:
            self._drains.pop(key, None)
        return True

    def start_heartbeat(
        self, operation: Coroutine[object, object, _T], *, name: str
    ) -> asyncio.Task[_T]:
        task = asyncio.create_task(operation, name=name)
        owned_task = cast("asyncio.Task[object]", task)
        self._heartbeat_tasks.add(owned_task)
        # Reading a cancelled Task can consume the only CancelledError that
        # retains its cleanup cause. The lease settlement owner chooses when
        # to observe that outcome; bookkeeping only releases local tracking.
        owned_task.add_done_callback(self._heartbeat_tasks.discard)
        return task

    def _track_task(self, task: asyncio.Task[_T], owned: set[asyncio.Task[object]]) -> None:
        owned_task = cast("asyncio.Task[object]", task)
        owned.add(owned_task)

        def settled(completed: asyncio.Task[object]) -> None:
            owned.discard(completed)
            with suppress(BaseException):
                completed.result()

        owned_task.add_done_callback(settled)

    async def run(
        self,
        key: str,
        operation: Callable[[], Awaitable[_T]],
        *,
        name: str,
        timeout_seconds: float,
        lease: LeasedAdapterLease | None = None,
        on_settled: Callable[[str, asyncio.Task[CapturedAwaitableOutcome[_T]]], None] | None = None,
    ) -> LeasedAdapterInvocation[_T]:
        try:
            task = asyncio.create_task(capture_awaitable_outcome(operation), name=name)
        finally:
            del operation
        self._track_task(task, self._adapter_tasks)
        if on_settled is not None:
            task.add_done_callback(lambda completed: on_settled(key, completed))
        release_watch = (
            None if lease is None or lease.marker is None else self._watch_lease(task, lease)
        )
        del lease
        outcome = await await_shielded_task_outcome(
            task,
            timeout_s=timeout_seconds,
            timeout_after_cancellation_s=0,
        )
        if task.done():
            self._adapter_tasks.discard(cast("asyncio.Task[object]", task))
            if release_watch is not None:
                release_watch()
        return LeasedAdapterInvocation(task=task, outcome=outcome)

    @staticmethod
    def _watch_lease(
        task: asyncio.Task[CapturedAwaitableOutcome[_T]], lease: LeasedAdapterLease
    ) -> Callable[[], None]:
        cancellation_requested = False

        def cancel_once() -> None:
            nonlocal cancellation_requested
            if cancellation_requested or task.done():
                return
            cancellation_requested = True
            task.cancel(lease.marker)

        def ownership_lost(completed: asyncio.Future[BaseException]) -> None:
            if completed.cancelled():
                return
            try:
                completed.result()
            except BaseException:
                return
            cancel_once()

        def heartbeat_ended(completed: asyncio.Task[None]) -> None:
            if completed.cancelled():
                return
            try:
                completed.result()
            except BaseException:
                cancel_once()

        lease.ownership_lost.add_done_callback(ownership_lost)
        lease.heartbeat.add_done_callback(heartbeat_ended)

        def release_watch() -> None:
            lease.ownership_lost.remove_done_callback(ownership_lost)
            lease.heartbeat.remove_done_callback(heartbeat_ended)

        task.add_done_callback(lambda _completed: release_watch())
        return release_watch

    async def run_to_settlement(
        self,
        action: Callable[[], Awaitable[_T]],
        heartbeat_action: Callable[[asyncio.Event], Awaitable[object]],
    ) -> _T:
        """Own worker work and its heartbeat until both are terminal.

        Worker callbacks may retain cleanup causes on cancellation. Capture
        them before asyncio normalizes task cancellation, and restore consumed
        caller requests only after observing both exact children.
        """
        stop = asyncio.Event()
        heartbeat = self.start_heartbeat(
            capture_owned_task_outcome(lambda: heartbeat_action(stop)),
            name="cayu-leased-worker-heartbeat",
        )
        work = asyncio.create_task(capture_owned_task_outcome(action))
        del action
        self._track_task(work, self._adapter_tasks)
        failure = None
        result = None
        consumed = 0
        try:
            done, _ = await asyncio.wait({work, heartbeat}, return_when=asyncio.FIRST_COMPLETED)
            if heartbeat in done:
                heartbeat_result = heartbeat.result()
                if heartbeat_result.error is not None:
                    raise heartbeat_result.error
            completed = await asyncio.shield(work)
            if completed.error is not None:
                raise completed.error
            result = completed.result
        except BaseException as error:
            failure = error
            forwarded = work.cancel()
            settled = await await_shielded_task_outcome(
                work,
                cancellation=error if isinstance(error, asyncio.CancelledError) else None,
            )
            consumed += settled.cancellation_requests_consumed
            child_error = settled.error or (
                None if settled.result is None else settled.result.error
            )
            if forwarded and isinstance(child_error, asyncio.CancelledError):
                child_error = exception_cause(child_error)
            failure = merge_owned_failures(failure, child_error)
            failure = merge_owned_failures(failure, settled.cancellation)
        finally:
            stop.set()
            settled_heartbeat = await await_shielded_task_outcome(heartbeat)
            consumed += settled_heartbeat.cancellation_requests_consumed
            heartbeat_error = settled_heartbeat.error or (
                None if settled_heartbeat.result is None else settled_heartbeat.result.error
            )
            failure = merge_owned_failures(failure, heartbeat_error)
            failure = merge_owned_failures(failure, settled_heartbeat.cancellation)
        restore_task_cancellation_requests(
            consumed, cancellation=failure if isinstance(failure, asyncio.CancelledError) else None
        )
        if failure is not None:
            raise failure from exception_cause(failure)
        return cast("_T", result)


async def capture_owned_task_outcome(
    factory: Callable[[], Awaitable[_T]],
) -> CapturedAwaitableOutcome[_T]:
    """Retain directly captured cancellation evidence through task completion."""
    outcome = await capture_awaitable_outcome(factory)
    if isinstance(outcome.error, asyncio.CancelledError):
        consume_pending_task_cancellation(outcome.error)
    return outcome


def merge_owned_failures(
    primary: BaseException | None, secondary: BaseException | None
) -> BaseException | None:
    """Preserve ordered failures while keeping current cancellation outward."""
    if primary is None:
        return secondary
    if secondary is None:
        return primary

    def contains(root, target):
        pending = [root]
        seen = set()
        while pending:
            current = pending.pop()
            if id(current) in seen:
                continue
            seen.add(id(current))
            if current is target:
                return True
            pending.extend(item for item in iter_exception_tree(current) if item is not current)
            cause = exception_cause(current)
            if cause is not None:
                pending.append(cause)
        return False

    if contains(primary, secondary):
        return primary
    if contains(secondary, primary):
        return secondary
    if isinstance(primary, asyncio.CancelledError):
        set_exception_cause(primary, merge_owned_failures(exception_cause(primary), secondary))
        return primary
    if isinstance(secondary, asyncio.CancelledError):
        set_exception_cause(secondary, merge_owned_failures(primary, exception_cause(secondary)))
        return secondary
    return BaseExceptionGroup("Verified task owned failures", [primary, secondary])
