"""Runtime ownership of one renewable watcher delivery and its cleanup."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Iterable
from typing import Any, TypeVar

from cayu._task_wait import LateFailures, await_shielded_task_outcome, wait_until_idle
from cayu.observability.watchers import (
    EventWatcher,
    EventWatcherClaim,
    EventWatcherContext,
    EventWatcherDelivery,
    EventWatcherDeliveryStatus,
    EventWatcherLeaseLost,
    EventWatcherStore,
    copy_event_watcher_claim,
    copy_event_watcher_delivery,
    event_watcher_error_payload,
    run_event_watcher_handler,
)
from cayu.runtime._durable_worker_loop import run_durable_lease_heartbeat
from cayu.vaults import SecretRedactor

_T = TypeVar("_T")

# A delivery that ended so could not record its outcome durably.
_UNSETTLED_DELIVERY = frozenset(
    {EventWatcherDeliveryStatus.PUBLICATION_FAILED, EventWatcherDeliveryStatus.LEASE_LOST}
)


class EventWatcherSupervisor:
    """Retain handler ownership even when the caller stops waiting for cleanup."""

    def __init__(self) -> None:
        self._active: dict[str, asyncio.Task[None]] = {}
        self._store_tasks: set[asyncio.Future[Any]] = set()
        self._late_failures = LateFailures("An event watcher delivery")

    def active(self, watcher_name: str) -> bool:
        task = self._active.get(watcher_name)
        return task is not None and not task.done()

    def _running(self) -> set[asyncio.Future[Any]]:
        return {
            *(task for task in self._active.values() if not task.done()),
            *(task for task in self._store_tasks if not task.done()),
        }

    @property
    def pending(self) -> bool:
        """Whether delivery work still runs, or failed without being reported."""

        return self._late_failures.pending or bool(self._running())

    def record_undelivered(self, deliveries: Iterable[EventWatcherDelivery]) -> None:
        """Record outcomes a cancelled caller collected but never received."""

        for delivery in deliveries:
            if delivery.status in _UNSETTLED_DELIVERY:
                self._late_failures.record(delivery.status.value)

    def _record_unobserved_outcome(self, done: asyncio.Future[EventWatcherDelivery]) -> None:
        # Its caller left, so nobody else sees whether the delivery settled.
        if done.cancelled():
            return
        error = done.exception()
        if error is not None:
            self._late_failures.record(type(error).__qualname__)
        elif done.result().status in _UNSETTLED_DELIVERY:
            self._late_failures.record(done.result().status.value)

    async def drain(self, *, timeout_s: float) -> bool:
        """Wait up to ``timeout_s`` for deliveries to finish, without cancelling them.

        A delivery whose caller was cancelled keeps renewing its lease until its
        handler settles, then publishes its outcome; a synchronous handler runs
        until it returns. Returns False if any work is still running, and raises,
        once, for a delivery whose caller left and whose outcome was not recorded.
        """

        idle = await wait_until_idle(self._running, timeout_s=timeout_s)
        self._late_failures.raise_once()
        return idle

    async def store_call(self, work: Awaitable[_T]) -> _T:
        # Discovery/admission may initialize a database connection before any
        # lease exists. In-lease renewal/publication use the tighter deadline.
        return await self._bounded(work, timeout_s=30.0)

    async def _bounded(self, work: Awaitable[_T], *, timeout_s: float) -> _T:
        task = asyncio.ensure_future(work)
        self._store_tasks.add(task)

        def finished(done: asyncio.Future[Any]) -> None:
            self._store_tasks.discard(done)
            if not done.cancelled():
                done.exception()

        task.add_done_callback(finished)
        try:
            done, _ = await asyncio.wait({task}, timeout=max(0, timeout_s))
            if task not in done:
                task.cancel()
                raise TimeoutError(
                    "Event watcher store operation exceeded its lease-safe deadline."
                )
            try:
                return task.result()
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if current is None or not current.cancelling():
                    raise RuntimeError("Event watcher store operation cancelled itself.") from None
                raise
        except BaseException:
            if not task.done():
                task.cancel()
            raise

    async def run(
        self,
        *,
        watcher: EventWatcher,
        store: EventWatcherStore,
        claim: EventWatcherClaim,
        context: EventWatcherContext,
        cursor_sequence: int,
        redactor: SecretRedactor,
    ) -> EventWatcherDelivery:
        if self.active(watcher.name):
            raise EventWatcherLeaseLost("A previous watcher handler has not finished cleanup.")
        result: asyncio.Future[EventWatcherDelivery] = asyncio.get_running_loop().create_future()
        cancel = asyncio.Event()
        worker = asyncio.create_task(
            self._drive(
                watcher=watcher,
                store=store,
                claim=claim,
                context=context,
                cursor_sequence=cursor_sequence,
                redactor=redactor,
                result=result,
                cancel=cancel,
            )
        )
        self._active[watcher.name] = worker

        def finished(task: asyncio.Task[None]) -> None:
            if self._active.get(watcher.name) is task:
                del self._active[watcher.name]
            if not task.cancelled():
                error = task.exception()
                if error is not None and not result.done():
                    result.set_exception(error)
            elif not result.done():
                result.cancel()

        worker.add_done_callback(finished)
        try:
            return await asyncio.shield(result)
        except asyncio.CancelledError:
            cancel.set()
            # The worker retains renewal until cooperative cancellation or a
            # synchronous callback actually settles. No caller wait is unbounded.
            result.add_done_callback(self._record_unobserved_outcome)
            raise

    async def _drive(
        self,
        *,
        watcher: EventWatcher,
        store: EventWatcherStore,
        claim: EventWatcherClaim,
        context: EventWatcherContext,
        cursor_sequence: int,
        redactor: SecretRedactor,
        result: asyncio.Future[EventWatcherDelivery],
        cancel: asyncio.Event,
    ) -> None:
        loop = asyncio.get_running_loop()
        operation_timeout = min(30.0, watcher.lease_seconds / 3)
        deadline = loop.time() + watcher.lease_seconds
        stop = asyncio.Event()
        handler: asyncio.Task[None] | None = None
        heartbeat: asyncio.Task[None] | None = None
        cancellation: asyncio.Task[bool] | None = None

        def fail(error: Exception, status: EventWatcherDeliveryStatus) -> None:
            if not result.done():
                result.set_result(
                    EventWatcherDelivery(
                        watcher_name=claim.watcher_name,
                        event_id=claim.event_id,
                        event_sequence=claim.event_sequence,
                        attempt=claim.attempt,
                        status=status,
                        cursor_sequence=cursor_sequence,
                        error=redactor.redact_text_bounded(
                            event_watcher_error_payload(error, redactor=redactor), max_bytes=4096
                        ),
                    )
                )

        async def renew() -> None:
            nonlocal claim, deadline
            started = loop.time()
            try:
                renewed = copy_event_watcher_claim(
                    await self._bounded(
                        store.renew_claim(claim, lease_seconds=watcher.lease_seconds),
                        timeout_s=min(operation_timeout, max(0, deadline - started)),
                    )
                )
            except Exception as error:
                raise EventWatcherLeaseLost(
                    "Event watcher lease renewal could not prove ownership."
                ) from error
            if renewed.model_dump(exclude={"lease_expires_at"}) != claim.model_dump(
                exclude={"lease_expires_at"}
            ):
                raise EventWatcherLeaseLost("Event watcher renewal returned another claim.")
            claim = renewed
            deadline = started + watcher.lease_seconds

        try:
            # Required even for custom stores, before invoking trusted callbacks.
            await renew()
            if cancel.is_set():
                result.cancel()
                return
            heartbeat = asyncio.create_task(
                run_durable_lease_heartbeat(
                    renew,
                    lease_seconds=watcher.lease_seconds,
                    stop=stop,
                    stopped_outcome=None,
                    lease_deadline=lambda: deadline,
                    deadline_failure=lambda: EventWatcherLeaseLost(
                        "Event watcher lease renewal expired."
                    ),
                    clock=loop.time,
                )
            )
            handler = asyncio.create_task(run_event_watcher_handler(watcher, context))
            cancellation = asyncio.create_task(cancel.wait())
            waiting: set[asyncio.Task[Any]] = {handler, heartbeat, cancellation}
            while not handler.done():
                done, _ = await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
                if cancellation in done:
                    handler.cancel()
                    waiting.discard(cancellation)
                if heartbeat in done:
                    try:
                        heartbeat.result()
                    except asyncio.CancelledError:
                        raise EventWatcherLeaseLost(
                            "Event watcher heartbeat cancelled itself."
                        ) from None
                    raise EventWatcherLeaseLost(
                        "Event watcher heartbeat stopped before handler settlement."
                    )
            # A concurrent renewal failure is authoritative even if the handler
            # completed at the same scheduler boundary.
            stop.set()
            await self._bounded(heartbeat, timeout_s=operation_timeout)
            handler_error: str | None = None
            try:
                handler.result()
            except asyncio.CancelledError:
                handler_error = "Event watcher handler was cancelled."
            except Exception as error:
                handler_error = redactor.redact_text_bounded(
                    event_watcher_error_payload(error, redactor=redactor), max_bytes=4096
                )
            if cancel.is_set():
                handler_error = "Event watcher caller cancelled delivery."
            await renew()
            # Publish under a freshly proven lease. Exact receipts make a single
            # retry safe when the first acknowledgement was lost after commit.
            for attempt in range(2):
                try:
                    publication = (
                        store.mark_success(claim)
                        if handler_error is None
                        else store.mark_failure(
                            claim, error=handler_error, max_attempts=watcher.max_attempts
                        )
                    )
                    delivery = copy_event_watcher_delivery(
                        await self._bounded(
                            publication,
                            timeout_s=min(operation_timeout, max(0, deadline - loop.time())),
                        )
                    )
                    if (
                        delivery.watcher_name,
                        delivery.event_id,
                        delivery.event_sequence,
                        delivery.attempt,
                    ) != (claim.watcher_name, claim.event_id, claim.event_sequence, claim.attempt):
                        raise ValueError("Event watcher publication returned another delivery.")
                    terminal = delivery.status in {
                        EventWatcherDeliveryStatus.SUCCEEDED,
                        EventWatcherDeliveryStatus.DEAD_LETTERED,
                    }
                    if (
                        delivery.status
                        not in {
                            EventWatcherDeliveryStatus.SUCCEEDED,
                            EventWatcherDeliveryStatus.FAILED,
                            EventWatcherDeliveryStatus.DEAD_LETTERED,
                        }
                        or (terminal and delivery.cursor_sequence != claim.event_sequence)
                        or (not terminal and delivery.cursor_sequence != cursor_sequence)
                    ):
                        raise ValueError(
                            "Event watcher publication returned an invalid settlement."
                        )
                    if not result.done():
                        result.set_result(delivery)
                    return
                except EventWatcherLeaseLost:
                    raise
                except Exception:
                    if attempt or loop.time() >= deadline:
                        raise
        except EventWatcherLeaseLost as error:
            fail(error, EventWatcherDeliveryStatus.LEASE_LOST)
        except Exception as error:
            fail(error, EventWatcherDeliveryStatus.PUBLICATION_FAILED)
        finally:
            stop.set()
            if cancellation is not None:
                cancellation.cancel()
            if heartbeat is not None and not heartbeat.done():
                heartbeat.cancel()
                self._store_tasks.add(heartbeat)
                heartbeat.add_done_callback(self._consume_retained)
            if handler is not None and not handler.done():
                handler.cancel()
                # This worker stays registered while cancellation-opaque app code
                # drains, fencing another local invocation of the same watcher.
                await await_shielded_task_outcome(handler)

    def _consume_retained(self, task: asyncio.Future[Any]) -> None:
        self._store_tasks.discard(task)
        if not task.cancelled():
            task.exception()
