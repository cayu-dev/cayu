"""Owned, renewable delivery of a foreground child's parent continuation."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar, cast

from cayu._task_wait import (
    CapturedAwaitableOutcome,
    await_shielded_task_outcome,
    capture_awaitable_outcome,
    unexpected_child_cancellation_error,
)
from cayu.runtime._durable_worker_loop import run_durable_lease_heartbeat
from cayu.runtime.sessions import (
    PersistedEventSideEffectClaim,
    PersistedEventSideEffectClaimLost,
    PersistedEventSideEffectDelivery,
    PersistedEventSideEffectStatus,
    SessionStore,
)

_T = TypeVar("_T")


class ForegroundChildDeliveryOwner:
    """Keep renewal and cleanup owned when a delivery caller stops waiting."""

    def __init__(self, store: SessionStore) -> None:
        self._store = store
        self._workers: dict[str, asyncio.Task[CapturedAwaitableOutcome[None]]] = {}
        self._store_tasks: set[asyncio.Future[Any]] = set()

    def active(self, child_session_id: str) -> bool:
        task = self._workers.get(child_session_id)
        return task is not None and not task.done()

    def _consume_store_task(self, task: asyncio.Future[Any]) -> None:
        self._store_tasks.discard(task)
        if not task.cancelled():
            task.exception()

    async def _bounded(self, operation: Awaitable[_T], timeout_s: float) -> _T:
        task = asyncio.create_task(capture_awaitable_outcome(lambda: operation))
        self._store_tasks.add(task)
        task.add_done_callback(self._consume_store_task)
        done, _ = await asyncio.wait({task}, timeout=max(0, timeout_s))
        if task not in done:
            # The store owns any already-dispatched write; retain its task rather
            # than treating a timeout as evidence that the write stopped.
            raise PersistedEventSideEffectClaimLost("Foreground delivery renewal timed out.")
        outcome = task.result()
        if outcome.error is not None:
            raise outcome.error
        return cast("_T", outcome.result)

    async def run(
        self,
        claim: PersistedEventSideEffectClaim,
        operation: Callable[[Callable[[], Awaitable[None]]], Awaitable[None]],
    ) -> None:
        claim = PersistedEventSideEffectClaim.model_validate_json(claim.model_dump_json())
        if self.active(claim.session_id):
            raise PersistedEventSideEffectClaimLost(
                "Foreground delivery is still settling locally."
            )
        result: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        cancel = asyncio.Event()
        worker = asyncio.create_task(
            capture_awaitable_outcome(lambda: self._drive(claim, operation, cancel))
        )
        self._workers[claim.session_id] = worker

        def finished(task: asyncio.Task[CapturedAwaitableOutcome[None]]) -> None:
            if self._workers.get(claim.session_id) is task:
                del self._workers[claim.session_id]
            if task.cancelled():
                if not result.done():
                    result.cancel()
            else:
                error = task.result().error
                if not result.done():
                    if error is None:
                        result.set_result(None)
                    elif isinstance(error, asyncio.CancelledError):
                        result.cancel()
                    else:
                        result.set_exception(error)

        worker.add_done_callback(finished)
        try:
            await asyncio.shield(result)
        except asyncio.CancelledError as error:
            current = asyncio.current_task()
            if current is None or not current.cancelling():
                raise unexpected_child_cancellation_error(
                    error, operation="Foreground continuation"
                ) from error
            cancel.set()
            result.add_done_callback(lambda done: None if done.cancelled() else done.exception())
            raise

    async def _drive(
        self,
        claim: PersistedEventSideEffectClaim,
        operation: Callable[[Callable[[], Awaitable[None]]], Awaitable[None]],
        cancel: asyncio.Event,
    ) -> None:
        loop = asyncio.get_running_loop()
        delivery = await self._bounded(
            self._store.get_persisted_event_side_effect_delivery(
                session_id=claim.session_id, event_id=claim.event_id
            ),
            30.0,
        )
        self._require_exact_delivery(claim, delivery)
        assert delivery is not None and delivery.lease_expires_at is not None
        lease_seconds = min(
            300.0, (delivery.lease_expires_at - delivery.updated_at).total_seconds()
        )
        if lease_seconds <= 0:
            raise PersistedEventSideEffectClaimLost("Foreground delivery has no live-shaped lease.")
        deadline = loop.time() + lease_seconds
        stop = asyncio.Event()
        heartbeat: asyncio.Task[CapturedAwaitableOutcome[None]] | None = None
        running: asyncio.Task[CapturedAwaitableOutcome[None]] | None = None
        cancellation: asyncio.Task[bool] | None = None
        errors: list[BaseException] = []
        operation_cancel_requested = False

        async def renew() -> None:
            nonlocal deadline
            started = loop.time()
            renewed = await self._bounded(
                self._store.renew_persisted_event_side_effect(claim, lease_seconds=lease_seconds),
                min(30.0, lease_seconds / 3, max(0, deadline - started)),
            )
            self._require_exact_delivery(claim, renewed)
            assert renewed.lease_expires_at is not None
            if (renewed.lease_expires_at - renewed.updated_at).total_seconds() < lease_seconds:
                raise PersistedEventSideEffectClaimLost(
                    "Foreground delivery renewal did not establish the requested lease."
                )
            deadline = started + lease_seconds

        try:
            # Store-clock renewal, not local clock comparison, establishes the
            # initial positive authority before recovery or execution begins.
            await renew()
            if cancel.is_set():
                return
            heartbeat = asyncio.create_task(
                capture_awaitable_outcome(
                    lambda: run_durable_lease_heartbeat(
                        renew,
                        lease_seconds=lease_seconds,
                        stop=stop,
                        stopped_outcome=None,
                        lease_deadline=lambda: deadline,
                        deadline_failure=lambda: PersistedEventSideEffectClaimLost(
                            "Foreground delivery ownership expired."
                        ),
                        clock=loop.time,
                    )
                )
            )
            running = asyncio.create_task(capture_awaitable_outcome(lambda: operation(renew)))
            cancellation = asyncio.create_task(cancel.wait())
            waiting: set[asyncio.Task[Any]] = {running, heartbeat, cancellation}
            while not running.done():
                done, _ = await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
                if cancellation in done:
                    operation_cancel_requested = running.cancel() or operation_cancel_requested
                    waiting.discard(cancellation)
                if heartbeat in done:
                    heartbeat_error = heartbeat.result().error
                    if heartbeat_error is not None:
                        raise heartbeat_error
                    raise PersistedEventSideEffectClaimLost(
                        "Foreground delivery heartbeat stopped."
                    )
            stop.set()
        except BaseException as error:
            errors.append(error)
        finally:
            if cancellation is not None:
                cancellation.cancel()
                await asyncio.gather(cancellation, return_exceptions=True)
            if running is not None:
                if not running.done():
                    operation_cancel_requested = running.cancel() or operation_cancel_requested
                # This is an owned background worker, not the foreground caller.
                # Keep renewal alive while the runtime settles cancellation.
                outcome = await await_shielded_task_outcome(running)
                operation_error = outcome.error if outcome.result is None else outcome.result.error
                if operation_error is not None and not (
                    errors
                    and operation_cancel_requested
                    and isinstance(operation_error, asyncio.CancelledError)
                ):
                    errors.append(operation_error)
                if outcome.cancellation is not None:
                    errors.append(outcome.cancellation)
            stop.set()
            if heartbeat is not None:
                outcome = await await_shielded_task_outcome(heartbeat)
                heartbeat_error = outcome.error if outcome.result is None else outcome.result.error
                if heartbeat_error is not None and all(heartbeat_error is not e for e in errors):
                    errors.append(heartbeat_error)
                if outcome.cancellation is not None:
                    errors.append(outcome.cancellation)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup(
                "Foreground continuation and ownership settlement failed.", errors
            )

    @staticmethod
    def _require_exact_delivery(
        claim: PersistedEventSideEffectClaim,
        delivery: PersistedEventSideEffectDelivery | None,
    ) -> None:
        if (
            delivery is None
            or (
                delivery.session_id,
                delivery.event_id,
                delivery.event_sequence,
                delivery.claim_id,
                delivery.attempts,
                delivery.status,
            )
            != (
                claim.session_id,
                claim.event_id,
                claim.event_sequence,
                claim.claim_id,
                claim.attempt,
                PersistedEventSideEffectStatus.LEASED,
            )
            or delivery.lease_expires_at is None
        ):
            raise PersistedEventSideEffectClaimLost("Foreground delivery no longer owns its claim.")

    async def drain(self, *, timeout_s: float) -> bool:
        if type(timeout_s) not in {int, float} or not 0 <= timeout_s < float("inf"):
            raise ValueError("timeout_s must be finite and non-negative.")
        pending = {*self._workers.values(), *self._store_tasks}
        if not pending:
            return True
        _, remaining = await asyncio.wait(pending, timeout=max(0, timeout_s))
        return not remaining and not self._workers and not self._store_tasks
