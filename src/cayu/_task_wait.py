from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Generic, TypeVar

_ResultT = TypeVar("_ResultT")


@dataclass(frozen=True)
class ShieldedTaskOutcome(Generic[_ResultT]):
    """One child outcome observed without discarding caller cancellation."""

    result: _ResultT | None = None
    error: BaseException | None = None
    cancellation: asyncio.CancelledError | None = None
    timed_out: bool = False


def unexpected_child_cancellation_error(
    cancellation: asyncio.CancelledError,
    *,
    operation: str,
) -> RuntimeError:
    """Classify child-only cancellation as an operational failure."""

    error = RuntimeError(f"{operation} was cancelled without caller cancellation.")
    error.__cause__ = cancellation
    return error


def consume_pending_task_cancellation(
    cancellation: asyncio.CancelledError | None = None,
    *,
    preserve_requests: int = 0,
) -> asyncio.CancelledError | None:
    """Normalize task state for an explicitly owned cancellation signal."""

    current_task = asyncio.current_task()
    if current_task is None or current_task.cancelling() <= preserve_requests:
        return cancellation
    if cancellation is not None:
        captured = cancellation
    else:
        # asyncio exposes no public cancellation-message accessor. CPython's
        # Task retains the most recent message here even after delivery, which
        # lets cleanup preserve the ordinary CancelledError shape instead of
        # replacing ``CancelledError("reason")`` with an empty instance.
        cancel_message = getattr(current_task, "_cancel_message", None)
        captured = (
            asyncio.CancelledError()
            if cancel_message is None
            else asyncio.CancelledError(cancel_message)
        )
    while current_task.cancelling() > preserve_requests:
        current_task.uncancel()
    return captured


async def await_shielded_task_outcome(
    task: asyncio.Task[_ResultT],
    *,
    cancellation: asyncio.CancelledError | None = None,
    timeout_s: float | None = None,
) -> ShieldedTaskOutcome[_ResultT]:
    """Await a child while retaining caller cancellation and fatal signals."""

    current_task = asyncio.current_task()
    historical_requests = 0 if current_task is None else current_task.cancelling()
    loop = asyncio.get_running_loop()
    deadline = None if timeout_s is None else loop.time() + timeout_s

    def completed_task_outcome() -> ShieldedTaskOutcome[_ResultT]:
        try:
            return ShieldedTaskOutcome(
                result=task.result(),
                cancellation=cancellation,
            )
        except BaseException as child_error:
            return ShieldedTaskOutcome(
                error=child_error,
                cancellation=cancellation,
            )

    def capture_pending_cancellation(
        observed: asyncio.CancelledError | None = None,
        *,
        preserve_requests: int,
    ) -> bool:
        nonlocal cancellation
        if current_task is None or current_task.cancelling() <= preserve_requests:
            return False
        # This helper deliberately suppresses cancellation only long enough to
        # classify the child outcome. Normalize every request now so nested
        # cleanup phases do not rediscover and duplicate the same cancellation;
        # the returned outcome makes its wrapper responsible for redelivery.
        cancellation = consume_pending_task_cancellation(
            cancellation or observed,
            preserve_requests=preserve_requests,
        )
        return True

    if cancellation is not None:
        # The caller has already observed and taken ownership of this signal.
        # One request belongs to the delivered signal; older handled requests
        # remain historical state and must not be claimed by this cleanup.
        preserve_requests = max(historical_requests - 1, 0)
        cancellation = consume_pending_task_cancellation(
            cancellation,
            preserve_requests=preserve_requests,
        )
        historical_requests = preserve_requests

    # Use a real cancellation checkpoint to distinguish a request that has not
    # yet been delivered from Task.cancelling()'s historical absolute count.
    # A previously handled cancellation does not raise here and must not be
    # fabricated into the outcome of an unrelated later operation.
    try:
        await asyncio.sleep(0)
    except asyncio.CancelledError as exc:
        requests_after_delivery = 0 if current_task is None else current_task.cancelling()
        # A higher count proves the delivered request arrived after entry, so
        # the whole entry count is historical. Otherwise the checkpoint
        # delivered one request already present at entry; claim only that one.
        preserve_requests = (
            historical_requests
            if requests_after_delivery > historical_requests
            else max(historical_requests - 1, 0)
        )
        capture_pending_cancellation(exc, preserve_requests=preserve_requests)
        historical_requests = preserve_requests

    # Completion and timeout can race at this checkpoint. Once the child is
    # terminal, observe its result (especially an exception) instead of losing
    # it behind a timeout or emitting "Task exception was never retrieved".
    if task.done():
        return completed_task_outcome()
    if deadline is not None and loop.time() >= deadline:
        return ShieldedTaskOutcome(
            cancellation=cancellation,
            timed_out=True,
        )
    while not task.done():
        try:
            if deadline is None:
                await asyncio.shield(task)
            else:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    return ShieldedTaskOutcome(
                        cancellation=cancellation,
                        timed_out=True,
                    )
                await asyncio.wait_for(asyncio.shield(task), timeout=remaining)
        except asyncio.CancelledError as exc:
            if capture_pending_cancellation(
                exc,
                preserve_requests=historical_requests,
            ):
                continue
            if task.done():
                break
            raise
        except TimeoutError:
            if not task.done():
                return ShieldedTaskOutcome(
                    cancellation=cancellation,
                    timed_out=True,
                )
        except BaseException:
            if not task.done():
                raise
            break
    return completed_task_outcome()
