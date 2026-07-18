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


def consume_pending_task_cancellation(
    cancellation: asyncio.CancelledError | None = None,
) -> asyncio.CancelledError | None:
    """Capture and normalize cancellation that the current task will redeliver."""

    current_task = asyncio.current_task()
    if current_task is None or current_task.cancelling() <= 0:
        return cancellation
    captured = cancellation or asyncio.CancelledError()
    while current_task.cancelling() > 0:
        current_task.uncancel()
    return captured


async def await_shielded_task_outcome(
    task: asyncio.Task[_ResultT],
    *,
    cancellation: asyncio.CancelledError | None = None,
    timeout_s: float | None = None,
) -> ShieldedTaskOutcome[_ResultT]:
    """Await a child while retaining caller cancellation and fatal signals."""

    def capture_pending_cancellation(
        observed: asyncio.CancelledError | None = None,
    ) -> None:
        nonlocal cancellation
        current_task = asyncio.current_task()
        if current_task is None or current_task.cancelling() <= 0:
            return
        # This helper deliberately suppresses cancellation only long enough to
        # classify the child outcome. Normalize every request now so nested
        # cleanup phases do not rediscover and duplicate the same cancellation;
        # the returned outcome makes its wrapper responsible for redelivery.
        cancellation = consume_pending_task_cancellation(cancellation or observed)

    capture_pending_cancellation()
    deadline = None if timeout_s is None else asyncio.get_running_loop().time() + timeout_s
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
            capture_pending_cancellation(exc)
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
    capture_pending_cancellation()
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
