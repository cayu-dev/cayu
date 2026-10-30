"""Bound a model step in flight by the run's ``max_elapsed_seconds``."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from typing import TypeVar

_T = TypeVar("_T")


class ElapsedLimitExpired(Exception):
    """The run reached ``max_elapsed_seconds`` while the bounded work was in flight.

    The work saw an ordinary cancellation and settled it before this was raised.
    """


async def elapsed_limit_stream(
    stream: AsyncIterator[_T],
    seconds_left: Callable[[], float | None],
    *,
    settled: Callable[[_T], bool],
) -> AsyncGenerator[_T, None]:
    """Cancel the producer when the run's time runs out while it is being driven.

    The timer covers only advancing the producer, never a yield to the caller,
    so it can't cancel the consumer's own work. ``seconds_left`` is read before
    each advance; ``None`` means no elapsed-time limit. Once ``settled`` accepts
    an item, the remaining items are local bookkeeping and run without a timer.

    A producer may catch the cut, yield its settlement (budget settlement, for
    example) and only then re-raise it. Those items pass through untimed, and
    the re-raised cancellation still ends as ``ElapsedLimitExpired``. A
    cancellation someone else also requested keeps its identity.
    """

    iterator = aiter(stream)
    task = asyncio.current_task()
    if task is None:
        raise RuntimeError("An elapsed-time limit requires an asyncio task.")
    armed = True
    cut = False
    try:
        while True:
            remaining = seconds_left() if armed and not cut else None
            cancellation_baseline = task.cancelling()
            timer = None if remaining is None else asyncio.timeout(max(remaining, 0.0))
            try:
                if timer is None:
                    item = await anext(iterator)
                else:
                    async with timer:
                        item = await anext(iterator)
            except StopAsyncIteration:
                return
            except (TimeoutError, asyncio.CancelledError) as stopped:
                ours = cut or (timer is not None and timer.expired())
                if ours and task.cancelling() <= cancellation_baseline:
                    raise ElapsedLimitExpired() from stopped
                raise
            if timer is not None and timer.expired():
                # The producer caught the cut and is settling it.
                cut = True
            if armed and settled(item):
                armed = False
            yield item
    finally:
        close = getattr(iterator, "aclose", None)
        if close is not None:
            await close()
