"""Finite ownership of actual guest channel tasks through their cleanup."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from cayu.runtime.browser_control import BrowserControlConflict


class BrowserGuestChannelSettlement:
    """Private route-owned evidence, independent of the propagated primary error."""

    def __init__(self) -> None:
        self._complete = False

    @property
    def complete(self) -> bool:
        return self._complete

    def confirm(self) -> None:
        """Call only after durable fencing and socket cleanup both succeed."""
        self._complete = True


class BrowserGuestChannels:
    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[None]] = set()
        self._failed_cancellations: set[asyncio.Task[None]] = set()
        self._settlements: dict[asyncio.Task[None], BrowserGuestChannelSettlement] = {}
        self._closed = False

    async def run(self, serve: Callable[[BrowserGuestChannelSettlement], Awaitable[None]]) -> None:
        if self._closed or len(self._tasks) >= 32:
            raise BrowserControlConflict("Browser guest channel capacity is unavailable.")

        settlement = BrowserGuestChannelSettlement()

        async def owned() -> None:
            try:
                await serve(settlement)
            except asyncio.CancelledError as failure:
                if failure.__cause__ is not None:
                    self._failed_cancellations.add(task)
                raise

        task = asyncio.create_task(owned(), name="cayu-browser-guest-channel")
        self._tasks.add(task)
        self._settlements[task] = settlement
        try:
            await task
        finally:
            # Awaiting the task includes its durable fence and socket cleanup.
            # Primary failure alone is not failed cleanup. Retain failures unless
            # the exact route owner positively confirmed fencing and closure.
            if self._settled(task):
                self._tasks.discard(task)
                self._failed_cancellations.discard(task)
                self._settlements.pop(task, None)

    def _settled(self, task: asyncio.Task[None]) -> bool:
        settlement = self._settlements.get(task)
        return task.done() and (
            (settlement is not None and settlement.complete)
            or (
                task not in self._failed_cancellations
                and (task.cancelled() or task.exception() is None)
            )
        )

    async def drain(self, *, timeout_s: float = 5.0) -> bool:
        if type(timeout_s) not in {int, float} or not 0 < timeout_s <= 30:
            raise ValueError("Browser channel drain timeout must be positive and bounded.")
        self._closed = True
        pending = {task for task in self._tasks if not task.done()}
        for task in pending:
            if not task.cancelling():
                task.cancel("browser control service shutdown")
        if pending:
            _, pending = await asyncio.wait(pending, timeout=timeout_s)
        for task in tuple(self._tasks):
            if self._settled(task):
                self._tasks.discard(task)
                self._failed_cancellations.discard(task)
                self._settlements.pop(task, None)
        return not pending and not self._tasks
