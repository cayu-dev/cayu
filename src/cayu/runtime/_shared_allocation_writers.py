"""Turns that a workflow run's steps and commands take on the sandbox they share.

``StepRunOptions(serialize_writers=True)`` and ``command(...,
serialize_writers=True)`` make a step or command wait for, and then hold, its
workflow run's writer turn: a step for its whole child session and the
copy-back after it, a command for its attach and its process. Only steps and
commands that ask for a turn take one; anything else in the run still runs
alongside them, as parallel steps always have.

The turns are process-local. They live in this process's memory, keyed by the
workflow run id, so they serialize what this process runs and nothing else. A
crash drops every turn with the process; a run resumed in another process
starts with free turns there, which is safe because the crashed process's
steps are gone. A superseded attempt still running in another process isn't
held back by them.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

__all__ = ["SharedAllocationWriterTurns"]


@dataclass(eq=False)
class _RunTurns:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    holder: str | None = None
    users: int = 0


class SharedAllocationWriterTurns:
    """Hand out one workflow run's writer turn, first come first served."""

    def __init__(self) -> None:
        self._runs: dict[str, _RunTurns] = {}

    @contextlib.asynccontextmanager
    async def turn(self, run_id: str, *, holder: str) -> AsyncIterator[None]:
        """Wait for ``run_id``'s turn, then hold it until the block exits.

        Cancelled while waiting, the caller leaves the queue without the turn.
        Leaving the block for any reason, cancellation included, gives it up.
        """

        turns = self._runs.get(run_id)
        if turns is None:
            turns = self._runs[run_id] = _RunTurns()
        turns.users += 1
        try:
            async with turns.lock:
                turns.holder = holder
                try:
                    yield
                finally:
                    turns.holder = None
        finally:
            turns.users -= 1
            if turns.users == 0 and self._runs.get(run_id) is turns:
                del self._runs[run_id]

    def holder(self, run_id: str) -> str | None:
        """Who holds ``run_id``'s turn now, if anyone (a step or command id)."""

        turns = self._runs.get(run_id)
        return None if turns is None else turns.holder

    def waiting(self, run_id: str) -> int:
        """How many steps and commands hold or wait for ``run_id``'s turn."""

        turns = self._runs.get(run_id)
        return 0 if turns is None else turns.users
