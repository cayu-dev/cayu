"""Process-local execution capacity shared by concurrent evaluation runs."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from threading import Lock

DEFAULT_EVAL_MAX_ACTIVE_TRIALS = 100
# Durable per-run concurrency crosses PostgreSQL INTEGER, SQLite INTEGER, and
# browser JSON number boundaries. Keep one explicit maximum that every public
# model and supported store can represent exactly. This is a representation
# limit, not the default or an operational recommendation.
EVAL_MAX_CONCURRENCY = 2**31 - 1


class EvalExecutionCapacity:
    """Bound aggregate active eval trials across cooperating run coordinators.

    One instance is one process-local capacity domain. Applications that launch
    multiple evaluation runs or coordinators concurrently share the same instance.
    Per-run ``max_concurrency`` remains a separate, narrower dispatch control.

    Capacity is intentionally operator-configurable and has no Runtime-defined
    upper ceiling. Concurrent use is confined to one event loop; after all
    active and waiting trials settle, the same object may be reused by a new
    loop during an application restart.
    """

    __slots__ = (
        "_active_trials",
        "_event_loop",
        "_max_active_trials",
        "_peak_active_trials",
        "_semaphore",
        "_state_lock",
        "_waiting_trials",
    )

    def __init__(self, max_active_trials: int = DEFAULT_EVAL_MAX_ACTIVE_TRIALS) -> None:
        if type(max_active_trials) is not int:
            raise TypeError("max_active_trials must be an int.")
        if max_active_trials < 1:
            raise ValueError("max_active_trials must be >= 1.")
        self._max_active_trials = max_active_trials
        self._semaphore: asyncio.Semaphore | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._active_trials = 0
        self._peak_active_trials = 0
        self._waiting_trials = 0
        self._state_lock = Lock()

    @property
    def max_active_trials(self) -> int:
        return self._max_active_trials

    @property
    def active_trials(self) -> int:
        with self._state_lock:
            return self._active_trials

    @property
    def peak_active_trials(self) -> int:
        with self._state_lock:
            return self._peak_active_trials

    def _semaphore_for_current_loop(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        with self._state_lock:
            if self._event_loop is None:
                self._event_loop = loop
                self._semaphore = asyncio.Semaphore(self._max_active_trials)
            elif self._event_loop is not loop:
                if self._active_trials != 0 or self._waiting_trials != 0:
                    raise RuntimeError(
                        "EvalExecutionCapacity cannot be shared concurrently across "
                        "multiple event loops."
                    )
                self._event_loop = loop
                self._semaphore = asyncio.Semaphore(self._max_active_trials)
            assert self._semaphore is not None
            self._waiting_trials += 1
            return self._semaphore

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Wait for one shared trial slot and release it on every exit path."""

        semaphore = self._semaphore_for_current_loop()
        try:
            await semaphore.acquire()
        except BaseException:
            with self._state_lock:
                self._waiting_trials -= 1
            raise
        with self._state_lock:
            self._waiting_trials -= 1
            self._active_trials += 1
            self._peak_active_trials = max(self._peak_active_trials, self._active_trials)
        try:
            yield
        finally:
            with self._state_lock:
                self._active_trials -= 1
            semaphore.release()


__all__ = [
    "DEFAULT_EVAL_MAX_ACTIVE_TRIALS",
    "EVAL_MAX_CONCURRENCY",
    "EvalExecutionCapacity",
]
