"""Tell a running task handler that its task was asked to stop."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar


class TaskCancellationRequested(RuntimeError):
    """A new top-level ``app.run`` was refused because its task was asked to stop."""


class TaskCancellationSignal:
    """Whether the task a handler is running has been asked to stop.

    ``run_task_worker`` binds one around every handler call and requests it as
    soon as its heartbeat sees a cancellation request for the task (for
    example from ``cayu task cancel``). Handler code reads it with
    :func:`current_task_cancellation`: ``requested`` is true from then on and
    ``wait()`` returns, so work that is not an ``app.run`` (a subprocess, a
    thread, an external call) can stop cooperatively. While a requested signal
    is bound, a new top-level ``app.run`` is refused with
    :class:`TaskCancellationRequested`.

    A hand-written worker can create one, ``bind()`` it around its own handler
    and call ``request()`` when it learns the task should stop.
    """

    def __init__(self) -> None:
        self._requested = asyncio.Event()

    @property
    def requested(self) -> bool:
        return self._requested.is_set()

    def request(self) -> None:
        """Mark the task as asked to stop. Calling it again does nothing."""
        self._requested.set()

    async def wait(self) -> None:
        """Return once the task has been asked to stop."""
        await self._requested.wait()

    @contextmanager
    def bind(self) -> Iterator[TaskCancellationSignal]:
        """Make this the signal :func:`current_task_cancellation` returns inside the block."""
        token = _CURRENT.set((self, _CURRENT.get()))
        try:
            yield self
        finally:
            _CURRENT.reset(token)


_Binding = tuple[TaskCancellationSignal, "_Binding | None"]
_CURRENT: ContextVar[_Binding | None] = ContextVar("cayu_task_cancellation", default=None)


def current_task_cancellation() -> TaskCancellationSignal | None:
    """The cancellation signal of the task handler running this code, if any."""
    binding = _CURRENT.get()
    return None if binding is None else binding[0]


def refuse_run_after_task_cancellation() -> None:
    """Refuse a new top-level run once any enclosing task was asked to stop."""
    binding = _CURRENT.get()
    while binding is not None:
        signal, binding = binding
        if signal.requested:
            raise TaskCancellationRequested(
                "The task running this handler was asked to stop; new runs are refused."
            )
