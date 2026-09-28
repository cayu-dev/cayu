"""Shared ownership of off-thread SQLite operations and caller cancellation."""

from __future__ import annotations

import asyncio
import contextvars
import sqlite3
from collections.abc import Callable
from concurrent.futures import Executor
from typing import TypeVar, cast

_T = TypeVar("_T")


async def _run_off_thread_with_connection_ownership(
    lock: asyncio.Lock,
    connection: sqlite3.Connection,
    operation: Callable[[sqlite3.Connection], _T],
    *,
    executor: Executor | None = None,
    worker_started: asyncio.Event | None = None,
    interrupt_on_cancellation: bool = False,
) -> _T:
    """Keep a SQLite connection owned until its off-thread operation terminates.

    Cancelling an ``asyncio.to_thread`` await does not stop the worker thread.
    For an interruptible read, request ``sqlite3_interrupt()`` after cancellation;
    in every case defer the signal while holding the connection lock so no
    subsequent operation or shutdown can reuse the connection before the worker
    has left it in a terminal transaction state.
    """

    if type(interrupt_on_cancellation) is not bool:
        raise TypeError("interrupt_on_cancellation must be a bool.")

    async with lock:

        def capture_outcome() -> tuple[bool, object]:
            try:
                return True, operation(connection)
            except BaseException as worker_failure:
                # The executor future must complete normally even when the
                # operation raises CancelledError. That makes every cancellation
                # from shield() unambiguously caller-owned and keeps ownership
                # tied to the executor's physical completion.
                return False, worker_failure

        loop = asyncio.get_running_loop()
        context = contextvars.copy_context()
        worker = loop.run_in_executor(executor, context.run, capture_outcome)
        if worker_started is not None:
            worker_started.set()
        cancellation: asyncio.CancelledError | None = None

        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError as exc:
                if cancellation is None:
                    cancellation = exc
                    if interrupt_on_cancellation:
                        connection.interrupt()
            except BaseException:
                if worker.done():
                    break
                raise

        succeeded, outcome = worker.result()
        if not succeeded:
            if not isinstance(outcome, BaseException):
                raise RuntimeError("SQLite worker returned an invalid failure outcome.")
            if cancellation is None:
                raise outcome
            cancellation.add_note(
                "SQLite worker failed while caller cancellation was pending: "
                f"{type(outcome).__name__}: {outcome}"
            )
            raise cancellation from outcome
        if cancellation is not None:
            raise cancellation
        return cast("_T", outcome)
