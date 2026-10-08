"""Preserve caller cancellation while completed tool results are published."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable
from contextlib import suppress
from typing import Never, TypeVar

from cayu._exception_groups import (
    exception_cause,
    iter_exception_tree,
    set_exception_cause,
)
from cayu.runners._cleanup import (
    attach_runner_cancellation_failure,
)

_PostToolResultT = TypeVar("_PostToolResultT")


def _contains_process_signal(error: BaseException | None) -> bool:
    if error is None:
        return False
    return any(
        isinstance(candidate, (GeneratorExit, KeyboardInterrupt, SystemExit))
        for candidate in iter_exception_tree(error)
    )


def _raise_preserved_post_tool_cancellation(
    cancellation: asyncio.CancelledError | None,
    failure: BaseException,
    *,
    restore_cancellation_requests: int,
) -> Never:
    """Keep an earlier caller cancellation authoritative over terminalization failure."""

    if cancellation is None or type(failure) is GeneratorExit or _contains_process_signal(failure):
        raise failure
    # Terminalization crosses stores, hooks, projections, and other extension
    # seams. Preserve the fact of the secondary failure without retaining its
    # potentially workload-derived message, traceback, or mutable state on the
    # public cancellation object.
    safe_failure = RuntimeError("Tool terminalization failed after caller cancellation.")
    del failure
    prior_cause = exception_cause(cancellation)
    if prior_cause is None:
        cause: BaseException = safe_failure
    else:
        cause = BaseExceptionGroup(
            "Post-tool cancellation and terminalization failures.",
            [prior_cause, safe_failure],
        )
    attach_runner_cancellation_failure(cancellation, cause)
    set_exception_cause(cancellation, cause)
    _raise_restored_post_tool_cancellation(
        cancellation,
        restore_cancellation_requests=restore_cancellation_requests,
        cause=cause,
    )


async def _receive_restored_post_tool_cancellation() -> None:
    """Observe restored control before publishing the durable tool outcome."""

    current_task = asyncio.current_task()
    if current_task is None or not current_task.cancelling():
        return
    # Observation and tool helpers restore requests before raising their owned
    # cancellation. On Python 3.11/3.12, uncancel() cannot rescind the queued
    # injection. Receive it before another publication await, while retaining
    # the original exception, its evidence, and every task cancellation count.
    with suppress(asyncio.CancelledError):
        await asyncio.sleep(0)


def _raise_restored_post_tool_cancellation(
    cancellation: asyncio.CancelledError,
    *,
    restore_cancellation_requests: int,
    cause: BaseException | None = None,
) -> Never:
    """Redeliver owned cancellation without erasing Task.cancelling() evidence."""

    current_task = asyncio.current_task()
    if current_task is not None:
        for _request in range(restore_cancellation_requests):
            current_task.cancel()
    raise cancellation from cause


async def _await_post_tool_operation(
    operation: Awaitable[_PostToolResultT],
    *,
    cancellation: asyncio.CancelledError | None,
    restore_cancellation_requests: int,
) -> _PostToolResultT:
    """Await terminalization without allowing it to replace owned cancellation."""

    if cancellation is None:
        return await operation
    try:
        return await operation
    except BaseException as failure:
        _raise_preserved_post_tool_cancellation(
            cancellation,
            failure,
            restore_cancellation_requests=restore_cancellation_requests,
        )


async def _iterate_post_tool_events(
    events: AsyncIterator[_PostToolResultT],
    *,
    cancellation: asyncio.CancelledError | None,
    restore_cancellation_requests: int,
) -> AsyncIterator[_PostToolResultT]:
    """Iterate terminalization events under the same cancellation authority."""

    if cancellation is None:
        async for event in events:
            yield event
        return
    try:
        async for event in events:
            yield event
    except BaseException as failure:
        _raise_preserved_post_tool_cancellation(
            cancellation,
            failure,
            restore_cancellation_requests=restore_cancellation_requests,
        )
