"""Caller-cancellation ownership for invocation-scoped extension operations."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

_ResultT = TypeVar("_ResultT")
_CHILD_CANCELLATION_MESSAGE = "Invocation operation cancelled by caller."


@dataclass(frozen=True, slots=True)
class InvocationOperationOutcome(Generic[_ResultT]):
    """One extension outcome plus independently authenticated caller cancellation."""

    result: _ResultT | None = None
    error: BaseException | None = None
    cancellation: asyncio.CancelledError | None = None


async def await_invocation_operation(
    operation_factory: Callable[[], Awaitable[_ResultT]],
) -> InvocationOperationOutcome[_ResultT]:
    """Run an extension in an owned task without surrendering caller cancellation.

    The caller task observes its own ``Task.cancel()`` deliveries. The child
    receives a separate fixed cancellation request so it can perform adapter
    cleanup, but it cannot replace, consume, or forge the caller's signal.
    Pending cancellation is checkpointed before dispatch, so an operation that
    has not started cannot run after the caller already requested cancellation.
    """

    if not callable(operation_factory):
        raise TypeError("Invocation operation factory must be callable.")
    current_task = asyncio.current_task()
    observed_requests = 0 if current_task is None else current_task.cancelling()
    cancellation: asyncio.CancelledError | None = None

    # A real checkpoint authenticates and consumes delivery of a request that
    # was already pending at entry. Historical, previously handled requests do
    # not raise here and remain outside this operation's ownership.
    try:
        await asyncio.sleep(0)
    except asyncio.CancelledError as exc:
        cancellation = exc
        observed_requests = 0 if current_task is None else current_task.cancelling()

    if cancellation is not None:
        return InvocationOperationOutcome(cancellation=cancellation)

    child: asyncio.Future[InvocationOperationOutcome[_ResultT]] | None = None
    factory_error: BaseException | None = None
    try:
        child = asyncio.ensure_future(_capture_child_operation(operation_factory))
    except BaseException as exc:
        factory_error = exc
    finally:
        del operation_factory

    if child is None:
        # A cross-thread cancellation can become pending while child-task
        # creation is running. Deliver it before returning that failure.
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError as exc:
            cancellation = exc
        return InvocationOperationOutcome(
            error=factory_error,
            cancellation=cancellation,
        )

    child_outcome: InvocationOperationOutcome[_ResultT] | None = None
    while not child.done():
        try:
            child_outcome = await asyncio.shield(child)
        except asyncio.CancelledError as exc:
            request_count = 0 if current_task is None else current_task.cancelling()
            if request_count > observed_requests:
                observed_requests = request_count
                if cancellation is None:
                    cancellation = exc
                child.cancel(_CHILD_CANCELLATION_MESSAGE)
                continue
            if child.done():
                break
            raise

    # Child completion and caller cancellation can become ready in the same
    # loop turn. Give the caller task one final real delivery point before
    # publishing the terminal child outcome; otherwise a just-arrived request
    # could be deferred until after a successful result escapes this boundary.
    try:
        await asyncio.sleep(0)
    except asyncio.CancelledError as exc:
        request_count = 0 if current_task is None else current_task.cancelling()
        if request_count > observed_requests:
            observed_requests = request_count
            if cancellation is None:
                cancellation = exc
        else:
            raise

    if child.done():
        try:
            child_outcome = child.result()
        except BaseException as exc:
            child_outcome = InvocationOperationOutcome(error=exc)
    child = None
    return InvocationOperationOutcome(
        result=None if child_outcome is None else child_outcome.result,
        error=None if child_outcome is None else child_outcome.error,
        cancellation=cancellation,
    )


async def _capture_child_operation(
    operation_factory: Callable[[], Awaitable[_ResultT]],
) -> InvocationOperationOutcome[_ResultT]:
    """Dispatch and await an extension entirely inside its owned child task."""

    operation: Awaitable[_ResultT] | None = None
    try:
        operation = operation_factory()
        result = await operation
    except BaseException as error:
        return InvocationOperationOutcome(error=error)
    finally:
        del operation_factory, operation
    return InvocationOperationOutcome(result=result)
