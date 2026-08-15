"""Caller-cancellation ownership for invocation-scoped extension operations."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from cayu._exception_groups import exception_tree_contains

_ResultT = TypeVar("_ResultT")
_CHILD_CANCELLATION_MESSAGE = "Invocation operation cancelled by caller."


@dataclass(frozen=True, slots=True)
class InvocationOperationOutcome(Generic[_ResultT]):
    """One extension outcome plus independently authenticated caller cancellation.

    ``operation_started=False`` is authoritative proof that the operation factory
    was never invoked. ``True`` is conservative: the operation may have dispatched
    external work even when no result is available.
    """

    operation_started: bool
    result: _ResultT | None = None
    error: BaseException | None = None
    cancellation: asyncio.CancelledError | None = None


@dataclass(slots=True)
class _InvocationOperationState:
    """Boundary-owned proof that the operation factory began executing."""

    started: bool = False


class _RetainedInvocationOperationProbe:
    """Retain one exact child until a later fence can prove it stopped."""

    __slots__ = ("__child",)

    def __init__(
        self,
        child: asyncio.Future[InvocationOperationOutcome[Any]],
    ) -> None:
        self.__child = child
        child.add_done_callback(_observe_retained_invocation_operation)

    async def __call__(self) -> bool:
        child = self.__child
        if not child.done() and child.get_loop() is not asyncio.get_running_loop():
            return False
        if not child.done():
            try:
                await asyncio.shield(child)
            except asyncio.CancelledError:
                raise
            except BaseException:
                if not child.done():
                    raise
        return child.done()

    async def outcome(self) -> InvocationOperationOutcome[Any] | None:
        """Return the exact retained outcome when its owner task settled safely."""

        child = self.__child
        if not child.done() and child.get_loop() is not asyncio.get_running_loop():
            return None
        if not child.done():
            try:
                await asyncio.shield(child)
            except asyncio.CancelledError:
                raise
            except BaseException:
                if not child.done():
                    raise
        if not child.done() or child.cancelled():
            return None
        try:
            outcome = child.result()
        except BaseException:
            return None
        return outcome if type(outcome) is InvocationOperationOutcome else None


def _observe_retained_invocation_operation(
    child: asyncio.Future[InvocationOperationOutcome[Any]],
) -> None:
    """Consume a detached child's task exception without discarding its result."""

    if child.cancelled():
        return
    try:
        child.exception()
    except BaseException:
        return


class InvocationOperationCapacityError(RuntimeError):
    """A bounded operation registry cannot admit another extension call."""


class BoundedInvocationOperationRegistry:
    """Retain and observe a bounded set of cancellation-abandonable operations."""

    def __init__(self, *, max_operations: int) -> None:
        if type(max_operations) is not int:
            raise TypeError("max_operations must be an int.")
        if max_operations <= 0:
            raise ValueError("max_operations must be greater than zero.")
        self._max_operations = max_operations
        self._reservations = 0
        self._operations: set[asyncio.Future[Any]] = set()

    def __len__(self) -> int:
        return len(self._operations) + self._reservations

    def reserve(self) -> bool:
        """Reserve capacity synchronously before an extension can be dispatched."""

        if len(self) >= self._max_operations:
            return False
        self._reservations += 1
        return True

    def release_reservation(self) -> None:
        if self._reservations <= 0:
            raise RuntimeError("Invocation operation registry has no reservation to release.")
        self._reservations -= 1

    def track(self, operation: asyncio.Future[Any]) -> None:
        """Convert one reservation into a retained operation."""

        self.release_reservation()
        self._operations.add(operation)
        operation.add_done_callback(self._operation_done)

    def release(self, operation: asyncio.Future[Any]) -> None:
        """Release one settled operation after its result has been consumed."""

        self._operations.discard(operation)

    def _operation_done(self, operation: asyncio.Future[Any]) -> None:
        self._operations.discard(operation)
        if operation.cancelled():
            return
        # The owned child normally returns an InvocationOperationOutcome even
        # for extension failures. Still consume an unexpected task exception so
        # a detached read cannot produce an unhandled event-loop diagnostic.
        operation.exception()


async def await_invocation_cancellation_checkpoint() -> asyncio.CancelledError | None:
    """Deliver one currently pending caller cancellation without dispatching work."""

    try:
        await asyncio.sleep(0)
    except asyncio.CancelledError as exc:
        return exc
    return None


async def await_invocation_operation(
    operation_factory: Callable[[], Awaitable[_ResultT]],
    *,
    request_child_cancellation: bool = True,
    cancellation: asyncio.CancelledError | None = None,
    abandon_on_caller_cancellation: bool = False,
    operation_registry: BoundedInvocationOperationRegistry | None = None,
    on_unsettled_supervisory_exit: (
        Callable[[_RetainedInvocationOperationProbe], None] | None
    ) = None,
) -> InvocationOperationOutcome[_ResultT]:
    """Run an extension in an owned task without surrendering caller cancellation.

    The caller task observes its own ``Task.cancel()`` deliveries. The child
    receives a separate fixed cancellation request by default so it can perform
    adapter cleanup, but it cannot replace, consume, or forge the caller's
    signal. Mutation owners with cancellation-opaque dependencies pass
    ``request_child_cancellation=False`` and remain shielded until the dispatched
    work positively settles.
    Pending cancellation is checkpointed before dispatch, so an operation that
    has not started cannot run after the caller already requested cancellation.
    A boundary that has already authenticated and retained caller cancellation
    may pass it back through ``cancellation`` to run required settlement work;
    later requests are observed without replacing the original signal.
    Side-effect-free operations may opt into prompt abandonment after caller
    cancellation by supplying ``abandon_on_caller_cancellation=True`` together
    with a bounded ``operation_registry`` and
    ``request_child_cancellation=False``. The child remains retained and
    observed until it naturally settles; cancellation delivery alone cannot
    prove that executor, subprocess, remote, or other opaque work stopped. This
    mode must not be used for a mutation whose settlement owns retry or cleanup
    authority.
    """

    if not callable(operation_factory):
        raise TypeError("Invocation operation factory must be callable.")
    if type(request_child_cancellation) is not bool:
        raise TypeError("request_child_cancellation must be a bool.")
    if type(abandon_on_caller_cancellation) is not bool:
        raise TypeError("abandon_on_caller_cancellation must be a bool.")
    if cancellation is not None and not isinstance(cancellation, asyncio.CancelledError):
        raise TypeError("cancellation must be a CancelledError or None.")
    if on_unsettled_supervisory_exit is not None and not callable(on_unsettled_supervisory_exit):
        raise TypeError("on_unsettled_supervisory_exit must be callable or None.")
    if abandon_on_caller_cancellation != (operation_registry is not None):
        raise ValueError(
            "abandon_on_caller_cancellation requires exactly one bounded operation registry."
        )
    if abandon_on_caller_cancellation and request_child_cancellation:
        raise ValueError(
            "Cancellation-abandonable operations must retain their child until natural settlement."
        )
    current_task = asyncio.current_task()
    observed_requests = 0 if current_task is None else current_task.cancelling()
    resume_after_cancellation = cancellation is not None

    # A real checkpoint authenticates and consumes delivery of a request that
    # was already pending at entry. Historical, previously handled requests do
    # not raise here and remain outside this operation's ownership.
    try:
        await asyncio.sleep(0)
    except asyncio.CancelledError as exc:
        if cancellation is None:
            cancellation = exc
        observed_requests = 0 if current_task is None else current_task.cancelling()

    if cancellation is not None and not resume_after_cancellation:
        return InvocationOperationOutcome(
            operation_started=False,
            cancellation=cancellation,
        )

    child: asyncio.Future[InvocationOperationOutcome[_ResultT]] | None = None
    operation_state = _InvocationOperationState()
    factory_error: BaseException | None = None
    registry_reserved = False
    if operation_registry is not None:
        registry_reserved = operation_registry.reserve()
        if not registry_reserved:
            factory_error = InvocationOperationCapacityError(
                "Invocation operation capacity is exhausted."
            )
    if factory_error is None:
        try:
            child = asyncio.ensure_future(
                _capture_child_operation(operation_factory, operation_state)
            )
            if operation_registry is not None:
                operation_registry.track(child)
                registry_reserved = False
        except BaseException as exc:
            factory_error = exc
        finally:
            if registry_reserved and operation_registry is not None:
                operation_registry.release_reservation()
    del operation_factory

    if child is None:
        # A cross-thread cancellation can become pending while child-task
        # creation is running. Deliver it before returning that failure.
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError as exc:
            if cancellation is None:
                cancellation = exc
        return InvocationOperationOutcome(
            operation_started=False,
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
                if request_child_cancellation:
                    child.cancel(_CHILD_CANCELLATION_MESSAGE)
                if abandon_on_caller_cancellation:
                    return InvocationOperationOutcome(
                        # The retained child can still begin after this caller
                        # stops waiting, so absence of an observed start is not
                        # authoritative no-dispatch evidence here.
                        operation_started=True,
                        cancellation=cancellation,
                    )
                continue
            if child.done():
                break
            raise
        except BaseExceptionGroup as exc:
            if on_unsettled_supervisory_exit is not None and exception_tree_contains(
                exc,
                (KeyboardInterrupt, SystemExit, GeneratorExit),
            ):
                on_unsettled_supervisory_exit(_RetainedInvocationOperationProbe(child))
            del child, on_unsettled_supervisory_exit, operation_state
            raise
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            if on_unsettled_supervisory_exit is not None:
                # A command child that completed concurrently can still carry
                # deferred external cleanup. Let the caller inspect its exact
                # outcome rather than treating task completion as quiescence.
                on_unsettled_supervisory_exit(_RetainedInvocationOperationProbe(child))
            del child, on_unsettled_supervisory_exit, operation_state
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
    except BaseExceptionGroup as exc:
        if on_unsettled_supervisory_exit is not None and exception_tree_contains(
            exc,
            (KeyboardInterrupt, SystemExit, GeneratorExit),
        ):
            on_unsettled_supervisory_exit(_RetainedInvocationOperationProbe(child))
        del child, on_unsettled_supervisory_exit, operation_state
        raise
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        if on_unsettled_supervisory_exit is not None:
            # Even a completed command coroutine may report deferred external
            # cleanup. Transfer its exact outcome before control leaves this
            # final post-completion checkpoint.
            on_unsettled_supervisory_exit(_RetainedInvocationOperationProbe(child))
        del child, on_unsettled_supervisory_exit, operation_state
        raise

    if child.done():
        try:
            child_outcome = child.result()
        except BaseException as exc:
            child_outcome = InvocationOperationOutcome(
                operation_started=operation_state.started,
                error=exc,
            )
    if operation_registry is not None:
        operation_registry.release(child)
    child = None
    return InvocationOperationOutcome(
        operation_started=(
            operation_state.started if child_outcome is None else child_outcome.operation_started
        ),
        result=None if child_outcome is None else child_outcome.result,
        error=None if child_outcome is None else child_outcome.error,
        cancellation=cancellation,
    )


async def _capture_child_operation(
    operation_factory: Callable[[], Awaitable[_ResultT]],
    operation_state: _InvocationOperationState,
) -> InvocationOperationOutcome[_ResultT]:
    """Dispatch and await an extension entirely inside its owned child task."""

    operation: Awaitable[_ResultT] | None = None
    try:
        operation_state.started = True
        operation = operation_factory()
        result = await operation
    except BaseException as error:
        return InvocationOperationOutcome(
            operation_started=True,
            error=error,
        )
    finally:
        del operation_factory, operation, operation_state
    return InvocationOperationOutcome(
        operation_started=True,
        result=result,
    )
