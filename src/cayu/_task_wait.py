from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

_ResultT = TypeVar("_ResultT")


@dataclass(frozen=True)
class ShieldedTaskOutcome(Generic[_ResultT]):
    """One child outcome observed without discarding caller cancellation."""

    result: _ResultT | None = None
    error: BaseException | None = None
    cancellation: asyncio.CancelledError | None = None
    subsequent_cancellation: asyncio.CancelledError | None = None
    cancellation_requests_consumed: int = 0
    timed_out: bool = False


@dataclass(frozen=True)
class CapturedAwaitableOutcome(Generic[_ResultT]):
    """One extension outcome captured before it can escape through task machinery."""

    result: _ResultT | None = None
    error: BaseException | None = None


async def capture_awaitable_outcome(
    operation_factory: Callable[[], Awaitable[_ResultT]],
) -> CapturedAwaitableOutcome[_ResultT]:
    """Return every extension outcome as data, including process-control signals."""

    if not callable(operation_factory):
        raise TypeError("operation_factory must be callable.")
    try:
        return CapturedAwaitableOutcome(result=await operation_factory())
    except BaseException as error:
        return CapturedAwaitableOutcome(error=error)


def unexpected_child_cancellation_error(
    cancellation: asyncio.CancelledError,
    *,
    operation: str,
) -> RuntimeError:
    """Classify child-only cancellation as an operational failure."""

    error = RuntimeError(f"{operation} was cancelled without caller cancellation.")
    error.__cause__ = cancellation
    return error


def consume_pending_task_cancellation(
    cancellation: asyncio.CancelledError | None = None,
    *,
    preserve_requests: int = 0,
) -> asyncio.CancelledError | None:
    """Normalize task state for an explicitly owned cancellation signal."""

    current_task = asyncio.current_task()
    if current_task is None or current_task.cancelling() <= preserve_requests:
        return cancellation
    if cancellation is not None:
        captured = cancellation
    else:
        # asyncio exposes no public cancellation-message accessor. CPython's
        # Task retains the most recent message here even after delivery, which
        # lets cleanup preserve the ordinary CancelledError shape instead of
        # replacing ``CancelledError("reason")`` with an empty instance.
        cancel_message = getattr(current_task, "_cancel_message", None)
        captured = (
            asyncio.CancelledError()
            if cancel_message is None
            else asyncio.CancelledError(cancel_message)
        )
    while current_task.cancelling() > preserve_requests:
        current_task.uncancel()
    return captured


def restore_task_cancellation_requests(
    count: int,
    *,
    cancellation: asyncio.CancelledError | None = None,
) -> None:
    """Restore shield-consumed requests immediately before control escapes.

    When the boundary retained the delivered cancellation, preserve its single
    task-owned message on every restored request.  Reissuing a bare request can
    otherwise replace the authoritative ``CancelledError`` at the next await.
    """

    if type(count) is not int or count < 0:
        raise ValueError("Consumed cancellation request count must be a non-negative int.")
    if cancellation is not None and not isinstance(cancellation, asyncio.CancelledError):
        raise TypeError("Restored cancellation must be a CancelledError.")
    cancellation_message = None
    has_cancellation_message = False
    if cancellation is not None:
        cancellation_args = BaseException.__dict__["args"].__get__(cancellation, BaseException)
        if type(cancellation_args) is tuple and len(cancellation_args) == 1:
            cancellation_message = cancellation_args[0]
            has_cancellation_message = True
    current_task = asyncio.current_task()
    if current_task is not None:
        for _request in range(count):
            if has_cancellation_message:
                current_task.cancel(cancellation_message)
            else:
                current_task.cancel()


async def await_shielded_task_outcome(
    task: asyncio.Future[_ResultT],
    *,
    cancellation: asyncio.CancelledError | None = None,
    timeout_s: float | None = None,
    timeout_after_cancellation_s: float | None = None,
) -> ShieldedTaskOutcome[_ResultT]:
    """Await a child while retaining caller cancellation and fatal signals."""

    current_task = asyncio.current_task()
    historical_requests = 0 if current_task is None else current_task.cancelling()
    cancellation_requests_consumed = 0
    subsequent_cancellation: asyncio.CancelledError | None = None
    loop = asyncio.get_running_loop()
    deadline = None if timeout_s is None else loop.time() + timeout_s
    if cancellation is not None and timeout_after_cancellation_s is not None:
        cancellation_deadline = loop.time() + timeout_after_cancellation_s
        deadline = (
            cancellation_deadline if deadline is None else min(deadline, cancellation_deadline)
        )

    def completed_task_outcome() -> ShieldedTaskOutcome[_ResultT]:
        try:
            return ShieldedTaskOutcome(
                result=task.result(),
                cancellation=cancellation,
                subsequent_cancellation=subsequent_cancellation,
                cancellation_requests_consumed=cancellation_requests_consumed,
            )
        except BaseException as child_error:
            return ShieldedTaskOutcome(
                error=child_error,
                cancellation=cancellation,
                subsequent_cancellation=subsequent_cancellation,
                cancellation_requests_consumed=cancellation_requests_consumed,
            )

    def capture_pending_cancellation(
        observed: asyncio.CancelledError | None = None,
        *,
        preserve_requests: int,
    ) -> bool:
        nonlocal cancellation, subsequent_cancellation
        nonlocal cancellation_requests_consumed, deadline
        if current_task is None or current_task.cancelling() <= preserve_requests:
            return False
        # This helper deliberately suppresses cancellation only long enough to
        # classify the child outcome. Normalize every request now so nested
        # cleanup phases do not rediscover and duplicate the same cancellation;
        # the returned outcome makes its wrapper responsible for redelivery.
        requests_before_consumption = current_task.cancelling()
        # The first delivered caller signal stays authoritative across repeated
        # cancellation. Retain one later delivery separately so a boundary with
        # its own structurally identified cancellation (for example a tool
        # deadline) can exclude that internal request without replacing the
        # first caller signal globally.
        if cancellation is None:
            cancellation = observed
        elif observed is not None and subsequent_cancellation is None:
            subsequent_cancellation = observed
        cancellation = consume_pending_task_cancellation(
            cancellation,
            preserve_requests=preserve_requests,
        )
        cancellation_requests_consumed += max(
            requests_before_consumption - current_task.cancelling(),
            0,
        )
        if timeout_after_cancellation_s is not None:
            cancellation_deadline = loop.time() + timeout_after_cancellation_s
            deadline = (
                cancellation_deadline if deadline is None else min(deadline, cancellation_deadline)
            )
        return True

    if cancellation is not None:
        # The caller has already observed and taken ownership of this signal.
        # One request belongs to the delivered signal; older handled requests
        # remain historical state and must not be claimed by this cleanup.
        preserve_requests = max(historical_requests - 1, 0)
        requests_before_consumption = historical_requests
        cancellation = consume_pending_task_cancellation(
            cancellation,
            preserve_requests=preserve_requests,
        )
        if current_task is not None:
            cancellation_requests_consumed += max(
                requests_before_consumption - current_task.cancelling(),
                0,
            )
        historical_requests = preserve_requests

    # Use a real cancellation checkpoint to distinguish a request that has not
    # yet been delivered from Task.cancelling()'s historical absolute count.
    # A previously handled cancellation does not raise here and must not be
    # fabricated into the outcome of an unrelated later operation.
    try:
        await asyncio.sleep(0)
    except asyncio.CancelledError as exc:
        requests_after_delivery = 0 if current_task is None else current_task.cancelling()
        # A higher count proves the delivered request arrived after entry, so
        # the whole entry count is historical. Otherwise the checkpoint
        # delivered one request already present at entry; claim only that one.
        preserve_requests = (
            historical_requests
            if requests_after_delivery > historical_requests
            else max(historical_requests - 1, 0)
        )
        capture_pending_cancellation(exc, preserve_requests=preserve_requests)
        historical_requests = preserve_requests

    # Completion and timeout can race at this checkpoint. Once the child is
    # terminal, observe its result (especially an exception) instead of losing
    # it behind a timeout or emitting "Task exception was never retrieved".
    if task.done():
        return completed_task_outcome()
    if deadline is not None and loop.time() >= deadline:
        return ShieldedTaskOutcome(
            cancellation=cancellation,
            subsequent_cancellation=subsequent_cancellation,
            cancellation_requests_consumed=cancellation_requests_consumed,
            timed_out=True,
        )
    while not task.done():
        try:
            if deadline is None:
                await asyncio.shield(task)
            else:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    return ShieldedTaskOutcome(
                        cancellation=cancellation,
                        subsequent_cancellation=subsequent_cancellation,
                        cancellation_requests_consumed=cancellation_requests_consumed,
                        timed_out=True,
                    )
                await asyncio.wait_for(asyncio.shield(task), timeout=remaining)
        except asyncio.CancelledError as exc:
            if capture_pending_cancellation(
                exc,
                preserve_requests=historical_requests,
            ):
                continue
            if task.done():
                break
            raise
        except TimeoutError:
            if not task.done():
                return ShieldedTaskOutcome(
                    cancellation=cancellation,
                    subsequent_cancellation=subsequent_cancellation,
                    cancellation_requests_consumed=cancellation_requests_consumed,
                    timed_out=True,
                )
        except BaseException:
            if not task.done():
                raise
            break
    return completed_task_outcome()


def _consume_detached_task_outcome(task: asyncio.Task[Any]) -> None:
    """Retrieve a timed-out task's eventual result after requesting cancellation."""

    with contextlib.suppress(asyncio.CancelledError):
        task.exception()


async def run_thread_to_completion(
    func: Callable[..., _ResultT], /, *args: Any, **kwargs: Any
) -> _ResultT:
    """Run ``func`` in a thread; if cancelled, wait for the thread before re-raising.

    A thread cannot be stopped, so the awaiting task stays alive until the work
    it dispatched has finished: owners that wait for that task, such as
    application shutdown, then also wait for the thread. The result of a
    cancelled call is discarded and the caller's cancellation is re-raised.
    """

    thread_task = asyncio.ensure_future(asyncio.to_thread(func, *args, **kwargs))
    outcome = await await_shielded_task_outcome(thread_task)
    if outcome.cancellation is not None:
        restore_task_cancellation_requests(
            outcome.cancellation_requests_consumed,
            cancellation=outcome.cancellation,
        )
        raise outcome.cancellation
    if outcome.error is not None:
        raise outcome.error
    return cast("_ResultT", outcome.result)


async def wait_until_idle(
    running: Callable[[], set[asyncio.Future[Any]]], *, timeout_s: float
) -> bool:
    """Wait up to ``timeout_s`` until ``running()`` is empty, without cancelling.

    ``running()`` is re-read after each completion, so work started meanwhile is
    also waited for. Returns False if work is still running at the deadline.
    """

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while pending := {future for future in running() if not future.done()}:
        remaining = deadline - loop.time()
        if remaining <= 0:
            return False
        await asyncio.wait(pending, timeout=remaining, return_when=asyncio.FIRST_COMPLETED)
    return True


class RetainedWorkFailed(RuntimeError):
    """Work kept past its caller failed; only the failures' types are reported."""


def retained_task_failure(task: asyncio.Task[Any]) -> BaseException | None:
    """Return how a retained task failed after its caller stopped waiting.

    A cancellation, including the one that detached the task, is not a failure.
    """

    if task.cancelled():
        return None
    error = task.exception()
    if error is None or isinstance(error, asyncio.CancelledError):
        return None
    return error


class LateFailures:
    """Failures of retained work that no caller observed, each reported once."""

    def __init__(self, work: str) -> None:
        self._work = work
        self._failures: dict[str, int] = {}

    @property
    def pending(self) -> bool:
        return bool(self._failures)

    def record(self, failure: str) -> None:
        """Record a failure by its type or status name, never its content."""

        self._failures[failure] = self._failures.get(failure, 0) + 1

    def raise_once(self) -> None:
        if not self._failures:
            return
        failures, self._failures = self._failures, {}
        summary = ", ".join(
            name if count == 1 else f"{name} x{count}" for name, count in failures.items()
        )
        raise RetainedWorkFailed(f"{self._work} failed after its caller left ({summary}).")


def combine_drain_results(results: Sequence[bool | BaseException]) -> bool:
    """Report every failure of concurrent drains; otherwise whether all settled."""

    failures = [result for result in results if isinstance(result, BaseException)]
    if len(failures) == 1:
        raise failures[0]
    if failures:
        raise BaseExceptionGroup("Retained work drains failed.", failures)
    return all(results)


def collect_failures(raisers: Iterable[Callable[[], None]]) -> list[BaseException]:
    """Run each owner's synchronous failure report and return what it raised.

    Combined drains call this only after every wait finished, with no await in
    between, so a cancelled drain consumes no owner's failures.
    """

    failures: list[BaseException] = []
    for raise_failures in raisers:
        try:
            raise_failures()
        except BaseException as failure:
            failures.append(failure)
    return failures
