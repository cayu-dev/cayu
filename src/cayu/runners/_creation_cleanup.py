"""Process-local ownership of rollback after a positively acknowledged allocation."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from weakref import WeakKeyDictionary

from cayu._task_wait import (
    CapturedAwaitableOutcome,
    await_shielded_task_outcome,
    capture_awaitable_outcome,
    restore_task_cancellation_requests,
)
from cayu.runners._cleanup import (
    RunnerFailureProgress,
    attach_runner_cancellation_failure,
    validate_cancel_timeout,
)
from cayu.runners.base import _contains_runner_fatal_signal

_OWNERS: set[_CreationCleanup] = set()
_ACQUISITIONS: dict[tuple[str, str], CreationLease] = {}
_RESTORATION_RETRIES: WeakKeyDictionary[asyncio.Task[None], Callable[[], Awaitable[None]]] = (
    WeakKeyDictionary()
)


def register_acquisition_restoration_retry(
    task: asyncio.Task[None], operation: Callable[[], Awaitable[None]]
) -> None:
    """Retain the producer's exact restoration operation, never a new allocation."""
    _RESTORATION_RETRIES[task] = operation


@dataclass(eq=False)
class CreationLease:
    key: tuple[str, str]
    retained: bool = False
    settlement_task: asyncio.Task[None] | None = None
    retry_task: asyncio.Task[None] | None = None
    restoration: Callable[[], Awaitable[None]] | None = None

    def __enter__(self) -> CreationLease:
        return self

    def __exit__(self, *args: object) -> None:
        if not self.retained:
            self.release()

    def release(self) -> None:
        if _ACQUISITIONS.get(self.key) is self:
            del _ACQUISITIONS[self.key]

    def retain_until(self, task: asyncio.Task[None]) -> None:
        self.retained = True
        self.settlement_task = task
        if (operation := _RESTORATION_RETRIES.get(task)) is not None:
            self.restoration = operation

        def settled(completed: asyncio.Task[None]) -> None:
            if (
                self.settlement_task is completed
                and not completed.cancelled()
                and completed.exception() is None
            ):
                self.release()

        task.add_done_callback(settled)


def acquire_creation_lease(adapter: str, identity: str) -> CreationLease:
    key = (adapter, identity)
    lease = CreationLease(key)
    existing = _ACQUISITIONS.setdefault(key, lease)
    if existing is not lease and existing.retry_task is asyncio.current_task():
        return existing
    if existing is not lease:
        raise RuntimeError("Runner acquisition or rollback is still pending for this resource.")
    return lease


def require_creation_cleanup_settled(adapter: str, identity: str) -> None:
    """Reject reuse of an exact allocation still owned by acquisition/rollback."""
    if (adapter, identity) in _ACQUISITIONS:
        raise RuntimeError("Runner acquisition or rollback is still pending for this resource.")


def retry_acquisition_settlement(
    previous: asyncio.Task[None], operation: Callable[[], Awaitable[None]], *, name: str
) -> asyncio.Task[None]:
    """Transfer only the exact retained restoration owner's explicit retry."""
    lease = next(
        (item for item in _ACQUISITIONS.values() if item.settlement_task is previous), None
    )
    if lease is not None and not previous.done():
        raise RuntimeError("Runner acquisition restoration is still running.")

    async def run() -> None:
        await operation()

    task = asyncio.create_task(run(), name=name)
    if lease is not None:
        lease.retry_task = task
        lease.retain_until(task)
    return task


async def drain_acquisition_restorations(adapter: str, *, timeout_s: float) -> int:
    """Join retained attachment cleanup and retry each settled failure once."""
    timeout = validate_cancel_timeout(timeout_s)
    leases = [
        lease
        for lease in _ACQUISITIONS.values()
        if lease.key[0] == adapter and lease.restoration is not None
    ]
    tasks: set[asyncio.Task[None]] = set()
    for lease in leases:
        task = lease.settlement_task
        assert task is not None and lease.restoration is not None
        if task.done():
            if not task.cancelled() and task.exception() is None:
                lease.release()
                continue
            task = retry_acquisition_settlement(
                task, lease.restoration, name="cayu-runner-attachment-restoration"
            )
        tasks.add(task)
    if tasks:
        await asyncio.wait(tasks, timeout=timeout)
    return sum(
        lease.key[0] == adapter and lease.restoration is not None
        for lease in _ACQUISITIONS.values()
    )


@dataclass
class CreationCleanupProgress(RunnerFailureProgress):
    reclaimed: bool = False


@dataclass(eq=False)
class _CreationCleanup:
    adapter: str
    operation: Callable[[CreationCleanupProgress], Awaitable[None]]
    identity: str | None = None
    acquisition: CreationLease | None = None
    progress: CreationCleanupProgress = field(default_factory=CreationCleanupProgress)
    task: asyncio.Task[CapturedAwaitableOutcome[None]] | None = None

    def start(self) -> asyncio.Task[CapturedAwaitableOutcome[None]]:
        if (
            self.task is not None
            and self.task.done()
            and not self.task.cancelled()
            and (self.task.result().error is None or self.progress.reclaimed)
        ):
            self.settled(self.task)
            return self.task
        if self.task is None or self.task.done():
            self.progress = CreationCleanupProgress()
            self.task = asyncio.create_task(
                capture_awaitable_outcome(lambda: self.operation(self.progress))
            )
            self.task.add_done_callback(self.settled)
        return self.task

    def settled(self, task: asyncio.Task[CapturedAwaitableOutcome[None]]) -> None:
        if (
            task is self.task
            and not task.cancelled()
            and (task.result().error is None or self.progress.reclaimed)
        ):
            _OWNERS.discard(self)
            if self.acquisition is not None:
                self.acquisition.release()


async def drain_creation_cleanups(adapter: str, *, timeout_s: float) -> int:
    """Join pending rollback; retry each settled failure once without reallocating."""
    timeout = validate_cancel_timeout(timeout_s)
    owners = [owner for owner in _OWNERS if owner.adapter == adapter]
    if owners:
        tasks = {owner.start() for owner in owners}
        done, _ = await asyncio.wait(tasks, timeout=timeout)
        for owner in owners:
            if owner.task in done:
                assert owner.task is not None
                owner.settled(owner.task)
    return sum(owner.adapter == adapter for owner in _OWNERS)


def retain_creation_cleanup(
    operation: Callable[[CreationCleanupProgress], Awaitable[None]],
    *,
    adapter: str,
    identity: str | None = None,
    acquisition: CreationLease | None = None,
    join_existing: bool = False,
) -> _CreationCleanup:
    """Own exact rollback synchronously, before any sibling observation can fail."""
    # Multiple reconciliation entrances can discover the same exact deletion.
    # Only callers whose identity fully defines that mutation may opt into joining.
    owner = next(
        (
            item
            for item in _OWNERS
            if join_existing
            and identity is not None
            and item.adapter == adapter
            and item.identity == identity
        ),
        None,
    )
    if owner is None and acquisition is None and identity is not None:
        acquisition = acquire_creation_lease(adapter, identity)
    if acquisition is not None:
        if (
            acquisition.key != (adapter, identity)
            or _ACQUISITIONS.get(acquisition.key) is not acquisition
        ):
            raise RuntimeError("Runner rollback does not own this acquisition.")
        acquisition.retained = True
    if owner is None:
        owner = _CreationCleanup(adapter, operation, identity, acquisition)
        _OWNERS.add(owner)
    return owner


async def settle_creation_cleanup(
    operation: Callable[[CreationCleanupProgress], Awaitable[None]],
    *,
    adapter: str,
    original: BaseException,
    message: str,
    timeout_s: float,
    identity: str | None = None,
    acquisition: CreationLease | None = None,
    join_existing: bool = False,
) -> None:
    timeout = validate_cancel_timeout(timeout_s)
    owner = retain_creation_cleanup(
        operation,
        adapter=adapter,
        identity=identity,
        acquisition=acquisition,
        join_existing=join_existing,
    )
    task = owner.start()
    outcome = await await_shielded_task_outcome(
        task,
        timeout_s=timeout,
        cancellation=original if isinstance(original, asyncio.CancelledError) else None,
    )
    failure = outcome.error if outcome.result is None else outcome.result.error
    if outcome.timed_out:
        failure = owner.progress.with_timeout("Runner constructor rollback has not settled.")
    if task.done():
        owner.settled(task)
    restore_task_cancellation_requests(
        outcome.cancellation_requests_consumed, cancellation=outcome.cancellation
    )
    if _contains_runner_fatal_signal(original) or (
        failure is not None and _contains_runner_fatal_signal(failure)
    ):
        failures = [original]
        if failure is not None and failure is not original:
            failures.append(failure)
        if outcome.cancellation is not None and all(
            outcome.cancellation is not item for item in failures
        ):
            failures.append(outcome.cancellation)
        if len(failures) == 1:
            raise original
        raise BaseExceptionGroup(message, failures) from None
    if outcome.cancellation is not None:
        evidence = [] if outcome.cancellation is original else [original]
        if failure is not None:
            evidence.append(failure)
        if evidence:
            attach_runner_cancellation_failure(
                outcome.cancellation,
                evidence[0] if len(evidence) == 1 else BaseExceptionGroup(message, evidence),
            )
        raise outcome.cancellation
    if failure is not None:
        raise BaseExceptionGroup(message, [original, failure]) from None
