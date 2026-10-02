"""Bounded ownership for knowledge publications retained across caller cancellation."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass
from math import isfinite
from typing import Any, Generic, TypeVar

_ResultT = TypeVar("_ResultT")


class KnowledgePublicationScope:
    """One application's share of the publications made through registered tools.

    Knowledge tools can be shared by several applications, so an application's
    shutdown must not seal or close a tool. Instead the application binds this
    scope to its tool calls: sealing it refuses that application's new
    publications, and draining it waits only for work that application started.
    The tool keeps owning that work; ``aclose`` on the tool stays the permanent
    close for whoever created it.
    """

    def __init__(self) -> None:
        self._sealed = False
        self._tasks: set[asyncio.Future[Any]] = set()

    @property
    def sealed(self) -> bool:
        return self._sealed

    @property
    def pending(self) -> bool:
        """Whether a publication this application started is still running."""

        return bool(self._tasks)

    def seal(self) -> None:
        """Synchronously refuse new publications started by this application."""

        self._sealed = True

    def track(self, task: asyncio.Future[Any]) -> None:
        """Wait for ``task`` when this scope drains; it stays owned by its tool."""

        if task.done() or task in self._tasks:
            return
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def drain(self, *, timeout_s: float) -> bool:
        """Seal, then wait for this application's work without cancelling it."""

        timeout = _positive_seconds(timeout_s)
        self.seal()
        if not self._tasks:
            return True
        _, pending = await asyncio.wait(tuple(self._tasks), timeout=timeout)
        return not pending


class KnowledgePublicationOwnerClosed(RuntimeError):
    """A publication owner has sealed its dispatch boundary."""


class KnowledgePublicationCapacityExhausted(RuntimeError):
    """A publication owner cannot retain another operation."""


class KnowledgePublicationOperationConflict(RuntimeError):
    """An in-flight operation identity was reused for different material."""


@dataclass(frozen=True, slots=True)
class RetainedKnowledgePublicationResult(Generic[_ResultT]):
    """The exact publication result and whether this caller joined its owner."""

    value: _ResultT
    joined: bool


@dataclass(frozen=True, slots=True)
class _RetainedKnowledgePublication(Generic[_ResultT]):
    fingerprint: str
    task: asyncio.Task[_ResultT]


class RetainedKnowledgePublicationOwner(Generic[_ResultT]):
    """Retain exact publication tasks until settlement or bounded shutdown.

    Caller cancellation never cancels a retained publication while the owner is
    open. Shutdown first seals dispatch, lets existing publications settle for
    the configured grace period, then requests cancellation from any remaining
    local awaiters. Durable operation identity and receipts remain authoritative
    when an adapter loses its acknowledgement during that final cancellation.
    """

    def __init__(self, *, max_publications: int) -> None:
        if type(max_publications) is not int:
            raise TypeError("max_publications must be an int.")
        if max_publications <= 0:
            raise ValueError("max_publications must be greater than zero.")
        self._max_publications = max_publications
        self._publications: dict[str, _RetainedKnowledgePublication[_ResultT]] = {}
        self._sealed = False
        self._closed = False
        self._close_task: asyncio.Task[bool] | None = None
        self._close_result: bool | None = None

    def __len__(self) -> int:
        return len(self._publications)

    @property
    def sealed(self) -> bool:
        return self._sealed

    @property
    def closed(self) -> bool:
        return self._closed

    def seal(self) -> None:
        """Synchronously reject publication dispatch beyond this boundary."""

        self._sealed = True

    async def join_existing(
        self,
        operation_id: str,
        fingerprint: str,
    ) -> RetainedKnowledgePublicationResult[_ResultT] | None:
        """Join an exact live operation without dispatching another publication."""

        owned = self._publications.get(operation_id)
        if owned is None:
            if self._sealed:
                raise KnowledgePublicationOwnerClosed
            return None
        if owned.fingerprint != fingerprint:
            raise KnowledgePublicationOperationConflict
        if self._closed:
            raise KnowledgePublicationOwnerClosed
        return RetainedKnowledgePublicationResult(
            value=await asyncio.shield(owned.task),
            joined=True,
        )

    async def run(
        self,
        operation_id: str,
        fingerprint: str,
        operation_factory: Callable[[], Awaitable[_ResultT]],
        *,
        scope: KnowledgePublicationScope | None = None,
    ) -> RetainedKnowledgePublicationResult[_ResultT]:
        """Join an exact operation or dispatch it under this owner's capacity.

        A sealed ``scope`` refuses a new dispatch but may still join exact work.
        """

        if not callable(operation_factory):
            raise TypeError("operation_factory must be callable.")
        owned = self._publications.get(operation_id)
        joined = owned is not None
        if owned is not None and owned.fingerprint != fingerprint:
            raise KnowledgePublicationOperationConflict
        if owned is None:
            if self._sealed or (scope is not None and scope.sealed):
                raise KnowledgePublicationOwnerClosed
            if len(self._publications) >= self._max_publications:
                raise KnowledgePublicationCapacityExhausted
            task = asyncio.create_task(_run_publication(operation_factory))
            owned = _RetainedKnowledgePublication(
                fingerprint=fingerprint,
                task=task,
            )
            self._publications[operation_id] = owned
            task.add_done_callback(
                lambda completed, operation_id=operation_id, owned=owned: self._release(
                    operation_id, owned, completed
                )
            )
        if scope is not None:
            scope.track(owned.task)
        return RetainedKnowledgePublicationResult(
            value=await asyncio.shield(owned.task),
            joined=joined,
        )

    async def aclose(self, *, timeout_s: float = 10.0) -> bool:
        """Seal and drain this owner once, returning whether it settled in grace."""

        timeout = _positive_seconds(timeout_s)
        self.seal()
        if self._close_result is not None:
            return self._close_result
        close_task = self._close_task
        if close_task is None:
            close_task = asyncio.create_task(self._close_started(timeout))
            close_task.add_done_callback(_observe_task_outcome)
            self._close_task = close_task
        return await asyncio.shield(close_task)

    async def _close_started(self, timeout_s: float) -> bool:
        tasks = tuple(owned.task for owned in self._publications.values())
        drained = True
        try:
            if tasks:
                _, pending = await asyncio.wait(tasks, timeout=timeout_s)
                drained = not pending
                if pending:
                    await _request_bounded_publication_stop(pending)
        except asyncio.CancelledError:
            drained = False
            await _request_bounded_publication_stop(task for task in tasks if not task.done())
            raise
        except BaseException:
            drained = False
            await _request_bounded_publication_stop(task for task in tasks if not task.done())
            raise
        finally:
            self._closed = True
            self._close_result = drained
        return drained

    def _release(
        self,
        operation_id: str,
        owned: _RetainedKnowledgePublication[_ResultT],
        completed: asyncio.Task[_ResultT],
    ) -> None:
        if self._publications.get(operation_id) is owned:
            self._publications.pop(operation_id, None)
        if not completed.cancelled():
            # Always retrieve detached failures. Public callers project their
            # own fixed diagnostics and never expose this exception text.
            with suppress(BaseException):
                completed.exception()


async def _run_publication(
    operation_factory: Callable[[], Awaitable[_ResultT]],
) -> _ResultT:
    return await operation_factory()


def _observe_task_outcome(task: asyncio.Task[Any]) -> None:
    if not task.cancelled():
        with suppress(BaseException):
            task.exception()


async def _request_bounded_publication_stop(
    tasks: Iterable[asyncio.Task[Any]],
) -> None:
    pending = tuple(task for task in tasks if not task.done())
    for task in pending:
        task.cancel("Knowledge publication owner is shutting down.")
    # A retained foreground publication contains a small, fixed number of
    # Cayu-owned wrapper tasks. Give cooperative adapters enough scheduling
    # turns to propagate cancellation and run their synchronous finalizers;
    # never await opaque adapter settlement after the grace period.
    for _ in range(8):
        if all(task.done() for task in pending):
            break
        await asyncio.sleep(0)


def _positive_seconds(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not isfinite(value)
        or value <= 0
    ):
        raise ValueError("timeout_s must be a finite positive number.")
    return float(value)
