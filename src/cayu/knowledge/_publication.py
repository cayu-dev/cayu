"""Bounded ownership for knowledge publications retained across caller cancellation."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass
from math import isfinite
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

_ResultT = TypeVar("_ResultT")


@runtime_checkable
class KnowledgePublicationLifecycle(Protocol):
    """Complete shutdown lifecycle exposed by one registered knowledge writer."""

    def seal(self) -> None:
        """Synchronously reject new publication dispatch."""

    async def aclose(self, *, timeout_s: float = 10.0) -> bool:
        """Drain all lifecycle-owned work within one shared deadline."""


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
    ) -> RetainedKnowledgePublicationResult[_ResultT]:
        """Join an exact operation or dispatch it under this owner's capacity."""

        if not callable(operation_factory):
            raise TypeError("operation_factory must be callable.")
        owned = self._publications.get(operation_id)
        joined = owned is not None
        if owned is not None and owned.fingerprint != fingerprint:
            raise KnowledgePublicationOperationConflict
        if owned is None:
            if self._sealed:
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
