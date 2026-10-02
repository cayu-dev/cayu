"""Bound observation without cancelling an opaque durable operation."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

from cayu.collaboration._contracts import CollaborationConflict
from cayu.collaboration._diagnostics import safe_failure
from cayu.collaboration.participants import CollaborationCapacityExceeded, CollaborationUnavailable
from cayu.vaults.redaction import SecretRedactor

T = TypeVar("T")


class _MutationOwners:
    def __init__(self) -> None:
        self.pending: set[asyncio.Task[Any]] = set()
        self._keys: dict[tuple[object, ...], tuple[bytes, asyncio.Task[Any]]] = {}
        self.closed = False
        self.observation_timeout = 10.0

    async def run(
        self,
        operation: Callable[[], Coroutine[Any, Any, T]],
        *,
        key: tuple[object, ...],
        expectation: bytes,
        redactor: SecretRedactor,
        failure_snapshot: Callable[[BaseException], BaseException] | None = None,
        result_failure: Callable[[T], BaseException | None] | None = None,
        wait_for_settlement: bool = False,
        track: Callable[[asyncio.Task[Any]], None] | None = None,
        observation_timeout: float | None = None,
    ) -> T:
        cancelled = False
        try:
            # Deliver pre-dispatch cancellation, without retaining its message.
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            cancelled = True
        if cancelled:
            raise asyncio.CancelledError("Collaboration observation cancelled before dispatch.")
        if self.closed:
            raise CollaborationUnavailable("Collaboration store is closing.")
        existing = self._keys.get(key)
        if existing is not None:
            previous, task = existing
            if previous != expectation:
                raise CollaborationConflict("Pending collaboration intent conflicts.")
        elif len(self.pending) >= 64:
            raise CollaborationCapacityExceeded(
                "Collaboration store has too many pending operations."
            )
        else:
            coroutine = operation()
            try:
                task = asyncio.create_task(coroutine, name="cayu-collaboration-mutation")
            except BaseException:
                coroutine.close()
                raise
            self.pending.add(task)
            self._keys[key] = (expectation, task)
            task.add_done_callback(lambda settled: self._settled(key, settled))
        if track is not None:
            track(task)
        try:
            # An enclosing runtime owner already has its execution deadline.
            # It must not turn this public observation bound into a failure of
            # the still-running mutation. Cancellation still leaves it owned.
            done, _ = await asyncio.wait(
                (task,),
                timeout=(
                    None
                    if wait_for_settlement
                    else self.observation_timeout
                    if observation_timeout is None
                    else observation_timeout
                ),
            )
        except asyncio.CancelledError:
            cancelled = True
            done = set()
        # Raise outside the handler so a caller's secret-bearing cancellation
        # is not retained through __context__. The operation remains owned.
        if cancelled:
            diagnostic = None
            if task.done() and not task.cancelled():
                error = task.exception()
                if error is None and result_failure is not None:
                    # Some owners return typed failure evidence for their own
                    # handoff logic. Cancellation must preserve that evidence
                    # through the same sanitizer as a raised task failure.
                    error = result_failure(task.result())
                if error is not None:
                    diagnostic = (
                        safe_failure(error, redactor=redactor)
                        if failure_snapshot is None
                        else failure_snapshot(error)
                    )
            raise asyncio.CancelledError(
                "Collaboration observation cancelled; reconcile the exact operation."
            ) from diagnostic
        if not done:
            raise CollaborationUnavailable(
                "Collaboration acknowledgement is pending; reconcile the exact operation."
            )
        if task.cancelled():
            raise CollaborationUnavailable(
                "Collaboration dependency cancelled; reconcile the exact operation."
            )
        return task.result()

    def _settled(self, key: tuple[object, ...], task: asyncio.Task[Any]) -> None:
        self.pending.discard(task)
        self._keys.pop(key, None)
        if not task.cancelled():
            # Consume late errors; exact owner readback, not notification, owns
            # durable outcome. Successful readback must never be guessed here.
            task.exception()

    def seal(self) -> None:
        """Refuse new operations without waiting for those already running."""

        self.closed = True

    def track(self, task: asyncio.Task[Any]) -> None:
        """Also wait for ``task``, run by another owner, when draining."""

        if not task.done() and task not in self.pending:
            self.pending.add(task)
            task.add_done_callback(self.pending.discard)

    def scope(self) -> _MutationScope:
        """A view of this store's mutations owned by one application."""

        return _MutationScope(self)

    def outstanding(self) -> set[asyncio.Task[Any]]:
        """Operations a drain waits for."""

        return set(self.pending)

    async def drain(self, *, timeout_s: float | None = None) -> None:
        self.closed = True
        await _await_pending(
            self.outstanding(), self.observation_timeout if timeout_s is None else timeout_s
        )


class _MutationScope:
    """One application's mutations against a shared collaboration store.

    Each task is also registered with the store's own owners, so the store keeps
    admitting its transactions and still waits for it when the store closes.
    Draining a scope refuses only this application's new work and never closes
    the shared store. It waits for this application's work and for every
    operation pending on the store: some application paths run directly on the
    store's owners, and those cannot be told apart from other applications'
    short operations, so waiting for all of them keeps a drain from reporting
    settled while this application's work still runs.
    """

    def __init__(self, parent: _MutationOwners) -> None:
        self._parent = parent
        self.pending: set[asyncio.Task[Any]] = set()
        self.closed = False
        # This application's own observation bound; the store's stays unchanged.
        self.observation_timeout = parent.observation_timeout

    async def run(self, operation: Callable[[], Coroutine[Any, Any, T]], **kwargs: Any) -> T:
        if self.closed:
            raise CollaborationUnavailable("Collaboration requests are closing.")
        return await self._parent.run(
            operation,
            track=self.track,
            observation_timeout=self.observation_timeout,
            **kwargs,
        )

    def seal(self) -> None:
        """Refuse this application's new operations; the shared store stays open."""

        self.closed = True

    def track(self, task: asyncio.Task[Any]) -> None:
        """Wait for ``task`` when this application's work drains."""

        if not task.done() and task not in self.pending:
            self.pending.add(task)
            task.add_done_callback(self.pending.discard)

    def outstanding(self) -> set[asyncio.Task[Any]]:
        """This application's operations and every operation pending on the store."""

        return self.pending | self._parent.pending

    async def drain(self, *, timeout_s: float | None = None) -> None:
        self.closed = True
        await _await_pending(
            self.outstanding(), self.observation_timeout if timeout_s is None else timeout_s
        )


async def _await_pending(pending: set[asyncio.Task[Any]], timeout_s: float) -> None:
    if pending:
        _, remaining = await asyncio.wait(tuple(pending), timeout=timeout_s)
        if remaining:
            raise CollaborationUnavailable("Collaboration mutations are still draining.")
