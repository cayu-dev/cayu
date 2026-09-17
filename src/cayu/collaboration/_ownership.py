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
        try:
            done, _ = await asyncio.wait((task,), timeout=self.observation_timeout)
        except asyncio.CancelledError:
            cancelled = True
            done = set()
        # Raise outside the handler so a caller's secret-bearing cancellation
        # is not retained through __context__. The operation remains owned.
        if cancelled:
            diagnostic = None
            if task.done() and not task.cancelled():
                error = task.exception()
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

    async def drain(self) -> None:
        self.closed = True
        if self.pending:
            _, pending = await asyncio.wait(tuple(self.pending), timeout=self.observation_timeout)
            if pending:
                raise CollaborationUnavailable("Collaboration mutations are still draining.")
