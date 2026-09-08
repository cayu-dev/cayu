"""Finite in-process owner of checkpoint publication and exact receipt readback.

No native operation is invoked here. A timed-out write remains owned and cannot
be replaced by a second local write; a fresh process can reconcile its durable
receipt. The enclosing server lifecycle must retain this owner until drain proves
all writes finished. Publication success is not guest-quiescence evidence.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from cayu._task_wait import (
    CapturedAwaitableOutcome,
    await_shielded_task_outcome,
    capture_awaitable_outcome,
    restore_task_cancellation_requests,
    unexpected_child_cancellation_error,
)
from cayu._validation import canonical_durable_json_bytes
from cayu.runtime._browser_control_publication import (
    BrowserControlFencePublication,
    BrowserControlPublication,
)
from cayu.runtime.browser_control import BrowserControlConflict, BrowserControlRecord
from cayu.runtime.sessions import SessionStore


class BrowserControlPublicationPending(RuntimeError):
    def __init__(self) -> None:
        super().__init__("Browser control publication is still owned and requires reconciliation.")


@dataclass(frozen=True, slots=True)
class _PendingPublication:
    command: BrowserControlPublication
    task: asyncio.Task[CapturedAwaitableOutcome[BrowserControlRecord]]


class BrowserControlPublisher:
    def __init__(self, store: SessionStore, *, maximum_pending: int = 32) -> None:
        if type(maximum_pending) is not int or not 1 <= maximum_pending <= 1024:
            raise ValueError("Browser control requires a bounded publication owner limit.")
        self._store = store
        self._maximum_pending = maximum_pending
        self._pending: dict[tuple[str, str], _PendingPublication] = {}
        self._closing = False

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    async def publish(
        self, command: BrowserControlPublication, *, timeout_s: float = 5.0
    ) -> BrowserControlRecord:
        """Publish or reconcile exactly this command, never replay native input.

        The caller must authorize before entry. A returned historical record must
        not be used to dispatch: current record and guest epoch still require CAS.
        """
        if type(timeout_s) not in {int, float} or not 0 < timeout_s <= 30:
            raise ValueError("Browser control publication wait must be finite and bounded.")
        owned = (
            BrowserControlFencePublication(command.owned_mutation)
            if type(command) is BrowserControlFencePublication
            else BrowserControlPublication(command.owned_mutation)
        )
        key = (owned.mutation.session_id, owned.storage_key)
        pending = self._pending.get(key)
        if pending is not None:
            if (
                type(pending.command) is not type(owned)
                or pending.command.mutation != owned.mutation
            ):
                raise BrowserControlConflict("Browser control publication identity conflicts.")
        else:
            if self._closing or len(self._pending) >= self._maximum_pending:
                raise BrowserControlPublicationPending()
            task = asyncio.create_task(
                capture_awaitable_outcome(lambda: self._publish_once(owned)),
                name="cayu-browser-control-publication",
            )
            pending = _PendingPublication(owned, task)
            self._pending[key] = pending
        outcome = await await_shielded_task_outcome(pending.task, timeout_s=timeout_s)
        captured = outcome.result
        failure = outcome.error if captured is None else captured.error
        result = None if captured is None else captured.result
        if pending.task.done() and self._pending.get(key) is pending:
            del self._pending[key]
        if outcome.cancellation is not None:
            restore_task_cancellation_requests(
                outcome.cancellation_requests_consumed, cancellation=outcome.cancellation
            )
            if failure is not None:
                if not isinstance(failure, Exception | asyncio.CancelledError):
                    raise BaseExceptionGroup(
                        "Browser control publication received cancellation and process control.",
                        [outcome.cancellation, failure],
                    )
                raise outcome.cancellation from failure
            raise outcome.cancellation
        if outcome.timed_out:
            raise BrowserControlPublicationPending()
        if failure is not None:
            if isinstance(failure, asyncio.CancelledError):
                raise unexpected_child_cancellation_error(
                    failure, operation="Browser control publication"
                )
            raise failure
        if result is None:
            raise BrowserControlConflict("Browser control publication has no receipt.")
        return result

    async def drain(self, *, timeout_s: float = 5.0) -> bool:
        """Stop new publications and report positive completion without cancellation.

        Existing exact publications may still be reconciled through ``publish``.
        Completed captured failures remain available there; draining neither
        erases them nor converts them into successful acknowledgement.
        """
        if type(timeout_s) not in {int, float} or not 0 < timeout_s <= 30:
            raise ValueError("Browser control drain wait must be finite and bounded.")
        self._closing = True
        tasks = {entry.task for entry in self._pending.values()}
        if not tasks:
            return True
        _, running = await asyncio.wait(tasks, timeout=timeout_s)
        return not running

    async def _readback(self, command: BrowserControlPublication) -> BrowserControlRecord | None:
        raw = await self._store.load_session_operation(
            command.mutation.session_id, command.storage_key
        )
        if raw is None:
            return None
        # Ordinary equality would accept True in place of schema_version=1.
        if canonical_durable_json_bytes(raw, "browser control receipt") != (
            canonical_durable_json_bytes(command.receipt(), "browser control receipt")
        ):
            raise BrowserControlConflict("Browser control receipt differs from the exact request.")
        return command.changed_record

    async def _publish_once(self, command: BrowserControlPublication) -> BrowserControlRecord:
        replay = await self._readback(command)
        if replay is not None:
            return replay
        try:
            with command.scope():
                await self._store.publish_session_operation_guarded_with_store_time(
                    command.mutation.session_id,
                    idempotency_key=command.storage_key,
                    operation_transform=command.transform,
                    commit_guard=lambda: None,
                    commit_time_guard=command.validate_commit_time,
                    expected_statuses=command.expected_statuses,
                    expected_run_epoch=command.expected_run_epoch,
                    events=[],
                )
        except Exception as publication_failure:
            try:
                replay = await self._readback(command)
            except Exception as readback_failure:
                raise ExceptionGroup(
                    "Browser control publication and reconciliation failed.",
                    [publication_failure, readback_failure],
                ) from None
            if replay is None:
                raise
            return replay
        result = await self._readback(command)
        if result is None:
            raise BrowserControlConflict("Browser control publication has no durable receipt.")
        return result
