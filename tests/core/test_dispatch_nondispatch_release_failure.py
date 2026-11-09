"""A failed requeue must not orphan a dispatch proven not to have run."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from tests.core.test_dispatch import _batch, _build, _dispatch_request

from cayu.messages import Message
from cayu.sessions.requests import RunRequest
from cayu.tasks.dispatch import TaskStoreDispatcher
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.records import TaskStatus


@pytest.mark.parametrize("fail_release", [False, True])
def test_retryable_nondispatch_release_failure_keeps_work_recoverable(
    monkeypatch, fail_release: bool
) -> None:
    asyncio.run(_scenario(monkeypatch, fail_release))


async def _scenario(monkeypatch, fail_release: bool) -> None:
    now = [datetime.now(UTC)]

    class FailOneReleaseStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self):
            super().__init__(ownership_clock=lambda: now[0])
            self.execution_started = False
            self.release_calls = 0

        async def mark_claimed_task_execution_started(self, *args, **kwargs):
            result = await super().mark_claimed_task_execution_started(*args, **kwargs)
            self.execution_started = True
            return result

        async def release_task(self, *args, **kwargs):
            self.release_calls += 1
            if fail_release and self.release_calls == 1:
                raise ConnectionError("temporary task-store release failure")
            return await super().release_task(*args, **kwargs)

    tasks = FailOneReleaseStore()
    h = _build([_batch("initial"), _batch("queued result")], task_store=tasks)
    session_id = "nondispatch-release-failure"
    async for _ in h.app.run(
        RunRequest(
            agent_name="assistant",
            session_id=session_id,
            messages=[Message.text("user", "initial")],
        )
    ):
        pass
    submitted = await h.app.dispatch(_dispatch_request(session_id, "retryable-work"))
    task_id = submitted.metadata["queue_task_id"]
    load_snapshot = h.app._queued_dispatch_coordinator._load_session_snapshot
    failed_execution_read = False

    async def fail_execution_snapshot_once(private_session_id):
        nonlocal failed_execution_read
        if tasks.execution_started and not failed_execution_read:
            failed_execution_read = True
            # The runtime fails before session admission. Its subsequent exact
            # settlement read is healthy and proves NOT_ADMITTED.
            raise ConnectionError("temporary session snapshot failure before admission")
        return await load_snapshot(private_session_id)

    monkeypatch.setattr(
        h.app._queued_dispatch_coordinator,
        "_load_session_snapshot",
        fail_execution_snapshot_once,
    )
    outcome = (
        await asyncio.gather(
            h.dispatcher.process_next(h.app, worker_id="original"),
            return_exceptions=True,
        )
    )[0]
    assert failed_execution_read
    assert len(h.provider.requests) == 1  # No queued provider work was invoked.

    # All stores are healthy now. Reclamation and ordinary replacement polling
    # must recover the runnable request without manual cancellation evidence.
    current = await tasks.load_task(task_id)
    assert current is not None
    if current.lease_expires_at is not None:
        now[0] = current.lease_expires_at + timedelta(seconds=1)
    await tasks.reclaim_expired()
    replacement = TaskStoreDispatcher(tasks)
    for _ in range(3):
        await replacement.process_next(h.app, worker_id="replacement")
    current = await tasks.load_task(task_id)
    assert current is not None
    assert current.status is TaskStatus.COMPLETED, (current.status_reason, repr(outcome))
    assert len(h.provider.requests) == 2
