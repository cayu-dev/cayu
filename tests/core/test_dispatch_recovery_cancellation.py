"""A busy-session recovery handoff must not strand a cancelled queue claim."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from tests.core.test_dispatch import _batch, _build, _dispatch_request

from cayu.messages import Message
from cayu.sessions.requests import ResumeRequest, RunRequest
from cayu.tasks.base import TaskClaimLost
from cayu.tasks.dispatch import TaskStoreDispatcher
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.records import TaskStatus


@pytest.mark.parametrize("interruption", ["owner", "operator", "heartbeat"])
def test_cancellation_during_busy_session_recovery_keeps_claim_settleable(
    monkeypatch, interruption
):
    asyncio.run(_cancellation_during_recovery(monkeypatch, interruption))


async def _cancellation_during_recovery(monkeypatch, interruption):
    cancel_owner = interruption == "owner"
    now = [datetime.now(UTC)]
    tasks = InMemoryTaskStore(ownership_clock=lambda: now[0])
    h = _build(
        [_batch("initial"), _batch("busy run"), _batch("queued result")],
        task_store=tasks,
    )
    session_id = f"cancel-recovery-{cancel_owner}"
    async for _ in h.app.run(
        RunRequest(
            agent_name="assistant",
            session_id=session_id,
            messages=[Message.text("user", "initial")],
        )
    ):
        pass
    submitted = await h.app.dispatch(_dispatch_request(session_id, f"queued-{cancel_owner}"))
    task_id = submitted.metadata["queue_task_id"]
    # Keep a real, independent invocation active, so queued execution is
    # positively NOT_ADMITTED and enters the busy-session recovery branch.
    busy_entered = asyncio.Event()
    release_busy = asyncio.Event()
    stream = h.provider.stream
    heartbeat = tasks.heartbeat
    replacement_renewed = asyncio.Event()

    async def busy_stream(request):
        busy_entered.set()
        await release_busy.wait()
        if interruption == "heartbeat" and len(h.provider.requests) == 2:
            # Make the replacement renew before it can finish, so a leaked
            # failure injector cannot hide behind a fast provider response.
            await asyncio.wait_for(replacement_renewed.wait(), 5)
        async for event in stream(request):
            yield event

    monkeypatch.setattr(h.provider, "stream", busy_stream)

    async def occupy_session():
        async for _ in h.app.resume(
            ResumeRequest(session_id=session_id, messages=[Message.text("user", "occupy")])
        ):
            pass

    busy = asyncio.create_task(occupy_session())
    await asyncio.wait_for(busy_entered.wait(), 5)
    recovery_entered = asyncio.Event()
    release_recovery = asyncio.Event()
    recover = h.app.recover_incomplete_session

    async def delayed_recovery(request):
        recovery_entered.set()
        await release_recovery.wait()
        return await recover(request)

    monkeypatch.setattr(h.app, "recover_incomplete_session", delayed_recovery)
    processing = asyncio.create_task(h.dispatcher.process_next(h.app, worker_id="owner"))
    try:
        await asyncio.wait_for(recovery_entered.wait(), 5)
        if cancel_owner:
            processing.cancel("stop dispatch worker")
        elif interruption == "heartbeat":
            failed = asyncio.Event()

            async def failed_heartbeat(*args, **kwargs):
                failed.set()
                raise TaskClaimLost("renewal acknowledgement timed out")

            monkeypatch.setattr(tasks, "heartbeat", failed_heartbeat)
            await asyncio.wait_for(failed.wait(), 5)
            await asyncio.sleep(0)
        else:
            requested = await tasks.cancel_task(task_id, {"reason": "operator cancelled"})
            assert requested.status_reason == "cancellation_requested"
        release_recovery.set()
        result = (await asyncio.gather(processing, return_exceptions=True))[0]
        if cancel_owner:
            assert isinstance(result, asyncio.CancelledError)

        # Let the unrelated invocation finish too: nothing is left executing
        # in the target session when the replacement dispatcher polls.
        release_busy.set()
        await asyncio.wait_for(busy, 5)
        current = await tasks.load_task(task_id)
        assert current is not None
        # Expiry and subsequent healthy polling must not strand a claim whose
        # dispatcher has already established non-admission and then exited.
        if current.lease_expires_at is not None:
            now[0] = current.lease_expires_at + timedelta(seconds=1)
            await tasks.reclaim_expired()

        async def healthy_heartbeat(task_id, worker_id, **kwargs):
            renewed = await heartbeat(task_id, worker_id, **kwargs)
            if worker_id == "replacement":
                replacement_renewed.set()
            return renewed

        # The fault belongs to the exited owner. Replacement polling must use
        # the real store renewal, including when execution crosses a heartbeat.
        monkeypatch.setattr(tasks, "heartbeat", healthy_heartbeat)
        restarted = TaskStoreDispatcher(tasks)
        for _ in range(3):
            await restarted.process_next(h.app, worker_id="replacement")
        current = await tasks.load_task(task_id)
        assert current is not None
        assert current.status is not TaskStatus.CLAIMED, (current.status_reason, result)
        if interruption == "operator":
            assert current.status is TaskStatus.CANCELLED
            assert len(h.provider.requests) == 2
        else:
            assert current.status is TaskStatus.COMPLETED
            assert len(h.provider.requests) == 3
            if interruption == "heartbeat":
                assert replacement_renewed.is_set()
    finally:
        release_recovery.set()
        release_busy.set()
        await asyncio.gather(processing, return_exceptions=True)
        await asyncio.wait_for(busy, 5)
