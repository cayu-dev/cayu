"""Retry deadline inspection must not outlive the worker lease."""

from __future__ import annotations

import asyncio
from time import monotonic

import pytest

from cayu.tasks.creation import TaskCreate
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.records import TaskClaimLost, TaskRetryPolicy
from cayu.tasks.worker import _heartbeat_until, _TaskLeaseAuthority


class BlockingRetryDeadlineStore(InMemoryTaskStore):
    verified_work_mutations_are_cancellation_quiescent = True

    def __init__(self) -> None:
        super().__init__()
        self.inspection_started = asyncio.Event()
        self.release_inspection = asyncio.Event()

    async def task_retry_deadline_elapsed(self, *args, **kwargs):
        self.inspection_started.set()
        await self.release_inspection.wait()
        return await super().task_retry_deadline_elapsed(*args, **kwargs)


def test_retry_deadline_inspection_cannot_hold_heartbeat_past_lease():
    async def scenario():
        store = BlockingRetryDeadlineStore()
        created = await store.create_task(
            TaskCreate(
                task_id="heartbeat-retry-deadline",
                type="job",
                retry_policy=TaskRetryPolicy(max_attempts=2, max_elapsed_seconds=60),
            )
        )
        claimed = await store.claim_task("owner", lease_seconds=1)
        assert claimed is not None and claimed.retry_series is not None
        authority = _TaskLeaseAuthority(
            lease_expires_at=claimed.lease_expires_at,
            handoff_id=None,
            task_id=created.id,
            worker_id="owner",
            claim_generation=claimed.claim_generation,
            deadline_monotonic=monotonic() + 0.7,
        )
        stop = asyncio.Event()
        heartbeat = asyncio.create_task(
            _heartbeat_until(
                store,
                created.id,
                "owner",
                1,
                stop,
                lease_authority=authority,
                claim_deadline_monotonic=authority.deadline_monotonic,
                enforce_retry_deadline=True,
            )
        )
        try:
            await asyncio.wait_for(store.inspection_started.wait(), 2)
            await asyncio.sleep(0.8)
            assert heartbeat.done(), "a retry-deadline read held the heartbeat past its lease"
            with pytest.raises(TaskClaimLost):
                heartbeat.result()
        finally:
            store.release_inspection.set()
            if not heartbeat.done():
                heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)

    asyncio.run(scenario())
