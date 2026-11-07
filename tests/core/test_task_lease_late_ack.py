"""A renewal whose acknowledgement never arrived must not fence its own holder."""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks.creation import TaskCreate
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.records import TaskStatus


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_claim_generation_survives_an_unacknowledged_renewal(backend, tmp_path):
    async def scenario():
        now = [datetime(2026, 1, 1, tzinfo=UTC)]
        store = (
            InMemoryTaskStore(ownership_clock=lambda: now[0])
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "tasks.sqlite", ownership_clock=lambda: now[0])
        )
        try:
            await store.create_task(TaskCreate(task_id="job", type="job"))
            claimed = await store.claim_task("holder", lease_seconds=30)
            assert claimed is not None
            now[0] += timedelta(seconds=1)
            # The renewal commits, but its acknowledgement is lost: the holder
            # still knows only what its claim returned.
            renewed = await store.heartbeat(
                "job", "holder", lease_expires_at=claimed.lease_expires_at, extend_seconds=30
            )
            assert renewed.lease_expires_at != claimed.lease_expires_at
            assert renewed.claim_generation == claimed.claim_generation
            now[0] += timedelta(seconds=1)
            await store.mark_claimed_task_execution_started(
                "job", "holder", claim_generation=claimed.claim_generation
            )
            await store.heartbeat("job", "holder", claim_generation=claimed.claim_generation)
            completed = await store.complete_task(
                "job",
                {"ok": True},
                worker_id="holder",
                claim_generation=claimed.claim_generation,
            )
            assert completed.status is TaskStatus.COMPLETED
            assert completed.claim_generation == claimed.claim_generation
        finally:
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())
