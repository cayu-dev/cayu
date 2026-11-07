"""A lost execution-start acknowledgement still settles after reclaim fences that claim."""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from tests.core.test_task_execution_settlement_owner import assert_settled, create_group

from cayu.applications import CayuApp
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks import TaskCreate, TaskQuery
from cayu.tasks._execution_settlement import TaskExecutionSettlement
from cayu.tasks.memory import InMemoryTaskStore


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_entry_readback_accepts_reclaims_fence_of_its_own_claim(backend, tmp_path, monkeypatch):
    async def scenario():
        now = [datetime(2026, 1, 1, tzinfo=UTC)]
        store = (
            InMemoryTaskStore(ownership_clock=lambda: now[0])
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "tasks.db", ownership_clock=lambda: now[0])
        )
        app = CayuApp(task_store=store, enable_logging=False)
        try:
            await create_group(app, TaskCreate(task_id="member", type="member"))
            claimed = await store.claim_task("owner", TaskQuery(type="member"), lease_seconds=30)
            assert claimed is not None

            original = type(store).mark_claimed_task_execution_started

            async def lose_acknowledgement(self, *args, **kwargs):
                await original(self, *args, **kwargs)
                raise OSError("Start acknowledgement lost.")

            monkeypatch.setattr(
                type(store), "mark_claimed_task_execution_started", lose_acknowledgement
            )
            settlement = TaskExecutionSettlement(store)
            # The store boundary detaches the lost acknowledgement into a redacted failure.
            with pytest.raises(RuntimeError):
                await settlement.enter(claimed)

            # The lease lapses; reclaim fences the started claim by requesting its
            # cancellation and advancing its generation by one.
            now[0] += timedelta(seconds=31)
            await store.reclaim_expired()
            fenced = await store.load_task(claimed.id)
            assert fenced.claim_generation == claimed.claim_generation + 1
            assert fenced.worker_id == "owner"

            await settlement.retry()
            await assert_settled(app, claimed.id)
        finally:
            await app.aclose()

    asyncio.run(scenario())
