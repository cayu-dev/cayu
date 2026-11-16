"""Started claims consume reclaim slots even though they are fenced, not returned."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from cayu.applications import CayuApp
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks import (
    TaskCreate,
    TaskQuery,
    TaskStatus,
    complete_managed_task,
    run_task_worker_until_idle,
)
from cayu.tasks.memory import InMemoryTaskStore


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("started_claims", [0, 1, 100])
def test_until_idle_reclaims_past_a_batch_containing_started_claims(
    backend, started_claims, tmp_path
):
    async def scenario():
        now = [datetime.now(UTC)]
        store = (
            InMemoryTaskStore(ownership_clock=lambda: now[0])
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "tasks.sqlite", ownership_clock=lambda: now[0])
        )
        app = CayuApp(task_store=store, enable_logging=False)
        handled = []
        try:
            # Keep the worker's default 100-row batch. A dead process started
            # some handlers, but died before dispatching its final claim.
            for index in range(101):
                created = await store.create_task(TaskCreate(task_id=f"job-{index:03}", type="job"))
                claimed = await store.claim_task(
                    "dead-worker", TaskQuery(type="job"), lease_seconds=30
                )
                assert claimed is not None and claimed.id == created.id
                if index < started_claims:
                    await store.mark_claimed_task_execution_started(
                        claimed.id, "dead-worker", claim_generation=claimed.claim_generation
                    )
                # Give the store an unambiguous oldest-first reclamation order.
                now[0] += timedelta(milliseconds=1)
            now[0] += timedelta(seconds=31)

            async def handler(_app, task, worker_id):
                handled.append(task.id)
                await complete_managed_task(store, task, worker_id, {"ok": True})

            result = await asyncio.wait_for(
                run_task_worker_until_idle(
                    app,
                    store,
                    handler,
                    worker_id="replacement",
                    query=TaskQuery(type="job"),
                    # A replacement elected as the only process can settle the
                    # stopped handlers while recovering the undispatched work.
                    reconcile_abandoned_after_s=0,
                    recover_interrupted_handoffs=False,
                    poll_interval_s=0.01,
                ),
                10,
            )
            last = await store.load_task("job-100")
            assert last is not None
            assert last.status is TaskStatus.COMPLETED, (
                "The worker returned idle before reclaiming undispatched work "
                "behind a batch containing started claims: "
                f"status={last.status}, owner={last.worker_id}, result={result}"
            )
            assert result.idle and not result.maintenance_required
            assert handled == [f"job-{i:03}" for i in range(started_claims, 101)]
            for index in range(started_claims):
                assert (await store.load_task(f"job-{index:03}")).status is TaskStatus.CANCELLED
        finally:
            await app.aclose()
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())
