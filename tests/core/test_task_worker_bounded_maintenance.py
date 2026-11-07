"""A bounded pool must keep recovering lost work while its last handler runs."""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

import cayu.tasks.worker as worker_module
from cayu.applications import CayuApp
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks import TaskCreate, TaskQuery, TaskStatus, complete_managed_task, run_task_worker
from cayu.tasks.memory import InMemoryTaskStore


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("bounded", [False, True])
def test_bounded_pool_keeps_maintenance_after_its_owner_finishes(
    backend, bounded, tmp_path, monkeypatch
):
    async def scenario():
        now = [datetime.now(UTC)]
        store = (
            InMemoryTaskStore(ownership_clock=lambda: now[0])
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "tasks.sqlite", ownership_clock=lambda: now[0])
        )
        app = CayuApp(task_store=store, enable_logging=False)
        await store.create_task(TaskCreate(task_id="lost", type="job"))
        lost = await store.claim_task("gone", lease_seconds=30)
        await store.mark_claimed_task_execution_started(
            lost.id, "gone", claim_generation=lost.claim_generation
        )
        for task_id in ("quick", "long"):
            await store.create_task(TaskCreate(task_id=task_id, type="job"))

        busy, release, owner_finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
        stop = asyncio.Event()
        stop_maintenance = worker_module._stop_busy_slot_maintenance

        async def observe_stop(maintenance, idle, failure):
            await stop_maintenance(maintenance, idle, failure)
            if failure is None:
                owner_finished.set()

        monkeypatch.setattr(worker_module, "_stop_busy_slot_maintenance", observe_stop)

        async def handler(_app, task, worker_id):
            if task.id == "long":
                busy.set()
                await release.wait()
            else:
                await busy.wait()
            await complete_managed_task(store, task, worker_id, {})

        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="pool",
                query=TaskQuery(type="job"),
                concurrency=2,
                max_tasks=2 if bounded else None,
                stop=stop,
                lease_seconds=3600,
                poll_interval_s=0.01,
                reclaim_every_s=0.02,
                reconcile_abandoned_after_s=0,
                recover_interrupted_handoffs=False,
            )
        )
        try:
            await asyncio.wait_for(busy.wait(), 5)
            await asyncio.wait_for(owner_finished.wait(), 5)
            # Start expiry only after the slot that owns maintenance has drained.
            now[0] = lost.lease_expires_at + timedelta(seconds=1)
            deadline = asyncio.get_running_loop().time() + 1.0
            while True:
                current = await store.load_task(lost.id)
                if current.status is TaskStatus.CANCELLED:
                    break
                if asyncio.get_running_loop().time() >= deadline:
                    break
                await asyncio.sleep(0.01)
            assert not worker.done(), "The long handler must still own its slot."
            assert current.status is TaskStatus.CANCELLED, (current.status, current.status_reason)
        finally:
            stop.set()
            release.set()
            assert await asyncio.wait_for(worker, 5) == 2
            await app.aclose()
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())
