"""A worker stopped after renewing an unstarted claim still returns that claim."""

import asyncio

import pytest

import cayu.tasks.worker as worker_module
from cayu.applications import CayuApp
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks import TaskCreate, TaskStatus
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.worker import run_task_worker


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_cancellation_after_pre_dispatch_renewal_returns_the_claim(backend, tmp_path, monkeypatch):
    async def scenario():
        store = (
            InMemoryTaskStore()
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "tasks.sqlite")
        )
        app = CayuApp(task_store=store, enable_logging=False)
        created = await store.create_task(TaskCreate(type="job"))
        renewed = asyncio.Event()

        renewals = []

        async def blocked_after_renewal(_store, task):
            renewals.append(task.lease_expires_at)
            renewed.set()
            await asyncio.Event().wait()
            return task

        monkeypatch.setattr(worker_module, "enforce_task_event_deadline", blocked_after_renewal)

        async def handler(_app, _task, _worker_id):
            pytest.fail("the handler must not start")

        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="renewing-worker",
                poll_interval_s=0.001,
                lease_seconds=60,
                max_tasks=1,
            )
        )
        try:
            await asyncio.wait_for(renewed.wait(), timeout=10)
            held = await store.load_task(created.id)
            assert held.status is TaskStatus.CLAIMED
            assert held.started_at is None
            # The pre-dispatch heartbeat renewed the claim's lease.
            assert renewals and held.lease_expires_at == renewals[-1]
            worker.cancel()
            with pytest.raises(asyncio.CancelledError):
                await worker
            released = await store.load_task(created.id)
            assert released.status is TaskStatus.PENDING
            assert released.worker_id is None
        finally:
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())
