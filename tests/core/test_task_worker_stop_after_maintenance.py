"""A stop observed during maintenance must prevent the next fresh claim."""

from __future__ import annotations

import asyncio

import pytest

import cayu.tasks.worker as task_worker
from cayu.applications import CayuApp
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks import InMemoryTaskStore, TaskCreate, TaskStatus, complete_managed_task
from cayu.tasks.handoff import InterruptedTaskContinuationClaimPage
from cayu.tasks.queries import TaskQuery
from cayu.tasks.worker import run_task_worker


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("continued", [False, True])
def test_stop_during_reclamation_does_not_dispatch_fresh_work(
    backend, tmp_path, monkeypatch, continued
):
    async def scenario():
        base = InMemoryTaskStore if backend == "memory" else SQLiteTaskStore
        maintaining = asyncio.Event()
        release = asyncio.Event()

        class BlockingMaintenanceStore(base):
            verified_work_mutations_are_cancellation_quiescent = True
            supports_interrupted_task_handoffs = True

            async def reclaim_expired(self, **kwargs):
                maintaining.set()
                await release.wait()
                return await super().reclaim_expired(**kwargs)

        store = BlockingMaintenanceStore(
            *(() if backend == "memory" else (tmp_path / "tasks.sqlite",))
        )
        app = CayuApp(task_store=store, enable_logging=False)
        created = await store.create_task(TaskCreate(type="job"))
        stop = asyncio.Event()
        ran = []

        async def handler(_app, task, worker_id):
            ran.append(task.id)
            await complete_managed_task(store, task, worker_id, {})

        async def continuation_scan(*args, **kwargs):
            resumed = await store.create_task(TaskCreate(type="continuation"))
            claimed = await store.claim_task(
                "owner", TaskQuery(type="continuation"), lease_seconds=60
            )
            assert claimed is not None and claimed.id == resumed.id
            await complete_managed_task(store, claimed, "owner", {})
            return InterruptedTaskContinuationClaimPage(
                task=resumed,
                next_after=(resumed.created_at, resumed.id),
                scanned_candidates=1,
                rejected_candidates=0,
                exhausted=True,
            )

        if continued:
            monkeypatch.setattr(
                task_worker, "_run_recovered_interrupted_task_handler", continuation_scan
            )

        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="owner",
                stop=stop,
                recover_interrupted_handoffs=False,
                recovered_interrupted_task_handler=handler if continued else None,
            )
        )
        try:
            await asyncio.wait_for(maintaining.wait(), 5)
            stop.set()
            release.set()
            handled = await asyncio.wait_for(worker, 5)
            current = await store.load_task(created.id)
            assert (handled, ran, current.status) == (int(continued), [], TaskStatus.PENDING)
        finally:
            stop.set()
            release.set()
            if not worker.done():
                worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            await app.aclose()
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())
