"""Execution-start storage latency must not admit work after its event expires."""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from cayu import CayuApp
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks import InMemoryTaskStore, TaskCreate, TaskRetryBudget, TaskStatus, run_task_worker


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_event_expiring_during_start_acknowledgement_never_dispatches(
    backend, tmp_path, monkeypatch
):
    async def scenario():
        # Keep the event live through admission, independently of setup latency.
        now = [datetime.now(UTC)]
        store = (
            InMemoryTaskStore(clock=lambda: now[0])
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "tasks.sqlite", clock=lambda: now[0])
        )
        app = CayuApp(task_store=store, enable_logging=False)
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []
        original_start = type(store).mark_claimed_task_execution_started

        async def delayed_start(self, *args, **kwargs):
            result = await original_start(self, *args, **kwargs)
            entered.set()
            # The store has committed execution entry; its acknowledgement is delayed.
            await release.wait()
            return result

        monkeypatch.setattr(type(store), "mark_claimed_task_execution_started", delayed_start)
        now[0] = datetime.now(UTC)
        created = await store.create_task(
            TaskCreate(
                type="job",
                retry_budget=TaskRetryBudget(max_elapsed_seconds=1.0),
            )
        )

        async def handler(_app, task, _worker_id):
            # Represents the first externally visible action of a callback.
            calls.append(task.id)
            await asyncio.Event().wait()

        worker = asyncio.create_task(
            run_task_worker(app, store, handler, worker_id="owner", max_tasks=1)
        )
        try:
            await asyncio.wait_for(entered.wait(), 5)
            assert not calls
            # Expire both the worker's local budget and the durable event clock,
            # while leaving its ordinary 300-second claim comfortably live.
            await asyncio.sleep(1.1)
            now[0] += timedelta(seconds=2)
            release.set()
            assert await asyncio.wait_for(worker, 5) == 1
            final = await store.load_task(created.id)
            assert final.status in {TaskStatus.CANCELLED, TaskStatus.FAILED}
            assert calls == [], "An expired event dispatched its handler after the start ack"
        finally:
            release.set()
            if not worker.done():
                worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            await app.aclose()
            if hasattr(store, "close"):
                await store.close()

    asyncio.run(scenario())
