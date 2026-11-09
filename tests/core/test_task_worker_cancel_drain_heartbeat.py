"""A slow cancellation-drain renewal must not hold the handler's interrupt timer."""

from __future__ import annotations

import asyncio

import pytest

from cayu.applications import CayuApp
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks import (
    InMemoryTaskStore,
    TaskCreate,
    TaskRetryPolicy,
    current_task_cancellation,
    run_task_worker,
)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("retry_series", [False, True])
def test_cancel_grace_expires_while_drain_heartbeat_is_waiting(
    backend, retry_series, tmp_path, monkeypatch
):
    async def scenario():
        store = (
            InMemoryTaskStore()
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "tasks.sqlite")
        )
        app = CayuApp(task_store=store, enable_logging=False)
        task = await store.create_task(
            TaskCreate(
                type="job",
                retry_policy=TaskRetryPolicy(max_attempts=2) if retry_series else None,
            )
        )
        entered = asyncio.Event()
        signalled = asyncio.Event()
        interrupted = asyncio.Event()
        drain_heartbeat_waiting = asyncio.Event()
        release_heartbeat = asyncio.Event()
        release_handler = asyncio.Event()
        original_heartbeat = type(store).heartbeat

        async def heartbeat(self, *args, **kwargs):
            # Only stall a renewal after the main heartbeat has already observed
            # the request and handed supervision over to the cancellation drain.
            if self is store and signalled.is_set():
                drain_heartbeat_waiting.set()
                await release_heartbeat.wait()
            return await original_heartbeat(self, *args, **kwargs)

        monkeypatch.setattr(type(store), "heartbeat", heartbeat)

        async def handler(_app, _task, _worker_id):
            signal = current_task_cancellation()
            assert signal is not None
            entered.set()
            try:
                await signal.wait()
                signalled.set()
                await release_handler.wait()
            except asyncio.CancelledError:
                interrupted.set()
                raise

        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="cancel-owner",
                max_tasks=1,
                lease_seconds=3,
                poll_interval_s=0.01,
                reclaim=False,
                recover_interrupted_handoffs=False,
                interrupt_on_cancel=True,
                execution_grace_s=1.2,
            )
        )
        interruption_waiter = asyncio.create_task(interrupted.wait())
        try:
            await asyncio.wait_for(entered.wait(), 5)
            await store.cancel_task(task.id, {"code": "operator_cancelled"})
            await asyncio.wait_for(signalled.wait(), 5)
            await asyncio.wait_for(drain_heartbeat_waiting.wait(), 5)
            # The drain renews after 1s, before its 1.2s grace expires. The
            # interrupt must arrive independently of that blocked store call.
            done, _ = await asyncio.wait({interruption_waiter}, timeout=0.8)
            assert done, (
                "A pending drain heartbeat prevented the expired cancellation "
                "grace from interrupting the handler"
            )
        finally:
            release_handler.set()
            release_heartbeat.set()
            interruption_waiter.cancel()
            await asyncio.gather(worker, interruption_waiter, return_exceptions=True)
            await app.aclose()
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())
