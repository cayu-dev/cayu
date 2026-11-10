"""Operator cancellation still reaches a retry handler draining worker shutdown."""

from __future__ import annotations

import asyncio

import pytest

import cayu.tasks.worker as worker_module
from cayu.applications import CayuApp
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks import (
    InMemoryTaskStore,
    TaskCreate,
    TaskRetryAttemptDisposition,
    TaskRetryAttemptReport,
    TaskRetryBudget,
    TaskRetryPolicy,
    TaskStatus,
    current_task_cancellation,
    run_task_worker,
)
from cayu.tasks.cancellation import _task_retry_cancellation_requested


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("shutdown_first", [False, True])
def test_operator_request_notifies_retry_handler_during_owner_shutdown(
    backend, shutdown_first, tmp_path, monkeypatch
):
    async def scenario():
        store = (
            InMemoryTaskStore()
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "tasks.sqlite")
        )
        app = CayuApp(task_store=store, enable_logging=False)
        created = await store.create_task(
            TaskCreate(
                type="job",
                retry_policy=TaskRetryPolicy(max_attempts=2),
                # This exercises shutdown, not the separate retry-deadline drain.
                retry_budget=TaskRetryBudget(max_elapsed_seconds=None),
            )
        )
        entered = asyncio.Event()
        shutdown_drain_entered = asyncio.Event()
        cancellation_renewed_twice = asyncio.Event()
        signalled = asyncio.Event()
        rescue = asyncio.Event()
        signals = []
        cancellation_renewals = 0
        original_heartbeat = store.heartbeat
        original_shutdown_drain = worker_module._raise_task_retry_owner_cancellation

        async def heartbeat(*args, **kwargs):
            nonlocal cancellation_renewals
            task = await original_heartbeat(*args, **kwargs)
            if _task_retry_cancellation_requested(task):
                cancellation_renewals += 1
                if cancellation_renewals >= 2:
                    cancellation_renewed_twice.set()
            return task

        async def observe_shutdown_drain(*args, **kwargs):
            shutdown_drain_entered.set()
            return await original_shutdown_drain(*args, **kwargs)

        monkeypatch.setattr(store, "heartbeat", heartbeat)
        monkeypatch.setattr(
            worker_module, "_raise_task_retry_owner_cancellation", observe_shutdown_drain
        )

        async def handler(_app, _task, _worker_id):
            signal = current_task_cancellation()
            assert signal is not None
            signals.append(signal)
            entered.set()
            signal_wait = asyncio.create_task(signal.wait())
            rescue_wait = asyncio.create_task(rescue.wait())
            try:
                await asyncio.wait((signal_wait, rescue_wait), return_when=asyncio.FIRST_COMPLETED)
                if signal.requested:
                    signalled.set()
                return TaskRetryAttemptReport(
                    idempotency_key="shutdown-handler-stopped",
                    disposition=TaskRetryAttemptDisposition.CANCELLED,
                    error={"code": "handler_stopped"},
                )
            finally:
                signal_wait.cancel()
                rescue_wait.cancel()
                await asyncio.gather(signal_wait, rescue_wait, return_exceptions=True)

        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="shutdown-owner",
                max_tasks=1,
                lease_seconds=3,
                poll_interval_s=0.01,
                reclaim=False,
                recover_interrupted_handoffs=False,
                execution_grace_s=0.1,
            )
        )
        signal_waiter = asyncio.create_task(signalled.wait())
        renewal_waiter = asyncio.create_task(cancellation_renewed_twice.wait())
        try:
            await asyncio.wait_for(entered.wait(), 5)
            if shutdown_first:
                worker.cancel("worker shutdown")
                await asyncio.wait_for(shutdown_drain_entered.wait(), 5)
            requested = await store.cancel_task(created.id, {"code": "operator_cancelled"})
            assert _task_retry_cancellation_requested(requested)
            done, _ = await asyncio.wait(
                (signal_waiter, renewal_waiter),
                return_when=asyncio.FIRST_COMPLETED,
                timeout=5,
            )
            assert done, "Neither cancellation delivery nor lease renewal made progress"
            # Give the heartbeat's caller a turn if it just renewed the row.
            await asyncio.sleep(0)
            assert signals[0].requested, (
                "The shutdown drain renewed an operator-cancelled attempt twice "
                "without notifying its still-running handler"
            )
            if shutdown_first:
                with pytest.raises(asyncio.CancelledError, match="worker shutdown"):
                    await asyncio.wait_for(worker, 5)
            else:
                assert await asyncio.wait_for(worker, 5) == 1
            assert (await store.load_task(created.id)).status is TaskStatus.CANCELLED
        finally:
            rescue.set()
            signal_waiter.cancel()
            renewal_waiter.cancel()
            await asyncio.gather(worker, signal_waiter, renewal_waiter, return_exceptions=True)
            await app.aclose()
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())
