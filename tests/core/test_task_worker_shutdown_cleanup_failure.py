"""Retry shutdown settles operator cancellation even when handler cleanup fails."""

from __future__ import annotations

import asyncio

import pytest

import cayu.tasks.worker as worker_module
from cayu._exception_groups import iter_exception_tree
from cayu.applications import CayuApp
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks import (
    InMemoryTaskStore,
    TaskCreate,
    TaskRetryBudget,
    TaskRetryPolicy,
    TaskStatus,
    current_task_cancellation,
    run_task_worker,
)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("interrupt", [False, True])
def test_retry_shutdown_preserves_cleanup_failure_and_settles_cancellation(
    backend, grouped, interrupt, tmp_path, monkeypatch
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
                retry_budget=TaskRetryBudget(max_elapsed_seconds=None),
            )
        )
        entered = asyncio.Event()
        draining = asyncio.Event()
        stopped = asyncio.Event()
        rescue = asyncio.Event()
        handler_tasks = []
        signals = []
        cleanup_failure = RuntimeError("handler cleanup failed")
        original_drain = worker_module._raise_task_retry_owner_cancellation

        async def observe_drain(*args, **kwargs):
            draining.set()
            return await original_drain(*args, **kwargs)

        monkeypatch.setattr(worker_module, "_raise_task_retry_owner_cancellation", observe_drain)

        async def handler(_app, _task, _worker_id):
            handler_tasks.append(asyncio.current_task())
            signal = current_task_cancellation()
            assert signal is not None
            signals.append(signal)
            entered.set()
            try:
                await signal.wait()
                if interrupt:
                    await rescue.wait()
            except asyncio.CancelledError as cancellation:
                if grouped:
                    raise BaseExceptionGroup(
                        "handler cancellation and cleanup failed", [cancellation, cleanup_failure]
                    ) from None
                raise cleanup_failure from None
            else:
                if grouped:
                    raise BaseExceptionGroup(
                        "handler cancellation and cleanup failed",
                        [asyncio.CancelledError("handler operation cancelled"), cleanup_failure],
                    )
                raise cleanup_failure
            finally:
                stopped.set()

        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="shutdown-owner",
                max_tasks=1,
                lease_seconds=3,
                reclaim=False,
                recover_interrupted_handoffs=False,
                execution_grace_s=0.1,
                interrupt_on_cancel=interrupt,
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 5)
            worker.cancel("worker shutdown")
            await asyncio.wait_for(draining.wait(), 5)
            operator_error = {"code": "operator_cancelled"}
            await store.cancel_task(created.id, operator_error)
            # Do not let wait_for cancel the worker again and hide the original
            # cancellation's secondary evidence.
            done, _ = await asyncio.wait({worker}, timeout=5)
            assert done, "Operator cancellation did not finish the shutdown drain"
            with pytest.raises(asyncio.CancelledError, match="worker shutdown") as caught:
                await worker
            assert signals[0].requested
            assert stopped.is_set()
            current = await store.load_task(created.id)
            assert current.status is TaskStatus.CANCELLED
            assert current.worker_id is None
            assert current.error == operator_error
            assert caught.value.__cause__ is not None
            assert any(
                error is cleanup_failure for error in iter_exception_tree(caught.value.__cause__)
            )
        finally:
            rescue.set()
            for handler_task in handler_tasks:
                if not handler_task.done():
                    handler_task.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            await app.aclose()
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("renewal_failure", ["none", "plain", "grouped"])
@pytest.mark.parametrize("interrupt", [False, True])
def test_retry_shutdown_supervises_handler_after_renewal_failure(
    backend, renewal_failure, interrupt, tmp_path, monkeypatch
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
                retry_budget=TaskRetryBudget(max_elapsed_seconds=None),
            )
        )
        entered = asyncio.Event()
        draining = asyncio.Event()
        renewed = asyncio.Event()
        stopped = asyncio.Event()
        rescue = asyncio.Event()
        handler_tasks = []
        signals = []
        maintenance_failure = RuntimeError("drain renewal failed")
        original_drain = worker_module._raise_task_retry_owner_cancellation
        original_heartbeat = store.heartbeat

        async def observe_drain(*args, **kwargs):
            draining.set()
            return await original_drain(*args, **kwargs)

        async def heartbeat(*args, **kwargs):
            if signals and signals[0].requested and not stopped.is_set():
                renewed.set()
                if renewal_failure == "grouped":
                    raise BaseExceptionGroup(
                        "renewal cancellation and cleanup failed",
                        [asyncio.CancelledError("renewal cancelled"), maintenance_failure],
                    )
                if renewal_failure == "plain":
                    raise maintenance_failure
            return await original_heartbeat(*args, **kwargs)

        monkeypatch.setattr(worker_module, "_raise_task_retry_owner_cancellation", observe_drain)
        monkeypatch.setattr(store, "heartbeat", heartbeat)

        async def handler(_app, _task, _worker_id):
            handler_tasks.append(asyncio.current_task())
            signal = current_task_cancellation()
            assert signal is not None
            signals.append(signal)
            entered.set()
            try:
                await rescue.wait()
            finally:
                stopped.set()

        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="shutdown-owner",
                max_tasks=1,
                lease_seconds=3,
                reclaim=False,
                recover_interrupted_handoffs=False,
                # Let a renewal fail before the grace-period interrupt is due.
                execution_grace_s=1.3,
                interrupt_on_cancel=interrupt,
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 5)
            worker.cancel("worker shutdown")
            await asyncio.wait_for(draining.wait(), 5)
            operator_error = {"code": "operator_cancelled"}
            await store.cancel_task(created.id, operator_error)
            await asyncio.wait_for(renewed.wait(), 5)
            await asyncio.sleep(0.05)
            assert not worker.done(), "Renewal failure abandoned a running handler"
            assert not stopped.is_set()
            if not interrupt:
                rescue.set()
            done, _ = await asyncio.wait({worker}, timeout=5)
            assert done, "Shutdown lost its grace-period interrupt or settlement owner"
            with pytest.raises(asyncio.CancelledError, match="worker shutdown") as caught:
                await worker
            assert stopped.is_set()
            current = await store.load_task(created.id)
            assert current.status is TaskStatus.CANCELLED
            assert current.worker_id is None
            assert current.error == operator_error
            if renewal_failure == "none":
                assert caught.value.__cause__ is None
            else:
                assert caught.value.__cause__ is not None
                assert any(
                    error is maintenance_failure
                    for failure in iter_exception_tree(caught.value.__cause__)
                    for error in iter_exception_tree(failure.__cause__ or failure)
                )
        finally:
            rescue.set()
            for handler_task in handler_tasks:
                if not handler_task.done():
                    handler_task.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            await app.aclose()
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())
