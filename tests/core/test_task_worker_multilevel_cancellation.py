"""Operator interrupts must cross every nested worker without aborting cleanup."""

from __future__ import annotations

import asyncio

import pytest

from cayu import CayuApp
from cayu.deadlines import current_handler_time_limit
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks import (
    InMemoryTaskStore,
    TaskCreate,
    TaskQuery,
    TaskStatus,
    complete_managed_task,
    run_task_worker,
)


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("operator_cancel", [False, True])
@pytest.mark.parametrize(
    ("restartable", "interrupt_on_cancel"), [(False, True), (True, True), (False, False)]
)
async def test_multilevel_worker_cancellation_preserves_cleanup(
    backend, operator_cancel, restartable, interrupt_on_cancel, tmp_path
):
    store = (
        InMemoryTaskStore() if backend == "memory" else SQLiteTaskStore(tmp_path / "tasks.sqlite")
    )
    app = CayuApp(task_store=store, enable_logging=False)
    root = await store.create_task(TaskCreate(type="root"))
    await store.create_task(TaskCreate(type="middle"))
    await store.create_task(TaskCreate(type="leaf", restartable=restartable))
    entered, interrupted, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    cleanup_started, release_cleanup = asyncio.Event(), asyncio.Event()
    cleanup_completed, cleanup_interrupted = asyncio.Event(), asyncio.Event()

    async def work(kind, handler):
        return await run_task_worker(
            app,
            store,
            handler,
            worker_id=kind,
            query=TaskQuery(type=kind),
            max_tasks=1,
            reclaim=False,
            recover_interrupted_handoffs=False,
            execution_grace_s=0,
            interrupt_on_cancel=interrupt_on_cancel if kind == "leaf" else True,
        )

    async def leaf(_app, task, worker_id):
        entered.set()
        try:
            await release.wait()
            await complete_managed_task(store, task, worker_id, {"done": True})
        except asyncio.CancelledError:
            interrupted.set()
            raise
        finally:
            cleanup_started.set()
            try:
                await release_cleanup.wait()
                cleanup_completed.set()
            except asyncio.CancelledError:
                cleanup_interrupted.set()
                raise

    async def middle(_app, task, worker_id):
        await work("leaf", leaf)
        await complete_managed_task(store, task, worker_id, {"done": True})

    async def outer(_app, task, worker_id):
        await work("middle", middle)
        await complete_managed_task(store, task, worker_id, {"done": True})

    worker = asyncio.create_task(work("root", outer))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        if operator_cancel:
            await store.cancel_task(root.id, {"code": "operator_cancelled"})
            # Only root's signal is requested. Middle's unrequested binding
            # must not hide that provenance from the leaf supervisor.
            done, _ = await asyncio.wait({worker}, timeout=2)
            assert interrupted.is_set() is interrupt_on_cancel
            assert cleanup_started.is_set() is interrupt_on_cancel
            assert not done
            worker.cancel("shutdown during cleanup")
        else:
            worker.cancel("worker shutdown")
        done, _ = await asyncio.wait({worker}, timeout=0.1)
        assert not done and not cleanup_interrupted.is_set()
        assert interrupted.is_set() is (operator_cancel and interrupt_on_cancel)
        release.set()
        release_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(worker, 5)
        assert cleanup_completed.is_set() and not cleanup_interrupted.is_set()
        assert (await store.load_task(root.id)).status is TaskStatus.CANCELLED
    finally:
        release.set()
        release_cleanup.set()
        await asyncio.wait_for(asyncio.gather(worker, return_exceptions=True), 10)
        await app.aclose()
        if backend == "sqlite":
            await store.close()


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
async def test_nested_operator_interrupt_disarms_leaf_cleanup_deadline(backend, tmp_path):
    store = (
        InMemoryTaskStore() if backend == "memory" else SQLiteTaskStore(tmp_path / "tasks.sqlite")
    )
    app = CayuApp(task_store=store, enable_logging=False)
    root = await store.create_task(TaskCreate(type="root"))
    child = await store.create_task(TaskCreate(type="leaf"))
    entered, cleaning, release_cleanup = asyncio.Event(), asyncio.Event(), asyncio.Event()
    cleaned, cleanup_interrupted = asyncio.Event(), asyncio.Event()
    limits = []

    async def leaf(_app, _task, _worker_id):
        limit = current_handler_time_limit()
        limits.append(limit)
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            try:
                await release_cleanup.wait()
                cleaned.set()
            except asyncio.CancelledError:
                cleanup_interrupted.set()
                raise

    async def work(kind, handler):
        return await run_task_worker(
            app,
            store,
            handler,
            worker_id=kind,
            query=TaskQuery(type=kind),
            max_tasks=1,
            reclaim=False,
            recover_interrupted_handoffs=False,
            execution_timeout_s=5 if kind == "leaf" else None,
            execution_grace_s=0,
        )

    async def outer(_app, _task, _worker_id):
        await work("leaf", leaf)

    worker = asyncio.create_task(work("root", outer))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        await store.cancel_task(root.id, {"code": "operator_cancelled"})
        await asyncio.wait_for(cleaning.wait(), 5)
        limit = limits[0]
        assert not limit.cancelled, "The operator must interrupt before the leaf deadline"
        # Keep cleanup running beyond the original hard deadline. The soft
        # deadline must still expire, but it must not deliver a second cancel.
        done, _ = await asyncio.wait({worker}, timeout=limit.remaining_seconds() + 0.1)
        assert limit.expired and not limit.cancelled
        assert not cleanup_interrupted.is_set()
        assert not done, "The worker must retain the leaf until cleanup finishes"
        held = await store.load_task(child.id)
        assert held.status is TaskStatus.CLAIMED and held.worker_id == "leaf"
        release_cleanup.set()
        assert await asyncio.wait_for(worker, 5) == 1
        assert cleaned.is_set()
        assert (await store.load_task(root.id)).status is TaskStatus.CANCELLED
        assert (await store.load_task(child.id)).status is TaskStatus.CANCELLED
    finally:
        release_cleanup.set()
        await asyncio.wait_for(asyncio.gather(worker, return_exceptions=True), 10)
        await app.aclose()
        if backend == "sqlite":
            await store.close()
