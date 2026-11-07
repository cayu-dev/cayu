"""A cancellation drain settles only its own claim, never a same-named later holder's."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from cayu.applications import CayuApp
from cayu.runtime._task_lease_authority import TaskLeaseAuthority
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks.creation import TaskCreate
from cayu.tasks.dispatch import TaskStoreDispatcher
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.records import TaskClaimLost, TaskStatus
from cayu.tasks.terminalization import TaskTerminalKind
from cayu.tasks.worker import (
    _request_task_cancellation_fence,
    _settle_ordinary_task_cancellation_after_quiescence,
    run_task_worker,
)

_WORKER = "shared-worker"


def _store(backend, tmp_path, now):
    if backend == "memory":
        return InMemoryTaskStore(ownership_clock=lambda: now[0])
    return SQLiteTaskStore(tmp_path / "tasks.sqlite", ownership_clock=lambda: now[0])


async def _successor_with_cancellation(store):
    """A drained claim's task, now held by a same-named later claim being cancelled."""

    await store.create_task(TaskCreate(task_id="drained", type="drain"))
    drained = await store.claim_task(_WORKER, lease_seconds=60)
    assert drained is not None
    await store.release_task(drained.id, _WORKER, claim_generation=drained.claim_generation)
    successor = await store.claim_task(_WORKER, lease_seconds=60)
    assert successor is not None and successor.claim_generation == drained.claim_generation + 2
    await store.mark_claimed_task_execution_started(
        successor.id, _WORKER, claim_generation=successor.claim_generation
    )
    await store.request_claimed_task_cancellation(
        successor.id,
        _WORKER,
        error={"reason": "operator"},
        claim_generation=successor.claim_generation,
    )
    held = await store.load_task(successor.id)
    assert held is not None and held.status_reason == "cancellation_requested"
    return drained, held


def _authority(task):
    return TaskLeaseAuthority(
        task.lease_expires_at,
        task_id=task.id,
        worker_id=_WORKER,
        claim_generation=task.claim_generation,
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_drains_leave_a_same_named_later_claim_alone(backend, tmp_path):
    async def scenario() -> None:
        now = [datetime(2026, 1, 1, tzinfo=UTC)]
        store = _store(backend, tmp_path, now)
        try:
            drained, held = await _successor_with_cancellation(store)
            with pytest.raises(TaskClaimLost, match="claim generation"):
                await _request_task_cancellation_fence(
                    store,
                    drained,
                    _WORKER,
                    {"reason": "drain"},
                    lease_authority=_authority(drained),
                )
            await _settle_ordinary_task_cancellation_after_quiescence(
                store, drained.id, _WORKER, claim_generation=drained.claim_generation
            )
            with pytest.raises(TaskClaimLost, match="claim generation"):
                await TaskStoreDispatcher(store)._settle_dispatch_cancellation_after_quiescence(
                    drained.id, _WORKER, claim_generation=drained.claim_generation
                )
            assert await store.load_task(drained.id) == held
        finally:
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("drain", ["worker", "dispatcher"])
def test_drain_settles_reclaims_fence_of_its_own_claim(backend, drain, tmp_path):
    """Reclaim fences a lost dispatched claim one generation on; its drain still settles it."""

    async def scenario() -> None:
        now = [datetime(2026, 1, 1, tzinfo=UTC)]
        store = _store(backend, tmp_path, now)
        try:
            await store.create_task(TaskCreate(task_id="lost", type="drain"))
            claimed = await store.claim_task(_WORKER, lease_seconds=10)
            assert claimed is not None
            await store.mark_claimed_task_execution_started(
                claimed.id, _WORKER, claim_generation=claimed.claim_generation
            )
            now[0] += timedelta(seconds=20)
            await store.reclaim_expired()
            fenced = await store.load_task(claimed.id)
            assert fenced is not None
            assert fenced.claim_generation == claimed.claim_generation + 1
            requested = await _request_task_cancellation_fence(
                store,
                claimed,
                _WORKER,
                {"reason": "drain"},
                lease_authority=_authority(claimed),
            )
            assert requested == fenced
            if drain == "worker":
                await _settle_ordinary_task_cancellation_after_quiescence(
                    store, claimed.id, _WORKER, claim_generation=claimed.claim_generation
                )
            else:
                await TaskStoreDispatcher(store)._settle_dispatch_cancellation_after_quiescence(
                    claimed.id, _WORKER, claim_generation=claimed.claim_generation
                )
            settled = await store.load_task(claimed.id)
            assert settled is not None and settled.status is TaskStatus.CANCELLED
        finally:
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())


def test_cancellation_drain_settles_under_its_running_heartbeat():
    """A settlement slower than the lease still commits as the worker's own write."""

    class SlowCancellationStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        async def terminalize_task(self, request):
            if request.kind is TaskTerminalKind.CANCELLED:
                await asyncio.sleep(2.5)
            return await super().terminalize_task(request)

    async def scenario() -> None:
        store = SlowCancellationStore()
        app = CayuApp(task_store=store, enable_logging=False)
        await store.create_task(TaskCreate(task_id="drained-slowly", type="job"))
        handler_started = asyncio.Event()
        release_handler = asyncio.Event()

        async def handler(_app, _task, _worker_id) -> None:
            handler_started.set()
            await release_handler.wait()

        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id=_WORKER,
                lease_seconds=1,
                poll_interval_s=0.001,
                reclaim=False,
                max_tasks=1,
            )
        )
        await asyncio.wait_for(handler_started.wait(), timeout=5)
        # Shutdown fences the claim with a cancellation request and drains the handler.
        worker.cancel()
        await asyncio.sleep(0.1)
        release_handler.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(worker, timeout=15)
        settled = await store.load_task("drained-slowly")
        assert settled is not None and settled.status is TaskStatus.CANCELLED
        # The worker's own terminalization committed, not an owner-lost reconciliation.
        assert "cancellation_reconciliation" not in (settled.status_payload or {})

    asyncio.run(scenario())
