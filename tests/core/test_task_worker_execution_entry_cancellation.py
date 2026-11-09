"""A worker cancelled while it marks a claim's execution started returns the claim.

No handler owns a claim until its execution-start mark is acknowledged. A
worker cancelled while that mark is in flight never dispatches the handler,
so the claim's own execution settlement must prove that, acknowledge any group
execution the mark opened, and release the claim for the next worker instead of
leaving it held by the exited worker until its lease expires.
"""

from __future__ import annotations

import asyncio
import contextlib
from enum import StrEnum

import pytest

from cayu import CayuApp
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks import (
    TaskCreate,
    TaskGraphCreate,
    TaskGraphNode,
    TaskGroupCreate,
    TaskGroupPolicy,
    TaskGroupQuiescencePolicy,
    TaskStatus,
    run_task_worker,
)
from cayu.tasks.memory import InMemoryTaskStore

_WORKER_TIMEOUT_S = 10


async def _create(app: CayuApp, store, *, grouped: bool) -> None:
    if not grouped:
        await store.create_task(TaskCreate(task_id="job", type="job"))
        return
    await app.create_task_group(
        TaskGroupCreate(
            group_id="jobs",
            graph=TaskGraphCreate(
                graph_id="jobs-graph",
                nodes=(
                    TaskGraphNode(task=TaskCreate(task_id="job", type="job")),
                    TaskGraphNode(task=TaskCreate(task_id="finalize", type="finalize")),
                ),
            ),
            member_task_ids=("job",),
            policy=TaskGroupPolicy(kind="all"),
            quiescence=TaskGroupQuiescencePolicy(timeout_seconds=60),
            finalizer_task_id="finalize",
        )
    )


class _EntryFault(StrEnum):
    # The worker is cancelled before or after the mark commits; either way the
    # mark itself runs to completion in its own task.
    CANCEL_ENTERED = "cancel-mark-entered"
    CANCEL_COMMITTED = "cancel-mark-committed"
    # The worker is cancelled while the mark fails without committing, so the
    # entry reads the claim back.
    RAISE = "cancel-mark-raises"


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("grouped", [False, True], ids=["task", "group-member"])
@pytest.mark.parametrize("fault", list(_EntryFault), ids=str)
def test_failed_execution_entry_releases_the_claim(tmp_path, backend, grouped, fault):
    async def scenario() -> None:
        base = InMemoryTaskStore if backend == "memory" else SQLiteTaskStore
        worker: list[asyncio.Task[int]] = []

        class FailingEntry(base):
            verified_work_mutations_are_cancellation_quiescent = True

            async def mark_claimed_task_execution_started(self, *args, **kwargs):
                if fault is _EntryFault.RAISE:
                    worker[0].cancel()
                    raise OSError("execution-start mark unavailable")
                if fault is _EntryFault.CANCEL_ENTERED:
                    worker[0].cancel()
                started = await super().mark_claimed_task_execution_started(*args, **kwargs)
                if fault is _EntryFault.CANCEL_COMMITTED:
                    worker[0].cancel()
                return started

        store = FailingEntry(*(() if backend == "memory" else (tmp_path / "t.sqlite",)))
        app = CayuApp(task_store=store, enable_logging=False)
        calls: list[str] = []

        async def handler(_app, task, _worker_id):
            calls.append(task.id)

        try:
            await _create(app, store, grouped=grouped)
            worker.append(asyncio.create_task(_run_worker(app, store, handler)))
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(worker[0], _WORKER_TIMEOUT_S)
            task = await store.load_task("job")
            assert calls == []
            assert (task.status, task.worker_id, task.started_at) == (
                TaskStatus.PENDING,
                None,
                None,
            )
            if grouped:
                snapshot = await app.load_task_group("jobs")
                executions = snapshot.quiescence.executions
                if fault is _EntryFault.RAISE:
                    assert executions == ()
                else:
                    (execution,) = executions
                    assert execution.worker_id == "owner"
                    assert execution.settled_at is not None
        finally:
            await app.aclose()
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_a_claim_whose_cancellation_was_requested_stays_with_its_drain(tmp_path, backend):
    async def scenario() -> None:
        base = InMemoryTaskStore if backend == "memory" else SQLiteTaskStore
        released: list[str] = []

        class CancelledBeforeEntry(base):
            verified_work_mutations_are_cancellation_quiescent = True

            async def mark_claimed_task_execution_started(
                self, task_id, worker_id, *, claim_generation=None
            ):
                # Cancellation is requested between the pre-dispatch checks and the
                # mark, so the mark refuses to start the claim.
                await self.request_claimed_task_cancellation(
                    task_id,
                    worker_id,
                    error={"code": "cancelled_before_entry"},
                    claim_generation=claim_generation,
                )
                return await super().mark_claimed_task_execution_started(
                    task_id, worker_id, claim_generation=claim_generation
                )

            async def release_task(self, task_id, *args, **kwargs):
                released.append(task_id)
                return await super().release_task(task_id, *args, **kwargs)

        store = CancelledBeforeEntry(*(() if backend == "memory" else (tmp_path / "t.sqlite",)))
        app = CayuApp(task_store=store, enable_logging=False)
        calls: list[str] = []

        async def handler(_app, task, _worker_id):
            calls.append(task.id)

        try:
            await _create(app, store, grouped=False)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(_run_worker(app, store, handler), _WORKER_TIMEOUT_S)
            assert calls == []
            # The cancellation drain keeps the claim; the release never tries it.
            assert released == []
        finally:
            await app.aclose()
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())


async def _run_worker(app: CayuApp, store, handler) -> int:
    return await run_task_worker(
        app,
        store,
        handler,
        worker_id="owner",
        max_tasks=1,
        reclaim=False,
        recover_interrupted_handoffs=False,
    )
