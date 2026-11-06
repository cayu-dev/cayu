"""A handler exception still proves its task-group execution has stopped."""

import asyncio

import pytest

from cayu.applications import CayuApp
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks import (
    TaskCreate,
    TaskGraphCreate,
    TaskGraphNode,
    TaskGroupCreate,
    TaskGroupFinalizerStatus,
    TaskGroupPolicy,
    TaskGroupQuiescencePolicy,
    TaskGroupQuiescenceStatus,
    TaskQuery,
    TaskRetryPolicy,
    TaskStatus,
)
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.worker import run_task_worker


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("retry_series", [False, True])
@pytest.mark.parametrize("raises", [False, True])
def test_failed_handler_releases_group_quiescence_barrier(backend, retry_series, raises, tmp_path):
    async def scenario():
        store = (
            InMemoryTaskStore()
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "tasks.sqlite")
        )
        app = CayuApp(task_store=store, enable_logging=False)
        try:
            await app.create_task_group(
                TaskGroupCreate(
                    group_id="race",
                    graph=TaskGraphCreate(
                        graph_id="race-graph",
                        nodes=(
                            TaskGraphNode(
                                task=TaskCreate(
                                    task_id="failed-member",
                                    type="failing",
                                    retry_policy=(
                                        TaskRetryPolicy(max_attempts=2) if retry_series else None
                                    ),
                                )
                            ),
                            TaskGraphNode(task=TaskCreate(task_id="winner", type="winner")),
                            TaskGraphNode(task=TaskCreate(task_id="finalizer", type="finalizer")),
                        ),
                    ),
                    member_task_ids=("failed-member", "winner"),
                    policy=TaskGroupPolicy(kind="first_success"),
                    quiescence=TaskGroupQuiescencePolicy(timeout_seconds=60),
                    finalizer_task_id="finalizer",
                )
            )
            stopped = asyncio.Event()

            async def handler(_app, _task, _worker_id):
                try:
                    if raises:
                        raise RuntimeError("member failed before producing a result")
                    # Control: the worker fails an unfinished return too, but
                    # that path must acknowledge exactly the same stopped work.
                    return None
                finally:
                    stopped.set()

            assert (
                await run_task_worker(
                    app,
                    store,
                    handler,
                    worker_id="group-owner",
                    max_tasks=1,
                    query=TaskQuery(type="failing"),
                )
                == 1
            )
            assert stopped.is_set()
            assert (await store.load_task("failed-member")).status is TaskStatus.FAILED
            await store.complete_task("winner", {"ok": True})
            snapshot = await app.reconcile_task_group("race")
            assert snapshot.quiescence.status is TaskGroupQuiescenceStatus.QUIESCENT, (
                snapshot.quiescence
            )
            assert all(item.settled_at is not None for item in snapshot.quiescence.executions)
            assert snapshot.quiescence.finalizer_status is TaskGroupFinalizerStatus.RELEASED
            assert (await store.load_task("finalizer")).status is TaskStatus.PENDING
        finally:
            await app.aclose()
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())
