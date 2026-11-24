"""``cancel_tasks``: cancel what a task query matches, and wait until it has settled."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from cayu.applications import CayuApp
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks.bulk_cancellation import cancel_tasks
from cayu.tasks.creation import TaskCreate
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.queries import TaskQuery
from cayu.tasks.records import TaskStatus
from cayu.tasks.store import TaskStore
from cayu.tasks.worker import run_task_worker


def _store(backend: str, tmp_path: Path) -> TaskStore:
    return InMemoryTaskStore() if backend == "memory" else SQLiteTaskStore(tmp_path / "t.db")


async def _seed(store: TaskStore) -> None:
    for task_id, labels in (
        ("job-a", {"job": "close"}),
        ("job-b", {"job": "close"}),
        ("other", {"job": "open"}),
    ):
        await store.create_task(TaskCreate(task_id=task_id, type="review", labels=labels))


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_idle_matches_are_cancelled_at_once_and_others_left(backend: str, tmp_path: Path) -> None:
    async def scenario() -> None:
        store = _store(backend, tmp_path)
        await _seed(store)
        outcome = await cancel_tasks(
            store, TaskQuery(labels={"job": "close"}), error={"code": "reset", "message": "x"}
        )
        assert sorted(outcome.settled) == ["job-a", "job-b"]
        assert outcome.done
        statuses = {task.id: task.status for task in await store.list_tasks(TaskQuery())}
        assert statuses == {
            "job-a": TaskStatus.CANCELLED,
            "job-b": TaskStatus.CANCELLED,
            "other": TaskStatus.PENDING,
        }
        # Finished tasks are not asked again.
        again = await cancel_tasks(store, TaskQuery(labels={"job": "close"}))
        assert again.settled == () and again.done

    asyncio.run(scenario())


def test_a_running_match_is_waited_for_until_its_worker_stops_it(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        await store.create_task(TaskCreate(task_id="long", type="review", labels={"job": "close"}))
        started = asyncio.Event()

        async def handler(_app: CayuApp, _task: object, _worker_id: str) -> None:
            started.set()
            await asyncio.Event().wait()

        stop = asyncio.Event()
        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="worker",
                query=TaskQuery(type="review"),
                lease_seconds=3,
                poll_interval_s=0.05,
                # The handler is cancelled half a second after the request reaches it.
                execution_timeout_s=600,
                execution_grace_s=0.5,
                stop=stop,
            )
        )
        try:
            await asyncio.wait_for(started.wait(), 10)
            waited = await cancel_tasks(
                store, TaskQuery(labels={"job": "close"}), wait_seconds=30, poll_seconds=0.05
            )
            assert waited.settled == ("long",) and waited.done
            task = await store.load_task("long")
            assert task is not None and task.status == TaskStatus.CANCELLED
        finally:
            stop.set()
            await asyncio.wait_for(worker, 30)
            await app.aclose()

    asyncio.run(scenario())


def test_without_a_wait_a_claimed_match_is_reported_stopping() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        await store.create_task(TaskCreate(task_id="long", type="review", labels={"job": "close"}))
        claimed = await store.claim_task("worker", TaskQuery(type="review"), lease_seconds=300)
        assert claimed is not None
        await store.mark_claimed_task_execution_started("long", "worker", claimed.lease_expires_at)
        outcome = await cancel_tasks(
            store, TaskQuery(labels={"job": "close"}), wait_seconds=0.2, poll_seconds=0.05
        )
        assert outcome.stopping == ("long",) and not outcome.done
        task = await store.load_task("long")
        assert task is not None and task.status_reason == "cancellation_requested"
        # Asked again, it is still stopping, not refused.
        again = await cancel_tasks(store, TaskQuery(labels={"job": "close"}))
        assert again.stopping == ("long",) and again.refused == ()

    asyncio.run(scenario())


def test_a_task_that_finished_after_it_was_listed_counts_as_settled() -> None:
    class FinishesFirst(InMemoryTaskStore):
        async def cancel_task(self, task_id, error=None):
            # Settled by something else (a prerequisite's cancellation, its worker)
            # between the listing and this call.
            await super().cancel_task(task_id, error)
            raise ValueError("Task is already terminal.")

    async def scenario() -> None:
        store = FinishesFirst()
        await _seed(store)
        outcome = await cancel_tasks(store, TaskQuery(labels={"job": "close"}))
        assert sorted(outcome.settled) == ["job-a", "job-b"] and outcome.done

    asyncio.run(scenario())


def test_matches_past_the_first_page_are_found_while_earlier_ones_leave() -> None:
    from cayu.tasks import bulk_cancellation

    async def scenario() -> None:
        store = InMemoryTaskStore()
        for index in range(5):
            await store.create_task(TaskCreate(task_id=f"t{index}", type="review"))
        original = store.query_task_index
        reads = 0

        async def index_then_claim(query, **kwargs):
            nonlocal reads
            reads += 1
            page = await original(query, **kwargs)
            if reads == 1:
                # A worker takes the first match between pages.
                await store.claim_task("worker", TaskQuery(type="review"), lease_seconds=300)
            return page

        store.query_task_index = index_then_claim  # type: ignore[method-assign]
        bulk_cancellation._PAGE = 2
        try:
            outcome = await cancel_tasks(store, TaskQuery(type="review", status=TaskStatus.PENDING))
        finally:
            bulk_cancellation._PAGE = 500
        handled = {*outcome.settled, *outcome.stopping, *(r.task_id for r in outcome.refused)}
        assert handled == {f"t{index}" for index in range(5)}

    asyncio.run(scenario())


def test_more_matches_than_max_tasks_say_so() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        await _seed(store)
        outcome = await cancel_tasks(store, TaskQuery(type="review"), max_tasks=2)
        assert len(outcome.settled) == 2 and outcome.has_more and not outcome.done
        rest = await cancel_tasks(store, TaskQuery(type="review"), max_tasks=2)
        assert len(rest.settled) == 1 and rest.done

    asyncio.run(scenario())


@pytest.mark.parametrize("arguments", [{"max_tasks": 0}, {"wait_seconds": -1}, {"poll_seconds": 0}])
def test_bad_bounds_are_refused(arguments: dict) -> None:
    query = arguments.pop("query", TaskQuery())
    with pytest.raises(ValueError):
        asyncio.run(cancel_tasks(InMemoryTaskStore(), query, **arguments))
