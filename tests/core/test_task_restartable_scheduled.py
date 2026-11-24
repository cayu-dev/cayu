"""Restartable tasks that are scheduled, or queued once per key with retries."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks import (
    TaskCreate,
    TaskSchedulePolicy,
    create_task_once,
    reconcile_abandoned_tasks,
)
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.queries import TaskQuery
from cayu.tasks.records import Task, TaskStatus
from cayu.tasks.scheduling import TaskScheduleEventType
from cayu.tasks.store import TaskStore

LEASE_SECONDS = 30
RESTART = {
    "live_worker_ids": ["worker-live"],
    "validator_id": "service.single-worker",
    "validator_version": "1",
    "settlement": "restart",
}


class _Clock:
    def __init__(self) -> None:
        self.value = datetime.now(UTC).replace(microsecond=0) - timedelta(hours=1)

    def __call__(self) -> datetime:
        return self.value


def _store(backend: str, tmp_path: Path, clock: _Clock) -> TaskStore:
    if backend == "memory":
        return InMemoryTaskStore(clock=clock, ownership_clock=clock)
    return SQLiteTaskStore(tmp_path / "tasks.db", clock=clock, ownership_clock=clock)


async def _close(store: TaskStore) -> None:
    close = getattr(store, "close", None)
    if close is not None:
        await close()


async def _claim_and_start(store: TaskStore, task_type: str, worker_id: str) -> Task:
    claimed = await store.claim_task(
        worker_id, TaskQuery(type=task_type), lease_seconds=LEASE_SECONDS
    )
    assert claimed is not None and claimed.lease_expires_at is not None
    await store.mark_claimed_task_execution_started(claimed.id, worker_id, claimed.lease_expires_at)
    return claimed


async def _lose_worker(store: TaskStore, clock: _Clock, task_type: str, worker_id: str) -> Task:
    claimed = await _claim_and_start(store, task_type, worker_id)
    assert claimed.lease_expires_at is not None
    clock.value = claimed.lease_expires_at + timedelta(seconds=1)
    await store.reclaim_expired(query=TaskQuery(type=task_type))
    fenced = await store.load_task(claimed.id)
    assert fenced is not None and fenced.status_reason == "cancellation_requested"
    return fenced


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_a_scheduled_task_restarts_in_the_occurrence_it_was_admitted_to(
    backend: str, tmp_path: Path
) -> None:
    async def scenario() -> None:
        clock = _Clock()
        store = _store(backend, tmp_path, clock)
        due = clock.value
        created = await store.create_task(
            TaskCreate(
                task_id="review-1",
                type="review",
                available_at=due,
                # Expires before the restart: admission happened once, so it still runs.
                schedule_policy=TaskSchedulePolicy(expires_at=due + timedelta(seconds=10)),
                restartable=True,
                max_restarts=1,
            )
        )
        assert created.restartable and created.schedule is not None
        fenced = await _lose_worker(store, clock, "review", "worker-a")
        assert fenced.schedule is not None and fenced.schedule.admitted_at == due

        [restarted] = await reconcile_abandoned_tasks(store, **RESTART)
        assert restarted.status is TaskStatus.PENDING and restarted.restart_count == 1
        assert restarted.available_at == due
        assert restarted.schedule == fenced.schedule

        claimed = await _claim_and_start(store, "review", "worker-b")
        assert claimed.id == "review-1" and claimed.restart_count == 1
        done = await store.complete_task(
            claimed.id,
            {"ok": True},
            worker_id="worker-b",
            claim_generation=claimed.claim_generation,
        )
        assert done.status is TaskStatus.COMPLETED
        events = [event.type for event in await store.list_task_schedule_events("review-1")]
        assert events[0] is TaskScheduleEventType.SCHEDULED
        assert events.count(TaskScheduleEventType.ELIGIBLE) == 1
        assert events.count(TaskScheduleEventType.STARTED) == 2
        assert events[-1] is TaskScheduleEventType.COMPLETED
        await _close(store)

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_a_once_per_key_task_restarts_then_retries_as_a_new_attempt(
    backend: str, tmp_path: Path
) -> None:
    async def scenario() -> None:
        clock = _Clock()
        store = _store(backend, tmp_path, clock)
        request = TaskCreate(type="reset", retry_key="reset-owner-1", restartable=True)
        first = await create_task_once(store, request, attempts=2)
        assert first.created and first.task is not None and first.task.restartable

        # A lost worker restarts the same attempt; the key still holds that task.
        await _lose_worker(store, clock, "reset", "worker-a")
        [restarted] = await reconcile_abandoned_tasks(store, **RESTART)
        assert restarted.id == first.task.id and restarted.status is TaskStatus.PENDING
        same = await create_task_once(store, request)
        assert not same.created and same.task is not None and same.task.id == first.task.id
        assert same.key is not None and same.key.attempts == 1

        # A failure uses the key's next attempt, which is restartable too.
        claimed = await _claim_and_start(store, "reset", "worker-b")
        await store.fail_task(
            claimed.id,
            {"message": "failed"},
            worker_id="worker-b",
            claim_generation=claimed.claim_generation,
        )
        retry = await create_task_once(store, request)
        assert retry.created and retry.task is not None
        assert retry.task.id != first.task.id and retry.task.restartable
        assert retry.key is not None and retry.key.attempts == 2
        await _close(store)

    asyncio.run(scenario())


def test_a_restartable_task_still_refuses_a_retry_series() -> None:
    with pytest.raises(ValueError, match=r"create_task_once\(attempts=N\)"):
        TaskCreate(type="job", restartable=True, retry_policy={"max_attempts": 2})
