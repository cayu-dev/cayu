"""A generation-fenced cancellation ignores a lease it also presents."""

import asyncio

import pytest

from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks import TaskCreate, TaskStatus
from cayu.tasks._cancellation import _task_cancellation_terminalization_request
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.records import TaskRetryPolicy
from cayu.tasks.retry import _task_retry_requested_cancellation_settlement


def _store(backend, tmp_path):
    return InMemoryTaskStore() if backend == "memory" else SQLiteTaskStore(tmp_path / "tasks.db")


async def _claimed_with_cancellation(store, **create):
    await store.create_task(TaskCreate(task_id="job", type="job", **create))
    claimed = await store.claim_task("owner", lease_seconds=60)
    assert claimed is not None
    await store.mark_claimed_task_execution_started(
        claimed.id, "owner", claim_generation=claimed.claim_generation
    )
    await store.cancel_task(claimed.id, {"code": "operator"})
    task = await store.load_task(claimed.id)
    assert task.status is TaskStatus.CLAIMED
    return task


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_ordinary_cancellation_with_generation_and_lease_commits(backend, tmp_path):
    async def scenario():
        store = _store(backend, tmp_path)
        try:
            task = await _claimed_with_cancellation(store)
            request = _task_cancellation_terminalization_request(
                task, worker_id="owner", claim_generation=task.claim_generation
            )
            assert request is not None
            # The caller also carries the claim's current lease, which the
            # generation fence ignores.
            request = request.model_copy(update={"lease_expires_at": task.lease_expires_at})
            terminal = await store.terminalize_task(request)
            assert terminal.status is TaskStatus.CANCELLED
        finally:
            if isinstance(store, SQLiteTaskStore):
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_retry_cancellation_with_generation_and_lease_commits(backend, tmp_path):
    async def scenario():
        store = _store(backend, tmp_path)
        try:
            task = await _claimed_with_cancellation(
                store, retry_policy=TaskRetryPolicy(max_attempts=2)
            )
            request = _task_retry_requested_cancellation_settlement(
                task, worker_id="owner", claim_generation=task.claim_generation
            )
            assert request is not None
            request = request.model_copy(update={"lease_expires_at": task.lease_expires_at})
            await store.settle_task_retry_attempt(request)
            assert (await store.load_task(task.id)).status is TaskStatus.CANCELLED
        finally:
            if isinstance(store, SQLiteTaskStore):
                await store.close()

    asyncio.run(scenario())
