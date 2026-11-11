"""A hold reason is caller text; it must not read as a cancellation fence.

``pause_task``, ``block_task`` and ``mark_task_needs_attention`` store the
caller's ``reason`` in ``Task.status_reason``. The cancellation fence is the
same field holding the literal ``"cancellation_requested"`` (or
``"retry_cancellation_requested"``) on a claimed or running task. A held task
that has no worker and no cancel request must stay cancellable and re-holdable
whatever its reason text is.

``assert_hold_reason_cancellation_fence_conformance`` runs the same contract,
and that a real fence still refuses, on every store including Postgres.
"""

from __future__ import annotations

import asyncio

import pytest

from cayu.storage.tasks_sqlite import SQLiteTaskStore
from cayu.tasks.cancellation import _task_cancellation_requested, _task_retry_cancellation_requested
from cayu.tasks.creation import TaskCreate
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.records import TaskStatus
from cayu.tasks.worker import _settle_task_refused_by_shutdown

_FENCE_REASONS = ("cancellation_requested", "retry_cancellation_requested")


@pytest.fixture(params=["memory", "sqlite"])
def store_factory(request, tmp_path):
    if request.param == "memory":
        return lambda: InMemoryTaskStore()
    return lambda: SQLiteTaskStore(tmp_path / "tasks.db")


async def _close(store) -> None:
    close = getattr(store, "close", None)
    if close is not None:
        await close()


@pytest.mark.parametrize("reason", _FENCE_REASONS)
@pytest.mark.parametrize("hold", ["pause_task", "block_task", "mark_task_needs_attention"])
def test_held_task_with_fence_text_reason_can_still_be_cancelled(store_factory, hold, reason):
    async def scenario():
        store = store_factory()
        try:
            task = await store.create_task(TaskCreate(type="job"))
            held = await getattr(store, hold)(task.id, reason=reason)
            # Nobody owns the task and nobody asked for cancellation.
            assert held.worker_id is None
            assert held.lease_expires_at is None
            assert held.status_reason == reason

            cancelled = await store.cancel_task(task.id)
            assert cancelled.status is TaskStatus.CANCELLED
        finally:
            await _close(store)

    asyncio.run(scenario())


@pytest.mark.parametrize("reason", _FENCE_REASONS)
def test_held_task_with_fence_text_reason_can_be_held_again(store_factory, reason):
    async def scenario():
        store = store_factory()
        try:
            task = await store.create_task(TaskCreate(type="job"))
            await store.pause_task(task.id, reason=reason)

            blocked = await store.block_task(task.id, reason="waiting for a human")
            assert blocked.status is TaskStatus.BLOCKED
            assert blocked.status_reason == "waiting for a human"
        finally:
            await _close(store)

    asyncio.run(scenario())


@pytest.mark.parametrize("reason", _FENCE_REASONS)
def test_fence_predicates_read_the_text_only_on_a_claimed_or_running_task(reason):
    async def scenario():
        store = InMemoryTaskStore()
        task = await store.create_task(TaskCreate(type="job"))
        held = await store.pause_task(task.id, reason=reason)
        for status in TaskStatus:
            candidate = held.model_copy(update={"status": status})
            fenced = _task_cancellation_requested(candidate) or (
                _task_retry_cancellation_requested(candidate)
            )
            assert fenced is (status in {TaskStatus.CLAIMED, TaskStatus.RUNNING})

    asyncio.run(scenario())


@pytest.mark.parametrize("reason", _FENCE_REASONS)
def test_shutdown_refusal_leaves_a_task_held_with_fence_text_reason_to_its_hold(reason):
    async def scenario():
        store = InMemoryTaskStore()
        task = await store.create_task(TaskCreate(type="job"))
        assert await store.claim_task("worker-a") is not None
        await store.pause_task(task.id, reason=reason)

        # No cancellation is draining: nothing is left for the failure path to settle.
        # The hold released the claim, so the lease authority is never consulted.
        assert await _settle_task_refused_by_shutdown(store, task.id, "worker-a", None) is True
        current = await store.load_task(task.id)
        assert (current.status, current.status_reason) == (TaskStatus.PAUSED, reason)

    asyncio.run(scenario())
