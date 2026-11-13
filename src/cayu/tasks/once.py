"""Queue a task once per key: one call, no exceptions to catch for a key already used."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from cayu.tasks.creation import TaskCreate
from cayu.tasks.records import Task, TaskStatus
from cayu.tasks.retry_budgets import (
    TaskEventAlreadyCompleted,
    TaskRetryBudget,
    TaskRetryBudgetExhausted,
    TaskRetryKeyState,
    TaskRetryKeyStatus,
)
from cayu.tasks.store import TaskStore

# A key's task that ended without completing; another may be queued while the key's
# budget allows.
_ENDED_WITHOUT_COMPLETING = frozenset(
    {TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.DEPENDENCY_SKIPPED}
)


@dataclass(frozen=True)
class TaskOnce:
    """What ``create_task_once`` did.

    ``task`` is the task that holds the key: the one this call created (``created``),
    or the one already queued, running, completed or failed for it. It is ``None`` only
    when that task was deleted or this caller can't read it. ``key`` is the key's
    budget and counts (``TaskStore.load_task_retry_key``).
    """

    task: Task | None
    created: bool
    key: TaskRetryKeyState | None


async def create_task_once(
    task_store: TaskStore, request: TaskCreate, *, attempts: int = 1
) -> TaskOnce:
    """Queue ``request`` unless its ``retry_key`` already has a task; return the task.

    The key (``TaskCreate.retry_key``, such as ``"digest-2026-11-12-am"``) holds one
    task at a time. While that task is queued or running, and once it completed,
    calling again returns it and creates nothing, also when several callers race. When
    it failed, was cancelled or was skipped, the next call queues another task for the
    key, up to ``attempts`` tasks in all; past that it returns the last one. With the
    default of one attempt, a key whose task failed stays failed: queue again under a
    new key. The first use of the key fixes its budget (``TaskRetryBudget(max_attempts=
    attempts)``, or the request's own ``retry_budget``), as ``create_task`` does; later
    calls keep it, whatever ``attempts`` they pass.

    The task IDs come from the key and the attempt number, so two callers queueing the
    same attempt collide on the ID and only one task is made. The request can't name
    its own ``task_id`` for that reason.
    """

    key = request.retry_key
    if key is None:
        raise ValueError("create_task_once needs the request's retry_key.")
    if type(attempts) is not int or attempts < 1:
        raise ValueError("attempts must be a whole number of at least 1.")
    if request.task_id is not None:
        raise ValueError("create_task_once chooses the task ID from the key; leave task_id unset.")
    state = await task_store.load_task_retry_key(key)
    # The key's first admission fixed its budget; a later call must send the same one.
    budget = (
        state.budget
        if state is not None
        else request.retry_budget
        or TaskRetryBudget(max_attempts=attempts, max_elapsed_seconds=None)
    )
    if state is not None:
        holder = await task_store.load_task(state.latest_task_id)
        if not _admits_another(state, holder):
            return TaskOnce(task=holder, created=False, key=state)
    attempt = 1 if state is None else state.attempts + 1
    task_id = _attempt_task_id(key, attempt)
    try:
        task = await task_store.create_task(
            request.model_copy(update={"task_id": task_id, "retry_budget": budget})
        )
    except (TaskRetryBudgetExhausted, TaskEventAlreadyCompleted) as exc:
        holder = await task_store.load_task(exc.state.latest_task_id)
        return TaskOnce(task=holder, created=False, key=exc.state)
    except ValueError:
        # Another caller queued this attempt first. The key may have moved on to a later
        # attempt since, so confirm the collision by the task itself, not by the key's
        # latest task, then return the task that holds the key now.
        existing = await task_store.load_task(task_id)
        if existing is None or existing.invocation.event_retry_key != key:
            raise
        state = await task_store.load_task_retry_key(key)
        if state is None or state.latest_task_id == task_id:
            return TaskOnce(task=existing, created=False, key=state)
        holder = await task_store.load_task(state.latest_task_id)
        return TaskOnce(task=holder, created=False, key=state)
    return TaskOnce(task=task, created=True, key=await task_store.load_task_retry_key(key))


def _admits_another(state: TaskRetryKeyState, holder: Task | None) -> bool:
    if state.status is not TaskRetryKeyStatus.ACTIVE or holder is None:
        return False
    remaining = state.attempts_remaining
    return holder.status in _ENDED_WITHOUT_COMPLETING and (remaining is None or remaining > 0)


def _attempt_task_id(key: str, attempt: int) -> str:
    base = "once-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
    return base if attempt == 1 else f"{base}-{attempt}"


__all__ = ["TaskOnce", "create_task_once"]
