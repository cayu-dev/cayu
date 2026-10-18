"""Explicit requeueing of positively quiescent, restartable queue jobs."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from cayu.tasks._lifecycle import _ensure_exact_owned_active_task_lease
from cayu.tasks.cancellation import _task_cancellation_requested, _task_retry_cancellation_requested
from cayu.tasks.records import Task, TaskStatus, copy_task
from cayu.tasks.terminalization import TaskTerminalizationConflict

if TYPE_CHECKING:
    from cayu.tasks.store import TaskStore


def _restarted_task(
    task: Task, worker_id: str, lease_expires_at: datetime, *, now: datetime
) -> Task:
    _ensure_exact_owned_active_task_lease(task, worker_id, lease_expires_at, now=now)
    if not task.restartable or task.interrupted_handoff_id is not None:
        raise ValueError("Only ordinary restartable root queue jobs can be requeued.")
    if _task_cancellation_requested(task) or _task_retry_cancellation_requested(task):
        raise TaskTerminalizationConflict("An explicit cancellation must not restart a task.")
    return copy_task(
        task.model_copy(
            update={
                "status": TaskStatus.PENDING,
                "session_id": None,
                "session_instance_id": None,
                "worker_id": None,
                "lease_expires_at": None,
                "started_at": None,
                "completed_at": None,
                "result": None,
                "error": None,
                "status_reason": None,
                "status_payload": {
                    "restart": {
                        "count": task.restart_count + 1,
                        "previous_session_id": task.session_id,
                    }
                },
                "restart_count": task.restart_count + 1,
                "updated_at": now,
            }
        )
    )


async def requeue_restartable_task(task_store: TaskStore, task: Task, worker_id: str) -> Task:
    """Requeue a handler the caller has already stopped and fully joined.

    This does not cancel or join work for the caller. Keep renewing ownership
    during shutdown; the exact current lease must still be live. A handler must
    leave no detached work capable of producing further effects. Operator
    cancellation, lost leases and non-restartable tasks are refused. Use a fresh
    session ID on the next attempt; ``restart_count`` distinguishes attempts.
    The generic worker performs this shutdown sequence automatically.
    """
    if task.lease_expires_at is None:
        raise ValueError("Restarting a task requires its exact worker lease.")
    return await task_store.restart_task(task.id, worker_id, lease_expires_at=task.lease_expires_at)
