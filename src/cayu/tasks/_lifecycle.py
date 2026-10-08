"""Task lifecycle transitions, exact lease checks and attachment authority."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from cayu._clock import normalize_utc_datetime
from cayu._validation import copy_durable_json_object
from cayu._validation import require_durable_nonblank as require_nonblank
from cayu.tasks._cancellation import _validate_ordinary_task_terminalization_against_cancellation
from cayu.tasks.cancellation import _task_cancellation_requested, _task_retry_cancellation_requested
from cayu.tasks.records import (
    _HELD_TASK_STATUSES,
    _TERMINAL_TASK_STATUSES,
    Task,
    TaskClaimLost,
    TaskRetrySeriesSnapshot,
    TaskStatus,
)
from cayu.tasks.scheduling import TaskScheduleConflict
from cayu.tasks.terminalization import (
    TaskTerminalizationConflict,
    TaskTerminalizationRequest,
    TaskTerminalKind,
)


def _ensure_can_transition(task: Task, next_status: TaskStatus) -> None:
    if (
        task.schedule is not None
        and task.schedule.admitted_at is None
        and next_status is TaskStatus.RUNNING
    ):
        raise TaskScheduleConflict("Managed schedules must be claimed before execution starts.")
    _ensure_task_status_can_transition(task.id, task.status, next_status)


def _ensure_retry_series_queue_attempt(
    retry_series: TaskRetrySeriesSnapshot | None,
) -> None:
    if retry_series is not None:
        raise ValueError(
            "Retry-series attempts are settled by task workers and cannot attach to sessions."
        )


def _ensure_task_status_can_transition(
    task_id: str,
    status: TaskStatus,
    next_status: TaskStatus,
) -> None:
    if status in {
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
        TaskStatus.DEPENDENCY_SKIPPED,
    }:
        raise ValueError(f"Task {task_id} is already terminal: {status}")
    if next_status == TaskStatus.RUNNING and status != TaskStatus.PENDING:
        raise ValueError(f"Task {task_id} cannot transition to running from {status}")


def _ensure_can_hold_task(task: Task, next_status: TaskStatus) -> None:
    if next_status not in _HELD_TASK_STATUSES:
        raise ValueError(f"Task {task.id} cannot be held as {next_status}.")
    _ensure_not_terminal(task)
    if _task_retry_cancellation_requested(task) or _task_cancellation_requested(task):
        raise TaskTerminalizationConflict(
            "Task cancellation is still draining under its current owner."
        )
    if task.status is TaskStatus.RUNNING and task.session_id is not None:
        raise ValueError(f"Task {task.id} is already attached to session {task.session_id}.")
    if task.status not in {
        TaskStatus.PENDING,
        TaskStatus.WAITING_DEPENDENCIES,
        TaskStatus.WAITING_GROUP,
        TaskStatus.CLAIMED,
        TaskStatus.RUNNING,
        *_HELD_TASK_STATUSES,
    }:
        raise ValueError(f"Task {task.id} cannot transition to {next_status} from {task.status}")


def _ensure_can_resume_task(task: Task) -> None:
    _ensure_not_terminal(task)
    if task.status not in _HELD_TASK_STATUSES:
        raise ValueError(f"Task {task.id} is not paused, blocked, or waiting for attention.")


def _ensure_not_terminal(task: Task) -> None:
    if task.status in _TERMINAL_TASK_STATUSES:
        raise ValueError(f"Task {task.id} is already terminal: {task.status}")


def _can_attach_claimed_task(
    task: Task,
    *,
    worker_id: str,
    now: datetime,
) -> bool:
    return _can_attach_claimed_task_state(
        status=task.status,
        session_id=task.session_id,
        worker_id=task.worker_id,
        lease_expires_at=task.lease_expires_at,
        expected_worker_id=worker_id,
        now=now,
    )


def _can_attach_claimed_task_state(
    *,
    status: TaskStatus,
    session_id: str | None,
    worker_id: str | None,
    lease_expires_at: datetime | None,
    expected_worker_id: str,
    now: datetime,
) -> bool:
    return (
        status is TaskStatus.CLAIMED
        and worker_id == expected_worker_id
        and session_id is None
        and lease_expires_at is not None
        and lease_expires_at > now
    )


def _ensure_active_task_lease(task: Task, worker_id: str, *, now: datetime) -> None:
    if task.lease_expires_at is None:
        raise TaskClaimLost(f"Task {task.id} has no active lease.")
    if task.lease_expires_at <= now:
        raise TaskClaimLost(f"Task {task.id} lease for worker {worker_id} has expired.")


def _ensure_owned_active_task_lease(
    task: Task,
    worker_id: str,
    *,
    now: datetime,
) -> None:
    """Require the supplied worker to own the task's current live lease."""

    if task.status not in {TaskStatus.CLAIMED, TaskStatus.RUNNING}:
        raise TaskClaimLost(f"Task {task.id} is not claimed or running.")
    if task.worker_id != worker_id:
        raise TaskClaimLost(f"Worker {worker_id} does not own task {task.id}.")
    _ensure_active_task_lease(task, worker_id, now=now)


def _ensure_exact_owned_active_task_lease(
    task: Task,
    worker_id: str,
    lease_expires_at: datetime,
    *,
    now: datetime,
) -> None:
    """Require one exact live worker-lease generation."""

    expected_lease = normalize_utc_datetime(lease_expires_at, "lease_expires_at")
    _ensure_owned_active_task_lease(task, worker_id, now=now)
    if task.lease_expires_at != expected_lease:
        raise TaskClaimLost(f"Worker {worker_id} does not own task {task.id} lease generation.")


def _ensure_task_terminalization_lease_authority(
    task: Task,
    request: TaskTerminalizationRequest,
    *,
    now: datetime,
) -> None:
    """Fence reclaimable claims exactly while preserving attached-task handoffs."""

    _ensure_owned_active_task_lease(task, request.worker_id, now=now)
    if request.lease_expires_at is not None:
        _ensure_exact_owned_active_task_lease(
            task,
            request.worker_id,
            request.lease_expires_at,
            now=now,
        )
        return
    if task.status is TaskStatus.CLAIMED and task.session_id is None:
        raise TaskClaimLost(
            "Unattached task terminalization requires the exact worker lease generation."
        )


def _ensure_recovered_attached_task_session(
    task: Task,
    *,
    session_id: str,
    session_instance_id: str,
) -> None:
    """Require a task to belong to one exact durable session incarnation."""

    if task.session_id != session_id or task.session_instance_id != session_instance_id:
        raise TaskClaimLost(
            "Attached-task recovery no longer owns the expected session incarnation."
        )


def _ensure_recovered_attached_task_failure_authority(
    task: Task,
    request: TaskTerminalizationRequest,
    *,
    session_id: str,
    session_instance_id: str,
    now: datetime,
) -> None:
    """Fence recovery failure to one exact expired attached-task generation."""

    now = normalize_utc_datetime(now, "now")
    _ensure_recovered_attached_task_session(
        task,
        session_id=session_id,
        session_instance_id=session_instance_id,
    )
    if request.kind is not TaskTerminalKind.FAILED:
        raise ValueError("Attached-task recovery can only publish task failure.")
    if task.retry_series is not None:
        raise ValueError(
            "Retry-series tasks require settle_task_retry_attempt for completion or failure."
        )
    if task.status is not TaskStatus.RUNNING:
        raise TaskClaimLost("Attached-task recovery requires a running task.")
    if (
        task.worker_id != request.worker_id
        or task.lease_expires_at is None
        or request.lease_expires_at is None
        or task.lease_expires_at != request.lease_expires_at
    ):
        raise TaskClaimLost(
            "Attached-task recovery no longer owns the expected worker lease generation."
        )
    _ensure_task_handoff_authority(task, request.handoff_id)
    if task.lease_expires_at > now:
        raise TaskClaimLost("Attached-task recovery owner lease is still active.")
    _validate_ordinary_task_terminalization_against_cancellation(task, request)


def _ensure_task_handoff_authority(task: Task, handoff_id: str | None) -> None:
    """Fence a task mutation to the exact continuation generation.

    A worker identifier is deliberately insufficient: an operator may reuse one
    stable worker name after a lease handoff. Both ``None`` and non-null values
    therefore compare exactly against the stored generation.
    """

    if task.interrupted_handoff_id != handoff_id:
        raise TaskClaimLost(
            f"Worker {task.worker_id} does not own task {task.id} handoff generation."
        )


def _require_active_attached_task_worker(
    task: Task,
    *,
    worker_id: str,
    session_id: str,
    session_instance_id: str,
    now: datetime,
) -> Task:
    """Validate and detach one store-authoritative resume owner snapshot."""

    _ensure_owned_active_task_lease(task, worker_id, now=now)
    if (
        task.status is not TaskStatus.RUNNING
        or task.session_id != session_id
        or task.session_instance_id != session_instance_id
    ):
        raise TaskClaimLost(f"Worker {worker_id} does not own the requested attached task session.")
    return task.model_copy(deep=True)


def _require_direct_attached_task_resume(
    task: Task,
    *,
    session_id: str,
    session_instance_id: str,
) -> Task:
    """Validate a workerless direct attachment with no handoff generation."""

    if (
        task.status is not TaskStatus.RUNNING
        or task.session_id != session_id
        or task.session_instance_id != session_instance_id
        or task.worker_id is not None
        or task.lease_expires_at is not None
        or task.interrupted_handoff_id is not None
    ):
        raise TaskClaimLost(
            "Ordinary resume does not own the requested direct attached task session."
        )
    return task.model_copy(deep=True)


def _raise_task_claim_attach_error(
    task: Task,
    worker_id: str,
    *,
    now: datetime,
) -> None:
    if task.status not in {TaskStatus.CLAIMED, TaskStatus.RUNNING}:
        raise TaskClaimLost(f"Task {task.id} is not claimed by worker {worker_id}.")
    _ensure_owned_active_task_lease(task, worker_id, now=now)
    if task.status is TaskStatus.RUNNING:
        if task.session_id is not None:
            raise ValueError(f"Task {task.id} is already attached to session {task.session_id}.")
        raise ValueError(f"Task {task.id} is already running.")
    if task.session_id is not None:
        raise ValueError(f"Task {task.id} is already attached to session {task.session_id}.")
    raise RuntimeError(f"Task {task.id} active claim could not be attached.")


def _copy_optional_status_reason(value: str | None) -> str | None:
    if value is None:
        return None
    return require_nonblank(value, "reason")


def _copy_optional_status_payload(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return copy_durable_json_object(value, "payload")
