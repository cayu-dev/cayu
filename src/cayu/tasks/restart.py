"""Requeueing restartable queue jobs: after a quiescent shutdown, or after worker loss."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from cayu._clock import normalize_utc_datetime
from cayu._validation import require_clean_nonblank
from cayu.tasks._cancellation import _task_cancellation_terminalization_request
from cayu.tasks._lifecycle import _ensure_exact_owned_active_task_lease
from cayu.tasks.cancellation import _task_cancellation_requested, _task_retry_cancellation_requested
from cayu.tasks.records import (
    TASK_MAX_RESTARTS_LIMIT,
    Task,
    TaskClaimLost,
    TaskRestartReason,
    TaskRestartRecord,
    TaskStatus,
    _validate_task_retry_reconciliation_identity,
    copy_task,
)
from cayu.tasks.terminalization import TaskTerminalizationConflict

if TYPE_CHECKING:
    from cayu.tasks.store import TaskStore

#: Restarts after worker loss a task may use when neither it nor the caller sets a cap.
DEFAULT_MAX_ABANDONED_RESTARTS = 2


class TaskRestartLimitReached(Exception):
    """A restartable task already used every restart after worker loss it may have."""

    def __init__(self, task_id: str, *, restarts: int, max_restarts: int) -> None:
        super().__init__(
            f"Task {task_id} was already restarted {restarts} time(s) after worker loss, "
            f"its limit of {max_restarts}."
        )
        self.task_id = task_id
        self.restarts = restarts
        self.max_restarts = max_restarts


class AbandonedTaskRestartRequest(BaseModel):
    """Requeue the exact fenced claim of a worker the caller knows is gone.

    ``max_restarts`` applies when the task sets no ``max_restarts`` of its own.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    task_id: str
    original_worker_id: str
    original_lease_expires_at: datetime
    validator_id: str
    validator_version: str
    evidence_id: str
    max_restarts: StrictInt = Field(
        default=DEFAULT_MAX_ABANDONED_RESTARTS, ge=0, le=TASK_MAX_RESTARTS_LIMIT
    )

    @field_validator(
        "task_id", "original_worker_id", "validator_id", "validator_version", "evidence_id"
    )
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _validate_task_retry_reconciliation_identity(value, info.field_name)

    @field_validator("original_lease_expires_at")
    @classmethod
    def normalize_lease(cls, value: datetime) -> datetime:
        return normalize_utc_datetime(value, "original_lease_expires_at")


def copy_abandoned_task_restart_request(
    request: AbandonedTaskRestartRequest,
) -> AbandonedTaskRestartRequest:
    if type(request) is not AbandonedTaskRestartRequest:
        raise TypeError("request must be an AbandonedTaskRestartRequest.")
    return AbandonedTaskRestartRequest.model_validate(request.model_dump())


def task_restart_limit(task: Task, default: int) -> int:
    """The restarts after worker loss ``task`` may use: its own cap, or ``default``."""

    return default if task.max_restarts is None else task.max_restarts


def abandoned_restarts_used(task: Task) -> int:
    """How many times ``task`` was restarted after worker loss."""

    return len(task.restart_history)


#: Fence error codes that only the task worker or expired-claim reclamation
#: record. Any other code, such as ``cancel_task``'s ``"cancelled"``, a handler
#: time limit or an error the caller passed, means someone asked to cancel the
#: task, and that request must not be undone by a restart.
_WORKER_LOSS_FENCE_CODES = frozenset(
    {
        "task_worker_lease_expired_after_dispatch",
        "task_worker_owner_cancelled",
        "task_worker_lease_authority_lost",
    }
)


def fenced_by_worker_loss(task: Task) -> bool:
    """Whether ``task``'s cancellation fence was left by worker loss, not asked for."""

    payload = task.status_payload
    error = payload.get("error") if type(payload) is dict else None
    code = error.get("code") if type(error) is dict else None
    return type(code) is str and code in _WORKER_LOSS_FENCE_CODES


def _same_evidence(record: TaskRestartRecord, request: AbandonedTaskRestartRequest) -> bool:
    return (record.validator_id, record.validator_version, record.evidence_id) == (
        request.validator_id,
        request.validator_version,
        request.evidence_id,
    )


def _requeued_task(task: Task, *, now: datetime, restart: dict[str, object]) -> Task:
    return task.model_copy(
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
                    **restart,
                }
            },
            "restart_count": task.restart_count + 1,
            "updated_at": now,
        }
    )


def _restarted_task(
    task: Task, worker_id: str, lease_expires_at: datetime, *, now: datetime
) -> Task:
    _ensure_exact_owned_active_task_lease(task, worker_id, lease_expires_at, now=now)
    if not task.restartable or task.interrupted_handoff_id is not None:
        raise ValueError("Only ordinary restartable root queue jobs can be requeued.")
    if _task_cancellation_requested(task) or _task_retry_cancellation_requested(task):
        raise TaskTerminalizationConflict("An explicit cancellation must not restart a task.")
    return copy_task(_requeued_task(task, now=now, restart={}))


def _abandoned_restarted_task(
    task: Task, request: AbandonedTaskRestartRequest, *, now: datetime
) -> Task:
    """Requeue a fenced task whose expired claim ``request`` names, or raise.

    Returns ``task`` itself, unchanged, when this exact restart already happened.
    """

    now = normalize_utc_datetime(now, "now")
    if task.id != request.task_id:
        raise TaskTerminalizationConflict("Abandoned task restart names a different task.")
    for record in task.restart_history:
        if not _same_evidence(record, request):
            continue
        if (record.previous_worker_id, record.previous_lease_expires_at) == (
            request.original_worker_id,
            request.original_lease_expires_at,
        ):
            return task
        raise TaskTerminalizationConflict(
            "This evidence already restarted the task after a different worker claim."
        )
    if task.retry_series is not None or task.session_id is not None:
        raise TaskTerminalizationConflict(
            "Only an ordinary task without a session can restart after worker loss."
        )
    if not task.restartable or task.interrupted_handoff_id is not None:
        raise TaskTerminalizationConflict(
            "Only a task created with restartable=True can restart after worker loss."
        )
    if (
        task.status not in {TaskStatus.CLAIMED, TaskStatus.RUNNING}
        or task.worker_id != request.original_worker_id
        or task.lease_expires_at != request.original_lease_expires_at
    ):
        raise TaskClaimLost("The task is no longer held by the abandoned worker claim.")
    if (
        _task_cancellation_terminalization_request(task, worker_id=request.original_worker_id)
        is None
    ):
        raise TaskTerminalizationConflict(
            "Only a task fenced after its worker was lost can restart after worker loss."
        )
    if not fenced_by_worker_loss(task):
        raise TaskTerminalizationConflict(
            "The task's cancellation was requested explicitly, so it must not restart."
        )
    if request.original_lease_expires_at > now:
        raise TaskClaimLost("The abandoned worker's lease has not expired yet.")
    limit = task_restart_limit(task, request.max_restarts)
    used = abandoned_restarts_used(task)
    if used >= limit:
        raise TaskRestartLimitReached(task.id, restarts=used, max_restarts=limit)
    record = TaskRestartRecord(
        restarted_at=now,
        reason=TaskRestartReason.WORKER_LOST,
        previous_worker_id=request.original_worker_id,
        previous_lease_expires_at=request.original_lease_expires_at,
        validator_id=request.validator_id,
        validator_version=request.validator_version,
        evidence_id=request.evidence_id,
    )
    requeued = _requeued_task(
        task,
        now=now,
        restart={
            "reason": record.reason.value,
            "previous_worker_id": record.previous_worker_id,
            "previous_lease_expires_at": record.previous_lease_expires_at.isoformat(),
            "validator_id": record.validator_id,
            "validator_version": record.validator_version,
            "evidence_id": record.evidence_id,
        },
    )
    return copy_task(
        requeued.model_copy(update={"restart_history": (*task.restart_history, record)})
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


async def restart_abandoned_task(
    task_store: TaskStore,
    task_id: str,
    *,
    validator_id: str,
    validator_version: str,
    evidence_id: str,
    max_restarts: int = DEFAULT_MAX_ABANDONED_RESTARTS,
) -> Task:
    """Requeue a restartable task fenced after its worker was lost; return it ``pending``.

    The restart counterpart of ``reconcile_abandoned_task_cancellation``, with
    the same preconditions: call it only once the application knows the
    original worker can no longer act. ``validator_id`` and
    ``validator_version`` name the check that made that judgement and
    ``evidence_id`` the fact it relied on. The store still refuses while the
    original lease is live.

    Only an ordinary task created with ``restartable=True`` and without an
    attached session restarts. It goes back to ``pending`` with
    ``restart_count`` one higher, a :class:`cayu.tasks.TaskRestartRecord`
    appended to ``restart_history``, and ``status_payload["restart"]`` naming
    the previous worker, the validator and the evidence, so the next handler
    can tell it is a restart. ``max_restarts`` caps restarts after worker loss
    when the task sets no ``max_restarts`` of its own; past it this raises
    :class:`TaskRestartLimitReached` and changes nothing, and the caller settles
    the task another way, for example with
    ``reconcile_abandoned_task_cancellation``. Restarts from
    ``requeue_restartable_task`` after a quiescent shutdown don't count.

    Only a fence that worker loss left is restarted: if someone asked to
    cancel the task (``cancel_task``, a handler time limit, or any other
    cancellation error), the request is refused, and the task is cancelled as
    asked. Calling it again with the same evidence after it succeeded returns
    the task unchanged, whoever holds it now.
    """

    task_id = require_clean_nonblank(task_id, "task_id")
    for name, value in (
        ("validator_id", validator_id),
        ("validator_version", validator_version),
        ("evidence_id", evidence_id),
    ):
        require_clean_nonblank(value, name)
    current = await task_store.load_task(task_id)
    if current is None:
        raise TaskTerminalizationConflict("Abandoned task restart lost its task.")
    if any(
        (record.validator_id, record.validator_version, record.evidence_id)
        == (validator_id, validator_version, evidence_id)
        for record in current.restart_history
    ):
        return current
    return await restart_observed_abandoned_task(
        task_store,
        current,
        validator_id=validator_id,
        validator_version=validator_version,
        evidence_id=evidence_id,
        max_restarts=max_restarts,
    )


async def restart_observed_abandoned_task(
    task_store: TaskStore,
    observed: Task,
    *,
    validator_id: str,
    validator_version: str,
    evidence_id: str,
    max_restarts: int,
) -> Task:
    """Restart the exact fenced claim ``observed`` shows; a changed claim is refused."""

    if observed.worker_id is None or observed.lease_expires_at is None:
        raise TaskTerminalizationConflict(
            "Only a claimed task fenced after worker loss can restart after worker loss."
        )
    return await task_store.restart_abandoned_task(
        AbandonedTaskRestartRequest(
            task_id=observed.id,
            original_worker_id=observed.worker_id,
            original_lease_expires_at=observed.lease_expires_at,
            validator_id=validator_id,
            validator_version=validator_version,
            evidence_id=evidence_id,
            max_restarts=max_restarts,
        )
    )
