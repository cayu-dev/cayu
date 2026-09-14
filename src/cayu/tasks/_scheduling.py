"""Pure scheduling transitions shared by native task-store transactions."""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from typing import TYPE_CHECKING, Literal

from cayu._validation import (
    MAX_PORTABLE_JSON_INTEGER,
    canonical_durable_json_bytes,
    revalidate_model_input,
)
from cayu.tasks.scheduling import (
    TaskRescheduleRequest,
    TaskScheduleCancelRequest,
    TaskScheduleConflict,
    TaskScheduleEligibility,
    TaskScheduleEvent,
    TaskScheduleEventType,
    TaskScheduleReceipt,
    TaskScheduleState,
    task_schedule_eligibility,
)

if TYPE_CHECKING:
    from cayu.tasks.base import Task, TaskCreate


def schedule_creation_digest(request: TaskCreate) -> str:
    # The owning entrance supplies a detached, validated TaskCreate. Include
    # private authority as well as public content; equal untrusted origin text
    # must never replay a server-verified creation.
    document = request.model_dump(mode="json", warnings="error")
    document["runtime_source"] = request._runtime_invocation_source
    document["verified_origin"] = (
        None
        if request._verified_invocation_origin is None
        else request._verified_invocation_origin.model_dump(mode="json", warnings="error")
    )
    document["runtime_session_binding"] = (
        None
        if request._runtime_session_binding is None
        else request._runtime_session_binding.model_dump(mode="json", warnings="error")
    )
    if document["runtime_source"] is not None:
        document["runtime_source"] = str(document["runtime_source"])
    return sha256(canonical_durable_json_bytes(document, "task schedule creation")).hexdigest()


def schedule_mutation_digest(request: TaskRescheduleRequest | TaskScheduleCancelRequest) -> str:
    if type(request) not in (TaskRescheduleRequest, TaskScheduleCancelRequest):
        raise TypeError("A typed schedule mutation is required.")
    request = revalidate_model_input(request, TaskRescheduleRequest, TaskScheduleCancelRequest)
    return sha256(
        canonical_durable_json_bytes(
            {
                "kind": "reschedule" if type(request) is TaskRescheduleRequest else "cancel",
                "request": request.model_dump(mode="json", warnings="error"),
            },
            "task schedule mutation",
        )
    ).hexdigest()


def require_schedule(task: Task) -> TaskScheduleState:
    if task.schedule is None or task.available_at is None:
        raise TaskScheduleConflict("Task has no managed one-shot schedule.")
    state = revalidate_model_input(task.schedule, TaskScheduleState)
    if type(state) is not TaskScheduleState:
        raise TaskScheduleConflict("Task schedule authority is malformed.")
    return state


def schedule_revision_after(state: TaskScheduleState) -> int:
    if state.revision >= MAX_PORTABLE_JSON_INTEGER:
        raise TaskScheduleConflict("Task schedule revision is exhausted.")
    return state.revision + 1


def require_schedule_mutation(task: Task, expected_revision: int) -> TaskScheduleState:
    state = require_schedule(task)
    if state.revision != expected_revision:
        raise TaskScheduleConflict("Task schedule revision changed.")
    if task.status in {"completed", "failed", "cancelled"}:
        raise TaskScheduleConflict("Terminal task schedules cannot be changed.")
    if task.status_reason in {"cancellation_requested", "retry_cancellation_requested"}:
        raise TaskScheduleConflict("Task schedule cancellation is already settling.")
    schedule_revision_after(state)
    return state


def rescheduled_task(task: Task, request: TaskRescheduleRequest, *, now: datetime) -> Task:
    from cayu.tasks.base import _rescheduled_initial_task_retry_series

    state = require_schedule_mutation(task, request.expected_revision)
    if state.admitted_at is not None or task.worker_id is not None or task.session_id is not None:
        raise TaskScheduleConflict("An admitted task cannot be rescheduled.")
    if task.status not in {"pending", "paused", "blocked", "needs_attention"}:
        raise TaskScheduleConflict("Task is not waiting for schedule admission.")
    return task.model_copy(
        update={
            "available_at": request.available_at,
            "retry_series": _rescheduled_initial_task_retry_series(
                task, available_at=request.available_at
            ),
            "updated_at": now,
            "schedule": state.model_copy(
                update={"revision": schedule_revision_after(state), "policy": request.policy}
            ),
        },
        deep=True,
    )


def admitted_schedule(task: Task, *, now: datetime) -> TaskScheduleState | None:
    if task.schedule is None:
        return None
    state = require_schedule(task)
    if state.admitted_at is not None:
        return state
    assert task.available_at is not None
    eligibility = task_schedule_eligibility(
        available_at=task.available_at, policy=state.policy, as_of=now
    )
    if eligibility not in {TaskScheduleEligibility.ELIGIBLE, TaskScheduleEligibility.MISFIRED}:
        raise TaskScheduleConflict("Task schedule does not permit admission at the store clock.")
    return state.model_copy(
        update={"revision": schedule_revision_after(state), "admitted_at": now}, deep=True
    )


def schedule_receipt(
    task: Task,
    request: TaskRescheduleRequest | TaskScheduleCancelRequest,
    *,
    now: datetime,
    kind: Literal[
        TaskScheduleEventType.RESCHEDULED,
        TaskScheduleEventType.CANCELLATION_REQUESTED,
        TaskScheduleEventType.CANCELLED,
    ],
) -> TaskScheduleReceipt:
    state = require_schedule(task)
    assert task.available_at is not None
    return TaskScheduleReceipt(
        task_id=task.id,
        operation_id=request.operation_id,
        request_sha256=schedule_mutation_digest(request),
        expected_revision=request.expected_revision,
        schedule=state,
        available_at=task.available_at,
        committed_at=now,
        type=kind,
    )


def schedule_transition_events(
    prior: Task | None,
    current: Task,
    *,
    first_sequence: int,
    operation_id: str | None = None,
) -> list[TaskScheduleEvent]:
    """Prepare bounded records before the owning store publishes a mutation."""
    if current.schedule is None:
        return []
    state = require_schedule(current)
    previous_state = None if prior is None else require_schedule(prior)
    assert current.available_at is not None
    kinds: list[TaskScheduleEventType] = []
    if prior is None or previous_state is None:
        kinds.append(TaskScheduleEventType.SCHEDULED)
    elif (
        prior.available_at != current.available_at
        or previous_state.policy != state.policy
        or (
            previous_state.revision != state.revision
            and prior.status == current.status
            and prior.status_reason == current.status_reason
            and previous_state.admitted_at == state.admitted_at
        )
    ):
        kinds.append(TaskScheduleEventType.RESCHEDULED)
    if state.admitted_at is not None and (
        previous_state is None or previous_state.admitted_at is None
    ):
        eligibility = task_schedule_eligibility(
            available_at=current.available_at, policy=state.policy, as_of=state.admitted_at
        )
        kinds.append(
            TaskScheduleEventType.MISFIRED
            if eligibility is TaskScheduleEligibility.MISFIRED
            else TaskScheduleEventType.ELIGIBLE
        )
    if current.status_reason in {"cancellation_requested", "retry_cancellation_requested"} and (
        prior is None or prior.status_reason != current.status_reason
    ):
        kinds.append(TaskScheduleEventType.CANCELLATION_REQUESTED)
    # Worker dispatch and later session attachment share one execution start.
    # The durable marker, not a second CLAIMED -> RUNNING transition, owns it.
    if (
        prior is not None
        and current.status in {"claimed", "running"}
        and prior.started_at is None
        and current.started_at is not None
    ):
        kinds.append(TaskScheduleEventType.STARTED)
    if prior is not None and prior.status != current.status:
        if current.status_reason == "schedule_expired":
            kinds.append(TaskScheduleEventType.EXPIRED)
        elif current.status_reason == "schedule_skipped":
            kinds.append(TaskScheduleEventType.SKIPPED)
        else:
            kind = {
                "claimed": TaskScheduleEventType.CLAIMED,
                "completed": TaskScheduleEventType.COMPLETED,
                "failed": TaskScheduleEventType.FAILED,
                "cancelled": TaskScheduleEventType.CANCELLED,
                "paused": TaskScheduleEventType.HELD,
                "blocked": TaskScheduleEventType.HELD,
                "needs_attention": TaskScheduleEventType.HELD,
                "pending": TaskScheduleEventType.RESUMED,
            }.get(current.status)
            if kind is not None:
                kinds.append(kind)
    return [
        TaskScheduleEvent(
            task_id=current.id,
            sequence=first_sequence + index,
            type=kind,
            revision=state.revision,
            occurred_at=current.updated_at,
            available_at=current.available_at,
            policy=state.policy,
            invocation_id=current.invocation.root_invocation_id,
            operation_id=operation_id,
        )
        for index, kind in enumerate(kinds)
    ]
