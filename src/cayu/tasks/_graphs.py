"""Pure bounded graph publication planning, shared by task-store backends."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from cayu.tasks.base import (
    Task,
    TaskRetryAttemptDisposition,
    TaskRetrySeriesDisposition,
    TaskRetrySettlementResult,
    TaskStatus,
    _runtime_task_retry_terminal_settlement,
    copy_task,
)
from cayu.tasks.graphs import (
    TaskGraphConflict,
    TaskGraphCreationReceipt,
    TaskGraphEvent,
    TaskGraphEventType,
    TaskGraphMember,
    TaskGraphUnavailable,
)

GRAPH_TERMINAL_STATUSES = frozenset(
    {
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
        TaskStatus.DEPENDENCY_SKIPPED,
    }
)
_FAILURE_STATUSES = GRAPH_TERMINAL_STATUSES - {TaskStatus.COMPLETED}
_EXECUTING_STATUSES = frozenset({TaskStatus.CLAIMED, TaskStatus.RUNNING, TaskStatus.COMPLETED})


@dataclass(frozen=True)
class GraphTransition:
    tasks: tuple[Task, ...]
    events: tuple[TaskGraphEvent, ...]
    retry_settlements: tuple[TaskRetrySettlementResult, ...] = ()


def dependency_skip_evidence(
    failed: tuple[str, ...], *, status_payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Use the same diagnostic shape for admission capacity and publication."""
    return {
        "status_reason": "dependency_failed",
        "status_payload": {
            **(status_payload or {}),
            "failed_prerequisite_task_ids": list(failed),
        },
        "error": {
            "code": "dependency_failed",
            "failed_prerequisite_task_ids": list(failed),
        },
    }


def member_from_task(task: Task) -> TaskGraphMember:
    return TaskGraphMember(
        task_id=task.id,
        prerequisite_task_ids=task.prerequisite_task_ids,
        status=task.status,
        failed_prerequisite_task_ids=(
            tuple((task.status_payload or {}).get("failed_prerequisite_task_ids", ()))
            if task.status is TaskStatus.DEPENDENCY_SKIPPED
            else ()
        ),
    )


def require_graph_membership(
    graph_id: str,
    receipt: TaskGraphCreationReceipt,
    prerequisites: Mapping[str, tuple[str, ...]],
) -> None:
    if receipt.graph_id != graph_id or tuple(sorted(prerequisites)) != receipt.task_ids:
        raise TaskGraphUnavailable("Graph membership does not match its admission receipt.")


def require_member_authority(graph_id: str, task: Task, prerequisites: tuple[str, ...]) -> None:
    if task.graph_id != graph_id or task.prerequisite_task_ids != prerequisites:
        raise TaskGraphUnavailable("Graph member does not match its admission authority.")


def plan_graph_transition(
    *,
    graph_id: str,
    prerequisites: Mapping[str, tuple[str, ...]],
    current: Mapping[str, Task],
    proposed: Task,
    proposed_readiness_recorded: bool,
    first_sequence: int,
    now: datetime,
    group_waiting_task_id: str | None = None,
) -> GraphTransition:
    """Validate a direct change and compute every resulting release/skip.

    The caller owns the complete graph transaction. Nothing here publishes or
    calls external code. A terminal member cannot be changed by a stale writer.
    """
    if set(current) != set(prerequisites) or proposed.id not in current:
        raise TaskGraphUnavailable("Complete graph membership is unavailable.")
    if any(not set(dependencies) <= current.keys() for dependencies in prerequisites.values()):
        raise TaskGraphUnavailable("Graph prerequisite evidence is missing.")
    prior = current[proposed.id]
    if proposed.id == group_waiting_task_id:
        if proposed.status in _EXECUTING_STATUSES:
            raise TaskGraphConflict("Task group has not released its finalizer.")
        if proposed.status is TaskStatus.PENDING:
            proposed = proposed.model_copy(update={"status": TaskStatus.WAITING_GROUP})
    if prior.status in GRAPH_TERMINAL_STATUSES and proposed != prior:
        raise TaskGraphConflict("Terminal graph member cannot be changed.")
    for task_id, task in current.items():
        if task.graph_id != graph_id or task.prerequisite_task_ids != prerequisites[task_id]:
            raise TaskGraphUnavailable("Task graph authority does not match membership.")
    if (
        proposed.graph_id != graph_id
        or proposed.prerequisite_task_ids != prerequisites[proposed.id]
    ):
        raise TaskGraphConflict("Task graph membership is immutable.")
    ready = all(
        current[identity].status is TaskStatus.COMPLETED for identity in prerequisites[proposed.id]
    )
    if proposed.status in _EXECUTING_STATUSES and not ready:
        raise TaskGraphConflict("Task prerequisites have not completed successfully.")
    if proposed.status is TaskStatus.DEPENDENCY_SKIPPED and prior.status is not proposed.status:
        raise TaskGraphConflict("Only dependency propagation may skip a graph member.")
    if proposed.status is TaskStatus.PENDING and not ready:
        proposed = proposed.model_copy(update={"status": TaskStatus.WAITING_DEPENDENCIES})
    tasks = dict(current)
    tasks[proposed.id] = copy_task(proposed)
    events: list[TaskGraphEvent] = []
    retry_settlements: list[TaskRetrySettlementResult] = []

    def event(task: Task, kind: TaskGraphEventType, causes: tuple[str, ...] = ()) -> None:
        events.append(
            TaskGraphEvent(
                graph_id=graph_id,
                sequence=first_sequence + len(events),
                type=kind,
                occurred_at=now,
                task_id=task.id,
                status=task.status,
                prerequisite_task_ids=causes,
            )
        )

    if prior.status not in GRAPH_TERMINAL_STATUSES and proposed.status in GRAPH_TERMINAL_STATUSES:
        event(proposed, TaskGraphEventType.TERMINAL)
    if (
        prior.status
        in {
            TaskStatus.PAUSED,
            TaskStatus.BLOCKED,
            TaskStatus.NEEDS_ATTENTION,
            TaskStatus.WAITING_GROUP,
        }
        and proposed.status is TaskStatus.PENDING
        and ready
        and not proposed_readiness_recorded
    ):
        # A hold can outlive prerequisite completion or group release. Resume
        # supplies PENDING directly, including for a finalizer without prerequisites.
        # The store's recorded-readiness check also keeps already-ready roots
        # from publishing a duplicate event after an ordinary hold/resume cycle.
        event(proposed, TaskGraphEventType.READY, prerequisites[proposed.id])
    changed = True
    while changed:
        changed = False
        for task_id in sorted(tasks):
            task = tasks[task_id]
            if task.status in GRAPH_TERMINAL_STATUSES:
                continue
            deps = prerequisites[task_id]
            failed = tuple(
                identity for identity in deps if tasks[identity].status in _FAILURE_STATUSES
            )
            if failed:
                if task.status in _EXECUTING_STATUSES or task.worker_id is not None:
                    raise TaskGraphUnavailable(
                        "An executing graph member has failed prerequisites."
                    )
                retry_settlement = None
                started_at = task.started_at
                if task.retry_series is not None:
                    retry_settlement = _runtime_task_retry_terminal_settlement(
                        task,
                        operation="dependency-skip",
                        request_disposition=TaskRetryAttemptDisposition.NON_RETRYABLE_FAILURE,
                        series_disposition=TaskRetrySeriesDisposition.NON_RETRYABLE_FAILURE,
                        status=TaskStatus.FAILED,
                        error={
                            "code": "dependency_failed",
                            "failed_prerequisite_task_ids": list(failed),
                        },
                        committed_at=now,
                    )
                    task = retry_settlement.task
                task = task.model_copy(
                    update={
                        "status": TaskStatus.DEPENDENCY_SKIPPED,
                        **dependency_skip_evidence(
                            failed,
                            status_payload=task.status_payload
                            if retry_settlement is not None
                            else None,
                        ),
                        "completed_at": now,
                        "started_at": started_at,
                        "updated_at": now,
                        "worker_id": None,
                        "lease_expires_at": None,
                        "interrupted_handoff_id": None,
                    }
                )
                tasks[task_id] = copy_task(task)
                if retry_settlement is not None:
                    retry_settlements.append(
                        TaskRetrySettlementResult.model_validate(
                            retry_settlement.model_copy(update={"task": tasks[task_id]}).model_dump(
                                mode="python",
                                warnings=False,
                            )
                        )
                    )
                event(task, TaskGraphEventType.SKIPPED, failed)
                changed = True
            elif task.status is TaskStatus.WAITING_DEPENDENCIES and all(
                tasks[identity].status is TaskStatus.COMPLETED for identity in deps
            ):
                task = task.model_copy(
                    update={
                        "status": TaskStatus.WAITING_GROUP
                        if task_id == group_waiting_task_id
                        else TaskStatus.PENDING,
                        "updated_at": now,
                    }
                )
                tasks[task_id] = copy_task(task)
                event(
                    task,
                    TaskGraphEventType.WAITING_GROUP
                    if task_id == group_waiting_task_id
                    else TaskGraphEventType.READY,
                    deps,
                )
                changed = True
    return GraphTransition(
        tasks=tuple(
            tasks[identity] for identity in sorted(tasks) if tasks[identity] != current[identity]
        ),
        events=tuple(events),
        retry_settlements=tuple(retry_settlements),
    )
