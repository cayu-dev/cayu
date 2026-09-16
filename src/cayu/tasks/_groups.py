"""Pure group publication plans; called under the graph's mutation owner."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from cayu._validation import MAX_PORTABLE_JSON_INTEGER
from cayu.tasks._graph_admission import GraphAdmission
from cayu.tasks._graphs import GRAPH_TERMINAL_STATUSES, GraphTransition, member_from_task
from cayu.tasks.base import TaskStatus
from cayu.tasks.graphs import task_graph_request_sha256
from cayu.tasks.groups import (
    TaskGroupConflict,
    TaskGroupCreate,
    TaskGroupCreationReceipt,
    TaskGroupDecision,
    TaskGroupEvent,
    TaskGroupEventType,
    TaskGroupSnapshot,
    TaskGroupStatus,
    TaskGroupUnavailable,
    task_group_request_sha256,
)


@dataclass(frozen=True)
class GroupPublication:
    snapshot: TaskGroupSnapshot
    events: tuple[TaskGroupEvent, ...]


def require_group_replay(
    snapshot: TaskGroupSnapshot, request: TaskGroupCreate, submitted_digest: str | None
) -> TaskGroupCreationReceipt:
    receipt = snapshot.receipt
    if (
        receipt.group_id != request.group_id
        or receipt.graph.graph_id != request.graph.graph_id
        or receipt.member_task_ids != request.member_task_ids
        or receipt.policy != request.policy
        or receipt.request_sha256 != task_group_request_sha256(request)
        or receipt.submitted_request_sha256
        != (submitted_digest or task_group_request_sha256(request))
        or receipt.graph.request_sha256 != task_graph_request_sha256(request.graph)
        or receipt.graph.task_ids != tuple(node.task.task_id for node in request.graph.nodes)
        or receipt.graph.submitted_request_sha256
        != (request.graph._submitted_request_sha256 or task_graph_request_sha256(request.graph))
    ):
        raise TaskGroupConflict("Group identity has different admission content.")
    return receipt.model_copy(deep=True)


def prepare_group_admission(
    request: TaskGroupCreate, admission: GraphAdmission, submitted_digest: str | None
) -> GroupPublication:
    if admission.receipt.request_sha256 != task_graph_request_sha256(request.graph):
        raise TaskGroupConflict("Group and graph admission authorities differ.")
    digest = task_group_request_sha256(request)
    receipt = TaskGroupCreationReceipt(
        group_id=request.group_id,
        graph=admission.receipt,
        member_task_ids=request.member_task_ids,
        policy=request.policy,
        request_sha256=digest,
        submitted_request_sha256=submitted_digest or digest,
    )
    tasks = {task.id: task for task in admission.tasks}
    snapshot = TaskGroupSnapshot(
        receipt=receipt,
        members=tuple(member_from_task(tasks[identity]) for identity in receipt.member_task_ids),
        last_sequence=1,
    )
    return GroupPublication(
        snapshot,
        (
            TaskGroupEvent(
                group_id=receipt.group_id,
                sequence=1,
                type=TaskGroupEventType.CREATED,
                occurred_at=receipt.graph.accepted_at,
            ),
        ),
    )


def plan_group_transition(
    snapshot: TaskGroupSnapshot, transition: GraphTransition, *, now: datetime
) -> GroupPublication:
    members = {member.task_id: member for member in snapshot.members}
    events: list[TaskGroupEvent] = []
    for task in transition.tasks:
        prior = members.get(task.id)
        if prior is None:
            continue
        member = member_from_task(task)
        if prior.prerequisite_task_ids != member.prerequisite_task_ids:
            raise TaskGroupUnavailable("Group member dependencies changed.")
        if prior.status in GRAPH_TERMINAL_STATUSES and prior != member:
            raise TaskGroupConflict("Terminal group member cannot change.")
        members[task.id] = member
        if prior.status not in GRAPH_TERMINAL_STATUSES and member.status in GRAPH_TERMINAL_STATUSES:
            events.append(
                TaskGroupEvent(
                    group_id=snapshot.receipt.group_id,
                    sequence=snapshot.last_sequence + len(events) + 1,
                    type=TaskGroupEventType.MEMBER_TERMINAL,
                    occurred_at=now,
                    task_id=task.id,
                    task_status=task.status,
                )
            )
    successful = tuple(
        identity for identity, member in members.items() if member.status is TaskStatus.COMPLETED
    )
    unsuccessful = tuple(
        identity
        for identity, member in members.items()
        if member.status in GRAPH_TERMINAL_STATUSES and member.status is not TaskStatus.COMPLETED
    )
    required = snapshot.receipt.policy.required_successes(len(members))
    decision = snapshot.decision
    if decision is None and (
        len(successful) >= required or len(members) - len(unsuccessful) < required
    ):
        succeeded = len(successful) >= required
        decision = TaskGroupDecision(
            status=TaskGroupStatus.SUCCEEDED if succeeded else TaskGroupStatus.FAILED,
            decided_at=now,
            successful_task_ids=successful[:required],
            unsuccessful_task_ids=unsuccessful,
            reason=None if succeeded else "completion_policy_impossible",
        )
        for kind in (
            (TaskGroupEventType.POLICY_SATISFIED, TaskGroupEventType.SUCCEEDED)
            if succeeded
            else (TaskGroupEventType.POLICY_IMPOSSIBLE, TaskGroupEventType.FAILED)
        ):
            events.append(
                TaskGroupEvent(
                    group_id=snapshot.receipt.group_id,
                    sequence=snapshot.last_sequence + len(events) + 1,
                    type=kind,
                    occurred_at=now,
                    decision=decision,
                )
            )
    return GroupPublication(
        TaskGroupSnapshot(
            receipt=snapshot.receipt,
            members=tuple(members.values()),
            decision=decision,
            last_sequence=snapshot.last_sequence + len(events),
        ),
        tuple(events),
    )


def validate_group_cursor(after_sequence: int, limit: int) -> None:
    if type(after_sequence) is not int or not 0 <= after_sequence <= MAX_PORTABLE_JSON_INTEGER:
        raise ValueError("Invalid group event cursor.")
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("Group event page size must be between 1 and 1000.")
