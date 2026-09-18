"""Prepare complete graph admission before any backend publishes members."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime

from cayu.tasks._graphs import dependency_skip_evidence
from cayu.tasks.base import (
    Task,
    TaskCreate,
    TaskInvocationSnapshot,
    TaskStatus,
    _task_from_create,
    copy_task,
    require_contract_bound_task_creation_snapshot,
)
from cayu.tasks.graphs import (
    TaskGraphConflict,
    TaskGraphCreate,
    TaskGraphCreationReceipt,
    TaskGraphEvent,
    TaskGraphEventType,
)


@dataclass(frozen=True)
class GraphAdmission:
    receipt: TaskGraphCreationReceipt
    tasks: tuple[Task, ...]
    events: tuple[TaskGraphEvent, ...]


def prepare_graph_admission(
    request: TaskGraphCreate,
    *,
    digest: str,
    now: datetime,
    parents: Mapping[str, Task | TaskInvocationSnapshot],
    validate_task: Callable[[TaskCreate], None],
    finalizer_task_id: str | None = None,
) -> GraphAdmission:
    if request._submitted_request_sha256 is not None:
        member_ids = {node.task.task_id for node in request.nodes}
        external_parents = {
            node.task.parent_task_id
            for node in request.nodes
            if node.task.parent_task_id is not None and node.task.parent_task_id not in member_ids
        }
        if {parent.id for parent in request._parent_invocations} != external_parents:
            raise TaskGraphConflict("Prepared graph has incomplete parent authority.")
    for expected in request._parent_invocations:
        actual = parents.get(expected.id)
        if actual is None:
            raise TaskGraphConflict("Validated graph parent authority is unavailable.")
        observed = TaskInvocationSnapshot(
            id=actual.id,
            session_id=actual.session_id,
            session_instance_id=actual.session_instance_id,
            invocation=actual.invocation,
        )
        if observed != expected:
            raise TaskGraphConflict("Graph parent authority changed after preflight.")
    prepared: dict[str, Task] = {}
    ids = {node.task.task_id for node in request.nodes}
    remaining = list(request.nodes)
    while remaining:
        progressed = False
        for node in tuple(remaining):
            req = node.task
            assert req.task_id is not None
            if req.parent_task_id in ids and req.parent_task_id not in prepared:
                continue
            parent = prepared.get(req.parent_task_id or "") or parents.get(req.parent_task_id or "")
            if req.parent_task_id is not None and parent is None:
                raise TaskGraphConflict("Graph parent authority is unavailable.")
            validate_task(req)
            task = _task_from_create(
                req,
                task_id=req.task_id,
                parent_task=parent,
                retry_started_at=now,
                supports_verified_work_contracts=True,
            )
            prepared[task.id] = copy_task(
                task.model_copy(
                    update={
                        "graph_id": request.graph_id,
                        "prerequisite_task_ids": node.prerequisite_task_ids,
                        "status": TaskStatus.WAITING_GROUP
                        if req.task_id == finalizer_task_id
                        else TaskStatus.WAITING_DEPENDENCIES
                        if node.prerequisite_task_ids
                        else TaskStatus.PENDING,
                    }
                )
            )
            if task.work_contract is not None:
                require_contract_bound_task_creation_snapshot(prepared[task.id])
                if node.prerequisite_task_ids:
                    # Every prerequisite can fail in one propagation transaction.
                    # Reserve its diagnostics in addition to the ordinary lifecycle
                    # headroom, checking both canonical bytes and JSON values.
                    require_contract_bound_task_creation_snapshot(
                        prepared[task.id].model_copy(
                            update=dependency_skip_evidence(node.prerequisite_task_ids)
                        )
                    )
            remaining.remove(node)
            progressed = True
        if not progressed:
            raise TaskGraphConflict("Graph parent lineage contains a cycle.")
    receipt = TaskGraphCreationReceipt(
        graph_id=request.graph_id,
        request_sha256=digest,
        submitted_request_sha256=request._submitted_request_sha256 or digest,
        task_ids=tuple(sorted(prepared)),
        accepted_at=now,
    )
    events = [
        TaskGraphEvent(
            graph_id=request.graph_id,
            sequence=1,
            type=TaskGraphEventType.CREATED,
            occurred_at=now,
        )
    ]
    for identity in receipt.task_ids:
        task = prepared[identity]
        events.append(
            TaskGraphEvent(
                graph_id=request.graph_id,
                sequence=len(events) + 1,
                type=TaskGraphEventType.WAITING_GROUP
                if task.status is TaskStatus.WAITING_GROUP
                else TaskGraphEventType.WAITING
                if task.prerequisite_task_ids
                else TaskGraphEventType.READY,
                occurred_at=now,
                task_id=task.id,
                status=task.status,
                prerequisite_task_ids=task.prerequisite_task_ids,
            )
        )
    return GraphAdmission(receipt=receipt, tasks=tuple(prepared.values()), events=tuple(events))
