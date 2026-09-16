"""Memory task-store graph transactions. Callers do not publish graph state."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cayu._validation import MAX_PORTABLE_JSON_INTEGER
from cayu.tasks._graphs import (
    GRAPH_TERMINAL_STATUSES,
    member_from_task,
    plan_graph_transition,
    require_graph_membership,
    require_member_authority,
)
from cayu.tasks.base import Task
from cayu.tasks.graphs import (
    TaskGraphConflict,
    TaskGraphCreate,
    TaskGraphCreationReceipt,
    TaskGraphEvent,
    TaskGraphEventType,
    TaskGraphSnapshot,
    TaskGraphUnavailable,
    copy_task_graph_create,
    graph_identifier,
    task_graph_request_sha256,
)

if TYPE_CHECKING:
    from cayu.tasks.base import InMemoryTaskStore


async def create_graph(
    store: InMemoryTaskStore, request: TaskGraphCreate
) -> TaskGraphCreationReceipt:
    request = copy_task_graph_create(request)
    digest = task_graph_request_sha256(request)
    async with store._lock:
        existing = store._task_graph_receipts.get(request.graph_id)
        if existing is not None:
            require_graph_membership(
                request.graph_id, existing, store._task_graph_members[request.graph_id]
            )
            if existing.request_sha256 != digest or existing.submitted_request_sha256 != (
                request._submitted_request_sha256 or digest
            ):
                raise TaskGraphConflict("Graph identity has different admission content.")
            return existing.model_copy(deep=True)
        ids = tuple(node.task.task_id for node in request.nodes)
        if any(
            identity in store._tasks or identity in store._task_graph_by_task for identity in ids
        ):
            raise TaskGraphConflict("A graph member identity is already occupied.")
        from cayu.tasks._graph_admission import prepare_graph_admission

        def validate_task(req):
            if req.work_contract is not None:
                store._require_work_contract(req.work_contract)
                store._ensure_contract_session_accepts_attachment(req.work_contract, req.session_id)
            if req.session_id in store._session_closure_claims:
                raise TaskGraphConflict("Graph task session is owned by closure.")

        parents = {}
        for node in request.nodes:
            assert node.task.task_id is not None
            if node.task.parent_task_id is not None and node.task.parent_task_id not in ids:
                parent = store._task_parent_for_create(node.task, task_id=node.task.task_id)
                if parent is not None:
                    parents[parent.id] = parent
        admission = prepare_graph_admission(
            request,
            digest=digest,
            # Admission initializes retry deadlines, not worker lease authority.
            now=store._clock(),
            parents=parents,
            validate_task=validate_task,
        )
        prepared = {task.id: task for task in admission.tasks}
        receipt = admission.receipt
        events = list(admission.events)
        writes = tuple(store._prepare_task_write(task) for task in prepared.values())
        # All task construction, parent/contract resolution and event validation
        # precede publication. There are no awaits in the mutation section.
        for write in writes:
            store._publish_prepared_task_write(write)
        store._task_graph_members[request.graph_id] = {
            task.id: task.prerequisite_task_ids for task in prepared.values()
        }
        store._task_graph_receipts[request.graph_id] = receipt
        store._task_graph_events[request.graph_id] = events
        store._task_graph_by_task.update({identity: request.graph_id for identity in prepared})
    store._publish_task_admission_broadcast()
    return receipt.model_copy(deep=True)


def graph_tasks(store: InMemoryTaskStore, graph_id: str) -> dict[str, Task]:
    require_graph_membership(
        graph_id, store._task_graph_receipts[graph_id], store._task_graph_members[graph_id]
    )
    result = {}
    for identity in store._task_graph_members[graph_id]:
        task = store._tasks.get(identity)
        if task is None:
            raise TaskGraphUnavailable("Graph member evidence is missing.")
        result[identity] = task
    return result


def store_graph_task(
    store: InMemoryTaskStore, task: Task, *, schedule_operation_id: str | None
) -> bool:
    graph_id = store._task_graph_by_task.get(task.id)
    if graph_id is None:
        if task.graph_id is not None or task.prerequisite_task_ids:
            raise TaskGraphUnavailable("Graph task has no admission authority.")
        return False
    if task.id not in store._tasks:
        raise TaskGraphConflict("Retained graph member identity cannot be reused.")
    transition = plan_graph_transition(
        graph_id=graph_id,
        prerequisites=store._task_graph_members[graph_id],
        current=graph_tasks(store, graph_id),
        proposed=task,
        proposed_readiness_recorded=any(
            event.type is TaskGraphEventType.READY and event.task_id == task.id
            for event in store._task_graph_events[graph_id]
        ),
        first_sequence=len(store._task_graph_events[graph_id]) + 1,
        now=store._ownership_clock(),
    )
    for updated in transition.tasks:
        if updated.session_id in store._session_closure_claims:
            raise TaskGraphConflict("Graph member is owned by session closure.")
    writes = tuple(
        store._prepare_task_write(
            updated, schedule_operation_id=schedule_operation_id if updated.id == task.id else None
        )
        for updated in transition.tasks
    )
    terminal_members = {
        updated.id: member_from_task(updated)
        for updated in transition.tasks
        if updated.status in GRAPH_TERMINAL_STATUSES
    }
    for write in writes:
        store._publish_prepared_task_write(write)
    for receipt in transition.retry_settlements:
        store._retry_settlements[(receipt.task_id, receipt.idempotency_key)] = receipt
    store._task_graph_terminal_members.update(terminal_members)
    store._task_graph_events[graph_id].extend(transition.events)
    if any(event.type is TaskGraphEventType.READY for event in transition.events):
        store._publish_task_admission_broadcast()
    return True


async def load_graph(store: InMemoryTaskStore, graph_id: str) -> TaskGraphSnapshot | None:
    graph_id = graph_identifier(graph_id)
    async with store._lock:
        receipt = store._task_graph_receipts.get(graph_id)
        if receipt is None:
            return None
        require_graph_membership(graph_id, receipt, store._task_graph_members[graph_id])
        members = []
        for identity in receipt.task_ids:
            task = store._tasks.get(identity)
            if task is not None:
                require_member_authority(
                    graph_id, task, store._task_graph_members[graph_id][identity]
                )
            member = (
                member_from_task(task)
                if task is not None
                else store._task_graph_terminal_members.get(identity)
            )
            if member is None:
                raise TaskGraphUnavailable("Graph member evidence is missing.")
            members.append(member)
        return TaskGraphSnapshot(
            receipt=receipt.model_copy(deep=True),
            members=tuple(members),
            last_sequence=len(store._task_graph_events[graph_id]),
        )


def require_graph_deletion_ready(store: InMemoryTaskStore, task_ids: tuple[str, ...]) -> None:
    for graph_id in {
        store._task_graph_by_task[identity]
        for identity in task_ids
        if identity in store._task_graph_by_task
    }:
        if any(
            identity not in store._task_graph_terminal_members
            for identity in store._task_graph_members[graph_id]
        ):
            raise TaskGraphConflict("Nonterminal graph retains its member tasks.")


async def list_graph_events(
    store: InMemoryTaskStore, graph_id: str, *, after_sequence: int, limit: int
) -> list[TaskGraphEvent]:
    graph_id = graph_identifier(graph_id)
    if type(after_sequence) is not int or not 0 <= after_sequence <= MAX_PORTABLE_JSON_INTEGER:
        raise ValueError("Invalid graph event cursor.")
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("Graph event page size must be between 1 and 1000.")
    async with store._lock:
        if graph_id not in store._task_graph_receipts:
            raise KeyError("Task graph not found.")
        return [
            event.model_copy(deep=True)
            for event in store._task_graph_events[graph_id][after_sequence : after_sequence + limit]
        ]
