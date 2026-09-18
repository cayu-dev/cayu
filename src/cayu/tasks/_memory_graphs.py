"""Memory task-store graph transactions. Callers do not publish graph state."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from cayu._validation import MAX_PORTABLE_JSON_INTEGER
from cayu.tasks._graphs import (
    GRAPH_TERMINAL_STATUSES,
    GraphTransition,
    member_from_task,
    plan_graph_transition,
    require_graph_membership,
    require_member_authority,
)
from cayu.tasks._group_quiescence import (
    plan_group_graph_transition,
    require_group_mutation,
    waiting_finalizer,
)
from cayu.tasks._groups import prepare_group_admission, require_group_replay
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
from cayu.tasks.groups import TaskGroupConflict, TaskGroupCreate, TaskGroupInvocationObligation

if TYPE_CHECKING:
    from cayu.tasks.base import InMemoryTaskStore


async def create_graph(
    store: InMemoryTaskStore,
    request: TaskGraphCreate,
    *,
    group: TaskGroupCreate | None = None,
    submitted_digest: str | None = None,
) -> TaskGraphCreationReceipt:
    request = copy_task_graph_create(request)
    digest = task_graph_request_sha256(request)
    async with store._lock:
        if group is not None:
            previous_group = store._task_groups.get(group.group_id)
            if previous_group is not None:
                require_group_replay(previous_group, group, submitted_digest)
            elif request.graph_id in store._task_graph_receipts:
                raise TaskGroupConflict("A group requires a newly admitted graph.")
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
            identity in store._tasks
            or identity in store._task_graph_by_task
            or identity in store._task_group_retry_lineage
            for identity in ids
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
            finalizer_task_id=None if group is None else group.finalizer_task_id,
        )
        prepared = {task.id: task for task in admission.tasks}
        receipt = admission.receipt
        events = list(admission.events)
        writes = tuple(store._prepare_task_write(task) for task in prepared.values())
        group_publication = (
            None if group is None else prepare_group_admission(group, admission, submitted_digest)
        )
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
        if group_publication is not None:
            snapshot = group_publication.snapshot
            store._task_groups[snapshot.receipt.group_id] = snapshot
            store._task_group_by_graph[request.graph_id] = snapshot.receipt.group_id
            store._task_group_events[snapshot.receipt.group_id] = list(group_publication.events)
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
    store: InMemoryTaskStore,
    task: Task,
    *,
    schedule_operation_id: str | None,
    settled_execution: tuple[str, str, datetime] | None = None,
    invocation: TaskGroupInvocationObligation | None = None,
    resolve_attention: bool = False,
    retry_successor: Task | None = None,
) -> bool:
    graph_id = store._task_graph_by_task.get(task.id)
    lineage_scope = store._task_group_retry_lineage.get(task.id)
    if graph_id is None and lineage_scope is not None:
        graph_id = lineage_scope[0]
    if graph_id is None:
        if task.graph_id is not None or task.prerequisite_task_ids:
            raise TaskGraphUnavailable("Graph task has no admission authority.")
        return False
    if task.id not in store._tasks:
        raise TaskGraphConflict("Retained graph member identity cannot be reused.")
    group_id = store._task_group_by_graph.get(graph_id)
    snapshot = None if group_id is None else store._task_groups[group_id]
    if snapshot is not None:
        require_group_mutation(
            snapshot,
            store._tasks[task.id],
            task,
            root_task_id=None if lineage_scope is None else lineage_scope[1],
        )
    current = graph_tasks(store, graph_id)
    transition = (
        GraphTransition(tasks=(task,), events=(), retry_settlements=())
        if lineage_scope is not None
        else plan_graph_transition(
            graph_id=graph_id,
            prerequisites=store._task_graph_members[graph_id],
            current=current,
            proposed=task,
            proposed_readiness_recorded=any(
                event.type is TaskGraphEventType.READY and event.task_id == task.id
                for event in store._task_graph_events[graph_id]
            ),
            first_sequence=len(store._task_graph_events[graph_id]) + 1,
            now=store._ownership_clock(),
            group_waiting_task_id=None if snapshot is None else waiting_finalizer(snapshot),
        )
    )
    group_publication = None
    lineage_roots = {
        identity: scope[1]
        for identity, scope in store._task_group_retry_lineage.items()
        if scope[0] == graph_id
    }
    current.update({identity: store._require_task(identity) for identity in lineage_roots})
    added_lineage = None
    if retry_successor is not None:
        transition = GraphTransition(
            tasks=(*transition.tasks, retry_successor),
            events=transition.events,
            retry_settlements=transition.retry_settlements,
        )
        if (
            snapshot is not None
            and snapshot.receipt.quiescence is not None
            and (task.id in snapshot.receipt.member_task_ids or lineage_scope is not None)
        ):
            root = task.id if lineage_scope is None else lineage_scope[1]
            lineage_roots[retry_successor.id] = root
            added_lineage = (retry_successor.id, (graph_id, root))
    if snapshot is not None:
        planned = plan_group_graph_transition(
            snapshot,
            transition,
            current=current,
            prerequisites=store._task_graph_members[graph_id],
            first_sequence=len(store._task_graph_events[graph_id]) + 1,
            now=store._ownership_clock(),
            settled_execution=settled_execution,
            invocation=invocation,
            resolve_attention=resolve_attention,
            lineage_roots=lineage_roots,
            unsettled_effects=frozenset(
                identity
                for identity in (*snapshot.receipt.member_task_ids, *lineage_roots)
                if store._task_has_unsettled_local_execution_attempt(identity)
            ),
        )
        transition, group_publication = planned.transition, planned.group
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
        and updated.id in store._task_graph_members[graph_id]
    }
    for write in writes:
        store._publish_prepared_task_write(write)
    if added_lineage is not None:
        store._task_group_retry_lineage[added_lineage[0]] = added_lineage[1]
    for receipt in transition.retry_settlements:
        store._retry_settlements[(receipt.task_id, receipt.idempotency_key)] = receipt
    store._task_graph_terminal_members.update(terminal_members)
    store._task_graph_events[graph_id].extend(transition.events)
    if group_publication is not None:
        group_id = group_publication.snapshot.receipt.group_id
        store._task_groups[group_id] = group_publication.snapshot
        store._task_group_events[group_id].extend(group_publication.events)
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
    from cayu.tasks._group_quiescence import require_group_deletion_ready

    for graph_id in {
        store._task_graph_by_task[identity]
        for identity in task_ids
        if identity in store._task_graph_by_task
    } | {
        store._task_group_retry_lineage[identity][0]
        for identity in task_ids
        if identity in store._task_group_retry_lineage
    }:
        group_id = store._task_group_by_graph.get(graph_id)
        if group_id is not None:
            snapshot = store._task_groups[group_id]
            require_group_deletion_ready(snapshot)
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
