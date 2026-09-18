"""Group access under the existing in-memory task-store lock."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from cayu.tasks._groups import validate_group_cursor
from cayu.tasks.graphs import graph_identifier
from cayu.tasks.groups import (
    TaskGroupConflict,
    TaskGroupCreate,
    TaskGroupCreationReceipt,
    TaskGroupEvent,
    TaskGroupInvocationObligation,
    TaskGroupQuiescenceResolution,
    TaskGroupSnapshot,
    copy_task_group_create,
)

if TYPE_CHECKING:
    from cayu.tasks.base import InMemoryTaskStore, Task


async def create_group(
    store: InMemoryTaskStore, request: TaskGroupCreate, *, submitted_digest: str | None = None
) -> TaskGroupCreationReceipt:
    from cayu.tasks._memory_graphs import create_graph

    request = copy_task_group_create(request)
    await create_graph(store, request.graph, group=request, submitted_digest=submitted_digest)
    async with store._lock:
        return store._task_groups[request.group_id].receipt.model_copy(deep=True)


async def load_group(store: InMemoryTaskStore, group_id: str) -> TaskGroupSnapshot | None:
    group_id = graph_identifier(group_id)
    async with store._lock:
        snapshot = store._task_groups.get(group_id)
        return None if snapshot is None else snapshot.model_copy(deep=True)


async def cancellation_requested(store: InMemoryTaskStore, task_id: str) -> bool:
    from cayu._validation import require_clean_nonblank

    task_id = require_clean_nonblank(task_id, "task_id")
    async with store._lock:
        return cancellation_requested_unlocked(store, task_id)


def cancellation_requested_unlocked(store: InMemoryTaskStore, task_id: str) -> bool:
    graph_id, root_id = store._task_group_retry_lineage.get(
        task_id, (store._task_graph_by_task.get(task_id), task_id)
    )
    group_id = None if graph_id is None else store._task_group_by_graph.get(graph_id)
    return (
        group_id is not None and root_id in store._task_groups[group_id].quiescence.loser_task_ids
    )


async def retains_execution(store: InMemoryTaskStore, task_id: str) -> bool:
    from cayu._validation import require_clean_nonblank

    task_id = require_clean_nonblank(task_id, "task_id")
    async with store._lock:
        graph_id, root_id = store._task_group_retry_lineage.get(
            task_id, (store._task_graph_by_task.get(task_id), task_id)
        )
        group_id = None if graph_id is None else store._task_group_by_graph.get(graph_id)
        if group_id is None:
            return False
        receipt = store._task_groups[group_id].receipt
        return receipt.quiescence is not None and root_id in receipt.member_task_ids


async def reconciliation_candidates(
    store: InMemoryTaskStore,
    *,
    after_group_id: str | None,
    limit: int,
) -> list[str]:
    from cayu.tasks._group_quiescence import validate_reconciliation_page
    from cayu.tasks.groups import TaskGroupQuiescenceStatus

    validate_reconciliation_page(after_group_id, limit)
    async with store._lock:
        return sorted(
            identity
            for identity, snapshot in store._task_groups.items()
            if snapshot.quiescence.status is TaskGroupQuiescenceStatus.DRAINING
            and (after_group_id is None or identity > after_group_id)
        )[:limit]


async def list_events(
    store: InMemoryTaskStore, group_id: str, *, after_sequence: int, limit: int
) -> list[TaskGroupEvent]:
    group_id = graph_identifier(group_id)
    validate_group_cursor(after_sequence, limit)
    async with store._lock:
        if group_id not in store._task_groups:
            raise KeyError("Task group not found.")
        return [
            event.model_copy(deep=True)
            for event in store._task_group_events[group_id][after_sequence : after_sequence + limit]
        ]


async def reconcile(
    store: InMemoryTaskStore,
    group_id: str,
    *,
    resolution: TaskGroupQuiescenceResolution | None = None,
    settled_execution: tuple[str, str, datetime] | None = None,
    invocation: TaskGroupInvocationObligation | None = None,
) -> TaskGroupSnapshot:
    from cayu.tasks._group_quiescence import (
        prepare_resolution,
        reconciliation_is_complete,
        require_resolution,
        retained_execution_settlement_matches,
        retained_invocation_settlement_matches,
    )
    from cayu.tasks._memory_graphs import store_graph_task

    group_id = graph_identifier(group_id)
    prepared = None if resolution is None else prepare_resolution(resolution)
    async with store._lock:
        snapshot = store._task_groups.get(group_id)
        if snapshot is None:
            raise KeyError("Task group not found.")
        if prepared is not None:
            resolution, digest = prepared
            prior = store._task_group_resolutions.get((group_id, resolution.idempotency_key))
            if prior is not None:
                if prior[0] != digest:
                    raise TaskGroupConflict("Resolution key has different content.")
                return prior[1].model_copy(deep=True)
            require_resolution(snapshot, resolution)
        if snapshot.receipt.quiescence is None:
            return snapshot.model_copy(deep=True)
        if (
            resolution is None
            and reconciliation_is_complete(snapshot)
            and (invocation is None or retained_invocation_settlement_matches(snapshot, invocation))
            and (
                settled_execution is None
                or retained_execution_settlement_matches(snapshot, settled_execution)
            )
        ):
            return snapshot.model_copy(deep=True)
        task = store._tasks.get(snapshot.receipt.member_task_ids[0])
        if task is None:
            raise TaskGroupConflict("Unsettled group lost task evidence.")
        store_graph_task(
            store,
            task,
            schedule_operation_id=None,
            settled_execution=settled_execution,
            invocation=invocation,
            resolve_attention=resolution is not None,
        )
        result = store._task_groups[group_id].model_copy(deep=True)
        if prepared is not None:
            store._task_group_resolutions[(group_id, prepared[0].idempotency_key)] = (
                prepared[1],
                result,
            )
        return result


async def settle_execution(store: InMemoryTaskStore, task: Task) -> None:
    if task.started_at is None or task.worker_id is None:
        return
    async with store._lock:
        lineage = store._task_group_retry_lineage.get(task.id)
        graph_id = task.graph_id if lineage is None else lineage[0]
        group_id = None if graph_id is None else store._task_group_by_graph.get(graph_id)
        snapshot = None if group_id is None else store._task_groups[group_id]
        if (
            snapshot is None
            or snapshot.receipt.quiescence is None
            or (task.id not in snapshot.receipt.member_task_ids and lineage is None)
        ):
            return
    assert group_id is not None
    await reconcile(store, group_id, settled_execution=(task.id, task.worker_id, task.started_at))


async def observe_result_resolution(
    store: InMemoryTaskStore, task_id: str, decision_id: str, owner_id: str, *, settled: bool
) -> None:
    from cayu.tasks._group_quiescence import observe_result_resolution as plan

    async with store._lock:
        graph_id, root_id = store._task_group_retry_lineage.get(
            task_id, (store._task_graph_by_task.get(task_id), task_id)
        )
        group_id = None if graph_id is None else store._task_group_by_graph.get(graph_id)
        if group_id is None:
            return
        store._task_groups[group_id] = plan(
            store._task_groups[group_id],
            root_id,
            task_id,
            decision_id,
            owner_id,
            settled=settled,
            now=store._ownership_clock(),
        )


async def observe_invocation(
    store: InMemoryTaskStore, invocation: TaskGroupInvocationObligation
) -> None:
    invocation = TaskGroupInvocationObligation.model_validate(
        invocation.model_dump(mode="python", warnings=False)
    )
    async with store._lock:
        task = store._tasks.get(invocation.task_id)
        lineage = store._task_group_retry_lineage.get(invocation.task_id)
        if lineage is not None:
            graph_id = lineage[0]
        elif task is not None:
            graph_id = task.graph_id
        else:
            graph_id = next(
                (
                    key
                    for key, members in store._task_graph_members.items()
                    if invocation.task_id in members
                ),
                None,
            )
        group_id = None if graph_id is None else store._task_group_by_graph.get(graph_id)
        snapshot = None if group_id is None else store._task_groups[group_id]
        if (
            snapshot is None
            or snapshot.receipt.quiescence is None
            or (task is not None and task.work_contract is not None)
            or (invocation.task_id not in snapshot.receipt.member_task_ids and lineage is None)
            or not any(
                item.task_id == invocation.task_id for item in snapshot.quiescence.executions
            )
        ):
            return
    assert group_id is not None
    await reconcile(store, group_id, invocation=invocation)
