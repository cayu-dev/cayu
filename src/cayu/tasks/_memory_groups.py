"""Group access under the existing in-memory task-store lock."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cayu.tasks._groups import validate_group_cursor
from cayu.tasks.graphs import graph_identifier
from cayu.tasks.groups import (
    TaskGroupCreate,
    TaskGroupCreationReceipt,
    TaskGroupEvent,
    TaskGroupSnapshot,
    copy_task_group_create,
)

if TYPE_CHECKING:
    from cayu.tasks.base import InMemoryTaskStore


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
