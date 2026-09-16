"""PostgreSQL group state uses the same ownership as its immutable graph."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cayu.storage import _postgres_support as pg
from cayu.tasks._groups import GroupPublication, require_group_replay, validate_group_cursor
from cayu.tasks.graphs import graph_identifier
from cayu.tasks.groups import (
    TaskGroupConflict,
    TaskGroupCreate,
    TaskGroupCreationReceipt,
    TaskGroupEvent,
    TaskGroupSnapshot,
    TaskGroupUnavailable,
    copy_task_group_create,
)

if TYPE_CHECKING:
    from cayu.storage.postgres import PostgresTaskStore


async def read_group(cur: Any, group_id: str) -> TaskGroupSnapshot | None:
    await cur.execute(
        "SELECT graph_id, snapshot_json FROM cayu_task_groups WHERE group_id = %s", (group_id,)
    )
    row = await cur.fetchone()
    if row is None:
        return None
    snapshot = TaskGroupSnapshot.model_validate(pg._loads(row[1]))
    if snapshot.receipt.group_id != group_id or snapshot.receipt.graph.graph_id != row[0]:
        raise TaskGroupUnavailable("Group index contradicts its authority.")
    return snapshot


async def check_admission(cur: Any, group: TaskGroupCreate, submitted_digest: str | None) -> None:
    existing = await read_group(cur, group.group_id)
    if existing is not None:
        require_group_replay(existing, group, submitted_digest)
    else:
        await cur.execute(
            "SELECT 1 FROM cayu_task_graphs WHERE graph_id = %s", (group.graph.graph_id,)
        )
        if await cur.fetchone() is not None:
            raise TaskGroupConflict("A group requires a newly admitted graph.")


async def publish(cur: Any, publication: GroupPublication, *, creating: bool = False) -> None:
    snapshot = publication.snapshot
    if creating:
        await cur.execute(
            "INSERT INTO cayu_task_groups (group_id, graph_id, snapshot_json) VALUES (%s, %s, %s)",
            (
                snapshot.receipt.group_id,
                snapshot.receipt.graph.graph_id,
                snapshot.model_dump_json(),
            ),
        )
    else:
        await cur.execute(
            "UPDATE cayu_task_groups SET snapshot_json = %s WHERE group_id = %s",
            (snapshot.model_dump_json(), snapshot.receipt.group_id),
        )
    for event in publication.events:
        await cur.execute(
            "INSERT INTO cayu_task_group_events (group_id, sequence, event_json) VALUES (%s, %s, %s)",
            (event.group_id, event.sequence, event.model_dump_json()),
        )


async def create_group(
    store: PostgresTaskStore, request: TaskGroupCreate
) -> TaskGroupCreationReceipt:
    from cayu.storage._postgres_task_graphs import create_graph

    request = copy_task_group_create(request)
    await create_graph(
        store, request.graph, group=request, submitted_digest=request._submitted_request_sha256
    )
    snapshot = await load_group(store, request.group_id)
    if snapshot is None:
        raise TaskGroupUnavailable("Group admission receipt is unavailable.")
    return snapshot.receipt


async def load_group(store: PostgresTaskStore, group_id: str) -> TaskGroupSnapshot | None:
    group_id = graph_identifier(group_id)
    await store._ensure_ready()
    async with store._connection() as conn, conn.cursor() as cur:
        return await read_group(cur, group_id)


async def list_events(
    store: PostgresTaskStore, group_id: str, *, after_sequence: int, limit: int
) -> list[TaskGroupEvent]:
    group_id = graph_identifier(group_id)
    validate_group_cursor(after_sequence, limit)
    await store._ensure_ready()
    async with store._connection() as conn, conn.cursor() as cur:
        if await read_group(cur, group_id) is None:
            raise KeyError("Task group not found.")
        await cur.execute(
            "SELECT sequence, event_json FROM cayu_task_group_events WHERE group_id = %s AND sequence > %s ORDER BY sequence LIMIT %s",
            (group_id, after_sequence, limit),
        )
        events = []
        for sequence, document in await cur.fetchall():
            event = TaskGroupEvent.model_validate(pg._loads(document))
            if (
                event.group_id != group_id
                or event.sequence != sequence
                or sequence != after_sequence + len(events) + 1
            ):
                raise TaskGroupUnavailable("Group event indexes contradict their authority.")
            events.append(event)
        return events
