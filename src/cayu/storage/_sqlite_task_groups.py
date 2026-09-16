"""SQLite group publication inside the native graph transaction."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    from cayu.storage.sqlite import SQLiteTaskStore


def read_group(store: SQLiteTaskStore, group_id: str) -> TaskGroupSnapshot | None:
    row = store._connection.execute(
        "SELECT graph_id, snapshot_json FROM cayu_task_groups WHERE group_id = ?", (group_id,)
    ).fetchone()
    if row is None:
        return None
    snapshot = TaskGroupSnapshot.model_validate_json(row[1])
    if snapshot.receipt.group_id != group_id or snapshot.receipt.graph.graph_id != row[0]:
        raise TaskGroupUnavailable("Group index contradicts its authority.")
    return snapshot


def check_admission(
    store: SQLiteTaskStore, group: TaskGroupCreate, submitted_digest: str | None
) -> None:
    existing = read_group(store, group.group_id)
    if existing is not None:
        require_group_replay(existing, group, submitted_digest)
    elif (
        store._connection.execute(
            "SELECT 1 FROM cayu_task_graphs WHERE graph_id = ?", (group.graph.graph_id,)
        ).fetchone()
        is not None
    ):
        raise TaskGroupConflict("A group requires a newly admitted graph.")


def publish(
    store: SQLiteTaskStore, publication: GroupPublication, *, creating: bool = False
) -> None:
    snapshot = publication.snapshot
    if creating:
        store._connection.execute(
            "INSERT INTO cayu_task_groups (group_id, graph_id, snapshot_json) VALUES (?, ?, ?)",
            (
                snapshot.receipt.group_id,
                snapshot.receipt.graph.graph_id,
                snapshot.model_dump_json(),
            ),
        )
    else:
        store._connection.execute(
            "UPDATE cayu_task_groups SET snapshot_json = ? WHERE group_id = ?",
            (snapshot.model_dump_json(), snapshot.receipt.group_id),
        )
    store._connection.executemany(
        "INSERT INTO cayu_task_group_events (group_id, sequence, event_json) VALUES (?, ?, ?)",
        [(event.group_id, event.sequence, event.model_dump_json()) for event in publication.events],
    )


async def create_group(
    store: SQLiteTaskStore, request: TaskGroupCreate
) -> TaskGroupCreationReceipt:
    from cayu.storage._sqlite_task_graphs import create_graph

    request = copy_task_group_create(request)
    await create_graph(
        store, request.graph, group=request, submitted_digest=request._submitted_request_sha256
    )
    snapshot = await load_group(store, request.group_id)
    if snapshot is None:
        raise TaskGroupUnavailable("Group admission receipt is unavailable.")
    return snapshot.receipt


async def load_group(store: SQLiteTaskStore, group_id: str) -> TaskGroupSnapshot | None:
    group_id = graph_identifier(group_id)
    async with store._lock:
        return read_group(store, group_id)


async def list_events(
    store: SQLiteTaskStore, group_id: str, *, after_sequence: int, limit: int
) -> list[TaskGroupEvent]:
    group_id = graph_identifier(group_id)
    validate_group_cursor(after_sequence, limit)
    async with store._lock:
        if read_group(store, group_id) is None:
            raise KeyError("Task group not found.")
        rows = store._connection.execute(
            "SELECT sequence, event_json FROM cayu_task_group_events WHERE group_id = ? AND sequence > ? ORDER BY sequence LIMIT ?",
            (group_id, after_sequence, limit),
        ).fetchall()
        events = []
        for sequence, document in rows:
            event = TaskGroupEvent.model_validate_json(document)
            if (
                event.group_id != group_id
                or event.sequence != sequence
                or sequence != after_sequence + len(events) + 1
            ):
                raise TaskGroupUnavailable("Group event indexes contradict their authority.")
            events.append(event)
        return events
