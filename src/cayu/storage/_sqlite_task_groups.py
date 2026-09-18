"""SQLite group publication inside the native graph transaction."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from cayu.tasks._groups import GroupPublication, require_group_replay, validate_group_cursor
from cayu.tasks.graphs import graph_identifier
from cayu.tasks.groups import (
    TaskGroupConflict,
    TaskGroupCreate,
    TaskGroupCreationReceipt,
    TaskGroupEvent,
    TaskGroupInvocationObligation,
    TaskGroupQuiescenceResolution,
    TaskGroupSnapshot,
    TaskGroupUnavailable,
    copy_task_group_create,
)

if TYPE_CHECKING:
    from cayu.storage.sqlite import SQLiteTaskStore
    from cayu.tasks.base import Task


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
    store._connection.execute(
        "UPDATE cayu_task_groups SET barrier_status = ?, barrier_deadline = ? WHERE group_id = ?",
        (
            snapshot.quiescence.status.value,
            None
            if snapshot.quiescence.deadline is None
            else snapshot.quiescence.deadline.isoformat(),
            snapshot.receipt.group_id,
        ),
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


async def reconciliation_candidates(
    store: SQLiteTaskStore,
    *,
    after_group_id: str | None,
    limit: int,
) -> list[str]:
    from cayu.tasks._group_quiescence import validate_reconciliation_page

    validate_reconciliation_page(after_group_id, limit)
    async with store._lock:
        rows = store._connection.execute(
            "SELECT group_id FROM cayu_task_groups WHERE barrier_status = 'draining' "
            "AND group_id > ? ORDER BY group_id LIMIT ?",
            (after_group_id or "", limit),
        ).fetchall()
        return [row[0] for row in rows]


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


async def reconcile(
    store: SQLiteTaskStore,
    group_id: str,
    *,
    resolution: TaskGroupQuiescenceResolution | None = None,
    settled_execution: tuple[str, str, datetime] | None = None,
    invocation: TaskGroupInvocationObligation | None = None,
) -> TaskGroupSnapshot:
    from cayu.storage._sqlite_task_graphs import record_transition
    from cayu.tasks._group_quiescence import (
        prepare_resolution,
        reconciliation_is_complete,
        require_resolution,
        retained_execution_settlement_matches,
        retained_invocation_settlement_matches,
    )

    group_id = graph_identifier(group_id)
    prepared = None if resolution is None else prepare_resolution(resolution)
    async with store._lock:
        with store._verified_transaction_unlocked():
            snapshot = read_group(store, group_id)
            if snapshot is None:
                raise KeyError("Task group not found.")
            if prepared is not None:
                resolution, digest = prepared
                prior = store._connection.execute(
                    "SELECT request_sha256, snapshot_json FROM cayu_task_group_resolutions "
                    "WHERE group_id = ? AND idempotency_key = ?",
                    (group_id, resolution.idempotency_key),
                ).fetchone()
                if prior is not None:
                    if prior[0] != digest:
                        raise TaskGroupConflict("Resolution key has different content.")
                    return TaskGroupSnapshot.model_validate_json(prior[1])
                require_resolution(snapshot, resolution)
            if snapshot.receipt.quiescence is None:
                return snapshot
            if (
                resolution is None
                and reconciliation_is_complete(snapshot)
                and (
                    invocation is None
                    or retained_invocation_settlement_matches(snapshot, invocation)
                )
                and (
                    settled_execution is None
                    or retained_execution_settlement_matches(snapshot, settled_execution)
                )
            ):
                return snapshot
            task = store._load_task_unlocked(snapshot.receipt.member_task_ids[0])
            if task is None:
                raise TaskGroupConflict("Unsettled group lost task evidence.")
            record_transition(
                store,
                task,
                task,
                settled_execution=settled_execution,
                invocation=invocation,
                resolve_attention=resolution is not None,
            )
            result = read_group(store, group_id)
            assert result is not None
            if prepared is not None:
                store._connection.execute(
                    "INSERT INTO cayu_task_group_resolutions "
                    "(group_id, idempotency_key, request_sha256, snapshot_json) VALUES (?, ?, ?, ?)",
                    (group_id, prepared[0].idempotency_key, prepared[1], result.model_dump_json()),
                )
            return result


async def settle_execution(store: SQLiteTaskStore, task: Task) -> None:
    if task.started_at is None or task.worker_id is None:
        return
    async with store._lock:
        row = store._connection.execute(
            "SELECT group_id FROM cayu_task_groups WHERE graph_id IN "
            "(SELECT graph_id FROM cayu_task_graph_members WHERE task_id = ? "
            "UNION SELECT graph_id FROM cayu_task_group_retry_lineage WHERE task_id = ?)",
            (task.id, task.id),
        ).fetchone()
        snapshot = None if row is None else read_group(store, row[0])
        if (
            snapshot is None
            or snapshot.receipt.quiescence is None
            or not any(item.task_id == task.id for item in snapshot.quiescence.executions)
        ):
            return
    await reconcile(
        store,
        snapshot.receipt.group_id,
        settled_execution=(task.id, task.worker_id, task.started_at),
    )


async def cancellation_requested(store: SQLiteTaskStore, task_id: str) -> bool:
    from cayu._validation import require_clean_nonblank

    task_id = require_clean_nonblank(task_id, "task_id")
    async with store._lock:
        return cancellation_requested_unlocked(store, task_id)


async def observe_result_resolution(
    store: SQLiteTaskStore, task_id: str, decision_id: str, owner_id: str, *, settled: bool
) -> None:
    from cayu.tasks._group_quiescence import observe_result_resolution as plan

    async with store._lock:
        with store._verified_transaction_unlocked():
            ownership = read_ownership(store, task_id)
            if ownership is None:
                return
            snapshot = plan(
                ownership[0],
                ownership[1],
                task_id,
                decision_id,
                owner_id,
                settled=settled,
                now=store._ownership_clock(),
            )
            store._connection.execute(
                "UPDATE cayu_task_groups SET snapshot_json = ? WHERE group_id = ?",
                (snapshot.model_dump_json(), snapshot.receipt.group_id),
            )


async def observe_invocation(
    store: SQLiteTaskStore, invocation: TaskGroupInvocationObligation
) -> None:
    invocation = TaskGroupInvocationObligation.model_validate(
        invocation.model_dump(mode="python", warnings=False)
    )
    async with store._lock:
        row = store._connection.execute(
            "SELECT group_id FROM cayu_task_groups WHERE graph_id IN "
            "(SELECT graph_id FROM cayu_task_graph_members WHERE task_id = ? "
            "UNION SELECT graph_id FROM cayu_task_group_retry_lineage WHERE task_id = ?)",
            (invocation.task_id, invocation.task_id),
        ).fetchone()
        snapshot = None if row is None else read_group(store, row[0])
        if (
            snapshot is None
            or snapshot.receipt.quiescence is None
            or not any(
                item.task_id == invocation.task_id for item in snapshot.quiescence.executions
            )
        ):
            return
    await reconcile(store, snapshot.receipt.group_id, invocation=invocation)


def cancellation_requested_unlocked(store: SQLiteTaskStore, task_id: str) -> bool:
    ownership = read_ownership(store, task_id)
    return ownership is not None and ownership[1] in ownership[0].quiescence.loser_task_ids


async def retains_execution(store: SQLiteTaskStore, task_id: str) -> bool:
    from cayu._validation import require_clean_nonblank

    task_id = require_clean_nonblank(task_id, "task_id")
    async with store._lock:
        ownership = read_ownership(store, task_id)
        if ownership is None:
            return False
        receipt = ownership[0].receipt
        return receipt.quiescence is not None and ownership[1] in receipt.member_task_ids


def read_ownership(store: SQLiteTaskStore, task_id: str) -> tuple[TaskGroupSnapshot, str] | None:
    row = store._connection.execute(
        "SELECT groups.snapshot_json, ownership.root_task_id FROM cayu_task_groups groups "
        "JOIN (SELECT graph_id, task_id AS root_task_id FROM cayu_task_graph_members WHERE task_id = ? "
        "UNION ALL SELECT graph_id, root_task_id FROM cayu_task_group_retry_lineage WHERE task_id = ?) ownership "
        "ON groups.graph_id = ownership.graph_id",
        (task_id, task_id),
    ).fetchone()
    return None if row is None else (TaskGroupSnapshot.model_validate_json(row[0]), row[1])
