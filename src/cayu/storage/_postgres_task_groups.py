"""PostgreSQL group state uses the same ownership as its immutable graph."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from cayu.storage import _postgres_support as pg
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
    from cayu.storage.postgres import PostgresTaskStore
    from cayu.tasks.base import Task


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
    await cur.execute(
        "UPDATE cayu_task_groups SET barrier_status = %s, barrier_deadline = %s WHERE group_id = %s",
        (snapshot.quiescence.status.value, snapshot.quiescence.deadline, snapshot.receipt.group_id),
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


async def reconciliation_candidates(
    store: PostgresTaskStore,
    *,
    after_group_id: str | None,
    limit: int,
) -> list[str]:
    from cayu.tasks._group_quiescence import validate_reconciliation_page

    validate_reconciliation_page(after_group_id, limit)
    await store._ensure_ready()
    async with store._connection() as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT group_id FROM cayu_task_groups WHERE barrier_status = 'draining' "
            "AND group_id > %s ORDER BY group_id LIMIT %s",
            (after_group_id or "", limit),
        )
        return [row[0] for row in await cur.fetchall()]


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


async def reconcile(
    store: PostgresTaskStore,
    group_id: str,
    *,
    resolution: TaskGroupQuiescenceResolution | None = None,
    settled_execution: tuple[str, str, datetime] | None = None,
    invocation: TaskGroupInvocationObligation | None = None,
) -> TaskGroupSnapshot:
    from cayu.storage._postgres_task_graphs import lock_graph, record_transition
    from cayu.tasks._group_quiescence import (
        prepare_resolution,
        reconciliation_is_complete,
        require_resolution,
        retained_execution_settlement_matches,
        retained_invocation_settlement_matches,
    )

    group_id = graph_identifier(group_id)
    prepared = None if resolution is None else prepare_resolution(resolution)
    await store._ensure_ready()
    async with store._connection() as conn, conn.cursor() as cur:
        snapshot = await read_group(cur, group_id)
        if snapshot is None:
            raise KeyError("Task group not found.")
        await lock_graph(cur, snapshot.receipt.graph.graph_id)
        snapshot = await read_group(cur, group_id)
        assert snapshot is not None
        if prepared is not None:
            resolution, digest = prepared
            await cur.execute(
                "SELECT request_sha256, snapshot_json FROM cayu_task_group_resolutions "
                "WHERE group_id = %s AND idempotency_key = %s",
                (group_id, resolution.idempotency_key),
            )
            prior = await cur.fetchone()
            if prior is not None:
                if prior[0] != digest:
                    raise TaskGroupConflict("Resolution key has different content.")
                return TaskGroupSnapshot.model_validate(pg._loads(prior[1]))
            require_resolution(snapshot, resolution)
        if snapshot.receipt.quiescence is None:
            return snapshot
        if (
            resolution is None
            and reconciliation_is_complete(snapshot)
            and (invocation is None or retained_invocation_settlement_matches(snapshot, invocation))
            and (
                settled_execution is None
                or retained_execution_settlement_matches(snapshot, settled_execution)
            )
        ):
            return snapshot
        task = await store._load_task(cur, snapshot.receipt.member_task_ids[0])
        if task is None:
            raise TaskGroupConflict("Unsettled group lost task evidence.")
        await record_transition(
            store,
            cur,
            task,
            task,
            settled_execution=settled_execution,
            invocation=invocation,
            resolve_attention=resolution is not None,
        )
        result = await read_group(cur, group_id)
        assert result is not None
        if prepared is not None:
            await cur.execute(
                "INSERT INTO cayu_task_group_resolutions "
                "(group_id, idempotency_key, request_sha256, snapshot_json) VALUES (%s, %s, %s, %s)",
                (group_id, prepared[0].idempotency_key, prepared[1], result.model_dump_json()),
            )
        return result


async def settle_execution(store: PostgresTaskStore, task: Task) -> None:
    if task.started_at is None or task.worker_id is None:
        return
    await store._ensure_ready()
    async with store._connection() as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT group_id FROM cayu_task_groups WHERE graph_id IN "
            "(SELECT graph_id FROM cayu_task_graph_members WHERE task_id = %s "
            "UNION SELECT graph_id FROM cayu_task_group_retry_lineage WHERE task_id = %s)",
            (task.id, task.id),
        )
        row = await cur.fetchone()
        snapshot = None if row is None else await read_group(cur, row[0])
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


async def cancellation_requested(store: PostgresTaskStore, task_id: str) -> bool:
    from cayu._validation import require_clean_nonblank

    task_id = require_clean_nonblank(task_id, "task_id")
    await store._ensure_ready()
    async with store._connection() as conn, conn.cursor() as cur:
        return await read_cancellation_requested(cur, task_id)


async def observe_result_resolution(
    store: PostgresTaskStore, task_id: str, decision_id: str, owner_id: str, *, settled: bool
) -> None:
    from cayu.storage._postgres_task_graphs import lock_task_graphs
    from cayu.tasks._group_quiescence import observe_result_resolution as plan

    await store._ensure_ready()
    async with store._connection() as conn, conn.transaction(), conn.cursor() as cur:
        await lock_task_graphs(cur, (task_id,))
        ownership = await read_ownership(cur, task_id)
        if ownership is None:
            return
        await cur.execute("SELECT clock_timestamp()")
        row = await cur.fetchone()
        assert row is not None
        snapshot = plan(
            ownership[0],
            ownership[1],
            task_id,
            decision_id,
            owner_id,
            settled=settled,
            now=row[0],
        )
        await cur.execute(
            "UPDATE cayu_task_groups SET snapshot_json = %s WHERE group_id = %s",
            (snapshot.model_dump_json(), snapshot.receipt.group_id),
        )


async def observe_invocation(
    store: PostgresTaskStore, invocation: TaskGroupInvocationObligation
) -> None:
    invocation = TaskGroupInvocationObligation.model_validate(
        invocation.model_dump(mode="python", warnings=False)
    )
    await store._ensure_ready()
    async with store._connection() as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT group_id FROM cayu_task_groups WHERE graph_id IN "
            "(SELECT graph_id FROM cayu_task_graph_members WHERE task_id = %s "
            "UNION SELECT graph_id FROM cayu_task_group_retry_lineage WHERE task_id = %s)",
            (invocation.task_id, invocation.task_id),
        )
        row = await cur.fetchone()
        snapshot = None if row is None else await read_group(cur, row[0])
        if (
            snapshot is None
            or snapshot.receipt.quiescence is None
            or not any(
                item.task_id == invocation.task_id for item in snapshot.quiescence.executions
            )
        ):
            return
    await reconcile(store, snapshot.receipt.group_id, invocation=invocation)


async def read_cancellation_requested(cur: Any, task_id: str) -> bool:
    ownership = await read_ownership(cur, task_id)
    return ownership is not None and ownership[1] in ownership[0].quiescence.loser_task_ids


async def retains_execution(store: PostgresTaskStore, task_id: str) -> bool:
    from cayu._validation import require_clean_nonblank

    task_id = require_clean_nonblank(task_id, "task_id")
    await store._ensure_ready()
    async with store._connection() as conn, conn.cursor() as cur:
        ownership = await read_ownership(cur, task_id)
        if ownership is None:
            return False
        receipt = ownership[0].receipt
        return receipt.quiescence is not None and ownership[1] in receipt.member_task_ids


async def read_ownership(cur: Any, task_id: str) -> tuple[TaskGroupSnapshot, str] | None:
    await cur.execute(
        "SELECT groups.group_id, ownership.root_task_id FROM cayu_task_groups groups "
        "JOIN (SELECT graph_id, task_id AS root_task_id FROM cayu_task_graph_members WHERE task_id = %s "
        "UNION ALL SELECT graph_id, root_task_id FROM cayu_task_group_retry_lineage WHERE task_id = %s) ownership "
        "ON groups.graph_id = ownership.graph_id",
        (task_id, task_id),
    )
    row = await cur.fetchone()
    if row is None:
        return None
    snapshot = await read_group(cur, row[0])
    if snapshot is None:
        raise TaskGroupUnavailable("Group cancellation ownership lost its snapshot.")
    return snapshot, row[1]
