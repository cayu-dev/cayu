"""Bounded graph transactions and graph-before-task ownership for PostgreSQL."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, LiteralString

from cayu._validation import MAX_PORTABLE_JSON_INTEGER
from cayu.runtime import _verified_work_policy as verified
from cayu.storage import _postgres_support as pg
from cayu.tasks._graph_admission import prepare_graph_admission
from cayu.tasks._graphs import (
    GRAPH_TERMINAL_STATUSES,
    member_from_task,
    plan_graph_transition,
    require_graph_membership,
    require_member_authority,
)
from cayu.tasks.base import Task, TaskInvocationSnapshot
from cayu.tasks.graphs import (
    TASK_GRAPH_MAX_NODES,
    TaskGraphConflict,
    TaskGraphCreate,
    TaskGraphCreationReceipt,
    TaskGraphEvent,
    TaskGraphEventType,
    TaskGraphMember,
    TaskGraphSnapshot,
    TaskGraphUnavailable,
    copy_task_graph_create,
    graph_identifier,
    task_graph_request_sha256,
)

if TYPE_CHECKING:
    from cayu.storage.postgres import PostgresTaskStore


async def lock_graph(cur: Any, graph_id: str) -> None:
    # Two-key advisory namespace. Hash collisions only serialize unrelated graphs;
    # exact graph identities are still checked against the durable receipt.
    await cur.execute("SELECT pg_advisory_xact_lock(791, hashtext(%s))", (graph_id,))


@dataclass(frozen=True)
class TaskCandidateQuery:
    """Internal, bounded, non-locking task-ID selection; never caller-authored SQL."""

    sql: LiteralString
    parameters: tuple[Any, ...]


async def select_graph_scopes(
    cur: Any,
    queries: tuple[TaskCandidateQuery, ...],
) -> tuple[tuple[str, ...], ...]:
    """Lock the union before any phase acquires task rows.

    Each phase must recheck its predicates and restrict graph row locking to its
    selected IDs or their bounded, already-owned graphs. Ordinary tasks have no
    graph scope and may still use the native SKIP LOCKED queue scan.
    """
    selections = []
    for query in queries:
        await cur.execute(query.sql, query.parameters)
        selections.append(tuple(row[0] for row in await cur.fetchall()))
    await lock_task_graphs(cur, tuple(sorted({identity for ids in selections for identity in ids})))
    return tuple(selections)


async def lock_task_graphs(cur: Any, task_ids: tuple[str, ...]) -> None:
    """Call before taking task-row or task-identity locks, in canonical order."""
    if not task_ids:
        return
    await cur.execute(
        "SELECT DISTINCT graph_id FROM cayu_task_graph_members "
        "WHERE task_id = ANY(%s) ORDER BY graph_id",
        (list(task_ids),),
    )
    graph_ids = [row[0] for row in await cur.fetchall()]
    for graph_id in graph_ids:
        await lock_graph(cur, graph_id)


async def require_graph_lock(cur: Any, graph_id: str) -> None:
    """Fail closed on a missed caller, rather than acquire a lock after a row."""
    await cur.execute(
        "SELECT 1 FROM pg_locks WHERE locktype = 'advisory' "
        "AND pid = pg_backend_pid() AND classid = 791 "
        "AND objid::bigint = (hashtext(%s)::bigint & 4294967295) "
        "AND objsubid = 2 AND mode = 'ExclusiveLock' AND granted",
        (graph_id,),
    )
    if await cur.fetchone() is None:
        raise TaskGraphUnavailable("Task graph mutation lacks graph-before-task ownership.")


async def require_unreserved_identity(cur: Any, task_id: str) -> None:
    await cur.execute("SELECT 1 FROM cayu_task_graph_members WHERE task_id = %s", (task_id,))
    if await cur.fetchone() is not None:
        raise TaskGraphConflict("Retained graph member identity cannot be reused.")


async def _receipt(cur: Any, graph_id: str) -> TaskGraphCreationReceipt | None:
    await cur.execute("SELECT receipt_json FROM cayu_task_graphs WHERE graph_id = %s", (graph_id,))
    row = await cur.fetchone()
    if row is None:
        return None
    receipt = TaskGraphCreationReceipt.model_validate(pg._loads(row[0]))
    if receipt.graph_id != graph_id:
        raise TaskGraphUnavailable("Graph receipt index contradicts its authority.")
    return receipt


async def _members(cur: Any, graph_id: str) -> tuple[dict[str, tuple[str, ...]], dict[str, Any]]:
    await cur.execute(
        "SELECT task_id, prerequisites_json, terminal_json FROM cayu_task_graph_members "
        "WHERE graph_id = %s ORDER BY task_id LIMIT %s",
        (graph_id, TASK_GRAPH_MAX_NODES + 1),
    )
    rows = await cur.fetchall()
    return ({row[0]: tuple(pg._loads(row[1])) for row in rows}, {row[0]: row[2] for row in rows})


async def insert_events(cur: Any, events: tuple[TaskGraphEvent, ...]) -> None:
    for event in events:
        await cur.execute(
            "INSERT INTO cayu_task_graph_events (graph_id, sequence, event_json) VALUES (%s, %s, %s)",
            (event.graph_id, event.sequence, event.model_dump_json()),
        )


async def notify_readiness(cur: Any) -> None:
    from cayu.storage.postgres import _TASK_ADMISSION_NOTIFY_CHANNEL

    # PostgreSQL delivers this only after commit. It contains no task payload.
    await cur.execute("SELECT pg_notify(%s, %s)", (_TASK_ADMISSION_NOTIFY_CHANNEL, ""))


async def create_graph(
    store: PostgresTaskStore, request: TaskGraphCreate
) -> TaskGraphCreationReceipt:
    request = copy_task_graph_create(request)
    digest = task_graph_request_sha256(request)
    await store._ensure_ready()

    async def operation(_conn: Any, cur: Any) -> TaskGraphCreationReceipt:
        await lock_graph(cur, request.graph_id)
        existing = await _receipt(cur, request.graph_id)
        if existing is not None:
            if (
                existing.request_sha256 != digest
                or existing.submitted_request_sha256
                != (request._submitted_request_sha256 or digest)
                or existing.task_ids != tuple(node.task.task_id for node in request.nodes)
            ):
                raise TaskGraphConflict("Graph identity has different admission content.")
            return existing
        ids = tuple(node.task.task_id for node in request.nodes)
        for identity in ids:
            assert identity is not None
            await store._lock_verified_work_identity(cur, "task", identity)
            await require_unreserved_identity(cur, identity)
            if await store._load_task(cur, identity) is not None:
                raise TaskGraphConflict("Graph member identity is already occupied.")
        parents: dict[str, TaskInvocationSnapshot] = {}
        for identity in sorted(
            {
                node.task.parent_task_id
                for node in request.nodes
                if node.task.parent_task_id is not None and node.task.parent_task_id not in ids
            }
        ):
            await cur.execute(
                f"SELECT {pg.TASK_COLUMNS} FROM cayu_tasks WHERE id = %s FOR KEY SHARE",
                (identity,),
            )
            row = await cur.fetchone()
            if row is None:
                raise TaskGraphConflict("Graph parent authority is unavailable.")
            task = pg.task_from_row(row)
            parents[identity] = TaskInvocationSnapshot(
                id=task.id,
                session_id=task.session_id,
                session_instance_id=task.session_instance_id,
                invocation=task.invocation,
            )
        # Keep shared contract/session locks ordered across multi-member admission.
        contracts = [
            node.task.work_contract for node in request.nodes if node.task.work_contract is not None
        ]
        # Do not deduplicate by registration identity: each member's expected
        # fingerprint must match, including conflicting references to one version.
        for reference in sorted(contracts, key=lambda ref: (ref.contract_id, ref.version)):
            verified.require_contract_reference(
                await store._load_work_contract_row(cur, reference), reference
            )
        sessions = sorted(
            {node.task.session_id for node in request.nodes if node.task.session_id is not None}
        )
        # Ordinary contracted-task creation takes execution authority before its
        # INSERT enters the closure guard. Acquire every such scope first,
        # rather than interleaving the two lock families across graph members.
        for session_id in sessions:
            if any(
                node.task.session_id == session_id and node.task.work_contract is not None
                for node in request.nodes
            ):
                await store._ensure_session_authority(cur, session_id, "contracted")
        for session_id in sessions:
            await cur.execute(
                "SELECT pg_advisory_xact_lock_shared(hashtextextended(%s, 0))",
                ("cayu-task-session-closure:" + session_id,),
            )
            await cur.execute(
                "SELECT 1 FROM cayu_task_session_closure_claims WHERE session_id = %s",
                (session_id,),
            )
            if await cur.fetchone() is not None:
                raise TaskGraphConflict("Graph task session is owned by closure.")
        admission = prepare_graph_admission(
            request,
            digest=digest,
            now=await store._verified_evidence_now(cur),
            parents=parents,
            validate_task=lambda _request: None,
        )
        await cur.execute(
            "INSERT INTO cayu_task_graphs (graph_id, receipt_json) VALUES (%s, %s)",
            (request.graph_id, admission.receipt.model_dump_json()),
        )
        for task in admission.tasks:
            await cur.execute(
                f"INSERT INTO cayu_tasks ({pg.TASK_COLUMNS}) VALUES ({', '.join(['%s'] * len(pg.TASK_COLUMNS.split(', ')))})",
                pg.task_insert_values(task),
            )
            await cur.execute(
                "INSERT INTO cayu_task_graph_members (task_id, graph_id, prerequisites_json) VALUES (%s, %s, %s)",
                (task.id, request.graph_id, json.dumps(task.prerequisite_task_ids)),
            )
            await store._record_schedule_transition(cur, None, task)
        await insert_events(cur, admission.events)
        await notify_readiness(cur)
        return admission.receipt

    return await store._run_verified_work_mutation(operation)


async def record_transition(
    store: PostgresTaskStore, cur: Any, prior: Task | None, current: Task
) -> None:
    await cur.execute(
        "SELECT graph_id FROM cayu_task_graph_members WHERE task_id = %s", (current.id,)
    )
    row = await cur.fetchone()
    if row is None:
        if current.graph_id is not None or current.prerequisite_task_ids:
            raise TaskGraphUnavailable("Task graph admission evidence is missing.")
        return
    if prior is None:
        raise TaskGraphConflict("Retained graph member identity cannot be reused.")
    graph_id = row[0]
    await require_graph_lock(cur, graph_id)
    receipt = await _receipt(cur, graph_id)
    if receipt is None:
        raise TaskGraphUnavailable("Graph admission receipt is missing.")
    prerequisites, _ = await _members(cur, graph_id)
    require_graph_membership(graph_id, receipt, prerequisites)
    tasks = {}
    for identity in receipt.task_ids:
        task = await store._load_task(cur, identity)
        if task is None:
            raise TaskGraphUnavailable("Graph member evidence is missing.")
        tasks[identity] = task
    tasks[prior.id] = prior
    await cur.execute(
        "SELECT COALESCE(MAX(sequence), 0) + 1, "
        "COALESCE(bool_or(event_json->>'type' = %s AND event_json->>'task_id' = %s), false) "
        "FROM cayu_task_graph_events WHERE graph_id = %s",
        (str(TaskGraphEventType.READY), current.id, graph_id),
    )
    sequence, readiness_recorded = await cur.fetchone()
    transition = plan_graph_transition(
        graph_id=graph_id,
        prerequisites=prerequisites,
        current=tasks,
        proposed=current,
        proposed_readiness_recorded=readiness_recorded,
        first_sequence=sequence,
        now=await store._verified_evidence_now(cur),
    )
    for task in transition.tasks:
        await store._update_task_snapshot(cur, task)
        if task.id != current.id:
            await store._record_schedule_transition(cur, tasks[task.id], task)
        if task.status in GRAPH_TERMINAL_STATUSES:
            await cur.execute(
                "UPDATE cayu_task_graph_members SET terminal_json = %s WHERE task_id = %s AND graph_id = %s",
                (member_from_task(task).model_dump_json(), task.id, graph_id),
            )
    for settlement in transition.retry_settlements:
        await cur.execute(
            "INSERT INTO cayu_task_retry_settlements "
            "(task_id, idempotency_key, request_sha256, receipt_json, committed_at) "
            "VALUES (%s, %s, %s, %s, %s)",
            (
                settlement.task_id,
                settlement.idempotency_key,
                settlement.request_sha256,
                settlement.model_dump_json(),
                settlement.committed_at,
            ),
        )
    await insert_events(cur, transition.events)
    if any(event.type is TaskGraphEventType.READY for event in transition.events):
        await notify_readiness(cur)


async def load_graph(store: PostgresTaskStore, graph_id: str) -> TaskGraphSnapshot | None:
    graph_id = graph_identifier(graph_id)
    await store._ensure_ready()
    async with store._connection() as conn, conn.cursor() as cur:
        await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        receipt = await _receipt(cur, graph_id)
        if receipt is None:
            return None
        prerequisites, terminal = await _members(cur, graph_id)
        require_graph_membership(graph_id, receipt, prerequisites)
        members = []
        for identity in receipt.task_ids:
            task = await store._load_task(cur, identity)
            if task is not None:
                require_member_authority(graph_id, task, prerequisites[identity])
                member = member_from_task(task)
            else:
                if terminal[identity] is None:
                    raise TaskGraphUnavailable("Graph member evidence is missing.")
                member = TaskGraphMember.model_validate(pg._loads(terminal[identity]))
                if (
                    member.task_id != identity
                    or member.prerequisite_task_ids != prerequisites[identity]
                    or member.status not in GRAPH_TERMINAL_STATUSES
                ):
                    raise TaskGraphUnavailable("Retained graph member contradicts its authority.")
            members.append(member)
        await cur.execute(
            "SELECT MAX(sequence) FROM cayu_task_graph_events WHERE graph_id = %s", (graph_id,)
        )
        last_sequence = (await cur.fetchone())[0]
        return TaskGraphSnapshot(
            receipt=receipt, members=tuple(members), last_sequence=last_sequence
        )


async def list_events(
    store: PostgresTaskStore, graph_id: str, *, after_sequence: int, limit: int
) -> list[TaskGraphEvent]:
    graph_id = graph_identifier(graph_id)
    if type(after_sequence) is not int or not 0 <= after_sequence <= MAX_PORTABLE_JSON_INTEGER:
        raise ValueError("Invalid graph event cursor.")
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("Invalid graph event page size.")
    await store._ensure_ready()
    async with store._connection() as conn, conn.cursor() as cur:
        await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        if await _receipt(cur, graph_id) is None:
            raise KeyError("Task graph not found.")
        await cur.execute(
            "SELECT sequence, event_json FROM cayu_task_graph_events WHERE graph_id = %s "
            "AND sequence > %s ORDER BY sequence LIMIT %s",
            (graph_id, after_sequence, limit),
        )
        events = []
        for sequence, document in await cur.fetchall():
            event = TaskGraphEvent.model_validate(pg._loads(document))
            if (
                event.graph_id != graph_id
                or event.sequence != sequence
                or sequence != after_sequence + len(events) + 1
            ):
                raise TaskGraphUnavailable("Graph event indexes contradict durable evidence.")
            events.append(event)
        return events


async def require_deletion_ready(cur: Any, task_ids: tuple[str, ...]) -> None:
    await cur.execute(
        "SELECT 1 FROM cayu_task_graph_members WHERE graph_id IN "
        "(SELECT graph_id FROM cayu_task_graph_members WHERE task_id = ANY(%s)) "
        "AND terminal_json IS NULL LIMIT 1",
        (list(task_ids),),
    )
    if await cur.fetchone() is not None:
        raise TaskGraphConflict("Nonterminal graph retains its member tasks.")
