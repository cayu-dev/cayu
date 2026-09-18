"""SQLite graph publication under the owning TaskStore transaction."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime
from typing import TYPE_CHECKING

from cayu._validation import MAX_PORTABLE_JSON_INTEGER
from cayu.storage import _sqlite_support as sql
from cayu.storage import _sqlite_task_groups as groups
from cayu.tasks._graph_admission import prepare_graph_admission
from cayu.tasks._graphs import (
    GRAPH_TERMINAL_STATUSES,
    GraphTransition,
    member_from_task,
    plan_graph_transition,
    require_graph_membership,
    require_member_authority,
)
from cayu.tasks._groups import prepare_group_admission
from cayu.tasks.base import Task, TaskCreate
from cayu.tasks.graphs import (
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
from cayu.tasks.groups import TaskGroupCreate, TaskGroupInvocationObligation

if TYPE_CHECKING:
    from cayu.storage.sqlite import SQLiteTaskStore


async def create_graph(
    store: SQLiteTaskStore,
    request: TaskGraphCreate,
    *,
    group: TaskGroupCreate | None = None,
    submitted_digest: str | None = None,
) -> TaskGraphCreationReceipt:
    request = copy_task_graph_create(request)
    digest = task_graph_request_sha256(request)
    async with store._lock:
        with store._verified_transaction_unlocked():
            if group is not None:
                groups.check_admission(store, group, submitted_digest)
            row = store._connection.execute(
                "SELECT receipt_json FROM cayu_task_graphs WHERE graph_id = ?", (request.graph_id,)
            ).fetchone()
            if row is not None:
                receipt = TaskGraphCreationReceipt.model_validate_json(row[0])
                if receipt.graph_id != request.graph_id or receipt.task_ids != tuple(
                    node.task.task_id for node in request.nodes
                ):
                    raise TaskGraphUnavailable("Graph admission receipt contradicts its identity.")
                if receipt.request_sha256 != digest or receipt.submitted_request_sha256 != (
                    request._submitted_request_sha256 or digest
                ):
                    raise TaskGraphConflict("Graph identity has different admission content.")
                return receipt
            ids = {node.task.task_id for node in request.nodes}
            parents = {}
            for node in request.nodes:
                assert node.task.task_id is not None
                if store._task_exists_unlocked(node.task.task_id):
                    raise TaskGraphConflict("Graph member identity is already occupied.")
                require_unreserved_identity(store, node.task.task_id)
                if node.task.parent_task_id is not None and node.task.parent_task_id not in ids:
                    parent = store._task_parent_for_create_unlocked(
                        node.task, task_id=node.task.task_id
                    )
                    if parent is not None:
                        parents[parent.id] = parent

            def validate_task(request: TaskCreate) -> None:
                if request.work_contract is not None:
                    from cayu.runtime import _verified_work_policy as verified

                    verified.require_contract_reference(
                        store._load_work_contract_unlocked(request.work_contract),
                        request.work_contract,
                    )
                    if request.session_id is not None:
                        store._ensure_session_execution_authority_unlocked(
                            request.session_id, "contracted"
                        )
                if request.session_id is not None:
                    row = store._connection.execute(
                        "SELECT 1 FROM cayu_task_session_closure_claims WHERE session_id = ?",
                        (request.session_id,),
                    ).fetchone()
                    if row is not None:
                        raise TaskGraphConflict("Graph member session is owned by closure.")

            admission = prepare_graph_admission(
                request,
                digest=digest,
                # Match ordinary task admission's retry/evidence clock domain.
                now=store._clock(),
                parents=parents,
                validate_task=validate_task,
                finalizer_task_id=None if group is None else group.finalizer_task_id,
            )
            group_publication = (
                None
                if group is None
                else prepare_group_admission(group, admission, submitted_digest)
            )
            for task in admission.tasks:
                store._insert_task_unlocked(task)
                store._connection.execute(
                    "UPDATE cayu_tasks SET graph_id = ?, prerequisite_task_ids_json = ? WHERE id = ?",
                    (request.graph_id, json.dumps(task.prerequisite_task_ids), task.id),
                )
                store._record_schedule_transition_unlocked(None, task)
            store._connection.execute(
                "INSERT INTO cayu_task_graphs (graph_id, receipt_json) VALUES (?, ?)",
                (request.graph_id, admission.receipt.model_dump_json()),
            )
            store._connection.executemany(
                "INSERT INTO cayu_task_graph_members (task_id, graph_id, prerequisites_json) VALUES (?, ?, ?)",
                [
                    (task.id, request.graph_id, json.dumps(task.prerequisite_task_ids))
                    for task in admission.tasks
                ],
            )
            insert_events(store, admission.events)
            if group_publication is not None:
                groups.publish(store, group_publication, creating=True)
    store._publish_task_admission_broadcast()
    return admission.receipt


def require_unreserved_identity(store: SQLiteTaskStore, task_id: str) -> None:
    if (
        store._connection.execute(
            "SELECT 1 FROM cayu_task_graph_members WHERE task_id = ? "
            "UNION ALL SELECT 1 FROM cayu_task_group_retry_lineage WHERE task_id = ?",
            (task_id, task_id),
        ).fetchone()
        is not None
    ):
        raise TaskGraphConflict("Retained graph member identity cannot be reused.")


def register_retry_successor(store: SQLiteTaskStore, source: Task, successor: Task) -> None:
    """Called only by automatic retry settlement, within its native transaction."""
    row = store._connection.execute(
        "SELECT graph_id, task_id FROM cayu_task_graph_members WHERE task_id = ? "
        "UNION ALL SELECT graph_id, root_task_id FROM cayu_task_group_retry_lineage WHERE task_id = ?",
        (source.id, source.id),
    ).fetchone()
    if row is None:
        return
    group = store._connection.execute(
        "SELECT group_id FROM cayu_task_groups WHERE graph_id = ?", (row[0],)
    ).fetchone()
    snapshot = None if group is None else groups.read_group(store, group[0])
    if (
        snapshot is None
        or snapshot.receipt.quiescence is None
        or row[1] not in snapshot.receipt.member_task_ids
    ):
        return
    store._connection.execute(
        "INSERT INTO cayu_task_group_retry_lineage (task_id, graph_id, root_task_id) VALUES (?, ?, ?)",
        (successor.id, row[0], row[1]),
    )


def insert_events(store: SQLiteTaskStore, events: tuple[TaskGraphEvent, ...]) -> None:
    store._connection.executemany(
        "INSERT INTO cayu_task_graph_events (graph_id, sequence, event_json) VALUES (?, ?, ?)",
        [(event.graph_id, event.sequence, event.model_dump_json()) for event in events],
    )


def _notify_committed_readiness(store: SQLiteTaskStore, event: TaskGraphEvent) -> None:
    """A deferred hint requires the exact READY event to have committed.

    SQLite mutation bodies contain no awaits. Deferring to the loop lets their
    native transaction finish, including callers using explicit BEGIN/COMMIT.
    A later publication failure rolls the event back and emits no wakeup.
    """
    try:
        if store._connection.in_transaction:
            return
        row = store._connection.execute(
            "SELECT event_json FROM cayu_task_graph_events WHERE graph_id = ? AND sequence = ?",
            (event.graph_id, event.sequence),
        ).fetchone()
        if row is not None and row[0] == event.model_dump_json():
            store._publish_task_admission_broadcast()
    except sqlite3.Error:
        # Closing/unavailable stores retain periodic polling as their fallback.
        return


def record_transition(
    store: SQLiteTaskStore,
    prior: Task | None,
    current: Task,
    *,
    settled_execution: tuple[str, str, datetime] | None = None,
    invocation: TaskGroupInvocationObligation | None = None,
    resolve_attention: bool = False,
) -> None:
    from cayu.tasks._group_quiescence import (
        plan_group_graph_transition,
        require_group_mutation,
        waiting_finalizer,
    )

    row = store._connection.execute(
        "SELECT graph_id, task_id FROM cayu_task_graph_members WHERE task_id = ? "
        "UNION ALL SELECT graph_id, root_task_id FROM cayu_task_group_retry_lineage WHERE task_id = ?",
        (current.id, current.id),
    ).fetchone()
    if row is None:
        if current.graph_id is not None:
            raise TaskGraphUnavailable("Task graph admission evidence is missing.")
        return
    if prior is None:
        raise TaskGraphConflict("Retained graph member identity cannot be reused.")
    if not store._connection.in_transaction:
        raise TaskGraphUnavailable("Graph publication requires an owned transaction.")
    graph_id = row[0]
    root_id = row[1]
    rows = store._connection.execute(
        "SELECT task_id, prerequisites_json FROM cayu_task_graph_members WHERE graph_id = ? ORDER BY task_id",
        (graph_id,),
    ).fetchall()
    prerequisites = {row[0]: tuple(json.loads(row[1])) for row in rows}
    receipt_row = store._connection.execute(
        "SELECT receipt_json FROM cayu_task_graphs WHERE graph_id = ?",
        (graph_id,),
    ).fetchone()
    if receipt_row is None:
        raise TaskGraphUnavailable("Graph admission receipt is missing.")
    require_graph_membership(
        graph_id, TaskGraphCreationReceipt.model_validate_json(receipt_row[0]), prerequisites
    )
    tasks = {identity: store._require_task_unlocked(identity) for identity in prerequisites}
    tasks[prior.id] = prior
    sequence, readiness_recorded = store._connection.execute(
        "SELECT COALESCE(MAX(sequence), 0) + 1, "
        "COALESCE(MAX(CASE WHEN json_extract(event_json, '$.type') = ? "
        "AND json_extract(event_json, '$.task_id') = ? THEN 1 ELSE 0 END), 0) "
        "FROM cayu_task_graph_events WHERE graph_id = ?",
        (str(TaskGraphEventType.READY), current.id, graph_id),
    ).fetchone()
    group_row = store._connection.execute(
        "SELECT group_id FROM cayu_task_groups WHERE graph_id = ?", (graph_id,)
    ).fetchone()
    snapshot = None if group_row is None else groups.read_group(store, group_row[0])
    if snapshot is not None:
        require_group_mutation(snapshot, prior, current, root_task_id=root_id)
    transition = (
        GraphTransition(tasks=(current,), events=(), retry_settlements=())
        if current.id not in prerequisites
        else plan_graph_transition(
            graph_id=graph_id,
            prerequisites=prerequisites,
            current=tasks,
            proposed=current,
            proposed_readiness_recorded=bool(readiness_recorded),
            first_sequence=sequence,
            now=store._ownership_clock(),
            group_waiting_task_id=None if snapshot is None else waiting_finalizer(snapshot),
        )
    )
    group_publication = None
    if snapshot is not None:
        lineage_roots = dict(
            store._connection.execute(
                "SELECT task_id, root_task_id FROM cayu_task_group_retry_lineage WHERE graph_id = ? ORDER BY task_id",
                (graph_id,),
            ).fetchall()
        )
        if lineage_roots:
            rows = store._connection.execute(
                "SELECT * FROM cayu_tasks WHERE id IN ("
                "SELECT task_id FROM cayu_task_group_retry_lineage WHERE graph_id = ?)",
                (graph_id,),
            ).fetchall()
            descendants = {task.id: task for task in map(sql.task_from_row, rows)}
            if descendants.keys() != lineage_roots.keys():
                raise TaskGraphUnavailable("Retry descendant evidence is missing.")
            tasks.update(descendants)
        tasks[prior.id] = prior
        effect_rows = store._connection.execute(
            "SELECT DISTINCT member.task_id FROM cayu_task_graph_members member "
            "JOIN cayu_tasks task ON task.id = member.task_id "
            "JOIN cayu_local_execution_attempts effect ON "
            "(effect.task_id = task.id OR effect.retry_series_id = "
            "json_extract(task.retry_series_json, '$.series_id')) "
            "WHERE member.graph_id = ? AND effect.retry_admissible = 0",
            (graph_id,),
        ).fetchall()
        planned = plan_group_graph_transition(
            snapshot,
            transition,
            current=tasks,
            prerequisites=prerequisites,
            first_sequence=sequence,
            now=store._ownership_clock(),
            settled_execution=settled_execution,
            invocation=invocation,
            resolve_attention=resolve_attention,
            unsettled_effects=frozenset(row[0] for row in effect_rows),
            lineage_roots=lineage_roots,
        )
        transition, group_publication = planned.transition, planned.group
    for task in transition.tasks:
        store._update_task_snapshot_unlocked(task)
        if task.id != current.id:
            store._record_schedule_transition_unlocked(tasks[task.id], task)
        if task.status in GRAPH_TERMINAL_STATUSES and task.id in prerequisites:
            store._connection.execute(
                "UPDATE cayu_task_graph_members SET terminal_json = ? WHERE task_id = ?",
                (member_from_task(task).model_dump_json(), task.id),
            )
    for settlement in transition.retry_settlements:
        store._connection.execute(
            "INSERT INTO cayu_task_retry_settlements "
            "(task_id, idempotency_key, request_sha256, receipt_json, committed_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                settlement.task_id,
                settlement.idempotency_key,
                settlement.request_sha256,
                settlement.model_dump_json(),
                sql.format_datetime(settlement.committed_at),
            ),
        )
    insert_events(store, transition.events)
    if group_publication is not None:
        groups.publish(store, group_publication)
    ready = next(
        (event for event in transition.events if event.type is TaskGraphEventType.READY), None
    )
    if ready is not None:
        asyncio.get_running_loop().call_soon(_notify_committed_readiness, store, ready)


async def load_graph(store: SQLiteTaskStore, graph_id: str) -> TaskGraphSnapshot | None:
    graph_id = graph_identifier(graph_id)
    async with store._lock:
        with sql._transaction(store._connection, begin_immediate=False):
            row = store._connection.execute(
                "SELECT receipt_json FROM cayu_task_graphs WHERE graph_id = ?", (graph_id,)
            ).fetchone()
            if row is None:
                return None
            receipt = TaskGraphCreationReceipt.model_validate_json(row[0])
            rows = store._connection.execute(
                "SELECT task_id, prerequisites_json, terminal_json FROM cayu_task_graph_members "
                "WHERE graph_id = ? ORDER BY task_id LIMIT 129",
                (graph_id,),
            ).fetchall()
            prerequisites = {row[0]: tuple(json.loads(row[1])) for row in rows}
            retained = {row[0]: row[2] for row in rows}
            require_graph_membership(graph_id, receipt, prerequisites)
            members = []
            for identity in receipt.task_ids:
                task = store._load_task_unlocked(identity)
                if task is not None:
                    require_member_authority(graph_id, task, prerequisites[identity])
                    members.append(member_from_task(task))
                else:
                    document = retained[identity]
                    if document is None:
                        raise TaskGraphUnavailable("Graph member evidence is missing.")
                    member = TaskGraphMember.model_validate_json(document)
                    if (
                        member.task_id != identity
                        or member.prerequisite_task_ids != prerequisites[identity]
                        or member.status not in GRAPH_TERMINAL_STATUSES
                    ):
                        raise TaskGraphUnavailable(
                            "Retained graph member contradicts its authority."
                        )
                    members.append(member)
            last_sequence = store._connection.execute(
                "SELECT MAX(sequence) FROM cayu_task_graph_events WHERE graph_id = ?", (graph_id,)
            ).fetchone()[0]
            return TaskGraphSnapshot(
                receipt=receipt, members=tuple(members), last_sequence=last_sequence
            )


async def list_events(
    store: SQLiteTaskStore, graph_id: str, *, after_sequence: int, limit: int
) -> list[TaskGraphEvent]:
    graph_id = graph_identifier(graph_id)
    if type(after_sequence) is not int or not 0 <= after_sequence <= MAX_PORTABLE_JSON_INTEGER:
        raise ValueError("Invalid graph event cursor.")
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("Invalid graph event page size.")
    async with store._lock:
        with sql._transaction(store._connection, begin_immediate=False):
            if (
                store._connection.execute(
                    "SELECT 1 FROM cayu_task_graphs WHERE graph_id = ?", (graph_id,)
                ).fetchone()
                is None
            ):
                raise KeyError("Task graph not found.")
            rows = store._connection.execute(
                "SELECT sequence, event_json FROM cayu_task_graph_events WHERE graph_id = ? AND sequence > ? ORDER BY sequence LIMIT ?",
                (graph_id, after_sequence, limit),
            ).fetchall()
            events = []
            for row in rows:
                event = TaskGraphEvent.model_validate_json(row[1])
                if (
                    event.graph_id != graph_id
                    or event.sequence != row[0]
                    or event.sequence != after_sequence + len(events) + 1
                ):
                    raise TaskGraphUnavailable("Graph event indexes contradict durable evidence.")
                events.append(event)
            return events


def require_deletion_ready(store: SQLiteTaskStore, task_ids: tuple[str, ...]) -> None:
    from cayu.tasks._group_quiescence import require_group_deletion_ready

    checked_groups: set[str] = set()
    for identity in task_ids:
        for row in store._connection.execute(
            "SELECT group_id FROM cayu_task_groups WHERE graph_id IN "
            "(SELECT graph_id FROM cayu_task_graph_members WHERE task_id = ? "
            "UNION SELECT graph_id FROM cayu_task_group_retry_lineage WHERE task_id = ?)",
            (identity, identity),
        ).fetchall():
            if row[0] in checked_groups:
                continue
            snapshot = groups.read_group(store, row[0])
            assert snapshot is not None
            require_group_deletion_ready(snapshot)
            checked_groups.add(row[0])
        if (
            store._connection.execute(
                "SELECT 1 FROM cayu_task_graph_members WHERE graph_id IN "
                "(SELECT graph_id FROM cayu_task_graph_members WHERE task_id = ? "
                "UNION SELECT graph_id FROM cayu_task_group_retry_lineage WHERE task_id = ?) AND terminal_json IS NULL LIMIT 1",
                (identity, identity),
            ).fetchone()
            is not None
        ):
            raise TaskGraphConflict("Nonterminal graph retains its member tasks.")
