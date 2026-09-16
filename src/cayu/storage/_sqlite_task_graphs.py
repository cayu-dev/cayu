"""SQLite graph publication under the owning TaskStore transaction."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from typing import TYPE_CHECKING

from cayu._validation import MAX_PORTABLE_JSON_INTEGER
from cayu.storage import _sqlite_support as sql
from cayu.tasks._graph_admission import prepare_graph_admission
from cayu.tasks._graphs import (
    GRAPH_TERMINAL_STATUSES,
    member_from_task,
    plan_graph_transition,
    require_graph_membership,
    require_member_authority,
)
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

if TYPE_CHECKING:
    from cayu.storage.sqlite import SQLiteTaskStore


async def create_graph(
    store: SQLiteTaskStore, request: TaskGraphCreate
) -> TaskGraphCreationReceipt:
    request = copy_task_graph_create(request)
    digest = task_graph_request_sha256(request)
    async with store._lock:
        with store._verified_transaction_unlocked():
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
    store._publish_task_admission_broadcast()
    return admission.receipt


def require_unreserved_identity(store: SQLiteTaskStore, task_id: str) -> None:
    if (
        store._connection.execute(
            "SELECT 1 FROM cayu_task_graph_members WHERE task_id = ?", (task_id,)
        ).fetchone()
        is not None
    ):
        raise TaskGraphConflict("Retained graph member identity cannot be reused.")


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


def record_transition(store: SQLiteTaskStore, prior: Task | None, current: Task) -> None:
    row = store._connection.execute(
        "SELECT graph_id FROM cayu_task_graph_members WHERE task_id = ?", (current.id,)
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
    transition = plan_graph_transition(
        graph_id=graph_id,
        prerequisites=prerequisites,
        current=tasks,
        proposed=current,
        proposed_readiness_recorded=bool(readiness_recorded),
        first_sequence=sequence,
        now=store._ownership_clock(),
    )
    for task in transition.tasks:
        store._update_task_snapshot_unlocked(task)
        if task.id != current.id:
            store._record_schedule_transition_unlocked(tasks[task.id], task)
        if task.status in GRAPH_TERMINAL_STATUSES:
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
    for identity in task_ids:
        if (
            store._connection.execute(
                "SELECT 1 FROM cayu_task_graph_members WHERE graph_id = "
                "(SELECT graph_id FROM cayu_task_graph_members WHERE task_id = ?) AND terminal_json IS NULL LIMIT 1",
                (identity,),
            ).fetchone()
            is not None
        ):
            raise TaskGraphConflict("Nonterminal graph retains its member tasks.")
