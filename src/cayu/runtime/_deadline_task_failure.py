"""Authenticate settlement-only recovery of a task failed by its deadline."""

from __future__ import annotations

from cayu.events import EventType
from cayu.runtime._continuation_task_failure import runtime_task_terminalization_idempotency_key
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.store import SessionStore
from cayu.tasks.records import Task, TaskStatus
from cayu.tasks.store import TaskStore
from cayu.tasks.terminalization import TaskTerminalKind


async def load_deadline_task_failure(
    task_store: TaskStore,
    session_store: SessionStore,
    *,
    task_id: str,
    session_id: str,
    session_instance_id: str,
    task_worker_id: str | None,
    task_handoff_id: str | None,
) -> Task | None:
    """Return exact terminal evidence, never authority to run another model step."""
    if not task_store.supports_idempotent_terminalization:
        return None
    task = await task_store.load_task(task_id)
    if (
        task is None
        or task.status is not TaskStatus.FAILED
        or task.session_id != session_id
        or task.session_instance_id != session_instance_id
        or task.worker_id is not None
        or task.lease_expires_at is not None
        or task.interrupted_handoff_id is not None
        or type(task.error) is not dict
        or task.error.get("error") != "ExecutionDeadlineExceeded"
    ):
        return None
    marker = task.error.get("deadline_failure")
    if (
        type(marker) is not dict
        or marker.get("session_id") != session_id
        or marker.get("session_instance_id") != session_instance_id
        or marker.get("task_handoff_id") != task_handoff_id
        or type(marker.get("terminal_event_id")) is not str
    ):
        return None
    session = await session_store.load(session_id)
    if session is None or session.instance_id != session_instance_id:
        return None
    records = await session_store.query_events(
        EventQuery(session_id=session_id, event_id=marker["terminal_event_id"], limit=2)
    )
    if len(records) != 1:
        return None
    terminal = records[0].event
    evidence = terminal.payload.get("failure_evidence")
    if (
        terminal.type != EventType.SESSION_INTERRUPTED
        or type(evidence) is not dict
        or evidence.get("classification") != "deadline"
        or evidence.get("session_id") != session_id
        or evidence.get("run_epoch") != marker.get("run_epoch")
        or evidence != task.error.get("failure_evidence")
    ):
        return None
    receipt = await task_store.load_task_terminalization_receipt(
        task_id,
        runtime_task_terminalization_idempotency_key(
            task_id=task_id, session_id=session_id, kind=TaskTerminalKind.FAILED
        ),
    )
    if task_worker_id is None:
        return task if receipt is None and task_handoff_id is None else None
    if (
        receipt is not None
        and receipt.worker_id == task_worker_id
        and receipt.kind is TaskTerminalKind.FAILED
        and receipt.task == task
    ):
        return task
    return None
