"""Freeze a terminal task set before permanently abandoning its session."""

from __future__ import annotations

from hashlib import sha256

from cayu.sessions.records import Session, SessionStatusConflict
from cayu.tasks.queries import TaskOrder, TaskQuery
from cayu.tasks.records import _TERMINAL_TASK_STATUSES, TaskSessionClosureClaim
from cayu.tasks.store import TaskStore


def abandonment_task_plan_id(session_id: str) -> str:
    return sha256(f"cayu:session-abandonment:v1:{session_id}".encode()).hexdigest()


async def claim_abandonment_task_fence(
    store: TaskStore | None, session: Session, *, task_id: str | None
) -> None:
    if store is None:
        if task_id is not None:
            raise SessionStatusConflict("Abandoning an attached task requires its task store.")
        return
    if not store.supports_session_closure_claims:
        raise NotImplementedError("Session abandonment requires atomic task-set closure claims.")
    if not store.supports_session_retention_epochs:
        raise NotImplementedError("Session abandonment requires task retention epoch fencing.")
    session_id = session.id
    plan_id = abandonment_task_plan_id(session_id)
    retained = await store.load_session_closure_claim(session_id)
    if retained is not None:
        if retained.session_id != session_id or retained.plan_id != plan_id:
            raise SessionStatusConflict("Session tasks already belong to another closure.")
        return
    if task_id is not None:
        attached = await store.load_task(task_id)
        if attached is not None and (
            attached.status not in _TERMINAL_TASK_STATUSES
            or attached.worker_id is not None
            or attached.lease_expires_at is not None
        ):
            raise SessionStatusConflict("Abandonment requires quiescent terminal tasks.")
    task_ids: list[str] = []
    while True:
        page = await store.list_tasks(
            TaskQuery(
                session_id=session_id,
                limit=1000,
                offset=len(task_ids),
                order_by=TaskOrder.CREATED_AT_ASC,
            )
        )
        if any(task.session_id != session_id for task in page) or len(page) > 1000:
            raise SessionStatusConflict("Task store returned an invalid session task set.")
        task_ids.extend(task.id for task in page)
        if len(task_ids) > 100_000:
            raise SessionStatusConflict("Session task set exceeds the abandonment claim limit.")
        if len(page) < 1000:
            break
    claim = TaskSessionClosureClaim(
        session_id=session_id,
        plan_id=plan_id,
        task_ids=tuple(task_ids),
        purpose="retention",
        retention_instance_id=session.instance_id,
        retention_run_epoch=session.run_epoch,
    )
    try:
        admitted = await store.claim_session_closure(claim)
    except ValueError as exc:
        raise SessionStatusConflict(
            "Session task set changed or is not quiescent and terminal."
        ) from exc
    if admitted != claim:
        raise SessionStatusConflict("Task store returned conflicting abandonment authority.")
