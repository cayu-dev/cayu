from __future__ import annotations

from uuid import uuid4

from cayu import (
    InvocationOrigin,
    InvocationOriginTrust,
    SessionExecutionSource,
    SessionInvocation,
    SessionInvocationBinding,
    SessionStore,
    TaskExecutionSource,
    TaskInvocation,
    TaskStore,
    session_invocation_from_task,
)


def unattributed_task_invocation() -> TaskInvocation:
    return TaskInvocation(
        origin=InvocationOrigin(trust=InvocationOriginTrust.UNATTRIBUTED),
        root_invocation_id=str(uuid4()),
        source=TaskExecutionSource.SDK_TASK,
    )


def unattributed_session_invocation_binding(session_id: str) -> SessionInvocationBinding:
    return SessionInvocationBinding(
        id=session_id,
        invocation=SessionInvocation(
            origin=InvocationOrigin(trust=InvocationOriginTrust.UNATTRIBUTED),
            root_invocation_id=str(uuid4()),
            root_session_id=session_id,
            source=SessionExecutionSource.SDK_RUN,
        ),
    )


async def stored_session_invocation(
    store: SessionStore,
    session_id: str,
) -> SessionInvocationBinding:
    snapshot = await store.load_invocation_snapshot(session_id)
    if snapshot is None:
        raise AssertionError(f"Session fixture not found: {session_id}")
    return SessionInvocationBinding(id=snapshot.id, invocation=snapshot.invocation)


async def task_backed_session_invocation(
    store: TaskStore,
    task_id: str,
    session_id: str,
) -> SessionInvocationBinding:
    task = await store.load_task(task_id)
    if task is None:
        raise AssertionError(f"Task fixture not found: {task_id}")
    return SessionInvocationBinding(
        id=session_id,
        invocation=session_invocation_from_task(
            task.invocation,
            session_id=session_id,
        ),
    )
