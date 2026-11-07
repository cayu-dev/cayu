"""Durable-state invariants checked after every injected fault and after recovery."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from cayu.events import EventType
from cayu.messages import ToolCallPart, ToolResultPart
from cayu.sessions.queries import SessionQuery
from cayu.sessions.records import SessionStatus
from cayu.sessions.store import SessionStore
from cayu.tasks.records import TaskStatus
from cayu.tasks.store import TaskStore

UNSETTLED_SESSION_STATUSES = frozenset({SessionStatus.RUNNING, SessionStatus.INTERRUPTING})
TERMINAL_SESSION_STATUSES = frozenset(
    {SessionStatus.COMPLETED, SessionStatus.FAILED, SessionStatus.INTERRUPTED}
)
_INTERACTION_OPENED = frozenset({EventType.INTERACTION_STARTED, EventType.INTERACTION_RESUMED})
_INTERACTION_SETTLED = frozenset(
    {
        EventType.INTERACTION_COMPLETED,
        EventType.INTERACTION_FAILED,
        EventType.INTERACTION_INTERRUPTED,
        EventType.INTERACTION_PAUSED,
    }
)
_HELD_EXECUTION_STATES = frozenset({"executing", "owner_lost"})
TERMINAL_TASK_STATUSES = frozenset({TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED})
_HELD_TASK_STATUSES = frozenset({TaskStatus.CLAIMED, TaskStatus.RUNNING})


class Stage:
    """Where in a case a violation was found."""

    SETUP = "setup"
    PHASE = "phase"
    CRASH = "crash"
    LIVE = "live"
    SHUTDOWN = "shutdown"
    AFTER_SHUTDOWN = "after-shutdown"
    RECOVERY = "recovery"
    TASK_RECOVERY = "task-recovery"
    RECOVERED = "recovered"
    LIVENESS = "liveness"
    RECOVERY_SHUTDOWN = "recovery-shutdown"
    TEARDOWN = "teardown"


class Invariant:
    """Names of the checked invariants; a violation is reported under one of them."""

    SESSION_UNSETTLED = "session-running-without-owner"
    EXECUTION_LEASE_HELD = "execution-lease-held-by-dead-owner"
    INTERACTION_UNSETTLED = "interaction-started-never-settled"
    TRANSCRIPT_INDEXES = "transcript-indexes-inconsistent"
    TRANSCRIPT_TOOL_PAIRING = "transcript-tool-result-without-call"
    TASK_UNSETTLED = "task-held-by-exited-worker"
    TASK_SESSION_DISAGREE = "task-outcome-disagrees-with-session"
    TASK_GROUP_OBLIGATION = "task-group-execution-obligation-unsettled"
    SHUTDOWN_HANG = "shutdown-or-recovery-hangs"
    NOT_RESUMABLE = "session-cannot-continue-after-recovery"


async def store_violations(store: SessionStore) -> list[tuple[str, str]]:
    """Violations across every session the scenario created."""

    violations: list[tuple[str, str]] = []
    for session in (await store.list_sessions(SessionQuery(limit=100))).sessions:
        violations.extend(
            (invariant, f"{session.id}: {detail}")
            for invariant, detail in await session_violations(store, session.id)
        )
    return violations


async def session_violations(store: SessionStore, session_id: str) -> list[tuple[str, str]]:
    """Violations of one session's settled durable state, with a short detail each."""

    session = await store.load(session_id)
    if session is None:
        return []
    violations: list[tuple[str, str]] = []
    if session.status in UNSETTLED_SESSION_STATUSES:
        violations.append((Invariant.SESSION_UNSETTLED, f"status={session.status.value}"))
    execution = await store.inspect_session_execution(session_id)
    if execution.state in _HELD_EXECUTION_STATES:
        violations.append(
            (
                Invariant.EXECUTION_LEASE_HELD,
                f"state={execution.state} owner={execution.owner_kind}",
            )
        )
    if session.status in TERMINAL_SESSION_STATUSES:
        opened: dict[str, EventType] = {}
        for event in await store.load_events(session_id):
            if event.interaction_id is None:
                continue
            if event.type in _INTERACTION_OPENED:
                opened[event.interaction_id] = event.type
            elif event.type in _INTERACTION_SETTLED:
                opened.pop(event.interaction_id, None)
        if opened:
            violations.append(
                (
                    Invariant.INTERACTION_UNSETTLED,
                    f"status={session.status.value} open={sorted(set(opened.values()))}",
                )
            )
    violations.extend(await transcript_violations(store, session_id))
    return violations


async def transcript_violations(store: SessionStore, session_id: str) -> list[tuple[str, str]]:
    snapshot = await store.load_transcript_snapshot(session_id)
    violations: list[tuple[str, str]] = []
    indexes = [record.index for record in snapshot.records]
    if indexes != list(range(snapshot.cursor)):
        violations.append(
            (Invariant.TRANSCRIPT_INDEXES, f"indexes={indexes} cursor={snapshot.cursor}")
        )
    called: set[str] = set()
    for record in snapshot.records:
        for part in record.message.content:
            if type(part) is ToolCallPart:
                called.add(part.tool_call_id)
            elif type(part) is ToolResultPart and part.tool_call_id not in called:
                violations.append(
                    (
                        Invariant.TRANSCRIPT_TOOL_PAIRING,
                        f"index={record.index} call={part.tool_call_id}",
                    )
                )
    return violations


async def task_violations(
    tasks: TaskStore, store: SessionStore, *, unacknowledged_at: datetime | None = None
) -> list[tuple[str, str]]:
    """Tasks held by a worker that has exited, or settled unlike their session.

    Every check runs after the phase's worker has returned or died, so a claimed
    or running task still names an owner that no longer exists. A task whose
    linked session completed must have completed with it, and the reverse, except
    that a durable cancellation request still decides a task whose session
    completed.
    An unacknowledged, unstarted claim may await its live lease expiry; recovery
    checks pass no exception and require that a second worker settles it.
    """

    violations: list[tuple[str, str]] = []
    for task in await tasks.list_tasks():
        awaiting_expiry = (
            unacknowledged_at is not None
            and task.status is TaskStatus.CLAIMED
            and task.started_at is None
            and task.lease_expires_at is not None
            and task.lease_expires_at > unacknowledged_at
        )
        if task.status in _HELD_TASK_STATUSES and not awaiting_expiry:
            violations.append(
                (
                    Invariant.TASK_UNSETTLED,
                    f"{task.id}: status={task.status.value} worker={task.worker_id}",
                )
            )
        if task.session_id is None:
            continue
        session = await store.load(task.session_id)
        if session is None:
            continue
        session_completed = session.status is SessionStatus.COMPLETED
        task_completed = task.status is TaskStatus.COMPLETED
        if session_completed and task.status is TaskStatus.CANCELLED:
            # A task reaches CANCELLED only through a durable cancellation request,
            # which wins task terminalization over its session's committed
            # completion: the session stays completed and the task cancelled.
            continue
        if session_completed != task_completed and (
            task.status in TERMINAL_TASK_STATUSES or session.status in TERMINAL_SESSION_STATUSES
        ):
            violations.append(
                (
                    Invariant.TASK_SESSION_DISAGREE,
                    f"{task.id}: task={task.status.value} session={session.status.value}",
                )
            )
    return violations


async def task_group_violations(app: Any, group_id: str) -> list[tuple[str, str]]:
    """Every execution a group recorded must be settled once its member is terminal."""

    snapshot = await app.load_task_group(group_id)
    if snapshot is None:
        return []
    tasks = {
        task_id: await app.task_store.load_task(task_id)
        for task_id in {execution.task_id for execution in snapshot.quiescence.executions}
    }
    return [
        (
            Invariant.TASK_GROUP_OBLIGATION,
            f"{execution.task_id}: worker={execution.worker_id} "
            f"task={tasks[execution.task_id].status.value} "
            f"quiescence={snapshot.quiescence.status.value}",
        )
        for execution in snapshot.quiescence.executions
        if execution.settled_at is None
        and tasks[execution.task_id] is not None
        and tasks[execution.task_id].status in TERMINAL_TASK_STATUSES
    ]
