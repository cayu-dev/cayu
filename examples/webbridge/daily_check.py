"""External cron tick -> idempotent task -> worker -> WebBridge agent."""

from __future__ import annotations

import hashlib
from datetime import date

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
    Message,
    ResumeRequest,
    RunRequest,
    Session,
    SessionExecutionSource,
    SessionInvocationBinding,
    SessionStatus,
    Task,
    TaskCreate,
    TaskHandlerOutcome,
    TaskQuery,
    TaskStatus,
    TaskStore,
    WebBridge,
    run_task_worker,
)
from cayu._validation import require_clean_nonblank

DAILY_CHECK_TASK_TYPE = "webbridge_daily_public_page"


def register_daily_checker(
    app: CayuApp,
    bridge: WebBridge,
    *,
    model: str,
    environment_name: str | None = None,
) -> AgentSpec:
    """Register the worker-owned agent with the selected application profile."""

    return bridge.register_agent(
        app,
        AgentSpec(name="daily_web_checker", model=model),
        environment_name=environment_name,
    )


async def external_cron_tick(
    app: CayuApp,
    *,
    target_url: str,
    scheduled_day: date,
    environment_name: str | None = None,
) -> Task:
    """Idempotently enqueue one date-and-target check from an external scheduler."""

    task_id = daily_check_task_id(target_url=target_url, scheduled_day=scheduled_day)
    task_input = {
        "target_url": target_url,
        "scheduled_day": scheduled_day.isoformat(),
    }
    if environment_name is not None:
        if (
            type(environment_name) is not str
            or not environment_name.strip()
            or environment_name != environment_name.strip()
        ):
            raise ValueError("environment_name must be a clean nonblank string.")
        task_input["environment_name"] = environment_name
    request = TaskCreate(
        task_id=task_id,
        type=DAILY_CHECK_TASK_TYPE,
        title=f"Daily public-page check for {scheduled_day.isoformat()}",
        assigned_agent_name="daily_web_checker",
        input=task_input,
    )
    try:
        return await app.create_task(request)
    except ValueError:
        if app.task_store is None:
            raise
        existing = await app.task_store.load_task(task_id)
        if (
            existing is None
            or existing.type != request.type
            or existing.assigned_agent_name != request.assigned_agent_name
            or existing.input != request.input
        ):
            raise
        return existing


def daily_check_task_id(*, target_url: str, scheduled_day: date) -> str:
    identity = f"{scheduled_day.isoformat()}\0{target_url}".encode()
    return "web_daily_" + hashlib.sha256(identity).hexdigest()


async def handle_daily_check(
    app: CayuApp,
    task: Task,
    worker_id: str,
) -> TaskHandlerOutcome | None:
    """Run the registered WebBridge agent; the task/session result is durable."""

    target_url = task.input.get("target_url")
    if type(target_url) is not str:
        raise ValueError("Daily check task is missing target_url.")
    environment_name = task.input.get("environment_name")
    if environment_name is not None and type(environment_name) is not str:
        raise ValueError("Daily check task has an invalid environment_name.")
    session_id = f"session_{task.id}"
    existing_session = await app.session_store.load(session_id)
    if existing_session is None:
        if task.session_id is not None:
            raise RuntimeError("Daily check task references a missing attached session.")
        stream = app.run(
            RunRequest(
                agent_name=task.assigned_agent_name or "daily_web_checker",
                session_id=session_id,
                task_id=task.id,
                task_worker_id=worker_id,
                environment_name=environment_name,
                messages=[
                    Message.text(
                        "user",
                        "Fetch this public page with web_fetch, report material changes, "
                        f"and retain its canonical final URL as evidence: {target_url}",
                    )
                ],
            )
        )
    else:
        if task.session_id not in {None, session_id}:
            raise RuntimeError("Daily check task is attached to a conflicting session.")
        if app.task_store is None:
            raise RuntimeError("Daily check recovery requires the configured task store.")
        if task.status is TaskStatus.CLAIMED:
            task = await app.task_store.attach_task(
                task.id,
                session_id=session_id,
                session_invocation=SessionInvocationBinding(
                    id=existing_session.id,
                    session_instance_id=existing_session.instance_id,
                    invocation=existing_session.invocation,
                ),
                worker_id=worker_id,
            )
        handled, outcome = await _recover_existing_daily_check(
            app,
            task,
            existing_session,
            environment_name=environment_name,
            settlement_worker_id=worker_id,
        )
        if not handled:
            raise RuntimeError("Attached daily check session still has a live recovery owner.")
        return outcome

    outcome = None
    async for event in stream:
        if event.type == EventType.SESSION_INTERRUPTED:
            outcome = TaskHandlerOutcome.SESSION_INTERRUPTED
    return outcome


async def _recover_existing_daily_check(
    app: CayuApp,
    task: Task,
    session: Session,
    *,
    environment_name: str | None,
    settlement_worker_id: str | None,
) -> tuple[bool, TaskHandlerOutcome | None]:
    """Recover one exact attached daily session without fresh-task redelivery."""

    if app.task_store is None:
        raise RuntimeError("Daily check recovery requires the configured task store.")
    session_id = f"session_{task.id}"
    _require_daily_task_session_authority(
        task,
        session,
        session_id=session_id,
        environment_name=environment_name,
    )
    recovery = await app.recover_incomplete_session(
        IncompleteSessionRecoveryRequest(session_id=session_id)
    )
    if any(
        action
        in {
            IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,
            IncompleteSessionRecoveryAction.SKIPPED_UNREGISTERED_AGENT,
        }
        for action in recovery.actions
    ):
        return False, None
    if recovery.pending_approval_id is not None or recovery.pending_user_input_id is not None:
        return True, TaskHandlerOutcome.SESSION_INTERRUPTED
    if recovery.status is not SessionStatus.INTERRUPTED:
        if recovery.status in {SessionStatus.COMPLETED, SessionStatus.FAILED}:
            recovered_session = await app.session_store.load(session_id)
            if recovered_session is None:
                raise RuntimeError("Recovered daily check session disappeared.")
            await _settle_daily_task_from_terminal_session(
                app.task_store,
                task,
                worker_id=settlement_worker_id,
                session=recovered_session,
                environment_name=environment_name,
            )
            return True, None
        raise RuntimeError("Attached daily check session is not ready for worker recovery.")
    if settlement_worker_id is not None and task.worker_id == settlement_worker_id:
        await app.task_store.release_attached_task_worker(task.id, settlement_worker_id)
    stream = app.resume(
        ResumeRequest(
            session_id=session_id,
            messages=[
                Message.text(
                    "user",
                    "Continue the attached daily page check after worker recovery; do not "
                    "repeat effects already recorded in the session.",
                )
            ],
        )
    )
    outcome = None
    async for event in stream:
        if event.type == EventType.SESSION_INTERRUPTED:
            outcome = TaskHandlerOutcome.SESSION_INTERRUPTED
    return True, outcome


async def _settle_daily_task_from_terminal_session(
    task_store: TaskStore,
    task: Task,
    *,
    worker_id: str | None,
    session: Session,
    environment_name: str | None,
) -> None:
    current = await task_store.load_task(task.id)
    if current is None:
        raise RuntimeError("Daily check task disappeared during session reconciliation.")
    if current.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}:
        return
    _require_daily_task_session_authority(
        current,
        session,
        session_id=session.id,
        environment_name=environment_name,
    )
    if session.status is SessionStatus.COMPLETED:
        result = {
            "session_id": session.id,
            "agent_name": session.agent_name,
            "environment_name": session.environment_name,
        }
        try:
            await task_store.complete_task(
                current.id,
                result,
                worker_id=worker_id,
            )
        except ValueError as conflict:
            settled = await task_store.load_task(current.id)
            if (
                settled is None
                or settled.status is not TaskStatus.COMPLETED
                or settled.result != result
                or settled.error is not None
            ):
                raise conflict
        return
    if session.status is SessionStatus.FAILED:
        error = {
            "session_id": session.id,
            "message": "The attached daily check session failed.",
        }
        try:
            await task_store.fail_task(
                current.id,
                error,
                worker_id=worker_id,
            )
        except ValueError as conflict:
            settled = await task_store.load_task(current.id)
            if (
                settled is None
                or settled.status is not TaskStatus.FAILED
                or settled.result is not None
                or settled.error != error
            ):
                raise conflict
        return
    raise RuntimeError("Daily check task settlement requires a terminal session.")


def _require_daily_task_session_authority(
    task: Task,
    session: Session,
    *,
    session_id: str,
    environment_name: str | None,
) -> None:
    expected_agent = task.assigned_agent_name or "daily_web_checker"
    task_invocation = task.invocation
    session_invocation = session.invocation
    if (
        task.session_id != session_id
        or session.id != session_id
        or session.agent_name != expected_agent
        or (environment_name is not None and session.environment_name != environment_name)
        or session_invocation.source is not SessionExecutionSource.TASK
        or session_invocation.root_session_id != session_id
        or task_invocation.origin != session_invocation.origin
        or task_invocation.root_invocation_id != session_invocation.root_invocation_id
        or (
            task_invocation.root_session_id is not None
            and task_invocation.root_session_id != session_invocation.root_session_id
        )
    ):
        raise RuntimeError("Daily check task does not own the existing session invocation.")


async def daily_check_worker(
    app: CayuApp,
    task_store: TaskStore,
    *,
    worker_id: str,
    max_tasks: int | None = None,
) -> int:
    worker_id = require_clean_nonblank(worker_id, "worker_id")
    if app.redact_json(worker_id) != worker_id:
        raise ValueError(
            "worker_id contains a workload secret and cannot be used as durable task authority."
        )
    if max_tasks is not None and max_tasks < 0:
        raise ValueError("max_tasks must be non-negative.")

    recovered = await _settle_ownerless_terminal_daily_checks(
        app,
        task_store,
        max_tasks=max_tasks,
    )
    remaining = None if max_tasks is None else max_tasks - recovered
    if remaining == 0:
        return recovered
    handled = await run_task_worker(
        app,
        task_store,
        handle_daily_check,
        worker_id=worker_id,
        query=TaskQuery(type=DAILY_CHECK_TASK_TYPE),
        max_tasks=remaining,
    )
    return recovered + handled


async def _settle_ownerless_terminal_daily_checks(
    app: CayuApp,
    task_store: TaskStore,
    *,
    max_tasks: int | None,
) -> int:
    """Settle bounded terminal work whose task no longer has a lease owner."""

    if max_tasks == 0:
        return 0
    candidates = await task_store.list_tasks(
        TaskQuery(
            status=TaskStatus.RUNNING,
            type=DAILY_CHECK_TASK_TYPE,
            limit=1000,
        )
    )
    recovered = 0
    for task in candidates:
        if max_tasks is not None and recovered >= max_tasks:
            break
        if task.session_id is None:
            continue
        if task.worker_id is not None or task.lease_expires_at is not None:
            continue
        expected_session_id = f"session_{task.id}"
        if task.session_id != expected_session_id:
            raise RuntimeError("Daily check task is attached to a conflicting session.")
        session = await app.session_store.load(expected_session_id)
        if session is None:
            raise RuntimeError("Daily check task references a missing attached session.")
        environment_name = task.input.get("environment_name")
        if environment_name is not None and type(environment_name) is not str:
            raise ValueError("Daily check task has an invalid environment_name.")
        _require_daily_task_session_authority(
            task,
            session,
            session_id=expected_session_id,
            environment_name=environment_name,
        )
        if session.status not in {SessionStatus.COMPLETED, SessionStatus.FAILED}:
            continue
        handled, _ = await _recover_existing_daily_check(
            app,
            task,
            session,
            environment_name=environment_name,
            settlement_worker_id=None,
        )
        if handled:
            recovered += 1
    return recovered


async def load_durable_daily_result(task_store: TaskStore, task_id: str) -> Task:
    """Load the terminal task that an app may translate into a notification."""

    task = await task_store.load_task(task_id)
    if task is None or task.status not in {TaskStatus.COMPLETED, TaskStatus.FAILED}:
        raise RuntimeError("Daily check has no durable terminal result yet.")
    return task
