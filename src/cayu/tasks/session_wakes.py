"""Durable wake tasks that continue sessions asleep in ``sleep_until``.

``SleepUntilTool`` parks a top-level session at its tool call under its own
``pending_wake`` pause and creates one durable task, due at the resolved
deadline. ``run_task_worker`` runs those tasks itself: when the task store makes
one due, the parked call receives its timer result and the same session
continues once. A failed attempt schedules a successor with capped backoff for
``SESSION_WAKE_RETRY_WINDOW`` after the deadline, and :func:`reconcile_session_wakes`
recreates any attempt a crash left out. The public names are exported from
``cayu.tasks``.

A session that a task owns parks that task: its worker hands the task off parked
on the wake, so no generic continuation claims it, and the wake attempt continues
it through a bound claim that presents the same wake. The continuation is part of
the owning task, which completes, fails or parks again as its session does.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING

from cayu.approvals.actors import ResolutionActor, ResolutionActorSource
from cayu.approvals.user_input import (
    PENDING_USER_INPUT_CHECKPOINT_KEY,
    PendingUserInput,
    UserInputResponse,
    session_wake_task_id,
    session_wake_tool_result,
    user_input_lifecycle_authority_from_checkpoint,
)
from cayu.approvals.wakes import pending_session_wake_from_checkpoint
from cayu.sessions._checkpoint_mutations import SessionRuntimePublicationConflict
from cayu.sessions.pending_action_contracts import PendingActionQuery
from cayu.sessions.records import PendingActionKind, SessionStatus
from cayu.sessions.requests import InterruptSessionRequest
from cayu.tasks.continuation_claims import TaskContinuationExpectation
from cayu.tasks.creation import TaskCreate
from cayu.tasks.handoff import TaskInterruptedHandoffReceipt, TaskParkedWait
from cayu.tasks.queries import TaskQuery
from cayu.tasks.records import TaskStatus
from cayu.tasks.scheduling import TaskScheduleCancelRequest, TaskSchedulePolicy

if TYPE_CHECKING:
    from cayu.applications import CayuApp
    from cayu.sessions.records import Session
    from cayu.tasks.records import Task
    from cayu.tools.base import Tool

SESSION_WAKE_TASK_TYPE = "cayu.session-wake.v1"
SESSION_WAKE_ACTOR_SUBJECT = "cayu:session-wake"
#: How long after its deadline a wake keeps retrying failed attempts.
SESSION_WAKE_RETRY_WINDOW = timedelta(hours=24)
_RETRY_BACKOFF = timedelta(seconds=30)
_MAX_RETRY_BACKOFF = timedelta(hours=1)
# Backoff and window bound the attempts; this only bounds a scan of malformed data.
_MAX_ATTEMPT_SCAN = 128
# Store and application clocks can disagree by a little. A due task whose deadline
# the application clock has not reached yet waits this long at most, holding its
# lease; a larger disagreement fails the attempt instead of resuming early.
MAX_WAKE_CLOCK_SKEW = timedelta(minutes=5)
_TERMINAL_TASK_STATUSES = frozenset({TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED})
# A wake can come due before the worker that ran its session has handed the task
# off. Its attempt waits this long for the handoff before failing into a successor.
_OWNER_HANDOFF_WAIT = timedelta(seconds=5)
_OWNER_HANDOFF_POLL = timedelta(milliseconds=100)
# How long the attempt that continues a task-owned session holds its claim at once.
_CONTINUATION_LEASE_SECONDS = 300

_logger = logging.getLogger(__name__)


class SessionWakeOutcome(StrEnum):
    """What one wake delivery found and did."""

    CONTINUED = "continued"
    """This delivery resolved the sleeping call and continued the session."""

    ALREADY_DELIVERED = "already_delivered"
    """An earlier delivery already resolved the call; nothing ran again."""

    CLOSED = "closed"
    """The session is gone or no longer sleeping on this wake; nothing ran."""

    MANUAL_RECOVERY_REQUIRED = "manual_recovery_required"
    """The woken round stopped on a tool that started without a result.

    No later attempt continues it: the session's pending action is a manual
    recovery of that tool, which a person settles with ``recover_user_input``.
    """


class SessionWakeNotDue(RuntimeError):
    """The application clock has not reached the wake's deadline yet."""

    def __init__(self, deadline: datetime) -> None:
        self.deadline = deadline
        super().__init__(f"Session wake is not due until {deadline.isoformat()}.")


class SessionWakeOwnerBusy(RuntimeError):
    """The task that owns the session has not handed it off yet; the wake is retried."""

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        super().__init__(
            f"Task {task_id} still holds the sleeping session; its wake continues it once "
            "the task is handed off."
        )


class SessionWakeNotContinued(RuntimeError):
    """A delivery ended before the sleeping call received its result; it is retried."""

    def __init__(self, wake_id: str) -> None:
        self.wake_id = wake_id
        super().__init__("The session stopped before its sleeping call continued.")


def _attempt_task_id(wake_id: str, attempt: int) -> str:
    base = session_wake_task_id(wake_id)
    return base if attempt == 1 else f"{base}.{attempt}"


def _attempt_task(
    session_id: str,
    wake_id: str,
    deadline: datetime,
    attempt: int,
    available_at: datetime,
    retry_until: datetime | None = None,
) -> TaskCreate:
    return TaskCreate(
        task_id=_attempt_task_id(wake_id, attempt),
        type=SESSION_WAKE_TASK_TYPE,
        title="Wake a sleeping session",
        available_at=available_at,
        schedule_policy=TaskSchedulePolicy(),
        input={
            "session_id": session_id,
            "wake_id": wake_id,
            "deadline": deadline.astimezone(UTC).isoformat(),
            "attempt": attempt,
            # Fixed by the first attempt; ``None`` on the first attempt itself.
            "retry_until": None if retry_until is None else retry_until.astimezone(UTC).isoformat(),
        },
    )


def _successor(previous: Task) -> TaskCreate | None:
    """The attempt after ``previous``, fixed by it so every publisher builds the same one.

    ``None`` once the retry window is spent. The window starts when the first
    attempt ran, or at the deadline if it ran on time, so a wake whose workers
    were all down past the deadline still gets its retries.
    """

    session_id, wake_id, deadline, attempt, retry_until = _wake_task_identity(previous)
    if retry_until is None:
        first_ran = previous.started_at or previous.available_at or deadline
        retry_until = max(deadline, first_ran) + SESSION_WAKE_RETRY_WINDOW
    backoff = min(_RETRY_BACKOFF * 2 ** (attempt - 1), _MAX_RETRY_BACKOFF)
    # The later of when it was due and when it ran: an attempt cancelled before
    # it was due still waits out its own due time first.
    base = max(
        moment
        for moment in (previous.available_at, previous.started_at, previous.created_at)
        if moment is not None
    )
    available_at = base + backoff
    if available_at > retry_until or attempt >= _MAX_ATTEMPT_SCAN:
        return None
    return _attempt_task(session_id, wake_id, deadline, attempt + 1, available_at, retry_until)


async def _publish_session_wake(app: CayuApp, pending: PendingUserInput) -> None:
    """Create the first wake attempt for a committed pause; a replay converges on it."""

    wake = pending.wake
    if wake is None:
        raise ValueError("Pending user input is not a session wake.")
    await app.create_task(
        _attempt_task(pending.session_id, pending.input_id, wake.deadline, 1, wake.deadline)
    )


async def _open_wake(app: CayuApp, session: Session) -> tuple[PendingUserInput | None, bool]:
    """Return the session's open wake and whether it is still asleep.

    After ``wake-close`` the round waits, handed over, under the shared pause
    until a delivery continues it or an interruption supersedes it.
    """

    checkpoint = await app.session_store.load_checkpoint(session.id)
    asleep = pending_session_wake_from_checkpoint(checkpoint)
    if asleep is not None:
        return asleep.continuation, True
    if checkpoint is None or checkpoint.get(PENDING_USER_INPUT_CHECKPOINT_KEY) is None:
        return None, False
    pending, _ = user_input_lifecycle_authority_from_checkpoint(
        checkpoint,
        redactor=app._secret_redactor,
        current_run_epoch=session.run_epoch,
        runtime_session=session,
    )
    return (pending, False) if pending is not None and pending.wake is not None else (None, False)


async def _delivered(app: CayuApp, session_id: str, wake_id: str) -> bool:
    """The call took its timer result: its pause closed, or its round waits on a subagent."""

    from cayu.sessions._foreground_child_checkpoint import foreground_child_state_from_checkpoint

    receipt = await app.session_store.load_runtime_publication_receipt(
        session_id, f"user-input-close:{wake_id}"
    )
    if receipt is not None:
        return True
    # A subagent in the woken round paused for a person. The pause closes when
    # the subagent ends and the round continues from the accepted timer result.
    checkpoint = await app.session_store.load_checkpoint(session_id)
    if foreground_child_state_from_checkpoint(checkpoint)[0] is None:
        return False
    _, accepted = user_input_lifecycle_authority_from_checkpoint(checkpoint)
    return accepted is not None and accepted.input_id == wake_id


async def _needs_manual_recovery(app: CayuApp, session_id: str, wake_id: str) -> bool:
    """The woken round stopped on a tool a person must settle, so no attempt may retry it."""

    page = await app.session_store.query_pending_actions(
        PendingActionQuery(
            session_id=session_id,
            kind=PendingActionKind.MANUAL_RECOVERY,
            statuses=frozenset({SessionStatus.INTERRUPTED}),
        )
    )
    return any(action.input_id == wake_id for action in page.actions)


async def deliver_session_wake(
    app: CayuApp, *, session_id: str, wake_id: str
) -> SessionWakeOutcome:
    """Resolve one due wake with its timer result and continue the session once.

    Duplicate, late or competing deliveries return ``ALREADY_DELIVERED`` without
    running the model again. A wake whose session was interrupted, deleted or
    otherwise moved on returns ``CLOSED`` and changes nothing. A woken round that
    stopped on a tool that started without a result returns
    ``MANUAL_RECOVERY_REQUIRED``; settle that tool with ``recover_user_input``.
    Raises :class:`SessionWakeNotDue` before the deadline.

    When a task owns the session, the delivery claims that task's parked handoff
    as its one trigger and continues the session under the task, which then
    completes, fails or parks again with it. A wake whose owning task has ended,
    for example because it was cancelled, stops the sleep instead and returns
    ``CLOSED``. Raises :class:`SessionWakeOwnerBusy` while the task's worker has
    not handed it off yet.
    """

    owner = await _owning_task(app, session_id)
    if owner is not None:
        return await _deliver_to_owning_task(
            app, owner, session_id=session_id, wake_id=wake_id, worker_id=None
        )
    return await _deliver(app, session_id=session_id, wake_id=wake_id)


async def _deliver(
    app: CayuApp,
    *,
    session_id: str,
    wake_id: str,
    task_worker_id: str | None = None,
    task_handoff_id: str | None = None,
) -> SessionWakeOutcome:
    from cayu.runtime._session_wake_publication import close_pending_wake

    session = await app.session_store.load(session_id)
    if session is None:
        return SessionWakeOutcome.CLOSED
    if await _delivered(app, session_id, wake_id):
        return SessionWakeOutcome.ALREADY_DELIVERED
    pending, asleep = await _open_wake(app, session)
    if pending is None or pending.input_id != wake_id:
        return SessionWakeOutcome.CLOSED
    timer = pending.wake
    assert timer is not None
    if app._clock() < timer.deadline:
        raise SessionWakeNotDue(timer.deadline)
    if not asleep and await _needs_manual_recovery(app, session_id, wake_id):
        return SessionWakeOutcome.MANUAL_RECOVERY_REQUIRED
    if pending.task_id is not None and task_worker_id is None:
        # Its task ended while it slept; never continue it without the task.
        await _stop_sleep(app, session_id)
        return SessionWakeOutcome.CLOSED
    try:
        if asleep:
            await close_pending_wake(
                session_store=app.session_store,
                event_writer=app._event_writer,
                session=session,
                transition="woke",
            )
        response = UserInputResponse(
            session_id=session_id,
            input_id=wake_id,
            answer=session_wake_tool_result(pending),
            resolved_by=ResolutionActor(
                subject=SESSION_WAKE_ACTOR_SUBJECT,
                source=ResolutionActorSource.SYSTEM,
            ),
            task_worker_id=task_worker_id,
            task_handoff_id=task_handoff_id,
        )
        async for _ in app._resolve_user_input_private(response, wake_delivery=True):
            pass
    except SessionRuntimePublicationConflict:
        if await _delivered(app, session_id, wake_id):
            return SessionWakeOutcome.ALREADY_DELIVERED
        refreshed = await app.session_store.load(session_id)
        if refreshed is None or (await _open_wake(app, refreshed))[0] is None:
            return SessionWakeOutcome.CLOSED
        raise
    if not await _delivered(app, session_id, wake_id):
        if await _needs_manual_recovery(app, session_id, wake_id):
            return SessionWakeOutcome.MANUAL_RECOVERY_REQUIRED
        # The continuation stopped before the call received its result, for
        # example when its environment could not reconnect; the wake stays open.
        raise SessionWakeNotContinued(wake_id)
    return SessionWakeOutcome.CONTINUED


def _wake_task_identity(task: Task) -> tuple[str, str, datetime, int, datetime | None]:
    payload = task.input
    session_id = payload.get("session_id")
    wake_id = payload.get("wake_id")
    deadline = _aware(payload.get("deadline"))
    attempt = payload.get("attempt")
    raw_retry_until = payload.get("retry_until")
    retry_until = _aware(raw_retry_until)
    if (
        set(payload) != {"session_id", "wake_id", "deadline", "attempt", "retry_until"}
        or (raw_retry_until is None) != (attempt == 1)
        or (raw_retry_until is not None and retry_until is None)
        or type(session_id) is not str
        or type(wake_id) is not str
        or not session_id
        or not wake_id
        or deadline is None
        or type(attempt) is not int
        or not 1 <= attempt <= _MAX_ATTEMPT_SCAN
        or task.id != _attempt_task_id(wake_id, attempt)
    ):
        raise ValueError("Session wake task is malformed.")
    return session_id, wake_id, deadline, attempt, retry_until


def _aware(value: object) -> datetime | None:
    if type(value) is not str:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


async def _deliver_when_due(
    app: CayuApp, *, session_id: str, wake_id: str, worker_id: str
) -> SessionWakeOutcome:
    owner = await _owning_task(app, session_id)
    if owner is not None:
        return await _deliver_to_owning_task(
            app, owner, session_id=session_id, wake_id=wake_id, worker_id=worker_id
        )
    try:
        return await _deliver(app, session_id=session_id, wake_id=wake_id)
    except SessionWakeNotDue as early:
        await _wait_out_clock_skew(app, early)
        return await _deliver(app, session_id=session_id, wake_id=wake_id)


async def _wait_out_clock_skew(app: CayuApp, early: SessionWakeNotDue) -> None:
    remaining = early.deadline - app._clock()
    if remaining > MAX_WAKE_CLOCK_SKEW:
        raise early
    await asyncio.sleep(max(remaining.total_seconds(), 0.0))


async def _owning_task(app: CayuApp, session_id: str) -> Task | None:
    """Return the running task that owns a session, or ``None`` for an unowned one."""

    task_store = app.task_store
    if task_store is None:
        return None
    owners = await task_store.list_tasks(
        TaskQuery(session_id=session_id, status=TaskStatus.RUNNING, limit=2)
    )
    if len(owners) > 1:
        raise RuntimeError("The sleeping session has ambiguous task ownership.")
    return owners[0] if owners else None


async def _deliver_to_owning_task(
    app: CayuApp,
    owner: Task,
    *,
    session_id: str,
    wake_id: str,
    worker_id: str | None,
) -> SessionWakeOutcome:
    """Continue a task-owned session through the task's handoff parked on this wake."""

    from cayu.tasks.worker import TaskHandlerOutcome, _run_recovered_interrupted_task_handler

    task_store = app.task_store
    assert task_store is not None
    session = await app.session_store.load(session_id)
    if session is not None and await _delivered(app, session_id, wake_id):
        return SessionWakeOutcome.ALREADY_DELIVERED
    pending = None if session is None else (await _open_wake(app, session))[0]
    asleep = pending is not None and pending.input_id == wake_id
    if asleep:
        assert pending is not None and pending.wake is not None
        if app._clock() < pending.wake.deadline:
            await _wait_out_clock_skew(app, SessionWakeNotDue(pending.wake.deadline))
    handed_off = await _handed_off_owner(app, owner, wake_id=wake_id, asleep=asleep)
    if handed_off is None:
        # A competing delivery may have held the task and continued the call meanwhile.
        if await _delivered(app, session_id, wake_id):
            return SessionWakeOutcome.ALREADY_DELIVERED
        return SessionWakeOutcome.CLOSED
    owner, parked = handed_off
    assert owner.session_id is not None and owner.session_instance_id is not None
    assert owner.interrupted_handoff_id is not None
    expected = TaskContinuationExpectation(
        task_id=owner.id,
        session_id=owner.session_id,
        session_instance_id=owner.session_instance_id,
        prior_handoff_id=owner.interrupted_handoff_id,
        operation_sha256=hashlib.sha256(
            f"cayu.session-wake-continuation.v1:{wake_id}".encode()
        ).hexdigest(),
        parked_wait=parked,
    )
    outcomes: list[SessionWakeOutcome] = []
    failures: list[Exception] = []

    async def continue_owned(owned_app: CayuApp, claimed: Task, claimed_by: str):
        try:
            outcomes.append(
                await _deliver(
                    owned_app,
                    session_id=session_id,
                    wake_id=wake_id,
                    task_worker_id=claimed_by,
                    task_handoff_id=claimed.interrupted_handoff_id,
                )
            )
        except Exception as error:
            failures.append(error)
        current = await owned_app.session_store.load(session_id)
        if current is not None and current.status is SessionStatus.INTERRUPTED:
            # Asleep again, still handed over, or stopped: the worker hands the
            # task off, parked when the session still sleeps.
            return TaskHandlerOutcome.SESSION_INTERRUPTED
        if failures:
            raise failures[0]
        return None

    page = await _run_recovered_interrupted_task_handler(
        app,
        task_store,
        continue_owned,
        worker_id=f"{worker_id or 'session-wake'}:continue:{wake_id}",
        expected=expected,
        apply_unproductive_backoff=False,
        lease_seconds=_CONTINUATION_LEASE_SECONDS,
        query=None,
        after=None,
    )
    if page.task is None:
        # Another delivery of this wake holds or finished the continuation.
        if await _delivered(app, session_id, wake_id):
            return SessionWakeOutcome.ALREADY_DELIVERED
        raise SessionWakeOwnerBusy(owner.id)
    if failures:
        raise failures[0]
    if not outcomes:
        raise SessionWakeNotContinued(wake_id)
    return outcomes[0]


async def _handed_off_owner(
    app: CayuApp, owner: Task, *, wake_id: str, asleep: bool
) -> tuple[Task, TaskParkedWait | None] | None:
    """Wait briefly for the owner's handoff and return it with the wait to present.

    This wake continues a handoff parked on it, and repairs an unparked one while
    the session still sleeps on it. A handoff parked on another wait, or an
    unparked one after the sleep closed, belongs to someone else: ``None``.
    """

    task_store = app.task_store
    assert task_store is not None
    parked = TaskParkedWait(kind="session_wake", wait_id=wake_id)
    waited = timedelta(0)
    current: Task | None = owner
    while current is not None and current.status is TaskStatus.RUNNING:
        if current.worker_id is None and current.interrupted_handoff_id is not None:
            receipt = await task_store.load_interrupted_task_handoff_receipt(
                current.id, current.interrupted_handoff_id
            )
            if receipt is None:
                return None
            presented = receipt.request.parked
            if presented == parked or (presented is None and asleep):
                return current, presented
            return None
        if not asleep:
            return None
        if waited >= _OWNER_HANDOFF_WAIT:
            raise SessionWakeOwnerBusy(current.id)
        await asyncio.sleep(_OWNER_HANDOFF_POLL.total_seconds())
        waited += _OWNER_HANDOFF_POLL
        current = await task_store.load_task(owner.id)
    return None


async def _stop_sleep(app: CayuApp, session_id: str) -> None:
    async for _ in app.interrupt_session(
        InterruptSessionRequest(
            session_id=session_id,
            reason="The task that owned this sleeping session ended before it woke.",
            requested_by=ResolutionActor(
                subject=SESSION_WAKE_ACTOR_SUBJECT, source=ResolutionActorSource.SYSTEM
            ),
        )
    ):
        pass


async def _handle_session_wake_task(app: CayuApp, task: Task, worker_id: str) -> None:
    """Run one wake attempt; ``run_task_worker`` routes every wake task here."""

    from cayu.tasks.worker import complete_managed_task

    task_store = app.task_store
    if task_store is None:
        raise RuntimeError("Session wakes require a task store.")
    session_id, wake_id, *_ = _wake_task_identity(task)
    try:
        outcome = await _deliver_when_due(
            app, session_id=session_id, wake_id=wake_id, worker_id=worker_id
        )
    except Exception:
        # The failed attempt is the inspectable evidence; its successor keeps the
        # still-open wake from sleeping forever within the retry window.
        successor = _successor(task)
        if successor is not None:
            await app.create_task(successor)
        raise
    await complete_managed_task(task_store, task, worker_id, {"outcome": outcome.value})


async def session_wake_attempts(app: CayuApp, wake_id: str) -> tuple[Task, ...]:
    """Return one wake's attempt tasks in order, including failed ones and their errors."""

    task_store = app.task_store
    if task_store is None:
        return ()
    attempts: list[Task] = []
    for attempt in range(1, _MAX_ATTEMPT_SCAN + 1):
        task = await task_store.load_task(_attempt_task_id(wake_id, attempt))
        if task is None:
            break
        attempts.append(task)
    return tuple(attempts)


async def _ensure_wake_attempt(app: CayuApp, pending: PendingUserInput) -> bool | None:
    """Create the next attempt when no live one exists.

    Returns whether one was created, or ``None`` when the retry window is spent.
    """

    timer = pending.wake
    assert timer is not None
    attempts = await session_wake_attempts(app, pending.input_id)
    if not attempts:
        request: TaskCreate | None = _attempt_task(
            pending.session_id, pending.input_id, timer.deadline, 1, timer.deadline
        )
    elif _attempt_is_live(attempts[-1], now=app._clock()):
        return False
    else:
        request = _successor(attempts[-1])
    if request is None:
        return None
    await app.create_task(request)
    return True


def _attempt_is_live(task: Task, *, now: datetime) -> bool:
    """An attempt that ended, lost its worker's lease or was fenced is not live."""

    from cayu.tasks import is_fenced_after_worker_loss

    if task.status in _TERMINAL_TASK_STATUSES or is_fenced_after_worker_loss(task):
        return False
    return not (
        task.status in {TaskStatus.CLAIMED, TaskStatus.RUNNING}
        and task.lease_expires_at is not None
        and task.lease_expires_at <= now
    )


async def reconcile_session_wakes(app: CayuApp, *, limit: int = 200) -> int:
    """Give every open wake a live attempt; return how many attempts were created.

    It covers a crash before the first attempt existed and an attempt that ended
    (failed, cancelled, or completed) while the session still sleeps. It reads
    bounded pages of sleeping sessions and is safe to repeat; a session it cannot
    reconcile is logged and skipped. ``run_task_worker`` calls it when it starts
    and on its expired-lease reclaim cadence.
    """

    if app.task_store is None:
        return 0
    created = 0
    spent: list[str] = []
    cursor: str | None = None
    while True:
        page = await app.session_store.query_pending_actions(
            PendingActionQuery(
                kind=PendingActionKind.WAKE,
                statuses=frozenset({SessionStatus.INTERRUPTED}),
                cursor=cursor,
                limit=min(limit, 200),
            )
        )
        for action in page.actions:
            try:
                session = await app.session_store.load(action.session.id)
                if session is None:
                    continue
                pending, _ = await _open_wake(app, session)
                if pending is None:
                    continue
                ensured = await _ensure_wake_attempt(app, pending)
                if ensured is None:
                    spent.append(pending.input_id)
                elif ensured:
                    created += 1
            except Exception as exc:
                # One damaged sleeping session must not stop the others or a worker.
                _logger.warning(
                    "Could not reconcile the wake of session %s (%s).",
                    action.session.id,
                    type(exc).__name__,
                )
        if not page.has_more or page.next_cursor is None:
            break
        cursor = page.next_cursor
    if spent:
        # One line per pass, naming the wakes to continue with deliver_session_wake.
        _logger.warning(
            "%d session wake(s) spent their retry window; continue them with "
            "deliver_session_wake: %s",
            len(spent),
            ", ".join(spent[:20]) + (" …" if len(spent) > 20 else ""),
        )
    return created


def _require_servable_pause(app: CayuApp, tool: Tool) -> None:
    """Refuse a pausing tool this application cannot serve, at registration."""

    if not getattr(tool, "pauses_session", False):
        return
    kind = getattr(tool, "pause_kind", "user_input")
    if kind == "user_input":
        return
    if kind != "wake" or not callable(getattr(tool, "plan_wake", None)):
        raise ValueError(
            f"Tool {tool.spec.name!r} declares an unsupported pause: a pausing tool "
            "is user input, or a wake with plan_wake(arguments, now=...)."
        )
    task_store = app.task_store
    if task_store is None or task_store.supports_task_scheduling is not True:
        raise ValueError(
            f"Tool {tool.spec.name!r} schedules a wake task; give the application a "
            "task store that supports task scheduling."
        )
    declared = type.__getattribute__(type(task_store), "__dict__")
    if not all(
        declared.get(capability) is True
        for capability in (
            "supports_bound_interrupted_task_handoffs",
            "supports_parked_interrupted_task_handoffs",
        )
    ):
        raise ValueError(
            f"Tool {tool.spec.name!r} parks the task that owns a sleeping session; give "
            "the application a task store that declares bound and parked "
            "interrupted-task handoffs (supports_bound_interrupted_task_handoffs and "
            "supports_parked_interrupted_task_handoffs), as the built-in stores do."
        )
    app._uses_session_wakes = True


async def _require_deletable(app: CayuApp, session_id: str) -> None:
    """Refuse to delete a sleeping session while the task that owns it still runs.

    Deletion refuses any session whose task is still active; this only names the
    way out. A parked task has no worker, so cancelling it settles it at once.
    """

    if await _open_wake_id(app, session_id) is None:
        return
    owner = await _owning_task(app, session_id)
    if owner is not None:
        raise ValueError(
            f"Session {session_id} is asleep in sleep_until inside task {owner.id}, which "
            "is still running. Cancel the task first (a parked task is cancelled at "
            "once), then delete the session."
        )


async def _cancel_session_wake(app: CayuApp, session_id: str) -> None:
    """Cancel the pending attempts of a sleep an admitted interruption stops.

    It covers a wake asleep, still opening, or handed over after a failed
    attempt. A wake attempt that runs anyway finds the pause closed. A task that
    owns the session stays parked on the closed wake until
    :func:`reconcile_parked_handoffs` releases it.
    """

    wake_id = await _open_wake_id(app, session_id)
    if wake_id is not None:
        await _cancel_session_wake_attempts(app, wake_id)


async def _cancel_session_wake_attempts(app: CayuApp, wake_id: str) -> None:
    """Cancel one wake's unstarted attempts; the closed pause stays the authority."""

    for task in await session_wake_attempts(app, wake_id):
        if task.status is not TaskStatus.PENDING or task.schedule is None:
            continue
        try:
            await app.cancel_scheduled_task(
                TaskScheduleCancelRequest(
                    task_id=task.id,
                    operation_id=f"cancel-{task.id}",
                    expected_revision=task.schedule.revision,
                )
            )
        except Exception:
            # A late attempt finds the pause closed and does nothing, so a failed
            # cancellation only leaves a no-op task behind.
            _logger.warning("Could not cancel wake task %s.", task.id, exc_info=True)


async def reconcile_parked_handoffs(app: CayuApp, *, limit: int = 100) -> int:
    """Release parked tasks whose session no longer waits on that wait; return how many.

    A task stays parked on a wake that an interruption closed, including one that
    closed while its worker was still handing the task off, or that a woken round
    left for manual recovery. Each such handoff is claimed with the wait it names
    and handed off again for what the session waits on now, usually nothing, so the
    generic continuation can take it. Bounded pages; safe to repeat. A task it
    cannot release is logged and skipped. ``run_task_worker`` calls it with
    :func:`reconcile_session_wakes`.
    """

    from cayu.tasks.worker import _session_parked_wait

    task_store = app.task_store
    if task_store is None:
        return 0
    released = 0
    after: str | None = None
    page_size = min(limit, 100)
    while True:
        page = await task_store.list_parked_interrupted_task_handoffs(after=after, limit=page_size)
        for receipt in page:
            request = receipt.request
            try:
                if await _session_parked_wait(app, request.session_id) == request.parked:
                    continue
                if await _release_parked_handoff(app, receipt):
                    released += 1
            except Exception as exc:
                _logger.warning(
                    "Could not release the parked task %s (%s).",
                    request.task_id,
                    type(exc).__name__,
                )
        if len(page) < page_size:
            return released
        after = page[-1].request.task_id


async def _release_parked_handoff(app: CayuApp, receipt: TaskInterruptedHandoffReceipt) -> bool:
    """Claim one stale parked handoff with its own wait and hand it off again."""

    from cayu.tasks.worker import TaskHandlerOutcome, _run_recovered_interrupted_task_handler

    task_store = app.task_store
    assert task_store is not None
    request = receipt.request
    expected = TaskContinuationExpectation(
        task_id=request.task_id,
        session_id=request.session_id,
        session_instance_id=request.session_instance_id,
        prior_handoff_id=request.handoff_id,
        operation_sha256=hashlib.sha256(
            f"cayu.parked-handoff-release.v1:{request.handoff_id}".encode()
        ).hexdigest(),
        parked_wait=request.parked,
    )

    async def hand_off_again(owned_app: CayuApp, claimed: Task, claimed_by: str):
        # The claim already proved the session interrupted; the worker hands the
        # task off for whatever the session waits on now.
        return TaskHandlerOutcome.SESSION_INTERRUPTED

    page = await _run_recovered_interrupted_task_handler(
        app,
        task_store,
        hand_off_again,
        worker_id=f"parked-handoff-release:{request.handoff_id[:16]}",
        expected=expected,
        apply_unproductive_backoff=False,
        lease_seconds=_CONTINUATION_LEASE_SECONDS,
        query=None,
        after=None,
    )
    return page.task is not None


async def _open_wake_id(app: CayuApp, session_id: str) -> str | None:
    """Name the wake a session's checkpoint holds, asleep or handed over."""

    checkpoint = await app.session_store.load_checkpoint(session_id)
    if checkpoint is None:
        return None
    wake = checkpoint.get("pending_wake")
    pending = (
        wake.get("continuation") if type(wake) is dict else checkpoint.get("pending_user_input")
    )
    if type(pending) is not dict or type(pending.get("wake")) is not dict:
        return None
    wake_id = pending.get("input_id")
    return wake_id if type(wake_id) is str else None
