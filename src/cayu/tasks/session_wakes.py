"""Durable wake tasks that continue sessions asleep in ``sleep_until``.

``SleepUntilTool`` parks a root session at its tool call under its own
``pending_wake`` pause and creates one durable task, due at the resolved
deadline. ``run_task_worker`` runs those tasks itself: when the task store makes
one due, the parked call receives its timer result and the same session
continues once. A failed attempt schedules a successor with capped backoff for
``SESSION_WAKE_RETRY_WINDOW`` after the deadline, and :func:`reconcile_session_wakes`
recreates any attempt a crash left out. The public names are exported from
``cayu.tasks``.
"""

from __future__ import annotations

import asyncio
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
from cayu.tasks.creation import TaskCreate
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
    """

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


async def _deliver_when_due(app: CayuApp, *, session_id: str, wake_id: str) -> SessionWakeOutcome:
    try:
        return await deliver_session_wake(app, session_id=session_id, wake_id=wake_id)
    except SessionWakeNotDue as early:
        remaining = early.deadline - app._clock()
        if remaining > MAX_WAKE_CLOCK_SKEW:
            raise
        await asyncio.sleep(max(remaining.total_seconds(), 0.0))
        return await deliver_session_wake(app, session_id=session_id, wake_id=wake_id)


async def _handle_session_wake_task(app: CayuApp, task: Task, worker_id: str) -> None:
    """Run one wake attempt; ``run_task_worker`` routes every wake task here."""

    from cayu.tasks.worker import complete_managed_task

    task_store = app.task_store
    if task_store is None:
        raise RuntimeError("Session wakes require a task store.")
    session_id, wake_id, *_ = _wake_task_identity(task)
    try:
        outcome = await _deliver_when_due(app, session_id=session_id, wake_id=wake_id)
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
    app._uses_session_wakes = True


async def _cancel_session_wake(app: CayuApp, session_id: str) -> None:
    """Cancel the pending attempts of a sleep an admitted interruption stops.

    It covers a wake asleep, still opening, or handed over after a failed
    attempt. A wake attempt that runs anyway finds the pause closed.
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
