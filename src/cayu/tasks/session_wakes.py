"""Durable wake tasks that continue sessions asleep in ``sleep_until``.

``SleepUntilTool`` parks a session at its tool call under its own
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

A subagent sleeps the same way. Its foreground parent's call waits on it; the task
that owns the parent, if any, parks on the subagent's wake, whose attempt continues
the subagent and then each parent up the chain under its claim on that task.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, NoReturn

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
from cayu.events import EventType
from cayu.sessions._checkpoint_mutations import SessionRuntimePublicationConflict
from cayu.sessions._foreground_child_checkpoint import (
    MAX_FOREGROUND_CHAIN_DEPTH,
    ForegroundChildWait,
    foreground_child_state_from_checkpoint,
)
from cayu.sessions._pending_tool_round_reader import pending_tool_round_from_checkpoint
from cayu.sessions.event_queries import EventOrder, EventQuery
from cayu.sessions.pending_action_contracts import PendingActionQuery
from cayu.sessions.pending_actions import woken_round_awaits_manual_recovery
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
    from cayu.tasks.store import TaskStore
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

# The task claims a wake holds while it continues a sleeping subagent, by the session
# of each claimed task: the parents it continues settle under that claim.
_HELD_TASK_CLAIMS: ContextVar[Mapping[str, tuple[str, str, str | None]] | None] = ContextVar(
    "cayu_session_wake_task_claims", default=None
)


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

    return await woken_round_awaits_manual_recovery(app.session_store, session_id, wake_id)


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

    owner = await _wake_owner(app, session_id)
    if owner is not None:
        return await _deliver_to_owning_task(
            app, owner, session_id=session_id, wake_id=wake_id, worker_id=None
        )
    if await _stop_ended_chain(app, session_id):
        return SessionWakeOutcome.CLOSED
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
    owner = await _wake_owner(app, session_id)
    if owner is not None:
        return await _deliver_to_owning_task(
            app, owner, session_id=session_id, wake_id=wake_id, worker_id=worker_id
        )
    if await _stop_ended_chain(app, session_id):
        return SessionWakeOutcome.CLOSED
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
    """Continue a task-owned session through the task's handoff parked on this wake.

    When the sleeping session is a subagent the task's session waits on, the attempt
    continues the subagent and then, under the same claim, each parent up the chain.
    It does so again after a crash between the two, while the task stays parked.
    """

    from cayu.tasks.worker import TaskHandlerOutcome, _run_recovered_interrupted_task_handler

    task_store = app.task_store
    assert task_store is not None
    subagent = owner.session_id != session_id
    session = await app.session_store.load(session_id)
    delivered = session is not None and await _delivered(app, session_id, wake_id)
    if delivered and not subagent:
        return SessionWakeOutcome.ALREADY_DELIVERED
    pending = None if session is None else (await _open_wake(app, session))[0]
    asleep = pending is not None and pending.input_id == wake_id
    if asleep:
        assert pending is not None and pending.wake is not None
        if app._clock() < pending.wake.deadline:
            await _wait_out_clock_skew(app, SessionWakeNotDue(pending.wake.deadline))
    parked = TaskParkedWait(
        kind="session_wake", wait_id=wake_id, session_id=session_id if subagent else None
    )
    handed_off = await _handed_off_owner(app, owner, parked=parked, asleep=asleep)
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
            if subagent:
                with _holding_task_claim(claimed):
                    outcome = await _deliver(owned_app, session_id=session_id, wake_id=wake_id)
                    if (
                        await _continue_waiting_parents(
                            owned_app, session_id, root=claimed.session_id or session_id
                        )
                        and outcome is SessionWakeOutcome.ALREADY_DELIVERED
                    ):
                        # The subagent had woken; this attempt continued its parents.
                        outcome = SessionWakeOutcome.CONTINUED
                    outcomes.append(outcome)
            else:
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
        current = await owned_app.session_store.load(claimed.session_id or session_id)
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
        worker_id=f"{worker_id or 'session-wake'}{_wake_continuation_suffix(wake_id)}",
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


async def _continue_waiting_parents(app: CayuApp, session_id: str, *, root: str) -> bool:
    """Bring each parent up the chain up to date with the subagent its call waits on.

    This attempt holds the task's claim, so it does so itself: a delivery holding no
    claim left the parent to the wake. An ended subagent's parent continues; a
    subagent asleep again refreshes its parent's reference before the task is handed
    back. Returns whether it continued any parent.
    """

    from cayu.runtime._foreground_child_continuation import continue_foreground_parent

    continued = False
    for _ in range(MAX_FOREGROUND_CHAIN_DEPTH):
        session = await app.session_store.load(session_id)
        if session is None or session_id == root or session.parent_session_id is None:
            break
        terminal = (await app.session_store.summarize_outcome(session_id)).terminal_event
        if (
            terminal is not None
            and (
                await _ended(app, session)
                or await _outstanding_wake_id(app, session_id) is not None
            )
            and await _waiting_parent(app, session) is not None
        ):
            # An ended subagent continues its parent; one asleep again refreshes its
            # parent's reference, so the task parks on its new wake.
            await continue_foreground_parent(app, terminal.event, run=_run_directly)
            # A delivery can be put off while the subagent still settles.
            continued = continued or await _waiting_parent(app, session) is None
        session_id = session.parent_session_id
    return continued


async def _run_directly(operation) -> None:
    """Run a continuation step under the task claim this attempt already holds."""

    async def no_further_guard() -> None:
        return None

    await operation(no_further_guard)


def _wake_continuation_suffix(wake_id: str) -> str:
    """How the worker id of a wake attempt's claim on its owning task ends."""

    return f":continue:{wake_id}"


async def _handed_off_owner(
    app: CayuApp, owner: Task, *, parked: TaskParkedWait, asleep: bool
) -> tuple[Task, TaskParkedWait | None] | None:
    """Wait briefly for the owner's handoff and return it with the wait to present.

    This wake continues a handoff parked on it, and repairs an unparked one while
    the session still sleeps on it. A handoff parked on another wait, or an
    unparked one after the sleep closed, belongs to someone else: ``None``.
    """

    task_store = app.task_store
    assert task_store is not None
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


async def wake_retries_spent(task_store: TaskStore, wake_id: str, *, now: datetime) -> bool:
    """Whether a wake has no live attempt and its retry window is spent.

    Nothing continues it on its own any more: ``deliver_session_wake`` does.
    """

    last: Task | None = None
    for attempt in range(1, _MAX_ATTEMPT_SCAN + 1):
        task = await task_store.load_task(_attempt_task_id(wake_id, attempt))
        if task is None:
            break
        last = task
    return last is not None and not _attempt_is_live(last, now=now) and _successor(last) is None


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
    reconcile is logged and skipped. It also stops a chain of subagent calls whose
    task ended after its subagent woke, when no attempt is left to stop it.
    ``run_task_worker`` calls it when it starts and on its expired-lease reclaim
    cadence.
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
    await _stop_chains_whose_task_ended(app, limit=limit)
    if spent:
        # One line per pass, naming the wakes to continue with deliver_session_wake.
        _logger.warning(
            "%d session wake(s) spent their retry window; continue them with "
            "deliver_session_wake: %s",
            len(spent),
            ", ".join(spent[:20]) + (" …" if len(spent) > 20 else ""),
        )
    return created


async def _stop_chains_whose_task_ended(app: CayuApp, *, limit: int) -> None:
    """Stop the calls of a chain whose task ended after its sleeping subagent woke.

    The wake's attempt stops such a chain when it runs. One that died after the
    subagent woke leaves no attempt behind, since the wake is no longer open, so the
    task's own call would wait on its ended subagents forever.
    """

    cursor: str | None = None
    while True:
        page = await app.session_store.query_pending_actions(
            PendingActionQuery(
                kind=PendingActionKind.DELEGATED_ACTION,
                statuses=frozenset({SessionStatus.INTERRUPTED}),
                cursor=cursor,
                limit=min(limit, 200),
            )
        )
        for action in page.actions:
            try:
                await _stop_chain_if_its_task_ended(app, action.session.id)
            except Exception as exc:
                _logger.warning(
                    "Could not stop the chain of session %s (%s).",
                    action.session.id,
                    type(exc).__name__,
                )
        if not page.has_more or page.next_cursor is None:
            return
        cursor = page.next_cursor


async def _stop_chain_if_its_task_ended(app: CayuApp, root_id: str) -> None:
    root = await app.session_store.load(root_id)
    if root is None or await _waiting_parent(app, root) is not None:
        return
    sleeping = await _sleeping_wait(app, root_id)
    if sleeping is None or sleeping[0] == root_id:
        return
    leaf, wake_id = sleeping
    attempts = await session_wake_attempts(app, wake_id)
    if await _outstanding_wake_id(app, leaf) is not None or (
        attempts and _attempt_is_live(attempts[-1], now=app._clock())
    ):
        # Its due attempt stops it.
        return
    # Stops only a chain whose task ended.
    await _stop_ended_chain(app, leaf)


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

    That task owns the session itself, or the chain of subagent calls waiting on it.
    Deletion refuses any session whose task is still active; this only names the way
    out. A parked task has no worker, so cancelling it settles it at once.
    """

    if await _outstanding_wake_id(app, session_id) is None:
        return
    owner = await _wake_owner(app, session_id)
    if owner is not None:
        held_by = "" if owner.session_id == session_id else f" through session {owner.session_id}"
        raise ValueError(
            f"Session {session_id} is asleep in sleep_until inside task {owner.id}{held_by}, "
            "which is still running. Cancel the task first (a parked task is cancelled at "
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
    generic continuation can take it. A task parked on a subagent's wake whose
    subagent woke and ended gets a wake attempt again if none is live, which
    continues its parents. Bounded pages; safe to repeat. A task it cannot release
    is logged and skipped. ``run_task_worker`` calls it with
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
                    if request.parked is not None and request.parked.session_id is not None:
                        await _ensure_parent_attempt(app, request.parked)
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


async def _ensure_parent_attempt(app: CayuApp, parked: TaskParkedWait) -> None:
    """Give a task parked on a subagent's wake an attempt once that subagent woke and ended.

    The attempt that continued the subagent can die before a call waiting above it
    continued. Once its claim expired and the task was parked again, and no attempt is
    live, a later attempt continues those calls under the task's claim.
    """

    assert parked.session_id is not None
    # The handoff is still parked on this wake, so a call above the subagent waits.
    subagent = await app.session_store.load(parked.session_id)
    if (
        subagent is None
        or not await _ended(app, subagent)
        or not await _delivered(app, parked.session_id, parked.wait_id)
    ):
        return
    attempts = await session_wake_attempts(app, parked.wait_id)
    if not attempts or _attempt_is_live(attempts[-1], now=app._clock()):
        return
    successor = _successor(attempts[-1])
    if successor is None:
        _logger.warning(
            "The wake %s that continues a parked task spent its retry window; continue it "
            "with deliver_session_wake.",
            parked.wait_id,
        )
        return
    await app.create_task(successor)


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


async def _call_closed(app: CayuApp, wait: ForegroundChildWait) -> bool:
    """Whether the parent's call that waits on a subagent was stopped, closing its interaction."""

    closed = await app.session_store.query_events(
        EventQuery(
            session_id=wait.parent_effect.session_id,
            interaction_id=wait.parent_effect.interaction_id,
            event_types=(
                EventType.INTERACTION_COMPLETED,
                EventType.INTERACTION_FAILED,
                EventType.INTERACTION_INTERRUPTED,
            ),
            limit=1,
        )
    )
    return bool(closed)


async def _waiting_parent(
    app: CayuApp, session: Session
) -> tuple[Session, ForegroundChildWait] | None:
    """The parent whose call still waits on this foreground subagent, with that wait."""

    if session.parent_session_id is None:
        return None
    parent = await app.session_store.load(session.parent_session_id)
    if parent is None:
        return None
    wait, _ = foreground_child_state_from_checkpoint(
        await app.session_store.load_checkpoint(parent.id)
    )
    if wait is None or wait.child_session_id != session.id or await _call_closed(app, wait):
        return None
    return parent, wait


def _chain_too_deep() -> NoReturn:
    """Fail a wake closed: past the depth no sleep is admitted at, ownership is unknown."""

    raise RuntimeError(
        f"A chain of subagent calls deeper than {MAX_FOREGROUND_CHAIN_DEPTH} waits on a "
        "sleeping session; no wake continues it without its task's claim."
    )


async def _chain_above(app: CayuApp, session_id: str) -> list[Session] | None:
    """The sessions above this one whose foreground subagent calls hold it, bottom up.

    A crash between two hops can leave an ended subagent whose parent ended too while
    the call above that one still waits; the chain is followed across it. Every walk
    that decides who owns or stops a chain follows this one. ``None`` means the chain
    is deeper than the runtime follows: no sleep is admitted there, so only a wake's
    own paths treat that as an error.
    """

    above: list[Session] = []
    current = await app.session_store.load(session_id)
    for _ in range(MAX_FOREGROUND_CHAIN_DEPTH):
        if current is None or current.parent_session_id is None:
            break
        waiting = await _waiting_parent(app, current)
        if waiting is not None:
            current = waiting[0]
            above.append(current)
            continue
        parent = await app.session_store.load(current.parent_session_id)
        subagent = current.metadata.get("subagent")
        if (
            parent is None
            or type(subagent) is not dict
            or subagent.get("mode") != "foreground"
            or not await _ended(app, current)
            or not await _ended(app, parent)
        ):
            break
        current = parent
        above.append(current)
    else:
        return None
    return above


async def _chain_root(app: CayuApp, session_id: str) -> str | None:
    """The session at the top of the foreground subagent calls that wait on this one.

    ``None`` when the chain is deeper than the runtime follows.
    """

    above = await _chain_above(app, session_id)
    if above is None:
        return None
    return above[-1].id if above else session_id


async def _wake_owner(app: CayuApp, session_id: str) -> Task | None:
    """The running task that owns a sleeping session, or the chain of subagent calls it ends.

    A subagent belongs to its parent's task while that parent's call waits on it. A
    wake fails closed on a chain deeper than the runtime follows.
    """

    root = await _chain_root(app, session_id)
    if root is None:
        _chain_too_deep()
    return await _owning_task(app, root)


async def _stop_ended_chain(app: CayuApp, session_id: str) -> bool:
    """Stop a subagent's sleep whose chain's task ended; return whether it did.

    As for a session the task owns itself, a wake never continues work without its
    task: every call still open in the chain, the top one included, and the sleep are
    stopped.
    """

    above = await _chain_above(app, session_id)
    if above is None:
        _chain_too_deep()
    if not above or app.task_store is None:
        return False
    root = above[-1]
    pending = pending_tool_round_from_checkpoint(await app.session_store.load_checkpoint(root.id))
    task = (
        None
        if pending is None or pending.task_id is None
        else await app.task_store.load_task(pending.task_id)
    )
    if task is None or task.status not in _TERMINAL_TASK_STATUSES:
        return False
    # Every call still open in the chain stops, from the top down, before the sleep
    # does: a middle subagent has no task of its own to fence it from continuing, and
    # one that already ended has nothing left to stop.
    for session in reversed(above):
        if session.status not in {SessionStatus.COMPLETED, SessionStatus.FAILED}:
            await _stop_sleep(app, session.id)
    if await _open_wake_id(app, session_id) is not None:
        # A subagent that already woke and ended has no sleep left to stop.
        await _stop_sleep(app, session_id)
    return True


@contextmanager
def _holding_task_claim(task: Task) -> Iterator[None]:
    assert task.session_id is not None and task.worker_id is not None
    held = {
        **(_HELD_TASK_CLAIMS.get() or {}),
        task.session_id: (task.id, task.worker_id, task.interrupted_handoff_id),
    }
    token = _HELD_TASK_CLAIMS.set(held)
    try:
        yield
    finally:
        _HELD_TASK_CLAIMS.reset(token)


def _held_task_claim(session_id: str) -> tuple[str, str, str | None] | None:
    """The task, worker and handoff of a claim a wake holds for this session, if any."""

    return (_HELD_TASK_CLAIMS.get() or {}).get(session_id)


async def _leaves_parent_to_its_wake(app: CayuApp, child_session_id: str) -> bool:
    """Whether a subagent's ended run leaves its parent to the wake that owns their task.

    While the task that owns the chain is parked on a sleeping subagent's wake, or
    claimed by that wake's attempt, only an attempt holding that exact claim
    continues the chain; any other delivery leaves the parent to it. A task handed
    off for anything else, such as a person, continues as an ordinary delivery.
    """

    if not app._uses_session_wakes or app.task_store is None:
        return False
    child = await app.session_store.load(child_session_id)
    if child is None or child.parent_session_id is None:
        return False
    root = await _chain_root(app, child.parent_session_id)
    if root is None:
        # Deeper than any sleep is admitted: an ordinary delivery.
        return False
    owner = await _owning_task(app, root)
    if owner is None:
        return False
    if _held_task_claim(root) == (owner.id, owner.worker_id, owner.interrupted_handoff_id):
        return False
    sleeping = await _sleeping_wait(app, root)
    if sleeping is None or sleeping[0] == root:
        return False
    leaf, wake_id = sleeping
    if owner.worker_id is not None:
        # Claimed: the wake's attempt holds it, alive or not, or someone else does.
        return owner.worker_id.endswith(_wake_continuation_suffix(wake_id))
    receipt = (
        None
        if owner.interrupted_handoff_id is None
        else await app.task_store.load_interrupted_task_handoff_receipt(
            owner.id, owner.interrupted_handoff_id
        )
    )
    return receipt is not None and receipt.request.parked == TaskParkedWait(
        kind="session_wake", wait_id=wake_id, session_id=leaf
    )


async def _ended(app: CayuApp, session: Session) -> bool:
    """Whether a subagent's run ended rather than paused on something of its own."""

    if session.status in {SessionStatus.COMPLETED, SessionStatus.FAILED}:
        return True
    if session.status is not SessionStatus.INTERRUPTED:
        return False
    page = await app.session_store.query_pending_actions(
        PendingActionQuery(session_id=session.id, limit=1)
    )
    return not page.actions


async def _ended_chain_wake(app: CayuApp, session: Session) -> tuple[str, str] | None:
    """The woken subagent wake an ended subagent last waited on, down its own chain.

    A parent still waiting on an ended subagent waits for that wake's attempt chain to
    continue it, after a crash between the hops.
    """

    for _ in range(MAX_FOREGROUND_CHAIN_DEPTH):
        records = await app.session_store.query_events(
            EventQuery(
                session_id=session.id,
                event_types=(
                    EventType.SESSION_INTERRUPTED,
                    EventType.SESSION_DELEGATED_ACTION_UPDATED,
                ),
                order_by=EventOrder.SEQUENCE_DESC,
                limit=1,
            )
        )
        payload = records[0].event.payload if records else {}
        child_id, kind, action_id = (
            payload.get("child_session_id"),
            payload.get("action_kind"),
            payload.get("action_id"),
        )
        if (
            payload.get("interruption_type") != "waiting_on_child_action"
            or type(child_id) is not str
            or type(action_id) is not str
        ):
            return None
        if kind == "wake":
            return (child_id, action_id) if await _delivered(app, child_id, action_id) else None
        child = await app.session_store.load(child_id)
        if kind != "delegated_action" or child is None or not await _ended(app, child):
            return None
        session = child
    return None


async def _sleeping_wait(app: CayuApp, session_id: str) -> tuple[str, str] | None:
    """The session and wake a session waits on, or ``None`` when it waits on no timer.

    That is its own open wake, or the wake of the foreground subagent its call waits
    on, followed down a chain of subagents. The parent's own wait names the wake, so it
    keeps waiting on it until its call continues, even after the subagent woke, and
    a parent still waiting on a subagent that already ended keeps waiting on the wake
    that subagent's chain woke from. It waits on no timer once its call was stopped or
    the subagent's sleep was stopped before it woke. A wake whose woken round stopped
    on a tool a person must settle waits on that person.
    """

    for _ in range(MAX_FOREGROUND_CHAIN_DEPTH):
        wake_id = await _outstanding_wake_id(app, session_id)
        if wake_id is None:
            wait, _ = foreground_child_state_from_checkpoint(
                await app.session_store.load_checkpoint(session_id)
            )
            if wait is None or await _call_closed(app, wait):
                return None
            if wait.child_action_kind == "delegated_action":
                child = await app.session_store.load(wait.child_session_id)
                if child is not None and await _ended(app, child):
                    return await _ended_chain_wake(app, child)
                session_id = wait.child_session_id
                continue
            if wait.child_action_kind != "wake":
                return None
            session_id, wake_id = wait.child_session_id, wait.child_action_id
            current = await _outstanding_wake_id(app, session_id)
            if current is not None:
                # The subagent sleeps again under a newer wake than the one its parent
                # recorded, when its parent's reference was not refreshed yet.
                wake_id = current
            elif not await _delivered(app, session_id, wake_id):
                # The subagent's sleep was stopped before it woke.
                return None
            else:
                child = await app.session_store.load(session_id)
                if child is not None and not await _ended(app, child):
                    # It woke and now waits on something else: follow what it waits on.
                    continue
        if await _needs_manual_recovery(app, session_id, wake_id):
            return None
        return session_id, wake_id
    # Deeper than any sleep is admitted, so nothing below waits on a timer.
    return None


async def _subagent_may_sleep(app: CayuApp, session: Session) -> bool:
    """Whether a subagent's sleep could be continued with the authority above it.

    The calls that wait on a foreground subagent hold its sleep. When a task owns
    the top of that chain, only a worker's claim can be handed off parked and
    continued by the wake, and only outside an approval or answer still being
    carried out, whose own continuation keeps the authority it was resolved with.
    """

    top: Session = session
    gated = False
    for _ in range(MAX_FOREGROUND_CHAIN_DEPTH):
        subagent = top.metadata.get("subagent")
        if (
            top.parent_session_id is None
            or type(subagent) is not dict
            or subagent.get("mode") != "foreground"
        ):
            break
        parent = await app.session_store.load(top.parent_session_id)
        if parent is None:
            break
        checkpoint = await app.session_store.load_checkpoint(parent.id) or {}
        gated = gated or any(
            checkpoint.get(key) is not None
            for key in ("pending_tool_approval", PENDING_USER_INPUT_CHECKPOINT_KEY)
        )
        top = parent
    else:
        # Deeper than the runtime follows a chain: nothing could vouch for its sleep.
        return False
    owner = None if top is session else await _owning_task(app, top.id)
    return owner is None or (owner.worker_id is not None and not gated)


async def _outstanding_wake_id(app: CayuApp, session_id: str) -> str | None:
    """A session's open wake that has not delivered its timer result yet.

    A woken round keeps its accepted wake until the round closes, also while a
    subagent it ran waits on something else; that wake is no longer a timer.
    """

    wake_id = await _open_wake_id(app, session_id)
    if wake_id is None or await _delivered(app, session_id, wake_id):
        return None
    return wake_id


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
