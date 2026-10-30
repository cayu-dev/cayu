"""Run the tasks that are due, then return with the time more work becomes due.

A host that doesn't keep a worker process up all the time (cron, a function, a
platform that wakes an application on a timer) starts a worker with
:func:`run_task_worker_until_idle`, lets it handle what is claimable now, and
uses the returned :class:`TaskWorkerIdleResult` to decide when to start it again.
``cayu worker NAME --until-idle`` does the same for a project's configured
worker entrypoint and prints the result as one JSON line.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt

from cayu.tasks.scheduling import TaskScheduleWakeup

if TYPE_CHECKING:
    from collections.abc import Iterator

    from cayu.applications import CayuApp
    from cayu.tasks.abandoned import FencedTaskMonitor
    from cayu.tasks.store import TaskStore
    from cayu.tasks.worker import TaskHandler, _InterruptedHandoffRecoveryPage

_UNTIL_IDLE_REQUESTED: ContextVar[bool] = ContextVar("cayu_worker_until_idle", default=False)


class TaskWorkerIdleResult(BaseModel):
    """What a worker run handled and when its tasks next need a worker.

    ``idle`` is true when the worker returned because nothing it could claim
    was left and no handler was running. It is false when ``max_tasks``, the
    ``stop`` event or the application closing ended the run first; work may
    then still be due, so start the worker again without waiting.

    The times are the store's observation, for the worker's ``query``, made
    just before the claim that found nothing (see
    :class:`cayu.tasks.TaskScheduleWakeup`). ``next_available_at`` is the
    earliest time a pending scheduled task becomes claimable.
    ``next_expiry_at`` is the earliest time a scheduled task stops being
    admissible, so a sweep can record that it expired, or, with
    ``reconcile_abandoned_after_s``, a task fenced after worker loss becomes
    due to be settled. Claims held by other workers, live or dead, are not
    reported: after a run that crashed, wait for its claims to expire
    (``lease_seconds``) before starting the worker again. The times are empty
    when nothing is scheduled or the run was not idle.

    Start the worker again at once when ``maintenance_required`` is true
    (expired or skipped tasks were still waiting to be recorded after the
    worker claimed again, recovering a session failed on a task for the first
    time, or looking for fenced tasks failed) or ``continuations_pending`` is
    true (a handled task may have handed off a session that this run, which
    scans for continuations once, didn't continue). If a run reports the same
    signals as the run before it, back off before starting it again.

    ``attention_required`` is true when a task in the worker's scope can't
    progress until someone acts: a task fenced after worker loss that this
    worker won't settle, a session whose recovery needs ``cayu recovery``
    (for example ``ModelCompletionManualRecoveryRequired``), or one whose
    recovery failed again when the run retried it at once. The worker logs a
    warning naming each one. It is also true when the run found more than
    100,000 fenced tasks, more than it reads in one run. Starting the worker
    again doesn't help; start it once that is done.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    handled: StrictInt = Field(ge=0)
    idle: StrictBool
    next_available_at: datetime | None = None
    next_expiry_at: datetime | None = None
    maintenance_required: StrictBool = False
    continuations_pending: StrictBool = False
    attention_required: StrictBool = False


@dataclass
class _TaskWorkerIdleExit:
    """State the slots of one until-idle worker share to decide when to return."""

    # Set by the maintenance slot once expired claims were reclaimed and a
    # whole pass of expired interrupted-handoff recovery found nothing more.
    maintenance_settled: bool = False
    # Set once the maintenance slot finished its one continuation scan.
    continuations_scanned: bool = False
    continuations_pending: bool = False
    # Slots inside a claim-and-handle step; a running handler may create work.
    steps_running: int = 0
    # Steps that handled work; one that ends during a claim makes its
    # observation stale.
    handled_steps: int = 0
    claimed_again: bool = False
    observation: TaskScheduleWakeup | None = None
    idle: bool = False
    # Tasks the latest whole pass of interrupted-handoff recovery failed on, and
    # those of them the pass before failed on too.
    recovery_failed: frozenset[str] = frozenset()
    recovery_repeated: frozenset[str] = frozenset()
    manual_recovery_required: bool = False
    # A pass that failed is retried once at once, before the run returns.
    recovery_retried: bool = False
    recovery_retry_pending: bool = False
    fenced: FencedTaskMonitor | None = None
    fenced_scan_failed: bool = False
    _pass_failed: set[str] = field(default_factory=set)
    _pass_manual: bool = False

    def may_return(self) -> bool:
        return self.maintenance_settled and self.continuations_scanned and self.steps_running == 1

    def record_recovery_page(
        self, page: _InterruptedHandoffRecoveryPage, *, first_page: bool
    ) -> None:
        if first_page:
            self._pass_failed = set()
            self._pass_manual = False
            self.recovery_retry_pending = False
        self._pass_failed |= page.failed
        self._pass_manual |= page.manual_recovery_required
        if not page.exhausted:
            return
        failed = frozenset(self._pass_failed)
        self.recovery_repeated = failed & self.recovery_failed
        self.recovery_failed = failed
        self.manual_recovery_required = self._pass_manual
        if failed and not self.recovery_retried:
            self.recovery_retried = True
            self.recovery_retry_pending = True

    def claim_again(self, observation: TaskScheduleWakeup | None, observed_after: int) -> bool:
        """Record an empty claim; return True to observe and claim once more first.

        Another slot's handler may have created work after ``observation``
        (taken when ``observed_after`` steps had handled work) and finished
        before the claim returned. The claim itself records expired and
        skipped tasks, so maintenance observed before it is usually settled by
        it. Claiming one more time confirms that without letting a sweep that
        never settles keep the worker running.
        """
        if self.handled_steps != observed_after:
            return True
        if observation is not None and observation.maintenance_required and not self.claimed_again:
            self.claimed_again = True
            return True
        self.observation = observation
        self.idle = True
        return False

    def result(self, handled: int) -> TaskWorkerIdleResult:
        observation = self.observation
        if not self.idle or observation is None:
            return TaskWorkerIdleResult(handled=handled, idle=self.idle)
        fenced = self.fenced
        expiries = [observation.next_expiry_at, None if fenced is None else fenced.next_settle_at]
        return TaskWorkerIdleResult(
            handled=handled,
            idle=True,
            next_available_at=observation.next_available_at,
            next_expiry_at=min((time for time in expiries if time is not None), default=None),
            maintenance_required=observation.maintenance_required
            or bool(self.recovery_failed - self.recovery_repeated)
            or self.fenced_scan_failed,
            continuations_pending=self.continuations_pending,
            # A failure the run's immediate retry hit again won't clear by itself.
            attention_required=self.manual_recovery_required
            or bool(self.recovery_repeated)
            or (fenced is not None and (fenced.attention_required or fenced.pass_unfinished)),
        )


# Hands the state to the run_task_worker call this module starts, and only it.
_UNTIL_IDLE_STATE: ContextVar[_TaskWorkerIdleExit | None] = ContextVar(
    "cayu_task_worker_until_idle_state", default=None
)


def _take_until_idle_state() -> _TaskWorkerIdleExit | None:
    state = _UNTIL_IDLE_STATE.get()
    if state is not None:
        # Workers that its handlers start run until stopped, as usual.
        _UNTIL_IDLE_STATE.set(None)
    return state


async def run_task_worker_until_idle(
    app: CayuApp,
    task_store: TaskStore,
    handler: TaskHandler,
    *,
    worker_id: str,
    **options: Any,
) -> TaskWorkerIdleResult:
    """Handle the tasks that are claimable now, then return.

    Takes the same arguments as :func:`cayu.tasks.run_task_worker`. The worker
    first does its startup maintenance: it reclaims expired claims and, when it
    recovers interrupted handoffs, finishes a whole recovery pass; with a
    ``recovered_interrupted_task_handler`` it also finishes one continuation
    scan, again after recovery hands off more sessions. It then claims and
    handles tasks as usual, parks contracted tasks, and returns once a claim
    finds nothing while no handler is running. With ``concurrency`` above one,
    a slot that finds nothing waits until the other slots' handlers finish,
    because they may create work that is due at once. ``max_tasks`` still caps
    the run; whichever comes first ends it.

    The store must report scheduled task times
    (``TaskStore.supports_task_scheduling``); all built-in stores do.
    """

    from cayu.tasks.worker import run_task_worker

    if not task_store.supports_task_scheduling:
        raise NotImplementedError(
            "run_task_worker_until_idle needs a task store that reports scheduled task "
            "times (supports_task_scheduling), so it can say when to start again."
        )
    state = _TaskWorkerIdleExit()
    context = copy_context()
    context.run(_UNTIL_IDLE_STATE.set, state)
    handled = await asyncio.create_task(
        run_task_worker(app, task_store, handler, worker_id=worker_id, **options),
        context=context,
    )
    return state.result(handled)


def worker_until_idle_requested() -> bool:
    """Whether ``cayu worker --until-idle`` started the running entrypoint.

    An entrypoint that supports the flag checks this and returns the result of
    :func:`run_task_worker_until_idle`, which the command prints.
    """

    return _UNTIL_IDLE_REQUESTED.get()


@contextmanager
def _worker_until_idle_requested() -> Iterator[None]:
    token = _UNTIL_IDLE_REQUESTED.set(True)
    try:
        yield
    finally:
        _UNTIL_IDLE_REQUESTED.reset(token)
