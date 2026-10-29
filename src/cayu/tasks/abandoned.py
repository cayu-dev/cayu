"""Tasks fenced after their worker was lost: recognize them, report them and settle them.

Cayu never hands a started task to another worker. When the lease of a worker
that dispatched a task expires, expired-claim reclamation fences the task as a
cancellation request: it stays ``claimed`` with ``status_reason``
``"cancellation_requested"`` (``"retry_cancellation_requested"`` for a retry
series). Cancelling ``run_task_worker`` records the same fence, so a process
that exits before its handler returns leaves the same state once its lease
expires. Cayu can't prove the old handler stopped, so the task stays claimed
until the application settles it. A deploy that replaces a worker during a
non-restartable task is the common way to get here.

Each part works on its own:

- :func:`is_fenced_after_worker_loss` tells such a task from a live claim.
- ``TaskQuery(fenced_after_worker_loss=True)`` lists only them, and ``False``
  leaves them out, for example from an "is a run already queued?" check.
- :func:`reconcile_abandoned_tasks` settles the ones held by workers the caller
  knows are gone, through ``reconcile_abandoned_task_cancellation``, or with
  ``settlement="restart"`` requeues the restartable ones through
  ``restart_abandoned_task`` and cancels the rest.
- ``run_task_worker(reconcile_abandoned_after_s=...)`` and ``cayu worker
  --reconcile-abandoned-after-seconds`` use it for a declared single-worker
  deployment (``restart_abandoned=True`` and ``--restart-abandoned`` choose the
  restart), and every task worker logs a warning for each fenced task it finds.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Collection, Iterable
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from math import isfinite
from time import monotonic

from cayu._clock import normalize_utc_datetime
from cayu._validation import require_clean_nonblank
from cayu.approvals.tools import ResolutionActor, ResolutionActorSource
from cayu.tasks.cancellation import (
    _TASK_CANCELLATION_REQUESTED_REASON,
    _TASK_RETRY_CANCELLATION_REQUESTED_REASON,
)
from cayu.tasks.queries import TaskOrder, TaskQuery, copy_task_query
from cayu.tasks.records import TASK_MAX_RESTARTS_LIMIT, Task, TaskClaimLost, TaskStatus
from cayu.tasks.restart import (
    DEFAULT_MAX_ABANDONED_RESTARTS,
    TaskRestartLimitReached,
    abandoned_restarts_used,
    fenced_by_worker_loss,
    restart_observed_abandoned_task,
    task_restart_limit,
)
from cayu.tasks.store import TaskStore
from cayu.tasks.terminalization import TaskTerminalizationConflict

#: The validator ``run_task_worker(reconcile_abandoned_after_s=...)`` records.
SINGLE_WORKER_VALIDATOR_ID = "cayu.task-worker.single-worker"
SINGLE_WORKER_VALIDATOR_VERSION = "1"
#: Set by ``cayu worker --reconcile-abandoned-after-seconds`` for the worker process.
RECONCILE_ABANDONED_AFTER_ENV = "CAYU_WORKER_RECONCILE_ABANDONED_AFTER_S"
#: Set by ``cayu worker --restart-abandoned`` for the worker process.
RESTART_ABANDONED_ENV = "CAYU_WORKER_RESTART_ABANDONED"
#: Set by ``cayu worker --max-abandoned-restarts`` for the worker process.
MAX_ABANDONED_RESTARTS_ENV = "CAYU_WORKER_MAX_ABANDONED_RESTARTS"

_FENCE_REASONS = frozenset(
    {_TASK_CANCELLATION_REQUESTED_REASON, _TASK_RETRY_CANCELLATION_REQUESTED_REASON}
)
_HELD_STATUSES = frozenset({TaskStatus.CLAIMED, TaskStatus.RUNNING})
# A worker reads fenced tasks this many at a time, and at most _SCAN_MAX per scan.
_SCAN_PAGE = 100
_SCAN_MAX = 1_000

logger = logging.getLogger(__name__)


class AbandonedTaskSettlement(StrEnum):
    """How a task fenced after worker loss is settled once its worker is known gone.

    ``CANCEL`` settles every such task as cancelled. ``RESTART`` requeues a task
    created with ``restartable=True`` (see ``restart_abandoned_task``) while it
    has restarts left, unless someone asked to cancel it, and cancels every
    other task as ``CANCEL`` does.
    """

    CANCEL = "cancel"
    RESTART = "restart"


def is_fenced_after_worker_loss(task: Task, now: datetime | None = None) -> bool:
    """Whether ``task`` is held only by a cancellation fence whose worker's lease expired.

    Such a task still lists as ``claimed`` (or ``running``), but no worker will
    run, claim or reclaim it again until it is reconciled. A live claim, and a
    cancellation a live worker is still draining, keep a lease that has not
    expired. ``now`` defaults to the local time, while stores decide lease
    expiry by their own clock; ``TaskQuery(fenced_after_worker_loss=True)``
    asks the store.
    """

    now = datetime.now(UTC) if now is None else normalize_utc_datetime(now, "now")
    return (
        has_cancellation_fence(task)
        and task.lease_expires_at is not None
        and task.lease_expires_at <= now
    )


def has_cancellation_fence(task: Task) -> bool:
    """Whether ``task`` is held by a worker with a pending cancellation, whatever its lease."""

    if type(task) is not Task:
        raise TypeError("task must be a Task.")
    return (
        task.status in _HELD_STATUSES
        and task.status_reason in _FENCE_REASONS
        and task.worker_id is not None
        and task.lease_expires_at is not None
    )


def abandoned_task_settlement(task: Task) -> str:
    """How the application settles a fenced task, in one sentence for logs and CLI output.

    It also says why a task can't be restarted after worker loss, when it can't.
    """

    if task.retry_series is not None:
        return (
            "it is a retry-series attempt, which can't be restarted after worker loss: "
            "settle it with TaskStore.reconcile_task_retry_cancellation and "
            "retry-specific evidence"
        )
    if task.session_id is not None:
        return (
            f"it is attached to session {task.session_id}, so it can't be restarted after "
            "worker loss: settle the session first, then reconcile the task with "
            "TaskStore.reconcile_task_cancellation"
        )
    if task.restartable:
        return (
            "once its worker's process is gone, restart it with restart_abandoned_task or "
            "settle it with reconcile_abandoned_task_cancellation, or declare a "
            "single-worker deployment with run_task_worker(reconcile_abandoned_after_s=..., "
            "restart_abandoned=True) or cayu worker --reconcile-abandoned-after-seconds "
            "--restart-abandoned"
        )
    return (
        "it wasn't created with restartable=True, so it can't be restarted after worker "
        "loss: once its worker's process is gone, settle it with "
        "reconcile_abandoned_task_cancellation, or declare a single-worker deployment "
        "with run_task_worker(reconcile_abandoned_after_s=...) or "
        "cayu worker --reconcile-abandoned-after-seconds"
    )


def validate_reconcile_abandoned_after_s(value: float, name: str) -> float:
    """Return a finite, non-negative number of seconds, or raise ``ValueError``."""

    if type(value) not in {int, float} or not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite, non-negative number of seconds.")
    return float(value)


def reconcile_abandoned_after_from_environment(value: float | None) -> float | None:
    """The worker option, falling back to ``CAYU_WORKER_RECONCILE_ABANDONED_AFTER_S``."""

    if value is not None:
        return validate_reconcile_abandoned_after_s(value, "reconcile_abandoned_after_s")
    raw = os.environ.get(RECONCILE_ABANDONED_AFTER_ENV)
    if raw is None or not raw.strip():
        return None
    try:
        parsed = float(raw)
    except ValueError:
        raise ValueError(f"{RECONCILE_ABANDONED_AFTER_ENV} must be a number of seconds.") from None
    return validate_reconcile_abandoned_after_s(parsed, RECONCILE_ABANDONED_AFTER_ENV)


def abandoned_task_settlement_choice(
    value: AbandonedTaskSettlement | str,
) -> AbandonedTaskSettlement:
    """Return ``value`` as an :class:`AbandonedTaskSettlement`, or raise ``ValueError``."""

    if not isinstance(value, str):
        raise TypeError("settlement must be 'cancel' or 'restart'.")
    try:
        return AbandonedTaskSettlement(value)
    except ValueError:
        raise ValueError("settlement must be 'cancel' or 'restart'.") from None


def validate_max_restarts(value: int, name: str) -> int:
    """Return a restart cap between 0 and ``TASK_MAX_RESTARTS_LIMIT``, or raise ``ValueError``."""

    if type(value) is not int or not 0 <= value <= TASK_MAX_RESTARTS_LIMIT:
        raise ValueError(f"{name} must be an int from 0 to {TASK_MAX_RESTARTS_LIMIT}.")
    return value


def restart_abandoned_from_environment(
    restart_abandoned: bool | None,
    max_restarts: int | None,
    *,
    reconcile_abandoned_after_s: float | None,
) -> tuple[AbandonedTaskSettlement, int]:
    """The worker's settlement and restart cap, falling back to the ``cayu worker`` variables."""

    if restart_abandoned is None:
        raw = os.environ.get(RESTART_ABANDONED_ENV, "").strip().lower()
        if raw in {"", "0", "false", "no"}:
            restart_abandoned = False
        elif raw in {"1", "true", "yes"}:
            restart_abandoned = True
        else:
            raise ValueError(f"{RESTART_ABANDONED_ENV} must be 1 or 0.")
    elif type(restart_abandoned) is not bool:
        raise TypeError("restart_abandoned must be a bool or None.")
    if max_restarts is None:
        raw = os.environ.get(MAX_ABANDONED_RESTARTS_ENV, "").strip()
        if raw:
            try:
                parsed = int(raw)
            except ValueError:
                raise ValueError(f"{MAX_ABANDONED_RESTARTS_ENV} must be a whole number.") from None
            max_restarts = validate_max_restarts(parsed, MAX_ABANDONED_RESTARTS_ENV)
    else:
        max_restarts = validate_max_restarts(max_restarts, "max_abandoned_restarts")
    if max_restarts is not None and not restart_abandoned:
        raise ValueError("max_abandoned_restarts only applies with restart_abandoned=True.")
    if restart_abandoned and reconcile_abandoned_after_s is None:
        raise ValueError(
            "restart_abandoned=True needs reconcile_abandoned_after_s: only a declared "
            "single worker settles fenced tasks itself."
        )
    return (
        AbandonedTaskSettlement.RESTART if restart_abandoned else AbandonedTaskSettlement.CANCEL,
        DEFAULT_MAX_ABANDONED_RESTARTS if max_restarts is None else max_restarts,
    )


def fenced_task_scan_query(query: TaskQuery | None = None) -> TaskQuery:
    """List fenced tasks within a worker claim query's scope, oldest change first."""

    query = copy_task_query(query)
    return TaskQuery(
        type=query.type,
        parent_task_id=query.parent_task_id,
        assigned_agent_name=query.assigned_agent_name,
        has_work_contract=query.has_work_contract,
        fenced_after_worker_loss=True,
        limit=_SCAN_PAGE,
        order_by=TaskOrder.UPDATED_AT_ASC,
    )


async def _list_fenced(
    task_store: TaskStore,
    query: TaskQuery,
    *,
    max_tasks: int | None = None,
) -> tuple[list[Task], int | None]:
    """Read fenced tasks page by page from ``query.offset``.

    Returns the tasks and the offset to continue from, or ``None`` once the last
    page was read. Fenced tasks don't change until they are settled, so the
    order is stable while the pages are read.
    """

    tasks: list[Task] = []
    offset = query.offset
    while True:
        page = await task_store.list_tasks(query.model_copy(update={"offset": offset}))
        tasks.extend(page)
        offset += len(page)
        if len(page) < query.limit:
            return tasks, None
        if max_tasks is not None and len(tasks) >= max_tasks:
            return tasks, offset


async def reconcile_abandoned_tasks(
    task_store: TaskStore,
    *,
    live_worker_ids: Collection[str],
    validator_id: str,
    validator_version: str,
    query: TaskQuery | None = None,
    reconciled_by: ResolutionActor | None = None,
    settlement: AbandonedTaskSettlement | str = AbandonedTaskSettlement.CANCEL,
    max_restarts: int = DEFAULT_MAX_ABANDONED_RESTARTS,
) -> list[Task]:
    """Settle fenced tasks held by workers the caller knows are gone; return the settled tasks.

    Every task the store reports as fenced after worker loss (by its own clock)
    whose worker is not in ``live_worker_ids`` is settled as cancelled through
    ``reconcile_abandoned_task_cancellation``, with ``validator_id`` and
    ``validator_version`` recorded as the check that made the judgement and the
    expired lease as the evidence. Call it only when ``live_worker_ids`` is
    proof that every other worker's handler stopped, for example from the
    platform's record of which worker processes are running. ``query`` narrows
    the tasks considered; every fenced task in its scope is read, ``limit`` at a
    time. Session-attached and retry-series tasks are left alone;
    :func:`abandoned_task_settlement` says how to settle them. A task another
    process settles or changes first is skipped.

    With ``settlement="restart"``, a task created with ``restartable=True`` is
    requeued through ``restart_abandoned_task`` instead, with the same validator
    and evidence: it returns ``pending`` with ``restart_count`` one higher and a
    record in ``restart_history``. ``max_restarts`` caps restarts after worker
    loss for tasks that set no ``max_restarts`` of their own. A task past its
    cap is settled as cancelled, with evidence id
    ``restart-limit-reached:<cap>:lease-expired-at:...`` recording why. Other
    tasks, and a restartable task whose cancellation someone asked for before
    its worker was lost, are cancelled as with ``"cancel"``.
    """

    if isinstance(live_worker_ids, str):
        raise TypeError("live_worker_ids must be a collection of worker ids.")
    live = frozenset(require_clean_nonblank(item, "live_worker_ids") for item in live_worker_ids)
    for name, value in (("validator_id", validator_id), ("validator_version", validator_version)):
        require_clean_nonblank(value, name)
    settlement = abandoned_task_settlement_choice(settlement)
    max_restarts = validate_max_restarts(max_restarts, "max_restarts")
    selected = copy_task_query(query)
    if selected.fenced_after_worker_loss is False:
        raise ValueError("reconcile_abandoned_tasks only considers fenced tasks.")
    selected = selected.model_copy(update={"fenced_after_worker_loss": True})
    tasks, _ = await _list_fenced(task_store, selected)
    return await _settle_abandoned(
        task_store,
        [task for task in tasks if _settleable(task, live_worker_ids=live)],
        validator_id=validator_id,
        validator_version=validator_version,
        reconciled_by=reconciled_by,
        settlement=settlement,
        max_restarts=max_restarts,
    )


def _settleable(task: Task, *, live_worker_ids: frozenset[str]) -> bool:
    # The store already decided the lease expired; this keeps only what the
    # helper may settle, also when a custom store doesn't apply the filter.
    return (
        has_cancellation_fence(task)
        and task.worker_id not in live_worker_ids
        and task.session_id is None
        and task.retry_series is None
    )


async def _settle_abandoned(
    task_store: TaskStore,
    tasks: Iterable[Task],
    *,
    validator_id: str,
    validator_version: str,
    reconciled_by: ResolutionActor | None,
    settlement: AbandonedTaskSettlement,
    max_restarts: int,
) -> list[Task]:
    from cayu.tasks.worker import _reconcile_observed_abandoned_task_cancellation

    settled: list[Task] = []
    for task in tasks:
        assert task.lease_expires_at is not None
        evidence_id = f"lease-expired-at:{task.lease_expires_at.isoformat()}"
        try:
            if (
                settlement is AbandonedTaskSettlement.RESTART
                and task.restartable
                and fenced_by_worker_loss(task)
            ):
                try:
                    settled.append(
                        await restart_observed_abandoned_task(
                            task_store,
                            task,
                            validator_id=validator_id,
                            validator_version=validator_version,
                            evidence_id=evidence_id,
                            max_restarts=max_restarts,
                        )
                    )
                    continue
                except TaskRestartLimitReached as exc:
                    logger.warning(
                        "Task %s (type %s) was already restarted %d time(s) after worker "
                        "loss, its limit, so it is settled as cancelled.",
                        task.id,
                        task.type,
                        exc.restarts,
                    )
                    evidence_id = f"restart-limit-reached:{exc.max_restarts}:{evidence_id}"
            # Bound to the claim that was listed, so a claim that changed since
            # (for example a replacement worker) is skipped, never cancelled.
            result = await _reconcile_observed_abandoned_task_cancellation(
                task_store,
                task,
                validator_id=validator_id,
                validator_version=validator_version,
                evidence_id=evidence_id,
                reconciled_by=reconciled_by,
            )
        except (TaskTerminalizationConflict, TaskClaimLost) as exc:
            logger.info("Fenced task %s was not reconciled: %s", task.id, exc)
            continue
        settled.append(result)
    return settled


class FencedTaskMonitor:
    """One task worker's view of fenced tasks: settle what it may, warn about the rest once.

    With ``reconcile_abandoned_after_s`` set, the worker declares it is the only
    one claiming tasks in ``query``'s scope, so a task fenced under another
    worker id has no live owner once it has stayed fenced that long. The store
    decides that the lease expired, by its own clock; the wait is counted on
    this process's monotonic ``clock`` from the first scan that saw the task
    fenced, so clock skew between the worker and the store can't shorten it.
    One scan reads at most ``_SCAN_MAX`` fenced tasks; when there are more, the
    next scan continues where it stopped, so every fenced task is reached in
    turn. What it remembers per task is dropped once a full pass over the scope
    no longer finds the task.
    """

    def __init__(
        self,
        task_store: TaskStore,
        *,
        worker_id: str,
        query: TaskQuery | None,
        reconcile_abandoned_after_s: float | None,
        clock: Callable[[], float] = monotonic,
        settlement: AbandonedTaskSettlement = AbandonedTaskSettlement.CANCEL,
        max_restarts: int = DEFAULT_MAX_ABANDONED_RESTARTS,
    ) -> None:
        self._task_store = task_store
        self._worker_id = worker_id
        self._query = fenced_task_scan_query(query)
        self._after_s = reconcile_abandoned_after_s
        self._clock = clock
        self._settlement = abandoned_task_settlement_choice(settlement)
        self._max_restarts = validate_max_restarts(max_restarts, "max_restarts")
        self._next_offset = 0
        self._reported: set[tuple[str, datetime | None]] = set()
        self._first_seen: dict[tuple[str, datetime | None], float] = {}
        # Tasks seen since the scan that started at offset zero.
        self._traversal_seen: set[tuple[str, datetime | None]] = set()

    async def scan(self) -> list[Task]:
        """Settle eligible fenced tasks and warn about the others; return the settled ones."""

        start = self._next_offset
        tasks, next_offset = await _list_fenced(
            self._task_store,
            self._query.model_copy(update={"offset": start}),
            max_tasks=_SCAN_MAX,
        )
        self._next_offset = next_offset or 0
        if start == 0:
            self._traversal_seen = set()
        fenced = [task for task in tasks if has_cancellation_fence(task)]
        observed_at = self._clock()
        seen = {(task.id, task.lease_expires_at) for task in fenced}
        for key in seen:
            self._first_seen.setdefault(key, observed_at)
        settled: list[Task] = []
        if self._after_s is not None:
            due = [
                task
                for task in fenced
                if _settleable(task, live_worker_ids=frozenset({self._worker_id}))
                and observed_at - self._first_seen[(task.id, task.lease_expires_at)]
                >= self._after_s
            ]
            settled = await _settle_abandoned(
                self._task_store,
                due,
                validator_id=SINGLE_WORKER_VALIDATOR_ID,
                validator_version=SINGLE_WORKER_VALIDATOR_VERSION,
                reconciled_by=_declaring_worker_actor(self._worker_id),
                settlement=self._settlement,
                max_restarts=self._max_restarts,
            )
            # Settled tasks left the list, so continue that many earlier.
            self._next_offset = max(self._next_offset - len(settled), 0)
            for task in settled:
                if task.status is TaskStatus.PENDING:
                    logger.warning(
                        "Restarted task %s (type %s): it was fenced after its worker stopped "
                        "renewing its lease, and worker %s is declared the only worker "
                        "(reconcile_abandoned_after_s=%g). This is restart %d of %d after "
                        "worker loss; the handler sees task.restart_count and "
                        "task.status_payload['restart'].",
                        task.id,
                        task.type,
                        self._worker_id,
                        self._after_s,
                        abandoned_restarts_used(task),
                        task_restart_limit(task, self._max_restarts),
                    )
                    continue
                logger.warning(
                    "Settled task %s (type %s) as cancelled: it was fenced after its worker "
                    "stopped renewing its lease, and worker %s is declared the only worker "
                    "(reconcile_abandoned_after_s=%g). Create a new task if the work should "
                    "run again.",
                    task.id,
                    task.type,
                    self._worker_id,
                    self._after_s,
                )
        settled_ids = {task.id for task in settled}
        self._warn([task for task in fenced if task.id not in settled_ids])
        self._traversal_seen |= seen
        if next_offset is None:
            # A full pass over the scope ended: forget only tasks it no longer
            # found, so memory follows the fenced tasks that exist and a task's
            # wait keeps counting across passes.
            self._reported &= self._traversal_seen
            self._first_seen = {
                key: value for key, value in self._first_seen.items() if key in self._traversal_seen
            }
        return settled

    def _warn(self, fenced: list[Task]) -> None:
        for task in fenced:
            key = (task.id, task.lease_expires_at)
            if key in self._reported:
                continue
            self._reported.add(key)
            assert task.lease_expires_at is not None
            logger.warning(
                "Task %s (type %s) is fenced: worker %s stopped renewing its lease at %s "
                "while the task was running, so it stays %s with a pending cancellation "
                "and no worker will run it until it is reconciled. %s.",
                task.id,
                task.type,
                task.worker_id,
                task.lease_expires_at.isoformat(),
                task.status.value,
                self._next_step(task),
            )

    def _next_step(self, task: Task) -> str:
        if self._after_s is None or task.session_id is not None or task.retry_series is not None:
            settlement = abandoned_task_settlement(task)
            return settlement[0].upper() + settlement[1:]
        if task.worker_id == self._worker_id:
            return (
                "It is held under this worker's own id, so this worker will not settle it; "
                "settle it with reconcile_abandoned_task_cancellation once its handler stopped"
            )
        wait = f"once it has stayed fenced for {self._after_s:g} seconds"
        if self._settlement is AbandonedTaskSettlement.RESTART:
            if not task.restartable:
                return (
                    f"This worker will settle it as cancelled {wait}; it wasn't created "
                    "with restartable=True, so it can't be restarted"
                )
            if not fenced_by_worker_loss(task):
                return (
                    f"This worker will settle it as cancelled {wait}; its cancellation was "
                    "requested, so it isn't restarted"
                )
            used = abandoned_restarts_used(task)
            limit = task_restart_limit(task, self._max_restarts)
            if used < limit:
                return (
                    f"This worker will restart it {wait} (restart {used + 1} of {limit} "
                    "after worker loss)"
                )
            return (
                f"This worker will settle it as cancelled {wait}: it was already restarted "
                f"{used} time(s) after worker loss, its limit"
            )
        return f"This worker will settle it as cancelled {wait}"


def _declaring_worker_actor(worker_id: str) -> ResolutionActor:
    # Worker ids may be long; the actor subject stays a bounded, stable identity.
    subject = (
        worker_id
        if len(worker_id.encode("utf-8")) <= 200
        else f"worker-sha256:{sha256(worker_id.encode('utf-8')).hexdigest()}"
    )
    return ResolutionActor(subject=subject, source=ResolutionActorSource.SYSTEM)


__all__ = [
    "MAX_ABANDONED_RESTARTS_ENV",
    "RECONCILE_ABANDONED_AFTER_ENV",
    "RESTART_ABANDONED_ENV",
    "SINGLE_WORKER_VALIDATOR_ID",
    "SINGLE_WORKER_VALIDATOR_VERSION",
    "AbandonedTaskSettlement",
    "abandoned_task_settlement",
    "has_cancellation_fence",
    "is_fenced_after_worker_loss",
    "reconcile_abandoned_tasks",
]
