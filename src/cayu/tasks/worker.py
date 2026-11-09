"""A generic durable task-worker loop for starting fresh sessions from tasks.

``TaskStoreDispatcher.run_worker`` resumes existing sessions from dispatch
requests. This helper covers the complementary shape used by, for example, the
PR-reviewer recipe: a worker that claims arbitrary :class:`Task`\\ s and starts a
*new* session for each. It adapts task-specific claim, handling, settlement, and
recovery rules to the shared durable worker scheduler and lease-heartbeat clock,
so a caller only supplies a handler that turns a claimed task into an
``app.run(...)``.

The handler owns the task's terminal state. Ordinary tasks may run with
``RunRequest(task_id=..., task_worker_id=..., task_lease_expires_at=...)`` so the
runtime completes/fails the
task, or call ``complete_managed_task``/``fail_managed_task`` explicitly.
Retry-series tasks instead return an explicit :class:`TaskRetryAttemptReport`;
generic runtime failures are never inferred to be retryable. A handler whose
attached session durably stops at an interrupted continuation boundary may return
``TaskHandlerOutcome.SESSION_INTERRUPTED`` to release worker ownership while a
control-plane process waits to resume that session. If the handler raises or
returns ``None`` while the task is still active, the worker marks the task failed
and keeps going -- one bad task does not kill it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
from time import monotonic
from typing import TYPE_CHECKING, Any, Literal

from cayu._exception_groups import (
    exception_cause,
    exception_tree_contains,
    iter_exception_tree,
    set_exception_cause,
)
from cayu._task_wait import (
    await_shielded_task_outcome,
    restore_task_cancellation_requests,
    unexpected_child_cancellation_error,
)
from cayu._validation import canonical_durable_json_bytes, require_clean_nonblank
from cayu.approvals.tools import ResolutionActor, ResolutionActorSource
from cayu.deadlines import (
    _DEFAULT_HANDLER_GRACE_SECONDS,
    ExecutionDeadline,
    ExecutionDeadlineExceeded,
    HandlerTimeLimit,
    default_run_deadline,
    handler_grace_seconds,
    require_handler_seconds,
)
from cayu.events import (
    Event,
    EventType,
    event_with_runtime_envelope_authority,
    event_with_runtime_generated_id,
    event_with_runtime_payload_authority,
)
from cayu.runtime._continuation_task_failure import (
    RuntimeTaskFailureTerminalizationPending,
    runtime_task_failure_identity_from_task,
    runtime_task_failure_terminalization_pending_from_error,
    runtime_task_failure_terminalization_request,
    runtime_task_failure_terminalization_request_sha256,
)
from cayu.runtime._durable_worker_loop import (
    DurableWorkerCadence,
    DurableWorkerDemandPolicy,
    DurableWorkerMetrics,
    DurableWorkerPollerGroup,
    DurableWorkerStep,
    record_task_admission_to_claim_latency,
    run_durable_lease_heartbeat,
    run_durable_worker_loop,
    validate_worker_interval,
    wait_or_stop,
    worker_event_loop_lag_monitor,
    worker_stop_requested,
)
from cayu.runtime._event_loop_lag_sessions import app_session_event_loop_lag_recorder
from cayu.runtime._task_lease_authority import (
    TaskLeaseAuthority as _TaskLeaseAuthority,
)
from cayu.runtime._task_lease_authority import (
    bind_task_lease_authority,
)
from cayu.runtime._task_store_operation_boundary import (
    capture_task_store_operation,
    raise_task_store_operation_failure,
    task_store_cancellation_reconciliation_capability_is_complete,
    task_store_interrupted_handoff_capability_is_complete,
    task_store_mutation_is_cancellation_quiescent,
)
from cayu.runtime.application_lifecycle import (
    ApplicationAdmission,
    ApplicationAdmissionsSealed,
    _current_operation_admission,
)
from cayu.sessions._execution_profile_checkpoint import (
    active_invocation_execution_profile_from_checkpoint,
)
from cayu.sessions._invocation_terminal_decision import (
    InvocationTerminalOutcome,
    invocation_terminal_decision_from_checkpoint,
    invocation_terminal_decision_matches_active_profile,
)
from cayu.sessions.authority import SessionRunFenced
from cayu.sessions.event_queries import EventOrder, EventQuery
from cayu.sessions.pending_action_contracts import PendingActionQuery
from cayu.sessions.records import Session, SessionStatus
from cayu.sessions.recovery import IncompleteSessionRecoveryRequest
from cayu.tasks._cancellation import (
    _cancellation_drain_holds,
    _task_cancellation_terminalization_request,
)
from cayu.tasks._event_budget_warnings import EventBudgetCheck, with_event_budget_expiry
from cayu.tasks._execution_settlement import TaskExecutionSettlement
from cayu.tasks._group_maintenance import TaskGroupMaintenance
from cayu.tasks._schedule_wakeup import next_schedule_wake_at, observe_schedule_wakeup
from cayu.tasks._task_run_settlement import (
    UnstartedClaim,
    acknowledge_stopped_handler,
    cancellation_carried_by,
    release_unstarted_claims,
)
from cayu.tasks._terminalization import (
    _terminalize_claimed_task,
    _terminalize_claimed_task_or_detect_peer_winner,
    settle_task_retry_attempt_with_retry,
)
from cayu.tasks.abandoned import (
    AbandonedTaskSettlement,
    FencedTaskMonitor,
    _declaring_worker_actor,
    reconcile_abandoned_after_from_environment,
    restart_abandoned_from_environment,
)
from cayu.tasks.cancellation import (
    _TASK_CANCELLATION_REQUESTED_REASON,
    _TASK_RETRY_CANCELLATION_REQUESTED_REASON,
    TaskCancellationReconciliationEvent,
    TaskCancellationReconciliationEvidence,
    TaskCancellationReconciliationOutcome,
    TaskCancellationReconciliationRequest,
    TaskRetryCancellationReconciliationEvent,
    TaskRetryCancellationReconciliationEvidence,
    TaskRetryCancellationReconciliationOutcome,
    TaskRetryCancellationReconciliationRequest,
    _task_cancellation_requested,
    _task_retry_cancellation_requested,
)
from cayu.tasks.continuation_claims import TaskContinuationExpectation, TaskContinuationUnsupported
from cayu.tasks.handler_cancellation import TaskCancellationSignal
from cayu.tasks.handoff import (
    InterruptedTaskContinuationClaimPage,
    TaskInterruptedHandoffConflict,
    TaskInterruptedHandoffReceipt,
    TaskInterruptedHandoffRequest,
    TaskParkedWait,
    claim_interrupted_task_handoff_request,
    interrupted_task_handoff_request,
    new_interrupted_task_continuation_handoff_id,
    prepare_interrupted_task_handoff,
)
from cayu.tasks.queries import TaskQuery
from cayu.tasks.records import (
    _TERMINAL_TASK_STATUSES,
    Task,
    TaskClaimLost,
    TaskRetrySeriesDisposition,
    TaskStatus,
    copy_task,
)
from cayu.tasks.retry import (
    TaskRetryAttemptDisposition,
    TaskRetryAttemptReport,
    TaskRetrySettlementRequest,
    TaskRetrySettlementResult,
    _task_retry_requested_cancellation_settlement,
    _task_retry_runtime_terminal_request,
)
from cayu.tasks.retry_budgets import enforce_task_event_deadline, reconcile_task_event_deadlines
from cayu.tasks.store import TaskStore
from cayu.tasks.terminalization import (
    TaskTerminalizationConflict,
    TaskTerminalizationRequest,
    TaskTerminalKind,
)
from cayu.tasks.until_idle import _take_until_idle_state

if TYPE_CHECKING:
    from cayu.applications import CayuApp
    from cayu.tasks.retry_budgets import TaskRetryKeyState
    from cayu.tasks.scheduling import TaskScheduleWakeup


class TaskHandlerOutcome(StrEnum):
    """Explicit non-terminal outcomes supported by :func:`run_task_worker`."""

    SESSION_INTERRUPTED = "session_interrupted"


class _TaskHeartbeatOutcome(StrEnum):
    STOPPED = "stopped"
    TERMINAL = "terminal"
    ELAPSED = "elapsed"
    CANCELLATION_REQUESTED = "cancellation_requested"


TaskHandler = Callable[
    ["CayuApp", Task, str],
    Awaitable[TaskHandlerOutcome | TaskRetryAttemptReport | None],
]
RecoveredInterruptedTaskHandler = Callable[
    ["CayuApp", Task, str],
    Awaitable[TaskHandlerOutcome | None],
]
# Matches the runtime's diagnostic bound so long validation errors and command
# output stay readable; truncation is marked instead of silent.
_MAX_TASK_FAILURE_MESSAGE_BYTES = 4 * 1024
_TASK_FAILURE_TRUNCATION_MARKER = "\u2026[truncated]"
# A cancelled handler that is still running this long after the grace period
# (never less than this floor) gets a durable cancellation request.
_HANDLER_STOP_WAIT_FLOOR_SECONDS = 1.0
_HANDLER_TIME_LIMIT_FENCE_CODE = "task_handler_timeout"
_INTERRUPTED_HANDOFF_MAX_ATTEMPTS = 3
_INTERRUPTED_HANDOFF_INITIAL_BACKOFF_SECONDS = 0.05
_INTERRUPTED_HANDOFF_MAX_BACKOFF_SECONDS = 1.0
_INTERRUPTED_HANDOFF_RECOVERY_BATCH_SIZE = 10
# Recorded on the cancellation of an attached task whose worker's lease expired.
_EXPIRED_ATTACHED_OWNER_VALIDATOR_ID = "cayu.task-worker.expired-attached-owner"
_EXPIRED_ATTACHED_OWNER_VALIDATOR_VERSION = "1"
_RECLAIM_BATCH_SIZE = 100
_INTERRUPTED_HANDOFF_RECOVERY_RESCAN_SECONDS = 30.0
_DEFAULT_MAX_UNPRODUCTIVE_INTERRUPTED_HANDOFFS = 8
_UNPRODUCTIVE_CONTINUATION_INITIAL_BACKOFF_SECONDS = 1.0
_UNPRODUCTIVE_CONTINUATION_MAX_BACKOFF_SECONDS = 30.0
# Enough handoffs for the backoff to reach its maximum when no bound is set.
_UNPRODUCTIVE_CONTINUATION_BACKOFF_COUNT_CAP = 8
# Far past the point where the backoff saturates, and small enough that the
# float conversion cannot overflow for any configured bound.
_UNPRODUCTIVE_CONTINUATION_MAX_BACKOFF_EXPONENT = 16
_UNPRODUCTIVE_HANDOFF_HISTORY_PAGE_SIZE = 500
# A continuation counts as productive once the session has new model output,
# finished tool work, or new input that a later continuation could act on.
_CONTINUATION_PROGRESS_EVENT_TYPES = (
    EventType.MODEL_COMPLETED,
    EventType.TOOL_CALL_COMPLETED,
    EventType.TOOL_CALL_APPROVED,
    EventType.TOOL_CALL_APPROVAL_DENIED,
    EventType.SESSION_MESSAGE_QUEUED,
)

logger = logging.getLogger(__name__)


async def complete_managed_task(
    task_store: TaskStore,
    task: Task,
    worker_id: str,
    result: dict[str, Any],
) -> Task:
    """Complete a managed worker task under the claim generation its handler received."""

    task = copy_task(task)
    return await task_store.complete_task(
        task.id,
        result,
        worker_id=worker_id,
        handoff_id=task.interrupted_handoff_id,
        claim_generation=task.claim_generation,
    )


async def fail_managed_task(
    task_store: TaskStore,
    task: Task,
    worker_id: str,
    error: dict[str, Any],
) -> Task:
    """Fail a managed worker task under the claim generation its handler received."""

    task = copy_task(task)
    return await task_store.fail_task(
        task.id,
        error,
        worker_id=worker_id,
        handoff_id=task.interrupted_handoff_id,
        claim_generation=task.claim_generation,
    )


async def reconcile_abandoned_task_cancellation(
    task_store: TaskStore,
    task_id: str,
    *,
    validator_id: str,
    validator_version: str,
    evidence_id: str,
    reconciled_by: ResolutionActor | None = None,
) -> Task:
    """Settle an ordinary task whose worker died while its cancellation was draining.

    Cayu never hands a started task to another worker after its lease expires:
    expired-claim reclamation fences it as a cancellation request instead, and
    cancelling ``run_task_worker`` records the same fence and waits for the handler
    to finish. The owning worker settles the fence once its handler is quiescent.
    If that process died first, the task stays claimed with ``status_reason``
    ``"cancellation_requested"`` until the application attests quiescence.

    Call this once the application knows the original worker can no longer act
    (for example, its process exited or its host restarted) and the handler left
    no partial external effect it would need to finish. ``validator_id`` and
    ``validator_version`` name the application check that made that judgement and
    ``evidence_id`` identifies the fact it relied on. The store still refuses while
    the original lease is live, but lease expiry does not prove that the handler
    stopped. Independently establish quiescence before calling this helper;
    a cancellation-fenced handler can still run after its lease expires.
    Session-attached tasks are not supported: once their lease expires, a task
    worker's interrupted-handoff recovery interrupts the session, which fences the
    old run, and settles the cancellation itself.

    Returns the cancelled task; a task this already settled is returned unchanged.
    """

    for name, value in (
        ("validator_id", validator_id),
        ("validator_version", validator_version),
        ("evidence_id", evidence_id),
    ):
        require_clean_nonblank(value, name)
    current = await task_store.load_task(task_id)
    if current is None:
        raise TaskTerminalizationConflict("Abandoned task cancellation lost its task.")
    if current.status is TaskStatus.CANCELLED:
        return current
    return await _reconcile_observed_abandoned_task_cancellation(
        task_store,
        current,
        validator_id=validator_id,
        validator_version=validator_version,
        evidence_id=evidence_id,
        reconciled_by=reconciled_by,
    )


async def _reconcile_observed_abandoned_task_cancellation(
    task_store: TaskStore,
    current: Task,
    *,
    validator_id: str,
    validator_version: str,
    evidence_id: str,
    reconciled_by: ResolutionActor | None,
) -> Task:
    """Cancel the exact fenced claim ``current`` shows; the store refuses a changed claim."""

    if current.session_id is not None:
        raise TaskTerminalizationConflict(
            "Session-attached tasks are not supported; a task worker's interrupted-handoff "
            "recovery interrupts the session and settles the cancellation once the lease "
            "expires."
        )
    if current.retry_series is not None:
        raise TaskTerminalizationConflict(
            "Retry-series cancellations require TaskStore.reconcile_task_retry_cancellation; "
            "reconcile_abandoned_task_cancellation only settles ordinary tasks."
        )
    worker_id = current.worker_id
    terminalization = (
        _task_cancellation_terminalization_request(current, worker_id=worker_id)
        if worker_id is not None
        else None
    )
    if worker_id is None or terminalization is None:
        raise TaskTerminalizationConflict(
            "Only a claimed task with a pending cancellation can be reconciled as abandoned."
        )
    evidence_sha256 = sha256(
        canonical_durable_json_bytes(
            {
                "schema": "cayu.abandoned-task-cancellation.v1",
                "task_id": current.id,
                "worker_id": worker_id,
                "cancellation_idempotency_key": terminalization.idempotency_key,
                "validator_id": validator_id,
                "validator_version": validator_version,
                "evidence_id": evidence_id,
            },
            "abandoned_task_cancellation",
        )
    ).hexdigest()
    request = _owner_lost_cancellation_request(
        current,
        worker_id,
        cancellation_idempotency_key=terminalization.idempotency_key,
        reconciliation_idempotency_key=f"abandoned-task-cancellation:v1:{evidence_sha256}",
        reconciled_by=reconciled_by
        or ResolutionActor(subject=validator_id, source=ResolutionActorSource.SYSTEM),
        validator_id=validator_id,
        validator_version=validator_version,
        evidence_id=evidence_id,
        evidence_sha256=evidence_sha256,
    )
    return (await task_store.reconcile_task_cancellation(request)).task


class _TaskRetryElapsed(Exception):
    """The durable retry-series deadline elapsed while its handler was active."""

    def __init__(
        self,
        owner_cancellation: asyncio.CancelledError | None = None,
        report: TaskRetryAttemptReport | None = None,
    ) -> None:
        super().__init__("Task retry deadline elapsed.")
        self.owner_cancellation = owner_cancellation
        self.report = report


class _TaskInterruptedHandoffRecoveryRequired(RuntimeError):
    """The task/session link is intact but ownership release needs recovery."""


class _TaskInterruptedHandoffReceiptConflict(TaskInterruptedHandoffConflict):
    """Returned receipt evidence conflicts independently of lease expiry."""


class _TaskRetryCancellationRequested(Exception):
    """An operator requested cancellation while the attempt remained owned."""

    def __init__(
        self,
        owner_cancellation: asyncio.CancelledError | None = None,
        report: TaskRetryAttemptReport | None = None,
    ) -> None:
        super().__init__("Task retry cancellation was requested.")
        self.owner_cancellation = owner_cancellation
        self.report = report


@dataclass(frozen=True)
class _TaskRetryQuiescence:
    """Owned evidence collected after a retry handler can no longer mutate."""

    owner_cancellation: asyncio.CancelledError | None
    report: TaskRetryAttemptReport | None


@dataclass(frozen=True)
class _TaskCancellationDelivery:
    """How a running handler hears that its task was asked to stop."""

    signal: TaskCancellationSignal
    # Seconds between the request and cancelling the handler; None never cancels it.
    interrupt_after_s: float | None
    # Whether the callback is still running and nothing else has cancelled it yet.
    may_interrupt: Callable[[], bool]


def _cancellation_grace_seconds(
    *,
    interrupt_on_cancel: bool,
    execution_timeout_s: float | None,
    execution_grace_s: float | None,
) -> float | None:
    """The grace a cancelled handler gets: the handler time limit's grace period."""

    if not interrupt_on_cancel:
        return None
    if execution_grace_s is not None:
        return execution_grace_s
    if execution_timeout_s is not None:
        return handler_grace_seconds(execution_timeout_s, None)
    return _DEFAULT_HANDLER_GRACE_SECONDS


@dataclass(frozen=True)
class _InterruptedHandoffRecoveryPage:
    """One scheduler-bounded expired-owner recovery turn."""

    recovered: int
    next_after: tuple[datetime, str] | None
    exhausted: bool
    # Tasks whose session's recovery failed; a later pass may succeed.
    failed: frozenset[str] = frozenset()
    # A session's recovery needs a person (``cayu recovery``) before any pass can.
    manual_recovery_required: bool = False


class _TaskRetryHandlerUnsettled(RuntimeError):
    """A handler remained live after its durable retry authority was lost."""


@dataclass(slots=True)
class _TaskWorkerBudget:
    """Event-loop-local reservations keep concurrent claims within max_tasks."""

    remaining: int
    reserved: int = 0

    @property
    def exhausted(self) -> bool:
        return self.remaining == 0 and self.reserved == 0

    def reserve(self) -> bool:
        if self.remaining == 0:
            return False
        self.remaining -= 1
        self.reserved += 1
        return True

    def settle(self, handled: int) -> None:
        self.remaining += 1 - handled
        self.reserved -= 1


async def run_task_worker(
    app: CayuApp,
    task_store: TaskStore,
    handler: TaskHandler,
    *,
    worker_id: str,
    query: TaskQuery | None = None,
    lease_seconds: int = 300,
    poll_interval_s: float = 1.0,
    metrics: DurableWorkerMetrics | None = None,
    minimum_idle_delay_s: float | None = None,
    maximum_idle_delay_s: float | None = None,
    idle_backoff_multiplier: float = 2.0,
    idle_jitter_ratio: float = 0.1,
    reclaim: bool = True,
    reclaim_every_s: float = 60.0,
    reconcile_abandoned_after_s: float | None = None,
    restart_abandoned: bool | None = None,
    max_abandoned_restarts: int | None = None,
    recover_interrupted_handoffs: bool = True,
    interrupted_handoff_recovery_every_s: float = (_INTERRUPTED_HANDOFF_RECOVERY_RESCAN_SECONDS),
    recovered_interrupted_task_handler: RecoveredInterruptedTaskHandler | None = None,
    continuation_poll_interval_s: float | None = None,
    max_unproductive_interrupted_handoffs: int | None = (
        _DEFAULT_MAX_UNPRODUCTIVE_INTERRUPTED_HANDOFFS
    ),
    stop: asyncio.Event | None = None,
    max_tasks: int | None = None,
    concurrency: int = 1,
    execution_timeout_s: float | None = None,
    execution_grace_s: float | None = None,
    interrupt_on_cancel: bool = True,
) -> int:
    """Claim and handle durable tasks until stopped; return the number handled.

    For each claimed task, ``handler(app, task, worker_id)`` is awaited while the
    task lease is heartbeated in the background. The handler typically builds a
    ``RunRequest(task_id=task.id, task_worker_id=worker_id,
    task_lease_expires_at=task.lease_expires_at, ...)`` and awaits
    ``app.run(...)`` so the runtime completes/fails the task. If that run ends at
    a durable interrupted boundary, return ``TaskHandlerOutcome.SESSION_INTERRUPTED``
    so the helper validates the session and releases only the task's worker lease.

    - ``query`` scopes which tasks this worker claims (e.g. by type / assigned agent).
    - ``lease_seconds`` is the claim lease; the lease is re-extended at ~1/3 of it.
    - ``poll_interval_s`` is the bounded dispatch-latency objective when idle.
    - ``minimum_idle_delay_s`` / ``maximum_idle_delay_s`` bound adaptive polling.
    - ``idle_backoff_multiplier`` and ``idle_jitter_ratio`` control empty-claim
      backoff without changing the dispatch-latency bound.
    - ``reclaim`` periodically returns expired claims from dead workers to pending
      when their handler never started. A started task whose owner is gone is
      fenced as a cancellation request instead of being repeated, and stays
      claimed until it is settled; see ``reconcile_abandoned_task_cancellation``
      for ordinary tasks and ``TaskStore.reconcile_task_retry_cancellation`` for
      retry series.
    - ``reclaim_every_s`` controls that cadence independently of task claims.
      At the same cadence the worker looks for tasks in its ``query``'s scope
      that are fenced after worker loss (see
      :func:`cayu.tasks.is_fenced_after_worker_loss`) and logs a warning once
      for each, naming the task and how to settle it. Both keep their cadence
      while a handler runs, so a worker busy with a long task still fences,
      and with ``reconcile_abandoned_after_s`` settles, a task another worker
      lost; a failure then is logged and retried at the next interval.
    - ``reconcile_abandoned_after_s`` declares that this is the only worker
      claiming tasks in its ``query``'s scope. A task the store reports fenced
      after worker loss, held by another worker id, is then settled once its
      lease expired this many seconds ago (by the store's lease clock, so a
      worker clock that runs ahead can't shorten the wait, and a worker started
      again, such as with ``--until-idle``, keeps counting from the lease; a
      store without ``supports_lease_clock`` counts from this process's
      first sight), as cancelled
      through ``reconcile_abandoned_task_cancellation``, with the
      declaration (``cayu.task-worker.single-worker``) recorded as the
      validator. ``None`` (the default) settles nothing, unless ``cayu worker
      --reconcile-abandoned-after-seconds`` set it for the process. Don't set
      it when several workers or processes claim the same tasks: a slow worker
      may still be running a task whose lease expired.
    - ``restart_abandoned=True`` makes that declared single worker requeue a
      task created with ``restartable=True`` instead of cancelling it, through
      ``restart_abandoned_task``: the task returns to ``pending`` with
      ``restart_count`` one higher and a record in ``restart_history``, and the
      next handler sees ``task.status_payload["restart"]``. It requires
      ``reconcile_abandoned_after_s``. Other tasks are still cancelled.
      ``max_abandoned_restarts`` (2 by default) caps restarts after worker loss
      for tasks that set no ``max_restarts`` of their own; a task past its cap
      is cancelled as before, with evidence that says so. ``None`` for either
      reads ``cayu worker --restart-abandoned`` and
      ``--max-abandoned-restarts`` through their environment variables.
    - ``recover_interrupted_handoffs`` owns bounded recovery of expired attached
      task workers. Elect one such scanner for each shared task store.
    - ``interrupted_handoff_recovery_every_s`` controls the scanner's exhausted
      rescan cadence independently of claim and expired-lease polling.
    - ``recovered_interrupted_task_handler`` makes this worker available to claim
      and continue receipt-backed workerless tasks. This execution capacity is
      independent of expired-owner scanner election.
    - ``continuation_poll_interval_s`` bounds rediscovery latency after an empty
      continuation scan. It defaults to ``poll_interval_s``.
    - ``max_unproductive_interrupted_handoffs`` bounds how many interrupted
      handoffs a task's session may record without model, tool, or input
      progress. A continuation claimed at that bound fails the task instead of
      running the handler, unless the session has a pending action. ``None``
      disables the bound; repeated unproductive continuations still back off.
    - ``stop`` is an ``asyncio.Event`` for graceful shutdown: no new claims, and
      handlers already running finish. Cancelling this coroutine instead fences
      each running task as a cancellation request and still waits for its handler;
      if the process exits first, the task needs
      ``reconcile_abandoned_task_cancellation``.
    - ``max_tasks`` bounds the total handled across all slots, including recovered
      continuations and contracted-task parking. To return once nothing is
      claimable, use :func:`cayu.tasks.run_task_worker_until_idle`.
    - ``concurrency`` is the positive number of independently heartbeated handler
      slots; it defaults to one. Handler failures are isolated. Setting ``stop``
      stops new claims and drains active handlers; cancelling this coroutine
      applies the existing cancellation/settlement rules to every active slot.
      Infrastructure errors stop and drain the group before propagating.
    - ``execution_timeout_s`` is the longest one handler call may take, grace
      period included, counted from handler start, whether or not its runs are
      attached to the task. ``None`` (the default) sets no limit. The last
      ``execution_grace_s`` seconds of it are the grace period (30 seconds by
      default, or half the limit if that is shorter; an explicit value must be
      smaller than the limit). With 600 and 30, new top-level runs in the call
      are refused with ``ExecutionDeadlineExceeded`` from 570 seconds, and
      :func:`cayu.deadlines.current_handler_time_limit` reports it expired. Runs
      the handler starts without their own ``execution_deadline`` get that
      570-second mark as their deadline, so a hung run ends
      ``session.interrupted`` with ``limit_reached`` and leaves the grace period
      to deliver. At 600 seconds the handler is cancelled; a run still in flight
      ends ``session.interrupted``. Once the handler has stopped, a task it did
      not settle fails with ``ExecutionDeadlineExceeded`` (scope
      ``task_handler``, with the IDs of the sessions it started); a task with a
      retry policy records a retryable failed attempt instead, and a pending
      cancellation still wins. The worker logs a warning when the soft limit
      passes and another when it fails the task. A restartable task is not
      requeued. A handler that keeps running after it is cancelled keeps its lease, and its task
      gets a durable cancellation request (code ``task_handler_timeout``) that
      settles when it returns. The limit is
      :class:`cayu.deadlines.HandlerTimeLimit`, which a hand-written worker can
      use on its own. The limit ends one handler call, not the work: an
      application that creates a new task for the same event should pass
      ``TaskCreate(retry_key=...)`` so the key's ``TaskRetryBudget`` stops it.
      A task's event budget also ends its handler when it runs out. Tasks
      created without ``retry_budget`` or ``retry_policy`` have no elapsed
      limit by default, so this limit decides. When the worker claims a task
      whose event budget is shorter than ``execution_timeout_s``, it logs a
      warning once per task type; the budget still applies.
    - ``interrupt_on_cancel`` decides what happens when a running task is
      cancelled (``TaskStore.cancel_task``, ``cayu task cancel``). The worker
      sees the request at its next heartbeat and requests the handler's
      :func:`cayu.tasks.current_task_cancellation` signal, so new top-level
      runs in the call are refused with ``TaskCancellationRequested`` and
      handler code can stop work that is not an ``app.run``. With ``True``
      (the default) the handler is then cancelled once the grace period
      passes (``execution_grace_s``; without it the time limit's default
      grace, or 30 seconds when there is no limit), so a run still in flight
      ends ``session.interrupted``. With ``False`` the worker only signals and
      waits, for handlers that must not be interrupted. Either way it keeps
      renewing the lease until the handler has stopped and then settles the
      task ``cancelled`` with the request's reason, whatever the handler did.
    """
    until_idle = _take_until_idle_state()
    if until_idle is not None and recovered_interrupted_task_handler is None:
        until_idle.continuations_scanned = True
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError("concurrency must be a positive int.")
    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive.")
    validate_worker_interval(poll_interval_s, "poll_interval_s")
    # Validates both settings before any claim.
    if execution_timeout_s is not None:
        HandlerTimeLimit(execution_timeout_s, grace_s=execution_grace_s)
    elif execution_grace_s is not None:
        require_handler_seconds(execution_grace_s, "execution_grace_s")
    if type(interrupt_on_cancel) is not bool:
        raise TypeError("interrupt_on_cancel must be a bool.")
    if metrics is not None and not isinstance(metrics, DurableWorkerMetrics):
        raise TypeError("metrics must be a DurableWorkerMetrics instance.")
    demand_policy = DurableWorkerDemandPolicy(
        dispatch_latency_s=poll_interval_s,
        minimum_idle_delay_s=minimum_idle_delay_s,
        maximum_idle_delay_s=maximum_idle_delay_s,
        backoff_multiplier=idle_backoff_multiplier,
        jitter_ratio=idle_jitter_ratio,
    )
    validate_worker_interval(reclaim_every_s, "reclaim_every_s")
    reconcile_abandoned_after_s = reconcile_abandoned_after_from_environment(
        reconcile_abandoned_after_s
    )
    abandoned_settlement, max_abandoned_restarts = restart_abandoned_from_environment(
        restart_abandoned,
        max_abandoned_restarts,
        reconcile_abandoned_after_s=reconcile_abandoned_after_s,
    )
    if type(recover_interrupted_handoffs) is not bool:
        raise TypeError("recover_interrupted_handoffs must be a bool.")
    validate_worker_interval(
        interrupted_handoff_recovery_every_s,
        "interrupted_handoff_recovery_every_s",
    )
    if continuation_poll_interval_s is None:
        continuation_poll_interval_s = poll_interval_s
    else:
        validate_worker_interval(
            continuation_poll_interval_s,
            "continuation_poll_interval_s",
        )
    if max_unproductive_interrupted_handoffs is not None and (
        type(max_unproductive_interrupted_handoffs) is not int
        or max_unproductive_interrupted_handoffs < 1
    ):
        raise ValueError("max_unproductive_interrupted_handoffs must be a positive int or None.")
    if max_tasks is not None and (type(max_tasks) is not int or max_tasks < 0):
        raise ValueError("max_tasks must be non-negative (an int or None).")
    worker_id = require_clean_nonblank(worker_id, "worker_id")
    if app.redact_json(worker_id) != worker_id:
        raise ValueError(
            "worker_id contains a workload secret and cannot be used as durable task authority."
        )
    if max_tasks == 0 or _worker_should_stop(app, stop):
        return 0
    if (
        abandoned_settlement is AbandonedTaskSettlement.RESTART
        and type(task_store).restart_abandoned_task is TaskStore.restart_abandoned_task
    ):
        raise NotImplementedError(
            "restart_abandoned=True needs a task store that implements restart_abandoned_task."
        )
    if not task_store_cancellation_reconciliation_capability_is_complete(task_store):
        raise NotImplementedError(
            "The generic task worker requires complete idempotent ordinary-task "
            "cancellation reconciliation before it can claim work."
        )
    handler = await _with_session_wakes(app, task_store, handler)
    materialized_work_contract_queue_supported = (
        task_store.supports_verified_work_contracts
        or type(task_store).hold_claimed_work_contract_task
        is not TaskStore.hold_claimed_work_contract_task
    )
    verified_mutation_methods = [
        ("hold_claimed_work_contract_task", "contracted-task parking"),
        ("claim_task", "task claiming"),
    ]
    if reclaim:
        verified_mutation_methods.append(("reclaim_expired", "expired-claim reclamation"))
    if materialized_work_contract_queue_supported:
        for method_name, operation_name in verified_mutation_methods:
            if task_store_mutation_is_cancellation_quiescent(task_store, method_name):
                continue
            raise NotImplementedError(
                f"The {operation_name} implementation must explicitly guarantee that "
                "verified-work mutations are cancellation-quiescent."
            )

    task_store_declarations = type.__getattribute__(type(task_store), "__dict__")
    interrupted_handoff_advertised = (
        task_store_declarations.get("supports_interrupted_task_handoffs") is True
    )
    interrupted_handoff_supported = (
        interrupted_handoff_advertised
        and task_store_interrupted_handoff_capability_is_complete(task_store)
    )
    if interrupted_handoff_advertised and not interrupted_handoff_supported:
        raise NotImplementedError(
            "The interrupted-task handoff capability requires stable receipt/recovery "
            "methods and cancellation-quiescent ownership mutations."
        )
    if recovered_interrupted_task_handler is not None and (
        not interrupted_handoff_supported
        or type(task_store).claim_interrupted_task_continuation
        is TaskStore.claim_interrupted_task_continuation
        or type(task_store).load_active_attached_task_worker
        is TaskStore.load_active_attached_task_worker
        or not task_store_mutation_is_cancellation_quiescent(
            task_store,
            "claim_interrupted_task_continuation",
        )
    ):
        raise NotImplementedError(
            "Recovered interrupted-task handlers require an atomic, "
            "cancellation-quiescent continuation claim."
        )

    budget = _TaskWorkerBudget(max_tasks) if concurrency > 1 and max_tasks is not None else None
    event_budget_check = EventBudgetCheck(execution_timeout_s)
    group_maintenance = TaskGroupMaintenance.for_store(task_store)
    fenced_monitor = (
        FencedTaskMonitor(
            task_store,
            worker_id=worker_id,
            query=query,
            reconcile_abandoned_after_s=reconcile_abandoned_after_s,
            settlement=abandoned_settlement,
            max_restarts=max_abandoned_restarts,
            # A run that exits once idle scans once, so it reads every fenced task.
            whole_pass=until_idle is not None,
        )
        if reclaim or reconcile_abandoned_after_s is not None
        else None
    )
    if until_idle is not None:
        until_idle.fenced = fenced_monitor
    infrastructure_stop = asyncio.Event()
    slot_stop = infrastructure_stop if concurrency > 1 else stop

    def slot_should_stop() -> bool:
        return infrastructure_stop.is_set() or _worker_should_stop(app, stop)

    async def run_slot(*, owns_maintenance: bool) -> int:
        expired_handoff_after: tuple[datetime, str] | None = None
        continuation_after: tuple[datetime, str] | None = None
        continuation_scan_scanned = 0
        continuation_scan_rejected = 0
        continuation_scan_filtered = 0
        next_interrupted_handoff_recovery_at = 0.0
        next_interrupted_continuation_scan_at = 0.0
        subscribe_to_poller = getattr(task_store, "_durable_worker_poller", None)
        if callable(subscribe_to_poller):
            poller = subscribe_to_poller((query,), demand_policy, clock=monotonic)
        else:
            poller = DurableWorkerPollerGroup().subscribe(demand_policy, clock=monotonic)
        if metrics is not None:
            poller.set_metrics(metrics)
        reclaim_cadence = DurableWorkerCadence(every_s=reclaim_every_s)
        reclaim_backlog = False
        fenced_cadence = DurableWorkerCadence(every_s=reclaim_every_s)
        # Set when reconciling session wakes released parked handoffs. The
        # continuation scan starts over once no scan in flight can overwrite it.
        handoffs_released = False

        def until_idle_skips_scan() -> bool:
            # An until-idle run scans once, in one slot: a session waiting on
            # a person would otherwise be continued again after every task.
            return until_idle is not None and (
                not owns_maintenance or until_idle.continuations_scanned
            )

        async def scan_fenced_tasks() -> bool:
            assert fenced_monitor is not None
            try:
                settled = await fenced_monitor.scan()
                if until_idle is not None:
                    until_idle.fenced_scan_failed = False
                return bool(settled)
            except Exception:
                logger.exception(
                    "Looking for fenced tasks failed; retrying in %g seconds.", reclaim_every_s
                )
                if until_idle is not None:
                    until_idle.fenced_scan_failed = True
                return False

        # An until-idle run names the bound so it can tell when one call left more.
        reclaim_bound = {"max_reclaims": _RECLAIM_BATCH_SIZE} if until_idle is not None else {}

        def rescan_continuations() -> None:
            nonlocal continuation_after, next_interrupted_continuation_scan_at
            # Newly handed-off work may sort behind the scan cursor, and an
            # until-idle run that already scanned must scan again before returning.
            next_interrupted_continuation_scan_at = 0.0
            if until_idle is not None and recovered_interrupted_task_handler is not None:
                continuation_after = None
                until_idle.continuations_scanned = False

        async def reclaim_expired_tasks() -> int:
            nonlocal handoffs_released
            # The same cadence restores wake attempts a crash or failure left out,
            # and releases parked tasks whose sleep has gone away.
            if await _reconcile_session_wakes(app, task_store):
                handoffs_released = True
            await _sweep_external_waits(app, task_store, query)
            if materialized_work_contract_queue_supported:
                reclaim_outcome = await capture_task_store_operation(
                    lambda: task_store.reclaim_expired(query=query, **reclaim_bound),
                    operation_name="Expired task-claim reclamation",
                    redactor=app._secret_redactor,
                    mutation_store=task_store,
                    mutation_method_name="reclaim_expired",
                )
                if reclaim_outcome.failure is not None:
                    failure = reclaim_outcome.failure
                    del reclaim_outcome
                    raise_task_store_operation_failure(failure)
                reclaimed = reclaim_outcome.result
                del reclaim_outcome
                reclaimed_count = 0 if reclaimed is None else len(reclaimed)
            else:
                reclaimed_count = len(
                    await task_store.reclaim_expired(query=query, **reclaim_bound)
                )
            # Reclamation has just recorded the fences of lost wake attempts.
            await _settle_wake_attempts_of_ended_tasks(app, task_store, worker_id)
            return reclaimed_count

        async def maintain_expired_claims() -> bool:
            """Reclaim expired claims and look for fenced tasks when each is due.

            Returns whether either changed a task.
            """

            nonlocal reclaim_backlog
            loop = asyncio.get_running_loop()
            changed = False
            if reclaim:
                reclaim_ran, reclaimed = await reclaim_cadence.run_if_due(
                    reclaim_expired_tasks,
                    now=loop.time(),
                    clock=loop.time,
                )
                if reclaim_ran:
                    poller.metrics.maintenance(reclaim=True)
                    # A full batch may leave more; an until-idle run reclaims them first.
                    reclaim_backlog = bool(reclaimed) and reclaimed >= _RECLAIM_BATCH_SIZE
                    if until_idle is not None and reclaim_backlog:
                        reclaim_cadence.expedite()
                changed = bool(reclaimed)
            if fenced_monitor is not None:
                _, settled_fenced = await fenced_cadence.run_if_due(
                    scan_fenced_tasks,
                    now=loop.time(),
                    clock=loop.time,
                )
                changed = changed or bool(settled_fenced)
            return changed

        def maintenance_wake_at() -> float | None:
            deadlines = [
                cadence.next_run_at
                for enabled, cadence in (
                    (reclaim, reclaim_cadence),
                    (fenced_monitor is not None, fenced_cadence),
                )
                if enabled and cadence.next_run_at is not None
            ]
            return min(deadlines) if deadlines else None

        def rescan_released_handoffs() -> None:
            nonlocal handoffs_released
            if handoffs_released:
                handoffs_released = False
                rescan_continuations()

        async def maintain_until(idle: asyncio.Event) -> None:
            # Maintenance otherwise waits for the claim loop, which a handler can
            # hold for hours: a task another worker lost would stay claimed and
            # unsettled meanwhile. So it keeps its cadence while the slot is busy.
            loop = asyncio.get_running_loop()
            while True:
                wake_at = maintenance_wake_at()
                assert wake_at is not None  # maintained_while_busy runs only with a cadence
                due_in = max(wake_at - loop.time(), 0.0)
                if await wait_or_stop(due_in, idle) or slot_should_stop():
                    return
                try:
                    await maintain_expired_claims()
                except Exception:
                    logger.exception(
                        "Reclaiming expired task claims failed while a handler was running; "
                        "retrying in %g seconds.",
                        reclaim_every_s,
                    )

        @contextlib.asynccontextmanager
        async def maintained_while_busy() -> AsyncIterator[None]:
            if not owns_maintenance or (not reclaim and fenced_monitor is None):
                yield
                return
            idle = asyncio.Event()
            maintenance = asyncio.create_task(maintain_until(idle))
            try:
                yield
            except BaseException as failure:
                await _stop_busy_slot_maintenance(maintenance, idle, failure)
                raise
            await _stop_busy_slot_maintenance(maintenance, idle, None)

        async def run_step(_now: float, handled: int) -> DurableWorkerStep:
            nonlocal expired_handoff_after
            nonlocal continuation_after
            nonlocal continuation_scan_scanned
            nonlocal continuation_scan_rejected
            nonlocal continuation_scan_filtered
            nonlocal next_interrupted_handoff_recovery_at
            nonlocal next_interrupted_continuation_scan_at

            loop = asyncio.get_running_loop()
            meaningful_activity = False
            if (
                owns_maintenance
                and recover_interrupted_handoffs
                and interrupted_handoff_supported
                and loop.time() >= next_interrupted_handoff_recovery_at
            ):
                poller.metrics.maintenance(recovery=True)
                recovery_page = await _recover_expired_interrupted_task_handoffs(
                    app,
                    task_store,
                    worker_id=worker_id,
                    after=expired_handoff_after,
                    limit=_INTERRUPTED_HANDOFF_RECOVERY_BATCH_SIZE,
                    stop=slot_stop,
                )
                if until_idle is not None:
                    until_idle.record_recovery_page(
                        recovery_page, first_page=expired_handoff_after is None
                    )
                if recovery_page.recovered:
                    meaningful_activity = True
                    # Recovery follows lease expiry and the scan creation time, so
                    # a recovered task may sort behind the cursor.
                    rescan_continuations()
                    if until_idle is not None:
                        # A task it settled may leave a lost wake attempt to settle,
                        # and fences to read again, before the run returns.
                        reclaim_cadence.expedite()
                        fenced_cadence.expedite()
                if recovery_page.exhausted:
                    expired_handoff_after = None
                    next_interrupted_handoff_recovery_at = (
                        0.0
                        if until_idle is not None and until_idle.recovery_retry_pending
                        else loop.time() + interrupted_handoff_recovery_every_s
                    )
                else:
                    expired_handoff_after = recovery_page.next_after
                    next_interrupted_handoff_recovery_at = 0.0
                if slot_should_stop():
                    return DurableWorkerStep(stop=True, activity=meaningful_activity)
            continuation_activity = False
            handled_this_step = 0
            if (
                recovered_interrupted_task_handler is not None
                and not until_idle_skips_scan()
                and loop.time() >= next_interrupted_continuation_scan_at
            ):
                # A handoff released meanwhile is rescanned after this page's
                # cursor is recorded, at this step's maintenance.
                async with maintained_while_busy():
                    continuation_page = await _run_recovered_interrupted_task_handler(
                        app,
                        task_store,
                        recovered_interrupted_task_handler,
                        worker_id=worker_id,
                        lease_seconds=lease_seconds,
                        query=query,
                        after=continuation_after,
                        max_unproductive_handoffs=max_unproductive_interrupted_handoffs,
                        stop=slot_stop,
                        execution_timeout_s=execution_timeout_s,
                        execution_grace_s=execution_grace_s,
                        interrupt_on_cancel=interrupt_on_cancel,
                        event_budget_check=event_budget_check,
                    )
                continuation_scan_scanned += continuation_page.scanned_candidates
                continuation_scan_rejected += continuation_page.rejected_candidates
                continuation_scan_filtered += continuation_page.filtered_candidates
                continuation_activity = continuation_page.task is not None
                meaningful_activity = meaningful_activity or continuation_activity
                if continuation_page.exhausted:
                    if continuation_scan_rejected:
                        logger.warning(
                            "Interrupted-task continuation scan rejected invalid authority: "
                            "rejected_candidates=%d filtered_candidates=%d "
                            "scanned_candidates=%d",
                            continuation_scan_rejected,
                            continuation_scan_filtered,
                            continuation_scan_scanned,
                        )
                    continuation_after = None
                    if until_idle is not None:
                        until_idle.continuations_scanned = True
                    continuation_scan_scanned = 0
                    continuation_scan_rejected = 0
                    continuation_scan_filtered = 0
                    next_interrupted_continuation_scan_at = (
                        loop.time() + continuation_poll_interval_s
                    )
                else:
                    continuation_after = continuation_page.next_after
                    next_interrupted_continuation_scan_at = 0.0
                if continuation_activity:
                    handled_this_step += 1
                    should_stop = (
                        max_tasks is not None and handled + handled_this_step >= max_tasks
                    ) or slot_should_stop()
                    if should_stop or budget is not None:
                        # A shared reservation authorizes only one handled task.
                        return DurableWorkerStep(
                            handled=handled_this_step,
                            stop=should_stop,
                            continue_immediately=not should_stop,
                            activity=True,
                        )
            if slot_should_stop():
                return DurableWorkerStep(
                    handled=handled_this_step,
                    stop=True,
                    activity=meaningful_activity,
                )
            if owns_maintenance:
                meaningful_activity = await maintain_expired_claims() or meaningful_activity
                rescan_released_handoffs()
            if (
                until_idle is not None
                and owns_maintenance
                and expired_handoff_after is None
                and not reclaim_backlog
                and not until_idle.recovery_retry_pending
            ):
                until_idle.maintenance_settled = True
            observation: TaskScheduleWakeup | None = None
            observed_after_steps = 0

            async def prepare_claim() -> None:
                nonlocal observation, observed_after_steps
                # Maintenance is a database poll too. Only the admitted worker may
                # scan; otherwise an idle cohort fans out before claim admission.
                await group_maintenance.step(task_store, app._secret_redactor, now=loop.time())
                await reconcile_task_event_deadlines(task_store)
                if until_idle is not None:
                    observed_after_steps = until_idle.handled_steps
                    observation = await observe_schedule_wakeup(task_store, query)

            async def claim_next_task() -> Task | None:
                if materialized_work_contract_queue_supported:
                    claim_outcome = await capture_task_store_operation(
                        lambda: task_store.claim_task(
                            worker_id,
                            query,
                            lease_seconds=lease_seconds,
                        ),
                        operation_name="Task claim",
                        redactor=app._secret_redactor,
                        mutation_store=task_store,
                        mutation_method_name="claim_task",
                    )
                    if claim_outcome.failure is not None:
                        failure = claim_outcome.failure
                        del claim_outcome
                        raise_task_store_operation_failure(failure)
                    task = claim_outcome.result
                    del claim_outcome
                    return task
                return await task_store.claim_task(
                    worker_id,
                    query,
                    lease_seconds=lease_seconds,
                )

            claimed_by_call: list[UnstartedClaim] = []
            claim_dispatched = False

            async def recorded_claim() -> Task | None:
                nonlocal claim_dispatched
                claim_dispatched = True
                claimed_task = await claim_next_task()
                if claimed_task is not None:
                    claimed_by_call.append(UnstartedClaim.of(claimed_task))
                return claimed_task

            try:
                claim = await poller.claim(
                    recorded_claim,
                    before_claim=prepare_claim,
                    maximum_active_s=lease_seconds,
                )
            except BaseException as claim_failure:
                # A lost acknowledgement carries no authority to release a claim.
                if claim_dispatched:
                    await release_unstarted_claims(
                        task_store,
                        worker_id,
                        claim_failure,
                        claims=claimed_by_call,
                    )
                raise
            task = claim.value
            # The poller's wrapper aliases the task. Drop it before fallible task
            # handling so detached failures cannot retain a second task reference.
            del claim
            if task is None:
                if until_idle is not None and poller.last_attempted and until_idle.may_return():
                    if until_idle.claim_again(observation, observed_after_steps):
                        return DurableWorkerStep(
                            handled=handled_this_step,
                            continue_immediately=True,
                            activity=meaningful_activity,
                        )
                    infrastructure_stop.set()
                    return DurableWorkerStep(
                        handled=handled_this_step,
                        stop=True,
                        activity=meaningful_activity,
                    )
                if continuation_activity and continuation_after is not None:
                    return DurableWorkerStep(
                        handled=handled_this_step,
                        continue_immediately=True,
                        activity=True,
                    )
                wake_deadlines: list[float] = []
                # Only the cohort's admitted poller inspects durable deadlines. An
                # idle handler denied a poll turn must not create another DB poll.
                schedule_deadline = (
                    await next_schedule_wake_at(task_store, (query,))
                    if poller.last_attempted
                    else None
                )
                if (
                    recovered_interrupted_task_handler is not None
                    and not until_idle_skips_scan()
                    and continuation_after is None
                ):
                    wake_deadlines.append(next_interrupted_continuation_scan_at)
                if (
                    owns_maintenance
                    and recover_interrupted_handoffs
                    and interrupted_handoff_supported
                    and expired_handoff_after is None
                ):
                    wake_deadlines.append(next_interrupted_handoff_recovery_at)
                if owns_maintenance and reclaim and reclaim_cadence.next_run_at is not None:
                    wake_deadlines.append(reclaim_cadence.next_run_at)
                if (
                    owns_maintenance
                    and fenced_monitor is not None
                    and fenced_cadence.next_run_at is not None
                ):
                    wake_deadlines.append(fenced_cadence.next_run_at)
                return DurableWorkerStep(
                    handled=handled_this_step,
                    idle=True,
                    next_wake_at=min(wake_deadlines) if wake_deadlines else None,
                    next_claim_at=schedule_deadline,
                    activity=meaningful_activity,
                )
            claimed = UnstartedClaim.of(task)
            try:
                record_task_admission_to_claim_latency(
                    poller.metrics,
                    admitted_at=task.created_at,
                    claimed_at=datetime.now(UTC),
                )
                task = copy_task(task)
                if task.work_contract is not None:
                    task_id = task.id
                    contract = task.work_contract
                    lease_expires_at = task.lease_expires_at
                    if lease_expires_at is None:  # pragma: no cover - claim contract
                        raise TaskClaimLost("Contracted task claim has no worker lease.")
                    parking_outcome = await capture_task_store_operation(
                        lambda task_id=task_id, contract=contract, generation=task.claim_generation: (
                            task_store.hold_claimed_work_contract_task(
                                task_id,
                                worker_id=worker_id,
                                claim_generation=generation,
                                contract=contract,
                            )
                        ),
                        operation_name="Contracted task parking",
                        redactor=app._secret_redactor,
                        mutation_store=task_store,
                        mutation_method_name="hold_claimed_work_contract_task",
                    )
                    if parking_outcome.failure is not None:
                        failure = parking_outcome.failure
                        del contract, lease_expires_at, parking_outcome, task, task_id
                        raise_task_store_operation_failure(failure)
                    del contract, parking_outcome, task_id
                    return DurableWorkerStep(
                        handled=handled_this_step + 1,
                        continue_immediately=True,
                        activity=True,
                    )
                poller.metrics.handler_started()
                try:
                    async with maintained_while_busy():
                        await _handle_with_heartbeat(
                            app,
                            task_store,
                            task,
                            handler,
                            worker_id,
                            lease_seconds,
                            execution_timeout_s=execution_timeout_s,
                            execution_grace_s=execution_grace_s,
                            interrupt_on_cancel=interrupt_on_cancel,
                            event_budget_check=event_budget_check,
                        )
                finally:
                    poller.metrics.handler_finished()
                rescan_released_handoffs()
                next_interrupted_continuation_scan_at = 0.0
                if (
                    until_idle is not None
                    and recovered_interrupted_task_handler is not None
                    and await _may_await_continuation(app, task_store, task)
                ):
                    # The handler may have handed off its session for continuation. A
                    # parked handoff is not: only its own trigger, a task, continues it.
                    until_idle.continuations_pending = True
                return DurableWorkerStep(
                    handled=handled_this_step + 1,
                    continue_immediately=True,
                    activity=True,
                )
            except BaseException as claim_exit:
                # No handler owns a claim before its execution starts.
                await release_unstarted_claims(task_store, worker_id, claim_exit, claims=[claimed])
                raise

        subscribe_to_admissions = getattr(task_store, "_task_admission_wakeup", None)
        admission_wakeup = None
        try:
            admission_wakeup = (
                None
                if not callable(subscribe_to_admissions)
                else await subscribe_to_admissions((query,))
            )

            async def counted_step(now: float, handled: int) -> DurableWorkerStep:
                # Checked before any maintenance: a closing app's stores may already
                # be closed. A step already running is waited for by shutdown.
                if slot_should_stop():
                    return DurableWorkerStep(stop=True)
                if budget is not None and not budget.reserve():
                    poller.defer_step()
                    if owns_maintenance and not budget.exhausted:
                        # Another slot's handler still runs on the last reservation.
                        # This slot owns the pool's maintenance, so it keeps
                        # reclaiming and settling lost work until the pool drains.
                        await app._run_worker_step(maintain_expired_claims)
                        return DurableWorkerStep(idle=True, next_wake_at=maintenance_wake_at())
                    return DurableWorkerStep(stop=budget.exhausted, idle=not budget.exhausted)
                completed = 0
                if until_idle is not None:
                    until_idle.steps_running += 1
                try:
                    result = await app._run_worker_step(lambda: run_step(now, handled))
                    completed = result.handled
                    return result
                finally:
                    if budget is not None:
                        budget.settle(completed)
                    if until_idle is not None:
                        until_idle.steps_running -= 1
                        if completed:
                            until_idle.handled_steps += 1

            return await run_durable_worker_loop(
                counted_step,
                poll_interval_s=poll_interval_s,
                stop=slot_stop,
                max_handled=max_tasks if budget is None else None,
                wait=(
                    _wait_or_stop if admission_wakeup is None else admission_wakeup.wait_for_worker
                ),
                demand_policy=demand_policy,
                poller=poller,
                metrics=metrics if metrics is not None else poller.metrics,
            )
        finally:
            if admission_wakeup is not None:
                admission_wakeup.close()
            poller.close()

    # One lag sampler per worker run, kept until every slot has drained. It also
    # records the lag on the sessions it slowed.
    session_lag = app_session_event_loop_lag_recorder(app)
    lag_monitor = worker_event_loop_lag_monitor(
        metrics,
        active_sessions=app._session_control.active_session_count,
        session_lag=session_lag,
    )
    lag_monitor.start()
    try:
        if concurrency == 1:
            return await run_slot(owns_maintenance=True)
        failures: list[Exception] = []

        async def drain_slot(*, owns_maintenance: bool) -> int:
            try:
                return await run_slot(owns_maintenance=owns_maintenance)
            except asyncio.CancelledError:
                infrastructure_stop.set()
                raise
            except Exception as exc:
                infrastructure_stop.set()
                failures.append(exc)
                return 0

        async def forward_stop() -> None:
            assert stop is not None
            await stop.wait()
            infrastructure_stop.set()

        try:
            async with asyncio.TaskGroup() as group:
                stop_watcher = group.create_task(forward_stop()) if stop is not None else None
                slots = [
                    group.create_task(drain_slot(owns_maintenance=index == 0))
                    for index in range(
                        min(concurrency, max_tasks) if max_tasks is not None else concurrency
                    )
                ]
                try:
                    await asyncio.wait(slots)
                finally:
                    if stop_watcher is not None:
                        stop_watcher.cancel()
        except BaseException as failure:
            for recorded in failures:
                _attach_task_worker_secondary_failure(
                    failure, recorded, label="Task-worker slot failures during shutdown"
                )
            raise
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise ExceptionGroup("Task-worker slot failures", failures)
        return sum(slot.result() for slot in slots)
    finally:
        lag_monitor.stop()
        if session_lag is not None:
            await session_lag.aclose()


async def _settle_wake_attempts_of_ended_tasks(
    app: CayuApp, task_store: TaskStore, worker_id: str
) -> None:
    """Settle the lost wake attempts of tasks that ended; never stops the worker."""

    from cayu.tasks.session_wakes import settle_wake_attempts_of_ended_tasks

    if not app._uses_session_wakes or task_store is not app.task_store:
        return
    try:
        await settle_wake_attempts_of_ended_tasks(app, actor=_declaring_worker_actor(worker_id))
    except Exception as exc:
        logger.warning(
            "Settling the wake attempts of ended tasks failed (%s); retrying later.",
            type(exc).__name__,
        )


async def _reconcile_session_wakes(app: CayuApp, task_store: TaskStore) -> int:
    """Give open ``sleep_until`` wakes a live attempt and release stale parked tasks.

    A task parked on an external wait that was retired, for example by an
    interruption, is released the same way.

    Returns how many parked tasks it released for the generic continuation. It
    never stops the worker over a failure.
    """

    from cayu.tasks.session_wakes import reconcile_parked_handoffs, reconcile_session_wakes

    if (
        not app._uses_session_wakes and app._external_wait_hosting is None
    ) or task_store is not app.task_store:
        return 0
    released = 0
    reconciles = (reconcile_session_wakes,) if app._uses_session_wakes else ()
    for reconcile in (*reconciles, reconcile_parked_handoffs):
        try:
            count = await reconcile(app)
        except Exception as exc:
            logger.warning(
                "Reconciling session wakes failed (%s); retrying later.", type(exc).__name__
            )
            continue
        if reconcile is reconcile_parked_handoffs:
            released = count
    return released


async def _sweep_external_waits(
    app: CayuApp, task_store: TaskStore, query: TaskQuery | None
) -> None:
    """Schedule the tasks ``app.host_external_waits`` still needs; never stops the worker.

    Only a worker that claims those tasks sweeps, so a narrow worker never makes
    work it leaves to another one.
    """

    from cayu.external_wait_hosting import claims_hosted_tasks

    hosting = app._external_wait_hosting
    if hosting is None or task_store is not app.task_store or not claims_hosted_tasks(query):
        return
    try:
        await hosting.sweep()
    except Exception as exc:
        logger.warning("Servicing external waits failed (%s); retrying later.", type(exc).__name__)


async def _with_session_wakes(
    app: CayuApp, task_store: TaskStore, handler: TaskHandler
) -> TaskHandler:
    """Run ``sleep_until`` wake tasks and hosted external-wait tasks here.

    No application handler has to know about either.
    """

    from cayu.tasks.session_wakes import SESSION_WAKE_TASK_TYPE, _handle_session_wake_task

    await _reconcile_session_wakes(app, task_store)
    hosting = app._external_wait_hosting
    if hosting is not None and task_store is app.task_store:
        handler = hosting._worker_handler(handler)

    async def handle(app: CayuApp, task: Task, worker_id: str):
        if task.type == SESSION_WAKE_TASK_TYPE:
            return await _handle_session_wake_task(app, task, worker_id)
        return await handler(app, task, worker_id)

    return handle


async def _handle_with_heartbeat(
    app: CayuApp,
    task_store: TaskStore,
    task: Task,
    handler: TaskHandler,
    worker_id: str,
    lease_seconds: int,
    *,
    recover_interrupted_authority_loss: bool = False,
    execution_timeout_s: float | None = None,
    execution_grace_s: float | None = None,
    interrupt_on_cancel: bool = True,
    event_budget_check: EventBudgetCheck | None = None,
) -> None:
    settlement = TaskExecutionSettlement(task_store, app._secret_redactor)
    primary: BaseException | None = None
    try:
        await _handle_with_heartbeat_owned(
            app,
            task_store,
            task,
            handler,
            worker_id,
            lease_seconds,
            settlement=settlement,
            recover_interrupted_authority_loss=recover_interrupted_authority_loss,
            execution_timeout_s=execution_timeout_s,
            execution_grace_s=execution_grace_s,
            interrupt_on_cancel=interrupt_on_cancel,
            event_budget_check=event_budget_check,
        )
        # This worker owns the settlement it created. A write that outlives one
        # observation window is slow, not failed: keep joining it for up to one
        # task lease instead of ending the worker loop. Repeated store failures
        # (as many as one attempt allows), or a write still running after the
        # lease, transfer the retry handle to the caller through ``finish``.
        if settlement.acknowledgement_in_flight:
            logger.warning(
                "Task group execution acknowledgement is slow; waiting for the "
                "in-flight store write: task_id=%s",
                app.redact_json(task.id),
            )
            wait_until = monotonic() + lease_seconds
            while settlement.acknowledgement_in_flight and monotonic() < wait_until:
                await settlement.attempt()
    except BaseException as exc:
        primary = exc
        raise
    finally:
        settlement.finish(primary)


async def _handle_with_heartbeat_owned(
    app: CayuApp,
    task_store: TaskStore,
    task: Task,
    handler: TaskHandler,
    worker_id: str,
    lease_seconds: int,
    *,
    settlement: TaskExecutionSettlement,
    recover_interrupted_authority_loss: bool,
    execution_timeout_s: float | None,
    execution_grace_s: float | None,
    interrupt_on_cancel: bool = True,
    event_budget_check: EventBudgetCheck | None = None,
) -> None:
    # Cancellation keeps the worker's grace even when the event budget below
    # leaves the time limit none.
    cancellation_grace_s = _cancellation_grace_seconds(
        interrupt_on_cancel=interrupt_on_cancel,
        execution_timeout_s=execution_timeout_s,
        execution_grace_s=execution_grace_s,
    )
    # The task timeout may be shorter, but it can never extend its event.
    event_key = task.invocation.event_retry_key
    event_limit = False
    event_deadline = None
    event_dispatch_deadline = None
    event_state = None
    if event_key is not None and task_store.supports_task_event_execution:
        observed_at = monotonic()
        event_state = await task_store.load_task_retry_key(event_key)
        if event_state is None:
            raise RuntimeError("Task event association has no durable budget record.")
        if event_budget_check is not None:
            event_budget_check.check(app, task, event_state)
        event_deadline = ExecutionDeadline(
            expires_at=event_state.elapsed_deadline, source="task_event"
        )
        remaining = await task_store.task_event_remaining_seconds(event_key)
        if remaining is not None:
            remaining = max(remaining - (monotonic() - observed_at), 0.0)
            event_dispatch_deadline = monotonic() + remaining
            if execution_timeout_s is None or remaining <= execution_timeout_s:
                # HandlerTimeLimit cancels the owned handler context; the existing
                # quiescence protocol still retains ownership through cleanup.
                execution_timeout_s = max(remaining, math.nextafter(0.0, 1.0))
                execution_grace_s = 0.0
                event_limit = True
    time_limit = (
        None
        if execution_timeout_s is None
        else HandlerTimeLimit(
            execution_timeout_s,
            grace_s=execution_grace_s,
            source="task_event" if event_limit else "task_worker",
        )
    )
    # Budget reads precede execution entry. Slow or failed reads leave a
    # provably undispatched claim that ordinary lease reclamation can retry.
    (
        task,
        claim_deadline_monotonic,
        retry_deadline_elapsed,
        cancellation_requested,
    ) = await _renew_task_claim_before_dispatch(
        task_store,
        task,
        worker_id,
        lease_seconds,
        settlement=settlement,
    )
    owner_task = asyncio.current_task()
    if owner_task is None:  # pragma: no cover - coroutine execution invariant
        raise RuntimeError("Task worker handler execution requires an owning task.")
    cancellation_baseline = owner_task.cancelling()
    stop_heartbeat = asyncio.Event()
    lease_authority = _TaskLeaseAuthority(
        lease_expires_at=_require_task_lease_expires_at(task),
        handoff_id=task.interrupted_handoff_id,
        task_id=task.id,
        worker_id=worker_id,
        # Fixed for the callback's lifetime: the claim this worker itself received.
        claim_generation=task.claim_generation,
        deadline_monotonic=claim_deadline_monotonic,
    )
    heartbeat_task = asyncio.create_task(
        _heartbeat_until(
            task_store,
            task.id,
            worker_id,
            lease_seconds,
            stop_heartbeat,
            lease_authority=lease_authority,
            claim_deadline_monotonic=claim_deadline_monotonic,
            enforce_retry_deadline=(
                task.retry_series is not None and task.retry_series.elapsed_deadline is not None
            ),
        )
    )
    handler_error: Exception | None = None
    pending_runtime_task_failure: RuntimeTaskFailureTerminalizationPending | None = None
    handler_outcome: TaskHandlerOutcome | TaskRetryAttemptReport | None = None
    interrupted_authority_loss = False
    retry_elapsed = retry_deadline_elapsed
    retry_cancellation_requested = cancellation_requested
    retry_elapsed_cancellation: asyncio.CancelledError | None = None
    retry_terminal_report: TaskRetryAttemptReport | None = None
    disposition_cancellation: asyncio.CancelledError | None = None
    heartbeat_authority_failure: Exception | None = None
    ownership_settled = False
    pending_failure: BaseException | None = None
    try:
        if not retry_elapsed and not retry_cancellation_requested:
            try:
                handler_outcome = await _await_task_handler(
                    app,
                    task_store,
                    task,
                    handler,
                    worker_id,
                    lease_seconds,
                    cancellation_baseline=cancellation_baseline,
                    heartbeat_task=heartbeat_task,
                    stop_heartbeat=stop_heartbeat,
                    lease_authority=lease_authority,
                    settlement=settlement,
                    time_limit=time_limit,
                    event_deadline=event_deadline,
                    event_dispatch_deadline=event_dispatch_deadline,
                    cancellation_grace_s=cancellation_grace_s,
                )
            except _TaskRetryElapsed as exc:
                retry_elapsed = True
                retry_elapsed_cancellation = exc.owner_cancellation
                retry_terminal_report = exc.report
            except _TaskRetryCancellationRequested as exc:
                retry_cancellation_requested = True
                retry_elapsed_cancellation = exc.owner_cancellation
                retry_terminal_report = exc.report
            except Exception as exc:  # a single bad task must not stop the worker
                if heartbeat_task.done() and not heartbeat_task.cancelled():
                    try:
                        heartbeat_task.result()
                    except Exception as heartbeat_failure:
                        heartbeat_authority_failure = heartbeat_failure
                        if exc is heartbeat_failure:
                            raise
                        raise BaseExceptionGroup(
                            "Task handler shutdown failed after heartbeat authority loss.",
                            [heartbeat_failure, exc],
                        ) from None
                if recover_interrupted_authority_loss and isinstance(
                    exc,
                    (SessionRunFenced, TaskClaimLost),
                ):
                    interrupted_authority_loss = True
                else:
                    pending_runtime_task_failure = (
                        runtime_task_failure_terminalization_pending_from_error(exc)
                    )
                    handler_error = exc

        if retry_elapsed or retry_cancellation_requested:
            terminalization_label = "cancellation" if retry_cancellation_requested else "deadline"
            if retry_cancellation_requested and task.retry_series is None:
                # The heartbeat may still be renewing (cancellation can win before dispatch,
                # with no handler to stop it); the write presents the claim's generation,
                # which renewals keep.
                current = await task_store.load_task(task.id)
                if current is None:
                    raise RuntimeError(
                        "Task cancellation lost its store-authoritative task after handler "
                        "quiescence."
                    )
                request = _task_cancellation_terminalization_request(
                    current,
                    worker_id=worker_id,
                    claim_generation=lease_authority.claim_generation,
                )
                if request is None:
                    raise RuntimeError(
                        "Task cancellation lost its durable request after handler quiescence."
                    )
                await _terminalize_claimed_task_or_detect_peer_winner(
                    task_store,
                    request,
                )
                ownership_settled = True
                if retry_elapsed_cancellation is not None:
                    raise retry_elapsed_cancellation
                return
            try:
                (
                    receipt,
                    retry_elapsed_cancellation,
                ) = (
                    await _settle_task_retry_cancellation_resisting_cancellation(
                        task_store,
                        task.id,
                        worker_id,
                        cancellation_baseline=cancellation_baseline,
                        owner_cancellation=retry_elapsed_cancellation,
                        report=retry_terminal_report,
                        lease_authority=lease_authority,
                    )
                    if retry_cancellation_requested
                    else await _enforce_task_retry_deadline_resisting_cancellation(
                        task_store,
                        task,
                        worker_id,
                        lease_authority=lease_authority,
                        cancellation_baseline=cancellation_baseline,
                        owner_cancellation=retry_elapsed_cancellation,
                        report=retry_terminal_report,
                    )
                )
            except Exception as terminalization_failure:
                if retry_elapsed_cancellation is not None:
                    existing_cause = retry_elapsed_cancellation.__cause__
                    secondary_failure = (
                        terminalization_failure
                        if existing_cause is None
                        else BaseExceptionGroup(
                            f"Task retry {terminalization_label} cleanup failures",
                            [existing_cause, terminalization_failure],
                        )
                    )
                    raise retry_elapsed_cancellation from secondary_failure
                raise
            if receipt is None:
                missing_receipt = RuntimeError(
                    f"Task retry {terminalization_label} lost its store-authoritative "
                    "terminal evidence after handler quiescence."
                )
                if retry_elapsed_cancellation is not None:
                    raise retry_elapsed_cancellation from missing_receipt
                raise missing_receipt
            ownership_settled = True
            if retry_elapsed_cancellation is not None:
                raise retry_elapsed_cancellation
            return

        if interrupted_authority_loss:
            ownership_settled = await _settle_recovered_continuation_authority_loss(
                app,
                task_store,
                lease_authority,
            )
            return

        async def finalize_handler_outcome() -> None:
            nonlocal handler_error

            if handler_error is not None:
                # Only this worker's own closing app counts as shutdown: a refusal
                # from another, closed app is an ordinary handler failure.
                if (
                    isinstance(handler_error, ApplicationAdmissionsSealed)
                    and app.lifecycle_state != "open"
                    and await _settle_task_refused_by_shutdown(
                        task_store,
                        task.id,
                        worker_id,
                        lease_authority,
                    )
                ):
                    handler_error = None
                    return
                if pending_runtime_task_failure is not None:
                    await _settle_pending_runtime_task_failure_owner(
                        app,
                        task_store,
                        task_id=task.id,
                        worker_id=worker_id,
                        lease_authority=lease_authority,
                        pending=pending_runtime_task_failure,
                    )
                    handler_error = None
                    return
                if time_limit is not None and time_limit.expired:
                    fail = _fail_task_for_handler_time_limit(
                        app,
                        task_store,
                        task,
                        worker_id,
                        time_limit,
                        handler_error,
                        lease_authority,
                        event_state=event_state,
                    )
                    handler_error = None
                    await fail
                    return
                failure_payload = _task_failure_payload(app, handler_error)
                handler_error = None
                if event_state is not None and failure_payload.get("code") == (
                    "event_elapsed_exhausted"
                ):
                    with_event_budget_expiry(failure_payload, task, event_state)
                await _safe_fail_payload(
                    app,
                    task_store,
                    task.id,
                    worker_id,
                    failure_payload,
                    handoff_id=task.interrupted_handoff_id,
                    lease_expires_at=lease_authority.lease_expires_at,
                    lease_authority=lease_authority,
                )
            elif type(handler_outcome) is TaskRetryAttemptReport:
                if task.retry_series is None:
                    await _safe_fail(
                        app,
                        task_store,
                        task.id,
                        worker_id,
                        TypeError("TaskRetryAttemptReport requires a task retry policy."),
                        lease_expires_at=lease_authority.lease_expires_at,
                        lease_authority=lease_authority,
                    )
                else:
                    await _settle_retry_attempt_report(
                        task_store,
                        task,
                        worker_id,
                        handler_outcome,
                        lease_authority=lease_authority,
                    )
            elif handler_outcome is TaskHandlerOutcome.SESSION_INTERRUPTED:
                await _handoff_interrupted_session(app, task_store, lease_authority)
            elif handler_outcome is None and time_limit is not None and time_limit.expired:
                # Returning after the limit without settling is still a timeout.
                await _fail_task_for_handler_time_limit(
                    app,
                    task_store,
                    task,
                    worker_id,
                    time_limit,
                    None,
                    lease_authority,
                    event_state=event_state,
                )
            elif handler_outcome is not None:
                await _safe_fail(
                    app,
                    task_store,
                    task.id,
                    worker_id,
                    TypeError(f"Unsupported task handler outcome: {handler_outcome!r}."),
                    lease_expires_at=lease_authority.lease_expires_at,
                    lease_authority=lease_authority,
                )
            else:
                await _safe_fail_unfinished(
                    app,
                    task_store,
                    task.id,
                    worker_id,
                    lease_expires_at=lease_authority.lease_expires_at,
                    lease_authority=lease_authority,
                )

        if task.retry_series is None:
            if not await _dispose_within_deadline(
                app, task, lease_authority, lease_seconds, finalize_handler_outcome()
            ):
                return
        else:
            (
                _receipt,
                disposition_cancellation,
            ) = await _complete_task_retry_terminalization_resisting_cancellation(
                finalize_handler_outcome(),
                label="attempt disposition",
                cancellation_baseline=cancellation_baseline,
                owner_cancellation=None,
            )
        ownership_settled = True
        if task.retry_series is not None and disposition_cancellation is not None:
            raise disposition_cancellation
    except BaseException as exc:
        pending_failure = exc
        raise
    finally:
        stop_heartbeat.set()
        try:
            if heartbeat_task.done():
                heartbeat_task.result()
            else:
                await heartbeat_task
        except Exception as heartbeat_failure:
            if heartbeat_authority_failure is not None:
                if heartbeat_failure is not heartbeat_authority_failure:
                    _attach_task_worker_secondary_failure(
                        pending_failure,
                        heartbeat_failure,
                        label="Task heartbeat shutdown failures",
                    )
            elif ownership_settled:
                _attach_task_worker_secondary_failure(
                    pending_failure,
                    heartbeat_failure,
                    label="Task disposition heartbeat failures",
                )
            elif pending_failure is not None:
                _attach_task_worker_secondary_failure(
                    pending_failure,
                    heartbeat_failure,
                    label="Task worker failures",
                )
            else:
                raise
            if isinstance(pending_failure, asyncio.CancelledError):
                cancellation_cause = exception_cause(pending_failure)
                if cancellation_cause is not None:
                    raise pending_failure from cancellation_cause


async def _renew_task_claim_before_dispatch(
    task_store: TaskStore,
    task: Task,
    worker_id: str,
    lease_seconds: int,
    *,
    settlement: TaskExecutionSettlement,
) -> tuple[Task, float, bool, bool]:
    """Prove that an acknowledged lease can still authorize handler dispatch."""

    renewal_started = monotonic()
    renewed = await task_store.heartbeat(
        task.id,
        worker_id,
        claim_generation=task.claim_generation,
        extend_seconds=lease_seconds,
    )
    claim_deadline_monotonic = _conservative_task_lease_deadline(
        renewal_started,
        lease_seconds,
    )
    if monotonic() >= claim_deadline_monotonic:
        raise TaskClaimLost(
            "Task heartbeat acknowledgement consumed the renewed lease before dispatch."
        )
    retry_deadline_elapsed = False
    if renewed.retry_series is not None and renewed.retry_series.elapsed_deadline is not None:
        retry_deadline_elapsed = await task_store.task_retry_deadline_elapsed(
            renewed.id,
            worker_id,
            claim_generation=task.claim_generation,
        )
        if monotonic() >= claim_deadline_monotonic:
            raise TaskClaimLost(
                "Task retry-deadline validation consumed the renewed lease before dispatch."
            )
        if not retry_deadline_elapsed:
            # The deadline query is a separate store operation. Cancellation can
            # commit while it is awaiting, so its earlier heartbeat result is no
            # longer sufficient admission evidence. Linearize dispatch through a
            # final renewal and inspect the returned authoritative task snapshot.
            renewal_started = monotonic()
            renewed = await task_store.heartbeat(
                renewed.id,
                worker_id,
                claim_generation=task.claim_generation,
                extend_seconds=lease_seconds,
            )
            claim_deadline_monotonic = _conservative_task_lease_deadline(
                renewal_started,
                lease_seconds,
            )
            if monotonic() >= claim_deadline_monotonic:
                raise TaskClaimLost(
                    "Task post-deadline heartbeat acknowledgement consumed the renewed "
                    "lease before dispatch."
                )
    if not retry_deadline_elapsed:
        renewed = await enforce_task_event_deadline(task_store, renewed)
        # Continuations skip execution entry, so cannot rely on its lease check.
        if monotonic() >= claim_deadline_monotonic:
            raise TaskClaimLost(
                "Task event-deadline inspection consumed the renewed lease before dispatch."
            )
    cancellation_requested = _task_retry_cancellation_requested(
        renewed
    ) or _task_cancellation_requested(renewed)
    if (
        not retry_deadline_elapsed
        and not cancellation_requested
        and renewed.interrupted_handoff_id is None
    ):
        if renewed.lease_expires_at is None:  # pragma: no cover - heartbeat contract
            raise TaskClaimLost("Task heartbeat returned no worker lease before dispatch.")
        renewed = await settlement.enter(renewed)
        if monotonic() >= claim_deadline_monotonic:
            # No callback has been dispatched. Retain the exact start receipt
            # before cancellation mutates or clears its worker authority.
            settlement.observe(renewed)
            await settlement.attempt()
            renewed = await task_store.request_claimed_task_cancellation(
                renewed.id,
                worker_id,
                error={"code": "task_execution_start_acknowledgement_consumed_lease"},
                claim_generation=task.claim_generation,
            )
            cancellation_requested = True
    return (
        copy_task(renewed),
        claim_deadline_monotonic,
        retry_deadline_elapsed,
        cancellation_requested,
    )


def _conservative_task_lease_deadline(started: float, lease_seconds: int) -> float:
    """Retain one-third of the durable lease for a fail-closed drain handoff."""

    return started + (lease_seconds * 2 / 3)


def _require_task_lease_expires_at(task: Task) -> datetime:
    if task.lease_expires_at is None:
        raise TaskClaimLost("Task claim has no worker lease.")
    return task.lease_expires_at


async def _await_task_handler(
    app: CayuApp,
    task_store: TaskStore,
    task: Task,
    handler: TaskHandler,
    worker_id: str,
    lease_seconds: int,
    *,
    cancellation_baseline: int,
    heartbeat_task: asyncio.Task[_TaskHeartbeatOutcome],
    stop_heartbeat: asyncio.Event,
    lease_authority: _TaskLeaseAuthority,
    settlement: TaskExecutionSettlement,
    time_limit: HandlerTimeLimit | None,
    event_deadline: ExecutionDeadline | None,
    event_dispatch_deadline: float | None,
    cancellation_grace_s: float | None = None,
) -> TaskHandlerOutcome | TaskRetryAttemptReport | None:
    # The callback may mutate its view. Settlement must remain bound to the
    # exact execution entered by this worker, not callback-authored authority.
    handler_view = copy_task(task)
    settlement_authority = copy_task(task)

    callback_done: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    # A cancellation request reaches the handler through this signal, then by
    # cancelling it after the grace period unless the worker opted out. A
    # callback that already returned (its acknowledgement may still be in
    # flight) or that its time limit already cancelled is not cancelled again.
    cancellation = _TaskCancellationDelivery(
        signal=TaskCancellationSignal(),
        interrupt_after_s=cancellation_grace_s,
        may_interrupt=lambda: (
            not callback_done.done() and (time_limit is None or not time_limit.cancelled)
        ),
    )

    async def invoke_handler() -> TaskHandlerOutcome | TaskRetryAttemptReport | None:
        from cayu.deadlines import task_event_deadline_scope
        from cayu.resource_access import execution_access

        if task.invocation.access_labels and task.invocation.resource_access is None:
            from cayu._resource_access_errors import ResourceAccessDenied

            raise ResourceAccessDenied()
        if event_dispatch_deadline is not None and time_limit is not None:
            # Task scheduling and access admission consume the existing budget;
            # entering the cancellation owner must not start that duration again.
            remaining = max(event_dispatch_deadline - monotonic(), math.nextafter(0.0, 1.0))
            if remaining <= time_limit.timeout_s:
                time_limit.timeout_s = remaining
                time_limit.grace_s = 0.0
                time_limit.source = "task_event"
        # Fixed at handler start, so retries within this call share it. The run
        # default comes from the entered limit, so the two cannot disagree.
        async with (
            contextlib.nullcontext() if time_limit is None else time_limit,
            contextlib.nullcontext()
            if event_deadline is None
            else task_event_deadline_scope(
                event_deadline, event_id=task.invocation.event_retry_key
            ),
            execution_access(
                task.invocation.resource_access,
                app.resource_access_policy,
                task.invocation.access_labels,
                kind="tasks",
            ),
        ):
            with (
                cancellation.signal.bind(),
                default_run_deadline(
                    ExecutionDeadline.after(None, source="task_worker")
                    if time_limit is None
                    else (
                        event_deadline
                        if time_limit.source == "task_event" and event_deadline is not None
                        else time_limit.run_deadline()
                    )
                ),
            ):
                # Entry acknowledgements and access admission can exhaust the event.
                # A scheduled timeout cannot prevent the callback's initial effects.
                if event_dispatch_deadline is not None and monotonic() >= event_dispatch_deadline:
                    assert event_deadline is not None
                    raise ExecutionDeadlineExceeded(event_deadline, "handler")
                result = await handler(app, handler_view, worker_id)
        return result

    async def owned_handler() -> TaskHandlerOutcome | TaskRetryAttemptReport | None:
        try:
            result = await invoke_handler()
        except BaseException as handler_failure:
            callback_done.set_result(None)
            # Raising also proves the callback stopped: settle its execution.
            await acknowledge_stopped_handler(settlement, settlement_authority, handler_failure)
            raise
        finally:
            if not callback_done.done():
                callback_done.set_result(None)
        settlement.observe(settlement_authority, result)
        # Storage acknowledgement is not part of the callback outcome. Ordinary
        # settlement failures stay owned until after result disposition.
        await settlement.attempt()
        return result

    async def settle_ordinary_cancellation(
        owner_cancellation: asyncio.CancelledError,
        *,
        code: str = "task_worker_owner_cancelled",
    ) -> None:
        owner_task = asyncio.current_task()
        assert owner_task is not None
        # Only a fresh root attempt can be requeued. A continuation of an
        # interrupted session keeps its handoff and settles like any other.
        if (
            task.restartable
            and task.interrupted_handoff_id is None
            and owner_task.cancelling() > cancellation_baseline
        ):
            await _restart_task_handler_after_shutdown(
                task_store,
                task,
                worker_id,
                handler_task,
                heartbeat_task,
                stop_heartbeat,
                lease_authority=lease_authority,
                lease_seconds=lease_seconds,
                owner_cancellation=owner_cancellation,
            )
            return
        await _drain_ordinary_task_handler_under_cancellation_fence(
            task_store,
            task,
            worker_id,
            handler_task,
            heartbeat_task,
            stop_heartbeat,
            lease_authority=lease_authority,
            lease_seconds=lease_seconds,
            cancellation_error={"code": code},
            primary_failure=owner_cancellation,
            owner_cancellation=owner_cancellation,
        )

    with bind_task_lease_authority(lease_authority):
        handler_task = asyncio.ensure_future(owned_handler())
    if time_limit is not None:
        # A task cancelled before its coroutine starts never enters the finally block.
        handler_task.add_done_callback(
            lambda _: callback_done.done() or callback_done.set_result(None)
        )
        # Retry disposition can finish while the fence is still writing. Count
        # the watcher separately so shutdown retains its store until it settles.
        _retain_handler_time_limit_watcher(
            asyncio.create_task(
                app._run_worker_step(
                    lambda: _fence_handler_that_outlives_its_time_limit(
                        app,
                        task_store,
                        task,
                        worker_id,
                        callback_done,
                        time_limit,
                        lease_authority=lease_authority,
                    )
                )
            )
        )
    try:
        completed, _pending = await asyncio.wait(
            (handler_task, heartbeat_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        heartbeat_result: _TaskHeartbeatOutcome | None = None
        if heartbeat_task.done():
            try:
                heartbeat_result = heartbeat_task.result()
            except Exception as heartbeat_failure:
                if task.retry_series is None:
                    await _drain_ordinary_task_handler_under_cancellation_fence(
                        task_store,
                        task,
                        worker_id,
                        handler_task,
                        heartbeat_task,
                        stop_heartbeat,
                        lease_authority=lease_authority,
                        lease_seconds=lease_seconds,
                        cancellation_error={"code": "task_worker_lease_authority_lost"},
                        primary_failure=heartbeat_failure,
                    )
                else:
                    await _drain_retry_task_handler_under_cancellation_fence(
                        task_store,
                        task,
                        worker_id,
                        handler_task,
                        heartbeat_task,
                        stop_heartbeat,
                        lease_authority=lease_authority,
                        lease_seconds=lease_seconds,
                        cancellation_baseline=cancellation_baseline,
                        primary_failure=heartbeat_failure,
                    )
                raise
        if handler_task in completed and heartbeat_result not in {
            _TaskHeartbeatOutcome.ELAPSED,
            _TaskHeartbeatOutcome.CANCELLATION_REQUESTED,
        }:
            # Heartbeat last: the claim keeps renewing until the disposition has
            # committed, and the owner stops the heartbeat after it. An ordinary
            # task's disposition already accounts for a requested cancellation,
            # so its heartbeat no longer stops to report one.
            if task.retry_series is None:
                lease_authority.keep_renewing_on_cancellation = True
            return await handler_task
        if heartbeat_result is _TaskHeartbeatOutcome.ELAPSED:
            quiescence = await _quiesce_task_retry_handler_before_terminalization(
                task_store,
                task.id,
                worker_id,
                lease_seconds,
                handler_task,
                cancellation_baseline,
                lease_authority,
            )
            raise _TaskRetryElapsed(quiescence.owner_cancellation, quiescence.report)
        if heartbeat_result is _TaskHeartbeatOutcome.CANCELLATION_REQUESTED:
            quiescence = await _quiesce_task_retry_handler_before_terminalization(
                task_store,
                task.id,
                worker_id,
                lease_seconds,
                handler_task,
                cancellation_baseline,
                lease_authority,
                cancellation=cancellation,
            )
            raise _TaskRetryCancellationRequested(
                quiescence.owner_cancellation,
                quiescence.report,
            )
        if heartbeat_result is _TaskHeartbeatOutcome.TERMINAL:
            return await handler_task
        final_heartbeat_result = await heartbeat_task
        if final_heartbeat_result is _TaskHeartbeatOutcome.ELAPSED:
            quiescence = await _quiesce_task_retry_handler_before_terminalization(
                task_store,
                task.id,
                worker_id,
                lease_seconds,
                handler_task,
                cancellation_baseline,
                lease_authority,
            )
            raise _TaskRetryElapsed(quiescence.owner_cancellation, quiescence.report)
        if final_heartbeat_result is _TaskHeartbeatOutcome.CANCELLATION_REQUESTED:
            quiescence = await _quiesce_task_retry_handler_before_terminalization(
                task_store,
                task.id,
                worker_id,
                lease_seconds,
                handler_task,
                cancellation_baseline,
                lease_authority,
                cancellation=cancellation,
            )
            raise _TaskRetryCancellationRequested(
                quiescence.owner_cancellation,
                quiescence.report,
            )
        return await handler_task
    except BaseExceptionGroup as grouped_failure:
        if task.retry_series is None:
            grouped_cancellation = cancellation_carried_by(
                grouped_failure, cancellation_baseline=cancellation_baseline
            )
            if grouped_cancellation is None or isinstance(grouped_failure, Exception):
                raise
            # A cancellation grouped with cleanup failures settles the claim
            # exactly as the bare cancellation would, then keeps its group.
            # Only a delivered one is the owner's; otherwise the handler's own.
            owner_task = asyncio.current_task()
            await settle_ordinary_cancellation(
                grouped_cancellation,
                code=(
                    "task_worker_owner_cancelled"
                    if owner_task is not None and owner_task.cancelling() > cancellation_baseline
                    else "task_handler_cancelled"
                ),
            )
            raise
        if exception_tree_contains(grouped_failure, (KeyboardInterrupt, SystemExit, GeneratorExit)):
            raise
        owner_task = asyncio.current_task()
        if owner_task is None:  # pragma: no cover - coroutine execution invariant
            raise RuntimeError(
                "Task retry cancellation classification requires an owning task."
            ) from grouped_failure
        if owner_task.cancelling() > cancellation_baseline:
            owner_cancellation = next(
                (
                    candidate
                    for candidate in iter_exception_tree(grouped_failure)
                    if isinstance(candidate, asyncio.CancelledError)
                ),
                None,
            )
            if owner_cancellation is None:
                cancel_message = getattr(owner_task, "_cancel_message", None)
                owner_cancellation = (
                    asyncio.CancelledError()
                    if cancel_message is None
                    else asyncio.CancelledError(cancel_message)
                )
            await _raise_task_retry_owner_cancellation(
                task_store,
                task,
                worker_id,
                lease_seconds,
                handler_task,
                heartbeat_task,
                stop_heartbeat,
                lease_authority,
                cancellation_baseline=cancellation_baseline,
                owner_cancellation=owner_cancellation,
            )
            raise AssertionError(
                "Task retry cancellation handling returned unexpectedly."
            ) from None
        if exception_tree_contains(grouped_failure, asyncio.CancelledError):
            raise RuntimeError(
                "Task retry handler was unexpectedly cancelled without cancellation "
                "of its owning worker."
            ) from grouped_failure
        raise
    except asyncio.CancelledError as owner_cancellation:
        owner_task = asyncio.current_task()
        if owner_task is None:  # pragma: no cover - coroutine execution invariant
            raise RuntimeError(
                "Task retry cancellation classification requires an owning task."
            ) from owner_cancellation
        if task.retry_series is not None and owner_task.cancelling() <= cancellation_baseline:
            raise RuntimeError(
                "Task retry handler was unexpectedly cancelled without cancellation "
                "of its owning worker."
            ) from owner_cancellation
        if task.retry_series is not None:
            await _raise_task_retry_owner_cancellation(
                task_store,
                task,
                worker_id,
                lease_seconds,
                handler_task,
                heartbeat_task,
                stop_heartbeat,
                lease_authority,
                cancellation_baseline=cancellation_baseline,
                owner_cancellation=owner_cancellation,
            )
            raise AssertionError(
                "Task retry cancellation handling returned unexpectedly."
            ) from None
        await settle_ordinary_cancellation(owner_cancellation)
        raise owner_cancellation from exception_cause(owner_cancellation)


_HANDLER_TIME_LIMIT_WATCHERS: set[asyncio.Task[None]] = set()


def _retain_handler_time_limit_watcher(watcher: asyncio.Task[None]) -> None:
    _HANDLER_TIME_LIMIT_WATCHERS.add(watcher)
    watcher.add_done_callback(_HANDLER_TIME_LIMIT_WATCHERS.discard)


async def _wait_for_handler_time_limit_expiry(
    callback_done: asyncio.Future[None],
    time_limit: HandlerTimeLimit,
) -> bool:
    """Wait until the limit passes; return False if the handler finished first."""

    while not callback_done.done():
        remaining = time_limit.remaining_seconds()
        if remaining == 0:
            return True
        # The handler may not have entered its limit yet. Recheck after the
        # soft interval, since the grace period is inside the total timeout.
        timeout = max(time_limit.soft_timeout_s, 0.01) if remaining is None else remaining
        await asyncio.wait((callback_done,), timeout=timeout)
    return False


async def _fail_task_for_handler_time_limit(
    app: CayuApp,
    task_store: TaskStore,
    task: Task,
    worker_id: str,
    time_limit: HandlerTimeLimit,
    handler_error: Exception | None,
    lease_authority: _TaskLeaseAuthority,
    *,
    event_state: TaskRetryKeyState | None = None,
) -> None:
    """Fail a task whose handler outlived its time limit; log it if the failure landed."""

    payload = _handler_time_limit_failure_payload(app, time_limit, handler_error)
    note = ""
    if time_limit.source == "task_event":
        payload["code"] = "event_elapsed_exhausted"
        note = " source=task_event"
        if event_state is not None:
            with_event_budget_expiry(payload, task, event_state)
            note = f"{note} {payload['message']}"
    del handler_error  # Not held across the store await.
    if not await _safe_fail_payload(
        app,
        task_store,
        task.id,
        worker_id,
        payload,
        lease_expires_at=lease_authority.lease_expires_at,
        lease_authority=lease_authority,
        retryable=True,
        handoff_id=task.interrupted_handoff_id,
    ):
        return
    logger.warning(
        "Task failed because its handler exceeded its time limit: task_id=%s timeout_s=%s "
        "grace_s=%s handler_cancelled=%s session_ids=%s%s",
        app.redact_json(task.id),
        time_limit.timeout_s,
        time_limit.grace_s,
        time_limit.cancelled,
        app.redact_json(list(time_limit.session_ids)),
        note,
    )


async def _fence_handler_that_outlives_its_time_limit(
    app: CayuApp,
    task_store: TaskStore,
    task: Task,
    worker_id: str,
    callback_done: asyncio.Future[None],
    time_limit: HandlerTimeLimit,
    *,
    lease_authority: _TaskLeaseAuthority,
) -> None:
    """Make a handler that ignores its time-limit cancellation durably visible.

    The worker keeps renewing the lease while the handler runs, so no other
    worker can take the task. The cancellation request shows why it is still
    claimed and settles the task as cancelled once the handler returns.
    """

    if not await _wait_for_handler_time_limit_expiry(callback_done, time_limit):
        return
    logger.warning(
        "Task handler passed its time limit; new runs are refused and the handler is "
        "cancelled after the grace period: task_id=%s timeout_s=%s grace_s=%s session_ids=%s",
        app.redact_json(task.id),
        time_limit.timeout_s,
        time_limit.grace_s,
        app.redact_json(list(time_limit.session_ids)),
    )
    cancelled = asyncio.ensure_future(time_limit.wait_cancelled())
    try:
        await asyncio.wait((callback_done, cancelled), return_when=asyncio.FIRST_COMPLETED)
    finally:
        cancelled.cancel()
    if callback_done.done():
        return
    await asyncio.wait(
        (callback_done,),
        timeout=max(time_limit.grace_s, _HANDLER_STOP_WAIT_FLOOR_SECONDS),
    )
    if callback_done.done():
        return
    logger.warning(
        "Task handler kept running after its time limit cancelled it; requesting "
        "cancellation and keeping the lease until it returns: task_id=%s",
        app.redact_json(task.id),
    )
    try:
        await _request_task_cancellation_fence(
            task_store,
            task,
            worker_id,
            app.redact_json(
                {
                    "code": _HANDLER_TIME_LIMIT_FENCE_CODE,
                    "error": "ExecutionDeadlineExceeded",
                    "message": "The task handler kept running after its time limit cancelled it.",
                    "session_ids": list(time_limit.session_ids),
                }
            ),
            lease_authority=lease_authority,
        )
    except Exception:
        logger.exception(
            "Could not record the time-limit cancellation request; the worker keeps the "
            "lease while the handler runs: task_id=%s",
            app.redact_json(task.id),
        )


def _heartbeat_during_quiescence(
    task_store: TaskStore,
    task: Task,
    worker_id: str,
    lease_seconds: int,
    heartbeat_task: asyncio.Task[_TaskHeartbeatOutcome],
    stop_heartbeat: asyncio.Event,
    *,
    lease_authority: _TaskLeaseAuthority,
) -> asyncio.Task[_TaskHeartbeatOutcome]:
    lease_authority.keep_renewing_on_cancellation = True
    if (
        heartbeat_task.done()
        and not heartbeat_task.cancelled()
        and heartbeat_task.exception() is None
        and heartbeat_task.result() is _TaskHeartbeatOutcome.CANCELLATION_REQUESTED
    ):
        # Cancellation may have been observed before the drain owner took over.
        # Continue from the last acknowledged lease, never grant a new deadline.
        return asyncio.create_task(
            _heartbeat_until(
                task_store,
                task.id,
                worker_id,
                lease_seconds,
                stop_heartbeat,
                lease_authority=lease_authority,
                claim_deadline_monotonic=lease_authority.deadline_monotonic,
            )
        )
    return heartbeat_task


async def _restart_task_handler_after_shutdown(
    task_store: TaskStore,
    task: Task,
    worker_id: str,
    handler_task: asyncio.Future[Any],
    heartbeat_task: asyncio.Task[_TaskHeartbeatOutcome],
    stop_heartbeat: asyncio.Event,
    *,
    lease_authority: _TaskLeaseAuthority,
    lease_seconds: int,
    owner_cancellation: asyncio.CancelledError,
) -> None:
    heartbeat_task = _heartbeat_during_quiescence(
        task_store,
        task,
        worker_id,
        lease_seconds,
        heartbeat_task,
        stop_heartbeat,
        lease_authority=lease_authority,
    )

    async def settle() -> None:
        stop_heartbeat.set()
        await heartbeat_task
        async with lease_authority.lock:
            current = await task_store.load_task(task.id)
            if current is None:
                raise TaskClaimLost("Restartable task disappeared during shutdown.")
            if current.status in _TERMINAL_TASK_STATUSES:
                return
            if not _task_cancellation_requested(current):
                try:
                    await task_store.restart_task(
                        task.id, worker_id, claim_generation=lease_authority.claim_generation
                    )
                    return
                except TaskTerminalizationConflict:
                    current = await task_store.load_task(task.id)
                    if (
                        current is None
                        or not _task_cancellation_requested(current)
                        or current.worker_id != worker_id
                        or current.claim_generation != lease_authority.claim_generation
                    ):
                        raise
            await _settle_ordinary_task_cancellation_after_quiescence(
                task_store,
                task.id,
                worker_id,
                claim_generation=lease_authority.claim_generation,
            )

    async def cleanup() -> None:
        # Lease renewal continues until cancellation cleanup has fully joined.
        # A handler suppressing cancellation retains ownership until it returns.
        handler_task.cancel()
        handler_failure: BaseException | None = None
        try:
            await handler_task
        except asyncio.CancelledError:
            pass
        except BaseException as error:
            handler_failure = error
        try:
            await settle()
        except BaseException as settlement_failure:
            if handler_failure is not None:
                _attach_task_worker_secondary_failure(
                    settlement_failure, handler_failure, label="Restartable handler cleanup failure"
                )
            raise
        if handler_failure is not None:
            raise handler_failure

    outcome = await await_shielded_task_outcome(
        asyncio.create_task(cleanup()), cancellation=owner_cancellation
    )
    if outcome.error is not None:
        _attach_task_worker_secondary_failure(
            owner_cancellation, outcome.error, label="Restartable task shutdown failure"
        )
    restore_task_cancellation_requests(
        outcome.cancellation_requests_consumed, cancellation=owner_cancellation
    )


async def _raise_task_retry_owner_cancellation(
    task_store: TaskStore,
    task: Task,
    worker_id: str,
    lease_seconds: int,
    handler_task: asyncio.Future[Any],
    heartbeat_task: asyncio.Task[_TaskHeartbeatOutcome],
    stop_heartbeat: asyncio.Event,
    lease_authority: _TaskLeaseAuthority,
    *,
    cancellation_baseline: int,
    owner_cancellation: asyncio.CancelledError,
) -> None:
    """Naturally drain one retry attempt before propagating owner cancellation."""

    async def cleanup() -> None:
        report: TaskRetryAttemptReport | None = None
        handler_failure: BaseException | None = None
        heartbeat_failure: BaseException | None = None
        heartbeat_outcome: _TaskHeartbeatOutcome | None = None

        completed, _pending = await asyncio.wait(
            (handler_task, heartbeat_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if heartbeat_task in completed:
            try:
                heartbeat_outcome = heartbeat_task.result()
            except BaseException as exc:
                heartbeat_failure = exc
            if (
                heartbeat_failure is None
                and heartbeat_outcome is _TaskHeartbeatOutcome.ELAPSED
                and not handler_task.done()
            ):
                quiescence = await _quiesce_task_retry_handler_before_terminalization(
                    task_store,
                    task.id,
                    worker_id,
                    lease_seconds,
                    handler_task,
                    cancellation_baseline=0,
                    lease_authority=lease_authority,
                )
                receipt = await _enforce_task_retry_deadline_with_retry(
                    task_store,
                    task,
                    worker_id,
                    lease_expires_at=lease_authority.lease_expires_at,
                    claim_generation=lease_authority.claim_generation,
                    token_count=(0 if quiescence.report is None else quiescence.report.token_count),
                    estimated_cost=(
                        Decimal(0)
                        if quiescence.report is None
                        else quiescence.report.estimated_cost
                    ),
                )
                if receipt is None:
                    raise RuntimeError(
                        "Task retry owner-cancellation drain lost its elapsed deadline receipt."
                    )
                return
            if heartbeat_failure is not None or heartbeat_outcome is (
                _TaskHeartbeatOutcome.CANCELLATION_REQUESTED
            ):
                await _drain_retry_task_handler_under_cancellation_fence(
                    task_store,
                    task,
                    worker_id,
                    handler_task,
                    heartbeat_task,
                    stop_heartbeat,
                    lease_authority=lease_authority,
                    lease_seconds=lease_seconds,
                    cancellation_baseline=cancellation_baseline,
                    cancellation_error={"code": "task_worker_owner_cancelled"},
                    primary_failure=heartbeat_failure or owner_cancellation,
                )
                if heartbeat_failure is not None:
                    raise heartbeat_failure
                return

        try:
            outcome = await handler_task
            if type(outcome) is TaskRetryAttemptReport:
                report = TaskRetryAttemptReport.model_validate(
                    outcome.model_dump(mode="python", warnings=False)
                )
        except asyncio.CancelledError as exc:
            handler_failure = unexpected_child_cancellation_error(
                exc,
                operation="Task retry handler owner-cancellation drain",
            )
        except (GeneratorExit, KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            handler_failure = exc

        stop_heartbeat.set()
        if heartbeat_outcome is None:
            try:
                heartbeat_outcome = await heartbeat_task
            except BaseException as exc:
                heartbeat_failure = exc
        if heartbeat_failure is not None or heartbeat_outcome is (
            _TaskHeartbeatOutcome.CANCELLATION_REQUESTED
        ):
            await _drain_retry_task_handler_under_cancellation_fence(
                task_store,
                task,
                worker_id,
                handler_task,
                heartbeat_task,
                stop_heartbeat,
                lease_authority=lease_authority,
                lease_seconds=lease_seconds,
                cancellation_baseline=cancellation_baseline,
                cancellation_error={"code": "task_worker_owner_cancelled"},
                primary_failure=heartbeat_failure or owner_cancellation,
            )
            if handler_failure is not None:
                _attach_task_worker_secondary_failure(
                    owner_cancellation,
                    handler_failure,
                    label="Task retry handler failures during owner cancellation",
                )
            if heartbeat_failure is not None:
                raise heartbeat_failure
            return

        if heartbeat_outcome is _TaskHeartbeatOutcome.ELAPSED:
            receipt = await _enforce_task_retry_deadline_with_retry(
                task_store,
                task,
                worker_id,
                lease_expires_at=lease_authority.lease_expires_at,
                claim_generation=lease_authority.claim_generation,
                token_count=0 if report is None else report.token_count,
                estimated_cost=(Decimal(0) if report is None else report.estimated_cost),
            )
            if receipt is None:
                raise RuntimeError(
                    "Task retry owner-cancellation drain lost its elapsed deadline receipt."
                )
            if handler_failure is not None:
                raise handler_failure
            return

        if report is not None:
            await _settle_retry_attempt_report(
                task_store,
                task,
                worker_id,
                report,
                lease_authority=lease_authority,
            )
        else:
            await task_store.release_task(
                task.id,
                worker_id,
                claim_generation=lease_authority.claim_generation,
            )
        if handler_failure is not None:
            raise handler_failure

    cleanup_failure: BaseException | None = None
    try:
        await _await_task_retry_cleanup_resisting_cancellation(
            cleanup(),
            label="owner-cancellation drain",
            owner_cancellation=owner_cancellation,
        )
    except BaseException as exc:
        cleanup_failure = exc
    if cleanup_failure is not None and exception_tree_contains(
        cleanup_failure,
        (GeneratorExit, KeyboardInterrupt, SystemExit),
    ):
        _attach_task_worker_secondary_failure(
            cleanup_failure,
            owner_cancellation,
            label="Task retry process-control and owner-cancellation evidence",
        )
        raise cleanup_failure from exception_cause(cleanup_failure)
    if cleanup_failure is not None:
        _attach_task_worker_secondary_failure(
            owner_cancellation,
            cleanup_failure,
            label="Task retry owner-cancellation drain failures",
        )
    _raise_task_retry_owner_cancellation_with_secondary(owner_cancellation, None)


def _raise_task_retry_owner_cancellation_with_secondary(
    owner_cancellation: asyncio.CancelledError,
    secondary: BaseException | None,
) -> None:
    if secondary is not None:
        _attach_task_worker_secondary_failure(
            owner_cancellation,
            secondary,
            label="Task retry cancellation failures",
        )
    cause = exception_cause(owner_cancellation)
    if cause is None:
        raise owner_cancellation from None
    raise owner_cancellation from cause


async def _stop_busy_slot_maintenance(
    maintenance: asyncio.Task[None],
    idle: asyncio.Event,
    failure: BaseException | None,
) -> None:
    """Join the maintenance a slot ran while busy, keeping the slot's own outcome.

    After a normal return a pass already under way finishes its store writes,
    unless the slot is cancelled meanwhile; after ``failure``, or that
    cancellation, the maintenance is cancelled and joined, the slot's
    cancellation requests are restored and the slot's error stays the one raised.
    """

    if failure is None:
        idle.set()
        try:
            await asyncio.wait({maintenance})
        except asyncio.CancelledError as cancellation:
            await _stop_busy_slot_maintenance(maintenance, idle, cancellation)
            raise
        # A failed pass is logged and retried, so only an unexpected error ends it.
        maintenance.result()
        return
    maintenance.cancel()
    outcome = await await_shielded_task_outcome(
        maintenance,
        cancellation=failure if isinstance(failure, asyncio.CancelledError) else None,
    )
    restore_task_cancellation_requests(
        outcome.cancellation_requests_consumed,
        cancellation=outcome.cancellation,
    )
    if outcome.error is not None and not isinstance(outcome.error, asyncio.CancelledError):
        _attach_task_worker_secondary_failure(
            failure, outcome.error, label="Busy-slot maintenance failure"
        )


def _attach_task_worker_secondary_failure(
    primary: BaseException | None,
    secondary: BaseException,
    *,
    label: str,
) -> None:
    """Retain one heartbeat failure without replacing an authoritative signal."""

    if (
        primary is None
        or primary is secondary
        or any(candidate is secondary for candidate in iter_exception_tree(primary))
    ):
        return
    existing_cause = exception_cause(primary)
    if existing_cause is None:
        set_exception_cause(primary, secondary)
        return
    if existing_cause is secondary or any(
        candidate is secondary for candidate in iter_exception_tree(existing_cause)
    ):
        return
    set_exception_cause(
        primary,
        BaseExceptionGroup(label, [existing_cause, secondary]),
    )


async def _drain_retry_task_handler_under_cancellation_fence(
    task_store: TaskStore,
    task: Task,
    worker_id: str,
    handler_task: asyncio.Future[Any],
    heartbeat_task: asyncio.Task[_TaskHeartbeatOutcome],
    stop_heartbeat: asyncio.Event,
    *,
    lease_authority: _TaskLeaseAuthority,
    lease_seconds: int,
    cancellation_baseline: int,
    cancellation_error: dict[str, Any] | None = None,
    primary_failure: BaseException,
    owner_cancellation: asyncio.CancelledError | None = None,
) -> None:
    """Fence a retry attempt before naturally draining cancellation-opaque work."""

    heartbeat_task = _heartbeat_during_quiescence(
        task_store,
        task,
        worker_id,
        lease_seconds,
        heartbeat_task,
        stop_heartbeat,
        lease_authority=lease_authority,
    )

    async def cleanup() -> None:
        failures: list[BaseException] = []
        requested: Task | None = None
        try:
            requested = await _request_task_cancellation_fence(
                task_store,
                task,
                worker_id,
                (
                    {"code": "task_worker_lease_authority_lost"}
                    if cancellation_error is None
                    else cancellation_error
                ),
                lease_authority=lease_authority,
            )
        except BaseException as exc:
            failures.append(exc)

        report: TaskRetryAttemptReport | None = None
        try:
            outcome = await handler_task
            if type(outcome) is TaskRetryAttemptReport:
                report = outcome
        except asyncio.CancelledError as exc:
            failures.append(
                unexpected_child_cancellation_error(
                    exc,
                    operation="Task retry handler drain",
                )
            )
        except BaseException as exc:
            failures.append(exc)

        stop_heartbeat.set()
        if not heartbeat_task.done():
            try:
                await heartbeat_task
            except BaseException as exc:
                failures.append(exc)

        try:
            if requested is None or not _task_retry_cancellation_requested(requested):
                raise TaskTerminalizationConflict(
                    "Task retry drain lost its durable cancellation request."
                )
            await _settle_retry_task_cancellation_after_quiescence(
                task_store,
                task.id,
                worker_id,
                cancellation_baseline=cancellation_baseline,
                report=report,
            )
        except BaseException as exc:
            failures.append(exc)

        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise BaseExceptionGroup(
                "Task retry cancellation-fence drain failures",
                failures,
            )

    cleanup_outcome = await await_shielded_task_outcome(
        asyncio.create_task(cleanup()),
        cancellation=owner_cancellation,
    )
    cleanup_failure = cleanup_outcome.error
    if cleanup_failure is not None and exception_tree_contains(
        cleanup_failure,
        (GeneratorExit, KeyboardInterrupt, SystemExit),
    ):
        _attach_task_worker_secondary_failure(
            cleanup_failure,
            primary_failure,
            label="Task retry process-control and ownership-loss evidence",
        )
        raise cleanup_failure from exception_cause(cleanup_failure)

    delivered_cancellation = cleanup_outcome.cancellation
    if delivered_cancellation is not None:
        if cleanup_failure is not None:
            _attach_task_worker_secondary_failure(
                primary_failure,
                cleanup_failure,
                label="Task retry ownership-loss and cancellation-fence drain failures",
            )
        _attach_task_worker_secondary_failure(
            delivered_cancellation,
            primary_failure,
            label="Task retry cancellation and ownership-loss evidence",
        )
        if cleanup_failure is not None:
            _attach_task_worker_secondary_failure(
                delivered_cancellation,
                cleanup_failure,
                label="Task retry cancellation-fence drain failures",
            )
        restore_task_cancellation_requests(
            cleanup_outcome.cancellation_requests_consumed,
            cancellation=delivered_cancellation,
        )
        if owner_cancellation is None:
            raise delivered_cancellation from exception_cause(delivered_cancellation)
        return

    if cleanup_failure is not None:
        if owner_cancellation is None:
            raise cleanup_failure from primary_failure
        _attach_task_worker_secondary_failure(
            primary_failure,
            cleanup_failure,
            label="Task retry cancellation-fence drain failures",
        )


async def _drain_ordinary_task_handler_under_cancellation_fence(
    task_store: TaskStore,
    task: Task,
    worker_id: str,
    handler_task: asyncio.Future[Any],
    heartbeat_task: asyncio.Task[_TaskHeartbeatOutcome],
    stop_heartbeat: asyncio.Event,
    *,
    lease_authority: _TaskLeaseAuthority,
    lease_seconds: int,
    cancellation_error: dict[str, Any],
    primary_failure: BaseException,
    owner_cancellation: asyncio.CancelledError | None = None,
) -> None:
    """Fence an ordinary task durably until its handler is positively quiescent."""

    heartbeat_task = _heartbeat_during_quiescence(
        task_store,
        task,
        worker_id,
        lease_seconds,
        heartbeat_task,
        stop_heartbeat,
        lease_authority=lease_authority,
    )

    async def cleanup() -> None:
        failures: list[BaseException] = []
        requested: Task | None = None
        try:
            requested = await _request_task_cancellation_fence(
                task_store,
                task,
                worker_id,
                cancellation_error,
                lease_authority=lease_authority,
            )
        except (GeneratorExit, KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            failures.append(exc)
        cancellation_requested = requested is not None and _task_cancellation_requested(requested)

        try:
            await handler_task
        except asyncio.CancelledError as exc:
            failures.append(
                unexpected_child_cancellation_error(
                    exc,
                    operation="Ordinary task handler drain",
                )
            )
        except (GeneratorExit, KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            failures.append(exc)

        # The settlement presents the claim's generation, so the heartbeat keeps the
        # lease live until it has committed.
        if cancellation_requested:
            try:
                await _settle_ordinary_task_cancellation_after_quiescence(
                    task_store,
                    task.id,
                    worker_id,
                    claim_generation=lease_authority.claim_generation,
                )
            except (GeneratorExit, KeyboardInterrupt, SystemExit):
                raise
            except BaseException as exc:
                failures.append(exc)

        stop_heartbeat.set()
        if not heartbeat_task.done():
            try:
                await heartbeat_task
            except (GeneratorExit, KeyboardInterrupt, SystemExit):
                raise
            except BaseException as exc:
                failures.append(exc)

        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise BaseExceptionGroup(
                "Ordinary task cancellation-fence cleanup failures",
                failures,
            )

    cleanup_outcome = await await_shielded_task_outcome(
        asyncio.create_task(cleanup()),
        cancellation=owner_cancellation,
    )
    cleanup_failure = cleanup_outcome.error
    if cleanup_failure is not None and exception_tree_contains(
        cleanup_failure,
        (GeneratorExit, KeyboardInterrupt, SystemExit),
    ):
        _attach_task_worker_secondary_failure(
            cleanup_failure,
            primary_failure,
            label="Task-worker process-control and ownership-loss evidence",
        )
        raise cleanup_failure from exception_cause(cleanup_failure)

    delivered_cancellation = cleanup_outcome.cancellation
    if delivered_cancellation is not None:
        _attach_task_worker_secondary_failure(
            delivered_cancellation,
            primary_failure,
            label="Task-worker cancellation and ownership-loss evidence",
        )
        if cleanup_failure is not None:
            _attach_task_worker_secondary_failure(
                delivered_cancellation,
                cleanup_failure,
                label="Task-worker cancellation-fence cleanup failures",
            )
        restore_task_cancellation_requests(
            cleanup_outcome.cancellation_requests_consumed,
            cancellation=delivered_cancellation,
        )
        if owner_cancellation is None:
            raise delivered_cancellation from exception_cause(delivered_cancellation)
        return

    if cleanup_failure is not None:
        _attach_task_worker_secondary_failure(
            primary_failure,
            cleanup_failure,
            label="Task-worker cancellation-fence cleanup failures",
        )


async def _request_task_cancellation_fence(
    task_store: TaskStore,
    task: Task,
    worker_id: str,
    cancellation_error: dict[str, Any],
    *,
    lease_authority: _TaskLeaseAuthority,
) -> Task | None:
    """Retry until durable state prevents another worker from dispatching the task."""

    claim_generation = lease_authority.claim_generation
    if claim_generation is None:
        raise TaskClaimLost("Task cancellation fence requires the claim's generation.")
    async with lease_authority.lock:
        while True:
            current = await task_store.load_task(task.id)
            if current is None:
                raise TaskClaimLost("Task cancellation fence target no longer exists.")
            if current.status in _TERMINAL_TASK_STATUSES:
                return current
            if not _cancellation_drain_holds(
                current, worker_id=worker_id, claim_generation=claim_generation
            ):
                raise TaskClaimLost(
                    "Task cancellation fence no longer owns the expected claim generation."
                )
            if _task_cancellation_requested(current) or _task_retry_cancellation_requested(current):
                return current
            try:
                requested = await task_store.request_claimed_task_cancellation(
                    task.id,
                    worker_id,
                    error=cancellation_error,
                    claim_generation=claim_generation,
                )
            except (GeneratorExit, KeyboardInterrupt, SystemExit):
                raise
            except BaseException as request_failure:
                if exception_tree_contains(
                    request_failure,
                    (GeneratorExit, KeyboardInterrupt, SystemExit),
                ):
                    raise
                try:
                    current = await task_store.load_task(task.id)
                except (GeneratorExit, KeyboardInterrupt, SystemExit):
                    raise
                except BaseException as reconciliation_failure:
                    if exception_tree_contains(
                        reconciliation_failure,
                        (GeneratorExit, KeyboardInterrupt, SystemExit),
                    ):
                        raise
                    await asyncio.sleep(0.05)
                    continue
                if current is None:
                    raise request_failure
                if current.status in _TERMINAL_TASK_STATUSES:
                    return current
                if not _cancellation_drain_holds(
                    current, worker_id=worker_id, claim_generation=claim_generation
                ):
                    raise TaskClaimLost(
                        "Task cancellation fence lost its claim generation."
                    ) from request_failure
                if _task_cancellation_requested(current) or _task_retry_cancellation_requested(
                    current
                ):
                    return current
                if isinstance(request_failure, TaskClaimLost):
                    raise
                await asyncio.sleep(0.05)
                continue
            if (
                requested.status
                in {
                    TaskStatus.COMPLETED,
                    TaskStatus.FAILED,
                    TaskStatus.CANCELLED,
                }
                or _task_cancellation_requested(requested)
                or _task_retry_cancellation_requested(requested)
            ):
                return requested
            await asyncio.sleep(0.05)


async def _settle_ordinary_task_cancellation_after_quiescence(
    task_store: TaskStore,
    task_id: str,
    worker_id: str,
    *,
    claim_generation: int,
) -> None:
    """Settle the exact cancellation after this runtime observed natural quiescence.

    Only claim ``claim_generation`` or reclaim's fence of it is settled. A later
    claim, possibly under the same worker name, observes and owns its own
    cancellation request.
    """

    current = await task_store.load_task(task_id)
    if (
        current is None
        or current.status in _TERMINAL_TASK_STATUSES
        or not _cancellation_drain_holds(
            current, worker_id=worker_id, claim_generation=claim_generation
        )
    ):
        return
    if current.claim_generation == claim_generation:
        terminalization = _task_cancellation_terminalization_request(
            current,
            worker_id=worker_id,
            claim_generation=claim_generation,
        )
        if terminalization is None:
            raise TaskTerminalizationConflict(
                "Ordinary task drain lost its durable cancellation request."
            )
        with contextlib.suppress(TaskClaimLost, TaskTerminalizationConflict):
            await _terminalize_claimed_task_or_detect_peer_winner(
                task_store,
                terminalization,
            )
        current = await task_store.load_task(task_id)
    # The claim's lease lapsed (reclaim may have fenced it): settle it as owner-lost,
    # with this runtime's observed quiescence as the evidence.
    if (
        current is None
        or current.status in _TERMINAL_TASK_STATUSES
        or current.lease_expires_at is None
        or not _task_cancellation_requested(current)
        or not _cancellation_drain_holds(
            current, worker_id=worker_id, claim_generation=claim_generation
        )
    ):
        return

    # Only the cancellation's idempotency key is read from it.
    terminalization = _task_cancellation_terminalization_request(current, worker_id=worker_id)
    if terminalization is None:  # pragma: no cover - guarded above
        raise AssertionError("Task cancellation request disappeared during reconciliation.")
    evidence_identity = canonical_durable_json_bytes(
        {
            "schema": "cayu.task-worker-quiescence.v1",
            "task_id": current.id,
            "worker_id": worker_id,
            "cancellation_idempotency_key": terminalization.idempotency_key,
        },
        "task_worker_quiescence",
    )
    evidence_sha256 = sha256(evidence_identity).hexdigest()
    request = _owner_lost_cancellation_request(
        current,
        worker_id,
        cancellation_idempotency_key=terminalization.idempotency_key,
        reconciliation_idempotency_key=f"task-worker-quiescence:v1:{evidence_sha256}",
        reconciled_by=ResolutionActor(
            subject="cayu:task-worker-quiescence",
            source=ResolutionActorSource.SYSTEM,
        ),
        validator_id="cayu.task-worker",
        validator_version="1",
        evidence_id=f"task-worker-quiescence:{evidence_sha256}",
        evidence_sha256=evidence_sha256,
    )
    await task_store.reconcile_task_cancellation(request)


def _owner_lost_cancellation_request(
    current: Task,
    worker_id: str,
    *,
    cancellation_idempotency_key: str,
    reconciliation_idempotency_key: str,
    reconciled_by: ResolutionActor,
    validator_id: str,
    validator_version: str,
    evidence_id: str,
    evidence_sha256: str,
) -> TaskCancellationReconciliationRequest:
    """Build the exact owner-lost reconciliation for a cancellation-fenced task's claim."""

    if current.lease_expires_at is None:
        raise TaskTerminalizationConflict("Task cancellation reconciliation lost its lease.")
    execution_profile_fingerprint = current.metadata.get("execution_profile_fingerprint")
    effect_fingerprint = current.metadata.get("effect_fingerprint")
    for name, value in (
        ("execution_profile_fingerprint", execution_profile_fingerprint),
        ("effect_fingerprint", effect_fingerprint),
    ):
        if value is not None and type(value) is not str:
            raise TaskTerminalizationConflict(
                f"Task cancellation reconciliation has invalid {name} authority."
            )
    cancellation_payload = current.status_payload
    if (
        type(cancellation_payload) is not dict
        or type(cancellation_payload.get("event")) is not dict
    ):
        raise TaskTerminalizationConflict(
            "Task cancellation reconciliation lost its request event."
        )
    cancellation_requested_at = TaskCancellationReconciliationEvent.model_validate(
        cancellation_payload["event"]
    ).occurred_at
    # This timestamp describes locally observed evidence only. The store still
    # decides owner-loss eligibility from its own transaction time; a skewed
    # evidence timestamp can only reject this reconciliation and retain the
    # cancellation fence.
    # Both values are stamped by the task store.  Worker wall time must not
    # decide whether an owner-lost reconciliation is accepted by a differently
    # clocked durable store.
    quiescence_validated_at = max(
        cancellation_requested_at,
        current.lease_expires_at,
    )
    return TaskCancellationReconciliationRequest(
        task_id=current.id,
        original_worker_id=worker_id,
        original_handoff_id=current.interrupted_handoff_id,
        original_lease_expires_at=current.lease_expires_at,
        cancellation_requested_at=cancellation_requested_at,
        cancellation_idempotency_key=cancellation_idempotency_key,
        reconciliation_idempotency_key=reconciliation_idempotency_key,
        reconciliation_requested_at=quiescence_validated_at,
        reconciled_by=reconciled_by,
        evidence=TaskCancellationReconciliationEvidence(
            outcome=TaskCancellationReconciliationOutcome.QUIESCENT,
            validator_id=validator_id,
            validator_version=validator_version,
            evidence_id=evidence_id,
            evidence_sha256=evidence_sha256,
            validated_at=quiescence_validated_at,
            execution_profile_fingerprint=execution_profile_fingerprint,
            effect_fingerprint=effect_fingerprint,
        ),
        expected_execution_profile_fingerprint=execution_profile_fingerprint,
        expected_effect_fingerprint=effect_fingerprint,
    )


async def _settle_retry_task_cancellation_after_quiescence(
    task_store: TaskStore,
    task_id: str,
    worker_id: str,
    *,
    cancellation_baseline: int,
    report: TaskRetryAttemptReport | None,
) -> None:
    """Settle a lease-loss cancellation from this worker's positive quiescence."""

    try:
        receipt, _cancellation = await _settle_task_retry_cancellation_resisting_cancellation(
            task_store,
            task_id,
            worker_id,
            cancellation_baseline=cancellation_baseline,
            owner_cancellation=None,
            report=report,
        )
    except (TaskClaimLost, TaskTerminalizationConflict):
        receipt = None
    if receipt is not None:
        return
    current = await task_store.load_task(task_id)
    if current is None or current.status in {
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    }:
        return
    series = current.retry_series
    payload = current.status_payload
    if (
        current.worker_id != worker_id
        or current.lease_expires_at is None
        or series is None
        or type(payload) is not dict
        or not _task_retry_cancellation_requested(current)
        or type(payload.get("event")) is not dict
        or type(payload.get("settlement_idempotency_key")) is not str
    ):
        raise TaskTerminalizationConflict(
            "Task retry quiescence lost its durable cancellation authority."
        )
    event = TaskRetryCancellationReconciliationEvent.model_validate(payload["event"])
    execution_profile_fingerprint = current.metadata.get("execution_profile_fingerprint")
    effect_fingerprint = current.metadata.get("effect_fingerprint")
    evidence_identity = canonical_durable_json_bytes(
        {
            "schema": "cayu.task-worker-retry-quiescence.v1",
            "task_id": current.id,
            "worker_id": worker_id,
            "cancellation_idempotency_key": payload["settlement_idempotency_key"],
        },
        "task_retry_worker_quiescence",
    )
    evidence_sha256 = sha256(evidence_identity).hexdigest()
    # Reconciliation is eligible only after this store-owned lease expiry. Use
    # durable task time for the evidence boundary rather than worker wall time.
    validated_at = max(event.occurred_at, current.lease_expires_at)
    await task_store.reconcile_task_retry_cancellation(
        TaskRetryCancellationReconciliationRequest(
            task_id=current.id,
            series_id=series.series_id,
            attempt=series.attempt,
            causal_budget_id=series.causal_budget_id,
            original_worker_id=worker_id,
            original_lease_expires_at=current.lease_expires_at,
            cancellation_requested_at=event.occurred_at,
            cancellation_idempotency_key=payload["settlement_idempotency_key"],
            reconciliation_idempotency_key=(f"task-worker-retry-quiescence:v1:{evidence_sha256}"),
            reconciliation_requested_at=validated_at,
            reconciled_by=ResolutionActor(
                subject="cayu:task-worker-retry-quiescence",
                source=ResolutionActorSource.SYSTEM,
            ),
            evidence=TaskRetryCancellationReconciliationEvidence(
                outcome=TaskRetryCancellationReconciliationOutcome.QUIESCENT,
                validator_id="cayu.task-worker",
                validator_version="1",
                evidence_id=f"task-worker-retry-quiescence:{evidence_sha256}",
                evidence_sha256=evidence_sha256,
                validated_at=validated_at,
                execution_profile_fingerprint=execution_profile_fingerprint,
                effect_fingerprint=effect_fingerprint,
            ),
            expected_execution_profile_fingerprint=execution_profile_fingerprint,
            expected_effect_fingerprint=effect_fingerprint,
        )
    )


async def _quiesce_task_retry_handler_before_terminalization(
    task_store: TaskStore,
    task_id: str,
    worker_id: str,
    lease_seconds: int,
    handler_task: asyncio.Future[Any],
    cancellation_baseline: int,
    lease_authority: _TaskLeaseAuthority,
    *,
    cancellation: _TaskCancellationDelivery | None = None,
) -> _TaskRetryQuiescence:
    """Retain the claim until the handler settles.

    With ``cancellation``, the handler is first told that its task was asked to
    stop and, unless the worker opted out, cancelled once the grace period
    passes. The claim is still retained until the handler is actually done.
    """

    interval = min(lease_seconds / 3, 1.0)
    owner_task = asyncio.current_task()
    if owner_task is None:  # pragma: no cover - coroutine execution invariant
        raise RuntimeError("Task retry handler quiescence requires an owning task.")
    owner_cancellation: asyncio.CancelledError | None = None
    lease_maintenance_failure: BaseException | None = None
    interrupt_at: float | None = None
    if cancellation is not None and not handler_task.done():
        cancellation.signal.request()
        if cancellation.interrupt_after_s is not None:
            interrupt_at = monotonic() + cancellation.interrupt_after_s
    while not handler_task.done():
        try:
            timeout = interval
            if interrupt_at is not None:
                remaining = interrupt_at - monotonic()
                if remaining <= 0:
                    # Cancel once, as the handler time limit does. A handler
                    # that resists keeps the claim until it returns.
                    interrupt_at = None
                    if cancellation is not None and cancellation.may_interrupt():
                        handler_task.cancel()
                    continue
                timeout = min(interval, remaining)
            completed, _pending = await asyncio.wait((handler_task,), timeout=timeout)
            if handler_task in completed:
                break
            renewed = await task_store.heartbeat(
                task_id,
                worker_id,
                claim_generation=lease_authority.claim_generation,
                extend_seconds=lease_seconds,
            )
            lease_authority.lease_expires_at = _require_task_lease_expires_at(renewed)
        except asyncio.CancelledError as exc:
            if owner_task.cancelling() <= cancellation_baseline:
                if lease_maintenance_failure is None:
                    lease_maintenance_failure = exc
            elif owner_cancellation is None:
                owner_cancellation = exc
        except Exception as exc:
            if lease_maintenance_failure is None:
                lease_maintenance_failure = exc
        except BaseException:
            # The durable cancellation request keeps replacement disabled.
            raise
    report: TaskRetryAttemptReport | None = None
    try:
        outcome = handler_task.result()
    except asyncio.CancelledError:
        pass
    except Exception:
        pass
    else:
        if type(outcome) is TaskRetryAttemptReport:
            report = TaskRetryAttemptReport.model_validate(
                outcome.model_dump(mode="python", warnings=False)
            )
    if owner_cancellation is not None:
        if lease_maintenance_failure is not None and owner_cancellation.__cause__ is None:
            owner_cancellation.__cause__ = lease_maintenance_failure
        return _TaskRetryQuiescence(owner_cancellation, report)
    if lease_maintenance_failure is not None:
        raise RuntimeError("Task retry lease maintenance failed while draining.") from (
            lease_maintenance_failure
        )
    return _TaskRetryQuiescence(None, report)


async def _await_task_retry_cleanup_resisting_cancellation(
    operation: Awaitable[Any],
    *,
    label: str,
    owner_cancellation: asyncio.CancelledError,
) -> Any:
    """Finish one ownership mutation before redelivering worker cancellation."""

    owner_task = asyncio.current_task()
    if owner_task is None:  # pragma: no cover - coroutine execution invariant
        raise RuntimeError("Task retry cleanup requires an owning task.")
    observed_cancellation_requests = owner_task.cancelling()
    operation_task = asyncio.ensure_future(operation)
    while True:
        try:
            return await asyncio.shield(operation_task)
        except asyncio.CancelledError as exc:
            current_requests = owner_task.cancelling()
            if current_requests > observed_cancellation_requests:
                observed_cancellation_requests = current_requests
                owner_cancellation.add_note(
                    f"Task retry {label} received an additional caller cancellation request."
                )
                continue
            if operation_task.done() and operation_task.cancelled():
                raise RuntimeError(f"Task retry {label} was unexpectedly cancelled.") from exc
            # ``owner_cancellation`` was already captured by the caller. A
            # shield can redeliver that same request at this nested boundary
            # without increasing ``Task.cancelling()``. Keep supervising the
            # dispatched cleanup so its eventual failure is observed and
            # attached exactly once instead of orphaning the task.
            continue


async def _handoff_interrupted_session(
    app: CayuApp,
    task_store: TaskStore,
    lease_authority: _TaskLeaseAuthority,
) -> None:
    """Hand off the session of the claim this callback holds, never a newer one.

    The disposition is bound to the callback's own claim generation. A row that
    shows another generation belongs to a later holder, possibly under the same
    worker name, and is left alone.
    """

    task_id = lease_authority.task_id
    worker_id = lease_authority.worker_id
    claim_generation = lease_authority.claim_generation
    assert task_id is not None and worker_id is not None and claim_generation is not None
    task = await task_store.load_task(task_id)
    if task is None or task.status in {
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    }:
        return
    if task.claim_generation != claim_generation:
        _log_superseded_disposition(app, task, lease_authority)
        return
    if task.status is not TaskStatus.RUNNING or task.session_id is None:
        await _safe_fail(
            app,
            task_store,
            task_id,
            worker_id,
            RuntimeError(
                "Task handler requested an interrupted-session handoff without a "
                "running attached task."
            ),
            lease_authority=lease_authority,
        )
        return
    if (
        task.worker_id != worker_id
        or task.lease_expires_at is None
        or task.session_instance_id is None
    ):
        raise TaskInterruptedHandoffConflict(
            "Interrupted-session handoff no longer has exact task ownership."
        )

    if not task_store_interrupted_handoff_capability_is_complete(task_store):
        raise NotImplementedError(
            "Interrupted-session handoff requires complete idempotent store support."
        )
    session = await app.session_store.load(task.session_id)
    if session is None:
        await _safe_fail(
            app,
            task_store,
            task_id,
            worker_id,
            RuntimeError(f"Attached session not found: {task.session_id}."),
            lease_authority=lease_authority,
        )
        return
    if session.status is not SessionStatus.INTERRUPTED:
        await _safe_fail(
            app,
            task_store,
            task_id,
            worker_id,
            RuntimeError(
                "Task handler requested an interrupted-session handoff while session "
                f"{task.session_id} was {session.status}."
            ),
            lease_authority=lease_authority,
        )
        return
    if session.instance_id != task.session_instance_id:
        await _safe_fail(
            app,
            task_store,
            task_id,
            worker_id,
            RuntimeError(
                "Task handler requested an interrupted-session handoff for a "
                "different session incarnation."
            ),
            lease_authority=lease_authority,
        )
        return
    request = claim_interrupted_task_handoff_request(
        task_id=task_id,
        worker_id=worker_id,
        claim_generation=claim_generation,
        prior_handoff_id=lease_authority.handoff_id,
        session_id=session.id,
        session_instance_id=session.instance_id,
        session_run_epoch=session.run_epoch,
        parked=await _session_parked_wait(app, session.id),
    )
    try:
        await _settle_interrupted_task_handoff_with_retry(
            app,
            task_store,
            request,
            recover_expired=False,
        )
    except _TaskInterruptedHandoffReceiptConflict:
        raise
    except TaskInterruptedHandoffConflict:
        # Another exact owner may terminalize the task after the session/task
        # preflight but before the handoff mutation acquires store authority.
        # That durable peer winner has already settled this worker's ownership;
        # it must not abort the worker loop.  Non-terminal conflicts remain
        # authoritative and fail closed.
        current = await task_store.load_task(task_id)
        if current is None or current.status in {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }:
            return
        if current.claim_generation != claim_generation:
            _log_superseded_disposition(app, current, lease_authority)
            return
        cancellation = _task_cancellation_terminalization_request(
            current,
            worker_id=worker_id,
            claim_generation=claim_generation,
        )
        if cancellation is not None:
            await _terminalize_claimed_task_or_detect_peer_winner(
                task_store,
                cancellation,
            )
            return
        raise


#: The shortest disposition deadline: a disposition renews its claim while it
#: runs, so this bounds how long a stuck store call can keep the claim alive.
_MIN_TASK_DISPOSITION_DEADLINE_S = 300.0


async def _dispose_within_deadline(
    app: CayuApp,
    task: Task,
    lease_authority: _TaskLeaseAuthority,
    lease_seconds: int,
    disposition: Awaitable[None],
) -> bool:
    """Run an ordinary task's disposition while its heartbeat keeps renewing.

    Returns False when the disposition outlived its deadline (one lease, at
    least five minutes): the claim stops renewing and is left to lease expiry
    and reclaim instead of being held by a stuck store call.
    """

    deadline_s = max(float(lease_seconds), _MIN_TASK_DISPOSITION_DEADLINE_S)
    deadline = asyncio.timeout(deadline_s)
    try:
        async with deadline:
            await disposition
    except TimeoutError:
        if not deadline.expired():
            raise
        logger.warning(
            "Task disposition did not commit within %ss; the claim stops renewing and is "
            "left to lease expiry: task_id=%s claim_generation=%s",
            deadline_s,
            app.redact_json(task.id),
            lease_authority.claim_generation,
        )
        return False
    return True


def _log_lost_disposition(
    app: CayuApp,
    task_id: str,
    claim_generation: int | None,
    lost: TaskClaimLost,
) -> None:
    logger.warning(
        "Task disposition lost its claim: task_id=%s claim_generation=%s reason=%s",
        app.redact_json(task_id),
        claim_generation,
        app.redact_json(str(lost)),
    )


def _log_superseded_disposition(
    app: CayuApp,
    task: Task,
    lease_authority: _TaskLeaseAuthority,
) -> None:
    logger.warning(
        "Task disposition lost its claim: task_id=%s claim_generation=%s "
        "current_claim_generation=%s",
        app.redact_json(task.id),
        lease_authority.claim_generation,
        task.claim_generation,
    )


async def _session_parked_wait(app: CayuApp, session_id: str) -> TaskParkedWait | None:
    """Name the wait an interrupted session is parked on, if it has one of its own.

    That is its own sleep, or the sleep of a foreground subagent it waits on, down
    its chain of subagents. A woken round that stopped on a tool a person must settle
    waits on that person, not on its timer, so its task is handed off unparked for
    them to continue. Otherwise it is the external-event wait its turn parked on.
    """

    from cayu.external_wait_tasks import external_parked_wait

    if not app._uses_session_wakes:
        return await external_parked_wait(app, session_id)
    from cayu.tasks.session_wakes import _sleeping_wait

    sleeping = await _sleeping_wait(app, session_id)
    if sleeping is None:
        return await external_parked_wait(app, session_id)
    sleeper, wake_id = sleeping
    return TaskParkedWait(
        kind="session_wake",
        wait_id=wake_id,
        session_id=None if sleeper == session_id else sleeper,
    )


async def _may_await_continuation(app: CayuApp, task_store: TaskStore, handled: Task) -> bool:
    """Whether a handled task may have left a session for the generic continuation.

    A wake task, and a hosted external wait's service or deadline task, continues
    its session inside its own attempt, so only an owning task it handed off
    unparked waits on a continuation. Any other task may, unless it ended parked on
    a trigger of its own. An unreadable state counts as pending, so the until-idle
    result asks for another run.
    """

    from cayu.tasks.session_wakes import SESSION_WAKE_TASK_TYPE

    try:
        hosting = app._external_wait_hosting
        if handled.type == SESSION_WAKE_TASK_TYPE:
            session_id = handled.input.get("session_id")
        elif hosting is not None and hosting.continues_sessions(handled):
            session_id = await hosting.continued_session_id(handled)
            if session_id is None:
                return False
        else:
            return await _handoff_state(task_store, handled.id) != "parked"
        owners = await task_store.list_tasks(
            TaskQuery(session_id=session_id, status=TaskStatus.RUNNING, limit=1)
        )
        return bool(owners) and await _handoff_state(task_store, owners[0].id) == "unparked"
    except Exception:
        return True


async def _handoff_state(
    task_store: TaskStore, task_id: str
) -> Literal["parked", "unparked"] | None:
    """How a task was handed off for continuation; ``None`` when it was not."""

    task = await task_store.load_task(task_id)
    if (
        task is None
        or task.status is not TaskStatus.RUNNING
        or task.worker_id is not None
        or task.interrupted_handoff_id is None
    ):
        return None
    receipt = await task_store.load_interrupted_task_handoff_receipt(
        task.id, task.interrupted_handoff_id
    )
    return "parked" if receipt is not None and receipt.request.parked is not None else "unparked"


async def _settle_interrupted_task_handoff_once(
    app: CayuApp,
    task_store: TaskStore,
    request: TaskInterruptedHandoffRequest,
    *,
    recover_expired: bool,
) -> TaskInterruptedHandoffReceipt:
    owner_task = asyncio.current_task()
    if owner_task is None:  # pragma: no cover - coroutine execution invariant
        raise RuntimeError("Interrupted-task handoff requires an owning task.")
    cancellation_baseline = owner_task.cancelling()
    operation = (
        task_store.recover_interrupted_task_worker(request)
        if recover_expired
        else task_store.release_interrupted_task_worker(request)
    )
    operation_task = asyncio.create_task(operation)
    owner_cancellation: asyncio.CancelledError | None = None
    # A cancellation request can already be pending when this helper starts.
    # The store operation is now dispatched, so use a real checkpoint to take
    # ownership of that signal and continue supervising the mutation until it
    # reaches a definite result.  Comparing only against the entry count would
    # misclassify this delivery as historical and orphan ``operation_task``.
    try:
        await asyncio.sleep(0)
    except asyncio.CancelledError as exc:
        owner_cancellation = exc
        cancellation_baseline = owner_task.cancelling()
    while True:
        owner_secondary_failure: RuntimeError | None = None
        child_cancellation_failure: RuntimeError | None = None
        try:
            # Wait on the owned task directly without an ``asyncio.shield``
            # proxy. Cancelling a shield proxy before the store settles can
            # make asyncio log the store's later raw exception even when this
            # owner subsequently observes and redacts it.
            await asyncio.wait((operation_task,))
            receipt = operation_task.result()
        except asyncio.CancelledError as exc:
            if owner_task.cancelling() > cancellation_baseline:
                cancellation_baseline = owner_task.cancelling()
                if owner_cancellation is None:
                    owner_cancellation = exc
                continue
            if operation_task.done() and operation_task.cancelled():
                failure = RuntimeError(
                    "Interrupted-task handoff store operation cancelled unexpectedly."
                )
                if owner_cancellation is not None:
                    owner_secondary_failure = failure
                else:
                    child_cancellation_failure = failure
            else:
                raise
        except (GeneratorExit, KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            if exception_tree_contains(exc, (GeneratorExit, KeyboardInterrupt, SystemExit)):
                raise
            if owner_cancellation is not None:
                diagnostic = app.redact_exception_diagnostic(
                    exc,
                    empty_message="interrupted-task handoff cleanup failed",
                    nonportable_message=(
                        "Interrupted-task handoff cleanup had a non-portable diagnostic."
                    ),
                )
                owner_secondary_failure = RuntimeError(diagnostic.message)
            else:
                raise
        else:
            validated_receipt: TaskInterruptedHandoffReceipt | None = None
            validation_failure: RuntimeError | None = None
            try:
                validated_receipt = _validate_interrupted_task_handoff_receipt(receipt, request)
            except (GeneratorExit, KeyboardInterrupt, SystemExit):
                raise
            except BaseException as exc:
                if exception_tree_contains(exc, (GeneratorExit, KeyboardInterrupt, SystemExit)):
                    raise
                if owner_cancellation is None:
                    raise
                diagnostic = app.redact_exception_diagnostic(
                    exc,
                    empty_message="interrupted-task handoff receipt validation failed",
                    nonportable_message=(
                        "Interrupted-task handoff receipt validation had a non-portable diagnostic."
                    ),
                )
                validation_failure = RuntimeError(diagnostic.message)
            if owner_cancellation is not None:
                if validation_failure is not None:
                    raise owner_cancellation from validation_failure
                raise owner_cancellation from None
            assert validated_receipt is not None
            return validated_receipt
        if owner_secondary_failure is not None:
            assert owner_cancellation is not None
            raise owner_cancellation from owner_secondary_failure
        if child_cancellation_failure is not None:
            raise child_cancellation_failure from None


async def _settle_interrupted_task_handoff_with_retry(
    app: CayuApp,
    task_store: TaskStore,
    request: TaskInterruptedHandoffRequest,
    *,
    recover_expired: bool,
) -> TaskInterruptedHandoffReceipt:
    delay = _INTERRUPTED_HANDOFF_INITIAL_BACKOFF_SECONDS
    last_failure_diagnostic: str | None = None
    # Read while this handoff still owns the session, so a progress event can
    # only follow this marker, even when the handoff event itself lands later.
    progress_sequence = await _session_progress_sequence(app, request.session_id)
    for attempt in range(1, _INTERRUPTED_HANDOFF_MAX_ATTEMPTS + 1):
        failed_attempt_status: Literal["pending", "retrying"] | None = None
        try:
            receipt, settled_by_recovery = await _settle_interrupted_task_handoff_attempt(
                app,
                task_store,
                request,
                recover_expired=recover_expired,
            )
        except (TaskInterruptedHandoffConflict, asyncio.CancelledError):
            raise
        except Exception as exc:
            last_failure_diagnostic = app.redact_exception_diagnostic(
                exc,
                empty_message="interrupted-task handoff was not acknowledged",
                nonportable_message=(
                    "Interrupted-task handoff failed with a non-portable diagnostic."
                ),
            ).message
            failed_attempt_status = "pending" if attempt == 1 else "retrying"
        else:
            await _emit_interrupted_task_handoff_event(
                app,
                request,
                handoff_status=(
                    "recovered"
                    if recover_expired or settled_by_recovery or attempt > 1
                    else "released"
                ),
                attempt=attempt,
                recovery_mode=recover_expired or settled_by_recovery,
                progress_sequence=progress_sequence,
            )
            return receipt
        assert failed_attempt_status is not None
        # Publish after leaving the store exception handler. If cancellation
        # interrupts event delivery, Python must not retain the unredacted store
        # failure in the CancelledError context chain.
        await _emit_interrupted_task_handoff_event(
            app,
            request,
            handoff_status=failed_attempt_status,
            attempt=attempt,
            recovery_mode=recover_expired,
            progress_sequence=progress_sequence,
        )
        try:
            receipt = await task_store.load_interrupted_task_handoff_receipt(
                request.task_id,
                request.handoff_id,
            )
        except Exception as exc:
            last_failure_diagnostic = app.redact_exception_diagnostic(
                exc,
                empty_message="interrupted-task handoff was not acknowledged",
                nonportable_message=(
                    "Interrupted-task handoff failed with a non-portable diagnostic."
                ),
            ).message
        else:
            if receipt is not None:
                receipt = _validate_interrupted_task_handoff_receipt(receipt, request)
                await _emit_interrupted_task_handoff_event(
                    app,
                    request,
                    handoff_status="recovered",
                    attempt=attempt,
                    recovery_mode=recover_expired,
                    progress_sequence=progress_sequence,
                )
                return receipt
        if attempt < _INTERRUPTED_HANDOFF_MAX_ATTEMPTS:
            await asyncio.sleep(delay)
            delay = min(delay * 2, _INTERRUPTED_HANDOFF_MAX_BACKOFF_SECONDS)

    await _emit_interrupted_task_handoff_event(
        app,
        request,
        handoff_status="recovery_required",
        attempt=_INTERRUPTED_HANDOFF_MAX_ATTEMPTS,
        recovery_mode=recover_expired,
        progress_sequence=progress_sequence,
    )
    raise _TaskInterruptedHandoffRecoveryRequired(
        last_failure_diagnostic or "Interrupted-task handoff was not acknowledged."
    ) from None


async def _settle_interrupted_task_handoff_attempt(
    app: CayuApp,
    task_store: TaskStore,
    request: TaskInterruptedHandoffRequest,
    *,
    recover_expired: bool,
) -> tuple[TaskInterruptedHandoffReceipt, bool]:
    """Settle once, upgrading an unchanged expired live lease to recovery."""

    try:
        receipt = await _settle_interrupted_task_handoff_once(
            app,
            task_store,
            request,
            recover_expired=recover_expired,
        )
    except _TaskInterruptedHandoffReceiptConflict:
        raise
    except TaskInterruptedHandoffConflict:
        if recover_expired:
            raise
        current = await task_store.load_task(request.task_id)
        if not _task_still_matches_interrupted_handoff_request(current, request):
            raise
        receipt = await _settle_interrupted_task_handoff_once(
            app,
            task_store,
            request,
            recover_expired=True,
        )
        return receipt, True
    return receipt, recover_expired


def _task_still_matches_interrupted_handoff_request(
    task: Task | None,
    request: TaskInterruptedHandoffRequest,
) -> bool:
    """Return whether store state still carries the exact frozen handoff tuple."""

    return bool(
        task is not None
        and task.status is TaskStatus.RUNNING
        and task.id == request.task_id
        and task.worker_id == request.worker_id
        and task.lease_expires_at is not None
        and (
            task.claim_generation == request.claim_generation
            if request.claim_generation is not None
            else task.lease_expires_at == request.lease_expires_at
        )
        and task.session_id == request.session_id
        and task.session_instance_id == request.session_instance_id
        and not _task_cancellation_requested(task)
        and not _task_retry_cancellation_requested(task)
    )


def _validate_interrupted_task_handoff_receipt(
    receipt: object,
    request: TaskInterruptedHandoffRequest,
) -> TaskInterruptedHandoffReceipt:
    if type(receipt) is not TaskInterruptedHandoffReceipt:
        raise _TaskInterruptedHandoffReceiptConflict(
            "Interrupted-task handoff store returned malformed receipt evidence."
        )
    try:
        copied_receipt = TaskInterruptedHandoffReceipt(
            request=receipt.request,
            request_sha256=receipt.request_sha256,
            task=receipt.task,
            committed_at=receipt.committed_at,
        )
    except (GeneratorExit, KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:
        if exception_tree_contains(exc, (GeneratorExit, KeyboardInterrupt, SystemExit)):
            raise
        raise _TaskInterruptedHandoffReceiptConflict(
            "Interrupted-task handoff store returned malformed receipt evidence."
        ) from None
    _, request_sha256 = prepare_interrupted_task_handoff(request)
    task = copied_receipt.task
    if (
        copied_receipt.request != request
        or copied_receipt.request_sha256 != request_sha256
        or task.id != request.task_id
        or task.status is not TaskStatus.RUNNING
        or task.session_id != request.session_id
        or task.session_instance_id != request.session_instance_id
        or task.worker_id is not None
        or task.lease_expires_at is not None
    ):
        raise _TaskInterruptedHandoffReceiptConflict(
            "Interrupted-task handoff receipt conflicts with exact authority."
        )
    return copied_receipt


async def _emit_interrupted_task_handoff_event(
    app: CayuApp,
    request: TaskInterruptedHandoffRequest,
    *,
    handoff_status: str,
    attempt: int,
    recovery_mode: bool,
    progress_sequence: int | None = None,
) -> Event | None:
    """Best-effort exact publication that never owns the task mutation."""

    identity = canonical_durable_json_bytes(
        {
            "schema": "cayu.task-interrupted-handoff-event.v1",
            "handoff_id": request.handoff_id,
            "handoff_status": handoff_status,
            "attempt": attempt,
            "recovery_mode": recovery_mode,
        },
        "task_interrupted_handoff_event",
    )
    event = event_with_runtime_generated_id(
        Event(
            id=f"task-handoff:v1:{sha256(identity).hexdigest()}",
            type=EventType.TASK_INTERRUPTED_HANDOFF,
            session_id=request.session_id,
            payload={
                "task_id": request.task_id,
                "handoff_id": request.handoff_id,
                "handoff_status": handoff_status,
                "attempt": attempt,
                "session_run_epoch": request.session_run_epoch,
                **({} if progress_sequence is None else {"progress_sequence": progress_sequence}),
                **(
                    {}
                    if request.parked is None
                    else {
                        "parked_wait_kind": request.parked.kind,
                        "parked_wait_id": request.parked.wait_id,
                    }
                ),
            },
        )
    )
    event = event_with_runtime_envelope_authority(event, "session_id")
    payload_authority = (
        ["handoff_id"] if request.parked is None else ["handoff_id", "parked_wait_id"]
    )
    if app.redact_json(request.task_id) == request.task_id:
        # The task identity is store-resolved, but its original value remains
        # application-authored. Preserve it as durable authority only after the
        # workload-secret boundary proves that exact value safe.
        payload_authority.append("task_id")
    event = event_with_runtime_payload_authority(event, *payload_authority)
    try:
        persisted = await app._event_writer.persist_exact_replay(event)
    except (asyncio.CancelledError, GeneratorExit, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        return None
    try:
        delivered = await app._event_writer.fan_out_persisted([persisted])
    except (asyncio.CancelledError, GeneratorExit, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        # The event and its side-effect handoff are already durable. Ordinary
        # sink delivery recovery owns any later retry.
        return persisted
    emitted = delivered[0]
    app._session_control.queue_out_of_band_event(emitted)
    return emitted


async def _completion_owes_task(app: CayuApp, session: Session, task_id: str) -> bool:
    """Whether a completed session's unsettled decision still owes this task."""

    if session.status is not SessionStatus.COMPLETED:
        return False
    from cayu.sessions._paired_writes import unsettled_completed_decision

    owed = unsettled_completed_decision(await app.session_store.load_checkpoint(session.id))
    return owed is not None and owed.task_id == task_id


async def _recover_expired_interrupted_task_handoffs(
    app: CayuApp,
    task_store: TaskStore,
    *,
    worker_id: str,
    after: tuple[datetime, str] | None,
    limit: int,
    stop: asyncio.Event | None,
) -> _InterruptedHandoffRecoveryPage:
    """Recover at most one store page before yielding to ordinary task work."""

    candidates = await task_store.list_expired_interrupted_task_handoff_candidates(
        after=after,
        limit=limit,
    )
    recovered = 0
    failed: set[str] = set()
    manual_recovery_required = False
    next_after = after
    for task in candidates:
        if _is_stopped(stop):
            return _InterruptedHandoffRecoveryPage(
                recovered=recovered,
                next_after=next_after,
                exhausted=False,
                failed=frozenset(failed),
                manual_recovery_required=manual_recovery_required,
            )
        if type(task) is not Task or task.session_id is None or task.lease_expires_at is None:
            raise TypeError("Interrupted-task handoff recovery returned an invalid task.")
        next_after = (task.lease_expires_at, task.id)
        session = await app.session_store.load(task.session_id)
        if (
            session is None
            or task.session_instance_id is None
            or session.instance_id != task.session_instance_id
        ):
            continue
        if _task_cancellation_requested(task):
            from cayu.tasks.session_wakes import in_subagent_chain

            # A foreground subagent may still run under the same lost claim: a
            # cancelled task in a chain of subagent calls is left fenced, as before,
            # and nothing of its sessions is touched.
            try:
                if await in_subagent_chain(app, session):
                    continue
            except Exception:
                failed.add(task.id)
                continue
        if session.status in {
            SessionStatus.PENDING,
            SessionStatus.RUNNING,
            SessionStatus.INTERRUPTING,
        } or await _completion_owes_task(app, session, task.id):
            from cayu.external_wait_tasks import recover_before_handoff

            try:
                # A turn that died before its external wait parked is recovered by
                # that wait, never plainly, which would leave it to rerun blind.
                if not await recover_before_handoff(app, session.id):
                    await app.recover_incomplete_session(
                        IncompleteSessionRecoveryRequest(
                            session_id=session.id,
                            inactive_for_seconds=None,
                            reason="expired_attached_task_owner",
                            metadata={"recovery_scope": "attached_task"},
                        )
                    )
            except (asyncio.CancelledError, GeneratorExit, KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                diagnostic = app.redact_exception_diagnostic(
                    exc,
                    empty_message="attached-session recovery failed",
                    nonportable_message=(
                        "Attached-session recovery failed with a non-portable diagnostic."
                    ),
                )
                logger.warning(
                    "Attached-session recovery failed before task handoff: "
                    "task_id=%s session_id=%s error_type=%s error=%s",
                    app.redact_json(task.id),
                    app.redact_json(session.id),
                    type(exc).__name__,
                    diagnostic.message,
                )
                if isinstance(exc, _manual_recovery_errors()):
                    manual_recovery_required = True
                else:
                    failed.add(task.id)
                continue
            session = await app.session_store.load(task.session_id)
            if session is None or session.instance_id != task.session_instance_id:
                continue
        if session.status is not SessionStatus.INTERRUPTED and not (
            # A completed session fences the lost worker's run as an interrupted
            # one does; its pending cancellation still decides the task.
            session.status is SessionStatus.COMPLETED and _task_cancellation_requested(task)
        ):
            continue
        from cayu.external_wait_tasks import stop_wait_left_arming

        try:
            # Another process may have interrupted a turn that died before parking on
            # its external wait; that wait must not stay armed and fence the task.
            await stop_wait_left_arming(app, session.id)
        except Exception as exc:
            logger.warning(
                "Stopping the external wait of interrupted session %s failed (%s); its task "
                "is handed off on a later pass.",
                app.redact_json(session.id),
                type(exc).__name__,
            )
            failed.add(task.id)
            continue
        if _task_cancellation_requested(task):
            # Nobody is left to drain the cancellation, and the interrupted session
            # fences the lost worker's run: settle it instead of handing it off. A
            # session in a chain of subagent calls was left alone above.
            try:
                if await _settle_expired_owner_cancellation(
                    task_store,
                    task,
                    actor=_declaring_worker_actor(worker_id),
                    evidence_id=(
                        f"session-{session.status.value}:{session.instance_id}:{session.run_epoch}"
                    ),
                ):
                    recovered += 1
                    logger.warning(
                        "Settled task %s as cancelled: its cancellation was pending when its "
                        "worker stopped renewing its lease, and its session has ended.",
                        app.redact_json(task.id),
                    )
            except Exception as exc:
                logger.warning(
                    "Settling the cancellation of task %s failed (%s); a later pass retries it.",
                    app.redact_json(task.id),
                    type(exc).__name__,
                )
                failed.add(task.id)
            continue
        request = interrupted_task_handoff_request(
            task,
            session_run_epoch=session.run_epoch,
            parked=await _session_parked_wait(app, session.id),
        )
        try:
            await _settle_interrupted_task_handoff_with_retry(
                app,
                task_store,
                request,
                recover_expired=True,
            )
        except TaskInterruptedHandoffConflict:
            current = await task_store.load_task(task.id)
            if (
                current is None
                or current.status
                in {
                    TaskStatus.COMPLETED,
                    TaskStatus.FAILED,
                    TaskStatus.CANCELLED,
                }
                or _task_cancellation_requested(current)
                or _task_retry_cancellation_requested(current)
                or current.worker_id is None
                or current.worker_id != request.worker_id
                or current.claim_generation != request.claim_generation
            ):
                continue
            raise
        recovered += 1
    return _InterruptedHandoffRecoveryPage(
        recovered=recovered,
        next_after=next_after,
        exhausted=len(candidates) < limit,
        failed=frozenset(failed),
        manual_recovery_required=manual_recovery_required,
    )


async def _settle_expired_owner_cancellation(
    task_store: TaskStore,
    task: Task,
    *,
    actor: ResolutionActor,
    evidence_id: str,
) -> bool:
    """Cancel an attached task whose worker's lease expired with its cancellation pending.

    Its session is interrupted or completed, which fences the lost worker's run, so
    nothing of it can continue; ``evidence_id`` names that session incarnation and run epoch. The
    store settles only the exact expired claim ``task`` shows; a claim that changed,
    a lease that is live again, or a settlement another worker made first is left
    alone. Returns whether this call settled the task.
    """

    worker_id = task.worker_id
    if worker_id is None:
        return False
    try:
        terminalization = _task_cancellation_terminalization_request(task, worker_id=worker_id)
        if terminalization is None:
            return False
        evidence_sha256 = sha256(
            canonical_durable_json_bytes(
                {
                    "schema": "cayu.expired-attached-task-cancellation.v1",
                    "task_id": task.id,
                    "worker_id": worker_id,
                    "handoff_id": task.interrupted_handoff_id,
                    "cancellation_idempotency_key": terminalization.idempotency_key,
                    "evidence_id": evidence_id,
                },
                "expired_attached_task_cancellation",
            )
        ).hexdigest()
        request = _owner_lost_cancellation_request(
            task,
            worker_id,
            cancellation_idempotency_key=terminalization.idempotency_key,
            reconciliation_idempotency_key=(
                f"expired-attached-task-cancellation:v1:{evidence_sha256}"
            ),
            # The recovering worker, so a peer that settled it first makes this conflict.
            reconciled_by=actor,
            validator_id=_EXPIRED_ATTACHED_OWNER_VALIDATOR_ID,
            validator_version=_EXPIRED_ATTACHED_OWNER_VALIDATOR_VERSION,
            evidence_id=evidence_id,
            evidence_sha256=evidence_sha256,
        )
        await task_store.reconcile_task_cancellation(request)
    except (TaskTerminalizationConflict, TaskClaimLost):
        return False
    else:
        return True


def _manual_recovery_errors() -> tuple[type[Exception], ...]:
    """Recovery failures that no later pass fixes until a person acts."""

    from cayu.runtime._approval_support import (
        RoundToolManualRecoveryRequired,
        ToolApprovalManualRecoveryRequired,
    )
    from cayu.runtime._model_completion_contracts import ModelCompletionManualRecoveryRequired

    return (
        ModelCompletionManualRecoveryRequired,
        RoundToolManualRecoveryRequired,
        ToolApprovalManualRecoveryRequired,
    )


async def _run_recovered_interrupted_task_handler(
    app: CayuApp,
    task_store: TaskStore,
    handler: RecoveredInterruptedTaskHandler,
    *,
    worker_id: str,
    expected: TaskContinuationExpectation | None = None,
    apply_unproductive_backoff: bool = True,
    lease_seconds: int,
    query: TaskQuery | None,
    after: tuple[datetime, str] | None,
    max_unproductive_handoffs: int | None = None,
    stop: asyncio.Event | None = None,
    execution_timeout_s: float | None = None,
    execution_grace_s: float | None = None,
    interrupt_on_cancel: bool = True,
    event_budget_check: EventBudgetCheck | None = None,
) -> InterruptedTaskContinuationClaimPage:
    """Scan one bounded page and supervise its optional continuation claim."""

    if expected is not None and (
        type.__getattribute__(type(task_store), "__dict__").get(
            "supports_bound_interrupted_task_handoffs"
        )
        is not True
        or not task_store_interrupted_handoff_capability_is_complete(task_store)
        or not task_store_mutation_is_cancellation_quiescent(
            task_store, "claim_interrupted_task_continuation"
        )
    ):
        raise TaskContinuationUnsupported("Controlled continuation requires bound atomic claims.")
    handoff_id = new_interrupted_task_continuation_handoff_id()

    async def claim_once():
        bound: dict[str, Any] = (
            {} if expected is None else {"task_id": expected.task_id, "expected": expected}
        )
        return await task_store.claim_interrupted_task_continuation(
            worker_id,
            query,
            handoff_id=handoff_id,
            lease_seconds=lease_seconds,
            after=after,
            scan_limit=_INTERRUPTED_HANDOFF_RECOVERY_BATCH_SIZE,
            **bound,
        )

    async def claim():
        try:
            return await claim_once()
        except (OSError, TimeoutError):
            if expected is None:
                raise
            # Same generation and expected tuple reconcile commit-before-ack loss.
            return await claim_once()

    claim_outcome = await capture_task_store_operation(
        claim,
        operation_name="Interrupted-task continuation claim",
        redactor=app._secret_redactor,
        mutation_store=task_store,
        mutation_method_name="claim_interrupted_task_continuation",
    )
    if claim_outcome.failure is not None:
        failure = claim_outcome.failure
        del claim_outcome
        raise_task_store_operation_failure(failure)
    page = claim_outcome.result
    del claim_outcome
    if type(page) is not InterruptedTaskContinuationClaimPage:
        raise RuntimeError("Interrupted-task continuation claim returned an invalid bounded page.")
    task = page.task
    if task is None:
        return page
    task = copy_task(task)
    if (
        task.status is not TaskStatus.RUNNING
        or task.worker_id != worker_id
        or task.lease_expires_at is None
        or task.session_id is None
        or task.session_instance_id is None
    ):
        raise RuntimeError(
            "Interrupted-task continuation claim returned invalid ownership authority."
        )

    async def continue_with_bound(
        owned_app: CayuApp, owned_task: Task, owned_worker_id: str
    ) -> TaskHandlerOutcome | None:
        return await _run_bounded_interrupted_task_handler(
            owned_app,
            task_store,
            owned_task,
            handler,
            owned_worker_id,
            max_unproductive_handoffs=max_unproductive_handoffs,
            apply_unproductive_backoff=apply_unproductive_backoff,
            stop=stop,
        )

    # Session/history reads and backoff can outlast a lease. Supervise the
    # entire continuation, including its bound decision, as owned handler work.
    await _handle_with_heartbeat(
        app,
        task_store,
        task,
        continue_with_bound,
        worker_id,
        lease_seconds,
        recover_interrupted_authority_loss=True,
        execution_timeout_s=execution_timeout_s,
        execution_grace_s=execution_grace_s,
        interrupt_on_cancel=interrupt_on_cancel,
        event_budget_check=event_budget_check,
    )
    return page


async def _run_bounded_interrupted_task_handler(
    app: CayuApp,
    task_store: TaskStore,
    task: Task,
    handler: RecoveredInterruptedTaskHandler,
    worker_id: str,
    *,
    max_unproductive_handoffs: int | None,
    apply_unproductive_backoff: bool = True,
    stop: asyncio.Event | None,
) -> TaskHandlerOutcome | None:
    assert task.session_id is not None
    session = await app.session_store.load(task.session_id)
    if (
        session is None
        or session.instance_id != task.session_instance_id
        or session.status is not SessionStatus.INTERRUPTED
    ):
        logger.warning(
            "Interrupted-task continuation session changed after claim; "
            "the durable lease remains available for owner-loss recovery: "
            "task_id=%s session_id=%s",
            app.redact_json(task.id),
            app.redact_json(task.session_id),
        )
        raise SessionRunFenced("Interrupted-task continuation session changed after claim.")
    unproductive_handoffs = await _count_unproductive_interrupted_handoffs(
        app,
        task,
        cap=max_unproductive_handoffs or _UNPRODUCTIVE_CONTINUATION_BACKOFF_COUNT_CAP,
    )
    if max_unproductive_handoffs is not None and unproductive_handoffs >= max_unproductive_handoffs:
        if await _session_has_pending_action(app, task.session_id):
            # Approvals, user input, delegated actions and manual tool recovery
            # wait on someone else. Keep throttling them but never fail them.
            logger.warning(
                "Interrupted-task continuation made no progress across %d handoffs "
                "but its session has a pending action; continuing with backoff: "
                "task_id=%s session_id=%s",
                unproductive_handoffs,
                app.redact_json(task.id),
                app.redact_json(task.session_id),
            )
        else:
            logger.warning(
                "Interrupted-task continuation made no progress across %d handoffs; "
                "failing the task instead of continuing it again: task_id=%s session_id=%s",
                unproductive_handoffs,
                app.redact_json(task.id),
                app.redact_json(task.session_id),
            )
            await _safe_fail_payload(
                app,
                task_store,
                task.id,
                worker_id,
                {
                    "error": "InterruptedTaskContinuationStalled",
                    "message": (
                        f"The attached session was handed off {unproductive_handoffs} times "
                        "without model, tool, or input progress, so its continuation was "
                        "stopped instead of being released again."
                    ),
                    "unproductive_interrupted_handoffs": unproductive_handoffs,
                },
                claim_generation=task.claim_generation,
                handoff_id=task.interrupted_handoff_id,
            )
            return None
    # Explicit owned human decisions do not throttle behind automated retry history.
    # Session identity, bound claims, heartbeat, stop and quiescent cleanup still apply.
    delay = (
        _unproductive_continuation_backoff(unproductive_handoffs)
        if apply_unproductive_backoff
        else 0
    )
    if delay > 0:
        logger.warning(
            "Interrupted-task continuation has made no progress across %d handoffs; "
            "backing off %.1fs before continuing: task_id=%s session_id=%s",
            unproductive_handoffs,
            delay,
            app.redact_json(task.id),
            app.redact_json(task.session_id),
        )
        if await _wait_or_stop(delay, stop) or _worker_should_stop(app, stop):
            # Hand the claim back so another worker can continue it promptly.
            return TaskHandlerOutcome.SESSION_INTERRUPTED
    if _worker_should_stop(app, stop):
        return TaskHandlerOutcome.SESSION_INTERRUPTED
    return await handler(app, task, worker_id)


def _unproductive_continuation_backoff(unproductive_handoffs: int) -> float:
    """Return the wait before continuing while heartbeat supervision runs.

    The first handoff follows the original run; each later one is a continuation
    that made no progress.
    """

    if unproductive_handoffs < 2:
        return 0.0
    exponent = min(unproductive_handoffs - 2, _UNPRODUCTIVE_CONTINUATION_MAX_BACKOFF_EXPONENT)
    return min(
        _UNPRODUCTIVE_CONTINUATION_INITIAL_BACKOFF_SECONDS * 2**exponent,
        _UNPRODUCTIVE_CONTINUATION_MAX_BACKOFF_SECONDS,
    )


async def _session_progress_sequence(app: CayuApp, session_id: str) -> int | None:
    """Return the session's latest progress event sequence, or None if unreadable."""

    try:
        records = await app.session_store.query_events(
            EventQuery(
                session_id=session_id,
                event_types=_CONTINUATION_PROGRESS_EVENT_TYPES,
                order_by=EventOrder.SEQUENCE_DESC,
                limit=1,
            )
        )
    except Exception:
        return None
    return records[0].sequence if records else 0


async def _session_has_pending_action(app: CayuApp, session_id: str) -> bool:
    """Return whether the session may be waiting on someone else's action."""

    try:
        result = await app.session_store.query_pending_actions(
            PendingActionQuery(session_id=session_id, limit=1)
        )
    except Exception:
        # An unreadable pending-action state is not proof that nothing waits.
        return True
    return bool(result.actions or result.issues or result.has_more)


async def _count_unproductive_interrupted_handoffs(
    app: CayuApp,
    task: Task,
    *,
    cap: int,
) -> int:
    """Count this task's handoffs decided since its session last made progress.

    Each handoff event records the progress sequence read while that handoff
    still owned the session. Only handoffs carrying the current sequence count,
    so a handoff event published after later progress is ignored. Handoff events
    are an observability projection: a missing event or a failed read can only
    lower the count.
    """

    assert task.session_id is not None
    progress_sequence = await _session_progress_sequence(app, task.session_id)
    if progress_sequence is None:
        logger.warning(
            "Could not read interrupted-task handoff history; continuing without "
            "the unproductive-continuation bound: task_id=%s session_id=%s",
            app.redact_json(task.id),
            app.redact_json(task.session_id),
        )
        return 0
    task_ids = {task.id, app.redact_json(task.id)}
    handoff_ids: set[str] = set()
    before_sequence: int | None = None
    # The descending cursor excludes newly appended events after the first
    # page. Scan that finite history until the requested count or exhaustion;
    # a fixed page budget can make an accepted bound permanently unreachable.
    while True:
        try:
            records = await app.session_store.query_events(
                EventQuery(
                    session_id=task.session_id,
                    event_type=EventType.TASK_INTERRUPTED_HANDOFF,
                    after_sequence=progress_sequence,
                    before_sequence=before_sequence,
                    order_by=EventOrder.SEQUENCE_DESC,
                    limit=_UNPRODUCTIVE_HANDOFF_HISTORY_PAGE_SIZE,
                )
            )
        except Exception:
            break
        for record in records:
            payload = record.event.payload
            recorded_progress = payload.get("progress_sequence")
            handoff_id = payload.get("handoff_id")
            if (
                payload.get("task_id") in task_ids
                and payload.get("handoff_status") in {"released", "recovered"}
                and type(recorded_progress) is int
                and recorded_progress == progress_sequence
                and type(handoff_id) is str
            ):
                handoff_ids.add(handoff_id)
                if len(handoff_ids) >= cap:
                    return cap
        if len(records) < _UNPRODUCTIVE_HANDOFF_HISTORY_PAGE_SIZE:
            break
        before_sequence = records[-1].sequence
        # In-memory adapters may finish a read without yielding. Give lease
        # renewal and cancellation a turn between pages of a long history.
        await asyncio.sleep(0)
    return len(handoff_ids)


async def _settle_recovered_continuation_authority_loss(
    app: CayuApp,
    task_store: TaskStore,
    lease_authority: _TaskLeaseAuthority,
) -> bool:
    """Release a still-interrupted continuation or defer it to lease recovery."""

    assert lease_authority.task_id is not None
    current = await task_store.load_task(lease_authority.task_id)
    if current is None or current.status in {
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    }:
        return True
    if (
        current.status is not TaskStatus.RUNNING
        or current.worker_id != lease_authority.worker_id
        or current.lease_expires_at is None
        or current.claim_generation != lease_authority.claim_generation
        or current.session_id is None
        or current.session_instance_id is None
    ):
        # A peer changed task authority. This worker must not publish a competing
        # terminal or handoff disposition from its stale snapshot.
        return True

    session = await app.session_store.load(current.session_id)
    if (
        session is not None
        and session.instance_id == current.session_instance_id
        and session.status is SessionStatus.INTERRUPTED
    ):
        await _handoff_interrupted_session(app, task_store, lease_authority)
        return True

    # A different session epoch may still be settling the task. Stop renewing
    # this worker's lease instead of converting that session fence into a task
    # failure. If no peer settles it, the exact lease expiry re-enters the
    # existing interrupted-owner recovery path.
    logger.warning(
        "Recovered continuation lost session authority; deferring task ownership "
        "to lease-expiry recovery: task_id=%s session_id=%s",
        app.redact_json(current.id),
        app.redact_json(current.session_id),
    )
    return False


async def _settle_pending_runtime_task_failure_owner(
    app: CayuApp,
    task_store: TaskStore,
    *,
    task_id: str,
    worker_id: str,
    lease_authority: _TaskLeaseAuthority,
    pending: RuntimeTaskFailureTerminalizationPending,
) -> None:
    """Settle only the worker whose exact durable failure decision is active."""

    if type(pending) is not RuntimeTaskFailureTerminalizationPending:
        raise TypeError("pending must be a RuntimeTaskFailureTerminalizationPending.")
    if pending.task_id != task_id:
        raise TaskClaimLost("Pending runtime task failure is bound to a different task.") from None

    current = await task_store.load_task(task_id)
    if current is None:
        raise TaskClaimLost("Pending runtime task failure lost its attached task.") from None
    if current.status is TaskStatus.FAILED:
        failure_identity = runtime_task_failure_identity_from_task(
            current,
            session_id=pending.session_id,
            session_instance_id=pending.session_instance_id,
        )
        if (
            failure_identity is not None
            and failure_identity.failure_id == pending.runtime_task_failure_id
            and failure_identity.run_epoch == pending.run_epoch
        ):
            return
        raise TaskClaimLost(
            "Pending runtime task failure conflicts with the terminal task evidence."
        ) from None
    if (
        current.status is not TaskStatus.RUNNING
        or current.session_id != pending.session_id
        or current.session_instance_id != pending.session_instance_id
    ):
        raise TaskClaimLost(
            "Pending runtime task failure conflicts with the attached task authority."
        ) from None

    session = await app.session_store.load(pending.session_id)
    checkpoint = await app.session_store.load_checkpoint(pending.session_id)
    decision = invocation_terminal_decision_from_checkpoint(checkpoint)
    active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
    if (
        session is None
        or session.instance_id != pending.session_instance_id
        or session.run_epoch != pending.run_epoch
        or session.status
        not in {SessionStatus.RUNNING, SessionStatus.INTERRUPTING, SessionStatus.FAILED}
        or decision is None
        or decision.outcome is not InvocationTerminalOutcome.FAILED
        or decision.decision_id != pending.decision_id
        or decision.runtime_task_failure_id != pending.runtime_task_failure_id
        or decision.task_id != task_id
        or active_profile is None
        or not invocation_terminal_decision_matches_active_profile(
            decision,
            session_id=session.id,
            session_instance_id=session.instance_id,
            run_epoch=session.run_epoch,
            interaction_id=active_profile.interaction_id,
            execution_profile_fingerprint=active_profile.profile.fingerprint,
        )
    ):
        raise TaskClaimLost(
            "Pending runtime task failure lost its exact durable decision authority."
        ) from None

    if (
        current.worker_id != worker_id
        or current.claim_generation != lease_authority.claim_generation
        or current.interrupted_handoff_id != lease_authority.handoff_id
    ):
        raise TaskClaimLost(
            "Pending runtime task failure no longer owns the worker claim."
        ) from None
    if (
        decision.task_error_payload is None
        or decision.task_terminalization_request_sha256 is None
        or decision.task_terminalization_request_sha256
        != runtime_task_failure_terminalization_request_sha256(
            task_id=task_id,
            task_worker_id=worker_id,
            task_handoff_id=lease_authority.handoff_id,
            session_id=pending.session_id,
            error=decision.task_error_payload,
        )
    ):
        raise TaskClaimLost(
            "Pending runtime task failure conflicts with its elected task mutation."
        ) from None
    terminal = await _terminalize_claimed_task(
        task_store,
        runtime_task_failure_terminalization_request(
            task_id=task_id,
            task_worker_id=worker_id,
            task_handoff_id=lease_authority.handoff_id,
            session_id=pending.session_id,
            error=decision.task_error_payload,
        ),
    )
    if (
        terminal.status is not TaskStatus.FAILED
        or terminal.session_id != pending.session_id
        or terminal.session_instance_id != pending.session_instance_id
        or terminal.error != decision.task_error_payload
    ):
        raise TaskClaimLost(
            "Task terminalization returned conflicting runtime failure evidence."
        ) from None


async def _heartbeat_until(
    task_store: TaskStore,
    task_id: str,
    worker_id: str,
    lease_seconds: int,
    stop: asyncio.Event,
    *,
    lease_authority: _TaskLeaseAuthority,
    claim_deadline_monotonic: float | None = None,
    enforce_retry_deadline: bool = False,
) -> _TaskHeartbeatOutcome:
    if claim_deadline_monotonic is None:
        deadline_started = monotonic()
        claim_deadline_monotonic = _conservative_task_lease_deadline(
            deadline_started,
            lease_seconds,
        )

    def current_deadline() -> float:
        assert claim_deadline_monotonic is not None
        return claim_deadline_monotonic

    def expired_lease() -> BaseException:
        return TaskClaimLost("Task heartbeat acknowledgement expired before lease renewal.")

    async def heartbeat() -> Task | None:
        nonlocal claim_deadline_monotonic
        updated, claim_deadline_monotonic = await _renew_task_lease_before_deadline(
            task_store,
            task_id,
            worker_id,
            lease_seconds,
            lease_authority=lease_authority,
            claim_deadline_monotonic=current_deadline(),
        )
        return updated

    async def inspect_heartbeat(updated: Task | None) -> _TaskHeartbeatOutcome | None:
        if updated is not None and not lease_authority.keep_renewing_on_cancellation:
            inspection = asyncio.create_task(enforce_task_event_deadline(task_store, updated))
            try:
                completed, _pending = await asyncio.wait(
                    {inspection}, timeout=max(0.0, current_deadline() - monotonic())
                )
            except asyncio.CancelledError:
                _retain_abandoned_task_lease_operation(inspection)
                raise
            if inspection not in completed:
                _retain_abandoned_task_lease_operation(inspection)
                raise expired_lease()
            updated = inspection.result()
        if (
            updated is not None
            and not lease_authority.keep_renewing_on_cancellation
            and (
                _task_retry_cancellation_requested(updated) or _task_cancellation_requested(updated)
            )
        ):
            return _TaskHeartbeatOutcome.CANCELLATION_REQUESTED
        if enforce_retry_deadline:
            retry_deadline_inspection = asyncio.create_task(
                task_store.task_retry_deadline_elapsed(
                    task_id,
                    worker_id,
                    claim_generation=lease_authority.claim_generation,
                )
            )
            try:
                completed, _pending = await asyncio.wait(
                    {retry_deadline_inspection},
                    timeout=max(0.0, current_deadline() - monotonic()),
                )
            except asyncio.CancelledError:
                _retain_abandoned_task_lease_operation(retry_deadline_inspection)
                raise
            if retry_deadline_inspection not in completed:
                _retain_abandoned_task_lease_operation(retry_deadline_inspection)
                raise expired_lease()
            if retry_deadline_inspection.result():
                return _TaskHeartbeatOutcome.ELAPSED
        return None

    async def reconcile_heartbeat_failure(
        heartbeat_error: Exception,
    ) -> _TaskHeartbeatOutcome | None:
        load_task = getattr(task_store, "load_task", None)
        if not callable(load_task):
            raise heartbeat_error
        reconciliation_task = asyncio.create_task(load_task(task_id))
        try:
            completed, _pending = await asyncio.wait(
                {reconciliation_task},
                timeout=max(0.0, current_deadline() - monotonic()),
            )
        except asyncio.CancelledError:
            _retain_abandoned_task_lease_operation(reconciliation_task)
            raise
        if reconciliation_task not in completed:
            _retain_abandoned_task_lease_operation(reconciliation_task)
            raise heartbeat_error
        try:
            task = reconciliation_task.result()
        except Exception as reconciliation_error:
            raise heartbeat_error from reconciliation_error
        if task is not None and task.status in {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }:
            elapsed_receipt = await _load_elapsed_task_retry_receipt(task_store, task)
            if elapsed_receipt is not None:
                return _TaskHeartbeatOutcome.ELAPSED
            return _TaskHeartbeatOutcome.TERMINAL
        raise heartbeat_error

    return await run_durable_lease_heartbeat(
        heartbeat,
        lease_seconds=lease_seconds,
        stop=stop,
        stopped_outcome=_TaskHeartbeatOutcome.STOPPED,
        maximum_interval_s=1.0,
        after_heartbeat=inspect_heartbeat,
        on_failure=reconcile_heartbeat_failure,
        wait=_wait_or_stop,
        lease_deadline=current_deadline,
        deadline_failure=expired_lease,
        clock=monotonic,
    )


async def _renew_task_lease_before_deadline(
    task_store: TaskStore,
    task_id: str,
    worker_id: str,
    lease_seconds: int,
    *,
    lease_authority: _TaskLeaseAuthority,
    claim_deadline_monotonic: float,
) -> tuple[Task | None, float]:
    """Renew the claim by its generation; the lock serializes only local deadline bookkeeping.

    The acknowledgement updates only the local lease deadline. A renewal whose
    acknowledgement is late never invalidates a write of the same claim, which
    presents the generation, not the expiry.
    """

    async with lease_authority.lock:
        renewal_started = monotonic()
        renewal_task = asyncio.create_task(
            task_store.heartbeat(
                task_id,
                worker_id,
                claim_generation=lease_authority.claim_generation,
                extend_seconds=lease_seconds,
            )
        )
        try:
            completed, _pending = await asyncio.wait(
                {renewal_task},
                timeout=max(0.0, claim_deadline_monotonic - monotonic()),
            )
        except asyncio.CancelledError:
            _retain_abandoned_task_lease_operation(renewal_task)
            raise
        if renewal_task not in completed:
            _retain_abandoned_task_lease_operation(renewal_task)
            raise TaskClaimLost(
                "Task heartbeat acknowledgement did not arrive before the last positively "
                "known lease deadline."
            )
        updated = renewal_task.result()
        if updated is not None:
            lease_authority.lease_expires_at = _require_task_lease_expires_at(updated)
        renewed_deadline = _conservative_task_lease_deadline(
            renewal_started,
            lease_seconds,
        )
        if monotonic() >= renewed_deadline:
            raise TaskClaimLost("Task heartbeat acknowledgement consumed the renewed lease.")
        lease_authority.deadline_monotonic = renewed_deadline
        return updated, renewed_deadline


# Lease renewals and reconciliation reads that outlived the worker waiting for
# them, by the application whose operation started them: its shutdown waits for
# them before releasing the task store.
_ABANDONED_TASK_LEASE_OPERATIONS: dict[asyncio.Task[Any], ApplicationAdmission | None] = {}


def _retain_abandoned_task_lease_operation(task: asyncio.Task[Any]) -> None:
    _ABANDONED_TASK_LEASE_OPERATIONS[task] = _current_operation_admission()
    task.add_done_callback(_consume_task_heartbeat_outcome)


def abandoned_task_lease_operations(application: ApplicationAdmission) -> set[asyncio.Future[Any]]:
    """This application's abandoned task-lease store calls still running on this loop."""

    loop = asyncio.get_running_loop()
    return {
        task
        for task, owner in tuple(_ABANDONED_TASK_LEASE_OPERATIONS.items())
        if owner is application and not task.done() and task.get_loop() is loop
    }


def _consume_task_heartbeat_outcome(task: asyncio.Task[Any]) -> None:
    """Observe a store heartbeat that outlived this worker's local authority."""

    _ABANDONED_TASK_LEASE_OPERATIONS.pop(task, None)
    with contextlib.suppress(BaseException):
        task.result()


async def _enforce_task_retry_deadline_with_retry(
    task_store: TaskStore,
    task: Task,
    worker_id: str,
    *,
    lease_expires_at: datetime,
    claim_generation: int,
    token_count: int = 0,
    estimated_cost: Decimal = Decimal(0),
) -> TaskRetrySettlementResult | None:
    """Enforce or authenticate a deadline commit whose acknowledgement was lost.

    The store fences by ``claim_generation``. The expected receipt is derived
    from ``lease_expires_at``, the lease the caller holds still under its lock.
    """

    expected_request, expected_request_sha256 = _task_retry_runtime_terminal_request(
        task.model_copy(update={"lease_expires_at": lease_expires_at}, deep=True),
        operation="elapsed",
        request_disposition=TaskRetryAttemptDisposition.RETRYABLE_FAILURE,
        error={"code": TaskRetrySeriesDisposition.ELAPSED_EXHAUSTED.value},
        token_count=token_count,
        estimated_cost=estimated_cost,
    )
    last_error: Exception | None = None
    for _attempt in range(3):
        try:
            receipt = await task_store.enforce_task_retry_deadline(
                task.id,
                worker_id,
                claim_generation=claim_generation,
                token_count=token_count,
                estimated_cost=estimated_cost,
            )
            if receipt is not None:
                _validate_elapsed_task_retry_receipt(
                    receipt,
                    expected_idempotency_key=expected_request.idempotency_key,
                    expected_request_sha256=expected_request_sha256,
                )
                return receipt
            current = await task_store.load_task(task.id)
            if current is not None:
                cancellation = _task_retry_requested_cancellation_settlement(
                    current,
                    worker_id=worker_id,
                    claim_generation=claim_generation,
                    token_count=token_count,
                    estimated_cost=estimated_cost,
                )
                if cancellation is not None:
                    return await settle_task_retry_attempt_with_retry(task_store, cancellation)
            return None
        except NotImplementedError:
            raise
        except Exception as exc:
            last_error = exc

        try:
            current = await task_store.load_task(task.id)
            receipt = (
                None
                if current is None
                else await _load_elapsed_task_retry_receipt(
                    task_store,
                    current,
                    expected_idempotency_key=expected_request.idempotency_key,
                    expected_request_sha256=expected_request_sha256,
                )
            )
            if receipt is not None:
                return receipt
        except Exception as exc:
            last_error = exc
        await asyncio.sleep(0)

    assert last_error is not None
    raise last_error


async def _enforce_task_retry_deadline_resisting_cancellation(
    task_store: TaskStore,
    task: Task,
    worker_id: str,
    *,
    lease_authority: _TaskLeaseAuthority,
    cancellation_baseline: int,
    owner_cancellation: asyncio.CancelledError | None,
    report: TaskRetryAttemptReport | None,
) -> tuple[TaskRetrySettlementResult | None, asyncio.CancelledError | None]:
    """Finish terminal ownership before redelivering caller cancellation."""

    async def enforce() -> TaskRetrySettlementResult | None:
        # Under the lease, at its current generation: the heartbeat may renew until now.
        async with lease_authority.lock:
            return await _enforce_task_retry_deadline_with_retry(
                task_store,
                task,
                worker_id,
                lease_expires_at=lease_authority.lease_expires_at,
                claim_generation=lease_authority.claim_generation,
                token_count=0 if report is None else report.token_count,
                estimated_cost=Decimal(0) if report is None else report.estimated_cost,
            )

    return await _complete_task_retry_terminalization_resisting_cancellation(
        enforce(),
        label="deadline",
        cancellation_baseline=cancellation_baseline,
        owner_cancellation=owner_cancellation,
    )


async def _complete_task_retry_terminalization_resisting_cancellation(
    terminalization: Awaitable[TaskRetrySettlementResult | None],
    *,
    label: str,
    cancellation_baseline: int,
    owner_cancellation: asyncio.CancelledError | None,
) -> tuple[TaskRetrySettlementResult | None, asyncio.CancelledError | None]:
    """Finish an owned terminal write before redelivering caller cancellation."""

    owner_task = asyncio.current_task()
    if owner_task is None:  # pragma: no cover - coroutine execution invariant
        raise RuntimeError("Task retry terminalization requires an owning task.")
    observed_cancellation_requests = (
        owner_task.cancelling() if owner_cancellation is not None else cancellation_baseline
    )
    terminalization_task = asyncio.ensure_future(terminalization)
    while True:
        try:
            receipt = await asyncio.shield(terminalization_task)
            return receipt, owner_cancellation
        except asyncio.CancelledError as exc:
            current_requests = owner_task.cancelling()
            if current_requests > observed_cancellation_requests:
                observed_cancellation_requests = current_requests
                if owner_cancellation is None:
                    owner_cancellation = exc
                else:
                    owner_cancellation.add_note(
                        f"Task retry {label} terminalization received an additional "
                        "caller cancellation request."
                    )
                continue
            if terminalization_task.done() and terminalization_task.cancelled():
                raise RuntimeError(
                    f"Task retry {label} terminalization was unexpectedly cancelled."
                ) from exc
            raise
        except Exception as terminalization_failure:
            if owner_cancellation is None:
                raise
            existing_cause = owner_cancellation.__cause__
            secondary_failure = (
                terminalization_failure
                if existing_cause is None
                else BaseExceptionGroup(
                    f"Task retry {label} cleanup failures",
                    [existing_cause, terminalization_failure],
                )
            )
            raise owner_cancellation from secondary_failure


async def _settle_task_retry_cancellation_resisting_cancellation(
    task_store: TaskStore,
    task_id: str,
    worker_id: str,
    *,
    cancellation_baseline: int,
    owner_cancellation: asyncio.CancelledError | None,
    report: TaskRetryAttemptReport | None,
    lease_authority: _TaskLeaseAuthority | None = None,
) -> tuple[TaskRetrySettlementResult | None, asyncio.CancelledError | None]:
    """Settle an operator cancellation without releasing its live owner early.

    With ``lease_authority`` (a heartbeat may still be renewing the lease), the attempt is
    read and settled under its lock, so no renewal makes the settlement's lease stale.
    """

    async def settle() -> TaskRetrySettlementResult:
        if lease_authority is None:
            return await attempt()
        async with lease_authority.lock:
            return await attempt()

    async def attempt() -> TaskRetrySettlementResult:
        task = await task_store.load_task(task_id)
        if task is None:
            raise RuntimeError("Task retry cancellation lost its durable attempt.")
        request = _task_retry_requested_cancellation_settlement(
            task,
            worker_id=worker_id,
            claim_generation=None if lease_authority is None else lease_authority.claim_generation,
            token_count=0 if report is None else report.token_count,
            estimated_cost=Decimal(0) if report is None else report.estimated_cost,
        )
        if request is None:
            raise RuntimeError("Task retry cancellation lost its durable request.")
        return await settle_task_retry_attempt_with_retry(task_store, request)

    return await _complete_task_retry_terminalization_resisting_cancellation(
        settle(),
        label="cancellation",
        cancellation_baseline=cancellation_baseline,
        owner_cancellation=owner_cancellation,
    )


async def _load_elapsed_task_retry_receipt(
    task_store: TaskStore,
    task: Task,
    *,
    expected_idempotency_key: str | None = None,
    expected_request_sha256: str | None = None,
) -> TaskRetrySettlementResult | None:
    if (
        task.status is not TaskStatus.FAILED
        or task.retry_series is None
        or task.retry_series.disposition is not TaskRetrySeriesDisposition.ELAPSED_EXHAUSTED
        or task.status_payload is None
    ):
        return None
    idempotency_key = task.status_payload.get("settlement_idempotency_key")
    if not isinstance(idempotency_key, str):
        return None
    receipt = await task_store.load_task_retry_settlement(task.id, idempotency_key)
    if receipt is None:
        return None
    if receipt.task != task or receipt.idempotency_key != idempotency_key:
        raise RuntimeError("Task retry deadline receipt conflicts with durable task state.")
    _validate_elapsed_task_retry_receipt(
        receipt,
        expected_idempotency_key=expected_idempotency_key,
        expected_request_sha256=expected_request_sha256,
    )
    return receipt


def _validate_elapsed_task_retry_receipt(
    receipt: TaskRetrySettlementResult,
    *,
    expected_idempotency_key: str | None,
    expected_request_sha256: str | None,
) -> None:
    if (
        expected_idempotency_key is not None and receipt.idempotency_key != expected_idempotency_key
    ) or (
        expected_request_sha256 is not None and receipt.request_sha256 != expected_request_sha256
    ):
        raise RuntimeError(
            "Task retry deadline receipt conflicts with the quiescent handler accounting."
        )


async def _safe_fail(
    app: CayuApp,
    task_store: TaskStore,
    task_id: str,
    worker_id: str,
    exc: Exception,
    *,
    lease_expires_at: datetime | None = None,
    lease_authority: _TaskLeaseAuthority | None = None,
) -> None:
    payload = _task_failure_payload(app, exc)
    del exc
    await _safe_fail_payload(
        app,
        task_store,
        task_id,
        worker_id,
        payload,
        lease_expires_at=lease_expires_at,
        lease_authority=lease_authority,
        handoff_id=lease_authority.handoff_id if lease_authority is not None else None,
    )


def _task_failure_payload(app: CayuApp, exc: Exception) -> dict[str, Any]:
    """Snapshot, redact, and bound one task failure before a store await."""

    diagnostic = app.redact_exception_diagnostic(
        exc,
        empty_message="task handler failed",
        nonportable_message="Task handler failed with a non-portable diagnostic.",
    )
    payload: dict[str, Any] = {
        "error": diagnostic.error_type,
        "message": diagnostic.message,
    }
    if isinstance(exc, ExecutionDeadlineExceeded) and exc.diagnostics.get("source") == "task_event":
        payload["code"] = "event_elapsed_exhausted"
    if diagnostic.durable_value_error_code is not None:
        payload["durable_value_error_code"] = diagnostic.durable_value_error_code
        payload["durable_value_error_path"] = diagnostic.durable_value_error_path
        if diagnostic.durable_value_error_limit is not None:
            payload["durable_value_error_limit"] = diagnostic.durable_value_error_limit
            payload["durable_value_error_observed_lower_bound"] = (
                diagnostic.durable_value_error_observed_lower_bound
            )
    diagnostic = None
    redacted_payload = app.redact_json(payload)
    payload.clear()
    if type(redacted_payload) is not dict:
        raise AssertionError("Task failure payload redaction returned a non-object.")
    return {
        key: _redact_and_bound_task_failure_text(app, value) if type(value) is str else value
        for key, value in redacted_payload.items()
    }


def _handler_time_limit_failure_payload(
    app: CayuApp,
    time_limit: HandlerTimeLimit,
    handler_error: Exception | None,
) -> dict[str, Any]:
    """Describe a handler call that outlived its time limit, before a store await."""

    message = (
        f"The task handler did not finish within its {time_limit.timeout_s:g}s time "
        "limit and was cancelled."
        if time_limit.cancelled
        else f"The task handler passed {time_limit.soft_timeout_s:g}s of its "
        f"{time_limit.timeout_s:g}s time limit, after which new runs are refused, and "
        "stopped without settling the task."
    )
    payload: dict[str, Any] = {
        "error": "ExecutionDeadlineExceeded",
        "message": f"{message} Anything it was doing then may or may not have taken effect.",
        "scope": "task_handler",
        **time_limit.inspection(),
    }
    if handler_error is not None and not (
        isinstance(handler_error, ExecutionDeadlineExceeded)
        and handler_error.diagnostics.get("scope") == "task_handler"
    ):
        payload["handler_failure"] = _task_failure_payload(app, handler_error)
    redacted = app.redact_json(payload)
    if type(redacted) is not dict:
        raise AssertionError("Task failure payload redaction returned a non-object.")
    return redacted


def _redact_and_bound_task_failure_text(app: CayuApp, value: str) -> str:
    redacted_value = app.redact_json(value)
    if type(redacted_value) is not str:
        raise AssertionError("Task failure text redaction returned a non-string.")
    encoded_value = redacted_value.encode("utf-8", "replace")
    if len(encoded_value) <= _MAX_TASK_FAILURE_MESSAGE_BYTES:
        return redacted_value
    keep = _MAX_TASK_FAILURE_MESSAGE_BYTES - len(_TASK_FAILURE_TRUNCATION_MARKER.encode())
    return encoded_value[:keep].decode("utf-8", "ignore") + _TASK_FAILURE_TRUNCATION_MARKER


async def _safe_fail_payload(
    app: CayuApp,
    task_store: TaskStore,
    task_id: str,
    worker_id: str,
    payload: dict[str, Any],
    *,
    lease_expires_at: datetime | None = None,
    claim_generation: int | None = None,
    lease_authority: _TaskLeaseAuthority | None = None,
    handoff_id: str | None = None,
    retryable: bool = False,
) -> bool:
    """Fail a live claimed task; return whether this call recorded the failure."""

    if claim_generation is None and lease_authority is not None:
        claim_generation = lease_authority.claim_generation
    task = await task_store.load_task(task_id)
    if task is None or task.status not in {TaskStatus.CLAIMED, TaskStatus.RUNNING}:
        return False
    if task.retry_series is not None:
        request = TaskRetrySettlementRequest(
            task_id=task_id,
            worker_id=worker_id,
            lease_expires_at=lease_expires_at,
            claim_generation=claim_generation,
            causal_budget_id=task.retry_series.causal_budget_id,
            disposition=(
                TaskRetryAttemptDisposition.RETRYABLE_FAILURE
                if retryable
                else TaskRetryAttemptDisposition.NON_RETRYABLE_FAILURE
            ),
            error=payload,
            idempotency_key=_worker_failure_terminalization_key(
                task_id,
                worker_id,
                claim_generation,
            ),
        )
        try:
            settled = await (
                _settle_task_retry_request_with_lease_authority(
                    task_store,
                    request,
                    lease_authority,
                )
                if lease_authority is not None
                else _settle_task_retry_request_honoring_cancellation(task_store, request)
            )
        except TaskClaimLost as lost:
            _log_lost_disposition(app, task_id, claim_generation, lost)
            return False
        # A cancellation committed first settles the attempt as cancelled.
        return settled.task.status is TaskStatus.FAILED
    try:
        peer_terminalization_won = await _terminalize_claimed_task_or_detect_peer_winner(
            task_store,
            TaskTerminalizationRequest(
                task_id=task_id,
                worker_id=worker_id,
                lease_expires_at=lease_expires_at,
                claim_generation=claim_generation,
                kind=TaskTerminalKind.FAILED,
                error=payload,
                handoff_id=handoff_id,
                idempotency_key=_worker_failure_terminalization_key(
                    task_id,
                    worker_id,
                    claim_generation,
                ),
            ),
        )
        if peer_terminalization_won:
            return False
    except TaskClaimLost as lost:
        _log_lost_disposition(app, task_id, claim_generation, lost)
        return False
    except ValueError:
        if task_store.supports_idempotent_terminalization:
            raise
        # A handler can terminalize its task and then raise. Preserve that
        # authoritative terminal outcome, but never hide validation failures
        # for a task that is still live.
        task = await task_store.load_task(task_id)
        if task is not None and task.status in {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }:
            return False
        raise
    return True


async def _settle_retry_attempt_report(
    task_store: TaskStore,
    task: Task,
    worker_id: str,
    report: TaskRetryAttemptReport,
    *,
    lease_authority: _TaskLeaseAuthority,
) -> None:
    copied = TaskRetryAttemptReport.model_validate(report.model_dump(mode="python", warnings=False))
    series = task.retry_series
    if series is None:
        raise TypeError("TaskRetryAttemptReport requires a task retry policy.")
    await _settle_task_retry_request_with_lease_authority(
        task_store,
        TaskRetrySettlementRequest(
            task_id=task.id,
            worker_id=worker_id,
            claim_generation=lease_authority.claim_generation,
            causal_budget_id=series.causal_budget_id,
            **copied.model_dump(mode="python", warnings=False),
        ),
        lease_authority,
    )


async def _settle_task_retry_request_with_lease_authority(
    task_store: TaskStore,
    request: TaskRetrySettlementRequest,
    lease_authority: _TaskLeaseAuthority,
) -> TaskRetrySettlementResult:
    """Settle under the claim's generation while heartbeats keep renewing it.

    Renewals keep the generation, so the settlement never races them: a
    ``TaskClaimLost`` here means the lease really lapsed or was superseded.
    """

    return await _settle_task_retry_request_honoring_cancellation(
        task_store,
        request.model_copy(
            update={"lease_expires_at": None, "claim_generation": lease_authority.claim_generation},
            deep=True,
        ),
    )


async def _settle_task_retry_request_honoring_cancellation(
    task_store: TaskStore,
    request: TaskRetrySettlementRequest,
) -> TaskRetrySettlementResult:
    """Let a cancellation committed before settlement win the attempt race."""

    try:
        return await settle_task_retry_attempt_with_retry(task_store, request)
    except TaskTerminalizationConflict:
        task = await task_store.load_task(request.task_id)
        if task is None:
            raise
        cancellation = _task_retry_requested_cancellation_settlement(
            task,
            worker_id=request.worker_id,
            # The generation the settlement presented: a heartbeat may move the
            # lease between this read and the settlement.
            claim_generation=request.claim_generation,
            token_count=request.token_count,
            estimated_cost=request.estimated_cost,
        )
        if cancellation is None:
            raise
        return await settle_task_retry_attempt_with_retry(task_store, cancellation)


def _worker_failure_terminalization_key(
    task_id: str, worker_id: str, claim_generation: int | None
) -> str:
    if claim_generation is None:
        identity = canonical_durable_json_bytes(
            {
                "schema": "cayu.task-worker-failure.v1",
                "task_id": task_id,
                "worker_id": worker_id,
            },
            "task_worker_failure",
        )
        return f"task-worker-failure:v1:{sha256(identity).hexdigest()}"
    # Two incarnations of one worker name never share a failure key.
    identity = canonical_durable_json_bytes(
        {
            "schema": "cayu.task-worker-failure.v2",
            "task_id": task_id,
            "worker_id": worker_id,
            "claim_generation": claim_generation,
        },
        "task_worker_failure",
    )
    return f"task-worker-failure:v2:{sha256(identity).hexdigest()}"


async def _safe_fail_unfinished(
    app: CayuApp,
    task_store: TaskStore,
    task_id: str,
    worker_id: str,
    *,
    lease_expires_at: datetime,
    lease_authority: _TaskLeaseAuthority | None = None,
) -> None:
    task = await task_store.load_task(task_id)
    if task is None or task.status not in {TaskStatus.CLAIMED, TaskStatus.RUNNING}:
        return
    await _safe_fail(
        app,
        task_store,
        task_id,
        worker_id,
        RuntimeError("Task handler returned without completing or failing the task."),
        lease_expires_at=lease_expires_at,
        lease_authority=lease_authority,
    )


def _is_stopped(stop: asyncio.Event | None) -> bool:
    return worker_stop_requested(stop)


async def _settle_task_refused_by_shutdown(
    task_store: TaskStore,
    task_id: str,
    worker_id: str,
    lease_authority: _TaskLeaseAuthority,
) -> bool:
    """Leave a claim the closing app refused for the next worker, without failing it.

    An unattached claim goes back to pending. A task already attached to a session
    or settled keeps its lease until it expires, and owner-loss recovery continues
    it as after a crash. Returns False when a requested cancellation must still
    settle through the ordinary failure path.
    """

    current = await task_store.load_task(task_id)
    if current is not None and current.status_reason in _TASK_CANCELLATION_REASONS:
        return False
    if (
        current is None
        or current.status is not TaskStatus.CLAIMED
        or current.worker_id != worker_id
        or current.session_id is not None
    ):
        return True
    # A claim that moved on or changed meanwhile is left to lease expiry.
    with contextlib.suppress(TaskClaimLost, TaskTerminalizationConflict):
        await task_store.release_task(
            task_id, worker_id, claim_generation=lease_authority.claim_generation
        )
    return True


_TASK_CANCELLATION_REASONS = frozenset(
    {_TASK_CANCELLATION_REQUESTED_REASON, _TASK_RETRY_CANCELLATION_REQUESTED_REASON}
)


def _worker_should_stop(app: CayuApp, stop: asyncio.Event | None) -> bool:
    # A closing application refuses new runs, so claiming more work would only
    # take tasks away from the next worker.
    return _is_stopped(stop) or app.lifecycle_state != "open"


async def _wait_or_stop(seconds: float, stop: asyncio.Event | None) -> bool:
    """Sleep for ``seconds`` or until ``stop`` is set. Returns True if stopped."""
    return await wait_or_stop(seconds, stop)
