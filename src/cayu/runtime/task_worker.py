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
from collections.abc import Awaitable, Callable
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
from cayu.core.events import (
    Event,
    EventType,
    event_with_runtime_envelope_authority,
    event_with_runtime_generated_id,
    event_with_runtime_payload_authority,
)
from cayu.core.runtime_authority import SessionRunFenced
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
    worker_stop_requested,
)
from cayu.runtime._invocation_terminal_decision import (
    InvocationTerminalOutcome,
    invocation_terminal_decision_from_checkpoint,
    invocation_terminal_decision_matches_active_profile,
)
from cayu.runtime._task_lease_authority import (
    TaskLeaseAuthority as _TaskLeaseAuthority,
)
from cayu.runtime._task_lease_authority import (
    bind_task_lease_authority,
    managed_task_lease_mutation,
)
from cayu.runtime._task_store_operation_boundary import (
    capture_task_store_operation,
    raise_task_store_operation_failure,
    task_store_cancellation_reconciliation_capability_is_complete,
    task_store_interrupted_handoff_capability_is_complete,
    task_store_mutation_is_cancellation_quiescent,
)
from cayu.runtime.approvals import ResolutionActor, ResolutionActorSource
from cayu.runtime.execution_profiles import (
    active_invocation_execution_profile_from_checkpoint,
)
from cayu.runtime.sessions import IncompleteSessionRecoveryRequest, SessionStatus
from cayu.runtime.tasks import (
    InterruptedTaskContinuationClaimPage,
    Task,
    TaskCancellationReconciliationEvent,
    TaskCancellationReconciliationEvidence,
    TaskCancellationReconciliationOutcome,
    TaskCancellationReconciliationRequest,
    TaskClaimLost,
    TaskInterruptedHandoffConflict,
    TaskInterruptedHandoffReceipt,
    TaskInterruptedHandoffRequest,
    TaskQuery,
    TaskRetryAttemptDisposition,
    TaskRetryAttemptReport,
    TaskRetryCancellationReconciliationEvent,
    TaskRetryCancellationReconciliationEvidence,
    TaskRetryCancellationReconciliationOutcome,
    TaskRetryCancellationReconciliationRequest,
    TaskRetrySeriesDisposition,
    TaskRetrySettlementRequest,
    TaskRetrySettlementResult,
    TaskStatus,
    TaskStore,
    TaskTerminalizationConflict,
    TaskTerminalizationRequest,
    TaskTerminalKind,
    _task_cancellation_requested,
    _task_cancellation_terminalization_request,
    _task_retry_cancellation_requested,
    _task_retry_requested_cancellation_settlement,
    _task_retry_runtime_terminal_request,
    _terminalize_claimed_task,
    _terminalize_claimed_task_or_detect_peer_winner,
    copy_task,
    interrupted_task_handoff_request,
    new_interrupted_task_continuation_handoff_id,
    prepare_interrupted_task_handoff,
    settle_task_retry_attempt_with_retry,
)

if TYPE_CHECKING:
    from cayu.runtime.app import CayuApp


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
_MAX_TASK_FAILURE_MESSAGE_BYTES = 500
_INTERRUPTED_HANDOFF_MAX_ATTEMPTS = 3
_INTERRUPTED_HANDOFF_INITIAL_BACKOFF_SECONDS = 0.05
_INTERRUPTED_HANDOFF_MAX_BACKOFF_SECONDS = 1.0
_INTERRUPTED_HANDOFF_RECOVERY_BATCH_SIZE = 10
_INTERRUPTED_HANDOFF_RECOVERY_RESCAN_SECONDS = 30.0

logger = logging.getLogger(__name__)


async def complete_managed_task(
    task_store: TaskStore,
    task: Task,
    worker_id: str,
    result: dict[str, Any],
) -> Task:
    """Complete a managed worker task against its latest acknowledged lease."""

    task = copy_task(task)
    async with managed_task_lease_mutation(
        task_id=task.id,
        worker_id=worker_id,
        handoff_id=task.interrupted_handoff_id,
        presented_lease_expires_at=task.lease_expires_at,
    ) as effective_lease:
        return await task_store.complete_task(
            task.id,
            result,
            worker_id=worker_id,
            lease_expires_at=effective_lease,
            handoff_id=task.interrupted_handoff_id,
        )


async def fail_managed_task(
    task_store: TaskStore,
    task: Task,
    worker_id: str,
    error: dict[str, Any],
) -> Task:
    """Fail a managed worker task against its latest acknowledged lease."""

    task = copy_task(task)
    async with managed_task_lease_mutation(
        task_id=task.id,
        worker_id=worker_id,
        handoff_id=task.interrupted_handoff_id,
        presented_lease_expires_at=task.lease_expires_at,
    ) as effective_lease:
        return await task_store.fail_task(
            task.id,
            error,
            worker_id=worker_id,
            lease_expires_at=effective_lease,
            handoff_id=task.interrupted_handoff_id,
        )


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
class _InterruptedHandoffRecoveryPage:
    """One scheduler-bounded expired-owner recovery turn."""

    recovered: int
    next_after: tuple[datetime, str] | None
    exhausted: bool


class _TaskRetryHandlerUnsettled(RuntimeError):
    """A handler remained live after its durable retry authority was lost."""


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
    recover_interrupted_handoffs: bool = True,
    interrupted_handoff_recovery_every_s: float = (_INTERRUPTED_HANDOFF_RECOVERY_RESCAN_SECONDS),
    recovered_interrupted_task_handler: RecoveredInterruptedTaskHandler | None = None,
    continuation_poll_interval_s: float | None = None,
    stop: asyncio.Event | None = None,
    max_tasks: int | None = None,
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
    - ``reclaim`` periodically returns expired leases from dead workers to pending.
    - ``reclaim_every_s`` controls that cadence independently of task claims.
    - ``recover_interrupted_handoffs`` owns bounded recovery of expired attached
      task workers. Elect one such scanner for each shared task store.
    - ``interrupted_handoff_recovery_every_s`` controls the scanner's exhausted
      rescan cadence independently of claim and expired-lease polling.
    - ``recovered_interrupted_task_handler`` makes this worker available to claim
      and continue receipt-backed workerless tasks. This execution capacity is
      independent of expired-owner scanner election.
    - ``continuation_poll_interval_s`` bounds rediscovery latency after an empty
      continuation scan. It defaults to ``poll_interval_s``.
    - ``stop`` is an ``asyncio.Event`` for graceful shutdown.
    - ``max_tasks`` bounds the loop (useful for tests and one-shot drains).
    """
    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive.")
    validate_worker_interval(poll_interval_s, "poll_interval_s")
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
    if max_tasks is not None and max_tasks < 0:
        raise ValueError("max_tasks must be non-negative.")
    worker_id = require_clean_nonblank(worker_id, "worker_id")
    if app.redact_json(worker_id) != worker_id:
        raise ValueError(
            "worker_id contains a workload secret and cannot be used as durable task authority."
        )
    if max_tasks == 0 or _is_stopped(stop):
        return 0
    if not task_store_cancellation_reconciliation_capability_is_complete(task_store):
        raise NotImplementedError(
            "The generic task worker requires complete idempotent ordinary-task "
            "cancellation reconciliation before it can claim work."
        )
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

    async def reclaim_expired_tasks() -> bool:
        if materialized_work_contract_queue_supported:
            reclaim_outcome = await capture_task_store_operation(
                lambda: task_store.reclaim_expired(query=query),
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
            return bool(reclaimed)
        else:
            return bool(await task_store.reclaim_expired(query=query))

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
            recover_interrupted_handoffs
            and interrupted_handoff_supported
            and loop.time() >= next_interrupted_handoff_recovery_at
        ):
            poller.metrics.maintenance(recovery=True)
            recovery_page = await _recover_expired_interrupted_task_handoffs(
                app,
                task_store,
                after=expired_handoff_after,
                limit=_INTERRUPTED_HANDOFF_RECOVERY_BATCH_SIZE,
                stop=stop,
            )
            if recovery_page.recovered:
                meaningful_activity = True
                next_interrupted_continuation_scan_at = 0.0
            if recovery_page.exhausted:
                expired_handoff_after = None
                next_interrupted_handoff_recovery_at = (
                    loop.time() + interrupted_handoff_recovery_every_s
                )
            else:
                expired_handoff_after = recovery_page.next_after
                next_interrupted_handoff_recovery_at = 0.0
            if _is_stopped(stop):
                return DurableWorkerStep(stop=True, activity=meaningful_activity)
        continuation_activity = False
        handled_this_step = 0
        if (
            recovered_interrupted_task_handler is not None
            and loop.time() >= next_interrupted_continuation_scan_at
        ):
            continuation_page = await _run_recovered_interrupted_task_handler(
                app,
                task_store,
                recovered_interrupted_task_handler,
                worker_id=worker_id,
                lease_seconds=lease_seconds,
                query=query,
                after=continuation_after,
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
                continuation_scan_scanned = 0
                continuation_scan_rejected = 0
                continuation_scan_filtered = 0
                next_interrupted_continuation_scan_at = loop.time() + continuation_poll_interval_s
            else:
                continuation_after = continuation_page.next_after
                next_interrupted_continuation_scan_at = 0.0
            if continuation_activity:
                handled_this_step += 1
                if (
                    max_tasks is not None and handled + handled_this_step >= max_tasks
                ) or _is_stopped(stop):
                    return DurableWorkerStep(
                        handled=handled_this_step,
                        stop=True,
                        activity=True,
                    )
        if _is_stopped(stop):
            return DurableWorkerStep(
                handled=handled_this_step,
                stop=True,
                activity=meaningful_activity,
            )
        if reclaim:
            reclaim_ran, reclaimed = await reclaim_cadence.run_if_due(
                reclaim_expired_tasks,
                now=loop.time(),
                clock=loop.time,
            )
            if reclaim_ran:
                poller.metrics.maintenance(reclaim=True)
            meaningful_activity = meaningful_activity or bool(reclaimed)

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

        claim = await poller.claim(
            claim_next_task,
            maximum_active_s=lease_seconds,
        )
        task = claim.value
        if task is None:
            if continuation_activity and continuation_after is not None:
                return DurableWorkerStep(
                    handled=handled_this_step,
                    continue_immediately=True,
                    activity=True,
                )
            wake_deadlines: list[float] = []
            if recovered_interrupted_task_handler is not None and continuation_after is None:
                wake_deadlines.append(next_interrupted_continuation_scan_at)
            if (
                recover_interrupted_handoffs
                and interrupted_handoff_supported
                and expired_handoff_after is None
            ):
                wake_deadlines.append(next_interrupted_handoff_recovery_at)
            if reclaim and reclaim_cadence.next_run_at is not None:
                wake_deadlines.append(reclaim_cadence.next_run_at)
            return DurableWorkerStep(
                handled=handled_this_step,
                idle=True,
                next_wake_at=min(wake_deadlines) if wake_deadlines else None,
                activity=meaningful_activity,
            )
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
                lambda task_id=task_id, contract=contract, lease_expires_at=lease_expires_at: (
                    task_store.hold_claimed_work_contract_task(
                        task_id,
                        worker_id=worker_id,
                        lease_expires_at=lease_expires_at,
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
            await _handle_with_heartbeat(app, task_store, task, handler, worker_id, lease_seconds)
        finally:
            poller.metrics.handler_finished()
        next_interrupted_continuation_scan_at = 0.0
        return DurableWorkerStep(
            handled=handled_this_step + 1,
            continue_immediately=True,
            activity=True,
        )

    subscribe_to_admissions = getattr(task_store, "_task_admission_wakeup", None)
    admission_wakeup = None
    try:
        admission_wakeup = (
            None
            if not callable(subscribe_to_admissions)
            else await subscribe_to_admissions((query,))
        )
        return await run_durable_worker_loop(
            run_step,
            poll_interval_s=poll_interval_s,
            stop=stop,
            max_handled=max_tasks,
            wait=(_wait_or_stop if admission_wakeup is None else admission_wakeup.wait_for_worker),
            demand_policy=demand_policy,
            poller=poller,
            metrics=metrics if metrics is not None else poller.metrics,
        )
    finally:
        if admission_wakeup is not None:
            admission_wakeup.close()
        poller.close()


async def _handle_with_heartbeat(
    app: CayuApp,
    task_store: TaskStore,
    task: Task,
    handler: TaskHandler,
    worker_id: str,
    lease_seconds: int,
    *,
    recover_interrupted_authority_loss: bool = False,
) -> None:
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
            handoff_id=task.interrupted_handoff_id,
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
                current = await task_store.load_task(task.id)
                if current is None:
                    raise RuntimeError(
                        "Task cancellation lost its store-authoritative task after handler "
                        "quiescence."
                    )
                request = _task_cancellation_terminalization_request(
                    current,
                    worker_id=worker_id,
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
                    )
                    if retry_cancellation_requested
                    else await _enforce_task_retry_deadline_resisting_cancellation(
                        task_store,
                        task,
                        worker_id,
                        lease_expires_at=lease_authority.lease_expires_at,
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
                task,
                worker_id,
            )
            return

        async def finalize_handler_outcome() -> None:
            nonlocal handler_error

            if handler_error is not None:
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
                failure_payload = _task_failure_payload(app, handler_error)
                handler_error = None
                await _safe_fail_payload(
                    task_store,
                    task.id,
                    worker_id,
                    failure_payload,
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
                await _handoff_interrupted_session(app, task_store, task.id, worker_id)
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
            async with lease_authority.lock:
                await finalize_handler_outcome()
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
) -> tuple[Task, float, bool, bool]:
    """Prove that an acknowledged lease can still authorize handler dispatch."""

    renewal_started = monotonic()
    renewed = await task_store.heartbeat(
        task.id,
        worker_id,
        lease_expires_at=_require_task_lease_expires_at(task),
        handoff_id=task.interrupted_handoff_id,
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
            lease_expires_at=_require_task_lease_expires_at(renewed),
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
                lease_expires_at=_require_task_lease_expires_at(renewed),
                handoff_id=renewed.interrupted_handoff_id,
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
        execution_lease_expires_at = renewed.lease_expires_at
        renewed = await task_store.mark_claimed_task_execution_started(
            renewed.id,
            worker_id,
            execution_lease_expires_at,
        )
        if monotonic() >= claim_deadline_monotonic:
            renewed = await task_store.request_claimed_task_cancellation(
                renewed.id,
                worker_id,
                execution_lease_expires_at,
                {"code": "task_execution_start_acknowledgement_consumed_lease"},
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
) -> TaskHandlerOutcome | TaskRetryAttemptReport | None:
    with bind_task_lease_authority(lease_authority):
        handler_task = asyncio.ensure_future(handler(app, task, worker_id))
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
                        cancellation_baseline=cancellation_baseline,
                        primary_failure=heartbeat_failure,
                    )
                raise
        if handler_task in completed and heartbeat_result not in {
            _TaskHeartbeatOutcome.ELAPSED,
            _TaskHeartbeatOutcome.CANCELLATION_REQUESTED,
        }:
            if task.retry_series is not None and not heartbeat_task.done():
                return await handler_task
            stop_heartbeat.set()
            try:
                final_heartbeat_result = await heartbeat_task
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
                        cancellation_baseline=cancellation_baseline,
                        primary_failure=heartbeat_failure,
                    )
                raise
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
                raise _TaskRetryElapsed(
                    quiescence.owner_cancellation,
                    quiescence.report,
                )
            if final_heartbeat_result is _TaskHeartbeatOutcome.CANCELLATION_REQUESTED:
                quiescence = await _quiesce_task_retry_handler_before_terminalization(
                    task_store,
                    task.id,
                    worker_id,
                    lease_seconds,
                    handler_task,
                    cancellation_baseline,
                    lease_authority,
                )
                raise _TaskRetryCancellationRequested(
                    quiescence.owner_cancellation,
                    quiescence.report,
                )
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
            )
            raise _TaskRetryCancellationRequested(
                quiescence.owner_cancellation,
                quiescence.report,
            )
        return await handler_task
    except BaseExceptionGroup as grouped_failure:
        if task.retry_series is None or exception_tree_contains(
            grouped_failure,
            (KeyboardInterrupt, SystemExit, GeneratorExit),
        ):
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
        await _drain_ordinary_task_handler_under_cancellation_fence(
            task_store,
            task,
            worker_id,
            handler_task,
            heartbeat_task,
            stop_heartbeat,
            lease_authority=lease_authority,
            cancellation_error={"code": "task_worker_owner_cancelled"},
            primary_failure=owner_cancellation,
            owner_cancellation=owner_cancellation,
        )
        raise owner_cancellation from exception_cause(owner_cancellation)


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
                lease_expires_at=lease_authority.lease_expires_at,
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
    cancellation_baseline: int,
    cancellation_error: dict[str, Any] | None = None,
    primary_failure: BaseException,
    owner_cancellation: asyncio.CancelledError | None = None,
) -> None:
    """Fence a retry attempt before naturally draining cancellation-opaque work."""

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
    cancellation_error: dict[str, Any],
    primary_failure: BaseException,
    owner_cancellation: asyncio.CancelledError | None = None,
) -> None:
    """Fence an ordinary task durably until its handler is positively quiescent."""

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

        stop_heartbeat.set()
        if not heartbeat_task.done():
            try:
                await heartbeat_task
            except (GeneratorExit, KeyboardInterrupt, SystemExit):
                raise
            except BaseException as exc:
                failures.append(exc)

        if cancellation_requested:
            try:
                await _settle_ordinary_task_cancellation_after_quiescence(
                    task_store,
                    task.id,
                    worker_id,
                )
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

    async with lease_authority.lock:
        while True:
            expected_lease_expires_at = lease_authority.lease_expires_at
            current = await task_store.load_task(task.id)
            if current is None:
                raise TaskClaimLost("Task cancellation fence target no longer exists.")
            if (
                current.status
                in {
                    TaskStatus.COMPLETED,
                    TaskStatus.FAILED,
                    TaskStatus.CANCELLED,
                }
                or _task_cancellation_requested(current)
                or _task_retry_cancellation_requested(current)
            ):
                return current
            if (
                current.worker_id != worker_id
                or current.lease_expires_at != expected_lease_expires_at
            ):
                raise TaskClaimLost(
                    "Task cancellation fence no longer owns the expected lease generation."
                )
            try:
                requested = await task_store.request_claimed_task_cancellation(
                    task.id,
                    worker_id,
                    expected_lease_expires_at,
                    cancellation_error,
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
                if (
                    current.status
                    in {
                        TaskStatus.COMPLETED,
                        TaskStatus.FAILED,
                        TaskStatus.CANCELLED,
                    }
                    or _task_cancellation_requested(current)
                    or _task_retry_cancellation_requested(current)
                ):
                    return current
                if (
                    current.worker_id != worker_id
                    or current.lease_expires_at != expected_lease_expires_at
                ):
                    raise TaskClaimLost(
                        "Task cancellation fence lost its exact worker lease."
                    ) from request_failure
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
) -> None:
    """Settle the exact cancellation after this runtime observed natural quiescence."""

    current = await task_store.load_task(task_id)
    if current is None or current.status in {
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    }:
        return
    if current.worker_id != worker_id:
        # A peer claim may have won immediately before the cancellation marker.
        # That peer now observes and owns the same durable cancellation request.
        return
    terminalization = _task_cancellation_terminalization_request(
        current,
        worker_id=worker_id,
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
    if current is None or current.status in {
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    }:
        return
    if (
        current is None
        or current.worker_id != worker_id
        or current.lease_expires_at is None
        or not _task_cancellation_requested(current)
    ):
        return

    terminalization = _task_cancellation_terminalization_request(
        current,
        worker_id=worker_id,
    )
    if terminalization is None:  # pragma: no cover - guarded above
        raise AssertionError("Task cancellation request disappeared during reconciliation.")
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
    request = TaskCancellationReconciliationRequest(
        task_id=current.id,
        original_worker_id=worker_id,
        original_lease_expires_at=current.lease_expires_at,
        cancellation_requested_at=cancellation_requested_at,
        cancellation_idempotency_key=terminalization.idempotency_key,
        reconciliation_idempotency_key=(f"task-worker-quiescence:v1:{evidence_sha256}"),
        reconciliation_requested_at=quiescence_validated_at,
        reconciled_by=ResolutionActor(
            subject="cayu:task-worker-quiescence",
            source=ResolutionActorSource.SYSTEM,
        ),
        evidence=TaskCancellationReconciliationEvidence(
            outcome=TaskCancellationReconciliationOutcome.QUIESCENT,
            validator_id="cayu.task-worker",
            validator_version="1",
            evidence_id=f"task-worker-quiescence:{evidence_sha256}",
            evidence_sha256=evidence_sha256,
            validated_at=quiescence_validated_at,
            execution_profile_fingerprint=execution_profile_fingerprint,
            effect_fingerprint=effect_fingerprint,
        ),
        expected_execution_profile_fingerprint=execution_profile_fingerprint,
        expected_effect_fingerprint=effect_fingerprint,
    )
    await task_store.reconcile_task_cancellation(request)


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
) -> _TaskRetryQuiescence:
    """Retain the claim until the handler naturally settles."""

    interval = min(lease_seconds / 3, 1.0)
    owner_task = asyncio.current_task()
    if owner_task is None:  # pragma: no cover - coroutine execution invariant
        raise RuntimeError("Task retry handler quiescence requires an owning task.")
    owner_cancellation: asyncio.CancelledError | None = None
    lease_maintenance_failure: BaseException | None = None
    while not handler_task.done():
        try:
            completed, _pending = await asyncio.wait((handler_task,), timeout=interval)
            if handler_task in completed:
                break
            renewed = await task_store.heartbeat(
                task_id,
                worker_id,
                lease_expires_at=lease_authority.lease_expires_at,
                handoff_id=lease_authority.handoff_id,
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
            # The durable cancellation request keeps replacement disabled. Do
            # not cancel an opaque wrapper: its delegated thread, subprocess,
            # adapter, or remote call may still be running after the wrapper
            # reports cancellation.
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
    task_id: str,
    worker_id: str,
) -> None:
    task = await task_store.load_task(task_id)
    if task is None or task.status in {
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    }:
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
            lease_expires_at=task.lease_expires_at,
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
            lease_expires_at=task.lease_expires_at,
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
            lease_expires_at=task.lease_expires_at,
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
            lease_expires_at=task.lease_expires_at,
        )
        return
    request = interrupted_task_handoff_request(
        task,
        session_run_epoch=session.run_epoch,
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
        current = await task_store.load_task(task.id)
        if current is None or current.status in {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }:
            return
        cancellation = _task_cancellation_terminalization_request(
            current,
            worker_id=worker_id,
        )
        if cancellation is not None:
            await _terminalize_claimed_task_or_detect_peer_winner(
                task_store,
                cancellation,
            )
            return
        raise


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
        and task.lease_expires_at == request.lease_expires_at
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
            },
        )
    )
    event = event_with_runtime_envelope_authority(event, "session_id")
    payload_authority = ["handoff_id"]
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


async def _recover_expired_interrupted_task_handoffs(
    app: CayuApp,
    task_store: TaskStore,
    *,
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
    next_after = after
    for task in candidates:
        if _is_stopped(stop):
            return _InterruptedHandoffRecoveryPage(
                recovered=recovered,
                next_after=next_after,
                exhausted=False,
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
        if session.status in {
            SessionStatus.PENDING,
            SessionStatus.RUNNING,
            SessionStatus.INTERRUPTING,
        }:
            try:
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
                continue
            session = await app.session_store.load(task.session_id)
            if session is None or session.instance_id != task.session_instance_id:
                continue
        if session.status is not SessionStatus.INTERRUPTED:
            continue
        request = interrupted_task_handoff_request(
            task,
            session_run_epoch=session.run_epoch,
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
                or current.lease_expires_at != request.lease_expires_at
            ):
                continue
            raise
        recovered += 1
    return _InterruptedHandoffRecoveryPage(
        recovered=recovered,
        next_after=next_after,
        exhausted=len(candidates) < limit,
    )


async def _run_recovered_interrupted_task_handler(
    app: CayuApp,
    task_store: TaskStore,
    handler: RecoveredInterruptedTaskHandler,
    *,
    worker_id: str,
    lease_seconds: int,
    query: TaskQuery | None,
    after: tuple[datetime, str] | None,
) -> InterruptedTaskContinuationClaimPage:
    """Scan one bounded page and supervise its optional continuation claim."""

    handoff_id = new_interrupted_task_continuation_handoff_id()
    claim_outcome = await capture_task_store_operation(
        lambda: task_store.claim_interrupted_task_continuation(
            worker_id,
            query,
            handoff_id=handoff_id,
            lease_seconds=lease_seconds,
            after=after,
            scan_limit=_INTERRUPTED_HANDOFF_RECOVERY_BATCH_SIZE,
        ),
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
        return page
    await _handle_with_heartbeat(
        app,
        task_store,
        task,
        handler,
        worker_id,
        lease_seconds,
        recover_interrupted_authority_loss=True,
    )
    return page


async def _settle_recovered_continuation_authority_loss(
    app: CayuApp,
    task_store: TaskStore,
    claimed: Task,
    worker_id: str,
) -> bool:
    """Release a still-interrupted continuation or defer it to lease recovery."""

    current = await task_store.load_task(claimed.id)
    if current is None or current.status in {
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    }:
        return True
    if (
        current.status is not TaskStatus.RUNNING
        or current.worker_id != worker_id
        or current.lease_expires_at is None
        or current.interrupted_handoff_id != claimed.interrupted_handoff_id
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
        await _handoff_interrupted_session(
            app,
            task_store,
            current.id,
            worker_id,
        )
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
        or current.lease_expires_at != lease_authority.lease_expires_at
        or current.interrupted_handoff_id != lease_authority.handoff_id
    ):
        raise TaskClaimLost(
            "Pending runtime task failure no longer owns the worker lease."
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
    handoff_id: str | None = None,
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
            handoff_id=handoff_id,
            claim_deadline_monotonic=current_deadline(),
        )
        return updated

    async def inspect_heartbeat(updated: Task | None) -> _TaskHeartbeatOutcome | None:
        if updated is not None and (
            _task_retry_cancellation_requested(updated) or _task_cancellation_requested(updated)
        ):
            return _TaskHeartbeatOutcome.CANCELLATION_REQUESTED
        if enforce_retry_deadline and await task_store.task_retry_deadline_elapsed(
            task_id,
            worker_id,
            lease_expires_at=lease_authority.lease_expires_at,
        ):
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
            reconciliation_task.add_done_callback(_consume_task_heartbeat_outcome)
            raise
        if reconciliation_task not in completed:
            reconciliation_task.add_done_callback(_consume_task_heartbeat_outcome)
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
    handoff_id: str | None,
    claim_deadline_monotonic: float,
) -> tuple[Task | None, float]:
    """Serialize one renewal with mutations that consume the exact lease token."""

    async with lease_authority.lock:
        renewal_started = monotonic()
        renewal_task = asyncio.create_task(
            task_store.heartbeat(
                task_id,
                worker_id,
                lease_expires_at=lease_authority.lease_expires_at,
                handoff_id=handoff_id,
                extend_seconds=lease_seconds,
            )
        )
        try:
            completed, _pending = await asyncio.wait(
                {renewal_task},
                timeout=max(0.0, claim_deadline_monotonic - monotonic()),
            )
        except asyncio.CancelledError:
            renewal_task.add_done_callback(_consume_task_heartbeat_outcome)
            raise
        if renewal_task not in completed:
            renewal_task.add_done_callback(_consume_task_heartbeat_outcome)
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


def _consume_task_heartbeat_outcome(task: asyncio.Task[Any]) -> None:
    """Observe a store heartbeat that outlived this worker's local authority."""

    with contextlib.suppress(BaseException):
        task.result()


async def _enforce_task_retry_deadline_with_retry(
    task_store: TaskStore,
    task: Task,
    worker_id: str,
    *,
    lease_expires_at: datetime,
    token_count: int = 0,
    estimated_cost: Decimal = Decimal(0),
) -> TaskRetrySettlementResult | None:
    """Enforce or authenticate a deadline commit whose acknowledgement was lost."""

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
                lease_expires_at=lease_expires_at,
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
    lease_expires_at: datetime,
    cancellation_baseline: int,
    owner_cancellation: asyncio.CancelledError | None,
    report: TaskRetryAttemptReport | None,
) -> tuple[TaskRetrySettlementResult | None, asyncio.CancelledError | None]:
    """Finish terminal ownership before redelivering caller cancellation."""

    return await _complete_task_retry_terminalization_resisting_cancellation(
        _enforce_task_retry_deadline_with_retry(
            task_store,
            task,
            worker_id,
            lease_expires_at=lease_expires_at,
            token_count=0 if report is None else report.token_count,
            estimated_cost=Decimal(0) if report is None else report.estimated_cost,
        ),
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
) -> tuple[TaskRetrySettlementResult | None, asyncio.CancelledError | None]:
    """Settle an operator cancellation without releasing its live owner early."""

    async def settle() -> TaskRetrySettlementResult:
        task = await task_store.load_task(task_id)
        if task is None:
            raise RuntimeError("Task retry cancellation lost its durable attempt.")
        request = _task_retry_requested_cancellation_settlement(
            task,
            worker_id=worker_id,
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
        task_store,
        task_id,
        worker_id,
        payload,
        lease_expires_at=lease_expires_at,
        lease_authority=lease_authority,
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
    if diagnostic.durable_value_error_code is not None:
        payload["durable_value_error_code"] = diagnostic.durable_value_error_code
        payload["durable_value_error_path"] = diagnostic.durable_value_error_path
    diagnostic = None
    redacted_payload = app.redact_json(payload)
    payload.clear()
    if type(redacted_payload) is not dict:
        raise AssertionError("Task failure payload redaction returned a non-object.")
    return {
        key: _redact_and_bound_task_failure_text(app, value) if type(value) is str else value
        for key, value in redacted_payload.items()
    }


def _redact_and_bound_task_failure_text(app: CayuApp, value: str) -> str:
    redacted_value = app.redact_json(value)
    if type(redacted_value) is not str:
        raise AssertionError("Task failure text redaction returned a non-string.")
    encoded_value = redacted_value.encode("utf-8", "replace")
    if len(encoded_value) <= _MAX_TASK_FAILURE_MESSAGE_BYTES:
        return redacted_value
    return encoded_value[:_MAX_TASK_FAILURE_MESSAGE_BYTES].decode("utf-8", "ignore")


async def _safe_fail_payload(
    task_store: TaskStore,
    task_id: str,
    worker_id: str,
    payload: dict[str, Any],
    *,
    lease_expires_at: datetime | None = None,
    lease_authority: _TaskLeaseAuthority | None = None,
) -> None:
    task = await task_store.load_task(task_id)
    if task is None or task.status not in {TaskStatus.CLAIMED, TaskStatus.RUNNING}:
        return
    if task.retry_series is not None:
        request = TaskRetrySettlementRequest(
            task_id=task_id,
            worker_id=worker_id,
            lease_expires_at=lease_expires_at,
            causal_budget_id=task.retry_series.causal_budget_id,
            disposition=TaskRetryAttemptDisposition.NON_RETRYABLE_FAILURE,
            error=payload,
            idempotency_key=_worker_failure_terminalization_key(
                task_id,
                worker_id,
            ),
        )
        try:
            await (
                _settle_task_retry_request_with_lease_authority(
                    task_store,
                    request,
                    lease_authority,
                )
                if lease_authority is not None
                else _settle_task_retry_request_honoring_cancellation(task_store, request)
            )
        except TaskClaimLost:
            return
        return
    try:
        peer_terminalization_won = await _terminalize_claimed_task_or_detect_peer_winner(
            task_store,
            TaskTerminalizationRequest(
                task_id=task_id,
                worker_id=worker_id,
                lease_expires_at=lease_expires_at,
                kind=TaskTerminalKind.FAILED,
                error=payload,
                idempotency_key=_worker_failure_terminalization_key(
                    task_id,
                    worker_id,
                ),
            ),
        )
        if peer_terminalization_won:
            return
    except TaskClaimLost:
        return
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
            return
        raise


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
            lease_expires_at=lease_authority.lease_expires_at,
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
    """Settle against the latest acknowledged lease without stopping heartbeats."""

    last_claim_loss: TaskClaimLost | None = None
    for _attempt in range(3):
        effective_lease = lease_authority.lease_expires_at
        effective_request = request.model_copy(
            update={"lease_expires_at": effective_lease},
            deep=True,
        )
        try:
            return await _settle_task_retry_request_honoring_cancellation(
                task_store,
                effective_request,
            )
        except TaskClaimLost as exc:
            last_claim_loss = exc

        receipt = await task_store.load_task_retry_settlement(
            effective_request.task_id,
            effective_request.idempotency_key,
        )
        if receipt is not None:
            return await _settle_task_retry_request_honoring_cancellation(
                task_store,
                effective_request,
            )
        if lease_authority.lease_expires_at == effective_lease:
            raise last_claim_loss

    assert last_claim_loss is not None
    raise last_claim_loss


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
            token_count=request.token_count,
            estimated_cost=request.estimated_cost,
        )
        if cancellation is None:
            raise
        return await settle_task_retry_attempt_with_retry(task_store, cancellation)


def _worker_failure_terminalization_key(task_id: str, worker_id: str) -> str:
    identity = canonical_durable_json_bytes(
        {
            "schema": "cayu.task-worker-failure.v1",
            "task_id": task_id,
            "worker_id": worker_id,
        },
        "task_worker_failure",
    )
    return f"task-worker-failure:v1:{sha256(identity).hexdigest()}"


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


async def _wait_or_stop(seconds: float, stop: asyncio.Event | None) -> bool:
    """Sleep for ``seconds`` or until ``stop`` is set. Returns True if stopped."""
    return await wait_or_stop(seconds, stop)
