"""A generic durable task-worker loop for starting fresh sessions from tasks.

``TaskStoreDispatcher.run_worker`` resumes existing sessions from dispatch
requests. This helper covers the complementary shape used by, for example, the
PR-reviewer recipe: a worker that claims arbitrary :class:`Task`\\ s and starts a
*new* session for each. It owns the claim -> heartbeat -> handle -> loop cycle
plus optional expired-lease reclaim, so a caller only supplies a handler that
turns a claimed task into an ``app.run(...)``.

The handler owns the task's terminal state. Legacy tasks may run with
``RunRequest(task_id=..., task_worker_id=...)`` so the runtime completes/fails the
task, or call ``task_store.complete_task``/``fail_task`` explicitly. Retry-series
tasks instead return an explicit :class:`TaskRetryAttemptReport`; generic runtime
failures are never inferred to be retryable. A handler whose attached session
durably stops at an interrupted continuation boundary may return
``TaskHandlerOutcome.SESSION_INTERRUPTED`` to release worker ownership while a
control-plane process waits to resume that session. If the handler raises or
returns ``None`` while the task is still active, the worker marks the task failed
and keeps going -- one bad task does not kill it.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
from math import isfinite
from typing import TYPE_CHECKING, Any

from cayu._exception_groups import (
    exception_cause,
    exception_tree_contains,
    iter_exception_tree,
    set_exception_cause,
)
from cayu._validation import canonical_durable_json_bytes, require_clean_nonblank
from cayu.runtime._task_store_operation_boundary import (
    capture_task_store_operation,
    raise_task_store_operation_failure,
    task_store_mutation_is_cancellation_quiescent,
)
from cayu.runtime.sessions import SessionStatus
from cayu.runtime.tasks import (
    Task,
    TaskClaimLost,
    TaskQuery,
    TaskRetryAttemptDisposition,
    TaskRetryAttemptReport,
    TaskRetrySeriesDisposition,
    TaskRetrySettlementRequest,
    TaskRetrySettlementResult,
    TaskStatus,
    TaskStore,
    TaskTerminalizationConflict,
    TaskTerminalizationRequest,
    TaskTerminalKind,
    _task_retry_cancellation_requested,
    _task_retry_requested_cancellation_settlement,
    _task_retry_runtime_terminal_request,
    _terminalize_claimed_task_or_detect_peer_winner,
    copy_task,
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
_MAX_TASK_FAILURE_MESSAGE_BYTES = 500
_TASK_RETRY_HANDLER_CANCELLATION_GRACE_SECONDS = 1.0


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
    reclaim: bool = True,
    stop: asyncio.Event | None = None,
    max_tasks: int | None = None,
) -> int:
    """Claim and handle durable tasks until stopped; return the number handled.

    For each claimed task, ``handler(app, task, worker_id)`` is awaited while the
    task lease is heartbeated in the background. The handler typically builds a
    ``RunRequest(task_id=task.id, task_worker_id=worker_id, ...)`` and awaits
    ``app.run(...)`` so the runtime completes/fails the task. If that run ends at
    a durable interrupted boundary, return ``TaskHandlerOutcome.SESSION_INTERRUPTED``
    so the helper validates the session and releases only the task's worker lease.

    - ``query`` scopes which tasks this worker claims (e.g. by type / assigned agent).
    - ``lease_seconds`` is the claim lease; the lease is re-extended at ~1/3 of it.
    - ``poll_interval_s`` is how long to wait when no task is available.
    - ``reclaim`` reclaims expired leases (from dead workers) before each claim.
    - ``stop`` is an ``asyncio.Event`` for graceful shutdown.
    - ``max_tasks`` bounds the loop (useful for tests and one-shot drains).
    """
    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive.")
    if not isfinite(poll_interval_s) or poll_interval_s <= 0:
        raise ValueError("poll_interval_s must be finite and positive.")
    if max_tasks is not None and max_tasks < 0:
        raise ValueError("max_tasks must be non-negative.")
    worker_id = require_clean_nonblank(worker_id, "worker_id")
    if app.redact_json(worker_id) != worker_id:
        raise ValueError(
            "worker_id contains a workload secret and cannot be used as durable task authority."
        )
    if max_tasks == 0 or _is_stopped(stop):
        return 0
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

    handled = 0
    while (max_tasks is None or handled < max_tasks) and not _is_stopped(stop):
        if reclaim:
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
                del reclaim_outcome
            else:
                await task_store.reclaim_expired(query=query)
        if materialized_work_contract_queue_supported:
            claim_outcome = await capture_task_store_operation(
                lambda: task_store.claim_task(worker_id, query, lease_seconds=lease_seconds),
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
        else:
            task = await task_store.claim_task(
                worker_id,
                query,
                lease_seconds=lease_seconds,
            )
        if task is None:
            if await _wait_or_stop(poll_interval_s, stop):
                break
            continue
        task = copy_task(task)
        if task.work_contract is not None:
            task_id = task.id
            contract = task.work_contract
            parking_outcome = await capture_task_store_operation(
                lambda task_id=task_id, contract=contract: (
                    task_store.hold_claimed_work_contract_task(
                        task_id,
                        worker_id=worker_id,
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
                del contract, parking_outcome, task, task_id
                raise_task_store_operation_failure(failure)
            del contract, parking_outcome, task_id
            handled += 1
            continue
        await _handle_with_heartbeat(app, task_store, task, handler, worker_id, lease_seconds)
        handled += 1
    return handled


async def _handle_with_heartbeat(
    app: CayuApp,
    task_store: TaskStore,
    task: Task,
    handler: TaskHandler,
    worker_id: str,
    lease_seconds: int,
) -> None:
    owner_task = asyncio.current_task()
    if owner_task is None:  # pragma: no cover - coroutine execution invariant
        raise RuntimeError("Task worker handler execution requires an owning task.")
    cancellation_baseline = owner_task.cancelling()
    stop_heartbeat = asyncio.Event()
    heartbeat_task = asyncio.create_task(
        _heartbeat_until(
            task_store,
            task.id,
            worker_id,
            lease_seconds,
            stop_heartbeat,
            enforce_retry_deadline=(
                task.retry_series is not None and task.retry_series.elapsed_deadline is not None
            ),
        )
    )
    handler_error: Exception | None = None
    handler_outcome: TaskHandlerOutcome | TaskRetryAttemptReport | None = None
    retry_elapsed = False
    retry_cancellation_requested = False
    retry_elapsed_cancellation: asyncio.CancelledError | None = None
    retry_terminal_report: TaskRetryAttemptReport | None = None
    disposition_cancellation: asyncio.CancelledError | None = None
    heartbeat_authority_failure: Exception | None = None
    ownership_settled = False
    pending_failure: BaseException | None = None
    try:
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
            )
        except _TaskRetryElapsed as exc:
            retry_elapsed = True
            retry_elapsed_cancellation = exc.owner_cancellation
            retry_terminal_report = exc.report
        except _TaskRetryCancellationRequested as exc:
            retry_cancellation_requested = True
            retry_elapsed_cancellation = exc.owner_cancellation
            retry_terminal_report = exc.report
        except _TaskRetryHandlerUnsettled:
            raise
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
            handler_error = exc

        if retry_elapsed or retry_cancellation_requested:
            terminalization_label = "cancellation" if retry_cancellation_requested else "deadline"
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

        async def finalize_handler_outcome() -> None:
            nonlocal handler_error

            if handler_error is not None:
                failure_payload = _task_failure_payload(app, handler_error)
                handler_error = None
                await _safe_fail_payload(task_store, task.id, worker_id, failure_payload)
            elif type(handler_outcome) is TaskRetryAttemptReport:
                if task.retry_series is None:
                    await _safe_fail(
                        app,
                        task_store,
                        task.id,
                        worker_id,
                        TypeError("TaskRetryAttemptReport requires a task retry policy."),
                    )
                else:
                    await _settle_retry_attempt_report(
                        task_store,
                        task,
                        worker_id,
                        handler_outcome,
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
                )
            else:
                await _safe_fail_unfinished(app, task_store, task.id, worker_id)

        if task.retry_series is None:
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
) -> TaskHandlerOutcome | TaskRetryAttemptReport | None:
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
            except Exception:
                await _cancel_task_handler_after_authority_loss(handler_task)
                raise
        if handler_task in completed and heartbeat_result not in {
            _TaskHeartbeatOutcome.ELAPSED,
            _TaskHeartbeatOutcome.CANCELLATION_REQUESTED,
        }:
            if task.retry_series is not None and not heartbeat_task.done():
                return await handler_task
            stop_heartbeat.set()
            final_heartbeat_result = await heartbeat_task
            if final_heartbeat_result is _TaskHeartbeatOutcome.ELAPSED:
                quiescence = await _quiesce_task_retry_handler_before_terminalization(
                    task_store,
                    task.id,
                    worker_id,
                    lease_seconds,
                    handler_task,
                    cancellation_baseline,
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
                cancellation_baseline=cancellation_baseline,
                owner_cancellation=owner_cancellation,
            )
            raise AssertionError(
                "Task retry cancellation handling returned unexpectedly."
            ) from None
        handler_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await handler_task
        raise


async def _raise_task_retry_owner_cancellation(
    task_store: TaskStore,
    task: Task,
    worker_id: str,
    lease_seconds: int,
    handler_task: asyncio.Future[Any],
    heartbeat_task: asyncio.Task[_TaskHeartbeatOutcome],
    stop_heartbeat: asyncio.Event,
    *,
    cancellation_baseline: int,
    owner_cancellation: asyncio.CancelledError,
) -> None:
    """Settle or release one retry attempt before propagating owner cancellation."""

    try:
        quiescence = await _quiesce_task_retry_handler_before_terminalization(
            task_store,
            task.id,
            worker_id,
            lease_seconds,
            handler_task,
            cancellation_baseline,
        )
    except BaseExceptionGroup as grouped_failure:
        if exception_tree_contains(
            grouped_failure,
            (KeyboardInterrupt, SystemExit, GeneratorExit),
        ):
            raise
        _attach_grouped_task_handler_failures_to_cancellation(
            owner_cancellation,
            grouped_failure,
        )
        quiescence = _TaskRetryQuiescence(None, None)
    if quiescence.owner_cancellation is not None:
        owner_cancellation.add_note(
            "Task retry handler quiescence received an additional caller cancellation request."
        )
        quiescence_failure = quiescence.owner_cancellation.__cause__
        if quiescence_failure is not None:
            _attach_task_worker_secondary_failure(
                owner_cancellation,
                quiescence_failure,
                label="Task retry handler-quiescence failures",
            )
    stop_heartbeat.set()
    heartbeat_failure: Exception | None = None
    heartbeat_outcome: _TaskHeartbeatOutcome | None = None
    try:
        heartbeat_outcome = await _await_task_retry_cleanup_resisting_cancellation(
            heartbeat_task,
            label="heartbeat shutdown",
            owner_cancellation=owner_cancellation,
        )
    except Exception as exc:
        heartbeat_failure = exc
    if heartbeat_outcome is _TaskHeartbeatOutcome.ELAPSED:
        raise _TaskRetryElapsed(owner_cancellation, quiescence.report) from None
    if heartbeat_outcome is _TaskHeartbeatOutcome.CANCELLATION_REQUESTED:
        raise _TaskRetryCancellationRequested(
            owner_cancellation,
            quiescence.report,
        ) from None
    if heartbeat_outcome is _TaskHeartbeatOutcome.TERMINAL:
        _raise_task_retry_owner_cancellation_with_secondary(
            owner_cancellation,
            heartbeat_failure,
        )
    if quiescence.report is not None:
        try:
            await _await_task_retry_cleanup_resisting_cancellation(
                _settle_retry_attempt_report(
                    task_store,
                    task,
                    worker_id,
                    quiescence.report,
                ),
                label="shutdown settlement",
                owner_cancellation=owner_cancellation,
            )
        except Exception as settlement_failure:
            secondary_failure: BaseException = settlement_failure
            if heartbeat_failure is not None:
                secondary_failure = BaseExceptionGroup(
                    "Task retry shutdown-settlement failures",
                    [heartbeat_failure, settlement_failure],
                )
            _raise_task_retry_owner_cancellation_with_secondary(
                owner_cancellation,
                secondary_failure,
            )
        _raise_task_retry_owner_cancellation_with_secondary(
            owner_cancellation,
            heartbeat_failure,
        )
    try:
        await _await_task_retry_cleanup_resisting_cancellation(
            task_store.release_task(task.id, worker_id),
            label="owner release",
            owner_cancellation=owner_cancellation,
        )
    except Exception as release_failure:
        try:
            current = await _await_task_retry_cleanup_resisting_cancellation(
                task_store.load_task(task.id),
                label="owner-release reconciliation",
                owner_cancellation=owner_cancellation,
            )
        except Exception as reconciliation_failure:
            failures: list[BaseException] = [release_failure, reconciliation_failure]
            if heartbeat_failure is not None:
                failures.insert(0, heartbeat_failure)
            _raise_task_retry_owner_cancellation_with_secondary(
                owner_cancellation,
                BaseExceptionGroup(
                    "Task retry owner-release failures",
                    failures,
                ),
            )
        if current is not None and _task_retry_cancellation_requested(current):
            secondary_failures: list[BaseException] = []
            existing_cause = exception_cause(owner_cancellation)
            if existing_cause is not None:
                secondary_failures.append(existing_cause)
            if heartbeat_failure is not None:
                secondary_failures.append(heartbeat_failure)
            if not isinstance(release_failure, TaskTerminalizationConflict):
                secondary_failures.append(release_failure)
            if len(secondary_failures) == 1:
                set_exception_cause(owner_cancellation, secondary_failures[0])
            elif secondary_failures:
                set_exception_cause(
                    owner_cancellation,
                    BaseExceptionGroup(
                        "Task retry owner-release failures",
                        secondary_failures,
                    ),
                )
            raise _TaskRetryCancellationRequested(
                owner_cancellation,
                quiescence.report,
            ) from None
        if current is not None and current.status in {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }:
            _raise_task_retry_owner_cancellation_with_secondary(
                owner_cancellation,
                heartbeat_failure,
            )
        secondary_failure: BaseException = release_failure
        if heartbeat_failure is not None:
            secondary_failure = BaseExceptionGroup(
                "Task retry owner-release failures",
                [heartbeat_failure, release_failure],
            )
        _raise_task_retry_owner_cancellation_with_secondary(
            owner_cancellation,
            secondary_failure,
        )
    _raise_task_retry_owner_cancellation_with_secondary(
        owner_cancellation,
        heartbeat_failure,
    )


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


def _attach_grouped_task_handler_failures_to_cancellation(
    owner_cancellation: asyncio.CancelledError,
    grouped_failure: BaseExceptionGroup,
) -> None:
    secondary_leaves = [
        candidate
        for candidate in iter_exception_tree(grouped_failure)
        if not isinstance(candidate, (BaseExceptionGroup, asyncio.CancelledError))
    ]
    if not secondary_leaves:
        return
    secondary: BaseException = (
        secondary_leaves[0]
        if len(secondary_leaves) == 1
        else BaseExceptionGroup(
            "Task retry handler failures concurrent with cancellation",
            secondary_leaves,
        )
    )
    _attach_task_worker_secondary_failure(
        owner_cancellation,
        secondary,
        label="Task retry handler failures concurrent with cancellation",
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


async def _cancel_task_handler_after_authority_loss(
    handler_task: asyncio.Future[Any],
) -> None:
    handler_task.cancel()
    completed, _pending = await asyncio.wait(
        (handler_task,),
        timeout=_TASK_RETRY_HANDLER_CANCELLATION_GRACE_SECONDS,
    )
    if handler_task in completed:
        with contextlib.suppress(asyncio.CancelledError):
            await handler_task
        return
    handler_task.add_done_callback(_consume_detached_handler_outcome)
    raise _TaskRetryHandlerUnsettled(
        "Task retry handler did not stop after its store authority was lost."
    )


async def _quiesce_task_retry_handler_before_terminalization(
    task_store: TaskStore,
    task_id: str,
    worker_id: str,
    lease_seconds: int,
    handler_task: asyncio.Future[Any],
    cancellation_baseline: int,
) -> _TaskRetryQuiescence:
    """Retain the claim until the handler acknowledges cancellation and settles."""

    handler_task.cancel()
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
            await task_store.heartbeat(task_id, worker_id, extend_seconds=lease_seconds)
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
            handler_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await handler_task
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
            raise


def _consume_detached_handler_outcome(handler_task: asyncio.Future[Any]) -> None:
    with contextlib.suppress(BaseException):
        handler_task.result()


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
        )
        return

    session = await app.session_store.load_state(task.session_id)
    if session is None:
        await _safe_fail(
            app,
            task_store,
            task_id,
            worker_id,
            RuntimeError(f"Attached session not found: {task.session_id}."),
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
        )
        return

    release_failure_payload: dict[str, str] | None = None
    try:
        await task_store.release_attached_task_worker(task_id, worker_id)
    except Exception as exc:
        release_failure_payload = _task_failure_payload(app, exc)
    if release_failure_payload is not None:
        await _safe_fail_payload(
            task_store,
            task_id,
            worker_id,
            release_failure_payload,
        )


async def _heartbeat_until(
    task_store: TaskStore,
    task_id: str,
    worker_id: str,
    lease_seconds: int,
    stop: asyncio.Event,
    *,
    enforce_retry_deadline: bool = False,
) -> _TaskHeartbeatOutcome:
    interval = min(lease_seconds / 3, 1.0)
    while not stop.is_set():
        if await _wait_or_stop(interval, stop):
            return _TaskHeartbeatOutcome.STOPPED
        try:
            updated = await task_store.heartbeat(
                task_id,
                worker_id,
                extend_seconds=lease_seconds,
            )
            if updated is not None and _task_retry_cancellation_requested(updated):
                return _TaskHeartbeatOutcome.CANCELLATION_REQUESTED
            if enforce_retry_deadline and await task_store.task_retry_deadline_elapsed(
                task_id, worker_id
            ):
                return _TaskHeartbeatOutcome.ELAPSED
        except Exception as heartbeat_error:
            try:
                task = await task_store.load_task(task_id)
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
    return _TaskHeartbeatOutcome.STOPPED


async def _enforce_task_retry_deadline_with_retry(
    task_store: TaskStore,
    task: Task,
    worker_id: str,
    *,
    token_count: int = 0,
    estimated_cost: Decimal = Decimal(0),
) -> TaskRetrySettlementResult | None:
    """Enforce or authenticate a deadline commit whose acknowledgement was lost."""

    expected_request, expected_request_sha256 = _task_retry_runtime_terminal_request(
        task,
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
) -> None:
    payload = _task_failure_payload(app, exc)
    del exc
    await _safe_fail_payload(task_store, task_id, worker_id, payload)


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
) -> None:
    task = await task_store.load_task(task_id)
    if task is None or task.status not in {TaskStatus.CLAIMED, TaskStatus.RUNNING}:
        return
    if task.retry_series is not None:
        try:
            await _settle_task_retry_request_honoring_cancellation(
                task_store,
                TaskRetrySettlementRequest(
                    task_id=task_id,
                    worker_id=worker_id,
                    causal_budget_id=task.retry_series.causal_budget_id,
                    disposition=TaskRetryAttemptDisposition.NON_RETRYABLE_FAILURE,
                    error=payload,
                    idempotency_key=_worker_failure_terminalization_key(
                        task_id,
                        worker_id,
                    ),
                ),
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
) -> None:
    copied = TaskRetryAttemptReport.model_validate(report.model_dump(mode="python", warnings=False))
    series = task.retry_series
    if series is None:
        raise TypeError("TaskRetryAttemptReport requires a task retry policy.")
    await _settle_task_retry_request_honoring_cancellation(
        task_store,
        TaskRetrySettlementRequest(
            task_id=task.id,
            worker_id=worker_id,
            causal_budget_id=series.causal_budget_id,
            **copied.model_dump(mode="python", warnings=False),
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
    )


def _is_stopped(stop: asyncio.Event | None) -> bool:
    return stop is not None and stop.is_set()


async def _wait_or_stop(seconds: float, stop: asyncio.Event | None) -> bool:
    """Sleep for ``seconds`` or until ``stop`` is set. Returns True if stopped."""
    if stop is None:
        await asyncio.sleep(seconds)
        return False
    try:
        await asyncio.wait_for(stop.wait(), timeout=seconds)
        return True
    except TimeoutError:
        return False
