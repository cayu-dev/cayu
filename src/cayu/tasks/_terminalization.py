"""Bounded task terminalization retries and exact receipt reconciliation."""

from __future__ import annotations

import asyncio
from datetime import datetime

from cayu.tasks._cancellation import _task_cancellation_terminalization_request
from cayu.tasks.records import (
    _TERMINAL_TASK_STATUSES,
    Task,
    TaskClaimLost,
)
from cayu.tasks.retry import (
    TaskRetrySettlementRequest,
    TaskRetrySettlementResult,
    _replay_task_retry_settlement,
    _validate_task_retry_settlement_receipt_identity,
    prepare_task_retry_settlement,
)
from cayu.tasks.store import TaskStore
from cayu.tasks.terminalization import (
    TaskTerminalizationConflict,
    TaskTerminalizationReceipt,
    TaskTerminalizationRequest,
    TaskTerminalizationRetryPolicy,
    TaskTerminalizationRetryResult,
    TaskTerminalizationUncertain,
    TaskTerminalKind,
    _replay_task_terminalization_receipt,
    _task_terminalization_request_matches_sha256,
    prepare_task_terminalization,
)


async def terminalize_task_with_retry(
    task_store: TaskStore,
    request: TaskTerminalizationRequest,
    *,
    policy: TaskTerminalizationRetryPolicy | None = None,
) -> TaskTerminalizationRetryResult:
    """Terminalize once, reconciling only acknowledgement-ambiguous failures."""

    if not isinstance(task_store, TaskStore):
        raise TypeError("task_store must be a TaskStore instance.")
    if not task_store.supports_idempotent_terminalization:
        raise ValueError("task_store must support idempotent task terminalization and receipts.")
    request, request_sha256 = prepare_task_terminalization(request)
    if policy is None:
        policy = TaskTerminalizationRetryPolicy()
    elif type(policy) is not TaskTerminalizationRetryPolicy:
        raise TypeError("policy must be a TaskTerminalizationRetryPolicy instance.")
    else:
        policy = TaskTerminalizationRetryPolicy.model_validate(policy.model_dump(mode="python"))

    clock = asyncio.get_running_loop()
    started_at = clock.time()
    applied_backoff_seconds = 0.0
    delay = min(policy.initial_backoff_seconds, policy.max_backoff_seconds)
    last_error_category = "store_error"
    for attempt in range(1, policy.max_attempts + 1):
        attempt_request = TaskTerminalizationRequest.model_validate(
            request.model_dump(mode="python")
        )
        try:
            task = await asyncio.wait_for(
                task_store.terminalize_task(attempt_request),
                timeout=policy.attempt_timeout_seconds,
            )
            return TaskTerminalizationRetryResult(
                task=task,
                attempt_count=attempt,
                receipt_reconciled=False,
                elapsed_seconds=max(0.0, clock.time() - started_at),
                applied_backoff_seconds=applied_backoff_seconds,
            )
        except Exception as exc:
            if not _task_terminalization_error_is_acknowledgement_ambiguous(exc):
                raise
            last_error_category = _task_terminalization_error_category(exc)

        try:
            receipt = await asyncio.wait_for(
                task_store.load_task_terminalization_receipt(
                    request.task_id,
                    request.idempotency_key,
                ),
                timeout=policy.attempt_timeout_seconds,
            )
        except Exception as exc:
            if not _task_terminalization_error_is_acknowledgement_ambiguous(exc):
                raise
            last_error_category = _task_terminalization_error_category(exc)
            receipt = None

        if receipt is not None:
            if type(receipt) is not TaskTerminalizationReceipt:
                raise TypeError(
                    "Task terminalization receipt loads must return "
                    "TaskTerminalizationReceipt instances."
                )
            if (
                receipt.task_id != request.task_id
                or receipt.idempotency_key != request.idempotency_key
                or receipt.worker_id != request.worker_id
                or receipt.kind is not request.kind
                or not _task_terminalization_request_matches_sha256(
                    request,
                    request_sha256=request_sha256,
                    candidate_sha256=receipt.request_sha256,
                )
            ):
                raise TaskTerminalizationConflict(
                    "Task terminalization receipt conflicts with the retry request."
                )
            try:
                current_task = await asyncio.wait_for(
                    task_store.load_task(request.task_id),
                    timeout=policy.attempt_timeout_seconds,
                )
            except Exception as exc:
                if not _task_terminalization_error_is_acknowledgement_ambiguous(exc):
                    raise
                last_error_category = _task_terminalization_error_category(exc)
            else:
                if current_task is not None and type(current_task) is not Task:
                    raise TypeError("Task loads must return Task instances.")
                reconciled_task = _replay_task_terminalization_receipt(
                    request=request,
                    request_sha256=request_sha256,
                    receipt=receipt,
                    current_task=current_task,
                )
                return TaskTerminalizationRetryResult(
                    task=reconciled_task,
                    attempt_count=attempt,
                    receipt_reconciled=True,
                    elapsed_seconds=max(0.0, clock.time() - started_at),
                    applied_backoff_seconds=applied_backoff_seconds,
                )

        if attempt == policy.max_attempts:
            raise TaskTerminalizationUncertain(
                task_id=request.task_id,
                idempotency_key=request.idempotency_key,
                attempt_count=attempt,
                error_category=last_error_category,
                elapsed_seconds=max(0.0, clock.time() - started_at),
                applied_backoff_seconds=applied_backoff_seconds,
            )
        if delay > 0:
            await asyncio.sleep(delay)
            applied_backoff_seconds += delay
        delay = min(delay * policy.backoff_multiplier, policy.max_backoff_seconds)

    raise AssertionError("Task terminalization retry loop exited without an outcome.")


async def settle_task_retry_attempt_with_retry(
    task_store: TaskStore,
    request: TaskRetrySettlementRequest,
    *,
    policy: TaskTerminalizationRetryPolicy | None = None,
) -> TaskRetrySettlementResult:
    """Settle once, reconciling only acknowledgement-ambiguous store failures."""

    if not isinstance(task_store, TaskStore):
        raise TypeError("task_store must be a TaskStore instance.")
    if not task_store.supports_task_retry_series:
        raise ValueError("task_store must support atomic task retry-series settlement.")
    request, request_sha256 = prepare_task_retry_settlement(request)
    if policy is None:
        policy = TaskTerminalizationRetryPolicy()
    elif type(policy) is not TaskTerminalizationRetryPolicy:
        raise TypeError("policy must be a TaskTerminalizationRetryPolicy instance.")
    else:
        policy = TaskTerminalizationRetryPolicy.model_validate(
            policy.model_dump(mode="python", warnings=False)
        )

    loop = asyncio.get_running_loop()
    started_at = loop.time()
    applied_backoff_seconds = 0.0
    delay = min(policy.initial_backoff_seconds, policy.max_backoff_seconds)
    last_error_category = "store_error"
    for attempt in range(1, policy.max_attempts + 1):
        try:
            receipt = await asyncio.wait_for(
                task_store.settle_task_retry_attempt(
                    TaskRetrySettlementRequest.model_validate(
                        request.model_dump(mode="python", warnings=False)
                    )
                ),
                timeout=policy.attempt_timeout_seconds,
            )
            return _validate_task_retry_settlement_receipt_identity(
                receipt,
                request=request,
                request_sha256=request_sha256,
            )
        except Exception as exc:
            if not _task_terminalization_error_is_acknowledgement_ambiguous(exc):
                raise
            last_error_category = _task_terminalization_error_category(exc)

        try:
            receipt = await asyncio.wait_for(
                task_store.load_task_retry_settlement(
                    request.task_id,
                    request.idempotency_key,
                ),
                timeout=policy.attempt_timeout_seconds,
            )
        except Exception as exc:
            if not _task_terminalization_error_is_acknowledgement_ambiguous(exc):
                raise
            last_error_category = _task_terminalization_error_category(exc)
            receipt = None

        if receipt is not None:
            receipt = _validate_task_retry_settlement_receipt_identity(
                receipt,
                request=request,
                request_sha256=request_sha256,
            )
            try:
                current_task = await asyncio.wait_for(
                    task_store.load_task(request.task_id),
                    timeout=policy.attempt_timeout_seconds,
                )
            except Exception as exc:
                if not _task_terminalization_error_is_acknowledgement_ambiguous(exc):
                    raise
                last_error_category = _task_terminalization_error_category(exc)
            else:
                return _replay_task_retry_settlement(
                    request=request,
                    request_sha256=request_sha256,
                    receipt=receipt,
                    current_task=current_task,
                )

        if attempt == policy.max_attempts:
            raise TaskTerminalizationUncertain(
                task_id=request.task_id,
                idempotency_key=request.idempotency_key,
                attempt_count=attempt,
                error_category=last_error_category,
                elapsed_seconds=max(0.0, loop.time() - started_at),
                applied_backoff_seconds=applied_backoff_seconds,
            )
        if delay > 0:
            await asyncio.sleep(delay)
            applied_backoff_seconds += delay
        delay = min(delay * policy.backoff_multiplier, policy.max_backoff_seconds)

    raise AssertionError("Task retry settlement loop exited without an outcome.")


async def _terminalize_claimed_task(
    task_store: TaskStore,
    request: TaskTerminalizationRequest,
) -> Task:
    """Use receipt-safe terminalization when supported, with a legacy fallback."""

    if task_store.supports_idempotent_terminalization:
        return (await terminalize_task_with_retry(task_store, request)).task

    request, _request_sha256 = prepare_task_terminalization(request)
    if request.kind is TaskTerminalKind.CANCELLED:
        return await task_store.cancel_task(request.task_id, request.error)

    async def apply(lease_expires_at: datetime | None) -> Task:
        if request.kind is TaskTerminalKind.COMPLETED:
            if request.result is None:  # pragma: no cover - request model invariant
                raise AssertionError("Completed task terminalization requires a result.")
            return await task_store.complete_task(
                request.task_id,
                request.result,
                worker_id=request.worker_id,
                lease_expires_at=lease_expires_at,
                handoff_id=request.handoff_id,
            )
        if request.error is None:  # pragma: no cover - request model invariant
            raise AssertionError("Failed task terminalization requires an error.")
        return await task_store.fail_task(
            request.task_id,
            request.error,
            worker_id=request.worker_id,
            lease_expires_at=lease_expires_at,
            handoff_id=request.handoff_id,
        )

    if request.worker_id is None or request.lease_expires_at is not None:
        return await apply(request.lease_expires_at)

    current = await task_store.load_task(request.task_id)
    if (
        current is None
        or current.worker_id != request.worker_id
        or current.interrupted_handoff_id != request.handoff_id
        or current.lease_expires_at is None
    ):
        raise TaskClaimLost("Task terminalization cannot reconstruct its exact live worker lease.")
    return await apply(current.lease_expires_at)


async def _terminalize_claimed_task_or_detect_peer_winner(
    task_store: TaskStore,
    request: TaskTerminalizationRequest,
) -> bool:
    """Terminalize the claim, or conservatively identify a peer winner.

    ``True`` means the task is already terminal and this request's key has no
    receipt, so another terminalization won. A receipt under this request's key
    remains an explicit conflict because it may prove changed intent.
    """

    request, _request_sha256 = prepare_task_terminalization(request)
    try:
        await _terminalize_claimed_task(task_store, request)
    except TaskTerminalizationConflict:
        if not task_store.supports_idempotent_terminalization:
            raise
        receipt = await task_store.load_task_terminalization_receipt(
            request.task_id,
            request.idempotency_key,
        )
        if receipt is not None:
            raise
        task = await task_store.load_task(request.task_id)
        if task is not None:
            cancellation = _task_cancellation_terminalization_request(
                task,
                worker_id=request.worker_id,
            )
            if cancellation is not None:
                await _terminalize_claimed_task(task_store, cancellation)
                return True
        if task is not None and task.status in _TERMINAL_TASK_STATUSES:
            return True
        raise
    return False


def _task_terminalization_error_is_acknowledgement_ambiguous(exc: Exception) -> bool:
    if isinstance(
        exc,
        (TaskClaimLost, TaskTerminalizationConflict, TypeError, ValueError),
    ):
        return False
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    for error_type in type(exc).__mro__:
        module = error_type.__module__
        name = error_type.__name__
        if module == "sqlite3" and name == "OperationalError":
            error_name = getattr(exc, "sqlite_errorname", None)
            return isinstance(error_name, str) and (
                error_name == "SQLITE_IOERR" or error_name.startswith("SQLITE_IOERR_")
            )
        if module.startswith("psycopg") and name == "OperationalError":
            sqlstate = getattr(exc, "sqlstate", None)
            return sqlstate is None or (isinstance(sqlstate, str) and sqlstate.startswith("08"))
    return False


def _task_terminalization_error_category(exc: Exception) -> str:
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, ConnectionError):
        return "connection"
    return "database_operational"
