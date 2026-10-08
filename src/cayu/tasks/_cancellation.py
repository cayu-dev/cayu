"""Shared task cancellation state transitions and recovery settlement rules."""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from typing import Any

from cayu._clock import normalize_utc_datetime
from cayu._validation import canonical_durable_json_bytes, copy_durable_json_object
from cayu.approvals.tools import ResolutionActor, copy_resolution_actor
from cayu.tasks.cancellation import (
    _TASK_CANCELLATION_REQUESTED_REASON,
    _TASK_RETRY_CANCELLATION_REQUESTED_REASON,
    TaskCancellationReconciliation,
    TaskCancellationReconciliationEvent,
    TaskCancellationReconciliationEventType,
    TaskCancellationReconciliationEvidence,
    TaskCancellationReconciliationRequest,
    TaskCancellationReconciliationResult,
    TaskRetryCancellationReconciliation,
    TaskRetryCancellationReconciliationEvent,
    TaskRetryCancellationReconciliationEventType,
    TaskRetryCancellationReconciliationEvidence,
    TaskRetryCancellationReconciliationRequest,
    _task_cancellation_reconciliation_conflict,
    _task_cancellation_reconciliation_event,
    _task_cancellation_requested,
    _task_cancellation_requested_event,
    _task_retry_cancellation_reconciliation_conflict,
    _task_retry_cancellation_reconciliation_event,
    _task_retry_cancellation_requested,
)
from cayu.tasks.records import (
    Task,
    TaskClaimLost,
    TaskRetrySeriesDisposition,
    TaskStatus,
    _validate_task_retry_reconciliation_identity,
)
from cayu.tasks.retry import (
    TaskRetrySettlementResult,
    _cancelled_task_retry_settlement,
    _task_retry_cancellation_requested_event,
    _task_retry_events,
    _task_retry_runtime_idempotency_key,
)
from cayu.tasks.terminalization import (
    TaskTerminalizationConflict,
    TaskTerminalizationReceipt,
    TaskTerminalizationRequest,
    TaskTerminalKind,
    prepare_task_terminalization,
)


def _task_cancellation_requested_task(
    task: Task,
    *,
    error: dict[str, Any] | None,
    updated_at: datetime,
) -> Task:
    """Fence an ordinary active task until its worker terminalizes cancellation."""

    if (
        task.retry_series is not None
        or task.status not in {TaskStatus.CLAIMED, TaskStatus.RUNNING}
        or task.worker_id is None
        or task.lease_expires_at is None
    ):
        raise TaskTerminalizationConflict("Task cannot drain ordinary cancellation.")
    _validate_task_retry_reconciliation_identity(task.id, "task_id")
    _validate_task_retry_reconciliation_identity(task.worker_id, "worker_id")
    if _task_cancellation_requested(task):
        payload = task.status_payload
        if type(payload) is not dict or set(payload) != {
            "terminalization_idempotency_key",
            "error",
            "event",
        }:
            raise TaskTerminalizationConflict(
                "Task cancellation request conflicts with active ownership."
            )
        key = payload["terminalization_idempotency_key"]
        event_payload = payload["event"]
        if type(key) is not str or type(event_payload) is not dict:
            raise TaskTerminalizationConflict(
                "Task cancellation request conflicts with active ownership."
            )
        event = TaskCancellationReconciliationEvent.model_validate(event_payload)
        if event != _task_cancellation_requested_event(
            task,
            cancellation_idempotency_key=key,
            occurred_at=event.occurred_at,
        ):
            raise TaskTerminalizationConflict(
                "Task cancellation request conflicts with its event identity."
            )
        return task.model_copy(deep=True)
    cancellation_error = (
        {"code": TaskStatus.CANCELLED.value}
        if error is None
        else copy_durable_json_object(error, "error")
    )
    identity = canonical_durable_json_bytes(
        {
            "schema": "cayu.task-cancellation.v1",
            "task_id": task.id,
            "worker_id": task.worker_id,
        },
        "task_cancellation",
    )
    cancellation_idempotency_key = f"task-cancellation:v1:{sha256(identity).hexdigest()}"
    requested_at = normalize_utc_datetime(updated_at, "updated_at")
    requested_event = _task_cancellation_requested_event(
        task,
        cancellation_idempotency_key=cancellation_idempotency_key,
        occurred_at=requested_at,
    )
    return task.model_copy(
        update={
            "status_reason": _TASK_CANCELLATION_REQUESTED_REASON,
            "status_payload": {
                "terminalization_idempotency_key": cancellation_idempotency_key,
                "error": cancellation_error,
                "event": requested_event.model_dump(mode="json", warnings=False),
            },
            "updated_at": requested_at,
        },
        deep=True,
    )


def _expired_dispatched_task_cancellation(
    task: Task,
    *,
    updated_at: datetime,
    error: dict[str, Any] | None = None,
) -> Task:
    """Retain an expired dispatched claim until positive quiescence evidence exists."""

    if task.started_at is None:
        raise TaskTerminalizationConflict(
            "A task without durable dispatch evidence cannot enter drain reconciliation."
        )
    cancellation_error = (
        {"code": "task_worker_lease_expired_after_dispatch"}
        if error is None
        else copy_durable_json_object(error, "error")
    )
    if task.retry_series is not None:
        return _task_retry_cancellation_requested_task(
            task,
            error=cancellation_error,
            updated_at=updated_at,
        )
    return _task_cancellation_requested_task(
        task,
        error=cancellation_error,
        updated_at=updated_at,
    )


def _task_cancellation_terminalization_request(
    task: Task,
    *,
    worker_id: str,
) -> TaskTerminalizationRequest | None:
    """Build the exact worker terminalization for a requested cancellation."""

    if not _task_cancellation_requested(task):
        return None
    if task.worker_id != worker_id or task.lease_expires_at is None:
        raise TaskClaimLost(f"Worker {worker_id} does not own task {task.id}.")
    payload = task.status_payload
    if type(payload) is not dict or set(payload) != {
        "terminalization_idempotency_key",
        "error",
        "event",
    }:
        raise TaskTerminalizationConflict("Task cancellation request payload is invalid.")
    key = payload["terminalization_idempotency_key"]
    error = payload["error"]
    event_payload = payload["event"]
    if type(key) is not str or type(error) is not dict or type(event_payload) is not dict:
        raise TaskTerminalizationConflict("Task cancellation request payload is invalid.")
    event = TaskCancellationReconciliationEvent.model_validate(event_payload)
    if event != _task_cancellation_requested_event(
        task,
        cancellation_idempotency_key=key,
        occurred_at=event.occurred_at,
    ):
        raise TaskTerminalizationConflict(
            "Task cancellation request conflicts with its event identity."
        )
    return TaskTerminalizationRequest(
        task_id=task.id,
        worker_id=worker_id,
        lease_expires_at=task.lease_expires_at,
        handoff_id=task.interrupted_handoff_id,
        kind=TaskTerminalKind.CANCELLED,
        error=error,
        idempotency_key=key,
    )


def _validate_ordinary_task_terminalization_against_cancellation(
    task: Task,
    request: TaskTerminalizationRequest,
) -> None:
    cancellation = _task_cancellation_terminalization_request(
        task,
        worker_id=request.worker_id,
    )
    if cancellation is None:
        if request.kind is TaskTerminalKind.CANCELLED:
            raise TaskTerminalizationConflict(
                "Worker cancellation requires a durable cancellation request."
            )
        return
    if cancellation != request:
        raise TaskTerminalizationConflict("Task cancellation request must win terminalization.")


def _validated_owner_lost_task_cancellation(
    task: Task,
    request: TaskCancellationReconciliationRequest,
    *,
    now: datetime,
) -> tuple[
    TaskTerminalizationRequest,
    TaskCancellationReconciliationEvent,
]:
    """Fence reconciliation to the exact expired ordinary-task owner."""

    return _validated_task_cancellation(
        task,
        request,
        now=now,
        require_owner_lost=True,
    )


def _validated_task_cancellation(
    task: Task,
    request: TaskCancellationReconciliationRequest,
    *,
    now: datetime,
    require_owner_lost: bool,
) -> tuple[
    TaskTerminalizationRequest,
    TaskCancellationReconciliationEvent,
]:
    """Fence reconciliation to the exact ordinary task and optional owner loss."""

    now = normalize_utc_datetime(now, "now")
    payload = task.status_payload
    conflict: str | None = None
    if (
        task.status not in {TaskStatus.CLAIMED, TaskStatus.RUNNING}
        or task.status_reason != request.expected_status_reason
        or task.retry_series is not None
        or type(payload) is not dict
        or set(payload) != {"terminalization_idempotency_key", "error", "event"}
        or type(payload.get("terminalization_idempotency_key")) is not str
        or type(payload.get("error")) is not dict
        or type(payload.get("event")) is not dict
    ):
        conflict = "Task is not the expected cancellation-requested ordinary task."
    elif (
        task.id != request.task_id
        or task.worker_id != request.original_worker_id
        or task.interrupted_handoff_id != request.original_handoff_id
        or task.lease_expires_at != request.original_lease_expires_at
        or payload["terminalization_idempotency_key"] != request.cancellation_idempotency_key
    ):
        conflict = "Task cancellation reconciliation identity is stale."
    elif require_owner_lost and (task.lease_expires_at is None or task.lease_expires_at > now):
        conflict = "Task cancellation owner lease is still active."
    elif request.reconciliation_requested_at > now:
        conflict = "Task cancellation reconciliation request is from the future."

    for metadata_key, expected in (
        (
            "execution_profile_fingerprint",
            request.expected_execution_profile_fingerprint,
        ),
        ("effect_fingerprint", request.expected_effect_fingerprint),
    ):
        if conflict is None and metadata_key in task.metadata:
            stored = task.metadata[metadata_key]
            if type(stored) is not str or stored != expected:
                conflict = (
                    f"Task cancellation reconciliation conflicts with the stored {metadata_key}."
                )

    if conflict is not None:
        raise _task_cancellation_reconciliation_conflict(request, conflict)

    assert task.worker_id is not None
    assert task.lease_expires_at is not None
    assert type(payload) is dict
    terminalization = _task_cancellation_terminalization_request(
        task,
        worker_id=request.original_worker_id,
    )
    if (
        terminalization is None
        or terminalization.idempotency_key != request.cancellation_idempotency_key
    ):
        raise _task_cancellation_reconciliation_conflict(
            request,
            "Task cancellation terminalization identity is stale.",
        )
    requested_event = TaskCancellationReconciliationEvent.model_validate(payload["event"])
    if requested_event.occurred_at != request.cancellation_requested_at:
        raise _task_cancellation_reconciliation_conflict(
            request,
            "Task cancellation request event identity is stale.",
        )
    return terminalization, requested_event


def _reconciled_task_cancellation(
    task: Task,
    request: TaskCancellationReconciliationRequest,
    *,
    request_sha256: str,
    committed_at: datetime,
) -> TaskCancellationReconciliationResult:
    """Build a cancelled terminal receipt with bounded reconciliation evidence."""

    committed_at = normalize_utc_datetime(committed_at, "committed_at")
    terminalization, requested_event = _validated_owner_lost_task_cancellation(
        task,
        request,
        now=committed_at,
    )
    durable_actor = copy_resolution_actor(request.reconciled_by)
    if durable_actor is None:  # pragma: no cover - required request invariant
        raise AssertionError("Task cancellation reconciliation lost actor provenance.")
    durable_actor = ResolutionActor(
        subject=durable_actor.subject,
        tenant=durable_actor.tenant,
        source=durable_actor.source,
        claims={},
    )
    evidence = TaskCancellationReconciliationEvidence.model_validate(
        request.evidence.model_dump(mode="python")
    )
    reconciliation = TaskCancellationReconciliation(
        request_sha256=request_sha256,
        task_id=request.task_id,
        original_worker_id=request.original_worker_id,
        original_handoff_id=request.original_handoff_id,
        original_lease_expires_at=request.original_lease_expires_at,
        cancellation_requested_at=request.cancellation_requested_at,
        cancellation_idempotency_key=request.cancellation_idempotency_key,
        reconciliation_idempotency_key=request.reconciliation_idempotency_key,
        reconciliation_requested_at=request.reconciliation_requested_at,
        reconciled_by=durable_actor,
        evidence=evidence,
        events=(
            requested_event,
            _task_cancellation_reconciliation_event(
                request,
                event_type=TaskCancellationReconciliationEventType.STARTED,
                occurred_at=request.reconciliation_requested_at,
            ),
            _task_cancellation_reconciliation_event(
                request,
                event_type=TaskCancellationReconciliationEventType.RECONCILED,
                occurred_at=committed_at,
            ),
        ),
    )
    terminal_task = task.model_copy(
        update={
            "status": TaskStatus.CANCELLED,
            "status_reason": None,
            "status_payload": {
                "cancellation_reconciliation": reconciliation.model_dump(
                    mode="json",
                    warnings=False,
                )
            },
            "result": None,
            "error": copy_durable_json_object(terminalization.error, "error"),
            "worker_id": None,
            "lease_expires_at": None,
            "interrupted_handoff_id": None,
            "started_at": task.started_at or committed_at,
            "completed_at": committed_at,
            "updated_at": committed_at,
        },
        deep=True,
    )
    _, terminalization_sha256 = prepare_task_terminalization(terminalization)
    receipt = TaskTerminalizationReceipt(
        task_id=request.task_id,
        idempotency_key=request.cancellation_idempotency_key,
        worker_id=request.original_worker_id,
        kind=TaskTerminalKind.CANCELLED,
        request_sha256=terminalization_sha256,
        task=terminal_task,
        committed_at=committed_at,
    )
    return TaskCancellationReconciliationResult(
        request_sha256=request_sha256,
        task=terminal_task,
        terminalization_receipt=receipt,
        reconciliation=reconciliation,
        committed_at=committed_at,
    )


def _task_retry_cancellation_requested_task(
    task: Task,
    *,
    error: dict[str, Any] | None,
    updated_at: datetime,
) -> Task:
    """Fence an active attempt while its worker proves dispatched work quiescent."""

    series = task.retry_series
    if (
        series is None
        or series.disposition is not TaskRetrySeriesDisposition.ACTIVE
        or task.status not in {TaskStatus.CLAIMED, TaskStatus.RUNNING}
        or task.worker_id is None
        or task.lease_expires_at is None
    ):
        raise TaskTerminalizationConflict("Task retry attempt cannot drain cancellation.")
    if _task_retry_cancellation_requested(task):
        payload = task.status_payload
        if type(payload) is dict and set(payload) == {
            "settlement_idempotency_key",
            "error",
            "event",
        }:
            return task.model_copy(deep=True)
        # Cayu 0.3.0 persisted the same fenced cancellation intent before the
        # bounded cancellation event was added. Replaying that exact request is
        # the only supported upgrade path: preserve its original occurrence
        # time and identities, then let the public reconciliation API validate
        # and settle it normally.
        expected_key = _task_retry_runtime_idempotency_key(task, "cancellation")
        if (
            type(payload) is not dict
            or set(payload) != {"settlement_idempotency_key", "error"}
            or payload.get("settlement_idempotency_key") != expected_key
            or type(payload.get("error")) is not dict
        ):
            raise TaskTerminalizationConflict(
                "Task retry cancellation request conflicts with active ownership."
            )
        requested_event = _task_retry_cancellation_requested_event(
            task,
            occurred_at=task.updated_at,
        )
        return task.model_copy(
            update={
                "status_payload": {
                    "settlement_idempotency_key": expected_key,
                    "error": copy_durable_json_object(payload["error"], "error"),
                    "event": requested_event.model_dump(mode="json", warnings=False),
                }
            },
            deep=True,
        )
    cancellation_error = (
        {"code": TaskRetrySeriesDisposition.CANCELLED.value}
        if error is None
        else copy_durable_json_object(error, "error")
    )
    requested_event = _task_retry_cancellation_requested_event(
        task,
        occurred_at=updated_at,
    )
    return task.model_copy(
        update={
            "status_reason": _TASK_RETRY_CANCELLATION_REQUESTED_REASON,
            "status_payload": {
                "settlement_idempotency_key": _task_retry_runtime_idempotency_key(
                    task,
                    "cancellation",
                ),
                "error": cancellation_error,
                "event": requested_event.model_dump(mode="json", warnings=False),
            },
            "updated_at": normalize_utc_datetime(updated_at, "updated_at"),
        },
        deep=True,
    )


def _validated_owner_lost_task_retry_cancellation(
    task: Task,
    request: TaskRetryCancellationReconciliationRequest,
    *,
    now: datetime,
) -> tuple[dict[str, Any], TaskRetryCancellationReconciliationEvent]:
    """Fence reconciliation to the exact expired owner and cancellation marker."""

    return _validated_task_retry_cancellation(
        task,
        request,
        now=now,
        require_owner_lost=True,
    )


def _validated_task_retry_cancellation(
    task: Task,
    request: TaskRetryCancellationReconciliationRequest,
    *,
    now: datetime,
    require_owner_lost: bool,
) -> tuple[dict[str, Any], TaskRetryCancellationReconciliationEvent]:
    """Fence reconciliation to the exact task and optionally an expired owner."""

    now = normalize_utc_datetime(now, "now")
    series = task.retry_series
    payload = task.status_payload
    conflict: str | None = None
    if (
        task.status not in {TaskStatus.CLAIMED, TaskStatus.RUNNING}
        or task.status_reason != request.expected_status_reason
        or series is None
        or series.disposition is not TaskRetrySeriesDisposition.ACTIVE
        or type(payload) is not dict
        or set(payload) != {"settlement_idempotency_key", "error", "event"}
        or type(payload.get("settlement_idempotency_key")) is not str
        or type(payload.get("error")) is not dict
        or type(payload.get("event")) is not dict
    ):
        conflict = "Task is not the expected cancellation-requested retry attempt."
    elif (
        task.id != request.task_id
        or series.series_id != request.series_id
        or series.attempt != request.attempt
        or series.causal_budget_id != request.causal_budget_id
        or task.worker_id != request.original_worker_id
        or task.lease_expires_at != request.original_lease_expires_at
        or payload["settlement_idempotency_key"] != request.cancellation_idempotency_key
        or request.cancellation_idempotency_key
        != _task_retry_runtime_idempotency_key(task, "cancellation")
    ):
        conflict = "Task retry cancellation reconciliation identity is stale."
    elif require_owner_lost and (task.lease_expires_at is None or task.lease_expires_at > now):
        conflict = "Task retry cancellation owner lease is still active."
    elif request.reconciliation_requested_at > now:
        conflict = "Task retry cancellation reconciliation request is from the future."

    for metadata_key, expected in (
        (
            "execution_profile_fingerprint",
            request.expected_execution_profile_fingerprint,
        ),
        ("effect_fingerprint", request.expected_effect_fingerprint),
    ):
        if conflict is None and metadata_key in task.metadata:
            stored = task.metadata[metadata_key]
            if type(stored) is not str or stored != expected:
                conflict = (
                    "Task retry cancellation reconciliation conflicts with the "
                    f"stored {metadata_key}."
                )

    if conflict is not None:
        raise _task_retry_cancellation_reconciliation_conflict(
            request,
            conflict,
        )

    assert task.lease_expires_at is not None
    assert type(payload) is dict
    assert type(payload["error"]) is dict
    requested_event = TaskRetryCancellationReconciliationEvent.model_validate(payload["event"])
    expected_event = _task_retry_cancellation_requested_event(
        task,
        occurred_at=requested_event.occurred_at,
    )
    if (
        requested_event != expected_event
        or requested_event.occurred_at != request.cancellation_requested_at
    ):
        raise _task_retry_cancellation_reconciliation_conflict(
            request,
            "Task retry cancellation request event identity is stale.",
        )
    return (
        copy_durable_json_object(payload["error"], "error"),
        requested_event,
    )


def _reconciled_task_retry_cancellation(
    task: Task,
    request: TaskRetryCancellationReconciliationRequest,
    *,
    request_sha256: str,
    committed_at: datetime,
) -> TaskRetrySettlementResult:
    """Build an ordinary cancelled receipt with bounded reconciliation evidence."""

    cancellation_error, requested_event = _validated_owner_lost_task_retry_cancellation(
        task,
        request,
        now=committed_at,
    )
    base = _cancelled_task_retry_settlement(
        task,
        error=cancellation_error,
        committed_at=committed_at,
    )
    durable_actor = copy_resolution_actor(request.reconciled_by)
    if durable_actor is None:  # pragma: no cover - required request invariant
        raise AssertionError("Task retry reconciliation lost actor provenance.")
    durable_actor = ResolutionActor(
        subject=durable_actor.subject,
        tenant=durable_actor.tenant,
        source=durable_actor.source,
        claims={},
    )
    evidence = TaskRetryCancellationReconciliationEvidence.model_validate(
        request.evidence.model_dump(mode="python")
    )
    reconciliation = TaskRetryCancellationReconciliation(
        request_sha256=request_sha256,
        task_id=request.task_id,
        series_id=request.series_id,
        attempt=request.attempt,
        causal_budget_id=request.causal_budget_id,
        original_worker_id=request.original_worker_id,
        original_lease_expires_at=request.original_lease_expires_at,
        cancellation_requested_at=request.cancellation_requested_at,
        cancellation_idempotency_key=request.cancellation_idempotency_key,
        reconciliation_idempotency_key=request.reconciliation_idempotency_key,
        reconciliation_requested_at=request.reconciliation_requested_at,
        reconciled_by=durable_actor,
        evidence=evidence,
        events=(
            requested_event,
            _task_retry_cancellation_reconciliation_event(
                request,
                event_type=TaskRetryCancellationReconciliationEventType.STARTED,
                occurred_at=request.reconciliation_requested_at,
            ),
            _task_retry_cancellation_reconciliation_event(
                request,
                event_type=TaskRetryCancellationReconciliationEventType.RECONCILED,
                occurred_at=committed_at,
            ),
        ),
    )
    status_payload = copy_durable_json_object(base.task.status_payload, "status_payload")
    status_payload["cancellation_reconciliation"] = reconciliation.model_dump(
        mode="json",
        warnings=False,
    )
    settled = base.task.model_copy(
        update={"status_payload": status_payload},
        deep=True,
    )
    return TaskRetrySettlementResult(
        task_id=request.task_id,
        idempotency_key=request.cancellation_idempotency_key,
        request_sha256=request_sha256,
        task=settled,
        successor=None,
        reconciliation=reconciliation,
        events=_task_retry_events(settled, occurred_at=committed_at),
        committed_at=committed_at,
    )
