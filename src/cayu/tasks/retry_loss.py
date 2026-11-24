"""Retry a retry-series attempt whose worker was lost, instead of cancelling the series.

A worker lost during a retry-series attempt leaves the attempt fenced
(``status_reason`` ``"retry_cancellation_requested"``). Once the caller knows the
worker is gone, :func:`retry_lost_task_attempt` settles that attempt as a
retryable failure through ``TaskStore.reconcile_task_retry_cancellation`` with
``retry_lost_attempt=True``. The series' own rules then decide whether it queues
the next attempt: the lost attempt uses up an attempt, its elapsed time counts,
and its usage counts against the token and cost caps. The usage a lost worker
spent is not known, so unless the caller passes it, it counts as everything the
series had left: a series with a token or cost cap ends as exhausted instead of
spending past it, and one without caps retries.
"""

from __future__ import annotations

from decimal import Decimal
from hashlib import sha256

from cayu._validation import canonical_durable_json_bytes, require_clean_nonblank
from cayu.approvals.tools import ResolutionActor, ResolutionActorSource
from cayu.tasks.cancellation import (
    TaskRetryCancellationReconciliationEvent,
    TaskRetryCancellationReconciliationEvidence,
    TaskRetryCancellationReconciliationOutcome,
    TaskRetryCancellationReconciliationRequest,
    _task_retry_cancellation_requested,
)
from cayu.tasks.records import Task
from cayu.tasks.retry import TaskRetrySettlementResult
from cayu.tasks.store import TaskStore
from cayu.tasks.terminalization import TaskTerminalizationConflict


def _fingerprint(task: Task, key: str) -> str | None:
    value = task.metadata.get(key)
    return value if type(value) is str else None


async def retry_lost_task_attempt(
    task_store: TaskStore,
    task: Task,
    *,
    validator_id: str,
    validator_version: str,
    evidence_id: str,
    reconciled_by: ResolutionActor | None = None,
    token_count: int | None = None,
    estimated_cost: Decimal | None = None,
) -> TaskRetrySettlementResult:
    """Settle ``task``, a retry attempt fenced after its worker was lost, as a retryable failure.

    Call it only when the caller can vouch that the attempt's worker stopped, as
    for ``reconcile_abandoned_tasks``; ``validator_id``, ``validator_version`` and
    ``evidence_id`` record that judgement. ``token_count`` and ``estimated_cost``
    are what the lost attempt spent, if known (for example from its session's
    usage); either one left out counts as all the series had left. Returns the
    settlement, whose ``successor`` is the next attempt when the series queued one.
    An attempt someone asked to cancel (also after the worker was lost) and a graph
    member are refused with ``TaskTerminalizationConflict``; an event-keyed task is
    retried only while its event's budget allows. Repeating the call for the same
    attempt and evidence returns the same settlement.
    """

    for name, value in (
        ("validator_id", validator_id),
        ("validator_version", validator_version),
        ("evidence_id", evidence_id),
    ):
        require_clean_nonblank(value, name)
    series = task.retry_series
    payload = task.status_payload
    if (
        series is None
        or task.worker_id is None
        or task.lease_expires_at is None
        or type(payload) is not dict
        or not _task_retry_cancellation_requested(task)
        or type(payload.get("event")) is not dict
        or type(payload.get("settlement_idempotency_key")) is not str
    ):
        raise TaskTerminalizationConflict(
            "Only a retry attempt fenced after its worker was lost can be retried this way."
        )
    event = TaskRetryCancellationReconciliationEvent.model_validate(payload["event"])
    profile = _fingerprint(task, "execution_profile_fingerprint")
    effect = _fingerprint(task, "effect_fingerprint")
    evidence_sha256 = sha256(
        canonical_durable_json_bytes(
            {
                "schema": "cayu.task-retry-lost-attempt.v1",
                "task_id": task.id,
                "worker_id": task.worker_id,
                "cancellation_idempotency_key": payload["settlement_idempotency_key"],
                "validator_id": validator_id,
                "validator_version": validator_version,
                "evidence_id": evidence_id,
            },
            "task_retry_lost_attempt",
        )
    ).hexdigest()
    validated_at = max(event.occurred_at, task.lease_expires_at)
    return await task_store.reconcile_task_retry_cancellation(
        TaskRetryCancellationReconciliationRequest(
            task_id=task.id,
            series_id=series.series_id,
            attempt=series.attempt,
            causal_budget_id=series.causal_budget_id,
            original_worker_id=task.worker_id,
            original_lease_expires_at=task.lease_expires_at,
            cancellation_requested_at=event.occurred_at,
            cancellation_idempotency_key=payload["settlement_idempotency_key"],
            reconciliation_idempotency_key=f"task-retry-lost-attempt:v1:{evidence_sha256}",
            reconciliation_requested_at=validated_at,
            reconciled_by=reconciled_by
            or ResolutionActor(subject=validator_id, source=ResolutionActorSource.SYSTEM),
            evidence=TaskRetryCancellationReconciliationEvidence(
                outcome=TaskRetryCancellationReconciliationOutcome.QUIESCENT,
                validator_id=validator_id,
                validator_version=validator_version,
                evidence_id=evidence_id,
                evidence_sha256=evidence_sha256,
                validated_at=validated_at,
                execution_profile_fingerprint=profile,
                effect_fingerprint=effect,
            ),
            expected_execution_profile_fingerprint=profile,
            expected_effect_fingerprint=effect,
            retry_lost_attempt=True,
            lost_attempt_token_count=token_count,
            lost_attempt_estimated_cost=estimated_cost,
        )
    )


__all__ = ["retry_lost_task_attempt"]
