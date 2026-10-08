"""Task retry outcomes, cumulative accounting and deterministic settlement rules."""

from __future__ import annotations

import math
from copy import deepcopy
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._clock import normalize_utc_datetime
from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    canonical_durable_json_bytes,
    copy_durable_json_object,
    copy_durable_metadata,
)
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.sessions.invocation import (
    copy_task_invocation,
)
from cayu.tasks._scheduling import require_schedule, schedule_revision_after
from cayu.tasks.cancellation import (
    TaskRetryCancellationReconciliation,
    TaskRetryCancellationReconciliationEvent,
    TaskRetryCancellationReconciliationEventType,
    TaskRetryCancellationReconciliationRequest,
    _task_retry_cancellation_reconciliation_conflict,
    _task_retry_cancellation_reconciliation_event_id,
    _task_retry_cancellation_requested,
    _task_retry_reconciliation_identity_sha256,
)
from cayu.tasks.records import (
    _TASK_RETRY_COST_MAX_DIGITS,
    _TASK_RETRY_TOTAL_COST_MAX_DIGITS,
    _TERMINAL_TASK_STATUSES,
    Task,
    TaskRetryPolicy,
    TaskRetrySeriesDisposition,
    TaskRetrySeriesSnapshot,
    TaskStatus,
    _bounded_task_retry_decimal,
    _task_retry_attempt_authority_sha256,
    _validate_task_retry_cost_currency,
    copy_task,
)
from cayu.tasks.scheduling import TaskScheduleConflict, TaskScheduleEligibility
from cayu.tasks.terminalization import (
    TaskTerminalizationConflict,
    _validate_task_terminalization_idempotency_key,
)

_TASK_RETRY_MAX_ATTEMPT_TOKEN_REPORT = MAX_DURABLE_JSON_INTEGER // 100


class TaskRetryAttemptDisposition(StrEnum):
    """Application-classified outcome for one retry-series attempt."""

    SUCCEEDED = "succeeded"
    RETRYABLE_FAILURE = "retryable_failure"
    NON_RETRYABLE_FAILURE = "non_retryable_failure"
    CANCELLED = "cancelled"


class TaskRetryEventType(StrEnum):
    """Stable task-level retry event types committed with a settlement."""

    ATTEMPT_SETTLED = "task.retry.attempt_settled"
    RETRY_SCHEDULED = "task.retry.scheduled"
    SERIES_TERMINAL = "task.retry.series_terminal"


class TaskRetryEvent(BaseModel):
    """Bounded, failure-payload-free retry evidence committed by a task store."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    id: str
    type: TaskRetryEventType
    task_id: str
    series_id: str
    causal_budget_id: str
    attempt: StrictInt = Field(ge=1, le=100)
    disposition: TaskRetrySeriesDisposition
    occurred_at: datetime
    attempts_remaining: StrictInt = Field(ge=0, le=99)
    tokens_remaining: StrictInt | None = Field(
        default=None,
        ge=0,
        le=MAX_DURABLE_JSON_INTEGER,
    )
    estimated_cost_remaining: Decimal | None = Field(default=None, ge=0)
    cost_currency: str
    elapsed_deadline: datetime | None = None
    next_eligible_at: datetime | None = None

    @field_validator("id", "task_id", "series_id", "causal_budget_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("cost_currency")
    @classmethod
    def validate_cost_currency(cls, value: str) -> str:
        return _validate_task_retry_cost_currency(value)

    @field_validator("occurred_at")
    @classmethod
    def normalize_occurred_at(cls, value: datetime) -> datetime:
        return normalize_utc_datetime(value, "occurred_at")

    @field_validator("elapsed_deadline", "next_eligible_at")
    @classmethod
    def normalize_optional_datetime(cls, value: datetime | None, info) -> datetime | None:
        if value is None:
            return None
        return normalize_utc_datetime(value, info.field_name)

    @field_validator("estimated_cost_remaining")
    @classmethod
    def validate_estimated_cost_remaining(cls, value: Decimal | None) -> Decimal | None:
        if value is None:
            return None
        return _bounded_task_retry_decimal(
            value,
            "estimated_cost_remaining",
            max_digits=_TASK_RETRY_COST_MAX_DIGITS,
        )


class _TaskRetryAttemptOutcome(BaseModel):
    """Shared validated application-owned outcome material."""

    model_config = ConfigDict(
        extra="forbid",
        hide_input_in_errors=True,
        allow_inf_nan=False,
    )

    idempotency_key: str
    disposition: TaskRetryAttemptDisposition
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    token_count: StrictInt = Field(
        default=0,
        ge=0,
        le=_TASK_RETRY_MAX_ATTEMPT_TOKEN_REPORT,
    )
    estimated_cost: Decimal = Field(default=Decimal(0), ge=0)
    retry_after_seconds: StrictFloat | None = Field(default=None, ge=0, le=86_400)

    @field_validator("idempotency_key")
    @classmethod
    def validate_idempotency_key(cls, value: str) -> str:
        return _validate_task_terminalization_idempotency_key(value)

    @field_validator("result", "error", mode="before")
    @classmethod
    def copy_payload(cls, value: dict[str, Any] | None, info) -> dict[str, Any] | None:
        if value is None:
            return None
        return copy_durable_json_object(value, info.field_name)

    @field_validator("estimated_cost")
    @classmethod
    def validate_estimated_cost(cls, value: Decimal) -> Decimal:
        return _bounded_task_retry_decimal(
            value,
            "estimated_cost",
            max_digits=_TASK_RETRY_COST_MAX_DIGITS,
        )

    @model_validator(mode="after")
    def validate_disposition_payload(self) -> _TaskRetryAttemptOutcome:
        if self.retry_after_seconds is not None and not math.isfinite(self.retry_after_seconds):
            raise ValueError("retry_after_seconds must be finite.")
        if self.disposition is TaskRetryAttemptDisposition.SUCCEEDED:
            if self.result is None or self.error is not None:
                raise ValueError("Succeeded attempts require result and forbid error.")
        elif self.result is not None or self.error is None:
            raise ValueError("Non-success attempts require error and forbid result.")
        if (
            self.retry_after_seconds is not None
            and self.disposition is not TaskRetryAttemptDisposition.RETRYABLE_FAILURE
        ):
            raise ValueError("Only retryable failures accept retry_after_seconds.")
        return self


class TaskRetrySettlementRequest(_TaskRetryAttemptOutcome):
    """One typed, claim-fenced retry-attempt outcome."""

    task_id: str
    worker_id: str
    lease_expires_at: datetime | None = None
    causal_budget_id: str

    @field_validator("task_id", "worker_id", "causal_budget_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("lease_expires_at")
    @classmethod
    def normalize_lease_expires_at(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return normalize_utc_datetime(value, "lease_expires_at")


class TaskRetryAttemptReport(_TaskRetryAttemptOutcome):
    """Application-owned classification and bounded accounting for one attempt."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        allow_inf_nan=False,
    )


class TaskRetrySettlementResult(BaseModel):
    """Immutable settlement receipt for one attempt and optional successor."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    task_id: str
    idempotency_key: str
    request_sha256: str
    task: Task
    successor: Task | None = None
    reconciliation: TaskRetryCancellationReconciliation | None = None
    events: tuple[TaskRetryEvent, ...] = Field(min_length=2, max_length=2)
    committed_at: datetime

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, value: str) -> str:
        return require_clean_nonblank(value, "task_id")

    @field_validator("idempotency_key")
    @classmethod
    def validate_result_idempotency_key(cls, value: str) -> str:
        return _validate_task_terminalization_idempotency_key(value)

    @field_validator("request_sha256")
    @classmethod
    def validate_request_sha256(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("request_sha256 must be a lowercase SHA-256 digest.")
        return value

    @field_validator("committed_at")
    @classmethod
    def normalize_committed_at(cls, value: datetime) -> datetime:
        return normalize_utc_datetime(value, "committed_at")

    @model_validator(mode="after")
    def validate_settlement_evidence(self) -> TaskRetrySettlementResult:
        series = self.task.retry_series
        if self.task.id != self.task_id or series is None:
            raise ValueError("Task retry receipt conflicts with its settled attempt.")
        if self.task.worker_id is not None or self.task.lease_expires_at is not None:
            raise ValueError("Task retry receipt retains live attempt ownership.")
        expected_status = {
            TaskRetrySeriesDisposition.SUCCEEDED: TaskStatus.COMPLETED,
            TaskRetrySeriesDisposition.CANCELLED: TaskStatus.CANCELLED,
        }.get(series.disposition, TaskStatus.FAILED)
        if (
            self.task.status is TaskStatus.DEPENDENCY_SKIPPED
            and series.disposition is TaskRetrySeriesDisposition.NON_RETRYABLE_FAILURE
            and self.task.graph_id is not None
            and self.task.prerequisite_task_ids
            and self.task.status_reason == "dependency_failed"
            and isinstance(self.task.status_payload, dict)
        ):
            causes = self.task.status_payload.get("failed_prerequisite_task_ids")
            if (
                isinstance(causes, list)
                and causes
                and all(type(identity) is str for identity in causes)
                and causes == sorted(set(causes))
                and set(causes) <= set(self.task.prerequisite_task_ids)
            ):
                expected_status = TaskStatus.DEPENDENCY_SKIPPED
        if self.task.status is not expected_status:
            raise ValueError("Task retry receipt conflicts with its series disposition.")

        if self.reconciliation is not None:
            reconciliation = self.reconciliation
            status_payload = self.task.status_payload
            if (
                series.disposition is not TaskRetrySeriesDisposition.CANCELLED
                or self.successor is not None
                or reconciliation.request_sha256 != self.request_sha256
                or reconciliation.task_id != self.task_id
                or reconciliation.series_id != series.series_id
                or reconciliation.attempt != series.attempt
                or reconciliation.causal_budget_id != series.causal_budget_id
                or reconciliation.cancellation_idempotency_key != self.idempotency_key
                or reconciliation.original_lease_expires_at > self.committed_at
                or reconciliation.reconciliation_requested_at > self.committed_at
                or reconciliation.events[1].occurred_at
                != reconciliation.reconciliation_requested_at
                or reconciliation.events[2].occurred_at != self.committed_at
                or type(status_payload) is not dict
                or status_payload.get("cancellation_reconciliation")
                != reconciliation.model_dump(mode="json", warnings=False)
            ):
                raise ValueError(
                    "Task retry cancellation reconciliation conflicts with its receipt."
                )

        scheduled = series.disposition is TaskRetrySeriesDisposition.RETRY_SCHEDULED
        if scheduled != (self.successor is not None):
            raise ValueError("Task retry receipt has contradictory successor evidence.")
        if self.successor is not None:
            successor_series = self.successor.retry_series
            if (
                self.successor.id != series.successor_task_id
                or self.successor.status is not TaskStatus.PENDING
                or self.successor.available_at != series.next_eligible_at
                or successor_series is None
                or successor_series.series_id != series.series_id
                or successor_series.attempt != series.attempt + 1
                or successor_series.predecessor_task_id != self.task.id
                or successor_series.disposition is not TaskRetrySeriesDisposition.ACTIVE
                or successor_series.policy != series.policy
                or successor_series.started_at != series.started_at
                or successor_series.causal_budget_id != series.causal_budget_id
                or successor_series.cumulative_tokens != series.cumulative_tokens
                or successor_series.cumulative_estimated_cost != series.cumulative_estimated_cost
                or successor_series.tokens_remaining != series.tokens_remaining
                or successor_series.estimated_cost_remaining != series.estimated_cost_remaining
                or successor_series.elapsed_deadline != series.elapsed_deadline
                or self.successor.type != self.task.type
                or self.successor.title != self.task.title
                or self.successor.description != self.task.description
                or self.successor.parent_task_id != self.task.parent_task_id
                or self.successor.assigned_agent_name != self.task.assigned_agent_name
                or self.successor.input != self.task.input
                or self.successor.metadata != self.task.metadata
                or self.successor.invocation != self.task.invocation
                or self.successor.session_id is not None
                or self.successor.worker_id is not None
                or self.successor.lease_expires_at is not None
                or self.successor.status_reason is not None
                or self.successor.status_payload is not None
                or self.successor.result is not None
                or self.successor.error is not None
                or self.successor.started_at is not None
                or self.successor.completed_at is not None
                or self.successor.created_at != self.committed_at
                or self.successor.updated_at != self.committed_at
            ):
                raise ValueError("Task retry receipt successor conflicts with the settled attempt.")

        expected_event_types = (
            TaskRetryEventType.ATTEMPT_SETTLED,
            (
                TaskRetryEventType.RETRY_SCHEDULED
                if scheduled
                else TaskRetryEventType.SERIES_TERMINAL
            ),
        )
        for event, expected_type in zip(self.events, expected_event_types, strict=True):
            if (
                event.type is not expected_type
                or event.task_id != self.task_id
                or event.series_id != series.series_id
                or event.causal_budget_id != series.causal_budget_id
                or event.attempt != series.attempt
                or event.disposition is not series.disposition
                or event.occurred_at != self.committed_at
                or event.attempts_remaining != series.attempts_remaining
                or event.tokens_remaining != series.tokens_remaining
                or event.estimated_cost_remaining != series.estimated_cost_remaining
                or event.cost_currency != series.policy.cost_currency
                or event.elapsed_deadline != series.elapsed_deadline
                or event.next_eligible_at != series.next_eligible_at
            ):
                raise ValueError("Task retry receipt event conflicts with the settled attempt.")
        return self


def _copy_task_retry_event(event: TaskRetryEvent) -> TaskRetryEvent:
    if type(event) is not TaskRetryEvent:
        raise TypeError("Task retry events must be TaskRetryEvent instances.")
    return TaskRetryEvent(
        id=event.id,
        type=event.type,
        task_id=event.task_id,
        series_id=event.series_id,
        causal_budget_id=event.causal_budget_id,
        attempt=event.attempt,
        disposition=event.disposition,
        occurred_at=event.occurred_at,
        attempts_remaining=event.attempts_remaining,
        tokens_remaining=event.tokens_remaining,
        estimated_cost_remaining=event.estimated_cost_remaining,
        cost_currency=event.cost_currency,
        elapsed_deadline=event.elapsed_deadline,
        next_eligible_at=event.next_eligible_at,
    )


def _copy_task_retry_settlement_result(
    receipt: TaskRetrySettlementResult,
) -> TaskRetrySettlementResult:
    if type(receipt) is not TaskRetrySettlementResult:
        raise TypeError(
            "Task retry settlement loads must return TaskRetrySettlementResult instances."
        )
    return TaskRetrySettlementResult(
        task_id=receipt.task_id,
        idempotency_key=receipt.idempotency_key,
        request_sha256=receipt.request_sha256,
        task=copy_task(receipt.task),
        successor=None if receipt.successor is None else copy_task(receipt.successor),
        reconciliation=(
            None
            if receipt.reconciliation is None
            else TaskRetryCancellationReconciliation.model_validate(
                receipt.reconciliation.model_dump(mode="python")
            )
        ),
        events=tuple(_copy_task_retry_event(event) for event in receipt.events),
        committed_at=receipt.committed_at,
    )


def prepare_task_retry_settlement(
    request: TaskRetrySettlementRequest,
) -> tuple[TaskRetrySettlementRequest, str]:
    """Detach and digest one typed retry-attempt report."""

    if type(request) is not TaskRetrySettlementRequest:
        raise TypeError("Task retry settlements require a TaskRetrySettlementRequest.")
    copied = TaskRetrySettlementRequest.model_validate(
        request.model_dump(mode="python", warnings=False)
    )
    request_sha256 = sha256(
        canonical_durable_json_bytes(
            {
                "schema": "cayu.task-retry-settlement.v2",
                **copied.model_dump(mode="json", warnings=False),
            },
            "task_retry_settlement",
        )
    ).hexdigest()
    return copied, request_sha256


def _legacy_task_retry_settlement_request_sha256(
    request: TaskRetrySettlementRequest,
) -> str:
    """Reconstruct the digest emitted before lease-generation fencing."""

    copied = TaskRetrySettlementRequest.model_validate(
        request.model_dump(mode="python", warnings=False)
    )
    return sha256(
        canonical_durable_json_bytes(
            {
                "schema": "cayu.task-retry-settlement.v1",
                **copied.model_dump(
                    mode="json",
                    warnings=False,
                    exclude={"lease_expires_at"},
                ),
            },
            "task_retry_settlement",
        )
    ).hexdigest()


def _task_retry_settlement_request_matches_sha256(
    request: TaskRetrySettlementRequest,
    *,
    request_sha256: str,
    candidate_sha256: str,
) -> bool:
    return candidate_sha256 in {
        request_sha256,
        _legacy_task_retry_settlement_request_sha256(request),
    }


def _replay_task_retry_settlement(
    *,
    request: TaskRetrySettlementRequest,
    request_sha256: str,
    receipt: TaskRetrySettlementResult,
    current_task: Task | None,
) -> TaskRetrySettlementResult:
    receipt = _copy_task_retry_settlement_result(receipt)
    current_task = None if current_task is None else copy_task(current_task)
    if not _task_retry_settlement_request_matches_sha256(
        request,
        request_sha256=request_sha256,
        candidate_sha256=receipt.request_sha256,
    ):
        raise TaskTerminalizationConflict(
            "Task retry settlement idempotency key is bound to another intent."
        )
    if current_task != receipt.task:
        raise TaskTerminalizationConflict(
            "Task retry settlement receipt conflicts with the current terminal attempt."
        )
    return receipt


def _replay_task_retry_cancellation_reconciliation(
    *,
    request: TaskRetryCancellationReconciliationRequest,
    request_sha256: str,
    receipt: TaskRetrySettlementResult,
    current_task: Task | None,
) -> TaskRetrySettlementResult:
    receipt = _copy_task_retry_settlement_result(receipt)
    current_task = None if current_task is None else copy_task(current_task)
    reconciliation = receipt.reconciliation
    if (
        reconciliation is None
        or receipt.idempotency_key != request.cancellation_idempotency_key
        or receipt.request_sha256 != request_sha256
        or reconciliation.request_sha256 != request_sha256
        or reconciliation.reconciliation_idempotency_key != request.reconciliation_idempotency_key
    ):
        raise _task_retry_cancellation_reconciliation_conflict(
            request,
            "Task retry cancellation settlement idempotency key is bound to another intent.",
        )
    if current_task != receipt.task:
        raise _task_retry_cancellation_reconciliation_conflict(
            request,
            "Task retry cancellation receipt conflicts with the current terminal attempt.",
        )
    return receipt


def _validate_task_retry_settlement_receipt_identity(
    receipt: TaskRetrySettlementResult,
    *,
    request: TaskRetrySettlementRequest,
    request_sha256: str,
) -> TaskRetrySettlementResult:
    receipt = _copy_task_retry_settlement_result(receipt)
    if (
        receipt.task_id != request.task_id
        or receipt.idempotency_key != request.idempotency_key
        or not _task_retry_settlement_request_matches_sha256(
            request,
            request_sha256=request_sha256,
            candidate_sha256=receipt.request_sha256,
        )
    ):
        raise TaskTerminalizationConflict(
            "Task retry settlement receipt conflicts with the requested operation."
        )
    return receipt


def _task_retry_series_id(task_id: str) -> str:
    material = canonical_durable_json_bytes(
        {"schema": "cayu.task-retry-series.v1", "first_task_id": task_id},
        "task_retry_series_id",
    )
    return f"task-retry-series:v1:{sha256(material).hexdigest()}"


def _task_retry_successor_id(series_id: str, attempt: int) -> str:
    material = canonical_durable_json_bytes(
        {"schema": "cayu.task-retry-attempt.v1", "series_id": series_id, "attempt": attempt},
        "task_retry_successor_id",
    )
    return f"task-retry-attempt:v1:{sha256(material).hexdigest()}"


def _rescheduled_initial_task_retry_series(
    task: Task, *, available_at: datetime
) -> TaskRetrySeriesSnapshot | None:
    """Rebind only an unstarted first attempt, without renewing its retry envelope."""
    series = task.retry_series
    if series is None:
        return None
    if (
        series.attempt != 1
        or series.predecessor_task_id is not None
        or series.disposition is not TaskRetrySeriesDisposition.ACTIVE
        or task.started_at is not None
    ):
        raise TaskScheduleConflict(
            "Retry successor eligibility is owned by its predecessor receipt."
        )
    digest = _task_retry_attempt_authority_sha256(
        task_id=task.id,
        task_type=task.type,
        title=task.title,
        description=task.description,
        parent_task_id=task.parent_task_id,
        assigned_agent_name=task.assigned_agent_name,
        available_at=available_at,
        created_at=task.created_at,
        task_input=task.input,
        metadata=task.metadata,
        invocation=task.invocation,
        series_id=series.series_id,
        causal_budget_id=series.causal_budget_id,
        attempt=series.attempt,
        policy=series.policy,
        started_at=series.started_at,
        cumulative_tokens=series.cumulative_tokens,
        cumulative_estimated_cost=series.cumulative_estimated_cost,
        predecessor_task_id=series.predecessor_task_id,
    )
    return series.model_copy(update={"authority_sha256": digest}, deep=True)


def _task_retry_runtime_idempotency_key(task: Task, operation: str) -> str:
    series = task.retry_series
    if series is None:
        raise ValueError("Task does not belong to a retry series.")
    operation = require_clean_nonblank(operation, "operation")
    material = canonical_durable_json_bytes(
        {
            "schema": "cayu.task-retry-runtime-settlement.v1",
            "operation": operation,
            "series_id": series.series_id,
            "task_id": task.id,
            "attempt": series.attempt,
        },
        "task_retry_runtime_idempotency_key",
    )
    return f"task-retry-{operation}:v1:{sha256(material).hexdigest()}"


def _task_retry_cancellation_requested_event(
    task: Task,
    *,
    occurred_at: datetime,
) -> TaskRetryCancellationReconciliationEvent:
    series = task.retry_series
    if series is None or task.worker_id is None:
        raise TaskTerminalizationConflict(
            "Task retry cancellation request lacks active attempt identity."
        )
    cancellation_idempotency_key = _task_retry_runtime_idempotency_key(
        task,
        "cancellation",
    )
    return TaskRetryCancellationReconciliationEvent(
        id=_task_retry_cancellation_reconciliation_event_id(
            event_type=(TaskRetryCancellationReconciliationEventType.CANCELLATION_REQUESTED),
            task_id=task.id,
            series_id=series.series_id,
            attempt=series.attempt,
            cancellation_idempotency_key=cancellation_idempotency_key,
            request_sha256=None,
            reconciliation_idempotency_key=None,
            evidence_sha256=None,
        ),
        type=TaskRetryCancellationReconciliationEventType.CANCELLATION_REQUESTED,
        task_id=task.id,
        series_id=series.series_id,
        attempt=series.attempt,
        causal_budget_id=series.causal_budget_id,
        original_worker_id_sha256=_task_retry_reconciliation_identity_sha256(
            task.worker_id,
            "worker_id",
        ),
        cancellation_idempotency_key_sha256=(
            _task_retry_reconciliation_identity_sha256(
                cancellation_idempotency_key,
                "cancellation_idempotency_key",
            )
        ),
        occurred_at=occurred_at,
    )


def _task_retry_requested_cancellation_settlement(
    task: Task,
    *,
    worker_id: str,
    token_count: int = 0,
    estimated_cost: Decimal = Decimal(0),
) -> TaskRetrySettlementRequest | None:
    """Reconstruct the exact cancellation intent owned by a draining attempt."""

    if not _task_retry_cancellation_requested(task):
        return None
    series = task.retry_series
    payload = task.status_payload
    if (
        task.status not in {TaskStatus.CLAIMED, TaskStatus.RUNNING}
        or series is None
        or series.disposition is not TaskRetrySeriesDisposition.ACTIVE
        or task.worker_id != worker_id
        or task.lease_expires_at is None
        or type(payload) is not dict
        or set(payload) != {"settlement_idempotency_key", "error", "event"}
        or type(payload.get("settlement_idempotency_key")) is not str
        or type(payload.get("error")) is not dict
        or type(payload.get("event")) is not dict
    ):
        raise TaskTerminalizationConflict(
            "Task retry cancellation request conflicts with active ownership."
        )
    expected_key = _task_retry_runtime_idempotency_key(task, "cancellation")
    if payload["settlement_idempotency_key"] != expected_key:
        raise TaskTerminalizationConflict(
            "Task retry cancellation request conflicts with its attempt identity."
        )
    requested_event = TaskRetryCancellationReconciliationEvent.model_validate(payload["event"])
    if requested_event != _task_retry_cancellation_requested_event(
        task,
        occurred_at=requested_event.occurred_at,
    ):
        raise TaskTerminalizationConflict(
            "Task retry cancellation request conflicts with its event identity."
        )
    return TaskRetrySettlementRequest(
        task_id=task.id,
        worker_id=worker_id,
        lease_expires_at=task.lease_expires_at,
        idempotency_key=expected_key,
        causal_budget_id=series.causal_budget_id,
        disposition=TaskRetryAttemptDisposition.CANCELLED,
        error=copy_durable_json_object(payload["error"], "error"),
        token_count=token_count,
        estimated_cost=estimated_cost,
    )


def _validated_task_retry_terminal_accounting(
    *,
    token_count: int,
    estimated_cost: Decimal,
) -> tuple[int, Decimal]:
    """Validate the accounting copied into a runtime-owned terminal settlement."""

    report = TaskRetryAttemptReport(
        idempotency_key="task-retry-terminal-accounting",
        disposition=TaskRetryAttemptDisposition.RETRYABLE_FAILURE,
        error={"code": "runtime_terminal"},
        token_count=token_count,
        estimated_cost=estimated_cost,
    )
    return report.token_count, report.estimated_cost


def _task_retry_attempt_elapsed(task: Task, *, series_now: datetime) -> bool:
    series_now = normalize_utc_datetime(series_now, "series_now")
    series = task.retry_series
    return bool(
        task.status is TaskStatus.PENDING
        and task.session_id is None
        and series is not None
        and series.disposition is TaskRetrySeriesDisposition.ACTIVE
        and series.elapsed_deadline is not None
        and series.elapsed_deadline <= series_now
    )


def _claimed_task_retry_attempt_elapsed(task: Task, *, series_now: datetime) -> bool:
    series_now = normalize_utc_datetime(series_now, "series_now")
    series = task.retry_series
    return bool(
        task.status in {TaskStatus.CLAIMED, TaskStatus.RUNNING}
        and task.session_id is None
        and not _task_retry_cancellation_requested(task)
        and series is not None
        and series.disposition is TaskRetrySeriesDisposition.ACTIVE
        and series.elapsed_deadline is not None
        and series.elapsed_deadline <= series_now
    )


def _elapsed_claimed_task_retry_settlement(
    task: Task,
    *,
    committed_at: datetime,
    token_count: int = 0,
    estimated_cost: Decimal = Decimal(0),
) -> TaskRetrySettlementResult:
    """Finalize a live claimed attempt after store time proves elapsed authority."""

    return _runtime_task_retry_terminal_settlement(
        task,
        operation="elapsed",
        request_disposition=TaskRetryAttemptDisposition.RETRYABLE_FAILURE,
        series_disposition=TaskRetrySeriesDisposition.ELAPSED_EXHAUSTED,
        status=TaskStatus.FAILED,
        error={"code": TaskRetrySeriesDisposition.ELAPSED_EXHAUSTED.value},
        committed_at=committed_at,
        token_count=token_count,
        estimated_cost=estimated_cost,
    )


def _expired_task_retry_settlement(
    task: Task,
    *,
    committed_at: datetime,
    series_now: datetime,
) -> TaskRetrySettlementResult:
    """Finalize an unclaimed attempt whose cumulative elapsed authority expired."""

    committed_at = normalize_utc_datetime(committed_at, "committed_at")
    series_now = normalize_utc_datetime(series_now, "series_now")
    if not _task_retry_attempt_elapsed(task, series_now=series_now):
        raise TaskTerminalizationConflict("Task retry attempt has not elapsed.")
    return _runtime_task_retry_terminal_settlement(
        task,
        operation="expiration",
        request_disposition=TaskRetryAttemptDisposition.RETRYABLE_FAILURE,
        series_disposition=TaskRetrySeriesDisposition.ELAPSED_EXHAUSTED,
        status=TaskStatus.FAILED,
        error={"code": TaskRetrySeriesDisposition.ELAPSED_EXHAUSTED.value},
        committed_at=committed_at,
    )


def _scheduled_task_nonexecution(
    task: Task,
    *,
    eligibility: TaskScheduleEligibility,
    now: datetime,
) -> tuple[Task, TaskRetrySettlementResult | None]:
    """Prepare an unadmitted schedule's terminal task and retry evidence."""
    state = require_schedule(task)
    if state.admitted_at is not None or task.worker_id is not None or task.session_id is not None:
        raise TaskScheduleConflict("Admitted schedules require their execution settlement owner.")
    if eligibility not in {TaskScheduleEligibility.EXPIRED, TaskScheduleEligibility.SKIPPED}:
        raise TaskScheduleConflict("Schedule remains eligible for execution.")
    reason = (
        "schedule_expired" if eligibility is TaskScheduleEligibility.EXPIRED else "schedule_skipped"
    )
    settlement = (
        _cancelled_task_retry_settlement(task, error={"code": reason}, committed_at=now)
        if task.retry_series is not None
        else None
    )
    terminal = task if settlement is None else settlement.task
    terminal = terminal.model_copy(
        update={
            "status": TaskStatus.CANCELLED,
            "status_reason": reason,
            "error": {"code": reason},
            "completed_at": now,
            "updated_at": now,
            "schedule": state.model_copy(update={"revision": schedule_revision_after(state)}),
        }
    )
    if settlement is not None:
        settlement = settlement.model_copy(update={"task": terminal}, deep=True)
    return terminal, settlement


def _cancelled_task_retry_settlement(
    task: Task,
    *,
    error: dict[str, Any] | None,
    committed_at: datetime,
) -> TaskRetrySettlementResult:
    series = task.retry_series
    if (
        series is None
        or series.disposition is not TaskRetrySeriesDisposition.ACTIVE
        or task.status in _TERMINAL_TASK_STATUSES
    ):
        raise TaskTerminalizationConflict("Task retry attempt is not cancellable.")
    return _runtime_task_retry_terminal_settlement(
        task,
        operation="cancellation",
        request_disposition=TaskRetryAttemptDisposition.CANCELLED,
        series_disposition=TaskRetrySeriesDisposition.CANCELLED,
        status=TaskStatus.CANCELLED,
        error=(
            {"code": TaskRetrySeriesDisposition.CANCELLED.value}
            if error is None
            else copy_durable_json_object(error, "error")
        ),
        committed_at=committed_at,
    )


def _runtime_task_retry_terminal_settlement(
    task: Task,
    *,
    operation: str,
    request_disposition: TaskRetryAttemptDisposition,
    series_disposition: TaskRetrySeriesDisposition,
    status: TaskStatus,
    error: dict[str, Any],
    committed_at: datetime,
    token_count: int = 0,
    estimated_cost: Decimal = Decimal(0),
) -> TaskRetrySettlementResult:
    committed_at = normalize_utc_datetime(committed_at, "committed_at")
    series = task.retry_series
    if series is None or series.disposition is not TaskRetrySeriesDisposition.ACTIVE:
        raise TaskTerminalizationConflict("Task retry attempt is not active.")
    request, request_sha256 = _task_retry_runtime_terminal_request(
        task,
        operation=operation,
        request_disposition=request_disposition,
        error=error,
        token_count=token_count,
        estimated_cost=estimated_cost,
    )
    idempotency_key = request.idempotency_key
    cumulative_tokens = series.cumulative_tokens + request.token_count
    if cumulative_tokens > MAX_DURABLE_JSON_INTEGER:
        raise ValueError("Task retry cumulative token count exceeds durable bounds.")
    cumulative_estimated_cost = series.cumulative_estimated_cost + request.estimated_cost
    _bounded_task_retry_decimal(
        cumulative_estimated_cost,
        "cumulative_estimated_cost",
        max_digits=_TASK_RETRY_TOTAL_COST_MAX_DIGITS,
    )
    settled_authority_sha256 = _task_retry_attempt_authority_sha256(
        task_id=task.id,
        task_type=task.type,
        title=task.title,
        description=task.description,
        parent_task_id=task.parent_task_id,
        assigned_agent_name=task.assigned_agent_name,
        available_at=task.available_at,
        created_at=task.created_at,
        task_input=task.input,
        metadata=task.metadata,
        invocation=task.invocation,
        series_id=series.series_id,
        causal_budget_id=series.causal_budget_id,
        attempt=series.attempt,
        policy=series.policy,
        started_at=series.started_at,
        cumulative_tokens=cumulative_tokens,
        cumulative_estimated_cost=cumulative_estimated_cost,
        predecessor_task_id=series.predecessor_task_id,
    )
    settled_series = _task_retry_series_snapshot(
        series_id=series.series_id,
        causal_budget_id=series.causal_budget_id,
        authority_sha256=settled_authority_sha256,
        attempt=series.attempt,
        policy=series.policy,
        started_at=series.started_at,
        cumulative_tokens=cumulative_tokens,
        cumulative_estimated_cost=cumulative_estimated_cost,
        disposition=series_disposition,
        predecessor_task_id=series.predecessor_task_id,
    )
    settled = task.model_copy(
        update={
            "status": status,
            "status_reason": series_disposition.value,
            "status_payload": {
                "retry_series_id": series.series_id,
                "attempt": series.attempt,
                "disposition": series_disposition.value,
                "settlement_idempotency_key": idempotency_key,
                "causal_budget_id": series.causal_budget_id,
                "cumulative_tokens": cumulative_tokens,
                "cumulative_estimated_cost": str(cumulative_estimated_cost),
                "cost_currency": series.policy.cost_currency,
                "next_eligible_at": None,
            },
            "result": None,
            "error": deepcopy(error),
            "worker_id": None,
            "lease_expires_at": None,
            "started_at": task.started_at or committed_at,
            "completed_at": committed_at,
            "updated_at": committed_at,
            "retry_series": settled_series,
        },
        deep=True,
    )
    return TaskRetrySettlementResult(
        task_id=task.id,
        idempotency_key=idempotency_key,
        request_sha256=request_sha256,
        task=settled,
        events=_task_retry_events(settled, occurred_at=committed_at),
        committed_at=committed_at,
    )


def _task_retry_runtime_terminal_request(
    task: Task,
    *,
    operation: str,
    request_disposition: TaskRetryAttemptDisposition,
    error: dict[str, Any],
    token_count: int = 0,
    estimated_cost: Decimal = Decimal(0),
) -> tuple[TaskRetrySettlementRequest, str]:
    """Build the exact request identity shared by mutation and reconciliation."""

    series = task.retry_series
    if series is None or series.disposition is not TaskRetrySeriesDisposition.ACTIVE:
        raise TaskTerminalizationConflict("Task retry attempt is not active.")
    operation = require_clean_nonblank(operation, "operation")
    return prepare_task_retry_settlement(
        TaskRetrySettlementRequest(
            task_id=task.id,
            worker_id=f"cayu-runtime-retry-{operation}",
            lease_expires_at=task.lease_expires_at,
            idempotency_key=_task_retry_runtime_idempotency_key(task, operation),
            causal_budget_id=series.causal_budget_id,
            disposition=request_disposition,
            error=error,
            token_count=token_count,
            estimated_cost=estimated_cost,
            retry_after_seconds=(
                0.0
                if request_disposition is TaskRetryAttemptDisposition.RETRYABLE_FAILURE
                else None
            ),
        )
    )


def _task_retry_backoff_seconds(policy: TaskRetryPolicy, completed_attempt: int) -> float:
    delay = policy.initial_backoff_seconds
    for _ in range(max(0, completed_attempt - 1)):
        if delay >= policy.max_backoff_seconds:
            return policy.max_backoff_seconds
        delay = min(policy.max_backoff_seconds, delay * policy.backoff_multiplier)
    return min(delay, policy.max_backoff_seconds)


def _task_retry_series_snapshot(
    *,
    series_id: str,
    causal_budget_id: str,
    authority_sha256: str,
    attempt: int,
    policy: TaskRetryPolicy,
    started_at: datetime,
    cumulative_tokens: int = 0,
    cumulative_estimated_cost: Decimal = Decimal(0),
    disposition: TaskRetrySeriesDisposition = TaskRetrySeriesDisposition.ACTIVE,
    predecessor_task_id: str | None = None,
    successor_task_id: str | None = None,
    next_eligible_at: datetime | None = None,
) -> TaskRetrySeriesSnapshot:
    return TaskRetrySeriesSnapshot(
        series_id=series_id,
        causal_budget_id=causal_budget_id,
        authority_sha256=authority_sha256,
        attempt=attempt,
        policy=policy,
        started_at=started_at,
        cumulative_tokens=cumulative_tokens,
        cumulative_estimated_cost=cumulative_estimated_cost,
        attempts_remaining=max(0, policy.max_attempts - attempt),
        tokens_remaining=(
            None
            if policy.max_total_tokens is None
            else max(0, policy.max_total_tokens - cumulative_tokens)
        ),
        estimated_cost_remaining=(
            None
            if policy.max_estimated_cost is None
            else max(Decimal(0), policy.max_estimated_cost - cumulative_estimated_cost)
        ),
        elapsed_deadline=(
            None
            if policy.max_elapsed_seconds is None
            else started_at + timedelta(seconds=policy.max_elapsed_seconds)
        ),
        disposition=disposition,
        predecessor_task_id=predecessor_task_id,
        successor_task_id=successor_task_id,
        next_eligible_at=next_eligible_at,
    )


def _task_retry_event_id(
    *,
    series_id: str,
    attempt: int,
    event_type: TaskRetryEventType,
) -> str:
    material = canonical_durable_json_bytes(
        {
            "schema": "cayu.task-retry-event.v1",
            "series_id": series_id,
            "attempt": attempt,
            "type": event_type.value,
        },
        "task_retry_event_id",
    )
    return f"task-retry-event:v1:{sha256(material).hexdigest()}"


def _task_retry_events(
    task: Task,
    *,
    occurred_at: datetime,
) -> tuple[TaskRetryEvent, TaskRetryEvent]:
    series = task.retry_series
    if series is None:  # pragma: no cover - internal construction invariant
        raise AssertionError("Retry settlement event requires retry-series evidence.")
    event_fields = {
        "task_id": task.id,
        "series_id": series.series_id,
        "causal_budget_id": series.causal_budget_id,
        "attempt": series.attempt,
        "disposition": series.disposition,
        "occurred_at": occurred_at,
        "attempts_remaining": series.attempts_remaining,
        "tokens_remaining": series.tokens_remaining,
        "estimated_cost_remaining": series.estimated_cost_remaining,
        "cost_currency": series.policy.cost_currency,
        "elapsed_deadline": series.elapsed_deadline,
        "next_eligible_at": series.next_eligible_at,
    }
    outcome_type = (
        TaskRetryEventType.RETRY_SCHEDULED
        if series.disposition is TaskRetrySeriesDisposition.RETRY_SCHEDULED
        else TaskRetryEventType.SERIES_TERMINAL
    )
    return (
        TaskRetryEvent(
            id=_task_retry_event_id(
                series_id=series.series_id,
                attempt=series.attempt,
                event_type=TaskRetryEventType.ATTEMPT_SETTLED,
            ),
            type=TaskRetryEventType.ATTEMPT_SETTLED,
            **event_fields,
        ),
        TaskRetryEvent(
            id=_task_retry_event_id(
                series_id=series.series_id,
                attempt=series.attempt,
                event_type=outcome_type,
            ),
            type=outcome_type,
            **event_fields,
        ),
    )


def _settled_task_retry_attempt(
    task: Task,
    request: TaskRetrySettlementRequest,
    *,
    now: datetime,
    series_now: datetime,
) -> tuple[Task, Task | None]:
    series = task.retry_series
    if series is None:
        raise ValueError("Task does not belong to a retry series.")
    if series.disposition is not TaskRetrySeriesDisposition.ACTIVE:
        raise TaskTerminalizationConflict("Task retry attempt is not active.")
    now = normalize_utc_datetime(now, "now")
    series_now = normalize_utc_datetime(series_now, "series_now")
    if request.causal_budget_id != series.causal_budget_id:
        raise TaskTerminalizationConflict(
            "Task retry settlement conflicts with the series causal budget."
        )
    requested_cancellation = _task_retry_requested_cancellation_settlement(
        task,
        worker_id=request.worker_id,
        token_count=request.token_count,
        estimated_cost=request.estimated_cost,
    )
    if requested_cancellation is not None and request != requested_cancellation:
        raise TaskTerminalizationConflict("Task retry attempt has a pending cancellation request.")
    token_authority_exceeded = (
        series.tokens_remaining is not None and request.token_count > series.tokens_remaining
    )
    cost_authority_exceeded = (
        series.estimated_cost_remaining is not None
        and request.estimated_cost > series.estimated_cost_remaining
    )
    cumulative_tokens = series.cumulative_tokens + request.token_count
    if cumulative_tokens > MAX_DURABLE_JSON_INTEGER:
        raise ValueError("Task retry cumulative token count exceeds durable bounds.")
    cumulative_cost = series.cumulative_estimated_cost + request.estimated_cost
    _bounded_task_retry_decimal(
        cumulative_cost,
        "cumulative_estimated_cost",
        max_digits=_TASK_RETRY_TOTAL_COST_MAX_DIGITS,
    )

    disposition: TaskRetrySeriesDisposition
    successor: Task | None = None
    next_eligible_at: datetime | None = None
    successor_task_id: str | None = None
    outcome_result = deepcopy(request.result)
    outcome_error = deepcopy(request.error)
    elapsed_deadline = series.elapsed_deadline
    if requested_cancellation is not None:
        status = TaskStatus.CANCELLED
        disposition = TaskRetrySeriesDisposition.CANCELLED
    elif elapsed_deadline is not None and series_now >= elapsed_deadline:
        status = TaskStatus.FAILED
        disposition = TaskRetrySeriesDisposition.ELAPSED_EXHAUSTED
        outcome_result = None
        outcome_error = {"code": disposition.value}
    elif token_authority_exceeded:
        status = TaskStatus.FAILED
        disposition = TaskRetrySeriesDisposition.TOKENS_EXHAUSTED
        outcome_result = None
        outcome_error = {"code": disposition.value}
    elif cost_authority_exceeded:
        status = TaskStatus.FAILED
        disposition = TaskRetrySeriesDisposition.COST_EXHAUSTED
        outcome_result = None
        outcome_error = {"code": disposition.value}
    elif request.disposition is TaskRetryAttemptDisposition.SUCCEEDED:
        status = TaskStatus.COMPLETED
        disposition = TaskRetrySeriesDisposition.SUCCEEDED
    elif request.disposition is TaskRetryAttemptDisposition.CANCELLED:
        status = TaskStatus.CANCELLED
        disposition = TaskRetrySeriesDisposition.CANCELLED
    elif request.disposition is TaskRetryAttemptDisposition.NON_RETRYABLE_FAILURE:
        status = TaskStatus.FAILED
        disposition = TaskRetrySeriesDisposition.NON_RETRYABLE_FAILURE
    else:
        status = TaskStatus.FAILED
        policy = series.policy
        backoff_seconds = max(
            _task_retry_backoff_seconds(policy, series.attempt),
            request.retry_after_seconds or 0.0,
        )
        next_eligible_at = series_now + timedelta(seconds=backoff_seconds)
        if series.attempt >= policy.max_attempts:
            disposition = TaskRetrySeriesDisposition.ATTEMPTS_EXHAUSTED
        elif elapsed_deadline is not None and next_eligible_at >= elapsed_deadline:
            disposition = TaskRetrySeriesDisposition.ELAPSED_EXHAUSTED
        elif policy.max_total_tokens is not None and cumulative_tokens >= policy.max_total_tokens:
            disposition = TaskRetrySeriesDisposition.TOKENS_EXHAUSTED
        elif policy.max_estimated_cost is not None and cumulative_cost >= policy.max_estimated_cost:
            disposition = TaskRetrySeriesDisposition.COST_EXHAUSTED
        else:
            disposition = TaskRetrySeriesDisposition.RETRY_SCHEDULED
            successor_task_id = _task_retry_successor_id(series.series_id, series.attempt + 1)

    settled_authority_sha256 = _task_retry_attempt_authority_sha256(
        task_id=task.id,
        task_type=task.type,
        title=task.title,
        description=task.description,
        parent_task_id=task.parent_task_id,
        assigned_agent_name=task.assigned_agent_name,
        available_at=task.available_at,
        created_at=task.created_at,
        task_input=task.input,
        metadata=task.metadata,
        invocation=task.invocation,
        series_id=series.series_id,
        causal_budget_id=series.causal_budget_id,
        attempt=series.attempt,
        policy=series.policy,
        started_at=series.started_at,
        cumulative_tokens=cumulative_tokens,
        cumulative_estimated_cost=cumulative_cost,
        predecessor_task_id=series.predecessor_task_id,
    )
    settled_series = _task_retry_series_snapshot(
        series_id=series.series_id,
        causal_budget_id=series.causal_budget_id,
        authority_sha256=settled_authority_sha256,
        attempt=series.attempt,
        policy=series.policy,
        started_at=series.started_at,
        cumulative_tokens=cumulative_tokens,
        cumulative_estimated_cost=cumulative_cost,
        disposition=disposition,
        predecessor_task_id=series.predecessor_task_id,
        successor_task_id=successor_task_id,
        next_eligible_at=next_eligible_at if successor_task_id is not None else None,
    )
    settled = task.model_copy(
        update={
            "status": status,
            "status_reason": disposition.value,
            "status_payload": {
                "retry_series_id": series.series_id,
                "attempt": series.attempt,
                "disposition": disposition.value,
                "settlement_idempotency_key": request.idempotency_key,
                "causal_budget_id": series.causal_budget_id,
                "cumulative_tokens": cumulative_tokens,
                "cumulative_estimated_cost": str(cumulative_cost),
                "cost_currency": series.policy.cost_currency,
                "next_eligible_at": (
                    None
                    if settled_series.next_eligible_at is None
                    else settled_series.next_eligible_at.isoformat()
                ),
            },
            "result": outcome_result,
            "error": outcome_error,
            "worker_id": None,
            "lease_expires_at": None,
            "started_at": task.started_at or now,
            "completed_at": now,
            "updated_at": now,
            "retry_series": settled_series,
        },
        deep=True,
    )
    if successor_task_id is not None:
        successor_authority_sha256 = _task_retry_attempt_authority_sha256(
            task_id=successor_task_id,
            task_type=task.type,
            title=task.title,
            description=task.description,
            parent_task_id=task.parent_task_id,
            assigned_agent_name=task.assigned_agent_name,
            available_at=next_eligible_at,
            created_at=now,
            task_input=task.input,
            metadata=task.metadata,
            invocation=task.invocation,
            series_id=series.series_id,
            causal_budget_id=series.causal_budget_id,
            attempt=series.attempt + 1,
            policy=series.policy,
            started_at=series.started_at,
            cumulative_tokens=cumulative_tokens,
            cumulative_estimated_cost=cumulative_cost,
            predecessor_task_id=task.id,
        )
        successor_series = _task_retry_series_snapshot(
            series_id=series.series_id,
            causal_budget_id=series.causal_budget_id,
            authority_sha256=successor_authority_sha256,
            attempt=series.attempt + 1,
            policy=series.policy,
            started_at=series.started_at,
            cumulative_tokens=cumulative_tokens,
            cumulative_estimated_cost=cumulative_cost,
            predecessor_task_id=task.id,
        )
        successor = Task(
            id=successor_task_id,
            type=task.type,
            title=task.title,
            description=task.description,
            status=TaskStatus.PENDING,
            parent_task_id=task.parent_task_id,
            assigned_agent_name=task.assigned_agent_name,
            available_at=next_eligible_at,
            input=copy_durable_json_object(task.input, "input"),
            metadata=copy_durable_metadata(task.metadata),
            created_at=now,
            updated_at=now,
            invocation=copy_task_invocation(task.invocation),
            retry_series=successor_series,
        )
    return settled, successor
