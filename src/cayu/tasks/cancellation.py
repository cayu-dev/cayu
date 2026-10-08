"""Task cancellation and reconciliation records, identities and replay validation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from enum import StrEnum
from hashlib import sha256
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from cayu._clock import normalize_utc_datetime
from cayu._validation import canonical_durable_json_bytes, revalidate_model_input
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.approvals.tools import ResolutionActor, resolution_actor_payload
from cayu.tasks.records import (
    _TASK_RETRY_RECONCILIATION_IDENTITY_MAX_BYTES,
    Task,
    TaskStatus,
    _validate_task_retry_reconciliation_identity,
    copy_task,
)
from cayu.tasks.terminalization import (
    TaskTerminalizationConflict,
    TaskTerminalizationReceipt,
    TaskTerminalizationRequest,
    TaskTerminalKind,
    _task_terminalization_request_matches_sha256,
    _validate_task_terminalization_idempotency_key,
    prepare_task_terminalization,
)

_TASK_CANCELLATION_REQUESTED_REASON = "cancellation_requested"


_TASK_RETRY_CANCELLATION_REQUESTED_REASON = "retry_cancellation_requested"


_TASK_RETRY_RECONCILIATION_EVIDENCE_ID_MAX_BYTES = 256


_TASK_RETRY_RECONCILIATION_VERSION_MAX_BYTES = 64


class TaskRetryCancellationReconciliationOutcome(StrEnum):
    """Application-validated state of the cancelled attempt's external effect."""

    QUIESCENT = "quiescent"
    EFFECT_COMPLETED = "effect_completed"
    EFFECT_FAILED = "effect_failed"
    UNRESOLVED = "unresolved"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    UNSUPPORTED = "unsupported"


class TaskCancellationReconciliationOutcome(StrEnum):
    """Application-validated state of an ordinary task's cancelled effect."""

    QUIESCENT = "quiescent"
    EFFECT_COMPLETED = "effect_completed"
    EFFECT_FAILED = "effect_failed"
    UNRESOLVED = "unresolved"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    UNSUPPORTED = "unsupported"


class TaskRetryCancellationReconciliationEventType(StrEnum):
    """Stable bounded lifecycle evidence for cancellation reconciliation."""

    CANCELLATION_REQUESTED = "task.retry.cancellation_requested"
    STARTED = "task.retry.cancellation_reconciliation_started"
    RECONCILED = "task.retry.cancellation_reconciled"
    REJECTED = "task.retry.cancellation_reconciliation_rejected"
    CONFLICT = "task.retry.cancellation_reconciliation_conflict"


_POSITIVE_TASK_RETRY_CANCELLATION_RECONCILIATION_OUTCOMES = frozenset(
    {
        TaskRetryCancellationReconciliationOutcome.QUIESCENT,
        TaskRetryCancellationReconciliationOutcome.EFFECT_COMPLETED,
        TaskRetryCancellationReconciliationOutcome.EFFECT_FAILED,
    }
)


_POSITIVE_TASK_CANCELLATION_RECONCILIATION_OUTCOMES = frozenset(
    {
        TaskCancellationReconciliationOutcome.QUIESCENT,
        TaskCancellationReconciliationOutcome.EFFECT_COMPLETED,
        TaskCancellationReconciliationOutcome.EFFECT_FAILED,
    }
)


def _task_retry_reconciliation_identity_is_bounded(value: str) -> bool:
    return len(value.encode("utf-8")) <= _TASK_RETRY_RECONCILIATION_IDENTITY_MAX_BYTES


def _validate_lowercase_sha256(value: str, field_name: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest.")
    return value


class TaskRetryCancellationReconciliationEvidence(BaseModel):
    """Bounded application-validated evidence; raw external payloads are excluded."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    outcome: TaskRetryCancellationReconciliationOutcome
    validator_id: str
    validator_version: str
    evidence_id: str
    evidence_sha256: str
    validated_at: datetime
    execution_profile_fingerprint: str | None = None
    effect_fingerprint: str | None = None

    @field_validator("validator_id", "evidence_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _validate_task_retry_reconciliation_identity(
            value,
            info.field_name,
            max_bytes=_TASK_RETRY_RECONCILIATION_EVIDENCE_ID_MAX_BYTES,
        )

    @field_validator("validator_version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        return _validate_task_retry_reconciliation_identity(
            value,
            "validator_version",
            max_bytes=_TASK_RETRY_RECONCILIATION_VERSION_MAX_BYTES,
        )

    @field_validator(
        "evidence_sha256",
        "execution_profile_fingerprint",
        "effect_fingerprint",
    )
    @classmethod
    def validate_sha256(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _validate_lowercase_sha256(value, info.field_name)

    @field_validator("validated_at")
    @classmethod
    def normalize_validated_at(cls, value: datetime) -> datetime:
        return normalize_utc_datetime(value, "validated_at")


class TaskCancellationReconciliationEvidence(TaskRetryCancellationReconciliationEvidence):
    """Ordinary-task evidence with the shared bounded validator schema."""

    outcome: TaskCancellationReconciliationOutcome


class TaskRetryCancellationReconciliationEvent(BaseModel):
    """Secret-free event projection for one owner-lost cancellation transition."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    id: str
    type: TaskRetryCancellationReconciliationEventType
    task_id: str
    series_id: str
    attempt: StrictInt = Field(ge=1, le=100)
    causal_budget_id: str
    original_worker_id_sha256: str
    cancellation_idempotency_key_sha256: str
    request_sha256: str | None = None
    reconciliation_idempotency_key_sha256: str | None = None
    evidence_sha256: str | None = None
    actor_sha256: str | None = None
    outcome: TaskRetryCancellationReconciliationOutcome | None = None
    occurred_at: datetime

    @field_validator("id", "task_id", "series_id", "causal_budget_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _validate_task_retry_reconciliation_identity(value, info.field_name)

    @field_validator(
        "original_worker_id_sha256",
        "cancellation_idempotency_key_sha256",
        "request_sha256",
        "reconciliation_idempotency_key_sha256",
        "evidence_sha256",
        "actor_sha256",
    )
    @classmethod
    def validate_sha256(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _validate_lowercase_sha256(value, info.field_name)

    @field_validator("occurred_at")
    @classmethod
    def normalize_occurred_at(cls, value: datetime) -> datetime:
        return normalize_utc_datetime(value, "occurred_at")

    @model_validator(mode="after")
    def validate_event_shape(self) -> TaskRetryCancellationReconciliationEvent:
        requested = self.type is TaskRetryCancellationReconciliationEventType.CANCELLATION_REQUESTED
        reconciliation_fields = (
            self.request_sha256,
            self.reconciliation_idempotency_key_sha256,
            self.evidence_sha256,
            self.actor_sha256,
            self.outcome,
        )
        if requested and any(value is not None for value in reconciliation_fields):
            raise ValueError("Cancellation-requested events cannot claim reconciliation evidence.")
        if not requested and any(value is None for value in reconciliation_fields):
            raise ValueError("Reconciliation events require bounded request evidence.")
        return self


class TaskRetryCancellationReconciliationRequest(BaseModel):
    """Exact owner-lost cancellation identity plus application-validated evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    task_id: str
    series_id: str
    attempt: StrictInt = Field(ge=1, le=100)
    causal_budget_id: str
    original_worker_id: str
    original_lease_expires_at: datetime
    cancellation_requested_at: datetime
    cancellation_idempotency_key: str
    expected_status_reason: Literal["retry_cancellation_requested"] = (
        _TASK_RETRY_CANCELLATION_REQUESTED_REASON
    )
    reconciliation_idempotency_key: str
    reconciliation_requested_at: datetime
    reconciled_by: ResolutionActor
    evidence: TaskRetryCancellationReconciliationEvidence
    expected_execution_profile_fingerprint: str | None = None
    expected_effect_fingerprint: str | None = None

    @field_validator(
        "task_id",
        "series_id",
        "causal_budget_id",
        "original_worker_id",
    )
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _validate_task_retry_reconciliation_identity(value, info.field_name)

    @field_validator("cancellation_idempotency_key", "reconciliation_idempotency_key")
    @classmethod
    def validate_idempotency_key(cls, value: str) -> str:
        return _validate_task_terminalization_idempotency_key(value)

    @field_validator(
        "original_lease_expires_at",
        "cancellation_requested_at",
        "reconciliation_requested_at",
    )
    @classmethod
    def normalize_identity_datetime(cls, value: datetime, info) -> datetime:
        return normalize_utc_datetime(value, info.field_name)

    @field_validator("reconciled_by", mode="before")
    @classmethod
    def copy_actor(cls, value: object) -> object:
        return revalidate_model_input(value, ResolutionActor)

    @field_validator("evidence", mode="before")
    @classmethod
    def copy_evidence(cls, value: object) -> object:
        return revalidate_model_input(value, TaskRetryCancellationReconciliationEvidence)

    @field_validator(
        "expected_execution_profile_fingerprint",
        "expected_effect_fingerprint",
    )
    @classmethod
    def validate_expected_fingerprint(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _validate_lowercase_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_evidence_identity(self) -> TaskRetryCancellationReconciliationRequest:
        if self.reconciled_by.source is None:
            raise ValueError("Task retry reconciliation requires an actor provenance source.")
        if (
            self.expected_execution_profile_fingerprint
            != self.evidence.execution_profile_fingerprint
        ):
            raise ValueError("Reconciliation evidence conflicts with the expected profile.")
        if self.expected_effect_fingerprint != self.evidence.effect_fingerprint:
            raise ValueError("Reconciliation evidence conflicts with the expected effect.")
        if self.evidence.validated_at < self.cancellation_requested_at:
            raise ValueError("Reconciliation evidence predates the cancellation request.")
        if self.reconciliation_requested_at < self.evidence.validated_at:
            raise ValueError("Reconciliation request predates its validated evidence.")
        for field_name, value in (
            ("reconciled_by.subject", self.reconciled_by.subject),
            ("reconciled_by.tenant", self.reconciled_by.tenant),
        ):
            if value is not None:
                _validate_task_retry_reconciliation_identity(value, field_name)
        return self


class TaskRetryCancellationReconciliation(BaseModel):
    """Durable positive evidence attached to the ordinary cancellation receipt."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    request_sha256: str
    task_id: str
    series_id: str
    attempt: StrictInt = Field(ge=1, le=100)
    causal_budget_id: str
    original_worker_id: str
    original_lease_expires_at: datetime
    cancellation_requested_at: datetime
    cancellation_idempotency_key: str
    reconciliation_idempotency_key: str
    reconciliation_requested_at: datetime
    reconciled_by: ResolutionActor
    evidence: TaskRetryCancellationReconciliationEvidence
    events: tuple[TaskRetryCancellationReconciliationEvent, ...] = Field(
        min_length=3,
        max_length=3,
    )

    @field_validator("request_sha256")
    @classmethod
    def validate_request_sha256(cls, value: str) -> str:
        return _validate_lowercase_sha256(value, "request_sha256")

    @field_validator(
        "task_id",
        "series_id",
        "causal_budget_id",
        "original_worker_id",
    )
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _validate_task_retry_reconciliation_identity(value, info.field_name)

    @field_validator("cancellation_idempotency_key", "reconciliation_idempotency_key")
    @classmethod
    def validate_idempotency_key(cls, value: str) -> str:
        return _validate_task_terminalization_idempotency_key(value)

    @field_validator(
        "original_lease_expires_at",
        "cancellation_requested_at",
        "reconciliation_requested_at",
    )
    @classmethod
    def normalize_identity_datetime(cls, value: datetime, info) -> datetime:
        return normalize_utc_datetime(value, info.field_name)

    @field_validator("reconciled_by", mode="before")
    @classmethod
    def copy_actor(cls, value: object) -> object:
        return revalidate_model_input(value, ResolutionActor)

    @field_validator("evidence", mode="before")
    @classmethod
    def copy_evidence(cls, value: object) -> object:
        return revalidate_model_input(value, TaskRetryCancellationReconciliationEvidence)

    @field_validator("events", mode="before")
    @classmethod
    def copy_events(cls, value: object) -> object:
        if isinstance(value, (str, bytes, bytearray, Mapping, BaseModel)):
            return value
        return tuple(
            revalidate_model_input(event, TaskRetryCancellationReconciliationEvent)
            for event in cast("Iterable[object]", value)
        )

    @model_validator(mode="after")
    def validate_projection(self) -> TaskRetryCancellationReconciliation:
        if self.reconciled_by.claims:
            raise ValueError("Durable reconciliation actors cannot retain authorization claims.")
        for field_name, value in (
            ("reconciled_by.subject", self.reconciled_by.subject),
            ("reconciled_by.tenant", self.reconciled_by.tenant),
        ):
            if value is not None:
                _validate_task_retry_reconciliation_identity(value, field_name)
        expected_types = (
            TaskRetryCancellationReconciliationEventType.CANCELLATION_REQUESTED,
            TaskRetryCancellationReconciliationEventType.STARTED,
            TaskRetryCancellationReconciliationEventType.RECONCILED,
        )
        original_worker_sha256 = _task_retry_reconciliation_identity_sha256(
            self.original_worker_id,
            "original_worker_id",
        )
        cancellation_sha256 = _task_retry_reconciliation_identity_sha256(
            self.cancellation_idempotency_key,
            "cancellation_idempotency_key",
        )
        reconciliation_sha256 = _task_retry_reconciliation_identity_sha256(
            self.reconciliation_idempotency_key,
            "reconciliation_idempotency_key",
        )
        actor_sha256 = _task_retry_reconciliation_actor_sha256(self.reconciled_by)
        for index, (event, event_type) in enumerate(zip(self.events, expected_types, strict=True)):
            requested = index == 0
            if (
                event.type is not event_type
                or event.task_id != self.task_id
                or event.series_id != self.series_id
                or event.attempt != self.attempt
                or event.causal_budget_id != self.causal_budget_id
                or event.original_worker_id_sha256 != original_worker_sha256
                or event.cancellation_idempotency_key_sha256 != cancellation_sha256
                or event.request_sha256 != (None if requested else self.request_sha256)
                or event.reconciliation_idempotency_key_sha256
                != (None if requested else reconciliation_sha256)
                or event.evidence_sha256 != (None if requested else self.evidence.evidence_sha256)
                or event.actor_sha256 != (None if requested else actor_sha256)
                or event.outcome != (None if requested else self.evidence.outcome)
                or event.occurred_at
                != (
                    self.cancellation_requested_at
                    if requested
                    else (
                        self.reconciliation_requested_at
                        if event_type is TaskRetryCancellationReconciliationEventType.STARTED
                        else self.events[2].occurred_at
                    )
                )
                or event.id
                != _task_retry_cancellation_reconciliation_event_id(
                    event_type=event_type,
                    task_id=self.task_id,
                    series_id=self.series_id,
                    attempt=self.attempt,
                    cancellation_idempotency_key=self.cancellation_idempotency_key,
                    request_sha256=None if requested else self.request_sha256,
                    reconciliation_idempotency_key=(
                        None if requested else self.reconciliation_idempotency_key
                    ),
                    evidence_sha256=(None if requested else self.evidence.evidence_sha256),
                )
            ):
                raise ValueError("Reconciliation event conflicts with its durable projection.")
        return self


class TaskRetryCancellationReconciliationConflict(TaskTerminalizationConflict):
    """A reconciliation request lost an exact identity or settlement race."""

    def __init__(
        self,
        message: str,
        *,
        event: TaskRetryCancellationReconciliationEvent,
    ) -> None:
        super().__init__(message)
        self.event = TaskRetryCancellationReconciliationEvent.model_validate(
            event.model_dump(mode="python")
        )


class TaskRetryCancellationReconciliationRejected(ValueError):
    """Application evidence was non-positive, so the attempt remains fenced."""

    def __init__(
        self,
        *,
        outcome: TaskRetryCancellationReconciliationOutcome,
        event: TaskRetryCancellationReconciliationEvent,
    ) -> None:
        super().__init__(f"Task retry cancellation reconciliation is {outcome.value}.")
        self.outcome = outcome
        self.event = TaskRetryCancellationReconciliationEvent.model_validate(
            event.model_dump(mode="python")
        )


class _TaskRetryCancellationReconciliationRejectionRecord(BaseModel):
    """Durable binding for one non-positive reconciliation request."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    task_id: str
    reconciliation_idempotency_key: str
    request_sha256: str
    outcome: TaskRetryCancellationReconciliationOutcome
    event: TaskRetryCancellationReconciliationEvent
    recorded_at: datetime

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, value: str) -> str:
        return _validate_task_retry_reconciliation_identity(value, "task_id")

    @field_validator("reconciliation_idempotency_key")
    @classmethod
    def validate_reconciliation_idempotency_key(cls, value: str) -> str:
        return _validate_task_terminalization_idempotency_key(value)

    @field_validator("request_sha256")
    @classmethod
    def validate_request_sha256(cls, value: str) -> str:
        return _validate_lowercase_sha256(value, "request_sha256")

    @field_validator("event", mode="before")
    @classmethod
    def copy_event(cls, value: object) -> object:
        return revalidate_model_input(value, TaskRetryCancellationReconciliationEvent)

    @field_validator("recorded_at")
    @classmethod
    def normalize_recorded_at(cls, value: datetime) -> datetime:
        return normalize_utc_datetime(value, "recorded_at")

    @model_validator(mode="after")
    def validate_rejection(self) -> _TaskRetryCancellationReconciliationRejectionRecord:
        event = self.event
        if (
            self.outcome in _POSITIVE_TASK_RETRY_CANCELLATION_RECONCILIATION_OUTCOMES
            or event.type is not TaskRetryCancellationReconciliationEventType.REJECTED
            or event.task_id != self.task_id
            or event.request_sha256 != self.request_sha256
            or event.reconciliation_idempotency_key_sha256
            != _task_retry_reconciliation_identity_sha256(
                self.reconciliation_idempotency_key,
                "reconciliation_idempotency_key",
            )
            or event.outcome is not self.outcome
            or event.occurred_at > self.recorded_at
        ):
            raise ValueError("Reconciliation rejection conflicts with its request binding.")
        return self


class TaskCancellationReconciliationEventType(StrEnum):
    """Stable bounded lifecycle evidence for ordinary-task cancellation recovery."""

    CANCELLATION_REQUESTED = "task.cancellation_requested"
    STARTED = "task.cancellation_reconciliation_started"
    RECONCILED = "task.cancellation_reconciled"
    REJECTED = "task.cancellation_reconciliation_rejected"
    CONFLICT = "task.cancellation_reconciliation_conflict"


class TaskCancellationReconciliationEvent(BaseModel):
    """Secret-free event projection for one owner-lost ordinary cancellation."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    id: str
    type: TaskCancellationReconciliationEventType
    task_id: str
    original_worker_id_sha256: str
    cancellation_idempotency_key_sha256: str
    request_sha256: str | None = None
    reconciliation_idempotency_key_sha256: str | None = None
    evidence_sha256: str | None = None
    actor_sha256: str | None = None
    outcome: TaskCancellationReconciliationOutcome | None = None
    occurred_at: datetime

    @field_validator("id", "task_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _validate_task_retry_reconciliation_identity(value, info.field_name)

    @field_validator(
        "original_worker_id_sha256",
        "cancellation_idempotency_key_sha256",
        "request_sha256",
        "reconciliation_idempotency_key_sha256",
        "evidence_sha256",
        "actor_sha256",
    )
    @classmethod
    def validate_sha256(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _validate_lowercase_sha256(value, info.field_name)

    @field_validator("occurred_at")
    @classmethod
    def normalize_occurred_at(cls, value: datetime) -> datetime:
        return normalize_utc_datetime(value, "occurred_at")

    @model_validator(mode="after")
    def validate_event_shape(self) -> TaskCancellationReconciliationEvent:
        requested = self.type is TaskCancellationReconciliationEventType.CANCELLATION_REQUESTED
        reconciliation_fields = (
            self.request_sha256,
            self.reconciliation_idempotency_key_sha256,
            self.evidence_sha256,
            self.actor_sha256,
            self.outcome,
        )
        if requested and any(value is not None for value in reconciliation_fields):
            raise ValueError("Cancellation-requested events cannot claim reconciliation evidence.")
        if not requested and any(value is None for value in reconciliation_fields):
            raise ValueError("Reconciliation events require bounded request evidence.")
        return self


class TaskCancellationReconciliationRequest(BaseModel):
    """Exact owner-lost ordinary cancellation plus application-validated evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    task_id: str
    original_worker_id: str
    original_handoff_id: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    original_lease_expires_at: datetime
    cancellation_requested_at: datetime
    cancellation_idempotency_key: str
    expected_status_reason: Literal["cancellation_requested"] = _TASK_CANCELLATION_REQUESTED_REASON
    reconciliation_idempotency_key: str
    reconciliation_requested_at: datetime
    reconciled_by: ResolutionActor
    evidence: TaskCancellationReconciliationEvidence
    expected_execution_profile_fingerprint: str | None = None
    expected_effect_fingerprint: str | None = None

    @field_validator("task_id", "original_worker_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _validate_task_retry_reconciliation_identity(value, info.field_name)

    @field_validator("original_handoff_id")
    @classmethod
    def validate_original_handoff_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, "original_handoff_id")

    @field_validator("cancellation_idempotency_key", "reconciliation_idempotency_key")
    @classmethod
    def validate_idempotency_key(cls, value: str) -> str:
        return _validate_task_terminalization_idempotency_key(value)

    @field_validator(
        "original_lease_expires_at",
        "cancellation_requested_at",
        "reconciliation_requested_at",
    )
    @classmethod
    def normalize_identity_datetime(cls, value: datetime, info) -> datetime:
        return normalize_utc_datetime(value, info.field_name)

    @field_validator("reconciled_by", mode="before")
    @classmethod
    def copy_actor(cls, value: object) -> object:
        return revalidate_model_input(value, ResolutionActor)

    @field_validator("evidence", mode="before")
    @classmethod
    def copy_evidence(cls, value: object) -> object:
        return revalidate_model_input(value, TaskCancellationReconciliationEvidence)

    @field_validator(
        "expected_execution_profile_fingerprint",
        "expected_effect_fingerprint",
    )
    @classmethod
    def validate_expected_fingerprint(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _validate_lowercase_sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_evidence_identity(self) -> TaskCancellationReconciliationRequest:
        if self.reconciled_by.source is None:
            raise ValueError("Task cancellation reconciliation requires actor provenance.")
        if (
            self.expected_execution_profile_fingerprint
            != self.evidence.execution_profile_fingerprint
        ):
            raise ValueError("Reconciliation evidence conflicts with the expected profile.")
        if self.expected_effect_fingerprint != self.evidence.effect_fingerprint:
            raise ValueError("Reconciliation evidence conflicts with the expected effect.")
        if self.evidence.validated_at < self.cancellation_requested_at:
            raise ValueError("Reconciliation evidence predates the cancellation request.")
        if self.reconciliation_requested_at < self.evidence.validated_at:
            raise ValueError("Reconciliation request predates its validated evidence.")
        for field_name, value in (
            ("reconciled_by.subject", self.reconciled_by.subject),
            ("reconciled_by.tenant", self.reconciled_by.tenant),
        ):
            if value is not None:
                _validate_task_retry_reconciliation_identity(value, field_name)
        return self


class TaskCancellationReconciliation(BaseModel):
    """Durable positive evidence attached to an ordinary cancellation receipt."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    request_sha256: str
    task_id: str
    original_worker_id: str
    original_handoff_id: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    original_lease_expires_at: datetime
    cancellation_requested_at: datetime
    cancellation_idempotency_key: str
    reconciliation_idempotency_key: str
    reconciliation_requested_at: datetime
    reconciled_by: ResolutionActor
    evidence: TaskCancellationReconciliationEvidence
    events: tuple[TaskCancellationReconciliationEvent, ...] = Field(
        min_length=3,
        max_length=3,
    )

    @field_validator("request_sha256")
    @classmethod
    def validate_request_sha256(cls, value: str) -> str:
        return _validate_lowercase_sha256(value, "request_sha256")

    @field_validator("task_id", "original_worker_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _validate_task_retry_reconciliation_identity(value, info.field_name)

    @field_validator("original_handoff_id")
    @classmethod
    def validate_original_handoff_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, "original_handoff_id")

    @field_validator("cancellation_idempotency_key", "reconciliation_idempotency_key")
    @classmethod
    def validate_idempotency_key(cls, value: str) -> str:
        return _validate_task_terminalization_idempotency_key(value)

    @field_validator(
        "original_lease_expires_at",
        "cancellation_requested_at",
        "reconciliation_requested_at",
    )
    @classmethod
    def normalize_identity_datetime(cls, value: datetime, info) -> datetime:
        return normalize_utc_datetime(value, info.field_name)

    @field_validator("reconciled_by", mode="before")
    @classmethod
    def copy_actor(cls, value: object) -> object:
        return revalidate_model_input(value, ResolutionActor)

    @field_validator("evidence", mode="before")
    @classmethod
    def copy_evidence(cls, value: object) -> object:
        return revalidate_model_input(value, TaskCancellationReconciliationEvidence)

    @field_validator("events", mode="before")
    @classmethod
    def copy_events(cls, value: object) -> object:
        if isinstance(value, (str, bytes, bytearray, Mapping, BaseModel)):
            return value
        return tuple(
            revalidate_model_input(event, TaskCancellationReconciliationEvent)
            for event in cast("Iterable[object]", value)
        )

    @model_validator(mode="after")
    def validate_projection(self) -> TaskCancellationReconciliation:
        if self.reconciled_by.claims:
            raise ValueError("Durable reconciliation actors cannot retain authorization claims.")
        for field_name, value in (
            ("reconciled_by.subject", self.reconciled_by.subject),
            ("reconciled_by.tenant", self.reconciled_by.tenant),
        ):
            if value is not None:
                _validate_task_retry_reconciliation_identity(value, field_name)
        expected_types = (
            TaskCancellationReconciliationEventType.CANCELLATION_REQUESTED,
            TaskCancellationReconciliationEventType.STARTED,
            TaskCancellationReconciliationEventType.RECONCILED,
        )
        original_worker_sha256 = _task_retry_reconciliation_identity_sha256(
            self.original_worker_id,
            "original_worker_id",
        )
        cancellation_sha256 = _task_retry_reconciliation_identity_sha256(
            self.cancellation_idempotency_key,
            "cancellation_idempotency_key",
        )
        reconciliation_sha256 = _task_retry_reconciliation_identity_sha256(
            self.reconciliation_idempotency_key,
            "reconciliation_idempotency_key",
        )
        actor_sha256 = _task_retry_reconciliation_actor_sha256(self.reconciled_by)
        for index, (event, event_type) in enumerate(zip(self.events, expected_types, strict=True)):
            requested = index == 0
            if (
                event.type is not event_type
                or event.task_id != self.task_id
                or event.original_worker_id_sha256 != original_worker_sha256
                or event.cancellation_idempotency_key_sha256 != cancellation_sha256
                or event.request_sha256 != (None if requested else self.request_sha256)
                or event.reconciliation_idempotency_key_sha256
                != (None if requested else reconciliation_sha256)
                or event.evidence_sha256 != (None if requested else self.evidence.evidence_sha256)
                or event.actor_sha256 != (None if requested else actor_sha256)
                or event.outcome != (None if requested else self.evidence.outcome)
                or event.occurred_at
                != (
                    self.cancellation_requested_at
                    if requested
                    else (
                        self.reconciliation_requested_at
                        if event_type is TaskCancellationReconciliationEventType.STARTED
                        else self.events[2].occurred_at
                    )
                )
                or event.id
                != _task_cancellation_reconciliation_event_id(
                    event_type=event_type,
                    task_id=self.task_id,
                    cancellation_idempotency_key=self.cancellation_idempotency_key,
                    request_sha256=None if requested else self.request_sha256,
                    reconciliation_idempotency_key=(
                        None if requested else self.reconciliation_idempotency_key
                    ),
                    evidence_sha256=(None if requested else self.evidence.evidence_sha256),
                )
            ):
                raise ValueError("Reconciliation event conflicts with its durable projection.")
        return self


class TaskCancellationReconciliationConflict(TaskTerminalizationConflict):
    """An ordinary reconciliation request lost an identity or settlement race."""

    def __init__(
        self,
        message: str,
        *,
        event: TaskCancellationReconciliationEvent,
    ) -> None:
        super().__init__(message)
        self.event = TaskCancellationReconciliationEvent.model_validate(
            event.model_dump(mode="python")
        )


class TaskCancellationReconciliationRejected(ValueError):
    """Application evidence was non-positive, so the ordinary task remains fenced."""

    def __init__(
        self,
        *,
        outcome: TaskCancellationReconciliationOutcome,
        event: TaskCancellationReconciliationEvent,
    ) -> None:
        super().__init__(f"Task cancellation reconciliation is {outcome.value}.")
        self.outcome = outcome
        self.event = TaskCancellationReconciliationEvent.model_validate(
            event.model_dump(mode="python")
        )


class _TaskCancellationReconciliationRejectionRecord(BaseModel):
    """Durable binding for one non-positive ordinary reconciliation request."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    task_id: str
    reconciliation_idempotency_key: str
    request_sha256: str
    outcome: TaskCancellationReconciliationOutcome
    event: TaskCancellationReconciliationEvent
    recorded_at: datetime

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, value: str) -> str:
        return _validate_task_retry_reconciliation_identity(value, "task_id")

    @field_validator("reconciliation_idempotency_key")
    @classmethod
    def validate_reconciliation_idempotency_key(cls, value: str) -> str:
        return _validate_task_terminalization_idempotency_key(value)

    @field_validator("request_sha256")
    @classmethod
    def validate_request_sha256(cls, value: str) -> str:
        return _validate_lowercase_sha256(value, "request_sha256")

    @field_validator("event", mode="before")
    @classmethod
    def copy_event(cls, value: object) -> object:
        return revalidate_model_input(value, TaskCancellationReconciliationEvent)

    @field_validator("recorded_at")
    @classmethod
    def normalize_recorded_at(cls, value: datetime) -> datetime:
        return normalize_utc_datetime(value, "recorded_at")

    @model_validator(mode="after")
    def validate_rejection(self) -> _TaskCancellationReconciliationRejectionRecord:
        event = self.event
        if (
            self.outcome in _POSITIVE_TASK_CANCELLATION_RECONCILIATION_OUTCOMES
            or event.type is not TaskCancellationReconciliationEventType.REJECTED
            or event.task_id != self.task_id
            or event.request_sha256 != self.request_sha256
            or event.reconciliation_idempotency_key_sha256
            != _task_retry_reconciliation_identity_sha256(
                self.reconciliation_idempotency_key,
                "reconciliation_idempotency_key",
            )
            or event.outcome is not self.outcome
            or event.occurred_at > self.recorded_at
        ):
            raise ValueError("Reconciliation rejection conflicts with its request binding.")
        return self


class TaskCancellationReconciliationResult(BaseModel):
    """Immutable ordinary cancellation receipt plus positive recovery evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    request_sha256: str
    task: Task
    terminalization_receipt: TaskTerminalizationReceipt
    reconciliation: TaskCancellationReconciliation
    committed_at: datetime

    @field_validator("request_sha256")
    @classmethod
    def validate_request_sha256(cls, value: str) -> str:
        return _validate_lowercase_sha256(value, "request_sha256")

    @field_validator("task", mode="before")
    @classmethod
    def copy_task(cls, value: object) -> object:
        return revalidate_model_input(value, Task)

    @field_validator("terminalization_receipt", mode="before")
    @classmethod
    def copy_receipt(cls, value: object) -> TaskTerminalizationReceipt:
        if type(value) is TaskTerminalizationReceipt:
            receipt = value
        elif type(value) is dict:
            payload = dict(value)
            task_payload = payload.get("task")
            if type(task_payload) is not dict:
                raise TypeError("terminalization_receipt.task must be a Task object.")
            payload["task"] = Task.model_validate(task_payload)
            receipt = TaskTerminalizationReceipt.model_validate(payload)
        else:
            raise TypeError(
                "terminalization_receipt must be a TaskTerminalizationReceipt or object."
            )
        return TaskTerminalizationReceipt(
            task_id=receipt.task_id,
            idempotency_key=receipt.idempotency_key,
            worker_id=receipt.worker_id,
            kind=receipt.kind,
            request_sha256=receipt.request_sha256,
            task=receipt.task.model_copy(deep=True),
            committed_at=receipt.committed_at,
        )

    @field_validator("reconciliation", mode="before")
    @classmethod
    def copy_reconciliation(cls, value: object) -> object:
        return revalidate_model_input(value, TaskCancellationReconciliation)

    @field_validator("committed_at")
    @classmethod
    def normalize_committed_at(cls, value: datetime) -> datetime:
        return normalize_utc_datetime(value, "committed_at")

    @model_validator(mode="after")
    def validate_result(self) -> TaskCancellationReconciliationResult:
        receipt = self.terminalization_receipt
        reconciliation = self.reconciliation
        terminalization = TaskTerminalizationRequest(
            task_id=reconciliation.task_id,
            worker_id=reconciliation.original_worker_id,
            lease_expires_at=reconciliation.original_lease_expires_at,
            handoff_id=reconciliation.original_handoff_id,
            kind=TaskTerminalKind.CANCELLED,
            error=self.task.error,
            idempotency_key=reconciliation.cancellation_idempotency_key,
        )
        _, terminalization_sha256 = prepare_task_terminalization(terminalization)
        expected_status_payload = {
            "cancellation_reconciliation": reconciliation.model_dump(
                mode="json",
                warnings=False,
            )
        }
        if (
            self.request_sha256 != reconciliation.request_sha256
            or self.task != receipt.task
            or self.task.id != reconciliation.task_id
            or self.task.status is not TaskStatus.CANCELLED
            or self.task.status_reason is not None
            or self.task.status_payload != expected_status_payload
            or receipt.kind is not TaskTerminalKind.CANCELLED
            or receipt.worker_id != reconciliation.original_worker_id
            or receipt.idempotency_key != reconciliation.cancellation_idempotency_key
            or not _task_terminalization_request_matches_sha256(
                terminalization,
                request_sha256=terminalization_sha256,
                candidate_sha256=receipt.request_sha256,
            )
            or receipt.committed_at != self.committed_at
            or reconciliation.events[2].occurred_at != self.committed_at
        ):
            raise ValueError("Cancellation reconciliation conflicts with its terminal receipt.")
        return self


def prepare_task_retry_cancellation_reconciliation(
    request: TaskRetryCancellationReconciliationRequest,
) -> tuple[TaskRetryCancellationReconciliationRequest, str]:
    """Detach and digest one evidence-bound owner-lost reconciliation request."""

    if type(request) is not TaskRetryCancellationReconciliationRequest:
        raise TypeError(
            "Task retry cancellation reconciliations require a "
            "TaskRetryCancellationReconciliationRequest."
        )
    copied = TaskRetryCancellationReconciliationRequest.model_validate(
        request.model_dump(mode="python", warnings=False)
    )
    request_material = copied.model_dump(
        mode="json",
        warnings=False,
        exclude={"reconciled_by"},
    )
    request_material["reconciled_by"] = resolution_actor_payload(copied.reconciled_by)
    request_sha256 = sha256(
        canonical_durable_json_bytes(
            {
                "schema": "cayu.task-retry-cancellation-reconciliation.v1",
                **request_material,
            },
            "task_retry_cancellation_reconciliation",
        )
    ).hexdigest()
    return copied, request_sha256


def prepare_task_cancellation_reconciliation(
    request: TaskCancellationReconciliationRequest,
) -> tuple[TaskCancellationReconciliationRequest, str]:
    """Detach and digest one evidence-bound ordinary reconciliation request."""

    if type(request) is not TaskCancellationReconciliationRequest:
        raise TypeError(
            "Task cancellation reconciliations require a TaskCancellationReconciliationRequest."
        )
    copied = TaskCancellationReconciliationRequest.model_validate(
        request.model_dump(mode="python", warnings=False)
    )
    request_material = copied.model_dump(
        mode="json",
        warnings=False,
        exclude={"reconciled_by"},
    )
    request_material["reconciled_by"] = resolution_actor_payload(copied.reconciled_by)
    request_sha256 = sha256(
        canonical_durable_json_bytes(
            {
                "schema": (
                    "cayu.task-cancellation-reconciliation.v1"
                    if copied.original_handoff_id is None
                    else "cayu.task-cancellation-reconciliation.v2"
                ),
                **request_material,
            },
            "task_cancellation_reconciliation",
        )
    ).hexdigest()
    return copied, request_sha256


def _copy_task_cancellation_reconciliation_result(
    result: TaskCancellationReconciliationResult,
) -> TaskCancellationReconciliationResult:
    if type(result) is not TaskCancellationReconciliationResult:
        raise TypeError("result must be a TaskCancellationReconciliationResult.")
    return result.model_copy(deep=True)


def _replay_task_cancellation_reconciliation(
    *,
    request: TaskCancellationReconciliationRequest,
    request_sha256: str,
    receipt: TaskTerminalizationReceipt,
    current_task: Task | None,
) -> TaskCancellationReconciliationResult:
    if type(receipt) is not TaskTerminalizationReceipt:
        raise TypeError("receipt must be a TaskTerminalizationReceipt.")
    receipt = receipt.model_copy(deep=True)
    current_task = None if current_task is None else copy_task(current_task)
    status_payload = receipt.task.status_payload
    reconciliation_payload = (
        None
        if type(status_payload) is not dict
        else status_payload.get("cancellation_reconciliation")
    )
    if type(reconciliation_payload) is not dict:
        raise _task_cancellation_reconciliation_conflict(
            request,
            "Task cancellation was terminalized without reconciliation evidence.",
        )
    reconciliation = TaskCancellationReconciliation.model_validate(reconciliation_payload)
    if (
        receipt.idempotency_key != request.cancellation_idempotency_key
        or reconciliation.request_sha256 != request_sha256
        or reconciliation.reconciliation_idempotency_key != request.reconciliation_idempotency_key
    ):
        raise _task_cancellation_reconciliation_conflict(
            request,
            "Task cancellation idempotency key is bound to another intent.",
        )
    if current_task != receipt.task:
        raise _task_cancellation_reconciliation_conflict(
            request,
            "Task cancellation receipt conflicts with the current terminal task.",
        )
    return TaskCancellationReconciliationResult(
        request_sha256=request_sha256,
        task=receipt.task,
        terminalization_receipt=receipt,
        reconciliation=reconciliation,
        committed_at=receipt.committed_at,
    )


def _task_retry_reconciliation_identity_sha256(value: str, field_name: str) -> str:
    value = _validate_task_retry_reconciliation_identity(value, field_name)
    return sha256(value.encode("utf-8")).hexdigest()


def _task_retry_reconciliation_actor_sha256(actor: ResolutionActor) -> str:
    projected = resolution_actor_payload(actor)
    if projected is None:  # pragma: no cover - required request invariant
        raise AssertionError("Task retry reconciliation requires actor provenance.")
    return sha256(
        canonical_durable_json_bytes(projected, "task_retry_reconciliation_actor")
    ).hexdigest()


def _task_retry_cancellation_reconciliation_event_id(
    *,
    event_type: TaskRetryCancellationReconciliationEventType,
    task_id: str,
    series_id: str,
    attempt: int,
    cancellation_idempotency_key: str,
    request_sha256: str | None,
    reconciliation_idempotency_key: str | None,
    evidence_sha256: str | None,
) -> str:
    material = canonical_durable_json_bytes(
        {
            "schema": "cayu.task-retry-cancellation-reconciliation-event.v1",
            "type": event_type.value,
            "task_id": task_id,
            "series_id": series_id,
            "attempt": attempt,
            "cancellation_idempotency_key": cancellation_idempotency_key,
            "request_sha256": request_sha256,
            "reconciliation_idempotency_key": reconciliation_idempotency_key,
            "evidence_sha256": evidence_sha256,
        },
        "task_retry_cancellation_reconciliation_event_id",
    )
    return f"task-retry-reconciliation-event:v1:{sha256(material).hexdigest()}"


def _task_retry_cancellation_reconciliation_event(
    request: TaskRetryCancellationReconciliationRequest,
    *,
    event_type: TaskRetryCancellationReconciliationEventType,
    occurred_at: datetime,
) -> TaskRetryCancellationReconciliationEvent:
    if event_type is TaskRetryCancellationReconciliationEventType.CANCELLATION_REQUESTED:
        raise ValueError("Reconciliation requests cannot fabricate cancellation events.")
    _, request_sha256 = prepare_task_retry_cancellation_reconciliation(request)
    return TaskRetryCancellationReconciliationEvent(
        id=_task_retry_cancellation_reconciliation_event_id(
            event_type=event_type,
            task_id=request.task_id,
            series_id=request.series_id,
            attempt=request.attempt,
            cancellation_idempotency_key=request.cancellation_idempotency_key,
            request_sha256=request_sha256,
            reconciliation_idempotency_key=request.reconciliation_idempotency_key,
            evidence_sha256=request.evidence.evidence_sha256,
        ),
        type=event_type,
        task_id=request.task_id,
        series_id=request.series_id,
        attempt=request.attempt,
        causal_budget_id=request.causal_budget_id,
        original_worker_id_sha256=_task_retry_reconciliation_identity_sha256(
            request.original_worker_id,
            "original_worker_id",
        ),
        cancellation_idempotency_key_sha256=(
            _task_retry_reconciliation_identity_sha256(
                request.cancellation_idempotency_key,
                "cancellation_idempotency_key",
            )
        ),
        request_sha256=request_sha256,
        reconciliation_idempotency_key_sha256=(
            _task_retry_reconciliation_identity_sha256(
                request.reconciliation_idempotency_key,
                "reconciliation_idempotency_key",
            )
        ),
        evidence_sha256=request.evidence.evidence_sha256,
        actor_sha256=_task_retry_reconciliation_actor_sha256(request.reconciled_by),
        outcome=request.evidence.outcome,
        occurred_at=occurred_at,
    )


def _task_retry_cancellation_reconciliation_conflict(
    request: TaskRetryCancellationReconciliationRequest,
    message: str,
) -> TaskRetryCancellationReconciliationConflict:
    return TaskRetryCancellationReconciliationConflict(
        message,
        event=_task_retry_cancellation_reconciliation_event(
            request,
            event_type=TaskRetryCancellationReconciliationEventType.CONFLICT,
            occurred_at=request.reconciliation_requested_at,
        ),
    )


def _task_retry_cancellation_reconciliation_rejection_record(
    request: TaskRetryCancellationReconciliationRequest,
    *,
    request_sha256: str,
    recorded_at: datetime,
) -> _TaskRetryCancellationReconciliationRejectionRecord | None:
    """Build the durable idempotency binding for a non-positive outcome."""

    if request.evidence.outcome in (_POSITIVE_TASK_RETRY_CANCELLATION_RECONCILIATION_OUTCOMES):
        return None
    return _TaskRetryCancellationReconciliationRejectionRecord(
        task_id=request.task_id,
        reconciliation_idempotency_key=request.reconciliation_idempotency_key,
        request_sha256=request_sha256,
        outcome=request.evidence.outcome,
        event=_task_retry_cancellation_reconciliation_event(
            request,
            event_type=TaskRetryCancellationReconciliationEventType.REJECTED,
            occurred_at=request.reconciliation_requested_at,
        ),
        recorded_at=recorded_at,
    )


def _rejected_task_retry_cancellation_reconciliation(
    record: _TaskRetryCancellationReconciliationRejectionRecord,
) -> TaskRetryCancellationReconciliationRejected:
    return TaskRetryCancellationReconciliationRejected(
        outcome=record.outcome,
        event=record.event,
    )


def _replay_task_retry_cancellation_reconciliation_rejection(
    request: TaskRetryCancellationReconciliationRequest,
    *,
    request_sha256: str,
    record: _TaskRetryCancellationReconciliationRejectionRecord,
) -> TaskRetryCancellationReconciliationRejected:
    """Replay one exact rejection or fail a changed idempotency-key intent."""

    durable = _TaskRetryCancellationReconciliationRejectionRecord.model_validate(
        record.model_dump(mode="python")
    )
    expected = _task_retry_cancellation_reconciliation_rejection_record(
        request,
        request_sha256=request_sha256,
        recorded_at=durable.recorded_at,
    )
    if expected is None or expected != durable:
        raise _task_retry_cancellation_reconciliation_conflict(
            request,
            "Reconciliation idempotency key is already bound to another request.",
        )
    return _rejected_task_retry_cancellation_reconciliation(durable)


def _task_cancellation_reconciliation_event_id(
    *,
    event_type: TaskCancellationReconciliationEventType,
    task_id: str,
    cancellation_idempotency_key: str,
    request_sha256: str | None,
    reconciliation_idempotency_key: str | None,
    evidence_sha256: str | None,
) -> str:
    material = canonical_durable_json_bytes(
        {
            "schema": "cayu.task-cancellation-reconciliation-event.v1",
            "type": event_type.value,
            "task_id": task_id,
            "cancellation_idempotency_key": cancellation_idempotency_key,
            "request_sha256": request_sha256,
            "reconciliation_idempotency_key": reconciliation_idempotency_key,
            "evidence_sha256": evidence_sha256,
        },
        "task_cancellation_reconciliation_event_id",
    )
    return f"task-reconciliation-event:v1:{sha256(material).hexdigest()}"


def _task_cancellation_requested_event(
    task: Task,
    *,
    cancellation_idempotency_key: str,
    occurred_at: datetime,
) -> TaskCancellationReconciliationEvent:
    if task.retry_series is not None or task.worker_id is None:
        raise TaskTerminalizationConflict(
            "Task cancellation request lacks ordinary active-owner identity."
        )
    return TaskCancellationReconciliationEvent(
        id=_task_cancellation_reconciliation_event_id(
            event_type=TaskCancellationReconciliationEventType.CANCELLATION_REQUESTED,
            task_id=task.id,
            cancellation_idempotency_key=cancellation_idempotency_key,
            request_sha256=None,
            reconciliation_idempotency_key=None,
            evidence_sha256=None,
        ),
        type=TaskCancellationReconciliationEventType.CANCELLATION_REQUESTED,
        task_id=task.id,
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


def _task_cancellation_reconciliation_event(
    request: TaskCancellationReconciliationRequest,
    *,
    event_type: TaskCancellationReconciliationEventType,
    occurred_at: datetime,
) -> TaskCancellationReconciliationEvent:
    if event_type is TaskCancellationReconciliationEventType.CANCELLATION_REQUESTED:
        raise ValueError("Reconciliation requests cannot fabricate cancellation events.")
    _, request_sha256 = prepare_task_cancellation_reconciliation(request)
    return TaskCancellationReconciliationEvent(
        id=_task_cancellation_reconciliation_event_id(
            event_type=event_type,
            task_id=request.task_id,
            cancellation_idempotency_key=request.cancellation_idempotency_key,
            request_sha256=request_sha256,
            reconciliation_idempotency_key=request.reconciliation_idempotency_key,
            evidence_sha256=request.evidence.evidence_sha256,
        ),
        type=event_type,
        task_id=request.task_id,
        original_worker_id_sha256=_task_retry_reconciliation_identity_sha256(
            request.original_worker_id,
            "original_worker_id",
        ),
        cancellation_idempotency_key_sha256=(
            _task_retry_reconciliation_identity_sha256(
                request.cancellation_idempotency_key,
                "cancellation_idempotency_key",
            )
        ),
        request_sha256=request_sha256,
        reconciliation_idempotency_key_sha256=(
            _task_retry_reconciliation_identity_sha256(
                request.reconciliation_idempotency_key,
                "reconciliation_idempotency_key",
            )
        ),
        evidence_sha256=request.evidence.evidence_sha256,
        actor_sha256=_task_retry_reconciliation_actor_sha256(request.reconciled_by),
        outcome=request.evidence.outcome,
        occurred_at=occurred_at,
    )


def _task_cancellation_reconciliation_conflict(
    request: TaskCancellationReconciliationRequest,
    message: str,
) -> TaskCancellationReconciliationConflict:
    return TaskCancellationReconciliationConflict(
        message,
        event=_task_cancellation_reconciliation_event(
            request,
            event_type=TaskCancellationReconciliationEventType.CONFLICT,
            occurred_at=request.reconciliation_requested_at,
        ),
    )


def _task_cancellation_reconciliation_rejection_record(
    request: TaskCancellationReconciliationRequest,
    *,
    request_sha256: str,
    recorded_at: datetime,
) -> _TaskCancellationReconciliationRejectionRecord | None:
    """Build the durable idempotency binding for one non-positive outcome."""

    if request.evidence.outcome in _POSITIVE_TASK_CANCELLATION_RECONCILIATION_OUTCOMES:
        return None
    return _TaskCancellationReconciliationRejectionRecord(
        task_id=request.task_id,
        reconciliation_idempotency_key=request.reconciliation_idempotency_key,
        request_sha256=request_sha256,
        outcome=request.evidence.outcome,
        event=_task_cancellation_reconciliation_event(
            request,
            event_type=TaskCancellationReconciliationEventType.REJECTED,
            occurred_at=request.reconciliation_requested_at,
        ),
        recorded_at=recorded_at,
    )


def _rejected_task_cancellation_reconciliation(
    record: _TaskCancellationReconciliationRejectionRecord,
) -> TaskCancellationReconciliationRejected:
    return TaskCancellationReconciliationRejected(
        outcome=record.outcome,
        event=record.event,
    )


def _replay_task_cancellation_reconciliation_rejection(
    request: TaskCancellationReconciliationRequest,
    *,
    request_sha256: str,
    record: _TaskCancellationReconciliationRejectionRecord,
) -> TaskCancellationReconciliationRejected:
    """Replay one exact rejection or fail a changed idempotency-key intent."""

    durable = _TaskCancellationReconciliationRejectionRecord.model_validate(
        record.model_dump(mode="python")
    )
    expected = _task_cancellation_reconciliation_rejection_record(
        request,
        request_sha256=request_sha256,
        recorded_at=durable.recorded_at,
    )
    if expected is None or expected != durable:
        raise _task_cancellation_reconciliation_conflict(
            request,
            "Reconciliation idempotency key is already bound to another request.",
        )
    return _rejected_task_cancellation_reconciliation(durable)


def _task_retry_cancellation_requested(task: Task) -> bool:
    """Return whether an owned retry attempt carries a durable cancel request."""

    return task.status_reason == _TASK_RETRY_CANCELLATION_REQUESTED_REASON


def _task_cancellation_requested(task: Task) -> bool:
    """Return whether an ordinary live task carries a durable cancel request."""

    return task.status_reason == _TASK_CANCELLATION_REQUESTED_REASON
