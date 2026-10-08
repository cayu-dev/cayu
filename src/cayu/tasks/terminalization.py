"""Task terminalization requests, receipts and replay validation."""

from __future__ import annotations

import math
from datetime import datetime
from enum import StrEnum
from hashlib import sha256
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._clock import normalize_utc_datetime
from cayu._validation import canonical_durable_json_bytes, copy_durable_json_object
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.tasks.records import (
    Task,
    TaskStatus,
)


class TaskTerminalizationConflict(ValueError):
    """An idempotency key is already bound to another terminalization intent."""


class TaskTerminalKind(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TASK_TERMINALIZATION_IDEMPOTENCY_KEY_MAX_BYTES = 256


class TaskTerminalizationRequest(BaseModel):
    """One claim-fenced, replay-safe completion, failure, or cancellation intent."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    task_id: str
    worker_id: str
    # Exact lease generation returned by claim/heartbeat. A worker name is
    # reusable and therefore is not sufficient terminalization authority.
    lease_expires_at: datetime | None = None
    # Exact interrupted-continuation generation. ``None`` is the authority for
    # ordinary task claims and direct attachments; recovered continuations must
    # present the non-null generation returned by their claim.
    handoff_id: str | None = None
    kind: TaskTerminalKind
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    idempotency_key: str

    @field_validator("task_id", "worker_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("handoff_id")
    @classmethod
    def validate_handoff_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, "handoff_id")

    @field_validator("lease_expires_at")
    @classmethod
    def normalize_lease_expires_at(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return normalize_utc_datetime(value, "lease_expires_at")

    @field_validator("idempotency_key")
    @classmethod
    def validate_idempotency_key(cls, value: str) -> str:
        return _validate_task_terminalization_idempotency_key(value)

    @field_validator("result", "error", mode="before")
    @classmethod
    def copy_payload(
        cls,
        value: dict[str, Any] | None,
        info,
    ) -> dict[str, Any] | None:
        if value is None:
            return None
        return copy_durable_json_object(value, info.field_name)

    @model_validator(mode="after")
    def validate_terminal_payload(self) -> TaskTerminalizationRequest:
        if self.kind is TaskTerminalKind.COMPLETED:
            if self.result is None or self.error is not None:
                raise ValueError("Completed terminalization requires result and forbids error.")
        elif self.error is None or self.result is not None:
            raise ValueError(
                "Failed or cancelled terminalization requires error and forbids result."
            )
        return self


class TaskTerminalizationReceipt(BaseModel):
    """Immutable commit evidence for one task terminalization intent."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    task_id: str
    idempotency_key: str
    worker_id: str
    kind: TaskTerminalKind
    request_sha256: str
    task: Task
    committed_at: datetime

    @field_validator("task_id", "worker_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("idempotency_key")
    @classmethod
    def validate_idempotency_key(cls, value: str) -> str:
        return _validate_task_terminalization_idempotency_key(value)

    @field_validator("request_sha256")
    @classmethod
    def validate_request_sha256(cls, value: str) -> str:
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("request_sha256 must be a lowercase SHA-256 digest.")
        return value

    @field_validator("task", mode="before")
    @classmethod
    def copy_terminal_task(cls, value: Task) -> Task:
        if type(value) is not Task:
            raise TypeError("task must be a Task instance.")
        return Task.model_validate(value.model_dump(mode="python"))

    @field_validator("committed_at")
    @classmethod
    def normalize_committed_at(cls, value: datetime) -> datetime:
        return normalize_utc_datetime(value, "committed_at")

    @model_validator(mode="after")
    def validate_receipt_task(self) -> TaskTerminalizationReceipt:
        expected_status = TaskStatus(self.kind.value)
        if self.task.id != self.task_id or self.task.status is not expected_status:
            raise ValueError("Terminalization receipt conflicts with its terminal task.")
        if self.task.worker_id is not None or self.task.lease_expires_at is not None:
            raise ValueError("Terminalization receipt task retains live claim ownership.")
        return self


class TaskTerminalizationRetryPolicy(BaseModel):
    """Finite retry and backoff bounds for acknowledgement-ambiguous writes."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    max_attempts: StrictInt = Field(default=3, ge=1, le=10)
    attempt_timeout_seconds: StrictFloat = Field(default=30.0, gt=0, le=300)
    initial_backoff_seconds: StrictFloat = Field(default=0.05, ge=0, le=60)
    backoff_multiplier: StrictFloat = Field(default=2.0, ge=1, le=10)
    max_backoff_seconds: StrictFloat = Field(default=1.0, ge=0, le=60)


class TaskTerminalizationRetryResult(BaseModel):
    """Detached terminal task plus observable retry/reconciliation evidence."""

    model_config = ConfigDict(
        extra="forbid",
        hide_input_in_errors=True,
        allow_inf_nan=False,
    )

    task: Task
    attempt_count: StrictInt = Field(ge=1, le=10)
    receipt_reconciled: StrictBool
    elapsed_seconds: StrictFloat = Field(default=0.0, ge=0)
    applied_backoff_seconds: StrictFloat = Field(default=0.0, ge=0)

    @field_validator("task", mode="before")
    @classmethod
    def copy_terminal_task(cls, value: Task) -> Task:
        if type(value) is not Task:
            raise TypeError("task must be a Task instance.")
        return Task.model_validate(value.model_dump(mode="python"))


class TaskTerminalizationUncertain(RuntimeError):
    """Bounded evidence that no exact terminalization receipt was observed."""

    def __init__(
        self,
        *,
        task_id: str,
        idempotency_key: str,
        attempt_count: int,
        error_category: str,
        elapsed_seconds: float = 0.0,
        applied_backoff_seconds: float = 0.0,
    ) -> None:
        for field_name, value in (
            ("elapsed_seconds", elapsed_seconds),
            ("applied_backoff_seconds", applied_backoff_seconds),
        ):
            if type(value) is not float:
                raise TypeError(f"{field_name} must be a float.")
            if value < 0 or not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite and non-negative.")
        self.task_id = _bounded_task_terminalization_evidence(task_id)
        self.idempotency_key = _bounded_task_terminalization_evidence(idempotency_key)
        self.attempt_count = attempt_count
        self.error_category = error_category
        self.elapsed_seconds = elapsed_seconds
        self.applied_backoff_seconds = applied_backoff_seconds
        super().__init__(
            "Task terminalization outcome is uncertain for "
            f"task {self.task_id} after {attempt_count} attempts "
            f"(category={error_category})."
        )


def prepare_task_terminalization(
    request: TaskTerminalizationRequest,
) -> tuple[TaskTerminalizationRequest, str]:
    """Detach and deterministically digest one validated logical request."""

    if type(request) is not TaskTerminalizationRequest:
        raise TypeError(
            "Task terminalization requests must be TaskTerminalizationRequest instances."
        )
    copied = TaskTerminalizationRequest.model_validate(request.model_dump(mode="python"))
    material = {
        "schema": "cayu.task-terminalization.v3",
        "task_id": copied.task_id,
        "idempotency_key": copied.idempotency_key,
        "worker_id": copied.worker_id,
        "lease_expires_at": (
            None if copied.lease_expires_at is None else copied.lease_expires_at.isoformat()
        ),
        "kind": copied.kind.value,
        "result": copied.result,
        "error": copied.error,
    }
    if copied.handoff_id is not None:
        material["handoff_id"] = copied.handoff_id
    request_sha256 = sha256(
        canonical_durable_json_bytes(material, "task_terminalization")
    ).hexdigest()
    return copied, request_sha256


def _legacy_task_terminalization_request_sha256(
    request: TaskTerminalizationRequest,
) -> str:
    """Reconstruct the digest emitted before lease-generation fencing."""

    copied = TaskTerminalizationRequest.model_validate(request.model_dump(mode="python"))
    material = {
        "schema": (
            "cayu.task-terminalization.v1"
            if copied.handoff_id is None
            else "cayu.task-terminalization.v2"
        ),
        "task_id": copied.task_id,
        "idempotency_key": copied.idempotency_key,
        "worker_id": copied.worker_id,
        "kind": copied.kind.value,
        "result": copied.result,
        "error": copied.error,
    }
    if copied.handoff_id is not None:
        material["handoff_id"] = copied.handoff_id
    return sha256(canonical_durable_json_bytes(material, "task_terminalization")).hexdigest()


def _task_terminalization_request_matches_sha256(
    request: TaskTerminalizationRequest,
    *,
    request_sha256: str,
    candidate_sha256: str,
) -> bool:
    return candidate_sha256 in {
        request_sha256,
        _legacy_task_terminalization_request_sha256(request),
    }


def prepare_task_terminalization_receipt_lookup(
    task_id: str,
    idempotency_key: str,
) -> tuple[str, str]:
    return (
        require_clean_nonblank(task_id, "task_id"),
        _validate_task_terminalization_idempotency_key(idempotency_key),
    )


def _validate_task_terminalization_idempotency_key(value: str) -> str:
    value = require_clean_nonblank(value, "idempotency_key")
    if len(value.encode("utf-8")) > TASK_TERMINALIZATION_IDEMPOTENCY_KEY_MAX_BYTES:
        raise ValueError(
            "idempotency_key must be at most "
            f"{TASK_TERMINALIZATION_IDEMPOTENCY_KEY_MAX_BYTES} UTF-8 bytes."
        )
    return value


def _replay_task_terminalization_receipt(
    *,
    request: TaskTerminalizationRequest,
    request_sha256: str,
    receipt: TaskTerminalizationReceipt,
    current_task: Task | None,
) -> Task:
    """Validate durable replay proof and return a detached terminal task."""

    if not _task_terminalization_request_matches_sha256(
        request,
        request_sha256=request_sha256,
        candidate_sha256=receipt.request_sha256,
    ):
        raise TaskTerminalizationConflict(
            "Task terminalization idempotency key conflicts with another intent."
        )
    if current_task != receipt.task:
        raise TaskTerminalizationConflict(
            "Task terminalization receipt conflicts with the current terminal task."
        )
    return receipt.task.model_copy(deep=True)


def _bounded_task_terminalization_evidence(value: str) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= TASK_TERMINALIZATION_IDEMPOTENCY_KEY_MAX_BYTES:
        return value
    suffix = f"...[sha256:{sha256(encoded).hexdigest()[:8]}]"
    prefix_bytes = TASK_TERMINALIZATION_IDEMPOTENCY_KEY_MAX_BYTES - len(suffix.encode("utf-8"))
    prefix = encoded[:prefix_bytes].decode("utf-8", "ignore")
    return f"{prefix}{suffix}"
