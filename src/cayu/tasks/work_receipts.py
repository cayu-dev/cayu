"""Durable receipts for verified work decisions and execution lifecycle settlement."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, StrictBool, field_validator, model_validator

from cayu._clock import normalize_utc_datetime
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu._validation import revalidate_model_input
from cayu.runtime.work_attempt_lifecycle import (
    WorkAttemptLifecycleSettlement,
    WorkAttemptPreparationHold,
    work_attempt_lifecycle_settlement_sha256,
    work_attempt_preparation_hold_sha256,
)
from cayu.tasks.contracts import (
    WORK_COMPLETION_APPLICATION_RECEIPT_MAX_BYTES,
    WORK_COMPLETION_APPLICATION_RECEIPT_MAX_ITEMS,
    require_bounded_work_completion_document,
    validate_work_completion_idempotency_key,
)
from cayu.tasks.records import (
    Task,
    TaskStatus,
    _preflight_bounded_task_payloads,
)


class CompletionDecisionApplicationReceipt(BaseModel):
    """Immutable evidence that one verifier decision was applied to its task."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    task_id: str
    decision_id: str
    verifier_profile_fingerprint: str
    idempotency_key: str
    request_sha256: str
    task: Task
    applied_at: datetime

    @field_validator("task_id", "decision_id", "idempotency_key")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        if info.field_name == "idempotency_key":
            return validate_work_completion_idempotency_key(value)
        return require_clean_nonblank(value, info.field_name)

    @field_validator("verifier_profile_fingerprint", "request_sha256")
    @classmethod
    def validate_sha256(cls, value: str, info) -> str:
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"{info.field_name} must be a lowercase SHA-256 digest.")
        return value

    @field_validator("task", mode="before")
    @classmethod
    def copy_task(cls, value: object) -> object:
        if type(value) is Task:
            _preflight_bounded_task_payloads(
                value,
                field_label="Decision-application receipt task",
            )
        return revalidate_model_input(value, Task)

    @field_validator("applied_at")
    @classmethod
    def normalize_applied_at(cls, value: datetime) -> datetime:
        return normalize_utc_datetime(value, "applied_at")

    @model_validator(mode="after")
    def validate_receipt_task(self) -> CompletionDecisionApplicationReceipt:
        if self.task.id != self.task_id:
            raise ValueError("Decision-application receipt conflicts with its task.")
        if self.task.work_contract is None:
            raise ValueError("Decision-application receipt requires a contract-bound task.")
        require_bounded_work_completion_document(
            self.model_dump(mode="json", warnings=False),
            "Completion decision application receipt",
            max_bytes=WORK_COMPLETION_APPLICATION_RECEIPT_MAX_BYTES,
            max_items=WORK_COMPLETION_APPLICATION_RECEIPT_MAX_ITEMS,
        )
        return self


class WorkAttemptPreparationHoldReceipt(BaseModel):
    """Original non-success result for one exact pre-admission callback failure."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    request: WorkAttemptPreparationHold
    request_sha256: str
    task: Task

    @field_validator("request", mode="before")
    @classmethod
    def copy_request(cls, value: object) -> object:
        return revalidate_model_input(value, WorkAttemptPreparationHold)

    @field_validator("task", mode="before")
    @classmethod
    def copy_result_task(cls, value: object) -> object:
        if type(value) is Task:
            _preflight_bounded_task_payloads(value, field_label="Preparation hold receipt task")
        return revalidate_model_input(value, Task)

    @model_validator(mode="after")
    def validate_authority(self) -> WorkAttemptPreparationHoldReceipt:
        if (
            self.request_sha256 != work_attempt_preparation_hold_sha256(self.request)
            or self.task.id != self.request.task_id
            or self.task.work_contract != self.request.contract
            or self.task.status
            is not (
                TaskStatus.CANCELLED
                if self.request.reason == "work_contract_group_cancelled"
                else TaskStatus.NEEDS_ATTENTION
            )
            or self.task.status_reason != self.request.reason
            or self.task.worker_id is not None
            or self.task.lease_expires_at is not None
            or self.task.session_id is not None
            or self.task.session_instance_id is not None
        ):
            raise ValueError("Preparation hold receipt conflicts with its exact non-success task.")
        require_bounded_work_completion_document(
            self.model_dump(mode="json", warnings=False),
            "Work-attempt preparation hold receipt",
            max_bytes=WORK_COMPLETION_APPLICATION_RECEIPT_MAX_BYTES + 32 * 1024,
            max_items=WORK_COMPLETION_APPLICATION_RECEIPT_MAX_ITEMS + 256,
        )
        return self


class WorkAttemptLifecycleReceipt(BaseModel):
    """Original final task result and exact invocation-release evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    request: WorkAttemptLifecycleSettlement
    request_sha256: str
    task: Task
    retired_contract_binding: StrictBool
    settled_at: datetime

    @field_validator("request", mode="before")
    @classmethod
    def copy_request(cls, value: object) -> object:
        return revalidate_model_input(value, WorkAttemptLifecycleSettlement)

    @field_validator("task", mode="before")
    @classmethod
    def copy_result_task(cls, value: object) -> object:
        if type(value) is Task:
            _preflight_bounded_task_payloads(
                value, field_label="Work-attempt lifecycle receipt task"
            )
        return revalidate_model_input(value, Task)

    @field_validator("settled_at")
    @classmethod
    def normalize_settled_at(cls, value: datetime) -> datetime:
        return normalize_utc_datetime(value, "settled_at")

    @model_validator(mode="after")
    def validate_authority(self) -> WorkAttemptLifecycleReceipt:
        if self.request_sha256 != work_attempt_lifecycle_settlement_sha256(self.request):
            raise ValueError("Work-attempt receipt conflicts with its exact settlement request.")
        if self.task.id != self.request.task_id or self.task.work_contract is None:
            raise ValueError("Work-attempt receipt requires its exact contract-bound task.")
        if self.task.session_id != self.request.release_evidence.session_id or (
            self.task.session_instance_id != self.request.release_evidence.session_instance_id
        ):
            raise ValueError("Work-attempt receipt conflicts with its released invocation.")
        if self.retired_contract_binding != (
            self.task.status is TaskStatus.COMPLETED
            or (
                self.request.kind == "group_cancellation"
                and self.task.status is TaskStatus.CANCELLED
            )
        ):
            raise ValueError(
                "Only accepted or quiescent group-cancelled work retires its contract binding."
            )
        if self.request.kind in {
            "runtime_stop",
            "proposal_deadline_stop",
            "continuation_deadline_stop",
        } and (
            self.task.status is not TaskStatus.NEEDS_ATTENTION
            or self.task.status_reason != self.request.stop_reason
        ):
            raise ValueError("Runtime-stop receipt requires its typed non-success result.")
        if self.request.kind == "group_cancellation" and (
            self.task.status is not TaskStatus.CANCELLED
            or self.task.status_reason != self.request.stop_reason
        ):
            raise ValueError("Group-stop receipt requires its typed cancellation result.")
        require_bounded_work_completion_document(
            self.model_dump(mode="json", warnings=False),
            "Work-attempt lifecycle receipt",
            max_bytes=WORK_COMPLETION_APPLICATION_RECEIPT_MAX_BYTES + 32 * 1024,
            max_items=WORK_COMPLETION_APPLICATION_RECEIPT_MAX_ITEMS + 256,
        )
        return self
