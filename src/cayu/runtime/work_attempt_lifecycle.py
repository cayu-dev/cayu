"""Exact settlement requests for the verified-task lifecycle owner."""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from cayu._clock import normalize_utc_datetime
from cayu._validation import canonical_durable_json_bytes, revalidate_model_input
from cayu.runtime.invocation_release import InvocationReleaseEvidence
from cayu.runtime.work_attempt_admission import WorkAttemptAdmission
from cayu.runtime.work_contracts import (
    WorkContractRef,
    require_bounded_work_completion_document,
    validate_work_completion_idempotency_key,
    validate_work_completion_linked_id,
)

WorkAttemptStopReason = Literal[
    "work_contract_elapsed_limit",
    "work_contract_budget_limit",
    "work_contract_handler_failed",
    "work_contract_execution_failed",
    "work_contract_execution_interrupted",
]


def runtime_stop_reason_for_execution_stop(
    admission: WorkAttemptAdmission,
) -> WorkAttemptStopReason | None:
    """Project a validated admission's immutable runtime limit decision."""
    stop = admission.execution_stop
    if stop is None:
        return None
    if stop.request.reason == "budget_limit":
        return "work_contract_budget_limit"
    if stop.request.reason == "elapsed_limit":
        return "work_contract_elapsed_limit"
    return None


class WorkAttemptPreparationHold(BaseModel):
    """Exact pre-admission failure, after the read-only callback has settled.

    This request carries no exception text or caller-supplied diagnostic payload.
    The worker must retain its claim while a cancelled callback is still running.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    hold_id: str
    task_id: str
    contract: WorkContractRef
    worker_id: str
    lease_expires_at: datetime
    reason: Literal[
        "work_contract_preparation_failed",
        "work_contract_preparation_timed_out",
        "work_contract_elapsed_limit",
    ]
    deadline_expires_at: datetime | None = None

    @field_validator("deadline_expires_at")
    @classmethod
    def normalize_deadline(cls, value: datetime | None) -> datetime | None:
        return None if value is None else normalize_utc_datetime(value, "deadline_expires_at")

    @model_validator(mode="after")
    def validate_deadline(self) -> WorkAttemptPreparationHold:
        if (self.reason == "work_contract_elapsed_limit") != (self.deadline_expires_at is not None):
            raise ValueError("Only elapsed preparation holds require an exact deadline.")
        return self

    @field_validator("hold_id")
    @classmethod
    def validate_key(cls, value: str) -> str:
        return validate_work_completion_idempotency_key(value)

    @field_validator("task_id", "worker_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return validate_work_completion_linked_id(value, info.field_name)

    @field_validator("contract", mode="before")
    @classmethod
    def copy_contract(cls, value: object) -> object:
        return revalidate_model_input(value, WorkContractRef)

    @field_validator("lease_expires_at")
    @classmethod
    def normalize_lease(cls, value: datetime) -> datetime:
        return normalize_utc_datetime(value, "lease_expires_at")


def copy_work_attempt_preparation_hold(
    value: WorkAttemptPreparationHold,
) -> WorkAttemptPreparationHold:
    if type(value) is not WorkAttemptPreparationHold:
        raise TypeError("Preparation hold requires WorkAttemptPreparationHold.")
    return cast(
        "WorkAttemptPreparationHold", revalidate_model_input(value, WorkAttemptPreparationHold)
    )


def work_attempt_preparation_hold_sha256(value: WorkAttemptPreparationHold) -> str:
    copied = copy_work_attempt_preparation_hold(value)
    return sha256(
        canonical_durable_json_bytes(
            copied.model_dump(mode="json"), "work_attempt_preparation_hold"
        )
    ).hexdigest()


class WorkAttemptLifecycleSettlement(BaseModel):
    """Bind final task settlement to exact admission and invocation readback.

    Only the runtime lifecycle owner produces release evidence. This store
    request is not a public replacement for that owner's cleanup protocol.
    The expected admission digest binds its entire canonical authority tuple,
    including execution generation, source settings and predecessor decision.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    settlement_id: str
    task_id: str
    admission_id: str
    expected_admission_sha256: str
    release_evidence: InvocationReleaseEvidence
    kind: Literal[
        "decision_application",
        "runtime_stop",
        "proposal_deadline_stop",
        "continuation_deadline_stop",
    ]
    proposal_id: str | None = None
    proposal_request_sha256: str | None = None
    decision_id: str | None = None
    application_idempotency_key: str | None = None
    stop_reason: WorkAttemptStopReason | None = None

    @field_validator("settlement_id", "application_idempotency_key")
    @classmethod
    def validate_key(cls, value: str | None) -> str | None:
        return None if value is None else validate_work_completion_idempotency_key(value)

    @field_validator("task_id", "admission_id", "decision_id", "proposal_id")
    @classmethod
    def validate_identity(cls, value: str | None, info) -> str | None:
        return None if value is None else validate_work_completion_linked_id(value, info.field_name)

    @field_validator("expected_admission_sha256")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("Expected admission identity must be lowercase SHA-256.")
        return value

    @field_validator("proposal_request_sha256")
    @classmethod
    def validate_proposal_digest(cls, value: str | None) -> str | None:
        return None if value is None else cls.validate_digest(value)

    @field_validator("release_evidence", mode="before")
    @classmethod
    def copy_release_evidence(cls, value: object) -> object:
        return revalidate_model_input(value, InvocationReleaseEvidence)

    @model_validator(mode="after")
    def validate_kind(self) -> WorkAttemptLifecycleSettlement:
        if self.kind in {"proposal_deadline_stop", "continuation_deadline_stop"}:
            if (
                self.proposal_id is None
                or self.proposal_request_sha256 is None
                or self.stop_reason != "work_contract_elapsed_limit"
            ):
                raise ValueError("Proposal expiry requires exact proposal and elapsed authority.")
        elif self.proposal_id is not None or self.proposal_request_sha256 is not None:
            raise ValueError("Only proposal expiry accepts proposal identity fields.")
        if self.kind in {"decision_application", "continuation_deadline_stop"}:
            if (
                self.decision_id is None
                or self.application_idempotency_key is None
                or (self.kind == "decision_application" and self.stop_reason is not None)
            ):
                raise ValueError("Decision settlement requires only exact application authority.")
        elif (
            self.stop_reason is None
            or self.decision_id is not None
            or self.application_idempotency_key is not None
        ):
            raise ValueError("Runtime-stop settlement requires only a typed stop reason.")
        require_bounded_work_completion_document(
            self.model_dump(mode="json", warnings=False),
            "work_attempt_lifecycle_settlement",
            max_bytes=32 * 1024,
            max_items=256,
        )
        return self


def copy_work_attempt_lifecycle_settlement(
    value: WorkAttemptLifecycleSettlement,
) -> WorkAttemptLifecycleSettlement:
    if type(value) is not WorkAttemptLifecycleSettlement:
        raise TypeError("Lifecycle settlement requires WorkAttemptLifecycleSettlement.")
    return cast(
        "WorkAttemptLifecycleSettlement",
        revalidate_model_input(value, WorkAttemptLifecycleSettlement),
    )


def work_attempt_lifecycle_settlement_sha256(value: WorkAttemptLifecycleSettlement) -> str:
    copied = copy_work_attempt_lifecycle_settlement(value)
    return sha256(
        canonical_durable_json_bytes(copied.model_dump(mode="json"), "work_attempt_settlement")
    ).hexdigest()


def work_attempt_admission_authority_sha256(value: WorkAttemptAdmission) -> str:
    copied = WorkAttemptAdmission.model_validate(
        revalidate_model_input(value, WorkAttemptAdmission)
    )
    return sha256(
        canonical_durable_json_bytes(copied.model_dump(mode="json"), "work_attempt_admission")
    ).hexdigest()
