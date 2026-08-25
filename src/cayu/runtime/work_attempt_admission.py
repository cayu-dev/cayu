"""Durable authority for contract-bound work-attempt admission and recovery."""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._clock import normalize_utc_datetime
from cayu._validation import (
    canonical_durable_json_bytes,
    require_durable_clean_nonblank,
    revalidate_model_input,
)
from cayu.runtime.invocation import SessionInvocationBinding
from cayu.runtime.work_contracts import (
    WORK_COMPLETION_DECISION_MAX_BYTES,
    WORK_CONTRACT_TASK_MAX_ITEMS,
    CompletionDecision,
    CompletionGap,
    CompletionProposal,
    CompletionProposalCreate,
    CompletionVerdict,
    WorkAttempt,
    WorkAttemptCreate,
    WorkContractRef,
    completion_proposal_request_sha256,
    copy_completion_proposal_create,
    validate_work_completion_linked_id,
    work_attempt_request_sha256,
)

# A continuation embeds one complete valid completion decision.  Reserve fixed
# headroom for the admission, invocation, claim, and attempt authority around
# that decision while retaining a single bounded durable receipt.
WORK_ATTEMPT_ADMISSION_MAX_BYTES = WORK_COMPLETION_DECISION_MAX_BYTES + (128 * 1024)
WORK_ATTEMPT_ADMISSION_MAX_ITEMS = WORK_CONTRACT_TASK_MAX_ITEMS
WORK_ATTEMPT_ADMISSION_ID_MAX_BYTES = 256
WORK_ATTEMPT_ADMISSION_LEASE_MAX_SECONDS = 3_600
WORK_ATTEMPT_ADMISSION_MAX_GENERATIONS = 64
WORK_ATTEMPT_RECOVERY_CHECKPOINT_KEY = "work_attempt_recovery_authority"


def _identity(value: str, field_name: str) -> str:
    value = require_durable_clean_nonblank(value, field_name)
    if len(value.encode("utf-8")) > WORK_ATTEMPT_ADMISSION_ID_MAX_BYTES:
        raise ValueError(
            f"{field_name} must be at most {WORK_ATTEMPT_ADMISSION_ID_MAX_BYTES} UTF-8 bytes."
        )
    return value


def _sha256(value: str, field_name: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest.")
    return value


def _model_sha256(value: BaseModel, field_name: str) -> str:
    return sha256(
        canonical_durable_json_bytes(
            value.model_dump(mode="json", warnings=False),
            field_name,
        )
    ).hexdigest()


class WorkAttemptAdmissionState(StrEnum):
    PREPARING = "preparing"
    ACTIVE = "active"
    RECOVERING = "recovering"
    RELEASED = "released"


class WorkAttemptAdmissionConflict(ValueError):
    """A stable admission identity is bound to conflicting authority."""


class WorkAttemptExecutionClaimLost(ValueError):
    """The caller no longer owns the exact active work-attempt generation."""


class WorkAttemptRecoveryRequired(RuntimeError):
    """Durable admission remains fenced pending exact session reconciliation."""


class WorkAttemptExecutionRequest(BaseModel):
    """Public, stable identities for one runtime-owned execution entrance."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    admission_id: str
    claim_id: str
    attempt_id: str
    interaction_id: str
    worker_id: str
    task_id: str | None = None
    predecessor_admission_id: str | None = None
    generation: StrictInt = Field(ge=1, le=WORK_ATTEMPT_ADMISSION_MAX_GENERATIONS)
    lease_seconds: StrictInt = Field(ge=1, le=WORK_ATTEMPT_ADMISSION_LEASE_MAX_SECONDS)

    @field_validator(
        "admission_id",
        "claim_id",
        "attempt_id",
        "interaction_id",
        "worker_id",
        "predecessor_admission_id",
    )
    @classmethod
    def validate_identity(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _identity(value, info.field_name)

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_work_completion_linked_id(value, "task_id")


class WorkAttemptClaimRenewalRequest(BaseModel):
    """Public request to renew the caller's current execution generation."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    admission_id: str
    claim_id: str
    worker_id: str
    generation: StrictInt = Field(ge=1, le=WORK_ATTEMPT_ADMISSION_MAX_GENERATIONS)
    lease_seconds: StrictInt = Field(ge=1, le=WORK_ATTEMPT_ADMISSION_LEASE_MAX_SECONDS)

    @field_validator("admission_id", "claim_id", "worker_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _identity(value, info.field_name)


class WorkAttemptRecoveryRequest(BaseModel):
    """Public request to replace one expired execution generation."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    admission_id: str
    claim_id: str
    worker_id: str
    generation: StrictInt = Field(ge=2, le=WORK_ATTEMPT_ADMISSION_MAX_GENERATIONS)
    lease_seconds: StrictInt = Field(ge=1, le=WORK_ATTEMPT_ADMISSION_LEASE_MAX_SECONDS)

    @field_validator("admission_id", "claim_id", "worker_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _identity(value, info.field_name)


class WorkAttemptAdmissionPrepare(BaseModel):
    """Store request that reserves one exact attempt before session mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    admission_id: str
    claim_id: str
    attempt_id: str
    task_id: str
    session_id: str
    interaction_id: str
    worker_id: str
    execution_owner_id: str
    generation: StrictInt = Field(ge=1, le=WORK_ATTEMPT_ADMISSION_MAX_GENERATIONS)
    lease_seconds: StrictInt = Field(ge=1, le=WORK_ATTEMPT_ADMISSION_LEASE_MAX_SECONDS)
    kind: Literal["initial", "continuation"]
    predecessor_admission_id: str | None = None
    source_request_sha256: str
    contract: WorkContractRef
    session_invocation: SessionInvocationBinding
    source_execution_profile_fingerprint: str

    @field_validator(
        "admission_id",
        "claim_id",
        "attempt_id",
        "interaction_id",
        "worker_id",
        "execution_owner_id",
        "predecessor_admission_id",
    )
    @classmethod
    def validate_identity(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _identity(value, info.field_name)

    @field_validator("task_id", "session_id")
    @classmethod
    def validate_linked_identity(cls, value: str, info) -> str:
        return validate_work_completion_linked_id(value, info.field_name)

    @field_validator("source_request_sha256", "source_execution_profile_fingerprint")
    @classmethod
    def validate_digest(cls, value: str, info) -> str:
        return _sha256(value, info.field_name)

    @field_validator("contract", mode="before")
    @classmethod
    def copy_contract(cls, value: object) -> object:
        return revalidate_model_input(value, WorkContractRef)

    @field_validator("session_invocation", mode="before")
    @classmethod
    def copy_session_invocation(cls, value: object) -> object:
        return revalidate_model_input(value, SessionInvocationBinding)

    @model_validator(mode="after")
    def validate_predecessor(self) -> WorkAttemptAdmissionPrepare:
        if (self.kind == "continuation") != (self.predecessor_admission_id is not None):
            raise ValueError(
                "Only continuation admission preparation may carry a predecessor admission."
            )
        return self


class WorkAttemptAdmissionActivate(BaseModel):
    """Exact session evidence required to publish one prepared admission."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    admission_id: str
    claim_id: str
    prepare_request_sha256: str
    session_evidence_sha256: str

    @field_validator("admission_id", "claim_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _identity(value, info.field_name)

    @field_validator("prepare_request_sha256", "session_evidence_sha256")
    @classmethod
    def validate_digest(cls, value: str, info) -> str:
        return _sha256(value, info.field_name)


class WorkAttemptExecutionClaimRequest(BaseModel):
    """Renew or replace one attempt's process-aware execution generation."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    admission_id: str
    claim_id: str
    worker_id: str
    execution_owner_id: str
    generation: StrictInt = Field(ge=1, le=WORK_ATTEMPT_ADMISSION_MAX_GENERATIONS)
    lease_seconds: StrictInt = Field(ge=1, le=WORK_ATTEMPT_ADMISSION_LEASE_MAX_SECONDS)

    @field_validator("admission_id", "claim_id", "worker_id", "execution_owner_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _identity(value, info.field_name)


class WorkAttemptRecoveryActivate(BaseModel):
    """Positive session-settlement evidence that activates a recovery generation."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    admission_id: str
    claim_id: str
    generation: StrictInt = Field(ge=2, le=WORK_ATTEMPT_ADMISSION_MAX_GENERATIONS)
    recovery_evidence_sha256: str

    @field_validator("admission_id", "claim_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _identity(value, info.field_name)

    @field_validator("recovery_evidence_sha256")
    @classmethod
    def validate_recovery_evidence_sha256(cls, value: str) -> str:
        return _sha256(value, "recovery_evidence_sha256")


class WorkAttemptExecutionClaim(BaseModel):
    """One immutable generation of renewable work-attempt execution authority."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    admission_id: str
    claim_id: str
    worker_id: str
    execution_owner_id: str
    generation: StrictInt = Field(ge=1, le=WORK_ATTEMPT_ADMISSION_MAX_GENERATIONS)
    request_sha256: str
    claimed_at: datetime
    lease_expires_at: datetime

    @field_validator("admission_id", "claim_id", "worker_id", "execution_owner_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _identity(value, info.field_name)

    @field_validator("request_sha256")
    @classmethod
    def validate_request_sha256(cls, value: str) -> str:
        return _sha256(value, "request_sha256")

    @field_validator("claimed_at", "lease_expires_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime, info) -> datetime:
        return normalize_utc_datetime(value, info.field_name)

    @model_validator(mode="after")
    def validate_lease_window(self) -> WorkAttemptExecutionClaim:
        if self.lease_expires_at <= self.claimed_at:
            raise ValueError("Execution-claim lease must expire after it is claimed.")
        return self


class WorkAttemptContinuationContext(BaseModel):
    """Exact bounded rejection evidence supplied to the next attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    prior_admission_id: str
    prior_attempt_id: str
    proposal_id: str
    decision: CompletionDecision
    application_idempotency_key: str
    gap_fingerprint: str

    @field_validator(
        "prior_admission_id",
        "prior_attempt_id",
        "proposal_id",
        "application_idempotency_key",
    )
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _identity(value, info.field_name)

    @field_validator("gap_fingerprint")
    @classmethod
    def validate_gap_fingerprint(cls, value: str) -> str:
        return _sha256(value, "gap_fingerprint")

    @field_validator("decision", mode="before")
    @classmethod
    def copy_decision(cls, value: object) -> object:
        return revalidate_model_input(value, CompletionDecision)

    @model_validator(mode="after")
    def validate_decision_gaps(self) -> WorkAttemptContinuationContext:
        if self.decision.gap_fingerprint != self.gap_fingerprint:
            raise ValueError("Continuation gap fingerprint conflicts with the decision.")
        return self

    @property
    def gaps(self) -> tuple[CompletionGap, ...]:
        """Return the decision-owned typed gaps without persisting a duplicate."""

        return self.decision.gaps


class WorkAttemptRecoverySessionAuthority(BaseModel):
    """Exact claim tuple retained while a recovered session awaits continuation."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    admission_id: str
    claim_id: str
    generation: StrictInt = Field(ge=1, le=WORK_ATTEMPT_ADMISSION_MAX_GENERATIONS)
    request_sha256: str

    @field_validator("admission_id", "claim_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _identity(value, info.field_name)

    @field_validator("request_sha256")
    @classmethod
    def validate_request_sha256(cls, value: str) -> str:
        return _sha256(value, "request_sha256")

    def checkpoint_value(self) -> dict[str, object]:
        return self.model_dump(mode="python", warnings=False)


def work_attempt_recovery_session_authority(
    claim: WorkAttemptExecutionClaim,
) -> WorkAttemptRecoverySessionAuthority:
    if type(claim) is not WorkAttemptExecutionClaim:
        raise TypeError("Recovery session authority requires a WorkAttemptExecutionClaim.")
    return WorkAttemptRecoverySessionAuthority(
        admission_id=claim.admission_id,
        claim_id=claim.claim_id,
        generation=claim.generation,
        request_sha256=claim.request_sha256,
    )


def work_attempt_recovery_session_authority_from_checkpoint(
    checkpoint: dict[str, Any] | None,
) -> WorkAttemptRecoverySessionAuthority | None:
    if checkpoint is None or WORK_ATTEMPT_RECOVERY_CHECKPOINT_KEY not in checkpoint:
        return None
    value = checkpoint[WORK_ATTEMPT_RECOVERY_CHECKPOINT_KEY]
    if type(value) is not dict:
        raise WorkAttemptRecoveryRequired(
            "Session recovery marker has malformed durable authority."
        )
    try:
        return WorkAttemptRecoverySessionAuthority.model_validate(value)
    except (TypeError, ValueError, UnicodeError):
        raise WorkAttemptRecoveryRequired(
            "Session recovery marker has malformed durable authority."
        ) from None


class WorkAttemptAdmission(BaseModel):
    """Durable, bounded admission intent or active receipt."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    admission_id: str
    prepare_request_sha256: str
    state: WorkAttemptAdmissionState
    attempt_id: str
    task_id: str
    session_id: str
    interaction_id: str
    kind: Literal["initial", "continuation"]
    source_request_sha256: str
    contract: WorkContractRef
    session_invocation: SessionInvocationBinding
    source_execution_profile_fingerprint: str
    claim: WorkAttemptExecutionClaim
    attempt: WorkAttempt | None = None
    continuation: WorkAttemptContinuationContext | None = None
    session_evidence_sha256: str | None = None
    recovery_evidence_sha256: str | None = None
    prepared_at: datetime
    activated_at: datetime | None = None

    @field_validator("admission_id", "attempt_id", "interaction_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _identity(value, info.field_name)

    @field_validator("task_id", "session_id")
    @classmethod
    def validate_linked_identity(cls, value: str, info) -> str:
        return validate_work_completion_linked_id(value, info.field_name)

    @field_validator(
        "prepare_request_sha256",
        "source_request_sha256",
        "source_execution_profile_fingerprint",
        "session_evidence_sha256",
        "recovery_evidence_sha256",
    )
    @classmethod
    def validate_digest(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _sha256(value, info.field_name)

    @field_validator("contract", mode="before")
    @classmethod
    def copy_contract(cls, value: object) -> object:
        return revalidate_model_input(value, WorkContractRef)

    @field_validator("session_invocation", mode="before")
    @classmethod
    def copy_session_invocation(cls, value: object) -> object:
        return revalidate_model_input(value, SessionInvocationBinding)

    @field_validator("claim", mode="before")
    @classmethod
    def copy_claim(cls, value: object) -> object:
        return revalidate_model_input(value, WorkAttemptExecutionClaim)

    @field_validator("attempt", mode="before")
    @classmethod
    def copy_attempt(cls, value: object) -> object:
        if value is None:
            return None
        return revalidate_model_input(value, WorkAttempt)

    @field_validator("continuation", mode="before")
    @classmethod
    def copy_continuation(cls, value: object) -> object:
        if value is None:
            return None
        return revalidate_model_input(value, WorkAttemptContinuationContext)

    @field_validator("prepared_at", "activated_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime | None, info) -> datetime | None:
        if value is None:
            return None
        return normalize_utc_datetime(value, info.field_name)

    @model_validator(mode="after")
    def validate_state(self) -> WorkAttemptAdmission:
        published = self.state is not WorkAttemptAdmissionState.PREPARING
        if published != (self.attempt is not None):
            raise ValueError("Published admissions must carry their work attempt.")
        if published != (self.session_evidence_sha256 is not None):
            raise ValueError("Published admissions must carry final session evidence.")
        if published != (self.activated_at is not None):
            raise ValueError("Published admissions must carry an activation timestamp.")
        if self.claim.admission_id != self.admission_id:
            raise ValueError("Execution claim belongs to another admission.")
        if (self.kind == "continuation") != (self.continuation is not None):
            raise ValueError("Only continuation admissions may carry prior decision authority.")
        if self.continuation is not None:
            decision = self.continuation.decision
            if (
                decision.verdict is not CompletionVerdict.REJECTED
                or decision.task_id != self.task_id
                or decision.attempt_id != self.continuation.prior_attempt_id
                or decision.proposal_id != self.continuation.proposal_id
                or decision.contract != self.contract
            ):
                raise ValueError("Continuation decision conflicts with the admission authority.")
        if self.attempt is not None and (
            self.attempt.attempt_id != self.attempt_id
            or self.attempt.task_id != self.task_id
            or self.attempt.session_id != self.session_id
            or self.attempt.contract != self.contract
            or self.attempt.execution_profile_fingerprint
            != self.source_execution_profile_fingerprint
            or self.attempt.request_sha256
            != work_attempt_request_sha256(
                WorkAttemptCreate(
                    attempt_id=self.attempt_id,
                    task_id=self.task_id,
                    session_id=self.session_id,
                    contract=self.contract,
                    execution_profile_fingerprint=(self.source_execution_profile_fingerprint),
                    worker_id=self.attempt.worker_id,
                )
            )
        ):
            raise ValueError("Published work attempt conflicts with its admission.")
        document = self.model_dump(mode="json", warnings=False)
        pending: list[object] = [document]
        item_count = 0
        while pending:
            item = pending.pop()
            item_count += 1
            if item_count > WORK_ATTEMPT_ADMISSION_MAX_ITEMS:
                raise ValueError("Work-attempt admission exceeds the durable item-count limit.")
            if type(item) is dict:
                pending.extend(item.values())
            elif type(item) is list:
                pending.extend(item)
        encoded = canonical_durable_json_bytes(
            document,
            "work_attempt_admission",
        )
        if len(encoded) > WORK_ATTEMPT_ADMISSION_MAX_BYTES:
            raise ValueError(
                f"Work-attempt admission must not exceed {WORK_ATTEMPT_ADMISSION_MAX_BYTES} bytes."
            )
        return self


class AdmittedCompletionProposalRequest(BaseModel):
    """A completion proposal fenced to an exact active execution claim."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    admission_id: str
    claim_id: str
    execution_owner_id: str
    generation: StrictInt = Field(ge=1, le=WORK_ATTEMPT_ADMISSION_MAX_GENERATIONS)
    proposal: CompletionProposalCreate

    @field_validator("admission_id", "claim_id", "execution_owner_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _identity(value, info.field_name)

    @field_validator("proposal", mode="before")
    @classmethod
    def copy_proposal(cls, value: object) -> object:
        if type(value) is CompletionProposalCreate:
            return copy_completion_proposal_create(value)
        return revalidate_model_input(value, CompletionProposalCreate)


class WorkAttemptProposalRequest(BaseModel):
    """Public proposal request completed with process authority by ``CayuApp``."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    admission_id: str
    claim_id: str
    generation: StrictInt = Field(ge=1, le=WORK_ATTEMPT_ADMISSION_MAX_GENERATIONS)
    proposal: CompletionProposalCreate

    @field_validator("admission_id", "claim_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _identity(value, info.field_name)

    @field_validator("proposal", mode="before")
    @classmethod
    def copy_proposal(cls, value: object) -> object:
        if type(value) is CompletionProposalCreate:
            return copy_completion_proposal_create(value)
        return revalidate_model_input(value, CompletionProposalCreate)


def copy_work_attempt_admission_prepare(
    value: WorkAttemptAdmissionPrepare,
) -> WorkAttemptAdmissionPrepare:
    if type(value) is not WorkAttemptAdmissionPrepare:
        raise TypeError("Admission preparation requires a WorkAttemptAdmissionPrepare.")
    return WorkAttemptAdmissionPrepare.model_validate(
        value.model_dump(mode="python", warnings=False)
    )


def copy_work_attempt_execution_request(
    value: WorkAttemptExecutionRequest,
) -> WorkAttemptExecutionRequest:
    if type(value) is not WorkAttemptExecutionRequest:
        raise TypeError("Work-attempt execution requires a WorkAttemptExecutionRequest.")
    return WorkAttemptExecutionRequest.model_validate(
        value.model_dump(mode="python", warnings=False)
    )


def copy_work_attempt_claim_renewal_request(
    value: WorkAttemptClaimRenewalRequest,
) -> WorkAttemptClaimRenewalRequest:
    if type(value) is not WorkAttemptClaimRenewalRequest:
        raise TypeError("Claim renewal requires a WorkAttemptClaimRenewalRequest.")
    return WorkAttemptClaimRenewalRequest.model_validate(
        value.model_dump(mode="python", warnings=False)
    )


def copy_work_attempt_recovery_request(
    value: WorkAttemptRecoveryRequest,
) -> WorkAttemptRecoveryRequest:
    if type(value) is not WorkAttemptRecoveryRequest:
        raise TypeError("Work-attempt recovery requires a WorkAttemptRecoveryRequest.")
    return WorkAttemptRecoveryRequest.model_validate(
        value.model_dump(mode="python", warnings=False)
    )


def copy_work_attempt_admission_activate(
    value: WorkAttemptAdmissionActivate,
) -> WorkAttemptAdmissionActivate:
    if type(value) is not WorkAttemptAdmissionActivate:
        raise TypeError("Admission activation requires a WorkAttemptAdmissionActivate.")
    return WorkAttemptAdmissionActivate.model_validate(
        value.model_dump(mode="python", warnings=False)
    )


def copy_work_attempt_execution_claim_request(
    value: WorkAttemptExecutionClaimRequest,
) -> WorkAttemptExecutionClaimRequest:
    if type(value) is not WorkAttemptExecutionClaimRequest:
        raise TypeError("Execution-claim mutation requires a WorkAttemptExecutionClaimRequest.")
    return WorkAttemptExecutionClaimRequest.model_validate(
        value.model_dump(mode="python", warnings=False)
    )


def copy_work_attempt_recovery_activate(
    value: WorkAttemptRecoveryActivate,
) -> WorkAttemptRecoveryActivate:
    if type(value) is not WorkAttemptRecoveryActivate:
        raise TypeError("Recovery activation requires a WorkAttemptRecoveryActivate.")
    return WorkAttemptRecoveryActivate.model_validate(
        value.model_dump(mode="python", warnings=False)
    )


def copy_admitted_completion_proposal_request(
    value: AdmittedCompletionProposalRequest,
) -> AdmittedCompletionProposalRequest:
    if type(value) is not AdmittedCompletionProposalRequest:
        raise TypeError("Admitted proposal requires an AdmittedCompletionProposalRequest.")
    return AdmittedCompletionProposalRequest.model_validate(
        value.model_dump(mode="python", warnings=False)
    )


def copy_work_attempt_proposal_request(
    value: WorkAttemptProposalRequest,
) -> WorkAttemptProposalRequest:
    if type(value) is not WorkAttemptProposalRequest:
        raise TypeError("Work-attempt proposal requires a WorkAttemptProposalRequest.")
    return WorkAttemptProposalRequest.model_validate(
        value.model_dump(mode="python", warnings=False)
    )


def work_attempt_admission_prepare_sha256(value: WorkAttemptAdmissionPrepare) -> str:
    return _model_sha256(copy_work_attempt_admission_prepare(value), "work_attempt_admission")


def work_attempt_execution_claim_request_sha256(
    value: WorkAttemptExecutionClaimRequest,
) -> str:
    return _model_sha256(
        copy_work_attempt_execution_claim_request(value),
        "work_attempt_execution_claim",
    )


def require_work_attempt_admission_result(
    value: object,
    *,
    operation_name: str,
) -> WorkAttemptAdmission:
    """Return one detached, fully validated admission supplied by a store."""

    if type(value) is not WorkAttemptAdmission:
        raise RuntimeError(f"{operation_name} returned invalid work-attempt authority.")
    try:
        return WorkAttemptAdmission.model_validate(value.model_dump(mode="python", warnings=False))
    except (TypeError, ValueError):
        raise RuntimeError(f"{operation_name} returned invalid work-attempt authority.") from None


def require_work_attempt_execution_claim_result(
    value: object,
    *,
    operation_name: str,
) -> WorkAttemptExecutionClaim:
    """Return one detached, fully validated historical execution claim."""

    if type(value) is not WorkAttemptExecutionClaim:
        raise RuntimeError(f"{operation_name} returned invalid execution-claim authority.")
    try:
        return WorkAttemptExecutionClaim.model_validate(
            value.model_dump(mode="python", warnings=False)
        )
    except (TypeError, ValueError):
        raise RuntimeError(
            f"{operation_name} returned invalid execution-claim authority."
        ) from None


def _work_attempt_preparation_authority(
    value: WorkAttemptAdmission,
) -> tuple[object, ...]:
    return (
        value.admission_id,
        value.prepare_request_sha256,
        value.attempt_id,
        value.task_id,
        value.session_id,
        value.interaction_id,
        value.kind,
        value.source_request_sha256,
        value.contract,
        value.session_invocation,
        value.source_execution_profile_fingerprint,
        value.continuation,
        value.prepared_at,
    )


def _work_attempt_publication_authority(
    value: WorkAttemptAdmission,
) -> tuple[object, ...]:
    return (
        *_work_attempt_preparation_authority(value),
        value.attempt,
        value.session_evidence_sha256,
        value.activated_at,
    )


def require_work_attempt_preparation_result(
    value: object,
    request: WorkAttemptAdmissionPrepare,
) -> WorkAttemptAdmission:
    """Bind a preparation result to every runtime-supplied request field."""

    request = copy_work_attempt_admission_prepare(request)
    admission = require_work_attempt_admission_result(
        value,
        operation_name="Work-attempt admission preparation",
    )
    claim_request = WorkAttemptExecutionClaimRequest(
        admission_id=request.admission_id,
        claim_id=request.claim_id,
        worker_id=request.worker_id,
        execution_owner_id=request.execution_owner_id,
        generation=request.generation,
        lease_seconds=request.lease_seconds,
    )
    claim = admission.claim
    if (
        admission.state is not WorkAttemptAdmissionState.PREPARING
        or admission.admission_id != request.admission_id
        or admission.prepare_request_sha256 != work_attempt_admission_prepare_sha256(request)
        or admission.attempt_id != request.attempt_id
        or admission.task_id != request.task_id
        or admission.session_id != request.session_id
        or admission.interaction_id != request.interaction_id
        or admission.kind != request.kind
        or admission.source_request_sha256 != request.source_request_sha256
        or admission.contract != request.contract
        or admission.session_invocation != request.session_invocation
        or admission.source_execution_profile_fingerprint
        != request.source_execution_profile_fingerprint
        or claim.admission_id != request.admission_id
        or claim.claim_id != request.claim_id
        or claim.worker_id != request.worker_id
        or claim.execution_owner_id != request.execution_owner_id
        or claim.generation != request.generation
        or claim.request_sha256 != work_attempt_execution_claim_request_sha256(claim_request)
        or claim.lease_expires_at - claim.claimed_at != timedelta(seconds=request.lease_seconds)
    ):
        raise RuntimeError("Work-attempt admission preparation returned conflicting authority.")
    return admission


def require_work_attempt_activation_result(
    value: object,
    prepared: WorkAttemptAdmission,
    request: WorkAttemptAdmissionActivate,
) -> WorkAttemptAdmission:
    """Validate publication without permitting immutable preparation drift."""

    prepared = require_work_attempt_admission_result(
        prepared,
        operation_name="Prepared work-attempt admission",
    )
    request = copy_work_attempt_admission_activate(request)
    active = require_work_attempt_admission_result(
        value,
        operation_name="Work-attempt admission activation",
    )
    if (
        _work_attempt_preparation_authority(active) != _work_attempt_preparation_authority(prepared)
        or active.state is not WorkAttemptAdmissionState.ACTIVE
        or active.claim != prepared.claim
        or active.attempt is None
        or active.attempt.worker_id != prepared.claim.worker_id
        or active.session_evidence_sha256 != request.session_evidence_sha256
        or active.recovery_evidence_sha256 != prepared.recovery_evidence_sha256
    ):
        raise RuntimeError("Work-attempt admission activation returned conflicting authority.")
    if prepared.state is WorkAttemptAdmissionState.ACTIVE and (
        _work_attempt_publication_authority(active) != _work_attempt_publication_authority(prepared)
    ):
        raise RuntimeError("Work-attempt admission activation changed its exact receipt.")
    return active


def require_work_attempt_claim_result(
    value: object,
    previous: WorkAttemptAdmission,
    request: WorkAttemptExecutionClaimRequest,
    *,
    operation_name: str,
    allowed_states: frozenset[WorkAttemptAdmissionState],
    allowed_previous_states: frozenset[WorkAttemptAdmissionState],
    renewal: bool = False,
) -> WorkAttemptAdmission:
    """Validate a renewed or replacement claim against prior durable authority."""

    previous = require_work_attempt_admission_result(
        previous,
        operation_name="Prior work-attempt admission lookup",
    )
    request = copy_work_attempt_execution_claim_request(request)
    result = require_work_attempt_admission_result(value, operation_name=operation_name)
    claim = result.claim
    expected_request_sha256 = (
        previous.claim.request_sha256
        if renewal
        else work_attempt_execution_claim_request_sha256(request)
    )
    if (
        _work_attempt_preparation_authority(result) != _work_attempt_preparation_authority(previous)
        or previous.state not in allowed_previous_states
        or result.state not in allowed_states
        or claim.admission_id != request.admission_id
        or claim.claim_id != request.claim_id
        or claim.worker_id != request.worker_id
        or claim.execution_owner_id != request.execution_owner_id
        or claim.generation != request.generation
        or claim.request_sha256 != expected_request_sha256
    ):
        raise RuntimeError(f"{operation_name} returned conflicting authority.")
    if previous.attempt is not None and (
        _work_attempt_publication_authority(result) != _work_attempt_publication_authority(previous)
    ):
        raise RuntimeError(f"{operation_name} changed immutable published authority.")
    if renewal and (
        previous.state is not WorkAttemptAdmissionState.ACTIVE
        or result.state is not WorkAttemptAdmissionState.ACTIVE
        or claim.claim_id != previous.claim.claim_id
        or claim.claimed_at != previous.claim.claimed_at
        or claim.lease_expires_at < previous.claim.lease_expires_at
        or result.recovery_evidence_sha256 != previous.recovery_evidence_sha256
    ):
        raise RuntimeError(f"{operation_name} returned conflicting renewed authority.")
    if not renewal:
        same_claim = claim.claim_id == previous.claim.claim_id
        if same_claim:
            if claim != previous.claim:
                raise RuntimeError(f"{operation_name} changed immutable claim authority.")
            forward_activation = (
                previous.state
                in {
                    WorkAttemptAdmissionState.PREPARING,
                    WorkAttemptAdmissionState.RECOVERING,
                }
                and result.state is WorkAttemptAdmissionState.ACTIVE
            )
            if not forward_activation and result != previous:
                raise RuntimeError(f"{operation_name} changed its exact replay receipt.")
            if previous.state is WorkAttemptAdmissionState.RECOVERING and (
                result.state is WorkAttemptAdmissionState.ACTIVE
                and result.recovery_evidence_sha256 is None
            ):
                raise RuntimeError(f"{operation_name} returned unauthenticated recovery authority.")
            if previous.state is WorkAttemptAdmissionState.PREPARING and (
                result.state is WorkAttemptAdmissionState.ACTIVE
                and result.recovery_evidence_sha256 is not None
            ):
                raise RuntimeError(f"{operation_name} returned conflicting activation authority.")
        else:
            replacement_states = (
                {
                    WorkAttemptAdmissionState.PREPARING,
                    WorkAttemptAdmissionState.ACTIVE,
                }
                if previous.state is WorkAttemptAdmissionState.PREPARING
                else {
                    WorkAttemptAdmissionState.RECOVERING,
                    WorkAttemptAdmissionState.ACTIVE,
                }
            )
            if (
                result.state not in replacement_states
                or claim.generation != previous.claim.generation + 1
                or claim.claimed_at < previous.claim.lease_expires_at
                or claim.lease_expires_at - claim.claimed_at
                != timedelta(seconds=request.lease_seconds)
            ):
                raise RuntimeError(f"{operation_name} returned conflicting replacement authority.")
            if (
                previous.state
                in {
                    WorkAttemptAdmissionState.ACTIVE,
                    WorkAttemptAdmissionState.RECOVERING,
                }
                and result.state is WorkAttemptAdmissionState.ACTIVE
                and result.recovery_evidence_sha256 is None
            ):
                raise RuntimeError(f"{operation_name} returned unauthenticated recovery authority.")
            if previous.state is WorkAttemptAdmissionState.PREPARING and (
                result.state is WorkAttemptAdmissionState.ACTIVE
                and result.recovery_evidence_sha256 is not None
            ):
                raise RuntimeError(f"{operation_name} returned conflicting activation authority.")
    if result.state is not WorkAttemptAdmissionState.ACTIVE and (
        result.recovery_evidence_sha256 is not None
    ):
        raise RuntimeError(f"{operation_name} returned premature recovery evidence.")
    return result


def require_work_attempt_recovery_activation_result(
    value: object,
    recovering: WorkAttemptAdmission,
    request: WorkAttemptRecoveryActivate,
) -> WorkAttemptAdmission:
    """Bind recovery activation to its exact claim and session evidence."""

    recovering = require_work_attempt_admission_result(
        recovering,
        operation_name="Recovering work-attempt admission",
    )
    request = copy_work_attempt_recovery_activate(request)
    active = require_work_attempt_admission_result(
        value,
        operation_name="Work-attempt recovery activation",
    )
    if (
        _work_attempt_publication_authority(active)
        != _work_attempt_publication_authority(recovering)
        or active.state is not WorkAttemptAdmissionState.ACTIVE
        or active.claim != recovering.claim
        or active.recovery_evidence_sha256 != request.recovery_evidence_sha256
    ):
        raise RuntimeError("Work-attempt recovery activation returned conflicting authority.")
    return active


def require_admitted_completion_proposal_result(
    value: object,
    request: AdmittedCompletionProposalRequest,
    admission: WorkAttemptAdmission,
) -> CompletionProposal:
    """Bind the public proposal result to the exact request and admission."""

    request = copy_admitted_completion_proposal_request(request)
    admission = require_work_attempt_admission_result(
        admission,
        operation_name="Admitted completion proposal authority lookup",
    )
    if type(value) is not CompletionProposal:
        raise RuntimeError("Task store returned an invalid completion proposal.")
    try:
        proposal = CompletionProposal.model_validate(
            value.model_dump(mode="python", warnings=False)
        )
    except (TypeError, ValueError):
        raise RuntimeError("Task store returned an invalid completion proposal.") from None
    expected = request.proposal
    if (
        admission.attempt is None
        or admission.attempt_id != expected.attempt_id
        or proposal.proposal_id != expected.proposal_id
        or proposal.attempt_id != expected.attempt_id
        or proposal.result != expected.result
        or proposal.evidence_references != expected.evidence_references
        or proposal.task_id != admission.task_id
        or proposal.contract != admission.contract
        or proposal.request_sha256 != completion_proposal_request_sha256(expected)
    ):
        raise RuntimeError("Task store returned a conflicting completion proposal.")
    return proposal


WorkAttemptAdmissionKind = Literal["initial", "continuation"]


__all__ = [
    "WORK_ATTEMPT_ADMISSION_LEASE_MAX_SECONDS",
    "WORK_ATTEMPT_ADMISSION_MAX_GENERATIONS",
    "AdmittedCompletionProposalRequest",
    "WorkAttemptAdmission",
    "WorkAttemptAdmissionActivate",
    "WorkAttemptAdmissionConflict",
    "WorkAttemptAdmissionKind",
    "WorkAttemptAdmissionPrepare",
    "WorkAttemptAdmissionState",
    "WorkAttemptClaimRenewalRequest",
    "WorkAttemptContinuationContext",
    "WorkAttemptExecutionClaim",
    "WorkAttemptExecutionClaimLost",
    "WorkAttemptExecutionClaimRequest",
    "WorkAttemptExecutionRequest",
    "WorkAttemptProposalRequest",
    "WorkAttemptRecoveryActivate",
    "WorkAttemptRecoveryRequest",
    "WorkAttemptRecoveryRequired",
    "WorkAttemptRecoverySessionAuthority",
    "copy_admitted_completion_proposal_request",
    "copy_work_attempt_admission_activate",
    "copy_work_attempt_admission_prepare",
    "copy_work_attempt_claim_renewal_request",
    "copy_work_attempt_execution_claim_request",
    "copy_work_attempt_execution_request",
    "copy_work_attempt_proposal_request",
    "copy_work_attempt_recovery_activate",
    "copy_work_attempt_recovery_request",
    "require_admitted_completion_proposal_result",
    "require_work_attempt_activation_result",
    "require_work_attempt_admission_result",
    "require_work_attempt_claim_result",
    "require_work_attempt_execution_claim_result",
    "require_work_attempt_preparation_result",
    "require_work_attempt_recovery_activation_result",
    "work_attempt_admission_prepare_sha256",
    "work_attempt_execution_claim_request_sha256",
    "work_attempt_recovery_session_authority",
    "work_attempt_recovery_session_authority_from_checkpoint",
]
