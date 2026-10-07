"""Durable provider-dispatch ledger for provider-backed completion verifiers.

Each record is one provider attempt made on behalf of a completion verifier. The
intent is persisted under the live verification claim before the provider is
entered, and the settlement is written once afterwards. Records live beside the
verified-work authority they serve and never in a worker session, so verifier
usage stays separate from the worker invocation it evaluates.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from hashlib import sha256
from typing import Literal

from pydantic import Field, StrictBool, StrictFloat, StrictInt, field_validator, model_validator

from cayu._clock import normalize_utc_datetime
from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    canonical_durable_json_bytes,
    require_durable_clean_nonblank,
    revalidate_model_input,
)
from cayu.budgets.usage import UsageMetrics
from cayu.tasks.contracts import (
    WORK_CONTRACT_IDENTIFIER_MAX_BYTES,
    WORK_VERIFICATION_LEASE_MAX_SECONDS,
    CompletionProposal,
    CompletionVerificationClaim,
    CompletionVerificationClaimLost,
    CompletionVerifierDecision,
    CompletionVerifierKind,
    CompletionVerifierRef,
    FrozenWorkContractModel,
    WorkCompletionConflict,
    WorkContractRef,
    copy_completion_verifier_decision,
    validate_work_completion_linked_id,
)

COMPLETION_VERIFIER_DISPATCH_SCHEMA_VERSION = 1
#: Hard ceiling for provider attempts recorded against one completion proposal.
COMPLETION_VERIFIER_DISPATCH_MAX_ATTEMPTS = 64
#: Hard ceiling for provider attempts inside one verifier execution.
COMPLETION_VERIFIER_DISPATCH_MAX_ATTEMPTS_PER_EXECUTION = 16

_CANONICAL_CODE_CHARACTERS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789._:-")


class CompletionVerifierDispatchBudgetExhausted(ValueError):
    """A provider verifier dispatch would exceed its durable verifier budget."""


class CompletionVerifierDispatchOutcome(StrEnum):
    """Runtime-observed outcome of one provider attempt."""

    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    OUTCOME_UNKNOWN = "outcome_unknown"


class CompletionVerifierUsageStatus(StrEnum):
    OBSERVED = "observed"
    MISSING = "missing"
    MALFORMED = "malformed"


class CompletionVerifierDecodeStatus(StrEnum):
    DECODED = "decoded"
    INVALID = "invalid"


def _identifier(value: str, field_name: str) -> str:
    value = require_durable_clean_nonblank(value, field_name)
    if (
        len(value) > WORK_CONTRACT_IDENTIFIER_MAX_BYTES
        or len(value.encode("utf-8")) > WORK_CONTRACT_IDENTIFIER_MAX_BYTES
    ):
        raise ValueError(
            f"{field_name} must not exceed {WORK_CONTRACT_IDENTIFIER_MAX_BYTES} UTF-8 bytes."
        )
    return value


def _digest(value: str, field_name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest.")
    return value


def _code(value: str, field_name: str) -> str:
    value = require_durable_clean_nonblank(value, field_name)
    if (
        len(value) > 128
        or value[0] not in "abcdefghijklmnopqrstuvwxyz0123456789"
        or any(character not in _CANONICAL_CODE_CHARACTERS for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase canonical code.")
    return value


class CompletionVerifierDispatchBudget(FrozenWorkContractModel):
    """Durable verifier budget shared by every provider attempt for one proposal.

    ``max_attempts`` counts every provider attempt for the proposal, including
    provider retries inside one verifier execution and attempts made by later
    verifier executions. Token ceilings count observed usage for settled
    attempts and the declared request envelope for attempts whose usage is
    unknown, so an unsettled dispatch can never free budget by being forgotten.
    """

    max_attempts: StrictInt = Field(ge=1, le=COMPLETION_VERIFIER_DISPATCH_MAX_ATTEMPTS)
    max_input_tokens: StrictInt | None = Field(default=None, gt=0, le=MAX_DURABLE_JSON_INTEGER)
    max_output_tokens: StrictInt | None = Field(default=None, gt=0, le=MAX_DURABLE_JSON_INTEGER)


class CompletionVerifierDispatchRequest(FrozenWorkContractModel):
    """Runtime-authored intent to enter one provider attempt under a live claim."""

    dispatch_id: str
    proposal_id: str
    claim_id: str
    worker_id: str
    execution_owner_id: str
    claim_attempt_number: StrictInt = Field(ge=1)
    verifier: CompletionVerifierRef
    verifier_profile_fingerprint: str
    provider_attempt: StrictInt = Field(
        ge=1, le=COMPLETION_VERIFIER_DISPATCH_MAX_ATTEMPTS_PER_EXECUTION
    )
    provider_name: str
    pricing_provider_name: str
    model: str
    request_sha256: str
    max_input_tokens: StrictInt = Field(gt=0, le=MAX_DURABLE_JSON_INTEGER)
    max_output_tokens: StrictInt = Field(gt=0, le=MAX_DURABLE_JSON_INTEGER)
    timeout_seconds: StrictFloat = Field(gt=0, le=WORK_VERIFICATION_LEASE_MAX_SECONDS)
    budget: CompletionVerifierDispatchBudget

    @field_validator(
        "dispatch_id",
        "proposal_id",
        "claim_id",
        "worker_id",
        "execution_owner_id",
        "provider_name",
        "pricing_provider_name",
        "model",
    )
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _identifier(value, info.field_name)

    @field_validator("verifier", mode="before")
    @classmethod
    def copy_verifier(cls, value: object) -> object:
        return revalidate_model_input(value, CompletionVerifierRef)

    @field_validator("budget", mode="before")
    @classmethod
    def copy_budget(cls, value: object) -> object:
        return revalidate_model_input(value, CompletionVerifierDispatchBudget)

    @field_validator("verifier_profile_fingerprint", "request_sha256")
    @classmethod
    def validate_digest(cls, value: str, info) -> str:
        return _digest(value, info.field_name)

    @model_validator(mode="after")
    def validate_dispatch(self) -> CompletionVerifierDispatchRequest:
        if self.verifier.kind is not CompletionVerifierKind.PROVIDER:
            raise ValueError("Only provider-backed verifiers record provider dispatches.")
        if self.dispatch_id != completion_verifier_dispatch_id(
            proposal_id=self.proposal_id,
            claim_id=self.claim_id,
            claim_attempt_number=self.claim_attempt_number,
            provider_attempt=self.provider_attempt,
        ):
            raise ValueError("dispatch_id conflicts with its runtime-derived identity.")
        if self.provider_attempt > self.budget.max_attempts:
            raise ValueError("provider_attempt exceeds the verifier attempt budget.")
        if (
            self.budget.max_input_tokens is not None
            and self.max_input_tokens > self.budget.max_input_tokens
        ) or (
            self.budget.max_output_tokens is not None
            and self.max_output_tokens > self.budget.max_output_tokens
        ):
            raise ValueError("A dispatch envelope cannot exceed the verifier token budget.")
        return self


class CompletionVerifierDispatchFailure(FrozenWorkContractModel):
    """Bounded, content-free classification of an unsuccessful attempt."""

    code: str
    status_code: StrictInt | None = Field(default=None, ge=100, le=599)
    retryable: StrictBool | None = None

    @field_validator("code")
    @classmethod
    def validate_code(cls, value: str) -> str:
        return _code(value, "code")


class CompletionVerifierDispatchSettlementRequest(FrozenWorkContractModel):
    """Runtime-observed terminal accounting for one provider attempt."""

    dispatch_id: str
    outcome: CompletionVerifierDispatchOutcome
    usage_status: CompletionVerifierUsageStatus
    usage: UsageMetrics | None = None
    latency_ms: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    failure: CompletionVerifierDispatchFailure | None = None
    response_sha256: str | None = None
    decode_status: CompletionVerifierDecodeStatus | None = None
    decision: CompletionVerifierDecision | None = None

    @field_validator("dispatch_id")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        return _identifier(value, "dispatch_id")

    @field_validator("usage", mode="before")
    @classmethod
    def copy_usage(cls, value: object) -> object:
        if value is None:
            return None
        return revalidate_model_input(value, UsageMetrics)

    @field_validator("failure", mode="before")
    @classmethod
    def copy_failure(cls, value: object) -> object:
        if value is None:
            return None
        return revalidate_model_input(value, CompletionVerifierDispatchFailure)

    @field_validator("decision", mode="before")
    @classmethod
    def copy_decision(cls, value: object) -> object:
        if value is None:
            return None
        return revalidate_model_input(value, CompletionVerifierDecision)

    @field_validator("response_sha256")
    @classmethod
    def validate_response_sha256(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _digest(value, "response_sha256")

    @model_validator(mode="after")
    def validate_settlement(self) -> CompletionVerifierDispatchSettlementRequest:
        if (self.usage_status is CompletionVerifierUsageStatus.OBSERVED) != (
            self.usage is not None
        ):
            raise ValueError("Observed usage requires usage metrics, and only observed usage.")
        if self.outcome is CompletionVerifierDispatchOutcome.COMPLETED:
            if self.response_sha256 is None or self.decode_status is None:
                raise ValueError("Completed dispatches require response and decode evidence.")
            decoded = self.decode_status is CompletionVerifierDecodeStatus.DECODED
            if decoded != (self.decision is not None):
                raise ValueError("Only a decoded response carries a verifier decision.")
            if decoded != (self.failure is None):
                raise ValueError("Only an invalid completed response carries a failure.")
        elif (
            self.response_sha256 is not None
            or self.decode_status is not None
            or self.decision is not None
            or self.failure is None
        ):
            raise ValueError("Unsuccessful dispatches carry a failure and no response or decision.")
        return self


class CompletionVerifierDispatchSettlement(CompletionVerifierDispatchSettlementRequest):
    request_sha256: str
    settled_at: datetime

    @field_validator("request_sha256")
    @classmethod
    def validate_request_sha256(cls, value: str) -> str:
        return _digest(value, "request_sha256")

    @field_validator("settled_at")
    @classmethod
    def normalize_settled_at(cls, value: datetime) -> datetime:
        return normalize_utc_datetime(value, "settled_at")


class CompletionVerifierDispatch(FrozenWorkContractModel):
    """One durable provider attempt and, once observed, its settlement."""

    schema_version: Literal[1] = COMPLETION_VERIFIER_DISPATCH_SCHEMA_VERSION
    dispatch_id: str
    proposal_id: str
    task_id: str
    attempt_id: str
    contract: WorkContractRef
    ordinal: StrictInt = Field(ge=1, le=COMPLETION_VERIFIER_DISPATCH_MAX_ATTEMPTS)
    request: CompletionVerifierDispatchRequest
    request_sha256: str
    dispatched_at: datetime
    settlement: CompletionVerifierDispatchSettlement | None = None

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1.")
        return value

    @field_validator("dispatch_id", "proposal_id", "attempt_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _identifier(value, info.field_name)

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, value: str) -> str:
        return validate_work_completion_linked_id(value, "task_id")

    @field_validator("contract", mode="before")
    @classmethod
    def copy_contract(cls, value: object) -> object:
        return revalidate_model_input(value, WorkContractRef)

    @field_validator("request", mode="before")
    @classmethod
    def copy_request(cls, value: object) -> object:
        return revalidate_model_input(value, CompletionVerifierDispatchRequest)

    @field_validator("settlement", mode="before")
    @classmethod
    def copy_settlement(cls, value: object) -> object:
        if value is None:
            return None
        return revalidate_model_input(value, CompletionVerifierDispatchSettlement)

    @field_validator("request_sha256")
    @classmethod
    def validate_request_sha256(cls, value: str) -> str:
        return _digest(value, "request_sha256")

    @field_validator("dispatched_at")
    @classmethod
    def normalize_dispatched_at(cls, value: datetime) -> datetime:
        return normalize_utc_datetime(value, "dispatched_at")

    @model_validator(mode="after")
    def validate_record(self) -> CompletionVerifierDispatch:
        if (
            self.request.dispatch_id != self.dispatch_id
            or self.request.proposal_id != self.proposal_id
        ):
            raise ValueError("Dispatch record conflicts with its request identity.")
        if self.request_sha256 != completion_verifier_dispatch_request_sha256(self.request):
            raise ValueError("Dispatch record conflicts with its request digest.")
        settlement = self.settlement
        if settlement is not None:
            if settlement.dispatch_id != self.dispatch_id:
                raise ValueError("Dispatch settlement belongs to another dispatch.")
            if settlement.request_sha256 != completion_verifier_dispatch_settlement_sha256(
                _settlement_request(settlement)
            ):
                raise ValueError("Dispatch settlement conflicts with its digest.")
            if settlement.settled_at < self.dispatched_at:
                raise ValueError("Dispatch settlement cannot precede its dispatch.")
            _require_settlement_target(self.request, settlement)
        return self


def completion_verifier_dispatch_id(
    *,
    proposal_id: str,
    claim_id: str,
    claim_attempt_number: int,
    provider_attempt: int,
) -> str:
    """Derive one stable attempt identity from its verifier execution authority."""

    digest = sha256(
        canonical_durable_json_bytes(
            {
                "domain": "cayu.completion-verifier-dispatch.v1",
                "proposal_id": proposal_id,
                "claim_id": claim_id,
                "claim_attempt_number": claim_attempt_number,
                "provider_attempt": provider_attempt,
            },
            "completion_verifier_dispatch_id",
        )
    ).hexdigest()
    return f"cvdisp_{digest[:48]}"


def completion_verifier_dispatch_request_sha256(value: CompletionVerifierDispatchRequest) -> str:
    copied = copy_completion_verifier_dispatch_request(value)
    return sha256(
        canonical_durable_json_bytes(
            copied.model_dump(mode="json", warnings=False),
            "completion_verifier_dispatch",
        )
    ).hexdigest()


def completion_verifier_dispatch_settlement_sha256(
    value: CompletionVerifierDispatchSettlementRequest,
) -> str:
    copied = copy_completion_verifier_dispatch_settlement_request(value)
    return sha256(
        canonical_durable_json_bytes(
            copied.model_dump(mode="json", warnings=False),
            "completion_verifier_dispatch_settlement",
        )
    ).hexdigest()


def copy_completion_verifier_dispatch_request(
    value: CompletionVerifierDispatchRequest,
) -> CompletionVerifierDispatchRequest:
    if type(value) is not CompletionVerifierDispatchRequest:
        raise TypeError("Dispatch intent must be a CompletionVerifierDispatchRequest.")
    return CompletionVerifierDispatchRequest.model_validate(
        value.model_dump(mode="python", warnings=False)
    )


def copy_completion_verifier_dispatch_settlement_request(
    value: CompletionVerifierDispatchSettlementRequest,
) -> CompletionVerifierDispatchSettlementRequest:
    if type(value) is not CompletionVerifierDispatchSettlementRequest:
        raise TypeError(
            "Dispatch settlement must be a CompletionVerifierDispatchSettlementRequest."
        )
    return CompletionVerifierDispatchSettlementRequest.model_validate(
        value.model_dump(mode="python", warnings=False)
    )


def copy_completion_verifier_dispatch(
    value: CompletionVerifierDispatch,
) -> CompletionVerifierDispatch:
    if type(value) is not CompletionVerifierDispatch:
        raise TypeError("Dispatch record must be a CompletionVerifierDispatch.")
    return CompletionVerifierDispatch.model_validate(
        value.model_dump(mode="python", warnings=False)
    )


def completion_verifier_dispatch_from_document(value: object) -> CompletionVerifierDispatch:
    """Rebuild one persisted record; every invariant is checked again."""

    return CompletionVerifierDispatch.model_validate(value)


def completion_verifier_dispatch_document(value: CompletionVerifierDispatch) -> dict[str, object]:
    return copy_completion_verifier_dispatch(value).model_dump(mode="json", warnings=False)


def _settlement_request(
    value: CompletionVerifierDispatchSettlement,
) -> CompletionVerifierDispatchSettlementRequest:
    return CompletionVerifierDispatchSettlementRequest.model_validate(
        {
            name: getattr(value, name)
            for name in CompletionVerifierDispatchSettlementRequest.model_fields
        }
    )


def _require_settlement_target(
    request: CompletionVerifierDispatchRequest,
    settlement: CompletionVerifierDispatchSettlementRequest,
) -> None:
    usage = settlement.usage
    if usage is not None and (
        usage.provider_name != request.pricing_provider_name
        or usage.requested_model != request.model
    ):
        raise ValueError("Dispatch usage conflicts with its provider target.")


def completion_verifier_dispatch_consumption(
    dispatches: tuple[CompletionVerifierDispatch, ...],
) -> tuple[int, int, int]:
    """Return attempts and conservatively counted input and output tokens."""

    input_tokens = 0
    output_tokens = 0
    for dispatch in dispatches:
        settlement = dispatch.settlement
        usage = None if settlement is None else settlement.usage
        if usage is None:
            input_tokens += dispatch.request.max_input_tokens
            output_tokens += dispatch.request.max_output_tokens
        else:
            input_tokens += usage.input_tokens
            output_tokens += usage.output_tokens
    return len(dispatches), input_tokens, output_tokens


def require_completion_verifier_dispatch_admission(
    request: CompletionVerifierDispatchRequest,
    *,
    proposal: CompletionProposal,
    contract_verifier: CompletionVerifierRef,
    claim: CompletionVerificationClaim | None,
    profile_fingerprint: str | None,
    decided: bool,
    existing: tuple[CompletionVerifierDispatch, ...],
    lease_now: datetime,
) -> None:
    """Validate a new dispatch intent against current durable authority.

    Stores call this inside the same atomic boundary that reads the claim,
    profile, decision index and prior dispatches, and that inserts the record.
    """

    if request.proposal_id != proposal.proposal_id:
        raise WorkCompletionConflict("Dispatch intent belongs to another proposal.")
    if request.verifier != contract_verifier:
        raise WorkCompletionConflict(
            "Dispatch intent uses a verifier other than the frozen contract verifier."
        )
    if profile_fingerprint is None or request.verifier_profile_fingerprint != profile_fingerprint:
        raise WorkCompletionConflict(
            "Dispatch intent requires the exact prepared verifier profile."
        )
    if decided:
        raise WorkCompletionConflict("Completion proposal already has a durable decision.")
    if (
        claim is None
        or claim.claim_id != request.claim_id
        or claim.worker_id != request.worker_id
        or claim.execution_owner_id != request.execution_owner_id
        or claim.attempt_number != request.claim_attempt_number
        or claim.verifier != request.verifier
        or claim.verifier_profile_fingerprint != request.verifier_profile_fingerprint
        or claim.lease_expires_at <= lease_now
    ):
        raise CompletionVerificationClaimLost(
            "Provider verifier dispatch requires the current live verifier claim."
        )
    for prior in existing:
        if (
            prior.request.verifier_profile_fingerprint != request.verifier_profile_fingerprint
            or prior.request.budget != request.budget
        ):
            raise WorkCompletionConflict(
                "Provider verifier dispatches for one proposal must share one profile and budget."
            )
    same_execution = tuple(
        prior
        for prior in existing
        if prior.request.claim_id == request.claim_id
        and prior.request.claim_attempt_number == request.claim_attempt_number
    )
    if request.provider_attempt != len(same_execution) + 1:
        raise WorkCompletionConflict(
            "Provider verifier attempts must be recorded in order within one execution."
        )
    if any(prior.settlement is None for prior in same_execution):
        raise WorkCompletionConflict(
            "A provider verifier attempt cannot start before the previous one settles."
        )
    attempts, input_tokens, output_tokens = completion_verifier_dispatch_consumption(existing)
    budget = request.budget
    if attempts >= budget.max_attempts:
        raise CompletionVerifierDispatchBudgetExhausted(
            "The provider verifier attempt budget for this proposal is exhausted."
        )
    if (
        budget.max_input_tokens is not None
        and input_tokens + request.max_input_tokens > budget.max_input_tokens
    ) or (
        budget.max_output_tokens is not None
        and output_tokens + request.max_output_tokens > budget.max_output_tokens
    ):
        raise CompletionVerifierDispatchBudgetExhausted(
            "The provider verifier token budget for this proposal is exhausted."
        )


def completion_verifier_dispatch_from_request(
    request: CompletionVerifierDispatchRequest,
    *,
    proposal: CompletionProposal,
    ordinal: int,
    dispatched_at: datetime,
) -> CompletionVerifierDispatch:
    return CompletionVerifierDispatch(
        dispatch_id=request.dispatch_id,
        proposal_id=proposal.proposal_id,
        task_id=proposal.task_id,
        attempt_id=proposal.attempt_id,
        contract=proposal.contract,
        ordinal=ordinal,
        request=request,
        request_sha256=completion_verifier_dispatch_request_sha256(request),
        dispatched_at=dispatched_at,
    )


def replay_completion_verifier_dispatch(
    existing: CompletionVerifierDispatch,
    request: CompletionVerifierDispatchRequest,
) -> CompletionVerifierDispatch:
    """Return an exact replay of one recorded intent or raise a conflict."""

    if existing.request_sha256 != completion_verifier_dispatch_request_sha256(request):
        raise WorkCompletionConflict("Dispatch identity is already bound to another intent.")
    return copy_completion_verifier_dispatch(existing)


def settle_completion_verifier_dispatch_record(
    existing: CompletionVerifierDispatch,
    request: CompletionVerifierDispatchSettlementRequest,
    *,
    settled_at: datetime,
) -> tuple[CompletionVerifierDispatch, bool]:
    """Apply one write-once settlement and report whether it changed the record.

    Settlement is accounting, not decision authority, so it does not require
    the claim to still be live: usage observed by a stale owner is retained.
    """

    request = copy_completion_verifier_dispatch_settlement_request(request)
    if request.dispatch_id != existing.dispatch_id:
        raise WorkCompletionConflict("Dispatch settlement belongs to another dispatch.")
    request_sha256 = completion_verifier_dispatch_settlement_sha256(request)
    if existing.settlement is not None:
        if existing.settlement.request_sha256 != request_sha256:
            raise WorkCompletionConflict("Dispatch is already settled with another outcome.")
        return copy_completion_verifier_dispatch(existing), False
    _require_settlement_target(existing.request, request)
    if request.decision is not None:
        copy_completion_verifier_decision(request.decision)
    settled_at = max(normalize_utc_datetime(settled_at, "settled_at"), existing.dispatched_at)
    settlement = CompletionVerifierDispatchSettlement(
        **request.model_dump(mode="python", warnings=False),
        request_sha256=request_sha256,
        settled_at=settled_at,
    )
    updated = CompletionVerifierDispatch.model_validate(
        {
            **existing.model_dump(mode="python", warnings=False),
            "settlement": settlement,
        }
    )
    return updated, True


__all__ = [
    "COMPLETION_VERIFIER_DISPATCH_MAX_ATTEMPTS",
    "COMPLETION_VERIFIER_DISPATCH_MAX_ATTEMPTS_PER_EXECUTION",
    "COMPLETION_VERIFIER_DISPATCH_SCHEMA_VERSION",
    "CompletionVerifierDecodeStatus",
    "CompletionVerifierDispatch",
    "CompletionVerifierDispatchBudget",
    "CompletionVerifierDispatchBudgetExhausted",
    "CompletionVerifierDispatchFailure",
    "CompletionVerifierDispatchOutcome",
    "CompletionVerifierDispatchRequest",
    "CompletionVerifierDispatchSettlement",
    "CompletionVerifierDispatchSettlementRequest",
    "CompletionVerifierUsageStatus",
    "completion_verifier_dispatch_consumption",
    "completion_verifier_dispatch_document",
    "completion_verifier_dispatch_from_document",
    "completion_verifier_dispatch_from_request",
    "completion_verifier_dispatch_id",
    "completion_verifier_dispatch_request_sha256",
    "completion_verifier_dispatch_settlement_sha256",
    "copy_completion_verifier_dispatch",
    "copy_completion_verifier_dispatch_request",
    "copy_completion_verifier_dispatch_settlement_request",
    "replay_completion_verifier_dispatch",
    "require_completion_verifier_dispatch_admission",
    "settle_completion_verifier_dispatch_record",
]
