"""Durable run ledger and receipts for independent completion evaluations.

An evaluation is an effectful, non-deterministic run of an application-owned
evaluator (a benchmark, test suite, canary or external grader) that is
independent of the agent being evaluated. Cayu persists each run's intent under
the live verification claim before the effect, settles it exactly once, and
hands the completed run to the verifier as an immutable evaluation receipt.
Records live beside the verified-work authority, apart from worker sessions and
provider-verifier dispatches.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from hashlib import sha256
from typing import Literal

from pydantic import Field, StrictInt, field_validator, model_validator

from cayu._clock import normalize_utc_datetime
from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    canonical_durable_json_bytes,
    copy_durable_json_object,
    require_durable_clean_nonblank,
    require_durable_nonblank,
    revalidate_model_input,
)
from cayu.tasks.contracts import (
    WORK_CONTRACT_IDENTIFIER_MAX_BYTES,
    WORK_EVALUATION_MAX_RUNS,
    CompletionEvaluationPolicy,
    CompletionEvaluatorRef,
    CompletionProposal,
    CompletionVerificationClaim,
    CompletionVerificationClaimLost,
    FrozenWorkContractModel,
    WorkCompletionConflict,
    WorkContractRef,
    preflight_work_completion_document,
    validate_work_completion_linked_id,
)

COMPLETION_EVALUATION_SCHEMA_VERSION = 1
COMPLETION_EVALUATION_EVIDENCE_MAX_BYTES = 256 * 1024
COMPLETION_EVALUATION_EVIDENCE_MAX_ITEMS = 8_192
COMPLETION_EVALUATION_USAGE_MAX_BYTES = 16 * 1024
COMPLETION_EVALUATION_USAGE_MAX_ITEMS = 256
COMPLETION_EVALUATION_SUMMARY_MAX_BYTES = 4 * 1024

_CODE_CHARACTERS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789._:-")


class CompletionEvaluationBudgetExhausted(ValueError):
    """Another evaluation run would exceed the contract's evaluator-run budget."""


class CompletionEvaluationOutcome(StrEnum):
    """Runtime-observed outcome of one evaluator run."""

    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    OUTCOME_UNKNOWN = "outcome_unknown"


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
        or any(character not in _CODE_CHARACTERS for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase canonical code.")
    return value


def _bounded_document(
    value: object, *, field_name: str, max_bytes: int, max_items: int
) -> dict[str, object]:
    preflight_work_completion_document(value, field_name, max_bytes=max_bytes, max_items=max_items)
    copied = copy_durable_json_object(value, field_name)
    if len(canonical_durable_json_bytes(copied, field_name)) > max_bytes:
        raise ValueError(f"{field_name} must not exceed {max_bytes} bytes.")
    return copied


def completion_evaluation_evidence_sha256(value: dict[str, object]) -> str:
    copied = copy_durable_json_object(value, "evidence")
    return sha256(
        canonical_durable_json_bytes(copied, "completion_evaluation_evidence")
    ).hexdigest()


def completion_evaluation_effect_id(
    *,
    proposal_id: str,
    evaluator: CompletionEvaluatorRef,
    run_ordinal: int,
) -> str:
    """Stable effect identity for one evaluator run, shared across claims.

    A recovering owner reconciles the same effect rather than inventing a new one.
    """

    digest = sha256(
        canonical_durable_json_bytes(
            {
                "domain": "cayu.completion-evaluation.v1",
                "proposal_id": proposal_id,
                "evaluator": evaluator.model_dump(mode="json", warnings=False),
                "run_ordinal": run_ordinal,
            },
            "completion_evaluation_effect_id",
        )
    ).hexdigest()
    return f"cveval_{digest[:48]}"


class CompletionEvaluationRunRequest(FrozenWorkContractModel):
    """Runtime-authored intent to perform one evaluator effect under a live claim."""

    effect_id: str
    proposal_id: str
    claim_id: str
    worker_id: str
    execution_owner_id: str
    claim_attempt_number: StrictInt = Field(ge=1)
    policy: CompletionEvaluationPolicy
    evaluator_profile_fingerprint: str
    run_ordinal: StrictInt = Field(ge=1, le=WORK_EVALUATION_MAX_RUNS)

    @field_validator("effect_id", "proposal_id", "claim_id", "worker_id", "execution_owner_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _identifier(value, info.field_name)

    @field_validator("policy", mode="before")
    @classmethod
    def copy_policy(cls, value: object) -> object:
        return revalidate_model_input(value, CompletionEvaluationPolicy)

    @field_validator("evaluator_profile_fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str) -> str:
        return _digest(value, "evaluator_profile_fingerprint")

    @model_validator(mode="after")
    def validate_run(self) -> CompletionEvaluationRunRequest:
        if self.run_ordinal > self.policy.max_runs:
            raise ValueError("run_ordinal exceeds the evaluation run budget.")
        if self.effect_id != completion_evaluation_effect_id(
            proposal_id=self.proposal_id,
            evaluator=self.policy.evaluator,
            run_ordinal=self.run_ordinal,
        ):
            raise ValueError("effect_id conflicts with its runtime-derived identity.")
        return self


class CompletionEvaluationFailure(FrozenWorkContractModel):
    """Bounded, content-free classification of an unsuccessful run."""

    code: str

    @field_validator("code")
    @classmethod
    def validate_code(cls, value: str) -> str:
        return _code(value, "code")


class CompletionEvaluationSettlementRequest(FrozenWorkContractModel):
    """Runtime-observed terminal outcome of one evaluator run.

    ``reported_usage`` is evaluator-reported accounting (for example run cost
    or task counts). It is recorded as reported, never as provider-observed.
    ``reconciled`` marks an outcome learned by reconciling an earlier owner's
    effect rather than by observing the run directly.
    """

    effect_id: str
    outcome: CompletionEvaluationOutcome
    latency_ms: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    reconciled: bool = False
    failure: CompletionEvaluationFailure | None = None
    evidence: dict[str, object] | None = None
    evidence_sha256: str | None = None
    summary: str | None = None
    reported_usage: dict[str, object] | None = None

    @field_validator("effect_id")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        return _identifier(value, "effect_id")

    @field_validator("failure", mode="before")
    @classmethod
    def copy_failure(cls, value: object) -> object:
        if value is None:
            return None
        return revalidate_model_input(value, CompletionEvaluationFailure)

    @field_validator("evidence", mode="before")
    @classmethod
    def copy_evidence(cls, value: object) -> object:
        if value is None:
            return None
        return _bounded_document(
            value,
            field_name="evidence",
            max_bytes=COMPLETION_EVALUATION_EVIDENCE_MAX_BYTES,
            max_items=COMPLETION_EVALUATION_EVIDENCE_MAX_ITEMS,
        )

    @field_validator("reported_usage", mode="before")
    @classmethod
    def copy_reported_usage(cls, value: object) -> object:
        if value is None:
            return None
        return _bounded_document(
            value,
            field_name="reported_usage",
            max_bytes=COMPLETION_EVALUATION_USAGE_MAX_BYTES,
            max_items=COMPLETION_EVALUATION_USAGE_MAX_ITEMS,
        )

    @field_validator("evidence_sha256")
    @classmethod
    def validate_evidence_sha256(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _digest(value, "evidence_sha256")

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = require_durable_nonblank(value, "summary")
        if len(value.encode("utf-8")) > COMPLETION_EVALUATION_SUMMARY_MAX_BYTES:
            raise ValueError(
                f"summary must not exceed {COMPLETION_EVALUATION_SUMMARY_MAX_BYTES} bytes."
            )
        return value

    @model_validator(mode="after")
    def validate_settlement(self) -> CompletionEvaluationSettlementRequest:
        completed = self.outcome is CompletionEvaluationOutcome.COMPLETED
        if completed != (self.evidence is not None) or completed == (self.failure is not None):
            raise ValueError(
                "Completed runs carry evidence and no failure; other runs carry a failure only."
            )
        if (self.evidence is None) != (self.evidence_sha256 is None):
            raise ValueError("Evaluation evidence requires its digest.")
        if self.evidence is not None and self.evidence_sha256 != (
            completion_evaluation_evidence_sha256(self.evidence)
        ):
            raise ValueError("Evaluation evidence conflicts with its digest.")
        if not completed and self.summary is not None:
            raise ValueError("Only completed runs carry a summary.")
        return self


class CompletionEvaluationSettlement(CompletionEvaluationSettlementRequest):
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


class CompletionEvaluationRun(FrozenWorkContractModel):
    """One durable evaluator run and, once observed, its settlement."""

    schema_version: Literal[1] = COMPLETION_EVALUATION_SCHEMA_VERSION
    effect_id: str
    proposal_id: str
    task_id: str
    attempt_id: str
    contract: WorkContractRef
    request: CompletionEvaluationRunRequest
    request_sha256: str
    started_at: datetime
    settlement: CompletionEvaluationSettlement | None = None

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1.")
        return value

    @field_validator("effect_id", "proposal_id", "attempt_id")
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
        return revalidate_model_input(value, CompletionEvaluationRunRequest)

    @field_validator("settlement", mode="before")
    @classmethod
    def copy_settlement(cls, value: object) -> object:
        if value is None:
            return None
        return revalidate_model_input(value, CompletionEvaluationSettlement)

    @field_validator("request_sha256")
    @classmethod
    def validate_request_sha256(cls, value: str) -> str:
        return _digest(value, "request_sha256")

    @field_validator("started_at")
    @classmethod
    def normalize_started_at(cls, value: datetime) -> datetime:
        return normalize_utc_datetime(value, "started_at")

    @property
    def run_ordinal(self) -> int:
        return self.request.run_ordinal

    @model_validator(mode="after")
    def validate_record(self) -> CompletionEvaluationRun:
        if self.request.effect_id != self.effect_id or self.request.proposal_id != self.proposal_id:
            raise ValueError("Evaluation run conflicts with its request identity.")
        if self.request_sha256 != completion_evaluation_run_request_sha256(self.request):
            raise ValueError("Evaluation run conflicts with its request digest.")
        settlement = self.settlement
        if settlement is not None:
            if settlement.effect_id != self.effect_id:
                raise ValueError("Evaluation settlement belongs to another run.")
            if settlement.request_sha256 != completion_evaluation_settlement_sha256(
                _settlement_request(settlement)
            ):
                raise ValueError("Evaluation settlement conflicts with its digest.")
            if settlement.settled_at < self.started_at:
                raise ValueError("Evaluation settlement cannot precede its run.")
        return self


class CompletionEvaluationReceipt(FrozenWorkContractModel):
    """Immutable evidence from one completed, independent evaluator run.

    Verifiers treat a receipt as trusted evidence recorded by the runtime; the
    worker's proposal and evidence remain claims to evaluate.
    """

    effect_id: str
    proposal_id: str
    task_id: str
    attempt_id: str
    contract: WorkContractRef
    evaluator: CompletionEvaluatorRef
    evaluator_profile_fingerprint: str
    run_ordinal: StrictInt = Field(ge=1, le=WORK_EVALUATION_MAX_RUNS)
    reconciled: bool
    evidence: dict[str, object]
    evidence_sha256: str
    summary: str | None = None
    started_at: datetime
    settled_at: datetime

    @field_validator("effect_id", "proposal_id", "attempt_id")
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

    @field_validator("evaluator", mode="before")
    @classmethod
    def copy_evaluator(cls, value: object) -> object:
        return revalidate_model_input(value, CompletionEvaluatorRef)

    @field_validator("evidence", mode="before")
    @classmethod
    def copy_evidence(cls, value: object) -> object:
        return _bounded_document(
            value,
            field_name="evidence",
            max_bytes=COMPLETION_EVALUATION_EVIDENCE_MAX_BYTES,
            max_items=COMPLETION_EVALUATION_EVIDENCE_MAX_ITEMS,
        )

    @field_validator("evaluator_profile_fingerprint", "evidence_sha256")
    @classmethod
    def validate_digest(cls, value: str, info) -> str:
        return _digest(value, info.field_name)

    @field_validator("started_at", "settled_at")
    @classmethod
    def normalize_timestamps(cls, value: datetime, info) -> datetime:
        return normalize_utc_datetime(value, info.field_name)

    @model_validator(mode="after")
    def validate_receipt(self) -> CompletionEvaluationReceipt:
        if self.evidence_sha256 != completion_evaluation_evidence_sha256(self.evidence):
            raise ValueError("Evaluation receipt evidence conflicts with its digest.")
        return self


def completion_evaluation_receipt(run: CompletionEvaluationRun) -> CompletionEvaluationReceipt:
    """Return the receipt for a completed run, or raise for any other run."""

    settlement = run.settlement
    if (
        settlement is None
        or settlement.outcome is not CompletionEvaluationOutcome.COMPLETED
        or settlement.evidence is None
        or settlement.evidence_sha256 is None
    ):
        raise WorkCompletionConflict("Only a completed evaluation run has a receipt.")
    return CompletionEvaluationReceipt(
        effect_id=run.effect_id,
        proposal_id=run.proposal_id,
        task_id=run.task_id,
        attempt_id=run.attempt_id,
        contract=run.contract,
        evaluator=run.request.policy.evaluator,
        evaluator_profile_fingerprint=run.request.evaluator_profile_fingerprint,
        run_ordinal=run.request.run_ordinal,
        reconciled=settlement.reconciled,
        evidence=settlement.evidence,
        evidence_sha256=settlement.evidence_sha256,
        summary=settlement.summary,
        started_at=run.started_at,
        settled_at=settlement.settled_at,
    )


def completion_evaluation_run_request_sha256(value: CompletionEvaluationRunRequest) -> str:
    copied = copy_completion_evaluation_run_request(value)
    return sha256(
        canonical_durable_json_bytes(
            copied.model_dump(mode="json", warnings=False), "completion_evaluation_run"
        )
    ).hexdigest()


def completion_evaluation_settlement_sha256(
    value: CompletionEvaluationSettlementRequest,
) -> str:
    copied = copy_completion_evaluation_settlement_request(value)
    return sha256(
        canonical_durable_json_bytes(
            copied.model_dump(mode="json", warnings=False), "completion_evaluation_settlement"
        )
    ).hexdigest()


def copy_completion_evaluation_run_request(
    value: CompletionEvaluationRunRequest,
) -> CompletionEvaluationRunRequest:
    if type(value) is not CompletionEvaluationRunRequest:
        raise TypeError("Evaluation intent must be a CompletionEvaluationRunRequest.")
    return CompletionEvaluationRunRequest.model_validate(
        value.model_dump(mode="python", warnings=False)
    )


def copy_completion_evaluation_settlement_request(
    value: CompletionEvaluationSettlementRequest,
) -> CompletionEvaluationSettlementRequest:
    if type(value) is not CompletionEvaluationSettlementRequest:
        raise TypeError("Evaluation settlement must be a CompletionEvaluationSettlementRequest.")
    return CompletionEvaluationSettlementRequest.model_validate(
        value.model_dump(mode="python", warnings=False)
    )


def copy_completion_evaluation_run(value: CompletionEvaluationRun) -> CompletionEvaluationRun:
    if type(value) is not CompletionEvaluationRun:
        raise TypeError("Evaluation run must be a CompletionEvaluationRun.")
    return CompletionEvaluationRun.model_validate(value.model_dump(mode="python", warnings=False))


def completion_evaluation_run_from_document(value: object) -> CompletionEvaluationRun:
    return CompletionEvaluationRun.model_validate(value)


def completion_evaluation_run_document(value: CompletionEvaluationRun) -> dict[str, object]:
    return copy_completion_evaluation_run(value).model_dump(mode="json", warnings=False)


def _settlement_request(
    value: CompletionEvaluationSettlement,
) -> CompletionEvaluationSettlementRequest:
    return CompletionEvaluationSettlementRequest.model_validate(
        {name: getattr(value, name) for name in CompletionEvaluationSettlementRequest.model_fields}
    )


def require_completion_evaluation_admission(
    request: CompletionEvaluationRunRequest,
    *,
    proposal: CompletionProposal,
    contract_evaluation: CompletionEvaluationPolicy | None,
    claim: CompletionVerificationClaim | None,
    decided: bool,
    existing: tuple[CompletionEvaluationRun, ...],
    lease_now: datetime,
) -> None:
    """Validate a new evaluation intent against current durable authority.

    Stores call this inside the same atomic boundary that reads the claim,
    decision index and prior runs, and that inserts the record.
    """

    if request.proposal_id != proposal.proposal_id:
        raise WorkCompletionConflict("Evaluation intent belongs to another proposal.")
    if contract_evaluation is None or request.policy != contract_evaluation:
        raise WorkCompletionConflict(
            "Evaluation intent conflicts with the frozen contract evaluation policy."
        )
    if decided:
        raise WorkCompletionConflict("Completion proposal already has a durable decision.")
    if (
        claim is None
        or claim.claim_id != request.claim_id
        or claim.worker_id != request.worker_id
        or claim.execution_owner_id != request.execution_owner_id
        or claim.attempt_number != request.claim_attempt_number
        or claim.lease_expires_at <= lease_now
    ):
        raise CompletionVerificationClaimLost(
            "Completion evaluation requires the current live verifier claim."
        )
    for prior in existing:
        if prior.request.evaluator_profile_fingerprint != request.evaluator_profile_fingerprint:
            raise WorkCompletionConflict(
                "Evaluation runs for one proposal must use one evaluator profile."
            )
        settlement = prior.settlement
        if settlement is None:
            raise WorkCompletionConflict(
                "An evaluation run cannot start before the previous one settles."
            )
        if settlement.outcome is CompletionEvaluationOutcome.COMPLETED:
            raise WorkCompletionConflict("Completion proposal already has an evaluation receipt.")
    if len(existing) >= request.policy.max_runs:
        raise CompletionEvaluationBudgetExhausted(
            "The evaluation run budget for this proposal is exhausted."
        )
    if request.run_ordinal != len(existing) + 1:
        raise WorkCompletionConflict("Evaluation runs must be recorded in order.")


def completion_evaluation_run_from_request(
    request: CompletionEvaluationRunRequest,
    *,
    proposal: CompletionProposal,
    started_at: datetime,
) -> CompletionEvaluationRun:
    return CompletionEvaluationRun(
        effect_id=request.effect_id,
        proposal_id=proposal.proposal_id,
        task_id=proposal.task_id,
        attempt_id=proposal.attempt_id,
        contract=proposal.contract,
        request=request,
        request_sha256=completion_evaluation_run_request_sha256(request),
        started_at=started_at,
    )


def replay_completion_evaluation_run(
    existing: CompletionEvaluationRun,
    request: CompletionEvaluationRunRequest,
) -> CompletionEvaluationRun:
    if existing.request_sha256 != completion_evaluation_run_request_sha256(request):
        raise WorkCompletionConflict("Evaluation effect is already bound to another intent.")
    return copy_completion_evaluation_run(existing)


def settle_completion_evaluation_run_record(
    existing: CompletionEvaluationRun,
    request: CompletionEvaluationSettlementRequest,
    *,
    settled_at: datetime,
) -> tuple[CompletionEvaluationRun, bool]:
    """Apply one write-once settlement and report whether it changed the record."""

    request = copy_completion_evaluation_settlement_request(request)
    if request.effect_id != existing.effect_id:
        raise WorkCompletionConflict("Evaluation settlement belongs to another run.")
    request_sha256 = completion_evaluation_settlement_sha256(request)
    if existing.settlement is not None:
        if existing.settlement.request_sha256 != request_sha256:
            raise WorkCompletionConflict("Evaluation run is already settled with another outcome.")
        return copy_completion_evaluation_run(existing), False
    settled_at = max(normalize_utc_datetime(settled_at, "settled_at"), existing.started_at)
    settlement = CompletionEvaluationSettlement(
        **request.model_dump(mode="python", warnings=False),
        request_sha256=request_sha256,
        settled_at=settled_at,
    )
    updated = CompletionEvaluationRun.model_validate(
        {**existing.model_dump(mode="python", warnings=False), "settlement": settlement}
    )
    return updated, True


__all__ = [
    "COMPLETION_EVALUATION_EVIDENCE_MAX_BYTES",
    "COMPLETION_EVALUATION_SCHEMA_VERSION",
    "CompletionEvaluationBudgetExhausted",
    "CompletionEvaluationFailure",
    "CompletionEvaluationOutcome",
    "CompletionEvaluationReceipt",
    "CompletionEvaluationRun",
    "CompletionEvaluationRunRequest",
    "CompletionEvaluationSettlement",
    "CompletionEvaluationSettlementRequest",
    "completion_evaluation_effect_id",
    "completion_evaluation_evidence_sha256",
    "completion_evaluation_receipt",
    "completion_evaluation_run_document",
    "completion_evaluation_run_from_document",
    "completion_evaluation_run_from_request",
    "completion_evaluation_run_request_sha256",
    "completion_evaluation_settlement_sha256",
    "copy_completion_evaluation_run",
    "copy_completion_evaluation_run_request",
    "copy_completion_evaluation_settlement_request",
    "replay_completion_evaluation_run",
    "require_completion_evaluation_admission",
    "settle_completion_evaluation_run_record",
]
