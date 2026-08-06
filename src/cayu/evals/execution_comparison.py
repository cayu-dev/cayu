from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, StrictBool, StrictStr, field_validator, model_validator

from cayu.evals.corpus import _sha256_revision
from cayu.evals.execution import CorpusExecutionResult


class CorpusComparisonReason(StrEnum):
    """Stable reason that two published executions cannot be compared as one contract."""

    TARGET_KEY_MISMATCH = "target_key_mismatch"
    CORPUS_REVISION_MISMATCH = "corpus_revision_mismatch"
    SUITE_ID_MISMATCH = "suite_id_mismatch"
    SUITE_REVISION_MISMATCH = "suite_revision_mismatch"
    EVIDENCE_POLICY_REVISION_MISMATCH = "evidence_policy_revision_mismatch"
    PRICING_PROFILE_FINGERPRINT_MISMATCH = "pricing_profile_fingerprint_mismatch"
    CASE_CONTRACT_MISMATCH = "case_contract_mismatch"
    ASSERTION_CONTRACT_MISMATCH = "assertion_contract_mismatch"


_REASON_ORDER = tuple(CorpusComparisonReason)


class CorpusComparisonCompatibility(BaseModel):
    """Typed precondition result for later regression comparison and UI adapters."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    schema_version: Literal[1] = 1
    baseline_result_revision: StrictStr
    current_result_revision: StrictStr
    comparable: StrictBool
    reasons: tuple[CorpusComparisonReason, ...] = ()

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1.")
        return value

    @field_validator("baseline_result_revision", "current_result_revision")
    @classmethod
    def validate_revisions(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("reasons", mode="before")
    @classmethod
    def validate_reason_sequence(cls, value: object) -> object:
        if not isinstance(value, list | tuple):
            raise ValueError("reasons must be an ordered array.")
        return value

    @model_validator(mode="after")
    def validate_contract(self) -> CorpusComparisonCompatibility:
        if self.reasons != tuple(reason for reason in _REASON_ORDER if reason in self.reasons):
            raise ValueError("Comparison reasons must be unique and in canonical order.")
        if self.comparable == bool(self.reasons):
            raise ValueError("Comparison comparable state contradicts its reasons.")
        return self


def _validated_execution(result: CorpusExecutionResult, field_name: str) -> CorpusExecutionResult:
    if type(result) is not CorpusExecutionResult:
        raise TypeError(f"{field_name} must be an exact CorpusExecutionResult.")
    return CorpusExecutionResult.model_validate(
        result.model_dump(mode="python", round_trip=True, warnings="none")
    )


def _case_contract(result: CorpusExecutionResult) -> tuple[tuple[str, str], ...]:
    return tuple((case.case_id, case.case_revision) for case in result.run.cases)


def _assertion_contract(
    result: CorpusExecutionResult,
) -> tuple[tuple[str, tuple[tuple[str, str], ...]], ...]:
    return tuple(
        (
            case.case_id,
            tuple(
                (assertion.assertion_id, assertion.assertion_revision)
                for assertion in case.trials[0].assertions
            ),
        )
        for case in result.run.cases
    )


def _uses_pricing(result: CorpusExecutionResult) -> bool:
    return any(
        assertion.detail.kind == "max_estimated_cost"
        for case in result.run.cases
        for trial in case.trials
        for assertion in trial.assertions
    )


def corpus_execution_compatibility(
    baseline: CorpusExecutionResult,
    current: CorpusExecutionResult,
) -> CorpusComparisonCompatibility:
    """Check evaluation-contract compatibility without comparing target releases."""

    baseline = _validated_execution(baseline, "baseline")
    current = _validated_execution(current, "current")
    reasons: set[CorpusComparisonReason] = set()
    if baseline.run.target_key != current.run.target_key:
        reasons.add(CorpusComparisonReason.TARGET_KEY_MISMATCH)
    if baseline.run.corpus_revision != current.run.corpus_revision:
        reasons.add(CorpusComparisonReason.CORPUS_REVISION_MISMATCH)
    if baseline.run.suite_id != current.run.suite_id:
        reasons.add(CorpusComparisonReason.SUITE_ID_MISMATCH)
    if baseline.run.suite_revision != current.run.suite_revision:
        reasons.add(CorpusComparisonReason.SUITE_REVISION_MISMATCH)
    if baseline.run.evidence_policy_revision != current.run.evidence_policy_revision:
        reasons.add(CorpusComparisonReason.EVIDENCE_POLICY_REVISION_MISMATCH)
    if (_uses_pricing(baseline) or _uses_pricing(current)) and (
        baseline.run.pricing_profile_fingerprint != current.run.pricing_profile_fingerprint
    ):
        reasons.add(CorpusComparisonReason.PRICING_PROFILE_FINGERPRINT_MISMATCH)
    if _case_contract(baseline) != _case_contract(current):
        reasons.add(CorpusComparisonReason.CASE_CONTRACT_MISMATCH)
    if _assertion_contract(baseline) != _assertion_contract(current):
        reasons.add(CorpusComparisonReason.ASSERTION_CONTRACT_MISMATCH)
    ordered_reasons = tuple(reason for reason in _REASON_ORDER if reason in reasons)
    return CorpusComparisonCompatibility(
        baseline_result_revision=baseline.revision,
        current_result_revision=current.revision,
        comparable=not ordered_reasons,
        reasons=ordered_reasons,
    )
