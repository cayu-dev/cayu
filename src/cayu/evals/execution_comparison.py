from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictStr,
    field_validator,
    model_validator,
)

from cayu.evals.corpus import EVAL_CORPUS_MAX_TRIALS, _sha256_revision
from cayu.evals.execution import CorpusExecutionResult
from cayu.evals.published import PublishedStatus
from cayu.evals.result_contract import EvalTrialDiagnosticCode
from cayu.evals.results import (
    CapturedEvaluationResultV1,
    EvalResultProjectionV1,
    eval_result_projection,
)


class CorpusComparisonReason(StrEnum):
    """Stable reason that two published executions cannot be compared as one contract."""

    TARGET_KEY_MISMATCH = "target_key_mismatch"
    EXTERNAL_TARGET_REVISION_MISMATCH = "external_target_revision_mismatch"
    CORPUS_REVISION_MISMATCH = "corpus_revision_mismatch"
    SUITE_ID_MISMATCH = "suite_id_mismatch"
    SUITE_REVISION_MISMATCH = "suite_revision_mismatch"
    EVIDENCE_POLICY_REVISION_MISMATCH = "evidence_policy_revision_mismatch"
    PRICING_PROFILE_FINGERPRINT_MISMATCH = "pricing_profile_fingerprint_mismatch"
    CASE_CONTRACT_MISMATCH = "case_contract_mismatch"
    ASSERTION_CONTRACT_MISMATCH = "assertion_contract_mismatch"


_REASON_ORDER = tuple(CorpusComparisonReason)


class CorpusRegressionScope(StrEnum):
    """Stable location of one compatible-result regression."""

    RUN = "run"
    CASE = "case"


class CorpusRegressionKind(StrEnum):
    """Stable dimension on which one compatible result regressed."""

    STATUS = "status"
    SCORE = "score"


_STATUS_SEVERITY: dict[PublishedStatus, int] = {
    "passed": 0,
    "failed": 1,
    "unavailable": 2,
    "error": 3,
}


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


class CorpusComparisonResultSummary(BaseModel):
    """Bounded public identity and aggregate outcome for one compared result."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    result_revision: StrictStr
    application_release_id: StrictStr = Field(min_length=1, max_length=256)
    app_manifest_fingerprint: StrictStr = Field(min_length=64, max_length=64)
    memory_attribution_support: Literal["unsupported"] = "unsupported"
    status: PublishedStatus
    score: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("result_revision")
    @classmethod
    def validate_result_revision(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("application_release_id")
    @classmethod
    def validate_release_id(cls, value: str) -> str:
        if value != value.strip() or not value.isprintable():
            raise ValueError("application_release_id must be clean nonblank text.")
        return value

    @field_validator("app_manifest_fingerprint")
    @classmethod
    def validate_manifest_fingerprint(cls, value: str) -> str:
        if any(character not in "0123456789abcdef" for character in value):
            raise ValueError("app_manifest_fingerprint must be a lowercase SHA-256 digest.")
        return value

    @model_validator(mode="after")
    def validate_score(self) -> CorpusComparisonResultSummary:
        if (self.status in {"unavailable", "error"}) != (self.score is None):
            raise ValueError("Comparison result status contradicts its score.")
        return self


class CorpusCaseComparison(BaseModel):
    """Aggregate outcomes for one case in two contract-compatible results."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    case_id: StrictStr = Field(min_length=1, max_length=128)
    baseline_status: PublishedStatus
    current_status: PublishedStatus
    baseline_score: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    current_score: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    baseline_trial_diagnostic_codes: tuple[EvalTrialDiagnosticCode, ...] = Field(
        default=(),
        max_length=EVAL_CORPUS_MAX_TRIALS,
        exclude_if=lambda value: not value,
    )
    current_trial_diagnostic_codes: tuple[EvalTrialDiagnosticCode, ...] = Field(
        default=(),
        max_length=EVAL_CORPUS_MAX_TRIALS,
        exclude_if=lambda value: not value,
    )

    @field_validator("case_id")
    @classmethod
    def validate_case_id(cls, value: str) -> str:
        if value != value.strip() or not value.isprintable():
            raise ValueError("case_id must be clean nonblank text.")
        return value

    @field_validator(
        "baseline_trial_diagnostic_codes",
        "current_trial_diagnostic_codes",
        mode="before",
    )
    @classmethod
    def validate_diagnostic_sequence(cls, value: object) -> object:
        if not isinstance(value, list | tuple):
            raise ValueError("Trial diagnostic codes must be an ordered array.")
        return value

    @model_validator(mode="after")
    def validate_scores(self) -> CorpusCaseComparison:
        for label, status, score in (
            ("baseline", self.baseline_status, self.baseline_score),
            ("current", self.current_status, self.current_score),
        ):
            if (status in {"unavailable", "error"}) != (score is None):
                raise ValueError(f"{label} case status contradicts its score.")
        return self


class CorpusExecutionRegression(BaseModel):
    """One typed regression derived from compatible published aggregates."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    scope: CorpusRegressionScope
    kind: CorpusRegressionKind
    case_id: StrictStr | None = Field(default=None, min_length=1, max_length=128)
    baseline_status: PublishedStatus | None = None
    current_status: PublishedStatus | None = None
    baseline_score: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    current_score: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("case_id")
    @classmethod
    def validate_case_id(cls, value: str | None) -> str | None:
        if value is not None and (value != value.strip() or not value.isprintable()):
            raise ValueError("case_id must be clean nonblank text.")
        return value

    @model_validator(mode="after")
    def validate_shape(self) -> CorpusExecutionRegression:
        if (self.scope is CorpusRegressionScope.CASE) != (self.case_id is not None):
            raise ValueError("Only case regressions carry case_id.")
        if self.kind is CorpusRegressionKind.STATUS:
            if self.baseline_status is None or self.current_status is None:
                raise ValueError("Status regressions require both statuses.")
            if self.baseline_score is not None or self.current_score is not None:
                raise ValueError("Status regressions cannot carry scores.")
            if _STATUS_SEVERITY[self.current_status] <= _STATUS_SEVERITY[self.baseline_status]:
                raise ValueError("A status regression must move to a worse status.")
        else:
            if self.baseline_score is None or self.current_score is None:
                raise ValueError("Score regressions require both scores.")
            if self.baseline_status is not None or self.current_status is not None:
                raise ValueError("Score regressions cannot carry statuses.")
            if self.current_score >= self.baseline_score:
                raise ValueError("A score regression must lower the score.")
        return self


class CorpusExecutionComparison(BaseModel):
    """Deterministic contract-aware comparison of two published corpus executions."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    schema_version: Literal[1] = 1
    compatibility: CorpusComparisonCompatibility
    score_tolerance: StrictFloat = Field(default=0.0, ge=0.0, le=1.0)
    baseline: CorpusComparisonResultSummary
    current: CorpusComparisonResultSummary
    cases: tuple[CorpusCaseComparison, ...] = Field(default_factory=tuple, max_length=1_000)
    regressions: tuple[CorpusExecutionRegression, ...] = Field(
        default_factory=tuple,
        max_length=2_002,
    )

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1.")
        return value

    @model_validator(mode="after")
    def validate_derived_contract(self) -> CorpusExecutionComparison:
        if self.compatibility.baseline_result_revision != self.baseline.result_revision:
            raise ValueError("Baseline summary does not match compatibility evidence.")
        if self.compatibility.current_result_revision != self.current.result_revision:
            raise ValueError("Current summary does not match compatibility evidence.")
        if not self.compatibility.comparable:
            if self.cases or self.regressions:
                raise ValueError("Incomparable results cannot carry outcome comparisons.")
            return self
        case_ids = tuple(case.case_id for case in self.cases)
        if not case_ids or case_ids != tuple(sorted(set(case_ids))):
            raise ValueError("Comparable case summaries must be nonempty, unique, and sorted.")
        expected = _regressions_from_summaries(
            self.baseline,
            self.current,
            self.cases,
            score_tolerance=self.score_tolerance,
        )
        if self.regressions != expected:
            raise ValueError("Comparison regressions do not match its aggregate outcomes.")
        return self


@dataclass(frozen=True)
class _CaseOutcome:
    case_id: str
    status: PublishedStatus
    score: float | None
    trial_diagnostic_codes: tuple[EvalTrialDiagnosticCode, ...]


@dataclass(frozen=True)
class _ExecutionProjection:
    result_revision: str
    target_key: str
    external_target_revision: str | None
    corpus_revision: str
    suite_id: str
    suite_revision: str
    evidence_policy_revision: str
    pricing_profile_fingerprint: str | None
    uses_pricing: bool
    case_contract: tuple[tuple[str, str], ...]
    assertion_contract: tuple[tuple[str, tuple[tuple[object, ...], ...]], ...]
    summary: CorpusComparisonResultSummary
    cases: tuple[_CaseOutcome, ...]


def _validated_projection(
    result: CorpusExecutionResult | CapturedEvaluationResultV1 | EvalResultProjectionV1,
    field_name: str,
) -> _ExecutionProjection:
    """Validate one potentially forged graph, then retain only bounded comparison data."""

    if type(result) is EvalResultProjectionV1:
        projection = EvalResultProjectionV1.model_validate(
            result.model_dump(mode="python", round_trip=True, warnings="none")
        )
    elif type(result) in {CorpusExecutionResult, CapturedEvaluationResultV1}:
        projection = eval_result_projection(
            cast("CorpusExecutionResult | CapturedEvaluationResultV1", result)
        )
    else:
        raise TypeError(
            f"{field_name} must be an exact CorpusExecutionResult, "
            "CapturedEvaluationResultV1, or EvalResultProjectionV1."
        )
    case_contract = tuple((case.case_id, case.case_revision) for case in projection.cases)
    assertion_contract = tuple(
        (
            case.case_id,
            tuple(
                (assertion.assertion_id, assertion.comparison_revision)
                for assertion in case.assertions
            ),
        )
        for case in projection.cases
    )
    summary = CorpusComparisonResultSummary(
        result_revision=projection.result_revision,
        application_release_id=projection.target.application_release_id,
        app_manifest_fingerprint=projection.target.app_manifest_fingerprint,
        memory_attribution_support=projection.memory_attribution_support,
        status=projection.status,
        score=projection.score,
    )
    cases = tuple(
        _CaseOutcome(
            case_id=case.case_id,
            status=case.status,
            score=case.score,
            trial_diagnostic_codes=case.trial_diagnostic_codes,
        )
        for case in projection.cases
    )
    return _ExecutionProjection(
        result_revision=projection.result_revision,
        target_key=projection.target.target_key,
        external_target_revision=projection.external_target_revision,
        corpus_revision=projection.corpus_revision,
        suite_id=projection.suite_id,
        suite_revision=projection.suite_revision,
        evidence_policy_revision=projection.evidence_policy_revision,
        pricing_profile_fingerprint=projection.pricing_profile_fingerprint,
        uses_pricing=projection.uses_pricing,
        case_contract=case_contract,
        assertion_contract=assertion_contract,
        summary=summary,
        cases=cases,
    )


def _compatibility_from_projections(
    baseline: _ExecutionProjection,
    current: _ExecutionProjection,
) -> CorpusComparisonCompatibility:
    reasons: set[CorpusComparisonReason] = set()
    if baseline.target_key != current.target_key:
        reasons.add(CorpusComparisonReason.TARGET_KEY_MISMATCH)
    if baseline.external_target_revision != current.external_target_revision:
        reasons.add(CorpusComparisonReason.EXTERNAL_TARGET_REVISION_MISMATCH)
    if baseline.corpus_revision != current.corpus_revision:
        reasons.add(CorpusComparisonReason.CORPUS_REVISION_MISMATCH)
    if baseline.suite_id != current.suite_id:
        reasons.add(CorpusComparisonReason.SUITE_ID_MISMATCH)
    if baseline.suite_revision != current.suite_revision:
        reasons.add(CorpusComparisonReason.SUITE_REVISION_MISMATCH)
    if baseline.evidence_policy_revision != current.evidence_policy_revision:
        reasons.add(CorpusComparisonReason.EVIDENCE_POLICY_REVISION_MISMATCH)
    if (baseline.uses_pricing or current.uses_pricing) and (
        baseline.pricing_profile_fingerprint != current.pricing_profile_fingerprint
    ):
        reasons.add(CorpusComparisonReason.PRICING_PROFILE_FINGERPRINT_MISMATCH)
    if baseline.case_contract != current.case_contract:
        reasons.add(CorpusComparisonReason.CASE_CONTRACT_MISMATCH)
    if baseline.assertion_contract != current.assertion_contract:
        reasons.add(CorpusComparisonReason.ASSERTION_CONTRACT_MISMATCH)
    ordered_reasons = tuple(reason for reason in _REASON_ORDER if reason in reasons)
    return CorpusComparisonCompatibility(
        baseline_result_revision=baseline.result_revision,
        current_result_revision=current.result_revision,
        comparable=not ordered_reasons,
        reasons=ordered_reasons,
    )


def corpus_execution_compatibility(
    baseline: CorpusExecutionResult,
    current: CorpusExecutionResult,
) -> CorpusComparisonCompatibility:
    """Check evaluation-contract compatibility without comparing target releases."""

    return _compatibility_from_projections(
        _validated_projection(baseline, "baseline"),
        _validated_projection(current, "current"),
    )


def eval_result_compatibility(
    baseline: CorpusExecutionResult | CapturedEvaluationResultV1 | EvalResultProjectionV1,
    current: CorpusExecutionResult | CapturedEvaluationResultV1 | EvalResultProjectionV1,
) -> CorpusComparisonCompatibility:
    """Check compatibility across captured and fresh immutable result origins."""

    return _compatibility_from_projections(
        _validated_projection(baseline, "baseline"),
        _validated_projection(current, "current"),
    )


def _regression_pair(
    *,
    scope: CorpusRegressionScope,
    case_id: str | None,
    baseline_status: PublishedStatus,
    current_status: PublishedStatus,
    baseline_score: float | None,
    current_score: float | None,
    score_tolerance: float,
) -> tuple[CorpusExecutionRegression, ...]:
    regressions: list[CorpusExecutionRegression] = []
    if _STATUS_SEVERITY[current_status] > _STATUS_SEVERITY[baseline_status]:
        regressions.append(
            CorpusExecutionRegression(
                scope=scope,
                kind=CorpusRegressionKind.STATUS,
                case_id=case_id,
                baseline_status=baseline_status,
                current_status=current_status,
            )
        )
    if (
        baseline_score is not None
        and current_score is not None
        and current_score < baseline_score - score_tolerance
    ):
        regressions.append(
            CorpusExecutionRegression(
                scope=scope,
                kind=CorpusRegressionKind.SCORE,
                case_id=case_id,
                baseline_score=baseline_score,
                current_score=current_score,
            )
        )
    return tuple(regressions)


def _regressions_from_summaries(
    baseline: CorpusComparisonResultSummary,
    current: CorpusComparisonResultSummary,
    cases: tuple[CorpusCaseComparison, ...],
    *,
    score_tolerance: float,
) -> tuple[CorpusExecutionRegression, ...]:
    regressions = list(
        _regression_pair(
            scope=CorpusRegressionScope.RUN,
            case_id=None,
            baseline_status=baseline.status,
            current_status=current.status,
            baseline_score=baseline.score,
            current_score=current.score,
            score_tolerance=score_tolerance,
        )
    )
    for case in cases:
        regressions.extend(
            _regression_pair(
                scope=CorpusRegressionScope.CASE,
                case_id=case.case_id,
                baseline_status=case.baseline_status,
                current_status=case.current_status,
                baseline_score=case.baseline_score,
                current_score=case.current_score,
                score_tolerance=score_tolerance,
            )
        )
    return tuple(regressions)


def _compare_projected_results(
    baseline: CorpusExecutionResult | CapturedEvaluationResultV1 | EvalResultProjectionV1,
    current: CorpusExecutionResult | CapturedEvaluationResultV1 | EvalResultProjectionV1,
    *,
    score_tolerance: float,
) -> CorpusExecutionComparison:
    if type(score_tolerance) not in (int, float) or isinstance(score_tolerance, bool):
        raise TypeError("score_tolerance must be a number.")
    if score_tolerance != score_tolerance or score_tolerance < 0 or score_tolerance > 1:
        raise ValueError("score_tolerance must be between 0 and 1.")
    baseline_projection = _validated_projection(baseline, "baseline")
    current_projection = _validated_projection(current, "current")
    compatibility = _compatibility_from_projections(baseline_projection, current_projection)
    baseline_summary = baseline_projection.summary
    current_summary = current_projection.summary
    tolerance = float(score_tolerance)
    if not compatibility.comparable:
        return CorpusExecutionComparison(
            compatibility=compatibility,
            score_tolerance=tolerance,
            baseline=baseline_summary,
            current=current_summary,
        )
    cases = tuple(
        CorpusCaseComparison(
            case_id=baseline_case.case_id,
            baseline_status=baseline_case.status,
            current_status=current_case.status,
            baseline_score=baseline_case.score,
            current_score=current_case.score,
            baseline_trial_diagnostic_codes=baseline_case.trial_diagnostic_codes,
            current_trial_diagnostic_codes=current_case.trial_diagnostic_codes,
        )
        for baseline_case, current_case in zip(
            baseline_projection.cases,
            current_projection.cases,
            strict=True,
        )
    )
    regressions = _regressions_from_summaries(
        baseline_summary,
        current_summary,
        cases,
        score_tolerance=tolerance,
    )
    return CorpusExecutionComparison(
        compatibility=compatibility,
        score_tolerance=tolerance,
        baseline=baseline_summary,
        current=current_summary,
        cases=cases,
        regressions=regressions,
    )


def compare_corpus_execution_results(
    baseline: CorpusExecutionResult,
    current: CorpusExecutionResult,
    *,
    score_tolerance: float = 0.0,
) -> CorpusExecutionComparison:
    """Compare published fresh results only when their complete contracts match."""

    if type(baseline) is not CorpusExecutionResult:
        raise TypeError("baseline must be an exact CorpusExecutionResult.")
    if type(current) is not CorpusExecutionResult:
        raise TypeError("current must be an exact CorpusExecutionResult.")
    return _compare_projected_results(
        baseline,
        current,
        score_tolerance=score_tolerance,
    )


def compare_eval_results(
    baseline: CorpusExecutionResult | CapturedEvaluationResultV1 | EvalResultProjectionV1,
    current: CorpusExecutionResult | CapturedEvaluationResultV1 | EvalResultProjectionV1,
    *,
    score_tolerance: float = 0.0,
) -> CorpusExecutionComparison:
    """Compare captured and fresh results through their shared public projection."""

    return _compare_projected_results(
        baseline,
        current,
        score_tolerance=score_tolerance,
    )
