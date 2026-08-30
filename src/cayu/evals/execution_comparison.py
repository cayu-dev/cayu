from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from cayu._validation import json_utf8_size_within_limit
from cayu.evals.corpus import (
    EVAL_CORPUS_MAX_ASSERTIONS_PER_CASE,
    EVAL_CORPUS_MAX_CASES,
    EVAL_CORPUS_MAX_JUDGE_CRITERIA,
    EVAL_CORPUS_MAX_TRIALS,
    _canonical_decimal_text,
    _exact_decimal_difference,
    _ordered_sequence_input,
    _portable_id,
    _PortableModel,
    _sha256_revision,
)
from cayu.evals.evidence import _canonical_decimal
from cayu.evals.execution import CorpusExecutionResult
from cayu.evals.json_subset import equal_json_values
from cayu.evals.published import (
    PublishedAssertionResult,
    PublishedModelJudgeDiagnostic,
    PublishedOutcome,
    PublishedStatus,
    PublishedToolArgumentsContainDetail,
    PublishedToolResultContainsDetail,
)
from cayu.evals.result_contract import EvalTrialDiagnosticCode
from cayu.evals.result_presentation import (
    EvalAssertionPresentationV1,
    EvalResultPresentationV2,
    EvalStructuredJudgePresentationV1,
    present_eval_result,
)
from cayu.evals.results import (
    CapturedEvaluationResultV1,
    EvalResultOrigin,
    EvalResultProjectionV1,
    EvalResultProjectionV2,
    eval_result_projection,
)
from cayu.evals.trial_policy import EvalSuiteTrialPolicyV1


class CorpusComparisonReason(StrEnum):
    """Stable reason that two published executions cannot be compared as one contract."""

    TARGET_KEY_MISMATCH = "target_key_mismatch"
    EXTERNAL_TARGET_REVISION_MISMATCH = "external_target_revision_mismatch"
    CORPUS_REVISION_MISMATCH = "corpus_revision_mismatch"
    SUITE_ID_MISMATCH = "suite_id_mismatch"
    SUITE_REVISION_MISMATCH = "suite_revision_mismatch"
    EVIDENCE_POLICY_REVISION_MISMATCH = "evidence_policy_revision_mismatch"
    PRICING_PROFILE_FINGERPRINT_MISMATCH = "pricing_profile_fingerprint_mismatch"
    TRIAL_POLICY_REVISION_MISMATCH = "trial_policy_revision_mismatch"
    ACCEPTED_EXPOSURE_CONTRACT_MISMATCH = "accepted_exposure_contract_mismatch"
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
    RELIABILITY = "reliability"


_STATUS_SEVERITY: dict[PublishedStatus, int] = {
    "passed": 0,
    "failed": 1,
    "unavailable": 2,
    "error": 3,
}

_EVALUATOR_SEVERITY: dict[PublishedModelJudgeDiagnostic, int] = {
    "judgment_recorded": 0,
    "evidence_unavailable": 1,
    "evaluator_error": 2,
}

EVAL_STRUCTURED_JUDGE_COMPARISON_MAX_ITEMS = (
    EVAL_CORPUS_MAX_CASES * EVAL_CORPUS_MAX_TRIALS * EVAL_CORPUS_MAX_ASSERTIONS_PER_CASE
)
CORPUS_EXECUTION_COMPARISON_MAX_BYTES = 96 << 20

EvalStructuredJudgeComparisonState = Literal[
    "compared",
    "contract_incompatible",
    "no_structured_judges",
    "observation_identity_mismatch",
    "source_detail_unavailable",
]
EvalStructuredJudgeAggregateChange = Literal[
    "improved",
    "regressed",
    "unchanged",
    "unavailable",
]
EvalStructuredJudgeEvaluatorChange = Literal["improved", "regressed", "unchanged"]
EvalToolJsonComparisonState = Literal[
    "compared",
    "contract_incompatible",
    "no_tool_json_assertions",
    "observation_identity_mismatch",
    "source_detail_unavailable",
]
EvalToolJsonObservedValueChange = Literal["changed", "unchanged", "unavailable"]
CorpusReliabilityChange = Literal["improved", "regressed", "changed", "unchanged"]


_UNAVAILABLE_TRIAL_DIAGNOSTICS = frozenset(
    {
        EvalTrialDiagnosticCode.ASSERTION_EVIDENCE_UNAVAILABLE,
        EvalTrialDiagnosticCode.TERMINAL_EVIDENCE_UNAVAILABLE,
        EvalTrialDiagnosticCode.INTERRUPTED_EVIDENCE_UNAVAILABLE,
        EvalTrialDiagnosticCode.CHILD_EVIDENCE_UNAVAILABLE,
        EvalTrialDiagnosticCode.EXTERNAL_TARGET_UNAVAILABLE,
        EvalTrialDiagnosticCode.EXTERNAL_TARGET_UNKNOWN,
        EvalTrialDiagnosticCode.EXTERNAL_TARGET_INCOMPLETE,
        EvalTrialDiagnosticCode.EXTERNAL_TARGET_IDENTITY_MISMATCH,
    }
)


def _canonical_signed_decimal(value: Decimal) -> str:
    if value == 0:
        return "0"
    if value < 0:
        return f"-{_canonical_decimal(value.copy_abs(), max_chars=64)}"
    return _canonical_decimal(value, max_chars=64)


def _validate_signed_unit_decimal(value: str, field_name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{field_name} must be a canonical decimal string.")
    negative = value.startswith("-")
    magnitude = value[1:] if negative else value
    magnitude = _canonical_decimal_text(magnitude, field_name, max_chars=64)
    if Decimal(magnitude) > 1:
        raise ValueError(f"{field_name} must be between -1 and 1.")
    canonical = f"-{magnitude}" if negative and magnitude != "0" else magnitude
    if value != canonical:
        raise ValueError(f"{field_name} must use canonical signed decimal form.")
    return value


class EvalStructuredJudgeCriterionComparisonV1(_PortableModel):
    """Exact criterion delta for one identity-matched structured judgment."""

    criterion_id: StrictStr
    weight: StrictStr
    baseline_score: StrictStr
    current_score: StrictStr
    score_delta: StrictStr
    baseline_explanation_state: Literal["available", "redacted", "unavailable"]
    current_explanation_state: Literal["available", "redacted", "unavailable"]

    @field_validator("criterion_id")
    @classmethod
    def validate_criterion_id(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("weight", "baseline_score", "current_score")
    @classmethod
    def validate_unit_decimal(cls, value: str, info) -> str:
        value = _canonical_decimal_text(value, info.field_name, max_chars=64)
        if Decimal(value) > 1:
            raise ValueError(f"{info.field_name} must be between 0 and 1.")
        return value

    @field_validator("score_delta")
    @classmethod
    def validate_score_delta(cls, value: str, info) -> str:
        return _validate_signed_unit_decimal(value, info.field_name)

    @model_validator(mode="after")
    def validate_delta(self) -> EvalStructuredJudgeCriterionComparisonV1:
        expected = _canonical_signed_decimal(
            _exact_decimal_difference(
                Decimal(self.current_score),
                Decimal(self.baseline_score),
            )
        )
        if self.score_delta != expected:
            raise ValueError("Criterion score_delta contradicts its retained scores.")
        return self


class EvalStructuredJudgeComparisonV1(_PortableModel):
    """Explainable comparison of one exact case/trial/assertion observation."""

    case_id: StrictStr
    trial_number: StrictInt | None = Field(default=None, ge=1, le=EVAL_CORPUS_MAX_TRIALS)
    assertion_id: StrictStr
    baseline_outcome: PublishedOutcome
    current_outcome: PublishedOutcome
    baseline: EvalStructuredJudgePresentationV1
    current: EvalStructuredJudgePresentationV1
    evaluator_change: EvalStructuredJudgeEvaluatorChange
    aggregate_change: EvalStructuredJudgeAggregateChange
    aggregate_delta: StrictStr | None
    regressed: StrictBool
    criteria: tuple[EvalStructuredJudgeCriterionComparisonV1, ...] = Field(
        default=(),
        max_length=EVAL_CORPUS_MAX_JUDGE_CRITERIA,
    )

    @field_validator("case_id", "assertion_id")
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("baseline", "current", mode="before")
    @classmethod
    def copy_judgment(cls, value: object) -> object:
        if type(value) is EvalStructuredJudgePresentationV1:
            return value.model_dump(mode="python", round_trip=True, warnings="none")
        if isinstance(value, BaseModel):
            raise TypeError(
                "Judgments must be exact EvalStructuredJudgePresentationV1 values or JSON objects."
            )
        return value

    @field_validator("criteria", mode="before")
    @classmethod
    def validate_criteria_order(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @field_validator("aggregate_delta")
    @classmethod
    def validate_aggregate_delta(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _validate_signed_unit_decimal(value, info.field_name)

    @model_validator(mode="after")
    def validate_comparison(self) -> EvalStructuredJudgeComparisonV1:
        baseline_detail = self.baseline.detail
        current_detail = self.current.detail
        if self.baseline_outcome != _structured_judge_outcome(
            baseline_detail.diagnostic,
            baseline_detail.aggregate_score,
            baseline_detail.threshold,
        ) or self.current_outcome != _structured_judge_outcome(
            current_detail.diagnostic,
            current_detail.aggregate_score,
            current_detail.threshold,
        ):
            raise ValueError("Compared outcomes contradict their retained judgments.")
        static_baseline = (
            baseline_detail.judge_profile,
            baseline_detail.candidate_route_relation,
            baseline_detail.rubric_id,
            baseline_detail.rubric_revision,
            baseline_detail.reference,
            baseline_detail.threshold,
            baseline_detail.evidence,
        )
        static_current = (
            current_detail.judge_profile,
            current_detail.candidate_route_relation,
            current_detail.rubric_id,
            current_detail.rubric_revision,
            current_detail.reference,
            current_detail.threshold,
            current_detail.evidence,
        )
        if static_baseline != static_current:
            raise ValueError(
                "Compared structured judgments must have identical admitted contracts."
            )
        expected_evaluator_change = _evaluator_change(
            baseline_detail.diagnostic,
            current_detail.diagnostic,
        )
        if self.evaluator_change != expected_evaluator_change:
            raise ValueError("evaluator_change contradicts retained judge diagnostics.")
        expected_criteria = _criterion_comparisons(self.baseline, self.current)
        if self.criteria != expected_criteria:
            raise ValueError("Criterion comparisons contradict retained judgments.")
        expected_delta = _aggregate_delta(self.baseline, self.current)
        if self.aggregate_delta != expected_delta:
            raise ValueError("aggregate_delta contradicts retained judgments.")
        return self


class EvalStructuredJudgeObservationMismatchV1(_PortableModel):
    """One structured observation present on only one side of a comparison."""

    case_id: StrictStr
    trial_number: StrictInt | None = Field(default=None, ge=1, le=EVAL_CORPUS_MAX_TRIALS)
    assertion_id: StrictStr
    availability: Literal["baseline_only", "current_only"]

    @field_validator("case_id", "assertion_id")
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)


class EvalToolJsonAssertionComparisonV1(_PortableModel):
    """Exact comparison of one safe retained tool-JSON observation."""

    case_id: StrictStr
    trial_number: StrictInt | None = Field(default=None, ge=1, le=EVAL_CORPUS_MAX_TRIALS)
    assertion_id: StrictStr
    baseline_outcome: PublishedOutcome
    current_outcome: PublishedOutcome
    baseline: PublishedToolArgumentsContainDetail | PublishedToolResultContainsDetail
    current: PublishedToolArgumentsContainDetail | PublishedToolResultContainsDetail
    evidence_state_changed: StrictBool
    observed_value_change: EvalToolJsonObservedValueChange
    outcome_changed: StrictBool
    regressed: StrictBool

    @field_validator("case_id", "assertion_id")
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("baseline", "current", mode="before")
    @classmethod
    def copy_detail(cls, value: object) -> object:
        if type(value) in {
            PublishedToolArgumentsContainDetail,
            PublishedToolResultContainsDetail,
        }:
            detail = cast(
                "PublishedToolArgumentsContainDetail | PublishedToolResultContainsDetail",
                value,
            )
            return detail.model_dump(mode="python", round_trip=True, warnings="none")
        if isinstance(value, BaseModel):
            raise TypeError(
                "Tool JSON details must be exact published tool-JSON detail values or JSON objects."
            )
        return value

    @model_validator(mode="after")
    def validate_comparison(self) -> EvalToolJsonAssertionComparisonV1:
        baseline_contract = (
            self.baseline.kind,
            self.baseline.tool_name,
            self.baseline.occurrence,
            self.baseline.expected_subset,
        )
        current_contract = (
            self.current.kind,
            self.current.tool_name,
            self.current.occurrence,
            self.current.expected_subset,
        )
        if baseline_contract != current_contract:
            raise ValueError("Compared tool JSON observations must share one assertion contract.")
        baseline_expected_outcome = _tool_json_outcome(self.baseline)
        current_expected_outcome = _tool_json_outcome(self.current)
        if (
            baseline_expected_outcome is not None
            and self.baseline_outcome != baseline_expected_outcome
        ) or (
            current_expected_outcome is not None
            and self.current_outcome != current_expected_outcome
        ):
            raise ValueError("Compared outcomes contradict their retained tool JSON evidence.")
        if (
            baseline_expected_outcome is None
            and self.baseline_outcome not in {"unavailable", "error"}
        ) or (
            current_expected_outcome is None
            and self.current_outcome not in {"unavailable", "error"}
        ):
            raise ValueError("Incomplete tool JSON evidence cannot have a scored outcome.")
        if self.evidence_state_changed != (
            self.baseline.observation_state != self.current.observation_state
        ):
            raise ValueError("evidence_state_changed contradicts retained observations.")
        expected_value_change = _tool_json_observed_value_change(
            self.baseline,
            self.current,
        )
        if self.observed_value_change != expected_value_change:
            raise ValueError("observed_value_change contradicts retained observations.")
        if self.outcome_changed != (self.baseline_outcome != self.current_outcome):
            raise ValueError("outcome_changed contradicts retained outcomes.")
        if self.regressed != (
            _STATUS_SEVERITY[self.current_outcome] > _STATUS_SEVERITY[self.baseline_outcome]
        ):
            raise ValueError("regressed contradicts retained outcomes.")
        return self


class EvalToolJsonObservationMismatchV1(_PortableModel):
    """One safe tool-JSON observation present on only one comparison side."""

    case_id: StrictStr
    trial_number: StrictInt | None = Field(default=None, ge=1, le=EVAL_CORPUS_MAX_TRIALS)
    assertion_id: StrictStr
    availability: Literal["baseline_only", "current_only"]

    @field_validator("case_id", "assertion_id")
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)


def _tool_json_outcome(
    detail: PublishedToolArgumentsContainDetail | PublishedToolResultContainsDetail,
) -> PublishedOutcome | None:
    if detail.observation_state in {
        "unavailable",
        "unsupported",
        "malformed",
        "incompatible",
        "limit_exceeded",
        "truncated",
        "redacted",
    }:
        return None
    return "passed" if detail.matched else "failed"


def _tool_json_observed_value_change(
    baseline: PublishedToolArgumentsContainDetail | PublishedToolResultContainsDetail,
    current: PublishedToolArgumentsContainDetail | PublishedToolResultContainsDetail,
) -> EvalToolJsonObservedValueChange:
    if baseline.actual is None or current.actual is None:
        return "unavailable"
    return "unchanged" if equal_json_values(baseline.actual, current.actual) else "changed"


def _evaluator_change(
    baseline: PublishedModelJudgeDiagnostic,
    current: PublishedModelJudgeDiagnostic,
) -> EvalStructuredJudgeEvaluatorChange:
    baseline_severity = _EVALUATOR_SEVERITY[baseline]
    current_severity = _EVALUATOR_SEVERITY[current]
    if current_severity > baseline_severity:
        return "regressed"
    if current_severity < baseline_severity:
        return "improved"
    return "unchanged"


def _structured_judge_outcome(
    diagnostic: PublishedModelJudgeDiagnostic,
    aggregate_score: str | None,
    threshold: str,
) -> PublishedOutcome:
    if diagnostic == "evidence_unavailable":
        return "unavailable"
    if diagnostic == "evaluator_error":
        return "error"
    if aggregate_score is None:
        raise ValueError("Recorded structured judgment requires an aggregate score.")
    return "passed" if Decimal(aggregate_score) >= Decimal(threshold) else "failed"


def _criterion_comparisons(
    baseline: EvalStructuredJudgePresentationV1,
    current: EvalStructuredJudgePresentationV1,
) -> tuple[EvalStructuredJudgeCriterionComparisonV1, ...]:
    if not baseline.criteria or not current.criteria:
        return ()
    baseline_ids = tuple(item.criterion_id for item in baseline.criteria)
    current_ids = tuple(item.criterion_id for item in current.criteria)
    if baseline_ids != current_ids:
        raise ValueError("Compared structured judgment criterion identities do not match.")
    comparisons: list[EvalStructuredJudgeCriterionComparisonV1] = []
    for baseline_item, current_item in zip(
        baseline.criteria,
        current.criteria,
        strict=True,
    ):
        if baseline_item.weight != current_item.weight:
            raise ValueError("Compared structured judgment criterion weights do not match.")
        comparisons.append(
            EvalStructuredJudgeCriterionComparisonV1(
                criterion_id=baseline_item.criterion_id,
                weight=baseline_item.weight,
                baseline_score=baseline_item.score,
                current_score=current_item.score,
                score_delta=_canonical_signed_decimal(
                    _exact_decimal_difference(
                        Decimal(current_item.score),
                        Decimal(baseline_item.score),
                    )
                ),
                baseline_explanation_state=baseline_item.explanation_state,
                current_explanation_state=current_item.explanation_state,
            )
        )
    return tuple(comparisons)


def _aggregate_delta(
    baseline: EvalStructuredJudgePresentationV1,
    current: EvalStructuredJudgePresentationV1,
) -> str | None:
    baseline_score = baseline.detail.aggregate_score
    current_score = current.detail.aggregate_score
    if baseline_score is None or current_score is None:
        return None
    return _canonical_signed_decimal(
        _exact_decimal_difference(Decimal(current_score), Decimal(baseline_score))
    )


def _aggregate_change(
    baseline: EvalStructuredJudgePresentationV1,
    current: EvalStructuredJudgePresentationV1,
    *,
    tolerance: Decimal,
) -> EvalStructuredJudgeAggregateChange:
    delta = _aggregate_delta(baseline, current)
    if delta is None:
        return "unavailable"
    decimal_delta = Decimal(delta)
    if decimal_delta < tolerance.copy_negate():
        return "regressed"
    if decimal_delta > tolerance:
        return "improved"
    return "unchanged"


def _structured_judge_regressed(
    *,
    baseline_outcome: PublishedOutcome,
    current_outcome: PublishedOutcome,
    evaluator_change: EvalStructuredJudgeEvaluatorChange,
    aggregate_change: EvalStructuredJudgeAggregateChange,
) -> bool:
    return (
        evaluator_change == "regressed"
        or aggregate_change == "regressed"
        or (baseline_outcome == "passed" and current_outcome == "failed")
    )


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
    target_key: StrictStr
    application_release_id: StrictStr = Field(min_length=1, max_length=256)
    app_manifest_fingerprint: StrictStr = Field(min_length=64, max_length=64)
    external_target_revision: StrictStr | None
    corpus_revision: StrictStr
    suite_id: StrictStr
    suite_revision: StrictStr
    evidence_policy_revision: StrictStr
    pricing_profile_fingerprint: StrictStr | None
    trial_policy_revision: StrictStr
    accepted_exposure_revision: StrictStr | None
    accepted_exposure_comparison_revision: StrictStr | None
    memory_attribution_support: Literal["unsupported"] = "unsupported"
    status: PublishedStatus
    score: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator(
        "result_revision",
        "external_target_revision",
        "corpus_revision",
        "suite_revision",
        "evidence_policy_revision",
        "pricing_profile_fingerprint",
        "trial_policy_revision",
        "accepted_exposure_revision",
        "accepted_exposure_comparison_revision",
    )
    @classmethod
    def validate_revisions(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _sha256_revision(value, info.field_name)

    @field_validator("target_key", "suite_id")
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

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
    def validate_exposure_revisions(self) -> CorpusComparisonResultSummary:
        if (self.accepted_exposure_revision is None) != (
            self.accepted_exposure_comparison_revision is None
        ):
            raise ValueError(
                "Accepted work exposure requires exact and comparison revisions together."
            )
        return self

    @model_validator(mode="after")
    def validate_score(self) -> CorpusComparisonResultSummary:
        if (self.status in {"unavailable", "error"}) != (self.score is None):
            raise ValueError("Comparison result status contradicts its score.")
        return self


class CorpusReliabilityDistributionV1(BaseModel):
    """Lossless trial-outcome counts used for cross-release reliability decisions."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    schema_version: Literal[1] = 1
    total_trials: StrictInt = Field(ge=1, le=EVAL_CORPUS_MAX_TRIALS)
    passed_trials: StrictInt = Field(ge=0, le=EVAL_CORPUS_MAX_TRIALS)
    candidate_failed_trials: StrictInt = Field(ge=0, le=EVAL_CORPUS_MAX_TRIALS)
    runtime_error_trials: StrictInt = Field(ge=0, le=EVAL_CORPUS_MAX_TRIALS)
    evaluator_error_trials: StrictInt = Field(ge=0, le=EVAL_CORPUS_MAX_TRIALS)
    unavailable_trials: StrictInt = Field(ge=0, le=EVAL_CORPUS_MAX_TRIALS)
    cancelled_trials: StrictInt = Field(ge=0, le=EVAL_CORPUS_MAX_TRIALS)

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1.")
        return value

    @model_validator(mode="after")
    def validate_distribution(self) -> CorpusReliabilityDistributionV1:
        classified = (
            self.passed_trials
            + self.candidate_failed_trials
            + self.runtime_error_trials
            + self.evaluator_error_trials
            + self.unavailable_trials
            + self.cancelled_trials
        )
        if classified != self.total_trials:
            raise ValueError("Reliability counts must classify every retained trial once.")
        return self

    @classmethod
    def from_diagnostic_codes(
        cls,
        codes: tuple[EvalTrialDiagnosticCode, ...],
    ) -> CorpusReliabilityDistributionV1:
        if not codes:
            raise ValueError("Reliability comparison requires retained trial diagnostics.")
        passed = candidate_failed = runtime_errors = evaluator_errors = unavailable = 0
        cancelled = 0
        for code in codes:
            if type(code) is not EvalTrialDiagnosticCode:
                raise TypeError("Reliability diagnostics must be EvalTrialDiagnosticCode values.")
            if code is EvalTrialDiagnosticCode.PASSED:
                passed += 1
            elif code is EvalTrialDiagnosticCode.ASSERTION_FAILED:
                candidate_failed += 1
            elif code is EvalTrialDiagnosticCode.ASSERTION_EVALUATION_FAILED:
                evaluator_errors += 1
            elif code is EvalTrialDiagnosticCode.EXTERNAL_TARGET_CANCELLED:
                cancelled += 1
            elif code in _UNAVAILABLE_TRIAL_DIAGNOSTICS:
                unavailable += 1
            else:
                runtime_errors += 1
        return cls(
            total_trials=len(codes),
            passed_trials=passed,
            candidate_failed_trials=candidate_failed,
            runtime_error_trials=runtime_errors,
            evaluator_error_trials=evaluator_errors,
            unavailable_trials=unavailable,
            cancelled_trials=cancelled,
        )


def _reliability_severity(
    distribution: CorpusReliabilityDistributionV1,
) -> tuple[int, int, int]:
    hard_errors = distribution.runtime_error_trials + distribution.evaluator_error_trials
    unavailable = hard_errors + distribution.unavailable_trials + distribution.cancelled_trials
    return (
        hard_errors,
        unavailable,
        unavailable + distribution.candidate_failed_trials,
    )


def _reliability_change(
    baseline: CorpusReliabilityDistributionV1,
    current: CorpusReliabilityDistributionV1,
) -> CorpusReliabilityChange:
    if baseline.total_trials != current.total_trials:
        raise ValueError("Comparable reliability distributions require equal trial counts.")
    if baseline == current:
        return "unchanged"
    baseline_severity = _reliability_severity(baseline)
    current_severity = _reliability_severity(current)
    if current_severity > baseline_severity:
        return "regressed"
    if current_severity < baseline_severity:
        return "improved"
    return "changed"


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
    baseline_reliability: CorpusReliabilityDistributionV1
    current_reliability: CorpusReliabilityDistributionV1
    reliability_change: CorpusReliabilityChange

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

    @field_validator("baseline_reliability", "current_reliability", mode="before")
    @classmethod
    def copy_reliability(cls, value: object) -> object:
        if type(value) is CorpusReliabilityDistributionV1:
            return value.model_dump(mode="python", round_trip=True, warnings="none")
        if isinstance(value, BaseModel):
            raise TypeError(
                "Reliability must be an exact CorpusReliabilityDistributionV1 or JSON object."
            )
        return value

    @model_validator(mode="after")
    def validate_scores(self) -> CorpusCaseComparison:
        for label, status, score in (
            ("baseline", self.baseline_status, self.baseline_score),
            ("current", self.current_status, self.current_score),
        ):
            if (status in {"unavailable", "error"}) != (score is None):
                raise ValueError(f"{label} case status contradicts its score.")
        expected_baseline = CorpusReliabilityDistributionV1.from_diagnostic_codes(
            self.baseline_trial_diagnostic_codes
        )
        expected_current = CorpusReliabilityDistributionV1.from_diagnostic_codes(
            self.current_trial_diagnostic_codes
        )
        if self.baseline_reliability != expected_baseline:
            raise ValueError("Baseline reliability contradicts retained trial diagnostics.")
        if self.current_reliability != expected_current:
            raise ValueError("Current reliability contradicts retained trial diagnostics.")
        expected_change = _reliability_change(expected_baseline, expected_current)
        if self.reliability_change != expected_change:
            raise ValueError("Reliability change contradicts retained trial distributions.")
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
    baseline_reliability: CorpusReliabilityDistributionV1 | None = None
    current_reliability: CorpusReliabilityDistributionV1 | None = None

    @field_validator("case_id")
    @classmethod
    def validate_case_id(cls, value: str | None) -> str | None:
        if value is not None and (value != value.strip() or not value.isprintable()):
            raise ValueError("case_id must be clean nonblank text.")
        return value

    @field_validator("baseline_reliability", "current_reliability", mode="before")
    @classmethod
    def copy_reliability(cls, value: object) -> object:
        if type(value) is CorpusReliabilityDistributionV1:
            return value.model_dump(mode="python", round_trip=True, warnings="none")
        if isinstance(value, BaseModel):
            raise TypeError(
                "Reliability must be an exact CorpusReliabilityDistributionV1 or JSON object."
            )
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
            if self.baseline_reliability is not None or self.current_reliability is not None:
                raise ValueError("Status regressions cannot carry reliability distributions.")
            if _STATUS_SEVERITY[self.current_status] <= _STATUS_SEVERITY[self.baseline_status]:
                raise ValueError("A status regression must move to a worse status.")
        elif self.kind is CorpusRegressionKind.SCORE:
            if self.baseline_score is None or self.current_score is None:
                raise ValueError("Score regressions require both scores.")
            if self.baseline_status is not None or self.current_status is not None:
                raise ValueError("Score regressions cannot carry statuses.")
            if self.baseline_reliability is not None or self.current_reliability is not None:
                raise ValueError("Score regressions cannot carry reliability distributions.")
            if self.current_score >= self.baseline_score:
                raise ValueError("A score regression must lower the score.")
        else:
            baseline = self.baseline_reliability
            current = self.current_reliability
            if baseline is None or current is None:
                raise ValueError("Reliability regressions require both distributions.")
            if any(
                value is not None
                for value in (
                    self.baseline_status,
                    self.current_status,
                    self.baseline_score,
                    self.current_score,
                )
            ):
                raise ValueError("Reliability regressions cannot carry statuses or scores.")
            if _reliability_change(baseline, current) != "regressed":
                raise ValueError("A reliability regression must worsen the trial distribution.")
        return self


class CorpusExecutionComparison(BaseModel):
    """Deterministic contract-aware comparison of two published corpus executions."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    schema_version: Literal[4] = 4
    compatibility: CorpusComparisonCompatibility
    score_tolerance: StrictFloat = Field(default=0.0, ge=0.0, le=1.0)
    baseline: CorpusComparisonResultSummary
    current: CorpusComparisonResultSummary
    cases: tuple[CorpusCaseComparison, ...] = Field(default_factory=tuple, max_length=1_000)
    regressions: tuple[CorpusExecutionRegression, ...] = Field(
        default_factory=tuple,
        max_length=3_002,
    )
    structured_judge_comparison_state: EvalStructuredJudgeComparisonState
    structured_judgments: tuple[EvalStructuredJudgeComparisonV1, ...] = Field(
        default=(),
        max_length=EVAL_STRUCTURED_JUDGE_COMPARISON_MAX_ITEMS,
    )
    structured_judge_observation_mismatches: tuple[
        EvalStructuredJudgeObservationMismatchV1, ...
    ] = Field(
        default=(),
        max_length=EVAL_STRUCTURED_JUDGE_COMPARISON_MAX_ITEMS * 2,
    )
    tool_json_comparison_state: EvalToolJsonComparisonState
    tool_json_assertions: tuple[EvalToolJsonAssertionComparisonV1, ...] = Field(
        default=(),
        max_length=EVAL_STRUCTURED_JUDGE_COMPARISON_MAX_ITEMS,
    )
    tool_json_observation_mismatches: tuple[EvalToolJsonObservationMismatchV1, ...] = Field(
        default=(),
        max_length=EVAL_STRUCTURED_JUDGE_COMPARISON_MAX_ITEMS * 2,
    )

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 4.")
        return value

    @field_validator(
        "structured_judgments",
        "structured_judge_observation_mismatches",
        "tool_json_assertions",
        "tool_json_observation_mismatches",
        mode="before",
    )
    @classmethod
    def validate_structured_judgments_order(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @model_validator(mode="after")
    def validate_derived_contract(self) -> CorpusExecutionComparison:
        if self.compatibility.baseline_result_revision != self.baseline.result_revision:
            raise ValueError("Baseline summary does not match compatibility evidence.")
        if self.compatibility.current_result_revision != self.current.result_revision:
            raise ValueError("Current summary does not match compatibility evidence.")
        if not self.compatibility.comparable:
            if (
                self.cases
                or self.regressions
                or self.structured_judgments
                or self.structured_judge_observation_mismatches
                or self.tool_json_assertions
                or self.tool_json_observation_mismatches
            ):
                raise ValueError("Incomparable results cannot carry outcome comparisons.")
            if self.structured_judge_comparison_state != "contract_incompatible":
                raise ValueError(
                    "Incomparable results require the contract_incompatible judge state."
                )
            if self.tool_json_comparison_state != "contract_incompatible":
                raise ValueError(
                    "Incomparable results require the contract_incompatible tool JSON state."
                )
            return self._validate_size()
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
        if self.structured_judge_comparison_state == "compared":
            if not self.structured_judgments:
                raise ValueError("Compared structured judges require retained observations.")
        elif self.structured_judgments:
            raise ValueError(
                "Structured judge observations require the compared presentation state."
            )
        if self.structured_judge_comparison_state == "observation_identity_mismatch":
            if not self.structured_judge_observation_mismatches:
                raise ValueError("Identity mismatch state requires exact unmatched observations.")
        elif self.structured_judge_observation_mismatches:
            raise ValueError("Unmatched observations require the identity mismatch state.")
        keys = tuple(
            (item.case_id, item.trial_number, item.assertion_id)
            for item in self.structured_judgments
        )
        if len(keys) != len(set(keys)):
            raise ValueError("Structured judge comparison identities must be unique.")
        mismatch_keys = tuple(
            (item.case_id, item.trial_number, item.assertion_id, item.availability)
            for item in self.structured_judge_observation_mismatches
        )
        if len(mismatch_keys) != len(set(mismatch_keys)):
            raise ValueError("Structured judge mismatch identities must be unique.")
        if self.tool_json_comparison_state == "compared":
            if not self.tool_json_assertions:
                raise ValueError("Compared tool JSON assertions require retained observations.")
        elif self.tool_json_assertions:
            raise ValueError("Tool JSON observations require the compared presentation state.")
        if self.tool_json_comparison_state == "observation_identity_mismatch":
            if not self.tool_json_observation_mismatches:
                raise ValueError(
                    "Tool JSON identity mismatch state requires exact unmatched observations."
                )
        elif self.tool_json_observation_mismatches:
            raise ValueError(
                "Unmatched tool JSON observations require the identity mismatch state."
            )
        tool_json_keys = tuple(
            (item.case_id, item.trial_number, item.assertion_id)
            for item in self.tool_json_assertions
        )
        if len(tool_json_keys) != len(set(tool_json_keys)):
            raise ValueError("Tool JSON comparison identities must be unique.")
        tool_json_mismatch_keys = tuple(
            (item.case_id, item.trial_number, item.assertion_id, item.availability)
            for item in self.tool_json_observation_mismatches
        )
        if len(tool_json_mismatch_keys) != len(set(tool_json_mismatch_keys)):
            raise ValueError("Tool JSON mismatch identities must be unique.")
        for item in self.structured_judgments:
            expected_change = _aggregate_change(
                item.baseline,
                item.current,
                tolerance=Decimal(str(self.score_tolerance)),
            )
            expected_regressed = _structured_judge_regressed(
                baseline_outcome=item.baseline_outcome,
                current_outcome=item.current_outcome,
                evaluator_change=item.evaluator_change,
                aggregate_change=expected_change,
            )
            if item.aggregate_change != expected_change or item.regressed != expected_regressed:
                raise ValueError(
                    "Structured judge change classification contradicts retained observations."
                )
        return self._validate_size()

    def _validate_size(self) -> CorpusExecutionComparison:
        if not json_utf8_size_within_limit(self, CORPUS_EXECUTION_COMPARISON_MAX_BYTES):
            raise ValueError(
                "Corpus execution comparison exceeds "
                f"{CORPUS_EXECUTION_COMPARISON_MAX_BYTES} canonical JSON bytes."
            )
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
    origin: EvalResultOrigin
    target_key: str
    external_target_revision: str | None
    corpus_revision: str
    suite_id: str
    suite_revision: str
    evidence_policy_revision: str
    pricing_profile_fingerprint: str | None
    trial_policy_revision: str
    accepted_exposure_revision: str | None
    accepted_exposure_comparison_revision: str | None
    uses_pricing: bool
    case_contract: tuple[tuple[str, str], ...]
    assertion_contract: tuple[tuple[str, tuple[tuple[object, ...], ...]], ...]
    summary: CorpusComparisonResultSummary
    cases: tuple[_CaseOutcome, ...]


@dataclass(frozen=True)
class _StructuredObservation:
    case_id: str
    trial_number: int | None
    assertion: EvalAssertionPresentationV1

    @property
    def key(self) -> tuple[str, int | None, str]:
        return self.case_id, self.trial_number, self.assertion.assertion_id


@dataclass(frozen=True)
class _ToolJsonObservation:
    case_id: str
    trial_number: int | None
    assertion: PublishedAssertionResult

    @property
    def key(self) -> tuple[str, int | None, str]:
        return self.case_id, self.trial_number, self.assertion.assertion_id


def _structured_observations(
    presentation: EvalResultPresentationV2,
) -> tuple[_StructuredObservation, ...]:
    observations = tuple(
        _StructuredObservation(
            case_id=case.case_id,
            trial_number=trial.trial_number,
            assertion=assertion,
        )
        for case in presentation.cases
        for trial in case.trials
        for assertion in trial.assertions
        if assertion.structured_judge is not None
    )
    keys = tuple(item.key for item in observations)
    if len(keys) != len(set(keys)):
        raise ValueError("Structured judge presentation observation identities must be unique.")
    return observations


def _structured_comparison(
    baseline: CorpusExecutionResult
    | CapturedEvaluationResultV1
    | EvalResultProjectionV1
    | EvalResultProjectionV2,
    current: CorpusExecutionResult
    | CapturedEvaluationResultV1
    | EvalResultProjectionV1
    | EvalResultProjectionV2,
    *,
    comparable: bool,
    score_tolerance: float,
) -> tuple[
    EvalStructuredJudgeComparisonState,
    tuple[EvalStructuredJudgeComparisonV1, ...],
    tuple[EvalStructuredJudgeObservationMismatchV1, ...],
]:
    if not comparable:
        return "contract_incompatible", (), ()
    if type(baseline) in {EvalResultProjectionV1, EvalResultProjectionV2} or type(current) in {
        EvalResultProjectionV1,
        EvalResultProjectionV2,
    }:
        return "source_detail_unavailable", (), ()
    baseline_presentation = present_eval_result(
        cast("CorpusExecutionResult | CapturedEvaluationResultV1", baseline)
    )
    current_presentation = present_eval_result(
        cast("CorpusExecutionResult | CapturedEvaluationResultV1", current)
    )
    baseline_observations = _structured_observations(baseline_presentation)
    current_observations = _structured_observations(current_presentation)
    if not baseline_observations and not current_observations:
        return "no_structured_judges", (), ()
    baseline_by_key = {item.key: item for item in baseline_observations}
    current_by_key = {item.key: item for item in current_observations}
    if baseline_by_key.keys() != current_by_key.keys():
        mismatches = tuple(
            EvalStructuredJudgeObservationMismatchV1(
                case_id=case_id,
                trial_number=trial_number,
                assertion_id=assertion_id,
                availability="baseline_only",
            )
            for case_id, trial_number, assertion_id in baseline_by_key
            if (case_id, trial_number, assertion_id) not in current_by_key
        ) + tuple(
            EvalStructuredJudgeObservationMismatchV1(
                case_id=case_id,
                trial_number=trial_number,
                assertion_id=assertion_id,
                availability="current_only",
            )
            for case_id, trial_number, assertion_id in current_by_key
            if (case_id, trial_number, assertion_id) not in baseline_by_key
        )
        return "observation_identity_mismatch", (), mismatches
    tolerance = Decimal(str(score_tolerance))
    comparisons: list[EvalStructuredJudgeComparisonV1] = []
    for baseline_observation in baseline_observations:
        current_observation = current_by_key[baseline_observation.key]
        baseline_judgment = baseline_observation.assertion.structured_judge
        current_judgment = current_observation.assertion.structured_judge
        if baseline_judgment is None or current_judgment is None:
            raise AssertionError("Structured observation lost its judgment presentation.")
        aggregate_change = _aggregate_change(
            baseline_judgment,
            current_judgment,
            tolerance=tolerance,
        )
        evaluator_change = _evaluator_change(
            baseline_judgment.detail.diagnostic,
            current_judgment.detail.diagnostic,
        )
        comparisons.append(
            EvalStructuredJudgeComparisonV1(
                case_id=baseline_observation.case_id,
                trial_number=baseline_observation.trial_number,
                assertion_id=baseline_observation.assertion.assertion_id,
                baseline_outcome=baseline_observation.assertion.outcome,
                current_outcome=current_observation.assertion.outcome,
                baseline=baseline_judgment,
                current=current_judgment,
                evaluator_change=evaluator_change,
                aggregate_change=aggregate_change,
                aggregate_delta=_aggregate_delta(baseline_judgment, current_judgment),
                regressed=_structured_judge_regressed(
                    baseline_outcome=baseline_observation.assertion.outcome,
                    current_outcome=current_observation.assertion.outcome,
                    evaluator_change=evaluator_change,
                    aggregate_change=aggregate_change,
                ),
                criteria=_criterion_comparisons(baseline_judgment, current_judgment),
            )
        )
    return "compared", tuple(comparisons), ()


def _tool_json_observations(
    result: CorpusExecutionResult | CapturedEvaluationResultV1,
) -> tuple[_ToolJsonObservation, ...]:
    if type(result) is CorpusExecutionResult:
        observations = tuple(
            _ToolJsonObservation(
                case_id=case.case_id,
                trial_number=trial.trial_number,
                assertion=assertion,
            )
            for case in result.run.cases
            for trial in case.trials
            for assertion in trial.assertions
            if isinstance(
                assertion.detail,
                (PublishedToolArgumentsContainDetail, PublishedToolResultContainsDetail),
            )
        )
    elif type(result) is CapturedEvaluationResultV1:
        observations = tuple(
            _ToolJsonObservation(
                case_id=result.score.case_id,
                trial_number=None,
                assertion=assertion,
            )
            for assertion in result.score.assertions
            if isinstance(
                assertion.detail,
                (PublishedToolArgumentsContainDetail, PublishedToolResultContainsDetail),
            )
        )
    else:  # pragma: no cover - closed by the public caller's exact-type checks
        raise TypeError("result must be a fresh or captured evaluation result.")
    keys = tuple(item.key for item in observations)
    if len(keys) != len(set(keys)):
        raise ValueError("Tool JSON observation identities must be unique.")
    return observations


def _tool_json_comparison(
    baseline: CorpusExecutionResult
    | CapturedEvaluationResultV1
    | EvalResultProjectionV1
    | EvalResultProjectionV2,
    current: CorpusExecutionResult
    | CapturedEvaluationResultV1
    | EvalResultProjectionV1
    | EvalResultProjectionV2,
    *,
    comparable: bool,
) -> tuple[
    EvalToolJsonComparisonState,
    tuple[EvalToolJsonAssertionComparisonV1, ...],
    tuple[EvalToolJsonObservationMismatchV1, ...],
]:
    if not comparable:
        return "contract_incompatible", (), ()
    if type(baseline) in {EvalResultProjectionV1, EvalResultProjectionV2} or type(current) in {
        EvalResultProjectionV1,
        EvalResultProjectionV2,
    }:
        return "source_detail_unavailable", (), ()
    validated_baseline = cast("CorpusExecutionResult | CapturedEvaluationResultV1", baseline)
    validated_current = cast("CorpusExecutionResult | CapturedEvaluationResultV1", current)
    baseline_observations = _tool_json_observations(validated_baseline)
    current_observations = _tool_json_observations(validated_current)
    if not baseline_observations and not current_observations:
        return "no_tool_json_assertions", (), ()
    baseline_by_key = {item.key: item for item in baseline_observations}
    current_by_key = {item.key: item for item in current_observations}
    if baseline_by_key.keys() != current_by_key.keys():
        mismatches = tuple(
            EvalToolJsonObservationMismatchV1(
                case_id=case_id,
                trial_number=trial_number,
                assertion_id=assertion_id,
                availability="baseline_only",
            )
            for case_id, trial_number, assertion_id in baseline_by_key
            if (case_id, trial_number, assertion_id) not in current_by_key
        ) + tuple(
            EvalToolJsonObservationMismatchV1(
                case_id=case_id,
                trial_number=trial_number,
                assertion_id=assertion_id,
                availability="current_only",
            )
            for case_id, trial_number, assertion_id in current_by_key
            if (case_id, trial_number, assertion_id) not in baseline_by_key
        )
        return "observation_identity_mismatch", (), mismatches
    comparisons: list[EvalToolJsonAssertionComparisonV1] = []
    for baseline_observation in baseline_observations:
        current_observation = current_by_key[baseline_observation.key]
        baseline_detail = baseline_observation.assertion.detail
        current_detail = current_observation.assertion.detail
        if not isinstance(
            baseline_detail,
            (PublishedToolArgumentsContainDetail, PublishedToolResultContainsDetail),
        ) or not isinstance(
            current_detail,
            (PublishedToolArgumentsContainDetail, PublishedToolResultContainsDetail),
        ):
            raise AssertionError("Tool JSON observation lost its published detail.")
        comparisons.append(
            EvalToolJsonAssertionComparisonV1(
                case_id=baseline_observation.case_id,
                trial_number=baseline_observation.trial_number,
                assertion_id=baseline_observation.assertion.assertion_id,
                baseline_outcome=baseline_observation.assertion.outcome,
                current_outcome=current_observation.assertion.outcome,
                baseline=baseline_detail,
                current=current_detail,
                evidence_state_changed=(
                    baseline_detail.observation_state != current_detail.observation_state
                ),
                observed_value_change=_tool_json_observed_value_change(
                    baseline_detail,
                    current_detail,
                ),
                outcome_changed=(
                    baseline_observation.assertion.outcome != current_observation.assertion.outcome
                ),
                regressed=(
                    _STATUS_SEVERITY[current_observation.assertion.outcome]
                    > _STATUS_SEVERITY[baseline_observation.assertion.outcome]
                ),
            )
        )
    return "compared", tuple(comparisons), ()


def _validated_projection(
    result: CorpusExecutionResult
    | CapturedEvaluationResultV1
    | EvalResultProjectionV1
    | EvalResultProjectionV2,
    field_name: str,
) -> _ExecutionProjection:
    """Validate one potentially forged graph, then retain only bounded comparison data."""

    if type(result) is EvalResultProjectionV2:
        projection = EvalResultProjectionV2.model_validate(
            result.model_dump(mode="python", round_trip=True, warnings="none")
        )
    elif type(result) is EvalResultProjectionV1:
        v1_projection = EvalResultProjectionV1.model_validate(
            result.model_dump(mode="python", round_trip=True, warnings="none")
        )
        trial_counts = {len(case.trial_diagnostic_codes) for case in v1_projection.cases}
        if len(trial_counts) != 1:
            raise ValueError("V1 result projection has inconsistent suite trial counts.")
        trial_policy = EvalSuiteTrialPolicyV1.create(trial_count=trial_counts.pop())
        projection = EvalResultProjectionV2(
            **v1_projection.model_dump(mode="python", exclude={"schema_version"}),
            trial_policy_revision=trial_policy.revision,
        )
    elif type(result) in {CorpusExecutionResult, CapturedEvaluationResultV1}:
        projection = eval_result_projection(
            cast("CorpusExecutionResult | CapturedEvaluationResultV1", result)
        )
    else:
        raise TypeError(
            f"{field_name} must be an exact CorpusExecutionResult, "
            "CapturedEvaluationResultV1, EvalResultProjectionV1, or EvalResultProjectionV2."
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
        target_key=projection.target.target_key,
        application_release_id=projection.target.application_release_id,
        app_manifest_fingerprint=projection.target.app_manifest_fingerprint,
        external_target_revision=projection.external_target_revision,
        corpus_revision=projection.corpus_revision,
        suite_id=projection.suite_id,
        suite_revision=projection.suite_revision,
        evidence_policy_revision=projection.evidence_policy_revision,
        pricing_profile_fingerprint=projection.pricing_profile_fingerprint,
        trial_policy_revision=projection.trial_policy_revision,
        accepted_exposure_revision=projection.accepted_exposure_revision,
        accepted_exposure_comparison_revision=(projection.accepted_exposure_comparison_revision),
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
        origin=projection.origin,
        target_key=projection.target.target_key,
        external_target_revision=projection.external_target_revision,
        corpus_revision=projection.corpus_revision,
        suite_id=projection.suite_id,
        suite_revision=projection.suite_revision,
        evidence_policy_revision=projection.evidence_policy_revision,
        pricing_profile_fingerprint=projection.pricing_profile_fingerprint,
        trial_policy_revision=projection.trial_policy_revision,
        accepted_exposure_revision=projection.accepted_exposure_revision,
        accepted_exposure_comparison_revision=(projection.accepted_exposure_comparison_revision),
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
    if baseline.trial_policy_revision != current.trial_policy_revision:
        reasons.add(CorpusComparisonReason.TRIAL_POLICY_REVISION_MISMATCH)
    if (
        baseline.origin == current.origin == EvalResultOrigin.FRESH_EXECUTION
        and baseline.accepted_exposure_comparison_revision
        != current.accepted_exposure_comparison_revision
    ):
        reasons.add(CorpusComparisonReason.ACCEPTED_EXPOSURE_CONTRACT_MISMATCH)
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
    baseline: CorpusExecutionResult
    | CapturedEvaluationResultV1
    | EvalResultProjectionV1
    | EvalResultProjectionV2,
    current: CorpusExecutionResult
    | CapturedEvaluationResultV1
    | EvalResultProjectionV1
    | EvalResultProjectionV2,
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
        if case.reliability_change == "regressed":
            regressions.append(
                CorpusExecutionRegression(
                    scope=CorpusRegressionScope.CASE,
                    kind=CorpusRegressionKind.RELIABILITY,
                    case_id=case.case_id,
                    baseline_reliability=case.baseline_reliability,
                    current_reliability=case.current_reliability,
                )
            )
    return tuple(regressions)


def _compare_projected_results(
    baseline: CorpusExecutionResult
    | CapturedEvaluationResultV1
    | EvalResultProjectionV1
    | EvalResultProjectionV2,
    current: CorpusExecutionResult
    | CapturedEvaluationResultV1
    | EvalResultProjectionV1
    | EvalResultProjectionV2,
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
    structured_state, structured_judgments, structured_mismatches = _structured_comparison(
        baseline,
        current,
        comparable=compatibility.comparable,
        score_tolerance=tolerance,
    )
    tool_json_state, tool_json_assertions, tool_json_mismatches = _tool_json_comparison(
        baseline,
        current,
        comparable=compatibility.comparable,
    )
    if not compatibility.comparable:
        return CorpusExecutionComparison(
            compatibility=compatibility,
            score_tolerance=tolerance,
            baseline=baseline_summary,
            current=current_summary,
            structured_judge_comparison_state=structured_state,
            structured_judgments=structured_judgments,
            structured_judge_observation_mismatches=structured_mismatches,
            tool_json_comparison_state=tool_json_state,
            tool_json_assertions=tool_json_assertions,
            tool_json_observation_mismatches=tool_json_mismatches,
        )
    compared_cases: list[CorpusCaseComparison] = []
    for baseline_case, current_case in zip(
        baseline_projection.cases,
        current_projection.cases,
        strict=True,
    ):
        baseline_reliability = CorpusReliabilityDistributionV1.from_diagnostic_codes(
            baseline_case.trial_diagnostic_codes
        )
        current_reliability = CorpusReliabilityDistributionV1.from_diagnostic_codes(
            current_case.trial_diagnostic_codes
        )
        compared_cases.append(
            CorpusCaseComparison(
                case_id=baseline_case.case_id,
                baseline_status=baseline_case.status,
                current_status=current_case.status,
                baseline_score=baseline_case.score,
                current_score=current_case.score,
                baseline_trial_diagnostic_codes=baseline_case.trial_diagnostic_codes,
                current_trial_diagnostic_codes=current_case.trial_diagnostic_codes,
                baseline_reliability=baseline_reliability,
                current_reliability=current_reliability,
                reliability_change=_reliability_change(
                    baseline_reliability,
                    current_reliability,
                ),
            )
        )
    cases = tuple(compared_cases)
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
        structured_judge_comparison_state=structured_state,
        structured_judgments=structured_judgments,
        structured_judge_observation_mismatches=structured_mismatches,
        tool_json_comparison_state=tool_json_state,
        tool_json_assertions=tool_json_assertions,
        tool_json_observation_mismatches=tool_json_mismatches,
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
    baseline: CorpusExecutionResult
    | CapturedEvaluationResultV1
    | EvalResultProjectionV1
    | EvalResultProjectionV2,
    current: CorpusExecutionResult
    | CapturedEvaluationResultV1
    | EvalResultProjectionV1
    | EvalResultProjectionV2,
    *,
    score_tolerance: float = 0.0,
) -> CorpusExecutionComparison:
    """Compare captured and fresh results through their shared public projection."""

    return _compare_projected_results(
        baseline,
        current,
        score_tolerance=score_tolerance,
    )
