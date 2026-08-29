from __future__ import annotations

import json
from collections.abc import Iterable
from decimal import Decimal
from functools import partial
from typing import Literal, cast

from pydantic import (
    BaseModel,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from cayu._validation import (
    copy_durable_json_object,
    durable_json_object_from_pairs,
    json_utf8_size_within_limit,
    parse_durable_json_integer_literal,
    reject_nonportable_json_constant,
)
from cayu.evals.corpus import (
    EVAL_CORPUS_MAX_ASSERTIONS_PER_CASE,
    EVAL_CORPUS_MAX_CASES,
    EVAL_CORPUS_MAX_JUDGE_CRITERIA,
    EVAL_CORPUS_MAX_TRIALS,
    _bounded_durable_text,
    _canonical_decimal_text,
    _exact_weighted_decimal,
    _ordered_sequence_input,
    _portable_id,
    _PortableModel,
    _sha256_revision,
)
from cayu.evals.evidence import _canonical_decimal
from cayu.evals.execution import CorpusExecutionResult
from cayu.evals.published import (
    PublishedArtifactDetail,
    PublishedAssertionResult,
    PublishedChildStatusDetail,
    PublishedEvalTrialResult,
    PublishedModelJudgeDetail,
    PublishedOutcome,
    PublishedProcessEventDetail,
    PublishedProcessEventsInOrderDetail,
    PublishedStatus,
    PublishedStructuredModelJudgeDetail,
    PublishedToolArgumentsContainDetail,
    PublishedToolResultContainsDetail,
    PublishedWorkspaceFileDetail,
    _published_score,
    _published_status_from_outcomes,
    _published_status_from_statuses,
)
from cayu.evals.result_contract import EvalTrialDiagnosticCode
from cayu.evals.results import CapturedEvaluationResultV1, EvalResultOrigin

EVAL_RESULT_PRESENTATION_SCHEMA_VERSION = 1
EVAL_RESULT_PRESENTATION_MAX_BYTES = 32 << 20
EVAL_RESULT_REPORT_SCHEMA_VERSION = 1
EVAL_RESULT_REPORT_MAX_BYTES = 96 << 20
_EVAL_RESULT_REPORT_UTF8_SCAN_CHARS = 64 << 10

EvalCandidateOutcome = Literal["passed", "failed", "not_scored"]
EvalAssertionDimension = Literal["passed", "failed", "unavailable", "error", "not_used"]
EvalEvaluatorHealth = Literal["healthy", "unavailable", "error", "not_used"]
EvalRuntimeOutcome = Literal["completed", "failed", "unavailable", "not_executed"]
EvalEvidenceState = Literal["complete", "incomplete", "unavailable"]
EvalAssertionCategory = Literal["deterministic", "semantic"]


class EvalResultOutcomeDimensionsV1(_PortableModel):
    """Orthogonal result states that must not be collapsed into one opaque status."""

    candidate: EvalCandidateOutcome
    deterministic_assertions: EvalAssertionDimension
    semantic_quality: EvalAssertionDimension
    evaluator_health: EvalEvaluatorHealth
    runtime: EvalRuntimeOutcome
    evidence: EvalEvidenceState


class EvalStructuredJudgeCriterionPresentationV1(_PortableModel):
    criterion_id: StrictStr
    weight: StrictStr
    score: StrictStr
    weighted_contribution: StrictStr
    explanation: StrictStr | None
    explanation_state: Literal["available", "redacted", "unavailable"]

    @field_validator("criterion_id")
    @classmethod
    def validate_criterion_id(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("weight", "score", "weighted_contribution")
    @classmethod
    def validate_decimal(cls, value: str, info) -> str:
        value = _canonical_decimal_text(value, info.field_name, max_chars=64)
        if Decimal(value) > 1:
            raise ValueError(f"{info.field_name} must be between 0 and 1.")
        return value


class EvalStructuredJudgePresentationV1(_PortableModel):
    """Decision-useful safe view of one admitted structured judgment."""

    detail: PublishedStructuredModelJudgeDetail
    threshold_passed: StrictBool | None
    criteria: tuple[EvalStructuredJudgeCriterionPresentationV1, ...] = Field(
        default=(),
        max_length=EVAL_CORPUS_MAX_JUDGE_CRITERIA,
    )

    @field_validator("detail", mode="before")
    @classmethod
    def copy_detail(cls, value: object) -> object:
        if type(value) is PublishedStructuredModelJudgeDetail:
            return value.model_dump(mode="python", round_trip=True, warnings="none")
        if isinstance(value, BaseModel):
            raise TypeError(
                "detail must be an exact PublishedStructuredModelJudgeDetail or JSON object."
            )
        return value

    @field_validator("criteria", mode="before")
    @classmethod
    def validate_criteria_order(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @field_validator("criteria")
    @classmethod
    def validate_criteria_match_detail(
        cls,
        value: tuple[EvalStructuredJudgeCriterionPresentationV1, ...],
        info,
    ) -> tuple[EvalStructuredJudgeCriterionPresentationV1, ...]:
        detail = info.data.get("detail")
        if type(detail) is not PublishedStructuredModelJudgeDetail:
            return value
        expected = tuple(
            (
                item.criterion_id,
                item.weight,
                item.score,
                _canonical_decimal(
                    _exact_weighted_decimal(((item.weight, Decimal(item.score)),)),
                    max_chars=64,
                ),
                item.explanation,
                item.explanation_state,
            )
            for item in detail.criteria
        )
        actual = tuple(
            (
                item.criterion_id,
                item.weight,
                item.score,
                item.weighted_contribution,
                item.explanation,
                item.explanation_state,
            )
            for item in value
        )
        if actual != expected:
            raise ValueError("Presented criteria do not match the admitted structured judgment.")
        return value

    @field_validator("threshold_passed")
    @classmethod
    def validate_threshold_passed(cls, value: bool | None, info) -> bool | None:
        detail = info.data.get("detail")
        if type(detail) is not PublishedStructuredModelJudgeDetail:
            return value
        expected = (
            None
            if detail.aggregate_score is None
            else Decimal(detail.aggregate_score) >= Decimal(detail.threshold)
        )
        if value != expected:
            raise ValueError("Presented threshold outcome contradicts the structured judgment.")
        return value


class EvalAssertionPresentationV1(_PortableModel):
    assertion_id: StrictStr
    assertion_revision: StrictStr
    kind: StrictStr
    category: EvalAssertionCategory
    outcome: PublishedOutcome
    score: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    structured_judge: EvalStructuredJudgePresentationV1 | None = None
    tool_json: PublishedToolArgumentsContainDetail | PublishedToolResultContainsDetail | None = None
    process: (
        PublishedChildStatusDetail
        | PublishedProcessEventDetail
        | PublishedProcessEventsInOrderDetail
        | None
    ) = None
    structure: PublishedWorkspaceFileDetail | PublishedArtifactDetail | None = None

    @field_validator("tool_json", mode="before")
    @classmethod
    def copy_tool_json(cls, value: object) -> object:
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
            raise TypeError("tool_json must be an exact published tool-JSON detail or JSON object.")
        return value

    @field_validator("process", mode="before")
    @classmethod
    def copy_process(cls, value: object) -> object:
        if type(value) in {
            PublishedChildStatusDetail,
            PublishedProcessEventDetail,
            PublishedProcessEventsInOrderDetail,
        }:
            detail = cast(
                "PublishedChildStatusDetail | PublishedProcessEventDetail | PublishedProcessEventsInOrderDetail",
                value,
            )
            return detail.model_dump(mode="python", round_trip=True, warnings="none")
        if isinstance(value, BaseModel):
            raise TypeError("process must be an exact published process detail or JSON object.")
        return value

    @field_validator("structure", mode="before")
    @classmethod
    def copy_structure(cls, value: object) -> object:
        if type(value) in {PublishedWorkspaceFileDetail, PublishedArtifactDetail}:
            detail = cast("PublishedWorkspaceFileDetail | PublishedArtifactDetail", value)
            return detail.model_dump(mode="python", round_trip=True, warnings="none")
        if isinstance(value, BaseModel):
            raise TypeError("structure must be an exact published structure detail or JSON object.")
        return value

    @field_validator("assertion_id")
    @classmethod
    def validate_assertion_id(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("assertion_revision")
    @classmethod
    def validate_assertion_revision(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("kind")
    @classmethod
    def validate_kind(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=64,
            nonblank=True,
            clean=True,
        )

    @model_validator(mode="after")
    def validate_assertion(self) -> EvalAssertionPresentationV1:
        semantic = self.kind in {"model_judge", "structured_model_judge"}
        if (self.category == "semantic") != semantic:
            raise ValueError("Assertion category contradicts its public kind.")
        if (self.kind == "structured_model_judge") != (self.structured_judge is not None):
            raise ValueError("Only structured-model-judge assertions carry structured detail.")
        tool_json_kind = self.kind in {"tool_arguments_contain", "tool_result_contains"}
        if tool_json_kind != (self.tool_json is not None):
            raise ValueError("Only tool-JSON assertions carry safe tool-JSON detail.")
        if self.tool_json is not None and self.tool_json.kind != self.kind:
            raise ValueError("Tool-JSON presentation kind contradicts its retained detail.")
        process_kind = self.kind in {
            "child_status",
            "process_event",
            "process_events_in_order",
        }
        if process_kind != (self.process is not None):
            raise ValueError("Only process assertions carry safe process detail.")
        if self.process is not None and self.process.kind != self.kind:
            raise ValueError("Process presentation kind contradicts its retained detail.")
        structure_kind = self.kind in {"workspace_file", "artifact"}
        if structure_kind != (self.structure is not None):
            raise ValueError("Only structural assertions carry safe structure detail.")
        if self.structure is not None and self.structure.kind != self.kind:
            raise ValueError("Structure presentation kind contradicts its retained detail.")
        if self.structured_judge is None:
            return self
        detail = self.structured_judge.detail
        if detail.diagnostic == "evidence_unavailable":
            expected_outcome: PublishedOutcome = "unavailable"
        elif detail.diagnostic == "evaluator_error":
            expected_outcome = "error"
        else:
            expected_outcome = (
                "passed"
                if Decimal(detail.aggregate_score or "0") >= Decimal(detail.threshold)
                else "failed"
            )
        expected_score = (
            None if detail.aggregate_score is None else float(Decimal(detail.aggregate_score))
        )
        if self.outcome != expected_outcome or self.score != expected_score:
            raise ValueError("Structured assertion outcome contradicts its admitted judgment.")
        return self


class EvalTrialPresentationV1(_PortableModel):
    trial_number: StrictInt | None = Field(default=None, ge=1, le=EVAL_CORPUS_MAX_TRIALS)
    status: PublishedStatus
    score: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    diagnostic_code: EvalTrialDiagnosticCode | None = None
    dimensions: EvalResultOutcomeDimensionsV1
    assertions: tuple[EvalAssertionPresentationV1, ...] = Field(
        min_length=1,
        max_length=EVAL_CORPUS_MAX_ASSERTIONS_PER_CASE,
    )

    @field_validator("assertions", mode="before")
    @classmethod
    def validate_assertions_order(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @model_validator(mode="after")
    def validate_trial(self) -> EvalTrialPresentationV1:
        assertion_ids = tuple(item.assertion_id for item in self.assertions)
        if len(assertion_ids) != len(set(assertion_ids)):
            raise ValueError("Presented trial assertion identities must be unique.")
        expected_status = _published_status_from_outcomes(
            assertion.outcome for assertion in self.assertions
        )
        expected_score = _published_score(assertion.score for assertion in self.assertions)
        if self.status != expected_status or self.score != expected_score:
            raise ValueError("Presented trial aggregates contradict its assertions.")
        if self.trial_number is None:
            if self.diagnostic_code is not None or self.dimensions.runtime != "not_executed":
                raise ValueError("Captured presentation cannot carry fresh runtime identity.")
        elif self.diagnostic_code is None or self.dimensions.runtime != _fresh_runtime(
            self.diagnostic_code
        ):
            raise ValueError("Fresh presentation runtime contradicts its diagnostic.")
        expected_dimensions = _presented_dimensions(
            self.assertions,
            runtime=self.dimensions.runtime,
            evidence=self.dimensions.evidence,
        )
        if self.dimensions != expected_dimensions:
            raise ValueError("Presented trial outcome dimensions contradict its assertions.")
        return self


class EvalCasePresentationV1(_PortableModel):
    case_id: StrictStr
    case_revision: StrictStr
    status: PublishedStatus
    score: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    dimensions: EvalResultOutcomeDimensionsV1
    trials: tuple[EvalTrialPresentationV1, ...] = Field(
        min_length=1,
        max_length=EVAL_CORPUS_MAX_TRIALS,
    )

    @field_validator("case_id")
    @classmethod
    def validate_case_id(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("case_revision")
    @classmethod
    def validate_case_revision(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("trials", mode="before")
    @classmethod
    def validate_trials_order(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @model_validator(mode="after")
    def validate_case(self) -> EvalCasePresentationV1:
        trial_numbers = tuple(
            item.trial_number for item in self.trials if item.trial_number is not None
        )
        if len(trial_numbers) != len(set(trial_numbers)):
            raise ValueError("Presented case trial identities must be unique.")
        expected_status = _published_status_from_statuses(trial.status for trial in self.trials)
        expected_score = _published_score(trial.score for trial in self.trials)
        if self.status != expected_status or self.score != expected_score:
            raise ValueError("Presented case aggregates contradict its trials.")
        if self.dimensions != _combine_dimensions(trial.dimensions for trial in self.trials):
            raise ValueError("Presented case dimensions contradict its trials.")
        return self


class EvalResultPresentationV1(_PortableModel):
    """Canonical explainable projection for one immutable public eval result."""

    schema_version: Literal[1] = EVAL_RESULT_PRESENTATION_SCHEMA_VERSION
    result_revision: StrictStr
    evaluation_revision: StrictStr
    origin: EvalResultOrigin
    target_key: StrictStr
    application_release_id: StrictStr
    app_manifest_fingerprint: StrictStr
    corpus_revision: StrictStr
    suite_id: StrictStr
    suite_revision: StrictStr
    evidence_policy_revision: StrictStr
    pricing_profile_fingerprint: StrictStr | None
    status: PublishedStatus
    score: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    dimensions: EvalResultOutcomeDimensionsV1
    cases: tuple[EvalCasePresentationV1, ...] = Field(
        min_length=1,
        max_length=EVAL_CORPUS_MAX_CASES,
    )

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1.")
        return value

    @field_validator(
        "result_revision",
        "evaluation_revision",
        "corpus_revision",
        "suite_revision",
        "evidence_policy_revision",
        "pricing_profile_fingerprint",
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
    def validate_application_release_id(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=256,
            nonblank=True,
            clean=True,
        )

    @field_validator("app_manifest_fingerprint")
    @classmethod
    def validate_app_manifest_fingerprint(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("app_manifest_fingerprint must be a lowercase SHA-256 digest.")
        return value

    @field_validator("cases", mode="before")
    @classmethod
    def validate_cases_order(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @model_validator(mode="after")
    def validate_result_size(self) -> EvalResultPresentationV1:
        case_ids = tuple(item.case_id for item in self.cases)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Presented result case identities must be unique.")
        if self.origin == EvalResultOrigin.CAPTURED_SESSION:
            if len(self.cases) != 1 or any(
                trial.trial_number is not None for case in self.cases for trial in case.trials
            ):
                raise ValueError("Captured presentation requires one identity-free retained trial.")
        elif any(trial.trial_number is None for case in self.cases for trial in case.trials):
            raise ValueError("Fresh presentation requires every retained trial identity.")
        expected_status = _published_status_from_statuses(case.status for case in self.cases)
        expected_score = _published_score(case.score for case in self.cases)
        if self.status != expected_status or self.score != expected_score:
            raise ValueError("Presented result aggregates contradict its cases.")
        if self.dimensions != _combine_dimensions(case.dimensions for case in self.cases):
            raise ValueError("Presented result dimensions contradict its cases.")
        if not json_utf8_size_within_limit(self, EVAL_RESULT_PRESENTATION_MAX_BYTES):
            raise ValueError(
                "Eval result presentation exceeds "
                f"{EVAL_RESULT_PRESENTATION_MAX_BYTES} canonical JSON bytes."
            )
        return self


class EvalResultReportV1(_PortableModel):
    """Portable report that binds immutable source evidence to its canonical view."""

    schema_version: Literal[1] = EVAL_RESULT_REPORT_SCHEMA_VERSION
    record_type: Literal["cayu.eval-result-report"] = "cayu.eval-result-report"
    result: CorpusExecutionResult | CapturedEvaluationResultV1
    presentation: EvalResultPresentationV1

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1.")
        return value

    @field_validator("result", mode="before")
    @classmethod
    def copy_result(cls, value: object) -> object:
        if type(value) is CorpusExecutionResult:
            return value.model_dump(mode="python", round_trip=True, warnings="none")
        if type(value) is CapturedEvaluationResultV1:
            return value.model_dump(mode="python", round_trip=True, warnings="none")
        if isinstance(value, BaseModel):
            raise TypeError(
                "result must be an exact CorpusExecutionResult, "
                "CapturedEvaluationResultV1, or JSON object."
            )
        return value

    @model_validator(mode="after")
    def validate_report(self) -> EvalResultReportV1:
        if self.presentation != present_eval_result(self.result):
            raise ValueError("Eval result report presentation does not match its result.")
        if not json_utf8_size_within_limit(self, EVAL_RESULT_REPORT_MAX_BYTES):
            raise ValueError(
                f"Eval result report exceeds {EVAL_RESULT_REPORT_MAX_BYTES} canonical JSON bytes."
            )
        return self


_SEMANTIC_DETAIL_TYPES = (PublishedModelJudgeDetail, PublishedStructuredModelJudgeDetail)
_RUNTIME_FAILED_CODES = {
    EvalTrialDiagnosticCode.EXTERNAL_TARGET_FAILED,
    EvalTrialDiagnosticCode.EXECUTION_FAILED,
    EvalTrialDiagnosticCode.SESSION_FAILED,
    EvalTrialDiagnosticCode.CASE_TIMEOUT,
}
_RUNTIME_UNAVAILABLE_CODES = {
    EvalTrialDiagnosticCode.EXTERNAL_TARGET_UNAVAILABLE,
    EvalTrialDiagnosticCode.EXTERNAL_TARGET_CANCELLED,
    EvalTrialDiagnosticCode.EXTERNAL_TARGET_UNKNOWN,
    EvalTrialDiagnosticCode.EXTERNAL_TARGET_INCOMPLETE,
    EvalTrialDiagnosticCode.EXTERNAL_TARGET_IDENTITY_MISMATCH,
}


def _aggregate_assertion_dimension(outcomes: Iterable[PublishedOutcome]) -> EvalAssertionDimension:
    values = tuple(outcomes)
    if not values:
        return "not_used"
    for state in ("error", "unavailable", "failed", "passed"):
        if state in values:
            return cast("EvalAssertionDimension", state)
    raise AssertionError("Unreachable assertion outcome.")


def _evaluator_health(assertions: Iterable[PublishedAssertionResult]) -> EvalEvaluatorHealth:
    diagnostics = tuple(
        assertion.detail.diagnostic
        for assertion in assertions
        if isinstance(assertion.detail, _SEMANTIC_DETAIL_TYPES)
    )
    if not diagnostics:
        return "not_used"
    if "evaluator_error" in diagnostics:
        return "error"
    if "evidence_unavailable" in diagnostics:
        return "unavailable"
    return "healthy"


def _candidate_outcome(
    *,
    runtime: EvalRuntimeOutcome,
    deterministic: EvalAssertionDimension,
    semantic: EvalAssertionDimension,
) -> EvalCandidateOutcome:
    if runtime in {"failed", "unavailable"}:
        return "not_scored"
    used = tuple(value for value in (deterministic, semantic) if value != "not_used")
    if any(value in {"error", "unavailable"} for value in used):
        return "not_scored"
    if "failed" in used:
        return "failed"
    if used and all(value == "passed" for value in used):
        return "passed"
    return "not_scored"


def _dimensions(
    assertions: tuple[PublishedAssertionResult, ...],
    *,
    runtime: EvalRuntimeOutcome,
    evidence: EvalEvidenceState,
) -> EvalResultOutcomeDimensionsV1:
    deterministic = _aggregate_assertion_dimension(
        assertion.outcome
        for assertion in assertions
        if not isinstance(assertion.detail, _SEMANTIC_DETAIL_TYPES)
    )
    semantic = _aggregate_assertion_dimension(
        assertion.outcome
        for assertion in assertions
        if isinstance(assertion.detail, _SEMANTIC_DETAIL_TYPES)
    )
    return EvalResultOutcomeDimensionsV1(
        candidate=_candidate_outcome(
            runtime=runtime,
            deterministic=deterministic,
            semantic=semantic,
        ),
        deterministic_assertions=deterministic,
        semantic_quality=semantic,
        evaluator_health=_evaluator_health(assertions),
        runtime=runtime,
        evidence=evidence,
    )


def _presented_dimensions(
    assertions: tuple[EvalAssertionPresentationV1, ...],
    *,
    runtime: EvalRuntimeOutcome,
    evidence: EvalEvidenceState,
) -> EvalResultOutcomeDimensionsV1:
    deterministic = _aggregate_assertion_dimension(
        assertion.outcome for assertion in assertions if assertion.category == "deterministic"
    )
    semantic_assertions = tuple(
        assertion for assertion in assertions if assertion.category == "semantic"
    )
    semantic = _aggregate_assertion_dimension(
        assertion.outcome for assertion in semantic_assertions
    )
    if not semantic_assertions:
        evaluator: EvalEvaluatorHealth = "not_used"
    elif any(assertion.outcome == "error" for assertion in semantic_assertions):
        evaluator = "error"
    elif any(assertion.outcome == "unavailable" for assertion in semantic_assertions):
        evaluator = "unavailable"
    else:
        evaluator = "healthy"
    return EvalResultOutcomeDimensionsV1(
        candidate=_candidate_outcome(
            runtime=runtime,
            deterministic=deterministic,
            semantic=semantic,
        ),
        deterministic_assertions=deterministic,
        semantic_quality=semantic,
        evaluator_health=evaluator,
        runtime=runtime,
        evidence=evidence,
    )


def _present_structured_judge(
    detail: PublishedStructuredModelJudgeDetail,
) -> EvalStructuredJudgePresentationV1:
    validated = PublishedStructuredModelJudgeDetail.model_validate(
        detail.model_dump(mode="python", round_trip=True, warnings="none")
    )
    criteria = tuple(
        EvalStructuredJudgeCriterionPresentationV1(
            criterion_id=item.criterion_id,
            weight=item.weight,
            score=item.score,
            weighted_contribution=_canonical_decimal(
                _exact_weighted_decimal(((item.weight, Decimal(item.score)),)),
                max_chars=64,
            ),
            explanation=item.explanation,
            explanation_state=item.explanation_state,
        )
        for item in validated.criteria
    )
    return EvalStructuredJudgePresentationV1(
        detail=validated,
        threshold_passed=(
            None
            if validated.aggregate_score is None
            else Decimal(validated.aggregate_score) >= Decimal(validated.threshold)
        ),
        criteria=criteria,
    )


def _present_assertion(assertion: PublishedAssertionResult) -> EvalAssertionPresentationV1:
    validated = PublishedAssertionResult.model_validate(
        assertion.model_dump(mode="python", round_trip=True, warnings="none")
    )
    detail = validated.detail
    semantic = isinstance(detail, _SEMANTIC_DETAIL_TYPES)
    tool_json = (
        cast(
            "PublishedToolArgumentsContainDetail | PublishedToolResultContainsDetail",
            detail,
        )
        if type(detail) in {PublishedToolArgumentsContainDetail, PublishedToolResultContainsDetail}
        else None
    )
    process = (
        cast(
            "PublishedChildStatusDetail | PublishedProcessEventDetail | PublishedProcessEventsInOrderDetail",
            detail,
        )
        if type(detail)
        in {
            PublishedChildStatusDetail,
            PublishedProcessEventDetail,
            PublishedProcessEventsInOrderDetail,
        }
        else None
    )
    structure = (
        cast("PublishedWorkspaceFileDetail | PublishedArtifactDetail", detail)
        if type(detail) in {PublishedWorkspaceFileDetail, PublishedArtifactDetail}
        else None
    )
    return EvalAssertionPresentationV1(
        assertion_id=validated.assertion_id,
        assertion_revision=validated.assertion_revision,
        kind=detail.kind,
        category="semantic" if semantic else "deterministic",
        outcome=validated.outcome,
        score=validated.score,
        structured_judge=(
            _present_structured_judge(detail)
            if type(detail) is PublishedStructuredModelJudgeDetail
            else None
        ),
        tool_json=tool_json,
        process=process,
        structure=structure,
    )


def _fresh_runtime(code: EvalTrialDiagnosticCode) -> EvalRuntimeOutcome:
    if code in _RUNTIME_FAILED_CODES:
        return "failed"
    if code in _RUNTIME_UNAVAILABLE_CODES:
        return "unavailable"
    return "completed"


def _present_fresh_trial(trial: PublishedEvalTrialResult) -> EvalTrialPresentationV1:
    validated = PublishedEvalTrialResult.model_validate(
        trial.model_dump(mode="python", round_trip=True, warnings="none")
    )
    assertions = tuple(_present_assertion(assertion) for assertion in validated.assertions)
    runtime = _fresh_runtime(validated.code)
    evidence: EvalEvidenceState = (
        "complete"
        if validated.evidence_complete
        else "unavailable"
        if validated.status == "unavailable"
        else "incomplete"
    )
    return EvalTrialPresentationV1(
        trial_number=validated.trial_number,
        status=validated.status,
        score=validated.score,
        diagnostic_code=validated.code,
        dimensions=_dimensions(
            validated.assertions,
            runtime=runtime,
            evidence=evidence,
        ),
        assertions=assertions,
    )


def _combine_dimension(values: Iterable[str], order: tuple[str, ...]) -> str:
    observed = set(values)
    for value in order:
        if value in observed:
            return value
    raise ValueError("Cannot combine an empty outcome dimension.")


def _combine_dimensions(
    values: Iterable[EvalResultOutcomeDimensionsV1],
) -> EvalResultOutcomeDimensionsV1:
    items = tuple(values)
    if not items:
        raise ValueError("Outcome dimensions require at least one observation.")
    return EvalResultOutcomeDimensionsV1(
        candidate=cast(
            "EvalCandidateOutcome",
            _combine_dimension(
                (item.candidate for item in items),
                ("not_scored", "failed", "passed"),
            ),
        ),
        deterministic_assertions=cast(
            "EvalAssertionDimension",
            _combine_dimension(
                (item.deterministic_assertions for item in items),
                ("error", "unavailable", "failed", "passed", "not_used"),
            ),
        ),
        semantic_quality=cast(
            "EvalAssertionDimension",
            _combine_dimension(
                (item.semantic_quality for item in items),
                ("error", "unavailable", "failed", "passed", "not_used"),
            ),
        ),
        evaluator_health=cast(
            "EvalEvaluatorHealth",
            _combine_dimension(
                (item.evaluator_health for item in items),
                ("error", "unavailable", "healthy", "not_used"),
            ),
        ),
        runtime=cast(
            "EvalRuntimeOutcome",
            _combine_dimension(
                (item.runtime for item in items),
                ("failed", "unavailable", "completed", "not_executed"),
            ),
        ),
        evidence=cast(
            "EvalEvidenceState",
            _combine_dimension(
                (item.evidence for item in items),
                ("unavailable", "incomplete", "complete"),
            ),
        ),
    )


def _present_fresh_result(result: CorpusExecutionResult) -> EvalResultPresentationV1:
    validated = CorpusExecutionResult.model_validate(
        result.model_dump(mode="python", round_trip=True, warnings="none")
    )
    cases: list[EvalCasePresentationV1] = []
    for case in validated.run.cases:
        trials = tuple(_present_fresh_trial(trial) for trial in case.trials)
        cases.append(
            EvalCasePresentationV1(
                case_id=case.case_id,
                case_revision=case.case_revision,
                status=case.status,
                score=case.score,
                dimensions=_combine_dimensions(trial.dimensions for trial in trials),
                trials=trials,
            )
        )
    ordered_cases = tuple(cases)
    presentation = EvalResultPresentationV1(
        result_revision=validated.revision,
        evaluation_revision=validated.run.revision,
        origin=EvalResultOrigin.FRESH_EXECUTION,
        target_key=validated.target.target_key,
        application_release_id=validated.target.application_release_id,
        app_manifest_fingerprint=validated.target.app_manifest_fingerprint,
        corpus_revision=validated.run.corpus_revision,
        suite_id=validated.run.suite_id,
        suite_revision=validated.run.suite_revision,
        evidence_policy_revision=validated.run.evidence_policy_revision,
        pricing_profile_fingerprint=validated.run.pricing_profile_fingerprint,
        status=validated.run.status,
        score=validated.run.score,
        dimensions=_combine_dimensions(case.dimensions for case in ordered_cases),
        cases=ordered_cases,
    )
    return _validate_presentation_size(presentation)


def _present_captured_result(result: CapturedEvaluationResultV1) -> EvalResultPresentationV1:
    validated = CapturedEvaluationResultV1.model_validate(
        result.model_dump(mode="python", round_trip=True, warnings="none")
    )
    score = validated.score
    assertions = tuple(_present_assertion(assertion) for assertion in score.assertions)
    evidence: EvalEvidenceState = (
        "unavailable"
        if any(assertion.outcome == "unavailable" for assertion in score.assertions)
        else "complete"
    )
    dimensions = _dimensions(
        score.assertions,
        runtime="not_executed",
        evidence=evidence,
    )
    trial = EvalTrialPresentationV1(
        status=score.status,
        score=score.score,
        dimensions=dimensions,
        assertions=assertions,
    )
    case = EvalCasePresentationV1(
        case_id=score.case_id,
        case_revision=score.case_revision,
        status=score.status,
        score=score.score,
        dimensions=dimensions,
        trials=(trial,),
    )
    presentation = EvalResultPresentationV1(
        result_revision=validated.revision,
        evaluation_revision=score.revision,
        origin=EvalResultOrigin.CAPTURED_SESSION,
        target_key=validated.target.target_key,
        application_release_id=validated.target.application_release_id,
        app_manifest_fingerprint=validated.target.app_manifest_fingerprint,
        corpus_revision=validated.corpus_revision,
        suite_id=validated.suite_id,
        suite_revision=validated.suite_revision,
        evidence_policy_revision=score.evidence_policy_revision,
        pricing_profile_fingerprint=score.pricing_profile_fingerprint,
        status=score.status,
        score=score.score,
        dimensions=dimensions,
        cases=(case,),
    )
    return _validate_presentation_size(presentation)


def _validate_presentation_size(
    presentation: EvalResultPresentationV1,
) -> EvalResultPresentationV1:
    # Revalidate the complete graph so forged nested instances cannot bypass the
    # public projection's derived-field and byte-bound contracts.
    return EvalResultPresentationV1.model_validate(
        presentation.model_dump(mode="python", round_trip=True, warnings="none")
    )


def present_eval_result(
    result: CorpusExecutionResult | CapturedEvaluationResultV1,
) -> EvalResultPresentationV1:
    """Compile one immutable public result into its canonical explainable view."""

    if type(result) is CorpusExecutionResult:
        return _present_fresh_result(result)
    if type(result) is CapturedEvaluationResultV1:
        return _present_captured_result(result)
    raise TypeError("result must be an exact CorpusExecutionResult or CapturedEvaluationResultV1.")


def _validate_report_text_utf8_size(source: str) -> None:
    if len(source) > EVAL_RESULT_REPORT_MAX_BYTES:
        raise ValueError(f"Eval result report JSON exceeds {EVAL_RESULT_REPORT_MAX_BYTES} bytes.")
    encoded_size = 0
    try:
        for start in range(0, len(source), _EVAL_RESULT_REPORT_UTF8_SCAN_CHARS):
            chunk = source[start : start + _EVAL_RESULT_REPORT_UTF8_SCAN_CHARS]
            encoded_size += len(chunk.encode("utf-8"))
            if encoded_size > EVAL_RESULT_REPORT_MAX_BYTES:
                raise ValueError(
                    f"Eval result report JSON exceeds {EVAL_RESULT_REPORT_MAX_BYTES} bytes."
                )
    except UnicodeEncodeError as exc:
        raise ValueError("Eval result report JSON must contain valid Unicode scalar text.") from exc


def eval_result_report_from_json(source: str | bytes | bytearray) -> EvalResultReportV1:
    """Parse one bounded explainable-result report without format guessing."""

    if type(source) is str:
        _validate_report_text_utf8_size(source)
        text = source
    elif type(source) is bytes or type(source) is bytearray:
        if len(source) > EVAL_RESULT_REPORT_MAX_BYTES:
            raise ValueError(
                f"Eval result report JSON exceeds {EVAL_RESULT_REPORT_MAX_BYTES} bytes."
            )
        try:
            text = source.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Eval result report JSON must be UTF-8.") from exc
    else:
        raise TypeError("Eval result report JSON must be text, bytes, or bytearray.")
    try:
        decoded = json.loads(
            text,
            parse_int=partial(
                parse_durable_json_integer_literal,
                field_name="eval result report JSON",
            ),
            parse_constant=partial(
                reject_nonportable_json_constant,
                field_name="eval result report JSON",
            ),
            object_pairs_hook=partial(
                durable_json_object_from_pairs,
                field_name="eval result report JSON",
            ),
        )
    except RecursionError as exc:
        raise ValueError("Eval result report JSON exceeds the supported nesting depth.") from exc
    document = copy_durable_json_object(decoded, "eval result report JSON")
    raw_version = document.get("schema_version")
    if type(raw_version) is not int or raw_version != EVAL_RESULT_REPORT_SCHEMA_VERSION:
        raise ValueError(
            "Eval result report has unsupported schema_version "
            f"{raw_version!r}; this Cayu version supports only "
            f"{EVAL_RESULT_REPORT_SCHEMA_VERSION}."
        )
    normalized = json.dumps(
        document,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return EvalResultReportV1.model_validate_json(normalized)


__all__ = [
    "EVAL_RESULT_PRESENTATION_MAX_BYTES",
    "EVAL_RESULT_PRESENTATION_SCHEMA_VERSION",
    "EVAL_RESULT_REPORT_MAX_BYTES",
    "EVAL_RESULT_REPORT_SCHEMA_VERSION",
    "EvalAssertionPresentationV1",
    "EvalCasePresentationV1",
    "EvalResultOutcomeDimensionsV1",
    "EvalResultPresentationV1",
    "EvalResultReportV1",
    "EvalStructuredJudgeCriterionPresentationV1",
    "EvalStructuredJudgePresentationV1",
    "EvalTrialPresentationV1",
    "eval_result_report_from_json",
    "present_eval_result",
]
