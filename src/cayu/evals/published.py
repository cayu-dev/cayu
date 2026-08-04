from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Annotated, Any, Literal, TypeAlias, cast

from pydantic import (
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from cayu._validation import MAX_DURABLE_JSON_INTEGER, json_utf8_size_within_limit
from cayu.evals.corpus import (
    _CURRENCY_PATTERN,
    EVAL_CORPUS_MAX_ASSERTIONS_PER_CASE,
    EVAL_CORPUS_MAX_CASES,
    EVAL_CORPUS_MAX_PUBLISHED_ASSERTION_RESULTS,
    EVAL_CORPUS_MAX_TOOL_NAMES,
    EVAL_CORPUS_MAX_TRIALS,
    EVIDENCE_MAX_CHILD_SESSIONS,
    EVIDENCE_MAX_MODEL_STEPS,
    EVIDENCE_MAX_TOOL_CALLS,
    EVIDENCE_MAX_TOTAL_TOKENS,
    AssertionSpec,
    ChildStatusAssertionSpec,
    EvalCaseSpec,
    EvalCorpusDocument,
    FinalOutputContainsAssertionSpec,
    FinalOutputEqualsAssertionSpec,
    MaxEstimatedCostAssertionSpec,
    MaxModelStepsAssertionSpec,
    MaxToolCallsAssertionSpec,
    MaxTotalTokensAssertionSpec,
    RootStatusAssertionSpec,
    ToolCalledAssertionSpec,
    ToolsCalledInOrderAssertionSpec,
    UsageRecordedAssertionSpec,
    _bounded_durable_text,
    _canonical_decimal_text,
    _content_revision,
    _model_content_revision,
    _ordered_sequence_input,
    _portable_id,
    _PortableModel,
    _SchemaV1PortableModel,
    _sha256_revision,
    assertion_spec_revision,
    eval_run_contract_for_corpus,
)
from cayu.evals.evidence import _canonical_decimal
from cayu.evals.models import (
    EvalAssertionResult,
    EvalRun,
    _model_instance_python_input,
)
from cayu.runtime.usage import AggregateCount, aggregate_usage_metrics_from_durable_payload

PUBLISHED_EVAL_SCHEMA_VERSION = 1
PUBLISHED_EVAL_MAX_BYTES = 32 << 20
PUBLISHED_EVAL_MAX_DURATION_MS = 2**63 - 1

PublishedStatus = Literal["passed", "failed", "unavailable", "error"]
PublishedOutcome = Literal["passed", "failed", "unavailable", "error"]

_ASSERTION_MESSAGE = {
    "passed": "Assertion passed.",
    "failed": "Assertion failed.",
    "unavailable": "Required evidence was unavailable.",
    "error": "Assertion evaluation failed.",
}
_TRIAL_MESSAGE = {
    "passed": "Trial passed.",
    "failed": "One or more assertions failed.",
    "unavailable": "Required trial evidence was unavailable.",
    "error": "Trial execution or assertion evaluation failed.",
}


class _PublishedAssertionDetail(_PortableModel):
    kind: StrictStr


class PublishedRootStatusDetail(_PublishedAssertionDetail):
    kind: Literal["root_status"] = "root_status"
    expected: Literal["completed", "failed"]
    actual: Literal["completed", "failed", "interrupted"] | None = None


class PublishedChildStatusDetail(_PublishedAssertionDetail):
    kind: Literal["child_status"] = "child_status"
    expected: Literal["completed", "failed"]
    min_count: StrictInt = Field(ge=0, le=EVIDENCE_MAX_CHILD_SESSIONS)
    max_count: StrictInt | None = Field(default=None, ge=0, le=EVIDENCE_MAX_CHILD_SESSIONS)
    matching_count: StrictInt | None = Field(default=None, ge=0, le=EVIDENCE_MAX_CHILD_SESSIONS)

    @model_validator(mode="after")
    def validate_count_range(self) -> PublishedChildStatusDetail:
        if self.max_count is not None and self.max_count < self.min_count:
            raise ValueError("max_count must be greater than or equal to min_count.")
        return self


class PublishedFinalOutputEqualsDetail(_PublishedAssertionDetail):
    kind: Literal["final_output_equals"] = "final_output_equals"
    matched: StrictBool | None = None


class PublishedFinalOutputContainsDetail(_PublishedAssertionDetail):
    kind: Literal["final_output_contains"] = "final_output_contains"
    matched: StrictBool | None = None


class PublishedToolCalledDetail(_PublishedAssertionDetail):
    kind: Literal["tool_called"] = "tool_called"
    tool_name: StrictStr
    min_count: StrictInt = Field(ge=0, le=EVIDENCE_MAX_TOOL_CALLS)
    max_count: StrictInt | None = Field(default=None, ge=0, le=EVIDENCE_MAX_TOOL_CALLS)
    matching_count: StrictInt | None = Field(default=None, ge=0, le=EVIDENCE_MAX_TOOL_CALLS)

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=256,
            nonblank=True,
            clean=True,
        )

    @model_validator(mode="after")
    def validate_count_range(self) -> PublishedToolCalledDetail:
        if self.max_count is not None and self.max_count < self.min_count:
            raise ValueError("max_count must be greater than or equal to min_count.")
        return self


class PublishedToolsCalledInOrderDetail(_PublishedAssertionDetail):
    kind: Literal["tools_called_in_order"] = "tools_called_in_order"
    expected_count: StrictInt = Field(ge=0, le=EVAL_CORPUS_MAX_TOOL_NAMES)
    actual_count: StrictInt | None = Field(default=None, ge=0, le=EVIDENCE_MAX_TOOL_CALLS)
    matched: StrictBool | None = None

    @model_validator(mode="after")
    def validate_observation(self) -> PublishedToolsCalledInOrderDetail:
        if (self.actual_count is None) != (self.matched is None):
            raise ValueError("Published tool-order observations must be present together.")
        if self.matched and self.actual_count != self.expected_count:
            raise ValueError("A matching tool order must have the expected call count.")
        if self.matched is False and self.expected_count == self.actual_count == 0:
            raise ValueError("Empty expected and actual tool orders must match.")
        return self


class PublishedMaxToolCallsDetail(_PublishedAssertionDetail):
    kind: Literal["max_tool_calls"] = "max_tool_calls"
    maximum: StrictInt = Field(ge=0, le=EVIDENCE_MAX_TOOL_CALLS)
    actual: StrictInt | None = Field(default=None, ge=0, le=EVIDENCE_MAX_TOOL_CALLS)


class PublishedMaxModelStepsDetail(_PublishedAssertionDetail):
    kind: Literal["max_model_steps"] = "max_model_steps"
    maximum: StrictInt = Field(ge=0, le=EVIDENCE_MAX_MODEL_STEPS)
    actual: StrictInt | None = Field(default=None, ge=0, le=EVIDENCE_MAX_MODEL_STEPS)


class PublishedUsageRecordedDetail(_PublishedAssertionDetail):
    kind: Literal["usage_recorded"] = "usage_recorded"
    minimum: StrictInt = Field(ge=0, le=EVIDENCE_MAX_TOTAL_TOKENS)
    actual: StrictInt | None = Field(default=None, ge=0, le=EVIDENCE_MAX_TOTAL_TOKENS)


class PublishedMaxTotalTokensDetail(_PublishedAssertionDetail):
    kind: Literal["max_total_tokens"] = "max_total_tokens"
    maximum: StrictInt = Field(ge=0, le=EVIDENCE_MAX_TOTAL_TOKENS)
    actual: StrictInt | None = Field(default=None, ge=0, le=EVIDENCE_MAX_TOTAL_TOKENS)


class PublishedMaxEstimatedCostDetail(_PublishedAssertionDetail):
    kind: Literal["max_estimated_cost"] = "max_estimated_cost"
    maximum: StrictStr
    currency: StrictStr
    estimated_cost: StrictStr | None = None
    priced_model_steps: StrictInt | None = Field(default=None, ge=0, le=EVIDENCE_MAX_MODEL_STEPS)
    unpriced_model_steps: StrictInt | None = Field(default=None, ge=0, le=EVIDENCE_MAX_MODEL_STEPS)

    @field_validator("maximum", "estimated_cost")
    @classmethod
    def validate_decimal_text(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _canonical_decimal_text(
            value,
            info.field_name,
            max_chars=64 if info.field_name == "maximum" else 128,
        )

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, value: str, info) -> str:
        value = _bounded_durable_text(
            value,
            info.field_name,
            max_chars=16,
            nonblank=True,
            clean=True,
        )
        if _CURRENCY_PATTERN.fullmatch(value) is None:
            raise ValueError("currency must be a portable uppercase identifier.")
        return value

    @model_validator(mode="after")
    def validate_cost_counts(self) -> PublishedMaxEstimatedCostDetail:
        observations = (
            self.estimated_cost,
            self.priced_model_steps,
            self.unpriced_model_steps,
        )
        if any(item is None for item in observations) and any(
            item is not None for item in observations
        ):
            raise ValueError("Published cost observations must be present together.")
        if (
            self.priced_model_steps is not None
            and self.unpriced_model_steps is not None
            and self.priced_model_steps + self.unpriced_model_steps > EVIDENCE_MAX_MODEL_STEPS
        ):
            raise ValueError("Published cost observations exceed the model-step evidence bound.")
        return self


PublishedAssertionDetail: TypeAlias = Annotated[
    PublishedRootStatusDetail
    | PublishedChildStatusDetail
    | PublishedFinalOutputEqualsDetail
    | PublishedFinalOutputContainsDetail
    | PublishedToolCalledDetail
    | PublishedToolsCalledInOrderDetail
    | PublishedMaxToolCallsDetail
    | PublishedMaxModelStepsDetail
    | PublishedUsageRecordedDetail
    | PublishedMaxTotalTokensDetail
    | PublishedMaxEstimatedCostDetail,
    Field(discriminator="kind"),
]


class PublishedAssertionResult(_PortableModel):
    assertion_id: StrictStr
    assertion_revision: StrictStr
    outcome: PublishedOutcome
    score: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    code: PublishedOutcome
    message: StrictStr
    detail: PublishedAssertionDetail

    @field_validator("assertion_id")
    @classmethod
    def validate_id(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("assertion_revision")
    @classmethod
    def validate_revision(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @model_validator(mode="after")
    def validate_contract(self) -> PublishedAssertionResult:
        if self.outcome in {"unavailable", "error"}:
            if self.score is not None:
                raise ValueError("Unavailable/error published assertions cannot have a score.")
        elif self.score != (1.0 if self.outcome == "passed" else 0.0):
            raise ValueError("Published deterministic assertion score is inconsistent.")
        if self.code != self.outcome or self.message != _ASSERTION_MESSAGE[self.outcome]:
            raise ValueError("Published assertion diagnostics do not match the outcome.")
        if self.outcome in {"passed", "failed"} and not _detail_has_observation(self.detail):
            raise ValueError("Scored published assertions require their observed evidence.")
        if self.outcome == "error" and _detail_has_observation(self.detail):
            raise ValueError("Errored published assertions cannot carry observed evidence.")
        if self.outcome == "unavailable" and _detail_has_observation(self.detail):
            if not isinstance(self.detail, PublishedMaxEstimatedCostDetail):
                raise ValueError(
                    "Unavailable published assertions cannot carry conclusive observations."
                )
            if not self.detail.unpriced_model_steps:
                raise ValueError(
                    "Unavailable published cost observations require unpriced model steps."
                )
        if (
            self.outcome in {"passed", "failed"}
            and isinstance(self.detail, PublishedMaxEstimatedCostDetail)
            and self.detail.unpriced_model_steps != 0
        ):
            raise ValueError("Scored published cost assertions require fully priced evidence.")
        expected_passed = _detail_expected_pass(self.detail)
        if self.outcome in {"passed", "failed"} and expected_passed is not None:
            expected_outcome = "passed" if expected_passed else "failed"
            if self.outcome != expected_outcome:
                raise ValueError("Published assertion outcome contradicts its detail evidence.")
        return self


class PublishedUsageSummaryV1(_PortableModel):
    model_steps: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    tool_calls: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    total_tokens: AggregateCount = Field(ge=0)


class PublishedEvalTrialResult(_PortableModel):
    trial_number: StrictInt = Field(ge=1, le=EVAL_CORPUS_MAX_TRIALS)
    status: PublishedStatus
    score: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    assertions: tuple[PublishedAssertionResult, ...] = Field(
        min_length=1,
        max_length=EVAL_CORPUS_MAX_ASSERTIONS_PER_CASE,
    )
    evidence_complete: StrictBool
    duration_ms: StrictInt = Field(ge=0, le=PUBLISHED_EVAL_MAX_DURATION_MS)
    usage: PublishedUsageSummaryV1 | None = None
    code: PublishedStatus
    message: StrictStr

    @field_validator("assertions", mode="before")
    @classmethod
    def validate_assertions_are_ordered(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @model_validator(mode="after")
    def validate_contract(self) -> PublishedEvalTrialResult:
        assertion_ids = tuple(assertion.assertion_id for assertion in self.assertions)
        if len(assertion_ids) != len(set(assertion_ids)):
            raise ValueError("Published assertion IDs must be unique within a trial.")
        expected_status = _published_status_from_outcomes(
            assertion.outcome for assertion in self.assertions
        )
        expected_score = _published_score(assertion.score for assertion in self.assertions)
        if self.status != expected_status or self.score != expected_score:
            raise ValueError("Published trial aggregates do not match its assertions.")
        if self.status in {"passed", "failed"} and not self.evidence_complete:
            raise ValueError("Scored published trials require complete evidence.")
        if self.evidence_complete and self.usage is None:
            raise ValueError("Complete published trials require exact usage.")
        if self.code != self.status or self.message != _TRIAL_MESSAGE[self.status]:
            raise ValueError("Published trial diagnostics do not match the status.")
        _validate_trial_observations(
            self.assertions,
            self.usage,
            evidence_complete=self.evidence_complete,
        )
        return self


class PublishedEvalCaseResult(_PortableModel):
    case_id: StrictStr
    case_revision: StrictStr
    status: PublishedStatus
    score: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    trials: tuple[PublishedEvalTrialResult, ...] = Field(
        min_length=1,
        max_length=EVAL_CORPUS_MAX_TRIALS,
    )
    duration_ms: StrictInt = Field(ge=0, le=PUBLISHED_EVAL_MAX_DURATION_MS)

    @field_validator("trials", mode="before")
    @classmethod
    def validate_trials_are_ordered(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @field_validator("case_id")
    @classmethod
    def validate_id(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("case_revision")
    @classmethod
    def validate_revision(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @model_validator(mode="after")
    def validate_contract(self) -> PublishedEvalCaseResult:
        if tuple(trial.trial_number for trial in self.trials) != tuple(
            range(1, len(self.trials) + 1)
        ):
            raise ValueError("Published trial numbers must be contiguous and ordered.")
        expected_status = _published_status_from_statuses(trial.status for trial in self.trials)
        expected_score = _published_score(trial.score for trial in self.trials)
        if self.status != expected_status or self.score != expected_score:
            raise ValueError("Published case aggregates do not match its trials.")
        if self.duration_ms != sum(trial.duration_ms for trial in self.trials):
            raise ValueError("Published case duration must equal its trial durations.")
        contracts = tuple(
            tuple(_assertion_contract(item) for item in trial.assertions) for trial in self.trials
        )
        if any(contract != contracts[0] for contract in contracts[1:]):
            raise ValueError("Published trials must share one ordered assertion contract.")
        return self


class PublishedEvalRun(_SchemaV1PortableModel):
    schema_version: Literal[1] = PUBLISHED_EVAL_SCHEMA_VERSION
    revision: StrictStr
    corpus_revision: StrictStr
    target_key: StrictStr
    suite_id: StrictStr
    suite_revision: StrictStr
    evidence_policy_revision: StrictStr
    pricing_profile_fingerprint: StrictStr | None = None
    status: PublishedStatus
    score: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    cases: tuple[PublishedEvalCaseResult, ...] = Field(
        min_length=1,
        max_length=EVAL_CORPUS_MAX_CASES,
    )
    duration_ms: StrictInt = Field(ge=0, le=PUBLISHED_EVAL_MAX_DURATION_MS)

    @field_validator("cases", mode="before")
    @classmethod
    def validate_cases_are_ordered(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @model_validator(mode="before")
    @classmethod
    def validate_expanded_result_limit(cls, value):
        """Reject oversized raw graphs before constructing their nested models."""

        def raw_sequence(container, field_name, path, model_type):
            if isinstance(container, Mapping):
                if field_name not in container:
                    raise ValueError(f"{path}.{field_name} is required.")
                sequence = container[field_name]
            elif isinstance(container, model_type):
                sequence = getattr(container, field_name)
            else:
                raise ValueError(f"{path} must be an object.")
            if not isinstance(sequence, list | tuple):
                raise ValueError(f"{path}.{field_name} must be an array.")
            return sequence

        if isinstance(value, Mapping):
            if "cases" not in value:
                raise ValueError("Published eval run cases are required.")
            cases = value["cases"]
        elif isinstance(value, cls):
            cases = value.cases
        else:
            raise ValueError("Published eval run input must be an object.")
        if not isinstance(cases, list | tuple):
            raise ValueError("Published eval run cases must be an array.")
        if not 1 <= len(cases) <= EVAL_CORPUS_MAX_CASES:
            raise ValueError(
                "Published eval run cases must contain between 1 and "
                f"{EVAL_CORPUS_MAX_CASES} items."
            )
        published_assertion_results = 0
        for case_index, case in enumerate(cases):
            trials = raw_sequence(
                case,
                "trials",
                f"Published eval run cases[{case_index}]",
                PublishedEvalCaseResult,
            )
            if not 1 <= len(trials) <= EVAL_CORPUS_MAX_TRIALS:
                raise ValueError(
                    f"Published eval run cases[{case_index}].trials must contain between 1 and "
                    f"{EVAL_CORPUS_MAX_TRIALS} items."
                )
            for trial_index, trial in enumerate(trials):
                assertions = raw_sequence(
                    trial,
                    "assertions",
                    f"Published eval run cases[{case_index}].trials[{trial_index}]",
                    PublishedEvalTrialResult,
                )
                if not 1 <= len(assertions) <= EVAL_CORPUS_MAX_ASSERTIONS_PER_CASE:
                    raise ValueError(
                        "Published eval run "
                        f"cases[{case_index}].trials[{trial_index}].assertions must contain "
                        f"between 1 and {EVAL_CORPUS_MAX_ASSERTIONS_PER_CASE} items."
                    )
                published_assertion_results += len(assertions)
                if published_assertion_results > EVAL_CORPUS_MAX_PUBLISHED_ASSERTION_RESULTS:
                    raise ValueError(
                        "Published eval run exceeds the corpus expanded assertion-result "
                        f"limit of {EVAL_CORPUS_MAX_PUBLISHED_ASSERTION_RESULTS}."
                    )
        return value

    @field_validator(
        "revision",
        "corpus_revision",
        "suite_revision",
        "evidence_policy_revision",
        "pricing_profile_fingerprint",
    )
    @classmethod
    def validate_revision_fields(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _sha256_revision(value, info.field_name)

    @field_validator("target_key", "suite_id")
    @classmethod
    def validate_id(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @model_validator(mode="after")
    def validate_contract(self) -> PublishedEvalRun:
        case_ids = tuple(case.case_id for case in self.cases)
        if case_ids != tuple(sorted(set(case_ids))):
            raise ValueError("Published cases must be unique and sorted by id.")
        expected_status = _published_status_from_statuses(case.status for case in self.cases)
        expected_score = _published_score(case.score for case in self.cases)
        if self.status != expected_status or self.score != expected_score:
            raise ValueError("Published run aggregates do not match its cases.")
        if self.duration_ms != sum(case.duration_ms for case in self.cases):
            raise ValueError("Published run duration must equal its case durations.")
        trial_counts = {len(case.trials) for case in self.cases}
        if len(trial_counts) != 1:
            raise ValueError("Published cases must share one suite-wide trial count.")
        published_assertion_results = sum(
            len(trial.assertions) for case in self.cases for trial in case.trials
        )
        if published_assertion_results > EVAL_CORPUS_MAX_PUBLISHED_ASSERTION_RESULTS:
            raise ValueError(
                "Published eval run exceeds the corpus expanded assertion-result limit of "
                f"{EVAL_CORPUS_MAX_PUBLISHED_ASSERTION_RESULTS}."
            )
        has_cost_assertion = any(
            assertion.detail.kind == "max_estimated_cost"
            for case in self.cases
            for trial in case.trials
            for assertion in trial.assertions
        )
        if has_cost_assertion and self.pricing_profile_fingerprint is None:
            raise ValueError("Published cost assertions require a pricing profile fingerprint.")
        if not json_utf8_size_within_limit(self, PUBLISHED_EVAL_MAX_BYTES):
            raise ValueError(
                f"Published eval run exceeds {PUBLISHED_EVAL_MAX_BYTES} canonical JSON bytes."
            )
        if self.revision != _model_content_revision(self, "published eval run"):
            raise ValueError("Published eval run revision does not match its content.")
        return self


def _published_status_from_outcomes(outcomes) -> PublishedStatus:
    values = tuple(outcomes)
    for outcome in ("error", "unavailable", "failed"):
        if outcome in values:
            return outcome
    return "passed"


def _detail_has_observation(detail: PublishedAssertionDetail) -> bool:
    if isinstance(detail, (PublishedFinalOutputEqualsDetail, PublishedFinalOutputContainsDetail)):
        return detail.matched is not None
    if isinstance(detail, PublishedRootStatusDetail):
        return detail.actual is not None
    if isinstance(detail, (PublishedChildStatusDetail, PublishedToolCalledDetail)):
        return detail.matching_count is not None
    if isinstance(detail, PublishedToolsCalledInOrderDetail):
        return detail.actual_count is not None and detail.matched is not None
    if isinstance(
        detail,
        (
            PublishedMaxToolCallsDetail,
            PublishedMaxModelStepsDetail,
            PublishedMaxTotalTokensDetail,
        ),
    ):
        return detail.actual is not None
    if isinstance(detail, PublishedUsageRecordedDetail):
        return detail.actual is not None
    if isinstance(detail, PublishedMaxEstimatedCostDetail):
        return detail.estimated_cost is not None
    return True


def _detail_expected_pass(detail: PublishedAssertionDetail) -> bool | None:
    if isinstance(detail, (PublishedFinalOutputEqualsDetail, PublishedFinalOutputContainsDetail)):
        return detail.matched
    if isinstance(detail, PublishedRootStatusDetail):
        return None if detail.actual is None else detail.actual == detail.expected
    if isinstance(detail, (PublishedChildStatusDetail, PublishedToolCalledDetail)):
        if detail.matching_count is None:
            return None
        return detail.matching_count >= detail.min_count and (
            detail.max_count is None or detail.matching_count <= detail.max_count
        )
    if isinstance(detail, PublishedMaxToolCallsDetail):
        return None if detail.actual is None else detail.actual <= detail.maximum
    if isinstance(detail, PublishedMaxModelStepsDetail):
        return None if detail.actual is None else detail.actual <= detail.maximum
    if isinstance(detail, PublishedUsageRecordedDetail):
        return None if detail.actual is None else detail.actual >= detail.minimum
    if isinstance(detail, PublishedMaxTotalTokensDetail):
        return None if detail.actual is None else detail.actual <= detail.maximum
    if isinstance(detail, PublishedMaxEstimatedCostDetail):
        if detail.estimated_cost is None:
            return None
        if detail.unpriced_model_steps:
            return None
        return Decimal(detail.estimated_cost) <= Decimal(detail.maximum)
    if isinstance(detail, PublishedToolsCalledInOrderDetail):
        return detail.matched
    return None


def _detail_evidence_area(detail: PublishedAssertionDetail) -> str:
    if isinstance(detail, PublishedRootStatusDetail):
        return "root-status"
    if isinstance(detail, PublishedChildStatusDetail):
        return "child-status"
    if isinstance(detail, (PublishedFinalOutputEqualsDetail, PublishedFinalOutputContainsDetail)):
        return "final-output"
    if isinstance(
        detail,
        (
            PublishedToolCalledDetail,
            PublishedToolsCalledInOrderDetail,
            PublishedMaxToolCallsDetail,
        ),
    ):
        return "tool"
    if isinstance(detail, PublishedMaxModelStepsDetail):
        return "model-step"
    if isinstance(detail, (PublishedUsageRecordedDetail, PublishedMaxTotalTokensDetail)):
        return "usage"
    if isinstance(detail, PublishedMaxEstimatedCostDetail):
        return "cost"
    raise AssertionError("Unreachable published assertion detail type.")


def _assertion_contract(assertion: PublishedAssertionResult) -> tuple[object, ...]:
    detail = assertion.detail
    if isinstance(detail, PublishedRootStatusDetail):
        static_detail: tuple[object, ...] = (detail.kind, detail.expected)
    elif isinstance(detail, PublishedChildStatusDetail):
        static_detail = (detail.kind, detail.expected, detail.min_count, detail.max_count)
    elif isinstance(detail, (PublishedFinalOutputEqualsDetail, PublishedFinalOutputContainsDetail)):
        static_detail = (detail.kind,)
    elif isinstance(detail, PublishedToolCalledDetail):
        static_detail = (detail.kind, detail.tool_name, detail.min_count, detail.max_count)
    elif isinstance(detail, PublishedToolsCalledInOrderDetail):
        static_detail = (detail.kind, detail.expected_count)
    elif isinstance(detail, (PublishedMaxToolCallsDetail, PublishedMaxModelStepsDetail)):
        static_detail = (detail.kind, detail.maximum)
    elif isinstance(detail, PublishedUsageRecordedDetail):
        static_detail = (detail.kind, detail.minimum)
    elif isinstance(detail, PublishedMaxTotalTokensDetail):
        static_detail = (detail.kind, detail.maximum)
    else:
        static_detail = (detail.kind, detail.maximum, detail.currency)
    return assertion.assertion_id, assertion.assertion_revision, *static_detail


def _validate_trial_observations(
    assertions: tuple[PublishedAssertionResult, ...],
    usage: PublishedUsageSummaryV1 | None,
    *,
    evidence_complete: bool,
) -> None:
    metric_values: dict[str, list[int]] = {
        "model_steps": [],
        "tool_calls": [],
        "total_tokens": [],
    }
    root_statuses: set[str] = set()
    child_counts: dict[str, set[int]] = {}
    tool_counts: dict[str, set[int]] = {}
    tool_order_counts: set[int] = set()
    cost_observations: dict[str, set[tuple[str, int, int]]] = {}
    availability_by_area: dict[str, set[bool]] = {}
    for assertion in assertions:
        detail = assertion.detail
        if assertion.outcome != "error":
            area = _detail_evidence_area(detail)
            availability_by_area.setdefault(area, set()).add(_detail_has_observation(detail))
        if isinstance(detail, PublishedRootStatusDetail) and detail.actual is not None:
            root_statuses.add(detail.actual)
        elif isinstance(detail, PublishedChildStatusDetail) and detail.matching_count is not None:
            child_counts.setdefault(detail.expected, set()).add(detail.matching_count)
        elif isinstance(detail, PublishedToolCalledDetail) and detail.matching_count is not None:
            tool_counts.setdefault(detail.tool_name, set()).add(detail.matching_count)
        elif (
            isinstance(detail, PublishedToolsCalledInOrderDetail)
            and detail.actual_count is not None
        ):
            tool_order_counts.add(detail.actual_count)
        elif isinstance(detail, PublishedMaxModelStepsDetail) and detail.actual is not None:
            metric_values["model_steps"].append(detail.actual)
        elif isinstance(detail, PublishedMaxToolCallsDetail) and detail.actual is not None:
            metric_values["tool_calls"].append(detail.actual)
        elif (
            isinstance(
                detail,
                (PublishedUsageRecordedDetail, PublishedMaxTotalTokensDetail),
            )
            and detail.actual is not None
        ):
            metric_values["total_tokens"].append(detail.actual)
        elif (
            isinstance(detail, PublishedMaxEstimatedCostDetail)
            and detail.estimated_cost is not None
            and detail.priced_model_steps is not None
            and detail.unpriced_model_steps is not None
        ):
            cost_observations.setdefault(detail.currency, set()).add(
                (
                    detail.estimated_cost,
                    detail.priced_model_steps,
                    detail.unpriced_model_steps,
                )
            )
            metric_values["model_steps"].append(
                detail.priced_model_steps + detail.unpriced_model_steps
            )
    inconsistent_areas = sorted(
        area for area, availability in availability_by_area.items() if len(availability) > 1
    )
    if inconsistent_areas:
        raise ValueError(
            "Published assertions must agree on evidence availability within each trial area: "
            + ", ".join(inconsistent_areas)
            + "."
        )
    if evidence_complete and availability_by_area.get("root-status") == {False}:
        raise ValueError("Complete published trial evidence requires a root-status observation.")
    if len(root_statuses) > 1:
        raise ValueError("Published root-status observations must agree within a trial.")
    if any(len(values) > 1 for values in child_counts.values()):
        raise ValueError(
            "Published child-status observations for the same status must agree within a trial."
        )
    observed_child_counts = [count for values in child_counts.values() for count in values]
    if sum(observed_child_counts) > EVIDENCE_MAX_CHILD_SESSIONS:
        raise ValueError(
            "Published child-status observations cannot exceed retained child evidence."
        )
    if any(len(values) > 1 for values in tool_counts.values()):
        raise ValueError(
            "Published tool-call observations for the same tool must agree within a trial."
        )
    if len(tool_order_counts) > 1:
        raise ValueError("Published tool-order observations must agree within a trial.")
    if any(len(values) > 1 for values in cost_observations.values()):
        raise ValueError(
            "Published cost observations for the same currency must agree within a trial."
        )
    observed_tool_counts = [count for values in tool_counts.values() for count in values]
    if usage is None and observed_tool_counts:
        raise ValueError("Published tool-call observations require trial usage.")
    if usage is not None and sum(observed_tool_counts) > usage.tool_calls:
        raise ValueError("Published tool-call observations cannot exceed trial tool calls.")
    for metric, values in metric_values.items():
        if len(set(values)) > 1:
            raise ValueError(f"Published {metric} observations must agree within a trial.")
        if usage is None and values:
            raise ValueError(f"Published {metric} observations require trial usage.")
        if usage is not None and values and values[0] != getattr(usage, metric):
            raise ValueError(f"Published {metric} observations must match trial usage.")


def _published_status_from_statuses(statuses) -> PublishedStatus:
    values = tuple(statuses)
    for status in ("error", "unavailable", "failed"):
        if status in values:
            return status
    return "passed"


def _published_score(scores) -> float | None:
    values = tuple(scores)
    if any(value is None for value in values):
        return None
    return sum(values) / len(values)


def _safe_metadata_int(result: EvalAssertionResult, key: str) -> int | None:
    value = result.metadata.get(key)
    return value if type(value) is int and 0 <= value <= MAX_DURABLE_JSON_INTEGER else None


def _safe_metadata_bool(result: EvalAssertionResult, key: str) -> bool | None:
    value = result.metadata.get(key)
    return value if type(value) is bool else None


def _safe_metadata_text(result: EvalAssertionResult, key: str, *, max_chars: int) -> str | None:
    value = result.metadata.get(key)
    if type(value) is not str:
        return None
    try:
        return _bounded_durable_text(
            value,
            key,
            max_chars=max_chars,
            nonblank=True,
            clean=True,
        )
    except ValueError:
        return None


def _safe_metadata_decimal(result: EvalAssertionResult, key: str) -> str | None:
    value = _safe_metadata_text(result, key, max_chars=128)
    if value is None:
        return None
    try:
        return _canonical_decimal_text(value, key, max_chars=128)
    except ValueError:
        return None


def _published_detail(
    spec: AssertionSpec,
    result: EvalAssertionResult,
) -> PublishedAssertionDetail:
    if type(spec) is RootStatusAssertionSpec:
        actual = result.metadata.get("actual")
        return PublishedRootStatusDetail(
            expected=spec.expected,
            actual=(
                actual
                if type(actual) is str and actual in {"completed", "failed", "interrupted"}
                else None
            ),
        )
    if type(spec) is ChildStatusAssertionSpec:
        return PublishedChildStatusDetail(
            expected=spec.expected,
            min_count=spec.min_count,
            max_count=spec.max_count,
            matching_count=_safe_metadata_int(result, "count"),
        )
    if type(spec) is FinalOutputEqualsAssertionSpec:
        return PublishedFinalOutputEqualsDetail(matched=_safe_metadata_bool(result, "matched"))
    if type(spec) is FinalOutputContainsAssertionSpec:
        return PublishedFinalOutputContainsDetail(matched=_safe_metadata_bool(result, "matched"))
    if type(spec) is ToolCalledAssertionSpec:
        return PublishedToolCalledDetail(
            tool_name=spec.tool_name,
            min_count=spec.min_count,
            max_count=spec.max_count,
            matching_count=_safe_metadata_int(result, "count"),
        )
    if type(spec) is ToolsCalledInOrderAssertionSpec:
        actual = result.metadata.get("actual")
        valid_actual = (
            type(actual) is list
            and len(actual) <= EVIDENCE_MAX_TOOL_CALLS
            and all(type(item) is str for item in actual)
        )
        return PublishedToolsCalledInOrderDetail(
            expected_count=len(spec.tool_names),
            actual_count=len(actual) if valid_actual else None,
            matched=(actual == list(spec.tool_names) if valid_actual else None),
        )
    if type(spec) is MaxToolCallsAssertionSpec:
        return PublishedMaxToolCallsDetail(
            maximum=spec.maximum,
            actual=_safe_metadata_int(result, "actual"),
        )
    if type(spec) is MaxModelStepsAssertionSpec:
        return PublishedMaxModelStepsDetail(
            maximum=spec.maximum,
            actual=_safe_metadata_int(result, "actual"),
        )
    if type(spec) is UsageRecordedAssertionSpec:
        return PublishedUsageRecordedDetail(
            minimum=spec.min_total_tokens,
            actual=_safe_metadata_int(result, "total_tokens"),
        )
    if type(spec) is MaxTotalTokensAssertionSpec:
        return PublishedMaxTotalTokensDetail(
            maximum=spec.maximum,
            actual=_safe_metadata_int(result, "actual"),
        )
    if type(spec) is MaxEstimatedCostAssertionSpec:
        estimated_cost = _safe_metadata_decimal(result, "estimated_cost")
        priced_steps = _safe_metadata_int(result, "priced_model_steps")
        unpriced_steps = _safe_metadata_int(result, "unpriced_model_steps")
        if any(item is None for item in (estimated_cost, priced_steps, unpriced_steps)):
            estimated_cost = None
            priced_steps = None
            unpriced_steps = None
        else:
            metadata_currency = _safe_metadata_text(result, "currency", max_chars=16)
            if metadata_currency != spec.currency:
                raise ValueError("Internal cost metadata currency does not match the corpus.")
        summary = result.cost_summary
        if summary is not None:
            summary_observation = (
                _canonical_decimal(summary.total_cost),
                summary.priced_model_steps,
                summary.unpriced_model_steps,
            )
            if summary.currency != spec.currency:
                raise ValueError("Internal cost summary currency does not match the corpus.")
            if summary.priced_model_steps + summary.unpriced_model_steps != summary.model_steps:
                raise ValueError(
                    "Internal cost summary step counts do not form an exact partition."
                )
            if (estimated_cost, priced_steps, unpriced_steps) != summary_observation:
                raise ValueError("Internal cost metadata does not match its exact cost summary.")
            estimated_cost, priced_steps, unpriced_steps = summary_observation
        return PublishedMaxEstimatedCostDetail(
            maximum=spec.maximum,
            currency=spec.currency,
            estimated_cost=estimated_cost,
            priced_model_steps=priced_steps,
            unpriced_model_steps=unpriced_steps,
        )
    raise AssertionError("Unreachable portable assertion detail type.")


def _published_assertion(
    spec: AssertionSpec,
    result: EvalAssertionResult,
) -> PublishedAssertionResult:
    if result.name != spec.id:
        raise ValueError("Internal assertion results do not match the corpus assertion IDs.")
    expected_revision = assertion_spec_revision(spec)
    if result.assertion_revision != expected_revision:
        raise ValueError(
            "Internal assertion result revision does not match the corpus assertion contract."
        )
    outcome = result.outcome.value
    return PublishedAssertionResult(
        assertion_id=spec.id,
        assertion_revision=expected_revision,
        outcome=outcome,
        score=result.score,
        code=outcome,
        message=_ASSERTION_MESSAGE[outcome],
        detail=_published_detail(spec, result),
    )


def _published_usage(value: Mapping[str, Any] | None) -> PublishedUsageSummaryV1 | None:
    if value is None:
        return None
    usage = value.get("usage")
    if type(usage) is not dict:
        return None
    model_steps = value.get("model_steps")
    tool_calls = value.get("tool_calls")
    if not all(type(item) is int and item >= 0 for item in (model_steps, tool_calls)):
        return None
    try:
        aggregate_usage = aggregate_usage_metrics_from_durable_payload(usage)
    except (TypeError, ValueError):
        return None
    return PublishedUsageSummaryV1(
        model_steps=cast("int", model_steps),
        tool_calls=cast("int", tool_calls),
        total_tokens=aggregate_usage.total_tokens,
    )


def _published_case(case: EvalCaseSpec, result) -> PublishedEvalCaseResult:
    trials: list[PublishedEvalTrialResult] = []
    for trial in result.trials:
        if len(trial.assertions) != len(case.assertions):
            raise ValueError("Internal trial assertions do not match the corpus contract.")
        assertions = tuple(
            _published_assertion(spec, assertion)
            for spec, assertion in zip(case.assertions, trial.assertions, strict=True)
        )
        status = trial.status.value
        if status == "skipped":
            raise ValueError("Portable corpus trials cannot be skipped.")
        trials.append(
            PublishedEvalTrialResult(
                trial_number=trial.trial_number,
                status=status,
                score=trial.score,
                assertions=assertions,
                evidence_complete=trial.evidence_complete,
                duration_ms=trial.duration_ms,
                usage=_published_usage(trial.usage_summary),
                code=status,
                message=_TRIAL_MESSAGE[status],
            )
        )
    status = _published_status_from_statuses(trial.status for trial in trials)
    return PublishedEvalCaseResult(
        case_id=case.id,
        case_revision=case.revision,
        status=status,
        score=_published_score(trial.score for trial in trials),
        trials=tuple(trials),
        duration_ms=sum(trial.duration_ms for trial in trials),
    )


def publish_eval_run(corpus: EvalCorpusDocument, run: EvalRun) -> PublishedEvalRun:
    """Project one lossless internal suite run into the public corpus result graph."""

    if type(corpus) is not EvalCorpusDocument:
        raise TypeError("corpus must be an exact EvalCorpusDocument.")
    if type(run) is not EvalRun:
        raise TypeError("run must be an exact EvalRun.")
    corpus = EvalCorpusDocument.model_validate(_model_instance_python_input(corpus))
    run = EvalRun.model_validate(_model_instance_python_input(run))
    suite = next((item for item in corpus.suites if item.id == run.suite_id), None)
    if suite is None:
        raise ValueError("Internal eval run references a suite absent from the corpus.")
    expected_run_contract = eval_run_contract_for_corpus(corpus, suite.id)
    if run.run_contract is None:
        raise ValueError("Portable publication requires an execution-time run contract.")
    if run.run_contract != expected_run_contract:
        raise ValueError("Internal eval run contract does not match the corpus contract.")
    case_specs = tuple(case for case in corpus.cases if case.suite_id == suite.id)
    result_by_id = {case.case_id: case for case in run.cases}
    if set(result_by_id) != {case.id for case in case_specs}:
        raise ValueError("Internal eval run cases do not match the complete corpus suite.")
    if any(len(result_by_id[case.id].trials) != suite.trial_request.trials for case in case_specs):
        raise ValueError("Internal eval run trial counts do not match the corpus suite.")
    cases = tuple(_published_case(case, result_by_id[case.id]) for case in case_specs)
    status = _published_status_from_statuses(case.status for case in cases)
    document: dict[str, Any] = {
        "schema_version": PUBLISHED_EVAL_SCHEMA_VERSION,
        "corpus_revision": corpus.revision,
        "target_key": corpus.target_key,
        "suite_id": suite.id,
        "suite_revision": suite.revision,
        "evidence_policy_revision": corpus.evidence_policy.revision,
        "pricing_profile_fingerprint": (
            None if corpus.pricing_profile is None else corpus.pricing_profile.fingerprint
        ),
        "status": status,
        "score": _published_score(case.score for case in cases),
        "cases": cases,
        "duration_ms": sum(case.duration_ms for case in cases),
    }
    revision_document = {
        **document,
        "cases": [case.model_dump(mode="json") for case in cases],
    }
    return PublishedEvalRun(
        revision=_content_revision(revision_document, "published eval run"),
        **document,
    )
