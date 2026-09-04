"""Portable suite reliability policy and preflight exposure contracts."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from decimal import Decimal
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from cayu._eval_limits import EVAL_SUITE_MAX_CONCURRENCY
from cayu._validation import canonical_durable_json_bytes, require_durable_clean_nonblank
from cayu.evals.result_contract import EvalTrialDiagnosticCode

EVAL_SUITE_TRIAL_POLICY_SCHEMA_VERSION = 1
EVAL_SUITE_RUN_EXPOSURE_SCHEMA_VERSION = 1
EVAL_SUITE_MAX_TRIALS = 100

_EVAL_SUITE_TRIAL_POLICY_DOMAIN = "eval suite trial policy"
_EVAL_SUITE_RUN_EXPOSURE_DOMAIN = "eval suite run exposure"
_EVAL_SUITE_RUN_EXPOSURE_COMPARISON_DOMAIN = "eval suite run exposure comparison"
_SHA256_REVISION_PREFIX = "sha256:"
_CANONICAL_NONNEGATIVE_DECIMAL = re.compile(
    r"(?:0|[1-9][0-9]*)(?:\.[0-9]*[1-9])?\Z",
    re.ASCII,
)
_PORTABLE_ID = re.compile(r"[a-z][a-z0-9._-]{0,127}\Z", re.ASCII)


def _revision(material: object, domain: str) -> str:
    return (
        _SHA256_REVISION_PREFIX
        + hashlib.sha256(canonical_durable_json_bytes(material, domain)).hexdigest()
    )


def _validate_revision(value: str, field_name: str) -> str:
    value = require_durable_clean_nonblank(value, field_name)
    if (
        len(value) != 71
        or not value.startswith(_SHA256_REVISION_PREFIX)
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{field_name} must be a lowercase sha256 revision.")
    return value


class _PortablePolicyModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        hide_input_in_errors=True,
    )


class EvalSuiteTrialPolicyV1(_PortablePolicyModel):
    """Immutable decision rule for the complete retained trial set of each case."""

    schema_version: Literal[1] = EVAL_SUITE_TRIAL_POLICY_SCHEMA_VERSION
    revision: StrictStr
    trial_count: StrictInt = Field(ge=1, le=EVAL_SUITE_MAX_TRIALS)
    minimum_passed_trials: StrictInt = Field(ge=1, le=EVAL_SUITE_MAX_TRIALS)
    max_concurrency: StrictInt = Field(ge=1, le=EVAL_SUITE_MAX_CONCURRENCY)
    require_zero_runtime_errors: Literal[True] = True
    require_zero_evaluator_errors: Literal[True] = True
    require_complete_required_evidence: Literal[True] = True

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1.")
        return value

    @field_validator(
        "require_zero_runtime_errors",
        "require_zero_evaluator_errors",
        "require_complete_required_evidence",
        mode="before",
    )
    @classmethod
    def validate_required_flag_types(cls, value: object, info) -> object:
        if type(value) is not bool or value is not True:
            raise ValueError(f"{info.field_name} must be true.")
        return value

    @field_validator("revision")
    @classmethod
    def validate_revision_shape(cls, value: str, info) -> str:
        return _validate_revision(value, info.field_name)

    @model_validator(mode="after")
    def validate_contract(self) -> EvalSuiteTrialPolicyV1:
        if self.minimum_passed_trials > self.trial_count:
            raise ValueError("minimum_passed_trials cannot exceed trial_count.")
        material = self.model_dump(mode="json", exclude={"revision"})
        if self.revision != _revision(material, _EVAL_SUITE_TRIAL_POLICY_DOMAIN):
            raise ValueError("Suite trial policy revision does not match its content.")
        return self

    @classmethod
    def create(
        cls,
        *,
        trial_count: int = 1,
        minimum_passed_trials: int | None = None,
        max_concurrency: int = 1,
    ) -> EvalSuiteTrialPolicyV1:
        if type(trial_count) is not int:
            raise TypeError("trial_count must be an int.")
        if minimum_passed_trials is None:
            minimum_passed_trials = trial_count
        elif type(minimum_passed_trials) is not int:
            raise TypeError("minimum_passed_trials must be an int or None.")
        if type(max_concurrency) is not int:
            raise TypeError("max_concurrency must be an int.")
        material = {
            "schema_version": EVAL_SUITE_TRIAL_POLICY_SCHEMA_VERSION,
            "trial_count": trial_count,
            "minimum_passed_trials": minimum_passed_trials,
            "max_concurrency": max_concurrency,
            "require_zero_runtime_errors": True,
            "require_zero_evaluator_errors": True,
            "require_complete_required_evidence": True,
        }
        return cls(
            revision=_revision(material, _EVAL_SUITE_TRIAL_POLICY_DOMAIN),
            **material,
        )


EvalReliabilityOutcome = Literal["passed", "failed", "unavailable", "error"]
EvalVariabilityKind = Literal[
    "single_trial",
    "candidate_evaluation_variability",
    "end_to_end_evaluation_variability",
]


class EvalCaseReliabilityV1(_PortablePolicyModel):
    """Lossless bounded distribution and policy decision for one published case."""

    schema_version: Literal[1] = 1
    trial_policy_revision: StrictStr
    outcome: EvalReliabilityOutcome
    total_trials: StrictInt = Field(ge=1, le=EVAL_SUITE_MAX_TRIALS)
    passed_trials: StrictInt = Field(ge=0, le=EVAL_SUITE_MAX_TRIALS)
    candidate_failed_trials: StrictInt = Field(ge=0, le=EVAL_SUITE_MAX_TRIALS)
    runtime_error_trials: StrictInt = Field(ge=0, le=EVAL_SUITE_MAX_TRIALS)
    evaluator_error_trials: StrictInt = Field(ge=0, le=EVAL_SUITE_MAX_TRIALS)
    unavailable_trials: StrictInt = Field(ge=0, le=EVAL_SUITE_MAX_TRIALS)
    cancelled_trials: StrictInt = Field(ge=0, le=EVAL_SUITE_MAX_TRIALS)
    scored_trials: StrictInt = Field(ge=0, le=EVAL_SUITE_MAX_TRIALS)
    minimum_score: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    mean_score: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    maximum_score: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    variability: EvalVariabilityKind

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1.")
        return value

    @field_validator("trial_policy_revision")
    @classmethod
    def validate_policy_revision(cls, value: str, info) -> str:
        return _validate_revision(value, info.field_name)

    @model_validator(mode="after")
    def validate_distribution(self) -> EvalCaseReliabilityV1:
        classified = (
            self.passed_trials
            + self.candidate_failed_trials
            + self.runtime_error_trials
            + self.evaluator_error_trials
            + self.unavailable_trials
            + self.cancelled_trials
        )
        if classified != self.total_trials:
            raise ValueError("Reliability trial counts must classify every retained trial once.")
        minimum_score = self.minimum_score
        mean_score = self.mean_score
        maximum_score = self.maximum_score
        scores = (minimum_score, mean_score, maximum_score)
        if self.scored_trials == 0:
            if any(score is not None for score in scores):
                raise ValueError("A reliability distribution without scores cannot carry stats.")
        elif minimum_score is None or mean_score is None or maximum_score is None:
            raise ValueError("A scored reliability distribution requires complete score stats.")
        elif not minimum_score <= mean_score <= maximum_score:
            raise ValueError("Reliability score statistics are inconsistent.")
        if (self.total_trials == 1) != (self.variability == "single_trial"):
            raise ValueError("Only a one-trial result may use the single-trial label.")
        return self

    @classmethod
    def create(
        cls,
        *,
        policy: EvalSuiteTrialPolicyV1,
        trials: Iterable[tuple[str, float | None, EvalTrialDiagnosticCode]],
        uses_model_judge: bool,
    ) -> EvalCaseReliabilityV1:
        if type(policy) is not EvalSuiteTrialPolicyV1:
            raise TypeError("policy must be an exact EvalSuiteTrialPolicyV1.")
        if type(uses_model_judge) is not bool:
            raise TypeError("uses_model_judge must be a bool.")
        retained = tuple(trials)
        if len(retained) != policy.trial_count:
            raise ValueError("Reliability trials must match the immutable trial policy.")
        passed = failed = runtime_errors = evaluator_errors = unavailable = cancelled = 0
        scores: list[float] = []
        for status, score, code in retained:
            if type(code) is not EvalTrialDiagnosticCode:
                raise TypeError("Reliability diagnostics must be EvalTrialDiagnosticCode values.")
            if score is not None:
                scores.append(score)
            if status == "passed":
                passed += 1
            elif status == "failed":
                failed += 1
            elif code is EvalTrialDiagnosticCode.EXTERNAL_TARGET_CANCELLED:
                cancelled += 1
            elif status == "unavailable":
                unavailable += 1
            elif status == "error" and code is EvalTrialDiagnosticCode.ASSERTION_EVALUATION_FAILED:
                evaluator_errors += 1
            elif status == "error":
                runtime_errors += 1
            else:
                raise ValueError("Reliability received an unsupported trial outcome.")
        if runtime_errors or evaluator_errors:
            outcome: EvalReliabilityOutcome = "error"
        elif unavailable or cancelled:
            outcome = "unavailable"
        elif passed >= policy.minimum_passed_trials:
            outcome = "passed"
        else:
            outcome = "failed"
        variability: EvalVariabilityKind
        if len(retained) == 1:
            variability = "single_trial"
        elif uses_model_judge:
            variability = "end_to_end_evaluation_variability"
        else:
            variability = "candidate_evaluation_variability"
        return cls(
            trial_policy_revision=policy.revision,
            outcome=outcome,
            total_trials=len(retained),
            passed_trials=passed,
            candidate_failed_trials=failed,
            runtime_error_trials=runtime_errors,
            evaluator_error_trials=evaluator_errors,
            unavailable_trials=unavailable,
            cancelled_trials=cancelled,
            scored_trials=len(scores),
            minimum_score=None if not scores else min(scores),
            mean_score=None if not scores else sum(scores) / len(scores),
            maximum_score=None if not scores else max(scores),
            variability=variability,
        )


class EvalMaximumCostTotalV1(_PortablePolicyModel):
    currency: StrictStr = Field(min_length=1, max_length=16)
    amount: StrictStr = Field(min_length=1, max_length=128)

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, value: str) -> str:
        value = require_durable_clean_nonblank(value, "currency")
        if not value.isascii() or any(
            not (character.isupper() or character.isdigit() or character in "._-")
            for character in value
        ):
            raise ValueError("currency must be a portable uppercase identifier.")
        return value

    @field_validator("amount")
    @classmethod
    def validate_amount(cls, value: str) -> str:
        value = require_durable_clean_nonblank(value, "amount")
        try:
            amount = Decimal(value)
        except Exception as exc:
            raise ValueError("amount must be a canonical decimal string.") from exc
        if (
            not amount.is_finite()
            or amount < 0
            or _CANONICAL_NONNEGATIVE_DECIMAL.fullmatch(value) is None
        ):
            raise ValueError("amount must be a canonical non-negative decimal string.")
        return value

    @classmethod
    def from_decimal(cls, *, currency: str, amount: Decimal) -> Self:
        if type(amount) is not Decimal:
            raise TypeError("amount must be an exact Decimal.")
        text = format(amount, "f")
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return cls(currency=currency, amount="0" if text in {"", "-0"} else text)


class EvalCandidateCostBudgetV1(EvalMaximumCostTotalV1):
    """One configured per-trial observed-cost interruption threshold."""

    @model_validator(mode="after")
    def validate_positive_budget(self) -> EvalCandidateCostBudgetV1:
        if Decimal(self.amount) <= 0:
            raise ValueError("Candidate cost budgets must be greater than zero.")
        return self


EvalMaximumCostUnavailableReason = Literal[
    "no_candidate_cost_ceiling",
    "candidate_cost_not_hard_bounded",
    "candidate_pricing_incomplete",
    "judge_cost_not_hard_bounded",
    "judge_pricing_incomplete",
]


class EvalMaximumCostExposureV1(_PortablePolicyModel):
    state: Literal["priced", "unavailable", "not_applicable"]
    totals: tuple[EvalMaximumCostTotalV1, ...] = Field(default=(), max_length=16)
    unavailable_reason: EvalMaximumCostUnavailableReason | None = None
    pricing_profile_fingerprints: tuple[StrictStr, ...] = Field(default=(), max_length=64)

    @field_validator("pricing_profile_fingerprints")
    @classmethod
    def validate_pricing_revisions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        validated = tuple(
            _validate_revision(item, "pricing_profile_fingerprints") for item in value
        )
        if validated != tuple(sorted(set(validated))):
            raise ValueError("Pricing profile fingerprints must be unique and sorted.")
        return validated

    @model_validator(mode="after")
    def validate_state(self) -> EvalMaximumCostExposureV1:
        currencies = tuple(item.currency for item in self.totals)
        if currencies != tuple(sorted(set(currencies))):
            raise ValueError("Maximum cost totals must have unique sorted currencies.")
        if self.state == "priced":
            if (
                not self.totals
                or not self.pricing_profile_fingerprints
                or self.unavailable_reason is not None
            ):
                raise ValueError(
                    "Priced maximum cost requires totals, pricing identity, and no "
                    "unavailable reason."
                )
        elif self.state == "unavailable":
            if self.totals or self.unavailable_reason is None:
                raise ValueError("Unavailable maximum cost requires only a typed reason.")
            if (
                self.unavailable_reason
                in {"candidate_cost_not_hard_bounded", "judge_cost_not_hard_bounded"}
                and not self.pricing_profile_fingerprints
            ):
                raise ValueError("A priced but non-hard-bounded cost requires pricing identity.")
        elif (
            self.totals or self.unavailable_reason is not None or self.pricing_profile_fingerprints
        ):
            raise ValueError(
                "Not-applicable maximum cost cannot carry totals, pricing, or a reason."
            )
        return self


class EvalExecutionProfileExposureV1(_PortablePolicyModel):
    """One exact candidate profile bound to the cases it will execute."""

    case_ids: tuple[StrictStr, ...] = Field(min_length=1, max_length=1_000)
    execution_profile_revision: StrictStr
    execution_profile_comparison_revision: StrictStr
    candidate_cost_budget: EvalCandidateCostBudgetV1 | None = None

    @field_validator("case_ids")
    @classmethod
    def validate_case_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        validated = tuple(require_durable_clean_nonblank(item, "case_ids") for item in value)
        if any(_PORTABLE_ID.fullmatch(item) is None for item in validated):
            raise ValueError("Execution-profile case IDs must be portable identifiers.")
        if validated != tuple(sorted(set(validated))):
            raise ValueError("Execution-profile case IDs must be unique and sorted.")
        return validated

    @field_validator("execution_profile_revision", "execution_profile_comparison_revision")
    @classmethod
    def validate_profile_revision(cls, value: str, info) -> str:
        return _validate_revision(value, info.field_name)


class EvalJudgeProfileExposureV1(_PortablePolicyModel):
    """One current judge route accepted for every assertion that names its key."""

    profile_key: StrictStr
    judge_profile_revision: StrictStr
    judge_profile_comparison_revision: StrictStr

    @field_validator("profile_key")
    @classmethod
    def validate_profile_key(cls, value: str) -> str:
        value = require_durable_clean_nonblank(value, "profile_key")
        if _PORTABLE_ID.fullmatch(value) is None:
            raise ValueError("profile_key must be a portable identifier.")
        return value

    @field_validator("judge_profile_revision", "judge_profile_comparison_revision")
    @classmethod
    def validate_profile_revision(cls, value: str, info) -> str:
        return _validate_revision(value, info.field_name)


class EvalSuiteRunExposureV1(_PortablePolicyModel):
    """Exact maximum configured work accepted for one authored-suite selection."""

    schema_version: Literal[1] = EVAL_SUITE_RUN_EXPOSURE_SCHEMA_VERSION
    revision: StrictStr
    selection_revision: StrictStr
    trial_policy_revision: StrictStr
    execution_profiles: tuple[EvalExecutionProfileExposureV1, ...] = Field(
        min_length=1,
        max_length=1_000,
    )
    judge_profiles: tuple[EvalJudgeProfileExposureV1, ...] = Field(default=(), max_length=64)
    candidate_trials: StrictInt = Field(ge=1, le=100_000)
    judge_evaluations: StrictInt = Field(ge=0, le=6_400_000)
    max_concurrency: StrictInt = Field(ge=1, le=EVAL_SUITE_MAX_CONCURRENCY)
    maximum_candidate_model_steps: StrictInt = Field(ge=1)
    maximum_candidate_total_tokens: StrictInt | None = Field(default=None, ge=1)
    maximum_judge_input_tokens: StrictInt | None = Field(default=None, ge=0)
    maximum_judge_output_tokens: StrictInt | None = Field(default=None, ge=0)
    maximum_judge_total_tokens: StrictInt | None = Field(default=None, ge=0)
    candidate_cost: EvalMaximumCostExposureV1
    judge_cost: EvalMaximumCostExposureV1

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1.")
        return value

    @field_validator(
        "revision",
        "selection_revision",
        "trial_policy_revision",
    )
    @classmethod
    def validate_revisions(cls, value: str, info) -> str:
        return _validate_revision(value, info.field_name)

    @field_validator("execution_profiles")
    @classmethod
    def validate_execution_profiles(
        cls,
        value: tuple[EvalExecutionProfileExposureV1, ...],
    ) -> tuple[EvalExecutionProfileExposureV1, ...]:
        validated = tuple(
            EvalExecutionProfileExposureV1.model_validate(
                item.model_dump(mode="python", round_trip=True, warnings="none")
            )
            for item in value
        )
        keys = tuple(item.case_ids for item in validated)
        if keys != tuple(sorted(keys)):
            raise ValueError("Execution profiles must be sorted by their case IDs.")
        all_case_ids = tuple(case_id for item in validated for case_id in item.case_ids)
        if len(all_case_ids) != len(set(all_case_ids)):
            raise ValueError("Execution profiles must bind every case at most once.")
        return validated

    @field_validator("judge_profiles")
    @classmethod
    def validate_judge_profiles(
        cls,
        value: tuple[EvalJudgeProfileExposureV1, ...],
    ) -> tuple[EvalJudgeProfileExposureV1, ...]:
        validated = tuple(
            EvalJudgeProfileExposureV1.model_validate(
                item.model_dump(mode="python", round_trip=True, warnings="none")
            )
            for item in value
        )
        keys = tuple(item.profile_key for item in validated)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("Judge profiles must have unique sorted keys.")
        return validated

    @model_validator(mode="after")
    def validate_contract(self) -> EvalSuiteRunExposureV1:
        if self.candidate_cost.state == "not_applicable":
            raise ValueError("Candidate work requires a maximum-cost exposure state.")
        if self.maximum_candidate_model_steps < self.candidate_trials:
            raise ValueError("Candidate model-step exposure is smaller than candidate trials.")
        judge_token_maxima = (
            self.maximum_judge_input_tokens,
            self.maximum_judge_output_tokens,
            self.maximum_judge_total_tokens,
        )
        if self.judge_evaluations == 0 and (
            judge_token_maxima != (0, 0, 0)
            or self.judge_cost.state != "not_applicable"
            or self.judge_profiles
        ):
            raise ValueError("A suite without judge work cannot carry judge exposure.")
        if self.judge_evaluations > 0:
            if not self.judge_profiles:
                raise ValueError("Judge work requires its exact accepted judge profiles.")
            if any(value is not None for value in judge_token_maxima):
                raise ValueError(
                    "Judge token maxima must remain unavailable until provider dispatch "
                    "enforces them."
                )
            if self.judge_cost.state != "unavailable":
                raise ValueError(
                    "Judge cost must remain unavailable until provider dispatch enforces it."
                )
            if self.judge_cost.unavailable_reason not in {
                "judge_cost_not_hard_bounded",
                "judge_pricing_incomplete",
            }:
                raise ValueError("Judge work requires a judge-specific cost diagnostic.")
        material = self.model_dump(mode="json", exclude={"revision"})
        if self.revision != _revision(material, _EVAL_SUITE_RUN_EXPOSURE_DOMAIN):
            raise ValueError("Suite run exposure revision does not match its content.")
        return self

    @property
    def comparison_revision(self) -> str:
        """Return the cross-release identity of the accepted execution contract."""

        validated = EvalSuiteRunExposureV1.model_validate(
            self.model_dump(mode="python", round_trip=True, warnings="none")
        )
        material = validated.model_dump(mode="json", exclude={"revision"})
        material["execution_profiles"] = [
            {
                "case_ids": list(item.case_ids),
                "execution_profile_comparison_revision": (
                    item.execution_profile_comparison_revision
                ),
                "candidate_cost_budget": (
                    None
                    if item.candidate_cost_budget is None
                    else item.candidate_cost_budget.model_dump(mode="json")
                ),
            }
            for item in validated.execution_profiles
        ]
        material["judge_profiles"] = [
            {
                "profile_key": item.profile_key,
                "judge_profile_comparison_revision": (item.judge_profile_comparison_revision),
            }
            for item in validated.judge_profiles
        ]
        return _revision(material, _EVAL_SUITE_RUN_EXPOSURE_COMPARISON_DOMAIN)

    @classmethod
    def create(cls, **values: Any) -> EvalSuiteRunExposureV1:
        draft = cls.model_construct(
            schema_version=EVAL_SUITE_RUN_EXPOSURE_SCHEMA_VERSION,
            revision=_SHA256_REVISION_PREFIX + "0" * 64,
            **values,
        )
        material = draft.model_dump(mode="json", exclude={"revision"})
        return cls(
            revision=_revision(material, _EVAL_SUITE_RUN_EXPOSURE_DOMAIN),
            **values,
        )


__all__ = [
    "EVAL_SUITE_RUN_EXPOSURE_SCHEMA_VERSION",
    "EVAL_SUITE_TRIAL_POLICY_SCHEMA_VERSION",
    "EvalCandidateCostBudgetV1",
    "EvalCaseReliabilityV1",
    "EvalExecutionProfileExposureV1",
    "EvalJudgeProfileExposureV1",
    "EvalMaximumCostExposureV1",
    "EvalMaximumCostTotalV1",
    "EvalMaximumCostUnavailableReason",
    "EvalSuiteRunExposureV1",
    "EvalSuiteTrialPolicyV1",
]
