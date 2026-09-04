from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from cayu.evals.result_contract import EvalTrialDiagnosticCode
from cayu.evals.trial_policy import (
    EvalCaseReliabilityV1,
    EvalExecutionProfileExposureV1,
    EvalMaximumCostExposureV1,
    EvalMaximumCostTotalV1,
    EvalSuiteRunExposureV1,
    EvalSuiteTrialPolicyV1,
)

_REVISION_A = "sha256:" + "a" * 64
_REVISION_B = "sha256:" + "b" * 64


def test_trial_policy_is_content_addressed_and_bounded() -> None:
    first = EvalSuiteTrialPolicyV1.create(
        trial_count=5,
        minimum_passed_trials=4,
        max_concurrency=2,
    )
    second = EvalSuiteTrialPolicyV1.create(
        trial_count=5,
        minimum_passed_trials=4,
        max_concurrency=2,
    )

    assert first == second
    assert first.revision.startswith("sha256:")
    assert first.require_zero_runtime_errors is True
    assert first.require_zero_evaluator_errors is True
    assert first.require_complete_required_evidence is True

    forged = first.model_dump(mode="json")
    forged["minimum_passed_trials"] = 3
    with pytest.raises(ValidationError, match="revision does not match"):
        EvalSuiteTrialPolicyV1.model_validate(forged)


def test_trial_policy_supports_one_hundred_way_execution() -> None:
    policy = EvalSuiteTrialPolicyV1.create(max_concurrency=100)

    assert policy.max_concurrency == 100
    with pytest.raises(ValidationError, match="less than or equal to 100"):
        EvalSuiteTrialPolicyV1.create(max_concurrency=101)


def test_maximum_cost_and_work_exposure_require_canonical_pricing_identity() -> None:
    total = EvalMaximumCostTotalV1.from_decimal(
        currency="USD",
        amount=Decimal("0.5000"),
    )
    assert total.amount == "0.5"

    with pytest.raises(ValidationError, match="canonical non-negative decimal"):
        EvalMaximumCostTotalV1(currency="USD", amount="0.50")
    with pytest.raises(ValidationError, match="pricing identity"):
        EvalMaximumCostExposureV1(state="priced", totals=(total,))

    exposure = EvalSuiteRunExposureV1.create(
        selection_revision=_REVISION_A,
        trial_policy_revision=_REVISION_B,
        execution_profiles=(
            EvalExecutionProfileExposureV1(
                case_ids=("case-one",),
                execution_profile_revision=_REVISION_A,
                execution_profile_comparison_revision=_REVISION_A,
            ),
        ),
        candidate_trials=2,
        judge_evaluations=0,
        max_concurrency=1,
        maximum_candidate_model_steps=4,
        maximum_candidate_total_tokens=200,
        maximum_judge_input_tokens=0,
        maximum_judge_output_tokens=0,
        maximum_judge_total_tokens=0,
        candidate_cost=EvalMaximumCostExposureV1(
            state="priced",
            totals=(total,),
            pricing_profile_fingerprints=(_REVISION_B,),
        ),
        judge_cost=EvalMaximumCostExposureV1(state="not_applicable"),
    )
    assert exposure.revision.startswith("sha256:")
    assert exposure.comparison_revision.startswith("sha256:")

    forged = exposure.model_dump(mode="json")
    forged["execution_profiles"] = [
        {
            "case_ids": ["case-one"],
            "execution_profile_revision": _REVISION_A,
            "execution_profile_comparison_revision": _REVISION_A,
        },
        {
            "case_ids": ["case-one"],
            "execution_profile_revision": _REVISION_B,
            "execution_profile_comparison_revision": _REVISION_B,
        },
    ]
    with pytest.raises(ValidationError, match="bind every case at most once"):
        EvalSuiteRunExposureV1.model_validate(forged)


def test_reliability_applies_threshold_and_retains_complete_distribution() -> None:
    policy = EvalSuiteTrialPolicyV1.create(
        trial_count=3,
        minimum_passed_trials=2,
        max_concurrency=2,
    )

    reliability = EvalCaseReliabilityV1.create(
        policy=policy,
        trials=(
            ("passed", 1.0, EvalTrialDiagnosticCode.PASSED),
            ("failed", 0.0, EvalTrialDiagnosticCode.ASSERTION_FAILED),
            ("passed", 0.8, EvalTrialDiagnosticCode.PASSED),
        ),
        uses_model_judge=False,
    )

    assert reliability.outcome == "passed"
    assert reliability.passed_trials == 2
    assert reliability.candidate_failed_trials == 1
    assert reliability.scored_trials == 3
    assert reliability.minimum_score == 0.0
    assert reliability.mean_score == pytest.approx(0.6)
    assert reliability.maximum_score == 1.0
    assert reliability.variability == "candidate_evaluation_variability"


@pytest.mark.parametrize(
    ("status", "code", "expected_count"),
    (
        ("error", EvalTrialDiagnosticCode.EXECUTION_FAILED, "runtime_error_trials"),
        (
            "error",
            EvalTrialDiagnosticCode.ASSERTION_EVALUATION_FAILED,
            "evaluator_error_trials",
        ),
        (
            "unavailable",
            EvalTrialDiagnosticCode.ASSERTION_EVIDENCE_UNAVAILABLE,
            "unavailable_trials",
        ),
        (
            "unavailable",
            EvalTrialDiagnosticCode.EXTERNAL_TARGET_CANCELLED,
            "cancelled_trials",
        ),
    ),
)
def test_reliability_fails_closed_after_reaching_pass_threshold(
    status: str,
    code: EvalTrialDiagnosticCode,
    expected_count: str,
) -> None:
    policy = EvalSuiteTrialPolicyV1.create(
        trial_count=3,
        minimum_passed_trials=2,
    )
    reliability = EvalCaseReliabilityV1.create(
        policy=policy,
        trials=(
            ("passed", 1.0, EvalTrialDiagnosticCode.PASSED),
            ("passed", 1.0, EvalTrialDiagnosticCode.PASSED),
            (status, None, code),
        ),
        uses_model_judge=True,
    )

    assert reliability.outcome == ("error" if status == "error" else "unavailable")
    assert getattr(reliability, expected_count) == 1
    assert reliability.variability == "end_to_end_evaluation_variability"
