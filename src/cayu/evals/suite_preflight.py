"""Deterministic authored-suite work and maximum-cost exposure compilation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal

from cayu.evals.corpus import (
    JudgeProfileIdentityV1,
    ModelJudgeAssertionSpec,
    StructuredModelJudgeAssertionSpec,
    eval_suite_trial_policy,
)
from cayu.evals.execution_profiles import EvalExecutionProfileV1
from cayu.evals.store import EvalRunCostBudget
from cayu.evals.suite_authoring import (
    EvalSuiteDocument,
    EvalSuiteSelectionV1,
    validate_eval_suite_selection,
    validate_expected_eval_suite_revision,
)
from cayu.evals.trial_policy import (
    EvalCandidateCostBudgetV1,
    EvalExecutionProfileExposureV1,
    EvalJudgeProfileExposureV1,
    EvalMaximumCostExposureV1,
    EvalMaximumCostTotalV1,
    EvalMaximumCostUnavailableReason,
    EvalSuiteRunExposureV1,
    EvalSuiteTrialPolicyV1,
)


@dataclass(frozen=True, slots=True)
class EvalCandidateLaunchExposure:
    """One preflighted launch group under one current execution profile."""

    case_ids: tuple[str, ...]
    execution_profile: EvalExecutionProfileV1
    cost_budget: EvalRunCostBudget | None = None


@dataclass(frozen=True, slots=True)
class EvalSuiteLaunchConcurrencyAllocation:
    """One run's durable lane and share of a suite-wide concurrency ceiling."""

    lane: int
    max_concurrency: int


def allocate_authored_suite_launch_concurrency(
    policy: EvalSuiteTrialPolicyV1,
    launch_count: int,
) -> tuple[EvalSuiteLaunchConcurrencyAllocation, ...]:
    """Partition a suite ceiling across durable runs without wasting available lanes."""

    if type(policy) is not EvalSuiteTrialPolicyV1:
        raise TypeError("policy must be an exact EvalSuiteTrialPolicyV1.")
    if type(launch_count) is not int or launch_count < 1:
        raise ValueError("launch_count must be a positive int.")
    lane_count = min(launch_count, policy.max_concurrency)
    base, extra = divmod(policy.max_concurrency, lane_count)
    lane_limits = tuple(base + (1 if lane < extra else 0) for lane in range(lane_count))
    return tuple(
        EvalSuiteLaunchConcurrencyAllocation(
            lane=index % lane_count,
            max_concurrency=lane_limits[index % lane_count],
        )
        for index in range(launch_count)
    )


def _cost_exposure(
    totals: dict[str, Decimal],
    *,
    required: bool,
    complete: bool,
    unavailable_reason: EvalMaximumCostUnavailableReason,
    pricing_profile_fingerprints: set[str],
) -> EvalMaximumCostExposureV1:
    if not required:
        return EvalMaximumCostExposureV1(state="not_applicable")
    if not complete:
        return EvalMaximumCostExposureV1(
            state="unavailable",
            unavailable_reason=unavailable_reason,
            pricing_profile_fingerprints=tuple(sorted(pricing_profile_fingerprints)),
        )
    return EvalMaximumCostExposureV1(
        state="priced",
        totals=tuple(
            EvalMaximumCostTotalV1.from_decimal(currency=currency, amount=amount)
            for currency, amount in sorted(totals.items())
        ),
        pricing_profile_fingerprints=tuple(sorted(pricing_profile_fingerprints)),
    )


def compile_authored_suite_run_exposure(
    document: EvalSuiteDocument,
    selection: EvalSuiteSelectionV1,
    launches: tuple[EvalCandidateLaunchExposure, ...],
    *,
    judge_profiles: tuple[JudgeProfileIdentityV1, ...],
    candidate_pricing_profile_fingerprint: str | None,
) -> EvalSuiteRunExposureV1:
    """Compile exact maximum configured work from already-preflighted authority."""

    validated = validate_expected_eval_suite_revision(document, document.revision)
    selected = validate_eval_suite_selection(selection, validated)
    policy = eval_suite_trial_policy(validated.suite)
    concurrency_allocations = allocate_authored_suite_launch_concurrency(
        policy,
        len(launches),
    )
    selected_ids = tuple(item.id for item in selected.cases)
    launch_ids = tuple(case_id for launch in launches for case_id in launch.case_ids)
    if tuple(sorted(launch_ids)) != selected_ids or len(launch_ids) != len(set(launch_ids)):
        raise ValueError("Exposure launch groups must partition the immutable selection.")
    for launch, allocation in zip(launches, concurrency_allocations, strict=True):
        if type(launch.execution_profile) is not EvalExecutionProfileV1:
            raise TypeError("Exposure launches require exact execution profiles.")
        if (
            policy.trial_count > launch.execution_profile.ceilings.max_trials
            or allocation.max_concurrency > launch.execution_profile.ceilings.max_concurrency
        ):
            raise ValueError("Suite trial policy exceeds a selected execution profile.")

    candidate_trials = len(selected_ids) * policy.trial_count
    maximum_candidate_model_steps = sum(
        len(launch.case_ids) * policy.trial_count * launch.execution_profile.ceilings.max_steps
        for launch in launches
    )
    # Runtime token limits stop subsequent work after observed usage reaches the
    # configured threshold. A single provider completion can legitimately cross
    # that threshold, so those limits are not a pre-dispatch maximum exposure.
    # Until an execution profile publishes a reservation-backed token bound, the
    # only truthful candidate token maximum is unavailable.
    maximum_candidate_total_tokens: int | None = None

    candidate_cost_totals: dict[str, Decimal] = defaultdict(Decimal)
    candidate_cost_complete = False
    for launch in launches:
        budget = launch.cost_budget
        if budget is None:
            continue
        candidate_cost_totals[budget.currency] += (
            budget.max_estimated_cost * len(launch.case_ids) * policy.trial_count
        )
    candidate_pricing = (
        set()
        if candidate_pricing_profile_fingerprint is None
        else {candidate_pricing_profile_fingerprint}
    )
    if candidate_cost_totals:
        # EvalRunCostBudget is an observed-usage interruption threshold. Without
        # an exact per-dispatch reservation, the completion that crosses it may
        # cost more than the configured value, so it must not be advertised as a
        # maximum priced exposure.
        candidate_reason: EvalMaximumCostUnavailableReason = (
            "candidate_cost_not_hard_bounded"
            if candidate_pricing
            else "candidate_pricing_incomplete"
        )
    else:
        candidate_reason = "no_candidate_cost_ceiling"
    candidate_cost = _cost_exposure(
        candidate_cost_totals,
        required=True,
        complete=candidate_cost_complete,
        unavailable_reason=candidate_reason,
        pricing_profile_fingerprints=candidate_pricing,
    )

    cases_by_id = {case.id: case for case in validated.cases}
    profile_by_key = {profile.key: profile for profile in judge_profiles}
    judge_evaluations = 0
    used_judge_profiles: dict[str, JudgeProfileIdentityV1] = {}
    judge_pricing: set[str] = set()
    judge_cost_complete = True
    for case_id in selected_ids:
        for assertion in cases_by_id[case_id].assertions:
            if type(assertion) is ModelJudgeAssertionSpec:
                profile_key = assertion.evaluator_key
            elif type(assertion) is StructuredModelJudgeAssertionSpec:
                profile_key = assertion.judge_profile_key
            else:
                continue
            profile = profile_by_key.get(profile_key)
            if profile is None:
                raise ValueError("Model judge exposure lacks its current profile.")
            if (
                type(assertion) is StructuredModelJudgeAssertionSpec
                and profile.revision != assertion.judge_profile_revision
            ):
                raise ValueError("Structured judge exposure lacks its exact current profile.")
            used_judge_profiles[profile.key] = profile
            calls = policy.trial_count
            judge_evaluations += calls
            if (
                profile.max_estimated_cost is None
                or profile.cost_currency is None
                or profile.pricing_profile_fingerprint is None
            ):
                judge_cost_complete = False
            else:
                judge_pricing.add(profile.pricing_profile_fingerprint)
    if judge_evaluations == 0:
        judge_cost = EvalMaximumCostExposureV1(state="not_applicable")
    else:
        judge_cost = EvalMaximumCostExposureV1(
            state="unavailable",
            unavailable_reason=(
                "judge_cost_not_hard_bounded" if judge_cost_complete else "judge_pricing_incomplete"
            ),
            pricing_profile_fingerprints=tuple(sorted(judge_pricing)),
        )

    return EvalSuiteRunExposureV1.create(
        selection_revision=selected.revision,
        trial_policy_revision=policy.revision,
        execution_profiles=tuple(
            EvalExecutionProfileExposureV1(
                case_ids=launch.case_ids,
                execution_profile_revision=launch.execution_profile.revision,
                execution_profile_comparison_revision=(
                    launch.execution_profile.comparison_revision
                ),
                candidate_cost_budget=(
                    None
                    if launch.cost_budget is None
                    else EvalCandidateCostBudgetV1.from_decimal(
                        currency=launch.cost_budget.currency,
                        amount=launch.cost_budget.max_estimated_cost,
                    )
                ),
            )
            for launch in sorted(launches, key=lambda item: item.case_ids)
        ),
        judge_profiles=tuple(
            EvalJudgeProfileExposureV1(
                profile_key=profile.key,
                judge_profile_revision=profile.revision,
                judge_profile_comparison_revision=profile.comparison_revision,
            )
            for profile in sorted(used_judge_profiles.values(), key=lambda item: item.key)
        ),
        candidate_trials=candidate_trials,
        judge_evaluations=judge_evaluations,
        max_concurrency=policy.max_concurrency,
        maximum_candidate_model_steps=maximum_candidate_model_steps,
        maximum_candidate_total_tokens=maximum_candidate_total_tokens,
        maximum_judge_input_tokens=0 if judge_evaluations == 0 else None,
        maximum_judge_output_tokens=0 if judge_evaluations == 0 else None,
        maximum_judge_total_tokens=0 if judge_evaluations == 0 else None,
        candidate_cost=candidate_cost,
        judge_cost=judge_cost,
    )


__all__ = [
    "EvalCandidateLaunchExposure",
    "EvalSuiteLaunchConcurrencyAllocation",
    "allocate_authored_suite_launch_concurrency",
    "compile_authored_suite_run_exposure",
]
