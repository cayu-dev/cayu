from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Literal

import pytest
from tests.evals.test_corpus_execution import _corpus, _provider, _target

from cayu.evals.corpus import (
    CorpusUserMessageSpec,
    EvaluationEvidencePolicySpec,
    JudgeProfileIdentityV1,
    ModelJudgeAssertionSpec,
    RootStatusAssertionSpec,
    RunInputSpec,
    StructuredModelJudgeAssertionSpec,
    StructuredRubricCriterionV1,
    StructuredRubricV1,
    _content_revision,
)
from cayu.evals.execution import run_corpus_suite
from cayu.evals.execution_comparison import (
    CorpusComparisonReason,
    eval_result_compatibility,
)
from cayu.evals.execution_profiles import (
    EvalExecutionCandidateIdentityV1,
    EvalExecutionProfileV1,
    EvalExecutionResourceCeilingsV1,
    EvalExecutionTargetMaterialIdentityV1,
)
from cayu.evals.results import eval_result_projection
from cayu.evals.store import EvalRunCostBudget
from cayu.evals.suite_authoring import (
    EvalCaseDraftV1,
    EvalCaseDraftV2,
    EvalSimpleInputStimulusV1,
    EvalSuiteDraftV1,
    EvalSuiteDraftV2,
    StructuredModelJudgeAssertionDraftV1,
    compile_eval_suite_draft,
    compile_eval_suite_draft_v2,
    eval_suite_selection,
)
from cayu.evals.suite_preflight import (
    EvalCandidateLaunchExposure,
    allocate_authored_suite_launch_concurrency,
    compile_authored_suite_run_exposure,
)
from cayu.evals.trial_policy import EvalSuiteTrialPolicyV1
from cayu.runtime.stop_policy import RunLimits


def _profile(
    evidence_policy: EvaluationEvidencePolicySpec,
    *,
    profile_id: str = "generated",
    label: str = "Generated profile",
    source: Literal["generated", "explicit"] = "generated",
    application_release_id: str = "release-1",
    app_manifest_fingerprint: str = "a" * 64,
    runtime_execution_profile_fingerprint: str = "b" * 64,
) -> EvalExecutionProfileV1:
    return EvalExecutionProfileV1.create(
        profile_id=profile_id,
        label=label,
        source=source,
        target_key="assistant.default",
        application_release_id=application_release_id,
        app_manifest_fingerprint=app_manifest_fingerprint,
        candidate=EvalExecutionCandidateIdentityV1(
            agent_name="assistant",
            provider_name="provider",
            model="model",
            runtime_execution_profile_schema_version=1,
            runtime_execution_profile_fingerprint=runtime_execution_profile_fingerprint,
        ),
        target_material=EvalExecutionTargetMaterialIdentityV1(
            kind="structural_sha256",
            fingerprint="c" * 64,
        ),
        fixture_strategy="none",
        reset_strategy="fresh_session_only",
        effect_posture="ordinary_application_authority",
        isolation_revision=None,
        evidence_policy=evidence_policy,
        ceilings=EvalExecutionResourceCeilingsV1(
            max_cases=1,
            max_trials=1,
            max_concurrency=1,
            max_timeout_seconds=120,
            max_bootstrap_messages=0,
            max_total_input_chars=1_000,
            max_compiled_input_chars=1_000,
            max_steps=4,
            run_limits=RunLimits(
                max_input_tokens=10,
                max_output_tokens=10,
                max_total_tokens=20,
            ),
        ),
    )


def _judge_profile(*, label: str = "Quality judge") -> JudgeProfileIdentityV1:
    values = {
        "schema_version": 1,
        "key": "quality-judge",
        "label": label,
        "provider_name": "provider",
        "model": "judge-model",
        "implementation_revision": "sha256:" + "a" * 64,
        "allowed_evidence": ["final_output"],
        "timeout_seconds": 120,
        "max_input_tokens": 1_000,
        "max_output_tokens": 200,
        "max_total_tokens": 1_200,
        "max_estimated_cost": "0.25",
        "cost_currency": "USD",
        "pricing_profile_fingerprint": "sha256:" + "b" * 64,
        "privacy_policy_key": "public-only",
        "privacy_policy_revision": "sha256:" + "c" * 64,
        "same_model_use": "forbidden",
    }
    return JudgeProfileIdentityV1(
        revision=_content_revision(values, "judge profile identity"),
        **values,
    )


def test_suite_concurrency_is_partitioned_across_durable_lanes() -> None:
    allocations = allocate_authored_suite_launch_concurrency(
        EvalSuiteTrialPolicyV1.create(
            trial_count=1,
            max_concurrency=5,
        ),
        7,
    )

    assert tuple(item.lane for item in allocations) == (0, 1, 2, 3, 4, 0, 1)
    assert tuple(item.max_concurrency for item in allocations) == (1, 1, 1, 1, 1, 1, 1)

    fewer_runs = allocate_authored_suite_launch_concurrency(
        EvalSuiteTrialPolicyV1.create(
            trial_count=1,
            max_concurrency=5,
        ),
        3,
    )
    assert tuple((item.lane, item.max_concurrency) for item in fewer_runs) == (
        (0, 2),
        (1, 2),
        (2, 1),
    )
    assert sum(item.max_concurrency for item in fewer_runs) == 5


def test_candidate_stop_thresholds_are_not_advertised_as_hard_maxima() -> None:
    document = compile_eval_suite_draft(
        EvalSuiteDraftV1(
            id="candidate-exposure",
            target_key="assistant.default",
            name="Candidate exposure",
            cases=(
                EvalCaseDraftV1(
                    id="case-one",
                    name="Case one",
                    stimulus=EvalSimpleInputStimulusV1(
                        input=RunInputSpec(messages=(CorpusUserMessageSpec(text="Evaluate this."),))
                    ),
                    assertions=(RootStatusAssertionSpec(id="completed", expected="completed"),),
                ),
            ),
        )
    )
    selection = eval_suite_selection(document)
    exposure = compile_authored_suite_run_exposure(
        document,
        selection,
        (
            EvalCandidateLaunchExposure(
                case_ids=("case-one",),
                execution_profile=_profile(EvaluationEvidencePolicySpec.create()),
                cost_budget=EvalRunCostBudget(
                    max_estimated_cost=Decimal("0.25"),
                    currency="USD",
                ),
            ),
        ),
        judge_profiles=(),
        candidate_pricing_profile_fingerprint="sha256:" + "d" * 64,
    )

    assert exposure.maximum_candidate_total_tokens is None
    assert exposure.candidate_cost.state == "unavailable"
    assert exposure.candidate_cost.totals == ()
    assert exposure.candidate_cost.unavailable_reason == "candidate_cost_not_hard_bounded"
    assert exposure.candidate_cost.pricing_profile_fingerprints == ("sha256:" + "d" * 64,)


def test_compiled_exposure_compares_across_releases_but_rejects_execution_drift() -> None:
    document = compile_eval_suite_draft(
        EvalSuiteDraftV1(
            id="cross-release-exposure",
            target_key="assistant.default",
            name="Cross-release exposure",
            cases=(
                EvalCaseDraftV1(
                    id="case-one",
                    name="Case one",
                    stimulus=EvalSimpleInputStimulusV1(
                        input=RunInputSpec(messages=(CorpusUserMessageSpec(text="Evaluate this."),))
                    ),
                    assertions=(RootStatusAssertionSpec(id="completed", expected="completed"),),
                ),
            ),
        )
    )
    selection = eval_suite_selection(document)

    def compile_exposure(profile: EvalExecutionProfileV1):
        return compile_authored_suite_run_exposure(
            document,
            selection,
            (
                EvalCandidateLaunchExposure(
                    case_ids=("case-one",),
                    execution_profile=profile,
                ),
            ),
            judge_profiles=(),
            candidate_pricing_profile_fingerprint=None,
        )

    baseline_profile = _profile(EvaluationEvidencePolicySpec.create())
    release_profile = _profile(
        EvaluationEvidencePolicySpec.create(),
        profile_id="release-profile",
        label="Release profile",
        source="explicit",
        application_release_id="release-2",
        app_manifest_fingerprint="d" * 64,
    )
    changed_execution_profile = _profile(
        EvaluationEvidencePolicySpec.create(),
        application_release_id="release-2",
        app_manifest_fingerprint="d" * 64,
        runtime_execution_profile_fingerprint="e" * 64,
    )
    baseline_exposure = compile_exposure(baseline_profile)
    release_exposure = compile_exposure(release_profile)
    changed_execution_exposure = compile_exposure(changed_execution_profile)

    assert baseline_profile.revision != release_profile.revision
    assert baseline_profile.comparison_revision == release_profile.comparison_revision
    assert baseline_exposure.revision != release_exposure.revision
    assert baseline_exposure.comparison_revision == release_exposure.comparison_revision
    assert baseline_exposure.comparison_revision != changed_execution_exposure.comparison_revision

    fresh = asyncio.run(
        run_corpus_suite(
            _target(_provider(trials=1)),
            _corpus(trials=1),
            "refund-regressions",
            max_concurrency=1,
        )
    )
    projection = eval_result_projection(fresh)

    def with_exposure(exposure):
        return projection.model_copy(
            update={
                "accepted_exposure_revision": exposure.revision,
                "accepted_exposure_comparison_revision": exposure.comparison_revision,
            }
        )

    baseline = with_exposure(baseline_exposure)
    release = with_exposure(release_exposure)
    changed_execution = with_exposure(changed_execution_exposure)
    assert eval_result_compatibility(baseline, release).comparable is True
    assert eval_result_compatibility(baseline, changed_execution).reasons == (
        CorpusComparisonReason.ACCEPTED_EXPOSURE_CONTRACT_MISMATCH,
    )

    forged_profile = baseline_profile.model_copy(
        update={"candidate": baseline_profile.candidate.model_copy(update={"model": "forged"})}
    )
    with pytest.raises(ValueError, match="revision does not match"):
        _ = forged_profile.comparison_revision
    forged_exposure = baseline_exposure.model_copy(update={"candidate_trials": 2})
    with pytest.raises(ValueError, match="revision does not match"):
        _ = forged_exposure.comparison_revision


def test_execution_exposure_preserves_case_to_profile_binding() -> None:
    document = compile_eval_suite_draft(
        EvalSuiteDraftV1(
            id="bound-profile-exposure",
            target_key="assistant.default",
            name="Bound profile exposure",
            cases=tuple(
                EvalCaseDraftV1(
                    id=case_id,
                    name=case_id,
                    stimulus=EvalSimpleInputStimulusV1(
                        input=RunInputSpec(
                            messages=(CorpusUserMessageSpec(text=f"Evaluate {case_id}."),)
                        )
                    ),
                    assertions=(RootStatusAssertionSpec(id="completed", expected="completed"),),
                )
                for case_id in ("case-a", "case-b")
            ),
        )
    )
    profile_a = _profile(
        EvaluationEvidencePolicySpec.create(),
        runtime_execution_profile_fingerprint="a" * 64,
    )
    profile_b = _profile(
        EvaluationEvidencePolicySpec.create(),
        runtime_execution_profile_fingerprint="b" * 64,
    )

    def compile_with(
        first: EvalExecutionProfileV1,
        second: EvalExecutionProfileV1,
        *,
        first_budget: Decimal | None = None,
        second_budget: Decimal | None = None,
    ):
        return compile_authored_suite_run_exposure(
            document,
            eval_suite_selection(document),
            (
                EvalCandidateLaunchExposure(
                    case_ids=("case-a",),
                    execution_profile=first,
                    cost_budget=(
                        None
                        if first_budget is None
                        else EvalRunCostBudget(max_estimated_cost=first_budget)
                    ),
                ),
                EvalCandidateLaunchExposure(
                    case_ids=("case-b",),
                    execution_profile=second,
                    cost_budget=(
                        None
                        if second_budget is None
                        else EvalRunCostBudget(max_estimated_cost=second_budget)
                    ),
                ),
            ),
            judge_profiles=(),
            candidate_pricing_profile_fingerprint=None,
        )

    original = compile_with(profile_a, profile_b)
    swapped = compile_with(profile_b, profile_a)

    assert original.revision != swapped.revision
    assert original.comparison_revision != swapped.comparison_revision
    assert original.execution_profiles[0].case_ids == ("case-a",)
    assert original.execution_profiles[0].execution_profile_revision == profile_a.revision

    budgeted = compile_with(
        profile_a,
        profile_b,
        first_budget=Decimal("0.1"),
        second_budget=Decimal("0.2"),
    )
    swapped_budgets = compile_with(
        profile_a,
        profile_b,
        first_budget=Decimal("0.2"),
        second_budget=Decimal("0.1"),
    )
    assert budgeted.candidate_cost == swapped_budgets.candidate_cost
    assert budgeted.revision != swapped_budgets.revision
    assert budgeted.comparison_revision != swapped_budgets.comparison_revision


def test_judge_stop_thresholds_are_not_advertised_as_hard_maxima() -> None:
    profile = _judge_profile()
    rubric = StructuredRubricV1.create(
        id="quality",
        criteria=(
            StructuredRubricCriterionV1(
                id="correctness",
                name="Correctness",
                description="The answer is correct.",
                weight="1",
            ),
        ),
    )
    judge_assertion = StructuredModelJudgeAssertionSpec(
        id="quality",
        judge_profile_key=profile.key,
        judge_profile_revision=profile.revision,
        rubric=rubric,
    )
    document = compile_eval_suite_draft_v2(
        EvalSuiteDraftV2(
            id="judge-exposure",
            target_key="assistant.default",
            name="Judge exposure",
            cases=(
                EvalCaseDraftV2(
                    id="case-one",
                    name="Case one",
                    stimulus=EvalSimpleInputStimulusV1(
                        input=RunInputSpec(messages=(CorpusUserMessageSpec(text="Evaluate this."),))
                    ),
                    assertions=(
                        StructuredModelJudgeAssertionDraftV1.from_assertion(judge_assertion),
                    ),
                ),
            ),
        )
    )
    exposure = compile_authored_suite_run_exposure(
        document,
        eval_suite_selection(document),
        (
            EvalCandidateLaunchExposure(
                case_ids=("case-one",),
                execution_profile=_profile(EvaluationEvidencePolicySpec.create()),
            ),
        ),
        judge_profiles=(profile,),
        candidate_pricing_profile_fingerprint=None,
    )

    assert exposure.judge_evaluations == 1
    assert exposure.maximum_judge_input_tokens is None
    assert exposure.maximum_judge_output_tokens is None
    assert exposure.maximum_judge_total_tokens is None
    assert exposure.judge_cost.state == "unavailable"
    assert exposure.judge_cost.totals == ()
    assert exposure.judge_cost.unavailable_reason == "judge_cost_not_hard_bounded"
    assert exposure.judge_cost.pricing_profile_fingerprints == ("sha256:" + "b" * 64,)
    assert exposure.judge_profiles[0].profile_key == profile.key


def test_rubric_string_model_judge_is_included_in_work_and_variability_exposure() -> None:
    profile = _judge_profile()
    document = compile_eval_suite_draft(
        EvalSuiteDraftV1(
            id="model-judge-exposure",
            target_key="assistant.default",
            name="Model judge exposure",
            cases=(
                EvalCaseDraftV1(
                    id="case-one",
                    name="Case one",
                    stimulus=EvalSimpleInputStimulusV1(
                        input=RunInputSpec(messages=(CorpusUserMessageSpec(text="Evaluate."),))
                    ),
                    assertions=(
                        ModelJudgeAssertionSpec(
                            id="quality",
                            evaluator_key=profile.key,
                            rubric="Judge correctness and completeness.",
                            rubric_version="v1",
                        ),
                    ),
                ),
            ),
        )
    )
    exposure = compile_authored_suite_run_exposure(
        document,
        eval_suite_selection(document),
        (
            EvalCandidateLaunchExposure(
                case_ids=("case-one",),
                execution_profile=_profile(EvaluationEvidencePolicySpec.create()),
            ),
        ),
        judge_profiles=(profile,),
        candidate_pricing_profile_fingerprint=None,
    )

    assert exposure.judge_evaluations == 1
    assert exposure.judge_profiles[0].judge_profile_revision == profile.revision
    assert exposure.judge_profiles[0].judge_profile_comparison_revision == (
        profile.comparison_revision
    )
    relabeled = _judge_profile(label="Renamed judge")
    assert relabeled.revision != profile.revision
    assert relabeled.comparison_revision == profile.comparison_revision
