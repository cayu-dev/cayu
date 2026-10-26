"""Adapt the canonical comparator once into consistently oriented read views."""

from __future__ import annotations

from dataclasses import dataclass

from cayu.evals.execution import CorpusExecutionResult
from cayu.evals.execution_comparison import (
    CorpusExecutionComparison,
    CorpusExecutionRegression,
    EvalStructuredJudgeComparisonV1,
    EvalToolJsonAssertionComparisonV1,
    compare_eval_results,
)
from cayu.evals.read_views import EvalRunViewSnapshot, EvalViewComparison, EvalViewImprovement
from cayu.evals.results import CapturedEvaluationResultV1
from cayu.evals.store import EvalRunStatus


@dataclass(frozen=True)
class EvalViewComparisonData:
    summary: EvalViewComparison
    improved: frozenset[str] = frozenset()
    regressed: frozenset[str] = frozenset()
    regressions: tuple[CorpusExecutionRegression, ...] = ()
    improvements: tuple[EvalViewImprovement, ...] = ()
    structured_judgments: tuple[EvalStructuredJudgeComparisonV1, ...] = ()
    tool_json_assertions: tuple[EvalToolJsonAssertionComparisonV1, ...] = ()


def _changed_cases(comparison: CorpusExecutionComparison) -> frozenset[str]:
    return frozenset(
        [item.case_id for item in comparison.regressions if item.case_id is not None]
        + [item.case_id for item in comparison.structured_judgments if item.regressed]
        + [item.case_id for item in comparison.tool_json_assertions if item.regressed]
    )


def compare_view_results(
    current: CorpusExecutionResult | CapturedEvaluationResultV1 | None,
    current_snapshot: EvalRunViewSnapshot | None,
    previous: CorpusExecutionResult | CapturedEvaluationResultV1 | None,
    previous_snapshot: EvalRunViewSnapshot | None,
    selected: str | None,
) -> EvalViewComparisonData:
    if current is None or (selected is not None and previous is None):
        snapshot = current_snapshot if current is None else previous_snapshot
        terminal = snapshot is not None and snapshot.run.status in {
            EvalRunStatus.FAILED,
            EvalRunStatus.CANCELLED,
        }
        reason = "current_run_has_no_result" if current is None else "compared_run_has_no_result"
        return EvalViewComparisonData(
            EvalViewComparison(
                state="unavailable" if terminal else "pending",
                compared_with=selected,
                unavailable_reasons=(reason,) if terminal else (),
            )
        )
    if previous is None:
        return EvalViewComparisonData(EvalViewComparison(state="no_baseline"))
    forward = compare_eval_results(previous, current)
    unavailable_reasons = tuple(
        label
        for state, label in (
            (
                forward.structured_judge_comparison_state,
                "structured_judge_observation_identity_mismatch",
            ),
            (forward.tool_json_comparison_state, "tool_json_observation_identity_mismatch"),
        )
        if state == "observation_identity_mismatch"
    )
    if not forward.compatibility.comparable or unavailable_reasons:
        return EvalViewComparisonData(
            EvalViewComparison(
                state="incompatible",
                compared_with=selected,
                compatibility=forward.compatibility,
                unavailable_reasons=unavailable_reasons,
            )
        )
    reverse = compare_eval_results(current, previous)
    improved, regressed = _changed_cases(reverse), _changed_cases(forward)
    # The reverse comparator detects improvements but names current as baseline.
    # Normalize every value pair here, before either run or case projection.
    improvements = tuple(
        EvalViewImprovement.model_validate(
            {
                **item.model_dump(mode="python"),
                "baseline_status": item.current_status,
                "current_status": item.baseline_status,
                "baseline_score": item.current_score,
                "current_score": item.baseline_score,
                "baseline_reliability": item.current_reliability,
                "current_reliability": item.baseline_reliability,
            }
        )
        for item in reverse.regressions
    )
    has_improvement = bool(improvements or improved)
    has_regression = bool(forward.regressions or regressed)
    verdict = (
        "tradeoff"
        if has_improvement and has_regression
        else "improvement"
        if has_improvement
        else "regression"
        if has_regression
        else "tie"
    )
    return EvalViewComparisonData(
        summary=EvalViewComparison(
            state="compared",
            compared_with=selected,
            verdict=verdict,
            compatibility=forward.compatibility,
            improved_cases=len(improved),
            regressed_cases=len(regressed),
            regressions=tuple(item for item in forward.regressions if item.case_id is None),
            improvements=tuple(item for item in improvements if item.case_id is None),
        ),
        improved=improved,
        regressed=regressed,
        regressions=forward.regressions,
        improvements=improvements,
        structured_judgments=forward.structured_judgments,
        tool_json_assertions=forward.tool_json_assertions,
    )
