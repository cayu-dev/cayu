"""Adapt the canonical comparator once into consistently oriented read views.

A comparison of two results is a pure function of the pair, so a store records it
(`EvalViewComparisonRecordV1`) and readers reuse it. `record_baseline_comparison`
records a result's comparison with its suite baseline when it is published.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from cayu.evals.execution import CorpusExecutionResult
from cayu.evals.execution_comparison import (
    CorpusExecutionComparison,
    CorpusExecutionRegression,
    EvalStructuredJudgeComparisonV1,
    EvalToolJsonAssertionComparisonV1,
    compare_eval_results,
)
from cayu.evals.read_views import EvalRunViewSnapshot, EvalViewComparison, EvalViewImprovement
from cayu.evals.results import CapturedEvaluationResultV1, eval_result_projection
from cayu.evals.store import EvalBaselineKey, EvalRunStatus
from cayu.evals.view_summaries import EvalViewComparisonRecordV1

if TYPE_CHECKING:
    from cayu.evals.store import EvalStore

_LOGGER = logging.getLogger(__name__)


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


def pending_comparison(
    current_snapshot: EvalRunViewSnapshot | None,
    previous_snapshot: EvalRunViewSnapshot | None,
    *,
    current_has_result: bool,
    selected: str | None,
) -> EvalViewComparisonData:
    """The comparison of a run whose own result, or the compared run's, is missing."""
    snapshot = current_snapshot if not current_has_result else previous_snapshot
    terminal = snapshot is not None and snapshot.run.status in {
        EvalRunStatus.FAILED,
        EvalRunStatus.CANCELLED,
    }
    reason = "current_run_has_no_result" if not current_has_result else "compared_run_has_no_result"
    return EvalViewComparisonData(
        EvalViewComparison(
            state="unavailable" if terminal else "pending",
            compared_with=selected,
            unavailable_reasons=(reason,) if terminal else (),
        )
    )


def comparison_record(data: EvalViewComparisonData) -> EvalViewComparisonRecordV1:
    """The selector-free record of a comparison of two results."""
    return EvalViewComparisonRecordV1(
        summary=data.summary.model_copy(update={"compared_with": None}),
        improved=tuple(sorted(data.improved)),
        regressed=tuple(sorted(data.regressed)),
        regressions=data.regressions,
        improvements=data.improvements,
        structured_judgments=data.structured_judgments,
        tool_json_assertions=data.tool_json_assertions,
    )


def comparison_from_record(
    record: EvalViewComparisonRecordV1, selected: str | None
) -> EvalViewComparisonData:
    summary = record.summary
    if summary.state != "no_baseline":
        summary = summary.model_copy(update={"compared_with": selected})
    return EvalViewComparisonData(
        summary=EvalViewComparison.model_validate(summary.model_dump(mode="python")),
        improved=frozenset(record.improved),
        regressed=frozenset(record.regressed),
        regressions=record.regressions,
        improvements=record.improvements,
        structured_judgments=record.structured_judgments,
        tool_json_assertions=record.tool_json_assertions,
    )


def compare_view_results(
    current: CorpusExecutionResult | CapturedEvaluationResultV1 | None,
    current_snapshot: EvalRunViewSnapshot | None,
    previous: CorpusExecutionResult | CapturedEvaluationResultV1 | None,
    previous_snapshot: EvalRunViewSnapshot | None,
    selected: str | None,
) -> EvalViewComparisonData:
    if current is None or (selected is not None and previous is None):
        return pending_comparison(
            current_snapshot,
            previous_snapshot,
            current_has_result=current is not None,
            selected=selected,
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


async def record_result_comparison(
    store: EvalStore,
    current: CorpusExecutionResult | CapturedEvaluationResultV1,
    previous: CorpusExecutionResult | CapturedEvaluationResultV1,
) -> EvalViewComparisonRecordV1:
    """Compare two stored results and record the comparison, unless it is recorded."""
    recorded = await store.load_result_comparison(current.revision, previous.revision)
    if recorded is not None:
        return recorded
    value = await asyncio.to_thread(compare_view_results, current, None, previous, None, None)
    record = comparison_record(value)
    await store.save_result_comparison(current.revision, previous.revision, record)
    return record


async def record_baseline_comparison(
    store: EvalStore, result: CorpusExecutionResult | CapturedEvaluationResultV1
) -> None:
    """Record a newly stored result's comparison with its suite's baseline, if any.

    Run views compare a run with the baseline, so a page read after publication
    finds the comparison recorded instead of decoding both results. This is an
    optimization: a failure is logged, and readers compare the pair themselves.
    """
    if not store.view_summaries:
        return
    try:
        projection = eval_result_projection(result)
        baseline = await store.load_baseline(
            EvalBaselineKey(
                target_key=projection.target.target_key,
                corpus_revision=projection.corpus_revision,
                suite_id=projection.suite_id,
            )
        )
        if baseline is None:
            return
        previous = (
            result
            if baseline.result_revision == result.revision
            else await store.load_result_by_revision(baseline.result_revision)
        )
        if previous is not None:
            await record_result_comparison(store, result, previous)
    except Exception:
        _LOGGER.warning("Could not record an eval result's baseline comparison.", exc_info=True)


async def record_suite_baseline_comparisons(
    store: EvalStore, key: EvalBaselineKey, *, limit: int = 20
) -> int:
    """Record the comparisons of a suite's latest results with its current baseline.

    After a baseline changes, the suite's pages compare its latest runs with the
    new baseline; recording those comparisons first keeps the pages from deriving
    them. Returns how many it recorded. Failures are logged, not raised.
    """
    from cayu.evals.read_views import EvalSuiteViewIdentity, EvalViewQuery

    if not store.view_summaries:
        return 0
    recorded = 0
    try:
        baseline = await store.load_baseline(key)
        if baseline is None:
            return 0
        previous = await store.load_result_by_revision(baseline.result_revision)
        if previous is None:
            return 0
        identity = EvalSuiteViewIdentity(
            source="corpus",
            target_key=key.target_key,
            revision=key.corpus_revision,
            suite_id=key.suite_id,
        )
        page = await store.list_view_index(
            EvalViewQuery(target_key=key.target_key, suite=identity, limit=limit)
        )
        for item in page.items:
            if item.source == "result":
                revision = item.revision
            elif item.source == "run":
                run = await store.load_run(item.revision)
                revision = None if run is None or run.result is None else run.result.revision
            else:
                continue
            if revision is None:
                continue
            if await store.load_result_comparison(revision, previous.revision) is not None:
                continue
            current = (
                previous
                if revision == previous.revision
                else await store.load_result_by_revision(revision)
            )
            if current is not None:
                await record_result_comparison(store, current, previous)
                recorded += 1
    except Exception:
        _LOGGER.warning("Could not record a suite's baseline comparisons.", exc_info=True)
    return recorded
