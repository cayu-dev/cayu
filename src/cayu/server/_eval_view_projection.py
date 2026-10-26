"""Pure projections of validated eval reads into dashboard views."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeVar

from fastapi import HTTPException

from cayu.evals.corpus import EvalCaseSpec, EvalSuiteSpec, eval_suite_trial_policy
from cayu.evals.execution import CorpusExecutionResult
from cayu.evals.published import _published_score
from cayu.evals.read_views import (
    EvalCaseResultView,
    EvalCaseView,
    EvalRunCaseFilter,
    EvalRunViewMetadata,
    EvalRunViewSnapshot,
    EvalSuiteViewIdentity,
    EvalTrialViewOutcome,
    EvalViewCaseCounts,
    EvalViewPage,
    EvalViewProgress,
    EvalViewUsage,
    decode_view_token,
    encode_view_token,
)
from cayu.evals.result_presentation import EvalCasePresentationV2, EvalResultPresentationV2
from cayu.evals.results import CapturedEvaluationResultV1
from cayu.evals.store import EvalBaselineRecord, EvalResultRecord, EvalRunStatus
from cayu.evals.suite_authoring import EvalCaseDefinition
from cayu.evals.trial_policy import EvalCaseReliabilityV1
from cayu.server._eval_view_comparison import EvalViewComparisonData

CaseRow = TypeVar("CaseRow", EvalCaseView, EvalCaseResultView)


@dataclass(frozen=True)
class EvalRunViewData:
    """Fully validated run/result, its exact corpus scope and canonical presentation."""

    selector: str
    snapshot: EvalRunViewSnapshot | None
    record: EvalResultRecord | None
    result: CorpusExecutionResult | CapturedEvaluationResultV1 | None
    identity: EvalSuiteViewIdentity
    suite: EvalSuiteSpec
    definitions: tuple[EvalCaseSpec, ...]
    presentation: EvalResultPresentationV2 | None


def _judge_score(
    cases: Sequence[EvalCasePresentationV2],
) -> tuple[float | None, Literal["available", "not_used", "unavailable"]]:
    scores = [
        assertion.score
        for case in cases
        for trial in case.trials
        for assertion in trial.assertions
        if assertion.category == "semantic"
    ]
    if not scores:
        return None, "not_used"
    score = _published_score(scores)
    return score, "unavailable" if score is None else "available"


def _check_label(assertion):
    kind = assertion["kind"]
    if kind == "root_status":
        return f"Session finishes {assertion['expected']}"
    if kind in {"final_output_contains", "final_output_equals"}:
        verb = "includes" if kind == "final_output_contains" else "equals"
        return f"Output {verb}: {assertion['expected'][:200]}"
    return kind.replace("_", " ").capitalize()


def case_view(
    case: EvalCaseSpec | EvalCaseDefinition, last: EvalCasePresentationV2 | None = None
) -> EvalCaseView:
    payload = case.model_dump(mode="json")
    stimulus = payload.get("stimulus")
    input_value = payload.get("input") if stimulus is None else stimulus
    checks = tuple(
        {
            "kind": assertion["kind"],
            "label": _check_label(assertion),
            "judge": assertion["kind"] in {"model_judge", "structured_model_judge"},
            "specification": assertion,
        }
        for assertion in payload["assertions"]
    )
    return EvalCaseView(id=case.id, name=case.name, input=input_value, checks=checks, last=last)


def case_page(
    items: Sequence[CaseRow], *, scope: object, cursor: str | None, limit: int
) -> EvalViewPage[CaseRow]:
    after = ""
    if cursor is not None:
        value = decode_view_token(cursor)
        if (
            not isinstance(value, list)
            or len(value) != 3
            or value[:2] != ["cases", scope]
            or type(value[2]) is not str
        ):
            raise ValueError("Case cursor scope mismatch.")
        after = value[2]

    def case_id(item: CaseRow) -> str:
        return item.case.id if isinstance(item, EvalCaseResultView) else item.id

    ordered = sorted(items, key=case_id)
    remaining = [item for item in ordered if case_id(item) > after]
    selected = remaining[:limit]
    more = len(remaining) > len(selected)
    next_cursor = None
    if more:
        last = selected[-1]
        next_cursor = encode_view_token(["cases", scope, case_id(last)])
    return EvalViewPage(
        items=tuple(selected), total=len(items), has_more=more, next_cursor=next_cursor
    )


def run_metadata(
    data: EvalRunViewData,
    *,
    baseline: EvalBaselineRecord | None,
    specialist: str,
) -> EvalRunViewMetadata:
    """Project a validated run, including its policy-bound live progress."""
    selector, snapshot, record, result = data.selector, data.snapshot, data.record, data.result
    identity, suite, definitions, presentation = (
        data.identity,
        data.suite,
        data.definitions,
        data.presentation,
    )
    counts = None
    reliabilities = None
    if presentation is not None:
        reliabilities = [item.reliability for item in presentation.cases]
    elif snapshot is not None and snapshot.run.status in {
        EvalRunStatus.QUEUED,
        EvalRunStatus.RUNNING,
        EvalRunStatus.CANCELLING,
    }:
        definitions_by_id = {case.id: case for case in definitions}
        policy = eval_suite_trial_policy(suite)
        trials_by_case: dict[str, list[EvalTrialViewOutcome]] = {}
        for trial in snapshot.terminal_trials:
            if trial.case_id not in definitions_by_id or trial.trial_number > policy.trial_count:
                raise HTTPException(status_code=409, detail="Eval progress contradicts its suite.")
            trials_by_case.setdefault(trial.case_id, []).append(trial)
        reliabilities = [
            EvalCaseReliabilityV1.create(
                policy=policy,
                trials=((trial.status, trial.score, trial.diagnostic_code) for trial in trials),
                uses_model_judge=any(
                    assertion.kind in {"model_judge", "structured_model_judge"}
                    for assertion in definitions_by_id[case_id].assertions
                ),
            )
            for case_id, trials in trials_by_case.items()
            if len(trials) == policy.trial_count
        ]
    if reliabilities is not None:
        statuses = Counter(item.outcome for item in reliabilities)
        flaky = sum(0 < item.passed_trials < item.total_trials for item in reliabilities)
        counts = EvalViewCaseCounts(
            total=len(presentation.cases) if presentation is not None else len(definitions),
            passed=statuses["passed"],
            failed=statuses["failed"],
            error=statuses["error"],
            unavailable=statuses["unavailable"],
            flaky=flaky,
        )
    run = None if snapshot is None else snapshot.run
    if run is None:
        assert record is not None
        origin, created_at = str(record.origin), record.created_at
    else:
        origin, created_at = str(run.spec.invocation.source), run.created_at
    actor = (
        run.spec.invocation.origin.subject
        if run and run.spec.invocation.origin is not None
        else None
    )
    usage = EvalViewUsage(state="unavailable", tokens=None, observed_tokens=None)
    if type(result) is CorpusExecutionResult:
        trials = [trial for case in result.run.cases for trial in case.trials]
        observed = [
            trial.usage
            for trial in trials
            if trial.usage is not None and trial.usage_evidence_state in {"complete", "partial"}
        ]
        complete = all(
            trial.usage is not None
            and trial.usage_evidence_state == "complete"
            and trial.evidence_complete
            for trial in trials
        )
        tokens = str(sum(item.total_tokens for item in observed)) if observed else None
        usage = EvalViewUsage(
            state="complete" if complete else "partial" if observed else "unavailable",
            tokens=tokens if complete else None,
            observed_tokens=tokens,
        )
    profile = None if run is None else run.spec.invocation.execution_profile_snapshot
    judge_score, judge_state = (
        (None, "unavailable") if presentation is None else _judge_score(presentation.cases)
    )
    return EvalRunViewMetadata(
        id=selector,
        native_run_id=None if run is None else run.id,
        authored_suite_revision=None
        if run is None
        else run.spec.invocation.authored_suite_revision,
        result_revision=None if record is None else record.revision,
        suite=identity,
        suite_name=suite.name,
        specialist=specialist,
        status=run.status if run else EvalRunStatus.COMPLETED,
        origin=origin,
        actor=actor,
        created_at=created_at,
        started_at=None if run is None else run.started_at,
        finished_at=None if run is None else run.finished_at,
        cases=counts,
        score=None if record is None else record.score,
        judge_score=judge_score,
        judge_score_state=judge_state,
        usage=usage,
        candidate=None if profile is None else profile.candidate,
        candidate_unavailable_reason="not_retained" if profile is None else None,
        baseline=baseline,
        is_baseline=baseline is not None
        and record is not None
        and baseline.result_revision == record.revision,
    )


def run_progress(data: EvalRunViewData) -> EvalViewProgress:
    # run_metadata has validated live slots against the same immutable suite.
    snapshot, result, suite = data.snapshot, data.result, data.suite
    definitions, presentation = data.definitions, data.presentation
    if presentation is not None:
        presented_ids = {case.case_id for case in presentation.cases}
        if type(result) is CorpusExecutionResult:
            if presented_ids != {case.id for case in definitions}:
                raise HTTPException(status_code=409, detail="Eval result case membership mismatch.")
        else:
            definitions = tuple(case for case in definitions if case.id in presented_ids)
    total_trials = (
        len(definitions) * suite.trial_request.trials
        if presentation is None
        else sum(case.reliability.total_trials for case in presentation.cases)
    )
    completed_trials = None
    completed_cases = None
    progress_state = "unavailable"
    if presentation is not None:
        completed_trials = sum(len(case.trials) for case in presentation.cases)
        completed_cases = len(presentation.cases)
        progress_state = "published"
    elif snapshot is not None and snapshot.run.status in {
        EvalRunStatus.QUEUED,
        EvalRunStatus.RUNNING,
        EvalRunStatus.CANCELLING,
    }:
        slots = snapshot.completed_trials_by_case
        completed_trials = sum(slots.values())
        completed_cases = sum(value == suite.trial_request.trials for value in slots.values())
        progress_state = "queued" if snapshot.run.status == EvalRunStatus.QUEUED else "checkpointed"
    return EvalViewProgress(
        total_cases=len(definitions),
        total_trials=total_trials,
        completed_cases=completed_cases,
        completed_trials=completed_trials,
        state=progress_state,
    )


def case_results(
    data: EvalRunViewData,
    previous_data: EvalRunViewData | None,
    comparison: EvalViewComparisonData,
    *,
    case_filter: EvalRunCaseFilter | None = None,
) -> list[EvalCaseResultView]:
    presentation, result, definitions = data.presentation, data.result, data.definitions
    previous = (
        previous_data.presentation
        if previous_data is not None and comparison.summary.state == "compared"
        else None
    )
    previous_result = None if previous_data is None else previous_data.result
    results: list[EvalCaseResultView] = []
    if presentation is not None:
        definition_map = {case.id: case for case in definitions}
        previous_cases = {} if previous is None else {case.case_id: case for case in previous.cases}
        evidence = (
            {case.case_id: case.model_dump(mode="json") for case in result.run.cases}
            if type(result) is CorpusExecutionResult
            else {}
        )
        previous_evidence = (
            {case.case_id: case.model_dump(mode="json") for case in previous_result.run.cases}
            if type(previous_result) is CorpusExecutionResult and previous is not None
            else {}
        )
        for case in presentation.cases:
            if case_filter is not None:
                matches = {
                    "regressed": case.case_id in comparison.regressed,
                    "improved": case.case_id in comparison.improved,
                    "failing": case.status in {"failed", "error", "unavailable"},
                    "flaky": 0 < case.reliability.passed_trials < case.reliability.total_trials,
                }
                if not matches[case_filter]:
                    continue
            definition = definition_map.get(case.case_id)
            if definition is None or definition.revision != case.case_revision:
                raise HTTPException(status_code=409, detail="Eval result case identity mismatch.")
            change = (
                "uncompared"
                if comparison.summary.state != "compared"
                else "tradeoff"
                if case.case_id in comparison.improved and case.case_id in comparison.regressed
                else "improved"
                if case.case_id in comparison.improved
                else "regressed"
                if case.case_id in comparison.regressed
                else "unchanged"
            )
            judge_score, judge_state = _judge_score((case,))
            results.append(
                EvalCaseResultView(
                    case=case_view(definition),
                    presentation=case,
                    judge_score=judge_score,
                    judge_score_state=judge_state,
                    evidence=evidence.get(case.case_id),
                    compared=previous_cases.get(case.case_id),
                    compared_evidence=previous_evidence.get(case.case_id),
                    structured_judgments=tuple(
                        item
                        for item in comparison.structured_judgments
                        if item.case_id == case.case_id
                    ),
                    tool_json_assertions=tuple(
                        item
                        for item in comparison.tool_json_assertions
                        if item.case_id == case.case_id
                    ),
                    change=change,
                    regressions=tuple(
                        item for item in comparison.regressions if item.case_id == case.case_id
                    ),
                    improvements=tuple(
                        item for item in comparison.improvements if item.case_id == case.case_id
                    ),
                )
            )
    return results
