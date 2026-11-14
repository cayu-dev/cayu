"""Pure projections of validated eval reads into dashboard views.

The projections read a run's lifecycle snapshot, its result's summary and its
cases' outcomes, never a whole result document. The reader gets those parts
from the store's summary tables or, for a store without them, derives them from
the decoded document with the same functions the store's writers use.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from fastapi import HTTPException

from cayu.evals.corpus import EvalCaseSpec, EvalSuiteSpec, eval_suite_trial_policy
from cayu.evals.read_views import (
    EvalCaseResultView,
    EvalCaseView,
    EvalReportedCaseView,
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
from cayu.evals.result_presentation import EvalCasePresentationV2
from cayu.evals.store import EvalBaselineRecord, EvalResultRecord, EvalRunStatus
from cayu.evals.suite_authoring import EvalCaseDefinition
from cayu.evals.trial_policy import EvalCaseReliabilityV1
from cayu.evals.view_comparison import EvalViewComparisonData
from cayu.evals.view_summaries import (
    EvalCaseOutcomeQuery,
    EvalCaseOutcomeV1,
    EvalResultViewSummaryV1,
    case_counts,
)

CaseRow = TypeVar("CaseRow", EvalCaseView, EvalCaseResultView, EvalReportedCaseView)

_MODEL_JUDGES = frozenset({"model_judge", "structured_model_judge"})
_LIVE = frozenset({EvalRunStatus.QUEUED, EvalRunStatus.RUNNING, EvalRunStatus.CANCELLING})


@dataclass(frozen=True, slots=True)
class EvalCaseDefinitionData:
    """What run views need of one case specification."""

    id: str
    revision: str
    view: EvalCaseView
    uses_model_judge: bool


@dataclass(frozen=True)
class EvalRunViewData:
    """A run or result, its exact corpus scope and its result's summary."""

    selector: str
    snapshot: EvalRunViewSnapshot | None
    record: EvalResultRecord | None
    identity: EvalSuiteViewIdentity
    suite: EvalSuiteSpec
    definitions: tuple[EvalCaseDefinitionData, ...]
    summary: EvalResultViewSummaryV1 | None
    fresh: bool

    @property
    def result_revision(self) -> str | None:
        return None if self.record is None else self.record.revision


def _check_label(assertion):
    kind = assertion["kind"]
    if kind == "root_status":
        return f"Session finishes {assertion['expected']}"
    if kind in {"final_output_contains", "final_output_equals"}:
        verb = "includes" if kind == "final_output_contains" else "equals"
        return f"Output {verb}: {assertion['expected'][:200]}"
    return kind.replace("_", " ").capitalize()


def case_view_from_payload(
    payload: dict[str, Any], last: EvalCasePresentationV2 | None = None
) -> EvalCaseView:
    """Case view of a case's JSON form, as the corpus document stores it."""
    stimulus = payload.get("stimulus")
    input_value = payload.get("input") if stimulus is None else stimulus
    checks = tuple(
        {
            "kind": assertion["kind"],
            "label": _check_label(assertion),
            "judge": assertion["kind"] in _MODEL_JUDGES,
            "specification": assertion,
        }
        for assertion in payload["assertions"]
    )
    return EvalCaseView(
        id=payload["id"], name=payload["name"], input=input_value, checks=checks, last=last
    )


def case_definition(payload: dict[str, Any]) -> EvalCaseDefinitionData:
    return EvalCaseDefinitionData(
        id=payload["id"],
        revision=payload["revision"],
        view=case_view_from_payload(payload),
        uses_model_judge=any(item["kind"] in _MODEL_JUDGES for item in payload["assertions"]),
    )


def case_definitions(
    cases: Sequence[EvalCaseSpec | EvalCaseDefinition],
) -> tuple[EvalCaseDefinitionData, ...]:
    return tuple(case_definition(case.model_dump(mode="json")) for case in cases)


def case_page(
    items: Sequence[CaseRow], *, scope: object, cursor: str | None, limit: int
) -> EvalViewPage[CaseRow]:
    after = case_cursor_after(cursor, scope)

    def case_id(item: CaseRow) -> str:
        return item.case.id if isinstance(item, EvalCaseResultView) else item.id

    ordered = sorted(items, key=case_id)
    remaining = [item for item in ordered if case_id(item) > (after or "")]
    selected = remaining[:limit]
    more = len(remaining) > len(selected)
    return EvalViewPage(
        items=tuple(selected),
        total=len(items),
        has_more=more,
        next_cursor=case_cursor(scope, case_id(selected[-1])) if more else None,
    )


def case_cursor_after(cursor: str | None, scope: object) -> str | None:
    if cursor is None:
        return None
    value = decode_view_token(cursor)
    if (
        not isinstance(value, list)
        or len(value) != 3
        or value[:2] != ["cases", scope]
        or type(value[2]) is not str
    ):
        raise ValueError("Case cursor scope mismatch.")
    return value[2]


def case_cursor(scope: object, case_id: str) -> str:
    return encode_view_token(["cases", scope, case_id])


def _live_counts(data: EvalRunViewData) -> EvalViewCaseCounts | None:
    snapshot = data.snapshot
    if snapshot is None or snapshot.run.status not in _LIVE:
        return None
    definitions = {case.id: case for case in data.definitions}
    policy = eval_suite_trial_policy(data.suite)
    trials_by_case: dict[str, list[EvalTrialViewOutcome]] = {}
    for trial in snapshot.terminal_trials:
        if trial.case_id not in definitions or trial.trial_number > policy.trial_count:
            raise HTTPException(status_code=409, detail="Eval progress contradicts its suite.")
        trials_by_case.setdefault(trial.case_id, []).append(trial)
    reliabilities = [
        EvalCaseReliabilityV1.create(
            policy=policy,
            trials=((trial.status, trial.score, trial.diagnostic_code) for trial in trials),
            uses_model_judge=definitions[case_id].uses_model_judge,
        )
        for case_id, trials in trials_by_case.items()
        if len(trials) == policy.trial_count
    ]
    return case_counts(reliabilities, total=len(data.definitions))


def run_metadata(
    data: EvalRunViewData,
    *,
    baseline: EvalBaselineRecord | None,
    specialist: str,
) -> EvalRunViewMetadata:
    """Project a validated run, including its policy-bound live progress."""
    snapshot, record, summary = data.snapshot, data.record, data.summary
    counts = summary.cases if summary is not None else _live_counts(data)
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
    usage = (
        summary.usage
        if summary is not None
        else EvalViewUsage(state="unavailable", tokens=None, observed_tokens=None)
    )
    profile = None if run is None else run.spec.invocation.execution_profile_snapshot
    return EvalRunViewMetadata(
        id=data.selector,
        native_run_id=None if run is None else run.id,
        authored_suite_revision=None
        if run is None
        else run.spec.invocation.authored_suite_revision,
        result_revision=data.result_revision,
        suite=data.identity,
        suite_name=data.suite.name,
        specialist=specialist,
        status=run.status if run else EvalRunStatus.COMPLETED,
        origin=origin,
        actor=actor,
        created_at=created_at,
        started_at=None if run is None else run.started_at,
        finished_at=None if run is None else run.finished_at,
        cases=counts,
        score=None if record is None else record.score,
        judge_score=None if summary is None else summary.judge_score,
        judge_score_state="unavailable" if summary is None else summary.judge_score_state,
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
    snapshot, summary, suite, definitions = (
        data.snapshot,
        data.summary,
        data.suite,
        data.definitions,
    )
    if summary is not None:
        presented = set(summary.case_ids)
        if data.fresh:
            if presented != {case.id for case in definitions}:
                raise HTTPException(status_code=409, detail="Eval result case membership mismatch.")
        else:
            definitions = tuple(case for case in definitions if case.id in presented)
        return EvalViewProgress(
            total_cases=len(definitions),
            total_trials=summary.total_trials,
            completed_cases=len(summary.case_ids),
            completed_trials=summary.completed_trials,
            state="published",
        )
    completed_trials = completed_cases = None
    state = "unavailable"
    if snapshot is not None and snapshot.run.status in _LIVE:
        slots = snapshot.completed_trials_by_case
        completed_trials = sum(slots.values())
        completed_cases = sum(value == suite.trial_request.trials for value in slots.values())
        state = "queued" if snapshot.run.status == EvalRunStatus.QUEUED else "checkpointed"
    return EvalViewProgress(
        total_cases=len(definitions),
        total_trials=len(definitions) * suite.trial_request.trials,
        completed_cases=completed_cases,
        completed_trials=completed_trials,
        state=state,
    )


def case_outcome_query(
    data: EvalRunViewData,
    comparison: EvalViewComparisonData,
    *,
    case_filter: EvalRunCaseFilter | None,
    after: str | None,
    limit: int,
) -> EvalCaseOutcomeQuery | None:
    """The outcome page a run view shows, or None when the run has no result yet."""
    if data.record is None or data.summary is None:
        return None
    case_ids = None
    store_filter = None
    if case_filter in {"regressed", "improved"}:
        selected = comparison.regressed if case_filter == "regressed" else comparison.improved
        case_ids = tuple(sorted(selected))
    elif case_filter == "failing":
        store_filter = "failing"
    elif case_filter == "flaky":
        store_filter = "flaky"
    return EvalCaseOutcomeQuery(
        revision=data.record.revision,
        case_ids=case_ids,
        after=after,
        limit=limit,
        case_filter=store_filter,
    )


def case_results(
    data: EvalRunViewData,
    outcomes: Sequence[EvalCaseOutcomeV1],
    compared: dict[str, EvalCaseOutcomeV1],
    comparison: EvalViewComparisonData,
) -> list[EvalCaseResultView]:
    """Case rows of one page of a result's outcomes, with the compared run's cases."""
    definitions = {case.id: case for case in data.definitions}
    compared_state = comparison.summary.state == "compared"
    rows = []
    for outcome in outcomes:
        case = outcome.presentation
        definition = definitions.get(case.case_id)
        if definition is None or definition.revision != case.case_revision:
            raise HTTPException(status_code=409, detail="Eval result case identity mismatch.")
        change = (
            "uncompared"
            if not compared_state
            else "tradeoff"
            if case.case_id in comparison.improved and case.case_id in comparison.regressed
            else "improved"
            if case.case_id in comparison.improved
            else "regressed"
            if case.case_id in comparison.regressed
            else "unchanged"
        )
        previous = compared.get(case.case_id) if compared_state else None
        rows.append(
            EvalCaseResultView(
                case=definition.view,
                presentation=case,
                judge_score=outcome.judge_score,
                judge_score_state=outcome.judge_score_state,
                evidence=outcome.evidence,
                compared=None if previous is None else previous.presentation,
                compared_evidence=None if previous is None else previous.evidence,
                structured_judgments=tuple(
                    item for item in comparison.structured_judgments if item.case_id == case.case_id
                ),
                tool_json_assertions=tuple(
                    item for item in comparison.tool_json_assertions if item.case_id == case.case_id
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
    return rows
