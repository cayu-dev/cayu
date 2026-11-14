"""What the Evals pages show about an immutable result, without the result itself.

A result document holds every trial's evidence and reaches 40 MB. The run and
suite pages show counts, scores and one page of cases. The functions here derive
those parts once, from a validated result and its canonical presentation:

- ``summarize_result`` gives the result's ``EvalResultViewSummaryV1`` and one
  ``EvalCaseOutcomeV1`` per case. Stores write them with the result record, and
  the reference reader derives the same values from a decoded document, so the
  two can be compared.
- ``EvalViewComparisonRecordV1`` holds the comparison of two results, which is a
  pure function of the pair, so it can be recorded once and read many times.

Each part is usable on its own: a store, the server reader and the ``cayu eval``
CLI all use the same models.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictFloat, StrictInt, StrictStr

from cayu._validation import compact_json_utf8_size
from cayu.evals.execution import CorpusExecutionResult
from cayu.evals.execution_comparison import (
    CorpusExecutionRegression,
    EvalStructuredJudgeComparisonV1,
    EvalToolJsonAssertionComparisonV1,
)
from cayu.evals.published import _published_score
from cayu.evals.read_views import (
    EvalViewCaseCounts,
    EvalViewComparison,
    EvalViewImprovement,
    EvalViewUsage,
)
from cayu.evals.result_presentation import (
    EvalCasePresentationV2,
    EvalResultPresentationV2,
    present_eval_result,
)
from cayu.evals.results import (
    CapturedEvaluationResultV1,
    EvalResultOrigin,
    EvalResultTargetIdentityV1,
    eval_result_projection,
)
from cayu.evals.store import EvalResultRecord, EvalRunStatus
from cayu.evals.trial_policy import EvalCaseReliabilityV1

EVAL_VIEW_SUMMARY_SCHEMA_VERSION = 1

JudgeScoreState = Literal["available", "not_used", "unavailable"]


class _SummaryModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvalResultViewSummaryV1(_SummaryModel):
    """Run-level view of one immutable result: counts, scores, usage and its cases.

    It repeats the result's catalog metadata (everything in its result record but
    the time it was stored), so a reader can tell that the record it holds
    describes the document this summary was derived from without decoding it.
    """

    schema_version: Literal[1] = EVAL_VIEW_SUMMARY_SCHEMA_VERSION
    revision: StrictStr
    origin: EvalResultOrigin
    target: EvalResultTargetIdentityV1
    corpus_revision: StrictStr
    suite_id: StrictStr
    suite_revision: StrictStr
    status: Literal["passed", "failed", "unavailable", "error"]
    score: StrictFloat | None
    document_bytes: StrictInt = Field(ge=1)
    case_ids: tuple[StrictStr, ...]
    total_trials: StrictInt = Field(ge=0)
    completed_trials: StrictInt = Field(ge=0)
    cases: EvalViewCaseCounts
    judge_score: StrictFloat | None
    judge_score_state: JudgeScoreState
    usage: EvalViewUsage

    def describes(self, record: EvalResultRecord) -> bool:
        """True when ``record`` is this result's catalog metadata."""
        return record.model_dump(mode="json", exclude={"created_at"}) == self.model_dump(
            mode="json", include=set(_RECORD_FIELDS)
        ) and type(record.score) is type(self.score)


_RECORD_FIELDS = frozenset(EvalResultRecord.model_fields) - {"created_at"}


class EvalCaseOutcomeV1(_SummaryModel):
    """One case of a result: its canonical presentation and, when loaded, its evidence."""

    schema_version: Literal[1] = EVAL_VIEW_SUMMARY_SCHEMA_VERSION
    presentation: EvalCasePresentationV2
    judge_score: StrictFloat | None
    judge_score_state: JudgeScoreState
    duration_ms: StrictInt | None = None
    evidence: JsonValue = None

    @property
    def case_id(self) -> str:
        return self.presentation.case_id

    @property
    def flaky(self) -> bool:
        reliability = self.presentation.reliability
        return 0 < reliability.passed_trials < reliability.total_trials

    @property
    def failing(self) -> bool:
        return self.presentation.status in {"failed", "error", "unavailable"}


class EvalViewComparisonRecordV1(_SummaryModel):
    """Comparison of a current result with a previous one, without the selector used."""

    schema_version: Literal[1] = EVAL_VIEW_SUMMARY_SCHEMA_VERSION
    summary: EvalViewComparison
    improved: tuple[StrictStr, ...] = ()
    regressed: tuple[StrictStr, ...] = ()
    regressions: tuple[CorpusExecutionRegression, ...] = ()
    improvements: tuple[EvalViewImprovement, ...] = ()
    structured_judgments: tuple[EvalStructuredJudgeComparisonV1, ...] = ()
    tool_json_assertions: tuple[EvalToolJsonAssertionComparisonV1, ...] = ()


class EvalCaseOutcomeQuery(_SummaryModel):
    """Keyset page of a result's case outcomes in case-ID order."""

    revision: StrictStr
    case_ids: tuple[StrictStr, ...] | None = None
    after: StrictStr | None = None
    limit: StrictInt | None = Field(default=None, ge=1, le=1000)
    case_filter: Literal["failing", "flaky"] | None = None
    include_evidence: bool = True


class EvalCaseOutcomePage(_SummaryModel):
    items: tuple[EvalCaseOutcomeV1, ...]
    total: StrictInt
    has_more: bool


class EvalRunViewPollV1(_SummaryModel):
    """What a live run view polls: the run's lifecycle and a version of its view.

    ``version`` changes whenever anything a run view shows about the run can
    change: its status, times, result, trial checkpoints or scenario progress.
    """

    run_id: StrictStr
    target_key: StrictStr
    corpus_revision: StrictStr
    suite_id: StrictStr
    status: EvalRunStatus
    completed_trials: StrictInt = Field(ge=0)
    result_revision: StrictStr | None
    version: StrictStr


def run_view_version(*parts: object) -> str:
    """Opaque version of the run row fields a run view depends on."""
    encoded = json.dumps(parts, default=str, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(encoded.encode()).hexdigest()


class EvalSuiteCatalogEntryV1(_SummaryModel):
    """A corpus suite as the catalog keeps it: its spec and its cases' specs as JSON."""

    target_key: StrictStr
    corpus_revision: StrictStr
    suite_id: StrictStr
    suite_revision: StrictStr
    name: StrictStr
    description: StrictStr | None
    case_count: StrictInt
    suite: JsonValue
    cases: tuple[JsonValue, ...]


@dataclass(frozen=True, slots=True)
class EvalResultViewRows:
    """A result's summary and case outcomes, with the JSON texts a store writes."""

    summary: EvalResultViewSummaryV1
    outcomes: tuple[EvalCaseOutcomeV1, ...]

    @property
    def summary_json(self) -> str:
        return self.summary.model_dump_json()

    def outcome_rows(self) -> list[tuple[object, ...]]:
        """(case_id, status, score, passed, total, duration_ms, outcome_json, evidence_json)."""
        rows = []
        for outcome in self.outcomes:
            presentation = outcome.presentation
            reliability = presentation.reliability
            rows.append(
                (
                    presentation.case_id,
                    str(presentation.status),
                    presentation.score,
                    reliability.passed_trials,
                    reliability.total_trials,
                    outcome.duration_ms,
                    outcome.model_dump_json(exclude={"evidence"}),
                    None
                    if outcome.evidence is None
                    else json.dumps(outcome.evidence, ensure_ascii=False, separators=(",", ":")),
                )
            )
        return rows


def judge_score(
    cases: Sequence[EvalCasePresentationV2],
) -> tuple[float | None, JudgeScoreState]:
    """Mean semantic-judge score over the cases' trials, as the run view shows it."""
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


def _usage(result: CorpusExecutionResult | CapturedEvaluationResultV1) -> EvalViewUsage:
    if type(result) is not CorpusExecutionResult:
        return EvalViewUsage(state="unavailable", tokens=None, observed_tokens=None)
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
    return EvalViewUsage(
        state="complete" if complete else "partial" if observed else "unavailable",
        tokens=tokens if complete else None,
        observed_tokens=tokens,
    )


def case_counts(
    reliabilities: Sequence[EvalCaseReliabilityV1], *, total: int
) -> EvalViewCaseCounts:
    """Case counts from the reliabilities of the cases that have finished."""
    statuses = Counter(item.outcome for item in reliabilities)
    return EvalViewCaseCounts(
        total=total,
        passed=statuses["passed"],
        failed=statuses["failed"],
        error=statuses["error"],
        unavailable=statuses["unavailable"],
        flaky=sum(0 < item.passed_trials < item.total_trials for item in reliabilities),
    )


def summarize_result(
    result: CorpusExecutionResult | CapturedEvaluationResultV1,
    presentation: EvalResultPresentationV2 | None = None,
    *,
    document_bytes: int | None = None,
) -> EvalResultViewRows:
    """Summary and case outcomes of a validated result, with each case's evidence.

    ``document_bytes`` is the size of the stored document; without it, the size of
    the result's compact JSON form.
    """
    if presentation is None:
        presentation = present_eval_result(result)
    if document_bytes is None:
        document_bytes = compact_json_utf8_size(result.model_dump(mode="json"))
    projection = eval_result_projection(result)
    if presentation.result_revision != result.revision:
        raise ValueError("Eval result presentation belongs to another result.")
    evidence = (
        {case.case_id: case for case in result.run.cases}
        if type(result) is CorpusExecutionResult
        else {}
    )
    score, state = judge_score(presentation.cases)
    summary = EvalResultViewSummaryV1(
        revision=result.revision,
        origin=projection.origin,
        target=projection.target,
        corpus_revision=projection.corpus_revision,
        suite_id=projection.suite_id,
        suite_revision=projection.suite_revision,
        status=projection.status,
        score=projection.score,
        document_bytes=document_bytes,
        case_ids=tuple(case.case_id for case in presentation.cases),
        total_trials=sum(case.reliability.total_trials for case in presentation.cases),
        completed_trials=sum(len(case.trials) for case in presentation.cases),
        cases=case_counts(
            [case.reliability for case in presentation.cases], total=len(presentation.cases)
        ),
        judge_score=score,
        judge_score_state=state,
        usage=_usage(result),
    )
    outcomes = []
    for case in presentation.cases:
        case_score, case_state = judge_score((case,))
        published = evidence.get(case.case_id)
        outcomes.append(
            EvalCaseOutcomeV1(
                presentation=case,
                judge_score=case_score,
                judge_score_state=case_state,
                duration_ms=None if published is None else published.duration_ms,
                evidence=None if published is None else published.model_dump(mode="json"),
            )
        )
    return EvalResultViewRows(summary=summary, outcomes=tuple(outcomes))


def page_case_outcomes(
    outcomes: Sequence[EvalCaseOutcomeV1], query: EvalCaseOutcomeQuery
) -> EvalCaseOutcomePage:
    """The reference keyset page that stores answer with SQL."""
    selected = sorted(outcomes, key=lambda item: item.case_id)
    if query.case_ids is not None:
        wanted = set(query.case_ids)
        selected = [item for item in selected if item.case_id in wanted]
    if query.case_filter == "failing":
        selected = [item for item in selected if item.failing]
    elif query.case_filter == "flaky":
        selected = [item for item in selected if item.flaky]
    total = len(selected)
    if query.after is not None:
        selected = [item for item in selected if item.case_id > query.after]
    more = query.limit is not None and len(selected) > query.limit
    if query.limit is not None:
        selected = selected[: query.limit]
    if not query.include_evidence:
        selected = [item.model_copy(update={"evidence": None}) for item in selected]
    return EvalCaseOutcomePage(items=tuple(selected), total=total, has_more=more)


__all__ = [
    "EVAL_VIEW_SUMMARY_SCHEMA_VERSION",
    "EvalCaseOutcomePage",
    "EvalCaseOutcomeQuery",
    "EvalCaseOutcomeV1",
    "EvalResultViewRows",
    "EvalResultViewSummaryV1",
    "EvalRunViewPollV1",
    "EvalSuiteCatalogEntryV1",
    "EvalViewComparisonRecordV1",
    "case_counts",
    "judge_score",
    "page_case_outcomes",
    "run_view_version",
    "summarize_result",
]
