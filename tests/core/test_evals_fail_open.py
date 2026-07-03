from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    EvalAssertion,
    EvalCase,
    EvalCaseResult,
    EvalContext,
    EvalRun,
    EvalStatus,
    EvalSuite,
    Message,
    RunRequest,
    ScriptedModelProvider,
    compare_eval_runs,
    run_eval_suite,
)
from cayu.evals import EvalAssertionResult, run_eval_case
from cayu.providers import ModelStreamEvent


def _app() -> CayuApp:
    app = CayuApp(enable_logging=False)
    app.register_provider(
        ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.text_delta("ok"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
            ]
            * 8
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="agent", model="fake-model"))
    return app


def _request() -> RunRequest:
    return RunRequest(agent_name="agent", messages=[Message.text("user", "go")], max_steps=1)


class _SequenceScoreAssertion(EvalAssertion):
    """Emits a preset score per evaluation, so a multi-trial run can be made to wobble."""

    def __init__(self, scores: list[float], *, threshold: float = 0.5) -> None:
        self._scores = list(scores)
        self._threshold = threshold
        self._index = 0

    @property
    def name(self) -> str:
        return "SequenceScore"

    async def evaluate(self, context: EvalContext) -> EvalAssertionResult:
        score = self._scores[self._index]
        self._index += 1
        return self.score_result(score, threshold=self._threshold, message="graded")


def test_zero_assertion_case_is_skipped_not_passed():
    # A case with no assertions used to pass at score 1.0 (fail-open); it must now be SKIPPED.
    suite = EvalSuite(
        id="skip",
        cases=[EvalCase(id="no-checks", request=_request())],
    )
    result = asyncio.run(run_eval_suite(_app(), suite))
    assert result.cases[0].status == EvalStatus.SKIPPED
    assert result.cases[0].score == 0.0
    # The run is not reported green while a case asserted nothing.
    assert result.status == EvalStatus.SKIPPED
    assert result.score == 0.0


def _case_result(case_id: str, status: EvalStatus, score: float) -> EvalCaseResult:
    now = datetime.now(UTC)
    return EvalCaseResult(
        case_id=case_id, status=status, score=score, started_at=now, completed_at=now
    )


def _run(status: EvalStatus, score: float, cases: list[EvalCaseResult]) -> EvalRun:
    return EvalRun(suite_id="s", status=status, score=score, cases=tuple(cases))


def test_score_tolerance_absorbs_stochastic_wobble():
    base = _run(EvalStatus.PASSED, 0.83, [_case_result("a", EvalStatus.PASSED, 0.83)])
    cur = _run(EvalStatus.PASSED, 0.82, [_case_result("a", EvalStatus.PASSED, 0.82)])

    # Zero tolerance flags the 0.01 wobble as a regression at both case and run level.
    strict = compare_eval_runs(base, cur)
    assert any("score regressed" in item for item in strict.regressions)

    # A small tolerance absorbs it: no regressions.
    tolerant = compare_eval_runs(base, cur, score_tolerance=0.05)
    assert tolerant.regressions == ()
    assert tolerant.cases[0].regressions == ()


def test_score_tolerance_still_flags_a_real_drop():
    base = _run(EvalStatus.PASSED, 0.90, [_case_result("a", EvalStatus.PASSED, 0.90)])
    cur = _run(EvalStatus.PASSED, 0.60, [_case_result("a", EvalStatus.PASSED, 0.60)])
    comparison = compare_eval_runs(base, cur, score_tolerance=0.05)
    assert any("score regressed" in item for item in comparison.regressions)


def test_score_tolerance_rejects_invalid_values():
    base = _run(EvalStatus.PASSED, 1.0, [_case_result("a", EvalStatus.PASSED, 1.0)])
    with pytest.raises(ValueError, match="score_tolerance"):
        compare_eval_runs(base, base, score_tolerance=-0.1)
    with pytest.raises(TypeError, match="score_tolerance"):
        compare_eval_runs(base, base, score_tolerance=True)


def test_trials_average_the_per_assertion_score():
    case = EvalCase(
        id="wobble",
        request=_request(),
        assertions=[_SequenceScoreAssertion([1.0, 0.8, 1.0, 0.8], threshold=0.5)],
    )
    result = asyncio.run(run_eval_case(_app(), case, suite_id="s", trials=4))
    assert result.status == EvalStatus.PASSED
    # mean of the four trial scores.
    assert result.assertions[0].score == pytest.approx(0.9)
    assert result.score == pytest.approx(0.9)
    assert result.metadata["trials"] == 4
    assert result.metadata["trial_scores"] == pytest.approx([1.0, 0.8, 1.0, 0.8])


def test_trials_mean_below_threshold_fails():
    case = EvalCase(
        id="mostly-fails",
        request=_request(),
        assertions=[_SequenceScoreAssertion([0.2, 0.4], threshold=0.5)],
    )
    result = asyncio.run(run_eval_case(_app(), case, suite_id="s", trials=2))
    # mean 0.3 < threshold 0.5 -> the aggregated assertion fails.
    assert result.status == EvalStatus.FAILED
    assert result.assertions[0].score == pytest.approx(0.3)


def test_trials_preserve_partial_execution_errors():
    app = CayuApp(enable_logging=False)
    app.register_provider(
        ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("ok"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="agent", model="fake-model"))
    case = EvalCase(
        id="partial-error",
        request=_request(),
        assertions=[_SequenceScoreAssertion([1.0], threshold=0.5)],
    )

    result = asyncio.run(run_eval_case(app, case, suite_id="s", trials=2))

    assert result.status == EvalStatus.ERROR
    assert result.score == 0.0
    assert result.error is not None
    assert "1 of 2 trials errored" in result.error
    assert result.metadata["trial_statuses"] == ["passed", "error"]


def test_trials_single_run_is_unchanged():
    case = EvalCase(
        id="one",
        request=_request(),
        assertions=[_SequenceScoreAssertion([0.7], threshold=0.5)],
    )
    result = asyncio.run(run_eval_case(_app(), case, suite_id="s"))
    # trials=1 returns the single run verbatim: no trial metadata, original message.
    assert result.assertions[0].message == "graded"
    assert "trials" not in result.metadata


def test_trials_rejects_invalid_values():
    case = EvalCase(id="x", request=_request(), assertions=[_SequenceScoreAssertion([1.0])])
    with pytest.raises(ValueError, match="trials"):
        asyncio.run(run_eval_case(_app(), case, suite_id="s", trials=0))
    with pytest.raises(TypeError, match="trials"):
        asyncio.run(run_eval_case(_app(), case, suite_id="s", trials=True))
