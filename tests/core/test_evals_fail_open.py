from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from cayu import (
    AgentSpec,
    BudgetLimit,
    BudgetPolicy,
    CayuApp,
    EvalAssertion,
    EvalCase,
    EvalCaseResult,
    EvalContext,
    EvalRun,
    EvalStatus,
    EvalSuite,
    Event,
    EventType,
    InMemorySessionStore,
    Message,
    ModelPricing,
    PricingCatalog,
    RunRequest,
    ScriptedModelProvider,
    SessionIdentity,
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


def _causal_budget_limit(key: str) -> BudgetLimit:
    return BudgetLimit(
        scope="causal",
        key=key,
        max_estimated_cost=Decimal("100"),
        pricing=PricingCatalog(
            prices=(
                ModelPricing(
                    provider_name="fake",
                    model="fake-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("1"),
                ),
            )
        ),
    )


def test_eval_case_rejects_task_linked_request():
    with pytest.raises(ValueError, match="task_id"):
        EvalCase(
            id="task-linked",
            request=RunRequest(
                agent_name="agent",
                messages=[Message.text("user", "go")],
                task_id="task-1",
                task_worker_id="worker-1",
            ),
        )


def test_preflight_error_does_not_report_nonexistent_trial_session():
    case = EvalCase(
        id="missing-agent",
        request=RunRequest(
            agent_name="missing",
            messages=[Message.text("user", "go")],
        ),
    )

    result = asyncio.run(run_eval_case(_app(), case, suite_id="s"))

    assert result.status == EvalStatus.ERROR
    assert result.session_id is None
    assert result.trial_session_ids == ()


def test_eval_trial_anchors_to_first_emitted_session(monkeypatch):
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)

    async def forwarding_run(request):
        identity = SessionIdentity(provider_name="fake", model="fake-model")
        root = await store.create(request, identity=identity)
        child = await store.create(
            RunRequest(
                agent_name="child",
                session_id="forwarded-child",
                parent_session_id=root.id,
                messages=[Message.text("user", "child work")],
            ),
            identity=identity,
        )
        root_event = Event(type=EventType.SESSION_STARTED, session_id=root.id)
        child_event = Event(type=EventType.SESSION_STARTED, session_id=child.id)
        await store.append_event(root.id, root_event)
        await store.append_event(child.id, child_event)
        yield root_event
        yield child_event

    monkeypatch.setattr(app, "run", forwarding_run)

    result = asyncio.run(
        run_eval_case(
            app,
            EvalCase(id="forwarded-events", request=_request()),
            suite_id="s",
        )
    )

    assert result.session_id == result.trial_session_ids[0]
    assert result.session_id != "forwarded-child"


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
    assert len(result.trial_session_ids) == 4
    assert len(set(result.trial_session_ids)) == 4
    assert result.session_id == result.trial_session_ids[-1]


def test_trials_keep_last_concrete_session_when_final_trial_has_none():
    from cayu.evals.runner import _aggregate_trials

    now = datetime.now(UTC)
    concrete = EvalCaseResult(
        case_id="partial",
        status=EvalStatus.ERROR,
        trial_session_ids=("concrete-session",),
        session_id="concrete-session",
        error="failed after session creation",
        started_at=now,
        completed_at=now,
    )
    no_session = EvalCaseResult(
        case_id="partial",
        status=EvalStatus.ERROR,
        error="failed before session creation",
        started_at=now,
        completed_at=now,
    )

    result = _aggregate_trials(
        EvalCase(id="partial", request=_request()),
        [concrete, no_session],
        started_at=now,
        completed_at=now,
        retain_trajectory=False,
    )

    assert result.trial_session_ids == ("concrete-session",)
    assert result.session_id == "concrete-session"


@pytest.mark.parametrize(
    ("authored_causal_budget_id", "budget_key"),
    [
        ("authored-budget", "authored-budget"),
        (None, "authored-session"),
    ],
)
def test_trials_replace_authored_session_but_preserve_causal_identity_and_isolate_state(
    authored_causal_budget_id,
    budget_key,
):
    app = _app()
    case = EvalCase(
        id="isolated",
        request=RunRequest(
            agent_name="agent",
            session_id="authored-session",
            causal_budget_id=authored_causal_budget_id,
            messages=[Message.text("user", "fresh trial")],
            max_steps=1,
            budget_limits=(_causal_budget_limit(budget_key),),
        ),
        assertions=[_SequenceScoreAssertion([1.0, 1.0])],
    )

    async def scenario():
        async for _ in app.run(
            RunRequest(
                agent_name="agent",
                session_id="authored-session",
                messages=[Message.text("user", "poisoned prior state")],
                max_steps=1,
            )
        ):
            pass
        result = await run_eval_case(app, case, suite_id="s", trials=2)
        transcripts = [
            await app.session_store.load_transcript(session_id)
            for session_id in result.trial_session_ids
        ]
        sessions = [
            await app.session_store.load(session_id) for session_id in result.trial_session_ids
        ]
        return result, transcripts, sessions

    result, transcripts, sessions = asyncio.run(scenario())

    assert case.request.session_id == "authored-session"
    assert case.request.causal_budget_id == authored_causal_budget_id
    assert case.request.budget_limits[0].key == budget_key
    assert result.status == EvalStatus.PASSED
    assert result.authored_session_id == "authored-session"
    assert len(result.trial_session_ids) == 2
    assert len(set(result.trial_session_ids)) == 2
    assert "authored-session" not in result.trial_session_ids
    assert all(session is not None for session in sessions)
    effective_causal_budget_id = authored_causal_budget_id or "authored-session"
    assert [session.causal_budget_id for session in sessions if session is not None] == [
        effective_causal_budget_id,
        effective_causal_budget_id,
    ]
    assert all(
        "poisoned prior state"
        not in " ".join(
            part.text for message in transcript for part in message.content if hasattr(part, "text")
        )
        for transcript in transcripts
    )


def test_trial_does_not_rewrite_mismatched_causal_budget_limit():
    app = _app()
    case = EvalCase(
        id="mismatched-budget",
        request=RunRequest(
            agent_name="agent",
            session_id="authored-session",
            causal_budget_id="authored-budget",
            messages=[Message.text("user", "go")],
            budget_limits=(_causal_budget_limit("different-budget"),),
        ),
        assertions=[_SequenceScoreAssertion([1.0])],
    )

    result = asyncio.run(run_eval_case(app, case, suite_id="s"))

    provider = app.get_provider()
    assert isinstance(provider, ScriptedModelProvider)
    assert provider.requests == []
    assert case.request.budget_limits[0].key == "different-budget"
    assert result.status == EvalStatus.ERROR
    assert result.error is not None
    assert "does not match" in result.error


def test_trials_without_authored_identity_default_causal_id_to_concrete_session():
    app = _app()
    case = EvalCase(
        id="anonymous-identity",
        request=_request(),
        assertions=[_SequenceScoreAssertion([1.0, 1.0])],
    )

    async def scenario():
        result = await run_eval_case(app, case, suite_id="s", trials=2)
        sessions = [
            await app.session_store.load(session_id) for session_id in result.trial_session_ids
        ]
        return result, sessions

    result, sessions = asyncio.run(scenario())

    assert result.status == EvalStatus.PASSED
    assert all(session is not None for session in sessions)
    assert [session.causal_budget_id for session in sessions if session is not None] == list(
        result.trial_session_ids
    )


def test_eval_trial_preserves_causal_identity_for_app_budget_policy():
    app = CayuApp(
        budget_policy=BudgetPolicy(limits=(_causal_budget_limit("authored-budget"),)),
        enable_logging=False,
    )
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
        id="app-causal-policy",
        request=RunRequest(
            agent_name="agent",
            session_id="authored-session",
            causal_budget_id="authored-budget",
            messages=[Message.text("user", "go")],
        ),
        assertions=[_SequenceScoreAssertion([1.0])],
    )

    async def scenario():
        result = await run_eval_case(app, case, suite_id="s")
        assert result.session_id is not None
        session = await app.session_store.load(result.session_id)
        events = await app.session_store.load_events(result.session_id)
        return result, session, events

    result, session, events = asyncio.run(scenario())

    assert result.status == EvalStatus.PASSED
    assert result.session_id != "authored-session"
    assert session is not None
    assert session.causal_budget_id == "authored-budget"
    assert EventType.BUDGET_CHECKED in [event.type for event in events]


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


def test_trials_single_run_skips_aggregation_metadata():
    case = EvalCase(
        id="one",
        request=_request(),
        assertions=[_SequenceScoreAssertion([0.7], threshold=0.5)],
    )
    result = asyncio.run(run_eval_case(_app(), case, suite_id="s"))
    # trials=1 keeps the assertion result verbatim and does not add aggregation metadata.
    assert result.assertions[0].message == "graded"
    assert "trials" not in result.metadata
    assert result.authored_session_id is None
    assert result.trial_session_ids == (result.session_id,)


def test_trials_rejects_invalid_values():
    case = EvalCase(id="x", request=_request(), assertions=[_SequenceScoreAssertion([1.0])])
    with pytest.raises(ValueError, match="trials"):
        asyncio.run(run_eval_case(_app(), case, suite_id="s", trials=0))
    with pytest.raises(TypeError, match="trials"):
        asyncio.run(run_eval_case(_app(), case, suite_id="s", trials=True))
