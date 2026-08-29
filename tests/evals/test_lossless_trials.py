from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from cayu import (
    AgentSpec,
    CayuApp,
    EvalAssertion,
    EvalAssertionResult,
    EvalCase,
    EvalCaseResult,
    EvalOutcome,
    EvalRun,
    EvalStatus,
    EvalSuite,
    EvalTrialResult,
    Event,
    EventType,
    FinalOutputContains,
    InMemorySessionStore,
    MaxEstimatedCost,
    Message,
    ModelPrice,
    ModelStreamEvent,
    PriceBook,
    RunRequest,
    ScriptedModelProvider,
    SessionCostSummary,
    eval_run_to_json,
    render_html_report,
    run_eval_case,
    run_eval_suite,
)
from cayu.core.events import event_with_runtime_payload_authority
from cayu.evals import runner as runner_module
from cayu.runtime.sessions import (
    SessionIdentity,
    SessionStatus,
    TerminalSessionEvidenceError,
    TerminalSessionEvidenceErrorCode,
)
from cayu.runtime.usage import SessionUsageSummary


def _scripted_app(*batches: list[ModelStreamEvent]) -> CayuApp:
    app = CayuApp(enable_logging=False)
    app.register_provider(ScriptedModelProvider(batches), default=True)
    app.register_agent(AgentSpec(name="agent", model="fake-model"))
    return app


def _request() -> RunRequest:
    return RunRequest(
        agent_name="agent",
        messages=[Message.text("user", "respond")],
        max_steps=1,
    )


def _assertion(outcome: EvalOutcome) -> EvalAssertionResult:
    if outcome == EvalOutcome.PASSED:
        return EvalAssertionResult(
            name="check",
            outcome=outcome,
            score=1.0,
        )
    if outcome == EvalOutcome.FAILED:
        return EvalAssertionResult(
            name="check",
            outcome=outcome,
            score=0.0,
        )
    return EvalAssertionResult(name="check", outcome=outcome, message=outcome.value)


def _trial(number: int, status: EvalStatus) -> EvalTrialResult:
    now = datetime.now(UTC)
    outcome = EvalOutcome(status.value) if status != EvalStatus.SKIPPED else None
    session_id = f"session-{number}"
    evidence_complete = status != EvalStatus.UNAVAILABLE
    return EvalTrialResult(
        trial_number=number,
        status=status,
        session_id=session_id,
        score=(
            None
            if status in (EvalStatus.ERROR, EvalStatus.UNAVAILABLE)
            else 1.0
            if status == EvalStatus.PASSED
            else 0.0
        ),
        assertions=() if outcome is None else (_assertion(outcome),),
        error="execution failed" if status == EvalStatus.ERROR else None,
        unavailable_reason="evidence missing" if status == EvalStatus.UNAVAILABLE else None,
        evidence_complete=evidence_complete,
        usage_summary=(
            SessionUsageSummary(session_id=session_id).model_dump(mode="json")
            if evidence_complete
            else None
        ),
        started_at=now,
        completed_at=now,
    )


def test_multi_trial_run_retains_distinct_evidence_and_reports_it():
    app = _scripted_app(
        [
            ModelStreamEvent.text_delta("alpha output"),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
                }
            ),
        ],
        [
            ModelStreamEvent.text_delta("beta output"),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
                }
            ),
        ],
    )
    case = EvalCase(
        id="distinct",
        request=_request(),
        assertions=[FinalOutputContains("alpha")],
    )

    result = asyncio.run(run_eval_case(app, case, suite_id="suite", trials=2))

    assert result.status == EvalStatus.FAILED
    assert result.score == pytest.approx(0.5)
    assert [trial.status for trial in result.trials] == [
        EvalStatus.PASSED,
        EvalStatus.FAILED,
    ]
    assert [trial.final_output for trial in result.trials] == ["alpha output", "beta output"]
    assert [trial.usage_summary["usage"]["total_tokens"] for trial in result.trials] == [3, 7]
    assert [trial.assertions[0].outcome for trial in result.trials] == [
        EvalOutcome.PASSED,
        EvalOutcome.FAILED,
    ]
    session_ids = [trial.session_id for trial in result.trials]
    assert None not in session_ids
    assert len(set(session_ids)) == 2

    reproduced = EvalCaseResult.from_trials(
        case_id=result.case_id,
        trials=result.trials,
        started_at=result.started_at,
        completed_at=result.completed_at,
        metadata=result.metadata,
    )
    assert reproduced.status == result.status
    assert reproduced.score == result.score
    assert reproduced.assertions == result.assertions

    run = asyncio.run(
        run_eval_suite(
            _scripted_app(
                [
                    ModelStreamEvent.text_delta("alpha output"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
                [
                    ModelStreamEvent.text_delta("beta output"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ),
            EvalSuite(id="suite", cases=[case]),
            trials=2,
        )
    )
    document = json.loads(eval_run_to_json(run))
    assert [trial["final_output"] for trial in document["cases"][0]["trials"]] == [
        "alpha output",
        "beta output",
    ]
    report = render_html_report(run)
    assert "Trial 1" in report
    assert "Trial 2" in report
    assert "alpha output" in report
    assert "beta output" in report


def test_multi_trial_run_retains_each_snapshot_derived_cost():
    app = _scripted_app(
        [
            ModelStreamEvent.text_delta("first"),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
                }
            ),
        ],
        [
            ModelStreamEvent.text_delta("second"),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
                }
            ),
        ],
    )
    pricing = PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="scripted",
                model="fake-model",
                input_per_million=Decimal("1000000"),
                output_per_million=Decimal("1000000"),
            ),
        )
    )
    case = EvalCase(
        id="costs",
        request=_request(),
        assertions=[MaxEstimatedCost(Decimal("10"), pricing=pricing)],
    )

    result = asyncio.run(run_eval_case(app, case, suite_id="suite", trials=2))

    costs = [trial.assertions[0].cost_summary for trial in result.trials]
    assert all(cost is not None for cost in costs)
    assert [cost.total_cost for cost in costs if cost is not None] == [
        Decimal("3"),
        Decimal("7"),
    ]


def test_unpriced_cost_trial_is_unavailable_and_retains_coverage():
    app = _scripted_app(
        [
            ModelStreamEvent.text_delta("complete output"),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
                }
            ),
        ]
    )
    case = EvalCase(
        id="unpriced-cost",
        request=_request(),
        assertions=[
            MaxEstimatedCost(
                Decimal("10"),
                pricing=PriceBook(
                    prices=(
                        ModelPrice.fixed(
                            provider_name="other-provider",
                            model="other-model",
                            input_per_million=Decimal("1"),
                            output_per_million=Decimal("1"),
                        ),
                    )
                ),
            )
        ],
    )

    result = asyncio.run(run_eval_case(app, case, suite_id="suite", retain_trajectory=True))

    trial = result.trials[0]
    assert result.status is EvalStatus.UNAVAILABLE
    assert result.score is None
    assert trial.status is EvalStatus.UNAVAILABLE
    assert trial.assertions[0].outcome is EvalOutcome.UNAVAILABLE
    assert trial.assertions[0].cost_summary is not None
    assert trial.assertions[0].cost_summary.unpriced_model_steps == 1
    assert trial.evidence_complete is True
    assert trial.trajectory is not None
    assert trial.trajectory.children_incomplete is False


def test_trial_rejects_usage_and_cost_summaries_from_other_sessions():
    app = _scripted_app(
        [
            ModelStreamEvent.text_delta("complete output"),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
                }
            ),
        ]
    )
    pricing = PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="scripted",
                model="fake-model",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("1"),
            ),
        )
    )
    case = EvalCase(
        id="attributed-summaries",
        request=_request(),
        assertions=[MaxEstimatedCost(Decimal("10"), pricing=pricing)],
    )
    trial = asyncio.run(run_eval_case(app, case, suite_id="suite")).trials[0]

    forged_usage = trial.model_dump(mode="python")
    forged_usage["usage_summary"]["session_id"] = "another-session"
    with pytest.raises(ValidationError, match="usage_summary must belong"):
        EvalTrialResult.model_validate(forged_usage)

    garbage_usage = trial.model_dump(mode="python")
    garbage_usage["usage_summary"] = {
        "session_id": trial.session_id,
        "garbage": "not a usage summary",
    }
    with pytest.raises(ValidationError):
        EvalTrialResult.model_validate(garbage_usage)

    forged_cost = trial.model_dump(mode="python")
    forged_cost["assertions"][0]["cost_summary"]["session_id"] = "wrong-session"
    with pytest.raises(ValidationError, match="cost summaries must belong"):
        EvalTrialResult.model_validate(forged_cost)


@pytest.mark.parametrize("outcome", list(EvalOutcome))
def test_assertion_outcomes_enforce_nullable_score_contract(outcome: EvalOutcome):
    if outcome in (EvalOutcome.PASSED, EvalOutcome.FAILED):
        score = 1.0 if outcome == EvalOutcome.PASSED else 0.0
        result = EvalAssertionResult(name="check", outcome=outcome, score=score)
        assert result.score == score
        with pytest.raises(ValidationError, match="require a score"):
            EvalAssertionResult(name="check", outcome=outcome)
    else:
        result = EvalAssertionResult(name="check", outcome=outcome)
        assert result.score is None
        with pytest.raises(ValidationError, match="cannot have a score"):
            EvalAssertionResult(name="check", outcome=outcome, score=0.0)


def test_trial_rejects_unavailable_status_when_any_assertion_errored():
    now = datetime.now(UTC)

    with pytest.raises(ValidationError, match="status does not match"):
        EvalTrialResult(
            trial_number=1,
            status=EvalStatus.UNAVAILABLE,
            score=None,
            assertions=(
                _assertion(EvalOutcome.UNAVAILABLE),
                EvalAssertionResult(
                    name="grader",
                    outcome=EvalOutcome.ERROR,
                    message="grader crashed",
                ),
            ),
            unavailable_reason="some evidence was missing",
            evidence_complete=False,
            started_at=now,
            completed_at=now,
        )


def test_complete_trial_evidence_requires_a_concrete_session():
    now = datetime.now(UTC)

    with pytest.raises(ValidationError, match="concrete session_id"):
        EvalTrialResult(
            trial_number=1,
            status=EvalStatus.ERROR,
            score=None,
            error="assertion crashed",
            evidence_complete=True,
            started_at=now,
            completed_at=now,
        )


def test_complete_unavailable_trial_requires_an_unavailable_assertion():
    now = datetime.now(UTC)

    with pytest.raises(ValidationError, match="unavailable assertion outcome"):
        EvalTrialResult(
            trial_number=1,
            status=EvalStatus.UNAVAILABLE,
            session_id="session-1",
            score=None,
            unavailable_reason="pricing unavailable",
            evidence_complete=True,
            started_at=now,
            completed_at=now,
        )


def test_result_timing_must_enclose_retained_children():
    started_at = datetime.now(UTC)
    completed_at = started_at + timedelta(seconds=10)
    trial = EvalTrialResult(
        trial_number=1,
        status=EvalStatus.SKIPPED,
        session_id="session-1",
        score=0.0,
        evidence_complete=True,
        usage_summary=SessionUsageSummary(session_id="session-1").model_dump(mode="json"),
        started_at=started_at,
        completed_at=completed_at,
        duration_ms=10_000,
    )

    with pytest.raises(ValidationError, match="Case timing must enclose"):
        EvalCaseResult.from_trials(
            case_id="case",
            trials=(trial,),
            started_at=started_at + timedelta(seconds=2),
            completed_at=started_at + timedelta(seconds=3),
        )

    case = EvalCaseResult.from_trials(case_id="case", trials=(trial,))
    with pytest.raises(ValidationError, match="Run timing must enclose"):
        EvalRun(
            suite_id="suite",
            status=EvalStatus.SKIPPED,
            score=0.0,
            cases=(case,),
            started_at=started_at + timedelta(seconds=2),
            completed_at=started_at + timedelta(seconds=3),
            duration_ms=1_000,
        )


def test_eval_run_rejects_empty_fail_open_result_graph():
    with pytest.raises(ValidationError, match="at least 1"):
        EvalRun(
            suite_id="empty",
            status=EvalStatus.PASSED,
            score=1.0,
            cases=(),
        )


def test_case_aggregation_precedence_is_error_unavailable_failed_passed():
    passed = EvalCaseResult.from_trials(
        case_id="case",
        trials=(_trial(1, EvalStatus.PASSED),),
    )
    failed = EvalCaseResult.from_trials(
        case_id="case",
        trials=(_trial(1, EvalStatus.PASSED), _trial(2, EvalStatus.FAILED)),
    )
    unavailable = EvalCaseResult.from_trials(
        case_id="case",
        trials=(
            _trial(1, EvalStatus.PASSED),
            _trial(2, EvalStatus.FAILED),
            _trial(3, EvalStatus.UNAVAILABLE),
        ),
    )
    errored = EvalCaseResult.from_trials(
        case_id="case",
        trials=(
            _trial(1, EvalStatus.PASSED),
            _trial(2, EvalStatus.FAILED),
            _trial(3, EvalStatus.UNAVAILABLE),
            _trial(4, EvalStatus.ERROR),
        ),
    )

    assert passed.status == EvalStatus.PASSED
    assert failed.status == EvalStatus.FAILED
    assert unavailable.status == EvalStatus.UNAVAILABLE
    assert unavailable.score is None
    assert errored.status == EvalStatus.ERROR
    assert errored.score is None


def test_case_model_rejects_forged_aggregate_fields():
    result = EvalCaseResult.from_trials(
        case_id="case",
        trials=(_trial(1, EvalStatus.PASSED),),
    )
    forged = result.model_dump(mode="python")
    forged["score"] = 0.25

    with pytest.raises(ValidationError, match="score does not match"):
        EvalCaseResult.model_validate(forged)


class _OversizedEvidenceStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    async def load_terminal_session_evidence(self, session_id: str, *, limits=None):
        raise TerminalSessionEvidenceError(
            TerminalSessionEvidenceErrorCode.TOTAL_BYTES_EXCEEDED,
            limit=1024,
            observed=2048,
        )


class _UnsupportedEvidenceStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1
    supports_terminal_session_evidence = False


def test_child_trajectory_captures_runner_owned_interrupted_descendant():
    async def run() -> None:
        store = InMemorySessionStore()

        async def create_running(session_id: str, *, parent_session_id: str | None = None) -> str:
            interaction_id = f"{session_id}-interaction"
            message = Message.text("user", "capture interrupted child")
            await store.create(
                RunRequest(
                    agent_name="agent",
                    session_id=session_id,
                    parent_session_id=parent_session_id,
                    messages=[message],
                ),
                identity=SessionIdentity(provider_name="scripted", model="fake-model"),
                interaction_started_event=Event(
                    id=f"{session_id}-interaction-started",
                    type=EventType.INTERACTION_STARTED,
                    session_id=session_id,
                    interaction_id=interaction_id,
                ),
                interaction_source_messages=[message],
            )
            await store.replace_initial_transcript_messages(
                session_id,
                [message],
                [message],
                interaction_id=interaction_id,
            )
            origin = Event(
                id=f"{session_id}-started",
                type=EventType.SESSION_STARTED,
                session_id=session_id,
                payload=(
                    {} if parent_session_id is None else {"parent_session_id": parent_session_id}
                ),
            )
            if parent_session_id is not None:
                origin = event_with_runtime_payload_authority(origin, "parent_session_id")
            await store.append_event(session_id, origin)
            return interaction_id

        root_id = "runner-owned-root"
        child_id = "runner-owned-interrupted-child"
        await create_running(root_id)
        child_interaction_id = await create_running(child_id, parent_session_id=root_id)
        await store.publish_interaction_transition(
            child_id,
            event=Event(
                id=f"{child_id}-interaction-interrupted",
                type=EventType.INTERACTION_INTERRUPTED,
                session_id=child_id,
                interaction_id=child_interaction_id,
            ),
            from_statuses={SessionStatus.RUNNING},
            to_status=SessionStatus.INTERRUPTED,
        )
        await store.append_event(
            child_id,
            Event(
                id=f"{child_id}-session-interrupted",
                type=EventType.SESSION_INTERRUPTED,
                session_id=child_id,
            ),
        )

        incomplete = runner_module._IncompleteFlag()
        children = await runner_module._build_child_trajectories(
            CayuApp(session_store=store, enable_logging=False),
            root_id,
            visited={root_id},
            incomplete=incomplete,
        )

        assert incomplete.value is False
        assert len(children) == 1
        assert children[0].session is not None
        assert children[0].session.id == child_id
        assert children[0].session.status is SessionStatus.INTERRUPTED
        assert children[0].events[-1].type is EventType.SESSION_INTERRUPTED

    asyncio.run(run())


def test_store_without_terminal_evidence_capability_is_explicitly_unavailable():
    app = CayuApp(session_store=_UnsupportedEvidenceStore(), enable_logging=False)
    app.register_provider(
        ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("complete output"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="agent", model="fake-model"))
    case = EvalCase(
        id="unsupported-store",
        request=_request(),
        assertions=[FinalOutputContains("complete")],
    )

    result = asyncio.run(run_eval_case(app, case, suite_id="suite"))

    assert result.status is EvalStatus.UNAVAILABLE
    assert result.score is None
    assert "does not support exact terminal evidence reads" in result.unavailable_reason


def test_typed_terminal_evidence_failure_is_unavailable_not_failed_or_zero():
    app = CayuApp(session_store=_OversizedEvidenceStore(), enable_logging=False)
    provider = ScriptedModelProvider(
        [
            ModelStreamEvent.text_delta("complete output"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="agent", model="fake-model"))
    case = EvalCase(
        id="oversized",
        request=_request(),
        assertions=[FinalOutputContains("complete")],
    )

    result = asyncio.run(run_eval_case(app, case, suite_id="suite"))
    trial = result.trials[0]

    assert len(provider.requests) == 1
    assert result.status == EvalStatus.UNAVAILABLE
    assert result.score is None
    assert trial.status == EvalStatus.UNAVAILABLE
    assert trial.score is None
    assert trial.evidence_complete is False
    assert trial.assertions[0].outcome == EvalOutcome.UNAVAILABLE
    assert "total_bytes_exceeded" in trial.unavailable_reason


class _RaisingAssertion(EvalAssertion):
    async def evaluate(self, context):
        raise RuntimeError("grader unavailable")


class _WrongCostOwnerAssertion(EvalAssertion):
    async def evaluate(self, context):
        return self.passed(
            cost_summary=SessionCostSummary(
                session_id="another-session",
                currency="USD",
                model_steps=0,
                priced_model_steps=0,
                unpriced_model_steps=0,
                total_cost=Decimal("0"),
            )
        )


def test_foreign_assertion_cost_summary_becomes_a_lossless_trial_error():
    app = _scripted_app(
        [
            ModelStreamEvent.text_delta("complete output"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    case = EvalCase(
        id="foreign-assertion-cost",
        request=_request(),
        assertions=[_WrongCostOwnerAssertion()],
    )

    result = asyncio.run(run_eval_case(app, case, suite_id="suite"))
    trial = result.trials[0]

    assert result.status is EvalStatus.ERROR
    assert result.score is None
    assert trial.status is EvalStatus.ERROR
    assert trial.score is None
    assert trial.assertions[0].outcome is EvalOutcome.ERROR
    assert trial.assertions[0].cost_summary is None
    assert "trajectory session" in trial.error


class _TrialVaryingAssertion(EvalAssertion):
    def __init__(self, field: str) -> None:
        self.field = field
        self.calls = 0

    async def evaluate(self, context):
        self.calls += 1
        return EvalAssertionResult(
            name="dynamic" if self.field == "threshold" else f"dynamic-{self.calls}",
            outcome=EvalOutcome.PASSED,
            score=1.0,
            threshold=0.4 + self.calls / 10 if self.field == "threshold" else 0.5,
        )


@pytest.mark.parametrize(
    ("field", "diagnostic"),
    [
        ("threshold", "same assertion threshold"),
        ("name", "same ordered assertion contract"),
    ],
)
def test_trial_inconsistent_assertion_contract_is_a_lossless_case_error(
    field: str,
    diagnostic: str,
):
    app = _scripted_app(
        [
            ModelStreamEvent.text_delta("complete output"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ],
        [
            ModelStreamEvent.text_delta("complete output"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ],
    )
    case = EvalCase(
        id=f"varying-{field}",
        request=_request(),
        assertions=[_TrialVaryingAssertion(field)],
    )

    result = asyncio.run(run_eval_case(app, case, suite_id="suite", trials=2))

    assert result.status is EvalStatus.ERROR
    assert result.score is None
    assert result.assertions == ()
    assert diagnostic in result.error
    assert [trial.status for trial in result.trials] == [
        EvalStatus.PASSED,
        EvalStatus.PASSED,
    ]
    assert [trial.score for trial in result.trials] == [1.0, 1.0]
    assert EvalCaseResult.model_validate_json(result.model_dump_json()) == result

    forged = result.model_dump(mode="python")
    forged["error"] = "different aggregation diagnostic"
    with pytest.raises(ValidationError, match="error does not match"):
        EvalCaseResult.model_validate(forged)


def test_trial_aggregation_error_does_not_cancel_concurrent_sibling_case():
    batch = [
        ModelStreamEvent.text_delta("complete output"),
        ModelStreamEvent.completed({"finish_reason": "stop"}),
    ]
    app = _scripted_app(batch, batch, batch, batch)
    suite = EvalSuite(
        id="concurrent-aggregation",
        cases=[
            EvalCase(
                id="dynamic",
                request=_request(),
                assertions=[_TrialVaryingAssertion("threshold")],
            ),
            EvalCase(
                id="stable",
                request=_request(),
                assertions=[FinalOutputContains("complete")],
            ),
        ],
    )

    result = asyncio.run(run_eval_suite(app, suite, max_concurrency=2, trials=2))

    assert result.status is EvalStatus.ERROR
    assert [case.case_id for case in result.cases] == ["dynamic", "stable"]
    assert result.cases[0].status is EvalStatus.ERROR
    assert result.cases[1].status is EvalStatus.PASSED
    assert len(result.cases[1].trials) == 2
    assert EvalRun.model_validate_json(eval_run_to_json(result)) == result


def test_assertion_exception_is_error_with_no_score():
    app = _scripted_app(
        [
            ModelStreamEvent.text_delta("complete output"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    case = EvalCase(id="grader-error", request=_request(), assertions=[_RaisingAssertion()])

    result = asyncio.run(run_eval_case(app, case, suite_id="suite"))
    trial = result.trials[0]

    assert result.status == EvalStatus.ERROR
    assert result.score is None
    assert trial.status == EvalStatus.ERROR
    assert trial.score is None
    assert trial.evidence_complete is True
    assert trial.assertions[0].outcome == EvalOutcome.ERROR
    assert "grader unavailable" in trial.error


class _RaisingProbeAssertion(EvalAssertion):
    def required_probes(self):
        raise RuntimeError("probe planning failed")

    async def evaluate(self, context):
        return self.passed()


def test_assertion_probe_planning_exception_is_error_instead_of_escaping():
    app = _scripted_app(
        [
            ModelStreamEvent.text_delta("complete output"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    case = EvalCase(
        id="probe-error",
        request=_request(),
        assertions=[_RaisingProbeAssertion()],
    )

    result = asyncio.run(run_eval_case(app, case, suite_id="suite"))
    trial = result.trials[0]

    assert result.status == EvalStatus.ERROR
    assert result.score is None
    assert trial.status == EvalStatus.ERROR
    assert trial.score is None
    assert trial.evidence_complete is False
    assert trial.assertions[0].outcome == EvalOutcome.ERROR
    assert "Failed to prepare eval assertion evidence" in trial.error
    assert "probe planning failed" in trial.error
