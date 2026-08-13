from __future__ import annotations

import asyncio
import inspect
from decimal import Decimal

import pytest
from tests._session_provenance import fixture_session_invocation

from cayu import AgentSpec, ModelProvider, ModelStreamEvent, RunRequest, ScriptedModelProvider
from cayu.core.events import Event, EventType
from cayu.core.messages import Message, ToolCallPart
from cayu.evals.assertions import (
    ChildSessionCompleted,
    FinalOutputContains,
    MaxEstimatedCost,
    MaxModelSteps,
    MaxToolCalls,
    MaxTotalTokens,
    ToolCalled,
    ToolNotCalled,
    ToolsCalledInOrder,
)
from cayu.evals.corpus import (
    EVIDENCE_MAX_CHILD_SESSIONS,
    EVIDENCE_MAX_FINAL_OUTPUT_CHARS,
    EVIDENCE_MAX_MODEL_STEPS,
    EVIDENCE_MAX_TOOL_CALLS,
    EVIDENCE_MAX_TOTAL_TOKENS,
    ChildStatusAssertionSpec,
    EvaluationEvidencePolicySpec,
    FinalOutputContainsAssertionSpec,
    FinalOutputEqualsAssertionSpec,
    MaxEstimatedCostAssertionSpec,
    MaxModelStepsAssertionSpec,
    MaxToolCallsAssertionSpec,
    MaxTotalTokensAssertionSpec,
    RootStatusAssertionSpec,
    ToolCalledAssertionSpec,
    ToolsCalledInOrderAssertionSpec,
    UsageRecordedAssertionSpec,
    assertion_spec_revision,
)
from cayu.evals.evidence import project_assertion_evidence_view
from cayu.evals.models import EvalContext, EvalOutcome, Trajectory
from cayu.evals.portable_assertions import compile_assertion_spec
from cayu.evals.portable_evaluation import evaluate_assertion_spec, evaluate_assertion_specs
from cayu.evals.runner import (
    EvalCase,
    EvalPlan,
    _evaluate_assertions,
    evaluate_assertions,
    run_eval_case,
    run_eval_suite,
)
from cayu.runtime.app import CayuApp
from cayu.runtime.costs import ModelPrice, PriceBook
from cayu.runtime.sessions import Session, SessionStatus
from cayu.runtime.usage import SessionUsageSummary, UsageMetrics, session_usage_summary
from cayu.vaults import REDACTED_SECRET, SecretRedactor


def _pricing() -> PriceBook:
    return PriceBook(
        price_book_version="v1",
        generated_at="2026-08-05T00:00:00Z",
        prices=(
            ModelPrice.fixed(
                provider_name="fixture",
                model="fixture-model",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("2"),
            ),
        ),
    )


def _terminal(session_id: str, status: SessionStatus) -> Event:
    return Event(
        type=(
            EventType.SESSION_COMPLETED
            if status is SessionStatus.COMPLETED
            else EventType.SESSION_FAILED
        ),
        session_id=session_id,
    )


def _trajectory(*, children_incomplete: bool = False) -> Trajectory:
    root_id = "root"
    events = (
        Event(
            type=EventType.MODEL_COMPLETED,
            session_id=root_id,
            payload={
                "usage_metrics": {
                    "provider_name": "fixture",
                    "model": "fixture-model",
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                }
            },
        ),
        Event(
            type=EventType.TOOL_CALL_STARTED,
            session_id=root_id,
            tool_name="lookup",
            payload={"tool_call_id": "call"},
        ),
        _terminal(root_id, SessionStatus.COMPLETED),
    )
    child_id = "child"
    child_events = (_terminal(child_id, SessionStatus.FAILED),)
    child = Trajectory(
        session=Session(
            id=child_id,
            agent_name="child",
            provider_name="fixture",
            model="fixture-model",
            causal_budget_id="budget",
            parent_session_id=root_id,
            invocation=fixture_session_invocation(
                child_id,
                parent_session_id=root_id,
            ),
            status=SessionStatus.FAILED,
        ),
        events=child_events,
        usage_summary=session_usage_summary(child_id, list(child_events)),
    )
    return Trajectory(
        session=Session(
            id=root_id,
            agent_name="agent",
            provider_name="fixture",
            model="fixture-model",
            causal_budget_id="budget",
            invocation=fixture_session_invocation(root_id),
            status=SessionStatus.COMPLETED,
        ),
        events=events,
        transcript=(
            Message.tool_call(
                calls=[
                    ToolCallPart(
                        tool_call_id="call",
                        tool_name="lookup",
                    )
                ]
            ),
            Message.text("assistant", "Approved"),
        ),
        usage_summary=session_usage_summary(root_id, list(events)),
        final_output="Approved",
        children=(child,),
        children_incomplete=children_incomplete,
    )


def _passing_specs():
    return (
        RootStatusAssertionSpec(id="root", expected="completed"),
        ChildStatusAssertionSpec(id="child", expected="failed", min_count=1, max_count=1),
        FinalOutputEqualsAssertionSpec(id="equals", expected="Approved"),
        FinalOutputContainsAssertionSpec(id="contains", expected="prove"),
        ToolCalledAssertionSpec(id="called", tool_name="lookup", min_count=1, max_count=1),
        ToolsCalledInOrderAssertionSpec(id="order", tool_names=("lookup",)),
        MaxToolCallsAssertionSpec(id="tools", maximum=1),
        MaxModelStepsAssertionSpec(id="steps", maximum=1),
        UsageRecordedAssertionSpec(id="usage", min_total_tokens=1),
        MaxTotalTokensAssertionSpec(id="tokens", maximum=15),
        MaxEstimatedCostAssertionSpec(id="cost", maximum="1", currency="USD"),
    )


def test_single_pure_assertion_result_carries_its_definition_revision():
    app = CayuApp(enable_logging=False)
    policy = EvaluationEvidencePolicySpec.standard()
    spec = RootStatusAssertionSpec(id="root", expected="completed")
    evidence = project_assertion_evidence_view(
        app,
        _trajectory(),
        evidence_policy=policy,
    )

    result = evaluate_assertion_spec(spec, evidence)

    assert result.outcome is EvalOutcome.PASSED
    assert result.assertion_revision == assertion_spec_revision(spec)


def test_every_portable_assertion_uses_one_shared_evidence_outcome_when_compiled(monkeypatch):
    import cayu.evals.portable_assertions as portable_assertions_module

    app = CayuApp(enable_logging=False)
    trajectory = _trajectory()
    pricing = _pricing()
    specs = _passing_specs()
    policy = EvaluationEvidencePolicySpec.standard()
    evidence = project_assertion_evidence_view(
        app,
        trajectory,
        evidence_policy=policy,
        pricing=pricing,
        cost_currencies=("USD",),
    )
    projection_calls = 0
    build_evidence = portable_assertions_module._build_assertion_evidence_view

    def counted_build(*args, **kwargs):
        nonlocal projection_calls
        projection_calls += 1
        return build_evidence(*args, **kwargs)

    monkeypatch.setattr(
        portable_assertions_module,
        "_build_assertion_evidence_view",
        counted_build,
    )

    pure = evaluate_assertion_specs(specs, evidence)
    compiled = asyncio.run(
        evaluate_assertions(
            trajectory,
            [
                compile_assertion_spec(
                    spec,
                    app=app,
                    evidence_policy=policy,
                    trusted_pricing=pricing,
                )
                for spec in specs
            ],
        )
    )

    assert [result.name for result in pure] == [spec.id for spec in specs]
    assert [result.outcome for result in pure] == [EvalOutcome.PASSED] * len(specs)
    assert [result.outcome for result in compiled] == [result.outcome for result in pure]
    expected_revisions = [assertion_spec_revision(spec) for spec in specs]
    assert [result.assertion_revision for result in pure] == expected_revisions
    assert [result.assertion_revision for result in compiled] == expected_revisions
    assert projection_calls == 1


def test_compiled_adapter_honors_explicitly_complete_synthetic_context():
    app = CayuApp(enable_logging=False)
    policy = EvaluationEvidencePolicySpec.standard()
    context = EvalContext(
        trajectory=Trajectory(),
        suite_id="suite",
        case_id="case",
        metadata={},
    )
    pairs = (
        (
            ChildSessionCompleted(min_count=0),
            ChildStatusAssertionSpec(id="children", expected="completed", min_count=0),
        ),
        (
            ToolCalled("missing", min_count=0, max_count=0),
            ToolCalledAssertionSpec(
                id="called",
                tool_name="missing",
                min_count=0,
                max_count=0,
            ),
        ),
        (
            ToolsCalledInOrder(()),
            ToolsCalledInOrderAssertionSpec(id="order", tool_names=()),
        ),
        (MaxToolCalls(0), MaxToolCallsAssertionSpec(id="tools", maximum=0)),
        (MaxModelSteps(0), MaxModelStepsAssertionSpec(id="steps", maximum=0)),
    )
    compiled = tuple(
        compile_assertion_spec(
            spec,
            app=app,
            evidence_policy=policy,
            trusted_pricing=None,
        )
        for _, spec in pairs
    )

    async def evaluate_all():
        direct = tuple(
            await asyncio.gather(*(assertion.evaluate(context) for assertion, _ in pairs))
        )
        adapter = tuple(
            await asyncio.gather(*(assertion.evaluate(context) for assertion in compiled))
        )
        shared = await _evaluate_assertions(compiled, context)
        return direct, adapter, shared

    direct, adapter, shared = asyncio.run(evaluate_all())

    assert [result.outcome for result in direct] == [EvalOutcome.PASSED] * len(pairs)
    assert [result.outcome for result in adapter] == [result.outcome for result in direct]
    assert [result.outcome for result in shared] == [result.outcome for result in direct]


def test_compiled_assertions_share_captured_redaction_for_output_and_tool_names():
    secret = "secret-token"
    redacted_tool = f"lookup-{REDACTED_SECRET}"
    app = CayuApp(
        enable_logging=False,
        secret_redactor=SecretRedactor(secret),
    )
    trajectory = _trajectory().model_copy(
        update={
            "events": (
                Event(
                    type=EventType.MODEL_COMPLETED,
                    session_id="root",
                    payload={
                        "usage_metrics": {
                            "provider_name": "fixture",
                            "model": "fixture-model",
                            "input_tokens": 10,
                            "output_tokens": 5,
                            "total_tokens": 15,
                        }
                    },
                ),
                Event(
                    type=EventType.TOOL_CALL_STARTED,
                    session_id="root",
                    tool_name=f"lookup-{secret}",
                    payload={"tool_call_id": "call"},
                ),
                _terminal("root", SessionStatus.COMPLETED),
            ),
            "transcript": (
                Message.tool_call(
                    calls=[
                        ToolCallPart(
                            tool_call_id="call",
                            tool_name=f"lookup-{secret}",
                        )
                    ]
                ),
                Message.text("assistant", f"Approved {secret}"),
            ),
            "final_output": f"Approved {secret}",
        }
    )
    policy = EvaluationEvidencePolicySpec.standard()
    specs = (
        FinalOutputEqualsAssertionSpec(
            id="output",
            expected=f"Approved {REDACTED_SECRET}",
        ),
        ToolCalledAssertionSpec(id="tool", tool_name=redacted_tool),
    )
    evidence = project_assertion_evidence_view(
        app,
        trajectory,
        evidence_policy=policy,
    )

    captured = evaluate_assertion_specs(specs, evidence)
    fresh = asyncio.run(
        evaluate_assertions(
            trajectory,
            tuple(
                compile_assertion_spec(
                    spec,
                    app=app,
                    evidence_policy=policy,
                    trusted_pricing=None,
                )
                for spec in specs
            ),
        )
    )

    assert evidence.final_output == f"Approved {REDACTED_SECRET}"
    assert evidence.started_tool_names == (redacted_tool,)
    assert [result.outcome for result in captured] == [EvalOutcome.PASSED] * 2
    assert [result.outcome for result in fresh] == [result.outcome for result in captured]


def test_fresh_compiled_assertions_use_the_executing_app_redaction_boundary():
    runtime_app = CayuApp(enable_logging=False)
    runtime_app.register_provider(
        ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("Approved"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {
                            "input_tokens": 1,
                            "output_tokens": 1,
                            "total_tokens": 2,
                        },
                    }
                ),
            ]
        ),
        default=True,
    )
    runtime_app.register_agent(AgentSpec(name="agent", model="fixture-model"))
    compiler_app = CayuApp(
        enable_logging=False,
        secret_redactor=SecretRedactor("Approved"),
    )
    spec = FinalOutputEqualsAssertionSpec(id="output", expected="Approved")
    case = EvalCase(
        id="case",
        request=RunRequest(
            agent_name="agent",
            messages=[Message.text("user", "Run the case.")],
            max_steps=1,
        ),
        assertions=[
            compile_assertion_spec(
                spec,
                app=compiler_app,
                evidence_policy=EvaluationEvidencePolicySpec.standard(),
                trusted_pricing=None,
            )
        ],
    )

    result = asyncio.run(run_eval_case(runtime_app, case, suite_id="suite"))

    assertion = result.trials[0].assertions[0]
    assert result.trials[0].final_output == "Approved"
    assert assertion.outcome is EvalOutcome.PASSED
    assert assertion.assertion_revision == assertion_spec_revision(spec)


class _FailingProvider(ModelProvider):
    name = "failing"

    async def stream(self, request):
        if request is not None:
            raise RuntimeError("model exploded")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


@pytest.mark.parametrize(
    ("expected", "outcome", "trial_status"),
    [
        ("completed", EvalOutcome.FAILED, "failed"),
        ("failed", EvalOutcome.PASSED, "passed"),
    ],
)
def test_fresh_compiled_root_status_owns_failed_session(expected, outcome, trial_status):
    app = CayuApp(enable_logging=False)
    app.register_provider(_FailingProvider(), default=True)
    app.register_agent(AgentSpec(name="agent", model="fixture-model"))
    spec = RootStatusAssertionSpec(id="root", expected=expected)
    case = EvalCase(
        id="case",
        request=RunRequest(
            agent_name="agent",
            messages=[Message.text("user", "Run the case.")],
            max_steps=1,
        ),
        assertions=[
            compile_assertion_spec(
                spec,
                app=app,
                evidence_policy=EvaluationEvidencePolicySpec.standard(),
                trusted_pricing=None,
            )
        ],
    )

    result = asyncio.run(run_eval_case(app, case, suite_id="suite"))

    trial = result.trials[0]
    assertion = trial.assertions[0]
    assert trial.status.value == trial_status
    assert trial.error is None
    assert assertion.outcome is outcome
    assert assertion.assertion_revision == assertion_spec_revision(spec)


def test_blocked_compiled_assertions_retain_their_definition_revision():
    app = CayuApp(enable_logging=False)
    spec = RootStatusAssertionSpec(id="root", expected="completed")
    case = EvalCase(
        id="case",
        request=RunRequest(
            agent_name="missing-agent",
            messages=[Message.text("user", "Run the case.")],
        ),
        assertions=[
            compile_assertion_spec(
                spec,
                app=app,
                evidence_policy=EvaluationEvidencePolicySpec.standard(),
                trusted_pricing=None,
            )
        ],
    )

    result = asyncio.run(run_eval_case(app, case, suite_id="suite"))

    assertion = result.trials[0].assertions[0]
    assert assertion.outcome is EvalOutcome.ERROR
    assert assertion.assertion_revision == assertion_spec_revision(spec)


def test_direct_and_compiled_assertions_agree_when_root_evidence_is_missing():
    app = CayuApp(enable_logging=False)
    policy = EvaluationEvidencePolicySpec.standard()
    direct = (
        ChildSessionCompleted(min_count=0),
        FinalOutputContains("x"),
        ToolCalled("lookup", min_count=0),
        ToolNotCalled("lookup"),
        ToolsCalledInOrder([]),
        MaxToolCalls(0),
        MaxModelSteps(0),
    )
    specs = (
        ChildStatusAssertionSpec(id="child", expected="completed", min_count=0),
        FinalOutputContainsAssertionSpec(id="output", expected="x"),
        ToolCalledAssertionSpec(id="tool", tool_name="lookup", min_count=0),
        ToolCalledAssertionSpec(id="tool-not", tool_name="lookup", min_count=0, max_count=0),
        ToolsCalledInOrderAssertionSpec(id="order", tool_names=()),
        MaxToolCallsAssertionSpec(id="tools", maximum=0),
        MaxModelStepsAssertionSpec(id="steps", maximum=0),
    )

    direct_results = asyncio.run(evaluate_assertions(Trajectory(), direct))
    compiled_results = asyncio.run(
        evaluate_assertions(
            Trajectory(),
            tuple(
                compile_assertion_spec(
                    spec,
                    app=app,
                    evidence_policy=policy,
                    trusted_pricing=None,
                )
                for spec in specs
            ),
        )
    )

    assert [result.outcome for result in direct_results] == [EvalOutcome.UNAVAILABLE] * 7
    assert [result.outcome for result in compiled_results] == [
        result.outcome for result in direct_results
    ]


def test_direct_and_compiled_model_step_assertions_ignore_orphan_usage():
    app = CayuApp(enable_logging=False)
    trajectory = Trajectory(
        usage_summary=SessionUsageSummary(
            session_id="orphan",
            model_steps=1,
            usage=UsageMetrics(total_tokens=1),
        )
    )
    compiled = compile_assertion_spec(
        MaxModelStepsAssertionSpec(id="steps", maximum=0),
        app=app,
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
        trusted_pricing=None,
    )

    direct_result = asyncio.run(evaluate_assertions(trajectory, (MaxModelSteps(0),)))[0]
    compiled_result = asyncio.run(evaluate_assertions(trajectory, (compiled,)))[0]

    assert direct_result.outcome is EvalOutcome.UNAVAILABLE
    assert compiled_result.outcome is EvalOutcome.UNAVAILABLE


def _output_boundary_trajectory(length: int) -> Trajectory:
    marker = "needle"
    output = "x" * (length - len(marker)) + marker
    trajectory = _trajectory()
    return trajectory.model_copy(
        update={
            "transcript": (
                trajectory.transcript[0],
                Message.text("assistant", output),
            ),
            "final_output": output,
        }
    )


def _tool_boundary_trajectory(count: int) -> Trajectory:
    root_id = "root"
    events = (
        *(
            Event(
                type=EventType.TOOL_CALL_STARTED,
                session_id=root_id,
                tool_name="lookup",
                payload={"tool_call_id": f"call-{index}"},
            )
            for index in range(count)
        ),
        _terminal(root_id, SessionStatus.COMPLETED),
    )
    calls = [
        ToolCallPart(tool_call_id=f"call-{index}", tool_name="lookup") for index in range(count)
    ]
    transcript = (Message.tool_call(calls=calls), Message.text("assistant", "Approved"))
    trajectory = _trajectory()
    return trajectory.model_copy(
        update={
            "events": events,
            "transcript": transcript,
            "usage_summary": session_usage_summary(root_id, list(events)),
        }
    )


def _model_boundary_trajectory(count: int) -> Trajectory:
    root_id = "root"
    events = (
        *(Event(type=EventType.MODEL_COMPLETED, session_id=root_id) for _ in range(count)),
        _terminal(root_id, SessionStatus.COMPLETED),
    )
    trajectory = _trajectory()
    return trajectory.model_copy(
        update={
            "events": events,
            "usage_summary": session_usage_summary(root_id, list(events)),
        }
    )


def _large_usage_trajectory() -> Trajectory:
    root_id = "root"
    per_step_tokens = 2**62
    events = (
        *(
            Event(
                type=EventType.MODEL_COMPLETED,
                session_id=root_id,
                payload={
                    "usage_metrics": {
                        "input_tokens": per_step_tokens,
                        "output_tokens": 0,
                        "total_tokens": per_step_tokens,
                    }
                },
            )
            for _ in range(2)
        ),
        _terminal(root_id, SessionStatus.COMPLETED),
    )
    trajectory = _trajectory()
    return trajectory.model_copy(
        update={
            "events": events,
            "transcript": (),
            "final_output": "",
            "usage_summary": session_usage_summary(root_id, list(events)),
        }
    )


def test_direct_and_compiled_token_assertions_reject_unbounded_aggregate_usage():
    trajectory = _large_usage_trajectory()
    app = CayuApp(enable_logging=False)
    compiled = compile_assertion_spec(
        MaxTotalTokensAssertionSpec(id="tokens", maximum=EVIDENCE_MAX_TOTAL_TOKENS),
        app=app,
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
        trusted_pricing=None,
    )

    direct_result = asyncio.run(
        evaluate_assertions(trajectory, (MaxTotalTokens(EVIDENCE_MAX_TOTAL_TOKENS),))
    )[0]
    compiled_result = asyncio.run(evaluate_assertions(trajectory, (compiled,)))[0]

    assert direct_result.outcome is EvalOutcome.UNAVAILABLE
    assert compiled_result.outcome is EvalOutcome.UNAVAILABLE
    assert direct_result.metadata["evidence_state"] == "limit_exceeded"
    assert compiled_result.metadata["evidence_state"] == "limit_exceeded"


def _child_boundary_trajectory(count: int) -> Trajectory:
    root = _trajectory()
    children = []
    for index in range(count):
        child_id = f"child-{index}"
        events = (_terminal(child_id, SessionStatus.COMPLETED),)
        children.append(
            Trajectory(
                session=Session(
                    id=child_id,
                    agent_name="child",
                    provider_name="fixture",
                    model="fixture-model",
                    causal_budget_id="budget",
                    parent_session_id="root",
                    invocation=fixture_session_invocation(
                        child_id,
                        parent_session_id="root",
                    ),
                    status=SessionStatus.COMPLETED,
                ),
                events=events,
                usage_summary=session_usage_summary(child_id, list(events)),
            )
        )
    return root.model_copy(update={"children": tuple(children)})


@pytest.mark.parametrize(
    ("trajectory", "direct", "spec", "expected"),
    [
        (
            _output_boundary_trajectory(EVIDENCE_MAX_FINAL_OUTPUT_CHARS),
            FinalOutputContains("needle"),
            FinalOutputContainsAssertionSpec(id="output", expected="needle"),
            EvalOutcome.PASSED,
        ),
        (
            _output_boundary_trajectory(EVIDENCE_MAX_FINAL_OUTPUT_CHARS + 1),
            FinalOutputContains("needle"),
            FinalOutputContainsAssertionSpec(id="output", expected="needle"),
            EvalOutcome.UNAVAILABLE,
        ),
        (
            _tool_boundary_trajectory(EVIDENCE_MAX_TOOL_CALLS),
            MaxToolCalls(EVIDENCE_MAX_TOOL_CALLS),
            MaxToolCallsAssertionSpec(id="tools", maximum=EVIDENCE_MAX_TOOL_CALLS),
            EvalOutcome.PASSED,
        ),
        (
            _tool_boundary_trajectory(EVIDENCE_MAX_TOOL_CALLS + 1),
            MaxToolCalls(EVIDENCE_MAX_TOOL_CALLS),
            MaxToolCallsAssertionSpec(id="tools", maximum=EVIDENCE_MAX_TOOL_CALLS),
            EvalOutcome.UNAVAILABLE,
        ),
        (
            _model_boundary_trajectory(EVIDENCE_MAX_MODEL_STEPS),
            MaxModelSteps(EVIDENCE_MAX_MODEL_STEPS),
            MaxModelStepsAssertionSpec(id="steps", maximum=EVIDENCE_MAX_MODEL_STEPS),
            EvalOutcome.PASSED,
        ),
        (
            _model_boundary_trajectory(EVIDENCE_MAX_MODEL_STEPS + 1),
            MaxModelSteps(EVIDENCE_MAX_MODEL_STEPS),
            MaxModelStepsAssertionSpec(id="steps", maximum=EVIDENCE_MAX_MODEL_STEPS),
            EvalOutcome.UNAVAILABLE,
        ),
        (
            _child_boundary_trajectory(EVIDENCE_MAX_CHILD_SESSIONS),
            ChildSessionCompleted(min_count=EVIDENCE_MAX_CHILD_SESSIONS),
            ChildStatusAssertionSpec(
                id="children",
                expected="completed",
                min_count=EVIDENCE_MAX_CHILD_SESSIONS,
            ),
            EvalOutcome.PASSED,
        ),
        (
            _child_boundary_trajectory(EVIDENCE_MAX_CHILD_SESSIONS + 1),
            ChildSessionCompleted(min_count=EVIDENCE_MAX_CHILD_SESSIONS),
            ChildStatusAssertionSpec(
                id="children",
                expected="completed",
                min_count=EVIDENCE_MAX_CHILD_SESSIONS,
            ),
            EvalOutcome.UNAVAILABLE,
        ),
    ],
)
def test_direct_and_compiled_assertions_share_portable_evidence_boundaries(
    trajectory,
    direct,
    spec,
    expected,
):
    app = CayuApp(enable_logging=False)
    compiled = compile_assertion_spec(
        spec,
        app=app,
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
        trusted_pricing=None,
    )

    direct_result = asyncio.run(evaluate_assertions(trajectory, (direct,)))[0]
    compiled_result = asyncio.run(evaluate_assertions(trajectory, (compiled,)))[0]

    assert direct_result.outcome is expected
    assert compiled_result.outcome is expected


def test_direct_and_compiled_cost_assertions_reject_over_limit_model_evidence():
    trajectory = _model_boundary_trajectory(EVIDENCE_MAX_MODEL_STEPS + 1)
    app = CayuApp(enable_logging=False)
    pricing = _pricing()
    spec = MaxEstimatedCostAssertionSpec(id="cost", maximum="1", currency="USD")
    compiled = compile_assertion_spec(
        spec,
        app=app,
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
        trusted_pricing=pricing,
    )

    direct_result = asyncio.run(evaluate_assertions(trajectory, (MaxEstimatedCost("1", pricing),)))[
        0
    ]
    compiled_result = asyncio.run(evaluate_assertions(trajectory, (compiled,)))[0]

    assert direct_result.outcome is EvalOutcome.UNAVAILABLE
    assert compiled_result.outcome is EvalOutcome.UNAVAILABLE


def test_compiled_assertion_revision_is_hashed_once_at_compile_time(monkeypatch):
    import cayu.evals.portable_assertions as portable_assertions_module
    import cayu.evals.portable_evaluation as portable_evaluation_module

    app = CayuApp(enable_logging=False)
    compiled = compile_assertion_spec(
        RootStatusAssertionSpec(id="root", expected="completed"),
        app=app,
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
        trusted_pricing=None,
    )

    def unexpected_rehash(*args, **kwargs):
        raise AssertionError("compiled assertion revision was recomputed")

    monkeypatch.setattr(
        portable_assertions_module,
        "assertion_spec_revision",
        unexpected_rehash,
    )
    monkeypatch.setattr(
        portable_evaluation_module,
        "assertion_spec_revision",
        unexpected_rehash,
    )

    result = asyncio.run(evaluate_assertions(_trajectory(), (compiled,)))[0]

    assert result.outcome is EvalOutcome.PASSED


def test_compiled_assertion_contract_cannot_be_reassigned_after_compilation():
    compiled = compile_assertion_spec(
        FinalOutputEqualsAssertionSpec(id="output", expected="Denied"),
        app=CayuApp(enable_logging=False),
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
        trusted_pricing=None,
    )

    with pytest.raises(AttributeError, match="immutable"):
        compiled.spec = FinalOutputEqualsAssertionSpec(id="output", expected="Approved")


@pytest.mark.parametrize("mutated_index", [0, 1])
def test_compiled_cost_assertions_reject_any_mutated_trusted_pricing(mutated_index):
    app = CayuApp(enable_logging=False)
    pricing_sources = (_pricing(), _pricing())
    compiled = tuple(
        compile_assertion_spec(
            MaxEstimatedCostAssertionSpec(id=f"cost-{index}", maximum="1", currency="USD"),
            app=app,
            evidence_policy=EvaluationEvidencePolicySpec.standard(),
            trusted_pricing=pricing_sources[index],
        )
        for index in range(2)
    )
    changed_pricing = PriceBook(
        price_book_version="v1",
        generated_at="2026-08-05T00:00:00Z",
        prices=(
            ModelPrice.fixed(
                provider_name="fixture",
                model="fixture-model",
                input_per_million=Decimal("1000000"),
                output_per_million=Decimal("2000000"),
            ),
        ),
    )
    pricing_sources[mutated_index].prices = changed_pricing.prices

    results = asyncio.run(evaluate_assertions(_trajectory(), compiled))

    assert [result.outcome for result in results] == [EvalOutcome.ERROR, EvalOutcome.ERROR]
    assert all(
        "pricing profile changed after assertion compilation" in result.message
        for result in results
    )
    assert [result.assertion_revision for result in results] == [
        assertion.assertion_revision for assertion in compiled
    ]


def test_compiled_cost_assertions_share_one_pricing_snapshot_per_trial(monkeypatch):
    import cayu.evals.portable_assertions as portable_assertions_module

    app = CayuApp(enable_logging=False)
    policy = EvaluationEvidencePolicySpec.standard()
    pricing = _pricing()
    compiled = tuple(
        compile_assertion_spec(
            MaxEstimatedCostAssertionSpec(
                id=f"cost-{index}",
                maximum="1",
                currency="USD",
            ),
            app=app,
            evidence_policy=policy,
            trusted_pricing=pricing,
        )
        for index in range(32)
    )

    bindings = tuple(
        object.__getattribute__(assertion, "_pricing_binding") for assertion in compiled
    )
    assert {id(binding.source) for binding in bindings} == {id(pricing)}

    validation_calls = 0
    validated_pricing = portable_assertions_module._validated_pricing

    def counted_validation(source):
        nonlocal validation_calls
        validation_calls += 1
        return validated_pricing(source)

    monkeypatch.setattr(
        portable_assertions_module,
        "_validated_pricing",
        counted_validation,
    )

    results = asyncio.run(evaluate_assertions(_trajectory(), compiled))

    assert [result.outcome for result in results] == [EvalOutcome.PASSED] * len(compiled)
    assert validation_calls == 1


def test_compiled_cost_assertions_accept_canonically_equivalent_price_books():
    primary = ModelPrice.fixed(
        provider_name="fixture",
        model="fixture-model",
        input_per_million=Decimal("1"),
        output_per_million=Decimal("2"),
    )
    secondary = ModelPrice.fixed(
        provider_name="other",
        model="other-model",
        input_per_million=Decimal("3"),
        output_per_million=Decimal("4"),
    )
    books = tuple(
        PriceBook(
            price_book_version="v1",
            generated_at="2026-08-05T00:00:00Z",
            prices=prices,
        )
        for prices in ((primary, secondary), (secondary, primary))
    )
    app = CayuApp(enable_logging=False)
    policy = EvaluationEvidencePolicySpec.standard()
    compiled = tuple(
        compile_assertion_spec(
            MaxEstimatedCostAssertionSpec(
                id=f"cost-{index}",
                maximum="1",
                currency="USD",
            ),
            app=app,
            evidence_policy=policy,
            trusted_pricing=book,
        )
        for index, book in enumerate(books)
    )

    results = asyncio.run(evaluate_assertions(_trajectory(), compiled))

    assert [result.outcome for result in results] == [EvalOutcome.PASSED, EvalOutcome.PASSED]


def test_compiled_cost_assertions_reject_too_many_distinct_currencies_before_projection():
    app = CayuApp(enable_logging=False)
    policy = EvaluationEvidencePolicySpec.standard()
    compiled = tuple(
        compile_assertion_spec(
            MaxEstimatedCostAssertionSpec(
                id=f"cost-{index}", maximum="1", currency=f"C{index:02d}"
            ),
            app=app,
            evidence_policy=policy,
            trusted_pricing=_pricing(),
        )
        for index in range(33)
    )

    results = asyncio.run(evaluate_assertions(_trajectory(), compiled))

    assert [result.outcome for result in results] == [EvalOutcome.ERROR] * len(compiled)
    assert all("currency limit of 32" in result.message for result in results)


def test_batch_evaluation_rejects_unordered_assertion_specs():
    evidence = project_assertion_evidence_view(
        CayuApp(enable_logging=False),
        _trajectory(),
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
    )

    with pytest.raises(TypeError, match="specs must be an ordered sequence"):
        evaluate_assertion_specs(
            {RootStatusAssertionSpec(id="root", expected="completed")},
            evidence,
        )


def test_unconstrained_runners_cannot_accept_a_portable_contract():
    assert "corpus" not in inspect.signature(EvalPlan).parameters
    assert "run_contract" not in inspect.signature(run_eval_suite).parameters


def test_started_tool_without_transcript_request_is_unavailable_not_an_empty_order_pass():
    app = CayuApp(enable_logging=False)
    source = _trajectory()
    data = source.model_dump(mode="python")
    data["transcript"] = (Message.text("assistant", "Approved"),)
    trajectory = Trajectory.model_validate(data)
    policy = EvaluationEvidencePolicySpec.standard()
    spec = ToolsCalledInOrderAssertionSpec(id="order", tool_names=())

    direct = asyncio.run(evaluate_assertions(trajectory, (ToolsCalledInOrder(()),)))[0]
    compiled = asyncio.run(
        evaluate_assertions(
            trajectory,
            (
                compile_assertion_spec(
                    spec,
                    app=app,
                    evidence_policy=policy,
                    trusted_pricing=None,
                ),
            ),
        )
    )[0]

    assert direct.outcome is EvalOutcome.UNAVAILABLE
    assert compiled.outcome is EvalOutcome.UNAVAILABLE


def test_direct_and_compiled_assertions_agree_on_incomplete_and_complete_negative_evidence():
    app = CayuApp(enable_logging=False)
    policy = EvaluationEvidencePolicySpec.standard()
    incomplete_direct = (ChildSessionCompleted(min_count=2),)
    incomplete_specs = (ChildStatusAssertionSpec(id="child", expected="completed", min_count=2),)
    incomplete = _trajectory(children_incomplete=True)

    direct_incomplete = asyncio.run(evaluate_assertions(incomplete, incomplete_direct))
    compiled_incomplete = asyncio.run(
        evaluate_assertions(
            incomplete,
            tuple(
                compile_assertion_spec(
                    spec,
                    app=app,
                    evidence_policy=policy,
                    trusted_pricing=None,
                )
                for spec in incomplete_specs
            ),
        )
    )

    direct_negative = (
        FinalOutputContains("Declined"),
        ToolCalled("missing"),
        ToolsCalledInOrder(["missing"]),
        MaxToolCalls(0),
        MaxModelSteps(0),
    )
    negative_specs = (
        FinalOutputContainsAssertionSpec(id="output", expected="Declined"),
        ToolCalledAssertionSpec(id="tool", tool_name="missing"),
        ToolsCalledInOrderAssertionSpec(id="order", tool_names=("missing",)),
        MaxToolCallsAssertionSpec(id="tools", maximum=0),
        MaxModelStepsAssertionSpec(id="steps", maximum=0),
    )
    complete = _trajectory()
    direct_complete = asyncio.run(evaluate_assertions(complete, direct_negative))
    compiled_complete = asyncio.run(
        evaluate_assertions(
            complete,
            tuple(
                compile_assertion_spec(
                    spec,
                    app=app,
                    evidence_policy=policy,
                    trusted_pricing=None,
                )
                for spec in negative_specs
            ),
        )
    )

    assert [result.outcome for result in direct_incomplete] == [EvalOutcome.UNAVAILABLE]
    assert [result.outcome for result in compiled_incomplete] == [EvalOutcome.UNAVAILABLE]
    assert [result.outcome for result in direct_complete] == [EvalOutcome.FAILED] * 5
    assert [result.outcome for result in compiled_complete] == [EvalOutcome.FAILED] * 5


def test_incomplete_evidence_never_passes_affected_assertions():
    app = CayuApp(enable_logging=False)
    evidence = project_assertion_evidence_view(
        app,
        _trajectory(children_incomplete=True),
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
    )
    results = evaluate_assertion_specs(
        (
            ChildStatusAssertionSpec(id="child", expected="failed"),
            MaxEstimatedCostAssertionSpec(id="cost", maximum="1", currency="USD"),
        ),
        evidence,
    )

    assert [result.outcome for result in results] == [
        EvalOutcome.UNAVAILABLE,
        EvalOutcome.UNAVAILABLE,
    ]
    assert all(result.score is None for result in results)


def test_portable_assertion_failures_are_observed_negative_evidence():
    evidence = project_assertion_evidence_view(
        CayuApp(enable_logging=False),
        _trajectory(),
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
    )
    results = evaluate_assertion_specs(
        (
            RootStatusAssertionSpec(id="root", expected="failed"),
            FinalOutputEqualsAssertionSpec(id="output", expected="Declined"),
            ToolCalledAssertionSpec(id="tool", tool_name="missing"),
            MaxTotalTokensAssertionSpec(id="tokens", maximum=14),
        ),
        evidence,
    )

    assert [result.outcome for result in results] == [EvalOutcome.FAILED] * 4
    assert all(result.score == 0.0 for result in results)


def test_compile_rejects_assertion_and_pricing_subclasses():
    class CustomAssertion(RootStatusAssertionSpec):
        pass

    class CustomPriceBook(PriceBook):
        pass

    app = CayuApp(enable_logging=False)
    policy = EvaluationEvidencePolicySpec.standard()

    with pytest.raises(TypeError, match="exact built-in"):
        compile_assertion_spec(
            CustomAssertion(id="custom", expected="completed"),
            app=app,
            evidence_policy=policy,
            trusted_pricing=None,
        )
    with pytest.raises(TypeError, match="exact PriceBook"):
        compile_assertion_spec(
            RootStatusAssertionSpec(id="root", expected="completed"),
            app=app,
            evidence_policy=policy,
            trusted_pricing=CustomPriceBook.model_validate(_pricing().model_dump(mode="python")),
        )
