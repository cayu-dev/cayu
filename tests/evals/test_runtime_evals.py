from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from cayu import (
    EVAL_SCHEMA_VERSION,
    AgentSpec,
    ArtifactCreated,
    CayuApp,
    Environment,
    EnvironmentSpec,
    EvalAssertion,
    EvalAssertionResult,
    EvalCase,
    EvalCaseResult,
    EvalContext,
    EvalRun,
    EvalStatus,
    EvalSuite,
    EventNotOccurred,
    EventOccurred,
    EventType,
    FinalOutputContains,
    LocalWorkspace,
    MaxModelSteps,
    MaxToolCalls,
    MaxTotalTokens,
    Message,
    RunRequest,
    ScriptedModelProvider,
    SessionCompleted,
    SessionFailed,
    SubagentSpec,
    SubagentTool,
    Tool,
    ToolArgsContain,
    ToolCalled,
    ToolContext,
    ToolNotCalled,
    ToolResult,
    ToolResultContains,
    ToolSpec,
    Trajectory,
    TrajectoryProbes,
    WorkspaceFileContains,
    compare_eval_runs,
    eval_run_to_json,
    load_eval_run,
    render_comparison_html,
    render_html_report,
    run_eval_suite,
)
from cayu.artifacts import ArtifactMetadata, ArtifactScope
from cayu.cli import main
from cayu.evals import (
    LLMJudge,
    WorkspaceFileExists,
    evaluate_assertions,
    load_trajectory,
    run_eval_case,
    write_trajectory_json,
)
from cayu.evals.runner import _build_child_trajectories
from cayu.providers import ModelProvider, ModelStreamEvent
from cayu.runtime import InMemorySessionStore, SessionIdentity
from cayu.runtime.sessions import Session


def _session(*, session_id: str = "sess_eval", environment_name: str | None = None) -> Session:
    return Session(
        id=session_id,
        agent_name="agent",
        provider_name="fake",
        model="fake-model",
        causal_budget_id="cb",
        environment_name=environment_name,
    )


def _context(
    *,
    session: Session | None = None,
    events: tuple = (),
    transcript: tuple = (),
    usage_summary=None,
    final_output: str = "",
    probes: TrajectoryProbes | None = None,
    metadata: dict | None = None,
) -> EvalContext:
    trajectory = Trajectory(
        session=session,
        events=events,
        transcript=transcript,
        usage_summary=usage_summary,
        final_output=final_output,
        probes=probes if probes is not None else TrajectoryProbes(),
        metadata=metadata or {},
    )
    return EvalContext(trajectory=trajectory, suite_id="s", case_id="c", metadata=metadata or {})


class EchoTool(Tool):
    spec = ToolSpec(
        name="echo",
        description="Echo text.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(content=f"echo: {args['text']}", structured={"text": args["text"]})


def test_eval_suite_runs_assertions_over_runtime_state(tmp_path):
    (tmp_path / "README.md").write_text("Installation\n\nUse cayu eval.\n", encoding="utf-8")
    app = CayuApp(enable_logging=False)
    app.register_provider(
        ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("Installation section added"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {
                            "input_tokens": 3,
                            "output_tokens": 4,
                            "total_tokens": 7,
                        },
                    }
                ),
            ]
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="coder", model="fake-model"))
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            workspace=LocalWorkspace(tmp_path),
        ),
        default=True,
    )
    suite = EvalSuite(
        id="readme",
        cases=[
            EvalCase(
                id="adds-installation",
                request=RunRequest(
                    agent_name="coder",
                    messages=[Message.text("user", "Update README.md")],
                    max_steps=1,
                ),
                assertions=[
                    SessionCompleted(),
                    FinalOutputContains("Installation"),
                    EventOccurred(EventType.MODEL_COMPLETED),
                    MaxModelSteps(1),
                    MaxToolCalls(0),
                    WorkspaceFileContains("README.md", "Installation"),
                ],
            )
        ],
    )

    result = asyncio.run(run_eval_suite(app, suite))

    assert result.status == EvalStatus.PASSED
    assert result.score == 1.0
    assert result.cases[0].session_id is not None
    assert result.cases[0].usage_summary["usage"]["total_tokens"] == 7


def test_eval_suite_asserts_tool_trajectory():
    app = CayuApp(enable_logging=False)
    app.register_provider(
        ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_1",
                        name="echo",
                        arguments={"text": "hi"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("echoed hi"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="coder", model="fake-model"), tools=[EchoTool()])
    suite = EvalSuite(
        id="tools",
        cases=[
            EvalCase(
                id="echo-call",
                request=RunRequest(
                    agent_name="coder",
                    messages=[Message.text("user", "echo hi")],
                    max_steps=2,
                ),
                assertions=[
                    SessionCompleted(),
                    ToolCalled("echo"),
                    ToolArgsContain("echo", {"text": "hi"}),
                    ToolResultContains("echo", "echo: hi"),
                    FinalOutputContains("echoed hi"),
                ],
            )
        ],
    )

    result = asyncio.run(run_eval_suite(app, suite))

    assert result.status == EvalStatus.PASSED
    assert result.cases[0].events_count >= 1


def test_eval_json_html_and_compare(tmp_path):
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
    suite = EvalSuite(
        id="basic",
        cases=[
            EvalCase(
                id="ok",
                request=RunRequest(
                    agent_name="agent",
                    messages=[Message.text("user", "say ok")],
                    max_steps=1,
                ),
                assertions=[SessionCompleted(), FinalOutputContains("ok")],
            )
        ],
    )
    result = asyncio.run(run_eval_suite(app, suite))
    output = tmp_path / "result.json"
    output.write_text(eval_run_to_json(result), encoding="utf-8")

    loaded = load_eval_run(output)
    html = render_html_report(loaded)
    comparison = compare_eval_runs(loaded, loaded)

    assert loaded == result
    assert "Cayu Eval Report" in html
    assert comparison.regressions == ()
    assert "Cayu Eval Comparison" in render_comparison_html(comparison)


def test_eval_cli_run_and_report(tmp_path, monkeypatch, capsys):
    module = tmp_path / "sample_eval.py"
    output = tmp_path / "results.json"
    report = tmp_path / "report.html"
    module.write_text(
        """
from cayu import (
    AgentSpec,
    CayuApp,
    EvalCase,
    EvalSuite,
    FinalOutputContains,
    Message,
    RunRequest,
    ScriptedModelProvider,
    SessionCompleted,
)
from cayu.providers import ModelStreamEvent


def build():
    app = CayuApp(enable_logging=False)
    app.register_provider(
        ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("hello eval"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="agent", model="fake-model"))
    suite = EvalSuite(
        id="cli",
        cases=[
            EvalCase(
                id="hello",
                request=RunRequest(
                    agent_name="agent",
                    messages=[Message.text("user", "hello")],
                    max_steps=1,
                ),
                assertions=[SessionCompleted(), FinalOutputContains("hello eval")],
            )
        ],
    )
    return app, suite
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    assert main(["eval", "run", "sample_eval:build", "--output", str(output)]) == 0
    run_data = json.loads(output.read_text(encoding="utf-8"))
    assert run_data["status"] == "passed"

    assert main(["eval", "report", str(output), "--output", str(report)]) == 0
    assert "Cayu Eval Report" in report.read_text(encoding="utf-8")

    captured = capsys.readouterr()
    assert captured.err == ""


class _FailingProvider(ModelProvider):
    name = "failing"

    async def stream(self, request):
        if request is not None:
            raise RuntimeError("model exploded")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})  # keeps this a generator


def _failing_app() -> CayuApp:
    app = CayuApp(enable_logging=False)
    app.register_provider(_FailingProvider(), default=True)
    app.register_agent(AgentSpec(name="agent", model="fake-model"))
    return app


def _failing_suite(suite_id, assertions) -> EvalSuite:
    return EvalSuite(
        id=suite_id,
        cases=[
            EvalCase(
                id="boom",
                request=RunRequest(
                    agent_name="agent",
                    messages=[Message.text("user", "go")],
                    max_steps=1,
                ),
                assertions=assertions,
            )
        ],
    )


def test_failed_run_reports_error_not_passed():
    # app.run() ends a failed run as SESSION_FAILED without raising; the eval must
    # surface that as ERROR, not score it on assertions alone.
    result = asyncio.run(
        run_eval_suite(_failing_app(), _failing_suite("fail", [FinalOutputContains("x")]))
    )
    assert result.cases[0].status == EvalStatus.ERROR
    assert result.status == EvalStatus.ERROR
    assert result.cases[0].error is not None


def test_failed_run_with_status_assertion_is_not_overridden():
    # A case that deliberately asserts a failed status owns the outcome.
    result = asyncio.run(
        run_eval_suite(_failing_app(), _failing_suite("expected-fail", [SessionFailed()]))
    )
    assert result.cases[0].status == EvalStatus.PASSED


class _HangingProvider(ModelProvider):
    name = "hanging"

    async def stream(self, request):
        await asyncio.sleep(60)
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _OverlapProbeProvider(ModelProvider):
    """Blocks every stream until `expected` are in flight — proves cases overlapped."""

    name = "overlap"

    def __init__(self, expected: int) -> None:
        self._expected = expected
        self._gate = asyncio.Event()
        self._active = 0
        self.max_active = 0

    async def stream(self, request):
        self._active += 1
        self.max_active = max(self.max_active, self._active)
        if self._active >= self._expected:
            self._gate.set()
        try:
            await asyncio.wait_for(self._gate.wait(), timeout=5)
        finally:
            self._active -= 1
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


def _app_with_provider(provider: ModelProvider) -> CayuApp:
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="agent", model="fake-model"))
    return app


def _case(case_id: str) -> EvalCase:
    return EvalCase(
        id=case_id,
        request=RunRequest(
            agent_name="agent",
            messages=[Message.text("user", "go")],
            max_steps=1,
        ),
        assertions=[FinalOutputContains("done")],
    )


def test_case_timeout_records_error_instead_of_hanging():
    result = asyncio.run(
        run_eval_suite(
            _app_with_provider(_HangingProvider()),
            EvalSuite(id="timeout", cases=[_case("hangs")]),
            case_timeout_seconds=0.05,
        )
    )
    assert result.status == EvalStatus.ERROR
    assert result.cases[0].status == EvalStatus.ERROR
    assert "timed out after 0.05 seconds" in result.cases[0].error


def test_max_concurrency_runs_cases_in_parallel_and_keeps_order():
    provider = _OverlapProbeProvider(expected=2)
    suite = EvalSuite(id="parallel", cases=[_case("a"), _case("b")])

    # Sequential execution would deadlock on the gate; overlap is what releases it.
    result = asyncio.run(run_eval_suite(_app_with_provider(provider), suite, max_concurrency=2))

    assert provider.max_active == 2
    assert [case.case_id for case in result.cases] == ["a", "b"]
    assert result.status == EvalStatus.PASSED


def test_max_concurrency_semaphore_caps_in_flight_cases():
    provider = _OverlapProbeProvider(expected=2)
    suite = EvalSuite(id="capped", cases=[_case("a"), _case("b"), _case("c")])

    result = asyncio.run(run_eval_suite(_app_with_provider(provider), suite, max_concurrency=2))

    assert provider.max_active == 2
    assert [case.case_id for case in result.cases] == ["a", "b", "c"]
    assert result.status == EvalStatus.PASSED


def test_run_eval_suite_rejects_invalid_concurrency_and_timeout():
    app = _app_with_provider(_FailingProvider())
    suite = _failing_suite("invalid", [])
    with pytest.raises(ValueError, match="max_concurrency"):
        asyncio.run(run_eval_suite(app, suite, max_concurrency=0))
    with pytest.raises(TypeError, match="max_concurrency"):
        asyncio.run(run_eval_suite(app, suite, max_concurrency=True))
    with pytest.raises(ValueError, match="case_timeout_seconds"):
        asyncio.run(run_eval_suite(app, suite, case_timeout_seconds=0))
    with pytest.raises(TypeError, match="case_timeout_seconds"):
        asyncio.run(run_eval_suite(app, suite, case_timeout_seconds="5"))


def _case_result(case_id, status, score) -> EvalCaseResult:
    now = datetime.now(UTC)
    return EvalCaseResult(
        case_id=case_id, status=status, score=score, started_at=now, completed_at=now
    )


def _run(status, score, cases, *, suite_id="s") -> EvalRun:
    return EvalRun(suite_id=suite_id, status=status, score=score, cases=tuple(cases))


def test_compare_detects_status_regression():
    base = _run(EvalStatus.PASSED, 1.0, [_case_result("a", EvalStatus.PASSED, 1.0)])
    cur = _run(EvalStatus.FAILED, 0.0, [_case_result("a", EvalStatus.FAILED, 0.0)])
    comparison = compare_eval_runs(base, cur)
    assert any("status regressed" in item for item in comparison.regressions)


def test_compare_flags_removed_case_but_not_added_case():
    base = _run(EvalStatus.PASSED, 1.0, [_case_result("a", EvalStatus.PASSED, 1.0)])
    cur = _run(EvalStatus.PASSED, 1.0, [_case_result("b", EvalStatus.PASSED, 1.0)])
    comparison = compare_eval_runs(base, cur)
    # 'a' was removed -> regression; 'b' was added -> NOT a regression.
    assert any("a:" in item for item in comparison.regressions)
    assert not any(item.startswith("b:") for item in comparison.regressions)


def test_compare_rejects_mismatched_suites():
    base = _run(
        EvalStatus.PASSED,
        1.0,
        [_case_result("a", EvalStatus.PASSED, 1.0)],
        suite_id="baseline-suite",
    )
    cur = _run(
        EvalStatus.PASSED,
        1.0,
        [_case_result("a", EvalStatus.PASSED, 1.0)],
        suite_id="current-suite",
    )
    with pytest.raises(ValueError, match="different suites"):
        compare_eval_runs(base, cur)


def _validation_case(case_id: str) -> EvalCase:
    return EvalCase(
        id=case_id,
        request=RunRequest(agent_name="coder", messages=[Message.text("user", "hi")]),
    )


def test_eval_suite_rejects_duplicate_case_ids():
    # compare_eval_runs indexes cases by id, so a duplicate would run but be silently dropped
    # from every baseline comparison; the suite must reject it at construction.
    with pytest.raises(ValidationError, match="case IDs must be unique; duplicated: dupe"):
        EvalSuite(id="suite", cases=[_validation_case("dupe"), _validation_case("dupe")])


def test_eval_suite_accepts_distinct_case_ids():
    suite = EvalSuite(id="suite", cases=[_validation_case("a"), _validation_case("b")])
    assert [case.id for case in suite.cases] == ["a", "b"]


def test_eval_run_exits_nonzero_on_failing_suite(tmp_path, monkeypatch):
    module = tmp_path / "failing_eval.py"
    module.write_text(
        """
from cayu import (
    AgentSpec,
    CayuApp,
    EvalCase,
    EvalSuite,
    FinalOutputContains,
    Message,
    RunRequest,
    ScriptedModelProvider,
)
from cayu.providers import ModelStreamEvent


def build():
    app = CayuApp(enable_logging=False)
    app.register_provider(
        ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("nope"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="agent", model="fake-model"))
    suite = EvalSuite(
        id="failing",
        cases=[
            EvalCase(
                id="wants-yes",
                request=RunRequest(
                    agent_name="agent",
                    messages=[Message.text("user", "say yes")],
                    max_steps=1,
                ),
                assertions=[FinalOutputContains("yes")],
            )
        ],
    )
    return app, suite
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    assert main(["eval", "run", "failing_eval:build", "--output", str(tmp_path / "out.json")]) == 1


def test_scripted_provider_requires_completed_event():
    with pytest.raises(ValueError, match="COMPLETED"):
        ScriptedModelProvider([ModelStreamEvent.text_delta("no completion")])


def test_event_not_occurred_pass_message_reads_naturally():
    app = CayuApp(enable_logging=False)
    app.register_provider(
        ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="agent", model="fake-model"))
    suite = EvalSuite(
        id="not-occurred",
        cases=[
            EvalCase(
                id="no-tools",
                request=RunRequest(
                    agent_name="agent",
                    messages=[Message.text("user", "hi")],
                    max_steps=1,
                ),
                assertions=[EventNotOccurred(EventType.TOOL_CALL_STARTED)],
            )
        ],
    )
    result = asyncio.run(run_eval_suite(app, suite))
    assert result.cases[0].status == EvalStatus.PASSED
    assert "did not occur" in result.cases[0].assertions[0].message


def test_artifact_created_scope_none_ignores_prior_env_artifact():
    # An environment-scoped artifact from a previous case must not satisfy scope=None
    # (which resolves to SESSION scope). The assertion filters the captured probe artifacts.
    prior = ArtifactMetadata(
        id="art_prior",
        filename="out.txt",
        content_type="text/plain",
        size_bytes=3,
        scope=ArtifactScope.ENVIRONMENT,
        session_id="other",
        environment_name="local",
    )
    context = _context(
        session=_session(session_id="sess_1", environment_name="local"),
        probes=TrajectoryProbes(artifacts_available=True, artifacts=(prior,)),
    )
    result = asyncio.run(ArtifactCreated(filename="out.txt").evaluate(context))
    assert result.passed is False


def _scored_app() -> CayuApp:
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
    return app


def test_assertion_results_carry_score_and_run_has_schema_version(tmp_path):
    suite = EvalSuite(
        id="scored",
        cases=[
            EvalCase(
                id="mixed",
                request=RunRequest(
                    agent_name="agent",
                    messages=[Message.text("user", "go")],
                    max_steps=1,
                ),
                # one passing + one failing check -> scores 1.0 and 0.0, case score 0.5.
                assertions=[SessionCompleted(), FinalOutputContains("nope")],
            )
        ],
    )
    result = asyncio.run(run_eval_suite(_scored_app(), suite))
    scores = {a.name: a.score for a in result.cases[0].assertions}
    assert scores["SessionCompleted"] == 1.0
    assert scores["FinalOutputContains"] == 0.0
    assert result.cases[0].score == 0.5  # mean of assertion scores

    output = tmp_path / "run.json"
    output.write_text(eval_run_to_json(result), encoding="utf-8")
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == 1
    assert load_eval_run(output) == result  # round-trips with the new fields


def test_load_eval_run_rejects_newer_schema_version(tmp_path):
    run = _run(EvalStatus.PASSED, 1.0, [_case_result("a", EvalStatus.PASSED, 1.0)])
    data = json.loads(eval_run_to_json(run))
    data["schema_version"] = EVAL_SCHEMA_VERSION + 1
    path = tmp_path / "future.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_eval_run(path)


def test_load_eval_run_rejects_non_int_schema_version(tmp_path):
    # A malformed schema_version (a JSON string) must raise a clean ValueError,
    # not a raw TypeError from the `>` comparison.
    run = _run(EvalStatus.PASSED, 1.0, [_case_result("a", EvalStatus.PASSED, 1.0)])
    data = json.loads(eval_run_to_json(run))
    data["schema_version"] = "2"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_eval_run(path)


def test_assertion_result_rejects_inconsistent_passed_and_score():
    # passed must agree with score >= threshold (1.0 when no threshold).
    with pytest.raises(ValidationError):
        EvalAssertionResult(name="x", passed=True, score=0.0)
    with pytest.raises(ValidationError):
        EvalAssertionResult(name="x", passed=False, score=0.6, threshold=0.5)
    # consistent pairs are accepted.
    assert EvalAssertionResult(name="x", passed=True, score=0.6, threshold=0.5).passed is True
    assert EvalAssertionResult(name="x", passed=False).score == 0.0


class _GradedAssertion(EvalAssertion):
    def __init__(self, score, threshold):
        self._score = score
        self._threshold = threshold

    async def evaluate(self, context):
        return self.score_result(self._score, threshold=self._threshold, message="graded")


def test_score_result_derives_pass_from_threshold():
    ctx = _context()
    passing = asyncio.run(_GradedAssertion(0.6, 0.5).evaluate(ctx))
    assert passing.passed is True and passing.score == 0.6 and passing.threshold == 0.5
    failing = asyncio.run(_GradedAssertion(0.4, 0.5).evaluate(ctx))
    assert failing.passed is False and failing.score == 0.4


def test_case_score_reflects_graded_assertion():
    suite = EvalSuite(
        id="graded",
        cases=[
            EvalCase(
                id="partial",
                request=RunRequest(
                    agent_name="agent",
                    messages=[Message.text("user", "go")],
                    max_steps=1,
                ),
                assertions=[_GradedAssertion(0.5, 0.0)],  # threshold 0 -> passes, score 0.5
            )
        ],
    )
    result = asyncio.run(run_eval_suite(_scored_app(), suite))
    assert result.cases[0].assertions[0].score == 0.5
    assert result.cases[0].score == 0.5  # graded score flows into the case score
    assert result.cases[0].status == EvalStatus.PASSED


def test_eval_case_result_normalizes_whitespace_error():
    # A captured exception string ending in whitespace must not crash result
    # construction (which would abort the whole suite).
    now = datetime.now(UTC)
    result = EvalCaseResult(
        case_id="c", status=EvalStatus.ERROR, error="boom\n  ", started_at=now, completed_at=now
    )
    assert result.error == "boom"
    blank = EvalCaseResult(
        case_id="c", status=EvalStatus.ERROR, error="   ", started_at=now, completed_at=now
    )
    assert blank.error is None


def test_max_total_tokens_fails_when_usage_missing():
    ctx = _context()
    result = asyncio.run(MaxTotalTokens(100).evaluate(ctx))
    assert result.passed is False


def test_tool_not_called_reports_when_tool_was_called():
    app = CayuApp(enable_logging=False)
    app.register_provider(
        ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(id="call_1", name="echo", arguments={"text": "hi"}),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="coder", model="fake-model"), tools=[EchoTool()])
    suite = EvalSuite(
        id="neg",
        cases=[
            EvalCase(
                id="echo-call",
                request=RunRequest(
                    agent_name="coder",
                    messages=[Message.text("user", "echo hi")],
                    max_steps=2,
                ),
                assertions=[ToolNotCalled("echo")],
            )
        ],
    )
    result = asyncio.run(run_eval_suite(app, suite))
    assertion = result.cases[0].assertions[0]
    assert assertion.passed is False
    assert "expected not to" in assertion.message


def test_load_eval_run_rejects_explicit_zero_schema_version(tmp_path):
    run = _run(EvalStatus.PASSED, 1.0, [_case_result("a", EvalStatus.PASSED, 1.0)])
    data = json.loads(eval_run_to_json(run))
    data["schema_version"] = 0
    path = tmp_path / "zero.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_eval_run(path)


def test_trajectory_json_round_trip(tmp_path):
    # The Trajectory is the serializable replay/export object: probe bytes (base64), a
    # probed-but-absent file (None), nested sub-agent children, and the session all survive.
    trajectory = Trajectory(
        session=_session(session_id="root", environment_name="local"),
        final_output="root output",
        probes=TrajectoryProbes(
            workspace_available=True,
            workspace_files={"a.txt": b"hello", "missing.txt": None},
            artifacts_available=True,
            artifacts=(
                ArtifactMetadata(id="art_1", filename="o.txt", size_bytes=5, session_id="root"),
            ),
        ),
        children=(Trajectory(final_output="child output"),),
    )
    path = tmp_path / "trajectory.json"
    write_trajectory_json(trajectory, path)
    restored = load_trajectory(path)
    assert restored.final_output == "root output"
    assert restored.session is not None and restored.session.id == "root"
    assert restored.probes.workspace_files == {"a.txt": b"hello", "missing.txt": None}
    assert restored.probes.artifacts[0].id == "art_1"
    assert restored.children[0].final_output == "child output"


def test_workspace_assertions_read_captured_probes():
    # Workspace assertions evaluate off the captured probe snapshot, never the live app.
    assert WorkspaceFileExists("f.txt").required_probes().workspace_paths == frozenset({"f.txt"})

    present = _context(
        session=_session(),
        probes=TrajectoryProbes(
            workspace_available=True, workspace_files={"f.txt": b"hello world"}
        ),
    )
    assert asyncio.run(WorkspaceFileExists("f.txt").evaluate(present)).passed is True
    assert asyncio.run(WorkspaceFileContains("f.txt", "world").evaluate(present)).passed is True
    assert asyncio.run(WorkspaceFileContains("f.txt", "absent").evaluate(present)).passed is False

    absent = _context(
        session=_session(),
        probes=TrajectoryProbes(workspace_available=True, workspace_files={"f.txt": None}),
    )
    assert asyncio.run(WorkspaceFileExists("f.txt").evaluate(absent)).passed is False

    no_workspace = _context(session=_session(), probes=TrajectoryProbes(workspace_available=False))
    result = asyncio.run(WorkspaceFileExists("f.txt").evaluate(no_workspace))
    assert result.passed is False
    assert "No workspace" in result.message


def test_workspace_assertion_distinguishes_uncaptured_from_absent():
    # Replaying against a path the run never probed (missing key) must report "not captured",
    # distinct from a captured-but-absent file (value None -> "not found"/"could not read").
    uncaptured = _context(
        session=_session(),
        probes=TrajectoryProbes(workspace_available=True, workspace_files={"other.txt": b"x"}),
    )
    r_exists = asyncio.run(WorkspaceFileExists("missing.txt").evaluate(uncaptured))
    assert r_exists.passed is False and "not captured" in r_exists.message
    r_contains = asyncio.run(WorkspaceFileContains("missing.txt", "x").evaluate(uncaptured))
    assert r_contains.passed is False and "not captured" in r_contains.message

    absent = _context(
        session=_session(),
        probes=TrajectoryProbes(workspace_available=True, workspace_files={"missing.txt": None}),
    )
    r_absent = asyncio.run(WorkspaceFileExists("missing.txt").evaluate(absent))
    assert r_absent.passed is False
    assert "not found" in r_absent.message and "not captured" not in r_absent.message


def test_eval_case_captures_sub_agent_children():
    # A parent agent that spawns a foreground sub-agent -> the runner captures the sub-agent
    # run as a child Trajectory (the full spawn -> parent_session_id link -> walk chain),
    # deterministically via a scripted provider (no live model).
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(
        ScriptedModelProvider(
            [
                [  # parent turn 1: delegate to the sub-agent
                    ModelStreamEvent.tool_call(
                        id="c1",
                        name="subagent",
                        arguments={"agent": "helper", "task": "Summarize."},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [  # child turn: the sub-agent answers
                    ModelStreamEvent.text_delta("subagent summary done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
                [  # parent turn 2: final answer
                    ModelStreamEvent.text_delta("parent finished"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="parent", model="fake-model"),
        tools=[SubagentTool(app, agents={"helper": SubagentSpec(agent_name="helper")})],
    )
    app.register_agent(AgentSpec(name="helper", model="fake-model"))

    case = EvalCase(
        id="with-subagent",
        request=RunRequest(
            agent_name="parent",
            session_id="parent",
            messages=[Message.text("user", "Delegate then summarize.")],
            max_steps=5,
        ),
        assertions=[SessionCompleted()],
    )
    result = asyncio.run(run_eval_case(app, case, suite_id="s", retain_trajectory=True))

    assert result.status == EvalStatus.PASSED
    assert result.trajectory is not None
    # the sub-agent run is captured as a child trajectory with parent linkage + its own data
    assert len(result.trajectory.children) == 1
    child = result.trajectory.children[0]
    assert child.session is not None
    assert child.session.agent_name == "helper"
    assert child.session.parent_session_id == "parent"
    assert child.final_output == "subagent summary done"


def test_build_child_trajectories_walks_sub_agent_tree():
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)

    async def scenario():
        identity = SessionIdentity(provider_name="fake", model="fake-model")
        await store.create(
            RunRequest(
                agent_name="parent", session_id="parent", messages=[Message.text("user", "hi")]
            ),
            identity=identity,
        )
        await store.create(
            RunRequest(
                agent_name="child",
                session_id="child",
                parent_session_id="parent",
                messages=[Message.text("user", "sub")],
            ),
            identity=identity,
        )
        return await _build_child_trajectories(app, "parent", visited={"parent"})

    children = asyncio.run(scenario())
    assert len(children) == 1
    assert children[0].session is not None
    assert children[0].session.id == "child"
    assert children[0].session.parent_session_id == "parent"


def _judge_app(judge_text: str) -> CayuApp:
    # A judge runtime whose model deterministically returns `judge_text` (no live model).
    app = CayuApp(enable_logging=False)
    app.register_provider(
        ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.text_delta(judge_text),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
            ]
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="judge", model="fake-model"))
    return app


def test_llm_judge_grades_and_passes_threshold():
    judge = LLMJudge(
        _judge_app('{"score": 0.8, "rationale": "helpful and correct"}'),
        agent_name="judge",
        rubric="Score how helpful the answer is.",
        threshold=0.6,
    )
    ctx = _context(
        session=_session(),
        final_output="Paris is the capital of France.",
        transcript=(Message.text("user", "What is the capital of France?"),),
    )
    result = asyncio.run(judge.evaluate(ctx))
    assert result.score == 0.8
    assert result.passed is True  # 0.8 >= 0.6
    assert "helpful" in result.message
    # auditable metadata: judge identity + rubric + exact prompt + raw output + parsed score
    m = result.metadata
    assert m["judge_agent"] == "judge"
    assert m["judge_model"] == "fake-model"
    assert m["rubric"] == "Score how helpful the answer is."
    assert "Score how helpful the answer is." in m["prompt"]
    assert m["judge_output"]
    assert m["score"] == 0.8
    assert m["rationale"] == "helpful and correct"


def test_llm_judge_parses_markdown_fenced_json():
    # Real models (e.g. Gemini) wrap JSON in a ```json ... ``` fence; the judge must unwrap it
    # and read the clean structured score + rationale (not the raw blob via the number fallback).
    judge = LLMJudge(
        _judge_app('```json\n{"score": 0.9, "rationale": "accurate and clear"}\n```'),
        agent_name="judge",
        rubric="Score.",
        threshold=0.5,
    )
    result = asyncio.run(judge.evaluate(_context(session=_session(), final_output="x")))
    assert result.score == 0.9
    assert result.passed is True
    assert result.message == "accurate and clear"


def test_llm_judge_parses_json_with_preamble():
    # Real models add preamble/fences around the JSON; the score must still parse and the
    # rationale stay clean (not fall back to grabbing a stray number).
    judge = LLMJudge(
        _judge_app('Here is my grade:\n```json\n{"score": 0.7, "rationale": "solid"}\n```'),
        agent_name="judge",
        rubric="Score.",
        threshold=0.5,
    )
    result = asyncio.run(judge.evaluate(_context(session=_session(), final_output="x")))
    assert result.score == 0.7
    assert result.message == "solid"


def test_llm_judge_records_rubric_version():
    judge = LLMJudge(
        _judge_app('{"score": 0.9, "rationale": "ok"}'),
        agent_name="judge",
        rubric="Score.",
        threshold=0.5,
        rubric_version="v2",
    )
    result = asyncio.run(judge.evaluate(_context(session=_session(), final_output="x")))
    assert result.metadata["rubric_version"] == "v2"


def test_llm_judge_rejects_non_finite_score():
    # A NaN/Infinity score (json.loads accepts them) must fail cleanly, never clamp to 1.0.
    judge = LLMJudge(
        _judge_app('{"score": NaN, "rationale": "broken"}'),
        agent_name="judge",
        rubric="Score.",
        threshold=0.5,
    )
    result = asyncio.run(judge.evaluate(_context(session=_session(), final_output="x")))
    assert result.passed is False
    assert "parseable" in result.message


def test_llm_judge_rejects_out_of_range_json_score():
    judge = LLMJudge(
        _judge_app('{"score": 2, "rationale": "wrong scale"}'),
        agent_name="judge",
        rubric="Score.",
        threshold=0.5,
    )
    result = asyncio.run(judge.evaluate(_context(session=_session(), final_output="x")))
    assert result.passed is False
    assert "parseable" in result.message


def test_llm_judge_rejects_out_of_range_labelled_score():
    judge = LLMJudge(
        _judge_app("score: 42"),
        agent_name="judge",
        rubric="Score.",
        threshold=0.5,
    )
    result = asyncio.run(judge.evaluate(_context(session=_session(), final_output="x")))
    assert result.passed is False
    assert "parseable" in result.message


def test_llm_judge_below_threshold_keeps_score():
    judge = LLMJudge(
        _judge_app('{"score": 0.3, "rationale": "incomplete"}'),
        agent_name="judge",
        rubric="Score.",
        threshold=0.6,
    )
    result = asyncio.run(judge.evaluate(_context(session=_session(), final_output="x")))
    assert result.score == 0.3  # continuous score preserved
    assert result.passed is False  # 0.3 < 0.6


def test_llm_judge_unparseable_output_fails():
    judge = LLMJudge(
        _judge_app("it is good but I will not give a score"),
        agent_name="judge",
        rubric="Score.",
        threshold=0.5,
    )
    result = asyncio.run(judge.evaluate(_context(session=_session(), final_output="x")))
    assert result.passed is False
    assert "parseable" in result.message


def test_llm_judge_score_flows_into_case_score():
    judge = LLMJudge(
        _judge_app('{"score": 0.5, "rationale": "ok"}'),
        agent_name="judge",
        rubric="Score.",
        threshold=0.0,  # always passes; isolates the score-flow check
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(
        ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.text_delta("answer"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
            ]
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="agent", model="fake-model"))
    case = EvalCase(
        id="judged",
        request=RunRequest(agent_name="agent", messages=[Message.text("user", "go")], max_steps=1),
        assertions=[judge],
    )
    result = asyncio.run(run_eval_case(app, case, suite_id="s"))
    assert result.assertions[0].score == 0.5
    assert result.status == EvalStatus.PASSED
    assert result.score == 0.5  # the continuous judge score flows into the case score


def test_capture_probes_survives_artifact_store_error():
    # An artifact-store failure must degrade (no artifacts) rather than crash the eval case.
    from cayu.evals.models import ProbeRequirements
    from cayu.evals.runner import _capture_probes

    class _RaisingStore:
        async def list(self, *, scope=None):
            raise RuntimeError("artifact backend down")

    class _FakeApp:
        def get_environment(self, name):
            return SimpleNamespace(
                environment=SimpleNamespace(artifact_store=_RaisingStore(), workspace=None)
            )

    probes = asyncio.run(
        _capture_probes(
            _FakeApp(),
            _session(environment_name="local"),
            ProbeRequirements(artifact_scopes=frozenset({ArtifactScope.SESSION})),
        )
    )
    assert probes.artifacts_available is True
    assert probes.artifacts == ()


def test_run_then_save_reload_replay(tmp_path):
    # Full lifecycle: run -> retain the trajectory -> save JSON -> reload -> replay the same
    # assertions offline (no live app/env), incl. the workspace probe surviving the round-trip.
    (tmp_path / "README.md").write_text("Installation\n", encoding="utf-8")
    app = CayuApp(enable_logging=False)
    app.register_provider(
        ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("Installation added"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="coder", model="fake-model"))
    app.register_environment(
        Environment(EnvironmentSpec(name="local"), workspace=LocalWorkspace(tmp_path)),
        default=True,
    )
    assertions = [
        FinalOutputContains("Installation"),
        WorkspaceFileContains("README.md", "Installation"),
    ]
    case = EvalCase(
        id="replayable",
        request=RunRequest(
            agent_name="coder",
            messages=[Message.text("user", "Update README.md")],
            max_steps=1,
        ),
        assertions=assertions,
    )

    # retain_trajectory=True exposes the probe-complete trajectory; default does not.
    result = asyncio.run(run_eval_case(app, case, suite_id="s", retain_trajectory=True))
    assert result.status == EvalStatus.PASSED
    assert result.trajectory is not None
    assert asyncio.run(run_eval_case(app, case, suite_id="s")).trajectory is None

    # save -> reload -> replay against the reloaded trajectory
    path = tmp_path / "trajectory.json"
    write_trajectory_json(result.trajectory, path)
    restored = load_trajectory(path)
    replayed = asyncio.run(evaluate_assertions(restored, assertions))
    assert [r.passed for r in replayed] == [True, True]

    # the trajectory is excluded from the persisted score-first eval-run JSON
    run = EvalRun(suite_id="s", status=result.status, cases=(result,))
    assert "trajectory" not in json.loads(eval_run_to_json(run))["cases"][0]


@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set; skipping the live integration eval (credential-gated).",
)
def test_integration_eval_against_gemini(tmp_path):
    # Integration mode: run the normal eval path against a REAL provider + real workspace, and
    # assert over the runtime-native surface (session status, events, workspace file, steps, output).
    # Credential-gated (skips without GEMINI_API_KEY), like the Docker-gated Postgres suite.
    from cayu.providers import ChatCompletionsProvider

    (tmp_path / "README.md").write_text("Installation\n", encoding="utf-8")
    app = CayuApp(enable_logging=False)
    app.register_provider(
        ChatCompletionsProvider(
            name="gemini",
            api_key_env="GEMINI_API_KEY",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model=os.environ.get("CAYU_GEMINI_MODEL", "gemini-2.5-flash"))
    )
    app.register_environment(
        Environment(EnvironmentSpec(name="local"), workspace=LocalWorkspace(tmp_path)),
        default=True,
    )
    case = EvalCase(
        id="live-ack",
        request=RunRequest(
            agent_name="assistant",
            messages=[
                Message.text(
                    "user", "Reply with only the single word ACKNOWLEDGED in capital letters."
                )
            ],
            max_steps=3,
        ),
        assertions=[
            SessionCompleted(),
            EventOccurred(EventType.MODEL_COMPLETED),
            WorkspaceFileContains("README.md", "Installation"),
            MaxModelSteps(3),
            FinalOutputContains("ACKNOWLEDGED"),
        ],
    )
    result = asyncio.run(run_eval_case(app, case, suite_id="integration", retain_trajectory=True))
    assert result.status == EvalStatus.PASSED, result.error or [
        (a.name, a.passed, a.message) for a in result.assertions
    ]
    # real runtime state was captured (real usage tokens, a linked session)
    assert result.session_id is not None
    assert result.trajectory is not None
    assert result.trajectory.usage_summary is not None
    assert result.trajectory.usage_summary.usage.total_tokens > 0
