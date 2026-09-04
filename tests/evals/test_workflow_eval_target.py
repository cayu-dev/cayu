from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    ChildSessionCompleted,
    CorpusComparisonReason,
    EvalCase,
    EvalPlan,
    EvalStatus,
    EvalSuite,
    ExecutionProfileBehaviorIdentity,
    FinalOutputContains,
    Message,
    MessageRole,
    ModelStreamEvent,
    RunRequest,
    ScriptedModelProvider,
    SessionStore,
    SQLiteSessionStore,
    Tool,
    ToolCalled,
    ToolContext,
    ToolResult,
    ToolsCalledInOrder,
    ToolSpec,
    WorkflowBase,
    WorkflowEvalExecution,
    WorkflowEvalInstanceScope,
    WorkflowEvalResult,
    WorkflowEvalTarget,
    WorkflowSpec,
    compare_corpus_execution_results,
    corpus_execution_result_from_json,
    corpus_execution_result_to_json,
    evaluate_assertions,
    load_eval_run,
    load_trajectory,
    parallel,
    render_corpus_execution_html,
    render_html_report,
    run_workflow_eval_suite,
    step,
    workflow_eval_trial_session_id,
    write_eval_run_json,
    write_trajectory_json,
)
from cayu.cli import main
from cayu.evals.corpus import (
    CorpusUserMessageSpec,
    EvalCaseSpec,
    EvalCorpusDocument,
    EvalSuiteSpec,
    EvaluationEvidencePolicySpec,
    EvaluationSourceIdentityV1,
    FinalOutputEqualsAssertionSpec,
    ProcessEventsInOrderAssertionSpec,
    RunInputSpec,
    ToolCalledAssertionSpec,
    TrialRequestSpec,
)
from cayu.evals.execution import _run_compiled_corpus_suite, compile_corpus_suite, run_corpus_suite
from cayu.evals.result_contract import EvalTrialDiagnosticCode

_REVISION = "sha256:" + "1" * 64
_SECOND_REVISION = "sha256:" + "2" * 64


class _EchoTool(Tool):
    spec = ToolSpec(
        name="echo",
        description="Echo text.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="tests:workflow-eval-echo-tool",
            behavior_version="1",
            implementation_version="1",
        ),
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del ctx
        return ToolResult(content=args["text"])


class _TwoChildWorkflow(WorkflowBase):
    spec = WorkflowSpec(name="two-child-eval")

    async def run(self, session_id: str):
        ctx = self.context(session_id)
        yield await ctx.start()
        await step(ctx, agent="first", step_id="first", prompt="first")
        second = await step(ctx, agent="second", step_id="second", prompt="second")
        yield await ctx.completed({"answer": second.text})


class _NoChildWorkflow(WorkflowBase):
    spec = WorkflowSpec(name="no-child-eval")

    async def run(self, session_id: str):
        ctx = self.context(session_id)
        yield await ctx.start()
        yield await ctx.completed({"answer": "done"})


class _MissingCompletionWorkflow(WorkflowBase):
    spec = WorkflowSpec(name="missing-completion-eval")

    async def run(self, session_id: str):
        ctx = self.context(session_id)
        yield await ctx.start()


class _DuplicateCompletionWorkflow(WorkflowBase):
    spec = WorkflowSpec(name="duplicate-completion-eval")

    async def run(self, session_id: str):
        ctx = self.context(session_id)
        yield await ctx.start()
        yield await ctx.completed({"answer": "first"})
        yield await ctx.completed({"answer": "second"})


class _FailingWorkflow(WorkflowBase):
    spec = WorkflowSpec(name="failing-workflow-eval")

    async def run(self, session_id: str):
        ctx = self.context(session_id)
        yield await ctx.start()
        raise RuntimeError("private workflow failure")


class _OverlappingWorkflow(WorkflowBase):
    spec = WorkflowSpec(name="overlapping-workflow-eval")

    def __init__(self, app: CayuApp, *, case_id: str, state: dict) -> None:
        super().__init__(app)
        self._case_id = case_id
        self._state = state

    async def run(self, session_id: str):
        ctx = self.context(session_id)
        yield await ctx.start()
        self._state["active"] += 1
        self._state["max_active"] = max(
            self._state["max_active"],
            self._state["active"],
        )
        if self._state["active"] == 2:
            self._state["overlap"].set()
        try:
            await asyncio.wait_for(self._state["overlap"].wait(), timeout=1)
            yield await ctx.completed({"answer": self._case_id})
        finally:
            self._state["active"] -= 1


class _BlockingWorkflow(WorkflowBase):
    spec = WorkflowSpec(name="blocking-workflow-eval")

    def __init__(self, app: CayuApp, *, started: asyncio.Event) -> None:
        super().__init__(app)
        self._started = started

    async def run(self, session_id: str):
        ctx = self.context(session_id)
        yield await ctx.start()
        self._started.set()
        await asyncio.Event().wait()
        yield await ctx.completed({"answer": "unreachable"})


class _GaiaShapedWorkflow(WorkflowBase):
    spec = WorkflowSpec(name="gaia-shaped-workflow-eval")

    async def run(self, session_id: str):
        ctx = self.context(session_id)
        yield await ctx.start()
        planner = await step(ctx, agent="planner", step_id="plan", prompt="plan")
        solver = await step(
            ctx,
            agent="solver",
            step_id="solve",
            prompt=planner.text,
        )
        verifiers = await parallel(
            [
                step(
                    ctx,
                    agent="verifier-a",
                    step_id="verify-a",
                    prompt=solver.text,
                ),
                step(
                    ctx,
                    agent="verifier-b",
                    step_id="verify-b",
                    prompt=solver.text,
                ),
            ]
        )
        verifier_outputs = verifiers.outputs
        branch = await step(
            ctx,
            agent="branch",
            step_id="branch",
            prompt="\n".join(str(output) for output in verifier_outputs),
        )
        adjudicator = await step(
            ctx,
            agent="adjudicator",
            step_id="adjudicate",
            prompt=branch.text,
        )
        yield await ctx.completed({"answer": adjudicator.text})


def _register_app(
    batches: list[list[ModelStreamEvent]] | None = None,
    *,
    session_store: SessionStore | None = None,
    provider: ScriptedModelProvider | None = None,
) -> CayuApp:
    app = CayuApp(session_store=session_store, enable_logging=False)
    app.register_provider(provider or ScriptedModelProvider(batches or []), default=True)
    app.register_agent(AgentSpec(name="first", model="scripted-model"), tools=[_EchoTool()])
    app.register_agent(AgentSpec(name="second", model="scripted-model"), tools=[_EchoTool()])
    for agent_name in (
        "planner",
        "solver",
        "verifier-a",
        "verifier-b",
        "branch",
        "adjudicator",
    ):
        app.register_agent(AgentSpec(name=agent_name, model="scripted-model"))
    return app


def _target(
    app: CayuApp,
    workflow_type: type[WorkflowBase],
    *,
    factory: Callable | None = None,
    projector: Callable | None = None,
    instance_scope: WorkflowEvalInstanceScope = WorkflowEvalInstanceScope.SHARED,
    application_context: dict | None = None,
    implementation_revision: str = _REVISION,
) -> WorkflowEvalTarget:
    return WorkflowEvalTarget(
        key="workflow-target",
        app=app,
        request_base=RunRequest(agent_name="first", messages=[]),
        application_release_id="workflow-release",
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
        workflow_spec=workflow_type.spec,
        implementation_revision=implementation_revision,
        result_projector_revision=_REVISION,
        execution_scope_revision=_REVISION,
        instance_scope=instance_scope,
        application_context={} if application_context is None else application_context,
        workflow_factory=(
            factory
            if factory is not None
            else lambda invocation: WorkflowEvalExecution(
                app=app,
                workflow=workflow_type(app),
            )
        ),
        result_projector=(
            projector
            if projector is not None
            else lambda evidence: WorkflowEvalResult(
                final_output=evidence.completion_event.payload["answer"],
                structured_output={"answer": evidence.completion_event.payload["answer"]},
            )
        ),
    )


def _suite(*assertions) -> EvalSuite:
    return EvalSuite(
        id="workflow-suite",
        cases=[
            EvalCase(
                id="workflow-case",
                request=RunRequest(
                    agent_name="first",
                    messages=[Message.text(MessageRole.USER, "run the workflow")],
                ),
                assertions=list(assertions),
            )
        ],
    )


def _corpus(*assertions) -> EvalCorpusDocument:
    policy = EvaluationEvidencePolicySpec.standard()
    suite = EvalSuiteSpec.create(
        id="workflow-suite",
        name="Workflow suite",
        trial_request=TrialRequestSpec(trials=1, timeout_seconds=30),
    )
    case = EvalCaseSpec.create(
        id="workflow-case",
        suite_id=suite.id,
        name="Workflow case",
        source=EvaluationSourceIdentityV1(
            application_release_id="captured-release",
            app_manifest_schema_version="7",
            app_manifest_fingerprint="a" * 64,
            evidence_revision=_REVISION,
        ),
        input=RunInputSpec(messages=(CorpusUserMessageSpec(text="run the workflow"),)),
        assertions=assertions,
    )
    return EvalCorpusDocument.create(
        target_key="workflow-target",
        evidence_policy=policy,
        suites=(suite,),
        cases=(case,),
    )


def build_cli_app() -> CayuApp:
    return _register_app()


def build_cli_workflow_eval_plan() -> EvalPlan:
    app = _register_app()
    return EvalPlan(
        workflow_target=_target(app, _NoChildWorkflow),
        suite=_suite(FinalOutputContains("done")),
    )


def test_workflow_eval_retains_root_children_tools_usage_and_typed_output(tmp_path) -> None:
    app = _register_app(
        [
            [
                ModelStreamEvent.tool_call(
                    id="first-call",
                    name="echo",
                    arguments={"text": "first"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("first done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="second-call",
                    name="echo",
                    arguments={"text": "second"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("second done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    result = asyncio.run(
        run_workflow_eval_suite(
            _target(app, _TwoChildWorkflow),
            _suite(
                ChildSessionCompleted(min_count=2),
                ToolCalled("echo"),
                ToolsCalledInOrder(["echo", "echo"]),
                FinalOutputContains("second done"),
            ),
            retain_trajectory=True,
        )
    )

    trial = result.cases[0].trials[0]
    assert result.status is EvalStatus.PASSED
    assert trial.status is EvalStatus.PASSED
    assert trial.final_output == "second done"
    assert trial.structured_output == {"answer": "second done"}
    assert trial.trajectory is not None
    assert trial.trajectory.workflow_output is not None
    assert trial.trajectory.workflow_output.structured_output == {"answer": "second done"}
    assert trial.trajectory.transcript == (Message.text(MessageRole.USER, "run the workflow"),)
    assert trial.trajectory.workflow_output.input_message_count == 1
    assert all(child.session is not None for child in trial.trajectory.children)
    assert [
        child.session.agent_name for child in trial.trajectory.children if child.session is not None
    ] == [
        "first",
        "second",
    ]
    assert trial.usage_summary is not None
    assert trial.usage_summary["model_steps"] == 4
    assert trial.usage_summary["tool_calls"] == 2
    assert trial.events_count == sum(
        len(node.events) for node in (trial.trajectory, *trial.trajectory.children)
    )
    tampered = trial.trajectory.model_copy(update={"final_output": "tampered"})
    with pytest.raises(ValueError, match="bound evidence"):
        asyncio.run(evaluate_assertions(tampered, [FinalOutputContains("tampered")]))
    tampered_input = trial.trajectory.model_copy(update={"transcript": ()})
    with pytest.raises(ValueError, match="input count"):
        asyncio.run(evaluate_assertions(tampered_input, [FinalOutputContains("second done")]))
    result_path = tmp_path / "workflow-run.json"
    write_eval_run_json(result, result_path)
    loaded = load_eval_run(result_path)
    assert loaded.cases[0].trials[0].structured_output == {"answer": "second done"}
    assert "second done" in render_html_report(loaded)
    trajectory_path = tmp_path / "workflow-trajectory.json"
    write_trajectory_json(trial.trajectory, trajectory_path)
    loaded_trajectory = load_trajectory(trajectory_path)
    assert loaded_trajectory.workflow_output == trial.trajectory.workflow_output


def test_gaia_shaped_parallel_workflow_retains_complete_bounded_child_evidence() -> None:
    app = _register_app(
        [
            [
                ModelStreamEvent.text_delta(output),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
            for output in (
                "plan",
                "solution",
                "verified-a",
                "verified-b",
                "branched",
                "final answer",
            )
        ]
    )
    result = asyncio.run(
        run_workflow_eval_suite(
            _target(app, _GaiaShapedWorkflow),
            _suite(
                ChildSessionCompleted(min_count=6),
                FinalOutputContains("final answer"),
            ),
            retain_trajectory=True,
        )
    )

    trial = result.cases[0].trials[0]
    assert result.status is EvalStatus.PASSED
    assert trial.trajectory is not None
    assert trial.usage_summary is not None
    assert trial.usage_summary["model_steps"] == 6
    assert [
        child.session.agent_name for child in trial.trajectory.children if child.session is not None
    ] == [
        "planner",
        "solver",
        "verifier-a",
        "verifier-b",
        "branch",
        "adjudicator",
    ]


def test_workflow_eval_runs_through_portable_corpus_and_publishes_identity() -> None:
    app = _register_app(
        [
            [
                ModelStreamEvent.tool_call(
                    id=f"{agent}-call",
                    name="echo",
                    arguments={"text": agent},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
            if phase == "tool"
            else [
                ModelStreamEvent.text_delta(f"{agent} done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
            for agent in ("first", "second")
            for phase in ("tool", "output")
        ]
    )
    target = _target(app, _TwoChildWorkflow)
    result = asyncio.run(
        run_corpus_suite(
            target,
            _corpus(
                FinalOutputEqualsAssertionSpec(id="output", expected="second done"),
                ToolCalledAssertionSpec(id="tools", tool_name="echo", min_count=2),
                ProcessEventsInOrderAssertionSpec(
                    id="process",
                    events=(
                        "session_started",
                        "session_completed",
                        "session_started",
                        "session_completed",
                    ),
                ),
            ),
            "workflow-suite",
            max_concurrency=1,
        )
    )

    assert result.run.status == "passed"
    assert result.run.cases[0].trials[0].code is EvalTrialDiagnosticCode.PASSED
    assert result.target.workflow == target.identity()
    assert result.target.external_process is None
    loaded = corpus_execution_result_from_json(corpus_execution_result_to_json(result))
    assert loaded.target.workflow == target.identity()
    assert "workflow-target" in render_corpus_execution_html(loaded)


def test_workflow_target_revision_change_makes_result_comparison_incompatible() -> None:
    app = _register_app()
    corpus = _corpus(FinalOutputEqualsAssertionSpec(id="output", expected="done"))
    baseline = asyncio.run(
        run_corpus_suite(
            _target(app, _NoChildWorkflow),
            corpus,
            "workflow-suite",
            max_concurrency=1,
        )
    )
    current = asyncio.run(
        run_corpus_suite(
            _target(
                app,
                _NoChildWorkflow,
                implementation_revision=_SECOND_REVISION,
            ),
            corpus,
            "workflow-suite",
            max_concurrency=1,
        )
    )

    comparison = compare_corpus_execution_results(baseline, current)
    assert comparison.compatibility.reasons == (
        CorpusComparisonReason.EXTERNAL_TARGET_REVISION_MISMATCH,
    )


def test_workflow_eval_runs_against_sqlite_session_store(tmp_path) -> None:
    async def run() -> None:
        store = SQLiteSessionStore(tmp_path / "workflow-eval.sqlite")
        app = _register_app(session_store=store)
        try:
            result = await run_workflow_eval_suite(
                _target(app, _NoChildWorkflow),
                _suite(FinalOutputContains("done")),
                retain_trajectory=True,
            )
        finally:
            await store.close()

        trial = result.cases[0].trials[0]
        assert result.status is EvalStatus.PASSED
        assert trial.trajectory is not None
        assert trial.trajectory.workflow_output is not None
        assert trial.trajectory.workflow_output.workflow_name == _NoChildWorkflow.spec.name

    asyncio.run(run())


def test_workflow_eval_runs_against_postgres_session_store(postgres_dsn) -> None:
    from tests.core.postgres_contention_support import drop_cayu_tables

    from cayu import PostgresSessionStore
    from cayu.storage.migrations import SchemaMode

    async def run() -> None:
        await drop_cayu_tables(postgres_dsn)
        store = PostgresSessionStore(
            postgres_dsn,
            min_size=1,
            max_size=4,
            schema_mode=SchemaMode.CREATE,
        )
        app = _register_app(session_store=store)
        try:
            result = await run_workflow_eval_suite(
                _target(app, _NoChildWorkflow),
                _suite(FinalOutputContains("done")),
                retain_trajectory=True,
            )
        finally:
            await store.close()

        trial = result.cases[0].trials[0]
        assert result.status is EvalStatus.PASSED
        assert trial.trajectory is not None
        assert trial.trajectory.workflow_output is not None

    asyncio.run(run())


def test_workflow_eval_recovery_replays_completed_children_without_provider_dispatch() -> None:
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="first-call",
                    name="echo",
                    arguments={"text": "first"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("first done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="second-call",
                    name="echo",
                    arguments={"text": "second"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("second done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = _register_app(provider=provider)
    target = _target(app, _TwoChildWorkflow)
    compiled = compile_corpus_suite(
        _corpus(FinalOutputEqualsAssertionSpec(id="output", expected="second done")),
        target,
        "workflow-suite",
    )
    checkpoints = []

    async def lose_acknowledgement(case_id, result, public_data) -> None:
        del case_id, public_data
        checkpoints.append(result)
        raise RuntimeError("simulated lost result acknowledgement")

    with pytest.raises(RuntimeError, match="lost result acknowledgement"):
        asyncio.run(
            _run_compiled_corpus_suite(
                target,
                compiled,
                max_concurrency=1,
                native_run_id="workflow-recovery-run",
                trial_completed=lose_acknowledgement,
            )
        )
    assert len(provider.requests) == 4
    assert checkpoints[0].trajectory is None
    assert checkpoints[0].final_output == ""
    assert checkpoints[0].structured_output is None

    resumed = asyncio.run(
        _run_compiled_corpus_suite(
            target,
            compiled,
            max_concurrency=1,
            native_run_id="workflow-recovery-run",
        )
    )

    assert resumed.run.status == "passed"
    assert len(provider.requests) == 4


def test_workflow_eval_recovery_after_completion_before_projection_reuses_children() -> None:
    class SimulatedProcessLoss(BaseException):
        pass

    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.text_delta(output),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
            for output in ("first done", "second done")
        ]
    )
    app = _register_app(provider=provider)
    projection_attempts = 0

    def projector(evidence):
        nonlocal projection_attempts
        projection_attempts += 1
        if projection_attempts == 1:
            raise SimulatedProcessLoss
        return WorkflowEvalResult(final_output=evidence.completion_event.payload["answer"])

    target = _target(app, _TwoChildWorkflow, projector=projector)
    compiled = compile_corpus_suite(
        _corpus(FinalOutputEqualsAssertionSpec(id="output", expected="second done")),
        target,
        "workflow-suite",
    )
    with pytest.raises(SimulatedProcessLoss):
        asyncio.run(
            _run_compiled_corpus_suite(
                target,
                compiled,
                max_concurrency=1,
                native_run_id="workflow-projector-crash",
            )
        )
    assert len(provider.requests) == 2

    resumed = asyncio.run(
        _run_compiled_corpus_suite(
            target,
            compiled,
            max_concurrency=1,
            native_run_id="workflow-projector-crash",
        )
    )

    assert resumed.run.status == "passed"
    assert projection_attempts == 2
    assert len(provider.requests) == 2


def test_workflow_eval_missing_completion_fails_closed_with_stable_diagnostic() -> None:
    app = _register_app()
    target = _target(
        app,
        _MissingCompletionWorkflow,
        projector=lambda evidence: pytest.fail("projector must not run without completion"),
    )
    result = asyncio.run(
        run_corpus_suite(
            target,
            _corpus(FinalOutputEqualsAssertionSpec(id="output", expected="done")),
            "workflow-suite",
            max_concurrency=1,
        )
    )

    trial = result.run.cases[0].trials[0]
    assert result.run.status == "error"
    assert trial.code is EvalTrialDiagnosticCode.WORKFLOW_COMPLETION_MISSING
    assert trial.output.evidence_state == "unavailable"


def test_workflow_eval_duplicate_completion_fails_closed_with_stable_diagnostic() -> None:
    app = _register_app()
    result = asyncio.run(
        run_corpus_suite(
            _target(app, _DuplicateCompletionWorkflow),
            _corpus(FinalOutputEqualsAssertionSpec(id="output", expected="done")),
            "workflow-suite",
            max_concurrency=1,
        )
    )

    trial = result.run.cases[0].trials[0]
    assert trial.code is EvalTrialDiagnosticCode.WORKFLOW_COMPLETION_CONFLICT
    assert trial.output.evidence_state == "unavailable"


def test_workflow_execution_failure_is_distinct_from_target_construction_failure() -> None:
    app = _register_app()
    result = asyncio.run(
        run_corpus_suite(
            _target(app, _FailingWorkflow),
            _corpus(FinalOutputEqualsAssertionSpec(id="output", expected="done")),
            "workflow-suite",
            max_concurrency=1,
        )
    )

    trial = result.run.cases[0].trials[0]
    assert trial.code is EvalTrialDiagnosticCode.WORKFLOW_EXECUTION_FAILED
    assert "private workflow failure" not in trial.message


def test_workflow_eval_rejects_attempt_superseded_during_projection() -> None:
    app = _register_app()
    workflow = _NoChildWorkflow(app)

    async def superseding_projector(evidence):
        async for _event in workflow.run(evidence.workflow_run_id):
            pass
        return WorkflowEvalResult(final_output="stale")

    result = asyncio.run(
        run_corpus_suite(
            _target(
                app,
                _NoChildWorkflow,
                factory=lambda invocation: WorkflowEvalExecution(app=app, workflow=workflow),
                projector=superseding_projector,
            ),
            _corpus(FinalOutputEqualsAssertionSpec(id="output", expected="done")),
            "workflow-suite",
            max_concurrency=1,
        )
    )

    trial = result.run.cases[0].trials[0]
    assert trial.code is EvalTrialDiagnosticCode.WORKFLOW_ATTEMPT_SUPERSEDED
    assert trial.output.evidence_state == "unavailable"


def test_workflow_eval_projector_failure_has_distinct_diagnostic() -> None:
    app = _register_app()

    def fail_projector(evidence):
        del evidence
        raise RuntimeError("private projector failure")

    result = asyncio.run(
        run_corpus_suite(
            _target(app, _NoChildWorkflow, projector=fail_projector),
            _corpus(FinalOutputEqualsAssertionSpec(id="output", expected="done")),
            "workflow-suite",
            max_concurrency=1,
        )
    )

    trial = result.run.cases[0].trials[0]
    assert trial.code is EvalTrialDiagnosticCode.WORKFLOW_PROJECTOR_FAILED
    assert "private projector failure" not in trial.message


def test_workflow_eval_rejects_untyped_projector_output() -> None:
    app = _register_app()
    result = asyncio.run(
        run_workflow_eval_suite(
            _target(
                app,
                _NoChildWorkflow,
                projector=lambda evidence: {"final_output": "not typed"},
            ),
            _suite(FinalOutputContains("done")),
        )
    )

    trial = result.cases[0].trials[0]
    assert trial.status is EvalStatus.ERROR
    assert trial.error == "Workflow result projector returned an invalid result."


def test_workflow_eval_requires_quiescence_before_publishing_output() -> None:
    app = _register_app()

    async def close() -> None:
        raise RuntimeError("private close detail")

    target = _target(
        app,
        _NoChildWorkflow,
        factory=lambda invocation: WorkflowEvalExecution(
            app=app,
            workflow=_NoChildWorkflow(app),
            close=close,
        ),
    )
    result = asyncio.run(
        run_corpus_suite(
            target,
            _corpus(FinalOutputEqualsAssertionSpec(id="output", expected="done")),
            "workflow-suite",
            max_concurrency=1,
        )
    )

    trial = result.run.cases[0].trials[0]
    assert trial.code is EvalTrialDiagnosticCode.WORKFLOW_QUIESCENCE_FAILED
    assert trial.output.evidence_state == "unavailable"
    assert "private close detail" not in trial.message


def test_workflow_eval_rejects_evidence_that_changes_during_quiescence() -> None:
    class LateChildWorkflow(WorkflowBase):
        spec = WorkflowSpec(name="late-child-workflow-eval")

        def __init__(self, app: CayuApp) -> None:
            super().__init__(app)
            self.release = asyncio.Event()
            self.child_task: asyncio.Task | None = None

        async def run_late_child(self, context) -> None:
            await self.release.wait()
            await step(
                context,
                agent="first",
                step_id="late-child",
                prompt="late child",
            )

        async def run(self, session_id: str):
            context = self.context(session_id)
            yield await context.start()
            self.child_task = asyncio.create_task(self.run_late_child(context))
            yield await context.completed({"answer": "done"})

        async def close(self) -> None:
            self.release.set()
            assert self.child_task is not None
            await self.child_task

    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.text_delta("late child done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ]
    )
    app = _register_app(provider=provider)
    workflow = LateChildWorkflow(app)
    target = _target(
        app,
        LateChildWorkflow,
        factory=lambda invocation: WorkflowEvalExecution(
            app=app,
            workflow=workflow,
            close=workflow.close,
        ),
    )

    result = asyncio.run(
        run_corpus_suite(
            target,
            _corpus(FinalOutputEqualsAssertionSpec(id="output", expected="done")),
            "workflow-suite",
            max_concurrency=1,
        )
    )

    trial = result.run.cases[0].trials[0]
    assert len(provider.requests) == 1
    assert result.run.status == "error"
    assert trial.code is EvalTrialDiagnosticCode.WORKFLOW_COMPLETION_CONFLICT
    assert trial.output.evidence_state == "unavailable"


def test_workflow_eval_rejects_factory_runtime_profile_mismatch() -> None:
    class VersionedScriptedProvider(ScriptedModelProvider):
        def __init__(self, events, *, behavior_version: str) -> None:
            super().__init__(events)
            self.behavior_version = behavior_version

        @property
        def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
            return ExecutionProfileBehaviorIdentity(
                name="tests:workflow-eval-runtime-provider",
                behavior_version=self.behavior_version,
                implementation_version="1",
            )

    class ProfiledWorkflow(WorkflowBase):
        spec = WorkflowSpec(name="profiled-workflow-eval")

        async def run(self, session_id: str):
            context = self.context(session_id)
            yield await context.start()
            child = await step(
                context,
                agent="first",
                step_id="profiled-child",
                prompt="run",
            )
            yield await context.completed({"answer": child.text})

    declared_provider = VersionedScriptedProvider([], behavior_version="declared-v1")
    runtime_provider = VersionedScriptedProvider(
        [
            [
                ModelStreamEvent.text_delta("runtime-v2"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ],
        behavior_version="runtime-v2",
    )
    declared_app = _register_app(provider=declared_provider)
    runtime_app = _register_app(provider=runtime_provider)
    assert declared_app.describe().fingerprint == runtime_app.describe().fingerprint
    target = _target(
        declared_app,
        ProfiledWorkflow,
        factory=lambda invocation: WorkflowEvalExecution(
            app=runtime_app,
            workflow=ProfiledWorkflow(runtime_app),
        ),
        instance_scope=WorkflowEvalInstanceScope.PER_TRIAL,
    )

    result = asyncio.run(
        run_corpus_suite(
            target,
            _corpus(FinalOutputEqualsAssertionSpec(id="output", expected="runtime-v2")),
            "workflow-suite",
            max_concurrency=1,
        )
    )

    trial = result.run.cases[0].trials[0]
    assert runtime_provider.requests == []
    assert result.run.status == "error"
    assert trial.code is EvalTrialDiagnosticCode.WORKFLOW_TARGET_FAILED
    assert trial.output.evidence_state == "unavailable"


def test_workflow_eval_timeout_closes_owned_execution_before_returning() -> None:
    async def run() -> None:
        app = _register_app()
        started = asyncio.Event()
        closed = asyncio.Event()

        async def close() -> None:
            closed.set()

        target = _target(
            app,
            _BlockingWorkflow,
            factory=lambda invocation: WorkflowEvalExecution(
                app=app,
                workflow=_BlockingWorkflow(app, started=started),
                close=close,
            ),
        )
        result = await run_workflow_eval_suite(
            target,
            _suite(FinalOutputContains("unreachable")),
            case_timeout_seconds=0.01,
        )

        trial = result.cases[0].trials[0]
        assert started.is_set()
        assert closed.is_set()
        assert trial.status is EvalStatus.ERROR
        assert "timed out" in (trial.error or "")
        assert trial.final_output == ""

    asyncio.run(run())


def test_cancelling_workflow_eval_closes_owned_execution() -> None:
    async def run() -> None:
        app = _register_app()
        started = asyncio.Event()
        closed = asyncio.Event()

        async def close() -> None:
            closed.set()

        target = _target(
            app,
            _BlockingWorkflow,
            factory=lambda invocation: WorkflowEvalExecution(
                app=app,
                workflow=_BlockingWorkflow(app, started=started),
                close=close,
            ),
        )
        task = asyncio.create_task(
            run_workflow_eval_suite(
                target,
                _suite(FinalOutputContains("unreachable")),
            )
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed.is_set()

    asyncio.run(run())


def test_cancelling_result_projection_closes_owned_execution_without_output() -> None:
    async def run() -> None:
        app = _register_app()
        projection_started = asyncio.Event()
        closed = asyncio.Event()

        async def projector(evidence):
            del evidence
            projection_started.set()
            await asyncio.Event().wait()
            return WorkflowEvalResult(final_output="unreachable")

        async def close() -> None:
            closed.set()

        target = _target(
            app,
            _NoChildWorkflow,
            factory=lambda invocation: WorkflowEvalExecution(
                app=app,
                workflow=_NoChildWorkflow(app),
                close=close,
            ),
            projector=projector,
        )
        task = asyncio.create_task(
            run_workflow_eval_suite(
                target,
                _suite(FinalOutputContains("unreachable")),
            )
        )
        await asyncio.wait_for(projection_started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed.is_set()

    asyncio.run(run())


def test_per_trial_factory_isolates_concurrent_workflow_instances_and_closes_them() -> None:
    seed_app = _register_app()
    invocations = []
    closed: list[str] = []
    apps: list[CayuApp] = []
    state = {
        "active": 0,
        "max_active": 0,
        "overlap": asyncio.Event(),
    }

    def factory(invocation):
        invocations.append(invocation)
        app = _register_app()
        apps.append(app)

        async def close() -> None:
            closed.append(invocation.workflow_run_id)

        return WorkflowEvalExecution(
            app=app,
            workflow=_OverlappingWorkflow(
                app,
                case_id=invocation.case_id,
                state=state,
            ),
            close=close,
        )

    target = _target(
        seed_app,
        _OverlappingWorkflow,
        factory=factory,
        instance_scope=WorkflowEvalInstanceScope.PER_TRIAL,
    )
    suite = EvalSuite(
        id="workflow-suite",
        cases=[
            EvalCase(
                id=case_id,
                request=RunRequest(
                    agent_name="first",
                    messages=[Message.text(MessageRole.USER, case_id)],
                ),
                assertions=[FinalOutputContains(case_id)],
            )
            for case_id in ("first-case", "second-case")
        ],
    )
    result = asyncio.run(
        run_workflow_eval_suite(
            target,
            suite,
            max_concurrency=2,
            retain_trajectory=True,
        )
    )

    assert result.status is EvalStatus.PASSED
    assert [case.case_id for case in result.cases] == ["first-case", "second-case"]
    assert state["max_active"] == 2
    assert len({id(app.session_store) for app in apps}) == 2
    assert len({invocation.workflow_run_id for invocation in invocations}) == 2
    assert sorted(closed) == sorted(invocation.workflow_run_id for invocation in invocations)
    assert all(len(invocation.messages) == 1 for invocation in invocations)
    for case in result.cases:
        trial = case.trials[0]
        assert trial.final_output == case.case_id
        assert trial.trajectory is not None
        assert trial.trajectory.transcript == (Message.text(MessageRole.USER, case.case_id),)


def test_shared_workflow_target_rejects_concurrent_execution() -> None:
    app = _register_app()
    with pytest.raises(ValueError, match="shared workflow target"):
        asyncio.run(
            run_workflow_eval_suite(
                _target(app, _NoChildWorkflow),
                _suite(FinalOutputContains("done")),
                max_concurrency=2,
            )
        )


def test_workflow_trial_session_identity_is_recovery_stable() -> None:
    arguments = {
        "target_revision": _REVISION,
        "run_id": "run",
        "suite_id": "suite",
        "case_id": "case",
        "trial_number": 1,
    }
    first = workflow_eval_trial_session_id(**arguments)
    assert first == workflow_eval_trial_session_id(**arguments)
    assert first != workflow_eval_trial_session_id(
        target_revision=_REVISION,
        run_id="run",
        suite_id="suite",
        case_id="case",
        trial_number=2,
    )


def test_workflow_target_identity_binds_application_context_and_omits_callbacks() -> None:
    app = _register_app()
    first = _target(app, _NoChildWorkflow, application_context={"tenant": "first"})
    second = _target(app, _NoChildWorkflow, application_context={"tenant": "second"})

    assert first.identity().revision != second.identity().revision
    dumped = first.model_dump(mode="python")
    assert "workflow_factory" not in dumped
    assert "result_projector" not in dumped
    assert "application_context" not in dumped


def test_cli_and_server_registry_preserve_workflow_target_authority() -> None:
    from cayu.cli.evals import _coerce_plan
    from cayu.server.evals_registry import explicit_eval_target_registry

    app = _register_app()
    target = _target(app, _NoChildWorkflow)
    plan = _coerce_plan(target)
    registry = explicit_eval_target_registry(target)
    registration = registry.registration(target.key)

    assert plan.workflow_target is target
    assert registration is not None
    assert type(registration.target) is WorkflowEvalTarget
    execution_target = registration.execution_target()
    assert type(execution_target) is WorkflowEvalTarget
    assert execution_target.identity() == target.identity()


def test_cli_project_eval_target_runs_workflow_without_a_custom_command(
    tmp_path,
    monkeypatch,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[tool.cayu]\n"
        f'factory = "{__name__}:build_cli_app"\n'
        f'eval_target = "{__name__}:build_cli_workflow_eval_plan"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert main(["eval", "run", "--output", "workflow-run.json"]) == 0

    loaded = load_eval_run(tmp_path / "workflow-run.json")
    assert loaded.status is EvalStatus.PASSED
    assert loaded.cases[0].trials[0].final_output == "done"


def test_durable_server_worker_executes_registered_workflow_target(tmp_path) -> None:
    from cayu.evals.store import EvalRunInvocation, EvalRunRequest, EvalRunStatus
    from cayu.server import EvalsConfig
    from cayu.server.evals_registry import (
        explicit_eval_target_registry,
        target_for_eval_invocation,
    )
    from cayu.server.evals_worker import EvalRunCoordinator
    from cayu.storage.evals_sqlite import SQLiteEvalStore

    async def run() -> None:
        session_store = SQLiteSessionStore(tmp_path / "workflow-sessions.sqlite")
        eval_store = SQLiteEvalStore(tmp_path / "workflow-evals.sqlite")
        app = _register_app(session_store=session_store)
        target = _target(app, _NoChildWorkflow)
        corpus = _corpus(FinalOutputEqualsAssertionSpec(id="output", expected="done"))
        registry = explicit_eval_target_registry(target)
        invocation = EvalRunInvocation()
        effective_target = target_for_eval_invocation(target, invocation)
        prepared = await registry.prepare_execution_profile(
            target.key,
            effective_target=effective_target,
        )
        invocation = invocation.model_copy(
            update={
                "execution_profile": prepared.binding,
                "execution_profile_snapshot": prepared.snapshot,
            },
            deep=True,
        )
        request = EvalRunRequest(
            run_id="workflow-server-run",
            idempotency_key="sha256:" + "3" * 64,
            corpus_revision=corpus.revision,
            target_key=target.key,
            suite_id=corpus.suites[0].id,
            suite_revision=corpus.suites[0].revision,
            max_concurrency=1,
            invocation=invocation,
        )
        coordinator = EvalRunCoordinator(
            EvalsConfig(
                target=target,
                store=eval_store,
                poll_interval_seconds=0.01,
                lease_seconds=5,
            )
        )
        try:
            await eval_store.save_corpus(corpus, redact_json=target.app.redact_json)
            await eval_store.admit_run(request, redact_json=target.app.redact_json)
            coordinator.start()
            deadline = asyncio.get_running_loop().time() + 5
            while asyncio.get_running_loop().time() < deadline:
                record = await eval_store.load_run(request.run_id)
                assert record is not None
                if record.status in {
                    EvalRunStatus.COMPLETED,
                    EvalRunStatus.FAILED,
                    EvalRunStatus.CANCELLED,
                }:
                    break
                await asyncio.sleep(0.01)
            assert record.status is EvalRunStatus.COMPLETED
            result = await eval_store.load_result(request.run_id)
            assert result is not None
            assert result.target.workflow == target.identity()
            assert result.run.status == "passed"
        finally:
            await coordinator.stop()
            await eval_store.close()
            await session_store.close()

    asyncio.run(run())
