"""Exercise the public workflow-root Evals surface from an installed wheel."""

from __future__ import annotations

import asyncio

from cayu import (
    AgentSpec,
    CayuApp,
    EvalCase,
    EvalPlan,
    EvalRun,
    EvalStatus,
    EvalSuite,
    FinalOutputContains,
    Message,
    MessageRole,
    RunRequest,
    ScriptedModelProvider,
    WorkflowBase,
    WorkflowEvalExecution,
    WorkflowEvalInstanceScope,
    WorkflowEvalResult,
    WorkflowEvalTarget,
    WorkflowSpec,
    run_eval_plan,
)

_REVISION = "sha256:" + "1" * 64


class _InstalledWorkflow(WorkflowBase):
    spec = WorkflowSpec(name="installed-wheel-workflow-eval")

    async def run(self, session_id: str):
        context = self.context(session_id)
        yield await context.start()
        yield await context.completed({"answer": "installed workflow passed"})


async def _run() -> None:
    app = CayuApp(enable_logging=False)
    app.register_provider(ScriptedModelProvider([]), default=True)
    app.register_agent(AgentSpec(name="entry", model="scripted-model"))
    workflow = _InstalledWorkflow(app)
    target = WorkflowEvalTarget(
        key="installed-workflow-target",
        app=app,
        request_base=RunRequest(agent_name="entry", messages=[]),
        application_release_id="installed-wheel-smoke",
        workflow_spec=workflow.spec,
        implementation_revision=_REVISION,
        result_projector_revision=_REVISION,
        execution_scope_revision=_REVISION,
        instance_scope=WorkflowEvalInstanceScope.SHARED,
        workflow_factory=lambda invocation: WorkflowEvalExecution(app=app, workflow=workflow),
        result_projector=lambda evidence: WorkflowEvalResult(
            final_output=evidence.completion_event.payload["answer"],
            structured_output={"answer": evidence.completion_event.payload["answer"]},
        ),
    )
    plan = EvalPlan(
        workflow_target=target,
        suite=EvalSuite(
            id="installed-workflow-suite",
            cases=[
                EvalCase(
                    id="installed-workflow-case",
                    request=RunRequest(
                        agent_name="entry",
                        messages=[Message.text(MessageRole.USER, "run")],
                    ),
                    assertions=[FinalOutputContains("installed workflow passed")],
                )
            ],
        ),
    )

    result = await run_eval_plan(plan, retain_trajectory=True)
    assert type(result) is EvalRun
    assert result.status is EvalStatus.PASSED
    trial = result.cases[0].trials[0]
    assert trial.structured_output == {"answer": "installed workflow passed"}
    assert trial.trajectory is not None
    assert trial.trajectory.workflow_output is not None


def main() -> int:
    asyncio.run(_run())
    print("built-wheel workflow eval smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
