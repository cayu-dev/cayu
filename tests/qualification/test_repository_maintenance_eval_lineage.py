"""Characterize the public workflow boundary before wiring the coding product."""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import cast

import pytest

from cayu import (
    ChildSessionCompleted,
    EvalStatus,
    Event,
    Message,
    ModelStreamEvent,
    RunRequest,
    SQLiteSessionStore,
    WorkflowBase,
    WorkflowSpec,
    run_workflow_eval_suite,
)
from tests.evals.test_workflow_eval_target import _register_app, _suite, _target


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("linked", [True, False])
def test_independent_sdk_execution_requires_explicit_workflow_lineage(tmp_path, backend, linked):
    async def scenario():
        store = SQLiteSessionStore(tmp_path / "sessions.sqlite3") if backend == "sqlite" else None
        app = _register_app(
            [[ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()]],
            session_store=store,
        )

        class ProductShapedWorkflow(WorkflowBase):
            spec = WorkflowSpec(name="maintenance-lineage-characterization")

            async def run(self, session_id):
                ctx = self.context(session_id)
                yield await ctx.start()
                # The product runner owns app.run, rather than workflow.step.
                # This fixture proves that entrance's lineage, not product sealing.
                request = RunRequest(
                    agent_name="first",
                    session_id="coding-child",
                    messages=[Message.text("user", "repair")],
                    parent_session_id=session_id if linked else None,
                    causal_budget_id=session_id if linked else None,
                )
                async with aclosing(
                    cast("AsyncGenerator[Event, None]", self.app.run(request))
                ) as events:
                    async for _ in events:
                        pass
                yield await ctx.completed({"answer": "done"})

        try:
            result = await run_workflow_eval_suite(
                _target(app, ProductShapedWorkflow),
                _suite(ChildSessionCompleted(min_count=1)),
                retain_trajectory=True,
            )
            trial = result.cases[0].trials[0]
            assert trial.trajectory is not None
            assert trial.usage_summary is not None
            child = await app.session_store.load("coding-child")
            assert child is not None
            if linked:
                assert result.status is EvalStatus.PASSED
                assert len(trial.trajectory.children) == 1
                assert trial.usage_summary["model_steps"] == 1
                assert child.parent_session_id is not None
                root = await app.session_store.load(child.parent_session_id)
                assert root is not None
                assert child.invocation.origin == root.invocation.origin
                assert child.invocation.root_invocation_id == root.invocation.root_invocation_id
                assert child.causal_budget_id == root.id
            else:
                assert result.status is EvalStatus.FAILED
                assert trial.trajectory.children == ()
                assert trial.usage_summary["model_steps"] == 0
                assert child.parent_session_id is None
        finally:
            if store is not None:
                await store.close()

    asyncio.run(scenario())
