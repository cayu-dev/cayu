"""Authored workflow steps propagate routing through the real child runtime."""

from __future__ import annotations

import asyncio

import pytest
from tests.core.test_model_failover_recovery import _RecoveryProvider
from tests.core.test_model_failover_stages import _StageMemoryStore, _StageSQLiteStore

from cayu import AgentSpec, CayuApp, EventType, ModelFailoverPolicy, ModelTarget, WorkflowSpec
from cayu.runtime.retry_policy import RetryPolicy
from cayu.workflows import StepRunOptions, WorkflowBase, step


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_workflow_child_failover_and_durable_step_replay(monkeypatch, tmp_path, backend):

    class FailoverWorkflow(WorkflowBase):
        spec = WorkflowSpec(name="failover-workflow")

        async def run(self, session_id):
            ctx = self.context(session_id)
            yield await ctx.start()
            result = await step(
                ctx,
                agent="agent",
                step_id="answer",
                prompt="answer",
                run_options=StepRunOptions(
                    retry_policy=RetryPolicy(max_attempts=1),
                    failover=ModelFailoverPolicy(
                        fallbacks=(ModelTarget(provider_name="backup", model="large"),),
                        max_total_attempts=2,
                    ),
                ),
            )
            self.child_id = result.session_id
            yield await ctx.completed()

    def application(store):
        app = CayuApp(session_store=store, enable_logging=False)
        primary, backup = _RecoveryProvider("primary"), _RecoveryProvider("backup")
        app.register_provider(primary, default=True)
        app.register_provider(backup)
        app.register_agent(AgentSpec(name="agent", model="small"))
        return app, primary, backup

    async def scenario():
        path = tmp_path / "workflow.sqlite"
        store = _StageMemoryStore() if backend == "memory" else _StageSQLiteStore(path)
        try:
            app, primary, backup = application(store)
            workflow = FailoverWorkflow(app)
            events = [event async for event in workflow.run("workflow")]
            assert events[-1].type is EventType.WORKFLOW_COMPLETED
            assert len(primary.requests) == len(backup.requests) == 1
            child_id = workflow.child_id
            child = await store.load(child_id)
            assert child is not None and child.provider_name == "primary"
            assert child.parent_session_id == "workflow"
            checkpoint = await store.load_checkpoint(child_id)
            assert checkpoint is not None
            assert checkpoint["model_failover"]["candidate_index"] == 1
            assert checkpoint["model_failover"]["attempts_used"] == 2
            child_events = await store.load_events(child_id)
            assert child_events[-1].type is EventType.SESSION_COMPLETED
            if isinstance(store, _StageSQLiteStore):
                await store.close()
                store = _StageSQLiteStore(path)
            app, primary, backup = application(store)
            replayed_workflow = FailoverWorkflow(app)
            replay = [event async for event in replayed_workflow.run("workflow")]
            assert replay[-1].type is EventType.WORKFLOW_COMPLETED
            assert replayed_workflow.child_id == child_id
            assert not primary.requests and not backup.requests
            assert await store.load_events(child_id) == child_events
            assert await store.load_checkpoint(child_id) == checkpoint
        finally:
            if isinstance(store, _StageSQLiteStore):
                await store.close()

    asyncio.run(scenario())
