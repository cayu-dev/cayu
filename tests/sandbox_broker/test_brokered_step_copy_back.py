"""A brokered sandbox shared by workflow steps copies back after each step when asked."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path

from pydantic import SecretStr
from tests.core.test_human_review import identity
from tests.sandbox_broker.fake_broker import FakeBroker
from tests.sandbox_broker.test_brokered_sandbox import _factory, _project
from tests.sandbox_broker.test_brokered_shared_allocation import _call, _done, _Solve, _WritePlan

from cayu import AgentSpec, CayuApp, EventType
from cayu.environments import AllocationSharingScope, EnvironmentSpec
from cayu.evals import ScriptedModelProvider
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.storage import SQLiteSessionStore
from cayu.workflows import StepRunOptions, WorkflowBase, WorkflowSpec, step


def test_each_step_copies_back_and_the_sandbox_lives_on(tmp_path: Path) -> None:
    source = _project(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")
    seen: dict[str, object] = {}

    class PlanThenSolve(WorkflowBase):
        spec = WorkflowSpec(name="plan-then-solve")

        async def run(self, session_id: str):
            ctx = self.context(session_id)
            yield await ctx.start()
            options = StepRunOptions(
                allocation_scope=AllocationSharingScope.PARENT, copy_back="after_step"
            )
            await step(ctx, agent="planner", step_id="plan", prompt="plan", run_options=options)
            seen["live_after_plan"] = [sandbox.id for sandbox in broker.live()]
            seen["plan_after_plan"] = (source / "plan.md").read_bytes()
            await step(ctx, agent="solver", step_id="solve", prompt="solve", run_options=options)
            seen["live_after_solve"] = [sandbox.id for sandbox in broker.live()]
            seen["solution_after_solve"] = (source / "solution.py").read_bytes()
            yield await ctx.completed()

    async def scenario():
        keyring = PublicAuthorityAliasKeyring(
            active_key_id="test",
            keys={"test": SecretStr(base64.urlsafe_b64encode(b"k" * 32).decode().rstrip("="))},
        )
        store = SQLiteSessionStore(
            tmp_path / "sessions.sqlite",
            public_authority_alias_codec=PublicAuthorityAliasCodec(keyring),
        )
        app = CayuApp(
            session_store=store, enable_logging=False, public_authority_alias_keyring=keyring
        )
        app.register_provider(
            ScriptedModelProvider([_call("write_plan"), _done(), _call("solve"), _done()]),
            default=True,
        )
        app.register_environment_factory(
            EnvironmentSpec(name="sandbox", execution_profile_identity=identity("brokered")),
            _factory(broker, source),
            default=True,
        )
        app.register_agent(AgentSpec(name="planner", model="scripted"), tools=[_WritePlan()])
        app.register_agent(AgentSpec(name="solver", model="scripted"), tools=[_Solve()])
        try:
            return [event async for event in PlanThenSolve(app).run("run-1")]
        finally:
            await store.close()

    events = asyncio.run(scenario())
    assert events[-1].type == EventType.WORKFLOW_COMPLETED, events[-1].payload
    (sandbox,) = broker.sandboxes.values()
    # The plan reached the source as soon as the planner's step ended, while
    # the one sandbox stayed up for the solver; completion then destroyed it.
    assert seen["live_after_plan"] == [sandbox.id] == seen["live_after_solve"]
    assert seen["plan_after_plan"] == b"replace a - b with a + b\n"
    assert seen["solution_after_solve"] == b"def add(a, b):\n    return a + b\n"
    assert sandbox.status == "destroyed" and sandbox.destroy_reason == "requested"
    assert broker.live() == []
    assert (source / "calc.py").read_bytes() == b"def add(a, b):\n    return a - b\n"
