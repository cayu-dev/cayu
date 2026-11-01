"""Workflow steps sharing one brokered sandbox, against a fake broker."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path

from pydantic import SecretStr
from tests.core.test_human_review import identity
from tests.sandbox_broker.fake_broker import FakeBroker
from tests.sandbox_broker.test_brokered_sandbox import _factory, _project

from cayu import AgentSpec, CayuApp, EventType
from cayu.environments import EnvironmentSpec
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.storage import SQLiteSessionStore
from cayu.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from cayu.workflows import StepRunOptions, WorkflowBase, WorkflowSpec, step


class _WritePlan(Tool):
    spec = ToolSpec(name="write_plan", parallel_safe=False, workspace_mutation=True)

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        assert ctx.workspace is not None
        await ctx.workspace.create_bytes("plan.md", b"replace a - b with a + b\n")
        return ToolResult(content="planned")


class _Solve(Tool):
    spec = ToolSpec(name="solve", parallel_safe=False, workspace_mutation=True)

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        assert ctx.workspace is not None
        plan = (await ctx.workspace.read_bytes("plan.md")).content
        assert plan.startswith(b"replace a - b")
        source = (await ctx.workspace.read_bytes("calc.py")).content
        await ctx.workspace.create_bytes("solution.py", source.replace(b"a - b", b"a + b"))
        return ToolResult(content="solved")


def _call(tool: str) -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.tool_call(id=f"call-{tool}", name=tool, arguments={}),
        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
    ]


def _done() -> list[ModelStreamEvent]:
    return [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed({})]


def test_workflow_steps_hand_files_over_in_one_brokered_sandbox(tmp_path: Path) -> None:
    source = _project(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")
    seen: dict[str, object] = {}

    class PlanThenSolve(WorkflowBase):
        spec = WorkflowSpec(name="plan-then-solve")

        async def run(self, session_id: str):
            ctx = self.context(session_id)
            yield await ctx.start()
            options = StepRunOptions(allocation_scope="parent")
            await step(ctx, agent="planner", step_id="plan", prompt="plan", run_options=options)
            # The planner's session ended, but its sandbox lives on for the
            # solver and nothing has been copied back yet.
            seen["live_after_plan"] = [sandbox.id for sandbox in broker.live()]
            seen["plan_in_source_after_plan"] = (source / "plan.md").exists()
            await step(ctx, agent="solver", step_id="solve", prompt="solve", run_options=options)
            seen["live_after_solve"] = [sandbox.id for sandbox in broker.live()]
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
    # One sandbox served both steps and was destroyed when the workflow completed.
    (sandbox,) = broker.sandboxes.values()
    assert seen["live_after_plan"] == [sandbox.id] == seen["live_after_solve"]
    assert seen["plan_in_source_after_plan"] is False
    assert sandbox.status == "destroyed" and sandbox.destroy_reason == "requested"
    assert broker.live() == []
    # Copy-back happened once, at the end, with both steps' files.
    assert (source / "plan.md").read_bytes() == b"replace a - b with a + b\n"
    assert (source / "solution.py").read_bytes() == b"def add(a, b):\n    return a + b\n"
    assert (source / "calc.py").read_bytes() == b"def add(a, b):\n    return a - b\n"
