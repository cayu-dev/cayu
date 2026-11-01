"""Workflow steps sharing one brokered sandbox, against a fake broker."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path

import pytest
from pydantic import SecretStr
from tests.core.test_human_review import identity
from tests.sandbox_broker.fake_broker import FakeBroker
from tests.sandbox_broker.test_brokered_sandbox import _factory, _project

from cayu import AgentSpec, CayuApp, EventType
from cayu.environments import EnvironmentSpec
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.sessions.records import SessionStatus
from cayu.sessions.recovery import (
    IncompleteSessionRecoveryAction,
    IncompleteSessionsRecoveryRequest,
)
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


class _SolveWhatIsThere(Tool):
    spec = ToolSpec(name="solve", parallel_safe=False, workspace_mutation=True)

    def __init__(self, seen: dict[str, object]) -> None:
        self._seen = seen

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        assert ctx.workspace is not None
        try:
            await ctx.workspace.read_bytes("plan.md")
        except Exception:
            self._seen["plan_seen_by_solver"] = False
        else:
            self._seen["plan_seen_by_solver"] = True
        source = (await ctx.workspace.read_bytes("calc.py")).content
        await ctx.workspace.create_bytes("solution.py", source.replace(b"a - b", b"a + b"))
        return ToolResult(content="solved")


class _Crash(Exception):
    """The process died here."""


class _ResumablePlanThenSolve(WorkflowBase):
    spec = WorkflowSpec(name="plan-then-solve")

    def __init__(self, app, *, crash_after_plan: bool) -> None:
        super().__init__(app)
        self._crash_after_plan = crash_after_plan

    async def run(self, session_id: str):
        ctx = self.context(session_id)
        yield await ctx.start()
        options = StepRunOptions(allocation_scope="parent")
        await step(ctx, agent="planner", step_id="plan", prompt="plan", run_options=options)
        if self._crash_after_plan:
            raise _Crash()
        await step(ctx, agent="solver", step_id="solve", prompt="solve", run_options=options)
        yield await ctx.completed()


def test_a_resumed_run_gets_a_new_brokered_sandbox_and_recovery_destroys_the_old(
    tmp_path: Path,
) -> None:
    source = _project(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")
    seen: dict[str, object] = {}
    keyring = PublicAuthorityAliasKeyring(
        active_key_id="test",
        keys={"test": SecretStr(base64.urlsafe_b64encode(b"k" * 32).decode().rstrip("="))},
    )
    stores: list[SQLiteSessionStore] = []

    def app(*turns) -> CayuApp:
        store = SQLiteSessionStore(
            tmp_path / "sessions.sqlite",
            public_authority_alias_codec=PublicAuthorityAliasCodec(keyring),
        )
        stores.append(store)
        built = CayuApp(
            session_store=store, enable_logging=False, public_authority_alias_keyring=keyring
        )
        built.register_provider(ScriptedModelProvider(list(turns)), default=True)
        built.register_environment_factory(
            EnvironmentSpec(name="sandbox", execution_profile_identity=identity("brokered")),
            _factory(broker, source),
            default=True,
        )
        built.register_agent(AgentSpec(name="planner", model="scripted"), tools=[_WritePlan()])
        built.register_agent(
            AgentSpec(name="solver", model="scripted"), tools=[_SolveWhatIsThere(seen)]
        )
        return built

    async def scenario():
        try:
            first = app(_call("write_plan"), _done())
            with pytest.raises(_Crash):
                async for _ in _ResumablePlanThenSolve(first, crash_after_plan=True).run("run-1"):
                    pass
            # The process dies: its lease keeper stops renewing, and nothing it
            # held in memory is released.
            registry = first._environment_lifecycle.shared_allocations
            for entry in tuple(registry._live.values()):
                await entry.runner.lease_keeper.stop()
                registry._forget(entry)
            registry._live.clear()
            (old,) = broker.live()
            second = app(_call("solve"), _done())
            resumed = _ResumablePlanThenSolve(second, crash_after_plan=False)
            events = [event async for event in resumed.run("run-1")]
            assert events[-1].type == EventType.WORKFLOW_COMPLETED, events[-1].payload
            # The old sandbox can't be reattached; it is left behind as an orphan.
            assert [sandbox.id for sandbox in broker.live()] == [old.id]
            page = await second.recover_incomplete_sessions(
                IncompleteSessionsRecoveryRequest(statuses={SessionStatus.COMPLETED})
            )
            return old.id, page
        finally:
            for store in stores:
                await store.close()

    old_id, page = asyncio.run(scenario())
    first_sandbox, second_sandbox = broker.sandboxes.values()
    assert first_sandbox.id == old_id
    # The resumed run worked in a new sandbox: the plan was lost with the old one.
    assert seen["plan_seen_by_solver"] is False
    assert not (source / "plan.md").exists()
    assert (source / "solution.py").read_bytes() == b"def add(a, b):\n    return a + b\n"
    assert second_sandbox.status == "destroyed" and second_sandbox.destroy_reason == "requested"
    # Recovery destroyed the old one through the factory, before its lease lapsed.
    (run,) = [result for result in page.results if result.session_id == "run-1"]
    assert IncompleteSessionRecoveryAction.REAPED_ALLOCATION in run.actions
    assert first_sandbox.status == "destroyed" and first_sandbox.destroy_reason == "requested"
    assert broker.live() == []
