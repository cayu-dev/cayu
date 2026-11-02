"""An interrupted owner's shared allocation leaves nothing behind that nobody owns.

Lambda MicroVM's binding suspends an interrupted MicroVM instead of
terminating it, so an ordinary session can resume it. A shared allocation's
record is forgotten once its owner releases it, so the release must not leave
a suspended allocation behind with no owner. It also must not keep a record of
one the binding did dispose of, or the owner's resume would keep failing to
reconnect to it.
"""

from __future__ import annotations

import asyncio

import pytest
from tests.environments.test_durable_shared_allocations import (
    _Crash,
    _PlanThenSolve,
    _ReconnectingFactory,
    _records,
    _SandboxBinding,
    _World,
)
from tests.sandbox_broker.test_brokered_shared_allocation import _call, _done

from cayu import EventType
from cayu.environments import EnvironmentFactory, EnvironmentFactoryOperation, SyncBinding


@pytest.fixture
def suspends_when_interrupted(monkeypatch):
    """The sandbox binding keeps its sandbox on an interrupted finalize, as Lambda does."""

    async def finalize(binding, bound, *, outcome=None, metadata=None):
        snapshot = await SyncBinding.finalize(binding, bound, outcome=outcome, metadata=metadata)
        binding._sandboxes.finalizations.append(outcome)
        if outcome != "interrupted":
            binding._sandboxes.dispose(binding._name)
        return snapshot

    monkeypatch.setattr(_SandboxBinding, "finalize", finalize)


async def _interrupt_after_plan(world: _World, run_id: str):
    """Run the plan step, then interrupt the run as a cancelled ``execute`` does."""

    first = world.app(_call("write_plan"), _done())
    with pytest.raises(_Crash):
        async for _ in _PlanThenSolve(first, crash_after_plan=True).run(run_id):
            pass
    await first._environment_lifecycle.shared_allocations.end_owner(run_id, outcome="interrupted")
    return first


def test_an_interrupted_release_disposes_of_what_its_binding_suspended(
    tmp_path, suspends_when_interrupted
):
    world = _World(tmp_path)

    async def scenario():
        try:
            first = await _interrupt_after_plan(world, "run-suspend")
            return await _records(first, "run-suspend")
        finally:
            await world.close()

    records = asyncio.run(scenario())
    # The binding only suspended the sandbox, so the release reconnected to it
    # and discarded it before forgetting its record: nothing is left unowned.
    created, reconnected = world.sandboxes.requests
    assert created.operation is EnvironmentFactoryOperation.CREATE
    assert reconnected.operation is EnvironmentFactoryOperation.RECONNECT
    assert reconnected.session_id == created.session_id
    assert world.sandboxes.finalizations == ["interrupted"]
    assert len(world.sandboxes.disposed) == 1
    assert not any(world.sandboxes.root.iterdir())
    assert records == {}


def test_an_interrupted_release_without_disposal_proof_lets_the_owner_resume(tmp_path, monkeypatch):
    # The binding disposes of the sandbox on every finalize, and the factory
    # can't prove an allocation gone (the base class's answer).
    monkeypatch.setattr(
        _ReconnectingFactory, "is_allocation_disposed", EnvironmentFactory.is_allocation_disposed
    )
    world = _World(tmp_path)

    async def scenario():
        try:
            first = await _interrupt_after_plan(world, "run-disposed")
            interrupted = await _records(first, "run-disposed")
            # The owner resumes: its plan step is journaled, and its solve step
            # gets a fresh sandbox (the interrupted copy-back put the plan in
            # the source, so the fresh copy-in brings it along).
            second = world.app(_call("solve"), _done())
            events = [event async for event in _PlanThenSolve(second).run("run-disposed")]
            return interrupted, events
        finally:
            await world.close()

    interrupted, events = asyncio.run(scenario())
    # The release's reconnect failed, so its disposal isn't confirmed: the
    # record is kept, but as an orphan. The resume never claims an orphan, so
    # it doesn't keep failing to reconnect to the disposed sandbox.
    assert interrupted["records"] == {}
    assert len(interrupted["orphans"]) == 1
    assert world.sandboxes.finalizations[0] == "interrupted"
    assert events[-1].type == EventType.WORKFLOW_COMPLETED, events[-1].payload


def test_a_resumed_run_rewrites_what_the_interrupted_copy_back_published(
    tmp_path, suspends_when_interrupted
):
    """The resumed copy-back compares against the source as the interruption left it."""

    from cayu import AgentSpec
    from cayu.tools.base import Tool, ToolContext, ToolResult, ToolSpec
    from cayu.workflows import StepRunOptions, WorkflowBase, WorkflowSpec, step

    class _Revise(Tool):
        spec = ToolSpec(name="revise", parallel_safe=False, workspace_mutation=True)

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            assert ctx.workspace is not None
            plan = (await ctx.workspace.read_bytes("plan.md")).content
            await ctx.workspace.write_bytes("plan.md", plan + b"revised\n")
            return ToolResult(content="revised")

    class _PlanThenRevise(WorkflowBase):
        spec = WorkflowSpec(name="plan-then-solve")

        def __init__(self, app, *, stop_after_plan=False) -> None:
            super().__init__(app)
            self._stop_after_plan = stop_after_plan

        async def run(self, session_id):
            ctx = self.context(session_id)
            shared = StepRunOptions(allocation_scope="parent")
            yield await ctx.start()
            await step(ctx, agent="planner", step_id="plan", prompt="plan", run_options=shared)
            if self._stop_after_plan:
                raise _Crash()
            await step(ctx, agent="reviser", step_id="revise", prompt="go", run_options=shared)
            yield await ctx.completed()

    world = _World(tmp_path)

    async def scenario():
        try:
            first = world.app(_call("write_plan"), _done())
            with pytest.raises(_Crash):
                async for _ in _PlanThenRevise(first, stop_after_plan=True).run("run-revise"):
                    pass
            await first._environment_lifecycle.shared_allocations.end_owner(
                "run-revise", outcome="interrupted"
            )
            published = (world.source / "plan.md").read_bytes()
            second = world.app(_call("revise"), _done())
            second.register_agent(AgentSpec(name="reviser", model="scripted"), tools=[_Revise()])
            events = [event async for event in _PlanThenRevise(second).run("run-revise")]
            return published, events
        finally:
            await world.close()

    published, events = asyncio.run(scenario())
    # The interrupted copy-back published the plan; the resumed run rewrote
    # that same file and its copy-back went through without a conflict.
    assert published == b"replace a - b with a + b\n"
    assert events[-1].type == EventType.WORKFLOW_COMPLETED, events[-1].payload
    assert (world.source / "plan.md").read_bytes() == published + b"revised\n"


def test_a_failed_disposal_after_an_interrupted_release_stays_owned_for_a_retry(
    tmp_path, suspends_when_interrupted, monkeypatch
):
    from cayu.sessions.records import SessionStatus
    from cayu.sessions.recovery import (
        IncompleteSessionRecoveryAction,
        IncompleteSessionsRecoveryRequest,
    )

    original = _ReconnectingFactory.create
    outage = [True]

    async def briefly_unreachable(factory, request):
        if outage[0] and request.operation is EnvironmentFactoryOperation.RECONNECT:
            raise ConnectionError("the provider is briefly unreachable")
        return await original(factory, request)

    monkeypatch.setattr(_ReconnectingFactory, "create", briefly_unreachable)
    monkeypatch.setattr(
        _ReconnectingFactory, "is_allocation_disposed", EnvironmentFactory.is_allocation_disposed
    )
    world = _World(tmp_path)

    async def scenario():
        try:
            first = await _interrupt_after_plan(world, "run-retry")
            kept = await _records(first, "run-retry")
            suspended = list(world.sandboxes.root.iterdir())
            outage[0] = False
            page = await first.recover_incomplete_sessions(
                IncompleteSessionsRecoveryRequest(statuses={SessionStatus.COMPLETED})
            )
            return kept, suspended, page, await _records(first, "run-retry")
        finally:
            await world.close()

    kept, suspended, page, records = asyncio.run(scenario())
    # The binding suspended the sandbox and discarding it failed: its record
    # stayed (as an orphan), so the sandbox wasn't left without an owner.
    assert len(suspended) == 1
    assert len(kept["orphans"]) == 1
    # Once the provider answers again, recovery disposes of it.
    actions = {result.session_id: result.actions for result in page.results}
    assert IncompleteSessionRecoveryAction.REAPED_ALLOCATION in actions["run-retry"]
    assert not any(world.sandboxes.root.iterdir())
    assert records == {}
