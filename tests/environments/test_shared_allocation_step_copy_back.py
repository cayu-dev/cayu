"""A step run with ``copy_back="after_step"`` copies the shared work back when it ends.

These use the reconnecting fake from ``test_durable_shared_allocations``: a
``SyncBinding`` with ``source_conflict_policy="require_revision"`` over a
sandbox directory that outlives every app, and a SQLite store per "process".
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from tests.core.test_human_review import identity
from tests.environments.test_durable_shared_allocations import (
    _Crash,
    _crash,
    _ReconnectingFactory,
    _records,
    _SandboxBinding,
    _World,
)
from tests.sandbox_broker.test_brokered_shared_allocation import _call, _done

from cayu import AgentSpec, CayuApp, EventType
from cayu.environments import (
    AllocationSharingScope,
    Environment,
    EnvironmentFactory,
    EnvironmentFactoryOperation,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
    EnvironmentSpec,
    SyncBinding,
)
from cayu.environments.bindings import SyncBindingSourceConflictError
from cayu.evals import ScriptedModelProvider
from cayu.messages import MessageRole
from cayu.runtime._shared_environment_allocations import (
    SharedAllocationCopyBackPendingError,
    SharedAllocationKey,
    SharedEnvironmentAllocations,
    _SharedAllocation,
)
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import Session
from cayu.storage import SQLiteSessionStore
from cayu.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from cayu.workflows import StepError, StepRunOptions, WorkflowBase, WorkflowSpec, parallel, step
from cayu.workflows import workflow as workflow_module
from cayu.workspaces import LocalWorkspace

_PLAN = b"replace a - b with a + b\n"
_CALC = b"def add(a, b):\n    return a - b\n"


class _PlanAndEditSource(Tool):
    """Writes the plan in the sandbox while someone edits the source checkout by hand."""

    spec = ToolSpec(name="plan_and_edit", parallel_safe=False, workspace_mutation=True)

    def __init__(self, source: Path) -> None:
        self._source = source

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        assert ctx.workspace is not None
        await ctx.workspace.create_bytes("plan.md", _PLAN)
        (self._source / "calc.py").write_bytes(b"# edited by hand\n" + _CALC)
        return ToolResult(content="planned")


class _EditPlan(Tool):
    """Changes the plan a previous step already copied back."""

    spec = ToolSpec(name="edit_plan", parallel_safe=False, workspace_mutation=True)

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        assert ctx.workspace is not None
        current = await ctx.workspace.read_bytes("plan.md")
        assert current.content == _PLAN
        await ctx.workspace.replace_bytes(
            "plan.md", _PLAN + b"checked\n", expected_revision=current.revision
        )
        return ToolResult(content="edited")


class _Steps(WorkflowBase):
    """Runs one step per agent, in order, all sharing the run's allocation."""

    spec = WorkflowSpec(name="copy-back-steps")

    def __init__(self, app, agents, *, copy_back="after_step", observe=None, crash_after=None):
        super().__init__(app)
        self._agents = agents
        self._options = StepRunOptions(
            allocation_scope=AllocationSharingScope.PARENT, copy_back=copy_back
        )
        self._observe = observe or (lambda agent: None)
        self._crash_after = crash_after

    async def run(self, session_id: str):
        ctx = self.context(session_id)
        yield await ctx.start()
        for agent in self._agents:
            await step(ctx, agent=agent, step_id=agent, prompt=agent, run_options=self._options)
            self._observe(agent)
            if agent == self._crash_after:
                raise _Crash()
        yield await ctx.completed()


def _app(world: _World, *turns) -> CayuApp:
    app = world.app(*turns)
    app.register_agent(
        AgentSpec(name="hasty-planner", model="scripted"), tools=[_PlanAndEditSource(world.source)]
    )
    app.register_agent(AgentSpec(name="editor", model="scripted"), tools=[_EditPlan()])
    return app


@pytest.fixture
def copy_backs(monkeypatch) -> list[str]:
    """Every mid-run copy-back the sandbox binding ran, by the session that bound it."""

    calls: list[str] = []
    original = _SandboxBinding._copy_back_while_bound

    async def recording(self, bound, *, outcome="completed"):
        snapshot = await original(self, bound, outcome=outcome)
        calls.append(outcome)
        return snapshot

    monkeypatch.setattr(_SandboxBinding, "_copy_back_while_bound", recording)
    return calls


async def _completed_steps(app: CayuApp, run_id: str) -> list[str]:
    events = await app.session_store.query_events(
        EventQuery(session_id=run_id, event_type=EventType.WORKFLOW_STEP_COMPLETED)
    )
    return [record.event.payload["step_id"] for record in events]


@pytest.mark.parametrize("copy_back", ["after_step", "on_release"])
def test_the_source_sees_a_steps_work_after_the_step_only_with_the_option(
    tmp_path, copy_back, copy_backs
):
    world = _World(tmp_path)
    seen: dict[str, dict[str, object]] = {}

    def observe(agent: str) -> None:
        seen[agent] = {
            "plan": (world.source / "plan.md").exists(),
            "solution": (world.source / "solution.py").exists(),
            "finalized": list(world.sandboxes.finalizations),
            "sandboxes": len(list(world.sandboxes.root.iterdir())),
        }

    async def scenario():
        try:
            app = _app(world, _call("write_plan"), _done(), _call("solve"), _done())
            workflow = _Steps(app, ["planner", "solver"], copy_back=copy_back, observe=observe)
            return [event async for event in workflow.run("run-1")]
        finally:
            await world.close()

    events = asyncio.run(scenario())
    assert events[-1].type == EventType.WORKFLOW_COMPLETED, events[-1].payload
    after_step = copy_back == "after_step"
    assert seen["planner"]["plan"] is after_step
    assert seen["solver"]["solution"] is after_step
    # Either way the sandbox stays up between the steps and only completion
    # finalizes it, and it was copied into once.
    for agent in ("planner", "solver"):
        assert seen[agent]["finalized"] == []
        assert seen[agent]["sandboxes"] == 1
    assert world.sandboxes.finalizations == ["completed"]
    assert len(world.sandboxes.copy_ins) == 1
    assert copy_backs == (["completed", "completed"] if after_step else [])
    assert (world.source / "plan.md").read_bytes() == _PLAN
    assert (world.source / "solution.py").read_bytes() == b"def add(a, b):\n    return a + b\n"


def test_a_copy_back_conflict_fails_the_step_and_keeps_the_allocation(tmp_path, copy_backs):
    world = _World(tmp_path)

    async def scenario():
        try:
            app = _app(world, _call("plan_and_edit"), _done(), _call("solve"), _done())
            with pytest.raises(StepError) as raised:
                async for _ in _Steps(app, ["hasty-planner", "solver"]).run("run-1"):
                    pass
            registry = app._environment_lifecycle.shared_allocations
            (record,) = (await _records(app, "run-1"))["records"].values()
            conflicted = {
                "error": raised.value,
                "live": len(registry.live_keys()),
                "finalized": list(world.sandboxes.finalizations),
                "sandbox_alive": world.sandboxes.root.exists()
                and len(list(world.sandboxes.root.iterdir())) == 1,
                "plan_in_source": (world.source / "plan.md").exists(),
                "journaled": await _completed_steps(app, "run-1"),
                "record_copied_back": record.get("copied_back", []),
            }
            # Resolve the conflict, then resume: the step's copy-back runs again
            # before its completion is journaled, and the run carries on.
            (world.source / "calc.py").write_bytes(_CALC)
            events = [
                event async for event in _Steps(app, ["hasty-planner", "solver"]).run("run-1")
            ]
            return conflicted, events, await _completed_steps(app, "run-1")
        finally:
            await world.close()

    conflicted, events, journaled = asyncio.run(scenario())
    error = conflicted["error"]
    assert isinstance(error.__cause__, SyncBindingSourceConflictError)
    assert "copying its shared work back" in str(error)
    assert error.step_id == "hasty-planner"
    # Like a conflict at the owner's release: nothing was disposed of, the
    # allocation is still live with the work in it, and nothing was published.
    assert conflicted["live"] == 1
    assert conflicted["finalized"] == []
    assert conflicted["sandbox_alive"] is True
    assert conflicted["plan_in_source"] is False
    assert conflicted["journaled"] == []
    assert conflicted["record_copied_back"] == []
    assert events[-1].type == EventType.WORKFLOW_COMPLETED, events[-1].payload
    assert journaled == ["hasty-planner", "solver"]
    # The resumed attempt reused the same sandbox; the retry and the solver
    # each copied back once, and completion finalized once.
    assert len(world.sandboxes.copy_ins) == 1
    assert copy_backs == ["completed", "completed"]
    assert world.sandboxes.finalizations == ["completed"]
    assert (world.source / "plan.md").read_bytes() == _PLAN
    assert (world.source / "calc.py").read_bytes() == _CALC


def test_a_crash_after_a_copy_back_reconnects_from_it_without_copying_back_again(
    tmp_path, copy_backs
):
    world = _World(tmp_path)

    async def scenario():
        try:
            first = _app(world, _call("write_plan"), _done())
            with pytest.raises(_Crash):
                async for _ in _Steps(first, ["planner"], crash_after="planner").run("run-1"):
                    pass
            _crash(first)
            (record,) = (await _records(first, "run-1"))["records"].values()
            second = _app(world, _call("edit_plan"), _done())
            events = [event async for event in _Steps(second, ["planner", "editor"]).run("run-1")]
            return record, events
        finally:
            await world.close()

    record, events = asyncio.run(scenario())
    assert events[-1].type == EventType.WORKFLOW_COMPLETED, events[-1].payload
    # The copy-back moved the recorded baseline: the plan is one of the paths
    # the source now has, and the step that copied it is remembered.
    assert "plan.md" in record["binding_state"]["source_paths"]
    assert len(record["copied_back"]) == 1
    # One sandbox, copied into once, reconnected after the crash. The planner's
    # copy-back wasn't repeated, and the editor's change to the already
    # copied plan published without a conflict.
    operations = [request.operation for request in world.sandboxes.requests]
    assert operations == [EnvironmentFactoryOperation.CREATE, EnvironmentFactoryOperation.RECONNECT]
    assert len(world.sandboxes.copy_ins) == 1
    assert copy_backs == ["completed", "completed"]
    assert world.sandboxes.finalizations == ["completed"]
    assert (world.source / "plan.md").read_bytes() == _PLAN + b"checked\n"


def test_a_crash_before_the_step_is_journaled_does_not_lose_or_repeat_its_copy_back(
    tmp_path, copy_backs, monkeypatch
):
    world = _World(tmp_path)
    original = workflow_module._append_step_completed

    async def crash_before_journaling(ctx, agent, result):
        if result.step_id == "planner" and ctx.app is crashing[0]:
            raise _Crash()
        return await original(ctx, agent, result)

    crashing: list[object] = [None]
    monkeypatch.setattr(workflow_module, "_append_step_completed", crash_before_journaling)

    async def scenario():
        try:
            first = _app(world, _call("write_plan"), _done())
            crashing[0] = first
            with pytest.raises(_Crash):
                async for _ in _Steps(first, ["planner", "editor"]).run("run-1"):
                    pass
            _crash(first)
            journaled_before = await _completed_steps(first, "run-1")
            second = _app(world, _call("edit_plan"), _done())
            events = [event async for event in _Steps(second, ["planner", "editor"]).run("run-1")]
            return journaled_before, events, await _completed_steps(second, "run-1")
        finally:
            await world.close()

    journaled_before, events, journaled = asyncio.run(scenario())
    assert journaled_before == []
    assert events[-1].type == EventType.WORKFLOW_COMPLETED, events[-1].payload
    assert journaled == ["planner", "editor"]
    # The planner's work reached the source before the crash. The resumed run
    # reconnected to journal the planner's step, saw from the record that its
    # copy-back had happened, and didn't run it again.
    operations = [request.operation for request in world.sandboxes.requests]
    assert operations == [EnvironmentFactoryOperation.CREATE, EnvironmentFactoryOperation.RECONNECT]
    assert copy_backs == ["completed", "completed"]
    assert len(world.sandboxes.copy_ins) == 1
    assert world.sandboxes.finalizations == ["completed"]
    assert (world.source / "plan.md").read_bytes() == _PLAN + b"checked\n"


def test_a_sync_binding_copies_back_while_bound_and_finalize_publishes_the_rest(tmp_path):
    (tmp_path / "source").mkdir()
    (tmp_path / "target").mkdir()
    (tmp_path / "source" / "calc.py").write_bytes(_CALC)
    source = LocalWorkspace(tmp_path / "source", workspace_id="source")
    target = LocalWorkspace(tmp_path / "target", workspace_id="target")
    binding = SyncBinding(
        target_workspace=target,
        source_conflict_policy="require_revision",
        max_file_bytes=1 << 20,
        max_total_bytes=1 << 24,
    )

    async def scenario():
        bound = await binding.bind(source, None, session_id="shared")
        assert bound.state_key is not None
        await target.write_bytes("plan.md", _PLAN)
        await binding._copy_back_while_bound(bound)
        state = binding._states[bound.state_key]
        seen: dict[str, object] = {
            "plan": (tmp_path / "source" / "plan.md").read_bytes(),
            "phase": state.phase,
            "deferred": state.defer_finalize_release,
        }
        # Later work, a change to a file already copied back and a deletion,
        # reaches the source at finalize without a conflict.
        current = await target.read_bytes("plan.md")
        assert current.revision is not None
        await target.replace_bytes(
            "plan.md", _PLAN + b"checked\n", expected_revision=current.revision
        )
        await target.delete("calc.py")
        await binding.finalize(bound, outcome="completed")
        seen["released"] = binding._states == {}
        # Finalize released the source's reservation, so it can be bound again.
        again = await binding.bind(source, None, session_id="next")
        binding.abandon(again)
        return seen

    seen = asyncio.run(scenario())
    assert seen == {"plan": _PLAN, "phase": "active", "deferred": False, "released": True}
    assert (tmp_path / "source" / "plan.md").read_bytes() == _PLAN + b"checked\n"
    assert not (tmp_path / "source" / "calc.py").exists()


@pytest.fixture
def sync_back(request, monkeypatch) -> str:
    """Give every sandbox binding the sync-back policy the test is parametrized with."""

    policy = request.param
    original = _SandboxBinding.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.sync_back = policy

    monkeypatch.setattr(_SandboxBinding, "__init__", init)
    return policy


@pytest.mark.parametrize("shutdown", [False, True])
@pytest.mark.parametrize("sync_back", ["always", "on_success"], indirect=True)
def test_a_failed_runs_release_keeps_a_conflicted_step_and_a_resume_publishes_it(
    tmp_path, sync_back, shutdown, copy_backs
):
    world = _World(tmp_path)
    in_source_after: dict[str, bool] = {}

    def observe(agent: str) -> None:
        in_source_after[agent] = (world.source / "plan.md").exists()

    async def scenario():
        try:
            app = _app(world, _call("plan_and_edit"), _done(), _call("solve"), _done())
            registry = app._environment_lifecycle.shared_allocations
            agents = ["hasty-planner", "solver"]
            # execute(), unlike run(), releases the allocation with "failed"
            # when the step's copy-back conflict ends the run.
            with pytest.raises(StepError):
                async for _ in _Steps(app, agents, observe=observe).execute("run-1"):
                    pass
            (record,) = (await _records(app, "run-1"))["records"].values()
            kept = {
                "closing": len(registry._closing),
                "finalized": list(world.sandboxes.finalizations),
                "disposed": list(world.sandboxes.disposed),
                "pending": record.get("pending_copy_back", []),
            }
            if shutdown:
                # With a durable store, shutdown leaves it for the run to
                # resume instead of discarding the work its release couldn't
                # publish; the resumed step then reconnects to it.
                await app.close_idle_environment_resources(timeout_s=5)
                kept["disposed"] = list(world.sandboxes.disposed)
            (world.source / "calc.py").write_bytes(_CALC)
            events = [
                event async for event in _Steps(app, agents, observe=observe).execute("run-1")
            ]
            return kept, events, await _completed_steps(app, "run-1")
        finally:
            await world.close()

    kept, events, journaled = asyncio.run(scenario())
    # The failed run's release didn't dispose of the only copy of the work:
    # "always" tried the copy-back again and hit the same conflict, and
    # "on_success", which wouldn't have copied back, refused to release.
    assert kept["closing"] == 1
    assert kept["finalized"] == [] and kept["disposed"] == []
    assert len(kept["pending"]) == 1
    assert events[-1].type == EventType.WORKFLOW_COMPLETED, events[-1].payload
    assert journaled == ["hasty-planner", "solver"]
    # The resumed step published its work before it was journaled.
    assert in_source_after == {"hasty-planner": True, "solver": True}
    assert copy_backs == ["completed", "completed"]
    assert len(world.sandboxes.copy_ins) == 1
    assert world.sandboxes.finalizations == ["completed"]
    assert (world.source / "plan.md").read_bytes() == _PLAN
    if shutdown:
        operations = [request.operation for request in world.sandboxes.requests]
        assert operations[-1] is EnvironmentFactoryOperation.RECONNECT


class _WriteQuick(Tool):
    spec = ToolSpec(name="write_quick", parallel_safe=False, workspace_mutation=True)

    def __init__(self) -> None:
        self.done = asyncio.Event()

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        assert ctx.workspace is not None
        await ctx.workspace.create_bytes("quick.txt", b"quick\n")
        self.done.set()
        return ToolResult(content="written")


class _WriteSlowly(Tool):
    """Leaves a half-written file in the sandbox until the test lets it finish."""

    spec = ToolSpec(name="write_slow", parallel_safe=False, workspace_mutation=True)

    def __init__(self) -> None:
        self.inside = asyncio.Event()
        self.finish = asyncio.Event()

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        assert ctx.workspace is not None
        await ctx.workspace.create_bytes("shared.txt", b"half\n")
        self.inside.set()
        await asyncio.wait_for(self.finish.wait(), timeout=30)
        current = await ctx.workspace.read_bytes("shared.txt")
        assert current.revision is not None
        await ctx.workspace.replace_bytes(
            "shared.txt", b"done\n", expected_revision=current.revision
        )
        return ToolResult(content="written")


def _respond_by_prompt(request):
    if any(message.role == MessageRole.TOOL for message in request.messages):
        return _done()
    prompt = next(message for message in request.messages if message.role == MessageRole.USER)
    return _call(f"write_{prompt.content[0].text}")


async def _drain(stream) -> None:
    async for _ in stream:
        pass


def test_a_copy_back_waits_for_parallel_writers_to_leave_the_sandbox(tmp_path, copy_backs):
    world = _World(tmp_path)
    quick, slow = _WriteQuick(), _WriteSlowly()
    seen: dict[str, object] = {}

    class QuickAndSlow(WorkflowBase):
        spec = WorkflowSpec(name="quick-and-slow")

        async def run(self, session_id: str):
            ctx = self.context(session_id)
            yield await ctx.start()
            options = StepRunOptions(
                allocation_scope=AllocationSharingScope.PARENT, copy_back="after_step"
            )
            outcome = await parallel(
                [
                    step(ctx, agent=name, step_id=name, prompt=name, run_options=options)
                    for name in ("quick", "slow")
                ]
            )
            assert not outcome.failures, outcome.failures
            yield await ctx.completed()

    async def scenario():
        store = SQLiteSessionStore(world.database)
        try:
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(
                ScriptedModelProvider(response_factory=_respond_by_prompt), default=True
            )
            app.register_environment_factory(
                EnvironmentSpec(name="sandbox", execution_profile_identity=identity("sandboxes")),
                _ReconnectingFactory(world.source, world.sandboxes),
                default=True,
            )
            app.register_agent(AgentSpec(name="quick", model="scripted"), tools=[quick])
            app.register_agent(AgentSpec(name="slow", model="scripted"), tools=[slow])
            run = asyncio.ensure_future(_drain(QuickAndSlow(app).run("run-1")))
            await asyncio.wait_for(asyncio.gather(quick.done.wait(), slow.inside.wait()), 30)
            # The quick step's session ends and its copy-back starts, but it
            # waits, gate closed, while the slow step is still writing.
            registry = app._environment_lifecycle.shared_allocations

            def copying_or_waiting() -> bool:
                gates = [
                    getattr(entry, "copy_back_gate", None) for entry in registry._live.values()
                ]
                return bool(copy_backs) or any(
                    gate is not None and not gate.is_set() for gate in gates
                )

            for _ in range(1000):
                if copying_or_waiting():
                    break
                await asyncio.sleep(0.01)
            for _ in range(20):
                await asyncio.sleep(0.01)
            seen["copied_while_writing"] = list(copy_backs)
            seen["source_while_writing"] = sorted(path.name for path in world.source.iterdir())
            slow.finish.set()
            await asyncio.wait_for(run, 30)
        finally:
            await store.close()

    asyncio.run(scenario())
    assert seen["copied_while_writing"] == []
    assert seen["source_while_writing"] == ["calc.py"]
    assert copy_backs == ["completed", "completed"]
    # Nothing half-written ever reached the source.
    assert (world.source / "shared.txt").read_bytes() == b"done\n"
    assert (world.source / "quick.txt").read_bytes() == b"quick\n"


class _NoFactory(EnvironmentFactory):
    @property
    def secret_resolution_scope(self):
        return "static"

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        raise AssertionError("the allocation is already live")


def test_a_subagent_left_behind_waits_at_a_closed_copy_back_gate():
    registry = SharedEnvironmentAllocations(session_store=InMemorySessionStore())
    factory = _NoFactory()
    key = SharedAllocationKey(AllocationSharingScope.PARENT, "run-1", "run-1", "sandbox")
    entry = _SharedAllocation(
        key=key,
        factory=factory,
        result=EnvironmentFactoryResult(environment=Environment(EnvironmentSpec(name="sandbox"))),
        bound=None,
    )
    request = EnvironmentFactoryRequest(
        session_id="run-1.step.sub", agent_name="helper", environment_name="sandbox"
    )
    # A background subagent of the finished step, arriving during its copy-back.
    subagent = cast("Session", SimpleNamespace(id="run-1.step.sub", parent_session_id="run-1.step"))

    async def scenario():
        registry.open_owner("run-1")
        registry._live[key] = entry
        copy_back = registry._attach(entry, "run-1.step", copy_back=True)
        entry.copy_back_gate.clear()
        arriving = asyncio.ensure_future(
            registry._acquire_attachable(key, factory, request, subagent)
        )
        for _ in range(10):
            await asyncio.sleep(0)
        waited = not arriving.done()
        entry.copy_back_gate.set()
        registry._detach(entry, copy_back)
        return waited, await asyncio.wait_for(arriving, 1)

    waited, acquired = asyncio.run(scenario())
    # The copy-back's own attachment names the step, but it isn't the step
    # still running, so the subagent doesn't get in until the gate opens.
    assert waited is True
    assert acquired is entry


@pytest.mark.parametrize("sync_back", ["on_success"], indirect=True)
def test_a_kept_release_can_be_retried_with_the_completed_outcome(tmp_path, sync_back, copy_backs):
    world = _World(tmp_path)
    seen: dict[str, object] = {}

    class ReleaseByHand(WorkflowBase):
        spec = WorkflowSpec(name="release-by-hand")

        async def run(self, session_id: str):
            ctx = self.context(session_id)
            yield await ctx.start()
            options = StepRunOptions(
                allocation_scope=AllocationSharingScope.PARENT, copy_back="after_step"
            )
            with pytest.raises(StepError):
                await step(
                    ctx,
                    agent="hasty-planner",
                    step_id="plan",
                    prompt="plan",
                    run_options=options,
                )
            # A failed release wouldn't copy back, so it keeps the work.
            with pytest.raises(SharedAllocationCopyBackPendingError):
                await ctx.release_environments(outcome="failed")
            seen["finalized_after_failed"] = list(world.sandboxes.finalizations)
            (world.source / "calc.py").write_bytes(_CALC)
            # With the source fixed, the same kept release publishes it.
            await ctx.release_environments(outcome="completed")
            yield await ctx.completed()

    async def scenario():
        try:
            app = _app(world, _call("plan_and_edit"), _done())
            return [event async for event in ReleaseByHand(app).run("run-1")]
        finally:
            await world.close()

    events = asyncio.run(scenario())
    assert events[-1].type == EventType.WORKFLOW_COMPLETED, events[-1].payload
    assert seen["finalized_after_failed"] == []
    assert world.sandboxes.finalizations == ["completed"]
    assert (world.source / "plan.md").read_bytes() == _PLAN
    assert copy_backs == []
