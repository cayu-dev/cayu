"""Workflow command steps run in the allocation the run's agent steps share."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from tests.environments.test_shared_allocations import _owners, _RecordingBinding

from cayu import AgentSpec, CayuApp, EventType
from cayu.environments import (
    Environment,
    EnvironmentFactory,
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
    EnvironmentSpec,
)
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.runners import LocalRunner
from cayu.runners.base import ExecResult
from cayu.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from cayu.workflows import (
    CommandResult,
    StepError,
    StepRunOptions,
    WorkflowBase,
    WorkflowSpec,
    command,
    step,
)
from cayu.workspaces import LocalWorkspace

PY = sys.executable
_SOLUTION = b"from pathlib import Path\nprint(Path('plan.md').read_text().strip())\n"


class _BlockingRunner(LocalRunner):
    """A local runner whose redacted exec, the one ``command`` uses, a test can hold open."""

    def __init__(self, root: Path, gate: asyncio.Event | None) -> None:
        super().__init__(root)
        self.gate = gate
        self.entered = asyncio.Event()

    async def exec_redacted(self, *args, **kwargs) -> ExecResult:
        self.entered.set()
        if self.gate is not None:
            await asyncio.wait_for(self.gate.wait(), timeout=30)
        return await super().exec_redacted(*args, **kwargs)


class _RunnerFactory(EnvironmentFactory):
    """Gives every allocation its own directory, workspace and local runner.

    A command sees a step's file only if both ran in the same allocation.
    """

    def __init__(self, root: Path, *, gate: asyncio.Event | None = None) -> None:
        self.root = root
        self.gate = gate
        self.requests: list[EnvironmentFactoryRequest] = []
        self.runners: list[_BlockingRunner] = []
        self.log: list[tuple[str, str | None]] = []
        self.discards = 0
        self.reader: _ReadSeed | None = None

    @property
    def secret_resolution_scope(self):
        return "static"

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        self.requests.append(request)
        root = self.root / f"allocation-{len(self.requests)}"
        root.mkdir()
        runner = _BlockingRunner(root, self.gate)
        self.runners.append(runner)

        async def release(action: EnvironmentFactoryReleaseAction) -> None:
            if action is EnvironmentFactoryReleaseAction.DISCARD:
                self.discards += 1

        return EnvironmentFactoryResult(
            environment=Environment(
                EnvironmentSpec(name=request.environment_name),
                workspace=LocalWorkspace(root, workspace_id=root.name),
                runner=runner,
                binding=_RecordingBinding(self.log, [0]),
            ),
            release=release,
        )

    def finalizations(self) -> list[str | None]:
        return [outcome for kind, outcome in self.log if kind == "finalize"]


class _WritePlan(Tool):
    spec = ToolSpec(name="write_plan", parallel_safe=False, workspace_mutation=True)

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        assert ctx.workspace is not None
        await ctx.workspace.create_bytes("plan.md", b"add a and b\n")
        return ToolResult(content="planned")


class _WriteSolution(Tool):
    spec = ToolSpec(name="write_solution", parallel_safe=False, workspace_mutation=True)

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        assert ctx.workspace is not None
        assert (await ctx.workspace.read_bytes("plan.md")).content == b"add a and b\n"
        await ctx.workspace.create_bytes("solution.py", _SOLUTION)
        return ToolResult(content="solved")


class _ReadSeed(Tool):
    spec = ToolSpec(name="read_seed", parallel_safe=False)

    def __init__(self) -> None:
        self.read: list[bytes] = []

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        assert ctx.workspace is not None
        self.read.append((await ctx.workspace.read_bytes("seed.txt")).content)
        return ToolResult(content="read")


def _call(tool: str) -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.tool_call(id=f"call-{tool}", name=tool, arguments={}),
        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
    ]


def _done() -> list[ModelStreamEvent]:
    return [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed({})]


def _app(
    tmp_path: Path,
    *,
    turns: list[list[ModelStreamEvent]],
    allocation_scope: str = "session",
    gate: asyncio.Event | None = None,
) -> tuple[CayuApp, _RunnerFactory]:
    factory = _RunnerFactory(tmp_path, gate=gate)
    app = CayuApp(enable_logging=False)
    app.register_provider(ScriptedModelProvider(turns), default=True)
    app.register_environment_factory(
        EnvironmentSpec(name="sandbox"), factory, default=True, allocation_scope=allocation_scope
    )
    app.register_agent(AgentSpec(name="planner", model="scripted"), tools=[_WritePlan()])
    app.register_agent(AgentSpec(name="solver", model="scripted"), tools=[_WriteSolution()])
    factory.reader = _ReadSeed()
    app.register_agent(AgentSpec(name="reader", model="scripted"), tools=[factory.reader])
    return app, factory


def _shared(app: CayuApp):
    return app._environment_lifecycle.shared_allocations


class _PlanSolveTest(WorkflowBase):
    spec = WorkflowSpec(name="plan-solve-test")

    def __init__(self, app, factory, *, scope: str | None = None) -> None:
        super().__init__(app)
        self._factory = factory
        self._scope = scope
        self.result: CommandResult | None = None
        self.seen: dict[str, object] = {}

    async def run(self, session_id: str):
        ctx = self.context(session_id)
        options = None if self._scope is None else StepRunOptions(allocation_scope=self._scope)
        yield await ctx.start()
        await step(ctx, agent="planner", step_id="plan", prompt="plan", run_options=options)
        await step(ctx, agent="solver", step_id="solve", prompt="solve", run_options=options)
        self.result = await command(
            ctx,
            step_id="tests",
            argv=[PY, "solution.py"],
            allocation_scope=self._scope,
        )
        # The command ran and detached; it released nothing.
        self.seen["finalized_after_command"] = self._factory.finalizations()
        self.seen["live_after_command"] = len(_shared(self.app).live_keys())
        yield await ctx.completed({"passed": self.result.passed})


@pytest.mark.parametrize(
    ("registered", "per_call"),
    [("parent", None), ("causal_budget", None), ("session", "parent")],
)
def test_a_command_runs_in_the_sandbox_the_steps_wrote_to(tmp_path, registered, per_call):
    turns = [_call("write_plan"), _done(), _call("write_solution"), _done()]
    app, factory = _app(tmp_path, turns=turns, allocation_scope=registered)
    workflow = _PlanSolveTest(app, factory, scope=per_call)

    async def scenario():
        return [event async for event in workflow.run("run-tests")]

    events = asyncio.run(scenario())
    assert events[-1].type == EventType.WORKFLOW_COMPLETED, events[-1].payload
    # One allocation for both steps and the command, owned by the workflow run.
    assert _owners(factory) == ["run-tests"]
    result = workflow.result
    assert result is not None and result.passed, result
    assert result.stdout.strip() == "add a and b"
    assert result.environment_name == "sandbox"
    assert not result.replayed
    # Only the run's completion released it, after the command had finished.
    assert workflow.seen == {"finalized_after_command": [], "live_after_command": 1}
    assert factory.finalizations() == ["completed"]
    assert not _shared(app).live_keys()


def test_a_replayed_command_does_not_attach_again(tmp_path):
    turns = [_call("write_plan"), _done(), _call("write_solution"), _done()]
    app, factory = _app(tmp_path, turns=turns, allocation_scope="parent")
    first = _PlanSolveTest(app, factory)
    again = _PlanSolveTest(app, factory)

    async def scenario():
        before = [event async for event in first.run("run-replay")]
        after = [event async for event in again.run("run-replay")]
        return before, after

    _, after = asyncio.run(scenario())
    assert after[-1].type == EventType.WORKFLOW_COMPLETED, after[-1].payload
    assert again.result is not None and again.result.replayed
    assert first.result is not None and again.result.stdout == first.result.stdout
    # Every step replayed, so nothing asked for a second allocation.
    assert _owners(factory) == ["run-replay"]
    assert factory.finalizations() == ["completed"]


class _CommandFirst(WorkflowBase):
    spec = WorkflowSpec(name="command-first")

    def __init__(self, app, factory) -> None:
        super().__init__(app)
        self._factory = factory
        self.seen: dict[str, object] = {}

    async def run(self, session_id: str):
        ctx = self.context(session_id)
        yield await ctx.start()
        seeded = await command(
            ctx, step_id="seed", argv=[PY, "-c", "open('seed.txt', 'w').write('from the command')"]
        )
        assert seeded.passed, seeded
        self.seen["requests_after_command"] = len(self._factory.requests)
        self.seen["finalized_after_command"] = self._factory.finalizations()
        await step(ctx, agent="reader", step_id="read", prompt="read")
        yield await ctx.completed()


def test_a_command_first_creates_the_allocation_the_steps_then_use(tmp_path):
    app, factory = _app(tmp_path, turns=[_call("read_seed"), _done()], allocation_scope="parent")
    workflow = _CommandFirst(app, factory)

    async def scenario():
        return [event async for event in workflow.run("run-first")]

    events = asyncio.run(scenario())
    assert events[-1].type == EventType.WORKFLOW_COMPLETED, events[-1].payload
    # The command created the allocation through the step path: requested for
    # the workflow run, under a fresh shared identity, and kept after the command.
    (request,) = factory.requests
    assert _owners(factory) == ["run-first"]
    assert request.agent_name == "command-first"
    assert request.parent_session_id is None
    assert workflow.seen == {"requests_after_command": 1, "finalized_after_command": []}
    # The step read what the command wrote, so it ran in the same allocation.
    assert factory.reader is not None and factory.reader.read == [b"from the command"]
    assert factory.finalizations() == ["completed"]


def test_a_command_is_refused_on_a_per_session_factory(tmp_path):
    app, factory = _app(tmp_path, turns=[])

    async def scenario():
        ctx = _CommandFirst(app, factory).context("run-refused")
        await ctx.start()
        with pytest.raises(ValueError, match="no session to own an allocation"):
            await command(ctx, step_id="check", argv=[PY, "-c", "pass"])
        with pytest.raises(ValueError, match="no session to own an allocation"):
            await command(
                ctx, step_id="check-session", argv=[PY, "-c", "pass"], allocation_scope="session"
            )
        with pytest.raises(ValueError, match="allocation_scope must be one of"):
            await command(ctx, step_id="check-x", argv=[PY, "-c", "pass"], allocation_scope="x")

    asyncio.run(scenario())
    assert factory.requests == []
    assert not _shared(app).live_keys()


def test_a_running_command_holds_the_allocation_but_never_releases_it(tmp_path):
    gate = asyncio.Event()
    app, factory = _app(tmp_path, turns=[], allocation_scope="parent", gate=gate)

    async def scenario():
        ctx = _CommandFirst(app, factory).context("run-hold")
        await ctx.start()
        running = asyncio.ensure_future(
            command(ctx, step_id="slow", argv=[PY, "-c", "print('ran')"])
        )
        for _ in range(3000):
            if factory.runners or running.done():
                break
            await asyncio.sleep(0.01)
        assert factory.runners, running
        await asyncio.wait_for(factory.runners[0].entered.wait(), timeout=30)
        # The owner ends while the command still runs: the release waits for it.
        await _shared(app).end_owner("run-hold", outcome="completed", attempt_id=ctx.attempt_id)
        assert factory.finalizations() == []
        gate.set()
        result = await running
        assert result.passed and result.stdout.strip() == "ran"
        # Detaching let the owner's release run; the command itself released nothing.
        for _ in range(100):
            if factory.finalizations():
                break
            await asyncio.sleep(0.01)
        assert not _shared(app).live_keys()

    asyncio.run(scenario())
    assert factory.finalizations() == ["completed"]
    assert factory.discards == 0


def test_a_command_after_its_owner_ended_is_refused(tmp_path):
    app, factory = _app(tmp_path, turns=[], allocation_scope="parent")

    async def scenario():
        ctx = _CommandFirst(app, factory).context("run-ended")
        await ctx.start()
        first = await command(ctx, step_id="one", argv=[PY, "-c", "pass"])
        assert first.passed
        await ctx.release_environments()
        assert factory.finalizations() == ["completed"]
        # A later command opens the run's scope again, like a later step would,
        # and gets a fresh allocation the run releases on completion.
        second = await command(ctx, step_id="two", argv=[PY, "-c", "pass"])
        assert second.passed
        await ctx.completed()

    asyncio.run(scenario())
    assert _owners(factory) == ["run-ended", "run-ended"]
    assert factory.finalizations() == ["completed", "completed"]


def test_a_factory_failure_is_a_step_error_without_a_journaled_start(tmp_path):
    class _Broken(_RunnerFactory):
        async def create(self, request):
            self.requests.append(request)
            raise RuntimeError("provider is down")

    factory = _Broken(tmp_path)
    app = CayuApp(enable_logging=False)
    app.register_environment_factory(
        EnvironmentSpec(name="sandbox"), factory, default=True, allocation_scope="parent"
    )

    async def scenario():
        ctx = _CommandFirst(app, factory).context("run-broken")
        await ctx.start()
        with pytest.raises(StepError, match="could not use the shared allocation") as raised:
            await command(ctx, step_id="check", argv=[PY, "-c", "pass"])
        assert isinstance(raised.value.__cause__, RuntimeError)
        events = await app.session_store.load_events("run-broken")
        assert not [event for event in events if event.type == EventType.WORKFLOW_STEP_STARTED]

    asyncio.run(scenario())
    assert len(factory.requests) == 1
