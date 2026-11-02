"""Parallel steps and commands sharing an allocation take turns only when they ask to."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from tests.environments.test_shared_allocation_commands import _call, _done, _RunnerFactory

from cayu import AgentSpec, CayuApp, EventType
from cayu.environments import EnvironmentSpec
from cayu.evals import ScriptedModelProvider
from cayu.messages import MessageRole
from cayu.providers.base import ModelRequest
from cayu.runtime._shared_allocation_writers import SharedAllocationWriterTurns
from cayu.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from cayu.workflows import (
    CommandResult,
    StepRunOptions,
    WorkflowBase,
    WorkflowSpec,
    command,
    parallel,
    step,
)

PY = sys.executable


def _respond(request: ModelRequest):
    """Each writer session calls its tool once, then finishes."""

    if any(message.role == MessageRole.TOOL for message in request.messages):
        return _done()
    return _call("write")


class _Write(Tool):
    """Records when each session enters and leaves its write, and what it saw.

    With ``meet`` set, each write waits until that many sessions are inside
    at once, which only happens when nothing serializes them.
    """

    spec = ToolSpec(name="write", parallel_safe=False, workspace_mutation=True)

    def __init__(self, *, meet: int = 0, hold: asyncio.Event | None = None) -> None:
        self.log: list[tuple[str, str]] = []
        self.inside = 0
        self.met = asyncio.Event()
        self.entered = asyncio.Event()
        self._meet = meet
        self._hold = hold

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        assert ctx.workspace is not None
        self.log.append(("enter", ctx.session_id))
        self.inside += 1
        self.entered.set()
        try:
            await ctx.workspace.write_bytes("shared.txt", f"half {ctx.session_id}\n".encode())
            if self._meet:
                if self.inside >= self._meet:
                    self.met.set()
                await asyncio.wait_for(self.met.wait(), timeout=10)
            if self._hold is not None:
                await self._hold.wait()
            for _ in range(5):
                await asyncio.sleep(0.01)
            await ctx.workspace.write_bytes("shared.txt", f"done {ctx.session_id}\n".encode())
        finally:
            self.inside -= 1
            self.log.append(("exit", ctx.session_id))
        return ToolResult(content="written")


def _app(tmp_path: Path, tool: _Write) -> tuple[CayuApp, _RunnerFactory]:
    factory = _RunnerFactory(tmp_path)
    app = CayuApp(enable_logging=False)
    app.register_provider(ScriptedModelProvider(response_factory=_respond), default=True)
    app.register_environment_factory(
        EnvironmentSpec(name="sandbox"), factory, default=True, allocation_scope="parent"
    )
    app.register_agent(AgentSpec(name="writer", model="scripted"), tools=[tool])
    return app, factory


def _turns(app: CayuApp) -> SharedAllocationWriterTurns:
    return app._environment_lifecycle.shared_allocations.writer_turns


class _TwoWriters(WorkflowBase):
    spec = WorkflowSpec(name="two-writers")

    def __init__(self, app, *, serialize: bool) -> None:
        super().__init__(app)
        self._options = StepRunOptions(serialize_writers=serialize)
        self.outcome = None

    async def run(self, session_id: str):
        ctx = self.context(session_id)
        yield await ctx.start()
        self.outcome = await parallel(
            [
                step(ctx, agent="writer", step_id=name, prompt=name, run_options=self._options)
                for name in ("a", "b")
            ]
        )
        yield await ctx.completed()


def _overlapped(log: list[tuple[str, str]]) -> bool:
    inside = 0
    for kind, _ in log:
        inside += 1 if kind == "enter" else -1
        if inside > 1:
            return True
    return False


def test_serialized_parallel_steps_never_overlap(tmp_path):
    tool = _Write()
    app, factory = _app(tmp_path, tool)
    workflow = _TwoWriters(app, serialize=True)

    async def scenario():
        return [event async for event in workflow.run("run-1")]

    events = asyncio.run(scenario())
    assert events[-1].type == EventType.WORKFLOW_COMPLETED, events[-1].payload
    assert workflow.outcome is not None and not workflow.outcome.failures
    # Both steps worked in the one shared sandbox, one after the other.
    assert len(factory.requests) == 1
    kinds = [kind for kind, _ in tool.log]
    assert kinds == ["enter", "exit", "enter", "exit"]
    assert tool.log[0][1] == tool.log[1][1] != tool.log[2][1] == tool.log[3][1]
    assert not _overlapped(tool.log)
    assert _turns(app).waiting("run-1") == 0


def test_parallel_steps_still_overlap_without_the_option(tmp_path):
    tool = _Write(meet=2)
    app, factory = _app(tmp_path, tool)
    workflow = _TwoWriters(app, serialize=False)

    async def scenario():
        return [event async for event in workflow.run("run-1")]

    events = asyncio.run(scenario())
    assert events[-1].type == EventType.WORKFLOW_COMPLETED, events[-1].payload
    assert workflow.outcome is not None and not workflow.outcome.failures
    assert len(factory.requests) == 1
    # Both were inside their write at once: the default still races.
    assert [kind for kind, _ in tool.log][:2] == ["enter", "enter"]
    assert _overlapped(tool.log)


def test_cancelling_a_run_releases_the_turn_it_held_and_the_one_it_waited_for(tmp_path):
    hold = asyncio.Event()
    tool = _Write(hold=hold)
    app, _ = _app(tmp_path, tool)
    workflow = _TwoWriters(app, serialize=True)
    seen: dict[str, object] = {}

    async def drain() -> None:
        async for _ in workflow.run("run-1"):
            pass

    async def scenario():
        task = asyncio.create_task(drain())
        await asyncio.wait_for(tool.entered.wait(), timeout=10)
        # One step holds the turn inside its write; the other waits for it.
        seen["holder"] = _turns(app).holder("run-1")
        seen["waiting"] = _turns(app).waiting("run-1")
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            seen["cancelled"] = True
        seen["holder_after"] = _turns(app).holder("run-1")
        seen["waiting_after"] = _turns(app).waiting("run-1")
        # The run's turn is free again for whatever runs next.
        async with asyncio.timeout(1):
            async with _turns(app).turn("run-1", holder="next"):
                seen["next"] = _turns(app).holder("run-1")
        await app.close_idle_environment_resources(timeout_s=5)

    asyncio.run(scenario())
    assert seen["cancelled"] is True
    assert seen["holder"] in {"a", "b"}
    assert seen["waiting"] == 2
    assert seen["holder_after"] is None
    assert seen["waiting_after"] == 0
    assert seen["next"] == "next"
    # Only the first writer got in; the cancelled waiter never started.
    assert [kind for kind, _ in tool.log] == ["enter", "exit"]


def test_a_command_waits_for_the_turn_a_serialized_step_holds(tmp_path):
    hold = asyncio.Event()
    tool = _Write(hold=hold)
    app, factory = _app(tmp_path, tool)
    seen: dict[str, object] = {}
    results: list[CommandResult] = []

    class StepThenCommand(WorkflowBase):
        spec = WorkflowSpec(name="step-and-command")

        async def run(self, session_id: str):
            ctx = self.context(session_id)
            yield await ctx.start()
            options = StepRunOptions(serialize_writers=True)
            writing = asyncio.ensure_future(
                step(ctx, agent="writer", step_id="a", prompt="a", run_options=options)
            )
            await asyncio.wait_for(tool.entered.wait(), timeout=10)
            checking = asyncio.ensure_future(
                command(
                    ctx,
                    step_id="check",
                    argv=[PY, "-c", "print(open('shared.txt').read().split()[0])"],
                    serialize_writers=True,
                )
            )
            for _ in range(10):
                await asyncio.sleep(0.01)
            seen["holder"] = _turns(app).holder(session_id)
            seen["waiting"] = _turns(app).waiting(session_id)
            seen["command_ran_early"] = any(runner.entered.is_set() for runner in factory.runners)
            hold.set()
            await writing
            results.append(await checking)
            yield await ctx.completed()

    async def scenario():
        return [event async for event in StepThenCommand(app).run("run-1")]

    events = asyncio.run(scenario())
    assert events[-1].type == EventType.WORKFLOW_COMPLETED, events[-1].payload
    assert seen["holder"] == "a"
    assert seen["waiting"] == 2
    assert seen["command_ran_early"] is False
    (result,) = results
    # The command ran after the step's write had finished, in the same sandbox.
    assert result.passed, result
    assert result.stdout.strip() == "done"
    assert len(factory.requests) == 1
