"""Foreground subagent cancellation: bounded cleanup that owns only its own cancel."""

from __future__ import annotations

import asyncio

import pytest

from cayu.core.events import Event, EventType
from cayu.core.tools import ToolContext
from cayu.tools import subagents
from cayu.tools.subagents import (
    SubagentSpec,
    SubagentTool,
)


class _BlockingChildRuntime:
    """SubagentRuntime stub whose child run blocks until cancelled."""

    def __init__(self, *, interrupt_hangs: bool = False) -> None:
        self.child_running = asyncio.Event()
        self.child_cancelled = asyncio.Event()
        self.interrupt_started = asyncio.Event()
        self._interrupt_hangs = interrupt_hangs

    def run(self, request):
        async def events():
            self.child_running.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.child_cancelled.set()
                raise
            yield  # pragma: no cover - unreachable

        return events()

    def interrupt_session(self, request):
        async def events():
            self.interrupt_started.set()
            if self._interrupt_hangs:
                await asyncio.Event().wait()
            yield Event(type=EventType.SESSION_INTERRUPTED, session_id=request.session_id)

        return events()


def _foreground_tool(runtime) -> SubagentTool:
    return SubagentTool(
        runtime,
        agents={"reviewer": SubagentSpec(agent_name="reviewer")},
    )


async def _cancel_tool_run(
    tool: SubagentTool,
    runtime: _BlockingChildRuntime,
    *,
    cancels: int = 1,
) -> tuple[asyncio.Task, asyncio.CancelledError]:
    """Run the tool, cancel it once running, and capture the raised instance."""
    captured: dict[str, asyncio.CancelledError] = {}

    async def call():
        try:
            await tool.run(
                ToolContext(session_id="sess_cancel_parent"),
                {"agent": "reviewer", "task": "review"},
            )
        except asyncio.CancelledError as exc:
            captured["exc"] = exc
            raise

    task = asyncio.create_task(call())
    await asyncio.wait_for(runtime.child_running.wait(), timeout=1)
    for _ in range(cancels):
        task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)
    return task, captured["exc"]


def test_subagent_cleanup_preserves_enclosing_timeout_classification():
    async def run():
        runtime = _BlockingChildRuntime()
        tool = _foreground_tool(runtime)
        task = asyncio.current_task()
        assert task is not None
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.05):
                await tool.run(
                    ToolContext(session_id="timeout-parent"),
                    {"agent": "reviewer", "task": "review"},
                )
        assert task.cancelling() == 0
        assert runtime.interrupt_started.is_set()
        assert runtime.child_cancelled.is_set()

    asyncio.run(run())


def test_subagent_cancellation_keeps_outer_cancellation_requests():
    runtime = _BlockingChildRuntime()
    tool = _foreground_tool(runtime)

    async def run():
        task, exc = await _cancel_tool_run(tool, runtime, cancels=2)
        return task, exc

    task, exc = asyncio.run(run())
    assert task.cancelled()
    # Neither the delivered request nor the additional request is consumed by
    # this re-raising cleanup boundary.
    assert task.cancelling() == 2
    assert getattr(exc, "artifacts", []) == []
    assert runtime.interrupt_started.is_set()
    assert runtime.child_cancelled.is_set()


def test_subagent_cancellation_cleanup_interrupts_child_and_reraises():
    runtime = _BlockingChildRuntime()
    tool = _foreground_tool(runtime)

    async def run():
        return await _cancel_tool_run(tool, runtime)

    task, exc = asyncio.run(run())
    assert task.cancelled()
    assert task.cancelling() == 1
    assert getattr(exc, "artifacts", []) == []
    assert runtime.interrupt_started.is_set()
    assert runtime.child_cancelled.is_set()


def test_subagent_cancellation_cleanup_is_bounded_when_interrupt_hangs(monkeypatch):
    monkeypatch.setattr(subagents, "SUBAGENT_CANCEL_CLEANUP_TIMEOUT_S", 0.05)
    runtime = _BlockingChildRuntime(interrupt_hangs=True)
    tool = _foreground_tool(runtime)

    async def run():
        return await _cancel_tool_run(tool, runtime)

    task, exc = asyncio.run(run())
    assert task.cancelled()
    assert task.cancelling() == 1
    artifacts = getattr(exc, "artifacts", [])
    assert len(artifacts) == 1
    assert artifacts[0]["type"] == "cayu.subagent_cleanup_error.v1"
    assert artifacts[0]["error_type"] == "TimeoutError"
    assert runtime.interrupt_started.is_set()
    # The backstop still tears down the child collector after the timeout.
    assert runtime.child_cancelled.is_set()
