"""Foreground subagent cancellation: bounded cleanup that owns only its own cancel."""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from cayu.core.events import Event, EventType
from cayu.core.tools import ToolContext
from cayu.tools import subagents
from cayu.tools.subagents import (
    SubagentSpec,
    SubagentTool,
    _uncancel_current_task,
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


def test_uncancel_current_task_consumes_only_one_pending_request():
    async def run():
        task = asyncio.current_task()
        assert task is not None
        task.cancel()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.sleep(0)
        assert task.cancelling() == 2
        _uncancel_current_task()
        assert task.cancelling() == 1
        # Second call keeps consuming one at a time and stays guarded at zero.
        _uncancel_current_task()
        _uncancel_current_task()
        assert task.cancelling() == 0

    asyncio.run(run())


def test_subagent_cancellation_keeps_outer_cancellation_requests():
    runtime = _BlockingChildRuntime()
    tool = _foreground_tool(runtime)

    async def run():
        task, exc = await _cancel_tool_run(tool, runtime, cancels=2)
        return task, exc

    task, exc = asyncio.run(run())
    assert task.cancelled()
    # The tool consumed exactly the one cancellation it caught; the second,
    # outer-owned request must survive (the old drain loop stripped it too).
    assert task.cancelling() == 1
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
    assert task.cancelling() == 0
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
    artifacts = getattr(exc, "artifacts", [])
    assert len(artifacts) == 1
    assert artifacts[0]["type"] == "cayu.subagent_cleanup_error.v1"
    assert artifacts[0]["error_type"] == "TimeoutError"
    assert runtime.interrupt_started.is_set()
    # The backstop still tears down the child collector after the timeout.
    assert runtime.child_cancelled.is_set()
