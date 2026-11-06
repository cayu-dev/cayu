"""A command cut off by the run's own time limit settles as timed out when its runner confirms the stop."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from tests.core.test_run_limit_elapsed_in_flight import (
    LIMIT_SECONDS,
    RESERVE_SECONDS,
    SLOW_SECONDS,
    SlowStepProvider,
    VirtualClockLoop,
    only,
    run_events,
    run_on_virtual_clock,
    tool_result,
    verdict_answer,
    verdict_spec,
)

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.budgets.run_limits import RunLimits
from cayu.configuration import CayuConfig, ToolExecutionConfig
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.events import EventType
from cayu.messages import Message
from cayu.providers import ModelStreamEvent
from cayu.runners.base import ExecCommand, ExecResult, Runner, attach_cancellation_artifacts
from cayu.sessions.requests import RunRequest
from cayu.tools.commands import ExecCommandTool


def kill_receipt(status: str) -> dict[str, Any]:
    return {
        "type": "cayu.runner_cleanup.v1",
        "adapter": "test-sandbox",
        "action": "kill_command",
        "status": status,
        "timeout_s": 5.0,
    }


class SlowCommandRunner(Runner):
    """Runs every command for ``SLOW_SECONDS`` and reports how a cancelled one stopped."""

    def __init__(self, kill_status: str | None) -> None:
        self.kill_status = kill_status
        self.started = 0
        self.cancelled = 0

    async def exec(self, command: ExecCommand, **kwargs: Any) -> ExecResult:
        self.started += 1
        try:
            await asyncio.sleep(SLOW_SECONDS)
        except asyncio.CancelledError as cancellation:
            self.cancelled += 1
            if self.kill_status is not None:
                attach_cancellation_artifacts(cancellation, [kill_receipt(self.kill_status)])
            raise
        return ExecResult(stdout="passed\n")


def exec_call() -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.tool_call(
            id="call-0",
            name="exec_command",
            arguments={"kind": "shell", "shell": "make test", "timeout_s": SLOW_SECONDS * 2},
        ),
        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
    ]


def command_app(
    provider: SlowStepProvider, runner: Runner, *, tool_timeout_seconds: float | None = None
) -> CayuApp:
    app = CayuApp(
        config=CayuConfig(
            tool_execution=ToolExecutionConfig(tool_timeout_seconds=tool_timeout_seconds)
        ),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(EnvironmentSpec(name="sandbox"), runner=runner), default=True
    )
    app.register_agent(
        AgentSpec(name="reviewer", model="fake"),
        tools=[ExecCommandTool(max_timeout_seconds=3_600)],
    )
    return app


def finish_request(session_id: str) -> RunRequest:
    return RunRequest(
        agent_name="reviewer",
        session_id=session_id,
        messages=[Message.text("user", "Review the fix.")],
        max_steps=None,
        limits=RunLimits(
            max_elapsed_seconds=LIMIT_SECONDS,
            on_limit="finish",
            finish_reserve_seconds=RESERVE_SECONDS,
        ),
        structured_output=verdict_spec(),
    )


# Without a tool timeout the call runs in the run's task; with a longer one, in its own.
@pytest.mark.parametrize("tool_timeout_seconds", [None, SLOW_SECONDS * 10])
def test_command_cut_by_the_run_limit_settles_as_timed_out_and_finishes(
    monkeypatch, tool_timeout_seconds
):
    runner = SlowCommandRunner(kill_status="completed")
    provider = SlowStepProvider([exec_call(), verdict_answer()], delays=[])
    app = command_app(provider, runner, tool_timeout_seconds=tool_timeout_seconds)

    async def scenario(loop: VirtualClockLoop) -> None:
        events = await run_events(app, finish_request("command-cut"))
        assert (runner.started, runner.cancelled) == (1, 1)
        stored = await app.session_store.load_events("command-cut")
        assert not any(event.type is EventType.TOOL_EFFECT_OUTCOME_UNKNOWN for event in stored)
        result = await tool_result(app, "command-cut", "call-0")
        cutoff = LIMIT_SECONDS - RESERVE_SECONDS
        assert result["is_error"] is True
        assert result["content"] == (
            f"Tool call stopped after {cutoff} seconds: the run reached its time limit. "
            "The command was stopped before it finished."
        )
        # A settled timeout, not a runtime failure whose effect is unknown.
        assert result["structured"] == {
            "timed_out": True,
            "stopped_by": "run_time_limit",
            "stopped_after_seconds": cutoff,
        }
        failed = [
            event
            for event in stored
            if event.type is EventType.TOOL_CALL_FAILED
            and event.payload.get("tool_call_id") == "call-0"
        ]
        assert len(failed) == 1
        assert "outcome_unknown" not in failed[0].payload
        # The finishing step sees the timed-out call and returns the answer.
        assert len(provider.requests) == 2
        validated = only(events, EventType.STRUCTURED_OUTPUT_VALIDATED)
        terminal = only(events, EventType.SESSION_INTERRUPTED)
        assert events.index(validated) < events.index(terminal)
        assert terminal.payload["reason"] == "limit_reached"
        assert terminal.payload["limit"] == "elapsed_seconds"
        assert terminal.payload["structured_output_pending"] is False
        assert terminal.payload["finish_up"] is True

    run_on_virtual_clock(monkeypatch, scenario)


@pytest.mark.parametrize("kill_status", [None, "failed"])
def test_command_cut_without_a_confirmed_stop_keeps_its_outcome_unknown(monkeypatch, kill_status):
    # No receipt, or a kill the runner could not confirm: the command may still run.
    runner = SlowCommandRunner(kill_status=kill_status)
    provider = SlowStepProvider([exec_call(), verdict_answer()], delays=[])
    app = command_app(provider, runner)

    async def scenario(loop: VirtualClockLoop) -> None:
        events = await run_events(app, finish_request("command-unknown"))
        assert (runner.started, runner.cancelled) == (1, 1)
        stored = await app.session_store.load_events("command-unknown")
        unknown = only(stored, EventType.TOOL_EFFECT_OUTCOME_UNKNOWN)
        assert unknown.payload["failure_evidence"]["classification"] == "timeout"
        assert unknown.payload["failure_evidence"]["settlement"] == "unknown"
        # The call waits for a person instead of a finishing step.
        assert len(provider.requests) == 1
        assert not any(event.type is EventType.STRUCTURED_OUTPUT_VALIDATED for event in events)

    run_on_virtual_clock(monkeypatch, scenario)


class TimeoutHonoringRunner(Runner):
    """Stops a command at its own timeout, the way a sandbox runner does."""

    def __init__(self) -> None:
        self.timeouts: list[int] = []

    async def exec(self, command: ExecCommand, **kwargs: Any) -> ExecResult:
        timeout_s = kwargs["timeout_s"]
        self.timeouts.append(timeout_s)
        if timeout_s >= SLOW_SECONDS:
            await asyncio.sleep(SLOW_SECONDS)
            return ExecResult(stdout="passed\n")
        await asyncio.sleep(timeout_s)
        return ExecResult(
            stdout="collected 40 items\n",
            exit_code=124,
            timed_out=True,
            artifacts=[kill_receipt("completed")],
        )


def test_command_timeout_is_shortened_to_fit_the_run_time_left(monkeypatch):
    runner = TimeoutHonoringRunner()
    provider = SlowStepProvider([exec_call(), verdict_answer()], delays=[])
    app = command_app(provider, runner)

    async def scenario(loop: VirtualClockLoop) -> None:
        events = await run_events(app, finish_request("command-shortened"))
        cutoff = LIMIT_SECONDS - RESERVE_SECONDS
        # The runner stops the command first, leaving it time to return its output.
        assert runner.timeouts == [cutoff - 5]
        result = await tool_result(app, "command-shortened", "call-0")
        assert result["is_error"] is True
        assert result["content"] == (
            f"Command timed out after {cutoff - 5} seconds (shortened from "
            f"{SLOW_SECONDS * 2} seconds to fit the run's time limit). The command was "
            "stopped.\n\nstdout:\ncollected 40 items"
        )
        assert result["structured"]["timed_out"] is True
        assert result["structured"]["timeout_s"] == cutoff - 5
        assert result["structured"]["requested_timeout_s"] == SLOW_SECONDS * 2
        stored = await app.session_store.load_events("command-shortened")
        assert not any(event.type is EventType.TOOL_EFFECT_OUTCOME_UNKNOWN for event in stored)
        # The output came back before the cut, so the next step answers in time.
        assert len(provider.requests) == 2
        only(events, EventType.STRUCTURED_OUTPUT_VALIDATED)
        assert not any(event.type is EventType.SESSION_INTERRUPTED for event in events)

    run_on_virtual_clock(monkeypatch, scenario)


def test_command_is_refused_when_the_run_has_no_time_left_for_it(monkeypatch):
    runner = TimeoutHonoringRunner()
    cutoff = LIMIT_SECONDS - RESERVE_SECONDS
    # The model step ends one second before ordinary tool calls must stop.
    provider = SlowStepProvider([exec_call(), verdict_answer()], delays=[cutoff - 1])
    app = command_app(provider, runner)

    async def scenario(loop: VirtualClockLoop) -> None:
        events = await run_events(app, finish_request("command-refused"))
        assert runner.timeouts == []
        result = await tool_result(app, "command-refused", "call-0")
        assert result["is_error"] is True
        assert result["content"] == (
            "Command not run: the run's time limit stops this call in 1.0 seconds, "
            "too little to run a command."
        )
        assert result["structured"] == {"error": "run_time_exhausted"}
        stored = await app.session_store.load_events("command-refused")
        assert not any(event.type is EventType.TOOL_EFFECT_OUTCOME_UNKNOWN for event in stored)
        only(events, EventType.STRUCTURED_OUTPUT_VALIDATED)

    run_on_virtual_clock(monkeypatch, scenario)
