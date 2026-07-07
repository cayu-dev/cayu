"""Tests for ``run_to_completion`` / ``RunOutcome``."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from cayu import (
    AgentSpec,
    CayuApp,
    Event,
    Message,
    ModelStreamEvent,
    RunOutcome,
    RunRequest,
    ScriptedModelProvider,
    SessionStatus,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
    run_to_completion,
)
from cayu.providers import ModelProvider, ModelRequest


def _request() -> RunRequest:
    return RunRequest(
        agent_name="assistant", session_id="s1", messages=[Message.text("user", "hi")]
    )


def test_run_to_completion_returns_final_text_and_ok() -> None:
    app = CayuApp()
    app.register_provider(
        ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.text_delta("Hello "),
                    ModelStreamEvent.text_delta("world"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
            ]
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="scripted-model"))

    outcome = asyncio.run(run_to_completion(app, _request()))

    assert isinstance(outcome, RunOutcome)
    assert outcome.ok
    assert outcome.status is SessionStatus.COMPLETED
    assert outcome.status == "completed"
    assert outcome.final_text == "Hello world"
    assert outcome.error is None
    assert outcome.session_id == "s1"


class _NoopTool(Tool):
    spec = ToolSpec(
        name="noop",
        description="noop",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
    )

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        return ToolResult(content="ok")


def test_run_to_completion_uses_latest_model_turn_even_when_empty() -> None:
    app = CayuApp()
    app.register_provider(
        ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.text_delta("draft-before-tool"),
                    ModelStreamEvent.tool_call(id="call_1", name="noop", arguments={}),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="scripted-model"), tools=[_NoopTool()])

    outcome = asyncio.run(run_to_completion(app, _request()))

    assert outcome.ok
    assert outcome.final_text == ""


class _FailsAfterToolProvider(ModelProvider):
    name = "fails_after_tool"

    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.calls += 1
        if self.calls == 1:
            yield ModelStreamEvent.text_delta("draft-before-tool")
            yield ModelStreamEvent.tool_call(id="call_1", name="noop", arguments={})
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        yield ModelStreamEvent.error("boom")


def test_run_to_completion_clears_prior_text_when_later_model_turn_fails() -> None:
    app = CayuApp()
    app.register_provider(_FailsAfterToolProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="scripted-model"), tools=[_NoopTool()])

    outcome = asyncio.run(run_to_completion(app, _request()))

    assert not outcome.ok
    assert outcome.status == "failed"
    assert outcome.final_text == ""
    assert outcome.error == "boom"


class _BoomProvider(ModelProvider):
    name = "boom"

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        raise RuntimeError("boom")
        yield  # pragma: no cover - makes this an async generator


def test_run_to_completion_reports_failure_without_raising() -> None:
    app = CayuApp()
    app.register_provider(_BoomProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="boom-model"))

    outcome = asyncio.run(run_to_completion(app, _request()))

    assert not outcome.ok
    assert outcome.status is SessionStatus.FAILED
    assert outcome.error is not None


def test_run_to_completion_reports_setup_failure_without_raising() -> None:
    app = CayuApp()

    outcome = asyncio.run(run_to_completion(app, _request()))

    assert not outcome.ok
    assert outcome.status is SessionStatus.FAILED
    assert outcome.error is not None
    assert "not registered" in outcome.error


class _PlainStrFailureApp:
    """Stand-in whose ``run`` replays a failure event exactly as a non-model JSON channel
    (webhook / SSE / JSONL) hands it back — ``.type`` is a plain ``str``, not the validator-
    coerced enum member. ``run_to_completion`` must still classify the run as FAILED.
    """

    async def run(self, request: RunRequest) -> AsyncIterator[Event]:
        # model_construct bypasses the Event validator, so `.type` stays a plain str.
        yield Event.model_construct(
            type="session.failed",
            session_id=request.session_id or "s1",
            payload={"error": "boom"},
        )


def test_run_to_completion_detects_failure_when_event_type_is_plain_str() -> None:
    # #125 footgun 1: `is` would silently miss a plain-str `.type` and leave the run stuck at the
    # default INTERRUPTED (the "document hung in processing forever" symptom); `==` detects FAILED.
    outcome = asyncio.run(run_to_completion(_PlainStrFailureApp(), _request()))  # type: ignore[arg-type]

    assert not outcome.ok
    assert outcome.status is SessionStatus.FAILED
    assert outcome.status == "failed"
    assert outcome.error == "boom"
