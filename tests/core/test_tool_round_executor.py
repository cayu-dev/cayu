"""Focused tests for the extracted tool-round execution boundary."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import pytest

from cayu.core import AgentSpec, Event, EventType, Message
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import (
    CayuApp,
    InMemorySessionStore,
    RetryPolicy,
    RunLimits,
    RunRequest,
    Session,
    SessionStatus,
)
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime._run_limits import RunLimitGate
from cayu.runtime._session_control import SessionInterruptedByRequest
from cayu.runtime._tool_round_executor import ToolRoundRun, _copy_agent_spec


class _FakeProvider(ModelProvider):
    name = "fake"

    def __init__(self, events: list[ModelStreamEvent]) -> None:
        self.events = events

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        for event in self.events:
            yield event


class _SideEffectTool(Tool):
    spec = ToolSpec(
        name="side_effect",
        description="Record execution.",
        input_schema={"type": "object", "properties": {}},
    )

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.calls.append(args)
        return ToolResult(content="recorded")


def _app_with_completed_session(
    session_id: str,
    *,
    total_tokens: int = 11,
) -> tuple[CayuApp, InMemorySessionStore, _SideEffectTool]:
    store = InMemorySessionStore()
    provider = _FakeProvider(
        [
            ModelStreamEvent.text_delta("final answer"),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {
                        "input_tokens": total_tokens - 4,
                        "output_tokens": 4,
                        "total_tokens": total_tokens,
                    },
                }
            ),
        ]
    )
    tool = _SideEffectTool()
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])

    async def run() -> None:
        async for _ in app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "answer")],
            )
        ):
            pass

    asyncio.run(run())
    return app, store, tool


def _limit_gate(
    app: CayuApp,
    session,
    *,
    limits: RunLimits,
) -> RunLimitGate:
    return RunLimitGate(
        app._run_limit_controller,
        session=session,
        agent_name="assistant",
        environment_name=None,
        limits=limits,
        budget_limits=(),
        run_started_at=time.monotonic(),
        run_baseline=None,
        budget_baseline_events=[],
        budget_notify_events=[],
    )


def _tool_round_run(
    app: CayuApp,
    session: Session,
    *,
    limits: RunLimits,
) -> ToolRoundRun:
    return app._tool_round_executor.create_run(
        session=session,
        registered_agent=app._get_registered_agent("assistant"),
        registered_environment=None,
        environment_name=None,
        limit_gate=_limit_gate(app, session, limits=limits),
        request_metadata={},
        task_id=None,
        structured_output=None,
        thinking=None,
        max_steps=16,
        limits=RunLimits(),
        budget_limits=(),
        retry_policy=RetryPolicy(),
        run_started_at=time.monotonic(),
        turn_usage_tracker=None,
        active_run=None,
    )


def _tool_call(call_id: str = "call_1") -> runtime_records.ToolCallRequest:
    return runtime_records.ToolCallRequest(id=call_id, name="side_effect", arguments={})


def test_tool_round_agent_copy_rejects_agent_spec_subclasses() -> None:
    class _DerivedAgentSpec(AgentSpec):
        pass

    with pytest.raises(TypeError, match="Agent registration requires an AgentSpec"):
        _copy_agent_spec(_DerivedAgentSpec(name="assistant", model="fake-model"))


def test_tool_round_interrupt_close_ignores_unrequested_cancellation():
    app, store, _ = _app_with_completed_session("sess_guard_cancel")

    async def scenario() -> tuple[list[Event], list[Message]]:
        session = await store.load("sess_guard_cancel")
        assert session is not None
        runner = _tool_round_run(app, session, limits=RunLimits())
        messages: list[Message] = []
        events = [
            event
            async for event in runner.close_after_interrupt(
                asyncio.CancelledError(),
                messages=messages,
                tool_calls=[_tool_call()],
                tool_outcomes=[],
                tool_round_id="round_1",
            )
        ]
        return events, messages

    events, messages = asyncio.run(scenario())

    assert events == []
    assert messages == []
    transcript = asyncio.run(store.load_transcript("sess_guard_cancel"))
    assert [message.role for message in transcript] == ["user", "assistant"]


def test_tool_round_interrupt_close_persists_missing_results():
    app, store, _ = _app_with_completed_session("sess_guard_interrupt")

    async def scenario() -> tuple[list[Event], list[Message]]:
        session = await store.load("sess_guard_interrupt")
        assert session is not None
        runner = _tool_round_run(app, session, limits=RunLimits())
        messages: list[Message] = []
        events = [
            event
            async for event in runner.close_after_interrupt(
                SessionInterruptedByRequest(session.id),
                messages=messages,
                tool_calls=[_tool_call()],
                tool_outcomes=[],
                tool_round_id="round_1",
            )
        ]
        return events, messages

    events, messages = asyncio.run(scenario())

    assert [event.type for event in events] == [EventType.TOOL_CALL_FAILED]
    assert events[0].payload["tool_call_id"] == "call_1"
    assert events[0].payload["tool_round_id"] == "round_1"
    assert [message.role for message in messages] == ["tool"]
    transcript = asyncio.run(store.load_transcript("sess_guard_interrupt"))
    assert [message.role for message in transcript] == ["user", "assistant", "tool"]


def test_tool_round_interrupt_close_handles_requested_cancellation():
    app, store, _ = _app_with_completed_session("sess_guard_cancel_interrupt")

    async def scenario() -> tuple[list[Event], list[Message]]:
        session = await store.update_status(
            "sess_guard_cancel_interrupt",
            SessionStatus.INTERRUPTING,
        )
        runner = _tool_round_run(app, session, limits=RunLimits())
        messages: list[Message] = []
        events = [
            event
            async for event in runner.close_after_interrupt(
                asyncio.CancelledError(),
                messages=messages,
                tool_calls=[_tool_call()],
                tool_outcomes=[],
                tool_round_id="round_1",
            )
        ]
        return events, messages

    events, messages = asyncio.run(scenario())

    assert [event.type for event in events] == [EventType.TOOL_CALL_FAILED]
    assert events[0].payload["tool_call_id"] == "call_1"
    assert [message.role for message in messages] == ["tool"]


def test_tool_round_interrupt_close_rejects_unrelated_exceptions():
    app, store, _ = _app_with_completed_session("sess_guard_type_error")

    async def scenario() -> None:
        session = await store.load("sess_guard_type_error")
        assert session is not None
        runner = _tool_round_run(app, session, limits=RunLimits())
        async for _ in runner.close_after_interrupt(
            ValueError("not an interrupt"),
            messages=[],
            tool_calls=[],
            tool_outcomes=[],
            tool_round_id=None,
        ):
            pass

    with pytest.raises(TypeError, match="Unsupported interrupt exception"):
        asyncio.run(scenario())


def test_tool_round_runner_stops_for_limit_before_tool_side_effects():
    app, store, tool = _app_with_completed_session("sess_runner_limit")

    async def scenario() -> tuple[list[Event], bool]:
        session = await store.load("sess_runner_limit")
        assert session is not None
        runner = _tool_round_run(
            app,
            session,
            limits=RunLimits(max_total_tokens=10),
        )
        events = [
            event
            async for event in runner.run(
                messages=[],
                tool_calls=[_tool_call()],
                tool_round_id="round_1",
            )
        ]
        return events, runner.stopped_for_limit

    events, stopped_for_limit = asyncio.run(scenario())

    assert stopped_for_limit is True
    assert [event.type for event in events] == [
        EventType.SESSION_LIMIT_REACHED,
        EventType.TOOL_CALL_FAILED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert events[1].payload["reason"] == "limit_reached"
    assert tool.calls == []


def test_tool_round_runner_executes_tool_round_and_persists_results():
    app, store, tool = _app_with_completed_session("sess_runner_execute")

    async def scenario() -> tuple[list[Event], list[Message], bool]:
        session = await store.load("sess_runner_execute")
        assert session is not None
        runner = _tool_round_run(
            app,
            session,
            limits=RunLimits(max_total_tokens=100),
        )
        messages: list[Message] = []
        events = [
            event
            async for event in runner.run(
                messages=messages,
                tool_calls=[_tool_call()],
                tool_round_id="round_1",
            )
        ]
        return events, messages, runner.stopped_for_limit

    events, messages, stopped_for_limit = asyncio.run(scenario())

    assert stopped_for_limit is False
    assert [event.type for event in events] == [
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
    ]
    assert tool.calls == [{}]
    assert [message.role for message in messages] == ["tool"]
    transcript = asyncio.run(store.load_transcript("sess_runner_execute"))
    assert transcript[-1].role == "tool"
