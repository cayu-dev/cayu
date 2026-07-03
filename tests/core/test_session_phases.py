"""Focused tests for the session-loop phase objects in cayu.runtime.app.

The session loop is composed of three phase objects: ``_LimitGate`` (the
shared run-limit / request-budget check used at every phase boundary),
``_InterruptGuard`` (the session-interrupt matrix around a tool round), and
``_ToolRoundRunner`` (policy planning, approval checkpointing, and tool
execution for one round). These tests exercise the phase objects directly.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import pytest

import cayu.runtime.app as runtime_app_module
from cayu.core import AgentSpec, Event, EventType, Message
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import (
    CayuApp,
    InMemorySessionStore,
    RunLimits,
    RunRequest,
    SessionStatus,
)
from cayu.runtime import _runtime_records as runtime_records


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
) -> runtime_app_module._LimitGate:
    return runtime_app_module._LimitGate(
        app,
        session=session,
        registered_agent=app._get_registered_agent("assistant"),
        registered_environment=None,
        environment_name=None,
        limits=limits,
        budget_limits=(),
        run_started_at=time.monotonic(),
        run_baseline=None,
        budget_baseline_events=[],
        budget_notify_events=[],
    )


def _interrupt_guard(app: CayuApp, session) -> runtime_app_module._InterruptGuard:
    return runtime_app_module._InterruptGuard(
        app,
        session=session,
        registered_agent=app._get_registered_agent("assistant"),
        registered_environment=None,
    )


def _tool_call(call_id: str = "call_1") -> runtime_records.ToolCallRequest:
    return runtime_records.ToolCallRequest(id=call_id, name="side_effect", arguments={})


def test_limit_gate_trips_and_stops_session_when_limit_is_reached():
    app, store, _ = _app_with_completed_session("sess_gate_trip")

    async def scenario() -> tuple[list[Event], bool]:
        session = await store.load("sess_gate_trip")
        assert session is not None
        gate = _limit_gate(app, session, limits=RunLimits(max_total_tokens=10))
        events = [event async for event in gate.evaluate_limits(messages=[])]
        return events, gate.tripped

    events, tripped = asyncio.run(scenario())

    assert tripped is True
    assert [event.type for event in events] == [
        EventType.SESSION_LIMIT_REACHED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert events[0].payload["limit"] == "total_tokens"
    assert events[0].payload["actual"] == 11
    assert events[0].payload["maximum"] == 10
    assert events[1].payload["interruption_type"] == "limit_reached"
    session = asyncio.run(store.load("sess_gate_trip"))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED


def test_limit_gate_does_not_trip_below_limit_and_resets_between_evaluations():
    app, store, _ = _app_with_completed_session("sess_gate_reset")

    async def scenario() -> tuple[list[Event], bool, list[Event], bool, bool]:
        session = await store.load("sess_gate_reset")
        assert session is not None
        below_gate = _limit_gate(app, session, limits=RunLimits(max_total_tokens=100))
        below_events = [event async for event in below_gate.evaluate_limits(messages=[])]
        below_tripped = below_gate.tripped

        tripping_gate = _limit_gate(app, session, limits=RunLimits(max_total_tokens=10))
        async for _ in tripping_gate.evaluate_limits(messages=[]):
            pass
        tripped_after_limits = tripping_gate.tripped
        # A later evaluation must reset the tripped flag: with no budget
        # policy configured the budget check passes cleanly.
        async for _ in tripping_gate.evaluate_budget(messages=[]):
            pass
        return (
            below_events,
            below_tripped,
            [],
            tripped_after_limits,
            tripping_gate.tripped,
        )

    below_events, below_tripped, _, tripped_after_limits, tripped_after_budget = asyncio.run(
        scenario()
    )

    assert below_events == []
    assert below_tripped is False
    assert tripped_after_limits is True
    assert tripped_after_budget is False


def test_interrupt_guard_ignores_cancellation_without_interrupt_request():
    app, store, _ = _app_with_completed_session("sess_guard_cancel")

    async def scenario() -> tuple[list[Event], list[Message]]:
        session = await store.load("sess_guard_cancel")
        assert session is not None
        guard = _interrupt_guard(app, session)
        messages: list[Message] = []
        events = [
            event
            async for event in guard.close_tool_round(
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


def test_interrupt_guard_closes_tool_round_on_interrupt_request():
    app, store, _ = _app_with_completed_session("sess_guard_interrupt")

    async def scenario() -> tuple[list[Event], list[Message]]:
        session = await store.load("sess_guard_interrupt")
        assert session is not None
        guard = _interrupt_guard(app, session)
        messages: list[Message] = []
        events = [
            event
            async for event in guard.close_tool_round(
                runtime_app_module._SessionInterruptedByRequest(session.id),
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


def test_interrupt_guard_closes_tool_round_on_cancellation_with_interrupt_request():
    app, store, _ = _app_with_completed_session("sess_guard_cancel_interrupt")

    async def scenario() -> tuple[list[Event], list[Message]]:
        session = await store.update_status(
            "sess_guard_cancel_interrupt",
            SessionStatus.INTERRUPTING,
        )
        guard = _interrupt_guard(app, session)
        messages: list[Message] = []
        events = [
            event
            async for event in guard.close_tool_round(
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


def test_interrupt_guard_rejects_unsupported_exceptions():
    app, store, _ = _app_with_completed_session("sess_guard_type_error")

    async def scenario() -> None:
        session = await store.load("sess_guard_type_error")
        assert session is not None
        guard = _interrupt_guard(app, session)
        async for _ in guard.close_tool_round(
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
        gate = _limit_gate(app, session, limits=RunLimits(max_total_tokens=10))
        guard = _interrupt_guard(app, session)
        runner = runtime_app_module._ToolRoundRunner(
            app,
            session=session,
            registered_agent=app._get_registered_agent("assistant"),
            registered_environment=None,
            environment_name=None,
            limit_gate=gate,
            interrupt_guard=guard,
            request_metadata={},
            task_id=None,
            structured_output=None,
            thinking=None,
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
        gate = _limit_gate(app, session, limits=RunLimits(max_total_tokens=100))
        guard = _interrupt_guard(app, session)
        runner = runtime_app_module._ToolRoundRunner(
            app,
            session=session,
            registered_agent=app._get_registered_agent("assistant"),
            registered_environment=None,
            environment_name=None,
            limit_gate=gate,
            interrupt_guard=guard,
            request_metadata={},
            task_id=None,
            structured_output=None,
            thinking=None,
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
