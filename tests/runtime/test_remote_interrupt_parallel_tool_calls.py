"""A durable interrupt during a tool round ends the session interrupted.

Both durably staged results must be published even if interruption cancels a
parallel call before it refreshes the round owner's in-memory stage set, and a
round already committed when the interrupt is seen has nothing left to refresh.
"""

from __future__ import annotations

import asyncio

import pytest
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_session_support import collect_events

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.events import EventType
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.runtime._durable_tool_round import DurableToolRound
from cayu.runtime._run_limits import RunLimitGate
from cayu.runtime.stop_policy import StopDecision, StopLimit
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionStatus
from cayu.sessions.requests import InterruptSessionRequest, RunRequest
from cayu.tools.base import Tool, ToolContext, ToolResult, ToolSpec


class Gate(Tool):  # parallel_safe defaults to True
    spec = ToolSpec(
        name="gate",
        max_terminal_payload_bytes=1024,
        input_schema={"type": "object", "properties": {"label": {"type": "string"}}},
    )

    def __init__(self):
        super().__init__()
        self.started, self.both, self.release = [], asyncio.Event(), asyncio.Event()

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.started.append(args["label"])
        if len(self.started) == 2:
            self.both.set()
        await self.release.wait()
        return ToolResult(content=f"done {args['label']}")


def _owner(store, tool) -> CayuApp:
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(
        FakeProvider(
            [
                ModelStreamEvent.tool_call(id="a", name="gate", arguments={"label": "a"}),
                ModelStreamEvent.tool_call(id="b", name="gate", arguments={"label": "b"}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])
    return app


async def _interrupt_from_another_worker(store):
    other = CayuApp(session_store=store, enable_logging=False)
    other.register_provider(FakeProvider([]), default=True)
    other.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[Gate()])
    try:
        async for _ in other.interrupt_session(InterruptSessionRequest(session_id="s")):
            pass
    except TimeoutError:  # "still finalizing": the owner must finish it
        pass


async def _mark_interrupting(store):
    await store.update_status("s", SessionStatus.INTERRUPTING)


@pytest.mark.parametrize(
    "interrupt", [_interrupt_from_another_worker, _mark_interrupting], ids=["worker", "status"]
)
def test_remote_interrupt_during_two_parallel_calls_ends_interrupted(interrupt):
    async def scenario():
        store, tool = InMemorySessionStore(), Gate()
        owner = _owner(store, tool)
        run = asyncio.create_task(
            collect_events(
                owner,
                RunRequest(
                    agent_name="assistant",
                    session_id="s",
                    messages=[Message.text("user", "go")],
                ),
            )
        )
        await asyncio.wait_for(tool.both.wait(), 10)
        await interrupt(store)
        tool.release.set()
        events = await asyncio.wait_for(run, 30)
        return store, events

    store, events = asyncio.run(scenario())
    types = [event.type for event in events]
    assert EventType.SESSION_FAILED not in types
    assert types[-1] == EventType.SESSION_INTERRUPTED
    session = asyncio.run(store.load("s"))
    assert session.status is SessionStatus.INTERRUPTED
    # Both calls returned, so both results are kept rather than lost.
    stored = asyncio.run(store.load_events("s"))
    results = sorted(
        event.payload["result"]["content"]
        for event in stored
        if event.type == EventType.TOOL_CALL_COMPLETED
    )
    assert results == ["done a", "done b"]


class Sequential(Tool):
    spec = ToolSpec(
        name="step",
        parallel_safe=False,
        max_terminal_payload_bytes=1024,
        input_schema={"type": "object", "properties": {}},
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(content="unreachable")


def test_interrupt_seen_after_a_limit_closed_the_round_ends_interrupted(monkeypatch):
    """The round is committed before the interrupt check that raises."""

    cutoff_checks = []

    def cutoff(self):
        cutoff_checks.append(None)
        if len(cutoff_checks) < 2:
            return None
        # The first sequential dispatch check finds only the finishing reserve left.
        return StopDecision(
            limit=StopLimit.ELAPSED_SECONDS,
            maximum=60,
            actual=59,
            message="Finishing before the elapsed-time limit.",
        )

    monkeypatch.setattr(RunLimitGate, "tool_cutoff_decision", cutoff)
    original_commit = DurableToolRound._commit_snapshot

    async def commit_after_remote_interrupt(self, *args, **kwargs):
        # Another worker marks the session while the round is being closed.
        await self._session_store.update_status(self._session.id, SessionStatus.INTERRUPTING)
        return await original_commit(self, *args, **kwargs)

    monkeypatch.setattr(DurableToolRound, "_commit_snapshot", commit_after_remote_interrupt)

    async def scenario():
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            FakeProvider(
                [
                    ModelStreamEvent.tool_call(id="a", name="step", arguments={}),
                    ModelStreamEvent.tool_call(id="b", name="step", arguments={}),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ]
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[Sequential()])
        events = await asyncio.wait_for(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="s",
                    messages=[Message.text("user", "go")],
                ),
            ),
            30,
        )
        return store, events

    store, events = asyncio.run(scenario())
    types = [event.type for event in events]
    assert EventType.SESSION_FAILED not in types
    assert types[-1] == EventType.SESSION_INTERRUPTED
    assert asyncio.run(store.load("s")).status is SessionStatus.INTERRUPTED
