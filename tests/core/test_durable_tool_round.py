"""Ordinary execution must respect its durable round owner's boundaries."""

import asyncio
from contextlib import aclosing

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    Event,
    EventType,
    InMemorySessionStore,
    Message,
    ModelStreamEvent,
    RunRequest,
    ScriptedModelProvider,
    Tool,
    ToolResult,
    ToolSpec,
)
from cayu.runtime._durable_tool_round import DurableToolRound
from cayu.runtime._invocation_secrets import InvocationPublicationSnapshot
from cayu.runtime._runtime_records import ToolCallOutcome, ToolCallRequest
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.tools.base import ToolEffect
from cayu.vaults.redaction import SecretRedactor


def _app(call_count, effect=ToolEffect.EXTERNAL):
    calls = []

    class Echo(Tool):
        spec = ToolSpec(
            name="echo",
            description="Echo the call index.",
            input_schema={"type": "object", "properties": {"index": {"type": "integer"}}},
            max_terminal_payload_bytes=65_536,
            effect=effect,
            execution_profile_identity=ExecutionProfileBehaviorIdentity(
                name="tests:durable-round:echo", behavior_version="1", implementation_version="1"
            ),
        )

        async def run(self, ctx, args):
            calls.append(args["index"])
            return ToolResult(content=str(args["index"]))

    provider = ScriptedModelProvider(
        [
            [
                *(
                    ModelStreamEvent.tool_call(
                        id=f"call-{index}", name="echo", arguments={"index": index}
                    )
                    for index in range(call_count)
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed({})],
        ]
    )
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="worker", model="scripted-model"), tools=[Echo()])
    return app, store, calls


@pytest.mark.parametrize("effect", [ToolEffect.NONE, ToolEffect.EXTERNAL])
def test_owner_requires_admission_before_evidence_or_publication(monkeypatch, effect):
    async def scenario():
        app, store, calls = _app(1, effect)
        admitted = []
        admit = DurableToolRound.admit

        async def checked_admit(owner):
            assert calls == []
            before = await store.load_checkpoint("admission")
            event = Event(type=EventType.TOOL_CALL_COMPLETED, session_id="admission")
            outcome = ToolCallOutcome(
                call=ToolCallRequest(id="call-0", name="echo", arguments={"index": 0}),
                result=ToolResult(content="premature"),
            )
            snapshot = InvocationPublicationSnapshot(redactor=SecretRedactor(), unsafe_output=False)
            with pytest.raises(RuntimeError, match="requires admission"):
                owner.observe_execution(event, outcome)
            with pytest.raises(RuntimeError, match="requires admission"):
                await owner.stage_terminal(event, outcome, False, False, snapshot)
            with pytest.raises(RuntimeError, match="requires admission"):
                await owner.record_publication_snapshot("call-0", snapshot)
            with pytest.raises(RuntimeError, match="requires admission"):
                await owner.record_workspace_capture(event)
            with pytest.raises(RuntimeError, match="requires admission"):
                async for _ in owner.publish([]):
                    pass
            assert await store.load_checkpoint("admission") == before
            assert app.tool_terminal_publication_status().active_round_reservations == 0

            await admit(owner)
            with pytest.raises(RuntimeError, match="already admitted"):
                await admit(owner)
            admitted.append(owner.defers_terminals)
            assert app.tool_terminal_publication_status().active_round_reservations == int(
                owner.defers_terminals
            )

        monkeypatch.setattr(DurableToolRound, "admit", checked_admit)
        events = [
            event
            async for event in app.run(
                RunRequest(
                    session_id="admission",
                    agent_name="worker",
                    messages=[Message.text("user", "go")],
                )
            )
        ]
        assert admitted == [effect is ToolEffect.EXTERNAL]
        assert events[-1].type is EventType.SESSION_COMPLETED
        assert calls == [0]
        metrics = app.tool_terminal_publication_status()
        assert metrics.active_round_reservations == metrics.staged_count == 0

    asyncio.run(scenario())


def test_closing_deferred_publication_closes_its_hook_stream_before_return(monkeypatch):
    async def scenario():
        app, _store, calls = _app(2)
        emit = app._tool_round_executor.emit_tool_call_result_with_hooks
        closed = []

        async def tracked_emit(**kwargs):
            try:
                async with aclosing(emit(**kwargs)) as stream:
                    async for item in stream:
                        yield item
            finally:
                if kwargs.get("terminal_event_emitter") is not None:
                    await asyncio.sleep(0)
                    closed.append(kwargs["tool_call"].id)

        monkeypatch.setattr(
            app._tool_round_executor, "emit_tool_call_result_with_hooks", tracked_emit
        )
        stream = app.run(RunRequest(agent_name="worker", messages=[Message.text("user", "go")]))
        async with aclosing(stream):
            async for event in stream:
                if event.type is EventType.TOOL_CALL_COMPLETED:
                    assert calls == [0, 1]
                    assert closed == []
                    await stream.aclose()
                    assert closed == ["call-0"]
                    return
        raise AssertionError("No staged terminal was published.")

    asyncio.run(scenario())
