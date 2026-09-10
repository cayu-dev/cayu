from __future__ import annotations

import asyncio

import pytest
from tests.core.test_runtime import FakeProvider, collect_events

from cayu import AgentSpec, CayuApp, Message, RunRequest, Tool, ToolEffect, ToolResult, ToolSpec
from cayu.core.events import EventType
from cayu.providers import ModelStreamEvent
from cayu.runtime.sessions import InMemorySessionStore, SessionQuery
from cayu.runtime.tool_terminal_publication import ToolTerminalPublicationGovernor
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.subagents import SubagentSpec, SubagentTool


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("mode", ["complete", "cancel", "capacity-rejected"])
def test_inline_child_capacity_uses_durable_family_without_releasing_parent(
    tmp_path, backend, mode
):
    async def scenario():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "nested.sqlite")
        )
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []

        class ChildTool(Tool):
            spec = ToolSpec(
                name="child_effect",
                description="Record one child dispatch.",
                effect=ToolEffect.EXTERNAL,
                max_terminal_payload_bytes=100_000,
            )

            async def run(self, ctx, args):
                calls.append(ctx.idempotency_key)
                entered.set()
                await release.wait()
                return ToolResult(content="child result")

        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="delegate",
                        name="subagent",
                        arguments={"agent": "child", "task": "work"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.tool_call(id="child-call", name="child_effect", arguments={}),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [ModelStreamEvent.completed({"finish_reason": "stop"})],
                [ModelStreamEvent.completed({"finish_reason": "stop"})],
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        governor = ToolTerminalPublicationGovernor(staged_capacity_bytes=256_000)
        app._tool_round_executor._terminal_publication_governor = governor
        delegate = SubagentTool(app, agents={"child": SubagentSpec(agent_name="child")})
        if mode == "capacity-rejected":
            delegate.spec = delegate.spec.model_copy(update={"max_terminal_payload_bytes": 100_000})
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="parent", model="fake-model"), tools=[delegate])
        app.register_agent(AgentSpec(name="child", model="fake-model"), tools=[ChildTool()])
        run = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="parent",
                    session_id="parent",
                    messages=[Message.text("user", "delegate")],
                ),
            )
        )
        competing = None
        try:
            if mode == "capacity-rejected":
                await asyncio.wait_for(asyncio.shield(run), timeout=20)
                assert not entered.is_set() and calls == []
            else:
                await asyncio.wait_for(entered.wait(), timeout=20)
                assert governor.snapshot().active_round_reservations == 2
                assert governor.snapshot().reserved_round_bytes == 256_000
                competing = asyncio.create_task(
                    governor.reserve_round(
                        session_id="unrelated",
                        tool_round_id="unrelated",
                        maximum_bytes=256_000,
                    )
                )
                await asyncio.sleep(0)
                assert not competing.done()
                if mode == "cancel":
                    assert run.cancelling() == 0
                    run.cancel("cancel parent with admitted child")
                    with pytest.raises(asyncio.CancelledError):
                        await asyncio.wait_for(asyncio.shield(run), timeout=20)
                    assert run.cancelled() and run.cancelling() == 1
                else:
                    release.set()
                    events = await asyncio.wait_for(asyncio.shield(run), timeout=20)
                    assert events[-1].type is EventType.SESSION_COMPLETED
                assert len(calls) == 1
                await asyncio.wait_for(competing, timeout=5)
                governor.release_round(session_id="unrelated", tool_round_id="unrelated")

            children = (
                await store.list_sessions(SessionQuery(parent_session_id="parent"))
            ).sessions
            assert len(children) == 1
            parent = await store.load("parent")
            assert parent.invocation.root_invocation_id == children[0].invocation.root_invocation_id
            child_events = await store.load_events(children[0].id)
            if mode == "capacity-rejected":
                assert not any(event.type is EventType.TOOL_CALL_STARTED for event in child_events)
                assert child_events[-1].type is EventType.SESSION_FAILED
            elif mode == "complete":
                assert (
                    sum(event.type is EventType.TOOL_CALL_COMPLETED for event in child_events) == 1
                )
            else:
                assert not any(
                    event.type is EventType.TOOL_CALL_COMPLETED for event in child_events
                )
                assert (
                    sum(
                        event.type is EventType.TOOL_EFFECT_OUTCOME_UNKNOWN
                        for event in child_events
                    )
                    == 1
                )
            assert governor.snapshot().active_round_reservations == 0
            assert governor.snapshot().reserved_round_bytes == 0
        finally:
            release.set()
            if not run.done():
                run.cancel()
            await asyncio.gather(run, return_exceptions=True)
            if competing is not None:
                competing.cancel()
                await asyncio.gather(competing, return_exceptions=True)
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())
