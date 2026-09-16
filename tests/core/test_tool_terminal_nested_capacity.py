from __future__ import annotations

import asyncio

import pytest
from tests.core.test_runtime import FakeProvider, collect_events

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.events import EventType
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions.base import InMemorySessionStore, RunRequest, SessionQuery
from cayu.storage.sqlite import SQLiteSessionStore, SQLiteTaskStore
from cayu.tasks.base import InMemoryTaskStore
from cayu.tasks.dispatch import TaskStoreDispatcher
from cayu.tools.base import Tool, ToolEffect, ToolResult, ToolSpec
from cayu.tools.subagents import (
    SubagentExecutionMode,
    SubagentResultTool,
    SubagentSpec,
    SubagentTool,
)
from cayu.tools.terminal_publication import ToolTerminalPublicationGovernor


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize(
    "mode", ["complete", "cancel", "capacity-rejected", "bounded-family", "bounded-family-cancel"]
)
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
        if mode == "capacity-rejected" or mode.startswith("bounded-family"):
            delegate.spec = delegate.spec.model_copy(update={"max_terminal_payload_bytes": 100_000})
        if mode == "capacity-rejected":
            await governor.reserve_round(
                session_id="existing-competitor",
                tool_round_id="existing-competitor",
                maximum_bytes=1,
            )
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
                if mode.startswith("bounded-family"):
                    assert governor.snapshot().reserved_round_bytes > 256_000
                else:
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
                if mode.endswith("cancel"):
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
            elif mode in {"complete", "bounded-family"}:
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
            if mode == "capacity-rejected":
                governor.release_round(
                    session_id="existing-competitor",
                    tool_round_id="existing-competitor",
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


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_queued_siblings_share_exclusive_aggregate_while_parent_waits(tmp_path, backend):
    async def scenario():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "queued.sqlite")
        )
        tasks = (
            InMemoryTaskStore()
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "queued.sqlite")
        )
        dispatcher = TaskStoreDispatcher(tasks)
        waiting, entered, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
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
                if len(calls) == 2:
                    entered.set()
                await release.wait()
                return ToolResult(content="child result")

        class WaitingResult(SubagentResultTool):
            async def run(self, ctx, args):
                waiting.set()
                # Ensure the queued worker is executing before polling. A
                # merely pending task legitimately returns not-ready at once.
                await entered.wait()
                return await super().run(ctx, args)

        profile = ExecutionProfileBehaviorIdentity(
            name="tests:nested-capacity", behavior_version="1", implementation_version="1"
        )
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="delegate", name="subagent", arguments={"agent": "child", "task": "work"}
                    ),
                    ModelStreamEvent.tool_call(
                        id="delegate-2",
                        name="subagent",
                        arguments={"agent": "child", "task": "work-2"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.tool_call(
                        id="wait-child",
                        name="subagent_result",
                        arguments={"all": True, "wait": True, "timeout_s": 30},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.tool_call(id="effect", name="child_effect", arguments={}),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.tool_call(id="effect-2", name="child_effect", arguments={}),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [ModelStreamEvent.completed({"finish_reason": "stop"})],
                [ModelStreamEvent.completed({"finish_reason": "stop"})],
                [ModelStreamEvent.completed({"finish_reason": "stop"})],
            ]
        )
        app = CayuApp(
            session_store=store, task_store=tasks, dispatcher=dispatcher, enable_logging=False
        )
        governor = ToolTerminalPublicationGovernor(staged_capacity_bytes=256_000)
        app._tool_round_executor._terminal_publication_governor = governor
        result = WaitingResult(store, task_store=tasks, execution_profile_identity=profile)
        result.spec = result.spec.model_copy(update={"max_terminal_payload_bytes": 100_000})
        delegate = SubagentTool(
            app,
            execution_profile_identity=profile,
            agents={"child": SubagentSpec(agent_name="child", mode=SubagentExecutionMode.DURABLE)},
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="parent", model="fake-model"), tools=[delegate, result])
        app.register_agent(AgentSpec(name="child", model="fake-model"), tools=[ChildTool()])
        parent = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="parent",
                    session_id="parent",
                    messages=[Message.text("user", "delegate")],
                ),
            )
        )
        workers = []
        try:
            await asyncio.wait_for(waiting.wait(), timeout=20)
            workers = [
                asyncio.create_task(dispatcher.process_next(app, worker_id=f"worker-{i}"))
                for i in range(2)
            ]
            await asyncio.wait_for(entered.wait(), timeout=20)
            assert governor.snapshot().active_round_reservations == 2
            assert governor.snapshot().reserved_round_bytes > 256_000
            release.set()
            handled = await asyncio.wait_for(asyncio.gather(*workers), timeout=20)
            assert all(handle.status.value == "completed" for handle in handled)
            events = await asyncio.wait_for(parent, timeout=20)
            assert events[-1].type is EventType.SESSION_COMPLETED
            assert not any(e.type is EventType.TOOL_CALL_FAILED for e in events)
            children = (
                await store.list_sessions(SessionQuery(parent_session_id="parent"))
            ).sessions
            assert len(children) == 2
            for child in children:
                assert (
                    child.invocation.root_invocation_id
                    == (await store.load("parent")).invocation.root_invocation_id
                )
                child_events = await store.load_events(child.id)
                assert sum(e.type is EventType.TOOL_CALL_COMPLETED for e in child_events) == 1
            assert len(calls) == len(set(calls)) == 2
            assert governor.snapshot().active_round_reservations == 0
            assert governor.snapshot().reserved_round_bytes == 0
        finally:
            release.set()
            pending = [parent, *workers]
            for task in pending:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            if backend == "sqlite":
                await tasks.close()
                await store.close()

    asyncio.run(scenario())
