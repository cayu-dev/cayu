from __future__ import annotations

import asyncio

import pytest
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_session_support import collect_events

from cayu._validation import DURABLE_EVENT_LIMITS
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.events import EventType
from cayu.messages import Message, TextPart, ToolResultPart
from cayu.providers.base import ModelStreamEvent
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.queries import SessionQuery
from cayu.sessions.requests import RunRequest
from cayu.storage.sqlite import SQLiteSessionStore, SQLiteTaskStore
from cayu.tasks.dispatch import TaskStoreDispatcher
from cayu.tasks.memory import InMemoryTaskStore
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
        app._tool_round_executor.invocation.terminals.governor = governor
        bounded = mode == "capacity-rejected" or mode.startswith("bounded-family")
        delegate = SubagentTool(
            app,
            agents={"child": SubagentSpec(agent_name="child")},
            max_terminal_payload_bytes=100_000 if bounded else None,
        )
        if not bounded:
            # An undeclared tool reserves the event-limit ceiling, which is
            # larger than this domain.
            delegate.spec = delegate.spec.model_copy(update={"max_terminal_payload_bytes": None})
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
                # Either way the family outgrows this small domain, so it is
                # admitted exclusively.
                assert governor.snapshot().reserved_round_bytes > 256_000
                assert governor.snapshot().active_exclusive_rounds == 2
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
        app._tool_round_executor.invocation.terminals.governor = governor
        result = WaitingResult(
            store,
            task_store=tasks,
            execution_profile_identity=profile,
            max_terminal_payload_bytes=100_000,
        )
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


def test_inline_child_round_skips_an_unrelated_round_queued_behind_its_parent():
    async def scenario():
        store = InMemorySessionStore()
        child_requested, proceed, entered, release = (asyncio.Event() for _ in range(4))

        class ChildTool(Tool):
            spec = ToolSpec(
                name="child_effect",
                description="Record one child dispatch.",
                effect=ToolEffect.EXTERNAL,
            )

            async def run(self, ctx, args):
                entered.set()
                await release.wait()
                return ToolResult(content="child result")

        class PausingProvider(FakeProvider):
            async def stream(self, request):
                if len(self.requests) == 1:
                    # The parent's round already holds its lease; hold the
                    # child's model step until an unrelated round is queued.
                    child_requested.set()
                    await proceed.wait()
                async for event in super().stream(request):
                    yield event

        provider = PausingProvider(
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
        governor = ToolTerminalPublicationGovernor(round_admission_timeout_s=None)
        app._tool_round_executor.invocation.terminals.governor = governor
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="parent", model="fake-model"),
            tools=[SubagentTool(app, agents={"child": SubagentSpec(agent_name="child")})],
        )
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
            await asyncio.wait_for(child_requested.wait(), timeout=20)
            assert governor.snapshot().active_round_reservations == 1
            competing = asyncio.create_task(
                governor.reserve_round(
                    session_id="unrelated", tool_round_id="unrelated", maximum_bytes=None
                )
            )
            for _ in range(2000):
                if governor.snapshot().round_reservation_waiters == 1:
                    break
                await asyncio.sleep(0.005)
            assert governor.snapshot().round_reservation_waiters == 1
            proceed.set()
            # The parent's lease waits for this child, so the child cannot wait
            # behind a round that is itself waiting for the parent's lease.
            await asyncio.wait_for(entered.wait(), timeout=20)
            assert not competing.done()
            release.set()
            events = await asyncio.wait_for(asyncio.shield(run), timeout=20)
            assert events[-1].type is EventType.SESSION_COMPLETED
            await asyncio.wait_for(competing, timeout=5)
            governor.release_round(session_id="unrelated", tool_round_id="unrelated")
            assert governor.snapshot().active_round_reservations == 0
        finally:
            proceed.set()
            release.set()
            pending = [run, *([] if competing is None else [competing])]
            for task in pending:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    asyncio.run(scenario())


class _RoutedProvider(FakeProvider):
    """Answer each session by its first user text and its completed tool rounds."""

    def __init__(self, scripts):
        super().__init__([])
        self._scripts = scripts

    async def stream(self, request):
        self.requests.append(request)
        text = " ".join(
            part.text
            for message in request.messages
            if message.role == "user"
            for part in message.content
            if type(part) is TextPart
        )
        script = next(script for marker, script in self._scripts.items() if marker in text)
        results = sum(
            type(part) is ToolResultPart for message in request.messages for part in message.content
        )
        calls = script(results)
        for call_id, name, arguments in calls:
            yield ModelStreamEvent.tool_call(id=call_id, name=name, arguments=arguments)
        yield ModelStreamEvent.completed({"finish_reason": "tool_calls" if calls else "stop"})


def test_sibling_children_queue_behind_an_unrelated_round_once_their_parent_finishes():
    async def scenario():
        store, tasks = InMemorySessionStore(), InMemoryTaskStore()
        dispatcher = TaskStoreDispatcher(tasks)
        gates = {call: asyncio.Event() for call in ("a-0", "a-1", "b-0", "b-1")}
        entered = {call: asyncio.Event() for call in gates}
        order = []

        class ProbeTool(Tool):
            # Unbounded terminal, so each sibling round fills the domain below.
            spec = ToolSpec(
                name="probe",
                description="Run one long probe.",
                effect=ToolEffect.EXTERNAL,
                input_schema={"type": "object", "properties": {"call": {"type": "string"}}},
            )

            async def run(self, ctx, args):
                call = args["call"]
                order.append(call)
                entered[call].set()
                await gates[call].wait()
                return ToolResult(content="probe done")

        class RootTool(Tool):
            def __init__(self, name):
                self.spec = ToolSpec(
                    name=name, description="Inspect one thing.", effect=ToolEffect.EXTERNAL
                )

            async def run(self, ctx, args):
                order.append(self.spec.name)
                return ToolResult(content="inspected")

        def child_script(prefix):
            return lambda results: (
                [(f"{prefix}-{results}", "probe", {"call": f"{prefix}-{results}"})]
                if results < 2
                else []
            )

        provider = _RoutedProvider(
            {
                "dispatch builders": lambda results: (
                    [
                        ("delegate-a", "subagent", {"agent": "builder", "task": "builder-a"}),
                        ("delegate-b", "subagent", {"agent": "builder", "task": "builder-b"}),
                    ]
                    if not results
                    else []
                ),
                "builder-a": child_script("a"),
                "builder-b": child_script("b"),
                "inspect mission": lambda results: (
                    [("root-0", "inspect_one", {}), ("root-1", "inspect_two", {})]
                    if not results
                    else []
                ),
            }
        )
        profile = ExecutionProfileBehaviorIdentity(
            name="tests:sibling-capacity", behavior_version="1", implementation_version="1"
        )
        app = CayuApp(
            session_store=store, task_store=tasks, dispatcher=dispatcher, enable_logging=False
        )
        # One undeclared call reserves the whole domain, so rounds take turns.
        governor = ToolTerminalPublicationGovernor(
            staged_capacity_bytes=DURABLE_EVENT_LIMITS.max_bytes,
            round_admission_timeout_s=None,
        )
        app._tool_round_executor.invocation.terminals.governor = governor
        delegate = SubagentTool(
            app,
            execution_profile_identity=profile,
            agents={
                "builder": SubagentSpec(agent_name="builder", mode=SubagentExecutionMode.DURABLE)
            },
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="parent", model="fake-model"), tools=[delegate])
        app.register_agent(AgentSpec(name="builder", model="fake-model"), tools=[ProbeTool()])
        app.register_agent(
            AgentSpec(name="root", model="fake-model"),
            tools=[RootTool("inspect_one"), RootTool("inspect_two")],
        )
        workers, root = [], None
        try:
            # The parent dispatches both builders and finishes, so it holds no
            # lease while they run: the builders are siblings, not nested work.
            parent_events = await asyncio.wait_for(
                collect_events(
                    app,
                    RunRequest(
                        agent_name="parent",
                        session_id="parent",
                        messages=[Message.text("user", "dispatch builders")],
                    ),
                ),
                timeout=20,
            )
            assert parent_events[-1].type is EventType.SESSION_COMPLETED
            assert governor.snapshot().active_round_reservations == 0
            workers.append(asyncio.create_task(dispatcher.process_next(app, worker_id="worker-a")))
            await asyncio.wait_for(entered["a-0"].wait(), timeout=20)
            root = asyncio.create_task(
                collect_events(
                    app,
                    RunRequest(
                        agent_name="root",
                        session_id="root",
                        messages=[Message.text("user", "inspect mission")],
                    ),
                )
            )
            for _ in range(2000):
                if governor.snapshot().round_reservation_waiters == 1:
                    break
                await asyncio.sleep(0.005)
            assert governor.snapshot().round_reservation_waiters == 1
            workers.append(asyncio.create_task(dispatcher.process_next(app, worker_id="worker-b")))
            # The second builder's first round queues behind the root instead of
            # joining its sibling's lease ahead of it.
            for _ in range(2000):
                if governor.snapshot().round_reservation_waiters == 2 or entered["b-0"].is_set():
                    break
                await asyncio.sleep(0.005)
            assert not entered["b-0"].is_set()
            gates["a-0"].set()
            root_events = await asyncio.wait_for(asyncio.shield(root), timeout=20)
            assert root_events[-1].type is EventType.SESSION_COMPLETED
            assert order[0] == "a-0"
            assert sorted(order[1:3]) == ["inspect_one", "inspect_two"]
            for gate in gates.values():
                gate.set()
            handled = await asyncio.wait_for(asyncio.gather(*workers), timeout=20)
            assert all(handle.status.value == "completed" for handle in handled)
            assert sorted(order) == ["a-0", "a-1", "b-0", "b-1", "inspect_one", "inspect_two"]
            assert governor.snapshot().active_round_reservations == 0
            assert governor.snapshot().round_reservation_waiters == 0
        finally:
            for gate in gates.values():
                gate.set()
            pending = [*workers, *([] if root is None else [root])]
            for task in pending:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    asyncio.run(scenario())
