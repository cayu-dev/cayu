from __future__ import annotations

import asyncio
import re

import pytest
from tests.core._workload_secret_support import FakeProvider

from cayu.core import AgentSpec, Message, Tool, ToolResult, ToolSpec
from cayu.core.events import Event, EventType
from cayu.core.tools import ToolContext
from cayu.providers import ModelStreamEvent
from cayu.runtime._child_session_identity import ChildSessionKind, generate_child_session_id
from cayu.runtime.app import CayuApp
from cayu.runtime.config import DEFAULT_MAX_STEPS
from cayu.runtime.sessions import InMemorySessionStore, RunRequest, SessionIdentity
from cayu.tools.subagents import (
    BackgroundSubagentTaskRegistry,
    SubagentExecutionMode,
    SubagentSpec,
    SubagentTool,
)


class _RecoveryMatcherNameOnlyTool(Tool):
    spec = ToolSpec(
        name="recovery_matcher_name_only",
        description="Exercise explicit child-session recovery registration.",
        input_schema={"type": "object", "properties": {}},
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del ctx, args
        return ToolResult(content="unused")

    def matches_recoverable_child(self, *args, **kwargs) -> bool:
        del args, kwargs
        return True


def test_only_explicit_child_session_recovery_capabilities_are_registered() -> None:
    class UncalledRuntime:
        def run(self, request):
            raise AssertionError("run must not be called")

        def interrupt_session(self, request):
            raise AssertionError("interrupt_session must not be called")

    app = CayuApp(enable_logging=False)
    subagent_tool = SubagentTool(
        UncalledRuntime(),
        agents={"reviewer": SubagentSpec(agent_name="reviewer")},
    )
    name_only_tool = _RecoveryMatcherNameOnlyTool()
    app.register_agent(
        AgentSpec(name="parent", model="fake-model"),
        tools=[subagent_tool, name_only_tool],
    )

    registered_tools = app._get_registered_agent("parent").tools

    assert registered_tools[subagent_tool.name].child_session_recovery is subagent_tool
    assert registered_tools[name_only_tool.name].child_session_recovery is None


def test_random_child_session_id_generation_has_full_strength_unique_identities() -> None:
    identities = {
        generate_child_session_id(
            kind=ChildSessionKind.SUBAGENT,
            parent_session_id="parent",
        )
        for _ in range(10_000)
    }

    assert len(identities) == 10_000
    assert all(
        re.fullmatch(r"cayu-child:v1:subagent:[0-9a-f]{32}", identity) for identity in identities
    )


def test_logical_spawn_identity_is_stable_and_scoped_to_parent_and_kind() -> None:
    first = generate_child_session_id(
        kind=ChildSessionKind.SUBAGENT,
        parent_session_id="parent-a",
        logical_spawn_id="spawn-a",
    )

    assert first == generate_child_session_id(
        kind=ChildSessionKind.SUBAGENT,
        parent_session_id="parent-a",
        logical_spawn_id="spawn-a",
    )
    assert first != generate_child_session_id(
        kind=ChildSessionKind.SUBAGENT,
        parent_session_id="parent-a",
        logical_spawn_id="spawn-b",
    )
    assert first != generate_child_session_id(
        kind=ChildSessionKind.SUBAGENT,
        parent_session_id="parent-b",
        logical_spawn_id="spawn-a",
    )
    assert first != generate_child_session_id(
        kind=ChildSessionKind.WORKFLOW_STEP,
        parent_session_id="parent-a",
        logical_spawn_id="spawn-a",
    )
    assert re.fullmatch(r"cayu-child:v1:subagent:[0-9a-f]{64}", first)


@pytest.mark.parametrize(
    "mode",
    [SubagentExecutionMode.FOREGROUND, SubagentExecutionMode.BACKGROUND],
)
def test_every_subagent_mode_builds_the_parent_scoped_tool_identity(mode) -> None:
    requests = []

    class RecordingRuntime:
        def run(self, request):
            requests.append(request)

            async def events():
                yield Event(type=EventType.SESSION_STARTED, session_id=request.session_id)
                yield Event(type=EventType.SESSION_COMPLETED, session_id=request.session_id)

            return events()

        def interrupt_session(self, request):
            raise AssertionError("interrupt_session must not be called")

    registry = BackgroundSubagentTaskRegistry()
    tool = SubagentTool(
        RecordingRuntime(),
        agents={
            "reviewer": SubagentSpec(
                agent_name="reviewer",
                mode=mode,
            )
        },
        background_registry=registry,
    )
    context = ToolContext(
        session_id="parent",
        idempotency_key="spawn-key",
        metadata={"tool_call_id": "call", "idempotency_key": "spawn-key"},
    )

    async def run_twice():
        results = []
        for _ in range(2):
            results.append(
                await tool.run(
                    context,
                    {"agent": "reviewer", "task": "review"},
                )
            )
            tasks = registry.active_tasks("parent")
            if tasks:
                await asyncio.gather(*tasks)
                await asyncio.sleep(0)
        return results

    results = asyncio.run(run_twice())
    expected = generate_child_session_id(
        kind=ChildSessionKind.SUBAGENT,
        parent_session_id="parent",
        logical_spawn_id="spawn-key",
    )

    assert [request.session_id for request in requests] == [expected, expected]
    assert [result.structured["child_session_id"] for result in results] == [expected, expected]
    assert all(
        request.metadata["subagent"]["idempotency_key"] == "spawn-key" for request in requests
    )
    assert all(request.max_steps == DEFAULT_MAX_STEPS for request in requests)
    assert all("max_steps" not in request.model_fields_set for request in requests)


@pytest.mark.parametrize(
    "mode",
    [SubagentExecutionMode.FOREGROUND, SubagentExecutionMode.BACKGROUND],
)
def test_every_subagent_mode_reuses_matching_durable_child(mode) -> None:
    async def run():
        store = InMemorySessionStore()
        provider = FakeProvider(
            [
                ModelStreamEvent.text_delta("review complete"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="reviewer", model="fake-model"))
        await store.create(
            RunRequest(
                agent_name="parent",
                session_id="parent",
                messages=[Message.text("user", "parent task")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        registry = BackgroundSubagentTaskRegistry()
        tool = SubagentTool(
            app,
            agents={"reviewer": SubagentSpec(agent_name="reviewer", mode=mode)},
            background_registry=registry,
        )
        context = ToolContext(
            session_id="parent",
            idempotency_key="spawn-key",
            metadata={"tool_call_id": "call", "idempotency_key": "spawn-key"},
        )
        arguments = {"agent": "reviewer", "task": "review"}

        first = await tool.run(context, arguments)
        tasks = registry.active_tasks("parent")
        if tasks:
            await asyncio.gather(*tasks)
            await asyncio.sleep(0)
        second = await tool.run(context, arguments)
        return provider, first, second

    provider, first, second = asyncio.run(run())

    assert first.structured["child_session_id"] == second.structured["child_session_id"]
    assert second.structured["reused"] is True
    assert second.structured["status"] == "completed"
    assert second.content == "review complete"
    assert len(provider.requests) == 1


@pytest.mark.parametrize(
    "mode",
    [SubagentExecutionMode.FOREGROUND, SubagentExecutionMode.BACKGROUND],
)
def test_every_subagent_mode_fails_closed_on_existing_identity_conflict(mode) -> None:
    async def run():
        store = InMemorySessionStore()
        provider = FakeProvider([])
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="reviewer", model="fake-model"))
        identity = SessionIdentity(provider_name="fake", model="fake-model")
        await store.create(
            RunRequest(
                agent_name="parent",
                session_id="parent",
                messages=[Message.text("user", "parent task")],
            ),
            identity=identity,
        )
        child_session_id = generate_child_session_id(
            kind=ChildSessionKind.SUBAGENT,
            parent_session_id="parent",
            logical_spawn_id="spawn-key",
        )
        await store.create(
            RunRequest(
                agent_name="reviewer",
                session_id=child_session_id,
                parent_session_id="parent",
                messages=[Message.text("user", "unrelated task")],
                metadata={"subagent": {"mode": mode.value, "agent": "unrelated"}},
            ),
            identity=identity,
        )
        tool = SubagentTool(
            app,
            agents={"reviewer": SubagentSpec(agent_name="reviewer", mode=mode)},
        )
        result = await tool.run(
            ToolContext(
                session_id="parent",
                idempotency_key="spawn-key",
                metadata={"tool_call_id": "call", "idempotency_key": "spawn-key"},
            ),
            {"agent": "reviewer", "task": "review"},
        )
        return store, provider, child_session_id, result

    store, provider, child_session_id, result = asyncio.run(run())

    assert result.is_error is True
    assert result.structured["status"] == "identity_conflict"
    assert result.structured["child_session_id"] == child_session_id
    loaded = asyncio.run(store.load(child_session_id))
    assert loaded is not None
    assert loaded.metadata["subagent"]["agent"] == "unrelated"
    assert provider.requests == []
