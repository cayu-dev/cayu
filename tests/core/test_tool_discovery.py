from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import pytest
from pydantic import ValidationError

from cayu.core import AgentSpec, EventType, ExecutionProfileBehaviorIdentity, Message
from cayu.core.tools import (
    DurableToolOperationConflict,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
)
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import (
    CayuApp,
    ForkSessionRequest,
    InMemorySessionStore,
    ResumeRequest,
    RunRequest,
)
from cayu.runtime.hooks import BeforeToolCallHookContext, RuntimeHook, ToolCallHookContext
from cayu.runtime.tool_catalogue import build_tool_catalog_snapshot, build_tool_descriptor
from cayu.runtime.tool_discovery import (
    TOOL_DISCOVERY_VIEW_OPERATION_KEY,
    ToolDiscoveryViewState,
    current_tool_discovery_view,
    initial_tool_discovery_operation_records,
    search_tool_descriptors,
    search_tools_spec,
)
from cayu.runtime.tool_exposure import ToolCapabilityCeiling
from cayu.runtime.tool_gateway import call_tool_spec
from cayu.runtime.tool_policy import (
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
)
from cayu.storage.sqlite import SQLiteSessionStore


class _RememberKnowledgeTool(Tool):
    spec = ToolSpec(
        name="remember_knowledge",
        description="Save an important reusable lesson in durable knowledge.",
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {"fact": {"type": "string"}},
            "required": ["fact"],
        },
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="tests:tool-discovery:remember-knowledge",
            behavior_version="1",
            implementation_version="1",
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, object]] = []

    async def run(self, ctx: ToolContext, args: dict[str, object]) -> ToolResult:
        del ctx
        self.calls.append(dict(args))
        return ToolResult(content=f"remembered: {args['fact']}")


class _NoiseTool(Tool):
    def __init__(self, index: int) -> None:
        super().__init__(
            ToolSpec(
                name=f"noise_{index:03d}",
                description=f"Unrelated capability number {index}.",
                input_schema={"type": "object", "additionalProperties": False},
            )
        )

    async def run(self, ctx: ToolContext, args: dict[str, object]) -> ToolResult:
        del ctx, args
        return ToolResult(content="noise")


class _DiscoveryProvider(ModelProvider):
    name = "discovery-test"

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:tool-discovery:provider",
            behavior_version="1",
            implementation_version="1",
        )

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self.tool_ref: str | None = None

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        request_number = len(self.requests)
        if request_number == 1:
            yield ModelStreamEvent.tool_call(
                id="search-call",
                name="search_tools",
                arguments={"query": "remember durable knowledge", "limit": 3},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        if request_number == 2:
            search_results = [
                part
                for message in request.messages
                if message.role == "tool"
                for part in message.content
                if part.type == "tool_result" and part.tool_name == "search_tools"
            ]
            search_result = search_results[-1]
            assert search_result.structured is not None
            [match] = search_result.structured["matches"]
            assert match["name"] == "remember_knowledge"
            assert match["descriptor_version"].startswith("sha256:")
            assert match["schema_fingerprint"].startswith("sha256:")
            assert match["readiness"] == "registered"
            assert match["input_schema"] == _RememberKnowledgeTool.spec.input_schema
            self.tool_ref = match["tool_ref"]
            yield ModelStreamEvent.tool_call(
                id="repeat-search-call",
                name="search_tools",
                arguments={"query": "remember durable knowledge", "limit": 3},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        if request_number == 3:
            search_results = [
                part
                for message in request.messages
                if message.role == "tool"
                for part in message.content
                if part.type == "tool_result" and part.tool_name == "search_tools"
            ]
            assert search_results[-1].structured is not None
            [reused_match] = search_results[-1].structured["matches"]
            assert reused_match["tool_ref"] == self.tool_ref
            assert search_results[-1].structured["view_revision"] == 1
            yield ModelStreamEvent.tool_call(
                id="gateway-call",
                name="call_tool",
                arguments={
                    "tool_ref": self.tool_ref,
                    "arguments": {"fact": "Keep discovery branch-local."},
                },
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        target_results = [
            part
            for message in request.messages
            if message.role == "tool"
            for part in message.content
            if part.type == "tool_result" and part.tool_name == "call_tool"
        ]
        if request_number == 4:
            assert target_results[-1].content == "remembered: Keep discovery branch-local."
            yield ModelStreamEvent.text_delta("done")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})
            return
        if request_number == 5:
            assert self.tool_ref is not None
            yield ModelStreamEvent.tool_call(
                id="copied-parent-gateway-call",
                name="call_tool",
                arguments={
                    "tool_ref": self.tool_ref,
                    "arguments": {"fact": "A child must not inherit this reference."},
                },
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        if request_number == 6:
            assert target_results[-1].is_error is True
            yield ModelStreamEvent.text_delta("parent reference rejected")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})
            return
        if request_number == 7:
            assert self.tool_ref is not None
            yield ModelStreamEvent.tool_call(
                id="resumed-gateway-call",
                name="call_tool",
                arguments={
                    "tool_ref": self.tool_ref,
                    "arguments": {"fact": "Discovery survives ordinary resume."},
                },
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        assert request_number == 8
        assert target_results[-1].content == "remembered: Discovery survives ordinary resume."
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _RecordingPolicy(ToolPolicy):
    def __init__(self) -> None:
        self.requests: list[ToolPolicyRequest] = []

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        self.requests.append(request)
        return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)


class _RecordingHook(RuntimeHook):
    def __init__(self) -> None:
        self.before: list[str] = []
        self.after: list[str] = []

    async def before_tool_call(self, context: BeforeToolCallHookContext) -> None:
        self.before.append(context.tool_name)

    async def after_tool_call(self, context: ToolCallHookContext) -> None:
        self.after.append(context.tool_name)


class _DiscoveryConflictOnceStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.discovery_conflicts = 0

    async def publish_session_operation(self, *args, **kwargs):
        if (
            kwargs.get("idempotency_key") == TOOL_DISCOVERY_VIEW_OPERATION_KEY
            and self.discovery_conflicts == 0
        ):
            self.discovery_conflicts += 1
            raise DurableToolOperationConflict("simulated discovery contention")
        return await super().publish_session_operation(*args, **kwargs)


class _NoAtomicOperationInitializationStore(InMemorySessionStore):
    supports_atomic_session_operation_initialization = False


def test_search_tools_vertical_keeps_catalogue_hidden_and_routes_effective_tool() -> None:
    async def run() -> None:
        store = _DiscoveryConflictOnceStore()
        provider = _DiscoveryProvider()
        remembered = _RememberKnowledgeTool()
        policy = _RecordingPolicy()
        hook = _RecordingHook()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=(remembered, *(_NoiseTool(index) for index in range(100))),
            tool_discovery_mode="search_tools",
            tool_policy=policy,
            runtime_hooks=(hook,),
        )

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="discovery-session",
                    messages=[Message.text("user", "Find and save the lesson.")],
                )
            )
        ]

        assert events[-1].type is EventType.SESSION_COMPLETED
        assert store.discovery_conflicts == 1
        assert remembered.calls == [{"fact": "Keep discovery branch-local."}]
        assert [request.tools for request in provider.requests] == [
            [search_tools_spec(), call_tool_spec()]
        ] * 4
        assert [request.tool_name for request in policy.requests] == [
            "search_tools",
            "search_tools",
            "remember_knowledge",
        ]
        assert hook.before == ["search_tools", "search_tools", "remember_knowledge"]
        assert hook.after == ["search_tools", "search_tools", "remember_knowledge"]
        state_raw = await store.load_session_operation(
            "discovery-session",
            TOOL_DISCOVERY_VIEW_OPERATION_KEY,
        )
        state = ToolDiscoveryViewState.model_validate(state_raw)
        assert state.revision == 1
        assert [grant.tool_name for grant in state.grants] == ["remember_knowledge"]
        assert state.grants[0].origin_model_step_id.startswith("mstep_")
        assert state.grants[0].created_at.tzinfo is not None
        assert "remember durable knowledge" not in state.model_dump_json()
        assert "input_schema" not in state.grants[0].model_dump(mode="json")
        assert provider.tool_ref == state.grants[0].tool_ref
        assert provider.tool_ref not in json.dumps(
            [event.model_dump(mode="json") for event in events],
            sort_keys=True,
        )
        assert "remember durable knowledge" not in json.dumps(
            [event.model_dump(mode="json") for event in events],
            sort_keys=True,
        )
        public_search_result = next(
            event.payload["result"]
            for event in events
            if event.type is EventType.TOOL_CALL_COMPLETED and event.tool_name == "search_tools"
        )
        assert public_search_result["structured"] == {
            "schema_version": 1,
            "match_count": 1,
            "view_revision": 1,
            "truncated": False,
        }
        durable_events = await store.load_events("discovery-session")
        assert [
            event.tool_name
            for event in durable_events
            if event.type is EventType.TOOL_CALL_COMPLETED
        ] == ["search_tools", "search_tools", "remember_knowledge"]

        with pytest.raises(ValueError, match="cannot change its durable capability ceiling"):
            _ = [
                event
                async for event in app.resume(
                    ResumeRequest(
                        session_id="discovery-session",
                        messages=[Message.text("user", "Drop every tool.")],
                        tool_capability_ceiling=ToolCapabilityCeiling(tool_names=()),
                    )
                )
            ]
        assert len(provider.requests) == 4
        assert (
            ToolDiscoveryViewState.model_validate(
                await store.load_session_operation(
                    "discovery-session",
                    TOOL_DISCOVERY_VIEW_OPERATION_KEY,
                )
            )
            == state
        )

        forked = [
            event
            async for event in app.fork_session(
                ForkSessionRequest(
                    source_session_id="discovery-session",
                    session_id="discovery-child",
                )
            )
        ]
        assert forked[0].type is EventType.SESSION_FORKED
        child_state = ToolDiscoveryViewState.model_validate(
            await store.load_session_operation(
                "discovery-child",
                TOOL_DISCOVERY_VIEW_OPERATION_KEY,
            )
        )
        assert child_state.session_id == "discovery-child"
        assert child_state.generation_id != state.generation_id
        assert child_state.revision == 0
        assert child_state.grants == ()
        assert (
            ToolDiscoveryViewState.model_validate(
                await store.load_session_operation(
                    "discovery-session",
                    TOOL_DISCOVERY_VIEW_OPERATION_KEY,
                )
            )
            == state
        )

        child_resumed = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id="discovery-child",
                    messages=[Message.text("user", "Try the copied parent reference.")],
                )
            )
        ]
        assert child_resumed[-1].type is EventType.SESSION_COMPLETED
        assert remembered.calls == [{"fact": "Keep discovery branch-local."}]
        child_rejection = next(
            event
            for event in child_resumed
            if event.type is EventType.TARGETED_TOOL_REFERENCE_REJECTED
        )
        assert child_rejection.payload["authority_kind"] == "tool_discovery"
        assert child_rejection.payload["rejection_reason"] == "unknown"
        assert provider.tool_ref not in child_rejection.model_dump_json()

        resumed = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id="discovery-session",
                    messages=[Message.text("user", "Save one more lesson.")],
                )
            )
        ]
        assert resumed[-1].type is EventType.SESSION_COMPLETED
        assert remembered.calls == [
            {"fact": "Keep discovery branch-local."},
            {"fact": "Discovery survives ordinary resume."},
        ]
        resumed_state_raw = await store.load_session_operation(
            "discovery-session",
            TOOL_DISCOVERY_VIEW_OPERATION_KEY,
        )
        assert ToolDiscoveryViewState.model_validate(resumed_state_raw) == state
        assert [request.tools for request in provider.requests] == [
            [search_tools_spec(), call_tool_spec()]
        ] * 8
        assert hook.before == [
            "search_tools",
            "search_tools",
            "remember_knowledge",
            "remember_knowledge",
        ]
        assert hook.after == [
            "search_tools",
            "search_tools",
            "remember_knowledge",
            "remember_knowledge",
        ]

    asyncio.run(run())


def test_discovery_rejects_a_store_without_atomic_view_initialization() -> None:
    async def run() -> None:
        store = _NoAtomicOperationInitializationStore()
        provider = _DiscoveryProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=(_RememberKnowledgeTool(),),
            tool_discovery_mode="search_tools",
        )

        with pytest.raises(RuntimeError, match="atomic session operation initialization"):
            _ = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="unsupported-discovery-store",
                        messages=[Message.text("user", "Find a tool.")],
                    )
                )
            ]

        assert await store.load("unsupported-discovery-store") is None
        assert provider.requests == []

    asyncio.run(run())


def test_sqlite_reconstruction_preserves_parent_view_and_empty_fork(tmp_path) -> None:
    async def run() -> None:
        database = tmp_path / "tool-discovery.sqlite"
        provider = _DiscoveryProvider()
        remembered = _RememberKnowledgeTool()

        first_store = SQLiteSessionStore(database)
        first_app = CayuApp(session_store=first_store, enable_logging=False)
        first_app.register_provider(provider, default=True)
        first_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=(remembered,),
            tool_discovery_mode="search_tools",
        )
        try:
            initial = [
                event
                async for event in first_app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="sqlite-discovery-parent",
                        messages=[Message.text("user", "Find and save the lesson.")],
                    )
                )
            ]
            assert initial[-1].type is EventType.SESSION_COMPLETED
            forked = [
                event
                async for event in first_app.fork_session(
                    ForkSessionRequest(
                        source_session_id="sqlite-discovery-parent",
                        session_id="sqlite-discovery-child",
                    )
                )
            ]
            assert forked[0].type is EventType.SESSION_FORKED
        finally:
            await first_store.close()

        reopened_store = SQLiteSessionStore(database)
        reopened_app = CayuApp(session_store=reopened_store, enable_logging=False)
        reopened_app.register_provider(provider, default=True)
        reopened_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=(remembered,),
            tool_discovery_mode="search_tools",
        )
        try:
            child = [
                event
                async for event in reopened_app.resume(
                    ResumeRequest(
                        session_id="sqlite-discovery-child",
                        messages=[Message.text("user", "Try the copied parent reference.")],
                    )
                )
            ]
            assert child[-1].type is EventType.SESSION_COMPLETED
            child_state = ToolDiscoveryViewState.model_validate(
                await reopened_store.load_session_operation(
                    "sqlite-discovery-child",
                    TOOL_DISCOVERY_VIEW_OPERATION_KEY,
                )
            )
            assert child_state.revision == 0
            assert child_state.grants == ()

            parent = [
                event
                async for event in reopened_app.resume(
                    ResumeRequest(
                        session_id="sqlite-discovery-parent",
                        messages=[Message.text("user", "Save one more lesson.")],
                    )
                )
            ]
            assert parent[-1].type is EventType.SESSION_COMPLETED
            parent_state = ToolDiscoveryViewState.model_validate(
                await reopened_store.load_session_operation(
                    "sqlite-discovery-parent",
                    TOOL_DISCOVERY_VIEW_OPERATION_KEY,
                )
            )
            assert parent_state.revision == 1
            assert [grant.tool_name for grant in parent_state.grants] == ["remember_knowledge"]
            assert remembered.calls == [
                {"fact": "Keep discovery branch-local."},
                {"fact": "Discovery survives ordinary resume."},
            ]
        finally:
            await reopened_store.close()

    asyncio.run(run())


def test_search_ranking_is_deterministic_bounded_and_excludes_direct_tools() -> None:
    descriptors = (
        build_tool_descriptor(
            name="remember_knowledge",
            description="Save reusable knowledge.",
            input_schema={
                "type": "object",
                "properties": {"lesson_text": {"type": "string"}},
            },
            parallel_safe=True,
            effect="external",
            publishes_arguments=True,
            workspace_mutation=False,
        ),
        build_tool_descriptor(
            name="search_notes",
            description="Search saved knowledge notes.",
            input_schema={},
            parallel_safe=True,
            effect="none",
            publishes_arguments=True,
            workspace_mutation=False,
        ),
    )
    catalogue = build_tool_catalog_snapshot(descriptors)
    ceiling = ToolCapabilityCeiling(tool_names=("remember_knowledge", "search_notes"))

    matches = search_tool_descriptors(
        "remember knowledge",
        catalogue=catalogue,
        ceiling=ceiling,
        excluded_names=("search_notes",),
    )

    assert [descriptor.name for descriptor in matches] == ["remember_knowledge"]
    assert [
        descriptor.name
        for descriptor in search_tool_descriptors(
            "lesson text",
            catalogue=catalogue,
            ceiling=ceiling,
        )
    ] == ["remember_knowledge"]
    assert (
        next(
            descriptor.name
            for descriptor in search_tool_descriptors(
                descriptors[0].tool_id,
                catalogue=catalogue,
                ceiling=ceiling,
            )
        )
        == "remember_knowledge"
    )
    assert (
        search_tool_descriptors(
            "member",
            catalogue=catalogue,
            ceiling=ceiling,
        )
        == ()
    )


def test_missing_foreign_or_stale_view_fails_closed() -> None:
    descriptor = build_tool_descriptor(
        name="remember_knowledge",
        description="Save reusable knowledge.",
        input_schema={},
        parallel_safe=True,
        effect="external",
        publishes_arguments=True,
        workspace_mutation=False,
    )
    catalogue = build_tool_catalog_snapshot((descriptor,))
    ceiling = ToolCapabilityCeiling(tool_names=(descriptor.name,))
    state = ToolDiscoveryViewState.model_validate(
        initial_tool_discovery_operation_records(
            session_id="parent",
            root_invocation_id="root",
            agent_name="assistant",
            catalogue=catalogue,
            ceiling=ceiling,
        )[TOOL_DISCOVERY_VIEW_OPERATION_KEY]
    )

    with pytest.raises(ValueError, match="not initialized"):
        current_tool_discovery_view(
            None,
            session_id="parent",
            generation_id=state.generation_id,
            agent_name="assistant",
            catalogue=catalogue,
            ceiling=ceiling,
        )

    with pytest.raises(ValueError, match="conflicts with current session authority"):
        current_tool_discovery_view(
            state.model_dump(mode="json"),
            session_id="child",
            generation_id=f"sha256:{'2' * 64}",
            agent_name="assistant",
            catalogue=catalogue,
            ceiling=ceiling,
        )

    changed_catalogue = build_tool_catalog_snapshot(
        (
            build_tool_descriptor(
                name="remember_knowledge",
                description="Save changed reusable knowledge.",
                input_schema={},
                parallel_safe=True,
                effect="external",
                publishes_arguments=True,
                workspace_mutation=False,
            ),
        )
    )
    with pytest.raises(ValueError, match="conflicts with current session authority"):
        current_tool_discovery_view(
            state.model_dump(mode="json"),
            session_id="parent",
            generation_id=state.generation_id,
            agent_name="assistant",
            catalogue=changed_catalogue,
            ceiling=ceiling,
        )

    with pytest.raises(ValueError, match="conflicts with current session authority"):
        current_tool_discovery_view(
            state.model_dump(mode="json"),
            session_id="parent",
            generation_id=state.generation_id,
            agent_name="assistant",
            catalogue=catalogue,
            ceiling=ToolCapabilityCeiling(tool_names=()),
        )


def test_malformed_discovery_view_fails_closed() -> None:
    descriptor = build_tool_descriptor(
        name="remember_knowledge",
        description="Save reusable knowledge.",
        input_schema={},
        parallel_safe=True,
        effect="external",
        publishes_arguments=True,
        workspace_mutation=False,
    )
    catalogue = build_tool_catalog_snapshot((descriptor,))
    ceiling = ToolCapabilityCeiling(tool_names=(descriptor.name,))

    with pytest.raises(ValidationError):
        current_tool_discovery_view(
            {"schema_version": 1, "session_id": "incomplete"},
            session_id="session",
            generation_id=f"sha256:{'3' * 64}",
            agent_name="assistant",
            catalogue=catalogue,
            ceiling=ceiling,
        )


def test_registration_rejects_invalid_discovery_modes() -> None:
    app = CayuApp(enable_logging=False)

    with pytest.raises(ValueError, match="tool_discovery_mode must be one of: search_tools"):
        app.register_agent(
            AgentSpec(name="assistant", model="model"),
            tool_discovery_mode="unknown",
        )
    with pytest.raises(TypeError, match="tool_discovery_mode must be a ToolDiscoveryMode"):
        app.register_agent(
            AgentSpec(name="assistant", model="model"),
            tool_discovery_mode=True,  # type: ignore[arg-type]
        )
