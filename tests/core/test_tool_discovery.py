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
from cayu.providers.base import (
    OPENAI_ADDITIONAL_TOOLS_PROTOCOL,
    OPENAI_CLIENT_TOOL_SEARCH_PROTOCOL,
)
from cayu.runtime import (
    CayuApp,
    ForkSessionRequest,
    InMemorySessionStore,
    MessageWindowContextPolicy,
    ResumeRequest,
    RunRequest,
    StaticToolExposurePolicy,
    TargetedToolGrant,
)
from cayu.runtime.hooks import BeforeToolCallHookContext, RuntimeHook, ToolCallHookContext
from cayu.runtime.tool_catalogue import build_tool_catalog_snapshot, build_tool_descriptor
from cayu.runtime.tool_discovery import (
    TOOL_DISCOVERY_VIEW_OPERATION_KEY,
    ToolDiscoveryMode,
    ToolDiscoveryProjectionKind,
    ToolDiscoveryViewInspection,
    ToolDiscoveryViewState,
    current_tool_discovery_view,
    initial_tool_discovery_operation_records,
    resolve_tool_discovery_projection,
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
from cayu.vaults import REDACTED_SECRET, SecretRedactor


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


class _NativeDiscoveryProvider(ModelProvider):
    name = "native-discovery-test"

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:tool-discovery:native-provider",
            behavior_version="1",
            implementation_version="1",
        )

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def supports_tool_discovery_projection(self, *, model: str, protocol: str) -> bool:
        return model == "fake-model" and protocol == OPENAI_CLIENT_TOOL_SEARCH_PROTOCOL

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        projection = request.tool_discovery_projection
        assert projection is not None
        if len(self.requests) == 1:
            assert projection.loaded_tool_names == ()
            yield ModelStreamEvent.tool_call(
                id="native-search",
                name="search_tools",
                arguments={"query": "remember durable knowledge", "limit": 3},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        if len(self.requests) == 2:
            assert projection.loaded_tool_names == ("remember_knowledge",)
            yield ModelStreamEvent.tool_call(
                id="native-tool-call",
                name="remember_knowledge",
                arguments={"fact": "Native discovery keeps durable authority."},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        if len(self.requests) == 4:
            assert projection.loaded_tool_names == ()
            yield ModelStreamEvent.tool_call(
                id="child-guessed-tool-call",
                name="remember_knowledge",
                arguments={"fact": "A child must not inherit or guess this capability."},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        if len(self.requests) == 5:
            assert projection.loaded_tool_names == ()
            rejected_results = [
                part
                for message in request.messages
                if message.role == "tool"
                for part in message.content
                if part.type == "tool_result" and part.tool_name == "remember_knowledge"
            ]
            assert rejected_results[-1].is_error is True
            yield ModelStreamEvent.text_delta("child guess rejected")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})
            return
        if len(self.requests) == 6:
            assert projection.loaded_tool_names == ("remember_knowledge",)
            yield ModelStreamEvent.tool_call(
                id="native-resumed-tool-call",
                name="remember_knowledge",
                arguments={"fact": "Native discovery survives ordinary resume."},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        assert len(self.requests) in {3, 7}
        results = [
            part
            for message in request.messages
            if message.role == "tool"
            for part in message.content
            if part.type == "tool_result" and part.tool_name == "remember_knowledge"
        ]
        expected_result = (
            "remembered: Native discovery keeps durable authority."
            if len(self.requests) == 3
            else "remembered: Native discovery survives ordinary resume."
        )
        assert results[-1].content == expected_result
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _SecretBearingDiscoveryTool(Tool):
    def __init__(self, secret: str, *, secret_schema_key: bool = False) -> None:
        property_name = secret if secret_schema_key else "note"
        super().__init__(
            ToolSpec(
                name="remember_private_note",
                description=f"Save a private note without exposing {secret}.",
                input_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        property_name: {
                            "type": "string",
                            "description": f"A note whose protected default is {secret}.",
                            "default": secret,
                        }
                    },
                },
            )
        )

    async def run(self, ctx: ToolContext, args: dict[str, object]) -> ToolResult:
        del ctx, args
        return ToolResult(content="unused")


class _NativeDiscoveryRedactionProvider(ModelProvider):
    name = "native-discovery-redaction-test"

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:tool-discovery:native-redaction-provider",
            behavior_version="1",
            implementation_version="1",
        )

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def supports_tool_discovery_projection(self, *, model: str, protocol: str) -> bool:
        return model == "fake-model" and protocol == OPENAI_CLIENT_TOOL_SEARCH_PROTOCOL

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        projection = request.tool_discovery_projection
        assert projection is not None
        if len(self.requests) == 1:
            assert projection.loaded_tool_names == ()
            yield ModelStreamEvent.tool_call(
                id="redaction-search",
                name="search_tools",
                arguments={"query": "remember private note", "limit": 1},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        assert len(self.requests) == 2
        assert projection.loaded_tool_names == ("remember_private_note",)
        yield ModelStreamEvent.text_delta("loaded safely")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _NativeDiscoveryTrimProvider(ModelProvider):
    name = "native-discovery-trim-test"

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:tool-discovery:native-trim-provider",
            behavior_version="1",
            implementation_version="1",
        )

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def supports_tool_discovery_projection(self, *, model: str, protocol: str) -> bool:
        return model == "fake-model" and protocol == OPENAI_CLIENT_TOOL_SEARCH_PROTOCOL

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        projection = request.tool_discovery_projection
        assert projection is not None
        request_number = len(self.requests)
        if request_number == 1:
            assert projection.loaded_tool_names == ()
            yield ModelStreamEvent.tool_call(
                id="trim-search",
                name="search_tools",
                arguments={"query": "remember durable knowledge", "limit": 1},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        if request_number == 2:
            assert projection.loaded_tool_names == ("remember_knowledge",)
            yield ModelStreamEvent.text_delta("loaded")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})
            return
        if request_number == 3:
            assert projection.loaded_tool_names == ()
            yield ModelStreamEvent.tool_call(
                id="trimmed-guessed-call",
                name="remember_knowledge",
                arguments={"fact": "A trimmed schema must not retain call authority."},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        assert request_number == 4
        assert projection.loaded_tool_names == ()
        rejected_results = [
            part
            for message in request.messages
            if message.role == "tool"
            for part in message.content
            if part.type == "tool_result" and part.tool_name == "remember_knowledge"
        ]
        assert rejected_results[-1].is_error is True
        yield ModelStreamEvent.text_delta("trimmed guess rejected")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _NativeDiscoveryAndTargetedProvider(ModelProvider):
    name = "native-discovery-and-targeted-test"

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:tool-discovery:native-and-targeted-provider",
            behavior_version="1",
            implementation_version="1",
        )

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def supports_targeted_tool_projection(self, *, model: str, protocol: str) -> bool:
        return model == "fake-model" and protocol == OPENAI_ADDITIONAL_TOOLS_PROTOCOL

    def supports_tool_discovery_projection(self, *, model: str, protocol: str) -> bool:
        return model == "fake-model" and protocol == OPENAI_CLIENT_TOOL_SEARCH_PROTOCOL

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        assert request.targeted_tool_projection is not None
        assert [tool["name"] for tool in request.targeted_tool_projection.tools] == [
            "remember_knowledge"
        ]
        discovery = request.tool_discovery_projection
        assert discovery is not None
        assert discovery.loaded_tool_names == ()
        if len(self.requests) == 1:
            yield ModelStreamEvent.tool_call(
                id="overlap-search",
                name="search_tools",
                arguments={"query": "remember durable knowledge", "limit": 1},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        if len(self.requests) == 2:
            search_results = [
                part
                for message in request.messages
                if message.role == "tool"
                for part in message.content
                if part.type == "tool_result" and part.tool_name == "search_tools"
            ]
            assert search_results[-1].structured is not None
            assert [match["name"] for match in search_results[-1].structured["matches"]] == [
                "remember_knowledge"
            ]
            yield ModelStreamEvent.tool_call(
                id="overlap-native-call",
                name="remember_knowledge",
                arguments={"fact": "Targeted authority wins an overlapping native name."},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        assert len(self.requests) == 3
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _NativeDirectExposureProvider(ModelProvider):
    name = "native-direct-exposure-test"

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:tool-discovery:native-direct-exposure-provider",
            behavior_version="1",
            implementation_version="1",
        )

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def supports_tool_discovery_projection(self, *, model: str, protocol: str) -> bool:
        return model == "fake-model" and protocol == OPENAI_CLIENT_TOOL_SEARCH_PROTOCOL

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        projection = request.tool_discovery_projection
        assert projection is not None
        assert projection.loaded_tool_names == ()
        if len(self.requests) == 1:
            assert [tool["name"] for tool in request.tools] == [
                "search_tools",
                "call_tool",
                "remember_knowledge",
            ]
            yield ModelStreamEvent.tool_call(
                id="direct-exposure-search",
                name="search_tools",
                arguments={"query": "remember knowledge", "limit": 1},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        assert len(self.requests) == 2
        search_results = [
            part
            for message in request.messages
            if message.role == "tool"
            for part in message.content
            if part.type == "tool_result" and part.tool_name == "search_tools"
        ]
        assert search_results[-1].structured is not None
        assert search_results[-1].structured["matches"] == []
        yield ModelStreamEvent.text_delta("direct tool omitted from discovery")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _RecordingPolicy(ToolPolicy):
    def __init__(self) -> None:
        self.requests: list[ToolPolicyRequest] = []

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        self.requests.append(request)
        return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)


class _RequireRememberApprovalPolicy(ToolPolicy):
    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        if request.tool_name == "remember_knowledge":
            return ToolPolicyResult(
                decision=ToolPolicyDecision.REQUIRE_APPROVAL,
                reason="Remembering knowledge requires review.",
            )
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
        request_footprints = [
            event.payload for event in events if event.type is EventType.REQUEST_FOOTPRINT_RECORDED
        ]
        assert [footprint["schema_version"] for footprint in request_footprints] == [6] * 4
        assert [
            (
                footprint["tool_discovery_view"]["revision"],
                footprint["tool_discovery_view"]["grant_count"],
            )
            for footprint in request_footprints
        ] == [(0, 0), (1, 1), (1, 1), (1, 1)]
        assert (
            len(
                {
                    json.dumps(footprint["fingerprints"]["tool_manifest"], sort_keys=True)
                    for footprint in request_footprints
                }
            )
            == 1
        )
        inspection = await app.inspect_tool_discovery_view("discovery-session")
        assert isinstance(inspection, ToolDiscoveryViewInspection)
        assert inspection.session_id == app.project_session_id_for_exposure(state.session_id)
        assert inspection.generation_id == state.generation_id
        assert inspection.revision == 1
        assert inspection.grant_count == 1
        assert inspection.grants_truncated is False
        assert [grant.tool_name for grant in inspection.grants] == ["remember_knowledge"]
        inspection_json = inspection.model_dump_json()
        assert state.grants[0].tool_ref not in inspection_json
        assert state.grants[0].grant_id not in inspection_json
        assert state.grants[0].origin_query_sha256 not in inspection_json
        assert "input_schema" not in inspection_json
        for invalid_limit in (True, 0, 257):
            with pytest.raises(ValueError, match="limit must be an integer from 1 through 256"):
                await app.inspect_tool_discovery_view(
                    "discovery-session",
                    limit=invalid_limit,  # type: ignore[arg-type]
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
        child_footprints = [
            event.payload
            for event in child_resumed
            if event.type is EventType.REQUEST_FOOTPRINT_RECORDED
        ]
        expected_child_view = {
            "generation_id": child_state.generation_id,
            "revision": 0,
            "catalogue_revision": child_state.catalogue_revision,
            "ceiling_fingerprint": child_state.ceiling_fingerprint,
            "grant_count": 0,
        }
        assert [footprint["tool_discovery_view"] for footprint in child_footprints] == [
            expected_child_view,
            expected_child_view,
        ]
        assert all(
            footprint["fingerprints"]["tool_manifest"]
            == request_footprints[0]["fingerprints"]["tool_manifest"]
            for footprint in child_footprints
        )

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
        resumed_footprints = [
            event.payload for event in resumed if event.type is EventType.REQUEST_FOOTPRINT_RECORDED
        ]
        assert [
            footprint["tool_discovery_view"]["grant_count"] for footprint in resumed_footprints
        ] == [1, 1]
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


def test_native_discovery_routes_loaded_name_through_the_same_grant_and_hooks() -> None:
    async def run() -> None:
        provider = _NativeDiscoveryProvider()
        remembered = _RememberKnowledgeTool()
        policy = _RecordingPolicy()
        hook = _RecordingHook()
        app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=(remembered,),
            tool_discovery_mode="openai_tool_search_client",
            tool_policy=policy,
            runtime_hooks=(hook,),
        )

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="native-discovery-session",
                    messages=[Message.text("user", "Find and save the lesson.")],
                )
            )
        ]

        assert events[-1].type is EventType.SESSION_COMPLETED
        assert remembered.calls == [{"fact": "Native discovery keeps durable authority."}]
        assert [
            request.tool_discovery_projection.loaded_tool_names for request in provider.requests
        ] == [
            (),
            ("remember_knowledge",),
            ("remember_knowledge",),
        ]
        assert [request.tool_name for request in policy.requests] == [
            "search_tools",
            "remember_knowledge",
        ]
        assert hook.before == ["search_tools", "remember_knowledge"]
        assert hook.after == ["search_tools", "remember_knowledge"]

        forked = [
            event
            async for event in app.fork_session(
                ForkSessionRequest(
                    source_session_id="native-discovery-session",
                    session_id="native-discovery-child",
                )
            )
        ]
        assert forked[0].type is EventType.SESSION_FORKED
        child_events = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id="native-discovery-child",
                    messages=[Message.text("user", "Continue independently.")],
                )
            )
        ]
        assert child_events[-1].type is EventType.SESSION_COMPLETED
        assert provider.requests[-1].tool_discovery_projection is not None
        assert provider.requests[-1].tool_discovery_projection.loaded_tool_names == ()
        assert remembered.calls == [{"fact": "Native discovery keeps durable authority."}]
        child_blocked = next(
            event for event in child_events if event.type is EventType.TOOL_CALL_BLOCKED
        )
        assert child_blocked.payload["reason"] == "not_exposed_in_request"

        resumed_events = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id="native-discovery-session",
                    messages=[Message.text("user", "Save one more lesson.")],
                )
            )
        ]
        assert resumed_events[-1].type is EventType.SESSION_COMPLETED
        assert remembered.calls == [
            {"fact": "Native discovery keeps durable authority."},
            {"fact": "Native discovery survives ordinary resume."},
        ]
        assert [
            request.tool_discovery_projection.loaded_tool_names
            for request in provider.requests[-2:]
        ] == [
            ("remember_knowledge",),
            ("remember_knowledge",),
        ]
        assert [request.tool_name for request in policy.requests] == [
            "search_tools",
            "remember_knowledge",
            "remember_knowledge",
        ]
        assert hook.before == [
            "search_tools",
            "remember_knowledge",
            "remember_knowledge",
        ]
        assert hook.after == [
            "search_tools",
            "remember_knowledge",
            "remember_knowledge",
        ]

    asyncio.run(run())


def test_native_discovery_redacts_loaded_definitions_before_provider_dispatch() -> None:
    async def run() -> None:
        secret = "native-discovery-secret-canary"
        provider = _NativeDiscoveryRedactionProvider()
        app = CayuApp(
            session_store=InMemorySessionStore(),
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=(_SecretBearingDiscoveryTool(secret),),
            tool_discovery_mode="openai_tool_search_client",
        )

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="native-discovery-redaction",
                    messages=[Message.text("user", "Find the private-note capability.")],
                )
            )
        ]

        assert events[-1].type is EventType.SESSION_COMPLETED
        assert len(provider.requests) == 2
        projection = provider.requests[-1].tool_discovery_projection
        assert projection is not None
        [loaded_tool] = projection.loaded_tools
        rendered_tool = json.dumps(loaded_tool, sort_keys=True)
        rendered_request = provider.requests[-1].model_dump_json()
        assert secret not in rendered_tool
        assert secret not in rendered_request
        assert REDACTED_SECRET in rendered_tool
        assert loaded_tool["name"] == "remember_private_note"

    asyncio.run(run())


def test_native_discovery_rejects_secret_bearing_schema_keys_before_dispatch() -> None:
    async def run() -> None:
        secret = "native-discovery-secret-schema-key"
        provider = _NativeDiscoveryRedactionProvider()
        app = CayuApp(
            session_store=InMemorySessionStore(),
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=(_SecretBearingDiscoveryTool(secret, secret_schema_key=True),),
            tool_discovery_mode="openai_tool_search_client",
        )

        with pytest.raises(ValueError) as exc_info:
            _ = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="native-discovery-secret-schema-key",
                        messages=[Message.text("user", "Find the private-note capability.")],
                    )
                )
            ]

        assert provider.requests == []
        assert secret not in repr((str(exc_info.value), vars(exc_info.value)))

    asyncio.run(run())


def test_targeted_native_grant_takes_precedence_over_same_name_discovery_grant() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        provider = _NativeDiscoveryAndTargetedProvider()
        remembered = _RememberKnowledgeTool()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=(remembered,),
            targeted_tool_mode="openai_additional_tools",
            tool_discovery_mode="openai_tool_search_client",
        )

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="native-discovery-targeted-overlap",
                    messages=[Message.text("user", "Find and save the lesson.")],
                    tool_grants=(
                        TargetedToolGrant(
                            request_id="targeted-overlap",
                            tool_id="cayu:remember_knowledge",
                            max_calls=1,
                            lifetime_seconds=60,
                        ),
                    ),
                )
            )
        ]

        assert events[-1].type is EventType.SESSION_COMPLETED
        assert remembered.calls == [{"fact": "Targeted authority wins an overlapping native name."}]
        [targeted_record] = await store.list_targeted_tool_grants(
            "native-discovery-targeted-overlap"
        )
        started = next(
            event
            for event in events
            if event.type is EventType.TOOL_CALL_STARTED and event.tool_name == "remember_knowledge"
        )
        assert started.payload["dispatch_kind"] == "native"
        assert started.payload["grant_id"] == targeted_record.grant_id
        discovery_view = await app.inspect_tool_discovery_view("native-discovery-targeted-overlap")
        assert discovery_view.grant_count == 1
        assert [grant.tool_name for grant in discovery_view.grants] == ["remember_knowledge"]
        assert [
            request.tool_discovery_projection.loaded_tool_names
            for request in provider.requests
            if request.tool_discovery_projection is not None
        ] == [(), (), ()]

    asyncio.run(run())


def test_native_discovery_uses_the_ordinary_tool_approval_boundary() -> None:
    async def run() -> None:
        provider = _NativeDiscoveryProvider()
        remembered = _RememberKnowledgeTool()
        app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=(remembered,),
            tool_discovery_mode="openai_tool_search_client",
            tool_policy=_RequireRememberApprovalPolicy(),
        )

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="native-discovery-approval",
                    messages=[Message.text("user", "Find and save the lesson.")],
                )
            )
        ]

        approval = next(
            event for event in events if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        assert approval.tool_name == "remember_knowledge"
        assert events[-1].type is EventType.SESSION_INTERRUPTED
        assert events[-1].payload["interruption_type"] == "tool_approval_required"
        assert remembered.calls == []
        assert [
            request.tool_discovery_projection.loaded_tool_names for request in provider.requests
        ] == [(), ("remember_knowledge",)]

    asyncio.run(run())


def test_native_discovery_unloads_a_grant_when_context_drops_its_schema_evidence() -> None:
    async def run() -> None:
        provider = _NativeDiscoveryTrimProvider()
        remembered = _RememberKnowledgeTool()
        app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=(remembered,),
            tool_discovery_mode="openai_tool_search_client",
            context_policy=MessageWindowContextPolicy(max_messages=2),
        )

        initial_events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="native-discovery-trim",
                    messages=[Message.text("user", "Find the memory tool.")],
                )
            )
        ]
        assert initial_events[-1].type is EventType.SESSION_COMPLETED

        resumed_events = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id="native-discovery-trim",
                    messages=[Message.text("user", "Try the remembered tool name.")],
                )
            )
        ]

        assert resumed_events[-1].type is EventType.SESSION_COMPLETED
        blocked = next(
            event for event in resumed_events if event.type is EventType.TOOL_CALL_BLOCKED
        )
        assert blocked.payload["reason"] == "not_exposed_in_request"
        assert remembered.calls == []
        assert [
            request.tool_discovery_projection.loaded_tool_names for request in provider.requests
        ] == [(), ("remember_knowledge",), (), ()]

    asyncio.run(run())


def test_native_discovery_omits_the_current_direct_exposure_from_search_results() -> None:
    async def run() -> None:
        provider = _NativeDirectExposureProvider()
        remembered = _RememberKnowledgeTool()
        app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=(remembered,),
            tool_discovery_mode="openai_tool_search_client",
            tool_exposure_policy=StaticToolExposurePolicy(
                profile_id="direct-memory",
                tools=("remember_knowledge",),
            ),
        )

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="native-discovery-direct-exposure",
                    messages=[Message.text("user", "Search for the visible memory tool.")],
                )
            )
        ]

        assert events[-1].type is EventType.SESSION_COMPLETED
        assert remembered.calls == []
        assert len(provider.requests) == 2

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


def test_discovery_projection_resolution_is_explicit_and_fallback_is_portable() -> None:
    portable_provider = _DiscoveryProvider()
    native_provider = _NativeDiscoveryProvider()

    assert (
        resolve_tool_discovery_projection(
            ToolDiscoveryMode.SEARCH_TOOLS,
            provider=portable_provider,
            model="fake-model",
        )
        is ToolDiscoveryProjectionKind.SEARCH_TOOLS
    )
    assert (
        resolve_tool_discovery_projection(
            ToolDiscoveryMode.OPENAI_TOOL_SEARCH_CLIENT_OR_SEARCH_TOOLS,
            provider=portable_provider,
            model="fake-model",
        )
        is ToolDiscoveryProjectionKind.SEARCH_TOOLS
    )
    assert (
        resolve_tool_discovery_projection(
            ToolDiscoveryMode.OPENAI_TOOL_SEARCH_CLIENT,
            provider=native_provider,
            model="fake-model",
        )
        is ToolDiscoveryProjectionKind.OPENAI_TOOL_SEARCH_CLIENT
    )
    with pytest.raises(ValueError, match="not established"):
        resolve_tool_discovery_projection(
            ToolDiscoveryMode.OPENAI_TOOL_SEARCH_CLIENT,
            provider=portable_provider,
            model="fake-model",
        )


def test_required_native_discovery_rejects_before_session_creation() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        provider = _DiscoveryProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=(_RememberKnowledgeTool(),),
            tool_discovery_mode="openai_tool_search_client",
        )

        with pytest.raises(ValueError, match="not established"):
            _ = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="unsupported-native-discovery",
                        messages=[Message.text("user", "Find a tool.")],
                    )
                )
            ]

        assert await store.load("unsupported-native-discovery") is None
        assert provider.requests == []

    asyncio.run(run())
