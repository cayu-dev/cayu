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
from cayu.providers import ModelProvider, ModelProviderError, ModelRequest, ModelStreamEvent
from cayu.providers.base import (
    OPENAI_ADDITIONAL_TOOLS_PROTOCOL,
    OPENAI_CLIENT_TOOL_SEARCH_PROTOCOL,
    OPENAI_HOSTED_TOOL_SEARCH_PROTOCOL,
    ToolDiscoveryProjectionResult,
)
from cayu.runtime import (
    CayuApp,
    ForkSessionRequest,
    InMemorySessionStore,
    MessageWindowContextPolicy,
    ResumeRequest,
    RetryPolicy,
    RunRequest,
    StaticToolExposurePolicy,
    TargetedToolGrant,
    ToolApprovalDecision,
    ToolApprovalRequest,
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


class _ChangedRememberKnowledgeTool(_RememberKnowledgeTool):
    spec = ToolSpec(
        name="remember_knowledge",
        description="Save changed knowledge with incompatible catalogue authority.",
        input_schema=_RememberKnowledgeTool.spec.input_schema,
        execution_profile_identity=_RememberKnowledgeTool.spec.execution_profile_identity,
    )


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


class _HostedDiscoveryProvider(ModelProvider):
    name = "hosted-discovery-test"

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:tool-discovery:hosted-provider",
            behavior_version="1",
            implementation_version="1",
        )

    def __init__(self, *, evidence: str = "valid") -> None:
        self.requests: list[ModelRequest] = []
        self.evidence = evidence

    def supports_tool_discovery_projection(self, *, model: str, protocol: str) -> bool:
        return model == "fake-model" and protocol == OPENAI_HOSTED_TOOL_SEARCH_PROTOCOL

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        projection = request.tool_discovery_projection
        assert projection is not None
        assert projection.protocol == OPENAI_HOSTED_TOOL_SEARCH_PROTOCOL
        assert projection.candidate_tool_names == ("remember_knowledge",)
        if len(self.requests) == 1:
            if self.evidence == "empty":
                yield ModelStreamEvent(
                    type="completed",
                    payload={"finish_reason": "stop"},
                    tool_discovery_result=ToolDiscoveryProjectionResult(),
                )
                return
            yield ModelStreamEvent.tool_call(
                id="hosted-native-tool-call",
                name="remember_knowledge",
                arguments={"fact": "Hosted discovery binds durable authority atomically."},
            )
            if self.evidence == "missing":
                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                return
            loaded_tools = projection.candidate_tools
            if self.evidence == "altered":
                loaded_tools = (
                    {
                        **projection.candidate_tools[0],
                        "description": "Provider-altered authority.",
                    },
                )
            elif self.evidence == "unrelated":
                loaded_tools = (
                    {
                        **projection.candidate_tools[0],
                        "name": "unrelated_tool",
                    },
                )
            yield ModelStreamEvent(
                type="completed",
                payload={"finish_reason": "tool_calls"},
                tool_discovery_result=ToolDiscoveryProjectionResult(
                    loaded_tools=loaded_tools,
                ),
            )
            return
        results = [
            part
            for message in request.messages
            if message.role == "tool"
            for part in message.content
            if part.type == "tool_result" and part.tool_name == "remember_knowledge"
        ]
        assert results[-1].content == (
            "remembered: Hosted discovery binds durable authority atomically."
        )
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _HostedReplayDiscoveryProvider(ModelProvider):
    name = "hosted-replay-discovery-test"

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:tool-discovery:hosted-replay-provider",
            behavior_version="1",
            implementation_version="1",
        )

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def supports_tool_discovery_projection(self, *, model: str, protocol: str) -> bool:
        return model == "fake-model" and protocol == OPENAI_HOSTED_TOOL_SEARCH_PROTOCOL

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        projection = request.tool_discovery_projection
        assert projection is not None
        assert projection.candidate_tool_names == ("remember_knowledge",)
        request_number = len(self.requests)
        if request_number == 1:
            assert projection.loaded_tool_names == ()
            yield ModelStreamEvent.tool_call(
                id="hosted-replay-first-call",
                name="remember_knowledge",
                arguments={"fact": "Hosted replay grants the tool."},
            )
            candidate = projection.candidate_tools[0]
            yield ModelStreamEvent(
                type="completed",
                payload={
                    "finish_reason": "tool_calls",
                    "provider_state": [
                        {
                            "provider": "openai",
                            "state": {
                                "type": "tool_search_call",
                                "execution": "server",
                                "call_id": None,
                                "status": "completed",
                                "arguments": {"paths": ["remember_knowledge"]},
                            },
                        },
                        {
                            "provider": "openai",
                            "state": {
                                "type": "tool_search_output",
                                "execution": "server",
                                "call_id": None,
                                "status": "completed",
                                "tools": [
                                    {
                                        "type": "function",
                                        "name": candidate["name"],
                                        "description": candidate["description"],
                                        "parameters": candidate["input_schema"],
                                        "strict": False,
                                        "defer_loading": True,
                                    }
                                ],
                            },
                        },
                    ],
                },
                tool_discovery_result=ToolDiscoveryProjectionResult(
                    loaded_tools=projection.candidate_tools,
                ),
            )
            return
        assert projection.loaded_tool_names == ("remember_knowledge",)
        if request_number == 2:
            yield ModelStreamEvent.text_delta("first call complete")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})
            return
        if request_number == 3:
            yield ModelStreamEvent.tool_call(
                id="hosted-replay-later-call",
                name="remember_knowledge",
                arguments={"fact": "Hosted replay reuses durable authority."},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        assert request_number == 4
        yield ModelStreamEvent.text_delta("replayed call complete")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _HostedDiscoveryAndTargetedProvider(ModelProvider):
    name = "hosted-discovery-and-targeted-test"

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:tool-discovery:hosted-and-targeted-provider",
            behavior_version="1",
            implementation_version="1",
        )

    def __init__(self, *, load_noise: bool = True) -> None:
        self.requests: list[ModelRequest] = []
        self.load_noise = load_noise

    def supports_targeted_tool_projection(self, *, model: str, protocol: str) -> bool:
        return model == "fake-model" and protocol == OPENAI_ADDITIONAL_TOOLS_PROTOCOL

    def supports_tool_discovery_projection(self, *, model: str, protocol: str) -> bool:
        return model == "fake-model" and protocol == OPENAI_HOSTED_TOOL_SEARCH_PROTOCOL

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        targeted = request.targeted_tool_projection
        assert targeted is not None
        assert [tool["name"] for tool in targeted.tools] == ["remember_knowledge"]
        discovery = request.tool_discovery_projection
        assert discovery is not None
        assert discovery.candidate_tool_names == (("noise_000",) if self.load_noise else ())
        if len(self.requests) == 1:
            yield ModelStreamEvent.tool_call(
                id="hosted-targeted-call",
                name="remember_knowledge",
                arguments={"fact": "Targeted authority wins hosted name precedence."},
            )
            if self.load_noise:
                yield ModelStreamEvent(
                    type="completed",
                    payload={"finish_reason": "tool_calls"},
                    tool_discovery_result=ToolDiscoveryProjectionResult(
                        loaded_tools=discovery.candidate_tools,
                    ),
                )
            else:
                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        assert len(self.requests) == 2
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _HostedMultiDiscoveryProvider(ModelProvider):
    name = "hosted-multi-discovery-test"

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:tool-discovery:hosted-multi-provider",
            behavior_version="1",
            implementation_version="1",
        )

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def supports_tool_discovery_projection(self, *, model: str, protocol: str) -> bool:
        return model == "fake-model" and protocol == OPENAI_HOSTED_TOOL_SEARCH_PROTOCOL

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        discovery = request.tool_discovery_projection
        assert discovery is not None
        assert discovery.candidate_tool_names == ("noise_000", "remember_knowledge")
        if len(self.requests) == 1:
            yield ModelStreamEvent.tool_call(
                id="hosted-multi-call",
                name="remember_knowledge",
                arguments={"fact": "Multi-load grants an exact canonical subset."},
            )
            yield ModelStreamEvent(
                type="completed",
                payload={"finish_reason": "tool_calls"},
                tool_discovery_result=ToolDiscoveryProjectionResult(
                    loaded_tools=tuple(reversed(discovery.candidate_tools)),
                ),
            )
            return
        assert len(self.requests) == 2
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _HostedDirectExposureProvider(ModelProvider):
    name = "hosted-direct-exposure-test"

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:tool-discovery:hosted-direct-exposure-provider",
            behavior_version="1",
            implementation_version="1",
        )

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def supports_tool_discovery_projection(self, *, model: str, protocol: str) -> bool:
        return model == "fake-model" and protocol == OPENAI_HOSTED_TOOL_SEARCH_PROTOCOL

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        discovery = request.tool_discovery_projection
        assert discovery is not None
        assert discovery.candidate_tool_names == ("noise_000",)
        assert [tool["name"] for tool in request.tools] == [
            "search_tools",
            "call_tool",
            "remember_knowledge",
        ]
        yield ModelStreamEvent(
            type="completed",
            payload={"finish_reason": "stop"},
            tool_discovery_result=ToolDiscoveryProjectionResult(),
        )


class _HostedZeroCandidateProvider(ModelProvider):
    name = "hosted-zero-candidate-test"

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:tool-discovery:hosted-zero-candidate-provider",
            behavior_version="1",
            implementation_version="1",
        )

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def supports_tool_discovery_projection(self, *, model: str, protocol: str) -> bool:
        return model == "fake-model" and protocol == OPENAI_HOSTED_TOOL_SEARCH_PROTOCOL

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        discovery = request.tool_discovery_projection
        assert discovery is not None
        assert discovery.candidate_tools == ()
        assert [tool["name"] for tool in request.tools] == [
            "search_tools",
            "call_tool",
            "remember_knowledge",
        ]
        yield ModelStreamEvent.text_delta("No discovery needed.")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _HostedForkProvider(ModelProvider):
    name = "hosted-fork-test"

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:tool-discovery:hosted-fork-provider",
            behavior_version="1",
            implementation_version="1",
        )

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def supports_tool_discovery_projection(self, *, model: str, protocol: str) -> bool:
        return model == "fake-model" and protocol == OPENAI_HOSTED_TOOL_SEARCH_PROTOCOL

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        discovery = request.tool_discovery_projection
        assert discovery is not None
        assert discovery.candidate_tool_names == ("remember_knowledge",)
        request_number = len(self.requests)
        if request_number in {1, 4}:
            fact = (
                "Parent hosted discovery authority."
                if request_number == 1
                else "Parent resume requires fresh hosted evidence."
            )
            yield ModelStreamEvent.tool_call(
                id=f"hosted-parent-call-{request_number}",
                name="remember_knowledge",
                arguments={"fact": fact},
            )
            yield ModelStreamEvent(
                type="completed",
                payload={"finish_reason": "tool_calls"},
                tool_discovery_result=ToolDiscoveryProjectionResult(
                    loaded_tools=discovery.candidate_tools,
                ),
            )
            return
        if request_number == 3:
            yield ModelStreamEvent.tool_call(
                id="hosted-child-guessed-call",
                name="remember_knowledge",
                arguments={"fact": "A child cannot inherit hosted authority."},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        assert request_number in {2, 5}
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _HostedRetryProvider(ModelProvider):
    name = "hosted-retry-test"

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:tool-discovery:hosted-retry-provider",
            behavior_version="1",
            implementation_version="1",
        )

    def __init__(self) -> None:
        self.projections = []

    def supports_tool_discovery_projection(self, *, model: str, protocol: str) -> bool:
        return model == "fake-model" and protocol == OPENAI_HOSTED_TOOL_SEARCH_PROTOCOL

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        discovery = request.tool_discovery_projection
        assert discovery is not None
        self.projections.append(discovery)
        if len(self.projections) == 1:
            raise ModelProviderError(
                "temporary hosted provider failure",
                provider=self.name,
                status_code=503,
                retryable=True,
            )
        assert len(self.projections) == 2
        yield ModelStreamEvent(
            type="completed",
            payload={"finish_reason": "stop"},
            tool_discovery_result=ToolDiscoveryProjectionResult(),
        )


class _HostedCancellationProvider(ModelProvider):
    name = "hosted-cancellation-test"

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:tool-discovery:hosted-cancellation-provider",
            behavior_version="1",
            implementation_version="1",
        )

    def __init__(self) -> None:
        self.call_emitted = asyncio.Event()
        self.cancelled = asyncio.Event()

    def supports_tool_discovery_projection(self, *, model: str, protocol: str) -> bool:
        return model == "fake-model" and protocol == OPENAI_HOSTED_TOOL_SEARCH_PROTOCOL

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        discovery = request.tool_discovery_projection
        assert discovery is not None
        assert discovery.candidate_tool_names == ("remember_knowledge",)
        yield ModelStreamEvent.tool_call(
            id="hosted-cancelled-call",
            name="remember_knowledge",
            arguments={"fact": "Uncommitted hosted selection must not execute."},
        )
        self.call_emitted.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.cancelled.set()


class _HostedCompactionProvider(ModelProvider):
    name = "hosted-compaction-test"

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:tool-discovery:hosted-compaction-provider",
            behavior_version="1",
            implementation_version="1",
        )

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def supports_tool_discovery_projection(self, *, model: str, protocol: str) -> bool:
        return model == "fake-model" and protocol == OPENAI_HOSTED_TOOL_SEARCH_PROTOCOL

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        discovery = request.tool_discovery_projection
        assert discovery is not None
        assert discovery.candidate_tool_names == ("remember_knowledge",)
        if len(self.requests) == 1:
            yield ModelStreamEvent.tool_call(
                id="hosted-compaction-initial-call",
                name="remember_knowledge",
                arguments={"fact": "Compacted hosted authority."},
            )
            yield ModelStreamEvent(
                type="completed",
                payload={"finish_reason": "tool_calls"},
                tool_discovery_result=ToolDiscoveryProjectionResult(
                    loaded_tools=discovery.candidate_tools,
                ),
            )
            return
        if len(self.requests) == 2:
            yield ModelStreamEvent.text_delta("done")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})
            return
        assert len(self.requests) == 3
        assert all(
            not (
                part.type == "provider_state"
                and part.state.get("type") in {"tool_search_call", "tool_search_output"}
            )
            for message in request.messages
            for part in message.content
        )
        yield ModelStreamEvent.tool_call(
            id="hosted-compaction-stale-call",
            name="remember_knowledge",
            arguments={"fact": "Compacted evidence must not remain callable."},
        )
        yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})


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


def test_hosted_discovery_publishes_loaded_grant_before_the_native_call() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        provider = _HostedDiscoveryProvider()
        remembered = _RememberKnowledgeTool()
        policy = _RecordingPolicy()
        hook = _RecordingHook()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=(remembered,),
            tool_discovery_mode="openai_tool_search_hosted",
            tool_policy=policy,
            runtime_hooks=(hook,),
        )

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="hosted-discovery-session",
                    messages=[Message.text("user", "Find and save this lesson.")],
                )
            )
        ]

        assert events[-1].type is EventType.SESSION_COMPLETED
        assert remembered.calls == [
            {"fact": "Hosted discovery binds durable authority atomically."}
        ]
        assert [request.tool_discovery_projection.protocol for request in provider.requests] == [
            OPENAI_HOSTED_TOOL_SEARCH_PROTOCOL,
            OPENAI_HOSTED_TOOL_SEARCH_PROTOCOL,
        ]
        assert [request.tool_name for request in policy.requests] == ["remember_knowledge"]
        assert hook.before == ["remember_knowledge"]
        assert hook.after == ["remember_knowledge"]
        state = ToolDiscoveryViewState.model_validate(
            await store.load_session_operation(
                "hosted-discovery-session",
                TOOL_DISCOVERY_VIEW_OPERATION_KEY,
            )
        )
        assert state.revision == 1
        assert [grant.tool_name for grant in state.grants] == ["remember_knowledge"]
        footprints = [
            event.payload for event in events if event.type is EventType.REQUEST_FOOTPRINT_RECORDED
        ]
        assert [footprint["schema_version"] for footprint in footprints] == [7, 7]
        assert [footprint["tool_discovery_projection"] for footprint in footprints] == [
            {
                "protocol": OPENAI_HOSTED_TOOL_SEARCH_PROTOCOL,
                "candidate_count": 1,
                "loaded_count": 0,
                "generation_id": state.generation_id,
            }
        ] * 2
        serialized_footprints = json.dumps(footprints, sort_keys=True)
        assert "remember_knowledge" not in serialized_footprints
        assert "Hosted discovery binds durable authority atomically" not in serialized_footprints

    asyncio.run(run())


def test_hosted_discovery_accepts_an_empty_selection_without_granting_authority() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        provider = _HostedDiscoveryProvider(evidence="empty")
        remembered = _RememberKnowledgeTool()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=(remembered,),
            tool_discovery_mode="openai_tool_search_hosted",
        )

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="hosted-discovery-empty-selection",
                    messages=[Message.text("user", "Use a tool only if one is relevant.")],
                )
            )
        ]

        assert events[-1].type is EventType.SESSION_COMPLETED
        assert remembered.calls == []
        state = ToolDiscoveryViewState.model_validate(
            await store.load_session_operation(
                "hosted-discovery-empty-selection",
                TOOL_DISCOVERY_VIEW_OPERATION_KEY,
            )
        )
        assert state.revision == 0
        assert state.grants == ()

    asyncio.run(run())


def test_hosted_discovery_reuses_a_grant_only_while_exact_replay_evidence_remains() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        provider = _HostedReplayDiscoveryProvider()
        remembered = _RememberKnowledgeTool()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=(remembered,),
            tool_discovery_mode="openai_tool_search_hosted",
        )
        session_id = "hosted-discovery-replay"

        initial = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "Find and save the first lesson.")],
                )
            )
        ]
        assert initial[-1].type is EventType.SESSION_COMPLETED

        resumed = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id=session_id,
                    messages=[Message.text("user", "Use the retained loaded tool again.")],
                )
            )
        ]

        assert resumed[-1].type is EventType.SESSION_COMPLETED
        assert remembered.calls == [
            {"fact": "Hosted replay grants the tool."},
            {"fact": "Hosted replay reuses durable authority."},
        ]
        view = ToolDiscoveryViewState.model_validate(
            await store.load_session_operation(session_id, TOOL_DISCOVERY_VIEW_OPERATION_KEY)
        )
        assert view.revision == 1
        assert [grant.tool_name for grant in view.grants] == ["remember_knowledge"]
        footprints = [
            event.payload
            for event in (*initial, *resumed)
            if event.type is EventType.REQUEST_FOOTPRINT_RECORDED
        ]
        assert [item["tool_discovery_projection"]["loaded_count"] for item in footprints] == [
            0,
            1,
            1,
            1,
        ]

    asyncio.run(run())


def test_hosted_discovery_atomically_grants_a_canonical_multi_load() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        provider = _HostedMultiDiscoveryProvider()
        remembered = _RememberKnowledgeTool()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=(remembered, _NoiseTool(0)),
            tool_discovery_mode="openai_tool_search_hosted",
        )

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="hosted-discovery-multi-load",
                    messages=[Message.text("user", "Load the exact relevant subset.")],
                )
            )
        ]

        assert events[-1].type is EventType.SESSION_COMPLETED
        assert remembered.calls == [{"fact": "Multi-load grants an exact canonical subset."}]
        state = ToolDiscoveryViewState.model_validate(
            await store.load_session_operation(
                "hosted-discovery-multi-load",
                TOOL_DISCOVERY_VIEW_OPERATION_KEY,
            )
        )
        assert state.revision == 1
        assert [grant.tool_name for grant in state.grants] == [
            "noise_000",
            "remember_knowledge",
        ]

    asyncio.run(run())


def test_hosted_discovery_excludes_current_direct_exposure_from_candidates() -> None:
    async def run() -> None:
        provider = _HostedDirectExposureProvider()
        app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=(_RememberKnowledgeTool(), _NoiseTool(0)),
            tool_discovery_mode="openai_tool_search_hosted",
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
                    session_id="hosted-discovery-direct-exposure",
                    messages=[Message.text("user", "Use the current direct surface.")],
                )
            )
        ]

        assert events[-1].type is EventType.SESSION_COMPLETED
        assert len(provider.requests) == 1

    asyncio.run(run())


def test_hosted_discovery_is_a_noop_when_direct_exposure_covers_the_ceiling() -> None:
    async def run() -> None:
        provider = _HostedZeroCandidateProvider()
        app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=(_RememberKnowledgeTool(),),
            tool_discovery_mode="openai_tool_search_hosted",
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
                    session_id="hosted-discovery-zero-candidate",
                    messages=[Message.text("user", "Use only the direct surface.")],
                )
            )
        ]

        assert events[-1].type is EventType.SESSION_COMPLETED
        assert len(provider.requests) == 1
        [footprint] = [
            event.payload for event in events if event.type is EventType.REQUEST_FOOTPRINT_RECORDED
        ]
        assert footprint["tool_discovery_projection"]["candidate_count"] == 0
        assert footprint["tools"]["count"] == 1

    asyncio.run(run())


def test_hosted_discovery_fork_resets_authority_and_parent_resume_reselects() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        provider = _HostedForkProvider()
        remembered = _RememberKnowledgeTool()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=(remembered,),
            tool_discovery_mode="openai_tool_search_hosted",
        )

        parent = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="hosted-fork-parent",
                    messages=[Message.text("user", "Find and save the parent lesson.")],
                )
            )
        ]
        assert parent[-1].type is EventType.SESSION_COMPLETED

        _ = [
            event
            async for event in app.fork_session(
                ForkSessionRequest(
                    source_session_id="hosted-fork-parent",
                    session_id="hosted-fork-child",
                )
            )
        ]
        child = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id="hosted-fork-child",
                    messages=[Message.text("user", "Try copied hosted authority.")],
                )
            )
        ]
        assert child[-1].type is EventType.SESSION_FAILED
        child_view = ToolDiscoveryViewState.model_validate(
            await store.load_session_operation(
                "hosted-fork-child",
                TOOL_DISCOVERY_VIEW_OPERATION_KEY,
            )
        )
        assert child_view.revision == 0
        assert child_view.grants == ()

        resumed = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id="hosted-fork-parent",
                    messages=[Message.text("user", "Save a second parent lesson.")],
                )
            )
        ]
        assert resumed[-1].type is EventType.SESSION_COMPLETED
        assert remembered.calls == [
            {"fact": "Parent hosted discovery authority."},
            {"fact": "Parent resume requires fresh hosted evidence."},
        ]
        generation_ids = [
            request.tool_discovery_projection.generation_id
            for request in provider.requests
            if request.tool_discovery_projection is not None
        ]
        assert generation_ids[0] == generation_ids[1] == generation_ids[3] == generation_ids[4]
        assert generation_ids[2] != generation_ids[0]

    asyncio.run(run())


def test_hosted_discovery_retry_preserves_the_exact_branch_candidate_projection() -> None:
    async def run() -> None:
        provider = _HostedRetryProvider()
        app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=(_RememberKnowledgeTool(),),
            tool_discovery_mode="openai_tool_search_hosted",
        )

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="hosted-discovery-retry",
                    messages=[Message.text("user", "Retry without changing authority.")],
                    retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
                )
            )
        ]

        assert events[-1].type is EventType.SESSION_COMPLETED
        assert len(provider.projections) == 2
        assert provider.projections[0] == provider.projections[1]

    asyncio.run(run())


def test_hosted_discovery_cancellation_cannot_publish_or_execute_unfinished_selection() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        provider = _HostedCancellationProvider()
        remembered = _RememberKnowledgeTool()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=(remembered,),
            tool_discovery_mode="openai_tool_search_hosted",
        )

        async def collect() -> list:
            return [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="hosted-discovery-cancelled",
                        messages=[Message.text("user", "Cancel after an uncommitted call.")],
                    )
                )
            ]

        task = asyncio.create_task(collect())
        await asyncio.wait_for(provider.call_emitted.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert provider.cancelled.is_set()
        assert remembered.calls == []
        state = ToolDiscoveryViewState.model_validate(
            await store.load_session_operation(
                "hosted-discovery-cancelled",
                TOOL_DISCOVERY_VIEW_OPERATION_KEY,
            )
        )
        assert state.revision == 0
        assert state.grants == ()

    asyncio.run(run())


def test_hosted_discovery_compaction_removes_replay_authority() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        provider = _HostedCompactionProvider()
        remembered = _RememberKnowledgeTool()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=(remembered,),
            tool_discovery_mode="openai_tool_search_hosted",
            context_policy=MessageWindowContextPolicy(max_messages=2),
        )

        initial = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="hosted-discovery-compaction",
                    messages=[Message.text("user", "Load and use the memory tool.")],
                )
            )
        ]
        assert initial[-1].type is EventType.SESSION_COMPLETED
        assert remembered.calls == [{"fact": "Compacted hosted authority."}]

        resumed = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id="hosted-discovery-compaction",
                    messages=[Message.text("user", "Try the old loaded function again.")],
                )
            )
        ]

        assert resumed[-1].type is EventType.SESSION_FAILED
        assert remembered.calls == [{"fact": "Compacted hosted authority."}]
        state = ToolDiscoveryViewState.model_validate(
            await store.load_session_operation(
                "hosted-discovery-compaction",
                TOOL_DISCOVERY_VIEW_OPERATION_KEY,
            )
        )
        assert state.revision == 1
        assert [grant.tool_name for grant in state.grants] == ["remember_knowledge"]

    asyncio.run(run())


def test_hosted_discovery_catalogue_drift_rejects_resume_before_provider_dispatch() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        initial_provider = _HostedDiscoveryProvider(evidence="empty")
        initial_app = CayuApp(session_store=store, enable_logging=False)
        initial_app.register_provider(initial_provider, default=True)
        initial_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=(_RememberKnowledgeTool(),),
            tool_discovery_mode="openai_tool_search_hosted",
        )
        initial = [
            event
            async for event in initial_app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="hosted-discovery-catalogue-drift",
                    messages=[Message.text("user", "Establish the original catalogue.")],
                )
            )
        ]
        assert initial[-1].type is EventType.SESSION_COMPLETED

        changed_provider = _HostedDiscoveryProvider(evidence="empty")
        changed_app = CayuApp(session_store=store, enable_logging=False)
        changed_app.register_provider(changed_provider, default=True)
        changed_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=(_ChangedRememberKnowledgeTool(),),
            tool_discovery_mode="openai_tool_search_hosted",
        )
        with pytest.raises(
            ValueError,
            match="Tool discovery view conflicts with current session authority",
        ):
            _ = [
                event
                async for event in changed_app.resume(
                    ResumeRequest(
                        session_id="hosted-discovery-catalogue-drift",
                        messages=[Message.text("user", "Use the changed catalogue.")],
                    )
                )
            ]

        assert changed_provider.requests == []

    asyncio.run(run())


@pytest.mark.parametrize("evidence", ["missing", "altered", "unrelated"])
def test_hosted_discovery_rejects_untrusted_selection_before_tool_execution(
    evidence: str,
) -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        provider = _HostedDiscoveryProvider(evidence=evidence)
        remembered = _RememberKnowledgeTool()
        policy = _RecordingPolicy()
        hook = _RecordingHook()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=(remembered,),
            tool_discovery_mode="openai_tool_search_hosted",
            tool_policy=policy,
            runtime_hooks=(hook,),
        )

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=f"hosted-discovery-rejected-{evidence}",
                    messages=[Message.text("user", "Do not trust malformed selection.")],
                )
            )
        ]

        assert events[-1].type is EventType.SESSION_FAILED
        assert remembered.calls == []
        assert policy.requests == []
        assert hook.before == []
        assert hook.after == []
        state = ToolDiscoveryViewState.model_validate(
            await store.load_session_operation(
                f"hosted-discovery-rejected-{evidence}",
                TOOL_DISCOVERY_VIEW_OPERATION_KEY,
            )
        )
        assert state.revision == 0
        assert state.grants == ()

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


def test_targeted_native_grant_takes_precedence_over_hosted_candidate_name() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        provider = _HostedDiscoveryAndTargetedProvider()
        remembered = _RememberKnowledgeTool()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=(remembered, _NoiseTool(0)),
            targeted_tool_mode="openai_additional_tools",
            tool_discovery_mode="openai_tool_search_hosted",
        )

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="hosted-discovery-targeted-overlap",
                    messages=[Message.text("user", "Save the lesson and load other tools.")],
                    tool_grants=(
                        TargetedToolGrant(
                            request_id="hosted-targeted-overlap",
                            tool_id="cayu:remember_knowledge",
                            max_calls=1,
                            lifetime_seconds=60,
                        ),
                    ),
                )
            )
        ]

        assert events[-1].type is EventType.SESSION_COMPLETED
        assert remembered.calls == [{"fact": "Targeted authority wins hosted name precedence."}]
        [targeted_record] = await store.list_targeted_tool_grants(
            "hosted-discovery-targeted-overlap"
        )
        started = next(
            event
            for event in events
            if event.type is EventType.TOOL_CALL_STARTED and event.tool_name == "remember_knowledge"
        )
        assert started.payload["grant_id"] == targeted_record.grant_id
        discovery_view = await app.inspect_tool_discovery_view("hosted-discovery-targeted-overlap")
        assert [grant.tool_name for grant in discovery_view.grants] == ["noise_000"]

    asyncio.run(run())


def test_targeted_native_grant_can_cover_the_complete_hosted_candidate_set() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        provider = _HostedDiscoveryAndTargetedProvider(load_noise=False)
        remembered = _RememberKnowledgeTool()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=(remembered,),
            targeted_tool_mode="openai_additional_tools",
            tool_discovery_mode="openai_tool_search_hosted",
        )

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="hosted-discovery-targeted-complete-overlap",
                    messages=[Message.text("user", "Use the targeted memory tool.")],
                    tool_grants=(
                        TargetedToolGrant(
                            request_id="hosted-targeted-complete-overlap",
                            tool_id="cayu:remember_knowledge",
                            max_calls=1,
                            lifetime_seconds=60,
                        ),
                    ),
                )
            )
        ]

        assert events[-1].type is EventType.SESSION_COMPLETED
        assert remembered.calls == [{"fact": "Targeted authority wins hosted name precedence."}]
        discovery_view = await app.inspect_tool_discovery_view(
            "hosted-discovery-targeted-complete-overlap"
        )
        assert discovery_view.revision == 0
        assert discovery_view.grants == ()
        assert all(
            request.tool_discovery_projection is not None
            and request.tool_discovery_projection.candidate_tools == ()
            for request in provider.requests
        )

    asyncio.run(run())


def test_hosted_discovery_approval_resume_uses_the_published_branch_grant() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        provider = _HostedDiscoveryProvider()
        remembered = _RememberKnowledgeTool()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=(remembered,),
            tool_discovery_mode="openai_tool_search_hosted",
            tool_policy=_RequireRememberApprovalPolicy(),
        )

        paused = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="hosted-discovery-approval",
                    messages=[Message.text("user", "Find and save the lesson.")],
                )
            )
        ]

        approval = next(
            event for event in paused if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        assert paused[-1].type is EventType.SESSION_INTERRUPTED
        assert remembered.calls == []
        view = ToolDiscoveryViewState.model_validate(
            await store.load_session_operation(
                "hosted-discovery-approval",
                TOOL_DISCOVERY_VIEW_OPERATION_KEY,
            )
        )
        assert [grant.tool_name for grant in view.grants] == ["remember_knowledge"]

        resumed = [
            event
            async for event in app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id="hosted-discovery-approval",
                    approval_id=approval.payload["approval_id"],
                    tool_round_id=approval.payload["tool_round_id"],
                    tool_call_id=approval.payload["tool_call_id"],
                    decision=ToolApprovalDecision.APPROVE,
                )
            )
        ]

        assert resumed[-1].type is EventType.SESSION_COMPLETED
        assert remembered.calls == [
            {"fact": "Hosted discovery binds durable authority atomically."}
        ]

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
