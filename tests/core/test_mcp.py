from __future__ import annotations

import asyncio
import base64
import hashlib
import inspect
import json
import logging
import sqlite3
import sys
import traceback
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr
from tests.provider_traceback_assertions import is_cayu_source_filename

import cayu.mcp as mcp_module
import cayu.runtime._execution_profile_admission as execution_profile_admission
from cayu import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    DEFAULT_MCP_MAX_LIST_ITEMS,
    DEFAULT_MCP_MAX_LIST_PAGES,
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentSpec,
    Event,
    EventQuery,
    EventType,
    ExecutionProfileMismatchError,
    ForkSessionRequest,
    HttpMcpSession,
    McpClient,
    McpIdleTimeoutError,
    McpInitializeResult,
    McpManifestBaseline,
    McpManifestBaselineLoadResult,
    McpManifestPolicy,
    McpManifestPolicyAction,
    McpManifestPublicationResult,
    McpProtocolError,
    McpResourceDefinition,
    McpResourceResult,
    McpServerSpec,
    McpSession,
    McpToolAdapter,
    McpToolDefinition,
    McpToolResult,
    McpToolset,
    McpToolsetRefreshBlocked,
    McpToolsetRefreshResult,
    McpToolsetRefreshState,
    McpToolsetUnavailable,
    Message,
    PublicAuthorityAliasCodec,
    PublicAuthorityAliasKeyring,
    ResumeRequest,
    RunRequest,
    SessionIdentity,
    SessionStore,
    SQLiteSessionStore,
    StaticToolExposurePolicy,
    StdioMcpProcessLifetime,
    StdioMcpSession,
    TargetedToolGrant,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolCapabilityCeiling,
    ToolContext,
    ToolEffect,
    connect_mcp_toolset,
    mcp_cayu_tool_name,
    mcp_tool_manifest_hash,
    mcp_tool_manifest_identity,
    mcp_tool_manifest_tools,
    mcp_toolset_manifest_diff,
)
from cayu import (
    StdioMcpClient as _StdioMcpClient,
)
from cayu.mcp._jsonrpc import MCP_PROTOCOL_VERSION
from cayu.mcp._stdio_process import stdio_mcp_parent_death_containment_platform_candidate
from cayu.mcp.base import _mcp_session_close_task, _retain_mcp_session_close
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.providers.base import (
    OPENAI_ADDITIONAL_TOOLS_PROTOCOL,
    OPENAI_CLIENT_TOOL_SEARCH_PROTOCOL,
    OPENAI_HOSTED_TOOL_SEARCH_PROTOCOL,
    ToolDiscoveryProjectionResult,
)
from cayu.runtime import (
    InMemorySessionStore,
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
)
from cayu.runtime._event_projection import public_event_sequence
from cayu.runtime.checkpoints import (
    ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
    INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
)
from cayu.runtime.hooks import BeforeToolCallHookContext, RuntimeHook, ToolCallHookContext
from cayu.runtime.sessions import (
    _mcp_authoritative_manifest_hash,
    _mcp_manifest_session_ref,
)
from cayu.storage import migrations as schema_migrations
from cayu.vaults import REDACTED_SECRET, SecretRedactor, SecretRef, StaticVault

_FAKE_SERVER = Path(__file__).resolve().parents[1] / "fixtures" / "fake_mcp_server.py"


def StdioMcpClient(*args: Any, **kwargs: Any) -> _StdioMcpClient:
    """Use the strong default where supported; explicitly opt down elsewhere."""

    if (
        "process_lifetime" not in kwargs
        and not stdio_mcp_parent_death_containment_platform_candidate()
    ):
        kwargs["process_lifetime"] = StdioMcpProcessLifetime.GRACEFUL_CLEANUP
    return _StdioMcpClient(*args, **kwargs)


class FakeProvider(ModelProvider):
    name = "fake"

    def __init__(self, events: list[list[ModelStreamEvent]]) -> None:
        self.events = events
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        for event in self.events[len(self.requests) - 1]:
            yield event


class RecordingRefreshPolicy(ToolPolicy):
    def __init__(self) -> None:
        self.requests: list[ToolPolicyRequest] = []

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        self.requests.append(request)
        return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)


class FakeMcpSession(McpSession):
    def __init__(
        self,
        *,
        definitions: tuple[McpToolDefinition, ...] = (),
        initialize_result: McpInitializeResult | None = None,
        list_tools_error: BaseException | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self.definitions = definitions
        self._initialize_result = initialize_result or McpInitializeResult(
            protocol_version="2025-06-18"
        )
        self.list_tools_error = list_tools_error
        self.close_error = close_error
        self.closed = False
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @property
    def initialize_result(self) -> McpInitializeResult:
        return self._initialize_result

    async def list_tools(self) -> tuple[McpToolDefinition, ...]:
        if self.list_tools_error is not None:
            raise self.list_tools_error
        return self.definitions

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> McpToolResult:
        self.calls.append((name, arguments))
        return McpToolResult(content=[{"type": "text", "text": "ok"}])

    async def list_resources(self) -> tuple[McpResourceDefinition, ...]:
        return ()

    async def read_resource(self, uri: str) -> McpResourceResult:
        raise NotImplementedError

    async def close(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


class BlockingRefreshMcpSession(FakeMcpSession):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.refresh_started = asyncio.Event()
        self.release_refresh = asyncio.Event()

    async def list_tools(self) -> tuple[McpToolDefinition, ...]:
        self.refresh_started.set()
        await self.release_refresh.wait()
        return await super().list_tools()


class BlockingCallMcpSession(FakeMcpSession):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.call_started = asyncio.Event()
        self.release_call = asyncio.Event()

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> McpToolResult:
        self.calls.append((name, arguments))
        self.call_started.set()
        await self.release_call.wait()
        return McpToolResult(content=[{"type": "text", "text": "old call settled"}])

    async def _call_tool_with_dispatch_signal(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        dispatch_signal,
    ) -> McpToolResult:
        self.calls.append((name, arguments))
        self.call_started.set()
        dispatch_signal.mark_dispatched()
        await self.release_call.wait()
        return McpToolResult(content=[{"type": "text", "text": "old call settled"}])


class PreDispatchBlockingMcpSession(FakeMcpSession):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.call_entered = asyncio.Event()
        self.release_dispatch = asyncio.Event()

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> McpToolResult:
        self.call_entered.set()
        await self.release_dispatch.wait()
        return await super().call_tool(name, arguments)


class BlockingCloseMcpSession(FakeMcpSession):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.close_started = asyncio.Event()
        self.release_close = asyncio.Event()
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1
        self.close_started.set()
        await self.release_close.wait()
        await super().close()


class FakeMcpClient(McpClient):
    def __init__(self, session: FakeMcpSession) -> None:
        self.session = session

    async def connect(self, server: McpServerSpec) -> McpSession:
        return self.session


class RacingManifestSessionStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self._publication_arrivals = 0
        self._publication_gate = asyncio.Event()
        self._publication_arrival_lock = asyncio.Lock()

    async def compare_and_publish_mcp_manifest_checks(self, *args, **kwargs):
        expected = kwargs["expected_generations"]
        if all(generation is None for generation in expected.values()):
            async with self._publication_arrival_lock:
                self._publication_arrivals += 1
                if self._publication_arrivals == 2:
                    self._publication_gate.set()
            await self._publication_gate.wait()
        return await super().compare_and_publish_mcp_manifest_checks(*args, **kwargs)


class ManifestMutationSessionStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.on_manifest_load: Callable[[], None] | None = None

    async def load_mcp_manifest_baselines(
        self,
        history_keys: tuple[str, ...],
    ) -> McpManifestBaselineLoadResult:
        callback = self.on_manifest_load
        self.on_manifest_load = None
        if callback is not None:
            callback()
        return await super().load_mcp_manifest_baselines(history_keys)


def test_stdio_mcp_client_lists_calls_and_reads_resources() -> None:
    async def run():
        client = StdioMcpClient()
        session = await client.connect(_fake_server_spec())
        try:
            initialize_result = session.initialize_result
            tools = await session.list_tools()
            tool_result = await session.call_tool("echo", {"text": "hello"})
            resources = await session.list_resources()
            resource_result = await session.read_resource("file:///hello.txt")
        finally:
            await session.close()
        return initialize_result, tools, tool_result, resources, resource_result

    initialize_result, tools, tool_result, resources, resource_result = asyncio.run(run())

    assert initialize_result.server_name == "fake-mcp"
    assert initialize_result.server_version == "1.0.0"
    assert initialize_result.instructions == "Use fake MCP tools only when explicitly requested."
    assert initialize_result.capabilities == {"tools": {}, "resources": {}}
    assert [tool.name for tool in tools] == ["echo"]
    assert tools[0].input_schema["required"] == ["text"]
    assert tool_result.content == [{"type": "text", "text": "echo: hello"}]
    assert tool_result.structured_content == {"echoed": "hello"}
    assert [resource.uri for resource in resources] == ["file:///hello.txt"]
    assert resource_result.contents[0]["text"] == "hello from resource"


def test_connect_mcp_toolset_returns_cayu_tool_adapters() -> None:
    async def run():
        toolset = await connect_mcp_toolset(_fake_server_spec(), client=StdioMcpClient())
        try:
            tools = toolset.tools
            result = await tools[0].run(
                ToolContext(session_id="sess_1", agent_name="assistant"),
                {"text": "from adapter"},
            )
            return toolset.initialize_result, toolset.manifest_hash, tools, result
        finally:
            await toolset.close()

    initialize_result, manifest_hash, tools, result = asyncio.run(run())

    assert initialize_result.server_name == "fake-mcp"
    assert len(tools) == 1
    assert tools[0].name == "mcp__local-mcp__echo"
    assert tools[0].mcp_manifest_hash == manifest_hash
    assert manifest_hash.startswith("sha256:")
    assert "original tool 'echo'" in tools[0].description
    assert "Use fake MCP tools only when explicitly requested." in tools[0].description
    assert tools[0].schema["required"] == ["text"]

    assert result.content == (
        'echo: from adapter\n\nStructured MCP content:\n{\n  "echoed": "from adapter"\n}'
    )
    assert result.structured == {
        "mcp_server": "local-mcp",
        "mcp_tool": "echo",
        "mcp_manifest_hash": manifest_hash,
        "mcp_content": [{"type": "text", "text": "echo: from adapter"}],
        "mcp_structured_content": {"echoed": "from adapter"},
    }
    assert result.is_error is False


def test_mcp_toolset_constructor_does_not_expose_refresh_authority() -> None:
    assert tuple(inspect.signature(McpToolset).parameters) == (
        "server",
        "session",
        "definitions",
    )


def test_static_mcp_adapter_registration_does_not_enable_live_refresh() -> None:
    async def run() -> None:
        toolset = _fake_toolset()
        app = CayuApp(enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=toolset.tools,
        )

        with pytest.raises(ValueError, match="explicit mcp_toolsets"):
            await app.refresh_mcp_toolset(toolset)

        assert toolset.refresh_state is McpToolsetRefreshState.READY
        await toolset.tools[0].run(
            ToolContext(session_id="static", agent_name="assistant"),
            {"text": "still callable"},
        )

    asyncio.run(run())


def test_static_mcp_registration_prevents_cross_application_refresh_ownership() -> None:
    toolset = _fake_toolset()
    static_app = CayuApp(enable_logging=False)
    static_app.register_agent(
        AgentSpec(name="static", model="fake-model"),
        tools=toolset.tools,
    )
    refresh_app = CayuApp(enable_logging=False)

    with pytest.raises(ValueError, match="static registrations"):
        refresh_app.register_agent(
            AgentSpec(name="refreshable", model="fake-model"),
            mcp_toolsets=(toolset,),
        )

    assert static_app.list_agents() == ("static",)
    assert refresh_app.list_agents() == ()


def test_distinct_toolset_wrappers_share_one_live_session_source_owner() -> None:
    definitions = _fake_tool_definitions("echo")
    session = FakeMcpSession(definitions=definitions)
    server = _fake_server_spec().model_copy(update={"connection_id": "shared-session"})
    static_toolset = McpToolset(
        server=server,
        session=session,
        definitions=definitions,
    )
    refreshable_toolset = McpToolset(
        server=server,
        session=session,
        definitions=definitions,
    )
    static_app = CayuApp(enable_logging=False)
    refreshable_app = CayuApp(enable_logging=False)

    assert static_toolset._refresh_source is refreshable_toolset._refresh_source
    static_app.register_agent(
        AgentSpec(name="static", model="fake-model"),
        tools=static_toolset.tools,
    )
    with pytest.raises(ValueError, match="static registrations"):
        refreshable_app.register_agent(
            AgentSpec(name="refreshable", model="fake-model"),
            mcp_toolsets=(refreshable_toolset,),
        )

    assert static_app.list_agents() == ("static",)
    assert refreshable_app.list_agents() == ()


def test_static_mcp_registration_can_share_one_source_across_applications() -> None:
    toolset = _fake_toolset()
    first = CayuApp(enable_logging=False)
    second = CayuApp(enable_logging=False)

    first.register_agent(
        AgentSpec(name="first", model="fake-model"),
        tools=toolset.tools,
    )
    second.register_agent(
        AgentSpec(name="second", model="fake-model"),
        tools=toolset.tools,
    )

    assert first.list_agents() == ("first",)
    assert second.list_agents() == ("second",)


def test_refresh_owned_mcp_source_prevents_cross_application_static_registration() -> None:
    toolset = _fake_toolset()
    refresh_app = CayuApp(enable_logging=False)
    refresh_app.register_agent(
        AgentSpec(name="refreshable", model="fake-model"),
        mcp_toolsets=(toolset,),
    )
    static_app = CayuApp(enable_logging=False)

    with pytest.raises(ValueError, match="refresh-owned"):
        static_app.register_agent(
            AgentSpec(name="static", model="fake-model"),
            tools=toolset.tools,
        )

    assert refresh_app.list_agents() == ("refreshable",)
    assert static_app.list_agents() == ()


def test_refreshable_mcp_registration_requires_stable_connection_identity() -> None:
    app = CayuApp(enable_logging=False)

    with pytest.raises(ValueError, match="connection_id"):
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            mcp_toolsets=(_fake_toolset(connection_id=None),),
        )


def test_refreshable_mcp_source_has_one_application_publication_owner() -> None:
    toolset = _fake_toolset()
    first = CayuApp(enable_logging=False)
    second = CayuApp(enable_logging=False)
    first.register_agent(
        AgentSpec(name="first", model="fake-model"),
        mcp_toolsets=(toolset,),
    )

    with pytest.raises(ValueError, match="only one CayuApp"):
        second.register_agent(
            AgentSpec(name="second", model="fake-model"),
            mcp_toolsets=(toolset,),
        )

    assert second.list_agents() == ()


def test_refreshable_mcp_sources_require_unique_application_connection_identities() -> None:
    app = CayuApp(enable_logging=False)
    app.register_agent(
        AgentSpec(name="first", model="fake-model"),
        mcp_toolsets=(_fake_toolset(definitions=_fake_tool_definitions("echo")),),
    )

    with pytest.raises(ValueError, match="unique connection identities"):
        app.register_agent(
            AgentSpec(name="second", model="fake-model"),
            mcp_toolsets=(_fake_toolset(definitions=_fake_tool_definitions("search")),),
        )

    assert app.list_agents() == ("first",)


def test_complete_mcp_source_registration_changes_execution_profile_identity() -> None:
    def resolve(app: CayuApp):
        return execution_profile_admission.resolve_execution_profile_identity(
            registered_agent=app._agents["assistant"],
            provider_name="fake",
            model="fake-model",
            durable_system_prompt=None,
            runtime_name="cayu",
            runtime_version="test",
            redactor=app._secret_redactor,
            process_identity="shared-test-process",
            registered_environment=app._get_registered_environment(None),
            runtime_hooks=app._runtime_hooks,
            loop_policies=app._loop_policies,
            loop_policy_identities=app._loop_policy_execution_profile_identities,
            invocation_loop_policies=(),
            invocation_loop_policy_identities=(),
        )

    static_toolset = _fake_toolset()
    complete_toolset = _fake_toolset()
    static_app = CayuApp(enable_logging=False)
    complete_app = CayuApp(enable_logging=False)
    static_app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=static_toolset.tools,
    )
    complete_app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        mcp_toolsets=(complete_toolset,),
    )

    assert (
        static_app._agents["assistant"].tool_catalogue.revision
        == complete_app._agents["assistant"].tool_catalogue.revision
    )
    assert resolve(static_app).fingerprint != resolve(complete_app).fingerprint


def test_mcp_refresh_keeps_unchanged_snapshot_and_generation() -> None:
    async def run():
        toolset = _fake_toolset()
        adapter = toolset.tools[0]
        app = CayuApp(enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            mcp_toolsets=(toolset,),
        )

        assert adapter._dispatch_authority_is_current()
        result = await app.refresh_mcp_toolset(toolset)
        assert adapter._dispatch_authority_is_current()
        await adapter.run(
            ToolContext(session_id="unchanged", agent_name="assistant"),
            {"text": "same"},
        )
        return result, app.get_agent("assistant"), toolset.session.calls

    result, registered, calls = asyncio.run(run())

    assert result.status == "unchanged"
    assert result.toolset is not None
    assert result.toolset.generation == 1
    assert result.previous_generation == result.generation == 1
    assert result.diff.changed is False
    assert tuple(registered.tools) == ("mcp__local-mcp__echo",)
    assert calls == [("echo", {"text": "same"})]


def test_mcp_refresh_atomically_replaces_every_registered_agent() -> None:
    async def run():
        toolset = _fake_toolset()
        session = toolset.session
        assert isinstance(session, FakeMcpSession)
        app = CayuApp(enable_logging=False)
        for name in ("first", "second"):
            app.register_agent(
                AgentSpec(name=name, model="fake-model"),
                mcp_toolsets=(toolset,),
            )
        stale_adapter = toolset.tools[0]
        session.definitions = _fake_tool_definitions("echo", "search")

        result = await app.refresh_mcp_toolset(toolset)
        first = app.get_agent("first")
        second = app.get_agent("second")
        assert not stale_adapter._dispatch_authority_is_current()
        with pytest.raises(McpToolsetUnavailable, match="stale"):
            await stale_adapter.run(
                ToolContext(session_id="stale", agent_name="first"),
                {"text": "must not dispatch"},
            )
        current_adapter = first.tools["mcp__local-mcp__search"].tool
        assert isinstance(current_adapter, McpToolAdapter)
        assert current_adapter._dispatch_authority_is_current()
        await current_adapter.run(
            ToolContext(session_id="current", agent_name="first"),
            {"text": "dispatch"},
        )
        return result, first, second, session.calls

    result, first, second, calls = asyncio.run(run())

    assert result.status == "accepted"
    assert result.previous_generation == 1
    assert result.generation == 2
    assert result.diff.added_tools == ("mcp__local-mcp__search",)
    assert result.diff.removed_tools == ()
    assert tuple(first.tools) == (
        "mcp__local-mcp__echo",
        "mcp__local-mcp__search",
    )
    assert tuple(second.tools) == tuple(first.tools)
    assert calls == [("search", {"text": "dispatch"})]


def test_mcp_refresh_publishes_new_catalogue_to_later_runtime_sessions() -> None:
    async def run():
        toolset = _fake_toolset()
        session = toolset.session
        assert isinstance(session, FakeMcpSession)
        provider = FakeProvider(
            [
                [ModelStreamEvent.completed({"finish_reason": "stop"})],
                [ModelStreamEvent.completed({"finish_reason": "stop"})],
            ]
        )
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            mcp_toolsets=(toolset,),
        )
        await _collect_events(
            app.run(
                RunRequest(
                    session_id="before-mcp-refresh",
                    agent_name="assistant",
                    messages=[Message.text("user", "before")],
                )
            )
        )

        session.definitions = _fake_tool_definitions("echo", "search")
        refresh = await app.refresh_mcp_toolset(toolset)
        await _collect_events(
            app.run(
                RunRequest(
                    session_id="after-mcp-refresh",
                    agent_name="assistant",
                    messages=[Message.text("user", "after")],
                )
            )
        )
        return refresh, provider.requests

    refresh, requests = asyncio.run(run())

    assert refresh.status == "accepted"
    assert [[tool["name"] for tool in request.tools] for request in requests] == [
        ["mcp__local-mcp__echo"],
        ["mcp__local-mcp__echo", "mcp__local-mcp__search"],
    ]


def test_mcp_refresh_does_not_widen_existing_session_or_fork_ceiling() -> None:
    async def run():
        toolset = _fake_toolset()
        session = toolset.session
        assert isinstance(session, FakeMcpSession)
        completed = [ModelStreamEvent.completed({"finish_reason": "stop"})]
        provider = FakeProvider([completed, completed])
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            mcp_toolsets=(toolset,),
        )
        source_session_id = "mcp-refresh-ceiling-source"
        await _collect_events(
            app.run(
                RunRequest(
                    session_id=source_session_id,
                    agent_name="assistant",
                    messages=[Message.text("user", "Start before the refresh.")],
                )
            )
        )

        session.definitions = _fake_tool_definitions("echo", "search")
        refresh = await app.refresh_mcp_toolset(toolset)
        await _collect_events(
            app.fork_session(
                ForkSessionRequest(
                    source_session_id=source_session_id,
                    session_id="mcp-refresh-ceiling-child",
                )
            )
        )
        resume_errors: list[ExecutionProfileMismatchError] = []
        for resumable_session_id in (
            source_session_id,
            "mcp-refresh-ceiling-child",
        ):
            with pytest.raises(ExecutionProfileMismatchError) as exc_info:
                await _collect_events(
                    app.resume(
                        ResumeRequest(
                            session_id=resumable_session_id,
                            messages=[Message.text("user", "Continue after the refresh.")],
                        )
                    )
                )
            resume_errors.append(exc_info.value)
        await _collect_events(
            app.run(
                RunRequest(
                    session_id="mcp-refresh-ceiling-fresh",
                    agent_name="assistant",
                    messages=[Message.text("user", "Start after the refresh.")],
                )
            )
        )
        source = await store.load(source_session_id)
        child = await store.load("mcp-refresh-ceiling-child")
        return refresh, provider.requests, source, child, resume_errors

    refresh, requests, source, child, resume_errors = asyncio.run(run())

    assert refresh.status == "accepted"
    assert [[tool["name"] for tool in request.tools] for request in requests] == [
        ["mcp__local-mcp__echo"],
        ["mcp__local-mcp__echo", "mcp__local-mcp__search"],
    ]
    assert source is not None and child is not None
    expected_ceiling = ToolCapabilityCeiling(tool_names=("mcp__local-mcp__echo",))
    assert source.tool_capability_ceiling == expected_ceiling
    assert child.tool_capability_ceiling == expected_ceiling
    assert all("direct_tools" in error.changed_component_classes for error in resume_errors)


@pytest.mark.parametrize("refresh_kind", ["changed", "removed"])
def test_mcp_refresh_rejects_frozen_direct_call_before_policy(refresh_kind: str) -> None:
    class BlockingToolCallProvider(ModelProvider):
        name = "blocking-mcp-refresh"

        def __init__(self, tool_name: str) -> None:
            self.tool_name = tool_name
            self.requests: list[ModelRequest] = []
            self.first_request_started = asyncio.Event()
            self.release_first_response = asyncio.Event()

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            if len(self.requests) == 1:
                self.first_request_started.set()
                await self.release_first_response.wait()
                yield ModelStreamEvent.tool_call(
                    id="stale-mcp-call",
                    name=self.tool_name,
                    arguments={"text": "must fail before policy"},
                )
                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                return
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    class RecordingHook(RuntimeHook):
        def __init__(self) -> None:
            self.before: list[str] = []
            self.after: list[str] = []

        async def before_tool_call(self, context: BeforeToolCallHookContext) -> None:
            self.before.append(context.tool_name)

        async def after_tool_call(self, context: ToolCallHookContext) -> None:
            self.after.append(context.tool_name)

    async def run():
        toolset = _fake_toolset()
        session = toolset.session
        assert isinstance(session, FakeMcpSession)
        provider = BlockingToolCallProvider(toolset.tools[0].name)
        policy = RecordingRefreshPolicy()
        hook = RecordingHook()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            mcp_toolsets=(toolset,),
            tool_policy=policy,
            runtime_hooks=(hook,),
        )

        invocation = asyncio.create_task(
            _collect_events(
                app.run(
                    RunRequest(
                        session_id="mcp-refresh-frozen-direct-call",
                        agent_name="assistant",
                        messages=[Message.text("user", "Call the MCP tool.")],
                    )
                )
            )
        )
        await provider.first_request_started.wait()
        session.definitions = (
            ()
            if refresh_kind == "removed"
            else _fake_tool_definitions(
                "echo",
                description="Changed after provider dispatch.",
            )
        )
        refresh = await app.refresh_mcp_toolset(toolset)
        provider.release_first_response.set()
        events = await invocation
        return refresh, events, policy.requests, hook, session.calls

    refresh, events, policy_requests, hook, calls = asyncio.run(run())

    assert refresh.status == "accepted"
    assert policy_requests == []
    assert hook.before == []
    assert hook.after == []
    assert calls == []
    assert not any(event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED for event in events)
    assert not any(event.type is EventType.TOOL_CALL_STARTED for event in events)
    [failed] = [event for event in events if event.type is EventType.TOOL_CALL_FAILED]
    assert failed.tool_name == "mcp__local-mcp__echo"
    assert failed.payload["blocked_by"] == "mcp_catalogue_authority"
    assert failed.payload["reason"] == "mcp_catalogue_authority_unavailable"


def test_mcp_authority_cannot_recover_after_policy_was_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checks: list[bool] = []

    def transient_authority(_adapter: McpToolAdapter) -> bool:
        current = len(checks) >= 2
        checks.append(current)
        return current

    monkeypatch.setattr(
        McpToolAdapter,
        "_dispatch_authority_is_current",
        transient_authority,
    )

    async def run():
        toolset = _fake_toolset()
        session = toolset.session
        assert isinstance(session, FakeMcpSession)
        adapter = toolset.tools[0]
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="transient-authority-mcp-call",
                        name=adapter.name,
                        arguments={"text": "must remain rejected"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [ModelStreamEvent.completed({"finish_reason": "stop"})],
            ]
        )
        policy = RecordingRefreshPolicy()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            mcp_toolsets=(toolset,),
            tool_policy=policy,
        )
        events = await _collect_events(
            app.run(
                RunRequest(
                    session_id="mcp-transient-policy-authority",
                    agent_name="assistant",
                    messages=[Message.text("user", "Call the MCP tool.")],
                )
            )
        )
        return adapter, events, policy.requests, session.calls

    adapter, events, policy_requests, calls = asyncio.run(run())

    assert checks == [False, False]
    assert adapter._dispatch_authority_is_current()
    assert checks == [False, False, True]
    assert policy_requests == []
    assert calls == []
    assert not any(event.type is EventType.TOOL_CALL_STARTED for event in events)


def test_mcp_refresh_revokes_a_direct_call_waiting_for_approval() -> None:
    class ApprovalProvider(ModelProvider):
        name = "approval-mcp-refresh"

        def __init__(self, tool_name: str) -> None:
            self.tool_name = tool_name
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            if len(self.requests) == 1:
                yield ModelStreamEvent.tool_call(
                    id="approval-mcp-call",
                    name=self.tool_name,
                    arguments={"text": "must be revoked while paused"},
                )
                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                return
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    class ApprovalPolicy(ToolPolicy):
        def __init__(self) -> None:
            self.requests: list[ToolPolicyRequest] = []

        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            self.requests.append(request)
            return ToolPolicyResult(
                decision=ToolPolicyDecision.REQUIRE_APPROVAL,
                reason="Test the refresh boundary while approval is pending.",
            )

    async def run():
        toolset = _fake_toolset()
        session = toolset.session
        assert isinstance(session, FakeMcpSession)
        provider = ApprovalProvider(toolset.tools[0].name)
        policy = ApprovalPolicy()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            mcp_toolsets=(toolset,),
            tool_policy=policy,
        )
        session_id = "mcp-refresh-pending-direct-approval"
        paused = await _collect_events(
            app.run(
                RunRequest(
                    session_id=session_id,
                    agent_name="assistant",
                    messages=[Message.text("user", "Call the MCP tool after approval.")],
                )
            )
        )
        approval = next(
            event for event in paused if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        session.definitions = _fake_tool_definitions(
            "echo",
            description="Changed while direct approval was pending.",
        )
        refresh = await app.refresh_mcp_toolset(toolset)
        with pytest.raises(ExecutionProfileMismatchError) as exc_info:
            await _collect_events(
                app.resolve_tool_approval(
                    ToolApprovalRequest(
                        session_id=session_id,
                        approval_id=approval.payload["approval"]["approval_id"],
                        tool_round_id=approval.payload["tool_round_id"],
                        tool_call_id=approval.payload["tool_call_id"],
                        decision=ToolApprovalDecision.APPROVE,
                    )
                )
            )
        return refresh, paused, exc_info.value, policy.requests, session.calls

    refresh, paused, error, policy_requests, calls = asyncio.run(run())

    assert refresh.status == "accepted"
    assert paused[-1].type is EventType.SESSION_INTERRUPTED
    assert calls == []
    assert len(policy_requests) == 1
    assert error.changed_component_classes == ("direct_tools",)


@pytest.mark.parametrize("refresh_kind", ["changed", "removed"])
@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_mcp_refresh_rejects_frozen_targeted_reference_without_consuming_it(
    refresh_kind: str,
    store_kind: str,
    tmp_path: Path,
) -> None:
    class BlockingTargetedProvider(ModelProvider):
        name = "blocking-mcp-targeted-refresh"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []
            self.first_request_started = asyncio.Event()
            self.release_first_response = asyncio.Event()
            self.tool_ref: str | None = None

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            if len(self.requests) == 1:
                self.first_request_started.set()
                await self.release_first_response.wait()
                assert self.tool_ref is not None
                yield ModelStreamEvent.tool_call(
                    id="stale-targeted-mcp-call",
                    name="call_tool",
                    arguments={
                        "tool_ref": self.tool_ref,
                        "arguments": {"text": "must not consume"},
                    },
                )
                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                return
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    async def run():
        toolset = _fake_toolset()
        session = toolset.session
        assert isinstance(session, FakeMcpSession)
        provider = BlockingTargetedProvider()
        policy = RecordingRefreshPolicy()
        store = (
            InMemorySessionStore()
            if store_kind == "memory"
            else SQLiteSessionStore(
                tmp_path / f"mcp-refresh-{refresh_kind}.db",
                public_authority_alias_codec=PublicAuthorityAliasCodec(
                    PublicAuthorityAliasKeyring(
                        active_key_id="mcp-refresh-test",
                        keys={
                            "mcp-refresh-test": SecretStr(
                                base64.urlsafe_b64encode(bytes([123]) * 32)
                                .decode("ascii")
                                .rstrip("=")
                            )
                        },
                    )
                ),
            )
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            mcp_toolsets=(toolset,),
            targeted_tool_mode="call_tool",
            tool_exposure_policy=StaticToolExposurePolicy(
                profile_id="targeted-mcp-only",
                tools=(),
            ),
            tool_policy=policy,
        )
        descriptor = app._agents["assistant"].tool_catalogue.descriptor_for_name(
            toolset.tools[0].name
        )
        session_id = "mcp-refresh-frozen-targeted-reference"
        invocation = asyncio.create_task(
            _collect_events(
                app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "Call the targeted MCP tool.")],
                        tool_grants=(
                            TargetedToolGrant(
                                request_id="targeted-mcp-echo",
                                tool_id=descriptor.tool_id,
                                max_calls=1,
                                lifetime_seconds=60,
                            ),
                        ),
                    )
                )
            )
        )
        await provider.first_request_started.wait()
        [issued] = await app.session_store.list_targeted_tool_grants(session_id)
        provider.tool_ref = issued.tool_ref
        session.definitions = (
            ()
            if refresh_kind == "removed"
            else _fake_tool_definitions(
                "echo",
                description="Changed after targeted reference issuance.",
            )
        )
        refresh = await app.refresh_mcp_toolset(toolset)
        provider.release_first_response.set()
        events = await invocation
        [current] = await app.session_store.list_targeted_tool_grants(session_id)
        outcome = (refresh, events, policy.requests, session.calls, current)
        close = getattr(store, "close", None)
        if close is not None:
            await close()
        return outcome

    refresh, events, policy_requests, calls, record = asyncio.run(run())

    assert refresh.status == "accepted"
    assert policy_requests == []
    assert calls == []
    assert record.used_calls == 0
    [rejected] = [
        event for event in events if event.type is EventType.TARGETED_TOOL_REFERENCE_REJECTED
    ]
    assert rejected.payload["rejection_reason"] == "catalogue_drift"


def test_mcp_refresh_rejects_frozen_native_targeted_call_without_consuming_it() -> None:
    class BlockingNativeTargetedProvider(ModelProvider):
        name = "blocking-native-mcp-targeted-refresh"

        def __init__(self, tool_name: str) -> None:
            self.tool_name = tool_name
            self.requests: list[ModelRequest] = []
            self.first_request_started = asyncio.Event()
            self.release_first_response = asyncio.Event()

        def supports_targeted_tool_projection(self, *, model: str, protocol: str) -> bool:
            return model == "fake-model" and protocol == OPENAI_ADDITIONAL_TOOLS_PROTOCOL

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            if len(self.requests) == 1:
                projection = request.targeted_tool_projection
                assert projection is not None
                assert [tool["name"] for tool in projection.tools] == [self.tool_name]
                self.first_request_started.set()
                await self.release_first_response.wait()
                yield ModelStreamEvent.tool_call(
                    id="stale-native-targeted-mcp-call",
                    name=self.tool_name,
                    arguments={"text": "must not consume"},
                )
                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                return
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    async def run():
        toolset = _fake_toolset()
        session = toolset.session
        assert isinstance(session, FakeMcpSession)
        provider = BlockingNativeTargetedProvider(toolset.tools[0].name)
        policy = RecordingRefreshPolicy()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            mcp_toolsets=(toolset,),
            targeted_tool_mode="openai_additional_tools",
            tool_exposure_policy=StaticToolExposurePolicy(
                profile_id="native-targeted-mcp-only",
                tools=(),
            ),
            tool_policy=policy,
        )
        descriptor = app._agents["assistant"].tool_catalogue.descriptor_for_name(
            toolset.tools[0].name
        )
        session_id = "mcp-refresh-frozen-native-targeted-call"
        invocation = asyncio.create_task(
            _collect_events(
                app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "Call the targeted MCP tool.")],
                        tool_grants=(
                            TargetedToolGrant(
                                request_id="native-targeted-mcp-echo",
                                tool_id=descriptor.tool_id,
                                max_calls=1,
                                lifetime_seconds=60,
                            ),
                        ),
                    )
                )
            )
        )
        await provider.first_request_started.wait()
        session.definitions = _fake_tool_definitions(
            "echo",
            description="Changed after native targeted projection.",
        )
        refresh = await app.refresh_mcp_toolset(toolset)
        provider.release_first_response.set()
        events = await invocation
        [current] = await app.session_store.list_targeted_tool_grants(session_id)
        return refresh, events, policy.requests, session.calls, current

    refresh, events, policy_requests, calls, record = asyncio.run(run())

    assert refresh.status == "accepted"
    assert policy_requests == []
    assert calls == []
    assert record.used_calls == 0
    [rejected] = [
        event for event in events if event.type is EventType.TARGETED_TOOL_REFERENCE_REJECTED
    ]
    assert rejected.tool_name == "mcp__local-mcp__echo"
    assert rejected.payload["rejection_reason"] == "catalogue_drift"


def test_mcp_refresh_rejects_frozen_discovery_reference_before_target_policy() -> None:
    class BlockingDiscoveryProvider(ModelProvider):
        name = "blocking-mcp-discovery-refresh"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []
            self.reference_ready = asyncio.Event()
            self.release_reference_call = asyncio.Event()

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            if len(self.requests) == 1:
                yield ModelStreamEvent.tool_call(
                    id="search-mcp-tools",
                    name="search_tools",
                    arguments={"query": "echo text", "limit": 1},
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
                [match] = search_results[-1].structured["matches"]
                self.reference_ready.set()
                await self.release_reference_call.wait()
                yield ModelStreamEvent.tool_call(
                    id="stale-discovered-mcp-call",
                    name="call_tool",
                    arguments={
                        "tool_ref": match["tool_ref"],
                        "arguments": {"text": "must fail before target policy"},
                    },
                )
                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                return
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    async def run():
        toolset = _fake_toolset()
        session = toolset.session
        assert isinstance(session, FakeMcpSession)
        provider = BlockingDiscoveryProvider()
        policy = RecordingRefreshPolicy()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            mcp_toolsets=(toolset,),
            tool_discovery_mode="search_tools",
            tool_policy=policy,
        )
        invocation = asyncio.create_task(
            _collect_events(
                app.run(
                    RunRequest(
                        session_id="mcp-refresh-frozen-discovery-reference",
                        agent_name="assistant",
                        messages=[Message.text("user", "Find and call the MCP tool.")],
                    )
                )
            )
        )
        await provider.reference_ready.wait()
        session.definitions = _fake_tool_definitions(
            "echo",
            description="Changed after discovery reference issuance.",
        )
        refresh = await app.refresh_mcp_toolset(toolset)
        provider.release_reference_call.set()
        events = await invocation
        return refresh, events, policy.requests, session.calls

    refresh, events, policy_requests, calls = asyncio.run(run())

    assert refresh.status == "accepted"
    assert [request.tool_name for request in policy_requests] == ["search_tools"]
    assert calls == []
    [rejected] = [
        event for event in events if event.type is EventType.TARGETED_TOOL_REFERENCE_REJECTED
    ]
    assert rejected.payload["rejection_reason"] == "catalogue_drift"


@pytest.mark.parametrize(
    ("discovery_mode", "protocol"),
    [
        ("openai_tool_search_client", OPENAI_CLIENT_TOOL_SEARCH_PROTOCOL),
        ("openai_tool_search_hosted", OPENAI_HOSTED_TOOL_SEARCH_PROTOCOL),
    ],
)
def test_mcp_refresh_rejects_frozen_native_discovery_projection_before_target_policy(
    discovery_mode: str,
    protocol: str,
) -> None:
    class BlockingNativeDiscoveryProvider(ModelProvider):
        name = "blocking-native-mcp-discovery-refresh"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []
            self.reference_ready = asyncio.Event()
            self.release_reference_call = asyncio.Event()

        def supports_tool_discovery_projection(self, *, model: str, protocol: str) -> bool:
            return model == "fake-model" and protocol in {
                OPENAI_CLIENT_TOOL_SEARCH_PROTOCOL,
                OPENAI_HOSTED_TOOL_SEARCH_PROTOCOL,
            }

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            projection = request.tool_discovery_projection
            assert projection is not None
            assert projection.protocol == protocol
            if protocol == OPENAI_CLIENT_TOOL_SEARCH_PROTOCOL and len(self.requests) == 1:
                assert projection.loaded_tool_names == ()
                yield ModelStreamEvent.tool_call(
                    id="native-search-mcp-tools",
                    name="search_tools",
                    arguments={"query": "echo text", "limit": 1},
                )
                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                return
            if (protocol == OPENAI_HOSTED_TOOL_SEARCH_PROTOCOL and len(self.requests) > 1) or (
                protocol == OPENAI_CLIENT_TOOL_SEARCH_PROTOCOL and len(self.requests) > 2
            ):
                yield ModelStreamEvent.completed({"finish_reason": "stop"})
                return

            if protocol == OPENAI_CLIENT_TOOL_SEARCH_PROTOCOL:
                assert projection.loaded_tool_names == ("mcp__local-mcp__echo",)
            else:
                assert projection.candidate_tool_names == ("mcp__local-mcp__echo",)
            self.reference_ready.set()
            await self.release_reference_call.wait()
            yield ModelStreamEvent.tool_call(
                id="stale-native-discovered-mcp-call",
                name="mcp__local-mcp__echo",
                arguments={"text": "must fail before target policy"},
            )
            if protocol == OPENAI_HOSTED_TOOL_SEARCH_PROTOCOL:
                yield ModelStreamEvent(
                    type="completed",
                    payload={"finish_reason": "tool_calls"},
                    tool_discovery_result=ToolDiscoveryProjectionResult(
                        loaded_tools=projection.candidate_tools,
                    ),
                )
                return
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})

    async def run():
        toolset = _fake_toolset()
        session = toolset.session
        assert isinstance(session, FakeMcpSession)
        provider = BlockingNativeDiscoveryProvider()
        policy = RecordingRefreshPolicy()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            mcp_toolsets=(toolset,),
            tool_discovery_mode=discovery_mode,
            tool_policy=policy,
        )
        invocation = asyncio.create_task(
            _collect_events(
                app.run(
                    RunRequest(
                        session_id=f"mcp-refresh-frozen-native-discovery-{protocol}",
                        agent_name="assistant",
                        messages=[Message.text("user", "Find and call the MCP tool.")],
                    )
                )
            )
        )
        await provider.reference_ready.wait()
        session.definitions = _fake_tool_definitions(
            "echo",
            description="Changed after native discovery projection.",
        )
        refresh = await app.refresh_mcp_toolset(toolset)
        provider.release_reference_call.set()
        events = await invocation
        return refresh, events, policy.requests, session.calls

    refresh, events, policy_requests, calls = asyncio.run(run())

    assert refresh.status == "accepted"
    expected_policy_tools = (
        ["search_tools"] if protocol == OPENAI_CLIENT_TOOL_SEARCH_PROTOCOL else []
    )
    assert [request.tool_name for request in policy_requests] == expected_policy_tools
    assert calls == []
    assert not any(
        event.type is EventType.TOOL_CALL_STARTED and event.tool_name == "mcp__local-mcp__echo"
        for event in events
    )
    [rejected] = [
        event for event in events if event.type is EventType.TARGETED_TOOL_REFERENCE_REJECTED
    ]
    assert rejected.payload["rejection_reason"] == "catalogue_drift"


def test_mcp_refresh_policy_block_quarantines_until_verified_retry() -> None:
    async def run():
        toolset = _fake_toolset()
        session = toolset.session
        assert isinstance(session, FakeMcpSession)
        app = CayuApp(
            enable_logging=False,
            mcp_manifest_policy=McpManifestPolicy(),
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            mcp_toolsets=(toolset,),
        )
        session.definitions = _fake_tool_definitions("echo", "search")

        with pytest.raises(McpToolsetRefreshBlocked, match="Policy action: block"):
            await app.refresh_mcp_toolset(toolset)
        assert toolset.refresh_state is McpToolsetRefreshState.QUARANTINED
        assert not toolset.tools[0]._dispatch_authority_is_current()
        with pytest.raises(McpToolsetUnavailable, match="not ready"):
            await toolset.tools[0].run(
                ToolContext(session_id="blocked", agent_name="assistant"),
                {"text": "must not dispatch"},
            )
        assert tuple(app.get_agent("assistant").tools) == ("mcp__local-mcp__echo",)

        session.definitions = _fake_tool_definitions("echo")
        retry = await app.refresh_mcp_toolset(toolset)
        assert toolset.tools[0]._dispatch_authority_is_current()
        await toolset.tools[0].run(
            ToolContext(session_id="restored", agent_name="assistant"),
            {"text": "safe"},
        )
        return retry, session.calls

    retry, calls = asyncio.run(run())

    assert retry.status == "unchanged"
    assert retry.generation == 1
    assert retry.toolset.refresh_state is McpToolsetRefreshState.READY
    assert calls == [("echo", {"text": "safe"})]


def test_mcp_refresh_applies_unchanged_manifest_policy_before_restoring_ready() -> None:
    async def run() -> McpToolsetRefreshState:
        toolset = _fake_toolset()
        app = CayuApp(
            enable_logging=False,
            mcp_manifest_policy=McpManifestPolicy(
                on_unchanged=McpManifestPolicyAction.BLOCK,
            ),
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            mcp_toolsets=(toolset,),
        )

        with pytest.raises(McpToolsetRefreshBlocked, match="unchanged"):
            await app.refresh_mcp_toolset(toolset)
        return toolset.refresh_state

    assert asyncio.run(run()) is McpToolsetRefreshState.QUARANTINED


def test_mcp_refresh_fences_calls_before_candidate_publication() -> None:
    async def run():
        definitions = _fake_tool_definitions("echo")
        session = BlockingRefreshMcpSession(definitions=definitions)
        toolset = McpToolset(
            server=_fake_server_spec().model_copy(update={"connection_id": "refresh-fence"}),
            session=session,
            definitions=definitions,
        )
        app = CayuApp(enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            mcp_toolsets=(toolset,),
        )
        session.definitions = _fake_tool_definitions("echo", "search")
        refresh = asyncio.create_task(app.refresh_mcp_toolset(toolset))
        await session.refresh_started.wait()

        assert toolset.refresh_state is McpToolsetRefreshState.REFRESHING
        assert not toolset.tools[0]._dispatch_authority_is_current()
        with pytest.raises(McpToolsetUnavailable, match="not ready"):
            await toolset.tools[0].run(
                ToolContext(session_id="refreshing", agent_name="assistant"),
                {"text": "must not dispatch"},
            )
        session.release_refresh.set()
        result = await refresh
        return result, session.calls

    result, calls = asyncio.run(run())

    assert result.status == "accepted"
    assert calls == []


def test_mcp_generation_fence_cancellation_does_not_retain_call_arguments() -> None:
    secret = "mcp-generation-fence-cancellation-canary"

    async def run() -> asyncio.CancelledError:
        toolset = _fake_toolset()
        source = toolset._refresh_source
        async with source.lock:
            call = asyncio.create_task(
                toolset.tools[0].run(
                    ToolContext(session_id="cancelled-fence", agent_name="assistant"),
                    {"text": secret},
                )
            )
            await asyncio.sleep(0)
            call.cancel()
            with pytest.raises(asyncio.CancelledError) as exc_info:
                await call
        return exc_info.value

    cancellation = asyncio.run(run())

    _assert_traceback_does_not_retain_text(cancellation, secret)


@pytest.mark.parametrize("dispatched", [False, True])
@pytest.mark.parametrize("child_outcome", ["cancelled", "suppressed", "translated"])
def test_mcp_generation_fence_keeps_caller_cancellation_authoritative(
    dispatched: bool,
    child_outcome: str,
) -> None:
    secret = "mcp-generation-fence-authority-canary"

    class CancellationOutcomeSession(FakeMcpSession):
        def __init__(self) -> None:
            super().__init__(definitions=_fake_tool_definitions("echo"))
            self._secret_redactor = SecretRedactor(secret)
            self.call_started = asyncio.Event()

        async def call_tool(self, name: str, arguments: dict[str, Any]) -> McpToolResult:
            del name, arguments
            self.call_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                if child_outcome == "translated":
                    raise RuntimeError(f"extension settlement exposed {secret}") from None
                if child_outcome == "suppressed":
                    return McpToolResult(content=[{"type": "text", "text": "suppressed"}])
                raise
            raise AssertionError("Cancellation test child returned unexpectedly.")

        async def _call_tool_with_dispatch_signal(
            self,
            name: str,
            arguments: dict[str, Any],
            *,
            dispatch_signal,
        ) -> McpToolResult:
            if dispatched:
                dispatch_signal.mark_dispatched()
            return await self.call_tool(name, arguments)

    async def run() -> tuple[asyncio.CancelledError, bool]:
        session = CancellationOutcomeSession()
        toolset = McpToolset(
            server=_fake_server_spec(),
            session=session,
            definitions=session.definitions,
        )
        call = asyncio.create_task(
            toolset.tools[0].run(
                ToolContext(session_id="cancelled-call", agent_name="assistant"),
                {"text": "cancel me"},
            )
        )
        await session.call_started.wait()
        call.cancel(f"caller authority {secret}")
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await call
        return exc_info.value, call.cancelled()

    cancellation, cancelled = asyncio.run(run())

    assert cancelled is True
    assert cancellation.__context__ is None
    assert secret not in "".join(traceback.format_exception(cancellation))
    assert REDACTED_SECRET in str(cancellation)
    if child_outcome == "translated":
        assert isinstance(cancellation.__cause__, McpProtocolError)
        assert REDACTED_SECRET in str(cancellation.__cause__)
    else:
        assert cancellation.__cause__ is None
    _assert_traceback_does_not_retain_text(cancellation, secret)


def test_mcp_refresh_allows_already_dispatched_call_to_settle() -> None:
    async def run():
        definitions = _fake_tool_definitions("echo")
        session = BlockingCallMcpSession(definitions=definitions)
        toolset = McpToolset(
            server=_fake_server_spec().model_copy(update={"connection_id": "in-flight-call"}),
            session=session,
            definitions=definitions,
        )
        app = CayuApp(enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            mcp_toolsets=(toolset,),
        )
        call = asyncio.create_task(
            toolset.tools[0].run(
                ToolContext(session_id="in-flight", agent_name="assistant"),
                {"text": "already dispatched"},
            )
        )
        await session.call_started.wait()
        session.definitions = _fake_tool_definitions("echo", "search")

        refresh = await app.refresh_mcp_toolset(toolset)
        assert call.done() is False
        session.release_call.set()
        result = await call
        return refresh, result, session.calls

    refresh, result, calls = asyncio.run(run())

    assert refresh.status == "accepted"
    assert result.content == "old call settled"
    assert calls == [("echo", {"text": "already dispatched"})]


def test_mcp_refresh_waits_for_custom_session_call_without_dispatch_proof() -> None:
    async def run():
        definitions = _fake_tool_definitions("echo")
        session = PreDispatchBlockingMcpSession(definitions=definitions)
        toolset = McpToolset(
            server=_fake_server_spec().model_copy(update={"connection_id": "pre-dispatch-call"}),
            session=session,
            definitions=definitions,
        )
        app = CayuApp(enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            mcp_toolsets=(toolset,),
        )
        call = asyncio.create_task(
            toolset.tools[0].run(
                ToolContext(session_id="pre-dispatch", agent_name="assistant"),
                {"text": "must run before refresh"},
            )
        )
        await session.call_entered.wait()
        session.definitions = _fake_tool_definitions("echo", "search")
        refresh = asyncio.create_task(app.refresh_mcp_toolset(toolset))
        await asyncio.sleep(0)

        assert refresh.done() is False
        assert toolset.refresh_state is McpToolsetRefreshState.READY
        assert session.calls == []

        session.release_dispatch.set()
        result = await call
        assert session.calls == [("echo", {"text": "must run before refresh"})]
        refreshed = await refresh
        return result, refreshed

    result, refreshed = asyncio.run(run())

    assert result.content == "ok"
    assert refreshed.status == "accepted"
    assert refreshed.generation == 2


def test_http_mcp_refresh_can_publish_after_transport_owns_pending_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run():
        server = McpServerSpec(
            name="http",
            connection_id="http-dispatch-proof",
            url="https://mcp.example/rpc",
        )
        definitions = _fake_tool_definitions("echo")
        session = HttpMcpSession(
            server=server,
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(lambda _: httpx.Response(500))
            ),
            url="https://mcp.example/rpc",
            client_name="cayu-test",
            client_version="1",
        )
        session._initialize_result = McpInitializeResult(protocol_version=MCP_PROTOCOL_VERSION)
        call_dispatched = asyncio.Event()
        release_call = asyncio.Event()

        async def send(
            payload: dict[str, Any],
            request_id: int,
            *,
            budget: Any,
            failure_redactor: SecretRedactor | None = None,
        ) -> dict[str, Any]:
            del budget, failure_redactor
            method = payload.get("method")
            payload.clear()
            if method == "tools/call":
                call_dispatched.set()
                await release_call.wait()
                result = {"content": [{"type": "text", "text": "old call settled"}]}
            elif method == "tools/list":
                result = {
                    "tools": [
                        definition.model_dump(mode="json", by_alias=True)
                        for definition in _fake_tool_definitions("echo", "search")
                    ]
                }
            else:  # pragma: no cover - the focused transport path has no other request
                raise AssertionError(f"Unexpected MCP request: {method}")
            return {"jsonrpc": "2.0", "id": request_id, "result": result}

        monkeypatch.setattr(session, "_send", send)
        toolset = McpToolset(server=server, session=session, definitions=definitions)
        app = CayuApp(enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            mcp_toolsets=(toolset,),
        )
        call = asyncio.create_task(
            toolset.tools[0].run(
                ToolContext(session_id="http-in-flight", agent_name="assistant"),
                {"text": "already dispatched"},
            )
        )
        try:
            await asyncio.wait_for(call_dispatched.wait(), timeout=1.0)
            refreshed = await asyncio.wait_for(app.refresh_mcp_toolset(toolset), timeout=1.0)
            assert call.done() is False
            release_call.set()
            result = await call
            return refreshed, result
        finally:
            release_call.set()
            with suppress(BaseException):
                await call
            await toolset.close()

    refreshed, result = asyncio.run(run())

    assert refreshed.status == "accepted"
    assert refreshed.generation == 2
    assert result.content == "old call settled"


def test_independent_mcp_sources_refresh_concurrently_without_lost_publication() -> None:
    async def run():
        first_definitions = _fake_tool_definitions("echo")
        second_definitions = _fake_tool_definitions("lookup")
        first_session = BlockingRefreshMcpSession(definitions=first_definitions)
        second_session = BlockingRefreshMcpSession(definitions=second_definitions)
        first_toolset = McpToolset(
            server=_fake_server_spec().model_copy(
                update={"name": "first", "connection_id": "first-source"}
            ),
            session=first_session,
            definitions=first_definitions,
        )
        second_toolset = McpToolset(
            server=_fake_server_spec().model_copy(
                update={"name": "second", "connection_id": "second-source"}
            ),
            session=second_session,
            definitions=second_definitions,
        )
        app = CayuApp(enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            mcp_toolsets=(first_toolset, second_toolset),
        )
        first_session.definitions = _fake_tool_definitions("echo", "search")
        second_session.definitions = _fake_tool_definitions("lookup", "inspect")

        first_refresh = asyncio.create_task(app.refresh_mcp_toolset(first_toolset))
        second_refresh = asyncio.create_task(app.refresh_mcp_toolset(second_toolset))
        await asyncio.wait_for(
            asyncio.gather(
                first_session.refresh_started.wait(),
                second_session.refresh_started.wait(),
            ),
            timeout=1.0,
        )
        first_session.release_refresh.set()
        second_session.release_refresh.set()
        results = await asyncio.gather(first_refresh, second_refresh)
        return results, tuple(app.get_agent("assistant").tools)

    results, tools = asyncio.run(run())

    assert [result.status for result in results] == ["accepted", "accepted"]
    assert [result.generation for result in results] == [2, 2]
    assert tools == (
        "mcp__first__echo",
        "mcp__first__search",
        "mcp__second__inspect",
        "mcp__second__lookup",
    )


def test_mcp_refresh_transport_failure_quarantines_until_retry() -> None:
    async def run():
        toolset = _fake_toolset()
        session = toolset.session
        assert isinstance(session, FakeMcpSession)
        app = CayuApp(enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            mcp_toolsets=(toolset,),
        )
        session.list_tools_error = McpProtocolError("refresh failed")

        with pytest.raises(McpProtocolError, match="refresh failed"):
            await app.refresh_mcp_toolset(toolset)
        assert toolset.refresh_state is McpToolsetRefreshState.QUARANTINED
        with pytest.raises(McpToolsetUnavailable, match="not ready"):
            await toolset.tools[0].run(
                ToolContext(session_id="failed", agent_name="assistant"),
                {"text": "must not dispatch"},
            )

        session.list_tools_error = None
        retry = await app.refresh_mcp_toolset(toolset)
        return retry, toolset.refresh_state

    retry, state = asyncio.run(run())

    assert retry.status == "unchanged"
    assert state is McpToolsetRefreshState.READY


def test_mcp_refresh_redacts_transport_failure_before_quarantine() -> None:
    secret = "mcp-refresh-transport-secret-canary"

    class SecretRefreshSession(FakeMcpSession):
        def __init__(self) -> None:
            super().__init__(definitions=_fake_tool_definitions("echo"))
            self._secret_redactor = SecretRedactor(secret)

        async def list_tools(self) -> tuple[McpToolDefinition, ...]:
            raise RuntimeError(f"server leaked {secret}")

    async def run() -> tuple[BaseException, McpToolsetRefreshState]:
        session = SecretRefreshSession()
        toolset = McpToolset(
            server=_fake_server_spec().model_copy(update={"connection_id": "redacted"}),
            session=session,
            definitions=session.definitions,
        )
        app = CayuApp(enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            mcp_toolsets=(toolset,),
        )
        with pytest.raises(McpProtocolError) as exc_info:
            await app.refresh_mcp_toolset(toolset)
        return exc_info.value, toolset.refresh_state

    error, state = asyncio.run(run())

    assert state is McpToolsetRefreshState.QUARANTINED
    assert secret not in "".join(traceback.format_exception(error))
    _assert_traceback_does_not_retain_text(error, secret)


def test_mcp_refresh_detaches_rejected_secret_definitions_before_quarantine() -> None:
    secret = "mcp-refresh-definition-secret-canary"
    definition = McpToolDefinition(
        name="duplicate",
        description=f"private description {secret}",
        input_schema={"type": "object"},
    )

    async def run() -> tuple[BaseException, McpToolsetRefreshState]:
        session = FakeMcpSession(definitions=_fake_tool_definitions("echo"))
        session._secret_redactor = SecretRedactor(secret)
        toolset = McpToolset(
            server=_fake_server_spec().model_copy(update={"connection_id": "redacted-definition"}),
            session=session,
            definitions=session.definitions,
        )
        app = CayuApp(enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            mcp_toolsets=(toolset,),
        )
        session.definitions = (definition, definition)
        with pytest.raises(McpProtocolError) as exc_info:
            await app.refresh_mcp_toolset(toolset)
        return exc_info.value, toolset.refresh_state

    error, state = asyncio.run(run())

    assert state is McpToolsetRefreshState.QUARANTINED
    assert secret not in "".join(traceback.format_exception(error))
    _assert_traceback_does_not_retain_text(error, secret)


def test_mcp_refresh_cancellation_quarantines_source() -> None:
    async def run():
        definitions = _fake_tool_definitions("echo")
        session = BlockingRefreshMcpSession(definitions=definitions)
        toolset = McpToolset(
            server=_fake_server_spec().model_copy(update={"connection_id": "cancelled"}),
            session=session,
            definitions=definitions,
        )
        app = CayuApp(enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            mcp_toolsets=(toolset,),
        )
        refresh = asyncio.create_task(app.refresh_mcp_toolset(toolset))
        await session.refresh_started.wait()
        refresh.cancel()
        with pytest.raises(asyncio.CancelledError):
            await refresh
        return toolset.refresh_state

    assert asyncio.run(run()) is McpToolsetRefreshState.QUARANTINED


def test_mcp_close_wins_in_flight_refresh_with_typed_fenced_outcome() -> None:
    async def run() -> tuple[BaseException, McpToolsetRefreshState]:
        definitions = _fake_tool_definitions("echo")
        session = BlockingRefreshMcpSession(definitions=definitions)
        toolset = McpToolset(
            server=_fake_server_spec().model_copy(update={"connection_id": "closed-refresh"}),
            session=session,
            definitions=definitions,
        )
        app = CayuApp(enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            mcp_toolsets=(toolset,),
        )
        refresh = asyncio.create_task(app.refresh_mcp_toolset(toolset))
        await session.refresh_started.wait()
        await toolset.close()
        session.release_refresh.set()

        with pytest.raises(McpToolsetUnavailable, match="closed during refresh") as exc_info:
            await refresh
        return exc_info.value, toolset.refresh_state

    error, state = asyncio.run(run())

    assert state is McpToolsetRefreshState.CLOSED
    assert error.__cause__ is None


def test_concurrent_mcp_close_callers_await_session_cleanup() -> None:
    async def run() -> tuple[bool, int, McpToolsetRefreshState]:
        definitions = _fake_tool_definitions("echo")
        session = BlockingCloseMcpSession(definitions=definitions)
        toolset = McpToolset(
            server=_fake_server_spec(),
            session=session,
            definitions=definitions,
        )
        first = asyncio.create_task(toolset.close())
        await session.close_started.wait()
        second = asyncio.create_task(toolset.close())
        await asyncio.sleep(0)
        second_returned_early = second.done()
        session.release_close.set()
        await asyncio.gather(first, second)
        return second_returned_early, session.close_calls, toolset.refresh_state

    second_returned_early, close_calls, state = asyncio.run(run())

    assert second_returned_early is False
    assert close_calls == 2
    assert state is McpToolsetRefreshState.CLOSED


def test_mcp_refresh_collision_rejects_candidate_without_partial_publication() -> None:
    async def run():
        toolset = _fake_toolset()
        session = toolset.session
        assert isinstance(session, FakeMcpSession)

        static_toolset = _fake_toolset(
            definitions=_fake_tool_definitions("search"),
            connection_id="static-collision",
        )
        colliding_tool = static_toolset.tools[0]
        app = CayuApp(enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=(colliding_tool,),
            mcp_toolsets=(toolset,),
        )
        previous_agent = app._agents["assistant"]
        session.definitions = _fake_tool_definitions("echo", "search")

        with pytest.raises(ValueError, match="collides"):
            await app.refresh_mcp_toolset(toolset)
        return toolset.refresh_state, previous_agent, app._agents["assistant"]

    state, previous_agent, current_agent = asyncio.run(run())

    assert state is McpToolsetRefreshState.QUARANTINED
    assert current_agent is previous_agent
    assert tuple(current_agent.tools) == (
        "mcp__local-mcp__search",
        "mcp__local-mcp__echo",
    )


def test_mcp_toolset_manifest_diff_rejects_unrelated_sources() -> None:
    first = _fake_toolset()
    second = _fake_toolset()

    with pytest.raises(ValueError, match="one source"):
        mcp_toolset_manifest_diff(first, second)


def test_mcp_refresh_classifies_provider_rename_with_stable_cayu_name_as_changed() -> None:
    async def run() -> McpToolsetRefreshResult:
        original = McpToolDefinition(
            name="lookup value",
            description="Lookup a value.",
            input_schema={"type": "object"},
        )
        renamed = original.model_copy(update={"name": "lookup@value"})
        session = FakeMcpSession(definitions=(original,))
        toolset = McpToolset(
            server=_fake_server_spec().model_copy(update={"connection_id": "stable-alias"}),
            session=session,
            definitions=session.definitions,
        )
        app = CayuApp(
            enable_logging=False,
            mcp_manifest_policy=McpManifestPolicy(
                on_tools_added=McpManifestPolicyAction.BLOCK,
                on_tools_removed=McpManifestPolicyAction.BLOCK,
                on_tools_changed=McpManifestPolicyAction.ALLOW,
            ),
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            mcp_toolsets=(toolset,),
        )
        session.definitions = (renamed,)

        return await app.refresh_mcp_toolset(toolset)

    result = asyncio.run(run())

    assert result.status == "accepted"
    assert result.policy_action == "allow"
    assert result.diff.added_tools == ()
    assert result.diff.removed_tools == ()
    assert result.diff.changed_tools == ("mcp__local-mcp__lookup_value",)


def test_mcp_refresh_classifies_private_rename_under_stable_public_name() -> None:
    first_private_name = "mcp-private-name-alpha"
    second_private_name = "mcp-private-name-beta"

    async def run() -> tuple[McpToolsetRefreshResult, str, list[tuple[str, dict[str, Any]]]]:
        original = _fake_tool_definitions(first_private_name)
        session = FakeMcpSession(definitions=original)
        session._secret_redactor = SecretRedactor((first_private_name, second_private_name))
        toolset = McpToolset(
            server=_fake_server_spec().model_copy(update={"connection_id": "private-rename"}),
            session=session,
            definitions=original,
        )
        public_name = toolset.tools[0].name
        app = CayuApp(
            enable_logging=False,
            mcp_manifest_policy=McpManifestPolicy(
                on_tools_added=McpManifestPolicyAction.BLOCK,
                on_tools_removed=McpManifestPolicyAction.BLOCK,
                on_tools_changed=McpManifestPolicyAction.ALLOW,
            ),
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            mcp_toolsets=(toolset,),
        )
        session.definitions = _fake_tool_definitions(second_private_name)

        result = await app.refresh_mcp_toolset(toolset)
        await result.toolset.tools[0].run(
            ToolContext(session_id="private-rename", agent_name="assistant"),
            {"text": "new binding"},
        )
        return result, public_name, session.calls

    result, public_name, calls = asyncio.run(run())

    assert result.status == "accepted"
    assert result.policy_action == "allow"
    assert result.diff.added_tools == ()
    assert result.diff.removed_tools == ()
    assert result.diff.changed_tools == (public_name,)
    assert calls == [(second_private_name, {"text": "new binding"})]
    public_result = repr(result.diff.policy_input())
    assert first_private_name not in public_result
    assert second_private_name not in public_result


def test_mcp_refresh_never_exposes_private_name_for_private_contract_change() -> None:
    private_name = "mcp-private-contract-name"
    first_private_description = "mcp-private-description-alpha"
    second_private_description = "mcp-private-description-beta"

    async def run() -> tuple[McpToolsetRefreshResult, str]:
        original = _fake_tool_definitions(
            private_name,
            description=first_private_description,
        )
        session = FakeMcpSession(definitions=original)
        session._secret_redactor = SecretRedactor(
            (private_name, first_private_description, second_private_description)
        )
        toolset = McpToolset(
            server=_fake_server_spec().model_copy(
                update={"connection_id": "private-contract-change"}
            ),
            session=session,
            definitions=original,
        )
        public_name = toolset.tools[0].name
        app = CayuApp(
            enable_logging=False,
            mcp_manifest_policy=McpManifestPolicy(
                on_tools_changed=McpManifestPolicyAction.ALLOW,
            ),
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            mcp_toolsets=(toolset,),
        )
        session.definitions = _fake_tool_definitions(
            private_name,
            description=second_private_description,
        )
        return await app.refresh_mcp_toolset(toolset), public_name

    result, public_name = asyncio.run(run())

    assert result.status == "accepted"
    assert result.diff.changed_tools == (public_name,)
    public_result = repr(result.diff.policy_input())
    assert private_name not in public_result
    assert first_private_description not in public_result
    assert second_private_description not in public_result


def test_agent_catalogue_reuses_authoritative_mcp_contract_identity() -> None:
    toolset = _fake_toolset()
    adapter = toolset.tools[0]
    binding = adapter._manifest_binding
    app = CayuApp(enable_logging=False)

    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=(adapter,),
    )

    catalogue = app._agents["assistant"].tool_catalogue
    descriptor = catalogue.descriptor_for_name(adapter.name)
    assert descriptor.provenance.kind == "mcp"
    assert descriptor.provenance.source_id == toolset.manifest_identity
    assert descriptor.provenance.source_contract_fingerprint == binding.manifest_contract_hash
    assert descriptor.provenance.source_tool_fingerprint is not None
    assert descriptor.tool_id.startswith("mcp:")
    assert binding.manifest_mcp_name not in descriptor.tool_id


def test_mcp_tool_adapter_includes_structured_content_in_model_text() -> None:
    async def run():
        toolset = await connect_mcp_toolset(_fake_server_spec(), client=StdioMcpClient())
        try:
            return await toolset.tools[0].run(
                ToolContext(session_id="sess_1", agent_name="assistant"),
                {"text": "structured", "structured_only": True},
            )
        finally:
            await toolset.close()

    result = asyncio.run(run())

    assert result.content == 'Structured MCP content:\n{\n  "echoed": "structured"\n}'
    assert result.structured["mcp_structured_content"] == {"echoed": "structured"}


def test_mcp_tool_adapter_rejects_deep_arguments_before_session_dispatch(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    recwarn: pytest.WarningsRecorder,
) -> None:
    secret = "mcp-adapter-deep-argument-secret-canary"
    definition = McpToolDefinition(name="echo", input_schema={"type": "object"})
    session = FakeMcpSession(definitions=(definition,))
    toolset = McpToolset(
        server=_fake_server_spec(),
        session=session,
        definitions=session.definitions,
    )
    value: Any = secret
    for _ in range(1_500):
        value = [value]
    arguments = {"large_integer_sibling": 10**5_000, "nested": value}

    from cayu.mcp import _transport as mcp_transport_module

    def fail_if_depth_walk_renders_key(*_args: Any, **_kwargs: Any) -> bool:
        raise AssertionError("depth-only MCP validation rendered an object key")

    monkeypatch.setattr(
        mcp_transport_module._McpJsonUtf8SizeCounter,
        "_string",
        fail_if_depth_walk_renders_key,
    )

    async def run() -> BaseException:
        with pytest.raises(McpProtocolError, match="supported JSON nesting") as exc_info:
            await toolset.tools[0].run(
                ToolContext(session_id="deep-argument", agent_name="test"),
                arguments,
            )
        return exc_info.value

    with caplog.at_level(logging.DEBUG):
        error = asyncio.run(run())

    assert session.calls == []
    assert secret not in "".join(traceback.format_exception(error))
    assert error.__cause__ is None
    assert error.__context__ is None
    _assert_traceback_does_not_retain_text(error, secret)
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in record.getMessage() for record in caplog.records)
    assert all(secret not in str(warning.message) for warning in recwarn)
    arguments.clear()


def test_mcp_tool_adapter_custom_session_owns_arguments_and_sanitizes_invalid_input() -> None:
    secret = "mcp-custom-session-invalid-argument-secret-canary"

    class SecretSession(FakeMcpSession):
        def __init__(self) -> None:
            super().__init__(
                definitions=(McpToolDefinition(name="echo", input_schema={"type": "object"}),)
            )
            self._secret_redactor = SecretRedactor(secret)

    session = SecretSession()
    toolset = McpToolset(
        server=_fake_server_spec(),
        session=session,
        definitions=session.definitions,
    )
    valid_arguments = {"nested": {"value": "original"}}

    async def run() -> tuple[BaseException, dict[str, Any]]:
        invalid_arguments = {secret: object()}
        with pytest.raises(McpProtocolError) as exc_info:
            await toolset.tools[0].run(
                ToolContext(session_id="invalid-custom-arguments", agent_name="test"),
                invalid_arguments,
            )
        await toolset.tools[0].run(
            ToolContext(session_id="valid-custom-arguments", agent_name="test"),
            valid_arguments,
        )
        valid_arguments["nested"]["value"] = "mutated"
        return exc_info.value, session.calls[0][1]

    error, received = asyncio.run(run())

    assert secret not in "".join(traceback.format_exception(error))
    _assert_traceback_does_not_retain_text(error, secret)
    assert session.calls[0][0] == "echo"
    assert received == {"nested": {"value": "original"}}


def test_mcp_tool_adapter_redacts_injected_secrets_echoed_by_server() -> None:
    # N3: a hostile/buggy MCP server can echo an injected secret (secret_env/
    # secret_headers) back through tool content/structured output. The toolset must
    # scrub it before it reaches model-visible context.
    secret = "sk-super-secret-mcp-value"

    class RedactingSession(FakeMcpSession):
        def __init__(self) -> None:
            super().__init__(
                definitions=(McpToolDefinition(name="echo", input_schema={"type": "object"}),)
            )
            self._secret_redactor = SecretRedactor((secret,))

        async def call_tool(self, name: str, arguments: dict[str, Any]) -> McpToolResult:
            return McpToolResult(
                content=[{"type": "text", "text": f"here is your token: {secret}"}],
                structured_content={"token": secret, "nested": {"also": secret}},
            )

    session = RedactingSession()
    toolset = McpToolset(
        server=_fake_server_spec(),
        session=session,
        definitions=session.definitions,
    )
    result = asyncio.run(
        toolset.tools[0].run(ToolContext(session_id="sess_1", agent_name="assistant"), {})
    )

    # Rendered model text is scrubbed.
    assert secret not in result.content
    assert REDACTED_SECRET in result.content
    # The raw content/structured echoes are scrubbed recursively too.
    assert secret not in json.dumps(result.model_dump(mode="json")["structured"])
    assert result.structured["mcp_content"][0]["text"] == f"here is your token: {REDACTED_SECRET}"
    assert result.structured["mcp_structured_content"]["token"] == REDACTED_SECRET
    assert result.structured["mcp_structured_content"]["nested"]["also"] == REDACTED_SECRET


@pytest.mark.parametrize("secret", ["text", "type"])
def test_mcp_tool_adapter_preserves_text_framing_for_short_schema_secret(
    secret: str,
) -> None:
    class RedactingSession(FakeMcpSession):
        def __init__(self) -> None:
            super().__init__(
                definitions=(McpToolDefinition(name="echo", input_schema={"type": "object"}),)
            )
            self._secret_redactor = SecretRedactor(secret)

        async def call_tool(self, name: str, arguments: dict[str, Any]) -> McpToolResult:
            del name, arguments
            return McpToolResult(
                content=[{"type": "text", "text": "hello world"}],
            )

    session = RedactingSession()
    toolset = McpToolset(
        server=_fake_server_spec(),
        session=session,
        definitions=session.definitions,
    )

    result = asyncio.run(
        toolset.tools[0].run(ToolContext(session_id="sess_1", agent_name="assistant"), {})
    )

    assert result.content == "hello world"
    assert result.structured["mcp_content"] == [{"type": "text", "text": "hello world"}]


@pytest.mark.parametrize(
    "secret_offset_from_boundary",
    [-128, -8, 64],
    ids=["before-boundary", "crosses-boundary", "after-boundary"],
)
def test_mcp_tool_adapter_redacts_structured_secret_before_byte_truncation(
    secret_offset_from_boundary: int,
) -> None:
    secret = "密钥🔐boundary-canary"
    rendered_prefix = 'Structured MCP content:\n{\n  "token": "'
    secret_start = 20_000 + secret_offset_from_boundary
    padding = "a" * (secret_start - len(rendered_prefix.encode("utf-8")))

    class RedactingSession(FakeMcpSession):
        def __init__(self) -> None:
            super().__init__(
                definitions=(McpToolDefinition(name="echo", input_schema={"type": "object"}),)
            )
            self._secret_redactor = SecretRedactor(secret)

        async def call_tool(self, name: str, arguments: dict[str, Any]) -> McpToolResult:
            del name, arguments
            return McpToolResult(
                content=[],
                structured_content={"token": padding + secret + "-suffix"},
            )

    session = RedactingSession()
    toolset = McpToolset(
        server=_fake_server_spec(),
        session=session,
        definitions=session.definitions,
    )
    result = asyncio.run(
        toolset.tools[0].run(ToolContext(session_id="sess_1", agent_name="assistant"), {})
    )

    assert secret not in result.content
    assert secret not in json.dumps(
        result.model_dump(mode="json")["structured"],
        ensure_ascii=False,
    )
    assert result.structured["mcp_structured_content"]["token"].endswith(
        f"{REDACTED_SECRET}-suffix"
    )
    if secret_offset_from_boundary == -8:
        assert "密钥" not in result.content


def test_mcp_toolset_redacts_session_secret_from_provider_definitions_and_manifest() -> None:
    secret = "mcp-definition-boundary-canary"

    class RedactingSession(FakeMcpSession):
        def __init__(self) -> None:
            super().__init__(
                initialize_result=McpInitializeResult(
                    protocol_version=MCP_PROTOCOL_VERSION,
                    server_name=f"server-{secret}",
                    server_version=f"version-{secret}",
                    instructions=f"Always authenticate with {secret}.",
                    capabilities={"authentication": secret},
                ),
                definitions=(
                    McpToolDefinition(
                        name="echo",
                        description=f"Echo using {secret}.",
                        input_schema={
                            "type": "object",
                            "properties": {"text": {"description": f"Authenticated by {secret}"}},
                        },
                        annotations={"title": f"Echo with {secret}"},
                    ),
                ),
            )
            self._secret_redactor = SecretRedactor(secret)

    session = RedactingSession()
    toolset = McpToolset(
        server=_fake_server_spec(),
        session=session,
        definitions=session.definitions,
    )

    serialized = json.dumps(
        {
            "initialize": toolset.initialize_result.model_dump(mode="json"),
            "definitions": [
                definition.model_dump(mode="json") for definition in toolset.definitions
            ],
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "schema": tool.schema,
                }
                for tool in toolset.tools
            ],
            "manifest_tools": toolset.manifest_tools,
            "manifest_hash": toolset.manifest_hash,
        }
    )

    assert secret not in serialized
    assert REDACTED_SECRET in serialized
    binding = toolset.tools[0]._manifest_binding
    assert binding.source_contract_hash != binding.manifest_contract_hash
    assert any(
        entry.mcp_name == binding.manifest_mcp_name
        and entry.contract_hash == binding.manifest_contract_hash
        for entry in toolset._manifest_snapshot.tools
    )


def test_runtime_composes_mcp_session_secrets_before_durable_tool_checkpoint() -> None:
    secret = "mcp-invocation-checkpoint-canary"

    class SecretSession(FakeMcpSession):
        def __init__(self) -> None:
            super().__init__(
                definitions=(
                    McpToolDefinition(
                        name="echo",
                        description="Echo text.",
                        input_schema={"type": "object"},
                    ),
                )
            )
            self._secret_redactor = SecretRedactor(secret)
            self.calls = 0

        async def call_tool(self, name: str, arguments: dict[str, Any]) -> McpToolResult:
            del name, arguments
            self.calls += 1
            return McpToolResult(content=[{"type": "text", "text": "unexpected"}])

    async def run():
        session = SecretSession()
        toolset = McpToolset(
            server=McpServerSpec(
                name="private-mcp",
                connection_id="private-mcp-checkpoint",
                command=["unused"],
            ),
            session=session,
            definitions=session.definitions,
        )
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_secret",
                        name=toolset.tools[0].name,
                        arguments={"token": secret},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ]
            ]
        )
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=toolset.tools,
        )
        events = await _collect_events(
            app.run(
                RunRequest(
                    session_id="mcp_invocation_checkpoint",
                    agent_name="assistant",
                    messages=[Message.text("user", "use the tool")],
                )
            )
        )
        checkpoint = await store.load_checkpoint("mcp_invocation_checkpoint")
        transcript = await store.load_transcript("mcp_invocation_checkpoint")
        return events, checkpoint, transcript, session.calls

    events, checkpoint, transcript, calls = asyncio.run(run())
    serialized = json.dumps(
        {
            "events": [event.model_dump(mode="json") for event in events],
            "checkpoint": checkpoint,
            "transcript": [message.model_dump(mode="json") for message in transcript],
        }
    )

    assert events[-1].type == EventType.SESSION_FAILED
    assert calls == 0
    assert checkpoint is not None
    checkpoint_without_active_profile = dict(checkpoint)
    assert (
        checkpoint_without_active_profile.pop(ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY)
        is not None
    )
    checkpoint_without_active_profile.pop(INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY)
    assert checkpoint_without_active_profile == {
        CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION
    }
    assert secret not in serialized


def test_runtime_uses_mcp_session_redactor_for_policy_denial_branches() -> None:
    secret = "mcp-policy-branch-canary"

    class SecretSession(FakeMcpSession):
        def __init__(self) -> None:
            super().__init__(
                definitions=(
                    McpToolDefinition(
                        name="echo",
                        description="Echo text.",
                        input_schema={"type": "object"},
                    ),
                )
            )
            self._secret_redactor = SecretRedactor(secret)
            self.calls = 0

        async def call_tool(self, name: str, arguments: dict[str, Any]) -> McpToolResult:
            del name, arguments
            self.calls += 1
            return McpToolResult(content=[{"type": "text", "text": "unexpected"}])

    class SecretDenyPolicy(ToolPolicy):
        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            del request
            return ToolPolicyResult(
                decision=ToolPolicyDecision.DENY,
                reason=f"server policy denied {secret}",
                metadata={"diagnostic": secret},
            )

    async def run():
        session = SecretSession()
        toolset = McpToolset(
            server=McpServerSpec(
                name="private-mcp",
                connection_id="private-mcp-policy",
                command=["unused"],
            ),
            session=session,
            definitions=session.definitions,
        )
        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_policy",
                        name=toolset.tools[0].name,
                        arguments={"text": "safe"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [ModelStreamEvent.completed({"finish_reason": "stop"})],
            ]
        )
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=toolset.tools,
            tool_policy=SecretDenyPolicy(),
        )
        emitted = await _collect_events(
            app.run(
                RunRequest(
                    session_id="mcp_policy_branch_redaction",
                    agent_name="assistant",
                    messages=[Message.text("user", "use the tool")],
                )
            )
        )
        persisted = await store.load_events("mcp_policy_branch_redaction")
        return emitted, persisted, session.calls

    emitted, persisted, calls = asyncio.run(run())

    assert calls == 0
    assert any(event.type == EventType.TOOL_CALL_BLOCKED for event in emitted)
    assert emitted[-1].type == EventType.SESSION_COMPLETED
    serialized = json.dumps(
        [event.model_dump(mode="json") for event in [*emitted, *persisted]],
        default=str,
    )
    assert secret not in serialized
    assert REDACTED_SECRET in serialized


def test_mcp_tool_adapter_derives_parallel_safe_from_read_only_hint() -> None:
    # MCP tools must feed the per-tool safety gate: only a server-declared read-only
    # tool may run concurrently. A write, un-annotated, or non-bool-hinted tool is a
    # barrier (parallel_safe=False) so it never races a sibling in a parallel round.
    definitions = (
        McpToolDefinition(name="read", input_schema={}, annotations={"readOnlyHint": True}),
        McpToolDefinition(name="idem", input_schema={}, annotations={"idempotentHint": True}),
        McpToolDefinition(
            name="read_idem",
            input_schema={},
            annotations={"readOnlyHint": True, "idempotentHint": True},
        ),
        McpToolDefinition(name="unknown", input_schema={}),
        McpToolDefinition(name="write", input_schema={}, annotations={"readOnlyHint": False}),
        McpToolDefinition(
            name="spoof",
            input_schema={},
            annotations={"readOnlyHint": "true", "idempotentHint": "true"},
        ),
    )
    session = FakeMcpSession(definitions=definitions)
    toolset = McpToolset(
        server=_fake_server_spec(),
        session=session,
        definitions=definitions,
    )
    parallel_safe = {tool.definition.name: tool.spec.parallel_safe for tool in toolset.tools}
    effects = {tool.definition.name: tool.spec.effect for tool in toolset.tools}

    assert parallel_safe["read"] is True
    assert parallel_safe["idem"] is False
    assert parallel_safe["read_idem"] is True
    assert parallel_safe["unknown"] is False
    assert parallel_safe["write"] is False
    assert parallel_safe["spoof"] is False
    assert effects["read"] is ToolEffect.NONE
    assert effects["idem"] is ToolEffect.IDEMPOTENT
    assert effects["read_idem"] is ToolEffect.NONE
    assert effects["unknown"] is ToolEffect.EXTERNAL
    assert effects["write"] is ToolEffect.EXTERNAL
    assert effects["spoof"] is ToolEffect.EXTERNAL


def test_mcp_tool_manifest_hash_is_stable_for_equivalent_json_order() -> None:
    server = _fake_server_spec()
    initialize_result = McpInitializeResult(
        protocol_version="2025-06-18",
        server_name="fake-mcp",
        server_version="1.0.0",
        instructions="Use fake MCP tools only when explicitly requested.",
    )
    first = (
        McpToolDefinition(
            name="echo",
            description="Echo text.",
            input_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "loud": {"type": "boolean"},
                },
                "required": ["text"],
            },
            annotations={"title": "Echo", "readOnlyHint": True},
        ),
    )
    second = (
        McpToolDefinition(
            name="echo",
            description="Echo text.",
            input_schema={
                "required": ["text"],
                "properties": {
                    "loud": {"type": "boolean"},
                    "text": {"type": "string"},
                },
                "type": "object",
            },
            annotations={"readOnlyHint": True, "title": "Echo"},
        ),
    )

    assert mcp_tool_manifest_hash(
        server=server,
        initialize_result=initialize_result,
        definitions=first,
    ) == mcp_tool_manifest_hash(
        server=server,
        initialize_result=initialize_result,
        definitions=second,
    )


def test_mcp_tool_manifest_hash_is_stable_for_equivalent_tool_order() -> None:
    server = _fake_server_spec()
    initialize_result = McpInitializeResult(protocol_version="2025-06-18")
    first = (
        McpToolDefinition(name="alpha", input_schema={"type": "object"}),
        McpToolDefinition(name="beta", input_schema={"type": "object"}),
    )
    second = (
        McpToolDefinition(name="beta", input_schema={"type": "object"}),
        McpToolDefinition(name="alpha", input_schema={"type": "object"}),
    )

    assert mcp_tool_manifest_hash(
        server=server,
        initialize_result=initialize_result,
        definitions=first,
    ) == mcp_tool_manifest_hash(
        server=server,
        initialize_result=initialize_result,
        definitions=second,
    )


def test_mcp_tool_manifest_hash_changes_when_schema_changes() -> None:
    server = _fake_server_spec()
    initialize_result = McpInitializeResult(protocol_version="2025-06-18")
    original = (
        McpToolDefinition(
            name="echo",
            input_schema={"type": "object", "required": ["text"]},
        ),
    )
    changed = (
        McpToolDefinition(
            name="echo",
            input_schema={"type": "object", "required": ["message"]},
        ),
    )

    assert mcp_tool_manifest_hash(
        server=server,
        initialize_result=initialize_result,
        definitions=original,
    ) != mcp_tool_manifest_hash(
        server=server,
        initialize_result=initialize_result,
        definitions=changed,
    )


def test_mcp_tool_manifest_tools_are_compact_and_stable() -> None:
    server = _fake_server_spec()
    first = (
        McpToolDefinition(
            name="echo",
            description="Echo text.",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
            },
        ),
    )
    second = (
        McpToolDefinition(
            name="echo",
            description="Echo text.",
            input_schema={
                "properties": {"text": {"type": "string"}},
                "type": "object",
            },
        ),
    )

    entries = mcp_tool_manifest_tools(server=server, definitions=first)

    assert entries == mcp_tool_manifest_tools(server=server, definitions=second)
    assert entries[0]["cayu_name"] == "mcp__local-mcp__echo"
    assert entries[0]["mcp_name"] == "echo"
    assert entries[0]["hash"].startswith("sha256:")
    assert "input_schema" not in entries[0]


def test_mcp_tool_manifest_identity_is_stable_across_manifest_revisions() -> None:
    server = _fake_server_spec()
    first = (
        McpToolDefinition(
            name="echo",
            description="Echo text.",
            input_schema={"type": "object", "required": ["text"]},
        ),
    )
    schema_changed = (
        McpToolDefinition(
            name="echo",
            description="Echo changed text.",
            input_schema={"type": "object", "required": ["message"]},
        ),
    )
    tool_changed = (
        McpToolDefinition(
            name="summarize",
            description="Summarize text.",
            input_schema={"type": "object", "required": ["text"]},
        ),
    )

    assert mcp_tool_manifest_identity(server=server, definitions=first) == (
        mcp_tool_manifest_identity(server=server, definitions=schema_changed)
    )
    assert mcp_tool_manifest_identity(server=server, definitions=first) == (
        mcp_tool_manifest_identity(server=server, definitions=tool_changed)
    )
    assert mcp_tool_manifest_identity(server=server) != mcp_tool_manifest_identity(
        server=server.model_copy(update={"connection_id": "second-connection"})
    )
    assert mcp_tool_manifest_identity(server=server) != mcp_tool_manifest_identity(
        server=server.model_copy(update={"connection_id": server.name})
    )
    explicit = server.model_copy(update={"connection_id": "tenant-a/orders"})
    assert mcp_tool_manifest_identity(server=explicit) == mcp_tool_manifest_identity(
        server=explicit.model_copy(update={"name": "renamed-display-alias"})
    )


def test_mcp_manifest_results_reject_mismatched_baseline_keys() -> None:
    event = Event(
        type=EventType.MCP_MANIFEST_CHECKED,
        session_id="mcp_manifest_result_key",
    )
    baseline = McpManifestBaseline(
        history_key="sha256:" + "1" * 64,
        generation=1,
        manifest_identity="sha256:" + "2" * 64,
        manifest_hash=_mcp_authoritative_manifest_hash(
            source_manifest_hash="sha256:" + "5" * 64,
            server_hash="sha256:" + "4" * 64,
            tools=(),
            exposed_tools=(),
        ),
        source_manifest_hash="sha256:" + "5" * 64,
        server_hash="sha256:" + "4" * 64,
        exposed_tools=(),
        accepted_session_ref=_mcp_manifest_session_ref(event.session_id),
        accepted_event_id=event.id,
        accepted_at=event.timestamp,
    )

    with pytest.raises(ValueError, match="load-result key"):
        McpManifestBaselineLoadResult(
            baselines={"sha256:" + "5" * 64: baseline},
        )
    with pytest.raises(ValueError, match="publication-result key"):
        McpManifestPublicationResult(
            published=True,
            baselines={"sha256:" + "5" * 64: baseline},
        )
    unsafe_baseline = baseline.model_dump()
    unsafe_baseline["tools"] = (
        {
            "cayu_name": "mcp__secret-server__secret-tool",
            "mcp_name": "secret-tool",
            "hash": "sha256:" + "6" * 64,
        },
    )
    with pytest.raises(ValueError, match="only tool_id and contract_hash"):
        McpManifestBaseline.model_validate(unsafe_baseline)

    invalid_hash = baseline.model_dump()
    invalid_hash["manifest_identity"] = "not-a-hash"
    with pytest.raises(ValueError, match="SHA-256"):
        McpManifestBaseline.model_validate(invalid_hash)

    duplicate_tools = baseline.model_dump()
    duplicate_tools["tools"] = (
        {
            "tool_id": "sha256:" + "6" * 64,
            "contract_hash": "sha256:" + "7" * 64,
        },
        {
            "tool_id": "sha256:" + "6" * 64,
            "contract_hash": "sha256:" + "8" * 64,
        },
    )
    with pytest.raises(ValueError, match="duplicate tool_id"):
        McpManifestBaseline.model_validate(duplicate_tools)

    object.__setattr__(baseline, "manifest_identity", "mutated-after-validation")
    with pytest.raises(ValueError, match="SHA-256"):
        McpManifestBaselineLoadResult(
            baselines={baseline.history_key: baseline},
        )


@pytest.mark.parametrize(
    "invalid_identifier",
    [
        "sha256:+" + "0" * 63,
        "sha256:-" + "0" * 63,
        "sha256:0_" + "0" * 62,
        "sha256:" + chr(0x0660) * 64,
        "sha256:" + "A" * 64,
        "sha256:" + "0" * 63 + " ",
    ],
)
def test_mcp_manifest_baseline_rejects_noncanonical_sha256_identifiers(
    invalid_identifier: str,
) -> None:
    event = Event(
        type=EventType.MCP_MANIFEST_CHECKED,
        session_id="mcp_manifest_noncanonical_hash",
    )
    baseline = {
        "history_key": "sha256:" + "1" * 64,
        "generation": 1,
        "manifest_identity": "sha256:" + "2" * 64,
        "manifest_hash": invalid_identifier,
        "source_manifest_hash": "sha256:" + "4" * 64,
        "server_hash": "sha256:" + "3" * 64,
        "tools": (),
        "exposed_tools": (),
        "accepted_session_ref": _mcp_manifest_session_ref(event.session_id),
        "accepted_event_id": event.id,
        "accepted_at": event.timestamp,
    }

    with pytest.raises(ValueError, match="SHA-256|whitespace"):
        McpManifestBaseline.model_validate(baseline)


@pytest.mark.parametrize("corruption", ["unbounded_identity", "wrong_identity"])
def test_runtime_fails_closed_on_invalid_loaded_manifest_baseline(corruption: str) -> None:
    secret = "sensitive-corrupt-baseline"

    class InvalidBaselineStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def load_mcp_manifest_baselines(self, history_keys):
            event = Event(
                type=EventType.MCP_MANIFEST_CHECKED,
                session_id="mcp_manifest_invalid_baseline_source",
            )
            manifest_identity = (
                secret + "x" * 10_000
                if corruption == "unbounded_identity"
                else "sha256:" + "f" * 64
            )
            baseline = McpManifestBaseline.model_construct(
                history_key=history_keys[0],
                generation=1,
                manifest_identity=manifest_identity,
                manifest_hash="sha256:" + "1" * 64,
                source_manifest_hash="sha256:" + "3" * 64,
                server_hash="sha256:" + "2" * 64,
                tools=(),
                exposed_tools=(),
                accepted_session_ref=_mcp_manifest_session_ref(event.session_id),
                accepted_event_id=event.id,
                accepted_at=event.timestamp,
            )
            return McpManifestBaselineLoadResult.model_construct(
                baselines={history_keys[0]: baseline},
            )

    async def run():
        store = InvalidBaselineStore()
        provider = FakeProvider([[ModelStreamEvent.completed({})]])
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=_fake_toolset().tools,
        )
        events = await _collect_events(
            app.run(
                RunRequest(
                    session_id=f"mcp_manifest_invalid_{corruption}",
                    agent_name="assistant",
                    messages=[Message.text("user", "hello")],
                )
            )
        )
        return events, provider.requests

    events, requests = asyncio.run(run())

    assert requests == []
    blocked = [event for event in events if event.type == EventType.MCP_MANIFEST_BLOCKED]
    assert len(blocked) == 1
    assert blocked[0].payload["reason"] == "authoritative_baseline_invalid"
    assert secret not in json.dumps([event.model_dump(mode="json") for event in events])


@pytest.mark.parametrize(
    "corruption",
    [
        "manifest_hash",
        "source_manifest_hash",
        "server_hash",
        "tools",
        "exposed_tools",
    ],
)
def test_runtime_recovers_after_malformed_authoritative_baseline_is_removed(
    corruption: str,
) -> None:
    malformed_hash = "sha256:+" + "0" * 63
    alternate_hash = "sha256:" + "f" * 64

    class ToggleMalformedBaselineStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1
        return_malformed_baseline = False

        async def load_stored_baselines(self, history_keys):
            return await super().load_mcp_manifest_baselines(history_keys)

        async def load_mcp_manifest_baselines(self, history_keys):
            loaded = await self.load_stored_baselines(history_keys)
            if not self.return_malformed_baseline:
                return loaded
            baseline = loaded.baselines[history_keys[0]]
            malformed = baseline.model_dump(mode="python")
            if corruption == "manifest_hash":
                malformed["manifest_hash"] = malformed_hash
            elif corruption in {"source_manifest_hash", "server_hash"}:
                malformed[corruption] = alternate_hash
            else:
                malformed[corruption] = (
                    {
                        "tool_id": alternate_hash,
                        "contract_hash": alternate_hash,
                    },
                )
            return McpManifestBaselineLoadResult.model_construct(
                baselines={
                    history_keys[0]: McpManifestBaseline.model_construct(**malformed),
                },
            )

    async def run():
        store = ToggleMalformedBaselineStore()
        toolset = _fake_toolset()

        async def execute(session_id: str):
            provider = FakeProvider([[ModelStreamEvent.completed({})]])
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=toolset.tools,
            )
            events = await _collect_events(
                app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "hello")],
                    )
                )
            )
            return events, provider.requests

        first_events, first_requests = await execute("mcp_manifest_valid_before_corruption")
        first_checked = next(
            event for event in first_events if event.type == EventType.MCP_MANIFEST_CHECKED
        )
        history_key = first_checked.payload["history_key"]
        accepted_before = (await store.load_stored_baselines((history_key,))).baselines[history_key]

        store.return_malformed_baseline = True
        malformed_events, malformed_requests = await execute(
            f"mcp_manifest_malformed_evidence_{corruption}"
        )
        accepted_during = (await store.load_stored_baselines((history_key,))).baselines[history_key]

        store.return_malformed_baseline = False
        recovered_events, recovered_requests = await execute(
            f"mcp_manifest_valid_after_{corruption}_corruption"
        )
        accepted_after = (await store.load_stored_baselines((history_key,))).baselines[history_key]
        return (
            first_events,
            first_requests,
            malformed_events,
            malformed_requests,
            recovered_events,
            recovered_requests,
            accepted_before,
            accepted_during,
            accepted_after,
        )

    (
        first_events,
        first_requests,
        malformed_events,
        malformed_requests,
        recovered_events,
        recovered_requests,
        accepted_before,
        accepted_during,
        accepted_after,
    ) = asyncio.run(run())

    assert len(first_requests) == 1
    assert [
        event.payload["status"]
        for event in first_events
        if event.type == EventType.MCP_MANIFEST_CHECKED
    ] == ["first_seen"]
    assert malformed_requests == []
    malformed_blocked = [
        event for event in malformed_events if event.type == EventType.MCP_MANIFEST_BLOCKED
    ]
    assert len(malformed_blocked) == 1
    assert malformed_blocked[0].payload["reason"] == "authoritative_baseline_invalid"
    assert malformed_hash not in json.dumps(
        [event.model_dump(mode="json") for event in malformed_events]
    )
    assert accepted_during == accepted_before
    assert len(recovered_requests) == 1
    assert [
        event.payload["status"]
        for event in recovered_events
        if event.type == EventType.MCP_MANIFEST_CHECKED
    ] == ["unchanged"]
    assert accepted_after == accepted_before


def test_registration_fails_closed_for_oversized_catalogue() -> None:
    provider = FakeProvider([[ModelStreamEvent.completed({})]])
    toolset = _fake_toolset(
        definitions=_fake_tool_definitions(*(f"tool_{index:05d}" for index in range(10_001)))
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)

    with pytest.raises(ValueError, match="descriptors cannot contain more than 10000 items"):
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=toolset.tools,
        )

    assert provider.requests == []
    assert "assistant" not in app._agents


def test_runtime_requires_explicit_identity_then_tracks_display_renames() -> None:
    async def run():
        store = InMemorySessionStore()
        missing_identity_toolset = _fake_toolset(connection_id=None)
        missing_identity_toolset.server.connection_id = "added-after-manifest-snapshot"
        await _run_mcp_manifest_session(
            store=store,
            session_id="mcp_manifest_missing_identity",
            toolset=missing_identity_toolset,
        )
        explicit_identity_toolset = _fake_toolset(
            connection_id="local-mcp",
            server_name="renamed-mcp",
        )
        explicit_identity_toolset.server.connection_id = "changed-after-manifest-snapshot"
        await _run_mcp_manifest_session(
            store=store,
            session_id="mcp_manifest_explicit_identity",
            toolset=explicit_identity_toolset,
        )
        await _run_mcp_manifest_session(
            store=store,
            session_id="mcp_manifest_explicit_renamed",
            toolset=_fake_toolset(
                connection_id="local-mcp",
                server_name="renamed-again-mcp",
            ),
        )
        return (
            await store.query_events(
                EventQuery(event_type=EventType.MCP_MANIFEST_CHECKED, limit=10)
            ),
            await store.query_events(
                EventQuery(event_type=EventType.MCP_MANIFEST_BLOCKED, limit=10)
            ),
        )

    records, blocked = asyncio.run(run())

    assert [record.event.payload["status"] for record in records] == [
        "first_seen",
        "changed",
    ]
    assert records[1].event.payload["history_key"] == records[0].event.payload["history_key"]
    assert records[1].event.payload["previous"]["event_id"] == records[0].event.id
    assert len(blocked) == 1
    assert blocked[0].event.payload["reason"] == "connection_identity_required"


def test_runtime_namespaces_same_name_connections_by_explicit_identity() -> None:
    async def run():
        store = InMemorySessionStore()
        for tenant in ("tenant-a/orders", "tenant-b/orders"):
            await _run_mcp_manifest_session(
                store=store,
                session_id=f"mcp_manifest_{tenant.split('/')[0]}",
                toolset=_fake_toolset(
                    connection_id=tenant,
                    server_name="shared-display-name",
                ),
            )
        return await store.query_events(
            EventQuery(event_type=EventType.MCP_MANIFEST_CHECKED, limit=10)
        )

    records = asyncio.run(run())

    assert [record.event.payload["status"] for record in records] == [
        "first_seen",
        "first_seen",
    ]
    assert records[0].event.payload["history_key"] != records[1].event.payload["history_key"]


def test_runtime_emits_first_seen_mcp_manifest_event() -> None:
    async def run():
        store = InMemorySessionStore()
        toolset = _fake_toolset()
        provider = FakeProvider(
            [[ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed({})]]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=toolset.tools,
        )
        events = await _collect_events(
            app.run(
                RunRequest(
                    session_id="mcp_manifest_first_seen",
                    agent_name="assistant",
                    messages=[Message.text("user", "hello")],
                )
            )
        )
        records = await store.query_events(
            EventQuery(event_type=EventType.MCP_MANIFEST_CHECKED, limit=10)
        )
        return events, records, toolset

    events, records, toolset = asyncio.run(run())

    manifest_events = [event for event in events if event.type == EventType.MCP_MANIFEST_CHECKED]
    assert len(manifest_events) == 1
    payload = manifest_events[0].payload
    assert payload["manifest_identity"] == toolset.manifest_identity
    assert payload["history_key"].startswith("sha256:")
    assert payload["source_manifest_hash"] == toolset.manifest_hash
    assert payload["manifest_hash"] != payload["source_manifest_hash"]
    assert payload["server_hash"] == toolset.manifest_server_hash
    assert payload["status"] == "first_seen"
    assert payload["outcome"] == "accepted"
    assert payload["previous"] is None
    assert payload["diff"]["server_changed"] is False
    assert payload["diff"]["added_tools"] == []
    assert payload["diff"]["removed_tools"] == []
    assert payload["diff"]["changed_tools"] == []
    assert payload["diff"]["truncated"] is False
    assert "tools" not in payload
    assert "server" not in payload
    assert len(records) == 1


def test_runtime_admits_sanitized_manifest_with_private_binding_hash() -> None:
    secret = "mcp-private-binding-canary"
    definitions = (
        McpToolDefinition(
            name="echo",
            description=f"Echo using {secret}.",
            input_schema={"type": "object"},
        ),
    )
    session = FakeMcpSession(definitions=definitions)
    session._secret_redactor = SecretRedactor(secret)
    toolset = McpToolset(
        server=McpServerSpec(
            name="private-mcp",
            connection_id="private-mcp-binding",
            command=["unused"],
        ),
        session=session,
        definitions=definitions,
    )

    async def run():
        store = InMemorySessionStore()
        await _run_mcp_manifest_session(
            store=store,
            session_id="mcp_private_binding",
            toolset=toolset,
        )
        return await store.query_events(
            EventQuery(event_type=EventType.MCP_MANIFEST_CHECKED, limit=10)
        )

    records = asyncio.run(run())

    assert len(records) == 1
    assert records[0].event.payload["outcome"] == "accepted"
    assert secret not in json.dumps(records[0].event.model_dump(mode="json"))


def test_runtime_marks_mcp_manifest_unchanged_across_sessions() -> None:
    async def run():
        store = InMemorySessionStore()
        await _run_mcp_manifest_session(
            store=store,
            session_id="mcp_manifest_unchanged_1",
            toolset=_fake_toolset(),
        )
        await _run_mcp_manifest_session(
            store=store,
            session_id="mcp_manifest_unchanged_2",
            toolset=_fake_toolset(),
        )
        return await store.query_events(
            EventQuery(event_type=EventType.MCP_MANIFEST_CHECKED, limit=10)
        )

    records = asyncio.run(run())

    assert [record.event.payload["status"] for record in records] == [
        "first_seen",
        "unchanged",
    ]
    assert records[1].event.payload["previous"]["event_id"] == records[0].event.id
    assert records[1].event.payload["previous"]["session_ref"].startswith("sha256:")
    assert "session_id" not in records[1].event.payload["previous"]
    assert records[1].event.payload["diff"]["server_changed"] is False
    assert records[1].event.payload["diff"]["added_tools"] == []
    assert records[1].event.payload["diff"]["removed_tools"] == []
    assert records[1].event.payload["diff"]["changed_tools"] == []


def test_runtime_marks_mcp_manifest_changed_across_sessions() -> None:
    async def run():
        store = InMemorySessionStore()
        await _run_mcp_manifest_session(
            store=store,
            session_id="mcp_manifest_changed_1",
            toolset=_fake_toolset(description="Echo text."),
        )
        await _run_mcp_manifest_session(
            store=store,
            session_id="mcp_manifest_changed_2",
            toolset=_fake_toolset(description="Echo changed text."),
        )
        return await store.query_events(
            EventQuery(event_type=EventType.MCP_MANIFEST_CHECKED, limit=10)
        )

    records = asyncio.run(run())

    assert [record.event.payload["status"] for record in records] == [
        "first_seen",
        "changed",
    ]
    changed_payload = records[1].event.payload
    assert changed_payload["previous"]["manifest_hash"] == records[0].event.payload["manifest_hash"]
    assert changed_payload["diff"]["server_changed"] is False
    assert changed_payload["diff"]["added_tools"] == []
    assert changed_payload["diff"]["removed_tools"] == []
    assert changed_payload["diff"]["changed_tools"] == [_opaque_mcp_tool_id("local-mcp", "echo")]


@pytest.mark.parametrize(
    ("change", "expected_changes"),
    [
        ("added", ["tools_added"]),
        ("removed", ["tools_removed"]),
        ("renamed", ["tools_added", "tools_removed"]),
        ("schema", ["tools_changed"]),
        ("annotations", ["tools_changed"]),
        ("description", ["tools_changed"]),
        ("instructions", ["server_changed", "tools_changed"]),
    ],
)
def test_runtime_policy_blocks_every_mcp_manifest_change_class(
    change: str,
    expected_changes: list[str],
) -> None:
    async def run():
        store = InMemorySessionStore()
        base_definitions = _fake_tool_definitions("echo")
        current_definitions = base_definitions
        base_initialize = McpInitializeResult(
            protocol_version="2025-06-18",
            instructions="Original instructions.",
        )
        current_initialize = base_initialize
        if change == "added":
            current_definitions = _fake_tool_definitions("echo", "new")
        elif change == "removed":
            base_definitions = _fake_tool_definitions("echo", "safety")
        elif change == "renamed":
            base_definitions = _fake_tool_definitions("old")
            current_definitions = _fake_tool_definitions("new")
        elif change == "schema":
            current_definitions = (
                McpToolDefinition(
                    name="echo",
                    description="Echo text.",
                    input_schema={"type": "object", "required": ["value"]},
                ),
            )
        elif change == "annotations":
            current_definitions = (
                McpToolDefinition(
                    name="echo",
                    description="Echo text.",
                    input_schema={
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                    },
                    annotations={"readOnlyHint": True},
                ),
            )
        elif change == "description":
            current_definitions = _fake_tool_definitions(
                "echo",
                description="Changed description.",
            )
        elif change == "instructions":
            current_initialize = McpInitializeResult(
                protocol_version="2025-06-18",
                instructions="Changed instructions.",
            )

        await _run_mcp_manifest_session(
            store=store,
            session_id=f"mcp_manifest_change_{change}_1",
            toolset=_fake_toolset(
                definitions=base_definitions,
                initialize_result=base_initialize,
            ),
        )
        provider = FakeProvider([[ModelStreamEvent.completed({})]])
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            mcp_manifest_policy=McpManifestPolicy(on_changed=McpManifestPolicyAction.BLOCK),
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=_fake_toolset(
                definitions=current_definitions,
                initialize_result=current_initialize,
            ).tools,
        )
        events = await _collect_events(
            app.run(
                RunRequest(
                    session_id=f"mcp_manifest_change_{change}_2",
                    agent_name="assistant",
                    messages=[Message.text("user", "hello")],
                )
            )
        )
        return events, provider.requests

    events, requests = asyncio.run(run())

    assert requests == []
    blocked = [event for event in events if event.type == EventType.MCP_MANIFEST_BLOCKED]
    assert len(blocked) == 1
    assert blocked[0].payload["policy"]["matched_changes"] == expected_changes


def test_mcp_manifest_events_and_baselines_bound_and_redact_inputs(tmp_path: Path) -> None:
    async def run():
        database = tmp_path / "mcp-redacted-baseline.sqlite"
        store = SQLiteSessionStore(database)
        secret = "sensitive-tenant-token"
        long_tool_name = f"tool-{secret}-" + "x" * 10_000
        long_session_id = "mcp-manifest-session-" + "y" * 2_000
        await _run_mcp_manifest_session(
            store=store,
            session_id=long_session_id,
            toolset=_fake_toolset(
                definitions=(
                    McpToolDefinition(
                        name=long_tool_name,
                        description=f"description-{secret}",
                        input_schema={
                            "type": "object",
                            "properties": {f"schema-{secret}": {"type": "string"}},
                        },
                        annotations={f"annotation-{secret}": True},
                    ),
                ),
                initialize_result=McpInitializeResult(
                    protocol_version="2025-06-18",
                    instructions=f"instructions-{secret}",
                ),
                connection_id=f"tenant/{secret}",
                server_name=f"server-{secret}",
            ),
        )
        initial_events = await store.query_events(
            EventQuery(event_type=EventType.MCP_MANIFEST_CHECKED, limit=10)
        )
        initial_history_key = initial_events[0].event.payload["history_key"]
        initial_baseline = (
            await store.load_mcp_manifest_baselines((initial_history_key,))
        ).baselines[initial_history_key]
        connection = sqlite3.connect(database)
        try:
            initial_baseline_json = connection.execute(
                "SELECT baseline_json FROM cayu_mcp_manifest_baselines WHERE history_key = ?",
                (initial_history_key,),
            ).fetchone()[0]
        finally:
            connection.close()
        changed = _fake_toolset(
            definitions=_fake_tool_definitions(*(f"tool_{index}" for index in range(150))),
            initialize_result=McpInitializeResult(
                protocol_version="2025-06-18",
                instructions=f"changed-instructions-{secret}",
            ),
            connection_id=f"tenant/{secret}",
            server_name=f"server-{secret}",
        )
        provider = FakeProvider([[ModelStreamEvent.completed({})]])
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=changed.tools,
        )
        events = await _collect_events(
            app.run(
                RunRequest(
                    session_id="mcp_manifest_redacted_2",
                    agent_name="assistant",
                    messages=[Message.text("user", "hello")],
                )
            )
        )
        await store.close()
        connection = sqlite3.connect(database)
        try:
            baseline_rows = [
                row[0]
                for row in connection.execute(
                    "SELECT baseline_json FROM cayu_mcp_manifest_baselines"
                ).fetchall()
            ]
        finally:
            connection.close()
        return (
            events,
            initial_baseline,
            long_tool_name,
            long_session_id,
            [initial_baseline_json, *baseline_rows],
        )

    events, initial_baseline, long_tool_name, long_session_id, baseline_rows = asyncio.run(run())

    checked = [event for event in events if event.type == EventType.MCP_MANIFEST_CHECKED]
    assert len(checked) == 1
    payload = checked[0].payload
    encoded = json.dumps(payload, sort_keys=True)
    encoded_baseline = initial_baseline.model_dump_json()
    assert "sensitive-tenant-token" not in encoded
    assert "sensitive-tenant-token" not in encoded_baseline
    assert long_tool_name not in encoded_baseline
    assert long_session_id not in encoded
    assert long_session_id not in encoded_baseline
    assert all("sensitive-tenant-token" not in row for row in baseline_rows)
    assert all(long_tool_name not in row for row in baseline_rows)
    assert all(long_session_id not in row for row in baseline_rows)
    assert len(initial_baseline.accepted_session_ref) == 71
    assert initial_baseline.tools == (
        {
            "tool_id": _opaque_mcp_tool_id(
                "server-sensitive-tenant-token",
                long_tool_name,
            ),
            "contract_hash": initial_baseline.tools[0]["contract_hash"],
        },
    )
    assert initial_baseline.exposed_tools[0]["tool_id"] == initial_baseline.tools[0]["tool_id"]
    assert len(encoded_baseline) < 1_500
    assert payload["diff"]["added_tools_count"] == 150
    assert len(payload["diff"]["added_tools"]) == 100
    assert payload["diff"]["truncated"] is True
    assert payload["change_classes"] == [
        "server_changed",
        "tools_added",
        "tools_removed",
    ]


def test_runtime_blocks_changed_mcp_manifest_before_model_request() -> None:
    async def run():
        store = InMemorySessionStore()
        await _run_mcp_manifest_session(
            store=store,
            session_id="mcp_manifest_blocked_1",
            toolset=_fake_toolset(description="Echo text."),
        )
        provider = FakeProvider(
            [[ModelStreamEvent.text_delta("should-not-run"), ModelStreamEvent.completed({})]]
        )
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            mcp_manifest_policy=McpManifestPolicy(on_changed=McpManifestPolicyAction.BLOCK),
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=_fake_toolset(description="Echo changed text.").tools,
        )
        events = await _collect_events(
            app.run(
                RunRequest(
                    session_id="mcp_manifest_blocked_2",
                    agent_name="assistant",
                    messages=[Message.text("user", "hello")],
                )
            )
        )
        return events, provider.requests

    events, requests = asyncio.run(run())

    assert requests == []
    checked = [event for event in events if event.type == EventType.MCP_MANIFEST_CHECKED]
    blocked = [event for event in events if event.type == EventType.MCP_MANIFEST_BLOCKED]
    failed = [event for event in events if event.type == EventType.SESSION_FAILED]
    assert [event.type for event in events if event.type == EventType.MODEL_STARTED] == []
    assert checked == []
    assert len(blocked) == 1
    assert len(failed) == 1
    assert blocked[0].payload["status"] == "changed"
    assert blocked[0].payload["policy"]["action"] == "block"
    assert blocked[0].payload["policy"]["matched_changes"] == ["tools_changed"]
    assert failed[0].payload["error_type"] == "McpManifestPolicyError"


def test_runtime_alerts_changed_mcp_manifest_without_blocking() -> None:
    async def run():
        store = InMemorySessionStore()
        await _run_mcp_manifest_session(
            store=store,
            session_id="mcp_manifest_alert_1",
            toolset=_fake_toolset(description="Echo text."),
        )
        provider = FakeProvider(
            [[ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed({})]]
        )
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            mcp_manifest_policy=McpManifestPolicy(on_changed=McpManifestPolicyAction.ALERT),
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=_fake_toolset(description="Echo changed text.").tools,
        )
        events = await _collect_events(
            app.run(
                RunRequest(
                    session_id="mcp_manifest_alert_2",
                    agent_name="assistant",
                    messages=[Message.text("user", "hello")],
                )
            )
        )
        return events, provider.requests

    events, requests = asyncio.run(run())

    assert len(requests) == 1
    checked = [event for event in events if event.type == EventType.MCP_MANIFEST_CHECKED]
    assert len(checked) == 1
    assert checked[0].payload["status"] == "changed"
    assert checked[0].payload["policy"]["action"] == "alert"
    assert [event for event in events if event.type == EventType.MCP_MANIFEST_BLOCKED] == []
    assert [event.type for event in events if event.type == EventType.SESSION_COMPLETED] == [
        EventType.SESSION_COMPLETED
    ]


def test_runtime_blocked_mcp_manifest_does_not_become_baseline() -> None:
    async def run():
        store = InMemorySessionStore()
        await _run_mcp_manifest_session(
            store=store,
            session_id="mcp_manifest_block_baseline_1",
            toolset=_fake_toolset(description="Echo text."),
        )
        for session_id in [
            "mcp_manifest_block_baseline_2",
            "mcp_manifest_block_baseline_3",
        ]:
            provider = FakeProvider(
                [[ModelStreamEvent.text_delta("should-not-run"), ModelStreamEvent.completed({})]]
            )
            app = CayuApp(
                session_store=store,
                enable_logging=False,
                mcp_manifest_policy=McpManifestPolicy(on_changed=McpManifestPolicyAction.BLOCK),
            )
            app.register_provider(provider, default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=_fake_toolset(description="Echo changed text.").tools,
            )
            await _collect_events(
                app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "hello")],
                    )
                )
            )
        checked_records = await store.query_events(
            EventQuery(event_type=EventType.MCP_MANIFEST_CHECKED, limit=10)
        )
        blocked_records = await store.query_events(
            EventQuery(event_type=EventType.MCP_MANIFEST_BLOCKED, limit=10)
        )
        return checked_records, blocked_records

    checked_records, blocked_records = asyncio.run(run())

    assert [record.event.payload["status"] for record in checked_records] == ["first_seen"]
    assert [record.event.payload["status"] for record in blocked_records] == [
        "changed",
        "changed",
    ]


def test_runtime_blocked_mcp_manifest_does_not_partially_accept_other_toolsets() -> None:
    async def run():
        store = InMemorySessionStore()
        echo_toolset = _fake_toolset(
            definitions=_fake_tool_definitions("echo"),
            connection_id="echo",
        )
        summarize_toolset = _fake_toolset(
            definitions=_fake_tool_definitions("summarize", description="Summarize text."),
            connection_id="summarize",
        )
        provider = FakeProvider(
            [[ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed({})]]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=summarize_toolset.tools,
        )
        await _collect_events(
            app.run(
                RunRequest(
                    session_id="mcp_manifest_partial_accept_1",
                    agent_name="assistant",
                    messages=[Message.text("user", "hello")],
                )
            )
        )

        changed_summarize_toolset = _fake_toolset(
            definitions=_fake_tool_definitions(
                "summarize",
                description="Summarize changed text.",
            ),
            connection_id="summarize",
        )
        blocked_provider = FakeProvider(
            [[ModelStreamEvent.text_delta("should-not-run"), ModelStreamEvent.completed({})]]
        )
        blocked_app = CayuApp(
            session_store=store,
            enable_logging=False,
            mcp_manifest_policy=McpManifestPolicy(on_changed=McpManifestPolicyAction.BLOCK),
        )
        blocked_app.register_provider(blocked_provider, default=True)
        blocked_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[*echo_toolset.tools, *changed_summarize_toolset.tools],
        )
        await _collect_events(
            blocked_app.run(
                RunRequest(
                    session_id="mcp_manifest_partial_accept_2",
                    agent_name="assistant",
                    messages=[Message.text("user", "hello")],
                )
            )
        )
        checked_records = await store.query_events(
            EventQuery(event_type=EventType.MCP_MANIFEST_CHECKED, limit=10)
        )
        blocked_records = await store.query_events(
            EventQuery(event_type=EventType.MCP_MANIFEST_BLOCKED, limit=10)
        )
        first_seen_sibling = next(
            record
            for record in checked_records
            if record.event.session_id == "mcp_manifest_partial_accept_2"
        )
        sibling_history_key = first_seen_sibling.event.payload["history_key"]
        sibling_baselines = await store.load_mcp_manifest_baselines((sibling_history_key,))
        return (
            checked_records,
            blocked_records,
            blocked_provider.requests,
            sibling_baselines,
        )

    checked_records, blocked_records, requests, sibling_baselines = asyncio.run(run())

    assert requests == []
    assert len(checked_records) == 2
    assert [record.event.session_id for record in checked_records] == [
        "mcp_manifest_partial_accept_1",
        "mcp_manifest_partial_accept_2",
    ]
    assert checked_records[-1].event.payload["status"] == "first_seen"
    assert checked_records[-1].event.payload["outcome"] == "batch_blocked"
    assert sibling_baselines.baselines == {}
    assert len(blocked_records) == 1
    assert blocked_records[0].event.payload["diff"]["changed_tools"] == [
        _opaque_mcp_tool_id("local-mcp", "summarize")
    ]


def test_runtime_mcp_manifest_policy_specific_change_overrides_generic_change() -> None:
    async def run():
        store = InMemorySessionStore()
        await _run_mcp_manifest_session(
            store=store,
            session_id="mcp_manifest_added_block_1",
            toolset=_fake_toolset(definitions=_fake_tool_definitions("echo")),
        )
        provider = FakeProvider(
            [[ModelStreamEvent.text_delta("should-not-run"), ModelStreamEvent.completed({})]]
        )
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            mcp_manifest_policy=McpManifestPolicy(
                on_changed=McpManifestPolicyAction.ALLOW,
                on_tools_added=McpManifestPolicyAction.BLOCK,
            ),
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=_fake_toolset(definitions=_fake_tool_definitions("echo", "summarize")).tools,
        )
        events = await _collect_events(
            app.run(
                RunRequest(
                    session_id="mcp_manifest_added_block_2",
                    agent_name="assistant",
                    messages=[Message.text("user", "hello")],
                )
            )
        )
        return events, provider.requests

    events, requests = asyncio.run(run())

    assert requests == []
    checked = [event for event in events if event.type == EventType.MCP_MANIFEST_CHECKED]
    blocked = [event for event in events if event.type == EventType.MCP_MANIFEST_BLOCKED]
    assert checked == []
    assert len(blocked) == 1
    assert blocked[0].payload["policy"]["action"] == "block"
    assert blocked[0].payload["policy"]["matched_changes"] == ["tools_added"]
    assert blocked[0].payload["diff"]["added_tools"] == [
        _opaque_mcp_tool_id("local-mcp", "summarize")
    ]


def test_runtime_mcp_manifest_policy_specific_override_can_be_less_strict() -> None:
    async def run():
        store = InMemorySessionStore()
        await _run_mcp_manifest_session(
            store=store,
            session_id="mcp_manifest_removed_alert_1",
            toolset=_fake_toolset(definitions=_fake_tool_definitions("echo", "summarize")),
        )
        provider = FakeProvider(
            [[ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed({})]]
        )
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            mcp_manifest_policy=McpManifestPolicy(
                on_changed=McpManifestPolicyAction.BLOCK,
                on_tools_removed=McpManifestPolicyAction.ALERT,
            ),
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=_fake_toolset(definitions=_fake_tool_definitions("echo")).tools,
        )
        events = await _collect_events(
            app.run(
                RunRequest(
                    session_id="mcp_manifest_removed_alert_2",
                    agent_name="assistant",
                    messages=[Message.text("user", "hello")],
                )
            )
        )
        return events, provider.requests

    events, requests = asyncio.run(run())

    assert len(requests) == 1
    checked = [event for event in events if event.type == EventType.MCP_MANIFEST_CHECKED]
    assert len(checked) == 1
    assert checked[0].payload["policy"]["action"] == "alert"
    assert checked[0].payload["policy"]["matched_changes"] == ["tools_removed"]
    assert checked[0].payload["diff"]["removed_tools"] == [
        _opaque_mcp_tool_id("local-mcp", "summarize")
    ]
    assert [event for event in events if event.type == EventType.MCP_MANIFEST_BLOCKED] == []


def test_runtime_marks_mcp_server_metadata_changed_across_sessions() -> None:
    async def run():
        store = InMemorySessionStore()
        await _run_mcp_manifest_session(
            store=store,
            session_id="mcp_manifest_server_changed_1",
            toolset=_fake_toolset(
                initialize_result=McpInitializeResult(
                    protocol_version="2025-06-18",
                    instructions="Use carefully.",
                )
            ),
        )
        await _run_mcp_manifest_session(
            store=store,
            session_id="mcp_manifest_server_changed_2",
            toolset=_fake_toolset(
                initialize_result=McpInitializeResult(
                    protocol_version="2025-06-18",
                    instructions="Use only after approval.",
                )
            ),
        )
        return await store.query_events(
            EventQuery(event_type=EventType.MCP_MANIFEST_CHECKED, limit=10)
        )

    records = asyncio.run(run())

    assert [record.event.payload["status"] for record in records] == [
        "first_seen",
        "changed",
    ]
    assert records[1].event.payload["diff"]["server_changed"] is True
    assert records[1].event.payload["diff"]["added_tools"] == []
    assert records[1].event.payload["diff"]["removed_tools"] == []
    assert records[1].event.payload["diff"]["changed_tools"] == [
        _opaque_mcp_tool_id("local-mcp", "echo")
    ]


def test_runtime_marks_mcp_added_and_removed_tools_across_sessions() -> None:
    async def run():
        store = InMemorySessionStore()
        await _run_mcp_manifest_session(
            store=store,
            session_id="mcp_manifest_tools_changed_1",
            toolset=_fake_toolset(definitions=_fake_tool_definitions("echo", "old")),
        )
        await _run_mcp_manifest_session(
            store=store,
            session_id="mcp_manifest_tools_changed_2",
            toolset=_fake_toolset(definitions=_fake_tool_definitions("echo", "new")),
        )
        return await store.query_events(
            EventQuery(event_type=EventType.MCP_MANIFEST_CHECKED, limit=10)
        )

    records = asyncio.run(run())

    assert [record.event.payload["status"] for record in records] == [
        "first_seen",
        "changed",
    ]
    assert records[1].event.payload["diff"]["server_changed"] is False
    assert records[1].event.payload["diff"]["added_tools"] == [
        _opaque_mcp_tool_id("local-mcp", "new")
    ]
    assert records[1].event.payload["diff"]["removed_tools"] == [
        _opaque_mcp_tool_id("local-mcp", "old")
    ]
    assert records[1].event.payload["diff"]["changed_tools"] == []


def test_runtime_audits_distinct_same_name_mcp_toolsets() -> None:
    async def run():
        store = InMemorySessionStore()
        echo_toolset = _fake_toolset(
            definitions=_fake_tool_definitions("echo"),
            connection_id="echo",
        )
        summarize_toolset = _fake_toolset(
            definitions=_fake_tool_definitions("summarize"),
            connection_id="summarize",
        )
        provider = FakeProvider(
            [[ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed({})]]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[*echo_toolset.tools, *summarize_toolset.tools],
        )
        await _collect_events(
            app.run(
                RunRequest(
                    session_id="mcp_manifest_same_server_two_toolsets",
                    agent_name="assistant",
                    messages=[Message.text("user", "hello")],
                )
            )
        )
        return await store.query_events(
            EventQuery(event_type=EventType.MCP_MANIFEST_CHECKED, limit=10)
        )

    records = asyncio.run(run())

    assert len(records) == 2
    assert [record.event.payload["status"] for record in records] == [
        "first_seen",
        "first_seen",
    ]
    assert (
        records[0].event.payload["manifest_identity"]
        != records[1].event.payload["manifest_identity"]
    )


def test_runtime_fails_closed_on_duplicate_mcp_connection_identities() -> None:
    async def run():
        store = InMemorySessionStore()
        provider = FakeProvider([[ModelStreamEvent.completed({})]])
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[
                *_fake_toolset(
                    definitions=_fake_tool_definitions("echo"),
                ).tools,
                *_fake_toolset(
                    definitions=_fake_tool_definitions("summarize"),
                ).tools,
            ],
        )
        events = await _collect_events(
            app.run(
                RunRequest(
                    session_id="mcp_manifest_duplicate_identity",
                    agent_name="assistant",
                    messages=[Message.text("user", "hello")],
                )
            )
        )
        return events, provider.requests

    events, requests = asyncio.run(run())

    assert requests == []
    blocked = [event for event in events if event.type == EventType.MCP_MANIFEST_BLOCKED]
    assert len(blocked) == 2
    assert {event.payload["reason"] for event in blocked} == {"duplicate_connection_identity"}
    failed = [event for event in events if event.type == EventType.SESSION_FAILED]
    assert len(failed) == 1
    assert failed[0].payload["error_type"] == "McpManifestHistoryConflict"


def test_runtime_fails_closed_when_session_store_lacks_manifest_history() -> None:
    class UnsupportedManifestHistoryStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1
        supports_mcp_manifest_history = False

    async def run():
        store = UnsupportedManifestHistoryStore()
        provider = FakeProvider([[ModelStreamEvent.completed({})]])
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=_fake_toolset().tools,
        )
        events = await _collect_events(
            app.run(
                RunRequest(
                    session_id="mcp_manifest_unsupported_store",
                    agent_name="assistant",
                    messages=[Message.text("user", "hello")],
                )
            )
        )
        return events, provider.requests

    events, requests = asyncio.run(run())

    assert requests == []
    blocked = [event for event in events if event.type == EventType.MCP_MANIFEST_BLOCKED]
    assert len(blocked) == 1
    assert blocked[0].payload["status"] == "history_unavailable"
    assert blocked[0].payload["reason"] == "session_store_manifest_history_unsupported"


def test_runtime_persists_complete_rejection_batch_before_first_yield() -> None:
    async def run():
        store = InMemorySessionStore()
        provider = FakeProvider([[ModelStreamEvent.completed({})]])
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[
                *_fake_toolset(
                    definitions=_fake_tool_definitions("echo"),
                    server_name="first",
                ).tools,
                *_fake_toolset(
                    definitions=_fake_tool_definitions("summarize"),
                    server_name="second",
                ).tools,
            ],
        )
        stream = app.run(
            RunRequest(
                session_id="mcp_manifest_rejection_abandoned",
                agent_name="assistant",
                messages=[Message.text("user", "hello")],
            )
        )
        async for event in stream:
            if event.type == EventType.MCP_MANIFEST_BLOCKED:
                break
        await stream.aclose()
        records = await store.query_events(
            EventQuery(event_type=EventType.MCP_MANIFEST_BLOCKED, limit=10)
        )
        return records, provider.requests

    records, requests = asyncio.run(run())

    assert requests == []
    assert len(records) == 2
    assert {record.event.payload["reason"] for record in records} == {
        "duplicate_connection_identity"
    }


def test_runtime_audits_explicit_sibling_when_identity_is_missing() -> None:
    async def run():
        store = InMemorySessionStore()
        provider = FakeProvider([[ModelStreamEvent.completed({})]])
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[
                *_fake_toolset(
                    definitions=_fake_tool_definitions("read"),
                    connection_id=None,
                    server_name="missing-identity",
                ).tools,
                *_fake_toolset(
                    definitions=_fake_tool_definitions("summarize"),
                    connection_id="tenant-a/summarizer",
                    server_name="explicit-identity",
                ).tools,
            ],
        )
        events = await _collect_events(
            app.run(
                RunRequest(
                    session_id="mcp_manifest_missing_identity_batch",
                    agent_name="assistant",
                    messages=[Message.text("user", "hello")],
                )
            )
        )
        checked = next(event for event in events if event.type == EventType.MCP_MANIFEST_CHECKED)
        baselines = await store.load_mcp_manifest_baselines((checked.payload["history_key"],))
        return events, baselines, provider.requests

    events, baselines, requests = asyncio.run(run())

    assert requests == []
    blocked = [event for event in events if event.type == EventType.MCP_MANIFEST_BLOCKED]
    checked = [event for event in events if event.type == EventType.MCP_MANIFEST_CHECKED]
    assert len(blocked) == 1
    assert blocked[0].payload["reason"] == "connection_identity_required"
    assert len(checked) == 1
    assert checked[0].payload["status"] == "not_evaluated"
    assert checked[0].payload["outcome"] == "batch_blocked"
    assert checked[0].payload["reason"] == "sibling_connection_identity_missing"
    assert baselines.baselines == {}


def test_runtime_classifies_internal_manifest_store_cancellation_as_failure() -> None:
    class InternallyCancelledManifestStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def load_mcp_manifest_baselines(self, history_keys):
            del history_keys
            child = asyncio.create_task(asyncio.sleep(60))
            child.cancel("storage child cancelled")
            return await child

    async def run():
        store = InternallyCancelledManifestStore()
        provider = FakeProvider([[ModelStreamEvent.completed({})]])
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=_fake_toolset().tools,
        )
        task = asyncio.create_task(
            _collect_events(
                app.run(
                    RunRequest(
                        session_id="mcp_manifest_internal_cancel",
                        agent_name="assistant",
                        messages=[Message.text("user", "hello")],
                    )
                )
            )
        )
        events = await task
        return events, provider.requests, task

    events, requests, task = asyncio.run(run())

    assert requests == []
    assert not task.cancelled()
    assert task.cancelling() == 0
    failed = [event for event in events if event.type == EventType.SESSION_FAILED]
    assert len(failed) == 1
    assert failed[0].payload["error_type"] == "RuntimeError"
    assert failed[0].payload["error"] == (
        "MCP manifest baseline loading was cancelled without caller cancellation."
    )


def test_runtime_does_not_reclassify_internal_manifest_cancellation_from_history() -> None:
    class InternallyCancelledManifestStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def load_mcp_manifest_baselines(self, history_keys):
            del history_keys
            child = asyncio.create_task(asyncio.sleep(60))
            child.cancel("storage child cancelled")
            return await child

    async def run_with_historical_cancellation():
        current_task = asyncio.current_task()
        assert current_task is not None
        current_task.cancel("historical caller cancellation")
        with pytest.raises(asyncio.CancelledError, match="historical caller cancellation"):
            await asyncio.sleep(0)
        assert current_task.cancelling() == 1

        store = InternallyCancelledManifestStore()
        provider = FakeProvider([[ModelStreamEvent.completed({})]])
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=_fake_toolset().tools,
        )
        events = await _collect_events(
            app.run(
                RunRequest(
                    session_id="mcp_manifest_internal_cancel_after_history",
                    agent_name="assistant",
                    messages=[Message.text("user", "hello")],
                )
            )
        )
        session = await store.load("mcp_manifest_internal_cancel_after_history")
        return events, provider.requests, session, current_task.cancelling()

    async def run():
        task = asyncio.create_task(run_with_historical_cancellation())
        result = await task
        return result, task

    ((events, requests, session, cancelling), task) = asyncio.run(run())

    assert requests == []
    assert session is not None
    assert session.status.value == "failed"
    assert cancelling == 1
    assert task.cancelling() == 1
    assert not task.cancelled()
    failed = [event for event in events if event.type == EventType.SESSION_FAILED]
    assert len(failed) == 1
    assert failed[0].payload["error_type"] == "RuntimeError"
    assert failed[0].payload["error"] == (
        "MCP manifest baseline loading was cancelled without caller cancellation."
    )


def test_runtime_preserves_real_cancellation_during_manifest_load() -> None:
    class BlockingManifestStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self):
            super().__init__()
            self.load_started = asyncio.Event()

        async def load_mcp_manifest_baselines(self, history_keys):
            del history_keys
            self.load_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    async def run():
        store = BlockingManifestStore()
        provider = FakeProvider([[ModelStreamEvent.completed({})]])
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=_fake_toolset().tools,
        )
        task = asyncio.create_task(
            _collect_events(
                app.run(
                    RunRequest(
                        session_id="mcp_manifest_caller_cancel",
                        agent_name="assistant",
                        messages=[Message.text("user", "hello")],
                    )
                )
            )
        )
        await store.load_started.wait()
        task.cancel("caller cancelled")
        cancelling = task.cancelling()
        with pytest.raises(asyncio.CancelledError, match="caller cancelled"):
            await task
        return task, cancelling, provider.requests

    task, cancelling, requests = asyncio.run(run())

    assert cancelling == 1
    assert task.cancelled()
    assert requests == []


def test_runtime_preserves_caller_cancellation_swallowed_during_manifest_load() -> None:
    class SwallowingManifestStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self):
            super().__init__()
            self.load_started = asyncio.Event()

        async def load_mcp_manifest_baselines(self, history_keys):
            self.load_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return await super().load_mcp_manifest_baselines(history_keys)
            raise AssertionError("unreachable")

    async def run():
        store = SwallowingManifestStore()
        provider = FakeProvider([[ModelStreamEvent.completed({})]])
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=_fake_toolset().tools,
        )
        task = asyncio.create_task(
            _collect_events(
                app.run(
                    RunRequest(
                        session_id="mcp_manifest_swallowed_load_cancel",
                        agent_name="assistant",
                        messages=[Message.text("user", "hello")],
                    )
                )
            )
        )
        await store.load_started.wait()
        task.cancel("caller cancelled")
        cancelling = task.cancelling()
        with pytest.raises(asyncio.CancelledError):
            await task
        return task, cancelling, provider.requests

    task, cancelling, requests = asyncio.run(run())

    assert cancelling == 1
    assert task.cancelled()
    assert task.cancelling() == 0
    assert requests == []


def test_runtime_preserves_cancellation_swallowed_after_manifest_commit() -> None:
    class CommitThenSwallowCancellationStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self):
            super().__init__()
            self.pause_next_publication = True
            self.publication_committed = asyncio.Event()

        async def compare_and_publish_mcp_manifest_checks(self, *args, **kwargs):
            result = await super().compare_and_publish_mcp_manifest_checks(*args, **kwargs)
            if result.published and self.pause_next_publication:
                self.pause_next_publication = False
                self.publication_committed.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    return result
                raise AssertionError("unreachable")
            return result

    async def run():
        store = CommitThenSwallowCancellationStore()
        toolset = _fake_toolset()
        cancelled_provider = FakeProvider([[ModelStreamEvent.completed({})]])
        cancelled_app = CayuApp(session_store=store, enable_logging=False)
        cancelled_app.register_provider(cancelled_provider, default=True)
        cancelled_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=toolset.tools,
        )
        cancelled_task = asyncio.create_task(
            _collect_events(
                cancelled_app.run(
                    RunRequest(
                        session_id="mcp_manifest_swallowed_commit_cancel",
                        agent_name="assistant",
                        messages=[Message.text("user", "cancel")],
                    )
                )
            )
        )
        await store.publication_committed.wait()
        cancelled_task.cancel("caller cancelled after commit")
        cancelling = cancelled_task.cancelling()
        with pytest.raises(asyncio.CancelledError):
            await cancelled_task

        committed_events = await store.query_events(
            EventQuery(event_type=EventType.MCP_MANIFEST_CHECKED, limit=10)
        )
        history_key = committed_events[0].event.payload["history_key"]
        committed_baseline = (await store.load_mcp_manifest_baselines((history_key,))).baselines[
            history_key
        ]

        retry_provider = FakeProvider([[ModelStreamEvent.completed({})]])
        retry_app = CayuApp(session_store=store, enable_logging=False)
        retry_app.register_provider(retry_provider, default=True)
        retry_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=toolset.tools,
        )
        retry_events = await _collect_events(
            retry_app.run(
                RunRequest(
                    session_id="mcp_manifest_swallowed_commit_retry",
                    agent_name="assistant",
                    messages=[Message.text("user", "retry")],
                )
            )
        )
        final_baseline = (await store.load_mcp_manifest_baselines((history_key,))).baselines[
            history_key
        ]
        return (
            cancelled_task,
            cancelling,
            cancelled_provider.requests,
            committed_events,
            committed_baseline,
            retry_events,
            retry_provider.requests,
            final_baseline,
        )

    (
        cancelled_task,
        cancelling,
        cancelled_requests,
        committed_events,
        committed_baseline,
        retry_events,
        retry_requests,
        final_baseline,
    ) = asyncio.run(run())

    assert cancelling == 1
    assert cancelled_task.cancelled()
    assert cancelled_task.cancelling() == 0
    assert cancelled_requests == []
    assert [record.event.payload["status"] for record in committed_events] == ["first_seen"]
    assert committed_baseline.generation == 1
    assert committed_baseline.accepted_event_id == committed_events[0].event.id
    assert len(retry_requests) == 1
    retry_checked = [
        event for event in retry_events if event.type == EventType.MCP_MANIFEST_CHECKED
    ]
    assert [event.payload["status"] for event in retry_checked] == ["unchanged"]
    assert final_baseline.generation == 1
    assert final_baseline.accepted_event_id == committed_baseline.accepted_event_id


def test_runtime_never_reclassifies_later_tool_name_revision_as_first_seen() -> None:
    async def run():
        store = InMemorySessionStore()
        echo_toolset = _fake_toolset(definitions=_fake_tool_definitions("echo"))
        await _run_mcp_manifest_session(
            store=store,
            session_id="mcp_manifest_same_server_initial",
            toolset=echo_toolset,
        )
        await _run_mcp_manifest_session(
            store=store,
            session_id="mcp_manifest_same_server_changed",
            toolset=_fake_toolset(definitions=_fake_tool_definitions("summarize")),
        )
        return await store.query_events(
            EventQuery(event_type=EventType.MCP_MANIFEST_CHECKED, limit=10)
        )

    records = asyncio.run(run())

    assert [record.event.payload["status"] for record in records] == [
        "first_seen",
        "changed",
    ]
    assert records[1].event.payload["previous"]["event_id"] == records[0].event.id
    assert records[1].event.payload["diff"]["added_tools"] == [
        _opaque_mcp_tool_id("local-mcp", "summarize")
    ]
    assert records[1].event.payload["diff"]["removed_tools"] == [
        _opaque_mcp_tool_id("local-mcp", "echo")
    ]


def test_runtime_blocks_safety_tool_removal_after_three_manifest_revisions() -> None:
    async def run():
        store = InMemorySessionStore()
        await _run_mcp_manifest_session(
            store=store,
            session_id="mcp_manifest_revision_1",
            toolset=_fake_toolset(definitions=_fake_tool_definitions("read", "safety_check")),
        )
        await _run_mcp_manifest_session(
            store=store,
            session_id="mcp_manifest_revision_2",
            toolset=_fake_toolset(
                definitions=_fake_tool_definitions(
                    "read",
                    "safety_check",
                    "summarize",
                )
            ),
        )

        provider = FakeProvider(
            [[ModelStreamEvent.text_delta("should-not-run"), ModelStreamEvent.completed({})]]
        )
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            mcp_manifest_policy=McpManifestPolicy(
                on_changed=McpManifestPolicyAction.ALLOW,
                on_tools_removed=McpManifestPolicyAction.BLOCK,
            ),
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=_fake_toolset(definitions=_fake_tool_definitions("read", "summarize")).tools,
        )
        events = await _collect_events(
            app.run(
                RunRequest(
                    session_id="mcp_manifest_revision_3",
                    agent_name="assistant",
                    messages=[Message.text("user", "hello")],
                )
            )
        )
        checked = await store.query_events(
            EventQuery(event_type=EventType.MCP_MANIFEST_CHECKED, limit=10)
        )
        return events, checked, provider.requests

    events, checked, requests = asyncio.run(run())

    assert requests == []
    assert [record.event.payload["status"] for record in checked] == [
        "first_seen",
        "changed",
    ]
    blocked = [event for event in events if event.type == EventType.MCP_MANIFEST_BLOCKED]
    assert len(blocked) == 1
    assert blocked[0].payload["status"] == "changed"
    assert blocked[0].payload["previous"]["event_id"] == checked[1].event.id
    assert blocked[0].payload["diff"]["removed_tools"] == [
        _opaque_mcp_tool_id("local-mcp", "safety_check")
    ]
    assert blocked[0].payload["policy"]["matched_changes"] == ["tools_removed"]


def test_runtime_manifest_authority_ignores_public_evidence_mutation() -> None:
    async def run():
        store = ManifestMutationSessionStore()
        await _run_mcp_manifest_session(
            store=store,
            session_id="mcp_manifest_immutable_initial",
            toolset=_fake_toolset(definitions=_fake_tool_definitions("read", "safety_check")),
        )

        candidate = _fake_toolset(definitions=_fake_tool_definitions("read"))
        original_identity = candidate.manifest_identity
        original_hash = candidate.manifest_hash
        original_server_hash = candidate.manifest_server_hash
        returned_tools = candidate.manifest_tools
        for field_name, replacement in (
            ("manifest_identity", "sha256:" + "a" * 64),
            ("manifest_hash", "sha256:" + "b" * 64),
            ("manifest_server_hash", "sha256:" + "c" * 64),
            ("manifest_tools", ()),
        ):
            with pytest.raises(AttributeError):
                setattr(candidate, field_name, replacement)

        def mutate_public_evidence_during_baseline_load() -> None:
            returned_tools[0]["mcp_name"] = "safety_check"
            returned_tools[0]["hash"] = "sha256:" + "f" * 64
            candidate.server.connection_id = "attacker-controlled-namespace"
            candidate.server.name = "attacker-controlled-name"

        store.on_manifest_load = mutate_public_evidence_during_baseline_load
        provider = FakeProvider([[ModelStreamEvent.completed({})]])
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            mcp_manifest_policy=McpManifestPolicy(
                on_first_seen=McpManifestPolicyAction.ALLOW,
                on_tools_removed=McpManifestPolicyAction.BLOCK,
            ),
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=candidate.tools,
        )
        events = await _collect_events(
            app.run(
                RunRequest(
                    session_id="mcp_manifest_immutable_candidate",
                    agent_name="assistant",
                    messages=[Message.text("user", "hello")],
                )
            )
        )
        return (
            candidate,
            original_identity,
            original_hash,
            original_server_hash,
            events,
            provider.requests,
        )

    (
        candidate,
        original_identity,
        original_hash,
        original_server_hash,
        events,
        requests,
    ) = asyncio.run(run())

    assert candidate.manifest_identity == original_identity
    assert candidate.manifest_hash == original_hash
    assert candidate.manifest_server_hash == original_server_hash
    assert candidate.manifest_tools[0]["mcp_name"] == "read"
    assert requests == []
    blocked = [event for event in events if event.type == EventType.MCP_MANIFEST_BLOCKED]
    assert len(blocked) == 1
    assert blocked[0].payload["status"] == "changed"
    assert blocked[0].payload["diff"]["removed_tools"] == [
        _opaque_mcp_tool_id("local-mcp", "safety_check")
    ]
    assert blocked[0].payload["policy"]["matched_changes"] == ["tools_removed"]


def test_runtime_blocks_registered_mcp_subset_that_removes_safety_tool() -> None:
    async def run():
        store = InMemorySessionStore()
        definitions = _fake_tool_definitions("read", "safety_check")
        await _run_mcp_manifest_session(
            store=store,
            session_id="mcp_manifest_exposure_full",
            toolset=_fake_toolset(definitions=definitions),
        )

        candidate = _fake_toolset(definitions=definitions)
        provider = FakeProvider([[ModelStreamEvent.completed({})]])
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            mcp_manifest_policy=McpManifestPolicy(
                on_changed=McpManifestPolicyAction.ALLOW,
                on_tools_removed=McpManifestPolicyAction.BLOCK,
            ),
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=(candidate.tools[0],),
        )
        events = await _collect_events(
            app.run(
                RunRequest(
                    session_id="mcp_manifest_exposure_subset",
                    agent_name="assistant",
                    messages=[Message.text("user", "hello")],
                )
            )
        )
        return events, provider.requests

    events, requests = asyncio.run(run())

    assert requests == []
    blocked = [event for event in events if event.type == EventType.MCP_MANIFEST_BLOCKED]
    assert len(blocked) == 1
    assert (
        blocked[0].payload["source_manifest_hash"]
        == blocked[0].payload["previous"]["source_manifest_hash"]
    )
    assert blocked[0].payload["advertised_tool_count"] == 2
    assert blocked[0].payload["tool_count"] == 1
    assert blocked[0].payload["diff"]["removed_tools"] == [
        _opaque_mcp_tool_id("local-mcp", "safety_check")
    ]
    assert blocked[0].payload["policy"]["matched_changes"] == ["tools_removed"]


def test_runtime_tracks_registered_mcp_alias_changes() -> None:
    async def run():
        store = InMemorySessionStore()
        first = _fake_toolset(definitions=_fake_tool_definitions("read"))
        first_alias = McpToolAdapter(
            toolset=first,
            definition=first.definitions[0],
            name="approved_read",
        )
        await _run_mcp_manifest_session(
            store=store,
            session_id="mcp_manifest_alias_first",
            toolset=first,
            tools=(first_alias,),
        )

        second = _fake_toolset(definitions=_fake_tool_definitions("read"))
        second_alias = McpToolAdapter(
            toolset=second,
            definition=second.definitions[0],
            name="renamed_read",
        )
        provider = FakeProvider([[ModelStreamEvent.completed({})]])
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            mcp_manifest_policy=McpManifestPolicy(on_changed=McpManifestPolicyAction.BLOCK),
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=(second_alias,),
        )
        events = await _collect_events(
            app.run(
                RunRequest(
                    session_id="mcp_manifest_alias_second",
                    agent_name="assistant",
                    messages=[Message.text("user", "hello")],
                )
            )
        )
        return events, provider.requests

    events, requests = asyncio.run(run())

    assert requests == []
    blocked = [event for event in events if event.type == EventType.MCP_MANIFEST_BLOCKED]
    assert len(blocked) == 1
    assert blocked[0].payload["change_classes"] == ["tools_added", "tools_removed"]


def test_runtime_blocks_registered_mcp_provider_contract_change() -> None:
    async def run():
        store = InMemorySessionStore()
        await _run_mcp_manifest_session(
            store=store,
            session_id="mcp_manifest_provider_contract_first",
            toolset=_fake_toolset(definitions=_fake_tool_definitions("read")),
        )

        candidate = _fake_toolset(definitions=_fake_tool_definitions("read"))
        candidate.tools[0].spec = candidate.tools[0].spec.model_copy(
            update={"description": "A newly privileged provider-facing contract."}
        )
        provider = FakeProvider([[ModelStreamEvent.completed({})]])
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            mcp_manifest_policy=McpManifestPolicy(
                on_changed=McpManifestPolicyAction.ALLOW,
                on_tools_changed=McpManifestPolicyAction.BLOCK,
            ),
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=candidate.tools,
        )
        events = await _collect_events(
            app.run(
                RunRequest(
                    session_id="mcp_manifest_provider_contract_second",
                    agent_name="assistant",
                    messages=[Message.text("user", "hello")],
                )
            )
        )
        return events, provider.requests

    events, requests = asyncio.run(run())

    assert requests == []
    blocked = [event for event in events if event.type == EventType.MCP_MANIFEST_BLOCKED]
    assert len(blocked) == 1
    assert (
        blocked[0].payload["source_manifest_hash"]
        == blocked[0].payload["previous"]["source_manifest_hash"]
    )
    assert blocked[0].payload["policy"]["matched_changes"] == ["tools_changed"]


def test_mcp_adapter_requires_an_advertised_definition_and_allows_distinct_aliases() -> None:
    toolset = _fake_toolset(definitions=_fake_tool_definitions("read"))

    with pytest.raises(ValueError, match="advertised"):
        McpToolAdapter(
            toolset=toolset,
            definition=McpToolDefinition(name="unadvertised", input_schema={"type": "object"}),
        )

    first = McpToolAdapter(
        toolset=toolset,
        definition=toolset.definitions[0],
        name="read_primary",
    )
    second = McpToolAdapter(
        toolset=toolset,
        definition=toolset.definitions[0],
        name="read_fallback",
    )

    assert first._manifest_binding.mcp_name == "read"
    assert second._manifest_binding.mcp_name == "read"
    assert first.name != second.name


def test_mcp_adapter_dispatch_remains_bound_to_accepted_definition_and_session() -> None:
    async def run():
        toolset = _fake_toolset(definitions=_fake_tool_definitions("read"))
        adapter = McpToolAdapter(
            toolset=toolset,
            definition=toolset.definitions[0],
            name="approved_read",
        )
        returned_definition = adapter.definition
        returned_definition.name = "dangerous_delete"
        returned_server = adapter.server
        returned_server.name = "attacker-server"
        for field_name, replacement in (
            ("definition", returned_definition),
            ("server", returned_server),
            ("toolset", _fake_toolset()),
            ("mcp_manifest_hash", "sha256:" + "f" * 64),
        ):
            with pytest.raises(AttributeError):
                setattr(adapter, field_name, replacement)
        with pytest.raises(AttributeError):
            toolset.session = FakeMcpSession()

        provider = FakeProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_read",
                        name="approved_read",
                        arguments={"text": "hello"},
                    ),
                    ModelStreamEvent.completed({}),
                ],
                [ModelStreamEvent.completed({})],
            ]
        )
        app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=(adapter,),
        )
        events = await _collect_events(
            app.run(
                RunRequest(
                    session_id="mcp_manifest_fixed_dispatch",
                    agent_name="assistant",
                    messages=[Message.text("user", "read")],
                )
            )
        )
        return events, toolset.session.calls, provider.requests

    events, calls, requests = asyncio.run(run())

    assert len(requests) == 2
    assert [tool["name"] for tool in requests[0].tools] == ["approved_read"]
    assert calls == [("read", {"text": "hello"})]
    assert any(event.type == EventType.TOOL_CALL_COMPLETED for event in events)


def test_sqlite_manifest_baseline_survives_app_reconstruction(tmp_path: Path) -> None:
    async def run():
        database = tmp_path / "mcp-history.sqlite"
        first_store = SQLiteSessionStore(database)
        await _run_mcp_manifest_session(
            store=first_store,
            session_id="sqlite_mcp_manifest_1",
            toolset=_fake_toolset(definitions=_fake_tool_definitions("safe")),
        )
        await first_store.close()

        second_store = SQLiteSessionStore(database)
        provider = FakeProvider(
            [[ModelStreamEvent.text_delta("should-not-run"), ModelStreamEvent.completed({})]]
        )
        app = CayuApp(
            session_store=second_store,
            enable_logging=False,
            mcp_manifest_policy=McpManifestPolicy(on_changed=McpManifestPolicyAction.BLOCK),
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=_fake_toolset(definitions=_fake_tool_definitions("unsafe")).tools,
        )
        events = await _collect_events(
            app.run(
                RunRequest(
                    session_id="sqlite_mcp_manifest_2",
                    agent_name="assistant",
                    messages=[Message.text("user", "hello")],
                )
            )
        )
        await second_store.close()
        return events, provider.requests

    events, requests = asyncio.run(run())

    assert requests == []
    blocked = [event for event in events if event.type == EventType.MCP_MANIFEST_BLOCKED]
    assert len(blocked) == 1
    assert blocked[0].payload["status"] == "changed"
    assert blocked[0].payload["previous"]["session_ref"].startswith("sha256:")
    assert "session_id" not in blocked[0].payload["previous"]


def test_sqlite_manifest_baseline_survives_accepted_session_deletion(tmp_path: Path) -> None:
    async def run():
        database = tmp_path / "mcp-history-retention.sqlite"
        first_store = SQLiteSessionStore(database)
        await _run_mcp_manifest_session(
            store=first_store,
            session_id="sqlite_mcp_manifest_retained_1",
            toolset=_fake_toolset(definitions=_fake_tool_definitions("safe")),
        )
        await first_store.delete_session("sqlite_mcp_manifest_retained_1")
        await first_store.close()

        second_store = SQLiteSessionStore(database)
        provider = FakeProvider([[ModelStreamEvent.completed({})]])
        app = CayuApp(
            session_store=second_store,
            enable_logging=False,
            mcp_manifest_policy=McpManifestPolicy(on_changed=McpManifestPolicyAction.BLOCK),
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=_fake_toolset(definitions=_fake_tool_definitions("unsafe")).tools,
        )
        events = await _collect_events(
            app.run(
                RunRequest(
                    session_id="sqlite_mcp_manifest_retained_2",
                    agent_name="assistant",
                    messages=[Message.text("user", "hello")],
                )
            )
        )
        await second_store.close()
        return events, provider.requests

    events, requests = asyncio.run(run())

    assert requests == []
    blocked = [event for event in events if event.type == EventType.MCP_MANIFEST_BLOCKED]
    assert len(blocked) == 1
    assert blocked[0].payload["status"] == "changed"
    assert blocked[0].payload["previous"]["session_ref"].startswith("sha256:")
    assert "session_id" not in blocked[0].payload["previous"]


@pytest.mark.parametrize("corruption_stage", ["load", "compare"])
@pytest.mark.parametrize("corruption", ["malformed_json", "invalid_evidence"])
def test_sqlite_runtime_sanitizes_corrupt_manifest_baseline(
    tmp_path: Path,
    corruption_stage: str,
    corruption: str,
) -> None:
    secret = f"raw-secret-{corruption_stage}-{corruption}"
    path = tmp_path / f"manifest-{corruption_stage}-{corruption}.sqlite3"

    class CorruptAfterLoadStore(SQLiteSessionStore):
        invocation_lifecycle_command_version = 1
        corrupt_after_load = False

        def corrupt_baseline(self, history_key: str) -> None:
            connection = sqlite3.connect(path)
            try:
                row = connection.execute(
                    "SELECT baseline_json FROM cayu_mcp_manifest_baselines WHERE history_key = ?",
                    (history_key,),
                ).fetchone()
                assert row is not None
                if corruption == "malformed_json":
                    corrupted = f'{{"manifest_hash":"{secret}"'
                else:
                    payload = json.loads(row[0])
                    payload["manifest_hash"] = secret
                    corrupted = json.dumps(payload)
                connection.execute(
                    "UPDATE cayu_mcp_manifest_baselines SET baseline_json = ? "
                    "WHERE history_key = ?",
                    (corrupted, history_key),
                )
                connection.commit()
            finally:
                connection.close()

        async def load_mcp_manifest_baselines(self, history_keys):
            loaded = await super().load_mcp_manifest_baselines(history_keys)
            if self.corrupt_after_load:
                self.corrupt_after_load = False
                self.corrupt_baseline(history_keys[0])
            return loaded

    async def run():
        store = CorruptAfterLoadStore(path)
        toolset = _fake_toolset()
        await _run_mcp_manifest_session(
            store=store,
            session_id="mcp_manifest_corruption_seed",
            toolset=toolset,
        )
        accepted = (
            await store.query_events(
                EventQuery(event_type=EventType.MCP_MANIFEST_CHECKED, limit=10)
            )
        )[0].event
        history_key = accepted.payload["history_key"]

        connection = sqlite3.connect(path)
        try:
            row = connection.execute(
                "SELECT baseline_json FROM cayu_mcp_manifest_baselines WHERE history_key = ?",
                (history_key,),
            ).fetchone()
            assert row is not None
            valid_baseline_json = row[0]
        finally:
            connection.close()

        if corruption_stage == "load":
            store.corrupt_baseline(history_key)
        else:
            store.corrupt_after_load = True

        blocked_provider = FakeProvider([[ModelStreamEvent.completed({})]])
        blocked_app = CayuApp(session_store=store, enable_logging=False)
        blocked_app.register_provider(blocked_provider, default=True)
        blocked_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=toolset.tools,
        )
        blocked_events = await _collect_events(
            blocked_app.run(
                RunRequest(
                    session_id=f"mcp_manifest_corrupt_{corruption_stage}_{corruption}",
                    agent_name="assistant",
                    messages=[Message.text("user", "blocked")],
                )
            )
        )
        durable_events = await store.query_events(EventQuery(limit=100))

        connection = sqlite3.connect(path)
        try:
            connection.execute(
                "UPDATE cayu_mcp_manifest_baselines SET baseline_json = ? WHERE history_key = ?",
                (valid_baseline_json, history_key),
            )
            connection.commit()
        finally:
            connection.close()

        recovered_provider = FakeProvider([[ModelStreamEvent.completed({})]])
        recovered_app = CayuApp(session_store=store, enable_logging=False)
        recovered_app.register_provider(recovered_provider, default=True)
        recovered_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=toolset.tools,
        )
        recovered_events = await _collect_events(
            recovered_app.run(
                RunRequest(
                    session_id=f"mcp_manifest_recovered_{corruption_stage}_{corruption}",
                    agent_name="assistant",
                    messages=[Message.text("user", "retry")],
                )
            )
        )
        recovered_baseline = (await store.load_mcp_manifest_baselines((history_key,))).baselines[
            history_key
        ]
        await store.close()
        return (
            blocked_events,
            blocked_provider.requests,
            durable_events,
            recovered_events,
            recovered_provider.requests,
            recovered_baseline,
        )

    (
        blocked_events,
        blocked_requests,
        durable_events,
        recovered_events,
        recovered_requests,
        recovered_baseline,
    ) = asyncio.run(run())

    assert blocked_requests == []
    blocked = [event for event in blocked_events if event.type == EventType.MCP_MANIFEST_BLOCKED]
    assert len(blocked) == 1
    assert blocked[0].payload["status"] == "history_conflict"
    assert blocked[0].payload["reason"] == "authoritative_baseline_invalid"
    failed = [event for event in blocked_events if event.type == EventType.SESSION_FAILED]
    assert len(failed) == 1
    assert failed[0].payload["error_type"] == "McpManifestHistoryConflict"
    assert secret not in json.dumps(
        [record.event.model_dump(mode="json") for record in durable_events]
    )
    assert len(recovered_requests) == 1
    recovered_checked = [
        event for event in recovered_events if event.type == EventType.MCP_MANIFEST_CHECKED
    ]
    assert [event.payload["status"] for event in recovered_checked] == ["unchanged"]
    assert recovered_baseline.generation == 1
    assert recovered_baseline.accepted_event_id != recovered_checked[0].id


def test_runtime_requires_explicit_identity_after_schema_upgrade(tmp_path: Path) -> None:
    async def run():
        database = tmp_path / "mcp-upgraded-history.sqlite"
        legacy_store = SQLiteSessionStore(database)
        await _seed_legacy_mcp_manifest_event(
            store=legacy_store,
            session_id="legacy_mcp_manifest_1",
            toolset=_fake_toolset(definitions=_fake_tool_definitions("read", "safety_check")),
        )
        await legacy_store.close()

        connection = sqlite3.connect(database)
        try:
            connection.execute("DELETE FROM cayu_schema_migrations WHERE revision = 22")
            connection.execute("DROP TABLE cayu_mcp_manifest_baselines")
            connection.execute("PRAGMA user_version = 21")
            connection.commit()
        finally:
            connection.close()

        store = SQLiteSessionStore(
            database,
            schema_mode=schema_migrations.SchemaMode.MIGRATE,
        )
        provider = FakeProvider([[ModelStreamEvent.completed({})]])
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            mcp_manifest_policy=McpManifestPolicy(
                on_first_seen=McpManifestPolicyAction.ALLOW,
                on_tools_removed=McpManifestPolicyAction.BLOCK,
            ),
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=_fake_toolset(
                definitions=_fake_tool_definitions("read"),
                connection_id=None,
            ).tools,
        )
        events = await _collect_events(
            app.run(
                RunRequest(
                    session_id="legacy_mcp_manifest_2",
                    agent_name="assistant",
                    messages=[Message.text("user", "hello")],
                )
            )
        )
        durable_blocked = await store.query_events(
            EventQuery(event_type=EventType.MCP_MANIFEST_BLOCKED, limit=10)
        )
        await store.close()
        return events, durable_blocked, provider.requests

    events, durable_blocked, requests = asyncio.run(run())

    assert requests == []
    blocked = [event for event in events if event.type == EventType.MCP_MANIFEST_BLOCKED]
    assert len(blocked) == 1
    assert blocked[0].payload["status"] == "history_conflict"
    assert blocked[0].payload["reason"] == "connection_identity_required"
    assert [record.sequence for record in durable_blocked] == [public_event_sequence(blocked[0].id)]


def test_concurrent_manifest_checks_serialize_different_first_baselines() -> None:
    async def run():
        store = RacingManifestSessionStore()
        providers = [
            FakeProvider([[ModelStreamEvent.completed({})]]),
            FakeProvider([[ModelStreamEvent.completed({})]]),
        ]
        apps = []
        for index, tool_name in enumerate(("first", "second")):
            app = CayuApp(
                session_store=store,
                enable_logging=False,
                mcp_manifest_policy=McpManifestPolicy(
                    on_first_seen=McpManifestPolicyAction.ALLOW,
                    on_changed=McpManifestPolicyAction.BLOCK,
                ),
            )
            app.register_provider(providers[index], default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=_fake_toolset(definitions=_fake_tool_definitions(tool_name)).tools,
            )
            apps.append(app)
        results = await asyncio.gather(
            *(
                _collect_events(
                    app.run(
                        RunRequest(
                            session_id=f"mcp_manifest_race_{index}",
                            agent_name="assistant",
                            messages=[Message.text("user", "hello")],
                        )
                    )
                )
                for index, app in enumerate(apps)
            )
        )
        checked = await store.query_events(
            EventQuery(event_type=EventType.MCP_MANIFEST_CHECKED, limit=10)
        )
        blocked = await store.query_events(
            EventQuery(event_type=EventType.MCP_MANIFEST_BLOCKED, limit=10)
        )
        return results, checked, blocked, providers

    _, checked, blocked, providers = asyncio.run(run())

    assert sum(len(provider.requests) for provider in providers) == 1
    assert [record.event.payload["status"] for record in checked] == ["first_seen"]
    assert [record.event.payload["status"] for record in blocked] == ["changed"]
    assert blocked[0].event.payload["previous"]["event_id"] == checked[0].event.id


def test_manifest_publication_lost_ack_is_fail_closed_and_retry_safe() -> None:
    class CommitThenRaiseManifestStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self):
            super().__init__()
            self.lose_next_ack = True

        async def compare_and_publish_mcp_manifest_checks(self, *args, **kwargs):
            result = await super().compare_and_publish_mcp_manifest_checks(*args, **kwargs)
            if result.published and self.lose_next_ack:
                self.lose_next_ack = False
                raise ConnectionError("manifest publication acknowledgement lost")
            return result

    async def run():
        store = CommitThenRaiseManifestStore()
        toolset = _fake_toolset()
        first_provider = FakeProvider([[ModelStreamEvent.completed({})]])
        first_app = CayuApp(session_store=store, enable_logging=False)
        first_app.register_provider(first_provider, default=True)
        first_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=toolset.tools,
        )
        first_events = await _collect_events(
            first_app.run(
                RunRequest(
                    session_id="mcp_manifest_lost_ack_1",
                    agent_name="assistant",
                    messages=[Message.text("user", "hello")],
                )
            )
        )

        second_provider = FakeProvider([[ModelStreamEvent.completed({})]])
        second_app = CayuApp(session_store=store, enable_logging=False)
        second_app.register_provider(second_provider, default=True)
        second_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=toolset.tools,
        )
        second_events = await _collect_events(
            second_app.run(
                RunRequest(
                    session_id="mcp_manifest_lost_ack_2",
                    agent_name="assistant",
                    messages=[Message.text("user", "retry")],
                )
            )
        )
        durable = await store.query_events(
            EventQuery(event_type=EventType.MCP_MANIFEST_CHECKED, limit=10)
        )
        baseline = (
            await store.load_mcp_manifest_baselines((durable[0].event.payload["history_key"],))
        ).baselines[durable[0].event.payload["history_key"]]
        return (
            first_events,
            first_provider.requests,
            second_events,
            second_provider.requests,
            durable,
            baseline,
        )

    (
        first_events,
        first_requests,
        second_events,
        second_requests,
        durable,
        baseline,
    ) = asyncio.run(run())

    assert first_requests == []
    assert first_events[-1].type == EventType.SESSION_FAILED
    assert len(second_requests) == 1
    second_checked = [
        event for event in second_events if event.type == EventType.MCP_MANIFEST_CHECKED
    ]
    assert [event.payload["status"] for event in second_checked] == ["unchanged"]
    assert [record.event.payload["status"] for record in durable] == [
        "first_seen",
        "unchanged",
    ]
    assert baseline.generation == 1
    assert baseline.accepted_event_id == durable[0].event.id


def test_manifest_publication_rejects_incomplete_success_ack_and_retries_safely() -> None:
    class IncompleteSuccessManifestStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self):
            super().__init__()
            self.return_incomplete_success = True

        async def compare_and_publish_mcp_manifest_checks(self, *args, **kwargs):
            result = await super().compare_and_publish_mcp_manifest_checks(*args, **kwargs)
            if result.published and self.return_incomplete_success:
                self.return_incomplete_success = False
                return McpManifestPublicationResult(
                    published=True,
                    baselines={},
                )
            return result

    async def run():
        store = IncompleteSuccessManifestStore()
        toolset = _fake_toolset()
        first_provider = FakeProvider([[ModelStreamEvent.completed({})]])
        first_app = CayuApp(session_store=store, enable_logging=False)
        first_app.register_provider(first_provider, default=True)
        first_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=toolset.tools,
        )
        first_events = await _collect_events(
            first_app.run(
                RunRequest(
                    session_id="mcp_manifest_incomplete_ack_1",
                    agent_name="assistant",
                    messages=[Message.text("user", "hello")],
                )
            )
        )

        second_provider = FakeProvider([[ModelStreamEvent.completed({})]])
        second_app = CayuApp(session_store=store, enable_logging=False)
        second_app.register_provider(second_provider, default=True)
        second_app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=toolset.tools,
        )
        second_events = await _collect_events(
            second_app.run(
                RunRequest(
                    session_id="mcp_manifest_incomplete_ack_2",
                    agent_name="assistant",
                    messages=[Message.text("user", "retry")],
                )
            )
        )
        durable = await store.query_events(
            EventQuery(event_type=EventType.MCP_MANIFEST_CHECKED, limit=10)
        )
        return (
            first_events,
            first_provider.requests,
            second_events,
            second_provider.requests,
            durable,
        )

    first_events, first_requests, second_events, second_requests, durable = asyncio.run(run())

    assert first_requests == []
    assert first_events[-1].type == EventType.SESSION_FAILED
    assert first_events[-1].payload["error_type"] == "McpManifestHistoryConflict"
    assert len(second_requests) == 1
    second_checked = [
        event for event in second_events if event.type == EventType.MCP_MANIFEST_CHECKED
    ]
    assert [event.payload["status"] for event in second_checked] == ["unchanged"]
    assert [record.event.payload["status"] for record in durable] == [
        "first_seen",
        "unchanged",
    ]


def test_runtime_scopes_mcp_manifest_comparison_by_environment() -> None:
    async def run():
        store = InMemorySessionStore()
        await _run_mcp_manifest_session(
            store=store,
            session_id="mcp_manifest_env_scoped_1",
            toolset=_fake_toolset(),
            environment_name="local",
        )
        await _run_mcp_manifest_session(
            store=store,
            session_id="mcp_manifest_env_scoped_2",
            toolset=_fake_toolset(),
            environment_name=None,
        )
        return await store.query_events(
            EventQuery(event_type=EventType.MCP_MANIFEST_CHECKED, limit=10)
        )

    records = asyncio.run(run())

    assert [record.event.environment_name for record in records] == ["local", None]
    assert [record.event.payload["status"] for record in records] == [
        "first_seen",
        "first_seen",
    ]


def test_stdio_mcp_client_replies_to_unsupported_server_requests() -> None:
    async def run():
        client = StdioMcpClient()
        session = await client.connect(_fake_server_spec())
        try:
            return await session.call_tool(
                "echo",
                {"text": "after server request", "server_request_first": True},
            )
        finally:
            await session.close()

    result = asyncio.run(run())

    assert result.content == [{"type": "text", "text": "echo: after server request"}]
    assert result.structured_content == {"echoed": "after server request"}


def test_stdio_mcp_client_routes_concurrent_out_of_order_responses() -> None:
    async def run():
        client = StdioMcpClient()
        session = await client.connect(_fake_server_spec())
        try:
            first, second = await asyncio.gather(
                session.call_tool("echo", {"text": "first", "defer_response": True}),
                session.call_tool("echo", {"text": "second"}),
            )
            return first, second
        finally:
            await session.close()

    first, second = asyncio.run(run())

    assert first.structured_content == {"echoed": "first"}
    assert second.structured_content == {"echoed": "second"}


def test_stdio_mcp_client_cleans_pending_request_on_cancellation() -> None:
    async def run():
        client = StdioMcpClient()
        session = await client.connect(_fake_server_spec())
        assert isinstance(session, StdioMcpSession)
        request_written = asyncio.Event()
        original_write_with_timeout = session._write_with_timeout

        async def capture_tool_call_write(
            payload: dict[str, Any],
            *,
            timeout_message: str,
            call_deadline: float | None = None,
        ) -> None:
            await original_write_with_timeout(
                payload,
                timeout_message=timeout_message,
                call_deadline=call_deadline,
            )
            if payload.get("method") == "tools/call":
                request_written.set()

        try:
            session._write_with_timeout = capture_tool_call_write
            task = asyncio.create_task(
                session.call_tool("echo", {"text": "cancelled", "defer_response": True})
            )
            await request_written.wait()
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            return dict(session._pending)
        finally:
            await session.close()

    pending = asyncio.run(run())

    assert pending == {}


def test_stdio_mcp_client_sends_cancelled_notification_when_request_is_cancelled() -> None:
    async def run():
        client = StdioMcpClient()
        session = await client.connect(_fake_server_spec())
        assert isinstance(session, StdioMcpSession)
        notifications: list[tuple[str, dict[str, Any]]] = []
        request_written = asyncio.Event()
        original_write_with_timeout = session._write_with_timeout

        async def capture_notify(method: str, params: dict[str, Any]) -> None:
            notifications.append((method, params))

        async def capture_tool_call_write(
            payload: dict[str, Any],
            *,
            timeout_message: str,
            call_deadline: float | None = None,
        ) -> None:
            await original_write_with_timeout(
                payload,
                timeout_message=timeout_message,
                call_deadline=call_deadline,
            )
            if payload.get("method") == "tools/call":
                request_written.set()

        try:
            session._write_with_timeout = capture_tool_call_write
            session._notify = capture_notify
            task = asyncio.create_task(
                session.call_tool("echo", {"text": "cancelled", "defer_response": True})
            )
            await request_written.wait()
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            return notifications
        finally:
            await session.close()

    notifications = asyncio.run(run())

    assert notifications == [
        (
            "notifications/cancelled",
            {
                "requestId": 2,
                "reason": "Cayu caller cancelled the request.",
            },
        )
    ]


def test_stdio_mcp_client_cleans_pending_request_when_write_is_cancelled() -> None:
    async def run():
        client = StdioMcpClient()
        session = await client.connect(_fake_server_spec())
        assert isinstance(session, StdioMcpSession)

        async def cancel_write(payload: dict[str, Any]) -> None:
            raise asyncio.CancelledError

        try:
            session._write = cancel_write
            with pytest.raises(McpProtocolError, match="cancelled unexpectedly"):
                await session._request("tools/list", {})
            with pytest.raises(McpProtocolError, match="closed"):
                await session.list_tools()
            await asyncio.wait_for(session.process.wait(), timeout=1)
            current = asyncio.current_task()
            assert current is not None
            return dict(session._pending), session.process.returncode, current.cancelling()
        finally:
            await session.close()

    pending, returncode, cancelling = asyncio.run(run())

    assert pending == {}
    assert returncode is not None
    assert cancelling == 0


def test_stdio_mcp_client_times_out_blocked_request_write() -> None:
    async def run():
        client = StdioMcpClient(write_timeout_s=0.01)
        session = await client.connect(_fake_server_spec())
        assert isinstance(session, StdioMcpSession)
        write_started = asyncio.Event()

        async def block_write(payload: dict[str, Any]) -> None:
            write_started.set()
            await asyncio.Event().wait()

        try:
            session._write = block_write
            with pytest.raises(TimeoutError, match="write timed out"):
                await session._request("tools/list", {})
            with pytest.raises(McpProtocolError, match="closed"):
                await session.list_tools()
            await asyncio.wait_for(session.process.wait(), timeout=1)
            return write_started.is_set(), dict(session._pending), session.process.returncode
        finally:
            await session.close()

    write_started, pending, returncode = asyncio.run(run())

    assert write_started is True
    assert pending == {}
    assert returncode is not None


def test_stdio_mcp_client_sends_cancelled_notification_when_request_times_out() -> None:
    async def run():
        client = StdioMcpClient()
        session = await client.connect(_fake_server_spec())
        assert isinstance(session, StdioMcpSession)
        notifications: list[tuple[str, dict[str, Any]]] = []

        async def capture_notify(method: str, params: dict[str, Any]) -> None:
            notifications.append((method, params))

        try:
            session.request_timeout_s = 0.01
            session._notify = capture_notify
            with pytest.raises(TimeoutError, match="timed out"):
                await session.call_tool("echo", {"text": "timeout", "defer_response": True})
            return notifications
        finally:
            await session.close()

    notifications = asyncio.run(run())

    assert notifications == [
        (
            "notifications/cancelled",
            {
                "requestId": 2,
                "reason": "Cayu request timed out.",
            },
        )
    ]


def test_stdio_mcp_client_cancelled_notification_is_timeout_bounded() -> None:
    async def run():
        client = StdioMcpClient(cancellation_notification_timeout_s=0.01)
        session = await client.connect(_fake_server_spec())
        assert isinstance(session, StdioMcpSession)
        notification_started = asyncio.Event()

        async def block_notify(method: str, params: dict[str, Any]) -> None:
            notification_started.set()
            await asyncio.Event().wait()

        try:
            session._notify = block_notify
            await asyncio.wait_for(
                session._send_request_cancelled_notification(
                    99,
                    method_name="tools/call",
                    reason="test",
                ),
                timeout=0.5,
            )
            return notification_started.is_set()
        finally:
            await session.close()

    assert asyncio.run(run()) is True


def test_stdio_mcp_client_closes_session_when_cancelled_notification_write_is_cancelled() -> None:
    async def run():
        client = StdioMcpClient(cancellation_notification_timeout_s=0.01)
        session = await client.connect(_fake_server_spec())
        assert isinstance(session, StdioMcpSession)

        async def block_notify_write(payload: dict[str, Any]) -> None:
            await asyncio.Event().wait()

        try:
            session._write = block_notify_write
            await session._send_request_cancelled_notification(
                99,
                method_name="tools/call",
                reason="test",
            )
            with pytest.raises(McpProtocolError, match="closed"):
                await session.list_tools()
            return session.process.returncode
        finally:
            await session.close()

    assert asyncio.run(run()) is not None


def test_stdio_mcp_client_rejects_unsupported_negotiated_protocol_version() -> None:
    spec = McpServerSpec(
        name="local-mcp",
        command=[sys.executable, str(_FAKE_SERVER)],
        env={"CAYU_FAKE_MCP_PROTOCOL_VERSION": "1999-01-01"},
    )

    with pytest.raises(McpProtocolError, match="unsupported protocol version"):
        asyncio.run(StdioMcpClient().connect(spec))


def test_stdio_session_constructor_revalidates_server_before_background_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_calls = 0

    def forbidden_create_task(coroutine: Any) -> Any:
        nonlocal task_calls
        task_calls += 1
        coroutine.close()
        raise AssertionError("background tasks must not start for invalid MCP configuration")

    monkeypatch.setattr(asyncio, "create_task", forbidden_create_task)
    server = McpServerSpec(name="server", command=["server"])
    server.command[0] = "invalid-stdio-session\x00"

    with pytest.raises(ValueError):
        StdioMcpSession(
            server=server,
            process=None,  # type: ignore[arg-type]
            request_timeout_s=1.0,
            write_timeout_s=1.0,
            graceful_shutdown_timeout_s=1.0,
            cancellation_notification_timeout_s=1.0,
            client_name="cayu",
            client_version="0.1.0",
        )

    assert task_calls == 0


def test_stdio_session_constructor_owns_server_snapshot() -> None:
    async def run() -> tuple[McpServerSpec, McpServerSpec, int | None]:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(_FAKE_SERVER),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        server = McpServerSpec(name="server", command=[sys.executable, str(_FAKE_SERVER)])
        expected = server.model_copy(deep=True)
        session = StdioMcpSession(
            server=server,
            process=process,
            request_timeout_s=1.0,
            write_timeout_s=1.0,
            graceful_shutdown_timeout_s=1.0,
            cancellation_notification_timeout_s=1.0,
            client_name="cayu",
            client_version="0.1.0",
        )
        server.name = "caller-mutated-server"
        try:
            retained = session.server
        finally:
            await session.close()
        return expected, retained, process.returncode

    expected, retained, returncode = asyncio.run(run())

    assert retained == expected
    assert returncode is not None


def test_stdio_mcp_client_accepts_older_supported_protocol_version() -> None:
    spec = McpServerSpec(
        name="local-mcp",
        command=[sys.executable, str(_FAKE_SERVER)],
        env={"CAYU_FAKE_MCP_PROTOCOL_VERSION": "2025-03-26"},
    )

    async def run():
        session = await StdioMcpClient().connect(spec)
        try:
            return session.initialize_result
        finally:
            await session.close()

    initialize_result = asyncio.run(run())
    assert initialize_result.protocol_version == "2025-03-26"


def test_stdio_mcp_client_list_tools_follows_next_cursor() -> None:
    spec = McpServerSpec(
        name="local-mcp",
        command=[sys.executable, str(_FAKE_SERVER)],
        env={"CAYU_FAKE_MCP_PAGINATE": "1"},
    )

    async def run():
        session = await StdioMcpClient().connect(spec)
        try:
            return await session.list_tools()
        finally:
            await session.close()

    tools = asyncio.run(run())
    assert [tool.name for tool in tools] == ["echo", "echo_page_2"]


def test_collect_paginated_rejects_repeated_cursor() -> None:
    from cayu.mcp._jsonrpc import McpPaginatedPage, collect_paginated

    async def request(method, params):
        # Always hand back the same cursor -> would loop forever without a guard.
        return McpPaginatedPage(
            {"tools": [{"name": params.get("cursor", "first")}]},
            "stuck",
        )

    with pytest.raises(McpProtocolError, match="repeated pagination cursor"):
        asyncio.run(collect_paginated(request, "tools/list", "tools"))


def test_mcp_pagination_defaults_are_public_and_bounded() -> None:
    assert DEFAULT_MCP_MAX_LIST_PAGES == mcp_module.DEFAULT_MCP_MAX_LIST_PAGES == 100
    assert DEFAULT_MCP_MAX_LIST_ITEMS == mcp_module.DEFAULT_MCP_MAX_LIST_ITEMS == 10_000


def test_collect_paginated_accepts_exact_page_and_item_limits() -> None:
    from cayu.mcp._jsonrpc import McpPaginatedPage, collect_paginated

    calls: list[dict[str, Any]] = []

    async def request(method, params):
        calls.append(dict(params))
        if not params:
            return McpPaginatedPage({"tools": [{"name": "first"}]}, "next")
        return McpPaginatedPage({"tools": [{"name": "second"}]})

    result = asyncio.run(
        collect_paginated(
            request,
            "tools/list",
            "tools",
            max_pages=2,
            max_items=2,
        )
    )

    assert [item["name"] for item in result] == ["first", "second"]
    assert calls == [{}, {"cursor": "next"}]


def test_collect_paginated_stops_before_requesting_page_over_limit() -> None:
    from cayu.mcp._jsonrpc import McpPaginatedPage, collect_paginated

    calls: list[dict[str, Any]] = []

    async def request(method, params):
        calls.append(dict(params))
        cursor_number = len(calls)
        return McpPaginatedPage(
            {"tools": [{"name": f"tool-{cursor_number}"}]},
            f"cursor-{cursor_number}",
        )

    with pytest.raises(
        McpProtocolError,
        match=r"tools/list.*after 2 pages.*page 3.*max_list_pages=2",
    ):
        asyncio.run(
            collect_paginated(
                request,
                "tools/list",
                "tools",
                max_pages=2,
                max_items=10,
            )
        )

    assert calls == [{}, {"cursor": "cursor-1"}]


def test_collect_paginated_rejects_oversized_cumulative_items() -> None:
    from cayu.mcp._jsonrpc import McpPaginatedPage, collect_paginated

    async def request(method, params):
        if not params:
            return McpPaginatedPage({"resources": [{"uri": "one"}]}, "next")
        return McpPaginatedPage({"resources": [{"uri": "two"}, {"uri": "three"}]})

    with pytest.raises(
        McpProtocolError,
        match=r"resources/list returned 3 items.*max_list_items=2",
    ):
        asyncio.run(
            collect_paginated(
                request,
                "resources/list",
                "resources",
                max_pages=10,
                max_items=2,
            )
        )


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    [
        ("max_pages", True, TypeError),
        ("max_pages", 0, ValueError),
        ("max_items", 1.5, TypeError),
        ("max_items", -1, ValueError),
    ],
)
def test_collect_paginated_rejects_invalid_limits(field, value, error_type) -> None:
    from cayu.mcp._jsonrpc import collect_paginated

    async def request(method, params):
        raise AssertionError("invalid limits must fail before making a request")

    kwargs = {"max_pages": 1, "max_items": 1, field: value}
    with pytest.raises(error_type, match=field):
        asyncio.run(collect_paginated(request, "tools/list", "tools", **kwargs))


def test_stdio_mcp_client_applies_configured_page_limit() -> None:
    spec = McpServerSpec(
        name="local-mcp",
        command=[sys.executable, str(_FAKE_SERVER)],
        env={"CAYU_FAKE_MCP_PAGINATE": "1"},
    )

    async def run() -> None:
        session = await StdioMcpClient(max_list_pages=1).connect(spec)
        try:
            with pytest.raises(McpProtocolError, match=r"tools/list.*max_list_pages=1"):
                await session.list_tools()
        finally:
            await session.close()

    asyncio.run(run())


def test_collect_paginated_treats_blank_cursor_as_end_of_list() -> None:
    from cayu.mcp._jsonrpc import McpPaginatedPage, collect_paginated

    async def request(method, params):
        assert params == {}
        return McpPaginatedPage({"tools": [{"name": "only"}]}, "")

    assert asyncio.run(collect_paginated(request, "tools/list", "tools")) == [{"name": "only"}]


def test_collect_paginated_rejects_non_string_cursor() -> None:
    from cayu.mcp._jsonrpc import McpPaginatedPage, collect_paginated

    async def request(method, params):
        return McpPaginatedPage({"tools": []}, 42)

    with pytest.raises(McpProtocolError, match="nextCursor must be a string"):
        asyncio.run(collect_paginated(request, "tools/list", "tools"))


def _assert_traceback_does_not_retain_text(error: BaseException, text: str) -> None:
    traceback = error.__traceback__
    while traceback is not None:
        if is_cayu_source_filename(traceback.tb_frame.f_code.co_filename):
            retained = {
                name: type(value).__name__
                for name, value in traceback.tb_frame.f_locals.items()
                if text in repr(value)
            }
            assert retained == {}, (
                traceback.tb_frame.f_code.co_filename,
                traceback.tb_frame.f_code.co_name,
                retained,
            )
        traceback = traceback.tb_next


def test_collect_paginated_does_not_retain_private_cursor_in_request_failure() -> None:
    from cayu.mcp._jsonrpc import McpPaginatedPage, collect_paginated

    cursor = "private-cursor-request-failure-canary"
    redactor = SecretRedactor("different-configured-workload-secret")
    calls = 0

    async def request(method, params):
        nonlocal calls
        del method
        calls += 1
        if calls == 1:
            return McpPaginatedPage({"tools": []}, cursor)
        raise RuntimeError(f"server rejected {params['cursor']}")

    with pytest.raises(McpProtocolError) as exc_info:
        asyncio.run(
            collect_paginated(
                request,
                "tools/list",
                "tools",
                redactor=redactor,
            )
        )

    assert cursor not in str(exc_info.value)
    assert REDACTED_SECRET in str(exc_info.value)
    _assert_traceback_does_not_retain_text(exc_info.value, cursor)


def test_collect_paginated_does_not_retain_private_cursor_on_later_page_validation() -> None:
    from cayu.mcp._jsonrpc import McpPaginatedPage, collect_paginated

    cursor = "private-cursor-later-validation-canary"
    calls = 0

    async def request(method, params):
        nonlocal calls
        del method
        calls += 1
        if calls == 1:
            return McpPaginatedPage({"tools": []}, cursor)
        assert params["cursor"] == cursor
        return McpPaginatedPage("invalid-result")

    with pytest.raises(McpProtocolError, match="result must be an object") as exc_info:
        asyncio.run(
            collect_paginated(
                request,
                "tools/list",
                "tools",
                redactor=SecretRedactor(cursor),
            )
        )

    _assert_traceback_does_not_retain_text(exc_info.value, cursor)


def test_collect_paginated_contains_hostile_exception_rendering() -> None:
    from cayu.mcp._jsonrpc import McpPaginatedPage, collect_paginated

    cursor = "private-cursor-hostile-rendering-canary"
    calls = 0

    class HostileRenderingError(RuntimeError):
        def __str__(self) -> str:
            raise KeyboardInterrupt(f"hostile renderer retained {cursor}")

    async def request(method, params):
        nonlocal calls
        del method
        calls += 1
        if calls == 1:
            return McpPaginatedPage({"tools": []}, cursor)
        assert params["cursor"] == cursor
        raise HostileRenderingError("private request failure")

    with pytest.raises(McpProtocolError, match="paginated request failed") as exc_info:
        asyncio.run(
            collect_paginated(
                request,
                "tools/list",
                "tools",
                redactor=SecretRedactor(cursor),
            )
        )

    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    _assert_traceback_does_not_retain_text(exc_info.value, cursor)


def test_collect_paginated_detaches_nested_group_with_private_cursor() -> None:
    from cayu.mcp._jsonrpc import McpPaginatedPage, collect_paginated

    cursor = "private-cursor-group-canary"
    calls = 0

    async def request(method, params):
        nonlocal calls
        del method
        calls += 1
        if calls == 1:
            return McpPaginatedPage({"tools": []}, cursor)
        assert params["cursor"] == cursor
        raise BaseExceptionGroup(
            f"outer group retained {cursor}",
            [
                BaseExceptionGroup(
                    f"inner group retained {cursor}",
                    [
                        asyncio.CancelledError(f"cancelled with {cursor}"),
                        RuntimeError(f"failed with {cursor}"),
                    ],
                )
            ],
        )

    with pytest.raises(BaseExceptionGroup) as exc_info:
        asyncio.run(
            collect_paginated(
                request,
                "tools/list",
                "tools",
                redactor=SecretRedactor("different-configured-workload-secret"),
            )
        )

    error = exc_info.value
    assert cursor not in repr(error)
    assert REDACTED_SECRET in repr(error)
    assert isinstance(error.exceptions[0], BaseExceptionGroup)
    nested = error.exceptions[0]
    assert isinstance(nested.exceptions[0], asyncio.CancelledError)
    assert isinstance(nested.exceptions[1], McpProtocolError)
    assert error.__cause__ is None
    assert error.__context__ is None
    _assert_traceback_does_not_retain_text(error, cursor)


@pytest.mark.parametrize("fatal_type", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_collect_paginated_detaches_scalar_fatal_signal_with_private_cursor(
    fatal_type: type[BaseException],
) -> None:
    from cayu.mcp._jsonrpc import McpPaginatedPage, collect_paginated

    cursor = f"private-cursor-{fatal_type.__name__}-canary"
    calls = 0

    async def request(method, params):
        nonlocal calls
        del method
        calls += 1
        if calls == 1:
            return McpPaginatedPage({"tools": []}, cursor)
        assert params["cursor"] == cursor
        raise fatal_type(f"fatal signal retained {params['cursor']}")

    async def scenario() -> BaseException:
        with pytest.raises(fatal_type) as exc_info:
            await collect_paginated(
                request,
                "tools/list",
                "tools",
                redactor=SecretRedactor("different-configured-workload-secret"),
            )
        return exc_info.value

    error = asyncio.run(scenario())
    assert cursor not in repr(error)
    assert REDACTED_SECRET in repr(error)
    assert error.__cause__ is None
    assert error.__context__ is None
    _assert_traceback_does_not_retain_text(error, cursor)


@pytest.mark.parametrize("historical_cancellation", [False, True])
def test_collect_paginated_classifies_callback_cancellation_as_failure(
    historical_cancellation: bool,
) -> None:
    from cayu.mcp._jsonrpc import McpPaginatedPage, collect_paginated

    cursor = "private-cursor-internal-cancellation-canary"
    calls = 0

    async def request(method, params):
        nonlocal calls
        del method
        calls += 1
        if calls == 1:
            return McpPaginatedPage({"resources": []}, cursor)
        assert params["cursor"] == cursor
        raise asyncio.CancelledError(f"callback cancelled with {cursor}")

    async def scenario() -> tuple[BaseException, int]:
        current = asyncio.current_task()
        assert current is not None
        if historical_cancellation:
            current.cancel("already delivered pagination cancellation")
            with suppress(asyncio.CancelledError):
                await asyncio.sleep(0)
        with pytest.raises(McpProtocolError, match="cancelled unexpectedly") as exc_info:
            await collect_paginated(
                request,
                "resources/list",
                "resources",
                redactor=SecretRedactor("different-configured-workload-secret"),
            )
        return exc_info.value, current.cancelling()

    error, cancelling = asyncio.run(scenario())

    assert cancelling == int(historical_cancellation)
    assert cursor not in "".join(traceback.format_exception(error))
    assert REDACTED_SECRET in str(error)


def test_collect_paginated_preserves_cancellation_pending_before_request() -> None:
    from cayu.mcp._jsonrpc import McpPaginatedPage, collect_paginated

    request_calls = 0

    async def request(method, params):
        nonlocal request_calls
        del method, params
        request_calls += 1
        return McpPaginatedPage({"resources": []})

    async def cancelled_operation() -> None:
        current = asyncio.current_task()
        assert current is not None
        current.cancel("pagination cancellation pending at request boundary")
        await collect_paginated(
            request,
            "resources/list",
            "resources",
        )

    async def scenario() -> tuple[int, bool, asyncio.CancelledError]:
        task = asyncio.create_task(cancelled_operation())
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        return task.cancelling(), task.cancelled(), exc_info.value

    cancelling, cancelled, error = asyncio.run(scenario())

    assert cancelling == 1
    assert cancelled is True
    assert error.args == ("pagination cancellation pending at request boundary",)
    assert request_calls == 0


def test_collect_paginated_does_not_retain_private_cursor_on_real_cancellation() -> None:
    from cayu.mcp._jsonrpc import McpPaginatedPage, collect_paginated

    cursor = "private-cursor-cancellation-canary"

    async def scenario() -> tuple[asyncio.CancelledError, int, bool]:
        second_request_started = asyncio.Event()
        calls = 0

        async def request(method, params):
            nonlocal calls
            del method
            calls += 1
            if calls == 1:
                return McpPaginatedPage({"resources": []}, cursor)
            assert params["cursor"] == cursor
            second_request_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise asyncio.CancelledError(
                    f"transport cancelled while sending {params['cursor']}"
                ) from None

        task = asyncio.create_task(
            collect_paginated(
                request,
                "resources/list",
                "resources",
                redactor=SecretRedactor("different-configured-workload-secret"),
            )
        )
        await second_request_started.wait()
        task.cancel("caller cancelled")
        cancelling = task.cancelling()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        return exc_info.value, cancelling, task.cancelled()

    cancellation, cancelling, cancelled = asyncio.run(scenario())

    assert cancelling == 1
    assert cancelled is True
    assert cancellation.args == (f"transport cancelled while sending {REDACTED_SECRET}",)
    _assert_traceback_does_not_retain_text(cancellation, cursor)


def test_collect_paginated_redacts_numeric_secret_from_real_cancellation(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    from cayu.mcp._jsonrpc import McpPaginatedPage, collect_paginated

    secret = "4829017351642089"

    async def scenario() -> tuple[asyncio.CancelledError, int, bool]:
        second_request_started = asyncio.Event()
        calls = 0

        async def request(method, params):
            nonlocal calls
            del method
            calls += 1
            if calls == 1:
                return McpPaginatedPage({"resources": []}, "next-page")
            assert params["cursor"] == "next-page"
            second_request_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        task = asyncio.create_task(
            collect_paginated(
                request,
                "resources/list",
                "resources",
                redactor=SecretRedactor(secret),
            )
        )
        await second_request_started.wait()
        task.cancel(int(secret))
        cancelling = task.cancelling()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        return exc_info.value, cancelling, task.cancelled()

    with caplog.at_level(logging.DEBUG):
        cancellation, cancelling, cancelled = asyncio.run(scenario())

    assert cancelling == 1
    assert cancelled is True
    assert cancellation.args == (REDACTED_SECRET,)
    assert secret not in repr(cancellation)
    assert secret not in "".join(traceback.format_exception(cancellation))
    _assert_traceback_does_not_retain_text(cancellation, secret)
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in record.getMessage() for record in caplog.records)
    assert all(secret not in str(warning.message) for warning in recwarn)


def test_stdio_mcp_client_times_out_blocked_initialized_notification_write() -> None:
    async def run():
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(_FAKE_SERVER),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        session = StdioMcpSession(
            server=_fake_server_spec(),
            process=process,
            request_timeout_s=1.0,
            write_timeout_s=0.01,
            graceful_shutdown_timeout_s=0.01,
            cancellation_notification_timeout_s=0.01,
            client_name="cayu",
            client_version="0.1.0",
        )
        original_write = session._write
        notification_write_started = asyncio.Event()

        async def block_initialized_notification(payload: dict[str, Any]) -> None:
            if payload.get("method") == "notifications/initialized":
                notification_write_started.set()
                await asyncio.Event().wait()
            await original_write(payload)

        session._write = block_initialized_notification
        try:
            with pytest.raises(TimeoutError, match="notifications/initialized write timed out"):
                await session.initialize()
            await asyncio.wait_for(session.process.wait(), timeout=1)
            return notification_write_started.is_set(), session.process.returncode
        finally:
            await session.close()

    notification_write_started, returncode = asyncio.run(run())

    assert notification_write_started is True
    assert returncode is not None


def test_stdio_mcp_session_close_uses_graceful_stdin_eof_before_terminate() -> None:
    async def run():
        client = StdioMcpClient()
        session = await client.connect(_fake_server_spec())
        assert isinstance(session, StdioMcpSession)
        await session.close()
        return session.process.returncode

    assert asyncio.run(run()) == 0


def test_stdio_mcp_session_close_finishes_cleanup_when_cancelled() -> None:
    async def run():
        client = StdioMcpClient()
        session = await client.connect(_fake_server_spec())
        assert isinstance(session, StdioMcpSession)
        original_close_stdin = session._close_stdin_for_graceful_shutdown
        close_started = asyncio.Event()

        async def delayed_close_stdin() -> None:
            close_started.set()
            await asyncio.sleep(0.01)
            await original_close_stdin()

        session._close_stdin_for_graceful_shutdown = delayed_close_stdin
        close_task = asyncio.create_task(session.close())
        await close_started.wait()
        close_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await close_task
        return (
            session.process.returncode,
            dict(session._pending),
            session._reader_task.done(),
            session._stderr_task.done(),
        )

    returncode, pending, reader_done, stderr_done = asyncio.run(run())

    assert returncode == 0
    assert pending == {}
    assert reader_done is True
    assert stderr_done is True


def test_stdio_mcp_session_close_preserves_cancellation_when_cleanup_fails() -> None:
    secret = "mcp-stdio-cancelled-close-secret-canary"

    async def run() -> tuple[int, bool, BaseException | None, int | None, bool, bool]:
        session = await StdioMcpClient(graceful_shutdown_timeout_s=0.01).connect(
            _fake_server_spec()
        )
        assert isinstance(session, StdioMcpSession)
        session._secret_redactor = SecretRedactor(secret)
        close_started = asyncio.Event()
        release_close = asyncio.Event()

        async def failing_close_stdin() -> None:
            close_started.set()
            await release_close.wait()
            raise RuntimeError(f"stdio cleanup exposed {secret}")

        session._close_stdin_for_graceful_shutdown = failing_close_stdin
        close_task = asyncio.create_task(session.close())
        await close_started.wait()
        close_task.cancel("cancel stdio MCP close")
        cancelling = close_task.cancelling()
        await asyncio.sleep(0)
        release_close.set()
        with pytest.raises(asyncio.CancelledError, match="cancel stdio MCP close") as exc_info:
            await close_task
        return (
            cancelling,
            close_task.cancelled(),
            exc_info.value.__cause__,
            session.process.returncode,
            session._reader_task.done(),
            session._stderr_task.done(),
        )

    cancelling, cancelled, cleanup_error, returncode, reader_done, stderr_done = asyncio.run(run())

    assert cancelling == 1
    assert cancelled is True
    assert isinstance(cleanup_error, McpProtocolError)
    assert secret not in "".join(traceback.format_exception(cleanup_error))
    assert returncode is not None
    assert reader_done is True
    assert stderr_done is True


def test_stdio_mcp_session_close_concurrent_callers_share_cleanup() -> None:
    async def run():
        client = StdioMcpClient()
        session = await client.connect(_fake_server_spec())
        assert isinstance(session, StdioMcpSession)
        original_close_stdin = session._close_stdin_for_graceful_shutdown
        close_started = asyncio.Event()
        allow_close = asyncio.Event()
        second_close_returned = False

        async def delayed_close_stdin() -> None:
            close_started.set()
            await allow_close.wait()
            await original_close_stdin()

        async def second_close() -> None:
            nonlocal second_close_returned
            await session.close()
            second_close_returned = True

        session._close_stdin_for_graceful_shutdown = delayed_close_stdin
        first = asyncio.create_task(session.close())
        await close_started.wait()
        second = asyncio.create_task(second_close())
        await asyncio.sleep(0)
        returned_while_first_in_progress = second_close_returned
        allow_close.set()
        await asyncio.gather(first, second)
        return returned_while_first_in_progress, second_close_returned, session.process.returncode

    returned_while_first_in_progress, second_close_returned, returncode = asyncio.run(run())

    assert returned_while_first_in_progress is False
    assert second_close_returned is True
    assert returncode == 0


def test_mcp_toolset_connect_classifies_internal_discovery_cancellation_as_failure() -> None:
    async def run():
        session = FakeMcpSession(list_tools_error=asyncio.CancelledError())
        with pytest.raises(McpProtocolError, match="cancelled unexpectedly"):
            await connect_mcp_toolset(_fake_server_spec(), client=FakeMcpClient(session))
        await asyncio.sleep(0)
        current = asyncio.current_task()
        assert current is not None
        return session.closed, current.cancelling()

    closed, cancelling = asyncio.run(run())

    assert closed is True
    assert cancelling == 0


def test_mcp_toolset_grouped_discovery_failure_closes_and_detaches_extension_session() -> None:
    secret = "mcp-grouped-discovery-secret-canary"
    raw_failure = BaseExceptionGroup(
        f"grouped discovery exposed {secret}",
        [
            asyncio.CancelledError(f"discovery cancelled with {secret}"),
            RuntimeError(f"discovery failed with {secret}"),
        ],
    )

    async def run() -> tuple[BaseExceptionGroup, bool, int]:
        session = FakeMcpSession(list_tools_error=raw_failure)
        session._secret_redactor = SecretRedactor(secret)
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await connect_mcp_toolset(_fake_server_spec(), client=FakeMcpClient(session))
        current = asyncio.current_task()
        assert current is not None
        return exc_info.value, session.closed, current.cancelling()

    error, closed, cancelling = asyncio.run(run())

    assert error is not raw_failure
    assert error.__context__ is None
    assert "CancelledError" in str(error.exceptions[0])
    assert "RuntimeError" in str(error.exceptions[1])
    assert secret not in "".join(traceback.format_exception(error))
    assert closed is True
    assert cancelling == 0


@pytest.mark.parametrize("fatal_type", [SystemExit, GeneratorExit])
@pytest.mark.parametrize("cleanup_error_type", [None, RuntimeError, SystemExit, KeyboardInterrupt])
def test_mcp_toolset_scalar_fatal_discovery_failure_finishes_extension_cleanup(
    fatal_type: type[BaseException],
    cleanup_error_type: type[BaseException] | None,
) -> None:
    secret = f"mcp-{fatal_type.__name__}-discovery-secret-canary"
    raw_failure = fatal_type(f"discovery exposed {secret}")
    cleanup_error = (
        cleanup_error_type(f"cleanup exposed {secret}") if cleanup_error_type is not None else None
    )

    async def run() -> tuple[BaseException, bool, int]:
        session = FakeMcpSession(
            list_tools_error=raw_failure,
            close_error=cleanup_error,
        )
        session._secret_redactor = SecretRedactor(secret)
        with pytest.raises(fatal_type) as exc_info:
            await connect_mcp_toolset(_fake_server_spec(), client=FakeMcpClient(session))
        current = asyncio.current_task()
        assert current is not None
        return exc_info.value, session.closed, current.cancelling()

    error, closed, cancelling = asyncio.run(run())

    assert type(error) is fatal_type
    assert error is not raw_failure
    assert closed is True
    assert cancelling == 0
    assert secret not in "".join(traceback.format_exception(error))
    assert REDACTED_SECRET in repr(error)
    assert error.__context__ is None
    if cleanup_error_type is not None:
        assert isinstance(error.__cause__, McpProtocolError)
        assert REDACTED_SECRET in str(error.__cause__)
    else:
        assert error.__cause__ is None
    _assert_traceback_does_not_retain_text(error, secret)


def test_mcp_toolset_scalar_fatal_discovery_failure_bounds_hostile_diagnostics(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    secret = "mcp-fatal-hostile-diagnostic-secret-canary"
    render_calls: list[str] = []

    class HostileDiagnostic:
        def __str__(self) -> str:
            render_calls.append("str")
            return secret

        def __repr__(self) -> str:
            render_calls.append("repr")
            return secret

    raw_failure = SystemExit(HostileDiagnostic(), secret * 10_000)

    async def run() -> tuple[SystemExit, bool]:
        session = FakeMcpSession(list_tools_error=raw_failure)
        session._secret_redactor = SecretRedactor(secret)
        with pytest.raises(SystemExit) as exc_info:
            await connect_mcp_toolset(_fake_server_spec(), client=FakeMcpClient(session))
        return exc_info.value, session.closed

    with caplog.at_level(logging.DEBUG):
        error, closed = asyncio.run(run())

    assert closed is True
    assert render_calls == []
    assert len(repr(error)) < 5_000
    assert secret not in "".join(traceback.format_exception(error))
    assert error.__context__ is None
    _assert_traceback_does_not_retain_text(error, secret)
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in record.getMessage() for record in caplog.records)
    assert all(secret not in str(warning.message) for warning in recwarn)


@pytest.mark.parametrize("safe_code", ["safe exit code", ""])
def test_mcp_toolset_scalar_system_exit_preserves_safe_code(safe_code: str) -> None:
    raw_failure = SystemExit(safe_code)

    async def run() -> tuple[SystemExit, bool]:
        session = FakeMcpSession(list_tools_error=raw_failure)
        with pytest.raises(SystemExit) as exc_info:
            await connect_mcp_toolset(_fake_server_spec(), client=FakeMcpClient(session))
        return exc_info.value, session.closed

    error, closed = asyncio.run(run())

    assert closed is True
    assert error is not raw_failure
    assert error.args == (safe_code,)
    assert error.code == safe_code
    assert error.__cause__ is None
    assert error.__context__ is None


def test_mcp_toolset_scalar_fatal_cleanup_preserves_real_caller_cancellation() -> None:
    secret = "mcp-fatal-cleanup-cancellation-secret-canary"

    class BlockingFailingCloseSession(FakeMcpSession):
        def __init__(self) -> None:
            super().__init__(list_tools_error=SystemExit(f"discovery exposed {secret}"))
            self._secret_redactor = SecretRedactor(secret)
            self.close_started = asyncio.Event()
            self.release_close = asyncio.Event()

        async def close(self) -> None:
            self.close_started.set()
            await self.release_close.wait()
            self.closed = True
            raise RuntimeError(f"cleanup exposed {secret}")

    async def run() -> tuple[int, bool, asyncio.CancelledError]:
        session = BlockingFailingCloseSession()
        task = asyncio.create_task(
            connect_mcp_toolset(_fake_server_spec(), client=FakeMcpClient(session))
        )
        await session.close_started.wait()
        task.cancel(f"cancel cleanup {secret}")
        cancelling = task.cancelling()
        await asyncio.sleep(0)
        session.release_close.set()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        return cancelling, task.cancelled(), exc_info.value

    cancelling, cancelled, error = asyncio.run(run())

    assert cancelling == 1
    assert cancelled is True
    assert error.__context__ is None
    assert isinstance(error.__cause__, BaseExceptionGroup)
    assert len(error.__cause__.exceptions) == 2
    assert all(isinstance(item, McpProtocolError) for item in error.__cause__.exceptions)
    assert secret not in "".join(traceback.format_exception(error))
    assert REDACTED_SECRET in "".join(traceback.format_exception(error.__cause__))


def test_mcp_toolset_discovery_rejects_unauthenticated_failure_handoffs(
    caplog,
    capsys,
) -> None:
    import logging
    import warnings

    from cayu.mcp._exception_handoffs import (
        MCP_HTTP_SETTLEMENT_TASK_ATTRIBUTE,
        MCP_SESSION_CLOSE_TASK_ATTRIBUTE,
    )

    secret = "forged-mcp-failure-handoff-secret-canary"
    render_calls: list[str] = []

    class SecretBearingHandoff:
        def __str__(self) -> str:
            render_calls.append("str")
            return secret

        def __repr__(self) -> str:
            render_calls.append("repr")
            return secret

    raw_failure = RuntimeError("extension discovery failed")
    raw_namespace = BaseException.__dict__["__dict__"].__get__(raw_failure, BaseException)
    raw_namespace[MCP_SESSION_CLOSE_TASK_ATTRIBUTE] = SecretBearingHandoff()
    raw_namespace[MCP_HTTP_SETTLEMENT_TASK_ATTRIBUTE] = SecretBearingHandoff()

    async def run() -> tuple[BaseException, bool]:
        session = FakeMcpSession(list_tools_error=raw_failure)
        session._secret_redactor = SecretRedactor(secret)
        with pytest.raises(McpProtocolError) as exc_info:
            await connect_mcp_toolset(_fake_server_spec(), client=FakeMcpClient(session))
        return exc_info.value, session.closed

    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        with caplog.at_level(logging.WARNING):
            error, closed = asyncio.run(run())

    public_namespace = BaseException.__dict__["__dict__"].__get__(error, BaseException)
    assert MCP_SESSION_CLOSE_TASK_ATTRIBUTE not in public_namespace
    assert MCP_HTTP_SETTLEMENT_TASK_ATTRIBUTE not in public_namespace
    assert secret not in "".join(traceback.format_exception(error))
    assert all(secret not in record.getMessage() for record in caplog.records)
    assert all(secret not in str(warning.message) for warning in captured_warnings)
    captured_output = capsys.readouterr()
    assert secret not in captured_output.out
    assert secret not in captured_output.err
    _assert_traceback_does_not_retain_text(error, secret)
    assert render_calls == []
    assert closed is True


def test_mcp_toolset_discovery_timeout_does_not_wait_for_retained_close() -> None:
    class BlockingCloseSession(FakeMcpSession):
        def __init__(self) -> None:
            super().__init__(list_tools_error=McpIdleTimeoutError("discovery idle timeout"))
            self.close_started = asyncio.Event()
            self.close_finished = asyncio.Event()
            self.release_close = asyncio.Event()

        def _fence_before_retained_close(self) -> bool:
            self.closed = True
            return True

        async def close(self) -> None:
            self.close_started.set()
            await self.release_close.wait()
            self.close_finished.set()

    async def run() -> tuple[float, bool]:
        session = BlockingCloseSession()
        started_at = asyncio.get_running_loop().time()
        with pytest.raises(McpIdleTimeoutError, match="idle timeout"):
            await connect_mcp_toolset(_fake_server_spec(), client=FakeMcpClient(session))
        elapsed = asyncio.get_running_loop().time() - started_at
        await asyncio.wait_for(session.close_started.wait(), timeout=0.1)
        close_was_pending = not session.close_finished.is_set()
        session.release_close.set()
        await asyncio.wait_for(session.close_finished.wait(), timeout=0.1)
        return elapsed, close_was_pending

    elapsed, close_was_pending = asyncio.run(run())

    assert elapsed < 0.05
    assert close_was_pending is True


def test_mcp_toolset_fencing_hook_cancellation_fails_closed_without_reclassification() -> None:
    class CancellingFenceSession(FakeMcpSession):
        def __init__(self) -> None:
            super().__init__(list_tools_error=McpIdleTimeoutError("discovery idle timeout"))

        def _fence_before_retained_close(self) -> bool:
            raise asyncio.CancelledError("extension hook cancelled internally")

    async def run() -> tuple[bool, int]:
        session = CancellingFenceSession()
        with pytest.raises(McpIdleTimeoutError, match="idle timeout"):
            await connect_mcp_toolset(_fake_server_spec(), client=FakeMcpClient(session))
        current = asyncio.current_task()
        assert current is not None
        return session.closed, current.cancelling()

    closed, cancelling = asyncio.run(run())

    assert closed is True
    assert cancelling == 0


@pytest.mark.parametrize("fatal_type", [SystemExit, GeneratorExit])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_mcp_toolset_fencing_hook_fatal_signal_fails_closed_and_finishes_cleanup(
    fatal_type: type[BaseException],
    cleanup_fails: bool,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    secret = "mcp-fencing-hook-fatal-secret-canary"

    class FatalFenceSession(FakeMcpSession):
        def __init__(self) -> None:
            super().__init__(
                list_tools_error=McpIdleTimeoutError("discovery idle timeout"),
                close_error=(
                    RuntimeError(f"discovery cleanup exposed {secret}") if cleanup_fails else None
                ),
            )
            self._secret_redactor = SecretRedactor(secret)

        def _fence_before_retained_close(self) -> bool:
            raise fatal_type(f"fencing probe exposed {secret}")

    async def run() -> tuple[BaseException, bool]:
        session = FatalFenceSession()
        with pytest.raises(McpIdleTimeoutError, match="idle timeout") as exc_info:
            await connect_mcp_toolset(_fake_server_spec(), client=FakeMcpClient(session))
        return exc_info.value, session.closed

    with caplog.at_level(logging.DEBUG):
        error, closed = asyncio.run(run())

    assert closed is True
    assert secret not in "".join(traceback.format_exception(error))
    if cleanup_fails:
        assert isinstance(error.__cause__, McpProtocolError)
        assert REDACTED_SECRET in str(error.__cause__)
    else:
        assert error.__cause__ is None
    _assert_traceback_does_not_retain_text(error, secret)
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in record.getMessage() for record in caplog.records)
    assert all(secret not in str(warning.message) for warning in recwarn)


def test_mcp_toolset_retained_close_failure_is_attached_and_redacted() -> None:
    secret = "mcp-retained-close-secret-canary"

    class FailingRetainedCloseSession(FakeMcpSession):
        def __init__(self) -> None:
            super().__init__(list_tools_error=McpIdleTimeoutError("discovery idle timeout"))
            self._secret_redactor = SecretRedactor(secret)
            self.close_started = asyncio.Event()
            self.release_close = asyncio.Event()

        def _fence_before_retained_close(self) -> bool:
            self.closed = True
            return True

        async def close(self) -> None:
            self.close_started.set()
            await self.release_close.wait()
            raise RuntimeError(f"retained cleanup exposed {secret}")

    async def run() -> tuple[BaseException, BaseException]:
        session = FailingRetainedCloseSession()
        with pytest.raises(McpIdleTimeoutError) as exc_info:
            await connect_mcp_toolset(_fake_server_spec(), client=FakeMcpClient(session))
        close_task = _mcp_session_close_task(exc_info.value)
        assert close_task is not None
        await session.close_started.wait()
        session.release_close.set()
        with pytest.raises(McpProtocolError) as close_exc_info:
            await close_task
        return exc_info.value, close_exc_info.value

    primary_error, close_error = asyncio.run(run())

    assert secret not in "".join(traceback.format_exception(primary_error))
    assert secret not in "".join(traceback.format_exception(close_error))
    assert REDACTED_SECRET in str(close_error)


def test_retained_close_terminal_observer_consumes_mixed_failure_group() -> None:
    raw_close_failure = BaseExceptionGroup(
        "mixed retained close failure",
        [
            asyncio.CancelledError("historical close cancellation"),
            RuntimeError("ordinary close failure"),
        ],
    )

    async def run() -> tuple[BaseException, list[dict[str, Any]]]:
        session = FakeMcpSession(close_error=raw_close_failure)
        loop = asyncio.get_running_loop()
        diagnostics: list[dict[str, Any]] = []
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: diagnostics.append(context))
        try:
            primary_error = McpIdleTimeoutError("discovery idle timeout")
            close_task = _retain_mcp_session_close(
                session,
                primary_error=primary_error,
            )
            outcome = (await asyncio.gather(close_task, return_exceptions=True))[0]
            await asyncio.sleep(0)
            return outcome, diagnostics
        finally:
            loop.set_exception_handler(previous_handler)

    outcome, diagnostics = asyncio.run(run())

    assert isinstance(outcome, BaseExceptionGroup)
    assert diagnostics == []


def test_mcp_toolset_discovery_timeout_waits_for_unfenced_extension_close() -> None:
    definition = McpToolDefinition(name="echo", input_schema={"type": "object"})

    class ReusedSession(FakeMcpSession):
        def __init__(self) -> None:
            super().__init__(definitions=(definition,))
            self.list_attempts = 0
            self.close_started = asyncio.Event()
            self.release_close = asyncio.Event()

        async def list_tools(self) -> tuple[McpToolDefinition, ...]:
            self.list_attempts += 1
            if self.closed:
                raise McpProtocolError("extension session is closed")
            if self.list_attempts == 1:
                raise McpIdleTimeoutError("discovery idle timeout")
            return self.definitions

        async def close(self) -> None:
            self.close_started.set()
            await self.release_close.wait()
            self.closed = True

    async def run() -> tuple[bool, bool, int]:
        session = ReusedSession()
        client = FakeMcpClient(session)
        first_attempt = asyncio.create_task(connect_mcp_toolset(_fake_server_spec(), client=client))
        await asyncio.wait_for(session.close_started.wait(), timeout=0.1)
        returned_before_close = first_attempt.done()
        session.release_close.set()
        with pytest.raises(McpIdleTimeoutError, match="idle timeout"):
            await first_attempt
        with pytest.raises(McpProtocolError, match="session is closed"):
            await connect_mcp_toolset(_fake_server_spec(), client=client)
        return returned_before_close, session.closed, session.list_attempts

    returned_before_close, closed, list_attempts = asyncio.run(run())

    assert returned_before_close is False
    assert closed is True
    assert list_attempts == 2


def test_mcp_toolset_builtin_subclasses_must_explicitly_prove_discovery_fencing() -> None:
    initialize_result = McpInitializeResult(protocol_version=MCP_PROTOCOL_VERSION)

    class ExtendedHttpSession(HttpMcpSession):
        def __init__(self) -> None:
            self.close_started = asyncio.Event()
            self.release_close = asyncio.Event()
            self.close_finished = False

        @property
        def initialize_result(self) -> McpInitializeResult:
            return initialize_result

        async def list_tools(self) -> tuple[McpToolDefinition, ...]:
            raise McpIdleTimeoutError("extended HTTP discovery timed out")

        async def close(self) -> None:
            self.close_started.set()
            await self.release_close.wait()
            self.close_finished = True

    class ExtendedStdioSession(StdioMcpSession):
        def __init__(self) -> None:
            self.close_started = asyncio.Event()
            self.release_close = asyncio.Event()
            self.close_finished = False

        @property
        def initialize_result(self) -> McpInitializeResult:
            return initialize_result

        async def list_tools(self) -> tuple[McpToolDefinition, ...]:
            raise McpIdleTimeoutError("extended stdio discovery timed out")

        async def close(self) -> None:
            self.close_started.set()
            await self.release_close.wait()
            self.close_finished = True

    async def exercise(session: ExtendedHttpSession | ExtendedStdioSession) -> None:
        task = asyncio.create_task(
            connect_mcp_toolset(_fake_server_spec(), client=FakeMcpClient(session))
        )
        await asyncio.wait_for(session.close_started.wait(), timeout=0.1)
        assert session._closed is True
        assert task.done() is False
        session.release_close.set()
        with pytest.raises(McpIdleTimeoutError, match="discovery timed out"):
            await task
        assert session.close_finished is True

    async def run() -> None:
        await exercise(ExtendedHttpSession())
        await exercise(ExtendedStdioSession())

    asyncio.run(run())


@pytest.mark.parametrize("cancel_discovery", [False, True])
def test_mcp_toolset_descendant_cannot_inherit_discovery_fencing_proof(
    cancel_discovery: bool,
) -> None:
    class OptedInSession(FakeMcpSession):
        def __init__(self) -> None:
            super().__init__()
            self.discovery_started = asyncio.Event()
            self.close_started = asyncio.Event()
            self.release_close = asyncio.Event()
            self.close_finished = False

        async def list_tools(self) -> tuple[McpToolDefinition, ...]:
            self.discovery_started.set()
            if cancel_discovery:
                await asyncio.Event().wait()
            raise McpIdleTimeoutError("descendant discovery timed out")

        def _fence_before_retained_close(self) -> bool:
            self.closed = True
            return True

        async def close(self) -> None:
            self.close_started.set()
            await self.release_close.wait()
            self.close_finished = True

    class DescendantSession(OptedInSession):
        pass

    async def run() -> tuple[int, bool]:
        session = DescendantSession()
        task = asyncio.create_task(
            connect_mcp_toolset(_fake_server_spec(), client=FakeMcpClient(session))
        )
        await asyncio.wait_for(session.discovery_started.wait(), timeout=0.1)
        cancelling = 0
        if cancel_discovery:
            task.cancel("cancel descendant discovery")
            cancelling = task.cancelling()
        await asyncio.wait_for(session.close_started.wait(), timeout=0.1)
        assert task.done() is False
        session.release_close.set()
        if cancel_discovery:
            with pytest.raises(asyncio.CancelledError, match="cancel descendant discovery"):
                await task
            assert task.cancelled() is True
        else:
            with pytest.raises(McpIdleTimeoutError, match="descendant discovery timed out"):
                await task
        return cancelling, session.close_finished

    cancelling, close_finished = asyncio.run(run())

    assert cancelling == int(cancel_discovery)
    assert close_finished is True


def test_mcp_toolset_real_cancellation_does_not_wait_for_retained_close() -> None:
    class BlockingDiscoverySession(FakeMcpSession):
        def __init__(self) -> None:
            super().__init__()
            self.discovery_started = asyncio.Event()
            self.close_started = asyncio.Event()
            self.close_finished = asyncio.Event()
            self.release_close = asyncio.Event()

        async def list_tools(self) -> tuple[McpToolDefinition, ...]:
            self.discovery_started.set()
            await asyncio.Event().wait()
            return ()

        def _fence_before_retained_close(self) -> bool:
            self.closed = True
            return True

        async def close(self) -> None:
            self.close_started.set()
            await self.release_close.wait()
            self.close_finished.set()

    async def run() -> tuple[int, bool, bool]:
        session = BlockingDiscoverySession()
        task = asyncio.create_task(
            connect_mcp_toolset(_fake_server_spec(), client=FakeMcpClient(session))
        )
        await session.discovery_started.wait()
        task.cancel("cancel discovery")
        cancelling = task.cancelling()
        done, _ = await asyncio.wait({task}, timeout=0.05)
        returned_before_close = task in done and not session.close_finished.is_set()
        await asyncio.wait_for(session.close_started.wait(), timeout=0.1)
        session.release_close.set()
        await asyncio.wait_for(session.close_finished.wait(), timeout=0.1)
        with pytest.raises(asyncio.CancelledError, match="cancel discovery"):
            await task
        return cancelling, task.cancelled(), returned_before_close

    cancelling, cancelled, returned_before_close = asyncio.run(run())

    assert cancelling == 1
    assert cancelled is True
    assert returned_before_close is True


def test_mcp_toolset_connect_closes_session_when_adapter_construction_fails() -> None:
    async def run():
        definition = McpToolDefinition(name="echo", input_schema={"type": "object"})
        session = FakeMcpSession(definitions=(definition, definition))
        with pytest.raises(ValueError, match="duplicate"):
            await connect_mcp_toolset(_fake_server_spec(), client=FakeMcpClient(session))
        return session.closed

    assert asyncio.run(run()) is True


def test_mcp_toolset_connect_preserves_original_error_when_cleanup_is_cancelled() -> None:
    async def run():
        session = FakeMcpSession(
            list_tools_error=RuntimeError("discovery failed"),
            close_error=asyncio.CancelledError(),
        )
        with pytest.raises(RuntimeError, match="discovery failed"):
            await connect_mcp_toolset(_fake_server_spec(), client=FakeMcpClient(session))
        return session.closed

    assert asyncio.run(run()) is True


def test_mcp_toolset_connect_attaches_redacted_cleanup_failure() -> None:
    secret = "mcp-discovery-close-secret-canary"

    async def run() -> BaseException:
        session = FakeMcpSession(
            list_tools_error=McpProtocolError("discovery failed"),
            close_error=RuntimeError(f"discovery cleanup exposed {secret}"),
        )
        session._secret_redactor = SecretRedactor(secret)
        with pytest.raises(McpProtocolError, match="discovery failed") as exc_info:
            await connect_mcp_toolset(_fake_server_spec(), client=FakeMcpClient(session))
        return exc_info.value

    error = asyncio.run(run())

    assert isinstance(error.__cause__, McpProtocolError)
    assert secret not in "".join(traceback.format_exception(error))
    assert REDACTED_SECRET in str(error.__cause__)


def test_mcp_toolset_cleanup_cancellation_detaches_prior_failures() -> None:
    secret = "mcp-discovery-cleanup-cancellation-secret-canary"

    class BlockingFailingCloseSession(FakeMcpSession):
        def __init__(self) -> None:
            super().__init__(list_tools_error=McpProtocolError(f"discovery exposed {secret}"))
            self._secret_redactor = SecretRedactor(secret)
            self.close_started = asyncio.Event()
            self.release_close = asyncio.Event()

        async def close(self) -> None:
            self.close_started.set()
            await self.release_close.wait()
            raise RuntimeError(f"cleanup exposed {secret}")

    async def run() -> tuple[int, bool, asyncio.CancelledError]:
        session = BlockingFailingCloseSession()
        task = asyncio.create_task(
            connect_mcp_toolset(_fake_server_spec(), client=FakeMcpClient(session))
        )
        await session.close_started.wait()
        task.cancel(f"cancel cleanup {secret}")
        cancelling = task.cancelling()
        await asyncio.sleep(0)
        session.release_close.set()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        return cancelling, task.cancelled(), exc_info.value

    cancelling, cancelled, error = asyncio.run(run())

    assert cancelling == 1
    assert cancelled is True
    assert error.__context__ is None
    assert isinstance(error.__cause__, BaseExceptionGroup)
    assert secret not in "".join(traceback.format_exception(error))
    assert REDACTED_SECRET in "".join(traceback.format_exception(error.__cause__))


def test_mcp_toolset_connect_redacts_discovery_failure_before_propagation() -> None:
    secret = "mcp-discovery-failure-canary"

    async def run():
        session = FakeMcpSession(list_tools_error=RuntimeError(f"discovery failed {secret}"))
        session._secret_redactor = SecretRedactor(secret)
        with pytest.raises(McpProtocolError) as excinfo:
            await connect_mcp_toolset(_fake_server_spec(), client=FakeMcpClient(session))
        return session.closed, excinfo.value

    closed, error = asyncio.run(run())

    assert closed is True
    assert secret not in str(error)
    assert REDACTED_SECRET in str(error)
    assert error.__cause__ is None
    assert error.__context__ is None


def test_mcp_toolset_connect_detaches_safe_discovery_error_from_secret_definitions() -> None:
    secret = "mcp-discovery-definition-secret-canary"
    definition = McpToolDefinition(
        name="duplicate",
        description=f"private description {secret}",
        input_schema={"type": "object"},
    )

    async def run():
        session = FakeMcpSession(definitions=(definition, definition))
        session._secret_redactor = SecretRedactor(secret)
        with pytest.raises(McpProtocolError, match="duplicate") as excinfo:
            await connect_mcp_toolset(_fake_server_spec(), client=FakeMcpClient(session))
        return session.closed, excinfo.value

    closed, error = asyncio.run(run())

    assert closed is True
    _assert_mcp_traceback_does_not_retain_secret(error, secret)


def test_mcp_toolset_internal_cancellation_preserves_redacted_cleanup_failure() -> None:
    secret = "mcp-cancel-cleanup-canary"

    async def run():
        session = FakeMcpSession(
            list_tools_error=asyncio.CancelledError(),
            close_error=RuntimeError(f"cleanup failed {secret}"),
        )
        session._secret_redactor = SecretRedactor(secret)
        with pytest.raises(McpProtocolError, match="cancelled unexpectedly") as excinfo:
            await connect_mcp_toolset(_fake_server_spec(), client=FakeMcpClient(session))
        await asyncio.sleep(0)
        return session.closed, excinfo.value

    closed, error = asyncio.run(run())

    assert closed is True
    assert isinstance(error.__cause__, McpProtocolError)
    assert secret not in "".join(traceback.format_exception(error))
    assert REDACTED_SECRET in str(error.__cause__)


def test_mcp_tool_adapter_runs_through_cayu_runtime() -> None:
    async def run():
        toolset = await connect_mcp_toolset(
            _fake_server_spec().model_copy(update={"connection_id": "local-mcp"}),
            client=StdioMcpClient(),
        )
        try:
            provider = FakeProvider(
                [
                    [
                        ModelStreamEvent.tool_call(
                            id="call_1",
                            name=toolset.tools[0].name,
                            arguments={"text": "runtime"},
                        ),
                        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                    ],
                    [
                        ModelStreamEvent.text_delta("done"),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ],
                ]
            )
            app = CayuApp()
            app.register_provider(provider, default=True)
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=toolset.tools,
            )
            events = await _collect_events(
                app.run(
                    RunRequest(
                        agent_name="assistant",
                        messages=[Message.text("user", "Use the MCP echo tool.")],
                    )
                )
            )
            return events, provider
        finally:
            await toolset.close()

    events, provider = asyncio.run(run())

    completed = [event for event in events if event.type == "tool.call.completed"]
    assert len(completed) == 1
    assert completed[0].tool_name == "mcp__local-mcp__echo"
    assert completed[0].payload["result"]["content"] == (
        'echo: runtime\n\nStructured MCP content:\n{\n  "echoed": "runtime"\n}'
    )
    assert provider.requests[1].messages[-1].content[0].content == (
        'echo: runtime\n\nStructured MCP content:\n{\n  "echoed": "runtime"\n}'
    )


def test_mcp_cayu_tool_name_is_provider_safe_and_stable() -> None:
    name = mcp_cayu_tool_name(
        "very.long/server name with spaces",
        "tool.name/with spaces and punctuation",
    )

    assert len(name) <= 64
    assert name.startswith("mcp__")
    assert all(character.isalnum() or character in {"_", "-"} for character in name)


def test_stdio_mcp_client_rejects_unresolved_secret_env() -> None:
    spec = McpServerSpec(
        name="secret-mcp",
        command=[sys.executable, str(_FAKE_SERVER)],
        secret_env={"TOKEN": {"name": "token"}},
    )

    with pytest.raises(ValueError, match="secret_env"):
        asyncio.run(StdioMcpClient().connect(spec))


def test_stdio_mcp_client_injects_secret_env_into_child_process() -> None:
    # The fake server echoes CAYU_FAKE_MCP_PROTOCOL_VERSION back as the
    # negotiated protocol version, so a vault-resolved value proves the secret
    # was injected into the child env (it never appears in argv).
    spec = McpServerSpec(
        name="secret-mcp",
        command=[sys.executable, str(_FAKE_SERVER)],
        secret_env={"CAYU_FAKE_MCP_PROTOCOL_VERSION": SecretRef(name="protocol")},
    )
    vault = StaticVault({"protocol": "1999-01-01"})

    with pytest.raises(McpProtocolError) as excinfo:
        asyncio.run(StdioMcpClient(secret_resolver=vault).connect(spec))
    assert "1999-01-01" not in str(excinfo.value)
    assert REDACTED_SECRET in str(excinfo.value)
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None
    assert not any("1999-01-01" in item for item in spec.command)


def test_stdio_mcp_client_connects_with_resolved_secret_env() -> None:
    async def run():
        spec = McpServerSpec(
            name="secret-mcp",
            command=[sys.executable, str(_FAKE_SERVER)],
            secret_env={"CAYU_FAKE_MCP_PROTOCOL_VERSION": SecretRef(name="protocol")},
        )
        vault = StaticVault({"protocol": MCP_PROTOCOL_VERSION})
        session = await StdioMcpClient(secret_resolver=vault).connect(spec)
        try:
            return session.initialize_result
        finally:
            await session.close()

    initialize_result = asyncio.run(run())

    assert initialize_result.protocol_version == MCP_PROTOCOL_VERSION


async def _collect_events(events):
    return [event async for event in events]


async def _run_mcp_manifest_session(
    *,
    store: SessionStore,
    session_id: str,
    toolset,
    environment_name: str | None = None,
    tools=None,
) -> None:
    provider = FakeProvider([[ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed({})]])
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    if environment_name is not None:
        app.register_environment(Environment(EnvironmentSpec(name=environment_name)), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=toolset.tools if tools is None else tools,
    )
    await _collect_events(
        app.run(
            RunRequest(
                session_id=session_id,
                agent_name="assistant",
                environment_name=environment_name,
                messages=[Message.text("user", "hello")],
            )
        )
    )


async def _seed_legacy_mcp_manifest_event(
    *,
    store: SessionStore,
    session_id: str,
    toolset: McpToolset,
    previous_event: Event | None = None,
) -> Event:
    await store.create(
        RunRequest(
            session_id=session_id,
            agent_name="assistant",
            messages=[Message.text("user", "legacy")],
        ),
        identity=SessionIdentity(provider_name="fake", model="fake-model"),
    )
    previous = (
        None
        if previous_event is None
        else {
            "event_id": previous_event.id,
            "session_id": previous_event.session_id,
            "manifest_identity": previous_event.payload["manifest_identity"],
            "manifest_hash": previous_event.payload["manifest_hash"],
            "server_hash": previous_event.payload["server_hash"],
        }
    )
    event = Event(
        type=EventType.MCP_MANIFEST_CHECKED,
        session_id=session_id,
        agent_name="assistant",
        payload={
            "server_name": toolset.server.name,
            "manifest_identity": f"legacy:{toolset.manifest_hash}",
            "manifest_hash": toolset.manifest_hash,
            "server_hash": toolset.manifest_server_hash,
            "status": "first_seen" if previous is None else "changed",
            "tool_count": len(toolset.definitions),
            "tools": list(toolset.manifest_tools),
            "server": {
                "protocol_version": toolset.initialize_result.protocol_version,
                "server_name": toolset.initialize_result.server_name,
                "server_version": toolset.initialize_result.server_version,
            },
            "previous": previous,
            "diff": {
                "server_changed": False,
                "added_tools": [],
                "removed_tools": [],
                "changed_tools": [],
            },
        },
    )
    await store.append_event(session_id, event)
    return event


def _fake_toolset(
    *,
    description: str = "Echo text.",
    definitions: tuple[McpToolDefinition, ...] | None = None,
    initialize_result: McpInitializeResult | None = None,
    connection_id: str | None = "local-mcp",
    server_name: str = "local-mcp",
):
    tool_definitions = (
        _fake_tool_definitions("echo", description=description)
        if definitions is None
        else definitions
    )
    return McpToolset(
        server=_fake_server_spec().model_copy(
            update={
                "connection_id": connection_id,
                "name": server_name,
            }
        ),
        session=FakeMcpSession(
            definitions=tool_definitions,
            initialize_result=initialize_result,
        ),
        definitions=tool_definitions,
    )


def _fake_tool_definitions(
    *names: str,
    description: str = "Echo text.",
) -> tuple[McpToolDefinition, ...]:
    return tuple(
        McpToolDefinition(
            name=name,
            description=description,
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
            },
        )
        for name in names
    )


def _opaque_mcp_tool_id(server_name: str, tool_name: str) -> str:
    encoded = json.dumps(
        {
            "schema": "cayu.mcp.audit_tool_identity.v1",
            "cayu_name": mcp_cayu_tool_name(server_name, tool_name),
            "mcp_name": tool_name,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _fake_server_spec() -> McpServerSpec:
    return McpServerSpec(
        name="local-mcp",
        command=[sys.executable, str(_FAKE_SERVER)],
    )


def test_base_child_env_uses_minimal_safelist_when_not_inheriting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.mcp.stdio import _MINIMAL_ENV_SAFELIST, _base_child_env

    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("HOME", "/home/agent")
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    monkeypatch.setenv("CAYU_SECRET_TOKEN", "do-not-leak")

    env = _base_child_env(False)

    assert env["PATH"] == "/usr/bin"
    assert env["HOME"] == "/home/agent"
    assert env["LANG"] == "en_US.UTF-8"
    assert "CAYU_SECRET_TOKEN" not in env
    assert set(env).issubset(set(_MINIMAL_ENV_SAFELIST))


def test_base_child_env_inherits_full_environment_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.mcp.stdio import _base_child_env

    monkeypatch.setenv("CAYU_SECRET_TOKEN", "inherited")

    env = _base_child_env(True)

    assert env["CAYU_SECRET_TOKEN"] == "inherited"


def test_stdio_mcp_client_surfaces_stderr_tail_on_startup_crash() -> None:
    async def run():
        script = (
            "import sys; sys.stderr.write('fatal: missing config value\\n'); sys.stderr.flush()"
        )
        spec = McpServerSpec(name="crash-mcp", command=[sys.executable, "-c", script])
        with pytest.raises(McpProtocolError) as excinfo:
            await StdioMcpClient().connect(spec)
        return str(excinfo.value)

    message = asyncio.run(run())

    assert "closed stdout" in message
    assert "fatal: missing config value" in message


def test_stdio_mcp_client_redacts_secret_stderr_tail_on_startup_crash() -> None:
    async def run():
        script = (
            "import os, sys; "
            "sys.stderr.write('fatal token=' + os.environ['TOKEN'] + '\\n'); "
            "sys.stderr.flush()"
        )
        spec = McpServerSpec(
            name="crash-secret-mcp",
            command=[sys.executable, "-c", script],
            secret_env={"TOKEN": SecretRef(name="token")},
        )
        vault = StaticVault({"token": "mcp-secret-token"})
        with pytest.raises(McpProtocolError) as excinfo:
            await StdioMcpClient(secret_resolver=vault).connect(spec)
        return str(excinfo.value)

    message = asyncio.run(run())

    assert "closed stdout" in message
    assert "mcp-secret-token" not in message
    assert REDACTED_SECRET in message


def test_stdio_mcp_client_redacts_secret_before_bounding_stderr_tail() -> None:
    secret = "mcp-stderr-split-boundary-canary"
    retained_secret_suffix = secret[len(secret) // 2 :]
    trailing_bytes = 8192 - len(retained_secret_suffix.encode())

    async def run():
        script = (
            "import os, sys; "
            "secret = os.environ['TOKEN']; "
            f"sys.stderr.write('p' * 100 + secret + 'z' * {trailing_bytes}); "
            "sys.stderr.flush()"
        )
        spec = McpServerSpec(
            name="crash-secret-mcp",
            command=[sys.executable, "-c", script],
            secret_env={"TOKEN": SecretRef(name="token")},
        )
        vault = StaticVault({"token": secret})
        with pytest.raises(McpProtocolError) as excinfo:
            await StdioMcpClient(secret_resolver=vault).connect(spec)
        return str(excinfo.value)

    message = asyncio.run(run())

    assert secret not in message
    assert retained_secret_suffix not in message
    assert REDACTED_SECRET in message


def test_stdio_invalid_json_error_does_not_retain_raw_decoder_document() -> None:
    secret = "mcp-stdio-invalid-json-canary"
    spec = McpServerSpec(
        name="invalid-json",
        command=[
            sys.executable,
            "-c",
            "import os; print('not-json-' + os.environ['TOKEN'], flush=True)",
        ],
        secret_env={"TOKEN": SecretRef(name="token")},
    )

    async def run():
        with pytest.raises(McpProtocolError) as excinfo:
            await StdioMcpClient(secret_resolver=StaticVault({"token": secret})).connect(spec)
        return excinfo.value

    error = asyncio.run(run())

    _assert_mcp_traceback_does_not_retain_secret(error, secret)


@pytest.mark.parametrize(
    ("response_mode", "error_match"),
    [
        ("non_object", "must be an object"),
        ("wrong_version", "jsonrpc='2.0'"),
    ],
)
def test_stdio_structurally_invalid_json_does_not_retain_secret_payload(
    response_mode: str,
    error_match: str,
) -> None:
    secret = f"mcp-stdio-{response_mode}-json-canary"
    spec = McpServerSpec(
        name="structurally-invalid-json",
        command=[sys.executable, str(_FAKE_SERVER)],
        env={"CAYU_FAKE_MCP_STRUCTURAL_RESPONSE": response_mode},
        secret_env={
            "CAYU_FAKE_MCP_STRUCTURAL_CANARY": SecretRef(name="token"),
        },
    )

    async def run():
        session = await StdioMcpClient(secret_resolver=StaticVault({"token": secret})).connect(spec)
        try:
            with pytest.raises(McpProtocolError, match=error_match) as excinfo:
                await session.list_tools()
            return excinfo.value
        finally:
            await session.close()

    error = asyncio.run(run())

    _assert_mcp_traceback_does_not_retain_secret(error, secret)


@pytest.mark.parametrize(
    ("response_mode", "error_match"),
    [
        ("non_finite", "invalid portable JSON"),
        ("non_finite_cursor", "invalid portable JSON"),
        ("unclean_identity", "clean nonblank"),
        ("ambiguous_identity_first", "invalid tool definition"),
        ("ambiguous_identity_last", "invalid tool definition"),
        ("ambiguous_identity_only", "invalid tool definition"),
    ],
)
def test_stdio_post_parse_failure_does_not_retain_secret_payload(
    response_mode: str,
    error_match: str,
) -> None:
    secret = f"mcp-stdio-{response_mode}-json-canary"
    spec = McpServerSpec(
        name="post-parse-invalid-json",
        command=[sys.executable, str(_FAKE_SERVER)],
        env={"CAYU_FAKE_MCP_STRUCTURAL_RESPONSE": response_mode},
        secret_env={
            "CAYU_FAKE_MCP_STRUCTURAL_CANARY": SecretRef(name="token"),
        },
    )

    async def run():
        session = await StdioMcpClient(secret_resolver=StaticVault({"token": secret})).connect(spec)
        assert isinstance(session, StdioMcpSession)
        try:
            with pytest.raises(McpProtocolError, match=error_match) as excinfo:
                await session.list_tools()
            return dict(session._tool_transport_names), excinfo.value
        finally:
            await session.close()

    mapping, error = asyncio.run(run())

    assert mapping == {}
    _assert_mcp_traceback_does_not_retain_secret(error, secret)


def test_stdio_private_tool_refresh_commits_only_the_accepted_transport_authority(
    tmp_path: Path,
) -> None:
    first_private_name = "mcp-stdio-private-refresh-name-alpha"
    second_private_name = "mcp-stdio-private-refresh-name-beta"
    first_private_description = "mcp-stdio-private-refresh-description-alpha"
    second_private_description = "mcp-stdio-private-refresh-description-beta"
    catalogue_path = tmp_path / "tools.json"

    def publish_catalogue(*, name: str, description: str) -> None:
        catalogue_path.write_text(
            json.dumps(
                [
                    {
                        "name": name,
                        "description": description,
                        "inputSchema": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                        },
                    }
                ]
            ),
            encoding="utf-8",
        )

    publish_catalogue(
        name=first_private_name,
        description=first_private_description,
    )
    spec = McpServerSpec(
        name="private-stdio-refresh",
        connection_id="private-stdio-refresh",
        command=[sys.executable, str(_FAKE_SERVER)],
        env={"CAYU_FAKE_MCP_TOOL_CATALOGUE_FILE": str(catalogue_path)},
        secret_env={
            "CAYU_FAKE_MCP_REFRESH_NAME_ALPHA": SecretRef(name="name_alpha"),
            "CAYU_FAKE_MCP_REFRESH_NAME_BETA": SecretRef(name="name_beta"),
            "CAYU_FAKE_MCP_REFRESH_DESCRIPTION_ALPHA": SecretRef(name="description_alpha"),
            "CAYU_FAKE_MCP_REFRESH_DESCRIPTION_BETA": SecretRef(name="description_beta"),
        },
    )
    vault = StaticVault(
        {
            "name_alpha": first_private_name,
            "name_beta": second_private_name,
            "description_alpha": first_private_description,
            "description_beta": second_private_description,
        }
    )

    async def run():
        toolset = await connect_mcp_toolset(
            spec,
            client=StdioMcpClient(secret_resolver=vault),
        )
        session = toolset.session
        assert isinstance(session, StdioMcpSession)
        app = CayuApp(
            enable_logging=False,
            mcp_manifest_policy=McpManifestPolicy(
                on_tools_changed=McpManifestPolicyAction.ALLOW,
            ),
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            mcp_toolsets=(toolset,),
        )
        try:
            publish_catalogue(
                name=second_private_name,
                description=second_private_description,
            )
            refresh = await app.refresh_mcp_toolset(toolset)
            call_result = await refresh.toolset.call_tool(
                refresh.toolset.definitions[0].name,
                {"text": "new private authority"},
            )
            return refresh, call_result, dict(session._tool_transport_names)
        finally:
            await toolset.close()

    refresh, call_result, mapping = asyncio.run(run())

    assert refresh.status == "accepted"
    assert refresh.diff.changed_tools == (refresh.toolset.tools[0].name,)
    assert mapping == {REDACTED_SECRET: second_private_name}
    assert call_result.content[0]["text"] == "echo: new private authority"
    public_refresh = repr(refresh.diff.policy_input())
    for secret in (
        first_private_name,
        second_private_name,
        first_private_description,
        second_private_description,
    ):
        assert secret not in public_refresh


@pytest.mark.parametrize(
    "response_mode",
    [
        "ambiguous_identity_first",
        "ambiguous_identity_last",
        "ambiguous_identity_only",
    ],
)
def test_stdio_ambiguous_resource_authority_never_commits_private_mapping(
    response_mode: str,
) -> None:
    secret = f"mcp-stdio-{response_mode}-resource-authority-canary"
    spec = McpServerSpec(
        name="post-parse-invalid-resource",
        command=[sys.executable, str(_FAKE_SERVER)],
        env={
            "CAYU_FAKE_MCP_STRUCTURAL_RESPONSE": response_mode,
            "CAYU_FAKE_MCP_STRUCTURAL_METHOD": "resources/list",
        },
        secret_env={
            "CAYU_FAKE_MCP_STRUCTURAL_CANARY": SecretRef(name="token"),
        },
    )

    async def run():
        session = await StdioMcpClient(secret_resolver=StaticVault({"token": secret})).connect(spec)
        assert isinstance(session, StdioMcpSession)
        try:
            with pytest.raises(
                McpProtocolError,
                match="invalid resource definition",
            ) as excinfo:
                await session.list_resources()
            return dict(session._resource_transport_uris), excinfo.value
        finally:
            await session.close()

    mapping, error = asyncio.run(run())

    assert mapping == {}
    _assert_mcp_traceback_does_not_retain_secret(error, secret)


def test_stdio_invalid_initialize_envelope_does_not_retain_secret_config() -> None:
    secret = "mcp-stdio-initialize-json-canary"
    spec = McpServerSpec(
        name="structurally-invalid-initialize",
        command=[sys.executable, str(_FAKE_SERVER)],
        env={
            "CAYU_FAKE_MCP_STRUCTURAL_RESPONSE": "wrong_version",
            "CAYU_FAKE_MCP_STRUCTURAL_METHOD": "initialize",
        },
        secret_env={
            "CAYU_FAKE_MCP_STRUCTURAL_CANARY": SecretRef(name="token"),
        },
    )

    async def run():
        with pytest.raises(McpProtocolError, match="jsonrpc='2.0'") as excinfo:
            await StdioMcpClient(secret_resolver=StaticVault({"token": secret})).connect(spec)
        return excinfo.value

    error = asyncio.run(run())

    _assert_mcp_traceback_does_not_retain_secret(error, secret)


def test_stdio_invalid_initialize_text_does_not_retain_resolved_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "mcp-stdio-initialize-portable-canary"
    spec = McpServerSpec(
        name="invalid-initialize-text",
        command=[sys.executable, str(_FAKE_SERVER)],
        env={"CAYU_FAKE_MCP_INVALID_PROTOCOL_TEXT": "1"},
        secret_env={
            "CAYU_FAKE_MCP_PROTOCOL_VERSION": SecretRef(name="token"),
        },
    )
    original_spawn = asyncio.create_subprocess_exec
    spawned_process: asyncio.subprocess.Process | None = None

    async def capture_spawn(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
        nonlocal spawned_process
        spawned_process = await original_spawn(*args, **kwargs)
        return spawned_process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", capture_spawn)

    async def run() -> McpProtocolError:
        with pytest.raises(
            McpProtocolError,
            match="initialize result contained invalid data",
        ) as excinfo:
            await StdioMcpClient(secret_resolver=StaticVault({"token": secret})).connect(spec)
        return excinfo.value

    error = asyncio.run(run())

    assert spawned_process is not None
    assert spawned_process.returncode is not None
    _assert_mcp_traceback_does_not_retain_secret(error, secret)


@pytest.mark.parametrize(
    ("method", "error_match"),
    [
        ("tools/call", "tools/call result contained invalid data"),
        ("resources/read", "resources/read result contained invalid data"),
    ],
)
def test_stdio_invalid_result_models_raise_detached_protocol_errors(
    method: str,
    error_match: str,
) -> None:
    secret = f"mcp-stdio-{method}-portable-canary"
    spec = McpServerSpec(
        name="invalid-result-model",
        command=[sys.executable, str(_FAKE_SERVER)],
        env={
            "CAYU_FAKE_MCP_STRUCTURAL_RESPONSE": "invalid_portable_result",
            "CAYU_FAKE_MCP_STRUCTURAL_METHOD": method,
        },
        secret_env={
            "CAYU_FAKE_MCP_STRUCTURAL_CANARY": SecretRef(name="token"),
        },
    )

    async def run() -> tuple[McpProtocolError, int | None]:
        session = await StdioMcpClient(secret_resolver=StaticVault({"token": secret})).connect(spec)
        assert isinstance(session, StdioMcpSession)
        try:
            with pytest.raises(McpProtocolError, match=error_match) as excinfo:
                if method == "tools/call":
                    await session.call_tool("echo", {})
                else:
                    await session.read_resource("file:///hello.txt")
            error = excinfo.value
        finally:
            await session.close()
        return error, session.process.returncode

    error, returncode = asyncio.run(run())

    assert returncode is not None
    _assert_mcp_traceback_does_not_retain_secret(error, secret)


def _assert_mcp_traceback_does_not_retain_secret(
    error: BaseException,
    secret: str,
) -> None:
    assert secret not in str(error)
    assert error.__cause__ is None
    assert error.__context__ is None
    traceback = error.__traceback__
    while traceback is not None:
        if is_cayu_source_filename(traceback.tb_frame.f_code.co_filename):
            for value in traceback.tb_frame.f_locals.values():
                assert secret not in repr(value)
        traceback = traceback.tb_next


def test_stdio_mcp_session_fast_fails_after_reader_stops() -> None:
    async def run():
        client = StdioMcpClient()
        session = await client.connect(_fake_server_spec())
        assert isinstance(session, StdioMcpSession)
        session.process.kill()
        await session.process.wait()
        # Let the reader observe the closed stdout and latch the session.
        with suppress(Exception):
            await asyncio.wait_for(asyncio.shield(session._reader_task), timeout=2.0)
        assert session._closed is True
        try:
            with pytest.raises(McpProtocolError, match="session is closed"):
                await asyncio.wait_for(session.list_tools(), timeout=1.0)
        finally:
            await session.close()

    asyncio.run(run())


def test_mcp_server_spec_rejects_secret_config_collisions() -> None:
    with pytest.raises(ValueError, match="env and secret_env"):
        McpServerSpec(
            name="secret-mcp",
            command=["server"],
            env={"TOKEN": "plain"},
            secret_env={"TOKEN": SecretRef(name="token")},
        )

    with pytest.raises(ValueError, match="headers and secret_headers"):
        McpServerSpec(
            name="secret-mcp",
            url="https://mcp.example/rpc",
            headers={"Authorization": "Bearer plain"},
            secret_headers={"authorization": SecretRef(name="token")},
        )
