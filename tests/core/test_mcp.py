from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest
from tests.provider_traceback_assertions import is_cayu_source_filename

import cayu.mcp as mcp_module
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
    McpClient,
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
    Message,
    RunRequest,
    SessionIdentity,
    SessionStore,
    SQLiteSessionStore,
    StdioMcpClient,
    StdioMcpSession,
    ToolContext,
    ToolEffect,
    connect_mcp_toolset,
    mcp_cayu_tool_name,
    mcp_tool_manifest_hash,
    mcp_tool_manifest_identity,
    mcp_tool_manifest_tools,
)
from cayu.mcp._jsonrpc import MCP_PROTOCOL_VERSION
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import (
    InMemorySessionStore,
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
)
from cayu.runtime._event_projection import public_event_sequence
from cayu.runtime.sessions import (
    _mcp_authoritative_manifest_hash,
    _mcp_manifest_session_ref,
)
from cayu.storage import migrations as schema_migrations
from cayu.vaults import REDACTED_SECRET, SecretRedactor, SecretRef, StaticVault

_FAKE_SERVER = Path(__file__).resolve().parents[1] / "fixtures" / "fake_mcp_server.py"


class FakeProvider(ModelProvider):
    name = "fake"

    def __init__(self, events: list[list[ModelStreamEvent]]) -> None:
        self.events = events
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        for event in self.events[len(self.requests) - 1]:
            yield event


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


class FakeMcpClient(McpClient):
    def __init__(self, session: FakeMcpSession) -> None:
        self.session = session

    async def connect(self, server: McpServerSpec) -> McpSession:
        return self.session


class RacingManifestSessionStore(InMemorySessionStore):
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
        toolset = await connect_mcp_toolset(_fake_server_spec())
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


def test_mcp_tool_adapter_includes_structured_content_in_model_text() -> None:
    async def run():
        toolset = await connect_mcp_toolset(_fake_server_spec())
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
    assert secret not in json.dumps(result.structured)
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
    assert secret not in json.dumps(result.structured, ensure_ascii=False)
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
    assert checkpoint == {CHECKPOINT_SCHEMA_VERSION_KEY: 1}
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


def test_runtime_fails_closed_with_bounded_evidence_for_oversized_manifest() -> None:
    async def run():
        store = InMemorySessionStore()
        provider = FakeProvider([[ModelStreamEvent.completed({})]])
        toolset = _fake_toolset(
            definitions=_fake_tool_definitions(*(f"tool_{index:05d}" for index in range(10_001)))
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
                    session_id="mcp_manifest_oversized",
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
    assert blocked[0].payload["reason"] == "manifest_tool_limit_exceeded"
    assert len(json.dumps(blocked[0].payload)) < 1_000


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
        ) -> None:
            await original_write_with_timeout(payload, timeout_message=timeout_message)
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
        ) -> None:
            await original_write_with_timeout(payload, timeout_message=timeout_message)
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
            with pytest.raises(asyncio.CancelledError):
                await session._request("tools/list", {})
            with pytest.raises(McpProtocolError, match="closed"):
                await session.list_tools()
            return dict(session._pending), session.process.returncode
        finally:
            await session.close()

    pending, returncode = asyncio.run(run())

    assert pending == {}
    assert returncode is not None


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


def test_mcp_toolset_connect_closes_session_when_discovery_is_cancelled() -> None:
    async def run():
        session = FakeMcpSession(list_tools_error=asyncio.CancelledError())
        with pytest.raises(asyncio.CancelledError):
            await connect_mcp_toolset(_fake_server_spec(), client=FakeMcpClient(session))
        return session.closed

    assert asyncio.run(run()) is True


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


def test_mcp_toolset_connect_preserves_original_cancellation_when_cleanup_fails() -> None:
    secret = "mcp-cancel-cleanup-canary"

    async def run():
        session = FakeMcpSession(
            list_tools_error=asyncio.CancelledError(),
            close_error=RuntimeError(f"cleanup failed {secret}"),
        )
        session._secret_redactor = SecretRedactor(secret)
        with pytest.raises(asyncio.CancelledError) as excinfo:
            await connect_mcp_toolset(_fake_server_spec(), client=FakeMcpClient(session))
        return session.closed, excinfo.value

    closed, error = asyncio.run(run())

    assert closed is True
    assert secret not in repr(error)


def test_mcp_tool_adapter_runs_through_cayu_runtime() -> None:
    async def run():
        toolset = await connect_mcp_toolset(
            _fake_server_spec().model_copy(update={"connection_id": "local-mcp"})
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
