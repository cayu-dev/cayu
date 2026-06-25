from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentSpec,
    EventQuery,
    EventType,
    McpClient,
    McpInitializeResult,
    McpProtocolError,
    McpResourceDefinition,
    McpResourceResult,
    McpServerSpec,
    McpSession,
    McpToolDefinition,
    McpToolResult,
    McpToolset,
    Message,
    RunRequest,
    StdioMcpClient,
    StdioMcpSession,
    ToolContext,
    connect_mcp_toolset,
    mcp_cayu_tool_name,
    mcp_tool_manifest_hash,
    mcp_tool_manifest_identity,
    mcp_tool_manifest_tools,
)
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import InMemorySessionStore

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

    @property
    def initialize_result(self) -> McpInitializeResult:
        return self._initialize_result

    async def list_tools(self) -> tuple[McpToolDefinition, ...]:
        if self.list_tools_error is not None:
            raise self.list_tools_error
        return self.definitions

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> McpToolResult:
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


def test_mcp_tool_manifest_identity_tracks_exposed_tool_names() -> None:
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
    assert mcp_tool_manifest_identity(server=server, definitions=first) != (
        mcp_tool_manifest_identity(server=server, definitions=tool_changed)
    )


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
    assert payload["server_name"] == "local-mcp"
    assert payload["manifest_hash"] == toolset.manifest_hash
    assert payload["server_hash"] == toolset.manifest_server_hash
    assert payload["status"] == "first_seen"
    assert payload["previous"] is None
    assert payload["diff"] == {
        "server_changed": False,
        "added_tools": [],
        "removed_tools": [],
        "changed_tools": [],
    }
    assert payload["tools"] == list(toolset.manifest_tools)
    assert len(records) == 1


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
    assert records[1].event.payload["previous"]["session_id"] == "mcp_manifest_unchanged_1"
    assert records[1].event.payload["diff"] == {
        "server_changed": False,
        "added_tools": [],
        "removed_tools": [],
        "changed_tools": [],
    }


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
    assert changed_payload["diff"] == {
        "server_changed": False,
        "added_tools": [],
        "removed_tools": [],
        "changed_tools": ["mcp__local-mcp__echo"],
    }


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
    assert records[1].event.payload["diff"] == {
        "server_changed": True,
        "added_tools": [],
        "removed_tools": [],
        "changed_tools": [],
    }


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
    assert records[1].event.payload["diff"] == {
        "server_changed": False,
        "added_tools": ["mcp__local-mcp__new"],
        "removed_tools": ["mcp__local-mcp__old"],
        "changed_tools": [],
    }


def test_runtime_audits_distinct_same_name_mcp_toolsets() -> None:
    async def run():
        store = InMemorySessionStore()
        echo_toolset = _fake_toolset(definitions=_fake_tool_definitions("echo"))
        summarize_toolset = _fake_toolset(definitions=_fake_tool_definitions("summarize"))
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
    assert [record.event.payload["server_name"] for record in records] == [
        "local-mcp",
        "local-mcp",
    ]
    assert [record.event.payload["tools"][0]["cayu_name"] for record in records] == [
        "mcp__local-mcp__echo",
        "mcp__local-mcp__summarize",
    ]
    assert (
        records[0].event.payload["manifest_identity"]
        != records[1].event.payload["manifest_identity"]
    )


def test_runtime_does_not_fallback_for_new_toolset_when_current_server_is_ambiguous() -> None:
    async def run():
        store = InMemorySessionStore()
        echo_toolset = _fake_toolset(definitions=_fake_tool_definitions("echo"))
        summarize_toolset = _fake_toolset(definitions=_fake_tool_definitions("summarize"))
        await _run_mcp_manifest_session(
            store=store,
            session_id="mcp_manifest_same_server_initial",
            toolset=echo_toolset,
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
                    session_id="mcp_manifest_same_server_added_second",
                    agent_name="assistant",
                    messages=[Message.text("user", "hello")],
                )
            )
        )
        return await store.query_events(
            EventQuery(event_type=EventType.MCP_MANIFEST_CHECKED, limit=10)
        )

    records = asyncio.run(run())

    assert [record.event.payload["tools"][0]["cayu_name"] for record in records] == [
        "mcp__local-mcp__echo",
        "mcp__local-mcp__echo",
        "mcp__local-mcp__summarize",
    ]
    assert [record.event.payload["status"] for record in records] == [
        "first_seen",
        "unchanged",
        "first_seen",
    ]
    assert records[2].event.payload["previous"] is None


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
        env={"CAYU_FAKE_MCP_PROTOCOL_VERSION": "2024-11-05"},
    )

    with pytest.raises(McpProtocolError, match="unsupported protocol version"):
        asyncio.run(StdioMcpClient().connect(spec))


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


def test_mcp_toolset_connect_preserves_original_cancellation_when_cleanup_fails() -> None:
    async def run():
        session = FakeMcpSession(
            list_tools_error=asyncio.CancelledError(),
            close_error=RuntimeError("cleanup failed"),
        )
        with pytest.raises(asyncio.CancelledError):
            await connect_mcp_toolset(_fake_server_spec(), client=FakeMcpClient(session))
        return session.closed

    assert asyncio.run(run()) is True


def test_mcp_tool_adapter_runs_through_cayu_runtime() -> None:
    async def run():
        toolset = await connect_mcp_toolset(_fake_server_spec())
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


async def _collect_events(events):
    return [event async for event in events]


async def _run_mcp_manifest_session(
    *,
    store: InMemorySessionStore,
    session_id: str,
    toolset,
    environment_name: str | None = None,
) -> None:
    provider = FakeProvider([[ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed({})]])
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    if environment_name is not None:
        app.register_environment(Environment(EnvironmentSpec(name=environment_name)), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=toolset.tools,
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


def _fake_toolset(
    *,
    description: str = "Echo text.",
    definitions: tuple[McpToolDefinition, ...] | None = None,
    initialize_result: McpInitializeResult | None = None,
):
    tool_definitions = (
        _fake_tool_definitions("echo", description=description)
        if definitions is None
        else definitions
    )
    return McpToolset(
        server=_fake_server_spec(),
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


def _fake_server_spec() -> McpServerSpec:
    return McpServerSpec(
        name="local-mcp",
        command=[sys.executable, str(_FAKE_SERVER)],
    )
