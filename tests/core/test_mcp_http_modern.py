from __future__ import annotations

import asyncio
import json
import traceback
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from tests.provider_traceback_assertions import is_cayu_source_filename

from cayu import (
    DEFAULT_MCP_CLIENT_VERSION,
    MCP_MODERN_PROTOCOL_VERSION,
    HttpMcpClient,
    McpCallDeadlineExceededError,
    McpProtocolEra,
    McpProtocolError,
    McpServerSpec,
    McpTransportLimits,
)
from cayu.mcp._http_protocol import (
    MCP_METHOD_HEADER,
    MCP_NAME_HEADER,
    MCP_PROTOCOL_VERSION_HEADER,
    MCP_SESSION_ID_HEADER,
    encode_mcp_http_header_value,
    mirrored_mcp_http_tool_headers,
    modern_http_tool_header_contract,
)
from cayu.mcp._jsonrpc import tool_result_from_payload
from cayu.mcp.http import _http_settlement_task


class _BlockingModernResponseStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.started.set()
        await asyncio.Event().wait()
        yield b""  # pragma: no cover - cancellation or timeout always wins

    async def aclose(self) -> None:
        self.closed = True


class ModernMcpHttpServer:
    def __init__(
        self,
        *,
        sse: bool = False,
        tools_list_changed: bool = False,
        response_session_id: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> None:
        self.sse = sse
        self.tools_list_changed = tools_list_changed
        self.response_session_id = response_session_id
        self.tools = tools or [
            {
                "name": "search",
                "description": "Search.",
                "inputSchema": {"type": "object"},
            }
        ]
        self.calls: list[tuple[dict[str, Any], dict[str, str]]] = []
        self.get_calls = 0
        self.delete_calls = 0
        self.error_once: tuple[str, int] | None = None
        self.result_overrides: dict[str, dict[str, Any]] = {}

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            self.get_calls += 1
            return httpx.Response(405)
        if request.method == "DELETE":
            self.delete_calls += 1
            return httpx.Response(405)
        body = json.loads(request.content)
        headers = {name.lower(): value for name, value in request.headers.items()}
        self.calls.append((body, headers))
        method = body["method"]
        if self.error_once is not None and self.error_once[0] == method:
            _method, status = self.error_once
            self.error_once = None
            return httpx.Response(
                status,
                headers={"content-type": "application/json"},
                json={
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "error": {"code": -32601, "message": "Method not found"},
                },
            )
        result = self.result_overrides.get(method, self._result_for(method))
        payload = {"jsonrpc": "2.0", "id": body["id"], "result": result}
        response_headers: dict[str, str] = {}
        if self.response_session_id is not None:
            response_headers[MCP_SESSION_ID_HEADER] = self.response_session_id
        if self.sse:
            response_headers["content-type"] = "text/event-stream"
            return httpx.Response(
                200,
                headers=response_headers,
                content=f"event: message\ndata: {json.dumps(payload)}\n\n".encode(),
            )
        return httpx.Response(200, headers=response_headers, json=payload)

    def _result_for(self, method: str) -> dict[str, Any]:
        if method == "server/discover":
            return {
                "resultType": "complete",
                "ttlMs": 0,
                "cacheScope": "private",
                "supportedVersions": [MCP_MODERN_PROTOCOL_VERSION],
                "capabilities": {
                    "tools": {"listChanged": self.tools_list_changed},
                    "resources": {},
                },
                "instructions": "Use the server carefully.",
                "_meta": {
                    "io.modelcontextprotocol/serverInfo": {
                        "name": "modern-fixture",
                        "version": "1.0",
                    }
                },
            }
        if method == "tools/list":
            return {
                "resultType": "complete",
                "ttlMs": 10,
                "cacheScope": "public",
                "tools": self.tools,
            }
        if method == "tools/call":
            return {
                "resultType": "complete",
                "content": [{"type": "text", "text": "ok"}],
                "structuredContent": ["one", 2],
                "isError": False,
            }
        if method == "resources/list":
            return {
                "resultType": "complete",
                "ttlMs": 0,
                "cacheScope": "private",
                "resources": [{"uri": "file://x", "name": "x"}],
            }
        if method == "resources/read":
            return {
                "resultType": "complete",
                "ttlMs": 0,
                "cacheScope": "private",
                "contents": [{"uri": "file://x", "text": "hello"}],
            }
        raise AssertionError(f"Unexpected method: {method}")

    def requests_for(self, method: str) -> list[tuple[dict[str, Any], dict[str, str]]]:
        return [call for call in self.calls if call[0]["method"] == method]


def _server_spec() -> McpServerSpec:
    return McpServerSpec(name="modern", url="https://mcp.example/rpc")


def _client(server: ModernMcpHttpServer) -> HttpMcpClient:
    return HttpMcpClient(
        protocol_era=McpProtocolEra.MODERN_2026_07_28,
        transport=server.transport,
    )


@pytest.mark.parametrize("sse", [False, True])
def test_modern_http_discovers_lists_calls_and_closes_statelessly(sse: bool) -> None:
    server = ModernMcpHttpServer(sse=sse)

    async def run():
        session = await _client(server).connect(_server_spec())
        try:
            metadata = session.initialize_result
            tools = await session.list_tools()
            tool_result = await session.call_tool("search", {"q": "cayu"})
            resources = await session.list_resources()
            resource = await session.read_resource("file://x")
            return metadata, tools, tool_result, resources, resource
        finally:
            await session.close()

    metadata, tools, tool_result, resources, resource = asyncio.run(run())

    assert metadata.protocol_version == MCP_MODERN_PROTOCOL_VERSION
    assert metadata.server_name == "modern-fixture"
    assert metadata.server_version == "1.0"
    assert metadata.instructions == "Use the server carefully."
    assert [tool.name for tool in tools] == ["search"]
    assert tool_result.content == [{"type": "text", "text": "ok"}]
    assert tool_result.structured_content == ["one", 2]
    assert [item.uri for item in resources] == ["file://x"]
    assert resource.contents == [{"uri": "file://x", "text": "hello"}]
    assert [body["method"] for body, _headers in server.calls] == [
        "server/discover",
        "tools/list",
        "tools/call",
        "resources/list",
        "resources/read",
    ]
    assert server.delete_calls == 0
    assert server.get_calls == 0

    expected_meta = {
        "io.modelcontextprotocol/protocolVersion": MCP_MODERN_PROTOCOL_VERSION,
        "io.modelcontextprotocol/clientInfo": {
            "name": "cayu",
            "version": DEFAULT_MCP_CLIENT_VERSION,
        },
        "io.modelcontextprotocol/clientCapabilities": {},
    }
    for body, headers in server.calls:
        assert body["params"]["_meta"] == expected_meta
        assert headers[MCP_PROTOCOL_VERSION_HEADER] == MCP_MODERN_PROTOCOL_VERSION
        assert headers[MCP_METHOD_HEADER] == body["method"]
        assert MCP_SESSION_ID_HEADER not in headers
    assert MCP_NAME_HEADER not in server.requests_for("server/discover")[0][1]
    assert MCP_NAME_HEADER not in server.requests_for("tools/list")[0][1]
    assert server.requests_for("tools/call")[0][1][MCP_NAME_HEADER] == "search"
    assert server.requests_for("resources/read")[0][1][MCP_NAME_HEADER] == "file://x"


def test_modern_http_mirrors_only_admitted_tool_header_authority() -> None:
    server = ModernMcpHttpServer(
        tools=[
            {
                "name": "écho",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "tenant": {"type": "string", "x-mcp-header": "Tenant"},
                        "empty": {"type": "string", "x-mcp-header": "Empty"},
                        "nested": {
                            "type": "object",
                            "properties": {
                                "enabled": {
                                    "type": "boolean",
                                    "x-mcp-header": "Enabled",
                                }
                            },
                        },
                    },
                },
            },
            {
                "name": "invalid",
                "inputSchema": {
                    "type": "object",
                    "properties": {"ratio": {"type": "number", "x-mcp-header": "Ratio"}},
                },
            },
        ]
    )

    async def run():
        session = await _client(server).connect(_server_spec())
        try:
            tools = await session.list_tools()
            await session.call_tool(
                "écho",
                {"tenant": "世界", "empty": "", "nested": {"enabled": True}},
            )
            return tools
        finally:
            await session.close()

    tools = asyncio.run(run())

    assert [tool.name for tool in tools] == ["écho"]
    _body, headers = server.requests_for("tools/call")[0]
    assert headers[MCP_NAME_HEADER] == "=?base64?w6ljaG8=?="
    assert headers["mcp-param-tenant"] == "=?base64?5LiW55WM?="
    assert headers["mcp-param-empty"] == "=?base64??="
    assert headers["mcp-param-enabled"] == "true"


def test_arbitrary_structured_content_is_modern_only_on_the_wire() -> None:
    legacy_payload = {
        "content": [{"type": "text", "text": "ok"}],
        "structuredContent": ["one", 2],
    }
    with pytest.raises(McpProtocolError, match="contained invalid data"):
        tool_result_from_payload(legacy_payload)

    modern_payload = {
        "content": [{"type": "text", "text": "ok"}],
        "structuredContent": ["one", 2],
    }
    parsed = tool_result_from_payload(
        modern_payload,
        allow_arbitrary_structured_content=True,
    )
    assert parsed.structured_content == ["one", 2]


def test_modern_http_counts_excluded_tools_toward_the_wire_item_limit() -> None:
    invalid_tool = {
        "name": "invalid",
        "inputSchema": {
            "type": "object",
            "properties": {"ratio": {"type": "number", "x-mcp-header": "Ratio"}},
        },
    }
    server = ModernMcpHttpServer(tools=[invalid_tool, {**invalid_tool, "name": "invalid-2"}])

    async def run() -> None:
        session = await HttpMcpClient(
            protocol_era=McpProtocolEra.MODERN_2026_07_28,
            transport=server.transport,
            max_list_items=1,
        ).connect(_server_spec())
        try:
            await session.list_tools()
        finally:
            await session.close()

    with pytest.raises(McpProtocolError, match="max_list_items=1"):
        asyncio.run(run())


def test_modern_http_requires_discovery_before_direct_tool_calls() -> None:
    server = ModernMcpHttpServer()

    async def run() -> None:
        session = await _client(server).connect(_server_spec())
        try:
            with pytest.raises(McpProtocolError, match="must be listed"):
                await session.call_tool("search", {})
        finally:
            await session.close()

    asyncio.run(run())
    assert server.requests_for("tools/call") == []


@pytest.mark.parametrize(
    ("method", "result", "error_match"),
    [
        (
            "server/discover",
            {
                "resultType": "complete",
                "ttlMs": 0,
                "cacheScope": "private",
                "supportedVersions": ["2025-06-18"],
                "capabilities": {},
            },
            "does not support pinned protocol version",
        ),
        (
            "server/discover",
            {
                "resultType": "complete",
                "cacheScope": "private",
                "supportedVersions": [MCP_MODERN_PROTOCOL_VERSION],
                "capabilities": {},
            },
            "ttlMs",
        ),
        (
            "tools/list",
            {
                "resultType": "complete",
                "ttlMs": 0,
                "cacheScope": "shared",
                "tools": [],
            },
            "cacheScope",
        ),
        (
            "tools/call",
            {"resultType": "input_required", "content": []},
            "resultType",
        ),
        (
            "resources/read",
            {
                "resultType": "complete",
                "ttlMs": True,
                "cacheScope": "private",
                "contents": [],
            },
            "ttlMs",
        ),
    ],
)
def test_modern_http_rejects_invalid_wire_results(
    method: str,
    result: dict[str, Any],
    error_match: str,
) -> None:
    server = ModernMcpHttpServer()
    server.result_overrides[method] = result

    async def run() -> None:
        if method == "server/discover":
            await _client(server).connect(_server_spec())
            return
        session = await _client(server).connect(_server_spec())
        try:
            if method == "tools/list":
                await session.list_tools()
            elif method == "tools/call":
                await session.list_tools()
                await session.call_tool("search", {})
            else:
                await session.read_resource("file://x")
        finally:
            await session.close()

    with pytest.raises(McpProtocolError, match=error_match):
        asyncio.run(run())


@pytest.mark.parametrize(
    "method",
    [
        "server/discover",
        "tools/list",
        "tools/call",
        "resources/list",
        "resources/read",
    ],
)
def test_modern_http_treats_missing_result_type_as_complete(method: str) -> None:
    server = ModernMcpHttpServer()
    result = server._result_for(method)
    result.pop("resultType")
    server.result_overrides[method] = result

    async def run() -> None:
        session = await _client(server).connect(_server_spec())
        try:
            if method == "tools/list":
                await session.list_tools()
            elif method == "tools/call":
                await session.list_tools()
                await session.call_tool("search", {})
            elif method == "resources/list":
                await session.list_resources()
            elif method == "resources/read":
                await session.read_resource("file://x")
        finally:
            await session.close()

    asyncio.run(run())


def test_modern_http_invalid_control_fields_do_not_retain_raw_values() -> None:
    server = ModernMcpHttpServer()
    canary = "modern-cache-control-secret-canary"
    server.result_overrides["tools/list"] = {
        "resultType": "complete",
        "ttlMs": canary,
        "cacheScope": "private",
        "tools": [],
    }

    async def run() -> BaseException:
        session = await _client(server).connect(_server_spec())
        try:
            with pytest.raises(McpProtocolError, match="ttlMs") as exc_info:
                await session.list_tools()
            return exc_info.value
        finally:
            await session.close()

    error = asyncio.run(run())
    assert canary not in "".join(traceback.format_exception(error))
    traceback_cursor = error.__traceback__
    while traceback_cursor is not None:
        if is_cayu_source_filename(traceback_cursor.tb_frame.f_code.co_filename):
            assert all(
                canary not in repr(value) for value in traceback_cursor.tb_frame.f_locals.values()
            )
        traceback_cursor = traceback_cursor.tb_next


def test_modern_http_invalid_result_type_does_not_retain_raw_value() -> None:
    server = ModernMcpHttpServer()
    canary = "modern-result-type-secret-canary"
    server.result_overrides["tools/call"] = {
        "resultType": canary,
        "content": [],
    }

    async def run() -> BaseException:
        session = await _client(server).connect(_server_spec())
        try:
            await session.list_tools()
            with pytest.raises(McpProtocolError, match="resultType") as exc_info:
                await session.call_tool("search", {})
            return exc_info.value
        finally:
            await session.close()

    error = asyncio.run(run())
    assert canary not in "".join(traceback.format_exception(error))
    traceback_cursor = error.__traceback__
    while traceback_cursor is not None:
        if is_cayu_source_filename(traceback_cursor.tb_frame.f_code.co_filename):
            assert all(
                canary not in repr(value) for value in traceback_cursor.tb_frame.f_locals.values()
            )
        traceback_cursor = traceback_cursor.tb_next


def test_modern_http_treats_malformed_server_identity_as_anonymous() -> None:
    server = ModernMcpHttpServer()
    result = server._result_for("server/discover")
    result["_meta"] = {"io.modelcontextprotocol/serverInfo": {"name": ["hostile"], "version": "1"}}
    server.result_overrides["server/discover"] = result

    async def run():
        session = await _client(server).connect(_server_spec())
        try:
            return session.initialize_result
        finally:
            await session.close()

    metadata = asyncio.run(run())
    assert metadata.server_name is None
    assert metadata.server_version is None


def test_modern_http_rejects_protocol_session_headers() -> None:
    server = ModernMcpHttpServer(response_session_id="forbidden")

    async def run() -> None:
        await _client(server).connect(_server_spec())

    with pytest.raises(McpProtocolError, match="must not mint"):
        asyncio.run(run())
    assert server.delete_calls == 0


@pytest.mark.parametrize(
    "header_name",
    [
        "Accept",
        "Accept-Encoding",
        "Content-Type",
        "MCP-Protocol-Version",
        "Mcp-Method",
        "Mcp-Name",
        "Mcp-Session-Id",
        "Mcp-Param-Tenant",
    ],
)
def test_modern_http_rejects_static_transport_owned_headers(header_name: str) -> None:
    server = ModernMcpHttpServer()

    async def run() -> None:
        await _client(server).connect(
            McpServerSpec(
                name="modern",
                url="https://mcp.example/rpc",
                headers={header_name: "forbidden"},
            )
        )

    with pytest.raises(ValueError, match="transport-owned"):
        asyncio.run(run())
    assert server.calls == []


def test_modern_http_does_not_install_the_legacy_list_changed_listener() -> None:
    server = ModernMcpHttpServer(tools_list_changed=True)

    async def run() -> tuple[bool, bool]:
        session = await _client(server).connect(_server_spec())
        try:
            continuity = session._set_tools_list_changed_continuity_handler(lambda _ready: None)
            listener = session._set_tools_list_changed_handler(lambda: None)
            await asyncio.sleep(0)
            return continuity, listener
        finally:
            await session.close()

    assert asyncio.run(run()) == (False, False)
    assert server.get_calls == 0


def test_modern_http_404_is_a_request_error_not_an_expired_session() -> None:
    server = ModernMcpHttpServer()
    server.error_once = ("tools/call", 404)

    async def run() -> None:
        session = await _client(server).connect(_server_spec())
        try:
            await session.list_tools()
            with pytest.raises(McpProtocolError, match="HTTP 404"):
                await session.call_tool("search", {})
            assert [tool.name for tool in await session.list_tools()] == ["search"]
        finally:
            await session.close()

    asyncio.run(run())


def test_modern_http_cancellation_closes_only_the_request_stream() -> None:
    server = ModernMcpHttpServer()
    blocked = _BlockingModernResponseStream()

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["method"] == "tools/call":
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                stream=blocked,
            )
        return server._handle(request)

    async def run() -> tuple[bool, bool, list[str]]:
        client = HttpMcpClient(
            protocol_era=McpProtocolEra.MODERN_2026_07_28,
            transport=httpx.MockTransport(handler),
        )
        session = await client.connect(_server_spec())
        try:
            await session.list_tools()
            call = asyncio.create_task(session.call_tool("search", {}))
            await asyncio.wait_for(blocked.started.wait(), timeout=0.2)
            call.cancel()
            with pytest.raises(asyncio.CancelledError) as exc_info:
                await call
            settlement = _http_settlement_task(exc_info.value)
            assert settlement is not None
            await asyncio.wait_for(settlement, timeout=0.2)
            tools = await session.list_tools()
            return session._closed, session._http.is_closed, [tool.name for tool in tools]
        finally:
            await session.close()

    session_closed, client_closed, tools = asyncio.run(run())
    assert blocked.closed is True
    assert session_closed is False
    assert client_closed is False
    assert tools == ["search"]


def test_modern_http_deadline_settles_only_the_timed_out_request() -> None:
    server = ModernMcpHttpServer()
    blocked = _BlockingModernResponseStream()

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["method"] == "tools/call":
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                stream=blocked,
            )
        return server._handle(request)

    async def run() -> tuple[bool, bool, list[str]]:
        client = HttpMcpClient(
            protocol_era=McpProtocolEra.MODERN_2026_07_28,
            transport=httpx.MockTransport(handler),
            transport_limits=McpTransportLimits(
                idle_timeout_s=1,
                total_call_timeout_s=0.02,
            ),
        )
        session = await client.connect(_server_spec())
        try:
            await session.list_tools()
            with pytest.raises(McpCallDeadlineExceededError) as exc_info:
                await session.call_tool("search", {})
            settlement = _http_settlement_task(exc_info.value)
            assert settlement is not None
            await asyncio.wait_for(settlement, timeout=0.2)
            tools = await session.list_tools()
            return session._closed, session._http.is_closed, [tool.name for tool in tools]
        finally:
            await session.close()

    session_closed, client_closed, tools = asyncio.run(run())
    assert blocked.closed is True
    assert session_closed is False
    assert client_closed is False
    assert tools == ["search"]


def test_modern_http_close_drains_an_inflight_request_without_delete() -> None:
    server = ModernMcpHttpServer()
    request_started = asyncio.Event()
    release_request = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["method"] == "tools/call":
            request_started.set()
            await release_request.wait()
        return server._handle(request)

    async def run() -> tuple[bool, bool, bool]:
        client = HttpMcpClient(
            protocol_era=McpProtocolEra.MODERN_2026_07_28,
            transport=httpx.MockTransport(handler),
        )
        session = await client.connect(_server_spec())
        await session.list_tools()
        call = asyncio.create_task(session.call_tool("search", {}))
        await asyncio.wait_for(request_started.wait(), timeout=0.2)
        close = asyncio.create_task(session.close())
        await asyncio.sleep(0)
        close_waited = not close.done()
        release_request.set()
        with pytest.raises(McpProtocolError, match="closed before the response"):
            await call
        await asyncio.wait_for(close, timeout=0.2)
        return close_waited, session._closed, session._http.is_closed

    close_waited, session_closed, client_closed = asyncio.run(run())
    assert close_waited is True
    assert session_closed is True
    assert client_closed is True
    assert server.delete_calls == 0


def test_http_client_requires_an_explicit_protocol_era_enum() -> None:
    with pytest.raises(TypeError, match="protocol_era"):
        HttpMcpClient(protocol_era="2026-07-28")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", "=?base64??="),
        ("plain", "plain"),
        (" padded ", "=?base64?IHBhZGRlZCA=?="),
        ("line1\nline2", "=?base64?bGluZTEKbGluZTI=?="),
        ("=?base64?literal?=", "=?base64?PT9iYXNlNjQ/bGl0ZXJhbD89?="),
    ],
)
def test_modern_http_header_value_encoding_matches_the_protocol(
    value: str,
    expected: str,
) -> None:
    assert encode_mcp_http_header_value(value, field_name="test") == expected


@pytest.mark.parametrize(
    "input_schema",
    [
        {"type": "object", "x-mcp-header": "Root"},
        {
            "type": "object",
            "allOf": [{"properties": {"tenant": {"type": "string", "x-mcp-header": "Tenant"}}}],
        },
        {
            "type": "object",
            "properties": {
                "first": {"type": "string", "x-mcp-header": "Tenant"},
                "second": {"type": "string", "x-mcp-header": "tenant"},
            },
        },
        {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "string", "x-mcp-header": "Item"},
                }
            },
        },
    ],
)
def test_modern_http_rejects_non_static_or_ambiguous_header_annotations(
    input_schema: dict[str, Any],
) -> None:
    with pytest.raises(McpProtocolError, match="x-mcp-header"):
        modern_http_tool_header_contract(input_schema)


def test_modern_http_does_not_treat_instance_data_as_header_annotations() -> None:
    schema = {
        "type": "object",
        "properties": {
            "x-mcp-header": {"type": "string"},
            "options": {
                "type": "object",
                "default": {"x-mcp-header": "ordinary-default-data"},
                "examples": [{"x-mcp-header": "ordinary-example-data"}],
                "const": {"x-mcp-header": "ordinary-const-data"},
                "enum": [{"x-mcp-header": "ordinary-enum-data"}],
            },
        },
        "$defs": {"x-mcp-header": {"type": "string"}},
    }

    assert modern_http_tool_header_contract(schema) == ()


@pytest.mark.parametrize(
    "keyword",
    [
        "additionalProperties",
        "allOf",
        "$defs",
        "dependentSchemas",
        "patternProperties",
    ],
)
def test_modern_http_rejects_annotations_in_non_reachable_subschemas(keyword: str) -> None:
    annotated_schema = {
        "type": "string",
        "x-mcp-header": "Tenant",
    }
    if keyword in {"$defs", "dependentSchemas", "patternProperties"}:
        forbidden_value: object = {"branch": annotated_schema}
    elif keyword == "allOf":
        forbidden_value = [annotated_schema]
    else:
        forbidden_value = annotated_schema

    with pytest.raises(McpProtocolError, match="statically reachable"):
        modern_http_tool_header_contract(
            {
                "type": "object",
                keyword: forbidden_value,
            }
        )


def test_modern_http_mirrored_headers_validate_values_and_omit_nulls() -> None:
    contract = modern_http_tool_header_contract(
        {
            "type": "object",
            "properties": {
                "count": {"type": "integer", "x-mcp-header": "Count"},
                "optional": {"type": "string", "x-mcp-header": "Optional"},
            },
        }
    )
    assert mirrored_mcp_http_tool_headers(
        contract,
        {"count": 42, "optional": None},
    ) == {"mcp-param-count": "42"}
    with pytest.raises(McpProtocolError, match="portable JSON integer"):
        mirrored_mcp_http_tool_headers(contract, {"count": 2**53})


def test_modern_http_mirrored_header_failure_does_not_retain_or_dispatch_value() -> None:
    server = ModernMcpHttpServer(
        tools=[
            {
                "name": "bounded",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "tenant": {"type": "string", "x-mcp-header": "Tenant"},
                    },
                },
            }
        ]
    )
    canary = "modern-mirrored-header-secret-canary"

    async def run() -> BaseException:
        session = await _client(server).connect(_server_spec())
        try:
            await session.list_tools()
            with pytest.raises(McpProtocolError, match="header value limit") as exc_info:
                await session.call_tool("bounded", {"tenant": canary + "x" * 8_192})
            return exc_info.value
        finally:
            await session.close()

    error = asyncio.run(run())
    assert server.requests_for("tools/call") == []
    assert canary not in "".join(traceback.format_exception(error))
    traceback_cursor = error.__traceback__
    while traceback_cursor is not None:
        if is_cayu_source_filename(traceback_cursor.tb_frame.f_code.co_filename):
            assert all(
                canary not in repr(value) for value in traceback_cursor.tb_frame.f_locals.values()
            )
        traceback_cursor = traceback_cursor.tb_next


def test_modern_http_bounds_mirrored_header_count_and_aggregate_bytes() -> None:
    oversized_schema = {
        "type": "object",
        "properties": {
            f"field_{index}": {
                "type": "string",
                "x-mcp-header": f"Field-{index}",
            }
            for index in range(65)
        },
    }
    with pytest.raises(McpProtocolError, match="header count limit"):
        modern_http_tool_header_contract(oversized_schema)

    contract = modern_http_tool_header_contract(
        {
            "type": "object",
            "properties": {
                f"field_{index}": {
                    "type": "string",
                    "x-mcp-header": f"Field-{index}",
                }
                for index in range(9)
            },
        }
    )
    with pytest.raises(McpProtocolError, match="aggregate limit"):
        mirrored_mcp_http_tool_headers(
            contract,
            {f"field_{index}": "x" * 8_000 for index in range(9)},
        )
