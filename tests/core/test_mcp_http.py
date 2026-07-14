from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from cayu import (
    DEFAULT_HTTP_MCP_CONNECT_TIMEOUT_S,
    DEFAULT_HTTP_MCP_TIMEOUT_S,
    HttpMcpClient,
    McpProtocolError,
    McpServerSpec,
    StdioMcpClient,
    connect_mcp_toolset,
)
from cayu.mcp._jsonrpc import MCP_PROTOCOL_VERSION
from cayu.mcp.http import MCP_PROTOCOL_VERSION_HEADER, MCP_SESSION_ID_HEADER, HttpMcpSession
from cayu.mcp.tools import _default_client_for
from cayu.vaults import SecretRef, StaticVault

_DEFAULT_TOOLS = [
    {"name": "search", "description": "Search the web.", "inputSchema": {"type": "object"}},
]


class FakeMcpHttpServer:
    """In-memory Streamable HTTP MCP server backing an ``httpx.MockTransport``."""

    def __init__(
        self,
        *,
        sse: bool = False,
        session_id: str | None = "sess-1",
        tools: list[dict[str, Any]] | None = None,
        protocol_version: str = MCP_PROTOCOL_VERSION,
        expire_after_init: bool = False,
        error_on: str | None = None,
        timeout_on: str | None = None,
        sse_content_type: str = "text/event-stream",
        sse_extra_events: list[dict[str, Any]] | None = None,
        sse_trailing_events: list[dict[str, Any]] | None = None,
        empty_sse_on: str | None = None,
        bad_jsonrpc_on: str | None = None,
        fold_sse: bool = False,
        paginate: bool = False,
    ) -> None:
        self.sse = sse
        self.paginate = paginate
        self.sse_content_type = sse_content_type
        self.fold_sse = fold_sse
        self.session_id = session_id
        self.tools = _DEFAULT_TOOLS if tools is None else tools
        self.protocol_version = protocol_version
        self.expire_after_init = expire_after_init
        self.error_on = error_on
        self.timeout_on = timeout_on
        self.sse_extra_events = sse_extra_events
        self.sse_trailing_events = sse_trailing_events
        self.empty_sse_on = empty_sse_on
        self.bad_jsonrpc_on = bad_jsonrpc_on
        self.calls: list[tuple[str, dict[str, str]]] = []  # (method, lowercased headers)
        self.initialized = False
        self.deleted = False

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            self.deleted = True
            return httpx.Response(200)
        body = json.loads(request.content)
        method = body.get("method")
        self.calls.append((method, {k.lower(): v for k, v in request.headers.items()}))
        if "id" not in body:
            if method == "notifications/initialized":
                self.initialized = True
            return httpx.Response(202)
        request_id = body["id"]
        if self.timeout_on is not None and method == self.timeout_on:
            raise httpx.ReadTimeout("simulated timeout", request=request)
        if self.expire_after_init and method != "initialize":
            return httpx.Response(404)
        if self.error_on is not None and method == self.error_on:
            return self._respond(request_id, method, error={"code": -32000, "message": "boom"})
        if self.empty_sse_on is not None and method == self.empty_sse_on:
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=b"")
        if self.bad_jsonrpc_on is not None and method == self.bad_jsonrpc_on:
            return self._respond(request_id, method, result=self._result_for(method), jsonrpc="1.0")
        params = body.get("params", {})
        return self._respond(request_id, method, result=self._result_for(method, params))

    def _result_for(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        params = params or {}
        if method == "initialize":
            return {
                "protocolVersion": self.protocol_version,
                "capabilities": {},
                "serverInfo": {"name": "fake", "version": "1.0"},
            }
        if method == "tools/list":
            if self.paginate and params.get("cursor") is None:
                return {"tools": self.tools, "nextCursor": "tools-page-2"}
            if self.paginate:
                return {"tools": [{**self.tools[0], "name": "search_page_2"}]}
            return {"tools": self.tools}
        if method == "tools/call":
            return {"content": [{"type": "text", "text": "ok"}], "isError": False}
        if method == "resources/list":
            return {"resources": [{"uri": "file://x", "name": "x"}]}
        if method == "resources/read":
            return {"contents": [{"uri": "file://x", "text": "hi"}]}
        return {}

    def _respond(
        self,
        request_id: int,
        method: str,
        *,
        result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
        jsonrpc: str = "2.0",
    ) -> httpx.Response:
        payload: dict[str, Any] = {"jsonrpc": jsonrpc, "id": request_id}
        if error is not None:
            payload["error"] = error
        else:
            payload["result"] = result
        headers: dict[str, str] = {}
        if method == "initialize" and self.session_id is not None:
            headers[MCP_SESSION_ID_HEADER] = self.session_id
        if self.sse:
            headers["content-type"] = self.sse_content_type
            # Optionally emit server->client notification events (no matching id)
            # before the response, to exercise the multi-event skip path.
            events: list[dict[str, Any]] = []
            if self.sse_extra_events and method != "initialize":
                events.extend(self.sse_extra_events)
            events.append(payload)
            # Optionally keep the stream open with events AFTER the response, to verify
            # the client returns on the response without waiting for / choking on them.
            if self.sse_trailing_events and method != "initialize":
                events.extend(self.sse_trailing_events)
            if self.fold_sse and method != "initialize":
                # Fold each event's JSON across multiple CRLF `data:` lines (per the
                # SSE spec, multiple data lines are joined with "\n").
                blocks = []
                for event in events:
                    data = "".join(
                        f"data: {line}\r\n" for line in json.dumps(event, indent=2).split("\n")
                    )
                    blocks.append(f"event: message\r\n{data}\r\n")
                body = "".join(blocks).encode()
            else:
                body = "".join(
                    f"event: message\ndata: {json.dumps(event)}\n\n" for event in events
                ).encode()
            return httpx.Response(200, headers=headers, content=body)
        return httpx.Response(200, headers=headers, json=payload)

    def headers_for(self, method: str) -> dict[str, str]:
        return next(headers for call_method, headers in self.calls if call_method == method)


def _server_spec(**overrides: Any) -> McpServerSpec:
    overrides.setdefault("name", "remote")
    overrides.setdefault("url", "https://mcp.example/rpc")
    return McpServerSpec(**overrides)


def test_http_json_transport_lists_and_calls_tools() -> None:
    server = FakeMcpHttpServer()

    async def run():
        session = await HttpMcpClient(transport=server.transport).connect(_server_spec())
        try:
            tools = await session.list_tools()
            result = await session.call_tool("search", {"q": "cayu"})
            resources = await session.list_resources()
            resource = await session.read_resource("file://x")
            return session.initialize_result, tools, result, resources, resource
        finally:
            await session.close()

    init, tools, result, resources, resource = asyncio.run(run())
    assert init.protocol_version == MCP_PROTOCOL_VERSION
    assert init.server_name == "fake"
    assert len(tools) == 1
    assert tools[0].name == "search"
    assert tools[0].annotations["mcp_server"] == "remote"
    assert result.is_error is False
    assert result.content == [{"type": "text", "text": "ok"}]
    assert resources[0].uri == "file://x"
    assert resource.contents[0]["text"] == "hi"
    assert server.initialized is True


def test_http_sse_transport_returns_matching_response() -> None:
    server = FakeMcpHttpServer(sse=True)

    async def run():
        session = await HttpMcpClient(transport=server.transport).connect(_server_spec())
        try:
            return await session.list_tools(), await session.call_tool("search", {})
        finally:
            await session.close()

    tools, result = asyncio.run(run())
    assert tools[0].name == "search"
    assert result.content[0]["text"] == "ok"


def test_http_sse_content_type_is_case_insensitive() -> None:
    # Media types are case-insensitive; a server may reply "Text/Event-Stream".
    server = FakeMcpHttpServer(sse=True, sse_content_type="Text/Event-Stream")

    async def run():
        session = await HttpMcpClient(transport=server.transport).connect(_server_spec())
        try:
            return await session.list_tools(), await session.call_tool("search", {})
        finally:
            await session.close()

    tools, result = asyncio.run(run())
    assert tools[0].name == "search"
    assert result.content[0]["text"] == "ok"


def test_http_sse_skips_non_matching_events() -> None:
    # Long-running shape: a progress notification (no matching id) precedes the
    # JSON-RPC response in the same SSE stream; the response must still be returned.
    server = FakeMcpHttpServer(
        sse=True,
        sse_extra_events=[
            {"jsonrpc": "2.0", "method": "notifications/progress", "params": {"pct": 50}}
        ],
    )

    async def run():
        session = await HttpMcpClient(transport=server.transport).connect(_server_spec())
        try:
            return await session.list_tools(), await session.call_tool("search", {})
        finally:
            await session.close()

    tools, result = asyncio.run(run())
    assert tools[0].name == "search"
    assert result.content[0]["text"] == "ok"


def test_http_empty_sse_raises() -> None:
    server = FakeMcpHttpServer(empty_sse_on="tools/list")

    async def run():
        session = await HttpMcpClient(transport=server.transport).connect(_server_spec())
        try:
            await session.list_tools()
        finally:
            await session.close()

    with pytest.raises(McpProtocolError, match="did not contain"):
        asyncio.run(run())


def test_http_rejects_non_2_0_jsonrpc() -> None:
    server = FakeMcpHttpServer(bad_jsonrpc_on="tools/list")

    async def run():
        session = await HttpMcpClient(transport=server.transport).connect(_server_spec())
        try:
            await session.list_tools()
        finally:
            await session.close()

    with pytest.raises(McpProtocolError, match="2.0"):
        asyncio.run(run())


def test_http_sends_session_id_protocol_and_accept_headers() -> None:
    server = FakeMcpHttpServer()

    async def run():
        session = await HttpMcpClient(transport=server.transport).connect(_server_spec())
        try:
            await session.list_tools()
        finally:
            await session.close()

    asyncio.run(run())
    init_headers = server.headers_for("initialize")
    assert init_headers["accept"] == "application/json, text/event-stream"
    assert init_headers["content-type"] == "application/json"
    # Per spec, the protocol-version header and session id are NOT sent on the
    # initialize request — they're learned/negotiated from the initialize response.
    assert MCP_PROTOCOL_VERSION_HEADER not in init_headers
    assert MCP_SESSION_ID_HEADER not in init_headers
    # Subsequent requests carry both the negotiated protocol version and session id.
    list_headers = server.headers_for("tools/list")
    assert list_headers[MCP_SESSION_ID_HEADER] == "sess-1"
    assert list_headers[MCP_PROTOCOL_VERSION_HEADER] == MCP_PROTOCOL_VERSION


def test_http_forwards_custom_headers() -> None:
    server = FakeMcpHttpServer()

    async def run():
        session = await HttpMcpClient(transport=server.transport).connect(
            _server_spec(headers={"authorization": "Bearer T"})
        )
        await session.close()

    asyncio.run(run())
    assert server.headers_for("initialize")["authorization"] == "Bearer T"


def test_http_resolves_secret_headers_through_secret_resolver() -> None:
    server = FakeMcpHttpServer()
    vault = StaticVault({"mcp_token": "Bearer sk-mcp-secret"})

    async def run():
        session = await HttpMcpClient(
            transport=server.transport,
            secret_resolver=vault,
        ).connect(_server_spec(secret_headers={"authorization": SecretRef(name="mcp_token")}))
        await session.close()

    asyncio.run(run())
    assert server.headers_for("initialize")["authorization"] == "Bearer sk-mcp-secret"


def test_http_rejects_secret_env() -> None:
    async def run():
        await HttpMcpClient(secret_resolver=StaticVault({"token": "x"})).connect(
            _server_spec(secret_env={"TOKEN": SecretRef(name="token")})
        )

    with pytest.raises(ValueError, match="secret_env"):
        asyncio.run(run())


def test_http_404_raises_session_expired() -> None:
    server = FakeMcpHttpServer(expire_after_init=True)

    async def run():
        session = await HttpMcpClient(transport=server.transport).connect(_server_spec())
        try:
            await session.list_tools()
        finally:
            await session.close()

    with pytest.raises(McpProtocolError, match="404"):
        asyncio.run(run())


def test_http_404_poisons_session() -> None:
    # After a 404 the session is unusable: a second call raises "closed", not a
    # silent request without the (now dead) session id.
    server = FakeMcpHttpServer(expire_after_init=True)

    async def run():
        session = await HttpMcpClient(transport=server.transport).connect(_server_spec())
        try:
            with pytest.raises(McpProtocolError, match="404"):
                await session.list_tools()
            with pytest.raises(McpProtocolError, match="closed"):
                await session.list_tools()
        finally:
            await session.close()

    asyncio.run(run())


def test_http_sse_rejects_server_initiated_request() -> None:
    # A server->client request (method + id) arriving in the SSE stream must fail the
    # session loudly instead of being silently skipped.
    server = FakeMcpHttpServer(
        sse=True,
        sse_extra_events=[
            {"jsonrpc": "2.0", "id": 999, "method": "sampling/createMessage", "params": {}}
        ],
    )

    async def run():
        session = await HttpMcpClient(transport=server.transport).connect(_server_spec())
        try:
            await session.call_tool("search", {})
        finally:
            await session.close()

    with pytest.raises(McpProtocolError, match="server request"):
        asyncio.run(run())


def test_http_sse_returns_response_before_trailing_events() -> None:
    # The response is returned as soon as it arrives, even though the server keeps
    # streaming notifications after it (the client must not wait for stream close).
    server = FakeMcpHttpServer(
        sse=True,
        sse_trailing_events=[
            {"jsonrpc": "2.0", "method": "notifications/progress", "params": {"done": True}}
        ],
    )

    async def run():
        session = await HttpMcpClient(transport=server.transport).connect(_server_spec())
        try:
            return await session.call_tool("search", {})
        finally:
            await session.close()

    result = asyncio.run(run())
    assert result.content[0]["text"] == "ok"


def test_http_jsonrpc_error_raises() -> None:
    server = FakeMcpHttpServer(error_on="tools/call")

    async def run():
        session = await HttpMcpClient(transport=server.transport).connect(_server_spec())
        try:
            await session.call_tool("search", {})
        finally:
            await session.close()

    with pytest.raises(McpProtocolError, match="failed"):
        asyncio.run(run())


def test_http_protocol_version_mismatch_raises() -> None:
    server = FakeMcpHttpServer(protocol_version="1.0")

    async def run():
        await HttpMcpClient(transport=server.transport).connect(_server_spec())

    with pytest.raises(McpProtocolError, match="unsupported protocol version"):
        asyncio.run(run())


def test_http_accepts_older_supported_protocol_version() -> None:
    # A server that pins an earlier-but-supported revision is accepted rather than
    # refused, and the negotiated version (not cayu's preferred one) is echoed on
    # every subsequent request per the spec.
    server = FakeMcpHttpServer(protocol_version="2025-03-26")

    async def run():
        session = await HttpMcpClient(transport=server.transport).connect(_server_spec())
        try:
            await session.list_tools()
            return session.initialize_result
        finally:
            await session.close()

    init = asyncio.run(run())
    assert init.protocol_version == "2025-03-26"
    assert server.headers_for("tools/list")[MCP_PROTOCOL_VERSION_HEADER] == "2025-03-26"


def test_http_list_tools_follows_next_cursor() -> None:
    server = FakeMcpHttpServer(paginate=True)

    async def run():
        session = await HttpMcpClient(transport=server.transport).connect(_server_spec())
        try:
            return await session.list_tools()
        finally:
            await session.close()

    tools = asyncio.run(run())
    assert [tool.name for tool in tools] == ["search", "search_page_2"]
    # The second tools/list request echoed the server's cursor.
    list_calls = [headers for method, headers in server.calls if method == "tools/list"]
    assert len(list_calls) == 2


def test_http_mcp_client_applies_configured_page_limit() -> None:
    server = FakeMcpHttpServer(paginate=True)

    async def run() -> None:
        session = await HttpMcpClient(
            transport=server.transport,
            max_list_pages=1,
        ).connect(_server_spec())
        try:
            with pytest.raises(McpProtocolError, match=r"tools/list.*max_list_pages=1"):
                await session.list_tools()
        finally:
            await session.close()

    asyncio.run(run())
    list_calls = [method for method, _headers in server.calls if method == "tools/list"]
    assert list_calls == ["tools/list"]


def test_http_close_sends_delete_and_is_idempotent() -> None:
    server = FakeMcpHttpServer()

    async def run():
        session = await HttpMcpClient(transport=server.transport).connect(_server_spec())
        await session.close()
        await session.close()
        return server.deleted

    assert asyncio.run(run()) is True


def test_http_rejects_secret_headers() -> None:
    server = FakeMcpHttpServer()

    async def run():
        await HttpMcpClient(transport=server.transport).connect(
            _server_spec(secret_headers={"authorization": SecretRef(name="token")})
        )

    with pytest.raises(ValueError, match="secret_headers"):
        asyncio.run(run())


def test_http_toolset_end_to_end() -> None:
    server = FakeMcpHttpServer()

    async def run():
        toolset = await connect_mcp_toolset(
            _server_spec(), client=HttpMcpClient(transport=server.transport)
        )
        try:
            names = [tool.spec.name for tool in toolset.tools]
            result = await toolset.call_tool("search", {"q": "x"})
            return names, result
        finally:
            await toolset.close()

    names, result = asyncio.run(run())
    assert names == ["mcp__remote__search"]
    assert result.is_error is False


def test_default_client_for_picks_transport_by_spec() -> None:
    assert isinstance(_default_client_for(_server_spec()), HttpMcpClient)
    assert isinstance(
        _default_client_for(McpServerSpec(name="local", command=["mcp-server"])),
        StdioMcpClient,
    )


def test_http_timeout_defaults_to_120s() -> None:
    server = FakeMcpHttpServer()

    async def run():
        session = await HttpMcpClient(transport=server.transport).connect(_server_spec())
        assert isinstance(session, HttpMcpSession)
        try:
            return session._http.timeout.read, session._http.timeout.connect
        finally:
            await session.close()

    read_timeout, connect_timeout = asyncio.run(run())
    assert read_timeout == DEFAULT_HTTP_MCP_TIMEOUT_S == 120.0
    assert connect_timeout == DEFAULT_HTTP_MCP_CONNECT_TIMEOUT_S == 10.0


def test_http_per_server_timeout_override_via_metadata() -> None:
    server = FakeMcpHttpServer()

    async def run():
        session = await HttpMcpClient(transport=server.transport).connect(
            _server_spec(metadata={"timeout": 5})
        )
        assert isinstance(session, HttpMcpSession)
        try:
            return session._http.timeout.read
        finally:
            await session.close()

    assert asyncio.run(run()) == 5.0


def test_http_resolve_transport_config_metadata_overrides_client_defaults() -> None:
    client = HttpMcpClient(timeout_s=120.0, proxy="http://default-proxy:8080")
    # No metadata -> client defaults.
    assert client._resolve_transport_config(_server_spec()) == (
        120.0,
        DEFAULT_HTTP_MCP_CONNECT_TIMEOUT_S,
        "http://default-proxy:8080",
    )
    # metadata overrides timeout and proxy.
    assert client._resolve_transport_config(
        _server_spec(metadata={"timeout": 30, "proxy": "http://corp-proxy:9090"})
    ) == (30.0, DEFAULT_HTTP_MCP_CONNECT_TIMEOUT_S, "http://corp-proxy:9090")


def test_http_timeout_raises_timeout_error() -> None:
    server = FakeMcpHttpServer(timeout_on="tools/list")

    async def run():
        session = await HttpMcpClient(transport=server.transport).connect(_server_spec())
        try:
            await session.list_tools()
        finally:
            await session.close()

    with pytest.raises(TimeoutError):
        asyncio.run(run())


def test_http_rejects_invalid_metadata_transport_config() -> None:
    client = HttpMcpClient()
    with pytest.raises(ValueError, match="metadata.timeout"):
        client._resolve_transport_config(_server_spec(metadata={"timeout": -1}))
    with pytest.raises(ValueError, match="metadata.proxy"):
        client._resolve_transport_config(_server_spec(metadata={"proxy": 123}))


@pytest.mark.parametrize("client_type", [HttpMcpClient, StdioMcpClient])
@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    [
        ("max_list_pages", True, TypeError),
        ("max_list_pages", 0, ValueError),
        ("max_list_items", 1.5, TypeError),
        ("max_list_items", -1, ValueError),
    ],
)
def test_mcp_clients_reject_invalid_list_limits(client_type, field, value, error_type) -> None:
    with pytest.raises(error_type, match=field):
        client_type(**{field: value})


def test_http_client_accepts_tls_verify_options() -> None:
    assert HttpMcpClient().verify is None
    assert HttpMcpClient(verify=False).verify is False
    assert HttpMcpClient(verify="/etc/ssl/corp-ca.pem").verify == "/etc/ssl/corp-ca.pem"
    bad_verify: Any = 123
    with pytest.raises(TypeError, match="verify"):
        HttpMcpClient(verify=bad_verify)


def test_connect_mcp_toolset_auto_selects_http_for_url(monkeypatch) -> None:
    # The no-client branch of McpToolset.connect must route a url spec to HTTP.
    server = FakeMcpHttpServer()
    monkeypatch.setattr(
        "cayu.mcp.tools._default_client_for",
        lambda spec: HttpMcpClient(transport=server.transport),
    )

    async def run():
        toolset = await connect_mcp_toolset(_server_spec())  # no explicit client
        try:
            return [tool.spec.name for tool in toolset.tools]
        finally:
            await toolset.close()

    assert asyncio.run(run()) == ["mcp__remote__search"]


def test_http_no_session_id_server() -> None:
    # A stateless server never issues Mcp-Session-Id: later requests omit it and
    # close() sends no DELETE.
    server = FakeMcpHttpServer(session_id=None)

    async def run():
        session = await HttpMcpClient(transport=server.transport).connect(_server_spec())
        try:
            await session.list_tools()
        finally:
            await session.close()

    asyncio.run(run())
    assert MCP_SESSION_ID_HEADER not in server.headers_for("tools/list")
    assert server.deleted is False


def test_http_sse_folds_multiline_data_and_crlf() -> None:
    # A response folded across multiple CRLF `data:` lines must still be parsed.
    server = FakeMcpHttpServer(sse=True, fold_sse=True)

    async def run():
        session = await HttpMcpClient(transport=server.transport).connect(_server_spec())
        try:
            return await session.call_tool("search", {})
        finally:
            await session.close()

    result = asyncio.run(run())
    assert result.content[0]["text"] == "ok"
