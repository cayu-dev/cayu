from __future__ import annotations

import asyncio
import gc
import json
from typing import Any

import httpx
import pytest
from tests.provider_traceback_assertions import is_cayu_source_filename

from cayu import (
    DEFAULT_HTTP_MCP_CONNECT_TIMEOUT_S,
    DEFAULT_HTTP_MCP_TIMEOUT_S,
    AgentSpec,
    CayuApp,
    HttpMcpClient,
    McpIdleTimeoutError,
    McpManifestPolicy,
    McpManifestPolicyAction,
    McpProtocolError,
    McpServerSpec,
    McpToolset,
    McpToolsetRefreshBlocked,
    McpToolsetRefreshState,
    McpTransportLimits,
    StdioMcpClient,
    connect_mcp_toolset,
)
from cayu.mcp._jsonrpc import MCP_PROTOCOL_VERSION
from cayu.mcp.http import (
    _MAX_ERROR_BODY_CHARS,
    MCP_PROTOCOL_VERSION_HEADER,
    MCP_SESSION_ID_HEADER,
    HttpMcpSession,
    _decode_jsonrpc,
)
from cayu.mcp.tools import _default_client_for
from cayu.vaults import REDACTED_SECRET, SecretRedactor, SecretRef, StaticVault

_DEFAULT_TOOLS = [
    {"name": "search", "description": "Search the web.", "inputSchema": {"type": "object"}},
]
_DEFAULT_RESOURCES = [{"uri": "file://x", "name": "x"}]


class FakeMcpHttpServer:
    """In-memory Streamable HTTP MCP server backing an ``httpx.MockTransport``."""

    def __init__(
        self,
        *,
        sse: bool = False,
        session_id: str | None = "sess-1",
        tools: list[dict[str, Any]] | None = None,
        resources: list[dict[str, Any]] | None = None,
        protocol_version: str = MCP_PROTOCOL_VERSION,
        expire_after_init: bool = False,
        error_on: str | None = None,
        error_message: str = "boom",
        instructions: str | None = None,
        timeout_on: str | None = None,
        sse_content_type: str = "text/event-stream",
        sse_extra_events: list[dict[str, Any]] | None = None,
        sse_extra_events_on: str | None = None,
        sse_trailing_events: list[dict[str, Any]] | None = None,
        empty_sse_on: str | None = None,
        bad_jsonrpc_on: str | None = None,
        raw_json_on: str | None = None,
        raw_json_document: str | None = None,
        invalid_portable_result_on: str | None = None,
        invalid_portable_canary: str = "",
        fold_sse: bool = False,
        paginate: bool = False,
        tools_list_changed: bool = False,
        get_sse_events: list[dict[str, Any]] | None = None,
        get_sse_event_ids: list[str | None] | None = None,
        get_status_after_first: int = 405,
        tools_before_second_get: list[dict[str, Any]] | None = None,
    ) -> None:
        self.sse = sse
        self.paginate = paginate
        self.sse_content_type = sse_content_type
        self.fold_sse = fold_sse
        self.session_id = session_id
        self.tools = _DEFAULT_TOOLS if tools is None else tools
        self.resources = _DEFAULT_RESOURCES if resources is None else resources
        self.protocol_version = protocol_version
        self.expire_after_init = expire_after_init
        self.error_on = error_on
        self.error_message = error_message
        self.instructions = instructions
        self.timeout_on = timeout_on
        self.sse_extra_events = sse_extra_events
        self.sse_extra_events_on = sse_extra_events_on
        self.sse_trailing_events = sse_trailing_events
        self.empty_sse_on = empty_sse_on
        self.bad_jsonrpc_on = bad_jsonrpc_on
        self.raw_json_on = raw_json_on
        self.raw_json_document = raw_json_document
        self.invalid_portable_result_on = invalid_portable_result_on
        self.invalid_portable_canary = invalid_portable_canary
        self.tools_list_changed = tools_list_changed
        self.get_sse_events = get_sse_events
        self.get_sse_event_ids = get_sse_event_ids
        self.get_status_after_first = get_status_after_first
        self.tools_before_second_get = tools_before_second_get
        self.get_calls = 0
        self.get_headers: list[dict[str, str]] = []
        self.calls: list[tuple[str, dict[str, str]]] = []  # (method, lowercased headers)
        self.tool_call_names: list[str] = []
        self.initialized = False
        self.deleted = False

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            self.deleted = True
            return httpx.Response(200)
        if request.method == "GET":
            self.get_calls += 1
            self.get_headers.append({key.lower(): value for key, value in request.headers.items()})
            if self.get_sse_events is None or self.get_calls > 1:
                if self.get_calls == 2 and self.tools_before_second_get is not None:
                    self.tools = self.tools_before_second_get
                return httpx.Response(self.get_status_after_first)
            event_ids = self.get_sse_event_ids or [None] * len(self.get_sse_events)
            if len(event_ids) != len(self.get_sse_events):
                raise AssertionError("GET/SSE event ids must match the event count")
            blocks = []
            for event, event_id in zip(self.get_sse_events, event_ids, strict=True):
                id_line = "" if event_id is None else f"id: {event_id}\n"
                blocks.append(f"event: message\n{id_line}data: {json.dumps(event)}\n\n")
            body = "".join(blocks).encode()
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=body,
            )
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
            return self._respond(
                request_id,
                method,
                error={"code": -32000, "message": self.error_message},
            )
        if self.empty_sse_on is not None and method == self.empty_sse_on:
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=b"")
        if self.raw_json_on is not None and method == self.raw_json_on:
            if self.raw_json_document is None:
                raise AssertionError("raw_json_document is required with raw_json_on")
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                content=self.raw_json_document.encode(),
            )
        if self.bad_jsonrpc_on is not None and method == self.bad_jsonrpc_on:
            return self._respond(request_id, method, result=self._result_for(method), jsonrpc="1.0")
        params = body.get("params", {})
        if method == "tools/call":
            self.tool_call_names.append(params.get("name"))
        return self._respond(request_id, method, result=self._result_for(method, params))

    def _result_for(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        params = params or {}
        if method == self.invalid_portable_result_on:
            invalid_text = f"{self.invalid_portable_canary}\x00"
            if method == "tools/call":
                return {"content": [{"type": "text", "text": invalid_text}]}
            if method == "resources/read":
                return {"contents": [{"text": invalid_text}]}
            raise AssertionError(f"Unsupported invalid portable result method: {method}")
        if method == "initialize":
            result = {
                "protocolVersion": self.protocol_version,
                "capabilities": (
                    {"tools": {"listChanged": True}} if self.tools_list_changed else {}
                ),
                "serverInfo": {"name": "fake", "version": "1.0"},
            }
            if self.instructions is not None:
                result["instructions"] = self.instructions
            return result
        if method == "tools/list":
            if self.paginate and params.get("cursor") is None:
                return {"tools": self.tools, "nextCursor": "tools-page-2"}
            if self.paginate:
                return {"tools": [{**self.tools[0], "name": "search_page_2"}]}
            return {"tools": self.tools}
        if method == "tools/call":
            return {"content": [{"type": "text", "text": "ok"}], "isError": False}
        if method == "resources/list":
            return {"resources": self.resources}
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
            if (
                self.sse_extra_events
                and method != "initialize"
                and (self.sse_extra_events_on is None or self.sse_extra_events_on == method)
            ):
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


class BlockingMcpServerMessageStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.closed = asyncio.Event()
        self._release = asyncio.Event()

    async def __aiter__(self):
        self.started.set()
        await self._release.wait()
        if False:  # pragma: no cover - retain the async-iterator type
            yield b""

    async def aclose(self) -> None:
        self.closed.set()
        self._release.set()


class BlockingGetMcpHttpServer(FakeMcpHttpServer):
    def __init__(self) -> None:
        super().__init__(tools_list_changed=True)
        self.stream = BlockingMcpServerMessageStream()

    def _handle(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            self.get_calls += 1
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=self.stream,
            )
        return super()._handle(request)


class HealthyKeepaliveMcpServerMessageStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.closed = asyncio.Event()
        self.chunks_sent = 0

    async def __aiter__(self):
        self.started.set()
        while True:
            await asyncio.sleep(0.005)
            # Complete, individually bounded SSE comments provide continuous
            # transport activity without carrying JSON-RPC authority.
            self.chunks_sent += 1
            yield b":" + (b"x" * 252) + b"\n\n"

    async def aclose(self) -> None:
        self.closed.set()


class HealthyKeepaliveGetMcpHttpServer(FakeMcpHttpServer):
    def __init__(self) -> None:
        super().__init__(tools_list_changed=True)
        self.stream = HealthyKeepaliveMcpServerMessageStream()

    def _handle(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            self.get_calls += 1
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=self.stream,
            )
        return super()._handle(request)


class FiniteLargeGetMcpHttpServer(FakeMcpHttpServer):
    def __init__(self) -> None:
        super().__init__(tools_list_changed=True)
        self.listener_body = (b":" + (b"x" * 60) + b"\n\n") * 20

    def _handle(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            self.get_calls += 1
            if self.get_calls == 1:
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=self.listener_body,
                )
            return httpx.Response(405)
        return super()._handle(request)


class ReplayCursorFailureMcpHttpServer(FakeMcpHttpServer):
    def __init__(self, cursor: str) -> None:
        super().__init__(tools_list_changed=True)
        self.cursor = cursor

    def _handle(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            self.get_calls += 1
            self.get_headers.append({key.lower(): value for key, value in request.headers.items()})
            if self.get_calls == 1:
                event = {
                    "jsonrpc": "2.0",
                    "method": "notifications/resources/list_changed",
                }
                body = f"id: {self.cursor}\ndata: {json.dumps(event)}\n\n".encode()
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=body,
                )
            raise RuntimeError(
                f"extension echoed replay authority {request.headers['last-event-id']}"
            )
        return super()._handle(request)


class StartupChangingGetMcpHttpServer(BlockingGetMcpHttpServer):
    def _handle(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            self.tools = [
                *_DEFAULT_TOOLS,
                {
                    "name": "summarize",
                    "description": "Summarize text.",
                    "inputSchema": {"type": "object"},
                },
            ]
        return super()._handle(request)


class CancellationResistantMcpServerMessageStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.cancelled = asyncio.Event()
        self.closed = asyncio.Event()
        self._release = asyncio.Event()

    async def __aiter__(self):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            await self._release.wait()
            raise RuntimeError("cancellation-resistant listener read failed") from None
        if False:  # pragma: no cover - retain the async-iterator type
            yield b""

    async def aclose(self) -> None:
        self.closed.set()
        self._release.set()


class CancellationResistantGetMcpHttpServer(FakeMcpHttpServer):
    def __init__(self) -> None:
        super().__init__(tools_list_changed=True)
        self.stream = CancellationResistantMcpServerMessageStream()

    def _handle(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            self.get_calls += 1
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=self.stream,
            )
        return super()._handle(request)


class RotatingGetMcpHttpServer(FakeMcpHttpServer):
    def __init__(self, *, retryable_failures: int = 0) -> None:
        super().__init__(tools_list_changed=True)
        self.retryable_failures = retryable_failures

    def _handle(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            self.get_calls += 1
            self.get_headers.append({key.lower(): value for key, value in request.headers.items()})
            if self.get_calls <= self.retryable_failures:
                return httpx.Response(503)
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=b"",
            )
        return super()._handle(request)


def _server_spec(**overrides: Any) -> McpServerSpec:
    overrides.setdefault("name", "remote")
    overrides.setdefault("url", "https://mcp.example/rpc")
    return McpServerSpec(**overrides)


async def _wait_for_http_mcp_tools(app: CayuApp, *tool_names: str) -> None:
    async with asyncio.timeout(2):
        while tuple(app.get_agent("assistant").tools) != tool_names:
            await asyncio.sleep(0.001)


async def _wait_for_http_mcp_refresh_state(
    toolset: McpToolset,
    state: McpToolsetRefreshState,
) -> None:
    async with asyncio.timeout(2):
        while toolset.refresh_state is not state:
            await asyncio.sleep(0.001)


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


@pytest.mark.parametrize(
    ("method", "error_match"),
    [
        ("tools/call", "tools/call result contained invalid data"),
        ("resources/read", "resources/read result contained invalid data"),
    ],
)
def test_http_invalid_result_models_raise_detached_protocol_errors(
    method: str,
    error_match: str,
) -> None:
    secret = f"mcp-http-{method}-portable-canary"
    server = FakeMcpHttpServer(
        invalid_portable_result_on=method,
        invalid_portable_canary=secret,
    )

    async def run() -> McpProtocolError:
        session = await HttpMcpClient(
            transport=server.transport,
            secret_resolver=StaticVault({"token": secret}),
        ).connect(_server_spec(secret_headers={"authorization": SecretRef(name="token")}))
        try:
            with pytest.raises(McpProtocolError, match=error_match) as excinfo:
                if method == "tools/call":
                    await session.call_tool("search", {})
                else:
                    await session.read_resource("file://x")
            return excinfo.value
        finally:
            await session.close()

    error = asyncio.run(run())

    assert server.deleted is True
    _assert_cayu_traceback_does_not_retain_secret(error, secret)


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


def test_http_interleaved_list_changed_notification_uses_shared_refresh_core() -> None:
    server = FakeMcpHttpServer(
        sse=True,
        tools_list_changed=True,
        sse_extra_events_on="resources/list",
        sse_extra_events=[
            {
                "jsonrpc": "2.0",
                "method": "notifications/tools/list_changed",
                "params": {"ignored": "server payload"},
            }
        ],
    )

    async def run():
        toolset = await connect_mcp_toolset(
            _server_spec(connection_id="http-interleaved-list-changed"),
            client=HttpMcpClient(transport=server.transport),
        )
        stale_adapter = toolset.tools[0]
        app = CayuApp(enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            mcp_toolsets=(toolset,),
        )
        await _wait_for_http_mcp_refresh_state(toolset, McpToolsetRefreshState.READY)
        server.tools = [
            *_DEFAULT_TOOLS,
            {
                "name": "summarize",
                "description": "Summarize text.",
                "inputSchema": {"type": "object"},
            },
        ]
        try:
            await toolset.session.list_resources()
            assert toolset.refresh_state in {
                McpToolsetRefreshState.DIRTY,
                McpToolsetRefreshState.REFRESHING,
                McpToolsetRefreshState.READY,
            }
            await _wait_for_http_mcp_tools(
                app,
                "mcp__remote__search",
                "mcp__remote__summarize",
            )
            return stale_adapter._dispatch_authority_is_current()
        finally:
            await toolset.close()

    assert asyncio.run(run()) is False


def test_http_get_sse_list_changed_listener_uses_shared_refresh_core() -> None:
    server = FakeMcpHttpServer(
        tools_list_changed=True,
        get_sse_events=[
            {
                "jsonrpc": "2.0",
                "method": "notifications/tools/list_changed",
            }
        ],
    )

    async def run():
        toolset = await connect_mcp_toolset(
            _server_spec(connection_id="http-get-list-changed"),
            client=HttpMcpClient(transport=server.transport),
        )
        app = CayuApp(enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            mcp_toolsets=(toolset,),
        )
        server.tools = [
            *_DEFAULT_TOOLS,
            {
                "name": "summarize",
                "description": "Summarize text.",
                "inputSchema": {"type": "object"},
            },
        ]
        try:
            await _wait_for_http_mcp_tools(
                app,
                "mcp__remote__search",
                "mcp__remote__summarize",
            )
            return server.get_calls
        finally:
            await toolset.close()

    assert asyncio.run(run()) >= 1


def test_http_get_sse_disconnect_fences_until_reconnect_reconciles_catalogue() -> None:
    refreshed_tools = [
        *_DEFAULT_TOOLS,
        {
            "name": "summarize",
            "description": "Summarize text.",
            "inputSchema": {"type": "object"},
        },
    ]
    server = FakeMcpHttpServer(
        tools_list_changed=True,
        get_sse_events=[],
        tools_before_second_get=refreshed_tools,
    )

    async def run():
        toolset = await connect_mcp_toolset(
            _server_spec(connection_id="http-listener-disconnect"),
            client=HttpMcpClient(transport=server.transport),
        )
        stale_adapter = toolset.tools[0]
        app = CayuApp(enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            mcp_toolsets=(toolset,),
        )
        try:
            await asyncio.sleep(0)
            assert toolset.refresh_state is McpToolsetRefreshState.DIRTY
            assert stale_adapter._dispatch_authority_is_current() is False
            async with asyncio.timeout(2):
                while tuple(app.get_agent("assistant").tools) != (
                    "mcp__remote__search",
                    "mcp__remote__summarize",
                ):
                    await asyncio.sleep(0.001)
            list_calls = sum(method == "tools/list" for method, _ in server.calls)
            return (
                toolset.refresh_state,
                list_calls,
                server.get_calls,
                stale_adapter._dispatch_authority_is_current(),
            )
        finally:
            await toolset.close()

    state, list_calls, get_calls, stale_is_current = asyncio.run(run())

    assert state is McpToolsetRefreshState.READY
    assert list_calls == 2
    assert get_calls == 2
    assert stale_is_current is False


def test_http_get_sse_reconnect_sends_the_last_safe_event_id() -> None:
    server = FakeMcpHttpServer(
        tools_list_changed=True,
        get_sse_events=[
            {
                "jsonrpc": "2.0",
                "method": "notifications/resources/list_changed",
            }
        ],
        get_sse_event_ids=["catalogue-cursor-7"],
    )

    async def run():
        toolset = await connect_mcp_toolset(
            _server_spec(connection_id="http-listener-replay-cursor"),
            client=HttpMcpClient(transport=server.transport),
        )
        app = CayuApp(enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            mcp_toolsets=(toolset,),
        )
        try:
            async with asyncio.timeout(2):
                while server.get_calls < 2:
                    await asyncio.sleep(0.001)
            return server.get_headers
        finally:
            await toolset.close()

    get_headers = asyncio.run(run())

    assert "last-event-id" not in get_headers[0]
    assert get_headers[1]["last-event-id"] == "catalogue-cursor-7"


def test_http_listener_failure_does_not_publish_private_replay_cursor() -> None:
    cursor = "private-replay-cursor-canary"
    server = ReplayCursorFailureMcpHttpServer(cursor)

    async def run() -> McpProtocolError:
        toolset = await connect_mcp_toolset(
            _server_spec(connection_id="http-listener-private-replay-cursor"),
            client=HttpMcpClient(transport=server.transport),
        )
        session = toolset.session
        assert isinstance(session, HttpMcpSession)
        app = CayuApp(enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            mcp_toolsets=(toolset,),
        )
        try:
            async with asyncio.timeout(2):
                while not session._closed:
                    await asyncio.sleep(0.001)
            with pytest.raises(McpProtocolError) as exc_info:
                await app.refresh_mcp_toolset(toolset)
            return exc_info.value
        finally:
            await toolset.close()

    error = asyncio.run(run())

    assert server.get_calls == 2
    assert server.get_headers[1]["last-event-id"] == cursor
    assert REDACTED_SECRET in str(error)
    _assert_cayu_traceback_does_not_retain_secret(error, cursor)


def test_http_get_sse_reconnect_backoff_resets_after_each_valid_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("cayu.mcp.http._MCP_SERVER_LISTENER_INITIAL_RECONNECT_DELAY_S", 0.02)
    monkeypatch.setattr("cayu.mcp.http._MCP_SERVER_LISTENER_MAX_RECONNECT_DELAY_S", 0.32)
    reconnect_delays: list[float] = []

    async def record_reconnect_delay(delay_s: float) -> None:
        reconnect_delays.append(delay_s)
        await asyncio.sleep(0)

    monkeypatch.setattr(
        "cayu.mcp.http._sleep_before_mcp_server_listener_reconnect",
        record_reconnect_delay,
    )
    server = RotatingGetMcpHttpServer(retryable_failures=3)

    async def run():
        toolset = await connect_mcp_toolset(
            _server_spec(connection_id="http-listener-rotating-stream"),
            client=HttpMcpClient(transport=server.transport),
        )
        app = CayuApp(enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            mcp_toolsets=(toolset,),
        )
        try:
            async with asyncio.timeout(2):
                while server.get_calls < 6:
                    await asyncio.sleep(0.001)
        finally:
            await toolset.close()

    asyncio.run(run())

    # Consecutive connection failures back off. A valid stream then proves
    # connectivity even when the server rotates it immediately, so routine
    # rotations restart from the initial delay instead of inheriting failures.
    assert reconnect_delays[:5] == pytest.approx((0.02, 0.04, 0.08, 0.02, 0.02))


def test_http_healthy_listener_outlives_finite_rpc_limits_without_polling() -> None:
    server = HealthyKeepaliveGetMcpHttpServer()
    limits = McpTransportLimits(
        max_message_bytes=1024,
        max_response_bytes=1024,
        idle_timeout_s=1,
        total_call_timeout_s=0.2,
    )

    async def run() -> tuple[bool, McpToolsetRefreshState, bool, int, int, bool, int]:
        toolset = await connect_mcp_toolset(
            _server_spec(connection_id="http-listener-healthy-keepalive"),
            client=HttpMcpClient(
                transport=server.transport,
                transport_limits=limits,
            ),
        )
        session = toolset.session
        assert isinstance(session, HttpMcpSession)
        adapter = toolset.tools[0]
        app = CayuApp(enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            mcp_toolsets=(toolset,),
        )
        try:
            await asyncio.wait_for(server.stream.started.wait(), timeout=1)
            await _wait_for_http_mcp_refresh_state(toolset, McpToolsetRefreshState.READY)
            await asyncio.sleep(0.5)
            return (
                session._closed,
                toolset.refresh_state,
                adapter._dispatch_authority_is_current(),
                server.get_calls,
                sum(method == "tools/list" for method, _ in server.calls),
                server.stream.closed.is_set(),
                server.stream.chunks_sent,
            )
        finally:
            await toolset.close()

    closed, state, current, get_calls, list_calls, stream_closed, chunks_sent = asyncio.run(run())

    assert closed is False
    assert state is McpToolsetRefreshState.READY
    assert current is True
    assert get_calls == 1
    assert list_calls == 2
    assert stream_closed is False
    assert chunks_sent * 255 > limits.max_response_bytes


def test_http_finite_listener_body_uses_event_limits_not_rpc_aggregate_limit() -> None:
    server = FiniteLargeGetMcpHttpServer()
    limits = McpTransportLimits(
        max_message_bytes=1024,
        max_response_bytes=1024,
        idle_timeout_s=1,
        total_call_timeout_s=0.2,
    )

    async def run() -> tuple[bool, McpToolsetRefreshState, bool, int, int]:
        toolset = await connect_mcp_toolset(
            _server_spec(connection_id="http-listener-finite-large-body"),
            client=HttpMcpClient(
                transport=server.transport,
                transport_limits=limits,
            ),
        )
        session = toolset.session
        assert isinstance(session, HttpMcpSession)
        adapter = toolset.tools[0]
        app = CayuApp(enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            mcp_toolsets=(toolset,),
        )
        try:
            async with asyncio.timeout(2):
                while (
                    server.get_calls < 2
                    or toolset.refresh_state is not McpToolsetRefreshState.READY
                ):
                    await asyncio.sleep(0.001)
            return (
                session._closed,
                toolset.refresh_state,
                adapter._dispatch_authority_is_current(),
                server.get_calls,
                sum(method == "tools/list" for method, _ in server.calls),
            )
        finally:
            await toolset.close()

    closed, state, current, get_calls, list_calls = asyncio.run(run())

    assert len(server.listener_body) > limits.max_response_bytes
    assert closed is False
    assert state is McpToolsetRefreshState.READY
    assert current is True
    assert get_calls == 2
    assert list_calls == 2


@pytest.mark.parametrize(
    "notification",
    [
        {
            "jsonrpc": "2.0",
            "method": "notifications/resources/list_changed",
            "params": {"ignored": True},
        },
        {
            "jsonrpc": "2.0",
            "method": "notifications/tools/list_changed",
            "params": [],
        },
    ],
)
def test_http_unknown_or_malformed_notification_does_not_start_refresh_work(
    notification: dict[str, Any],
) -> None:
    server = FakeMcpHttpServer(
        sse=True,
        tools_list_changed=True,
        sse_extra_events_on="resources/list",
        sse_extra_events=[notification],
    )

    async def run():
        toolset = await connect_mcp_toolset(
            _server_spec(connection_id="http-unknown-notification"),
            client=HttpMcpClient(transport=server.transport),
        )
        app = CayuApp(enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            mcp_toolsets=(toolset,),
        )
        try:
            await _wait_for_http_mcp_refresh_state(toolset, McpToolsetRefreshState.READY)
            baseline_list_calls = sum(method == "tools/list" for method, _ in server.calls)
            await toolset.session.list_resources()
            await asyncio.sleep(0.1)
            list_calls = sum(method == "tools/list" for method, _ in server.calls)
            return toolset.refresh_state, baseline_list_calls, list_calls
        finally:
            await toolset.close()

    state, baseline_list_calls, list_calls = asyncio.run(run())

    assert state is McpToolsetRefreshState.READY
    # Listener activation performs one reconciliation before async
    # notification ownership becomes dispatchable.
    assert baseline_list_calls == 2
    assert list_calls == baseline_list_calls


def test_http_get_405_keeps_manual_refresh_as_the_non_polling_fallback() -> None:
    server = FakeMcpHttpServer(tools_list_changed=True)

    async def run():
        toolset = await connect_mcp_toolset(
            _server_spec(connection_id="http-listener-unavailable"),
            client=HttpMcpClient(transport=server.transport),
        )
        app = CayuApp(enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            mcp_toolsets=(toolset,),
        )
        async with asyncio.timeout(2):
            while (
                server.get_calls < 1
                or toolset.refresh_state is not McpToolsetRefreshState.READY
                or sum(method == "tools/list" for method, _ in server.calls) < 2
            ):
                await asyncio.sleep(0.001)
        server.tools = [
            *_DEFAULT_TOOLS,
            {
                "name": "summarize",
                "description": "Summarize text.",
                "inputSchema": {"type": "object"},
            },
        ]
        try:
            await asyncio.sleep(0.1)
            before = tuple(app.get_agent("assistant").tools)
            refresh = await app.refresh_mcp_toolset(toolset)
            after = tuple(app.get_agent("assistant").tools)
            return before, after, refresh.status, server.get_calls
        finally:
            await toolset.close()

    before, after, status, get_calls = asyncio.run(run())

    assert before == ("mcp__remote__search",)
    assert after == ("mcp__remote__search", "mcp__remote__summarize")
    assert status == "accepted"
    assert get_calls == 1


def test_http_first_listener_activation_fences_and_reconciles_startup_gap() -> None:
    server = StartupChangingGetMcpHttpServer()

    async def run():
        toolset = await connect_mcp_toolset(
            _server_spec(connection_id="http-listener-startup-gap"),
            client=HttpMcpClient(transport=server.transport),
        )
        stale_adapter = toolset.tools[0]
        app = CayuApp(enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            mcp_toolsets=(toolset,),
        )
        try:
            assert toolset.refresh_state is McpToolsetRefreshState.DIRTY
            assert stale_adapter._dispatch_authority_is_current() is False
            await asyncio.wait_for(server.stream.started.wait(), timeout=1)
            await _wait_for_http_mcp_tools(
                app,
                "mcp__remote__search",
                "mcp__remote__summarize",
            )
            return (
                toolset.refresh_state,
                sum(method == "tools/list" for method, _ in server.calls),
                stale_adapter._dispatch_authority_is_current(),
            )
        finally:
            await toolset.close()

    state, list_calls, stale_is_current = asyncio.run(run())

    assert state is McpToolsetRefreshState.READY
    assert list_calls == 2
    assert stale_is_current is False


def test_http_listener_activation_rolls_back_before_failed_app_registration() -> None:
    first_server = FakeMcpHttpServer(tools_list_changed=True)
    occupied_server = FakeMcpHttpServer()

    async def run():
        first = await connect_mcp_toolset(
            _server_spec(name="first", connection_id="first"),
            client=HttpMcpClient(transport=first_server.transport),
        )
        occupied = await connect_mcp_toolset(
            _server_spec(name="occupied", connection_id="occupied"),
            client=HttpMcpClient(transport=occupied_server.transport),
        )
        occupied_app = CayuApp(enable_logging=False)
        occupied_app.register_agent(
            AgentSpec(name="occupied", model="fake-model"),
            mcp_toolsets=(occupied,),
        )
        rejected_app = CayuApp(enable_logging=False)
        try:
            with pytest.raises(ValueError, match="only one CayuApp"):
                rejected_app.register_agent(
                    AgentSpec(name="rejected", model="fake-model"),
                    mcp_toolsets=(first, occupied),
                )
            await asyncio.sleep(0)
            first_session = first.session
            assert isinstance(first_session, HttpMcpSession)
            return (
                first.refresh_state,
                first._refresh_source.refresh_owner,
                first_session._tools_list_changed_handler,
                first_server.get_calls,
            )
        finally:
            await first.close()
            await occupied.close()

    state, owner, handler, get_calls = asyncio.run(run())

    assert state is McpToolsetRefreshState.READY
    assert owner is None
    assert handler is None
    assert get_calls == 0


def test_http_listener_start_does_not_block_synchronous_multi_agent_registration() -> None:
    server = BlockingGetMcpHttpServer()

    async def run():
        toolset = await connect_mcp_toolset(
            _server_spec(connection_id="http-listener-shared-registration"),
            client=HttpMcpClient(transport=server.transport),
        )
        app = CayuApp(enable_logging=False)
        app.register_agent(
            AgentSpec(name="first", model="fake-model"),
            mcp_toolsets=(toolset,),
        )
        app.register_agent(
            AgentSpec(name="second", model="fake-model"),
            mcp_toolsets=(toolset,),
        )
        try:
            await asyncio.wait_for(server.stream.started.wait(), timeout=1)
            return tuple(app.get_agent("first").tools), tuple(app.get_agent("second").tools)
        finally:
            await toolset.close()

    first_tools, second_tools = asyncio.run(run())

    assert first_tools == ("mcp__remote__search",)
    assert second_tools == first_tools


def test_http_get_sse_server_request_fences_session_without_reconnect() -> None:
    cursor = "private-server-request-event-id-canary"
    server = FakeMcpHttpServer(
        tools_list_changed=True,
        get_sse_events=[
            {
                "jsonrpc": "2.0",
                "id": 91,
                "method": f"sampling/createMessage/{cursor}",
                "params": {},
            }
        ],
        get_sse_event_ids=[cursor],
    )

    async def run():
        toolset = await connect_mcp_toolset(
            _server_spec(connection_id="http-listener-server-request"),
            client=HttpMcpClient(transport=server.transport),
        )
        session = toolset.session
        assert isinstance(session, HttpMcpSession)
        stale_adapter = toolset.tools[0]
        app = CayuApp(enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            mcp_toolsets=(toolset,),
        )
        try:
            async with asyncio.timeout(2):
                while not session._closed:
                    await asyncio.sleep(0.001)
            await asyncio.sleep(0.3)
            with pytest.raises(McpProtocolError, match="server request") as exc_info:
                await app.refresh_mcp_toolset(toolset)
            return (
                toolset.refresh_state,
                stale_adapter._dispatch_authority_is_current(),
                server.get_calls,
                exc_info.value,
            )
        finally:
            await toolset.close()

    state, stale_is_current, get_calls, error = asyncio.run(run())

    assert state in {
        McpToolsetRefreshState.DIRTY,
        McpToolsetRefreshState.QUARANTINED,
    }
    assert stale_is_current is False
    assert get_calls == 1
    assert REDACTED_SECRET in str(error)
    _assert_cayu_traceback_does_not_retain_secret(error, cursor)


def test_http_get_non_sse_success_fences_session_without_reconnect() -> None:
    server = FakeMcpHttpServer(
        tools_list_changed=True,
        get_status_after_first=200,
    )

    async def run():
        toolset = await connect_mcp_toolset(
            _server_spec(connection_id="http-listener-invalid-representation"),
            client=HttpMcpClient(transport=server.transport),
        )
        session = toolset.session
        assert isinstance(session, HttpMcpSession)
        app = CayuApp(enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            mcp_toolsets=(toolset,),
        )
        try:
            async with asyncio.timeout(2):
                while not session._closed:
                    await asyncio.sleep(0.001)
            await asyncio.sleep(0.3)
            with pytest.raises(McpProtocolError, match="did not return text/event-stream"):
                await app.refresh_mcp_toolset(toolset)
            return toolset.refresh_state, server.get_calls
        finally:
            await toolset.close()

    state, get_calls = asyncio.run(run())

    assert state is McpToolsetRefreshState.DIRTY
    assert get_calls == 1


def test_http_get_404_fences_the_refreshable_source() -> None:
    server = FakeMcpHttpServer(
        tools_list_changed=True,
        get_status_after_first=404,
    )

    async def run():
        toolset = await connect_mcp_toolset(
            _server_spec(connection_id="http-listener-session-expired"),
            client=HttpMcpClient(transport=server.transport),
        )
        adapter = toolset.tools[0]
        app = CayuApp(enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            mcp_toolsets=(toolset,),
        )
        try:
            async with asyncio.timeout(2):
                while toolset.refresh_state not in {
                    McpToolsetRefreshState.DIRTY,
                    McpToolsetRefreshState.QUARANTINED,
                }:
                    await asyncio.sleep(0.001)
            return toolset.refresh_state, adapter._dispatch_authority_is_current()
        finally:
            await toolset.close()

    state, current = asyncio.run(run())

    assert state in {
        McpToolsetRefreshState.DIRTY,
        McpToolsetRefreshState.QUARANTINED,
    }
    assert current is False


def test_http_fatal_post_timeout_fences_refresh_owned_catalogue_authority() -> None:
    server = BlockingGetMcpHttpServer()

    async def run() -> tuple[bool, McpToolsetRefreshState, bool]:
        toolset = await connect_mcp_toolset(
            _server_spec(connection_id="http-listener-post-timeout"),
            client=HttpMcpClient(transport=server.transport),
        )
        session = toolset.session
        assert isinstance(session, HttpMcpSession)
        stale_adapter = toolset.tools[0]
        app = CayuApp(enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            mcp_toolsets=(toolset,),
        )
        await asyncio.wait_for(server.stream.started.wait(), timeout=1)
        await _wait_for_http_mcp_refresh_state(toolset, McpToolsetRefreshState.READY)
        assert stale_adapter._dispatch_authority_is_current() is True
        server.timeout_on = "tools/call"
        try:
            with pytest.raises(McpIdleTimeoutError):
                await session.call_tool("search", {})
            return (
                session._closed,
                toolset.refresh_state,
                stale_adapter._dispatch_authority_is_current(),
            )
        finally:
            await toolset.close()

    assert asyncio.run(run()) == (
        True,
        McpToolsetRefreshState.DIRTY,
        False,
    )


def test_http_post_404_fences_refresh_owned_catalogue_authority() -> None:
    server = BlockingGetMcpHttpServer()

    async def run() -> tuple[bool, McpToolsetRefreshState, bool]:
        toolset = await connect_mcp_toolset(
            _server_spec(connection_id="http-listener-post-session-expired"),
            client=HttpMcpClient(transport=server.transport),
        )
        session = toolset.session
        assert isinstance(session, HttpMcpSession)
        stale_adapter = toolset.tools[0]
        app = CayuApp(enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            mcp_toolsets=(toolset,),
        )
        await asyncio.wait_for(server.stream.started.wait(), timeout=1)
        await _wait_for_http_mcp_refresh_state(toolset, McpToolsetRefreshState.READY)
        assert stale_adapter._dispatch_authority_is_current() is True
        server.expire_after_init = True
        try:
            with pytest.raises(McpProtocolError, match="expired or was not found"):
                await session.call_tool("search", {})
            return (
                session._closed,
                toolset.refresh_state,
                stale_adapter._dispatch_authority_is_current(),
            )
        finally:
            await toolset.close()

    assert asyncio.run(run()) == (
        True,
        McpToolsetRefreshState.DIRTY,
        False,
    )


def test_http_get_sse_listener_is_cancelled_and_joined_on_toolset_close() -> None:
    server = BlockingGetMcpHttpServer()

    async def run():
        toolset = await connect_mcp_toolset(
            _server_spec(connection_id="http-listener-close"),
            client=HttpMcpClient(transport=server.transport),
        )
        app = CayuApp(enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            mcp_toolsets=(toolset,),
        )
        await asyncio.wait_for(server.stream.started.wait(), timeout=1)
        await asyncio.wait_for(toolset.close(), timeout=1)
        return server.stream.closed.is_set()

    assert asyncio.run(run()) is True


def test_http_listener_settles_cancellation_resistant_reads_without_task_leaks() -> None:
    server = CancellationResistantGetMcpHttpServer()
    limits = McpTransportLimits(
        max_message_bytes=1024,
        max_response_bytes=2048,
        idle_timeout_s=0.01,
        total_call_timeout_s=0.05,
    )

    async def run():
        loop = asyncio.get_running_loop()
        prior_exception_handler = loop.get_exception_handler()
        loop_failures: list[dict[str, Any]] = []
        loop.set_exception_handler(lambda _loop, context: loop_failures.append(context))
        toolset = await connect_mcp_toolset(
            _server_spec(connection_id="http-listener-resistant-read"),
            client=HttpMcpClient(
                transport=server.transport,
                transport_limits=limits,
            ),
        )
        session = toolset.session
        assert isinstance(session, HttpMcpSession)
        app = CayuApp(enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            mcp_toolsets=(toolset,),
        )
        try:
            await asyncio.wait_for(server.stream.cancelled.wait(), timeout=1)
            async with asyncio.timeout(2):
                while not session._closed:
                    await asyncio.sleep(0.001)
        finally:
            await toolset.close()
            for _ in range(3):
                gc.collect()
                await asyncio.sleep(0)
            loop.set_exception_handler(prior_exception_handler)
        return loop_failures, server.stream.closed.is_set()

    loop_failures, stream_closed = asyncio.run(run())

    assert loop_failures == []
    assert stream_closed is True


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


def test_http_rejects_non_2_0_jsonrpc_without_retaining_secret_payload() -> None:
    secret = "mcp-http-structural-json-canary"
    server = FakeMcpHttpServer(
        bad_jsonrpc_on="tools/list",
        tools=[
            {
                "name": "search",
                "description": secret,
                "inputSchema": {"type": "object"},
            }
        ],
    )
    vault = StaticVault({"token": secret})

    async def run():
        session = await HttpMcpClient(
            transport=server.transport,
            secret_resolver=vault,
        ).connect(_server_spec(secret_headers={"authorization": SecretRef(name="token")}))
        try:
            with pytest.raises(McpProtocolError, match="2.0") as excinfo:
                await session.list_tools()
            return excinfo.value
        finally:
            await session.close()

    error = asyncio.run(run())

    _assert_cayu_traceback_does_not_retain_secret(error, secret)


def test_http_rejects_non_object_jsonrpc_without_retaining_secret_payload() -> None:
    secret = "mcp-http-non-object-json-canary"
    server = FakeMcpHttpServer(
        raw_json_on="tools/list",
        raw_json_document=json.dumps([secret]),
    )
    vault = StaticVault({"token": secret})

    async def run():
        session = await HttpMcpClient(
            transport=server.transport,
            secret_resolver=vault,
        ).connect(_server_spec(secret_headers={"authorization": SecretRef(name="token")}))
        try:
            with pytest.raises(McpProtocolError, match="must be an object") as excinfo:
                await session.list_tools()
            return excinfo.value
        finally:
            await session.close()

    error = asyncio.run(run())

    _assert_cayu_traceback_does_not_retain_secret(error, secret)


def test_http_invalid_initialize_envelope_does_not_retain_secret_config() -> None:
    secret = "mcp-http-initialize-json-canary"
    server = FakeMcpHttpServer(
        bad_jsonrpc_on="initialize",
        instructions=secret,
    )
    vault = StaticVault({"token": secret})

    async def run():
        with pytest.raises(McpProtocolError, match="2.0") as excinfo:
            await HttpMcpClient(
                transport=server.transport,
                secret_resolver=vault,
            ).connect(_server_spec(secret_headers={"authorization": SecretRef(name="token")}))
        return excinfo.value

    error = asyncio.run(run())

    _assert_cayu_traceback_does_not_retain_secret(error, secret)


def test_http_invalid_initialize_text_does_not_retain_resolved_secret() -> None:
    secret = "mcp-http-initialize-portable-canary"
    server = FakeMcpHttpServer(protocol_version=f"{secret}\x00")
    vault = StaticVault({"token": secret})

    async def run() -> McpProtocolError:
        with pytest.raises(
            McpProtocolError, match="initialize result contained invalid data"
        ) as excinfo:
            await HttpMcpClient(
                transport=server.transport,
                secret_resolver=vault,
            ).connect(_server_spec(secret_headers={"authorization": SecretRef(name="token")}))
        return excinfo.value

    error = asyncio.run(run())

    assert server.deleted is True
    _assert_cayu_traceback_does_not_retain_secret(error, secret)


def test_http_session_constructor_revalidates_server_before_request() -> None:
    fake_server = FakeMcpHttpServer()

    async def run() -> None:
        http_client = httpx.AsyncClient(transport=fake_server.transport)
        server = _server_spec()
        server.url = "invalid-http-session\x00"
        try:
            with pytest.raises(ValueError):
                HttpMcpSession(
                    server=server,
                    http_client=http_client,
                    url="https://mcp.example/rpc",
                    client_name="cayu",
                    client_version="0.1.0",
                )
        finally:
            await http_client.aclose()

    asyncio.run(run())

    assert fake_server.calls == []


def test_http_session_constructor_owns_server_snapshot() -> None:
    async def run() -> tuple[McpServerSpec, McpServerSpec]:
        server = _server_spec()
        expected = server.model_copy(deep=True)
        session = HttpMcpSession(
            server=server,
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(lambda _: httpx.Response(200))
            ),
            url="https://mcp.example/rpc",
            client_name="cayu",
            client_version="0.1.0",
        )
        server.name = "caller-mutated-server"
        try:
            return expected, session.server
        finally:
            await session.close()

    expected, retained = asyncio.run(run())

    assert retained == expected


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


def test_http_redacts_session_secrets_from_mcp_metadata_and_protocol_errors() -> None:
    secret = "mcp-response-boundary-canary"
    server = FakeMcpHttpServer(
        tools=[
            {
                "name": "search",
                "description": f"Send {secret} when searching.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"description": f"Authenticated by {secret}"}},
                },
                "annotations": {"title": f"Search with {secret}"},
            }
        ],
        instructions=f"Always use {secret}.",
        error_on="tools/call",
        error_message=f"upstream rejected {secret}",
    )
    vault = StaticVault({"mcp_token": secret})

    async def run():
        session = await HttpMcpClient(
            transport=server.transport,
            secret_resolver=vault,
        ).connect(_server_spec(secret_headers={"authorization": SecretRef(name="mcp_token")}))
        try:
            tools = await session.list_tools()
            with pytest.raises(McpProtocolError) as excinfo:
                await session.call_tool("search", {})
            return session.initialize_result, tools, excinfo.value
        finally:
            await session.close()

    initialize_result, tools, error = asyncio.run(run())
    serialized = json.dumps(
        {
            "initialize": initialize_result.model_dump(mode="json"),
            "tools": [tool.model_dump(mode="json") for tool in tools],
            "error": str(error),
        }
    )

    assert secret not in serialized
    assert error.__cause__ is None
    assert error.__context__ is None
    assert REDACTED_SECRET in serialized


def test_http_keeps_raw_mcp_identities_private_while_calls_use_transport_mapping() -> None:
    tool_secret = "mcp-tool-identity-canary"
    resource_secret = "file://x"
    server = FakeMcpHttpServer(
        tools=[
            {
                "name": tool_secret,
                "description": "Private transport identity.",
                "inputSchema": {"type": "object"},
            }
        ]
    )
    vault = StaticVault(
        {
            "tool_token": tool_secret,
            "resource_token": resource_secret,
        }
    )

    async def run():
        session = await HttpMcpClient(
            transport=server.transport,
            secret_resolver=vault,
        ).connect(
            _server_spec(
                secret_headers={
                    "x-tool-token": SecretRef(name="tool_token"),
                    "x-resource-token": SecretRef(name="resource_token"),
                }
            )
        )
        try:
            tools = await session.list_tools()
            resources = await session.list_resources()
            tool_result = await session.call_tool(tools[0].name, {})
            resource_result = await session.read_resource(resources[0].uri)
            return tools, resources, tool_result, resource_result
        finally:
            await session.close()

    tools, resources, tool_result, resource_result = asyncio.run(run())
    serialized = json.dumps(
        {
            "tools": [tool.model_dump(mode="json") for tool in tools],
            "resources": [resource.model_dump(mode="json") for resource in resources],
            "tool_result": tool_result.model_dump(mode="json"),
            "resource_result": resource_result.model_dump(mode="json"),
        }
    )

    assert tool_secret not in serialized
    assert resource_secret not in serialized
    assert tools[0].name == REDACTED_SECRET
    assert resources[0].uri == REDACTED_SECRET


def test_http_rejects_cross_page_authority_collisions_without_partial_mapping() -> None:
    first_secret = "mcp-page-one-authority-canary"
    second_secret = "search_page_2"
    server = FakeMcpHttpServer(
        paginate=True,
        tools=[
            {
                "name": first_secret,
                "description": "First page private identity.",
                "inputSchema": {"type": "object"},
            }
        ],
    )
    vault = StaticVault({"first": first_secret, "second": second_secret})

    async def run():
        session = await HttpMcpClient(
            transport=server.transport,
            secret_resolver=vault,
        ).connect(
            _server_spec(
                secret_headers={
                    "x-first": SecretRef(name="first"),
                    "x-second": SecretRef(name="second"),
                }
            )
        )
        assert isinstance(session, HttpMcpSession)
        try:
            with pytest.raises(McpProtocolError, match="collide") as excinfo:
                await session.list_tools()
            return dict(session._tool_transport_names), excinfo.value
        finally:
            await session.close()

    mapping, error = asyncio.run(run())

    assert mapping == {}
    _assert_cayu_traceback_does_not_retain_secret(error, first_secret)
    _assert_cayu_traceback_does_not_retain_secret(error, second_secret)


def test_http_paginated_mapping_failure_does_not_retain_private_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_secret = "mcp-http-private-first-authority-canary"
    second_secret = "mcp-http-private-second-authority-canary"
    cursor = "mcp-http-private-cursor-canary"
    redactor = SecretRedactor([first_secret, second_secret, cursor])

    async def run() -> McpProtocolError:
        session = HttpMcpSession(
            server=_server_spec(),
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(lambda _: httpx.Response(200))
            ),
            url="https://mcp.example/rpc",
            client_name="cayu",
            client_version="0.1.0",
            secret_redactor=redactor,
        )

        async def send(
            _payload: dict[str, Any],
            request_id: int,
            *,
            budget: Any,
            failure_redactor: SecretRedactor | None = None,
        ) -> dict[str, Any]:
            del budget
            assert failure_redactor is redactor
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": [
                        {"name": first_secret},
                        {"name": second_secret},
                    ],
                    "nextCursor": cursor,
                },
            }

        monkeypatch.setattr(session, "_send", send)
        try:
            with pytest.raises(McpProtocolError, match="collide") as exc_info:
                await session._request(
                    "tools/list",
                    {},
                    authority_mapping={},
                    paginated=True,
                )
            return exc_info.value
        finally:
            await session.close()

    error = asyncio.run(run())

    _assert_cayu_traceback_does_not_retain_secret(error, first_secret)
    _assert_cayu_traceback_does_not_retain_secret(error, second_secret)
    _assert_cayu_traceback_does_not_retain_secret(error, cursor)


def test_http_invalid_paginated_cursor_does_not_retain_secret_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "mcp-http-invalid-cursor-secret-canary"
    redactor = SecretRedactor(secret)

    async def run() -> McpProtocolError:
        session = HttpMcpSession(
            server=_server_spec(),
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(lambda _: httpx.Response(200))
            ),
            url="https://mcp.example/rpc",
            client_name="cayu",
            client_version="0.1.0",
            secret_redactor=redactor,
        )

        async def send(
            _payload: dict[str, Any],
            request_id: int,
            *,
            budget: Any,
            failure_redactor: SecretRedactor | None = None,
        ) -> dict[str, Any]:
            del budget
            assert failure_redactor is redactor
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": [
                        {
                            "name": "echo",
                            "description": secret,
                            "inputSchema": {"type": "object"},
                        }
                    ],
                    "nextCursor": float("nan"),
                },
            }

        monkeypatch.setattr(session, "_send", send)
        try:
            with pytest.raises(McpProtocolError, match="invalid portable JSON") as exc_info:
                await session._request(
                    "tools/list",
                    {},
                    authority_mapping={},
                    paginated=True,
                )
            return exc_info.value
        finally:
            await session.close()

    error = asyncio.run(run())

    _assert_cayu_traceback_does_not_retain_secret(error, secret)


def test_http_rejects_invalid_portable_response_without_retaining_secret_payload() -> None:
    secret = "mcp-http-invalid-portable-json-canary"
    server = FakeMcpHttpServer(
        raw_json_on="tools/list",
        raw_json_document=json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {
                    "tools": [
                        {
                            "name": "search",
                            "description": secret,
                            "inputSchema": {"type": "object"},
                        }
                    ],
                    "invalid": float("nan"),
                },
            }
        ),
    )
    vault = StaticVault({"token": secret})

    async def run():
        session = await HttpMcpClient(
            transport=server.transport,
            secret_resolver=vault,
        ).connect(_server_spec(secret_headers={"authorization": SecretRef(name="token")}))
        try:
            with pytest.raises(McpProtocolError, match="invalid portable JSON") as excinfo:
                await session.list_tools()
            return excinfo.value
        finally:
            await session.close()

    error = asyncio.run(run())

    _assert_cayu_traceback_does_not_retain_secret(error, secret)


def test_http_rejects_unclean_private_authority_without_partial_mapping() -> None:
    secret = "mcp-http-unclean-authority-canary"
    server = FakeMcpHttpServer(
        tools=[
            {
                "name": f" {secret}",
                "description": "Unclean private transport identity.",
                "inputSchema": {"type": "object"},
            }
        ]
    )
    vault = StaticVault({"token": secret})

    async def run():
        session = await HttpMcpClient(
            transport=server.transport,
            secret_resolver=vault,
        ).connect(_server_spec(secret_headers={"authorization": SecretRef(name="token")}))
        assert isinstance(session, HttpMcpSession)
        try:
            with pytest.raises(McpProtocolError, match="clean nonblank") as excinfo:
                await session.list_tools()
            return dict(session._tool_transport_names), excinfo.value
        finally:
            await session.close()

    mapping, error = asyncio.run(run())

    assert mapping == {}
    _assert_cayu_traceback_does_not_retain_secret(error, secret)


@pytest.mark.parametrize(
    "tool_order",
    [
        "ambiguous_first",
        "ambiguous_last",
        "ambiguous_only",
    ],
)
def test_http_ambiguous_authority_never_commits_private_mapping(
    tool_order: str,
) -> None:
    secret = f"mcp-http-{tool_order}-authority-canary"
    valid_tool = {
        "name": secret,
        "description": "Valid private transport identity.",
        "inputSchema": {"type": "object"},
    }
    ambiguous_tool = {
        "name": True,
        "description": "Wrong-type transport identity.",
        "inputSchema": {"type": "object"},
    }
    tools = (
        [ambiguous_tool]
        if tool_order == "ambiguous_only"
        else (
            [ambiguous_tool, valid_tool]
            if tool_order == "ambiguous_first"
            else [valid_tool, ambiguous_tool]
        )
    )
    server = FakeMcpHttpServer(tools=tools)
    vault = StaticVault({"token": secret})

    async def run():
        session = await HttpMcpClient(
            transport=server.transport,
            secret_resolver=vault,
        ).connect(_server_spec(secret_headers={"authorization": SecretRef(name="token")}))
        assert isinstance(session, HttpMcpSession)
        try:
            with pytest.raises(McpProtocolError, match="invalid tool definition") as excinfo:
                await session.list_tools()
            return dict(session._tool_transport_names), excinfo.value
        finally:
            await session.close()

    mapping, error = asyncio.run(run())

    assert mapping == {}
    _assert_cayu_traceback_does_not_retain_secret(error, secret)


@pytest.mark.parametrize(
    "resource_order",
    [
        "ambiguous_first",
        "ambiguous_last",
        "ambiguous_only",
    ],
)
def test_http_ambiguous_resource_authority_never_commits_private_mapping(
    resource_order: str,
) -> None:
    secret = f"mcp-http-{resource_order}-resource-authority-canary"
    valid_resource = {
        "uri": secret,
        "name": "Valid private transport identity.",
    }
    ambiguous_resource = {
        "uri": True,
        "name": "Wrong-type transport identity.",
    }
    resources = (
        [ambiguous_resource]
        if resource_order == "ambiguous_only"
        else (
            [ambiguous_resource, valid_resource]
            if resource_order == "ambiguous_first"
            else [valid_resource, ambiguous_resource]
        )
    )
    server = FakeMcpHttpServer(resources=resources)
    vault = StaticVault({"token": secret})

    async def run():
        session = await HttpMcpClient(
            transport=server.transport,
            secret_resolver=vault,
        ).connect(_server_spec(secret_headers={"authorization": SecretRef(name="token")}))
        assert isinstance(session, HttpMcpSession)
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
    _assert_cayu_traceback_does_not_retain_secret(error, secret)


def test_http_invalid_json_error_does_not_retain_raw_decoder_document() -> None:
    secret = "mcp-invalid-json-canary"

    with pytest.raises(McpProtocolError) as excinfo:
        _decode_jsonrpc(f"not-json-{secret}")

    _assert_cayu_traceback_does_not_retain_secret(excinfo.value, secret)


def _assert_cayu_traceback_does_not_retain_secret(
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


def test_http_redacts_secret_before_bounding_error_body() -> None:
    secret = "mcp-http-split-boundary-canary"
    visible_prefix_length = len(REDACTED_SECRET)
    body = "x" * (_MAX_ERROR_BODY_CHARS - visible_prefix_length) + secret + "-unbounded-suffix"

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text=body, request=request)

    async def run() -> str:
        session = HttpMcpSession(
            server=_server_spec(),
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
            url="https://mcp.example/rpc",
            client_name="cayu",
            client_version="0.1.0",
            secret_redactor=SecretRedactor(secret),
        )
        try:
            with pytest.raises(McpProtocolError) as excinfo:
                await session.initialize()
            return str(excinfo.value)
        finally:
            await session.close()

    error = asyncio.run(run())

    assert secret not in error
    assert secret[:visible_prefix_length] not in error
    assert REDACTED_SECRET in error


def test_http_redacts_transport_error_without_retaining_raw_exception_cause() -> None:
    secret = "mcp-http-transport-canary"

    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"upstream rejected {secret}", request=request)

    async def run() -> McpProtocolError:
        session = HttpMcpSession(
            server=_server_spec(),
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(fail)),
            url="https://mcp.example/rpc",
            client_name="cayu",
            client_version="0.1.0",
            secret_redactor=SecretRedactor(secret),
        )
        try:
            with pytest.raises(McpProtocolError) as excinfo:
                await session.initialize()
            return excinfo.value
        finally:
            await session.close()

    error = asyncio.run(run())

    assert secret not in str(error)
    assert REDACTED_SECRET in str(error)
    assert error.__cause__ is None
    assert error.__context__ is None


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


def test_http_subclass_inheriting_list_tools_keeps_builtin_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExtendedHttpSession(HttpMcpSession):
        pass

    class DelegatingHttpSession(HttpMcpSession):
        async def list_tools(self):
            return await super().list_tools()

    async def exercise(session_type):
        session = session_type(
            server=_server_spec(),
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(lambda _: httpx.Response(200))
            ),
            url="https://mcp.example/rpc",
            client_name="cayu",
            client_version="0.1.0",
            secret_redactor=SecretRedactor(),
        )

        async def send(
            _payload: dict[str, Any],
            request_id: int,
            *,
            budget: Any,
            failure_redactor: SecretRedactor | None = None,
        ) -> dict[str, Any]:
            del budget
            assert failure_redactor is session.secret_redactor
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"tools": _DEFAULT_TOOLS},
            }

        monkeypatch.setattr(session, "_send", send)
        try:
            return await session.list_tools()
        finally:
            await session.close()

    async def run():
        return await exercise(ExtendedHttpSession), await exercise(DelegatingHttpSession)

    inherited_definitions, delegated_definitions = asyncio.run(run())

    assert [definition.name for definition in inherited_definitions] == ["search"]
    assert [definition.name for definition in delegated_definitions] == ["search"]


def test_http_private_contract_refresh_is_blocked_without_committing_transport_authority() -> None:
    private_name = "mcp-http-private-refresh-name"
    first_description = "mcp-http-private-refresh-description-alpha"
    second_description = "mcp-http-private-refresh-description-beta"
    initial_tools = [
        {
            "name": private_name,
            "description": first_description,
            "inputSchema": {"type": "object"},
        }
    ]
    server = FakeMcpHttpServer(tools=initial_tools)
    vault = StaticVault(
        {
            "name": private_name,
            "first_description": first_description,
            "second_description": second_description,
        }
    )
    spec = _server_spec(
        connection_id="private-http-block",
        secret_headers={
            "x-private-name": SecretRef(name="name"),
            "x-private-description-alpha": SecretRef(name="first_description"),
            "x-private-description-beta": SecretRef(name="second_description"),
        },
    )

    async def run():
        toolset = await connect_mcp_toolset(
            spec,
            client=HttpMcpClient(transport=server.transport, secret_resolver=vault),
        )
        session = toolset.session
        assert isinstance(session, HttpMcpSession)
        initial_mapping = dict(session._tool_transport_names)
        app = CayuApp(
            enable_logging=False,
            mcp_manifest_policy=McpManifestPolicy(
                on_tools_changed=McpManifestPolicyAction.BLOCK,
            ),
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            mcp_toolsets=(toolset,),
        )
        try:
            server.tools = [
                {
                    "name": private_name,
                    "description": second_description,
                    "inputSchema": {"type": "object"},
                }
            ]
            with pytest.raises(McpToolsetRefreshBlocked) as exc_info:
                await app.refresh_mcp_toolset(toolset)
            blocked_mapping = dict(session._tool_transport_names)
            blocked_state = toolset.refresh_state

            server.tools = initial_tools
            recovery = await app.refresh_mcp_toolset(toolset)
            return (
                initial_mapping,
                blocked_mapping,
                blocked_state,
                recovery,
                recovery.toolset.refresh_state,
                exc_info.value,
            )
        finally:
            await toolset.close()

    initial_mapping, blocked_mapping, blocked_state, recovery, recovery_state, error = asyncio.run(
        run()
    )

    assert initial_mapping == {REDACTED_SECRET: private_name}
    assert blocked_mapping == initial_mapping
    assert blocked_state == "quarantined"
    assert recovery.status == "unchanged"
    assert recovery_state == "ready"
    public_error = repr(error)
    assert private_name not in public_error
    assert first_description not in public_error
    assert second_description not in public_error


def test_http_private_tool_rename_commits_exact_new_transport_authority() -> None:
    first_private_name = "mcp-http-private-refresh-name-alpha"
    second_private_name = "mcp-http-private-refresh-name-beta"
    server = FakeMcpHttpServer(
        tools=[
            {
                "name": first_private_name,
                "description": "Private rename test.",
                "inputSchema": {"type": "object"},
            }
        ]
    )
    vault = StaticVault({"first_name": first_private_name, "second_name": second_private_name})
    spec = _server_spec(
        connection_id="private-http-rename",
        secret_headers={
            "x-private-name-alpha": SecretRef(name="first_name"),
            "x-private-name-beta": SecretRef(name="second_name"),
        },
    )

    async def run():
        toolset = await connect_mcp_toolset(
            spec,
            client=HttpMcpClient(transport=server.transport, secret_resolver=vault),
        )
        session = toolset.session
        assert isinstance(session, HttpMcpSession)
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
            server.tools = [
                {
                    "name": second_private_name,
                    "description": "Private rename test.",
                    "inputSchema": {"type": "object"},
                }
            ]
            refresh = await app.refresh_mcp_toolset(toolset)
            call_result = await refresh.toolset.call_tool(
                refresh.toolset.definitions[0].name,
                {},
            )
            return refresh, call_result, dict(session._tool_transport_names)
        finally:
            await toolset.close()

    refresh, call_result, mapping = asyncio.run(run())

    assert refresh.status == "accepted"
    assert refresh.diff.changed_tools == (refresh.toolset.tools[0].name,)
    assert mapping == {REDACTED_SECRET: second_private_name}
    assert server.tool_call_names[-1] == second_private_name
    assert call_result.is_error is False
    public_refresh = repr(refresh.diff.policy_input())
    assert first_private_name not in public_refresh
    assert second_private_name not in public_refresh


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
