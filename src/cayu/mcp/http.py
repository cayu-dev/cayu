"""MCP client for remote servers over the Streamable HTTP transport.

JSON-RPC messages are sent as HTTP POST bodies to a single endpoint (`server.url`).
The server replies either with `application/json` (one JSON-RPC response) or
`text/event-stream` (SSE). The SSE stream is consumed incrementally and the matching
JSON-RPC response is returned the moment it arrives — the call does not wait for the
server to close the stream. Interim server->client notifications are ignored, but a
server-initiated request (which this request/response client cannot answer) fails the
session with a protocol error rather than being silently dropped. The request is
bounded by the configured timeout (httpx's read timeout also caps the gap between
events). A session id (`Mcp-Session-Id`) issued on `initialize` is echoed on every
later request, and `MCP-Protocol-Version` is sent on every request after
initialization. The JSON<->model parsing is the shared logic in `cayu.mcp._jsonrpc`,
identical to the stdio transport.

Two deliberate deviations from the spec's SHOULD/MUST guidance, suited to a
request/response session: on HTTP 404 (expired session) we raise and mark the session
unusable rather than transparently re-initializing — the caller/toolset reconnects to
start a new session; and to cancel we drop the connection rather than sending a
`CancelledNotification` (a possible future enhancement).
"""

from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from typing import Any

import certifi
import httpx

from cayu._validation import copy_json_value, require_clean_nonblank
from cayu.mcp._jsonrpc import (
    DEFAULT_MCP_CLIENT_NAME,
    DEFAULT_MCP_CLIENT_VERSION,
    MCP_PROTOCOL_VERSION,
    McpProtocolError,
    initialize_params,
    initialize_result_from_payload,
    jsonrpc_notification_payload,
    jsonrpc_request_payload,
    resource_definition_from_payload,
    result_from_jsonrpc_response,
    tool_definition_from_payload,
    tool_result_from_payload,
    validate_positive_number,
)
from cayu.mcp.base import (
    McpClient,
    McpInitializeResult,
    McpResourceDefinition,
    McpResourceResult,
    McpServerSpec,
    McpSession,
    McpToolDefinition,
    McpToolResult,
)
from cayu.vaults import (
    SecretRedactor,
    SecretResolver,
    resolve_secret_env,
    validate_secret_resolver,
)

# Remote tool calls can be slow, so the HTTP default is generous (the stdio default
# is 30s). Both are overridable per-server via McpServerSpec.metadata["timeout"].
DEFAULT_HTTP_MCP_TIMEOUT_S = 120.0
DEFAULT_HTTP_MCP_CONNECT_TIMEOUT_S = 10.0
MCP_SESSION_ID_HEADER = "mcp-session-id"
MCP_PROTOCOL_VERSION_HEADER = "mcp-protocol-version"
_JSON_CONTENT_TYPE = "application/json"
_SSE_CONTENT_TYPE = "text/event-stream"
_ACCEPT_HEADER = f"{_JSON_CONTENT_TYPE}, {_SSE_CONTENT_TYPE}"
_MAX_ERROR_BODY_CHARS = 2_000


class HttpMcpClient(McpClient):
    """MCP client for remote servers over the Streamable HTTP transport."""

    def __init__(
        self,
        *,
        timeout_s: float = DEFAULT_HTTP_MCP_TIMEOUT_S,
        connect_timeout_s: float = DEFAULT_HTTP_MCP_CONNECT_TIMEOUT_S,
        proxy: str | None = None,
        verify: bool | str | None = None,
        client_name: str = DEFAULT_MCP_CLIENT_NAME,
        client_version: str = DEFAULT_MCP_CLIENT_VERSION,
        transport: httpx.AsyncBaseTransport | None = None,
        secret_resolver: SecretResolver | None = None,
    ) -> None:
        self.timeout_s = validate_positive_number(timeout_s, "timeout_s")
        self.connect_timeout_s = validate_positive_number(connect_timeout_s, "connect_timeout_s")
        self.proxy = _validate_optional_proxy(proxy, "proxy")
        # TLS verification passed to httpx: None -> default certifi bundle; a path
        # to a custom CA bundle (corporate/internal CA); or False to disable
        # verification (e.g. a self-signed dev server — not for production).
        if verify is not None and type(verify) not in {bool, str}:
            raise TypeError("verify must be a bool, a CA-bundle path string, or None.")
        self.verify = verify
        self.client_name = require_clean_nonblank(client_name, "client_name")
        self.client_version = require_clean_nonblank(client_version, "client_version")
        self._transport = transport
        if secret_resolver is not None:
            validate_secret_resolver(secret_resolver)
        self.secret_resolver = secret_resolver

    async def connect(self, server: McpServerSpec) -> McpSession:
        if type(server) is not McpServerSpec:
            raise TypeError("server must be an McpServerSpec.")
        if server.url is None:
            raise ValueError("HttpMcpClient requires an MCP server url.")
        if server.command is not None:
            raise ValueError("HttpMcpClient does not support command MCP servers.")
        if server.secret_env:
            raise ValueError(
                "HttpMcpClient does not support MCP secret_env; a remote server's "
                "process environment cannot be set by the client."
            )
        if server.secret_headers and self.secret_resolver is None:
            raise ValueError(
                "HttpMcpClient cannot resolve MCP secret_headers without a secret_resolver. "
                "Pass secret_resolver= (a Vault or CredentialProxy) to the client."
            )
        timeout_s, connect_timeout_s, proxy = self._resolve_transport_config(server)
        headers = {
            "content-type": _JSON_CONTENT_TYPE,
            "accept": _ACCEPT_HEADER,
            **server.headers,
        }
        secret_redactor = SecretRedactor()
        if server.secret_headers and self.secret_resolver is not None:
            # Secret header values are resolved at connect time and injected
            # directly into the HTTP client, never into model-visible config.
            resolved = await resolve_secret_env(
                server.secret_headers,
                self.secret_resolver,
                scope={"mcp_server": server.name, "destination": server.url},
            )
            for name, secret in resolved.items():
                headers[name] = secret.value.get_secret_value()
            # A hostile server can echo these values back through tool output; scrub them.
            secret_redactor = SecretRedactor(tuple(resolved.values()))
        client_kwargs: dict[str, Any] = {
            "headers": headers,
            "timeout": httpx.Timeout(timeout_s, connect=connect_timeout_s),
        }
        if self._transport is not None:
            # A caller-injected transport (e.g. tests) handles its own routing, so
            # verify/proxy are not applicable.
            client_kwargs["transport"] = self._transport
        else:
            client_kwargs["verify"] = certifi.where() if self.verify is None else self.verify
            if proxy is not None:
                client_kwargs["proxy"] = proxy
        session = HttpMcpSession(
            server=server,
            http_client=httpx.AsyncClient(**client_kwargs),
            url=server.url,
            client_name=self.client_name,
            client_version=self.client_version,
            secret_redactor=secret_redactor,
        )
        try:
            await session.initialize()
        except BaseException:
            await session.close()
            raise
        return session

    def _resolve_transport_config(self, server: McpServerSpec) -> tuple[float, float, str | None]:
        """Client defaults, overridden per-server by metadata["timeout"]/["proxy"]."""
        timeout_s = self.timeout_s
        proxy = self.proxy
        if "timeout" in server.metadata:
            timeout_s = validate_positive_number(server.metadata["timeout"], "metadata.timeout")
        if "proxy" in server.metadata:
            proxy = _validate_optional_proxy(server.metadata["proxy"], "metadata.proxy")
        return timeout_s, self.connect_timeout_s, proxy


class HttpMcpSession(McpSession):
    def __init__(
        self,
        *,
        server: McpServerSpec,
        http_client: httpx.AsyncClient,
        url: str,
        client_name: str,
        client_version: str,
        secret_redactor: SecretRedactor | None = None,
    ) -> None:
        self.server = server
        self._secret_redactor = secret_redactor or SecretRedactor()
        self.client_name = client_name
        self.client_version = client_version
        self._http = http_client
        self._url = url
        self._initialize_result: McpInitializeResult | None = None
        self._session_id: str | None = None
        self._next_id = 1
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    @property
    def initialize_result(self) -> McpInitializeResult:
        if self._initialize_result is None:
            raise McpProtocolError("MCP session has not been initialized.")
        return self._initialize_result

    async def initialize(self) -> None:
        result = await self._request(
            "initialize",
            initialize_params(self.client_name, self.client_version),
        )
        if type(result) is not dict:
            raise McpProtocolError("MCP initialize result must be an object.")
        self._initialize_result = initialize_result_from_payload(result)
        if self._initialize_result.protocol_version != MCP_PROTOCOL_VERSION:
            raise McpProtocolError(
                "MCP server negotiated unsupported protocol version "
                f"{self._initialize_result.protocol_version!r}."
            )
        await self._notify("notifications/initialized", {})

    async def list_tools(self) -> tuple[McpToolDefinition, ...]:
        result = await self._request("tools/list", {})
        if type(result) is not dict:
            raise McpProtocolError("MCP tools/list result must be an object.")
        tools = result.get("tools", [])
        if not isinstance(tools, list):
            raise McpProtocolError("MCP tools/list result tools must be a list.")
        return tuple(tool_definition_from_payload(tool, self.server.name) for tool in tools)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> McpToolResult:
        tool_name = require_clean_nonblank(name, "tool name")
        copied_arguments = copy_json_value(arguments, "arguments")
        if type(copied_arguments) is not dict:
            raise TypeError("MCP tool arguments must be an object.")
        result = await self._request(
            "tools/call",
            {"name": tool_name, "arguments": copied_arguments},
        )
        if type(result) is not dict:
            raise McpProtocolError("MCP tools/call result must be an object.")
        return tool_result_from_payload(result)

    async def list_resources(self) -> tuple[McpResourceDefinition, ...]:
        result = await self._request("resources/list", {})
        if type(result) is not dict:
            raise McpProtocolError("MCP resources/list result must be an object.")
        resources = result.get("resources", [])
        if not isinstance(resources, list):
            raise McpProtocolError("MCP resources/list result resources must be a list.")
        return tuple(
            resource_definition_from_payload(resource, self.server.name) for resource in resources
        )

    async def read_resource(self, uri: str) -> McpResourceResult:
        resource_uri = require_clean_nonblank(uri, "resource uri")
        result = await self._request("resources/read", {"uri": resource_uri})
        if type(result) is not dict:
            raise McpProtocolError("MCP resources/read result must be an object.")
        contents = result.get("contents", [])
        if not isinstance(contents, list):
            raise McpProtocolError("MCP resources/read result contents must be a list.")
        return McpResourceResult(contents=contents)

    async def close(self) -> None:
        # Run cleanup in a shielded task so the DELETE + aclose still complete even
        # if the caller is cancelled mid-close (mirrors StdioMcpSession.close).
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(self._close_impl())
        cleanup_task = self._close_task
        was_cancelled = False
        while True:
            try:
                await asyncio.shield(cleanup_task)
                break
            except asyncio.CancelledError:
                was_cancelled = True
                if cleanup_task.done():
                    break
        if was_cancelled:
            raise asyncio.CancelledError

    async def _close_impl(self) -> None:
        if self._session_id is not None:
            # Best-effort session termination; the server may not allow it (405).
            with suppress(Exception):
                response = await self._http.delete(self._url, headers=self._protocol_headers())
                await response.aread()
        await self._http.aclose()

    async def _request(self, method: str, params: dict[str, Any]) -> Any:
        method_name = require_clean_nonblank(method, "method")
        request_id = self._next_id
        self._next_id += 1
        message = await self._send(
            jsonrpc_request_payload(request_id, method_name, params), request_id
        )
        return result_from_jsonrpc_response(message, method_name)

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        method_name = require_clean_nonblank(method, "method")
        await self._send(jsonrpc_notification_payload(method_name, params), None)

    async def _send(self, payload: dict[str, Any], request_id: int | None) -> dict[str, Any]:
        # Stream the response so an SSE reply is returned as soon as the matching
        # JSON-RPC event arrives, without waiting for the server to close the stream.
        if self._closed:
            raise McpProtocolError("MCP HTTP session is closed.")
        try:
            async with self._http.stream(
                "POST",
                self._url,
                content=json.dumps(payload).encode("utf-8"),
                headers=self._protocol_headers(),
            ) as response:
                await self._handle_status(response)
                session_id = response.headers.get(MCP_SESSION_ID_HEADER)
                if session_id:
                    self._session_id = session_id
                if request_id is None:
                    # Notification: the server replies 202 Accepted with no body.
                    await response.aread()
                    return {}
                content_type = response.headers.get("content-type", "")
                # Media types are case-insensitive (RFC 9110); httpx returns it as sent.
                if content_type.split(";", 1)[0].strip().lower() == _SSE_CONTENT_TYPE:
                    return await _read_sse_response(response, request_id)
                await response.aread()
                message = _decode_jsonrpc(response.text)
                if message.get("id") != request_id:
                    raise McpProtocolError("MCP HTTP response id did not match the request.")
                return message
        except httpx.TimeoutException as exc:
            raise TimeoutError(f"MCP HTTP request to {self._url} timed out.") from exc
        except httpx.RequestError as exc:
            raise McpProtocolError(f"MCP HTTP request failed for {self._url}: {exc}") from exc

    async def _handle_status(self, response: httpx.Response) -> None:
        if response.status_code == 404:
            # The session is gone (spec): poison the session so callers can't keep
            # using it, and drop the dead id so close() skips the doomed DELETE.
            self._session_id = None
            self._closed = True
            await response.aread()
            raise McpProtocolError("MCP HTTP session expired or was not found (HTTP 404).")
        if response.status_code >= 400:
            await response.aread()
            raise McpProtocolError(
                f"MCP HTTP request failed with HTTP {response.status_code}: {_safe_body(response)}"
            )

    def _protocol_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        # The spec requires MCP-Protocol-Version on requests AFTER initialization
        # (the negotiated version); sending it on the initialize request itself can
        # make a server 400 before version negotiation, so omit it until initialized.
        if self._initialize_result is not None:
            headers[MCP_PROTOCOL_VERSION_HEADER] = MCP_PROTOCOL_VERSION
        if self._session_id is not None:
            headers[MCP_SESSION_ID_HEADER] = self._session_id
        return headers


def _validate_optional_proxy(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str:
        raise ValueError(f"{field_name} must be a string.")
    return require_clean_nonblank(value, field_name)


def _decode_jsonrpc(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except ValueError as exc:
        raise McpProtocolError("MCP HTTP response was not valid JSON.") from exc
    if type(payload) is not dict:
        raise McpProtocolError("MCP JSON-RPC message must be an object.")
    if payload.get("jsonrpc") != "2.0":
        raise McpProtocolError("MCP JSON-RPC message must use jsonrpc='2.0'.")
    return payload


async def _read_sse_response(response: httpx.Response, request_id: int) -> dict[str, Any]:
    """Read the SSE stream incrementally and return the response matching the request.

    Events are dispatched on blank lines (per the SSE spec). The matching JSON-RPC
    response is returned the moment it arrives, so the call does not wait for the
    server to close the stream. A server-initiated request (which a request/response
    client cannot service) fails the session loudly instead of being silently dropped.
    """
    data_lines: list[str] = []
    async for line in response.aiter_lines():
        if line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())
        elif line == "":
            message = _sse_event_message(data_lines)
            data_lines = []
            if message is None:
                continue
            if message.get("id") == request_id:
                return message
            _reject_server_message(message)
        # Other SSE fields (event:, id:, retry:, comments) carry no JSON-RPC payload.
    message = _sse_event_message(data_lines)
    if message is not None:
        if message.get("id") == request_id:
            return message
        _reject_server_message(message)
    raise McpProtocolError("MCP HTTP SSE stream did not contain the response.")


def _sse_event_message(data_lines: list[str]) -> dict[str, Any] | None:
    if not data_lines:
        return None
    return _decode_jsonrpc("\n".join(data_lines))


def _reject_server_message(message: dict[str, Any]) -> None:
    # A server-initiated request (method + id) cannot be answered by this
    # request/response client, so fail loudly rather than leave the server waiting
    # (stdio likewise refuses server requests). Server notifications (method, no id)
    # and stray responses carry no obligation and are ignored.
    if "method" in message and "id" in message:
        raise McpProtocolError(
            f"Cayu does not service MCP server requests over HTTP: {message['method']!r}."
        )


def _safe_body(response: httpx.Response) -> str:
    try:
        return response.text[:_MAX_ERROR_BODY_CHARS]
    except Exception:
        return "<unreadable response body>"
