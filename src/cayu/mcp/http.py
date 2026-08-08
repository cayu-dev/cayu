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
    DEFAULT_MCP_MAX_LIST_ITEMS,
    DEFAULT_MCP_MAX_LIST_PAGES,
    JsonrpcAuthorityMappingResult,
    McpPaginatedPage,
    McpProtocolError,
    collect_paginated,
    initialize_params,
    initialize_result_from_payload,
    jsonrpc_authority_mapping,
    jsonrpc_notification_payload,
    jsonrpc_request_payload,
    merge_jsonrpc_authority_mapping,
    resource_definition_from_payload,
    resource_result_from_payload,
    result_from_jsonrpc_response,
    safely_redact_jsonrpc_response,
    tool_definition_from_payload,
    tool_result_from_payload,
    validate_negotiated_protocol_version,
    validate_positive_integer,
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
    copy_mcp_server_spec,
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
        max_list_pages: int = DEFAULT_MCP_MAX_LIST_PAGES,
        max_list_items: int = DEFAULT_MCP_MAX_LIST_ITEMS,
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
        self.max_list_pages = validate_positive_integer(max_list_pages, "max_list_pages")
        self.max_list_items = validate_positive_integer(max_list_items, "max_list_items")
        self._transport = transport
        if secret_resolver is not None:
            validate_secret_resolver(secret_resolver)
        self.secret_resolver = secret_resolver

    async def connect(self, server: McpServerSpec) -> McpSession:
        server = copy_mcp_server_spec(server)
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
            resolved.clear()
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
        try:
            session = HttpMcpSession(
                server=server,
                http_client=httpx.AsyncClient(**client_kwargs),
                url=server.url,
                client_name=self.client_name,
                client_version=self.client_version,
                max_list_pages=self.max_list_pages,
                max_list_items=self.max_list_items,
                secret_redactor=secret_redactor,
            )
        finally:
            # httpx owns copied headers/config after successful construction. Do
            # not retain injected values if construction or initialize fails and
            # diagnostics capture traceback locals.
            headers.clear()
            client_kwargs.clear()
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
        max_list_pages: int = DEFAULT_MCP_MAX_LIST_PAGES,
        max_list_items: int = DEFAULT_MCP_MAX_LIST_ITEMS,
        secret_redactor: SecretRedactor | None = None,
    ) -> None:
        server = copy_mcp_server_spec(server)
        self.server = server
        self._secret_redactor = secret_redactor or SecretRedactor()
        self.client_name = client_name
        self.client_version = client_version
        self.max_list_pages = validate_positive_integer(max_list_pages, "max_list_pages")
        self.max_list_items = validate_positive_integer(max_list_items, "max_list_items")
        self._http = http_client
        self._url = url
        self._initialize_result: McpInitializeResult | None = None
        self._session_id: str | None = None
        self._next_id = 1
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None
        self._tool_transport_names: dict[str, str] = {}
        self._resource_transport_uris: dict[str, str] = {}
        self._authority_mapping_lock = asyncio.Lock()

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
        initialize_result = initialize_result_from_payload(result)
        validation_error: McpProtocolError | None = None
        try:
            validate_negotiated_protocol_version(initialize_result.protocol_version)
        except McpProtocolError as exc:
            validation_error = McpProtocolError(self._secret_redactor.redact_text(str(exc)))
        if validation_error is not None:
            del initialize_result
            raise validation_error from None
        self._initialize_result = initialize_result
        await self._notify("notifications/initialized", {})

    async def list_tools(self) -> tuple[McpToolDefinition, ...]:
        transport_names: dict[str, str] = {}

        async def request_page(method: str, params: dict[str, Any]) -> Any:
            return await self._request(
                method,
                params,
                authority_mapping=transport_names,
                paginated=True,
            )

        tools = await collect_paginated(
            request_page,
            "tools/list",
            "tools",
            max_pages=self.max_list_pages,
            max_items=self.max_list_items,
            redactor=self._secret_redactor,
        )
        definitions: tuple[McpToolDefinition, ...] = ()
        definition_error = False
        try:
            definitions = tuple(
                tool_definition_from_payload(tool, self.server.name) for tool in tools
            )
        except (McpProtocolError, TypeError, ValueError):
            definition_error = True
        if definition_error:
            transport_names.clear()
            tools.clear()
            raise McpProtocolError("MCP tools/list returned an invalid tool definition.") from None
        async with self._authority_mapping_lock:
            merge_result = merge_jsonrpc_authority_mapping(
                self._tool_transport_names,
                transport_names,
                max_items=self.max_list_items,
            )
            if merge_result.error is not None:
                transport_names.clear()
                raise McpProtocolError(
                    self._secret_redactor.redact_text(merge_result.error)
                ) from None
            self._tool_transport_names = {
                public: raw for public, raw in merge_result.mapping.items() if public != raw
            }
            merge_result.mapping.clear()
            transport_names.clear()
        return definitions

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> McpToolResult:
        tool_name = require_clean_nonblank(name, "tool name")
        copied_arguments = copy_json_value(arguments, "arguments")
        if type(copied_arguments) is not dict:
            raise TypeError("MCP tool arguments must be an object.")
        result = await self._request(
            "tools/call",
            {
                "name": self._tool_transport_names.get(tool_name, tool_name),
                "arguments": copied_arguments,
            },
        )
        if type(result) is not dict:
            raise McpProtocolError("MCP tools/call result must be an object.")
        return tool_result_from_payload(result)

    async def list_resources(self) -> tuple[McpResourceDefinition, ...]:
        transport_uris: dict[str, str] = {}

        async def request_page(method: str, params: dict[str, Any]) -> Any:
            return await self._request(
                method,
                params,
                authority_mapping=transport_uris,
                paginated=True,
            )

        resources = await collect_paginated(
            request_page,
            "resources/list",
            "resources",
            max_pages=self.max_list_pages,
            max_items=self.max_list_items,
            redactor=self._secret_redactor,
        )
        definitions: tuple[McpResourceDefinition, ...] = ()
        definition_error = False
        try:
            definitions = tuple(
                resource_definition_from_payload(resource, self.server.name)
                for resource in resources
            )
        except (McpProtocolError, TypeError, ValueError):
            definition_error = True
        if definition_error:
            transport_uris.clear()
            resources.clear()
            raise McpProtocolError(
                "MCP resources/list returned an invalid resource definition."
            ) from None
        async with self._authority_mapping_lock:
            merge_result = merge_jsonrpc_authority_mapping(
                self._resource_transport_uris,
                transport_uris,
                max_items=self.max_list_items,
            )
            if merge_result.error is not None:
                transport_uris.clear()
                raise McpProtocolError(
                    self._secret_redactor.redact_text(merge_result.error)
                ) from None
            self._resource_transport_uris = {
                public: raw for public, raw in merge_result.mapping.items() if public != raw
            }
            merge_result.mapping.clear()
            transport_uris.clear()
        return definitions

    async def read_resource(self, uri: str) -> McpResourceResult:
        resource_uri = require_clean_nonblank(uri, "resource uri")
        result = await self._request(
            "resources/read",
            {"uri": self._resource_transport_uris.get(resource_uri, resource_uri)},
        )
        if type(result) is not dict:
            raise McpProtocolError("MCP resources/read result must be an object.")
        return resource_result_from_payload(result)

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

    async def _request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        authority_mapping: dict[str, str] | None = None,
        paginated: bool = False,
    ) -> Any:
        method_name = require_clean_nonblank(method, "method")
        request_id = self._next_id
        self._next_id += 1
        sanitized_error: McpProtocolError | None = None
        try:
            message = await self._send(
                jsonrpc_request_payload(request_id, method_name, params), request_id
            )
        except McpProtocolError as exc:
            redacted_error = self._secret_redactor.redact_text(str(exc))
            if redacted_error == str(exc):
                raise
            sanitized_error = McpProtocolError(redacted_error)
        if sanitized_error is not None:
            raise sanitized_error
        redaction_result = safely_redact_jsonrpc_response(
            message,
            method=method_name,
            redactor=self._secret_redactor,
        )
        redacted_message = redaction_result.response
        mapping_result = JsonrpcAuthorityMappingResult({})
        mapping_error = redaction_result.error
        private_cursor: Any = None
        raw_result: Any = None
        if paginated and mapping_error is None:
            raw_result = message.get("result")
            if type(raw_result) is dict and "nextCursor" in raw_result:
                try:
                    private_cursor = copy_json_value(
                        raw_result["nextCursor"],
                        "nextCursor",
                    )
                except (TypeError, ValueError):
                    mapping_error = f"MCP {method_name} response has an invalid nextCursor."
        if mapping_error is None:
            mapping_result = jsonrpc_authority_mapping(
                message,
                redacted_message,
                method=method_name,
            )
            mapping_error = mapping_result.error
        if mapping_error is None and authority_mapping is not None:
            merge_result = merge_jsonrpc_authority_mapping(
                authority_mapping,
                mapping_result.mapping,
                max_items=self.max_list_items,
            )
            mapping_error = merge_result.error
            if mapping_error is None:
                authority_mapping.clear()
                authority_mapping.update(merge_result.mapping)
            merge_result.mapping.clear()
        if mapping_error is not None and authority_mapping is not None:
            authority_mapping.clear()
        mapping_result.mapping.clear()
        raw_result = None
        # The redacted response is the only representation needed after
        # authority extraction. Do not retain the private server payload in a
        # traceback if validation or result parsing fails.
        message.clear()
        message = {}
        if mapping_error is not None:
            private_cursor = None
            raise McpProtocolError(
                self._secret_redactor.redact_text_bounded(
                    mapping_error,
                    max_bytes=4096,
                )
            ) from None
        try:
            result = result_from_jsonrpc_response(redacted_message, method_name)
        except BaseException:
            private_cursor = None
            raise
        if not paginated:
            return result
        if type(result) is dict:
            result.pop("nextCursor", None)
        return McpPaginatedPage(
            result=result,
            next_cursor=private_cursor,
        )

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        method_name = require_clean_nonblank(method, "method")
        await self._send(jsonrpc_notification_payload(method_name, params), None)

    async def _send(self, payload: dict[str, Any], request_id: int | None) -> dict[str, Any]:
        # Stream the response so an SSE reply is returned as soon as the matching
        # JSON-RPC event arrives, without waiting for the server to close the stream.
        if self._closed:
            raise McpProtocolError("MCP HTTP session is closed.")
        transport_error: TimeoutError | McpProtocolError | None = None
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
                    return await _read_sse_response(
                        response,
                        request_id,
                        redactor=self._secret_redactor,
                    )
                await response.aread()
                response_text = response.text
                # Do not retain the raw body through a protocol-error traceback.
                # The stream context owns its response independently of this local.
                response = None
                try:
                    message = _decode_jsonrpc(response_text)
                finally:
                    response_text = ""
                if message.get("id") != request_id:
                    message.clear()
                    message = {}
                    raise McpProtocolError("MCP HTTP response id did not match the request.")
                return message
        except httpx.TimeoutException:
            message = self._secret_redactor.redact_text(
                f"MCP HTTP request to {self._url} timed out."
            )
            transport_error = TimeoutError(message)
        except httpx.RequestError as exc:
            message = self._secret_redactor.redact_text(
                f"MCP HTTP request failed for {self._url}: {exc}"
            )
            transport_error = McpProtocolError(message)
        if transport_error is not None:
            raise transport_error
        raise AssertionError("MCP HTTP request returned without a response or transport error.")

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
                f"MCP HTTP request failed with HTTP {response.status_code}: "
                f"{_safe_body(response, redactor=self._secret_redactor)}"
            )

    def _protocol_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        # The spec requires MCP-Protocol-Version on requests AFTER initialization
        # (the negotiated version); sending it on the initialize request itself can
        # make a server 400 before version negotiation, so omit it until initialized.
        if self._initialize_result is not None:
            headers[MCP_PROTOCOL_VERSION_HEADER] = self._initialize_result.protocol_version
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
    payload: Any = None
    protocol_error: str | None = None
    try:
        payload = json.loads(text)
    except ValueError:
        protocol_error = "MCP HTTP response was not valid JSON."
    # Drop the raw document before any public parser error is raised. A
    # syntactically valid but structurally invalid response can carry the same
    # injected secrets as an undecodable response.
    text = ""
    if protocol_error is None:
        if type(payload) is not dict:
            protocol_error = "MCP JSON-RPC message must be an object."
        elif payload.get("jsonrpc") != "2.0":
            protocol_error = "MCP JSON-RPC message must use jsonrpc='2.0'."
    if protocol_error is not None:
        payload = None
        raise McpProtocolError(protocol_error)
    return payload


async def _read_sse_response(
    response: httpx.Response,
    request_id: int,
    *,
    redactor: SecretRedactor,
) -> dict[str, Any]:
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
            _reject_server_message(message, redactor=redactor)
        # Other SSE fields (event:, id:, retry:, comments) carry no JSON-RPC payload.
    # The final loop value can itself be a raw data line. Clear it before a
    # parser failure can expose this frame through traceback-local capture.
    line = ""
    message = _sse_event_message(data_lines)
    if message is not None:
        if message.get("id") == request_id:
            return message
        _reject_server_message(message, redactor=redactor)
    raise McpProtocolError("MCP HTTP SSE stream did not contain the response.")


def _sse_event_message(data_lines: list[str]) -> dict[str, Any] | None:
    if not data_lines:
        return None
    text = "\n".join(data_lines)
    data_lines.clear()
    try:
        return _decode_jsonrpc(text)
    finally:
        text = ""


def _reject_server_message(
    message: dict[str, Any],
    *,
    redactor: SecretRedactor,
) -> None:
    # A server-initiated request (method + id) cannot be answered by this
    # request/response client, so fail loudly rather than leave the server waiting
    # (stdio likewise refuses server requests). Server notifications (method, no id)
    # and stray responses carry no obligation and are ignored.
    if "method" in message and "id" in message:
        safe_method = redactor.redact_text(repr(message["method"]))
        message.clear()
        raise McpProtocolError(
            f"Cayu does not service MCP server requests over HTTP: {safe_method}."
        )


def _safe_body(response: httpx.Response, *, redactor: SecretRedactor) -> str:
    try:
        return redactor.redact_text(response.text)[:_MAX_ERROR_BODY_CHARS]
    except Exception:
        return "<unreadable response body>"
