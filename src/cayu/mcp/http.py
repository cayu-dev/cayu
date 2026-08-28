"""MCP client for remote servers over the Streamable HTTP transport.

JSON-RPC messages are sent as HTTP POST bodies to a single endpoint (`server.url`).
The server replies either with `application/json` (one JSON-RPC response) or
`text/event-stream` (SSE). The SSE stream is consumed incrementally and the matching
JSON-RPC response is returned the moment it arrives — the call does not wait for the
server to close the stream. Interim server->client notifications are ignored, but a
server-initiated request (which this request/response client cannot answer) fails the
session with a protocol error rather than being silently dropped. Cayu incrementally
enforces message/event and aggregate-response byte limits, an inbound idle timeout,
and an absolute call deadline. A session id (`Mcp-Session-Id`) issued on `initialize` is echoed on every
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
import re
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Coroutine, Sequence
from contextlib import AbstractAsyncContextManager, aclosing, suppress
from dataclasses import dataclass
from typing import Any, TypeVar, cast

import certifi
import httpx

from cayu._exception_groups import exception_cause, set_exception_cause
from cayu._validation import (
    copy_json_value,
    json_utf8_size_within_limit,
    require_clean_nonblank,
)
from cayu.mcp._exception_handoffs import (
    attach_mcp_http_settlement_task,
    mcp_http_settlement_task,
)
from cayu.mcp._jsonrpc import (
    DEFAULT_MCP_CLIENT_NAME,
    DEFAULT_MCP_CLIENT_VERSION,
    DEFAULT_MCP_MAX_LIST_ITEMS,
    DEFAULT_MCP_MAX_LIST_PAGES,
    SUPPORTED_MCP_PROTOCOL_VERSIONS,
    JsonrpcAuthorityMappingResult,
    McpPaginatedPage,
    McpProtocolError,
    collect_paginated,
    initialize_params,
    initialize_result_from_payload,
    jsonrpc_authority_mapping,
    jsonrpc_notification_payload,
    jsonrpc_request_payload,
    jsonrpc_tool_contract_evidence,
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
from cayu.mcp._transport import (
    McpCallDeadlineExceededError,
    McpIdleTimeoutError,
    McpMessageTooLargeError,
    McpPeerClosedError,
    McpResponseTooLargeError,
    McpTransportLimits,
    _capture_mcp_owned_task_fatal_signal,
    _unwrap_mcp_owned_task_result,
    copy_mcp_transport_limits,
    credential_safe_mcp_fatal_signal,
    credential_safe_mcp_transport_failure,
    mcp_json_value_nesting_too_deep,
    mcp_jsonrpc_request_preflight,
    resolve_mcp_transport_limits,
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
    _await_mcp_session_cleanup_task,
    _credential_safe_mcp_cancellation,
    _McpCallerCancellationBoundary,
    _McpToolDiscovery,
    _McpToolDispatchSignal,
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
_TRUNCATED_ERROR_BODY_DETAIL = "<response body omitted because safe retention was incomplete>"
_SSE_LINE_ENDING = re.compile(rb"[\r\n]")
_T = TypeVar("_T")


class _McpUnexpectedTransportCancellationError(McpProtocolError):
    """An owned HTTP child stopped without cancellation of its caller."""


class _McpExtensionTransportProtocolError(RuntimeError):
    """An injected HTTP response stream raised an MCP protocol error."""


class _McpHttpStatusError(McpProtocolError):
    """A bounded HTTP response carried a non-successful status."""


@dataclass(slots=True)
class _HttpExchangeOwner:
    """Own one entered httpx stream context through bounded settlement."""

    stream_context: AbstractAsyncContextManager[httpx.Response]
    response: httpx.Response | None = None
    entered: bool = False
    exit_task: asyncio.Task[Any] | None = None
    settlement: asyncio.Future[None] | None = None

    def record_entered_response(self, response: httpx.Response) -> None:
        self.response = response
        self.entered = True

    def start_exit(self) -> asyncio.Task[Any]:
        if not self.entered:
            raise RuntimeError("MCP HTTP stream context was not entered.")
        if self.exit_task is None:
            self.exit_task = asyncio.create_task(
                _capture_mcp_owned_task_fatal_signal(_exit_http_stream_context(self.stream_context))
            )
        return self.exit_task

    def mark_settled(self) -> None:
        settlement = self.settlement
        if settlement is not None and not settlement.done():
            settlement.set_result(None)


@dataclass(slots=True)
class _HttpResponseEnvelope:
    """A parsed response plus authority staged until semantic validation."""

    message: dict[str, Any]
    staged_session_id: str | None = None

    def take_session_id(self) -> str | None:
        session_id = self.staged_session_id
        self.staged_session_id = None
        return session_id

    def clear_private_authority(self) -> None:
        self.staged_session_id = None


@dataclass(slots=True)
class _RetainedHttpErrorBody:
    """Track whether a retained diagnostic contains the complete response body."""

    content: bytearray
    source_complete: bool = True


class _McpHttpResponseContentError(McpProtocolError):
    """A response was received completely but its bounded content was invalid."""

    def __init__(
        self,
        message: str,
        *,
        cleanup_protocol_version: str | None = None,
    ) -> None:
        super().__init__(message)
        self.cleanup_protocol_version = cleanup_protocol_version


def _initialize_cleanup_protocol_version(message: dict[str, Any]) -> str | None:
    """Return positive protocol authority suitable only for session cleanup."""

    result = message.get("result")
    if type(result) is not dict:
        return None
    protocol_version = result.get("protocolVersion")
    if type(protocol_version) is str and protocol_version in SUPPORTED_MCP_PROTOCOL_VERSIONS:
        return protocol_version
    return None


class HttpMcpClient(McpClient):
    """MCP client for remote servers over the Streamable HTTP transport."""

    def __init__(
        self,
        *,
        timeout_s: float | None = None,
        transport_limits: McpTransportLimits | None = None,
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
        self.transport_limits = resolve_mcp_transport_limits(
            transport_limits,
            legacy_timeout_s=timeout_s,
            default_timeout_s=DEFAULT_HTTP_MCP_TIMEOUT_S,
            legacy_field_name="timeout_s",
        )
        self._has_explicit_transport_limits = transport_limits is not None
        # Retain the legacy observable attribute for callers that inspect it.
        self.timeout_s = self.transport_limits.total_call_timeout_s
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
        _timeout_s, connect_timeout_s, proxy = self._resolve_transport_config(server)
        transport_limits = self._resolve_transport_limits(server)
        headers = {
            "content-type": _JSON_CONTENT_TYPE,
            "accept": _ACCEPT_HEADER,
            "accept-encoding": "identity",
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
            "timeout": httpx.Timeout(
                transport_limits.idle_timeout_s,
                connect=min(connect_timeout_s, transport_limits.idle_timeout_s),
            ),
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
                transport_limits=transport_limits,
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
        except BaseException as error:
            # A deadline/cancellation failure already carries the exact retained
            # settlement owner. Waiting for close here would extend the public
            # initialization deadline with a fresh cleanup timeout.
            settlement_task = _http_settlement_task(error)
            initialization_session_close = (
                settlement_task is not None and settlement_task is session._close_task
            )
            bounded_control_failure = isinstance(
                error,
                (TimeoutError, asyncio.CancelledError),
            )
            if settlement_task is None or (
                initialization_session_close and not bounded_control_failure
            ):
                try:
                    await session.close()
                except asyncio.CancelledError as cancellation:
                    _attach_http_cleanup_failure(cancellation, error)
                    raise
                except BaseException as cleanup_error:
                    _attach_http_cleanup_failure(error, cleanup_error)
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

    def _resolve_transport_limits(self, server: McpServerSpec) -> McpTransportLimits:
        if self._has_explicit_transport_limits:
            if "timeout" in server.metadata:
                raise ValueError("metadata.timeout cannot be combined with transport_limits.")
            return copy_mcp_transport_limits(self.transport_limits)
        timeout_s = self.timeout_s
        if "timeout" in server.metadata:
            timeout_s = validate_positive_number(server.metadata["timeout"], "metadata.timeout")
        return McpTransportLimits(
            max_message_bytes=self.transport_limits.max_message_bytes,
            max_response_bytes=self.transport_limits.max_response_bytes,
            idle_timeout_s=timeout_s,
            total_call_timeout_s=timeout_s,
        )


class HttpMcpSession(McpSession):
    def __init__(
        self,
        *,
        server: McpServerSpec,
        http_client: httpx.AsyncClient,
        url: str,
        client_name: str,
        client_version: str,
        timeout_s: float | None = None,
        transport_limits: McpTransportLimits | None = None,
        max_list_pages: int = DEFAULT_MCP_MAX_LIST_PAGES,
        max_list_items: int = DEFAULT_MCP_MAX_LIST_ITEMS,
        secret_redactor: SecretRedactor | None = None,
    ) -> None:
        server = copy_mcp_server_spec(server)
        resolved_limits = resolve_mcp_transport_limits(
            transport_limits,
            legacy_timeout_s=timeout_s,
            default_timeout_s=DEFAULT_HTTP_MCP_TIMEOUT_S,
            legacy_field_name="timeout_s",
        )
        self.server = server
        self._secret_redactor = secret_redactor or SecretRedactor()
        self.client_name = client_name
        self.client_version = client_version
        self.transport_limits = resolved_limits
        self.max_list_pages = validate_positive_integer(max_list_pages, "max_list_pages")
        self.max_list_items = validate_positive_integer(max_list_items, "max_list_items")
        self._http = http_client
        self._url = url
        self._initialize_result: McpInitializeResult | None = None
        self._negotiated_protocol_version: str | None = None
        self._session_id: str | None = None
        self._cleanup_session_id: str | None = None
        self._cleanup_protocol_version: str | None = None
        self._initializing = False
        self._next_id = 1
        self._closed = False
        self._fenced = False
        self._close_task: asyncio.Task[None] | None = None
        self._failed_close_settlement_task: asyncio.Task[None] | None = None
        self._client_close_task: asyncio.Task[None] | None = None
        self._client_close_failure: BaseException | None = None
        self._active_exchange_settlements: set[asyncio.Future[None]] = set()
        self._settlement_tasks: set[asyncio.Task[None]] = set()
        self._tool_transport_names: dict[str, str] = {}
        self._resource_transport_uris: dict[str, str] = {}
        self._authority_mapping_lock = asyncio.Lock()

    @property
    def initialize_result(self) -> McpInitializeResult:
        if self._initialize_result is None:
            raise McpProtocolError("MCP session has not been initialized.")
        return self._initialize_result

    async def initialize(self) -> None:
        def parse_initialize_result(result: Any) -> McpInitializeResult:
            initialize_result = initialize_result_from_payload(result)
            validation_error: McpProtocolError | None = None
            try:
                validate_negotiated_protocol_version(initialize_result.protocol_version)
            except McpProtocolError as error:
                validation_error = McpProtocolError(self._secret_redactor.redact_text(str(error)))
            if validation_error is not None:
                del initialize_result
                raise validation_error from None
            return initialize_result

        if self._initializing:
            raise McpProtocolError("MCP HTTP session initialization is already in progress.")
        self._initializing = True
        try:
            initialize_result = await self._request(
                "initialize",
                initialize_params(self.client_name, self.client_version),
                result_parser=parse_initialize_result,
            )
            # The initialize response establishes the version required on every
            # subsequent HTTP request. Retain it privately so a failed initialized
            # notification can still send a conforming session DELETE without
            # exposing this session as fully initialized.
            self._negotiated_protocol_version = initialize_result.protocol_version
            await self._notify(
                "notifications/initialized",
                {},
                protocol_version=initialize_result.protocol_version,
                session_id=self._cleanup_session_id,
            )
            if self._closed:
                raise McpProtocolError("MCP HTTP session was closed during initialization.")
            # Publish the session identity and initialized metadata together, after
            # the initialized notification and its response owner have settled.
            self._session_id = self._cleanup_session_id
            self._cleanup_session_id = None
            self._cleanup_protocol_version = None
            self._initialize_result = initialize_result
        except BaseException as error:
            self._initialize_result = None
            self._closed = True
            if _http_settlement_task(error) is None:
                self._begin_failed_session_close(error)
            raise
        finally:
            self._initializing = False

    async def list_tools(self) -> tuple[McpToolDefinition, ...]:
        discovery = await self._discover_builtin_tools_for_toolset()
        await discovery.commit()
        return discovery.definitions

    async def _discover_tools_for_toolset(self) -> _McpToolDiscovery:
        if type(self).list_tools is not HttpMcpSession.list_tools:
            return await McpSession._discover_tools_for_toolset(self)
        return await self._discover_builtin_tools_for_toolset()

    async def _discover_builtin_tools_for_toolset(self) -> _McpToolDiscovery:
        transport_names: dict[str, str] = {}
        private_contract_hashes: list[str] = []
        parsed_tool_count = 0

        def parse_tools_page(result: Any) -> dict[str, Any]:
            nonlocal parsed_tool_count
            if type(result) is not dict:
                raise McpProtocolError("MCP tools/list result must be an object.")
            tools = result.get("tools", [])
            if type(tools) is not list:
                raise McpProtocolError("MCP tools/list result tools must be a list.")
            observed_items = parsed_tool_count + len(tools)
            if observed_items > self.max_list_items:
                result.clear()
                raise McpProtocolError(
                    f"MCP tools/list returned {observed_items} items, exceeding "
                    f"max_list_items={self.max_list_items}."
                )
            parse_failed = False
            definitions: list[McpToolDefinition] = []
            try:
                definitions = [
                    tool_definition_from_payload(tool, self.server.name) for tool in tools
                ]
            except (McpProtocolError, TypeError, ValueError):
                parse_failed = True
            finally:
                result.clear()
            if parse_failed:
                definitions.clear()
                raise McpProtocolError(
                    "MCP tools/list returned an invalid tool definition."
                ) from None
            parsed_tool_count = observed_items
            return {"tools": definitions}

        async def request_page(method: str, params: dict[str, Any]) -> Any:
            return await self._request(
                method,
                params,
                authority_mapping=transport_names,
                private_tool_contract_hashes=private_contract_hashes,
                paginated=True,
                result_parser=parse_tools_page,
            )

        try:
            tools = await collect_paginated(
                request_page,
                "tools/list",
                "tools",
                max_pages=self.max_list_pages,
                max_items=self.max_list_items,
                redactor=self._secret_redactor,
            )
        except BaseException:
            transport_names.clear()
            private_contract_hashes.clear()
            raise
        definitions = tuple(tools)
        private_hashes = tuple(private_contract_hashes)
        private_contract_hashes.clear()

        async def commit_transport_names() -> None:
            async with self._authority_mapping_lock:
                if self._closed:
                    transport_names.clear()
                    raise McpProtocolError(
                        "MCP HTTP session closed before tool discovery was published."
                    )
                self._tool_transport_names = {
                    public: raw for public, raw in transport_names.items() if public != raw
                }
                transport_names.clear()

        try:
            return _McpToolDiscovery(
                definitions,
                private_contract_hashes=private_hashes,
                commit=commit_transport_names,
                discard=transport_names.clear,
            )
        except BaseException:
            transport_names.clear()
            raise

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> McpToolResult:
        call = self._call_tool_request(name, arguments, dispatch_signal=None)
        name = ""
        arguments = {}
        return await call

    async def _call_tool_with_dispatch_signal(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        dispatch_signal: _McpToolDispatchSignal,
    ) -> McpToolResult:
        if type(self) is not HttpMcpSession:
            call = McpSession._call_tool_with_dispatch_signal(
                self,
                name,
                arguments,
                dispatch_signal=dispatch_signal,
            )
        else:
            call = self._call_tool_request(
                name,
                arguments,
                dispatch_signal=dispatch_signal,
            )
        name = ""
        arguments = {}
        return await call

    async def _call_tool_request(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        dispatch_signal: _McpToolDispatchSignal | None,
    ) -> McpToolResult:
        tool_name = require_clean_nonblank(name, "tool name")
        if type(arguments) is not dict:
            if mcp_json_value_nesting_too_deep(arguments):
                arguments = {}
                raise McpProtocolError(
                    "MCP tool arguments exceeded the supported JSON nesting."
                ) from None
            copied_arguments = copy_json_value(arguments, "arguments")
            if type(copied_arguments) is not dict:
                raise TypeError("MCP tool arguments must be an object.")
            arguments = copied_arguments
        request_params = {
            "name": self._tool_transport_names.get(tool_name, tool_name),
            "arguments": arguments,
        }
        name = ""
        tool_name = ""
        arguments = {}
        return await self._request(
            "tools/call",
            request_params,
            result_parser=tool_result_from_payload,
            dispatch_signal=dispatch_signal,
        )

    async def list_resources(self) -> tuple[McpResourceDefinition, ...]:
        transport_uris: dict[str, str] = {}
        parsed_resource_count = 0

        def parse_resources_page(result: Any) -> dict[str, Any]:
            nonlocal parsed_resource_count
            if type(result) is not dict:
                raise McpProtocolError("MCP resources/list result must be an object.")
            resources = result.get("resources", [])
            if type(resources) is not list:
                raise McpProtocolError("MCP resources/list result resources must be a list.")
            observed_items = parsed_resource_count + len(resources)
            if observed_items > self.max_list_items:
                result.clear()
                raise McpProtocolError(
                    f"MCP resources/list returned {observed_items} items, exceeding "
                    f"max_list_items={self.max_list_items}."
                )
            parse_failed = False
            definitions: list[McpResourceDefinition] = []
            try:
                definitions = [
                    resource_definition_from_payload(resource, self.server.name)
                    for resource in resources
                ]
            except (McpProtocolError, TypeError, ValueError):
                parse_failed = True
            finally:
                result.clear()
            if parse_failed:
                definitions.clear()
                raise McpProtocolError(
                    "MCP resources/list returned an invalid resource definition."
                ) from None
            parsed_resource_count = observed_items
            return {"resources": definitions}

        async def request_page(method: str, params: dict[str, Any]) -> Any:
            return await self._request(
                method,
                params,
                authority_mapping=transport_uris,
                paginated=True,
                result_parser=parse_resources_page,
            )

        try:
            resources = await collect_paginated(
                request_page,
                "resources/list",
                "resources",
                max_pages=self.max_list_pages,
                max_items=self.max_list_items,
                redactor=self._secret_redactor,
            )
        except BaseException:
            transport_uris.clear()
            raise
        definitions = tuple(resources)
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
        return await self._request(
            "resources/read",
            {"uri": self._resource_transport_uris.get(resource_uri, resource_uri)},
            result_parser=resource_result_from_payload,
        )

    async def close(self) -> None:
        # Run cleanup in a shielded task so the DELETE + aclose still complete even
        # if the caller is cancelled mid-close (mirrors StdioMcpSession.close).
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(self._close_impl())
        sanitized_cancellation: asyncio.CancelledError | None = None
        try:
            await _await_mcp_session_cleanup_task(
                self._close_task,
                redactor=self._secret_redactor,
                context="MCP HTTP session cleanup failed",
            )
        except asyncio.CancelledError as cancellation:
            if exception_cause(cancellation) is None:
                if self._client_close_failure is not None:
                    _attach_http_cleanup_failure(cancellation, self._client_close_failure)
                elif self._client_close_task is not None and not self._client_close_task.done():
                    _attach_http_settlement_task(cancellation, self._client_close_task)
            sanitized_cancellation = _credential_safe_mcp_cancellation(
                cancellation,
                redactor=self._secret_redactor,
            )
        if sanitized_cancellation is not None:
            raise sanitized_cancellation
        if self._client_close_failure is not None:
            raise self._client_close_failure from None

    def _fence_before_retained_close(self) -> bool:
        # Even when a subclass must finish its own close synchronously, the
        # inherited HTTP request entrances can be fenced immediately.
        self._closed = True
        # A subclass may add work or reuse entrances that ``_closed`` does not
        # fence. It must explicitly override this proof hook before discovery
        # cleanup can safely continue in the background.
        return type(self) is HttpMcpSession

    def _begin_failed_session_close(
        self,
        error: BaseException,
        *,
        terminate_fenced_session: bool = False,
    ) -> None:
        """Retain full cleanup after a failed session operation.

        A completed response that expires during semantic processing may
        terminate its known server-side session despite the logical fence. An
        uncertain transport exchange must retain its exact settlement instead.
        """

        if not terminate_fenced_session:
            # Ordinary initialization failures retain the historical close owner.
            # HttpMcpClient.connect() joins it for non-control failures, while its
            # bounded implementation preserves deadline/cancellation latency.
            if self._close_task is None:
                self._close_task = asyncio.create_task(
                    self._close_impl(report_server_cleanup_failure=True)
                )
                self._close_task.add_done_callback(_consume_task_result)
            _attach_http_settlement_task(error, self._close_task)
            return

        settlement_task = self._failed_close_settlement_task
        if settlement_task is None and self._close_task is None:
            settlement_task = asyncio.create_task(self._settle_failed_session_close())
            self._failed_close_settlement_task = settlement_task
            self._retain_settlement_task(settlement_task)
            self._close_task = asyncio.create_task(
                self._join_fenced_settlement_for_close(settlement_task)
            )
            self._close_task.add_done_callback(_consume_task_result)
        if settlement_task is None:
            # An explicit close already owns cleanup. Reuse that exact owner
            # rather than issuing a duplicate DELETE or client close.
            assert self._close_task is not None
            settlement_task = self._close_task
        _attach_http_settlement_task(error, settlement_task)

    async def _settle_failed_session_close(self) -> None:
        """Terminate a failed session, then close its client with full evidence."""

        cleanup_failures: list[BaseException] = []
        cleanup_session_id = self._session_id or self._cleanup_session_id
        cleanup_protocol_version = self._cleanup_protocol_version
        if cleanup_session_id is not None:
            delete_settlements: list[asyncio.Task[None]] = []
            cancellation_boundary = _McpCallerCancellationBoundary()
            try:
                await cancellation_boundary.checkpoint()
                await self._delete_session(
                    budget=_HttpCallBudget(self.transport_limits),
                    session_id=cleanup_session_id,
                    protocol_version=cleanup_protocol_version,
                    retained_settlements=delete_settlements,
                )
            except asyncio.CancelledError as error:
                if cancellation_boundary.caller_cancelled():
                    raise
                cleanup_failures.append(
                    credential_safe_mcp_transport_failure(
                        error,
                        redactor=self._secret_redactor,
                        context="MCP HTTP server-session termination failed",
                    )
                )
            except BaseException as error:
                cleanup_failures.append(
                    credential_safe_mcp_transport_failure(
                        error,
                        redactor=self._secret_redactor,
                        context="MCP HTTP server-session termination failed",
                    )
                )
            finally:
                cleanup_session_id = None
                cleanup_protocol_version = None

            # DELETE can transfer an uncertain response to a retained owner. It
            # must settle before the shared client closes, and its failure remains
            # secondary evidence on the original failed operation.
            for delete_settlement in delete_settlements:
                cancellation_boundary = _McpCallerCancellationBoundary()
                try:
                    await cancellation_boundary.checkpoint()
                    await asyncio.shield(delete_settlement)
                except asyncio.CancelledError as error:
                    if cancellation_boundary.caller_cancelled():
                        raise
                    cleanup_failures.append(
                        credential_safe_mcp_transport_failure(
                            error,
                            redactor=self._secret_redactor,
                            context="MCP HTTP server-session settlement failed",
                        )
                    )
                except BaseException as error:
                    # The retained settlement is runtime-owned and has already
                    # detached and redacted extension failures.
                    cleanup_failures.append(error)
            delete_settlements.clear()

        client_close_task = self._ensure_client_close_task()
        cancellation_boundary = _McpCallerCancellationBoundary()
        try:
            await cancellation_boundary.checkpoint()
            await asyncio.shield(client_close_task)
        except asyncio.CancelledError as error:
            if cancellation_boundary.caller_cancelled():
                raise
            cleanup_failures.append(
                credential_safe_mcp_transport_failure(
                    error,
                    redactor=self._secret_redactor,
                    context="MCP HTTP client cleanup failed",
                )
            )
        except BaseException as error:
            # The client-close owner sanitizes its own extension boundary.
            cleanup_failures.append(error)
        del client_close_task

        if len(cleanup_failures) == 1:
            raise cleanup_failures[0]
        if cleanup_failures:
            raise BaseExceptionGroup(
                "MCP HTTP failed-session cleanup failures.",
                cleanup_failures,
            )

    async def _close_impl(self, *, report_server_cleanup_failure: bool = False) -> None:
        budget = _HttpCallBudget(self.transport_limits)
        active_wait_task = asyncio.create_task(self._wait_for_active_http_exchanges())
        active_exchanges_settled = await _wait_for_http_close_task(
            active_wait_task,
            budget=budget,
        )
        if not active_exchanges_settled:
            # Keep close() bounded, but do not snapshot session authority, issue
            # DELETE, or close the shared client ahead of a dispatched exchange.
            # This retained owner resumes cleanup after every exact exchange owner
            # reaches a definite outcome; _closed already fences new publication.
            draining_close_task = asyncio.create_task(
                self._finish_close_after_active_exchanges(
                    active_wait_task,
                    report_server_cleanup_failure=report_server_cleanup_failure,
                )
            )
            self._retain_settlement_task(draining_close_task)
            return
        await self._close_after_active_exchanges(
            budget,
            report_server_cleanup_failure=report_server_cleanup_failure,
        )

    async def _finish_close_after_active_exchanges(
        self,
        active_wait_task: asyncio.Task[None],
        *,
        report_server_cleanup_failure: bool,
    ) -> None:
        await asyncio.shield(active_wait_task)
        await self._close_after_active_exchanges(
            _HttpCallBudget(self.transport_limits),
            report_server_cleanup_failure=report_server_cleanup_failure,
        )

    async def _close_after_active_exchanges(
        self,
        budget: _HttpCallBudget,
        *,
        report_server_cleanup_failure: bool,
    ) -> None:
        server_cleanup_failures: list[BaseException] = []
        delete_settlements: list[asyncio.Task[None]] = []
        # A response that completed after close was requested can carry newer
        # cleanup-only authority than the last publicly committed session ID.
        cleanup_session_id = self._cleanup_session_id or self._session_id
        cleanup_protocol_version = self._cleanup_protocol_version
        self._session_id = None
        self._cleanup_session_id = None
        self._cleanup_protocol_version = None
        try:
            if cleanup_session_id is not None and not self._fenced:
                # Best-effort session termination; the server may not allow it (405).
                cancellation_boundary = _McpCallerCancellationBoundary()
                await cancellation_boundary.checkpoint()
                await self._delete_session(
                    budget=budget,
                    session_id=cleanup_session_id,
                    protocol_version=cleanup_protocol_version,
                    retained_settlements=(
                        delete_settlements if report_server_cleanup_failure else None
                    ),
                )
        except asyncio.CancelledError as error:
            if cancellation_boundary.caller_cancelled():
                raise
            if report_server_cleanup_failure:
                server_cleanup_failures.append(
                    credential_safe_mcp_transport_failure(
                        error,
                        redactor=self._secret_redactor,
                        context="MCP HTTP server-session termination failed",
                        preserve_cause=True,
                    )
                )
            error = None
        except BaseException as error:
            # DELETE is advisory. Extension-produced exception groups (including
            # cancellation leaves) cannot skip the authoritative client close.
            if report_server_cleanup_failure:
                server_cleanup_failures.append(
                    credential_safe_mcp_transport_failure(
                        error,
                        redactor=self._secret_redactor,
                        context="MCP HTTP server-session termination failed",
                        preserve_cause=True,
                    )
                )
            error = None
        finally:
            # Fenced sessions intentionally skip DELETE because the preceding
            # exchange has an uncertain outcome. Their raw authority must still
            # disappear before client-close failure can capture this frame.
            cleanup_session_id = None
            cleanup_protocol_version = None
        if report_server_cleanup_failure:
            for delete_settlement in delete_settlements:
                cancellation_boundary = _McpCallerCancellationBoundary()
                try:
                    await cancellation_boundary.checkpoint()
                    await asyncio.shield(delete_settlement)
                except asyncio.CancelledError as error:
                    if cancellation_boundary.caller_cancelled():
                        raise
                    server_cleanup_failures.append(
                        credential_safe_mcp_transport_failure(
                            error,
                            redactor=self._secret_redactor,
                            context="MCP HTTP server-session settlement failed",
                            preserve_cause=True,
                        )
                    )
                    error = None
                except BaseException as error:
                    server_cleanup_failures.append(
                        credential_safe_mcp_transport_failure(
                            error,
                            redactor=self._secret_redactor,
                            context="MCP HTTP server-session settlement failed",
                            preserve_cause=True,
                        )
                    )
                    error = None
        delete_settlements.clear()
        client_close_failure = await _wait_for_http_client_close_task(
            self._ensure_client_close_task(),
            budget=budget,
        )
        pending_settlements = tuple(task for task in self._settlement_tasks if not task.done())
        current_task = asyncio.current_task()
        pending_settlements = tuple(
            task for task in pending_settlements if task is not current_task
        )
        if pending_settlements:
            # A close remains bounded even when an injected transport suppresses
            # cancellation. The retained settlement tasks continue to own that work
            # and the fenced session can never be reused.
            settlement_wait = asyncio.gather(
                *pending_settlements,
                return_exceptions=True,
            )
            try:
                await _wait_for_http_close_task(
                    settlement_wait,
                    budget=budget,
                )
            finally:
                if not settlement_wait.done():
                    settlement_wait.add_done_callback(_consume_task_result)
        cleanup_failures = list(server_cleanup_failures)
        server_cleanup_failures.clear()
        if client_close_failure is not None:
            cleanup_failures.append(client_close_failure)
        if len(cleanup_failures) == 1:
            raise cleanup_failures[0] from None
        if cleanup_failures:
            raise BaseExceptionGroup(
                "MCP HTTP session cleanup failures.",
                cleanup_failures,
            )

    def _ensure_client_close_task(self) -> asyncio.Task[None]:
        task = self._client_close_task
        if task is None:
            task = asyncio.create_task(self._close_http_client_with_safe_failure())
            task.add_done_callback(self._record_http_client_close_outcome)
            self._client_close_task = task
        return task

    async def _close_http_client_with_safe_failure(self) -> None:
        safe_failure: BaseException | None = None
        try:
            # A fenced sibling may initiate client cleanup while another healthy
            # request still owns an entered or entering response. Closing the
            # shared client is ordered after every registered exchange settles.
            await self._wait_for_active_http_exchanges()
            await self._http.aclose()
        except BaseException as error:
            safe_failure = credential_safe_mcp_transport_failure(
                error,
                redactor=self._secret_redactor,
                context="MCP HTTP client cleanup failed",
            )
            error = None
        if safe_failure is not None:
            raise safe_failure from None

    def _record_http_client_close_outcome(self, task: asyncio.Task[None]) -> None:
        try:
            task.result()
        except BaseException as error:
            self._client_close_failure = error

    def _retain_settlement_task(self, task: asyncio.Task[None]) -> None:
        self._settlement_tasks.add(task)

        def settled(completed: asyncio.Task[None]) -> None:
            self._settlement_tasks.discard(completed)
            _consume_task_result(completed)

        task.add_done_callback(settled)

    def _register_http_exchange(self, exchange_owner: _HttpExchangeOwner) -> None:
        settlement = asyncio.get_running_loop().create_future()
        exchange_owner.settlement = settlement
        self._active_exchange_settlements.add(settlement)
        settlement.add_done_callback(self._active_exchange_settlements.discard)

    async def _wait_for_active_http_exchanges(self) -> None:
        # _closed is set before cleanup starts, so no new exchange can register
        # after the current task yields. Loop defensively for an owner that passed
        # its entrance check immediately before the fence.
        while True:
            pending = tuple(
                settlement
                for settlement in self._active_exchange_settlements
                if not settlement.done()
            )
            if not pending:
                return
            await asyncio.wait(pending)

    def _begin_fenced_settlement(
        self,
        *,
        exchange_owner: _HttpExchangeOwner,
        abandoned_tasks: Sequence[asyncio.Future[Any]],
        error: BaseException,
        redactor: SecretRedactor,
        initialize_request: bool,
        already_reported_failure_ids: frozenset[int] = frozenset(),
    ) -> None:
        self._closed = True
        self._fenced = True
        initialization_cleanup: Callable[[], Coroutine[Any, Any, None]] | None = None
        if self._initialize_result is None:
            cleanup_session_id = self._session_id or self._cleanup_session_id
            cleanup_protocol_version = (
                self._cleanup_protocol_version or self._negotiated_protocol_version
            )
            if cleanup_session_id is not None:
                # Transfer the cleanup-only authority into the retained owner so
                # initialize(), close(), and client.connect() cannot issue a
                # duplicate DELETE or close the client ahead of this exchange.
                self._session_id = None
                self._cleanup_session_id = None
                self._cleanup_protocol_version = None

            async def terminate_initialization_session() -> None:
                nonlocal cleanup_session_id, cleanup_protocol_version
                resolved_session_id = cleanup_session_id
                resolved_protocol_version = cleanup_protocol_version
                cleanup_session_id = None
                cleanup_protocol_version = None
                late_response: httpx.Response | None = None
                if resolved_session_id is None and initialize_request:
                    # A cancellation-resistant transport may publish response
                    # headers only after the public initialization deadline. The
                    # retained enter owner settles before this callback runs, so
                    # capture its cleanup-only authority before closing the client.
                    late_response = exchange_owner.response
                    if late_response is not None and 200 <= late_response.status_code < 300:
                        resolved_session_id = late_response.headers.get(MCP_SESSION_ID_HEADER)
                    late_response = None
                if resolved_session_id is None:
                    return
                delete_settlements: list[asyncio.Task[None]] = []
                cleanup_failures: list[BaseException] = []
                try:
                    await self._delete_session(
                        budget=_HttpCallBudget(self.transport_limits),
                        session_id=resolved_session_id,
                        protocol_version=resolved_protocol_version,
                        retained_settlements=delete_settlements,
                    )
                except BaseException as cleanup_error:
                    cleanup_failures.append(cleanup_error)
                finally:
                    resolved_session_id = None
                    resolved_protocol_version = None
                # DELETE may itself transfer an uncertain response into a
                # nested retained settlement. Do not close the shared client
                # until that exact owner has finished.
                for delete_settlement in delete_settlements:
                    try:
                        await asyncio.shield(delete_settlement)
                    except BaseException as cleanup_error:
                        cleanup_failures.append(cleanup_error)
                delete_settlements.clear()
                if len(cleanup_failures) == 1:
                    raise cleanup_failures[0]
                if cleanup_failures:
                    raise BaseExceptionGroup(
                        "MCP HTTP initialization cleanup failures.",
                        cleanup_failures,
                    )

            if cleanup_session_id is not None or initialize_request:
                initialization_cleanup = terminate_initialization_session
        client_close_task: asyncio.Task[None] | None = None
        client_close_factory: Callable[[], asyncio.Task[None]] | None = None
        if initialization_cleanup is None:
            client_close_task = self._ensure_client_close_task()
        else:
            client_close_factory = self._ensure_client_close_task
        settlement_task = asyncio.create_task(
            _settle_registered_http_exchange(
                exchange_owner=exchange_owner,
                abandoned_tasks=abandoned_tasks,
                redactor=redactor,
                already_reported_failure_ids=already_reported_failure_ids,
                initialization_cleanup=initialization_cleanup,
                client_close_task=client_close_task,
                client_close_factory=client_close_factory,
            )
        )
        self._retain_settlement_task(settlement_task)
        if initialization_cleanup is not None and self._close_task is None:
            # close() must join this exact owner rather than start client close
            # concurrently while response/session cleanup is still in flight.
            # Keep the public close wait bounded even if an extension transport
            # never settles; the retained task continues to own that work.
            self._close_task = asyncio.create_task(
                self._join_fenced_settlement_for_close(settlement_task)
            )
            self._close_task.add_done_callback(_consume_task_result)
        _attach_http_settlement_task(error, settlement_task)

    async def _join_fenced_settlement_for_close(
        self,
        settlement_task: asyncio.Task[None],
    ) -> None:
        await _wait_for_http_close_task(
            settlement_task,
            budget=_HttpCallBudget(self.transport_limits),
        )

    def _raise_completed_response_deadline(
        self,
        *,
        method_name: str,
        redactor: SecretRedactor,
    ) -> None:
        """Fence a response that completed transport cleanup but missed its deadline."""

        error = McpCallDeadlineExceededError(
            "MCP HTTP exchange exceeded its total call deadline during response processing."
        )
        self._closed = True
        self._fenced = True
        if self._session_id is not None or self._cleanup_session_id is not None:
            # Transport cleanup has completed and the session id is known.
            # Retain bounded DELETE-before-client-close ownership without
            # extending the public request deadline.
            self._begin_failed_session_close(
                error,
                terminate_fenced_session=True,
            )
        else:
            _attach_http_settlement_task(error, self._ensure_client_close_task())
        safe_error = credential_safe_mcp_transport_failure(
            error,
            redactor=redactor,
            context=f"MCP {method_name} request timed out",
            preserve_cause=True,
        )
        raise safe_error from None

    async def _request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        authority_mapping: dict[str, str] | None = None,
        private_tool_contract_hashes: list[str] | None = None,
        paginated: bool = False,
        result_parser: Callable[[Any], Any] | None = None,
        dispatch_signal: _McpToolDispatchSignal | None = None,
    ) -> Any:
        budget = _HttpCallBudget(self.transport_limits)
        method_name = require_clean_nonblank(method, "method")
        failure_redactor = self._secret_redactor
        private_request_cursor = params.get("cursor") if paginated else None
        if type(private_request_cursor) is str and private_request_cursor.strip():
            failure_redactor = failure_redactor.with_secret(private_request_cursor)
        private_request_cursor = None
        request_id = self._next_id
        self._next_id += 1
        sanitized_error: BaseException | None = None
        response_envelope: _HttpResponseEnvelope | None = None
        try:
            try:
                if self._initializing and method_name != "initialize":
                    raise McpProtocolError("MCP HTTP session initialization is still in progress.")
                request_preflight = mcp_jsonrpc_request_preflight(
                    request_id,
                    method_name,
                    params,
                    max_bytes=self.transport_limits.max_message_bytes,
                )
                if request_preflight.exceeds_limit:
                    size_error = McpMessageTooLargeError(
                        "MCP HTTP JSON-RPC request exceeded "
                        f"{self.transport_limits.max_message_bytes} bytes."
                    )
                    raise size_error from None
                if request_preflight.nesting_too_deep:
                    raise McpProtocolError(
                        "MCP HTTP JSON-RPC request exceeded the supported JSON nesting."
                    ) from None
                budget.check_total_deadline()
                payload = jsonrpc_request_payload(request_id, method_name, params)
            finally:
                # This shallow container is transport-owned. Scrub every preparation
                # exit without mutating nested dictionaries still owned by the caller.
                params.clear()
            if dispatch_signal is not None:
                # There is no await between this mark and _send() registering its
                # owned HTTP exchange. Once this task yields, the request may have
                # reached the remote target and must be allowed to settle.
                dispatch_signal.mark_dispatched()
            send_result = await self._send(
                payload,
                request_id,
                budget=budget,
                failure_redactor=failure_redactor,
            )
            # Preserve the long-standing internal test/extension seam whose
            # injected send doubles return the raw JSON-RPC object.
            if type(send_result) is _HttpResponseEnvelope:
                response_envelope = send_result
            else:
                response_envelope = _HttpResponseEnvelope(cast("dict[str, Any]", send_result))
            send_result = None
            if self._closed:
                self._cleanup_session_id = (
                    response_envelope.take_session_id() or self._cleanup_session_id
                )
                response_envelope.message.clear()
                response_envelope.clear_private_authority()
                raise McpProtocolError(
                    "MCP HTTP session closed before the response could be published."
                )
        except asyncio.CancelledError as exc:
            sanitized_error = _credential_safe_mcp_cancellation(
                exc,
                redactor=failure_redactor,
            )
        except TimeoutError as exc:
            sanitized_error = credential_safe_mcp_transport_failure(
                exc,
                redactor=failure_redactor,
                context=f"MCP {method_name} request timed out",
                preserve_cause=True,
            )
        except BaseExceptionGroup as exc:
            sanitized_error = credential_safe_mcp_transport_failure(
                exc,
                redactor=failure_redactor,
                context=f"MCP {method_name} request failed",
                preserve_cause=True,
            )
        except (RecursionError, TypeError, ValueError) as exc:
            sanitized_error = credential_safe_mcp_transport_failure(
                exc,
                redactor=failure_redactor,
                context=f"MCP {method_name} request preparation failed",
            )
        except McpProtocolError as exc:
            sanitized_error = credential_safe_mcp_transport_failure(
                exc,
                redactor=failure_redactor,
                context=f"MCP {method_name} request failed",
                preserve_cause=True,
            )
        except (KeyboardInterrupt, SystemExit, GeneratorExit) as exc:
            sanitized_error = credential_safe_mcp_fatal_signal(
                exc,
                redactor=failure_redactor,
                context=f"MCP {method_name} request failed",
            )
        if sanitized_error is not None:
            raise sanitized_error
        assert response_envelope is not None
        if method_name == "initialize":
            # A malformed initialize response may still have allocated a
            # server-side session. Keep its ID cleanup-only until the complete
            # initialize result passes semantic validation.
            self._cleanup_session_id = response_envelope.staged_session_id
            self._cleanup_protocol_version = (
                _initialize_cleanup_protocol_version(response_envelope.message)
                if self._cleanup_session_id is not None
                else None
            )
        message = response_envelope.message
        if mcp_json_value_nesting_too_deep(message):
            # The response is complete and its transport owner has settled, so an
            # ordinary HTTP session remains reusable. Initialization still owns
            # the cleanup-only session authority staged above.
            message.clear()
            message = {}
            response_envelope.clear_private_authority()
            raise McpProtocolError(
                "MCP HTTP JSON-RPC response exceeded the supported JSON nesting."
            ) from None
        redaction_result = safely_redact_jsonrpc_response(
            message,
            method=method_name,
            redactor=self._secret_redactor,
        )
        redacted_message = redaction_result.response
        mapping_result = JsonrpcAuthorityMappingResult({})
        mapping_error = redaction_result.error
        private_evidence_error: str | None = None
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
        if mapping_error is None and private_tool_contract_hashes is not None:
            evidence_result = jsonrpc_tool_contract_evidence(
                message,
                server_name=self.server.name,
            )
            private_evidence_error = evidence_result.error
            if private_evidence_error is None:
                private_tool_contract_hashes.extend(evidence_result.hashes)
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
        if mapping_error is not None and private_tool_contract_hashes is not None:
            private_tool_contract_hashes.clear()
        mapping_result.mapping.clear()
        raw_result = None
        # The redacted response is the only representation needed after
        # authority extraction. Do not retain the private server payload in a
        # traceback if validation or result parsing fails.
        message.clear()
        message = {}
        if budget.total_deadline_expired():
            if authority_mapping is not None:
                authority_mapping.clear()
            private_cursor = None
            redacted_message.clear()
            response_envelope.clear_private_authority()
            self._raise_completed_response_deadline(
                method_name=method_name,
                redactor=failure_redactor,
            )
        if mapping_error is not None:
            private_cursor = None
            response_envelope.clear_private_authority()
            raise McpProtocolError(
                self._secret_redactor.redact_text_bounded(
                    mapping_error,
                    max_bytes=4096,
                )
            ) from None
        result: Any = None
        try:
            result = result_from_jsonrpc_response(redacted_message, method_name)
            redacted_message.clear()
            redaction_result = None
            if result_parser is not None:
                result = result_parser(result)
        except BaseException:
            private_cursor = None
            redacted_message.clear()
            redaction_result = None
            if type(result) in {dict, list}:
                result.clear()
            result = None
            response_envelope.clear_private_authority()
            if budget.total_deadline_expired():
                if authority_mapping is not None:
                    authority_mapping.clear()
                redacted_message.clear()
                self._raise_completed_response_deadline(
                    method_name=method_name,
                    redactor=failure_redactor,
                )
            raise
        if budget.total_deadline_expired():
            if authority_mapping is not None:
                authority_mapping.clear()
            private_cursor = None
            if type(result) in {dict, list}:
                result.clear()
            redacted_message.clear()
            response_envelope.clear_private_authority()
            self._raise_completed_response_deadline(
                method_name=method_name,
                redactor=failure_redactor,
            )
        if private_evidence_error is not None:
            if private_tool_contract_hashes is not None:
                private_tool_contract_hashes.clear()
            private_cursor = None
            if type(result) in {dict, list}:
                result.clear()
            response_envelope.clear_private_authority()
            raise McpProtocolError(private_evidence_error) from None
        staged_session_id = response_envelope.take_session_id()
        if method_name == "initialize":
            if staged_session_id:
                self._cleanup_session_id = staged_session_id
        elif staged_session_id:
            self._session_id = staged_session_id
        if not paginated:
            return result
        if type(result) is dict:
            result.pop("nextCursor", None)
        return McpPaginatedPage(
            result=result,
            next_cursor=private_cursor,
        )

    async def _notify(
        self,
        method: str,
        params: dict[str, Any],
        *,
        protocol_version: str | None = None,
        session_id: str | None = None,
    ) -> None:
        method_name = require_clean_nonblank(method, "method")
        budget = _HttpCallBudget(self.transport_limits)
        sanitized_error: BaseException | None = None
        response_envelope: _HttpResponseEnvelope | None = None
        try:
            response_envelope = await self._send(
                jsonrpc_notification_payload(method_name, params),
                None,
                budget=budget,
                protocol_version=protocol_version,
                session_id=session_id,
            )
            if self._closed:
                self._cleanup_session_id = (
                    response_envelope.take_session_id() or self._cleanup_session_id
                )
                response_envelope.message.clear()
                response_envelope.clear_private_authority()
                raise McpProtocolError(
                    "MCP HTTP session closed before the notification could be published."
                )
        except asyncio.CancelledError as exc:
            sanitized_error = _credential_safe_mcp_cancellation(
                exc,
                redactor=self._secret_redactor,
            )
        except TimeoutError as exc:
            sanitized_error = credential_safe_mcp_transport_failure(
                exc,
                redactor=self._secret_redactor,
                context=f"MCP {method_name} notification timed out",
                preserve_cause=True,
            )
        except BaseExceptionGroup as exc:
            sanitized_error = credential_safe_mcp_transport_failure(
                exc,
                redactor=self._secret_redactor,
                context=f"MCP {method_name} notification failed",
                preserve_cause=True,
            )
        except McpProtocolError as exc:
            sanitized_error = credential_safe_mcp_transport_failure(
                exc,
                redactor=self._secret_redactor,
                context=f"MCP {method_name} notification failed",
                preserve_cause=True,
            )
        except (KeyboardInterrupt, SystemExit, GeneratorExit) as exc:
            sanitized_error = credential_safe_mcp_fatal_signal(
                exc,
                redactor=self._secret_redactor,
                context=f"MCP {method_name} notification failed",
            )
        finally:
            # A cleanup-only session ID can be a server echo of workload-secret
            # material. Never retain it in the public notification traceback.
            session_id = None
            protocol_version = None
        if sanitized_error is not None:
            raise sanitized_error
        assert response_envelope is not None
        staged_session_id = response_envelope.take_session_id()
        if staged_session_id:
            if self._initializing and method_name == "notifications/initialized":
                self._cleanup_session_id = staged_session_id
            else:
                self._session_id = staged_session_id

    async def _send(
        self,
        payload: dict[str, Any],
        request_id: int | None,
        *,
        budget: _HttpCallBudget,
        failure_redactor: SecretRedactor | None = None,
        protocol_version: str | None = None,
        session_id: str | None = None,
    ) -> _HttpResponseEnvelope:
        # Stream the response so an SSE reply is returned as soon as the matching
        # JSON-RPC event arrives, without waiting for the server to close the stream.
        failure_redactor = failure_redactor or self._secret_redactor
        initialize_request = payload.get("method") == "initialize"
        content = b""
        try:
            if self._closed:
                raise McpProtocolError("MCP HTTP session is closed.")
            budget.check_total_deadline()
            if not json_utf8_size_within_limit(
                payload,
                self.transport_limits.max_message_bytes,
                ensure_ascii=True,
            ):
                raise McpMessageTooLargeError(
                    "MCP HTTP JSON-RPC request exceeded "
                    f"{self.transport_limits.max_message_bytes} bytes."
                )
            content = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        finally:
            # ``payload`` is the defensive transport copy. Do not retain it when
            # deadline or serialization validation fails before dispatch.
            payload.clear()
        if len(content) > self.transport_limits.max_message_bytes:
            content = b""
            raise McpMessageTooLargeError(
                "MCP HTTP JSON-RPC request exceeded "
                f"{self.transport_limits.max_message_bytes} bytes."
            )
        # Size preflight and serialization are part of the absolute lifecycle,
        # but they cannot make the remote outcome uncertain. Reject an overrun
        # before creating the stream context so the untouched session remains
        # reusable for ordinary calls.
        try:
            budget.check_total_deadline()
        except McpCallDeadlineExceededError:
            content = b""
            raise
        response: httpx.Response | None = None
        exchange_owner: _HttpExchangeOwner | None = None
        dispatch_started = False
        primary_failed = True
        primary_error: BaseException | None = None
        reuse_forbidden = False
        deferred_transport_error: BaseException | None = None
        deferred_transport_cause: BaseException | None = None
        staged_session_id: str | None = None
        content_type = ""
        try:
            stream_context = self._http.stream(
                "POST",
                self._url,
                content=content,
                headers=self._protocol_headers(
                    protocol_version=protocol_version,
                    session_id=session_id,
                ),
                follow_redirects=False,
            )
            exchange_owner = _HttpExchangeOwner(stream_context)
            self._register_http_exchange(exchange_owner)

            def record_dispatch_started() -> None:
                nonlocal dispatch_started
                dispatch_started = True

            response = await budget.wait(
                stream_context.__aenter__(),
                on_started=record_dispatch_started,
            )
            exchange_owner.record_entered_response(response)
            budget.note_activity()
            self._record_response_ownership(
                response,
                initialize_request=initialize_request,
            )
            _validate_http_response_headers(response, limits=self.transport_limits)
            await self._handle_status(
                response,
                budget=budget,
                redactor=failure_redactor,
            )
            staged_session_id = response.headers.get(MCP_SESSION_ID_HEADER)
            if request_id is None:
                await _consume_http_notification_response(
                    response,
                    budget=budget,
                    limits=self.transport_limits,
                )
                primary_failed = False
                return _HttpResponseEnvelope({}, staged_session_id=staged_session_id)
            content_type = response.headers.get("content-type", "")
            # Media types are case-insensitive (RFC 9110); httpx returns it as sent.
            try:
                if content_type.split(";", 1)[0].strip().lower() == _SSE_CONTENT_TYPE:
                    message = await _read_sse_response(
                        response,
                        request_id,
                        initialize_request=initialize_request,
                        redactor=failure_redactor,
                        limits=self.transport_limits,
                        budget=budget,
                    )
                else:
                    message = await _read_json_response(
                        response,
                        budget=budget,
                        limits=self.transport_limits,
                    )
                    if message.get("id") != request_id:
                        cleanup_protocol_version = (
                            _initialize_cleanup_protocol_version(message)
                            if initialize_request
                            else None
                        )
                        message.clear()
                        message = {}
                        raise _McpHttpResponseContentError(
                            "MCP HTTP response id did not match the request.",
                            cleanup_protocol_version=cleanup_protocol_version,
                        )
            except _McpHttpResponseContentError as content_error:
                if initialize_request and self._cleanup_session_id is not None:
                    # A decoded rejection can strengthen cleanup authority with
                    # the server's supported protocol version without publishing
                    # the rejected initialize result.
                    self._cleanup_protocol_version = content_error.cleanup_protocol_version
                content_error.cleanup_protocol_version = None
                raise
            primary_failed = False
            return _HttpResponseEnvelope(
                message,
                staged_session_id=staged_session_id,
            )
        except httpx.TimeoutException:
            reuse_forbidden = True
            message = failure_redactor.redact_text(
                f"MCP HTTP request to {self._url} exceeded its idle timeout."
            )
            primary_error = McpIdleTimeoutError(message)
            deferred_transport_error = primary_error
        except httpx.RemoteProtocolError:
            reuse_forbidden = True
            primary_error = McpPeerClosedError(
                "MCP HTTP peer closed before completing the response."
            )
            deferred_transport_error = primary_error
        except httpx.RequestError as exc:
            reuse_forbidden = True
            safe_transport_error = credential_safe_mcp_transport_failure(
                exc,
                redactor=failure_redactor,
                context="MCP HTTP request failed",
                preserve_cause=True,
            )
            primary_error = safe_transport_error
            deferred_transport_error = primary_error
        except _McpUnexpectedTransportCancellationError as exc:
            reuse_forbidden = True
            primary_error = McpProtocolError(str(exc))
            deferred_transport_error = primary_error
        except McpPeerClosedError as exc:
            reuse_forbidden = True
            primary_error = exc
            raise
        except (
            McpCallDeadlineExceededError,
            McpIdleTimeoutError,
            asyncio.CancelledError,
        ) as exc:
            reuse_forbidden = True
            primary_error = exc
            raise
        except McpProtocolError as exc:
            if not dispatch_started or not budget.total_deadline_expired():
                primary_error = exc
                raise
            safe_protocol_error = credential_safe_mcp_transport_failure(
                exc,
                redactor=failure_redactor,
                context="MCP HTTP response processing failed",
                preserve_cause=True,
            )
            primary_error = McpCallDeadlineExceededError(
                "MCP HTTP exchange exceeded its total call deadline during response processing."
            )
            reuse_forbidden = dispatch_started
            deferred_transport_error = primary_error
            deferred_transport_cause = safe_protocol_error
        except Exception as exc:
            primary_error = McpProtocolError("MCP HTTP transport operation failed; session closed.")
            safe_transport_error = credential_safe_mcp_transport_failure(
                exc,
                redactor=failure_redactor,
                context="MCP HTTP transport operation failed",
            )
            _attach_http_cleanup_failure(primary_error, safe_transport_error)
            reuse_forbidden = True
            deferred_transport_error = primary_error
            deferred_transport_cause = safe_transport_error
        except (BaseExceptionGroup, KeyboardInterrupt, SystemExit, GeneratorExit) as exc:
            # Current caller cancellation is handled above. Process-control and
            # grouped transport signals retain their public classification, but
            # their uncertain outcome must not leave the logical session reusable.
            reuse_forbidden = True
            primary_error = exc
            raise
        except BaseException as exc:
            # An arbitrary scalar BaseException is extension failure evidence,
            # not a process-control signal understood by Cayu. Detach it before
            # retaining settlement ownership or crossing the public boundary.
            safe_transport_error = credential_safe_mcp_transport_failure(
                exc,
                redactor=failure_redactor,
                context="MCP HTTP transport operation failed",
            )
            reuse_forbidden = True
            primary_error = safe_transport_error
            deferred_transport_error = safe_transport_error
            exc = None
        finally:
            content = b""
            # Response authority is staged above only after the bounded body is
            # accepted. The request owner commits it after semantic validation.
            # Do not retain rejected header values in a body/cleanup traceback.
            staged_session_id = None
            content_type = ""
            if exchange_owner is not None:
                if response is not None and not (primary_failed and reuse_forbidden):
                    deferred_cleanup_error: BaseException | None = None
                    deferred_cleanup_cause: BaseException | None = None
                    exit_task = exchange_owner.start_exit()
                    try:
                        # The absolute call budget owns the entered stream context's
                        # normal exit. If it expires, that same exit continues under
                        # retained settlement ownership.
                        await budget.wait_for_owned_task(exit_task)
                        exchange_owner.mark_settled()
                    except asyncio.CancelledError as cancellation:
                        if primary_error is not None:
                            _attach_http_cleanup_failure(cancellation, primary_error)
                        self._begin_fenced_settlement(
                            exchange_owner=exchange_owner,
                            abandoned_tasks=budget.take_abandoned_tasks(),
                            error=cancellation,
                            redactor=failure_redactor,
                            initialize_request=initialize_request,
                        )
                        exchange_owner = None
                        exit_task = None
                        response = None
                        stream_context = None
                        raise
                    except (McpCallDeadlineExceededError, McpIdleTimeoutError) as timeout_error:
                        authoritative_error = primary_error or timeout_error
                        if primary_error is not None:
                            _attach_http_cleanup_failure(primary_error, timeout_error)
                        self._begin_fenced_settlement(
                            exchange_owner=exchange_owner,
                            abandoned_tasks=budget.take_abandoned_tasks(),
                            error=authoritative_error,
                            redactor=failure_redactor,
                            initialize_request=initialize_request,
                        )
                        exchange_owner = None
                        exit_task = None
                        response = None
                        stream_context = None
                        if primary_error is None:
                            if request_id is not None:
                                message.clear()
                                message = {}
                            raise
                    except BaseException as cleanup_error:
                        safe_cleanup_error = credential_safe_mcp_transport_failure(
                            cleanup_error,
                            redactor=failure_redactor,
                            context="MCP HTTP response cleanup failed",
                        )
                        if primary_error is None:
                            cleanup_failure = McpProtocolError(
                                "MCP HTTP response cleanup failed; session closed."
                            )
                            _attach_http_cleanup_failure(cleanup_failure, safe_cleanup_error)
                            authoritative_error = cleanup_failure
                        else:
                            _attach_http_cleanup_failure(primary_error, safe_cleanup_error)
                            authoritative_error = primary_error
                        self._begin_fenced_settlement(
                            exchange_owner=exchange_owner,
                            abandoned_tasks=budget.take_abandoned_tasks(),
                            error=authoritative_error,
                            redactor=failure_redactor,
                            initialize_request=initialize_request,
                            already_reported_failure_ids=frozenset({id(cleanup_error)}),
                        )
                        if primary_error is None:
                            deferred_cleanup_error = authoritative_error
                            deferred_cleanup_cause = safe_cleanup_error
                        exchange_owner = None
                        exit_task = None
                        response = None
                        stream_context = None
                    if deferred_cleanup_error is not None:
                        raise deferred_cleanup_error from deferred_cleanup_cause
                elif dispatch_started and primary_failed and primary_error is not None:
                    # The exact entered-or-entering context remains the settlement
                    # owner even when no response handle was available at the public
                    # deadline or cancellation boundary.
                    self._begin_fenced_settlement(
                        exchange_owner=exchange_owner,
                        abandoned_tasks=budget.take_abandoned_tasks(),
                        error=primary_error,
                        redactor=failure_redactor,
                        initialize_request=initialize_request,
                    )
                    exchange_owner = None
                    response = None
                    stream_context = None
                else:
                    # The budget rejected the enter awaitable before dispatch.
                    exchange_owner.mark_settled()

        if deferred_transport_error is not None:
            # Raise after leaving the raw transport exception handler so Python
            # cannot retain that private exception as an implicit __context__.
            raise deferred_transport_error from deferred_transport_cause
        raise AssertionError("MCP HTTP transport returned without a result or error.")

    def _record_response_ownership(
        self,
        response: httpx.Response,
        *,
        initialize_request: bool,
    ) -> None:
        """Record status/session ownership before representation validation."""

        if response.status_code == 404:
            # Status is authoritative even when representation headers are
            # malformed. Never let a rejected body leave an expired logical
            # session reusable.
            self._session_id = None
            self._cleanup_session_id = None
            self._cleanup_protocol_version = None
            self._closed = True
        if initialize_request and 200 <= response.status_code < 300:
            # A successful initialize response can allocate a server session
            # independently of whether Cayu accepts its body representation.
            # Keep this authority cleanup-only until initialization publishes.
            cleanup_session_id = response.headers.get(MCP_SESSION_ID_HEADER)
            if cleanup_session_id:
                self._cleanup_session_id = cleanup_session_id
                self._cleanup_protocol_version = None

    async def _handle_status(
        self,
        response: httpx.Response,
        *,
        budget: _HttpCallBudget,
        redactor: SecretRedactor,
    ) -> None:
        content_type = response.headers.get("content-type", "")
        media_type = content_type.split(";", 1)[0].strip().lower()
        content_type = ""
        json_message = media_type == _JSON_CONTENT_TYPE
        sse_events = media_type == _SSE_CONTENT_TYPE
        media_type = ""
        if response.status_code == 404:
            # The session is gone (spec): poison the session so callers can't keep
            # using it, and drop the dead id so close() skips the doomed DELETE.
            self._session_id = None
            self._cleanup_session_id = None
            self._cleanup_protocol_version = None
            self._closed = True
            await _read_http_error_body(
                response,
                budget=budget,
                limits=self.transport_limits,
                json_message=json_message,
                sse_events=sse_events,
                retain=False,
            )
            raise _McpHttpStatusError("MCP HTTP session expired or was not found (HTTP 404).")
        if not 200 <= response.status_code < 300:
            body, source_complete = await _read_http_error_body(
                response,
                budget=budget,
                limits=self.transport_limits,
                json_message=json_message,
                sse_events=sse_events,
                retain=True,
            )
            try:
                detail = _safe_body(
                    body,
                    redactor=redactor,
                    source_complete=source_complete,
                )
            finally:
                body = b""
                source_complete = False
            raise _McpHttpStatusError(
                f"MCP HTTP request failed with HTTP {response.status_code}: {detail}"
            )

    async def _delete_session(
        self,
        *,
        budget: _HttpCallBudget,
        session_id: str,
        protocol_version: str | None = None,
        retained_settlements: list[asyncio.Task[None]] | None = None,
    ) -> None:
        response: httpx.Response | None = None
        exchange_owner: _HttpExchangeOwner | None = None
        try:
            stream_context = self._http.stream(
                "DELETE",
                self._url,
                headers=self._protocol_headers(
                    protocol_version=protocol_version,
                    session_id=session_id,
                ),
                follow_redirects=False,
            )
            exchange_owner = _HttpExchangeOwner(stream_context)
            self._register_http_exchange(exchange_owner)
            response = await budget.wait(stream_context.__aenter__())
            exchange_owner.record_entered_response(response)
            budget.note_activity()
            known_status_failure: _McpHttpStatusError | None = None
            if not 200 <= response.status_code < 300:
                known_status_failure = _McpHttpStatusError(
                    f"MCP HTTP request failed with HTTP {response.status_code}."
                )
            try:
                _validate_http_response_headers(response, limits=self.transport_limits)
                await self._handle_status(
                    response,
                    budget=budget,
                    redactor=self._secret_redactor,
                )
            except _McpHttpStatusError:
                # _handle_status() already retained the known status and the
                # complete bounded diagnostic body.
                known_status_failure = None
                raise
            except BaseException as representation_failure:
                if known_status_failure is None:
                    raise
                raise BaseExceptionGroup(
                    "MCP HTTP session termination response failures.",
                    [known_status_failure, representation_failure],
                ) from None
            finally:
                known_status_failure = None
            await _read_http_body(response, budget=budget, retain=False)
        finally:
            session_id = ""
            protocol_version = None
            abandoned_tasks = budget.take_abandoned_tasks()
            # A caller collecting the nested settlement remains the diagnostic
            # owner for the shared client-close task. The nested exchange still
            # performs and awaits that exact close, but must not report its
            # failure before the caller observes the same memoized task.
            report_nested_client_close_failure = retained_settlements is None
            if exchange_owner is not None and abandoned_tasks:
                self._fenced = True
                settlement_task = asyncio.create_task(
                    _settle_registered_http_exchange(
                        exchange_owner=exchange_owner,
                        client_close_factory=(
                            self._ensure_client_close_task if retained_settlements is None else None
                        ),
                        abandoned_tasks=abandoned_tasks,
                        redactor=self._secret_redactor,
                        defer_client_close_to_parent=retained_settlements is not None,
                        report_client_close_failure=report_nested_client_close_failure,
                    )
                )
                self._retain_settlement_task(settlement_task)
                if retained_settlements is not None:
                    retained_settlements.append(settlement_task)
            elif exchange_owner is not None and exchange_owner.entered:
                exit_task = exchange_owner.start_exit()
                exit_cancelled = False
                try:
                    exit_completed = await _wait_for_http_close_task(
                        exit_task,
                        budget=budget,
                    )
                except asyncio.CancelledError:
                    exit_completed = False
                    exit_cancelled = True
                if not exit_completed:
                    self._fenced = True
                    settlement_task = asyncio.create_task(
                        _settle_registered_http_exchange(
                            exchange_owner=exchange_owner,
                            client_close_factory=(
                                self._ensure_client_close_task
                                if retained_settlements is None
                                else None
                            ),
                            abandoned_tasks=(),
                            redactor=self._secret_redactor,
                            defer_client_close_to_parent=retained_settlements is not None,
                            report_client_close_failure=report_nested_client_close_failure,
                        )
                    )
                    self._retain_settlement_task(settlement_task)
                    if retained_settlements is not None:
                        retained_settlements.append(settlement_task)
                else:
                    exchange_owner.mark_settled()
                if exit_cancelled:
                    raise asyncio.CancelledError
            elif exchange_owner is not None:
                # Enter completed with failure and no response context exists.
                exchange_owner.mark_settled()

    def _protocol_headers(
        self,
        *,
        protocol_version: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, str]:
        headers: dict[str, str] = {"accept-encoding": "identity"}
        # The spec requires MCP-Protocol-Version on requests AFTER initialization
        # (the negotiated version); sending it on the initialize request itself can
        # make a server 400 before version negotiation, so omit it until initialized.
        if protocol_version is not None:
            headers[MCP_PROTOCOL_VERSION_HEADER] = protocol_version
        elif self._initialize_result is not None:
            headers[MCP_PROTOCOL_VERSION_HEADER] = self._initialize_result.protocol_version
        elif self._negotiated_protocol_version is not None:
            headers[MCP_PROTOCOL_VERSION_HEADER] = self._negotiated_protocol_version
        resolved_session_id = self._session_id if session_id is None else session_id
        if resolved_session_id is not None:
            headers[MCP_SESSION_ID_HEADER] = resolved_session_id
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
    cleanup_protocol_version: str | None = None
    try:
        payload = json.loads(text)
    except RecursionError:
        protocol_error = "MCP JSON-RPC response exceeded the supported JSON nesting."
    except ValueError:
        protocol_error = "MCP HTTP response was not valid JSON."
    # Drop the raw document before any public parser error is raised. A
    # syntactically valid but structurally invalid response can carry the same
    # injected secrets as an undecodable response.
    text = ""
    if protocol_error is None:
        if type(payload) is dict:
            cleanup_protocol_version = _initialize_cleanup_protocol_version(payload)
        if type(payload) is not dict:
            protocol_error = "MCP JSON-RPC message must be an object."
        elif mcp_json_value_nesting_too_deep(payload):
            protocol_error = "MCP JSON-RPC response exceeded the supported JSON nesting."
        elif payload.get("jsonrpc") != "2.0":
            protocol_error = "MCP JSON-RPC message must use jsonrpc='2.0'."
    if protocol_error is not None:
        if type(payload) is dict:
            payload.clear()
        payload = None
        raise _McpHttpResponseContentError(
            protocol_error,
            cleanup_protocol_version=cleanup_protocol_version,
        ) from None
    return payload


def _decode_jsonrpc_bytes(data: bytes) -> dict[str, Any]:
    try:
        text = data.decode("utf-8", "strict")
    except UnicodeDecodeError:
        raise _McpHttpResponseContentError("MCP HTTP response was not valid UTF-8.") from None
    data = b""
    try:
        return _decode_jsonrpc(text)
    finally:
        text = ""


class _HttpCallBudget:
    """Monotonic idle and total-deadline accounting for one HTTP exchange."""

    __slots__ = (
        "_abandoned_tasks",
        "_deadline",
        "_idle_timeout_s",
        "_last_activity",
        "_loop",
        "max_response_bytes",
    )

    def __init__(self, limits: McpTransportLimits) -> None:
        self._loop = asyncio.get_running_loop()
        self._abandoned_tasks: set[asyncio.Future[Any]] = set()
        started_at = self._loop.time()
        self._last_activity = started_at
        self._idle_timeout_s = limits.idle_timeout_s
        self._deadline = started_at + limits.total_call_timeout_s
        self.max_response_bytes = limits.max_response_bytes

    def note_activity(self) -> None:
        self._last_activity = self._loop.time()

    def total_deadline_expired(self) -> bool:
        return self._loop.time() >= self._deadline

    def check_total_deadline(self) -> None:
        if self.total_deadline_expired():
            raise McpCallDeadlineExceededError(
                "MCP HTTP exchange exceeded its total call deadline."
            )

    def check_wait_deadlines(self) -> None:
        self.check_total_deadline()
        if self._loop.time() >= self._last_activity + self._idle_timeout_s:
            raise McpIdleTimeoutError("MCP HTTP exchange exceeded its idle timeout.")

    def take_abandoned_tasks(self) -> tuple[asyncio.Future[Any], ...]:
        tasks = tuple(self._abandoned_tasks)
        self._abandoned_tasks.clear()
        return tasks

    def _cancel_and_retain(self, task: asyncio.Future[Any]) -> None:
        task.cancel()
        self._abandoned_tasks.add(task)

    async def wait(
        self,
        awaitable: Awaitable[_T],
        *,
        on_started: Callable[[], None] | None = None,
    ) -> _T:
        now = self._loop.time()
        if now >= self._deadline:
            self._close_unstarted_awaitable(awaitable)
            raise McpCallDeadlineExceededError(
                "MCP HTTP exchange exceeded its total call deadline."
            )
        # Idle time measures one owned transport/cleanup await, not synchronous
        # request preparation or processing of bytes already received.
        self._last_activity = now
        idle_deadline = self._last_activity + self._idle_timeout_s
        task = asyncio.ensure_future(_capture_mcp_owned_task_fatal_signal(awaitable))
        if on_started is not None:
            on_started()
        try:
            done, _ = await asyncio.wait(
                {task},
                timeout=min(self._deadline, idle_deadline) - now,
            )
        except asyncio.CancelledError:
            self._cancel_and_retain(task)
            raise
        if done:
            if task.cancelled():
                try:
                    self.check_wait_deadlines()
                except (McpCallDeadlineExceededError, McpIdleTimeoutError):
                    self._abandoned_tasks.add(task)
                    raise
                raise _McpUnexpectedTransportCancellationError(
                    "MCP HTTP transport operation was cancelled unexpectedly."
                )
            try:
                self.check_wait_deadlines()
            except (McpCallDeadlineExceededError, McpIdleTimeoutError):
                # The elapsed-time decision is authoritative, but a completed
                # child can still own a late response or diagnostic. Retain that
                # exact future for the exchange settlement instead of discarding
                # its result or emitting an unobserved task exception.
                self._abandoned_tasks.add(task)
                raise
            return _unwrap_mcp_owned_task_result(task.result())
        self._cancel_and_retain(task)
        now = self._loop.time()
        if now >= self._deadline:
            raise McpCallDeadlineExceededError(
                "MCP HTTP exchange exceeded its total call deadline."
            )
        raise McpIdleTimeoutError("MCP HTTP exchange exceeded its idle timeout.")

    async def wait_for_owned_task(self, task: asyncio.Future[_T]) -> _T:
        """Wait within this budget without cancelling or wrapping an existing owner."""

        now = self._loop.time()
        if now >= self._deadline:
            raise McpCallDeadlineExceededError(
                "MCP HTTP exchange exceeded its total call deadline."
            )
        self._last_activity = now
        idle_deadline = self._last_activity + self._idle_timeout_s
        done, _ = await asyncio.wait(
            {task},
            timeout=min(self._deadline, idle_deadline) - now,
        )
        if done:
            if task.cancelled():
                self.check_wait_deadlines()
                raise _McpUnexpectedTransportCancellationError(
                    "MCP HTTP transport operation was cancelled unexpectedly."
                )
            self.check_wait_deadlines()
            return _unwrap_mcp_owned_task_result(task.result())
        now = self._loop.time()
        if now >= self._deadline:
            raise McpCallDeadlineExceededError(
                "MCP HTTP exchange exceeded its total call deadline."
            )
        raise McpIdleTimeoutError("MCP HTTP exchange exceeded its idle timeout.")

    def _close_unstarted_awaitable(self, awaitable: Awaitable[Any]) -> None:
        if isinstance(awaitable, asyncio.Future):
            self._cancel_and_retain(awaitable)
            return
        close = getattr(awaitable, "close", None)
        if callable(close):
            close()


def _validate_http_response_headers(
    response: httpx.Response,
    *,
    limits: McpTransportLimits,
) -> None:
    content_encoding = ""
    encodings: list[str] = []
    content_length: str | None = None
    stripped = ""
    try:
        content_encoding = response.headers.get("content-encoding", "")
        encodings = [item.strip().lower() for item in content_encoding.split(",") if item.strip()]
        if any(encoding != "identity" for encoding in encodings):
            raise McpProtocolError("MCP HTTP response used an unsupported content encoding.")
        content_length = response.headers.get("content-length")
        if content_length is None:
            return
        stripped = content_length.strip()
        if not stripped.isascii() or not stripped.isdecimal():
            raise McpProtocolError("MCP HTTP response had an invalid Content-Length header.")
        if _decimal_exceeds_limit(stripped, limits.max_response_bytes):
            raise McpResponseTooLargeError(
                f"MCP HTTP response exceeded {limits.max_response_bytes} bytes."
            )
    finally:
        content_encoding = ""
        encodings.clear()
        content_length = None
        stripped = ""


def _decimal_exceeds_limit(value: str, limit: int) -> bool:
    normalized = value.lstrip("0") or "0"
    limit_text = str(limit)
    return len(normalized) > len(limit_text) or (
        len(normalized) == len(limit_text) and normalized > limit_text
    )


async def _iter_bounded_http_body(
    response: httpx.Response,
    *,
    budget: _HttpCallBudget,
    max_response_bytes: int,
) -> AsyncGenerator[bytes, None]:
    total_bytes = 0
    if response.is_stream_consumed:
        content = response.content
        for offset in range(0, len(content), 65_536):
            await asyncio.sleep(0)
            budget.check_total_deadline()
            chunk = content[offset : offset + 65_536]
            if not chunk:
                continue
            budget.note_activity()
            total_bytes += len(chunk)
            if total_bytes > max_response_bytes:
                content = b""
                chunk = b""
                raise McpResponseTooLargeError(
                    f"MCP HTTP response exceeded {max_response_bytes} bytes."
                )
            try:
                yield chunk
            finally:
                chunk = b""
        content = b""
        budget.check_total_deadline()
        return
    # Preserve transport chunk boundaries so every received chunk resets the
    # application-owned idle timer. A fixed rechunking size would buffer slow,
    # active peers inside httpx and falsely classify them as idle.
    iterator = response.aiter_raw().__aiter__()
    while True:
        try:
            chunk = await budget.wait(_next_http_body_chunk(iterator))
        except StopAsyncIteration:
            budget.check_wait_deadlines()
            return
        if not chunk:
            continue
        budget.note_activity()
        total_bytes += len(chunk)
        if total_bytes > max_response_bytes:
            chunk = b""
            raise McpResponseTooLargeError(
                f"MCP HTTP response exceeded {max_response_bytes} bytes."
            )
        try:
            yield chunk
        finally:
            chunk = b""


async def _next_http_body_chunk(iterator: AsyncIterator[bytes]) -> bytes:
    try:
        return await iterator.__anext__()
    except StopAsyncIteration:
        raise
    except McpProtocolError:
        # An extension can raise the same public type as Cayu's bounded parser,
        # but it cannot prove that the transport stopped cleanly. Keep that
        # provenance structurally distinct so _send fences the logical session.
        raise _McpExtensionTransportProtocolError(
            "MCP HTTP response stream raised a protocol failure."
        ) from None


async def _read_http_body(
    response: httpx.Response,
    *,
    budget: _HttpCallBudget,
    retain: bool,
    max_message_bytes: int | None = None,
) -> bytes:
    content_length = response.headers.get("content-length")
    message_too_large = (
        max_message_bytes is not None
        and content_length is not None
        and _decimal_exceeds_limit(content_length.strip(), max_message_bytes)
    )
    content_length = None
    if message_too_large:
        raise McpMessageTooLargeError(
            f"MCP HTTP JSON-RPC error message exceeded {max_message_bytes} bytes."
        )
    body = bytearray()
    message_bytes = 0
    chunks = _iter_bounded_http_body(
        response,
        budget=budget,
        max_response_bytes=budget.max_response_bytes,
    )
    try:
        async with aclosing(chunks):
            async for chunk in chunks:
                message_bytes += len(chunk)
                if max_message_bytes is not None and message_bytes > max_message_bytes:
                    body.clear()
                    chunk = b""
                    raise McpMessageTooLargeError(
                        f"MCP HTTP JSON-RPC error message exceeded {max_message_bytes} bytes."
                    )
                if retain:
                    body.extend(chunk)
                chunk = b""
    except BaseException:
        body.clear()
        raise
    try:
        return bytes(body)
    finally:
        body.clear()
        message_bytes = 0


async def _read_http_error_body(
    response: httpx.Response,
    *,
    budget: _HttpCallBudget,
    limits: McpTransportLimits,
    json_message: bool,
    sse_events: bool,
    retain: bool,
) -> tuple[bytes, bool]:
    """Read an HTTP error while applying the advertised representation's limit."""

    if not sse_events:
        return (
            await _read_http_body(
                response,
                budget=budget,
                retain=retain,
                max_message_bytes=limits.max_message_bytes if json_message else None,
            ),
            True,
        )

    retained = _RetainedHttpErrorBody(bytearray())
    raw_chunks = _iter_bounded_http_body(
        response,
        budget=budget,
        max_response_bytes=limits.max_response_bytes,
    )
    chunks = _retain_http_body_chunks(
        raw_chunks,
        retained=retained,
        retain=retain,
        max_retain_bytes=limits.max_message_bytes,
    )
    lines = _aiter_bounded_sse_lines(
        chunks,
        max_event_bytes=limits.max_message_bytes,
    )
    try:
        async with aclosing(raw_chunks), aclosing(chunks), aclosing(lines):
            async for _line in lines:
                del _line
    except BaseException:
        retained.content.clear()
        retained.source_complete = False
        raise
    try:
        return bytes(retained.content), retained.source_complete
    finally:
        retained.content.clear()
        retained.source_complete = False


async def _retain_http_body_chunks(
    chunks: AsyncIterator[bytes],
    *,
    retained: _RetainedHttpErrorBody,
    retain: bool,
    max_retain_bytes: int,
) -> AsyncGenerator[bytes, None]:
    async for chunk in chunks:
        if retain:
            remaining = max_retain_bytes - len(retained.content)
            if len(chunk) > max(0, remaining):
                retained.source_complete = False
            if remaining > 0:
                retained.content.extend(chunk[:remaining])
        try:
            yield chunk
        finally:
            chunk = b""


async def _consume_http_notification_response(
    response: httpx.Response,
    *,
    budget: _HttpCallBudget,
    limits: McpTransportLimits,
) -> None:
    """Drain a notification response under both aggregate and message bounds."""

    content_type = response.headers.get("content-type", "")
    media_type = content_type.split(";", 1)[0].strip().lower()
    content_type = ""
    content_length: str | None = None
    message_bytes = 0
    try:
        if media_type == _SSE_CONTENT_TYPE:
            chunks = _iter_bounded_http_body(
                response,
                budget=budget,
                max_response_bytes=limits.max_response_bytes,
            )
            lines = _aiter_bounded_sse_lines(
                chunks,
                max_event_bytes=limits.max_message_bytes,
            )
            async with aclosing(chunks), aclosing(lines):
                async for line in lines:
                    del line
            return

        content_length = response.headers.get("content-length")
        if content_length is not None and _decimal_exceeds_limit(
            content_length.strip(),
            limits.max_message_bytes,
        ):
            raise McpMessageTooLargeError(
                f"MCP HTTP notification response exceeded {limits.max_message_bytes} bytes."
            )
        chunks = _iter_bounded_http_body(
            response,
            budget=budget,
            max_response_bytes=limits.max_response_bytes,
        )
        async with aclosing(chunks):
            async for chunk in chunks:
                message_bytes += len(chunk)
                if message_bytes > limits.max_message_bytes:
                    chunk = b""
                    raise McpMessageTooLargeError(
                        f"MCP HTTP notification response exceeded {limits.max_message_bytes} bytes."
                    )
                chunk = b""
    finally:
        media_type = ""
        content_length = None
        message_bytes = 0


async def _read_json_response(
    response: httpx.Response,
    *,
    budget: _HttpCallBudget,
    limits: McpTransportLimits,
) -> dict[str, Any]:
    content_length = response.headers.get("content-length")
    message_too_large = False
    if content_length is not None:
        message_too_large = _decimal_exceeds_limit(
            content_length.strip(),
            limits.max_message_bytes,
        )
    content_length = None
    if message_too_large:
        raise McpMessageTooLargeError(
            f"MCP HTTP JSON-RPC message exceeded {limits.max_message_bytes} bytes."
        )
    body = bytearray()
    chunks = _iter_bounded_http_body(
        response,
        budget=budget,
        max_response_bytes=limits.max_response_bytes,
    )
    try:
        async with aclosing(chunks):
            async for chunk in chunks:
                if len(body) + len(chunk) > limits.max_message_bytes:
                    body.clear()
                    chunk = b""
                    raise McpMessageTooLargeError(
                        f"MCP HTTP JSON-RPC message exceeded {limits.max_message_bytes} bytes."
                    )
                body.extend(chunk)
                chunk = b""
    except BaseException:
        body.clear()
        raise
    data = bytes(body)
    body.clear()
    if not data:
        raise McpPeerClosedError("MCP HTTP peer closed before sending a JSON-RPC response.")
    try:
        return _decode_jsonrpc_bytes(data)
    finally:
        data = b""


async def _read_sse_response(
    response: httpx.Response,
    request_id: int,
    *,
    initialize_request: bool,
    redactor: SecretRedactor,
    limits: McpTransportLimits,
    budget: _HttpCallBudget,
) -> dict[str, Any]:
    """Read the SSE stream incrementally and return the response matching the request.

    Events are dispatched on blank lines (per the SSE spec). The matching JSON-RPC
    response is returned the moment it arrives, so the call does not wait for the
    server to close the stream. A server-initiated request (which a request/response
    client cannot service) fails the session loudly instead of being silently dropped.
    """
    chunks = _iter_bounded_http_body(
        response,
        budget=budget,
        max_response_bytes=limits.max_response_bytes,
    )
    data_lines: list[str] = []
    lines = _aiter_bounded_sse_lines(
        chunks,
        max_event_bytes=limits.max_message_bytes,
    )
    try:
        async with aclosing(chunks), aclosing(lines):
            async for line in lines:
                try:
                    if line == "":
                        message = _sse_event_message(data_lines)
                        data_lines = []
                        if message is None:
                            continue
                        if message.get("id") == request_id:
                            return message
                        if initialize_request and "method" not in message:
                            cleanup_protocol_version = _initialize_cleanup_protocol_version(message)
                            message.clear()
                            message = {}
                            raise _McpHttpResponseContentError(
                                "MCP HTTP response id did not match the request.",
                                cleanup_protocol_version=cleanup_protocol_version,
                            )
                        _reject_server_message(message, redactor=redactor)
                        message.clear()
                        message = {}
                        continue
                    if line.startswith("data:"):
                        data = line[len("data:") :]
                        if data.startswith(" "):
                            data = data[1:]
                        data_lines.append(data)
                        data = ""
                    # Other SSE fields carry no JSON-RPC payload.
                finally:
                    line = ""
    except BaseException:
        data_lines.clear()
        line = ""
        raise
    # SSE dispatches only on a blank line. A peer EOF is not an implicit event
    # delimiter, so an unfinished event cannot be accepted as a response.
    data_lines.clear()
    raise McpPeerClosedError("MCP HTTP SSE peer closed; stream did not contain the response.")


async def _aiter_bounded_sse_lines(
    chunks: AsyncIterator[bytes],
    *,
    max_event_bytes: int,
) -> AsyncGenerator[str, None]:
    """Yield strict UTF-8 SSE lines while bounding the unfinished event."""

    line = bytearray()
    event_bytes = 0
    # None means no pending CR. A bool records whether the line already emitted
    # at that CR was nonblank, so a following LF can be counted as the second
    # byte of CRLF without delaying valid standalone-CR dispatch.
    pending_cr_nonblank: bool | None = None

    def account_completed_line(delimiter_bytes: int, *, commit: bool = True) -> None:
        nonlocal event_bytes
        if not line:
            # The blank dispatch line is not part of the event-size contract.
            event_bytes = 0
            return
        if event_bytes + len(line) + delimiter_bytes > max_event_bytes:
            line.clear()
            raise McpMessageTooLargeError(f"MCP HTTP SSE event exceeded {max_event_bytes} bytes.")
        if commit:
            event_bytes += len(line) + delimiter_bytes

    try:
        async for chunk in chunks:
            chunk_view = memoryview(chunk)
            try:
                offset = 0
                while offset < len(chunk):
                    if pending_cr_nonblank is not None:
                        if chunk[offset] == 0x0A:
                            if pending_cr_nonblank:
                                if event_bytes + 1 > max_event_bytes:
                                    line.clear()
                                    raise McpMessageTooLargeError(
                                        f"MCP HTTP SSE event exceeded {max_event_bytes} bytes."
                                    )
                                event_bytes += 1
                            offset += 1
                            pending_cr_nonblank = None
                            continue
                        pending_cr_nonblank = None
                    delimiter_match = _SSE_LINE_ENDING.search(chunk, offset)
                    delimiter = len(chunk) if delimiter_match is None else delimiter_match.start()
                    segment_bytes = delimiter - offset
                    if event_bytes + len(line) + segment_bytes > max_event_bytes:
                        line.clear()
                        delimiter_match = None
                        raise McpMessageTooLargeError(
                            f"MCP HTTP SSE event exceeded {max_event_bytes} bytes."
                        )
                    line.extend(chunk_view[offset:delimiter])
                    if delimiter == len(chunk):
                        break
                    if chunk[delimiter] == 0x0D:
                        # CR is a complete SSE delimiter. Emit immediately so an
                        # active stream cannot withhold a complete event merely by
                        # stalling before an optional LF. If LF arrives later, the
                        # pending state above accounts and swallows it as CRLF.
                        nonblank = bool(line)
                        account_completed_line(1)
                        decoded = _decode_sse_line(line)
                        line.clear()
                        pending_cr_nonblank = nonblank
                        offset = delimiter + 1
                        try:
                            yield decoded
                        finally:
                            decoded = ""
                        continue
                    account_completed_line(1)
                    decoded = _decode_sse_line(line)
                    line.clear()
                    offset = delimiter + 1
                    try:
                        yield decoded
                    finally:
                        decoded = ""
            finally:
                chunk_view.release()
                chunk = b""
        if line:
            account_completed_line(0)
            decoded = _decode_sse_line(line)
            line.clear()
            try:
                yield decoded
            finally:
                decoded = ""
    finally:
        line.clear()


def _decode_sse_line(line: bytearray) -> str:
    try:
        return line.decode("utf-8", "strict")
    except UnicodeDecodeError:
        line.clear()
        raise _McpHttpResponseContentError("MCP HTTP SSE response was not valid UTF-8.") from None


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


def _attach_http_cleanup_failure(
    primary_error: BaseException,
    cleanup_error: BaseException,
) -> None:
    prior_cause = exception_cause(primary_error)
    if prior_cause is cleanup_error:
        return
    combined = (
        cleanup_error
        if prior_cause is None
        else BaseExceptionGroup(
            "MCP HTTP cleanup failures.",
            [prior_cause, cleanup_error],
        )
    )
    set_exception_cause(primary_error, combined)


def _attach_http_settlement_task(
    error: BaseException,
    task: asyncio.Task[None],
) -> None:
    attach_mcp_http_settlement_task(error, task)


def _http_settlement_task(error: BaseException) -> asyncio.Task[None] | None:
    return mcp_http_settlement_task(error)


async def _exit_http_stream_context(
    stream_context: AbstractAsyncContextManager[httpx.Response],
) -> None:
    await stream_context.__aexit__(None, None, None)


async def _close_http_response_stream(stream: httpx.AsyncByteStream) -> None:
    await stream.aclose()


async def _settle_registered_http_exchange(
    *,
    exchange_owner: _HttpExchangeOwner,
    **kwargs: Any,
) -> None:
    """Settle one registered owner and always release its session-wide barrier."""

    try:
        await _settle_http_exchange(exchange_owner=exchange_owner, **kwargs)
    finally:
        exchange_owner.mark_settled()
        kwargs.clear()
        del exchange_owner


async def _settle_http_exchange(
    *,
    exchange_owner: _HttpExchangeOwner,
    abandoned_tasks: Sequence[asyncio.Future[Any]],
    redactor: SecretRedactor,
    already_reported_failure_ids: frozenset[int] = frozenset(),
    initialization_cleanup: Callable[[], Coroutine[Any, Any, None]] | None = None,
    client_close_task: asyncio.Task[None] | None = None,
    client_close_factory: Callable[[], asyncio.Task[None]] | None = None,
    defer_client_close_to_parent: bool = False,
    report_client_close_failure: bool = True,
) -> None:
    has_one_client_close_owner = (client_close_task is None) != (client_close_factory is None)
    if defer_client_close_to_parent:
        if has_one_client_close_owner:
            raise RuntimeError("Deferred MCP HTTP settlement cannot own client close.")
    elif not has_one_client_close_owner:
        raise RuntimeError("MCP HTTP settlement requires exactly one client close owner.")
    failures: list[BaseException] = []
    seen_tasks: set[int] = set()
    seen_failures = set(already_reported_failure_ids)
    entered_at_start = exchange_owner.entered
    exit_prestarted = exchange_owner.exit_task is not None
    abandoned_response_close_uncertain = False

    def retain_safe_failure(error: BaseException, *, context: str) -> None:
        if id(error) in seen_failures:
            return
        seen_failures.add(id(error))
        failures.append(
            credential_safe_mcp_transport_failure(
                error,
                redactor=redactor,
                context=context,
            )
        )

    # An enter or body-read task that suppressed cancellation remains the exact
    # owner until it settles. A late successful enter must still be paired with
    # this exchange's context exit.
    for task in abandoned_tasks:
        if id(task) in seen_tasks:
            continue
        seen_tasks.add(id(task))
        cancellation_boundary = _McpCallerCancellationBoundary()
        try:
            await cancellation_boundary.checkpoint()
            result = _unwrap_mcp_owned_task_result(await asyncio.shield(task))
        except StopAsyncIteration:
            continue
        except asyncio.CancelledError:
            if cancellation_boundary.caller_cancelled():
                raise
            if entered_at_start and not exit_prestarted:
                abandoned_response_close_uncertain = True
        except BaseException as error:
            if entered_at_start and not exit_prestarted:
                abandoned_response_close_uncertain = True
            retain_safe_failure(
                error,
                context="MCP HTTP abandoned transport operation failed",
            )
        else:
            if not exchange_owner.entered:
                if isinstance(result, httpx.Response):
                    exchange_owner.record_entered_response(result)
                else:
                    failures.append(
                        McpProtocolError("MCP HTTP stream enter returned an invalid response.")
                    )
            result = None
    if abandoned_tasks:
        del task
    abandoned_tasks = ()

    exit_failed = False
    response_was_closed_before_exit = False
    if exchange_owner.entered:
        response = exchange_owner.response
        assert response is not None
        response_was_closed_before_exit = response.is_closed
        exit_task = exchange_owner.start_exit()
        cancellation_boundary = _McpCallerCancellationBoundary()
        try:
            await cancellation_boundary.checkpoint()
            _unwrap_mcp_owned_task_result(await asyncio.shield(exit_task))
        except asyncio.CancelledError:
            if cancellation_boundary.caller_cancelled():
                raise
            exit_failed = True
            failures.append(McpProtocolError("MCP HTTP stream context exit was cancelled."))
        except BaseException as error:
            exit_failed = True
            retain_safe_failure(
                error,
                context="MCP HTTP stream context exit failed",
            )

        # httpx marks a response closed before its stream close await finishes.
        # If an abandoned body owner ended uncertainly, a successful context exit
        # can therefore be a no-op. Retry the stream directly only in that proven
        # uncertain case, or after an exit failure.
        retry_stream_close = exit_failed or (
            not exit_prestarted
            and abandoned_response_close_uncertain
            and response_was_closed_before_exit
        )
        if retry_stream_close:
            direct_close_task = _start_http_response_close(response)
            cancellation_boundary = _McpCallerCancellationBoundary()
            try:
                await cancellation_boundary.checkpoint()
                _unwrap_mcp_owned_task_result(await asyncio.shield(direct_close_task))
            except asyncio.CancelledError:
                if cancellation_boundary.caller_cancelled():
                    raise
                failures.append(McpProtocolError("MCP HTTP stream close retry was cancelled."))
            except BaseException as error:
                retain_safe_failure(
                    error,
                    context="MCP HTTP stream close retry failed",
                )
            del direct_close_task
        del exit_task
        del response

    if initialization_cleanup is not None:
        cleanup_task = asyncio.create_task(
            _capture_mcp_owned_task_fatal_signal(initialization_cleanup())
        )
        initialization_cleanup = None
        cancellation_boundary = _McpCallerCancellationBoundary()
        try:
            await cancellation_boundary.checkpoint()
            _unwrap_mcp_owned_task_result(await asyncio.shield(cleanup_task))
        except asyncio.CancelledError:
            if cancellation_boundary.caller_cancelled():
                raise
            failures.append(
                McpProtocolError("MCP HTTP initialization session cleanup was cancelled.")
            )
        except BaseException as error:
            retain_safe_failure(
                error,
                context="MCP HTTP initialization session cleanup failed",
            )
        del cleanup_task

    # This exchange no longer owns remote I/O. Release the session-wide barrier
    # before joining the shared client close task so that task cannot wait on us.
    exchange_owner.mark_settled()
    if not defer_client_close_to_parent:
        if client_close_task is None:
            assert client_close_factory is not None
            client_close_task = client_close_factory()
        client_close_factory = None
        cancellation_boundary = _McpCallerCancellationBoundary()
        try:
            await cancellation_boundary.checkpoint()
            await asyncio.shield(client_close_task)
        except asyncio.CancelledError:
            if cancellation_boundary.caller_cancelled():
                raise
            if report_client_close_failure:
                failures.append(McpProtocolError("MCP HTTP client cleanup was cancelled."))
        except BaseException as error:
            if report_client_close_failure:
                retain_safe_failure(
                    error,
                    context="MCP HTTP client cleanup failed",
                )
        del client_close_task
    client_close_factory = None
    del exchange_owner

    if len(failures) == 1:
        raise failures[0]
    if failures:
        raise BaseExceptionGroup("MCP HTTP settlement failures.", failures)


def _start_http_response_close(response: httpx.Response) -> asyncio.Task[Any]:
    stream = cast("httpx.AsyncByteStream", response.stream)
    return asyncio.create_task(
        _capture_mcp_owned_task_fatal_signal(_close_http_response_stream(stream))
    )


async def _wait_for_http_close_task(
    close_task: asyncio.Future[Any],
    *,
    budget: _HttpCallBudget,
) -> bool:
    try:
        await budget.wait_for_owned_task(close_task)
    except asyncio.CancelledError:
        raise
    except BaseException:
        return False
    finally:
        for abandoned_task in budget.take_abandoned_tasks():
            _consume_task_result(abandoned_task)
    return True


async def _wait_for_http_client_close_task(
    close_task: asyncio.Future[Any],
    *,
    budget: _HttpCallBudget,
) -> BaseException | None:
    try:
        await budget.wait_for_owned_task(close_task)
    except (McpCallDeadlineExceededError, McpIdleTimeoutError):
        return None
    except asyncio.CancelledError:
        raise
    except BaseException as error:
        return error
    finally:
        for abandoned_task in budget.take_abandoned_tasks():
            _consume_task_result(abandoned_task)
    return None


def _consume_task_result(task: asyncio.Future[Any]) -> None:
    # This callback is the terminal observer for intentionally retained work.
    # Consume process-control leaves and mixed BaseExceptionGroups as well as
    # ordinary failures so the event loop cannot publish a second diagnostic.
    with suppress(BaseException):
        task.result()


def _safe_body(
    body: bytes,
    *,
    redactor: SecretRedactor,
    source_complete: bool,
) -> str:
    if not source_complete:
        return _TRUNCATED_ERROR_BODY_DETAIL
    try:
        text = body.decode("utf-8", "replace")
        body = b""
        try:
            return redactor.redact_text(text)[:_MAX_ERROR_BODY_CHARS]
        finally:
            text = ""
    except Exception:
        return "<unreadable response body>"
