from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from cayu._validation import (
    copy_json_value,
    json_utf8_size_within_limit,
    require_clean_nonblank,
)
from cayu.mcp._jsonrpc import (
    DEFAULT_MCP_CLIENT_NAME,
    DEFAULT_MCP_CLIENT_VERSION,
    DEFAULT_MCP_MAX_LIST_ITEMS,
    DEFAULT_MCP_MAX_LIST_PAGES,
    DEFAULT_MCP_REQUEST_TIMEOUT_S,
    JSONRPC_METHOD_NOT_FOUND,
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
from cayu.mcp._stdio_process import (
    DEFAULT_MCP_CONTAINMENT_KILL_TIMEOUT_S,
    DEFAULT_MCP_CONTAINMENT_STARTUP_TIMEOUT_S,
    DEFAULT_MCP_CONTAINMENT_TERM_TIMEOUT_S,
    ContainedStdioMcpProcess,
    StdioMcpProcessLifetime,
    _prepare_stdio_mcp_containment_rendezvous,
    create_contained_stdio_mcp_process,
    create_direct_stdio_mcp_process,
    preflight_stdio_mcp_parent_death_containment,
    stdio_mcp_process_capability_evidence,
    stdio_mcp_process_capability_evidence_for_process,
    validate_containment_platform,
    validate_stdio_mcp_containment_timeout,
    validate_stdio_mcp_process_lifetime,
)
from cayu.mcp._transport import (
    McpCallDeadlineExceededError,
    McpIdleTimeoutError,
    McpMessageTooLargeError,
    McpPeerClosedError,
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
    _attach_mcp_session_cleanup_task,
    _await_mcp_session_cleanup_task,
    _close_mcp_session_after_primary_failure,
    _credential_safe_mcp_cancellation,
    _mcp_session_close_task,
    _McpCallerCancellationBoundary,
    _McpToolDiscovery,
    _McpToolDispatchSignal,
    _retain_mcp_session_close,
    copy_mcp_server_spec,
)
from cayu.vaults import (
    SecretRedactionTail,
    SecretRedactor,
    SecretResolver,
    resolve_secret_env,
    validate_secret_resolver,
)

DEFAULT_MCP_WRITE_TIMEOUT_S = 5.0
DEFAULT_MCP_GRACEFUL_SHUTDOWN_TIMEOUT_S = 2.0
DEFAULT_MCP_CANCELLATION_NOTIFICATION_TIMEOUT_S = 1.0

# When we do not inherit the full parent environment, npx/uvx and other stdio
# launchers still need a handful of variables (a PATH to find the binary, a HOME
# for their package caches, and a locale) or they fail to start at all. Copy only
# this minimal safelist through so the child stays isolated from the rest of the
# parent env while remaining launchable.
_MINIMAL_ENV_SAFELIST = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "SYSTEMROOT",
    "APPDATA",
    "USERPROFILE",
)

# Retain at most this many bytes of the child's most recent stderr output so a
# startup crash surfaces in the resulting protocol error instead of being lost.
DEFAULT_MCP_STDERR_CAPTURE_BYTES = 8192
# Best-effort grace period to let the stderr drain finish (and reach EOF) after
# the child closes stdout, so a crash message lands in the captured tail.
DEFAULT_MCP_STDERR_DRAIN_GRACE_S = 0.2
_RETAINED_STDIO_WRITE_SETTLEMENT_TASKS: set[asyncio.Task[None]] = set()
_RETAINED_STDIO_SHUTDOWN_TASKS: set[asyncio.Task[Any]] = set()


class _StdioPreDispatchMessageTooLargeError(McpMessageTooLargeError):
    """Internal proof that stdio size validation rejected before dispatch."""


@dataclass(frozen=True, slots=True)
class _StdioPendingTiming:
    settled_at: float
    last_read_activity: float
    expired_idle_gap: tuple[float, float] | None
    response_received: bool


@dataclass(frozen=True, slots=True)
class _StdioClientConnectionConfig:
    """One defensively owned stdio connection configuration."""

    transport_limits: McpTransportLimits
    request_timeout_s: float | None
    write_timeout_s: float
    graceful_shutdown_timeout_s: float
    cancellation_notification_timeout_s: float
    client_name: str
    client_version: str
    max_list_pages: int
    max_list_items: int
    inherit_env: bool
    secret_resolver: SecretResolver | None
    process_lifetime: StdioMcpProcessLifetime
    containment_startup_timeout_s: float
    containment_term_timeout_s: float
    containment_kill_timeout_s: float


def _validate_stdio_client_timeout(value: float, field_name: str) -> float:
    """Own one finite positive timeout before any connection side effect."""

    timeout_s = validate_positive_number(value, field_name)
    if not math.isfinite(timeout_s):
        raise ValueError(f"{field_name} must be finite and greater than zero.")
    return timeout_s


def _clear_completed_stdio_response(
    future: asyncio.Future[dict[str, Any]],
) -> None:
    """Drop a private response that lost a timeout or cancellation race."""

    if not future.done() or future.cancelled():
        return
    try:
        response = future.result()
    except BaseException:
        # Reading the exception prevents asyncio from publishing a second,
        # uncontrolled diagnostic. Pending failures are already credential-safe.
        return
    response.clear()


def _base_child_env(inherit_env: bool) -> dict[str, str]:
    """Build the base child environment before per-server overrides.

    Inherits the full parent env when requested; otherwise copies only the
    minimal safelist so launchers such as npx/uvx can still be found and run.
    """
    if inherit_env:
        return dict(os.environ)
    env: dict[str, str] = {}
    for name in _MINIMAL_ENV_SAFELIST:
        value = os.environ.get(name)
        if value is not None:
            env[name] = value
    return env


def _stdio_containment_rendezvous_identity(server: McpServerSpec) -> str:
    """Bind replacement admission to one secret-free logical command identity."""

    assert server.command is not None
    payload = {
        "command": server.command,
        "connection_identity": (
            {"connection_id": server.connection_id, "kind": "connection_id"}
            if server.connection_id is not None
            else {"connection_id": server.name, "kind": "server_name"}
        ),
        "schema": "cayu.mcp.stdio_containment_rendezvous.v1",
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class StdioMcpClient(McpClient):
    """MCP client for local stdio servers."""

    def __init__(
        self,
        *,
        request_timeout_s: float | None = None,
        transport_limits: McpTransportLimits | None = None,
        write_timeout_s: float = DEFAULT_MCP_WRITE_TIMEOUT_S,
        graceful_shutdown_timeout_s: float = DEFAULT_MCP_GRACEFUL_SHUTDOWN_TIMEOUT_S,
        cancellation_notification_timeout_s: float = DEFAULT_MCP_CANCELLATION_NOTIFICATION_TIMEOUT_S,
        client_name: str = DEFAULT_MCP_CLIENT_NAME,
        client_version: str = DEFAULT_MCP_CLIENT_VERSION,
        max_list_pages: int = DEFAULT_MCP_MAX_LIST_PAGES,
        max_list_items: int = DEFAULT_MCP_MAX_LIST_ITEMS,
        inherit_env: bool = False,
        secret_resolver: SecretResolver | None = None,
        process_lifetime: StdioMcpProcessLifetime | str = (
            StdioMcpProcessLifetime.PARENT_DEATH_CONTAINMENT
        ),
        containment_startup_timeout_s: float = DEFAULT_MCP_CONTAINMENT_STARTUP_TIMEOUT_S,
        containment_term_timeout_s: float = DEFAULT_MCP_CONTAINMENT_TERM_TIMEOUT_S,
        containment_kill_timeout_s: float = DEFAULT_MCP_CONTAINMENT_KILL_TIMEOUT_S,
    ) -> None:
        self.transport_limits = resolve_mcp_transport_limits(
            transport_limits,
            legacy_timeout_s=request_timeout_s,
            default_timeout_s=DEFAULT_MCP_REQUEST_TIMEOUT_S,
            legacy_field_name="request_timeout_s",
        )
        self._has_explicit_transport_limits = transport_limits is not None
        # Retain the legacy observable attribute for callers that inspect it.
        self.request_timeout_s = self.transport_limits.total_call_timeout_s
        self.write_timeout_s = validate_positive_number(
            write_timeout_s,
            "write_timeout_s",
        )
        self.graceful_shutdown_timeout_s = validate_positive_number(
            graceful_shutdown_timeout_s,
            "graceful_shutdown_timeout_s",
        )
        self.cancellation_notification_timeout_s = validate_positive_number(
            cancellation_notification_timeout_s,
            "cancellation_notification_timeout_s",
        )
        self.client_name = require_clean_nonblank(client_name, "client_name")
        self.client_version = require_clean_nonblank(client_version, "client_version")
        self.max_list_pages = validate_positive_integer(max_list_pages, "max_list_pages")
        self.max_list_items = validate_positive_integer(max_list_items, "max_list_items")
        if type(inherit_env) is not bool:
            raise TypeError("inherit_env must be a bool.")
        self.inherit_env = inherit_env
        if secret_resolver is not None:
            validate_secret_resolver(secret_resolver)
        self.secret_resolver = secret_resolver
        self.process_lifetime = validate_stdio_mcp_process_lifetime(process_lifetime)
        self.containment_startup_timeout_s = validate_stdio_mcp_containment_timeout(
            containment_startup_timeout_s,
            "containment_startup_timeout_s",
        )
        self.containment_term_timeout_s = validate_stdio_mcp_containment_timeout(
            containment_term_timeout_s,
            "containment_term_timeout_s",
        )
        self.containment_kill_timeout_s = validate_stdio_mcp_containment_timeout(
            containment_kill_timeout_s,
            "containment_kill_timeout_s",
        )

    @property
    def process_capability_evidence(self):
        lifetime = validate_stdio_mcp_process_lifetime(self.process_lifetime)
        return stdio_mcp_process_capability_evidence(lifetime).model_copy(deep=True)

    def _snapshot_connection_config(self) -> _StdioClientConnectionConfig:
        """Revalidate every mutable client field before the first suspension."""

        transport_limits = copy_mcp_transport_limits(self.transport_limits)
        if type(self._has_explicit_transport_limits) is not bool:
            raise TypeError("_has_explicit_transport_limits must be a bool.")
        request_timeout_s = (
            None
            if self._has_explicit_transport_limits
            else _validate_stdio_client_timeout(self.request_timeout_s, "request_timeout_s")
        )
        inherit_env = self.inherit_env
        if type(inherit_env) is not bool:
            raise TypeError("inherit_env must be a bool.")
        secret_resolver = self.secret_resolver
        if secret_resolver is not None:
            validate_secret_resolver(secret_resolver)
        return _StdioClientConnectionConfig(
            transport_limits=transport_limits,
            request_timeout_s=request_timeout_s,
            write_timeout_s=_validate_stdio_client_timeout(
                self.write_timeout_s,
                "write_timeout_s",
            ),
            graceful_shutdown_timeout_s=_validate_stdio_client_timeout(
                self.graceful_shutdown_timeout_s,
                "graceful_shutdown_timeout_s",
            ),
            cancellation_notification_timeout_s=_validate_stdio_client_timeout(
                self.cancellation_notification_timeout_s,
                "cancellation_notification_timeout_s",
            ),
            client_name=require_clean_nonblank(self.client_name, "client_name"),
            client_version=require_clean_nonblank(self.client_version, "client_version"),
            max_list_pages=validate_positive_integer(self.max_list_pages, "max_list_pages"),
            max_list_items=validate_positive_integer(self.max_list_items, "max_list_items"),
            inherit_env=inherit_env,
            secret_resolver=secret_resolver,
            process_lifetime=validate_stdio_mcp_process_lifetime(self.process_lifetime),
            containment_startup_timeout_s=validate_stdio_mcp_containment_timeout(
                self.containment_startup_timeout_s,
                "containment_startup_timeout_s",
            ),
            containment_term_timeout_s=validate_stdio_mcp_containment_timeout(
                self.containment_term_timeout_s,
                "containment_term_timeout_s",
            ),
            containment_kill_timeout_s=validate_stdio_mcp_containment_timeout(
                self.containment_kill_timeout_s,
                "containment_kill_timeout_s",
            ),
        )

    async def connect(self, server: McpServerSpec) -> StdioMcpSession:
        server = copy_mcp_server_spec(server)
        # These public client attributes can be changed after construction.
        # Own and revalidate one exact launch configuration before any secret
        # lookup or process side effect, then carry it across every await.
        config = self._snapshot_connection_config()
        if server.command is None:
            raise ValueError("StdioMcpClient requires an MCP server command.")
        if server.url is not None:
            raise ValueError("StdioMcpClient does not support URL MCP servers.")
        if server.secret_env and config.secret_resolver is None:
            raise ValueError(
                "StdioMcpClient cannot resolve MCP secret_env without a secret_resolver. "
                "Pass secret_resolver= (a Vault or CredentialProxy) to the client."
            )
        if server.secret_headers:
            raise ValueError("StdioMcpClient does not support MCP secret_headers.")
        # Platform support is caller-visible configuration and must be checked
        # before secret lookup or process creation.
        validate_containment_platform(config.process_lifetime)
        # Own the selected host-environment view before preflight suspends. The
        # process environment and the mutable client must not be able to change
        # which host values the admitted launch receives.
        child_env = _base_child_env(config.inherit_env)
        child_env.update(server.env)
        secret_redactor = SecretRedactor()
        process_capability_evidence = None
        process = None
        prepared_containment_rendezvous = None
        containment_rendezvous_identity = None
        try:
            containment_preflight_proof = None
            if config.process_lifetime is StdioMcpProcessLifetime.PARENT_DEATH_CONTAINMENT:
                containment_preflight_proof = await preflight_stdio_mcp_parent_death_containment(
                    config.containment_startup_timeout_s
                )
                containment_rendezvous_identity = _stdio_containment_rendezvous_identity(server)
                prepared_containment_rendezvous = _prepare_stdio_mcp_containment_rendezvous(
                    containment_rendezvous_identity
                )
            process_capability_evidence = stdio_mcp_process_capability_evidence(
                config.process_lifetime,
                _parent_death_containment_proved=containment_preflight_proof is not None,
            )
            if server.secret_env and config.secret_resolver is not None:
                # Secret values go straight into the child process env — never into
                # argv — and stay wrapped until this final injection point.
                resolved = await resolve_secret_env(
                    server.secret_env,
                    config.secret_resolver,
                    scope={"mcp_server": server.name},
                )
                for name, secret in resolved.items():
                    child_env[name] = secret.value.get_secret_value()
                # A hostile server can echo these values back through tool output; scrub them.
                secret_redactor = SecretRedactor(tuple(resolved.values()))
                resolved.clear()
            if config.process_lifetime is StdioMcpProcessLifetime.PARENT_DEATH_CONTAINMENT:
                process = await create_contained_stdio_mcp_process(
                    *server.command,
                    env=child_env,
                    limit=config.transport_limits.max_message_bytes + 2,
                    startup_timeout_s=config.containment_startup_timeout_s,
                    term_timeout_s=config.containment_term_timeout_s,
                    kill_timeout_s=config.containment_kill_timeout_s,
                    _preflight_proof=containment_preflight_proof,
                    _rendezvous_identity=containment_rendezvous_identity,
                    _prepared_rendezvous=prepared_containment_rendezvous,
                )
            else:
                process = await create_direct_stdio_mcp_process(
                    *server.command,
                    env=child_env,
                    limit=config.transport_limits.max_message_bytes + 2,
                    lifetime=config.process_lifetime,
                )
        finally:
            # The child owns its copied environment after a successful spawn. Do
            # not retain injected values if spawning or initialize later fails.
            child_env.clear()
            if prepared_containment_rendezvous is not None:
                prepared_containment_rendezvous.close()
        assert process is not None
        assert process_capability_evidence is not None
        session = StdioMcpSession(
            server=server,
            process=process,
            request_timeout_s=config.request_timeout_s,
            transport_limits=(
                config.transport_limits if config.request_timeout_s is None else None
            ),
            write_timeout_s=config.write_timeout_s,
            graceful_shutdown_timeout_s=config.graceful_shutdown_timeout_s,
            cancellation_notification_timeout_s=config.cancellation_notification_timeout_s,
            client_name=config.client_name,
            client_version=config.client_version,
            max_list_pages=config.max_list_pages,
            max_list_items=config.max_list_items,
            secret_redactor=secret_redactor,
        )
        if session.process_capability_evidence != process_capability_evidence:
            # Process-derived evidence is authoritative. This assertion also
            # makes a lost private direct-process registration fail closed
            # before initialization can expose an incorrect lifecycle claim.
            await session.close()
            raise McpProtocolError("MCP stdio process lifecycle evidence did not match its launch.")
        cleanup_cancellation: asyncio.CancelledError | None = None
        try:
            await session.initialize()
        except asyncio.CancelledError as error:
            if _mcp_session_close_task(error) is None:
                _retain_mcp_session_close(session, primary_error=error)
            raise
        except TimeoutError as error:
            if _mcp_session_close_task(error) is None:
                _retain_mcp_session_close(session, primary_error=error)
            raise
        except Exception as error:
            cleanup_cancellation = await _close_mcp_session_after_primary_failure(
                session,
                primary_error=error,
                primary_context="MCP stdio initialization failed",
                cleanup_context="MCP stdio initialization cleanup failed",
            )
            if cleanup_cancellation is None:
                raise
        else:
            return session
        assert cleanup_cancellation is not None
        raise cleanup_cancellation


class StdioMcpSession(McpSession):
    def __init__(
        self,
        *,
        server: McpServerSpec,
        process: asyncio.subprocess.Process | ContainedStdioMcpProcess,
        request_timeout_s: float | None = None,
        transport_limits: McpTransportLimits | None = None,
        write_timeout_s: float,
        graceful_shutdown_timeout_s: float,
        cancellation_notification_timeout_s: float,
        client_name: str,
        client_version: str,
        max_list_pages: int = DEFAULT_MCP_MAX_LIST_PAGES,
        max_list_items: int = DEFAULT_MCP_MAX_LIST_ITEMS,
        secret_redactor: SecretRedactor | None = None,
    ) -> None:
        server = copy_mcp_server_spec(server)
        resolved_limits = resolve_mcp_transport_limits(
            transport_limits,
            legacy_timeout_s=request_timeout_s,
            default_timeout_s=DEFAULT_MCP_REQUEST_TIMEOUT_S,
            legacy_field_name="request_timeout_s",
        )
        self.server = server
        self.process = process
        self._process_capability_evidence = stdio_mcp_process_capability_evidence_for_process(
            process
        )
        self._secret_redactor = secret_redactor or SecretRedactor()
        self.transport_limits = resolved_limits
        self._uses_legacy_request_timeout = transport_limits is None
        self._request_timeout_s = resolved_limits.total_call_timeout_s
        self.write_timeout_s = write_timeout_s
        self.graceful_shutdown_timeout_s = graceful_shutdown_timeout_s
        self.cancellation_notification_timeout_s = cancellation_notification_timeout_s
        self.client_name = client_name
        self.client_version = client_version
        self.max_list_pages = validate_positive_integer(max_list_pages, "max_list_pages")
        self.max_list_items = validate_positive_integer(max_list_items, "max_list_items")
        self._initialize_result: McpInitializeResult | None = None
        self._next_id = 1
        self._closed = False
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._pending_timing: dict[int, _StdioPendingTiming] = {}
        self._stdout_buffer = bytearray()
        self._last_read_activity = asyncio.get_running_loop().time()
        self._last_expired_idle_gap: tuple[float, float] | None = None
        self._peer_closed_at: float | None = None
        self._write_lock = asyncio.Lock()
        self._stderr_tail = SecretRedactionTail(
            self._secret_redactor,
            max_bytes=DEFAULT_MCP_STDERR_CAPTURE_BYTES,
        )
        self._reader_task = asyncio.create_task(self._read_loop())
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        self._close_task: asyncio.Task[None] | None = None
        self._request_cleanup_tasks: set[asyncio.Task[None]] = set()
        self._tool_transport_names: dict[str, str] = {}
        self._resource_transport_uris: dict[str, str] = {}
        self._authority_mapping_lock = asyncio.Lock()

    @property
    def initialize_result(self) -> McpInitializeResult:
        if self._initialize_result is None:
            raise McpProtocolError("MCP session has not been initialized.")
        return self._initialize_result

    @property
    def request_timeout_s(self) -> float:
        return self._request_timeout_s

    @property
    def process_capability_evidence(self):
        return self._process_capability_evidence.model_copy(deep=True)

    @request_timeout_s.setter
    def request_timeout_s(self, value: float) -> None:
        timeout_s = validate_positive_number(value, "request_timeout_s")
        limits = self.transport_limits
        updated_limits = McpTransportLimits(
            max_message_bytes=limits.max_message_bytes,
            max_response_bytes=limits.max_response_bytes,
            idle_timeout_s=timeout_s,
            total_call_timeout_s=timeout_s,
        )
        if getattr(self, "_uses_legacy_request_timeout", False):
            self.transport_limits = updated_limits
        self._request_timeout_s = timeout_s

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

        try:
            initialize_result = await self._request(
                "initialize",
                initialize_params(self.client_name, self.client_version),
                result_parser=parse_initialize_result,
            )
            await self._notify("notifications/initialized", {})
            self._initialize_result = initialize_result
        except BaseException as error:
            self._initialize_result = None
            self._closed = True
            _retain_mcp_session_close(self, primary_error=error)
            raise

    async def list_tools(self) -> tuple[McpToolDefinition, ...]:
        discovery = await self._discover_builtin_tools_for_toolset()
        await discovery.commit()
        return discovery.definitions

    async def _discover_tools_for_toolset(self) -> _McpToolDiscovery:
        if type(self).list_tools is not StdioMcpSession.list_tools:
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
                        "MCP stdio session closed before tool discovery was published."
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
        if type(self) is not StdioMcpSession:
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
        # The request owns its shallow parameter object. Drop caller-owned input
        # from this public frame before a typed outbound-overflow traceback can
        # cross the MCP boundary.
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
        sanitized_cancellation: asyncio.CancelledError | None = None
        try:
            current_task = asyncio.current_task()
            request_cleanup_tasks = tuple(
                task
                for task in self._request_cleanup_tasks
                if task is not current_task and not task.done()
            )
            if request_cleanup_tasks:
                await asyncio.gather(
                    *(asyncio.shield(task) for task in request_cleanup_tasks),
                    return_exceptions=True,
                )
            self._schedule_close()
            assert self._close_task is not None
            await _await_mcp_session_cleanup_task(
                self._close_task,
                redactor=self._secret_redactor,
                context="MCP stdio session cleanup failed",
            )
        except asyncio.CancelledError as cancellation:
            sanitized_cancellation = _credential_safe_mcp_cancellation(
                cancellation,
                redactor=self._secret_redactor,
            )
        if sanitized_cancellation is not None:
            raise sanitized_cancellation

    def _fence_before_retained_close(self) -> bool:
        # Even when a subclass must finish its own close synchronously, the
        # inherited stdio request entrances can be fenced immediately.
        self._closed = True
        # Subclasses can own additional dispatch paths or pending work. They must
        # opt in with their own positive fencing proof rather than inheriting
        # authority established only for this implementation.
        return type(self) is StdioMcpSession

    def _schedule_close(self) -> None:
        self._closed = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close_impl_with_safe_failure())
            self._close_task.add_done_callback(_consume_task_result)

    async def _close_impl_with_safe_failure(self) -> None:
        safe_failure: BaseException | None = None
        try:
            await self._close_impl()
        except BaseException as error:
            safe_failure = credential_safe_mcp_transport_failure(
                error,
                redactor=self._secret_redactor,
                context="MCP stdio session cleanup failed",
            )
            error = None
        if safe_failure is not None:
            raise safe_failure from None

    async def _close_impl(self) -> None:
        failures: list[BaseException] = []
        try:
            # Logical closure is already authoritative. Settle callers before
            # waiting on the external process tree so a bounded TERM/KILL
            # sequence cannot turn their outcome into an unrelated call timeout.
            self._fail_pending(McpProtocolError("MCP stdio session closed."))
        except BaseException as error:
            failures.append(error)

        async def settle_shutdown_task(
            task: asyncio.Task[Any],
            *,
            timeout_s: float | None = None,
        ) -> bool:
            done, _pending = await asyncio.wait(
                (task,),
                timeout=(self.graceful_shutdown_timeout_s if timeout_s is None else timeout_s),
            )
            if task not in done:
                _retain_cancelled_stdio_shutdown_task(task)
                return False
            try:
                task.result()
            except BaseException as error:
                failures.append(error)
            return True

        contained_settlement_checked = False

        async def wait_for_process_exit(*, final: bool = False) -> bool:
            nonlocal contained_settlement_checked
            contained = isinstance(self.process, ContainedStdioMcpProcess)
            if self.process.returncode is not None and not (final and contained):
                return True
            if final and contained:
                wait_task = asyncio.create_task(self.process.wait_for_settlement())
            else:
                wait_task = asyncio.create_task(self.process.wait())
            final_timeout_s = None
            if final and contained:
                # Await the complete configured process-tree settlement bound.
                # A short transport grace period must not truncate a larger
                # TERM/KILL and reaping allowance selected by the caller.
                final_timeout_s = self.process.settlement_timeout_s
            settled = await settle_shutdown_task(wait_task, timeout_s=final_timeout_s)
            if contained and (settled or final):
                # A final wait consumes this close operation's one complete
                # settlement allowance even if the retained owner needs longer
                # to reach quiescence. Do not spend that allowance twice.
                contained_settlement_checked = True
            process_exited = self.process.returncode is not None
            if final and not settled and (contained or not process_exited):
                failures.append(
                    McpProtocolError(
                        "MCP stdio process did not provide authenticated settlement "
                        "after it was killed."
                    )
                )
            return process_exited

        if self.process.returncode is None:
            stdin_close_task = asyncio.create_task(self._close_stdin_for_graceful_shutdown())
            await settle_shutdown_task(stdin_close_task)
            process_exited = self.process.returncode is not None
            if not process_exited:
                # Preserve the existing stdio contract: EOF receives one full
                # graceful interval before TERM is requested.
                process_exited = await wait_for_process_exit()
            if not process_exited and self.process.returncode is None:
                try:
                    self.process.terminate()
                except ProcessLookupError:
                    pass
                except BaseException as error:
                    failures.append(error)
                process_exited = self.process.returncode is not None
                if not process_exited:
                    process_exited = await wait_for_process_exit()
            if not process_exited and self.process.returncode is None:
                try:
                    self.process.kill()
                except ProcessLookupError:
                    pass
                except BaseException as error:
                    failures.append(error)
                if self.process.returncode is None:
                    await wait_for_process_exit(final=True)
        if isinstance(self.process, ContainedStdioMcpProcess) and not contained_settlement_checked:
            # A supervisor can exit before close begins. Its exit code is not
            # positive process-tree settlement evidence, so always validate the
            # authenticated settled receipt before reporting successful close.
            await wait_for_process_exit(final=True)
        try:
            self._stdout_buffer.clear()
        except BaseException as error:
            failures.append(error)
        for task in (self._reader_task, self._stderr_task):
            try:
                await self._cancel_background_task(task)
            except BaseException as error:
                failures.append(error)
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise BaseExceptionGroup("MCP stdio session cleanup failures.", failures)

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
        if self._closed:
            raise McpProtocolError("MCP stdio session is closed.")
        loop = asyncio.get_running_loop()
        started_at = loop.time()
        # Legacy sessions retain their mutable timeout alias for compatibility;
        # an explicit immutable limits object remains authoritative.
        call_timeout_s = (
            self.request_timeout_s
            if self._uses_legacy_request_timeout
            else self.transport_limits.total_call_timeout_s
        )
        idle_timeout_s = self.transport_limits.idle_timeout_s
        call_deadline = started_at + call_timeout_s
        method_name = require_clean_nonblank(method, "method")
        request_id = self._next_id
        self._next_id += 1
        request_written = False
        sanitized_cancellation: asyncio.CancelledError | None = None
        sanitized_failure: BaseException | None = None
        preparation_error: BaseException | None = None
        try:
            try:
                request_preflight = mcp_jsonrpc_request_preflight(
                    request_id,
                    method_name,
                    params,
                    max_bytes=self.transport_limits.max_message_bytes,
                )
                if request_preflight.exceeds_limit:
                    size_error = McpMessageTooLargeError(
                        "MCP stdio JSON-RPC message exceeded "
                        f"{self.transport_limits.max_message_bytes} bytes."
                    )
                    raise size_error from None
                if request_preflight.nesting_too_deep:
                    raise McpProtocolError(
                        "MCP stdio JSON-RPC request exceeded the supported JSON nesting."
                    ) from None
                if loop.time() >= call_deadline:
                    self._schedule_close()
                    raise McpCallDeadlineExceededError(
                        "MCP stdio exchange exceeded its total call deadline."
                    )
                payload = jsonrpc_request_payload(request_id, method_name, params)
            except (RecursionError, TypeError, ValueError) as error:
                preparation_error = credential_safe_mcp_transport_failure(
                    error,
                    redactor=self._secret_redactor,
                    context=f"MCP {method_name} request preparation failed",
                )
                error = None
        finally:
            # This shallow container is transport-owned. Scrub every preparation
            # exit without mutating nested dictionaries still owned by the caller.
            params.clear()
        if preparation_error is not None:
            raise preparation_error from None
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        write_cancellation_boundary = _McpCallerCancellationBoundary()
        try:
            await write_cancellation_boundary.checkpoint()
            if dispatch_signal is not None:
                # _write_with_timeout() creates its owned write task before its
                # first suspension. Once this task yields, target execution is an
                # uncertain outcome and refresh must let the exact call settle.
                dispatch_signal.mark_dispatched()
            await self._write_with_timeout(
                payload,
                timeout_message=f"MCP request {request_id} write timed out.",
                call_deadline=call_deadline,
            )
            request_written = True
            payload.clear()
        except asyncio.CancelledError as error:
            self._pending.pop(request_id, None)
            self._pending_timing.pop(request_id, None)
            future.cancel()
            if self._close_task is not None and _mcp_session_close_task(error) is None:
                _attach_mcp_session_cleanup_task(error, self._close_task)
            payload.clear()
            if write_cancellation_boundary.caller_cancelled():
                sanitized_cancellation = _credential_safe_mcp_cancellation(
                    error,
                    redactor=self._secret_redactor,
                )
            else:
                sanitized_failure = credential_safe_mcp_transport_failure(
                    error,
                    redactor=self._secret_redactor,
                    context="MCP stdio transport write was cancelled unexpectedly",
                    preserve_cause=True,
                )
                if _mcp_session_close_task(sanitized_failure) is None:
                    self._schedule_close()
                    assert self._close_task is not None
                    _attach_mcp_session_cleanup_task(sanitized_failure, self._close_task)
        except TimeoutError as error:
            self._pending.pop(request_id, None)
            self._pending_timing.pop(request_id, None)
            future.cancel()
            if self._close_task is not None and _mcp_session_close_task(error) is None:
                _attach_mcp_session_cleanup_task(error, self._close_task)
            payload.clear()
            sanitized_failure = credential_safe_mcp_transport_failure(
                error,
                redactor=self._secret_redactor,
                context=f"MCP {method_name} request write timed out",
                preserve_cause=True,
            )
        except (BaseExceptionGroup, Exception) as error:
            self._pending.pop(request_id, None)
            self._pending_timing.pop(request_id, None)
            _clear_completed_stdio_response(future)
            future.cancel()
            if self._close_task is not None and _mcp_session_close_task(error) is None:
                _attach_mcp_session_cleanup_task(error, self._close_task)
            payload.clear()
            raise
        except (KeyboardInterrupt, SystemExit, GeneratorExit) as error:
            self._pending.pop(request_id, None)
            self._pending_timing.pop(request_id, None)
            _clear_completed_stdio_response(future)
            future.cancel()
            payload.clear()
            sanitized_failure = credential_safe_mcp_fatal_signal(
                error,
                redactor=self._secret_redactor,
                context=f"MCP {method_name} request write failed",
            )
            error = None
        if sanitized_failure is not None:
            raise sanitized_failure
        if sanitized_cancellation is not None:
            raise sanitized_cancellation
        response_cancellation_boundary = _McpCallerCancellationBoundary()
        try:
            await response_cancellation_boundary.checkpoint()
            response = await self._wait_for_response(
                future,
                request_id=request_id,
                started_at=started_at,
                call_deadline=call_deadline,
                idle_timeout_s=idle_timeout_s,
            )
        except TimeoutError as error:
            self._pending.pop(request_id, None)
            self._pending_timing.pop(request_id, None)
            _clear_completed_stdio_response(future)
            future.cancel()
            if request_written:
                self._fence_uncertain_request()
                self._schedule_uncertain_request_cleanup(
                    request_id,
                    method_name=method_name,
                    reason="Cayu request timed out.",
                    primary_error=error,
                )
            sanitized_failure = credential_safe_mcp_transport_failure(
                error,
                redactor=self._secret_redactor,
                context=f"MCP {method_name} request timed out",
                preserve_cause=True,
            )
        except asyncio.CancelledError as error:
            self._pending.pop(request_id, None)
            self._pending_timing.pop(request_id, None)
            _clear_completed_stdio_response(future)
            future.cancel()
            caller_cancelled = response_cancellation_boundary.caller_cancelled()
            primary_error: BaseException | None = error
            if not caller_cancelled:
                primary_error = credential_safe_mcp_transport_failure(
                    error,
                    redactor=self._secret_redactor,
                    context="MCP stdio response wait was cancelled unexpectedly",
                    preserve_cause=True,
                )
            if request_written:
                assert primary_error is not None
                self._fence_uncertain_request()
                self._schedule_uncertain_request_cleanup(
                    request_id,
                    method_name=method_name,
                    reason=(
                        "Cayu caller cancelled the request."
                        if caller_cancelled
                        else "Cayu response wait was cancelled unexpectedly."
                    ),
                    primary_error=primary_error,
                )
            if caller_cancelled:
                sanitized_cancellation = _credential_safe_mcp_cancellation(
                    error,
                    redactor=self._secret_redactor,
                )
            else:
                sanitized_failure = primary_error
            primary_error = None
        if sanitized_failure is not None:
            raise sanitized_failure
        if sanitized_cancellation is not None:
            raise sanitized_cancellation
        redaction_result = safely_redact_jsonrpc_response(
            response,
            method=method_name,
            redactor=self._secret_redactor,
        )
        redacted_response = redaction_result.response
        mapping_result = JsonrpcAuthorityMappingResult({})
        mapping_error = redaction_result.error
        private_evidence_error: str | None = None
        private_cursor: Any = None
        raw_result: Any = None
        if paginated and mapping_error is None:
            raw_result = response.get("result")
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
                response,
                redacted_response,
                method=method_name,
            )
            mapping_error = mapping_result.error
        if mapping_error is None and private_tool_contract_hashes is not None:
            evidence_result = jsonrpc_tool_contract_evidence(
                response,
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
        response.clear()
        response = {}
        if loop.time() >= call_deadline:
            if authority_mapping is not None:
                authority_mapping.clear()
            private_cursor = None
            redacted_response.clear()
            self._raise_completed_response_deadline(
                request_id=request_id,
                method_name=method_name,
            )
        if mapping_error is not None:
            private_cursor = None
            raise McpProtocolError(
                self._secret_redactor.redact_text_bounded(
                    mapping_error,
                    max_bytes=4096,
                )
            ) from None
        result: Any = None
        try:
            result = result_from_jsonrpc_response(redacted_response, method_name)
            redacted_response.clear()
            redaction_result = None
            if result_parser is not None:
                result = result_parser(result)
        except BaseException:
            private_cursor = None
            redacted_response.clear()
            redaction_result = None
            if type(result) in {dict, list}:
                result.clear()
            result = None
            if loop.time() >= call_deadline:
                if authority_mapping is not None:
                    authority_mapping.clear()
                redacted_response.clear()
                self._raise_completed_response_deadline(
                    request_id=request_id,
                    method_name=method_name,
                )
            raise
        if loop.time() >= call_deadline:
            if authority_mapping is not None:
                authority_mapping.clear()
            private_cursor = None
            if type(result) in {dict, list}:
                result.clear()
            redacted_response.clear()
            self._raise_completed_response_deadline(
                request_id=request_id,
                method_name=method_name,
            )
        if private_evidence_error is not None:
            if private_tool_contract_hashes is not None:
                private_tool_contract_hashes.clear()
            private_cursor = None
            if type(result) in {dict, list}:
                result.clear()
            raise McpProtocolError(private_evidence_error) from None
        if not paginated:
            return result
        if type(result) is dict:
            result.pop("nextCursor", None)
        return McpPaginatedPage(
            result=result,
            next_cursor=private_cursor,
        )

    def _raise_completed_response_deadline(
        self,
        *,
        request_id: int,
        method_name: str,
    ) -> None:
        """Fence a completed response that missed its final publication deadline."""

        error = McpCallDeadlineExceededError(
            f"MCP request {request_id} exceeded its total call deadline during response processing."
        )
        self._fence_uncertain_request()
        self._schedule_close()
        assert self._close_task is not None
        _attach_mcp_session_cleanup_task(error, self._close_task)
        safe_error = credential_safe_mcp_transport_failure(
            error,
            redactor=self._secret_redactor,
            context=f"MCP {method_name} request timed out",
            preserve_cause=True,
        )
        raise safe_error from None

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        method_name = require_clean_nonblank(method, "method")
        sanitized_cancellation: asyncio.CancelledError | None = None
        sanitized_failure: BaseException | None = None
        cancellation_boundary = _McpCallerCancellationBoundary()
        try:
            await cancellation_boundary.checkpoint()
            await self._write_with_timeout(
                jsonrpc_notification_payload(method_name, params),
                timeout_message=f"MCP notification {method_name} write timed out.",
            )
        except asyncio.CancelledError as cancellation:
            if cancellation_boundary.caller_cancelled():
                sanitized_cancellation = _credential_safe_mcp_cancellation(
                    cancellation,
                    redactor=self._secret_redactor,
                )
            else:
                sanitized_failure = credential_safe_mcp_transport_failure(
                    cancellation,
                    redactor=self._secret_redactor,
                    context=f"MCP notification {method_name} write was cancelled unexpectedly",
                    preserve_cause=True,
                )
                if _mcp_session_close_task(sanitized_failure) is None:
                    self._schedule_close()
                    assert self._close_task is not None
                    _attach_mcp_session_cleanup_task(sanitized_failure, self._close_task)
        except (KeyboardInterrupt, SystemExit, GeneratorExit) as error:
            sanitized_failure = credential_safe_mcp_fatal_signal(
                error,
                redactor=self._secret_redactor,
                context=f"MCP notification {method_name} write failed",
            )
            error = None
        if sanitized_failure is not None:
            raise sanitized_failure
        if sanitized_cancellation is not None:
            raise sanitized_cancellation

    async def _wait_for_response(
        self,
        future: asyncio.Future[dict[str, Any]],
        *,
        request_id: int,
        started_at: float,
        call_deadline: float,
        idle_timeout_s: float,
    ) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        while True:
            now = loop.time()
            last_activity = max(started_at, self._last_read_activity)
            idle_deadline = last_activity + idle_timeout_s
            peer_closed_at = getattr(self, "_peer_closed_at", None)
            if (
                peer_closed_at is not None
                and peer_closed_at < call_deadline
                and peer_closed_at < idle_deadline
            ):
                # Peer closure is authoritative as soon as stdout reports EOF.
                # Stderr enrichment remains optional and cannot extend delivery
                # beyond this request's original total deadline.
                await self._await_stderr_drain(deadline=call_deadline)
                self._pending_timing.pop(request_id, None)
                _clear_completed_stdio_response(future)
                raise self._peer_closed_error("MCP stdio process closed stdout.")
            if now >= call_deadline:
                raise McpCallDeadlineExceededError(
                    f"MCP request {request_id} timed out at its total call deadline."
                )
            if future.done():
                if loop.time() >= call_deadline:
                    raise McpCallDeadlineExceededError(
                        f"MCP request {request_id} timed out at its total call deadline."
                    )
                timing = self._pending_timing.pop(request_id, None)
                if timing is not None and _stdio_pending_crossed_idle_deadline(
                    timing,
                    started_at=started_at,
                    idle_timeout_s=idle_timeout_s,
                ):
                    raise McpIdleTimeoutError(
                        f"MCP request {request_id} exceeded its idle timeout."
                    )
                return future.result()
            if _stdio_gap_crossed_idle_deadline(
                self._last_expired_idle_gap,
                started_at=started_at,
                idle_timeout_s=idle_timeout_s,
            ):
                raise McpIdleTimeoutError(f"MCP request {request_id} exceeded its idle timeout.")
            if now >= idle_deadline:
                raise McpIdleTimeoutError(f"MCP request {request_id} exceeded its idle timeout.")
            await asyncio.wait(
                {future},
                timeout=min(call_deadline, idle_deadline) - now,
            )

    async def _write_with_timeout(
        self,
        payload: dict[str, Any],
        *,
        timeout_message: str,
        call_deadline: float | None = None,
    ) -> None:
        loop = asyncio.get_running_loop()
        if call_deadline is None:
            call_timeout_s = (
                self.request_timeout_s
                if self._uses_legacy_request_timeout
                else self.transport_limits.total_call_timeout_s
            )
            call_deadline = loop.time() + call_timeout_s
        now = loop.time()
        remaining = call_deadline - now
        if remaining <= 0:
            self._schedule_close()
            raise McpCallDeadlineExceededError(
                "MCP stdio exchange exceeded its total call deadline."
            )
        write_deadline = min(call_deadline, now + self.write_timeout_s)
        write_task = asyncio.create_task(_capture_mcp_owned_task_fatal_signal(self._write(payload)))
        payload = {}
        cancellation_boundary = _McpCallerCancellationBoundary()
        unexpected_wait_error: McpProtocolError | None = None
        try:
            await cancellation_boundary.checkpoint()
            now = loop.time()
            if now >= write_deadline:
                if now >= call_deadline:
                    checkpoint_timeout: TimeoutError = McpCallDeadlineExceededError(
                        "MCP stdio exchange exceeded its total call deadline."
                    )
                else:
                    checkpoint_timeout = TimeoutError(timeout_message)
                self._schedule_interrupted_write_cleanup(
                    write_task,
                    primary_error=checkpoint_timeout,
                )
                raise checkpoint_timeout from None
            done, _ = await asyncio.wait(
                {write_task},
                timeout=write_deadline - now,
            )
        except asyncio.CancelledError as error:
            if cancellation_boundary.caller_cancelled():
                self._schedule_interrupted_write_cleanup(
                    write_task,
                    primary_error=error,
                )
                raise
            unexpected_wait_error = McpProtocolError(
                "MCP stdio write wait was cancelled unexpectedly; session closed."
            )
            self._schedule_interrupted_write_cleanup(
                write_task,
                primary_error=unexpected_wait_error,
            )
            done = set()
        if unexpected_wait_error is not None:
            raise unexpected_wait_error from None
        observed_at = loop.time()
        if not done or observed_at >= write_deadline:
            if observed_at >= call_deadline:
                timeout_error: TimeoutError = McpCallDeadlineExceededError(
                    "MCP stdio exchange exceeded its total call deadline."
                )
            else:
                timeout_error = TimeoutError(timeout_message)
            self._schedule_interrupted_write_cleanup(
                write_task,
                primary_error=timeout_error,
            )
            raise timeout_error from None
        if write_task.cancelled():
            unexpected_error = McpProtocolError(
                "MCP stdio transport write was cancelled unexpectedly; session closed."
            )
            self._schedule_interrupted_write_cleanup(
                write_task,
                primary_error=unexpected_error,
            )
            raise unexpected_error from None
        peer_closed_error: McpPeerClosedError | None = None
        pre_dispatch_size_error: McpMessageTooLargeError | None = None
        uncertain_write_error: BaseException | None = None
        try:
            _unwrap_mcp_owned_task_result(write_task.result())
        except _StdioPreDispatchMessageTooLargeError as error:
            # This private type is produced only by validation before
            # stdin.write(). The public type alone is not proof of provenance.
            pre_dispatch_size_error = McpMessageTooLargeError(str(error))
        except (BrokenPipeError, ConnectionResetError):
            try:
                await self._await_stderr_drain(deadline=call_deadline)
            except asyncio.CancelledError as cancellation:
                self._schedule_interrupted_write_cleanup(
                    write_task,
                    primary_error=cancellation,
                )
                raise
            peer_closed_error = self._peer_closed_error("MCP stdio process closed stdin.")
            self._schedule_interrupted_write_cleanup(
                write_task,
                primary_error=peer_closed_error,
            )
        except (BaseExceptionGroup, Exception) as error:
            uncertain_write_error = credential_safe_mcp_transport_failure(
                error,
                redactor=self._secret_redactor,
                context="MCP stdio transport write failed",
            )
            self._schedule_interrupted_write_cleanup(
                write_task,
                primary_error=uncertain_write_error,
            )
        except (KeyboardInterrupt, SystemExit, GeneratorExit) as error:
            uncertain_write_error = credential_safe_mcp_fatal_signal(
                error,
                redactor=self._secret_redactor,
                context="MCP stdio transport write failed",
            )
            self._schedule_interrupted_write_cleanup(
                write_task,
                primary_error=uncertain_write_error,
            )
            error = None
        except BaseException as error:
            uncertain_write_error = credential_safe_mcp_transport_failure(
                error,
                redactor=self._secret_redactor,
                context="MCP stdio transport write failed",
            )
            self._schedule_interrupted_write_cleanup(
                write_task,
                primary_error=uncertain_write_error,
            )
            error = None
        if peer_closed_error is not None:
            # Raise outside the raw pipe exception handler so it cannot survive as
            # implicit context on the public, credential-safe transport failure.
            raise peer_closed_error from None
        if pre_dispatch_size_error is not None:
            done.clear()
            del write_task
            raise pre_dispatch_size_error from None
        if uncertain_write_error is not None:
            # The retained settlement task owns the raw completed writer. Remove
            # this frame's reference so public traceback locals cannot render its
            # extension-controlled exception while cleanup is still settling.
            done.clear()
            del write_task
            raise uncertain_write_error from None

    def _schedule_interrupted_write_cleanup(
        self,
        write_task: asyncio.Task[Any],
        *,
        primary_error: BaseException,
    ) -> None:
        """Fence promptly while retaining the exact possibly-running writer."""

        self._closed = True
        write_task.cancel()
        self._schedule_close()
        assert self._close_task is not None
        close_task = self._close_task
        retained_write_task: asyncio.Task[Any] | None = write_task

        async def settle_write_and_session() -> None:
            nonlocal retained_write_task
            owned_write_task = retained_write_task
            assert owned_write_task is not None
            failures: list[BaseException] = []
            try:
                await asyncio.shield(close_task)
            except BaseException as error:
                failures.append(
                    credential_safe_mcp_transport_failure(
                        error,
                        redactor=self._secret_redactor,
                        context="MCP stdio interrupted-write session cleanup failed",
                    )
                )
            cancellation_boundary = _McpCallerCancellationBoundary()
            try:
                await cancellation_boundary.checkpoint()
                _unwrap_mcp_owned_task_result(await asyncio.shield(owned_write_task))
            except asyncio.CancelledError:
                if cancellation_boundary.caller_cancelled():
                    raise
                # Cancellation of the owned writer was requested above. Session
                # closure is the positive fence; a cancelled writer adds no failure.
            except BaseException as error:
                failures.append(
                    credential_safe_mcp_transport_failure(
                        error,
                        redactor=self._secret_redactor,
                        context="MCP stdio interrupted transport write failed",
                    )
                )
            retained_write_task = None
            del owned_write_task
            if len(failures) == 1:
                raise failures[0]
            if failures:
                raise BaseExceptionGroup(
                    "MCP stdio interrupted-write cleanup failures.",
                    failures,
                )

        settlement_task = asyncio.create_task(settle_write_and_session())
        _RETAINED_STDIO_WRITE_SETTLEMENT_TASKS.add(settlement_task)
        _attach_mcp_session_cleanup_task(primary_error, settlement_task)

        def completed(task: asyncio.Task[None]) -> None:
            _RETAINED_STDIO_WRITE_SETTLEMENT_TASKS.discard(task)
            _consume_task_result(task)

        settlement_task.add_done_callback(completed)

    async def _write(self, payload: dict[str, Any]) -> None:
        if self.process.stdin is None:
            raise McpProtocolError("MCP stdio process stdin is unavailable.")
        if not json_utf8_size_within_limit(
            payload,
            self.transport_limits.max_message_bytes,
            ensure_ascii=True,
        ):
            payload.clear()
            raise _StdioPreDispatchMessageTooLargeError(
                "MCP stdio JSON-RPC message exceeded "
                f"{self.transport_limits.max_message_bytes} bytes."
            )
        data = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        payload = {}
        try:
            if len(data) > self.transport_limits.max_message_bytes:
                raise _StdioPreDispatchMessageTooLargeError(
                    "MCP stdio JSON-RPC message exceeded "
                    f"{self.transport_limits.max_message_bytes} bytes."
                )
            async with self._write_lock:
                self.process.stdin.write(data + b"\n")
                await self.process.stdin.drain()
        finally:
            data = b""

    async def _close_stdin_for_graceful_shutdown(self) -> None:
        stdin = self.process.stdin
        if stdin is None:
            return
        stdin.close()
        wait_closed = getattr(stdin, "wait_closed", None)
        if wait_closed is not None:
            with suppress(BrokenPipeError, ConnectionResetError):
                await wait_closed()

    async def _cancel_background_task(self, task: asyncio.Task) -> None:
        if task.done():
            _consume_task_result(task)
            return
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    async def _send_request_cancelled_notification(
        self,
        request_id: int,
        *,
        method_name: str,
        reason: str,
    ) -> None:
        if method_name == "initialize":
            return
        notify_task = asyncio.create_task(
            _capture_mcp_owned_task_fatal_signal(
                self._notify(
                    "notifications/cancelled",
                    {"requestId": request_id, "reason": reason},
                )
            )
        )
        try:
            notify_outcome = await asyncio.wait_for(
                asyncio.shield(notify_task),
                timeout=self.cancellation_notification_timeout_s,
            )
            _unwrap_mcp_owned_task_result(notify_outcome)
        except (Exception, asyncio.CancelledError):
            notify_task.cancel()
            notify_task.add_done_callback(_consume_task_result)
            self._schedule_close()
            assert self._close_task is not None
            while True:
                try:
                    await asyncio.shield(self._close_task)
                    break
                except asyncio.CancelledError:
                    if self._close_task.done():
                        break

    def _schedule_uncertain_request_cleanup(
        self,
        request_id: int,
        *,
        method_name: str,
        reason: str,
        primary_error: BaseException,
    ) -> None:
        async def notify_then_close() -> None:
            try:
                await self._send_request_cancelled_notification(
                    request_id,
                    method_name=method_name,
                    reason=reason,
                )
            finally:
                self._schedule_close()
                assert self._close_task is not None
                await _await_mcp_session_cleanup_task(
                    self._close_task,
                    redactor=self._secret_redactor,
                    context="MCP stdio uncertain-request cleanup failed",
                )

        async def owned_cleanup() -> None:
            safe_failure: BaseException | None = None
            try:
                await notify_then_close()
            except BaseException as error:
                safe_failure = credential_safe_mcp_transport_failure(
                    error,
                    redactor=self._secret_redactor,
                    context="MCP stdio uncertain-request cleanup failed",
                )
                error = None
            if safe_failure is not None:
                raise safe_failure from None

        cleanup_task = asyncio.create_task(owned_cleanup())
        self._request_cleanup_tasks.add(cleanup_task)
        _attach_mcp_session_cleanup_task(primary_error, cleanup_task)

        def completed(task: asyncio.Task[None]) -> None:
            self._request_cleanup_tasks.discard(task)
            _consume_task_result(task)

        cleanup_task.add_done_callback(completed)

    def _fence_uncertain_request(self) -> None:
        self._closed = True
        self._fail_pending(
            McpProtocolError("MCP stdio session closed after an uncertain request outcome.")
        )

    async def _read_loop(self) -> None:
        error: BaseException | None = None
        try:
            while True:
                message = await self._read_message()
                sanitized_error: McpProtocolError | None = None
                try:
                    await self._handle_message(message)
                except Exception as exc:
                    sanitized_error = McpProtocolError(
                        self._secret_redactor.redact_text_bounded(
                            str(exc),
                            max_bytes=4096,
                        )
                    )
                finally:
                    message.clear()
                    message = {}
                if sanitized_error is not None:
                    raise sanitized_error from None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = exc
        finally:
            # The reader loop only exits once the transport is dead. Latch the
            # session closed so subsequent requests fast-fail immediately instead
            # of blocking for the full request timeout on a future no reader will
            # ever resolve.
            self._stdout_buffer.clear()
            self._fail_pending(
                error if error is not None else McpProtocolError("MCP stdio reader stopped."),
            )
            self._closed = True
            if error is not None:
                self._schedule_close()

    async def _handle_message(self, message: dict[str, Any]) -> None:
        message_id = message.get("id")
        if "method" in message:
            if message_id is not None:
                await self._write_server_request_error(message)
            return
        if message_id is None:
            raise McpProtocolError("MCP JSON-RPC response is missing an id.")
        if type(message_id) is not int:
            raise McpProtocolError("MCP JSON-RPC response id must be an integer.")
        future = self._pending.get(message_id)
        if future is None or future.done():
            return
        # Detach the request result so the reader can clear its raw transport
        # frame before waiting for the next message.
        copy_error: McpProtocolError | None = None
        try:
            copied_message = copy_json_value(message, "MCP stdio response")
        except ValueError:
            copied_message = None
            copy_error = McpProtocolError("MCP response contained invalid portable JSON.")
        self._pending.pop(message_id, None)
        if copy_error is not None:
            self._record_pending_timing(message_id, response_received=True)
            future.set_exception(copy_error)
            return
        if type(copied_message) is not dict:
            raise AssertionError("MCP stdio response copy returned a non-object.")
        self._record_pending_timing(message_id, response_received=True)
        future.set_result(copied_message)

    async def _write_server_request_error(self, message: dict[str, Any]) -> None:
        request_id = message.get("id")
        method = message.get("method")
        method_name = method if isinstance(method, str) else "unknown"
        safe_method_name = self._secret_redactor.redact_text(method_name)
        await self._write_with_timeout(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": JSONRPC_METHOD_NOT_FOUND,
                    "message": f"Cayu does not support MCP server request: {safe_method_name}",
                },
            },
            timeout_message=f"MCP server request rejection for {safe_method_name} write timed out.",
        )

    def _fail_pending(self, error: BaseException) -> None:
        pending = self._pending
        self._pending = {}
        for request_id, future in pending.items():
            if not future.done():
                self._record_pending_timing(request_id, response_received=False)
                # Exceptions retain mutable traceback, cause, note, and private
                # handoff state. Publish a detached instance per waiter so one
                # concurrent caller cannot mutate another caller's diagnostic.
                future.set_exception(
                    credential_safe_mcp_transport_failure(
                        error,
                        redactor=self._secret_redactor,
                        context="MCP stdio transport failed",
                        preserve_cause=True,
                    )
                )

    def _record_pending_timing(self, request_id: int, *, response_received: bool) -> None:
        self._pending_timing[request_id] = _StdioPendingTiming(
            settled_at=asyncio.get_running_loop().time(),
            last_read_activity=self._last_read_activity,
            expired_idle_gap=self._last_expired_idle_gap,
            response_received=response_received,
        )

    async def _read_message(self) -> dict[str, Any]:
        if self.process.stdout is None:
            raise McpProtocolError("MCP stdio process stdout is unavailable.")
        line = await self._read_frame()
        decoded_line = ""
        payload: Any = None
        protocol_error: str | None = None
        try:
            decoded_line = line.decode("utf-8", "strict")
        except UnicodeDecodeError:
            protocol_error = "MCP stdio process wrote invalid UTF-8."
        if protocol_error is None:
            try:
                payload = json.loads(decoded_line)
            except RecursionError:
                protocol_error = "MCP stdio JSON-RPC response exceeded the supported JSON nesting."
            except ValueError:
                protocol_error = "MCP stdio process wrote invalid JSON."
        # Drop the transport document before structural validation raises. A
        # valid JSON value with an invalid JSON-RPC envelope remains untrusted
        # and may contain a resolved secret echoed by the server.
        line = b""
        decoded_line = ""
        if protocol_error is None:
            if mcp_json_value_nesting_too_deep(payload):
                protocol_error = "MCP stdio JSON-RPC response exceeded the supported JSON nesting."
            elif type(payload) is not dict:
                protocol_error = "MCP JSON-RPC message must be an object."
            elif payload.get("jsonrpc") != "2.0":
                protocol_error = "MCP JSON-RPC message must use jsonrpc='2.0'."
        if protocol_error is not None:
            payload = None
            raise self._protocol_error(protocol_error)
        return payload

    async def _read_frame(self) -> bytes:
        stdout = self.process.stdout
        if stdout is None:
            raise McpProtocolError("MCP stdio process stdout is unavailable.")
        max_bytes = self.transport_limits.max_message_bytes
        while True:
            newline_index = self._stdout_buffer.find(b"\n")
            if newline_index >= 0:
                frame = bytes(self._stdout_buffer[:newline_index])
                del self._stdout_buffer[: newline_index + 1]
                if frame.endswith(b"\r"):
                    frame = frame[:-1]
                if len(frame) > max_bytes:
                    frame = b""
                    self._stdout_buffer.clear()
                    raise McpMessageTooLargeError(f"MCP stdio message exceeded {max_bytes} bytes.")
                return frame
            buffered = len(self._stdout_buffer)
            if buffered > max_bytes and not (
                buffered == max_bytes + 1 and self._stdout_buffer.endswith(b"\r")
            ):
                self._stdout_buffer.clear()
                raise McpMessageTooLargeError(f"MCP stdio message exceeded {max_bytes} bytes.")
            allocation_ceiling = max_bytes + 2
            if buffered >= allocation_ceiling:
                self._stdout_buffer.clear()
                raise McpMessageTooLargeError(f"MCP stdio message exceeded {max_bytes} bytes.")
            chunk = await stdout.read(min(65_536, allocation_ceiling - buffered))
            if not chunk:
                self._stdout_buffer.clear()
                self._peer_closed_at = asyncio.get_running_loop().time()
                # Wake every current waiter before optional stderr enrichment.
                # The per-request wait path uses the recorded timestamp to keep
                # peer closure distinct from later idle/total deadline expiry.
                self._fail_pending(McpPeerClosedError("MCP stdio process closed stdout."))
                self._schedule_close()
                await self._await_stderr_drain()
                raise self._peer_closed_error("MCP stdio process closed stdout.")
            received_at = asyncio.get_running_loop().time()
            previous_activity = self._last_read_activity
            if received_at >= previous_activity + self.transport_limits.idle_timeout_s:
                self._last_expired_idle_gap = (previous_activity, received_at)
            self._last_read_activity = received_at
            self._stdout_buffer.extend(chunk)
            chunk = b""

    async def _drain_stderr(self) -> None:
        stderr = self.process.stderr
        if stderr is None:
            self._stderr_tail.finish_complete()
            return
        try:
            while True:
                chunk = await stderr.read(4096)
                if not chunk:
                    self._stderr_tail.finish_complete()
                    return
                self._append_stderr(chunk)
        except BaseException:
            self._stderr_tail.abort()
            raise

    def _append_stderr(self, chunk: bytes) -> None:
        self._stderr_tail.feed(chunk)

    def _stderr_snapshot(self) -> str:
        return self._stderr_tail.text().strip()

    def _protocol_error(self, message: str) -> McpProtocolError:
        tail = self._stderr_snapshot()
        if not tail:
            return McpProtocolError(message)
        return McpProtocolError(f"{message} MCP server stderr (tail): {tail}")

    def _peer_closed_error(self, message: str) -> McpPeerClosedError:
        tail = self._stderr_snapshot()
        if not tail:
            return McpPeerClosedError(message)
        return McpPeerClosedError(f"{message} MCP server stderr (tail): {tail}")

    async def _await_stderr_drain(self, *, deadline: float | None = None) -> None:
        task = self._stderr_task
        if task is None or task.done():
            return
        timeout_s = DEFAULT_MCP_STDERR_DRAIN_GRACE_S
        if deadline is not None:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return
            timeout_s = min(timeout_s, remaining)
        with suppress(Exception):
            await asyncio.wait_for(
                asyncio.shield(task),
                timeout=timeout_s,
            )


def _stdio_pending_crossed_idle_deadline(
    timing: _StdioPendingTiming,
    *,
    started_at: float,
    idle_timeout_s: float,
) -> bool:
    gap = timing.expired_idle_gap
    if _stdio_gap_crossed_idle_deadline(
        gap,
        started_at=started_at,
        idle_timeout_s=idle_timeout_s,
        settled_at=timing.settled_at,
    ):
        return True
    if timing.response_received:
        return False
    request_idle_started_at = max(started_at, timing.last_read_activity)
    return timing.settled_at >= request_idle_started_at + idle_timeout_s


def _stdio_gap_crossed_idle_deadline(
    gap: tuple[float, float] | None,
    *,
    started_at: float,
    idle_timeout_s: float,
    settled_at: float | None = None,
) -> bool:
    if gap is None:
        return False
    gap_started_at, activity_resumed_at = gap
    if settled_at is not None and activity_resumed_at > settled_at:
        return False
    request_idle_started_at = max(started_at, gap_started_at)
    return activity_resumed_at >= request_idle_started_at + idle_timeout_s


def _consume_task_result(task: asyncio.Task) -> None:
    # Retained cleanup may deliberately aggregate process-control and ordinary
    # failures. Consume the complete group at this terminal observer.
    with suppress(BaseException):
        task.result()


def _retain_cancelled_stdio_shutdown_task(task: asyncio.Task[Any]) -> None:
    """Cancel a timed-out shutdown awaitable without abandoning its exact owner."""

    task.cancel()
    _RETAINED_STDIO_SHUTDOWN_TASKS.add(task)

    def completed(completed_task: asyncio.Task[Any]) -> None:
        _RETAINED_STDIO_SHUTDOWN_TASKS.discard(completed_task)
        _consume_task_result(completed_task)

    task.add_done_callback(completed)
