from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass
from hashlib import sha1
from typing import Any

from cayu._validation import copy_json_value, require_clean_nonblank
from cayu.core.tools import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.mcp._jsonrpc import McpProtocolError
from cayu.mcp._transport import (
    credential_safe_mcp_fatal_signal,
    credential_safe_mcp_transport_failure,
    mcp_json_value_nesting_too_deep,
)
from cayu.mcp.base import (
    McpClient,
    McpInitializeResult,
    McpServerSpec,
    McpSession,
    McpToolDefinition,
    McpToolResult,
    _close_mcp_session_after_primary_failure,
    _credential_safe_mcp_cancellation,
    _McpCallerCancellationBoundary,
    _retain_mcp_session_close_if_fenced,
    copy_mcp_server_spec,
)
from cayu.mcp.http import HttpMcpClient, HttpMcpSession
from cayu.mcp.stdio import StdioMcpClient, StdioMcpSession
from cayu.vaults import SecretRedactor

_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_UNSAFE_TOOL_NAME_CHARS_RE = re.compile(r"[^A-Za-z0-9_-]+")
_MAX_STRUCTURED_CONTENT_TEXT_BYTES = 20_000
_MAX_SERVER_INSTRUCTIONS_DESCRIPTION_CHARS = 1_000
_MAX_MCP_DISCOVERY_ERROR_BYTES = 4096


@dataclass(frozen=True, slots=True)
class _McpManifestToolEvidence:
    cayu_name: str
    mcp_name: str
    contract_hash: str


@dataclass(frozen=True, slots=True)
class _McpManifestSnapshot:
    identity_is_explicit: bool
    identity: str
    manifest_hash: str
    server_hash: str
    tools: tuple[_McpManifestToolEvidence, ...]
    tool_count: int


@dataclass(frozen=True, slots=True)
class _McpAdapterBinding:
    """Immutable construction-time authority for one MCP adapter."""

    toolset: McpToolset
    mcp_name: str
    source_contract_hash: str
    manifest_mcp_name: str
    manifest_contract_hash: str


class McpToolAdapter(Tool):
    """Expose one MCP server tool as a Cayu tool."""

    def __init__(
        self,
        *,
        toolset: McpToolset,
        definition: McpToolDefinition,
        name: str | None = None,
    ) -> None:
        if not isinstance(toolset, McpToolset):
            raise TypeError("toolset must be an McpToolset.")
        if type(definition) is not McpToolDefinition:
            raise TypeError("definition must be an McpToolDefinition.")
        binding = toolset._bind_adapter_definition(definition)
        public_definition = _redact_tool_definition(
            definition,
            redactor=toolset.secret_redactor,
        )
        tool_name = name or mcp_cayu_tool_name(
            toolset.server.name,
            public_definition.name,
        )
        if not _TOOL_NAME_RE.fullmatch(tool_name):
            raise ValueError(
                "MCP Cayu tool names must contain 1-64 letters, numbers, underscores, or hyphens."
            )
        self.__binding = binding
        self.__mcp_manifest_hash = toolset.manifest_hash
        self.__server = toolset.server
        self.__definition = public_definition
        super().__init__(
            spec=ToolSpec(
                name=tool_name,
                description=_tool_description(toolset, self.__definition),
                input_schema=self.__definition.input_schema,
                parallel_safe=_mcp_tool_parallel_safe(self.__definition),
                effect=_mcp_tool_effect(self.__definition),
            )
        )

    @property
    def _manifest_binding(self) -> _McpAdapterBinding:
        """Return the immutable dispatch binding used by runtime admission."""

        return self.__binding

    @property
    def toolset(self) -> McpToolset:
        return self.__binding.toolset

    @property
    def mcp_manifest_hash(self) -> str:
        return self.__mcp_manifest_hash

    @property
    def server(self) -> McpServerSpec:
        return self.__server.model_copy(deep=True)

    @property
    def definition(self) -> McpToolDefinition:
        return self.__definition.model_copy(deep=True)

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        if type(args) is not dict:
            raise TypeError("MCP tool arguments must be an object.")
        call = self.__binding.toolset.call_tool(
            self.__binding.mcp_name,
            args,
        )
        args = {}
        try:
            result = await call
        except BaseException:
            call = None
            raise
        call = None
        redactor = self.__binding.toolset.secret_redactor
        mcp_content = result.content
        mcp_structured_content = result.structured_content
        if redactor.has_values:
            # Redact the complete server values before rendering can truncate a
            # secret across the model-visible structured-content byte boundary.
            mcp_content = _redact_mcp_content(result.content, redactor=redactor)
            mcp_structured_content = redactor.redact_json(result.structured_content)
        content = _mcp_tool_result_text(
            mcp_content,
            structured_content=mcp_structured_content,
        )
        if redactor.has_values:
            # A hostile MCP server can echo injected secrets (secret_env/secret_headers)
            # back through its result. Keep a final text pass as defense in depth.
            content = redactor.redact_text(content)
        return ToolResult(
            content=content,
            structured={
                "mcp_server": self.__server.name,
                "mcp_tool": self.__definition.name,
                "mcp_manifest_hash": self.__mcp_manifest_hash,
                "mcp_content": mcp_content,
                "mcp_structured_content": mcp_structured_content,
            },
            is_error=result.is_error,
        )


class McpToolset:
    """Persistent initialized MCP server connection plus Cayu tool adapters."""

    def __init__(
        self,
        *,
        server: McpServerSpec,
        session: McpSession,
        definitions: tuple[McpToolDefinition, ...],
    ) -> None:
        if type(server) is not McpServerSpec:
            raise TypeError("server must be an McpServerSpec.")
        if not isinstance(session, McpSession):
            raise TypeError("session must be an McpSession.")
        self.__session = session
        redactor = session.secret_redactor
        raw_definitions = tuple(definition.model_copy(deep=True) for definition in definitions)
        raw_server = server.model_copy(deep=True)
        raw_initialize_result = session.initialize_result
        self.__binding_server = raw_server
        self.__server = _redact_server_spec(raw_server, redactor=redactor)
        self.__initialize_result = _redact_initialize_result(
            session.initialize_result,
            redactor=redactor,
        )
        self.__definitions = tuple(
            _redact_tool_definition(definition, redactor=redactor) for definition in raw_definitions
        )
        binding_manifest_hash = mcp_tool_manifest_hash(
            server=raw_server,
            initialize_result=raw_initialize_result,
            definitions=raw_definitions,
        )
        manifest_identity = mcp_tool_manifest_identity(
            server=raw_server,
        )
        binding_server_hash = mcp_tool_manifest_server_hash(
            server=raw_server,
            initialize_result=raw_initialize_result,
        )
        binding_manifest_tools = mcp_tool_manifest_tools(
            server=raw_server,
            definitions=raw_definitions,
        )
        self.__binding_snapshot = _McpManifestSnapshot(
            identity_is_explicit=raw_server.connection_id is not None,
            identity=manifest_identity,
            manifest_hash=binding_manifest_hash,
            server_hash=binding_server_hash,
            tools=tuple(
                _McpManifestToolEvidence(
                    cayu_name=entry["cayu_name"],
                    mcp_name=entry["mcp_name"],
                    contract_hash=entry["hash"],
                )
                for entry in binding_manifest_tools
            ),
            tool_count=len(raw_definitions),
        )
        source_tool_keys = [
            (entry.cayu_name, entry.mcp_name) for entry in self.__binding_snapshot.tools
        ]
        if len(source_tool_keys) != len(set(source_tool_keys)):
            raise ValueError("MCP tool definitions must not contain duplicate tools.")

        manifest_hash = mcp_tool_manifest_hash(
            server=self.__server,
            initialize_result=self.__initialize_result,
            definitions=self.__definitions,
        )
        manifest_server_hash = mcp_tool_manifest_server_hash(
            server=self.__server,
            initialize_result=self.__initialize_result,
        )
        manifest_tools = mcp_tool_manifest_tools(
            server=self.__server,
            definitions=self.__definitions,
        )
        self.__manifest_snapshot = _McpManifestSnapshot(
            identity_is_explicit=raw_server.connection_id is not None,
            identity=manifest_identity,
            manifest_hash=manifest_hash,
            server_hash=manifest_server_hash,
            tools=tuple(
                _McpManifestToolEvidence(
                    cayu_name=entry["cayu_name"],
                    mcp_name=entry["mcp_name"],
                    contract_hash=entry["hash"],
                )
                for entry in manifest_tools
            ),
            tool_count=len(self.__definitions),
        )
        self.__tools = tuple(
            McpToolAdapter(toolset=self, definition=definition) for definition in raw_definitions
        )
        _validate_unique_tool_names(list(self.__tools))

    @property
    def server(self) -> McpServerSpec:
        return self.__server.model_copy(deep=True)

    @property
    def session(self) -> McpSession:
        return self.__session

    @property
    def definitions(self) -> tuple[McpToolDefinition, ...]:
        return tuple(definition.model_copy(deep=True) for definition in self.__definitions)

    @property
    def tools(self) -> tuple[McpToolAdapter, ...]:
        return self.__tools

    def _bind_adapter_definition(self, definition: McpToolDefinition) -> _McpAdapterBinding:
        """Bind an adapter only to a definition advertised by this toolset."""

        entry = mcp_tool_manifest_tools(
            server=self.__binding_server,
            definitions=(definition.model_copy(deep=True),),
        )[0]
        matches = [
            candidate
            for candidate in self.__binding_snapshot.tools
            if candidate.mcp_name == entry["mcp_name"] and candidate.contract_hash == entry["hash"]
        ]
        if len(matches) != 1:
            raise ValueError(
                "MCP adapters must bind exactly one definition advertised by their toolset."
            )
        public_definition = _redact_tool_definition(
            definition,
            redactor=self.secret_redactor,
        )
        public_entry = mcp_tool_manifest_tools(
            server=self.__server,
            definitions=(public_definition,),
        )[0]
        public_matches = [
            candidate
            for candidate in self.__manifest_snapshot.tools
            if candidate.mcp_name == public_entry["mcp_name"]
            and candidate.contract_hash == public_entry["hash"]
        ]
        if len(public_matches) != 1:
            raise ValueError("MCP adapters must map to exactly one sanitized manifest definition.")
        return _McpAdapterBinding(
            toolset=self,
            mcp_name=matches[0].mcp_name,
            source_contract_hash=matches[0].contract_hash,
            manifest_mcp_name=public_matches[0].mcp_name,
            manifest_contract_hash=public_matches[0].contract_hash,
        )

    @property
    def _manifest_snapshot(self) -> _McpManifestSnapshot:
        """Return the immutable construction-time evidence used by runtime admission."""

        return self.__manifest_snapshot

    @property
    def manifest_identity_is_explicit(self) -> bool:
        """Whether the cached manifest identity came from an explicit connection ID."""

        return self.__manifest_snapshot.identity_is_explicit

    @property
    def manifest_identity(self) -> str:
        return self.__manifest_snapshot.identity

    @property
    def manifest_hash(self) -> str:
        return self.__manifest_snapshot.manifest_hash

    @property
    def manifest_server_hash(self) -> str:
        return self.__manifest_snapshot.server_hash

    @property
    def manifest_tools(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "cayu_name": entry.cayu_name,
                "mcp_name": entry.mcp_name,
                "hash": entry.contract_hash,
            }
            for entry in self.__manifest_snapshot.tools
        )

    @classmethod
    async def connect(
        cls,
        server: McpServerSpec,
        *,
        client: McpClient | None = None,
    ) -> McpToolset:
        authoritative_server = copy_mcp_server_spec(server)
        mcp_client = client if client is not None else _default_client_for(authoritative_server)
        session = await mcp_client.connect(copy_mcp_server_spec(authoritative_server))
        sanitized_error: BaseException | None = None
        discovery_cancellation_boundary = _McpCallerCancellationBoundary()
        try:
            await discovery_cancellation_boundary.checkpoint()
            definitions = await session.list_tools()
            return cls(
                server=authoritative_server,
                session=session,
                definitions=definitions,
            )
        except asyncio.CancelledError as exc:
            if not discovery_cancellation_boundary.caller_cancelled():
                public_error = credential_safe_mcp_transport_failure(
                    exc,
                    redactor=session.secret_redactor,
                    context="MCP tool discovery was cancelled unexpectedly",
                    max_message_bytes=_MAX_MCP_DISCOVERY_ERROR_BYTES,
                    preserve_cause=True,
                )
                if not _retain_mcp_session_close_if_fenced(
                    session,
                    primary_error=public_error,
                ):
                    cleanup_cancellation = await _close_mcp_session_after_primary_failure(
                        session,
                        primary_error=public_error,
                        primary_context="MCP tool discovery failed",
                        cleanup_context="MCP tool discovery cleanup failed",
                    )
                    if cleanup_cancellation is not None:
                        sanitized_error = cleanup_cancellation
                if sanitized_error is None:
                    sanitized_error = public_error
                definitions = ()
            else:
                public_cancellation = _credential_safe_mcp_cancellation(
                    exc,
                    redactor=session.secret_redactor,
                )
                if not _retain_mcp_session_close_if_fenced(
                    session,
                    primary_error=public_cancellation,
                ):
                    cleanup_cancellation = await _close_mcp_session_after_primary_failure(
                        session,
                        primary_error=public_cancellation,
                        primary_context="MCP tool discovery was cancelled",
                        cleanup_context="MCP tool discovery cleanup failed",
                    )
                    if cleanup_cancellation is not None:
                        sanitized_error = cleanup_cancellation
                        definitions = ()
                    else:
                        sanitized_error = public_cancellation
                        definitions = ()
                else:
                    sanitized_error = public_cancellation
                    definitions = ()
        except TimeoutError as exc:
            public_error: BaseException = exc
            if session.secret_redactor.has_values:
                public_error = credential_safe_mcp_transport_failure(
                    exc,
                    redactor=session.secret_redactor,
                    context="MCP tool discovery failed",
                    max_message_bytes=_MAX_MCP_DISCOVERY_ERROR_BYTES,
                    preserve_cause=True,
                )
            if not _retain_mcp_session_close_if_fenced(
                session,
                primary_error=public_error,
            ):
                cleanup_cancellation = await _close_mcp_session_after_primary_failure(
                    session,
                    primary_error=public_error,
                    primary_context="MCP tool discovery timed out",
                    cleanup_context="MCP tool discovery cleanup failed",
                )
                if cleanup_cancellation is not None:
                    sanitized_error = cleanup_cancellation
                    definitions = ()
            if sanitized_error is None:
                if public_error is exc:
                    raise
                sanitized_error = public_error
                definitions = ()
        except (BaseExceptionGroup, Exception) as exc:
            cleanup_cancellation = await _close_mcp_session_after_primary_failure(
                session,
                primary_error=exc,
                primary_context="MCP tool discovery failed",
                cleanup_context="MCP tool discovery cleanup failed",
            )
            if cleanup_cancellation is not None:
                sanitized_error = cleanup_cancellation
                definitions = ()
            elif not session.secret_redactor.has_values:
                raise
            else:
                sanitized_error = credential_safe_mcp_transport_failure(
                    exc,
                    redactor=session.secret_redactor,
                    context="MCP tool discovery failed",
                    max_message_bytes=_MAX_MCP_DISCOVERY_ERROR_BYTES,
                    preserve_cause=True,
                )
                definitions = ()
        except BaseException as fatal:
            cleanup_cancellation = await _close_mcp_session_after_primary_failure(
                session,
                primary_error=fatal,
                primary_context="MCP tool discovery failed",
                cleanup_context="MCP tool discovery cleanup failed",
            )
            if cleanup_cancellation is not None:
                sanitized_error = cleanup_cancellation
            else:
                sanitized_error = credential_safe_mcp_fatal_signal(
                    fatal,
                    redactor=session.secret_redactor,
                    context="MCP tool discovery failed",
                    max_message_bytes=_MAX_MCP_DISCOVERY_ERROR_BYTES,
                )
            definitions = ()
        if sanitized_error is not None:
            raise sanitized_error
        raise AssertionError("MCP tool discovery returned without a toolset or error.")

    @property
    def initialize_result(self) -> McpInitializeResult:
        return self.__initialize_result.model_copy(deep=True)

    @property
    def secret_redactor(self) -> SecretRedactor:
        """Redactor for secrets injected into this server's session (empty if none)."""
        return self.__session.secret_redactor

    @property
    def process_capability_evidence(self):
        """Configured process-lifecycle evidence for stdio-backed toolsets."""

        if isinstance(self.__session, StdioMcpSession):
            return self.__session.process_capability_evidence
        return None

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> McpToolResult:
        # Built-in sessions own a bounded preflight before their defensive copy.
        # Third-party sessions have no common limits contract, so retain the
        # historical toolset-owned copy before handing arguments to extensions.
        call_name = name
        call_arguments = arguments
        name = ""
        arguments = {}
        preparation_error: BaseException | None = None
        if type(self.__session) not in {HttpMcpSession, StdioMcpSession}:
            # Built-in sessions combine nesting and byte validation in their
            # bounded transport preflight. Custom sessions have no shared size
            # contract, so retain the adapter's historical validation boundary.
            if mcp_json_value_nesting_too_deep(call_arguments):
                call_name = ""
                call_arguments = {}
                raise McpProtocolError(
                    "MCP tool arguments exceeded the supported JSON nesting."
                ) from None
            try:
                call_arguments = copy_json_value(call_arguments, "arguments")
            except (RecursionError, TypeError, ValueError) as error:
                preparation_error = credential_safe_mcp_transport_failure(
                    error,
                    redactor=self.secret_redactor,
                    context="MCP tool arguments were invalid",
                )
                error = None
            if preparation_error is not None:
                call_name = ""
                call_arguments = {}
                raise preparation_error from None
            if type(call_arguments) is not dict:
                call_name = ""
                call_arguments = {}
                raise TypeError("MCP tool arguments must be an object.")
        call = self.__session.call_tool(call_name, call_arguments)
        call_name = ""
        call_arguments = {}
        try:
            return await call
        except BaseException:
            call = None
            raise
        finally:
            call = None

    async def close(self) -> None:
        await self.__session.close()


async def connect_mcp_toolset(
    server: McpServerSpec,
    *,
    client: McpClient | None = None,
) -> McpToolset:
    """Connect to one MCP server and return its initialized toolset."""

    return await McpToolset.connect(server, client=client)


def _redact_server_spec(
    server: McpServerSpec,
    *,
    redactor: SecretRedactor,
) -> McpServerSpec:
    payload = redactor.redact_json_values(server.model_dump(mode="json"))
    if type(payload) is not dict:
        raise AssertionError("MCP server redaction returned a non-object.")
    return McpServerSpec(**payload)


def _redact_initialize_result(
    result: McpInitializeResult,
    *,
    redactor: SecretRedactor,
) -> McpInitializeResult:
    payload = redactor.redact_json_values(
        result.model_dump(mode="json"),
        preserve_string_fields={"protocol_version"},
    )
    if type(payload) is not dict:
        raise AssertionError("MCP initialize-result redaction returned a non-object.")
    return McpInitializeResult(**payload)


def _redact_tool_definition(
    definition: McpToolDefinition,
    *,
    redactor: SecretRedactor,
) -> McpToolDefinition:
    payload = redactor.redact_json_values(definition.model_dump(mode="json"))
    if type(payload) is not dict:
        raise AssertionError("MCP tool-definition redaction returned a non-object.")
    return McpToolDefinition(**payload)


def _default_client_for(server: McpServerSpec) -> McpClient:
    """Pick the transport from the spec: a URL server uses HTTP, a command server stdio."""
    if server.url is not None:
        return HttpMcpClient()
    return StdioMcpClient()


def mcp_cayu_tool_name(server_name: str, tool_name: str) -> str:
    server_slug = _tool_name_slug(server_name, "server_name")
    tool_slug = _tool_name_slug(tool_name, "tool_name")
    candidate = f"mcp__{server_slug}__{tool_slug}"
    if len(candidate) <= 64:
        return candidate
    digest = sha1(candidate.encode("utf-8")).hexdigest()[:10]
    budget = 64 - len("mcp__") - len("__") - len("_") - len(digest)
    server_budget = max(8, budget // 3)
    tool_budget = max(8, budget - server_budget)
    return f"mcp__{server_slug[:server_budget]}__{tool_slug[:tool_budget]}_{digest}"


def mcp_tool_manifest_hash(
    *,
    server: McpServerSpec,
    initialize_result: McpInitializeResult,
    definitions: tuple[McpToolDefinition, ...],
) -> str:
    """Return a stable hash of the MCP tool contract Cayu exposes."""

    if type(server) is not McpServerSpec:
        raise TypeError("server must be an McpServerSpec.")
    if type(initialize_result) is not McpInitializeResult:
        raise TypeError("initialize_result must be an McpInitializeResult.")
    if not isinstance(definitions, tuple):
        raise TypeError("definitions must be a tuple of McpToolDefinition instances.")
    payload = _mcp_tool_manifest_payload(
        server=server,
        initialize_result=initialize_result,
        definitions=definitions,
    )
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def mcp_tool_manifest_identity(
    *,
    server: McpServerSpec,
    definitions: tuple[McpToolDefinition, ...] | None = None,
) -> str:
    """Return an opaque candidate identity for one MCP server connection.

    ``definitions`` remains accepted for source compatibility with the original
    helper, but manifest contents intentionally do not participate in identity.
    Only an explicit ``McpServerSpec.connection_id`` is authoritative for
    runtime manifest history. The server-name form exists solely so a rejected
    ID-less toolset can carry bounded, non-identifying audit evidence.
    """

    if type(server) is not McpServerSpec:
        raise TypeError("server must be an McpServerSpec.")
    if definitions is not None:
        if not isinstance(definitions, tuple):
            raise TypeError("definitions must be a tuple of McpToolDefinition instances.")
        if any(type(definition) is not McpToolDefinition for definition in definitions):
            raise TypeError("definitions must contain McpToolDefinition instances.")
    payload = {
        "schema": "cayu.mcp.connection_identity.v1",
        # Domain-separate authoritative explicit identities from the audit-only
        # server-name candidate. An explicit ID equal to the presentation name
        # must still establish a genuinely new authorization namespace.
        "identity_kind": ("connection_id" if server.connection_id is not None else "server_name"),
        "connection_id": (
            server.connection_id if server.connection_id is not None else server.name
        ),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def mcp_tool_manifest_tools(
    *,
    server: McpServerSpec,
    definitions: tuple[McpToolDefinition, ...],
) -> tuple[dict[str, Any], ...]:
    """Return compact per-tool manifest entries for drift auditing."""

    if type(server) is not McpServerSpec:
        raise TypeError("server must be an McpServerSpec.")
    if not isinstance(definitions, tuple):
        raise TypeError("definitions must be a tuple of McpToolDefinition instances.")
    entries: list[dict[str, Any]] = []
    for definition in definitions:
        if type(definition) is not McpToolDefinition:
            raise TypeError("definitions must contain McpToolDefinition instances.")
        cayu_name = mcp_cayu_tool_name(server.name, definition.name)
        payload = {
            "cayu_name": cayu_name,
            "mcp_name": definition.name,
            "description": definition.description,
            "input_schema": definition.input_schema,
            "annotations": definition.annotations,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        entries.append(
            {
                "cayu_name": cayu_name,
                "mcp_name": definition.name,
                "hash": f"sha256:{hashlib.sha256(encoded).hexdigest()}",
            }
        )
    entries.sort(key=lambda entry: (entry["cayu_name"], entry["mcp_name"]))
    return tuple(entries)


def mcp_tool_manifest_server_hash(
    *,
    server: McpServerSpec,
    initialize_result: McpInitializeResult,
) -> str:
    """Return a stable hash of MCP server metadata that affects the manifest."""

    if type(server) is not McpServerSpec:
        raise TypeError("server must be an McpServerSpec.")
    if type(initialize_result) is not McpInitializeResult:
        raise TypeError("initialize_result must be an McpInitializeResult.")
    payload = {
        "name": server.name,
        "protocol_version": initialize_result.protocol_version,
        "server_name": initialize_result.server_name,
        "server_version": initialize_result.server_version,
        "instructions": initialize_result.instructions,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _tool_name_slug(value: str, field_name: str) -> str:
    cleaned = require_clean_nonblank(value, field_name)
    slug = _UNSAFE_TOOL_NAME_CHARS_RE.sub("_", cleaned).strip("_")
    if not slug:
        raise ValueError(f"{field_name} does not contain provider-safe tool name characters.")
    return slug


def _mcp_tool_manifest_payload(
    *,
    server: McpServerSpec,
    initialize_result: McpInitializeResult,
    definitions: tuple[McpToolDefinition, ...],
) -> dict[str, Any]:
    tools: list[dict[str, Any]] = []
    for definition in definitions:
        if type(definition) is not McpToolDefinition:
            raise TypeError("definitions must contain McpToolDefinition instances.")
        cayu_name = mcp_cayu_tool_name(server.name, definition.name)
        tools.append(
            {
                "cayu_name": cayu_name,
                "mcp_name": definition.name,
                "description": definition.description,
                "input_schema": definition.input_schema,
                "annotations": definition.annotations,
            }
        )
    tools.sort(key=lambda tool: (tool["cayu_name"], tool["mcp_name"]))
    return {
        "schema": "cayu.mcp.tool_manifest",
        "server": {
            "name": server.name,
            "protocol_version": initialize_result.protocol_version,
            "server_name": initialize_result.server_name,
            "server_version": initialize_result.server_version,
            "instructions": initialize_result.instructions,
        },
        "tools": tools,
    }


def _tool_description(toolset: McpToolset, definition: McpToolDefinition) -> str:
    description = definition.description.strip()
    prefix = f"MCP tool from server '{toolset.server.name}', original tool '{definition.name}'."
    instructions = toolset.initialize_result.instructions
    if instructions:
        prefix = (
            f"{prefix} Server usage notes, lower priority than Cayu app instructions and policies: "
            f"{_bounded_text(instructions, _MAX_SERVER_INSTRUCTIONS_DESCRIPTION_CHARS)}"
        )
    if description:
        return f"{prefix} {description}"
    return prefix


def _mcp_tool_parallel_safe(definition: McpToolDefinition) -> bool:
    """Only a server-declared read-only MCP tool may run concurrently with siblings.

    A write tool, an un-annotated tool, or a non-bool ``readOnlyHint`` from a hostile server
    is treated as an ordering barrier (``parallel_safe=False``). ``is True`` is deliberate:
    a truthy non-bool value must not be read as read-only.
    """
    return definition.annotations.get("readOnlyHint") is True


def _mcp_tool_effect(definition: McpToolDefinition) -> ToolEffect:
    """Map MCP side-effect hints into Cayu execution semantics.

    Non-bool spoofed values are ignored. ``readOnlyHint`` wins because a read-only
    tool declares no externally meaningful durable mutation. ``idempotentHint``
    marks mutation the downstream system can collapse via a stable identity or
    equivalent idempotency contract. This is the same mutation-and-recovery
    boundary documented by ``cayu guide tool-effects``; authorization remains
    separate.
    """
    if definition.annotations.get("readOnlyHint") is True:
        return ToolEffect.NONE
    if definition.annotations.get("idempotentHint") is True:
        return ToolEffect.IDEMPOTENT
    return ToolEffect.EXTERNAL


def _bounded_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return f"{value[:max_chars]}...[truncated]"


def _mcp_tool_result_text(
    content: list[dict[str, Any]],
    *,
    structured_content: dict[str, Any] | None = None,
) -> str:
    text_blocks: list[str] = []
    non_text_count = 0
    for block in content:
        if type(block) is not dict:
            non_text_count += 1
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            text_blocks.append(block["text"])
        else:
            non_text_count += 1
    result = "\n\n".join(text_blocks).strip()
    structured_text = _structured_content_text(structured_content)
    if structured_text:
        result = f"{result}\n\n{structured_text}".strip() if result else structured_text
    if non_text_count:
        note = f"[MCP returned {non_text_count} non-text content block(s).]"
        result = f"{result}\n\n{note}".strip() if result else note
    return result


def _redact_mcp_content(
    content: list[dict[str, Any]],
    *,
    redactor: SecretRedactor,
) -> list[dict[str, Any]]:
    """Redact MCP blocks while preserving the typed text envelope used for rendering."""

    redacted: list[dict[str, Any]] = []
    for block in content:
        if type(block) is not dict:
            raise AssertionError("MCP tool content must contain objects.")
        block_type = block.get("type")
        text = block.get("text")
        if block_type == "text" and type(text) is str:
            untrusted = {key: value for key, value in block.items() if key not in {"type", "text"}}
            redacted_untrusted = redactor.redact_json(untrusted)
            if type(redacted_untrusted) is not dict:
                raise AssertionError("MCP text block redaction returned a non-object.")
            redacted.append(
                {
                    "type": "text",
                    "text": redactor.redact_text(text),
                    **redacted_untrusted,
                }
            )
            continue
        redacted_block = redactor.redact_json(block)
        if type(redacted_block) is not dict:
            raise AssertionError("MCP content block redaction returned a non-object.")
        redacted.append(redacted_block)
    return redacted


def _structured_content_text(structured_content: dict[str, Any] | None) -> str:
    if structured_content is None:
        return ""
    encoded = json.dumps(
        structured_content,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )
    data = encoded.encode("utf-8")
    if len(data) <= _MAX_STRUCTURED_CONTENT_TEXT_BYTES:
        return f"Structured MCP content:\n{encoded}"
    truncated = data[:_MAX_STRUCTURED_CONTENT_TEXT_BYTES].decode("utf-8", errors="replace")
    return f"Structured MCP content:\n{truncated}\n\n[structured content truncated]"


def _validate_unique_tool_names(adapters: list[McpToolAdapter]) -> None:
    names = [adapter.name for adapter in adapters]
    if len(names) != len(set(names)):
        raise ValueError("Discovered MCP tools produced duplicate Cayu tool names.")
