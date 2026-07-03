from __future__ import annotations

import asyncio
import hashlib
import json
import re
from hashlib import sha1
from typing import Any

from cayu._validation import copy_json_value, require_clean_nonblank
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.mcp.base import (
    McpClient,
    McpInitializeResult,
    McpServerSpec,
    McpSession,
    McpToolDefinition,
    McpToolResult,
)
from cayu.mcp.http import HttpMcpClient
from cayu.mcp.stdio import StdioMcpClient
from cayu.vaults import SecretRedactor

_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_UNSAFE_TOOL_NAME_CHARS_RE = re.compile(r"[^A-Za-z0-9_-]+")
_MAX_STRUCTURED_CONTENT_TEXT_BYTES = 20_000
_MAX_SERVER_INSTRUCTIONS_DESCRIPTION_CHARS = 1_000


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
        tool_name = name or mcp_cayu_tool_name(toolset.server.name, definition.name)
        if not _TOOL_NAME_RE.fullmatch(tool_name):
            raise ValueError(
                "MCP Cayu tool names must contain 1-64 letters, numbers, underscores, or hyphens."
            )
        self.toolset = toolset
        self.mcp_manifest_hash = toolset.manifest_hash
        self.server = toolset.server.model_copy(deep=True)
        self.definition = definition.model_copy(deep=True)
        super().__init__(
            spec=ToolSpec(
                name=tool_name,
                description=_tool_description(toolset, definition),
                input_schema=definition.input_schema,
            )
        )

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        if type(args) is not dict:
            raise TypeError("MCP tool arguments must be an object.")
        arguments = copy_json_value(args, "arguments")
        if type(arguments) is not dict:
            raise TypeError("MCP tool arguments must be an object.")
        result = await self.toolset.call_tool(self.definition.name, arguments)
        content = _mcp_tool_result_text(
            result.content,
            structured_content=result.structured_content,
        )
        mcp_content = result.content
        mcp_structured_content = result.structured_content
        redactor = self.toolset.secret_redactor
        if redactor.has_values:
            # A hostile MCP server can echo injected secrets (secret_env/secret_headers)
            # back through its result; scrub the rendered text AND the raw content/
            # structured echoes before they reach model-visible context.
            content = redactor.redact_text(content)
            mcp_content = redactor.redact_json(result.content)
            mcp_structured_content = redactor.redact_json(result.structured_content)
        return ToolResult(
            content=content,
            structured={
                "mcp_server": self.server.name,
                "mcp_tool": self.definition.name,
                "mcp_manifest_hash": self.mcp_manifest_hash,
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
        self.server = server.model_copy(deep=True)
        self.session = session
        self.definitions = tuple(definition.model_copy(deep=True) for definition in definitions)
        self.manifest_hash = mcp_tool_manifest_hash(
            server=self.server,
            initialize_result=self.initialize_result,
            definitions=self.definitions,
        )
        self.manifest_identity = mcp_tool_manifest_identity(
            server=self.server,
            definitions=self.definitions,
        )
        self.manifest_server_hash = mcp_tool_manifest_server_hash(
            server=self.server,
            initialize_result=self.initialize_result,
        )
        self.manifest_tools = mcp_tool_manifest_tools(
            server=self.server,
            definitions=self.definitions,
        )
        self.tools = tuple(
            McpToolAdapter(toolset=self, definition=definition) for definition in self.definitions
        )
        _validate_unique_tool_names(list(self.tools))

    @classmethod
    async def connect(
        cls,
        server: McpServerSpec,
        *,
        client: McpClient | None = None,
    ) -> McpToolset:
        if type(server) is not McpServerSpec:
            raise TypeError("server must be an McpServerSpec.")
        mcp_client = client if client is not None else _default_client_for(server)
        session = await mcp_client.connect(server)
        try:
            definitions = await session.list_tools()
            return cls(server=server, session=session, definitions=definitions)
        except asyncio.CancelledError:
            await _close_session_after_failed_toolset_connect(session)
            raise
        except Exception:
            await _close_session_after_failed_toolset_connect(session)
            raise

    @property
    def initialize_result(self) -> McpInitializeResult:
        return self.session.initialize_result

    @property
    def secret_redactor(self) -> SecretRedactor:
        """Redactor for secrets injected into this server's session (empty if none)."""
        return self.session.secret_redactor

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> McpToolResult:
        return await self.session.call_tool(name, arguments)

    async def close(self) -> None:
        await self.session.close()


async def connect_mcp_toolset(
    server: McpServerSpec,
    *,
    client: McpClient | None = None,
) -> McpToolset:
    """Connect to one MCP server and return its initialized toolset."""

    return await McpToolset.connect(server, client=client)


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
    definitions: tuple[McpToolDefinition, ...],
) -> str:
    """Return a stable identity for comparing one exposed MCP toolset over time."""

    if type(server) is not McpServerSpec:
        raise TypeError("server must be an McpServerSpec.")
    if not isinstance(definitions, tuple):
        raise TypeError("definitions must be a tuple of McpToolDefinition instances.")
    cayu_names: list[str] = []
    for definition in definitions:
        if type(definition) is not McpToolDefinition:
            raise TypeError("definitions must contain McpToolDefinition instances.")
        cayu_names.append(mcp_cayu_tool_name(server.name, definition.name))
    payload = {
        "server_name": server.name,
        "cayu_tool_names": sorted(cayu_names),
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


async def _close_session_after_failed_toolset_connect(session: McpSession) -> None:
    close_task = asyncio.create_task(session.close())
    while True:
        try:
            await asyncio.shield(close_task)
            return
        except Exception:
            return
        except asyncio.CancelledError:
            if close_task.done():
                return
