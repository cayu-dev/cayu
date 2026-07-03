"""Transport-agnostic MCP JSON-RPC helpers.

Shared by every `McpClient`/`McpSession` transport (stdio, HTTP, …) so the
JSON-RPC framing and the JSON->model parsing stay identical across transports.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any, cast

from cayu._validation import copy_json_value, require_clean_nonblank, require_nonblank
from cayu.mcp.base import (
    McpInitializeResult,
    McpResourceDefinition,
    McpToolDefinition,
    McpToolResult,
)

# The protocol version cayu advertises in the initialize request (its preferred,
# latest-known revision).
MCP_PROTOCOL_VERSION = "2025-06-18"
# Every protocol revision cayu can speak. A server negotiates by echoing one of
# these in its initialize response; anything outside the set is rejected. Older
# servers that pin an earlier revision are accepted rather than refused outright.
SUPPORTED_MCP_PROTOCOL_VERSIONS = frozenset(
    {
        "2024-11-05",
        "2025-03-26",
        "2025-06-18",
    }
)
DEFAULT_MCP_REQUEST_TIMEOUT_S = 30.0
DEFAULT_MCP_CLIENT_NAME = "cayu"
DEFAULT_MCP_CLIENT_VERSION = "0.1.0"
JSONRPC_METHOD_NOT_FOUND = -32601


class McpProtocolError(RuntimeError):
    """Raised when an MCP server violates the expected JSON-RPC contract."""


def validate_positive_number(value: float, field_name: str) -> float:
    if type(value) not in {float, int}:
        raise TypeError(f"{field_name} must be a number.")
    numeric = float(value)
    if numeric <= 0:
        raise ValueError(f"{field_name} must be greater than zero.")
    return numeric


def initialize_params(client_name: str, client_version: str) -> dict[str, Any]:
    return {
        "protocolVersion": MCP_PROTOCOL_VERSION,
        "capabilities": {},
        "clientInfo": {"name": client_name, "version": client_version},
    }


def jsonrpc_request_payload(request_id: int, method: str, params: dict[str, Any]) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": copy_json_value(params, "params"),
    }


def jsonrpc_notification_payload(method: str, params: dict[str, Any]) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "method": method,
        "params": copy_json_value(params, "params"),
    }


def result_from_jsonrpc_response(response: dict[str, Any], method: str) -> Any:
    """Return the JSON-RPC ``result`` or raise on an ``error``/missing result."""
    if "error" in response:
        error = response["error"]
        if isinstance(error, Mapping):
            message = error.get("message", "MCP request failed.")
            raise McpProtocolError(f"MCP {method} failed: {message}")
        raise McpProtocolError(f"MCP {method} failed.")
    if "result" not in response:
        raise McpProtocolError(f"MCP {method} response missing result.")
    return copy_json_value(response["result"], "result")


def validate_negotiated_protocol_version(version: str) -> None:
    """Raise if the server negotiated a protocol revision cayu cannot speak."""
    if version not in SUPPORTED_MCP_PROTOCOL_VERSIONS:
        raise McpProtocolError(f"MCP server negotiated unsupported protocol version {version!r}.")


async def collect_paginated(
    request: Callable[[str, dict[str, Any]], Awaitable[Any]],
    method: str,
    item_key: str,
) -> list[Any]:
    """Drain a paginated list request, following ``nextCursor`` across pages.

    A server may return a page of items plus an opaque ``nextCursor``; the next
    page is fetched by echoing that cursor. Without this the first page would be
    silently treated as the complete list — truncating tool/resource manifests
    (and their hashes) into a stable-looking but incomplete view.
    """
    items: list[Any] = []
    cursor: str | None = None
    seen_cursors: set[str] = set()
    while True:
        params: dict[str, Any] = {} if cursor is None else {"cursor": cursor}
        result = await request(method, params)
        if type(result) is not dict:
            raise McpProtocolError(f"MCP {method} result must be an object.")
        page = result.get(item_key, [])
        if not isinstance(page, list):
            raise McpProtocolError(f"MCP {method} result {item_key} must be a list.")
        items.extend(page)
        next_cursor = result.get("nextCursor")
        if next_cursor is None:
            return items
        if not isinstance(next_cursor, str):
            raise McpProtocolError(f"MCP {method} nextCursor must be a string.")
        cursor = require_nonblank(next_cursor, "nextCursor")
        if cursor in seen_cursors:
            raise McpProtocolError(
                f"MCP {method} repeated pagination cursor {cursor!r}; refusing to loop."
            )
        seen_cursors.add(cursor)


def initialize_result_from_payload(payload: dict[str, Any]) -> McpInitializeResult:
    protocol_version = payload.get("protocolVersion")
    if not isinstance(protocol_version, str):
        raise McpProtocolError("MCP initialize protocolVersion must be a string.")
    capabilities = payload.get("capabilities", {})
    if type(capabilities) is not dict:
        raise McpProtocolError("MCP initialize capabilities must be an object.")
    server_info = payload.get("serverInfo", {})
    if server_info is None:
        server_info = {}
    if type(server_info) is not dict:
        raise McpProtocolError("MCP initialize serverInfo must be an object.")
    instructions = payload.get("instructions")
    if instructions is not None and type(instructions) is not str:
        raise McpProtocolError("MCP initialize instructions must be a string.")
    return McpInitializeResult(
        protocol_version=protocol_version,
        server_name=optional_mapping_string(server_info, "name"),
        server_version=optional_mapping_string(server_info, "version"),
        instructions=instructions,
        capabilities=capabilities,
    )


def tool_definition_from_payload(payload: object, server_name: str) -> McpToolDefinition:
    if type(payload) is not dict:
        raise McpProtocolError("MCP tool definitions must be objects.")
    payload = cast("dict[str, Any]", payload)
    name = mapping_string(payload, "name")
    description = optional_mapping_string(payload, "description") or ""
    input_schema = payload.get("inputSchema", {})
    if type(input_schema) is not dict:
        raise McpProtocolError("MCP tool inputSchema must be an object.")
    annotations = payload.get("annotations", {})
    if type(annotations) is not dict:
        raise McpProtocolError("MCP tool annotations must be an object.")
    return McpToolDefinition(
        name=name,
        description=description,
        input_schema=input_schema,
        annotations={
            **annotations,
            "mcp_server": server_name,
        },
    )


def tool_result_from_payload(payload: dict[str, Any]) -> McpToolResult:
    content = payload.get("content", [])
    if not isinstance(content, list):
        raise McpProtocolError("MCP tool result content must be a list.")
    structured_content = payload.get("structuredContent")
    if structured_content is not None and type(structured_content) is not dict:
        raise McpProtocolError("MCP structuredContent must be an object.")
    is_error = payload.get("isError", False)
    if type(is_error) is not bool:
        raise McpProtocolError("MCP tool result isError must be a bool.")
    return McpToolResult(
        content=content,
        structured_content=structured_content,
        is_error=is_error,
    )


def resource_definition_from_payload(payload: object, server_name: str) -> McpResourceDefinition:
    if type(payload) is not dict:
        raise McpProtocolError("MCP resource definitions must be objects.")
    payload = cast("dict[str, Any]", payload)
    uri = mapping_string(payload, "uri")
    metadata = {
        "mcp_server": server_name,
    }
    return McpResourceDefinition(
        uri=uri,
        name=optional_mapping_string(payload, "name"),
        description=optional_mapping_string(payload, "description"),
        mime_type=optional_mapping_string(payload, "mimeType"),
        metadata=metadata,
    )


def mapping_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise McpProtocolError(f"MCP {key} must be a string.")
    return require_clean_nonblank(value, key)


def optional_mapping_string(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise McpProtocolError(f"MCP {key} must be a string.")
    return require_nonblank(value, key)
