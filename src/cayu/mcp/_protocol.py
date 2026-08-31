"""Protocol-era behavior shared by MCP transports.

The 2026-07-28 revision is a different wire era, not a newer value for the
legacy initialization handshake.  This module owns only transport-neutral
request metadata and result validation; HTTP routing headers and stdio process
semantics remain in their transport modules.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, cast

from cayu._validation import (
    MAX_PORTABLE_JSON_INTEGER,
    copy_json_value,
    require_clean_nonblank,
    require_nonblank,
)
from cayu.mcp._jsonrpc import MCP_MODERN_PROTOCOL_VERSION, McpProtocolError
from cayu.mcp.base import McpInitializeResult

_CLIENT_CAPABILITIES_META_KEY = "io.modelcontextprotocol/clientCapabilities"
_CLIENT_INFO_META_KEY = "io.modelcontextprotocol/clientInfo"
_PROTOCOL_VERSION_META_KEY = "io.modelcontextprotocol/protocolVersion"
_SERVER_INFO_META_KEY = "io.modelcontextprotocol/serverInfo"

_MODERN_CACHEABLE_METHODS = frozenset(
    {
        "server/discover",
        "tools/list",
        "resources/list",
        "resources/read",
    }
)


class McpProtocolEra(StrEnum):
    """MCP wire era selected once for one transport connection."""

    LEGACY = "legacy"
    MODERN_2026_07_28 = MCP_MODERN_PROTOCOL_VERSION


class LegacyMcpWireProtocol:
    """Transport-neutral behavior for the 2025 initialization era."""

    era = McpProtocolEra.LEGACY
    establishment_method = "initialize"
    supports_legacy_listener = True
    validates_modern_results = False

    def prepare_request_params(
        self,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        del method
        return params


class ModernMcpWireProtocol:
    """Transport-neutral behavior for stateless MCP 2026-07-28 requests."""

    era = McpProtocolEra.MODERN_2026_07_28
    establishment_method = "server/discover"
    supports_legacy_listener = False
    validates_modern_results = True

    def __init__(self, *, client_name: str, client_version: str) -> None:
        self._request_meta = {
            _PROTOCOL_VERSION_META_KEY: MCP_MODERN_PROTOCOL_VERSION,
            _CLIENT_INFO_META_KEY: {
                "name": client_name,
                "version": client_version,
            },
            _CLIENT_CAPABILITIES_META_KEY: {},
        }

    def prepare_request_params(
        self,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        del method
        if "_meta" in params:
            raise McpProtocolError(
                "MCP 2026 request parameters cannot override the reserved _meta envelope."
            )
        params["_meta"] = copy_json_value(self._request_meta, "MCP 2026 request metadata")
        return params


def validate_modern_mcp_result(result: object, *, method: str) -> dict[str, Any]:
    """Validate and remove 2026 wire-only result discrimination/cache fields."""

    if type(result) is not dict:
        # Do not retain a malformed scalar result in this public traceback.
        result = None
        raise McpProtocolError(f"MCP {method} result must be an object.")
    result = cast("dict[str, Any]", result)
    protocol_error: str | None = None
    try:
        return _validate_modern_mcp_result(result, method=method)
    except McpProtocolError as error:
        protocol_error = str(error)
    finally:
        if protocol_error is not None:
            # Modern control fields bypass ordinary payload redaction so their
            # exact types can be validated. Never retain a rejected raw value in
            # the validator traceback that crosses the transport boundary.
            result.clear()
    raise McpProtocolError(protocol_error) from None


def _validate_modern_mcp_result(
    result: dict[str, Any],
    *,
    method: str,
) -> dict[str, Any]:
    if "resultType" in result and result.get("resultType") != "complete":
        raise McpProtocolError(f"MCP {method} returned an unsupported resultType.")
    result.pop("resultType", None)
    if method in _MODERN_CACHEABLE_METHODS:
        ttl_ms = result.get("ttlMs")
        if type(ttl_ms) is not int or not 0 <= ttl_ms <= MAX_PORTABLE_JSON_INTEGER:
            raise McpProtocolError(
                f"MCP {method} result ttlMs must be a non-negative portable JSON integer."
            )
        cache_scope = result.get("cacheScope")
        if type(cache_scope) is not str or cache_scope not in {"private", "public"}:
            raise McpProtocolError(f"MCP {method} result cacheScope must be 'private' or 'public'.")
        result.pop("ttlMs")
        result.pop("cacheScope")
    return result


def modern_discover_result_from_payload(payload: object) -> McpInitializeResult:
    """Normalize validated 2026 discovery into Cayu's server-metadata contract."""

    if type(payload) is not dict:
        raise McpProtocolError("MCP server/discover result must be an object.")
    payload = cast("dict[str, Any]", payload)
    parsed: McpInitializeResult | None = None
    protocol_error: str | None = None
    try:
        parsed = _modern_discover_result_from_payload(payload)
    except McpProtocolError as error:
        protocol_error = str(error)
    except (TypeError, ValueError):
        protocol_error = "MCP server/discover result contained invalid data."
    finally:
        payload.clear()
    if protocol_error is not None:
        raise McpProtocolError(protocol_error) from None
    if parsed is None:
        raise AssertionError("MCP discovery parser returned no result or error.")
    return parsed


def _modern_discover_result_from_payload(
    payload: dict[str, Any],
) -> McpInitializeResult:
    supported_versions = payload.get("supportedVersions")
    if type(supported_versions) is not list or len(supported_versions) > 64:
        raise McpProtocolError("MCP server/discover supportedVersions must be a bounded array.")
    validated_versions: list[str] = []
    for version in supported_versions:
        if type(version) is not str:
            raise McpProtocolError("MCP server/discover supportedVersions entries must be strings.")
        version = require_clean_nonblank(version, "supported protocol version")
        if len(version.encode("utf-8")) > 128:
            raise McpProtocolError(
                "MCP server/discover supported protocol version exceeded 128 bytes."
            )
        validated_versions.append(version)
    if MCP_MODERN_PROTOCOL_VERSION not in validated_versions:
        raise McpProtocolError("MCP server does not support pinned protocol version 2026-07-28.")
    capabilities = payload.get("capabilities")
    if type(capabilities) is not dict:
        raise McpProtocolError("MCP server/discover capabilities must be an object.")
    instructions = payload.get("instructions")
    if instructions is not None and type(instructions) is not str:
        raise McpProtocolError("MCP server/discover instructions must be a string.")

    server_name: str | None = None
    server_version: str | None = None
    meta = payload.get("_meta")
    server_info = meta.get(_SERVER_INFO_META_KEY) if type(meta) is dict else None
    if type(server_info) is dict:
        candidate_name = server_info.get("name")
        candidate_version = server_info.get("version")
        if type(candidate_name) is str and type(candidate_version) is str:
            try:
                server_name = require_nonblank(candidate_name, "server name")
                server_version = require_nonblank(candidate_version, "server version")
            except (TypeError, ValueError):
                server_name = None
                server_version = None
    return McpInitializeResult(
        protocol_version=MCP_MODERN_PROTOCOL_VERSION,
        server_name=server_name,
        server_version=server_version,
        instructions=instructions,
        capabilities=capabilities,
    )
