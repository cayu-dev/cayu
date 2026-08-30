"""Wire-specific helpers for MCP over Streamable HTTP.

The 2026-07-28 revision is a different protocol era, not a newer value for the
legacy initialization handshake.  Keeping its request metadata, routing
headers, and tool-header authority here prevents those rules from leaking into
catalogue admission and tool execution.
"""

from __future__ import annotations

import base64
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
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

MCP_PROTOCOL_VERSION_HEADER = "mcp-protocol-version"
MCP_SESSION_ID_HEADER = "mcp-session-id"
MCP_METHOD_HEADER = "mcp-method"
MCP_NAME_HEADER = "mcp-name"
MCP_PARAMETER_HEADER_PREFIX = "mcp-param-"

MAX_MCP_HTTP_MIRRORED_HEADERS_PER_TOOL = 64
MAX_MCP_HTTP_HEADER_NAME_BYTES = 256
MAX_MCP_HTTP_HEADER_VALUE_BYTES = 8_192
MAX_MCP_HTTP_MIRRORED_HEADER_BYTES = 65_536

_CLIENT_CAPABILITIES_META_KEY = "io.modelcontextprotocol/clientCapabilities"
_CLIENT_INFO_META_KEY = "io.modelcontextprotocol/clientInfo"
_PROTOCOL_VERSION_META_KEY = "io.modelcontextprotocol/protocolVersion"
_SERVER_INFO_META_KEY = "io.modelcontextprotocol/serverInfo"
_BASE64_SENTINEL_PREFIX = "=?base64?"
_BASE64_SENTINEL_SUFFIX = "?="
_HTTP_TOKEN = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_NON_REACHABLE_SCHEMA_KEYWORDS = (
    "items",
    "prefixItems",
    "contains",
    "additionalProperties",
    "unevaluatedProperties",
    "unevaluatedItems",
    "propertyNames",
    "patternProperties",
    "dependentSchemas",
    "oneOf",
    "anyOf",
    "allOf",
    "not",
    "if",
    "then",
    "else",
    "$defs",
    "definitions",
)
_OBJECT_VALUED_SCHEMA_KEYWORDS = frozenset(
    {
        "patternProperties",
        "dependentSchemas",
        "$defs",
        "definitions",
    }
)


class McpProtocolEra(StrEnum):
    """MCP wire era selected for one transport connection."""

    LEGACY = "legacy"
    MODERN_2026_07_28 = MCP_MODERN_PROTOCOL_VERSION


@dataclass(frozen=True, slots=True)
class McpHttpToolHeaderBinding:
    """Private routing-header authority derived from one admitted tool schema."""

    argument_path: tuple[str, ...]
    header_name: str
    value_type: str


McpHttpToolHeaderContract = tuple[McpHttpToolHeaderBinding, ...]

_MODERN_CACHEABLE_METHODS = frozenset(
    {
        "server/discover",
        "tools/list",
        "resources/list",
        "resources/read",
    }
)


class LegacyHttpMcpWireProtocol:
    """2025-era initialized/session-oriented Streamable HTTP codec."""

    era = McpProtocolEra.LEGACY
    establishment_method = "initialize"
    supports_legacy_listener = True
    uses_protocol_sessions = True
    validates_modern_results = False

    def prepare_request_params(
        self,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        del method
        return params

    def request_headers(
        self,
        payload: Mapping[str, Any],
        *,
        initialized_protocol_version: str | None,
        negotiated_protocol_version: str | None,
        session_id: str | None,
        protocol_version_override: str | None = None,
        session_id_override: str | None = None,
        mirrored_headers: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        del payload, mirrored_headers
        headers = {"accept-encoding": "identity"}
        protocol_version = (
            protocol_version_override or initialized_protocol_version or negotiated_protocol_version
        )
        if protocol_version is not None:
            headers[MCP_PROTOCOL_VERSION_HEADER] = protocol_version
        resolved_session_id = session_id if session_id_override is None else session_id_override
        if resolved_session_id is not None:
            headers[MCP_SESSION_ID_HEADER] = resolved_session_id
        return headers

    def response_session_id(self, headers: Mapping[str, str]) -> str | None:
        return headers.get(MCP_SESSION_ID_HEADER)


class ModernHttpMcpWireProtocol:
    """Stateless MCP 2026-07-28 Streamable HTTP codec."""

    era = McpProtocolEra.MODERN_2026_07_28
    establishment_method = "server/discover"
    supports_legacy_listener = False
    uses_protocol_sessions = False
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
        if "_meta" in params:
            raise McpProtocolError(
                "MCP 2026 request parameters cannot override the reserved _meta envelope."
            )
        if not _plain_http_header_value(method):
            raise McpProtocolError("MCP 2026 request method was not HTTP-header safe.")
        name_field = _modern_request_name_field(method)
        if name_field is not None:
            name = params.get(name_field)
            if type(name) is not str:
                raise McpProtocolError(f"MCP 2026 {method} params.{name_field} must be a string.")
            encode_mcp_http_header_value(name, field_name="Mcp-Name")
        params["_meta"] = copy_json_value(self._request_meta, "MCP 2026 request metadata")
        return params

    def request_headers(
        self,
        payload: Mapping[str, Any],
        *,
        initialized_protocol_version: str | None,
        negotiated_protocol_version: str | None,
        session_id: str | None,
        protocol_version_override: str | None = None,
        session_id_override: str | None = None,
        mirrored_headers: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        del (
            initialized_protocol_version,
            negotiated_protocol_version,
            session_id,
            protocol_version_override,
            session_id_override,
        )
        method = payload.get("method")
        if type(method) is not str or not _plain_http_header_value(method):
            raise McpProtocolError("MCP 2026 request method was not HTTP-header safe.")
        headers = {
            "accept-encoding": "identity",
            MCP_PROTOCOL_VERSION_HEADER: MCP_MODERN_PROTOCOL_VERSION,
            MCP_METHOD_HEADER: method,
        }
        params = payload.get("params")
        if type(params) is not dict:
            raise McpProtocolError("MCP 2026 request params must be an object.")
        meta = params.get("_meta")
        if type(meta) is not dict or any(
            meta.get(key) != expected for key, expected in self._request_meta.items()
        ):
            raise McpProtocolError(
                "MCP 2026 request metadata did not match the selected protocol era."
            )
        name_field = _modern_request_name_field(method)
        if name_field is not None:
            name = params.get(name_field)
            if type(name) is not str:
                raise McpProtocolError(f"MCP 2026 {method} params.{name_field} must be a string.")
            headers[MCP_NAME_HEADER] = encode_mcp_http_header_value(
                name,
                field_name="Mcp-Name",
            )
        if mirrored_headers is not None:
            for header_name, header_value in mirrored_headers.items():
                normalized_name = header_name.lower()
                if normalized_name in headers:
                    raise McpProtocolError(
                        "MCP 2026 mirrored headers collided with routing headers."
                    )
                headers[normalized_name] = header_value
        return headers

    def response_session_id(self, headers: Mapping[str, str]) -> None:
        if headers.get(MCP_SESSION_ID_HEADER) is not None:
            raise McpProtocolError("MCP 2026 HTTP responses must not mint an Mcp-Session-Id.")
        return None


def modern_http_tool_header_contract(input_schema: object) -> McpHttpToolHeaderContract:
    """Validate and compile all ``x-mcp-header`` annotations in one schema."""

    if type(input_schema) is not dict:
        return ()
    bindings: list[McpHttpToolHeaderBinding] = []
    header_names: set[str] = set()

    def non_reachable_subschemas(schema: dict[str, Any]) -> Iterator[object]:
        for keyword in _NON_REACHABLE_SCHEMA_KEYWORDS:
            value = schema.get(keyword)
            if value is None:
                continue
            if type(value) is list:
                yield from value
            elif type(value) is dict and keyword in _OBJECT_VALUED_SCHEMA_KEYWORDS:
                yield from value.values()
            else:
                yield value

    def reject_forbidden_annotation(schema: object) -> None:
        if type(schema) is not dict:
            return
        schema = cast("dict[str, Any]", schema)
        if "x-mcp-header" in schema:
            raise McpProtocolError(
                "MCP tool x-mcp-header annotations must be statically reachable "
                "through object properties."
            )
        properties = schema.get("properties")
        if type(properties) is dict:
            for child in properties.values():
                reject_forbidden_annotation(child)
        for child in non_reachable_subschemas(schema):
            reject_forbidden_annotation(child)

    def visit_reachable_schema(schema: object, path: tuple[str, ...]) -> None:
        if type(schema) is not dict:
            reject_forbidden_annotation(schema)
            return
        schema = cast("dict[str, Any]", schema)
        annotation = schema.get("x-mcp-header")
        if annotation is not None or "x-mcp-header" in schema:
            if not path:
                raise McpProtocolError(
                    "MCP tool x-mcp-header annotations cannot be declared at the schema root."
                )
            if (
                type(annotation) is not str
                or not annotation
                or not _HTTP_TOKEN.fullmatch(annotation)
            ):
                raise McpProtocolError("MCP tool x-mcp-header names must use HTTP token syntax.")
            header_name = f"{MCP_PARAMETER_HEADER_PREFIX}{annotation}"
            if len(header_name.encode("ascii")) > MAX_MCP_HTTP_HEADER_NAME_BYTES:
                raise McpProtocolError("MCP tool x-mcp-header name exceeded the HTTP header limit.")
            normalized_name = header_name.lower()
            if normalized_name in header_names:
                raise McpProtocolError(
                    "MCP tool x-mcp-header names must be case-insensitively unique."
                )
            value_type = schema.get("type")
            if type(value_type) is not str or value_type not in {
                "boolean",
                "integer",
                "string",
            }:
                raise McpProtocolError(
                    "MCP tool x-mcp-header annotations require boolean, integer, or string values."
                )
            header_names.add(normalized_name)
            bindings.append(
                McpHttpToolHeaderBinding(
                    argument_path=path,
                    header_name=normalized_name,
                    value_type=value_type,
                )
            )
            if len(bindings) > MAX_MCP_HTTP_MIRRORED_HEADERS_PER_TOOL:
                raise McpProtocolError("MCP tool schema exceeded the mirrored-header count limit.")

        properties = schema.get("properties")
        if type(properties) is dict:
            for property_name, property_schema in properties.items():
                if type(property_name) is not str:
                    raise McpProtocolError("MCP tool schema property names must be strings.")
                visit_reachable_schema(property_schema, (*path, property_name))
        for child in non_reachable_subschemas(schema):
            reject_forbidden_annotation(child)

    visit_reachable_schema(input_schema, ())
    return tuple(bindings)


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


def mirrored_mcp_http_tool_headers(
    contract: McpHttpToolHeaderContract,
    arguments: Mapping[str, Any],
) -> dict[str, str]:
    """Extract bounded routing headers from arguments using admitted authority."""

    headers: dict[str, str] | None = None
    protocol_error: str | None = None
    try:
        headers = _mirrored_mcp_http_tool_headers(contract, arguments)
    except McpProtocolError as error:
        protocol_error = str(error)
    contract = ()
    arguments = {}
    if protocol_error is not None:
        raise McpProtocolError(protocol_error) from None
    if headers is None:
        raise AssertionError("MCP mirrored-header extraction returned no result or error.")
    return headers


def _mirrored_mcp_http_tool_headers(
    contract: McpHttpToolHeaderContract,
    arguments: Mapping[str, Any],
) -> dict[str, str]:
    headers: dict[str, str] = {}
    total_bytes = 0
    for binding in contract:
        current: object = arguments
        present = True
        for segment in binding.argument_path:
            if type(current) is not dict or segment not in current:
                present = False
                break
            current = current[segment]
        if not present or current is None:
            continue
        if binding.value_type == "boolean":
            if type(current) is not bool:
                raise McpProtocolError("MCP mirrored-header argument must be a boolean.")
            value = "true" if current else "false"
        elif binding.value_type == "integer":
            if (
                type(current) is not int
                or not -MAX_PORTABLE_JSON_INTEGER <= current <= MAX_PORTABLE_JSON_INTEGER
            ):
                raise McpProtocolError(
                    "MCP mirrored-header argument must be a portable JSON integer."
                )
            value = str(current)
        else:
            if type(current) is not str:
                raise McpProtocolError("MCP mirrored-header argument must be a string.")
            value = current
        encoded = encode_mcp_http_header_value(
            value,
            field_name="MCP mirrored-header value",
        )
        total_bytes += len(binding.header_name) + len(encoded)
        if total_bytes > MAX_MCP_HTTP_MIRRORED_HEADER_BYTES:
            headers.clear()
            raise McpProtocolError("MCP mirrored tool headers exceeded the aggregate limit.")
        headers[binding.header_name] = encoded
    return headers


def encode_mcp_http_header_value(value: str, *, field_name: str) -> str:
    """Encode one MCP name/parameter using the specification's Base64 sentinel."""

    encoded: str | None = None
    protocol_error: str | None = None
    try:
        encoded = _encode_mcp_http_header_value(value, field_name=field_name)
    except McpProtocolError as error:
        protocol_error = str(error)
    value = ""
    field_name = ""
    if protocol_error is not None:
        raise McpProtocolError(protocol_error) from None
    if encoded is None:
        raise AssertionError("MCP HTTP header encoding returned no result or error.")
    return encoded


def _encode_mcp_http_header_value(value: str, *, field_name: str) -> str:
    try:
        raw = value.encode("utf-8")
    except UnicodeEncodeError:
        raise McpProtocolError(f"{field_name} was not valid UTF-8 text.") from None
    if _plain_http_header_value(value) and not (
        value.startswith(_BASE64_SENTINEL_PREFIX) and value.endswith(_BASE64_SENTINEL_SUFFIX)
    ):
        encoded = value
    else:
        encoded = (
            _BASE64_SENTINEL_PREFIX
            + base64.b64encode(raw).decode("ascii")
            + _BASE64_SENTINEL_SUFFIX
        )
    if len(encoded.encode("ascii")) > MAX_MCP_HTTP_HEADER_VALUE_BYTES:
        raise McpProtocolError(f"{field_name} exceeded the MCP HTTP header value limit.")
    return encoded


def _plain_http_header_value(value: str) -> bool:
    if not value or value[:1] in {" ", "\t"} or value[-1:] in {" ", "\t"}:
        return False
    return all(character == "\t" or " " <= character <= "~" for character in value)


def _modern_request_name_field(method: str) -> str | None:
    if method in {"tools/call", "prompts/get"}:
        return "name"
    if method == "resources/read":
        return "uri"
    return None
