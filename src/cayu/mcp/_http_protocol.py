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
from typing import Any, cast

from cayu._validation import MAX_PORTABLE_JSON_INTEGER
from cayu.mcp._jsonrpc import MCP_MODERN_PROTOCOL_VERSION, McpProtocolError
from cayu.mcp._protocol import McpProtocolEra, ModernMcpWireProtocol

MCP_PROTOCOL_VERSION_HEADER = "mcp-protocol-version"
MCP_SESSION_ID_HEADER = "mcp-session-id"
MCP_METHOD_HEADER = "mcp-method"
MCP_NAME_HEADER = "mcp-name"
MCP_PARAMETER_HEADER_PREFIX = "mcp-param-"

MAX_MCP_HTTP_MIRRORED_HEADERS_PER_TOOL = 64
MAX_MCP_HTTP_HEADER_NAME_BYTES = 256
MAX_MCP_HTTP_HEADER_VALUE_BYTES = 8_192
MAX_MCP_HTTP_MIRRORED_HEADER_BYTES = 65_536

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


@dataclass(frozen=True, slots=True)
class McpHttpToolHeaderBinding:
    """Private routing-header authority derived from one admitted tool schema."""

    argument_path: tuple[str, ...]
    header_name: str
    value_type: str


McpHttpToolHeaderContract = tuple[McpHttpToolHeaderBinding, ...]


@dataclass(frozen=True, slots=True)
class McpHttpToolHeaderFilterResult:
    """Contracts aligned with retained tools plus the excluded wire count."""

    contracts: tuple[McpHttpToolHeaderContract, ...]
    excluded_count: int


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


class ModernHttpMcpWireProtocol(ModernMcpWireProtocol):
    """Stateless MCP 2026-07-28 Streamable HTTP codec."""

    uses_protocol_sessions = False

    def __init__(self, *, client_name: str, client_version: str) -> None:
        super().__init__(client_name=client_name, client_version=client_version)

    def prepare_request_params(
        self,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        if not _plain_http_header_value(method):
            raise McpProtocolError("MCP 2026 request method was not HTTP-header safe.")
        name_field = _modern_request_name_field(method)
        if name_field is not None:
            name = params.get(name_field)
            if type(name) is not str:
                raise McpProtocolError(f"MCP 2026 {method} params.{name_field} must be a string.")
            encode_mcp_http_header_value(name, field_name="Mcp-Name")
        return super().prepare_request_params(method, params)

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


def filter_invalid_modern_http_tool_headers(
    response: dict[str, Any],
) -> McpHttpToolHeaderFilterResult | None:
    """Drop tools with invalid HTTP-header annotations and retain aligned contracts."""

    result = response.get("result")
    if type(result) is not dict:
        return None
    tools = result.get("tools")
    if type(tools) is not list:
        return None
    retained_tools: list[Any] = []
    contracts: list[McpHttpToolHeaderContract] = []
    excluded_count = 0
    for tool in tools:
        input_schema = tool.get("inputSchema", {}) if type(tool) is dict else None
        try:
            contract = modern_http_tool_header_contract(input_schema)
        except McpProtocolError:
            excluded_count += 1
            continue
        retained_tools.append(tool)
        contracts.append(contract)
    tools.clear()
    tools.extend(retained_tools)
    retained_tools.clear()
    return McpHttpToolHeaderFilterResult(
        contracts=tuple(contracts),
        excluded_count=excluded_count,
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
