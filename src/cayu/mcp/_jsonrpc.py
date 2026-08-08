"""Transport-agnostic MCP JSON-RPC helpers.

Shared by every `McpClient`/`McpSession` transport (stdio, HTTP, …) so the
JSON-RPC framing and the JSON->model parsing stay identical across transports.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeVar, cast

from cayu._exception_groups import rebuild_exception_group
from cayu._validation import copy_json_value, require_clean_nonblank, require_nonblank
from cayu.mcp.base import (
    McpInitializeResult,
    McpResourceDefinition,
    McpResourceResult,
    McpToolDefinition,
    McpToolResult,
)
from cayu.vaults import SecretRedactor

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
DEFAULT_MCP_MAX_LIST_PAGES = 100
DEFAULT_MCP_MAX_LIST_ITEMS = 10_000
JSONRPC_METHOD_NOT_FOUND = -32601

_ParsedMcpModel = TypeVar("_ParsedMcpModel")


class McpProtocolError(RuntimeError):
    """Raised when an MCP server violates the expected JSON-RPC contract."""


@dataclass(frozen=True)
class JsonrpcAuthorityMappingResult:
    """Exception-free result for private authority extraction and merging."""

    mapping: dict[str, str]
    error: str | None = None


@dataclass(frozen=True)
class JsonrpcRedactionResult:
    """Exception-free result for redacting one untrusted transport response."""

    response: dict[str, Any]
    error: str | None = None


@dataclass
class McpPaginatedPage:
    """Public page data plus private transport authority for the next request."""

    result: Any
    next_cursor: Any = None

    def clear_private_authority(self) -> None:
        self.next_cursor = None


@dataclass(frozen=True)
class _PageRequestOutcome:
    page: McpPaginatedPage | None = None
    error: BaseException | None = None
    cancellation_args: tuple[Any, ...] | None = None


def validate_positive_number(value: float, field_name: str) -> float:
    if type(value) not in {float, int}:
        raise TypeError(f"{field_name} must be a number.")
    numeric = float(value)
    if numeric <= 0:
        raise ValueError(f"{field_name} must be greater than zero.")
    return numeric


def validate_positive_integer(value: int, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero.")
    return value


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


def redact_jsonrpc_response(
    response: dict[str, Any],
    *,
    method: str,
    redactor: SecretRedactor,
) -> dict[str, Any]:
    """Redact exposable response data while preserving transport authority fields."""

    redacted: dict[str, Any] = {}
    for field_name in ("jsonrpc", "id"):
        if field_name in response:
            redacted[field_name] = copy_json_value(response[field_name], field_name)
    for field_name in ("result", "error"):
        if field_name in response:
            redacted[field_name] = redactor.redact_json(response[field_name])
    raw_result = response.get("result")
    redacted_result = redacted.get("result")
    if type(raw_result) is not dict or type(redacted_result) is not dict:
        return redacted

    if method == "initialize" and "protocolVersion" in raw_result:
        redacted_result["protocolVersion"] = copy_json_value(
            raw_result["protocolVersion"],
            "protocolVersion",
        )
    return redacted


def safely_redact_jsonrpc_response(
    response: dict[str, Any],
    *,
    method: str,
    redactor: SecretRedactor,
) -> JsonrpcRedactionResult:
    """Redact without propagating validation frames that retain the response."""

    try:
        redacted = redact_jsonrpc_response(
            response,
            method=method,
            redactor=redactor,
        )
    except ValueError:
        return JsonrpcRedactionResult(
            {},
            "MCP response contained invalid portable JSON.",
        )
    return JsonrpcRedactionResult(redacted)


def jsonrpc_authority_mapping(
    raw_response: dict[str, Any],
    redacted_response: dict[str, Any],
    *,
    method: str,
) -> JsonrpcAuthorityMappingResult:
    """Map redacted public MCP identities to their private transport identities."""

    if method == "tools/list":
        item_key, authority_key = "tools", "name"
    elif method == "resources/list":
        item_key, authority_key = "resources", "uri"
    else:
        return JsonrpcAuthorityMappingResult({})
    raw_result = raw_response.get("result")
    redacted_result = redacted_response.get("result")
    if type(raw_result) is not dict or type(redacted_result) is not dict:
        return JsonrpcAuthorityMappingResult({})
    raw_items = raw_result.get(item_key)
    redacted_items = redacted_result.get(item_key)
    if type(raw_items) is not list or type(redacted_items) is not list:
        return JsonrpcAuthorityMappingResult({})
    mapping: dict[str, str] = {}
    for raw_item, redacted_item in zip(raw_items, redacted_items, strict=False):
        if type(raw_item) is dict and type(redacted_item) is dict and authority_key in raw_item:
            raw_object = cast("dict[str, Any]", raw_item)
            redacted_object = cast("dict[str, Any]", redacted_item)
            raw_authority = raw_object[authority_key]
            redacted_authority = redacted_object.get(authority_key)
            if type(raw_authority) is not str or type(redacted_authority) is not str:
                continue
            previous = mapping.get(redacted_authority)
            if previous is not None and previous != raw_authority:
                return JsonrpcAuthorityMappingResult(
                    {},
                    f"MCP {item_key} identities collide after workload-secret redaction.",
                )
            mapping[redacted_authority] = raw_authority
    return JsonrpcAuthorityMappingResult(mapping)


def merge_jsonrpc_authority_mapping(
    current: dict[str, str],
    incoming: dict[str, str],
    *,
    max_items: int,
) -> JsonrpcAuthorityMappingResult:
    """Atomically merge a bounded private transport-identity mapping."""

    if type(current) is not dict or type(incoming) is not dict:
        raise TypeError("MCP authority mappings must be objects.")
    if type(max_items) is not int or max_items <= 0:
        raise ValueError("max_items must be a positive integer.")
    merged = dict(current)
    for public_identity, raw_identity in incoming.items():
        try:
            public_identity = require_clean_nonblank(public_identity, "public MCP identity")
            raw_identity = require_clean_nonblank(raw_identity, "raw MCP identity")
        except ValueError:
            merged.clear()
            return JsonrpcAuthorityMappingResult(
                {},
                "MCP transport identities must be clean nonblank strings.",
            )
        previous = merged.get(public_identity)
        if previous is not None and previous != raw_identity:
            return JsonrpcAuthorityMappingResult(
                {},
                "MCP transport identities collide after workload-secret redaction.",
            )
        merged[public_identity] = raw_identity
        if len(merged) > max_items:
            return JsonrpcAuthorityMappingResult(
                {},
                f"MCP private transport-identity mapping exceeded {max_items} items.",
            )
    return JsonrpcAuthorityMappingResult(merged)


def validate_negotiated_protocol_version(version: str) -> None:
    """Raise if the server negotiated a protocol revision cayu cannot speak."""
    if version not in SUPPORTED_MCP_PROTOCOL_VERSIONS:
        raise McpProtocolError(f"MCP server negotiated unsupported protocol version {version!r}.")


async def collect_paginated(
    request: Callable[[str, dict[str, Any]], Awaitable[McpPaginatedPage]],
    method: str,
    item_key: str,
    *,
    max_pages: int = DEFAULT_MCP_MAX_LIST_PAGES,
    max_items: int = DEFAULT_MCP_MAX_LIST_ITEMS,
    redactor: SecretRedactor | None = None,
) -> list[Any]:
    """Drain a paginated list request, following ``nextCursor`` across pages.

    A server may return a page of items plus an opaque ``nextCursor``; the next
    page is fetched by echoing that cursor. Without this the first page would be
    silently treated as the complete list — truncating tool/resource manifests
    (and their hashes) into a stable-looking but incomplete view.
    """
    page_limit = validate_positive_integer(max_pages, "max_pages")
    item_limit = validate_positive_integer(max_items, "max_items")
    resolved_redactor = redactor or SecretRedactor()
    items: list[Any] = []
    cursor: str | None = None
    seen_cursor_digests: set[bytes] = set()
    page_count = 0
    while True:
        params: dict[str, Any] = {} if cursor is None else {"cursor": cursor}
        paginated_page: McpPaginatedPage | None = None
        next_cursor: Any = None
        result: Any = None
        try:
            outcome = await _request_paginated_page(
                request,
                method,
                params,
                redactor=resolved_redactor,
            )
            params.clear()
            if outcome.cancellation_args is not None:
                raise asyncio.CancelledError(*outcome.cancellation_args) from None
            if outcome.error is not None:
                raise outcome.error from None
            paginated_page = outcome.page
            if paginated_page is None:
                raise AssertionError("MCP page request completed without a page.")
            result = paginated_page.result
            page_count += 1
            if type(result) is not dict:
                raise McpProtocolError(f"MCP {method} result must be an object.")
            page = result.get(item_key, [])
            if not isinstance(page, list):
                raise McpProtocolError(f"MCP {method} result {item_key} must be a list.")
            observed_items = len(items) + len(page)
            if observed_items > item_limit:
                raise McpProtocolError(
                    f"MCP {method} returned {observed_items} items, exceeding "
                    f"max_list_items={item_limit}."
                )
            items.extend(page)
            next_cursor = paginated_page.next_cursor
            paginated_page.clear_private_authority()
            paginated_page = None
            result = None
            if next_cursor is None:
                return items
            if not isinstance(next_cursor, str):
                raise McpProtocolError(f"MCP {method} nextCursor must be a string.")
            if next_cursor == "":
                return items
            if not next_cursor.strip():
                raise McpProtocolError(f"MCP {method} nextCursor must not be blank.")
            if any(0xD800 <= ord(character) <= 0xDFFF for character in next_cursor):
                raise McpProtocolError(
                    f"MCP {method} nextCursor must contain Unicode scalar values."
                )
            cursor = next_cursor
            next_cursor = None
            cursor_digest = hashlib.sha256(cursor.encode("utf-8")).digest()
            if cursor_digest in seen_cursor_digests:
                raise McpProtocolError(
                    f"MCP {method} repeated pagination cursor; refusing to loop."
                )
            seen_cursor_digests.add(cursor_digest)
            cursor_digest = b""
            if page_count >= page_limit:
                raise McpProtocolError(
                    f"MCP {method} returned a nextCursor after {page_count} pages; "
                    f"page {page_count + 1} would exceed max_list_pages={page_limit}."
                )
        except BaseException:
            if paginated_page is not None:
                paginated_page.clear_private_authority()
            paginated_page = None
            result = None
            next_cursor = None
            cursor = None
            seen_cursor_digests.clear()
            params.clear()
            raise


async def _request_paginated_page(
    request: Callable[[str, dict[str, Any]], Awaitable[McpPaginatedPage]],
    method: str,
    params: dict[str, Any],
    *,
    redactor: SecretRedactor,
) -> _PageRequestOutcome:
    """Detach private request frames before a public error or cancellation escapes."""

    request_redactor = redactor
    private_cursor = params.get("cursor")
    if type(private_cursor) is str and private_cursor.strip():
        request_redactor = request_redactor.with_secret(private_cursor)
    private_cursor = None
    try:
        page = await request(method, params)
    except asyncio.CancelledError as cancellation:
        params.clear()
        cancellation_args = _safe_redacted_cancellation_args(
            cancellation,
            redactor=request_redactor,
        )
        return _PageRequestOutcome(cancellation_args=cancellation_args)
    except BaseExceptionGroup as error:
        params.clear()
        detached = _redacted_page_request_group(
            error,
            method=method,
            redactor=request_redactor,
        )
        del error
        return _PageRequestOutcome(error=detached)
    except Exception as exc:
        params.clear()
        error = _redacted_page_request_error(
            exc,
            method=method,
            redactor=request_redactor,
        )
        return _PageRequestOutcome(error=error)
    except BaseException as fatal:
        params.clear()
        error = _redacted_page_request_fatal(
            fatal,
            redactor=request_redactor,
        )
        return _PageRequestOutcome(error=error)
    finally:
        params.clear()
    if type(page) is not McpPaginatedPage:
        return _PageRequestOutcome(
            error=McpProtocolError(f"MCP {method} page request returned an invalid envelope.")
        )
    return _PageRequestOutcome(page=page)


def _redacted_page_request_group(
    error: BaseExceptionGroup,
    *,
    method: str,
    redactor: SecretRedactor,
) -> BaseExceptionGroup:
    """Rebuild a grouped page failure without retaining private exception state."""

    group_message = f"MCP {method} paginated request reported multiple failures."
    try:
        args = BaseException.__dict__["args"].__get__(error, BaseException)
        if type(args) is tuple and args and type(args[0]) is str:
            group_message = redactor.redact_text_bounded(args[0], max_bytes=2048)
    except BaseException:
        pass

    def detach_leaf(leaf: BaseException) -> BaseException:
        if isinstance(leaf, asyncio.CancelledError):
            return asyncio.CancelledError(
                *_safe_redacted_cancellation_args(leaf, redactor=redactor)
            )
        if isinstance(leaf, Exception):
            return _redacted_page_request_error(
                leaf,
                method=method,
                redactor=redactor,
            )
        args = _safe_redacted_base_exception_args(leaf, redactor=redactor)
        if isinstance(leaf, KeyboardInterrupt):
            return KeyboardInterrupt(*args)
        if isinstance(leaf, SystemExit):
            return SystemExit(*args)
        if isinstance(leaf, GeneratorExit):
            return GeneratorExit(*args)
        return BaseException(*args)

    return rebuild_exception_group(
        error,
        group_message=group_message,
        leaf_mapper=detach_leaf,
        invalid_leaf_factory=lambda: McpProtocolError(
            f"MCP {method} paginated request reported an invalid failure group."
        ),
    )


def _redacted_page_request_error(
    error: Exception,
    *,
    method: str,
    redactor: SecretRedactor,
) -> Exception:
    """Copy a safe diagnostic without retaining the private exception traceback."""

    try:
        message = str(error).strip()
    except BaseException:
        message = ""
    if not message:
        message = "MCP paginated request failed"
    error_type = redactor.redact_text_bounded(
        _safe_exception_type_name(error),
        max_bytes=256,
    )
    message = redactor.redact_text_bounded(message, max_bytes=2048)
    if isinstance(error, TimeoutError):
        return TimeoutError(message)
    if isinstance(error, McpProtocolError):
        return McpProtocolError(message)
    return McpProtocolError(f"MCP {method} page request failed: {error_type}: {message}")


def _redacted_page_request_fatal(
    error: BaseException,
    *,
    redactor: SecretRedactor,
) -> BaseException:
    """Rebuild a scalar fatal signal without retaining private exception state."""

    args = _safe_redacted_base_exception_args(error, redactor=redactor)
    if isinstance(error, KeyboardInterrupt):
        return KeyboardInterrupt(*args)
    if isinstance(error, SystemExit):
        return SystemExit(*args)
    if isinstance(error, GeneratorExit):
        return GeneratorExit(*args)
    return BaseException(*args)


def _safe_exception_type_name(error: BaseException) -> str:
    try:
        name = type.__getattribute__(type(error), "__name__")
    except BaseException:
        return "Exception"
    return name if type(name) is str else "Exception"


def _safe_redacted_cancellation_args(
    cancellation: asyncio.CancelledError,
    *,
    redactor: SecretRedactor,
) -> tuple[Any, ...]:
    """Copy cancellation detail without retaining the private cancellation object."""

    try:
        args = tuple(cancellation.args)
    except BaseException:
        return ()
    copied: list[Any] = []
    for value in args:
        if value is None or type(value) in {bool, int, float}:
            copied.append(value)
            continue
        try:
            rendered = value if type(value) is str else str(value)
        except BaseException:
            rendered = "cancelled"
        copied.append(redactor.redact_text_bounded(rendered, max_bytes=2048))
    return tuple(copied)


def _safe_redacted_base_exception_args(
    error: BaseException,
    *,
    redactor: SecretRedactor,
) -> tuple[Any, ...]:
    """Copy fatal-signal arguments without invoking extension-owned accessors."""

    try:
        args = BaseException.__dict__["args"].__get__(error, BaseException)
    except BaseException:
        return ()
    if type(args) is not tuple:
        return ()
    copied: list[Any] = []
    for value in args:
        if value is None or type(value) in {bool, int, float}:
            copied.append(value)
            continue
        try:
            rendered = value if type(value) is str else str(value)
        except BaseException:
            rendered = "MCP paginated request failed"
        copied.append(redactor.redact_text_bounded(rendered, max_bytes=2048))
    return tuple(copied)


def _consume_model_payload(
    payload: dict[str, Any],
    parser: Callable[[dict[str, Any]], _ParsedMcpModel],
    *,
    failure_message: str,
) -> _ParsedMcpModel:
    """Parse one transport result without retaining rejected response data."""

    parsed: _ParsedMcpModel | None = None
    parse_failed = False
    try:
        parsed = parser(payload)
    except (McpProtocolError, TypeError, ValueError):
        parse_failed = True
    finally:
        payload.clear()
    if parse_failed:
        raise McpProtocolError(failure_message) from None
    if parsed is None:
        raise AssertionError("MCP model parser returned no result or error.")
    return parsed


def initialize_result_from_payload(payload: dict[str, Any]) -> McpInitializeResult:
    return _consume_model_payload(
        payload,
        _initialize_result_from_payload,
        failure_message="MCP initialize result contained invalid data.",
    )


def _initialize_result_from_payload(payload: dict[str, Any]) -> McpInitializeResult:
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
    return _consume_model_payload(
        payload,
        _tool_result_from_payload,
        failure_message="MCP tools/call result contained invalid data.",
    )


def _tool_result_from_payload(payload: dict[str, Any]) -> McpToolResult:
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


def resource_result_from_payload(payload: dict[str, Any]) -> McpResourceResult:
    return _consume_model_payload(
        payload,
        _resource_result_from_payload,
        failure_message="MCP resources/read result contained invalid data.",
    )


def _resource_result_from_payload(payload: dict[str, Any]) -> McpResourceResult:
    contents = payload.get("contents", [])
    if not isinstance(contents, list):
        raise McpProtocolError("MCP resources/read result contents must be a list.")
    return McpResourceResult(contents=contents)


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
