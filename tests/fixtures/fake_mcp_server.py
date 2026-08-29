from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def main() -> None:
    deferred_tool_response = None
    emitted_post_discovery_list_changed = False
    for line in sys.stdin:
        message = json.loads(line)
        method = message.get("method")
        if method == os.environ.get("CAYU_FAKE_MCP_LIST_CHANGED_ON_METHOD"):
            count = int(os.environ.get("CAYU_FAKE_MCP_LIST_CHANGED_COUNT", "1"))
            for _ in range(count):
                notification = {
                    "jsonrpc": "2.0",
                    "method": "notifications/tools/list_changed",
                }
                notification_params = os.environ.get("CAYU_FAKE_MCP_LIST_CHANGED_PARAMS")
                if notification_params is not None:
                    notification["params"] = json.loads(notification_params)
                _write(notification)
            response_delay_s = os.environ.get("CAYU_FAKE_MCP_LIST_CHANGED_RESPONSE_DELAY_S")
            if response_delay_s is not None:
                time.sleep(float(response_delay_s))
        if "id" not in message:
            continue
        request_id = message["id"]
        paginated_failure_method = os.environ.get("CAYU_FAKE_MCP_PAGINATED_FAILURE_METHOD")
        if paginated_failure_method is not None and method == paginated_failure_method:
            cursor = message.get("params", {}).get("cursor")
            if cursor is not None:
                sys.stdout.write("{invalid-json\n")
                sys.stdout.flush()
                continue
            private_identity = os.environ["CAYU_FAKE_MCP_PAGINATED_PRIVATE_IDENTITY"]
            page_canary = os.environ["CAYU_FAKE_MCP_PAGINATED_PAGE_CANARY"]
            if method == "resources/list":
                result = {
                    "resources": [
                        {
                            "uri": private_identity,
                            "description": page_canary,
                        }
                    ],
                    "nextCursor": "failing-page",
                }
            else:
                result = {
                    "tools": [
                        {
                            "name": private_identity,
                            "description": page_canary,
                            "inputSchema": {"type": "object"},
                        }
                    ],
                    "nextCursor": "failing-page",
                }
            _write({"jsonrpc": "2.0", "id": request_id, "result": result})
            continue
        structural_response = os.environ.get("CAYU_FAKE_MCP_STRUCTURAL_RESPONSE")
        structural_method = os.environ.get(
            "CAYU_FAKE_MCP_STRUCTURAL_METHOD",
            "tools/list",
        )
        if structural_response is not None and method == structural_method:
            canary = os.environ.get("CAYU_FAKE_MCP_STRUCTURAL_CANARY", "")
            if structural_response == "non_object":
                _write([canary])
            elif structural_response == "wrong_version":
                _write(
                    {
                        "jsonrpc": "1.0",
                        "id": request_id,
                        "result": {"secret": canary},
                    }
                )
            elif structural_response == "non_finite":
                _write(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "tools": [
                                {
                                    "name": "echo",
                                    "description": canary,
                                    "inputSchema": {"type": "object"},
                                }
                            ],
                            "invalid": float("nan"),
                        },
                    }
                )
            elif structural_response == "non_finite_cursor":
                _write(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "tools": [
                                {
                                    "name": "echo",
                                    "description": canary,
                                    "inputSchema": {"type": "object"},
                                }
                            ],
                            "nextCursor": float("nan"),
                        },
                    }
                )
            elif structural_response == "unclean_identity":
                _write(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "tools": [
                                {
                                    "name": f" {canary}",
                                    "description": "Unclean private transport identity.",
                                    "inputSchema": {"type": "object"},
                                }
                            ]
                        },
                    }
                )
            elif structural_response in {
                "ambiguous_identity_first",
                "ambiguous_identity_last",
                "ambiguous_identity_only",
            }:
                if structural_method == "resources/list":
                    valid_item = {
                        "uri": canary,
                        "name": "Valid private transport identity.",
                    }
                    ambiguous_item = {
                        "uri": True,
                        "name": "Wrong-type transport identity.",
                    }
                    result_key = "resources"
                else:
                    valid_item = {
                        "name": canary,
                        "description": "Valid private transport identity.",
                        "inputSchema": {"type": "object"},
                    }
                    ambiguous_item = {
                        "name": True,
                        "description": "Wrong-type transport identity.",
                        "inputSchema": {"type": "object"},
                    }
                    result_key = "tools"
                items = (
                    [ambiguous_item]
                    if structural_response == "ambiguous_identity_only"
                    else (
                        [ambiguous_item, valid_item]
                        if structural_response == "ambiguous_identity_first"
                        else [valid_item, ambiguous_item]
                    )
                )
                _write(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {result_key: items},
                    }
                )
            elif structural_response == "invalid_portable_result":
                invalid_text = f"{canary}\x00"
                if structural_method == "tools/call":
                    result = {"content": [{"type": "text", "text": invalid_text}]}
                elif structural_method == "resources/read":
                    result = {"contents": [{"text": invalid_text}]}
                else:
                    raise ValueError(
                        "invalid_portable_result requires tools/call or resources/read"
                    )
                _write({"jsonrpc": "2.0", "id": request_id, "result": result})
            else:
                raise ValueError(
                    f"Unsupported CAYU_FAKE_MCP_STRUCTURAL_RESPONSE: {structural_response}"
                )
            continue
        if method == "initialize":
            initialize_delay_s = os.environ.get("CAYU_FAKE_MCP_INITIALIZE_DELAY_S")
            if initialize_delay_s is not None:
                time.sleep(float(initialize_delay_s))
            protocol_version = os.environ.get(
                "CAYU_FAKE_MCP_PROTOCOL_VERSION",
                "2025-06-18",
            )
            if os.environ.get("CAYU_FAKE_MCP_INVALID_PROTOCOL_TEXT") == "1":
                protocol_version += "\x00"
            _write(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": protocol_version,
                        "capabilities": {
                            "tools": (
                                {"listChanged": True}
                                if os.environ.get("CAYU_FAKE_MCP_LIST_CHANGED") == "1"
                                else {}
                            ),
                            "resources": {},
                        },
                        "serverInfo": {
                            "name": "fake-mcp",
                            "version": "1.0.0",
                        },
                        "instructions": "Use fake MCP tools only when explicitly requested.",
                    },
                }
            )
        elif method == "tools/list":
            catalogue_path = os.environ.get("CAYU_FAKE_MCP_TOOL_CATALOGUE_FILE")
            if catalogue_path is not None:
                tools = json.loads(Path(catalogue_path).read_text(encoding="utf-8"))
                _write(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {"tools": tools},
                    }
                )
                if not emitted_post_discovery_list_changed:
                    emitted_post_discovery_list_changed = (
                        _emit_post_discovery_list_changed_if_configured()
                    )
                continue
            echo_tool = {
                "name": "echo",
                "description": "Echo text.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            }
            if os.environ.get("CAYU_FAKE_MCP_PAGINATE") == "1":
                cursor = message.get("params", {}).get("cursor")
                if cursor is None:
                    result = {"tools": [echo_tool], "nextCursor": "tools-page-2"}
                else:
                    result = {
                        "tools": [{**echo_tool, "name": "echo_page_2"}],
                    }
            else:
                result = {"tools": [echo_tool]}
            _write({"jsonrpc": "2.0", "id": request_id, "result": result})
            if not emitted_post_discovery_list_changed:
                emitted_post_discovery_list_changed = (
                    _emit_post_discovery_list_changed_if_configured()
                )
        elif method == "tools/call":
            params = message.get("params", {})
            name = params.get("name")
            arguments = params.get("arguments", {})
            if arguments.get("close_stdout"):
                sys.stdout.close()
                return
            if arguments.get("close_stdout_keep_stderr_s") is not None:
                os.close(sys.stdout.fileno())
                time.sleep(float(arguments["close_stdout_keep_stderr_s"]))
                return
            if arguments.get("invalid_utf8"):
                sys.stdout.buffer.write(b"\xff\n")
                sys.stdout.buffer.flush()
                continue
            if arguments.get("server_request_first"):
                _write(
                    {
                        "jsonrpc": "2.0",
                        "id": "server_request_1",
                        "method": "roots/list",
                        "params": {},
                    }
                )
            catalogue_path = os.environ.get("CAYU_FAKE_MCP_TOOL_CATALOGUE_FILE")
            known_tool_names = {"echo"}
            if catalogue_path is not None:
                known_tool_names = {
                    tool.get("name")
                    for tool in json.loads(Path(catalogue_path).read_text(encoding="utf-8"))
                    if type(tool) is dict
                }
            if name not in known_tool_names:
                _write(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32601, "message": "Unknown tool."},
                    }
                )
                continue
            text = arguments.get("text", "")
            deep_response_depth = arguments.get("deep_response_depth")
            if deep_response_depth is not None:
                _write_deep_tool_response(
                    request_id,
                    depth=int(deep_response_depth),
                    canary=str(arguments.get("deep_response_canary", "")),
                )
                continue
            response = _tool_response(
                request_id,
                text=text,
                structured_only=bool(arguments.get("structured_only")),
            )
            exact_response_bytes = arguments.get("exact_response_bytes")
            if exact_response_bytes is not None:
                secret_prefix = (
                    os.environ.get("CAYU_FAKE_MCP_LIMIT_CANARY", "")
                    if arguments.get("include_limit_canary")
                    else ""
                )
                response = _tool_response_with_exact_bytes(
                    request_id,
                    exact_response_bytes,
                    prefix=secret_prefix,
                )
            if arguments.get("slow_response_delay_s") is not None:
                _write_slow(response, float(arguments["slow_response_delay_s"]))
                continue
            if arguments.get("defer_response"):
                deferred_tool_response = response
                continue
            _write(response)
            if deferred_tool_response is not None:
                _write(deferred_tool_response)
                deferred_tool_response = None
        elif method == "resources/list":
            _write(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "resources": [
                            {
                                "uri": "file:///hello.txt",
                                "name": "hello.txt",
                                "description": "Greeting resource.",
                                "mimeType": "text/plain",
                            }
                        ]
                    },
                }
            )
        elif method == "resources/read":
            _write(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "contents": [
                            {
                                "uri": "file:///hello.txt",
                                "mimeType": "text/plain",
                                "text": "hello from resource",
                            }
                        ]
                    },
                }
            )
        else:
            _write(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": f"Unsupported method: {method}"},
                }
            )


def _write(message: object) -> None:
    sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _emit_post_discovery_list_changed_if_configured() -> bool:
    delay_s = os.environ.get("CAYU_FAKE_MCP_POST_DISCOVERY_LIST_CHANGED_DELAY_S")
    if delay_s is None:
        return False
    time.sleep(float(delay_s))
    _write({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"})
    return True


def _write_slow(message: object, delay_s: float) -> None:
    data = (json.dumps(message, separators=(",", ":")) + "\n").encode()
    for byte in data:
        sys.stdout.buffer.write(bytes((byte,)))
        sys.stdout.buffer.flush()
        time.sleep(delay_s)


def _write_deep_tool_response(request_id: int, *, depth: int, canary: str) -> None:
    prefix = ('{"jsonrpc":"2.0","id":' + str(request_id) + ',"result":{"nested":').encode()
    leaf = json.dumps(canary, separators=(",", ":")).encode()
    sys.stdout.buffer.write(prefix + b"[" * depth + leaf + b"]" * depth + b"}}\n")
    sys.stdout.buffer.flush()


def _tool_response_with_exact_bytes(
    request_id: int,
    target_bytes: int,
    *,
    prefix: str = "",
) -> dict:
    response = {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "content": [{"type": "text", "text": prefix}],
            "structuredContent": {},
        },
    }
    encoded = json.dumps(response, separators=(",", ":")).encode()
    if target_bytes < len(encoded):
        raise ValueError("exact_response_bytes is smaller than the response envelope")
    padding = "x" * (target_bytes - len(encoded))
    response["result"]["content"][0]["text"] = prefix + padding
    return response


def _tool_response(request_id, *, text: str, structured_only: bool) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "content": ([] if structured_only else [{"type": "text", "text": f"echo: {text}"}]),
            "structuredContent": {"echoed": text},
        },
    }


if __name__ == "__main__":
    main()
