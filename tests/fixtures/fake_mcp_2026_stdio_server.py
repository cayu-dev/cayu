from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

MCP_PROTOCOL_VERSION = "2026-07-28"


def main() -> None:
    request_log = Path(os.environ["CAYU_FAKE_MCP_REQUEST_LOG"])
    result_overrides = json.loads(os.environ.get("CAYU_FAKE_MCP_RESULT_OVERRIDES", "{}"))
    if type(result_overrides) is not dict:
        raise TypeError("CAYU_FAKE_MCP_RESULT_OVERRIDES must be an object")

    for line in sys.stdin:
        message = json.loads(line)
        _append_request(request_log, message)
        method = message.get("method")
        if "id" not in message:
            continue
        request_id = message["id"]
        fault_method = os.environ.get("CAYU_FAKE_MCP_FAULT_METHOD", "tools/call")
        if method == fault_method and _inject_fault(os.environ.get("CAYU_FAKE_MCP_FAULT")):
            continue
        if method in result_overrides:
            _respond(request_id, result_overrides[method])
            continue
        if method == "server/discover":
            result: Any = {
                "resultType": "complete",
                "ttlMs": 0,
                "cacheScope": "private",
                "supportedVersions": [MCP_PROTOCOL_VERSION],
                "capabilities": {
                    "tools": {"listChanged": True},
                    "resources": {},
                },
                "instructions": "Use the modern stdio fixture carefully.",
                "_meta": {
                    "io.modelcontextprotocol/serverInfo": {
                        "name": "modern-stdio-fixture",
                        "version": "1.0",
                    }
                },
            }
        elif method == "tools/list":
            result = {
                "resultType": "complete",
                "ttlMs": 10,
                "cacheScope": "public",
                "tools": [
                    {
                        "name": "search",
                        "description": "Search the fixture.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "x-mcp-header": "Query",
                                }
                            },
                        },
                    },
                    {
                        "name": "invalid-header-tool",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "ratio": {
                                    "type": "number",
                                    "x-mcp-header": "Ratio",
                                }
                            },
                        },
                    },
                ],
            }
        elif method == "tools/call":
            arguments = message.get("params", {}).get("arguments", {})
            if arguments.get("defer_response") is True:
                continue
            result = {
                "resultType": "complete",
                "content": [{"type": "text", "text": "ok"}],
                "structuredContent": ["one", 2],
                "isError": False,
            }
        elif method == "resources/list":
            result = {
                "resultType": "complete",
                "ttlMs": 0,
                "cacheScope": "private",
                "resources": [{"uri": "file://fixture", "name": "fixture"}],
            }
        elif method == "resources/read":
            result = {
                "resultType": "complete",
                "ttlMs": 0,
                "cacheScope": "private",
                "contents": [{"uri": "file://fixture", "text": "hello"}],
            }
        else:
            _write(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": f"Unsupported method: {method}"},
                }
            )
            continue
        _respond(request_id, result)


def _inject_fault(fault: str | None) -> bool:
    if fault == "disconnect":
        sys.exit(0)
    if fault == "stderr-crash":
        sys.stderr.write("fatal: fixture configuration unavailable\n")
        sys.stderr.flush()
        sys.exit(1)
    if fault == "malformed-json":
        sys.stdout.write("{not-json}\n")
        sys.stdout.flush()
        return True
    if fault == "oversized-response":
        sys.stdout.write("x" * 8192 + "\n")
        sys.stdout.flush()
        return True
    if fault == "idle":
        return True
    if fault == "stream-until-deadline":
        # Keep receiving bytes without completing a frame; the total deadline
        # must bound this peer even when the idle timer keeps being refreshed.
        while True:
            sys.stdout.write(" ")
            sys.stdout.flush()
            time.sleep(0.01)
    return False


def _append_request(path: Path, message: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n")


def _respond(request_id: object, result: object) -> None:
    _write({"jsonrpc": "2.0", "id": request_id, "result": result})


def _write(message: object) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
