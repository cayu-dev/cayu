from __future__ import annotations

import json
import os
import sys


def main() -> None:
    deferred_tool_response = None
    for line in sys.stdin:
        message = json.loads(line)
        method = message.get("method")
        if "id" not in message:
            continue
        request_id = message["id"]
        if method == "initialize":
            _write(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": os.environ.get(
                            "CAYU_FAKE_MCP_PROTOCOL_VERSION",
                            "2025-06-18",
                        ),
                        "capabilities": {
                            "tools": {},
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
            _write(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "tools": [
                            {
                                "name": "echo",
                                "description": "Echo text.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {"text": {"type": "string"}},
                                    "required": ["text"],
                                },
                            }
                        ]
                    },
                }
            )
        elif method == "tools/call":
            params = message.get("params", {})
            name = params.get("name")
            arguments = params.get("arguments", {})
            if arguments.get("server_request_first"):
                _write(
                    {
                        "jsonrpc": "2.0",
                        "id": "server_request_1",
                        "method": "roots/list",
                        "params": {},
                    }
                )
            if name != "echo":
                _write(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32601, "message": "Unknown tool."},
                    }
                )
                continue
            text = arguments.get("text", "")
            response = _tool_response(
                request_id,
                text=text,
                structured_only=bool(arguments.get("structured_only")),
            )
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


def _write(message: dict) -> None:
    sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
    sys.stdout.flush()


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
