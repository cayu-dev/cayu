"""Test-only malformed peers for the isolated-tool transport boundary."""

from __future__ import annotations

import json
import os
import signal
import sys

from cayu.core.tools import ToolResult
from cayu.runtime._isolated_tool_protocol import (
    encode_isolated_tool_success,
    encode_isolated_tool_terminal_frame,
)


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 2 or arguments[0] != "--result-fd":
        return 64
    result_fd = int(arguments[1])
    request = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    mode = request["factory_config"]["mode"]
    max_response_bytes = int(request["limits"]["max_response_bytes"])
    if mode == "malformed_wire":
        terminal = b"not-json"
        wire = encode_isolated_tool_terminal_frame(
            terminal,
            max_bytes=max_response_bytes,
        )
    elif mode == "multiple_wire":
        terminal = encode_isolated_tool_success(
            request_sha256=request["request_sha256"],
            result=ToolResult(content="must not be accepted"),
            max_bytes=max_response_bytes,
        )
        frame = encode_isolated_tool_terminal_frame(
            terminal,
            max_bytes=max_response_bytes,
        )
        wire = frame + frame
        frame = b""
    elif mode == "oversized_wire":
        terminal = b""
        wire = b"CIT1" + (max_response_bytes + 1).to_bytes(4, "big")
    elif mode == "secret_invalid_wire":
        terminal = b"ISOLATED_WIRE_SECRET_CANARY"
        wire = encode_isolated_tool_terminal_frame(
            terminal,
            max_bytes=max_response_bytes,
        )
    elif mode == "terminal_then_stdout_overflow":
        terminal = encode_isolated_tool_success(
            request_sha256=request["request_sha256"],
            result=ToolResult(content="must not be accepted"),
            max_bytes=max_response_bytes,
        )
        wire = encode_isolated_tool_terminal_frame(
            terminal,
            max_bytes=max_response_bytes,
        )
    else:
        return 65
    with os.fdopen(result_fd, "wb", closefd=True) as stream:
        stream.write(wire)
        stream.flush()
    terminal = b""
    wire = b""
    if mode == "terminal_then_stdout_overflow":
        sys.stdout.write("x" * 4096)
        sys.stdout.flush()
    while True:
        signal.pause()


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess tests
    raise SystemExit(main())
