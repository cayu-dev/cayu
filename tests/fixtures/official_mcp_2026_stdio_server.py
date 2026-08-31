"""Real MCP Python SDK server; the SDK owns framing and protocol behavior."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mcp.server import MCPServer
from mcp_types import CallToolResult, TextContent

if TYPE_CHECKING:
    from mcp.server.context import CallNext, HandlerResult, ServerRequestContext


async def record_request(ctx: ServerRequestContext[Any, Any], call_next: CallNext) -> HandlerResult:
    path = Path(os.environ["CAYU_SDK_MCP_REQUEST_LOG"])
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps({"method": ctx.method, "params": ctx.params}) + "\n")
    return await call_next(ctx)


server = MCPServer(
    "official-stdio-fixture",
    version="1.0",
    instructions="Exercise the official SDK over stdio.",
    middleware=[record_request],
)


@server.tool()
def search(query: str) -> CallToolResult:
    """Return the query as text and arbitrary structured JSON."""
    return CallToolResult(
        content=[TextContent(type="text", text=query)],
        structured_content=[query, 2, None],
    )


@server.resource("fixture://message", name="message")
def message() -> str:
    """A static resource for the stdio interoperability test."""
    return "hello from the official SDK"


if __name__ == "__main__":
    server.run()
