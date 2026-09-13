"""Official SDK fixture for shared-channel subscription and cancellation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mcp.server import MCPServer
from mcp.server.subscriptions import InMemorySubscriptionBus, ToolsListChanged

if TYPE_CHECKING:
    from mcp.server.context import CallNext, HandlerResult, ServerRequestContext


async def record(ctx: ServerRequestContext[Any, Any], call_next: CallNext) -> HandlerResult:
    path = Path(os.environ["CAYU_SDK_MCP_REQUEST_LOG"])
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps({"method": ctx.method, "params": ctx.params}) + "\n")
    try:
        return await call_next(ctx)
    finally:
        if ctx.method == "subscriptions/listen":
            with path.open("a", encoding="utf-8") as output:
                output.write(json.dumps({"subscriptionClosed": True}) + "\n")


bus = InMemorySubscriptionBus()
server = MCPServer(
    "stdio-subscription-fixture", version="1.0", subscriptions=bus, middleware=[record]
)


@server.tool()
def echo(text: str) -> str:
    """Echo text."""
    return text


def added() -> str:
    """A newly available tool."""
    return "new"


@server.resource("control://advance")
async def advance() -> str:
    """Add a tool and notify every interested subscription."""
    server.tool()(added)
    await bus.publish(ToolsListChanged())
    return "advanced"


if __name__ == "__main__":
    server.run()
