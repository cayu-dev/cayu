"""Official SDK server for real HTTP subscription and catalogue-refresh tests."""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path
from typing import TYPE_CHECKING, Any

import uvicorn
from mcp.server import MCPServer
from mcp.server.subscriptions import InMemorySubscriptionBus, ToolsListChanged

if TYPE_CHECKING:
    from mcp.server.context import CallNext, HandlerResult, ServerRequestContext


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready-path", type=Path, required=True)
    parser.add_argument("--request-log", type=Path, required=True)
    args = parser.parse_args()

    async def record(ctx: ServerRequestContext[Any, Any], call_next: CallNext) -> HandlerResult:
        with args.request_log.open("a", encoding="utf-8") as output:
            output.write(json.dumps({"method": ctx.method, "params": ctx.params}) + "\n")
        try:
            return await call_next(ctx)
        finally:
            if ctx.method == "subscriptions/listen":
                with args.request_log.open("a", encoding="utf-8") as output:
                    output.write(json.dumps({"subscriptionClosed": True}) + "\n")

    bus = InMemorySubscriptionBus()
    mcp = MCPServer("subscription-fixture", version="1.0", subscriptions=bus, middleware=[record])

    @mcp.tool()
    def echo(text: str) -> str:
        """Echo the supplied text."""
        return text

    def added() -> str:
        """A newly available tool."""
        return "new"

    @mcp.resource("control://advance")
    async def advance() -> str:
        """Add a tool and publish a correlated tool-list-change event."""
        mcp.tool()(added)
        await bus.publish(ToolsListChanged())
        return "catalogue advanced"

    app = mcp.streamable_http_app()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(128)
        args.ready_path.write_text(
            f"http://127.0.0.1:{sock.getsockname()[1]}/mcp", encoding="utf-8"
        )
        uvicorn.Server(uvicorn.Config(app, log_level="error")).run(sockets=[sock])


if __name__ == "__main__":
    main()
