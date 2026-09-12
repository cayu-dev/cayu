"""Observe a modern HTTP server's changing tool catalogue without an LLM.

Run against a server advertising tools.listChanged:
    uv run python examples/mcp_http_subscriptions.py http://127.0.0.1:8000/mcp

For the repository's official SDK fixture, add --trigger-resource control://advance.
"""

from __future__ import annotations

import argparse
import asyncio

from cayu import (
    AgentSpec,
    CayuApp,
    HttpMcpClient,
    McpProtocolEra,
    McpServerSpec,
    McpToolsetRefreshState,
    connect_mcp_toolset,
)


async def observe(url: str, *, trigger_resource: str | None, duration_s: float) -> None:
    toolset = await connect_mcp_toolset(
        McpServerSpec(name="remote", connection_id="example/modern-http-subscription", url=url),
        client=HttpMcpClient(protocol_era=McpProtocolEra.MODERN_2026_07_28),
    )
    try:
        app = CayuApp(enable_logging=False)
        app.register_agent(
            AgentSpec(name="observer", model="unused-by-this-example"), mcp_toolsets=(toolset,)
        )
        async with asyncio.timeout(10):
            while toolset.refresh_state is not McpToolsetRefreshState.READY:
                await asyncio.sleep(0.01)
        previous = tuple(app.get_agent("observer").tools)
        print("initial", previous)
        if trigger_resource is not None:
            await toolset.session.read_resource(trigger_resource)
        deadline = asyncio.get_running_loop().time() + duration_s
        while asyncio.get_running_loop().time() < deadline:
            current = tuple(app.get_agent("observer").tools)
            if current != previous:
                print("refreshed", current)
                previous = current
            await asyncio.sleep(0.05)
    finally:
        await toolset.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--trigger-resource")
    parser.add_argument("--observe-seconds", type=float, default=30)
    args = parser.parse_args()
    asyncio.run(
        observe(args.url, trigger_resource=args.trigger_resource, duration_s=args.observe_seconds)
    )
