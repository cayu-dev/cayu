"""Observe a modern stdio tool catalogue without an LLM.

Run: uv run python examples/mcp_stdio_subscriptions.py -- <server> [arguments ...]
On platforms without default process containment, explicitly select --graceful-cleanup.
The official SDK fixture supports --trigger-resource control://advance and requires
CAYU_SDK_MCP_REQUEST_LOG in its environment (pass an `env NAME=value ...` command).
"""

from __future__ import annotations

import argparse
import asyncio
import math

from cayu import (
    AgentSpec,
    CayuApp,
    McpProtocolEra,
    McpServerSpec,
    McpToolsetRefreshState,
    StdioMcpClient,
    StdioMcpProcessLifetime,
    connect_mcp_toolset,
)


async def observe(
    command: list[str], *, graceful: bool, trigger: str | None, duration: float
) -> None:
    client = (
        StdioMcpClient(
            protocol_era=McpProtocolEra.MODERN_2026_07_28,
            process_lifetime=StdioMcpProcessLifetime.GRACEFUL_CLEANUP,
        )
        if graceful
        else StdioMcpClient(protocol_era=McpProtocolEra.MODERN_2026_07_28)
    )
    toolset = await connect_mcp_toolset(
        McpServerSpec(name="local", connection_id="example/stdio-subscription", command=command),
        client=client,
    )
    try:
        app = CayuApp(enable_logging=False)
        app.register_agent(AgentSpec(name="observer", model="unused"), mcp_toolsets=(toolset,))
        async with asyncio.timeout(10):
            while toolset.refresh_state is not McpToolsetRefreshState.READY:
                await asyncio.sleep(0.01)
        previous = tuple(app.get_agent("observer").tools)
        print("initial", previous)
        if trigger is not None:
            await toolset.session.read_resource(trigger)
        deadline = asyncio.get_running_loop().time() + duration
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
    parser.add_argument("--graceful-cleanup", action="store_true")
    parser.add_argument("--trigger-resource")
    parser.add_argument("--observe-seconds", type=float, default=10)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("provide a server command after --")
    if not math.isfinite(args.observe_seconds) or args.observe_seconds <= 0:
        parser.error("--observe-seconds must be positive and finite")
    asyncio.run(
        observe(
            command,
            graceful=args.graceful_cleanup,
            trigger=args.trigger_resource,
            duration=args.observe_seconds,
        )
    )
