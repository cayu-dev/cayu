from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path


async def main() -> None:
    """Run a credential-free MCP catalogue-change notification end to end."""

    from cayu import (
        AgentSpec,
        CayuApp,
        McpServerSpec,
        McpToolAdapter,
        McpToolsetUnavailable,
        StdioMcpClient,
        StdioMcpProcessLifetime,
        ToolContext,
        connect_mcp_toolset,
    )

    server = McpServerSpec(
        name="changing-mcp",
        connection_id="example/mcp-list-changed/changing-mcp",
        command=[sys.executable, str(Path(__file__).resolve()), "--server"],
    )
    toolset = await connect_mcp_toolset(
        server,
        client=StdioMcpClient(
            process_lifetime=StdioMcpProcessLifetime.GRACEFUL_CLEANUP,
        ),
    )
    try:
        app = CayuApp(enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="unused-by-this-example"),
            mcp_toolsets=(toolset,),
        )
        initial_adapter = toolset.tools[0]
        print("initial", tuple(app.get_agent("assistant").tools))

        # This demo-only resource mutates the server catalogue and emits the
        # standard MCP tools/list_changed notification before replying.
        await toolset.session.read_resource("control://advance-catalogue")
        await _wait_for_refresh(toolset)

        current = app.get_agent("assistant")
        current_adapter = current.tools["mcp__changing-mcp__remember"].tool
        assert isinstance(current_adapter, McpToolAdapter)
        old_snapshot_is_stale = False
        try:
            await initial_adapter.run(
                ToolContext(session_id="example", agent_name="assistant"),
                {},
            )
        except McpToolsetUnavailable:
            old_snapshot_is_stale = True
        print("refreshed", tuple(current.tools))
        print("generation", current_adapter.toolset.generation)
        print("old_snapshot_is_stale", old_snapshot_is_stale)
    finally:
        await toolset.close()


async def _wait_for_refresh(toolset) -> None:
    from cayu import McpToolsetRefreshState

    async with asyncio.timeout(2):
        while toolset.refresh_state is not McpToolsetRefreshState.READY:
            await asyncio.sleep(0.001)


def server_main() -> None:
    catalogue_revision = 1
    for line in sys.stdin:
        message = json.loads(line)
        method = message.get("method")
        if "id" not in message:
            continue
        request_id = message["id"]
        if method == "initialize":
            _write_result(
                request_id,
                {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {
                        "tools": {"listChanged": True},
                        "resources": {},
                    },
                    "serverInfo": {"name": "changing-mcp", "version": "1.0.0"},
                },
            )
        elif method == "tools/list":
            tools = [_tool_definition("echo")]
            if catalogue_revision >= 2:
                tools.append(_tool_definition("remember"))
            _write_result(request_id, {"tools": tools})
        elif method == "resources/read":
            catalogue_revision = 2
            _write_notification("notifications/tools/list_changed")
            _write_result(
                request_id,
                {
                    "contents": [
                        {
                            "uri": "control://advance-catalogue",
                            "mimeType": "text/plain",
                            "text": "catalogue advanced",
                        }
                    ]
                },
            )


def _tool_definition(name: str) -> dict:
    return {
        "name": name,
        "description": f"Deterministic {name} tool.",
        "inputSchema": {"type": "object", "additionalProperties": False},
    }


def _write_notification(method: str) -> None:
    _write({"jsonrpc": "2.0", "method": method})


def _write_result(request_id: int, result: dict) -> None:
    _write({"jsonrpc": "2.0", "id": request_id, "result": result})


def _write(message: dict) -> None:
    sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    if "--server" in sys.argv:
        server_main()
    else:
        asyncio.run(main())
