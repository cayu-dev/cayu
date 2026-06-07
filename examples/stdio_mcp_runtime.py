from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path

from cayu import (
    AgentSpec,
    CayuApp,
    McpServerSpec,
    Message,
    RunRequest,
    connect_mcp_toolset,
)
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent


class FakeProvider(ModelProvider):
    name = "fake"

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if len(self.requests) == 1:
            yield ModelStreamEvent.tool_call(
                id="call_1",
                name=self.tool_name,
                arguments={"text": "hello from MCP"},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        yield ModelStreamEvent.text_delta("final answer after MCP tool")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


async def main() -> None:
    server = McpServerSpec(
        name="local-mcp",
        command=[sys.executable, str(Path(__file__).resolve()), "--server"],
    )
    toolset = await connect_mcp_toolset(server)
    try:
        if not toolset.tools:
            raise RuntimeError("MCP server did not advertise tools.")

        app = CayuApp()
        app.register_provider(FakeProvider(toolset.tools[0].name), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=toolset.tools,
        )

        print("mcp_tool", toolset.tools[0].name)
        print("mcp_server", toolset.initialize_result.server_name)
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "Use the MCP echo tool.")],
            )
        ):
            print(event.type, event.tool_name or "-", event.payload)
    finally:
        await toolset.close()


def server_main() -> None:
    for line in sys.stdin:
        message = json.loads(line)
        method = message.get("method")
        if "id" not in message:
            continue
        request_id = message["id"]
        if method == "initialize":
            _write(
                request_id,
                {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "example-mcp", "version": "1.0.0"},
                    "instructions": "Echo only text explicitly supplied by the caller.",
                },
            )
        elif method == "tools/list":
            _write(
                request_id,
                {
                    "tools": [
                        {
                            "name": "echo",
                            "description": "Echo text from the MCP example server.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"text": {"type": "string"}},
                                "required": ["text"],
                            },
                        }
                    ]
                },
            )
        elif method == "tools/call":
            text = message.get("params", {}).get("arguments", {}).get("text", "")
            _write(
                request_id,
                {
                    "content": [{"type": "text", "text": f"echo: {text}"}],
                    "structuredContent": {"echoed": text},
                },
            )


def _write(request_id: int, result: dict) -> None:
    sys.stdout.write(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result,
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    sys.stdout.flush()


if __name__ == "__main__":
    if "--server" in sys.argv:
        server_main()
    else:
        asyncio.run(main())
