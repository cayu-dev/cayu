"""Thin Cayu client driver for pinned official HTTP conformance scenarios.

The official runner supplies the loopback URL and scenario environment. All wire
behavior belongs to Cayu: this driver never constructs requests or patches peers.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from urllib.parse import urlsplit

from cayu import HttpMcpClient, McpProtocolEra, McpServerSpec

SCENARIOS = frozenset(
    {
        "initialize",
        "tools_call",
        "http-standard-headers",
        "http-custom-headers",
        "http-invalid-tool-headers",
        "json-schema-ref-no-deref",
        "json-schema-2020-12-preservation",
    }
)


async def run(url: str, scenario: str, version: str, context: dict) -> None:
    if scenario not in SCENARIOS:
        raise ValueError("Unsupported conformance scenario.")
    eras = {
        "2025-06-18": McpProtocolEra.LEGACY,
        "2026-07-28": McpProtocolEra.MODERN_2026_07_28,
    }
    if version not in eras or (scenario == "initialize" and version != "2025-06-18"):
        raise ValueError("Unsupported conformance protocol version.")
    parsed = urlsplit(url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
        or parsed.username
        or parsed.password
    ):
        raise ValueError("Conformance peers must be credential-free loopback HTTP URLs.")
    session = await HttpMcpClient(protocol_era=eras[version]).connect(
        McpServerSpec(name="conformance", url=url)
    )
    try:
        tools = await session.list_tools()
        if scenario in {"initialize", "json-schema-ref-no-deref"}:
            return
        if scenario == "tools_call":
            result = await session.call_tool("add_numbers", {"a": 2, "b": 3})
            if result.is_error or result.content != [
                {"type": "text", "text": "The sum of 2 and 3 is 5"}
            ]:
                raise AssertionError("Tool call did not preserve the official fixture result.")
        elif scenario == "http-standard-headers":
            for tool in tools:
                await session.call_tool(tool.name, {})
            for resource in await session.list_resources():
                await session.read_resource(resource.uri)
        elif scenario == "http-custom-headers":
            for call in context["toolCalls"]:
                await session.call_tool(call["name"], call["arguments"])
        elif scenario == "http-invalid-tool-headers":
            # Exercise the catalogue Cayu actually admitted; do not silently
            # filter the fixture's invalid tools in this adapter.
            for tool in tools:
                await session.call_tool(tool.name, {"region": "us-east1", "value": "test"})
        elif scenario == "json-schema-2020-12-preservation":
            focal = next(tool for tool in tools if tool.name == "json_schema_2020_12_tool")
            await session.call_tool("json_schema_echo", {"schema": focal.input_schema})
    finally:
        await session.close()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Expected one conformance server URL.")
    token = os.environ["CAYU_MCP_CONFORMANCE_TOKEN"]
    if not token:
        raise ValueError("The gate must provide a fresh completion token.")
    scenario = os.environ["MCP_CONFORMANCE_SCENARIO"]
    version = os.environ["MCP_CONFORMANCE_PROTOCOL_VERSION"]
    asyncio.run(
        run(
            sys.argv[1],
            scenario,
            version,
            json.loads(os.environ.get("MCP_CONFORMANCE_CONTEXT", "{}")),
        )
    )
    # Emit only after operations, session.close(), and event-loop shutdown.
    print(
        json.dumps({"cayu_completion": token, "scenario": scenario, "version": version}), flush=True
    )


if __name__ == "__main__":
    main()
