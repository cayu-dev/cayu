"""A custom Tool that consumes runtime services from ``ToolContext``.

``ToolContext`` hands a tool live runtime handles — ``ctx.workspace`` (files) and
``ctx.runner`` (command execution) — the same seam the framework's own
``ExecCommandTool`` uses. Any bespoke tool (a coverage parser, a screenshot
runner, a flaky-test detector) reaches for these. This example writes a file
through the workspace, counts its lines through the runner, and reads it back.

Run it (no API key; a scripted provider drives the one tool call):

    PYTHONPATH=src python examples/custom_runner_tool.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentSpec,
    ExecCommand,
    LocalRunner,
    LocalWorkspace,
    Message,
    RunRequest,
    ScriptedModelProvider,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
)
from cayu.providers import ModelStreamEvent


class LineCountTool(Tool):
    """Write text to a workspace file, then count its lines with the runner."""

    spec = ToolSpec(
        name="line_count",
        description="Write text to a file in the workspace and count its lines.",
        input_schema={
            "type": "object",
            "properties": {"filename": {"type": "string"}, "text": {"type": "string"}},
            "required": ["filename", "text"],
            "additionalProperties": False,
        },
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        if ctx.workspace is None or ctx.runner is None:
            return ToolResult(
                content="line_count needs a workspace and a runner in the environment.",
                is_error=True,
            )
        filename, text = args["filename"], args["text"]

        # 1) Write a file through the workspace handle.
        await ctx.workspace.write_bytes(filename, text.encode())
        # 2) Run a command through the runner handle.
        result = await ctx.runner.exec(ExecCommand.process("wc", "-l", filename))
        # 3) Read the file back through the workspace handle. read_bytes returns a
        # WorkspaceReadResult (content bytes + total_bytes + truncated), not raw bytes.
        read_back = await ctx.workspace.read_bytes(filename)

        return ToolResult(
            content=f"wc -l said: {result.stdout.strip()}",
            structured={"exit_code": result.exit_code, "bytes_read_back": read_back.total_bytes},
        )


async def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="cayu_custom_tool_"))
    app = CayuApp()
    app.register_provider(
        ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="c1",
                        name="line_count",
                        arguments={"filename": "notes.txt", "text": "one\ntwo\nthree\n"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("counted"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        ),
        default=True,
    )
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            workspace=LocalWorkspace(root, workspace_id="ws"),
            runner=LocalRunner(root),
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="scripted-model"), tools=[LineCountTool()])

    async for event in app.run(
        RunRequest(
            agent_name="assistant",
            session_id="demo",
            environment_name="local",
            messages=[Message.text("user", "count the lines")],
        )
    ):
        print(event.type, event.tool_name or "-", str(event.payload)[:120])


if __name__ == "__main__":
    asyncio.run(main())
