from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from cayu import (
    AgentSpec,
    AnthropicProvider,
    CayuApp,
    Environment,
    EnvironmentSpec,
    ExecCommandTool,
    ListFilesTool,
    LocalRunner,
    LocalWorkspace,
    Message,
    ReadFileTool,
    RunRequest,
    WriteFileTool,
)


async def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY to run this live Anthropic example.")
        return

    model = os.environ.get("CAYU_ANTHROPIC_MODEL", "claude-sonnet-4-6")
    root = (
        Path(__file__).resolve().parents[1]
        / ".examples-workspaces"
        / "anthropic-local-tools"
    )
    root.mkdir(parents=True, exist_ok=True)
    workspace = LocalWorkspace(root, workspace_id="anthropic-local-demo")
    runner = LocalRunner(root, inherit_env=False)

    print("workspace_root", root)

    app = CayuApp()
    app.register_provider(AnthropicProvider(), default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local-dev", metadata={"kind": "local"}),
            workspace=workspace,
            runner=runner,
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(
            name="assistant",
            model=model,
            system_prompt=(
                "You are testing Cayu local tools. Use tools when needed. "
                "Do not write progress narration before tool calls. Use exactly "
                "one tool call per assistant turn. Cayu currently executes tool "
                "calls sequentially, so never say work ran simultaneously, in "
                "parallel, concurrently, or at the same time. Keep the final "
                "answer short."
            ),
        ),
        tools=[
            WriteFileTool(),
            ReadFileTool(),
            ListFilesTool(),
            ExecCommandTool(),
        ],
    )

    request = RunRequest(
        agent_name="assistant",
        session_id="demo_anthropic_local_tools",
        messages=[
            Message.text(
                "user",
                (
                    "Create notes/result.txt with the text 'anthropic ok'. "
                    "After that tool result is returned, list txt files. "
                    "After that tool result is returned, run a process command "
                    "that prints the Python executable path using "
                    f"{sys.executable!r}. Do not describe progress until all "
                    "tool calls are complete."
                ),
            )
        ],
    )
    async for event in app.run(request):
        print(
            event.type,
            event.environment_name or "-",
            event.tool_name or "-",
            event.payload,
        )

    print("workspace_files", list((await workspace.list("**/*")).paths))


if __name__ == "__main__":
    asyncio.run(main())
