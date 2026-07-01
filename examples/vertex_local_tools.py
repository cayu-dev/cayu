from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from cayu import (
    AgentSpec,
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
    VertexProvider,
    WriteFileTool,
)


async def main() -> None:
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        print(
            "Set GOOGLE_CLOUD_PROJECT (and authenticate with "
            "`gcloud auth application-default login`, plus `pip install cayu[vertex]`) "
            "to run this live Vertex AI example."
        )
        return

    model = os.environ.get("CAYU_VERTEX_MODEL", "claude-sonnet-4-6")
    region = os.environ.get("CAYU_VERTEX_REGION", "global")
    root = Path(__file__).resolve().parents[1] / ".examples-workspaces" / "vertex-local-tools"
    root.mkdir(parents=True, exist_ok=True)
    workspace = LocalWorkspace(root, workspace_id="vertex-local-demo")
    runner = LocalRunner(root, inherit_env=False)

    print("workspace_root", root)

    app = CayuApp()
    # No explicit credentials -> Application Default Credentials (ADC).
    app.register_provider(VertexProvider(project_id=project_id, region=region), default=True)
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
        session_id="demo_vertex_local_tools",
        messages=[
            Message.text(
                "user",
                (
                    "Create notes/result.txt with the text 'vertex ok'. "
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
