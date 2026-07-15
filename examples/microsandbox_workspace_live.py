from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator

from _live_checks import require_equal
from _workspace_conformance import (
    verify_portable_workspace_path_safety,
    verify_portable_workspace_round_trip,
)
from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentSpec,
    ExecCommandTool,
    ListFilesTool,
    Message,
    MicrosandboxRunner,
    MicrosandboxWorkspace,
    ReadFileTool,
    RunRequest,
    WriteFileTool,
)
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent


class FakeProvider(ModelProvider):
    """Deterministic provider proving file tools and exec share one sandbox workspace."""

    name = "fake"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self._batches = [
            [
                ModelStreamEvent.tool_call(
                    id="call_write",
                    name="write_file",
                    arguments={"path": "notes/result.txt", "content": "sandbox workspace"},
                ),
                ModelStreamEvent.tool_call(
                    id="call_read",
                    name="read_file",
                    arguments={"path": "notes/result.txt"},
                ),
                ModelStreamEvent.tool_call(
                    id="call_list",
                    name="list_files",
                    arguments={"pattern": "**/*.txt"},
                ),
                ModelStreamEvent.tool_call(
                    id="call_cat",
                    name="exec_command",
                    arguments={
                        "kind": "process",
                        "argv": ["cat", "notes/result.txt"],
                    },
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("microsandbox workspace finished"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        for event in self._batches[len(self.requests) - 1]:
            yield event


async def main() -> None:
    sandbox_name = os.environ.get("CAYU_MICROSANDBOX_NAME", "cayu-workspace-live")
    image = os.environ.get("CAYU_MICROSANDBOX_IMAGE", "python:3.13-alpine")

    print(f"sandbox_name {sandbox_name}")
    print(f"image {image}")
    print("creating sandbox")

    async with await MicrosandboxRunner.create(
        sandbox_name,
        image=image,
        replace=True,
        close_action="remove",
    ) as runner:
        print("sandbox ready")

        workspace = MicrosandboxWorkspace(
            runner,
            workspace_id="sandbox-workspace",
        )
        await verify_portable_workspace_round_trip(workspace, adapter="microsandbox-live")
        await verify_portable_workspace_path_safety(workspace, adapter="microsandbox-live")
        print("workspace_conformance portable-round-trip,path-safety")
        provider = FakeProvider()
        app = CayuApp()
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="microsandbox-workspace-live", metadata={"kind": "sandbox"}),
                runner=runner,
                workspace=workspace,
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[
                WriteFileTool(),
                ReadFileTool(),
                ListFilesTool(),
                ExecCommandTool(),
            ],
        )

        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id="demo_microsandbox_workspace",
                messages=[Message.text("user", "write, read, list, and cat the sandbox file")],
            )
        ):
            print(
                event.type,
                event.environment_name or "-",
                event.tool_name or "-",
                event.payload,
            )

        direct_read = await workspace.read_bytes("notes/result.txt")
        direct_text = direct_read.content.decode("utf-8")
        require_equal(direct_text, "sandbox workspace", "direct_workspace_read")
        require_equal(len(provider.requests), 2, "model_requests")
        print(f"direct_workspace_read {direct_text}")
        print("model_requests", len(provider.requests))
        print("closing sandbox")

    print("completed")


if __name__ == "__main__":
    asyncio.run(main())
