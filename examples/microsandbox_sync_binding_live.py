from __future__ import annotations

import asyncio
import importlib.util
import os
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentSpec,
    ExecCommandTool,
    ListFilesTool,
    LocalWorkspace,
    Message,
    MicrosandboxRunner,
    MicrosandboxWorkspace,
    ReadFileTool,
    RunRequest,
    SyncBinding,
    WriteFileTool,
)
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent


class FakeProvider(ModelProvider):
    name = "fake"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self._batches = [
            [
                ModelStreamEvent.tool_call(
                    id="call_read",
                    name="read_file",
                    arguments={"path": "notes/input.txt"},
                ),
                ModelStreamEvent.tool_call(
                    id="call_write",
                    name="write_file",
                    arguments={
                        "path": "notes/input.txt",
                        "content": "updated in microsandbox workspace",
                    },
                ),
                ModelStreamEvent.tool_call(
                    id="call_new",
                    name="write_file",
                    arguments={
                        "path": "notes/output.txt",
                        "content": "created in microsandbox workspace",
                    },
                ),
                ModelStreamEvent.tool_call(
                    id="call_exec",
                    name="exec_command",
                    arguments={
                        "kind": "process",
                        "argv": ["cat", "notes/input.txt"],
                    },
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("microsandbox sync binding finished"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        for event in self._batches[len(self.requests) - 1]:
            yield event


async def main() -> None:
    if importlib.util.find_spec("microsandbox") is None:
        print(
            "Install the optional microsandbox package to run this live "
            "Microsandbox SyncBinding example: `pip install cayu[microsandbox]`."
        )
        return

    sandbox_name = os.environ.get("CAYU_MICROSANDBOX_NAME", "cayu-sync-binding-live")
    image = os.environ.get("CAYU_MICROSANDBOX_IMAGE", "python:3.13-alpine")

    print(f"sandbox_name {sandbox_name}")
    print(f"image {image}")
    print("creating sandbox")

    with tempfile.TemporaryDirectory(prefix="cayu-microsandbox-sync-") as directory:
        source_root = Path(directory) / "source"
        source_root.mkdir()
        (source_root / "notes").mkdir()
        (source_root / "notes" / "input.txt").write_text(
            "original source file",
            encoding="utf-8",
        )

        async with await MicrosandboxRunner.create(
            sandbox_name,
            image=image,
            replace=True,
            close_action="remove",
        ) as runner:
            print("sandbox ready")

            source = LocalWorkspace(source_root, workspace_id="source-workspace")
            target = MicrosandboxWorkspace(
                runner,
                workspace_id="microsandbox-bound-workspace",
            )
            binding = SyncBinding(target_workspace=target, path=runner.default_cwd)
            provider = FakeProvider()
            app = CayuApp(enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_environment(
                Environment(
                    EnvironmentSpec(name="microsandbox-sync-live"),
                    workspace=source,
                    runner=runner,
                    binding=binding,
                ),
                default=True,
            )
            app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[
                    ReadFileTool(),
                    WriteFileTool(),
                    ListFilesTool(),
                    ExecCommandTool(),
                ],
            )

            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="demo_microsandbox_sync_binding",
                    messages=[Message.text("user", "sync and update the workspace")],
                )
            ):
                print(
                    event.type,
                    event.environment_name or "-",
                    event.tool_name or "-",
                    event.payload,
                )

            print("source_files", (await source.list("**/*.txt")).paths)
            print(
                "source_input",
                (source_root / "notes" / "input.txt").read_text(encoding="utf-8"),
            )
            print(
                "source_output",
                (source_root / "notes" / "output.txt").read_text(encoding="utf-8"),
            )
            print("model_requests", len(provider.requests))
            print("closing sandbox")

    print("completed")


if __name__ == "__main__":
    asyncio.run(main())
