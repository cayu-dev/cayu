from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

from cayu import (
    AgentSpec,
    CayuApp,
    DockerRunner,
    Environment,
    EnvironmentSpec,
    ExecCommand,
    ExecCommandTool,
    ListFilesTool,
    LocalWorkspace,
    Message,
    ReadFileTool,
    RunnerWorkspace,
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
                        "content": "updated in docker workspace",
                    },
                ),
                ModelStreamEvent.tool_call(
                    id="call_new",
                    name="write_file",
                    arguments={
                        "path": "notes/output.txt",
                        "content": "created in docker workspace",
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
                ModelStreamEvent.text_delta("docker sync binding finished"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        for event in self._batches[len(self.requests) - 1]:
            yield event


async def main() -> None:
    docker_path = os.environ.get("CAYU_DOCKER_PATH") or shutil.which("docker")
    if docker_path is None:
        print("Install Docker or set CAYU_DOCKER_PATH to run this live Docker example.")
        return

    container_name = os.environ.get("CAYU_DOCKER_NAME", "cayu-sync-binding-live")
    image = os.environ.get("CAYU_DOCKER_IMAGE", "python:3.13-alpine")

    print(f"container_name {container_name}")
    print(f"image {image}")
    print("creating container")

    with tempfile.TemporaryDirectory(prefix="cayu-docker-sync-") as directory:
        source_root = Path(directory) / "source"
        source_root.mkdir()
        (source_root / "notes").mkdir()
        (source_root / "notes" / "input.txt").write_text(
            "original source file",
            encoding="utf-8",
        )

        async with await DockerRunner.create(
            container_name,
            image=image,
            docker_path=docker_path,
            replace=True,
            close_action="remove",
        ) as runner:
            python_check = await runner.exec(ExecCommand.process("python3", "--version"))
            if python_check.exit_code != 0:
                raise RuntimeError(
                    "RunnerWorkspace requires python3 inside the Docker container. "
                    "Use a Python image or install python3 with setup_commands."
                )

            print("container ready")

            source = LocalWorkspace(source_root, workspace_id="source-workspace")
            target = RunnerWorkspace(
                runner,
                workspace_id="docker-bound-workspace",
            )
            binding = SyncBinding(target_workspace=target, path=runner.default_cwd)
            provider = FakeProvider()
            app = CayuApp(enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_environment(
                Environment(
                    EnvironmentSpec(name="docker-sync-live"),
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
                    session_id="demo_docker_sync_binding",
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
            print("closing container")

    print("completed")


if __name__ == "__main__":
    asyncio.run(main())
