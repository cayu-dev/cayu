from __future__ import annotations

import asyncio
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentSpec,
    ListFilesTool,
    LocalWorkspace,
    Message,
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
                        "content": "updated in bound workspace",
                    },
                ),
                ModelStreamEvent.tool_call(
                    id="call_new",
                    name="write_file",
                    arguments={
                        "path": "notes/output.txt",
                        "content": "created in bound workspace",
                    },
                ),
                ModelStreamEvent.tool_call(
                    id="call_list",
                    name="list_files",
                    arguments={"pattern": "**/*.txt"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("sync binding finished"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        for event in self._batches[len(self.requests) - 1]:
            yield event


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="cayu-sync-binding-") as directory:
        root = Path(directory)
        source_root = root / "source"
        target_root = root / "target"
        source_root.mkdir()
        target_root.mkdir()
        (source_root / "notes").mkdir()
        (source_root / "notes" / "input.txt").write_text(
            "original source file",
            encoding="utf-8",
        )
        (target_root / "stale.txt").write_text("removed during bind", encoding="utf-8")

        source = LocalWorkspace(source_root, workspace_id="source-workspace")
        target = LocalWorkspace(target_root, workspace_id="bound-workspace")
        binding = SyncBinding(target_workspace=target, path="/workspace")

        provider = FakeProvider()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="sync-local"),
                workspace=source,
                binding=binding,
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[ReadFileTool(), WriteFileTool(), ListFilesTool()],
        )

        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id="demo_sync_binding_local",
                messages=[Message.text("user", "inspect and update the workspace")],
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
        print("target_stale_exists", (target_root / "stale.txt").exists())


if __name__ == "__main__":
    asyncio.run(main())
