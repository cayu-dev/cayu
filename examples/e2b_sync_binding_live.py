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
    E2BRunner,
    E2BWorkspace,
    Environment,
    EnvironmentSpec,
    ExecCommandTool,
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
                        "content": "updated in e2b workspace",
                    },
                ),
                ModelStreamEvent.tool_call(
                    id="call_new",
                    name="write_file",
                    arguments={
                        "path": "notes/output.txt",
                        "content": "created in e2b workspace",
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
                ModelStreamEvent.text_delta("e2b sync binding finished"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        for event in self._batches[len(self.requests) - 1]:
            yield event


async def main() -> None:
    if importlib.util.find_spec("e2b") is None:
        print(
            "Install the optional e2b package to run this live E2B "
            "SyncBinding example: `pip install cayu[e2b]`."
        )
        return

    if not os.environ.get("E2B_API_KEY"):
        print("Set E2B_API_KEY to run this live E2B SyncBinding example.")
        return

    template = os.environ.get("CAYU_E2B_TEMPLATE")
    sandbox_timeout_s = int(os.environ.get("CAYU_E2B_SANDBOX_TIMEOUT_S", "300"))
    print(f"template {template or '<e2b-default>'}")
    print("creating sandbox")

    with tempfile.TemporaryDirectory(prefix="cayu-e2b-sync-") as directory:
        source_root = Path(directory) / "source"
        source_root.mkdir()
        (source_root / "notes").mkdir()
        (source_root / "notes" / "input.txt").write_text(
            "original source file",
            encoding="utf-8",
        )

        async with await E2BRunner.create(
            template=template,
            sandbox_timeout_s=sandbox_timeout_s,
            close_action="kill",
        ) as runner:
            print(f"sandbox_id {runner.sandbox_id}")
            print("sandbox ready")

            source = LocalWorkspace(source_root, workspace_id="source-workspace")
            target = E2BWorkspace(runner, workspace_id="e2b-bound-workspace")
            binding = SyncBinding(target_workspace=target, path=runner.default_cwd)
            provider = FakeProvider()
            app = CayuApp(enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_environment(
                Environment(
                    EnvironmentSpec(name="e2b-sync-live"),
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
                    session_id="demo_e2b_sync_binding",
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
