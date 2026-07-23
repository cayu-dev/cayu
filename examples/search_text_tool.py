from __future__ import annotations

import asyncio
import shutil
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentSpec,
    LocalRunner,
    LocalWorkspace,
    Message,
    ReadFileTool,
    RunRequest,
    SearchTextTool,
)
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent


class FakeProvider(ModelProvider):
    """Drive the recommended search progression without a model API call."""

    name = "fake"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self._batches = [
            [
                ModelStreamEvent.tool_call(
                    id="locate",
                    name="search_text",
                    arguments={
                        "pattern": "load_config",
                        "mode": "files",
                        "glob": "*.py",
                    },
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="inspect",
                    name="search_text",
                    arguments={
                        "pattern": "load_config",
                        "path": "src",
                        "mode": "content",
                        "limit": 20,
                    },
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="expand",
                    name="read_file",
                    arguments={"path": "src/config.py", "max_bytes": 256},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("Located and inspected src/config.py."),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        for event in self._batches[len(self.requests) - 1]:
            yield event


async def main() -> None:
    if shutil.which("rg") is None:
        print("Install ripgrep in the runner image to use SearchTextTool.")
        return

    with tempfile.TemporaryDirectory(prefix="cayu-search-text-") as directory:
        root = Path(directory)
        source = root / "src" / "config.py"
        source.parent.mkdir()
        source.write_text(
            "def load_config(path: str) -> dict:\n    return {'path': path, 'loaded': True}\n"
        )

        provider = FakeProvider()
        app = CayuApp()
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local-search"),
                workspace=LocalWorkspace(root, workspace_id="search-demo"),
                runner=LocalRunner(root),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[SearchTextTool(), ReadFileTool()],
        )

        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id="demo_search_text",
                messages=[Message.text("user", "Find and inspect load_config.")],
            )
        ):
            print(event.type, event.tool_name or "-", event.payload)


if __name__ == "__main__":
    asyncio.run(main())
