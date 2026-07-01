"""Environment factory example: local workspace + local runner (native binding).

Usage:
    uv sync --extra dev
    PYTHONPATH=src .venv/bin/python examples/environments/local_native.py

API-key-free. The simplest ``EnvironmentFactory``: it builds a fresh ``LocalWorkspace`` +
``LocalRunner`` per session (each session gets its own directory), joined by
``NativeBinding`` — compute and files already share a filesystem, so the binding is a
pass-through. This is the no-surprises baseline for the environment-factory pattern.
Register it with ``app.register_environment_factory`` instead of a single static
``register_environment`` when each session needs its own freshly-provisioned environment.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from tempfile import TemporaryDirectory

from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentFactory,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
    EnvironmentSpec,
    ListFilesTool,
    LocalRunner,
    LocalWorkspace,
    Message,
    NativeBinding,
    RunRequest,
    WriteFileTool,
)
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent


class FakeProvider(ModelProvider):
    name = "fake"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self._batches = [
            # turn 1: the model asks to write a file then list files
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="write_file",
                    arguments={"path": "notes/result.txt", "content": "native ok"},
                ),
                ModelStreamEvent.tool_call(
                    id="call_2", name="list_files", arguments={"pattern": "**/*.txt"}
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            # turn 2: the model finishes after seeing the tool results
            [
                ModelStreamEvent.text_delta("local native environment finished"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        for event in self._batches[len(self.requests) - 1]:
            yield event


class LocalNativeFactory(EnvironmentFactory):
    """Provision a fresh per-session local environment (workspace + runner + native binding)."""

    def __init__(self, base_root: Path) -> None:
        self._base_root = base_root

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        root = self._base_root / request.session_id
        root.mkdir(parents=True, exist_ok=True)
        return EnvironmentFactoryResult(
            environment=Environment(
                EnvironmentSpec(name=request.environment_name, metadata={"kind": "local"}),
                workspace=LocalWorkspace(root, workspace_id=f"ws-{request.session_id}"),
                runner=LocalRunner(root),
                binding=NativeBinding(),
            ),
            metadata={"root": str(root)},
        )


async def main() -> None:
    with TemporaryDirectory(prefix="cayu-local-native-") as base:
        app = CayuApp()
        app.register_provider(FakeProvider(), default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="local"),
            LocalNativeFactory(Path(base)),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[WriteFileTool(), ListFilesTool()],
        )

        # The event stream shows the environment/binding lifecycle events interleaved with the
        # write_file and list_files tool events, ending in session.completed.
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id="demo_local_native",
                messages=[Message.text("user", "write and list")],
            )
        ):
            print(event.type, event.environment_name or "-", event.tool_name or "-")


if __name__ == "__main__":
    asyncio.run(main())
