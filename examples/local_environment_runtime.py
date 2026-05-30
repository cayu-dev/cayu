from __future__ import annotations

import asyncio
import sys
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentSpec,
    Event,
    EventType,
    ExecCommandTool,
    ListFilesTool,
    LocalRunner,
    LocalWorkspace,
    Message,
    ReadFileTool,
    RunRequest,
    WriteFileTool,
)
from cayu.providers import (
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
    ModelStreamEventType,
)


class FakeProvider(ModelProvider):
    name = "fake"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self._batches = [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="write_file",
                    arguments={"path": "notes/result.txt", "content": "local ok"},
                ),
                ModelStreamEvent.tool_call(
                    id="call_2",
                    name="exec_command",
                    arguments={
                        "argv": [
                            sys.executable,
                            "-c",
                            (
                                "from pathlib import Path; "
                                "print(Path('notes/result.txt').read_text())"
                            ),
                        ],
                    },
                ),
                ModelStreamEvent.tool_call(
                    id="call_3",
                    name="list_files",
                    arguments={"pattern": "**/*.txt"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("local environment finished"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        for event in self._batches[len(self.requests) - 1]:
            yield event

    def to_event(
        self,
        stream_event: ModelStreamEvent,
        *,
        session_id: str,
        agent_name: str | None = None,
    ) -> Event:
        if stream_event.type == ModelStreamEventType.TEXT_DELTA:
            return Event(
                type=EventType.MODEL_TEXT_DELTA,
                session_id=session_id,
                agent_name=agent_name,
                payload={"delta": stream_event.delta},
            )
        if stream_event.type == ModelStreamEventType.COMPLETED:
            return Event(
                type=EventType.MODEL_COMPLETED,
                session_id=session_id,
                agent_name=agent_name,
                payload=stream_event.payload,
            )
        return Event(
            type=f"custom.provider.{stream_event.type}",
            session_id=session_id,
            agent_name=agent_name,
            payload=stream_event.payload,
        )


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="cayu-local-env-") as directory:
        root = Path(directory)
        workspace = LocalWorkspace(root, workspace_id="local-demo")
        runner = LocalRunner(root)

        provider = FakeProvider()
        app = CayuApp()
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local-dev", metadata={"kind": "local"}),
                workspace=workspace,
                runner=runner,
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
                session_id="demo_local_environment",
                messages=[Message.text("user", "write and run")],
            )
        ):
            print(
                event.type,
                event.environment_name or "-",
                event.tool_name or "-",
                event.payload,
            )

        print("workspace_files", list((await workspace.list("**/*")).paths))


if __name__ == "__main__":
    asyncio.run(main())
