from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator

from _live_checks import require_equal
from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentSpec,
    ExecCommandTool,
    Message,
    MicrosandboxRunner,
    RunRequest,
)
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent


class FakeProvider(ModelProvider):
    """Deterministic provider that calls exec_command through Cayu runtime."""

    name = "fake"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self._batches = [
            [
                ModelStreamEvent.tool_call(
                    id="call_pwd",
                    name="exec_command",
                    arguments={"kind": "process", "argv": ["pwd"]},
                ),
                ModelStreamEvent.tool_call(
                    id="call_env",
                    name="exec_command",
                    arguments={
                        "kind": "process",
                        "argv": [
                            "sh",
                            "-c",
                            (
                                'if [ -n "$CAYU_HOST_SECRET_SHOULD_NOT_LEAK" ]; '
                                "then echo visible; else echo hidden; fi"
                            ),
                        ],
                    },
                ),
                ModelStreamEvent.tool_call(
                    id="call_explicit_env",
                    name="exec_command",
                    arguments={
                        "kind": "process",
                        "argv": ["sh", "-c", "printf '%s\\n' \"$CAYU_EXPLICIT_ENV\""],
                        "env": {"CAYU_EXPLICIT_ENV": "visible"},
                    },
                ),
                ModelStreamEvent.tool_call(
                    id="call_shell",
                    name="exec_command",
                    arguments={
                        "kind": "shell",
                        "shell": "printf abcdef; printf uvwxyz >&2",
                        "max_output_bytes": 3,
                    },
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("microsandbox runtime finished"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        for event in self._batches[len(self.requests) - 1]:
            yield event


async def main() -> None:
    sandbox_name = os.environ.get("CAYU_MICROSANDBOX_NAME", "cayu-runtime-live")
    image = os.environ.get("CAYU_MICROSANDBOX_IMAGE", "alpine")

    os.environ["CAYU_HOST_SECRET_SHOULD_NOT_LEAK"] = "hidden"

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

        provider = FakeProvider()
        app = CayuApp()
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="microsandbox-live", metadata={"kind": "sandbox"}),
                runner=runner,
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[ExecCommandTool()],
        )

        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id="demo_microsandbox_runtime",
                messages=[Message.text("user", "run sandbox commands")],
            )
        ):
            print(
                event.type,
                event.environment_name or "-",
                event.tool_name or "-",
                event.payload,
            )

        require_equal(len(provider.requests), 2, "model_requests")
        print("model_requests", len(provider.requests))
        print("closing sandbox")

    print("completed")


if __name__ == "__main__":
    asyncio.run(main())
