from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from cayu import (
    AgentSpec,
    CayuApp,
    Event,
    EventType,
    Message,
    RunRequest,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
)
from cayu.providers import (
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
    ModelStreamEventType,
)


class FakeProvider(ModelProvider):
    """Deterministic provider that requests one tool call, then returns text."""

    name = "fake"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self._batches = [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="echo",
                    arguments={"text": "hello from tool"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("final answer after tool"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        batch_index = len(self.requests) - 1
        for event in self._batches[batch_index]:
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


class EchoTool(Tool):
    spec = ToolSpec(
        name="echo",
        description="Echo text.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(
            content=args["text"],
            structured={
                "agent": ctx.agent_name,
                "echoed": args["text"],
            },
        )


async def main() -> None:
    provider = FakeProvider()
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
    )

    request = RunRequest(
        agent_name="assistant",
        session_id="demo_echo",
        messages=[Message.text("user", "echo something")],
    )

    async for event in app.run(request):
        print(event.type, event.tool_name or "-", event.payload)

    print("model_requests", len(provider.requests))
    print("second_request_last_message", provider.requests[1].messages[-1].model_dump())


if __name__ == "__main__":
    asyncio.run(main())
