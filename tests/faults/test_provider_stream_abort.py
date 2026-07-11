from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    Message,
    RetryPolicy,
    RunRequest,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
)
from cayu.providers import ChatCompletionsProvider
from cayu.runtime import SessionStatus
from cayu.storage import SQLiteSessionStore


class RecordingSideEffectTool(Tool):
    spec = ToolSpec(
        name="side_effect",
        description="Record whether an interrupted provider stream executes a tool.",
        input_schema={"type": "object", "properties": {}},
    )

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, Any]] = []

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        self.calls.append(args)
        return ToolResult(content="executed")


class AbortingChatCompletionsServer:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.handler_errors: list[BaseException] = []
        self._server: asyncio.Server | None = None

    async def __aenter__(self) -> AbortingChatCompletionsServer:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        return self

    async def __aexit__(self, *args: object) -> None:
        assert self._server is not None
        self._server.close()
        await self._server.wait_closed()

    @property
    def base_url(self) -> str:
        assert self._server is not None
        socket = self._server.sockets[0]
        port = socket.getsockname()[1]
        return f"http://127.0.0.1:{port}/v1"

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            header_bytes = await reader.readuntil(b"\r\n\r\n")
            headers = header_bytes.decode("latin-1").split("\r\n")
            content_length = next(
                int(line.partition(":")[2].strip())
                for line in headers
                if line.lower().startswith("content-length:")
            )
            body = await reader.readexactly(content_length)
            decoded = json.loads(body)
            assert isinstance(decoded, dict)
            self.requests.append(decoded)

            event = {
                "id": "chatcmpl_abort",
                "model": "abort-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": "partial response",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_abort",
                                    "type": "function",
                                    "function": {"name": "side_effect", "arguments": "{}"},
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ],
            }
            sse = f"data: {json.dumps(event)}\n\n".encode()
            response = (
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/event-stream\r\n"
                b"Transfer-Encoding: chunked\r\n"
                b"Connection: close\r\n\r\n" + f"{len(sse):X}\r\n".encode() + sse + b"\r\n"
            )
            writer.write(response)
            # drain() queues the valid delta before the loopback RST; the test's
            # MODEL_TEXT_DELTA assertion fails closed if a platform reverses delivery.
            await writer.drain()
            writer.transport.abort()
        except BaseException as exc:
            self.handler_errors.append(exc)
            writer.transport.abort()


@pytest.mark.anyio
async def test_real_provider_transport_abort_fails_durably_without_tool_execution(
    tmp_path,
) -> None:
    store_path = tmp_path / "provider-abort.sqlite"
    store = SQLiteSessionStore(store_path)
    tool = RecordingSideEffectTool()

    async with AbortingChatCompletionsServer() as endpoint:
        provider = ChatCompletionsProvider(
            api_key="local-contract-key",
            name="abort_chat",
            base_url=endpoint.base_url,
            allow_http=True,
            stream_include_usage=False,
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(
                name="abort-assistant",
                model="abort-model",
                provider_name="abort_chat",
            ),
            tools=[tool],
        )
        try:
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="abort-assistant",
                        session_id="provider-stream-abort",
                        max_steps=1,
                        messages=[Message.text("user", "Attempt the side effect.")],
                        retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
                    )
                )
            ]
        finally:
            await provider.aclose()

    assert endpoint.handler_errors == []
    assert len(endpoint.requests) == 2
    request = endpoint.requests[0]
    assert request["stream"] is True
    assert request["model"] == "abort-model"
    assert request["tools"][0]["function"]["name"] == "side_effect"

    event_types = [event.type for event in events]
    assert EventType.MODEL_TEXT_DELTA in event_types
    assert event_types.count(EventType.MODEL_RETRY) == 1
    assert EventType.MODEL_COMPLETED not in event_types
    assert EventType.TOOL_CALL_STARTED not in event_types
    assert EventType.TOOL_CALL_COMPLETED not in event_types
    assert event_types[-1] == EventType.SESSION_FAILED
    assert tool.calls == []

    model_error = next(event for event in events if event.type == EventType.MODEL_ERROR)
    assert model_error.payload["error_type"] == "ChatCompletionsAPIError"
    assert model_error.payload["provider"] == "chat_completions"
    assert model_error.payload["retryable"] is True
    assert isinstance(model_error.payload["provider_error_type"], str)
    model_retry = next(event for event in events if event.type == EventType.MODEL_RETRY)
    assert model_retry.payload["reason"] == "connection"

    await store.close()
    reopened = SQLiteSessionStore(store_path)
    try:
        session = await reopened.load("provider-stream-abort")
        persisted_events = await reopened.load_events("provider-stream-abort")
        transcript = await reopened.load_transcript("provider-stream-abort")
    finally:
        await reopened.close()

    assert session is not None
    assert session.status == SessionStatus.FAILED
    assert persisted_events[-1].type == EventType.SESSION_FAILED
    assert persisted_events[-1].payload == events[-1].payload
    assert (
        next(event for event in persisted_events if event.type == EventType.MODEL_ERROR).payload[
            "error_type"
        ]
        == "ChatCompletionsAPIError"
    )
    assert EventType.MODEL_COMPLETED not in [event.type for event in persisted_events]
    assert EventType.TOOL_CALL_STARTED not in [event.type for event in persisted_events]
    assert [message.role for message in transcript] == ["user"]
