"""OpenAI Responses over a persistent WebSocket, against an in-process fake server.

The fake server keeps each connection's latest completed response the way the
Responses WebSocket protocol does, and records, for every request, the bytes it
received and the full input the model would see after resolving
``previous_response_id``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, cast

import pytest
from websockets.asyncio.server import ServerConnection, serve

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.messages import Message, MessageRole, ProviderStatePart
from cayu.providers import OpenAIProvider, WebSocketOpenAITransport
from cayu.providers.base import ModelRequest, ModelStreamEventType
from cayu.providers.openai import (
    OpenAIAPIError,
    OpenAIContextOverflowError,
    _execution_profile_material,
)
from cayu.sessions.requests import RunRequest
from cayu.tools.base import Tool, ToolContext, ToolResult, ToolSpec

pytestmark = pytest.mark.anyio

_MODEL = "gpt-test"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@dataclass
class Received:
    connection: int
    bytes: int
    body: dict[str, Any]
    # The input the model sees: the body's input after resolving the chain.
    model_input: list[Any] | None


@dataclass
class FakeResponsesServer:
    """A Responses WebSocket endpoint that answers from a script of outputs."""

    outputs: list[list[dict[str, Any]]]
    statuses: list[str] = field(default_factory=list)
    empty_completed_output: bool = False
    close_after_each_response: bool = False
    forget_held_response: bool = False
    received: list[Received] = field(default_factory=list)
    handshake_headers: list[dict[str, str]] = field(default_factory=list)
    connections: int = 0

    async def handle(self, websocket: ServerConnection) -> None:
        connection = self.connections
        self.connections += 1
        assert websocket.request is not None
        self.handshake_headers.append(
            {name.lower(): value for name, value in websocket.request.headers.raw_items()}
        )
        held: tuple[str, list[Any], list[Any]] | None = None
        async for message in websocket:
            assert isinstance(message, str)
            body = json.loads(message)
            assert body.pop("type") == "response.create"
            previous = body.get("previous_response_id")
            if previous is not None:
                if self.forget_held_response:
                    self.forget_held_response = False
                    held = None
                if held is None or held[0] != previous:
                    self.received.append(Received(connection, len(message.encode()), body, None))
                    await websocket.send(json.dumps(_missing_previous_response(previous)))
                    continue
                model_input = [*held[1], *held[2], *body["input"]]
            else:
                model_input = list(body["input"])
            self.received.append(Received(connection, len(message.encode()), body, model_input))
            response_id = f"resp_{len(self.received)}"
            output = self.outputs.pop(0)
            status = self.statuses.pop(0) if self.statuses else "completed"
            for event in _response_events(
                response_id,
                output,
                status=status,
                empty_completed_output=self.empty_completed_output,
            ):
                await websocket.send(json.dumps(event))
            held = (response_id, model_input, output) if status == "completed" else None
            if self.close_after_each_response:
                await websocket.close()
                return

    @property
    def sent_bodies(self) -> list[dict[str, Any]]:
        return [received.body for received in self.received]


def _missing_previous_response(response_id: str) -> dict[str, Any]:
    return {
        "type": "error",
        "status": 404,
        "error": {
            "type": "invalid_request_error",
            "code": "previous_response_not_found",
            "message": f"Previous response with id '{response_id}' not found.",
            "param": "previous_response_id",
        },
    }


def _response_events(
    response_id: str,
    output: list[dict[str, Any]],
    *,
    status: str = "completed",
    empty_completed_output: bool = False,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = [
        {
            "type": "response.created",
            "response": {"id": response_id, "model": _MODEL, "status": "in_progress"},
        }
    ]
    for index, item in enumerate(output):
        if item["type"] == "function_call":
            events.append(
                {
                    "type": "response.output_item.added",
                    "output_index": index,
                    "item": {**item, "arguments": "", "status": "in_progress"},
                }
            )
            events.append(
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": item["id"],
                    "output_index": index,
                    "name": item["name"],
                    "arguments": item["arguments"],
                }
            )
        elif item["type"] == "message":
            events.append(
                {
                    "type": "response.output_text.delta",
                    "item_id": item["id"],
                    "output_index": index,
                    "content_index": 0,
                    "delta": item["content"][0]["text"],
                }
            )
        events.append({"type": "response.output_item.done", "output_index": index, "item": item})
    terminal_type = f"response.{status}"
    response: dict[str, Any] = {
        "id": response_id,
        "model": _MODEL,
        "status": status,
        "output": [] if empty_completed_output else output,
        "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
    }
    if status == "incomplete":
        response["incomplete_details"] = {"reason": "max_output_tokens"}
    events.append({"type": terminal_type, "response": response})
    return events


def _reasoning(index: int) -> dict[str, Any]:
    return {
        "type": "reasoning",
        "id": f"rs_{index}",
        "summary": [],
        "encrypted_content": f"encrypted-{index}",
    }


def _call(index: int, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": f"fc_{index}",
        "call_id": f"call_{index}",
        "name": "read_page",
        "arguments": json.dumps(arguments, separators=(",", ":")),
        "status": "completed",
    }


def _message(index: int, text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "id": f"msg_{index}",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _tool_round_outputs(rounds: int) -> list[list[dict[str, Any]]]:
    outputs = [[_reasoning(index), _call(index, {"page": index})] for index in range(1, rounds)]
    outputs.append([_message(rounds, "All pages read.")])
    return outputs


class ReadPageTool(Tool):
    """Returns a fixed-size page, standing in for a file read in a coding round."""

    page_bytes = 4_000

    spec = ToolSpec(
        name="read_page",
        description="Read one page.",
        input_schema={
            "type": "object",
            "properties": {"page": {"type": "integer"}},
            "required": ["page"],
        },
    )

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        line = f"page {args['page']}: " + "lorem ipsum dolor sit amet " * 3
        return ToolResult(content=(line + "\n") * (self.page_bytes // (len(line) + 1)))


@asynccontextmanager
async def _running(server: FakeResponsesServer) -> AsyncIterator[str]:
    async with serve(server.handle, "127.0.0.1", 0) as websocket_server:
        port = websocket_server.sockets[0].getsockname()[1]
        yield f"http://127.0.0.1:{port}"


def _provider(base_url: str, transport: Any) -> OpenAIProvider:
    return OpenAIProvider(
        api_key="sk-test-websocket-key",
        base_url=base_url,
        allow_http=True,
        transport=transport,
    )


def _app(provider: OpenAIProvider) -> CayuApp:
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(
            name="reader",
            model=_MODEL,
            system_prompt="Read every page you are asked to read, one call at a time.",
        ),
        tools=[ReadPageTool()],
    )
    return app


async def _run_session(app: CayuApp, prompt: str) -> None:
    events = [
        event
        async for event in app.run(
            RunRequest(agent_name="reader", messages=[Message.text("user", prompt)])
        )
    ]
    assert events[-1].type == "session.completed", [event.type for event in events[-3:]]


async def _run_over_websocket(
    rounds: int, prompt: str, *, incremental_input: bool
) -> FakeResponsesServer:
    server = FakeResponsesServer(outputs=_tool_round_outputs(rounds))
    async with _running(server) as base_url:
        transport = WebSocketOpenAITransport(allow_http=True, incremental_input=incremental_input)
        provider = _provider(base_url, transport)
        try:
            await _run_session(_app(provider), prompt)
        finally:
            await provider.aclose()
    return server


def _without(body: Mapping[str, Any], *names: str) -> dict[str, Any]:
    return {name: value for name, value in body.items() if name not in names}


class _RecordingHttpTransport:
    """Stands in for the HTTP transport; streams the same scripted responses."""

    def __init__(self, outputs: list[list[dict[str, Any]]]) -> None:
        self.outputs = outputs
        self.payloads: list[dict[str, Any]] = []

    async def create_response(self, **_: Any) -> Mapping[str, Any]:
        raise AssertionError("not used")

    async def stream_response_events(self, *, payload: Mapping[str, Any], **_: Any):
        self.payloads.append(json.loads(json.dumps(payload)))
        response_id = f"resp_{len(self.payloads)}"
        for event in _response_events(response_id, self.outputs.pop(0)):
            yield event


async def test_tool_rounds_send_only_new_items_and_the_model_sees_the_same_input() -> None:
    prompt = "Read pages 1 to 3."
    over_http = _RecordingHttpTransport(_tool_round_outputs(4))
    await _run_session(_app(_provider("https://api.openai.com", over_http)), prompt)

    server = await _run_over_websocket(4, prompt, incremental_input=True)

    bodies = server.sent_bodies
    assert len(bodies) == len(over_http.payloads) == 4
    # One connection carried the whole session.
    assert {received.connection for received in server.received} == {0}
    assert "previous_response_id" not in bodies[0]
    for index in range(1, 4):
        assert bodies[index]["previous_response_id"] == f"resp_{index}"
        # Only the tool result is new: the reasoning and call are held by the server.
        assert [item["type"] for item in bodies[index]["input"]] == ["function_call_output"]

    # The model sees exactly what the HTTP transport sends, request by request,
    # with every other field unchanged.
    cache_keys = {payload["prompt_cache_key"] for payload in over_http.payloads}
    assert len(cache_keys) == 1
    websocket_cache_keys = {body["prompt_cache_key"] for body in bodies}
    assert len(websocket_cache_keys) == 1
    for http_payload, received in zip(over_http.payloads, server.received, strict=True):
        assert received.model_input == http_payload["input"]
        assert _without(received.body, "input", "previous_response_id", "prompt_cache_key") == (
            _without(http_payload, "input", "stream", "prompt_cache_key")
        )
        assert http_payload["store"] is False


async def test_full_input_mode_sends_what_http_sends() -> None:
    prompt = "Read pages 1 and 2."
    over_http = _RecordingHttpTransport(_tool_round_outputs(3))
    await _run_session(_app(_provider("https://api.openai.com", over_http)), prompt)

    server = await _run_over_websocket(3, prompt, incremental_input=False)

    assert {received.connection for received in server.received} == {0}
    for http_payload, body in zip(over_http.payloads, server.sent_bodies, strict=True):
        assert "previous_response_id" not in body
        assert _without(body, "prompt_cache_key") == _without(
            http_payload, "stream", "prompt_cache_key"
        )


async def test_bytes_sent_per_tool_round_drop_to_the_new_items() -> None:
    # A 120 KB opening context and a 4 KB tool result per round, about the shape
    # of a coding round that reads one file.
    prompt = "Review this module.\n" + ("def handler(event):\n    return event\n" * 3_300)
    rounds = 8
    full = await _run_over_websocket(rounds, prompt, incremental_input=False)
    incremental = await _run_over_websocket(rounds, prompt, incremental_input=True)

    full_bytes = [received.bytes for received in full.received]
    incremental_bytes = [received.bytes for received in incremental.received]
    assert len(full_bytes) == len(incremental_bytes) == rounds
    # The first request carries the whole context either way.
    assert incremental_bytes[0] == full_bytes[0]
    # Full requests grow with the transcript; incremental ones stay at one tool
    # result plus the request's fixed fields (instructions, tools, options).
    assert full_bytes == sorted(full_bytes)
    assert full_bytes[-1] > full_bytes[0] + (rounds - 2) * ReadPageTool.page_bytes
    for later in incremental_bytes[1:]:
        assert later < ReadPageTool.page_bytes * 2
    assert sum(incremental_bytes) * 4 < sum(full_bytes)
    # Both servers saw the same conversation.
    for full_received, incremental_received in zip(
        full.received, incremental.received, strict=True
    ):
        assert incremental_received.model_input == full_received.model_input


def _user(text: str) -> Message:
    return Message.text("user", text)


def _assistant(output: list[dict[str, Any]]) -> Message:
    return Message(
        role=MessageRole.ASSISTANT,
        content=tuple(ProviderStatePart(provider="openai", state=item) for item in output),
    )


def _request(*messages: Message, **fields: Any) -> ModelRequest:
    return ModelRequest(
        model=_MODEL,
        messages=[Message.text("system", "Be brief."), *messages],
        cache_affinity_key="lineage-1",
        **fields,
    )


async def _complete(provider: OpenAIProvider, request: ModelRequest) -> list[Any]:
    events = [event async for event in provider.stream(request)]
    assert events[-1].type is ModelStreamEventType.COMPLETED, events[-1]
    return events


_FIRST = [_message(1, "First answer.")]
_SECOND = [_message(2, "Second answer.")]


def _two_turns() -> tuple[ModelRequest, ModelRequest]:
    first = _request(_user("one"))
    second = _request(_user("one"), _assistant(_FIRST), _user("two"))
    return first, second


async def _two_requests(
    server: FakeResponsesServer,
    first: ModelRequest,
    second: ModelRequest,
    **transport_options: Any,
) -> FakeResponsesServer:
    async with _running(server) as base_url:
        provider = _provider(
            base_url, WebSocketOpenAITransport(allow_http=True, **transport_options)
        )
        try:
            await _complete(provider, first)
            await _complete(provider, second)
        finally:
            await provider.aclose()
    return server


async def test_follow_up_turn_sends_new_items_with_previous_response_id() -> None:
    server = await _two_requests(FakeResponsesServer(outputs=[_FIRST, _SECOND]), *_two_turns())

    first, second = server.sent_bodies
    assert second["previous_response_id"] == "resp_1"
    assert second["input"] == [{"role": "user", "content": [{"type": "input_text", "text": "two"}]}]
    assert server.received[1].model_input == [*first["input"], *_FIRST, *second["input"]]
    assert second["prompt_cache_key"] == first["prompt_cache_key"] == "lineage-1"
    assert second["instructions"] == "Be brief."
    assert server.handshake_headers[0]["authorization"] == "Bearer sk-test-websocket-key"
    assert "content-type" not in server.handshake_headers[0]


async def test_output_items_sent_only_in_output_item_done_still_chain() -> None:
    server = await _two_requests(
        FakeResponsesServer(outputs=[_FIRST, _SECOND], empty_completed_output=True),
        *_two_turns(),
    )

    assert server.sent_bodies[1]["previous_response_id"] == "resp_1"


async def test_reconnect_sends_the_full_input() -> None:
    first, second = _two_turns()
    server = await _two_requests(
        FakeResponsesServer(outputs=[_FIRST, _SECOND], close_after_each_response=True),
        first,
        second,
    )

    assert [received.connection for received in server.received] == [0, 1]
    assert "previous_response_id" not in server.sent_bodies[1]
    assert server.received[1].model_input == [
        *server.received[0].body["input"],
        *_FIRST,
        {"role": "user", "content": [{"type": "input_text", "text": "two"}]},
    ]


async def test_expired_previous_response_is_resent_in_full_on_the_same_connection() -> None:
    server = FakeResponsesServer(outputs=[_FIRST, _SECOND])
    async with _running(server) as base_url:
        provider = _provider(base_url, WebSocketOpenAITransport(allow_http=True))
        first, second = _two_turns()
        try:
            await _complete(provider, first)
            server.forget_held_response = True
            events = await _complete(provider, second)
        finally:
            await provider.aclose()

    # The caller saw one ordinary response; the refused delta never reached it.
    assert not [event for event in events if event.type is ModelStreamEventType.ERROR]
    assert [received.connection for received in server.received] == [0, 0, 0]
    refused, resent = server.sent_bodies[1:]
    assert refused["previous_response_id"] == "resp_1"
    assert server.received[1].model_input is None
    assert "previous_response_id" not in resent
    assert len(resent["input"]) == 3


@pytest.mark.parametrize(
    "case",
    [
        "compacted",
        "redacted",
        "edited_earlier",
        "instructions_changed",
        "tools_changed",
        "options_changed",
    ],
)
async def test_changed_history_or_settings_send_the_full_input(case: str) -> None:
    first = _request(_user("one"))
    two = _user("two")
    if case == "compacted":
        second = _request(_user("Summary: the user said one."), two)
    elif case == "redacted":
        redacted = _message(1, "[REDACTED]")
        second = _request(_user("one"), _assistant([redacted]), two)
    elif case == "edited_earlier":
        second = _request(_user("one, edited"), _assistant(_FIRST), two)
    elif case == "instructions_changed":
        second = _request(_user("one"), _assistant(_FIRST), two)
        second.messages[0] = Message.text("system", "Be thorough.")
    elif case == "tools_changed":
        second = _request(
            _user("one"),
            _assistant(_FIRST),
            two,
            tools=[{"name": "lookup", "description": "Look up.", "input_schema": {}}],
        )
    else:
        second = _request(
            _user("one"), _assistant(_FIRST), two, options={"openai": {"max_output_tokens": 500}}
        )
    server = await _two_requests(FakeResponsesServer(outputs=[_FIRST, _SECOND]), first, second)

    assert [received.connection for received in server.received] == [0, 0]
    assert "previous_response_id" not in server.sent_bodies[1]
    assert server.received[1].model_input == server.sent_bodies[1]["input"]


async def test_incomplete_previous_response_is_not_chained() -> None:
    server = await _two_requests(
        FakeResponsesServer(outputs=[_FIRST, _SECOND], statuses=["incomplete", "completed"]),
        *_two_turns(),
    )

    assert "previous_response_id" not in server.sent_bodies[1]


async def test_full_input_mode_never_chains() -> None:
    server = await _two_requests(
        FakeResponsesServer(outputs=[_FIRST, _SECOND]), *_two_turns(), incremental_input=False
    )

    assert [received.connection for received in server.received] == [0, 0]
    assert "previous_response_id" not in server.sent_bodies[1]


async def test_request_the_provider_already_chained_is_sent_as_given() -> None:
    server = FakeResponsesServer(outputs=[_FIRST, _SECOND])
    async with _running(server) as base_url:
        provider = OpenAIProvider(
            api_key="sk-test-websocket-key",
            base_url=base_url,
            allow_http=True,
            reasoning_state="server",
            transport=WebSocketOpenAITransport(allow_http=True),
        )
        try:
            events = await _complete(provider, _request(_user("one")))
            completed = events[-1]
            held = tuple(
                ProviderStatePart(provider=state["provider"], state=state["state"])
                for state in completed.payload["provider_state"]
            )
            assistant = Message(role=MessageRole.ASSISTANT, content=held)
            await _complete(provider, _request(_user("one"), assistant, _user("two")))
        finally:
            await provider.aclose()

    second = server.sent_bodies[1]
    assert second["store"] is True
    assert second["previous_response_id"] == "resp_1"
    assert len(second["input"]) == 1


async def test_concurrent_calls_of_one_lineage_use_separate_connections() -> None:
    server = FakeResponsesServer(outputs=[_FIRST, _FIRST, _SECOND])
    async with _running(server) as base_url:
        provider = _provider(base_url, WebSocketOpenAITransport(allow_http=True))
        first, second = _two_turns()
        try:
            await asyncio.gather(_complete(provider, first), _complete(provider, first))
            await _complete(provider, second)
        finally:
            await provider.aclose()

    assert sorted(received.connection for received in server.received[:2]) == [0, 1]
    assert server.sent_bodies[2]["previous_response_id"] in {"resp_1", "resp_2"}


async def test_abandoned_stream_drops_its_connection() -> None:
    server = FakeResponsesServer(outputs=[_FIRST, _SECOND])
    async with _running(server) as base_url:
        provider = _provider(base_url, WebSocketOpenAITransport(allow_http=True))
        first, second = _two_turns()
        try:
            stream = cast("AsyncGenerator[Any, None]", provider.stream(first))
            async with aclosing(stream):
                async for _event in stream:
                    break
            await _complete(provider, second)
        finally:
            await provider.aclose()

    assert [received.connection for received in server.received] == [0, 1]
    assert "previous_response_id" not in server.sent_bodies[1]


async def test_refused_handshake_is_classified_like_the_http_response() -> None:
    from websockets.datastructures import Headers
    from websockets.http11 import Response

    def refuse(connection: ServerConnection, request: Any) -> Response:
        body = json.dumps(
            {
                "error": {
                    "type": "invalid_request_error",
                    "code": "invalid_api_key",
                    "message": "bad",
                }
            }
        ).encode()
        return Response(401, "Unauthorized", Headers({"content-type": "application/json"}), body)

    async def never(websocket: ServerConnection) -> None:  # pragma: no cover - refused first
        raise AssertionError("handshake should be refused")

    async with serve(never, "127.0.0.1", 0, process_request=refuse) as websocket_server:
        port = websocket_server.sockets[0].getsockname()[1]
        provider = _provider(f"http://127.0.0.1:{port}", WebSocketOpenAITransport(allow_http=True))
        try:
            events = [event async for event in provider.stream(_request(_user("one")))]
        finally:
            await provider.aclose()

    error = events[-1]
    assert error.type is ModelStreamEventType.ERROR
    assert error.payload["status_code"] == 401
    assert error.payload["retryable"] is False
    assert "sk-test-websocket-key" not in json.dumps(error.payload)


async def test_context_overflow_error_event_is_typed() -> None:
    async def overflow(websocket: ServerConnection) -> None:
        async for _message_text in websocket:
            await websocket.send(
                json.dumps(
                    {
                        "type": "error",
                        "status": 400,
                        "error": {
                            "type": "invalid_request_error",
                            "code": "context_length_exceeded",
                            "message": "Your input exceeds the context window of this model.",
                            "param": "input",
                        },
                    }
                )
            )

    async with serve(overflow, "127.0.0.1", 0) as websocket_server:
        port = websocket_server.sockets[0].getsockname()[1]
        provider = _provider(f"http://127.0.0.1:{port}", WebSocketOpenAITransport(allow_http=True))
        try:
            with pytest.raises(OpenAIContextOverflowError):
                async for _event in provider.stream(_request(_user("one"))):
                    pass
        finally:
            await provider.aclose()


async def test_connection_lost_before_any_event_is_retryable() -> None:
    async def hang_up(websocket: ServerConnection) -> None:
        await websocket.recv()
        websocket.transport.abort()

    async with serve(hang_up, "127.0.0.1", 0) as websocket_server:
        port = websocket_server.sockets[0].getsockname()[1]
        transport = WebSocketOpenAITransport(allow_http=True)
        try:
            with pytest.raises(OpenAIAPIError) as failure:
                async for _event in transport.stream_response_events(
                    url=f"http://127.0.0.1:{port}/v1/responses",
                    headers={"authorization": "Bearer sk-test"},
                    payload={"model": _MODEL, "input": [], "stream": True},
                    timeout_s=5,
                    transport_idle_timeout_s=5,
                    protocol_idle_timeout_s=5,
                    semantic_progress_timeout_s=5,
                    absolute_stream_timeout_s=5,
                ):
                    pass
        finally:
            await transport.aclose()

    assert failure.value.error_type == "RemoteProtocolError"
    assert failure.value.retryable is True


def test_websocket_transport_has_its_own_execution_profile_material() -> None:
    http = _execution_profile_material(OpenAIProvider(api_key="sk-test"))
    websocket = _execution_profile_material(
        OpenAIProvider(api_key="sk-test", transport=WebSocketOpenAITransport())
    )
    assert http is not None and websocket is not None
    assert "responses_transport" not in http
    assert websocket == {**http, "responses_transport": "websocket", "incremental_input": True}


def test_websocket_transport_rejects_https_downgrade() -> None:
    from cayu.providers.openai import OpenAIError

    transport = WebSocketOpenAITransport()

    async def stream() -> None:
        async for _event in transport.stream_response_events(
            url="http://api.example.test/v1/responses",
            headers={},
            payload={"model": _MODEL, "input": []},
            timeout_s=1,
            transport_idle_timeout_s=1,
            protocol_idle_timeout_s=1,
            semantic_progress_timeout_s=1,
            absolute_stream_timeout_s=1,
        ):
            pass

    with pytest.raises((ValueError, OpenAIError)):
        asyncio.run(stream())


async def test_silent_server_hits_the_stream_deadline_and_drops_the_connection() -> None:
    from cayu.providers.base import ModelStreamDeadlineError

    async def silent(websocket: ServerConnection) -> None:
        async for _message_text in websocket:
            await asyncio.sleep(1)

    async with serve(silent, "127.0.0.1", 0) as websocket_server:
        port = websocket_server.sockets[0].getsockname()[1]
        transport = WebSocketOpenAITransport(allow_http=True)
        try:
            with pytest.raises(ModelStreamDeadlineError):
                async for _event in transport.stream_response_events(
                    url=f"http://127.0.0.1:{port}/v1/responses",
                    headers={"authorization": "Bearer sk-test"},
                    payload={"model": _MODEL, "input": [], "stream": True},
                    timeout_s=5,
                    transport_idle_timeout_s=0.2,
                    protocol_idle_timeout_s=0.2,
                    semantic_progress_timeout_s=5,
                    absolute_stream_timeout_s=5,
                ):
                    pass
            assert transport._idle == {}
        finally:
            await transport.aclose()


async def test_handshake_redirect_is_refused_without_following_it() -> None:
    from websockets.datastructures import Headers
    from websockets.http11 import Response

    elsewhere = FakeResponsesServer(outputs=[_FIRST])

    async with _running(elsewhere) as other_base_url:
        location = other_base_url.replace("http://", "ws://") + "/v1/responses"

        def redirect(connection: ServerConnection, request: Any) -> Response:
            return Response(307, "Temporary Redirect", Headers({"Location": location}), b"")

        async def never(websocket: ServerConnection) -> None:  # pragma: no cover
            raise AssertionError("handshake should be redirected")

        async with serve(never, "127.0.0.1", 0, process_request=redirect) as websocket_server:
            port = websocket_server.sockets[0].getsockname()[1]
            provider = OpenAIProvider(
                api_key="sk-test-websocket-key",
                base_url=f"http://127.0.0.1:{port}",
                allow_http=True,
                extra_headers={"api-key": "gateway-secret"},
                transport=WebSocketOpenAITransport(allow_http=True),
            )
            try:
                events = [event async for event in provider.stream(_request(_user("one")))]
            finally:
                await provider.aclose()

    assert events[-1].type is ModelStreamEventType.ERROR
    assert events[-1].payload["status_code"] == 307
    assert elsewhere.connections == 0
    assert elsewhere.received == []


async def test_handshake_that_loses_the_deadline_race_is_closed() -> None:
    import time

    from cayu.providers.base import ModelStreamDeadlineError

    accepted: list[ServerConnection] = []

    async def stall(websocket: ServerConnection) -> None:
        accepted.append(websocket)
        # Hold the event loop past the client's idle deadlines right after the
        # handshake, so the deadline and the finished handshake race.
        time.sleep(0.4)
        await websocket.wait_closed()

    async with serve(stall, "127.0.0.1", 0) as websocket_server:
        port = websocket_server.sockets[0].getsockname()[1]
        transport = WebSocketOpenAITransport(allow_http=True)
        try:
            with pytest.raises((ModelStreamDeadlineError, OpenAIAPIError)):
                async for _event in transport.stream_response_events(
                    url=f"http://127.0.0.1:{port}/v1/responses",
                    headers={"authorization": "Bearer sk-test"},
                    payload={"model": _MODEL, "input": [], "stream": True},
                    timeout_s=5,
                    transport_idle_timeout_s=0.2,
                    protocol_idle_timeout_s=0.2,
                    semantic_progress_timeout_s=5,
                    absolute_stream_timeout_s=5,
                ):
                    pass
            for _attempt in range(100):
                if accepted and all(websocket.close_code is not None for websocket in accepted):
                    break
                await asyncio.sleep(0.02)
            assert accepted
            assert all(websocket.close_code is not None for websocket in accepted)
        finally:
            await transport.aclose()


async def test_fragments_of_a_slow_message_keep_the_transport_clock_alive() -> None:
    created = json.dumps(
        {
            "type": "response.created",
            "response": {"id": "resp_1", "model": _MODEL, "status": "in_progress"},
        }
    )

    async def fragments() -> AsyncIterator[str]:
        for start in range(0, len(created), 20):
            yield created[start : start + 20]
            await asyncio.sleep(0.05)

    async def slow(websocket: ServerConnection) -> None:
        async for _message_text in websocket:
            await websocket.send(fragments())
            await websocket.send(json.dumps(_response_events("resp_1", _FIRST)[-1]))

    async with serve(slow, "127.0.0.1", 0) as websocket_server:
        port = websocket_server.sockets[0].getsockname()[1]
        transport = WebSocketOpenAITransport(allow_http=True)
        try:
            events = [
                event
                async for event in transport.stream_response_events(
                    url=f"http://127.0.0.1:{port}/v1/responses",
                    headers={"authorization": "Bearer sk-test"},
                    payload={"model": _MODEL, "input": [], "stream": True},
                    timeout_s=5,
                    # The whole message takes longer than this; each fragment does not.
                    transport_idle_timeout_s=0.3,
                    protocol_idle_timeout_s=10,
                    semantic_progress_timeout_s=10,
                    absolute_stream_timeout_s=10,
                )
            ]
        finally:
            await transport.aclose()

    assert [event["type"] for event in events] == ["response.created", "response.completed"]


async def test_cancelled_aclose_still_drops_every_pooled_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = FakeResponsesServer(outputs=[_FIRST, _FIRST])
    async with _running(server) as base_url:
        transport = WebSocketOpenAITransport(allow_http=True)
        provider = _provider(base_url, transport)
        first, _second = _two_turns()
        await asyncio.gather(_complete(provider, first), _complete(provider, first))
        pooled = [connection for idle in transport._idle.values() for connection in idle]
        assert len(pooled) == 2

        async def stuck_close(*_: Any, **__: Any) -> None:
            await asyncio.sleep(60)

        monkeypatch.setattr(pooled[0].websocket, "close", stuck_close)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(provider.aclose(), 0.2)

        assert transport._idle == {}
        assert all(connection.websocket.transport.is_closing() for connection in pooled)


async def test_bytes_during_a_consumer_pause_do_not_extend_the_transport_clock() -> None:
    from cayu.providers.base import ModelStreamDeadlineError

    created = _response_events("resp_1", _FIRST)[0]
    finished = asyncio.Event()

    async def stalled_message() -> AsyncIterator[str]:
        yield '{"type": "response.output_text.delta", '
        await finished.wait()

    async def stall_mid_message(websocket: ServerConnection) -> None:
        async for _message_text in websocket:
            await websocket.send(json.dumps(created))
            # One fragment arrives while the consumer is paused, then nothing.
            await asyncio.sleep(0.9)
            await websocket.send(stalled_message())

    async with serve(stall_mid_message, "127.0.0.1", 0) as websocket_server:
        port = websocket_server.sockets[0].getsockname()[1]
        transport = WebSocketOpenAITransport(allow_http=True)
        resumed_at = 0.0
        loop = asyncio.get_running_loop()
        try:
            with pytest.raises(ModelStreamDeadlineError):
                async for _event in transport.stream_response_events(
                    url=f"http://127.0.0.1:{port}/v1/responses",
                    headers={"authorization": "Bearer sk-test"},
                    payload={"model": _MODEL, "input": [], "stream": True},
                    timeout_s=5,
                    transport_idle_timeout_s=0.5,
                    protocol_idle_timeout_s=30,
                    semantic_progress_timeout_s=30,
                    absolute_stream_timeout_s=30,
                ):
                    await asyncio.sleep(1.0)
                    resumed_at = loop.time()
            # The pause is excluded once: the stall expires half a second after
            # the consumer resumes, not after the pause plus the late fragment.
            assert loop.time() - resumed_at < 1.0
        finally:
            finished.set()
            await transport.aclose()
