from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import httpx
import pytest

import cayu.providers.openai as openai_module
from cayu import (
    RESOLVED_FILE_ATTACHMENTS_OPTION,
    AgentSpec,
    CayuApp,
    FileAttachmentKind,
    Message,
    ResumeRequest,
    RunRequest,
    file_attachment,
)
from cayu.core.messages import ProviderStatePart, TextPart, ToolCallPart
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.providers import (
    HttpxOpenAITransport,
    ModelRequest,
    ModelStreamEventType,
    OpenAIAPIError,
    OpenAIProtocolError,
    OpenAIProvider,
    build_openai_payload,
    openai_response_events,
)


class RecordingTransport:
    def __init__(
        self,
        stream_events: list[list[Mapping[str, Any]]] | None = None,
    ) -> None:
        self.stream_event_batches = list(stream_events or [])
        self.calls: list[dict[str, Any]] = []

    async def stream_response_events(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
        stream_idle_timeout_s: float,
    ):
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "payload": dict(payload),
                "timeout_s": timeout_s,
                "stream_idle_timeout_s": stream_idle_timeout_s,
            }
        )
        if not self.stream_event_batches:
            raise AssertionError("No fake OpenAI stream queued.")
        for event in self.stream_event_batches.pop(0):
            yield event


class BlankFailingTransport:
    async def stream_response_events(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
        stream_idle_timeout_s: float,
    ):
        raise RuntimeError()
        yield {}


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

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        return ToolResult(
            content=args["text"],
            structured={"agent": ctx.agent_name, "echoed": args["text"]},
        )


def test_build_openai_payload_translates_cayu_messages() -> None:
    request = ModelRequest(
        model="gpt-test",
        messages=[
            Message.text("system", "You are a careful assistant."),
            Message.text("user", "Read a file."),
            Message(
                role="assistant",
                content=[
                    TextPart(text="I will inspect it."),
                    ToolCallPart(
                        tool_call_id="call_1",
                        tool_name="read_file",
                        arguments={"path": "README.md"},
                    ),
                ],
            ),
            Message.tool_result(
                tool_call_id="call_1",
                tool_name="read_file",
                content="README content",
                structured={"ignored_by_provider": True},
            ),
        ],
        tools=[
            {
                "name": "read_file",
                "description": "Read a file.",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            }
        ],
        options={"openai": {"temperature": 0.2}},
    )

    payload = build_openai_payload(request)

    assert payload == {
        "model": "gpt-test",
        "instructions": "You are a careful assistant.",
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Read a file."}],
            },
            {
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "I will inspect it."},
                ],
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "read_file",
                "arguments": '{"path":"README.md"}',
                "status": "completed",
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "README content",
            },
        ],
        "tools": [
            {
                "type": "function",
                "name": "read_file",
                "description": "Read a file.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
                "strict": False,
            }
        ],
        "store": False,
        "temperature": 0.2,
    }


def test_build_openai_payload_translates_file_attachments() -> None:
    attachment = file_attachment(
        artifact_id="art_pdf",
        kind=FileAttachmentKind.DOCUMENT,
        filename="invoice.pdf",
        content_type="application/pdf",
        size_bytes=5,
    )
    request = ModelRequest(
        model="gpt-test",
        messages=[
            Message.text("user", "Read the invoice."),
            Message.tool_call(
                tool_call_id="call_1",
                tool_name="read_file",
                arguments={"artifact_id": "art_pdf"},
            ),
            Message.tool_result(
                tool_call_id="call_1",
                tool_name="read_file",
                content="Attached PDF artifact art_pdf: invoice.pdf.",
                artifacts=[attachment],
            ),
        ],
        options={
            RESOLVED_FILE_ATTACHMENTS_OPTION: {
                "art_pdf": {
                    "artifact_id": "art_pdf",
                    "kind": "document",
                    "filename": "invoice.pdf",
                    "content_type": "application/pdf",
                    "data_base64": "aGVsbG8=",
                    "metadata": {},
                }
            }
        },
    )

    payload = build_openai_payload(request)

    assert payload["input"][2] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "Attached PDF artifact art_pdf: invoice.pdf.",
    }
    assert payload["input"][3] == {
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": "The previous tool result returned file content for inspection.",
            },
            {
                "type": "input_file",
                "filename": "invoice.pdf",
                "file_data": "data:application/pdf;base64,aGVsbG8=",
            },
        ],
    }


@pytest.mark.anyio
async def test_openai_provider_emits_text_and_completed_events() -> None:
    transport = RecordingTransport(
        stream_events=[
            [
                {"type": "response.created", "response": {"id": "resp_1"}},
                {"type": "response.output_text.delta", "delta": "hello"},
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_1",
                        "model": "gpt-test",
                        "status": "completed",
                        "output": [
                            {
                                "type": "message",
                                "id": "msg_1",
                                "status": "completed",
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": "hello",
                                        "annotations": [],
                                    }
                                ],
                            }
                        ],
                        "usage": {"input_tokens": 10, "output_tokens": 2},
                    },
                },
            ]
        ]
    )
    provider = OpenAIProvider(api_key="test-key", transport=transport)
    request = ModelRequest(
        model="gpt-test",
        messages=[Message.text("user", "Say hello.")],
    )

    events = [event async for event in provider.stream(request)]

    assert [event.type for event in events] == [
        ModelStreamEventType.TEXT_DELTA,
        ModelStreamEventType.COMPLETED,
    ]
    assert events[0].delta == "hello"
    assert events[1].payload["status"] == "completed"
    assert transport.calls[0]["url"] == "https://api.openai.com/v1/responses"
    assert transport.calls[0]["headers"]["authorization"] == "Bearer test-key"
    assert transport.calls[0]["payload"]["store"] is False
    assert transport.calls[0]["payload"]["stream"] is True


@pytest.mark.anyio
async def test_openai_provider_emits_tool_call_events() -> None:
    transport = RecordingTransport(
        stream_events=[
            [
                {
                    "type": "response.output_item.added",
                    "output_index": 1,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "echo",
                        "arguments": "",
                    },
                },
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": "fc_1",
                    "output_index": 1,
                    "delta": '{"text":',
                },
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": "fc_1",
                    "output_index": 1,
                    "delta": '"hello"}',
                },
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": "fc_1",
                    "output_index": 1,
                    "name": "echo",
                    "arguments": '{"text":"hello"}',
                },
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_1",
                        "model": "gpt-test",
                        "status": "completed",
                        "output": [
                            {
                                "type": "reasoning",
                                "id": "rs_1",
                                "summary": [],
                                "phase": "tool_use",
                            },
                            {
                                "type": "function_call",
                                "id": "fc_1",
                                "call_id": "call_1",
                                "name": "echo",
                                "arguments": '{"text":"hello"}',
                                "status": "completed",
                            },
                        ],
                    },
                },
            ]
        ]
    )
    provider = OpenAIProvider(api_key="test-key", transport=transport)
    request = ModelRequest(
        model="gpt-test",
        messages=[Message.text("user", "Use a tool.")],
    )

    events = [event async for event in provider.stream(request)]

    assert [event.type for event in events] == [
        ModelStreamEventType.TOOL_CALL,
        ModelStreamEventType.COMPLETED,
    ]
    assert events[0].payload == {
        "id": "call_1",
        "name": "echo",
        "arguments": {"text": "hello"},
    }
    assert events[1].payload["status"] == "completed"


@pytest.mark.anyio
async def test_openai_provider_round_trips_runtime_tool_results() -> None:
    transport = RecordingTransport(
        stream_events=[
            [
                {
                    "type": "response.output_item.added",
                    "output_index": 1,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "echo",
                        "arguments": "",
                    },
                },
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": "fc_1",
                    "output_index": 1,
                    "delta": '{"text":"hello from openai"}',
                },
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": "fc_1",
                    "output_index": 1,
                    "name": "echo",
                    "arguments": '{"text":"hello from openai"}',
                },
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_1",
                        "model": "gpt-test",
                        "status": "completed",
                        "output": [
                            {
                                "type": "reasoning",
                                "id": "rs_1",
                                "summary": [],
                                "phase": "tool_use",
                            },
                            {
                                "type": "function_call",
                                "id": "fc_1",
                                "call_id": "call_1",
                                "name": "echo",
                                "arguments": '{"text":"hello from openai"}',
                                "status": "completed",
                            },
                        ],
                    },
                },
            ],
            [
                {"type": "response.output_text.delta", "delta": "final answer"},
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_2",
                        "model": "gpt-test",
                        "status": "completed",
                        "output": [
                            {
                                "type": "message",
                                "id": "msg_2",
                                "status": "completed",
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": "final answer",
                                        "annotations": [],
                                    }
                                ],
                            }
                        ],
                    },
                },
            ],
        ]
    )
    provider = OpenAIProvider(api_key="test-key", transport=transport)
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(
            name="assistant",
            model="gpt-test",
            system_prompt="Use tools when needed.",
        ),
        tools=[EchoTool()],
    )

    events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "Echo this.")],
            )
        )
    ]

    assert events[-1].type == "session.completed"
    model_completed_events = [event for event in events if event.type == "model.completed"]
    assert "provider_state" not in model_completed_events[0].payload
    assert len(transport.calls) == 2
    assert transport.calls[0]["payload"]["input"] == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "Echo this."}],
        }
    ]
    assert transport.calls[1]["payload"]["input"] == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "Echo this."}],
        },
        {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [],
            "phase": "tool_use",
        },
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "echo",
            "arguments": '{"text":"hello from openai"}',
            "status": "completed",
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "hello from openai",
        },
    ]


@pytest.mark.anyio
async def test_openai_provider_replays_streamed_function_call_when_completed_lacks_output() -> None:
    transport = RecordingTransport(
        stream_events=[
            [
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "echo",
                        "arguments": "",
                    },
                },
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": "fc_1",
                    "output_index": 0,
                    "delta": '{"text":"hello from openai"}',
                },
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": "fc_1",
                    "output_index": 0,
                    "name": "echo",
                    "arguments": '{"text":"hello from openai"}',
                },
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_1",
                        "model": "gpt-test",
                        "status": "completed",
                    },
                },
            ],
            [
                {"type": "response.output_text.delta", "delta": "final answer"},
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_2",
                        "model": "gpt-test",
                        "status": "completed",
                        "output": [
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": "final answer"}],
                            }
                        ],
                    },
                },
            ],
        ]
    )
    provider = OpenAIProvider(api_key="test-key", transport=transport)
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="gpt-test", system_prompt="Use tools."),
        tools=[EchoTool()],
    )

    events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "Echo this.")],
            )
        )
    ]

    assert events[-1].type == "session.completed"
    assert transport.calls[1]["payload"]["input"] == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "Echo this."}],
        },
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "echo",
            "arguments": '{"text":"hello from openai"}',
            "status": "completed",
            "id": "fc_1",
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "hello from openai",
        },
    ]


@pytest.mark.anyio
async def test_openai_provider_replays_streamed_text_when_completed_lacks_output() -> None:
    transport = RecordingTransport(
        stream_events=[
            [
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {
                        "type": "message",
                        "id": "msg_1",
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [],
                    },
                },
                {"type": "response.output_text.delta", "delta": "hello"},
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": {
                        "type": "message",
                        "id": "msg_1",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "hello"}],
                    },
                },
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_1",
                        "model": "gpt-test",
                        "status": "completed",
                    },
                },
            ],
            [
                {"type": "response.output_text.delta", "delta": "second answer"},
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_2",
                        "model": "gpt-test",
                        "status": "completed",
                        "output": [
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": "second answer"}],
                            }
                        ],
                    },
                },
            ],
        ]
    )
    provider = OpenAIProvider(api_key="test-key", transport=transport)
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="gpt-test", system_prompt="Be direct."))

    run_events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id="sess_openai_text_replay",
                messages=[Message.text("user", "Say hello.")],
            )
        )
    ]
    resume_events = [
        event
        async for event in app.resume(
            ResumeRequest(
                session_id="sess_openai_text_replay",
                messages=[Message.text("user", "Again.")],
            )
        )
    ]

    assert run_events[-1].type == "session.completed"
    assert resume_events[-1].type == "session.completed"
    assert transport.calls[1]["payload"]["input"] == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "Say hello."}],
        },
        {
            "type": "message",
            "id": "msg_1",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hello"}],
        },
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "Again."}],
        },
    ]


def test_openai_response_events_rejects_malformed_function_call() -> None:
    with pytest.raises(OpenAIProtocolError, match="arguments were not valid JSON"):
        openai_response_events(
            {
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "echo",
                        "arguments": "{",
                    }
                ]
            }
        )


def test_openai_response_events_sanitizes_response_error() -> None:
    with pytest.raises(OpenAIProtocolError) as exc_info:
        openai_response_events(
            {
                "error": {
                    "type": "invalid_request_error",
                    "code": "bad_request",
                    "message": "bad request",
                    "debug": "not persisted",
                },
                "output": [],
            }
        )

    message = str(exc_info.value)
    assert (
        message == 'OpenAI response error: {"code":"bad_request",'
        '"message":"bad request","type":"invalid_request_error"}'
    )
    assert "debug" not in message
    assert "not persisted" not in message


def test_openai_response_events_emits_refusal_text() -> None:
    events = openai_response_events(
        {
            "id": "resp_1",
            "model": "gpt-test",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "refusal",
                            "refusal": "I cannot help with that.",
                        }
                    ],
                }
            ],
        }
    )

    assert [event.type for event in events] == [
        ModelStreamEventType.TEXT_DELTA,
        ModelStreamEventType.COMPLETED,
    ]
    assert events[0].delta == "I cannot help with that."


@pytest.mark.anyio
async def test_openai_stream_events_emits_incomplete_terminal_response() -> None:
    async def raw_events():
        yield {"type": "response.output_text.delta", "delta": "partial"}
        yield {
            "type": "response.incomplete",
            "response": {
                "id": "resp_1",
                "model": "gpt-test",
                "status": "incomplete",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "partial"}],
                    }
                ],
                "incomplete_details": {"reason": "max_output_tokens"},
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        }

    events = [event async for event in openai_module.openai_stream_events(raw_events())]

    assert [event.type for event in events] == [
        ModelStreamEventType.TEXT_DELTA,
        ModelStreamEventType.COMPLETED,
    ]
    assert events[0].delta == "partial"
    assert events[1].payload["status"] == "incomplete"
    assert events[1].payload["incomplete_details"] == {"reason": "max_output_tokens"}


@pytest.mark.anyio
async def test_openai_stream_events_uses_done_function_call_arguments() -> None:
    async def raw_events():
        yield {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "echo",
                "arguments": "",
            },
        }
        yield {
            "type": "response.function_call_arguments.done",
            "item_id": "fc_1",
            "output_index": 0,
            "name": "echo",
            "arguments": '{"text":"from done event"}',
            "sequence_number": 2,
        }
        yield {
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "model": "gpt-test",
                "status": "completed",
            },
        }

    events = [event async for event in openai_module.openai_stream_events(raw_events())]

    assert [event.type for event in events] == [
        ModelStreamEventType.TOOL_CALL,
        ModelStreamEventType.COMPLETED,
    ]
    assert events[0].payload == {
        "id": "call_1",
        "name": "echo",
        "arguments": {"text": "from done event"},
    }


@pytest.mark.anyio
async def test_openai_stream_events_emits_refusal_text() -> None:
    async def raw_events():
        yield {
            "type": "response.refusal.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "I cannot help with that.",
            "sequence_number": 1,
        }
        yield {
            "type": "response.refusal.done",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "refusal": "I cannot help with that.",
            "sequence_number": 2,
        }
        yield {
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "model": "gpt-test",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_1",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "refusal",
                                "refusal": "I cannot help with that.",
                            }
                        ],
                    }
                ],
            },
        }

    events = [event async for event in openai_module.openai_stream_events(raw_events())]

    assert [event.type for event in events] == [
        ModelStreamEventType.TEXT_DELTA,
        ModelStreamEventType.COMPLETED,
    ]
    assert events[0].delta == "I cannot help with that."


@pytest.mark.anyio
async def test_openai_stream_events_rejects_function_call_done_without_call_id() -> None:
    async def raw_events():
        yield {
            "type": "response.function_call_arguments.done",
            "item_id": "fc_1",
            "output_index": 0,
            "name": "echo",
            "arguments": '{"text":"hello"}',
            "sequence_number": 1,
        }

    with pytest.raises(OpenAIProtocolError, match="arrived before output_item.added"):
        [event async for event in openai_module.openai_stream_events(raw_events())]


@pytest.mark.anyio
async def test_openai_stream_events_extracts_response_failed_error() -> None:
    async def raw_events():
        yield {
            "type": "response.failed",
            "response": {
                "id": "resp_1",
                "status": "failed",
                "error": {
                    "code": "server_error",
                    "message": "The model failed.",
                    "debug": "not persisted",
                },
            },
            "sequence_number": 1,
        }

    with pytest.raises(OpenAIAPIError) as exc_info:
        [event async for event in openai_module.openai_stream_events(raw_events())]

    message = str(exc_info.value)
    assert message == (
        'OpenAI streaming error: {"code":"server_error","message":"The model failed."}'
    )
    assert "debug" not in message
    assert "not persisted" not in message


@pytest.mark.anyio
async def test_openai_stream_events_extracts_top_level_error_event() -> None:
    async def raw_events():
        yield {
            "type": "error",
            "code": "rate_limit_exceeded",
            "message": "Too many requests.",
            "param": None,
            "sequence_number": 1,
        }

    with pytest.raises(OpenAIAPIError) as exc_info:
        [event async for event in openai_module.openai_stream_events(raw_events())]

    assert str(exc_info.value) == (
        'OpenAI streaming error: {"code":"rate_limit_exceeded",'
        '"message":"Too many requests.","type":"error"}'
    )


def test_openai_response_events_rejects_unsupported_output_item() -> None:
    with pytest.raises(OpenAIProtocolError, match="Unsupported OpenAI output item"):
        openai_response_events({"output": [{"type": "web_search_call"}]})


def test_openai_response_events_ignores_reasoning_items() -> None:
    events = openai_response_events(
        {
            "id": "resp_1",
            "model": "gpt-test",
            "status": "completed",
            "output": [
                {"type": "reasoning", "id": "rs_1", "summary": []},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "done"}],
                },
            ],
        }
    )

    assert [event.type for event in events] == [
        ModelStreamEventType.TEXT_DELTA,
        ModelStreamEventType.COMPLETED,
    ]
    assert events[0].delta == "done"


@pytest.mark.parametrize("reserved_option", ["input", "store", "previous_response_id"])
def test_openai_options_must_not_override_reserved_payload_fields(
    reserved_option: str,
) -> None:
    request = ModelRequest(
        model="gpt-test",
        messages=[Message.text("user", "hello")],
        options={"openai": {reserved_option: "bad"}},
    )

    with pytest.raises(ValueError, match="reserved"):
        build_openai_payload(request)


def test_openai_payload_replays_provider_state_items() -> None:
    request = ModelRequest(
        model="gpt-test",
        messages=[
            Message.text("user", "hello"),
            Message(
                role="assistant",
                content=[
                    ProviderStatePart(
                        provider="openai",
                        state={
                            "type": "reasoning",
                            "id": "rs_1",
                            "summary": [],
                            "phase": "tool_use",
                        },
                    ),
                    ProviderStatePart(
                        provider="openai",
                        state={
                            "type": "function_call",
                            "id": "fc_1",
                            "call_id": "call_1",
                            "name": "echo",
                            "arguments": '{"text":"hello"}',
                            "status": "completed",
                        },
                    ),
                    ToolCallPart(
                        tool_call_id="call_1",
                        tool_name="echo",
                        arguments={"text": "hello"},
                    ),
                ],
            ),
            Message.tool_result(
                tool_call_id="call_1",
                tool_name="echo",
                content="hello",
            ),
        ],
    )

    payload = build_openai_payload(request)

    assert payload["input"] == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        },
        {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [],
            "phase": "tool_use",
        },
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "echo",
            "arguments": '{"text":"hello"}',
            "status": "completed",
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "hello",
        },
    ]


def test_openai_payload_ignores_other_provider_state_items() -> None:
    request = ModelRequest(
        model="gpt-test",
        messages=[
            Message.text("user", "hello"),
            Message(
                role="assistant",
                content=[
                    ProviderStatePart(
                        provider="other-provider",
                        state={"type": "opaque", "id": "state_1"},
                    ),
                    TextPart(text="assistant text"),
                ],
            ),
        ],
    )

    payload = build_openai_payload(request)

    assert payload["input"] == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        },
        {
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "assistant text"}],
        },
    ]


def test_openai_provider_rejects_invalid_tool_names() -> None:
    request = ModelRequest(
        model="gpt-test",
        messages=[Message.text("user", "hello")],
        tools=[
            {
                "name": "invalid tool name",
                "description": "Bad name.",
                "input_schema": {"type": "object"},
            }
        ],
    )

    with pytest.raises(ValueError, match="tool names"):
        build_openai_payload(request)


def test_openai_provider_rejects_protected_extra_headers() -> None:
    with pytest.raises(ValueError, match="extra_headers cannot override"):
        OpenAIProvider(
            api_key="test-key",
            extra_headers={"authorization": "other-key"},
        )


def test_openai_provider_requires_https_base_url() -> None:
    with pytest.raises(ValueError, match="https"):
        OpenAIProvider(
            api_key="test-key",
            base_url="http://api.openai.com",
        )


@pytest.mark.anyio
async def test_httpx_openai_transport_requires_https_url() -> None:
    with pytest.raises(ValueError, match="https"):
        await HttpxOpenAITransport().create_response(
            url="http://api.openai.com/v1/responses",
            headers={},
            payload={},
            timeout_s=1,
        )


@pytest.mark.anyio
async def test_httpx_openai_transport_includes_url_in_network_errors(monkeypatch) -> None:
    class FailingClient:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> FailingClient:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def post(
            self,
            url: str,
            *,
            headers: dict[str, str],
            json: dict[str, Any],
        ) -> httpx.Response:
            request = httpx.Request("POST", url)
            raise httpx.ConnectError(
                "[Errno 8] nodename nor servname provided, or not known",
                request=request,
            )

    monkeypatch.setattr(
        "cayu.providers.openai.httpx.AsyncClient",
        FailingClient,
    )

    with pytest.raises(OpenAIAPIError, match="https://api.openai.com/v1/responses"):
        await HttpxOpenAITransport().create_response(
            url="https://api.openai.com/v1/responses",
            headers={},
            payload={},
            timeout_s=1,
        )


@pytest.mark.anyio
async def test_httpx_openai_transport_sanitizes_error_body(monkeypatch) -> None:
    class FailingClient:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> FailingClient:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def post(
            self,
            url: str,
            *,
            headers: dict[str, str],
            json: dict[str, Any],
        ) -> httpx.Response:
            request = httpx.Request("POST", url)
            response = httpx.Response(
                400,
                request=request,
                headers={"content-type": "application/json"},
                json={
                    "error": {
                        "type": "invalid_request_error",
                        "code": "bad_request",
                        "message": "bad request",
                        "debug": "not persisted",
                    },
                    "request_id": "req_123",
                    "extra": "not persisted",
                },
            )
            raise httpx.HTTPStatusError(
                "bad request",
                request=request,
                response=response,
            )

    monkeypatch.setattr(
        "cayu.providers.openai.httpx.AsyncClient",
        FailingClient,
    )

    with pytest.raises(OpenAIAPIError) as exc_info:
        await HttpxOpenAITransport().create_response(
            url="https://api.openai.com/v1/responses",
            headers={},
            payload={},
            timeout_s=1,
        )

    message = str(exc_info.value)
    assert (
        message == "OpenAI API request failed with HTTP 400: "
        '{"code":"bad_request","message":"bad request",'
        '"request_id":"req_123","type":"invalid_request_error"}'
    )
    assert "debug" not in message
    assert "not persisted" not in message


@pytest.mark.anyio
async def test_openai_provider_emits_nonblank_error_for_blank_exception() -> None:
    provider = OpenAIProvider(api_key="test-key", transport=BlankFailingTransport())
    request = ModelRequest(
        model="gpt-test",
        messages=[Message.text("user", "hello")],
    )

    events = [event async for event in provider.stream(request)]

    assert [event.type for event in events] == [ModelStreamEventType.ERROR]
    assert events[0].payload == {"error": "RuntimeError: OpenAI provider failed"}


@pytest.mark.anyio
async def test_openai_sse_parser_does_not_let_keepalives_reset_event_idle_timeout() -> None:
    async def lines():
        yield ": keepalive"
        await asyncio.sleep(0.01)
        yield ""

    with pytest.raises(TimeoutError, match="no SSE events"):
        [
            event
            async for event in openai_module._aiter_sse_json_events(
                lines(),
                idle_timeout_s=0.001,
            )
        ]
