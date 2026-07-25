from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from cayu.core import AgentSpec, EventType, Message, ToolCallPart, ToolResultPart
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import CayuApp, InMemorySessionStore, RunRequest


class _SequencedProvider(ModelProvider):
    name = "sequenced"

    def __init__(self, responses: list[list[ModelStreamEvent]]) -> None:
        self._responses = responses
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        response = self._responses[len(self.requests)]
        self.requests.append(request)
        for event in response:
            yield event


class _RecordingTool(Tool):
    spec = ToolSpec(
        name="record",
        description="Record a value.",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        },
    )

    def __init__(self) -> None:
        super().__init__()
        self.values: list[int] = []

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        value = args["value"]
        self.values.append(value)
        return ToolResult(content=str(value))


def _tool_call_response(value: int) -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.tool_call(
            id="provider-reused-call-id",
            name="record",
            arguments={"value": value},
        ),
        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
    ]


def test_runtime_links_reused_provider_call_ids_to_distinct_tool_rounds() -> None:
    store = InMemorySessionStore()
    provider = _SequencedProvider(
        [
            _tool_call_response(1),
            _tool_call_response(2),
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    tool = _RecordingTool()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="sequenced-model"),
        tools=[tool],
    )

    async def scenario():
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_reused_tool_call_id",
                    messages=[Message.text("user", "record twice")],
                )
            )
        ]
        return events, await store.load_transcript("sess_reused_tool_call_id")

    events, transcript = asyncio.run(scenario())

    completed_rounds = [
        event
        for event in events
        if event.type == EventType.MODEL_COMPLETED and "tool_round_id" in event.payload
    ]
    started_calls = [event for event in events if event.type == EventType.TOOL_CALL_STARTED]
    completed_calls = [event for event in events if event.type == EventType.TOOL_CALL_COMPLETED]

    assert len(completed_rounds) == len(started_calls) == len(completed_calls) == 2
    assert tool.values == [1, 2]

    identity_fields = ("model_step_id", "model_attempt_id", "tool_round_id")
    for model_event, started_event, completed_event in zip(
        completed_rounds,
        started_calls,
        completed_calls,
        strict=True,
    ):
        expected_identity = {field: model_event.payload[field] for field in identity_fields}
        assert {field: started_event.payload[field] for field in identity_fields} == (
            expected_identity
        )
        assert {field: completed_event.payload[field] for field in identity_fields} == (
            expected_identity
        )
        assert started_event.payload["tool_call_id"] == "provider-reused-call-id"
        assert completed_event.payload["tool_call_id"] == "provider-reused-call-id"

    assert (
        completed_rounds[0].payload["tool_round_id"] != completed_rounds[1].payload["tool_round_id"]
    )

    transcript_parts = [
        part
        for message in transcript
        for part in message.content
        if isinstance(part, (ToolCallPart, ToolResultPart))
    ]
    assert len(transcript_parts) == 4
    for round_index, (call_part, result_part) in enumerate(
        zip(transcript_parts[::2], transcript_parts[1::2], strict=True)
    ):
        assert isinstance(call_part, ToolCallPart)
        assert isinstance(result_part, ToolResultPart)
        expected_event = completed_rounds[round_index]
        assert call_part.tool_call_id == result_part.tool_call_id == "provider-reused-call-id"
        for field in identity_fields:
            assert getattr(call_part, field) == expected_event.payload[field]
            assert getattr(result_part, field) == expected_event.payload[field]


def test_runtime_rejects_duplicate_call_ids_within_one_tool_round() -> None:
    store = InMemorySessionStore()
    provider = _SequencedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="duplicate-call-id",
                    name="record",
                    arguments={"value": 1},
                ),
                ModelStreamEvent.tool_call(
                    id="duplicate-call-id",
                    name="record",
                    arguments={"value": 2},
                ),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "tool_calls",
                        "usage": {
                            "input_tokens": 7,
                            "output_tokens": 3,
                            "total_tokens": 10,
                        },
                    }
                ),
            ]
        ]
    )
    tool = _RecordingTool()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="sequenced-model"),
        tools=[tool],
    )

    async def scenario():
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_duplicate_tool_call_id",
                    messages=[Message.text("user", "record twice")],
                )
            )
        ]
        return events, await store.load_transcript("sess_duplicate_tool_call_id")

    events, transcript = asyncio.run(scenario())

    assert len(provider.requests) == 1
    assert tool.values == []
    assert not any(
        event.type
        in {
            EventType.TOOL_CALL_STARTED,
            EventType.TOOL_CALL_COMPLETED,
            EventType.TOOL_CALL_FAILED,
            EventType.TOOL_CALL_BLOCKED,
        }
        for event in events
    )
    completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
    assert completed.payload["usage_metrics"]["total_tokens"] == 10
    assert completed.payload["completion_outcome"] == "invalid_transcript_state"
    assert completed.payload["completion_error"]["provider_error_code"] == (
        "invalid_model_completion_transcript"
    )
    assert "tool_round_id" not in completed.payload
    assert completed.payload["model_step_id"].startswith("mstep_")
    assert completed.payload["model_attempt_id"].startswith("matt_")
    assert transcript == [Message.text("user", "record twice")]
    assert events[-1].type == EventType.SESSION_FAILED


def test_runtime_removes_provider_spoofed_execution_identity() -> None:
    spoofed_identity = {
        "model_step_id": f"mstep_{'a' * 32}",
        "model_attempt_id": f"matt_{'b' * 32}",
        "tool_round_id": f"tround_{'c' * 32}",
    }
    provider = _SequencedProvider(
        [
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        **spoofed_identity,
                    }
                ),
            ]
        ]
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="sequenced-model"))

    async def scenario():
        return [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_provider_spoofed_execution_identity",
                    messages=[Message.text("user", "finish")],
                )
            )
        ]

    events = asyncio.run(scenario())

    completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
    assert completed.payload["model_step_id"] != spoofed_identity["model_step_id"]
    assert completed.payload["model_attempt_id"] != spoofed_identity["model_attempt_id"]
    assert "tool_round_id" not in completed.payload
