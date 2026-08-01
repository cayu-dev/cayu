from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import cast

import pytest
from tests.core._event_projection_support import private_events_for_public_events

from cayu.core import AgentSpec, EventType, Message, ToolCallPart, ToolResultPart
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import (
    CayuApp,
    InMemorySessionStore,
    RunRequest,
    SessionIdentity,
    SessionStore,
)
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.runtime import _transcript as transcript_helpers
from cayu.runtime.execution_units import ToolRoundIdentity


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


class _TranscriptOnlyStore:
    def __init__(self, transcript: list[Message]) -> None:
        self._transcript = transcript

    async def load_transcript(self, _session_id: str) -> list[Message]:
        return [message.model_copy(deep=True) for message in self._transcript]


def _tool_call_response(value: int) -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.tool_call(
            id="provider-reused-call-id",
            name="record",
            arguments={"value": value},
        ),
        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
    ]


def _tool_round_identity(value: str) -> ToolRoundIdentity:
    return ToolRoundIdentity(
        model_step_id=f"mstep_{value * 32}",
        model_attempt_id=f"matt_{value * 32}",
        tool_round_id=f"tround_{value * 32}",
    )


def _tool_call(
    *,
    call_id: str = "call-1",
    name: str = "side_effect",
    arguments: dict | None = None,
) -> runtime_records.ToolCallRequest:
    return runtime_records.ToolCallRequest(
        id=call_id,
        name=name,
        arguments={} if arguments is None else arguments,
    )


def _mixed_identity_result_message(
    identity: ToolRoundIdentity,
    conflicting_identity: ToolRoundIdentity,
    *,
    tool_name: str,
) -> Message:
    return Message.tool_result(
        results=[
            ToolResultPart(
                tool_call_id="call-1",
                tool_name=tool_name,
                content="expected result",
                **identity.payload(),
            ),
            ToolResultPart(
                tool_call_id="call-1",
                tool_name=tool_name,
                content="conflicting result",
                **conflicting_identity.payload(),
            ),
        ]
    )


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
        return (
            await private_events_for_public_events(store, events),
            await store.load_transcript("sess_reused_tool_call_id"),
        )

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
        return (
            await private_events_for_public_events(store, events),
            await store.load_transcript("sess_duplicate_tool_call_id"),
        )

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


def test_pending_round_rejects_conflicting_transcript_tool_descriptor() -> None:
    identity = _tool_round_identity("1")
    transcript = [
        Message.tool_call(
            tool_call_id="call-1",
            tool_name="side_effect",
            arguments={},
            **identity.payload(),
        ),
        Message.tool_result(
            tool_call_id="call-1",
            tool_name="different_tool",
            content="wrong result",
            **identity.payload(),
        ),
    ]

    with pytest.raises(ValueError, match="result descriptor"):
        asyncio.run(
            transcript_helpers.tool_round_has_result_messages(
                cast("SessionStore", _TranscriptOnlyStore(transcript)),
                "session-1",
                [_tool_call()],
                tool_round_identity=identity,
            )
        )


def test_pending_round_accepts_one_complete_call_and_result_message() -> None:
    identity = _tool_round_identity("1")
    transcript = [
        Message.tool_call(
            tool_call_id="call-1",
            tool_name="side_effect",
            arguments={},
            **identity.payload(),
        ),
        Message.tool_result(
            tool_call_id="call-1",
            tool_name="side_effect",
            content="recorded",
            **identity.payload(),
        ),
    ]

    closed = asyncio.run(
        transcript_helpers.tool_round_has_result_messages(
            cast("SessionStore", _TranscriptOnlyStore(transcript)),
            "session-1",
            [_tool_call()],
            tool_round_identity=identity,
        )
    )

    assert closed is True


@pytest.mark.parametrize(
    ("case", "error_pattern"),
    [
        pytest.param("result_without_call", "before the pending round call boundary", id="no-call"),
        pytest.param("result_before_call", "before the pending round call boundary", id="order"),
        pytest.param("incomplete_calls", "does not match the pending call set", id="membership"),
        pytest.param("duplicate_calls", "duplicate calls", id="duplicate-calls"),
        pytest.param("duplicate_results", "duplicate results", id="duplicate-results"),
    ],
)
def test_pending_round_rejects_malformed_transcript_grammar(
    case: str,
    error_pattern: str,
) -> None:
    identity = _tool_round_identity("1")
    call_parts = [
        ToolCallPart(
            tool_call_id="call-1",
            tool_name="side_effect",
            arguments={},
            **identity.payload(),
        )
    ]
    result_parts = [
        ToolResultPart(
            tool_call_id="call-1",
            tool_name="side_effect",
            content="recorded",
            **identity.payload(),
        )
    ]
    pending_calls = [_tool_call()]
    if case == "result_without_call":
        transcript = [Message.tool_result(results=result_parts)]
    elif case == "result_before_call":
        transcript = [
            Message.tool_result(results=result_parts),
            Message.tool_call(calls=call_parts),
        ]
    elif case == "incomplete_calls":
        pending_calls.append(_tool_call(call_id="call-2"))
        transcript = [
            Message.tool_call(calls=call_parts),
            Message.tool_result(
                results=[
                    *result_parts,
                    ToolResultPart(
                        tool_call_id="call-2",
                        tool_name="side_effect",
                        content="recorded",
                        **identity.payload(),
                    ),
                ]
            ),
        ]
    elif case == "duplicate_calls":
        transcript = [
            Message.tool_call(calls=[*call_parts, *call_parts]),
            Message.tool_result(results=result_parts),
        ]
    else:
        transcript = [
            Message.tool_call(calls=call_parts),
            Message.tool_result(results=[*result_parts, *result_parts]),
        ]

    with pytest.raises(ValueError, match=error_pattern):
        asyncio.run(
            transcript_helpers.tool_round_has_result_messages(
                cast("SessionStore", _TranscriptOnlyStore(transcript)),
                "session-1",
                pending_calls,
                tool_round_identity=identity,
            )
        )


def test_pending_round_rejects_mixed_transcript_tool_round_identities() -> None:
    identity = _tool_round_identity("1")
    conflicting_identity = _tool_round_identity("a")
    transcript = [
        _mixed_identity_result_message(
            identity,
            conflicting_identity,
            tool_name="side_effect",
        )
    ]

    with pytest.raises(ValueError, match="conflicting tool-round identities"):
        asyncio.run(
            transcript_helpers.tool_round_has_result_messages(
                cast("SessionStore", _TranscriptOnlyStore(transcript)),
                "session-1",
                [_tool_call()],
                tool_round_identity=identity,
            )
        )


def test_pending_round_rejects_newer_conflicting_transcript_message() -> None:
    identity = _tool_round_identity("1")
    conflicting_identity = _tool_round_identity("a")
    transcript = [
        Message.tool_call(
            tool_call_id="call-1",
            tool_name="side_effect",
            arguments={},
            **identity.payload(),
        ),
        Message.tool_result(
            tool_call_id="call-1",
            tool_name="side_effect",
            content="expected result",
            **identity.payload(),
        ),
        Message.tool_result(
            tool_call_id="call-1",
            tool_name="side_effect",
            content="newer conflicting result",
            **conflicting_identity.payload(),
        ),
    ]

    with pytest.raises(ValueError, match="newer conflicting tool-round evidence"):
        asyncio.run(
            transcript_helpers.tool_round_has_result_messages(
                cast("SessionStore", _TranscriptOnlyStore(transcript)),
                "session-1",
                [_tool_call()],
                tool_round_identity=identity,
            )
        )


def test_pending_round_rejects_duplicate_result_messages() -> None:
    identity = _tool_round_identity("1")
    result = Message.tool_result(
        tool_call_id="call-1",
        tool_name="side_effect",
        content="recorded",
        **identity.payload(),
    )

    with pytest.raises(ValueError, match="duplicate tool-round result messages"):
        asyncio.run(
            transcript_helpers.tool_round_has_result_messages(
                cast(
                    "SessionStore",
                    _TranscriptOnlyStore(
                        [
                            Message.tool_call(
                                tool_call_id="call-1",
                                tool_name="side_effect",
                                arguments={},
                                **identity.payload(),
                            ),
                            result,
                            result,
                        ]
                    ),
                ),
                "session-1",
                [_tool_call()],
                tool_round_identity=identity,
            )
        )


def test_pending_round_recovery_retains_checkpoint_without_call_boundary() -> None:
    async def scenario() -> None:
        session_id = "sess_result_without_tool_call_boundary"
        identity = _tool_round_identity("1")
        tool_call = _tool_call(name="record", arguments={"value": 1})
        checkpoint, _pending_round = tool_round_recovery.checkpoint_with_pending_tool_round(
            None,
            agent_name="assistant",
            environment_name=None,
            task_id=None,
            tool_calls=[tool_call],
            policy_outcomes=None,
            structured_output=None,
            tool_round_identity=identity,
        )
        transcript = [
            Message.tool_result(
                tool_call_id="call-1",
                tool_name="record",
                content="unattributable result",
                **identity.payload(),
            )
        ]
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        tool = _RecordingTool()
        app.register_provider(_SequencedProvider([]), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="sequenced-model"),
            tools=[tool],
        )
        await store.create(
            RunRequest(
                session_id=session_id,
                agent_name="assistant",
                messages=[Message.text("user", "record")],
            ),
            identity=SessionIdentity(
                provider_name="sequenced",
                model="sequenced-model",
            ),
        )
        await store.append_transcript_messages(session_id, transcript)
        await store.checkpoint(session_id, checkpoint)
        session = await store.load(session_id)
        assert session is not None

        with pytest.raises(ValueError, match="before the pending round call boundary"):
            async for _event in app._recovery_coordinator.recover_pending_tool_round(
                session=session,
                registered_agent=app._get_registered_agent("assistant"),
                registered_environment=None,
                messages=[message.model_copy(deep=True) for message in transcript],
            ):
                pass

        assert await store.load_checkpoint(session_id) == checkpoint
        assert await store.load_transcript(session_id) == transcript
        assert await store.load_events(session_id) == []
        assert tool.values == []

    asyncio.run(scenario())


def test_pending_round_recovery_retains_checkpoint_for_mixed_transcript_identity() -> None:
    async def scenario() -> None:
        session_id = "sess_mixed_transcript_tool_round_identity"
        identity = _tool_round_identity("1")
        conflicting_identity = _tool_round_identity("a")
        tool_call = _tool_call(name="record", arguments={"value": 1})
        checkpoint, _pending_round = tool_round_recovery.checkpoint_with_pending_tool_round(
            None,
            agent_name="assistant",
            environment_name=None,
            task_id=None,
            tool_calls=[tool_call],
            policy_outcomes=None,
            structured_output=None,
            tool_round_identity=identity,
        )
        transcript = [
            Message.tool_call(
                tool_call_id="call-1",
                tool_name="record",
                arguments={"value": 1},
                **identity.payload(),
            ),
            Message.tool_result(
                tool_call_id="call-1",
                tool_name="record",
                content="expected result",
                **identity.payload(),
            ),
            Message.tool_result(
                tool_call_id="call-1",
                tool_name="record",
                content="newer conflicting result",
                **conflicting_identity.payload(),
            ),
        ]
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        tool = _RecordingTool()
        app.register_provider(_SequencedProvider([]), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="sequenced-model"),
            tools=[tool],
        )
        await store.create(
            RunRequest(
                session_id=session_id,
                agent_name="assistant",
                messages=[Message.text("user", "record")],
            ),
            identity=SessionIdentity(
                provider_name="sequenced",
                model="sequenced-model",
            ),
        )
        await store.append_transcript_messages(session_id, transcript)
        await store.checkpoint(session_id, checkpoint)
        session = await store.load(session_id)
        assert session is not None

        with pytest.raises(ValueError, match="newer conflicting tool-round evidence"):
            async for _event in app._recovery_coordinator.recover_pending_tool_round(
                session=session,
                registered_agent=app._get_registered_agent("assistant"),
                registered_environment=None,
                messages=[message.model_copy(deep=True) for message in transcript],
            ):
                pass

        assert await store.load_checkpoint(session_id) == checkpoint
        assert await store.load_transcript(session_id) == transcript
        assert await store.load_events(session_id) == []
        assert tool.values == []

    asyncio.run(scenario())


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
