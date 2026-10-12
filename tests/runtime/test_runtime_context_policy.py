"""Context policy contracts."""

from __future__ import annotations

import asyncio
import json
import math
from typing import Any

import pytest
from pydantic import ValidationError
from tests.runtime.runtime_provider_support import CountingProvider, FakeProvider
from tests.runtime.runtime_session_support import (
    _test_session,
    assert_only_model_step_publication_checkpoint,
    checkpoint_without_model_step_publication,
    collect_events,
    collect_resume_events,
)
from tests.runtime.runtime_tool_support import EchoTool

import cayu.context.base as runtime_context_module
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.artifacts.attachments import file_attachment
from cayu.context.base import (
    CheckpointCompactionContextPolicy,
    ContextBuildResult,
    ContextPolicy,
    ContextPressureEstimate,
    ContextPressureOverhead,
    ContextRequest,
    ContextUsageState,
    DefaultContextPolicy,
    MessageWindowContextPolicy,
    ObservedDeltaContextEstimator,
    RecentTurnsContextPolicy,
    RuntimeManagedContextPolicy,
    UsageTriggeredContextPolicy,
)
from cayu.events import EventType
from cayu.messages import Message, TextPart
from cayu.providers.base import (
    InputTokenCountConfidence,
    InputTokenCountMethod,
    InputTokenCountResult,
    ModelContextPressureProfile,
    ModelStreamEvent,
)
from cayu.sessions.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
)
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.requests import ResumeRequest, RunRequest
from cayu.tools.base import Tool, ToolContext, ToolResult, ToolSpec


def test_cayu_app_context_policy_can_trim_model_facing_messages():
    class LastMessagePolicy(ContextPolicy):
        def __init__(self) -> None:
            self.seen: list[ContextRequest] = []

        async def build(self, request: ContextRequest) -> list[Message]:
            self.seen.append(request)
            # Messages are frozen; policies cannot corrupt the transcript.
            with pytest.raises(ValidationError):
                request.messages[0].content[0].text = "mutated"  # type: ignore[misc]
            return [request.messages[-1]]

    policy = LastMessagePolicy()
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("trimmed"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=policy,
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_trim_context",
                messages=[
                    Message.text("user", "old context"),
                    Message.text("user", "current request"),
                ],
                metadata={"source": "test"},
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert len(policy.seen) == 1
    assert policy.seen[0].session.id == "sess_trim_context"
    assert policy.seen[0].agent.name == "assistant"
    assert policy.seen[0].step == 1
    assert policy.seen[0].metadata == {"source": "test"}
    assert [message.content[0].text for message in provider.requests[0].messages] == [
        "current request"
    ]

    transcript = asyncio.run(store.load_transcript("sess_trim_context"))
    assert [message.content[0].text for message in transcript] == [
        "old context",
        "current request",
        "trimmed",
    ]


def test_cayu_app_context_policy_can_replace_tool_results_for_model_only():
    class CompactToolResultPolicy(ContextPolicy):
        async def build(self, request: ContextRequest) -> list[Message]:
            compacted: list[Message] = []
            for message in request.messages:
                if message.role == "tool":
                    compacted.append(
                        Message.tool_result(
                            tool_call_id=message.content[0].tool_call_id,
                            tool_name=message.content[0].tool_name,
                            content="[tool result compacted]",
                            structured={"compacted": True},
                        )
                    )
                else:
                    compacted.append(message)
            return compacted

    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="echo",
                    arguments={"text": "large tool output"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
        context_policy=CompactToolResultPolicy(),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_tool_result",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    provider_tool_result = provider.requests[1].messages[-1].content[0]
    assert provider_tool_result.content == "[tool result compacted]"
    assert provider_tool_result.structured == {"compacted": True}

    transcript = asyncio.run(store.load_transcript("sess_compact_tool_result"))
    stored_tool_result = transcript[2].content[0]
    assert stored_tool_result.content == "large tool output"
    assert stored_tool_result.structured == {
        "agent": "assistant",
        "echoed": "large tool output",
    }


def test_cayu_app_fails_cleanly_when_context_policy_returns_invalid_output():
    class BadPolicy(ContextPolicy):
        async def build(self, request: ContextRequest):
            return tuple(request.messages)

    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("should not run"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=BadPolicy(),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_bad_context_policy",
                messages=[Message.text("user", "hi")],
            ),
        )
    )

    assert provider.requests == []
    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    assert (
        events[-1].payload["error"]
        == "ContextPolicy.build() must return a list of Message instances."
    )
    transcript = asyncio.run(store.load_transcript("sess_bad_context_policy"))
    assert [message.content[0].text for message in transcript] == ["hi"]


def test_cayu_app_rejects_context_policy_that_cuts_through_tool_round():
    class OrphanToolResultPolicy(ContextPolicy):
        async def build(self, request: ContextRequest) -> list[Message]:
            return [request.messages[-1]]

    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="echo",
                    arguments={"text": "large tool output"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("should not run"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
        context_policy=OrphanToolResultPolicy(),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_orphan_tool_result",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    assert len(provider.requests) == 1
    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_FAILED,
    ]
    assert "tool results without preceding assistant tool calls" in events[-1].payload["error"]


def test_builtin_context_policies_trim_recent_history_safely():
    messages = [
        Message.text("system", "You are careful."),
        Message.text("user", "old"),
        Message.text("assistant", "old answer"),
        Message.text("user", "current"),
    ]

    turns_policy = RecentTurnsContextPolicy(max_user_turns=1)
    messages_policy = MessageWindowContextPolicy(max_messages=2)

    async def build_contexts() -> tuple[list[Message], list[Message]]:
        request = ContextRequest(
            session=_test_session(),
            agent=AgentSpec(name="assistant", model="fake-model"),
            messages=messages,
            step=1,
        )
        return (
            await turns_policy.build(request),
            await messages_policy.build(request),
        )

    turns_context, messages_context = asyncio.run(build_contexts())

    assert [message.role for message in turns_context] == ["system", "user"]
    assert [message.content[0].text for message in turns_context] == [
        "You are careful.",
        "current",
    ]
    assert [message.role for message in messages_context] == ["system", "user"]
    assert [message.content[0].text for message in messages_context] == [
        "You are careful.",
        "current",
    ]


def test_context_policy_receives_previous_actual_context_usage_on_next_call():
    class UsageAwarePolicy(ContextPolicy):
        def __init__(self) -> None:
            self.requests: list[ContextRequest] = []

        async def build(self, request: ContextRequest) -> list[Message]:
            self.requests.append(request)
            if (
                request.context_usage.last_input_tokens is not None
                and request.context_usage.last_input_tokens >= 50
            ):
                return [request.messages[-1]]
            return request.messages

    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 100, "output_tokens": 4}}),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 20, "output_tokens": 2}}),
            ],
            [
                ModelStreamEvent.text_delta("third answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 1}}),
            ],
        ]
    )
    policy = UsageAwarePolicy()
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=policy,
    )

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="usage_context_policy",
                messages=[Message.text("user", "first request")],
            ),
        )
    )
    asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="usage_context_policy",
                messages=[Message.text("user", "second request")],
            ),
        )
    )
    asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="usage_context_policy",
                messages=[Message.text("user", "third request")],
            ),
        )
    )

    assert len(policy.requests) == 3
    assert policy.requests[0].context_usage.last_input_tokens is None
    assert policy.requests[1].context_usage.last_input_tokens == 100
    assert policy.requests[1].context_usage.last_output_tokens == 4
    assert policy.requests[1].context_usage.last_total_tokens == 104
    assert policy.requests[1].context_usage.last_provider_name == "fake"
    assert policy.requests[1].context_usage.last_requested_model == "fake-model"
    assert policy.requests[1].context_usage.last_model == "fake-model"
    assert policy.requests[2].context_usage.last_input_tokens == 20
    assert len(provider.requests[0].messages) == 1
    assert len(provider.requests[1].messages) == 1
    assert provider.requests[1].messages[0].content[0].text == "second request"
    assert len(provider.requests[2].messages) > 1
    assert provider.requests[2].messages[-1].content[0].text == "third request"


def test_observed_delta_context_estimator_estimates_only_messages_after_cursor():
    estimator = ObservedDeltaContextEstimator(chars_per_token=4, json_chars_per_token=2)
    usage = ContextUsageState(
        last_input_tokens=40,
        last_transcript_cursor=2,
        input_coverage=runtime_context_module.context_input_coverage(
            [
                Message.text("user", "already counted"),
                Message.text("assistant", "already counted answer"),
            ],
            transcript_cursor=2,
        ),
    )
    messages = [
        Message.text("user", "already counted"),
        Message.text("assistant", "already counted answer"),
        Message.text("user", "abcd" * 5),
        Message.tool_result(
            tool_call_id="call_1",
            tool_name="lookup",
            content="xy" * 5,
            structured={"ok": True},
        ),
    ]

    estimate = estimator.estimate(usage=usage, messages=messages)

    assert isinstance(estimate, ContextPressureEstimate)
    assert estimate.method == "observed_plus_estimated_delta"
    assert estimate.confidence == "estimated"
    assert estimate.observed_context_input_tokens == 40
    assert estimate.estimated_delta_input_tokens == 8
    assert estimate.estimated_context_input_tokens == 48
    assert estimate.reserved_output_tokens == 0
    assert estimate.estimated_context_window_tokens == 48
    assert estimate.anchor_transcript_cursor == 2
    assert estimate.current_transcript_cursor == 4
    assert estimate.estimated_message_count == 2


def test_observed_delta_context_estimator_can_estimate_full_request_overhead():
    estimator = ObservedDeltaContextEstimator(chars_per_token=4, json_chars_per_token=2)
    usage = ContextUsageState(last_input_tokens=40, last_transcript_cursor=2)
    messages = [Message.text("user", "abcd" * 5)]
    overhead = ContextPressureOverhead(
        tools=[
            {
                "name": "lookup",
                "description": "x" * 40,
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
            }
        ],
        structured_output_instruction="return structured output",
        request_options={
            "step": 2,
            "structured_output": {"name": "answer"},
            "openai": {"tool_choice": "auto"},
        },
    )

    estimate = estimator.estimate_full_request(
        usage=usage,
        messages=messages,
        overhead=overhead,
    )

    assert estimate.method == "local_full_request_estimate"
    assert estimate.observed_context_input_tokens == 40
    assert estimate.estimated_message_input_tokens == 5
    assert estimate.estimated_tool_schema_input_tokens > 0
    assert estimate.estimated_structured_output_input_tokens > 0
    assert estimate.estimated_request_options_input_tokens > 0
    assert estimate.estimated_context_input_tokens == (
        estimate.estimated_message_input_tokens
        + estimate.estimated_tool_schema_input_tokens
        + estimate.estimated_structured_output_input_tokens
        + estimate.estimated_request_options_input_tokens
    )
    assert estimate.estimated_context_window_tokens == estimate.estimated_context_input_tokens


def test_observed_delta_context_estimator_counts_only_overhead_delta_after_anchor():
    estimator = ObservedDeltaContextEstimator(chars_per_token=4, json_chars_per_token=2)
    overhead = ContextPressureOverhead(
        tools=[
            {
                "name": "lookup",
                "description": "x" * 80,
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
            }
        ],
        request_options={"openai": {"tool_choice": "auto"}},
    )
    overhead_tokens = estimator.estimate_full_request(
        usage=ContextUsageState(),
        messages=[Message.text("user", "seed")],
        overhead=overhead,
    ).estimated_request_overhead_input_tokens
    usage = ContextUsageState(
        last_input_tokens=100,
        last_transcript_cursor=1,
        input_coverage=runtime_context_module.context_input_coverage(
            [Message.text("user", "already counted")],
            transcript_cursor=1,
        ),
        last_context_overhead_input_tokens=overhead_tokens,
    )
    messages = [
        Message.text("user", "already counted"),
        Message.text("user", "abcd" * 5),
    ]

    stable = estimator.estimate_anchored_request(
        usage=usage,
        messages=messages,
        overhead=overhead,
    )
    increased = estimator.estimate_anchored_request(
        usage=usage.model_copy(update={"last_context_overhead_input_tokens": overhead_tokens - 5}),
        messages=messages,
        overhead=overhead,
    )
    decreased = estimator.estimate_anchored_request(
        usage=usage.model_copy(update={"last_context_overhead_input_tokens": overhead_tokens + 5}),
        messages=messages,
        overhead=overhead,
    )

    assert stable.estimated_message_input_tokens == 5
    assert stable.estimated_request_overhead_input_tokens == overhead_tokens
    assert stable.previous_request_overhead_input_tokens == overhead_tokens
    assert stable.estimated_request_overhead_delta_tokens == 0
    assert stable.estimated_delta_input_tokens == 5
    assert stable.estimated_context_input_tokens == 105

    assert increased.estimated_request_overhead_delta_tokens == 5
    assert increased.estimated_context_input_tokens == 110
    assert decreased.estimated_request_overhead_delta_tokens == -5
    assert decreased.estimated_context_input_tokens == 100


def test_observed_delta_context_estimator_uses_provider_specific_image_floor():
    estimator = ObservedDeltaContextEstimator(chars_per_token=5)
    from cayu.messages import FilePart

    user_image_message = Message(
        role="user",
        content=[
            TextPart(text="look"),
            FilePart(
                attachment=file_attachment(
                    artifact_id="img_user",
                    kind="image",
                    filename="user.png",
                    content_type="image/png",
                    size_bytes=68,
                )
            ),
        ],
    )
    image_message = Message.tool_result(
        tool_call_id="call_image",
        tool_name="image_tool",
        content="image result",
        artifacts=[
            file_attachment(
                artifact_id="img_1",
                kind="image",
                filename="pixel.png",
                content_type="image/png",
                size_bytes=68,
            )
        ],
    )
    document_message = Message.tool_result(
        tool_call_id="call_document",
        tool_name="document_tool",
        content="document result",
        artifacts=[
            file_attachment(
                artifact_id="doc_1",
                kind="document",
                filename="tiny.pdf",
                content_type="application/pdf",
                size_bytes=120,
            )
        ],
    )

    generic_tokens = estimator.estimate_full_request(
        usage=ContextUsageState(),
        messages=[user_image_message, image_message, document_message],
        overhead=ContextPressureOverhead(image_min_tokens=32),
    )
    anthropic_tokens = estimator.estimate_full_request(
        usage=ContextUsageState(),
        messages=[user_image_message, image_message, document_message],
        overhead=ContextPressureOverhead(
            image_min_tokens=100,
            document_min_tokens=1800,
        ),
    )

    assert generic_tokens.estimated_attachment_input_tokens < 200
    assert anthropic_tokens.estimated_attachment_input_tokens >= 2000
    assert (
        anthropic_tokens.estimated_attachment_input_tokens
        > generic_tokens.estimated_attachment_input_tokens
    )


def test_observed_delta_context_estimator_uses_adaptive_text_density():
    prose = (
        "Remote sandbox Git authentication should use a brokered Git HTTP proxy. "
        "Credentials stay outside the sandbox boundary. " * 100
    )
    dense_json = (
        '{"path":"src/cayu/context/base.py","line":123,"value":"x_y-z.abc/def","ok":true}\n' * 100
    )
    pretty_json = (
        '{\n  "path": "src/cayu/context/base.py",\n  "line": 123,\n'
        '  "value": "x_y-z.abc/def",\n  "ok": true\n}\n' * 100
    )
    logs = (
        "2026-07-02T12:34:56Z INFO session=sess_123 step=4 "
        "tool=search_knowledge latency_ms=42 status=ok input_tokens=14947\n" * 100
    )
    csv = (
        "timestamp,session_id,agent,model,input_tokens,output_tokens,status\n"
        "2026-07-02T12:34:56Z,sess_123,assistant,gpt-5.5,14947,7,ok\n" * 100
    )
    code = (
        "def estimate_context_pressure(request):\n"
        "    if request.count_input_tokens is None:\n"
        "        return estimate\n"
        "    return request.count_input_tokens(messages)\n" * 100
    )
    markdown = (
        "# Heading\n\n- `code/path.py`: explanation with punctuation, numbers 12345, "
        'and JSON {"ok": true}.\n' * 100
    )
    estimator = ObservedDeltaContextEstimator(chars_per_token=5)

    prose_tokens = estimator.estimate_message_tokens(Message.text("user", prose))
    dense_json_tokens = estimator.estimate_message_tokens(Message.text("user", dense_json))
    pretty_json_tokens = estimator.estimate_message_tokens(Message.text("user", pretty_json))
    logs_tokens = estimator.estimate_message_tokens(Message.text("user", logs))
    csv_tokens = estimator.estimate_message_tokens(Message.text("user", csv))
    code_tokens = estimator.estimate_message_tokens(Message.text("user", code))
    markdown_tokens = estimator.estimate_message_tokens(Message.text("user", markdown))

    assert prose_tokens == math.ceil(len(prose) / 5)
    assert dense_json_tokens == math.ceil(len(dense_json) / 3)
    assert pretty_json_tokens == math.ceil(len(pretty_json) / 3)
    assert logs_tokens == math.ceil(len(logs) / 2.75)
    assert csv_tokens == math.ceil(len(csv) / 2.5)
    assert code_tokens == math.ceil(len(code) / 5)
    assert markdown_tokens == math.ceil(len(markdown) / 3.75)


def test_observed_delta_context_estimator_uses_tool_schema_density_override():
    estimator = ObservedDeltaContextEstimator(chars_per_token=5, json_chars_per_token=5)
    tools = [
        {
            "name": "heavy",
            "description": "Search many records. " * 200,
            "input_schema": {
                "type": "object",
                "properties": {
                    f"field_{index}": {
                        "type": "string",
                        "description": "Filter value " * 20,
                    }
                    for index in range(20)
                },
            },
        }
    ]

    default_tokens = estimator.estimate_tool_schema_tokens(tools)
    openai_profile_tokens = estimator.estimate_tool_schema_tokens(
        tools,
        chars_per_token=6,
    )

    assert default_tokens > openai_profile_tokens
    assert openai_profile_tokens == math.ceil(
        len(json.dumps(tools[0], ensure_ascii=False, separators=(",", ":"), sort_keys=True)) / 6
    )


def test_context_policy_receives_estimated_context_pressure_on_next_call():
    class PressureAwarePolicy(ContextPolicy):
        def __init__(self) -> None:
            self.requests: list[ContextRequest] = []

        async def build(self, request: ContextRequest) -> list[Message]:
            self.requests.append(request)
            return request.messages

    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 40, "output_tokens": 4}}),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 2}}),
            ],
        ]
    )
    policy = PressureAwarePolicy()
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=policy,
    )

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="context_pressure_estimate",
                messages=[Message.text("user", "first request")],
            ),
        )
    )
    asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="context_pressure_estimate",
                messages=[Message.text("user", "abcd" * 4)],
            ),
        )
    )

    first_completed = next(
        event for event in first_events if event.type == EventType.MODEL_COMPLETED
    )
    assert first_completed.payload["transcript_cursor"] == 2
    assert len(policy.requests) == 2
    pressure = policy.requests[1].context_usage.input_pressure
    assert pressure is not None
    assert pressure.observed_context_input_tokens == 40
    assert pressure.estimated_delta_input_tokens == 7
    assert pressure.estimated_context_input_tokens == 47
    assert pressure.estimated_context_window_tokens == 47
    assert pressure.anchor_transcript_cursor == 1
    assert pressure.current_transcript_cursor == 3
    assert pressure.estimated_message_count == 2


def test_context_policy_input_pressure_uses_provider_profile_image_floor():
    class ImageFloorProvider(FakeProvider):
        @property
        def context_pressure_profile(self) -> ModelContextPressureProfile:
            return ModelContextPressureProfile(
                image_min_tokens=100,
                document_min_tokens=1800,
            )

    class PressureAwarePolicy(ContextPolicy):
        def __init__(self) -> None:
            self.requests: list[ContextRequest] = []

        async def build(self, request: ContextRequest) -> list[Message]:
            self.requests.append(request)
            return request.messages

    provider = ImageFloorProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 40, "output_tokens": 4}}),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 50, "output_tokens": 2}}),
            ],
        ]
    )
    policy = PressureAwarePolicy()
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=policy,
    )

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="context_pressure_provider_profile",
                messages=[Message.text("user", "first request")],
            ),
        )
    )
    asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="context_pressure_provider_profile",
                messages=[
                    Message.tool_result(
                        tool_call_id="call_image",
                        tool_name="image_tool",
                        content="image result",
                        artifacts=[
                            file_attachment(
                                artifact_id="img_1",
                                kind="image",
                                filename="pixel.png",
                                content_type="image/png",
                                size_bytes=68,
                            )
                        ],
                    ),
                    Message.tool_result(
                        tool_call_id="call_document",
                        tool_name="document_tool",
                        content="document result",
                        artifacts=[
                            file_attachment(
                                artifact_id="doc_1",
                                kind="document",
                                filename="tiny.pdf",
                                content_type="application/pdf",
                                size_bytes=120,
                            )
                        ],
                    ),
                ],
            ),
        )
    )

    assert len(policy.requests) == 2
    pressure = policy.requests[1].context_usage.input_pressure
    assert pressure is not None
    assert pressure.estimated_attachment_input_tokens >= 1900


def test_usage_triggered_context_policy_can_trigger_on_estimated_context_pressure():
    class LargeSchemaTool(Tool):
        spec = ToolSpec(
            name="large_schema",
            description="x" * 200,
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "y" * 200,
                    }
                },
            },
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            return ToolResult(content="")

    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 40, "output_tokens": 4}}),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 2}}),
            ],
        ]
    )
    policy = UsageTriggeredContextPolicy(
        base_policy=DefaultContextPolicy(),
        triggered_policy=MessageWindowContextPolicy(max_messages=1),
        trigger_estimated_context_tokens=45,
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[LargeSchemaTool()],
        context_policy=policy,
    )

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="usage_triggered_estimated_pressure",
                messages=[Message.text("user", "first request")],
            ),
        )
    )
    second_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="usage_triggered_estimated_pressure",
                messages=[Message.text("user", "abcd" * 5)],
            ),
        )
    )

    assert len(provider.requests[0].messages) == 1
    assert len(provider.requests[1].messages) == 1
    assert provider.requests[1].messages[0].content[0].text == "abcd" * 5
    checkpoint_events = [
        event for event in first_events if event.type == EventType.SESSION_CHECKPOINTED
    ]
    assert len(checkpoint_events) == 1
    payload = checkpoint_events[0].payload
    assert payload["checkpoint"] == "usage_triggered_context"
    assert payload["min_input_tokens"] is None
    assert payload["trigger_estimated_context_tokens"] == 45
    assert payload["reserved_output_tokens"] == 0
    assert payload["min_total_tokens"] is None
    assert payload["last_input_tokens"] is None
    assert payload["last_total_tokens"] is None
    assert payload["last_transcript_cursor"] is None
    assert payload["estimated_context_input_tokens"] >= 45
    assert payload["estimated_context_window_tokens"] >= 45
    assert payload["estimated_delta_input_tokens"] == payload["estimated_context_input_tokens"]
    assert not [event for event in second_events if event.type == EventType.SESSION_CHECKPOINTED]


def test_usage_triggered_context_policy_does_not_double_count_stable_overhead():
    class LargeSchemaTool(Tool):
        spec = ToolSpec(
            name="large_schema",
            description="x" * 1000,
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "y" * 1000,
                    }
                },
            },
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            return ToolResult(content="")

    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 1000, "output_tokens": 4}}),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 1010, "output_tokens": 2}}),
            ],
        ]
    )
    policy = UsageTriggeredContextPolicy(
        base_policy=DefaultContextPolicy(),
        triggered_policy=MessageWindowContextPolicy(max_messages=1),
        trigger_estimated_context_tokens=1250,
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[LargeSchemaTool()],
        context_policy=policy,
    )

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="usage_triggered_stable_overhead",
                messages=[Message.text("user", "first request")],
            ),
        )
    )
    second_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="usage_triggered_stable_overhead",
                messages=[Message.text("user", "abcd" * 5)],
            ),
        )
    )

    first_completed = next(
        event for event in first_events if event.type == EventType.MODEL_COMPLETED
    )
    assert (
        first_completed.payload["context_pressure"]["estimated_request_overhead_input_tokens"] > 300
    )
    assert len(provider.requests[1].messages) == 3
    assert not [event for event in second_events if event.type == EventType.SESSION_CHECKPOINTED]


def test_usage_triggered_context_policy_full_estimates_same_length_projection():
    class ReplacingPolicy(ContextPolicy):
        async def build(self, request: ContextRequest) -> list[Message]:
            return [Message.text("user", "x" * 1000) for _ in request.messages]

    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 100, "output_tokens": 4}}),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 1010, "output_tokens": 2}}),
            ],
        ]
    )
    policy = UsageTriggeredContextPolicy(
        base_policy=ReplacingPolicy(),
        triggered_policy=MessageWindowContextPolicy(max_messages=1),
        trigger_estimated_context_tokens=450,
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=policy,
    )

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="usage_triggered_same_length_projection",
                messages=[Message.text("user", "first request")],
            ),
        )
    )
    second_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="usage_triggered_same_length_projection",
                messages=[Message.text("user", "second request")],
            ),
        )
    )

    assert not [event for event in first_events if event.type == EventType.SESSION_CHECKPOINTED]
    assert len(provider.requests[0].messages) == 1
    assert provider.requests[0].messages[0].content[0].text == "x" * 1000
    assert len(provider.requests[1].messages) == 1
    assert provider.requests[1].messages[0].content[0].text == "second request"
    checkpoint_event = next(
        event for event in second_events if event.type == EventType.SESSION_CHECKPOINTED
    )
    assert checkpoint_event.payload["estimated_context_input_tokens"] >= 300


def test_usage_triggered_context_policy_estimates_base_policy_model_facing_context():
    class InjectingPolicy(ContextPolicy):
        async def build(self, request: ContextRequest) -> list[Message]:
            return [
                *request.messages,
                Message.text("user", "injected context " + ("x" * 400)),
            ]

    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 40, "output_tokens": 4}}),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 2}}),
            ],
        ]
    )
    policy = UsageTriggeredContextPolicy(
        base_policy=InjectingPolicy(),
        triggered_policy=MessageWindowContextPolicy(max_messages=1),
        trigger_estimated_context_tokens=80,
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=policy,
    )

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="usage_triggered_estimated_policy_context",
                messages=[Message.text("user", "first request")],
            ),
        )
    )
    second_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="usage_triggered_estimated_policy_context",
                messages=[Message.text("user", "second request")],
            ),
        )
    )

    assert len(provider.requests[0].messages) == 1
    assert len(provider.requests[1].messages) == 1
    assert provider.requests[1].messages[0].content[0].text == "second request"
    checkpoint_event = next(
        event for event in first_events if event.type == EventType.SESSION_CHECKPOINTED
    )
    assert checkpoint_event.payload["estimated_context_input_tokens"] >= 80
    assert not [event for event in second_events if event.type == EventType.SESSION_CHECKPOINTED]


def test_usage_triggered_context_policy_includes_reserved_output_in_estimated_trigger():
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 20, "output_tokens": 4}}),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 2}}),
            ],
        ]
    )
    policy = UsageTriggeredContextPolicy(
        base_policy=DefaultContextPolicy(),
        triggered_policy=MessageWindowContextPolicy(max_messages=1),
        trigger_estimated_context_tokens=94,
        reserved_output_tokens=90,
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=policy,
    )

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="usage_triggered_reserved_output",
                messages=[Message.text("user", "abcd" * 5)],
            ),
        )
    )

    assert len(provider.requests[0].messages) == 1
    checkpoint_event = next(
        event for event in first_events if event.type == EventType.SESSION_CHECKPOINTED
    )
    assert checkpoint_event.payload["trigger_estimated_context_tokens"] == 94
    assert checkpoint_event.payload["reserved_output_tokens"] == 90
    assert checkpoint_event.payload["estimated_context_input_tokens"] < 100
    assert checkpoint_event.payload["estimated_context_window_tokens"] >= 94


def test_usage_triggered_context_policy_provider_count_can_prevent_estimated_false_positive():
    class LargeSchemaTool(Tool):
        spec = ToolSpec(
            name="large_schema",
            description="x" * 400,
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "y" * 400,
                    }
                },
            },
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            return ToolResult(content="")

    provider = CountingProvider(
        [
            ModelStreamEvent.text_delta("answer"),
            ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 2}}),
        ],
        count_result=InputTokenCountResult(
            input_tokens=10,
            method=InputTokenCountMethod.OFFICIAL,
            confidence=InputTokenCountConfidence.HIGH,
        ),
    )
    policy = UsageTriggeredContextPolicy(
        base_policy=DefaultContextPolicy(),
        triggered_policy=MessageWindowContextPolicy(max_messages=1),
        trigger_estimated_context_tokens=45,
        verify_estimate_with_provider_count=True,
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[LargeSchemaTool()],
        context_policy=policy,
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="usage_triggered_provider_count_false_positive",
                messages=[Message.text("user", "first request")],
            ),
        )
    )

    assert len(provider.count_requests) == 1
    assert provider.count_requests[0].tools
    assert not [event for event in events if event.type == EventType.SESSION_CHECKPOINTED]


def test_usage_triggered_context_policy_provider_count_is_not_called_when_far_from_threshold():
    provider = CountingProvider(
        [
            ModelStreamEvent.text_delta("answer"),
            ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 2}}),
        ],
        count_result=InputTokenCountResult(
            input_tokens=10,
            method=InputTokenCountMethod.OFFICIAL,
            confidence=InputTokenCountConfidence.HIGH,
        ),
    )
    policy = UsageTriggeredContextPolicy(
        base_policy=DefaultContextPolicy(),
        triggered_policy=MessageWindowContextPolicy(max_messages=1),
        trigger_estimated_context_tokens=10_000,
        verify_estimate_with_provider_count=True,
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=policy,
    )

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="usage_triggered_provider_count_far",
                messages=[Message.text("user", "first request")],
            ),
        )
    )

    assert provider.count_requests == []


def test_usage_triggered_context_policy_large_delta_can_request_provider_count():
    provider = CountingProvider(
        [
            ModelStreamEvent.text_delta("answer"),
            ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 2}}),
        ],
        count_result=InputTokenCountResult(
            input_tokens=12_000,
            method=InputTokenCountMethod.OFFICIAL,
            confidence=InputTokenCountConfidence.HIGH,
        ),
    )
    policy = UsageTriggeredContextPolicy(
        base_policy=DefaultContextPolicy(),
        triggered_policy=MessageWindowContextPolicy(max_messages=1),
        trigger_estimated_context_tokens=10_000,
        verify_estimate_with_provider_count=True,
        provider_count_min_delta_tokens=20,
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=policy,
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="usage_triggered_provider_count_large_delta",
                messages=[Message.text("user", "x" * 200)],
            ),
        )
    )

    assert len(provider.count_requests) == 1
    checkpoint_event = next(
        event for event in events if event.type == EventType.SESSION_CHECKPOINTED
    )
    assert checkpoint_event.payload["provider_count_input_tokens"] == 12_000
    assert checkpoint_event.payload["provider_count_context_window_tokens"] == 12_000


def test_usage_triggered_context_policy_stays_triggered_after_previous_actual_usage():
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 100, "output_tokens": 4}}),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 20, "output_tokens": 2}}),
            ],
            [
                ModelStreamEvent.text_delta("third answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 1}}),
            ],
        ]
    )
    policy = UsageTriggeredContextPolicy(
        base_policy=DefaultContextPolicy(),
        triggered_policy=MessageWindowContextPolicy(max_messages=1),
        min_input_tokens=50,
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=policy,
    )

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="usage_triggered_policy",
                messages=[Message.text("user", "first request")],
            ),
        )
    )
    second_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="usage_triggered_policy",
                messages=[Message.text("user", "second request")],
            ),
        )
    )
    asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="usage_triggered_policy",
                messages=[Message.text("user", "third request")],
            ),
        )
    )

    assert len(provider.requests[0].messages) == 1
    assert len(provider.requests[1].messages) == 1
    assert provider.requests[1].messages[0].content[0].text == "second request"
    assert len(provider.requests[2].messages) == 1
    assert provider.requests[2].messages[0].content[0].text == "third request"
    checkpoint_events = [
        event for event in second_events if event.type == EventType.SESSION_CHECKPOINTED
    ]
    assert len(checkpoint_events) == 1
    second_model_started = next(
        event for event in second_events if event.type == EventType.MODEL_STARTED
    )
    assert checkpoint_events[0].payload == {
        "checkpoint": "usage_triggered_context",
        "execution_profile_fingerprint": second_model_started.payload[
            "execution_profile_fingerprint"
        ],
        "min_input_tokens": 50,
        "min_total_tokens": None,
        "last_input_tokens": 100,
        "last_total_tokens": 104,
        "model_step_id": second_model_started.payload["model_step_id"],
    }
    checkpoint = asyncio.run(app.session_store.load_checkpoint("usage_triggered_policy"))
    assert checkpoint_without_model_step_publication(checkpoint) == {
        CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
        "usage_triggered_context": {
            "version": 1,
            "min_input_tokens": 50,
            "min_total_tokens": None,
            "last_input_tokens": 100,
            "last_total_tokens": 104,
        },
    }


def test_usage_triggered_context_policy_preserves_marker_with_checkpoint_policy():
    class ReplacingCheckpointPolicy(RuntimeManagedContextPolicy):
        def __init__(self) -> None:
            self.calls = 0

        async def build_with_checkpoint(
            self,
            request: ContextRequest,
            *,
            checkpoint: dict[str, Any] | None,
        ) -> ContextBuildResult:
            self.calls += 1
            return ContextBuildResult(
                messages=[request.messages[-1]],
                checkpoint={"triggered_policy": {"calls": self.calls}},
                checkpoint_event_payload={
                    "checkpoint": "triggered_policy",
                    "calls": self.calls,
                },
            )

    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 100, "output_tokens": 4}}),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 20, "output_tokens": 2}}),
            ],
            [
                ModelStreamEvent.text_delta("third answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 1}}),
            ],
        ]
    )
    triggered_policy = ReplacingCheckpointPolicy()
    policy = UsageTriggeredContextPolicy(
        base_policy=DefaultContextPolicy(),
        triggered_policy=triggered_policy,
        min_input_tokens=50,
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=policy,
    )

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="usage_triggered_policy_checkpoint_merge",
                messages=[Message.text("user", "first request")],
            ),
        )
    )
    asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="usage_triggered_policy_checkpoint_merge",
                messages=[Message.text("user", "second request")],
            ),
        )
    )
    asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="usage_triggered_policy_checkpoint_merge",
                messages=[Message.text("user", "third request")],
            ),
        )
    )

    assert triggered_policy.calls == 2
    assert len(provider.requests[1].messages) == 1
    assert len(provider.requests[2].messages) == 1
    checkpoint = asyncio.run(
        app.session_store.load_checkpoint("usage_triggered_policy_checkpoint_merge")
    )
    assert checkpoint_without_model_step_publication(checkpoint) == {
        CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
        "triggered_policy": {"calls": 2},
        "usage_triggered_context": {
            "version": 1,
            "min_input_tokens": 50,
            "min_total_tokens": None,
            "last_input_tokens": 100,
            "last_total_tokens": 104,
        },
    }


def test_usage_triggered_context_policy_can_be_last_call_only():
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 100, "output_tokens": 4}}),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 20, "output_tokens": 2}}),
            ],
            [
                ModelStreamEvent.text_delta("third answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 1}}),
            ],
        ]
    )
    policy = UsageTriggeredContextPolicy(
        base_policy=DefaultContextPolicy(),
        triggered_policy=MessageWindowContextPolicy(max_messages=1),
        min_input_tokens=50,
        sticky=False,
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=policy,
    )

    asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="usage_triggered_policy_last_call_only",
                messages=[Message.text("user", "first request")],
            ),
        )
    )
    asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="usage_triggered_policy_last_call_only",
                messages=[Message.text("user", "second request")],
            ),
        )
    )
    asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="usage_triggered_policy_last_call_only",
                messages=[Message.text("user", "third request")],
            ),
        )
    )

    assert len(provider.requests[0].messages) == 1
    assert len(provider.requests[1].messages) == 1
    assert provider.requests[1].messages[0].content[0].text == "second request"
    assert len(provider.requests[2].messages) > 1
    assert provider.requests[2].messages[-1].content[0].text == "third request"
    checkpoint = asyncio.run(
        app.session_store.load_checkpoint("usage_triggered_policy_last_call_only")
    )
    assert_only_model_step_publication_checkpoint(checkpoint)


def test_usage_triggered_context_policy_supports_total_token_threshold():
    policy = UsageTriggeredContextPolicy(
        base_policy=MessageWindowContextPolicy(max_messages=3),
        triggered_policy=MessageWindowContextPolicy(max_messages=1),
        min_total_tokens=100,
    )
    messages = [
        Message.text("system", "You are careful."),
        Message.text("user", "old"),
        Message.text("assistant", "old answer"),
        Message.text("user", "current"),
    ]

    async def build_context() -> list[Message]:
        return await policy.build(
            ContextRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=messages,
                step=2,
                context_usage=ContextUsageState(last_total_tokens=100),
            )
        )

    context = asyncio.run(build_context())

    assert [message.role for message in context] == ["system", "user"]
    assert [message.content[0].text for message in context] == ["You are careful.", "current"]


def test_usage_triggered_context_policy_requires_a_threshold():
    with pytest.raises(ValueError, match="At least one usage threshold"):
        UsageTriggeredContextPolicy(
            triggered_policy=MessageWindowContextPolicy(max_messages=1),
        )


def test_usage_triggered_context_policy_rejects_runtime_managed_estimated_base_policy():
    with pytest.raises(ValueError, match="side-effect-free base_policy"):
        UsageTriggeredContextPolicy(
            base_policy=CheckpointCompactionContextPolicy(),
            triggered_policy=MessageWindowContextPolicy(max_messages=1),
            trigger_estimated_context_tokens=100,
        )


def test_usage_triggered_context_policy_requires_bool_sticky():
    with pytest.raises(TypeError, match="sticky must be a bool"):
        UsageTriggeredContextPolicy(
            triggered_policy=MessageWindowContextPolicy(max_messages=1),
            min_input_tokens=1,
            sticky=1,
        )
