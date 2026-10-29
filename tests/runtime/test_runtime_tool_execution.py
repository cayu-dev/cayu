"""Tool execution contracts."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
from pydantic import ValidationError
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_session_support import (
    _private_events_for_public_events,
    collect_events,
    tool_round_identity_payload,
)
from tests.runtime.runtime_tool_support import EchoTool, SideEffectTool, UpperTool

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.budgets.run_limits import RunLimits
from cayu.configuration import CayuConfig, ToolExecutionConfig
from cayu.events import EventType
from cayu.messages import Message
from cayu.observability.hooks import RuntimeHook, ToolCallHookContext
from cayu.providers.base import ModelRequest, ModelStreamEvent
from cayu.runtime.execution_profiles import EXECUTION_PROFILE_FINGERPRINT_FIELD
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionStatus
from cayu.sessions.requests import RunRequest
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools._argument_publication import arguments_identity_fields
from cayu.tools.base import (
    _POLICY_DENIAL_TEXT_MAX_BYTES,
    _POLICY_DENIAL_TRUNCATION_MARKER,
    Tool,
    ToolContext,
    ToolEffect,
    ToolResult,
    ToolSpec,
    _bound_policy_denial_text,
)
from cayu.tools.policy import ToolPolicy, ToolPolicyDecision, ToolPolicyRequest, ToolPolicyResult


class MutatingProvider(FakeProvider):
    def __init__(self) -> None:
        super().__init__(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_1",
                        name="echo",
                        arguments={"text": "from tool"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )

    mutation_blocked = False

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        async for event in super().stream(request):
            if len(self.requests) == 1:
                # Messages are frozen; the runtime history cannot be corrupted
                # even by a misbehaving provider.
                try:
                    request.messages[0].content[0].text = "mutated by provider"
                except ValidationError:
                    self.mutation_blocked = True
            yield event


class MetadataMutatingProvider(FakeProvider):
    def __init__(self) -> None:
        super().__init__(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_1",
                        name="echo",
                        arguments={"text": "from tool"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        async for event in super().stream(request):
            if len(self.requests) == 1:
                request.options["agent_metadata"]["nested"]["value"] = "mutated"
            yield event


class FailingTool(Tool):
    spec = ToolSpec(
        name="fail",
        description="Fail intentionally.",
        effect=ToolEffect.NONE,
        input_schema={"type": "object", "properties": {}},
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        raise RuntimeError("intentional tool failure")


class BlankFailingTool(Tool):
    spec = ToolSpec(
        name="blank_fail",
        description="Fail intentionally without a message.",
        effect=ToolEffect.NONE,
        input_schema={"type": "object", "properties": {}},
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        raise RuntimeError()


class BlankErrorResultTool(Tool):
    spec = ToolSpec(
        name="blank_error_result",
        description="Return an error result without a message.",
        input_schema={"type": "object", "properties": {}},
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(is_error=True)


class InvalidResultTool(Tool):
    spec = ToolSpec(
        name="invalid_result",
        description="Return the wrong result type.",
        effect=ToolEffect.NONE,
        input_schema={"type": "object", "properties": {}},
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return {"content": "not a ToolResult"}  # type: ignore[return-value]


class InvalidConstructedResultTool(Tool):
    spec = ToolSpec(
        name="invalid_constructed_result",
        description="Return a ToolResult that bypassed validation.",
        input_schema={"type": "object", "properties": {}},
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        class BadStructured(dict):
            def items(self):
                raise RuntimeError("tool result traversal should not run")

        return ToolResult.model_construct(
            content="ok",
            structured=BadStructured({"bad": "value"}),
            artifacts=[],
            is_error=False,
        )


class ArgumentMutatingTool(Tool):
    spec = ToolSpec(
        name="mutate_args",
        description="Mutate call arguments.",
        input_schema={"type": "object", "properties": {}},
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        args["nested"]["value"] = "mutated"
        return ToolResult(content="mutated")


class ResultHoldingTool(Tool):
    spec = ToolSpec(
        name="hold_result",
        description="Keep returned result for later mutation.",
        input_schema={"type": "object", "properties": {}},
    )

    def __init__(self) -> None:
        super().__init__()
        self.result: ToolResult | None = None

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.result = ToolResult(
            content="held",
            structured={"nested": {"value": "original"}},
            artifacts=[{"nested": {"value": "original"}}],
        )
        return self.result


def test_cayu_app_executes_tool_call_and_records_result():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="echo",
                    arguments={"text": "from tool"},
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
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_tool",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    private_events = asyncio.run(_private_events_for_public_events(store, "sess_tool", events))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert len(provider.requests) == 2
    assert provider.requests[0].tools == [
        {
            "name": "echo",
            "description": "Echo text.",
            "input_schema": EchoTool.spec.input_schema,
        }
    ]
    assert private_events[3].tool_name == "echo"
    assert private_events[3].payload == {
        "tool_call_id": "call_1",
        "idempotency_key": private_events[3].payload["idempotency_key"],
        "effect": "external",
        "arguments_state": "quarantined",
        **arguments_identity_fields(
            {"text": "from tool"}, session_id="sess_tool", publish_arguments=True
        ),
        "argument_presence": {"keys": ["text"], "key_count": 1, "unlisted_key_count": 0},
        EXECUTION_PROFILE_FINGERPRINT_FIELD: private_events[3].payload[
            EXECUTION_PROFILE_FINGERPRINT_FIELD
        ],
        **tool_round_identity_payload(private_events[3]),
    }
    assert private_events[4].payload["tool_round_id"] == private_events[3].payload["tool_round_id"]
    assert events[4].payload["result"]["content"] == "from tool"
    assert events[4].payload["result"]["structured"] == {
        "agent": "assistant",
        "echoed": "from tool",
    }
    assert provider.requests[1].messages[-2].role == "assistant"
    tool_call_part = provider.requests[1].messages[-2].content[0]
    assert tool_call_part.type == "tool_call"
    assert tool_call_part.tool_call_id == "call_1"
    assert tool_call_part.tool_name == "echo"
    assert tool_call_part.arguments == {"text": "from tool"}

    assert provider.requests[1].messages[-1].role == "tool"
    tool_result_part = provider.requests[1].messages[-1].content[0]
    assert tool_result_part.type == "tool_result"
    assert tool_result_part.tool_call_id == "call_1"
    assert tool_result_part.tool_name == "echo"
    assert tool_result_part.content == "from tool"
    assert tool_result_part.structured == {
        "agent": "assistant",
        "echoed": "from tool",
    }
    assert tool_result_part.artifacts == []
    assert tool_result_part.is_error is False

    transcript = asyncio.run(store.load_transcript("sess_tool"))
    assert [message.role for message in transcript] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert transcript[0].content[0].text == "use the tool"
    assert transcript[1].content[0].type == "tool_call"
    assert transcript[2].content[0].type == "tool_result"
    assert transcript[3].content[0].text == "done"


def test_cayu_app_returns_tool_failure_to_model():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="fail",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("recovered"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[FailingTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_tool_failure",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    session = asyncio.run(store.load("sess_tool_failure"))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_FAILED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert len(provider.requests) == 2
    tool_call_part = provider.requests[1].messages[-2].content[0]
    assert tool_call_part.type == "tool_call"
    assert tool_call_part.tool_call_id == "call_1"
    assert tool_call_part.tool_name == "fail"

    tool_result_part = provider.requests[1].messages[-1].content[0]
    assert tool_result_part.type == "tool_result"
    assert tool_result_part.is_error is True
    assert tool_result_part.content == "intentional tool failure"
    assert session is not None
    assert session.status == SessionStatus.COMPLETED


def test_cayu_app_returns_nonblank_tool_failure_when_exception_message_is_blank():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="blank_fail",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("recovered"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[BlankFailingTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_blank_tool_failure",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    assert events[4].type == EventType.TOOL_CALL_FAILED
    assert events[4].payload["result"]["content"] == ("RuntimeError: tool execution failed")

    tool_result_part = provider.requests[1].messages[-1].content[0]
    assert tool_result_part.type == "tool_result"
    assert tool_result_part.is_error is True
    assert tool_result_part.content == "RuntimeError: tool execution failed"


def test_cayu_app_returns_nonblank_tool_failure_when_tool_error_content_is_blank():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="blank_error_result",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("recovered"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[BlankErrorResultTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_blank_tool_error_result",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    assert events[4].type == EventType.TOOL_CALL_FAILED
    assert events[4].payload["result"]["content"] == ("Tool returned an error without details.")

    tool_result_part = provider.requests[1].messages[-1].content[0]
    assert tool_result_part.type == "tool_result"
    assert tool_result_part.is_error is True
    assert tool_result_part.content == "Tool returned an error without details."


def test_cayu_app_returns_clear_tool_failure_for_invalid_result_type():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="invalid_result",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("recovered"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[InvalidResultTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_invalid_tool_result",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    assert events[4].type == EventType.TOOL_CALL_FAILED
    assert events[4].payload["result"]["content"] == (
        "Tool returned invalid result type: dict. Expected ToolResult."
    )

    tool_result_part = provider.requests[1].messages[-1].content[0]
    assert tool_result_part.type == "tool_result"
    assert tool_result_part.is_error is True
    assert tool_result_part.content == (
        "Tool returned invalid result type: dict. Expected ToolResult."
    )


def test_cayu_app_fences_external_tool_with_invalid_constructed_result():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="invalid_constructed_result",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("recovered"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[InvalidConstructedResultTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_invalid_constructed_tool_result",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    session = asyncio.run(store.load("sess_invalid_constructed_tool_result"))

    assert events[-1].type is EventType.SESSION_INTERRUPTED
    persisted = asyncio.run(store.load_events("sess_invalid_constructed_tool_result"))
    assert not any(
        event.type in {EventType.TOOL_CALL_FAILED, EventType.TOOL_CALL_COMPLETED}
        for event in persisted
    )
    unknown = [event for event in persisted if event.type is EventType.TOOL_EFFECT_OUTCOME_UNKNOWN]
    assert len(unknown) == 1
    assert unknown[0].payload["tool_call_id"] == "call_1"
    assert len(provider.requests) == 1
    checkpoint = asyncio.run(store.load_checkpoint("sess_invalid_constructed_tool_result"))
    assert checkpoint is not None and "pending_tool_round" in checkpoint
    transcript = asyncio.run(store.load_transcript("sess_invalid_constructed_tool_result"))
    assert not any(message.role == "tool" for message in transcript)
    assert session is not None
    assert session.status is SessionStatus.INTERRUPTED


def test_cayu_app_keeps_text_and_tool_calls_in_one_assistant_turn():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("I will check. "),
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="echo",
                    arguments={"text": "from tool"},
                ),
                ModelStreamEvent.text_delta(" After that."),
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
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_mixed_assistant_turn",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]

    assistant_message = provider.requests[1].messages[-2]
    assert assistant_message.role == "assistant"
    assert [part.type for part in assistant_message.content] == [
        "text",
        "tool_call",
        "text",
    ]
    assert assistant_message.content[0].text == "I will check. "
    assert assistant_message.content[1].tool_call_id == "call_1"
    assert assistant_message.content[1].tool_name == "echo"
    assert assistant_message.content[2].text == " After that."

    tool_result_message = provider.requests[1].messages[-1]
    assert tool_result_message.role == "tool"
    assert tool_result_message.content[0].tool_call_id == "call_1"


def test_cayu_app_preserves_whitespace_text_deltas():
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("Hello"),
                ModelStreamEvent.text_delta(" "),
                ModelStreamEvent.text_delta("world"),
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="echo",
                    arguments={"text": "from tool"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_text_delta_whitespace",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assistant_message = provider.requests[1].messages[-2]
    assert assistant_message.role == "assistant"
    assert assistant_message.content[0].text == "Hello world"


def test_cayu_app_protects_tool_call_arguments_from_tool_mutation():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="mutate_args",
                    arguments={"nested": {"value": "original"}},
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
        tools=[ArgumentMutatingTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_mutating_tool_args",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert events[3].type == EventType.TOOL_CALL_STARTED
    assert events[3].payload["arguments_state"] == "quarantined"
    assert "arguments" not in events[3].payload
    completed = next(event for event in events if event.type == EventType.TOOL_CALL_COMPLETED)
    assert completed.payload["arguments_state"] == "finalized"
    assert completed.payload["arguments"] == {"nested": {"value": "original"}}

    assistant_message = provider.requests[1].messages[-2]
    tool_call_part = assistant_message.content[0]
    assert tool_call_part.type == "tool_call"
    assert tool_call_part.arguments == {"nested": {"value": "original"}}


@pytest.fixture(params=("memory", "sqlite"), ids=("memory", "sqlite"))
def tool_result_session_store(request, tmp_path):
    store = (
        InMemorySessionStore()
        if request.param == "memory"
        else SQLiteSessionStore(tmp_path / "tool-result-read-only.sqlite")
    )
    yield store
    close = getattr(store, "close", None)
    if close is not None:
        asyncio.run(close())


def test_cayu_app_protects_tool_result_messages_from_result_mutation(
    tool_result_session_store,
):
    store = tool_result_session_store
    tool = ResultHoldingTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="hold_result",
                    arguments={},
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
        tools=[tool],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_mutating_tool_result",
                messages=[Message.text("user", "use the tool")],
            ),
        )
    )
    assert events[-1].type == EventType.SESSION_COMPLETED
    assert tool.result is not None

    with pytest.raises(TypeError, match="Frozen JSON values cannot be mutated"):
        tool.result.structured["nested"]["value"] = "mutated"  # type: ignore[index]
    with pytest.raises(TypeError, match="Frozen JSON values cannot be mutated"):
        tool.result.artifacts[0]["nested"]["value"] = "mutated"  # type: ignore[index]

    tool_result_part = provider.requests[1].messages[-1].content[0]
    assert tool_result_part.type == "tool_result"
    assert tool_result_part.structured == {"nested": {"value": "original"}}
    assert tool_result_part.artifacts == [{"nested": {"value": "original"}}]

    completed = next(event for event in events if event.type == EventType.TOOL_CALL_COMPLETED)
    assert type(completed.payload["result"]["structured"]) is dict
    assert type(completed.payload["result"]["artifacts"]) is list
    assert completed.payload["result"]["structured"] == {"nested": {"value": "original"}}
    assert completed.payload["result"]["artifacts"] == [{"nested": {"value": "original"}}]

    transcript = asyncio.run(store.load_transcript("sess_mutating_tool_result"))
    stored_result = transcript[2].content[0]
    assert stored_result.structured == {"nested": {"value": "original"}}
    assert stored_result.artifacts == [{"nested": {"value": "original"}}]


def test_cayu_app_protects_runtime_history_from_provider_mutation():
    provider = MutatingProvider()
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[EchoTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_provider_mutation",
                messages=[Message.text("user", "original")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert provider.mutation_blocked
    assert provider.requests[0].messages[0].content[0].text == "original"
    assert provider.requests[1].messages[0].content[0].text == "original"


def test_cayu_app_protects_agent_metadata_from_provider_mutation():
    provider = MetadataMutatingProvider()
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(
            name="assistant",
            model="fake-model",
            metadata={"nested": {"value": "original"}},
        ),
        tools=[EchoTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_metadata_mutation",
                messages=[Message.text("user", "original")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert provider.requests[0].options["agent_metadata"] == {"nested": {"value": "mutated"}}
    assert provider.requests[1].options["agent_metadata"] == {"nested": {"value": "original"}}


def test_cayu_app_passes_agent_provider_options_to_model_request():
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(
            name="assistant",
            model="fake-model",
            provider_options={
                "openai": {
                    "prompt_cache_key": "tenant-a-agent",
                    "prompt_cache_retention": "24h",
                }
            },
        )
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_provider_options",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert provider.requests[0].options["openai"] == {
        "prompt_cache_key": "tenant-a-agent",
        "prompt_cache_retention": "24h",
    }
    assert provider.requests[0].options["agent_metadata"] == {}
    assert provider.requests[0].options["step"] == 1


def test_cayu_app_groups_multiple_tool_calls_and_results_in_history():
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="echo",
                    arguments={"text": "one"},
                ),
                ModelStreamEvent.tool_call(
                    id="call_2",
                    name="upper",
                    arguments={"text": "two"},
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
        tools=[EchoTool(), UpperTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_multi_tool",
                messages=[Message.text("user", "use both tools")],
            ),
        )
    )

    tool_types = {EventType.TOOL_CALL_STARTED, EventType.TOOL_CALL_COMPLETED}
    assert [event.type for event in events if event.type not in tool_types] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    # Independent calls may overlap, and dynamic rounds defer terminal
    # publication. Each call still starts once and completes before model reuse.
    assert all(event.type in tool_types for event in events[3:7])
    for tool_name in ("echo", "upper"):
        assert [event.type for event in events[3:7] if event.tool_name == tool_name] == [
            EventType.TOOL_CALL_STARTED,
            EventType.TOOL_CALL_COMPLETED,
        ]

    tool_call_message = provider.requests[1].messages[-2]
    assert tool_call_message.role == "assistant"
    assert [part.type for part in tool_call_message.content] == [
        "tool_call",
        "tool_call",
    ]
    assert [part.tool_call_id for part in tool_call_message.content] == [
        "call_1",
        "call_2",
    ]

    tool_result_message = provider.requests[1].messages[-1]
    assert tool_result_message.role == "tool"
    assert [part.type for part in tool_result_message.content] == [
        "tool_result",
        "tool_result",
    ]
    assert [part.content for part in tool_result_message.content] == ["one", "TWO"]


def test_tool_call_times_out_and_session_continues():
    class SlowTool(Tool):
        spec = ToolSpec(
            name="slow_tool",
            description="Sleep past the configured tool timeout.",
            effect=ToolEffect.NONE,
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            await asyncio.sleep(30)
            return ToolResult(content="unexpected")

    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(id="call_slow", name="slow_tool", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("recovered"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(
        session_store=store,
        config=CayuConfig(tool_execution=ToolExecutionConfig(tool_timeout_seconds=0.05)),
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[SlowTool()],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_tool_timeout",
                messages=[Message.text("user", "go")],
            ),
        )
    )

    failed_events = [event for event in events if event.type == EventType.TOOL_CALL_FAILED]
    assert len(failed_events) == 1
    assert failed_events[0].payload["result"]["content"] == (
        "Tool call timed out after 0.05 seconds."
    )
    assert failed_events[0].payload["result"]["is_error"] is True
    assert events[-1].type == EventType.SESSION_COMPLETED

    transcript = asyncio.run(store.load_transcript("sess_tool_timeout"))
    tool_message = next(message for message in transcript if message.role == "tool")
    assert tool_message.content[0].content == "Tool call timed out after 0.05 seconds."
    assert tool_message.content[0].is_error is True


def test_run_tool_does_not_mislabel_tool_raised_timeout_error():
    from cayu.runtime._tool_execution import run_tool
    from cayu.vaults.redaction import SecretRedactor

    class TimeoutRaisingTool(Tool):
        spec = ToolSpec(
            name="timeout_raiser",
            description="Raise TimeoutError from inside the tool.",
            input_schema={"type": "object", "properties": {}},
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            raise TimeoutError("upstream service timed out")

    outcome = asyncio.run(
        run_tool(
            tool=TimeoutRaisingTool(),
            effect=ToolEffect.EXTERNAL,
            ctx=ToolContext(session_id="sess_tool_raised_timeout"),
            arguments={},
            timeout_seconds=5,
            redactor=SecretRedactor,
        )
    )

    result = outcome.result
    assert result.is_error is True
    assert "upstream service timed out" in result.content
    assert "timed out after" not in result.content


def test_tool_error_discards_recorded_policy_denial_before_terminal_classification():
    class RecordDenialThenRaiseTool(Tool):
        spec = ToolSpec(
            name="record_denial_then_raise",
            description="Record a nested denial and then fail.",
            effect=ToolEffect.NONE,
            input_schema={"type": "object", "properties": {}},
        )

        def __init__(self) -> None:
            self.calls = 0

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del args
            self.calls += 1
            denial_result = ToolResult(
                content="Nested command denied by policy.",
                structured={"error": "command_denied", "decision": "deny"},
                is_error=True,
            )
            ctx._record_policy_denial(
                source=self,
                denied_by="command_policy",
                decision="deny",
                reason="Nested command denied by policy.",
                result=denial_result,
            )
            raise RuntimeError("tool failed after recording denial")

    tool = RecordDenialThenRaiseTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_record_then_error",
                    name=tool.spec.name,
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [ModelStreamEvent.completed({"finish_reason": "stop"})],
        ]
    )
    app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_record_denial_then_error",
                messages=[Message.text("user", "run")],
            ),
        )
    )

    assert tool.calls == 1
    assert all(event.type != EventType.TOOL_CALL_BLOCKED for event in events)
    failed = [event for event in events if event.type == EventType.TOOL_CALL_FAILED]
    assert len(failed) == 1
    assert failed[0].payload["result"]["content"] == "tool failed after recording denial"
    assert events[-1].type == EventType.SESSION_COMPLETED


def test_parallel_tool_round_executes_calls_concurrently_by_default():
    class RendezvousTool(Tool):
        spec = ToolSpec(
            name="rendezvous",
            description="Wait until every call in the round has started.",
            input_schema={
                "type": "object",
                "properties": {"slot": {"type": "string"}},
                "required": ["slot"],
            },
        )

        def __init__(self, expected_arrivals: int) -> None:
            self.expected_arrivals = expected_arrivals
            self.arrivals = 0
            self.all_arrived = asyncio.Event()

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            self.arrivals += 1
            if self.arrivals >= self.expected_arrivals:
                self.all_arrived.set()
            # Deadlocks under sequential execution: each call waits for the
            # other call in the same round to start.
            await asyncio.wait_for(self.all_arrived.wait(), timeout=2)
            return ToolResult(content=args["slot"])

    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_a",
                    name="rendezvous",
                    arguments={"slot": "a"},
                ),
                ModelStreamEvent.tool_call(
                    id="call_b",
                    name="rendezvous",
                    arguments={"slot": "b"},
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
        tools=[RendezvousTool(expected_arrivals=2)],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_parallel_round",
                messages=[Message.text("user", "fan out")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    private_events = asyncio.run(
        _private_events_for_public_events(store, "sess_parallel_round", events)
    )
    tool_events = [
        (event.type, event.payload["tool_call_id"])
        for event in private_events
        if event.type in {EventType.TOOL_CALL_STARTED, EventType.TOOL_CALL_COMPLETED}
    ]
    # The unknown secret scope stages terminals until all calls have sealed.
    # Starts and finalized terminals each preserve the model's tool-call order.
    assert tool_events == [
        (EventType.TOOL_CALL_STARTED, "call_a"),
        (EventType.TOOL_CALL_STARTED, "call_b"),
        (EventType.TOOL_CALL_COMPLETED, "call_a"),
        (EventType.TOOL_CALL_COMPLETED, "call_b"),
    ]

    transcript = asyncio.run(store.load_transcript("sess_parallel_round"))
    tool_message = next(message for message in transcript if message.role == "tool")
    assert [part.tool_call_id for part in tool_message.content] == ["call_a", "call_b"]
    assert [part.content for part in tool_message.content] == ["a", "b"]


def test_parallel_tool_policy_denials_emit_one_canonical_block_per_call():
    reason = "parallel policy denial 🙂 " * 500

    class OversizedDenyPolicy(ToolPolicy):
        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            del request
            return ToolPolicyResult(decision=ToolPolicyDecision.DENY, reason=reason)

    class ParallelBlockProbe(RuntimeHook):
        def __init__(self) -> None:
            self.arrivals = 0
            self.current = 0
            self.max_concurrent = 0
            self.all_arrived = asyncio.Event()

        async def after_tool_call(self, context: ToolCallHookContext) -> None:
            if context.tool_event.type != EventType.TOOL_CALL_BLOCKED:
                return
            self.arrivals += 1
            self.current += 1
            self.max_concurrent = max(self.max_concurrent, self.current)
            if self.arrivals == 2:
                self.all_arrived.set()
            # This times out if the denial paths are executed sequentially.
            await asyncio.wait_for(self.all_arrived.wait(), timeout=2)
            await asyncio.sleep(0)
            self.current -= 1

    store = InMemorySessionStore()
    tool = SideEffectTool()
    probe = ParallelBlockProbe()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(id="call_a", name="side_effect", arguments={}),
                ModelStreamEvent.tool_call(id="call_b", name="side_effect", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [ModelStreamEvent.completed({"finish_reason": "stop"})],
        ]
    )
    app = CayuApp(
        session_store=store,
        enable_logging=False,
        config=CayuConfig(tool_execution=ToolExecutionConfig(max_parallel_tool_calls=2)),
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=OversizedDenyPolicy(),
        runtime_hooks=[probe],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_parallel_policy_denials",
                messages=[Message.text("user", "try both calls")],
            ),
        )
    )

    assert probe.max_concurrent == 2
    assert tool.calls == []
    blocked = [event for event in events if event.type == EventType.TOOL_CALL_BLOCKED]
    private_blocked = asyncio.run(
        _private_events_for_public_events(store, "sess_parallel_policy_denials", blocked)
    )
    assert [event.payload["tool_call_id"] for event in private_blocked] == [
        "call_a",
        "call_b",
    ]
    assert all(event.payload["denied_by"] == "tool_policy" for event in blocked)
    assert all(event.payload["decision"] == "deny" for event in blocked)
    bounded_reason = _bound_policy_denial_text(reason)
    assert all(event.payload["reason"] == bounded_reason for event in blocked)
    assert all(event.payload["result"]["content"] == bounded_reason for event in blocked)
    assert all(
        event.payload["result"]["structured"]["reason"] == bounded_reason for event in blocked
    )
    assert len(bounded_reason.encode("utf-8")) <= _POLICY_DENIAL_TEXT_MAX_BYTES
    assert bounded_reason.endswith(_POLICY_DENIAL_TRUNCATION_MARKER)
    assert len({event.payload["idempotency_key"] for event in private_blocked}) == 2
    assert all(event.type != EventType.TOOL_CALL_FAILED for event in events)

    transcript = asyncio.run(store.load_transcript("sess_parallel_policy_denials"))
    tool_message = next(message for message in transcript if message.role == "tool")
    assert [part.tool_call_id for part in tool_message.content] == ["call_a", "call_b"]
    assert all(part.content == bounded_reason for part in tool_message.content)


def test_parallel_tool_round_composes_with_tool_call_limit_and_usage_counting():
    # Composition pin for issue #101's real-world trigger: a genuinely
    # concurrent tool round under RunLimits. The whole round is gated against
    # limits BEFORE execution (mid-round re-evaluation is skipped), and the
    # usage summary must count every tool.call.started the parallel round
    # appended — the undercount the watermark race used to cause.
    class RendezvousTool(Tool):
        spec = ToolSpec(
            name="rendezvous",
            description="Wait until every call in the round has started.",
            input_schema={
                "type": "object",
                "properties": {"slot": {"type": "string"}},
                "required": ["slot"],
            },
        )

        def __init__(self, expected_arrivals: int) -> None:
            self.expected_arrivals = expected_arrivals
            self.arrivals = 0
            self.all_arrived = asyncio.Event()

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            self.arrivals += 1
            if self.arrivals >= self.expected_arrivals:
                self.all_arrived.set()
            # Deadlocks under sequential execution: each call waits for the
            # other call in the same round to start.
            await asyncio.wait_for(self.all_arrived.wait(), timeout=2)
            return ToolResult(content=args["slot"])

    store = InMemorySessionStore()
    tool = RendezvousTool(expected_arrivals=2)
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(id="call_a", name="rendezvous", arguments={"slot": "a"}),
                ModelStreamEvent.tool_call(id="call_b", name="rendezvous", arguments={"slot": "b"}),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "tool_calls",
                        "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
                    }
                ),
            ],
            [
                ModelStreamEvent.tool_call(id="call_c", name="rendezvous", arguments={"slot": "c"}),
                ModelStreamEvent.tool_call(id="call_d", name="rendezvous", arguments={"slot": "d"}),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "tool_calls",
                        "usage": {"input_tokens": 20, "output_tokens": 4, "total_tokens": 24},
                    }
                ),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_parallel_limit",
                messages=[Message.text("user", "fan out twice")],
                limits=RunLimits(max_tool_calls=3),
            ),
        )
    )

    # Round 1 (2 parallel calls) fits the limit and genuinely ran concurrently.
    assert tool.arrivals == 2
    private_events = asyncio.run(
        _private_events_for_public_events(store, "sess_parallel_limit", events)
    )
    started_ids = [
        event.payload["tool_call_id"]
        for event in private_events
        if event.type == EventType.TOOL_CALL_STARTED
    ]
    assert started_ids == ["call_a", "call_b"]

    # Round 2 (2 more calls) would exceed max_tool_calls=3; the whole round is
    # gated before execution, so neither call starts.
    limit_event = next(event for event in events if event.type == EventType.SESSION_LIMIT_REACHED)
    assert limit_event.payload["limit"] == "tool_calls"
    assert limit_event.payload["actual"] == 4
    failed_ids = {
        event.payload["tool_call_id"]
        for event in private_events
        if event.type == EventType.TOOL_CALL_FAILED
    }
    assert failed_ids == {"call_c", "call_d"}
    assert events[-1].type == EventType.SESSION_INTERRUPTED

    # The usage summary counts every started call of the parallel round and
    # both model steps — the undercount the watermark race used to cause.
    summary = asyncio.run(app.get_session_usage("sess_parallel_limit"))
    assert summary.tool_calls == 2
    assert summary.model_steps == 2
    assert summary.usage.total_tokens == 36


def test_parallel_tool_round_concurrency_is_capped_by_semaphore():
    class ConcurrencyProbeTool(Tool):
        spec = ToolSpec(
            name="probe",
            description="Record how many calls run concurrently.",
            input_schema={
                "type": "object",
                "properties": {"step": {"type": "integer"}},
                "required": ["step"],
            },
        )

        def __init__(self) -> None:
            self.current = 0
            self.max_concurrent = 0

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            self.current += 1
            self.max_concurrent = max(self.max_concurrent, self.current)
            await asyncio.sleep(0.02)
            self.current -= 1
            return ToolResult(content=str(args["step"]))

    probe = ConcurrencyProbeTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(id="call_1", name="probe", arguments={"step": 1}),
                ModelStreamEvent.tool_call(id="call_2", name="probe", arguments={"step": 2}),
                ModelStreamEvent.tool_call(id="call_3", name="probe", arguments={"step": 3}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(config=CayuConfig(tool_execution=ToolExecutionConfig(max_parallel_tool_calls=2)))
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[probe],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_parallel_capped",
                messages=[Message.text("user", "fan out")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert probe.max_concurrent == 2
    private_events = asyncio.run(
        _private_events_for_public_events(
            app.session_store,
            "sess_parallel_capped",
            events,
        )
    )
    completed_ids = [
        event.payload["tool_call_id"]
        for event in private_events
        if event.type == EventType.TOOL_CALL_COMPLETED
    ]
    assert completed_ids == ["call_1", "call_2", "call_3"]


def test_parallel_tool_call_timeouts_do_not_serialize_the_round():
    class SleepyTool(Tool):
        spec = ToolSpec(
            name="sleepy",
            description="Sleep forever; the per-call timeout stops it.",
            input_schema={"type": "object", "properties": {}},
        )

        def __init__(self) -> None:
            self.active = 0
            self.max_concurrent = 0

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            self.active += 1
            self.max_concurrent = max(self.max_concurrent, self.active)
            try:
                await asyncio.sleep(30)
                return ToolResult(content="unexpected")
            finally:
                self.active -= 1

    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(id="call_1", name="sleepy", arguments={}),
                ModelStreamEvent.tool_call(id="call_2", name="sleepy", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(config=CayuConfig(tool_execution=ToolExecutionConfig(tool_timeout_seconds=0.5)))
    app.register_provider(provider, default=True)
    sleepy = SleepyTool()
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[sleepy],
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_parallel_timeouts",
                messages=[Message.text("user", "fan out")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_INTERRUPTED
    persisted = asyncio.run(app.session_store.load_events("sess_parallel_timeouts"))
    assert not any(
        event.type in {EventType.TOOL_CALL_FAILED, EventType.TOOL_CALL_COMPLETED}
        for event in persisted
    )
    unknown = [event for event in persisted if event.type == EventType.TOOL_EFFECT_OUTCOME_UNKNOWN]
    assert len(unknown) == 2
    assert {event.payload["tool_call_id"] for event in unknown} == {"call_1", "call_2"}
    checkpoint = asyncio.run(app.session_store.load_checkpoint("sess_parallel_timeouts"))
    assert checkpoint is not None and "pending_tool_round" in checkpoint
    assert len(provider.requests) == 1
    assert sleepy.max_concurrent == 2
    assert sleepy.active == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tool_timeout_seconds": 0},
        {"tool_timeout_seconds": "5"},
        {"max_parallel_tool_calls": 0},
        {"max_parallel_tool_calls": 2.5},
    ],
)
def test_cayu_config_validates_tool_execution_settings(kwargs):
    with pytest.raises(ValueError):
        ToolExecutionConfig(**kwargs)
