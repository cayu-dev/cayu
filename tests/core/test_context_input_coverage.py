from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from cayu.core import (
    AgentSpec,
    EventType,
    ExecutionProfileBehaviorIdentity,
    Message,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
)
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import (
    CayuApp,
    ContextPolicy,
    ContextRequest,
    ContextUsageState,
    EventQuery,
    InMemorySessionStore,
    ObservedDeltaContextEstimator,
    ResumeRequest,
    RunRequest,
    StructuredOutputSpec,
    context_input_coverage,
)
from cayu.runtime._model_step_executor import _context_usage_state_for_session
from cayu.runtime.context import _prompt_cache_extension_messages
from cayu.runtime.structured_output import STRUCTURED_OUTPUT_TOOL_NAME
from cayu.storage import SQLiteSessionStore


def test_large_assistant_completion_is_not_covered_by_previous_input() -> None:
    messages = [Message.text("user", "hi"), Message.text("assistant", "x" * 8000)]
    usage = ContextUsageState(
        last_input_tokens=10,
        last_transcript_cursor=2,
        input_coverage=context_input_coverage(messages[:1], transcript_cursor=1),
    )
    estimate = ObservedDeltaContextEstimator(chars_per_token=4).estimate(
        usage=usage, messages=messages
    )
    assert estimate is not None
    assert estimate.estimated_message_count == 1
    assert estimate.estimated_delta_input_tokens >= 2000
    assert estimate.observed_context_input_tokens == 10
    assert estimate.anchor_transcript_cursor == 1


@pytest.mark.parametrize("change", ["compaction", "same_length", "retention", "legacy", "no_usage"])
def test_changed_or_unknown_input_projection_uses_full_estimate(change: str) -> None:
    original = [Message.text("user", "old input"), Message.text("assistant", "old answer")]
    messages = [*original, Message.text("user", "next")]
    coverage = context_input_coverage(original, transcript_cursor=20)
    if change == "compaction":
        messages = [Message.text("system", "summary"), Message.text("user", "next")]
    elif change == "same_length":
        messages[0] = Message.text("user", "replacement")
    elif change == "retention":
        messages = messages[-1:]
    usage = ContextUsageState(
        last_input_tokens=None if change == "no_usage" else 100,
        last_transcript_cursor=21,
        input_coverage=None if change == "legacy" else coverage,
    )
    estimator = ObservedDeltaContextEstimator()
    assert estimator.estimate(usage=usage, messages=messages) is None
    estimate = estimator.estimate_anchored_request(
        usage=usage, messages=messages, reserved_output_tokens=13
    )
    assert estimate.method == "local_full_request_estimate"
    assert estimate.estimated_message_count == len(messages)
    assert estimate.estimated_context_window_tokens == estimate.estimated_context_input_tokens + 13


class CapturePolicy(ContextPolicy):
    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests.coverage.capture", behavior_version="1", implementation_version="1"
        )

    def __init__(self) -> None:
        self.requests: list[ContextRequest] = []

    async def build(self, request: ContextRequest) -> list[Message]:
        self.requests.append(request)
        return request.messages


class ScriptedProvider(ModelProvider):
    name = "coverage"
    supports_native_structured_output = True

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests.coverage.provider", behavior_version="1", implementation_version="1"
        )

    def __init__(self, batches: list[list[ModelStreamEvent]]) -> None:
        self.batches = batches
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request.model_copy(deep=True))
        for event in self.batches[len(self.requests) - 1]:
            yield event


def completion(input_tokens: int = 10) -> ModelStreamEvent:
    return ModelStreamEvent.completed({"usage": {"input_tokens": input_tokens, "output_tokens": 1}})


@pytest.mark.parametrize("backend", ["memory", "sqlite_reopen"])
@pytest.mark.parametrize("answer", ["x" * 8000, ""], ids=["large", "empty"])
def test_next_request_and_reconstructed_usage_count_completion_once(
    tmp_path, backend, answer
) -> None:
    async def run() -> None:
        path = tmp_path / "sessions.sqlite"
        store = InMemorySessionStore() if backend == "memory" else SQLiteSessionStore(path)
        provider = ScriptedProvider(
            [
                [*([ModelStreamEvent.text_delta(answer)] if answer else []), completion()],
                [ModelStreamEvent.text_delta("second"), completion(20)],
                [ModelStreamEvent.text_delta("third"), completion(30)],
            ]
        )
        policy = CapturePolicy()

        def app_for(selected_store) -> CayuApp:
            app = CayuApp(session_store=selected_store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="assistant", model="fake"), context_policy=policy)
            return app

        app = app_for(store)
        first = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="coverage",
                    messages=[Message.text("user", "hi")],
                )
            )
        ]
        completed = next(event for event in first if event.type == EventType.MODEL_COMPLETED)
        assert completed.payload["input_coverage"] == context_input_coverage(
            provider.requests[0].messages, transcript_cursor=1
        ).model_dump(mode="json")
        before = await _context_usage_state_for_session(session_store=store, session_id="coverage")
        assert before.input_coverage == context_input_coverage(
            provider.requests[0].messages, transcript_cursor=1
        )
        if backend == "sqlite_reopen":
            await store.close()
            store = SQLiteSessionStore(path)
            app = app_for(store)
        restored = await _context_usage_state_for_session(
            session_store=store, session_id="coverage"
        )
        assert restored == before
        second = [
            event
            async for event in app.resume(
                ResumeRequest(session_id="coverage", messages=[Message.text("user", "next")])
            )
        ]
        assert any(event.type == EventType.MODEL_COMPLETED for event in second)
        pressure = policy.requests[1].context_usage.input_pressure
        assert pressure is not None
        expected = ObservedDeltaContextEstimator().estimate_message_tokens(
            Message.text("user", "next")
        )
        if answer:
            expected += ObservedDeltaContextEstimator().estimate_message_tokens(
                Message.text("assistant", answer)
            )
        assert pressure.estimated_delta_input_tokens == expected
        assert pressure.estimated_message_count == 1 + bool(answer)
        assert pressure.anchor_transcript_cursor == 1
        assert (
            _prompt_cache_extension_messages(policy.requests[1], max_attachment_results=2)
            == policy.requests[1].messages
        )
        third = [
            event
            async for event in app.resume(
                ResumeRequest(session_id="coverage", messages=[Message.text("user", "last")])
            )
        ]
        assert any(event.type == EventType.MODEL_COMPLETED for event in third)
        pressure = policy.requests[2].context_usage.input_pressure
        assert pressure is not None
        assert pressure.observed_context_input_tokens == 20
        assert pressure.estimated_message_count == 2
        assert pressure.estimated_delta_input_tokens < 10
        if backend != "memory":
            await store.close()

    asyncio.run(run())


class LookupTool(Tool):
    spec = ToolSpec(
        name="lookup",
        description="Look up a value",
        input_schema={"type": "object", "properties": {}},
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(content="result")


def test_tool_call_and_result_are_both_after_input_coverage() -> None:
    async def run() -> None:
        provider = ScriptedProvider(
            [
                [
                    ModelStreamEvent.tool_call(name="lookup", arguments={}, id="lookup-1"),
                    completion(),
                ],
                [ModelStreamEvent.text_delta("done"), completion()],
            ]
        )
        policy = CapturePolicy()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake"), context_policy=policy, tools=[LookupTool()]
        )
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="tool-coverage",
                    messages=[Message.text("user", "look up")],
                )
            )
        ]
        assert len(provider.requests) == 2
        pressure = policy.requests[1].context_usage.input_pressure
        assert pressure is not None
        assert pressure.anchor_transcript_cursor == 1
        assert pressure.estimated_message_count == 2
        assert pressure.estimated_delta_input_tokens > 0
        durable = await app.session_store.query_events(
            EventQuery(session_id="tool-coverage", event_type=EventType.MODEL_COMPLETED)
        )
        assert durable[0].event.payload["input_coverage"]["transcript_cursor"] == 1
        assert any(event.type == EventType.TOOL_CALL_COMPLETED for event in events)

    asyncio.run(run())


@pytest.mark.parametrize("strategy", ["tool", "native"])
def test_structured_output_records_actual_pre_response_projection(strategy: str) -> None:
    async def run() -> None:
        first_output = (
            ModelStreamEvent.tool_call(
                name=STRUCTURED_OUTPUT_TOOL_NAME, arguments={"output": {"answer": "ok"}}, id="final"
            )
            if strategy == "tool"
            else ModelStreamEvent.text_delta('{"answer":"ok"}')
        )
        provider = ScriptedProvider([[first_output, completion()], [first_output, completion()]])
        policy = CapturePolicy()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake"), context_policy=policy)
        spec = StructuredOutputSpec(
            name="answer",
            strategy=strategy,
            json_schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        )
        first = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="structured-coverage",
                    messages=[Message.text("user", "answer")],
                    structured_output=spec,
                )
            )
        ]
        completed = next(event for event in first if event.type == EventType.MODEL_COMPLETED)
        assert completed.payload["input_coverage"] == context_input_coverage(
            provider.requests[0].messages, transcript_cursor=1
        ).model_dump(mode="json")
        assert any(event.type == EventType.STRUCTURED_OUTPUT_VALIDATED for event in first)
        second = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id="structured-coverage",
                    messages=[Message.text("user", "again")],
                    structured_output=spec,
                )
            )
        ]
        assert any(event.type == EventType.STRUCTURED_OUTPUT_VALIDATED for event in second)
        # TOOL adds a request-only system instruction: raw transcript pressure
        # must fall back rather than bind that projection to transcript offsets.
        usage = policy.requests[1].context_usage
        estimate = ObservedDeltaContextEstimator().estimate_anchored_request(
            usage=usage, messages=policy.requests[1].messages
        )
        if strategy == "tool":
            assert estimate.method == "local_full_request_estimate"
        else:
            assert estimate.method == "observed_plus_estimated_delta_with_overhead"
            assert estimate.estimated_message_count == 2

    asyncio.run(run())


def test_absolute_input_cursor_is_separate_from_retained_message_count() -> None:
    retained = [Message.text("user", "retained input")]
    usage = ContextUsageState(
        last_input_tokens=10,
        last_transcript_cursor=21,
        input_coverage=context_input_coverage(retained, transcript_cursor=20),
    )
    estimate = ObservedDeltaContextEstimator().estimate(
        usage=usage,
        messages=[*retained, Message.text("assistant", "new answer")],
    )
    assert estimate is not None
    assert estimate.anchor_transcript_cursor == 20
    assert estimate.current_transcript_cursor == 21
    assert estimate.estimated_message_count == 1
