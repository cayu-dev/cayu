from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from cayu.core import AgentSpec, Event, EventType, Message
from cayu.providers import ModelProvider, ModelProviderError, ModelRequest, ModelStreamEvent
from cayu.runtime import CayuApp, RetryPolicy, RunRequest, StructuredOutputSpec
from cayu.runtime.structured_output import STRUCTURED_OUTPUT_TOOL_NAME


class _ScriptedStructuredOutputProvider(ModelProvider):
    name = "structured-identity"
    supports_native_structured_output = True

    def __init__(
        self,
        outcomes: list[list[ModelStreamEvent] | ModelProviderError],
    ) -> None:
        self._outcomes = outcomes
        self.calls = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        outcome = self._outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, ModelProviderError):
            raise outcome
        for event in outcome:
            yield event


def _collect_events(
    provider: ModelProvider,
    *,
    session_id: str,
    structured_output: StructuredOutputSpec,
    retry_policy: RetryPolicy | None = None,
) -> list[Event]:
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="identity-model"))

    async def collect() -> list[Event]:
        _ = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "Return the structured answer.")],
                    structured_output=structured_output,
                    retry_policy=retry_policy,
                )
            )
        ]
        return await app.session_store.load_events(session_id)

    return asyncio.run(collect())


def _attempt_identity(event: Event) -> dict[str, object]:
    return {
        "model_step_id": event.payload.get("model_step_id"),
        "model_attempt_id": event.payload.get("model_attempt_id"),
    }


def _tool_round_identity(event: Event) -> dict[str, object]:
    return {
        **_attempt_identity(event),
        "tool_round_id": event.payload.get("tool_round_id"),
    }


def test_native_structured_output_uses_successful_provider_attempt_identity_after_retry() -> None:
    provider = _ScriptedStructuredOutputProvider(
        [
            ModelProviderError(
                "retry the provider request",
                provider="structured-identity",
                retryable=True,
            ),
            [
                ModelStreamEvent.text_delta('{"answer":"ok"}'),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    events = _collect_events(
        provider,
        session_id="sess_native_structured_identity_after_provider_retry",
        structured_output=StructuredOutputSpec(
            strategy="native",
            json_schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        ),
        retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0),
    )

    starts = [event for event in events if event.type == EventType.MODEL_STARTED]
    completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
    structured_events = [
        event
        for event in events
        if event.type
        in {
            EventType.STRUCTURED_OUTPUT_VALIDATING,
            EventType.STRUCTURED_OUTPUT_VALIDATED,
        }
    ]

    assert provider.calls == 2
    assert len(starts) == 2
    assert starts[0].payload["model_step_id"] == starts[1].payload["model_step_id"]
    assert starts[0].payload["model_attempt_id"] != starts[1].payload["model_attempt_id"]
    assert completed.payload["attempt"] == 2
    assert [event.payload["attempt"] for event in structured_events] == [1, 1]
    assert all(
        _attempt_identity(event) == _attempt_identity(completed) for event in structured_events
    )
    assert all("tool_round_id" not in event.payload for event in structured_events)


def test_tool_structured_output_repair_preserves_each_completion_round_identity() -> None:
    provider = _ScriptedStructuredOutputProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_invalid",
                    name=STRUCTURED_OUTPUT_TOOL_NAME,
                    arguments={"output": {"wrong": "value"}},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="call_valid",
                    name=STRUCTURED_OUTPUT_TOOL_NAME,
                    arguments={"output": {"answer": "fixed"}},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
        ]
    )
    events = _collect_events(
        provider,
        session_id="sess_tool_structured_repair_execution_identities",
        structured_output=StructuredOutputSpec(
            json_schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
            max_retries=1,
        ),
    )

    completions = [event for event in events if event.type == EventType.MODEL_COMPLETED]
    structured_events = [
        event
        for event in events
        if event.type
        in {
            EventType.STRUCTURED_OUTPUT_VALIDATING,
            EventType.STRUCTURED_OUTPUT_VALIDATED,
            EventType.STRUCTURED_OUTPUT_FAILED,
            EventType.STRUCTURED_OUTPUT_RETRY,
        }
    ]

    assert len(completions) == 2
    assert [event.type for event in structured_events] == [
        EventType.STRUCTURED_OUTPUT_VALIDATING,
        EventType.STRUCTURED_OUTPUT_FAILED,
        EventType.STRUCTURED_OUTPUT_RETRY,
        EventType.STRUCTURED_OUTPUT_VALIDATING,
        EventType.STRUCTURED_OUTPUT_VALIDATED,
    ]
    assert all(
        _tool_round_identity(event) == _tool_round_identity(completions[0])
        for event in structured_events[:3]
    )
    assert all(
        _tool_round_identity(event) == _tool_round_identity(completions[1])
        for event in structured_events[3:]
    )
    assert _tool_round_identity(completions[0]) != _tool_round_identity(completions[1])
