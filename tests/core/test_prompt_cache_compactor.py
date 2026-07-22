from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    CheckpointCompactionContextPolicy,
    CompactionRequest,
    ContextPressureOverhead,
    ContextRequest,
    ContextUsageState,
    Environment,
    EnvironmentSpec,
    EventType,
    LocalArtifactStore,
    Message,
    PromptCacheCompactor,
    ResumeRequest,
    RetryPolicy,
    RunRequest,
    Session,
    TextPart,
    ThinkingConfig,
)
from cayu.artifacts import RESOLVED_FILE_ATTACHMENTS_OPTION
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.providers import ModelProvider, ModelProviderError, ModelRequest, ModelStreamEvent
from cayu.runtime.context import ContextBuildError


class RecordingProvider(ModelProvider):
    name = "recording"

    def __init__(self, events: list[ModelStreamEvent]) -> None:
        self.events = events
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        for event in self.events:
            yield event


class SequencedProvider(ModelProvider):
    name = "sequenced"

    def __init__(self, responses: list[list[ModelStreamEvent]]) -> None:
        self.responses = responses
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        for event in self.responses[len(self.requests) - 1]:
            yield event


class RetryOnceProvider(ModelProvider):
    name = "retry-once"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if len(self.requests) == 1:
            raise ModelProviderError(
                "provider overloaded",
                provider=self.name,
                status_code=529,
                retryable=True,
            )
        yield ModelStreamEvent.text_delta("recovered summary")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class ToolCallFailureProvider(ModelProvider):
    name = "tool-call-failure"

    def __init__(self, failure_kind: str) -> None:
        self.failure_kind = failure_kind
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if len(self.requests) == 1:
            yield ModelStreamEvent.tool_call(id="call_1", name="inspect_report", arguments={})
            if self.failure_kind == "event":
                yield ModelStreamEvent.error("stream failed after tool call")
                return
            if self.failure_kind == "post_completion_exception":
                yield ModelStreamEvent.completed(
                    {
                        "finish_reason": "tool_use",
                        "usage": {"input_tokens": 100, "output_tokens": 10},
                    }
                )
            raise RuntimeError("transport failed after tool call")
        yield ModelStreamEvent.text_delta("bounded summary")
        yield ModelStreamEvent.completed(
            {
                "finish_reason": "stop",
                "usage": {"input_tokens": 20, "output_tokens": 5},
            }
        )


class ToolThenBoundedProviderErrorProvider(ModelProvider):
    name = "tool-then-bounded-error"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self.bounded_error = ModelProviderError(
            "bounded provider unavailable",
            provider=self.name,
            status_code=503,
            error_code="service_unavailable",
            retryable=False,
            retry_after_s=2.5,
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if len(self.requests) == 1:
            yield ModelStreamEvent.tool_call(
                id="call_1",
                name="inspect_report",
                arguments={},
            )
            return
        raise self.bounded_error


class InspectReportTool(Tool):
    spec = ToolSpec(
        name="inspect_report",
        description="Inspect a report.",
        input_schema={"type": "object", "properties": {}},
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(content="inspected")


async def collect_events(stream) -> list:
    return [event async for event in stream]


def test_prompt_cache_compactor_extends_the_exact_model_request_prefix() -> None:
    provider = RecordingProvider(
        [
            ModelStreamEvent.text_delta("cache-aware summary"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    cached_request = ModelRequest(
        model="claude-sonnet-4-6",
        messages=[
            Message.text("system", "You are careful."),
            Message.text("user", "Inspect the attached report with the registered tool."),
        ],
        tools=[
            {
                "name": "inspect_report",
                "description": "Inspect a report.",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
        options={
            "anthropic": {
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
                "max_tokens": 4096,
            },
            "thinking": {"enabled": True, "effort": "high"},
            "structured_output": {"strategy": "native"},
            RESOLVED_FILE_ATTACHMENTS_OPTION: {
                "report": {
                    "artifact_id": "report",
                    "kind": "document",
                    "filename": "report.pdf",
                    "content_type": "application/pdf",
                    "data_base64": "JVBERg==",
                    "metadata": {},
                }
            },
        },
    )
    compactor = PromptCacheCompactor(
        provider=provider,
        options={"anthropic": {"max_tokens": 512}},
    )

    result = asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=Session(
                    id="prompt-cache-prefix",
                    agent_name="assistant",
                    provider_name="recording",
                    model="claude-sonnet-4-6",
                ),
                agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
                messages=[Message.text("user", "newly compactable")],
                context_messages=cached_request.messages,
                cache_prefix_request=cached_request,
            )
        )
    )

    sent = provider.requests[0]
    assert result.summary == "cache-aware summary"
    assert sent.model == cached_request.model
    assert sent.tools == cached_request.tools
    assert sent.messages[:-1] == cached_request.messages
    assert sent.options["thinking"] == {"enabled": True, "effort": "high"}
    assert (
        sent.options[RESOLVED_FILE_ATTACHMENTS_OPTION]
        == cached_request.options[RESOLVED_FILE_ATTACHMENTS_OPTION]
    )
    assert sent.options["structured_output"] is None
    assert sent.options["anthropic"] == {
        "cache_control": {"type": "ephemeral", "ttl": "1h"},
        "max_tokens": 512,
    }


def test_prompt_cache_compactor_uses_bounded_cross_model_fallback_for_override() -> None:
    provider = RecordingProvider(
        [
            ModelStreamEvent.text_delta("cross-model summary"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    cached_request = ModelRequest(
        model="claude-sonnet-4-6",
        messages=[Message.text("user", "cached context")],
    )

    result = asyncio.run(
        PromptCacheCompactor(
            provider=provider,
            model="different-model",
        ).compact(
            CompactionRequest(
                session=Session(
                    id="prompt-cache-model-mismatch",
                    agent_name="assistant",
                    provider_name="recording",
                    model="claude-sonnet-4-6",
                ),
                agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
                messages=[Message.text("user", "newly compactable context")],
                context_messages=cached_request.messages,
                cache_prefix_request=cached_request,
            )
        )
    )

    assert result.summary == "cross-model summary"
    assert result.metadata["compactor"] == "ModelCompactor"
    assert provider.requests[0].model == "different-model"
    assert provider.requests[0].tools == []
    assert "newly compactable context" in provider.requests[0].messages[1].content[0].text


def test_prompt_cache_compactor_uses_bounded_fallback_when_session_model_changed() -> None:
    provider = RecordingProvider(
        [
            ModelStreamEvent.text_delta("current-model summary"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    cached_request = ModelRequest(
        model="old-model",
        messages=[Message.text("user", "full cached context")],
        tools=[
            {
                "name": "inspect_report",
                "description": "Inspect a report.",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
    )

    result = asyncio.run(
        PromptCacheCompactor(provider=provider).compact(
            CompactionRequest(
                session=Session(
                    id="prompt-cache-session-model-changed",
                    agent_name="assistant",
                    provider_name="recording",
                    model="new-model",
                ),
                agent=AgentSpec(name="assistant", model="new-model"),
                messages=[Message.text("user", "newly compactable context")],
                context_messages=cached_request.messages,
                cache_prefix_request=cached_request,
            )
        )
    )

    assert result.summary == "current-model summary"
    assert result.metadata["compactor"] == "ModelCompactor"
    assert provider.requests[0].model == "new-model"
    assert provider.requests[0].tools == []
    assert "full cached context" not in provider.requests[0].messages[1].content[0].text


def test_prompt_cache_compactor_requires_model_for_cross_provider_compaction() -> None:
    provider = RecordingProvider([])
    cached_request = ModelRequest(
        model="claude-sonnet-4-6",
        messages=[Message.text("user", "full cached context")],
    )

    with pytest.raises(
        ValueError,
        match="model is required when the compactor provider differs",
    ):
        asyncio.run(
            PromptCacheCompactor(provider=provider).compact(
                CompactionRequest(
                    session=Session(
                        id="prompt-cache-provider-model-required",
                        agent_name="assistant",
                        provider_name="original-provider",
                        model="claude-sonnet-4-6",
                    ),
                    agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
                    messages=[Message.text("user", "newly compactable context")],
                    context_messages=cached_request.messages,
                    cache_prefix_request=cached_request,
                    force_bounded_compaction=True,
                )
            )
        )

    assert provider.requests == []


def test_prompt_cache_compactor_uses_explicit_model_for_cross_provider_compaction() -> None:
    provider = RecordingProvider(
        [
            ModelStreamEvent.text_delta("cross-provider summary"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    cached_request = ModelRequest(
        model="claude-sonnet-4-6",
        messages=[Message.text("user", "full cached context")],
        tools=[
            {
                "name": "inspect_report",
                "description": "Inspect a report.",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
        options={
            RESOLVED_FILE_ATTACHMENTS_OPTION: {
                "report": {
                    "artifact_id": "report",
                    "kind": "document",
                    "filename": "report.pdf",
                    "content_type": "application/pdf",
                    "data_base64": "JVBERg==",
                    "metadata": {},
                }
            }
        },
    )

    result = asyncio.run(
        PromptCacheCompactor(provider=provider, model="gpt-4.1-mini").compact(
            CompactionRequest(
                session=Session(
                    id="prompt-cache-provider-mismatch",
                    agent_name="assistant",
                    provider_name="original-provider",
                    model="claude-sonnet-4-6",
                ),
                agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
                messages=[Message.text("user", "newly compactable context")],
                context_messages=cached_request.messages,
                cache_prefix_request=cached_request,
            )
        )
    )

    assert result.summary == "cross-provider summary"
    assert result.metadata["compactor"] == "ModelCompactor"
    assert provider.requests[0].model == "gpt-4.1-mini"
    assert provider.requests[0].tools == []
    assert RESOLVED_FILE_ATTACHMENTS_OPTION not in provider.requests[0].options
    assert "full cached context" not in provider.requests[0].messages[1].content[0].text
    assert "newly compactable context" in provider.requests[0].messages[1].content[0].text


def test_prompt_cache_compactor_uses_bounded_fallback_for_tool_structured_output() -> None:
    provider = RecordingProvider(
        [
            ModelStreamEvent.text_delta("plain summary"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    cached_request = ModelRequest(
        model="claude-sonnet-4-6",
        messages=[
            Message.text("system", "You are careful."),
            Message.text(
                "system",
                "Call `__cayu_submit_structured_output` with the final answer.",
            ),
            Message.text("user", "return a report"),
        ],
        tools=[
            {
                "name": "__cayu_submit_structured_output",
                "description": "Submit structured output.",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
        options={"structured_output": {"strategy": "tool"}},
    )

    result = asyncio.run(
        PromptCacheCompactor(provider=provider).compact(
            CompactionRequest(
                session=Session(
                    id="prompt-cache-tool-structured-output",
                    agent_name="assistant",
                    provider_name="recording",
                    model="claude-sonnet-4-6",
                ),
                agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
                messages=[Message.text("user", "newly compactable context")],
                context_messages=cached_request.messages,
                cache_prefix_request=cached_request,
            )
        )
    )

    assert result.summary == "plain summary"
    assert result.metadata["compactor"] == "ModelCompactor"
    assert provider.requests[0].tools == []
    assert [message.role for message in provider.requests[0].messages] == ["system", "user"]
    assert "__cayu_submit_structured_output" not in str(provider.requests[0].model_dump())


def test_prompt_cache_compactor_degrades_exact_tool_call_to_bounded_input() -> None:
    provider = SequencedProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="inspect_report",
                    arguments={},
                ),
                ModelStreamEvent.completed(
                    {
                        "model": "claude-sonnet-4-6-20260601",
                        "finish_reason": "tool_use",
                        "usage": {"input_tokens": 100, "output_tokens": 10},
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta("bounded summary"),
                ModelStreamEvent.completed(
                    {
                        "model": "claude-sonnet-4-6-20260601",
                        "finish_reason": "stop",
                        "usage": {"input_tokens": 20, "output_tokens": 5},
                    }
                ),
            ],
        ]
    )
    cached_request = ModelRequest(
        model="claude-sonnet-4-6",
        messages=[Message.text("user", "full cached context")],
        tools=[
            {
                "name": "inspect_report",
                "description": "Inspect a report.",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
    )

    result = asyncio.run(
        PromptCacheCompactor(provider=provider).compact(
            CompactionRequest(
                session=Session(
                    id="prompt-cache-tool-call-degradation",
                    agent_name="assistant",
                    provider_name="sequenced",
                    model="claude-sonnet-4-6",
                ),
                agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
                messages=[Message.text("user", "newly compactable context")],
                context_messages=cached_request.messages,
                cache_prefix_request=cached_request,
            )
        )
    )

    assert result.summary == "bounded summary"
    assert len(provider.requests) == 2
    assert provider.requests[0].tools == cached_request.tools
    assert provider.requests[1].tools == []
    assert "full cached context" not in str(provider.requests[1].model_dump())
    assert result.metadata["prompt_cache_exact_attempt"] == "rejected_tool_call"
    assert [payload["compactor"] for payload in result.model_completed_payloads] == [
        "PromptCacheCompactor",
        "ModelCompactor",
    ]
    assert [
        payload.get("usage_metrics", {}).get("input_tokens")
        for payload in result.model_completed_payloads
    ] == [100, 20]


@pytest.mark.parametrize("failure_kind", ["event", "exception"])
def test_prompt_cache_compactor_degrades_when_exact_tool_call_stream_fails(
    failure_kind: str,
) -> None:
    provider = ToolCallFailureProvider(failure_kind)
    cached_request = ModelRequest(
        model="claude-sonnet-4-6",
        messages=[Message.text("user", "full cached context")],
        tools=[
            {
                "name": "inspect_report",
                "description": "Inspect a report.",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
    )

    result = asyncio.run(
        PromptCacheCompactor(provider=provider).compact(
            CompactionRequest(
                session=Session(
                    id=f"prompt-cache-tool-call-{failure_kind}",
                    agent_name="assistant",
                    provider_name=provider.name,
                    model="claude-sonnet-4-6",
                ),
                agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
                messages=[Message.text("user", "newly compactable context")],
                context_messages=cached_request.messages,
                cache_prefix_request=cached_request,
            )
        )
    )

    assert result.summary == "bounded summary"
    assert len(provider.requests) == 2
    assert provider.requests[1].tools == []
    assert result.model_completed_payloads[0]["compaction_outcome"] == ("rejected_tool_call")
    assert result.model_completed_payloads[0]["usage_unavailable_reason"] == (
        "compaction tool-call attempt ended without provider completion usage"
    )
    assert "usage_metrics" not in result.model_completed_payloads[0]


def test_prompt_cache_compactor_preserves_bounded_provider_error_after_tool_degradation() -> None:
    provider = ToolThenBoundedProviderErrorProvider()
    cached_request = ModelRequest(
        model="claude-sonnet-4-6",
        messages=[Message.text("user", "full cached context")],
        tools=[
            {
                "name": "inspect_report",
                "description": "Inspect a report.",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
    )

    with pytest.raises(ModelProviderError) as exc_info:
        asyncio.run(
            PromptCacheCompactor(provider=provider).compact(
                CompactionRequest(
                    session=Session(
                        id="prompt-cache-tool-call-bounded-provider-error",
                        agent_name="assistant",
                        provider_name=provider.name,
                        model="claude-sonnet-4-6",
                    ),
                    agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
                    messages=[Message.text("user", "newly compactable context")],
                    context_messages=cached_request.messages,
                    cache_prefix_request=cached_request,
                )
            )
        )

    assert len(provider.requests) == 2
    assert provider.requests[1].tools == []
    assert exc_info.value is provider.bounded_error
    assert exc_info.value.status_code == 503
    assert exc_info.value.error_code == "service_unavailable"
    assert exc_info.value.retryable is False
    assert exc_info.value.retry_after_s == 2.5


def test_prompt_cache_compaction_failure_telemetry_is_invocation_scoped() -> None:
    provider = ToolThenBoundedProviderErrorProvider()
    policy = CheckpointCompactionContextPolicy(
        compactor=PromptCacheCompactor(provider=provider),
        max_user_turns=1,
        compact_after_messages=2,
    )
    messages = [
        Message.text("user", "old request"),
        Message.text("assistant", "old answer"),
        Message.text("user", "current request"),
    ]

    async def build_cache_prefix_request(context_messages: list[Message]) -> ModelRequest:
        return ModelRequest(
            model="claude-sonnet-4-6",
            messages=context_messages,
            tools=[
                {
                    "name": "inspect_report",
                    "description": "Inspect a report.",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
        )

    def context_request(*, force_bounded_compaction: bool) -> ContextRequest:
        return ContextRequest(
            session=Session(
                id="prompt-cache-failure-telemetry-scope",
                agent_name="assistant",
                provider_name=provider.name,
                model="claude-sonnet-4-6",
            ),
            agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
            messages=messages,
            step=1,
            context_usage=ContextUsageState(
                last_transcript_cursor=2,
                last_provider_name=provider.name,
                last_requested_model="claude-sonnet-4-6",
            ),
            build_cache_prefix_request=build_cache_prefix_request,
            force_bounded_compaction=force_bounded_compaction,
        )

    with pytest.raises(ContextBuildError) as exact_failure:
        asyncio.run(
            policy.build_with_checkpoint(
                context_request(force_bounded_compaction=False),
                checkpoint=None,
            )
        )

    assert exact_failure.value.cause is provider.bounded_error
    assert [telemetry.event_type for telemetry in exact_failure.value.compaction_telemetry] == [
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.MODEL_COMPLETED,
        EventType.CONTEXT_COMPACTION_FAILED,
    ]
    exact_attempts = [
        telemetry
        for telemetry in exact_failure.value.compaction_telemetry
        if telemetry.event_type == EventType.MODEL_COMPLETED
    ]
    assert [item.payload["compaction_outcome"] for item in exact_attempts] == [
        "rejected_tool_call",
        "provider_error",
    ]

    with pytest.raises(ContextBuildError) as bounded_only_failure:
        asyncio.run(
            policy.build_with_checkpoint(
                context_request(force_bounded_compaction=True),
                checkpoint=None,
            )
        )

    assert len(provider.requests) == 3
    assert bounded_only_failure.value.cause is provider.bounded_error
    assert [
        telemetry.event_type for telemetry in bounded_only_failure.value.compaction_telemetry
    ] == [
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.CONTEXT_COMPACTION_FAILED,
    ]
    bounded_attempt = bounded_only_failure.value.compaction_telemetry[1]
    assert bounded_attempt.payload["compaction_outcome"] == "provider_error"
    assert bounded_attempt.payload["usage_unavailable_reason"] == (
        "compaction provider dispatch failed without completion usage"
    )


def test_prompt_cache_compactor_retains_usage_before_post_completion_stream_failure() -> None:
    provider = ToolCallFailureProvider("post_completion_exception")
    cached_request = ModelRequest(
        model="claude-sonnet-4-6",
        messages=[Message.text("user", "full cached context")],
        tools=[
            {
                "name": "inspect_report",
                "description": "Inspect a report.",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
    )

    result = asyncio.run(
        PromptCacheCompactor(provider=provider).compact(
            CompactionRequest(
                session=Session(
                    id="prompt-cache-tool-call-post-completion-error",
                    agent_name="assistant",
                    provider_name=provider.name,
                    model="claude-sonnet-4-6",
                ),
                agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
                messages=[Message.text("user", "newly compactable context")],
                context_messages=cached_request.messages,
                cache_prefix_request=cached_request,
            )
        )
    )

    assert result.summary == "bounded summary"
    assert len(provider.requests) == 2
    assert result.model_completed_payloads[0]["compaction_outcome"] == ("rejected_tool_call")
    assert result.model_completed_payloads[0]["usage_metrics"]["input_tokens"] == 100
    assert "usage_unavailable_reason" not in result.model_completed_payloads[0]


def test_prompt_cache_compactor_falls_back_without_an_exact_request() -> None:
    provider = RecordingProvider([])

    result = asyncio.run(
        PromptCacheCompactor(provider=provider).compact(
            CompactionRequest(
                session=Session(
                    id="prompt-cache-unavailable",
                    agent_name="assistant",
                    provider_name="recording",
                    model="claude-sonnet-4-6",
                ),
                agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
                messages=[Message.text("user", "newly compactable context")],
                context_messages=[Message.text("user", "not a real provider request")],
            )
        )
    )

    assert result.metadata["compactor"] == "TranscriptDigestCompactor"
    assert provider.requests == []


def test_prompt_cache_compactor_accounts_for_usage_and_ignores_thinking() -> None:
    provider = RecordingProvider(
        [
            ModelStreamEvent.thinking("internal compaction reasoning"),
            ModelStreamEvent.text_delta("summary"),
            ModelStreamEvent.completed(
                {
                    "model": "claude-sonnet-4-6-20260601",
                    "usage": {"input_tokens": 100, "output_tokens": 10},
                }
            ),
        ]
    )
    cached_request = ModelRequest(
        model="claude-sonnet-4-6",
        messages=[Message.text("user", "long cached context")],
    )

    result = asyncio.run(
        PromptCacheCompactor(provider=provider).compact(
            CompactionRequest(
                session=Session(
                    id="prompt-cache-usage",
                    agent_name="assistant",
                    provider_name="recording",
                    model="claude-sonnet-4-6",
                ),
                agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
                messages=[Message.text("user", "long cached context")],
                context_messages=cached_request.messages,
                cache_prefix_request=cached_request,
            )
        )
    )

    assert result.summary == "summary"
    assert result.model_completed_payloads == [
        {
            "model": "claude-sonnet-4-6-20260601",
            "usage": {"input_tokens": 100, "output_tokens": 10},
            "provider_name": "recording",
            "requested_model": "claude-sonnet-4-6",
            "purpose": "context_compaction",
            "compactor": "PromptCacheCompactor",
            "usage_metrics": {
                "provider_name": "recording",
                "requested_model": "claude-sonnet-4-6",
                "model": "claude-sonnet-4-6-20260601",
                "input_tokens": 100,
                "output_tokens": 10,
                "total_tokens": 110,
                "reasoning_output_tokens": 0,
                "cache": {
                    "read_tokens": 0,
                    "write_tokens": 0,
                    "cached_input_tokens": 0,
                    "uncached_input_tokens": 100,
                },
            },
        }
    ]


def test_prompt_cache_compactor_retries_structured_provider_errors() -> None:
    provider = RetryOnceProvider()
    cached_request = ModelRequest(
        model="claude-sonnet-4-6",
        messages=[Message.text("user", "long cached context")],
    )

    result = asyncio.run(
        PromptCacheCompactor(
            provider=provider,
            retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
        ).compact(
            CompactionRequest(
                session=Session(
                    id="prompt-cache-retry",
                    agent_name="assistant",
                    provider_name=provider.name,
                    model="claude-sonnet-4-6",
                ),
                agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
                messages=[Message.text("user", "long cached context")],
                context_messages=cached_request.messages,
                cache_prefix_request=cached_request,
            )
        )
    )

    assert result.summary == "recovered summary"
    assert len(provider.requests) == 2


def test_checkpoint_policy_builds_the_cache_prefix_with_runtime_request_shape() -> None:
    provider = RecordingProvider(
        [
            ModelStreamEvent.text_delta("summary"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    policy = CheckpointCompactionContextPolicy(
        compactor=PromptCacheCompactor(provider=provider),
        max_user_turns=1,
        compact_after_messages=2,
    )
    messages = [
        Message.text("system", "You are careful."),
        Message.text("user", "old request"),
        Message.text("assistant", "old answer"),
        Message.text("user", "current request"),
    ]

    async def build_cache_prefix_request(context_messages: list[Message]) -> ModelRequest:
        return ModelRequest(
            model="claude-sonnet-4-6",
            messages=context_messages,
            tools=[
                {
                    "name": "inspect_report",
                    "description": "Inspect a report.",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            options={"thinking": {"enabled": True, "effort": "high"}},
        )

    result = asyncio.run(
        policy.build_with_checkpoint(
            ContextRequest(
                session=Session(
                    id="checkpoint-cache-prefix",
                    agent_name="assistant",
                    provider_name="recording",
                    model="claude-sonnet-4-6",
                ),
                agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
                messages=messages,
                step=1,
                context_usage=ContextUsageState(
                    last_transcript_cursor=3,
                    last_provider_name="recording",
                    last_requested_model="claude-sonnet-4-6",
                ),
                build_cache_prefix_request=build_cache_prefix_request,
            ),
            checkpoint=None,
        )
    )

    assert result.checkpoint is not None
    assert provider.requests[0].tools[0]["name"] == "inspect_report"
    assert provider.requests[0].options["thinking"] == {
        "enabled": True,
        "effort": "high",
    }
    assert provider.requests[0].messages[:-1] == messages


def test_checkpoint_policy_reports_start_when_cache_prefix_build_fails() -> None:
    provider = RecordingProvider([])
    policy = CheckpointCompactionContextPolicy(
        compactor=PromptCacheCompactor(provider=provider),
        max_user_turns=1,
        compact_after_messages=2,
    )

    async def failing_builder(context_messages: list[Message]) -> ModelRequest:
        assert context_messages
        raise RuntimeError("cache prefix construction failed")

    with pytest.raises(ContextBuildError, match="cache prefix construction failed") as exc_info:
        asyncio.run(
            policy.build_with_checkpoint(
                ContextRequest(
                    session=Session(
                        id="checkpoint-cache-prefix-failure",
                        agent_name="assistant",
                        provider_name="recording",
                        model="claude-sonnet-4-6",
                    ),
                    agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
                    messages=[
                        Message.text("user", "old request"),
                        Message.text("assistant", "old answer"),
                        Message.text("user", "current request"),
                    ],
                    step=1,
                    context_usage=ContextUsageState(
                        last_transcript_cursor=2,
                        last_provider_name="recording",
                        last_requested_model="claude-sonnet-4-6",
                    ),
                    build_cache_prefix_request=failing_builder,
                ),
                checkpoint=None,
            )
        )

    assert [telemetry.event_type for telemetry in exc_info.value.compaction_telemetry] == [
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.CONTEXT_COMPACTION_FAILED,
    ]
    assert all(
        "bounded_input" not in telemetry.payload
        for telemetry in exc_info.value.compaction_telemetry
    )
    assert provider.requests == []


def test_prompt_cache_digest_exhaustion_can_progress_on_bounded_followup() -> None:
    provider = RecordingProvider(
        [
            ModelStreamEvent.text_delta("provider summary"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    policy = CheckpointCompactionContextPolicy(
        compactor=PromptCacheCompactor(provider=provider),
        max_user_turns=1,
        compact_after_messages=1,
    )
    messages = [
        Message.text("user", "oversized " + "x" * 10_000),
        Message.text("user", "current"),
    ]

    first = asyncio.run(
        policy.build_with_checkpoint(
            ContextRequest(
                session=Session(
                    id="prompt-cache-digest-exhaustion",
                    agent_name="assistant",
                    provider_name="recording",
                    model="claude-sonnet-4-6",
                ),
                agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
                messages=messages,
                step=1,
            ),
            checkpoint=None,
        )
    )

    assert first.checkpoint is not None
    assert first.checkpoint["context_compaction"]["compacted_transcript_cursor"] == 0
    assert first.checkpoint["context_compaction"]["progress"]["exhausted"] is True
    assert provider.requests == []

    second = asyncio.run(
        policy.build_with_checkpoint(
            ContextRequest(
                session=Session(
                    id="prompt-cache-digest-exhaustion",
                    agent_name="assistant",
                    provider_name="recording",
                    model="claude-sonnet-4-6",
                ),
                agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
                messages=messages,
                step=2,
            ),
            checkpoint=first.checkpoint,
        )
    )

    assert second.checkpoint is not None
    assert second.checkpoint["context_compaction"]["compacted_transcript_cursor"] == 1
    assert second.checkpoint["context_compaction"]["summary"] == "provider summary"
    assert "progress" not in second.checkpoint["context_compaction"]
    assert len(provider.requests) == 1


@pytest.mark.parametrize("last_transcript_cursor", [None, 0, 3])
def test_prompt_cache_digest_exhaustion_uses_fallback_key_without_valid_usage_cursor(
    last_transcript_cursor: int | None,
) -> None:
    provider = RecordingProvider([])
    policy = CheckpointCompactionContextPolicy(
        compactor=PromptCacheCompactor(provider=provider),
        max_user_turns=1,
        compact_after_messages=1,
    )
    messages = [
        Message.text("user", "oversized " + "x" * 10_000),
        Message.text("user", "current"),
    ]

    async def unexpected_cache_prefix_builder(_messages: list[Message]) -> ModelRequest:
        raise AssertionError("an invalid usage cursor cannot reconstruct a cache prefix")

    request = ContextRequest(
        session=Session(
            id="prompt-cache-missing-usage-cursor",
            agent_name="assistant",
            provider_name="recording",
            model="claude-sonnet-4-6",
        ),
        agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
        messages=messages,
        step=1,
        context_usage=ContextUsageState(
            last_transcript_cursor=last_transcript_cursor,
            last_provider_name="recording",
            last_requested_model="claude-sonnet-4-6",
        ),
        build_cache_prefix_request=unexpected_cache_prefix_builder,
    )
    first = asyncio.run(policy.build_with_checkpoint(request, checkpoint=None))

    assert first.checkpoint is not None
    compacted = first.checkpoint["context_compaction"]
    assert compacted["compacted_transcript_cursor"] == 0
    assert compacted["progress"]["exhausted"] is True
    assert compacted["progress"]["key"].startswith("transcript-digest:v2:")
    assert provider.requests == []


def test_checkpoint_policy_skips_cache_request_after_provider_identity_changes() -> None:
    provider = RecordingProvider(
        [
            ModelStreamEvent.text_delta("bounded summary"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    policy = CheckpointCompactionContextPolicy(
        compactor=PromptCacheCompactor(provider=provider),
        max_user_turns=1,
        compact_after_messages=2,
    )
    messages = [
        Message.text("user", "old request"),
        Message.text("assistant", "old answer"),
        Message.text("user", "current request"),
    ]

    async def unexpected_builder(context_messages: list[Message]) -> ModelRequest:
        raise AssertionError("a different provider cannot reuse the prior request cache")

    result = asyncio.run(
        policy.build_with_checkpoint(
            ContextRequest(
                session=Session(
                    id="checkpoint-provider-changed",
                    agent_name="assistant",
                    provider_name="recording",
                    model="claude-sonnet-4-6",
                ),
                agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
                messages=messages,
                step=1,
                context_usage=ContextUsageState(
                    last_transcript_cursor=2,
                    last_provider_name="previous-provider",
                    last_requested_model="claude-sonnet-4-6",
                ),
                build_cache_prefix_request=unexpected_builder,
            ),
            checkpoint=None,
        )
    )

    assert result.checkpoint is not None
    assert result.checkpoint["context_compaction"]["metadata"]["compactor"] == "ModelCompactor"
    assert len(provider.requests) == 1
    assert provider.requests[0].tools == []


def test_checkpoint_policy_skips_cache_request_after_requested_model_changes() -> None:
    provider = RecordingProvider(
        [
            ModelStreamEvent.text_delta("bounded summary"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    policy = CheckpointCompactionContextPolicy(
        compactor=PromptCacheCompactor(provider=provider),
        max_user_turns=1,
        compact_after_messages=2,
    )
    messages = [
        Message.text("user", "old request"),
        Message.text("assistant", "old answer"),
        Message.text("user", "current request"),
    ]

    async def unexpected_builder(context_messages: list[Message]) -> ModelRequest:
        raise AssertionError("a different requested model cannot reuse the prior request cache")

    result = asyncio.run(
        policy.build_with_checkpoint(
            ContextRequest(
                session=Session(
                    id="checkpoint-requested-model-changed",
                    agent_name="assistant",
                    provider_name="recording",
                    model="new-model",
                ),
                agent=AgentSpec(name="assistant", model="new-model"),
                messages=messages,
                step=1,
                context_usage=ContextUsageState(
                    last_transcript_cursor=2,
                    last_provider_name="recording",
                    last_requested_model="old-model",
                ),
                build_cache_prefix_request=unexpected_builder,
            ),
            checkpoint=None,
        )
    )

    assert result.checkpoint is not None
    assert result.checkpoint["context_compaction"]["metadata"]["compactor"] == "ModelCompactor"
    assert len(provider.requests) == 1
    assert provider.requests[0].model == "new-model"
    assert provider.requests[0].tools == []


def test_checkpoint_policy_skips_exact_builder_for_tool_structured_output() -> None:
    provider = RecordingProvider(
        [
            ModelStreamEvent.text_delta("bounded summary"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    policy = CheckpointCompactionContextPolicy(
        compactor=PromptCacheCompactor(provider=provider),
        max_user_turns=1,
        compact_after_messages=2,
    )
    messages = [
        Message.text("user", "old request"),
        Message.text("assistant", "old answer"),
        Message.text("user", "current request"),
    ]

    async def unexpected_builder(context_messages: list[Message]) -> ModelRequest:
        raise AssertionError("tool structured output must take the bounded path directly")

    result = asyncio.run(
        policy.build_with_checkpoint(
            ContextRequest(
                session=Session(
                    id="checkpoint-tool-structured-output",
                    agent_name="assistant",
                    provider_name="recording",
                    model="claude-sonnet-4-6",
                ),
                agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
                messages=messages,
                step=1,
                context_usage=ContextUsageState(
                    last_transcript_cursor=2,
                    last_provider_name="recording",
                    last_requested_model="claude-sonnet-4-6",
                ),
                pressure_overhead=ContextPressureOverhead(
                    structured_output_instruction="Call the reserved output tool."
                ),
                build_cache_prefix_request=unexpected_builder,
            ),
            checkpoint=None,
        )
    )

    assert result.checkpoint is not None
    assert result.checkpoint["context_compaction"]["metadata"]["compactor"] == "ModelCompactor"
    assert len(provider.requests) == 1
    assert provider.requests[0].tools == []


def test_checkpoint_policy_keeps_the_initial_prefix_until_it_can_compact() -> None:
    policy = CheckpointCompactionContextPolicy(
        max_user_turns=1,
        compact_after_messages=4,
    )
    messages = [
        Message.text("system", "You are careful."),
        Message.text("user", "old request"),
        Message.text("assistant", "old answer"),
        Message.text("user", "current request"),
    ]

    result = asyncio.run(
        policy.build_with_checkpoint(
            ContextRequest(
                session=Session(
                    id="checkpoint-warm-prefix",
                    agent_name="assistant",
                    provider_name="recording",
                    model="claude-sonnet-4-6",
                ),
                agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
                messages=messages,
                step=1,
            ),
            checkpoint=None,
        )
    )

    assert result.checkpoint is None
    assert result.messages == messages


def test_checkpoint_policy_falls_back_without_a_completed_request_cursor() -> None:
    provider = RecordingProvider([])
    policy = CheckpointCompactionContextPolicy(
        compactor=PromptCacheCompactor(provider=provider),
        max_user_turns=1,
        compact_after_messages=2,
    )
    messages = [
        Message.text("user", "old request"),
        Message.text("assistant", "old answer"),
        Message.text("user", "current request"),
    ]

    async def unexpected_builder(context_messages: list[Message]) -> ModelRequest:
        raise AssertionError("no completed request exists to rebuild")

    result = asyncio.run(
        policy.build_with_checkpoint(
            ContextRequest(
                session=Session(
                    id="checkpoint-cache-unavailable",
                    agent_name="assistant",
                    provider_name="recording",
                    model="claude-sonnet-4-6",
                ),
                agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
                messages=messages,
                step=1,
                build_cache_prefix_request=unexpected_builder,
            ),
            checkpoint=None,
        )
    )

    assert result.checkpoint is not None
    assert result.checkpoint["context_compaction"]["metadata"]["compactor"] == (
        "TranscriptDigestCompactor"
    )
    assert provider.requests == []


def test_checkpoint_policy_skips_cache_request_for_cross_model_override() -> None:
    provider = RecordingProvider(
        [
            ModelStreamEvent.text_delta("cross-model summary"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    policy = CheckpointCompactionContextPolicy(
        compactor=PromptCacheCompactor(provider=provider, model="different-model"),
        max_user_turns=1,
        compact_after_messages=2,
    )
    messages = [
        Message.text("user", "old request"),
        Message.text("assistant", "old answer"),
        Message.text("user", "current request"),
    ]

    async def unexpected_builder(context_messages: list[Message]) -> ModelRequest:
        raise AssertionError("cross-model compaction cannot reuse the request cache")

    result = asyncio.run(
        policy.build_with_checkpoint(
            ContextRequest(
                session=Session(
                    id="checkpoint-cross-model",
                    agent_name="assistant",
                    provider_name="recording",
                    model="claude-sonnet-4-6",
                ),
                agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
                messages=messages,
                step=1,
                context_usage=ContextUsageState(last_transcript_cursor=2),
                build_cache_prefix_request=unexpected_builder,
            ),
            checkpoint=None,
        )
    )

    assert result.checkpoint is not None
    assert result.checkpoint["context_compaction"]["metadata"]["compactor"] == "ModelCompactor"
    assert provider.requests[0].model == "different-model"


def test_checkpoint_policy_does_not_build_a_model_request_for_digest_compaction() -> None:
    policy = CheckpointCompactionContextPolicy(
        max_user_turns=1,
        compact_after_messages=2,
    )
    messages = [
        Message.text("user", "old request"),
        Message.text("assistant", "old answer"),
        Message.text("user", "current request"),
    ]

    async def unexpected_builder(context_messages: list[Message]) -> ModelRequest:
        raise AssertionError("digest compaction must not build or resolve a provider request")

    result = asyncio.run(
        policy.build_with_checkpoint(
            ContextRequest(
                session=Session(
                    id="checkpoint-digest-lazy",
                    agent_name="assistant",
                    provider_name="recording",
                    model="claude-sonnet-4-6",
                ),
                agent=AgentSpec(name="assistant", model="claude-sonnet-4-6"),
                messages=messages,
                step=1,
                build_cache_prefix_request=unexpected_builder,
            ),
            checkpoint=None,
        )
    )

    assert result.checkpoint is not None
    assert result.checkpoint["context_compaction"]["metadata"]["compactor"] == (
        "TranscriptDigestCompactor"
    )


def test_cayu_app_uses_cache_prefix_then_bounded_delta_and_accounts_for_both() -> None:
    provider = SequencedProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 2}}),
            ],
            [
                ModelStreamEvent.text_delta("compacted summary"),
                ModelStreamEvent.completed(
                    {
                        "usage": {
                            "input_tokens": 100,
                            "output_tokens": 10,
                            "cache_read_input_tokens": 80,
                        }
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 11, "output_tokens": 2}}),
            ],
            [
                ModelStreamEvent.text_delta("updated compacted summary"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 50, "output_tokens": 5}}),
            ],
            [
                ModelStreamEvent.text_delta("third answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 12, "output_tokens": 3}}),
            ],
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="claude-sonnet-4-6", system_prompt="Be careful."),
        tools=[InspectReportTool()],
        context_policy=CheckpointCompactionContextPolicy(
            compactor=PromptCacheCompactor(provider=provider),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )
    thinking = ThinkingConfig(effort="high")

    asyncio.run(
        collect_events(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="app-cache-prefix",
                    messages=[Message.text("user", "first request")],
                    thinking=thinking,
                )
            )
        )
    )
    first_resume_events = asyncio.run(
        collect_events(
            app.resume(
                ResumeRequest(
                    session_id="app-cache-prefix",
                    messages=[Message.text("user", "second request")],
                    thinking=thinking,
                )
            )
        )
    )
    second_resume_events = asyncio.run(
        collect_events(
            app.resume(
                ResumeRequest(
                    session_id="app-cache-prefix",
                    messages=[Message.text("user", "third request")],
                    thinking=thinking,
                )
            )
        )
    )

    initial_request, cached_compaction, second_request, delta_compaction, final_request = (
        provider.requests
    )
    assert cached_compaction.messages[: len(initial_request.messages)] == initial_request.messages
    assert cached_compaction.tools == initial_request.tools
    assert cached_compaction.options["thinking"] == initial_request.options["thinking"]
    assert (
        second_request.messages[1]
        .content[0]
        .text.startswith("Previous session context summary:\ncompacted summary")
    )
    assert delta_compaction.tools == []
    assert [message.role for message in delta_compaction.messages] == ["system", "user"]
    delta_prompt = delta_compaction.messages[1].content[0].text
    assert "Existing summary:\ncompacted summary" in delta_prompt
    assert "user: second request" in delta_prompt
    assert "assistant: second answer" in delta_prompt
    assert "user: first request" not in delta_prompt
    assert (
        final_request.messages[1]
        .content[0]
        .text.startswith("Previous session context summary:\nupdated compacted summary")
    )

    compaction_events = [
        event
        for event in first_resume_events + second_resume_events
        if event.type == EventType.MODEL_COMPLETED
        and event.payload.get("purpose") == "context_compaction"
    ]
    assert [event.payload["compactor"] for event in compaction_events] == [
        "PromptCacheCompactor",
        "ModelCompactor",
    ]
    usage = asyncio.run(app.get_session_usage("app-cache-prefix"))
    assert usage.model_steps == 5
    assert usage.usage.input_tokens == 263
    assert usage.usage.output_tokens == 22
    assert usage.usage.cache.read_tokens == 80
    assert usage.usage.cache.uncached_input_tokens == 183


def test_cayu_app_resume_model_override_cannot_reuse_previous_model_cache() -> None:
    provider = SequencedProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 2}}),
            ],
            [
                ModelStreamEvent.text_delta("bounded summary"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 8, "output_tokens": 2}}),
            ],
            [
                ModelStreamEvent.text_delta("new model answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 6, "output_tokens": 2}}),
            ],
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="old-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=PromptCacheCompactor(provider=provider),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    asyncio.run(
        collect_events(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="app-cache-model-override",
                    messages=[Message.text("user", "first request")],
                )
            )
        )
    )
    resume_events = asyncio.run(
        collect_events(
            app.resume(
                ResumeRequest(
                    session_id="app-cache-model-override",
                    model="new-model",
                    messages=[Message.text("user", "second request")],
                )
            )
        )
    )

    assert [request.model for request in provider.requests] == [
        "old-model",
        "new-model",
        "new-model",
    ]
    assert provider.requests[1].tools == []
    assert [message.role for message in provider.requests[1].messages] == ["system", "user"]
    assert any(
        event.type == EventType.MODEL_COMPLETED
        and event.payload.get("purpose") == "context_compaction"
        for event in resume_events
    )
    compaction_completed = [
        event for event in resume_events if event.type == EventType.CONTEXT_COMPACTION_COMPLETED
    ]
    assert len(compaction_completed) == 1
    assert compaction_completed[0].payload["compactor"] == "PromptCacheCompactor"
    assert compaction_completed[0].payload["chunk_mode"] == "single_request"
    assert resume_events[-1].type == EventType.SESSION_COMPLETED


def test_cayu_app_preserves_resolved_attachment_bytes_in_cached_prefix(tmp_path) -> None:
    provider = SequencedProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {"input_tokens": 10, "output_tokens": 2},
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {"input_tokens": 20, "output_tokens": 2},
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta("compacted summary"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("third answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            artifact_store=LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts"),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="claude-sonnet-4-6"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=PromptCacheCompactor(provider=provider),
            max_user_turns=1,
            compact_after_messages=3,
        ),
    )

    async def run_three_turns() -> tuple[str, str]:
        from io import BytesIO

        from PIL import Image

        buffer = BytesIO()
        Image.new("RGB", (1, 1), "white").save(buffer, format="PNG")
        old_part = await app.attach_file(
            buffer.getvalue(),
            filename="old-report.png",
            kind="image",
            session_id="app-cache-attachment",
        )
        await collect_events(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="app-cache-attachment",
                    messages=[
                        Message(
                            role="user",
                            content=[TextPart(text="inspect the old report"), old_part],
                        )
                    ],
                )
            )
        )
        current_part = await app.attach_file(
            buffer.getvalue(),
            filename="current-report.png",
            kind="image",
            session_id="app-cache-attachment",
        )
        await collect_events(
            app.resume(
                ResumeRequest(
                    session_id="app-cache-attachment",
                    messages=[
                        Message(
                            role="user",
                            content=[TextPart(text="inspect the current report"), current_part],
                        )
                    ],
                )
            )
        )
        await collect_events(
            app.resume(
                ResumeRequest(
                    session_id="app-cache-attachment",
                    messages=[Message.text("user", "compare the findings")],
                )
            )
        )
        return (
            old_part.attachment["artifact_id"],
            current_part.attachment["artifact_id"],
        )

    old_artifact_id, current_artifact_id = asyncio.run(run_three_turns())

    initial_request, warm_request, compaction_request, final_request = provider.requests
    assert set(initial_request.options[RESOLVED_FILE_ATTACHMENTS_OPTION]) == {old_artifact_id}
    warm_resolved = warm_request.options[RESOLVED_FILE_ATTACHMENTS_OPTION]
    assert set(warm_resolved) == {current_artifact_id}
    assert compaction_request.messages[: len(warm_request.messages)] == warm_request.messages
    assert compaction_request.options[RESOLVED_FILE_ATTACHMENTS_OPTION] == warm_resolved
    assert old_artifact_id not in compaction_request.options[RESOLVED_FILE_ATTACHMENTS_OPTION]
    assert final_request.options[RESOLVED_FILE_ATTACHMENTS_OPTION] == {}


def test_prompt_cache_compactor_uses_bounded_incremental_compaction_after_checkpoint() -> None:
    compaction_instruction = (
        "Preserve the mandatory retention token across every compaction. Return only a summary."
    )
    provider = SequencedProvider(
        [
            [
                ModelStreamEvent.text_delta("first summary"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("updated summary"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    compactor = PromptCacheCompactor(
        provider=provider,
        compaction_instruction=compaction_instruction,
    )
    session = Session(
        id="repeated-compaction",
        agent_name="assistant",
        provider_name="sequenced",
        model="claude-sonnet-4-6",
    )
    agent = AgentSpec(name="assistant", model="claude-sonnet-4-6")
    first_cached_request = ModelRequest(
        model="claude-sonnet-4-6",
        messages=[Message.text("user", "first full cached context")],
        tools=[
            {
                "name": "inspect_report",
                "description": "Inspect a report.",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
    )

    first = asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=session,
                agent=agent,
                messages=[Message.text("user", "first full cached context")],
                context_messages=first_cached_request.messages,
                cache_prefix_request=first_cached_request,
            )
        )
    )
    second = asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=session,
                agent=agent,
                messages=[Message.text("user", "new work since the checkpoint")],
                existing_summary=first.summary,
                context_messages=[
                    Message.text("user", "first full cached context"),
                    Message.text("assistant", "first answer"),
                    Message.text("user", "new work since the checkpoint"),
                ],
                cache_prefix_request=ModelRequest(
                    model="claude-sonnet-4-6",
                    messages=[Message.text("user", "an ever-growing raw transcript")],
                    tools=first_cached_request.tools,
                ),
            )
        )
    )

    incremental_request = provider.requests[1]
    assert first.metadata["compactor"] == "PromptCacheCompactor"
    assert second.metadata["compactor"] == "ModelCompactor"
    assert incremental_request.tools == []
    assert [message.role for message in incremental_request.messages] == ["system", "user"]
    assert incremental_request.messages[0].content[0].text == compaction_instruction
    incremental_prompt = incremental_request.messages[1].content[0].text
    assert "Existing summary:\nfirst summary" in incremental_prompt
    assert "user: new work since the checkpoint" in incremental_prompt
    assert "an ever-growing raw transcript" not in incremental_prompt
