"""Prompt cache compaction contracts."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from uuid import UUID

import pytest
from tests._session_provenance import fixture_session_invocation
from tests.runtime.runtime_compaction_support import (
    RecordingCompactor,
    _expected_compaction_failure,
)
from tests.runtime.runtime_provider_support import ContextOverflowProvider, FakeProvider
from tests.runtime.runtime_session_support import (
    _expected_failure_evidence,
    _test_session,
    collect_events,
    collect_resume_events,
)

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.artifacts.attachments import RESOLVED_FILE_ATTACHMENTS_OPTION
from cayu.budgets.pricing import ModelPrice, PriceBook
from cayu.context.base import (
    CheckpointCompactionContextPolicy,
    CompactionRequest,
    CompactionResult,
    ModelCompactor,
    PromptCacheCompactor,
)
from cayu.context.footprints import RequestFootprintConfig
from cayu.events import EventType
from cayu.messages import Message, MessageRole, TextPart
from cayu.providers.base import ModelContextOverflowError, ModelRequest, ModelStreamEvent
from cayu.runtime._event_projection import PRIVATE_EVENT_AUTHORITY
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.records import Session
from cayu.sessions.requests import ResumeRequest, RunRequest


def test_prompt_cache_compactor_sends_context_messages_as_prefix():
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("session cache "),
            ModelStreamEvent.text_delta("summary"),
            ModelStreamEvent.completed({"usage": {"output_tokens": 3}}),
        ]
    )
    compactor = PromptCacheCompactor(provider=provider)

    context = [
        Message.text("system", "You are helpful."),
        Message.text("user", "first request"),
        Message.text("assistant", "first answer"),
        Message.text("user", "second request"),
        Message.text("assistant", "second answer"),
    ]
    result = asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[Message.text("user", "old request")],
                context_messages=context,
                cache_prefix_request=ModelRequest(model="fake-model", messages=context),
            )
        )
    )

    assert result.summary == "session cache summary"
    assert result.metadata["compactor"] == "PromptCacheCompactor"
    assert result.metadata["provider"] == "fake"
    assert result.metadata["model"] == "fake-model"
    assert result.metadata["prompt_cache_compaction"] is True
    assert result.metadata["context_message_count"] == 5
    assert result.metadata["attachment_results_preserved"] == 0

    assert len(provider.requests) == 1
    request = provider.requests[0]
    assert request.model == "fake-model"
    assert request.tools == []
    assert len(request.messages) == len(context) + 1
    for i, original in enumerate(context):
        assert request.messages[i].role == original.role
        assert request.messages[i].content[0].text == original.content[0].text
    instruction_message = request.messages[-1]
    assert instruction_message.role == MessageRole.USER
    assert "Summarize the conversation above" in instruction_message.content[0].text


def test_prompt_cache_compactor_falls_back_when_context_messages_empty():
    provider = FakeProvider([ModelStreamEvent.text_delta("should not be called")])
    compactor = PromptCacheCompactor(provider=provider)

    result = asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[Message.text("user", "old request")],
            )
        )
    )

    assert len(provider.requests) == 0
    assert "old request" in result.summary


def test_prompt_cache_compactor_uses_session_model_by_default():
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    compactor = PromptCacheCompactor(provider=provider)
    session = Session(
        id="sess_model_test",
        agent_name="assistant",
        provider_name="fake",
        model="claude-sonnet-4-20260601",
        invocation=fixture_session_invocation("sess_model_test"),
    )

    asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=session,
                agent=AgentSpec(name="assistant", model="claude-sonnet-4-20260601"),
                messages=[Message.text("user", "old")],
                context_messages=[Message.text("user", "hello")],
                cache_prefix_request=ModelRequest(
                    model="claude-sonnet-4-20260601",
                    messages=[Message.text("user", "hello")],
                ),
            )
        )
    )

    assert provider.requests[0].model == "claude-sonnet-4-20260601"


def test_prompt_cache_compactor_uses_model_override():
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    compactor = PromptCacheCompactor(provider=provider, model="override-model")

    asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[Message.text("user", "old")],
                context_messages=[Message.text("user", "hello")],
                cache_prefix_request=ModelRequest(
                    model="fake-model",
                    messages=[Message.text("user", "hello")],
                ),
            )
        )
    )

    assert provider.requests[0].model == "override-model"


def test_prompt_cache_compactor_marks_unpriced_exact_tool_call_attempt():
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(id="call_1", name="echo", arguments={"text": "bad"}),
            ],
            [
                ModelStreamEvent.text_delta("bounded summary"),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {"input_tokens": 20, "output_tokens": 5},
                    }
                ),
            ],
        ]
    )
    compactor = PromptCacheCompactor(provider=provider)

    result = asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[Message.text("user", "old")],
                context_messages=[Message.text("user", "hello")],
                cache_prefix_request=ModelRequest(
                    model="fake-model",
                    messages=[Message.text("user", "hello")],
                ),
            )
        )
    )

    assert result.summary == "bounded summary"
    assert len(provider.requests) == 2
    assert result.metadata["prompt_cache_exact_attempt"] == "rejected_tool_call"
    assert result.model_completed_payloads[0]["compaction_outcome"] == ("rejected_tool_call")
    assert result.model_completed_payloads[0]["usage_unavailable_reason"] == (
        "compaction tool-call attempt ended without provider completion usage"
    )
    assert "usage_metrics" not in result.model_completed_payloads[0]
    assert result.model_completed_payloads[1]["usage_metrics"]["input_tokens"] == 20


def test_prompt_cache_compactor_bounds_exact_context_overflow_once():
    provider = ContextOverflowProvider(
        success_events=[
            ModelStreamEvent.text_delta("bounded summary"),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {"input_tokens": 20, "output_tokens": 5},
                }
            ),
        ]
    )
    compactor = PromptCacheCompactor(provider=provider)
    exact_tool = {
        "name": "exact_only_tool",
        "description": "Only present on the cached request.",
        "input_schema": {"type": "object", "properties": {}},
    }
    session = Session(
        id="sess_prompt_cache_overflow",
        agent_name="assistant",
        provider_name=provider.name,
        model="fake-model",
        invocation=fixture_session_invocation("sess_prompt_cache_overflow"),
    )

    result = asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=session,
                agent=AgentSpec(
                    name="assistant",
                    model="fake-model",
                    provider_options={
                        "bounded_only": {"enabled": True},
                        RESOLVED_FILE_ATTACHMENTS_OPTION: {"remove": True},
                    },
                ),
                messages=[Message.text("user", "bounded transcript only")],
                context_messages=[Message.text("user", "context projection")],
                cache_prefix_request=ModelRequest(
                    model="fake-model",
                    messages=[Message.text("user", "cached prefix only")],
                    tools=[exact_tool],
                    options={
                        "exact_only": {"enabled": True},
                        RESOLVED_FILE_ATTACHMENTS_OPTION: {"cached": True},
                    },
                ),
            )
        )
    )

    assert result.summary == "bounded summary"
    assert result.metadata["prompt_cache_exact_attempt"] == "context_overflow"
    assert len(provider.requests) == 2
    exact_request, bounded_request = provider.requests
    assert exact_request.tools == [exact_tool]
    assert exact_request.options["exact_only"] == {"enabled": True}
    assert RESOLVED_FILE_ATTACHMENTS_OPTION in exact_request.options
    assert bounded_request.tools == []
    assert bounded_request.options == {"bounded_only": {"enabled": True}}
    bounded_text = "\n".join(
        part.text
        for message in bounded_request.messages
        for part in message.content
        if isinstance(part, TextPart)
    )
    assert "bounded transcript only" in bounded_text
    assert "cached prefix only" not in bounded_text

    exact_attempt = result.model_completed_payloads[0]
    assert exact_attempt["compaction_outcome"] == "context_overflow"
    assert exact_attempt["context_overflow"] is True
    assert exact_attempt["error_type"] == "ModelContextOverflowError"
    assert exact_attempt["provider"] == provider.name
    assert exact_attempt["status_code"] == 400
    assert exact_attempt["provider_error_code"] == "context_length_exceeded"
    assert exact_attempt["retryable"] is False
    assert exact_attempt["usage_unavailable_reason"] == (
        "exact prompt-cache compaction overflowed without provider completion usage"
    )
    assert "usage" not in exact_attempt
    assert "usage_metrics" not in exact_attempt
    assert result.model_completed_payloads[1]["usage_metrics"]["input_tokens"] == 20


def test_prompt_cache_compactor_surfaces_bounded_failure_after_exact_overflow():
    provider = ContextOverflowProvider(
        success_events=[ModelStreamEvent.error("bounded compaction failed")]
    )
    compactor = PromptCacheCompactor(provider=provider)
    session = Session(
        id="sess_prompt_cache_bounded_failure",
        agent_name="assistant",
        provider_name=provider.name,
        model="fake-model",
        invocation=fixture_session_invocation("sess_prompt_cache_bounded_failure"),
    )

    with pytest.raises(RuntimeError, match="bounded compaction failed") as exc_info:
        asyncio.run(
            compactor.compact(
                CompactionRequest(
                    session=session,
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[Message.text("user", "bounded transcript")],
                    context_messages=[Message.text("user", "context projection")],
                    cache_prefix_request=ModelRequest(
                        model="fake-model",
                        messages=[Message.text("user", "cached prefix")],
                    ),
                )
            )
        )

    assert len(provider.requests) == 2
    assert type(exc_info.value) is RuntimeError
    assert str(exc_info.value) == "bounded compaction failed"


def test_prompt_cache_compactor_does_not_loop_when_bounded_attempt_also_overflows():
    provider = ContextOverflowProvider(overflow_requests=2)
    compactor = PromptCacheCompactor(provider=provider)
    session = Session(
        id="sess_prompt_cache_double_overflow",
        agent_name="assistant",
        provider_name=provider.name,
        model="fake-model",
        invocation=fixture_session_invocation("sess_prompt_cache_double_overflow"),
    )

    with pytest.raises(RuntimeError, match="context too large") as exc_info:
        asyncio.run(
            compactor.compact(
                CompactionRequest(
                    session=session,
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[Message.text("user", "bounded transcript")],
                    context_messages=[Message.text("user", "context projection")],
                    cache_prefix_request=ModelRequest(
                        model="fake-model",
                        messages=[Message.text("user", "cached prefix")],
                    ),
                )
            )
        )

    assert len(provider.requests) == 2
    assert isinstance(exc_info.value, ModelContextOverflowError)
    assert exc_info.value.error_code == "context_length_exceeded"


def test_prompt_cache_policy_deduplicates_exact_overflow_and_completes_with_bounded_result():
    overflow = ModelContextOverflowError(
        "context too large",
        provider="fake",
        status_code=400,
        error_code="context_length_exceeded",
    )
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 2}}),
            ],
            [ModelStreamEvent.error(str(overflow), cause=overflow)],
            [
                ModelStreamEvent.text_delta("bounded summary"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 20, "output_tokens": 5}}),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 8, "output_tokens": 2}}),
            ],
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=PromptCacheCompactor(provider=provider),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_prompt_cache_policy_overflow_success",
                messages=[Message.text("user", "first request")],
            ),
        )
    )
    assert first_events[-1].type == EventType.SESSION_COMPLETED

    resume_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_prompt_cache_policy_overflow_success",
                messages=[Message.text("user", "second request")],
            ),
        )
    )

    assert len(provider.requests) == 4
    exact_request, bounded_request = provider.requests[1:3]
    exact_text = "\n".join(
        part.text
        for message in exact_request.messages
        for part in message.content
        if isinstance(part, TextPart)
    )
    assert "first request" in exact_text
    assert "first answer" in exact_text
    assert len(exact_request.messages) > 2
    assert [message.role for message in bounded_request.messages] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
    ]
    assert bounded_request.tools == []

    compaction_attempts = [
        event
        for event in resume_events
        if event.type == EventType.MODEL_COMPLETED
        and event.payload.get("purpose") == "context_compaction"
    ]
    # The exact attempt is both recorded before fallback and returned by the compactor.
    # Its preserved internal attempt id must replace, rather than duplicate, that event.
    assert len(compaction_attempts) == 2
    assert (
        sum(
            event.payload.get("compaction_outcome") == "context_overflow"
            for event in compaction_attempts
        )
        == 1
    )
    assert compaction_attempts[0].payload["compaction_outcome"] == "context_overflow"
    assert "compaction_attempt_id" not in compaction_attempts[0].payload
    assert "usage_metrics" not in compaction_attempts[0].payload
    assert compaction_attempts[1].payload["usage_metrics"]["input_tokens"] == 20
    assert "compaction_attempt_id" not in compaction_attempts[1].payload
    completed = next(
        event for event in resume_events if event.type == EventType.CONTEXT_COMPACTION_COMPLETED
    )
    assert completed.payload["chunk_mode"] == "single_request"
    assert EventType.CONTEXT_COMPACTION_FAILED not in [event.type for event in resume_events]
    assert resume_events[-1].type == EventType.SESSION_COMPLETED
    usage = asyncio.run(app.get_session_usage("sess_prompt_cache_policy_overflow_success"))
    assert usage.model_steps == 4
    assert usage.usage.input_tokens == 38
    assert usage.usage.output_tokens == 9


def test_prompt_cache_compactor_records_exact_overflow_when_bounded_attempt_fails():
    overflow = ModelContextOverflowError(
        "context too large",
        provider="fake",
        status_code=400,
        error_code="context_length_exceeded",
    )
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 2}}),
            ],
            [ModelStreamEvent.error(str(overflow), cause=overflow)],
            [ModelStreamEvent.error("bounded compaction failed")],
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=PromptCacheCompactor(provider=provider),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_prompt_cache_failed_fallback_telemetry",
                messages=[Message.text("user", "first request")],
            ),
        )
    )
    assert first_events[-1].type == EventType.SESSION_COMPLETED

    resume_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_prompt_cache_failed_fallback_telemetry",
                messages=[Message.text("user", "second request")],
            ),
        )
    )

    assert len(provider.requests) == 3
    compaction_events = [
        event
        for event in resume_events
        if event.type
        in {
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.MODEL_COMPLETED,
            EventType.CONTEXT_COMPACTION_FAILED,
        }
    ]
    assert [event.type for event in compaction_events] == [
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.MODEL_COMPLETED,
        EventType.CONTEXT_COMPACTION_FAILED,
    ]
    assert "bounded_input" not in compaction_events[0].payload
    exact_attempt = compaction_events[1]
    assert exact_attempt.payload["purpose"] == "context_compaction"
    assert exact_attempt.payload["compaction_outcome"] == "context_overflow"
    assert exact_attempt.payload["usage_unavailable_reason"] == (
        "exact prompt-cache compaction overflowed without provider completion usage"
    )
    assert "usage_metrics" not in exact_attempt.payload
    bounded_attempt = compaction_events[2]
    assert bounded_attempt.payload["compaction_outcome"] == "unfinished_stream"
    assert bounded_attempt.payload["usage_unavailable_reason"]
    assert "usage_metrics" not in bounded_attempt.payload
    failed_compaction = compaction_events[3]
    assert failed_compaction.payload["error_type"] == "RuntimeError"
    assert "error" not in failed_compaction.payload
    assert failed_compaction.payload["error_type"] == "RuntimeError"
    assert resume_events[-1].type == EventType.SESSION_FAILED
    assert resume_events[-1].payload["session_run_operation_id"] == PRIVATE_EVENT_AUTHORITY
    terminal_records = asyncio.run(
        app.session_store.query_events(
            EventQuery(
                session_id="sess_prompt_cache_failed_fallback_telemetry",
                event_type=EventType.SESSION_FAILED,
            )
        )
    )
    terminal_operation_id = terminal_records[-1].event.payload["session_run_operation_id"]
    assert str(UUID(terminal_operation_id)) == terminal_operation_id
    assert {
        key: value
        for key, value in resume_events[-1].payload.items()
        if key != "session_run_operation_id"
    } == {
        "compaction_failure": _expected_compaction_failure(
            failed_compaction,
            reason="provider_failed",
        ),
        "failure_evidence": _expected_failure_evidence(
            "sess_prompt_cache_failed_fallback_telemetry",
            "RuntimeError",
            "ContextBuildError",
            run_epoch=3,
        ),
        "error": "bounded compaction failed",
        "error_type": "RuntimeError",
    }
    usage = asyncio.run(app.get_session_usage("sess_prompt_cache_failed_fallback_telemetry"))
    assert usage.model_steps == 3
    cost = asyncio.run(
        app.get_session_cost(
            "sess_prompt_cache_failed_fallback_telemetry",
            PriceBook(
                prices=(
                    ModelPrice.fixed(
                        provider_name="fake",
                        model="fake-model",
                        input_per_million=Decimal("1"),
                        output_per_million=Decimal("10"),
                    ),
                )
            ),
        )
    )
    assert cost.model_steps == 3
    assert cost.priced_model_steps == 1
    assert cost.unpriced_model_steps == 2
    assert cost.line_items[0].priced is True
    for line_item in cost.line_items[1:]:
        assert line_item.priced is False
        assert line_item.total_cost == Decimal("0")
        assert line_item.missing_pricing_reason == (
            "model.completed event has no token usage metrics"
        )


def test_prompt_cache_compactor_preserves_attempt_order_after_recovered_invalid_summary():
    class RecoveringPromptCacheCompactor(PromptCacheCompactor):
        async def compact(self, request: CompactionRequest) -> CompactionResult:
            try:
                await ModelCompactor(
                    provider=self.provider,
                    model=request.session.model,
                ).compact(request)
            except ValueError:
                return await super().compact(request)
            raise AssertionError("the first compaction unexpectedly returned a valid summary")

    overflow = ModelContextOverflowError(
        "context too large",
        provider="fake",
        status_code=400,
        error_code="context_length_exceeded",
    )
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 2}}),
            ],
            [ModelStreamEvent.completed({"usage": {"input_tokens": 15, "output_tokens": 0}})],
            [ModelStreamEvent.error(str(overflow), cause=overflow)],
            [
                ModelStreamEvent.text_delta(" \n\t"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 20, "output_tokens": 1}}),
            ],
        ]
    )
    app = CayuApp(request_footprint=RequestFootprintConfig(enabled=False))
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=RecoveringPromptCacheCompactor(provider=provider),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    first_events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_prompt_cache_invalid_bounded_summary",
                messages=[Message.text("user", "first request")],
            ),
        )
    )
    assert first_events[-1].type == EventType.SESSION_COMPLETED

    resume_events = asyncio.run(
        collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_prompt_cache_invalid_bounded_summary",
                messages=[Message.text("user", "second request")],
            ),
        )
    )

    compaction_events = [
        event
        for event in resume_events
        if event.type
        in {
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.MODEL_COMPLETED,
            EventType.CONTEXT_COMPACTION_FAILED,
        }
    ]
    assert [event.type for event in compaction_events] == [
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.MODEL_COMPLETED,
        EventType.MODEL_COMPLETED,
        EventType.CONTEXT_COMPACTION_FAILED,
    ]
    recovered_attempt, exact_attempt, bounded_attempt = compaction_events[1:4]
    assert recovered_attempt.payload["compaction_outcome"] == "invalid_summary"
    assert recovered_attempt.payload["usage_metrics"]["input_tokens"] == 15
    assert exact_attempt.payload["compaction_outcome"] == "context_overflow"
    assert "usage_metrics" not in exact_attempt.payload
    assert bounded_attempt.payload["compaction_outcome"] == "invalid_summary"
    assert bounded_attempt.payload["usage_metrics"]["input_tokens"] == 20
    assert bounded_attempt.payload["usage_metrics"]["output_tokens"] == 1
    assert compaction_events[-1].payload["error_type"] == "ValueError"
    assert "error" not in compaction_events[-1].payload
    assert resume_events[-1].type == EventType.SESSION_FAILED


def test_prompt_cache_compactor_fails_on_provider_error():
    provider = FakeProvider([ModelStreamEvent.error("provider unavailable")])
    compactor = PromptCacheCompactor(provider=provider)

    with pytest.raises(RuntimeError, match="provider unavailable"):
        asyncio.run(
            compactor.compact(
                CompactionRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[Message.text("user", "old")],
                    context_messages=[Message.text("user", "hello")],
                    cache_prefix_request=ModelRequest(
                        model="fake-model",
                        messages=[Message.text("user", "hello")],
                    ),
                )
            )
        )

    assert len(provider.requests) == 1


def test_prompt_cache_compactor_custom_instruction():
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("custom done"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    compactor = PromptCacheCompactor(
        provider=provider,
        compaction_instruction="Make it short.",
    )

    asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[Message.text("user", "old")],
                context_messages=[Message.text("user", "hello")],
                cache_prefix_request=ModelRequest(
                    model="fake-model",
                    messages=[Message.text("user", "hello")],
                ),
            )
        )
    )

    instruction = provider.requests[0].messages[-1].content[0].text
    assert instruction == "Make it short."


def test_prompt_cache_compactor_custom_fallback():
    provider = FakeProvider([ModelStreamEvent.text_delta("should not be called")])
    recording = RecordingCompactor()
    compactor = PromptCacheCompactor(
        provider=provider,
        fallback_compactor=recording,
    )

    result = asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[Message.text("user", "old request")],
            )
        )
    )

    assert len(provider.requests) == 0
    assert len(recording.requests) == 1
    assert "old request" in result.summary


def test_prompt_cache_compactor_merges_cached_request_options_with_overrides():
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    compactor = PromptCacheCompactor(
        provider=provider,
        options={"anthropic": {"max_tokens": 512}, "custom": {"enabled": True}},
    )

    asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(
                    name="assistant",
                    model="fake-model",
                    provider_options={
                        "anthropic": {
                            "cache_control": {"type": "ephemeral", "ttl": "5m"},
                            "max_tokens": 2048,
                        },
                        "openai": {"prompt_cache_key": "agent-cache"},
                    },
                ),
                messages=[Message.text("user", "old")],
                context_messages=[Message.text("user", "hello")],
                cache_prefix_request=ModelRequest(
                    model="fake-model",
                    messages=[Message.text("user", "hello")],
                    options={
                        "anthropic": {
                            "cache_control": {"type": "ephemeral", "ttl": "5m"},
                            "max_tokens": 2048,
                        },
                        "openai": {"prompt_cache_key": "agent-cache"},
                    },
                ),
            )
        )
    )

    assert provider.requests[0].options == {
        "anthropic": {
            "cache_control": {"type": "ephemeral", "ttl": "5m"},
            "max_tokens": 512,
        },
        "openai": {"prompt_cache_key": "agent-cache"},
        "custom": {"enabled": True},
    }


def test_prompt_cache_compactor_rejects_invalid_provider():
    with pytest.raises(TypeError, match="provider must be a ModelProvider"):
        PromptCacheCompactor(provider="not a provider")


def test_prompt_cache_compactor_rejects_invalid_fallback():
    provider = FakeProvider([])
    with pytest.raises(TypeError, match="fallback_compactor must be a ContextCompactor"):
        PromptCacheCompactor(provider=provider, fallback_compactor="not a compactor")
