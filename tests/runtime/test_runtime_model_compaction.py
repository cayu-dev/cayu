"""Model compaction contracts."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tests._session_provenance import fixture_session_invocation
from tests.runtime.runtime_compaction_support import (
    CompletedThenRetryableCompactionProvider,
    FlakyCompactionProvider,
)
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_session_support import (
    _private_record_for_public_event,
    _test_session,
    collect_events,
)

import cayu.context.base as runtime_context_module
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.artifacts.attachments import file_attachment
from cayu.budgets.base import BudgetWindow, InMemoryBudgetStore
from cayu.context.base import (
    CheckpointCompactionContextPolicy,
    CompactionPrompt,
    CompactionRequest,
    CompactionResult,
    ContextBuildError,
    ContextRequest,
    ModelCompactor,
    PromptCacheCompactor,
)
from cayu.events import EventType
from cayu.messages import FilePart, Message, MessageRole, ProviderStatePart, TextPart
from cayu.providers.base import (
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelStreamEvent,
    ModelStreamEventType,
)
from cayu.providers.retry_policy import RetryPolicy
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.records import Session
from cayu.sessions.requests import RunRequest


def test_model_compactor_ignores_thinking_events():
    # A reasoning-capable compaction provider may emit THINKING events; compaction must
    # ignore them (the summary is built from text) rather than crash on an unknown type.
    provider = FakeProvider(
        [
            ModelStreamEvent.thinking("internal reasoning", provider_state={"signature": "S"}),
            ModelStreamEvent.text_delta("summary "),
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"usage": {"output_tokens": 2}}),
        ]
    )
    compactor = ModelCompactor(provider=provider, model="summary-model")

    result = asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[Message.text("user", "old request")],
            )
        )
    )

    assert result.summary == "summary done"


def test_model_compactor_summarizes_context_with_provider():
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("summary "),
            ModelStreamEvent.text_delta("done"),
            ModelStreamEvent.completed({"usage": {"output_tokens": 2}}),
        ]
    )
    compactor = ModelCompactor(
        provider=provider,
        model="summary-model",
        options={"anthropic": {"max_tokens": 512}},
    )

    result = asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[
                    Message.text("user", "old request"),
                    Message.text("assistant", "old answer"),
                ],
                existing_summary="previous context",
            )
        )
    )

    assert result.summary == "summary done"
    assert result.metadata == {
        "compactor": "ModelCompactor",
        "provider": "fake",
        "model": "summary-model",
        "input_truncated": False,
        "max_input_chars": 120000,
        "completed": {"usage": {"output_tokens": 2}},
    }
    assert len(provider.requests) == 1
    request = provider.requests[0]
    assert request.model == "summary-model"
    assert request.tools == []
    assert request.options == {"anthropic": {"max_tokens": 512}}
    assert [message.role for message in request.messages] == ["system", "user"]
    prompt = request.messages[1].content[0].text
    assert "Existing summary:\nprevious context" in prompt
    assert "user: old request" in prompt
    assert "assistant: old answer" in prompt


def test_model_compactor_reports_model_completed_payload_with_usage_metrics():
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("summary"),
            ModelStreamEvent.completed(
                {
                    "model": "summary-model-2026-01-01",
                    "usage": {"input_tokens": 40, "output_tokens": 8},
                    "provider_state": {"opaque": True},
                }
            ),
        ]
    )
    compactor = ModelCompactor(provider=provider, model="summary-model")

    result = asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[Message.text("user", "old request")],
            )
        )
    )

    assert result.model_completed_payloads == [
        {
            "model": "summary-model-2026-01-01",
            "usage": {"input_tokens": 40, "output_tokens": 8},
            "provider_name": "fake",
            "requested_model": "summary-model",
            "purpose": "context_compaction",
            "compactor": "ModelCompactor",
            "usage_metrics": {
                "provider_name": "fake",
                "requested_model": "summary-model",
                "model": "summary-model-2026-01-01",
                "input_tokens": 40,
                "output_tokens": 8,
                "total_tokens": 48,
                "reasoning_output_tokens": 0,
                "cache": {
                    "read_tokens": 0,
                    "write_tokens": 0,
                    "cached_input_tokens": 0,
                    "uncached_input_tokens": 40,
                },
            },
        }
    ]


def test_model_compactor_retries_transient_provider_errors():
    provider = FlakyCompactionProvider(
        failures=1,
        error=ModelProviderError(
            "provider overloaded",
            provider="flaky",
            status_code=503,
            retryable=True,
        ),
    )
    compactor = ModelCompactor(
        provider=provider,
        model="summary-model",
        retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
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

    assert result.summary == "recovered summary"
    assert len(provider.requests) == 2


def test_model_compactor_retries_transient_provider_error_events():
    typed_error = ModelProviderError(
        "provider overloaded",
        provider="fake",
        status_code=529,
        retryable=True,
        retry_after_s=0.0,
    )
    provider = FakeProvider(
        [
            [ModelStreamEvent.error("provider overloaded", cause=typed_error)],
            [
                ModelStreamEvent.text_delta("recovered summary"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    compactor = ModelCompactor(
        provider=provider,
        model="summary-model",
        retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
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

    assert result.summary == "recovered summary"
    assert len(provider.requests) == 2


def test_model_compactor_reports_every_completion_when_retry_follows_completed():
    provider = CompletedThenRetryableCompactionProvider()
    result = asyncio.run(
        ModelCompactor(
            provider=provider,
            model="summary-model",
            retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
        ).compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[Message.text("user", "old request")],
            )
        )
    )

    assert result.summary == "summary 2"
    assert len(provider.requests) == 2
    assert [
        payload["usage_metrics"]["input_tokens"] for payload in result.model_completed_payloads
    ] == [1, 2]
    assert all(
        "compaction_attempt_id" not in payload for payload in result.model_completed_payloads
    )


def test_prompt_cache_compactor_reports_every_completion_when_retry_follows_completed():
    provider = CompletedThenRetryableCompactionProvider()
    context_messages = [
        Message.text("user", "cached request"),
        Message.text("assistant", "cached answer"),
    ]
    result = asyncio.run(
        PromptCacheCompactor(
            provider=provider,
            retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
        ).compact(
            CompactionRequest(
                session=Session(
                    id="sess_prompt_cache_retry_after_completed",
                    agent_name="assistant",
                    provider_name=provider.name,
                    model="fake-model",
                    invocation=fixture_session_invocation(
                        "sess_prompt_cache_retry_after_completed"
                    ),
                ),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[Message.text("user", "old request")],
                context_messages=context_messages,
                cache_prefix_request=ModelRequest(
                    model="fake-model",
                    messages=context_messages,
                ),
            )
        )
    )

    assert result.summary == "summary 2"
    assert len(provider.requests) == 2
    assert [
        payload["usage_metrics"]["input_tokens"] for payload in result.model_completed_payloads
    ] == [1, 2]
    assert all(
        "compaction_attempt_id" not in payload for payload in result.model_completed_payloads
    )


def test_runtime_compaction_outer_ledger_does_not_duplicate_retry_completions():
    compaction_provider = CompletedThenRetryableCompactionProvider()
    runtime_provider = FakeProvider([ModelStreamEvent.completed({})])
    budget_store = InMemoryBudgetStore()
    app = CayuApp(budget_store=budget_store)
    app.register_provider(runtime_provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        context_policy=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(
                provider=compaction_provider,
                model="summary-model",
                retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
            ),
            max_user_turns=1,
            compact_after_messages=2,
        ),
    )

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_compaction_completed_retry",
                messages=[
                    Message.text("user", "old"),
                    Message.text("assistant", "old answer"),
                    Message.text("user", "current"),
                ],
            ),
        )
    )

    completions = [
        event
        for event in events
        if event.type == EventType.MODEL_COMPLETED
        and event.payload.get("purpose") == "context_compaction"
    ]
    assert len(compaction_provider.requests) == 2
    assert [event.payload["usage_metrics"]["input_tokens"] for event in completions] == [1, 2]
    assert all("compaction_attempt_id" not in event.payload for event in completions)
    budget_events = asyncio.run(
        budget_store.load_events_for_budget(
            scope="app",
            key=None,
            window=BudgetWindow.all_time(),
        )
    )
    session_records = asyncio.run(
        app.session_store.query_events(
            EventQuery(session_id="sess_compaction_completed_retry", limit=100)
        )
    )
    assert [
        event.id for event in budget_events if event.payload.get("purpose") == "context_compaction"
    ] == [
        _private_record_for_public_event(session_records, event).event.id for event in completions
    ]


def test_model_compactor_can_disable_retries_explicitly():
    provider = FlakyCompactionProvider(
        failures=1,
        error=ModelProviderError(
            "provider overloaded",
            provider="flaky",
            status_code=503,
            retryable=True,
        ),
    )
    compactor = ModelCompactor(
        provider=provider,
        model="summary-model",
        retry_policy=RetryPolicy(max_attempts=1),
    )

    with pytest.raises(ModelProviderError, match="provider overloaded"):
        asyncio.run(
            compactor.compact(
                CompactionRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[Message.text("user", "old request")],
                )
            )
        )
    assert len(provider.requests) == 1


def test_model_compactor_does_not_retry_non_retryable_provider_error_events():
    typed_error = ModelProviderError(
        "request timed out",
        provider="fake",
        status_code=503,
        retryable=False,
        retry_after_s=2.5,
    )
    provider = FakeProvider([[ModelStreamEvent.error("request timed out", cause=typed_error)]])
    compactor = ModelCompactor(
        provider=provider,
        model="summary-model",
        retry_policy=RetryPolicy(max_attempts=3, initial_delay_s=0.0),
    )

    with pytest.raises(ModelProviderError, match="request timed out") as exc_info:
        asyncio.run(
            compactor.compact(
                CompactionRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[Message.text("user", "old request")],
                )
            )
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.retryable is False
    assert exc_info.value.retry_after_s == 2.5
    assert len(provider.requests) == 1


def test_model_compactor_keeps_provider_only_error_events_untyped():
    provider = FakeProvider(
        [
            [
                ModelStreamEvent(
                    type=ModelStreamEventType.ERROR,
                    payload={"error": "provider unavailable", "provider": "fake"},
                )
            ],
            [
                ModelStreamEvent.text_delta("should not retry"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    compactor = ModelCompactor(
        provider=provider,
        model="summary-model",
        retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
    )

    with pytest.raises(RuntimeError, match="provider unavailable"):
        asyncio.run(
            compactor.compact(
                CompactionRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[Message.text("user", "old request")],
                )
            )
        )

    assert len(provider.requests) == 1


def test_model_compactor_does_not_retry_terminal_provider_errors():
    provider = FlakyCompactionProvider(
        failures=1,
        error=ModelProviderError(
            "invalid request",
            provider="flaky",
            status_code=400,
            retryable=False,
        ),
    )
    compactor = ModelCompactor(
        provider=provider,
        model="summary-model",
        retry_policy=RetryPolicy(max_attempts=3, initial_delay_s=0.0),
    )

    with pytest.raises(ModelProviderError, match="invalid request"):
        asyncio.run(
            compactor.compact(
                CompactionRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[Message.text("user", "old request")],
                )
            )
        )
    assert len(provider.requests) == 1


def test_model_compactor_rejects_tool_calls_from_compaction_model():
    provider = FakeProvider(
        [
            ModelStreamEvent.tool_call(
                id="call_1",
                name="echo",
                arguments={"text": "bad"},
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    )
    compactor = ModelCompactor(provider=provider, model="summary-model")

    with pytest.raises(RuntimeError, match="must not call tools"):
        asyncio.run(
            compactor.compact(
                CompactionRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[Message.text("user", "old request")],
                )
            )
        )


def test_model_compactor_fails_on_provider_error():
    provider = FakeProvider([ModelStreamEvent.error("provider unavailable")])
    compactor = ModelCompactor(provider=provider, model="summary-model")

    with pytest.raises(RuntimeError, match="provider unavailable"):
        asyncio.run(
            compactor.compact(
                CompactionRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[Message.text("user", "old request")],
                )
            )
        )


def test_model_compactor_bounds_large_compaction_input():
    completed_batch = [
        ModelStreamEvent.text_delta("bounded summary"),
        ModelStreamEvent.completed({"finish_reason": "stop"}),
    ]
    provider = FakeProvider([completed_batch for _index in range(16)])
    compactor = ModelCompactor(
        provider=provider,
        model="summary-model",
        max_input_chars=1000,
    )

    result = asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[Message.text("user", "x" * 5000)],
                existing_summary="important previous summary",
            )
        )
    )

    assert len(provider.requests) > 1
    assert all(len(request.messages[1].content[0].text) <= 1000 for request in provider.requests)
    assert "x" * 100 in provider.requests[0].messages[1].content[0].text
    assert any(
        "important previous summary" in request.messages[1].content[0].text
        for request in provider.requests
    )
    assert result.covered_message_count == 1
    assert result.source_chunk_count > 1
    assert result.source_chunk_mode == "hierarchical_atomic_unit"
    assert result.bounded_input is True
    assert result.metadata["input_truncated"] is True
    assert result.metadata["max_input_chars"] == 1000


def test_model_compactor_bounds_by_complete_message_prefix_and_reports_coverage():
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("bounded summary"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    compactor = ModelCompactor(
        provider=provider,
        model="summary-model",
        max_input_chars=1000,
    )
    messages = [
        Message.text("user", "OLDEST_MUST_SURVIVE " + "x" * 400),
        Message.text("assistant", "LATEST_COMPACTABLE " + "x" * 5000),
    ]

    result = asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=messages,
            )
        )
    )

    prompt = provider.requests[0].messages[1].content[0].text
    assert result.covered_message_count == 1
    assert "OLDEST_MUST_SURVIVE" in prompt
    assert "LATEST_COMPACTABLE" not in prompt


def test_compaction_result_rejects_missing_or_negative_coverage():
    with pytest.raises(ValueError):
        CompactionResult(summary="summary")
    with pytest.raises(ValueError):
        CompactionResult(summary="summary", covered_message_count=-1)
    with pytest.raises(ValueError):
        CompactionPrompt(prompt="custom prompt", covered_message_count=0)
    with pytest.raises(ValueError):
        CompactionResult(
            summary="summary",
            covered_message_count=0,
            source_chunk_count=-1,
        )
    with pytest.raises(ValueError):
        CompactionResult(
            summary="summary",
            covered_message_count=0,
            source_chunk_mode=" ",
        )
    for invalid_sha256 in ("abc", "A" * 64, "g" * 64):
        with pytest.raises(ValueError, match="lowercase SHA-256"):
            CompactionResult(
                summary="summary",
                covered_message_count=1,
                represented_existing_summary_sha256=invalid_sha256,
            )
    for invalid_mode in ("private\nmode", "x" * 513):
        with pytest.raises(ValueError):
            CompactionResult(
                summary="summary",
                covered_message_count=0,
                source_chunk_mode=invalid_mode,
            )
    for invalid_progress_key in ("private\nkey", "x" * 513):
        with pytest.raises(ValueError):
            CompactionResult(
                summary="summary",
                covered_message_count=0,
                progress_exhausted=True,
                progress_key=invalid_progress_key,
            )
    with pytest.raises(ValueError, match="requires progress_key"):
        CompactionResult(
            summary="summary",
            covered_message_count=0,
            progress_exhausted=True,
        )
    with pytest.raises(ValueError, match="zero coverage"):
        CompactionResult(
            summary="summary",
            covered_message_count=1,
            progress_exhausted=True,
            progress_key="test:v1",
        )
    for invalid_count in (True, "1", 1.0):
        with pytest.raises(ValueError):
            CompactionResult(
                summary="summary",
                covered_message_count=invalid_count,
            )
        with pytest.raises(ValueError):
            CompactionPrompt(
                prompt="custom prompt",
                covered_message_count=invalid_count,
            )


@pytest.mark.parametrize("second_size", [100, 600, 1200, 5000])
def test_model_compactor_coverage_never_crosses_a_message_boundary(second_size):
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("summary"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    messages = [
        Message.text("user", "first-marker"),
        Message.text("assistant", "second-marker " + "x" * second_size),
    ]
    result = asyncio.run(
        ModelCompactor(
            provider=provider,
            model="summary-model",
            max_input_chars=1000,
        ).compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=messages,
            )
        )
    )
    prompt = provider.requests[0].messages[-1].content[0].text
    assert result.covered_message_count in {1, 2}
    assert "first-marker" in prompt
    assert ("second-marker" in prompt) is (result.covered_message_count == 2)


@settings(max_examples=40, deadline=None)
@given(
    source_shapes=st.lists(
        st.tuples(
            st.one_of(
                st.sampled_from([0, 1, 700, 799, 800, 801, 999, 1000, 1001, 2000, 4000]),
                st.integers(min_value=0, max_value=4000),
            ),
            st.booleans(),
        ),
        min_size=1,
        max_size=5,
    )
)
def test_model_compaction_property_preserves_every_covered_source_marker(
    source_shapes: list[tuple[int, bool]],
) -> None:
    markers = [f"CAYU_COVERAGE_MARKER_{index:02d}" for index in range(len(source_shapes))]

    class MarkerPreservingProvider(ModelProvider):
        name = "property-marker-preserving"

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            prompt = request.messages[-1].content[0].text
            represented = [marker for marker in markers if marker in prompt]
            yield ModelStreamEvent.text_delta(" ".join(represented) or "bounded fragment")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    source_messages = [
        Message.text(
            MessageRole.USER if index % 2 == 0 else MessageRole.ASSISTANT,
            marker + " " + (("😀é" * ((size + 1) // 2))[:size] if unicode_filler else "x" * size),
        )
        for index, (marker, (size, unicode_filler)) in enumerate(
            zip(
                markers,
                source_shapes,
                strict=True,
            )
        )
    ]
    messages = [*source_messages, Message.text(MessageRole.USER, "current turn")]
    original_messages = [message.model_copy(deep=True) for message in messages]
    policy = CheckpointCompactionContextPolicy(
        compactor=ModelCompactor(
            provider=MarkerPreservingProvider(),
            model="summary-model",
            max_input_chars=1000,
        ),
        max_user_turns=1,
        compact_after_messages=1,
    )

    checkpoint = None
    previous_cursor = 0
    for _attempt in range(len(source_messages)):
        result = asyncio.run(
            policy.build_with_checkpoint(
                ContextRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=messages,
                    step=1,
                    force_compaction=True,
                ),
                checkpoint=checkpoint,
            )
        )
        assert result.checkpoint is not None
        checkpoint = result.checkpoint
        compaction_checkpoint = checkpoint["context_compaction"]
        cursor = compaction_checkpoint["compacted_transcript_cursor"]
        assert previous_cursor < cursor <= len(source_messages)
        assert compaction_checkpoint["summary"].split().count(markers[cursor - 1]) == 1

        projection = json.dumps(
            [message.model_dump(mode="json") for message in result.messages],
            ensure_ascii=False,
        )
        summary = compaction_checkpoint["summary"]
        for index, marker in enumerate(markers):
            assert projection.count(marker) == 1
            assert summary.split().count(marker) == (1 if index < cursor else 0)

        assert messages == original_messages
        previous_cursor = cursor
        if cursor == len(source_messages):
            break

    assert previous_cursor == len(source_messages)


def test_bounded_model_compactor_finds_largest_prefix_with_logarithmic_rebuilds(
    monkeypatch,
):
    messages = [
        Message.text(
            "user" if index % 2 == 0 else "assistant",
            f"message-{index:04d} " + "x" * 20,
        )
        for index in range(1024)
    ]
    request = CompactionRequest(
        session=_test_session(),
        agent=AgentSpec(name="assistant", model="fake-model"),
        messages=messages,
    )
    original = runtime_context_module._default_compaction_prompt_parts
    rebuild_count = 0

    def counted_prompt_parts(candidate):
        nonlocal rebuild_count
        rebuild_count += 1
        return original(candidate)

    monkeypatch.setattr(
        runtime_context_module,
        "_default_compaction_prompt_parts",
        counted_prompt_parts,
    )
    prompt, truncated, covered = runtime_context_module._bounded_default_compaction_prompt(
        request,
        max_chars=4000,
    )

    assert prompt is not None
    assert truncated is True
    assert 0 < covered < len(messages)
    assert len(prompt) <= 4000
    assert rebuild_count <= 12


def test_model_compactor_never_splits_tool_call_from_result_at_input_bound():
    completed_batch = [
        ModelStreamEvent.text_delta("summary"),
        ModelStreamEvent.completed({"finish_reason": "stop"}),
    ]
    tool_round = [
        Message.tool_call(
            tool_call_id="call_atomic",
            tool_name="inspect",
            arguments={"query": "TOOL_CALL_MARKER " + "x" * 1500},
        ),
        Message.tool_result(
            tool_call_id="call_atomic",
            tool_name="inspect",
            content="TOOL_RESULT_MARKER",
        ),
    ]

    prefix_provider = FakeProvider(completed_batch)
    prefix_result = asyncio.run(
        ModelCompactor(
            provider=prefix_provider,
            model="summary-model",
            max_input_chars=1000,
        ).compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[Message.text("user", "safe-prefix"), *tool_round],
            )
        )
    )
    prefix_prompt = prefix_provider.requests[0].messages[-1].content[0].text
    assert prefix_result.covered_message_count == 1
    assert "safe-prefix" in prefix_prompt
    assert "TOOL_CALL_MARKER" not in prefix_prompt
    assert "TOOL_RESULT_MARKER" not in prefix_prompt

    hierarchy_provider = FakeProvider([completed_batch for _index in range(16)])
    hierarchy_result = asyncio.run(
        ModelCompactor(
            provider=hierarchy_provider,
            model="summary-model",
            max_input_chars=1000,
        ).compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=tool_round,
            )
        )
    )
    hierarchy_prompts = [
        request.messages[-1].content[0].text for request in hierarchy_provider.requests
    ]
    assert hierarchy_result.covered_message_count == 2
    assert hierarchy_result.source_chunk_mode == "hierarchical_atomic_unit"
    assert any("TOOL_CALL_MARKER" in prompt for prompt in hierarchy_prompts)
    assert any("TOOL_RESULT_MARKER" in prompt for prompt in hierarchy_prompts)


@pytest.mark.parametrize(
    ("message", "marker"),
    [
        (
            Message(
                role=MessageRole.USER,
                content=(
                    TextPart(text="UNICODE_BOUNDARY_😀 " + "x" * 2000),
                    FilePart(
                        attachment=file_attachment(
                            artifact_id="HIERARCHY_FILE_MARKER",
                            kind="image",
                            filename="hierarchy.png",
                            content_type="image/png",
                            size_bytes=5,
                        )
                    ),
                ),
            ),
            "HIERARCHY_FILE_MARKER",
        ),
        (
            Message(
                role=MessageRole.ASSISTANT,
                content=(
                    TextPart(text="x" * 2000),
                    ProviderStatePart(
                        provider="fake",
                        state={"opaque": "HIERARCHY_PROVIDER_STATE"},
                    ),
                ),
            ),
            "provider_state provider=fake",
        ),
    ],
    ids=["file_and_unicode", "provider_state"],
)
def test_hierarchical_compaction_represents_boundary_message_parts(message, marker):
    completed_batch = [
        ModelStreamEvent.text_delta("summary"),
        ModelStreamEvent.completed({"finish_reason": "stop"}),
    ]
    provider = FakeProvider([completed_batch for _index in range(16)])
    result = asyncio.run(
        ModelCompactor(
            provider=provider,
            model="summary-model",
            max_input_chars=1000,
        ).compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[message],
            )
        )
    )
    prompts = [request.messages[-1].content[0].text for request in provider.requests]
    assert result.covered_message_count == 1
    assert result.source_chunk_mode == "hierarchical_atomic_unit"
    assert any(marker in prompt for prompt in prompts)
    if message.role == MessageRole.USER:
        assert any("UNICODE_BOUNDARY_😀" in prompt for prompt in prompts)


def test_hierarchical_compaction_cancellation_never_returns_partial_coverage():
    class CancelSecondHierarchyCallProvider(ModelProvider):
        name = "cancel-hierarchy"

        def __init__(self) -> None:
            self.calls = 0
            self.second_started = asyncio.Event()

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.calls += 1
            if self.calls == 2:
                self.second_started.set()
                await asyncio.Event().wait()
                raise AssertionError("unreachable")
            yield ModelStreamEvent.text_delta("first fragment summary")
            yield ModelStreamEvent.completed({"usage": {"input_tokens": 5, "output_tokens": 2}})

    async def run() -> None:
        provider = CancelSecondHierarchyCallProvider()
        task = asyncio.create_task(
            ModelCompactor(
                provider=provider,
                model="summary-model",
                max_input_chars=1000,
            ).compact(
                CompactionRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[Message.text("user", "oversized " + "😀" * 3000)],
                )
            )
        )
        await provider.second_started.wait()
        task.cancel("cancel hierarchy")
        with pytest.raises(asyncio.CancelledError, match="cancel hierarchy"):
            await task
        assert provider.calls == 2

    asyncio.run(run())


def test_hierarchical_compaction_call_limit_fails_before_unbounded_dispatch():
    provider = FakeProvider([])
    compactor = ModelCompactor(
        provider=provider,
        model="summary-model",
        max_input_chars=1000,
        max_hierarchy_calls=2,
    )
    with pytest.raises(ValueError, match="max_hierarchy_calls before dispatch"):
        asyncio.run(
            compactor.compact(
                CompactionRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[Message.text("user", "oversized " + "x" * 5000)],
                )
            )
        )
    assert provider.requests == []


def test_hierarchical_compaction_retry_consumes_the_actual_dispatch_limit():
    class RetryThenCompleteProvider(ModelProvider):
        name = "retry-then-complete"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            if len(self.requests) == 1:
                raise ModelProviderError(
                    "retry first hierarchy dispatch",
                    provider=self.name,
                    status_code=503,
                    retryable=True,
                )
            yield ModelStreamEvent.text_delta("first fragment summary")
            yield ModelStreamEvent.completed(
                {
                    "model": request.model,
                    "usage": {"input_tokens": 5, "output_tokens": 2},
                }
            )

    provider = RetryThenCompleteProvider()
    compactor = ModelCompactor(
        provider=provider,
        model="summary-model",
        max_input_chars=1000,
        max_hierarchy_calls=3,
        retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
    )

    with pytest.raises(ValueError, match="exceeded max_hierarchy_calls"):
        asyncio.run(
            compactor.compact(
                CompactionRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[Message.text("user", "oversized " + "x" * 1100)],
                )
            )
        )

    # The retry is a real provider dispatch. The final logical hierarchy call is
    # rejected before it can become an uncounted fourth request.
    assert len(provider.requests) == 3


def test_hierarchical_compaction_preserves_instructions_within_every_bounded_prompt():
    completed_batch = [
        ModelStreamEvent.text_delta("short partial summary"),
        ModelStreamEvent.completed({"finish_reason": "stop"}),
    ]
    provider = FakeProvider([completed_batch for _index in range(16)])
    instruction = "INSTRUCTION_MARKER: retain exact identifiers."
    result = asyncio.run(
        ModelCompactor(
            provider=provider,
            model="summary-model",
            max_input_chars=1000,
        ).compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[Message.text("user", "oversized " + "x" * 5000)],
                instructions=instruction,
            )
        )
    )
    prompts = [request.messages[-1].content[0].text for request in provider.requests]
    assert result.covered_message_count == 1
    assert len(prompts) > 1
    assert all(instruction in prompt for prompt in prompts)
    assert all(len(prompt) <= 1000 for prompt in prompts)


def test_hierarchical_compaction_fails_closed_when_instructions_leave_no_source_capacity():
    provider = FakeProvider([])
    with pytest.raises(
        ValueError,
        match="max_input_chars is too small for hierarchical compaction",
    ):
        asyncio.run(
            ModelCompactor(
                provider=provider,
                model="summary-model",
                max_input_chars=1000,
            ).compact(
                CompactionRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[Message.text("user", "oversized " + "x" * 5000)],
                    instructions="i" * 1000,
                )
            )
        )
    assert provider.requests == []


def test_hierarchical_compaction_validates_merge_capacity_before_dispatch():
    provider = FakeProvider([])
    with pytest.raises(
        ValueError,
        match="max_input_chars is too small for hierarchical assembly",
    ):
        asyncio.run(
            ModelCompactor(
                provider=provider,
                model="summary-model",
                max_input_chars=1000,
            ).compact(
                CompactionRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[Message.text("user", "x" * 100)],
                    instructions="i" * 700,
                )
            )
        )
    assert provider.requests == []


def test_hierarchical_compaction_preflights_existing_summary_call_capacity():
    provider = FakeProvider([])
    with pytest.raises(
        ValueError,
        match="max_hierarchy_calls before dispatch",
    ):
        asyncio.run(
            ModelCompactor(
                provider=provider,
                model="summary-model",
                max_input_chars=1000,
                max_hierarchy_calls=2,
            ).compact(
                CompactionRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[Message.text("user", "small new source")],
                    existing_summary="prior " + "x" * 5000,
                )
            )
        )
    assert provider.requests == []


def test_hierarchical_compaction_preflights_every_existing_summary_merge_level():
    provider = FakeProvider([])
    with pytest.raises(
        ValueError,
        match="max_hierarchy_calls before dispatch",
    ):
        asyncio.run(
            ModelCompactor(
                provider=provider,
                model="summary-model",
                max_input_chars=1000,
                # The first merge level needs 70 calls. One source call plus
                # one assumed final call used to make 72 appear sufficient,
                # even though the optimistic second level still needs two
                # calls before the actual final merge.
                max_hierarchy_calls=72,
            ).compact(
                CompactionRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[Message.text("user", "small new source")],
                    existing_summary="prior " + "x" * 50_000,
                )
            )
        )
    assert provider.requests == []


def test_hierarchical_compaction_merges_existing_summary_exactly_once():
    class MarkerPreservingProvider(ModelProvider):
        name = "marker-preserving"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            prompt = request.messages[-1].content[0].text
            markers = [
                marker for marker in ("PRIOR_RANGE_MARKER", "NEW_RANGE_MARKER") if marker in prompt
            ]
            yield ModelStreamEvent.text_delta(" ".join(markers) or "fragment")
            yield ModelStreamEvent.completed({})

    provider = MarkerPreservingProvider()
    result = asyncio.run(
        ModelCompactor(
            provider=provider,
            model="summary-model",
            max_input_chars=1000,
        ).compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[
                    Message.text("user", "NEW_RANGE_MARKER " + "x" * 4000),
                    Message.text("assistant", "unacknowledged suffix"),
                ],
                existing_summary="PRIOR_RANGE_MARKER",
            )
        )
    )
    assert result.covered_message_count == 1
    assert result.summary.count("PRIOR_RANGE_MARKER") == 1
    assert result.summary.count("NEW_RANGE_MARKER") == 1
    assert all(
        "unacknowledged suffix" not in request.messages[-1].content[0].text
        for request in provider.requests
    )


def test_hierarchical_compaction_allows_initial_expansion_then_current_pass_shrinkage():
    class ExpandThenShrinkProvider(ModelProvider):
        name = "expand-then-shrink"

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.calls += 1
            prompt = request.messages[-1].content[0].text
            if prompt.startswith("Summarize this ordered fragment"):
                summary = "L" * 1500
            elif "L" * 100 in prompt:
                summary = "M" * 600
            elif "M" * 100 in prompt:
                summary = "N" * 300
            elif "N" * 100 in prompt:
                summary = "reduced"
            else:
                summary = "final"
            yield ModelStreamEvent.text_delta(summary)
            yield ModelStreamEvent.completed({})

    provider = ExpandThenShrinkProvider()
    result = asyncio.run(
        ModelCompactor(
            provider=provider,
            model="summary-model",
            max_input_chars=1000,
        ).compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[Message.text("user", "oversized " + "x" * 1500)],
            )
        )
    )

    assert result.covered_message_count == 1
    assert result.summary == "final"
    assert provider.calls > result.source_chunk_count


def test_hierarchical_failure_reports_completed_chunks_without_checkpoint():
    class FailSecondHierarchyCallProvider(ModelProvider):
        name = "fail-hierarchy"

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.calls += 1
            if self.calls == 2:
                yield ModelStreamEvent.error("second hierarchy fragment failed")
                return
            yield ModelStreamEvent.text_delta("first fragment summary")
            yield ModelStreamEvent.completed({"usage": {"input_tokens": 5, "output_tokens": 2}})

    provider = FailSecondHierarchyCallProvider()
    policy = CheckpointCompactionContextPolicy(
        compactor=ModelCompactor(
            provider=provider,
            model="summary-model",
            max_input_chars=1000,
        ),
        max_user_turns=1,
        compact_after_messages=1,
    )
    with pytest.raises(ContextBuildError, match="second hierarchy fragment failed") as exc_info:
        asyncio.run(
            policy.build_with_checkpoint(
                ContextRequest(
                    session=_test_session(),
                    agent=AgentSpec(name="assistant", model="fake-model"),
                    messages=[
                        Message.text("user", "oversized " + "x" * 5000),
                        Message.text("user", "current"),
                    ],
                    step=1,
                    force_compaction=True,
                ),
                checkpoint=None,
            )
        )
    event_types = [item.event_type for item in exc_info.value.compaction_telemetry]
    assert event_types == [
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.MODEL_COMPLETED,
        EventType.CONTEXT_COMPACTION_FAILED,
    ]
    assert exc_info.value.compaction_telemetry[0].payload["bounded_input"] is True
    uncertain = exc_info.value.compaction_telemetry[2].payload
    assert uncertain["compaction_outcome"] == "unfinished_stream"
    assert uncertain["usage_unavailable_reason"]
    failure = exc_info.value.compaction_telemetry[-1].payload
    assert failure["represented_message_count"] == 0
    assert failure["chunk_count"] == 2
    assert failure["chunk_mode"] == "failed"
    assert failure["bounded_input"] is True


def test_model_compactor_uses_bounded_default_prompt():
    provider = FakeProvider(
        [
            ModelStreamEvent.text_delta("default summary"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    compactor = ModelCompactor(
        provider=provider,
        model="summary-model",
        max_input_chars=1000,
    )

    result = asyncio.run(
        compactor.compact(
            CompactionRequest(
                session=_test_session(),
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[
                    Message.text("user", "old request"),
                    Message.text("user", "uncovered marker " + "x" * 2000),
                ],
            )
        )
    )

    assert result.summary == "default summary"
    assert result.covered_message_count == 1
    prompt = provider.requests[0].messages[1].content[0].text
    assert "Transcript to compact:\nuser: old request" in prompt
    assert "uncovered marker" not in prompt
