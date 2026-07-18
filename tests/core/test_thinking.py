from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
from pydantic import ValidationError

from cayu import (
    AgentSpec,
    CayuApp,
    Message,
    RunRequest,
    ThinkingConfig,
    ThinkingPart,
)
from cayu.core.events import Event, EventType
from cayu.core.messages import MessageRole, ProviderStatePart, TextPart, copy_message_part
from cayu.core.thinking import thinking_config_payload
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent, ModelStreamEventType
from cayu.providers.anthropic import (
    _anthropic_message,
    anthropic_response_events,
    build_anthropic_payload,
)
from cayu.providers.base import ModelCompletion, ModelFinishReason
from cayu.providers.chat_completions import (
    _assistant_message as chat_assistant_message,
)
from cayu.providers.chat_completions import (
    build_chat_completions_payload,
    chat_completions_stream_events,
)
from cayu.providers.openai import (
    _openai_input_items,
    build_openai_payload,
    openai_response_events,
)
from cayu.runtime._transcript import AssistantThinkingPart, _materialize_thinking
from cayu.runtime.model_steps import (
    AssistantStepResult,
    StepClassificationType,
    classify_assistant_step,
)
from cayu.runtime.sessions import (
    TranscriptQuery,
    TranscriptRecord,
    filter_transcript_records,
)
from cayu.runtime.usage import normalize_usage_metrics


# --------------------------------------------------------------------------- #
# ThinkingConfig + provider mapping (field-driven)
# --------------------------------------------------------------------------- #
def test_thinking_config_rejects_budget_below_minimum() -> None:
    with pytest.raises(ValidationError):
        ThinkingConfig(max_tokens=512)
    assert ThinkingConfig(max_tokens=1024).max_tokens == 1024


def test_thinking_config_rejects_unknown_effort() -> None:
    with pytest.raises(ValidationError):
        ThinkingConfig(effort="extreme")  # type: ignore[arg-type]


def _anthropic_payload(config: ThinkingConfig) -> dict:
    request = ModelRequest(
        model="claude-x",
        messages=[Message.text("user", "hi")],
        options={"thinking": thinking_config_payload(config)},
    )
    return build_anthropic_payload(request)


def _openai_payload(config: ThinkingConfig, **option_overrides) -> dict:
    request = ModelRequest(
        model="gpt-x",
        messages=[Message.text("user", "hi")],
        options={"thinking": thinking_config_payload(config), **option_overrides},
    )
    return build_openai_payload(request)


def test_anthropic_payload_effort_uses_adaptive_output_config() -> None:
    payload = _anthropic_payload(ThinkingConfig(effort="high"))
    assert payload["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert payload["output_config"] == {"effort": "high"}


def test_anthropic_payload_budget_uses_legacy_enabled_and_raises_max_tokens() -> None:
    payload = _anthropic_payload(ThinkingConfig(max_tokens=8000))
    assert payload["thinking"] == {
        "type": "enabled",
        "budget_tokens": 8000,
        "display": "summarized",
    }
    # Anthropic requires max_tokens > budget_tokens, so it must be bumped above 8000.
    assert payload["max_tokens"] > 8000


def test_anthropic_payload_disabled_and_default() -> None:
    assert _anthropic_payload(ThinkingConfig(enabled=False))["thinking"] == {"type": "disabled"}
    assert _anthropic_payload(ThinkingConfig())["thinking"] == {
        "type": "adaptive",
        "display": "summarized",
    }


def test_anthropic_payload_preserves_raw_output_config_sibling() -> None:
    # A caller's unrelated output_config key survives the typed effort merge.
    request = ModelRequest(
        model="claude-x",
        messages=[Message.text("user", "hi")],
        options={
            "thinking": thinking_config_payload(ThinkingConfig(effort="high")),
            "anthropic": {"output_config": {"format": {"x": 1}}},
        },
    )
    assert build_anthropic_payload(request)["output_config"] == {
        "format": {"x": 1},
        "effort": "high",
    }


def test_openai_payload_effort_and_disabled() -> None:
    assert _openai_payload(ThinkingConfig(effort="low"))["reasoning"] == {
        "effort": "low",
        "summary": "auto",
    }
    assert _openai_payload(ThinkingConfig())["reasoning"] == {"summary": "auto"}
    assert "reasoning" not in _openai_payload(ThinkingConfig(enabled=False))


def test_openai_payload_preserves_raw_reasoning_sibling() -> None:
    payload = _openai_payload(
        ThinkingConfig(effort="high"), openai={"reasoning": {"generate_summary": "x"}}
    )
    assert payload["reasoning"] == {"generate_summary": "x", "summary": "auto", "effort": "high"}


def _chat_completions_payload(config: ThinkingConfig) -> dict:
    request = ModelRequest(
        model="gemini-x",
        messages=[Message.text("user", "hi")],
        options={"thinking": thinking_config_payload(config)},
    )
    return build_chat_completions_payload(request)


def test_chat_completions_payload_maps_effort_to_reasoning_effort() -> None:
    assert _chat_completions_payload(ThinkingConfig(effort="low"))["reasoning_effort"] == "low"
    # enabled=False is a no-op for the generic adapter (disabling isn't portable; the
    # non-portable "none" is left to a raw provider_options override).
    assert "reasoning_effort" not in _chat_completions_payload(ThinkingConfig(enabled=False))
    # Enabled with no effort -> let the provider/model default decide (no knob sent).
    assert "reasoning_effort" not in _chat_completions_payload(ThinkingConfig())


def test_chat_completions_stream_surfaces_reasoning_content() -> None:
    async def chunks():
        yield {"choices": [{"delta": {"reasoning_content": "thinking..."}}]}
        yield {"choices": [{"delta": {"content": "answer"}}]}
        yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}

    async def collect():
        return [event async for event in chat_completions_stream_events(chunks())]

    events = asyncio.run(collect())
    thinking = [e for e in events if e.type == ModelStreamEventType.THINKING]
    text = [e for e in events if e.type == ModelStreamEventType.TEXT_DELTA]
    assert [e.delta for e in thinking] == ["thinking..."]
    assert [e.delta for e in text] == ["answer"]


def test_neutral_thinking_overrides_raw_provider_keys() -> None:
    # The typed config wins over a raw provider thinking/reasoning the caller also set.
    anthropic = ModelRequest(
        model="claude-x",
        messages=[Message.text("user", "hi")],
        options={
            "thinking": thinking_config_payload(ThinkingConfig(effort="high")),
            "anthropic": {"thinking": {"type": "enabled", "budget_tokens": 9999}},
        },
    )
    assert build_anthropic_payload(anthropic)["thinking"] == {
        "type": "adaptive",
        "display": "summarized",
    }
    openai = ModelRequest(
        model="gpt-x",
        messages=[Message.text("user", "hi")],
        options={
            "thinking": thinking_config_payload(ThinkingConfig(effort="high")),
            "openai": {"reasoning": {"effort": "low", "summary": "detailed"}},
        },
    )
    payload = build_openai_payload(openai)
    assert payload["reasoning"]["effort"] == "high"  # typed effort wins...
    assert payload["reasoning"]["summary"] == "detailed"  # ...but a raw summary survives


def test_payload_tolerates_non_dict_raw_sibling() -> None:
    # A malformed (non-dict) raw sibling must not crash the merge.
    anthropic = ModelRequest(
        model="claude-x",
        messages=[Message.text("user", "hi")],
        options={
            "thinking": thinking_config_payload(ThinkingConfig(effort="high")),
            "anthropic": {"output_config": "bogus"},
        },
    )
    assert build_anthropic_payload(anthropic)["output_config"] == {"effort": "high"}


def test_chat_completions_empty_reasoning_content_falls_back() -> None:
    async def chunks():
        yield {"choices": [{"delta": {"reasoning_content": "", "reasoning": "actual"}}]}
        yield {"choices": [{"delta": {"content": "answer"}}]}
        yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}

    async def collect():
        return [event async for event in chat_completions_stream_events(chunks())]

    events = asyncio.run(collect())
    thinking = [e.delta for e in events if e.type == ModelStreamEventType.THINKING]
    assert thinking == ["actual"]


def test_thinking_config_is_frozen() -> None:
    config = ThinkingConfig(effort="high")
    with pytest.raises(ValidationError):
        config.effort = "low"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# ThinkingPart message model
# --------------------------------------------------------------------------- #
def test_thinking_part_allowed_on_assistant_and_copied() -> None:
    message = Message(
        role=MessageRole.ASSISTANT,
        content=[
            ThinkingPart(text="reason", provider_state={"signature": "S"}),
            TextPart(text="answer"),
        ],
    )
    copied = copy_message_part(message.content[0])
    assert isinstance(copied, ThinkingPart)
    assert copied.text == "reason"
    assert copied.provider_state == {"signature": "S"}
    # provider_state is deep-copied, not aliased
    assert copied.provider_state is not message.content[0].provider_state


def test_thinking_part_text_may_be_empty() -> None:
    part = ThinkingPart(provider_state={"type": "redacted_thinking", "data": "B"})
    assert part.text == ""


def test_thinking_part_rejected_on_user_message() -> None:
    with pytest.raises(ValidationError):
        Message(role=MessageRole.USER, content=[ThinkingPart(text="x")])


# --------------------------------------------------------------------------- #
# Anthropic provider: parse + round-trip
# --------------------------------------------------------------------------- #
def test_anthropic_parses_thinking_blocks_without_crashing() -> None:
    response = {
        "content": [
            {"type": "thinking", "thinking": "step by step", "signature": "SIG"},
            {"type": "redacted_thinking", "data": "BLOB"},
            {"type": "text", "text": "the answer"},
        ],
        "usage": {"input_tokens": 1, "output_tokens": 2},
    }
    events = anthropic_response_events(response)
    thinking = [e for e in events if e.type == ModelStreamEventType.THINKING]
    assert [e.delta for e in thinking] == ["step by step", ""]
    assert thinking[0].payload["provider_state"] == {"type": "thinking", "signature": "SIG"}
    assert thinking[1].payload["provider_state"] == {"type": "redacted_thinking", "data": "BLOB"}


def test_anthropic_round_trips_thinking_verbatim_first() -> None:
    message = Message(
        role=MessageRole.ASSISTANT,
        content=[
            ThinkingPart(text="reason", provider_state={"type": "thinking", "signature": "SIG"}),
            ThinkingPart(
                provider_state={"type": "redacted_thinking", "data": "BLOB"},
            ),
            TextPart(text="answer"),
        ],
    )
    rendered = _anthropic_message(message, resolved_attachments={})
    assert rendered is not None
    assert rendered["content"] == [
        {"type": "thinking", "thinking": "reason", "signature": "SIG"},
        {"type": "redacted_thinking", "data": "BLOB"},
        {"type": "text", "text": "answer"},
    ]


def test_anthropic_drops_thinking_without_signature() -> None:
    message = Message(
        role=MessageRole.ASSISTANT,
        content=[ThinkingPart(text="orphan reasoning"), TextPart(text="answer")],
    )
    rendered = _anthropic_message(message, resolved_attachments={})
    assert rendered is not None
    assert rendered["content"] == [{"type": "text", "text": "answer"}]


def test_anthropic_request_passes_thinking_options_through() -> None:
    request = ModelRequest(
        model="claude-x",
        messages=[Message.text("user", "hi")],
        options={
            "anthropic": {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}
        },
    )
    payload = build_anthropic_payload(request)
    assert payload["thinking"] == {"type": "adaptive"}
    assert payload["output_config"] == {"effort": "high"}


# --------------------------------------------------------------------------- #
# OpenAI provider: surface reasoning summary, preserve encrypted round-trip
# --------------------------------------------------------------------------- #
def test_openai_surfaces_reasoning_summary_and_preserves_state() -> None:
    response = {
        "output": [
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{"type": "summary_text", "text": "Let me reason."}],
                "encrypted_content": "ENC",
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "a"}],
            },
        ],
        "usage": {"input_tokens": 1, "output_tokens": 2},
    }
    events = openai_response_events(response)
    thinking = [e for e in events if e.type == ModelStreamEventType.THINKING]
    assert [e.delta for e in thinking] == ["Let me reason."]
    # reasoning item with encrypted_content is still captured as provider state
    completed = next(e for e in events if e.type == ModelStreamEventType.COMPLETED)
    state = completed.payload["provider_state"]
    assert any(item["state"].get("encrypted_content") == "ENC" for item in state)


def test_openai_input_ignores_thinking_part_round_trips_via_state() -> None:
    message = Message(
        role=MessageRole.ASSISTANT,
        content=[
            ThinkingPart(text="Let me reason."),
            ProviderStatePart(
                provider="openai",
                state={"type": "reasoning", "id": "rs_1", "encrypted_content": "ENC"},
            ),
            TextPart(text="answer"),
        ],
    )
    items = _openai_input_items(
        message, resolved_attachments={}, reasoning_state="inline", use_provider_state=True
    )
    assert any(
        item.get("type") == "reasoning" and item.get("encrypted_content") == "ENC" for item in items
    )
    assert all(item.get("type") != "thinking" for item in items)


def test_openai_input_without_state_skips_thinking_part() -> None:
    message = Message(
        role=MessageRole.ASSISTANT,
        content=[ThinkingPart(text="display only"), TextPart(text="answer")],
    )
    items = _openai_input_items(
        message, resolved_attachments={}, reasoning_state="inline", use_provider_state=True
    )
    # No provider state -> only the assistant text message item, thinking ignored.
    assert [item.get("type") for item in items] == ["message"]


# --------------------------------------------------------------------------- #
# ChatCompletions tolerates ThinkingPart on assistant input
# --------------------------------------------------------------------------- #
def test_chat_completions_ignores_thinking_part() -> None:
    message = Message(
        role=MessageRole.ASSISTANT,
        content=[ThinkingPart(text="reason"), TextPart(text="answer")],
    )
    rendered = chat_assistant_message(message)
    assert rendered["content"] == "answer"


# --------------------------------------------------------------------------- #
# Usage: surface Anthropic thinking tokens
# --------------------------------------------------------------------------- #
def test_usage_surfaces_anthropic_thinking_tokens() -> None:
    metrics = normalize_usage_metrics(
        provider_name="anthropic",
        model="claude-x",
        raw_usage={
            "input_tokens": 10,
            "output_tokens": 40,
            "output_tokens_details": {"thinking_tokens": 25},
        },
    )
    assert metrics.reasoning_output_tokens == 25


# --------------------------------------------------------------------------- #
# Step classification: thinking-only step is THINK_ONLY, not INVALID
# --------------------------------------------------------------------------- #
def test_thinking_only_step_is_think_only() -> None:
    result = AssistantStepResult(
        session_id="s",
        step=0,
        assistant_message=None,
        tool_calls=[],
        completion=ModelCompletion(finish_reason=ModelFinishReason.STOP),
        text_content="",
        has_user_visible_content=False,
        provider_state_count=0,
        thinking_count=1,
    )
    assert classify_assistant_step(result).type == StepClassificationType.THINK_ONLY


# --------------------------------------------------------------------------- #
# Runtime end-to-end
# --------------------------------------------------------------------------- #
class _ThinkingProvider(ModelProvider):
    name = "fake"

    def __init__(self) -> None:
        self.options: dict[str, object] | None = None

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.options = request.options
        yield ModelStreamEvent.thinking(
            "let me think", provider_state={"type": "thinking", "signature": "SIG"}
        )
        yield ModelStreamEvent.text_delta("42")
        yield ModelStreamEvent.completed(
            {"usage": {"input_tokens": 1, "output_tokens": 2}, "stop_reason": "end_turn"}
        )


async def _run(
    *,
    agent_thinking: ThinkingConfig | None,
    run_thinking: ThinkingConfig | None = None,
) -> tuple[_ThinkingProvider, list[Event], list[Message]]:
    provider = _ThinkingProvider()
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="a", model="m", thinking=agent_thinking))
    events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="a",
                messages=[Message.text("user", "hi")],
                thinking=run_thinking,
            )
        )
    ]
    transcript = await app.session_store.load_transcript(events[0].session_id)
    return provider, events, transcript


def _thinking_parts(transcript: list[Message]) -> list[ThinkingPart]:
    return [
        part
        for message in transcript
        if message.role == MessageRole.ASSISTANT
        for part in message.content
        if isinstance(part, ThinkingPart)
    ]


def test_agentspec_thinking_reaches_request_options() -> None:
    provider, _events, transcript = asyncio.run(_run(agent_thinking=ThinkingConfig(effort="high")))
    assert provider.options is not None
    assert provider.options["thinking"] == thinking_config_payload(ThinkingConfig(effort="high"))
    assert _thinking_parts(transcript)[0].text == "let me think"


def test_run_request_thinking_overrides_agentspec() -> None:
    provider, _events, _transcript = asyncio.run(
        _run(
            agent_thinking=ThinkingConfig(effort="low"),
            run_thinking=ThinkingConfig(max_tokens=5000),
        )
    )
    assert provider.options is not None
    assert provider.options["thinking"] == thinking_config_payload(ThinkingConfig(max_tokens=5000))


def test_thinking_part_persisted_with_signature() -> None:
    _provider, _events, transcript = asyncio.run(_run(agent_thinking=ThinkingConfig(effort="high")))
    parts = _thinking_parts(transcript)
    assert parts[0].provider_state == {"type": "thinking", "signature": "SIG"}


def test_include_in_transcript_false_keeps_signed_block_intact() -> None:
    # A signed Anthropic block must keep its ORIGINAL text even when opted out of the
    # transcript: blanking it would break the signature on a tool-use continuation.
    _provider, _events, transcript = asyncio.run(
        _run(agent_thinking=ThinkingConfig(effort="high", include_in_transcript=False))
    )
    parts = _thinking_parts(transcript)
    assert len(parts) == 1
    assert parts[0].text == "let me think"
    assert parts[0].provider_state == {"type": "thinking", "signature": "SIG"}


def test_model_thinking_delta_event_emitted() -> None:
    _provider, events, _transcript = asyncio.run(_run(agent_thinking=ThinkingConfig(effort="high")))
    deltas = [
        event.payload.get("delta")
        for event in events
        if event.type == EventType.MODEL_THINKING_DELTA
    ]
    assert deltas == ["let me think"]


def test_no_thinking_config_leaves_options_unchanged() -> None:
    provider, _events, _transcript = asyncio.run(_run(agent_thinking=None))
    assert provider.options is not None
    assert "thinking" not in provider.options


# --------------------------------------------------------------------------- #
# Transcript query: exclude thinking parts (issue #57 AC#9)
# --------------------------------------------------------------------------- #
def test_filter_transcript_records_excludes_thinking() -> None:
    records = [
        TranscriptRecord(
            index=0,
            message=Message(
                role=MessageRole.ASSISTANT,
                content=[
                    ThinkingPart(text="reason", provider_state={"signature": "S"}),
                    TextPart(text="answer"),
                ],
            ),
        ),
        TranscriptRecord(
            index=1,
            message=Message(
                role=MessageRole.ASSISTANT,
                content=[ThinkingPart(text="only reasoning", provider_state={"signature": "S"})],
            ),
        ),
    ]
    # include_thinking=True -> unchanged.
    assert filter_transcript_records(records, include_thinking=True) == records
    # include_thinking=False -> first record stripped to text; thinking-only record dropped.
    out = filter_transcript_records(records, include_thinking=False)
    assert len(out) == 1
    assert out[0].index == 0
    assert [type(part).__name__ for part in out[0].message.content] == ["TextPart"]


def test_query_transcript_can_exclude_thinking() -> None:
    async def go():
        provider = _ThinkingProvider()
        app = CayuApp()
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="a", model="m", thinking=ThinkingConfig(effort="high")))
        events = [
            event
            async for event in app.run(
                RunRequest(agent_name="a", messages=[Message.text("user", "hi")])
            )
        ]
        session_id = events[0].session_id
        included = await app.session_store.query_transcript(
            TranscriptQuery(session_id=session_id, role="assistant")
        )
        excluded = await app.session_store.query_transcript(
            TranscriptQuery(session_id=session_id, role="assistant", include_thinking=False)
        )
        return included, excluded

    included, excluded = asyncio.run(go())
    included_types = [type(p).__name__ for r in included.records for p in r.message.content]
    excluded_types = [type(p).__name__ for r in excluded.records for p in r.message.content]
    assert "ThinkingPart" in included_types  # default keeps thinking
    assert "ThinkingPart" not in excluded_types  # filtered out
    assert "TextPart" in excluded_types  # the answer survives


# --------------------------------------------------------------------------- #
# Regression: round-trip (the gaps the first pass missed; payload-level gaps now
# covered by the build_anthropic_payload tests above)
# --------------------------------------------------------------------------- #
def test_include_in_transcript_false_round_trips_original_text() -> None:
    # The signed block opted out of the transcript must still echo its ORIGINAL text +
    # signature back to Anthropic (a blanked text would be a modified block -> 400).
    _provider, _events, transcript = asyncio.run(
        _run(agent_thinking=ThinkingConfig(effort="high", include_in_transcript=False))
    )
    assistant = next(m for m in transcript if m.role == MessageRole.ASSISTANT)
    rendered = _anthropic_message(assistant, resolved_attachments={})
    assert rendered is not None
    thinking_blocks = [b for b in rendered["content"] if b.get("type") == "thinking"]
    assert thinking_blocks[0]["thinking"] == "let me think"
    assert thinking_blocks[0]["signature"] == "SIG"


def test_materialize_thinking_drops_only_display_only_when_excluded() -> None:
    # Display-only reasoning (no provider_state) is dropped when excluded...
    assert _materialize_thinking(AssistantThinkingPart(text="x", include=False)) is None
    # ...but a signed block is kept intact so its signature stays valid for round-trip.
    kept = _materialize_thinking(
        AssistantThinkingPart(text="x", provider_state={"signature": "S"}, include=False)
    )
    assert kept is not None
    assert kept.text == "x"
    assert kept.provider_state == {"signature": "S"}


# --------------------------------------------------------------------------- #
# Per-run override threading (approval / dispatch / server)
# --------------------------------------------------------------------------- #
def test_thinking_threads_through_request_copies() -> None:
    from cayu.runtime.approvals import (
        PendingToolApproval,
        PendingToolCallApproval,
        ToolApprovalDecision,
        ToolApprovalRequest,
        copy_pending_tool_approval,
        copy_tool_approval_request,
    )
    from cayu.runtime.dispatch import DispatchRequest, copy_dispatch_request

    cfg = ThinkingConfig(effort="high")
    approval = ToolApprovalRequest(
        session_id="s", approval_id="a", decision=ToolApprovalDecision.APPROVE, thinking=cfg
    )
    assert copy_tool_approval_request(approval).thinking == cfg

    pending = PendingToolApproval(
        approval_id="a",
        tool_call_id="t",
        tool_name="x",
        agent_name="ag",
        tool_calls=[PendingToolCallApproval(tool_call_id="t", tool_name="x")],
        thinking=cfg,
    )
    assert copy_pending_tool_approval(pending).thinking == cfg

    dispatch = DispatchRequest(
        session_id="s", messages=[Message.text("user", "hi")], dispatch_id="d", thinking=cfg
    )
    assert copy_dispatch_request(dispatch).thinking == cfg
