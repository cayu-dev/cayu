from __future__ import annotations

from cayu.core import Message, MessageRole, ProviderStatePart, TextPart
from cayu.providers import (
    ModelCompletion,
    ModelFinishReason,
    ModelStreamEvent,
    ModelStreamEventType,
    normalize_model_completion,
)
from cayu.runtime._runtime_records import ToolCallRequest
from cayu.runtime.model_steps import (
    AssistantStepResult,
    StepClassificationType,
    classify_assistant_step,
)


def _step_result(
    *,
    finish_reason: ModelFinishReason = ModelFinishReason.STOP,
    message: Message | None = None,
    tool_calls: list[ToolCallRequest] | None = None,
) -> AssistantStepResult:
    text_content = ""
    provider_state_count = 0
    if message is not None:
        text_content = "".join(part.text for part in message.content if type(part) is TextPart)
        provider_state_count = sum(1 for part in message.content if type(part) is ProviderStatePart)
    return AssistantStepResult(
        session_id="sess_1",
        step=1,
        assistant_message=message,
        tool_calls=[] if tool_calls is None else tool_calls,
        completion=ModelCompletion(finish_reason=finish_reason),
        text_content=text_content,
        has_user_visible_content=bool(text_content.strip()),
        provider_state_count=provider_state_count,
    )


def test_normalize_model_completion_maps_common_provider_reasons() -> None:
    assert (
        normalize_model_completion({"finish_reason": "stop"}).finish_reason
        == ModelFinishReason.STOP
    )
    assert (
        normalize_model_completion({"stop_reason": "tool_use"}).finish_reason
        == ModelFinishReason.TOOL_CALLS
    )
    assert (
        normalize_model_completion(
            {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}
        ).finish_reason
        == ModelFinishReason.LENGTH
    )
    assert normalize_model_completion({"status": "failed"}).finish_reason == ModelFinishReason.ERROR


def test_model_stream_event_normalizes_direct_completed_construction() -> None:
    event = ModelStreamEvent(
        type=ModelStreamEventType.COMPLETED,
        payload={"stop_reason": "end_turn"},
    )

    assert event.completion is not None
    assert event.completion.finish_reason == ModelFinishReason.STOP
    assert event.completion.raw_finish_reason == "end_turn"


def test_classify_assistant_step_continues_for_tool_calls() -> None:
    result = _step_result(
        finish_reason=ModelFinishReason.STOP,
        message=Message.text("assistant", "I will call a tool."),
        tool_calls=[ToolCallRequest(id="call_1", name="echo", arguments={})],
    )

    classification = classify_assistant_step(result)

    assert classification.type == StepClassificationType.CONTINUE


def test_classify_assistant_step_final_for_visible_text() -> None:
    result = _step_result(message=Message.text("assistant", "done"))

    classification = classify_assistant_step(result)

    assert classification.type == StepClassificationType.FINAL


def test_classify_assistant_step_length_and_filter_stop_before_final() -> None:
    assert (
        classify_assistant_step(
            _step_result(
                finish_reason=ModelFinishReason.LENGTH,
                message=Message.text("assistant", "partial"),
            )
        ).type
        == StepClassificationType.LENGTH
    )
    assert (
        classify_assistant_step(_step_result(finish_reason=ModelFinishReason.CONTENT_FILTER)).type
        == StepClassificationType.FILTERED
    )


def test_classify_assistant_step_detects_think_only_and_invalid() -> None:
    think_only = _step_result(
        message=Message(
            role=MessageRole.ASSISTANT,
            content=[ProviderStatePart(provider="openai", state={"type": "reasoning"})],
        )
    )

    assert classify_assistant_step(think_only).type == StepClassificationType.THINK_ONLY
    assert classify_assistant_step(_step_result()).type == StepClassificationType.INVALID
