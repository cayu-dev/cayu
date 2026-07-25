from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from cayu.core.messages import Message, MessageRole, ProviderStatePart, TextPart, ThinkingPart
from cayu.providers import ModelCompletion, ModelFinishReason
from cayu.runtime._runtime_records import ToolCallRequest
from cayu.runtime.execution_units import (
    ModelAttemptIdentity,
    ToolRoundIdentity,
    copy_tool_round_identity,
)


class StepClassificationType(StrEnum):
    CONTINUE = "continue"
    FINAL = "final"
    LENGTH = "length"
    FILTERED = "filtered"
    FAILED = "failed"
    INVALID = "invalid"
    THINK_ONLY = "think_only"


@dataclass(frozen=True)
class AssistantStepResult:
    session_id: str
    step: int
    model_step_id: str
    model_attempt_id: str
    tool_round_identity: ToolRoundIdentity | None
    assistant_message: Message | None
    tool_calls: list[ToolCallRequest]
    completion: ModelCompletion
    text_content: str
    has_user_visible_content: bool
    provider_state_count: int
    thinking_count: int = 0

    def __post_init__(self) -> None:
        ModelAttemptIdentity(
            model_step_id=self.model_step_id,
            model_attempt_id=self.model_attempt_id,
        )
        if self.tool_round_identity is not None:
            identity = copy_tool_round_identity(self.tool_round_identity)
            if (
                identity.model_step_id != self.model_step_id
                or identity.model_attempt_id != self.model_attempt_id
            ):
                raise ValueError(
                    "Tool round identity must belong to the assistant step's model attempt."
                )
        if bool(self.tool_calls) != (self.tool_round_identity is not None):
            raise ValueError("Assistant tool calls require exactly one tool-round identity.")


@dataclass(frozen=True)
class StepClassification:
    type: StepClassificationType
    reason: str

    def payload(self) -> dict[str, str]:
        return {
            "type": self.type.value,
            "reason": self.reason,
        }


def classify_assistant_step(result: AssistantStepResult) -> StepClassification:
    """Classify what the runtime should do with one completed assistant step."""

    if result.tool_calls:
        return StepClassification(
            type=StepClassificationType.CONTINUE,
            reason="assistant requested tool calls",
        )

    finish_reason = result.completion.finish_reason
    if finish_reason == ModelFinishReason.ERROR:
        return StepClassification(
            type=StepClassificationType.FAILED,
            reason="provider reported a failed model step",
        )
    if finish_reason == ModelFinishReason.LENGTH:
        return StepClassification(
            type=StepClassificationType.LENGTH,
            reason="provider stopped because the model reached an output limit",
        )
    if finish_reason == ModelFinishReason.CONTENT_FILTER:
        return StepClassification(
            type=StepClassificationType.FILTERED,
            reason="provider stopped because output was filtered",
        )
    if completion_requests_follow_up(result.completion):
        return StepClassification(
            type=StepClassificationType.CONTINUE,
            reason="provider requested another model step",
        )
    if result.has_user_visible_content:
        return StepClassification(
            type=StepClassificationType.FINAL,
            reason="assistant produced user-visible content",
        )
    if result.provider_state_count > 0 or result.thinking_count > 0:
        return StepClassification(
            type=StepClassificationType.THINK_ONLY,
            reason="assistant produced reasoning or provider state but no user-visible content",
        )
    return StepClassification(
        type=StepClassificationType.INVALID,
        reason="assistant produced no tool calls and no user-visible content",
    )


def completion_requests_follow_up(completion: ModelCompletion) -> bool:
    """Return whether a normal completion explicitly requests another model step."""

    if type(completion) is not ModelCompletion:
        raise TypeError("Completion must be a ModelCompletion instance.")
    return completion.end_turn is False and completion.finish_reason not in {
        ModelFinishReason.ERROR,
        ModelFinishReason.LENGTH,
        ModelFinishReason.CONTENT_FILTER,
    }


def assistant_text_content(message: Message | None) -> str:
    if message is None:
        return ""
    if message.role != MessageRole.ASSISTANT:
        raise ValueError("Assistant step result requires an assistant message.")
    text_parts: list[str] = []
    for part in message.content:
        if type(part) is TextPart:
            text_parts.append(part.text)
    return "".join(text_parts)


def _count_parts(message: Message | None, part_type: type) -> int:
    if message is None:
        return 0
    return sum(1 for part in message.content if type(part) is part_type)


def provider_state_count(message: Message | None) -> int:
    return _count_parts(message, ProviderStatePart)


def thinking_count(message: Message | None) -> int:
    return _count_parts(message, ThinkingPart)
