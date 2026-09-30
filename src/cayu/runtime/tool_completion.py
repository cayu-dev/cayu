"""Explicit completion from a successful application tool's durable result."""

from __future__ import annotations

from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator, model_validator

from cayu._validation import require_clean_nonblank
from cayu.messages import ToolCallPart, ToolResultPart, copy_message_part
from cayu.tools.base import ToolEffect


class ToolCompletionPolicy(BaseModel):
    """Complete a single-call round from a designated successful tool.

    The host renders the tool result. Eligible application tools declare
    ``NONE`` or ``IDEMPOTENT`` effects. Rounds with sibling calls continue
    through the ordinary model loop. Supply this policy on each new run or
    resumed turn; approval and crash recovery retain the admitted policy.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    tool_names: tuple[StrictStr, ...] = Field(min_length=1, max_length=64)

    @field_validator("tool_names", mode="before")
    @classmethod
    def validate_tool_names(cls, value: object) -> tuple[str, ...]:
        if type(value) not in (tuple, list):
            raise ValueError("tool_names must be a list or tuple of tool names.")
        value = cast("list[object] | tuple[object, ...]", value)
        if not 1 <= len(value) <= 64:
            raise ValueError("tool_names must contain between 1 and 64 names.")
        if any(type(name) is not str for name in value):
            raise ValueError("tool_names must contain strings.")
        names = tuple(require_clean_nonblank(cast("str", name), "tool_names") for name in value)
        if any(len(name) > 128 for name in names):
            raise ValueError("tool_names must not exceed 128 characters per name.")
        if len(set(names)) != len(names):
            raise ValueError("tool_names must not contain duplicates.")
        return tuple(sorted(names))


class ToolCompletionResult(BaseModel):
    """Detached host-rendering basis from the canonical published tool outcome."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )

    reason: Literal["host_rendered_tool"] = "host_rendered_tool"
    status: Literal["completed"] = "completed"
    call: ToolCallPart
    effect: ToolEffect
    result: ToolResultPart

    @field_validator("call", "result", mode="before")
    @classmethod
    def detach_parts(cls, value: object) -> object:
        if isinstance(value, (ToolCallPart, ToolResultPart)):
            return copy_message_part(value)
        return value

    @model_validator(mode="after")
    def validate_success(self) -> ToolCompletionResult:
        if self.effect not in (ToolEffect.NONE, ToolEffect.IDEMPOTENT):
            raise ValueError("Tool completion requires a none or idempotent effect.")
        if self.result.is_error:
            raise ValueError("Tool completion requires a successful result.")
        fields = ("tool_call_id", "tool_name", "model_step_id", "model_attempt_id", "tool_round_id")
        if any(getattr(self.call, name) != getattr(self.result, name) for name in fields):
            raise ValueError("Tool completion call and result identities must agree.")
        if self.call.tool_round_id is None:
            raise ValueError("Tool completion requires a durable tool-round identity.")
        return self


def copy_tool_completion_policy(value: object) -> ToolCompletionPolicy | None:
    """Revalidate caller values, including unchecked model copies."""
    if value is None:
        return None
    if type(value) is ToolCompletionPolicy:
        value = value.model_dump(mode="python")
    return ToolCompletionPolicy.model_validate(value)
