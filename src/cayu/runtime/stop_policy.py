from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from cayu.runtime.usage import SessionUsageSummary


class StopLimit(StrEnum):
    INPUT_TOKENS = "input_tokens"
    OUTPUT_TOKENS = "output_tokens"
    TOTAL_TOKENS = "total_tokens"
    TOOL_CALLS = "tool_calls"
    ELAPSED_SECONDS = "elapsed_seconds"


class RunLimits(BaseModel):
    """Optional hard limits for one session run or resume call.

    Token and tool-call limits are evaluated against the durable session event
    stream. Elapsed time is evaluated against the current runtime invocation.
    """

    model_config = ConfigDict(extra="forbid")

    max_input_tokens: StrictInt | None = Field(default=None, ge=1)
    max_output_tokens: StrictInt | None = Field(default=None, ge=1)
    max_total_tokens: StrictInt | None = Field(default=None, ge=1)
    max_tool_calls: StrictInt | None = Field(default=None, ge=1)
    max_elapsed_seconds: StrictInt | None = Field(default=None, ge=1)
    scope: Literal["session", "run"] = "session"


class StopDecision(BaseModel):
    """Decision returned when a run limit is reached."""

    model_config = ConfigDict(extra="forbid")

    limit: StopLimit
    maximum: StrictInt
    actual: StrictInt
    message: str


def copy_run_limits(limits: RunLimits | None) -> RunLimits:
    if limits is None:
        return RunLimits()
    if type(limits) is not RunLimits:
        raise TypeError("Run limits must be a RunLimits instance.")
    return RunLimits(
        max_input_tokens=limits.max_input_tokens,
        max_output_tokens=limits.max_output_tokens,
        max_total_tokens=limits.max_total_tokens,
        max_tool_calls=limits.max_tool_calls,
        max_elapsed_seconds=limits.max_elapsed_seconds,
        scope=limits.scope,
    )


def has_run_limits(limits: RunLimits) -> bool:
    limits = copy_run_limits(limits)
    return any(
        value is not None
        for value in (
            limits.max_input_tokens,
            limits.max_output_tokens,
            limits.max_total_tokens,
            limits.max_tool_calls,
            limits.max_elapsed_seconds,
        )
    )


def first_reached_limit(
    *,
    limits: RunLimits,
    usage: SessionUsageSummary,
    elapsed_seconds: int,
    pending_tool_calls: int = 0,
) -> StopDecision | None:
    limits = copy_run_limits(limits)
    if type(usage) is not SessionUsageSummary:
        raise TypeError("usage must be a SessionUsageSummary.")
    if elapsed_seconds < 0:
        raise ValueError("elapsed_seconds must be greater than or equal to zero.")
    if pending_tool_calls < 0:
        raise ValueError("pending_tool_calls must be greater than or equal to zero.")

    checks = (
        (
            StopLimit.INPUT_TOKENS,
            limits.max_input_tokens,
            usage.usage.input_tokens,
            True,
        ),
        (
            StopLimit.OUTPUT_TOKENS,
            limits.max_output_tokens,
            usage.usage.output_tokens,
            True,
        ),
        (
            StopLimit.TOTAL_TOKENS,
            limits.max_total_tokens,
            usage.usage.total_tokens,
            True,
        ),
        (
            StopLimit.TOOL_CALLS,
            limits.max_tool_calls,
            usage.tool_calls + pending_tool_calls,
            False,
        ),
        (
            StopLimit.ELAPSED_SECONDS,
            limits.max_elapsed_seconds,
            elapsed_seconds,
            True,
        ),
    )
    for limit, maximum, actual, stop_on_equal in checks:
        if maximum is not None and (actual > maximum or (stop_on_equal and actual == maximum)):
            return StopDecision(
                limit=limit,
                maximum=maximum,
                actual=actual,
                message=f"Run limit reached: {limit.value} {actual} >= {maximum}.",
            )
    return None
