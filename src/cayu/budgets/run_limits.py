"""Portable hard-limit configuration shared by requests and runtime policies."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from cayu._validation import MAX_DURABLE_JSON_INTEGER


class RunLimits(BaseModel):
    """Optional hard limits for one session run or resume call.

    Every limit is evaluated against the clock selected by ``scope``:

    - ``scope="run"`` (the default): token, tool-call, and elapsed-time
      limits all measure the current runtime invocation — usage deltas since
      the original run entered and active time since the run started. Durable
      approval and user-input continuations retain the baseline and exclude the
      human wait interval. A fresh run or clean-boundary resume resets it.
    - ``scope="session"``: token, tool-call, and elapsed-time limits all
      measure the whole durable session — cumulative usage from the session
      event stream and wall time since the session was created. A resumed
      session that already meets a limit stops immediately.
    """

    model_config = ConfigDict(extra="forbid")

    max_input_tokens: StrictInt | None = Field(default=None, ge=1, le=MAX_DURABLE_JSON_INTEGER)
    max_output_tokens: StrictInt | None = Field(default=None, ge=1, le=MAX_DURABLE_JSON_INTEGER)
    max_total_tokens: StrictInt | None = Field(default=None, ge=1, le=MAX_DURABLE_JSON_INTEGER)
    max_tool_calls: StrictInt | None = Field(default=None, ge=1, le=MAX_DURABLE_JSON_INTEGER)
    max_elapsed_seconds: StrictInt | None = Field(default=None, ge=1, le=MAX_DURABLE_JSON_INTEGER)
    scope: Literal["session", "run"] = "run"


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
