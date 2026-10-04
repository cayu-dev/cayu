from __future__ import annotations

from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, StrictInt, field_validator

from cayu.budgets.run_limits import RunLimits as RunLimits
from cayu.budgets.run_limits import copy_run_limits as copy_run_limits
from cayu.budgets.run_limits import has_run_limits as has_run_limits
from cayu.budgets.usage import SessionUsageSummary
from cayu.tools.inference import InferenceLimits, copy_inference_limits


class StopLimit(StrEnum):
    """Kinds of limits that can stop a session run.

    ``first_reached_limit`` produces the token, tool-call, and elapsed-time
    kinds. ``ESTIMATED_COST`` is produced by request-budget evaluation
    (``BudgetLimit``), whose ``StopDecision`` carries ``Decimal`` cost values
    in ``maximum`` and ``actual``. ``MODEL_STEPS`` is produced when durable
    queued input reaches an otherwise-complete run at its final model step;
    neither is returned by ``first_reached_limit``.
    """

    INPUT_TOKENS = "input_tokens"
    OUTPUT_TOKENS = "output_tokens"
    TOTAL_TOKENS = "total_tokens"
    TOOL_CALLS = "tool_calls"
    MODEL_STEPS = "model_steps"
    ELAPSED_SECONDS = "elapsed_seconds"
    ESTIMATED_COST = "estimated_cost"


class StopDecision(BaseModel):
    """Decision returned when a run limit is reached."""

    model_config = ConfigDict(extra="forbid")

    limit: StopLimit
    maximum: StrictInt | Decimal
    actual: StrictInt | Decimal
    message: str

    @field_validator("maximum", "actual")
    @classmethod
    def validate_numeric_values(cls, value: int | Decimal, info) -> int | Decimal:
        if type(value) is Decimal and not value.is_finite():
            raise ValueError(f"{info.field_name} must be finite.")
        return value


def auxiliary_token_admission(
    *, limits: RunLimits, usage: SessionUsageSummary, request_limits: InferenceLimits
) -> StopDecision | None:
    """Check a proposed envelope against already scope-adjusted usage.

    The runtime owner must serialize admission through attempt settlement; this
    pure check neither reserves resources nor grants dispatch authority. Missing
    usage is not zero and cannot establish hard-token headroom.
    """

    limits = copy_run_limits(limits)
    request_limits = copy_inference_limits(request_limits)
    if type(usage) is not SessionUsageSummary:
        raise TypeError("usage must be a SessionUsageSummary.")
    usage = SessionUsageSummary.model_validate(
        {name: getattr(usage, name) for name in SessionUsageSummary.model_fields}
    )
    checks = (
        (
            StopLimit.INPUT_TOKENS,
            limits.max_input_tokens,
            usage.usage.input_tokens + request_limits.max_input_tokens,
        ),
        (
            StopLimit.OUTPUT_TOKENS,
            limits.max_output_tokens,
            usage.usage.output_tokens + request_limits.max_output_tokens,
        ),
        (
            StopLimit.TOTAL_TOKENS,
            limits.max_total_tokens,
            usage.usage.total_tokens
            + request_limits.max_input_tokens
            + request_limits.max_output_tokens,
        ),
    )
    if usage.unmeasured_model_attempts and any(maximum is not None for _, maximum, _ in checks):
        raise ValueError("Unmeasured model attempts prevent authoritative token admission.")
    for limit, maximum, proposed in checks:
        if maximum is not None and proposed > maximum:
            return StopDecision(
                limit=limit,
                maximum=maximum,
                actual=proposed,
                message=f"Auxiliary request exceeds remaining {limit.value} headroom.",
            )
    return None


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
            comparator = ">=" if stop_on_equal else ">"
            return StopDecision(
                limit=limit,
                maximum=maximum,
                actual=actual,
                message=f"Run limit reached: {limit.value} {actual} {comparator} {maximum}.",
            )
    return None
