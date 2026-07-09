from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator
from pydantic.json_schema import SkipJsonSchema  # noqa: TC002 - Pydantic needs this at runtime.

from cayu._validation import copy_json_value, require_clean_nonblank, require_nonblank
from cayu.core.thinking import ThinkingConfig
from cayu.runtime.approvals import ToolApprovalRecoveryOutcome
from cayu.runtime.budgets import BudgetLimit, copy_request_budget_limits
from cayu.runtime.loop_policies import LoopPolicy, validate_loop_policies
from cayu.runtime.retry_policy import RetryPolicy, copy_retry_policy
from cayu.runtime.stop_policy import RunLimits, copy_run_limits
from cayu.runtime.structured_output import StructuredOutputSpec, copy_structured_output_spec


class ToolRoundRecoveryRequest(BaseModel):
    """Caller-supplied terminal outcome for a crashed ordinary tool call.

    Targets one started-but-unresolved tool call in a session's pending tool
    round (a round that crashed outside approval/user-input flows). The
    operator supplies the externally verified COMPLETED/FAILED outcome as
    evidence; Cayu persists it as the call's terminal result and never re-runs
    the tool. ``max_steps``, ``limits``, ``budget_limits``, and
    ``retry_policy`` default to ``None``, which applies the runtime defaults
    for the resumed run (the pending tool round persists no run configuration).
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    session_id: str
    round_id: str
    tool_call_id: str
    outcome: ToolApprovalRecoveryOutcome
    message: str
    structured: dict[str, Any] | None = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    max_steps: StrictInt | None = Field(default=None, ge=1, le=256)
    limits: RunLimits | None = None
    budget_limits: tuple[BudgetLimit, ...] | None = None
    retry_policy: RetryPolicy | None = None
    structured_output: StructuredOutputSpec | None = None
    thinking: ThinkingConfig | None = None
    loop_policies: SkipJsonSchema[tuple[LoopPolicy, ...]] = Field(
        default_factory=tuple,
        exclude=True,
    )

    @field_validator("session_id", "round_id", "tool_call_id")
    @classmethod
    def validate_nonblank_ids(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str, info) -> str:
        return require_nonblank(value, info.field_name)

    @field_validator("reason")
    @classmethod
    def validate_optional_reason(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_nonblank(value, info.field_name)

    @field_validator("structured", "artifacts", "metadata", mode="before")
    @classmethod
    def copy_json_fields(cls, value, info):
        return copy_json_value(value, info.field_name)

    @field_validator("structured_output")
    @classmethod
    def copy_structured_output(
        cls,
        value: StructuredOutputSpec | None,
    ) -> StructuredOutputSpec | None:
        return copy_structured_output_spec(value)

    @field_validator("limits")
    @classmethod
    def copy_limits(cls, value: RunLimits | None) -> RunLimits | None:
        if value is None:
            return None
        return copy_run_limits(value)

    @field_validator("budget_limits", mode="before")
    @classmethod
    def copy_budget_limits(cls, value) -> tuple[BudgetLimit, ...] | None:
        if value is None:
            return None
        return copy_request_budget_limits(value)


def copy_tool_round_recovery_request(
    request: ToolRoundRecoveryRequest,
) -> ToolRoundRecoveryRequest:
    if type(request) is not ToolRoundRecoveryRequest:
        raise TypeError("Tool round recovery requires a ToolRoundRecoveryRequest.")
    return ToolRoundRecoveryRequest(
        session_id=request.session_id,
        round_id=request.round_id,
        tool_call_id=request.tool_call_id,
        outcome=request.outcome,
        message=request.message,
        structured=copy_json_value(request.structured, "structured"),
        artifacts=copy_json_value(request.artifacts, "artifacts"),
        reason=request.reason,
        metadata=copy_json_value(request.metadata, "metadata"),
        max_steps=request.max_steps,
        limits=copy_run_limits(request.limits) if request.limits is not None else None,
        budget_limits=(
            copy_request_budget_limits(request.budget_limits)
            if request.budget_limits is not None
            else None
        ),
        retry_policy=copy_retry_policy(request.retry_policy) if request.retry_policy else None,
        structured_output=copy_structured_output_spec(request.structured_output),
        thinking=request.thinking,
        loop_policies=validate_loop_policies(request.loop_policies, field_name="loop_policies"),
    )
