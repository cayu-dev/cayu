from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator
from pydantic.json_schema import SkipJsonSchema  # noqa: TC002 - Pydantic needs this at runtime.

from cayu._validation import copy_json_value, require_clean_nonblank, require_nonblank
from cayu.core.thinking import ThinkingConfig
from cayu.runtime.budgets import BudgetLimit, copy_request_budget_limits
from cayu.runtime.loop_policies import LoopPolicy, validate_loop_policies
from cayu.runtime.retry_policy import RetryPolicy, copy_retry_policy
from cayu.runtime.stop_policy import RunLimits, copy_run_limits
from cayu.runtime.structured_output import StructuredOutputSpec, copy_structured_output_spec


class ToolApprovalDecision(StrEnum):
    APPROVE = "approve"
    DENY = "deny"


class ToolApprovalRecoveryOutcome(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"


class ToolApprovalRequest(BaseModel):
    """Caller decision for a pending tool approval."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    session_id: str
    approval_id: str
    decision: ToolApprovalDecision
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    max_steps: StrictInt = Field(default=16, ge=1, le=256)
    limits: RunLimits = Field(default_factory=RunLimits)
    budget_limits: tuple[BudgetLimit, ...] = Field(default_factory=tuple)
    retry_policy: RetryPolicy | None = None
    structured_output: StructuredOutputSpec | None = None
    thinking: ThinkingConfig | None = None
    loop_policies: SkipJsonSchema[tuple[LoopPolicy, ...]] = Field(
        default_factory=tuple,
        exclude=True,
    )

    @field_validator("session_id", "approval_id")
    @classmethod
    def validate_nonblank_ids(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("reason")
    @classmethod
    def validate_optional_reason(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_nonblank(value, info.field_name)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "metadata")

    @field_validator("structured_output")
    @classmethod
    def copy_structured_output(
        cls,
        value: StructuredOutputSpec | None,
    ) -> StructuredOutputSpec | None:
        return copy_structured_output_spec(value)

    @field_validator("budget_limits", mode="before")
    @classmethod
    def copy_budget_limits(cls, value) -> tuple[BudgetLimit, ...]:
        return copy_request_budget_limits(value)

    @field_validator("loop_policies", mode="before")
    @classmethod
    def copy_loop_policies(cls, value) -> tuple[LoopPolicy, ...]:
        return validate_loop_policies(value, field_name="loop_policies")


class ToolApprovalRecoveryRequest(BaseModel):
    """Caller-supplied terminal outcome for an approved tool with unknown result."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    session_id: str
    approval_id: str
    tool_call_id: str
    outcome: ToolApprovalRecoveryOutcome
    message: str
    structured: dict[str, Any] | None = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    max_steps: StrictInt = Field(default=16, ge=1, le=256)
    limits: RunLimits = Field(default_factory=RunLimits)
    budget_limits: tuple[BudgetLimit, ...] = Field(default_factory=tuple)
    retry_policy: RetryPolicy | None = None
    structured_output: StructuredOutputSpec | None = None
    thinking: ThinkingConfig | None = None
    loop_policies: SkipJsonSchema[tuple[LoopPolicy, ...]] = Field(
        default_factory=tuple,
        exclude=True,
    )

    @field_validator("session_id", "approval_id", "tool_call_id")
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

    @field_validator("budget_limits", mode="before")
    @classmethod
    def copy_budget_limits(cls, value) -> tuple[BudgetLimit, ...]:
        return copy_request_budget_limits(value)

    @field_validator("loop_policies", mode="before")
    @classmethod
    def copy_loop_policies(cls, value) -> tuple[LoopPolicy, ...]:
        return validate_loop_policies(value, field_name="loop_policies")


class PendingToolCallApproval(BaseModel):
    """One tool call captured inside a pending approval round."""

    model_config = ConfigDict(extra="forbid")

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    policy_decision: str | None = None
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tool_call_id", "tool_name")
    @classmethod
    def validate_nonblank_fields(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("policy_decision", "reason")
    @classmethod
    def validate_optional_nonblank_fields(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        if info.field_name == "reason":
            return require_nonblank(value, info.field_name)
        return require_clean_nonblank(value, info.field_name)

    @field_validator("arguments", "metadata", mode="before")
    @classmethod
    def copy_json_fields(cls, value: dict[str, Any], info) -> dict[str, Any]:
        return copy_json_value(value, info.field_name)


class PendingToolApproval(BaseModel):
    """Durable checkpoint state for a tool call waiting on caller approval."""

    model_config = ConfigDict(extra="forbid")

    approval_id: str
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    agent_name: str
    environment_name: str | None = None
    workspace_id: str | None = None
    task_id: str | None = None
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    tool_calls: list[PendingToolCallApproval]
    structured_output: StructuredOutputSpec | None = None
    thinking: ThinkingConfig | None = None

    @field_validator("approval_id", "tool_call_id", "tool_name", "agent_name")
    @classmethod
    def validate_nonblank_fields(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("environment_name", "workspace_id", "task_id", "reason")
    @classmethod
    def validate_optional_nonblank_fields(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        if info.field_name == "reason":
            return require_nonblank(value, info.field_name)
        return require_clean_nonblank(value, info.field_name)

    @field_validator("arguments", "metadata", mode="before")
    @classmethod
    def copy_json_fields(cls, value: dict[str, Any], info) -> dict[str, Any]:
        return copy_json_value(value, info.field_name)

    @field_validator("structured_output")
    @classmethod
    def copy_structured_output(
        cls,
        value: StructuredOutputSpec | None,
    ) -> StructuredOutputSpec | None:
        return copy_structured_output_spec(value)

    @field_validator("tool_calls")
    @classmethod
    def copy_tool_calls(
        cls,
        value: list[PendingToolCallApproval],
    ) -> list[PendingToolCallApproval]:
        copied = [copy_pending_tool_call_approval(call) for call in value]
        if not copied:
            raise ValueError("Pending tool approval must include tool calls.")
        return copied


def copy_tool_approval_request(request: ToolApprovalRequest) -> ToolApprovalRequest:
    if type(request) is not ToolApprovalRequest:
        raise TypeError("Tool approval resolution requires a ToolApprovalRequest.")
    return ToolApprovalRequest(
        session_id=request.session_id,
        approval_id=request.approval_id,
        decision=request.decision,
        reason=request.reason,
        metadata=copy_json_value(request.metadata, "metadata"),
        max_steps=request.max_steps,
        limits=copy_run_limits(request.limits),
        budget_limits=copy_request_budget_limits(request.budget_limits),
        retry_policy=copy_retry_policy(request.retry_policy) if request.retry_policy else None,
        structured_output=copy_structured_output_spec(request.structured_output),
        thinking=request.thinking,
        loop_policies=validate_loop_policies(request.loop_policies, field_name="loop_policies"),
    )


def copy_tool_approval_recovery_request(
    request: ToolApprovalRecoveryRequest,
) -> ToolApprovalRecoveryRequest:
    if type(request) is not ToolApprovalRecoveryRequest:
        raise TypeError("Tool approval recovery requires a ToolApprovalRecoveryRequest.")
    return ToolApprovalRecoveryRequest(
        session_id=request.session_id,
        approval_id=request.approval_id,
        tool_call_id=request.tool_call_id,
        outcome=request.outcome,
        message=request.message,
        structured=copy_json_value(request.structured, "structured"),
        artifacts=copy_json_value(request.artifacts, "artifacts"),
        reason=request.reason,
        metadata=copy_json_value(request.metadata, "metadata"),
        max_steps=request.max_steps,
        limits=copy_run_limits(request.limits),
        budget_limits=copy_request_budget_limits(request.budget_limits),
        retry_policy=copy_retry_policy(request.retry_policy) if request.retry_policy else None,
        structured_output=copy_structured_output_spec(request.structured_output),
        thinking=request.thinking,
        loop_policies=validate_loop_policies(request.loop_policies, field_name="loop_policies"),
    )


def copy_pending_tool_approval(approval: PendingToolApproval) -> PendingToolApproval:
    if type(approval) is not PendingToolApproval:
        raise TypeError("Pending tool approval must be a PendingToolApproval.")
    return PendingToolApproval(
        approval_id=approval.approval_id,
        tool_call_id=approval.tool_call_id,
        tool_name=approval.tool_name,
        arguments=copy_json_value(approval.arguments, "arguments"),
        agent_name=approval.agent_name,
        environment_name=approval.environment_name,
        workspace_id=approval.workspace_id,
        task_id=approval.task_id,
        reason=approval.reason,
        metadata=copy_json_value(approval.metadata, "metadata"),
        tool_calls=[copy_pending_tool_call_approval(call) for call in approval.tool_calls],
        structured_output=copy_structured_output_spec(approval.structured_output),
        thinking=approval.thinking,
    )


def copy_pending_tool_call_approval(
    call: PendingToolCallApproval,
) -> PendingToolCallApproval:
    if type(call) is not PendingToolCallApproval:
        raise TypeError("Pending tool call approval must be a PendingToolCallApproval.")
    return PendingToolCallApproval(
        tool_call_id=call.tool_call_id,
        tool_name=call.tool_name,
        arguments=copy_json_value(call.arguments, "arguments"),
        policy_decision=call.policy_decision,
        reason=call.reason,
        metadata=copy_json_value(call.metadata, "metadata"),
    )
