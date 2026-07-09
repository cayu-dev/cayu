from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator
from pydantic.json_schema import SkipJsonSchema  # noqa: TC002 - Pydantic needs this at runtime.

from cayu._validation import copy_json_value, require_clean_nonblank, require_nonblank
from cayu.core.thinking import ThinkingConfig
from cayu.runtime.approvals import (
    PendingToolCallApproval,
    ResolutionActor,
    ToolApprovalRecoveryOutcome,
    copy_pending_tool_call_approval,
    copy_resolution_actor,
)
from cayu.runtime.budgets import BudgetLimit, copy_budget_limits, copy_request_budget_limits
from cayu.runtime.loop_policies import LoopPolicy, validate_loop_policies
from cayu.runtime.retry_policy import RetryPolicy, copy_retry_policy
from cayu.runtime.stop_policy import RunLimits, copy_run_limits
from cayu.runtime.structured_output import StructuredOutputSpec, copy_structured_output_spec

PENDING_USER_INPUT_CHECKPOINT_KEY = "pending_user_input"


class UserInputResponse(BaseModel):
    """Caller-supplied answer that resumes a session paused by ``ask_user``.

    ``max_steps``, ``limits``, ``budget_limits``, and ``retry_policy`` default to ``None``,
    which means "inherit the original run's configuration" as persisted on the pending
    user-input checkpoint. Passing an explicit value overrides it for the resumed run.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    session_id: str
    input_id: str
    answer: str
    structured: dict[str, Any] | None = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    resolved_by: ResolutionActor | None = None
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

    @field_validator("session_id", "input_id")
    @classmethod
    def validate_nonblank_ids(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("answer")
    @classmethod
    def validate_answer(cls, value: str, info) -> str:
        return require_nonblank(value, info.field_name)

    @field_validator("structured", "artifacts", "metadata", mode="before")
    @classmethod
    def copy_json_fields(cls, value, info):
        return copy_json_value(value, info.field_name)

    @field_validator("resolved_by")
    @classmethod
    def copy_resolved_by(cls, value: ResolutionActor | None) -> ResolutionActor | None:
        return copy_resolution_actor(value)

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

    @field_validator("loop_policies", mode="before")
    @classmethod
    def copy_loop_policies(cls, value) -> tuple[LoopPolicy, ...]:
        return validate_loop_policies(value, field_name="loop_policies")


class PendingUserInput(BaseModel):
    """Durable checkpoint state for a session paused on an ``ask_user`` question.

    ``max_steps``, ``limits``, ``budget_limits``, and ``retry_policy`` persist the original
    run's configuration across the pause so resolving the question resumes with the same
    config instead of fresh defaults (unless the resolution request overrides them). They
    are optional so checkpoints written before this state existed still load.
    """

    model_config = ConfigDict(extra="forbid")

    input_id: str
    tool_call_id: str
    tool_name: str
    question: str
    options: list[str] = Field(default_factory=list)
    arguments: dict[str, Any] = Field(default_factory=dict)
    agent_name: str
    environment_name: str | None = None
    workspace_id: str | None = None
    task_id: str | None = None
    tool_calls: list[PendingToolCallApproval]
    structured_output: StructuredOutputSpec | None = None
    thinking: ThinkingConfig | None = None
    max_steps: StrictInt | None = Field(default=None, ge=1, le=256)
    limits: RunLimits | None = None
    budget_limits: tuple[BudgetLimit, ...] | None = None
    retry_policy: RetryPolicy | None = None

    @field_validator("input_id", "tool_call_id", "tool_name", "agent_name")
    @classmethod
    def validate_nonblank_fields(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str, info) -> str:
        return require_nonblank(value, info.field_name)

    @field_validator("environment_name", "workspace_id", "task_id")
    @classmethod
    def validate_optional_nonblank_fields(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("options")
    @classmethod
    def validate_options(cls, value: list[str], info) -> list[str]:
        return [require_nonblank(option, "option") for option in value]

    @field_validator("arguments", mode="before")
    @classmethod
    def copy_arguments(cls, value: dict[str, Any], info) -> dict[str, Any]:
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
            raise ValueError("Pending user input must include tool calls.")
        return copied

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
        return copy_budget_limits(value, field_name="budget_limits")

    @field_validator("retry_policy")
    @classmethod
    def copy_retry(cls, value: RetryPolicy | None) -> RetryPolicy | None:
        if value is None:
            return None
        return copy_retry_policy(value)


def copy_user_input_response(response: UserInputResponse) -> UserInputResponse:
    if type(response) is not UserInputResponse:
        raise TypeError("User input resolution requires a UserInputResponse.")
    return UserInputResponse(
        session_id=response.session_id,
        input_id=response.input_id,
        answer=response.answer,
        structured=copy_json_value(response.structured, "structured"),
        artifacts=copy_json_value(response.artifacts, "artifacts"),
        metadata=copy_json_value(response.metadata, "metadata"),
        resolved_by=copy_resolution_actor(response.resolved_by),
        max_steps=response.max_steps,
        limits=copy_run_limits(response.limits) if response.limits is not None else None,
        budget_limits=(
            copy_request_budget_limits(response.budget_limits)
            if response.budget_limits is not None
            else None
        ),
        retry_policy=copy_retry_policy(response.retry_policy) if response.retry_policy else None,
        structured_output=copy_structured_output_spec(response.structured_output),
        thinking=response.thinking,
        loop_policies=validate_loop_policies(response.loop_policies, field_name="loop_policies"),
    )


def copy_pending_user_input(pending: PendingUserInput) -> PendingUserInput:
    if type(pending) is not PendingUserInput:
        raise TypeError("Pending user input must be a PendingUserInput.")
    return PendingUserInput(
        input_id=pending.input_id,
        tool_call_id=pending.tool_call_id,
        tool_name=pending.tool_name,
        question=pending.question,
        options=list(pending.options),
        arguments=copy_json_value(pending.arguments, "arguments"),
        agent_name=pending.agent_name,
        environment_name=pending.environment_name,
        workspace_id=pending.workspace_id,
        task_id=pending.task_id,
        tool_calls=[copy_pending_tool_call_approval(call) for call in pending.tool_calls],
        structured_output=copy_structured_output_spec(pending.structured_output),
        thinking=pending.thinking,
        max_steps=pending.max_steps,
        limits=copy_run_limits(pending.limits) if pending.limits is not None else None,
        budget_limits=(
            copy_budget_limits(pending.budget_limits, field_name="budget_limits")
            if pending.budget_limits is not None
            else None
        ),
        retry_policy=copy_retry_policy(pending.retry_policy)
        if pending.retry_policy is not None
        else None,
    )


def pending_user_input_from_checkpoint(
    checkpoint: dict[str, Any] | None,
) -> PendingUserInput | None:
    if checkpoint is None:
        return None
    copied_checkpoint = copy_json_value(checkpoint, "checkpoint")
    value = copied_checkpoint.get(PENDING_USER_INPUT_CHECKPOINT_KEY)
    if value is None:
        return None
    if type(value) is not dict:
        raise ValueError("Pending user input checkpoint must be an object.")
    return PendingUserInput(**value)


class UserInputRecoveryRequest(BaseModel):
    """Caller-supplied terminal outcome for a paused round's tool with an unknown result.

    Used when `resolve_user_input` reports `manual_recovery_required`: a tool in the paused
    round started on a prior resume but recorded no terminal event (a crash mid-tool), so it
    cannot be re-run automatically. The caller supplies the externally verified outcome for
    that `tool_call_id`; `answer` is re-supplied so the `ask_user` result is available if it
    was not already recorded before the crash.

    ``max_steps``, ``limits``, ``budget_limits``, and ``retry_policy`` default to ``None`` =
    "inherit the original run's configuration" from the pending checkpoint.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    session_id: str
    input_id: str
    answer: str
    tool_call_id: str
    outcome: ToolApprovalRecoveryOutcome
    message: str
    structured: dict[str, Any] | None = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    resolved_by: ResolutionActor | None = None
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

    @field_validator("session_id", "input_id", "tool_call_id")
    @classmethod
    def validate_nonblank_ids(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("resolved_by")
    @classmethod
    def copy_resolved_by(cls, value: ResolutionActor | None) -> ResolutionActor | None:
        return copy_resolution_actor(value)

    @field_validator("answer", "message")
    @classmethod
    def validate_nonblank_text(cls, value: str, info) -> str:
        return require_nonblank(value, info.field_name)

    @field_validator("reason")
    @classmethod
    def validate_optional_reason(cls, value: str | None, info) -> str | None:
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

    @field_validator("loop_policies", mode="before")
    @classmethod
    def copy_loop_policies(cls, value) -> tuple[LoopPolicy, ...]:
        return validate_loop_policies(value, field_name="loop_policies")


def copy_user_input_recovery_request(
    request: UserInputRecoveryRequest,
) -> UserInputRecoveryRequest:
    if type(request) is not UserInputRecoveryRequest:
        raise TypeError("User input recovery requires a UserInputRecoveryRequest.")
    return UserInputRecoveryRequest(
        session_id=request.session_id,
        input_id=request.input_id,
        answer=request.answer,
        tool_call_id=request.tool_call_id,
        outcome=request.outcome,
        message=request.message,
        structured=copy_json_value(request.structured, "structured"),
        artifacts=copy_json_value(request.artifacts, "artifacts"),
        reason=request.reason,
        metadata=copy_json_value(request.metadata, "metadata"),
        resolved_by=copy_resolution_actor(request.resolved_by),
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
