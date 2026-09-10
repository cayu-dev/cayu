"""Portable run settings bound by work-attempt admission before dispatch."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from cayu._clock import normalize_utc_datetime
from cayu._validation import (
    copy_durable_json_object,
    require_durable_clean_nonblank,
    revalidate_model_input,
)
from cayu.core.thinking import ThinkingConfig
from cayu.deadlines import ExecutionDeadline
from cayu.runtime.budgets import BudgetLimit, copy_request_budget_limits
from cayu.runtime.config import MAX_STEPS
from cayu.runtime.retry_policy import RetryPolicy
from cayu.runtime.stop_policy import RunLimits
from cayu.runtime.structured_output import StructuredOutputSpec
from cayu.runtime.tool_exposure import ToolCapabilityCeiling
from cayu.runtime.work_contracts import require_bounded_work_completion_document

WORK_ATTEMPT_RUN_SEMANTICS_MAX_BYTES = 64 * 1024
WORK_ATTEMPT_RUN_SEMANTICS_MAX_ITEMS = 8_192
WORK_ATTEMPT_RUN_SEMANTICS_MAX_BUDGETS = 64


class WorkAttemptRunSemantics(BaseModel):
    """Detached settings, not authentication of a live invocation or collaborator.

    The admission owner binds this value to its source execution profile. Every
    store/runtime boundary revalidates it; a profile fingerprint alone cannot
    reconstruct these settings after a pre-dispatch process loss.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal[1] = 1
    max_steps: StrictInt = Field(ge=1, le=MAX_STEPS)
    limits: RunLimits = Field(default_factory=RunLimits)
    budget_limits: tuple[BudgetLimit, ...] = ()
    causal_budget_id: str | None = None
    tool_capability_ceiling: ToolCapabilityCeiling | None = None
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    structured_output: StructuredOutputSpec | None = None
    thinking: ThinkingConfig | None = None
    request_metadata: dict[str, Any] = Field(default_factory=dict)
    deadline_expires_at: datetime | None = None
    deadline_source: str = "runtime"
    deadline_scope: str = "execution"

    @property
    def deadline(self) -> ExecutionDeadline:
        """Reconstruct local timer state; only portable authority is persisted."""
        return ExecutionDeadline(
            expires_at=self.deadline_expires_at,
            source=self.deadline_source,
            scope=self.deadline_scope,
        )

    @field_validator("deadline_expires_at")
    @classmethod
    def normalize_deadline(cls, value: datetime | None) -> datetime | None:
        return None if value is None else normalize_utc_datetime(value, "deadline_expires_at")

    @field_validator("causal_budget_id")
    @classmethod
    def validate_causal_budget_id(cls, value: str | None) -> str | None:
        return None if value is None else require_durable_clean_nonblank(value, "causal_budget_id")

    @field_validator(
        "limits",
        "retry_policy",
        "structured_output",
        "thinking",
        "tool_capability_ceiling",
        mode="before",
    )
    @classmethod
    def copy_nested_settings(cls, value: object, info) -> object:
        model_type = {
            "limits": RunLimits,
            "retry_policy": RetryPolicy,
            "structured_output": StructuredOutputSpec,
            "thinking": ThinkingConfig,
            "tool_capability_ceiling": ToolCapabilityCeiling,
        }[info.field_name]
        return revalidate_model_input(value, model_type)

    @field_validator("budget_limits", mode="before")
    @classmethod
    def copy_budgets(cls, value: object) -> tuple[BudgetLimit, ...]:
        if type(value) is not list and type(value) is not tuple:
            raise TypeError("Work-attempt budgets must be a list or tuple.")
        if len(value) > WORK_ATTEMPT_RUN_SEMANTICS_MAX_BUDGETS:
            raise ValueError("Work-attempt budgets exceed the collection limit.")
        limits = tuple(
            BudgetLimit.model_validate(revalidate_model_input(item, BudgetLimit)) for item in value
        )
        return copy_request_budget_limits(limits)

    @field_validator("request_metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: object) -> dict[str, Any]:
        return copy_durable_json_object(value, "work_attempt_request_metadata")

    @model_validator(mode="after")
    def require_bounded_settings(self) -> WorkAttemptRunSemantics:
        # Deadline validation stays with its owner, without storing its private
        # process-local timer attributes in durable model equality.
        _ = self.deadline
        require_bounded_work_completion_document(
            self.model_dump(mode="json", warnings=False),
            "work_attempt_run_semantics",
            max_bytes=WORK_ATTEMPT_RUN_SEMANTICS_MAX_BYTES,
            max_items=WORK_ATTEMPT_RUN_SEMANTICS_MAX_ITEMS,
        )
        return self


def copy_work_attempt_run_semantics(value: WorkAttemptRunSemantics) -> WorkAttemptRunSemantics:
    if type(value) is not WorkAttemptRunSemantics:
        raise TypeError("Work-attempt settings require WorkAttemptRunSemantics.")
    return cast("WorkAttemptRunSemantics", revalidate_model_input(value, WorkAttemptRunSemantics))
