"""Detached settings retained by atomic invocation admission."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from cayu._validation import require_durable_clean_nonblank
from cayu.budgets.run_limits import RunLimits, copy_run_limits
from cayu.configuration import MAX_STEPS
from cayu.context.structured_output import StructuredOutputSpec, copy_structured_output_spec
from cayu.providers.retry_policy import RetryPolicy, copy_retry_policy
from cayu.runtime.tool_completion import ToolCompletionPolicy, copy_tool_completion_policy
from cayu.sessions.checkpoints import (
    INVOCATION_CONFIGURATION_CHECKPOINT_KEY as INVOCATION_CONFIGURATION_CHECKPOINT_KEY,
)


class InvocationConfiguration(BaseModel):
    """Profile-bound configuration evidence, never permission to execute a run."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, hide_input_in_errors=True, revalidate_instances="always"
    )

    record_type: Literal["cayu.invocation-configuration"] = "cayu.invocation-configuration"
    schema_version: Literal[1] = 1
    session_id: str
    session_instance_id: str
    execution_profile_fingerprint: str
    structured_output: StructuredOutputSpec | None
    max_steps: StrictInt | None = Field(ge=1, le=MAX_STEPS)
    limits: RunLimits
    retry_policy: RetryPolicy
    tool_completion: ToolCompletionPolicy | None

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_version(cls, value: object) -> int:
        if type(value) is not int or value != 1:
            raise ValueError("Invocation configuration version is unsupported.")
        return value

    @field_validator("session_id", "session_instance_id", "execution_profile_fingerprint")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("structured_output", mode="after")
    @classmethod
    def copy_spec(cls, value: StructuredOutputSpec | None) -> StructuredOutputSpec | None:
        return copy_structured_output_spec(value)

    @field_validator("limits", mode="after")
    @classmethod
    def copy_limits(cls, value: RunLimits) -> RunLimits:
        return copy_run_limits(value)

    @field_validator("retry_policy", mode="after")
    @classmethod
    def copy_retry(cls, value: RetryPolicy) -> RetryPolicy:
        return copy_retry_policy(value)

    @field_validator("tool_completion", mode="after")
    @classmethod
    def copy_completion(cls, value: ToolCompletionPolicy | None) -> ToolCompletionPolicy | None:
        return copy_tool_completion_policy(value)
