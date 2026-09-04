"""Typed, redacted inspection records for effective runtime configuration."""

from __future__ import annotations

from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, StrictInt, field_validator

from cayu.core.thinking import ThinkingConfig
from cayu.runtime.config import CayuConfigSource
from cayu.runtime.execution_profiles import ExecutionProfileIdentity
from cayu.runtime.retry_policy import RetryPolicy

_ValueT = TypeVar("_ValueT")

_INSPECTION_MODEL = ConfigDict(
    extra="forbid",
    frozen=True,
    hide_input_in_errors=True,
)


class EffectiveConfigurationField(BaseModel, Generic[_ValueT]):
    """One resolved non-secret value and its semantic provenance."""

    model_config = _INSPECTION_MODEL

    value: _ValueT
    owner: str
    source: CayuConfigSource


class EffectiveRunLimits(BaseModel):
    """Immutable projection of the mutable request-building ``RunLimits`` type."""

    model_config = _INSPECTION_MODEL

    max_input_tokens: StrictInt | None = None
    max_output_tokens: StrictInt | None = None
    max_total_tokens: StrictInt | None = None
    max_tool_calls: StrictInt | None = None
    max_elapsed_seconds: StrictInt | None = None
    scope: Literal["session", "run"] = "run"


class EffectiveRunConfiguration(BaseModel):
    """Redacted effective run controls and the exact durable profile they produce.

    The record intentionally excludes messages, metadata, callbacks, policies, and
    source inputs. ``execution_profile`` is the Runtime's already-redacted durable
    identity computed by the same preflight used for session admission.
    """

    model_config = _INSPECTION_MODEL

    max_steps: EffectiveConfigurationField[StrictInt]
    limits: EffectiveConfigurationField[EffectiveRunLimits]
    retry_policy: EffectiveConfigurationField[RetryPolicy]
    thinking: EffectiveConfigurationField[ThinkingConfig | None]
    execution_profile: ExecutionProfileIdentity

    @field_validator("execution_profile", mode="before")
    @classmethod
    def detach_execution_profile(cls, value: object) -> ExecutionProfileIdentity:
        if isinstance(value, ExecutionProfileIdentity):
            value = value.model_dump(mode="json", warnings=False)
        return ExecutionProfileIdentity.model_validate(value)
