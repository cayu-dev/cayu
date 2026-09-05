"""Cohesive application-owned defaults for the Cayu Runtime.

This module contains the small public configuration surface that callers may
reasonably tune for an application. Protocol constants, durable schema
versions, and internal safety bounds remain owned by their feature modules.
"""

from __future__ import annotations

from math import isfinite
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from cayu._eval_limits import EVAL_SUITE_MAX_CONCURRENCY
from cayu.artifacts.attachments import (
    DEFAULT_MAX_FILE_ATTACHMENT_BYTES,
    DEFAULT_MAX_FILE_ATTACHMENTS_PER_REQUEST,
    DEFAULT_MAX_TOTAL_FILE_ATTACHMENT_BYTES,
)
from cayu.core.thinking import ThinkingConfig
from cayu.runtime.recovery_cleanup import (
    RecoveryCleanupPolicy,
    copy_recovery_cleanup_policy,
)
from cayu.runtime.retry_policy import RetryPolicy, copy_retry_policy
from cayu.runtime.stop_policy import RunLimits

DEFAULT_MAX_STEPS = 64
MAX_STEPS = 256
DEFAULT_MAX_PARALLEL_TOOL_CALLS = 4
DEFAULT_MAX_ENVIRONMENT_LIFECYCLE_OWNERS = 256

CayuConfigSource = Literal["framework", "application", "explicit"]


_CONFIG_MODEL = ConfigDict(
    extra="forbid",
    frozen=True,
    hide_input_in_errors=True,
    validate_default=True,
)


class _FrozenRunLimits(RunLimits):
    """The immutable representation reachable through application config."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


def _copy_thinking_config(config: ThinkingConfig | None) -> ThinkingConfig | None:
    if config is None:
        return None
    return ThinkingConfig.model_validate(config.model_dump(mode="python", warnings=False))


class RunDefaults(BaseModel):
    """Defaults applied when one run or resume request omits an override."""

    model_config = _CONFIG_MODEL

    max_steps: StrictInt = Field(default=DEFAULT_MAX_STEPS, ge=1, le=MAX_STEPS)
    limits: RunLimits = Field(default_factory=RunLimits)
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    thinking: ThinkingConfig | None = None

    @field_validator("limits", mode="after")
    @classmethod
    def detach_limits(cls, value: RunLimits) -> RunLimits:
        # RunLimits remains mutable for request-building compatibility. Freeze a
        # detached subtype here so the CayuConfig object graph is immutable.
        return _FrozenRunLimits.model_validate(value.model_dump(mode="python", warnings=False))

    @field_validator("retry_policy", mode="after")
    @classmethod
    def detach_retry_policy(cls, value: RetryPolicy) -> RetryPolicy:
        return copy_retry_policy(value)

    @field_validator("thinking", mode="after")
    @classmethod
    def detach_thinking(cls, value: ThinkingConfig | None) -> ThinkingConfig | None:
        return _copy_thinking_config(value)

    def copy_limits(self) -> RunLimits:
        """Return ordinary mutable request limits detached from this config."""

        return RunLimits.model_validate(self.limits.model_dump(mode="python", warnings=False))


class ToolExecutionConfig(BaseModel):
    """Application-wide attachment and tool-execution defaults."""

    model_config = _CONFIG_MODEL

    max_file_attachment_bytes: StrictInt = Field(
        default=DEFAULT_MAX_FILE_ATTACHMENT_BYTES,
        ge=1,
    )
    max_total_file_attachment_bytes: StrictInt = Field(
        default=DEFAULT_MAX_TOTAL_FILE_ATTACHMENT_BYTES,
        ge=1,
    )
    max_file_attachments_per_request: StrictInt = Field(
        default=DEFAULT_MAX_FILE_ATTACHMENTS_PER_REQUEST,
        ge=1,
    )
    tool_timeout_seconds: float | None = None
    max_parallel_tool_calls: StrictInt = Field(
        default=DEFAULT_MAX_PARALLEL_TOOL_CALLS,
        ge=1,
    )

    @field_validator("tool_timeout_seconds", mode="before")
    @classmethod
    def validate_tool_timeout_seconds(cls, value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("tool_timeout_seconds must be a number.")
        resolved = float(value)
        if not isfinite(resolved) or resolved <= 0:
            raise ValueError("tool_timeout_seconds must be a finite positive number.")
        return resolved


class EvalConfig(BaseModel):
    """Defaults for new evaluation runs without an explicit suite policy."""

    model_config = _CONFIG_MODEL

    max_concurrency: StrictInt = Field(default=1, ge=1, le=EVAL_SUITE_MAX_CONCURRENCY)


class OperationsConfig(BaseModel):
    """Application-wide bounds for Runtime-owned operational work."""

    model_config = _CONFIG_MODEL

    max_environment_lifecycle_owners: StrictInt = Field(
        default=DEFAULT_MAX_ENVIRONMENT_LIFECYCLE_OWNERS,
        ge=1,
    )
    recovery_cleanup_policy: RecoveryCleanupPolicy = Field(
        default_factory=RecoveryCleanupPolicy,
    )

    @field_validator("recovery_cleanup_policy", mode="after")
    @classmethod
    def detach_recovery_cleanup_policy(
        cls,
        value: RecoveryCleanupPolicy,
    ) -> RecoveryCleanupPolicy:
        return copy_recovery_cleanup_policy(value)


class CayuConfig(BaseModel):
    """Small, immutable root configuration for one :class:`CayuApp`."""

    model_config = _CONFIG_MODEL

    run: RunDefaults = Field(default_factory=RunDefaults)
    tool_execution: ToolExecutionConfig = Field(default_factory=ToolExecutionConfig)
    operations: OperationsConfig = Field(default_factory=OperationsConfig)
    evals: EvalConfig = Field(default_factory=EvalConfig)

    @field_validator("run", mode="after")
    @classmethod
    def detach_run_defaults(cls, value: RunDefaults) -> RunDefaults:
        return RunDefaults.model_validate(
            {field_name: getattr(value, field_name) for field_name in value.model_fields_set}
        )

    @field_validator("tool_execution", mode="after")
    @classmethod
    def detach_tool_execution_config(
        cls,
        value: ToolExecutionConfig,
    ) -> ToolExecutionConfig:
        return ToolExecutionConfig.model_validate(
            {field_name: getattr(value, field_name) for field_name in value.model_fields_set}
        )

    @field_validator("evals", mode="after")
    @classmethod
    def detach_evals(cls, value: EvalConfig) -> EvalConfig:
        return EvalConfig.model_validate(
            {field_name: getattr(value, field_name) for field_name in value.model_fields_set}
        )

    @field_validator("operations", mode="after")
    @classmethod
    def detach_operations_config(cls, value: OperationsConfig) -> OperationsConfig:
        return OperationsConfig.model_validate(
            {field_name: getattr(value, field_name) for field_name in value.model_fields_set}
        )


def copy_cayu_config(config: CayuConfig | None) -> CayuConfig:
    """Return a detached, validated application configuration."""

    if config is None:
        return CayuConfig()
    if type(config) is not CayuConfig:
        raise TypeError("config must be a CayuConfig.")
    return CayuConfig.model_validate(
        config.model_dump(mode="python", warnings=False),
    )
