from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, field_validator

from cayu._validation import require_clean_nonblank


class ContextCountingMode(StrEnum):
    OFF = "off"
    OBSERVE = "observe"


class ContextCountingConfig(BaseModel):
    """Runtime controls for pre-call input token counting.

    `observe` records provider counts and reconciles them against completed
    model usage when available. It does not enforce budgets or mutate context.
    Provider-backed counters may make an extra provider request, so this is
    observability/debugging infrastructure rather than a default overflow guard.
    """

    model_config = ConfigDict(extra="forbid")

    mode: ContextCountingMode = ContextCountingMode.OFF

    @field_validator("mode", mode="before")
    @classmethod
    def validate_mode(cls, value: object) -> ContextCountingMode:
        if isinstance(value, ContextCountingMode):
            return value
        if not isinstance(value, str):
            raise ValueError("`mode` must be a string.")
        return ContextCountingMode(require_clean_nonblank(value, "mode"))


def copy_context_counting_config(
    config: ContextCountingConfig | None,
) -> ContextCountingConfig:
    if config is None:
        return ContextCountingConfig()
    if type(config) is not ContextCountingConfig:
        raise TypeError("Context counting config must be a ContextCountingConfig instance.")
    return ContextCountingConfig(mode=config.mode)
