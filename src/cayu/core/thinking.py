from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, StrictBool, StrictInt, field_validator, model_validator

ThinkingEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]

MIN_THINKING_BUDGET_TOKENS = 1024


class ThinkingConfig(BaseModel):
    """Provider-neutral thinking/reasoning configuration.

    Effort names are native values, not a portable scale. Supported vocabulary is
    ``none``, ``minimal``, ``low``, ``medium``, ``high``, ``xhigh``, and ``max``;
    adapters reject known incompatible model/transport combinations before dispatch.
    Unknown model compatibility remains backend-dependent. ``xlow`` is not a verified
    native value in the maintained adapters and is not an alias for ``minimal``.

    Explicit effort maps unchanged to OpenAI ``reasoning.effort``, compatible Chat
    Completions ``reasoning_effort``, or Anthropic adaptive thinking plus
    ``output_config.effort``. Bedrock Converse has no typed effort mapping and rejects
    it. See ``cayu guide thinking`` for the matrix.

    ``none`` requests the backend's native non-reasoning effort; ``minimal`` still
    permits reasoning. Both require ``enabled=True`` (the default), which enables
    this configuration, not a promise that the model thinks. Neither is silently
    mapped to ``enabled=False``. The latter retains its historical best-effort
    behavior: Anthropic disables thinking; OpenAI and generic Chat Completions no-op.

    ``effort`` and ``max_tokens`` are mutually exclusive. A token budget selects
    Anthropic legacy thinking (OpenAI and generic Chat Completions historically
    ignore it). Disabling thinking cannot be combined with either control. With
    neither control, enabled thinking uses existing provider defaults. Typed effort
    wins over conflicting raw effort; unrelated provider options remain intact.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
    )

    enabled: StrictBool = True
    effort: ThinkingEffort | None = None
    max_tokens: StrictInt | None = None
    # False keeps newly-produced readable reasoning out of the persisted transcript. It
    # cannot suppress everything: live ``model.thinking.delta`` events still stream, and an
    # Anthropic signed block is retained verbatim (its signature is needed to continue a
    # tool-use loop), so that block stays in the transcript.
    include_in_transcript: StrictBool = True

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value < MIN_THINKING_BUDGET_TOKENS:
            raise ValueError(f"thinking max_tokens must be at least {MIN_THINKING_BUDGET_TOKENS}.")
        return value

    @model_validator(mode="after")
    def validate_mode(self) -> ThinkingConfig:
        if not self.enabled and (self.effort is not None or self.max_tokens is not None):
            raise ValueError("thinking enabled=False cannot be combined with effort or max_tokens.")
        if self.effort is not None and self.max_tokens is not None:
            raise ValueError("thinking effort and max_tokens are mutually exclusive.")
        return self


def thinking_config_payload(config: ThinkingConfig) -> dict[str, Any]:
    """The neutral ``options["thinking"]`` payload each provider maps from."""
    return config.model_dump()
