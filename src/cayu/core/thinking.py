from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, StrictBool, StrictInt, field_validator, model_validator

ThinkingEffort = Literal["low", "medium", "high"]

MIN_THINKING_BUDGET_TOKENS = 1024


class ThinkingConfig(BaseModel):
    """Provider-neutral thinking/reasoning configuration.

    The runtime carries this as a neutral ``options["thinking"]`` payload and each
    provider maps it to its own request shape (field-driven, no model lookup):

    - ``effort`` set -> adaptive thinking (Anthropic ``thinking={"type": "adaptive"}``
      + ``output_config={"effort": ...}``; OpenAI ``reasoning={"effort": ...}``). This
      is the path the current Claude and OpenAI reasoning models use.
    - ``max_tokens`` set (and no ``effort``) -> legacy budgeted thinking (Anthropic
      ``thinking={"type": "enabled", "budget_tokens": ...}``). Only the older Claude
      models accept a token budget; OpenAI has no budget knob and ignores it.
    - neither set, ``enabled=True`` -> adaptive thinking with provider defaults (on the
      Chat Completions path this is a no-op: the model reasons by its own default).
    - ``enabled=False`` is best-effort and provider-dependent: Anthropic disables
      (``thinking={"type": "disabled"}``); OpenAI reasoning models cannot be disabled, so
      it is a no-op (the model still reasons); the generic Chat Completions adapter also
      no-ops (disabling isn't portable — pass a raw ``reasoning_effort`` via
      ``provider_options`` to target a backend like Gemini that accepts ``"none"``).

    ``effort`` and ``max_tokens`` select mutually exclusive modes. Disabling thinking
    cannot be combined with either enabled-mode control. Provider-specific support is
    still validated by the provider; contradictory neutral controls fail here before a
    provider request can be built.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

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
