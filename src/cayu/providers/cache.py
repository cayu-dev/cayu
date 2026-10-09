from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from cayu._validation import copy_json_value


class CacheBreakpoint(StrEnum):
    SYSTEM_PROMPT = "system_prompt"
    TOOL_DEFINITIONS = "tool_definitions"
    CONVERSATION_PREFIX = "conversation_prefix"


class CachePolicy(BaseModel):
    """Controls prompt-cache marker placement for Anthropic-shaped providers.

    A breakpoint marks the end of a stable, cacheable prefix. The default marks the
    system prompt, the tool definitions and the conversation history before the
    newest message, so a growing conversation reuses everything it already sent.
    That is three of the four markers Anthropic allows per request.
    ``AnthropicProvider`` and ``VertexProvider`` apply it as ``cache_control``;
    ``BedrockProvider`` applies it as Converse ``cachePoint`` blocks.

    Disable caching with ``CachePolicy(breakpoints=())``, or per request with
    ``options["cache_policy"] = {"breakpoints": []}``.
    """

    model_config = ConfigDict(extra="forbid")

    breakpoints: tuple[CacheBreakpoint, ...] = (
        CacheBreakpoint.SYSTEM_PROMPT,
        CacheBreakpoint.TOOL_DEFINITIONS,
        CacheBreakpoint.CONVERSATION_PREFIX,
    )
    conversation_prefix_strategy: Literal["all_but_last", "all_but_last_n", "none"] = "all_but_last"
    conversation_prefix_n: StrictInt = Field(default=1, ge=1)
    ttl: Literal["standard", "extended"] | None = None

    @property
    def uses_extended_ttl(self) -> bool:
        return self.ttl == "extended"

    def marker(self) -> dict[str, str]:
        if self.uses_extended_ttl:
            return {"type": "ephemeral", "ttl": "1h"}
        return {"type": "ephemeral"}


class RequestCacheProjection(BaseModel):
    """Ephemeral provider-owned cache evidence for one prepared request.

    ``conversation_prefix`` contains only the projected message material through
    the marker the adapter actually applied. The runtime may fingerprint this
    value, but it never persists the value itself.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    policy: CachePolicy
    conversation_prefix: tuple[dict[str, Any], ...] | None = None

    @field_validator("conversation_prefix", mode="before")
    @classmethod
    def copy_conversation_prefix(cls, value: object) -> object:
        if value is None:
            return None
        if not isinstance(value, list | tuple):
            raise TypeError("conversation_prefix must be a list or tuple of JSON objects.")
        copied = copy_json_value(list(value), "request cache conversation prefix")
        if not copied or any(type(item) is not dict for item in copied):
            raise ValueError("conversation_prefix must contain one or more JSON objects.")
        return tuple(copied)

    @model_validator(mode="after")
    def validate_conversation_prefix(self) -> RequestCacheProjection:
        if (
            self.conversation_prefix is not None
            and CacheBreakpoint.CONVERSATION_PREFIX not in self.policy.breakpoints
        ):
            raise ValueError(
                "conversation_prefix requires an effective conversation-prefix breakpoint."
            )
        return self


def resolve_cache_policy(
    default: CachePolicy | None,
    options: Mapping[str, Any],
) -> CachePolicy | None:
    """Pick the effective policy: a per-request ``options['cache_policy']`` override
    (a mapping, since request options are JSON-only) is merged field-by-field onto the
    provider default, so overriding one field (e.g. ``ttl``) does not silently reset the
    others. With no provider default the override stands alone."""
    override = options.get("cache_policy")
    if override is None:
        return default
    if not isinstance(override, Mapping):
        raise ValueError("ModelRequest options['cache_policy'] must be a mapping.")
    if default is None:
        return CachePolicy.model_validate(dict(override))
    return CachePolicy.model_validate({**default.model_dump(), **dict(override)})
