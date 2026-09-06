"""Local effort compatibility, not backend acceptance or a model allowlist.

The dated source audit and portability boundaries live in guides/thinking.md.
Only exact known aliases and their dated snapshots receive model-specific checks.
Unknown identifiers retain native values and defer acceptance to the backend.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Literal

from cayu.core.thinking import ThinkingConfig

_STANDARD = frozenset({"low", "medium", "high"})
_OPENAI_MODELS = {
    "gpt-5": _STANDARD | {"minimal"},
    "gpt-5-mini": _STANDARD | {"minimal"},
    "gpt-5-nano": _STANDARD | {"minimal"},
    "gpt-5-pro": frozenset({"high"}),
    "gpt-5.1": _STANDARD | {"none"},
    "gpt-5.1-codex": _STANDARD,
    "gpt-5.1-codex-max": _STANDARD | {"xhigh"},
    "gpt-5.2": _STANDARD | {"none", "xhigh"},
    "gpt-5.4": _STANDARD | {"none", "xhigh"},
    "gpt-5.5": _STANDARD | {"none", "xhigh"},
    "gpt-5.6": _STANDARD | {"none", "xhigh", "max"},
    "gpt-5.6-sol": _STANDARD | {"none", "xhigh", "max"},
    "gpt-5.6-terra": _STANDARD | {"none", "xhigh", "max"},
    "gpt-5.6-luna": _STANDARD | {"none", "xhigh", "max"},
    "gpt-6-astra": _STANDARD | {"xhigh", "max"},
    "o1": _STANDARD,
    "o3": _STANDARD,
    "o3-mini": _STANDARD,
    "o4-mini": _STANDARD,
    "gpt-4o": frozenset(),
    "gpt-4o-mini": frozenset(),
    "gpt-4.1": frozenset(),
    "gpt-4.1-mini": frozenset(),
    "gpt-4.1-nano": frozenset(),
}
_ANTHROPIC_MODELS = {
    **dict.fromkeys(
        ("claude-opus-4-6", "claude-sonnet-4-6", "claude-mythos-preview"),
        _STANDARD | {"max"},
    ),
    **dict.fromkeys(
        (
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-opus-5",
            "claude-sonnet-5",
            "claude-fable-5",
            "claude-fable-5-1",
            "claude-mythos-5",
            "claude-mythos-5-1",
        ),
        _STANDARD | {"xhigh", "max"},
    ),
    # These models do not support the adaptive mode selected by typed effort.
    **dict.fromkeys(
        (
            "claude-opus-4-5",
            "claude-sonnet-4-5",
            "claude-haiku-4-5",
            "claude-opus-4-1",
            "claude-opus-4",
            "claude-sonnet-4",
            "claude-3-7-sonnet",
            "claude-3-5-sonnet",
            "claude-3-5-haiku",
        ),
        frozenset(),
    ),
}

_GOOGLE_CHAT_MODELS = {
    **dict.fromkeys(
        (
            "gemini-3.1-pro-preview",
            "gemini-3.1-flash-lite-preview",
            "gemini-3-flash-preview",
            "gemini-2.5-pro",
        ),
        _STANDARD | {"minimal"},
    ),
    **dict.fromkeys(("gemini-2.5-flash", "gemini-2.5-flash-lite"), _STANDARD | {"none", "minimal"}),
}


def validate_thinking_effort(
    options: Mapping[str, Any],
    *,
    protocol: Literal["openai", "anthropic", "chat_completions", "bedrock"],
    model: str = "",
) -> None:
    """Reject incompatible typed controls without echoing user input or secrets.

    No network, model discovery, normalization of effort, or fallback. Recognizable
    model IDs retain their declared semantics even on compatible endpoints. This
    function succeeding establishes local validity only.
    """
    neutral = options.get("thinking")
    if neutral is None:
        return
    try:
        config = ThinkingConfig.model_validate(neutral)
    except ValueError:
        raise ValueError(
            "Invalid thinking configuration; use ThinkingConfig with mutually exclusive controls."
        ) from None
    effort = config.effort
    if effort is None:
        return
    if protocol == "bedrock":
        raise ValueError(
            "Local thinking incompatibility: Bedrock Converse has no typed effort mapping. "
            "Select an adapter with documented thinking effort support."
        )
    # OpenAI snapshots use -YYYY-MM-DD; Anthropic uses -YYYYMMDD or @YYYYMMDD
    # on Vertex. Do not guess semantics for arbitrary suffixes or future families.
    alias = re.sub(r"(?:-\d{4}-\d{2}-\d{2}|[-@]\d{8}|-latest)$", "", model)
    if protocol == "anthropic":
        supported = _ANTHROPIC_MODELS.get(alias, _STANDARD | {"xhigh", "max"})
    else:
        supported = _OPENAI_MODELS.get(alias)
        if protocol == "chat_completions" and alias in {
            "gpt-5-pro",
            "gpt-5.1-codex",
            "gpt-5.1-codex-max",
        }:
            supported = frozenset()
        if protocol == "chat_completions" and alias in _GOOGLE_CHAT_MODELS:
            supported = _GOOGLE_CHAT_MODELS[alias]
    if supported is not None and effort not in supported:
        allowed = ", ".join(sorted(supported)) or "no typed effort on this model/transport"
        raise ValueError(
            f"Local thinking incompatibility: {protocol} does not support the requested effort "
            f"for this model/transport; allowed: {allowed}. "
            "See cayu guide thinking."
        )
