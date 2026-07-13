from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING

from examples.prompt_cache_compaction.scenario import run_scenario

from cayu import (
    AnthropicProvider,
    CacheBreakpoint,
    CachePolicy,
    ThinkingConfig,
)
from cayu.runtime import load_model_catalog

if TYPE_CHECKING:
    from examples._advanced_support import ScenarioResult

_DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
_BUDGETED_THINKING_MODELS = frozenset({_DEFAULT_ANTHROPIC_MODEL})


def _thinking_for_model(model: str, *, mode: str | None = None) -> ThinkingConfig:
    selected = mode or ("budgeted" if model in _BUDGETED_THINKING_MODELS else "adaptive")
    if selected == "budgeted":
        return ThinkingConfig(max_tokens=1024)
    if selected == "adaptive":
        return ThinkingConfig(effort="low")
    raise ValueError("thinking mode must be 'budgeted' or 'adaptive'.")


async def run(root: Path, provider_name: str | None = None) -> ScenarioResult:
    selected = (provider_name or "anthropic").strip().lower()
    if selected != "anthropic":
        raise ValueError("Prompt-cache compaction live verification requires --provider anthropic.")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("Set ANTHROPIC_API_KEY to run this live verification.")
    lines = int(os.environ.get("CAYU_ANTHROPIC_CACHE_LINES", "400"))
    model = os.environ.get("CAYU_ANTHROPIC_MODEL", _DEFAULT_ANTHROPIC_MODEL)
    thinking = _thinking_for_model(
        model,
        mode=os.environ.get("CAYU_ANTHROPIC_THINKING_MODE"),
    )
    provider = AnthropicProvider(
        cache_policy=CachePolicy(
            # Keep the live assertion specific: a nonzero read can only come from the
            # conversation/tool-result prefix, not an independently cached system or tool tier.
            breakpoints=(CacheBreakpoint.CONVERSATION_PREFIX,)
        )
    )
    baseline_provider = AnthropicProvider()
    catalog_path = os.environ.get("CAYU_PROMPT_CACHE_MODEL_CATALOG")
    model_catalog = load_model_catalog(Path(catalog_path)) if catalog_path else None
    return await run_scenario(
        root,
        provider=provider,
        baseline_provider=baseline_provider,
        model=model,
        mode="live",
        stable_context_lines=lines,
        provider_options={
            "anthropic": {"max_tokens": 2048 if thinking.max_tokens is not None else 512}
        },
        system_prompt_suffix=(
            " The stable tool output is deliberately large enough to exercise Anthropic's "
            "cache counters; do not repeat it in responses."
        ),
        thinking=thinking,
        model_catalog=model_catalog,
    )


if __name__ == "__main__":
    asyncio.run(run(Path.cwd()))
