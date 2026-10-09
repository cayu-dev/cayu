from __future__ import annotations

import os
from pathlib import Path

from examples._advanced_support import ScenarioResult
from examples.late_system_message_caching.scenario import (
    HAS_CACHE_AFFINITY_FIELD,
    LIVE_MIN_CACHED_SHARE,
    SCENARIO,
    VARIANTS,
    run_conversation,
    summarize,
)

from cayu import AnthropicProvider, OpenAIProvider

# Price-book model IDs, so every turn is priced by cayu.default_price_book().
_MODELS = {"openai": "gpt-6-luna", "anthropic": "claude-haiku-4-5"}
_OPTIONS = {
    "openai": {"openai": {"reasoning": {"effort": "low"}, "max_output_tokens": 400}},
    "anthropic": {"anthropic": {"max_tokens": 200}},
}
_KEY_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


def _provider(name: str) -> OpenAIProvider | AnthropicProvider:
    # Default construction on purpose: the measurement is what an application
    # gets without tuning cache settings.
    if name == "openai":
        return OpenAIProvider()
    return AnthropicProvider()


async def run(root: Path, provider_name: str | None = None) -> ScenarioResult:
    selected = (provider_name or "openai").strip().lower()
    if selected not in _MODELS:
        raise ValueError("Late-system-message caching runs with --provider openai or anthropic.")
    if not os.environ.get(_KEY_ENV[selected]):
        raise RuntimeError(f"Set {_KEY_ENV[selected]} to run this live measurement.")
    model = os.environ.get("CAYU_LATE_SYSTEM_CACHE_MODEL", _MODELS[selected])
    provider = _provider(selected)
    try:
        runs = [
            await run_conversation(
                provider,
                provider_name=selected,
                model=model,
                variant=variant,
                options=_OPTIONS[selected],
            )
            for variant in VARIANTS
        ]
    finally:
        await provider.aclose()

    summary = summarize(runs)
    late = summary["late_system"]
    late_turns = late["turns"][1:]
    assertions = {
        "late_system_turns_2_plus_read_most_input_from_cache": all(
            turn["cached_share"] >= LIVE_MIN_CACHED_SHARE for turn in late_turns
        ),
        "every_turn_reported_usage": all(
            turn["input_tokens"] > 0 for run in summary.values() for turn in run["turns"]
        ),
    }
    result = ScenarioResult(
        scenario=SCENARIO,
        mode="live",
        status="verified" if all(assertions.values()) else "failed",
        assertions=assertions,
        sessions=[],
        provider_name=selected,
        model=model,
        metrics={
            "variants": summary,
            "cache_affinity_key_field": HAS_CACHE_AFFINITY_FIELD,
            "min_cached_share_required": LIVE_MIN_CACHED_SHARE,
        },
    )
    result.write(root)
    result.require_verified()
    return result
