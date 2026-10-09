from __future__ import annotations

import asyncio
import os
from pathlib import Path

from examples._advanced_support import ScenarioResult, live_provider
from examples.cache_aware_research_council.scenario import run_scenario

from cayu import CacheBreakpoint, CachePolicy
from cayu.budgets.pricing import load_price_book

# Cache the conversation prefix explicitly so the source's warm request can be
# reused by the cache-aware compactor. The system prompt alone is below the
# smallest cacheable prompt for some Claude models.
ANTHROPIC_CACHE_POLICY = CachePolicy(
    breakpoints=(
        CacheBreakpoint.SYSTEM_PROMPT,
        CacheBreakpoint.TOOL_DEFINITIONS,
        CacheBreakpoint.CONVERSATION_PREFIX,
    )
)


async def run(root: Path, provider_name: str | None = None) -> ScenarioResult:
    selected = (provider_name or os.environ.get("CAYU_ADVANCED_PROVIDER", "gemini")).strip().lower()
    provider, model = live_provider(selected, anthropic_cache_policy=ANTHROPIC_CACHE_POLICY)
    price_book_path = os.environ.get("CAYU_RESEARCH_COUNCIL_PRICE_BOOK")
    price_book = load_price_book(Path(price_book_path)) if price_book_path else None
    return await run_scenario(
        root,
        provider=provider,
        model=model,
        mode="live",
        price_book=price_book,
        # Anthropic reports prompt-cache reads; other providers keep the
        # paired token assertions without a provider-specific cache claim.
        require_compaction_cache_read=selected == "anthropic",
    )


if __name__ == "__main__":
    asyncio.run(run(Path.cwd()))
