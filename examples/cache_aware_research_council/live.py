from __future__ import annotations

import asyncio
import os
from pathlib import Path

from examples._advanced_support import ScenarioResult, live_provider
from examples.cache_aware_research_council.scenario import run_scenario

from cayu.runtime import load_price_book


async def run(root: Path, provider_name: str | None = None) -> ScenarioResult:
    provider, model = live_provider(provider_name)
    price_book_path = os.environ.get("CAYU_RESEARCH_COUNCIL_PRICE_BOOK")
    price_book = load_price_book(Path(price_book_path)) if price_book_path else None
    return await run_scenario(
        root,
        provider=provider,
        model=model,
        mode="live",
        price_book=price_book,
    )


if __name__ == "__main__":
    asyncio.run(run(Path.cwd()))
