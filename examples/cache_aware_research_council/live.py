from __future__ import annotations

import asyncio
from pathlib import Path

from examples._advanced_support import ScenarioResult, live_provider
from examples.cache_aware_research_council.scenario import run_scenario


async def run(root: Path, provider_name: str | None = None) -> ScenarioResult:
    provider, model = live_provider(provider_name)
    return await run_scenario(root, provider=provider, model=model, mode="live")


if __name__ == "__main__":
    asyncio.run(run(Path.cwd()))
