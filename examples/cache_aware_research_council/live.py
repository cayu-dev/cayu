from __future__ import annotations

import asyncio
import os
from pathlib import Path

from examples._advanced_support import ScenarioResult, live_provider
from examples.cache_aware_research_council.scenario import run_scenario

from cayu.runtime import load_model_catalog


async def run(root: Path, provider_name: str | None = None) -> ScenarioResult:
    provider, model = live_provider(provider_name)
    catalog_path = os.environ.get("CAYU_RESEARCH_COUNCIL_MODEL_CATALOG")
    model_catalog = load_model_catalog(Path(catalog_path)) if catalog_path else None
    return await run_scenario(
        root,
        provider=provider,
        model=model,
        mode="live",
        model_catalog=model_catalog,
    )


if __name__ == "__main__":
    asyncio.run(run(Path.cwd()))
