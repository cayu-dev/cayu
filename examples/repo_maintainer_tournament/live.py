from __future__ import annotations

import asyncio
from pathlib import Path

from examples._advanced_support import ScenarioResult, live_provider
from examples.repo_maintainer_tournament.real_repository import (
    LiveRepositoryConfig,
    RealRepositoryBoundary,
    promote_tournament_result,
)
from examples.repo_maintainer_tournament.scenario import RepositorySourceContext, run_scenario


async def run(root: Path, provider_name: str | None = None) -> ScenarioResult:
    provider, model = live_provider(provider_name)
    repository = LiveRepositoryConfig.from_environment()
    if repository is None:
        return await run_scenario(root, provider=provider, model=model, mode="live")
    boundary = RealRepositoryBoundary(repository)
    preparation = await boundary.prepare(root)
    result = await run_scenario(
        root,
        provider=provider,
        model=model,
        mode="live",
        source_context=RepositorySourceContext(
            pull=preparation.pull,
            files=preparation.files,
            baseline=preparation.baseline,
        ),
    )
    return await promote_tournament_result(
        result,
        root=root,
        boundary=boundary,
        preparation=preparation,
    )


if __name__ == "__main__":
    asyncio.run(run(Path.cwd()))
