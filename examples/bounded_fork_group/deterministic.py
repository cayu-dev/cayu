from __future__ import annotations

import asyncio
from pathlib import Path

from examples._advanced_support import ScenarioResult, completed_batch, structured_batch
from examples.bounded_fork_group.scenario import run_scenario

from cayu import ScriptedModelProvider


async def run(root: Path) -> ScenarioResult:
    provider = ScriptedModelProvider(
        [
            completed_batch("Shared source prepared."),
            structured_batch(
                {
                    "proposal": "small focused implementation",
                    "quality": 9,
                    "risk": "low",
                },
                call_id="focused-candidate",
            ),
            structured_batch(
                {
                    "proposal": "broader extensible implementation",
                    "quality": 8,
                    "risk": "medium",
                },
                call_id="extensible-candidate",
            ),
            structured_batch(
                {
                    "proposal": "safer bounded extensible replacement",
                    "quality": 8,
                    "risk": "low",
                },
                call_id="extensible-replacement",
            ),
            structured_batch(
                {
                    "dispositions": [
                        {
                            "branch_id": "focused",
                            "disposition": "selected",
                            "reason": "Highest quality with the lower declared risk.",
                        },
                        {
                            "branch_id": "extensible",
                            "disposition": "rejected",
                            "reason": "Valid, but broader than this bounded task requires.",
                        },
                    ]
                },
                call_id="fork-group-judgment",
            ),
        ]
    )
    return await run_scenario(
        root,
        provider=provider,
        model="scripted-model",
        mode="deterministic",
    )


if __name__ == "__main__":
    asyncio.run(run(Path.cwd()))
