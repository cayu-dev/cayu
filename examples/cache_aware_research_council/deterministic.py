from __future__ import annotations

import asyncio
from decimal import Decimal
from pathlib import Path

from examples._advanced_support import ScenarioResult, completed_batch, structured_batch
from examples.cache_aware_research_council.scenario import run_scenario

from cayu import (
    ModelCatalog,
    ModelInfo,
    PriceTier,
    Provenance,
    ScriptedModelProvider,
    TieredPricing,
)


async def run(root: Path) -> ScenarioResult:
    reports = [
        {
            "strategy": "primary-source-audit",
            "claim": "Checkpoint forks can reuse prepared context while preserving lineage.",
            "evidence": ["runtime fork event", "shared causal budget"],
            "uncertainties": ["provider cache lifetime varies"],
        },
        {
            "strategy": "contrarian-cost-check",
            "claim": "Forking only saves money when shared input dominates branch-specific work.",
            "evidence": ["paired baseline required", "cached input metrics required"],
            "uncertainties": ["no paired baseline is present yet"],
        },
        {
            "strategy": "operator-recovery-review",
            "claim": "Durable checkpoints make branch evaluation recoverable.",
            "evidence": ["persisted session lineage", "restartable child session"],
            "uncertainties": ["promotion remains application-owned"],
        },
    ]
    provider = ScriptedModelProvider(
        [
            completed_batch("Shared research context prepared."),
            completed_batch("Compacted checkpoint prepared for branch creation."),
            *[
                structured_batch(
                    report,
                    call_id=f"baseline-report-{index}",
                    input_tokens=240,
                    output_tokens=10,
                )
                for index, report in enumerate(reports, start=1)
            ],
            *[
                structured_batch(
                    report,
                    call_id=f"report-{index}",
                    input_tokens=80,
                    output_tokens=10,
                )
                for index, report in enumerate(reports, start=1)
            ],
            structured_batch(
                {
                    "winner": "contrarian-cost-check",
                    "weakness": "The council lacks a paired baseline for the cost claim.",
                    "repair_instruction": "Add a paired baseline and state remaining uncertainty.",
                },
                call_id="evaluation",
            ),
            structured_batch(
                {
                    "fixed_weakness": "Added the missing paired baseline comparison.",
                    "added_evidence": ["baseline and candidate record the same prepared context"],
                    "remaining_uncertainty": "Live provider cache metrics still require calibration.",
                },
                call_id="repair",
            ),
        ]
    )
    model_catalog = ModelCatalog(
        catalog_version="deterministic-fixture-v1",
        generated_at="2026-01-01T00:00:00Z",
        models=(
            ModelInfo(
                provider_name="scripted",
                model="scripted-model",
                match="exact",
                pricing=TieredPricing(
                    standard=(
                        PriceTier(
                            input_per_million=Decimal("1.00"),
                            output_per_million=Decimal("5.00"),
                        ),
                    ),
                ),
                provenance=Provenance(
                    source="deterministic fixture; not provider pricing",
                    url="https://example.invalid/cayu/research-council-pricing-fixture",
                    as_of="2026-01-01",
                ),
            ),
        ),
    )
    return await run_scenario(
        root,
        provider=provider,
        model="scripted-model",
        mode="deterministic",
        model_catalog=model_catalog,
    )


if __name__ == "__main__":
    asyncio.run(run(Path.cwd()))
