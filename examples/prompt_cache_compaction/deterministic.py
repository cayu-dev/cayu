from __future__ import annotations

import asyncio
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

from examples.prompt_cache_compaction.scenario import run_scenario

from cayu import (
    ModelPrice,
    ModelStreamEvent,
    PriceBook,
    Provenance,
    ScriptedModelProvider,
)

if TYPE_CHECKING:
    from examples._advanced_support import ScenarioResult


def _completed(
    text: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_counters_reported: bool = False,
):
    usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    if cache_read_tokens or cache_counters_reported:
        usage["cache_read_input_tokens"] = cache_read_tokens
        usage["cache_creation_input_tokens"] = 0
    return [
        ModelStreamEvent.text_delta(text),
        ModelStreamEvent.completed({"finish_reason": "stop", "usage": usage}),
    ]


async def run(root: Path) -> ScenarioResult:
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="load-context",
                    name="load_stable_context",
                    arguments={"topic": "cache contracts"},
                ),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "tool_use",
                        "usage": {"input_tokens": 20, "output_tokens": 4},
                    }
                ),
            ],
            _completed("Context loaded.", input_tokens=1200, output_tokens=4),
            _completed("Retain concrete constraints.", input_tokens=1240, output_tokens=5),
            _completed(
                "CACHE_RETENTION_OK. The stable context and pending work are retained.",
                input_tokens=40,
                output_tokens=8,
                cache_read_tokens=1200,
            ),
            _completed("Retain pending work too.", input_tokens=60, output_tokens=5),
            _completed(
                "CACHE_RETENTION_OK. The summary includes the next retained constraint.",
                input_tokens=55,
                output_tokens=9,
            ),
            _completed("CACHE_RETENTION_OK", input_tokens=62, output_tokens=5),
        ]
    )
    baseline_provider = ScriptedModelProvider(
        [
            _completed(
                "CACHE_RETENTION_OK. Bounded baseline summary.",
                input_tokens=1220,
                output_tokens=8,
                cache_counters_reported=True,
            )
        ]
    )
    price_book = PriceBook(
        price_book_version="deterministic-fixture-v1",
        generated_at="2026-01-01T00:00:00Z",
        prices=(
            ModelPrice.fixed(
                provider_name="scripted",
                model="scripted-model",
                input_per_million=Decimal("3.00"),
                output_per_million=Decimal("15.00"),
                cache_read_input_per_million=Decimal("0.30"),
                cache_write_input_per_million=Decimal("3.75"),
                provenance=Provenance(
                    source="deterministic fixture; not provider pricing",
                    url="https://example.invalid/cayu/prompt-cache-pricing-fixture",
                    as_of="2026-01-01",
                ),
            ),
        ),
    )
    return await run_scenario(
        root,
        provider=provider,
        baseline_provider=baseline_provider,
        model="scripted-model",
        mode="deterministic",
        stable_context_lines=40,
        price_book=price_book,
    )


if __name__ == "__main__":
    asyncio.run(run(Path.cwd()))
