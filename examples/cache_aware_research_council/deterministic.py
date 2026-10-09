from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterable
from decimal import Decimal
from pathlib import Path

from examples._advanced_support import ScenarioResult, completed_batch, structured_batch
from examples.cache_aware_research_council.scenario import run_scenario

from cayu import (
    ModelPrice,
    PriceBook,
    Provenance,
    ScriptedModelProvider,
)
from cayu.providers import ModelRequest, ModelStreamEvent

# Deterministic stand-in for a provider tokenizer: about four characters per token.
_CHARS_PER_TOKEN = 4

ScriptedResponse = Callable[[int], list[ModelStreamEvent]]


def request_input_tokens(request: ModelRequest) -> int:
    """Return deterministic input usage derived from the exact request sent.

    Fixed scripted usage cannot tell a compacted branch from one that silently
    received the full transcript. Measuring the serialized model-facing request
    keeps the paired token assertions tied to what the runtime actually sent.
    """

    serialized = json.dumps(
        {
            "messages": [message.model_dump(mode="json") for message in request.messages],
            "tools": request.tools,
        },
        sort_keys=True,
    )
    return max(1, len(serialized) // _CHARS_PER_TOKEN)


def request_sized_provider(responses: Iterable[ScriptedResponse]) -> ScriptedModelProvider:
    """Replay scripted responses in order, reporting request-derived input usage."""

    pending = iter(responses)

    def respond(request: ModelRequest) -> list[ModelStreamEvent]:
        response = next(pending, None)
        if response is None:
            raise AssertionError("No scripted research-council response remains.")
        return response(request_input_tokens(request))

    return ScriptedModelProvider(response_factory=respond)


def _text(text: str, *, cache_read_input_tokens: int | None = None) -> ScriptedResponse:
    if cache_read_input_tokens is None:
        return lambda input_tokens: completed_batch(text, input_tokens=input_tokens)

    def respond(input_tokens: int) -> list[ModelStreamEvent]:
        return [
            ModelStreamEvent.text_delta(text),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": 5,
                        "total_tokens": input_tokens + 5,
                        "cache_read_input_tokens": cache_read_input_tokens,
                        "cache_creation_input_tokens": 0,
                    },
                }
            ),
        ]

    return respond


def _structured(output: dict[str, object], *, call_id: str) -> ScriptedResponse:
    return lambda input_tokens: structured_batch(
        output,
        call_id=call_id,
        input_tokens=input_tokens,
        output_tokens=10,
    )


def scripted_provider(
    *,
    compaction_cache_read_input_tokens: int | None = None,
) -> ScriptedModelProvider:
    """Build the scripted council provider.

    ``compaction_cache_read_input_tokens`` makes the compaction call report a
    provider prompt-cache read, so tests can exercise the live-only assertion.
    """

    reports: list[dict[str, object]] = [
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
    return request_sized_provider(
        [
            _text("Shared research context prepared."),
            _text(
                "1. Which lineage evidence survives recovery? 2. What do cache receipts "
                "prove? 3. When does branching save cost?"
            ),
            _text(
                "Research brief: compare durable agent runtimes on lineage, recovery "
                "receipts, and cache accounting under one prompt and provider "
                "configuration. The evidence notebook repeats that requirement in 100 "
                "observations.",
                cache_read_input_tokens=compaction_cache_read_input_tokens,
            ),
            _text("Compacted checkpoint prepared for branch creation."),
            *[
                _structured(report, call_id=f"baseline-report-{index}")
                for index, report in enumerate(reports, start=1)
            ],
            *[
                _structured(report, call_id=f"report-{index}")
                for index, report in enumerate(reports, start=1)
            ],
            _structured(
                {
                    "winner": "contrarian-cost-check",
                    "weakness": "The council lacks a paired baseline for the cost claim.",
                    "repair_instruction": "Add a paired baseline and state remaining uncertainty.",
                },
                call_id="evaluation",
            ),
            _structured(
                {
                    "fixed_weakness": "Added the missing paired baseline comparison.",
                    "added_evidence": ["baseline and candidate record the same prepared context"],
                    "remaining_uncertainty": "Live provider cache metrics still require calibration.",
                },
                call_id="repair",
            ),
        ]
    )


async def run(
    root: Path,
    *,
    provider: ScriptedModelProvider | None = None,
    require_compaction_cache_read: bool = False,
) -> ScenarioResult:
    price_book = PriceBook(
        price_book_version="deterministic-fixture-v1",
        generated_at="2026-01-01T00:00:00Z",
        prices=(
            ModelPrice.fixed(
                provider_name="scripted",
                model="scripted-model",
                match="exact",
                input_per_million=Decimal("1.00"),
                output_per_million=Decimal("5.00"),
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
        provider=scripted_provider() if provider is None else provider,
        model="scripted-model",
        mode="deterministic",
        price_book=price_book,
        require_compaction_cache_read=require_compaction_cache_read,
    )


if __name__ == "__main__":
    asyncio.run(run(Path.cwd()))
