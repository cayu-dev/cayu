from __future__ import annotations

import asyncio
from pathlib import Path

from examples.cache_aware_research_council.deterministic import run


def test_research_council_forks_strategies_and_repairs_seeded_weakness(
    tmp_path: Path,
) -> None:
    result = asyncio.run(run(tmp_path))

    assert result.status == "verified"
    assert result.scenario == "cache-aware-research-council"
    assert result.assertions == {
        "causal_budget_shared": True,
        "compaction_checkpoint_persisted": True,
        "paired_baseline_recorded": True,
        "compacted_candidate_first_attempt_context_is_smaller": True,
        "compacted_candidate_used_fewer_input_tokens": True,
        "evaluator_found_material_weakness": True,
        "repair_addressed_critique": True,
        "fork_lineage_persisted": True,
        "strategies_are_distinct": True,
    }
    assert len(result.sessions) == 9
    assert {session.role for session in result.sessions} == {
        "source",
        "primary-sources",
        "contrarian",
        "practitioner",
        "evaluator",
        "repair",
        "baseline-primary-sources",
        "baseline-contrarian",
        "baseline-practitioner",
    }
    assert result.metrics["model_requests"] == 10
    observation = result.metrics["paired_token_observation"]
    assert observation["input_token_delta"] > 0
    assert observation["first_attempt_input_token_delta"] > 0
    assert observation["baseline_output_tokens"] == 30
    assert observation["candidate_output_tokens"] == 30
    assert observation["measurement"] == "total-provider-input-with-first-attempt-control"
    assert observation["baseline_model_steps"] == 3
    assert observation["candidate_model_steps"] == 3
    cost = result.metrics["paired_cost_evidence"]
    assert cost == {
        "status": "priced",
        "currency": "USD",
        "candidate_cost": "0.00039",
        "paired_baseline_cost": "0.00087",
        "savings": "0.00048",
        "savings_percent": "55.17",
        "price_book_version": "deterministic-fixture-v1",
        "price_book_generated_at": "2026-01-01T00:00:00Z",
        "pricing_provider_name": "scripted",
        "pricing_model": "scripted-model",
        "pricing_match": "exact",
        "pricing_tier_max_input_tokens": None,
        "pricing_provenance": {
            "source": "deterministic fixture; not provider pricing",
            "url": "https://example.invalid/cayu/research-council-pricing-fixture",
            "as_of": "2026-01-01",
        },
    }
    assert all(session.model_steps >= 1 for session in result.sessions)
    assert result.output_path is not None
    assert result.output_path.exists()
