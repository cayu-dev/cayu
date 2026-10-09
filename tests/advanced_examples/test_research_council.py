from __future__ import annotations

import asyncio
from decimal import Decimal
from pathlib import Path

import pytest
from examples.cache_aware_research_council.deterministic import run, scripted_provider


def test_research_council_forks_strategies_and_repairs_seeded_weakness(
    tmp_path: Path,
) -> None:
    result = asyncio.run(run(tmp_path))

    assert result.status == "verified"
    assert result.scenario == "cache-aware-research-council"
    assert result.assertions == {
        "causal_budget_shared": True,
        "compaction_checkpoint_persisted": True,
        "compaction_represented_source_context": True,
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
    # Two source preparation turns, prompt-cache compaction, the source resume,
    # 3 + 3 branches, evaluator, and repair.
    assert result.metrics["model_requests"] == 12
    checkpoint = result.metrics["cache_window"]["checkpoint"]
    assert checkpoint["coverage_mode"] == "full"
    assert checkpoint["represented_message_count"] == 4
    assert checkpoint["cache_read_input_tokens"] == 0
    assert checkpoint["metadata"]["compactor"] == "PromptCacheCompactor"
    assert checkpoint["metadata"]["prompt_cache_compaction"] is True
    observation = result.metrics["paired_token_observation"]
    # The fixture reports input usage measured from each serialized request, so
    # these ratios fail if compacted forks are sent the full source transcript.
    baseline_first = observation["baseline_first_attempt_input_tokens"]
    candidate_first = observation["candidate_first_attempt_input_tokens"]
    assert candidate_first * 5 < baseline_first * 3
    assert observation["input_token_delta"] == baseline_first - candidate_first
    assert observation["baseline_output_tokens"] == 30
    assert observation["candidate_output_tokens"] == 30
    assert observation["measurement"] == "total-provider-input-with-first-attempt-control"
    assert observation["baseline_model_steps"] == 3
    assert observation["candidate_model_steps"] == 3
    cost_report = result.metrics["paired_cost_evidence"]
    assert cost_report["schema_version"] == 3
    assert cost_report["status"] == "verified"
    pair = cost_report["pairs"][0]
    assert Decimal(pair["savings"]) == Decimal(pair["baseline_cost"]) - Decimal(
        pair["candidate_cost"]
    )
    assert Decimal(pair["savings_percentage"]) > Decimal("40")
    assert [item["operation"] for item in pair["candidate"]["operations"]] == [
        "agent_step",
    ]
    assert pair["candidate"]["whole_harness"]["attempt_count"] == 3
    assert pair["candidate"]["pricing_provenance"] == [
        {
            "source": "deterministic fixture; not provider pricing",
            "url": "https://example.invalid/cayu/research-council-pricing-fixture",
            "as_of": "2026-01-01",
        }
    ]
    assert all(session.model_steps >= 1 for session in result.sessions)
    assert result.output_path is not None
    assert result.output_path.exists()


def test_research_council_requires_compaction_cache_read_when_enabled(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="compaction_read_warm_prompt_cache"):
        asyncio.run(run(tmp_path / "cold", require_compaction_cache_read=True))

    result = asyncio.run(
        run(
            tmp_path / "warm",
            provider=scripted_provider(compaction_cache_read_input_tokens=6_000),
            require_compaction_cache_read=True,
        )
    )

    assert result.assertions["compaction_read_warm_prompt_cache"] is True
    assert result.metrics["cache_window"]["checkpoint"]["cache_read_input_tokens"] == 6_000
