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
    assert observation["measurement"] == "total-provider-input-with-first-attempt-control"
    assert observation["baseline_model_steps"] == 3
    assert observation["candidate_model_steps"] == 3
    assert all(session.model_steps >= 1 for session in result.sessions)
    assert result.output_path is not None
    assert result.output_path.exists()
