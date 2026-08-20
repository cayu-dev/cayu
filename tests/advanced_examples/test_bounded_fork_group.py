from __future__ import annotations

import asyncio
from pathlib import Path

from examples.bounded_fork_group.deterministic import run


def test_bounded_fork_group_runs_and_replays_the_public_coordinator(tmp_path: Path) -> None:
    result = asyncio.run(run(tmp_path))

    assert result.status == "verified"
    assert result.assertions == {
        "all_deterministic_gates_passed": True,
        "bounded_group_completed": True,
        "dispositions_cover_all_and_select_one": True,
        "economic_evidence_is_complete": True,
        "evaluator_is_structurally_isolated": True,
        "exact_checkpoint_and_profile_are_frozen": True,
        "replay_did_not_rerun_models": True,
        "siblings_share_causal_budget": True,
    }
    assert result.metrics["branch_count"] == 2
    assert result.metrics["model_requests"] == 4
    assert result.metrics["selected_branch"] == "focused"
    assert result.metrics["total_tokens"] == 145
