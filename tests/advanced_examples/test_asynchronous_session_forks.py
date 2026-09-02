from __future__ import annotations

import asyncio

from examples.asynchronous_session_forks import run_asynchronous_fork_trace
from examples.asynchronous_session_forks.process_recovery import (
    run_fresh_process_fork_recovery_trace,
)


def test_asynchronous_session_forks_settle_and_recover_independently() -> None:
    result = asyncio.run(run_asynchronous_fork_trace())

    assert result.first_result == "result:B"
    assert result.selected_child == "B"
    assert result.child_statuses == {
        "A": "completed",
        "B": "completed",
        "C": "interrupted",
        "D": "failed",
    }
    assert len(set(result.queue_task_ids.values())) == 4
    assert result.provider_calls["checkpoint:14"] == 1
    assert result.provider_calls["checkpoint:15"] == 1
    assert result.provider_calls["child:A"] == 2
    assert result.provider_calls["child:B"] == 1
    assert result.provider_calls["child:C"] == 1
    assert result.provider_calls["child:D"] == 1
    assert result.provider_completions["child:A"] == 2
    assert result.provider_completions["child:B"] == 1
    assert result.provider_completions["child:C"] == 1
    assert result.provider_completions["checkpoint:14"] == 1
    assert result.provider_completions["checkpoint:15"] == 1
    assert "child:D" not in result.provider_completions
    assert result.provider_cancellations == {}
    assert result.tool_invocations == 1
    assert result.tool_mutations == 1
    assert result.trace == (
        "trunk_checkpoint_14_complete",
        "fork_admission_replayed_after_producer_reconstruction",
        "all_children_durably_dispatched_before_model_execution",
        "trunk_checkpoint_15_complete_while_children_pending",
        "worker_reconstructed_after_claim_boundary",
        "child_b_observed_while_siblings_running",
        "child_b_result_used_before_siblings_settled",
        "child_d_failed_without_altering_siblings",
        "worker_lost_after_durable_terminal_publication",
        "terminal_publication_recovered_without_redispatch",
        "child_a_completed_later",
        "worker_lost_after_durable_provider_completion",
        "provider_completion_recovered_without_redispatch",
        "child_c_interrupted_independently",
        "application_owned_evaluator_selected_settled_child",
        "terminal_reconstruction_created_no_duplicate_work",
    )


def test_asynchronous_session_forks_recover_in_fresh_processes() -> None:
    result = run_fresh_process_fork_recovery_trace()

    assert result.boundaries == (
        "claim",
        "provider_completion",
        "terminal_publication",
    )
    assert result.session_statuses == {
        "claim": "completed",
        "provider_completion": "interrupted",
        "terminal_publication": "completed",
    }
    assert result.task_statuses == {
        "claim": "completed",
        "provider_completion": "cancelled",
        "terminal_publication": "cancelled",
    }
    assert result.provider_calls == {
        boundary: {"child": 1, "source": 1} for boundary in result.boundaries
    }
    assert all(result.queue_task_ids.values())
    assert len(set(result.queue_task_ids.values())) == 1
