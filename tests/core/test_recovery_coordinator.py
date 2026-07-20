from __future__ import annotations

import asyncio
from decimal import Decimal

from cayu.core.thinking import ThinkingConfig
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime._recovery_coordinator import (
    _DEFAULT_APPROVAL_MAX_STEPS,
    _effective_approval_budget_limits,
    _effective_approval_max_steps,
    _effective_approval_retry_policy,
    _effective_approval_run_limits,
    _effective_approval_thinking,
    _interrupted_tool_round_results,
    _recovery_abandonment_signal,
)
from cayu.runtime.approvals import PendingToolApproval, PendingToolCallApproval
from cayu.runtime.budgets import BudgetLimit
from cayu.runtime.costs import ModelPrice, PriceBook
from cayu.runtime.retry_policy import RetryPolicy
from cayu.runtime.stop_policy import RunLimits


def _pending_approval(**kwargs) -> PendingToolApproval:
    return PendingToolApproval(
        approval_id="appr_1",
        tool_call_id="call_1",
        tool_name="side_effect",
        agent_name="assistant",
        tool_calls=[PendingToolCallApproval(tool_call_id="call_1", tool_name="side_effect")],
        **kwargs,
    )


def _budget_limit(max_estimated_cost: str) -> BudgetLimit:
    return BudgetLimit(
        max_estimated_cost=Decimal(max_estimated_cost),
        pricing=PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="fake",
                    model="fake-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("10"),
                ),
            )
        ),
    )


def test_effective_approval_run_config_prefers_override_then_pending_then_default() -> None:
    persisted = _pending_approval(
        max_steps=9,
        limits=RunLimits(max_tool_calls=2, scope="session"),
        budget_limits=(_budget_limit("1.00"),),
        retry_policy=RetryPolicy(max_attempts=4),
    )
    legacy = _pending_approval()

    assert _effective_approval_max_steps(max_steps=3, pending_approval=persisted) == 3
    assert _effective_approval_run_limits(
        limits=RunLimits(max_tool_calls=5),
        pending_approval=persisted,
    ) == RunLimits(max_tool_calls=5)
    assert (
        _effective_approval_budget_limits(
            budget_limits=(),
            pending_approval=persisted,
        )
        == ()
    )
    override_policy = RetryPolicy(max_attempts=2)
    assert (
        _effective_approval_retry_policy(
            retry_policy=override_policy,
            pending_approval=persisted,
        )
        is override_policy
    )

    assert _effective_approval_max_steps(max_steps=None, pending_approval=persisted) == 9
    assert _effective_approval_run_limits(
        limits=None,
        pending_approval=persisted,
    ) == RunLimits(max_tool_calls=2, scope="session")
    assert _effective_approval_budget_limits(
        budget_limits=None,
        pending_approval=persisted,
    ) == (_budget_limit("1.00"),)
    assert _effective_approval_retry_policy(
        retry_policy=None,
        pending_approval=persisted,
    ) == RetryPolicy(max_attempts=4)

    assert (
        _effective_approval_max_steps(max_steps=None, pending_approval=legacy)
        == _DEFAULT_APPROVAL_MAX_STEPS
    )
    assert _effective_approval_run_limits(limits=None, pending_approval=legacy) == RunLimits()
    assert _effective_approval_budget_limits(budget_limits=None, pending_approval=legacy) == ()
    assert (
        _effective_approval_retry_policy(
            retry_policy=None,
            pending_approval=legacy,
        )
        is None
    )


def test_effective_approval_thinking_restores_pending_run_config() -> None:
    pending = _pending_approval(thinking=ThinkingConfig(effort="high"))

    restored = _effective_approval_thinking(thinking=None, pending_approval=pending)
    assert restored is not None
    assert restored.effort == "high"

    override = ThinkingConfig(effort="low")
    assert _effective_approval_thinking(thinking=override, pending_approval=pending) is override


def test_interrupted_tool_round_results_attaches_artifacts_by_tool_call_id() -> None:
    # Parallel cleanup artifacts stay with their producing call. The unkeyed
    # sequential fallback belongs only to the first unfinished call.
    a = runtime_records.ToolCallRequest(id="A", name="tool_a", arguments={})
    b = runtime_records.ToolCallRequest(id="B", name="tool_b", arguments={})

    keyed = _interrupted_tool_round_results(
        tool_calls=[a, b],
        completed_outcomes=[],
        cancellation_artifacts_by_id={"B": [{"producer": "B"}]},
    )
    by_id = {outcome.call.id: outcome for outcome in keyed}
    assert by_id["B"].result.artifacts == [{"producer": "B"}]
    assert by_id["A"].result.artifacts == []

    fallback = _interrupted_tool_round_results(
        tool_calls=[a, b],
        completed_outcomes=[],
        cancellation_artifacts=[{"producer": "unknown"}],
    )
    by_id = {outcome.call.id: outcome for outcome in fallback}
    assert by_id["A"].result.artifacts == [{"producer": "unknown"}]
    assert by_id["B"].result.artifacts == []


def test_recovery_abandonment_signal_finds_nested_grouped_cancellation() -> None:
    cancellation = asyncio.CancelledError("cancel recovery")
    grouped = BaseExceptionGroup(
        "recovery failed during cancellation",
        [
            GeneratorExit(),
            RuntimeError("cleanup failed"),
            BaseExceptionGroup("nested", [cancellation]),
        ],
    )

    assert _recovery_abandonment_signal(grouped) is cancellation
    assert (
        _recovery_abandonment_signal(ExceptionGroup("ordinary", [RuntimeError("failed")])) is None
    )
