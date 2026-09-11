from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from cayu import AgentSpec, Message, ScriptedModelProvider
from cayu.runtime import (
    BudgetLimit,
    BudgetPolicy,
    CayuApp,
    RunRequest,
    default_price_book,
)


def _policy(maximum: str = "10") -> BudgetPolicy:
    return BudgetPolicy(
        limits=(
            BudgetLimit(
                scope="app",
                max_estimated_cost=Decimal(maximum),
                pricing=default_price_book(),
            ),
        )
    )


def test_budget_policy_constructor_and_getter_are_defensive() -> None:
    source = _policy()
    app = CayuApp(budget_policy=source, enable_logging=False)

    source.limits[0].max_estimated_cost = Decimal("99")
    exposed = app.budget_policy
    assert exposed is not None
    exposed.limits[0].max_estimated_cost = Decimal("77")

    assert app.budget_policy is not exposed
    assert app.budget_policy.limits[0].max_estimated_cost == Decimal("10")


def test_budget_policy_replacement_uses_same_defensive_owner() -> None:
    app = CayuApp(budget_policy=_policy("10"), enable_logging=False)
    replacement = _policy("20")

    app.budget_policy = replacement
    replacement.limits[0].max_estimated_cost = Decimal("99")

    assert app.budget_policy.limits[0].max_estimated_cost == Decimal("20")


def test_budget_policy_invalid_replacement_preserves_existing_value() -> None:
    app = CayuApp(budget_policy=_policy("10"), enable_logging=False)

    with pytest.raises(TypeError, match="Budget policy"):
        app.budget_policy = object()  # type: ignore[assignment]

    assert app.budget_policy.limits[0].max_estimated_cost == Decimal("10")


def test_budget_policy_none_removes_app_policy() -> None:
    app = CayuApp(budget_policy=_policy(), enable_logging=False)

    app.budget_policy = None

    assert app.budget_policy is None


def test_replacement_changes_only_later_candidate_profile() -> None:
    app = CayuApp(budget_policy=_policy("10"), enable_logging=False)
    app.register_provider(ScriptedModelProvider([]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="scripted-model"))
    request = RunRequest(
        agent_name="assistant",
        session_id="budget-profile-replacement",
        messages=[Message.text("user", "inspect")],
    )

    before = asyncio.run(app.inspect_run_execution_profile(request))
    app.budget_policy = _policy("20")
    after = asyncio.run(app.inspect_run_execution_profile(request))

    assert before != after
