from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Literal

from cayu.runtime import (
    BudgetLimit,
    BudgetPolicy,
    BudgetReservation,
    InMemoryBudgetLedger,
    ModelPrice,
    PriceBook,
)
from cayu.runtime.budgets import (
    budget_limits_for_session,
    request_budget_limits_for_session,
)
from cayu.storage import SQLiteBudgetLedger


def _price_book() -> PriceBook:
    return PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="fake",
                model="fake-model",
                input_per_million=Decimal("1000000"),
                output_per_million=Decimal("1000000"),
            ),
        )
    )


def _limit(
    *,
    maximum: str = "1.5",
    action: Literal["interrupt", "notify"] = "interrupt",
) -> BudgetLimit:
    return BudgetLimit(
        scope="app",
        max_estimated_cost=Decimal(maximum),
        pricing=_price_book(),
        action=action,
        reservation=BudgetReservation(max_input_tokens=1, max_output_tokens=0),
    )


def _resolve_policy(policy: BudgetPolicy):
    return budget_limits_for_session(
        policy=policy,
        agent_name="assistant",
        causal_budget_id="job_1",
    )


def test_effective_limit_ids_are_stable_distinct_and_semantic() -> None:
    duplicate_policy = BudgetPolicy(limits=(_limit(), _limit()))

    first = _resolve_policy(duplicate_policy)
    reconstructed = _resolve_policy(
        BudgetPolicy.model_validate(duplicate_policy.model_dump(mode="json"))
    )
    with_unrelated_entry = _resolve_policy(
        BudgetPolicy(limits=(_limit(maximum="9"), *duplicate_policy.limits))
    )
    changed = _resolve_policy(BudgetPolicy(limits=(_limit(maximum="2"),)))

    assert first[0].budget_limit_id != first[1].budget_limit_id
    assert [limit.budget_limit_id for limit in reconstructed] == [
        limit.budget_limit_id for limit in first
    ]
    assert [limit.budget_limit_id for limit in with_unrelated_entry[1:]] == [
        limit.budget_limit_id for limit in first
    ]
    assert changed[0].budget_limit_id != first[0].budget_limit_id
    assert all(limit.budget_limit_id.startswith("blim_") for limit in first)
    assert all("fake-model" not in limit.budget_limit_id for limit in first)


def test_duplicate_membership_change_does_not_reuse_a_sibling_identity() -> None:
    duplicated = _resolve_policy(BudgetPolicy(limits=(_limit(), _limit())))
    one_survivor = _resolve_policy(BudgetPolicy(limits=(_limit(),)))

    duplicated_ids = {limit.budget_limit_id for limit in duplicated}

    assert one_survivor[0].budget_limit_id not in duplicated_ids


def test_app_and_request_limit_namespaces_do_not_alias() -> None:
    configured = _limit()

    app_limit = _resolve_policy(BudgetPolicy(limits=(configured,)))[0]
    request_limit = request_budget_limits_for_session(
        limits=(configured,),
        agent_name="assistant",
        causal_budget_id="job_1",
    )[0]

    assert app_limit.budget_limit_id != request_limit.budget_limit_id


def test_in_memory_ledger_partitions_identical_limits_by_exact_id() -> None:
    async def scenario() -> None:
        limits = _resolve_policy(BudgetPolicy(limits=(_limit(), _limit())))
        ledger = InMemoryBudgetLedger()

        first = await ledger.reserve(
            limit=limits[0],
            session_id="session_1",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
        )
        second = await ledger.reserve(
            limit=limits[1],
            session_id="session_1",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
        )
        repeated_first = await ledger.reserve(
            limit=limits[0],
            session_id="session_2",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
        )

        assert first.accepted is True
        assert second.accepted is True
        assert repeated_first.accepted is False
        assert first.record is not None
        assert second.record is not None
        assert first.record.budget_limit_id == limits[0].budget_limit_id
        assert second.record.budget_limit_id == limits[1].budget_limit_id

    asyncio.run(scenario())


def test_sqlite_ledger_reconstructs_exact_limit_and_separates_semantic_change(
    tmp_path,
) -> None:
    path = tmp_path / "budget-identities.sqlite"
    policy = BudgetPolicy(limits=(_limit(),))

    async def scenario() -> None:
        original = _resolve_policy(policy)[0]
        first_ledger = SQLiteBudgetLedger(path)
        try:
            accepted = await first_ledger.reserve(
                limit=original,
                session_id="session_1",
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
            )
        finally:
            await first_ledger.close()

        reconstructed = _resolve_policy(
            BudgetPolicy.model_validate(policy.model_dump(mode="json"))
        )[0]
        changed = _resolve_policy(BudgetPolicy(limits=(_limit(maximum="2"),)))[0]
        second_ledger = SQLiteBudgetLedger(path)
        try:
            rejected = await second_ledger.reserve(
                limit=reconstructed,
                session_id="session_2",
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
            )
            independently_accepted = await second_ledger.reserve(
                limit=changed,
                session_id="session_3",
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
            )
        finally:
            await second_ledger.close()

        assert accepted.accepted is True
        assert reconstructed.budget_limit_id == original.budget_limit_id
        assert rejected.accepted is False
        assert changed.budget_limit_id != original.budget_limit_id
        assert independently_accepted.accepted is True

    asyncio.run(scenario())
