"""Every selected target retains strict, target-specific budget admission."""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest
from tests.core.test_model_failover_recovery import _RecoveryProvider
from tests.core.test_model_failover_stages import _StageMemoryStore, _StageSQLiteStore

from cayu import (
    AgentSpec,
    BudgetLimit,
    BudgetPolicy,
    BudgetReservation,
    CayuApp,
    EventType,
    Message,
    ModelFailoverPolicy,
    ModelPrice,
    ModelTarget,
    PriceBook,
    RunRequest,
)
from cayu.runtime.retry_policy import RetryPolicy
from cayu.storage.budget_ledger import SQLiteBudgetLedger


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("pricing_case", ["priced", "missing", "wrong_currency"])
@pytest.mark.parametrize("reserved", [True, False])
@pytest.mark.parametrize("budget_source", ["policy", "request"])
def test_public_fallback_pricing_admission(
    monkeypatch, tmp_path, backend, pricing_case, reserved, budget_source
):

    async def scenario():
        path = tmp_path / "pricing.sqlite"
        ledger_path = tmp_path / "pricing-ledger.sqlite"
        store = _StageMemoryStore() if backend == "memory" else _StageSQLiteStore(path)
        ledger = SQLiteBudgetLedger(ledger_path) if backend == "sqlite" else None
        prices = [
            ModelPrice.fixed(
                provider_name="primary",
                model="small",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("0"),
            )
        ]
        if pricing_case != "missing":
            prices.append(
                ModelPrice.fixed(
                    provider_name="backup",
                    model="large",
                    input_per_million=Decimal("2"),
                    output_per_million=Decimal("0"),
                    currency="EUR" if pricing_case == "wrong_currency" else "USD",
                )
            )
        policy = BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("10"),
                    pricing=PriceBook(prices=tuple(prices)),
                    reservation=(
                        BudgetReservation(max_input_tokens=1_000_000, max_output_tokens=0)
                        if reserved
                        else None
                    ),
                ),
            )
        )
        app = CayuApp(
            session_store=store,
            budget_ledger=ledger,
            budget_policy=policy if budget_source == "policy" else None,
            enable_logging=False,
        )
        primary, backup = _RecoveryProvider("primary"), _RecoveryProvider("backup")
        app.register_provider(primary, default=True)
        app.register_provider(backup)
        app.register_agent(AgentSpec(name="agent", model="small"))
        try:
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="agent",
                        session_id="pricing",
                        messages=[Message.text("user", "answer")],
                        retry_policy=RetryPolicy(max_attempts=1),
                        budget_limits=policy.limits if budget_source == "request" else (),
                        failover=ModelFailoverPolicy(
                            fallbacks=(ModelTarget(provider_name="backup", model="large"),),
                            max_total_attempts=2,
                        ),
                    )
                )
            ]
            allowed = pricing_case == "priced"
            assert len(primary.requests) == 1
            assert len(backup.requests) == int(allowed)
            assert events[-1].type is (
                EventType.SESSION_COMPLETED if allowed else EventType.SESSION_INTERRUPTED
            ), events[-1].payload
            persisted = await store.load_events("pricing")
            reservations = [event for event in persisted if event.type is EventType.BUDGET_RESERVED]
            assert len(reservations) == (1 + int(allowed) if reserved else 0)
            records = []
            for event in reservations:
                record = await app.budget_ledger.load_reservation(event.payload["reservation_id"])
                assert record is not None and record.status == "reconciled"
                records.append(record)
            if reserved:
                assert records[0].provider_name == "primary" and records[0].model == "small"
                assert records[0].reserved_amount == records[0].actual_amount == Decimal("1")
            if allowed and reserved:
                assert records[1].provider_name == "backup" and records[1].model == "large"
                assert records[1].reserved_amount == Decimal("2")
                assert records[1].actual_amount == Decimal("0.000002")
            assert not any(event.type is EventType.MODEL_FAILOVER_EXHAUSTED for event in events)
            checkpoint = await store.load_checkpoint("pricing")
            if isinstance(store, _StageSQLiteStore):
                await store.close()
                store = _StageSQLiteStore(path)
                assert await store.load_events("pricing") == persisted
                assert await store.load_checkpoint("pricing") == checkpoint
                assert ledger is not None
                await ledger.close()
                ledger = SQLiteBudgetLedger(ledger_path)
                assert [
                    await ledger.load_reservation(record.reservation_id) for record in records
                ] == records
        finally:
            if ledger is not None:
                await ledger.close()
            if isinstance(store, _StageSQLiteStore):
                await store.close()

    asyncio.run(scenario())
