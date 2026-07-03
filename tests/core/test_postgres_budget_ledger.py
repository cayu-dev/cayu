"""Postgres budget-ledger parity tests.

Mirror the SQLite ledger assertions in ``test_usage.py`` against a real
Dockerized Postgres. They skip automatically when Docker is unavailable.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from cayu.runtime import (
    BudgetLimit,
    BudgetReservation,
    BudgetWindow,
    ModelPricing,
    PricingCatalog,
)

pytestmark = pytest.mark.usefixtures("postgres_dsn")

_TABLES = (
    "cayu_budget_reservations",
    "cayu_knowledge_labels",
    "cayu_knowledge_aspects",
    "cayu_knowledge_impact_targets",
    "cayu_knowledge_chunks",
    "cayu_knowledge_entries",
    "cayu_event_watcher_state",
    "cayu_events",
    "cayu_session_labels",
    "cayu_transcript_messages",
    "cayu_checkpoints",
    "cayu_tasks",
    "cayu_sessions",
    "cayu_schema_migrations",
)


class MutableClock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value


def _reservation_budget_limit(
    max_cost: str = "1",
    *,
    window: BudgetWindow | str | None = None,
) -> BudgetLimit:
    return BudgetLimit(
        scope="app",
        max_estimated_cost=Decimal(max_cost),
        window=BudgetWindow.all_time() if window is None else window,
        pricing=PricingCatalog(
            prices=(
                ModelPricing(
                    provider_name="fake",
                    model="fake-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                    cache_read_input_per_million=Decimal("0.25"),
                    cache_write_input_per_million=Decimal("1.25"),
                ),
            )
        ),
        reservation=BudgetReservation(
            max_input_tokens=100_000,
            max_output_tokens=50_000,
            max_cache_read_input_tokens=40_000,
            max_cache_write_input_tokens=8_000,
        ),
    )


async def _drop_all(dsn: str) -> None:
    import psycopg

    async with await psycopg.AsyncConnection.connect(dsn) as conn:
        async with conn.cursor() as cur:
            for table in _TABLES:
                await cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        await conn.commit()


def _new_ledger(dsn: str, **kwargs):
    from cayu import PostgresBudgetLedger
    from cayu.storage.migrations import SchemaMode

    # Tests own a throwaway database and (re)create the schema each run.
    return PostgresBudgetLedger(
        dsn,
        min_size=1,
        max_size=4,
        schema_mode=SchemaMode.CREATE,
        **kwargs,
    )


def _run(dsn: str, coro_factory, **ledger_kwargs) -> object:
    async def runner():
        await _drop_all(dsn)
        ledger = _new_ledger(dsn, **ledger_kwargs)
        try:
            return await coro_factory(ledger)
        finally:
            await ledger.close()

    return asyncio.run(runner())


async def _reserve(ledger, limit, session_id: str):
    return await ledger.reserve(
        limit=limit,
        session_id=session_id,
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
    )


def test_postgres_budget_ledger_reserves_reconciles_and_releases(postgres_dsn) -> None:
    async def ops(ledger):
        limit = _reservation_budget_limit(max_cost="0.25")
        first = await _reserve(ledger, limit, "sess_1")
        assert first.accepted is True
        assert first.record is not None
        blocked = await _reserve(ledger, limit, "sess_2")
        assert blocked.accepted is False
        reconciled = await ledger.reconcile(
            reservation_id=first.record.reservation_id,
            actual_amount=Decimal("0.01"),
            reason="actual usage",
        )
        retry = await _reserve(ledger, limit, "sess_2")
        assert retry.accepted is True
        assert retry.record is not None
        released = await ledger.release(
            reservation_id=retry.record.reservation_id,
            reason="unused",
        )
        return blocked, reconciled, released

    blocked, reconciled, released = _run(postgres_dsn, ops)

    assert blocked.actual == Decimal("0.44")
    assert reconciled.released_amount == Decimal("0.21")
    assert released.status == "released"


def test_postgres_budget_ledger_survives_ledger_restart(postgres_dsn) -> None:
    # The whole point of the durable ledger: reservations written by one worker
    # bind budget for a different worker process on a fresh connection pool.
    async def runner():
        await _drop_all(postgres_dsn)
        first_worker = _new_ledger(postgres_dsn)
        limit = _reservation_budget_limit(max_cost="0.25")
        try:
            first = await _reserve(first_worker, limit, "sess_1")
            assert first.accepted is True
        finally:
            await first_worker.close()
        second_worker = _new_ledger(postgres_dsn)
        try:
            return await _reserve(second_worker, limit, "sess_2")
        finally:
            await second_worker.close()

    blocked = asyncio.run(runner())

    assert blocked.accepted is False
    assert blocked.actual == Decimal("0.44")


def test_postgres_budget_ledger_window_bounds_active_reservations(postgres_dsn) -> None:
    clock = MutableClock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))

    async def ops(ledger):
        rolling_limit = _reservation_budget_limit(
            max_cost="0.25",
            window=BudgetWindow.rolling(seconds=60),
        )
        all_time_limit = _reservation_budget_limit(max_cost="0.25")
        rolling_first = await _reserve(ledger, rolling_limit, "sess_rolling_1")
        all_time_first = await _reserve(ledger, all_time_limit, "sess_all_time_1")
        clock.value = datetime(2026, 1, 1, 12, 0, 30, tzinfo=UTC)
        blocked_now = await _reserve(ledger, rolling_limit, "sess_rolling_blocked")
        clock.value = datetime(2026, 1, 1, 12, 2, tzinfo=UTC)
        rolling_second = await _reserve(ledger, rolling_limit, "sess_rolling_2")
        all_time_second = await _reserve(ledger, all_time_limit, "sess_all_time_2")
        return rolling_first, blocked_now, rolling_second, all_time_first, all_time_second

    rolling_first, blocked_now, rolling_second, all_time_first, all_time_second = _run(
        postgres_dsn,
        ops,
        clock=clock,
    )

    assert rolling_first.accepted is True
    assert blocked_now.accepted is False
    assert blocked_now.actual == Decimal("0.44")
    assert rolling_second.accepted is True
    assert rolling_second.actual == Decimal("0.22")
    assert all_time_first.accepted is True
    assert all_time_second.accepted is False
    assert all_time_second.actual == Decimal("0.44")


def test_postgres_budget_ledger_reaps_expired_active_reservations(postgres_dsn) -> None:
    clock = MutableClock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))

    async def ops(ledger):
        limit = _reservation_budget_limit(max_cost="0.25")
        orphaned = await _reserve(ledger, limit, "sess_orphaned")
        assert orphaned.accepted is True
        assert orphaned.record is not None
        clock.value = datetime(2026, 1, 1, 12, 1, tzinfo=UTC)
        recovered = await _reserve(ledger, limit, "sess_recovered")
        # V3: reconciling a reservation reaped while still in flight records the actual
        # spend instead of crashing the billed run (which would also undercount).
        reconciled = await ledger.reconcile(
            reservation_id=orphaned.record.reservation_id,
            actual_amount=Decimal("0.01"),
        )
        return recovered, reconciled

    recovered, reconciled = _run(postgres_dsn, ops, clock=clock, reservation_ttl_seconds=60)

    assert recovered.accepted is True
    assert recovered.actual == Decimal("0.22")
    assert reconciled.status == "reconciled"
    assert reconciled.actual_amount == Decimal("0.01")


def test_postgres_budget_ledger_release_tolerates_ttl_reaped_reservation(postgres_dsn) -> None:
    clock = MutableClock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))

    async def ops(ledger):
        limit = _reservation_budget_limit(max_cost="0.25")
        orphaned = await _reserve(ledger, limit, "sess_orphaned")
        assert orphaned.accepted is True
        assert orphaned.record is not None
        clock.value = datetime(2026, 1, 1, 12, 1, tzinfo=UTC)
        recovered = await _reserve(ledger, limit, "sess_recovered")
        released = await ledger.release(
            reservation_id=orphaned.record.reservation_id,
            reason="cleanup",
        )
        return recovered, released

    recovered, released = _run(postgres_dsn, ops, clock=clock, reservation_ttl_seconds=60)

    assert recovered.accepted is True
    assert released.status == "released"
    assert released.actual_amount is None
    assert released.released_amount == released.reserved_amount
    assert released.reason == "Reservation expired: not reconciled within 60s."


def test_postgres_budget_ledger_rejects_unknown_reservation(postgres_dsn) -> None:
    async def ops(ledger):
        with pytest.raises(KeyError, match="bres_missing"):
            await ledger.reconcile(
                reservation_id="bres_missing",
                actual_amount=Decimal("0.01"),
            )
        with pytest.raises(KeyError, match="bres_missing"):
            await ledger.release(reservation_id="bres_missing", reason="unused")
        return True

    assert _run(postgres_dsn, ops) is True
