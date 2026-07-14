from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from cayu.core import Event, EventType, Message
from cayu.providers import UsageDialect
from cayu.runtime import (
    BudgetLimit,
    BudgetPolicy,
    BudgetReservation,
    BudgetWindow,
    CayuApp,
    InMemoryBudgetLedger,
    InMemorySessionStore,
    ModelPricing,
    PricingCatalog,
    RunRequest,
    SessionBudgetStore,
    SessionIdentity,
    default_model_catalog,
)
from cayu.runtime.budgets import (
    InMemoryBudgetStore,
    budget_check_from_events,
    copy_budget_window,
    copy_request_budget_limits,
    events_for_budget_window,
    request_budget_limits_for_session,
)
from cayu.runtime.costs import (
    estimate_causal_budget_cost,
    estimate_session_cost,
)
from cayu.runtime.stop_policy import (
    RunLimits,
    StopDecision,
    StopLimit,
    copy_run_limits,
    first_reached_limit,
    has_run_limits,
)
from cayu.runtime.usage import (
    causal_budget_usage_summary,
    normalize_usage_metrics,
    session_usage_summary,
    usage_metrics_from_event_payload,
)
from cayu.storage import SQLiteBudgetLedger, SQLiteSessionStore


class MutableClock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value


def test_normalize_openai_usage_metrics() -> None:
    metrics = normalize_usage_metrics(
        provider_name="openai",
        model="gpt-5.5",
        raw_usage={
            "input_tokens": 100,
            "input_tokens_details": {"cached_tokens": 60},
            "output_tokens": 20,
            "output_tokens_details": {"reasoning_tokens": 5},
            "total_tokens": 120,
        },
    )

    assert metrics is not None
    assert metrics.provider_name == "openai"
    assert metrics.model == "gpt-5.5"
    assert metrics.input_tokens == 100
    assert metrics.output_tokens == 20
    assert metrics.total_tokens == 120
    assert metrics.reasoning_output_tokens == 5
    assert metrics.cache.read_tokens == 60
    assert metrics.cache.write_tokens == 0
    assert metrics.cache.cached_input_tokens == 60
    assert metrics.cache.uncached_input_tokens == 40


def test_normalize_openai_chat_usage_shape() -> None:
    metrics = normalize_usage_metrics(
        provider_name="openai",
        model="gpt-5.5",
        raw_usage={
            "prompt_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 40},
            "completion_tokens": 10,
            "completion_tokens_details": {"reasoning_tokens": 3},
        },
    )

    assert metrics is not None
    assert metrics.input_tokens == 100
    assert metrics.output_tokens == 10
    assert metrics.total_tokens == 110
    assert metrics.reasoning_output_tokens == 3
    assert metrics.cache.read_tokens == 40
    assert metrics.cache.cached_input_tokens == 40
    assert metrics.cache.uncached_input_tokens == 60


def test_budget_policy_validates_scope_keys_and_duplicates() -> None:
    pricing = PricingCatalog(
        prices=(
            ModelPricing(
                provider_name="fake",
                model="fake-model",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("1"),
            ),
        )
    )

    policy = BudgetPolicy(
        limits=(
            BudgetLimit(
                scope="app",
                max_estimated_cost=Decimal("1"),
                pricing=pricing,
            ),
            BudgetLimit(
                scope="agent",
                key="builder",
                max_estimated_cost=Decimal("2"),
                pricing=pricing,
            ),
            BudgetLimit(
                scope="causal",
                key="job_1",
                max_estimated_cost=Decimal("3"),
                pricing=pricing,
            ),
        )
    )

    assert len(policy.limits) == 3
    tiered_policy = BudgetPolicy(
        limits=(
            BudgetLimit(
                scope="app",
                max_estimated_cost=Decimal("1"),
                pricing=pricing,
                action="notify",
            ),
            BudgetLimit(
                scope="app",
                max_estimated_cost=Decimal("2"),
                pricing=pricing,
                action="interrupt",
            ),
        )
    )
    assert [limit.action for limit in tiered_policy.limits] == ["notify", "interrupt"]
    from_mapping = BudgetPolicy(
        limits=(
            {
                "scope": "app",
                "max_estimated_cost": "3",
                "pricing": pricing.model_dump(mode="json"),
            },
        )
    )
    assert from_mapping.limits[0].max_estimated_cost == Decimal("3")
    with pytest.raises(ValueError, match="must not set key"):
        BudgetLimit(
            scope="app",
            key="global",
            max_estimated_cost=Decimal("1"),
            pricing=pricing,
        )
    with pytest.raises(ValueError, match="require key"):
        BudgetLimit(
            scope="agent",
            max_estimated_cost=Decimal("1"),
            pricing=pricing,
        )
    with pytest.raises(ValueError, match="require key"):
        BudgetLimit(
            scope="causal",
            max_estimated_cost=Decimal("1"),
            pricing=pricing,
        )
    with pytest.raises(ValueError, match="duplicate"):
        BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("1"),
                    pricing=pricing,
                ),
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("1"),
                    pricing=pricing,
                ),
            )
        )
    with pytest.raises(ValueError, match="request-scoped"):
        BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="session",
                    max_estimated_cost=Decimal("1"),
                    pricing=pricing,
                ),
            )
        )
    with pytest.raises(ValueError, match="require priced"):
        BudgetLimit(
            scope="app",
            max_estimated_cost=Decimal("1"),
            pricing=pricing,
            allow_unpriced=True,
            reservation=BudgetReservation(max_input_tokens=1, max_output_tokens=0),
        )
    with pytest.raises(ValueError, match="action='interrupt'"):
        BudgetLimit(
            scope="app",
            max_estimated_cost=Decimal("1"),
            pricing=pricing,
            action="notify",
            reservation=BudgetReservation(max_input_tokens=1, max_output_tokens=0),
        )


def test_budget_window_calendar_bounds_and_storage_key() -> None:
    window = BudgetWindow.calendar(period="day", timezone="America/New_York")
    since, until = window.bounds(now=datetime(2026, 6, 19, 13, 30, tzinfo=UTC))

    assert window.storage_key == "calendar:day:America/New_York"
    assert since == datetime(2026, 6, 19, 4, 0, tzinfo=UTC)
    assert until == datetime(2026, 6, 20, 4, 0, tzinfo=UTC)

    copied = BudgetWindow.model_validate({"kind": "calendar", "period": "month", "timezone": "UTC"})
    assert copied.storage_key == "calendar:month:UTC"
    assert copy_budget_window("calendar:week:UTC") == BudgetWindow.calendar(
        period="week",
        timezone="UTC",
    )
    assert BudgetWindow.calendar(period="week", timezone="UTC").bounds(
        now=datetime(2026, 6, 19, 13, 30, tzinfo=UTC)
    ) == (
        datetime(2026, 6, 15, 0, 0, tzinfo=UTC),
        datetime(2026, 6, 22, 0, 0, tzinfo=UTC),
    )


def test_budget_window_rejects_invalid_calendar_fields() -> None:
    with pytest.raises(ValueError, match="Unknown budget window timezone"):
        BudgetWindow.calendar(period="day", timezone="Not/AZone")
    with pytest.raises(ValueError, match="Calendar budget windows require timezone"):
        BudgetWindow(kind="calendar", period="day")
    with pytest.raises(ValueError, match="Rolling budget windows must not set calendar details"):
        BudgetWindow(kind="rolling", duration_seconds=60, timezone="UTC")


def _reservation_budget_limit(
    max_cost: str = "1",
    *,
    window: BudgetWindow | str | None = None,
    key: str | None = None,
) -> BudgetLimit:
    return BudgetLimit(
        scope="app" if key is None else "agent",
        key=key,
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


def test_budget_reservation_requires_reserved_tokens() -> None:
    with pytest.raises(ValueError, match="at least one token"):
        BudgetReservation(max_input_tokens=0, max_output_tokens=0)


def test_in_memory_budget_ledger_reserves_reconciles_and_releases() -> None:
    async def run():
        ledger = InMemoryBudgetLedger()
        limit = _reservation_budget_limit(max_cost="0.50")
        first = await ledger.reserve(
            limit=limit,
            session_id="sess_1",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
        )
        assert first.accepted is True
        assert first.requested == Decimal("0.22")
        assert first.record is not None
        second = await ledger.reserve(
            limit=limit,
            session_id="sess_2",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
        )
        assert second.accepted is True
        third = await ledger.reserve(
            limit=limit,
            session_id="sess_3",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
        )
        assert third.accepted is False
        reconciled = await ledger.reconcile(
            reservation_id=first.record.reservation_id,
            actual_amount=Decimal("0.05"),
            reason="actual usage",
        )
        assert reconciled.status == "reconciled"
        assert reconciled.released_amount == Decimal("0.17")
        retry = await ledger.reserve(
            limit=limit,
            session_id="sess_3",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
        )
        assert retry.accepted is True
        assert retry.record is not None
        released = await ledger.release(
            reservation_id=retry.record.reservation_id,
            reason="model failed",
        )
        assert released.status == "released"
        return third, retry, released

    third, retry, released = asyncio.run(run())

    assert "reservation failed" in third.message.lower()
    assert retry.actual == Decimal("0.49")
    assert released.released_amount == Decimal("0.22")


def test_in_memory_budget_ledger_window_bounds_active_reservations() -> None:
    async def run():
        clock = MutableClock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
        ledger = InMemoryBudgetLedger(clock=clock)
        rolling_limit = _reservation_budget_limit(
            max_cost="0.25",
            window=BudgetWindow.rolling(seconds=60),
        )
        all_time_limit = _reservation_budget_limit(max_cost="0.25")
        rolling_first = await ledger.reserve(
            limit=rolling_limit,
            session_id="sess_rolling_1",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
        )
        all_time_first = await ledger.reserve(
            limit=all_time_limit,
            session_id="sess_all_time_1",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
        )
        clock.value = datetime(2026, 1, 1, 12, 0, 30, tzinfo=UTC)
        blocked_now = await ledger.reserve(
            limit=rolling_limit,
            session_id="sess_rolling_blocked",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
        )
        clock.value = datetime(2026, 1, 1, 12, 2, tzinfo=UTC)
        rolling_second = await ledger.reserve(
            limit=rolling_limit,
            session_id="sess_rolling_2",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
        )
        all_time_second = await ledger.reserve(
            limit=all_time_limit,
            session_id="sess_all_time_2",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
        )
        return rolling_first, blocked_now, rolling_second, all_time_first, all_time_second

    rolling_first, blocked_now, rolling_second, all_time_first, all_time_second = asyncio.run(run())

    assert rolling_first.accepted is True
    assert blocked_now.accepted is False
    assert blocked_now.actual == Decimal("0.44")
    assert rolling_second.accepted is False
    assert rolling_second.actual == Decimal("0.44")
    assert all_time_first.accepted is True
    assert all_time_second.accepted is False
    assert all_time_second.actual == Decimal("0.44")


def test_in_memory_budget_ledger_uses_reconciliation_time_for_rolling_window() -> None:
    async def run():
        clock = MutableClock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
        ledger = InMemoryBudgetLedger(clock=clock)
        limit = _reservation_budget_limit(
            max_cost="0.25",
            window=BudgetWindow.rolling(seconds=60),
        )
        first = await ledger.reserve(
            limit=limit,
            session_id="sess_1",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
        )
        assert first.record is not None
        await ledger.reconcile(
            reservation_id=first.record.reservation_id,
            actual_amount=Decimal("0.22"),
            occurred_at=datetime(2026, 1, 1, 12, 2, tzinfo=UTC),
        )
        clock.value = datetime(2026, 1, 1, 12, 2, 30, tzinfo=UTC)
        blocked = await ledger.reserve(
            limit=limit,
            session_id="sess_2",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
        )
        clock.value = datetime(2026, 1, 1, 12, 3, 1, tzinfo=UTC)
        accepted = await ledger.reserve(
            limit=limit,
            session_id="sess_3",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
        )
        return blocked, accepted

    blocked, accepted = asyncio.run(run())

    assert blocked.accepted is False
    assert blocked.actual == Decimal("0.44")
    assert accepted.accepted is True
    assert accepted.actual == Decimal("0.22")


def test_in_memory_budget_ledger_uses_reconciliation_time_for_calendar_window() -> None:
    async def run():
        clock = MutableClock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
        ledger = InMemoryBudgetLedger(clock=clock)
        limit = _reservation_budget_limit(
            max_cost="0.25",
            window=BudgetWindow.calendar(period="day", timezone="UTC"),
        )
        first = await ledger.reserve(
            limit=limit,
            session_id="sess_1",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
        )
        assert first.record is not None
        await ledger.reconcile(
            reservation_id=first.record.reservation_id,
            actual_amount=Decimal("0.22"),
            occurred_at=datetime(2026, 1, 1, 23, 59, tzinfo=UTC),
        )
        clock.value = datetime(2026, 1, 2, 0, 1, tzinfo=UTC)
        next_day = await ledger.reserve(
            limit=limit,
            session_id="sess_2",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
        )
        active = await ledger.reserve(
            limit=limit,
            session_id="sess_3",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
        )
        return next_day, active

    next_day, active = asyncio.run(run())

    assert next_day.accepted is True
    assert active.accepted is False
    assert active.actual == Decimal("0.44")


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_budget_ledgers_keep_active_reservation_across_calendar_boundary(
    tmp_path,
    backend: str,
) -> None:
    async def run():
        clock = MutableClock(datetime(2026, 1, 1, 23, 59, tzinfo=UTC))
        ledger = (
            InMemoryBudgetLedger(clock=clock)
            if backend == "memory"
            else SQLiteBudgetLedger(tmp_path / "budget.sqlite", clock=clock)
        )
        try:
            limit = _reservation_budget_limit(
                max_cost="0.25",
                window=BudgetWindow.calendar(period="day", timezone="UTC"),
            )
            first = await ledger.reserve(
                limit=limit,
                session_id="sess_before_midnight",
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
            )
            clock.value = datetime(2026, 1, 2, 0, 1, tzinfo=UTC)
            blocked = await ledger.reserve(
                limit=limit,
                session_id="sess_after_midnight",
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
            )
            return first, blocked
        finally:
            close = getattr(ledger, "close", None)
            if close is not None:
                await close()

    first, blocked = asyncio.run(run())

    assert first.accepted is True
    assert blocked.accepted is False
    assert blocked.actual == Decimal("0.44")


def test_sqlite_budget_ledger_reserves_reconciles_and_releases(tmp_path) -> None:
    async def run():
        ledger = SQLiteBudgetLedger(tmp_path / "budget.sqlite")
        try:
            limit = _reservation_budget_limit(max_cost="0.25")
            first = await ledger.reserve(
                limit=limit,
                session_id="sess_1",
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
            )
            assert first.accepted is True
            assert first.record is not None
            blocked = await ledger.reserve(
                limit=limit,
                session_id="sess_2",
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
            )
            assert blocked.accepted is False
            reconciled = await ledger.reconcile(
                reservation_id=first.record.reservation_id,
                actual_amount=Decimal("0.01"),
                reason="actual usage",
            )
            retry = await ledger.reserve(
                limit=limit,
                session_id="sess_2",
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
            )
            assert retry.accepted is True
            assert retry.record is not None
            released = await ledger.release(
                reservation_id=retry.record.reservation_id,
                reason="unused",
            )
            return blocked, reconciled, released
        finally:
            await ledger.close()

    blocked, reconciled, released = asyncio.run(run())

    assert blocked.actual == Decimal("0.44")
    assert reconciled.released_amount == Decimal("0.21")
    assert released.status == "released"


def test_sqlite_budget_ledger_persists_rolling_window_key(tmp_path) -> None:
    async def run():
        ledger = SQLiteBudgetLedger(tmp_path / "budget.sqlite")
        try:
            rolling_limit = _reservation_budget_limit(
                max_cost="0.25",
                window=BudgetWindow.rolling(seconds=60),
            )
            all_time_limit = _reservation_budget_limit(max_cost="0.25")
            first = await ledger.reserve(
                limit=rolling_limit,
                session_id="sess_1",
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
            )
            blocked = await ledger.reserve(
                limit=rolling_limit,
                session_id="sess_2",
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
            )
            all_time = await ledger.reserve(
                limit=all_time_limit,
                session_id="sess_3",
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
            )
            return first, blocked, all_time
        finally:
            await ledger.close()

    first, blocked, all_time = asyncio.run(run())

    assert first.accepted is True
    assert first.window.storage_key == "rolling:60s"
    assert blocked.accepted is False
    assert blocked.actual == Decimal("0.44")
    assert all_time.accepted is True
    assert all_time.window.storage_key == "all_time"


def test_sqlite_budget_ledger_window_bounds_active_reservations(tmp_path) -> None:
    async def run():
        clock = MutableClock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
        ledger = SQLiteBudgetLedger(tmp_path / "budget.sqlite", clock=clock)
        try:
            rolling_limit = _reservation_budget_limit(
                max_cost="0.25",
                window=BudgetWindow.rolling(seconds=60),
            )
            all_time_limit = _reservation_budget_limit(max_cost="0.25")
            rolling_first = await ledger.reserve(
                limit=rolling_limit,
                session_id="sess_rolling_1",
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
            )
            all_time_first = await ledger.reserve(
                limit=all_time_limit,
                session_id="sess_all_time_1",
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
            )
            clock.value = datetime(2026, 1, 1, 12, 0, 30, tzinfo=UTC)
            blocked_now = await ledger.reserve(
                limit=rolling_limit,
                session_id="sess_rolling_blocked",
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
            )
            clock.value = datetime(2026, 1, 1, 12, 2, tzinfo=UTC)
            rolling_second = await ledger.reserve(
                limit=rolling_limit,
                session_id="sess_rolling_2",
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
            )
            all_time_second = await ledger.reserve(
                limit=all_time_limit,
                session_id="sess_all_time_2",
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
            )
            return rolling_first, blocked_now, rolling_second, all_time_first, all_time_second
        finally:
            await ledger.close()

    rolling_first, blocked_now, rolling_second, all_time_first, all_time_second = asyncio.run(run())

    assert rolling_first.accepted is True
    assert blocked_now.accepted is False
    assert blocked_now.actual == Decimal("0.44")
    assert rolling_second.accepted is False
    assert rolling_second.actual == Decimal("0.44")
    assert all_time_first.accepted is True
    assert all_time_second.accepted is False
    assert all_time_second.actual == Decimal("0.44")


def test_sqlite_budget_ledger_uses_reconciliation_time_for_rolling_window(tmp_path) -> None:
    async def run():
        clock = MutableClock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
        ledger = SQLiteBudgetLedger(tmp_path / "budget.sqlite", clock=clock)
        try:
            limit = _reservation_budget_limit(
                max_cost="0.25",
                window=BudgetWindow.rolling(seconds=60),
            )
            first = await ledger.reserve(
                limit=limit,
                session_id="sess_1",
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
            )
            assert first.record is not None
            await ledger.reconcile(
                reservation_id=first.record.reservation_id,
                actual_amount=Decimal("0.22"),
                occurred_at=datetime(2026, 1, 1, 12, 2, tzinfo=UTC),
            )
            clock.value = datetime(2026, 1, 1, 12, 2, 30, tzinfo=UTC)
            blocked = await ledger.reserve(
                limit=limit,
                session_id="sess_2",
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
            )
            clock.value = datetime(2026, 1, 1, 12, 3, 1, tzinfo=UTC)
            accepted = await ledger.reserve(
                limit=limit,
                session_id="sess_3",
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
            )
            return blocked, accepted
        finally:
            await ledger.close()

    blocked, accepted = asyncio.run(run())

    assert blocked.accepted is False
    assert blocked.actual == Decimal("0.44")
    assert accepted.accepted is True
    assert accepted.actual == Decimal("0.22")


def test_sqlite_budget_ledger_uses_reconciliation_time_for_calendar_window(tmp_path) -> None:
    async def run():
        clock = MutableClock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
        ledger = SQLiteBudgetLedger(tmp_path / "budget.sqlite", clock=clock)
        try:
            limit = _reservation_budget_limit(
                max_cost="0.25",
                window=BudgetWindow.calendar(period="day", timezone="UTC"),
            )
            first = await ledger.reserve(
                limit=limit,
                session_id="sess_1",
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
            )
            assert first.record is not None
            await ledger.reconcile(
                reservation_id=first.record.reservation_id,
                actual_amount=Decimal("0.22"),
                occurred_at=datetime(2026, 1, 1, 23, 59, tzinfo=UTC),
            )
            clock.value = datetime(2026, 1, 2, 0, 1, tzinfo=UTC)
            next_day = await ledger.reserve(
                limit=limit,
                session_id="sess_2",
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
            )
            active = await ledger.reserve(
                limit=limit,
                session_id="sess_3",
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
            )
            return next_day, active
        finally:
            await ledger.close()

    next_day, active = asyncio.run(run())

    assert next_day.accepted is True
    assert next_day.window.storage_key == "calendar:day:UTC"
    assert active.accepted is False
    assert active.actual == Decimal("0.44")


def test_in_memory_budget_ledger_reaps_expired_active_reservations() -> None:
    async def run():
        clock = MutableClock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
        ledger = InMemoryBudgetLedger(clock=clock, reservation_ttl_seconds=60)
        limit = _reservation_budget_limit(max_cost="0.25")
        orphaned = await ledger.reserve(
            limit=limit,
            session_id="sess_orphaned",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
        )
        assert orphaned.accepted is True
        assert orphaned.record is not None
        clock.value = datetime(2026, 1, 1, 12, 1, tzinfo=UTC)
        recovered = await ledger.reserve(
            limit=limit,
            session_id="sess_recovered",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
        )
        # V3: reconciling a reservation reaped while still in flight records the actual
        # spend instead of crashing the billed run (which would also undercount).
        reconciled = await ledger.reconcile(
            reservation_id=orphaned.record.reservation_id,
            actual_amount=Decimal("0.01"),
        )
        return recovered, reconciled

    recovered, reconciled = asyncio.run(run())

    assert recovered.accepted is True
    assert recovered.actual == Decimal("0.22")
    assert reconciled.status == "reconciled"
    assert reconciled.actual_amount == Decimal("0.01")


def test_in_memory_budget_ledger_heartbeat_keeps_live_reservation_active() -> None:
    async def run():
        clock = MutableClock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
        ledger = InMemoryBudgetLedger(clock=clock, reservation_ttl_seconds=60)
        limit = _reservation_budget_limit(max_cost="0.25")
        first = await ledger.reserve(
            limit=limit,
            session_id="sess_live",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
        )
        assert first.record is not None
        clock.value = datetime(2026, 1, 1, 12, 0, 30, tzinfo=UTC)
        assert await ledger.heartbeat(reservation_id=first.record.reservation_id) is True
        clock.value = datetime(2026, 1, 1, 12, 1, 1, tzinfo=UTC)
        blocked = await ledger.reserve(
            limit=limit,
            session_id="sess_blocked",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
        )
        clock.value = datetime(2026, 1, 1, 12, 1, 30, tzinfo=UTC)
        late_heartbeat = await ledger.heartbeat(reservation_id=first.record.reservation_id)
        recovered = await ledger.reserve(
            limit=limit,
            session_id="sess_recovered",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
        )
        return blocked, late_heartbeat, recovered

    blocked, late_heartbeat, recovered = asyncio.run(run())

    assert blocked.accepted is False
    assert late_heartbeat is False
    assert recovered.accepted is True


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_budget_ledgers_reap_only_the_matching_budget(tmp_path, backend: str) -> None:
    async def run():
        clock = MutableClock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
        ledger = (
            InMemoryBudgetLedger(clock=clock, reservation_ttl_seconds=60)
            if backend == "memory"
            else SQLiteBudgetLedger(
                tmp_path / "budget.sqlite",
                clock=clock,
                reservation_ttl_seconds=60,
            )
        )
        try:
            first_limit = _reservation_budget_limit(max_cost="0.25", key="first")
            second_limit = _reservation_budget_limit(max_cost="0.25", key="second")
            first = await ledger.reserve(
                limit=first_limit,
                session_id="sess_first_expired",
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
            )
            second = await ledger.reserve(
                limit=second_limit,
                session_id="sess_second_expired",
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
            )
            assert first.record is not None
            assert second.record is not None
            clock.value = datetime(2026, 1, 1, 12, 1, tzinfo=UTC)
            replacement = await ledger.reserve(
                limit=first_limit,
                session_id="sess_first_replacement",
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
            )
            untouched = await ledger.release(
                reservation_id=second.record.reservation_id,
                reason="manual cleanup",
            )
            return replacement, untouched
        finally:
            close = getattr(ledger, "close", None)
            if close is not None:
                await close()

    replacement, untouched = asyncio.run(run())

    assert replacement.accepted is True
    assert untouched.reason == "manual cleanup"


def test_in_memory_budget_ledger_release_tolerates_ttl_reaped_reservation() -> None:
    async def run():
        clock = MutableClock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
        ledger = InMemoryBudgetLedger(clock=clock, reservation_ttl_seconds=60)
        limit = _reservation_budget_limit(max_cost="0.25")
        orphaned = await ledger.reserve(
            limit=limit,
            session_id="sess_orphaned",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
        )
        assert orphaned.accepted is True
        assert orphaned.record is not None
        clock.value = datetime(2026, 1, 1, 12, 1, tzinfo=UTC)
        recovered = await ledger.reserve(
            limit=limit,
            session_id="sess_recovered",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
        )
        released = await ledger.release(
            reservation_id=orphaned.record.reservation_id,
            reason="cleanup",
        )
        return recovered, released

    recovered, released = asyncio.run(run())

    assert recovered.accepted is True
    assert released.status == "released"
    assert released.actual_amount is None
    assert released.released_amount == released.reserved_amount
    assert released.reason == "Reservation expired: not reconciled within 60s."


def test_in_memory_budget_ledger_reservation_ttl_none_disables_reap() -> None:
    async def run():
        clock = MutableClock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
        ledger = InMemoryBudgetLedger(clock=clock, reservation_ttl_seconds=None)
        limit = _reservation_budget_limit(max_cost="0.25")
        first = await ledger.reserve(
            limit=limit,
            session_id="sess_1",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
        )
        assert first.accepted is True
        clock.value = datetime(2026, 1, 2, 12, 0, tzinfo=UTC)
        blocked = await ledger.reserve(
            limit=limit,
            session_id="sess_2",
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
        )
        return blocked

    blocked = asyncio.run(run())

    assert blocked.accepted is False
    assert blocked.actual == Decimal("0.44")


def test_budget_ledgers_reject_invalid_reservation_ttl(tmp_path) -> None:
    with pytest.raises(ValueError, match="positive integer or None"):
        InMemoryBudgetLedger(reservation_ttl_seconds=0)
    with pytest.raises(ValueError, match="positive integer or None"):
        SQLiteBudgetLedger(tmp_path / "budget.sqlite", reservation_ttl_seconds=-5)


def test_sqlite_budget_ledger_reaps_expired_active_reservations(tmp_path) -> None:
    async def run():
        clock = MutableClock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
        ledger = SQLiteBudgetLedger(
            tmp_path / "budget.sqlite",
            clock=clock,
            reservation_ttl_seconds=60,
        )
        try:
            limit = _reservation_budget_limit(max_cost="0.25")
            orphaned = await ledger.reserve(
                limit=limit,
                session_id="sess_orphaned",
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
            )
            assert orphaned.accepted is True
            assert orphaned.record is not None
            clock.value = datetime(2026, 1, 1, 12, 1, tzinfo=UTC)
            recovered = await ledger.reserve(
                limit=limit,
                session_id="sess_recovered",
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
            )
            # V3: reconciling a reservation reaped while still in flight records the
            # actual spend instead of crashing the billed run (which would undercount).
            reconciled = await ledger.reconcile(
                reservation_id=orphaned.record.reservation_id,
                actual_amount=Decimal("0.01"),
            )
            return recovered, reconciled
        finally:
            await ledger.close()

    recovered, reconciled = asyncio.run(run())

    assert recovered.accepted is True
    assert recovered.actual == Decimal("0.22")
    assert reconciled.status == "reconciled"
    assert reconciled.actual_amount == Decimal("0.01")


def test_sqlite_budget_ledger_heartbeat_keeps_live_reservation_active(tmp_path) -> None:
    async def run():
        clock = MutableClock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
        ledger = SQLiteBudgetLedger(
            tmp_path / "budget.sqlite",
            clock=clock,
            reservation_ttl_seconds=60,
        )
        try:
            limit = _reservation_budget_limit(max_cost="0.25")
            first = await ledger.reserve(
                limit=limit,
                session_id="sess_live",
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
            )
            assert first.record is not None
            clock.value = datetime(2026, 1, 1, 12, 0, 30, tzinfo=UTC)
            assert await ledger.heartbeat(reservation_id=first.record.reservation_id) is True
            clock.value = datetime(2026, 1, 1, 12, 1, 1, tzinfo=UTC)
            blocked = await ledger.reserve(
                limit=limit,
                session_id="sess_blocked",
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
            )
            clock.value = datetime(2026, 1, 1, 12, 1, 30, tzinfo=UTC)
            late_heartbeat = await ledger.heartbeat(reservation_id=first.record.reservation_id)
            recovered = await ledger.reserve(
                limit=limit,
                session_id="sess_recovered",
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
            )
            return blocked, late_heartbeat, recovered
        finally:
            await ledger.close()

    blocked, late_heartbeat, recovered = asyncio.run(run())

    assert blocked.accepted is False
    assert late_heartbeat is False
    assert recovered.accepted is True


def test_sqlite_budget_ledger_release_tolerates_ttl_reaped_reservation(tmp_path) -> None:
    async def run():
        clock = MutableClock(datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
        ledger = SQLiteBudgetLedger(
            tmp_path / "budget.sqlite",
            clock=clock,
            reservation_ttl_seconds=60,
        )
        try:
            limit = _reservation_budget_limit(max_cost="0.25")
            orphaned = await ledger.reserve(
                limit=limit,
                session_id="sess_orphaned",
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
            )
            assert orphaned.accepted is True
            assert orphaned.record is not None
            clock.value = datetime(2026, 1, 1, 12, 1, tzinfo=UTC)
            recovered = await ledger.reserve(
                limit=limit,
                session_id="sess_recovered",
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
            )
            released = await ledger.release(
                reservation_id=orphaned.record.reservation_id,
                reason="cleanup",
            )
            return recovered, released
        finally:
            await ledger.close()

    recovered, released = asyncio.run(run())

    assert recovered.accepted is True
    assert released.status == "released"
    assert released.actual_amount is None
    assert released.released_amount == released.reserved_amount
    assert released.reason == "Reservation expired: not reconciled within 60s."


def test_sqlite_budget_ledger_database_can_be_shared_with_session_store(tmp_path) -> None:
    async def run():
        path = tmp_path / "shared.sqlite"
        ledger = SQLiteBudgetLedger(path)
        await ledger.close()
        session_store = SQLiteSessionStore(path)
        try:
            session = await session_store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_shared_budget_db",
                    messages=[Message.text("user", "hello")],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
            return session
        finally:
            await session_store.close()

    session = asyncio.run(run())

    assert session.id == "sess_shared_budget_db"


def test_sqlite_budget_ledger_migrates_legacy_unprefixed_table(tmp_path) -> None:
    # Before ADR 0001 revision 8 the ledger created an ad-hoc unprefixed
    # `budget_reservations` table. Opening such a database must carry active
    # reservations into `cayu_budget_reservations` and drop the legacy table.
    path = tmp_path / "budget.sqlite"
    now = datetime.now(UTC).isoformat()
    legacy = sqlite3.connect(path)
    legacy.execute(
        """
        CREATE TABLE budget_reservations (
            reservation_id TEXT PRIMARY KEY,
            scope TEXT NOT NULL,
            budget_key TEXT,
            window TEXT NOT NULL,
            currency TEXT NOT NULL,
            session_id TEXT NOT NULL,
            agent_name TEXT NOT NULL,
            provider_name TEXT NOT NULL,
            model TEXT NOT NULL,
            reserved_amount TEXT NOT NULL,
            actual_amount TEXT,
            status TEXT NOT NULL,
            reason TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    legacy.execute(
        "INSERT INTO budget_reservations VALUES "
        "(?, 'app', NULL, 'all_time', 'USD', 'sess_legacy', 'assistant', "
        "'fake', 'fake-model', '0.22', NULL, 'active', NULL, ?, ?)",
        ("bres_legacy", now, now),
    )
    legacy.commit()
    legacy.close()

    async def run():
        ledger = SQLiteBudgetLedger(path)
        try:
            limit = _reservation_budget_limit(max_cost="0.25")
            blocked = await ledger.reserve(
                limit=limit,
                session_id="sess_1",
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
            )
            reconciled = await ledger.reconcile(
                reservation_id="bres_legacy",
                actual_amount=Decimal("0.01"),
            )
            accepted = await ledger.reserve(
                limit=limit,
                session_id="sess_2",
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
            )
            return blocked, reconciled, accepted
        finally:
            await ledger.close()

    blocked, reconciled, accepted = asyncio.run(run())

    # The migrated legacy reservation still counts against the budget…
    assert blocked.accepted is False
    assert blocked.actual == Decimal("0.44")
    # …and remains reconcilable under its original reservation_id.
    assert reconciled.released_amount == Decimal("0.21")
    assert accepted.accepted is True

    inspector = sqlite3.connect(path)
    try:
        tables = {
            row[0]
            for row in inspector.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    finally:
        inspector.close()
    assert "cayu_budget_reservations" in tables
    assert "budget_reservations" not in tables


def test_in_memory_budget_store_filters_app_and_agent_events() -> None:
    async def run():
        store = InMemoryBudgetStore()
        await store.append_event(
            Event(
                type=EventType.MODEL_COMPLETED,
                session_id="sess_1",
                agent_name="builder",
                payload={
                    "usage_metrics": {
                        "provider_name": "fake",
                        "model": "fake-model",
                        "input_tokens": 1,
                        "output_tokens": 0,
                        "total_tokens": 1,
                    }
                },
            )
        )
        await store.append_event(
            Event(
                type=EventType.MODEL_COMPLETED,
                session_id="sess_2",
                agent_name="researcher",
                payload={
                    "usage_metrics": {
                        "provider_name": "fake",
                        "model": "fake-model",
                        "input_tokens": 2,
                        "output_tokens": 0,
                        "total_tokens": 2,
                    }
                },
            )
        )
        app_events = await store.load_events_for_budget(
            scope="app",
            key=None,
            window="all_time",
        )
        builder_events = await store.load_events_for_budget(
            scope="agent",
            key="builder",
            window="all_time",
        )
        return app_events, builder_events

    app_events, builder_events = asyncio.run(run())

    assert len(app_events) == 2
    assert len(builder_events) == 1
    assert builder_events[0].agent_name == "builder"


def test_in_memory_budget_store_filters_rolling_window_events() -> None:
    async def run():
        store = InMemoryBudgetStore()
        await store.append_event(
            Event(
                type=EventType.MODEL_COMPLETED,
                session_id="sess_old",
                agent_name="builder",
                timestamp=datetime.now(UTC) - timedelta(seconds=120),
                payload={
                    "usage_metrics": {
                        "provider_name": "fake",
                        "model": "fake-model",
                        "input_tokens": 1,
                        "output_tokens": 0,
                        "total_tokens": 1,
                    }
                },
            )
        )
        await store.append_event(
            Event(
                type=EventType.MODEL_COMPLETED,
                session_id="sess_recent",
                agent_name="builder",
                timestamp=datetime.now(UTC),
                payload={
                    "usage_metrics": {
                        "provider_name": "fake",
                        "model": "fake-model",
                        "input_tokens": 2,
                        "output_tokens": 0,
                        "total_tokens": 2,
                    }
                },
            )
        )

        all_time_events = await store.load_events_for_budget(
            scope="app",
            key=None,
            window=BudgetWindow.all_time(),
        )
        rolling_events = await store.load_events_for_budget(
            scope="app",
            key=None,
            window=BudgetWindow.rolling(seconds=60),
        )
        return all_time_events, rolling_events

    all_time_events, rolling_events = asyncio.run(run())

    assert [event.session_id for event in all_time_events] == ["sess_old", "sess_recent"]
    assert [event.session_id for event in rolling_events] == ["sess_recent"]


def test_events_for_budget_window_uses_caller_supplied_now() -> None:
    now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    boundary_event = Event(
        type=EventType.MODEL_COMPLETED,
        session_id="sess_boundary",
        timestamp=now - timedelta(seconds=60),
    )
    old_event = Event(
        type=EventType.MODEL_COMPLETED,
        session_id="sess_old",
        timestamp=now - timedelta(seconds=61),
    )

    filtered = events_for_budget_window(
        [old_event, boundary_event],
        BudgetWindow.rolling(seconds=60),
        now=now,
    )

    assert [event.session_id for event in filtered] == ["sess_boundary"]


def test_events_for_budget_window_filters_calendar_day() -> None:
    now = datetime(2026, 6, 19, 13, 30, tzinfo=UTC)
    previous_day = Event(
        type=EventType.MODEL_COMPLETED,
        session_id="sess_previous_day",
        timestamp=datetime(2026, 6, 19, 3, 59, 59, tzinfo=UTC),
    )
    current_day = Event(
        type=EventType.MODEL_COMPLETED,
        session_id="sess_current_day",
        timestamp=datetime(2026, 6, 19, 4, 0, tzinfo=UTC),
    )
    next_day = Event(
        type=EventType.MODEL_COMPLETED,
        session_id="sess_next_day",
        timestamp=datetime(2026, 6, 20, 4, 0, tzinfo=UTC),
    )

    filtered = events_for_budget_window(
        [previous_day, current_day, next_day],
        BudgetWindow.calendar(period="day", timezone="America/New_York"),
        now=now,
    )

    assert [event.session_id for event in filtered] == ["sess_current_day"]


def test_session_budget_store_reads_model_events_from_session_store() -> None:
    async def run():
        session_store = InMemorySessionStore()
        await session_store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_builder",
                causal_budget_id="job_shared",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await session_store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_other_job",
                causal_budget_id="job_other",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await session_store.append_event(
            "sess_builder",
            Event(
                type=EventType.MODEL_COMPLETED,
                session_id="sess_builder",
                agent_name="builder",
                timestamp=datetime.now(UTC) - timedelta(seconds=120),
                payload={
                    "usage_metrics": {
                        "provider_name": "fake",
                        "model": "fake-model",
                        "input_tokens": 1,
                        "output_tokens": 0,
                        "total_tokens": 1,
                    }
                },
            ),
        )
        await session_store.append_event(
            "sess_other_job",
            Event(
                type=EventType.MODEL_COMPLETED,
                session_id="sess_other_job",
                agent_name="builder",
                timestamp=datetime.now(UTC),
                payload={
                    "usage_metrics": {
                        "provider_name": "fake",
                        "model": "fake-model",
                        "input_tokens": 2,
                        "output_tokens": 0,
                        "total_tokens": 2,
                    }
                },
            ),
        )
        budget_store = SessionBudgetStore(session_store)
        agent_events = await budget_store.load_events_for_budget(
            scope="agent",
            key="builder",
            window="all_time",
        )
        causal_events = await budget_store.load_events_for_budget(
            scope="causal",
            key="job_shared",
            window="all_time",
        )
        rolling_events = await budget_store.load_events_for_budget(
            scope="agent",
            key="builder",
            window=BudgetWindow.rolling(seconds=60),
        )
        return agent_events, causal_events, rolling_events

    agent_events, causal_events, rolling_events = asyncio.run(run())

    assert len(agent_events) == 2
    assert len(causal_events) == 1
    assert causal_events[0].type == EventType.MODEL_COMPLETED
    assert causal_events[0].session_id == "sess_builder"
    assert [event.session_id for event in rolling_events] == ["sess_other_job"]


def test_session_budget_store_filters_calendar_window_events() -> None:
    async def run():
        session_store = InMemorySessionStore()
        window = BudgetWindow.calendar(period="day", timezone="UTC")
        since, until = window.bounds()
        assert since is not None
        assert until is not None
        for session_id in ("sess_previous", "sess_current", "sess_next"):
            await session_store.create(
                RunRequest(
                    agent_name="builder",
                    session_id=session_id,
                    messages=[Message.text("user", session_id)],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
        for session_id, timestamp in (
            ("sess_previous", since - timedelta(seconds=1)),
            ("sess_current", since + ((until - since) / 2)),
            ("sess_next", until),
        ):
            await session_store.append_event(
                session_id,
                Event(
                    type=EventType.MODEL_COMPLETED,
                    session_id=session_id,
                    agent_name="builder",
                    timestamp=timestamp,
                    payload={
                        "usage_metrics": {
                            "provider_name": "fake",
                            "model": "fake-model",
                            "input_tokens": 1,
                            "output_tokens": 0,
                            "total_tokens": 1,
                        }
                    },
                ),
            )
        budget_store = SessionBudgetStore(session_store)
        return await budget_store.load_events_for_budget(
            scope="agent",
            key="builder",
            window=window,
        )

    events = asyncio.run(run())

    assert [event.session_id for event in events] == ["sess_current"]


def test_session_budget_store_reads_model_events_from_sqlite_store(tmp_path) -> None:
    async def run():
        session_store = SQLiteSessionStore(tmp_path / "sessions.sqlite")
        try:
            await session_store.create(
                RunRequest(
                    agent_name="builder",
                    session_id="sess_builder",
                    causal_budget_id="job_shared",
                    messages=[Message.text("user", "hello")],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
            await session_store.create(
                RunRequest(
                    agent_name="researcher",
                    session_id="sess_researcher",
                    causal_budget_id="job_other",
                    messages=[Message.text("user", "hello")],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
            await session_store.append_event(
                "sess_builder",
                Event(
                    type=EventType.MODEL_COMPLETED,
                    session_id="sess_builder",
                    agent_name="builder",
                    payload={
                        "usage_metrics": {
                            "provider_name": "fake",
                            "model": "fake-model",
                            "input_tokens": 1,
                            "output_tokens": 0,
                            "total_tokens": 1,
                        }
                    },
                ),
            )
            await session_store.append_event(
                "sess_researcher",
                Event(
                    type=EventType.MODEL_COMPLETED,
                    session_id="sess_researcher",
                    agent_name="researcher",
                    payload={
                        "usage_metrics": {
                            "provider_name": "fake",
                            "model": "fake-model",
                            "input_tokens": 2,
                            "output_tokens": 0,
                            "total_tokens": 2,
                        }
                    },
                ),
            )
            budget_store = SessionBudgetStore(session_store)
            app_events = await budget_store.load_events_for_budget(
                scope="app",
                key=None,
                window="all_time",
            )
            builder_events = await budget_store.load_events_for_budget(
                scope="agent",
                key="builder",
                window="all_time",
            )
            causal_events = await budget_store.load_events_for_budget(
                scope="causal",
                key="job_shared",
                window="all_time",
            )
            return app_events, builder_events, causal_events
        finally:
            await session_store.close()

    app_events, builder_events, causal_events = asyncio.run(run())

    assert len(app_events) == 2
    assert len(builder_events) == 1
    assert builder_events[0].session_id == "sess_builder"
    assert len(causal_events) == 1
    assert causal_events[0].session_id == "sess_builder"


def test_cayu_app_exposes_causal_budget_usage_and_cost() -> None:
    async def run():
        session_store = InMemorySessionStore()
        app = CayuApp(session_store=session_store)
        await session_store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_parent",
                causal_budget_id="job_shared",
                messages=[Message.text("user", "parent")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await session_store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_child",
                causal_budget_id="job_shared",
                messages=[Message.text("user", "child")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await session_store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_other",
                causal_budget_id="job_other",
                messages=[Message.text("user", "other")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        for session_id, input_tokens in (
            ("sess_parent", 1000),
            ("sess_child", 500),
            ("sess_other", 9000),
        ):
            await session_store.append_event(
                session_id,
                Event(
                    type=EventType.MODEL_COMPLETED,
                    session_id=session_id,
                    payload={
                        "usage_metrics": {
                            "provider_name": "fake",
                            "model": "fake-model",
                            "input_tokens": input_tokens,
                            "output_tokens": 100,
                            "total_tokens": input_tokens + 100,
                        }
                    },
                ),
            )
        await session_store.append_event(
            "sess_child",
            Event(
                type=EventType.TOOL_CALL_STARTED,
                session_id="sess_child",
                tool_name="read_file",
            ),
        )
        pricing = PricingCatalog(
            prices=(
                ModelPricing(
                    provider_name="fake",
                    model="fake-model",
                    input_per_million=Decimal("2"),
                    output_per_million=Decimal("8"),
                ),
            )
        )
        usage = await app.get_causal_budget_usage("job_shared")
        cost = await app.get_causal_budget_cost("job_shared", pricing)
        return usage, cost

    usage, cost = asyncio.run(run())

    assert usage.causal_budget_id == "job_shared"
    assert usage.session_ids == ["sess_parent", "sess_child"]
    assert usage.session_count == 2
    assert usage.model_steps == 2
    assert usage.tool_calls == 1
    assert usage.usage.input_tokens == 1500
    assert usage.usage.output_tokens == 200
    assert [item.session_id for item in usage.session_summaries] == [
        "sess_parent",
        "sess_child",
    ]
    assert usage.session_summaries[0].usage.input_tokens == 1000
    assert usage.session_summaries[1].usage.input_tokens == 500
    assert cost.causal_budget_id == "job_shared"
    assert cost.session_ids == ["sess_parent", "sess_child"]
    assert cost.session_count == 2
    assert cost.model_steps == 2
    assert cost.total_cost == Decimal("0.0046")
    assert [item.session_id for item in cost.session_costs] == [
        "sess_parent",
        "sess_child",
    ]
    assert cost.session_costs[0].total_cost == Decimal("0.0028")
    assert cost.session_costs[1].total_cost == Decimal("0.0018")


def test_budget_check_fails_closed_for_unpriced_model_steps() -> None:
    pricing = PricingCatalog(
        prices=(
            ModelPricing(
                provider_name="fake",
                model="other-model",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("1"),
            ),
        )
    )
    check = budget_check_from_events(
        limit=BudgetLimit(
            scope="app",
            max_estimated_cost=Decimal("1"),
            pricing=pricing,
        ),
        events=[
            Event(
                type=EventType.MODEL_COMPLETED,
                session_id="sess_1",
                payload={
                    "usage_metrics": {
                        "provider_name": "fake",
                        "model": "fake-model",
                        "input_tokens": 1,
                        "output_tokens": 0,
                        "total_tokens": 1,
                    }
                },
            )
        ],
    )

    assert check.limit_reached is True
    assert check.unpriced_model_steps == 1
    assert "no matching pricing" in check.message


def test_normalize_anthropic_usage_metrics() -> None:
    metrics = normalize_usage_metrics(
        provider_name="anthropic",
        model="claude-sonnet-4-6",
        raw_usage={
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_read_input_tokens": 70,
            "cache_creation_input_tokens": 0,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 2,
                "ephemeral_1h_input_tokens": 3,
            },
        },
    )

    assert metrics is not None
    assert metrics.provider_name == "anthropic"
    assert metrics.model == "claude-sonnet-4-6"
    assert metrics.input_tokens == 175
    assert metrics.output_tokens == 20
    assert metrics.total_tokens == 195
    assert metrics.cache.read_tokens == 70
    assert metrics.cache.write_tokens == 5
    assert metrics.cache.cached_input_tokens == 70
    assert metrics.cache.uncached_input_tokens == 100


def test_normalize_vertex_usage_metrics_matches_anthropic() -> None:
    # Claude on Vertex returns the Anthropic-shaped usage payload, so the "vertex"
    # provider must fold cache tokens into input exactly like "anthropic".
    raw_usage = {
        "input_tokens": 100,
        "output_tokens": 20,
        "cache_read_input_tokens": 70,
        "cache_creation_input_tokens": 0,
        "cache_creation": {
            "ephemeral_5m_input_tokens": 2,
            "ephemeral_1h_input_tokens": 3,
        },
    }
    metrics = normalize_usage_metrics(
        provider_name="vertex", model="claude-sonnet-4-6", raw_usage=dict(raw_usage)
    )

    assert metrics is not None
    assert metrics.input_tokens == 175
    assert metrics.total_tokens == 195
    assert metrics.cache.read_tokens == 70
    assert metrics.cache.write_tokens == 5
    assert metrics.cache.cached_input_tokens == 70
    assert metrics.cache.uncached_input_tokens == 100


def test_normalize_bedrock_anthropic_shape_by_payload() -> None:
    # Claude reached through Bedrock/a gateway registers under a name that is not
    # in the built-in alias set. The payload shape (Anthropic-only cache fields)
    # must still drive cache-token folding so tokens are not undercounted.
    raw_usage = {
        "input_tokens": 100,
        "output_tokens": 20,
        "cache_read_input_tokens": 70,
        "cache_creation_input_tokens": 5,
    }
    metrics = normalize_usage_metrics(
        provider_name="bedrock", model="claude-sonnet-4-6", raw_usage=dict(raw_usage)
    )

    assert metrics is not None
    assert metrics.input_tokens == 175
    assert metrics.total_tokens == 195
    assert metrics.cache.read_tokens == 70
    assert metrics.cache.write_tokens == 5
    assert metrics.cache.cached_input_tokens == 70
    assert metrics.cache.uncached_input_tokens == 100


def test_normalize_declared_anthropic_dialect_overrides_unknown_name() -> None:
    # A renamed adapter can declare its dialect even when the payload carries no
    # cache fields yet (so shape detection alone cannot classify it).
    metrics = normalize_usage_metrics(
        provider_name="claude-proxy",
        model="claude-sonnet-4-6",
        raw_usage={
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_read_input_tokens": 70,
            "cache_creation_input_tokens": 0,
        },
        usage_dialect=UsageDialect.ANTHROPIC,
    )

    assert metrics is not None
    assert metrics.input_tokens == 170
    assert metrics.cache.read_tokens == 70
    assert metrics.cache.cached_input_tokens == 70
    assert metrics.cache.uncached_input_tokens == 100


def test_normalize_declared_dialect_beats_name_alias() -> None:
    # An explicit dialect wins over the built-in name allowlist: a payload routed
    # through a provider registered as "vertex" but declaring the OpenAI dialect
    # must not fold cache tokens into input.
    metrics = normalize_usage_metrics(
        provider_name="vertex",
        model="gpt-5.5",
        raw_usage={
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 40},
        },
        usage_dialect="openai",
    )

    assert metrics is not None
    assert metrics.input_tokens == 100
    assert metrics.cache.read_tokens == 40
    assert metrics.cache.cached_input_tokens == 40
    assert metrics.cache.uncached_input_tokens == 60


def test_normalize_openai_usage_dialect_enum_marks_cached_tokens_as_reads() -> None:
    metrics = normalize_usage_metrics(
        provider_name="renamed-chat-provider",
        model="chat-model",
        raw_usage={
            "prompt_tokens": 9,
            "completion_tokens": 2,
            "prompt_tokens_details": {"cached_tokens": 3},
        },
        usage_dialect=UsageDialect.OPENAI,
    )

    assert metrics is not None
    assert metrics.cache.read_tokens == 3
    assert metrics.cache.cached_input_tokens == 3


def test_normalize_unknown_dialect_falls_back_to_detection() -> None:
    # An unrecognized dialect string is treated as "auto" and detection applies.
    metrics = normalize_usage_metrics(
        provider_name="gateway",
        model="claude-sonnet-4-6",
        raw_usage={
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_read_input_tokens": 30,
        },
        usage_dialect="mystery",
    )

    assert metrics is not None
    assert metrics.input_tokens == 130
    assert metrics.cache.read_tokens == 30


def test_normalize_anthropic_top_level_cache_write_counter() -> None:
    metrics = normalize_usage_metrics(
        provider_name="anthropic",
        model="claude-sonnet-4-6",
        raw_usage={
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_creation_input_tokens": 10,
        },
    )

    assert metrics is not None
    assert metrics.cache.write_tokens == 10


def test_session_usage_summary_aggregates_model_steps_and_tools() -> None:
    events = [
        Event(
            type=EventType.MODEL_COMPLETED,
            session_id="session_1",
            payload={
                "usage_metrics": {
                    "provider_name": "openai",
                    "model": "gpt-5.5",
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "total_tokens": 120,
                    "reasoning_output_tokens": 4,
                    "cache": {
                        "read_tokens": 60,
                        "write_tokens": 0,
                        "cached_input_tokens": 60,
                        "uncached_input_tokens": 40,
                    },
                }
            },
        ),
        Event(type=EventType.TOOL_CALL_STARTED, session_id="session_1", tool_name="read_file"),
        Event(
            type=EventType.MODEL_COMPLETED,
            session_id="session_1",
            payload={
                "usage": {
                    "input_tokens": 50,
                    "output_tokens": 10,
                    "cache_read_input_tokens": 5,
                },
                "provider_name": "anthropic",
                "model": "claude-sonnet-4-6",
            },
        ),
    ]

    summary = session_usage_summary("session_1", events)

    assert summary.session_id == "session_1"
    assert summary.model_steps == 2
    assert summary.tool_calls == 1
    assert summary.provider_names == ["openai", "anthropic"]
    assert summary.models == ["gpt-5.5", "claude-sonnet-4-6"]
    assert summary.usage.input_tokens == 155
    assert summary.usage.output_tokens == 30
    assert summary.usage.total_tokens == 185
    assert summary.usage.reasoning_output_tokens == 4
    assert summary.usage.cache.read_tokens == 65
    assert summary.usage.cache.cached_input_tokens == 65
    assert summary.usage.cache.uncached_input_tokens == 90


def test_causal_budget_usage_summary_aggregates_related_sessions() -> None:
    events = [
        Event(
            type=EventType.MODEL_COMPLETED,
            session_id="sess_parent",
            payload={
                "usage_metrics": {
                    "provider_name": "openai",
                    "model": "gpt-5.5",
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "total_tokens": 120,
                }
            },
        ),
        Event(type=EventType.TOOL_CALL_STARTED, session_id="sess_child", tool_name="read_file"),
        Event(
            type=EventType.MODEL_COMPLETED,
            session_id="sess_child",
            payload={
                "usage_metrics": {
                    "provider_name": "openai",
                    "model": "gpt-5.5",
                    "input_tokens": 50,
                    "output_tokens": 10,
                    "total_tokens": 60,
                }
            },
        ),
    ]

    summary = causal_budget_usage_summary(
        causal_budget_id="job_shared",
        session_ids=["sess_parent", "sess_child"],
        events=events,
    )

    assert summary.causal_budget_id == "job_shared"
    assert summary.session_ids == ["sess_parent", "sess_child"]
    assert summary.session_count == 2
    assert summary.model_steps == 2
    assert summary.tool_calls == 1
    assert summary.provider_names == ["openai"]
    assert summary.models == ["gpt-5.5"]
    assert summary.usage.input_tokens == 150
    assert summary.usage.output_tokens == 30
    assert summary.usage.total_tokens == 180
    assert [item.session_id for item in summary.session_summaries] == [
        "sess_parent",
        "sess_child",
    ]
    assert summary.session_summaries[0].usage.input_tokens == 100
    assert summary.session_summaries[1].tool_calls == 1


def test_estimate_session_cost_prices_each_model_step() -> None:
    events = [
        Event(
            type=EventType.MODEL_COMPLETED,
            session_id="session_1",
            payload={
                "usage_metrics": {
                    "provider_name": "openai",
                    "model": "gpt-5.5-2026-04-23",
                    "input_tokens": 1000,
                    "output_tokens": 200,
                    "total_tokens": 1200,
                    "reasoning_output_tokens": 0,
                    "cache": {
                        "read_tokens": 400,
                        "write_tokens": 0,
                        "cached_input_tokens": 400,
                        "uncached_input_tokens": 600,
                    },
                }
            },
        ),
        Event(
            type=EventType.MODEL_COMPLETED,
            session_id="session_1",
            payload={
                "usage_metrics": {
                    "provider_name": "anthropic",
                    "model": "claude-sonnet-4-6",
                    "input_tokens": 1000,
                    "output_tokens": 100,
                    "total_tokens": 1100,
                    "reasoning_output_tokens": 0,
                    "cache": {
                        "read_tokens": 300,
                        "write_tokens": 200,
                        "cached_input_tokens": 300,
                        "uncached_input_tokens": 500,
                    },
                }
            },
        ),
    ]
    pricing = PricingCatalog(
        prices=(
            ModelPricing(
                provider_name="openai",
                model="gpt-5.5",
                match="prefix",
                input_per_million=Decimal("2"),
                output_per_million=Decimal("8"),
                cache_read_input_per_million=Decimal("0.5"),
            ),
            ModelPricing(
                provider_name="anthropic",
                model="claude-sonnet-4-6",
                input_per_million=Decimal("3"),
                output_per_million=Decimal("15"),
                cache_read_input_per_million=Decimal("0.3"),
                cache_write_input_per_million=Decimal("3.75"),
            ),
        )
    )

    summary = estimate_session_cost(session_id="session_1", events=events, pricing=pricing)

    assert summary.model_steps == 2
    assert summary.priced_model_steps == 2
    assert summary.unpriced_model_steps == 0
    assert summary.line_items[0].pricing_model == "gpt-5.5"
    assert summary.line_items[0].pricing_match == "prefix"
    assert summary.line_items[0].input_cost == Decimal("0.0012")
    assert summary.line_items[0].cache_read_input_cost == Decimal("0.0002")
    assert summary.line_items[0].output_cost == Decimal("0.0016")
    assert summary.line_items[0].total_cost == Decimal("0.0030")
    assert summary.line_items[1].input_cost == Decimal("0.0015")
    assert summary.line_items[1].cache_read_input_cost == Decimal("0.00009")
    assert summary.line_items[1].cache_write_input_cost == Decimal("0.00075")
    assert summary.line_items[1].output_cost == Decimal("0.0015")
    assert summary.line_items[1].total_cost == Decimal("0.00384")
    assert summary.total_cost == Decimal("0.00684")


def test_estimate_session_cost_matches_a_provider_dated_model_prefix() -> None:
    events = [
        Event(
            type=EventType.MODEL_COMPLETED,
            session_id="session_1",
            payload={
                "usage_metrics": {
                    "provider_name": "openai",
                    "requested_model": "gpt-5.4-mini",
                    "model": "gpt-5.4-mini-2026-06-01",
                    "input_tokens": 1000,
                    "output_tokens": 100,
                    "total_tokens": 1100,
                    "reasoning_output_tokens": 0,
                    "cache": {
                        "read_tokens": 0,
                        "write_tokens": 0,
                        "cached_input_tokens": 0,
                        "uncached_input_tokens": 1000,
                    },
                }
            },
        )
    ]
    pricing = PricingCatalog(
        prices=(
            ModelPricing(
                provider_name="openai",
                model="gpt-5.4-mini",
                match="prefix",
                input_per_million=Decimal("0.25"),
                output_per_million=Decimal("2.00"),
            ),
        )
    )

    summary = estimate_session_cost(session_id="session_1", events=events, pricing=pricing)

    assert summary.priced_model_steps == 1
    assert summary.unpriced_model_steps == 0
    assert summary.line_items[0].model == "gpt-5.4-mini-2026-06-01"
    assert summary.line_items[0].requested_model == "gpt-5.4-mini"
    assert summary.line_items[0].pricing_model == "gpt-5.4-mini"
    assert summary.line_items[0].pricing_match == "prefix"
    assert summary.total_cost == Decimal("0.00045")


def test_estimate_causal_budget_cost_prices_related_sessions() -> None:
    events = [
        Event(
            type=EventType.MODEL_COMPLETED,
            session_id="sess_parent",
            payload={
                "usage_metrics": {
                    "provider_name": "openai",
                    "model": "gpt-5.5",
                    "input_tokens": 1000,
                    "output_tokens": 100,
                    "total_tokens": 1100,
                    "cache": {
                        "read_tokens": 200,
                        "write_tokens": 0,
                        "cached_input_tokens": 200,
                        "uncached_input_tokens": 800,
                    },
                }
            },
        ),
        Event(
            type=EventType.MODEL_COMPLETED,
            session_id="sess_child",
            payload={
                "usage_metrics": {
                    "provider_name": "openai",
                    "model": "gpt-5.5",
                    "input_tokens": 500,
                    "output_tokens": 50,
                    "total_tokens": 550,
                }
            },
        ),
    ]
    pricing = PricingCatalog(
        prices=(
            ModelPricing(
                provider_name="openai",
                model="gpt-5.5",
                input_per_million=Decimal("2"),
                output_per_million=Decimal("8"),
                cache_read_input_per_million=Decimal("0.5"),
            ),
        )
    )

    summary = estimate_causal_budget_cost(
        causal_budget_id="job_shared",
        session_ids=["sess_parent", "sess_child"],
        events=events,
        pricing=pricing,
    )

    assert summary.causal_budget_id == "job_shared"
    assert summary.session_ids == ["sess_parent", "sess_child"]
    assert summary.session_count == 2
    assert summary.model_steps == 2
    assert summary.priced_model_steps == 2
    assert summary.unpriced_model_steps == 0
    assert summary.line_items[0].total_cost == Decimal("0.0025")
    assert summary.line_items[1].total_cost == Decimal("0.0014")
    assert summary.total_cost == Decimal("0.0039")
    assert [item.session_id for item in summary.session_costs] == [
        "sess_parent",
        "sess_child",
    ]
    assert summary.session_costs[0].total_cost == Decimal("0.0025")
    assert summary.session_costs[1].total_cost == Decimal("0.0014")


def test_estimate_session_cost_reports_unpriced_model_steps() -> None:
    events = [
        Event(
            type=EventType.MODEL_COMPLETED,
            session_id="session_1",
            payload={
                "usage_metrics": {
                    "provider_name": "openai",
                    "model": "gpt-unknown",
                    "input_tokens": 100,
                    "output_tokens": 10,
                    "total_tokens": 110,
                    "reasoning_output_tokens": 0,
                    "cache": {
                        "read_tokens": 0,
                        "write_tokens": 0,
                        "cached_input_tokens": 0,
                        "uncached_input_tokens": 100,
                    },
                }
            },
        ),
        Event(type=EventType.MODEL_COMPLETED, session_id="session_1", payload={}),
    ]
    pricing = PricingCatalog(
        prices=(
            ModelPricing(
                provider_name="openai",
                model="gpt-5.5",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("2"),
            ),
        )
    )

    summary = estimate_session_cost(session_id="session_1", events=events, pricing=pricing)

    assert summary.total_cost == Decimal("0")
    assert summary.priced_model_steps == 0
    assert summary.unpriced_model_steps == 2
    assert summary.line_items[0].missing_pricing_reason == "no matching model pricing"
    assert (
        summary.line_items[1].missing_pricing_reason
        == "model.completed event has no token usage metrics"
    )


def test_estimate_session_cost_rejects_currency_mismatch() -> None:
    events = [
        Event(
            type=EventType.MODEL_COMPLETED,
            session_id="session_1",
            payload={
                "usage_metrics": {
                    "provider_name": "openai",
                    "model": "gpt-5.5",
                    "input_tokens": 100,
                    "output_tokens": 10,
                    "total_tokens": 110,
                    "reasoning_output_tokens": 0,
                    "cache": {
                        "read_tokens": 0,
                        "write_tokens": 0,
                        "cached_input_tokens": 0,
                        "uncached_input_tokens": 100,
                    },
                }
            },
        )
    ]
    pricing = PricingCatalog(
        prices=(
            ModelPricing(
                provider_name="openai",
                model="gpt-5.5",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("2"),
                currency="EUR",
            ),
        )
    )

    summary = estimate_session_cost(
        session_id="session_1",
        events=events,
        pricing=pricing,
        currency="USD",
    )

    assert summary.total_cost == Decimal("0")
    assert summary.unpriced_model_steps == 1
    assert summary.line_items[0].missing_pricing_reason == (
        "pricing currency EUR does not match requested USD"
    )


def test_estimate_session_cost_prefers_exact_pricing_over_prefix_pricing() -> None:
    events = [
        Event(
            type=EventType.MODEL_COMPLETED,
            session_id="session_1",
            payload={
                "usage_metrics": {
                    "provider_name": "openai",
                    "model": "gpt-5.5-special",
                    "input_tokens": 1000,
                    "output_tokens": 0,
                    "total_tokens": 1000,
                    "reasoning_output_tokens": 0,
                    "cache": {
                        "read_tokens": 0,
                        "write_tokens": 0,
                        "cached_input_tokens": 0,
                        "uncached_input_tokens": 1000,
                    },
                }
            },
        )
    ]
    pricing = PricingCatalog(
        prices=(
            ModelPricing(
                provider_name="openai",
                model="gpt-5.5",
                match="prefix",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("1"),
            ),
            ModelPricing(
                provider_name="openai",
                model="gpt-5.5-special",
                match="exact",
                input_per_million=Decimal("10"),
                output_per_million=Decimal("1"),
            ),
        )
    )

    summary = estimate_session_cost(
        session_id="session_1",
        events=events,
        pricing=pricing,
        currency="usd",
    )

    assert summary.currency == "USD"
    assert summary.line_items[0].pricing_match == "exact"
    assert summary.total_cost == Decimal("0.01")


def test_estimate_session_cost_respects_explicit_zero_cache_prices() -> None:
    events = [
        Event(
            type=EventType.MODEL_COMPLETED,
            session_id="session_1",
            payload={
                "usage_metrics": {
                    "provider_name": "openai",
                    "model": "gpt-5.5",
                    "input_tokens": 1000,
                    "output_tokens": 0,
                    "total_tokens": 1000,
                    "reasoning_output_tokens": 0,
                    "cache": {
                        "read_tokens": 800,
                        "write_tokens": 100,
                        "cached_input_tokens": 800,
                        "uncached_input_tokens": 100,
                    },
                }
            },
        )
    ]
    pricing = PricingCatalog(
        prices=(
            ModelPricing(
                provider_name="openai",
                model="gpt-5.5",
                input_per_million=Decimal("10"),
                output_per_million=Decimal("10"),
                cache_read_input_per_million=Decimal("0"),
                cache_write_input_per_million=Decimal("0"),
            ),
        )
    )

    summary = estimate_session_cost(session_id="session_1", events=events, pricing=pricing)

    assert summary.line_items[0].input_cost == Decimal("0.001")
    assert summary.line_items[0].cache_read_input_cost == Decimal("0")
    assert summary.line_items[0].cache_write_input_cost == Decimal("0")
    assert summary.total_cost == Decimal("0.001")


def test_request_budget_limits_validate_currency_and_copy_pricing() -> None:
    pricing = PricingCatalog(
        prices=(
            ModelPricing(
                provider_name="openai",
                model="gpt-5.5",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("2"),
            ),
        )
    )

    budget = BudgetLimit(
        scope="run",
        max_estimated_cost=Decimal("0.01"),
        pricing=pricing,
        currency="usd",
    )
    (copied,) = copy_request_budget_limits((budget,))

    assert copied is not budget
    assert copied.currency == "USD"
    assert copied.scope == "run"
    assert copied.pricing is not budget.pricing
    assert copied.pricing.prices[0] is not budget.pricing.prices[0]


def test_request_budget_limits_reject_reservations_on_session_and_run_scopes() -> None:
    pricing = PricingCatalog(
        prices=(
            ModelPricing(
                provider_name="openai",
                model="gpt-5.5",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("2"),
            ),
        )
    )

    for scope in ("session", "run"):
        budget = BudgetLimit(
            scope=scope,
            max_estimated_cost=Decimal("0.01"),
            pricing=pricing,
            reservation=BudgetReservation(max_input_tokens=1, max_output_tokens=0),
        )
        with pytest.raises(ValueError, match="must not use reservations"):
            copy_request_budget_limits((budget,))


def test_request_budget_limits_allow_reservations_on_shared_scopes() -> None:
    pricing = PricingCatalog(
        prices=(
            ModelPricing(
                provider_name="openai",
                model="gpt-5.5",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("2"),
            ),
        )
    )

    limits = copy_request_budget_limits(
        (
            BudgetLimit(
                scope="app",
                max_estimated_cost=Decimal("0.01"),
                pricing=pricing,
                reservation=BudgetReservation(max_input_tokens=1, max_output_tokens=0),
            ),
            BudgetLimit(
                scope="agent",
                key="assistant",
                max_estimated_cost=Decimal("0.02"),
                pricing=pricing,
                reservation=BudgetReservation(max_input_tokens=2, max_output_tokens=1),
            ),
            BudgetLimit(
                scope="causal",
                key="job_1",
                max_estimated_cost=Decimal("0.03"),
                pricing=pricing,
                reservation=BudgetReservation(max_input_tokens=3, max_output_tokens=0),
            ),
        )
    )

    assert [limit.scope for limit in limits] == ["app", "agent", "causal"]
    assert all(limit.reservation is not None for limit in limits)
    assert limits[1].reservation is not None
    assert limits[1].reservation.max_input_tokens == 2


def test_request_budget_limits_validate_agent_and_causal_keys() -> None:
    pricing = PricingCatalog(
        prices=(
            ModelPricing(
                provider_name="openai",
                model="gpt-5.5",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("2"),
            ),
        )
    )

    limits = request_budget_limits_for_session(
        limits=(
            BudgetLimit(
                scope="agent",
                key="assistant",
                max_estimated_cost=Decimal("0.01"),
                pricing=pricing,
            ),
            BudgetLimit(
                scope="causal",
                key="job_1",
                max_estimated_cost=Decimal("0.02"),
                pricing=pricing,
            ),
        ),
        agent_name="assistant",
        causal_budget_id="job_1",
    )

    assert [limit.scope for limit in limits] == ["agent", "causal"]
    with pytest.raises(ValueError, match="does not match"):
        request_budget_limits_for_session(
            limits=(
                BudgetLimit(
                    scope="agent",
                    key="other",
                    max_estimated_cost=Decimal("0.01"),
                    pricing=pricing,
                ),
            ),
            agent_name="assistant",
            causal_budget_id="job_1",
        )


def test_usage_metrics_from_event_payload_rejects_non_usage_payload() -> None:
    assert usage_metrics_from_event_payload({"usage": "bad"}) is None
    assert usage_metrics_from_event_payload({"usage": {}}) is None


def test_run_limits_detect_reached_token_budget() -> None:
    summary = session_usage_summary(
        "session_1",
        [
            Event(
                type=EventType.MODEL_COMPLETED,
                session_id="session_1",
                payload={
                    "usage_metrics": {
                        "input_tokens": 7,
                        "output_tokens": 3,
                        "total_tokens": 10,
                        "reasoning_output_tokens": 0,
                        "cache": {
                            "read_tokens": 0,
                            "write_tokens": 0,
                            "cached_input_tokens": 0,
                            "uncached_input_tokens": 7,
                        },
                    }
                },
            )
        ],
    )

    decision = first_reached_limit(
        limits=RunLimits(max_total_tokens=10),
        usage=summary,
        elapsed_seconds=0,
    )

    assert decision is not None
    assert decision.limit == StopLimit.TOTAL_TOKENS
    assert decision.maximum == 10
    assert decision.actual == 10
    assert decision.message == "Run limit reached: total_tokens 10 >= 10."


def test_run_limits_allow_tool_call_until_capacity_is_exceeded() -> None:
    summary = session_usage_summary(
        "session_1",
        [Event(type=EventType.TOOL_CALL_STARTED, session_id="session_1")],
    )

    allowed = first_reached_limit(
        limits=RunLimits(max_tool_calls=2),
        usage=summary,
        elapsed_seconds=0,
        pending_tool_calls=1,
    )
    blocked = first_reached_limit(
        limits=RunLimits(max_tool_calls=2),
        usage=summary,
        elapsed_seconds=0,
        pending_tool_calls=2,
    )

    assert allowed is None
    assert blocked is not None
    assert blocked.limit == StopLimit.TOOL_CALLS
    assert blocked.actual == 3
    # Tool calls only stop past capacity, so the message uses a strict ">".
    assert blocked.message == "Run limit reached: tool_calls 3 > 2."


def test_has_run_limits_detects_empty_and_configured_limits() -> None:
    assert not has_run_limits(RunLimits())
    assert has_run_limits(RunLimits(max_elapsed_seconds=1))


def test_run_limits_scope_defaults_to_run() -> None:
    assert RunLimits().scope == "run"


def test_copy_run_limits_preserves_scope() -> None:
    copied = copy_run_limits(RunLimits(scope="run", max_tool_calls=3))
    assert copied.scope == "run"
    assert copied.max_tool_calls == 3


def test_run_limits_scope_alone_is_not_a_limit() -> None:
    assert not has_run_limits(RunLimits(scope="run"))


def test_stop_decision_supports_estimated_cost_with_decimal_values() -> None:
    decision = StopDecision(
        limit=StopLimit.ESTIMATED_COST,
        maximum=Decimal("0.50"),
        actual=Decimal("0.51"),
        message="Budget reached: 0.51 >= 0.50 USD.",
    )

    assert decision.limit == StopLimit.ESTIMATED_COST
    assert decision.maximum == Decimal("0.50")
    assert decision.actual == Decimal("0.51")


def test_cayu_app_get_session_cost_uses_durable_events() -> None:
    app = CayuApp()
    asyncio.run(
        app.session_store.create(
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "hi")],
                session_id="cost_session",
            ),
            identity=SessionIdentity(
                provider_name="openai",
                model="gpt-5.6",
                runtime_name="cayu",
                runtime_version=None,
            ),
        )
    )
    asyncio.run(
        app.session_store.append_event(
            "cost_session",
            Event(
                type=EventType.MODEL_COMPLETED,
                session_id="cost_session",
                payload={
                    "usage_metrics": {
                        "provider_name": "openai",
                        "model": "gpt-5.5",
                        "input_tokens": 1000,
                        "output_tokens": 100,
                        "total_tokens": 1100,
                        "reasoning_output_tokens": 0,
                        "cache": {
                            "read_tokens": 0,
                            "write_tokens": 0,
                            "cached_input_tokens": 0,
                            "uncached_input_tokens": 1000,
                        },
                    }
                },
            ),
        )
    )
    pricing = PricingCatalog(
        prices=(
            ModelPricing(
                provider_name="openai",
                model="gpt-5.5",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("10"),
            ),
        )
    )

    summary = asyncio.run(app.get_session_cost("cost_session", pricing))

    assert summary.total_cost == Decimal("0.002")


def test_cayu_app_cost_methods_preserve_catalog_projection_semantics() -> None:
    async def run():
        app = CayuApp()
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "hi")],
                session_id="tiered_cost_session",
                causal_budget_id="tiered_cost_job",
            ),
            identity=SessionIdentity(
                provider_name="openai",
                model="gpt-5.5",
                runtime_name="cayu",
                runtime_version=None,
            ),
        )
        await app.session_store.append_event(
            "tiered_cost_session",
            Event(
                type=EventType.MODEL_COMPLETED,
                session_id="tiered_cost_session",
                payload={
                    "usage_metrics": {
                        "provider_name": "openai",
                        "model": "gpt-5.6",
                        "input_tokens": 300_000,
                        "output_tokens": 0,
                        "total_tokens": 300_000,
                        "reasoning_output_tokens": 0,
                        "cache": {
                            "read_tokens": 0,
                            "write_tokens": 0,
                            "cached_input_tokens": 0,
                            "uncached_input_tokens": 300_000,
                        },
                    }
                },
            ),
        )
        catalog = default_model_catalog()
        projected = catalog.pricing_catalog()
        return (
            await app.get_session_cost("tiered_cost_session", catalog),
            await app.get_session_cost("tiered_cost_session", projected),
            await app.get_causal_budget_cost("tiered_cost_job", catalog),
            await app.get_causal_budget_cost("tiered_cost_job", projected),
        )

    direct_session, projected_session, direct_causal, projected_causal = asyncio.run(run())

    assert projected_session == direct_session
    assert projected_causal == direct_causal
    model = default_model_catalog().resolve(provider_name="openai", model="gpt-5.6")
    assert model is not None
    expected = model.pricing_at(300_000).input_per_million * Decimal("0.3")
    assert direct_session.total_cost == expected
    assert direct_session.line_items[0].pricing_provenance is not None
