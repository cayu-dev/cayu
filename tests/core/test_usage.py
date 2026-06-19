from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from cayu.core import Event, EventType, Message
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
                    max_estimated_cost=Decimal("2"),
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


def test_in_memory_budget_ledger_keeps_active_reservations_inside_rolling_window() -> None:
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
        clock.value = datetime(2026, 1, 1, 12, 2, tzinfo=UTC)
        rolling_blocked = await ledger.reserve(
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
        return rolling_first, rolling_blocked, all_time_first, all_time_second

    rolling_first, rolling_blocked, all_time_first, all_time_second = asyncio.run(run())

    assert rolling_first.accepted is True
    assert rolling_blocked.accepted is False
    assert rolling_blocked.actual == Decimal("0.44")
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


def test_sqlite_budget_ledger_keeps_active_reservations_inside_rolling_window(tmp_path) -> None:
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
            clock.value = datetime(2026, 1, 1, 12, 2, tzinfo=UTC)
            rolling_blocked = await ledger.reserve(
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
            return rolling_first, rolling_blocked, all_time_first, all_time_second
        finally:
            await ledger.close()

    rolling_first, rolling_blocked, all_time_first, all_time_second = asyncio.run(run())

    assert rolling_first.accepted is True
    assert rolling_blocked.accepted is False
    assert rolling_blocked.actual == Decimal("0.44")
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


def test_request_budget_limits_reject_reservations() -> None:
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
        scope="session",
        max_estimated_cost=Decimal("0.01"),
        pricing=pricing,
        reservation=BudgetReservation(max_input_tokens=1, max_output_tokens=0),
    )

    with pytest.raises(ValueError, match="must not use reservations"):
        copy_request_budget_limits((budget,))


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


def test_has_run_limits_detects_empty_and_configured_limits() -> None:
    assert not has_run_limits(RunLimits())
    assert has_run_limits(RunLimits(max_elapsed_seconds=1))


def test_run_limits_scope_defaults_to_session() -> None:
    assert RunLimits().scope == "session"


def test_copy_run_limits_preserves_scope() -> None:
    copied = copy_run_limits(RunLimits(scope="run", max_tool_calls=3))
    assert copied.scope == "run"
    assert copied.max_tool_calls == 3


def test_run_limits_scope_alone_is_not_a_limit() -> None:
    assert not has_run_limits(RunLimits(scope="run"))


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
                model="gpt-5.5",
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
