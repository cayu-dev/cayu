from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from cayu import (
    BudgetLimit,
    BudgetReservation,
    CayuApp,
    Event,
    EventType,
    InMemoryBudgetLedger,
    ModelInfo,
    ModelPrice,
    PriceBook,
    PriceSchedule,
    PriceTier,
    Provenance,
    TieredPricing,
    default_model_catalog,
    default_price_book,
    estimate_session_cost,
)
from cayu.runtime.budgets import budget_check_from_events

_PROVENANCE = Provenance(
    source="official",
    url="https://example.com/pricing",
    as_of="2026-07-14",
)


def _pricing(input_price: str, output_price: str) -> TieredPricing:
    return TieredPricing(
        standard=(
            PriceTier(
                max_input_tokens=None,
                input_per_million=Decimal(input_price),
                output_per_million=Decimal(output_price),
            ),
        )
    )


def _completed(*, model: str, timestamp: datetime) -> Event:
    return Event(
        type=EventType.MODEL_COMPLETED,
        session_id="session-1",
        timestamp=timestamp,
        payload={
            "usage_metrics": {
                "provider_name": "gateway",
                "model": model,
                "input_tokens": 1_000_000,
                "output_tokens": 0,
                "total_tokens": 1_000_000,
                "reasoning_output_tokens": 0,
                "cache": {
                    "read_tokens": 0,
                    "write_tokens": 0,
                    "cached_input_tokens": 0,
                    "uncached_input_tokens": 1_000_000,
                },
            }
        },
    )


def test_default_model_catalog_and_price_book_are_independent_offline_resources() -> None:
    models = default_model_catalog()
    prices = default_price_book()

    assert "pricing" not in ModelInfo.model_fields
    assert models.models
    assert prices.prices
    assert default_model_catalog() is not models
    assert default_price_book() is not prices


def test_price_book_uses_event_date_to_select_effective_schedule() -> None:
    prices = PriceBook(
        price_book_version="test",
        generated_at="2026-07-14",
        prices=(
            ModelPrice(
                provider_name="gateway",
                model="frontier",
                match="exact",
                schedules=(
                    PriceSchedule(
                        effective_from=None,
                        effective_through=date(2026, 8, 31),
                        pricing=_pricing("2", "10"),
                        provenance=_PROVENANCE,
                    ),
                    PriceSchedule(
                        effective_from=date(2026, 9, 1),
                        effective_through=None,
                        pricing=_pricing("3", "15"),
                        provenance=_PROVENANCE,
                    ),
                ),
            ),
        ),
    )

    summary = estimate_session_cost(
        session_id="session-1",
        events=[
            _completed(
                model="frontier",
                # UTC date is still August 31 despite the local September 1 date.
                timestamp=datetime(
                    2026,
                    9,
                    1,
                    0,
                    30,
                    tzinfo=timezone(timedelta(hours=1)),
                ),
            ),
            _completed(
                model="frontier",
                # UTC date is September 1 despite the local August 31 date.
                timestamp=datetime(
                    2026,
                    8,
                    31,
                    17,
                    tzinfo=timezone(timedelta(hours=-7)),
                ),
            ),
        ],
        pricing=prices,
    )

    assert [item.total_cost for item in summary.line_items] == [Decimal("2"), Decimal("3")]
    assert summary.line_items[0].pricing_effective_through == date(2026, 8, 31)
    assert summary.line_items[1].pricing_effective_from == date(2026, 9, 1)


def test_price_book_rejects_overlapping_effective_schedules() -> None:
    with pytest.raises(ValidationError, match="overlap"):
        ModelPrice(
            provider_name="gateway",
            model="frontier",
            schedules=(
                PriceSchedule(
                    effective_from=None,
                    effective_through=date(2026, 9, 1),
                    pricing=_pricing("2", "10"),
                    provenance=_PROVENANCE,
                ),
                PriceSchedule(
                    effective_from=date(2026, 9, 1),
                    effective_through=None,
                    pricing=_pricing("3", "15"),
                    provenance=_PROVENANCE,
                ),
            ),
        )


def test_price_book_rejects_reversed_and_unordered_effective_schedules() -> None:
    with pytest.raises(ValidationError, match="must not follow"):
        PriceSchedule(
            effective_from=date(2026, 9, 1),
            effective_through=date(2026, 8, 31),
            pricing=_pricing("2", "10"),
            provenance=_PROVENANCE,
        )

    with pytest.raises(ValidationError, match="must be ordered"):
        ModelPrice(
            provider_name="gateway",
            model="frontier",
            schedules=(
                PriceSchedule(
                    effective_from=date(2026, 9, 1),
                    pricing=_pricing("3", "15"),
                    provenance=_PROVENANCE,
                ),
                PriceSchedule(
                    effective_through=date(2026, 8, 31),
                    pricing=_pricing("2", "10"),
                    provenance=_PROVENANCE,
                ),
            ),
        )


def test_price_book_prices_gateway_identity_without_model_metadata() -> None:
    assert default_model_catalog().resolve(provider_name="gateway", model="frontier") is None

    prices = PriceBook(
        price_book_version="test",
        generated_at="2026-07-14",
        prices=(
            ModelPrice(
                provider_name="gateway",
                model="frontier",
                schedules=(PriceSchedule(pricing=_pricing("1", "2"), provenance=_PROVENANCE),),
            ),
        ),
    )

    summary = estimate_session_cost(
        session_id="session-1",
        events=[_completed(model="frontier", timestamp=datetime(2026, 7, 14, tzinfo=UTC))],
        pricing=prices,
    )

    assert summary.priced_model_steps == 1
    assert summary.total_cost == Decimal("1")


def test_budget_preflight_fails_closed_when_price_schedule_has_expired() -> None:
    prices = PriceBook(
        prices=(
            ModelPrice(
                provider_name="gateway",
                model="frontier",
                schedules=(
                    PriceSchedule(
                        effective_through=date(2026, 8, 31),
                        pricing=_pricing("2", "10"),
                        provenance=_PROVENANCE,
                    ),
                ),
            ),
        ),
    )
    check = budget_check_from_events(
        limit=BudgetLimit(max_estimated_cost=Decimal("10"), pricing=prices),
        events=[],
        provider_name="gateway",
        model="frontier",
        effective_at=datetime(2026, 9, 1, tzinfo=UTC),
    )

    assert check.limit_reached is True
    assert "pricing schedule expired" in check.message


def test_budget_reservation_uses_the_ledgers_injected_clock() -> None:
    prices = PriceBook(
        prices=(
            ModelPrice(
                provider_name="gateway",
                model="frontier",
                schedules=(
                    PriceSchedule(
                        effective_through=date(2026, 8, 31),
                        pricing=_pricing("2", "10"),
                        provenance=_PROVENANCE,
                    ),
                    PriceSchedule(
                        effective_from=date(2026, 9, 1),
                        pricing=_pricing("3", "15"),
                        provenance=_PROVENANCE,
                    ),
                ),
            ),
        ),
    )
    ledger = InMemoryBudgetLedger(
        clock=lambda: datetime(2026, 9, 1, tzinfo=UTC),
    )
    result = asyncio.run(
        ledger.reserve(
            limit=BudgetLimit(
                max_estimated_cost=Decimal("10"),
                pricing=prices,
                reservation=BudgetReservation(
                    max_input_tokens=1_000_000,
                    max_output_tokens=0,
                ),
            ),
            session_id="session-1",
            agent_name="agent",
            provider_name="gateway",
            model="frontier",
        )
    )

    assert result.accepted is True
    assert result.requested == Decimal("3")


def test_default_app_budget_ledger_inherits_the_runtime_clock() -> None:
    prices = PriceBook(
        prices=(
            ModelPrice(
                provider_name="gateway",
                model="frontier",
                schedules=(
                    PriceSchedule(
                        effective_through=date(2026, 8, 31),
                        pricing=_pricing("2", "10"),
                        provenance=_PROVENANCE,
                    ),
                    PriceSchedule(
                        effective_from=date(2026, 9, 1),
                        pricing=_pricing("3", "15"),
                        provenance=_PROVENANCE,
                    ),
                ),
            ),
        ),
    )
    app = CayuApp(
        enable_logging=False,
        clock=lambda: datetime(2026, 9, 1, tzinfo=UTC),
    )

    result = asyncio.run(
        app.budget_ledger.reserve(
            limit=BudgetLimit(
                max_estimated_cost=Decimal("10"),
                pricing=prices,
                reservation=BudgetReservation(
                    max_input_tokens=1_000_000,
                    max_output_tokens=0,
                ),
            ),
            session_id="session-1",
            agent_name="agent",
            provider_name="gateway",
            model="frontier",
        )
    )

    assert result.accepted is True
    assert result.requested == Decimal("3")
