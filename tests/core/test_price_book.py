from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from cayu import (
    BillingIdentity,
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
    PricingResourceMapping,
    Provenance,
    TieredPricing,
    default_model_catalog,
    default_price_book,
    estimate_session_cost,
)
from cayu.core.billing import resolved_billing_identity
from cayu.providers import bedrock_billing_identity
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


def _bedrock_completed(identity: BillingIdentity) -> Event:
    return Event(
        type=EventType.MODEL_COMPLETED,
        session_id="bedrock-session",
        timestamp=datetime(2026, 7, 16, tzinfo=UTC),
        payload={
            "billing_identity": identity.model_dump(mode="json"),
            "usage_metrics": {
                "provider_name": "bedrock",
                "requested_model": identity.resource_id,
                "model": identity.resource_id,
                "billing_identity": identity.model_dump(mode="json"),
                "input_tokens": 1_000_000,
                "output_tokens": 1_000_000,
                "total_tokens": 2_000_000,
                "reasoning_output_tokens": 0,
                "cache": {
                    "read_tokens": 0,
                    "write_tokens": 0,
                    "cached_input_tokens": 0,
                    "uncached_input_tokens": 1_000_000,
                },
            },
        },
    )


def _bedrock_prices() -> PriceBook:
    return PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="bedrock",
                model="global.anthropic.claude-sonnet-4-6",
                match="exact",
                input_per_million=Decimal("3"),
                output_per_million=Decimal("15"),
                pricing_context={
                    "source_region": ("us-east-1",),
                    "service_tier": ("default",),
                },
            ),
        )
    )


@pytest.mark.parametrize(
    "identity",
    [
        bedrock_billing_identity(
            invoked_model="us.anthropic.claude-sonnet-4-6",
            source_region="us-east-1",
            resource_type="inference_profile",
            profile_scope="geographic",
        ),
        bedrock_billing_identity(
            invoked_model="global.anthropic.claude-sonnet-4-6",
            source_region="eu-west-1",
            resource_type="inference_profile",
            profile_scope="global",
        ),
        bedrock_billing_identity(
            invoked_model="global.anthropic.claude-sonnet-4-6",
            source_region="us-east-1",
            resource_type="inference_profile",
            profile_scope="global",
            requested_service_tier="flex",
        ),
    ],
)
def test_bedrock_pricing_does_not_cross_match_profile_region_or_tier(
    identity: BillingIdentity,
) -> None:
    summary = estimate_session_cost(
        session_id="bedrock-session",
        events=[_bedrock_completed(identity)],
        pricing=_bedrock_prices(),
    )
    assert summary.unpriced_model_steps == 1
    assert summary.line_items[0].billing_identity == identity


def test_bedrock_application_profile_requires_explicit_mapping() -> None:
    arn = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/profile-1"
    identity = bedrock_billing_identity(
        invoked_model=arn,
        source_region="us-east-1",
        resource_type="application_inference_profile",
    )
    unmapped = estimate_session_cost(
        session_id="bedrock-session",
        events=[_bedrock_completed(identity)],
        pricing=_bedrock_prices(),
    )
    mapped_prices = _bedrock_prices().model_copy(
        update={
            "resource_mappings": (
                PricingResourceMapping(
                    provider_name="bedrock",
                    resource_id=arn,
                    pricing_model="global.anthropic.claude-sonnet-4-6",
                ),
            )
        }
    )
    mapped = estimate_session_cost(
        session_id="bedrock-session",
        events=[_bedrock_completed(identity)],
        pricing=mapped_prices,
    )
    assert unmapped.unpriced_model_steps == 1
    assert mapped.priced_model_steps == 1
    assert mapped.total_cost == Decimal("18")
    assert mapped.line_items[0].pricing_match == "resource_mapping"


def test_bedrock_billing_identity_uses_canonical_provider_for_renamed_adapter() -> None:
    identity = bedrock_billing_identity(
        invoked_model="global.anthropic.claude-sonnet-4-6",
        source_region="us-east-1",
        resource_type="inference_profile",
        profile_scope="global",
        effective_service_tier="default",
    )
    event = _bedrock_completed(identity)
    metrics = event.payload["usage_metrics"]
    assert isinstance(metrics, dict)
    metrics["provider_name"] = "bedrock-us-east"

    summary = estimate_session_cost(
        session_id="bedrock-session",
        events=[event],
        pricing=_bedrock_prices(),
    )

    assert summary.priced_model_steps == 1
    assert summary.total_cost == Decimal("18")
    assert summary.line_items[0].provider_name == "bedrock-us-east"
    assert summary.line_items[0].pricing_provider_name == "bedrock"


def test_price_book_rejects_noncontextual_bedrock_price() -> None:
    with pytest.raises(ValidationError, match="requires exact contextual dimensions"):
        PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="bedrock",
                    model="anthropic.claude-sonnet-4-6",
                    input_per_million=Decimal("3"),
                    output_per_million=Decimal("15"),
                ),
            )
        )


def test_bedrock_posthoc_cost_stays_unpriced_without_billing_identity() -> None:
    identity = bedrock_billing_identity(
        invoked_model="global.anthropic.claude-sonnet-4-6",
        source_region="us-east-1",
        resource_type="inference_profile",
        profile_scope="global",
        effective_service_tier="default",
    )
    event = _bedrock_completed(identity)
    event.payload.pop("billing_identity")
    metrics = event.payload["usage_metrics"]
    assert isinstance(metrics, dict)
    metrics.pop("billing_identity")

    summary = estimate_session_cost(
        session_id="identity-less-bedrock-session",
        events=[event],
        pricing=_bedrock_prices(),
    )

    assert summary.priced_model_steps == 0
    assert summary.unpriced_model_steps == 1
    assert summary.total_cost == Decimal("0")


def test_price_book_rejects_overlapping_bedrock_contexts() -> None:
    with pytest.raises(ValidationError, match="overlapping pricing contexts"):
        PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="bedrock",
                    model="global.anthropic.claude-sonnet-4-6",
                    match="exact",
                    input_per_million=Decimal("3"),
                    output_per_million=Decimal("15"),
                    pricing_context={
                        "source_region": ("us-east-1", "us-west-2"),
                        "service_tier": ("default",),
                    },
                ),
                ModelPrice.fixed(
                    provider_name="bedrock",
                    model="global.anthropic.claude-sonnet-4-6",
                    match="exact",
                    input_per_million=Decimal("4"),
                    output_per_million=Decimal("20"),
                    pricing_context={
                        "source_region": ("us-east-1", "eu-west-1"),
                        "service_tier": ("default",),
                    },
                ),
            )
        )


def test_bedrock_budget_preflight_defers_until_request_identity_is_known() -> None:
    limit = BudgetLimit(max_estimated_cost=Decimal("100"), pricing=_bedrock_prices())

    before_request = budget_check_from_events(
        limit=limit,
        events=[],
        provider_name="bedrock",
        model="global.anthropic.claude-sonnet-4-6",
        effective_at=datetime(2026, 7, 16, tzinfo=UTC),
    )
    matching_request = budget_check_from_events(
        limit=limit,
        events=[],
        provider_name="bedrock",
        model="global.anthropic.claude-sonnet-4-6",
        billing_identity_state=resolved_billing_identity(
            bedrock_billing_identity(
                invoked_model="global.anthropic.claude-sonnet-4-6",
                source_region="us-east-1",
                resource_type="inference_profile",
                profile_scope="global",
            )
        ),
        effective_at=datetime(2026, 7, 16, tzinfo=UTC),
    )
    wrong_region = budget_check_from_events(
        limit=limit,
        events=[],
        provider_name="bedrock",
        model="global.anthropic.claude-sonnet-4-6",
        billing_identity_state=resolved_billing_identity(
            bedrock_billing_identity(
                invoked_model="global.anthropic.claude-sonnet-4-6",
                source_region="eu-west-1",
                resource_type="inference_profile",
                profile_scope="global",
            )
        ),
        effective_at=datetime(2026, 7, 16, tzinfo=UTC),
    )

    assert before_request.limit_reached is False
    assert matching_request.limit_reached is False
    assert wrong_region.limit_reached is True
    assert "no matching model pricing" in wrong_region.message


def test_bedrock_reserved_budget_preflight_checks_both_possible_effective_tiers() -> None:
    identity = bedrock_billing_identity(
        invoked_model="global.anthropic.claude-sonnet-4-6",
        source_region="us-east-1",
        resource_type="inference_profile",
        profile_scope="global",
        requested_service_tier="reserved",
    )

    def price(service_tier: str, input_price: str) -> ModelPrice:
        return ModelPrice.fixed(
            provider_name="bedrock",
            model=identity.resource_id,
            match="exact",
            input_per_million=Decimal(input_price),
            output_per_million=Decimal("1"),
            pricing_context={
                "source_region": ("us-east-1",),
                "service_tier": (service_tier,),
            },
        )

    complete_pricing = PriceBook(
        prices=(
            price("reserved", "2"),
            price("default", "3"),
        )
    )
    limit = BudgetLimit(
        max_estimated_cost=Decimal("10"),
        pricing=complete_pricing,
        reservation=BudgetReservation(max_input_tokens=1_000_000, max_output_tokens=0),
    )

    check = budget_check_from_events(
        limit=limit,
        events=[],
        provider_name="bedrock",
        model=identity.resource_id,
        billing_identity_state=resolved_billing_identity(identity),
        effective_at=datetime(2026, 7, 16, tzinfo=UTC),
    )
    reserved = asyncio.run(
        InMemoryBudgetLedger().reserve(
            limit=limit,
            session_id="bedrock-session",
            agent_name="assistant",
            provider_name="bedrock",
            model=identity.resource_id,
            billing_identity=identity,
        )
    )
    incomplete = budget_check_from_events(
        limit=limit.model_copy(update={"pricing": PriceBook(prices=(price("reserved", "2"),))}),
        events=[],
        provider_name="bedrock",
        model=identity.resource_id,
        billing_identity_state=resolved_billing_identity(identity),
        effective_at=datetime(2026, 7, 16, tzinfo=UTC),
    )

    assert check.limit_reached is False
    assert reserved.accepted is True
    assert reserved.requested == Decimal("3")
    assert incomplete.limit_reached is True
    assert "no matching model pricing" in incomplete.message


def test_bedrock_cache_write_reservation_requires_declared_ttl_rate() -> None:
    identity = bedrock_billing_identity(
        invoked_model="global.anthropic.claude-sonnet-4-6",
        source_region="us-east-1",
        resource_type="inference_profile",
        profile_scope="global",
    )
    ledger = InMemoryBudgetLedger()

    with pytest.raises(ValueError, match="cache-write TTLs"):
        asyncio.run(
            ledger.reserve(
                limit=BudgetLimit(
                    max_estimated_cost=Decimal("10"),
                    pricing=_bedrock_prices(),
                    reservation=BudgetReservation(
                        max_input_tokens=0,
                        max_output_tokens=0,
                        max_cache_write_input_tokens=1_000_000,
                    ),
                ),
                session_id="bedrock-session",
                agent_name="assistant",
                provider_name="bedrock",
                model=identity.resource_id,
                billing_identity=identity,
            )
        )


def test_budget_reservation_preserves_a_zero_cache_read_rate() -> None:
    pricing = PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="gateway",
                model="frontier",
                match="exact",
                input_per_million=Decimal("3"),
                output_per_million=Decimal("15"),
                cache_read_input_per_million=Decimal("0"),
            ),
        )
    )

    result = asyncio.run(
        InMemoryBudgetLedger().reserve(
            limit=BudgetLimit(
                max_estimated_cost=Decimal("1"),
                pricing=pricing,
                reservation=BudgetReservation(
                    max_input_tokens=0,
                    max_output_tokens=0,
                    max_cache_read_input_tokens=1_000_000,
                ),
            ),
            session_id="session-1",
            agent_name="assistant",
            provider_name="gateway",
            model="frontier",
        )
    )

    assert result.accepted is True
    assert result.requested == Decimal("0")


def test_non_bedrock_cache_write_reservation_matches_cost_engine_rate() -> None:
    pricing = PriceBook(
        prices=(
            ModelPrice(
                provider_name="anthropic",
                model="claude-sonnet-4-6",
                schedules=(
                    PriceSchedule(
                        pricing=TieredPricing(
                            standard=(
                                PriceTier(
                                    input_per_million=Decimal("3"),
                                    output_per_million=Decimal("15"),
                                    cache_write_input_per_million=Decimal("3.75"),
                                ),
                            ),
                            cache_write_1h_per_million=Decimal("6"),
                        ),
                        provenance=_PROVENANCE,
                    ),
                ),
            ),
        )
    )

    result = asyncio.run(
        InMemoryBudgetLedger().reserve(
            limit=BudgetLimit(
                max_estimated_cost=Decimal("10"),
                pricing=pricing,
                reservation=BudgetReservation(
                    max_input_tokens=0,
                    max_output_tokens=0,
                    max_cache_write_input_tokens=1_000_000,
                ),
            ),
            session_id="session-1",
            agent_name="assistant",
            provider_name="anthropic",
            model="claude-sonnet-4-6",
        )
    )

    assert result.accepted is True
    assert result.requested == Decimal("3.75")


def test_bundled_bedrock_price_applies_exact_cache_ttl_and_identity() -> None:
    identity = bedrock_billing_identity(
        invoked_model="global.anthropic.claude-sonnet-4-6",
        source_region="us-east-1",
        resource_type="inference_profile",
        profile_scope="global",
        effective_service_tier="default",
    )
    event = _bedrock_completed(identity)
    metrics = event.payload["usage_metrics"]
    assert isinstance(metrics, dict)
    metrics["input_tokens"] = 2_000_000
    metrics["total_tokens"] = 3_000_000
    metrics["cache"] = {
        "read_tokens": 0,
        "write_tokens": 1_000_000,
        "write_5m_tokens": 1_000_000,
        "write_1h_tokens": 0,
        "write_unknown_ttl_tokens": 0,
        "cached_input_tokens": 0,
        "uncached_input_tokens": 1_000_000,
    }

    summary = estimate_session_cost(
        session_id="bedrock-session",
        events=[event],
        pricing=default_price_book(),
    )

    assert summary.priced_model_steps == 1
    assert summary.total_cost == Decimal("21.75")
    assert summary.line_items[0].billing_identity == identity


def test_bundled_bedrock_price_rejects_unsupported_one_hour_cache_write() -> None:
    identity = bedrock_billing_identity(
        invoked_model="global.anthropic.claude-sonnet-4-6",
        source_region="us-east-1",
        resource_type="inference_profile",
        profile_scope="global",
    )
    event = _bedrock_completed(identity)
    metrics = event.payload["usage_metrics"]
    assert isinstance(metrics, dict)
    metrics["cache"] = {
        "read_tokens": 0,
        "write_tokens": 1,
        "write_5m_tokens": 0,
        "write_1h_tokens": 1,
        "write_unknown_ttl_tokens": 0,
        "cached_input_tokens": 0,
        "uncached_input_tokens": 1_000_000,
    }

    summary = estimate_session_cost(
        session_id="bedrock-session",
        events=[event],
        pricing=default_price_book(),
    )

    assert summary.unpriced_model_steps == 1
    assert summary.line_items[0].missing_pricing_reason == (
        "pricing does not declare 1-hour cache writes"
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
