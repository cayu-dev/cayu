from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from cayu import (
    BillingIdentity,
    BudgetLimit,
    BudgetReservation,
    ContextualPricingRequirement,
    Event,
    EventType,
    InMemoryBudgetLedger,
    ModelPrice,
    PriceBook,
    PricingContext,
    PricingContextSelector,
    ResolvedBillingIdentity,
    estimate_session_cost,
)
from cayu.core.billing import completed_billing_identity


def _identity(*, contexts: tuple[PricingContext, ...]) -> BillingIdentity:
    return BillingIdentity(
        provider_name="commercial-cloud",
        resource_id="opaque-resource-1",
        request_evidence={"tenant": "tenant-a", "nested": {"plan": "reserved"}},
        pricing_contexts=contexts,
    )


def test_billing_identity_is_deeply_immutable() -> None:
    identity = _identity(
        contexts=(PricingContext(dimensions={"zone": "north", "tier": "reserved"}),)
    )

    with pytest.raises(TypeError, match="cannot be mutated"):
        identity.request_evidence["tenant"] = "tenant-b"  # type: ignore[index]
    nested = identity.request_evidence["nested"]
    assert isinstance(nested, Mapping)
    with pytest.raises(TypeError, match="cannot be mutated"):
        nested["plan"] = "standard"  # type: ignore[index]
    with pytest.raises(TypeError, match="cannot be mutated"):
        identity.pricing_contexts[0].dimensions["zone"] = "south"  # type: ignore[index]


def test_pricing_context_selector_is_deeply_immutable() -> None:
    selector = PricingContextSelector(dimensions={"zone": ("north", "south")})

    assert selector.dimensions["zone"] == ("north", "south")
    with pytest.raises(TypeError, match="cannot be mutated"):
        selector.dimensions["zone"] = ("south",)  # type: ignore[index]


def test_resolved_state_revalidates_provider_supplied_identity() -> None:
    identity = _identity(contexts=()).model_copy(
        update={"request_evidence": {"tenant": "tenant-a"}}
    )

    state = ResolvedBillingIdentity(identity=identity)

    assert state.identity is not None
    with pytest.raises(TypeError, match="cannot be mutated"):
        state.identity.request_evidence["tenant"] = "tenant-b"  # type: ignore[index]


def test_completion_may_narrow_contexts_and_add_evidence() -> None:
    north = PricingContext(dimensions={"zone": "north", "tier": "reserved"})
    fallback = PricingContext(dimensions={"zone": "north", "tier": "standard"})
    requested = _identity(contexts=(north, fallback))
    completed = BillingIdentity(
        provider_name=requested.provider_name,
        resource_id=requested.resource_id,
        request_evidence=requested.request_evidence,
        completion_evidence={"effective_tier": "standard"},
        pricing_contexts=(fallback,),
    )

    assert completed_billing_identity(requested, completed) == completed


@pytest.mark.parametrize(
    "completed",
    [
        None,
        BillingIdentity(
            provider_name="other-cloud",
            resource_id="opaque-resource-1",
        ),
        BillingIdentity(
            provider_name="commercial-cloud",
            resource_id="different-resource",
        ),
        BillingIdentity(
            provider_name="commercial-cloud",
            resource_id="opaque-resource-1",
            request_evidence={"tenant": "tenant-b"},
        ),
        _identity(contexts=()),
        _identity(
            contexts=(
                PricingContext(dimensions={"zone": "north", "tier": "reserved"}),
                PricingContext(dimensions={"zone": "south", "tier": "reserved"}),
            )
        ),
    ],
)
def test_completion_rejects_identity_replacement_or_context_widening(
    completed: BillingIdentity | None,
) -> None:
    requested = _identity(
        contexts=(PricingContext(dimensions={"zone": "north", "tier": "reserved"}),)
    )

    with pytest.raises(ValueError):
        completed_billing_identity(requested, completed)


def test_completion_cannot_introduce_an_identity() -> None:
    with pytest.raises(ValueError, match="no request identity"):
        completed_billing_identity(None, _identity(contexts=()))


def test_context_free_identity_remains_context_free_at_completion() -> None:
    identity = _identity(contexts=())

    assert completed_billing_identity(identity, identity) == identity


def test_non_bedrock_contextual_identity_prices_end_to_end() -> None:
    identity = _identity(
        contexts=(PricingContext(dimensions={"zone": "north", "tier": "reserved"}),)
    )
    pricing = PriceBook(
        contextual_pricing_requirements=(
            ContextualPricingRequirement(
                provider_name="commercial-cloud",
                dimensions=("zone", "tier"),
            ),
        ),
        prices=(
            ModelPrice.fixed(
                provider_name="commercial-cloud",
                model=identity.resource_id,
                match="exact",
                pricing_context={
                    "zone": ("north",),
                    "tier": ("reserved",),
                },
                input_per_million=Decimal("2"),
                output_per_million=Decimal("6"),
            ),
        ),
    )
    event = Event(
        type=EventType.MODEL_COMPLETED,
        session_id="neutral-billing-session",
        timestamp=datetime(2026, 7, 17, tzinfo=UTC),
        payload={
            "provider_name": "renamed-commercial-cloud",
            "model": identity.resource_id,
            "billing_identity": identity.model_dump(mode="json"),
            "usage_metrics": {
                "provider_name": "renamed-commercial-cloud",
                "model": identity.resource_id,
                "billing_identity": identity.model_dump(mode="json"),
                "input_tokens": 1_000_000,
                "output_tokens": 1_000_000,
                "total_tokens": 2_000_000,
            },
        },
    )

    summary = estimate_session_cost(
        session_id=event.session_id,
        events=[event],
        pricing=pricing,
    )

    assert summary.total_cost == Decimal("8")
    assert summary.line_items[0].pricing_provider_name == "commercial-cloud"
    assert summary.line_items[0].billing_identity == identity


def test_context_free_identity_uses_ordinary_price_and_cache_rate() -> None:
    identity = _identity(contexts=())
    pricing = PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name=identity.provider_name,
                model=identity.resource_id,
                match="exact",
                input_per_million=Decimal("2"),
                output_per_million=Decimal("6"),
                cache_write_input_per_million=Decimal("3"),
            ),
        ),
    )
    event = Event(
        type=EventType.MODEL_COMPLETED,
        session_id="context-free-billing-session",
        timestamp=datetime(2026, 7, 17, tzinfo=UTC),
        payload={
            "billing_identity": identity.model_dump(mode="json"),
            "usage_metrics": {
                "provider_name": "renamed-commercial-cloud",
                "model": identity.resource_id,
                "input_tokens": 1_000_000,
                "output_tokens": 0,
                "total_tokens": 1_000_000,
                "cache": {
                    "write_tokens": 1_000_000,
                    "write_unknown_ttl_tokens": 1_000_000,
                    "uncached_input_tokens": 0,
                },
            },
        },
    )

    summary = estimate_session_cost(
        session_id=event.session_id,
        events=[event],
        pricing=pricing,
    )

    assert summary.total_cost == Decimal("3")
    assert summary.line_items[0].pricing_provider_name == identity.provider_name
    assert summary.line_items[0].billing_identity == identity


def test_contextual_identity_uses_generic_cache_rate_for_cost_and_reservation() -> None:
    identity = _identity(
        contexts=(PricingContext(dimensions={"zone": "north", "tier": "reserved"}),)
    )
    pricing = PriceBook(
        contextual_pricing_requirements=(
            ContextualPricingRequirement(
                provider_name=identity.provider_name,
                dimensions=("zone", "tier"),
            ),
        ),
        prices=(
            ModelPrice.fixed(
                provider_name=identity.provider_name,
                model=identity.resource_id,
                match="exact",
                pricing_context={
                    "zone": ("north",),
                    "tier": ("reserved",),
                },
                input_per_million=Decimal("2"),
                output_per_million=Decimal("6"),
                cache_write_input_per_million=Decimal("3"),
            ),
        ),
    )
    event = Event(
        type=EventType.MODEL_COMPLETED,
        session_id="generic-cache-billing-session",
        timestamp=datetime(2026, 7, 17, tzinfo=UTC),
        payload={
            "billing_identity": identity.model_dump(mode="json"),
            "usage_metrics": {
                "provider_name": "renamed-commercial-cloud",
                "model": identity.resource_id,
                "input_tokens": 1_000_000,
                "output_tokens": 0,
                "total_tokens": 1_000_000,
                "cache": {
                    "write_tokens": 1_000_000,
                    "write_unknown_ttl_tokens": 1_000_000,
                    "uncached_input_tokens": 0,
                },
            },
        },
    )
    summary = estimate_session_cost(
        session_id=event.session_id,
        events=[event],
        pricing=pricing,
    )
    reservation = asyncio.run(
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
            session_id=event.session_id,
            agent_name="assistant",
            provider_name="renamed-commercial-cloud",
            model=identity.resource_id,
            billing_identity=identity,
        )
    )

    assert summary.total_cost == Decimal("3")
    assert reservation.accepted is True
    assert reservation.requested == Decimal("3")
