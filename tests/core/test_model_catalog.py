from __future__ import annotations

import asyncio
from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from cayu import (
    BudgetLimit,
    BudgetReservation,
    Event,
    EventType,
    InMemoryBudgetLedger,
    ModelCatalog,
    ModelInfo,
    ModelPricing,
    PriceTier,
    PricingCatalog,
    Provenance,
    TieredPricing,
    default_model_catalog,
    estimate_session_cost,
)
from cayu.runtime.budgets import budget_actual_cost_for_event, budget_check_from_events
from cayu.runtime.costs import dump_model_catalog, load_model_catalog

_PROV = Provenance(source="official", url="https://example/pricing", as_of="2026-01-01")


def _tiered(
    *, base_in: str, base_out: str, over_in: str, over_out: str, boundary: int
) -> TieredPricing:
    return TieredPricing(
        standard=(
            PriceTier(
                max_input_tokens=boundary,
                input_per_million=Decimal(base_in),
                output_per_million=Decimal(base_out),
            ),
            PriceTier(
                max_input_tokens=None,
                input_per_million=Decimal(over_in),
                output_per_million=Decimal(over_out),
            ),
        ),
        cache_write_5m_per_million=Decimal("6.25"),
    )


def _opus() -> ModelInfo:
    return ModelInfo(
        provider_name="anthropic",
        model="claude-opus-4-8",
        family="claude-opus",
        aliases=("opus",),
        context_window=1_000_000,
        max_output_tokens=64_000,
        modalities_in=("text", "image"),
        tool_calling=True,
        reasoning=True,
        structured_output=True,
        prompt_caching=True,
        release_date=date(2026, 1, 1),
        pricing=_tiered(base_in="5", base_out="25", over_in="10", over_out="50", boundary=200_000),
        provenance=_PROV,
    )


def _cache_tier_catalog() -> ModelCatalog:
    return ModelCatalog(
        catalog_version="test",
        generated_at="2026-01-01",
        models=(
            ModelInfo(
                provider_name="anthropic",
                model="claude-test",
                pricing=TieredPricing(
                    standard=(
                        PriceTier(
                            max_input_tokens=200_000,
                            input_per_million=Decimal("1"),
                            output_per_million=Decimal("2"),
                            cache_read_input_per_million=Decimal("0.1"),
                            cache_write_input_per_million=Decimal("0.2"),
                        ),
                        PriceTier(
                            max_input_tokens=None,
                            input_per_million=Decimal("10"),
                            output_per_million=Decimal("20"),
                            cache_read_input_per_million=Decimal("1"),
                            cache_write_input_per_million=Decimal("2"),
                        ),
                    )
                ),
                provenance=_PROV,
            ),
        ),
    )


def _model_completed(
    *,
    provider_name: str,
    model: str,
    input_tokens: int,
    output_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    cache_write_input_tokens: int = 0,
    session_id: str = "s",
    requested_model: str | None = None,
) -> Event:
    uncached_input_tokens = input_tokens - cache_read_input_tokens - cache_write_input_tokens
    return Event(
        type=EventType.MODEL_COMPLETED,
        session_id=session_id,
        payload={
            "usage_metrics": {
                "provider_name": provider_name,
                "model": model,
                "requested_model": requested_model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "reasoning_output_tokens": 0,
                "cache": {
                    "read_tokens": cache_read_input_tokens,
                    "write_tokens": cache_write_input_tokens,
                    "cached_input_tokens": cache_read_input_tokens,
                    "uncached_input_tokens": uncached_input_tokens,
                },
            }
        },
    )


def test_model_info_defaults_and_base_projection() -> None:
    info = _opus()
    # Both catalog records and one-off prices accept dated/suffixed runtime ids by default.
    assert info.match == "prefix"
    assert ModelPricing.model_fields["match"].default == "prefix"
    # to_model_pricing() uses the base (smallest-context) tier + the 5m cache-write rate.
    base = info.to_model_pricing()
    assert base.input_per_million == Decimal("5")
    assert base.output_per_million == Decimal("25")
    assert base.cache_write_input_per_million == Decimal("6.25")
    assert base.match == "prefix"


def test_one_off_pricing_defaults_to_prefix_and_exact_is_explicit() -> None:
    prefix = PricingCatalog(
        prices=(
            ModelPricing(
                provider_name="openai",
                model="gpt-5.5",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("2"),
            ),
        )
    )

    assert prefix.match_price(provider_name="openai", model="gpt-5.5") is not None
    assert prefix.match_price(provider_name="openai", model="gpt-5.5-2026-07-01") is not None
    assert prefix.match_price(provider_name="openai", model="gpt-5.50") is not None
    assert "pricing_tiers" not in prefix.prices[0].model_dump(mode="json")
    assert "provenance" not in prefix.prices[0].model_dump(mode="json")

    exact = PricingCatalog(prices=(prefix.prices[0].model_copy(update={"match": "exact"}),))
    assert exact.match_price(provider_name="openai", model="gpt-5.5") is not None
    assert exact.match_price(provider_name="openai", model="gpt-5.5-2026-07-01") is None


def test_pricing_catalog_selects_the_longest_matching_prefix() -> None:
    broad = ModelPricing(
        provider_name="openai",
        model="gpt-5",
        input_per_million=Decimal("1"),
        output_per_million=Decimal("2"),
    )
    specific = ModelPricing(
        provider_name="openai",
        model="gpt-5.5",
        input_per_million=Decimal("3"),
        output_per_million=Decimal("4"),
    )
    catalog = PricingCatalog(prices=(broad, specific))

    matched = catalog.match_price(provider_name="openai", model="gpt-5.5-2026-07-01")

    assert matched == specific


def test_bundled_exact_alias_does_not_match_a_nearby_model_family() -> None:
    catalog = default_model_catalog()

    matched = catalog.resolve(provider_name="openai", model="gpt-5.6")
    assert matched is not None
    assert matched.model == "gpt-5.6-sol"
    assert catalog.resolve(provider_name="openai", model="gpt-5.60") is None
    assert catalog.resolve(provider_name="openai", model="gpt-5.6-mini") is None

    projected = catalog.pricing_catalog()
    assert projected.match_price(provider_name="openai", model="gpt-5.6") is not None
    assert projected.match_price(provider_name="openai", model="gpt-5.60") is None
    assert projected.match_price(provider_name="openai", model="gpt-5.6-mini") is None


def test_default_model_catalog_is_bundled_validated_and_offline() -> None:
    first = default_model_catalog()
    second = default_model_catalog()

    assert first == second
    assert first is not second
    assert first.catalog_version == first.generated_at
    assert {model.provider_name for model in first.models} == {
        "anthropic",
        "azure",
        "google",
        "openai",
        "vertex",
    }
    assert all(model.tool_calling for model in first.models)
    assert all(not model.deprecated for model in first.models)
    assert all(model.pricing.currency == "USD" for model in first.models)
    assert all(model.provenance.source == "official" for model in first.models)
    assert all(model.provenance.url.startswith("https://") for model in first.models)


def test_default_model_catalog_prices_exact_aliases_with_provenance() -> None:
    catalog = default_model_catalog()
    expected = next(
        model
        for model in catalog.models
        if model.provider_name == "openai" and model.model == "gpt-5.6-sol"
    )
    runtime_model = "gpt-5.6"
    resolved = catalog.resolve(provider_name="OPENAI", model=runtime_model)
    assert resolved is not None
    assert resolved == expected

    summary = estimate_session_cost(
        session_id="s",
        events=[
            _model_completed(
                provider_name="openai",
                model=runtime_model,
                input_tokens=1_000_000,
                output_tokens=1_000_000,
            )
        ],
        catalog=catalog,
    )

    assert summary.priced_model_steps == 1
    selected_price = expected.pricing_at(1_000_000)
    assert summary.total_cost == (
        selected_price.input_per_million + selected_price.output_per_million
    )
    assert summary.line_items[0].pricing_model == "gpt-5.6"
    assert summary.line_items[0].pricing_match == "exact"
    assert summary.line_items[0].pricing_provenance == resolved.provenance


def test_every_bundled_match_prefix_resolves_without_broadening_model_families() -> None:
    catalog = default_model_catalog()

    for model in catalog.models:
        assert model.match == "exact"
        assert catalog.match(provider_name=model.provider_name, model=model.model) == model
        for alias in model.aliases:
            assert catalog.resolve(provider_name=model.provider_name, model=alias) == model
            assert catalog.resolve(provider_name=model.provider_name, model=f"{alias}-mini") is None
        for prefix in model.match_prefixes:
            assert (
                catalog.resolve(
                    provider_name=model.provider_name,
                    model=f"{prefix}260101",
                )
                == model
            )
        assert (
            catalog.resolve(
                provider_name=model.provider_name,
                model=f"{model.model}-mini",
            )
            is None
        )


def test_default_model_catalog_leaves_unknown_models_unpriced() -> None:
    summary = estimate_session_cost(
        session_id="s",
        events=[
            _model_completed(
                provider_name="openai",
                model="future-model",
                input_tokens=1_000,
            )
        ],
        catalog=default_model_catalog(),
    )

    assert summary.priced_model_steps == 0
    assert summary.unpriced_model_steps == 1
    assert summary.line_items[0].missing_pricing_reason == "no matching model pricing"


def test_provider_resolved_unknown_model_does_not_fall_back_to_requested_pricing() -> None:
    summary = estimate_session_cost(
        session_id="s",
        events=[
            _model_completed(
                provider_name="openai",
                requested_model="gpt-5.5",
                model="future-expensive-model",
                input_tokens=1_000,
            )
        ],
        catalog=default_model_catalog(),
    )

    assert summary.priced_model_steps == 0
    assert summary.unpriced_model_steps == 1
    assert summary.line_items[0].model == "future-expensive-model"
    assert summary.line_items[0].requested_model == "gpt-5.5"
    assert summary.line_items[0].missing_pricing_reason == "no matching model pricing"


def test_provider_resolved_unknown_model_fails_closed_after_usage() -> None:
    check = budget_check_from_events(
        limit=BudgetLimit(
            max_estimated_cost=Decimal("100"),
            pricing=default_model_catalog(),
        ),
        events=[
            _model_completed(
                provider_name="openai",
                requested_model="gpt-5.5",
                model="future-expensive-model",
                input_tokens=1_000,
            )
        ],
        provider_name="openai",
        model="gpt-5.5",
    )

    assert check.limit_reached is True
    assert check.unpriced_model_steps == 1
    assert "no matching pricing" in check.message


def test_budget_limit_accepts_default_model_catalog() -> None:
    catalog = default_model_catalog()
    limit = BudgetLimit(
        max_estimated_cost=Decimal("1"),
        pricing=catalog,
    )

    assert isinstance(limit.pricing, ModelCatalog)

    expected = next(
        model
        for model in catalog.models
        if model.provider_name == "openai" and model.model == "gpt-5.6-sol"
    )
    info = catalog.resolve(
        provider_name=expected.provider_name,
        model="gpt-5.6",
    )
    assert info is not None
    assert info == expected


def test_default_catalog_known_model_supports_causal_reservation_and_reconciliation() -> None:
    async def run():
        catalog = default_model_catalog()
        expected = next(
            model
            for model in catalog.models
            if model.provider_name == "openai" and model.model == "gpt-5.6-sol"
        )
        runtime_model = "gpt-5.6"
        limit = BudgetLimit(
            scope="causal",
            key="job-default-catalog",
            max_estimated_cost=Decimal("100"),
            pricing=catalog,
            reservation=BudgetReservation(max_input_tokens=1, max_output_tokens=0),
        )
        ledger = InMemoryBudgetLedger()
        reserved = await ledger.reserve(
            limit=limit,
            session_id="s",
            agent_name="assistant",
            provider_name=expected.provider_name,
            model=runtime_model,
        )
        assert reserved.record is not None
        actual = budget_actual_cost_for_event(
            limit=limit,
            event=_model_completed(
                provider_name=expected.provider_name,
                model=runtime_model,
                input_tokens=1,
            ),
        )
        reconciled = await ledger.reconcile(
            reservation_id=reserved.record.reservation_id,
            actual_amount=actual,
            reason="model completed",
        )
        return expected, reserved, reconciled

    expected, reserved, reconciled = asyncio.run(run())
    expected_cost = expected.pricing_at(1).input_per_million / Decimal("1000000")

    assert reserved.accepted is True
    assert reserved.requested == expected_cost
    assert reconciled.actual_amount == expected_cost
    assert reconciled.released_amount == Decimal("0")


def test_default_catalog_unknown_model_budget_fails_closed_before_usage() -> None:
    check = budget_check_from_events(
        limit=BudgetLimit(
            max_estimated_cost=Decimal("100"),
            pricing=default_model_catalog(),
        ),
        events=[],
        provider_name="openai",
        model="future-unknown-model",
    )

    assert check.limit_reached is True
    assert check.model_steps == 0
    assert "no matching pricing" in check.message


def test_default_catalog_allow_unpriced_keeps_unknown_usage_explicitly_unpriced() -> None:
    check = budget_check_from_events(
        limit=BudgetLimit(
            max_estimated_cost=Decimal("100"),
            pricing=default_model_catalog(),
            allow_unpriced=True,
        ),
        events=[
            _model_completed(
                provider_name="openai",
                model="future-unknown-model",
                input_tokens=1_000,
            )
        ],
        provider_name="openai",
        model="future-unknown-model",
    )

    assert check.limit_reached is False
    assert check.unpriced_model_steps == 1
    assert check.actual == Decimal("0")
    assert check.cost_summary.line_items[0].priced is False
    assert check.cost_summary.line_items[0].missing_pricing_reason == "no matching model pricing"


def test_budget_limit_uses_the_matching_context_tier() -> None:
    catalog = ModelCatalog(
        catalog_version="test",
        generated_at="2026-01-01",
        models=(_opus(),),
    )
    event = _model_completed(
        provider_name="anthropic",
        model="claude-opus-4-8-20260101",
        input_tokens=300_000,
    )

    check = budget_check_from_events(
        limit=BudgetLimit(max_estimated_cost=Decimal("2.5"), pricing=catalog),
        events=[event],
        provider_name="anthropic",
        model="claude-opus-4-8-20260101",
    )

    assert check.actual == Decimal("3")
    assert check.limit_reached is True


def test_context_tier_selects_its_cache_write_rate() -> None:
    catalog = _cache_tier_catalog()
    summary = estimate_session_cost(
        session_id="s",
        events=[
            _model_completed(
                provider_name="anthropic",
                model="claude-test",
                input_tokens=300_000,
                cache_write_input_tokens=300_000,
            )
        ],
        catalog=catalog,
    )

    assert summary.line_items[0].cache_write_input_cost == Decimal("0.6")
    projected = estimate_session_cost(
        session_id="s",
        events=[
            _model_completed(
                provider_name="anthropic",
                model="claude-test",
                input_tokens=300_000,
                cache_write_input_tokens=300_000,
            )
        ],
        pricing=catalog.pricing_catalog(),
    )
    assert projected == summary


def test_causal_reservation_selects_tier_from_all_reserved_input() -> None:
    async def run():
        ledger = InMemoryBudgetLedger()
        limit = BudgetLimit(
            scope="causal",
            key="job-tiered",
            max_estimated_cost=Decimal("10"),
            pricing=_cache_tier_catalog(),
            reservation=BudgetReservation(
                max_input_tokens=100_000,
                max_output_tokens=0,
                max_cache_read_input_tokens=150_000,
            ),
        )
        reserved = await ledger.reserve(
            limit=limit,
            session_id="s",
            agent_name="assistant",
            provider_name="anthropic",
            model="claude-test-2026-07-01",
        )
        assert reserved.record is not None
        actual = budget_actual_cost_for_event(
            limit=limit,
            event=_model_completed(
                provider_name="anthropic",
                model="claude-test-2026-07-01",
                input_tokens=250_000,
                cache_read_input_tokens=150_000,
            ),
        )
        reconciled = await ledger.reconcile(
            reservation_id=reserved.record.reservation_id,
            actual_amount=actual,
            reason="model completed",
        )
        return reserved, reconciled

    reserved, reconciled = asyncio.run(run())

    assert reserved.accepted is True
    assert reserved.requested == Decimal("1.15")
    assert reconciled.actual_amount == Decimal("1.15")
    assert reconciled.released_amount == Decimal("0")


def test_projected_pricing_catalog_preserves_reservation_tiers() -> None:
    async def run():
        return await InMemoryBudgetLedger().reserve(
            limit=BudgetLimit(
                scope="causal",
                key="job-projected-tiered",
                max_estimated_cost=Decimal("10"),
                pricing=_cache_tier_catalog().pricing_catalog(),
                reservation=BudgetReservation(
                    max_input_tokens=100_000,
                    max_output_tokens=0,
                    max_cache_read_input_tokens=150_000,
                ),
            ),
            session_id="s",
            agent_name="assistant",
            provider_name="anthropic",
            model="claude-test-2026-07-01",
        )

    result = asyncio.run(run())

    assert result.accepted is True
    assert result.requested == Decimal("1.15")


def test_reservation_uses_upper_tier_cache_write_rate() -> None:
    async def run():
        return await InMemoryBudgetLedger().reserve(
            limit=BudgetLimit(
                scope="causal",
                key="job-cache-write-tiered",
                max_estimated_cost=Decimal("10"),
                pricing=_cache_tier_catalog(),
                reservation=BudgetReservation(
                    max_input_tokens=100_000,
                    max_output_tokens=0,
                    max_cache_write_input_tokens=150_000,
                ),
            ),
            session_id="s",
            agent_name="assistant",
            provider_name="anthropic",
            model="claude-test",
        )

    result = asyncio.run(run())

    assert result.accepted is True
    assert result.requested == Decimal("1.3")


@pytest.mark.parametrize(
    ("uncached_input_tokens", "expected"),
    [
        (50_000, Decimal("0.065")),
        (50_001, Decimal("0.65001")),
    ],
)
def test_reservation_total_input_tier_boundary(
    uncached_input_tokens: int,
    expected: Decimal,
) -> None:
    async def run():
        return await InMemoryBudgetLedger().reserve(
            limit=BudgetLimit(
                scope="causal",
                key="job-boundary",
                max_estimated_cost=Decimal("10"),
                pricing=_cache_tier_catalog(),
                reservation=BudgetReservation(
                    max_input_tokens=uncached_input_tokens,
                    max_output_tokens=0,
                    max_cache_read_input_tokens=150_000,
                ),
            ),
            session_id="s",
            agent_name="assistant",
            provider_name="anthropic",
            model="claude-test",
        )

    result = asyncio.run(run())

    assert result.accepted is True
    assert result.requested == expected


def test_tier_for_selects_the_right_band() -> None:
    pricing = _tiered(base_in="5", base_out="25", over_in="10", over_out="50", boundary=200_000)
    assert pricing.tier_for(1).input_per_million == Decimal("5")
    assert pricing.tier_for(200_000).input_per_million == Decimal("5")  # exactly at the boundary
    assert pricing.tier_for(200_001).input_per_million == Decimal("10")  # one over rolls up
    assert pricing.tier_for(10_000_000).input_per_million == Decimal(
        "10"
    )  # open-ended catches overflow


def test_tier_for_without_open_ended_tier_falls_back_to_top_band() -> None:
    pricing = TieredPricing(
        standard=(
            PriceTier(
                max_input_tokens=100,
                input_per_million=Decimal("1"),
                output_per_million=Decimal("2"),
            ),
            PriceTier(
                max_input_tokens=200,
                input_per_million=Decimal("3"),
                output_per_million=Decimal("4"),
            ),
        )
    )
    assert pricing.tier_for(500).input_per_million == Decimal(
        "3"
    )  # past all bounds -> largest band


def test_tiered_pricing_rejects_malformed_tiers() -> None:
    with pytest.raises(ValidationError):
        TieredPricing(standard=())  # empty
    with pytest.raises(ValidationError):  # not ascending
        TieredPricing(
            standard=(
                PriceTier(
                    max_input_tokens=200,
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
                PriceTier(
                    max_input_tokens=100,
                    input_per_million=Decimal("3"),
                    output_per_million=Decimal("4"),
                ),
            )
        )
    with pytest.raises(ValidationError):  # open-ended tier not last
        TieredPricing(
            standard=(
                PriceTier(
                    max_input_tokens=None,
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
                PriceTier(
                    max_input_tokens=100,
                    input_per_million=Decimal("3"),
                    output_per_million=Decimal("4"),
                ),
            )
        )
    with pytest.raises(ValidationError, match="base rates must match"):
        ModelPricing(
            provider_name="anthropic",
            model="claude-opus-4-8",
            input_per_million=Decimal("1"),
            output_per_million=Decimal("2"),
            pricing_tiers=_tiered(
                base_in="5",
                base_out="25",
                over_in="10",
                over_out="50",
                boundary=200_000,
            ).standard,
        )


def test_entities_are_frozen_and_forbid_extra() -> None:
    info = _opus()
    with pytest.raises(ValidationError):
        info.deprecated = True  # frozen
    with pytest.raises(ValidationError):
        ModelInfo.model_validate({**info.model_dump(), "surprise": 1})  # extra forbidden


def test_model_info_rejects_blank_optional_and_list_string_fields() -> None:
    base = {
        "provider_name": "anthropic",
        "model": "claude-opus-4-8",
        "pricing": _tiered(
            base_in="5", base_out="25", over_in="10", over_out="50", boundary=200_000
        ),
        "provenance": _PROV,
    }
    # A provided family/alias/modality must be a clean non-blank string.
    with pytest.raises(ValidationError):
        ModelInfo(**base, family="   ")
    with pytest.raises(ValidationError):
        ModelInfo(**base, aliases=("opus", " "))
    with pytest.raises(ValidationError):
        ModelInfo(**base, match_prefixes=("claude-opus-4-8-20", " "))
    with pytest.raises(ValidationError):
        ModelInfo(**base, modalities_in=("", "image"))
    with pytest.raises(ValidationError):
        ModelInfo(**base, modalities_out=("text", "  "))
    # But absent/empty is fine, and clean values pass.
    ok = ModelInfo(**base, family=None, aliases=(), modalities_in=("text", "image"))
    assert ok.family is None
    assert ModelInfo(**base, family="claude-opus").family == "claude-opus"


def test_pricing_catalog_projection_retains_tiers_and_provenance() -> None:
    catalog = ModelCatalog(catalog_version="1", generated_at="2026-01-01", models=(_opus(),))
    projected = catalog.pricing_catalog()
    price = projected.prices[0]

    assert price.input_per_million == Decimal("5")
    assert price.output_per_million == Decimal("25")
    assert price.cache_write_input_per_million == Decimal("6.25")
    assert price.match == "prefix"
    assert price.pricing_tiers is not None
    assert tuple(tier.cache_write_input_per_million for tier in price.pricing_tiers) == (
        Decimal("6.25"),
        Decimal("6.25"),
    )
    assert price.provenance == _PROV
    assert price.pricing_tiers[0].model_dump(mode="json")["cache_write_input_per_million"] == (
        "6.25"
    )
    assert PricingCatalog.model_validate_json(projected.model_dump_json()) == projected


def test_pricing_catalog_projection_materializes_each_tier_cache_write_fallback() -> None:
    info = ModelInfo(
        provider_name="anthropic",
        model="claude-fallback-test",
        pricing=TieredPricing(
            standard=(
                PriceTier(
                    max_input_tokens=200_000,
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                    cache_write_input_per_million=Decimal("2"),
                ),
                PriceTier(
                    max_input_tokens=None,
                    input_per_million=Decimal("10"),
                    output_per_million=Decimal("20"),
                ),
            ),
            cache_write_5m_per_million=Decimal("7"),
        ),
        provenance=_PROV,
    )
    catalog = ModelCatalog(catalog_version="test", generated_at="2026-01-01", models=(info,))
    projected = catalog.pricing_catalog()
    round_tripped = PricingCatalog.model_validate_json(projected.model_dump_json())
    assert round_tripped.prices[0].pricing_tiers is not None
    assert tuple(
        tier.cache_write_input_per_million for tier in round_tripped.prices[0].pricing_tiers
    ) == (Decimal("2"), Decimal("7"))

    event = _model_completed(
        provider_name="anthropic",
        model="claude-fallback-test",
        input_tokens=300_000,
        cache_write_input_tokens=300_000,
    )
    direct = estimate_session_cost(session_id="s", events=[event], catalog=catalog)
    projected_summary = estimate_session_cost(
        session_id="s",
        events=[event],
        pricing=round_tripped,
    )

    assert direct.line_items[0].cache_write_input_cost == Decimal("2.1")
    assert projected_summary == direct


def test_model_catalog_match_and_route() -> None:
    opus = _opus()
    old = ModelInfo(
        provider_name="anthropic",
        model="claude-2",
        context_window=100_000,
        tool_calling=False,
        deprecated=True,
        pricing=_tiered(base_in="8", base_out="24", over_in="8", over_out="24", boundary=50_000),
        provenance=_PROV,
    )
    catalog = ModelCatalog(catalog_version="1", generated_at="2026-01-01", models=(opus, old))

    assert catalog.match(provider_name="anthropic", model="claude-opus-4-8") is opus
    assert catalog.match(provider_name="anthropic", model="nope") is None
    # deprecated excluded by default; capability + context filters apply.
    assert catalog.route() == [opus]
    assert catalog.route(include_deprecated=True) == [opus, old]
    assert catalog.route(min_context=500_000, needs_vision=True, needs_tools=True) == [opus]
    assert catalog.route(needs_vision=True, include_deprecated=True) == [opus]  # old is text-only


def test_route_provider_filter_is_case_insensitive() -> None:
    # route()'s provider filter must normalize case/whitespace like resolve()/pricing, so a
    # differently-cased provider from user config doesn't silently hide valid models.
    catalog = ModelCatalog(catalog_version="1", generated_at="2026-01-01", models=(_opus(),))
    assert catalog.route(provider_name="Anthropic") == [_opus()]
    assert catalog.route(provider_name=" ANTHROPIC ") == [_opus()]
    assert catalog.route(provider_name="anthropic") == [_opus()]
    assert catalog.route(provider_name="openai") == []  # absent provider still filters out
    # route and resolve now agree on casing.
    assert catalog.resolve(provider_name="Anthropic", model="claude-opus-4-8") is not None


def test_dumps_is_deterministic_and_load_round_trips(tmp_path) -> None:
    b = _opus()
    a = ModelInfo(
        provider_name="anthropic",
        model="claude-haiku-4-5",
        context_window=200_000,
        pricing=_tiered(base_in="1", base_out="5", over_in="1", over_out="5", boundary=200_000),
        provenance=_PROV,
    )
    unsorted = ModelCatalog(catalog_version="1", generated_at="2026-01-01", models=(b, a))
    sorted_ = ModelCatalog(catalog_version="1", generated_at="2026-01-01", models=(a, b))
    # Serialization is order-independent (models sorted by provider/model) and ends with a newline.
    assert dump_model_catalog(unsorted) == dump_model_catalog(sorted_)
    assert dump_model_catalog(unsorted).endswith("\n")
    # Round-trip through a file.
    path = tmp_path / "catalog.json"
    path.write_text(dump_model_catalog(unsorted))
    assert load_model_catalog(path) == sorted_


def test_estimate_session_cost_with_catalog_selects_the_input_token_tier() -> None:
    catalog = ModelCatalog(catalog_version="1", generated_at="2026-01-01", models=(_opus(),))
    # A dated model id must prefix-match the catalog entry; input crosses the 200k boundary.
    events = [
        _model_completed(
            provider_name="anthropic", model="claude-opus-4-8-20260101", input_tokens=100_000
        ),
        _model_completed(
            provider_name="anthropic", model="claude-opus-4-8-20260101", input_tokens=300_000
        ),
    ]

    tiered = estimate_session_cost(session_id="s", events=events, catalog=catalog)
    # 100k step priced at the base band (5/M): 100000*5/1e6 = 0.5; band ceiling recorded.
    assert tiered.line_items[0].input_cost == Decimal("0.5")
    assert tiered.line_items[0].pricing_tier_max_input_tokens == 200_000
    # 300k step priced at the open-ended band (10/M): 300000*10/1e6 = 3.0; open-ended band -> None.
    assert tiered.line_items[1].input_cost == Decimal("3.0")
    assert tiered.line_items[1].pricing_model == "claude-opus-4-8"
    assert tiered.line_items[1].pricing_tier_max_input_tokens is None

    projected = estimate_session_cost(
        session_id="s",
        events=events,
        pricing=catalog.pricing_catalog(),
    )
    assert projected == tiered


def test_estimate_session_cost_catalog_falls_back_to_flat_pricing() -> None:
    catalog = ModelCatalog(catalog_version="1", generated_at="2026-01-01", models=(_opus(),))
    flat = PricingCatalog(
        prices=(
            ModelPricing(
                provider_name="openai",
                model="gpt-5.5",
                match="prefix",
                input_per_million=Decimal("2"),
                output_per_million=Decimal("8"),
            ),
        )
    )
    # openai model is absent from the catalog but present in the flat pricing -> flat used.
    events = [_model_completed(provider_name="openai", model="gpt-5.5-2026", input_tokens=1_000)]
    summary = estimate_session_cost(session_id="s", events=events, pricing=flat, catalog=catalog)
    assert summary.priced_model_steps == 1
    assert summary.line_items[0].pricing_model == "gpt-5.5"
    assert summary.line_items[0].input_cost == Decimal("0.002")


def test_flat_pricing_overrides_same_currency_catalog_match() -> None:
    catalog = ModelCatalog(catalog_version="1", generated_at="2026-01-01", models=(_opus(),))
    flat = PricingCatalog(
        prices=(
            ModelPricing(
                provider_name="anthropic",
                model="claude-opus-4-8",
                match="prefix",
                input_per_million=Decimal("2"),
                output_per_million=Decimal("8"),
            ),
        )
    )
    events = [
        _model_completed(
            provider_name="anthropic",
            model="claude-opus-4-8-20260101",
            input_tokens=100_000,
        )
    ]

    summary = estimate_session_cost(session_id="s", events=events, pricing=flat, catalog=catalog)

    assert summary.priced_model_steps == 1
    assert summary.line_items[0].input_cost == Decimal("0.2")
    assert summary.line_items[0].pricing_provenance is None


def test_catalog_currency_match_beats_mismatched_flat_override() -> None:
    catalog = ModelCatalog(catalog_version="1", generated_at="2026-01-01", models=(_opus(),))
    flat = PricingCatalog(
        prices=(
            ModelPricing(
                provider_name="anthropic",
                model="claude-opus-4-8",
                match="prefix",
                currency="EUR",
                input_per_million=Decimal("2"),
                output_per_million=Decimal("8"),
            ),
        )
    )
    events = [
        _model_completed(
            provider_name="anthropic",
            model="claude-opus-4-8-20260101",
            input_tokens=100_000,
        )
    ]

    summary = estimate_session_cost(session_id="s", events=events, pricing=flat, catalog=catalog)

    assert summary.priced_model_steps == 1
    assert summary.line_items[0].input_cost == Decimal("0.5")
    assert summary.line_items[0].pricing_provenance == _PROV


def test_estimate_session_cost_requires_pricing_or_catalog() -> None:
    with pytest.raises(ValueError, match="requires pricing or catalog"):
        estimate_session_cost(session_id="s", events=[])


def test_model_catalog_rejects_duplicate_models_at_construction() -> None:
    # Duplicate (provider, model, match) must fail at construction, not deep in cost estimation.
    with pytest.raises(ValidationError, match="duplicate provider/model/match"):
        ModelCatalog(catalog_version="1", generated_at="2026-01-01", models=(_opus(), _opus()))


def test_model_catalog_rejects_duplicate_explicit_match_prefixes() -> None:
    first = _opus().model_copy(update={"match": "exact", "match_prefixes": ("claude-snapshot-20",)})
    second = _opus().model_copy(
        update={
            "model": "claude-sonnet-4-6",
            "match": "exact",
            "match_prefixes": ("claude-snapshot-20",),
        }
    )

    with pytest.raises(ValidationError, match="duplicate provider/model/match"):
        ModelCatalog(catalog_version="1", generated_at="2026-01-01", models=(first, second))


def test_resolve_is_prefix_and_case_insensitive_matching_the_cost_engine() -> None:
    catalog = ModelCatalog(catalog_version="1", generated_at="2026-01-01", models=(_opus(),))
    # A dated runtime id + differently-cased provider resolves to the base record (like pricing).
    resolved = catalog.resolve(provider_name="Anthropic", model="claude-opus-4-8-20260101")
    assert resolved is not None
    assert resolved.model == "claude-opus-4-8"
    # match() stays strictly exact.
    assert catalog.match(provider_name="Anthropic", model="claude-opus-4-8-20260101") is None
    assert catalog.resolve(provider_name="anthropic", model="gpt-5.5") is None


def test_catalog_currency_mismatch_falls_back_to_flat_pricing_instead_of_shadowing() -> None:
    # Same model in the catalog (EUR) and the flat catalog (USD); requesting USD must use the
    # flat USD price, not mark the step unpriced because the catalog entry is EUR.
    eur = ModelInfo(
        provider_name="anthropic",
        model="claude-opus-4-8",
        pricing=TieredPricing(
            currency="EUR",
            standard=(
                PriceTier(
                    max_input_tokens=None,
                    input_per_million=Decimal("4"),
                    output_per_million=Decimal("20"),
                ),
            ),
        ),
        provenance=_PROV,
    )
    catalog = ModelCatalog(catalog_version="1", generated_at="2026-01-01", models=(eur,))
    flat = PricingCatalog(
        prices=(
            ModelPricing(
                provider_name="anthropic",
                model="claude-opus-4-8",
                match="prefix",
                input_per_million=Decimal("5"),
                output_per_million=Decimal("25"),
            ),
        )
    )
    events = [
        _model_completed(
            provider_name="anthropic", model="claude-opus-4-8-20260101", input_tokens=100_000
        )
    ]
    summary = estimate_session_cost(session_id="s", events=events, pricing=flat, catalog=catalog)
    assert summary.priced_model_steps == 1  # priced from flat USD, not shadowed by the EUR catalog
    assert summary.line_items[0].input_cost == Decimal("0.5")  # 100000 * 5 (USD) / 1e6
