from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from cayu import (
    Event,
    EventType,
    ModelCatalog,
    ModelInfo,
    ModelPricing,
    PriceTier,
    PricingCatalog,
    Provenance,
    TieredPricing,
    estimate_session_cost,
)
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


def _model_completed(
    *,
    provider_name: str,
    model: str,
    input_tokens: int,
    output_tokens: int = 0,
    session_id: str = "s",
) -> Event:
    return Event(
        type=EventType.MODEL_COMPLETED,
        session_id=session_id,
        payload={
            "usage_metrics": {
                "provider_name": provider_name,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "reasoning_output_tokens": 0,
                "cache": {
                    "read_tokens": 0,
                    "write_tokens": 0,
                    "cached_input_tokens": 0,
                    "uncached_input_tokens": 0,
                },
            }
        },
    )


def test_model_info_defaults_and_base_projection() -> None:
    info = _opus()
    # ModelInfo defaults to prefix matching; the flat ModelPricing default is unchanged.
    assert info.match == "prefix"
    assert ModelPricing.model_fields["match"].default == "exact"
    # to_model_pricing() uses the base (smallest-context) tier + the 5m cache-write rate.
    base = info.to_model_pricing()
    assert base.input_per_million == Decimal("5")
    assert base.output_per_million == Decimal("25")
    assert base.cache_write_input_per_million == Decimal("6.25")
    assert base.match == "prefix"


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
        ModelInfo(**base, modalities_in=("", "image"))
    with pytest.raises(ValidationError):
        ModelInfo(**base, modalities_out=("text", "  "))
    # But absent/empty is fine, and clean values pass.
    ok = ModelInfo(**base, family=None, aliases=(), modalities_in=("text", "image"))
    assert ok.family is None
    assert ModelInfo(**base, family="claude-opus").family == "claude-opus"


def test_pricing_catalog_projection_parity_with_flat_catalog() -> None:
    catalog = ModelCatalog(catalog_version="1", generated_at="2026-01-01", models=(_opus(),))
    projected = catalog.pricing_catalog()
    expected = PricingCatalog(
        prices=(
            ModelPricing(
                provider_name="anthropic",
                model="claude-opus-4-8",
                input_per_million=Decimal("5"),
                output_per_million=Decimal("25"),
                cache_write_input_per_million=Decimal("6.25"),
                match="prefix",
            ),
        )
    )
    assert projected == expected


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

    # Without the catalog, the flat projection prices BOTH steps at the base tier — proving the
    # catalog path is what selects the higher band for the big prompt (and flat records no band).
    flat = estimate_session_cost(session_id="s", events=events, pricing=catalog.pricing_catalog())
    assert flat.line_items[0].input_cost == Decimal("0.5")
    assert flat.line_items[1].input_cost == Decimal("1.5")  # 300000*5/1e6, NOT tier-aware
    assert flat.line_items[0].pricing_tier_max_input_tokens is None


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


def test_estimate_session_cost_requires_pricing_or_catalog() -> None:
    with pytest.raises(ValueError, match="requires pricing or catalog"):
        estimate_session_cost(session_id="s", events=[])


def test_model_catalog_rejects_duplicate_models_at_construction() -> None:
    # Duplicate (provider, model, match) must fail at construction, not deep in cost estimation.
    with pytest.raises(ValidationError, match="duplicate provider/model/match"):
        ModelCatalog(catalog_version="1", generated_at="2026-01-01", models=(_opus(), _opus()))


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
