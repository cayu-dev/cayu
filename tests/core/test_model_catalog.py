from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from cayu import (
    Event,
    EventType,
    ModelCatalog,
    ModelInfo,
    ModelPrice,
    PriceBook,
    PriceSchedule,
    PriceTier,
    Provenance,
    TieredPricing,
    default_model_catalog,
    default_price_book,
    dump_model_catalog,
    dump_price_book,
    estimate_session_cost,
    load_model_catalog,
    load_price_book,
)

_PROVENANCE = Provenance(
    source="official",
    url="https://example.com/models",
    as_of="2026-07-14",
)


def _opus(**updates: object) -> ModelInfo:
    values: dict[str, object] = {
        "provider_name": "anthropic",
        "model": "claude-opus-4-8",
        "family": "claude-opus",
        "aliases": ("opus",),
        "context_window": 1_000_000,
        "max_output_tokens": 64_000,
        "modalities_in": ("text", "image"),
        "tool_calling": True,
        "reasoning": True,
        "structured_output": True,
        "prompt_caching": True,
        "release_date": date(2026, 1, 1),
        "provenance": _PROVENANCE,
    }
    values.update(updates)
    return ModelInfo(**values)


def _catalog(*models: ModelInfo) -> ModelCatalog:
    return ModelCatalog(
        catalog_version="test",
        generated_at="2026-07-14",
        models=models or (_opus(),),
    )


def _completed(
    *,
    provider_name: str = "anthropic",
    model: str = "claude-opus-4-8",
    requested_model: str | None = None,
    input_tokens: int,
    output_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    cache_write_input_tokens: int = 0,
) -> Event:
    return Event(
        type=EventType.MODEL_COMPLETED,
        session_id="session-1",
        timestamp=datetime(2026, 7, 14, tzinfo=UTC),
        payload={
            "usage_metrics": {
                "provider_name": provider_name,
                "requested_model": requested_model,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "reasoning_output_tokens": 0,
                "cache": {
                    "read_tokens": cache_read_input_tokens,
                    "write_tokens": cache_write_input_tokens,
                    "cached_input_tokens": cache_read_input_tokens,
                    "uncached_input_tokens": (
                        input_tokens - cache_read_input_tokens - cache_write_input_tokens
                    ),
                },
            }
        },
    )


def _tiered_price_book() -> PriceBook:
    return PriceBook(
        price_book_version="test",
        generated_at="2026-07-14",
        prices=(
            ModelPrice(
                provider_name="anthropic",
                model="claude-opus-4-8",
                aliases=("opus",),
                schedules=(
                    PriceSchedule(
                        pricing=TieredPricing(
                            standard=(
                                PriceTier(
                                    max_input_tokens=200_000,
                                    input_per_million=Decimal("5"),
                                    output_per_million=Decimal("25"),
                                    cache_read_input_per_million=Decimal("0.5"),
                                ),
                                PriceTier(
                                    max_input_tokens=None,
                                    input_per_million=Decimal("10"),
                                    output_per_million=Decimal("50"),
                                    cache_read_input_per_million=Decimal("1"),
                                ),
                            ),
                            cache_write_5m_per_million=Decimal("6.25"),
                        ),
                        provenance=_PROVENANCE,
                    ),
                ),
            ),
        ),
    )


def test_model_info_is_metadata_only() -> None:
    info = _opus()

    assert "pricing" not in ModelInfo.model_fields
    assert info.context_window == 1_000_000
    assert info.tool_calling is True
    assert info.match == "prefix"


def test_model_catalog_resolves_aliases_and_the_most_specific_prefix() -> None:
    broad = _opus(model="claude-opus", aliases=())
    specific = _opus(model="claude-opus-4-8", aliases=("opus",))
    catalog = _catalog(broad, specific)

    assert catalog.resolve(provider_name="ANTHROPIC", model="opus") == specific
    assert catalog.resolve(provider_name="anthropic", model="claude-opus-4-8-20260714") == specific
    assert catalog.resolve(provider_name="anthropic", model="claude-opus-future") == broad


def test_model_catalog_exact_match_does_not_capture_a_nearby_family() -> None:
    catalog = _catalog(_opus(match="exact"))

    assert catalog.resolve(provider_name="anthropic", model="claude-opus-4-8") is not None
    assert catalog.resolve(provider_name="anthropic", model="claude-opus-4-80") is None


def test_model_catalog_match_is_canonical_record_lookup() -> None:
    catalog = _catalog()

    assert catalog.match(provider_name="anthropic", model="claude-opus-4-8") == _opus()
    assert catalog.match(provider_name="ANTHROPIC", model="claude-opus-4-8") is None
    assert catalog.match(provider_name="anthropic", model="opus") is None


def test_model_catalog_routes_on_objective_capabilities() -> None:
    vision = _opus()
    text = _opus(
        model="claude-text",
        aliases=(),
        modalities_in=("text",),
        tool_calling=False,
        context_window=100_000,
    )
    retired = _opus(model="claude-retired", aliases=(), deprecated=True)
    catalog = _catalog(vision, text, retired)

    assert catalog.route(needs_vision=True, needs_tools=True, min_context=200_000) == [vision]
    assert retired not in catalog.route(include_deprecated=False)
    assert retired in catalog.route(include_deprecated=True)
    assert catalog.route(provider_name="ANTHROPIC") == [vision, text]


def test_model_catalog_rejects_duplicate_matching_identities() -> None:
    with pytest.raises(ValidationError, match="duplicate provider/model/match"):
        _catalog(_opus(), _opus())

    with pytest.raises(ValidationError, match="duplicate provider/model/match"):
        _catalog(
            _opus(match="exact", match_prefixes=("claude-snapshot",)),
            _opus(
                model="other",
                aliases=(),
                match="exact",
                match_prefixes=("claude-snapshot",),
            ),
        )


def test_model_entities_reject_blank_members_extra_fields_and_mutation() -> None:
    with pytest.raises(ValidationError, match="family.*cannot be blank"):
        _opus(family="  ")
    with pytest.raises(ValidationError, match="aliases.*cannot be blank"):
        _opus(aliases=("  ",))
    with pytest.raises(ValidationError, match="Extra inputs"):
        ModelInfo(**_opus().model_dump(), pricing={})

    info = _opus()
    with pytest.raises(ValidationError, match="frozen"):
        info.model = "changed"


def test_default_model_catalog_is_freshly_validated_metadata_only_data() -> None:
    first = default_model_catalog()
    second = default_model_catalog()

    assert first == second
    assert first is not second
    assert first.catalog_version == first.generated_at
    assert {model.provider_name for model in first.models} == {
        "anthropic",
        "azure",
        "bedrock",
        "google",
        "openai",
        "vertex",
    }
    assert all("pricing" not in model.model_dump() for model in first.models)
    assert all(model.provenance.source == "official" for model in first.models)
    assert all(model.provenance.url.startswith("https://") for model in first.models)


def test_bundled_model_alias_and_price_identity_resolve_independently() -> None:
    models = default_model_catalog()
    prices = default_price_book()

    info = models.resolve(provider_name="openai", model="gpt-5.6")
    summary = estimate_session_cost(
        session_id="session-1",
        events=[
            _completed(
                provider_name="openai",
                model="gpt-5.6",
                input_tokens=1_000_000,
            )
        ],
        pricing=prices,
    )

    assert info is not None
    assert info.model == "gpt-5.6-sol"
    assert summary.priced_model_steps == 1
    assert summary.line_items[0].pricing_provenance is not None
    assert models.resolve(provider_name="openai", model="gpt-5.60") is None


def test_model_catalog_dump_is_deterministic_and_round_trips(tmp_path) -> None:
    first = _opus(model="a", aliases=())
    second = _opus(model="b", aliases=())
    unsorted = _catalog(second, first)
    sorted_catalog = _catalog(first, second)

    assert dump_model_catalog(unsorted) == dump_model_catalog(sorted_catalog)
    assert dump_model_catalog(unsorted).endswith("\n")
    path = tmp_path / "models.json"
    path.write_text(dump_model_catalog(unsorted))
    assert load_model_catalog(path) == sorted_catalog


def test_hosted_web_search_is_optional_strict_and_round_trips(tmp_path) -> None:
    legacy = _opus().model_dump(mode="json")
    del legacy["hosted_web_search"]
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(
        ModelCatalog.model_validate(
            {"catalog_version": "old", "generated_at": "2026-07-14", "models": [legacy]}
        ).model_dump_json()
    )

    # Catalogs written before the field existed load as "not established".
    assert ModelInfo.model_validate(legacy).hosted_web_search is False
    assert load_model_catalog(legacy_path).models[0].hosted_web_search is False

    supported = _catalog(_opus(hosted_web_search=True))
    path = tmp_path / "models.json"
    path.write_text(dump_model_catalog(supported))
    assert '"hosted_web_search": true' in path.read_text()
    assert load_model_catalog(path) == supported

    for invalid in ("yes", 1, None):
        with pytest.raises(ValidationError, match="hosted_web_search"):
            _opus(hosted_web_search=invalid)


def test_bundled_hosted_web_search_facts_are_openai_only() -> None:
    catalog = default_model_catalog()
    supported = {
        (model.provider_name, model.model) for model in catalog.models if model.hosted_web_search
    }

    assert {provider for provider, _ in supported} == {"openai"}
    assert {
        ("openai", "gpt-5.6-luna"),
        ("openai", "gpt-5.6-sol"),
        ("openai", "gpt-5.6-terra"),
    } <= supported
    alias = catalog.resolve(provider_name="openai", model="gpt-5.6")
    assert alias is not None and alias.hosted_web_search is True


def test_fixed_model_price_is_an_indefinite_application_owned_schedule() -> None:
    price = ModelPrice.fixed(
        provider_name="gateway",
        model="frontier",
        input_per_million=Decimal("1"),
        output_per_million=Decimal("2"),
    )

    assert price.schedules[0].effective_from is None
    assert price.schedules[0].effective_through is None
    assert price.schedules[0].provenance.source == "application"
    assert price.match == "prefix"


def test_price_book_uses_the_most_specific_prefix() -> None:
    prices = PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="openai",
                model="gpt-5",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("2"),
            ),
            ModelPrice.fixed(
                provider_name="openai",
                model="gpt-5.5",
                input_per_million=Decimal("3"),
                output_per_million=Decimal("4"),
            ),
        )
    )
    summary = estimate_session_cost(
        session_id="session-1",
        events=[
            _completed(provider_name="openai", model="gpt-5.5-snapshot", input_tokens=1_000_000)
        ],
        pricing=prices,
    )

    assert summary.total_cost == Decimal("3")
    assert summary.line_items[0].pricing_model == "gpt-5.5"


def test_exact_price_does_not_match_a_suffixed_runtime_model() -> None:
    prices = PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="openai",
                model="gpt-5.5",
                match="exact",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("2"),
            ),
        )
    )
    summary = estimate_session_cost(
        session_id="session-1",
        events=[_completed(provider_name="openai", model="gpt-5.5-snapshot", input_tokens=1)],
        pricing=prices,
    )

    assert summary.unpriced_model_steps == 1
    assert summary.line_items[0].missing_pricing_reason == "no matching model pricing"


def test_provider_resolved_unknown_model_does_not_fall_back_to_requested_price() -> None:
    prices = PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="openai",
                model="gpt-5.5",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("2"),
            ),
        )
    )
    summary = estimate_session_cost(
        session_id="session-1",
        events=[
            _completed(
                provider_name="openai",
                requested_model="gpt-5.5",
                model="future-expensive-model",
                input_tokens=1_000,
            )
        ],
        pricing=prices,
    )

    assert summary.priced_model_steps == 0
    assert summary.unpriced_model_steps == 1
    assert summary.line_items[0].model == "future-expensive-model"
    assert summary.line_items[0].requested_model == "gpt-5.5"
    assert summary.line_items[0].missing_pricing_reason == "no matching model pricing"


def test_price_book_selects_context_tiers_cache_rates_and_provenance() -> None:
    prices = _tiered_price_book()
    summary = estimate_session_cost(
        session_id="session-1",
        events=[
            _completed(input_tokens=100_000),
            _completed(input_tokens=300_000),
            _completed(
                input_tokens=300_000,
                cache_read_input_tokens=100_000,
                cache_write_input_tokens=100_000,
            ),
        ],
        pricing=prices,
    )

    assert [item.total_cost for item in summary.line_items] == [
        Decimal("0.5"),
        Decimal("3"),
        Decimal("1.725"),
    ]
    assert summary.line_items[1].pricing_tier_max_input_tokens is None
    assert summary.line_items[1].pricing_provenance == _PROVENANCE


def test_tiered_pricing_rejects_malformed_tiers() -> None:
    with pytest.raises(ValidationError, match="open-ended pricing tier must be last"):
        TieredPricing(
            standard=(
                PriceTier(max_input_tokens=None, input_per_million=1, output_per_million=1),
                PriceTier(max_input_tokens=100, input_per_million=1, output_per_million=1),
            )
        )
    with pytest.raises(ValidationError, match="standard pricing tiers must be ascending"):
        TieredPricing(
            standard=(
                PriceTier(max_input_tokens=200, input_per_million=1, output_per_million=1),
                PriceTier(max_input_tokens=100, input_per_million=1, output_per_million=1),
            )
        )


def test_price_book_rejects_duplicate_matching_identities() -> None:
    price = ModelPrice.fixed(
        provider_name="gateway",
        model="frontier",
        input_per_million=Decimal("1"),
        output_per_million=Decimal("2"),
    )
    with pytest.raises(ValidationError, match="duplicate provider/model/match"):
        PriceBook(prices=(price, price))


def test_price_book_dump_is_deterministic_and_round_trips(tmp_path) -> None:
    first = ModelPrice.fixed(
        provider_name="gateway",
        model="a",
        input_per_million=Decimal("1"),
        output_per_million=Decimal("2"),
    )
    second = first.model_copy(update={"model": "b"})
    unsorted = PriceBook(prices=(second, first))
    sorted_book = PriceBook(prices=(first, second))

    assert dump_price_book(unsorted) == dump_price_book(sorted_book)
    assert dump_price_book(unsorted).endswith("\n")
    path = tmp_path / "prices.json"
    path.write_text(dump_price_book(unsorted))
    assert load_price_book(path) == sorted_book


@pytest.mark.parametrize(
    ("provider", "model", "when", "tokens", "cached", "written", "output", "expected"),
    [
        ("openai", "gpt-6-astra", "2026-09-24", 272000, 0, 0, 1000, "2.77"),
        ("openai", "gpt-6-astra", "2026-09-24", 272001, 0, 0, 1000, "5.51502"),
        ("openai", "gpt-6-sol", "2026-09-24", 272000, 1000, 1000, 1000, "0.5527"),
        ("openai", "gpt-6-sol", "2026-09-24", 272001, 1000, 1000, 1000, "1.100404"),
        ("openai", "gpt-6-luna", "2026-09-24", 100000, 0, 0, 1000, "0.0105"),
        ("anthropic", "claude-fable-5-1", "2026-09-24", 100000, 10000, 0, 1000, "0.9525"),
        ("anthropic", "claude-opus-5-5", "2026-09-24", 100000, 10000, 0, 1000, "0.382"),
        ("google", "gemini-3.5-flash-lite", "2026-09-24", 100000, 0, 0, 1000, "0.0325"),
        ("google", "gemini-3.8-flash", "2026-12-31", 100000, 0, 0, 1000, "0.07875"),
        ("google", "gemini-3.8-flash", "2027-01-01", 100000, 0, 0, 1000, "0.1575"),
    ],
)
def test_september_catalog_prices_new_models_at_context_and_date_boundaries(
    provider, model, when, tokens, cached, written, output, expected
):
    info = default_model_catalog().resolve(provider_name=provider, model=model)
    assert info is not None and info.tool_calling
    event = _completed(
        provider_name=provider,
        model=model,
        input_tokens=tokens,
        cache_read_input_tokens=cached,
        cache_write_input_tokens=written,
        output_tokens=output,
    ).model_copy(update={"timestamp": datetime.fromisoformat(when).replace(tzinfo=UTC)})
    summary = estimate_session_cost(
        session_id="session-1", events=[event], pricing=default_price_book()
    )
    assert summary.priced_model_steps == 1
    assert summary.line_items[0].total_cost == Decimal(expected)


@pytest.mark.parametrize(
    ("model", "expected"),
    [("gpt-6-astra", "0.025"), ("gpt-6-sol", "0.013"), ("gpt-6-luna", "0.01015")],
)
def test_gpt6_hosted_search_prices_tokens_and_search_calls(model, expected):
    event = _completed(provider_name="openai", model=model, input_tokens=1000, output_tokens=100)
    payload = dict(event.payload)
    payload["usage_metrics"] = {
        **payload["usage_metrics"],
        "hosted_tools": {"web_search_calls": 1},
    }
    event = event.model_copy(
        update={"timestamp": datetime(2026, 9, 24, tzinfo=UTC), "payload": payload}
    )
    summary = estimate_session_cost(
        session_id="session-1", events=[event], pricing=default_price_book()
    )
    assert summary.priced_model_steps == 1
    assert summary.unpriced_model_steps == 0
    assert summary.line_items[0].web_search_cost == Decimal("0.01")
    assert summary.total_cost == Decimal(expected)
