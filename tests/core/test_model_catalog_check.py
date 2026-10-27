"""The deterministic release policy for the bundled model catalog and price book."""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import pytest

from cayu.budgets.pricing import (
    ModelInfo,
    ModelPrice,
    default_model_catalog,
    default_price_book,
)
from maintenance.model_catalog.policy import (
    CATALOG_MAX_AGE_DAYS,
    VERIFY_MAX_AGE_DAYS,
    VERIFY_SCHEDULE_DAYS,
    model_policy_errors,
    price_policy_errors,
    suspicious_price_changes,
    validate_catalog,
    validate_price_book,
    validate_resource_pair,
)


def _provider_model(provider_name: str) -> ModelInfo:
    return next(
        model for model in default_model_catalog().models if model.provider_name == provider_name
    )


def _provider_price(provider_name: str) -> ModelPrice:
    model = _provider_model(provider_name)
    return next(
        price
        for price in default_price_book().prices
        if price.provider_name == model.provider_name and price.model == model.model
    )


def test_bedrock_opus_5_release_date_matches_aws_model_card() -> None:
    model = default_model_catalog().match(
        provider_name="bedrock", model="global.anthropic.claude-opus-5"
    )

    assert model is not None
    assert model.release_date == date(2026, 7, 24)


@pytest.mark.parametrize(
    ("model_name", "release_date"),
    [
        ("claude-fable-5", date(2026, 6, 9)),
        ("claude-opus-4-8", date(2026, 5, 28)),
        ("claude-sonnet-4-6", date(2026, 2, 17)),
        ("claude-sonnet-5", date(2026, 6, 30)),
    ],
)
def test_vertex_claude_facts_match_google_model_cards(model_name: str, release_date: date) -> None:
    model = default_model_catalog().match(provider_name="vertex", model=model_name)

    assert model is not None
    assert model.context_window == 1_000_000
    assert model.max_output_tokens == 128_000
    assert model.release_date == release_date
    assert model.provenance.url == (
        "https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/"
        f"partner-models/claude/{model_name.removeprefix('claude-')}"
    )


@pytest.mark.parametrize(
    ("model", "effective_from"),
    [
        ("gpt-5.4-mini", date(2026, 3, 1)),
        ("gpt-5.4-nano", date(2026, 3, 1)),
        ("gpt-5.5", date(2026, 5, 1)),
        ("gpt-5.6-luna", date(2026, 7, 1)),
        ("gpt-5.6-sol", date(2026, 7, 1)),
        ("gpt-5.6-terra", date(2026, 7, 1)),
    ],
)
def test_azure_prices_start_on_their_retail_meter_effective_date(
    model: str, effective_from: date
) -> None:
    price = next(
        item
        for item in default_price_book().prices
        if item.provider_name == "azure" and item.model == model
    )

    assert price.schedule_on(effective_from - timedelta(days=1)) is None
    schedule = price.schedule_on(effective_from)
    assert schedule is not None
    assert schedule.effective_from == effective_from


@pytest.mark.parametrize(
    ("model", "input_price", "output_price", "cache_read_price"),
    [
        ("gpt-5.4-mini", "0.375", "2.25", "0.0375"),
        ("gpt-5.4-nano", "0.1", "0.625", "0.01"),
    ],
)
def test_azure_gpt_5_4_prices_include_published_batch_rates(
    model: str,
    input_price: str,
    output_price: str,
    cache_read_price: str,
) -> None:
    price = next(
        item
        for item in default_price_book().prices
        if item.provider_name == "azure" and item.model == model
    )
    schedule = price.schedule_on(date(2026, 8, 3))

    assert schedule is not None
    batch = schedule.pricing.batch
    assert batch is not None
    assert batch.input_per_million == Decimal(input_price)
    assert batch.output_per_million == Decimal(output_price)
    assert batch.cache_read_input_per_million == Decimal(cache_read_price)


_VERIFICATION_DATE = date(2026, 7, 13)


def _pricing(price: ModelPrice):
    schedule = price.schedule_on(_VERIFICATION_DATE)
    assert schedule is not None
    return schedule.pricing


def _with_pricing(price: ModelPrice, pricing) -> ModelPrice:
    active = price.schedule_on(_VERIFICATION_DATE)
    assert active is not None
    schedules = tuple(
        schedule.model_copy(update={"pricing": pricing}) if schedule is active else schedule
        for schedule in price.schedules
    )
    return price.model_copy(update={"schedules": schedules})


def test_default_catalog_satisfies_release_policy() -> None:
    catalog = default_model_catalog()
    price_book = default_price_book()
    validate_catalog(catalog, today=date.fromisoformat(catalog.generated_at))
    validate_price_book(price_book, today=date.fromisoformat(price_book.generated_at))
    validate_resource_pair(catalog, price_book)


def test_resource_pair_rejects_undocumented_price_only_identity() -> None:
    catalog = default_model_catalog()
    price_book = default_price_book()
    gateway_price = ModelPrice.fixed(
        provider_name="openai",
        model="gateway-billing-only",
        input_per_million=Decimal("1"),
        output_per_million=Decimal("2"),
    )
    candidate = price_book.model_copy(update={"prices": (*price_book.prices, gateway_price)})

    with pytest.raises(ValueError, match="undocumented bundled price-only identities"):
        validate_resource_pair(catalog, candidate)

    validate_resource_pair(
        catalog,
        candidate,
        allowed_price_only_identities={("openai", "gateway-billing-only")},
    )

    missing_canonical_price = price_book.model_copy(update={"prices": price_book.prices[1:]})
    with pytest.raises(ValueError, match="bundled model identities without canonical prices"):
        validate_resource_pair(catalog, missing_canonical_price)


def test_release_policy_rejects_stale_provenance() -> None:
    catalog = default_model_catalog()
    oldest = min(date.fromisoformat(model.provenance.as_of) for model in catalog.models)
    stale_day = max(
        date.fromisoformat(catalog.generated_at),
        oldest + timedelta(days=31),
    )
    with pytest.raises(ValueError, match="provenance is stale"):
        validate_catalog(catalog, today=stale_day)


def test_release_policy_rejects_non_official_provider_host() -> None:
    catalog = default_model_catalog()
    model = catalog.models[0]
    bad_model = model.model_copy(
        update={
            "provenance": model.provenance.model_copy(update={"url": "https://example.com/pricing"})
        }
    )
    candidate = catalog.model_copy(update={"models": (bad_model, *catalog.models[1:])})

    with pytest.raises(ValueError, match="allowed official host"):
        validate_catalog(candidate, today=date.fromisoformat(catalog.generated_at))


def test_release_policy_rejects_provider_owned_but_unapproved_subdomain() -> None:
    catalog = default_model_catalog()
    index, model = next(
        (index, model)
        for index, model in enumerate(catalog.models)
        if model.provider_name == "vertex"
    )
    bad_model = model.model_copy(
        update={
            "provenance": model.provenance.model_copy(
                update={"url": "https://sites.google.com/view/model-pricing"}
            )
        }
    )
    models = list(catalog.models)
    models[index] = bad_model

    with pytest.raises(ValueError, match="allowed official host"):
        validate_catalog(
            catalog.model_copy(update={"models": tuple(models)}),
            today=date.fromisoformat(catalog.generated_at),
        )


def test_release_policy_enforces_narrow_bundled_match_rules() -> None:
    model = default_model_catalog().models[0]
    broad = model.model_copy(update={"match": "prefix"})
    unrelated = model.model_copy(update={"match_prefixes": ("other-model-20",)})
    unsafe_delimiter = model.model_copy(update={"match_prefixes": (f"{model.model}x20",)})

    assert any(
        "canonical model match must be 'exact'" in error
        for error in model_policy_errors(broad, today=date(2026, 7, 14), max_age_days=None)
    )
    for candidate in (unrelated, unsafe_delimiter):
        assert any(
            "must strictly extend the canonical model" in error
            for error in model_policy_errors(candidate, today=date(2026, 7, 14), max_age_days=None)
        )


def test_verification_schedule_leaves_release_freshness_margin() -> None:
    assert VERIFY_MAX_AGE_DAYS == 14
    assert VERIFY_MAX_AGE_DAYS + VERIFY_SCHEDULE_DAYS < CATALOG_MAX_AGE_DAYS


def test_price_policy_freshness_applies_only_to_the_active_schedule() -> None:
    price = next(
        item
        for item in default_price_book().prices
        if item.provider_name == "anthropic" and item.model == "claude-sonnet-5"
    )
    # Exercise a fixed transition without depending on a provider's current
    # promotional schedule (the announced Sonnet 5 increase was cancelled).
    current = price.schedules[0].model_copy(
        update={"effective_from": None, "effective_through": date(2026, 8, 31)}
    )
    future = current.model_copy(
        update={"effective_from": date(2026, 9, 1), "effective_through": None}
    )
    candidate = price.model_copy(
        update={
            "schedules": (
                current.model_copy(
                    update={
                        "provenance": current.provenance.model_copy(update={"as_of": "2026-08-01"})
                    }
                ),
                future.model_copy(
                    update={
                        "provenance": future.provenance.model_copy(update={"as_of": "2026-07-01"})
                    }
                ),
            )
        }
    )

    before_transition = price_policy_errors(
        candidate,
        today=date(2026, 8, 13),
        max_age_days=30,
    )
    after_transition = price_policy_errors(
        candidate,
        today=date(2026, 9, 1),
        max_age_days=30,
    )

    assert not any("provenance is stale" in error for error in before_transition)
    assert any("schedule[1]: provenance is stale" in error for error in after_transition)


def test_release_policy_rejects_zero_and_implausibly_large_prices() -> None:
    price_book = default_price_book()
    original_price = price_book.prices[0]
    for price, message in (("0", "greater than zero"), ("1001", "exceeds 1000")):
        original_pricing = _pricing(original_price)
        tier = original_pricing.base().model_copy(update={"input_per_million": Decimal(price)})
        changed = _with_pricing(
            original_price,
            original_pricing.model_copy(update={"standard": (tier,)}),
        )
        candidate = price_book.model_copy(update={"prices": (changed, *price_book.prices[1:])})
        with pytest.raises(ValueError, match=message):
            validate_price_book(
                candidate,
                today=date.fromisoformat(price_book.generated_at),
            )


def test_automated_price_changes_flag_large_deltas() -> None:
    model = default_model_catalog().models[0]
    price = default_price_book().prices[0]
    pricing = _pricing(price)
    tier = pricing.base().model_copy(
        update={"input_per_million": pricing.base().input_per_million * 5}
    )
    changed_price = _with_pricing(
        price,
        pricing.model_copy(update={"standard": (tier,)}),
    )

    warnings = suspicious_price_changes(model, model, price, changed_price)

    assert any("schedule[0].standard[0].input changed" in warning for warning in warnings)


def test_catalog_policy_bounds_context_window_updates() -> None:
    model = default_model_catalog().models[0]
    too_large = model.model_copy(update={"context_window": 10_000_001})
    wrong_type = model.model_copy(update={"context_window": True})
    nonpositive = model.model_copy(update={"context_window": 0})

    assert any(
        "context_window exceeds 10000000" in error
        for error in model_policy_errors(too_large, today=date(2026, 7, 13), max_age_days=None)
    )
    assert any(
        "context_window must be a positive integer" in error
        for error in model_policy_errors(wrong_type, today=date(2026, 7, 13), max_age_days=None)
    )
    assert any(
        "context_window must be a positive integer" in error
        for error in model_policy_errors(nonpositive, today=date(2026, 7, 13), max_age_days=None)
    )
    price = default_price_book().prices[0]
    assert suspicious_price_changes(model, nonpositive, price, price) == []


def test_automated_context_window_changes_flag_large_deltas() -> None:
    model = default_model_catalog().models[0].model_copy(update={"context_window": 100_000})
    changed = model.model_copy(update={"context_window": 500_000})

    price = default_price_book().prices[0]
    warnings = suspicious_price_changes(model, changed, price, price)

    assert "context_window changed from 100000 to 500000" in warnings


def test_automated_hosted_web_search_removal_requires_review() -> None:
    model = _provider_model("openai").model_copy(update={"hosted_web_search": True})
    removed = model.model_copy(update={"hosted_web_search": False})
    price = _provider_price("openai")

    assert "hosted_web_search removed: True -> False" in suspicious_price_changes(
        model, removed, price, price
    )
    assert suspicious_price_changes(removed, model, price, price) == []
