from decimal import Decimal

import pytest

import cayu.budgets.base as budget_base
from cayu.budgets import BudgetLimit, default_price_book
from cayu.budgets.base import (
    _budget_limit_definition_digest,
    _computed_budget_limit_definition_digest,
    _effective_budget_limits,
)
from cayu.budgets.pricing import ContextualPricingRequirement, ModelPrice, PriceBook


def _small_book(version: str = "copies-1") -> PriceBook:
    return PriceBook(
        price_book_version=version,
        prices=(
            ModelPrice.fixed(
                provider_name="openai",
                model="copy-test-model",
                match="exact",
                input_per_million=Decimal("1.50"),
                output_per_million=Decimal("6"),
            ),
        ),
    )


def _contextual_book() -> PriceBook:
    return PriceBook(
        price_book_version="copies-contextual",
        contextual_pricing_requirements=(
            ContextualPricingRequirement(provider_name="fake", dimensions=("tier",)),
        ),
        prices=(
            ModelPrice.fixed(
                provider_name="fake",
                model="fake-model",
                match="exact",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("2"),
                cache_read_input_per_million=Decimal("0.5"),
                pricing_context={"tier": ("a", "b")},
            ),
        ),
    )


def _unvalidated_twin(book: PriceBook, corruption: str) -> PriceBook:
    """The book rebuilt with model_copy(update=...), which skips validation."""

    (price,) = book.prices
    if corruption == "dimension string":
        selector = price.pricing_context.model_copy(update={"dimensions": {"tier": "ab"}})
        price = price.model_copy(update={"pricing_context": selector})
    else:
        (schedule,) = price.schedules
        (tier,) = schedule.pricing.standard
        tier = tier.model_copy(update={"cache_read_input_per_million": Decimal("NaN")})
        pricing = schedule.pricing.model_copy(update={"standard": (tier,)})
        price = price.model_copy(
            update={"schedules": (schedule.model_copy(update={"pricing": pricing}),)}
        )
    return book.model_copy(update={"prices": (price,)})


@pytest.mark.parametrize("pricing", [None, _small_book()])
def test_cached_definition_digest_matches_the_computed_one(pricing) -> None:
    limit = BudgetLimit.usd(Decimal("5"), pricing=pricing)

    first = _budget_limit_definition_digest(limit)

    assert first == _computed_budget_limit_definition_digest(limit)[0]
    assert _budget_limit_definition_digest(limit) == first
    respelled = BudgetLimit.usd(Decimal("5.000"), pricing=pricing)
    assert _budget_limit_definition_digest(respelled) == first


def test_a_limit_changed_after_its_digest_was_cached_gets_its_own_digest() -> None:
    limit = BudgetLimit.usd(Decimal("5"), pricing=_small_book())
    before = _budget_limit_definition_digest(limit)

    limit.pricing.price_book_version = "copies-2"

    after = _budget_limit_definition_digest(limit)
    assert after != before
    assert after == _computed_budget_limit_definition_digest(limit)[0]


def test_an_effective_limit_with_a_corrupted_price_book_is_still_rejected() -> None:
    (effective,) = _effective_budget_limits(
        (BudgetLimit.usd(Decimal("5"), pricing=_small_book()),),
        identity_namespace="request",
    )

    effective.pricing.prices = ()

    with pytest.raises(ValueError):
        _effective_budget_limits(
            (effective,), identity_namespace="request", preserve_effective=True
        )


def test_price_book_copies_stay_detached_in_both_directions() -> None:
    book = _small_book()
    first = BudgetLimit.usd(Decimal("5"), pricing=book)

    book.price_book_version = "changed-by-caller"
    first.pricing.price_book_version = "changed-through-limit"
    second = BudgetLimit.usd(Decimal("5"), pricing=_small_book())

    assert first.pricing is not book
    assert second.pricing.price_book_version == "copies-1"
    assert second.pricing is not first.pricing


def test_copying_a_limit_revalidates_its_price_book_once(monkeypatch) -> None:
    calls = []
    revalidate = budget_base.revalidate_model_input

    def counted(value, *model_types):
        if model_types == (PriceBook,):
            calls.append(value)
        return revalidate(value, *model_types)

    monkeypatch.setattr(budget_base, "revalidate_model_input", counted)
    book = default_price_book().with_prices(
        ModelPrice.fixed(
            provider_name="openai",
            model="copy-count-model",
            match="exact",
            input_per_million=Decimal("2"),
            output_per_million=Decimal("8"),
        ),
        price_book_version="copy-count",
    )
    limits = (
        BudgetLimit.usd(Decimal("5"), pricing=book),
        BudgetLimit.usd(Decimal("60"), scope="causal", key="run-1", pricing=book),
    )

    for _ in range(10):
        effective = _effective_budget_limits(limits, identity_namespace="request")
        _effective_budget_limits(effective, identity_namespace="request", preserve_effective=True)

    assert len(calls) == 1


@pytest.mark.parametrize("corruption", ["dimension string", "nan rate"])
def test_an_unvalidated_book_is_rejected_after_its_valid_twin_was_copied(corruption) -> None:
    valid = _contextual_book()
    BudgetLimit.usd(Decimal("5"), pricing=valid)
    twin = _unvalidated_twin(valid, corruption)

    with pytest.raises(ValueError):
        BudgetLimit.usd(Decimal("5"), pricing=twin)


@pytest.mark.parametrize("corruption", ["dimension string", "nan rate"])
def test_an_effective_limit_given_an_unvalidated_book_fails_its_identity_check(corruption) -> None:
    valid = _contextual_book()
    (effective,) = _effective_budget_limits(
        (BudgetLimit.usd(Decimal("5"), pricing=valid),),
        identity_namespace="request",
    )
    _effective_budget_limits((effective,), identity_namespace="request", preserve_effective=True)

    effective.pricing.prices = _unvalidated_twin(valid, corruption).prices

    with pytest.raises(ValueError):
        _effective_budget_limits(
            (effective,), identity_namespace="request", preserve_effective=True
        )
