from decimal import Decimal

from examples._advanced_support import paired_cost_evidence

from cayu import (
    Event,
    EventType,
    ModelPrice,
    PriceBook,
    PriceSchedule,
    PriceTier,
    Provenance,
    SessionCostSummary,
    TieredPricing,
    estimate_session_cost,
)


def _cost(
    *,
    session_id: str,
    input_tokens: int,
    price_book: PriceBook,
    currency: str = "USD",
):
    event = Event(
        type=EventType.MODEL_COMPLETED,
        session_id=session_id,
        payload={
            "usage_metrics": {
                "provider_name": "provider",
                "model": "model",
                "input_tokens": input_tokens,
                "output_tokens": 0,
                "total_tokens": input_tokens,
                "reasoning_output_tokens": 0,
                "cache": {
                    "read_tokens": 0,
                    "write_tokens": 0,
                    "cached_input_tokens": 0,
                    "uncached_input_tokens": input_tokens,
                },
            }
        },
    )
    return estimate_session_cost(
        session_id=session_id,
        events=[event],
        pricing=price_book,
        currency=currency,
    )


def _price_book(
    *,
    provider_name: str = "provider",
    model: str = "model",
    price_book_version: str = "tiered-v1",
) -> PriceBook:
    return PriceBook(
        price_book_version=price_book_version,
        generated_at="2026-07-13T00:00:00Z",
        prices=(
            ModelPrice(
                provider_name=provider_name,
                model=model,
                match="exact",
                schedules=(
                    PriceSchedule(
                        pricing=TieredPricing(
                            standard=(
                                PriceTier(
                                    max_input_tokens=100,
                                    input_per_million=Decimal("1"),
                                    output_per_million=Decimal("5"),
                                ),
                                PriceTier(
                                    input_per_million=Decimal("2"),
                                    output_per_million=Decimal("10"),
                                ),
                            )
                        ),
                        provenance=Provenance(
                            source="test fixture",
                            url="https://example.invalid/pricing",
                            as_of="2026-07-13",
                        ),
                    ),
                ),
            ),
        ),
    )


def test_paired_cost_evidence_fails_closed_when_pair_is_missing() -> None:
    evidence = paired_cost_evidence(
        candidate=None,
        baseline=(),
        price_book=_price_book(),
        baseline_cost_field="paired_baseline_cost",
    )

    assert evidence == {
        "status": "unavailable",
        "reason": "paired completion usage is missing",
    }


def test_paired_cost_evidence_fails_closed_without_catalog() -> None:
    evidence = paired_cost_evidence(
        candidate=(),
        baseline=(),
        price_book=None,
        baseline_cost_field="paired_baseline_cost",
    )

    assert evidence == {
        "status": "unpriced",
        "reason": "no caller-supplied price book",
        "candidate_cost": None,
        "paired_baseline_cost": None,
        "savings": None,
        "savings_percent": None,
    }


def test_paired_cost_evidence_fails_closed_for_empty_summaries() -> None:
    evidence = paired_cost_evidence(
        candidate=(),
        baseline=(),
        price_book=_price_book(),
        baseline_cost_field="paired_baseline_cost",
    )

    assert evidence == {
        "status": "unavailable",
        "reason": "paired cost summaries are empty",
    }


def test_paired_cost_evidence_fails_closed_for_unpriced_attempt() -> None:
    price_book = _price_book()
    unpriced_price_book = _price_book(
        provider_name="other-provider",
        model="other-model",
        price_book_version="empty-v1",
    )
    evidence = paired_cost_evidence(
        candidate=(_cost(session_id="candidate", input_tokens=50, price_book=unpriced_price_book),),
        baseline=(_cost(session_id="baseline", input_tokens=50, price_book=price_book),),
        price_book=price_book,
        baseline_cost_field="paired_baseline_cost",
    )

    assert evidence == {
        "status": "unpriced",
        "reason": "the caller-supplied price book did not price every paired attempt",
        "candidate_cost": None,
        "paired_baseline_cost": None,
        "savings": None,
        "savings_percent": None,
        "price_book_version": "tiered-v1",
        "price_book_generated_at": "2026-07-13T00:00:00Z",
    }


def test_paired_cost_evidence_fails_closed_for_different_currencies() -> None:
    price_book = _price_book()
    evidence = paired_cost_evidence(
        candidate=(
            SessionCostSummary(
                session_id="candidate",
                currency="USD",
                model_steps=0,
                priced_model_steps=0,
                unpriced_model_steps=0,
                total_cost=Decimal("0"),
            ),
        ),
        baseline=(
            SessionCostSummary(
                session_id="baseline",
                currency="EUR",
                model_steps=0,
                priced_model_steps=0,
                unpriced_model_steps=0,
                total_cost=Decimal("0"),
            ),
        ),
        price_book=price_book,
        baseline_cost_field="paired_baseline_cost",
    )

    assert evidence == {
        "status": "unavailable",
        "reason": "paired costs use different currencies",
    }


def test_paired_cost_evidence_fails_closed_when_attempts_use_different_pricing_tiers() -> None:
    price_book = _price_book()

    evidence = paired_cost_evidence(
        candidate=(_cost(session_id="candidate", input_tokens=50, price_book=price_book),),
        baseline=(_cost(session_id="baseline", input_tokens=150, price_book=price_book),),
        price_book=price_book,
        baseline_cost_field="paired_baseline_cost",
    )

    assert evidence == {
        "status": "unavailable",
        "reason": "paired attempts resolved different pricing rows or tiers",
        "price_book_version": "tiered-v1",
        "price_book_generated_at": "2026-07-13T00:00:00Z",
    }
