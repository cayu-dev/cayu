from decimal import Decimal

from examples._advanced_support import paired_cost_evidence

from cayu import (
    Event,
    EventType,
    ModelCatalog,
    ModelInfo,
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
    catalog: ModelCatalog,
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
        catalog=catalog,
        currency=currency,
    )


def _catalog(
    *,
    provider_name: str = "provider",
    model: str = "model",
    catalog_version: str = "tiered-v1",
) -> ModelCatalog:
    return ModelCatalog(
        catalog_version=catalog_version,
        generated_at="2026-07-13T00:00:00Z",
        models=(
            ModelInfo(
                provider_name=provider_name,
                model=model,
                match="exact",
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
    )


def test_paired_cost_evidence_fails_closed_when_pair_is_missing() -> None:
    evidence = paired_cost_evidence(
        candidate=None,
        baseline=(),
        catalog=_catalog(),
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
        catalog=None,
        baseline_cost_field="paired_baseline_cost",
    )

    assert evidence == {
        "status": "unpriced",
        "reason": "no caller-supplied model catalog",
        "candidate_cost": None,
        "paired_baseline_cost": None,
        "savings": None,
        "savings_percent": None,
    }


def test_paired_cost_evidence_fails_closed_for_empty_summaries() -> None:
    evidence = paired_cost_evidence(
        candidate=(),
        baseline=(),
        catalog=_catalog(),
        baseline_cost_field="paired_baseline_cost",
    )

    assert evidence == {
        "status": "unavailable",
        "reason": "paired cost summaries are empty",
    }


def test_paired_cost_evidence_fails_closed_for_unpriced_attempt() -> None:
    catalog = _catalog()
    unpriced_catalog = _catalog(
        provider_name="other-provider",
        model="other-model",
        catalog_version="empty-v1",
    )
    evidence = paired_cost_evidence(
        candidate=(_cost(session_id="candidate", input_tokens=50, catalog=unpriced_catalog),),
        baseline=(_cost(session_id="baseline", input_tokens=50, catalog=catalog),),
        catalog=catalog,
        baseline_cost_field="paired_baseline_cost",
    )

    assert evidence == {
        "status": "unpriced",
        "reason": "the caller-supplied catalog did not price every paired attempt",
        "candidate_cost": None,
        "paired_baseline_cost": None,
        "savings": None,
        "savings_percent": None,
        "catalog_version": "tiered-v1",
        "catalog_generated_at": "2026-07-13T00:00:00Z",
    }


def test_paired_cost_evidence_fails_closed_for_different_currencies() -> None:
    catalog = _catalog()
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
        catalog=catalog,
        baseline_cost_field="paired_baseline_cost",
    )

    assert evidence == {
        "status": "unavailable",
        "reason": "paired costs use different currencies",
    }


def test_paired_cost_evidence_fails_closed_when_attempts_use_different_pricing_tiers() -> None:
    catalog = _catalog()

    evidence = paired_cost_evidence(
        candidate=(_cost(session_id="candidate", input_tokens=50, catalog=catalog),),
        baseline=(_cost(session_id="baseline", input_tokens=150, catalog=catalog),),
        catalog=catalog,
        baseline_cost_field="paired_baseline_cost",
    )

    assert evidence == {
        "status": "unavailable",
        "reason": "paired attempts resolved different pricing rows or tiers",
        "catalog_version": "tiered-v1",
        "catalog_generated_at": "2026-07-13T00:00:00Z",
    }
