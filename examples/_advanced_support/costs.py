from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from typing import Any, Literal

from cayu import PriceBook, SessionCostSummary


def paired_cost_evidence(
    *,
    candidate: Sequence[SessionCostSummary] | None,
    baseline: Sequence[SessionCostSummary] | None,
    price_book: PriceBook | None,
    baseline_cost_field: Literal["bounded_baseline_cost", "paired_baseline_cost"],
) -> dict[str, Any]:
    """Build one fail-closed, provenance-bearing paired cost envelope."""
    if candidate is None or baseline is None:
        return {"status": "unavailable", "reason": "paired completion usage is missing"}
    if price_book is None:
        return _unpriced(
            reason="no caller-supplied price book",
            baseline_cost_field=baseline_cost_field,
        )
    if not candidate or not baseline:
        return {"status": "unavailable", "reason": "paired cost summaries are empty"}
    summaries = [*candidate, *baseline]
    if any(summary.unpriced_model_steps for summary in summaries):
        return _unpriced(
            reason="the caller-supplied price book did not price every paired attempt",
            baseline_cost_field=baseline_cost_field,
            price_book=price_book,
        )

    currencies = {summary.currency for summary in summaries}
    if len(currencies) != 1:
        return {"status": "unavailable", "reason": "paired costs use different currencies"}
    line_items = [item for summary in summaries for item in summary.line_items]
    pricing_resolutions = {
        (
            item.pricing_provider_name,
            item.pricing_model,
            item.pricing_match,
            item.pricing_tier_max_input_tokens,
        )
        for item in line_items
        if item.priced
    }
    if len(pricing_resolutions) != 1:
        return {
            "status": "unavailable",
            "reason": "paired attempts resolved different pricing rows or tiers",
            "price_book_version": price_book.price_book_version,
            "price_book_generated_at": price_book.generated_at,
        }
    provider_name, model, match, tier_max_input_tokens = pricing_resolutions.pop()
    pricing_provenances = {
        item.pricing_provenance.model_dump_json() if item.pricing_provenance is not None else None
        for item in line_items
        if item.priced
    }
    if len(pricing_provenances) != 1 or None in pricing_provenances:
        return _unpriced(
            reason="paired pricing provenance could not be resolved",
            baseline_cost_field=baseline_cost_field,
            price_book=price_book,
        )

    candidate_total = sum((summary.total_cost for summary in candidate), Decimal("0"))
    baseline_total = sum((summary.total_cost for summary in baseline), Decimal("0"))
    savings = baseline_total - candidate_total
    savings_percent = (
        None
        if baseline_total == 0
        else (savings * Decimal("100") / baseline_total).quantize(Decimal("0.01"))
    )
    return {
        "status": "priced",
        "currency": currencies.pop(),
        "candidate_cost": str(candidate_total),
        baseline_cost_field: str(baseline_total),
        "savings": str(savings),
        "savings_percent": None if savings_percent is None else str(savings_percent),
        "price_book_version": price_book.price_book_version,
        "price_book_generated_at": price_book.generated_at,
        "pricing_provider_name": provider_name,
        "pricing_model": model,
        "pricing_match": match,
        "pricing_tier_max_input_tokens": tier_max_input_tokens,
        "pricing_provenance": next(
            item.pricing_provenance.model_dump(mode="json")
            for item in line_items
            if item.pricing_provenance is not None
        ),
    }


def _unpriced(
    *,
    reason: str,
    baseline_cost_field: str,
    price_book: PriceBook | None = None,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "status": "unpriced",
        "reason": reason,
        "candidate_cost": None,
        baseline_cost_field: None,
        "savings": None,
        "savings_percent": None,
    }
    if price_book is not None:
        evidence["price_book_version"] = price_book.price_book_version
        evidence["price_book_generated_at"] = price_book.generated_at
    return evidence
