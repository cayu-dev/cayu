from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from cayu import (
    ComparableGenerationSettings,
    ComparableOutputBudget,
    ComparisonCostLineItem,
    ComparisonPricingCatalog,
    CostQualityAttemptOperation,
    Event,
    EventType,
    ModelPrice,
    PairedCostAttempt,
    PairedCostQualityComparisonRequest,
    PairedCostQualityPair,
    PairedCostQualitySide,
    PairedQualityEvidence,
    PriceBook,
    Provenance,
    compare_paired_cost_quality,
    estimate_session_cost,
)


@dataclass(frozen=True)
class ComparisonSessionEvidence:
    """Example-side attribution needed to project durable events into one report."""

    session_id: str
    events: tuple[Event, ...]
    operation: CostQualityAttemptOperation
    role: str
    branch_id: str | None = None
    source_checkpoint_id: str | None = None


_UNPRICED_PRICE_BOOK = PriceBook(
    price_book_version="cayu-unpriced-placeholder-v1",
    generated_at="not-a-pricing-catalog",
    prices=(
        ModelPrice.fixed(
            provider_name="urn:cayu:unpriced-placeholder",
            model="urn:cayu:unpriced-placeholder",
            match="exact",
            input_per_million=Decimal("0"),
            output_per_million=Decimal("0"),
            provenance=Provenance(
                source="Cayu usage-only placeholder; not pricing",
                url="https://cayu.dev/contracts/cost-quality-comparison",
                as_of="not-applicable",
            ),
        ),
    ),
)


def paired_cost_quality_report(
    *,
    pair_id: str,
    workload_id: str,
    task_id: str,
    source_id: str,
    role: str,
    output_budget: ComparableOutputBudget,
    generation_settings: ComparableGenerationSettings,
    baseline_strategy_id: str,
    candidate_strategy_id: str,
    baseline: Sequence[ComparisonSessionEvidence] | None,
    candidate: Sequence[ComparisonSessionEvidence] | None,
    baseline_quality: PairedQualityEvidence | None,
    candidate_quality: PairedQualityEvidence | None,
    price_book: PriceBook | None,
) -> dict[str, Any]:
    """Map example events to the public contract without owning any arithmetic."""

    catalog = (
        None
        if price_book is None
        else ComparisonPricingCatalog(
            price_book_version=price_book.price_book_version,
            generated_at=price_book.generated_at,
        )
    )
    pair = PairedCostQualityPair(
        pair_id=pair_id,
        baseline=_side(
            strategy_id=baseline_strategy_id,
            workload_id=workload_id,
            task_id=task_id,
            source_id=source_id,
            role=role,
            output_budget=output_budget,
            generation_settings=generation_settings,
            pricing_catalog=catalog,
            quality=baseline_quality,
            sessions=baseline,
            price_book=price_book,
        ),
        candidate=_side(
            strategy_id=candidate_strategy_id,
            workload_id=workload_id,
            task_id=task_id,
            source_id=source_id,
            role=role,
            output_budget=output_budget,
            generation_settings=generation_settings,
            pricing_catalog=catalog,
            quality=candidate_quality,
            sessions=candidate,
            price_book=price_book,
        ),
    )
    return compare_paired_cost_quality(
        PairedCostQualityComparisonRequest(pairs=(pair,))
    ).model_dump(mode="json")


def _side(
    *,
    strategy_id: str,
    workload_id: str,
    task_id: str,
    source_id: str,
    role: str,
    output_budget: ComparableOutputBudget,
    generation_settings: ComparableGenerationSettings,
    pricing_catalog: ComparisonPricingCatalog | None,
    quality: PairedQualityEvidence | None,
    sessions: Sequence[ComparisonSessionEvidence] | None,
    price_book: PriceBook | None,
) -> PairedCostQualitySide | None:
    if sessions is None:
        return None
    attempts = tuple(
        attempt for session in sessions for attempt in _attempts(session, price_book=price_book)
    )
    return PairedCostQualitySide(
        strategy_id=strategy_id,
        workload_id=workload_id,
        task_id=task_id,
        source_id=source_id,
        role=role,
        output_budget=output_budget,
        generation_settings=generation_settings,
        pricing_catalog=pricing_catalog,
        quality=quality,
        attempts=attempts,
    )


def _attempts(
    evidence: ComparisonSessionEvidence,
    *,
    price_book: PriceBook | None,
) -> tuple[PairedCostAttempt, ...]:
    summary = estimate_session_cost(
        session_id=evidence.session_id,
        events=list(evidence.events),
        pricing=price_book or _UNPRICED_PRICE_BOOK,
    )
    completed_events = tuple(
        event for event in evidence.events if event.type == EventType.MODEL_COMPLETED
    )
    completed_costs = {
        event.id: item for event, item in zip(completed_events, summary.line_items, strict=True)
    }
    relevant_types = {
        EventType.MODEL_STARTED,
        EventType.MODEL_RETRY,
        EventType.MODEL_ATTEMPT_DISCARDED,
        EventType.MODEL_COMPLETED,
    }
    records: dict[str, dict[str, Any]] = {}
    default_operation_id = f"{evidence.session_id}:{evidence.operation.value}"
    for event_index, event in enumerate(evidence.events):
        if event.type not in relevant_types:
            continue
        payload = event.payload
        model_attempt_id = _optional_text(payload.get("model_attempt_id"))
        model_step_id = _optional_text(payload.get("model_step_id"))
        step = _positive_int(payload.get("step"))
        ordinal = _positive_int(payload.get("attempt"))
        operation_id = model_step_id or (
            default_operation_id if step is None else f"{default_operation_id}:{step}"
        )
        if model_attempt_id is not None:
            key = f"attempt:{model_attempt_id}"
        elif ordinal is not None:
            key = f"operation:{operation_id}:attempt:{ordinal}"
        else:
            key = f"event:{event.id}"
        record = records.setdefault(
            key,
            {
                "event_index": event_index,
                "operation_id": operation_id,
                "model_attempt_id": model_attempt_id,
                "ordinal": ordinal,
                "provider_name": None,
                "model": None,
                "cost": None,
                "usage_unavailable_reason": (
                    "provider attempt has no model.completed usage evidence"
                ),
            },
        )
        record["provider_name"] = record["provider_name"] or _optional_text(
            payload.get("provider_name") or payload.get("provider")
        )
        record["model"] = record["model"] or _optional_text(
            payload.get("model") or payload.get("requested_model")
        )
        item = completed_costs.get(event.id)
        if item is not None:
            record["provider_name"] = item.provider_name or record["provider_name"]
            record["model"] = item.model or item.requested_model or record["model"]
            if _usage_is_missing(item.missing_pricing_reason):
                record["usage_unavailable_reason"] = item.missing_pricing_reason
            else:
                record["cost"] = ComparisonCostLineItem.from_cost_line_item(item)
                record["usage_unavailable_reason"] = None

    ordered_records = sorted(records.values(), key=lambda item: item["event_index"])
    next_ordinal: dict[str, int] = {}
    attempts: list[PairedCostAttempt] = []
    for record in ordered_records:
        operation_id = record["operation_id"]
        ordinal = record["ordinal"]
        if ordinal is None:
            ordinal = next_ordinal.get(operation_id, 1)
        next_ordinal[operation_id] = max(next_ordinal.get(operation_id, 1), ordinal + 1)
        attempt_id = record["model_attempt_id"] or f"{operation_id}:{ordinal}"
        attempts.append(
            PairedCostAttempt(
                attempt_id=attempt_id,
                operation_id=operation_id,
                session_id=evidence.session_id,
                branch_id=evidence.branch_id,
                role=evidence.role,
                operation=evidence.operation,
                attempt_ordinal=ordinal,
                provider_name=record["provider_name"],
                model=record["model"],
                source_checkpoint_id=evidence.source_checkpoint_id,
                cost=record["cost"],
                usage_unavailable_reason=record["usage_unavailable_reason"],
            )
        )
    return tuple(attempts)


def _optional_text(value: object) -> str | None:
    if type(value) is not str or not value.strip():
        return None
    return value


def _positive_int(value: object) -> int | None:
    if type(value) is not int or value < 1:
        return None
    return value


def _usage_is_missing(reason: str | None) -> bool:
    return reason in {
        "model.completed event has no valid normalized usage metrics",
        "model.completed event has no token usage metrics",
    }
