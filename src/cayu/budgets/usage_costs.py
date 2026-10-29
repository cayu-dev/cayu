"""Bounded cost reporting composed from native snapshots and canonical pricing."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cayu._validation import revalidate_model_input, revalidate_model_inputs
from cayu.budgets.aggregates import (
    EXACT_AGGREGATE,
    AggregateAccuracy,
    AggregateAccuracyKind,
    UsageAggregateBreakdown,
    UsageAggregateTotals,
    UsageCostRollup,
    UsagePricingInput,
    UsageRollupStoreResult,
    UsageSessionAggregateBreakdown,
    UsageSessionAggregateGroup,
    UsageSessionAggregateRemainder,
    UsageSessionCostBreakdown,
    UsageSessionCostSummary,
    _combined_aggregate_accuracy,
    _estimate_usage_cost,
    _estimate_usage_session_cost_breakdown,
    _session_cost_summary,
    _sum_usage_aggregate_totals,
    _unevaluated_session_cost,
)
from cayu.budgets.pricing import PriceBook
from cayu.budgets.usage import PositiveAggregateCount

UsageSessionOrder = Literal["tokens", "cost"]
CostRankingUnavailableReason = Literal[
    "pricing_not_configured",
    "candidate_limit",
    "pricing_projection_incomplete",
    "prices_incomplete",
    "mixed_currencies",
    "candidate_byte_limit",
]


class UsageSessionRanking(BaseModel):
    model_config = ConfigDict(extra="forbid")
    requested_order: UsageSessionOrder
    effective_order: UsageSessionOrder
    unavailable_reason: CostRankingUnavailableReason | None = None
    currency: str | None = None

    @model_validator(mode="after")
    def truthful_order(self) -> UsageSessionRanking:
        if (self.requested_order != self.effective_order) != (self.unavailable_reason is not None):
            raise ValueError("A ranking fallback requires an explicit unavailable reason.")
        if self.effective_order != "cost" and self.currency is not None:
            raise ValueError("Token ordering has no comparison currency.")
        return self


class UsageDimensionCostGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider_name: str | None = None
    model: str | None = None
    cost: UsageSessionCostSummary

    @field_validator("cost", mode="before")
    @classmethod
    def copy_cost(cls, value: object) -> object:
        return revalidate_model_input(value, UsageSessionCostSummary)


class UsageDimensionCostRemainder(BaseModel):
    model_config = ConfigDict(extra="forbid")
    group_count: PositiveAggregateCount
    cost: UsageSessionCostSummary

    @field_validator("cost", mode="before")
    @classmethod
    def copy_cost(cls, value: object) -> object:
        return revalidate_model_input(value, UsageSessionCostSummary)


class UsageDimensionCostBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")
    price_book_version: str
    price_book_generated_at: str
    groups: tuple[UsageDimensionCostGroup, ...] = Field(max_length=100)
    remainder: UsageDimensionCostRemainder | None = None
    accuracy: AggregateAccuracy

    @field_validator("groups", mode="before")
    @classmethod
    def copy_groups(cls, value: object) -> object:
        return revalidate_model_inputs(value, UsageDimensionCostGroup)

    @field_validator("remainder", "accuracy", mode="before")
    @classmethod
    def copy_nested(cls, value: object) -> object:
        return revalidate_model_input(value, UsageDimensionCostRemainder, AggregateAccuracy)


@dataclass(frozen=True)
class UsageCostReport:
    result: UsageRollupStoreResult
    cost: UsageCostRollup | None
    session_ranking: UsageSessionRanking | None
    session_cost_breakdown: UsageSessionCostBreakdown | None
    provider_cost_breakdown: UsageDimensionCostBreakdown | None
    model_cost_breakdown: UsageDimensionCostBreakdown | None


class _CostReporter:
    def __init__(self, result: UsageRollupStoreResult, pricing: PriceBook):
        self.result = result
        self.pricing = pricing
        self.cache: dict[tuple[date, str | None], object] = {}

    def summarize(
        self,
        totals: UsageAggregateTotals,
        inputs: tuple[UsagePricingInput, ...],
        *,
        session: bool = False,
    ) -> UsageSessionCostSummary:
        accuracy = self.result.pricing_inputs_accuracy
        if session:
            accuracy = _combined_aggregate_accuracy(
                accuracy, self.result.session_pricing_inputs_accuracy
            )
        return _session_cost_summary(
            _estimate_usage_cost(
                totals=totals,
                totals_accuracy=self.result.totals_accuracy,
                pricing_inputs=inputs,
                pricing_inputs_accuracy=accuracy,
                pricing=self.pricing,
                billing_group_limit=1,
                estimate_cache=self.cache,
            )
        )

    def dimension(self, dimension: Literal["provider", "model"]) -> UsageDimensionCostBreakdown:
        breakdown: UsageAggregateBreakdown = getattr(self.result, f"{dimension}_breakdown")
        # Strict pricing validation can reject metrics whose tolerant usage
        # projection still retained provider/model counters. Their attribution
        # cannot be reconstructed from metrics=None: do not invent an unknown
        # group or silently drop those occurrences from a real group's cost.
        unavailable = (
            AggregateAccuracy(
                kind=AggregateAccuracyKind.TRUNCATED,
                reason="Provider/model cost attribution is unavailable because some pricing inputs have no valid normalized metrics.",
            )
            if any(item.metrics is None for item in self.result.pricing_inputs)
            else None
        )

        def summarize(
            totals: UsageAggregateTotals, inputs: tuple[UsagePricingInput, ...]
        ) -> UsageSessionCostSummary:
            if unavailable is not None:
                return _unevaluated_session_cost(
                    model_steps=totals.model_steps, accuracy=unavailable
                )
            return self.summarize(totals, inputs)

        inputs: dict[tuple[str | None, str | None], list[UsagePricingInput]] = defaultdict(list)
        for item in self.result.pricing_inputs:
            metrics = item.metrics
            key = (
                None if metrics is None else metrics.provider_name,
                None if metrics is None or dimension == "provider" else metrics.model,
            )
            inputs[key].append(item)
        groups = tuple(
            UsageDimensionCostGroup(
                provider_name=group.provider_name,
                model=group.model,
                cost=summarize(
                    group.totals, tuple(inputs.pop((group.provider_name, group.model), ()))
                ),
            )
            for group in breakdown.groups
        )
        remainder = None
        if breakdown.remainder is not None:
            remainder = UsageDimensionCostRemainder(
                group_count=breakdown.remainder.group_count,
                cost=summarize(
                    breakdown.remainder.totals,
                    tuple(item for values in inputs.values() for item in values),
                ),
            )
        return UsageDimensionCostBreakdown(
            price_book_version=self.pricing.price_book_version,
            price_book_generated_at=self.pricing.generated_at,
            groups=groups,
            remainder=remainder,
            accuracy=_combined_aggregate_accuracy(
                _combined_aggregate_accuracy(
                    breakdown.accuracy, self.result.pricing_inputs_accuracy
                ),
                unavailable or EXACT_AGGREGATE,
            ),
        )

    def ordered_sessions(
        self, cost: UsageCostRollup
    ) -> tuple[tuple[UsageSessionAggregateGroup, ...] | None, CostRankingUnavailableReason | None]:
        breakdown = self.result.session_breakdown
        assert breakdown is not None
        if breakdown.remainder is not None:
            return None, "candidate_limit"
        if any(
            accuracy.kind is not AggregateAccuracyKind.EXACT
            for accuracy in (
                breakdown.accuracy,
                self.result.totals_accuracy,
                self.result.pricing_inputs_accuracy,
                self.result.session_pricing_inputs_accuracy,
            )
        ):
            return None, "pricing_projection_incomplete"
        if cost.unpriced_reasons or cost.unpriced_model_steps or cost.unpriced_auxiliary_attempts:
            return None, "prices_incomplete"
        if len(cost.currencies) > 1:
            return None, "mixed_currencies"
        inputs: dict[str, list[UsagePricingInput]] = defaultdict(list)
        for item in self.result.session_pricing_inputs:
            assert item.session_id is not None
            inputs[item.session_id].append(item)
        values: dict[str, Decimal] = {}
        for group in breakdown.groups:
            summary = self.summarize(group.totals, tuple(inputs[group.session_id]), session=True)
            # Validate each candidate, including zero-step auxiliary/hosted work.
            if (
                summary.unpriced_reasons
                or summary.unpriced_model_steps
                or summary.unpriced_auxiliary_attempts
            ):
                return None, "prices_incomplete"
            values[group.session_id] = (
                summary.currencies[0].total_cost if summary.currencies else Decimal(0)
            )
        return tuple(
            sorted(
                breakdown.groups,
                key=lambda group: (
                    values[group.session_id].copy_negate(),
                    -group.totals.usage.total_tokens,
                    -group.totals.model_steps,
                    group.session_id,
                ),
            )
        ), None


def _select_sessions(
    result: UsageRollupStoreResult,
    groups: tuple[UsageSessionAggregateGroup, ...],
    *,
    limit: int,
    order: UsageSessionOrder,
) -> UsageRollupStoreResult:
    """Project a validated snapshot, preserving exact omitted usage and pricing."""
    breakdown = result.session_breakdown
    assert breakdown is not None
    selected = groups[:limit]
    selected_ids = {group.session_id for group in selected}
    omitted = [group for group in breakdown.groups if group.session_id not in selected_ids]
    totals = [group.totals for group in omitted]
    count = len(omitted)
    active = sum(int(group.active) for group in omitted)
    if breakdown.remainder is not None:
        count += breakdown.remainder.group_count
        active += breakdown.remainder.active_session_count
        totals.append(breakdown.remainder.totals)
    remainder = (
        None
        if not count
        else UsageSessionAggregateRemainder(
            group_count=count,
            active_session_count=active,
            totals=_sum_usage_aggregate_totals(totals),
        )
    )
    selected_breakdown = UsageSessionAggregateBreakdown(
        groups=selected,
        order=order,
        remainder=remainder,
        accuracy=_combined_aggregate_accuracy(
            breakdown.accuracy,
            EXACT_AGGREGATE
            if remainder is None
            else AggregateAccuracy(
                kind=AggregateAccuracyKind.TRUNCATED,
                reason="Matching sessions exceed session_group_limit.",
                limit=limit,
            ),
        ),
    )
    inputs = tuple(
        item for item in result.session_pricing_inputs if item.session_id in selected_ids
    )
    # Reconstruct only after validated native provenance; this owner derives a new
    # selection and its attribution commitment from that same snapshot, without
    # upgrading the snapshot's accuracy.
    payload = result.model_dump(mode="python")
    payload.update(session_breakdown=selected_breakdown, session_pricing_inputs=inputs)
    if result.session_pricing_inputs_accuracy.kind is AggregateAccuracyKind.EXACT:
        payload["session_pricing_input_group_count"] = len(inputs)
    return UsageRollupStoreResult.model_validate(payload)


def build_usage_cost_report(
    result: UsageRollupStoreResult,
    pricing: PriceBook | None,
    *,
    session_order: UsageSessionOrder,
    session_limit: int | None,
    group_limit: int,
    unavailable_reason: CostRankingUnavailableReason | None = None,
) -> UsageCostReport:
    reporter = None if pricing is None else _CostReporter(result, pricing)
    cost = (
        None
        if reporter is None
        else _estimate_usage_cost(
            totals=result.totals,
            totals_accuracy=result.totals_accuracy,
            pricing_inputs=result.pricing_inputs,
            pricing_inputs_accuracy=result.pricing_inputs_accuracy,
            pricing=pricing,
            billing_group_limit=group_limit,
            estimate_cache=reporter.cache,
        )
    )
    ranking = None
    if result.session_breakdown is not None:
        assert session_limit is not None
        groups = result.session_breakdown.groups
        effective: UsageSessionOrder = "tokens"
        reason = unavailable_reason
        if session_order == "cost" and reason is None:
            if reporter is None:
                reason = "pricing_not_configured"
            else:
                assert cost is not None
                ordered, reason = reporter.ordered_sessions(cost)
                if ordered is not None:
                    effective, groups = "cost", ordered
        ranking = UsageSessionRanking(
            requested_order=session_order,
            effective_order=effective,
            unavailable_reason=reason,
            currency=cost.currencies[0].currency
            if effective == "cost" and cost is not None and cost.currencies
            else None,
        )
        result = _select_sessions(result, groups, limit=session_limit, order=effective)
    return UsageCostReport(
        result=result,
        cost=cost,
        session_ranking=ranking,
        session_cost_breakdown=None
        if reporter is None or result.session_breakdown is None
        else _estimate_usage_session_cost_breakdown(result, pricing, estimate_cache=reporter.cache),
        provider_cost_breakdown=None if reporter is None else reporter.dimension("provider"),
        model_cost_breakdown=None if reporter is None else reporter.dimension("model"),
    )
