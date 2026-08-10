from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import cast

import pytest
from pydantic import BaseModel, ValidationError

from cayu.core import Event, EventType
from cayu.core.billing import BillingIdentity
from cayu.runtime.aggregates import (
    AggregateAccuracy,
    AggregateAccuracyKind,
    UsageAggregateBreakdown,
    UsageAggregateGroup,
    UsageAggregateRemainder,
    UsageAggregateTotals,
    UsageBillingCostBreakdown,
    UsageBillingCostGroup,
    UsageBillingIdentity,
    UsageCostRollup,
    UsageCurrencyCost,
    UsagePricingInput,
    UsageRollupStoreResult,
    UsageSessionAggregateBreakdown,
    UsageSessionAggregateGroup,
    UsageSessionAggregateRemainder,
    UsageSessionCostBreakdown,
    UsageSessionCostGroup,
    UsageSessionCostRemainder,
    UsageSessionCostSummary,
)
from cayu.runtime.budgets import (
    BudgetCheck,
    BudgetLimit,
    BudgetReconciliation,
    BudgetSettlementRecord,
    budget_reconciliation_payload,
    budget_settlement_event_id,
    budget_settlement_id,
)
from cayu.runtime.costs import (
    CausalBudgetCostSummary,
    CostLineItem,
    ModelPrice,
    PriceBook,
    SessionCostSummary,
)
from cayu.runtime.usage import (
    AggregateCacheUsageMetrics,
    AggregateUsageMetrics,
    CacheUsageMetrics,
    CausalBudgetUsageSummary,
    SessionUsageSummary,
    UsageMetrics,
)

_SnapshotCase = Callable[[], tuple[BaseModel, Callable[[], None]]]


def _aggregate_usage(*, input_tokens: int = 3) -> AggregateUsageMetrics:
    return AggregateUsageMetrics(
        input_tokens=input_tokens,
        output_tokens=2,
        total_tokens=input_tokens + 2,
        reasoning_output_tokens=1,
        cache=AggregateCacheUsageMetrics(
            read_tokens=1,
            write_tokens=2,
            write_5m_tokens=0,
            write_1h_tokens=0,
            write_unknown_ttl_tokens=0,
            cached_input_tokens=1,
            uncached_input_tokens=2,
        ),
    )


def _totals(*, input_tokens: int = 3) -> UsageAggregateTotals:
    return UsageAggregateTotals(
        session_count=1,
        model_steps=1,
        model_steps_with_usage=1,
        tool_calls=1,
        usage=_aggregate_usage(input_tokens=input_tokens),
    )


def _exact() -> AggregateAccuracy:
    return AggregateAccuracy(kind=AggregateAccuracyKind.EXACT)


def _mutate_model(model: BaseModel, field: str, value: object) -> None:
    object.__setattr__(model, field, value)


def _usage_metrics_case() -> tuple[BaseModel, Callable[[], None]]:
    cache = CacheUsageMetrics(read_tokens=1)
    identity = BillingIdentity(provider_name="provider", resource_id="model")
    result = UsageMetrics(
        billing_identity=identity,
        input_tokens=1,
        total_tokens=1,
        cache=cache,
    )

    def mutate() -> None:
        cache.read_tokens = 99
        cast("dict[str, str]", identity.request_evidence)["region"] = "mutated"

    return result, mutate


def _aggregate_usage_case() -> tuple[BaseModel, Callable[[], None]]:
    cache = AggregateCacheUsageMetrics(
        read_tokens=1,
        write_tokens=0,
        write_5m_tokens=0,
        write_1h_tokens=0,
        write_unknown_ttl_tokens=0,
        cached_input_tokens=0,
        uncached_input_tokens=1,
    )
    result = AggregateUsageMetrics(
        input_tokens=1,
        output_tokens=0,
        total_tokens=1,
        reasoning_output_tokens=0,
        cache=cache,
    )
    return result, lambda: _mutate_model(cache, "read_tokens", 99)


def _session_usage_case() -> tuple[BaseModel, Callable[[], None]]:
    usage = _aggregate_usage()
    result = SessionUsageSummary(session_id="session", model_steps=1, usage=usage)
    return result, lambda: _mutate_model(usage.cache, "read_tokens", 99)


def _causal_usage_case() -> tuple[BaseModel, Callable[[], None]]:
    usage = _aggregate_usage()
    session = SessionUsageSummary(session_id="session", model_steps=1, usage=usage)
    result = CausalBudgetUsageSummary(
        causal_budget_id="budget",
        session_ids=["session"],
        session_count=1,
        model_steps=1,
        usage=usage,
        session_summaries=(session,),
    )

    def mutate() -> None:
        _mutate_model(usage.cache, "read_tokens", 99)
        session.provider_names.append("mutated")

    return result, mutate


def _usage_totals_case() -> tuple[BaseModel, Callable[[], None]]:
    usage = _aggregate_usage()
    result = UsageAggregateTotals(
        session_count=1,
        model_steps=1,
        model_steps_with_usage=1,
        tool_calls=1,
        usage=usage,
    )
    return result, lambda: _mutate_model(usage.cache, "read_tokens", 99)


def _usage_group_case() -> tuple[BaseModel, Callable[[], None]]:
    totals = _totals()
    result = UsageAggregateGroup(provider_name="provider", model="model", totals=totals)
    return result, lambda: _mutate_model(totals.usage.cache, "read_tokens", 99)


def _usage_remainder_case() -> tuple[BaseModel, Callable[[], None]]:
    totals = _totals()
    result = UsageAggregateRemainder(group_count=1, totals=totals)
    return result, lambda: _mutate_model(totals.usage.cache, "read_tokens", 99)


def _usage_breakdown_case() -> tuple[BaseModel, Callable[[], None]]:
    group = UsageAggregateGroup(provider_name="provider", model="model", totals=_totals())
    accuracy = _exact()
    result = UsageAggregateBreakdown(groups=(group,), remainder=None, accuracy=accuracy)

    def mutate() -> None:
        _mutate_model(group.totals.usage.cache, "read_tokens", 99)
        _mutate_model(accuracy, "kind", AggregateAccuracyKind.SAMPLED)

    return result, mutate


def _session_usage_group_case() -> tuple[BaseModel, Callable[[], None]]:
    totals = _totals()
    result = UsageSessionAggregateGroup(
        session_id="session",
        status="completed",
        active=False,
        totals=totals,
    )
    return result, lambda: _mutate_model(totals.usage.cache, "read_tokens", 99)


def _session_usage_remainder_case() -> tuple[BaseModel, Callable[[], None]]:
    totals = _totals()
    result = UsageSessionAggregateRemainder(
        group_count=1,
        active_session_count=0,
        totals=totals,
    )
    return result, lambda: _mutate_model(totals.usage.cache, "read_tokens", 99)


def _session_usage_breakdown_case() -> tuple[BaseModel, Callable[[], None]]:
    group = UsageSessionAggregateGroup(
        session_id="session",
        status="completed",
        active=False,
        totals=_totals(),
    )
    accuracy = _exact()
    result = UsageSessionAggregateBreakdown(
        groups=(group,),
        remainder=None,
        accuracy=accuracy,
    )

    def mutate() -> None:
        _mutate_model(group.totals.usage.cache, "read_tokens", 99)
        _mutate_model(accuracy, "kind", AggregateAccuracyKind.SAMPLED)

    return result, mutate


def _usage_rollup_case() -> tuple[BaseModel, Callable[[], None]]:
    start = datetime(2026, 8, 1, tzinfo=UTC)
    totals = _totals()
    totals_accuracy = _exact()
    provider_group = UsageAggregateGroup(
        provider_name="provider",
        model=None,
        totals=_totals(),
    )
    model_group = UsageAggregateGroup(
        provider_name="provider",
        model="model",
        totals=_totals(),
    )
    provider_breakdown = UsageAggregateBreakdown(
        groups=(provider_group,), remainder=None, accuracy=_exact()
    )
    model_breakdown = UsageAggregateBreakdown(
        groups=(model_group,), remainder=None, accuracy=_exact()
    )
    session_group = UsageSessionAggregateGroup(
        session_id="session",
        status="completed",
        active=False,
        totals=_totals(),
    )
    session_breakdown = UsageSessionAggregateBreakdown(
        groups=(session_group,), remainder=None, accuracy=_exact()
    )
    pricing_metrics = UsageMetrics(
        provider_name="provider",
        model="model",
        input_tokens=3,
        output_tokens=2,
        total_tokens=5,
        reasoning_output_tokens=1,
        cache=CacheUsageMetrics(
            read_tokens=1,
            write_tokens=2,
            cached_input_tokens=1,
            uncached_input_tokens=2,
        ),
    )
    pricing_input = UsagePricingInput(
        effective_on=start.date(), occurrences=1, metrics=pricing_metrics
    )
    session_pricing_input = UsagePricingInput(
        session_id="session",
        effective_on=start.date(),
        occurrences=1,
        metrics=pricing_metrics,
    )
    pricing_accuracy = _exact()
    session_pricing_accuracy = _exact()
    shared_pricing_metrics = pricing_input.metrics
    retained_session_pricing_metrics = session_pricing_input.metrics
    assert shared_pricing_metrics is not None
    assert retained_session_pricing_metrics is not None
    result = UsageRollupStoreResult(
        as_of=start + timedelta(days=2),
        start_at=start,
        end_at=start + timedelta(days=1),
        totals=totals,
        totals_accuracy=totals_accuracy,
        provider_breakdown=provider_breakdown,
        model_breakdown=model_breakdown,
        session_breakdown=session_breakdown,
        pricing_inputs=(pricing_input,),
        pricing_inputs_included=True,
        pricing_input_group_count=1,
        pricing_inputs_accuracy=pricing_accuracy,
        session_pricing_inputs=(session_pricing_input,),
        session_pricing_inputs_included=True,
        session_pricing_input_group_count=1,
        session_pricing_inputs_accuracy=session_pricing_accuracy,
        active_session_count=0,
        matching_session_count=1,
    )

    def mutate() -> None:
        _mutate_model(totals.usage.cache, "read_tokens", 99)
        _mutate_model(totals_accuracy, "kind", AggregateAccuracyKind.SAMPLED)
        _mutate_model(provider_group.totals.usage.cache, "read_tokens", 98)
        _mutate_model(model_group.totals.usage.cache, "read_tokens", 97)
        _mutate_model(session_group.totals.usage.cache, "read_tokens", 96)
        _mutate_model(shared_pricing_metrics, "input_tokens", 95)
        _mutate_model(retained_session_pricing_metrics, "input_tokens", 94)
        _mutate_model(pricing_accuracy, "kind", AggregateAccuracyKind.SAMPLED)
        _mutate_model(session_pricing_accuracy, "kind", AggregateAccuracyKind.SAMPLED)

    return result, mutate


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(_usage_metrics_case, id="step-metrics"),
        pytest.param(_aggregate_usage_case, id="aggregate-metrics"),
        pytest.param(_session_usage_case, id="session-summary"),
        pytest.param(_causal_usage_case, id="causal-summary"),
        pytest.param(_usage_totals_case, id="aggregate-totals"),
        pytest.param(_usage_group_case, id="aggregate-group"),
        pytest.param(_usage_remainder_case, id="aggregate-remainder"),
        pytest.param(_usage_breakdown_case, id="aggregate-breakdown"),
        pytest.param(_session_usage_group_case, id="session-group"),
        pytest.param(_session_usage_remainder_case, id="session-remainder"),
        pytest.param(_session_usage_breakdown_case, id="session-breakdown"),
        pytest.param(_usage_rollup_case, id="store-rollup"),
    ],
)
def test_usage_contracts_capture_transitively_detached_snapshots(case: _SnapshotCase) -> None:
    accepted, mutate_sources = case()
    snapshot = accepted.model_dump(mode="python")

    mutate_sources()

    assert accepted.model_dump(mode="python") == snapshot


def test_usage_parent_revalidates_a_previously_accepted_nested_model() -> None:
    cache = AggregateCacheUsageMetrics(
        read_tokens=1,
        write_tokens=0,
        write_5m_tokens=0,
        write_1h_tokens=0,
        write_unknown_ttl_tokens=0,
        cached_input_tokens=0,
        uncached_input_tokens=1,
    )
    _mutate_model(cache, "read_tokens", -1)

    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        AggregateUsageMetrics(
            input_tokens=1,
            output_tokens=0,
            total_tokens=1,
            reasoning_output_tokens=0,
            cache=cache,
        )


def _unpriced_line_item() -> CostLineItem:
    return CostLineItem(
        model_step=1,
        priced=False,
        currency="USD",
        input_tokens=1,
        output_tokens=0,
        cache_read_input_tokens=0,
        cache_write_input_tokens=0,
        uncached_input_tokens=1,
        input_cost=Decimal("0"),
        output_cost=Decimal("0"),
        cache_read_input_cost=Decimal("0"),
        cache_write_input_cost=Decimal("0"),
        total_cost=Decimal("0"),
        missing_pricing_reason="missing price",
    )


def _session_cost(*, line_item: CostLineItem | None = None) -> SessionCostSummary:
    return SessionCostSummary(
        session_id="session",
        currency="USD",
        model_steps=1,
        priced_model_steps=0,
        unpriced_model_steps=1,
        total_cost=Decimal("0"),
        line_items=(_unpriced_line_item() if line_item is None else line_item,),
    )


def _session_cost_case() -> tuple[BaseModel, Callable[[], None]]:
    item = _unpriced_line_item()
    result = _session_cost(line_item=item)
    return result, lambda: _mutate_model(item, "missing_pricing_reason", "mutated")


def _causal_cost_case() -> tuple[BaseModel, Callable[[], None]]:
    item = _unpriced_line_item()
    session = _session_cost(line_item=item)
    result = CausalBudgetCostSummary(
        causal_budget_id="budget",
        session_ids=["session"],
        session_count=1,
        currency="USD",
        model_steps=1,
        priced_model_steps=0,
        unpriced_model_steps=1,
        total_cost=Decimal("0"),
        line_items=(item,),
        session_costs=(session,),
    )

    def mutate() -> None:
        _mutate_model(item, "missing_pricing_reason", "mutated")
        _mutate_model(session, "total_cost", Decimal("99"))

    return result, mutate


def _budget_check_case() -> tuple[BaseModel, Callable[[], None]]:
    summary = _session_cost()
    result = BudgetCheck(
        budget_limit_id=f"blim_{'0' * 64}",
        scope="session",
        currency="USD",
        maximum=Decimal("10"),
        actual=Decimal("0"),
        model_steps=1,
        unpriced_model_steps=1,
        limit_reached=False,
        message="within budget",
        cost_summary=summary,
    )
    return result, lambda: _mutate_model(summary, "total_cost", Decimal("99"))


def _billing_group_case() -> tuple[BaseModel, Callable[[], None]]:
    identity = UsageBillingIdentity(provider_name="provider", resource_id="model")
    result = UsageBillingCostGroup(
        billing_identity=identity,
        pricing_provider_name="provider",
        pricing_model="model",
        priced=True,
        model_steps=1,
        currency="USD",
        total_cost=Decimal("1.25"),
    )
    return result, lambda: _mutate_model(identity, "resource_id", "mutated")


def _billing_breakdown_case() -> tuple[BaseModel, Callable[[], None]]:
    group, _ = _billing_group_case()
    assert isinstance(group, UsageBillingCostGroup)
    accuracy = _exact()
    result = UsageBillingCostBreakdown(
        identified_model_steps=1,
        groups=(group,),
        remainder=None,
        accuracy=accuracy,
    )

    def mutate() -> None:
        _mutate_model(group.billing_identity, "resource_id", "mutated")
        _mutate_model(accuracy, "kind", AggregateAccuracyKind.SAMPLED)

    return result, mutate


def _usage_session_cost_summary_case() -> tuple[BaseModel, Callable[[], None]]:
    accuracy = _exact()
    currency = UsageCurrencyCost(currency="USD", model_steps=1, total_cost=Decimal("1.25"))
    result = UsageSessionCostSummary(
        accuracy=accuracy,
        evaluated_model_steps=1,
        priced_model_steps=1,
        unpriced_model_steps=0,
        unevaluated_model_steps=0,
        currencies=(currency,),
        unpriced_reasons=(),
    )

    def mutate() -> None:
        _mutate_model(accuracy, "kind", AggregateAccuracyKind.SAMPLED)
        _mutate_model(currency, "total_cost", Decimal("99"))

    return result, mutate


def _usage_session_cost_group_case() -> tuple[BaseModel, Callable[[], None]]:
    cost, _ = _usage_session_cost_summary_case()
    assert isinstance(cost, UsageSessionCostSummary)
    result = UsageSessionCostGroup(session_id="session", cost=cost)
    return result, lambda: _mutate_model(cost.currencies[0], "total_cost", Decimal("99"))


def _usage_session_cost_remainder_case() -> tuple[BaseModel, Callable[[], None]]:
    cost, _ = _usage_session_cost_summary_case()
    assert isinstance(cost, UsageSessionCostSummary)
    result = UsageSessionCostRemainder(group_count=1, cost=cost)
    return result, lambda: _mutate_model(cost.currencies[0], "total_cost", Decimal("99"))


def _usage_session_cost_breakdown_case() -> tuple[BaseModel, Callable[[], None]]:
    cost, _ = _usage_session_cost_summary_case()
    assert isinstance(cost, UsageSessionCostSummary)
    group = UsageSessionCostGroup(session_id="session", cost=cost)
    accuracy = _exact()
    result = UsageSessionCostBreakdown(
        price_book_version="v1",
        price_book_generated_at="2026-08-01",
        groups=(group,),
        remainder=None,
        accuracy=accuracy,
    )

    def mutate() -> None:
        _mutate_model(cost.currencies[0], "total_cost", Decimal("99"))
        _mutate_model(accuracy, "kind", AggregateAccuracyKind.SAMPLED)

    return result, mutate


def _usage_cost_rollup_case() -> tuple[BaseModel, Callable[[], None]]:
    accuracy = _exact()
    currency = UsageCurrencyCost(currency="USD", model_steps=1, total_cost=Decimal("1.25"))
    billing, _ = _billing_breakdown_case()
    assert isinstance(billing, UsageBillingCostBreakdown)
    result = UsageCostRollup(
        price_book_version="v1",
        price_book_generated_at="2026-08-01",
        accuracy=accuracy,
        evaluated_model_steps=1,
        priced_model_steps=1,
        unpriced_model_steps=0,
        unevaluated_model_steps=0,
        currencies=(currency,),
        unpriced_reasons=(),
        billing_breakdown=billing,
    )

    def mutate() -> None:
        _mutate_model(accuracy, "kind", AggregateAccuracyKind.SAMPLED)
        _mutate_model(currency, "total_cost", Decimal("99"))
        _mutate_model(billing.groups[0].billing_identity, "resource_id", "mutated")

    return result, mutate


def _budget_limit_case() -> tuple[BaseModel, Callable[[], None]]:
    pricing = PriceBook(
        price_book_version="v1",
        generated_at="2026-08-01",
        prices=(
            ModelPrice.fixed(
                provider_name="provider",
                model="model",
                match="exact",
                input_per_million=Decimal("1.25"),
                output_per_million=Decimal("2.50"),
            ),
        ),
    )
    result = BudgetLimit(max_estimated_cost=Decimal("10"), pricing=pricing)
    return result, lambda: _mutate_model(pricing, "price_book_version", "mutated")


def _budget_settlement_case() -> tuple[BaseModel, Callable[[], None]]:
    reservation_id = "reservation"
    settlement_id = budget_settlement_id(reservation_id)
    settled_at = datetime(2026, 8, 1, tzinfo=UTC)
    reconciliation = BudgetReconciliation(
        reservation_id=reservation_id,
        settlement_id=settlement_id,
        settlement_kind="completed",
        budget_limit_id=f"blim_{'0' * 64}",
        model_step_id=f"mstep_{'0' * 32}",
        model_attempt_id=f"matt_{'0' * 32}",
        status="reconciled",
        reserved_amount=Decimal("1"),
        actual_amount=Decimal("1"),
        released_amount=Decimal("0"),
        settled_at=settled_at,
    )
    event = Event(
        id=budget_settlement_event_id(settlement_id),
        type=EventType.BUDGET_RECONCILED,
        timestamp=settled_at,
        session_id="session",
        agent_name="agent",
        payload={**budget_reconciliation_payload(reconciliation), "audit_marker": "original"},
    )
    result = BudgetSettlementRecord(
        settlement_id=settlement_id,
        reservation_id=reservation_id,
        settlement_kind="completed",
        session_id="session",
        agent_name="agent",
        reconciliation=reconciliation,
        event=event,
    )

    def mutate() -> None:
        event.payload["audit_marker"] = "mutated"
        _mutate_model(reconciliation, "reason", "mutated")

    return result, mutate


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(_session_cost_case, id="session-cost"),
        pytest.param(_causal_cost_case, id="causal-cost"),
        pytest.param(_budget_check_case, id="budget-check"),
        pytest.param(_billing_group_case, id="billing-group"),
        pytest.param(_billing_breakdown_case, id="billing-breakdown"),
        pytest.param(_usage_session_cost_summary_case, id="session-cost-summary"),
        pytest.param(_usage_session_cost_group_case, id="session-cost-group"),
        pytest.param(_usage_session_cost_remainder_case, id="session-cost-remainder"),
        pytest.param(_usage_session_cost_breakdown_case, id="session-cost-breakdown"),
        pytest.param(_usage_cost_rollup_case, id="cost-rollup"),
        pytest.param(_budget_limit_case, id="budget-limit"),
        pytest.param(_budget_settlement_case, id="settlement-record"),
    ],
)
def test_financial_contracts_capture_transitively_detached_snapshots(case: _SnapshotCase) -> None:
    accepted, mutate_sources = case()
    snapshot = accepted.model_dump(mode="python")

    mutate_sources()

    assert accepted.model_dump(mode="python") == snapshot


def test_cost_parent_revalidates_a_previously_accepted_nested_model() -> None:
    item = _unpriced_line_item()
    _mutate_model(item, "total_cost", Decimal("-1"))

    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        _session_cost(line_item=item)


def test_budget_settlement_rejects_an_event_subclass_alias() -> None:
    accepted, _ = _budget_settlement_case()
    assert isinstance(accepted, BudgetSettlementRecord)

    class CallerEvent(Event):
        def __getattribute__(self, name: str):
            if name == "type":
                raise AssertionError("event attributes must not be read")
            return super().__getattribute__(name)

    supplied = CallerEvent.model_construct(**accepted.event.model_dump(mode="python"))

    with pytest.raises(TypeError, match="Events must be Event instances"):
        BudgetSettlementRecord(
            settlement_id=accepted.settlement_id,
            reservation_id=accepted.reservation_id,
            settlement_kind=accepted.settlement_kind,
            session_id=accepted.session_id,
            agent_name=accepted.agent_name,
            reconciliation=accepted.reconciliation,
            event=supplied,
        )
