from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._validation import copy_json_value, require_clean_nonblank
from cayu.core.events import Event, EventType
from cayu.runtime.usage import UsageMetrics, usage_metrics_from_event_payload

_TOKENS_PER_MILLION = Decimal("1000000")


class ModelPricing(BaseModel):
    """User-supplied model pricing expressed as currency units per 1M tokens."""

    model_config = ConfigDict(extra="forbid")

    provider_name: str
    model: str
    input_per_million: Decimal = Field(ge=0)
    output_per_million: Decimal = Field(ge=0)
    cache_read_input_per_million: Decimal | None = Field(default=None, ge=0)
    cache_write_input_per_million: Decimal | None = Field(default=None, ge=0)
    currency: str = "USD"
    match: Literal["exact", "prefix"] = "exact"

    @field_validator("provider_name", "model", "currency")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator(
        "input_per_million",
        "output_per_million",
        "cache_read_input_per_million",
        "cache_write_input_per_million",
    )
    @classmethod
    def validate_decimal(cls, value: Decimal | None, info) -> Decimal | None:
        if value is None:
            return None
        if not value.is_finite():
            raise ValueError(f"{info.field_name} must be finite.")
        return value


class PricingCatalog(BaseModel):
    """Collection of user-supplied model prices used for cost estimation."""

    model_config = ConfigDict(extra="forbid")

    prices: tuple[ModelPricing, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_matches(self) -> PricingCatalog:
        seen: set[tuple[str, str, str]] = set()
        for price in self.prices:
            key = (
                price.provider_name.strip().lower(),
                price.model.strip(),
                price.match,
            )
            if key in seen:
                raise ValueError("Pricing catalog contains duplicate provider/model/match entries.")
            seen.add(key)
        return self

    def match_price(self, *, provider_name: str | None, model: str | None) -> ModelPricing | None:
        provider = provider_name.strip().lower() if type(provider_name) is str else None
        model_name = model.strip() if type(model) is str else None
        if provider is None or model_name is None:
            return None

        matches: list[ModelPricing] = []
        for price in self.prices:
            if price.provider_name.strip().lower() != provider:
                continue
            configured_model = price.model.strip()
            if (price.match == "exact" and model_name == configured_model) or (
                price.match == "prefix" and model_name.startswith(configured_model)
            ):
                matches.append(price)

        if not matches:
            return None
        return sorted(matches, key=_pricing_specificity, reverse=True)[0]


class CostLineItem(BaseModel):
    """Estimated cost for one model.completed event."""

    model_config = ConfigDict(extra="forbid")

    model_step: StrictInt = Field(ge=1)
    provider_name: str | None = None
    model: str | None = None
    pricing_provider_name: str | None = None
    pricing_model: str | None = None
    pricing_match: Literal["exact", "prefix"] | None = None
    priced: StrictBool
    currency: str
    input_tokens: StrictInt = Field(ge=0)
    output_tokens: StrictInt = Field(ge=0)
    cache_read_input_tokens: StrictInt = Field(ge=0)
    cache_write_input_tokens: StrictInt = Field(ge=0)
    uncached_input_tokens: StrictInt = Field(ge=0)
    input_cost: Decimal = Field(ge=0)
    output_cost: Decimal = Field(ge=0)
    cache_read_input_cost: Decimal = Field(ge=0)
    cache_write_input_cost: Decimal = Field(ge=0)
    total_cost: Decimal = Field(ge=0)
    missing_pricing_reason: str | None = None

    @field_validator(
        "provider_name",
        "model",
        "pricing_provider_name",
        "pricing_model",
        "currency",
        "missing_pricing_reason",
    )
    @classmethod
    def validate_optional_strings(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)


class SessionCostSummary(BaseModel):
    """Estimated session cost derived from durable model.completed events."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    currency: str
    model_steps: StrictInt = Field(ge=0)
    priced_model_steps: StrictInt = Field(ge=0)
    unpriced_model_steps: StrictInt = Field(ge=0)
    total_cost: Decimal = Field(ge=0)
    line_items: tuple[CostLineItem, ...] = Field(default_factory=tuple)

    @field_validator("session_id", "currency")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)


class CausalBudgetCostSummary(BaseModel):
    """Estimated cost for all sessions sharing one causal budget id."""

    model_config = ConfigDict(extra="forbid")

    causal_budget_id: str
    session_ids: list[str] = Field(default_factory=list)
    session_count: StrictInt = Field(default=0, ge=0)
    currency: str
    model_steps: StrictInt = Field(ge=0)
    priced_model_steps: StrictInt = Field(ge=0)
    unpriced_model_steps: StrictInt = Field(ge=0)
    total_cost: Decimal = Field(ge=0)
    line_items: tuple[CostLineItem, ...] = Field(default_factory=tuple)
    session_costs: tuple[SessionCostSummary, ...] = Field(default_factory=tuple)

    @field_validator("causal_budget_id", "currency")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("session_ids", mode="before")
    @classmethod
    def copy_session_ids(cls, value: list[str], info) -> list[str]:
        return _copy_string_list(value, info.field_name)


def copy_pricing_catalog(catalog: PricingCatalog) -> PricingCatalog:
    if type(catalog) is not PricingCatalog:
        raise TypeError("Pricing catalog must be a PricingCatalog instance.")
    return PricingCatalog(prices=tuple(price.model_copy(deep=True) for price in catalog.prices))


def estimate_session_cost(
    *,
    session_id: str,
    events: list[Event],
    pricing: PricingCatalog,
    currency: str = "USD",
) -> SessionCostSummary:
    session_id = require_clean_nonblank(session_id, "session_id")
    currency = require_clean_nonblank(currency, "currency").upper()
    pricing = copy_pricing_catalog(pricing)
    if type(events) is not list:
        raise TypeError("events must be a list.")

    line_items: list[CostLineItem] = []
    model_step = 0
    for event in events:
        if type(event) is not Event:
            raise TypeError("events must contain Event instances.")
        if event.type != EventType.MODEL_COMPLETED:
            continue
        model_step += 1
        metrics = usage_metrics_from_event_payload(event.payload)
        if metrics is None:
            line_items.append(
                _unpriced_line_item(
                    model_step=model_step,
                    provider_name=_optional_nonblank(event.payload.get("provider_name")),
                    model=_optional_nonblank(event.payload.get("model")),
                    currency=currency,
                    reason="model.completed event has no token usage metrics",
                )
            )
            continue
        line_items.append(
            _cost_line_item(
                model_step=model_step,
                metrics=metrics,
                pricing=pricing,
                currency=currency,
            )
        )

    total_cost = sum((item.total_cost for item in line_items), Decimal("0"))
    priced_model_steps = sum(1 for item in line_items if item.priced)
    unpriced_model_steps = len(line_items) - priced_model_steps
    return SessionCostSummary(
        session_id=session_id,
        currency=currency,
        model_steps=len(line_items),
        priced_model_steps=priced_model_steps,
        unpriced_model_steps=unpriced_model_steps,
        total_cost=total_cost,
        line_items=tuple(line_items),
    )


def estimate_causal_budget_cost(
    *,
    causal_budget_id: str,
    session_ids: list[str],
    events: list[Event],
    pricing: PricingCatalog,
    currency: str = "USD",
) -> CausalBudgetCostSummary:
    causal_budget_id = require_clean_nonblank(causal_budget_id, "causal_budget_id")
    copied_session_ids = _copy_string_list(session_ids, "session_ids")
    known_session_ids = set(copied_session_ids)
    filtered_events = [event for event in events if event.session_id in known_session_ids]
    summary = estimate_session_cost(
        session_id=causal_budget_id,
        events=filtered_events,
        pricing=pricing,
        currency=currency,
    )
    session_costs = tuple(
        estimate_session_cost(
            session_id=session_id,
            events=[event for event in filtered_events if event.session_id == session_id],
            pricing=pricing,
            currency=currency,
        )
        for session_id in copied_session_ids
    )
    return CausalBudgetCostSummary(
        causal_budget_id=causal_budget_id,
        session_ids=copied_session_ids,
        session_count=len(copied_session_ids),
        currency=summary.currency,
        model_steps=summary.model_steps,
        priced_model_steps=summary.priced_model_steps,
        unpriced_model_steps=summary.unpriced_model_steps,
        total_cost=summary.total_cost,
        line_items=summary.line_items,
        session_costs=session_costs,
    )


def _cost_line_item(
    *,
    model_step: int,
    metrics: UsageMetrics,
    pricing: PricingCatalog,
    currency: str,
) -> CostLineItem:
    price = pricing.match_price(provider_name=metrics.provider_name, model=metrics.model)
    if price is None:
        return _unpriced_line_item(
            model_step=model_step,
            provider_name=metrics.provider_name,
            model=metrics.model,
            currency=currency,
            reason="no matching model pricing",
            metrics=metrics,
        )

    if price.currency.upper() != currency.upper():
        return _unpriced_line_item(
            model_step=model_step,
            provider_name=metrics.provider_name,
            model=metrics.model,
            currency=currency,
            reason=f"pricing currency {price.currency} does not match requested {currency}",
            metrics=metrics,
        )

    uncached_input_tokens = metrics.cache.uncached_input_tokens
    if (
        uncached_input_tokens == 0
        and metrics.input_tokens > 0
        and metrics.cache.read_tokens == 0
        and metrics.cache.write_tokens == 0
    ):
        uncached_input_tokens = metrics.input_tokens

    cache_read_price = (
        price.cache_read_input_per_million
        if price.cache_read_input_per_million is not None
        else price.input_per_million
    )
    cache_write_price = (
        price.cache_write_input_per_million
        if price.cache_write_input_per_million is not None
        else price.input_per_million
    )

    input_cost = _token_cost(uncached_input_tokens, price.input_per_million)
    output_cost = _token_cost(metrics.output_tokens, price.output_per_million)
    cache_read_cost = _token_cost(metrics.cache.read_tokens, cache_read_price)
    cache_write_cost = _token_cost(metrics.cache.write_tokens, cache_write_price)
    total_cost = input_cost + output_cost + cache_read_cost + cache_write_cost

    return CostLineItem(
        model_step=model_step,
        provider_name=metrics.provider_name,
        model=metrics.model,
        pricing_provider_name=price.provider_name,
        pricing_model=price.model,
        pricing_match=price.match,
        priced=True,
        currency=currency.upper(),
        input_tokens=metrics.input_tokens,
        output_tokens=metrics.output_tokens,
        cache_read_input_tokens=metrics.cache.read_tokens,
        cache_write_input_tokens=metrics.cache.write_tokens,
        uncached_input_tokens=uncached_input_tokens,
        input_cost=input_cost,
        output_cost=output_cost,
        cache_read_input_cost=cache_read_cost,
        cache_write_input_cost=cache_write_cost,
        total_cost=total_cost,
    )


def _unpriced_line_item(
    *,
    model_step: int,
    provider_name: str | None,
    model: str | None,
    currency: str,
    reason: str,
    metrics: UsageMetrics | None = None,
) -> CostLineItem:
    return CostLineItem(
        model_step=model_step,
        provider_name=provider_name,
        model=model,
        priced=False,
        currency=currency.upper(),
        input_tokens=0 if metrics is None else metrics.input_tokens,
        output_tokens=0 if metrics is None else metrics.output_tokens,
        cache_read_input_tokens=0 if metrics is None else metrics.cache.read_tokens,
        cache_write_input_tokens=0 if metrics is None else metrics.cache.write_tokens,
        uncached_input_tokens=0 if metrics is None else metrics.cache.uncached_input_tokens,
        input_cost=Decimal("0"),
        output_cost=Decimal("0"),
        cache_read_input_cost=Decimal("0"),
        cache_write_input_cost=Decimal("0"),
        total_cost=Decimal("0"),
        missing_pricing_reason=reason,
    )


def _token_cost(tokens: int, price_per_million: Decimal) -> Decimal:
    return Decimal(tokens) * price_per_million / _TOKENS_PER_MILLION


def _pricing_specificity(price: ModelPricing) -> tuple[int, int]:
    match_score = 1 if price.match == "exact" else 0
    return match_score, len(price.model)


def _optional_nonblank(value: object) -> str | None:
    if type(value) is str and value.strip():
        return value
    return None


def _copy_string_list(value: list[str], field_name: str) -> list[str]:
    copied = copy_json_value(value, field_name)
    if type(copied) is not list:
        raise ValueError(f"{field_name} must be a list.")
    result: list[str] = []
    for index, item in enumerate(copied):
        if type(item) is not str:
            raise ValueError(f"{field_name}[{index}] must be a string.")
        result.append(require_clean_nonblank(item, f"{field_name}[{index}]"))
    return result
