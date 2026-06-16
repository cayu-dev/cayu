from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from decimal import Decimal
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._validation import require_clean_nonblank
from cayu.core.events import Event, EventType, copy_event
from cayu.runtime.costs import PricingCatalog, SessionCostSummary, estimate_session_cost
from cayu.runtime.sessions import EventQuery, SessionStore

BudgetScope = Literal["app", "agent"]
BudgetWindow = Literal["all_time"]


class BudgetLimit(BaseModel):
    """Estimated-cost budget that applies across one durable runtime scope."""

    model_config = ConfigDict(extra="forbid")

    scope: BudgetScope
    max_estimated_cost: Decimal = Field(gt=0)
    pricing: PricingCatalog
    currency: str = "USD"
    window: BudgetWindow = "all_time"
    key: str | None = None
    allow_unpriced: StrictBool = False

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name).upper()

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("max_estimated_cost")
    @classmethod
    def validate_cost(cls, value: Decimal, info) -> Decimal:
        if not value.is_finite():
            raise ValueError(f"{info.field_name} must be finite.")
        return value

    @model_validator(mode="after")
    def validate_scope_key(self) -> BudgetLimit:
        if self.scope == "app" and self.key is not None:
            raise ValueError("App budget limits must not set key.")
        if self.scope == "agent" and self.key is None:
            raise ValueError("Agent budget limits require key.")
        return self


class BudgetPolicy(BaseModel):
    """App-level budget policy applied automatically by the runtime."""

    model_config = ConfigDict(extra="forbid")

    limits: tuple[BudgetLimit, ...] = Field(default_factory=tuple)

    @field_validator("limits", mode="before")
    @classmethod
    def copy_limits(
        cls, value: Iterable[BudgetLimit | Mapping[str, Any]] | None
    ) -> tuple[BudgetLimit, ...]:
        if value is None:
            return ()
        if isinstance(value, str | bytes):
            raise ValueError("Budget policy limits must be an iterable of BudgetLimit values.")
        return tuple(_coerce_budget_limit(limit) for limit in value)

    @model_validator(mode="after")
    def validate_unique_limits(self) -> BudgetPolicy:
        seen: set[tuple[str, str, str | None]] = set()
        for limit in self.limits:
            key = (limit.scope, limit.window, limit.key)
            if key in seen:
                raise ValueError("Budget policy contains duplicate scope/window/key limits.")
            seen.add(key)
        return self


class BudgetCheck(BaseModel):
    """Result of evaluating one budget limit."""

    model_config = ConfigDict(extra="forbid")

    scope: BudgetScope
    key: str | None = None
    window: BudgetWindow = "all_time"
    currency: str
    maximum: Decimal = Field(gt=0)
    actual: Decimal = Field(ge=0)
    model_steps: StrictInt = Field(ge=0)
    unpriced_model_steps: StrictInt = Field(ge=0)
    limit_reached: StrictBool
    message: str
    cost_summary: SessionCostSummary

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name).upper()

    @field_validator("maximum", "actual")
    @classmethod
    def validate_decimal(cls, value: Decimal, info) -> Decimal:
        if not value.is_finite():
            raise ValueError(f"{info.field_name} must be finite.")
        return value


class BudgetStore(ABC):
    """Durable source for cross-session budget accounting."""

    @abstractmethod
    async def append_event(self, event: Event) -> None:
        """Observe one cost-bearing runtime event for budget accounting."""

    @abstractmethod
    async def load_events_for_budget(
        self,
        *,
        scope: BudgetScope,
        key: str | None,
        window: BudgetWindow,
    ) -> list[Event]:
        """Return events that contribute to the given budget scope."""


class InMemoryBudgetStore(BudgetStore):
    """In-memory budget store for tests and local apps."""

    def __init__(self) -> None:
        self._events: list[Event] = []
        self._lock = asyncio.Lock()

    async def append_event(self, event: Event) -> None:
        copied = copy_event(event)
        async with self._lock:
            self._events.append(copied)

    async def load_events_for_budget(
        self,
        *,
        scope: BudgetScope,
        key: str | None,
        window: BudgetWindow,
    ) -> list[Event]:
        if window != "all_time":
            raise ValueError(f"Unsupported budget window: {window}")
        async with self._lock:
            events = [copy_event(event) for event in self._events]
        if scope == "app":
            return events
        if scope == "agent":
            budget_key = require_clean_nonblank(key or "", "key")
            return [event for event in events if event.agent_name == budget_key]
        raise ValueError(f"Unsupported budget scope: {scope}")


class SessionBudgetStore(BudgetStore):
    """Budget store backed by the existing durable session event stream."""

    def __init__(self, session_store: SessionStore) -> None:
        if not isinstance(session_store, SessionStore):
            raise TypeError("session_store must be a SessionStore.")
        self._session_store = session_store

    async def append_event(self, event: Event) -> None:
        if type(event) is not Event:
            raise TypeError("event must be an Event.")

    async def load_events_for_budget(
        self,
        *,
        scope: BudgetScope,
        key: str | None,
        window: BudgetWindow,
    ) -> list[Event]:
        if window != "all_time":
            raise ValueError(f"Unsupported budget window: {window}")
        agent_name: str | None = None
        if scope == "agent":
            agent_name = require_clean_nonblank(key or "", "key")
        elif scope != "app":
            raise ValueError(f"Unsupported budget scope: {scope}")

        records = []
        after_sequence: int | None = None
        while True:
            page = await self._session_store.query_events(
                EventQuery(
                    event_type=EventType.MODEL_COMPLETED,
                    agent_name=agent_name,
                    after_sequence=after_sequence,
                    limit=5000,
                )
            )
            if not page:
                break
            records.extend(page)
            after_sequence = page[-1].sequence
            if len(page) < 5000:
                break
        return [copy_event(record.event) for record in records]


def copy_budget_limit(limit: BudgetLimit) -> BudgetLimit:
    return _copy_budget_limit(limit)


def copy_budget_policy(policy: BudgetPolicy | None) -> BudgetPolicy | None:
    if policy is None:
        return None
    if type(policy) is not BudgetPolicy:
        raise TypeError("Budget policy must be a BudgetPolicy instance.")
    return BudgetPolicy(limits=tuple(_copy_budget_limit(limit) for limit in policy.limits))


def budget_limits_for_session(
    *,
    policy: BudgetPolicy | None,
    agent_name: str,
) -> tuple[BudgetLimit, ...]:
    policy = copy_budget_policy(policy)
    if policy is None:
        return ()
    agent_name = require_clean_nonblank(agent_name, "agent_name")
    matched: list[BudgetLimit] = []
    for limit in policy.limits:
        if limit.scope == "app" or (limit.scope == "agent" and limit.key == agent_name):
            matched.append(_copy_budget_limit(limit))
    return tuple(matched)


def budget_check_from_events(
    *,
    limit: BudgetLimit,
    events: list[Event],
    provider_name: str | None = None,
    model: str | None = None,
) -> BudgetCheck:
    if type(limit) is not BudgetLimit:
        raise TypeError("limit must be a BudgetLimit.")
    summary = estimate_session_cost(
        session_id=_budget_summary_id(limit),
        events=events,
        pricing=limit.pricing,
        currency=limit.currency,
    )
    limit_reached = False
    if summary.unpriced_model_steps > 0 and not limit.allow_unpriced:
        limit_reached = True
        message = (
            "Budget cannot be verified because "
            f"{summary.unpriced_model_steps} model step(s) have no matching pricing."
        )
    elif summary.total_cost >= limit.max_estimated_cost:
        limit_reached = True
        message = (
            f"Budget reached: {summary.total_cost} >= {limit.max_estimated_cost} {limit.currency}."
        )
    elif (
        not limit.allow_unpriced
        and (preflight_error := _budget_preflight_error(limit, provider_name, model)) is not None
    ):
        limit_reached = True
        message = preflight_error
    else:
        message = (
            f"Budget checked: {summary.total_cost} < {limit.max_estimated_cost} {limit.currency}."
        )
    return BudgetCheck(
        scope=limit.scope,
        key=limit.key,
        window=limit.window,
        currency=limit.currency,
        maximum=limit.max_estimated_cost,
        actual=summary.total_cost,
        model_steps=summary.model_steps,
        unpriced_model_steps=summary.unpriced_model_steps,
        limit_reached=limit_reached,
        message=message,
        cost_summary=summary,
    )


def budget_check_payload(check: BudgetCheck) -> dict[str, Any]:
    if type(check) is not BudgetCheck:
        raise TypeError("check must be a BudgetCheck.")
    return {
        "scope": check.scope,
        "key": check.key,
        "window": check.window,
        "currency": check.currency,
        "maximum": str(check.maximum),
        "actual": str(check.actual),
        "model_steps": check.model_steps,
        "unpriced_model_steps": check.unpriced_model_steps,
        "limit_reached": check.limit_reached,
        "message": check.message,
        "cost_summary": check.cost_summary.model_dump(mode="json"),
    }


def _copy_budget_limit(limit: BudgetLimit) -> BudgetLimit:
    if type(limit) is not BudgetLimit:
        raise TypeError("Budget policy limits must be BudgetLimit instances.")
    return BudgetLimit(
        scope=limit.scope,
        max_estimated_cost=limit.max_estimated_cost,
        pricing=limit.pricing.model_copy(deep=True),
        currency=limit.currency,
        window=limit.window,
        key=limit.key,
        allow_unpriced=limit.allow_unpriced,
    )


def _coerce_budget_limit(limit: BudgetLimit | Mapping[str, Any]) -> BudgetLimit:
    if type(limit) is BudgetLimit:
        return _copy_budget_limit(limit)
    if isinstance(limit, Mapping):
        return BudgetLimit.model_validate(dict(limit))
    raise TypeError("Budget policy limits must be BudgetLimit instances or mappings.")


def _budget_summary_id(limit: BudgetLimit) -> str:
    key = "all" if limit.key is None else limit.key
    return f"budget:{limit.scope}:{limit.window}:{key}"


def _budget_preflight_error(
    limit: BudgetLimit,
    provider_name: str | None,
    model: str | None,
) -> str | None:
    price = limit.pricing.match_price(provider_name=provider_name, model=model)
    if price is None:
        return f"Budget cannot be verified because {provider_name}/{model} has no matching pricing."
    if price.currency.upper() != limit.currency.upper():
        return (
            "Budget cannot be verified because "
            f"{provider_name}/{model} pricing currency {price.currency} "
            f"does not match requested {limit.currency}."
        )
    return None
