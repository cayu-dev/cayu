from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from decimal import Decimal
from typing import Any, Literal
from uuid import uuid4

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

BudgetScope = Literal["app", "agent", "causal"]
BudgetWindow = Literal["all_time"]
BudgetReservationStatus = Literal["active", "reconciled", "released"]
_TOKENS_PER_MILLION = Decimal("1000000")


class BudgetReservation(BaseModel):
    """Conservative per-model-step reservation configured by the app."""

    model_config = ConfigDict(extra="forbid")

    max_input_tokens: StrictInt = Field(ge=0)
    max_output_tokens: StrictInt = Field(ge=0)
    max_cache_read_input_tokens: StrictInt = Field(default=0, ge=0)
    max_cache_write_input_tokens: StrictInt = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_nonzero_reservation(self) -> BudgetReservation:
        if (
            self.max_input_tokens
            + self.max_output_tokens
            + self.max_cache_read_input_tokens
            + self.max_cache_write_input_tokens
            <= 0
        ):
            raise ValueError("Budget reservation must reserve at least one token.")
        return self


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
    reservation: BudgetReservation | None = None

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
        if self.scope in ("agent", "causal") and self.key is None:
            raise ValueError(f"{self.scope.title()} budget limits require key.")
        if self.reservation is not None and self.allow_unpriced:
            raise ValueError("Budget reservations require priced model usage.")
        return self

    @field_validator("reservation")
    @classmethod
    def copy_reservation(cls, value: BudgetReservation | None) -> BudgetReservation | None:
        if value is None:
            return None
        if type(value) is not BudgetReservation:
            raise TypeError("reservation must be a BudgetReservation.")
        return copy_budget_reservation(value)


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


class BudgetReservationRecord(BaseModel):
    """One reserved budget amount for a model step."""

    model_config = ConfigDict(extra="forbid")

    reservation_id: str = Field(default_factory=lambda: f"bres_{uuid4().hex}")
    scope: BudgetScope
    key: str | None = None
    window: BudgetWindow = "all_time"
    currency: str
    session_id: str
    agent_name: str
    provider_name: str
    model: str
    reserved_amount: Decimal = Field(ge=0)
    actual_amount: Decimal | None = Field(default=None, ge=0)
    status: BudgetReservationStatus = "active"
    reason: str | None = None

    @field_validator(
        "reservation_id",
        "currency",
        "session_id",
        "agent_name",
        "provider_name",
        "model",
    )
    @classmethod
    def validate_nonblank_strings(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("key", "reason")
    @classmethod
    def validate_optional_strings(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("currency")
    @classmethod
    def validate_record_currency(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name).upper()

    @field_validator("reserved_amount", "actual_amount")
    @classmethod
    def validate_record_decimal(cls, value: Decimal | None, info) -> Decimal | None:
        if value is None:
            return None
        if not value.is_finite():
            raise ValueError(f"{info.field_name} must be finite.")
        return value


class BudgetReservationResult(BaseModel):
    """Result of attempting to reserve budget before a model step."""

    model_config = ConfigDict(extra="forbid")

    accepted: StrictBool
    scope: BudgetScope
    key: str | None = None
    window: BudgetWindow = "all_time"
    currency: str
    maximum: Decimal = Field(gt=0)
    requested: Decimal = Field(ge=0)
    actual: Decimal = Field(ge=0)
    message: str
    record: BudgetReservationRecord | None = None

    @field_validator("key")
    @classmethod
    def validate_result_key(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("currency", "message")
    @classmethod
    def validate_result_strings(cls, value: str, info) -> str:
        value = require_clean_nonblank(value, info.field_name)
        return value.upper() if info.field_name == "currency" else value

    @field_validator("maximum", "requested", "actual")
    @classmethod
    def validate_result_decimal(cls, value: Decimal, info) -> Decimal:
        if not value.is_finite():
            raise ValueError(f"{info.field_name} must be finite.")
        return value


class BudgetReconciliation(BaseModel):
    """Result of reconciling a reservation after the model step completes."""

    model_config = ConfigDict(extra="forbid")

    reservation_id: str
    status: BudgetReservationStatus
    reserved_amount: Decimal = Field(ge=0)
    actual_amount: Decimal | None = Field(default=None, ge=0)
    released_amount: Decimal = Field(ge=0)
    reason: str | None = None

    @field_validator("reservation_id")
    @classmethod
    def validate_reconciliation_id(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("reason")
    @classmethod
    def validate_reconciliation_reason(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("reserved_amount", "actual_amount", "released_amount")
    @classmethod
    def validate_reconciliation_decimal(cls, value: Decimal | None, info) -> Decimal | None:
        if value is None:
            return None
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


class BudgetLedger(ABC):
    """Atomic reservation ledger for strict budget enforcement."""

    @abstractmethod
    async def reserve(
        self,
        *,
        limit: BudgetLimit,
        session_id: str,
        agent_name: str,
        provider_name: str,
        model: str,
    ) -> BudgetReservationResult:
        """Reserve budget for one model step if capacity remains."""

    @abstractmethod
    async def reconcile(
        self,
        *,
        reservation_id: str,
        actual_amount: Decimal,
        reason: str | None = None,
    ) -> BudgetReconciliation:
        """Replace an active reservation with the actual charged amount."""

    @abstractmethod
    async def release(
        self,
        *,
        reservation_id: str,
        reason: str,
    ) -> BudgetReconciliation:
        """Release an active reservation without charging it."""


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
        if scope == "causal":
            raise ValueError(
                "InMemoryBudgetStore cannot resolve causal budget scope. "
                "Use SessionBudgetStore for causal budgets."
            )
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
        causal_budget_id: str | None = None
        if scope == "agent":
            agent_name = require_clean_nonblank(key or "", "key")
        elif scope == "causal":
            causal_budget_id = require_clean_nonblank(key or "", "key")
        elif scope != "app":
            raise ValueError(f"Unsupported budget scope: {scope}")

        records = []
        after_sequence: int | None = None
        while True:
            page = await self._session_store.query_events(
                EventQuery(
                    event_type=EventType.MODEL_COMPLETED,
                    causal_budget_id=causal_budget_id,
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


class InMemoryBudgetLedger(BudgetLedger):
    """In-memory reservation ledger for single-process apps and tests."""

    def __init__(self) -> None:
        self._records: dict[str, BudgetReservationRecord] = {}
        self._lock = asyncio.Lock()

    async def reserve(
        self,
        *,
        limit: BudgetLimit,
        session_id: str,
        agent_name: str,
        provider_name: str,
        model: str,
    ) -> BudgetReservationResult:
        request = _budget_reservation_amount(
            limit=limit,
            provider_name=provider_name,
            model=model,
        )
        async with self._lock:
            current = _ledger_used_amount(
                self._records.values(),
                scope=limit.scope,
                key=limit.key,
                window=limit.window,
                currency=limit.currency,
            )
            projected = current + request
            if projected > limit.max_estimated_cost:
                return _reservation_result(
                    limit=limit,
                    accepted=False,
                    requested=request,
                    actual=projected,
                    message=(
                        "Budget reservation failed: "
                        f"{projected} > {limit.max_estimated_cost} {limit.currency}."
                    ),
                )
            record = BudgetReservationRecord(
                scope=limit.scope,
                key=limit.key,
                window=limit.window,
                currency=limit.currency,
                session_id=session_id,
                agent_name=agent_name,
                provider_name=provider_name,
                model=model,
                reserved_amount=request,
            )
            self._records[record.reservation_id] = record
            return _reservation_result(
                limit=limit,
                accepted=True,
                requested=request,
                actual=projected,
                message=(
                    f"Budget reserved: {request} {limit.currency} for {provider_name}/{model}."
                ),
                record=record,
            )

    async def reconcile(
        self,
        *,
        reservation_id: str,
        actual_amount: Decimal,
        reason: str | None = None,
    ) -> BudgetReconciliation:
        reservation_id = require_clean_nonblank(reservation_id, "reservation_id")
        actual_amount = _validate_amount(actual_amount, "actual_amount")
        async with self._lock:
            record = self._active_record(reservation_id)
            reconciled = _reconciled_record(record, actual_amount=actual_amount, reason=reason)
            self._records[reservation_id] = reconciled
            return _reconciliation_from_record(reconciled)

    async def release(
        self,
        *,
        reservation_id: str,
        reason: str,
    ) -> BudgetReconciliation:
        reservation_id = require_clean_nonblank(reservation_id, "reservation_id")
        reason = require_clean_nonblank(reason, "reason")
        async with self._lock:
            record = self._active_record(reservation_id)
            released = record.model_copy(
                update={
                    "status": "released",
                    "reason": reason,
                },
                deep=True,
            )
            self._records[reservation_id] = released
            return _reconciliation_from_record(released)

    def _active_record(self, reservation_id: str) -> BudgetReservationRecord:
        record = self._records.get(reservation_id)
        if record is None:
            raise KeyError(f"Budget reservation not found: {reservation_id}")
        if record.status != "active":
            raise ValueError(f"Budget reservation is not active: {reservation_id}")
        return record


def copy_budget_reservation(reservation: BudgetReservation) -> BudgetReservation:
    if type(reservation) is not BudgetReservation:
        raise TypeError("reservation must be a BudgetReservation.")
    return BudgetReservation(
        max_input_tokens=reservation.max_input_tokens,
        max_output_tokens=reservation.max_output_tokens,
        max_cache_read_input_tokens=reservation.max_cache_read_input_tokens,
        max_cache_write_input_tokens=reservation.max_cache_write_input_tokens,
    )


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
    causal_budget_id: str,
) -> tuple[BudgetLimit, ...]:
    policy = copy_budget_policy(policy)
    if policy is None:
        return ()
    agent_name = require_clean_nonblank(agent_name, "agent_name")
    causal_budget_id = require_clean_nonblank(causal_budget_id, "causal_budget_id")
    matched: list[BudgetLimit] = []
    for limit in policy.limits:
        if (
            limit.scope == "app"
            or (limit.scope == "agent" and limit.key == agent_name)
            or (limit.scope == "causal" and limit.key == causal_budget_id)
        ):
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
        reservation=(
            None if limit.reservation is None else copy_budget_reservation(limit.reservation)
        ),
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


def budget_reservation_payload(result: BudgetReservationResult) -> dict[str, Any]:
    if type(result) is not BudgetReservationResult:
        raise TypeError("result must be a BudgetReservationResult.")
    payload: dict[str, Any] = {
        "accepted": result.accepted,
        "scope": result.scope,
        "key": result.key,
        "window": result.window,
        "currency": result.currency,
        "maximum": str(result.maximum),
        "requested": str(result.requested),
        "actual": str(result.actual),
        "message": result.message,
    }
    if result.record is not None:
        payload["reservation_id"] = result.record.reservation_id
        payload["session_id"] = result.record.session_id
        payload["agent_name"] = result.record.agent_name
        payload["provider_name"] = result.record.provider_name
        payload["model"] = result.record.model
    return payload


def budget_reconciliation_payload(reconciliation: BudgetReconciliation) -> dict[str, Any]:
    if type(reconciliation) is not BudgetReconciliation:
        raise TypeError("reconciliation must be a BudgetReconciliation.")
    return {
        "reservation_id": reconciliation.reservation_id,
        "status": reconciliation.status,
        "reserved_amount": str(reconciliation.reserved_amount),
        "actual_amount": (
            None if reconciliation.actual_amount is None else str(reconciliation.actual_amount)
        ),
        "released_amount": str(reconciliation.released_amount),
        "reason": reconciliation.reason,
    }


def budget_actual_cost_for_event(*, limit: BudgetLimit, event: Event) -> Decimal:
    if type(limit) is not BudgetLimit:
        raise TypeError("limit must be a BudgetLimit.")
    if type(event) is not Event:
        raise TypeError("event must be an Event.")
    summary = estimate_session_cost(
        session_id=_budget_summary_id(limit),
        events=[event],
        pricing=limit.pricing,
        currency=limit.currency,
    )
    if summary.unpriced_model_steps > 0:
        raise ValueError("Cannot reconcile budget reservation from unpriced model usage.")
    return summary.total_cost


def _budget_reservation_amount(
    *,
    limit: BudgetLimit,
    provider_name: str,
    model: str,
) -> Decimal:
    if limit.reservation is None:
        raise ValueError("Budget limit does not define a reservation policy.")
    price = limit.pricing.match_price(provider_name=provider_name, model=model)
    if price is None:
        raise ValueError(f"Budget reservation cannot be priced for {provider_name}/{model}.")
    if price.currency.upper() != limit.currency.upper():
        raise ValueError(
            f"Budget reservation currency {price.currency} does not match {limit.currency}."
        )
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
    reservation = limit.reservation
    return (
        _token_cost(reservation.max_input_tokens, price.input_per_million)
        + _token_cost(reservation.max_output_tokens, price.output_per_million)
        + _token_cost(reservation.max_cache_read_input_tokens, cache_read_price)
        + _token_cost(reservation.max_cache_write_input_tokens, cache_write_price)
    )


def _token_cost(tokens: int, price_per_million: Decimal) -> Decimal:
    return Decimal(tokens) * price_per_million / _TOKENS_PER_MILLION


def _ledger_used_amount(
    records: Iterable[BudgetReservationRecord],
    *,
    scope: BudgetScope,
    key: str | None,
    window: BudgetWindow,
    currency: str,
) -> Decimal:
    total = Decimal("0")
    for record in records:
        if (
            record.scope != scope
            or record.key != key
            or record.window != window
            or record.currency.upper() != currency.upper()
        ):
            continue
        if record.status == "active":
            total += record.reserved_amount
        elif record.status == "reconciled":
            total += record.actual_amount or Decimal("0")
    return total


def _reservation_result(
    *,
    limit: BudgetLimit,
    accepted: bool,
    requested: Decimal,
    actual: Decimal,
    message: str,
    record: BudgetReservationRecord | None = None,
) -> BudgetReservationResult:
    return BudgetReservationResult(
        accepted=accepted,
        scope=limit.scope,
        key=limit.key,
        window=limit.window,
        currency=limit.currency,
        maximum=limit.max_estimated_cost,
        requested=requested,
        actual=actual,
        message=message,
        record=record,
    )


def _reconciled_record(
    record: BudgetReservationRecord,
    *,
    actual_amount: Decimal,
    reason: str | None,
) -> BudgetReservationRecord:
    return record.model_copy(
        update={
            "actual_amount": actual_amount,
            "status": "reconciled",
            "reason": reason,
        },
        deep=True,
    )


def _reconciliation_from_record(record: BudgetReservationRecord) -> BudgetReconciliation:
    actual_amount = record.actual_amount
    released_amount = Decimal("0")
    if record.status == "released":
        released_amount = record.reserved_amount
    elif actual_amount is not None and record.reserved_amount > actual_amount:
        released_amount = record.reserved_amount - actual_amount
    return BudgetReconciliation(
        reservation_id=record.reservation_id,
        status=record.status,
        reserved_amount=record.reserved_amount,
        actual_amount=actual_amount,
        released_amount=released_amount,
        reason=record.reason,
    )


def _validate_amount(value: Decimal, field_name: str) -> Decimal:
    if type(value) is not Decimal:
        value = Decimal(str(value))
    if not value.is_finite() or value < 0:
        raise ValueError(f"{field_name} must be a finite non-negative Decimal.")
    return value
