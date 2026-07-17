from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any, Literal, NamedTuple, cast
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

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
from cayu.core.billing import (
    UNRESOLVED_BILLING_IDENTITY,
    BillingIdentity,
    BillingIdentityState,
    PricingContext,
    ResolvedBillingIdentity,
    billing_identity_value,
    completed_billing_identity,
)
from cayu.core.events import Event, EventType, copy_event
from cayu.runtime.costs import (
    CostLineItem,
    PriceBook,
    Provenance,
    SessionCostSummary,
    _normalize_provider,
    _resolve_price_book,
    _ResolvedPrice,
    estimate_session_cost,
    resolve_price_book,
)

BudgetScope = Literal["app", "agent", "causal", "session", "run"]
BudgetWindowKind = Literal["all_time", "rolling", "calendar"]
BudgetCalendarPeriod = Literal["day", "week", "month"]
BudgetAction = Literal["interrupt", "notify"]
BudgetReservationStatus = Literal["active", "reconciled", "released"]
_TOKENS_PER_MILLION = Decimal("1000000")
DEFAULT_RESERVATION_TTL_SECONDS = 3600
_ALL_TIME_WINDOW = "all_time"
_ROLLING_PREFIX = "rolling:"
_ROLLING_SUFFIX = "s"
_CALENDAR_PREFIX = "calendar:"


class BudgetWindow(BaseModel):
    """Time window used when selecting events for budget accounting."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: BudgetWindowKind = "all_time"
    duration_seconds: StrictInt | None = Field(default=None, ge=1)
    period: BudgetCalendarPeriod | None = None
    timezone: str | None = None

    @classmethod
    def all_time(cls) -> BudgetWindow:
        return cls(kind="all_time")

    @classmethod
    def rolling(cls, *, seconds: int) -> BudgetWindow:
        return cls(kind="rolling", duration_seconds=seconds)

    @classmethod
    def calendar(
        cls,
        *,
        period: BudgetCalendarPeriod,
        timezone: str = "UTC",
    ) -> BudgetWindow:
        return cls(kind="calendar", period=period, timezone=timezone)

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        timezone = require_clean_nonblank(value, info.field_name)
        try:
            ZoneInfo(timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Unknown budget window timezone: {timezone}") from exc
        return timezone

    @model_validator(mode="after")
    def validate_window_fields(self) -> BudgetWindow:
        if self.kind == "all_time":
            if (
                self.duration_seconds is not None
                or self.period is not None
                or self.timezone is not None
            ):
                raise ValueError("All-time budget windows must not set window details.")
        elif self.kind == "rolling":
            if self.duration_seconds is None:
                raise ValueError("Rolling budget windows require duration_seconds.")
            if self.period is not None or self.timezone is not None:
                raise ValueError("Rolling budget windows must not set calendar details.")
        elif self.kind == "calendar":
            if self.duration_seconds is not None:
                raise ValueError("Calendar budget windows must not set duration_seconds.")
            if self.period is None:
                raise ValueError("Calendar budget windows require period.")
            if self.timezone is None:
                raise ValueError("Calendar budget windows require timezone.")
        return self

    @property
    def storage_key(self) -> str:
        if self.kind == "all_time":
            return "all_time"
        if self.kind == "rolling":
            return f"rolling:{self.duration_seconds}s"
        return f"calendar:{self.period}:{self.timezone}"

    def since(self, now: datetime | None = None) -> datetime | None:
        return self.bounds(now=now)[0]

    def until(self, now: datetime | None = None) -> datetime | None:
        return self.bounds(now=now)[1]

    def bounds(self, now: datetime | None = None) -> tuple[datetime | None, datetime | None]:
        reference = datetime.now(UTC) if now is None else _utc_datetime(now, "now")
        if self.kind == "all_time":
            return None, None
        if self.kind == "rolling":
            return reference - timedelta(seconds=self.duration_seconds or 0), reference
        return _calendar_window_bounds(
            reference,
            period=self.period or "day",
            timezone=self.timezone or "UTC",
        )


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

    scope: BudgetScope = "session"
    max_estimated_cost: Decimal = Field(gt=0)
    pricing: PriceBook
    currency: str = "USD"
    window: BudgetWindow = Field(default_factory=BudgetWindow.all_time)
    key: str | None = None
    allow_unpriced: StrictBool = False
    reservation: BudgetReservation | None = None
    action: BudgetAction = "interrupt"

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

    @field_validator("window", mode="before")
    @classmethod
    def copy_window(cls, value) -> BudgetWindow:
        return copy_budget_window(value)

    @field_validator("max_estimated_cost")
    @classmethod
    def validate_cost(cls, value: Decimal, info) -> Decimal:
        if not value.is_finite():
            raise ValueError(f"{info.field_name} must be finite.")
        return value

    @model_validator(mode="after")
    def validate_scope_key(self) -> BudgetLimit:
        if self.scope in ("app", "session", "run") and self.key is not None:
            raise ValueError(f"{self.scope.title()} budget limits must not set key.")
        if self.scope in ("agent", "causal") and self.key is None:
            raise ValueError(f"{self.scope.title()} budget limits require key.")
        if self.reservation is not None and self.allow_unpriced:
            raise ValueError("Budget reservations require priced model usage.")
        if self.reservation is not None and self.action != "interrupt":
            raise ValueError("Budget reservations require action='interrupt'.")
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
        seen: set[tuple[str, str, str | None, str, Decimal]] = set()
        for limit in self.limits:
            if limit.scope in {"session", "run"}:
                raise ValueError(
                    f"{limit.scope.title()} budget limits are request-scoped, "
                    "not app policy limits."
                )
            key = (
                limit.scope,
                limit.window.storage_key,
                limit.key,
                limit.action,
                limit.max_estimated_cost,
            )
            if key in seen:
                raise ValueError("Budget policy contains duplicate scope/window/key limits.")
            seen.add(key)
        return self


class BudgetCheck(BaseModel):
    """Result of evaluating one budget limit."""

    model_config = ConfigDict(extra="forbid")

    scope: BudgetScope
    key: str | None = None
    window: BudgetWindow = Field(default_factory=BudgetWindow.all_time)
    currency: str
    maximum: Decimal = Field(gt=0)
    actual: Decimal = Field(ge=0)
    action: BudgetAction = "interrupt"
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

    @field_validator("window", mode="before")
    @classmethod
    def copy_window(cls, value) -> BudgetWindow:
        return copy_budget_window(value)

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
    window: BudgetWindow = Field(default_factory=BudgetWindow.all_time)
    currency: str
    session_id: str
    agent_name: str
    provider_name: str
    model: str
    billing_identity: BillingIdentity | None = None
    reserved_amount: Decimal = Field(ge=0)
    actual_amount: Decimal | None = Field(default=None, ge=0)
    status: BudgetReservationStatus = "active"
    reason: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

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

    @field_validator("window", mode="before")
    @classmethod
    def copy_window(cls, value) -> BudgetWindow:
        return copy_budget_window(value)

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

    @field_validator("created_at", "updated_at")
    @classmethod
    def validate_record_timestamp(cls, value: datetime, info) -> datetime:
        return _utc_datetime(value, info.field_name)


class BudgetReservationResult(BaseModel):
    """Result of attempting to reserve budget before a model step."""

    model_config = ConfigDict(extra="forbid")

    accepted: StrictBool
    scope: BudgetScope
    key: str | None = None
    window: BudgetWindow = Field(default_factory=BudgetWindow.all_time)
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

    @field_validator("window", mode="before")
    @classmethod
    def copy_window(cls, value) -> BudgetWindow:
        return copy_budget_window(value)

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
    billing_identity: BillingIdentity | None = None
    pricing_provider_name: str | None = None
    pricing_model: str | None = None
    pricing_match: Literal["exact", "prefix", "resource_mapping"] | None = None
    pricing_provenance: Provenance | None = None
    pricing_effective_from: date | None = None
    pricing_effective_through: date | None = None
    pricing_tier_max_input_tokens: StrictInt | None = Field(default=None, gt=0)

    @field_validator("reservation_id")
    @classmethod
    def validate_reconciliation_id(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("reason", "pricing_provider_name", "pricing_model")
    @classmethod
    def validate_reconciliation_reason(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_pricing_evidence(self) -> BudgetReconciliation:
        identity = (
            self.pricing_provider_name,
            self.pricing_model,
            self.pricing_match,
            self.pricing_provenance,
        )
        evidence = (
            *identity,
            self.pricing_effective_from,
            self.pricing_effective_through,
            self.pricing_tier_max_input_tokens,
        )
        if any(value is not None for value in evidence) and not all(
            value is not None for value in identity
        ):
            raise ValueError("reconciliation pricing identity and provenance must be complete")
        if (
            self.pricing_effective_from is not None
            and self.pricing_effective_through is not None
            and self.pricing_effective_from > self.pricing_effective_through
        ):
            raise ValueError("reconciliation pricing effective window is reversed")
        return self

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
        """Observe one cost-bearing event, idempotently by session and event id.

        Runtime delivery is at-least-once across crashes. Implementations must
        return without adding a second charge when the same immutable event is
        retried, and reject a conflicting event with the same identity.
        """

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

    @property
    def reservation_ttl_seconds(self) -> int | None:
        """Active-reservation lease duration, or ``None`` for non-expiring ledgers.

        Custom ledgers remain non-expiring by default. A ledger that advertises
        a finite TTL must also implement :meth:`heartbeat` so the runtime can
        keep live provider calls reserved.
        """

        return None

    async def heartbeat(self, *, reservation_id: str) -> bool:
        """Renew one active reservation lease.

        Return ``False`` when the reservation is terminal or its lease already
        expired. Unknown reservation ids raise ``KeyError``. The default is
        intentionally unsupported because the base ledger advertises no TTL.
        """

        raise NotImplementedError("This budget ledger does not use expiring reservations.")

    @abstractmethod
    async def reserve(
        self,
        *,
        limit: BudgetLimit,
        session_id: str,
        agent_name: str,
        provider_name: str,
        model: str,
        billing_identity: BillingIdentity | None = None,
    ) -> BudgetReservationResult:
        """Reserve budget for one provider dispatch if capacity remains."""

    @abstractmethod
    async def reconcile(
        self,
        *,
        reservation_id: str,
        actual_amount: Decimal,
        reason: str | None = None,
        occurred_at: datetime | None = None,
        billing_identity: BillingIdentity | None = None,
    ) -> BudgetReconciliation:
        """Replace a reservation with the charged amount.

        Implementations must return the existing result for an identical retry
        without changing its accounting timestamp, and reject a conflicting
        terminal outcome for the same reservation.
        """

    @abstractmethod
    async def release(
        self,
        *,
        reservation_id: str,
        reason: str,
    ) -> BudgetReconciliation:
        """Release an active reservation without charging it.

        Implementations must return the existing result for an identical retry
        and reject a conflicting terminal outcome for the same reservation.
        """


class InMemoryBudgetStore(BudgetStore):
    """In-memory budget store for tests and local apps."""

    def __init__(self) -> None:
        self._events: list[Event] = []
        self._events_by_id: dict[tuple[str, str], Event] = {}
        self._lock = asyncio.Lock()

    async def append_event(self, event: Event) -> None:
        copied = copy_event(event)
        async with self._lock:
            identity = (copied.session_id, copied.id)
            existing = self._events_by_id.get(identity)
            if existing is not None:
                if existing != copied:
                    raise ValueError(
                        "Budget event identity was reused with conflicting contents: "
                        f"{copied.session_id}/{copied.id}"
                    )
                return
            self._events_by_id[identity] = copied
            self._events.append(copied)

    async def load_events_for_budget(
        self,
        *,
        scope: BudgetScope,
        key: str | None,
        window: BudgetWindow,
    ) -> list[Event]:
        async with self._lock:
            events = [copy_event(event) for event in self._events]
        events = events_for_budget_window(events, window)
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

    def __init__(self, session_store: Any) -> None:
        from cayu.runtime.sessions import SessionStore

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
        from cayu.runtime.sessions import EventQuery

        window = copy_budget_window(window)
        since, until = window.bounds()
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
                    since=since,
                    until=until,
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

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] | None = None,
        reservation_ttl_seconds: int | None = DEFAULT_RESERVATION_TTL_SECONDS,
    ) -> None:
        self._records: dict[str, BudgetReservationRecord] = {}
        self._lock = asyncio.Lock()
        self._clock = _clock_or_utc_now(clock)
        self._reservation_ttl_seconds = _validate_reservation_ttl(reservation_ttl_seconds)

    @property
    def reservation_ttl_seconds(self) -> int | None:
        return self._reservation_ttl_seconds

    async def reserve(
        self,
        *,
        limit: BudgetLimit,
        session_id: str,
        agent_name: str,
        provider_name: str,
        model: str,
        billing_identity: BillingIdentity | None = None,
    ) -> BudgetReservationResult:
        async with self._lock:
            now = self._clock()
            request = _budget_reservation_amount(
                limit=limit,
                provider_name=provider_name,
                model=model,
                effective_at=now,
                billing_identity=billing_identity,
            )
            self._reap_expired_unlocked(now, limit=limit)
            current = _ledger_used_amount(
                self._records.values(),
                limit=limit,
                now=now,
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
                billing_identity=billing_identity,
                reserved_amount=request,
                created_at=now,
                updated_at=now,
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

    async def heartbeat(self, *, reservation_id: str) -> bool:
        reservation_id = require_clean_nonblank(reservation_id, "reservation_id")
        now = self._clock()
        async with self._lock:
            record = self._records.get(reservation_id)
            if record is None:
                raise KeyError(f"Budget reservation not found: {reservation_id}")
            if record.status != "active" or _reservation_is_expired(
                record,
                now=now,
                ttl_seconds=self._reservation_ttl_seconds,
            ):
                return False
            self._records[reservation_id] = record.model_copy(
                update={"updated_at": now},
                deep=True,
            )
            return True

    async def reconcile(
        self,
        *,
        reservation_id: str,
        actual_amount: Decimal,
        reason: str | None = None,
        occurred_at: datetime | None = None,
        billing_identity: BillingIdentity | None = None,
    ) -> BudgetReconciliation:
        reservation_id = require_clean_nonblank(reservation_id, "reservation_id")
        actual_amount = _validate_amount(actual_amount, "actual_amount")
        reconciled_at = _utc_datetime(occurred_at, "occurred_at") if occurred_at else self._clock()
        async with self._lock:
            record = self._reconcilable_record(reservation_id)
            reconciled = _reconciled_record(
                record,
                actual_amount=actual_amount,
                reason=reason,
                updated_at=reconciled_at,
                billing_identity=billing_identity,
            )
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
        released_at = self._clock()
        async with self._lock:
            record = self._releasable_record(reservation_id)
            released = _released_record(
                record,
                reason=reason,
                updated_at=released_at,
            )
            self._records[reservation_id] = released
            return _reconciliation_from_record(released)

    def _reap_expired_unlocked(self, now: datetime, *, limit: BudgetLimit) -> None:
        if self._reservation_ttl_seconds is None:
            return
        for reservation_id, record in self._records.items():
            if (
                record.status != "active"
                or not _reservation_matches_limit(record, limit)
                or not _reservation_is_expired(
                    record,
                    now=now,
                    ttl_seconds=self._reservation_ttl_seconds,
                )
            ):
                continue
            self._records[reservation_id] = record.model_copy(
                update={
                    "status": "released",
                    "reason": _expired_reservation_reason(self._reservation_ttl_seconds),
                    "updated_at": now,
                },
                deep=True,
            )

    def _active_record(self, reservation_id: str) -> BudgetReservationRecord:
        record = self._records.get(reservation_id)
        if record is None:
            raise KeyError(f"Budget reservation not found: {reservation_id}")
        if record.status != "active":
            raise ValueError(f"Budget reservation is not active: {reservation_id}")
        return record

    def _releasable_record(self, reservation_id: str) -> BudgetReservationRecord:
        record = self._records.get(reservation_id)
        if record is None:
            raise KeyError(f"Budget reservation not found: {reservation_id}")
        if record.status in {"active", "released"}:
            return record
        raise ValueError(f"Budget reservation is not active: {reservation_id}")

    def _reconcilable_record(self, reservation_id: str) -> BudgetReservationRecord:
        record = self._records.get(reservation_id)
        if record is None:
            raise KeyError(f"Budget reservation not found: {reservation_id}")
        if record.status in {"active", "reconciled"}:
            return record
        if record.status == "released" and _is_expired_reservation_reason(record.reason):
            # Reaped by the TTL while still in flight (a long step or a wall-clock jump).
            # Reconcile it anyway so the actual spend is recorded rather than crashing the
            # billed run and silently undercounting the shared budget window.
            return record
        raise ValueError(f"Budget reservation is not active: {reservation_id}")


def copy_budget_reservation(reservation: BudgetReservation) -> BudgetReservation:
    if type(reservation) is not BudgetReservation:
        raise TypeError("reservation must be a BudgetReservation.")
    return BudgetReservation(
        max_input_tokens=reservation.max_input_tokens,
        max_output_tokens=reservation.max_output_tokens,
        max_cache_read_input_tokens=reservation.max_cache_read_input_tokens,
        max_cache_write_input_tokens=reservation.max_cache_write_input_tokens,
    )


def copy_budget_window(value: BudgetWindow | Mapping[str, Any] | str | None) -> BudgetWindow:
    if value is None:
        return BudgetWindow.all_time()
    if type(value) is BudgetWindow:
        return value.model_copy(deep=True)
    if isinstance(value, str):
        return _budget_window_from_string(value)
    if isinstance(value, Mapping):
        return BudgetWindow.model_validate(dict(value))
    raise TypeError("Budget window must be a BudgetWindow, mapping, string, or None.")


def events_for_budget_window(
    events: Iterable[Event],
    window: BudgetWindow | Mapping[str, Any] | str | None,
    *,
    now: datetime | None = None,
) -> list[Event]:
    window = copy_budget_window(window)
    since, until = window.bounds(now=now)
    copied = [copy_event(event) for event in events]
    if since is None and until is None:
        return copied
    return [event for event in copied if _event_in_window(event, since=since, until=until)]


def copy_budget_limit(limit: BudgetLimit) -> BudgetLimit:
    return _copy_budget_limit(limit)


def copy_budget_policy(policy: BudgetPolicy | None) -> BudgetPolicy | None:
    if policy is None:
        return None
    if type(policy) is not BudgetPolicy:
        raise TypeError("Budget policy must be a BudgetPolicy instance.")
    return BudgetPolicy(limits=tuple(_copy_budget_limit(limit) for limit in policy.limits))


def copy_budget_limits(
    limits: Iterable[BudgetLimit | Mapping[str, Any]] | None,
    *,
    field_name: str = "budget_limits",
) -> tuple[BudgetLimit, ...]:
    if limits is None:
        return ()
    if isinstance(limits, str | bytes):
        raise ValueError(f"{field_name} must be an iterable of BudgetLimit values.")
    return tuple(_coerce_budget_limit(limit) for limit in limits)


def copy_request_budget_limits(
    limits: Iterable[BudgetLimit | Mapping[str, Any]] | None,
) -> tuple[BudgetLimit, ...]:
    """Copy per-request budget limits, allowing reservations on shared scopes.

    Request limits scoped ``app``/``agent``/``causal`` may configure a
    reservation: those budgets are shared across sessions, so the runtime
    routes them through the atomic budget ledger before each model step,
    keeping concurrent sessions from jointly overshooting the limit.

    ``session``/``run`` scoped limits must not reserve. They are accounted
    from a single session's own event stream, and a session executes its
    model steps sequentially, so there is no cross-session race to close.
    Without a reservation the residual race for those scopes is only the
    usage of the one model step that is in flight when the read-then-act
    check passes.
    """
    copied = copy_budget_limits(limits, field_name="budget_limits")
    for limit in copied:
        if limit.reservation is not None and limit.scope in ("session", "run"):
            raise ValueError(
                "Request budget limits must not use reservations for "
                f"{limit.scope!r} scope; reservations require a shared "
                "budget scope (app, agent, or causal)."
            )
    return copied


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


def request_budget_limits_for_session(
    *,
    limits: Iterable[BudgetLimit | Mapping[str, Any]] | None,
    agent_name: str,
    causal_budget_id: str,
) -> tuple[BudgetLimit, ...]:
    copied = copy_request_budget_limits(limits)
    agent_name = require_clean_nonblank(agent_name, "agent_name")
    causal_budget_id = require_clean_nonblank(causal_budget_id, "causal_budget_id")
    for limit in copied:
        if limit.scope == "agent" and limit.key != agent_name:
            raise ValueError(
                f"Request agent budget limit key {limit.key!r} does not match "
                f"session agent {agent_name!r}."
            )
        if limit.scope == "causal" and limit.key != causal_budget_id:
            raise ValueError(
                f"Request causal budget limit key {limit.key!r} does not match "
                f"session causal_budget_id {causal_budget_id!r}."
            )
    return copied


def budget_check_from_events(
    *,
    limit: BudgetLimit,
    events: list[Event],
    provider_name: str | None = None,
    model: str | None = None,
    billing_identity_state: BillingIdentityState = UNRESOLVED_BILLING_IDENTITY,
    effective_at: datetime | None = None,
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
        and (
            preflight_error := _budget_preflight_error(
                limit,
                provider_name,
                model,
                billing_identity_state=billing_identity_state,
                effective_at=effective_at,
            )
        )
        is not None
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
        action=limit.action,
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
        "window": check.window.storage_key,
        "window_details": check.window.model_dump(mode="json"),
        "currency": check.currency,
        "maximum": str(check.maximum),
        "actual": str(check.actual),
        "action": check.action,
        "model_steps": check.model_steps,
        "unpriced_model_steps": check.unpriced_model_steps,
        "limit_reached": check.limit_reached,
        "message": check.message,
        "cost_summary": check.cost_summary.model_dump(mode="json"),
    }


def _copy_budget_limit(limit: BudgetLimit) -> BudgetLimit:
    if type(limit) is not BudgetLimit:
        raise TypeError("Budget limits must be BudgetLimit instances.")
    return BudgetLimit(
        scope=limit.scope,
        max_estimated_cost=limit.max_estimated_cost,
        pricing=limit.pricing.model_copy(deep=True),
        currency=limit.currency,
        window=limit.window,
        key=limit.key,
        allow_unpriced=limit.allow_unpriced,
        action=limit.action,
        reservation=(
            None if limit.reservation is None else copy_budget_reservation(limit.reservation)
        ),
    )


def _coerce_budget_limit(limit: BudgetLimit | Mapping[str, Any]) -> BudgetLimit:
    if type(limit) is BudgetLimit:
        return _copy_budget_limit(limit)
    if isinstance(limit, Mapping):
        return BudgetLimit.model_validate(dict(limit))
    raise TypeError("Budget limits must be BudgetLimit instances or mappings.")


def _budget_summary_id(limit: BudgetLimit) -> str:
    key = "all" if limit.key is None else limit.key
    return f"budget:{limit.scope}:{limit.window.storage_key}:{key}"


def _budget_preflight_error(
    limit: BudgetLimit,
    provider_name: str | None,
    model: str | None,
    *,
    billing_identity_state: BillingIdentityState,
    effective_at: datetime | None,
) -> str | None:
    billing_identity = billing_identity_value(billing_identity_state)
    reference = (
        datetime.now(UTC) if effective_at is None else _utc_datetime(effective_at, "effective_at")
    )
    for pricing_context in _budget_pricing_contexts(billing_identity):
        resolution = _resolve_price_book(
            limit.pricing,
            provider_name=provider_name,
            model=model,
            input_tokens=0,
            effective_on=reference.astimezone(UTC).date(),
            billing_identity=billing_identity,
            pricing_context=pricing_context,
        )
        price = resolution.resolved
        if price is None:
            # Contextual dimensions are request-specific. The run loop repeats
            # preflight with the provider-resolved identity immediately before
            # reserving and dispatching.
            if not isinstance(
                billing_identity_state, ResolvedBillingIdentity
            ) and has_deferred_contextual_price(
                limit.pricing,
                provider_name=provider_name,
                model=model,
            ):
                return None
            reason = resolution.missing_reason or "no matching model pricing"
            return f"Budget cannot be verified because {provider_name}/{model}: {reason}."
        if price.currency.upper() != limit.currency.upper():
            return (
                "Budget cannot be verified because "
                f"{provider_name}/{model} pricing currency {price.currency} "
                f"does not match requested {limit.currency}."
            )
    return None


def has_deferred_contextual_price(
    pricing: PriceBook,
    *,
    provider_name: str | None,
    model: str | None,
) -> bool:
    if provider_name is None or model is None:
        return False
    # A directly matching context-free price belongs to the declared provider and
    # does not need deferred resolution. This also prevents an unrelated contextual
    # row with the same model identifier from shadowing it.
    if (
        pricing._resolve_match(
            provider_name=provider_name,
            model=model,
            billing_identity=None,
        )
        is not None
    ):
        return False
    pricing_model = next(
        (
            mapping.pricing_model
            for mapping in pricing.resource_mappings
            if _normalize_provider(mapping.provider_name) == _normalize_provider(provider_name)
            and mapping.resource_id == model
        ),
        model,
    )
    return any(
        _normalize_provider(price.provider_name) == _normalize_provider(provider_name)
        and price.model == pricing_model
        and price.pricing_context is not None
        for price in pricing.prices
    )


def budget_reservation_payload(result: BudgetReservationResult) -> dict[str, Any]:
    if type(result) is not BudgetReservationResult:
        raise TypeError("result must be a BudgetReservationResult.")
    payload: dict[str, Any] = {
        "accepted": result.accepted,
        "scope": result.scope,
        "key": result.key,
        "window": result.window.storage_key,
        "window_details": result.window.model_dump(mode="json"),
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
        if result.record.billing_identity is not None:
            payload["billing_identity"] = result.record.billing_identity.model_dump(mode="json")
    return payload


def budget_reconciliation_payload(reconciliation: BudgetReconciliation) -> dict[str, Any]:
    if type(reconciliation) is not BudgetReconciliation:
        raise TypeError("reconciliation must be a BudgetReconciliation.")
    pricing = None
    if reconciliation.pricing_provider_name is not None:
        pricing = {
            "provider_name": reconciliation.pricing_provider_name,
            "model": reconciliation.pricing_model,
            "match": reconciliation.pricing_match,
            "provenance": reconciliation.pricing_provenance.model_dump(mode="json")
            if reconciliation.pricing_provenance is not None
            else None,
            "effective_from": (
                None
                if reconciliation.pricing_effective_from is None
                else reconciliation.pricing_effective_from.isoformat()
            ),
            "effective_through": (
                None
                if reconciliation.pricing_effective_through is None
                else reconciliation.pricing_effective_through.isoformat()
            ),
            "tier_max_input_tokens": reconciliation.pricing_tier_max_input_tokens,
        }
    return {
        "reservation_id": reconciliation.reservation_id,
        "status": reconciliation.status,
        "reserved_amount": str(reconciliation.reserved_amount),
        "actual_amount": (
            None if reconciliation.actual_amount is None else str(reconciliation.actual_amount)
        ),
        "released_amount": str(reconciliation.released_amount),
        "reason": reconciliation.reason,
        "pricing": pricing,
        "billing_identity": (
            None
            if reconciliation.billing_identity is None
            else reconciliation.billing_identity.model_dump(mode="json")
        ),
    }


class _BudgetActualCost(NamedTuple):
    amount: Decimal
    line_item: CostLineItem


def budget_actual_cost_for_event(*, limit: BudgetLimit, event: Event) -> _BudgetActualCost:
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
    if len(summary.line_items) != 1:
        raise ValueError("Budget reconciliation requires exactly one priced model step.")
    return _BudgetActualCost(summary.total_cost, summary.line_items[0])


def budget_reconciliation_with_pricing(
    reconciliation: BudgetReconciliation,
    line_item: CostLineItem,
) -> BudgetReconciliation:
    if type(reconciliation) is not BudgetReconciliation:
        raise TypeError("reconciliation must be a BudgetReconciliation.")
    if type(line_item) is not CostLineItem or not line_item.priced:
        raise TypeError("line_item must be a priced CostLineItem.")
    return BudgetReconciliation.model_validate(
        {
            **reconciliation.model_dump(mode="python"),
            "pricing_provider_name": line_item.pricing_provider_name,
            "pricing_model": line_item.pricing_model,
            "pricing_match": line_item.pricing_match,
            "pricing_provenance": (
                None
                if line_item.pricing_provenance is None
                else line_item.pricing_provenance.model_copy(deep=True)
            ),
            "pricing_effective_from": line_item.pricing_effective_from,
            "pricing_effective_through": line_item.pricing_effective_through,
            "pricing_tier_max_input_tokens": line_item.pricing_tier_max_input_tokens,
            "billing_identity": line_item.billing_identity,
        }
    )


def _budget_reservation_amount(
    *,
    limit: BudgetLimit,
    provider_name: str,
    model: str,
    effective_at: datetime,
    billing_identity: BillingIdentity | None = None,
) -> Decimal:
    if limit.reservation is None:
        raise ValueError("Budget limit does not define a reservation policy.")
    reservation = limit.reservation
    reserved_input_tokens = (
        reservation.max_input_tokens
        + reservation.max_cache_read_input_tokens
        + reservation.max_cache_write_input_tokens
    )
    amounts: list[Decimal] = []
    for pricing_context in _budget_pricing_contexts(billing_identity):
        price = budget_price(
            limit,
            provider_name=provider_name,
            model=model,
            input_tokens=reserved_input_tokens,
            effective_at=effective_at,
            billing_identity=billing_identity,
            pricing_context=pricing_context,
        )
        if price is None:
            raise ValueError(f"Budget reservation cannot be priced for {provider_name}/{model}.")
        if price.currency.upper() != limit.currency.upper():
            raise ValueError(
                f"Budget reservation currency {price.currency} does not match {limit.currency}."
            )
        cache_read_price = (
            price.input_per_million
            if price.cache_read_input_per_million is None
            else price.cache_read_input_per_million
        )
        if not price.requires_cache_write_ttls:
            cache_write_price = (
                price.cache_write_input_per_million
                if price.cache_write_input_per_million is not None
                else price.input_per_million
            )
        else:
            supported_rates = {
                "5m": price.cache_write_5m_per_million,
                "1h": price.cache_write_1h_per_million,
            }
            cache_write_rates = [supported_rates[ttl] for ttl in price.price.cache_write_ttls]
            if reservation.max_cache_write_input_tokens and (
                not cache_write_rates or any(rate is None for rate in cache_write_rates)
            ):
                raise ValueError("Budget reservation cannot price the declared cache-write TTLs.")
            cache_write_price = max(
                (rate for rate in cache_write_rates if rate is not None),
                default=Decimal("0"),
            )
        amounts.append(
            _token_cost(reservation.max_input_tokens, price.input_per_million)
            + _token_cost(reservation.max_output_tokens, price.output_per_million)
            + _token_cost(reservation.max_cache_read_input_tokens, cache_read_price)
            + _token_cost(reservation.max_cache_write_input_tokens, cache_write_price)
        )
    return max(amounts)


def _budget_pricing_contexts(
    billing_identity: BillingIdentity | None,
) -> tuple[PricingContext | None, ...]:
    """Return every pricing outcome established before provider dispatch."""

    if billing_identity is None:
        return (None,)
    return billing_identity.pricing_contexts or (None,)


def _token_cost(tokens: int, price_per_million: Decimal) -> Decimal:
    return Decimal(tokens) * price_per_million / _TOKENS_PER_MILLION


def budget_price(
    limit: BudgetLimit,
    *,
    provider_name: str | None,
    model: str | None,
    input_tokens: int = 0,
    effective_at: datetime | None = None,
    billing_identity: BillingIdentity | None = None,
    pricing_context: PricingContext | None = None,
) -> _ResolvedPrice | None:
    effective_at = (
        datetime.now(UTC) if effective_at is None else _utc_datetime(effective_at, "effective_at")
    )
    return resolve_price_book(
        limit.pricing,
        provider_name=provider_name,
        model=model,
        input_tokens=input_tokens,
        effective_on=effective_at.astimezone(UTC).date(),
        billing_identity=billing_identity,
        pricing_context=pricing_context,
    )


def _ledger_used_amount(
    records: Iterable[BudgetReservationRecord],
    *,
    limit: BudgetLimit,
    now: datetime | None = None,
) -> Decimal:
    total = Decimal("0")
    since, until = limit.window.bounds(now=now)
    for record in records:
        if not _reservation_matches_limit(record, limit):
            continue
        if record.status == "active":
            total += record.reserved_amount
        elif record.status == "reconciled":
            if since is not None and record.updated_at < since:
                continue
            if until is not None and record.updated_at >= until:
                continue
            total += record.actual_amount or Decimal("0")
    return total


def _reservation_matches_limit(record: BudgetReservationRecord, limit: BudgetLimit) -> bool:
    return (
        record.scope == limit.scope
        and record.key == limit.key
        and record.window.storage_key == limit.window.storage_key
        and record.currency.upper() == limit.currency.upper()
    )


def _validate_reservation_ttl(value: int | None) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 1:
        raise ValueError("reservation_ttl_seconds must be a positive integer or None.")
    return value


def _reservation_is_expired(
    record: BudgetReservationRecord,
    *,
    now: datetime,
    ttl_seconds: int | None,
) -> bool:
    if ttl_seconds is None:
        return False
    cutoff = now - timedelta(seconds=ttl_seconds)
    return record.updated_at <= cutoff


_EXPIRED_RESERVATION_REASON_PREFIX = "Reservation expired:"


def _expired_reservation_reason(ttl_seconds: int) -> str:
    return f"{_EXPIRED_RESERVATION_REASON_PREFIX} not reconciled within {ttl_seconds}s."


def _is_expired_reservation_reason(reason: str | None) -> bool:
    """True when a released reservation was reaped by the TTL (vs. explicitly released)."""
    return reason is not None and reason.startswith(_EXPIRED_RESERVATION_REASON_PREFIX)


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
    updated_at: datetime | None = None,
    billing_identity: BillingIdentity | None = None,
) -> BudgetReservationRecord:
    resolved_identity = _reconciled_billing_identity(record.billing_identity, billing_identity)
    if record.status == "reconciled":
        if (
            record.actual_amount == actual_amount
            and record.reason == reason
            and record.billing_identity == resolved_identity
        ):
            return record
        raise ValueError(
            f"Budget reservation has a conflicting reconciliation: {record.reservation_id}"
        )
    if record.status != "active" and not (
        record.status == "released" and _is_expired_reservation_reason(record.reason)
    ):
        raise ValueError(f"Budget reservation is not active: {record.reservation_id}")
    reconciled_at = (
        _utc_datetime(updated_at, "updated_at") if updated_at is not None else datetime.now(UTC)
    )
    return record.model_copy(
        update={
            "actual_amount": actual_amount,
            "status": "reconciled",
            "reason": reason,
            "updated_at": reconciled_at,
            "billing_identity": resolved_identity,
        },
        deep=True,
    )


def _released_record(
    record: BudgetReservationRecord,
    *,
    reason: str,
    updated_at: datetime | None = None,
) -> BudgetReservationRecord:
    if record.status == "released":
        if record.reason == reason or _is_expired_reservation_reason(record.reason):
            return record
        raise ValueError(f"Budget reservation has a conflicting release: {record.reservation_id}")
    if record.status != "active":
        raise ValueError(f"Budget reservation is not active: {record.reservation_id}")
    released_at = (
        _utc_datetime(updated_at, "updated_at") if updated_at is not None else datetime.now(UTC)
    )
    return record.model_copy(
        update={
            "status": "released",
            "reason": reason,
            "updated_at": released_at,
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
        billing_identity=record.billing_identity,
    )


def _reconciled_billing_identity(
    requested: BillingIdentity | None,
    completed: BillingIdentity | None,
) -> BillingIdentity | None:
    if completed is None:
        return requested
    return completed_billing_identity(requested, completed)


def _validate_amount(value: Decimal, field_name: str) -> Decimal:
    if type(value) is not Decimal:
        value = Decimal(str(value))
    if not value.is_finite() or value < 0:
        raise ValueError(f"{field_name} must be a finite non-negative Decimal.")
    return value


def _budget_window_from_string(value: str) -> BudgetWindow:
    text = require_clean_nonblank(value, "window")
    if text == _ALL_TIME_WINDOW:
        return BudgetWindow.all_time()
    if text.startswith(_ROLLING_PREFIX) and text.endswith(_ROLLING_SUFFIX):
        raw_seconds = text[len(_ROLLING_PREFIX) : -len(_ROLLING_SUFFIX)]
        try:
            seconds = int(raw_seconds)
        except ValueError as exc:
            raise ValueError(f"Invalid rolling budget window: {text}") from exc
        return BudgetWindow.rolling(seconds=seconds)
    if text.startswith(_CALENDAR_PREFIX):
        parts = text.split(":", 2)
        if len(parts) != 3:
            raise ValueError(f"Invalid calendar budget window: {text}")
        return BudgetWindow.calendar(
            period=_calendar_period(parts[1]),
            timezone=parts[2],
        )
    raise ValueError(f"Unsupported budget window: {text}")


def _utc_datetime(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware.")
    return value.astimezone(UTC)


def _clock_or_utc_now(clock: Callable[[], datetime] | None) -> Callable[[], datetime]:
    if clock is None:
        return lambda: datetime.now(UTC)
    if not callable(clock):
        raise TypeError("clock must be callable.")

    def _checked_clock() -> datetime:
        return _utc_datetime(clock(), "clock()")

    return _checked_clock


def _event_in_window(
    event: Event,
    *,
    since: datetime | None,
    until: datetime | None,
) -> bool:
    timestamp = _utc_datetime(event.timestamp, "event.timestamp")
    if since is not None and timestamp < since:
        return False
    return not (until is not None and timestamp >= until)


def _calendar_period(value: str) -> BudgetCalendarPeriod:
    if value in {"day", "week", "month"}:
        return cast("BudgetCalendarPeriod", value)
    raise ValueError(f"Unsupported calendar budget period: {value}")


def _calendar_window_bounds(
    now: datetime,
    *,
    period: BudgetCalendarPeriod,
    timezone: str,
) -> tuple[datetime, datetime]:
    zone = ZoneInfo(timezone)
    local_now = now.astimezone(zone)
    if period == "day":
        local_start = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
        local_end = local_start + timedelta(days=1)
    elif period == "week":
        local_day_start = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
        local_start = local_day_start - timedelta(days=local_now.weekday())
        local_end = local_start + timedelta(days=7)
    elif period == "month":
        local_start = local_now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if local_start.month == 12:
            local_end = local_start.replace(year=local_start.year + 1, month=1)
        else:
            local_end = local_start.replace(month=local_start.month + 1)
    else:
        raise ValueError(f"Unsupported calendar budget period: {period}")
    return local_start.astimezone(UTC), local_end.astimezone(UTC)
