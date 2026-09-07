"""Bounded exact pricing of a store snapshot, grouped by model-attempt identity."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from cayu._validation import MAX_DURABLE_JSON_INTEGER, JsonUtf8SizeCounter, require_clean_nonblank
from cayu.core import Event, EventType
from cayu.runtime.aggregates import aggregate_hosted_tool_usage_metrics_from_event_payload
from cayu.runtime.costs import (
    CausalBudgetCostSummary,
    CostLineItem,
    PriceBook,
    SessionCostSummary,
    SessionCostTotals,
    _combine_hosted_cost_metrics,
    _cost_line_item,
    _cost_usage_metrics_from_event_payload,
    _effective_date,
    _merge_hosted_cost_metrics,
    _missing_usage_pricing_reason,
    _optional_billing_identity,
    _optional_execution_profile_fingerprint,
    _optional_nonblank,
    _unpriced_line_item,
    add_cost_amounts,
    copy_price_book,
)
from cayu.runtime.usage import UsageMetrics

if TYPE_CHECKING:
    from cayu.runtime.sessions import EventQuery

COST_ACCOUNTING_PAGE_SIZE = 256
COST_ACCOUNTING_MAX_PENDING_EVENTS = 256
COST_EVENT_TYPES = (EventType.MODEL_COMPLETED, EventType.MODEL_HOSTED_TOOL_CALL)
CostGroupKey = tuple[str, bool, str]


class CostAccountingOutputTooLarge(ValueError):
    """Requested detailed output exceeds the caller's explicit byte bound."""


class CostAccountingCursor(BaseModel):
    """Bind a refresh to the exact scope, price book, and prior window membership."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    scope_digest: str
    pricing_digest: str
    generation: StrictInt = Field(default=0, ge=0, le=MAX_DURABLE_JSON_INTEGER)
    signature: str = ""
    since: datetime | None = None
    until: datetime | None = None


class CostAccountingSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    through_sequence: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    totals: SessionCostTotals
    session_totals: tuple[SessionCostTotals, ...] = ()
    details: SessionCostSummary | None = None
    session_details: tuple[SessionCostSummary, ...] = ()
    durable_totals: SessionCostTotals | None = None
    cursor: CostAccountingCursor | None = None


def cost_group_key(event: Event) -> CostGroupKey:
    attempt = _optional_nonblank(event.payload.get("model_attempt_id"))
    # Attempt identity is session-scoped, so aggregate cost always equals the
    # sum of unique per-session costs, even for caller-supplied colliding IDs.
    return event.session_id, attempt is not None, attempt or event.id


def cost_accounting_query(query: EventQuery) -> EventQuery:
    from cayu.runtime.sessions import EventOrder, copy_event_query

    query = copy_event_query(query)
    if query.event_type is not None or query.event_types or query.exclude_event_types:
        raise ValueError("Cost accounting owns its cost-bearing event filter.")
    return copy_event_query(
        query,
        update={
            "event_types": COST_EVENT_TYPES,
            "after_sequence": query.after_sequence or 0,
            "order_by": EventOrder.SEQUENCE_ASC,
            "limit": COST_ACCOUNTING_PAGE_SIZE,
        },
    )


def cost_pending_events(query: EventQuery, events: tuple[Event, ...]) -> tuple[Event, ...]:
    """Copy and filter a bounded in-flight tail; stores additionally check causal membership."""
    from cayu.core.events import copy_event
    from cayu.runtime.sessions import EventRecord, _event_record_matches, copy_event_query

    if type(events) is not tuple:
        raise TypeError("additional_events must be a tuple.")
    if len(events) > COST_ACCOUNTING_MAX_PENDING_EVENTS:
        raise ValueError("Cost accounting in-flight events exceed the working-set bound.")
    query = copy_event_query(query, update={"after_sequence": None, "before_sequence": None})
    kinds = frozenset(str(kind) for kind in COST_EVENT_TYPES)
    pending: dict[tuple[str, str], Event] = {}
    for event in events:
        event = copy_event(event)
        if _event_record_matches(EventRecord(sequence=1, event=event), query, kinds, frozenset()):
            pending.setdefault((event.session_id, event.id), event)
    return tuple(pending.values())


@dataclass
class _Totals:
    model_steps: int = 0
    priced_model_steps: int = 0
    unpriced_model_steps: int = 0
    missing_usage_model_steps: int = 0
    missing_pricing_model_steps: int = 0
    unsupported_pricing_model_steps: int = 0

    total_cost: Decimal = Decimal(0)

    def add(self, item: CostLineItem) -> None:
        if item.model_step:
            self.model_steps += 1
            self.priced_model_steps += int(item.priced)
            self.unpriced_model_steps += int(not item.priced)
            self.missing_usage_model_steps += int(
                not item.priced and item.unpriced_reason == "missing_usage"
            )
            self.missing_pricing_model_steps += int(
                not item.priced and item.unpriced_reason == "missing_pricing"
            )
            self.unsupported_pricing_model_steps += int(
                not item.priced and item.unpriced_reason == "unsupported_pricing"
            )

        self.total_cost = add_cost_amounts(self.total_cost, item.total_cost)

    def summary(self, session_id: str, currency: str) -> SessionCostTotals:
        return SessionCostTotals(
            session_id=session_id,
            currency=currency,
            model_steps=self.model_steps,
            priced_model_steps=self.priced_model_steps,
            unpriced_model_steps=self.unpriced_model_steps,
            missing_usage_model_steps=self.missing_usage_model_steps,
            missing_pricing_model_steps=self.missing_pricing_model_steps,
            unsupported_pricing_model_steps=self.unsupported_pricing_model_steps,
            total_cost=self.total_cost,
        )


class CostAccountingReducer:
    """One attempt's hosted evidence plus totals; detailed rows are opt-in output."""

    def __init__(
        self,
        query: EventQuery,
        pricing: PriceBook,
        *,
        currency: str,
        details: bool = False,
        by_session: bool = False,
        additional_events: tuple[Event, ...] = (),
        max_detail_bytes: int | None = None,
        _pricing_is_snapshot: bool = False,
    ) -> None:
        if type(details) is not bool or type(by_session) is not bool:
            raise TypeError("Cost accounting output flags must be bools.")
        if len(additional_events) > COST_ACCOUNTING_MAX_PENDING_EVENTS:
            raise ValueError("Cost accounting in-flight events exceed the working-set bound.")
        self._pricing = pricing if _pricing_is_snapshot else copy_price_book(pricing)
        self._currency = require_clean_nonblank(currency, "currency").upper()
        self._session_id = query.session_id or query.causal_budget_id or "cost-query"
        self._through_sequence = query.after_sequence or 0
        if max_detail_bytes is not None and (
            type(max_detail_bytes) is not int or max_detail_bytes < 1
        ):
            raise ValueError("max_detail_bytes must be a positive integer.")
        self._detail_size = (
            None if max_detail_bytes is None else JsonUtf8SizeCounter(max_detail_bytes)
        )
        self._details = details
        self._by_session = by_session
        self._totals = _Totals()
        self._session_totals: dict[str, _Totals] = {}
        self._lines: list[tuple[int, int, str, CostLineItem]] = []
        self._pending: dict[CostGroupKey, list[tuple[int, Event]]] = {}
        for index, event in enumerate(additional_events):
            self._pending.setdefault(cost_group_key(event), []).append(
                (MAX_DURABLE_JSON_INTEGER + index + 1, event)
            )
        self._key: CostGroupKey | None = None
        self._hosted: UsageMetrics | None = None
        self._hosted_time: datetime | None = None
        self._hosted_profile: str | None = None
        self._hosted_sequence = 0
        self._completion_seen = False
        self._pending_hosted_added = False
        self._current_pending: list[tuple[int, Event]] = []

    def add(self, sequence: int, event: Event) -> None:
        self._through_sequence = max(self._through_sequence, sequence)
        key = cost_group_key(event)
        if key != self._key:
            self._finish_group()
            self._start_group(key)
        self._current_pending = [row for row in self._current_pending if row[1].id != event.id]
        if event.type == EventType.MODEL_HOSTED_TOOL_CALL:
            if self._completion_seen:
                raise ValueError(
                    "Cost accounting requires hosted evidence before completions in each group."
                )
            self._add_hosted(sequence, event)
        elif event.type == EventType.MODEL_COMPLETED:
            self._add_pending_hosted()
            self._add_completion(sequence, event)

    def _start_group(self, key: CostGroupKey) -> None:
        self._key = key
        self._hosted = None
        self._hosted_time = None
        self._hosted_profile = None
        self._hosted_sequence = 0
        self._completion_seen = False
        self._pending_hosted_added = False
        self._current_pending = self._pending.pop(key, [])

    def _add_hosted(self, sequence: int, event: Event) -> None:
        metrics = aggregate_hosted_tool_usage_metrics_from_event_payload(event.payload)
        if metrics is None:
            return
        profile = _optional_execution_profile_fingerprint(
            event.payload.get("execution_profile_fingerprint")
        )
        if self._hosted is None:
            self._hosted = metrics
            self._hosted_time = event.timestamp
            self._hosted_profile = profile
            self._hosted_sequence = sequence
        else:
            self._hosted = _combine_hosted_cost_metrics(self._hosted, metrics)
            if profile != self._hosted_profile:
                self._hosted_profile = None

    def _add_pending_hosted(self) -> None:
        if self._pending_hosted_added:
            return
        self._pending_hosted_added = True
        for sequence, event in self._current_pending:
            if event.type == EventType.MODEL_HOSTED_TOOL_CALL:
                self._add_hosted(sequence, event)

    def _add_completion(self, sequence: int, event: Event) -> None:
        metrics = _cost_usage_metrics_from_event_payload(event.payload)
        profile = _optional_execution_profile_fingerprint(
            event.payload.get("execution_profile_fingerprint")
        )
        if not self._completion_seen and self._hosted is not None:
            metrics = _merge_hosted_cost_metrics(metrics, self._hosted)
        self._completion_seen = True
        if metrics is None:
            item = _unpriced_line_item(
                model_step=1,
                provider_name=_optional_nonblank(event.payload.get("provider_name")),
                requested_model=_optional_nonblank(event.payload.get("requested_model")),
                model=_optional_nonblank(event.payload.get("model")),
                execution_profile_fingerprint=profile,
                currency=self._currency,
                reason=_missing_usage_pricing_reason(event.payload),
                billing_identity=_optional_billing_identity(event.payload.get("billing_identity")),
            )
        else:
            item = _cost_line_item(
                model_step=1,
                metrics=metrics,
                pricing=self._pricing,
                currency=self._currency,
                effective_on=_effective_date(event.timestamp),
                execution_profile_fingerprint=profile,
            )
        self._add_item(0, sequence, event.session_id, item)

    def _add_item(self, category: int, sequence: int, session_id: str, item: CostLineItem) -> None:
        self._totals.add(item)
        if self._by_session:
            self._session_totals.setdefault(session_id, _Totals()).add(item)
        if self._details:
            if self._detail_size is not None and not self._detail_size.value(item):
                raise CostAccountingOutputTooLarge("Detailed cost output exceeds max_detail_bytes.")
            self._lines.append((category, sequence, session_id, item))

    def _finish_group(self) -> None:
        if self._key is None:
            return
        self._add_pending_hosted()
        for sequence, event in self._current_pending:
            if event.type == EventType.MODEL_COMPLETED:
                self._add_completion(sequence, event)
        if not self._completion_seen and self._hosted is not None:
            assert self._hosted_time is not None
            self._add_item(
                2 if self._key[1] else 1,
                self._hosted_sequence,
                self._key[0],
                _cost_line_item(
                    model_step=0,
                    metrics=self._hosted,
                    pricing=self._pricing,
                    currency=self._currency,
                    effective_on=_effective_date(self._hosted_time),
                    execution_profile_fingerprint=self._hosted_profile,
                ),
            )
        self._key = None

    def snapshot(self) -> CostAccountingSnapshot:
        self._finish_group()
        while self._pending:
            self._start_group(next(iter(self._pending)))
            self._finish_group()
        totals = self._totals.summary(self._session_id, self._currency)
        session_totals = tuple(
            total.summary(session_id, self._currency)
            for session_id, total in sorted(self._session_totals.items())
        )
        details = None
        session_details: tuple[SessionCostSummary, ...] = ()
        if self._details:
            self._lines.sort(key=lambda row: (row[0], row[1]))
            step = 0
            lines: list[CostLineItem] = []
            per_session_lines: dict[str, list[CostLineItem]] = {}
            per_session_steps: dict[str, int] = {}
            for category, _sequence, session_id, item in self._lines:
                if category == 0:
                    step += 1
                lines.append(item.model_copy(update={"model_step": step if category == 0 else 0}))
                if self._by_session:
                    local_step = per_session_steps.get(session_id, 0) + int(category == 0)
                    per_session_steps[session_id] = local_step
                    per_session_lines.setdefault(session_id, []).append(
                        item.model_copy(update={"model_step": local_step if category == 0 else 0})
                    )
            details = SessionCostSummary(**totals.model_dump(), line_items=tuple(lines))
            session_details = tuple(
                SessionCostSummary(
                    **total.model_dump(),
                    line_items=tuple(per_session_lines.get(total.session_id, [])),
                )
                for total in session_totals
            )
        return CostAccountingSnapshot(
            through_sequence=self._through_sequence,
            totals=totals,
            session_totals=session_totals,
            details=details,
            session_details=session_details,
        )


def causal_cost_summary(
    snapshot: CostAccountingSnapshot,
    causal_budget_id: str,
    session_ids: list[str],
) -> CausalBudgetCostSummary:
    """Preserve explicit inspection's ordered, unique, zero-filled session output."""
    if snapshot.details is None:
        raise RuntimeError("Cost accounting store omitted requested details.")
    session_ids = list(dict.fromkeys(session_ids))
    per_session = {row.session_id: row for row in snapshot.session_details}
    return CausalBudgetCostSummary(
        causal_budget_id=causal_budget_id,
        session_ids=session_ids,
        session_count=len(session_ids),
        **snapshot.totals.model_dump(exclude={"session_id"}),
        line_items=snapshot.details.line_items,
        session_costs=tuple(
            per_session[session_id]
            if session_id in per_session
            else SessionCostSummary(
                session_id=session_id,
                currency=snapshot.totals.currency,
                model_steps=0,
                priced_model_steps=0,
                unpriced_model_steps=0,
                total_cost=Decimal(0),
            )
            for session_id in session_ids
        ),
    )
