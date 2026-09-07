"""Refresh exact totals by repricing only changed attempt groups, with fixed state."""

from __future__ import annotations

import hmac
import secrets
from hashlib import sha256
from typing import TYPE_CHECKING

from cayu._validation import canonical_durable_json_bytes, require_clean_nonblank
from cayu.core import Event
from cayu.runtime._cost_accounting import (
    COST_EVENT_TYPES,
    CostAccountingCursor,
    CostAccountingReducer,
    CostAccountingSnapshot,
    CostGroupKey,
    cost_group_key,
)
from cayu.runtime.costs import PriceBook, SessionCostTotals, add_cost_amounts, copy_price_book

if TYPE_CHECKING:
    from cayu.runtime.sessions import EventQuery, EventRecord


class CostAccountingAuthority:
    """Authenticate ephemeral cursor totals to one store instance and deletion revision."""

    def __init__(self) -> None:
        self._key = secrets.token_bytes(32)

    def _signature(self, snapshot: CostAccountingSnapshot) -> str:
        if snapshot.cursor is None or snapshot.durable_totals is None:
            raise ValueError("Incremental cost cursor is incomplete.")
        message = {
            "cursor": snapshot.cursor.model_dump(mode="json", exclude={"signature"}),
            "through_sequence": snapshot.through_sequence,
            "durable_totals": snapshot.durable_totals.model_dump(mode="json"),
        }
        return hmac.new(
            self._key, canonical_durable_json_bytes(message, "cost cursor"), sha256
        ).hexdigest()

    def verify(self, snapshot: CostAccountingSnapshot) -> bool:
        if snapshot.cursor is None:
            return False
        try:
            return hmac.compare_digest(snapshot.cursor.signature, self._signature(snapshot))
        except (TypeError, ValueError):
            return False

    def authorize(self, snapshot: CostAccountingSnapshot) -> CostAccountingSnapshot:
        assert snapshot.cursor is not None
        cursor = snapshot.cursor.model_copy(update={"signature": self._signature(snapshot)})
        return snapshot.model_copy(update={"cursor": cursor})


def _cursor(query: EventQuery, pricing: PriceBook, currency: str) -> CostAccountingCursor:
    scope = query.model_dump(mode="json", exclude={"since", "until", "limit", "order_by"})
    prices = {"currency": currency.upper(), "pricing": pricing.model_dump(mode="json")}
    return CostAccountingCursor(
        scope_digest=sha256(canonical_durable_json_bytes(scope, "cost scope")).hexdigest(),
        pricing_digest=sha256(canonical_durable_json_bytes(prices, "cost pricing")).hexdigest(),
        since=query.since,
        until=query.until,
    )


def _apply_delta(
    total: SessionCostTotals,
    new: SessionCostTotals,
    old: SessionCostTotals,
) -> SessionCostTotals:
    return SessionCostTotals(
        session_id=total.session_id,
        currency=total.currency,
        model_steps=total.model_steps + new.model_steps - old.model_steps,
        priced_model_steps=total.priced_model_steps
        + new.priced_model_steps
        - old.priced_model_steps,
        unpriced_model_steps=total.unpriced_model_steps
        + new.unpriced_model_steps
        - old.unpriced_model_steps,
        total_cost=add_cost_amounts(total.total_cost, new.total_cost, old.total_cost.copy_negate()),
    )


class CostAccountingRead:
    """Cold stream or prior totals plus exact old/new contributions of changed groups.

    The store supplies each changed group's complete evidence in one snapshot,
    hosted rows before completions. No historical contributions are cached here.
    A pending tail affects only the returned effective totals, never the durable
    baseline retained for the next refresh.
    """

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
        previous: CostAccountingSnapshot | None = None,
        generation: int = 0,
        through_sequence: int = 0,
        authority: CostAccountingAuthority | None = None,
    ) -> None:
        from cayu.runtime.sessions import copy_event_query

        if type(details) is not bool or type(by_session) is not bool:
            raise TypeError("Cost accounting output flags must be bools.")
        if max_detail_bytes is not None and (
            type(max_detail_bytes) is not int or max_detail_bytes < 1
        ):
            raise ValueError("max_detail_bytes must be a positive integer.")
        self.query = query
        self._pricing = copy_price_book(pricing)
        self._currency = require_clean_nonblank(currency, "currency").upper()
        self._authority = authority or CostAccountingAuthority()
        self._cursor = _cursor(query, self._pricing, self._currency).model_copy(
            update={"generation": generation}
        )
        if previous is not None and type(previous) is not CostAccountingSnapshot:
            raise TypeError("previous must be a CostAccountingSnapshot.")
        self.previous = None
        if previous is not None and not details and not by_session:
            if type(previous) is not CostAccountingSnapshot:
                raise TypeError("previous must be a CostAccountingSnapshot.")
            cursor = previous.cursor
            if cursor is not None and (
                cursor.scope_digest == self._cursor.scope_digest
                and cursor.pricing_digest == self._cursor.pricing_digest
                and previous.durable_totals is not None
                and cursor.generation == generation
                and self._authority.verify(previous)
            ):
                self.previous = previous
        self.incremental = self.previous is not None
        self._cold: CostAccountingReducer | None = None
        self._cold_durable: CostAccountingReducer | None = None
        self._pending: dict[CostGroupKey, tuple[Event, ...]] = {}
        for event in additional_events:
            key = cost_group_key(event)
            self._pending[key] = (*self._pending.get(key, ()), event)
        self._key: CostGroupKey | None = None
        self._old: CostAccountingReducer | None = None
        self._new: CostAccountingReducer | None = None
        self._new_durable: CostAccountingReducer | None = None
        self._through = max(query.after_sequence or 0, through_sequence)
        self.old_query = query
        self.source_query = query
        self._total: SessionCostTotals | None = None
        self._durable_total: SessionCostTotals | None = None
        if self.previous is None:
            self._cold = CostAccountingReducer(
                query,
                self._pricing,
                currency=currency,
                details=details,
                by_session=by_session,
                additional_events=additional_events,
                max_detail_bytes=max_detail_bytes,
                _pricing_is_snapshot=True,
            )
            if additional_events:
                self._cold_durable = self._reducer(query)
        else:
            assert self.previous.cursor is not None
            assert self.previous.durable_totals is not None
            self._total = SessionCostTotals.model_validate(
                self.previous.durable_totals.model_dump()
            )
            self._durable_total = self._total
            self._through = max(self.previous.through_sequence, through_sequence)
            self.old_query = copy_event_query(
                query,
                update={
                    "since": self.previous.cursor.since,
                    "until": self.previous.cursor.until,
                },
            )
            since = (
                None
                if query.since is None or self.old_query.since is None
                else min(query.since, self.old_query.since)
            )
            until = (
                None
                if query.until is None or self.old_query.until is None
                else max(query.until, self.old_query.until)
            )
            self.source_query = copy_event_query(query, update={"since": since, "until": until})

    def _reducer(self, query: EventQuery, pending: tuple[Event, ...] = ()) -> CostAccountingReducer:
        return CostAccountingReducer(
            query,
            self._pricing,
            currency=self._currency,
            additional_events=pending,
            _pricing_is_snapshot=True,
        )

    def _matches(self, record: EventRecord, query: EventQuery) -> bool:
        from cayu.runtime.sessions import _event_record_matches

        return _event_record_matches(
            record, query, frozenset(str(kind) for kind in COST_EVENT_TYPES), frozenset()
        )

    @property
    def remaining_pending_keys(self) -> tuple[CostGroupKey, ...]:
        """At most the explicit 256-event tail, for store lookups before finishing."""
        return tuple(self._pending)

    def add(self, sequence: int, event: Event) -> None:
        from cayu.runtime.sessions import EventRecord

        if self._cold is not None:
            self._cold.add(sequence, event)
            if self._cold_durable is not None:
                self._cold_durable.add(sequence, event)
            return
        key = cost_group_key(event)
        if key != self._key:
            self._finish_group()
            self._start_group(key)
        record = EventRecord(sequence=sequence, event=event)
        assert self.previous is not None
        assert self._new is not None and self._old is not None
        if sequence <= self.previous.through_sequence and self._matches(record, self.old_query):
            self._old.add(sequence, event)
        if self._matches(record, self.query):
            self._new.add(sequence, event)
            if self._new_durable is not None:
                self._new_durable.add(sequence, event)
            self._through = max(self._through, sequence)

    def _start_group(self, key: CostGroupKey) -> None:
        self._key = key
        pending = self._pending.pop(key, ())
        self._old = self._reducer(self.old_query)
        self._new = self._reducer(self.query, pending)
        self._new_durable = self._reducer(self.query) if pending else None

    def _finish_group(self) -> None:
        if self._key is None:
            return
        assert self._new is not None and self._old is not None
        assert self._total is not None and self._durable_total is not None
        new = self._new.snapshot().totals
        old = self._old.snapshot().totals
        durable = self._new_durable.snapshot().totals if self._new_durable is not None else new
        self._total = _apply_delta(self._total, new, old)
        self._durable_total = _apply_delta(self._durable_total, durable, old)
        self._key = None

    def snapshot(self) -> CostAccountingSnapshot:
        if self._cold is not None:
            result = self._cold.snapshot()
            durable = (
                self._cold_durable.snapshot().totals
                if self._cold_durable is not None
                else result.totals
            )
            return self._authority.authorize(
                result.model_copy(
                    update={
                        "durable_totals": durable,
                        "cursor": self._cursor,
                        "through_sequence": max(result.through_sequence, self._through),
                    }
                )
            )
        self._finish_group()
        while self._pending:
            self._start_group(next(iter(self._pending)))
            self._finish_group()
        assert self._total is not None and self._durable_total is not None
        return self._authority.authorize(
            CostAccountingSnapshot(
                through_sequence=self._through,
                totals=self._total,
                durable_totals=self._durable_total,
                cursor=self._cursor,
            )
        )
