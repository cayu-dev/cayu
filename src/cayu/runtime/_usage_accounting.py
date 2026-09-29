"""Exact usage reduction with a fixed event page and one ordered watermark."""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from cayu._validation import MAX_DURABLE_JSON_INTEGER
from cayu.budgets.aggregates import (
    AggregateUsageMetrics,
    add_aggregate_usage,
    build_aggregate_usage_metrics,
    summary_usage_metrics_from_event_payload,
)
from cayu.budgets.usage import (
    USAGE_BEARING_EVENT_TYPES,
    CausalBudgetUsageSummary,
    SessionUsageSummary,
    combine_session_usage_summaries,
    session_usage_summary,
)
from cayu.events import EventType

if TYPE_CHECKING:
    from cayu.sessions.base import EventQuery, EventRecord

USAGE_ACCOUNTING_PAGE_SIZE = 256
USAGE_ACCOUNTING_CACHE_MAX_SESSIONS = 4096


class UsageIdentitySummary(BaseModel):
    """Requested identity breakdown, including its distinct session membership."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    provider_name: str | None
    model: str | None
    session_ids: tuple[str, ...]
    model_steps: StrictInt = Field(ge=0)
    usage: AggregateUsageMetrics


class UsageAccountingSnapshot(BaseModel):
    """Exact usage through a store-owned sequence boundary, without source events."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    through_sequence: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    generation: StrictInt = Field(default=0, ge=0, le=MAX_DURABLE_JSON_INTEGER)
    summary: SessionUsageSummary
    session_summaries: tuple[SessionUsageSummary, ...] = ()
    provider_summaries: tuple[UsageIdentitySummary, ...] = ()
    model_summaries: tuple[UsageIdentitySummary, ...] = ()


def usage_accounting_query(query: EventQuery) -> EventQuery:
    from cayu.sessions.base import EventOrder, copy_event_query

    query = copy_event_query(query)
    if query.event_type is not None or query.event_types or query.exclude_event_types:
        raise ValueError("Usage accounting owns its usage-bearing event filter.")
    return copy_event_query(
        query,
        update={
            "event_types": USAGE_BEARING_EVENT_TYPES,
            "after_sequence": query.after_sequence or 0,
            "order_by": EventOrder.SEQUENCE_ASC,
            "limit": USAGE_ACCOUNTING_PAGE_SIZE,
        },
    )


@dataclass(frozen=True, slots=True)
class _SessionUsageEntry:
    through_sequence: int
    scanned_through: int
    summary: SessionUsageSummary


class SessionUsageCache:
    """Carry exact whole-session usage forward inside one store instance.

    An entry is reused only while the durable accounting generation is
    unchanged, so deleting usage-bearing evidence forces a cold read. Built-in
    stores serialize appends per session, so every row committed after an entry
    was taken has a larger sequence than the entry's scanned boundary. A read
    therefore reduces only rows after that boundary.
    """

    def __init__(self, max_sessions: int = USAGE_ACCOUNTING_CACHE_MAX_SESSIONS) -> None:
        if type(max_sessions) is not int or max_sessions < 1:
            raise ValueError("max_sessions must be a positive integer.")
        self._lock = threading.Lock()
        self._max_sessions = max_sessions
        self._generation = -1
        self._entries: OrderedDict[str, _SessionUsageEntry] = OrderedDict()

    @staticmethod
    def session_scope(query: EventQuery, *, by_session: bool, by_identity: bool) -> str | None:
        """Return the session for a plain whole-session read, else ``None``.

        ``query`` must already be normalized by ``usage_accounting_query``.
        Filters, windows, grouped output, and access-bounded reads bypass the
        cache so it never serves a result computed under a different scope.
        """
        from cayu.sessions.access import _query_bounds
        from cayu.sessions.base import EventQuery

        session_id = query.session_id
        if by_session or by_identity or session_id is None or _query_bounds.get() is not None:
            return None
        if query != usage_accounting_query(EventQuery(session_id=session_id)):
            return None
        return session_id

    def resume_after(self, session_id: str, generation: int) -> _SessionUsageEntry | None:
        with self._lock:
            if generation > self._generation:
                self._entries.clear()
                self._generation = generation
            if generation != self._generation:
                return None
            entry = self._entries.get(session_id)
            if entry is not None:
                self._entries.move_to_end(session_id)
            return entry

    def settle(
        self,
        session_id: str,
        generation: int,
        prior: _SessionUsageEntry | None,
        delta: UsageAccountingSnapshot,
        *,
        boundary: int,
    ) -> UsageAccountingSnapshot:
        """Combine a prior entry with rows read after it, then retain the result.

        ``boundary`` is the session's highest sequence in the same read
        snapshot, so an unrelated event tail is not inspected again.
        """
        if prior is None:
            summary = delta.summary
            through = delta.through_sequence
            scanned = max(boundary, through)
        elif delta.through_sequence > prior.scanned_through:
            summary = combine_session_usage_summaries(session_id, (prior.summary, delta.summary))
            through = delta.through_sequence
            scanned = max(boundary, through)
        else:
            summary = prior.summary
            through = prior.through_sequence
            scanned = max(boundary, prior.scanned_through)
        entry = _SessionUsageEntry(
            through_sequence=through,
            scanned_through=scanned,
            summary=summary.model_copy(deep=True),
        )
        with self._lock:
            if generation > self._generation:
                self._entries.clear()
                self._generation = generation
            current = self._entries.get(session_id)
            if generation == self._generation and (
                current is None or current.scanned_through <= scanned
            ):
                self._entries[session_id] = entry
                self._entries.move_to_end(session_id)
                while len(self._entries) > self._max_sessions:
                    self._entries.popitem(last=False)
        return UsageAccountingSnapshot(
            through_sequence=through,
            generation=generation,
            summary=summary.model_copy(deep=True),
        )


class UsageAccountingReducer:
    """Retain totals and requested result groups; never retain historical events."""

    def __init__(self, query: EventQuery, *, by_session: bool, by_identity: bool = False) -> None:
        if type(by_session) is not bool or type(by_identity) is not bool:
            raise TypeError("Usage accounting output flags must be bools.")
        self._summary = SessionUsageSummary(
            session_id=query.session_id or query.causal_budget_id or "usage-query"
        )
        self._through_sequence = query.after_sequence or 0
        self._by_session = by_session
        self._sessions: dict[str, SessionUsageSummary] = {}
        self._by_identity = by_identity
        self._identities: dict[tuple[bool, str | None, str | None], UsageIdentitySummary] = {}
        self._identity_sessions: dict[tuple[bool, str | None, str | None], dict[str, None]] = {}

    def add_page(self, records: Sequence[EventRecord]) -> None:
        if len(records) > USAGE_ACCOUNTING_PAGE_SIZE:
            raise ValueError("Usage accounting page exceeds its working-set bound.")
        previous = self._through_sequence
        for record in records:
            if record.sequence <= previous:
                raise ValueError("Usage accounting requires one strictly ordered watermark.")
            previous = record.sequence
        events = [record.event for record in records]
        self._summary = combine_session_usage_summaries(
            self._summary.session_id,
            (self._summary, session_usage_summary(self._summary.session_id, events)),
        )
        if self._by_session:
            # At most one page of references, regardless of historical event count.
            groups = dict.fromkeys(event.session_id for event in events)
            for session_id in groups:
                page_summary = session_usage_summary(
                    session_id, [event for event in events if event.session_id == session_id]
                )
                prior = self._sessions.get(session_id)
                self._sessions[session_id] = (
                    page_summary
                    if prior is None
                    else combine_session_usage_summaries(session_id, (prior, page_summary))
                )
        if self._by_identity:
            for event in events:
                if event.type not in {
                    EventType.MODEL_COMPLETED,
                    EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED,
                }:
                    continue
                try:
                    metrics = summary_usage_metrics_from_event_payload(event.payload)
                except (TypeError, ValueError):
                    continue
                if metrics is None:
                    continue
                for by_model in (False, True):
                    key = (by_model, metrics.provider_name, metrics.model if by_model else None)
                    prior = self._identities.get(key)
                    membership = self._identity_sessions.setdefault(key, {})
                    membership[event.session_id] = None
                    self._identities[key] = UsageIdentitySummary(
                        provider_name=key[1],
                        model=key[2],
                        session_ids=(),
                        model_steps=(0 if prior is None else prior.model_steps)
                        + int(event.type == EventType.MODEL_COMPLETED),
                        usage=add_aggregate_usage(
                            build_aggregate_usage_metrics() if prior is None else prior.usage,
                            metrics,
                        ),
                    )
        self._through_sequence = previous

    def snapshot(self) -> UsageAccountingSnapshot:
        return UsageAccountingSnapshot(
            through_sequence=self._through_sequence,
            summary=self._summary,
            session_summaries=tuple(self._sessions.values()),
            provider_summaries=tuple(
                row.model_copy(update={"session_ids": tuple(self._identity_sessions[key])})
                for key, row in self._identities.items()
                if not key[0]
            ),
            model_summaries=tuple(
                row.model_copy(update={"session_ids": tuple(self._identity_sessions[key])})
                for key, row in self._identities.items()
                if key[0]
            ),
        )


def causal_usage_summary(
    snapshot: UsageAccountingSnapshot,
    causal_budget_id: str,
    session_ids: list[str],
) -> CausalBudgetUsageSummary:
    session_ids = list(dict.fromkeys(session_ids))
    per_session = {row.session_id: row for row in snapshot.session_summaries}
    total = snapshot.summary
    return CausalBudgetUsageSummary(
        causal_budget_id=causal_budget_id,
        session_ids=session_ids,
        session_count=len(session_ids),
        model_steps=total.model_steps,
        unmeasured_model_attempts=total.unmeasured_model_attempts,
        tool_calls=total.tool_calls,
        provider_names=total.provider_names,
        models=total.models,
        usage=total.usage,
        session_summaries=tuple(
            per_session.get(session_id, SessionUsageSummary(session_id=session_id))
            for session_id in session_ids
        ),
    )
