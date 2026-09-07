"""Exact usage reduction with a fixed event page and one ordered watermark."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from cayu._validation import MAX_DURABLE_JSON_INTEGER
from cayu.core import EventType
from cayu.runtime.aggregates import (
    AggregateUsageMetrics,
    add_aggregate_usage,
    build_aggregate_usage_metrics,
    summary_usage_metrics_from_event_payload,
)
from cayu.runtime.usage import (
    USAGE_BEARING_EVENT_TYPES,
    CausalBudgetUsageSummary,
    SessionUsageSummary,
    combine_session_usage_summaries,
    session_usage_summary,
)

if TYPE_CHECKING:
    from cayu.runtime.sessions import EventQuery, EventRecord

USAGE_ACCOUNTING_PAGE_SIZE = 256


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
    from cayu.runtime.sessions import EventOrder, copy_event_query

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
                if event.type != EventType.MODEL_COMPLETED:
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
                        model_steps=1 if prior is None else prior.model_steps + 1,
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
        tool_calls=total.tool_calls,
        provider_names=total.provider_names,
        models=total.models,
        usage=total.usage,
        session_summaries=tuple(
            per_session.get(session_id, SessionUsageSummary(session_id=session_id))
            for session_id in session_ids
        ),
    )
