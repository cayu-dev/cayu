"""Bounded session usage queries and aggregation of durable event records."""

from __future__ import annotations

import heapq
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, datetime, timedelta
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

from cayu.budgets.aggregates import (
    EXACT_AGGREGATE,
    AggregateAccuracy,
    AggregateAccuracyKind,
    AggregateUsageMetrics,
    BoundedUsagePricingInputAccumulator,
    UsageAggregateBreakdown,
    UsageAggregateGroup,
    UsageAggregateRemainder,
    UsageAggregateTotals,
    UsageRollupStoreResult,
    UsageSessionAggregateBreakdown,
    UsageSessionAggregateGroup,
    UsageSessionAggregateRemainder,
    add_aggregate_usage,
    aggregate_hosted_tool_usage_metrics_from_event_payload,
    aggregate_usage_metrics_from_event_payload,
    build_aggregate_usage_metrics,
    normalize_aggregate_event_timestamp,
    require_bounded_usage_session_id,
)
from cayu.budgets.reported import ReportedCostCollector
from cayu.budgets.usage import UsageMetrics
from cayu.events import Event, EventType
from cayu.sessions.queries import SessionAggregateFilter
from cayu.sessions.records import EventRecord, SessionStatus

MAX_USAGE_ROLLUP_WINDOW = timedelta(days=366)


class UsageRollupQuery(BaseModel):
    """Bounded event-time usage query with explicit current-session filtering."""

    model_config = ConfigDict(extra="forbid")

    start_at: datetime
    end_at: datetime
    sessions: SessionAggregateFilter = Field(default_factory=SessionAggregateFilter)
    group_limit: StrictInt = Field(default=20, ge=1, le=100)
    session_group_limit: StrictInt | None = Field(default=None, ge=1, le=100)
    include_pricing_inputs: StrictBool = False
    pricing_input_limit: StrictInt = Field(default=1000, ge=1, le=5000)

    @field_validator("start_at", "end_at")
    @classmethod
    def normalize_window_timestamp(cls, value: datetime, info) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{info.field_name} must be timezone-aware.")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_window(self) -> UsageRollupQuery:
        if self.start_at >= self.end_at:
            raise ValueError("Usage rollup start_at must be before end_at.")
        if self.end_at - self.start_at > MAX_USAGE_ROLLUP_WINDOW:
            raise ValueError(
                f"Usage rollup window cannot exceed {MAX_USAGE_ROLLUP_WINDOW.days} days."
            )
        return self


def copy_usage_rollup_query(query: UsageRollupQuery) -> UsageRollupQuery:
    if type(query) is not UsageRollupQuery:
        raise TypeError("Usage aggregate queries must be UsageRollupQuery instances.")
    return UsageRollupQuery.model_validate(query.model_dump(mode="python"))


@dataclass
class _UsageAccumulator:
    session_count: int = 0
    model_steps: int = 0
    model_steps_with_usage: int = 0
    usage: AggregateUsageMetrics = dataclass_field(default_factory=build_aggregate_usage_metrics)

    def add(self, metrics: UsageMetrics | None) -> None:
        self.model_steps += 1
        if metrics is None:
            return
        self.model_steps_with_usage += 1
        self.usage = add_aggregate_usage(self.usage, metrics)

    def add_usage_only(self, metrics: UsageMetrics) -> None:
        self.usage = add_aggregate_usage(self.usage, metrics)

    def merge(self, other: _UsageAccumulator) -> None:
        self.session_count += other.session_count
        self.model_steps += other.model_steps
        self.model_steps_with_usage += other.model_steps_with_usage
        self.usage = add_aggregate_usage(self.usage, other.usage)

    def totals(self, *, tool_calls: int = 0) -> UsageAggregateTotals:
        return UsageAggregateTotals(
            session_count=self.session_count,
            model_steps=self.model_steps,
            model_steps_with_usage=self.model_steps_with_usage,
            tool_calls=tool_calls,
            usage=self.usage,
        )


_IN_MEMORY_USAGE_GROUP_CANDIDATE_LIMIT = 512


_UsageGroupKey = tuple[str | None, str | None]


_UsageGroupSortKey = tuple[bool, str, bool, str]


_SessionRecordsFactory = Callable[
    [],
    Iterable[tuple[str, SessionStatus, Iterable[EventRecord]]],
]


@dataclass
class _SessionUsageAccumulator:
    session_id: str
    status: SessionStatus
    usage: _UsageAccumulator = dataclass_field(default_factory=_UsageAccumulator)
    tool_calls: int = 0
    has_activity: bool = False

    def add_event(self, event: Event) -> None:
        if event.type == EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED:
            self.has_activity = True
            metrics = aggregate_usage_metrics_from_event_payload(event.payload)
            if metrics is not None:
                self.usage.add_usage_only(metrics)
            return
        if event.type == EventType.TOOL_CALL_STARTED:
            self.tool_calls += 1
            self.has_activity = True
            return
        if event.type == EventType.MODEL_HOSTED_TOOL_CALL:
            metrics = aggregate_hosted_tool_usage_metrics_from_event_payload(event.payload)
            if metrics is not None:
                self.has_activity = True
                self.usage.add_usage_only(metrics)
            return
        if event.type != EventType.MODEL_COMPLETED:
            return
        self.has_activity = True
        self.usage.add(aggregate_usage_metrics_from_event_payload(event.payload))

    def candidate(self) -> _SessionUsageCandidate:
        self.usage.session_count = int(self.has_activity)
        return _SessionUsageCandidate(
            session_id=self.session_id,
            status=self.status,
            active=self.status
            in {
                SessionStatus.PENDING,
                SessionStatus.RUNNING,
                SessionStatus.INTERRUPTING,
            },
            totals=self.usage.totals(tool_calls=self.tool_calls),
        )


@dataclass(frozen=True)
class _SessionUsageCandidate:
    """Internal group candidate whose identity may remain in the remainder."""

    session_id: str
    status: SessionStatus
    active: bool
    totals: UsageAggregateTotals

    def retained_group(self) -> UsageSessionAggregateGroup:
        # Only identities crossing the public retained-group contract need its
        # independent byte bound. Omitted candidates contribute counters only.
        require_bounded_usage_session_id(self.session_id)
        return UsageSessionAggregateGroup(
            session_id=self.session_id,
            status=self.status.value,
            active=self.active,
            totals=self.totals,
        )


@dataclass
class _SessionUsageRemainderAccumulator:
    group_count: int = 0
    active_session_count: int = 0
    totals: _UsageAccumulator = dataclass_field(default_factory=_UsageAccumulator)
    tool_calls: int = 0

    def add(self, group: _SessionUsageCandidate) -> None:
        self.group_count += 1
        self.active_session_count += int(group.active)
        group_totals = _UsageAccumulator(
            session_count=group.totals.session_count,
            model_steps=group.totals.model_steps,
            model_steps_with_usage=group.totals.model_steps_with_usage,
            usage=group.totals.usage,
        )
        self.totals.merge(group_totals)
        self.tool_calls += group.totals.tool_calls

    def result(self) -> UsageSessionAggregateRemainder | None:
        if not self.group_count:
            return None
        return UsageSessionAggregateRemainder(
            group_count=self.group_count,
            active_session_count=self.active_session_count,
            totals=self.totals.totals(tool_calls=self.tool_calls),
        )


@dataclass
class _InMemoryUsageGroupCandidates:
    """Bounded heavy-hitter candidates for one in-memory breakdown dimension."""

    limit: int = _IN_MEMORY_USAGE_GROUP_CANDIDATE_LIMIT
    _estimates: dict[_UsageGroupKey, tuple[int, int, int]] = dataclass_field(default_factory=dict)
    _heap: list[tuple[int, int, bool, str, bool, str, int, _UsageGroupKey]] = dataclass_field(
        default_factory=list
    )
    _generation: int = 0
    sampled: bool = False

    def observe(
        self,
        key: _UsageGroupKey,
        metrics: UsageMetrics | None,
        *,
        model_step: bool = True,
    ) -> None:
        token_weight = 0 if metrics is None else metrics.total_tokens
        step_weight = int(model_step)
        current = self._estimates.get(key)
        if current is not None:
            self._record(key, current[0] + token_weight, current[1] + step_weight)
            return
        if len(self._estimates) < self.limit:
            self._record(key, token_weight, step_weight)
            return

        self.sampled = True
        minimum_tokens, minimum_steps, minimum_key = self._pop_minimum()
        del self._estimates[minimum_key]
        self._record(
            key,
            minimum_tokens + token_weight,
            minimum_steps + step_weight,
        )

    @property
    def keys(self) -> tuple[_UsageGroupKey, ...]:
        return tuple(self._estimates)

    def _record(self, key: _UsageGroupKey, tokens: int, steps: int) -> None:
        self._generation += 1
        generation = self._generation
        self._estimates[key] = tokens, steps, generation
        heapq.heappush(
            self._heap,
            (
                tokens,
                steps,
                *_usage_group_identity_sort_key(key),
                generation,
                key,
            ),
        )
        if len(self._heap) > self.limit * 2:
            self._heap = [
                (
                    item_tokens,
                    item_steps,
                    *_usage_group_identity_sort_key(item_key),
                    item_generation,
                    item_key,
                )
                for item_key, (item_tokens, item_steps, item_generation) in (
                    self._estimates.items()
                )
            ]
            heapq.heapify(self._heap)

    def _pop_minimum(self) -> tuple[int, int, _UsageGroupKey]:
        while self._heap:
            tokens, steps, _, _, _, _, generation, key = heapq.heappop(self._heap)
            if self._estimates.get(key) == (tokens, steps, generation):
                return tokens, steps, key
        raise RuntimeError("In-memory usage candidate heap lost its retained groups.")


def _usage_rollup_from_session_records(
    *,
    session_records: _SessionRecordsFactory,
    query: UsageRollupQuery,
    as_of: datetime,
    matching_session_count: int,
    active_session_count: int,
) -> UsageRollupStoreResult:
    reported = ReportedCostCollector()
    totals = _UsageAccumulator()
    pricing = BoundedUsagePricingInputAccumulator(query.pricing_input_limit)
    session_pricing = BoundedUsagePricingInputAccumulator(query.pricing_input_limit)
    provider_candidates = _InMemoryUsageGroupCandidates()
    model_candidates = _InMemoryUsageGroupCandidates()
    retained_session_groups: list[_SessionUsageCandidate] = []
    session_remainder = _SessionUsageRemainderAccumulator()
    activity_session_count = 0
    tool_calls = 0

    for session_id, status, records in session_records():
        session_has_activity = False
        session_accumulator = (
            None
            if query.session_group_limit is None
            else _SessionUsageAccumulator(session_id=session_id, status=status)
        )
        for record in records:
            event = record.event
            event_timestamp = normalize_aggregate_event_timestamp(event.timestamp)
            if event_timestamp < query.start_at or event_timestamp >= query.end_at:
                continue
            if event.type == EventType.MODEL_COMPLETED:
                reported.add(
                    session_id=session_id,
                    event_id=event.id,
                    timestamp=event_timestamp,
                    payload=event.payload,
                )
            if session_accumulator is not None:
                session_accumulator.add_event(event)
            if event.type == EventType.TOOL_CALL_STARTED:
                session_has_activity = True
                tool_calls += 1
                continue
            if event.type == EventType.MODEL_HOSTED_TOOL_CALL:
                metrics = aggregate_hosted_tool_usage_metrics_from_event_payload(event.payload)
                if metrics is None:
                    continue
                session_has_activity = True
                totals.add_usage_only(metrics)
                provider_candidates.observe(
                    _usage_group_key(metrics, dimension="provider"),
                    metrics,
                    model_step=False,
                )
                model_candidates.observe(
                    _usage_group_key(metrics, dimension="model"),
                    metrics,
                    model_step=False,
                )
                if query.include_pricing_inputs and not pricing.truncated:
                    pricing.add_payload(
                        event_type=EventType.MODEL_HOSTED_TOOL_CALL,
                        effective_on=event_timestamp.date(),
                        occurrences=1,
                        payload=event.payload,
                    )
                continue
            if event.type not in {
                EventType.MODEL_COMPLETED,
                EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED,
            }:
                continue

            session_has_activity = True
            metrics = aggregate_usage_metrics_from_event_payload(event.payload)
            model_step = event.type == EventType.MODEL_COMPLETED
            if model_step:
                totals.add(metrics)
            elif metrics is not None:
                totals.add_usage_only(metrics)
            provider_candidates.observe(
                _usage_group_key(metrics, dimension="provider"),
                metrics,
                model_step=model_step,
            )
            model_candidates.observe(
                _usage_group_key(metrics, dimension="model"),
                metrics,
                model_step=model_step,
            )

            if not query.include_pricing_inputs or pricing.truncated:
                continue
            pricing.add_payload(
                event_type=EventType.MODEL_COMPLETED
                if model_step
                else EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED,
                effective_on=event_timestamp.date(),
                occurrences=1,
                payload=event.payload,
            )
        if session_has_activity:
            activity_session_count += 1
        if session_accumulator is not None:
            assert query.session_group_limit is not None
            _retain_bounded_session_usage_group(
                retained_session_groups,
                session_remainder,
                session_accumulator.candidate(),
                limit=query.session_group_limit,
            )

    pricing_items, pricing_group_count, pricing_accuracy = pricing.result()
    session_breakdown = _in_memory_session_usage_breakdown(
        retained_session_groups,
        session_remainder,
        limit=query.session_group_limit,
    )
    if query.include_pricing_inputs and session_breakdown is not None and session_breakdown.groups:
        visible_session_ids = {group.session_id for group in session_breakdown.groups}
        for session_id, _, records in session_records():
            if session_id not in visible_session_ids:
                continue
            for record in records:
                event = record.event
                if not _aggregate_model_event_is_in_window(event, query):
                    continue
                session_pricing.add_payload(
                    event_type=EventType(event.type),
                    session_id=session_id,
                    effective_on=normalize_aggregate_event_timestamp(event.timestamp).date(),
                    occurrences=1,
                    payload=event.payload,
                )
                if session_pricing.truncated:
                    break
            if session_pricing.truncated:
                break
    (
        session_pricing_items,
        session_pricing_group_count,
        session_pricing_accuracy,
    ) = session_pricing.result()

    totals.session_count = activity_session_count
    return UsageRollupStoreResult(
        reported_costs=reported.page(),
        as_of=as_of,
        start_at=query.start_at,
        end_at=query.end_at,
        totals=totals.totals(tool_calls=tool_calls),
        provider_breakdown=_bounded_in_memory_usage_breakdown(
            session_records,
            query=query,
            limit=query.group_limit,
            dimension="provider",
            candidates=provider_candidates,
        ),
        model_breakdown=_bounded_in_memory_usage_breakdown(
            session_records,
            query=query,
            limit=query.group_limit,
            dimension="model",
            candidates=model_candidates,
        ),
        session_breakdown=session_breakdown,
        pricing_inputs=pricing_items,
        pricing_inputs_included=query.include_pricing_inputs,
        pricing_input_group_count=pricing_group_count,
        pricing_inputs_accuracy=pricing_accuracy,
        session_pricing_inputs=session_pricing_items,
        session_pricing_inputs_included=(
            query.include_pricing_inputs and query.session_group_limit is not None
        ),
        session_pricing_input_group_count=session_pricing_group_count,
        session_pricing_inputs_accuracy=session_pricing_accuracy,
        active_session_count=active_session_count,
        matching_session_count=matching_session_count,
    )


def _retain_bounded_session_usage_group(
    retained: list[_SessionUsageCandidate],
    remainder: _SessionUsageRemainderAccumulator,
    group: _SessionUsageCandidate,
    *,
    limit: int,
) -> None:
    if len(retained) < limit:
        retained.append(group)
        return
    worst = max(retained, key=_session_usage_group_rank_key)
    if _session_usage_group_rank_key(group) < _session_usage_group_rank_key(worst):
        retained.remove(worst)
        remainder.add(worst)
        retained.append(group)
        return
    remainder.add(group)


def _in_memory_session_usage_breakdown(
    retained: list[_SessionUsageCandidate],
    remainder: _SessionUsageRemainderAccumulator,
    *,
    limit: int | None,
) -> UsageSessionAggregateBreakdown | None:
    if limit is None:
        return None
    groups = tuple(
        candidate.retained_group()
        for candidate in sorted(retained, key=_session_usage_group_rank_key)
    )
    remainder_result = remainder.result()
    return UsageSessionAggregateBreakdown(
        groups=groups,
        remainder=remainder_result,
        accuracy=(
            EXACT_AGGREGATE.model_copy()
            if remainder_result is None
            else AggregateAccuracy(
                kind=AggregateAccuracyKind.TRUNCATED,
                reason="Matching sessions exceed session_group_limit.",
                limit=limit,
            )
        ),
    )


def _session_usage_group_rank_key(
    group: _SessionUsageCandidate,
) -> tuple[int, int, str]:
    return (
        -group.totals.usage.total_tokens,
        -group.totals.model_steps,
        group.session_id,
    )


def _bounded_in_memory_usage_breakdown(
    session_records: _SessionRecordsFactory,
    *,
    query: UsageRollupQuery,
    limit: int,
    dimension: Literal["provider", "model"],
    candidates: _InMemoryUsageGroupCandidates,
) -> UsageAggregateBreakdown:
    accumulators = _accumulate_usage_group_batch(
        session_records,
        query=query,
        dimension=dimension,
        keys=candidates.keys,
    )
    visible_items = sorted(accumulators.items(), key=_usage_group_rank_key)[:limit]

    visible = tuple(
        UsageAggregateGroup(
            provider_name=key[0],
            model=key[1],
            totals=accumulator.totals(),
        )
        for key, accumulator in visible_items
    )
    if candidates.sampled:
        return UsageAggregateBreakdown(
            groups=visible,
            remainder=None,
            accuracy=AggregateAccuracy(
                kind=AggregateAccuracyKind.SAMPLED,
                reason=(
                    f"Distinct {dimension} groups exceed the bounded in-memory "
                    "heavy-hitter candidate limit."
                ),
                limit=candidates.limit,
            ),
        )

    group_count = len(candidates.keys)
    if group_count <= limit:
        return UsageAggregateBreakdown(
            groups=visible,
            remainder=None,
            accuracy=EXACT_AGGREGATE.model_copy(),
        )

    remainder = _accumulate_usage_remainder(
        session_records,
        query=query,
        dimension=dimension,
        visible_keys={(group.provider_name, group.model) for group in visible},
    )
    return UsageAggregateBreakdown(
        groups=visible,
        remainder=UsageAggregateRemainder(
            group_count=group_count - len(visible),
            totals=remainder.totals(),
        ),
        accuracy=AggregateAccuracy(
            kind=AggregateAccuracyKind.TRUNCATED,
            reason=f"Distinct {dimension} groups exceed group_limit.",
            limit=limit,
        ),
    )


def _accumulate_usage_group_batch(
    session_records: _SessionRecordsFactory,
    *,
    query: UsageRollupQuery,
    dimension: Literal["provider", "model"],
    keys: tuple[_UsageGroupKey, ...],
) -> dict[_UsageGroupKey, _UsageAccumulator]:
    accumulators = {key: _UsageAccumulator() for key in keys}
    for _, _, records in session_records():
        seen: set[_UsageGroupKey] = set()
        for record in records:
            event = record.event
            projected = _aggregate_usage_event_in_window(event, query)
            if projected is None:
                continue
            metrics, model_step = projected
            key = _usage_group_key(metrics, dimension=dimension)
            accumulator = accumulators.get(key)
            if accumulator is None:
                continue
            if model_step:
                accumulator.add(metrics)
            elif metrics is not None:
                accumulator.add_usage_only(metrics)
            seen.add(key)
        for key in seen:
            accumulators[key].session_count += 1
    return accumulators


def _accumulate_usage_remainder(
    session_records: _SessionRecordsFactory,
    *,
    query: UsageRollupQuery,
    dimension: Literal["provider", "model"],
    visible_keys: set[_UsageGroupKey],
) -> _UsageAccumulator:
    remainder = _UsageAccumulator()
    for _, _, records in session_records():
        session_has_remainder = False
        for record in records:
            event = record.event
            projected = _aggregate_usage_event_in_window(event, query)
            if projected is None:
                continue
            metrics, model_step = projected
            if _usage_group_key(metrics, dimension=dimension) in visible_keys:
                continue
            if model_step:
                remainder.add(metrics)
            elif metrics is not None:
                remainder.add_usage_only(metrics)
            session_has_remainder = True
        remainder.session_count += session_has_remainder
    return remainder


def _aggregate_model_event_is_in_window(event: Event, query: UsageRollupQuery) -> bool:
    if event.type == EventType.MODEL_HOSTED_TOOL_CALL:
        if aggregate_hosted_tool_usage_metrics_from_event_payload(event.payload) is None:
            return False
    elif event.type not in {EventType.MODEL_COMPLETED, EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED}:
        return False
    timestamp = normalize_aggregate_event_timestamp(event.timestamp)
    return query.start_at <= timestamp < query.end_at


def _aggregate_usage_event_in_window(
    event: Event,
    query: UsageRollupQuery,
) -> tuple[UsageMetrics | None, bool] | None:
    timestamp = normalize_aggregate_event_timestamp(event.timestamp)
    if timestamp < query.start_at or timestamp >= query.end_at:
        return None
    if event.type == EventType.MODEL_COMPLETED:
        return aggregate_usage_metrics_from_event_payload(event.payload), True
    if event.type == EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED:
        return aggregate_usage_metrics_from_event_payload(event.payload), False
    if event.type == EventType.MODEL_HOSTED_TOOL_CALL:
        metrics = aggregate_hosted_tool_usage_metrics_from_event_payload(event.payload)
        if metrics is not None:
            return metrics, False
    return None


def _usage_group_key(
    metrics: UsageMetrics | None,
    *,
    dimension: Literal["provider", "model"],
) -> _UsageGroupKey:
    provider_name = None if metrics is None else metrics.provider_name
    model = None if metrics is None or dimension == "provider" else metrics.model
    return provider_name, model


def _usage_group_identity_sort_key(key: _UsageGroupKey) -> _UsageGroupSortKey:
    return key[0] is None, key[0] or "", key[1] is None, key[1] or ""


def _usage_group_rank_key(
    item: tuple[_UsageGroupKey, _UsageAccumulator],
) -> tuple[int, int, bool, str, bool, str]:
    key, accumulator = item
    return (
        -accumulator.usage.total_tokens,
        -accumulator.model_steps,
        *_usage_group_identity_sort_key(key),
    )
