from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._validation import json_utf8_size_within_limit, require_clean_nonblank
from cayu.runtime.usage import CacheUsageMetrics, UsageMetrics


class AggregateAccuracyKind(StrEnum):
    """How completely an aggregate projection represents its requested scope."""

    EXACT = "exact"
    SAMPLED = "sampled"
    TRUNCATED = "truncated"


class AggregateAccuracy(BaseModel):
    """Truthfulness metadata for a bounded aggregate or one of its sections."""

    model_config = ConfigDict(extra="forbid")

    kind: AggregateAccuracyKind
    reason: str | None = None
    limit: StrictInt | None = Field(default=None, ge=1)

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, "reason")

    @model_validator(mode="after")
    def validate_details(self) -> AggregateAccuracy:
        if self.kind is AggregateAccuracyKind.EXACT:
            if self.reason is not None or self.limit is not None:
                raise ValueError("Exact aggregates cannot include truncation details.")
        elif self.reason is None:
            raise ValueError("Non-exact aggregates require a reason.")
        return self


EXACT_AGGREGATE = AggregateAccuracy(kind=AggregateAccuracyKind.EXACT)
MAX_AGGREGATE_USAGE_COUNTER = 2**63 - 1
MAX_USAGE_PRICING_INPUT_BYTES = 8 * 1024 * 1024
MAX_USAGE_PRICING_RAW_CANDIDATES = 5000

# Python's ``str.strip`` set, written explicitly so SQLite and PostgreSQL can
# apply the same identity contract instead of relying on database-specific
# ASCII whitespace defaults.
AGGREGATE_IDENTITY_TRIM_CHARACTERS = (
    "\u0009\u000a\u000b\u000c\u000d"
    "\u001c\u001d\u001e\u001f"
    "\u0020\u0085\u00a0\u1680"
    "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
    "\u2028\u2029\u202f\u205f\u3000"
)


class UsageAggregateTotals(BaseModel):
    """Identity-free activity and token totals for one aggregate scope."""

    model_config = ConfigDict(extra="forbid")

    session_count: StrictInt = Field(ge=0)
    model_steps: StrictInt = Field(ge=0)
    model_steps_with_usage: StrictInt = Field(ge=0)
    tool_calls: StrictInt = Field(ge=0)
    usage: UsageMetrics

    @field_validator("usage")
    @classmethod
    def require_identity_free_usage(cls, value: UsageMetrics) -> UsageMetrics:
        if (
            value.provider_name is not None
            or value.requested_model is not None
            or value.model is not None
            or value.billing_identity is not None
        ):
            raise ValueError("Aggregate usage totals cannot carry one model identity.")
        return value.model_copy(deep=True)

    @model_validator(mode="after")
    def validate_reported_steps(self) -> UsageAggregateTotals:
        if self.model_steps_with_usage > self.model_steps:
            raise ValueError("model_steps_with_usage cannot exceed model_steps.")
        return self


def add_aggregate_usage(left: UsageMetrics, right: UsageMetrics) -> UsageMetrics:
    """Add token counters while deliberately discarding per-step identity fields."""

    return UsageMetrics(
        input_tokens=left.input_tokens + right.input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        total_tokens=left.total_tokens + right.total_tokens,
        reasoning_output_tokens=(left.reasoning_output_tokens + right.reasoning_output_tokens),
        cache=CacheUsageMetrics(
            read_tokens=left.cache.read_tokens + right.cache.read_tokens,
            write_tokens=left.cache.write_tokens + right.cache.write_tokens,
            write_5m_tokens=left.cache.write_5m_tokens + right.cache.write_5m_tokens,
            write_1h_tokens=left.cache.write_1h_tokens + right.cache.write_1h_tokens,
            write_unknown_ttl_tokens=(
                left.cache.write_unknown_ttl_tokens + right.cache.write_unknown_ttl_tokens
            ),
            cached_input_tokens=(left.cache.cached_input_tokens + right.cache.cached_input_tokens),
            uncached_input_tokens=(
                left.cache.uncached_input_tokens + right.cache.uncached_input_tokens
            ),
        ),
    )


def aggregate_usage_metrics_from_event_payload(
    payload: dict[str, object],
) -> UsageMetrics | None:
    """Project normalized counters without trusting arbitrary event JSON.

    Aggregate queries intentionally consume only the durable ``usage_metrics``
    projection. A present object counts as reported usage; malformed counters
    contribute zero and malformed optional identity becomes unknown. This
    definition is simple enough for SQL stores to implement exactly and keeps a
    custom event from failing an otherwise valid operational report.
    """

    raw_metrics = payload.get("usage_metrics")
    if type(raw_metrics) is not dict:
        return None
    raw_metrics = cast("dict[str, object]", raw_metrics)
    raw_cache = raw_metrics.get("cache")
    raw_cache = {} if type(raw_cache) is not dict else cast("dict[str, object]", raw_cache)
    return UsageMetrics(
        provider_name=aggregate_identity_value(raw_metrics.get("provider_name")),
        requested_model=aggregate_identity_value(raw_metrics.get("requested_model")),
        model=aggregate_identity_value(raw_metrics.get("model")),
        input_tokens=_aggregate_counter(raw_metrics.get("input_tokens")),
        output_tokens=_aggregate_counter(raw_metrics.get("output_tokens")),
        total_tokens=_aggregate_counter(raw_metrics.get("total_tokens")),
        reasoning_output_tokens=_aggregate_counter(raw_metrics.get("reasoning_output_tokens")),
        cache=CacheUsageMetrics(
            read_tokens=_aggregate_counter(raw_cache.get("read_tokens")),
            write_tokens=_aggregate_counter(raw_cache.get("write_tokens")),
            write_5m_tokens=_aggregate_counter(raw_cache.get("write_5m_tokens")),
            write_1h_tokens=_aggregate_counter(raw_cache.get("write_1h_tokens")),
            write_unknown_ttl_tokens=_aggregate_counter(raw_cache.get("write_unknown_ttl_tokens")),
            cached_input_tokens=_aggregate_counter(raw_cache.get("cached_input_tokens")),
            uncached_input_tokens=_aggregate_counter(raw_cache.get("uncached_input_tokens")),
        ),
    )


def pricing_usage_metrics_from_event_payload(
    payload: dict[str, object],
    *,
    max_bytes: int = MAX_USAGE_PRICING_INPUT_BYTES,
) -> UsageMetrics | None:
    """Return strict price-relevant metrics with unbounded evidence removed."""

    from cayu.runtime.usage import usage_metrics_from_event_payload

    raw_metrics = payload.get("usage_metrics")
    if type(raw_metrics) is not dict:
        return None
    raw_metrics = cast("dict[str, object]", raw_metrics)
    event_identity = payload.get("billing_identity")
    bounded_payload_view: dict[str, object] = {"usage_metrics": _PricingMetricsMapping(raw_metrics)}
    if type(event_identity) is dict:
        bounded_payload_view["billing_identity"] = _EvidenceFreeMapping(
            cast("dict[str, object]", event_identity)
        )
    if not json_utf8_size_within_limit(bounded_payload_view, max_bytes):
        raise _UsagePricingInputTooLarge

    bounded_metrics = dict(raw_metrics)
    nested_identity = bounded_metrics.get("billing_identity")
    if type(nested_identity) is dict:
        bounded_metrics["billing_identity"] = _billing_identity_without_evidence(
            cast("dict[str, object]", nested_identity)
        )
    bounded_payload: dict[str, object] = {"usage_metrics": bounded_metrics}
    if type(event_identity) is dict:
        bounded_payload["billing_identity"] = _billing_identity_without_evidence(
            cast("dict[str, object]", event_identity)
        )
    try:
        metrics = usage_metrics_from_event_payload(bounded_payload)
    except (TypeError, ValueError):
        return None
    if metrics is not None and not _usage_metrics_within_aggregate_counter_limit(metrics):
        return None
    return metrics


class _UsagePricingInputTooLarge(ValueError):
    """A price-relevant event projection exceeds its application-memory bound."""


class _EvidenceFreeMapping(Mapping[str, object]):
    """Read-only identity view that skips unbounded evidence without copying it."""

    _EXCLUDED = frozenset({"request_evidence", "completion_evidence"})

    def __init__(self, value: dict[str, object]) -> None:
        self._value = value

    def __getitem__(self, key: str) -> object:
        if key in self._EXCLUDED:
            raise KeyError(key)
        return self._value[key]

    def __iter__(self) -> Iterator[str]:
        return (key for key in self._value if key not in self._EXCLUDED)

    def __len__(self) -> int:
        return len(self._value) - sum(key in self._value for key in self._EXCLUDED)


class _PricingMetricsMapping(Mapping[str, object]):
    """Read-only usage view that substitutes the evidence-free nested identity."""

    def __init__(self, value: dict[str, object]) -> None:
        self._value = value

    def __getitem__(self, key: str) -> object:
        item = self._value[key]
        if key == "billing_identity" and type(item) is dict:
            return _EvidenceFreeMapping(cast("dict[str, object]", item))
        return item

    def __iter__(self) -> Iterator[str]:
        return iter(self._value)

    def __len__(self) -> int:
        return len(self._value)


def _usage_metrics_within_aggregate_counter_limit(metrics: UsageMetrics) -> bool:
    return all(
        value <= MAX_AGGREGATE_USAGE_COUNTER
        for value in (
            metrics.input_tokens,
            metrics.output_tokens,
            metrics.total_tokens,
            metrics.reasoning_output_tokens,
            metrics.cache.read_tokens,
            metrics.cache.write_tokens,
            metrics.cache.write_5m_tokens,
            metrics.cache.write_1h_tokens,
            metrics.cache.write_unknown_ttl_tokens,
            metrics.cache.cached_input_tokens,
            metrics.cache.uncached_input_tokens,
        )
    )


def _billing_identity_without_evidence(identity: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in identity.items()
        if key not in {"request_evidence", "completion_evidence"}
    }


def _aggregate_counter(value: object) -> int:
    return value if type(value) is int and 0 <= value <= MAX_AGGREGATE_USAGE_COUNTER else 0


def aggregate_identity_value(value: object) -> str | None:
    """Return a valid aggregate identity or represent malformed input as unknown."""

    if type(value) is not str or not value or value != value.strip():
        return None
    return value


def normalize_aggregate_event_timestamp(value: datetime) -> datetime:
    """Normalize event time using the durable stores' naive-as-UTC convention."""

    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def usage_pricing_metrics_key(metrics: UsageMetrics | None) -> str | None:
    """Return the canonical store-independent key for price-relevant metrics."""

    if metrics is None:
        return None
    return json.dumps(
        metrics.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


class UsageAggregateGroup(BaseModel):
    """One provider/model grouping inside a bounded usage breakdown."""

    model_config = ConfigDict(extra="forbid")

    provider_name: str | None
    model: str | None
    totals: UsageAggregateTotals

    @field_validator("provider_name", "model")
    @classmethod
    def validate_optional_identity(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)


class UsageAggregateRemainder(BaseModel):
    """Exact totals represented by groups omitted from a bounded breakdown."""

    model_config = ConfigDict(extra="forbid")

    group_count: StrictInt = Field(ge=1)
    totals: UsageAggregateTotals


class UsageAggregateBreakdown(BaseModel):
    """Deterministic top groups plus an exact remainder when detail is bounded."""

    model_config = ConfigDict(extra="forbid")

    groups: tuple[UsageAggregateGroup, ...] = Field(max_length=100)
    remainder: UsageAggregateRemainder | None
    accuracy: AggregateAccuracy

    @model_validator(mode="after")
    def validate_accuracy(self) -> UsageAggregateBreakdown:
        if self.remainder is None and self.accuracy.kind is AggregateAccuracyKind.TRUNCATED:
            raise ValueError("A truncated breakdown must include an exact remainder.")
        if self.remainder is not None and self.accuracy.kind is not AggregateAccuracyKind.TRUNCATED:
            raise ValueError("A bounded breakdown with a remainder must report truncation.")
        return self


class UsagePricingInput(BaseModel):
    """One bounded group of identical, price-relevant model-step inputs."""

    model_config = ConfigDict(extra="forbid")

    effective_on: date
    occurrences: StrictInt = Field(ge=1)
    metrics: UsageMetrics | None = None

    @field_validator("metrics")
    @classmethod
    def copy_metrics(cls, value: UsageMetrics | None) -> UsageMetrics | None:
        return None if value is None else value.model_copy(deep=True)


def coalesce_usage_pricing_inputs(
    inputs: Iterable[UsagePricingInput],
) -> tuple[UsagePricingInput, ...]:
    """Merge store-native candidate groups after canonical usage validation."""

    grouped: dict[tuple[date, str | None], tuple[int, UsageMetrics | None]] = {}
    for item in inputs:
        metrics_key = usage_pricing_metrics_key(item.metrics)
        key = (item.effective_on, metrics_key)
        occurrences, _ = grouped.get(key, (0, item.metrics))
        grouped[key] = (occurrences + item.occurrences, item.metrics)
    return tuple(
        sorted(
            (
                UsagePricingInput(
                    effective_on=effective_on,
                    occurrences=occurrences,
                    metrics=metrics,
                )
                for (effective_on, _), (occurrences, metrics) in grouped.items()
            ),
            key=lambda item: (
                -item.occurrences,
                item.effective_on,
                "" if item.metrics is None else usage_pricing_metrics_key(item.metrics) or "",
            ),
        )
    )


@dataclass
class BoundedUsagePricingInputAccumulator:
    """Canonicalize price inputs within group-count and serialized-byte bounds."""

    limit: int
    max_bytes: int = MAX_USAGE_PRICING_INPUT_BYTES
    _groups: dict[tuple[date, str | None], tuple[int, UsageMetrics | None]] = dataclass_field(
        default_factory=dict,
        init=False,
    )
    _retained_bytes: int = dataclass_field(default=0, init=False)
    _truncated_group_count: int = dataclass_field(default=0, init=False)
    _truncation_reason: str | None = dataclass_field(default=None, init=False)
    _truncation_limit: int | None = dataclass_field(default=None, init=False)
    truncated: bool = dataclass_field(default=False, init=False)

    def __post_init__(self) -> None:
        if type(self.limit) is not int or self.limit < 1:
            raise ValueError("limit must be a positive integer.")
        if type(self.max_bytes) is not int or self.max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer.")

    def add(self, item: UsagePricingInput) -> None:
        if self.truncated:
            return
        metrics_key = usage_pricing_metrics_key(item.metrics)
        key = (item.effective_on, metrics_key)
        existing = self._groups.get(key)
        if existing is not None:
            self._groups[key] = (existing[0] + item.occurrences, existing[1])
            return
        observed_group_count = len(self._groups) + 1
        if len(self._groups) >= self.limit:
            self._truncate(
                group_count=observed_group_count,
                reason="Price-relevant model-step groups exceed pricing_input_limit.",
                limit=self.limit,
            )
            return
        retained_bytes = 0 if metrics_key is None else len(metrics_key.encode("utf-8"))
        if self._retained_bytes + retained_bytes > self.max_bytes:
            self._truncate(
                group_count=observed_group_count,
                reason="Price-relevant model-step groups exceed the serialized-byte limit.",
                limit=self.max_bytes,
            )
            return
        self._groups[key] = (item.occurrences, item.metrics)
        self._retained_bytes += retained_bytes

    def add_payload(
        self,
        *,
        effective_on: date,
        occurrences: int,
        payload: dict[str, object],
    ) -> None:
        """Validate and add one raw candidate without serializing it unboundedly."""

        if self.truncated:
            return
        try:
            metrics = pricing_usage_metrics_from_event_payload(
                payload,
                max_bytes=self.max_bytes,
            )
        except _UsagePricingInputTooLarge:
            self.reject_oversized_candidate()
            return
        self.add(
            UsagePricingInput(
                effective_on=effective_on,
                occurrences=occurrences,
                metrics=metrics,
            )
        )

    def reject_oversized_candidate(self) -> None:
        """Discard a store-classified candidate that cannot cross the memory boundary."""

        if self.truncated:
            return
        self._truncate(
            group_count=len(self._groups) + 1,
            reason="Price-relevant model-step groups exceed the serialized-byte limit.",
            limit=self.max_bytes,
        )

    def reject_candidate_row_overflow(
        self,
        *,
        limit: int = MAX_USAGE_PRICING_RAW_CANDIDATES,
    ) -> None:
        """Reject a raw store projection that exceeds bounded canonicalization work."""

        if self.truncated:
            return
        if type(limit) is not int or limit < 1:
            raise ValueError("Raw pricing candidate limit must be a positive integer.")
        self._truncate(
            group_count=len(self._groups) + 1,
            reason=(
                "Store-native price-input candidate rows exceed the bounded "
                "canonicalization-work limit."
            ),
            limit=limit,
        )

    def _truncate(self, *, group_count: int, reason: str, limit: int) -> None:
        self._groups.clear()
        self._retained_bytes = 0
        self._truncated_group_count = group_count
        self._truncation_reason = reason
        self._truncation_limit = limit
        self.truncated = True

    def result(
        self,
    ) -> tuple[tuple[UsagePricingInput, ...], int, AggregateAccuracy]:
        if self.truncated:
            assert self._truncation_reason is not None
            assert self._truncation_limit is not None
            return (
                (),
                self._truncated_group_count,
                AggregateAccuracy(
                    kind=AggregateAccuracyKind.TRUNCATED,
                    reason=self._truncation_reason,
                    limit=self._truncation_limit,
                ),
            )
        inputs = coalesce_usage_pricing_inputs(
            UsagePricingInput(
                effective_on=effective_on,
                occurrences=occurrences,
                metrics=metrics,
            )
            for (effective_on, _), (occurrences, metrics) in self._groups.items()
        )
        return inputs, len(inputs), EXACT_AGGREGATE.model_copy()


class UsageRollupStoreResult(BaseModel):
    """Bounded store projection used to construct the public usage rollup."""

    model_config = ConfigDict(extra="forbid")

    as_of: datetime
    start_at: datetime
    end_at: datetime
    totals: UsageAggregateTotals
    totals_accuracy: AggregateAccuracy = Field(default_factory=lambda: EXACT_AGGREGATE.model_copy())
    provider_breakdown: UsageAggregateBreakdown
    model_breakdown: UsageAggregateBreakdown
    pricing_inputs: tuple[UsagePricingInput, ...] = Field(default=(), max_length=5000)
    pricing_inputs_included: StrictBool = False
    pricing_input_group_count: StrictInt = Field(
        default=0,
        ge=0,
        description=(
            "Exact canonical group count when pricing_inputs_accuracy is exact; "
            "the number observed before bounded collection stopped when truncated."
        ),
    )
    pricing_inputs_accuracy: AggregateAccuracy = Field(
        default_factory=lambda: EXACT_AGGREGATE.model_copy()
    )
    active_session_count: StrictInt = Field(default=0, ge=0)
    matching_session_count: StrictInt = Field(default=0, ge=0)
    includes_active_sessions: StrictBool = True

    @field_validator("as_of", "start_at", "end_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime, info) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{info.field_name} must be timezone-aware.")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_scope(self) -> UsageRollupStoreResult:
        if self.start_at >= self.end_at:
            raise ValueError("Usage rollup start_at must be before end_at.")
        if self.pricing_input_group_count < len(self.pricing_inputs):
            raise ValueError("pricing_input_group_count cannot be smaller than returned groups.")
        if (
            self.pricing_input_group_count == len(self.pricing_inputs)
            and self.pricing_inputs_accuracy.kind is not AggregateAccuracyKind.EXACT
        ):
            raise ValueError("Complete pricing inputs must report exact accuracy.")
        if (
            self.pricing_input_group_count > len(self.pricing_inputs)
            and self.pricing_inputs_accuracy.kind is not AggregateAccuracyKind.TRUNCATED
        ):
            raise ValueError("Omitted pricing inputs must report truncation.")
        if (
            self.pricing_inputs_accuracy.kind is AggregateAccuracyKind.TRUNCATED
            and self.pricing_inputs
        ):
            raise ValueError("Truncated pricing projections cannot return partial inputs.")
        if not self.pricing_inputs_included and (
            self.pricing_inputs or self.pricing_input_group_count
        ):
            raise ValueError("Pricing inputs cannot be returned when they were not requested.")
        if (
            self.pricing_inputs_included
            and self.pricing_inputs_accuracy.kind is AggregateAccuracyKind.EXACT
            and sum(item.occurrences for item in self.pricing_inputs) != self.totals.model_steps
        ):
            raise ValueError("Exact pricing inputs must account for every model step.")
        for breakdown in (self.provider_breakdown, self.model_breakdown):
            if (
                self.totals_accuracy.kind is AggregateAccuracyKind.SAMPLED
                and breakdown.accuracy.kind is AggregateAccuracyKind.EXACT
            ):
                raise ValueError("A sampled usage rollup cannot include an exact breakdown.")
            if (
                self.totals_accuracy.kind is AggregateAccuracyKind.TRUNCATED
                and breakdown.accuracy.kind is not AggregateAccuracyKind.TRUNCATED
            ):
                raise ValueError("A truncated usage rollup must report truncated breakdowns.")
        return self


class UsageCurrencyCost(BaseModel):
    """Exact estimated cost for one currency; currencies are never combined."""

    model_config = ConfigDict(extra="forbid")

    currency: str
    model_steps: StrictInt = Field(ge=1)
    total_cost: Decimal = Field(ge=0)

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, value: str) -> str:
        return require_clean_nonblank(value, "currency").upper()


class UsageUnpricedReason(BaseModel):
    """Model-step count that could not be priced for one explicit reason."""

    model_config = ConfigDict(extra="forbid")

    reason: str
    model_steps: StrictInt = Field(ge=1)

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        return require_clean_nonblank(value, "reason")


class UsageCostRollup(BaseModel):
    """Bounded cost result derived from grouped price-relevant inputs."""

    model_config = ConfigDict(extra="forbid")

    price_book_version: str
    price_book_generated_at: str
    accuracy: AggregateAccuracy
    evaluated_model_steps: StrictInt = Field(ge=0)
    priced_model_steps: StrictInt = Field(ge=0)
    unpriced_model_steps: StrictInt = Field(ge=0)
    unevaluated_model_steps: StrictInt = Field(ge=0)
    currencies: tuple[UsageCurrencyCost, ...] = Field(max_length=5000)
    unpriced_reasons: tuple[UsageUnpricedReason, ...] = Field(max_length=5000)

    @field_validator("price_book_version", "price_book_generated_at")
    @classmethod
    def validate_price_book_identity(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_step_accounting(self) -> UsageCostRollup:
        if self.priced_model_steps + self.unpriced_model_steps != self.evaluated_model_steps:
            raise ValueError("Priced and unpriced steps must sum to evaluated_model_steps.")
        if sum(item.model_steps for item in self.currencies) != self.priced_model_steps:
            raise ValueError("Currency model-step counts must sum to priced_model_steps.")
        if sum(item.model_steps for item in self.unpriced_reasons) != self.unpriced_model_steps:
            raise ValueError("Unpriced reasons must sum to unpriced_model_steps.")
        if self.accuracy.kind is AggregateAccuracyKind.EXACT and self.unevaluated_model_steps:
            raise ValueError("Exact cost rollups cannot contain unevaluated model steps.")
        if self.accuracy.kind is AggregateAccuracyKind.TRUNCATED and self.evaluated_model_steps:
            raise ValueError("Truncated cost rollups cannot report partial evaluated totals.")
        return self


def estimate_usage_rollup_cost(
    result: UsageRollupStoreResult,
    pricing,
) -> UsageCostRollup:
    """Price an exact bounded input projection without reporting partial totals."""

    from cayu.runtime.costs import PriceBook, estimate_model_step_cost

    if type(result) is not UsageRollupStoreResult:
        raise TypeError("result must be a UsageRollupStoreResult.")
    if type(pricing) is not PriceBook:
        raise TypeError("pricing must be a PriceBook.")
    if not result.pricing_inputs_included:
        raise ValueError("Usage rollup did not request pricing inputs.")
    cost_accuracy = _combined_aggregate_accuracy(
        result.totals_accuracy,
        result.pricing_inputs_accuracy,
    )
    if (
        result.pricing_inputs_accuracy.kind is not AggregateAccuracyKind.EXACT
        or result.totals_accuracy.kind is AggregateAccuracyKind.TRUNCATED
    ):
        return UsageCostRollup(
            price_book_version=pricing.price_book_version,
            price_book_generated_at=pricing.generated_at,
            accuracy=cost_accuracy,
            evaluated_model_steps=0,
            priced_model_steps=0,
            unpriced_model_steps=0,
            unevaluated_model_steps=result.totals.model_steps,
            currencies=(),
            unpriced_reasons=(),
        )

    currency_costs: dict[str, Decimal] = {}
    currency_steps: dict[str, int] = {}
    unpriced_reasons: dict[str, int] = {}
    priced_model_steps = 0
    unpriced_model_steps = 0
    for item in result.pricing_inputs:
        if item.metrics is None:
            reason = "model.completed event has no valid normalized usage metrics"
            unpriced_reasons[reason] = unpriced_reasons.get(reason, 0) + item.occurrences
            unpriced_model_steps += item.occurrences
            continue
        estimate = estimate_model_step_cost(
            metrics=item.metrics,
            pricing=pricing,
            effective_on=item.effective_on,
        )
        if not estimate.priced:
            reason = estimate.missing_pricing_reason or "no matching model pricing"
            unpriced_reasons[reason] = unpriced_reasons.get(reason, 0) + item.occurrences
            unpriced_model_steps += item.occurrences
            continue
        assert estimate.currency is not None
        currency = estimate.currency.upper()
        currency_costs[currency] = currency_costs.get(currency, Decimal(0)) + (
            estimate.total_cost * item.occurrences
        )
        currency_steps[currency] = currency_steps.get(currency, 0) + item.occurrences
        priced_model_steps += item.occurrences

    return UsageCostRollup(
        price_book_version=pricing.price_book_version,
        price_book_generated_at=pricing.generated_at,
        accuracy=cost_accuracy,
        evaluated_model_steps=priced_model_steps + unpriced_model_steps,
        priced_model_steps=priced_model_steps,
        unpriced_model_steps=unpriced_model_steps,
        unevaluated_model_steps=0,
        currencies=tuple(
            UsageCurrencyCost(
                currency=currency,
                model_steps=currency_steps[currency],
                total_cost=currency_costs[currency],
            )
            for currency in sorted(currency_costs)
        ),
        unpriced_reasons=tuple(
            UsageUnpricedReason(reason=reason, model_steps=model_steps)
            for reason, model_steps in sorted(
                unpriced_reasons.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ),
    )


def _combined_aggregate_accuracy(
    left: AggregateAccuracy,
    right: AggregateAccuracy,
) -> AggregateAccuracy:
    """Return the least-complete accuracy without hiding either limitation."""

    if left.kind is AggregateAccuracyKind.EXACT:
        return right.model_copy(deep=True)
    if right.kind is AggregateAccuracyKind.EXACT:
        return left.model_copy(deep=True)

    kind = (
        AggregateAccuracyKind.TRUNCATED
        if AggregateAccuracyKind.TRUNCATED in {left.kind, right.kind}
        else AggregateAccuracyKind.SAMPLED
    )
    reasons = tuple(dict.fromkeys(item.reason for item in (left, right) if item.reason))
    limits = {item.limit for item in (left, right) if item.limit is not None}
    return AggregateAccuracy(
        kind=kind,
        reason="; ".join(reasons),
        limit=next(iter(limits)) if len(limits) == 1 else None,
    )
