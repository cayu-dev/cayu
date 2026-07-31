from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    StrictBool,
    StrictInt,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticSerializationError

from cayu._validation import (
    JsonUtf8SizeCounter,
    json_utf8_size_within_limit,
    require_clean_nonblank,
)
from cayu.core.billing import BillingIdentity
from cayu.core.events import Event, EventType
from cayu.runtime.usage import (
    AggregateCacheUsageMetrics,  # noqa: F401 - compatibility re-export
    AggregateCount,
    AggregateUsageMetrics,
    CacheUsageMetrics,
    PositiveAggregateCount,
    UsageMetrics,
    add_aggregate_usage,
    build_aggregate_usage_metrics,
)

if TYPE_CHECKING:
    from cayu.runtime.sessions import UsageRollupQuery


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
MAX_USAGE_ROLLUP_SESSION_ID_BYTES = 1024
MAX_USAGE_ROLLUP_STORE_RESULT_BYTES = 24 * 1024 * 1024


class UsageRollupInconsistent(ValueError):
    """A store returned a usage projection that does not honor its query."""


class UsageRollupResultTooLarge(ValueError):
    """A usage projection cannot cross its bounded public contract safely."""


def _validation_error_is_only_usage_rollup_size_limits(error: ValidationError) -> bool:
    """Whether canonical validation failed exclusively on explicit byte ceilings."""

    details = error.errors(include_url=False, include_input=False)
    return bool(details) and all(
        isinstance((detail.get("ctx") or {}).get("error"), UsageRollupResultTooLarge)
        for detail in details
    )


def require_bounded_usage_session_id(value: str) -> str:
    """Validate a session identity before it enters a usage projection."""

    cleaned = require_clean_nonblank(value, "session_id")
    try:
        encoded = cleaned.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("session_id must contain portable Unicode text.") from exc
    if len(encoded) > MAX_USAGE_ROLLUP_SESSION_ID_BYTES:
        raise UsageRollupResultTooLarge("A session identity exceeds the usage-rollup byte limit.")
    return cleaned


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

    session_count: AggregateCount = Field(ge=0)
    model_steps: AggregateCount = Field(ge=0)
    model_steps_with_usage: AggregateCount = Field(ge=0)
    tool_calls: AggregateCount = Field(ge=0)
    usage: AggregateUsageMetrics

    @model_validator(mode="after")
    def validate_reported_steps(self) -> UsageAggregateTotals:
        if self.model_steps_with_usage > self.model_steps:
            raise ValueError("model_steps_with_usage cannot exceed model_steps.")
        return self


def _sum_usage_aggregate_totals(
    values: Iterable[UsageAggregateTotals],
) -> UsageAggregateTotals:
    session_count = 0
    model_steps = 0
    model_steps_with_usage = 0
    tool_calls = 0
    usage = build_aggregate_usage_metrics()
    for value in values:
        session_count += value.session_count
        model_steps += value.model_steps
        model_steps_with_usage += value.model_steps_with_usage
        tool_calls += value.tool_calls
        usage = add_aggregate_usage(usage, value.usage)
    return UsageAggregateTotals(
        session_count=session_count,
        model_steps=model_steps,
        model_steps_with_usage=model_steps_with_usage,
        tool_calls=tool_calls,
        usage=usage,
    )


def aggregate_usage_metrics_from_event_payload(
    payload: dict[str, object],
) -> UsageMetrics | None:
    """Project normalized counters without trusting arbitrary event JSON.

    Aggregate queries intentionally consume only the durable ``usage_metrics``
    projection. An explicit normalization failure is authoritative; otherwise,
    a present object counts as reported usage, malformed counters contribute
    zero, and malformed optional identity becomes unknown. This definition is
    simple enough for SQL stores to implement exactly and keeps a custom event
    from failing an otherwise valid operational report.
    """

    if payload.get("usage_normalization_failed") is True:
        return None
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


def summary_usage_metrics_from_event_payload(
    payload: dict[str, object],
) -> UsageMetrics | None:
    """Return metrics for an application-memory usage summary.

    A durable normalized projection is authoritative and uses the same
    tolerant parsing contract as native store rollups. The strict raw-usage
    normalization fallback is retained only for older events that do not carry
    ``usage_metrics``.
    """

    if "usage_metrics" in payload:
        return aggregate_usage_metrics_from_event_payload(payload)

    from cayu.runtime.usage import usage_metrics_from_event_payload

    return usage_metrics_from_event_payload(payload)


def project_aggregate_usage_inspection_event(
    event: Event,
) -> tuple[Event, UsageMetrics | None]:
    """Return the bounded event projection and metrics consumed by inspection.

    Parsing happens before projection with the same tolerant contract as native
    rollups: malformed optional identity and counter fields become unknown or
    zero without discarding otherwise valid counters.
    """

    metrics = (
        aggregate_usage_metrics_from_event_payload(event.payload)
        if event.type == EventType.MODEL_COMPLETED
        else None
    )
    payload = {"usage_metrics": metrics.model_dump(mode="json")} if metrics is not None else {}
    return event.model_copy(update={"payload": payload}), metrics


def pricing_usage_metrics_from_event_payload(
    payload: dict[str, object],
    *,
    max_bytes: int = MAX_USAGE_PRICING_INPUT_BYTES,
) -> UsageMetrics | None:
    """Return strict price-relevant metrics with unbounded evidence removed."""

    from cayu.runtime.usage import usage_metrics_from_event_payload

    if payload.get("usage_normalization_failed") is True:
        return None
    raw_metrics = payload.get("usage_metrics")
    if type(raw_metrics) is not dict:
        return None
    raw_metrics = cast("dict[str, object]", raw_metrics)
    event_identity = payload.get("billing_identity")
    bounded_metrics = dict(raw_metrics)
    nested_identity = bounded_metrics.get("billing_identity")
    if type(nested_identity) is dict:
        bounded_metrics["billing_identity"] = _billing_identity_for_aggregate(
            cast("dict[str, object]", nested_identity)
        )
    bounded_payload_view: dict[str, object] = {"usage_metrics": bounded_metrics}
    if type(event_identity) is dict:
        bounded_payload_view["billing_identity"] = _billing_identity_for_aggregate(
            cast("dict[str, object]", event_identity)
        )
    if not json_utf8_size_within_limit(bounded_payload_view, max_bytes):
        raise _UsagePricingInputTooLarge

    bounded_payload: dict[str, object] = {"usage_metrics": bounded_metrics}
    if type(event_identity) is dict:
        bounded_payload["billing_identity"] = _billing_identity_for_aggregate(
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


_BEDROCK_AGGREGATE_REQUEST_EVIDENCE = (
    "source_region",
    "resource_type",
    "profile_scope",
    "requested_service_tier",
)
_BEDROCK_AGGREGATE_COMPLETION_EVIDENCE = ("effective_service_tier",)


def _billing_identity_for_aggregate(identity: dict[str, object]) -> dict[str, object]:
    """Retain only bounded, Cayu-owned commercial display fields.

    Provider evidence is arbitrary durable JSON and must not enter an aggregate
    response by default. Bedrock's adapter owns a small string-only schema used by
    its pricing diagnostics, so those exact fields are safe to project while all
    unknown evidence (including customer data) remains excluded.
    """

    projected = {
        key: value
        for key, value in identity.items()
        if key not in {"request_evidence", "completion_evidence"}
    }
    if identity.get("provider_name") != "bedrock":
        return projected
    for field_name, allowed_fields in (
        ("request_evidence", _BEDROCK_AGGREGATE_REQUEST_EVIDENCE),
        ("completion_evidence", _BEDROCK_AGGREGATE_COMPLETION_EVIDENCE),
    ):
        raw_evidence = identity.get(field_name)
        if type(raw_evidence) is not dict:
            continue
        evidence = cast("dict[str, object]", raw_evidence)
        selected = {
            key: value
            for key in allowed_fields
            if type(value := evidence.get(key)) is str and value and value == value.strip()
        }
        if selected:
            projected[field_name] = selected
    return projected


def _billing_identity_for_breakdown(identity: BillingIdentity) -> UsageBillingIdentity:
    """Return the safe display projection used by aggregate response groups."""

    projected = _billing_identity_for_aggregate(identity.model_dump(mode="json"))
    request_evidence = projected.get("request_evidence", {})
    completion_evidence = projected.get("completion_evidence", {})
    assert type(request_evidence) is dict
    assert type(completion_evidence) is dict
    return UsageBillingIdentity(
        provider_name=identity.provider_name,
        resource_id=identity.resource_id,
        request_evidence=cast("dict[str, str]", request_evidence),
        completion_evidence=cast("dict[str, str]", completion_evidence),
    )


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

    group_count: PositiveAggregateCount = Field(ge=1)
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


UsageSessionStatus = Literal[
    "pending",
    "running",
    "interrupting",
    "completed",
    "failed",
    "interrupted",
]
_ACTIVE_USAGE_SESSION_STATUSES = frozenset({"pending", "running", "interrupting"})


class UsageSessionAggregateGroup(BaseModel):
    """Exact usage for one retained current session inside a bounded breakdown."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    status: UsageSessionStatus
    active: StrictBool
    totals: UsageAggregateTotals

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        return require_bounded_usage_session_id(value)

    @model_validator(mode="after")
    def validate_active_state(self) -> UsageSessionAggregateGroup:
        if self.active is not (self.status in _ACTIVE_USAGE_SESSION_STATUSES):
            raise ValueError("Session usage active state must match the current status.")
        if self.totals.session_count not in {0, 1}:
            raise ValueError("One session usage group can report at most one activity session.")
        has_activity = bool(self.totals.model_steps or self.totals.tool_calls)
        if self.totals.session_count != int(has_activity):
            raise ValueError("Session usage count must match activity in the requested window.")
        return self


class UsageSessionAggregateRemainder(BaseModel):
    """Exact usage represented by current sessions omitted from bounded detail."""

    model_config = ConfigDict(extra="forbid")

    group_count: PositiveAggregateCount = Field(ge=1)
    active_session_count: AggregateCount = Field(ge=0)
    totals: UsageAggregateTotals

    @model_validator(mode="after")
    def validate_counts(self) -> UsageSessionAggregateRemainder:
        if self.active_session_count > self.group_count:
            raise ValueError("Remainder active sessions cannot exceed omitted groups.")
        if self.totals.session_count > self.group_count:
            raise ValueError("Remainder activity sessions cannot exceed omitted groups.")
        return self


class UsageSessionAggregateBreakdown(BaseModel):
    """Deterministic session groups plus an exact aggregate remainder."""

    model_config = ConfigDict(extra="forbid")

    groups: tuple[UsageSessionAggregateGroup, ...] = Field(max_length=100)
    remainder: UsageSessionAggregateRemainder | None
    accuracy: AggregateAccuracy

    @model_validator(mode="after")
    def validate_breakdown(self) -> UsageSessionAggregateBreakdown:
        session_ids = [group.session_id for group in self.groups]
        if len(set(session_ids)) != len(session_ids):
            raise ValueError("Session usage groups must have distinct session IDs.")
        if self.remainder is None and self.accuracy.kind is AggregateAccuracyKind.TRUNCATED:
            raise ValueError("A truncated session breakdown must include a remainder.")
        if self.remainder is not None and self.accuracy.kind is not AggregateAccuracyKind.TRUNCATED:
            raise ValueError("A session breakdown remainder must report truncation.")
        if self.groups != tuple(
            sorted(
                self.groups,
                key=lambda group: (
                    -group.totals.usage.total_tokens,
                    -group.totals.model_steps,
                    group.session_id,
                ),
            )
        ):
            raise ValueError("Session usage groups must use canonical usage ordering.")
        return self


class UsagePricingInput(BaseModel):
    """One bounded group of identical, price-relevant model-step inputs."""

    model_config = ConfigDict(extra="forbid")

    session_id: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    effective_on: date
    occurrences: StrictInt = Field(ge=1)
    metrics: UsageMetrics | None = None

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_bounded_usage_session_id(value)

    @field_validator("metrics")
    @classmethod
    def copy_metrics(cls, value: UsageMetrics | None) -> UsageMetrics | None:
        return None if value is None else value.model_copy(deep=True)


def coalesce_usage_pricing_inputs(
    inputs: Iterable[UsagePricingInput],
) -> tuple[UsagePricingInput, ...]:
    """Merge store-native candidate groups after canonical usage validation."""

    grouped: dict[
        tuple[str | None, date, str | None],
        tuple[int, UsageMetrics | None],
    ] = {}
    for item in inputs:
        metrics_key = usage_pricing_metrics_key(item.metrics)
        key = (item.session_id, item.effective_on, metrics_key)
        occurrences, _ = grouped.get(key, (0, item.metrics))
        grouped[key] = (occurrences + item.occurrences, item.metrics)
    return tuple(
        sorted(
            (
                UsagePricingInput(
                    session_id=session_id,
                    effective_on=effective_on,
                    occurrences=occurrences,
                    metrics=metrics,
                )
                for (session_id, effective_on, _), (occurrences, metrics) in grouped.items()
            ),
            key=lambda item: (
                item.session_id is None,
                item.session_id or "",
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
    _groups: dict[
        tuple[str | None, date, str | None],
        tuple[int, UsageMetrics | None],
    ] = dataclass_field(default_factory=dict, init=False)
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
        key = (item.session_id, item.effective_on, metrics_key)
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
        retained_bytes = (
            0 if item.session_id is None else len(item.session_id.encode("utf-8"))
        ) + (0 if metrics_key is None else len(metrics_key.encode("utf-8")))
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
        session_id: str | None = None,
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
                session_id=session_id,
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
                session_id=session_id,
                effective_on=effective_on,
                occurrences=occurrences,
                metrics=metrics,
            )
            for (session_id, effective_on, _), (occurrences, metrics) in self._groups.items()
        )
        return inputs, len(inputs), EXACT_AGGREGATE.model_copy()


def _session_pricing_remainder_groups(
    *,
    shared: tuple[UsagePricingInput, ...],
    retained: tuple[UsagePricingInput, ...],
) -> dict[tuple[date, str | None], tuple[int, UsageMetrics | None]]:
    """Subtract retained occurrences without allocating another input projection."""

    shared_groups: dict[
        tuple[date, str | None],
        tuple[int, UsageMetrics | None],
    ] = {}
    for item in shared:
        key = (item.effective_on, usage_pricing_metrics_key(item.metrics))
        occurrences, _ = shared_groups.get(key, (0, item.metrics))
        shared_groups[key] = occurrences + item.occurrences, item.metrics
    retained_occurrences: dict[tuple[date, str | None], int] = {}
    for item in retained:
        key = (item.effective_on, usage_pricing_metrics_key(item.metrics))
        retained_occurrences[key] = retained_occurrences.get(key, 0) + item.occurrences

    remainder: dict[tuple[date, str | None], tuple[int, UsageMetrics | None]] = {}
    for key, (shared_count, metrics) in shared_groups.items():
        retained_count = retained_occurrences.pop(key, 0)
        if retained_count > shared_count:
            raise ValueError("Session pricing inputs exceed shared pricing occurrences.")
        if retained_count < shared_count:
            remainder[key] = shared_count - retained_count, metrics
    if retained_occurrences:
        raise ValueError("Session pricing inputs are absent from shared pricing inputs.")
    return remainder


def _session_pricing_input_remainder(
    *,
    shared: tuple[UsagePricingInput, ...],
    retained: tuple[UsagePricingInput, ...],
) -> tuple[UsagePricingInput, ...]:
    """Build exact canonical inputs for sessions omitted from bounded detail."""

    remainder = _session_pricing_remainder_groups(shared=shared, retained=retained)
    return coalesce_usage_pricing_inputs(
        UsagePricingInput(
            effective_on=effective_on,
            occurrences=occurrences,
            metrics=metrics,
        )
        for (effective_on, _), (occurrences, metrics) in remainder.items()
    )


def _scaled_pricing_usage(
    metrics: UsageMetrics,
    occurrences: int,
) -> AggregateUsageMetrics:
    """Scale one valid pricing input into identity-free aggregate counters."""

    return build_aggregate_usage_metrics(
        input_tokens=metrics.input_tokens * occurrences,
        output_tokens=metrics.output_tokens * occurrences,
        total_tokens=metrics.total_tokens * occurrences,
        reasoning_output_tokens=metrics.reasoning_output_tokens * occurrences,
        cache_read_tokens=metrics.cache.read_tokens * occurrences,
        cache_write_tokens=metrics.cache.write_tokens * occurrences,
        cache_write_5m_tokens=metrics.cache.write_5m_tokens * occurrences,
        cache_write_1h_tokens=metrics.cache.write_1h_tokens * occurrences,
        cache_write_unknown_ttl_tokens=(metrics.cache.write_unknown_ttl_tokens * occurrences),
        cached_input_tokens=metrics.cache.cached_input_tokens * occurrences,
        uncached_input_tokens=metrics.cache.uncached_input_tokens * occurrences,
    )


def _aggregate_usage_counter_values(
    usage: AggregateUsageMetrics,
) -> tuple[int, ...]:
    """Return counters in one stable order for component-wise validation."""

    return (
        usage.input_tokens,
        usage.output_tokens,
        usage.total_tokens,
        usage.reasoning_output_tokens,
        usage.cache.read_tokens,
        usage.cache.write_tokens,
        usage.cache.write_5m_tokens,
        usage.cache.write_1h_tokens,
        usage.cache.write_unknown_ttl_tokens,
        usage.cache.cached_input_tokens,
        usage.cache.uncached_input_tokens,
    )


_SESSION_PRICING_ATTRIBUTION_DOMAIN = b"cayu.session-pricing-attribution.v1\x00"


def _session_pricing_attribution_fingerprint(
    inputs: Iterable[UsagePricingInput],
) -> str:
    """Commit to every price-relevant field for one retained session."""

    digest = hashlib.sha256(_SESSION_PRICING_ATTRIBUTION_DOMAIN)
    canonical_inputs: list[tuple[bytes, bytes, bytes]] = []
    for item in inputs:
        metrics_key = (
            b"\x00"
            if item.metrics is None
            else b"\x01" + (usage_pricing_metrics_key(item.metrics) or "").encode("utf-8")
        )
        canonical_inputs.append(
            (
                item.effective_on.isoformat().encode("ascii"),
                str(item.occurrences).encode("ascii"),
                metrics_key,
            )
        )
    for components in sorted(canonical_inputs):
        for component in components:
            digest.update(len(component).to_bytes(8, byteorder="big", signed=False))
            digest.update(component)
    return digest.hexdigest()


def _session_pricing_attribution_commitment(
    *,
    session_ids: Iterable[str],
    inputs: Iterable[UsagePricingInput],
) -> tuple[tuple[str, str], ...]:
    inputs_by_session: dict[str, list[UsagePricingInput]] = {}
    for item in inputs:
        if item.session_id is None:
            raise ValueError("Session pricing attribution requires session identity.")
        inputs_by_session.setdefault(item.session_id, []).append(item)
    return tuple(
        (
            session_id,
            _session_pricing_attribution_fingerprint(
                inputs_by_session.get(session_id, ()),
            ),
        )
        for session_id in session_ids
    )


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
    session_breakdown: UsageSessionAggregateBreakdown | None = None
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
    session_pricing_inputs: tuple[UsagePricingInput, ...] = Field(
        default=(),
        max_length=5000,
    )
    session_pricing_inputs_included: StrictBool = False
    session_pricing_input_group_count: StrictInt = Field(default=0, ge=0)
    session_pricing_inputs_accuracy: AggregateAccuracy = Field(
        default_factory=lambda: EXACT_AGGREGATE.model_copy()
    )
    active_session_count: StrictInt = Field(default=0, ge=0)
    matching_session_count: StrictInt = Field(default=0, ge=0)
    includes_active_sessions: StrictBool = True
    _session_pricing_attribution: tuple[tuple[str, str], ...] = PrivateAttr(default=())

    @field_validator("as_of", "start_at", "end_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime, info) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{info.field_name} must be timezone-aware.")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_scope(self) -> UsageRollupStoreResult:
        self._validate_projection_invariants()
        if not self._session_pricing_attribution:
            self._session_pricing_attribution = self._current_session_pricing_attribution()
        return self

    def _current_session_pricing_attribution(self) -> tuple[tuple[str, str], ...]:
        if (
            not self.session_pricing_inputs_included
            or self.session_pricing_inputs_accuracy.kind is not AggregateAccuracyKind.EXACT
            or self.session_breakdown is None
        ):
            return ()
        return _session_pricing_attribution_commitment(
            session_ids=(group.session_id for group in self.session_breakdown.groups),
            inputs=self.session_pricing_inputs,
        )

    def _validate_projection_invariants(self) -> None:
        """Validate conservation rules across independently bounded projections."""

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
        if any(item.session_id is not None for item in self.pricing_inputs):
            raise ValueError("Shared pricing inputs cannot retain session identity.")
        pricing_input_keys = [
            (item.effective_on, usage_pricing_metrics_key(item.metrics))
            for item in self.pricing_inputs
        ]
        if len(set(pricing_input_keys)) != len(pricing_input_keys):
            raise ValueError("Shared pricing inputs must have distinct canonical keys.")
        if (
            self.pricing_inputs_included
            and self.pricing_inputs_accuracy.kind is AggregateAccuracyKind.EXACT
            and sum(item.occurrences for item in self.pricing_inputs) != self.totals.model_steps
        ):
            raise ValueError("Exact pricing inputs must account for every model step.")
        if self.session_pricing_input_group_count < len(self.session_pricing_inputs):
            raise ValueError(
                "session_pricing_input_group_count cannot be smaller than returned groups."
            )
        if (
            self.session_pricing_input_group_count == len(self.session_pricing_inputs)
            and self.session_pricing_inputs_accuracy.kind is not AggregateAccuracyKind.EXACT
        ):
            raise ValueError("Complete session pricing inputs must report exact accuracy.")
        if (
            self.session_pricing_input_group_count > len(self.session_pricing_inputs)
            and self.session_pricing_inputs_accuracy.kind is not AggregateAccuracyKind.TRUNCATED
        ):
            raise ValueError("Omitted session pricing inputs must report truncation.")
        if (
            self.session_pricing_inputs_accuracy.kind is AggregateAccuracyKind.TRUNCATED
            and self.session_pricing_inputs
        ):
            raise ValueError("Truncated session pricing projections cannot return partial inputs.")
        if not self.session_pricing_inputs_included and (
            self.session_pricing_inputs or self.session_pricing_input_group_count
        ):
            raise ValueError(
                "Session pricing inputs cannot be returned when they were not requested."
            )
        if self.session_pricing_inputs_included and self.session_breakdown is None:
            raise ValueError("Session pricing inputs require a session breakdown.")
        if self.session_pricing_inputs_included and not self.pricing_inputs_included:
            raise ValueError("Session pricing inputs require the shared pricing projection.")
        visible_session_ids = (
            set()
            if self.session_breakdown is None
            else {group.session_id for group in self.session_breakdown.groups}
        )
        if any(
            item.session_id is None or item.session_id not in visible_session_ids
            for item in self.session_pricing_inputs
        ):
            raise ValueError("Session pricing inputs must belong to retained session groups.")
        session_pricing_input_keys = [
            (
                item.session_id,
                item.effective_on,
                usage_pricing_metrics_key(item.metrics),
            )
            for item in self.session_pricing_inputs
        ]
        if len(set(session_pricing_input_keys)) != len(session_pricing_input_keys):
            raise ValueError("Session pricing inputs must have distinct canonical keys.")
        self._validate_exact_session_pricing_attribution()
        self._validate_exact_session_pricing_projection()
        if self.session_breakdown is not None:
            remainder = self.session_breakdown.remainder
            represented_group_count = len(self.session_breakdown.groups)
            represented_active_count = sum(
                int(group.active) for group in self.session_breakdown.groups
            )
            represented_totals = [group.totals for group in self.session_breakdown.groups]
            if remainder is not None:
                represented_group_count += remainder.group_count
                represented_active_count += remainder.active_session_count
                represented_totals.append(remainder.totals)
            if represented_group_count != self.matching_session_count:
                raise ValueError(
                    "Session breakdown groups must account for every matching session."
                )
            if represented_active_count != self.active_session_count:
                raise ValueError("Session breakdown groups must account for every active session.")
            if _sum_usage_aggregate_totals(represented_totals) != self.totals:
                raise ValueError("Session breakdown totals must equal shared usage totals.")
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

    def validate_for_query(self, query: UsageRollupQuery) -> UsageRollupStoreResult:
        """Return a bounded, canonically revalidated result for one query."""

        from cayu.runtime.sessions import UsageRollupQuery

        if type(query) is not UsageRollupQuery:
            raise TypeError("Usage rollup result validation requires a UsageRollupQuery.")
        try:
            size_counter = JsonUtf8SizeCounter(MAX_USAGE_ROLLUP_STORE_RESULT_BYTES)
            if not size_counter.value(self):
                if size_counter.encountered_unsupported_value:
                    raise UsageRollupInconsistent(
                        "Usage rollup store result contains a non-serializable value."
                    )
                if size_counter.exceeded_limit:
                    raise UsageRollupResultTooLarge(
                        "Usage rollup store result exceeds its internal byte limit."
                    )
                raise UsageRollupInconsistent(
                    "Usage rollup store result could not be size-classified."
                )
            payload = self.model_dump(
                mode="python",
                round_trip=True,
                warnings="error",
            )
            try:
                validated = UsageRollupStoreResult.model_validate(payload)
            except ValidationError as exc:
                if _validation_error_is_only_usage_rollup_size_limits(exc):
                    raise UsageRollupResultTooLarge(
                        "Usage rollup store result exceeds a nested byte limit."
                    ) from exc
                raise
            validated._validate_query_contract(query)
            if self._session_pricing_attribution != validated._session_pricing_attribution:
                raise UsageRollupInconsistent(
                    "Session pricing attribution changed after store projection."
                )
        except UsageRollupResultTooLarge:
            raise
        except UsageRollupInconsistent:
            raise
        except (
            AssertionError,
            AttributeError,
            PydanticSerializationError,
            RecursionError,
            TypeError,
            ValueError,
        ) as exc:
            raise UsageRollupInconsistent(
                "Usage rollup result violates its projection invariants."
            ) from exc
        return validated

    def _validate_query_contract(self, query: UsageRollupQuery) -> None:
        """Check requested opt-ins and limits before publishing a store result."""

        if self.start_at != query.start_at or self.end_at != query.end_at:
            raise UsageRollupInconsistent(
                "Usage rollup result window does not match the requested window."
            )
        for name, breakdown in (
            ("provider", self.provider_breakdown),
            ("model", self.model_breakdown),
        ):
            if len(breakdown.groups) > query.group_limit:
                raise UsageRollupInconsistent(
                    f"Usage rollup {name} groups exceed the requested limit."
                )
            if (
                breakdown.accuracy.kind is AggregateAccuracyKind.TRUNCATED
                and breakdown.accuracy.limit != query.group_limit
            ):
                raise UsageRollupInconsistent(
                    f"Usage rollup {name} truncation does not match the requested limit."
                )

        session_projection_requested = query.session_group_limit is not None
        if (self.session_breakdown is not None) is not session_projection_requested:
            raise UsageRollupInconsistent(
                "Usage rollup session projection does not match the query."
            )
        if self.session_breakdown is not None:
            session_group_limit = query.session_group_limit
            if session_group_limit is None:
                raise UsageRollupInconsistent(
                    "Usage rollup returned an unrequested session projection."
                )
            for group in self.session_breakdown.groups:
                require_bounded_usage_session_id(group.session_id)
            if len(self.session_breakdown.groups) > session_group_limit:
                raise UsageRollupInconsistent(
                    "Usage rollup session groups exceed the requested limit."
                )
            if (
                self.session_breakdown.accuracy.kind is AggregateAccuracyKind.TRUNCATED
                and self.session_breakdown.accuracy.limit != session_group_limit
            ):
                raise UsageRollupInconsistent(
                    "Usage rollup session truncation does not match the requested limit."
                )

        if self.pricing_inputs_included is not query.include_pricing_inputs:
            raise UsageRollupInconsistent(
                "Usage rollup pricing projection does not match the query."
            )
        if len(self.pricing_inputs) > query.pricing_input_limit:
            raise UsageRollupInconsistent("Usage rollup pricing inputs exceed the requested limit.")
        session_pricing_requested = query.include_pricing_inputs and session_projection_requested
        if self.session_pricing_inputs_included is not session_pricing_requested:
            raise UsageRollupInconsistent(
                "Usage rollup session pricing projection does not match the query."
            )
        if len(self.session_pricing_inputs) > query.pricing_input_limit:
            raise UsageRollupInconsistent(
                "Usage rollup session pricing inputs exceed the requested limit."
            )
        for item in self.session_pricing_inputs:
            if item.session_id is None:
                raise UsageRollupInconsistent(
                    "Usage rollup session pricing inputs require session identity."
                )
            require_bounded_usage_session_id(item.session_id)

    def _validate_exact_session_pricing_attribution(self) -> None:
        """Reconcile exact retained pricing inputs with visible session usage."""

        if (
            not self.session_pricing_inputs_included
            or self.session_pricing_inputs_accuracy.kind is not AggregateAccuracyKind.EXACT
            or self.session_breakdown is None
        ):
            return

        occurrences_by_session: dict[str, int] = {}
        valid_usage_steps_by_session: dict[str, int] = {}
        known_usage_by_session: dict[str, AggregateUsageMetrics] = {}
        for item in self.session_pricing_inputs:
            assert item.session_id is not None
            session_id = item.session_id
            occurrences_by_session[session_id] = (
                occurrences_by_session.get(session_id, 0) + item.occurrences
            )
            if item.metrics is None:
                continue
            valid_usage_steps_by_session[session_id] = (
                valid_usage_steps_by_session.get(session_id, 0) + item.occurrences
            )
            known_usage_by_session[session_id] = add_aggregate_usage(
                known_usage_by_session.get(
                    session_id,
                    build_aggregate_usage_metrics(),
                ),
                _scaled_pricing_usage(item.metrics, item.occurrences),
            )

        for group in self.session_breakdown.groups:
            session_id = group.session_id
            if occurrences_by_session.get(session_id, 0) != group.totals.model_steps:
                raise ValueError(
                    "Exact session pricing inputs must account for each retained session's "
                    "model steps."
                )
            valid_usage_steps = valid_usage_steps_by_session.get(session_id, 0)
            known_usage = known_usage_by_session.get(
                session_id,
                build_aggregate_usage_metrics(),
            )
            if valid_usage_steps > group.totals.model_steps_with_usage or any(
                known > reported
                for known, reported in zip(
                    _aggregate_usage_counter_values(known_usage),
                    _aggregate_usage_counter_values(group.totals.usage),
                    strict=True,
                )
            ):
                raise ValueError(
                    "Exact session pricing inputs exceed the retained session's usage totals."
                )
            if (
                valid_usage_steps == group.totals.model_steps_with_usage
                and known_usage != group.totals.usage
            ):
                raise ValueError(
                    "Exact session pricing inputs do not reproduce the retained session's "
                    "usage totals."
                )

    def _validate_exact_session_pricing_projection(self) -> None:
        """Require exact session inputs to be a lossless partition of shared inputs."""

        if (
            not self.session_pricing_inputs_included
            or self.pricing_inputs_accuracy.kind is not AggregateAccuracyKind.EXACT
            or self.session_pricing_inputs_accuracy.kind is not AggregateAccuracyKind.EXACT
        ):
            return
        if not self.pricing_inputs_included or self.session_breakdown is None:
            raise UsageRollupInconsistent(
                "Exact session pricing inputs require the shared usage projection."
            )
        try:
            remainder_groups = _session_pricing_remainder_groups(
                shared=self.pricing_inputs,
                retained=self.session_pricing_inputs,
            )
        except ValueError as exc:
            raise UsageRollupInconsistent(
                "Exact session pricing inputs do not match the shared usage projection."
            ) from exc
        expected_remainder_steps = (
            0
            if self.session_breakdown.remainder is None
            else self.session_breakdown.remainder.totals.model_steps
        )
        if sum(item[0] for item in remainder_groups.values()) != expected_remainder_steps:
            raise UsageRollupInconsistent(
                "Exact session pricing inputs do not account for the shared usage projection."
            )


class UsageCurrencyCost(BaseModel):
    """Exact estimated cost for one currency; currencies are never combined."""

    model_config = ConfigDict(extra="forbid")

    currency: str
    model_steps: PositiveAggregateCount = Field(ge=1)
    total_cost: Decimal = Field(ge=0)

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, value: str) -> str:
        return require_clean_nonblank(value, "currency").upper()


class UsageUnpricedReason(BaseModel):
    """Model-step count that could not be priced for one explicit reason."""

    model_config = ConfigDict(extra="forbid")

    reason: str
    model_steps: PositiveAggregateCount = Field(ge=1)

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        return require_clean_nonblank(value, "reason")


class UsageBillingIdentity(BaseModel):
    """Bounded commercial identity safe to return in an aggregate response."""

    model_config = ConfigDict(extra="forbid")

    provider_name: str
    resource_id: str
    request_evidence: dict[str, str] = Field(default_factory=dict, max_length=4)
    completion_evidence: dict[str, str] = Field(default_factory=dict, max_length=1)

    @field_validator("provider_name", "resource_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("request_evidence", "completion_evidence")
    @classmethod
    def copy_evidence(cls, value: dict[str, str], info) -> dict[str, str]:
        copied: dict[str, str] = {}
        for key, item in value.items():
            clean_key = require_clean_nonblank(key, f"{info.field_name} key")
            copied[clean_key] = require_clean_nonblank(item, f"{info.field_name}.{clean_key}")
        return copied

    @model_validator(mode="after")
    def validate_display_contract(self) -> UsageBillingIdentity:
        request_fields = set(self.request_evidence)
        completion_fields = set(self.completion_evidence)
        if self.provider_name == "bedrock":
            if not request_fields.issubset(_BEDROCK_AGGREGATE_REQUEST_EVIDENCE) or not (
                completion_fields.issubset(_BEDROCK_AGGREGATE_COMPLETION_EVIDENCE)
            ):
                raise ValueError("Bedrock aggregate identity contains unsupported display fields.")
        elif request_fields or completion_fields:
            raise ValueError("Provider evidence is not part of the generic aggregate contract.")
        return self


class UsageBillingCostGroup(BaseModel):
    """One bounded, identity-bearing group of equivalent pricing outcomes."""

    model_config = ConfigDict(extra="forbid")

    billing_identity: UsageBillingIdentity
    pricing_provider_name: str | None = None
    pricing_model: str | None = None
    priced: StrictBool
    model_steps: PositiveAggregateCount = Field(ge=1)
    currency: str | None = None
    total_cost: Decimal = Field(ge=0)
    missing_pricing_reason: str | None = None

    @field_validator(
        "pricing_provider_name",
        "pricing_model",
        "currency",
        "missing_pricing_reason",
    )
    @classmethod
    def validate_optional_text(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        value = require_clean_nonblank(value, info.field_name)
        return value.upper() if info.field_name == "currency" else value

    @model_validator(mode="after")
    def validate_outcome(self) -> UsageBillingCostGroup:
        if self.priced:
            if (
                self.currency is None
                or self.pricing_provider_name is None
                or self.pricing_model is None
                or self.missing_pricing_reason is not None
            ):
                raise ValueError(
                    "Priced billing groups require currency, pricing identity, and no missing "
                    "reason."
                )
        elif (
            self.currency is not None
            or self.total_cost != 0
            or self.pricing_provider_name is not None
            or self.pricing_model is not None
            or self.missing_pricing_reason is None
        ):
            raise ValueError(
                "Unpriced billing groups require a missing reason, no pricing identity, and zero "
                "currency-local cost."
            )
        return self


class UsageBillingCostRemainder(BaseModel):
    """Exact counts for identity-bearing pricing groups omitted from bounded detail."""

    model_config = ConfigDict(extra="forbid")

    group_count: PositiveAggregateCount = Field(ge=1)
    model_steps: PositiveAggregateCount = Field(ge=1)
    priced_model_steps: AggregateCount = Field(ge=0)
    unpriced_model_steps: AggregateCount = Field(ge=0)

    @model_validator(mode="after")
    def validate_step_accounting(self) -> UsageBillingCostRemainder:
        if self.priced_model_steps + self.unpriced_model_steps != self.model_steps:
            raise ValueError("Remainder priced and unpriced steps must sum to model_steps.")
        return self


class UsageBillingCostBreakdown(BaseModel):
    """Bounded identity detail for evaluated cost inputs, with an exact remainder."""

    model_config = ConfigDict(extra="forbid")

    identified_model_steps: AggregateCount = Field(ge=0)
    groups: tuple[UsageBillingCostGroup, ...] = Field(max_length=100)
    remainder: UsageBillingCostRemainder | None
    accuracy: AggregateAccuracy

    @model_validator(mode="after")
    def validate_scope(self) -> UsageBillingCostBreakdown:
        represented_steps = sum(group.model_steps for group in self.groups)
        if self.remainder is not None:
            represented_steps += self.remainder.model_steps
            if self.accuracy.kind is not AggregateAccuracyKind.TRUNCATED:
                raise ValueError("A bounded billing remainder must report truncation.")
        if represented_steps != self.identified_model_steps:
            raise ValueError(
                "Billing groups and their remainder must sum to identified_model_steps."
            )
        if self.accuracy.kind is AggregateAccuracyKind.EXACT and self.remainder is not None:
            raise ValueError("An exact billing breakdown cannot include a remainder.")
        return self


class UsageCostRollup(BaseModel):
    """Bounded cost result derived from grouped price-relevant inputs."""

    model_config = ConfigDict(extra="forbid")

    price_book_version: str
    price_book_generated_at: str
    accuracy: AggregateAccuracy
    evaluated_model_steps: AggregateCount = Field(ge=0)
    priced_model_steps: AggregateCount = Field(ge=0)
    unpriced_model_steps: AggregateCount = Field(ge=0)
    unevaluated_model_steps: AggregateCount = Field(ge=0)
    currencies: tuple[UsageCurrencyCost, ...] = Field(max_length=5000)
    unpriced_reasons: tuple[UsageUnpricedReason, ...] = Field(max_length=5000)
    billing_breakdown: UsageBillingCostBreakdown

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
        if self.billing_breakdown.identified_model_steps > self.evaluated_model_steps:
            raise ValueError("Billing-identified model steps cannot exceed evaluated_model_steps.")
        if (
            self.accuracy.kind is AggregateAccuracyKind.SAMPLED
            and self.billing_breakdown.accuracy.kind is AggregateAccuracyKind.EXACT
        ):
            raise ValueError("Sampled cost rollups cannot contain an exact billing breakdown.")
        if (
            self.accuracy.kind is AggregateAccuracyKind.TRUNCATED
            and self.billing_breakdown.accuracy.kind is not AggregateAccuracyKind.TRUNCATED
        ):
            raise ValueError("Truncated cost rollups require a truncated billing breakdown.")
        return self


class UsageSessionCostSummary(BaseModel):
    """Cost accounting for one session group or the exact omitted remainder."""

    model_config = ConfigDict(extra="forbid")

    accuracy: AggregateAccuracy
    evaluated_model_steps: AggregateCount = Field(ge=0)
    priced_model_steps: AggregateCount = Field(ge=0)
    unpriced_model_steps: AggregateCount = Field(ge=0)
    unevaluated_model_steps: AggregateCount = Field(ge=0)
    currencies: tuple[UsageCurrencyCost, ...] = Field(max_length=5000)
    unpriced_reasons: tuple[UsageUnpricedReason, ...] = Field(max_length=5000)

    @model_validator(mode="after")
    def validate_step_accounting(self) -> UsageSessionCostSummary:
        if self.priced_model_steps + self.unpriced_model_steps != self.evaluated_model_steps:
            raise ValueError("Priced and unpriced steps must sum to evaluated_model_steps.")
        if sum(item.model_steps for item in self.currencies) != self.priced_model_steps:
            raise ValueError("Currency model-step counts must sum to priced_model_steps.")
        if sum(item.model_steps for item in self.unpriced_reasons) != self.unpriced_model_steps:
            raise ValueError("Unpriced reasons must sum to unpriced_model_steps.")
        if self.accuracy.kind is AggregateAccuracyKind.EXACT and self.unevaluated_model_steps:
            raise ValueError("Exact session costs cannot contain unevaluated model steps.")
        if self.accuracy.kind is AggregateAccuracyKind.TRUNCATED and self.evaluated_model_steps:
            raise ValueError("Truncated session costs cannot report partial evaluated totals.")
        return self


class UsageSessionCostGroup(BaseModel):
    """Cost accounting for one retained session."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    cost: UsageSessionCostSummary

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        return require_clean_nonblank(value, "session_id")


class UsageSessionCostRemainder(BaseModel):
    """Exact aggregate cost for sessions omitted from bounded detail."""

    model_config = ConfigDict(extra="forbid")

    group_count: PositiveAggregateCount = Field(ge=1)
    cost: UsageSessionCostSummary


class UsageSessionCostBreakdown(BaseModel):
    """Bounded per-session costs plus an aggregate remainder."""

    model_config = ConfigDict(extra="forbid")

    price_book_version: str
    price_book_generated_at: str
    groups: tuple[UsageSessionCostGroup, ...] = Field(max_length=100)
    remainder: UsageSessionCostRemainder | None
    accuracy: AggregateAccuracy

    @field_validator("price_book_version", "price_book_generated_at")
    @classmethod
    def validate_price_book_identity(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_breakdown(self) -> UsageSessionCostBreakdown:
        session_ids = [group.session_id for group in self.groups]
        if len(set(session_ids)) != len(session_ids):
            raise ValueError("Session cost groups must have distinct session IDs.")
        if self.remainder is not None and self.accuracy.kind is not AggregateAccuracyKind.TRUNCATED:
            raise ValueError("A bounded session cost remainder must report truncation.")
        return self


@dataclass
class _UsageBillingCostAccumulator:
    billing_identity: UsageBillingIdentity
    pricing_provider_name: str | None
    pricing_model: str | None
    priced: bool
    currency: str | None
    missing_pricing_reason: str | None
    model_steps: int = 0
    total_cost: Decimal = Decimal(0)


def estimate_usage_rollup_cost(
    result: UsageRollupStoreResult,
    pricing,
    *,
    billing_group_limit: int = 20,
) -> UsageCostRollup:
    """Price an exact bounded input projection without reporting partial totals."""

    from cayu.runtime.costs import PriceBook

    if type(result) is not UsageRollupStoreResult:
        raise TypeError("result must be a UsageRollupStoreResult.")
    if type(pricing) is not PriceBook:
        raise TypeError("pricing must be a PriceBook.")
    if type(billing_group_limit) is not int or not 1 <= billing_group_limit <= 100:
        raise ValueError("billing_group_limit must be between 1 and 100.")
    if not result.pricing_inputs_included:
        raise ValueError("Usage rollup did not request pricing inputs.")
    return _estimate_usage_cost(
        totals=result.totals,
        totals_accuracy=result.totals_accuracy,
        pricing_inputs=result.pricing_inputs,
        pricing_inputs_accuracy=result.pricing_inputs_accuracy,
        pricing=pricing,
        billing_group_limit=billing_group_limit,
    )


def _estimate_usage_cost(
    *,
    totals: UsageAggregateTotals,
    totals_accuracy: AggregateAccuracy,
    pricing_inputs: tuple[UsagePricingInput, ...],
    pricing_inputs_accuracy: AggregateAccuracy,
    pricing,
    billing_group_limit: int,
    estimate_cache: dict[tuple[date, str | None], object] | None = None,
) -> UsageCostRollup:
    from cayu.runtime.costs import ModelStepCostEstimate, estimate_model_step_cost

    cost_accuracy = _combined_aggregate_accuracy(
        totals_accuracy,
        pricing_inputs_accuracy,
    )
    if (
        pricing_inputs_accuracy.kind is not AggregateAccuracyKind.EXACT
        or totals_accuracy.kind is AggregateAccuracyKind.TRUNCATED
    ):
        return UsageCostRollup(
            price_book_version=pricing.price_book_version,
            price_book_generated_at=pricing.generated_at,
            accuracy=cost_accuracy,
            evaluated_model_steps=0,
            priced_model_steps=0,
            unpriced_model_steps=0,
            unevaluated_model_steps=totals.model_steps,
            currencies=(),
            unpriced_reasons=(),
            billing_breakdown=UsageBillingCostBreakdown(
                identified_model_steps=0,
                groups=(),
                remainder=None,
                accuracy=cost_accuracy,
            ),
        )

    currency_costs: dict[str, Decimal] = {}
    currency_steps: dict[str, int] = {}
    unpriced_reasons: dict[str, int] = {}
    billing_groups: dict[str, _UsageBillingCostAccumulator] = {}
    priced_model_steps = 0
    unpriced_model_steps = 0
    for item in pricing_inputs:
        if item.metrics is None:
            reason = "model.completed event has no valid normalized usage metrics"
            unpriced_reasons[reason] = unpriced_reasons.get(reason, 0) + item.occurrences
            unpriced_model_steps += item.occurrences
            continue
        estimate_key = (item.effective_on, usage_pricing_metrics_key(item.metrics))
        estimate = None if estimate_cache is None else estimate_cache.get(estimate_key)
        if estimate is None:
            estimate = estimate_model_step_cost(
                metrics=item.metrics,
                pricing=pricing,
                effective_on=item.effective_on,
            )
            if estimate_cache is not None:
                estimate_cache[estimate_key] = estimate
        else:
            estimate = cast("ModelStepCostEstimate", estimate)
        pricing_identity = item.metrics.billing_identity
        if pricing_identity is not None:
            billing_identity = _billing_identity_for_breakdown(pricing_identity)
            group_key = json.dumps(
                {
                    "billing_identity": billing_identity.model_dump(mode="json"),
                    "pricing_provider_name": estimate.pricing_provider_name,
                    "pricing_model": estimate.pricing_model,
                    "priced": estimate.priced,
                    "currency": estimate.currency,
                    "missing_pricing_reason": estimate.missing_pricing_reason,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            group = billing_groups.get(group_key)
            if group is None:
                group = _UsageBillingCostAccumulator(
                    billing_identity=billing_identity.model_copy(deep=True),
                    pricing_provider_name=estimate.pricing_provider_name,
                    pricing_model=estimate.pricing_model,
                    priced=estimate.priced,
                    currency=estimate.currency,
                    missing_pricing_reason=estimate.missing_pricing_reason,
                )
                billing_groups[group_key] = group
            group.model_steps += item.occurrences
            group.total_cost += estimate.total_cost * item.occurrences
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

    ordered_billing_groups = sorted(
        (
            UsageBillingCostGroup(
                billing_identity=group.billing_identity,
                pricing_provider_name=group.pricing_provider_name,
                pricing_model=group.pricing_model,
                priced=group.priced,
                model_steps=group.model_steps,
                currency=group.currency,
                total_cost=group.total_cost,
                missing_pricing_reason=group.missing_pricing_reason,
            )
            for group in billing_groups.values()
        ),
        key=lambda group: (
            -group.model_steps,
            group.billing_identity.provider_name,
            group.billing_identity.resource_id,
            json.dumps(
                group.billing_identity.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "" if group.pricing_provider_name is None else group.pricing_provider_name,
            "" if group.pricing_model is None else group.pricing_model,
            "" if group.currency is None else group.currency,
            "" if group.missing_pricing_reason is None else group.missing_pricing_reason,
        ),
    )
    retained_billing_groups = tuple(ordered_billing_groups[:billing_group_limit])
    omitted_billing_groups = ordered_billing_groups[billing_group_limit:]
    billing_remainder = (
        None
        if not omitted_billing_groups
        else UsageBillingCostRemainder(
            group_count=len(omitted_billing_groups),
            model_steps=sum(group.model_steps for group in omitted_billing_groups),
            priced_model_steps=sum(
                group.model_steps for group in omitted_billing_groups if group.priced
            ),
            unpriced_model_steps=sum(
                group.model_steps for group in omitted_billing_groups if not group.priced
            ),
        )
    )
    billing_accuracy = cost_accuracy
    if billing_remainder is not None:
        billing_accuracy = _combined_aggregate_accuracy(
            cost_accuracy,
            AggregateAccuracy(
                kind=AggregateAccuracyKind.TRUNCATED,
                reason="Billing identity groups exceed group_limit.",
                limit=billing_group_limit,
            ),
        )

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
        billing_breakdown=UsageBillingCostBreakdown(
            identified_model_steps=sum(group.model_steps for group in ordered_billing_groups),
            groups=retained_billing_groups,
            remainder=billing_remainder,
            accuracy=billing_accuracy,
        ),
    )


def estimate_usage_session_cost_breakdown(
    result: UsageRollupStoreResult,
    pricing,
) -> UsageSessionCostBreakdown:
    """Price retained session groups without publishing partial session costs."""

    from cayu.runtime.costs import PriceBook

    if type(result) is not UsageRollupStoreResult:
        raise TypeError("result must be a UsageRollupStoreResult.")
    if type(pricing) is not PriceBook:
        raise TypeError("pricing must be a PriceBook.")
    if result.session_breakdown is None:
        raise ValueError("Usage rollup did not request a session breakdown.")
    if not result.pricing_inputs_included:
        raise ValueError("Usage rollup did not request shared pricing inputs.")
    if not result.session_pricing_inputs_included:
        raise ValueError("Usage rollup did not request session pricing inputs.")

    evaluation_accuracy = _combined_aggregate_accuracy(
        _combined_aggregate_accuracy(
            result.totals_accuracy,
            result.pricing_inputs_accuracy,
        ),
        result.session_pricing_inputs_accuracy,
    )
    breakdown_accuracy = _combined_aggregate_accuracy(
        result.session_breakdown.accuracy,
        evaluation_accuracy,
    )
    if evaluation_accuracy.kind is not AggregateAccuracyKind.EXACT:
        groups = tuple(
            UsageSessionCostGroup(
                session_id=group.session_id,
                cost=_unevaluated_session_cost(
                    model_steps=group.totals.model_steps,
                    accuracy=evaluation_accuracy,
                ),
            )
            for group in result.session_breakdown.groups
        )
        remainder = (
            None
            if result.session_breakdown.remainder is None
            else UsageSessionCostRemainder(
                group_count=result.session_breakdown.remainder.group_count,
                cost=_unevaluated_session_cost(
                    model_steps=result.session_breakdown.remainder.totals.model_steps,
                    accuracy=evaluation_accuracy,
                ),
            )
        )
        return UsageSessionCostBreakdown(
            price_book_version=pricing.price_book_version,
            price_book_generated_at=pricing.generated_at,
            groups=groups,
            remainder=remainder,
            accuracy=breakdown_accuracy,
        )

    estimate_cache: dict[tuple[date, str | None], object] = {}
    inputs_by_session: dict[str, list[UsagePricingInput]] = {}
    for item in result.session_pricing_inputs:
        assert item.session_id is not None
        inputs_by_session.setdefault(item.session_id, []).append(item)
    groups = tuple(
        UsageSessionCostGroup(
            session_id=group.session_id,
            cost=_session_cost_summary(
                _estimate_usage_cost(
                    totals=group.totals,
                    totals_accuracy=EXACT_AGGREGATE,
                    pricing_inputs=tuple(inputs_by_session.get(group.session_id, ())),
                    pricing_inputs_accuracy=EXACT_AGGREGATE,
                    pricing=pricing,
                    billing_group_limit=1,
                    estimate_cache=estimate_cache,
                )
            ),
        )
        for group in result.session_breakdown.groups
    )
    remainder = None
    if result.session_breakdown.remainder is not None:
        remainder_inputs = _session_pricing_input_remainder(
            shared=result.pricing_inputs,
            retained=result.session_pricing_inputs,
        )
        if (
            sum(item.occurrences for item in remainder_inputs)
            != result.session_breakdown.remainder.totals.model_steps
        ):
            raise ValueError("Session pricing remainder does not match omitted usage totals.")
        remainder_cost = _session_cost_summary(
            _estimate_usage_cost(
                totals=result.session_breakdown.remainder.totals,
                totals_accuracy=EXACT_AGGREGATE,
                pricing_inputs=remainder_inputs,
                pricing_inputs_accuracy=EXACT_AGGREGATE,
                pricing=pricing,
                billing_group_limit=1,
                estimate_cache=estimate_cache,
            )
        )
        remainder = UsageSessionCostRemainder(
            group_count=result.session_breakdown.remainder.group_count,
            cost=remainder_cost,
        )
    return UsageSessionCostBreakdown(
        price_book_version=pricing.price_book_version,
        price_book_generated_at=pricing.generated_at,
        groups=groups,
        remainder=remainder,
        accuracy=breakdown_accuracy,
    )


def _session_cost_summary(cost: UsageCostRollup) -> UsageSessionCostSummary:
    return UsageSessionCostSummary(
        accuracy=cost.accuracy.model_copy(deep=True),
        evaluated_model_steps=cost.evaluated_model_steps,
        priced_model_steps=cost.priced_model_steps,
        unpriced_model_steps=cost.unpriced_model_steps,
        unevaluated_model_steps=cost.unevaluated_model_steps,
        currencies=tuple(item.model_copy(deep=True) for item in cost.currencies),
        unpriced_reasons=tuple(item.model_copy(deep=True) for item in cost.unpriced_reasons),
    )


def _unevaluated_session_cost(
    *,
    model_steps: int,
    accuracy: AggregateAccuracy,
) -> UsageSessionCostSummary:
    return UsageSessionCostSummary(
        accuracy=accuracy.model_copy(deep=True),
        evaluated_model_steps=0,
        priced_model_steps=0,
        unpriced_model_steps=0,
        unevaluated_model_steps=model_steps,
        currencies=(),
        unpriced_reasons=(),
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
