from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from cayu._validation import (
    copy_json_value,
    require_clean_nonblank,
    require_durable_json_text,
)
from cayu.core.billing import BillingIdentity
from cayu.core.events import Event, EventType


class ModelCompletionPurpose(StrEnum):
    """Runtime-owned purpose markers for auxiliary model completions."""

    CONTEXT_COMPACTION = "context_compaction"


def is_conversational_model_completion_payload(payload: dict[str, Any]) -> bool:
    """Whether one completion may anchor the next conversational context.

    A valid transcript cursor is the sole runtime-owned evidence that the
    completion belongs to an ordinary model step, even when provider metadata
    supplied a colliding ``purpose`` key. Cursorless completions are ineligible
    regardless of whether they carry an explicit purpose.
    """

    transcript_cursor = payload.get("transcript_cursor")
    return type(transcript_cursor) is int and transcript_cursor >= 0


class CacheUsageMetrics(BaseModel):
    """Provider-neutral cache token counters for one model step."""

    model_config = ConfigDict(extra="forbid")

    read_tokens: StrictInt = Field(default=0, ge=0)
    write_tokens: StrictInt = Field(default=0, ge=0)
    write_5m_tokens: StrictInt = Field(default=0, ge=0, exclude_if=lambda value: value == 0)
    write_1h_tokens: StrictInt = Field(default=0, ge=0, exclude_if=lambda value: value == 0)
    write_unknown_ttl_tokens: StrictInt = Field(
        default=0, ge=0, exclude_if=lambda value: value == 0
    )
    cached_input_tokens: StrictInt = Field(default=0, ge=0)
    uncached_input_tokens: StrictInt = Field(default=0, ge=0)


class UsageMetrics(BaseModel):
    """Provider-neutral token counters for one model step."""

    model_config = ConfigDict(extra="forbid")

    provider_name: str | None = None
    requested_model: str | None = None
    model: str | None = None
    billing_identity: BillingIdentity | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    input_tokens: StrictInt = Field(default=0, ge=0)
    output_tokens: StrictInt = Field(default=0, ge=0)
    total_tokens: StrictInt = Field(default=0, ge=0)
    reasoning_output_tokens: StrictInt = Field(default=0, ge=0)
    cache: CacheUsageMetrics = Field(default_factory=CacheUsageMetrics)

    @field_validator("provider_name", "requested_model", "model")
    @classmethod
    def validate_optional_nonblank(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)


class SessionUsageSummary(BaseModel):
    """Usage totals derived from durable session events."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    model_steps: StrictInt = Field(default=0, ge=0)
    tool_calls: StrictInt = Field(default=0, ge=0)
    provider_names: list[str] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)
    usage: UsageMetrics = Field(default_factory=UsageMetrics)

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("provider_names", "models", mode="before")
    @classmethod
    def copy_string_lists(cls, value: list[str], info) -> list[str]:
        copied = copy_json_value(value, info.field_name)
        if type(copied) is not list:
            raise ValueError(f"{info.field_name} must be a list.")
        result: list[str] = []
        for index, item in enumerate(copied):
            if type(item) is not str:
                raise ValueError(f"{info.field_name}[{index}] must be a string.")
            result.append(require_clean_nonblank(item, f"{info.field_name}[{index}]"))
        return result


class CausalBudgetUsageSummary(BaseModel):
    """Usage totals for all sessions sharing one causal budget id."""

    model_config = ConfigDict(extra="forbid")

    causal_budget_id: str
    session_ids: list[str] = Field(default_factory=list)
    session_count: StrictInt = Field(default=0, ge=0)
    model_steps: StrictInt = Field(default=0, ge=0)
    tool_calls: StrictInt = Field(default=0, ge=0)
    provider_names: list[str] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)
    usage: UsageMetrics = Field(default_factory=UsageMetrics)
    session_summaries: tuple[SessionUsageSummary, ...] = Field(default_factory=tuple)

    @field_validator("causal_budget_id")
    @classmethod
    def validate_causal_budget_id(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("session_ids", "provider_names", "models", mode="before")
    @classmethod
    def copy_string_lists(cls, value: list[str], info) -> list[str]:
        copied = copy_json_value(value, info.field_name)
        if type(copied) is not list:
            raise ValueError(f"{info.field_name} must be a list.")
        result: list[str] = []
        for index, item in enumerate(copied):
            if type(item) is not str:
                raise ValueError(f"{info.field_name}[{index}] must be a string.")
            result.append(require_clean_nonblank(item, f"{info.field_name}[{index}]"))
        return result


# Canonical usage dialects (string values mirror ``providers.base.UsageDialect``
# so a declared enum can be passed straight through as ``str``).
_DIALECT_ANTHROPIC = "anthropic"
_DIALECT_OPENAI = "openai"
_DIALECT_GENERIC = "generic"
_KNOWN_DIALECTS = frozenset({_DIALECT_ANTHROPIC, _DIALECT_OPENAI, _DIALECT_GENERIC})

# Registered names whose raw usage payload follows the Anthropic shape (cache
# tokens in separate fields, excluded from input_tokens). Claude on Vertex AI is
# one of them. Retained as a fallback for the built-in aliases; unknown names
# (Bedrock, gateways, renamed adapters) are resolved from a declared dialect or
# the payload shape so their cache tokens are not silently undercounted.
_ANTHROPIC_SHAPED_PROVIDERS = frozenset({"anthropic", "vertex"})


def normalize_usage_metrics(
    *,
    provider_name: str | None,
    model: str | None,
    raw_usage: Any,
    usage_dialect: str | None = None,
    requested_model: str | None = None,
    billing_identity: BillingIdentity | None = None,
) -> UsageMetrics | None:
    """Normalize provider usage payloads without hiding the original raw usage.

    ``usage_dialect`` (a ``providers.base.UsageDialect`` value or its string) lets
    a provider declare how it encodes cache tokens. When omitted or ``"auto"``,
    the dialect is inferred from the provider name and then the payload shape, so
    Anthropic-shaped payloads (Claude via Bedrock/gateways/renamed adapters) fold
    cache read/write tokens back into ``input_tokens`` instead of undercounting.
    """

    if type(raw_usage) is not dict:
        return None

    provider = provider_name.strip().lower() if type(provider_name) is str else None
    dialect = _resolve_usage_dialect(
        provider=provider,
        declared_dialect=usage_dialect,
        raw_usage=raw_usage,
    )
    authoritative_dialect = _authoritative_usage_dialect(
        provider=provider,
        declared_dialect=usage_dialect,
    )

    strict_input_tokens = _equivalent_nonnegative_int_fields(
        raw_usage,
        ("input_tokens", "prompt_tokens"),
    )
    strict_output_tokens = _equivalent_nonnegative_int_fields(
        raw_usage,
        ("output_tokens", "completion_tokens"),
    )
    strict_total_tokens = _optional_nonnegative_int_field(raw_usage, "total_tokens")
    strict_cache_read_tokens = _optional_nonnegative_int_field(
        raw_usage,
        "cache_read_input_tokens",
    )
    strict_cache_write_tokens = _strict_cache_write_tokens(raw_usage)
    if (
        strict_input_tokens is None
        or strict_output_tokens is None
        or strict_total_tokens is None
        or strict_cache_read_tokens is None
        or strict_cache_write_tokens is None
    ):
        return None

    input_tokens, has_input_tokens = strict_input_tokens
    output_tokens, has_output_tokens = strict_output_tokens
    reported_total_tokens, has_reported_total_tokens = strict_total_tokens
    cache_read_tokens, has_explicit_cache_read = strict_cache_read_tokens
    cache_write_tokens = strict_cache_write_tokens
    if dialect == _DIALECT_OPENAI and (not has_input_tokens or not has_output_tokens):
        return None

    if not _has_usage_counter(raw_usage) and dialect != _DIALECT_OPENAI:
        return None
    nested_cached_input = _nested_cached_input_tokens(raw_usage)
    has_nested_cached_container = _has_nested_cached_input_container(raw_usage)
    if nested_cached_input is None and (
        dialect in {_DIALECT_OPENAI, _DIALECT_GENERIC}
        or (dialect == _DIALECT_ANTHROPIC and has_nested_cached_container)
    ):
        return None
    cached_input_tokens, has_nested_cached_input = (
        nested_cached_input if nested_cached_input is not None else (0, False)
    )
    mixed_cache_evidence = has_nested_cached_input and _has_top_level_anthropic_cache_evidence(
        raw_usage
    )
    if mixed_cache_evidence and authoritative_dialect not in {
        _DIALECT_OPENAI,
        _DIALECT_ANTHROPIC,
    }:
        return None

    computed_primary_total = input_tokens + output_tokens
    if has_reported_total_tokens:
        accepted_reported_totals = {computed_primary_total}
        if dialect == _DIALECT_ANTHROPIC:
            # Bedrock surfaces both conventions across supported model payloads:
            # totalTokens may be the primary input-plus-output sum, or may already
            # include the separately reported cache read/write counters.
            accepted_reported_totals.add(
                computed_primary_total + cache_read_tokens + cache_write_tokens
            )
        if reported_total_tokens not in accepted_reported_totals:
            return None
    total_tokens = reported_total_tokens if has_reported_total_tokens else computed_primary_total

    if dialect in {_DIALECT_OPENAI, _DIALECT_GENERIC} and cached_input_tokens > input_tokens:
        return None
    strict_reasoning_output = (
        _nested_reasoning_output_tokens(raw_usage) if dialect == _DIALECT_OPENAI else None
    )
    if dialect == _DIALECT_OPENAI and strict_reasoning_output is None:
        return None
    reasoning_output_tokens = 0
    input_details = raw_usage.get("input_tokens_details")
    if type(input_details) is not dict:
        input_details = raw_usage.get("prompt_tokens_details")
    if dialect == _DIALECT_ANTHROPIC and type(input_details) is dict:
        cached_input_tokens = _nonnegative_int(input_details.get("cached_tokens"))

    if strict_reasoning_output is not None:
        reasoning_output_tokens, _ = strict_reasoning_output
        if reasoning_output_tokens > output_tokens:
            return None
    else:
        output_details = raw_usage.get("output_tokens_details")
        if type(output_details) is not dict:
            output_details = raw_usage.get("completion_tokens_details")
        if type(output_details) is dict:
            reasoning_output_tokens = _nonnegative_int(output_details.get("reasoning_tokens"))
            if reasoning_output_tokens == 0:
                # Anthropic reports extended-thinking tokens as `thinking_tokens` (already
                # billed inside output_tokens); surface them in the same neutral field.
                reasoning_output_tokens = _nonnegative_int(output_details.get("thinking_tokens"))

    cache_write_5m_tokens = 0
    cache_write_1h_tokens = 0
    cache_write_unknown_ttl_tokens = 0
    cache_details = raw_usage.get("cache_details")
    has_cache_details = type(cache_details) is list
    if has_cache_details:
        for detail in cache_details:
            if type(detail) is not dict:
                continue
            tokens = _nonnegative_int(detail.get("input_tokens"))
            ttl = detail.get("ttl")
            if ttl == "5m":
                cache_write_5m_tokens += tokens
            elif ttl == "1h":
                cache_write_1h_tokens += tokens
            else:
                cache_write_unknown_ttl_tokens += tokens
    cache_creation = raw_usage.get("cache_creation")
    if type(cache_creation) is dict:
        cache_creation_tokens = sum(_nonnegative_int(value) for value in cache_creation.values())
        if cache_creation_tokens > 0:
            cache_write_tokens = max(cache_write_tokens, cache_creation_tokens)
        if not has_cache_details:
            cache_write_5m_tokens += _nonnegative_int(
                cache_creation.get("ephemeral_5m_input_tokens")
            )
            cache_write_1h_tokens += _nonnegative_int(
                cache_creation.get("ephemeral_1h_input_tokens")
            )
            cache_write_unknown_ttl_tokens += sum(
                _nonnegative_int(value)
                for key, value in cache_creation.items()
                if key
                not in {
                    "ephemeral_5m_input_tokens",
                    "ephemeral_1h_input_tokens",
                }
            )

    detailed_cache_writes = (
        cache_write_5m_tokens + cache_write_1h_tokens + cache_write_unknown_ttl_tokens
    )
    if detailed_cache_writes > cache_write_tokens:
        cache_write_tokens = detailed_cache_writes
    elif cache_write_tokens > detailed_cache_writes:
        cache_write_unknown_ttl_tokens += cache_write_tokens - detailed_cache_writes

    anthropic_shaped = dialect == _DIALECT_ANTHROPIC
    if dialect == _DIALECT_OPENAI:
        if cache_write_tokens > 0:
            return None
        if has_explicit_cache_read and cache_read_tokens != cached_input_tokens:
            return None
        cache_read_tokens = cached_input_tokens
    elif anthropic_shaped:
        if has_nested_cached_input and (
            not has_explicit_cache_read or cache_read_tokens != cached_input_tokens
        ):
            return None
        cached_input_tokens = max(cached_input_tokens, cache_read_tokens)
        input_tokens = input_tokens + cache_read_tokens + cache_write_tokens
    else:
        if cache_read_tokens > 0 or cache_write_tokens > 0:
            return None
        # An undeclared provider may expose OpenAI-shaped cache details, but the
        # runtime has no authoritative dialect proving how to bill them. Keep all
        # reported input in the ordinary bucket rather than subtracting a cached
        # portion whose cache-read price would otherwise be missing.
        cached_input_tokens = 0

    if anthropic_shaped:
        computed_total_tokens = input_tokens + output_tokens
        total_tokens = computed_total_tokens

    uncached_input_tokens = max(input_tokens - cached_input_tokens, 0)
    if anthropic_shaped:
        uncached_input_tokens = _first_nonnegative_int(raw_usage, ("input_tokens",))

    return UsageMetrics(
        provider_name=provider_name,
        requested_model=requested_model,
        model=model,
        billing_identity=billing_identity,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        reasoning_output_tokens=reasoning_output_tokens,
        cache=CacheUsageMetrics(
            read_tokens=cache_read_tokens,
            write_tokens=cache_write_tokens,
            write_5m_tokens=cache_write_5m_tokens,
            write_1h_tokens=cache_write_1h_tokens,
            write_unknown_ttl_tokens=cache_write_unknown_ttl_tokens,
            cached_input_tokens=cached_input_tokens,
            uncached_input_tokens=uncached_input_tokens,
        ),
    )


def usage_metrics_payload(metrics: UsageMetrics | None) -> dict[str, Any] | None:
    if metrics is None:
        return None
    return metrics.model_dump()


def durable_model_completed_payload(
    payload: dict[str, Any],
    *,
    fallback_fields: dict[str, Any],
    unavailable_reason: str,
) -> dict[str, Any]:
    """Project provider completion metadata onto the cross-store JSON boundary.

    Provider payloads normally remain equivalent JSON. When one top-level field
    contains text that a supported durable store cannot represent, retain every
    independently durable non-derived field plus the caller's runtime-owned
    identity fallback. Usage fails closed whenever raw or normalized evidence
    crosses a lossy persistence boundary.
    """

    copied = copy_json_value(payload, "model_completed_payload")
    try:
        require_durable_json_text(copied, "model_completed_payload")
    except ValueError:
        projected: dict[str, Any] = {}
        for key, value in copied.items():
            candidate = {key: value}
            try:
                require_durable_json_text(candidate, f"model_completed_payload.{key}")
            except ValueError:
                continue
            projected[key] = value

        runtime_fields = copy_json_value(fallback_fields, "fallback_fields")
        for key, value in runtime_fields.items():
            candidate = {key: value}
            try:
                require_durable_json_text(candidate, f"fallback_fields.{key}")
            except ValueError:
                continue
            projected.setdefault(key, value)

        lost_raw_usage = "usage" in copied and "usage" not in projected
        lost_billing_identity = "billing_identity" in copied and "billing_identity" not in projected
        if lost_raw_usage or lost_billing_identity:
            # Normalized counters cannot remain authoritative after their raw
            # provider evidence or commercial identity crosses a lossy
            # persistence boundary. Dropping the derived representation keeps
            # every cost and aggregate reader fail-closed instead of letting
            # optimized readers price a projection whose billing basis changed.
            projected.pop("usage_metrics", None)
        lost_normalized_usage = "usage_metrics" in copied and "usage_metrics" not in projected
        lost_all_usage = (
            ("usage" in copied or "usage_metrics" in copied)
            and "usage" not in projected
            and "usage_metrics" not in projected
        )
        if (
            copied.get("usage_normalization_failed") is True
            or lost_raw_usage
            or lost_billing_identity
            or lost_normalized_usage
            or lost_all_usage
        ):
            projected["usage_normalization_failed"] = True
            projected["usage_unavailable_reason"] = require_clean_nonblank(
                unavailable_reason,
                "unavailable_reason",
            )
        require_durable_json_text(projected, "model_completed_payload")
        return projected
    return copied


def strip_provider_billing_identity(payload: dict[str, Any]) -> None:
    """Remove billing identity fields that only the runtime may populate."""

    payload.pop("billing_identity", None)
    usage_metrics = payload.get("usage_metrics")
    if type(usage_metrics) is dict:
        usage_metrics.pop("billing_identity", None)


# The event types usage accounting reads and folds: `tool.call.started`
# drives the tool-call counter, `model.completed` carries token usage and
# cost inputs. The runtime's usage readers (the session usage tracker and the
# session/causal-budget summaries) query exactly this tuple; a new
# usage-bearing type must be added here AND handled in
# `session_usage_summary` below.
USAGE_BEARING_EVENT_TYPES: tuple[EventType, ...] = (
    EventType.MODEL_COMPLETED,
    EventType.TOOL_CALL_STARTED,
)


def project_usage_inspection_event(event: Event) -> Event:
    """Retain only the bounded fields consumed by session usage inspection."""
    payload: dict[str, Any] = {}
    if event.type == EventType.MODEL_COMPLETED:
        try:
            metrics = usage_metrics_from_event_payload(event.payload)
        except (TypeError, ValueError):
            metrics = None
        if metrics is not None:
            payload["usage_metrics"] = metrics.model_dump(mode="json")
    return event.model_copy(update={"payload": payload})


def session_usage_summary(session_id: str, events: list[Event]) -> SessionUsageSummary:
    provider_names: list[str] = []
    models: list[str] = []
    usage = UsageMetrics()
    model_steps = 0
    tool_calls = 0

    for event in events:
        if event.type == EventType.TOOL_CALL_STARTED:
            tool_calls += 1
            continue
        if event.type != EventType.MODEL_COMPLETED:
            continue
        model_steps += 1
        try:
            metrics = usage_metrics_from_event_payload(event.payload)
        except (TypeError, ValueError):
            metrics = None
        if metrics is None:
            continue
        usage = _add_usage(usage, metrics)
        if metrics.provider_name is not None and metrics.provider_name not in provider_names:
            provider_names.append(metrics.provider_name)
        if metrics.model is not None and metrics.model not in models:
            models.append(metrics.model)

    return SessionUsageSummary(
        session_id=session_id,
        model_steps=model_steps,
        tool_calls=tool_calls,
        provider_names=provider_names,
        models=models,
        usage=usage,
    )


def count_model_steps_with_usage(events: Iterable[Event]) -> int:
    """Count durable model completions carrying valid normalized usage."""
    count = 0
    for event in events:
        if event.type != EventType.MODEL_COMPLETED:
            continue
        try:
            metrics = usage_metrics_from_event_payload(event.payload)
        except (TypeError, ValueError):
            continue
        if metrics is not None:
            count += 1
    return count


def causal_budget_usage_summary(
    *,
    causal_budget_id: str,
    session_ids: list[str],
    events: list[Event],
) -> CausalBudgetUsageSummary:
    causal_budget_id = require_clean_nonblank(causal_budget_id, "causal_budget_id")
    session_ids = _copy_string_list(session_ids, "session_ids")
    known_session_ids = set(session_ids)
    filtered_events = [event for event in events if event.session_id in known_session_ids]
    summary = session_usage_summary(causal_budget_id, filtered_events)
    return CausalBudgetUsageSummary(
        causal_budget_id=causal_budget_id,
        session_ids=session_ids,
        session_count=len(session_ids),
        model_steps=summary.model_steps,
        tool_calls=summary.tool_calls,
        provider_names=summary.provider_names,
        models=summary.models,
        usage=summary.usage,
        session_summaries=tuple(
            session_usage_summary(
                session_id,
                [event for event in filtered_events if event.session_id == session_id],
            )
            for session_id in session_ids
        ),
    )


def usage_metrics_from_event_payload(payload: dict[str, Any]) -> UsageMetrics | None:
    raw_identity = payload.get("billing_identity")
    event_identity = (
        BillingIdentity.model_validate(raw_identity) if type(raw_identity) is dict else None
    )
    # New runtime events record a failed normalization decision explicitly so
    # durable readers do not reinterpret the same raw payload later without the
    # provider's authoritative dialect. Events written before this marker was
    # introduced retain the legacy raw-usage fallback below.
    if payload.get("usage_normalization_failed") is True:
        return None
    metrics = payload.get("usage_metrics")
    if type(metrics) is dict:
        parsed = UsageMetrics(**copy_json_value(metrics, "usage_metrics"))
        if event_identity is None:
            return parsed
        if parsed.billing_identity is not None and parsed.billing_identity != event_identity:
            raise ValueError("model.completed billing identities do not match.")
        return parsed.model_copy(update={"billing_identity": event_identity}, deep=True)

    return normalize_usage_metrics(
        provider_name=_optional_string(payload.get("provider_name")),
        model=_optional_string(payload.get("model")),
        requested_model=_optional_string(payload.get("requested_model")),
        raw_usage=payload.get("usage"),
        billing_identity=event_identity,
    )


def _add_usage(left: UsageMetrics, right: UsageMetrics) -> UsageMetrics:
    return UsageMetrics(
        input_tokens=left.input_tokens + right.input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        total_tokens=left.total_tokens + right.total_tokens,
        reasoning_output_tokens=left.reasoning_output_tokens + right.reasoning_output_tokens,
        cache=CacheUsageMetrics(
            read_tokens=left.cache.read_tokens + right.cache.read_tokens,
            write_tokens=left.cache.write_tokens + right.cache.write_tokens,
            write_5m_tokens=left.cache.write_5m_tokens + right.cache.write_5m_tokens,
            write_1h_tokens=left.cache.write_1h_tokens + right.cache.write_1h_tokens,
            write_unknown_ttl_tokens=(
                left.cache.write_unknown_ttl_tokens + right.cache.write_unknown_ttl_tokens
            ),
            cached_input_tokens=left.cache.cached_input_tokens + right.cache.cached_input_tokens,
            uncached_input_tokens=left.cache.uncached_input_tokens
            + right.cache.uncached_input_tokens,
        ),
    )


def _copy_string_list(value: list[str], field_name: str) -> list[str]:
    copied = copy_json_value(value, field_name)
    if type(copied) is not list:
        raise ValueError(f"{field_name} must be a list.")
    result: list[str] = []
    for index, item in enumerate(copied):
        if type(item) is not str:
            raise ValueError(f"{field_name}[{index}] must be a string.")
        result.append(require_clean_nonblank(item, f"{field_name}[{index}]"))
    return result


def _resolve_usage_dialect(
    *,
    provider: str | None,
    declared_dialect: str | None,
    raw_usage: dict[str, Any],
) -> str:
    """Pick the usage dialect from an explicit declaration, name, then shape.

    An explicit non-``auto`` declaration wins. Otherwise the built-in name
    aliases are honored for backward compatibility, and finally the payload
    shape is inspected so Anthropic-shaped usage from an unknown provider name
    (Bedrock, gateways, renamed adapters) is still folded correctly.
    """

    declared = declared_dialect.strip().lower() if isinstance(declared_dialect, str) else None
    if declared in _KNOWN_DIALECTS:
        return declared
    if provider in _ANTHROPIC_SHAPED_PROVIDERS:
        return _DIALECT_ANTHROPIC
    if provider == _DIALECT_OPENAI:
        return _DIALECT_OPENAI
    if _is_anthropic_shaped_payload(raw_usage):
        return _DIALECT_ANTHROPIC
    # OpenAI-style nested ``*_tokens_details.cached_tokens`` is left as generic
    # unless a provider declares the ``openai`` dialect. Without authoritative
    # semantics, the generic path keeps every reported input token in the
    # ordinary billable bucket instead of treating an unclassified portion as free.
    return _DIALECT_GENERIC


def _authoritative_usage_dialect(
    *,
    provider: str | None,
    declared_dialect: str | None,
) -> str | None:
    """Return a declared or built-in dialect, excluding payload-shape inference."""

    declared = declared_dialect.strip().lower() if isinstance(declared_dialect, str) else None
    if declared in _KNOWN_DIALECTS:
        return declared
    if provider in _ANTHROPIC_SHAPED_PROVIDERS:
        return _DIALECT_ANTHROPIC
    if provider == _DIALECT_OPENAI:
        return _DIALECT_OPENAI
    return None


def _is_anthropic_shaped_payload(raw_usage: dict[str, Any]) -> bool:
    """True when the payload carries Anthropic-only cache-token fields.

    These top-level counters (separate cache read/write fields excluded from
    ``input_tokens``) are unique to the Anthropic Messages usage shape, so their
    presence classifies Claude reached through an unrecognized provider name
    (Bedrock, gateways, renamed adapters) without a name allowlist.
    """

    if "cache_read_input_tokens" in raw_usage or "cache_creation_input_tokens" in raw_usage:
        return True
    return type(raw_usage.get("cache_creation")) is dict


def _has_top_level_anthropic_cache_evidence(raw_usage: dict[str, Any]) -> bool:
    """Return whether the payload carries any non-null Anthropic cache field."""

    return any(
        key in raw_usage and raw_usage[key] is not None
        for key in (
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
            "cache_creation",
            "cache_details",
        )
    )


def _nonnegative_int(value: Any) -> int:
    if type(value) is bool:
        return 0
    if type(value) is int and value >= 0:
        return value
    return 0


def _optional_nonnegative_int_field(values: dict[str, Any], key: str) -> tuple[int, bool] | None:
    """Return a present non-negative integer field, or ``None`` when malformed."""

    if key not in values:
        return 0, False
    value = values[key]
    if type(value) is not int or value < 0:
        return None
    return value, True


def _equivalent_nonnegative_int_fields(
    values: dict[str, Any],
    keys: tuple[str, ...],
) -> tuple[int, bool] | None:
    """Read equivalent counters once and reject malformed or conflicting aliases."""

    observed: list[int] = []
    for key in keys:
        parsed = _optional_nonnegative_int_field(values, key)
        if parsed is None:
            return None
        value, present = parsed
        if present:
            observed.append(value)
    if not observed:
        return 0, False
    if any(value != observed[0] for value in observed[1:]):
        return None
    return observed[0], True


def _nested_cached_input_tokens(raw_usage: dict[str, Any]) -> tuple[int, bool] | None:
    """Read equivalent OpenAI cached-token details once and reject contradictions."""

    observed: list[int] = []
    for key in ("input_tokens_details", "prompt_tokens_details"):
        if key not in raw_usage:
            continue
        details = raw_usage[key]
        if details is None:
            continue
        if type(details) is not dict:
            return None
        if "cached_tokens" not in details:
            continue
        value = details["cached_tokens"]
        if value is None:
            continue
        if type(value) is not int or value < 0:
            return None
        observed.append(value)
    if not observed:
        return 0, False
    if any(value != observed[0] for value in observed[1:]):
        return None
    return observed[0], True


def _has_nested_cached_input_container(raw_usage: dict[str, Any]) -> bool:
    """Return whether a non-null OpenAI input-details container was supplied."""

    for key in ("input_tokens_details", "prompt_tokens_details"):
        if key in raw_usage and raw_usage[key] is not None:
            return True
    return False


def _nested_reasoning_output_tokens(
    raw_usage: dict[str, Any],
) -> tuple[int, bool] | None:
    """Read equivalent optional reasoning counters without lossy coercion."""

    observed: list[int] = []
    for key in ("output_tokens_details", "completion_tokens_details"):
        if key not in raw_usage:
            continue
        details = raw_usage[key]
        if details is None:
            continue
        if type(details) is not dict:
            return None
        for detail_key in ("reasoning_tokens", "thinking_tokens"):
            if detail_key not in details or details[detail_key] is None:
                continue
            value = details[detail_key]
            if type(value) is not int or value < 0:
                return None
            observed.append(value)
    if not observed:
        return 0, False
    if any(value != observed[0] for value in observed[1:]):
        return None
    return observed[0], True


def _strict_cache_write_tokens(raw_usage: dict[str, Any]) -> int | None:
    """Validate cache-write evidence before an OpenAI-dialect payload can be priced."""

    counter = _optional_nonnegative_int_field(raw_usage, "cache_creation_input_tokens")
    if counter is None:
        return None
    total, _ = counter

    cache_creation = raw_usage.get("cache_creation")
    if "cache_creation" in raw_usage and cache_creation is not None:
        if type(cache_creation) is not dict:
            return None
        creation_total = 0
        for value in cache_creation.values():
            if type(value) is not int or value < 0:
                return None
            creation_total += value
        total = max(total, creation_total)

    cache_details = raw_usage.get("cache_details")
    if "cache_details" in raw_usage and cache_details is not None:
        if type(cache_details) is not list:
            return None
        details_total = 0
        for detail in cache_details:
            if type(detail) is not dict:
                return None
            tokens = _optional_nonnegative_int_field(detail, "input_tokens")
            if tokens is None:
                return None
            value, present = tokens
            if not present:
                return None
            details_total += value
        total = max(total, details_total)
    return total


def _has_usage_counter(values: dict[str, Any]) -> bool:
    keys = (
        "input_tokens",
        "prompt_tokens",
        "output_tokens",
        "completion_tokens",
        "total_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
    )
    if any(_nonnegative_int(values.get(key)) > 0 for key in keys):
        return True
    input_details = values.get("input_tokens_details")
    if type(input_details) is not dict:
        input_details = values.get("prompt_tokens_details")
    if type(input_details) is dict and _nonnegative_int(input_details.get("cached_tokens")) > 0:
        return True
    output_details = values.get("output_tokens_details")
    if type(output_details) is not dict:
        output_details = values.get("completion_tokens_details")
    if (
        type(output_details) is dict
        and _nonnegative_int(output_details.get("reasoning_tokens")) > 0
    ):
        return True
    cache_creation = values.get("cache_creation")
    return type(cache_creation) is dict and any(
        _nonnegative_int(value) > 0 for value in cache_creation.values()
    )


def _first_nonnegative_int(values: dict[str, Any], keys: tuple[str, ...]) -> int:
    for key in keys:
        value = _nonnegative_int(values.get(key))
        if value > 0:
            return value
    return 0


def _optional_string(value: Any) -> str | None:
    if type(value) is str and value.strip():
        return value
    return None
