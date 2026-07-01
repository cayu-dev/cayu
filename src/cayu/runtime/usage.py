from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from cayu._validation import copy_json_value, require_clean_nonblank
from cayu.core.events import Event, EventType


class CacheUsageMetrics(BaseModel):
    """Provider-neutral cache token counters for one model step."""

    model_config = ConfigDict(extra="forbid")

    read_tokens: StrictInt = Field(default=0, ge=0)
    write_tokens: StrictInt = Field(default=0, ge=0)
    cached_input_tokens: StrictInt = Field(default=0, ge=0)
    uncached_input_tokens: StrictInt = Field(default=0, ge=0)


class UsageMetrics(BaseModel):
    """Provider-neutral token counters for one model step."""

    model_config = ConfigDict(extra="forbid")

    provider_name: str | None = None
    model: str | None = None
    input_tokens: StrictInt = Field(default=0, ge=0)
    output_tokens: StrictInt = Field(default=0, ge=0)
    total_tokens: StrictInt = Field(default=0, ge=0)
    reasoning_output_tokens: StrictInt = Field(default=0, ge=0)
    cache: CacheUsageMetrics = Field(default_factory=CacheUsageMetrics)

    @field_validator("provider_name", "model")
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


# Providers whose raw usage payload follows the Anthropic shape (cache tokens in
# separate fields, excluded from input_tokens). Claude on Vertex AI is one of them.
_ANTHROPIC_SHAPED_PROVIDERS = frozenset({"anthropic", "vertex"})


def normalize_usage_metrics(
    *,
    provider_name: str | None,
    model: str | None,
    raw_usage: Any,
) -> UsageMetrics | None:
    """Normalize provider usage payloads without hiding the original raw usage."""

    if type(raw_usage) is not dict:
        return None
    if not _has_usage_counter(raw_usage):
        return None

    input_tokens = _first_nonnegative_int(raw_usage, ("input_tokens", "prompt_tokens"))
    output_tokens = _first_nonnegative_int(
        raw_usage,
        ("output_tokens", "completion_tokens"),
    )
    total_tokens = _nonnegative_int(raw_usage.get("total_tokens"))
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens

    cached_input_tokens = 0
    reasoning_output_tokens = 0
    input_details = raw_usage.get("input_tokens_details")
    if type(input_details) is not dict:
        input_details = raw_usage.get("prompt_tokens_details")
    if type(input_details) is dict:
        cached_input_tokens = _nonnegative_int(input_details.get("cached_tokens"))

    output_details = raw_usage.get("output_tokens_details")
    if type(output_details) is not dict:
        output_details = raw_usage.get("completion_tokens_details")
    if type(output_details) is dict:
        reasoning_output_tokens = _nonnegative_int(output_details.get("reasoning_tokens"))
        if reasoning_output_tokens == 0:
            # Anthropic reports extended-thinking tokens as `thinking_tokens` (already
            # billed inside output_tokens); surface them in the same neutral field.
            reasoning_output_tokens = _nonnegative_int(output_details.get("thinking_tokens"))

    cache_read_tokens = _nonnegative_int(raw_usage.get("cache_read_input_tokens"))
    cache_write_tokens = _nonnegative_int(raw_usage.get("cache_creation_input_tokens"))
    cache_creation = raw_usage.get("cache_creation")
    if type(cache_creation) is dict:
        cache_creation_tokens = sum(_nonnegative_int(value) for value in cache_creation.values())
        if cache_creation_tokens > 0:
            cache_write_tokens = cache_creation_tokens

    provider = provider_name.strip().lower() if type(provider_name) is str else None
    anthropic_shaped = provider in _ANTHROPIC_SHAPED_PROVIDERS
    if provider == "openai":
        cache_read_tokens = max(cache_read_tokens, cached_input_tokens)
    elif anthropic_shaped:
        cached_input_tokens = max(cached_input_tokens, cache_read_tokens)
        input_tokens = input_tokens + cache_read_tokens + cache_write_tokens

    if total_tokens == 0 or anthropic_shaped:
        total_tokens = input_tokens + output_tokens

    uncached_input_tokens = max(input_tokens - cached_input_tokens, 0)
    if anthropic_shaped:
        uncached_input_tokens = _first_nonnegative_int(raw_usage, ("input_tokens",))

    return UsageMetrics(
        provider_name=provider_name,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        reasoning_output_tokens=reasoning_output_tokens,
        cache=CacheUsageMetrics(
            read_tokens=cache_read_tokens,
            write_tokens=cache_write_tokens,
            cached_input_tokens=cached_input_tokens,
            uncached_input_tokens=uncached_input_tokens,
        ),
    )


def usage_metrics_payload(metrics: UsageMetrics | None) -> dict[str, Any] | None:
    if metrics is None:
        return None
    return metrics.model_dump()


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
        metrics = usage_metrics_from_event_payload(event.payload)
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
    metrics = payload.get("usage_metrics")
    if type(metrics) is dict:
        return UsageMetrics(**copy_json_value(metrics, "usage_metrics"))

    return normalize_usage_metrics(
        provider_name=_optional_string(payload.get("provider_name")),
        model=_optional_string(payload.get("model")),
        raw_usage=payload.get("usage"),
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


def _nonnegative_int(value: Any) -> int:
    if type(value) is bool:
        return 0
    if type(value) is int and value >= 0:
        return value
    return 0


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
