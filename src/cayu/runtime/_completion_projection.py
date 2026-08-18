from __future__ import annotations

from typing import Any, cast

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    DurableValueError,
    copy_durable_json_object,
    copy_durable_json_value,
    require_durable_nonblank,
    require_durable_text,
)
from cayu.runtime.execution_units import RUNTIME_OWNED_EXECUTION_IDENTITY_FIELDS
from cayu.runtime.usage import normalize_usage_metrics_with_overflow_error

_RUNTIME_OWNED_USAGE_FIELDS = frozenset(
    {
        "usage_metrics",
        "usage_normalization_failed",
        "usage_unavailable_reason",
        "usage_metrics_rejected",
        "rejected_usage_evidence",
    }
)


def portable_model_completion_projection(
    payload: dict[str, Any],
    *,
    provider_name: str,
    requested_model: str,
    usage_dialect: str | None,
) -> dict[str, Any]:
    """Project portable completion facts while retaining recognized usage.

    Provider completion payloads are untrusted and may mix valid billing
    counters with malformed auxiliary metadata.  Projection is intentionally
    top-level and input-value-independent: rejected keys are not echoed, the
    provider-owned ``usage_metrics`` namespace is discarded, and malformed raw
    ``usage`` is reduced to runtime-owned normalized counters when possible.
    """

    if type(payload) is not dict:
        raise TypeError("Model completion payload must be an object.")
    provider_name = _durable_nonblank(provider_name, "provider_name")
    requested_model = _durable_nonblank(requested_model, "requested_model")
    source = _exact_string_key_object(payload)
    if source is None:  # Guard the type contract above for static analyzers.
        raise TypeError("Model completion payload must be an object.")
    resolved_model = source.get("model")

    projected: dict[str, Any] = {}
    for key, value in source.items():
        if key in _RUNTIME_OWNED_USAGE_FIELDS or key in RUNTIME_OWNED_EXECUTION_IDENTITY_FIELDS:
            # These namespaces are runtime-owned; providers cannot inject
            # durable identity, normalized accounting evidence, or its failure
            # classification.
            continue
        if key == "usage":
            try:
                projected["usage"] = copy_durable_json_object(value, "completed usage")
            except DurableValueError:
                overflow_evidence = _portable_overflow_usage_projection(
                    value,
                    provider_name=provider_name,
                    requested_model=requested_model,
                    resolved_model=resolved_model,
                    usage_dialect=usage_dialect,
                )
                if overflow_evidence is not None:
                    projected["usage"] = overflow_evidence
                else:
                    # The original provider evidence cannot cross the durable
                    # boundary. Do not price a lossy reconstruction even when its
                    # recognized counters happen to normalize successfully.
                    projected["usage_normalization_failed"] = True
            continue
        try:
            portable_key = require_durable_text(key, "completed key")
            portable_value = copy_durable_json_value(value, "completed value")
        except DurableValueError:
            continue
        projected[portable_key] = portable_value
    return projected


def _portable_overflow_usage_projection(
    raw_usage: object,
    *,
    provider_name: str,
    requested_model: str,
    resolved_model: object,
    usage_dialect: str | None,
) -> dict[str, Any] | None:
    """Retain bounded counters only when they prove a derived int64 overflow."""

    portable_usage = _portable_usage_counter_projection(raw_usage)
    if not portable_usage:
        return None
    try:
        model = _durable_nonblank(resolved_model, "model")
    except ValueError:
        model = requested_model
    try:
        normalize_usage_metrics_with_overflow_error(
            provider_name=provider_name,
            model=model,
            requested_model=requested_model,
            raw_usage=portable_usage,
            usage_dialect=usage_dialect,
        )
    except (TypeError, ValueError):
        # The fixed-vocabulary projection contains only independently valid
        # counters. A raised validation failure therefore comes from a derived
        # UsageMetrics/cache aggregate crossing the signed-int64 boundary.
        return portable_usage
    return None


def _portable_usage_counter_projection(raw_usage: object) -> dict[str, Any]:
    """Project recognized portable counters from a malformed usage object.

    The result has a fixed vocabulary. Provider-owned auxiliary keys and text
    never cross the boundary, while independently valid counters remain
    available as terminal evidence when a derived aggregate exceeds int64.
    Cache-detail arrays are folded into at most two entries per TTL category,
    so malformed metadata cannot amplify the durable record size.
    """

    usage = _exact_string_key_object(raw_usage)
    if usage is None:
        return {}

    projected: dict[str, Any] = {}
    for key in (
        "input_tokens",
        "prompt_tokens",
        "output_tokens",
        "completion_tokens",
        "total_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
    ):
        counter = _portable_usage_counter(usage.get(key))
        if counter is not None:
            projected[key] = counter

    for container_key, counter_keys in (
        ("input_tokens_details", ("cached_tokens",)),
        ("prompt_tokens_details", ("cached_tokens",)),
        ("output_tokens_details", ("reasoning_tokens", "thinking_tokens")),
        ("completion_tokens_details", ("reasoning_tokens", "thinking_tokens")),
    ):
        container = usage.get(container_key)
        container = _exact_string_key_object(container)
        if container is None:
            continue
        counters = {
            key: counter
            for key in counter_keys
            if (counter := _portable_usage_counter(container.get(key))) is not None
        }
        if counters:
            projected[container_key] = counters

    cache_details = usage.get("cache_details")
    if type(cache_details) is list:
        totals = {"5m": 0, "1h": 0, "unknown": 0}
        for detail in cache_details:
            detail = _exact_string_key_object(detail)
            if detail is None:
                continue
            counter = _portable_usage_counter(detail.get("input_tokens"))
            if counter is None:
                continue
            ttl = detail.get("ttl")
            category = ttl if type(ttl) is str and ttl in {"5m", "1h"} else "unknown"
            totals[category] += counter
        folded_details: list[dict[str, Any]] = []
        for category in ("5m", "1h", "unknown"):
            for counter in _bounded_counter_sum_evidence(totals[category]):
                folded_detail: dict[str, Any] = {"input_tokens": counter}
                if category != "unknown":
                    folded_detail["ttl"] = category
                folded_details.append(folded_detail)
        if folded_details:
            projected["cache_details"] = folded_details

    cache_creation = usage.get("cache_creation")
    cache_creation = _exact_string_key_object(cache_creation)
    if cache_creation is not None:
        projected_creation: dict[str, int] = {}
        known_keys = {
            "ephemeral_5m_input_tokens",
            "ephemeral_1h_input_tokens",
        }
        for key in known_keys:
            counter = _portable_usage_counter(cache_creation.get(key))
            if counter is not None:
                projected_creation[key] = counter
        unknown_total = sum(
            counter
            for key, value in cache_creation.items()
            if key not in known_keys and (counter := _portable_usage_counter(value)) is not None
        )
        for index, counter in enumerate(_bounded_counter_sum_evidence(unknown_total)):
            projected_creation[f"other_input_tokens_{index}"] = counter
        if projected_creation:
            projected["cache_creation"] = projected_creation

    return projected


def _exact_string_key_object(value: object) -> dict[str, object] | None:
    """Copy an untrusted exact dict without invoking provider-owned key methods."""

    if type(value) is not dict:
        return None
    copied: dict[str, object] = {}
    for key, item in value.items():
        if type(key) is str:
            copied[key] = item
    return copied


def _portable_usage_counter(value: object) -> int | None:
    if type(value) is int and 0 <= value <= MAX_DURABLE_JSON_INTEGER:
        return value
    return None


def _bounded_counter_sum_evidence(total: int) -> tuple[int, ...]:
    if total <= 0:
        return ()
    if total <= MAX_DURABLE_JSON_INTEGER:
        return (total,)
    # The exact aggregate is not portable. MAX+1 is a truthful lower bound
    # proving overflow while keeping the projection fixed-size and int64-safe.
    return (MAX_DURABLE_JSON_INTEGER, 1)


def _durable_nonblank(value: object, field_name: str) -> str:
    return require_durable_nonblank(cast("str", value), field_name)
