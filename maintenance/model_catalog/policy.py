"""Review policy for Cayu's committed default model catalog.

This module is repository maintenance code. Runtime catalog loading never imports it and
never performs network access.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from urllib.parse import urlparse

from cayu import ModelCatalog, ModelInfo

CATALOG_MAX_AGE_DAYS = 30
VERIFY_MAX_AGE_DAYS = 14
VERIFY_SCHEDULE_DAYS = 7
if VERIFY_MAX_AGE_DAYS + VERIFY_SCHEDULE_DAYS >= CATALOG_MAX_AGE_DAYS:
    raise RuntimeError("catalog verification policy leaves no freshness review margin")
MAX_CATALOG_PRICE_PER_MILLION_USD = Decimal("1000")
MAX_AUTOMATED_PRICE_CHANGE_FACTOR = Decimal("4")
MAX_CATALOG_CONTEXT_WINDOW_TOKENS = 10_000_000
MAX_AUTOMATED_CONTEXT_WINDOW_CHANGE_FACTOR = Decimal("4")
REQUIRED_PROVIDERS = frozenset({"openai", "anthropic", "google", "vertex", "azure"})
OFFICIAL_HOSTS = {
    "anthropic": frozenset({"docs.anthropic.com", "platform.claude.com"}),
    "azure": frozenset({"azure.microsoft.com", "learn.microsoft.com", "prices.azure.com"}),
    "google": frozenset({"ai.google.dev"}),
    "openai": frozenset({"developers.openai.com", "openai.com", "platform.openai.com"}),
    "vertex": frozenset({"cloud.google.com", "docs.cloud.google.com"}),
}
_MATCH_PREFIX_DELIMITERS = frozenset("-_/:@")


def _is_host_allowed(hostname: str, allowed_hosts: frozenset[str]) -> bool:
    return hostname in allowed_hosts


def _price_points(model: ModelInfo) -> dict[str, Decimal]:
    points: dict[str, Decimal] = {}
    for index, tier in enumerate(model.pricing.standard):
        prefix = f"standard[{index}]"
        points[f"{prefix}.input"] = tier.input_per_million
        points[f"{prefix}.output"] = tier.output_per_million
        if tier.cache_read_input_per_million is not None:
            points[f"{prefix}.cache_read"] = tier.cache_read_input_per_million
        if tier.cache_write_input_per_million is not None:
            points[f"{prefix}.cache_write"] = tier.cache_write_input_per_million
    if model.pricing.batch is not None:
        points["batch.input"] = model.pricing.batch.input_per_million
        points["batch.output"] = model.pricing.batch.output_per_million
        if model.pricing.batch.cache_read_input_per_million is not None:
            points["batch.cache_read"] = model.pricing.batch.cache_read_input_per_million
    if model.pricing.cache_write_5m_per_million is not None:
        points["cache_write_5m"] = model.pricing.cache_write_5m_per_million
    if model.pricing.cache_write_1h_per_million is not None:
        points["cache_write_1h"] = model.pricing.cache_write_1h_per_million
    return points


def model_policy_errors(
    model: ModelInfo,
    *,
    today: date,
    max_age_days: int | None = CATALOG_MAX_AGE_DAYS,
    allow_deprecated: bool = False,
) -> list[str]:
    """Structural, source, and price-sanity errors for one bundled record."""

    identity = f"{model.provider_name}/{model.model}"
    errors: list[str] = []
    if model.deprecated and not allow_deprecated:
        errors.append(f"{identity}: deprecated models are not bundled by default")
    if not model.tool_calling:
        errors.append(f"{identity}: default models must support tool calling")
    if model.pricing.currency != "USD":
        errors.append(f"{identity}: only USD pricing is supported by the default snapshot")
    if model.provenance.source != "official":
        errors.append(f"{identity}: provenance source must be 'official'")
    if model.match != "exact":
        errors.append(f"{identity}: bundled canonical model match must be 'exact'")
    for prefix in model.match_prefixes:
        suffix = prefix[len(model.model) :] if prefix.startswith(model.model) else ""
        if len(suffix) < 2 or suffix[0] not in _MATCH_PREFIX_DELIMITERS:
            errors.append(
                f"{identity}: match prefix {prefix!r} must strictly extend the canonical model "
                "with an approved delimiter"
            )
    if model.context_window is not None:
        if type(model.context_window) is not int or model.context_window <= 0:
            errors.append(f"{identity}: context_window must be a positive integer")
        elif model.context_window > MAX_CATALOG_CONTEXT_WINDOW_TOKENS:
            errors.append(
                f"{identity}: context_window exceeds {MAX_CATALOG_CONTEXT_WINDOW_TOKENS} tokens"
            )

    parsed = urlparse(model.provenance.url)
    hostname = (parsed.hostname or "").lower()
    allowed_hosts = OFFICIAL_HOSTS.get(model.provider_name, frozenset())
    if parsed.scheme != "https" or not _is_host_allowed(hostname, allowed_hosts):
        errors.append(f"{identity}: provenance URL is not on an allowed official host")

    try:
        verified_at = date.fromisoformat(model.provenance.as_of)
    except ValueError:
        errors.append(f"{identity}: provenance as_of is not an ISO date")
    else:
        age = (today - verified_at).days
        if age < 0:
            errors.append(f"{identity}: provenance as_of is in the future")
        elif max_age_days is not None and age > max_age_days:
            errors.append(f"{identity}: provenance is stale ({age}d > {max_age_days}d)")

    for field, value in _price_points(model).items():
        if value <= 0:
            errors.append(f"{identity}: {field} price must be greater than zero")
        elif value > MAX_CATALOG_PRICE_PER_MILLION_USD:
            errors.append(
                f"{identity}: {field} price exceeds "
                f"{MAX_CATALOG_PRICE_PER_MILLION_USD} USD per million"
            )
    return errors


def suspicious_price_changes(
    before: ModelInfo,
    after: ModelInfo,
    *,
    max_factor: Decimal = MAX_AUTOMATED_PRICE_CHANGE_FACTOR,
) -> list[str]:
    """Changes too risky for the automated verifier to place in a candidate snapshot."""

    warnings: list[str] = []
    before_bounds = tuple(tier.max_input_tokens for tier in before.pricing.standard)
    after_bounds = tuple(tier.max_input_tokens for tier in after.pricing.standard)
    if before_bounds != after_bounds:
        warnings.append(f"context tier boundaries changed: {before_bounds} -> {after_bounds}")

    before_context = before.context_window
    after_context = after.context_window
    if before_context is not None and after_context is None:
        warnings.append(f"context_window removed: {before_context} -> None")
    elif (
        type(before_context) is int
        and type(after_context) is int
        and before_context > 0
        and after_context > 0
        and before_context != after_context
    ):
        low, high = sorted((Decimal(before_context), Decimal(after_context)))
        if high / low > MAX_AUTOMATED_CONTEXT_WINDOW_CHANGE_FACTOR:
            warnings.append(f"context_window changed from {before_context} to {after_context}")

    old = _price_points(before)
    new = _price_points(after)
    added = sorted(new.keys() - old.keys())
    removed = sorted(old.keys() - new.keys())
    if added:
        warnings.append(f"price dimensions added: {', '.join(added)}")
    if removed:
        warnings.append(f"price dimensions removed: {', '.join(removed)}")
    for field in sorted(old.keys() & new.keys()):
        low, high = sorted((old[field], new[field]))
        if low == 0 or high / low > max_factor:
            warnings.append(f"{field} changed from {old[field]} to {new[field]}")
    return warnings


def validate_catalog(
    catalog: ModelCatalog,
    *,
    today: date | None = None,
    max_age_days: int | None = CATALOG_MAX_AGE_DAYS,
    require_provider_coverage: bool = True,
) -> None:
    """Raise ``ValueError`` when a candidate is unsafe to bundle.

    ``max_age_days=None`` is useful while preparing a review PR: structural and source
    checks still run, while stale records can remain visible in the review report. Release
    CI uses the default and therefore cannot publish a stale committed snapshot.
    """

    if type(catalog) is not ModelCatalog:
        raise TypeError("catalog must be a ModelCatalog instance.")
    reference = today or datetime.now(UTC).date()
    errors: list[str] = []

    if catalog.catalog_version != catalog.generated_at:
        errors.append("catalog_version must equal the dated generated_at snapshot version")

    try:
        generated_at = date.fromisoformat(catalog.generated_at)
    except ValueError:
        errors.append(f"catalog generated_at is not an ISO date: {catalog.generated_at!r}")
    else:
        if generated_at > reference:
            errors.append(f"catalog generated_at is in the future: {generated_at}")

    providers = {model.provider_name for model in catalog.models}
    identities = [(model.provider_name, model.model) for model in catalog.models]
    if identities != sorted(identities):
        errors.append("models must be sorted by provider_name and model")
    missing = sorted(REQUIRED_PROVIDERS - providers)
    extra = sorted(providers - REQUIRED_PROVIDERS)
    if require_provider_coverage and missing:
        errors.append(f"missing required providers: {', '.join(missing)}")
    if extra:
        errors.append(f"providers outside the advertised default set: {', '.join(extra)}")

    for model in catalog.models:
        errors.extend(model_policy_errors(model, today=reference, max_age_days=max_age_days))

    if errors:
        raise ValueError("Model catalog policy failed:\n- " + "\n- ".join(errors))
