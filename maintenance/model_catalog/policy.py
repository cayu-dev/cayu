"""Review policy for Cayu's committed default model catalog.

This module is repository maintenance code. Runtime catalog loading never imports it and
never performs network access.
"""

from __future__ import annotations

from collections.abc import Collection
from datetime import UTC, date, datetime
from decimal import Decimal
from urllib.parse import urlparse

from cayu import ModelCatalog, ModelInfo, ModelPrice, PriceBook

CATALOG_MAX_AGE_DAYS = 30
VERIFY_MAX_AGE_DAYS = 14
VERIFY_SCHEDULE_DAYS = 7
if VERIFY_MAX_AGE_DAYS + VERIFY_SCHEDULE_DAYS >= CATALOG_MAX_AGE_DAYS:
    raise RuntimeError("catalog verification policy leaves no freshness review margin")
MAX_BUNDLED_PRICE_PER_MILLION_USD = Decimal("1000")
MAX_AUTOMATED_PRICE_CHANGE_FACTOR = Decimal("4")
MAX_CATALOG_CONTEXT_WINDOW_TOKENS = 10_000_000
MAX_AUTOMATED_CONTEXT_WINDOW_CHANGE_FACTOR = Decimal("4")
REQUIRED_PROVIDERS = frozenset({"openai", "anthropic", "google", "vertex", "azure"})
# A bundled billing identity that is deliberately not a routable model must be named here
# and documented in docs/model-catalog.md. Application-owned books do not use this policy.
BUNDLED_PRICE_ONLY_IDENTITIES: frozenset[tuple[str, str]] = frozenset()
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


def _price_points(model: ModelPrice) -> dict[str, Decimal]:
    points: dict[str, Decimal] = {}
    for schedule_index, schedule in enumerate(model.schedules):
        pricing = schedule.pricing
        schedule_prefix = f"schedule[{schedule_index}]"
        for tier_index, tier in enumerate(pricing.standard):
            prefix = f"{schedule_prefix}.standard[{tier_index}]"
            points[f"{prefix}.input"] = tier.input_per_million
            points[f"{prefix}.output"] = tier.output_per_million
            if tier.cache_read_input_per_million is not None:
                points[f"{prefix}.cache_read"] = tier.cache_read_input_per_million
            if tier.cache_write_input_per_million is not None:
                points[f"{prefix}.cache_write"] = tier.cache_write_input_per_million
        if pricing.batch is not None:
            points[f"{schedule_prefix}.batch.input"] = pricing.batch.input_per_million
            points[f"{schedule_prefix}.batch.output"] = pricing.batch.output_per_million
            if pricing.batch.cache_read_input_per_million is not None:
                points[f"{schedule_prefix}.batch.cache_read"] = (
                    pricing.batch.cache_read_input_per_million
                )
        if pricing.cache_write_5m_per_million is not None:
            points[f"{schedule_prefix}.cache_write_5m"] = pricing.cache_write_5m_per_million
        if pricing.cache_write_1h_per_million is not None:
            points[f"{schedule_prefix}.cache_write_1h"] = pricing.cache_write_1h_per_million
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

    return errors


def price_policy_errors(
    model: ModelPrice,
    *,
    today: date,
    max_age_days: int | None = CATALOG_MAX_AGE_DAYS,
) -> list[str]:
    """Structural, source, validity, and rate-sanity errors for one bundled price."""

    identity = f"{model.provider_name}/{model.model}"
    errors: list[str] = []
    if model.match != "exact":
        errors.append(f"{identity}: bundled canonical price match must be 'exact'")
    for prefix in model.match_prefixes:
        suffix = prefix[len(model.model) :] if prefix.startswith(model.model) else ""
        if len(suffix) < 2 or suffix[0] not in _MATCH_PREFIX_DELIMITERS:
            errors.append(
                f"{identity}: price match prefix {prefix!r} must strictly extend the canonical "
                "model with an approved delimiter"
            )
    active_schedule = model.schedule_on(today)
    for index, schedule in enumerate(model.schedules):
        pricing = schedule.pricing
        schedule_id = f"{identity} schedule[{index}]"
        if pricing.currency != "USD":
            errors.append(f"{schedule_id}: only USD pricing is supported")
        if schedule.provenance.source != "official":
            errors.append(f"{schedule_id}: provenance source must be 'official'")
        parsed = urlparse(schedule.provenance.url)
        hostname = (parsed.hostname or "").lower()
        allowed_hosts = OFFICIAL_HOSTS.get(model.provider_name, frozenset())
        if parsed.scheme != "https" or not _is_host_allowed(hostname, allowed_hosts):
            errors.append(f"{schedule_id}: provenance URL is not on an allowed official host")
        try:
            verified_at = date.fromisoformat(schedule.provenance.as_of)
        except ValueError:
            errors.append(f"{schedule_id}: provenance as_of is not an ISO date")
        else:
            age = (today - verified_at).days
            if age < 0:
                errors.append(f"{schedule_id}: provenance as_of is in the future")
            elif schedule is active_schedule and max_age_days is not None and age > max_age_days:
                errors.append(f"{schedule_id}: provenance is stale ({age}d > {max_age_days}d)")

    for field, value in _price_points(model).items():
        if value <= 0:
            errors.append(f"{identity}: {field} price must be greater than zero")
        elif value > MAX_BUNDLED_PRICE_PER_MILLION_USD:
            errors.append(
                f"{identity}: {field} price exceeds "
                f"{MAX_BUNDLED_PRICE_PER_MILLION_USD} USD per million"
            )
    return errors


def suspicious_price_changes(
    before: ModelInfo,
    after: ModelInfo,
    before_price: ModelPrice,
    after_price: ModelPrice,
    *,
    max_factor: Decimal = MAX_AUTOMATED_PRICE_CHANGE_FACTOR,
    effective_on: date | None = None,
) -> list[str]:
    """Changes too risky for the automated verifier to place in a candidate snapshot."""

    if effective_on is not None:
        before_schedule = before_price.schedule_on(effective_on)
        after_schedule = after_price.schedule_on(effective_on)
        if before_schedule is not None:
            before_price = before_price.model_copy(update={"schedules": (before_schedule,)})
        if after_schedule is not None:
            after_price = after_price.model_copy(update={"schedules": (after_schedule,)})
    warnings: list[str] = []
    before_bounds = tuple(
        tier.max_input_tokens
        for schedule in before_price.schedules
        for tier in schedule.pricing.standard
    )
    after_bounds = tuple(
        tier.max_input_tokens
        for schedule in after_price.schedules
        for tier in schedule.pricing.standard
    )
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

    old = _price_points(before_price)
    new = _price_points(after_price)
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


def validate_price_book(
    price_book: PriceBook,
    *,
    today: date | None = None,
    max_age_days: int | None = CATALOG_MAX_AGE_DAYS,
    require_provider_coverage: bool = True,
) -> None:
    """Raise ``ValueError`` when a bundled price-book candidate is unsafe."""

    if type(price_book) is not PriceBook:
        raise TypeError("price_book must be a PriceBook instance.")
    reference = today or datetime.now(UTC).date()
    errors: list[str] = []
    if price_book.price_book_version != price_book.generated_at:
        errors.append("price_book_version must equal the dated generated_at snapshot version")
    try:
        generated_at = date.fromisoformat(price_book.generated_at)
    except ValueError:
        errors.append(f"price book generated_at is not an ISO date: {price_book.generated_at!r}")
    else:
        if generated_at > reference:
            errors.append(f"price book generated_at is in the future: {generated_at}")

    providers = {price.provider_name for price in price_book.prices}
    identities = [(price.provider_name, price.model) for price in price_book.prices]
    if identities != sorted(identities):
        errors.append("prices must be sorted by provider_name and model")
    missing = sorted(REQUIRED_PROVIDERS - providers)
    extra = sorted(providers - REQUIRED_PROVIDERS)
    if require_provider_coverage and missing:
        errors.append(f"missing required pricing providers: {', '.join(missing)}")
    if extra:
        errors.append(f"pricing providers outside the advertised default set: {', '.join(extra)}")
    for price in price_book.prices:
        errors.extend(price_policy_errors(price, today=reference, max_age_days=max_age_days))
    if errors:
        raise ValueError("Price book policy failed:\n- " + "\n- ".join(errors))


def validate_resource_pair(
    catalog: ModelCatalog,
    price_book: PriceBook,
    *,
    allowed_price_only_identities: Collection[tuple[str, str]] = (BUNDLED_PRICE_ONLY_IDENTITIES),
) -> None:
    """Validate the reviewed contract between Cayu's two bundled resources."""

    if type(catalog) is not ModelCatalog:
        raise TypeError("catalog must be a ModelCatalog instance.")
    if type(price_book) is not PriceBook:
        raise TypeError("price_book must be a PriceBook instance.")

    errors: list[str] = []
    if catalog.generated_at != price_book.generated_at:
        errors.append(
            "model catalog and price book must share one generated_at for atomic maintenance"
        )

    model_identities = {(model.provider_name.casefold(), model.model) for model in catalog.models}
    price_identities = {
        (price.provider_name.casefold(), price.model) for price in price_book.prices
    }
    allowed = {
        (provider_name.casefold(), model) for provider_name, model in allowed_price_only_identities
    }
    unexplained = sorted(
        (price.provider_name, price.model)
        for price in price_book.prices
        if (price.provider_name.casefold(), price.model) not in model_identities
        and (price.provider_name.casefold(), price.model) not in allowed
    )
    if unexplained:
        rendered = ", ".join(f"{provider}/{model}" for provider, model in unexplained)
        errors.append(f"undocumented bundled price-only identities: {rendered}")
    missing_prices = sorted(model_identities - price_identities)
    if missing_prices:
        rendered = ", ".join(f"{provider}/{model}" for provider, model in missing_prices)
        errors.append(f"bundled model identities without canonical prices: {rendered}")

    if errors:
        raise ValueError("Bundled resource-pair policy failed:\n- " + "\n- ".join(errors))
