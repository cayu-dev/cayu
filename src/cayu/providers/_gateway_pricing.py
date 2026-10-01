"""Projection of Gateway catalog prices into ordinary local execution estimates."""

from __future__ import annotations

import hashlib
import re
from decimal import ROUND_CEILING, Decimal, localcontext
from typing import Any, TypeGuard

from cayu.budgets.pricing import ModelPrice, PriceBook, Provenance

_TOKEN_FIELDS = {
    "input_tokens": "input_per_million",
    "output_tokens": "output_per_million",
    "cached_input_tokens": "cache_read_input_per_million",
    "cache_write_tokens": "cache_write_input_per_million",
}
_DIMENSIONS = {*_TOKEN_FIELDS, "reasoning_tokens", "requests", "provider_tool_calls"}
_MAX_QUANTITY = (1 << 53) - 1


def _identity(value: Any) -> TypeGuard[str]:
    return type(value) is str and re.fullmatch(r"[A-Za-z0-9_.:/-]{1,128}", value) is not None


def _version(value: dict[str, Any]) -> bool:
    version = value.get("schema_version", 1)
    return type(version) is int and version == 1


def _price(row: dict[str, Any], *, url: str) -> ModelPrice | None:
    metadata = row.get("cayu")
    if (
        not _version(row)
        or type(metadata) is not dict
        or not _version(metadata)
        or metadata.get("protocol") != "chat_completions"
        or metadata.get("currency") != "USD"
        or not _identity(metadata.get("price_id"))
    ):
        return None
    components = metadata.get("components")
    if type(components) is not list or not 1 <= len(components) <= 16:
        return None
    rates: dict[str, Decimal] = {}
    for part in components:
        if (
            type(part) is not dict
            or not _version(part)
            or set(part) - {"schema_version", "dimension", "nano_usd", "per_units"}
        ):
            return None
        dimension = part.get("dimension")
        amount, units = part.get("nano_usd"), part.get("per_units")
        if (
            type(dimension) is not str
            or dimension not in _DIMENSIONS
            or dimension in rates
            or type(amount) is not int
            or not 0 <= amount <= _MAX_QUANTITY
            or type(units) is not int
            or not 1 <= units <= _MAX_QUANTITY
        ):
            return None
        # Round repeating rational prices upward, independently of caller context.
        with localcontext() as context:
            context.prec = 50
            context.rounding = ROUND_CEILING
            rates[dimension] = Decimal(amount) / Decimal(units * 1000)
    if not {"input_tokens", "output_tokens"} <= rates.keys():
        return None
    if rates.get("requests", 0) or rates.get("provider_tool_calls", 0):
        return None
    rates["output_tokens"] = max(rates["output_tokens"], rates.get("reasoning_tokens", Decimal(0)))
    return ModelPrice.fixed(
        provider_name="cayu_gateway",
        model=row["id"],
        match="exact",
        input_per_million=rates["input_tokens"],
        output_per_million=rates["output_tokens"],
        cache_read_input_per_million=rates.get("cached_input_tokens"),
        cache_write_input_per_million=rates.get("cache_write_tokens"),
        provenance=Provenance(
            source=f"Gateway catalog price_id={metadata['price_id']}", url=url, as_of="unspecified"
        ),
    )


def catalog_price_book(models: list[dict[str, Any]], *, url: str) -> PriceBook:
    """Build one detached snapshot; unsupported rows never become zero prices."""
    if len(models) > 256:
        raise ValueError("Gateway catalog exceeds its model bound.")
    # The catalog supplies no authoritative timestamp. Local fetch time would
    # change ordinary budget/profile identities even when prices are unchanged.
    seen: set[str] = set()
    prices = []
    for row in models:
        model = row.get("id")
        if not _identity(model) or model in seen:
            raise ValueError("Gateway catalog has invalid or duplicate model identities.")
        seen.add(model)
        price = _price(row, url=url)
        if price is not None:
            prices.append(price)
    if not prices:
        raise ValueError("Gateway catalog has no supported token prices.")
    prices.sort(key=lambda price: price.model)
    digest = hashlib.sha256("\n".join(p.model_dump_json() for p in prices).encode()).hexdigest()
    return PriceBook(price_book_version=f"gateway-catalog:{digest}", prices=tuple(prices))
