"""Structured-output schema the verifier agent returns, and the parse that turns it
into a corrected ModelInfo (provenance 'official') or a flagged outcome.

The schema spans the UNION of how different providers price (flat vs size-tiered rates;
cached-read discount; Anthropic-style 5m/1h cache WRITE tiers; batch). Each field says when it
applies so the agent fills what THIS provider exposes and nulls the rest. `evidence` is required:
the agent must quote the verbatim pricing row it read each number from — that grounding is what
makes the numbers (context window included) trustworthy instead of guessed.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from datetime import date, timedelta
from decimal import Decimal
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from cayu import ModelInfo, ModelPrice, PriceSchedule, PriceTier, Provenance
from maintenance.model_catalog.verify import VerifyOutcome

_PRICE = {"type": ["string", "number", "null"]}

VERIFIED_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "confirmed": {
            "type": "boolean",
            "description": "true ONLY if you read these values off an authoritative "
            "official page for this exact model.",
        },
        "context_window": {
            "type": ["integer", "null"],
            "minimum": 1,
            "description": "Advertised max INPUT context as a single round number "
            "(e.g. 200000, 1000000). Do NOT add max-output tokens or sum two values. "
            "Null if the page does not clearly state it.",
        },
        "input_per_million": dict(_PRICE, description="Standard input price, USD per 1M tokens."),
        "output_per_million": dict(_PRICE, description="Standard output price, USD per 1M tokens."),
        "cache_read_per_million": dict(
            _PRICE,
            description="Discounted price for a CACHE HIT on "
            "input (cached/'cached input' tokens). Most providers have this.",
        ),
        "cache_write_5m_per_million": dict(
            _PRICE,
            description="Price to WRITE input into a prompt "
            "cache with the default ~5-minute TTL. Provider-specific "
            "(Anthropic). Null if the provider caches automatically "
            "with no separate write charge (e.g. OpenAI, Google).",
        ),
        "cache_write_1h_per_million": dict(
            _PRICE,
            description="Price to WRITE input into a prompt "
            "cache with the extended 1-hour TTL — i.e. the cost of "
            "keeping the cache alive longer. Provider-specific "
            "(Anthropic). Null if not offered.",
        ),
        "batch_input_per_million": dict(
            _PRICE,
            description="Input price under the batch/async API "
            "(often ~50% off). Null if not offered.",
        ),
        "batch_output_per_million": dict(
            _PRICE, description="Output price under the batch API. Null if not offered."
        ),
        "context_tiers": {
            "type": ["array", "null"],
            "description": "ONLY if the provider charges different rates by input size (e.g. Google "
            "≤200k vs >200k, OpenAI >272k). One entry per band, ascending. Omit/null for flat pricing.",
            "items": {
                "type": "object",
                "properties": {
                    "up_to_tokens": {
                        "type": ["integer", "null"],
                        "minimum": 1,
                        "description": "Upper bound of this band; null = 'and above'.",
                    },
                    "input_per_million": _PRICE,
                    "output_per_million": _PRICE,
                    "cache_read_per_million": _PRICE,
                    "cache_write_per_million": dict(
                        _PRICE,
                        description="Context-band cache-write price when it changes with input size.",
                    ),
                },
            },
        },
        "deprecated": {
            "type": "boolean",
            "description": "true if the page marks the model deprecated/retired/legacy.",
        },
        "source_url": {"type": "string", "description": "URL of the official page you read."},
        "pricing_effective_from": {
            "type": ["string", "null"],
            "description": "Published ISO date when the reported rates begin. Use null when "
            "the rates apply now and the page gives no explicit start date. For an announced "
            "future transition, report the future rates and their future start date.",
        },
        "evidence": {
            "type": "string",
            "maxLength": 2000,
            "description": "REQUIRED. Verbatim quote of the exact pricing row(s)/cell(s) you "
            "read — include the model name and every number you reported. This is how the "
            "values are trusted; if you cannot quote it, set confirmed=false.",
        },
        "model_source_url": {
            "type": ["string", "null"],
            "description": "Official model/capability page used for model facts, or null when "
            "model facts were not independently verified.",
        },
        "model_evidence": {
            "type": ["string", "null"],
            "maxLength": 2000,
            "description": "Verbatim model-fact evidence supporting context/lifecycle updates, "
            "or null when only pricing was verified.",
        },
        "note": {"type": "string"},
    },
}

# NATIVE structured output uses OpenAI strict mode: every object needs additionalProperties:false
# and ALL keys in `required` (optionality is expressed by nullable types, which our price/tier
# fields already are). Derive these so the lists never drift from `properties`.
VERIFIED_SCHEMA["additionalProperties"] = False
VERIFIED_SCHEMA["required"] = list(VERIFIED_SCHEMA["properties"])
_TIER_ITEMS: dict[str, Any] = VERIFIED_SCHEMA["properties"]["context_tiers"]["items"]
_TIER_ITEMS["additionalProperties"] = False
_TIER_ITEMS["required"] = list(_TIER_ITEMS["properties"])


_NUMBER_TOKEN = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"
_NUMBER_RANGE = re.compile(
    rf"{_NUMBER_TOKEN}\s*(?:-|\u2013|\u2014|to)\s*{_NUMBER_TOKEN}",
    re.IGNORECASE,
)


def _dec(v: Any) -> Decimal | None:
    """Parse a price the model returned. It may arrive as a number or a noisy string
    ("$5.00", "5.00 USD", "free", "N/A"). Extract the first numeric token; if there isn't
    one, return None so the field falls back to the curated value instead of crashing."""
    if v is None:
        return None
    if type(v) in {int, float}:
        return Decimal(str(v))
    if not isinstance(v, str):
        return None
    normalized = v.replace(",", "")
    if _NUMBER_RANGE.search(normalized):
        return None
    m = re.search(_NUMBER_TOKEN, normalized)
    return Decimal(m.group()) if m else None


def _tier(entry: dict[str, Any]) -> PriceTier | None:
    """Build a PriceTier from a context_tiers entry; None if it lacks usable input+output."""
    inp, out = _dec(entry.get("input_per_million")), _dec(entry.get("output_per_million"))
    if inp is None or out is None:
        return None
    up = entry.get("up_to_tokens")
    if up is not None and (type(up) is not int or up <= 0):
        return None
    return PriceTier(
        max_input_tokens=up,
        input_per_million=inp,
        output_per_million=out,
        cache_read_input_per_million=_dec(entry.get("cache_read_per_million")),
        cache_write_input_per_million=_dec(entry.get("cache_write_per_million")),
    )


def normalized_source_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, "", ""))


def parse_verified(
    data: dict[str, Any],
    original: ModelInfo,
    original_price: ModelPrice,
    *,
    as_of: str,
    browsed_urls: Collection[str],
    browsed_pricing_modes: Mapping[str, Collection[str]],
) -> VerifyOutcome:
    if not data.get("confirmed"):
        return VerifyOutcome(
            verified=False, note=data.get("note", "not confirmed by official page")
        )
    evidence = data.get("evidence")
    if not isinstance(evidence, str) or not evidence.strip():
        return VerifyOutcome(
            verified=False,
            note="official-page verification returned no grounding evidence",
        )
    source_url = data.get("source_url")
    if not isinstance(source_url, str) or not source_url.strip():
        return VerifyOutcome(verified=False, note="verification returned no official source URL")
    normalized_browsed = {
        normalized_source_url(url): url
        for url in sorted(url for url in browsed_urls if isinstance(url, str) and url.strip())
    }
    normalized_claim = normalized_source_url(source_url)
    if normalized_claim not in normalized_browsed:
        return VerifyOutcome(
            verified=False,
            note="claimed source URL was not visited by the page-reading tools",
        )
    normalized_modes: dict[str, set[str]] = {}
    for url, modes in browsed_pricing_modes.items():
        if isinstance(url, str) and url.strip():
            normalized_modes.setdefault(normalized_source_url(url), set()).update(modes)
    source_modes = normalized_modes.get(normalized_claim, set())
    if "standard" not in source_modes:
        return VerifyOutcome(
            verified=False,
            note="standard pricing mode was not verified on the claimed source URL",
        )
    # Provenance is derived from the browser trace, not copied from the model's claim.
    source_url = normalized_source_url(normalized_browsed[normalized_claim])

    model_source = data.get("model_source_url")
    model_evidence = data.get("model_evidence")
    model_provenance: Provenance | None = None
    if model_source is not None or model_evidence is not None:
        if not isinstance(model_source, str) or not model_source.strip():
            return VerifyOutcome(
                verified=False,
                note="model-fact verification returned no official source URL",
            )
        if not isinstance(model_evidence, str) or not model_evidence.strip():
            return VerifyOutcome(
                verified=False,
                note="model-fact verification returned no grounding evidence",
            )
        normalized_model_claim = normalized_source_url(model_source)
        if normalized_model_claim not in normalized_browsed:
            return VerifyOutcome(
                verified=False,
                note="claimed model source URL was not visited by the page-reading tools",
            )
        model_provenance = Provenance(
            source="official",
            url=normalized_source_url(normalized_browsed[normalized_model_claim]),
            as_of=as_of,
        )

    try:
        verified_on = date.fromisoformat(as_of)
    except ValueError:
        return VerifyOutcome(verified=False, note="as_of is not an ISO date")
    raw_pricing_effective_from = data.get("pricing_effective_from")
    if raw_pricing_effective_from is None:
        pricing_effective_from = verified_on
    elif isinstance(raw_pricing_effective_from, str):
        try:
            pricing_effective_from = date.fromisoformat(raw_pricing_effective_from)
        except ValueError:
            return VerifyOutcome(
                verified=False,
                note="pricing_effective_from is not an ISO date",
            )
    else:
        return VerifyOutcome(
            verified=False,
            note="pricing_effective_from must be an ISO date or null",
        )
    if pricing_effective_from == date.min:
        return VerifyOutcome(
            verified=False,
            note="pricing_effective_from must be later than 0001-01-01",
        )

    target_schedule = original_price.schedule_on(pricing_effective_from)
    reference_schedule = target_schedule or original_price.schedule_on(verified_on)
    if reference_schedule is None:
        reference_schedule = original_price.schedules[-1]
    original_pricing = reference_schedule.pricing

    # An explicit null/empty context_tiers value means flat pricing and removes curated upper
    # tiers. Missing or malformed output is not authoritative and retains them. A valid non-empty
    # list replaces the complete tier set.
    raw_tiers = data.get("context_tiers")
    tier_entries = raw_tiers if isinstance(raw_tiers, list) else None
    parsed_tiers = (
        [_tier(entry) for entry in tier_entries if isinstance(entry, dict)]
        if tier_entries is not None
        else None
    )
    tiers_valid = (
        parsed_tiers is not None
        and len(parsed_tiers) == len(tier_entries or [])
        and all(tier is not None for tier in parsed_tiers)
    )
    tiers = [tier for tier in (parsed_tiers or []) if tier is not None]
    if tiers and tiers_valid:
        tiers.sort(key=lambda t: (t.max_input_tokens is None, t.max_input_tokens or 0))
        new_standard = tuple(tiers)
    else:
        base = original_pricing.base()
        base_over: dict[str, Any] = {}
        for field, src in (
            ("input_per_million", "input_per_million"),
            ("output_per_million", "output_per_million"),
            ("cache_read_input_per_million", "cache_read_per_million"),
        ):
            raw = data.get(src)
            dec = _dec(raw)
            if raw is None and src in data and field == "cache_read_input_per_million":
                base_over[field] = None
            elif dec is not None:  # skip unparseable / absent -> keep curated
                base_over[field] = dec
        keep_upper_tiers = not (
            "context_tiers" in data and (raw_tiers is None or (tier_entries == [] and tiers_valid))
        )
        new_standard = (
            base.model_copy(update=base_over),
            *(original_pricing.standard[1:] if keep_upper_tiers else ()),
        )

    pricing_over: dict[str, Any] = {"standard": new_standard}
    for field, src in (
        ("cache_write_5m_per_million", "cache_write_5m_per_million"),
        ("cache_write_1h_per_million", "cache_write_1h_per_million"),
    ):
        raw = data.get(src)
        dec = _dec(raw)
        if raw is None and src in data:
            pricing_over[field] = None
        elif dec is not None:
            pricing_over[field] = dec
    raw_bi = data.get("batch_input_per_million")
    raw_bo = data.get("batch_output_per_million")
    bi, bo = _dec(raw_bi), _dec(raw_bo)
    batch_values_reported = raw_bi is not None or raw_bo is not None
    if batch_values_reported and "batch" not in source_modes:
        return VerifyOutcome(
            verified=False,
            note="batch pricing was reported without verified Batch mode on the claimed source URL",
        )
    if "batch" in source_modes:
        if (
            "batch_input_per_million" in data
            and "batch_output_per_million" in data
            and raw_bi is None
            and raw_bo is None
        ):
            pricing_over["batch"] = None
        elif bi is not None and bo is not None:
            pricing_over["batch"] = PriceTier(input_per_million=bi, output_per_million=bo)
    new_pricing = original_pricing.model_copy(update=pricing_over)

    pricing_provenance = Provenance(source="official", url=source_url, as_of=as_of)
    if target_schedule is None:
        future_starts = tuple(
            schedule.effective_from
            for schedule in original_price.schedules
            if schedule.effective_from is not None
            and schedule.effective_from > pricing_effective_from
        )
        updated_schedule = PriceSchedule(
            effective_from=pricing_effective_from,
            effective_through=(min(future_starts) - timedelta(days=1) if future_starts else None),
            pricing=new_pricing,
            provenance=pricing_provenance,
        )
        updated_schedules = tuple(
            sorted(
                (*original_price.schedules, updated_schedule),
                key=lambda schedule: schedule.effective_from or date.min,
            )
        )
    else:
        target_index = next(
            index
            for index, schedule in enumerate(original_price.schedules)
            if schedule is target_schedule
        )
        if (
            new_pricing != target_schedule.pricing
            and target_schedule.effective_from != pricing_effective_from
        ):
            previous_schedule = target_schedule.model_copy(
                update={"effective_through": pricing_effective_from - timedelta(days=1)}
            )
            updated_schedule = PriceSchedule(
                effective_from=pricing_effective_from,
                effective_through=target_schedule.effective_through,
                pricing=new_pricing,
                provenance=pricing_provenance,
            )
            updated_schedules = (
                *original_price.schedules[:target_index],
                previous_schedule,
                updated_schedule,
                *original_price.schedules[target_index + 1 :],
            )
        else:
            updated_schedule = target_schedule.model_copy(
                update={"pricing": new_pricing, "provenance": pricing_provenance}
            )
            updated_schedules = (
                *original_price.schedules[:target_index],
                updated_schedule,
                *original_price.schedules[target_index + 1 :],
            )
    corrected_price = ModelPrice.model_validate(
        original_price.model_copy(update={"schedules": updated_schedules}).model_dump(mode="json")
    )

    model_updates: dict[str, Any] = {}
    if model_provenance is not None:
        model_updates["provenance"] = model_provenance
        if type(data.get("context_window")) is int:
            model_updates["context_window"] = data["context_window"]
        if "deprecated" in data:
            model_updates["deprecated"] = bool(data["deprecated"])

    corrected = ModelInfo.model_validate(
        original.model_copy(update=model_updates).model_dump(mode="json")
    )
    return VerifyOutcome(
        verified=True,
        model=corrected,
        price=corrected_price,
        note=data.get("note", "official-verified"),
        evidence=evidence.strip(),
        pricing_provenance=pricing_provenance,
        pricing_effective_from=pricing_effective_from,
        model_evidence=(model_evidence.strip() if isinstance(model_evidence, str) else ""),
    )
