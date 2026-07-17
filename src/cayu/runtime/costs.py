from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, date, datetime
from decimal import Decimal
from importlib import resources
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, NamedTuple, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_serializer,
    field_validator,
    model_validator,
)

from cayu._validation import (
    FrozenJsonDict,
    copy_json_value,
    require_clean_nonblank,
)
from cayu.core.billing import BillingIdentity, PricingContext
from cayu.core.events import Event, EventType
from cayu.runtime.usage import UsageMetrics, usage_metrics_from_event_payload

_TOKENS_PER_MILLION = Decimal("1000000")

_MatchT = TypeVar("_MatchT")


_MODEL_ENTITY_CONFIG = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())


class Provenance(BaseModel):
    """Where model or pricing facts came from and when they were checked."""

    model_config = _MODEL_ENTITY_CONFIG

    source: str
    url: str
    as_of: str

    @field_validator("source", "url", "as_of")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)


class PriceTier(BaseModel):
    """One context band of tiered pricing.

    Tiers are ordered ascending by ``max_input_tokens``; the final tier may use
    ``None`` to mean "and above".
    """

    model_config = _MODEL_ENTITY_CONFIG

    max_input_tokens: StrictInt | None = Field(default=None, gt=0)
    input_per_million: Decimal = Field(ge=0)
    output_per_million: Decimal = Field(ge=0)
    cache_read_input_per_million: Decimal | None = Field(default=None, ge=0)
    cache_write_input_per_million: Decimal | None = Field(
        default=None,
        ge=0,
        exclude_if=lambda value: value is None,
    )

    @field_validator(
        "input_per_million",
        "output_per_million",
        "cache_read_input_per_million",
        "cache_write_input_per_million",
    )
    @classmethod
    def validate_decimal(cls, value: Decimal | None, info) -> Decimal | None:
        if value is None:
            return None
        if not value.is_finite():
            raise ValueError(f"{info.field_name} must be finite.")
        return value


class TieredPricing(BaseModel):
    """Full pricing for one model: context-band tiers plus optional batch and
    cache-write-duration rates.

    Only the context-band ``standard`` tiers feed the cost engine today. A tier-local
    cache-write rate takes precedence over the model-wide
    ``cache_write_5m_per_million`` fallback.
    ``batch`` is carried for completeness and future use. Contextual pricing
    applies ``cache_write_1h_per_million`` only to usage explicitly attributed
    to a one-hour cache write.
    """

    model_config = _MODEL_ENTITY_CONFIG

    currency: str = "USD"
    standard: tuple[PriceTier, ...] = Field(min_length=1)
    batch: PriceTier | None = None
    cache_write_5m_per_million: Decimal | None = Field(default=None, ge=0)
    cache_write_1h_per_million: Decimal | None = Field(default=None, ge=0)

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("cache_write_5m_per_million", "cache_write_1h_per_million")
    @classmethod
    def validate_decimal(cls, value: Decimal | None, info) -> Decimal | None:
        if value is None:
            return None
        if not value.is_finite():
            raise ValueError(f"{info.field_name} must be finite.")
        return value

    @model_validator(mode="after")
    def validate_tier_order(self) -> TieredPricing:
        bounded = [
            tier.max_input_tokens for tier in self.standard if tier.max_input_tokens is not None
        ]
        if bounded != sorted(bounded):
            raise ValueError("standard pricing tiers must be ascending by max_input_tokens.")
        if len(bounded) != len(set(bounded)):
            raise ValueError("standard pricing tiers must have distinct max_input_tokens.")
        open_ended = [
            index for index, tier in enumerate(self.standard) if tier.max_input_tokens is None
        ]
        if len(open_ended) > 1:
            raise ValueError("standard pricing may have at most one open-ended tier.")
        if open_ended and open_ended[0] != len(self.standard) - 1:
            raise ValueError("the open-ended pricing tier must be last.")
        return self

    def base(self) -> PriceTier:
        """The smallest-context (first) tier — the base price."""
        return self.standard[0]

    def tier_for(self, input_tokens: int) -> PriceTier:
        """The tier whose band contains ``input_tokens`` (the open-ended tier, or
        the largest bounded tier if none is open-ended and the count overflows)."""
        for tier in self.standard:
            if tier.max_input_tokens is None or input_tokens <= tier.max_input_tokens:
                return tier
        return self.standard[-1]


class PriceSchedule(BaseModel):
    """One price schedule, effective on inclusive UTC calendar dates."""

    model_config = _MODEL_ENTITY_CONFIG

    effective_from: date | None = None
    effective_through: date | None = None
    pricing: TieredPricing
    provenance: Provenance

    @model_validator(mode="after")
    def validate_window(self) -> PriceSchedule:
        if (
            self.effective_from is not None
            and self.effective_through is not None
            and self.effective_from > self.effective_through
        ):
            raise ValueError("price schedule effective_from must not follow effective_through.")
        return self

    def applies_on(self, effective_on: date) -> bool:
        return (self.effective_from is None or effective_on >= self.effective_from) and (
            self.effective_through is None or effective_on <= self.effective_through
        )


class _PriceMatchRule(NamedTuple):
    model: str
    match: Literal["exact", "prefix"]


class PricingContextSelector(BaseModel):
    """Exact pricing dimensions accepted by one contextual price row."""

    model_config = _MODEL_ENTITY_CONFIG

    dimensions: Mapping[str, tuple[str, ...]]

    @field_validator("dimensions", mode="before")
    @classmethod
    def validate_dimensions(cls, value: Any) -> dict[str, tuple[str, ...]]:
        if not isinstance(value, Mapping) or not value:
            raise ValueError("pricing context dimensions must be a non-empty object.")
        result: dict[str, tuple[str, ...]] = {}
        for key, raw_values in value.items():
            if type(key) is not str:
                raise ValueError("pricing context dimension names must be strings.")
            clean_key = require_clean_nonblank(key, "dimension name")
            if not isinstance(raw_values, (list, tuple)) or not raw_values:
                raise ValueError(
                    f"Pricing context dimension {clean_key!r} must have allowed values."
                )
            values_list: list[str] = []
            for item in raw_values:
                if type(item) is not str:
                    raise ValueError(
                        f"Pricing context dimension {clean_key!r} values must be strings."
                    )
                values_list.append(require_clean_nonblank(item, f"dimensions.{clean_key}"))
            values = tuple(values_list)
            if len(values) != len(set(values)):
                raise ValueError(
                    f"Pricing context dimension {clean_key!r} values must be distinct."
                )
            result[clean_key] = values
        return result

    @field_validator("dimensions")
    @classmethod
    def freeze_dimensions(
        cls,
        value: Mapping[str, tuple[str, ...]],
    ) -> Mapping[str, tuple[str, ...]]:
        return FrozenJsonDict((key, tuple(allowed_values)) for key, allowed_values in value.items())

    @field_serializer("dimensions")
    def serialize_dimensions(
        self,
        value: Mapping[str, tuple[str, ...]],
    ) -> dict[str, list[str]]:
        return {key: list(values) for key, values in value.items()}

    def storage_key(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        return tuple(
            (key, tuple(sorted(values))) for key, values in sorted(self.dimensions.items())
        )

    def matches(self, context: PricingContext) -> bool:
        return set(self.dimensions) == set(context.dimensions) and all(
            context.dimensions[key] in values for key, values in self.dimensions.items()
        )


class ContextualPricingRequirement(BaseModel):
    """Commercial constraints every contextual price for one provider must declare."""

    model_config = _MODEL_ENTITY_CONFIG

    provider_name: str
    dimensions: tuple[str, ...]
    requires_cache_write_ttls: StrictBool = False

    @field_validator("provider_name")
    @classmethod
    def validate_provider_name(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("dimensions")
    @classmethod
    def validate_requirement_dimensions(
        cls,
        value: tuple[str, ...],
        info,
    ) -> tuple[str, ...]:
        result = tuple(require_clean_nonblank(item, info.field_name) for item in value)
        if not result or len(result) != len(set(result)):
            raise ValueError(
                "contextual pricing requirement dimensions must be non-empty/distinct."
            )
        return result


class PricingResourceMapping(BaseModel):
    """Map one opaque provider resource to an explicit pricing model."""

    model_config = _MODEL_ENTITY_CONFIG

    provider_name: str
    resource_id: str
    pricing_model: str

    @field_validator("provider_name", "resource_id", "pricing_model")
    @classmethod
    def validate_mapping_identity(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)


_BUILTIN_CONTEXTUAL_PRICING_REQUIREMENTS = (
    ContextualPricingRequirement(
        provider_name="bedrock",
        dimensions=("source_region", "service_tier"),
        requires_cache_write_ttls=True,
    ),
)


class ModelPrice(BaseModel):
    """Pricing identity and its non-overlapping effective schedules."""

    model_config = _MODEL_ENTITY_CONFIG

    provider_name: str
    model: str
    aliases: tuple[str, ...] = ()
    match: Literal["exact", "prefix"] = "prefix"
    match_prefixes: tuple[str, ...] = ()
    pricing_context: PricingContextSelector | None = None
    cache_write_ttls: tuple[Literal["5m", "1h"], ...] = ()
    schedules: tuple[PriceSchedule, ...] = Field(min_length=1)

    @classmethod
    def fixed(
        cls,
        *,
        provider_name: str,
        model: str,
        input_per_million: Decimal,
        output_per_million: Decimal,
        cache_read_input_per_million: Decimal | None = None,
        cache_write_input_per_million: Decimal | None = None,
        currency: str = "USD",
        aliases: tuple[str, ...] = (),
        match: Literal["exact", "prefix"] = "prefix",
        match_prefixes: tuple[str, ...] = (),
        pricing_context: PricingContextSelector | Mapping[str, tuple[str, ...]] | None = None,
        cache_write_ttls: tuple[Literal["5m", "1h"], ...] = (),
        provenance: Provenance | None = None,
    ) -> ModelPrice:
        """Build one application-owned price with no validity bound.

        Use explicit ``PriceSchedule`` values when a rate has a known start or end.
        """
        source = provenance or Provenance(
            source="application",
            url="application://price-book",
            as_of="unspecified",
        )
        if pricing_context is None or type(pricing_context) is PricingContextSelector:
            resolved_pricing_context = pricing_context
        else:
            resolved_pricing_context = PricingContextSelector(dimensions=dict(pricing_context))
        return cls(
            provider_name=provider_name,
            model=model,
            aliases=aliases,
            match=match,
            match_prefixes=match_prefixes,
            pricing_context=resolved_pricing_context,
            cache_write_ttls=cache_write_ttls,
            schedules=(
                PriceSchedule(
                    pricing=TieredPricing(
                        currency=currency,
                        standard=(
                            PriceTier(
                                input_per_million=input_per_million,
                                output_per_million=output_per_million,
                                cache_read_input_per_million=cache_read_input_per_million,
                                cache_write_input_per_million=cache_write_input_per_million,
                            ),
                        ),
                    ),
                    provenance=source,
                ),
            ),
        )

    @field_validator("provider_name", "model")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("aliases", "match_prefixes")
    @classmethod
    def validate_string_members(cls, value: tuple[str, ...], info) -> tuple[str, ...]:
        return tuple(require_clean_nonblank(member, info.field_name) for member in value)

    @model_validator(mode="after")
    def validate_schedules(self) -> ModelPrice:
        if self.pricing_context is not None and (
            self.match != "exact" or self.aliases or self.match_prefixes
        ):
            raise ValueError("Contextual pricing must use one exact model identity.")
        if self.cache_write_ttls and self.pricing_context is None:
            raise ValueError("cache_write_ttls require contextual pricing.")
        if len(self.cache_write_ttls) != len(set(self.cache_write_ttls)):
            raise ValueError("cache_write_ttls must be distinct.")
        ordered = sorted(
            self.schedules,
            key=lambda schedule: (
                date.min if schedule.effective_from is None else schedule.effective_from
            ),
        )
        if list(self.schedules) != ordered:
            raise ValueError("price schedules must be ordered by effective_from.")
        for previous, current in pairwise(ordered):
            if previous.effective_through is None or current.effective_from is None:
                raise ValueError("price schedules overlap.")
            if current.effective_from <= previous.effective_through:
                raise ValueError("price schedules overlap.")
        for schedule in self.schedules:
            if (
                "5m" in self.cache_write_ttls
                and schedule.pricing.cache_write_5m_per_million is None
                and any(
                    tier.cache_write_input_per_million is None for tier in schedule.pricing.standard
                )
            ):
                raise ValueError("5-minute cache TTL requires a 5-minute write rate.")
            if (
                "1h" in self.cache_write_ttls
                and schedule.pricing.cache_write_1h_per_million is None
            ):
                raise ValueError("1-hour cache TTL requires a 1-hour write rate.")
        return self

    def _match_rules(self) -> tuple[_PriceMatchRule, ...]:
        return (
            _PriceMatchRule(model=self.model, match=self.match),
            *(_PriceMatchRule(model=alias, match="exact") for alias in self.aliases),
            *(_PriceMatchRule(model=prefix, match="prefix") for prefix in self.match_prefixes),
        )

    def schedule_on(self, effective_on: date) -> PriceSchedule | None:
        return next(
            (schedule for schedule in self.schedules if schedule.applies_on(effective_on)),
            None,
        )


class _PriceBookMatch(NamedTuple):
    price: ModelPrice
    model: str
    match: Literal["exact", "prefix"]
    resource_mapping: bool = False


def _price_book_match_key(candidate: _PriceBookMatch) -> tuple[str, str, str]:
    return (candidate.price.provider_name, candidate.model, candidate.match)


def _price_context_matches(
    price: ModelPrice,
    context: PricingContext | None,
) -> bool:
    if context is None:
        return price.pricing_context is None
    return price.pricing_context is not None and price.pricing_context.matches(context)


def _pricing_context_selectors_overlap(
    left: PricingContextSelector,
    right: PricingContextSelector,
) -> bool:
    return set(left.dimensions) == set(right.dimensions) and all(
        set(values) & set(right.dimensions[key]) for key, values in left.dimensions.items()
    )


class PriceBook(BaseModel):
    """A versioned collection of application-owned commercial model prices."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    price_book_version: str = "application"
    generated_at: str = "unspecified"
    prices: tuple[ModelPrice, ...] = Field(min_length=1)
    contextual_pricing_requirements: tuple[ContextualPricingRequirement, ...] = ()
    resource_mappings: tuple[PricingResourceMapping, ...] = ()

    @field_validator("price_book_version", "generated_at")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_unique_matches(self) -> PriceBook:
        requirements = {
            _normalize_provider(requirement.provider_name): requirement
            for requirement in (
                *_BUILTIN_CONTEXTUAL_PRICING_REQUIREMENTS,
                *self.contextual_pricing_requirements,
            )
        }
        if len(requirements) != (
            len(_BUILTIN_CONTEXTUAL_PRICING_REQUIREMENTS)
            + len(self.contextual_pricing_requirements)
        ):
            raise ValueError("Contextual pricing requirements must have unique providers.")
        seen: set[tuple[str, str, str, tuple[tuple[str, tuple[str, ...]], ...] | None]] = set()
        contextual_selectors: dict[
            tuple[str, str, str],
            list[PricingContextSelector],
        ] = {}
        for price in self.prices:
            requirement = requirements.get(_normalize_provider(price.provider_name))
            required_dimensions = None if requirement is None else set(requirement.dimensions)
            if requirement is not None and (
                price.pricing_context is None
                or set(price.pricing_context.dimensions) != required_dimensions
            ):
                dimensions = ", ".join(sorted(requirement.dimensions))
                raise ValueError(
                    f"{price.provider_name} pricing requires exact contextual dimensions: "
                    f"{dimensions}."
                )
            for rule in price._match_rules():
                match_key = _match_key(price.provider_name, rule.model, rule.match)
                key = (
                    *match_key,
                    (
                        None
                        if price.pricing_context is None
                        else price.pricing_context.storage_key()
                    ),
                )
                if key in seen:
                    raise ValueError("Price book contains duplicate provider/model/match entries.")
                seen.add(key)
                if price.pricing_context is not None:
                    existing = contextual_selectors.setdefault(match_key, [])
                    if any(
                        _pricing_context_selectors_overlap(price.pricing_context, other)
                        for other in existing
                    ):
                        raise ValueError("Price book contains overlapping pricing contexts.")
                    existing.append(price.pricing_context)
        mapping_keys = [
            (_normalize_provider(mapping.provider_name), mapping.resource_id)
            for mapping in self.resource_mappings
        ]
        if len(mapping_keys) != len(set(mapping_keys)):
            raise ValueError("Price book contains duplicate resource mappings.")
        return self

    def _requires_cache_write_ttls(self, provider_name: str) -> bool:
        requirement = next(
            (
                requirement
                for requirement in (
                    *_BUILTIN_CONTEXTUAL_PRICING_REQUIREMENTS,
                    *self.contextual_pricing_requirements,
                )
                if _normalize_provider(requirement.provider_name)
                == _normalize_provider(provider_name)
            ),
            None,
        )
        return requirement is not None and requirement.requires_cache_write_ttls

    def _resolve_match(
        self,
        *,
        provider_name: str | None,
        model: str | None,
        billing_identity: BillingIdentity | None = None,
        pricing_context: PricingContext | None = None,
    ) -> _PriceBookMatch | None:
        pricing_provider_name = (
            billing_identity.provider_name if billing_identity is not None else provider_name
        )
        pricing_model = billing_identity.resource_id if billing_identity is not None else model
        mapped = False
        if billing_identity is not None:
            mapping = next(
                (
                    item
                    for item in self.resource_mappings
                    if _normalize_provider(item.provider_name)
                    == _normalize_provider(billing_identity.provider_name)
                    and item.resource_id == billing_identity.resource_id
                ),
                None,
            )
            if mapping is not None:
                pricing_model = mapping.pricing_model
                mapped = True
        candidates: tuple[_PriceBookMatch, ...] = tuple(
            _PriceBookMatch(
                price=price,
                model=rule.model,
                match=rule.match,
                resource_mapping=mapped,
            )
            for price in self.prices
            for rule in price._match_rules()
            if _price_context_matches(price, pricing_context)
        )
        return _best_match_record(
            candidates,
            provider_name=pricing_provider_name,
            model=pricing_model,
            key=_price_book_match_key,
        )


class ModelInfo(BaseModel):
    """A model's identity, objective capabilities, lifecycle, and provenance."""

    model_config = _MODEL_ENTITY_CONFIG

    provider_name: str
    model: str
    family: str | None = None
    aliases: tuple[str, ...] = ()
    # capabilities
    context_window: StrictInt | None = Field(default=None, gt=0)
    max_output_tokens: StrictInt | None = Field(default=None, gt=0)
    modalities_in: tuple[str, ...] = ("text",)
    modalities_out: tuple[str, ...] = ("text",)
    tool_calling: StrictBool = False
    reasoning: StrictBool = False
    structured_output: StrictBool = False
    prompt_caching: StrictBool = False
    # lifecycle
    release_date: date | None = None
    knowledge_cutoff: date | None = None
    deprecated: StrictBool = False
    retirement_date: date | None = None
    match: Literal["exact", "prefix"] = "prefix"
    match_prefixes: tuple[str, ...] = ()
    provenance: Provenance

    @field_validator("provider_name", "model")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("family")
    @classmethod
    def validate_optional_family(cls, value: str | None, info) -> str | None:
        # Optional, but a provided value must be a clean non-blank label (not "  ").
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("aliases", "match_prefixes", "modalities_in", "modalities_out")
    @classmethod
    def validate_string_members(cls, value: tuple[str, ...], info) -> tuple[str, ...]:
        return tuple(require_clean_nonblank(member, info.field_name) for member in value)

    def _match_rules(self) -> tuple[_ModelMatchRule, ...]:
        return (
            _ModelMatchRule(model=self.model, match=self.match),
            *(_ModelMatchRule(model=alias, match="exact") for alias in self.aliases),
            *(_ModelMatchRule(model=prefix, match="prefix") for prefix in self.match_prefixes),
        )


class _ModelMatchRule(NamedTuple):
    model: str
    match: Literal["exact", "prefix"]


class _CatalogMatch(NamedTuple):
    info: ModelInfo
    model: str
    match: Literal["exact", "prefix"]


class ModelCatalog(BaseModel):
    """A versioned metadata registry used for model discovery and routing."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    catalog_version: str
    generated_at: str
    models: tuple[ModelInfo, ...] = Field(min_length=1)

    @field_validator("catalog_version", "generated_at")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_unique_models(self) -> ModelCatalog:
        seen: set[tuple[str, str, str]] = set()
        for info in self.models:
            for rule in info._match_rules():
                key = _match_key(info.provider_name, rule.model, rule.match)
                if key in seen:
                    raise ValueError(
                        "Model catalog contains duplicate provider/model/match entries."
                    )
                seen.add(key)
        return self

    def _resolve_match(
        self,
        *,
        provider_name: str | None,
        model: str | None,
    ) -> _CatalogMatch | None:
        candidates = (
            _CatalogMatch(info=info, model=rule.model, match=rule.match)
            for info in self.models
            for rule in info._match_rules()
        )
        matched = _best_match_record(
            candidates,
            provider_name=provider_name,
            model=model,
            key=lambda candidate: (
                candidate.info.provider_name,
                candidate.model,
                candidate.match,
            ),
        )
        return matched

    def resolve(self, *, provider_name: str | None, model: str | None) -> ModelInfo | None:
        """The most-specific metadata record whose declared identity matches the model."""

        matched = self._resolve_match(provider_name=provider_name, model=model)
        return None if matched is None else matched.info

    def match(self, *, provider_name: str, model: str) -> ModelInfo | None:
        """Exact ``(provider_name, model)`` record lookup (see ``resolve`` for declared
        prefix matching)."""
        for info in self.models:
            if info.provider_name == provider_name and info.model == model:
                return info
        return None

    def route(
        self,
        *,
        provider_name: str | None = None,
        min_context: int = 0,
        needs_vision: bool = False,
        needs_tools: bool = False,
        include_deprecated: bool = False,
    ) -> list[ModelInfo]:
        """Non-deprecated models (by default) matching the capability filters."""
        selected: list[ModelInfo] = []
        for info in self.models:
            if not include_deprecated and info.deprecated:
                continue
            if provider_name is not None and _normalize_provider(
                info.provider_name
            ) != _normalize_provider(provider_name):
                continue
            if min_context and (info.context_window or 0) < min_context:
                continue
            if needs_vision and "image" not in info.modalities_in:
                continue
            if needs_tools and not info.tool_calling:
                continue
            selected.append(info)
        return selected


def dump_model_catalog(catalog: ModelCatalog) -> str:
    """Deterministic JSON: models sorted by (provider_name, model), sorted keys,
    trailing newline — stable enough to commit and diff."""
    if type(catalog) is not ModelCatalog:
        raise TypeError("catalog must be a ModelCatalog instance.")
    ordered = tuple(sorted(catalog.models, key=lambda model: (model.provider_name, model.model)))
    data = catalog.model_copy(update={"models": ordered}).model_dump(mode="json")
    return json.dumps(data, sort_keys=True, indent=2) + "\n"


def load_model_catalog(path: str | Path) -> ModelCatalog:
    """Load a catalog from a JSON file written by ``dump_model_catalog``."""
    return ModelCatalog.model_validate_json(Path(path).read_text())


def default_model_catalog() -> ModelCatalog:
    """Load Cayu's bundled, dated model-catalog snapshot.

    The resource is parsed and validated on every call so callers never share mutable
    ``ModelCatalog`` state. Loading is deterministic and performs no network access.
    """
    resource = resources.files("cayu.data").joinpath("default_model_catalog.json")
    return ModelCatalog.model_validate_json(resource.read_text(encoding="utf-8"))


def dump_price_book(price_book: PriceBook) -> str:
    """Serialize a price book deterministically for review and packaging."""
    if type(price_book) is not PriceBook:
        raise TypeError("price_book must be a PriceBook instance.")
    ordered = tuple(
        sorted(
            price_book.prices,
            key=lambda price: (
                price.provider_name,
                price.model,
                (() if price.pricing_context is None else price.pricing_context.storage_key()),
            ),
        )
    )
    data = price_book.model_copy(update={"prices": ordered}).model_dump(mode="json")
    data["resource_mappings"] = [
        mapping.model_dump(mode="json")
        for mapping in sorted(
            price_book.resource_mappings,
            key=lambda mapping: (mapping.provider_name, mapping.resource_id),
        )
    ]
    return json.dumps(data, sort_keys=True, indent=2) + "\n"


def load_price_book(path: str | Path) -> PriceBook:
    """Load a price book from a JSON file written by ``dump_price_book``."""
    return PriceBook.model_validate_json(Path(path).read_text())


def default_price_book() -> PriceBook:
    """Load Cayu's bundled, dated price book without network access."""
    resource = resources.files("cayu.data").joinpath("default_price_book.json")
    return PriceBook.model_validate_json(resource.read_text(encoding="utf-8"))


class CostLineItem(BaseModel):
    """Estimated cost for one model.completed event."""

    model_config = ConfigDict(extra="forbid")

    model_step: StrictInt = Field(ge=1)
    provider_name: str | None = None
    requested_model: str | None = None
    model: str | None = None
    billing_identity: BillingIdentity | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    pricing_provider_name: str | None = None
    pricing_model: str | None = None
    pricing_match: Literal["exact", "prefix", "resource_mapping"] | None = None
    pricing_provenance: Provenance | None = None
    pricing_effective_from: date | None = None
    pricing_effective_through: date | None = None
    # Ceiling of the tiered-pricing band that priced this step (None = open-ended).
    pricing_tier_max_input_tokens: StrictInt | None = None
    priced: StrictBool
    currency: str
    input_tokens: StrictInt = Field(ge=0)
    output_tokens: StrictInt = Field(ge=0)
    cache_read_input_tokens: StrictInt = Field(ge=0)
    cache_write_input_tokens: StrictInt = Field(ge=0)
    cache_write_5m_input_tokens: StrictInt = Field(
        default=0,
        ge=0,
        exclude_if=lambda value: value == 0,
    )
    cache_write_1h_input_tokens: StrictInt = Field(
        default=0,
        ge=0,
        exclude_if=lambda value: value == 0,
    )
    cache_write_unknown_ttl_input_tokens: StrictInt = Field(
        default=0,
        ge=0,
        exclude_if=lambda value: value == 0,
    )
    uncached_input_tokens: StrictInt = Field(ge=0)
    input_cost: Decimal = Field(ge=0)
    output_cost: Decimal = Field(ge=0)
    cache_read_input_cost: Decimal = Field(ge=0)
    cache_write_input_cost: Decimal = Field(ge=0)
    total_cost: Decimal = Field(ge=0)
    missing_pricing_reason: str | None = None

    @field_validator(
        "provider_name",
        "requested_model",
        "model",
        "pricing_provider_name",
        "pricing_model",
        "currency",
        "missing_pricing_reason",
    )
    @classmethod
    def validate_optional_strings(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)


class SessionCostSummary(BaseModel):
    """Estimated session cost derived from durable model.completed events."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    currency: str
    model_steps: StrictInt = Field(ge=0)
    priced_model_steps: StrictInt = Field(ge=0)
    unpriced_model_steps: StrictInt = Field(ge=0)
    total_cost: Decimal = Field(ge=0)
    line_items: tuple[CostLineItem, ...] = Field(default_factory=tuple)

    @field_validator("session_id", "currency")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)


class CausalBudgetCostSummary(BaseModel):
    """Estimated cost for all sessions sharing one causal budget id."""

    model_config = ConfigDict(extra="forbid")

    causal_budget_id: str
    session_ids: list[str] = Field(default_factory=list)
    session_count: StrictInt = Field(default=0, ge=0)
    currency: str
    model_steps: StrictInt = Field(ge=0)
    priced_model_steps: StrictInt = Field(ge=0)
    unpriced_model_steps: StrictInt = Field(ge=0)
    total_cost: Decimal = Field(ge=0)
    line_items: tuple[CostLineItem, ...] = Field(default_factory=tuple)
    session_costs: tuple[SessionCostSummary, ...] = Field(default_factory=tuple)

    @field_validator("causal_budget_id", "currency")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("session_ids", mode="before")
    @classmethod
    def copy_session_ids(cls, value: list[str], info) -> list[str]:
        return _copy_string_list(value, info.field_name)


def copy_price_book(price_book: PriceBook) -> PriceBook:
    if type(price_book) is not PriceBook:
        raise TypeError("price_book must be a PriceBook instance.")
    return PriceBook(
        price_book_version=price_book.price_book_version,
        generated_at=price_book.generated_at,
        prices=tuple(price.model_copy(deep=True) for price in price_book.prices),
        contextual_pricing_requirements=tuple(
            requirement.model_copy(deep=True)
            for requirement in price_book.contextual_pricing_requirements
        ),
        resource_mappings=tuple(
            mapping.model_copy(deep=True) for mapping in price_book.resource_mappings
        ),
    )


def estimate_session_cost(
    *,
    session_id: str,
    events: list[Event],
    pricing: PriceBook,
    currency: str = "USD",
) -> SessionCostSummary:
    session_id = require_clean_nonblank(session_id, "session_id")
    currency = require_clean_nonblank(currency, "currency").upper()
    pricing = copy_price_book(pricing)
    if type(events) is not list:
        raise TypeError("events must be a list.")

    return _estimate_session_cost(
        session_id=session_id,
        events=events,
        pricing=pricing,
        currency=currency,
    )


def _estimate_session_cost(
    *,
    session_id: str,
    events: list[Event],
    pricing: PriceBook,
    currency: str,
) -> SessionCostSummary:
    """Estimate one session from already validated scalar inputs and a pricing snapshot."""

    line_items: list[CostLineItem] = []
    model_step = 0
    for event in events:
        if type(event) is not Event:
            raise TypeError("events must contain Event instances.")
        if event.type != EventType.MODEL_COMPLETED:
            continue
        model_step += 1
        metrics = usage_metrics_from_event_payload(event.payload)
        if metrics is None:
            raw_identity = event.payload.get("billing_identity")
            line_items.append(
                _unpriced_line_item(
                    model_step=model_step,
                    provider_name=_optional_nonblank(event.payload.get("provider_name")),
                    requested_model=_optional_nonblank(event.payload.get("requested_model")),
                    model=_optional_nonblank(event.payload.get("model")),
                    currency=currency,
                    reason="model.completed event has no token usage metrics",
                    billing_identity=(
                        BillingIdentity.model_validate(raw_identity)
                        if type(raw_identity) is dict
                        else None
                    ),
                )
            )
            continue
        line_items.append(
            _cost_line_item(
                model_step=model_step,
                metrics=metrics,
                pricing=pricing,
                currency=currency,
                effective_on=_effective_date(event.timestamp),
            )
        )

    total_cost = sum((item.total_cost for item in line_items), Decimal("0"))
    priced_model_steps = sum(1 for item in line_items if item.priced)
    unpriced_model_steps = len(line_items) - priced_model_steps
    return SessionCostSummary(
        session_id=session_id,
        currency=currency,
        model_steps=len(line_items),
        priced_model_steps=priced_model_steps,
        unpriced_model_steps=unpriced_model_steps,
        total_cost=total_cost,
        line_items=tuple(line_items),
    )


def estimate_causal_budget_cost(
    *,
    causal_budget_id: str,
    session_ids: list[str],
    events: list[Event],
    pricing: PriceBook,
    currency: str = "USD",
) -> CausalBudgetCostSummary:
    causal_budget_id = require_clean_nonblank(causal_budget_id, "causal_budget_id")
    copied_session_ids = _copy_string_list(session_ids, "session_ids")
    known_session_ids = set(copied_session_ids)
    filtered_events: list[Event] = []
    events_by_session: dict[str, list[Event]] = {session_id: [] for session_id in known_session_ids}
    for event in events:
        if event.session_id not in known_session_ids:
            continue
        if type(event) is not Event:
            raise TypeError("events must contain Event instances.")
        filtered_events.append(event)
        events_by_session[event.session_id].append(event)

    pricing = copy_price_book(pricing)
    currency = require_clean_nonblank(currency, "currency").upper()
    summary = _estimate_session_cost(
        session_id=causal_budget_id,
        events=filtered_events,
        pricing=pricing,
        currency=currency,
    )
    session_costs = tuple(
        _estimate_session_cost(
            session_id=session_id,
            events=events_by_session[session_id],
            pricing=pricing,
            currency=currency,
        )
        for session_id in copied_session_ids
    )
    return CausalBudgetCostSummary(
        causal_budget_id=causal_budget_id,
        session_ids=copied_session_ids,
        session_count=len(copied_session_ids),
        currency=summary.currency,
        model_steps=summary.model_steps,
        priced_model_steps=summary.priced_model_steps,
        unpriced_model_steps=summary.unpriced_model_steps,
        total_cost=summary.total_cost,
        line_items=summary.line_items,
        session_costs=session_costs,
    )


class _ResolvedPrice(NamedTuple):
    price: ModelPrice
    matched_model: str
    match: Literal["exact", "prefix", "resource_mapping"]
    schedule: PriceSchedule
    tier: PriceTier
    requires_cache_write_ttls: bool

    @property
    def currency(self) -> str:
        return self.schedule.pricing.currency

    @property
    def input_per_million(self) -> Decimal:
        return self.tier.input_per_million

    @property
    def output_per_million(self) -> Decimal:
        return self.tier.output_per_million

    @property
    def cache_read_input_per_million(self) -> Decimal | None:
        return self.tier.cache_read_input_per_million

    @property
    def cache_write_input_per_million(self) -> Decimal | None:
        if self.tier.cache_write_input_per_million is not None:
            return self.tier.cache_write_input_per_million
        return self.schedule.pricing.cache_write_5m_per_million

    @property
    def cache_write_5m_per_million(self) -> Decimal | None:
        if self.tier.cache_write_input_per_million is not None:
            return self.tier.cache_write_input_per_million
        return self.schedule.pricing.cache_write_5m_per_million

    @property
    def cache_write_1h_per_million(self) -> Decimal | None:
        return self.schedule.pricing.cache_write_1h_per_million


class _PriceResolution(NamedTuple):
    resolved: _ResolvedPrice | None
    missing_reason: str | None


def resolve_price_book(
    price_book: PriceBook,
    *,
    provider_name: str | None,
    model: str | None,
    input_tokens: int,
    effective_on: date,
    billing_identity: BillingIdentity | None = None,
    pricing_context: PricingContext | None = None,
) -> _ResolvedPrice | None:
    """Resolve one price-book entry, effective schedule, and context tier."""
    return _resolve_price_book(
        price_book,
        provider_name=provider_name,
        model=model,
        input_tokens=input_tokens,
        effective_on=effective_on,
        billing_identity=billing_identity,
        pricing_context=pricing_context,
    ).resolved


def _resolve_price_book(
    price_book: PriceBook,
    *,
    provider_name: str | None,
    model: str | None,
    input_tokens: int,
    effective_on: date,
    billing_identity: BillingIdentity | None = None,
    pricing_context: PricingContext | None = None,
) -> _PriceResolution:
    if billing_identity is not None and pricing_context is None:
        if len(billing_identity.pricing_contexts) != 1:
            if billing_identity.pricing_contexts:
                return _PriceResolution(None, "billing identity has ambiguous pricing contexts")
        else:
            pricing_context = billing_identity.pricing_contexts[0]
    matched = price_book._resolve_match(
        provider_name=provider_name,
        model=model,
        billing_identity=billing_identity,
        pricing_context=pricing_context,
    )
    if matched is None:
        return _PriceResolution(None, "no matching model pricing")
    schedule = matched.price.schedule_on(effective_on)
    if schedule is None:
        schedules = matched.price.schedules
        latest = schedules[-1].effective_through
        reason = (
            "pricing schedule expired"
            if latest is not None and effective_on > latest
            else "no applicable pricing schedule"
        )
        return _PriceResolution(None, reason)
    return _PriceResolution(
        _ResolvedPrice(
            price=matched.price,
            matched_model=matched.model,
            match=("resource_mapping" if matched.resource_mapping else matched.match),
            schedule=schedule,
            tier=schedule.pricing.tier_for(input_tokens),
            requires_cache_write_ttls=(
                bool(matched.price.cache_write_ttls)
                or price_book._requires_cache_write_ttls(matched.price.provider_name)
            ),
        ),
        None,
    )


def _resolve_price(
    *,
    metrics: UsageMetrics,
    pricing: PriceBook,
    effective_on: date,
) -> _PriceResolution:
    # A provider-reported model is authoritative for billing. Falling back from a
    # present-but-unpriced resolved model to the requested model could silently apply
    # the wrong rate after routing; use the request only when no resolved model exists.
    billing_model = (
        metrics.billing_identity.resource_id
        if metrics.billing_identity is not None
        else metrics.model
        if metrics.model is not None
        else metrics.requested_model
    )
    return _resolve_price_book(
        pricing,
        provider_name=metrics.provider_name,
        model=billing_model,
        input_tokens=metrics.input_tokens,
        effective_on=effective_on,
        billing_identity=metrics.billing_identity,
    )


def _cost_line_item(
    *,
    model_step: int,
    metrics: UsageMetrics,
    pricing: PriceBook,
    currency: str,
    effective_on: date,
) -> CostLineItem:
    resolution = _resolve_price(metrics=metrics, pricing=pricing, effective_on=effective_on)
    if resolution.resolved is None:
        return _unpriced_line_item(
            model_step=model_step,
            provider_name=metrics.provider_name,
            requested_model=metrics.requested_model,
            model=metrics.model,
            currency=currency,
            reason=resolution.missing_reason or "no matching model pricing",
            metrics=metrics,
            billing_identity=metrics.billing_identity,
        )
    price = resolution.resolved

    if price.currency.upper() != currency.upper():
        return _unpriced_line_item(
            model_step=model_step,
            provider_name=metrics.provider_name,
            requested_model=metrics.requested_model,
            model=metrics.model,
            currency=currency,
            reason=f"pricing currency {price.currency} does not match requested {currency}",
            metrics=metrics,
            billing_identity=metrics.billing_identity,
        )

    uncached_input_tokens = metrics.cache.uncached_input_tokens
    if (
        uncached_input_tokens == 0
        and metrics.input_tokens > 0
        and metrics.cache.read_tokens == 0
        and metrics.cache.write_tokens == 0
    ):
        uncached_input_tokens = metrics.input_tokens

    cache_read_price = (
        price.cache_read_input_per_million
        if price.cache_read_input_per_million is not None
        else price.input_per_million
    )
    cache_write_price = (
        price.cache_write_input_per_million
        if price.cache_write_input_per_million is not None
        else price.input_per_million
    )

    input_cost = _token_cost(uncached_input_tokens, price.input_per_million)
    output_cost = _token_cost(metrics.output_tokens, price.output_per_million)
    cache_read_cost = _token_cost(metrics.cache.read_tokens, cache_read_price)
    if not price.requires_cache_write_ttls:
        cache_write_cost = _token_cost(metrics.cache.write_tokens, cache_write_price)
    else:
        ttl_resolution = _contextual_cache_write_cost(metrics=metrics, price=price)
        if isinstance(ttl_resolution, str):
            return _unpriced_line_item(
                model_step=model_step,
                provider_name=metrics.provider_name,
                requested_model=metrics.requested_model,
                model=metrics.model,
                currency=currency,
                reason=ttl_resolution,
                metrics=metrics,
                billing_identity=metrics.billing_identity,
            )
        cache_write_cost = ttl_resolution
    total_cost = input_cost + output_cost + cache_read_cost + cache_write_cost

    return CostLineItem(
        model_step=model_step,
        provider_name=metrics.provider_name,
        requested_model=metrics.requested_model,
        model=metrics.model,
        billing_identity=metrics.billing_identity,
        pricing_provider_name=price.price.provider_name,
        pricing_model=price.matched_model,
        pricing_match=price.match,
        pricing_provenance=price.schedule.provenance,
        pricing_effective_from=price.schedule.effective_from,
        pricing_effective_through=price.schedule.effective_through,
        pricing_tier_max_input_tokens=price.tier.max_input_tokens,
        priced=True,
        currency=currency.upper(),
        input_tokens=metrics.input_tokens,
        output_tokens=metrics.output_tokens,
        cache_read_input_tokens=metrics.cache.read_tokens,
        cache_write_input_tokens=metrics.cache.write_tokens,
        cache_write_5m_input_tokens=metrics.cache.write_5m_tokens,
        cache_write_1h_input_tokens=metrics.cache.write_1h_tokens,
        cache_write_unknown_ttl_input_tokens=metrics.cache.write_unknown_ttl_tokens,
        uncached_input_tokens=uncached_input_tokens,
        input_cost=input_cost,
        output_cost=output_cost,
        cache_read_input_cost=cache_read_cost,
        cache_write_input_cost=cache_write_cost,
        total_cost=total_cost,
    )


def _unpriced_line_item(
    *,
    model_step: int,
    provider_name: str | None,
    model: str | None,
    currency: str,
    reason: str,
    metrics: UsageMetrics | None = None,
    requested_model: str | None = None,
    billing_identity: BillingIdentity | None = None,
) -> CostLineItem:
    return CostLineItem(
        model_step=model_step,
        provider_name=provider_name,
        requested_model=requested_model,
        model=model,
        billing_identity=billing_identity,
        priced=False,
        currency=currency.upper(),
        input_tokens=0 if metrics is None else metrics.input_tokens,
        output_tokens=0 if metrics is None else metrics.output_tokens,
        cache_read_input_tokens=0 if metrics is None else metrics.cache.read_tokens,
        cache_write_input_tokens=0 if metrics is None else metrics.cache.write_tokens,
        cache_write_5m_input_tokens=0 if metrics is None else metrics.cache.write_5m_tokens,
        cache_write_1h_input_tokens=0 if metrics is None else metrics.cache.write_1h_tokens,
        cache_write_unknown_ttl_input_tokens=(
            0 if metrics is None else metrics.cache.write_unknown_ttl_tokens
        ),
        uncached_input_tokens=0 if metrics is None else metrics.cache.uncached_input_tokens,
        input_cost=Decimal("0"),
        output_cost=Decimal("0"),
        cache_read_input_cost=Decimal("0"),
        cache_write_input_cost=Decimal("0"),
        total_cost=Decimal("0"),
        missing_pricing_reason=reason,
    )


def _contextual_cache_write_cost(
    *,
    metrics: UsageMetrics,
    price: _ResolvedPrice,
) -> Decimal | str:
    five_minute_rate = price.cache_write_5m_per_million
    one_hour_rate = price.cache_write_1h_per_million
    supported = price.price.cache_write_ttls
    if metrics.cache.write_5m_tokens and "5m" not in supported:
        return "pricing does not declare 5-minute cache writes"
    if metrics.cache.write_1h_tokens and "1h" not in supported:
        return "pricing does not declare 1-hour cache writes"
    if metrics.cache.write_5m_tokens and five_minute_rate is None:
        return "pricing has no 5-minute cache-write rate"
    if metrics.cache.write_1h_tokens and one_hour_rate is None:
        return "pricing has no 1-hour cache-write rate"

    unknown_rate: Decimal | None = None
    if metrics.cache.write_unknown_ttl_tokens:
        if supported == ("5m",):
            unknown_rate = five_minute_rate
        elif supported == ("1h",):
            unknown_rate = one_hour_rate
        elif (
            set(supported) == {"5m", "1h"}
            and five_minute_rate is not None
            and five_minute_rate == one_hour_rate
        ):
            unknown_rate = five_minute_rate
        if unknown_rate is None:
            return "cache-write TTL is unknown and applicable rates are ambiguous"

    return (
        _token_cost(metrics.cache.write_5m_tokens, five_minute_rate or Decimal("0"))
        + _token_cost(metrics.cache.write_1h_tokens, one_hour_rate or Decimal("0"))
        + _token_cost(metrics.cache.write_unknown_ttl_tokens, unknown_rate or Decimal("0"))
    )


def _token_cost(tokens: int, price_per_million: Decimal) -> Decimal:
    return Decimal(tokens) * price_per_million / _TOKENS_PER_MILLION


def _effective_date(timestamp: datetime) -> date:
    """Normalize runtime timestamps to the UTC calendar date used by price schedules."""
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC).date()


def _normalize_provider(provider_name: str) -> str:
    """Case/whitespace-insensitive provider key, shared by matching, dedup, and routing."""
    return provider_name.strip().lower()


def _match_key(provider_name: str, model: str, match: str) -> tuple[str, str, str]:
    """The dedup/identity key for a priced record: provider is case-insensitive."""
    return (_normalize_provider(provider_name), model.strip(), match)


def _best_match_record(
    records: Iterable[_MatchT],
    *,
    provider_name: str | None,
    model: str | None,
    key: Callable[[_MatchT], tuple[str, str, str]],
) -> _MatchT | None:
    """Most-specific prefix/exact match, or ``None``.

    ``key(record)`` yields ``(provider_name, model, match)`` with ``match`` in
    {"exact", "prefix"}. Provider is compared case-insensitively; an exact match, then a
    longer configured model, wins ties, and the earliest record wins an outright tie — the
    single matching rule shared by price books and model catalogs.
    Records are read directly (no throwaway projection list per call).
    """
    provider = _normalize_provider(provider_name) if type(provider_name) is str else None
    model_name = model.strip() if type(model) is str else None
    if provider is None or model_name is None:
        return None

    best: tuple[tuple[int, int], _MatchT] | None = None
    for record in records:
        cand_provider, cand_model, cand_match = key(record)
        if _normalize_provider(cand_provider) != provider:
            continue
        configured_model = cand_model.strip()
        if (cand_match == "exact" and model_name == configured_model) or (
            cand_match == "prefix" and model_name.startswith(configured_model)
        ):
            score = (1 if cand_match == "exact" else 0, len(cand_model))
            if best is None or score > best[0]:
                best = (score, record)
    return None if best is None else best[1]


def _optional_nonblank(value: object) -> str | None:
    if type(value) is str and value.strip():
        return value
    return None


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
