from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from datetime import date
from decimal import Decimal
from importlib import resources
from pathlib import Path
from typing import Literal, NamedTuple, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._validation import copy_json_value, require_clean_nonblank
from cayu.core.events import Event, EventType
from cayu.runtime.usage import UsageMetrics, usage_metrics_from_event_payload

_TOKENS_PER_MILLION = Decimal("1000000")

_MatchT = TypeVar("_MatchT")


class ModelPricing(BaseModel):
    """Model pricing per 1M tokens, with optional projected tiers and provenance."""

    model_config = ConfigDict(extra="forbid")

    provider_name: str
    model: str
    input_per_million: Decimal = Field(ge=0)
    output_per_million: Decimal = Field(ge=0)
    cache_read_input_per_million: Decimal | None = Field(default=None, ge=0)
    cache_write_input_per_million: Decimal | None = Field(default=None, ge=0)
    currency: str = "USD"
    match: Literal["exact", "prefix"] = "prefix"
    pricing_tiers: tuple[PriceTier, ...] | None = Field(
        default=None,
        min_length=1,
        exclude_if=lambda value: value is None,
    )
    provenance: Provenance | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )

    @field_validator("provider_name", "model", "currency")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

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

    @model_validator(mode="after")
    def validate_pricing_tiers(self) -> ModelPricing:
        if self.pricing_tiers is None:
            return self
        tiered = TieredPricing(currency=self.currency, standard=self.pricing_tiers)
        base = tiered.base()
        if (
            self.input_per_million != base.input_per_million
            or self.output_per_million != base.output_per_million
            or self.cache_read_input_per_million != base.cache_read_input_per_million
            or (
                base.cache_write_input_per_million is not None
                and self.cache_write_input_per_million != base.cache_write_input_per_million
            )
        ):
            raise ValueError("ModelPricing base rates must match the first pricing tier.")
        return self


class PricingCatalog(BaseModel):
    """Collection of user-supplied model prices used for cost estimation."""

    model_config = ConfigDict(extra="forbid")

    prices: tuple[ModelPricing, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_matches(self) -> PricingCatalog:
        seen: set[tuple[str, str, str]] = set()
        for price in self.prices:
            key = _match_key(price.provider_name, price.model, price.match)
            if key in seen:
                raise ValueError("Pricing catalog contains duplicate provider/model/match entries.")
            seen.add(key)
        return self

    def match_price(self, *, provider_name: str | None, model: str | None) -> ModelPricing | None:
        def price_key(price: ModelPricing) -> tuple[str, str, str]:
            return (price.provider_name, price.model, price.match)

        return _best_match_record(
            self.prices,
            provider_name=provider_name,
            model=model,
            key=price_key,
        )


_MODEL_ENTITY_CONFIG = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())


class Provenance(BaseModel):
    """Where a model's facts came from and when (audit metadata; not priced)."""

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

    Only the context-band ``standard`` tiers feed the cost engine today (via
    ``ModelInfo.to_model_pricing``/``pricing_at``). A tier-local cache-write rate takes
    precedence over the legacy model-wide ``cache_write_5m_per_million`` fallback.
    ``batch`` and ``cache_write_1h_per_million`` are carried for completeness and future
    use — they are NOT auto-applied by ``estimate_session_cost``.
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
        """The smallest-context (first) tier — the flat back-compat price."""
        return self.standard[0]

    def tier_for(self, input_tokens: int) -> PriceTier:
        """The tier whose band contains ``input_tokens`` (the open-ended tier, or
        the largest bounded tier if none is open-ended and the count overflows)."""
        for tier in self.standard:
            if tier.max_input_tokens is None or input_tokens <= tier.max_input_tokens:
                return tier
        return self.standard[-1]


class ModelInfo(BaseModel):
    """A model's full profile: identity, objective capabilities, lifecycle, and
    tiered pricing.

    Projects into the compatibility cost shape via ``to_model_pricing()`` while retaining
    its tiers, or into one selected tier via ``pricing_at(input_tokens)``. ``match_prefixes``
    declares additional narrow prefix keys without broadening the canonical model key.
    Capabilities are objective facts only — no quality/benchmark scores.
    """

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
    # pricing
    match: Literal["exact", "prefix"] = "prefix"
    match_prefixes: tuple[str, ...] = ()
    pricing: TieredPricing
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

    def _model_pricing(
        self,
        tier: PriceTier,
        *,
        include_projection_metadata: bool = False,
    ) -> ModelPricing:
        def effective_cache_write_price(candidate: PriceTier) -> Decimal | None:
            return (
                candidate.cache_write_input_per_million
                if candidate.cache_write_input_per_million is not None
                else self.pricing.cache_write_5m_per_million
            )

        projected_tiers = (
            tuple(
                candidate.model_copy(
                    update={"cache_write_input_per_million": effective_cache_write_price(candidate)}
                )
                for candidate in self.pricing.standard
            )
            if include_projection_metadata
            else None
        )
        return ModelPricing(
            provider_name=self.provider_name,
            model=self.model,
            input_per_million=tier.input_per_million,
            output_per_million=tier.output_per_million,
            cache_read_input_per_million=tier.cache_read_input_per_million,
            cache_write_input_per_million=effective_cache_write_price(tier),
            currency=self.pricing.currency,
            match=self.match,
            pricing_tiers=projected_tiers,
            provenance=self.provenance,
        )

    def to_model_pricing(self) -> ModelPricing:
        """Compatibility row retaining the tier/provenance metadata used for pricing."""
        return self._model_pricing(self.pricing.base(), include_projection_metadata=True)

    def pricing_at(self, input_tokens: int) -> ModelPricing:
        """Flat ``ModelPricing`` from the tier that prices ``input_tokens``."""
        return self._model_pricing(self.pricing.tier_for(input_tokens))

    def _match_rules(self) -> tuple[_ModelMatchRule, ...]:
        return (
            _ModelMatchRule(model=self.model, match=self.match),
            *(_ModelMatchRule(model=alias, match="exact") for alias in self.aliases),
            *(_ModelMatchRule(model=prefix, match="prefix") for prefix in self.match_prefixes),
        )

    def _pricing_rows(self) -> tuple[ModelPricing, ...]:
        primary = self.to_model_pricing()
        return tuple(
            primary.model_copy(update={"model": rule.model, "match": rule.match})
            for rule in self._match_rules()
        )


class _ModelMatchRule(NamedTuple):
    model: str
    match: Literal["exact", "prefix"]


class _CatalogMatch(NamedTuple):
    info: ModelInfo
    model: str
    match: Literal["exact", "prefix"]


class ModelCatalog(BaseModel):
    """A versioned set of ``ModelInfo`` records: the model catalog.

    Cost and budget APIs can consume it directly to preserve context tiers. It can also
    project into the compatible ``PricingCatalog`` contract without losing pricing semantics
    via ``pricing_catalog()``. Use ``resolve()`` to find the record whose canonical key or
    exact alias, or explicit match prefix prices a runtime model id; ``match()`` is a strict
    canonical-record lookup.
    """

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
        # Mirror PricingCatalog.validate_unique_matches: reject duplicate matching keys at
        # construction rather than crashing later when the catalog is projected/priced.
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

    def pricing_catalog(self) -> PricingCatalog:
        return PricingCatalog(
            prices=tuple(price for model in self.models for price in model._pricing_rows())
        )

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
        """The most-specific record whose primary or explicit prefix matches the model."""

        matched = self._resolve_match(provider_name=provider_name, model=model)
        return None if matched is None else matched.info

    def _price_at(
        self,
        *,
        provider_name: str | None,
        model: str | None,
        input_tokens: int,
    ) -> tuple[ModelPricing, PriceTier] | None:
        matched = self._resolve_match(provider_name=provider_name, model=model)
        if matched is None:
            return None
        tier = matched.info.pricing.tier_for(input_tokens)
        price = matched.info._model_pricing(tier).model_copy(
            update={"model": matched.model, "match": matched.match}
        )
        return price, tier

    def match(self, *, provider_name: str, model: str) -> ModelInfo | None:
        """Exact ``(provider_name, model)`` record lookup (see ``resolve`` for pricing-style
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


PricingSource = PricingCatalog | ModelCatalog


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


class CostLineItem(BaseModel):
    """Estimated cost for one model.completed event."""

    model_config = ConfigDict(extra="forbid")

    model_step: StrictInt = Field(ge=1)
    provider_name: str | None = None
    requested_model: str | None = None
    model: str | None = None
    pricing_provider_name: str | None = None
    pricing_model: str | None = None
    pricing_match: Literal["exact", "prefix"] | None = None
    pricing_provenance: Provenance | None = None
    # Ceiling of the tiered-pricing band that priced this step (None = the open-ended tier, or
    # flat/no-catalog pricing). Lets an audit see which context band applied without the catalog.
    pricing_tier_max_input_tokens: StrictInt | None = None
    priced: StrictBool
    currency: str
    input_tokens: StrictInt = Field(ge=0)
    output_tokens: StrictInt = Field(ge=0)
    cache_read_input_tokens: StrictInt = Field(ge=0)
    cache_write_input_tokens: StrictInt = Field(ge=0)
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


def copy_pricing_catalog(catalog: PricingCatalog) -> PricingCatalog:
    if type(catalog) is not PricingCatalog:
        raise TypeError("Pricing catalog must be a PricingCatalog instance.")
    return PricingCatalog(prices=tuple(price.model_copy(deep=True) for price in catalog.prices))


def estimate_session_cost(
    *,
    session_id: str,
    events: list[Event],
    pricing: PricingSource | None = None,
    currency: str = "USD",
    catalog: ModelCatalog | None = None,
) -> SessionCostSummary:
    session_id = require_clean_nonblank(session_id, "session_id")
    currency = require_clean_nonblank(currency, "currency").upper()
    pricing, catalog = _split_pricing_sources(pricing, catalog)
    if type(events) is not list:
        raise TypeError("events must be a list.")

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
            line_items.append(
                _unpriced_line_item(
                    model_step=model_step,
                    provider_name=_optional_nonblank(event.payload.get("provider_name")),
                    requested_model=_optional_nonblank(event.payload.get("requested_model")),
                    model=_optional_nonblank(event.payload.get("model")),
                    currency=currency,
                    reason="model.completed event has no token usage metrics",
                )
            )
            continue
        line_items.append(
            _cost_line_item(
                model_step=model_step,
                metrics=metrics,
                pricing=pricing,
                currency=currency,
                catalog=catalog,
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
    pricing: PricingSource | None = None,
    currency: str = "USD",
    catalog: ModelCatalog | None = None,
) -> CausalBudgetCostSummary:
    causal_budget_id = require_clean_nonblank(causal_budget_id, "causal_budget_id")
    copied_session_ids = _copy_string_list(session_ids, "session_ids")
    known_session_ids = set(copied_session_ids)
    filtered_events = [event for event in events if event.session_id in known_session_ids]
    summary = estimate_session_cost(
        session_id=causal_budget_id,
        events=filtered_events,
        pricing=pricing,
        currency=currency,
        catalog=catalog,
    )
    session_costs = tuple(
        estimate_session_cost(
            session_id=session_id,
            events=[event for event in filtered_events if event.session_id == session_id],
            pricing=pricing,
            currency=currency,
            catalog=catalog,
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


def _split_pricing_sources(
    pricing: PricingSource | None,
    catalog: ModelCatalog | None,
) -> tuple[PricingCatalog | None, ModelCatalog | None]:
    if pricing is None and catalog is None:
        raise ValueError("estimate_session_cost requires pricing or catalog.")
    if isinstance(pricing, ModelCatalog):
        if catalog is not None:
            raise ValueError("Pass a ModelCatalog as pricing or catalog, not both.")
        catalog = pricing
        pricing = None
    elif pricing is not None:
        pricing = copy_pricing_catalog(pricing)
    if catalog is not None and type(catalog) is not ModelCatalog:
        raise TypeError("catalog must be a ModelCatalog instance.")
    return pricing, catalog


class _ResolvedPrice(NamedTuple):
    price: ModelPricing
    tier_max_input_tokens: int | None
    provenance: Provenance | None


def pricing_source_price(
    source: PricingSource,
    *,
    provider_name: str | None,
    model: str | None,
    input_tokens: int = 0,
) -> ModelPricing | None:
    """Resolve one pricing source with its matching and context-tier semantics."""
    resolved = _resolve_source_price(
        source,
        provider_name=provider_name,
        model=model,
        input_tokens=input_tokens,
    )
    return None if resolved is None else resolved.price


def _resolve_source_price(
    source: PricingSource,
    *,
    provider_name: str | None,
    model: str | None,
    input_tokens: int,
) -> _ResolvedPrice | None:
    if isinstance(source, ModelCatalog):
        match = source._price_at(
            provider_name=provider_name,
            model=model,
            input_tokens=input_tokens,
        )
        if match is None:
            return None
        price, tier = match
        return _ResolvedPrice(
            price,
            tier.max_input_tokens,
            price.provenance,
        )

    price = source.match_price(provider_name=provider_name, model=model)
    if price is None or price.pricing_tiers is None:
        return None if price is None else _ResolvedPrice(price, None, price.provenance)
    tier = TieredPricing(currency=price.currency, standard=price.pricing_tiers).tier_for(
        input_tokens
    )
    return _ResolvedPrice(
        price.model_copy(
            update={
                "input_per_million": tier.input_per_million,
                "output_per_million": tier.output_per_million,
                "cache_read_input_per_million": tier.cache_read_input_per_million,
                "cache_write_input_per_million": (
                    tier.cache_write_input_per_million
                    if tier.cache_write_input_per_million is not None
                    else price.cache_write_input_per_million
                ),
                "pricing_tiers": None,
            },
            deep=True,
        ),
        tier.max_input_tokens,
        price.provenance,
    )


def _resolve_price(
    *,
    metrics: UsageMetrics,
    pricing: PricingCatalog | None,
    catalog: ModelCatalog | None,
    currency: str,
) -> _ResolvedPrice | None:
    """Resolve the per-step price and the tiered band that produced it.

    An explicit flat price is treated as an application override when both sources match in
    the requested currency; the tier-aware catalog fills models absent from that override.
    Currency-matching results always beat mismatched results. If neither source uses the
    requested currency, the flat override is returned first so the caller reports that
    mismatch. ``tier_max_input_tokens`` is the ceiling of the catalog band used (``None`` for
    the open-ended tier or a flat price).
    """
    candidates: list[_ResolvedPrice] = []
    for source in (pricing, catalog):
        if source is None:
            continue
        resolved = _match_with_requested_fallback(
            metrics,
            lambda model, source=source: _resolve_source_price(
                source,
                provider_name=metrics.provider_name,
                model=model,
                input_tokens=metrics.input_tokens,
            ),
        )
        if resolved is not None:
            candidates.append(resolved)
    if not candidates:
        return None
    for resolved in candidates:
        if resolved.price.currency.upper() == currency.upper():
            return resolved
    return candidates[0]


def _match_with_requested_fallback(
    metrics: UsageMetrics, lookup: Callable[[str | None], _MatchT | None]
) -> _MatchT | None:
    """Look up the provider-resolved model, or the request only when none was reported."""
    result = lookup(metrics.model)
    if result is None and metrics.model is None:
        result = lookup(metrics.requested_model)
    return result


def _cost_line_item(
    *,
    model_step: int,
    metrics: UsageMetrics,
    pricing: PricingCatalog | None,
    currency: str,
    catalog: ModelCatalog | None = None,
) -> CostLineItem:
    resolved = _resolve_price(metrics=metrics, pricing=pricing, catalog=catalog, currency=currency)
    if resolved is None:
        return _unpriced_line_item(
            model_step=model_step,
            provider_name=metrics.provider_name,
            requested_model=metrics.requested_model,
            model=metrics.model,
            currency=currency,
            reason="no matching model pricing",
            metrics=metrics,
        )
    price = resolved.price

    if price.currency.upper() != currency.upper():
        return _unpriced_line_item(
            model_step=model_step,
            provider_name=metrics.provider_name,
            requested_model=metrics.requested_model,
            model=metrics.model,
            currency=currency,
            reason=f"pricing currency {price.currency} does not match requested {currency}",
            metrics=metrics,
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
    cache_write_cost = _token_cost(metrics.cache.write_tokens, cache_write_price)
    total_cost = input_cost + output_cost + cache_read_cost + cache_write_cost

    return CostLineItem(
        model_step=model_step,
        provider_name=metrics.provider_name,
        requested_model=metrics.requested_model,
        model=metrics.model,
        pricing_provider_name=price.provider_name,
        pricing_model=price.model,
        pricing_match=price.match,
        pricing_provenance=resolved.provenance,
        pricing_tier_max_input_tokens=resolved.tier_max_input_tokens,
        priced=True,
        currency=currency.upper(),
        input_tokens=metrics.input_tokens,
        output_tokens=metrics.output_tokens,
        cache_read_input_tokens=metrics.cache.read_tokens,
        cache_write_input_tokens=metrics.cache.write_tokens,
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
) -> CostLineItem:
    return CostLineItem(
        model_step=model_step,
        provider_name=provider_name,
        requested_model=requested_model,
        model=model,
        priced=False,
        currency=currency.upper(),
        input_tokens=0 if metrics is None else metrics.input_tokens,
        output_tokens=0 if metrics is None else metrics.output_tokens,
        cache_read_input_tokens=0 if metrics is None else metrics.cache.read_tokens,
        cache_write_input_tokens=0 if metrics is None else metrics.cache.write_tokens,
        uncached_input_tokens=0 if metrics is None else metrics.cache.uncached_input_tokens,
        input_cost=Decimal("0"),
        output_cost=Decimal("0"),
        cache_read_input_cost=Decimal("0"),
        cache_write_input_cost=Decimal("0"),
        total_cost=Decimal("0"),
        missing_pricing_reason=reason,
    )


def _token_cost(tokens: int, price_per_million: Decimal) -> Decimal:
    return Decimal(tokens) * price_per_million / _TOKENS_PER_MILLION


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
    single matching rule shared by ``PricingCatalog.match_price`` and ``ModelCatalog.resolve``.
    Records are read directly (no throwaway projection list per call).
    """
    provider = _normalize_provider(provider_name) if type(provider_name) is str else None
    model_name = model.strip() if type(model) is str else None
    if provider is None or model_name is None:
        return None

    best: tuple[tuple[int, int], int, _MatchT] | None = None
    for index, record in enumerate(records):
        cand_provider, cand_model, cand_match = key(record)
        if _normalize_provider(cand_provider) != provider:
            continue
        configured_model = cand_model.strip()
        if (cand_match == "exact" and model_name == configured_model) or (
            cand_match == "prefix" and model_name.startswith(configured_model)
        ):
            score = (1 if cand_match == "exact" else 0, len(cand_model))
            if best is None or score > best[0]:
                best = (score, index, record)
    return None if best is None else best[2]


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
