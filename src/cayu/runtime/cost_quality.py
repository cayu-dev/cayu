from __future__ import annotations

import re
from collections import defaultdict
from datetime import date
from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext
from enum import StrEnum
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    json_utf8_size_within_limit,
    require_clean_nonblank,
    revalidate_model_input,
    revalidate_model_inputs,
)
from cayu.runtime.costs import CostLineItem, Provenance

COST_QUALITY_COMPARISON_SCHEMA_VERSION = 1

_MAX_PAIRS = 256
_MAX_ATTEMPTS_PER_SIDE = 2_048
_MAX_QUALITY_REFERENCES = 64
_MAX_TOTAL_ATTEMPTS = 4_096
_MAX_REQUEST_JSON_BYTES = 8 * 1024 * 1024
_MAX_IDENTITY_CHARS = 512
_MAX_REFERENCE_CHARS = 2_048
_MAX_DECIMAL_DIGITS = 64
_MAX_COST_DECIMAL_PLACES = 64
_MAX_COST_VALUE = Decimal(MAX_DURABLE_JSON_INTEGER)
_PERCENT_QUANTUM = Decimal("0.01")
_COMPARISON_DECIMAL_CONTEXT = Context(
    prec=_MAX_DECIMAL_DIGITS + _MAX_COST_DECIMAL_PLACES,
    rounding=ROUND_HALF_EVEN,
)
_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)
_SAFE_REFERENCE_KIND_PATTERN = re.compile(r"[a-z][a-z0-9._-]{0,127}\Z", re.ASCII)
_SHA256_CONTENT_IDENTIFIER_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z", re.ASCII)


class ComparisonPricingProvenance(BaseModel):
    """Bounded, allowlisted pricing source retained in comparison reports."""

    model_config = _MODEL_CONFIG

    source: str = Field(max_length=_MAX_REFERENCE_CHARS)
    url: str = Field(max_length=_MAX_REFERENCE_CHARS)
    as_of: str = Field(max_length=_MAX_IDENTITY_CHARS)

    @classmethod
    def from_provenance(cls, provenance: Provenance) -> ComparisonPricingProvenance:
        if type(provenance) is not Provenance:
            raise TypeError("provenance must be a Provenance.")
        return cls(source=provenance.source, url=provenance.url, as_of=provenance.as_of)

    @field_validator("source", "url", "as_of")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)


class ComparisonCostLineItem(BaseModel):
    """Bounded, safe accounting projection of one runtime cost line item."""

    model_config = _MODEL_CONFIG

    model_step: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    provider_name: str | None = Field(default=None, max_length=_MAX_IDENTITY_CHARS)
    requested_model: str | None = Field(default=None, max_length=_MAX_IDENTITY_CHARS)
    model: str | None = Field(default=None, max_length=_MAX_IDENTITY_CHARS)
    pricing_provider_name: str | None = Field(default=None, max_length=_MAX_IDENTITY_CHARS)
    pricing_model: str | None = Field(default=None, max_length=_MAX_IDENTITY_CHARS)
    pricing_match: Literal["exact", "prefix", "resource_mapping"] | None = None
    pricing_provenance: ComparisonPricingProvenance | None = None
    pricing_effective_from: date | None = None
    pricing_effective_through: date | None = None
    pricing_tier_max_input_tokens: StrictInt | None = Field(
        default=None,
        gt=0,
        le=MAX_DURABLE_JSON_INTEGER,
    )
    priced: StrictBool
    currency: str = Field(max_length=_MAX_IDENTITY_CHARS)
    input_tokens: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    output_tokens: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    cache_read_input_tokens: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    cache_write_input_tokens: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    cache_write_5m_input_tokens: StrictInt = Field(
        default=0,
        ge=0,
        le=MAX_DURABLE_JSON_INTEGER,
    )
    cache_write_1h_input_tokens: StrictInt = Field(
        default=0,
        ge=0,
        le=MAX_DURABLE_JSON_INTEGER,
    )
    cache_write_unknown_ttl_input_tokens: StrictInt = Field(
        default=0,
        ge=0,
        le=MAX_DURABLE_JSON_INTEGER,
    )
    uncached_input_tokens: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    input_cost: Decimal = Field(ge=0, le=_MAX_COST_VALUE)
    output_cost: Decimal = Field(ge=0, le=_MAX_COST_VALUE)
    cache_read_input_cost: Decimal = Field(ge=0, le=_MAX_COST_VALUE)
    cache_write_input_cost: Decimal = Field(ge=0, le=_MAX_COST_VALUE)
    total_cost: Decimal = Field(ge=0, le=_MAX_COST_VALUE)
    missing_pricing_reason: str | None = Field(default=None, max_length=_MAX_REFERENCE_CHARS)

    @classmethod
    def from_cost_line_item(cls, item: CostLineItem) -> ComparisonCostLineItem:
        if type(item) is not CostLineItem:
            raise TypeError("item must be a CostLineItem.")
        return cls(
            model_step=item.model_step,
            provider_name=item.provider_name,
            requested_model=item.requested_model,
            model=item.model,
            pricing_provider_name=item.pricing_provider_name,
            pricing_model=item.pricing_model,
            pricing_match=item.pricing_match,
            pricing_provenance=(
                None
                if item.pricing_provenance is None
                else ComparisonPricingProvenance.from_provenance(item.pricing_provenance)
            ),
            pricing_effective_from=item.pricing_effective_from,
            pricing_effective_through=item.pricing_effective_through,
            pricing_tier_max_input_tokens=item.pricing_tier_max_input_tokens,
            priced=item.priced,
            currency=item.currency,
            input_tokens=item.input_tokens,
            output_tokens=item.output_tokens,
            cache_read_input_tokens=item.cache_read_input_tokens,
            cache_write_input_tokens=item.cache_write_input_tokens,
            cache_write_5m_input_tokens=item.cache_write_5m_input_tokens,
            cache_write_1h_input_tokens=item.cache_write_1h_input_tokens,
            cache_write_unknown_ttl_input_tokens=item.cache_write_unknown_ttl_input_tokens,
            uncached_input_tokens=item.uncached_input_tokens,
            input_cost=item.input_cost,
            output_cost=item.output_cost,
            cache_read_input_cost=item.cache_read_input_cost,
            cache_write_input_cost=item.cache_write_input_cost,
            total_cost=item.total_cost,
            missing_pricing_reason=item.missing_pricing_reason,
        )

    @field_validator("pricing_provenance", mode="before")
    @classmethod
    def copy_provenance(cls, value: object) -> object:
        if type(value) is Provenance:
            return ComparisonPricingProvenance.from_provenance(value)
        return revalidate_model_input(value, ComparisonPricingProvenance)

    @field_validator(
        "provider_name",
        "requested_model",
        "model",
        "pricing_provider_name",
        "pricing_model",
        "missing_pricing_reason",
    )
    @classmethod
    def validate_optional_text(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, value: str) -> str:
        return require_clean_nonblank(value, "currency")

    @field_validator(
        "input_cost",
        "output_cost",
        "cache_read_input_cost",
        "cache_write_input_cost",
        "total_cost",
    )
    @classmethod
    def validate_cost(cls, value: Decimal, info) -> Decimal:
        if not value.is_finite():
            raise ValueError(f"{info.field_name} must be finite.")
        if len(value.as_tuple().digits) > _MAX_DECIMAL_DIGITS:
            raise ValueError(
                f"{info.field_name} must use at most {_MAX_DECIMAL_DIGITS} decimal digits."
            )
        exponent = value.as_tuple().exponent
        if not isinstance(exponent, int):
            raise ValueError(f"{info.field_name} must be finite.")
        if exponent < -_MAX_COST_DECIMAL_PLACES:
            raise ValueError(
                f"{info.field_name} must use at most {_MAX_COST_DECIMAL_PLACES} decimal places."
            )
        return value


class CostQualityComparisonStatus(StrEnum):
    """Strength of the cost-and-quality claim supported by retained evidence."""

    VERIFIED = "verified"
    MEASURED_UNMATCHED = "measured_unmatched"
    UNPRICED = "unpriced"
    UNAVAILABLE = "unavailable"


class QualityEvidenceStatus(StrEnum):
    """Outcome of one application-owned quality gate."""

    PASSED = "passed"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


class CostQualityAttemptOperation(StrEnum):
    """Why one provider attempt was made."""

    AGENT_STEP = "agent_step"
    COMPACTION = "compaction"
    STRUCTURED_OUTPUT_REPAIR = "structured_output_repair"
    EVALUATION = "evaluation"
    REPAIR = "repair"
    COMPARISON_CONTROL = "comparison_control"


class CostQualityFindingCode(StrEnum):
    """Stable reason a pair or aggregate cannot make a stronger claim."""

    MISSING_SIDE = "missing_side"
    WORKLOAD_MISMATCH = "workload_mismatch"
    TASK_MISMATCH = "task_mismatch"
    SOURCE_MISMATCH = "source_mismatch"
    SOURCE_CHECKPOINT_MISMATCH = "source_checkpoint_mismatch"
    ROLE_MISMATCH = "role_mismatch"
    OUTPUT_BUDGET_MISMATCH = "output_budget_mismatch"
    GENERATION_SETTINGS_MISMATCH = "generation_settings_mismatch"
    MISSING_ATTEMPTS = "missing_attempts"
    DUPLICATE_ATTEMPT = "duplicate_attempt"
    ATTEMPT_SEQUENCE_INVALID = "attempt_sequence_invalid"
    MISSING_ATTEMPT_IDENTITY = "missing_attempt_identity"
    MISSING_USAGE = "missing_usage"
    CONTRADICTORY_USAGE = "contradictory_usage"
    MISSING_PRICING_PROVENANCE = "missing_pricing_provenance"
    PRICING_CATALOG_MISMATCH = "pricing_catalog_mismatch"
    PRICING_PROVENANCE_MISMATCH = "pricing_provenance_mismatch"
    MIXED_CURRENCY = "mixed_currency"
    UNPRICED_ATTEMPT = "unpriced_attempt"
    QUALITY_EVIDENCE_MISSING = "quality_evidence_missing"
    QUALITY_CONTRACT_MISMATCH = "quality_contract_mismatch"
    QUALITY_EVIDENCE_INVALID = "quality_evidence_invalid"
    QUALITY_GATE_FAILED = "quality_gate_failed"
    QUALITY_GATE_UNAVAILABLE = "quality_gate_unavailable"
    AGGREGATE_COHORT_MISMATCH = "aggregate_cohort_mismatch"


class SavingsPercentageState(StrEnum):
    """Why a savings percentage is present or absent."""

    COMPUTED = "computed"
    ZERO_BASELINE = "zero_baseline"
    UNAVAILABLE = "unavailable"


class CostDirection(StrEnum):
    """Observed direction of candidate cost relative to baseline cost."""

    SAVINGS = "savings"
    BREAK_EVEN = "break_even"
    INCREASED_COST = "increased_cost"
    UNAVAILABLE = "unavailable"


class QualityEvidenceReference(BaseModel):
    """Opaque, non-secret identifier for application-owned quality evidence."""

    model_config = _MODEL_CONFIG

    kind: str = Field(max_length=128)
    reference: str = Field(max_length=71)

    @field_validator("kind")
    @classmethod
    def validate_kind(cls, value: str, info) -> str:
        value = require_clean_nonblank(value, info.field_name)
        if _SAFE_REFERENCE_KIND_PATTERN.fullmatch(value) is None:
            raise ValueError(
                f"{info.field_name} must start with a lowercase ASCII letter and contain only "
                "lowercase ASCII letters, digits, '.', '_', or '-'."
            )
        return value

    @field_validator("reference")
    @classmethod
    def validate_reference(cls, value: str) -> str:
        value = require_clean_nonblank(value, "reference")
        if _SHA256_CONTENT_IDENTIFIER_PATTERN.fullmatch(value) is None:
            raise ValueError("reference must be a sha256 content identifier.")
        return value


class PairedQualityEvidence(BaseModel):
    """Application-owned result from one declared quality contract."""

    model_config = _MODEL_CONFIG

    contract_name: str = Field(max_length=_MAX_IDENTITY_CHARS)
    contract_version: str = Field(max_length=_MAX_IDENTITY_CHARS)
    threshold: Decimal = Field(ge=0, le=1)
    role: str = Field(max_length=_MAX_IDENTITY_CHARS)
    status: QualityEvidenceStatus
    score: Decimal | None = Field(default=None, ge=0, le=1)
    references: tuple[QualityEvidenceReference, ...] = Field(
        default_factory=tuple,
        max_length=_MAX_QUALITY_REFERENCES,
    )

    @field_validator("references")
    @classmethod
    def order_references(
        cls,
        value: tuple[QualityEvidenceReference, ...],
    ) -> tuple[QualityEvidenceReference, ...]:
        return tuple(sorted(value, key=lambda item: (item.kind, item.reference)))

    @field_validator("contract_name", "contract_version", "role")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("threshold", "score")
    @classmethod
    def validate_decimal(cls, value: Decimal | None, info) -> Decimal | None:
        if value is not None and not value.is_finite():
            raise ValueError(f"{info.field_name} must be finite.")
        return value


class ComparableOutputBudget(BaseModel):
    """The final-output envelope that must match across a verified pair."""

    model_config = _MODEL_CONFIG

    output_mode: Literal["text", "structured"] = "text"
    max_output_tokens: StrictInt = Field(gt=0, le=MAX_DURABLE_JSON_INTEGER)


class ComparableGenerationSettings(BaseModel):
    """Opaque identity of the output-affecting settings declared for one side."""

    model_config = _MODEL_CONFIG

    revision: str = Field(max_length=71)

    @field_validator("revision")
    @classmethod
    def validate_revision(cls, value: str) -> str:
        value = require_clean_nonblank(value, "revision")
        if _SHA256_CONTENT_IDENTIFIER_PATTERN.fullmatch(value) is None:
            raise ValueError("revision must be a sha256 content identifier.")
        return value


class ComparisonPricingCatalog(BaseModel):
    """Identity of the application-selected price book used for one side."""

    model_config = _MODEL_CONFIG

    price_book_version: str = Field(max_length=_MAX_IDENTITY_CHARS)
    generated_at: str = Field(max_length=_MAX_IDENTITY_CHARS)

    @field_validator("price_book_version", "generated_at")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)


class PairedCostAttempt(BaseModel):
    """One attributable provider attempt and its provider-reported cost evidence."""

    model_config = _MODEL_CONFIG

    attempt_id: str = Field(max_length=_MAX_IDENTITY_CHARS)
    operation_id: str = Field(max_length=_MAX_IDENTITY_CHARS)
    session_id: str = Field(max_length=_MAX_IDENTITY_CHARS)
    branch_id: str | None = Field(default=None, max_length=_MAX_IDENTITY_CHARS)
    role: str = Field(max_length=_MAX_IDENTITY_CHARS)
    operation: CostQualityAttemptOperation
    attempt_ordinal: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    provider_name: str | None = Field(default=None, max_length=_MAX_IDENTITY_CHARS)
    model: str | None = Field(default=None, max_length=_MAX_IDENTITY_CHARS)
    source_checkpoint_id: str | None = Field(default=None, max_length=_MAX_IDENTITY_CHARS)
    cost: ComparisonCostLineItem | None = None
    usage_unavailable_reason: str | None = Field(default=None, max_length=_MAX_REFERENCE_CHARS)

    @field_validator("cost", mode="before")
    @classmethod
    def copy_cost(cls, value: object) -> object:
        if isinstance(value, CostLineItem):
            return ComparisonCostLineItem.from_cost_line_item(value)
        if type(value) is dict and "billing_identity" in value:
            return ComparisonCostLineItem.from_cost_line_item(CostLineItem.model_validate(value))
        return revalidate_model_input(value, ComparisonCostLineItem)

    @field_validator(
        "attempt_id",
        "operation_id",
        "session_id",
        "role",
    )
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator(
        "branch_id",
        "provider_name",
        "model",
        "source_checkpoint_id",
        "usage_unavailable_reason",
    )
    @classmethod
    def validate_optional_text(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)


class PairedCostQualitySide(BaseModel):
    """One baseline or candidate strategy under a matched comparison contract."""

    model_config = _MODEL_CONFIG

    strategy_id: str = Field(max_length=_MAX_IDENTITY_CHARS)
    workload_id: str = Field(max_length=_MAX_IDENTITY_CHARS)
    task_id: str = Field(max_length=_MAX_IDENTITY_CHARS)
    source_id: str = Field(max_length=_MAX_IDENTITY_CHARS)
    role: str = Field(max_length=_MAX_IDENTITY_CHARS)
    output_budget: ComparableOutputBudget
    generation_settings: ComparableGenerationSettings
    pricing_catalog: ComparisonPricingCatalog | None = None
    quality: PairedQualityEvidence | None = None
    attempts: tuple[PairedCostAttempt, ...] = Field(
        default_factory=tuple,
        max_length=_MAX_ATTEMPTS_PER_SIDE,
    )

    @field_validator(
        "output_budget",
        "generation_settings",
        "pricing_catalog",
        "quality",
        mode="before",
    )
    @classmethod
    def copy_nested_contracts(cls, value: object) -> object:
        return revalidate_model_input(
            value,
            ComparableOutputBudget,
            ComparableGenerationSettings,
            ComparisonPricingCatalog,
            PairedQualityEvidence,
        )

    @field_validator("attempts", mode="before")
    @classmethod
    def copy_attempts(cls, value: object) -> object:
        return revalidate_model_inputs(value, PairedCostAttempt)

    @field_validator("strategy_id", "workload_id", "task_id", "source_id", "role")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)


class PairedCostQualityPair(BaseModel):
    """A stable pair identity with optional sides so absence stays explicit."""

    model_config = _MODEL_CONFIG

    pair_id: str = Field(max_length=_MAX_IDENTITY_CHARS)
    baseline: PairedCostQualitySide | None = None
    candidate: PairedCostQualitySide | None = None

    @field_validator("baseline", "candidate", mode="before")
    @classmethod
    def copy_sides(cls, value: object) -> object:
        return revalidate_model_input(value, PairedCostQualitySide)

    @field_validator("pair_id")
    @classmethod
    def validate_pair_id(cls, value: str) -> str:
        return require_clean_nonblank(value, "pair_id")


class PairedCostQualityComparisonRequest(BaseModel):
    """Bounded input to the canonical paired cost-and-quality comparison."""

    model_config = _MODEL_CONFIG

    pairs: tuple[PairedCostQualityPair, ...] = Field(min_length=1, max_length=_MAX_PAIRS)

    @field_validator("pairs")
    @classmethod
    def order_pairs(
        cls,
        value: tuple[PairedCostQualityPair, ...],
    ) -> tuple[PairedCostQualityPair, ...]:
        pair_ids = [pair.pair_id for pair in value]
        if len(pair_ids) != len(set(pair_ids)):
            raise ValueError("pairs must use distinct pair_id values.")
        return tuple(sorted(value, key=lambda pair: pair.pair_id))

    @model_validator(mode="after")
    def validate_request_work(self) -> PairedCostQualityComparisonRequest:
        attempt_count = sum(
            len(side.attempts)
            for pair in self.pairs
            for side in (pair.baseline, pair.candidate)
            if side is not None
        )
        if attempt_count > _MAX_TOTAL_ATTEMPTS:
            raise ValueError(f"comparison request exceeds {_MAX_TOTAL_ATTEMPTS} total attempts.")
        if not json_utf8_size_within_limit(self, _MAX_REQUEST_JSON_BYTES):
            raise ValueError(
                f"comparison request exceeds {_MAX_REQUEST_JSON_BYTES} canonical JSON bytes."
            )
        return self


class CostQualityFinding(BaseModel):
    """One typed, bounded explanation of proof loss."""

    model_config = _MODEL_CONFIG

    code: CostQualityFindingCode
    field: str = Field(max_length=_MAX_IDENTITY_CHARS)
    detail: str = Field(max_length=_MAX_REFERENCE_CHARS)

    @field_validator("field", "detail")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)


class CostCurrencyTotal(BaseModel):
    """Priced subtotal in one currency; currencies are never combined."""

    model_config = _MODEL_CONFIG

    currency: str = Field(max_length=_MAX_IDENTITY_CHARS)
    priced_cost: Decimal = Field(ge=0)

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, value: str) -> str:
        return require_clean_nonblank(value, "currency").upper()

    @field_validator("priced_cost")
    @classmethod
    def validate_cost(cls, value: Decimal) -> Decimal:
        if not value.is_finite():
            raise ValueError("priced_cost must be finite.")
        return value


class CostAccountingTotals(BaseModel):
    """Recomputed usage and priced subtotals for one evidence scope."""

    model_config = _MODEL_CONFIG

    attempt_count: StrictInt = Field(ge=0)
    first_attempt_count: StrictInt = Field(ge=0)
    retry_attempt_count: StrictInt = Field(ge=0)
    priced_attempt_count: StrictInt = Field(ge=0)
    unpriced_attempt_count: StrictInt = Field(ge=0)
    missing_usage_attempt_count: StrictInt = Field(ge=0)
    input_tokens: StrictInt = Field(ge=0)
    output_tokens: StrictInt = Field(ge=0)
    cache_read_input_tokens: StrictInt = Field(ge=0)
    cache_write_input_tokens: StrictInt = Field(ge=0)
    uncached_input_tokens: StrictInt = Field(ge=0)
    currencies: tuple[CostCurrencyTotal, ...] = ()

    @field_validator("currencies", mode="before")
    @classmethod
    def copy_currencies(cls, value: object) -> object:
        return revalidate_model_inputs(value, CostCurrencyTotal)


class CostOperationTotals(BaseModel):
    model_config = _MODEL_CONFIG

    operation: CostQualityAttemptOperation
    totals: CostAccountingTotals

    @field_validator("totals", mode="before")
    @classmethod
    def copy_totals(cls, value: object) -> object:
        return revalidate_model_input(value, CostAccountingTotals)


class CostSessionTotals(BaseModel):
    model_config = _MODEL_CONFIG

    session_id: str
    totals: CostAccountingTotals

    @field_validator("totals", mode="before")
    @classmethod
    def copy_totals(cls, value: object) -> object:
        return revalidate_model_input(value, CostAccountingTotals)


class CostBranchTotals(BaseModel):
    model_config = _MODEL_CONFIG

    branch_id: str
    totals: CostAccountingTotals

    @field_validator("totals", mode="before")
    @classmethod
    def copy_totals(cls, value: object) -> object:
        return revalidate_model_input(value, CostAccountingTotals)


class PairedCostQualitySideReport(BaseModel):
    """Safe, deterministic accounting projection for one supplied side."""

    model_config = _MODEL_CONFIG

    strategy_id: str
    workload_id: str
    task_id: str
    source_id: str
    role: str
    output_budget: ComparableOutputBudget
    generation_settings: ComparableGenerationSettings
    pricing_catalog: ComparisonPricingCatalog | None
    pricing_provenance: tuple[ComparisonPricingProvenance, ...]
    quality: PairedQualityEvidence | None
    attempts: tuple[PairedCostAttempt, ...]
    whole_harness: CostAccountingTotals
    operations: tuple[CostOperationTotals, ...]
    sessions: tuple[CostSessionTotals, ...]
    branches: tuple[CostBranchTotals, ...]

    @field_validator(
        "output_budget",
        "generation_settings",
        "pricing_catalog",
        "quality",
        "whole_harness",
        mode="before",
    )
    @classmethod
    def copy_nested_contracts(cls, value: object) -> object:
        return revalidate_model_input(
            value,
            ComparableOutputBudget,
            ComparableGenerationSettings,
            ComparisonPricingCatalog,
            PairedQualityEvidence,
            CostAccountingTotals,
        )

    @field_validator("pricing_provenance", mode="before")
    @classmethod
    def copy_provenance(cls, value: object) -> object:
        return revalidate_model_inputs(value, ComparisonPricingProvenance)

    @field_validator("attempts", mode="before")
    @classmethod
    def copy_attempts(cls, value: object) -> object:
        return revalidate_model_inputs(value, PairedCostAttempt)

    @field_validator("operations", "sessions", "branches", mode="before")
    @classmethod
    def copy_breakdowns(cls, value: object, info) -> object:
        model = {
            "operations": CostOperationTotals,
            "sessions": CostSessionTotals,
            "branches": CostBranchTotals,
        }[info.field_name]
        return revalidate_model_inputs(value, model)


class PairedCostQualityPairReport(BaseModel):
    """One pair's raw accounting and strongest supportable claim."""

    model_config = _MODEL_CONFIG

    pair_id: str
    status: CostQualityComparisonStatus
    findings: tuple[CostQualityFinding, ...]
    baseline: PairedCostQualitySideReport | None
    candidate: PairedCostQualitySideReport | None
    eligible_for_verified_aggregate: bool
    currency: str | None
    baseline_cost: Decimal | None
    candidate_cost: Decimal | None
    savings: Decimal | None
    savings_percentage: Decimal | None
    savings_percentage_state: SavingsPercentageState
    cost_direction: CostDirection

    @field_validator("findings", mode="before")
    @classmethod
    def copy_findings(cls, value: object) -> object:
        return revalidate_model_inputs(value, CostQualityFinding)

    @field_validator("baseline", "candidate", mode="before")
    @classmethod
    def copy_sides(cls, value: object) -> object:
        return revalidate_model_input(value, PairedCostQualitySideReport)


class CostQualityPairExclusion(BaseModel):
    """Why one pair does not contribute to the verified aggregate percentage."""

    model_config = _MODEL_CONFIG

    pair_id: str
    status: CostQualityComparisonStatus
    findings: tuple[CostQualityFinding, ...]

    @field_validator("findings", mode="before")
    @classmethod
    def copy_findings(cls, value: object) -> object:
        return revalidate_model_inputs(value, CostQualityFinding)


class CostQualityAggregateReport(BaseModel):
    """Verified-pair arithmetic plus every typed exclusion."""

    model_config = _MODEL_CONFIG

    status: CostQualityComparisonStatus
    pair_count: StrictInt = Field(ge=1)
    eligible_pair_ids: tuple[str, ...]
    exclusions: tuple[CostQualityPairExclusion, ...]
    findings: tuple[CostQualityFinding, ...]
    currency: str | None
    baseline_cost: Decimal | None
    candidate_cost: Decimal | None
    savings: Decimal | None
    savings_percentage: Decimal | None
    savings_percentage_state: SavingsPercentageState
    cost_direction: CostDirection

    @field_validator("exclusions", mode="before")
    @classmethod
    def copy_exclusions(cls, value: object) -> object:
        return revalidate_model_inputs(value, CostQualityPairExclusion)

    @field_validator("findings", mode="before")
    @classmethod
    def copy_findings(cls, value: object) -> object:
        return revalidate_model_inputs(value, CostQualityFinding)


class PairedCostQualityComparisonReport(BaseModel):
    """Versioned, serialization-safe result of one pure comparison."""

    model_config = _MODEL_CONFIG

    schema_version: Literal[1] = COST_QUALITY_COMPARISON_SCHEMA_VERSION
    status: CostQualityComparisonStatus
    pairs: tuple[PairedCostQualityPairReport, ...]
    aggregate: CostQualityAggregateReport

    @field_validator("pairs", mode="before")
    @classmethod
    def copy_pairs(cls, value: object) -> object:
        return revalidate_model_inputs(value, PairedCostQualityPairReport)

    @field_validator("aggregate", mode="before")
    @classmethod
    def copy_aggregate(cls, value: object) -> object:
        return revalidate_model_input(value, CostQualityAggregateReport)


def compare_paired_cost_quality(
    request: PairedCostQualityComparisonRequest,
) -> PairedCostQualityComparisonReport:
    """Return one deterministic report from bounded caller-owned evidence."""

    if type(request) is not PairedCostQualityComparisonRequest:
        raise TypeError("request must be a PairedCostQualityComparisonRequest.")
    with localcontext(_COMPARISON_DECIMAL_CONTEXT):
        copied = request.model_copy(deep=True)
        pairs = tuple(_compare_pair(pair) for pair in copied.pairs)
        aggregate = _aggregate_pairs(pairs)
        return PairedCostQualityComparisonReport(
            status=aggregate.status,
            pairs=pairs,
            aggregate=aggregate,
        )


def _compare_pair(pair: PairedCostQualityPair) -> PairedCostQualityPairReport:
    findings: list[CostQualityFinding] = []
    if pair.baseline is None:
        findings.append(_finding(CostQualityFindingCode.MISSING_SIDE, "baseline"))
    if pair.candidate is None:
        findings.append(_finding(CostQualityFindingCode.MISSING_SIDE, "candidate"))

    baseline_report = _side_report(pair.baseline)
    candidate_report = _side_report(pair.candidate)
    unpriced = False
    unpriced_findings: list[CostQualityFinding] = []
    quality_findings: list[CostQualityFinding] = []
    if pair.baseline is not None:
        side_findings, side_unpriced = _side_findings(pair.baseline, "baseline")
        findings.extend(side_findings)
        unpriced = unpriced or side_unpriced
        if side_unpriced:
            unpriced_findings.append(
                _finding(CostQualityFindingCode.UNPRICED_ATTEMPT, "baseline.attempts")
            )
    if pair.candidate is not None:
        side_findings, side_unpriced = _side_findings(pair.candidate, "candidate")
        findings.extend(side_findings)
        unpriced = unpriced or side_unpriced
        if side_unpriced:
            unpriced_findings.append(
                _finding(CostQualityFindingCode.UNPRICED_ATTEMPT, "candidate.attempts")
            )

    if pair.baseline is not None and pair.candidate is not None:
        findings.extend(_pair_identity_findings(pair.baseline, pair.candidate))
        quality_findings.extend(_pair_quality_findings(pair.baseline, pair.candidate))

    status: CostQualityComparisonStatus
    if findings:
        status = CostQualityComparisonStatus.UNAVAILABLE
    elif unpriced:
        status = CostQualityComparisonStatus.UNPRICED
    elif quality_findings:
        status = CostQualityComparisonStatus.MEASURED_UNMATCHED
    else:
        status = CostQualityComparisonStatus.VERIFIED
    all_findings = tuple([*findings, *unpriced_findings, *quality_findings])

    currency, baseline_cost, candidate_cost = _pair_costs(baseline_report, candidate_report)
    savings, percentage, percentage_state, direction = _savings(
        baseline_cost,
        candidate_cost,
    )
    return PairedCostQualityPairReport(
        pair_id=pair.pair_id,
        status=status,
        findings=all_findings,
        baseline=baseline_report,
        candidate=candidate_report,
        eligible_for_verified_aggregate=status is CostQualityComparisonStatus.VERIFIED,
        currency=currency,
        baseline_cost=baseline_cost,
        candidate_cost=candidate_cost,
        savings=savings,
        savings_percentage=percentage,
        savings_percentage_state=percentage_state,
        cost_direction=direction,
    )


def _pair_identity_findings(
    baseline: PairedCostQualitySide,
    candidate: PairedCostQualitySide,
) -> list[CostQualityFinding]:
    findings: list[CostQualityFinding] = []
    comparisons = (
        ("workload_id", CostQualityFindingCode.WORKLOAD_MISMATCH),
        ("task_id", CostQualityFindingCode.TASK_MISMATCH),
        ("source_id", CostQualityFindingCode.SOURCE_MISMATCH),
        ("role", CostQualityFindingCode.ROLE_MISMATCH),
        ("output_budget", CostQualityFindingCode.OUTPUT_BUDGET_MISMATCH),
        ("generation_settings", CostQualityFindingCode.GENERATION_SETTINGS_MISMATCH),
    )
    for field_name, code in comparisons:
        if getattr(baseline, field_name) != getattr(candidate, field_name):
            findings.append(_finding(code, field_name))
    baseline_checkpoints = {
        attempt.source_checkpoint_id
        for attempt in baseline.attempts
        if attempt.source_checkpoint_id is not None
    }
    candidate_checkpoints = {
        attempt.source_checkpoint_id
        for attempt in candidate.attempts
        if attempt.source_checkpoint_id is not None
    }
    if baseline_checkpoints != candidate_checkpoints:
        findings.append(
            _finding(
                CostQualityFindingCode.SOURCE_CHECKPOINT_MISMATCH,
                "source_checkpoint_id",
            )
        )
    if (
        baseline.pricing_catalog is not None
        and candidate.pricing_catalog is not None
        and baseline.pricing_catalog != candidate.pricing_catalog
    ):
        findings.append(
            _finding(CostQualityFindingCode.PRICING_CATALOG_MISMATCH, "pricing_catalog")
        )
    currencies = {
        item.cost.currency.upper()
        for side in (baseline, candidate)
        for item in side.attempts
        if item.cost is not None
    }
    if len(currencies) > 1:
        findings.append(_finding(CostQualityFindingCode.MIXED_CURRENCY, "currency"))
    pricing_rows: dict[tuple[str, str, str], set[tuple[str, str, str]]] = defaultdict(set)
    for side in (baseline, candidate):
        for attempt in side.attempts:
            cost = attempt.cost
            if (
                cost is None
                or not cost.priced
                or cost.pricing_provider_name is None
                or cost.pricing_model is None
                or cost.pricing_match is None
                or cost.pricing_provenance is None
            ):
                continue
            pricing_rows[(cost.pricing_provider_name, cost.pricing_model, cost.pricing_match)].add(
                (
                    cost.pricing_provenance.model_dump_json(),
                    str(cost.pricing_effective_from),
                    str(cost.pricing_effective_through),
                )
            )
    if any(len(resolutions) > 1 for resolutions in pricing_rows.values()):
        findings.append(
            _finding(
                CostQualityFindingCode.PRICING_PROVENANCE_MISMATCH,
                "pricing_provenance",
            )
        )
    return findings


def _pair_quality_findings(
    baseline: PairedCostQualitySide,
    candidate: PairedCostQualitySide,
) -> list[CostQualityFinding]:
    if baseline.quality is None or candidate.quality is None:
        return [_finding(CostQualityFindingCode.QUALITY_EVIDENCE_MISSING, "quality")]
    baseline_quality = baseline.quality
    candidate_quality = candidate.quality

    def contract(quality: PairedQualityEvidence) -> tuple[str, str, Decimal, str]:
        return (
            quality.contract_name,
            quality.contract_version,
            quality.threshold,
            quality.role,
        )

    findings: list[CostQualityFinding] = []
    if contract(baseline_quality) != contract(candidate_quality):
        findings.append(_finding(CostQualityFindingCode.QUALITY_CONTRACT_MISMATCH, "quality"))
    for field_name, quality, side_role in (
        ("baseline.quality", baseline_quality, baseline.role),
        ("candidate.quality", candidate_quality, candidate.role),
    ):
        if quality.role != side_role or not _quality_evidence_consistent(quality):
            findings.append(_finding(CostQualityFindingCode.QUALITY_EVIDENCE_INVALID, field_name))
        elif quality.status is QualityEvidenceStatus.FAILED:
            findings.append(_finding(CostQualityFindingCode.QUALITY_GATE_FAILED, field_name))
        elif quality.status is QualityEvidenceStatus.UNAVAILABLE:
            findings.append(_finding(CostQualityFindingCode.QUALITY_GATE_UNAVAILABLE, field_name))
    return findings


def _quality_evidence_consistent(quality: PairedQualityEvidence) -> bool:
    if quality.status is QualityEvidenceStatus.UNAVAILABLE:
        return quality.score is None
    if quality.score is None:
        return False
    passed = quality.score >= quality.threshold
    return passed == (quality.status is QualityEvidenceStatus.PASSED)


def _side_findings(
    side: PairedCostQualitySide,
    side_name: str,
) -> tuple[list[CostQualityFinding], bool]:
    findings: list[CostQualityFinding] = []
    if not side.attempts:
        findings.append(_finding(CostQualityFindingCode.MISSING_ATTEMPTS, f"{side_name}.attempts"))
        return findings, False
    attempt_ids = [attempt.attempt_id for attempt in side.attempts]
    if len(attempt_ids) != len(set(attempt_ids)):
        findings.append(_finding(CostQualityFindingCode.DUPLICATE_ATTEMPT, f"{side_name}.attempts"))
    ordinals: dict[str, list[int]] = defaultdict(list)
    for attempt in side.attempts:
        ordinals[attempt.operation_id].append(attempt.attempt_ordinal)
    if any(sorted(values) != list(range(1, len(values) + 1)) for values in ordinals.values()):
        findings.append(
            _finding(
                CostQualityFindingCode.ATTEMPT_SEQUENCE_INVALID,
                f"{side_name}.attempts",
            )
        )

    unpriced = False
    for attempt in side.attempts:
        field_name = f"{side_name}.attempts.{attempt.attempt_id}"
        if attempt.provider_name is None or attempt.model is None:
            findings.append(_finding(CostQualityFindingCode.MISSING_ATTEMPT_IDENTITY, field_name))
        if attempt.cost is None:
            findings.append(_finding(CostQualityFindingCode.MISSING_USAGE, field_name))
            continue
        cost = attempt.cost
        if not _cost_is_consistent(attempt):
            findings.append(_finding(CostQualityFindingCode.CONTRADICTORY_USAGE, field_name))
        if cost.priced:
            if side.pricing_catalog is None or cost.pricing_provenance is None:
                findings.append(
                    _finding(CostQualityFindingCode.MISSING_PRICING_PROVENANCE, field_name)
                )
        else:
            unpriced = True
    return findings, unpriced


def _cost_is_consistent(attempt: PairedCostAttempt) -> bool:
    cost = attempt.cost
    if cost is None:
        return False
    if attempt.usage_unavailable_reason is not None:
        return False
    if (
        cost.provider_name is not None
        and attempt.provider_name is not None
        and cost.provider_name != attempt.provider_name
    ):
        return False
    if cost.model is not None and attempt.model is not None and cost.model != attempt.model:
        return False
    if (
        cost.cache_read_input_tokens + cost.cache_write_input_tokens + cost.uncached_input_tokens
        != cost.input_tokens
    ):
        return False
    cache_write_ttl_tokens = (
        cost.cache_write_5m_input_tokens
        + cost.cache_write_1h_input_tokens
        + cost.cache_write_unknown_ttl_input_tokens
    )
    if cache_write_ttl_tokens and cache_write_ttl_tokens != cost.cache_write_input_tokens:
        return False
    component_cost = (
        cost.input_cost
        + cost.output_cost
        + cost.cache_read_input_cost
        + cost.cache_write_input_cost
    )
    if component_cost != cost.total_cost:
        return False
    if cost.priced:
        return (
            all(
                value is not None
                for value in (
                    cost.pricing_provider_name,
                    cost.pricing_model,
                    cost.pricing_match,
                )
            )
            and cost.missing_pricing_reason is None
        )
    return (
        cost.total_cost == 0
        and cost.input_cost == 0
        and cost.output_cost == 0
        and cost.cache_read_input_cost == 0
        and cost.cache_write_input_cost == 0
        and cost.missing_pricing_reason is not None
    )


def _side_report(side: PairedCostQualitySide | None) -> PairedCostQualitySideReport | None:
    if side is None:
        return None
    attempts = tuple(
        sorted(
            side.attempts,
            key=lambda attempt: (
                attempt.session_id,
                attempt.operation_id,
                attempt.attempt_ordinal,
                attempt.attempt_id,
            ),
        )
    )
    by_operation: dict[CostQualityAttemptOperation, list[PairedCostAttempt]] = defaultdict(list)
    by_session: dict[str, list[PairedCostAttempt]] = defaultdict(list)
    by_branch: dict[str, list[PairedCostAttempt]] = defaultdict(list)
    provenance: dict[str, ComparisonPricingProvenance] = {}
    for attempt in attempts:
        by_operation[attempt.operation].append(attempt)
        by_session[attempt.session_id].append(attempt)
        if attempt.branch_id is not None:
            by_branch[attempt.branch_id].append(attempt)
        if attempt.cost is not None and attempt.cost.pricing_provenance is not None:
            item = attempt.cost.pricing_provenance
            if type(item) is Provenance:
                item = ComparisonPricingProvenance.from_provenance(item)
            provenance[item.model_dump_json()] = item
    return PairedCostQualitySideReport(
        strategy_id=side.strategy_id,
        workload_id=side.workload_id,
        task_id=side.task_id,
        source_id=side.source_id,
        role=side.role,
        output_budget=side.output_budget,
        generation_settings=side.generation_settings,
        pricing_catalog=side.pricing_catalog,
        pricing_provenance=tuple(provenance[key] for key in sorted(provenance)),
        quality=side.quality,
        attempts=attempts,
        whole_harness=_totals(attempts),
        operations=tuple(
            CostOperationTotals(operation=operation, totals=_totals(by_operation[operation]))
            for operation in sorted(by_operation, key=lambda value: value.value)
        ),
        sessions=tuple(
            CostSessionTotals(session_id=session_id, totals=_totals(by_session[session_id]))
            for session_id in sorted(by_session)
        ),
        branches=tuple(
            CostBranchTotals(branch_id=branch_id, totals=_totals(by_branch[branch_id]))
            for branch_id in sorted(by_branch)
        ),
    )


def _totals(attempts) -> CostAccountingTotals:
    selected = tuple(attempts)
    priced_costs: dict[str, Decimal] = defaultdict(lambda: Decimal(0))
    counters = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_write_input_tokens": 0,
        "uncached_input_tokens": 0,
    }
    priced_attempts = 0
    unpriced_attempts = 0
    missing_attempts = 0
    for attempt in selected:
        cost = attempt.cost
        if cost is None:
            missing_attempts += 1
            continue
        for field_name in counters:
            counters[field_name] += getattr(cost, field_name)
        if cost.priced:
            priced_attempts += 1
            priced_costs[cost.currency.upper()] += cost.total_cost
        else:
            unpriced_attempts += 1
    first_attempts = sum(attempt.attempt_ordinal == 1 for attempt in selected)
    return CostAccountingTotals(
        attempt_count=len(selected),
        first_attempt_count=first_attempts,
        retry_attempt_count=len(selected) - first_attempts,
        priced_attempt_count=priced_attempts,
        unpriced_attempt_count=unpriced_attempts,
        missing_usage_attempt_count=missing_attempts,
        **counters,
        currencies=tuple(
            CostCurrencyTotal(currency=currency, priced_cost=priced_costs[currency])
            for currency in sorted(priced_costs)
        ),
    )


def _pair_costs(
    baseline: PairedCostQualitySideReport | None,
    candidate: PairedCostQualitySideReport | None,
) -> tuple[str | None, Decimal | None, Decimal | None]:
    if baseline is None or candidate is None:
        return None, None, None
    baseline_total = _complete_cost(baseline.whole_harness)
    candidate_total = _complete_cost(candidate.whole_harness)
    if baseline_total is None or candidate_total is None:
        return None, None, None
    baseline_currency, baseline_cost = baseline_total
    candidate_currency, candidate_cost = candidate_total
    if baseline_currency != candidate_currency:
        return None, None, None
    return baseline_currency, baseline_cost, candidate_cost


def _complete_cost(totals: CostAccountingTotals) -> tuple[str, Decimal] | None:
    if (
        totals.attempt_count == 0
        or totals.unpriced_attempt_count
        or totals.missing_usage_attempt_count
        or len(totals.currencies) != 1
    ):
        return None
    currency = totals.currencies[0]
    return currency.currency, currency.priced_cost


def _savings(
    baseline_cost: Decimal | None,
    candidate_cost: Decimal | None,
) -> tuple[Decimal | None, Decimal | None, SavingsPercentageState, CostDirection]:
    if baseline_cost is None or candidate_cost is None:
        return None, None, SavingsPercentageState.UNAVAILABLE, CostDirection.UNAVAILABLE
    savings = baseline_cost - candidate_cost
    if savings > 0:
        direction = CostDirection.SAVINGS
    elif savings < 0:
        direction = CostDirection.INCREASED_COST
    else:
        direction = CostDirection.BREAK_EVEN
    if baseline_cost == 0:
        return savings, None, SavingsPercentageState.ZERO_BASELINE, direction
    percentage = (savings * Decimal(100) / baseline_cost).quantize(
        _PERCENT_QUANTUM,
        rounding=ROUND_HALF_EVEN,
    )
    return savings, percentage, SavingsPercentageState.COMPUTED, direction


def _aggregate_pairs(
    pairs: tuple[PairedCostQualityPairReport, ...],
) -> CostQualityAggregateReport:
    status = max((pair.status for pair in pairs), key=_status_rank)
    eligible = [pair for pair in pairs if pair.eligible_for_verified_aggregate]
    aggregate_findings: list[CostQualityFinding] = []
    eligible_currencies = {pair.currency for pair in eligible}
    if None in eligible_currencies or len(eligible_currencies) > 1:
        if eligible:
            aggregate_findings.append(
                _finding(CostQualityFindingCode.MIXED_CURRENCY, "aggregate.currency")
            )
            status = CostQualityComparisonStatus.UNAVAILABLE
        eligible = []
    eligible_cohorts = {_aggregate_cohort(pair) for pair in eligible}
    if len(eligible_cohorts) > 1:
        aggregate_findings.append(
            _finding(
                CostQualityFindingCode.AGGREGATE_COHORT_MISMATCH,
                "aggregate.cohort",
            )
        )
        status = CostQualityComparisonStatus.UNAVAILABLE
        eligible = []
    exclusions = [
        CostQualityPairExclusion(
            pair_id=pair.pair_id,
            status=pair.status,
            findings=pair.findings,
        )
        for pair in pairs
        if pair not in eligible
    ]
    if not eligible:
        currency = None
        baseline_cost = None
        candidate_cost = None
    else:
        currency = eligible[0].currency
        baseline_cost = sum(
            (pair.baseline_cost for pair in eligible if pair.baseline_cost is not None),
            Decimal(0),
        )
        candidate_cost = sum(
            (pair.candidate_cost for pair in eligible if pair.candidate_cost is not None),
            Decimal(0),
        )
    savings, percentage, percentage_state, direction = _savings(
        baseline_cost,
        candidate_cost,
    )
    return CostQualityAggregateReport(
        status=status,
        pair_count=len(pairs),
        eligible_pair_ids=tuple(pair.pair_id for pair in eligible),
        exclusions=tuple(exclusions),
        findings=tuple(aggregate_findings),
        currency=currency,
        baseline_cost=baseline_cost,
        candidate_cost=candidate_cost,
        savings=savings,
        savings_percentage=percentage,
        savings_percentage_state=percentage_state,
        cost_direction=direction,
    )


def _aggregate_cohort(
    pair: PairedCostQualityPairReport,
) -> tuple[
    str,
    str,
    str,
    str,
    str,
    tuple[str, str],
    tuple[str, str, Decimal, str],
]:
    side = pair.baseline
    if side is None or side.quality is None or side.pricing_catalog is None:
        raise RuntimeError(
            "Verified aggregate pairs must retain baseline quality and pricing evidence."
        )
    quality = side.quality
    catalog = side.pricing_catalog
    return (
        side.workload_id,
        side.task_id,
        side.role,
        side.output_budget.model_dump_json(),
        side.generation_settings.model_dump_json(),
        (catalog.price_book_version, catalog.generated_at),
        (
            quality.contract_name,
            quality.contract_version,
            quality.threshold,
            quality.role,
        ),
    )


def _status_rank(status: CostQualityComparisonStatus) -> int:
    return {
        CostQualityComparisonStatus.VERIFIED: 0,
        CostQualityComparisonStatus.MEASURED_UNMATCHED: 1,
        CostQualityComparisonStatus.UNPRICED: 2,
        CostQualityComparisonStatus.UNAVAILABLE: 3,
    }[status]


def _finding(
    code: CostQualityFindingCode,
    field: str,
) -> CostQualityFinding:
    return CostQualityFinding(
        code=code,
        field=field,
        detail={
            CostQualityFindingCode.MISSING_SIDE: "A required comparison side is missing.",
            CostQualityFindingCode.WORKLOAD_MISMATCH: "The pair uses different workloads.",
            CostQualityFindingCode.TASK_MISMATCH: "The pair uses different tasks.",
            CostQualityFindingCode.SOURCE_MISMATCH: "The pair uses different source evidence.",
            CostQualityFindingCode.SOURCE_CHECKPOINT_MISMATCH: (
                "The pair uses different available source-checkpoint evidence."
            ),
            CostQualityFindingCode.ROLE_MISMATCH: "The pair uses different comparison roles.",
            CostQualityFindingCode.OUTPUT_BUDGET_MISMATCH: (
                "The pair uses different final-output budgets."
            ),
            CostQualityFindingCode.GENERATION_SETTINGS_MISMATCH: (
                "The pair uses different declared generation settings."
            ),
            CostQualityFindingCode.MISSING_ATTEMPTS: "Required provider attempts are absent.",
            CostQualityFindingCode.DUPLICATE_ATTEMPT: "Attempt identities are not unique.",
            CostQualityFindingCode.ATTEMPT_SEQUENCE_INVALID: (
                "Provider attempt ordinals are not a complete sequence."
            ),
            CostQualityFindingCode.MISSING_ATTEMPT_IDENTITY: (
                "A required attempt lacks provider or model identity."
            ),
            CostQualityFindingCode.MISSING_USAGE: (
                "A required attempt has no provider-reported usage."
            ),
            CostQualityFindingCode.CONTRADICTORY_USAGE: (
                "Attempt usage, identity, or cost components contradict each other."
            ),
            CostQualityFindingCode.MISSING_PRICING_PROVENANCE: (
                "Priced evidence lacks its catalog or pricing provenance."
            ),
            CostQualityFindingCode.PRICING_CATALOG_MISMATCH: (
                "The pair uses incompatible pricing catalogs."
            ),
            CostQualityFindingCode.PRICING_PROVENANCE_MISMATCH: (
                "Equivalent pricing rows resolved incompatible schedules or provenance."
            ),
            CostQualityFindingCode.MIXED_CURRENCY: (
                "The selected evidence cannot be combined across currencies."
            ),
            CostQualityFindingCode.UNPRICED_ATTEMPT: (
                "At least one required attempt has usage but no matching price."
            ),
            CostQualityFindingCode.QUALITY_EVIDENCE_MISSING: (
                "A required quality result is missing."
            ),
            CostQualityFindingCode.QUALITY_CONTRACT_MISMATCH: (
                "The pair uses different quality contracts or thresholds."
            ),
            CostQualityFindingCode.QUALITY_EVIDENCE_INVALID: (
                "A quality outcome contradicts its role, score, or threshold."
            ),
            CostQualityFindingCode.QUALITY_GATE_FAILED: "The declared quality gate failed.",
            CostQualityFindingCode.QUALITY_GATE_UNAVAILABLE: (
                "The declared quality gate was unavailable."
            ),
            CostQualityFindingCode.AGGREGATE_COHORT_MISMATCH: (
                "Verified pairs belong to different workload, quality, or pricing cohorts."
            ),
        }[code],
    )
