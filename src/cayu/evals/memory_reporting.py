"""Deterministic paired repeated-trial reports for fixed memory interventions."""

from __future__ import annotations

import html
import json
from decimal import Decimal
from enum import StrEnum
from functools import partial
from hashlib import sha256
from statistics import median
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    SkipValidation,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationInfo,
    field_validator,
    model_validator,
)

from cayu._validation import (
    canonical_durable_json_bytes,
    compact_json_utf8_size,
    copy_durable_json_object,
    durable_json_object_from_pairs,
    parse_durable_json_integer_literal,
    reject_nonportable_json_constant,
    require_durable_clean_nonblank,
    require_unicode_scalar_text,
    revalidate_model_input,
)
from cayu.evals.execution import CorpusExecutionResult
from cayu.evals.execution_profiles import (
    EvalExecutionProfileBindingV1,
    EvalExecutionProfileV1,
)
from cayu.evals.memory_attribution import (
    EvalMemoryEvidenceCompleteness,
    eval_memory_attribution_fingerprint,
)
from cayu.evals.published import (
    PublishedEvalTrialResult,
    PublishedModelJudgeDetail,
    PublishedOutcome,
    PublishedStructuredModelJudgeDetail,
)
from cayu.memory_intervention_execution import (
    MemoryInterventionExecutionRecord,
    MemoryInterventionExecutionStatus,
)
from cayu.memory_interventions import (
    MemoryInterventionComparability,
    MemoryInterventionComparabilityStatus,
    MemoryInterventionKind,
    MemoryInterventionSpec,
    MemoryInterventionTrialBinding,
)
from cayu.runtime.cost_quality import (
    CostQualityComparisonStatus,
    PairedCostQualityComparisonReport,
    PairedCostQualityComparisonRequest,
    PairedCostQualityPair,
    PairedCostQualityPairReport,
    PairedCostQualitySide,
    compare_paired_cost_quality,
)
from cayu.runtime.execution_profiles import (
    ExecutionProfileComponentClass,
)
from cayu.runtime.usage import AggregateCount

MEMORY_EXPERIMENT_REPORT_SCHEMA_VERSION = 1
MEMORY_EXPERIMENT_REPORT_MAX_BYTES = 32 << 20
MEMORY_EXPERIMENT_REPORT_MAX_CASES = 1_000
MEMORY_EXPERIMENT_REPORT_MAX_VARIANTS = 128
MEMORY_EXPERIMENT_REPORT_MAX_REPETITIONS = 1_000
MEMORY_EXPERIMENT_REPORT_MAX_ROWS = 100_000
MEMORY_EXPERIMENT_REPORT_MAX_METRICS = 32
MEMORY_EXPERIMENT_REPORT_MAX_PAIRS = 256
MEMORY_EXPERIMENT_REPORT_MAX_HTML_BYTES = 32 << 20

_MODEL_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    hide_input_in_errors=True,
    revalidate_instances="always",
    validate_default=True,
)


def _clean(value: str, field_name: str, *, maximum: int = 256) -> str:
    value = require_durable_clean_nonblank(value, field_name)
    value = require_unicode_scalar_text(value, field_name)
    if len(value) > maximum:
        raise ValueError(f"{field_name} exceeds {maximum} characters.")
    return value


def _fingerprint(value: str, field_name: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be lowercase SHA-256 hex.")
    return value


def _revision(value: str, field_name: str) -> str:
    if (
        len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{field_name} must be a sha256 revision.")
    return value


def _content_revision(value: object, field_name: str) -> str:
    return "sha256:" + sha256(canonical_durable_json_bytes(value, field_name)).hexdigest()


def _signed_aggregate_count_from_json(value: object, info: ValidationInfo) -> object:
    if info.mode != "json":
        return value
    if type(value) is not str or not value:
        raise ValueError("Signed aggregate counters in JSON must be canonical decimal strings.")
    unsigned = value[1:] if value.startswith("-") else value
    if (
        not unsigned
        or not unsigned.isascii()
        or not unsigned.isdigit()
        or (len(unsigned) > 1 and unsigned.startswith("0"))
        or value == "-0"
    ):
        raise ValueError("Signed aggregate counters in JSON must be canonical decimal strings.")
    return int(value)


SignedAggregateCountString = Annotated[str, Field(pattern=r"^(0|[1-9]\d*|-[1-9]\d*)$")]
SignedAggregateCount = Annotated[
    StrictInt,
    BeforeValidator(
        _signed_aggregate_count_from_json,
        json_schema_input_type=SignedAggregateCountString,
    ),
    PlainSerializer(str, return_type=SignedAggregateCountString, when_used="json"),
]


class _ReportModel(BaseModel):
    model_config = _MODEL_CONFIG


class MemoryMetricRole(StrEnum):
    TASK_QUALITY = "task_quality"
    FACTUAL_SUPPORT = "factual_support"
    HALLUCINATION_AVOIDANCE = "hallucination_avoidance"
    SAFETY = "safety"
    PRIVACY = "privacy"
    STALE_EXPOSURE_AVOIDANCE = "stale_exposure_avoidance"
    FALSE_EXPOSURE_AVOIDANCE = "false_exposure_avoidance"
    UNAUTHORIZED_EXPOSURE_AVOIDANCE = "unauthorized_exposure_avoidance"


class MemoryMetricDirection(StrEnum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


_RANKABLE_MEMORY_METRIC_ROLES = frozenset({MemoryMetricRole.TASK_QUALITY})


class MemoryMetricAvailability(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


class MemoryTrialAvailability(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    MISSING = "missing"
    UNMATCHED = "unmatched"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    OUTCOME_UNKNOWN = "outcome_unknown"
    CONFLICTING = "conflicting"
    INDETERMINATE = "indeterminate"


class MemoryPairStatus(StrEnum):
    COMPARABLE = "comparable"
    INCOMPARABLE = "incomparable"
    UNAVAILABLE = "unavailable"


class MemoryVariantDispositionStatus(StrEnum):
    SELECTED = "selected"
    BASELINE_SUPERSEDED = "baseline_superseded"
    ELIGIBLE_NOT_SELECTED = "eligible_not_selected"
    REJECTED = "rejected"
    INCOMPARABLE = "incomparable"
    UNAVAILABLE = "unavailable"
    NOT_BETTER = "not_better_than_baseline"


class MemoryOperationalDimension(StrEnum):
    LATENCY_MS = "latency_ms"
    TOTAL_TOKENS = "total_tokens"
    MEMORY_PREPARATION_DURATION_MS = "memory_preparation_duration_ms"
    MEMORY_CONTEXT_TOKENS = "memory_context_tokens"
    MEMORY_CONTEXT_BYTES = "memory_context_bytes"


class MemoryMetricBinding(_ReportModel):
    role: MemoryMetricRole
    assertion_id: StrictStr = Field(max_length=128)
    assertion_revision: StrictStr = Field(min_length=71, max_length=71)
    direction: MemoryMetricDirection = MemoryMetricDirection.HIGHER_IS_BETTER

    @field_validator("assertion_id")
    @classmethod
    def validate_id(cls, value: str, info) -> str:
        return _clean(value, info.field_name, maximum=128)

    @field_validator("assertion_revision")
    @classmethod
    def validate_assertion_revision(cls, value: str, info) -> str:
        return _revision(value, info.field_name)


class MemoryMetricGate(_ReportModel):
    role: MemoryMetricRole
    minimum: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    maximum: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_threshold(self) -> Self:
        if (self.minimum is None) == (self.maximum is None):
            raise ValueError("A metric gate requires exactly one threshold direction.")
        return self


class MemoryExperimentGatePolicy(_ReportModel):
    required_metric_roles: tuple[MemoryMetricRole, ...] = ()
    metric_gates: tuple[MemoryMetricGate, ...] = ()
    minimum_comparable_pairs: StrictInt = Field(default=1, ge=1)
    require_priced_cost: StrictBool = False
    maximum_candidate_cost: Decimal | None = Field(default=None, ge=Decimal(0))
    cost_currency: StrictStr | None = Field(default=None, min_length=3, max_length=16)

    @field_validator("required_metric_roles", mode="before")
    @classmethod
    def order_required_roles(cls, value: object) -> object:
        if not isinstance(value, list | tuple):
            raise TypeError("required_metric_roles must be an ordered array.")
        roles = tuple(MemoryMetricRole(item) for item in value)
        if roles != tuple(sorted(set(roles), key=str)):
            raise ValueError("required_metric_roles must be unique and sorted.")
        return roles

    @field_validator("metric_gates", mode="before")
    @classmethod
    def order_metric_gates(cls, value: object) -> object:
        if not isinstance(value, list | tuple):
            raise TypeError("metric_gates must be an ordered array.")
        gates = tuple(MemoryMetricGate.model_validate(item) for item in value)
        if tuple(gate.role for gate in gates) != tuple(
            sorted({gate.role for gate in gates}, key=str)
        ):
            raise ValueError("metric_gates must have unique, sorted roles.")
        return gates

    @model_validator(mode="after")
    def validate_cost_gate(self) -> Self:
        if (self.maximum_candidate_cost is None) != (self.cost_currency is None):
            raise ValueError("A cost ceiling requires exactly one currency.")
        return self


class MemoryRankingTerm(_ReportModel):
    role: MemoryMetricRole
    direction: MemoryMetricDirection


def _validate_metric_policy(
    metric_bindings: tuple[MemoryMetricBinding, ...],
    ranking: tuple[MemoryRankingTerm, ...],
    gates: MemoryExperimentGatePolicy,
) -> None:
    roles = tuple(item.role for item in metric_bindings)
    if roles != tuple(sorted(set(roles), key=str)):
        raise ValueError("Metric bindings must have unique, sorted roles.")
    assertion_ids = tuple(item.assertion_id for item in metric_bindings)
    if len(assertion_ids) != len(set(assertion_ids)):
        raise ValueError("Each assertion may bind to exactly one memory metric role.")
    ranking_roles = tuple(item.role for item in ranking)
    if len(ranking_roles) != len(set(ranking_roles)) or any(
        role not in roles for role in ranking_roles
    ):
        raise ValueError("Ranking terms must be unique and reference declared metrics.")
    if any(role not in _RANKABLE_MEMORY_METRIC_ROLES for role in ranking_roles):
        raise ValueError(
            "Ranking terms must use task-quality dimensions; safety and evidence "
            "dimensions belong in gates."
        )
    directions = {item.role: item.direction for item in metric_bindings}
    if any(directions[item.role] is not item.direction for item in ranking):
        raise ValueError("Ranking directions must match their declared metric directions.")
    if any(role not in roles for role in gates.required_metric_roles) or any(
        gate.role not in roles for gate in gates.metric_gates
    ):
        raise ValueError("Gate roles must reference declared metrics.")


class MemoryExperimentCase(_ReportModel):
    case_id: StrictStr = Field(max_length=128)
    case_revision: StrictStr = Field(min_length=71, max_length=71)

    @field_validator("case_id")
    @classmethod
    def validate_case_id(cls, value: str, info) -> str:
        return _clean(value, info.field_name, maximum=128)

    @field_validator("case_revision")
    @classmethod
    def validate_case_revision(cls, value: str, info) -> str:
        return _revision(value, info.field_name)


class MemoryExperimentVariant(_ReportModel):
    variant_id: StrictStr = Field(max_length=128)
    candidate_id: StrictStr = Field(max_length=256)
    spec: MemoryInterventionSpec
    execution_profile: EvalExecutionProfileV1
    execution_profile_binding: EvalExecutionProfileBindingV1
    evaluator_fingerprint: StrictStr = Field(min_length=64, max_length=64)

    @field_validator("variant_id", "candidate_id")
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        return _clean(
            value,
            info.field_name,
            maximum=128 if info.field_name == "variant_id" else 256,
        )

    @field_validator("evaluator_fingerprint")
    @classmethod
    def validate_evaluator_fingerprint(cls, value: str, info) -> str:
        return _fingerprint(value, info.field_name)

    @field_validator("spec", mode="before")
    @classmethod
    def copy_spec(cls, value: object) -> object:
        return revalidate_model_input(value, MemoryInterventionSpec)

    @field_validator("execution_profile", mode="before")
    @classmethod
    def copy_profile(cls, value: object) -> object:
        return revalidate_model_input(value, EvalExecutionProfileV1)

    @field_validator("execution_profile_binding", mode="before")
    @classmethod
    def copy_binding(cls, value: object) -> object:
        return revalidate_model_input(value, EvalExecutionProfileBindingV1)

    @model_validator(mode="after")
    def validate_profile(self) -> Self:
        runtime = self.execution_profile_binding.runtime_execution_profile
        candidate = self.execution_profile.candidate
        if (
            self.execution_profile.revision != self.execution_profile_binding.profile_revision
            or candidate.runtime_execution_profile_schema_version != runtime.schema_version
            or candidate.runtime_execution_profile_fingerprint != runtime.fingerprint
        ):
            raise ValueError("Memory variant execution profile evidence conflicts.")
        profile_changed = self.spec.execution_profile_fingerprint != runtime.fingerprint
        if profile_changed != (self.spec.kind is MemoryInterventionKind.AUTOMATIC_RECALL_OFF):
            raise ValueError(
                "Only automatic_recall_off may use an effective runtime profile that differs "
                "from the frozen starting profile."
            )
        return self


def _require_experiment_variant_execution_authority(
    variants: tuple[MemoryExperimentVariant, ...],
    baseline_variant_id: str,
) -> None:
    by_id = {item.variant_id: item for item in variants}
    baseline = by_id[baseline_variant_id]
    baseline_runtime = baseline.execution_profile_binding.runtime_execution_profile
    for variant in variants:
        if variant.variant_id == baseline_variant_id:
            continue
        if variant.spec.execution_profile_fingerprint != baseline_runtime.fingerprint:
            raise ValueError(
                "Every memory intervention must start from the baseline execution profile."
            )
        candidate_runtime = variant.execution_profile_binding.runtime_execution_profile
        if variant.spec.kind is MemoryInterventionKind.AUTOMATIC_RECALL_OFF:
            baseline_components = {
                item.component_class: item for item in baseline_runtime.components
            }
            candidate_components = {
                item.component_class: item for item in candidate_runtime.components
            }
            changed = tuple(
                component_class
                for component_class in sorted(
                    set(baseline_components) | set(candidate_components),
                    key=str,
                )
                if baseline_components.get(component_class)
                != candidate_components.get(component_class)
            )
            if changed != (ExecutionProfileComponentClass.AUTOMATIC_RECALL,):
                raise ValueError(
                    "automatic_recall_off must change exactly the automatic-recall "
                    "execution-profile component."
                )
        elif candidate_runtime.fingerprint != baseline_runtime.fingerprint:
            raise ValueError(
                "Memory interventions other than automatic_recall_off cannot change the "
                "runtime execution profile."
            )


def _eval_profile_comparison_material(
    profile: EvalExecutionProfileV1,
    *,
    automatic_recall_variant: bool,
) -> dict[str, object]:
    material = profile.model_dump(mode="json", exclude={"revision"})
    if automatic_recall_variant:
        # Only the runtime-owned automatic-recall component may change. The
        # application manifest remains part of the generic experiment contract;
        # otherwise an unrelated application change could be attributed to the
        # intervention.
        candidate = material["candidate"]
        assert isinstance(candidate, dict)
        candidate.pop("runtime_execution_profile_fingerprint")
    return material


def _generic_experiment_identity_matches(
    baseline_row: MemoryTrialReportRow,
    candidate_row: MemoryTrialReportRow,
    baseline_variant: MemoryExperimentVariant,
    candidate_variant: MemoryExperimentVariant,
) -> bool:
    automatic_recall_variant = (
        candidate_variant.spec.kind is MemoryInterventionKind.AUTOMATIC_RECALL_OFF
    )
    return (
        baseline_row.case_revision == candidate_row.case_revision
        and baseline_row.corpus_revision == candidate_row.corpus_revision
        and baseline_row.suite_id == candidate_row.suite_id
        and baseline_row.suite_revision == candidate_row.suite_revision
        and baseline_row.evidence_policy_revision == candidate_row.evidence_policy_revision
        and baseline_row.pricing_profile_fingerprint == candidate_row.pricing_profile_fingerprint
        and baseline_row.evaluator_fingerprint == candidate_row.evaluator_fingerprint
        and baseline_row.provider_name == candidate_row.provider_name
        and baseline_row.model == candidate_row.model
        and _eval_profile_comparison_material(
            baseline_variant.execution_profile,
            automatic_recall_variant=automatic_recall_variant,
        )
        == _eval_profile_comparison_material(
            candidate_variant.execution_profile,
            automatic_recall_variant=automatic_recall_variant,
        )
    )


class MemoryPreparationOverheadEvidence(_ReportModel):
    evidence_revision: StrictStr = Field(min_length=71, max_length=71)
    preparation_duration_ms: StrictInt | None = Field(default=None, ge=0)
    context_tokens: StrictInt | None = Field(default=None, ge=0)
    context_bytes: StrictInt | None = Field(default=None, ge=0)

    @field_validator("evidence_revision")
    @classmethod
    def validate_revision(cls, value: str, info) -> str:
        return _revision(value, info.field_name)

    @model_validator(mode="after")
    def validate_content_revision(self) -> Self:
        material = self.model_dump(mode="json", exclude={"evidence_revision"})
        if self.evidence_revision != _content_revision(
            material,
            "memory preparation overhead evidence",
        ):
            raise ValueError("Memory preparation overhead revision does not match its content.")
        if all(
            value is None
            for value in (
                self.preparation_duration_ms,
                self.context_tokens,
                self.context_bytes,
            )
        ):
            raise ValueError("Memory preparation overhead requires observed evidence.")
        return self

    @classmethod
    def create(
        cls,
        *,
        preparation_duration_ms: int | None = None,
        context_tokens: int | None = None,
        context_bytes: int | None = None,
    ) -> MemoryPreparationOverheadEvidence:
        material = {
            "preparation_duration_ms": preparation_duration_ms,
            "context_tokens": context_tokens,
            "context_bytes": context_bytes,
        }
        return cls(
            evidence_revision=_content_revision(
                material,
                "memory preparation overhead evidence",
            ),
            **material,
        )


class MemoryPublishedResultEvidence(_ReportModel):
    """One complete content-addressed EvalStore result available to report rows."""

    run_id: StrictStr = Field(max_length=512)
    # CorpusExecutionResult has a deliberate JSON-string/Python-int aggregate
    # counter boundary. Validate it exactly here and prevent the outer model's
    # JSON mode from revalidating the returned typed instance as raw JSON.
    result: SkipValidation[CorpusExecutionResult]

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str, info) -> str:
        return _clean(value, info.field_name, maximum=512)

    @field_validator("result", mode="before")
    @classmethod
    def copy_result(cls, value: object) -> object:
        if isinstance(value, dict):
            try:
                return CorpusExecutionResult.model_validate(value)
            except (TypeError, ValueError):
                pass
            try:
                encoded = json.dumps(
                    value,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            except (TypeError, ValueError, UnicodeError) as exc:
                raise ValueError("Published result evidence must be portable JSON.") from exc
            if len(encoded) > MEMORY_EXPERIMENT_REPORT_MAX_BYTES:
                raise ValueError("Published result evidence exceeds the report byte bound.")
            return CorpusExecutionResult.model_validate_json(encoded)
        copied = revalidate_model_input(value, CorpusExecutionResult)
        if type(copied) is not CorpusExecutionResult:
            raise ValueError("Published result evidence must be an exact corpus execution result.")
        return copied


class MemoryExperimentTrialEvidence(_ReportModel):
    case_id: StrictStr = Field(max_length=128)
    case_revision: StrictStr = Field(min_length=71, max_length=71)
    repetition: StrictInt = Field(ge=1, le=MEMORY_EXPERIMENT_REPORT_MAX_REPETITIONS)
    variant_id: StrictStr = Field(max_length=128)
    execution: MemoryInterventionExecutionRecord
    intervention_binding: MemoryInterventionTrialBinding | None = None
    published_result_revision: StrictStr | None = Field(default=None, max_length=71)
    accounting_side: PairedCostQualitySide | None = None
    memory_overhead: MemoryPreparationOverheadEvidence | None = None

    @field_validator("case_id", "variant_id")
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        return _clean(value, info.field_name, maximum=128)

    @field_validator("case_revision")
    @classmethod
    def validate_case_revision(cls, value: str, info) -> str:
        return _revision(value, info.field_name)

    @field_validator("execution", mode="before")
    @classmethod
    def copy_execution(cls, value: object) -> object:
        return revalidate_model_input(value, MemoryInterventionExecutionRecord)

    @field_validator("intervention_binding", mode="before")
    @classmethod
    def copy_intervention_binding(cls, value: object) -> object:
        return revalidate_model_input(value, MemoryInterventionTrialBinding)

    @field_validator("published_result_revision")
    @classmethod
    def validate_published_result_revision(cls, value: str | None, info) -> str | None:
        return None if value is None else _revision(value, info.field_name)

    @field_validator("accounting_side", mode="before")
    @classmethod
    def copy_accounting_side(cls, value: object) -> object:
        return revalidate_model_input(value, PairedCostQualitySide)

    @field_validator("memory_overhead", mode="before")
    @classmethod
    def copy_memory_overhead(cls, value: object) -> object:
        return revalidate_model_input(value, MemoryPreparationOverheadEvidence)


class MemoryExperimentReportRequest(_ReportModel):
    schema_version: Literal[1] = 1
    experiment_id: StrictStr = Field(max_length=128)
    cases: tuple[MemoryExperimentCase, ...] = Field(
        min_length=1, max_length=MEMORY_EXPERIMENT_REPORT_MAX_CASES
    )
    repetitions: StrictInt = Field(ge=1, le=MEMORY_EXPERIMENT_REPORT_MAX_REPETITIONS)
    baseline_variant_id: StrictStr = Field(max_length=128)
    variants: tuple[MemoryExperimentVariant, ...] = Field(
        min_length=2, max_length=MEMORY_EXPERIMENT_REPORT_MAX_VARIANTS
    )
    metric_bindings: tuple[MemoryMetricBinding, ...] = Field(
        default=(), max_length=MEMORY_EXPERIMENT_REPORT_MAX_METRICS
    )
    ranking: tuple[MemoryRankingTerm, ...] = Field(
        default=(), max_length=MEMORY_EXPERIMENT_REPORT_MAX_METRICS
    )
    gates: MemoryExperimentGatePolicy = Field(default_factory=MemoryExperimentGatePolicy)
    published_results: tuple[MemoryPublishedResultEvidence, ...] = Field(
        default=(), max_length=MEMORY_EXPERIMENT_REPORT_MAX_ROWS
    )
    trials: tuple[MemoryExperimentTrialEvidence, ...] = Field(
        default=(), max_length=MEMORY_EXPERIMENT_REPORT_MAX_ROWS
    )

    @field_validator("experiment_id", "baseline_variant_id")
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        return _clean(value, info.field_name, maximum=128)

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        case_keys = tuple((item.case_id, item.case_revision) for item in self.cases)
        case_ids = tuple(item.case_id for item in self.cases)
        if case_keys != tuple(sorted(case_keys)) or len(case_ids) != len(set(case_ids)):
            raise ValueError("Experiment case IDs must be unique and cases sorted.")
        variant_ids = tuple(item.variant_id for item in self.variants)
        if variant_ids != tuple(sorted(set(variant_ids))):
            raise ValueError("Experiment variants must be unique and sorted.")
        variants = {item.variant_id: item for item in self.variants}
        baseline = variants.get(self.baseline_variant_id)
        if baseline is None or baseline.spec.kind is not MemoryInterventionKind.AS_DECLARED:
            raise ValueError("The baseline variant must be the exact as-declared intervention.")
        if any(
            item.variant_id != self.baseline_variant_id
            and item.spec.kind is MemoryInterventionKind.AS_DECLARED
            for item in self.variants
        ):
            raise ValueError("Only the baseline variant may use the as-declared intervention.")
        _require_experiment_variant_execution_authority(
            self.variants,
            self.baseline_variant_id,
        )
        _validate_metric_policy(self.metric_bindings, self.ranking, self.gates)
        expected_rows = len(self.cases) * self.repetitions * len(self.variants)
        if expected_rows > MEMORY_EXPERIMENT_REPORT_MAX_ROWS:
            raise ValueError("The expanded memory experiment exceeds its row bound.")
        expected_pairs = len(self.cases) * self.repetitions * (len(self.variants) - 1)
        if expected_pairs > MEMORY_EXPERIMENT_REPORT_MAX_PAIRS:
            raise ValueError(
                "The expanded memory experiment exceeds the canonical paired-comparison bound."
            )
        evidence_keys = tuple(
            (item.case_id, item.repetition, item.variant_id) for item in self.trials
        )
        if evidence_keys != tuple(sorted(set(evidence_keys))):
            raise ValueError("Trial evidence must have unique, sorted matrix coordinates.")
        execution_ids = tuple(item.execution.execution_id for item in self.trials)
        if len(execution_ids) != len(set(execution_ids)):
            raise ValueError("Trial evidence execution identities must be unique.")
        trial_ids = tuple(item.execution.trial_id for item in self.trials)
        if len(trial_ids) != len(set(trial_ids)):
            raise ValueError("Trial evidence trial identities must be unique.")
        accounting_attempt_ids = tuple(
            attempt.attempt_id
            for item in self.trials
            if item.accounting_side is not None
            for attempt in item.accounting_side.attempts
        )
        if len(accounting_attempt_ids) != len(set(accounting_attempt_ids)):
            raise ValueError("Accounting attempt identities must belong to exactly one trial row.")
        published_revisions = tuple(item.result.revision for item in self.published_results)
        if published_revisions != tuple(sorted(set(published_revisions))):
            raise ValueError("Published result evidence must be unique and sorted by revision.")
        referenced_revisions = tuple(
            sorted(
                {
                    item.published_result_revision
                    for item in self.trials
                    if item.published_result_revision is not None
                }
            )
        )
        if published_revisions != referenced_revisions:
            raise ValueError(
                "Published result evidence must exactly match the referenced result graph."
            )
        published_run_ids = tuple(item.run_id for item in self.published_results)
        if len(published_run_ids) != len(set(published_run_ids)):
            raise ValueError("Published result evidence must use unique eval run IDs.")
        published_results = {item.result.revision: item.result for item in self.published_results}
        cases = dict(case_keys)
        for item in self.trials:
            case_revision = cases.get(item.case_id)
            variant = variants.get(item.variant_id)
            if case_revision != item.case_revision or variant is None:
                raise ValueError("Trial evidence references an undeclared matrix coordinate.")
            if item.repetition > self.repetitions:
                raise ValueError("Trial evidence repetition exceeds the experiment declaration.")
            if item.published_result_revision is not None:
                result = published_results.get(item.published_result_revision)
                if result is None:
                    raise ValueError("Trial evidence references an unavailable published result.")
                if (
                    result.target.target_key != variant.execution_profile.target_key
                    or result.target.application_release_id
                    != variant.execution_profile.application_release_id
                    or result.target.app_manifest.fingerprint
                    != variant.execution_profile.app_manifest_fingerprint
                ):
                    raise ValueError(
                        "Published result target conflicts with its experiment variant."
                    )
                if (
                    result.run.evidence_policy_revision
                    != variant.execution_profile.evidence_policy.revision
                ):
                    raise ValueError(
                        "Published result evidence policy conflicts with its experiment variant."
                    )
            _validate_trial_evidence(item, variant, self.experiment_id, published_results)
        if compact_json_utf8_size(self.model_dump(mode="json")) > (
            MEMORY_EXPERIMENT_REPORT_MAX_BYTES
        ):
            raise ValueError("Memory experiment request exceeds its byte bound.")
        return self


class MemoryMetricObservation(_ReportModel):
    role: MemoryMetricRole
    assertion_id: StrictStr
    assertion_revision: StrictStr
    availability: MemoryMetricAvailability
    outcome: PublishedOutcome | None = None
    value: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    evaluator_key: StrictStr | None = Field(default=None, max_length=128)
    evaluator_implementation_revision: StrictStr | None = Field(
        default=None,
        min_length=71,
        max_length=71,
    )

    @field_validator("assertion_id")
    @classmethod
    def validate_assertion_id(cls, value: str, info) -> str:
        return _clean(value, info.field_name, maximum=128)

    @field_validator("assertion_revision")
    @classmethod
    def validate_assertion_revision(cls, value: str, info) -> str:
        return _revision(value, info.field_name)

    @field_validator("evaluator_key")
    @classmethod
    def validate_evaluator_key(cls, value: str | None, info) -> str | None:
        return None if value is None else _clean(value, info.field_name, maximum=128)

    @field_validator("evaluator_implementation_revision")
    @classmethod
    def validate_evaluator_revision(cls, value: str | None, info) -> str | None:
        return None if value is None else _revision(value, info.field_name)

    @model_validator(mode="after")
    def validate_availability(self) -> Self:
        if (self.evaluator_key is None) != (self.evaluator_implementation_revision is None):
            raise ValueError("Model-judge metric evidence requires complete evaluator identity.")
        expected_availability = (
            MemoryMetricAvailability.UNAVAILABLE
            if self.outcome in {None, "unavailable"}
            else MemoryMetricAvailability.ERROR
            if self.outcome == "error"
            else MemoryMetricAvailability.AVAILABLE
        )
        if self.availability is not expected_availability:
            raise ValueError("Metric availability contradicts its published outcome.")
        if self.availability is MemoryMetricAvailability.AVAILABLE and self.value is None:
            raise ValueError("Available metric evidence requires a numeric value.")
        if self.availability is not MemoryMetricAvailability.AVAILABLE and self.value is not None:
            raise ValueError("Unavailable metric evidence cannot carry a numeric value.")
        return self


def _require_execution_variant_authority(
    execution: MemoryInterventionExecutionRecord,
    *,
    case_id: str,
    case_revision: str,
    variant: MemoryExperimentVariant,
) -> None:
    if (
        execution.case_id != case_id
        or execution.case_revision != case_revision
        or execution.candidate_id != variant.candidate_id
        or execution.spec_fingerprint != variant.spec.fingerprint
        or execution.required_execution_profile_fingerprint
        != variant.spec.execution_profile_fingerprint
        or execution.runtime_execution_profile_fingerprint
        != variant.execution_profile_binding.runtime_execution_profile.fingerprint
        or execution.evaluator_fingerprint != variant.evaluator_fingerprint
    ):
        raise ValueError("Trial execution conflicts with its experiment authority.")


def _require_terminal_execution_binding(
    execution: MemoryInterventionExecutionRecord,
    binding: MemoryInterventionTrialBinding | None,
) -> str | None:
    if execution.status is MemoryInterventionExecutionStatus.ACTIVE:
        raise ValueError("Memory experiment reports require terminal trial executions.")
    if (binding is None) != (execution.final_binding_fingerprint is None):
        raise ValueError("Trial binding evidence is incomplete.")
    if binding is None:
        return None
    execution_lineage = {
        "spec_fingerprint": execution.spec_fingerprint,
        "materialization_fingerprint": execution.materialization_fingerprint,
        "trial_binding_fingerprint": execution.trial_binding_fingerprint,
        "operation_fingerprint": execution.operation_fingerprint,
        "receipt_fingerprint": execution.receipt_fingerprint,
        "snapshot_result_fingerprint": execution.snapshot_result_fingerprint,
        "final_binding_fingerprint": execution.final_binding_fingerprint,
        "evaluator_fingerprint": execution.evaluator_fingerprint,
        "session_id": execution.session_id,
        "runtime_evidence_fingerprint": execution.runtime_evidence_fingerprint,
        "eval_result_revision": execution.eval_result_revision,
        "terminal_disposition": execution.status.value,
    }
    binding_lineage = _intervention_binding_lineage(binding)
    if binding_lineage != execution_lineage:
        raise ValueError("Trial binding conflicts with its complete execution lineage.")
    return _content_revision(
        execution_lineage,
        "memory intervention execution binding lineage",
    )


def _intervention_binding_lineage(
    binding: MemoryInterventionTrialBinding,
) -> dict[str, object]:
    return {
        "spec_fingerprint": binding.spec.fingerprint,
        "materialization_fingerprint": binding.operation.materialization_fingerprint,
        "trial_binding_fingerprint": binding.trial.fingerprint,
        "operation_fingerprint": binding.operation.fingerprint,
        "receipt_fingerprint": binding.receipt.fingerprint,
        "snapshot_result_fingerprint": binding.result.fingerprint,
        "final_binding_fingerprint": binding.fingerprint,
        "evaluator_fingerprint": binding.trial.evaluator_fingerprint,
        "session_id": binding.result.session_id,
        "runtime_evidence_fingerprint": binding.result.runtime_evidence_fingerprint,
        "eval_result_revision": binding.result.eval_result_revision,
        "terminal_disposition": binding.result.terminal_disposition.value,
    }


class MemoryTrialReportRow(_ReportModel):
    row_id: StrictStr = Field(min_length=71, max_length=71)
    case_id: StrictStr
    case_revision: StrictStr
    repetition: StrictInt = Field(ge=1, le=MEMORY_EXPERIMENT_REPORT_MAX_REPETITIONS)
    variant_id: StrictStr
    intervention_spec_fingerprint: StrictStr
    snapshot_fingerprint: StrictStr
    execution_profile_revision: StrictStr
    runtime_execution_profile_fingerprint: StrictStr
    provider_name: StrictStr
    model: StrictStr
    evaluator_fingerprint: StrictStr
    execution_id: StrictStr | None = None
    trial_id: StrictStr | None = None
    execution_revision: StrictInt | None = Field(default=None, ge=0)
    final_binding_fingerprint: StrictStr | None = None
    execution_binding_lineage_revision: StrictStr | None = None
    intervention_binding: MemoryInterventionTrialBinding | None = None
    published_result_revision: StrictStr | None = None
    published_run_id: StrictStr | None = None
    source_trial_revision: StrictStr | None = None
    corpus_revision: StrictStr | None = None
    suite_id: StrictStr | None = None
    suite_revision: StrictStr | None = None
    evidence_policy_revision: StrictStr | None = None
    pricing_profile_fingerprint: StrictStr | None = None
    intervention_attribution_fingerprint: StrictStr | None = None
    attribution_evidence_revision: StrictStr | None = None
    attribution_status: EvalMemoryEvidenceCompleteness | None = None
    availability: MemoryTrialAvailability
    execution_status: MemoryInterventionExecutionStatus | None = None
    published_status: PublishedOutcome | None = None
    metrics: tuple[MemoryMetricObservation, ...] = ()
    duration_ms: StrictInt | None = Field(default=None, ge=0)
    total_tokens: AggregateCount | None = Field(default=None, ge=0)
    accounting_side: PairedCostQualitySide | None = None
    memory_overhead: MemoryPreparationOverheadEvidence | None = None

    @field_validator(
        "row_id",
        "case_revision",
        "execution_profile_revision",
        "execution_binding_lineage_revision",
        "published_result_revision",
        "corpus_revision",
        "suite_revision",
        "evidence_policy_revision",
        "attribution_evidence_revision",
    )
    @classmethod
    def validate_revisions(cls, value: str | None, info) -> str | None:
        return None if value is None else _revision(value, info.field_name)

    @field_validator(
        "intervention_spec_fingerprint",
        "snapshot_fingerprint",
        "runtime_execution_profile_fingerprint",
        "evaluator_fingerprint",
        "pricing_profile_fingerprint",
        "final_binding_fingerprint",
        "intervention_attribution_fingerprint",
    )
    @classmethod
    def validate_fingerprints(cls, value: str | None, info) -> str | None:
        return None if value is None else _fingerprint(value, info.field_name)

    @field_validator("source_trial_revision")
    @classmethod
    def validate_source_trial_revision(cls, value: str | None, info) -> str | None:
        return None if value is None else _fingerprint(value, info.field_name)

    @field_validator("intervention_binding", mode="before")
    @classmethod
    def copy_intervention_binding(cls, value: object) -> object:
        return revalidate_model_input(value, MemoryInterventionTrialBinding)

    @field_validator("suite_id")
    @classmethod
    def validate_suite_id(cls, value: str | None, info) -> str | None:
        return None if value is None else _clean(value, info.field_name, maximum=128)

    @model_validator(mode="after")
    def validate_evidence_shape(self) -> Self:
        if self.source_trial_revision is None and any(
            observation.availability is not MemoryMetricAvailability.UNAVAILABLE
            or observation.outcome is not None
            or observation.value is not None
            or observation.evaluator_key is not None
            or observation.evaluator_implementation_revision is not None
            for observation in self.metrics
        ):
            raise ValueError("A row without a published trial cannot retain metric evidence.")
        if self.execution_id is None:
            if self.availability is not MemoryTrialAvailability.MISSING or any(
                value is not None
                for value in (
                    self.trial_id,
                    self.execution_revision,
                    self.final_binding_fingerprint,
                    self.execution_binding_lineage_revision,
                    self.intervention_binding,
                    self.published_result_revision,
                    self.published_run_id,
                    self.source_trial_revision,
                    self.corpus_revision,
                    self.suite_id,
                    self.suite_revision,
                    self.evidence_policy_revision,
                    self.pricing_profile_fingerprint,
                    self.intervention_attribution_fingerprint,
                    self.attribution_evidence_revision,
                    self.attribution_status,
                    self.execution_status,
                    self.published_status,
                    self.duration_ms,
                    self.total_tokens,
                    self.accounting_side,
                    self.memory_overhead,
                )
            ):
                raise ValueError("A missing trial row cannot retain execution evidence.")
            return self
        if (
            self.trial_id is None
            or self.execution_revision is None
            or self.execution_status is None
        ):
            raise ValueError("An observed trial row requires exact execution identity.")
        if self.execution_status is MemoryInterventionExecutionStatus.ACTIVE:
            raise ValueError("Memory experiment reports require terminal trial executions.")
        if (self.intervention_binding is None) != (self.final_binding_fingerprint is None):
            raise ValueError(
                "An observed trial row requires complete intervention binding evidence."
            )
        if (
            self.intervention_binding is not None
            and self.intervention_binding.fingerprint != self.final_binding_fingerprint
        ):
            raise ValueError("Intervention binding evidence conflicts with its fingerprint.")
        if (self.intervention_binding is None) != (self.execution_binding_lineage_revision is None):
            raise ValueError("Trial row execution-binding lineage evidence is incomplete.")
        if self.intervention_binding is not None and (
            self.execution_binding_lineage_revision
            != _content_revision(
                _intervention_binding_lineage(self.intervention_binding),
                "memory intervention execution binding lineage",
            )
            or self.intervention_binding.result.terminal_disposition.value
            != self.execution_status.value
        ):
            raise ValueError("Trial binding conflicts with its complete execution lineage.")
        if (self.published_result_revision is None) != (self.published_run_id is None):
            raise ValueError("Published result and run identities must be present together.")
        published_generic_identity = (
            self.corpus_revision,
            self.suite_id,
            self.suite_revision,
            self.evidence_policy_revision,
        )
        if (self.published_result_revision is None) != all(
            value is None for value in published_generic_identity
        ):
            raise ValueError(
                "Published result evidence requires its complete generic experiment identity."
            )
        if self.source_trial_revision is not None and self.published_result_revision is None:
            raise ValueError("Published trial identity requires its result identity.")
        published_trial_fields = (
            self.published_status,
            self.attribution_evidence_revision,
            self.attribution_status,
            self.duration_ms,
        )
        if self.source_trial_revision is None:
            if any(value is not None for value in published_trial_fields):
                raise ValueError(
                    "A row without a published trial cannot retain published trial evidence."
                )
        elif any(value is None for value in published_trial_fields):
            raise ValueError("A published trial row requires its complete status evidence.")
        expected_availability = _trial_availability_from_statuses(
            self.execution_status,
            published=self.source_trial_revision is not None,
            published_status=self.published_status,
            attribution_status=self.attribution_status,
        )
        if self.availability is not expected_availability:
            raise ValueError("Trial availability contradicts its retained status evidence.")
        if self.availability is MemoryTrialAvailability.AVAILABLE and any(
            value is None
            for value in (
                self.final_binding_fingerprint,
                self.source_trial_revision,
                self.published_status,
                self.intervention_attribution_fingerprint,
                self.attribution_evidence_revision,
            )
        ):
            raise ValueError("An available trial row requires complete result lineage.")
        if (
            self.availability is MemoryTrialAvailability.AVAILABLE
            and self.attribution_status is not EvalMemoryEvidenceCompleteness.COMPLETE
        ):
            raise ValueError("An available trial row requires complete attribution evidence.")
        if (
            self.availability is MemoryTrialAvailability.UNMATCHED
            and self.execution_status is not MemoryInterventionExecutionStatus.COMPLETED
        ):
            raise ValueError("Only a completed execution may be unmatched.")
        return self


class MemoryMetricDelta(_ReportModel):
    role: MemoryMetricRole
    baseline: StrictFloat
    candidate: StrictFloat
    delta: StrictFloat


class MemoryOperationalDelta(_ReportModel):
    dimension: MemoryOperationalDimension
    baseline: AggregateCount = Field(ge=0)
    candidate: AggregateCount = Field(ge=0)
    delta: SignedAggregateCount

    @model_validator(mode="after")
    def validate_delta(self) -> Self:
        if self.delta != self.candidate - self.baseline:
            raise ValueError("Memory operational delta arithmetic conflicts.")
        return self


class MemoryTrialPairComparison(_ReportModel):
    pair_id: StrictStr = Field(min_length=71, max_length=71)
    case_id: StrictStr
    case_revision: StrictStr
    repetition: StrictInt
    baseline_variant_id: StrictStr
    candidate_variant_id: StrictStr
    baseline_row_id: StrictStr
    candidate_row_id: StrictStr
    status: MemoryPairStatus
    reasons: tuple[StrictStr, ...] = ()
    memory_comparability_fingerprint: StrictStr | None = None
    metric_deltas: tuple[MemoryMetricDelta, ...] = ()
    operational_deltas: tuple[MemoryOperationalDelta, ...] = ()
    accounting_pair: PairedCostQualityPairReport | None = None

    @model_validator(mode="after")
    def validate_pair(self) -> Self:
        if self.baseline_variant_id == self.candidate_variant_id:
            raise ValueError("Memory trial pair requires distinct baseline and candidate variants.")
        if self.pair_id != _pair_id(
            self.case_id,
            self.repetition,
            self.candidate_variant_id,
        ):
            raise ValueError("Memory trial pair identity conflicts with its coordinates.")
        if self.baseline_row_id != _row_id(
            self.case_id,
            self.repetition,
            self.baseline_variant_id,
        ) or self.candidate_row_id != _row_id(
            self.case_id,
            self.repetition,
            self.candidate_variant_id,
        ):
            raise ValueError("Memory trial pair row identities conflict with its coordinates.")
        if self.accounting_pair is not None and self.accounting_pair.pair_id != self.pair_id:
            raise ValueError("Memory trial pair accounting identity conflicts.")
        if self.status is MemoryPairStatus.UNAVAILABLE:
            if self.memory_comparability_fingerprint is not None:
                raise ValueError(
                    "Unavailable memory trial pair cannot claim comparability evidence."
                )
        elif self.memory_comparability_fingerprint is None:
            raise ValueError("Classified memory trial pair requires comparability evidence.")
        else:
            _fingerprint(
                self.memory_comparability_fingerprint,
                "memory_comparability_fingerprint",
            )
        if self.reasons != tuple(sorted(set(self.reasons))):
            raise ValueError("Memory trial pair reasons must be unique and sorted.")
        roles = tuple(item.role for item in self.metric_deltas)
        if roles != tuple(sorted(set(roles), key=str)):
            raise ValueError("Memory trial pair metric deltas must be unique and sorted.")
        if any(item.delta != item.candidate - item.baseline for item in self.metric_deltas):
            raise ValueError("Memory trial pair metric delta arithmetic conflicts.")
        dimensions = tuple(item.dimension for item in self.operational_deltas)
        if dimensions != tuple(sorted(set(dimensions), key=str)):
            raise ValueError("Memory trial pair operational deltas must be unique and sorted.")
        if self.status is not MemoryPairStatus.COMPARABLE and (
            self.metric_deltas or self.operational_deltas
        ):
            raise ValueError("Only comparable memory trial pairs may retain deltas.")
        return self


class MemoryMetricDistribution(_ReportModel):
    role: MemoryMetricRole
    pair_count: StrictInt = Field(ge=0)
    available_count: StrictInt = Field(ge=0)
    unavailable_count: StrictInt = Field(ge=0)
    incomparable_count: StrictInt = Field(ge=0)
    baseline_values: tuple[StrictFloat, ...] = ()
    candidate_values: tuple[StrictFloat, ...] = ()
    deltas: tuple[StrictFloat, ...] = ()
    mean_delta: StrictFloat | None = None
    median_delta: StrictFloat | None = None

    @model_validator(mode="after")
    def validate_distribution(self) -> Self:
        if (
            self.available_count + self.unavailable_count + self.incomparable_count
            != self.pair_count
            or len(self.baseline_values) != self.available_count
            or len(self.candidate_values) != self.available_count
            or len(self.deltas) != self.available_count
        ):
            raise ValueError("Memory metric distribution counts conflict.")
        if self.available_count == 0:
            if self.mean_delta is not None or self.median_delta is not None:
                raise ValueError("Unavailable metric distribution cannot carry aggregates.")
        else:
            if self.mean_delta is None or self.median_delta is None:
                raise ValueError("Available metric distribution requires its aggregates.")
            if any(
                delta != candidate - baseline
                for baseline, candidate, delta in zip(
                    self.baseline_values,
                    self.candidate_values,
                    self.deltas,
                    strict=True,
                )
            ):
                raise ValueError("Memory metric distribution delta arithmetic conflicts.")
            if self.mean_delta != sum(self.deltas) / len(self.deltas) or (
                self.median_delta != float(median(self.deltas))
            ):
                raise ValueError("Memory metric distribution aggregates conflict.")
        return self


class MemoryOperationalDistribution(_ReportModel):
    dimension: MemoryOperationalDimension
    pair_count: StrictInt = Field(ge=0)
    available_count: StrictInt = Field(ge=0)
    unavailable_count: StrictInt = Field(ge=0)
    incomparable_count: StrictInt = Field(ge=0)
    baseline_values: tuple[AggregateCount, ...] = ()
    candidate_values: tuple[AggregateCount, ...] = ()
    deltas: tuple[SignedAggregateCount, ...] = ()
    mean_delta: StrictFloat | None = None
    median_delta: StrictFloat | None = None

    @model_validator(mode="after")
    def validate_distribution(self) -> Self:
        if (
            self.available_count + self.unavailable_count + self.incomparable_count
            != self.pair_count
            or len(self.baseline_values) != self.available_count
            or len(self.candidate_values) != self.available_count
            or len(self.deltas) != self.available_count
        ):
            raise ValueError("Memory operational distribution counts conflict.")
        if self.available_count == 0:
            if self.mean_delta is not None or self.median_delta is not None:
                raise ValueError("Unavailable operational distribution cannot carry aggregates.")
        else:
            if self.mean_delta is None or self.median_delta is None:
                raise ValueError("Available operational distribution requires its aggregates.")
            if self.mean_delta != sum(self.deltas) / len(self.deltas) or (
                self.median_delta != float(median(self.deltas))
            ):
                raise ValueError("Memory operational distribution aggregates conflict.")
        return self


class MemoryCaseComparison(_ReportModel):
    schema_version: Literal[1] = 1
    revision: StrictStr = Field(min_length=71, max_length=71)
    case_id: StrictStr
    case_revision: StrictStr
    candidate_variant_id: StrictStr
    pairs: tuple[MemoryTrialPairComparison, ...]
    metric_roles: tuple[MemoryMetricRole, ...]
    distributions: tuple[MemoryMetricDistribution, ...]
    operational_distributions: tuple[MemoryOperationalDistribution, ...]

    @model_validator(mode="after")
    def validate_revision(self) -> Self:
        if tuple(item.repetition for item in self.pairs) != tuple(range(1, len(self.pairs) + 1)):
            raise ValueError("Memory case comparison repetitions must be contiguous.")
        if len({item.baseline_variant_id for item in self.pairs}) != 1:
            raise ValueError("Memory case comparison requires one exact baseline variant.")
        if any(
            item.case_id != self.case_id
            or item.case_revision != self.case_revision
            or item.candidate_variant_id != self.candidate_variant_id
            for item in self.pairs
        ):
            raise ValueError("Memory case comparison pair identities conflict.")
        if self.metric_roles != tuple(sorted(set(self.metric_roles), key=str)):
            raise ValueError("Memory case comparison metric roles must be unique and sorted.")
        roles = tuple(item.role for item in self.distributions)
        if roles != self.metric_roles:
            raise ValueError(
                "Memory case comparison must retain every declared metric distribution."
            )
        dimensions = tuple(item.dimension for item in self.operational_distributions)
        if dimensions != tuple(sorted(MemoryOperationalDimension, key=str)):
            raise ValueError("Memory case comparison must retain every operational distribution.")
        expected_distributions = tuple(
            _distribution(item.role, self.pairs) for item in self.distributions
        )
        if self.distributions != expected_distributions:
            raise ValueError("Memory case comparison distributions conflict with their pairs.")
        expected_operational_distributions = tuple(
            _operational_distribution(item.dimension, self.pairs)
            for item in self.operational_distributions
        )
        if self.operational_distributions != expected_operational_distributions:
            raise ValueError("Memory case operational distributions conflict with their pairs.")
        material = self.model_dump(mode="json", exclude={"revision"})
        if self.revision != _content_revision(material, "memory case comparison"):
            raise ValueError("Memory case comparison revision does not match its content.")
        return self


class MemoryVariantDisposition(_ReportModel):
    variant_id: StrictStr
    status: MemoryVariantDispositionStatus
    reasons: tuple[StrictStr, ...]
    comparable_pair_count: StrictInt = Field(ge=0)
    unavailable_pair_count: StrictInt = Field(ge=0)
    incomparable_pair_count: StrictInt = Field(ge=0)
    ranking_values: tuple[StrictFloat | None, ...] = ()


class MemoryVariantCostQualityReport(_ReportModel):
    """Canonical #300 evidence scoped to one candidate versus the baseline."""

    candidate_variant_id: StrictStr = Field(max_length=128)
    report: PairedCostQualityComparisonReport | None = None

    @field_validator("candidate_variant_id")
    @classmethod
    def validate_candidate_variant_id(cls, value: str, info) -> str:
        return _clean(value, info.field_name, maximum=128)

    @field_validator("report", mode="before")
    @classmethod
    def copy_report(cls, value: object) -> object:
        return revalidate_model_input(value, PairedCostQualityComparisonReport)


class MemoryVariantOperationalReport(_ReportModel):
    candidate_variant_id: StrictStr = Field(max_length=128)
    distributions: tuple[MemoryOperationalDistribution, ...]

    @field_validator("candidate_variant_id")
    @classmethod
    def validate_candidate_variant_id(cls, value: str, info) -> str:
        return _clean(value, info.field_name, maximum=128)

    @model_validator(mode="after")
    def validate_distributions(self) -> Self:
        dimensions = tuple(item.dimension for item in self.distributions)
        if dimensions != tuple(sorted(set(dimensions), key=str)):
            raise ValueError("Variant operational dimensions must be unique and sorted.")
        return self


class MemoryExperimentReport(_ReportModel):
    record_type: Literal["cayu.memory-experiment-report"] = "cayu.memory-experiment-report"
    schema_version: Literal[1] = MEMORY_EXPERIMENT_REPORT_SCHEMA_VERSION
    revision: StrictStr = Field(min_length=71, max_length=71)
    experiment_id: StrictStr
    baseline_variant_id: StrictStr
    repetitions: StrictInt = Field(ge=1, le=MEMORY_EXPERIMENT_REPORT_MAX_REPETITIONS)
    cases: tuple[MemoryExperimentCase, ...]
    variants: tuple[MemoryExperimentVariant, ...]
    metric_bindings: tuple[MemoryMetricBinding, ...]
    ranking: tuple[MemoryRankingTerm, ...]
    gates: MemoryExperimentGatePolicy
    rows: tuple[MemoryTrialReportRow, ...]
    comparisons: tuple[MemoryCaseComparison, ...]
    operational_summary: tuple[MemoryVariantOperationalReport, ...]
    cost_quality: tuple[MemoryVariantCostQualityReport, ...]
    selected_variant_id: StrictStr
    dispositions: tuple[MemoryVariantDisposition, ...]

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        case_keys = tuple((item.case_id, item.case_revision) for item in self.cases)
        case_ids = tuple(item.case_id for item in self.cases)
        variant_ids = tuple(item.variant_id for item in self.variants)
        if case_keys != tuple(sorted(case_keys)) or len(case_ids) != len(set(case_ids)):
            raise ValueError("Memory experiment report case IDs must be unique and cases sorted.")
        if variant_ids != tuple(sorted(set(variant_ids))):
            raise ValueError("Memory experiment report variants must be unique and sorted.")
        variants = {item.variant_id: item for item in self.variants}
        baseline = variants.get(self.baseline_variant_id)
        if baseline is None or baseline.spec.kind is not MemoryInterventionKind.AS_DECLARED:
            raise ValueError("Memory experiment report has no exact baseline variant.")
        _require_experiment_variant_execution_authority(
            self.variants,
            self.baseline_variant_id,
        )
        metric_roles = tuple(item.role for item in self.metric_bindings)
        if metric_roles != tuple(sorted(set(metric_roles), key=str)):
            raise ValueError("Memory experiment report metrics must be unique and sorted.")
        _validate_metric_policy(self.metric_bindings, self.ranking, self.gates)
        if not self.rows:
            raise ValueError("Memory experiment report requires its complete trial matrix.")
        expected_row_keys = tuple(
            (case.case_id, repetition, variant.variant_id)
            for case in self.cases
            for repetition in range(1, self.repetitions + 1)
            for variant in self.variants
        )
        if (
            tuple((row.case_id, row.repetition, row.variant_id) for row in self.rows)
            != expected_row_keys
        ):
            raise ValueError("Memory experiment report does not retain its complete matrix.")
        cases = dict(case_keys)
        execution_ids = tuple(row.execution_id for row in self.rows if row.execution_id is not None)
        if len(execution_ids) != len(set(execution_ids)):
            raise ValueError("Memory experiment execution identities must be unique.")
        trial_ids = tuple(row.trial_id for row in self.rows if row.trial_id is not None)
        if len(trial_ids) != len(set(trial_ids)):
            raise ValueError("Memory experiment trial identities must be unique.")
        for row in self.rows:
            variant = variants[row.variant_id]
            if (
                row.row_id != _row_id(row.case_id, row.repetition, row.variant_id)
                or row.case_revision != cases[row.case_id]
                or row.intervention_spec_fingerprint != variant.spec.fingerprint
                or row.snapshot_fingerprint != variant.spec.snapshot_fingerprint
                or row.execution_profile_revision != variant.execution_profile.revision
                or row.runtime_execution_profile_fingerprint
                != variant.execution_profile_binding.runtime_execution_profile.fingerprint
                or row.provider_name != variant.execution_profile.candidate.provider_name
                or row.model != variant.execution_profile.candidate.model
                or row.evaluator_fingerprint != variant.evaluator_fingerprint
                or (
                    row.published_result_revision is not None
                    and row.evidence_policy_revision
                    != variant.execution_profile.evidence_policy.revision
                )
            ):
                raise ValueError("Memory experiment row conflicts with its frozen variant.")
            binding = row.intervention_binding
            if binding is not None and (
                binding.spec.fingerprint != variant.spec.fingerprint
                or binding.operation.case_id != row.case_id
                or binding.operation.candidate_id != variant.candidate_id
                or binding.operation.trial_id != row.trial_id
                or binding.attribution_fingerprint != row.intervention_attribution_fingerprint
            ):
                raise ValueError(
                    "Memory experiment row binding conflicts with its frozen authority."
                )
            _validate_accounting_side_authority(
                row.accounting_side,
                variant=variant,
                experiment_id=self.experiment_id,
                case_revision=row.case_revision,
                repetition=row.repetition,
            )
            row_roles = tuple(item.role for item in row.metrics)
            if row_roles != metric_roles:
                raise ValueError("Memory experiment row does not retain every metric dimension.")
            for observation, binding in zip(row.metrics, self.metric_bindings, strict=True):
                if (
                    observation.assertion_id != binding.assertion_id
                    or observation.assertion_revision != binding.assertion_revision
                ):
                    raise ValueError("Memory experiment metric conflicts with its declaration.")
        accounting_attempt_ids = tuple(
            attempt.attempt_id
            for row in self.rows
            if row.accounting_side is not None
            for attempt in row.accounting_side.attempts
        )
        if len(accounting_attempt_ids) != len(set(accounting_attempt_ids)):
            raise ValueError(
                "Memory experiment accounting attempts cannot belong to multiple rows."
            )
        expected_comparison_keys = tuple(
            (case.case_id, variant.variant_id)
            for case in self.cases
            for variant in self.variants
            if variant.variant_id != self.baseline_variant_id
        )
        if (
            tuple((item.case_id, item.candidate_variant_id) for item in self.comparisons)
            != expected_comparison_keys
        ):
            raise ValueError("Memory experiment comparisons must cover every candidate case.")
        row_map = {(row.case_id, row.repetition, row.variant_id): row for row in self.rows}
        retained_cost_pairs = {
            pair.pair_id: pair
            for item in self.cost_quality
            if item.report is not None
            for pair in item.report.pairs
        }
        for comparison in self.comparisons:
            if comparison.case_revision != cases[comparison.case_id]:
                raise ValueError("Memory case comparison conflicts with its case authority.")
            if comparison.metric_roles != metric_roles:
                raise ValueError("Memory case comparison conflicts with declared report metrics.")
            if len(comparison.pairs) != self.repetitions:
                raise ValueError("Memory case comparison omits repeated trial pairs.")
            for pair in comparison.pairs:
                baseline = row_map[(pair.case_id, pair.repetition, self.baseline_variant_id)]
                candidate = row_map[(pair.case_id, pair.repetition, pair.candidate_variant_id)]
                expected_pair = _build_pair(
                    baseline,
                    candidate,
                    variants[self.baseline_variant_id],
                    variants[pair.candidate_variant_id],
                    baseline.intervention_binding,
                    candidate.intervention_binding,
                    retained_cost_pairs,
                )
                if pair != expected_pair:
                    raise ValueError(
                        "Memory trial pair classification conflicts with its exact evidence."
                    )
            expected_distributions = tuple(
                _distribution(binding.role, comparison.pairs) for binding in self.metric_bindings
            )
            if comparison.distributions != expected_distributions:
                raise ValueError("Memory case comparison distributions conflict with their pairs.")
            expected_operational_distributions = tuple(
                _operational_distribution(dimension, comparison.pairs)
                for dimension in sorted(MemoryOperationalDimension, key=str)
            )
            if comparison.operational_distributions != expected_operational_distributions:
                raise ValueError("Memory case operational distributions conflict with their pairs.")
        expected_operational_variants = tuple(
            item.variant_id for item in self.variants if item.variant_id != self.baseline_variant_id
        )
        if tuple(item.candidate_variant_id for item in self.operational_summary) != (
            expected_operational_variants
        ):
            raise ValueError("Memory experiment operational summaries must cover every candidate.")
        for summary in self.operational_summary:
            candidate_pairs = tuple(
                pair
                for comparison in self.comparisons
                if comparison.candidate_variant_id == summary.candidate_variant_id
                for pair in comparison.pairs
            )
            expected_distributions = tuple(
                _operational_distribution(dimension, candidate_pairs)
                for dimension in sorted(MemoryOperationalDimension, key=str)
            )
            if summary.distributions != expected_distributions:
                raise ValueError(
                    "Memory experiment operational summary conflicts with trial pairs."
                )
        if tuple(item.variant_id for item in self.dispositions) != tuple(
            item.variant_id for item in self.variants
        ):
            raise ValueError("Memory experiment dispositions must cover every variant.")
        selected = tuple(
            item.variant_id
            for item in self.dispositions
            if item.status is MemoryVariantDispositionStatus.SELECTED
        )
        if selected != (self.selected_variant_id,):
            raise ValueError("Memory experiment report requires one selected disposition.")
        expected_selected, expected_dispositions = _select_memory_variant(
            self.variants,
            self.baseline_variant_id,
            self.comparisons,
            self.cost_quality,
            self.metric_bindings,
            self.ranking,
            self.gates,
        )
        if (
            self.selected_variant_id != expected_selected
            or self.dispositions != expected_dispositions
        ):
            raise ValueError(
                "Memory experiment recommendation conflicts with its retained evidence."
            )
        expected_cost_variants = tuple(
            item.variant_id for item in self.variants if item.variant_id != self.baseline_variant_id
        )
        if tuple(item.candidate_variant_id for item in self.cost_quality) != (
            expected_cost_variants
        ):
            raise ValueError("Memory experiment accounting must cover every candidate in order.")
        for scoped in self.cost_quality:
            expected_inputs: list[PairedCostQualityPair] = []
            for case in self.cases:
                for repetition in range(1, self.repetitions + 1):
                    baseline_row = row_map[(case.case_id, repetition, self.baseline_variant_id)]
                    candidate_row = row_map[(case.case_id, repetition, scoped.candidate_variant_id)]
                    expected_inputs.append(
                        PairedCostQualityPair(
                            pair_id=_pair_id(
                                case.case_id,
                                repetition,
                                scoped.candidate_variant_id,
                            ),
                            baseline=baseline_row.accounting_side,
                            candidate=candidate_row.accounting_side,
                        )
                    )
            expected_report = compare_paired_cost_quality(
                PairedCostQualityComparisonRequest(pairs=tuple(expected_inputs))
            )
            if scoped.report != expected_report:
                raise ValueError("Memory experiment accounting conflicts with its trial evidence.")
        accounting_pairs = {
            pair.pair_id: pair.accounting_pair
            for comparison in self.comparisons
            for pair in comparison.pairs
            if pair.accounting_pair is not None
        }
        if retained_cost_pairs != accounting_pairs:
            raise ValueError("Memory experiment accounting conflicts with trial pairs.")
        material = self.model_dump(mode="json", exclude={"revision"})
        if self.revision != _content_revision(material, "memory experiment report"):
            raise ValueError("Memory experiment report revision does not match its content.")
        if (
            compact_json_utf8_size(self.model_dump(mode="json"))
            > MEMORY_EXPERIMENT_REPORT_MAX_BYTES
        ):
            raise ValueError("Memory experiment report exceeds its byte bound.")
        return self


def _validate_trial_evidence(
    evidence: MemoryExperimentTrialEvidence,
    variant: MemoryExperimentVariant,
    experiment_id: str,
    published_results: dict[str, CorpusExecutionResult],
) -> None:
    execution = evidence.execution
    _require_execution_variant_authority(
        execution,
        case_id=evidence.case_id,
        case_revision=evidence.case_revision,
        variant=variant,
    )
    binding = evidence.intervention_binding
    _require_terminal_execution_binding(execution, binding)
    published = _published_trial_for_evidence(evidence, published_results)
    if published is not None:
        if execution.eval_result_revision is None:
            raise ValueError("Published trial has no durable source result identity.")
        if (
            published.trial_number != evidence.repetition
            or published.source_trial_revision != execution.eval_result_revision
        ):
            raise ValueError("Published trial conflicts with its execution result.")
        if binding is not None:
            root_source = next(
                (
                    source
                    for source in published.memory_attribution.sources
                    if source.source.tree_path == ()
                ),
                None,
            )
            if (
                root_source is not None
                and root_source.attribution is not None
                and binding.attribution_fingerprint
                != eval_memory_attribution_fingerprint(root_source.attribution)
            ):
                raise ValueError(
                    "Published memory attribution conflicts with its intervention binding."
                )
    _validate_accounting_side_authority(
        evidence.accounting_side,
        variant=variant,
        experiment_id=experiment_id,
        case_revision=evidence.case_revision,
        repetition=evidence.repetition,
    )


def _validate_accounting_side_authority(
    side: PairedCostQualitySide | None,
    *,
    variant: MemoryExperimentVariant,
    experiment_id: str,
    case_revision: str,
    repetition: int,
) -> None:
    if side is not None:
        profile = variant.execution_profile.candidate
        if (
            side.strategy_id != variant.variant_id
            or side.workload_id != experiment_id
            or side.task_id != memory_experiment_accounting_task_id(experiment_id)
            or side.source_id
            != memory_experiment_accounting_source_id(
                case_revision,
                repetition,
            )
            or any(
                (
                    attempt.operation.value not in {"evaluation", "comparison_control"}
                    and (
                        (
                            attempt.provider_name is not None
                            and attempt.provider_name != profile.provider_name
                        )
                        or (attempt.model is not None and attempt.model != profile.model)
                    )
                )
                for attempt in side.attempts
            )
        ):
            raise ValueError("Accounting evidence conflicts with its exact trial authority.")


def _published_trial_for_evidence(
    evidence: MemoryExperimentTrialEvidence,
    published_results: dict[str, CorpusExecutionResult],
) -> PublishedEvalTrialResult | None:
    if evidence.published_result_revision is None:
        return None
    result = published_results.get(evidence.published_result_revision)
    if result is None:
        raise ValueError("Trial evidence references an unavailable published result.")
    run = result.run
    case = next(
        (
            item
            for item in run.cases
            if item.case_id == evidence.case_id and item.case_revision == evidence.case_revision
        ),
        None,
    )
    if case is None:
        raise ValueError("Published run does not contain the exact report case.")
    return next(
        (item for item in case.trials if item.trial_number == evidence.repetition),
        None,
    )


def _row_id(case_id: str, repetition: int, variant_id: str) -> str:
    return _content_revision(
        {"case_id": case_id, "repetition": repetition, "variant_id": variant_id},
        "memory experiment row identity",
    )


def memory_experiment_accounting_task_id(experiment_id: str) -> str:
    """Return the shared #300 task identity for one experiment cohort."""

    normalized_experiment_id = _clean(experiment_id, "experiment_id", maximum=128)
    return _content_revision(
        {"experiment_id": normalized_experiment_id},
        "memory experiment accounting task identity",
    )


def memory_experiment_accounting_source_id(case_revision: str, repetition: int) -> str:
    """Return the exact #300 source identity for one repeated case row."""

    case_revision = _revision(case_revision, "case_revision")
    if (
        type(repetition) is not int
        or not 1 <= repetition <= MEMORY_EXPERIMENT_REPORT_MAX_REPETITIONS
    ):
        raise ValueError("repetition must be a supported positive integer.")
    return _content_revision(
        {"case_revision": case_revision, "repetition": repetition},
        "memory experiment accounting source identity",
    )


def _pair_id(case_id: str, repetition: int, candidate_variant_id: str) -> str:
    return _content_revision(
        {
            "case_id": case_id,
            "repetition": repetition,
            "candidate_variant_id": candidate_variant_id,
        },
        "memory experiment pair identity",
    )


def _trial_availability(
    execution: MemoryInterventionExecutionRecord | None,
    published: PublishedEvalTrialResult | None,
) -> MemoryTrialAvailability:
    return _trial_availability_from_statuses(
        None if execution is None else execution.status,
        published=published is not None,
        published_status=None if published is None else published.status,
        attribution_status=(
            None if published is None else published.memory_attribution.completeness
        ),
    )


def _trial_availability_from_statuses(
    execution_status: MemoryInterventionExecutionStatus | None,
    *,
    published: bool,
    published_status: PublishedOutcome | None,
    attribution_status: EvalMemoryEvidenceCompleteness | None,
) -> MemoryTrialAvailability:
    if published != (published_status is not None and attribution_status is not None):
        raise ValueError("Published trial status evidence is incomplete.")
    if execution_status is None:
        return MemoryTrialAvailability.MISSING
    if execution_status is MemoryInterventionExecutionStatus.ACTIVE:
        raise ValueError("Memory experiment reports require terminal trial executions.")
    mapped = {
        MemoryInterventionExecutionStatus.FAILED: MemoryTrialAvailability.FAILED,
        MemoryInterventionExecutionStatus.CANCELLED: MemoryTrialAvailability.CANCELLED,
        MemoryInterventionExecutionStatus.TIMED_OUT: MemoryTrialAvailability.TIMED_OUT,
        MemoryInterventionExecutionStatus.OUTCOME_UNKNOWN: (
            MemoryTrialAvailability.OUTCOME_UNKNOWN
        ),
        MemoryInterventionExecutionStatus.CONFLICTING: MemoryTrialAvailability.CONFLICTING,
        MemoryInterventionExecutionStatus.INDETERMINATE: (MemoryTrialAvailability.INDETERMINATE),
    }
    if execution_status in mapped:
        return mapped[execution_status]
    if execution_status is not MemoryInterventionExecutionStatus.COMPLETED or not published:
        return (
            MemoryTrialAvailability.UNMATCHED
            if execution_status is MemoryInterventionExecutionStatus.COMPLETED
            else MemoryTrialAvailability.MISSING
        )
    if published_status == "unavailable":
        return MemoryTrialAvailability.UNAVAILABLE
    if published_status == "error":
        return MemoryTrialAvailability.FAILED
    if attribution_status is not EvalMemoryEvidenceCompleteness.COMPLETE:
        return MemoryTrialAvailability.UNAVAILABLE
    return MemoryTrialAvailability.AVAILABLE


def _observations(
    published: PublishedEvalTrialResult | None,
    bindings: tuple[MemoryMetricBinding, ...],
) -> tuple[MemoryMetricObservation, ...]:
    assertions = (
        {} if published is None else {item.assertion_id: item for item in published.assertions}
    )
    observations: list[MemoryMetricObservation] = []
    for binding in bindings:
        assertion = assertions.get(binding.assertion_id)
        if assertion is None or assertion.assertion_revision != binding.assertion_revision:
            observations.append(
                MemoryMetricObservation(
                    role=binding.role,
                    assertion_id=binding.assertion_id,
                    assertion_revision=binding.assertion_revision,
                    availability=MemoryMetricAvailability.UNAVAILABLE,
                )
            )
            continue
        availability = (
            MemoryMetricAvailability.AVAILABLE
            if assertion.score is not None
            else MemoryMetricAvailability.ERROR
            if assertion.outcome == "error"
            else MemoryMetricAvailability.UNAVAILABLE
        )
        evaluator_key, evaluator_implementation_revision = _published_evaluator_identity(
            assertion.detail
        )
        observations.append(
            MemoryMetricObservation(
                role=binding.role,
                assertion_id=binding.assertion_id,
                assertion_revision=binding.assertion_revision,
                availability=availability,
                outcome=assertion.outcome,
                value=assertion.score,
                evaluator_key=evaluator_key,
                evaluator_implementation_revision=evaluator_implementation_revision,
            )
        )
    return tuple(observations)


def _published_evaluator_identity(detail: object) -> tuple[str | None, str | None]:
    if isinstance(detail, PublishedModelJudgeDetail):
        return detail.evaluator_key, detail.evaluator_implementation_revision
    if isinstance(detail, PublishedStructuredModelJudgeDetail):
        return detail.judge_profile.key, detail.judge_profile.implementation_revision
    return None, None


def _build_row(
    case: MemoryExperimentCase,
    repetition: int,
    variant: MemoryExperimentVariant,
    evidence: MemoryExperimentTrialEvidence | None,
    published_result: CorpusExecutionResult | None,
    published: PublishedEvalTrialResult | None,
    published_run_id: str | None,
    metric_bindings: tuple[MemoryMetricBinding, ...],
) -> MemoryTrialReportRow:
    execution = None if evidence is None else evidence.execution
    attribution = None if published is None else published.memory_attribution
    usage = None if published is None else published.usage
    return MemoryTrialReportRow(
        row_id=_row_id(case.case_id, repetition, variant.variant_id),
        case_id=case.case_id,
        case_revision=case.case_revision,
        repetition=repetition,
        variant_id=variant.variant_id,
        intervention_spec_fingerprint=variant.spec.fingerprint,
        snapshot_fingerprint=variant.spec.snapshot_fingerprint,
        execution_profile_revision=variant.execution_profile.revision,
        runtime_execution_profile_fingerprint=(
            variant.execution_profile_binding.runtime_execution_profile.fingerprint
        ),
        provider_name=variant.execution_profile.candidate.provider_name,
        model=variant.execution_profile.candidate.model,
        evaluator_fingerprint=variant.evaluator_fingerprint,
        execution_id=None if execution is None else execution.execution_id,
        trial_id=None if execution is None else execution.trial_id,
        execution_revision=None if execution is None else execution.revision,
        final_binding_fingerprint=(
            None if execution is None else execution.final_binding_fingerprint
        ),
        execution_binding_lineage_revision=(
            None
            if evidence is None
            else _require_terminal_execution_binding(
                evidence.execution,
                evidence.intervention_binding,
            )
        ),
        intervention_binding=(None if evidence is None else evidence.intervention_binding),
        published_result_revision=(
            None if evidence is None else evidence.published_result_revision
        ),
        published_run_id=published_run_id,
        source_trial_revision=None if published is None else published.source_trial_revision,
        corpus_revision=(
            None if published_result is None else published_result.run.corpus_revision
        ),
        suite_id=None if published_result is None else published_result.run.suite_id,
        suite_revision=(None if published_result is None else published_result.run.suite_revision),
        evidence_policy_revision=(
            None if published_result is None else published_result.run.evidence_policy_revision
        ),
        pricing_profile_fingerprint=(
            None if published_result is None else published_result.run.pricing_profile_fingerprint
        ),
        intervention_attribution_fingerprint=(
            None
            if evidence is None or evidence.intervention_binding is None
            else evidence.intervention_binding.attribution_fingerprint
        ),
        attribution_evidence_revision=(None if attribution is None else attribution.revision),
        attribution_status=None if attribution is None else attribution.completeness,
        availability=_trial_availability(execution, published),
        execution_status=None if execution is None else execution.status,
        published_status=None if published is None else published.status,
        metrics=_observations(published, metric_bindings),
        duration_ms=None if published is None else published.duration_ms,
        total_tokens=None if usage is None else usage.total_tokens,
        accounting_side=None if evidence is None else evidence.accounting_side,
        memory_overhead=None if evidence is None else evidence.memory_overhead,
    )


def _metric_deltas(
    baseline: MemoryTrialReportRow,
    candidate: MemoryTrialReportRow,
) -> tuple[MemoryMetricDelta, ...]:
    left = {item.role: item for item in baseline.metrics}
    right = {item.role: item for item in candidate.metrics}
    result: list[MemoryMetricDelta] = []
    for role in sorted(set(left) & set(right), key=str):
        baseline_item = left[role]
        candidate_item = right[role]
        if (
            baseline_item.value is None
            or candidate_item.value is None
            or baseline_item.assertion_revision != candidate_item.assertion_revision
            or baseline_item.evaluator_key != candidate_item.evaluator_key
            or baseline_item.evaluator_implementation_revision
            != candidate_item.evaluator_implementation_revision
        ):
            continue
        result.append(
            MemoryMetricDelta(
                role=role,
                baseline=baseline_item.value,
                candidate=candidate_item.value,
                delta=candidate_item.value - baseline_item.value,
            )
        )
    return tuple(result)


def _operational_value(
    row: MemoryTrialReportRow,
    dimension: MemoryOperationalDimension,
) -> int | None:
    if dimension is MemoryOperationalDimension.LATENCY_MS:
        return row.duration_ms
    if dimension is MemoryOperationalDimension.TOTAL_TOKENS:
        return row.total_tokens
    overhead = row.memory_overhead
    if overhead is None:
        return None
    if dimension is MemoryOperationalDimension.MEMORY_PREPARATION_DURATION_MS:
        return overhead.preparation_duration_ms
    if dimension is MemoryOperationalDimension.MEMORY_CONTEXT_TOKENS:
        return overhead.context_tokens
    return overhead.context_bytes


def _operational_deltas(
    baseline: MemoryTrialReportRow,
    candidate: MemoryTrialReportRow,
) -> tuple[MemoryOperationalDelta, ...]:
    result: list[MemoryOperationalDelta] = []
    for dimension in sorted(MemoryOperationalDimension, key=str):
        baseline_value = _operational_value(baseline, dimension)
        candidate_value = _operational_value(candidate, dimension)
        if baseline_value is None or candidate_value is None:
            continue
        result.append(
            MemoryOperationalDelta(
                dimension=dimension,
                baseline=baseline_value,
                candidate=candidate_value,
                delta=candidate_value - baseline_value,
            )
        )
    return tuple(result)


def _distribution(
    role: MemoryMetricRole,
    pairs: tuple[MemoryTrialPairComparison, ...],
) -> MemoryMetricDistribution:
    deltas = tuple(
        delta
        for pair in pairs
        if pair.status is MemoryPairStatus.COMPARABLE
        for delta in pair.metric_deltas
        if delta.role is role
    )
    values = tuple(item.delta for item in deltas)
    return MemoryMetricDistribution(
        role=role,
        pair_count=len(pairs),
        available_count=len(values),
        unavailable_count=sum(
            pair.status is MemoryPairStatus.UNAVAILABLE
            or (
                pair.status is MemoryPairStatus.COMPARABLE
                and not any(delta.role is role for delta in pair.metric_deltas)
            )
            for pair in pairs
        ),
        incomparable_count=sum(pair.status is MemoryPairStatus.INCOMPARABLE for pair in pairs),
        baseline_values=tuple(item.baseline for item in deltas),
        candidate_values=tuple(item.candidate for item in deltas),
        deltas=values,
        mean_delta=None if not values else sum(values) / len(values),
        median_delta=None if not values else float(median(values)),
    )


def _operational_distribution(
    dimension: MemoryOperationalDimension,
    pairs: tuple[MemoryTrialPairComparison, ...],
) -> MemoryOperationalDistribution:
    deltas = tuple(
        delta
        for pair in pairs
        if pair.status is MemoryPairStatus.COMPARABLE
        for delta in pair.operational_deltas
        if delta.dimension is dimension
    )
    values = tuple(item.delta for item in deltas)
    return MemoryOperationalDistribution(
        dimension=dimension,
        pair_count=len(pairs),
        available_count=len(values),
        unavailable_count=sum(
            pair.status is MemoryPairStatus.UNAVAILABLE
            or (
                pair.status is MemoryPairStatus.COMPARABLE
                and not any(delta.dimension is dimension for delta in pair.operational_deltas)
            )
            for pair in pairs
        ),
        incomparable_count=sum(pair.status is MemoryPairStatus.INCOMPARABLE for pair in pairs),
        baseline_values=tuple(item.baseline for item in deltas),
        candidate_values=tuple(item.candidate for item in deltas),
        deltas=values,
        mean_delta=None if not values else sum(values) / len(values),
        median_delta=None if not values else float(median(values)),
    )


def _build_pair(
    baseline: MemoryTrialReportRow,
    candidate: MemoryTrialReportRow,
    baseline_variant: MemoryExperimentVariant,
    candidate_variant: MemoryExperimentVariant,
    baseline_binding: MemoryInterventionTrialBinding | None,
    candidate_binding: MemoryInterventionTrialBinding | None,
    cost_pairs: dict[str, PairedCostQualityPairReport],
) -> MemoryTrialPairComparison:
    pair_id = _pair_id(candidate.case_id, candidate.repetition, candidate.variant_id)
    reasons: set[str] = set()
    comparability = None
    if (
        baseline.availability is not MemoryTrialAvailability.AVAILABLE
        or candidate.availability is not MemoryTrialAvailability.AVAILABLE
        or baseline_binding is None
        or candidate_binding is None
        or baseline.attribution_status is not EvalMemoryEvidenceCompleteness.COMPLETE
        or candidate.attribution_status is not EvalMemoryEvidenceCompleteness.COMPLETE
    ):
        status = MemoryPairStatus.UNAVAILABLE
        reasons.add("trial_evidence_unavailable")
    else:
        comparability = MemoryInterventionComparability.create(
            baseline=baseline_binding,
            intervention=candidate_binding,
        )
        if comparability.status is MemoryInterventionComparabilityStatus.INCOMPARABLE:
            status = MemoryPairStatus.INCOMPARABLE
            reasons.update(str(reason) for reason in comparability.mismatch_reasons)
        elif not _generic_experiment_identity_matches(
            baseline,
            candidate,
            baseline_variant,
            candidate_variant,
        ):
            status = MemoryPairStatus.INCOMPARABLE
            reasons.add("generic_experiment_identity")
        elif any(
            (
                left.evaluator_key,
                left.evaluator_implementation_revision,
            )
            != (
                right.evaluator_key,
                right.evaluator_implementation_revision,
            )
            for left, right in zip(baseline.metrics, candidate.metrics, strict=True)
        ):
            status = MemoryPairStatus.INCOMPARABLE
            reasons.add("evaluator_implementation_identity")
        else:
            status = MemoryPairStatus.COMPARABLE
    accounting = cost_pairs.get(pair_id)
    if (
        accounting is not None
        and accounting.status is CostQualityComparisonStatus.MEASURED_UNMATCHED
    ):
        reasons.add("accounting_not_comparable")
        if status is MemoryPairStatus.COMPARABLE:
            status = MemoryPairStatus.INCOMPARABLE
    elif accounting is not None and accounting.status is not CostQualityComparisonStatus.VERIFIED:
        reasons.add("accounting_not_verified")
    return MemoryTrialPairComparison(
        pair_id=pair_id,
        case_id=candidate.case_id,
        case_revision=candidate.case_revision,
        repetition=candidate.repetition,
        baseline_variant_id=baseline.variant_id,
        candidate_variant_id=candidate.variant_id,
        baseline_row_id=baseline.row_id,
        candidate_row_id=candidate.row_id,
        status=status,
        reasons=tuple(sorted(reasons)),
        memory_comparability_fingerprint=(
            None if comparability is None else comparability.fingerprint
        ),
        metric_deltas=(
            _metric_deltas(baseline, candidate) if status is MemoryPairStatus.COMPARABLE else ()
        ),
        operational_deltas=(
            _operational_deltas(baseline, candidate)
            if status is MemoryPairStatus.COMPARABLE
            else ()
        ),
        accounting_pair=accounting,
    )


def _variant_disposition(
    variant: MemoryExperimentVariant,
    baseline_variant_id: str,
    comparisons: tuple[MemoryCaseComparison, ...],
    scoped_cost_quality: MemoryVariantCostQualityReport | None,
    bindings: tuple[MemoryMetricBinding, ...],
    ranking: tuple[MemoryRankingTerm, ...],
    gates: MemoryExperimentGatePolicy,
) -> tuple[MemoryVariantDisposition, tuple[float, ...] | None]:
    if variant.variant_id == baseline_variant_id:
        return (
            MemoryVariantDisposition(
                variant_id=variant.variant_id,
                status=MemoryVariantDispositionStatus.SELECTED,
                reasons=("baseline_default",),
                comparable_pair_count=0,
                unavailable_pair_count=0,
                incomparable_pair_count=0,
            ),
            (),
        )
    pairs = tuple(
        pair
        for comparison in comparisons
        if comparison.candidate_variant_id == variant.variant_id
        for pair in comparison.pairs
    )
    comparable = tuple(pair for pair in pairs if pair.status is MemoryPairStatus.COMPARABLE)
    unavailable_count = sum(pair.status is MemoryPairStatus.UNAVAILABLE for pair in pairs)
    incomparable_count = sum(pair.status is MemoryPairStatus.INCOMPARABLE for pair in pairs)
    reasons: set[str] = set()
    if not pairs or not comparable:
        status = (
            MemoryVariantDispositionStatus.INCOMPARABLE
            if any(pair.status is MemoryPairStatus.INCOMPARABLE for pair in pairs)
            else MemoryVariantDispositionStatus.UNAVAILABLE
        )
        reasons.add("no_comparable_pairs")
        return (
            MemoryVariantDisposition(
                variant_id=variant.variant_id,
                status=status,
                reasons=tuple(sorted(reasons)),
                comparable_pair_count=len(comparable),
                unavailable_pair_count=unavailable_count,
                incomparable_pair_count=incomparable_count,
            ),
            None,
        )
    if len(comparable) != len(pairs):
        incomparable = any(pair.status is MemoryPairStatus.INCOMPARABLE for pair in pairs)
        return (
            MemoryVariantDisposition(
                variant_id=variant.variant_id,
                status=(
                    MemoryVariantDispositionStatus.INCOMPARABLE
                    if incomparable
                    else MemoryVariantDispositionStatus.UNAVAILABLE
                ),
                reasons=("incomplete_repeated_trial_matrix",),
                comparable_pair_count=len(comparable),
                unavailable_pair_count=unavailable_count,
                incomparable_pair_count=incomparable_count,
            ),
            None,
        )
    if len(comparable) < gates.minimum_comparable_pairs:
        reasons.add("insufficient_comparable_pairs")
    deltas_by_role = {
        role: tuple(
            delta.delta for pair in comparable for delta in pair.metric_deltas if delta.role is role
        )
        for role in (binding.role for binding in bindings)
    }
    for role in gates.required_metric_roles:
        if len(deltas_by_role.get(role, ())) != len(comparable):
            reasons.add(f"required_metric_unavailable:{role.value}")
    candidate_values = {
        role: tuple(
            delta.candidate
            for pair in comparable
            for delta in pair.metric_deltas
            if delta.role is role
        )
        for role in (binding.role for binding in bindings)
    }
    for gate in gates.metric_gates:
        values = candidate_values.get(gate.role, ())
        if len(values) != len(comparable):
            reasons.add(f"gate_evidence_unavailable:{gate.role.value}")
        elif (gate.minimum is not None and min(values) < gate.minimum) or (
            gate.maximum is not None and max(values) > gate.maximum
        ):
            reasons.add(f"gate_failed:{gate.role.value}")
    aggregate = (
        None
        if scoped_cost_quality is None or scoped_cost_quality.report is None
        else scoped_cost_quality.report.aggregate
    )
    complete_verified_cost = (
        aggregate is not None
        and aggregate.pair_count == len(pairs)
        and aggregate.status is CostQualityComparisonStatus.VERIFIED
    )
    if gates.require_priced_cost and not complete_verified_cost:
        reasons.add("priced_cost_unavailable")
    if gates.maximum_candidate_cost is not None:
        if (
            not complete_verified_cost
            or aggregate is None
            or aggregate.candidate_cost is None
            or aggregate.currency != gates.cost_currency
        ):
            reasons.add("budget_evidence_unavailable")
        elif aggregate.candidate_cost > gates.maximum_candidate_cost:
            reasons.add("budget_exceeded")
    if reasons:
        return (
            MemoryVariantDisposition(
                variant_id=variant.variant_id,
                status=MemoryVariantDispositionStatus.REJECTED,
                reasons=tuple(sorted(reasons)),
                comparable_pair_count=len(comparable),
                unavailable_pair_count=unavailable_count,
                incomparable_pair_count=incomparable_count,
            ),
            None,
        )
    if not ranking:
        return (
            MemoryVariantDisposition(
                variant_id=variant.variant_id,
                status=MemoryVariantDispositionStatus.NOT_BETTER,
                reasons=("no_task_quality_ranking",),
                comparable_pair_count=len(comparable),
                unavailable_pair_count=unavailable_count,
                incomparable_pair_count=incomparable_count,
            ),
            None,
        )
    ranking_values: list[float] = []
    for term in ranking:
        values = tuple(
            (
                delta.delta
                if term.direction is MemoryMetricDirection.HIGHER_IS_BETTER
                else -delta.delta
            )
            for pair in comparable
            for delta in pair.metric_deltas
            if delta.role is term.role
        )
        if len(values) != len(comparable):
            return (
                MemoryVariantDisposition(
                    variant_id=variant.variant_id,
                    status=MemoryVariantDispositionStatus.UNAVAILABLE,
                    reasons=(f"ranking_metric_unavailable:{term.role.value}",),
                    comparable_pair_count=len(comparable),
                    unavailable_pair_count=unavailable_count,
                    incomparable_pair_count=incomparable_count,
                ),
                None,
            )
        ranking_values.append(sum(values) / len(values))
    return (
        MemoryVariantDisposition(
            variant_id=variant.variant_id,
            status=MemoryVariantDispositionStatus.NOT_BETTER,
            reasons=("eligible_for_ranking",),
            comparable_pair_count=len(comparable),
            unavailable_pair_count=unavailable_count,
            incomparable_pair_count=incomparable_count,
            ranking_values=tuple(ranking_values),
        ),
        tuple(ranking_values),
    )


def _select_memory_variant(
    variants: tuple[MemoryExperimentVariant, ...],
    baseline_variant_id: str,
    comparisons: tuple[MemoryCaseComparison, ...],
    cost_quality: tuple[MemoryVariantCostQualityReport, ...],
    bindings: tuple[MemoryMetricBinding, ...],
    ranking: tuple[MemoryRankingTerm, ...],
    gates: MemoryExperimentGatePolicy,
) -> tuple[str, tuple[MemoryVariantDisposition, ...]]:
    dispositions: list[MemoryVariantDisposition] = []
    ranks: dict[str, tuple[float, ...]] = {}
    cost_quality_by_variant = {item.candidate_variant_id: item for item in cost_quality}
    for variant in variants:
        disposition, rank = _variant_disposition(
            variant,
            baseline_variant_id,
            comparisons,
            cost_quality_by_variant.get(variant.variant_id),
            bindings,
            ranking,
            gates,
        )
        dispositions.append(disposition)
        if rank is not None and variant.variant_id != baseline_variant_id:
            ranks[variant.variant_id] = rank
    rankable = tuple(
        (rank, variant_id) for variant_id, rank in ranks.items() if rank > tuple(0.0 for _ in rank)
    )
    selected = baseline_variant_id
    best_rank: tuple[float, ...] | None = None
    if rankable:
        best_rank = max(item[0] for item in rankable)
        selected = min(item[1] for item in rankable if item[0] == best_rank)
    final: list[MemoryVariantDisposition] = []
    for item in dispositions:
        if item.variant_id == selected:
            final.append(
                item.model_copy(
                    update={
                        "status": MemoryVariantDispositionStatus.SELECTED,
                        "reasons": (
                            ("baseline_default",)
                            if selected == baseline_variant_id
                            else ("highest_eligible_lexicographic_rank",)
                        ),
                    }
                )
            )
            continue
        if item.variant_id == baseline_variant_id:
            final.append(
                item.model_copy(
                    update={
                        "status": MemoryVariantDispositionStatus.BASELINE_SUPERSEDED,
                        "reasons": ("higher_ranked_candidate_selected",),
                    }
                )
            )
            continue
        rank = ranks.get(item.variant_id)
        if rank is not None and rank > tuple(0.0 for _ in rank):
            assert best_rank is not None
            final.append(
                item.model_copy(
                    update={
                        "status": MemoryVariantDispositionStatus.ELIGIBLE_NOT_SELECTED,
                        "reasons": (
                            ("eligible_rank_tie_broken_by_variant_id",)
                            if rank == best_rank
                            else ("lower_eligible_lexicographic_rank",)
                        ),
                    }
                )
            )
            continue
        if rank is not None:
            final.append(
                item.model_copy(
                    update={
                        "status": MemoryVariantDispositionStatus.NOT_BETTER,
                        "reasons": ("not_better_than_baseline",),
                    }
                )
            )
            continue
        final.append(item)
    return selected, tuple(final)


def build_memory_experiment_report(
    request: MemoryExperimentReportRequest,
) -> MemoryExperimentReport:
    """Build one complete deterministic report without launching work."""

    if type(request) is not MemoryExperimentReportRequest:
        raise TypeError("request must be an exact MemoryExperimentReportRequest.")
    request = MemoryExperimentReportRequest.model_validate(
        request.model_dump(mode="python", round_trip=True, warnings="none")
    )
    evidence = {(item.case_id, item.repetition, item.variant_id): item for item in request.trials}
    published_results = {item.result.revision: item.result for item in request.published_results}
    published_run_ids = {item.result.revision: item.run_id for item in request.published_results}
    built_rows: list[MemoryTrialReportRow] = []
    for case in request.cases:
        for repetition in range(1, request.repetitions + 1):
            for variant in request.variants:
                item = evidence.get((case.case_id, repetition, variant.variant_id))
                result = (
                    None
                    if item is None or item.published_result_revision is None
                    else published_results[item.published_result_revision]
                )
                built_rows.append(
                    _build_row(
                        case,
                        repetition,
                        variant,
                        item,
                        result,
                        (
                            None
                            if item is None
                            else _published_trial_for_evidence(item, published_results)
                        ),
                        (
                            None
                            if item is None or item.published_result_revision is None
                            else published_run_ids[item.published_result_revision]
                        ),
                        request.metric_bindings,
                    )
                )
    rows = tuple(built_rows)
    row_map = {(row.case_id, row.repetition, row.variant_id): row for row in rows}
    cost_quality: list[MemoryVariantCostQualityReport] = []
    cost_pairs_by_id: dict[str, PairedCostQualityPairReport] = {}
    for variant in request.variants:
        if variant.variant_id == request.baseline_variant_id:
            continue
        cost_inputs: list[PairedCostQualityPair] = []
        for case in request.cases:
            for repetition in range(1, request.repetitions + 1):
                baseline_evidence = evidence.get(
                    (case.case_id, repetition, request.baseline_variant_id)
                )
                candidate_evidence = evidence.get((case.case_id, repetition, variant.variant_id))
                baseline_side = (
                    None if baseline_evidence is None else baseline_evidence.accounting_side
                )
                candidate_side = (
                    None if candidate_evidence is None else candidate_evidence.accounting_side
                )
                cost_inputs.append(
                    PairedCostQualityPair(
                        pair_id=_pair_id(case.case_id, repetition, variant.variant_id),
                        baseline=baseline_side,
                        candidate=candidate_side,
                    )
                )
        cost_report = compare_paired_cost_quality(
            PairedCostQualityComparisonRequest(pairs=tuple(cost_inputs))
        )
        cost_quality.append(
            MemoryVariantCostQualityReport(
                candidate_variant_id=variant.variant_id,
                report=cost_report,
            )
        )
        cost_pairs_by_id.update((item.pair_id, item) for item in cost_report.pairs)
    comparisons: list[MemoryCaseComparison] = []
    for case in request.cases:
        for variant in request.variants:
            if variant.variant_id == request.baseline_variant_id:
                continue
            pairs = tuple(
                _build_pair(
                    row_map[(case.case_id, repetition, request.baseline_variant_id)],
                    row_map[(case.case_id, repetition, variant.variant_id)],
                    next(
                        item
                        for item in request.variants
                        if item.variant_id == request.baseline_variant_id
                    ),
                    variant,
                    (
                        None
                        if (
                            item := evidence.get(
                                (case.case_id, repetition, request.baseline_variant_id)
                            )
                        )
                        is None
                        else item.intervention_binding
                    ),
                    (
                        None
                        if (item := evidence.get((case.case_id, repetition, variant.variant_id)))
                        is None
                        else item.intervention_binding
                    ),
                    cost_pairs_by_id,
                )
                for repetition in range(1, request.repetitions + 1)
            )
            distributions = tuple(
                _distribution(binding.role, pairs) for binding in request.metric_bindings
            )
            operational_distributions = tuple(
                _operational_distribution(dimension, pairs)
                for dimension in sorted(MemoryOperationalDimension, key=str)
            )
            material = {
                "schema_version": 1,
                "case_id": case.case_id,
                "case_revision": case.case_revision,
                "candidate_variant_id": variant.variant_id,
                "pairs": [item.model_dump(mode="json") for item in pairs],
                "metric_roles": [binding.role.value for binding in request.metric_bindings],
                "distributions": [item.model_dump(mode="json") for item in distributions],
                "operational_distributions": [
                    item.model_dump(mode="json") for item in operational_distributions
                ],
            }
            comparisons.append(
                MemoryCaseComparison(
                    revision=_content_revision(material, "memory case comparison"),
                    case_id=case.case_id,
                    case_revision=case.case_revision,
                    candidate_variant_id=variant.variant_id,
                    pairs=pairs,
                    metric_roles=tuple(binding.role for binding in request.metric_bindings),
                    distributions=distributions,
                    operational_distributions=operational_distributions,
                )
            )
    comparison_tuple = tuple(comparisons)
    operational_summary = tuple(
        MemoryVariantOperationalReport(
            candidate_variant_id=variant.variant_id,
            distributions=tuple(
                _operational_distribution(
                    dimension,
                    tuple(
                        pair
                        for comparison in comparison_tuple
                        if comparison.candidate_variant_id == variant.variant_id
                        for pair in comparison.pairs
                    ),
                )
                for dimension in sorted(MemoryOperationalDimension, key=str)
            ),
        )
        for variant in request.variants
        if variant.variant_id != request.baseline_variant_id
    )
    selected, final_dispositions = _select_memory_variant(
        request.variants,
        request.baseline_variant_id,
        comparison_tuple,
        tuple(cost_quality),
        request.metric_bindings,
        request.ranking,
        request.gates,
    )
    material = {
        "record_type": "cayu.memory-experiment-report",
        "schema_version": MEMORY_EXPERIMENT_REPORT_SCHEMA_VERSION,
        "experiment_id": request.experiment_id,
        "baseline_variant_id": request.baseline_variant_id,
        "repetitions": request.repetitions,
        "cases": [item.model_dump(mode="json") for item in request.cases],
        "variants": [item.model_dump(mode="json") for item in request.variants],
        "metric_bindings": [item.model_dump(mode="json") for item in request.metric_bindings],
        "ranking": [item.model_dump(mode="json") for item in request.ranking],
        "gates": request.gates.model_dump(mode="json"),
        "rows": [item.model_dump(mode="json") for item in rows],
        "comparisons": [item.model_dump(mode="json") for item in comparison_tuple],
        "operational_summary": [item.model_dump(mode="json") for item in operational_summary],
        "cost_quality": [item.model_dump(mode="json") for item in cost_quality],
        "selected_variant_id": selected,
        "dispositions": [item.model_dump(mode="json") for item in final_dispositions],
    }
    return MemoryExperimentReport(
        revision=_content_revision(material, "memory experiment report"),
        experiment_id=request.experiment_id,
        baseline_variant_id=request.baseline_variant_id,
        repetitions=request.repetitions,
        cases=request.cases,
        variants=request.variants,
        metric_bindings=request.metric_bindings,
        ranking=request.ranking,
        gates=request.gates,
        rows=rows,
        comparisons=comparison_tuple,
        operational_summary=operational_summary,
        cost_quality=tuple(cost_quality),
        selected_variant_id=selected,
        dispositions=final_dispositions,
    )


def memory_experiment_report_to_json(report: MemoryExperimentReport) -> str:
    if type(report) is not MemoryExperimentReport:
        raise TypeError("report must be an exact MemoryExperimentReport.")
    validated = MemoryExperimentReport.model_validate(
        report.model_dump(mode="python", round_trip=True, warnings="none")
    )
    encoded = json.dumps(
        validated.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    if len(encoded.encode("utf-8")) > MEMORY_EXPERIMENT_REPORT_MAX_BYTES:
        raise ValueError("Memory experiment report exceeds its JSON byte bound.")
    return encoded


def memory_experiment_report_from_json(source: str | bytes) -> MemoryExperimentReport:
    normalized = _memory_experiment_json_document(
        source,
        field_name="memory experiment report JSON",
        oversized_message="Memory experiment report exceeds its JSON byte bound.",
    )
    return MemoryExperimentReport.model_validate_json(normalized)


def memory_experiment_request_from_json(source: str | bytes) -> MemoryExperimentReportRequest:
    normalized = _memory_experiment_json_document(
        source,
        field_name="memory experiment request JSON",
        oversized_message="Memory experiment request exceeds its JSON byte bound.",
    )
    return MemoryExperimentReportRequest.model_validate_json(normalized)


def _memory_experiment_json_document(
    source: str | bytes,
    *,
    field_name: str,
    oversized_message: str,
) -> bytes:
    if type(source) is str:
        try:
            raw = source.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError(f"{field_name} must contain valid Unicode scalar text.") from exc
    elif type(source) is bytes:
        raw = source
    else:
        raise TypeError(f"{field_name} must be text or bytes.")
    if len(raw) > MEMORY_EXPERIMENT_REPORT_MAX_BYTES:
        raise ValueError(oversized_message)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{field_name} must be UTF-8.") from exc
    try:
        decoded = json.loads(
            text,
            parse_int=partial(
                parse_durable_json_integer_literal,
                field_name=field_name,
            ),
            parse_constant=partial(
                reject_nonportable_json_constant,
                field_name=field_name,
            ),
            object_pairs_hook=partial(
                durable_json_object_from_pairs,
                field_name=field_name,
            ),
        )
    except RecursionError as exc:
        raise ValueError(f"{field_name} exceeds the supported nesting depth.") from exc
    document = copy_durable_json_object(decoded, field_name)
    return json.dumps(
        document,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def render_memory_experiment_report_html(report: MemoryExperimentReport) -> str:
    report = MemoryExperimentReport.model_validate(
        report.model_dump(mode="python", round_trip=True, warnings="none")
    )
    case_authority_rows = "".join(
        "<tr>"
        f"<td><code>{html.escape(item.case_id)}</code></td>"
        f"<td><code>{html.escape(item.case_revision)}</code></td>"
        "</tr>"
        for item in report.cases
    )
    variant_authority_rows = "".join(
        "<tr>"
        f"<td><code>{html.escape(item.variant_id)}</code></td>"
        f"<td><code>{html.escape(item.candidate_id)}</code></td>"
        f"<td>{html.escape(item.spec.kind.value)}</td>"
        f"<td><code>{html.escape(item.spec.fingerprint)}</code></td>"
        f"<td><code>{html.escape(item.spec.snapshot_fingerprint)}</code></td>"
        f"<td><code>{html.escape(item.execution_profile.revision)}</code></td>"
        f"<td><code>{html.escape(item.execution_profile_binding.runtime_execution_profile.fingerprint)}</code></td>"
        f"<td>{html.escape(item.execution_profile.candidate.provider_name)}</td>"
        f"<td>{html.escape(item.execution_profile.candidate.model)}</td>"
        f"<td><code>{html.escape(item.evaluator_fingerprint)}</code></td>"
        "</tr>"
        for item in report.variants
    )
    ranking_rows = (
        "".join(
            "<tr>"
            f"<td>{index}</td>"
            f"<td>{html.escape(item.role.value)}</td>"
            f"<td>{html.escape(item.direction.value)}</td>"
            "</tr>"
            for index, item in enumerate(report.ranking, start=1)
        )
        or '<tr><td colspan="3">No task-quality ranking declared.</td></tr>'
    )
    metric_gate_rows = (
        "".join(
            "<tr>"
            f"<td>{html.escape(item.role.value)}</td>"
            f"<td>{'-' if item.minimum is None else f'{item.minimum:g}'}</td>"
            f"<td>{'-' if item.maximum is None else f'{item.maximum:g}'}</td>"
            "</tr>"
            for item in report.gates.metric_gates
        )
        or '<tr><td colspan="3">No metric thresholds declared.</td></tr>'
    )
    required_gate_roles = (
        ", ".join(item.value for item in report.gates.required_metric_roles) or "none"
    )
    maximum_cost = (
        "none"
        if report.gates.maximum_candidate_cost is None
        else f"{report.gates.maximum_candidate_cost} {report.gates.cost_currency}"
    )
    disposition_rows = "".join(
        "<tr>"
        f"<td><code>{html.escape(item.variant_id)}</code></td>"
        f"<td>{html.escape(item.status.value)}</td>"
        f"<td>{item.comparable_pair_count}</td>"
        f"<td>{item.incomparable_pair_count}</td>"
        f"<td>{item.unavailable_pair_count}</td>"
        f"<td>{html.escape(', '.join(item.reasons))}</td>"
        "</tr>"
        for item in report.dispositions
    )
    case_rows = "".join(
        "<tr>"
        f"<td><code>{html.escape(item.case_id)}</code></td>"
        f"<td><code>{html.escape(item.candidate_variant_id)}</code></td>"
        f"<td>{sum(pair.status is MemoryPairStatus.COMPARABLE for pair in item.pairs)}</td>"
        f"<td>{sum(pair.status is MemoryPairStatus.INCOMPARABLE for pair in item.pairs)}</td>"
        f"<td>{sum(pair.status is MemoryPairStatus.UNAVAILABLE for pair in item.pairs)}</td>"
        "</tr>"
        for item in report.comparisons
    )
    trial_rows = "".join(
        "<tr>"
        f"<td><code>{html.escape(row.case_id)}</code></td>"
        f"<td>{row.repetition}</td>"
        f"<td><code>{html.escape(row.variant_id)}</code></td>"
        f"<td><code>{html.escape(row.execution_id or '-')}</code></td>"
        f"<td><code>{html.escape(row.trial_id or '-')}</code></td>"
        f"<td>{'-' if row.execution_revision is None else row.execution_revision}</td>"
        f"<td>{html.escape(row.availability.value)}</td>"
        f"<td>{html.escape('-' if row.execution_status is None else row.execution_status.value)}</td>"
        f"<td>{html.escape('-' if row.published_status is None else str(row.published_status))}</td>"
        f"<td><code>{html.escape(row.final_binding_fingerprint or '-')}</code></td>"
        f"<td><code>{html.escape(row.execution_binding_lineage_revision or '-')}</code></td>"
        f"<td><code>{html.escape(row.source_trial_revision or '-')}</code></td>"
        f"<td><code>{html.escape(row.intervention_attribution_fingerprint or '-')}</code></td>"
        f"<td><code>{html.escape(row.attribution_evidence_revision or '-')}</code></td>"
        f"<td>{html.escape('-' if row.attribution_status is None else row.attribution_status)}</td>"
        f"<td>{html.escape(_metric_observations_summary(row.metrics))}</td>"
        f"<td>{'-' if row.duration_ms is None else row.duration_ms}</td>"
        f"<td>{'-' if row.total_tokens is None else row.total_tokens}</td>"
        f"<td>{html.escape(_overhead_summary(row.memory_overhead))}</td>"
        f"<td><code>{html.escape('-' if row.published_result_revision is None else row.published_result_revision)}</code></td>"
        "</tr>"
        for row in report.rows
    )
    pair_rows = "".join(
        "<tr>"
        f"<td><code>{html.escape(pair.case_id)}</code></td>"
        f"<td>{pair.repetition}</td>"
        f"<td><code>{html.escape(pair.candidate_variant_id)}</code></td>"
        f"<td>{html.escape(pair.status.value)}</td>"
        f"<td>{html.escape(', '.join(pair.reasons) or '-')}</td>"
        f"<td>{html.escape(', '.join(f'{item.role.value}:{item.baseline:g}/{item.candidate:g}/{item.delta:+g}' for item in pair.metric_deltas) or '-')}</td>"
        f"<td>{html.escape(', '.join(f'{item.dimension.value}:{item.baseline}/{item.candidate}/{item.delta:+d}' for item in pair.operational_deltas) or '-')}</td>"
        f"<td>{html.escape(_accounting_summary(pair.accounting_pair))}</td>"
        "</tr>"
        for comparison in report.comparisons
        for pair in comparison.pairs
    )
    distribution_rows = "".join(
        "<tr>"
        f"<td><code>{html.escape(comparison.case_id)}</code></td>"
        f"<td><code>{html.escape(comparison.candidate_variant_id)}</code></td>"
        f"<td>{html.escape(distribution.role.value)}</td>"
        f"<td>{distribution.pair_count}</td>"
        f"<td>{distribution.available_count}</td>"
        f"<td>{distribution.unavailable_count}</td>"
        f"<td>{distribution.incomparable_count}</td>"
        f"<td>{html.escape(_number_sequence(distribution.baseline_values))}</td>"
        f"<td>{html.escape(_number_sequence(distribution.candidate_values))}</td>"
        f"<td>{html.escape(_number_sequence(distribution.deltas))}</td>"
        "</tr>"
        for comparison in report.comparisons
        for distribution in comparison.distributions
    )
    operational_distribution_rows = "".join(
        "<tr>"
        f"<td><code>{html.escape(comparison.case_id)}</code></td>"
        f"<td><code>{html.escape(comparison.candidate_variant_id)}</code></td>"
        f"<td>{html.escape(distribution.dimension.value)}</td>"
        f"<td>{distribution.pair_count}</td>"
        f"<td>{distribution.available_count}</td>"
        f"<td>{distribution.unavailable_count}</td>"
        f"<td>{distribution.incomparable_count}</td>"
        f"<td>{html.escape(_integer_sequence(distribution.baseline_values))}</td>"
        f"<td>{html.escape(_integer_sequence(distribution.candidate_values))}</td>"
        f"<td>{html.escape(_integer_sequence(distribution.deltas))}</td>"
        "</tr>"
        for comparison in report.comparisons
        for distribution in comparison.operational_distributions
    )
    experiment_operational_rows = "".join(
        "<tr>"
        f"<td><code>{html.escape(summary.candidate_variant_id)}</code></td>"
        f"<td>{html.escape(distribution.dimension.value)}</td>"
        f"<td>{distribution.pair_count}</td>"
        f"<td>{distribution.available_count}</td>"
        f"<td>{distribution.unavailable_count}</td>"
        f"<td>{distribution.incomparable_count}</td>"
        f"<td>{html.escape(_integer_sequence(distribution.baseline_values))}</td>"
        f"<td>{html.escape(_integer_sequence(distribution.candidate_values))}</td>"
        f"<td>{html.escape(_integer_sequence(distribution.deltas))}</td>"
        "</tr>"
        for summary in report.operational_summary
        for distribution in summary.distributions
    )
    cost_aggregate_rows = "".join(
        "<tr>"
        f"<td><code>{html.escape(scoped.candidate_variant_id)}</code></td>"
        f"<td>{html.escape(_cost_aggregate_summary(scoped.report))}</td>"
        "</tr>"
        for scoped in report.cost_quality
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cayu Memory Experiment Report</title><style>
body{{font:15px/1.5 system-ui,sans-serif;margin:0;background:#f7f7f4;color:#18211d}}main{{max-width:1120px;margin:auto;padding:32px 24px}}table{{width:100%;border-collapse:collapse;background:white;margin:16px 0}}th,td{{border:1px solid #d9dfdc;padding:9px;text-align:left;vertical-align:top}}th{{background:#eef2f0}}code{{overflow-wrap:anywhere}}.note{{color:#5d6864}}
</style></head><body><main>
	<h1>Memory experiment report</h1>
	<p>Experiment <code>{html.escape(report.experiment_id)}</code></p>
	<p>Schema version <code>{report.schema_version}</code>; report revision <code>{html.escape(report.revision)}</code></p>
	<p>Baseline <code>{html.escape(report.baseline_variant_id)}</code>; repetitions <code>{report.repetitions}</code></p>
	<p><strong>Recommended fixed candidate:</strong> <code>{html.escape(report.selected_variant_id)}</code></p>
	<p class="note">Safety, evidence, and budget gates are evaluated before deterministic quality ranking. This report describes paired observed outcomes; it does not claim that memory caused or was used in an answer.</p>
	<h2>Case authority</h2><table><thead><tr><th>Case</th><th>Case revision</th></tr></thead><tbody>{case_authority_rows}</tbody></table>
	<h2>Variant authority</h2><table><thead><tr><th>Variant</th><th>Candidate</th><th>Intervention</th><th>Spec fingerprint</th><th>Snapshot fingerprint</th><th>Eval profile revision</th><th>Runtime profile fingerprint</th><th>Provider</th><th>Model</th><th>Evaluator fingerprint</th></tr></thead><tbody>{variant_authority_rows}</tbody></table>
	<h2>Gate policy</h2>
	<p>Required evidence roles: <code>{html.escape(required_gate_roles)}</code>; minimum comparable pairs: <code>{report.gates.minimum_comparable_pairs}</code>; require priced cost: <code>{str(report.gates.require_priced_cost).lower()}</code>; maximum candidate cost: <code>{html.escape(maximum_cost)}</code>.</p>
	<table><thead><tr><th>Metric</th><th>Minimum</th><th>Maximum</th></tr></thead><tbody>{metric_gate_rows}</tbody></table>
	<h2>Task-quality ranking</h2><table><thead><tr><th>Priority</th><th>Metric</th><th>Direction</th></tr></thead><tbody>{ranking_rows}</tbody></table>
	<h2>Variant dispositions</h2><table><thead><tr><th>Variant</th><th>Status</th><th>Comparable</th><th>Incomparable</th><th>Unavailable</th><th>Reasons</th></tr></thead><tbody>{disposition_rows}</tbody></table>
<h2>Case comparisons</h2><table><thead><tr><th>Case</th><th>Candidate</th><th>Comparable</th><th>Incomparable</th><th>Unavailable</th></tr></thead><tbody>{case_rows}</tbody></table>
<h2>Complete trial matrix</h2><table><thead><tr><th>Case</th><th>Repetition</th><th>Variant</th><th>Execution ID</th><th>Trial ID</th><th>Execution revision</th><th>Availability</th><th>Execution</th><th>Published</th><th>Final binding fingerprint</th><th>Execution-binding lineage revision</th><th>Source trial revision</th><th>Intervention attribution fingerprint</th><th>Attribution evidence revision</th><th>Attribution</th><th>Metrics/outcomes/evaluator authority</th><th>Latency ms</th><th>Tokens</th><th>Memory overhead</th><th>Published result</th></tr></thead><tbody>{trial_rows}</tbody></table>
<h2>Paired evidence</h2><table><thead><tr><th>Case</th><th>Repetition</th><th>Candidate</th><th>Status</th><th>Reasons</th><th>Metric baseline/candidate/delta</th><th>Operational baseline/candidate/delta</th><th>Canonical cost/usage evidence</th></tr></thead><tbody>{pair_rows}</tbody></table>
<h2>Metric distributions</h2><table><thead><tr><th>Case</th><th>Candidate</th><th>Metric</th><th>Pairs</th><th>Available</th><th>Unavailable</th><th>Incomparable</th><th>Baseline values</th><th>Candidate values</th><th>Deltas</th></tr></thead><tbody>{distribution_rows}</tbody></table>
<h2>Case operational distributions</h2><table><thead><tr><th>Case</th><th>Candidate</th><th>Dimension</th><th>Pairs</th><th>Available</th><th>Unavailable</th><th>Incomparable</th><th>Baseline values</th><th>Candidate values</th><th>Deltas</th></tr></thead><tbody>{operational_distribution_rows}</tbody></table>
<h2>Experiment operational distributions</h2><table><thead><tr><th>Candidate</th><th>Dimension</th><th>Pairs</th><th>Available</th><th>Unavailable</th><th>Incomparable</th><th>Baseline values</th><th>Candidate values</th><th>Deltas</th></tr></thead><tbody>{experiment_operational_rows}</tbody></table>
<h2>Canonical cost/usage aggregates</h2><table><thead><tr><th>Candidate</th><th>Aggregate</th></tr></thead><tbody>{cost_aggregate_rows}</tbody></table>
<p class="note">The JSON form retains exact profile, attribution, accounting-attempt, and content-fingerprint evidence. Missing dimensions remain explicitly unavailable in both forms.</p>
</main></body></html>"""
    if len(document.encode("utf-8")) > MEMORY_EXPERIMENT_REPORT_MAX_HTML_BYTES:
        raise ValueError("Memory experiment report HTML exceeds its byte bound.")
    return document


def _number_sequence(values: tuple[float, ...]) -> str:
    return "-" if not values else ", ".join(f"{value:g}" for value in values)


def _integer_sequence(values: tuple[int, ...]) -> str:
    return "-" if not values else ", ".join(str(value) for value in values)


def _metric_observations_summary(
    observations: tuple[MemoryMetricObservation, ...],
) -> str:
    if not observations:
        return "-"
    return ", ".join(
        (
            f"{item.role.value}={item.availability.value}"
            f";outcome={item.outcome or '-'}"
            f";value={'-' if item.value is None else f'{item.value:g}'}"
            f";evaluator={item.evaluator_key or '-'}"
            f"@{item.evaluator_implementation_revision or '-'}"
        )
        for item in observations
    )


def _overhead_summary(evidence: MemoryPreparationOverheadEvidence | None) -> str:
    if evidence is None:
        return "unavailable"
    values = (
        ("prepare_ms", evidence.preparation_duration_ms),
        ("context_tokens", evidence.context_tokens),
        ("context_bytes", evidence.context_bytes),
    )
    return ", ".join(f"{name}={value}" for name, value in values if value is not None)


def _accounting_summary(evidence: PairedCostQualityPairReport | None) -> str:
    if evidence is None:
        return "unavailable"
    baseline_retries = (
        None if evidence.baseline is None else evidence.baseline.whole_harness.retry_attempt_count
    )
    candidate_retries = (
        None if evidence.candidate is None else evidence.candidate.whole_harness.retry_attempt_count
    )
    return (
        f"{evidence.status.value}; currency={evidence.currency or '-'}; "
        f"cost={evidence.baseline_cost if evidence.baseline_cost is not None else '-'}"
        f"/{evidence.candidate_cost if evidence.candidate_cost is not None else '-'}; "
        f"retries={baseline_retries if baseline_retries is not None else '-'}"
        f"/{candidate_retries if candidate_retries is not None else '-'}"
    )


def _cost_aggregate_summary(
    report: PairedCostQualityComparisonReport | None,
) -> str:
    if report is None:
        return "unavailable"
    aggregate = report.aggregate
    return (
        f"status={aggregate.status.value}; pairs={aggregate.pair_count}; "
        f"eligible={','.join(aggregate.eligible_pair_ids) or '-'}; "
        f"excluded={len(aggregate.exclusions)}; currency={aggregate.currency or '-'}; "
        f"cost={aggregate.baseline_cost if aggregate.baseline_cost is not None else '-'}"
        f"/{aggregate.candidate_cost if aggregate.candidate_cost is not None else '-'}; "
        f"savings={aggregate.savings if aggregate.savings is not None else '-'}; "
        f"savings_percentage={aggregate.savings_percentage if aggregate.savings_percentage is not None else '-'}; "
        f"direction={aggregate.cost_direction.value}"
    )


__all__ = [
    "MEMORY_EXPERIMENT_REPORT_MAX_BYTES",
    "MEMORY_EXPERIMENT_REPORT_SCHEMA_VERSION",
    "MemoryCaseComparison",
    "MemoryExperimentCase",
    "MemoryExperimentGatePolicy",
    "MemoryExperimentReport",
    "MemoryExperimentReportRequest",
    "MemoryExperimentTrialEvidence",
    "MemoryExperimentVariant",
    "MemoryMetricAvailability",
    "MemoryMetricBinding",
    "MemoryMetricDelta",
    "MemoryMetricDirection",
    "MemoryMetricDistribution",
    "MemoryMetricGate",
    "MemoryMetricObservation",
    "MemoryMetricRole",
    "MemoryOperationalDelta",
    "MemoryOperationalDimension",
    "MemoryOperationalDistribution",
    "MemoryPairStatus",
    "MemoryPreparationOverheadEvidence",
    "MemoryPublishedResultEvidence",
    "MemoryRankingTerm",
    "MemoryTrialAvailability",
    "MemoryTrialPairComparison",
    "MemoryTrialReportRow",
    "MemoryVariantCostQualityReport",
    "MemoryVariantDisposition",
    "MemoryVariantDispositionStatus",
    "MemoryVariantOperationalReport",
    "build_memory_experiment_report",
    "memory_experiment_accounting_source_id",
    "memory_experiment_accounting_task_id",
    "memory_experiment_report_from_json",
    "memory_experiment_report_to_json",
    "memory_experiment_request_from_json",
    "render_memory_experiment_report_html",
]
