"""Bounded, read-only planning and evaluation for knowledge maintenance.

The router decides which exact active revisions are eligible for review.  This
module supplies the next boundary: an injected planner may propose a strict
consolidation draft, Cayu validates its structure independently, and a separate
injected evaluator judges semantic safety.  Nothing here persists a proposal or
changes knowledge lifecycle state.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable, Iterable
from contextlib import suppress
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from typing import Any, Literal, Protocol, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from cayu._clock import utc_clock
from cayu._validation import (
    canonical_durable_json_bytes,
    copy_label_map,
    require_durable_clean_nonblank,
    require_durable_nonblank,
    require_finite,
)
from cayu.knowledge_maintenance import (
    MAX_KNOWLEDGE_MAINTENANCE_ROUTING_SIGNALS,
    KnowledgeMaintenanceCandidateSignal,
    KnowledgeMaintenanceRoutedCandidate,
    KnowledgeMaintenanceRoutingRequest,
    KnowledgeMaintenanceRoutingResult,
)
from cayu.storage.memory import (
    BUILTIN_KNOWLEDGE_KINDS,
    MAX_KNOWLEDGE_MAINTENANCE_BYTES,
    MAX_KNOWLEDGE_MAINTENANCE_SOURCES,
    MAX_KNOWLEDGE_MAINTENANCE_TEXT_BYTES,
    MAX_KNOWLEDGE_REVISION,
    KnowledgeAccessScope,
    KnowledgeEntry,
    KnowledgeEntryReadLimitExceeded,
    KnowledgeRelationKind,
    KnowledgeRevisionRef,
    copy_knowledge_access_scope,
    copy_knowledge_revision_ref,
    knowledge_entry_payload_bytes,
)

KNOWLEDGE_MAINTENANCE_PLANNING_SCHEMA_VERSION = 1
KNOWLEDGE_MAINTENANCE_DETERMINISTIC_EVALUATOR_VERSION = (
    "cayu.knowledge-maintenance-deterministic-evaluator.v1"
)
MAX_KNOWLEDGE_MAINTENANCE_PLANNING_BYTES = 4 * 1024 * 1024
MAX_KNOWLEDGE_MAINTENANCE_PLAN_CLAIMS = 256
MAX_KNOWLEDGE_MAINTENANCE_EVALUATION_FINDINGS = 256
MAX_KNOWLEDGE_MAINTENANCE_MODEL_CALLS = 16
MAX_KNOWLEDGE_MAINTENANCE_COST_MICRO_USD = 1_000_000_000
MAX_KNOWLEDGE_MAINTENANCE_TOKEN_COUNT = 2**63 - 1
MAX_KNOWLEDGE_MAINTENANCE_PLANNING_TIMEOUT_SECONDS = 300.0

_IDENTITY_MAX_BYTES = 256
_CODE_MAX_BYTES = 128
_TITLE_MAX_BYTES = 4_096
_MAX_REPLACEMENT_KINDS = 100
_MAX_MODEL_IDENTITIES = 16
_SAFE_CODE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*\Z")
_SHA256_HEX_RE = re.compile(r"[0-9a-f]{64}\Z")


class _PlanningModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        validate_default=True,
    )


def _clean(value: object, field_name: str, *, max_bytes: int = _IDENTITY_MAX_BYTES) -> str:
    if type(value) is not str:
        raise ValueError(f"`{field_name}` must be a string.")
    value = require_durable_clean_nonblank(value, field_name)
    if len(value.encode("utf-8")) > max_bytes:
        raise ValueError(f"`{field_name}` must be at most {max_bytes} UTF-8 bytes.")
    return value


def _bounded_text(value: str, field_name: str, *, max_bytes: int) -> str:
    value = require_durable_nonblank(value, field_name)
    if len(value.encode("utf-8")) > max_bytes:
        raise ValueError(f"`{field_name}` must be at most {max_bytes} UTF-8 bytes.")
    return value


def _safe_code(value: object, field_name: str) -> str:
    value = _clean(value, field_name, max_bytes=_CODE_MAX_BYTES)
    if _SAFE_CODE_RE.fullmatch(value) is None:
        raise ValueError(f"`{field_name}` must be a safe machine-readable code.")
    return value


def _fingerprint(value: object, field_name: str) -> str:
    return sha256(canonical_durable_json_bytes(value, field_name)).hexdigest()


def _validate_fingerprint(value: str, field_name: str) -> str:
    if type(value) is not str or _SHA256_HEX_RE.fullmatch(value) is None:
        raise ValueError(f"`{field_name}` must be lowercase SHA-256 hex.")
    return value


def _utc(value: datetime, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"`{field_name}` must be a datetime.")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"`{field_name}` must be timezone-aware.")
    return value.astimezone(UTC)


def _copy_revision_ref(value: object) -> KnowledgeRevisionRef:
    if isinstance(value, KnowledgeRevisionRef):
        return copy_knowledge_revision_ref(value)
    return KnowledgeRevisionRef.model_validate(value)


def _copy_revision_refs(
    value: object,
    field_name: str,
    *,
    allow_empty: bool = False,
    allow_same_entry: bool = False,
) -> tuple[KnowledgeRevisionRef, ...]:
    if not isinstance(value, list | tuple):
        raise ValueError(f"`{field_name}` must be an ordered array.")
    if len(value) > MAX_KNOWLEDGE_MAINTENANCE_SOURCES:
        raise ValueError(
            f"`{field_name}` must contain at most {MAX_KNOWLEDGE_MAINTENANCE_SOURCES} values."
        )
    copied = tuple(
        sorted(
            (_copy_revision_ref(item) for item in value),
            key=lambda item: (item.entry_id, item.revision),
        )
    )
    if not allow_empty and not copied:
        raise ValueError(f"`{field_name}` cannot be empty.")
    if len({(item.entry_id, item.revision) for item in copied}) != len(copied):
        raise ValueError(f"`{field_name}` cannot repeat an exact revision.")
    if not allow_same_entry and len({item.entry_id for item in copied}) != len(copied):
        raise ValueError(f"`{field_name}` cannot repeat a logical entry.")
    return copied


def _copy_safe_codes(
    value: object,
    field_name: str,
    *,
    allow_empty: bool = False,
    maximum: int = MAX_KNOWLEDGE_MAINTENANCE_PLAN_CLAIMS,
) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        raise ValueError(f"`{field_name}` must be an ordered array.")
    if len(value) > maximum:
        raise ValueError(f"`{field_name}` must contain at most {maximum} values.")
    copied = tuple(sorted(_safe_code(item, field_name) for item in value))
    if not allow_empty and not copied:
        raise ValueError(f"`{field_name}` cannot be empty.")
    if len(copied) != len(set(copied)):
        raise ValueError(f"`{field_name}` cannot contain duplicates.")
    return copied


def _copy_model_identities(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        raise ValueError(f"`{field_name}` must be an ordered array.")
    if len(value) > _MAX_MODEL_IDENTITIES:
        raise ValueError(f"`{field_name}` must contain at most {_MAX_MODEL_IDENTITIES} values.")
    copied = tuple(sorted(_clean(item, field_name) for item in value))
    if len(copied) != len(set(copied)):
        raise ValueError(f"`{field_name}` cannot contain duplicates.")
    return copied


def _snapshot_payload_bytes(
    candidates: Iterable[KnowledgeMaintenanceRoutedCandidate],
    signals: Iterable[KnowledgeMaintenanceCandidateSignal],
) -> int:
    return len(
        canonical_durable_json_bytes(
            {
                "contract": "cayu.knowledge-maintenance-routed-payload.v1",
                "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
                "signals": [signal.model_dump(mode="json") for signal in signals],
            },
            "knowledge maintenance routed payload",
        )
    )


class KnowledgeMaintenancePlanEndpointKind(StrEnum):
    """The two endpoint roles representable in an unpersisted relation draft."""

    REPLACEMENT = "replacement"
    SOURCE = "source"


class KnowledgeMaintenanceEvaluationVerdict(StrEnum):
    """Independent semantic disposition of one structurally valid plan."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"


class KnowledgeMaintenanceEvaluationFindingKind(StrEnum):
    """Closed, privacy-safe reasons a maintenance plan cannot proceed."""

    ROUTING_BINDING_INVALID = "routing_binding_invalid"
    SOURCE_OUTSIDE_ROUTE = "source_outside_route"
    SOURCE_COVERAGE_INCOMPLETE = "source_coverage_incomplete"
    RELATION_COVERAGE_INVALID = "relation_coverage_invalid"
    RELATION_ORIENTATION_INVALID = "relation_orientation_invalid"
    SOURCE_REVISION_EXHAUSTED = "source_revision_exhausted"
    EVIDENCE_COVERAGE_INCOMPLETE = "evidence_coverage_incomplete"
    EVIDENCE_SOURCE_INVALID = "evidence_source_invalid"
    REPLACEMENT_KIND_DISALLOWED = "replacement_kind_disallowed"
    STALE_SOURCE = "stale_source"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    INFORMATION_LOSS = "information_loss"
    CONTRADICTION_MISHANDLED = "contradiction_mishandled"
    RETENTION_VIOLATION = "retention_violation"
    POLICY_VIOLATION = "policy_violation"
    PROMPT_INJECTION = "prompt_injection"


class KnowledgeMaintenanceEvaluationFindingCode(StrEnum):
    """Closed runtime-owned diagnostic vocabulary for one finding."""

    ROUTING_BINDING_INVALID = "routing_binding_invalid"
    SOURCE_OUTSIDE_ROUTE = "source_outside_route"
    SOURCE_COVERAGE_INCOMPLETE = "source_coverage_incomplete"
    RELATION_COVERAGE_INVALID = "relation_coverage_invalid"
    RELATION_ORIENTATION_INVALID = "relation_orientation_invalid"
    SOURCE_REVISION_EXHAUSTED = "source_revision_exhausted"
    EVIDENCE_COVERAGE_INCOMPLETE = "evidence_coverage_incomplete"
    RELATION_EVIDENCE_MAPPING_INVALID = "relation_evidence_mapping_invalid"
    UNREFERENCED_EVIDENCE_MAPPING = "unreferenced_evidence_mapping"
    EVIDENCE_SOURCE_INVALID = "evidence_source_invalid"
    REPLACEMENT_KIND_DISALLOWED = "replacement_kind_disallowed"
    STALE_SOURCE = "stale_source"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    INFORMATION_LOSS = "information_loss"
    CONTRADICTION_MISHANDLED = "contradiction_mishandled"
    RETENTION_VIOLATION = "retention_violation"
    POLICY_VIOLATION = "policy_violation"
    PROMPT_INJECTION = "prompt_injection"


class KnowledgeMaintenancePlanningOutcome(StrEnum):
    """Exhaustive result of one read-only planning attempt."""

    NO_CANDIDATES = "no_candidates"
    ROUTING_INCOMPLETE = "routing_incomplete"
    SOURCE_SET_CHANGED = "source_set_changed"
    SOURCE_REVALIDATION_FAILED = "source_revalidation_failed"
    SOURCE_REVALIDATION_FAILED_AFTER_PLANNING = "source_revalidation_failed_after_planning"
    SOURCE_REVALIDATION_FAILED_AFTER_EVALUATION = "source_revalidation_failed_after_evaluation"
    PLANNER_TIMED_OUT = "planner_timed_out"
    PLANNER_FAILED = "planner_failed"
    PLANNER_INVALID = "planner_invalid"
    PLANNER_OVER_BUDGET = "planner_over_budget"
    DETERMINISTIC_REJECTED = "deterministic_rejected"
    EVALUATOR_TIMED_OUT = "evaluator_timed_out"
    EVALUATOR_FAILED = "evaluator_failed"
    EVALUATOR_INVALID = "evaluator_invalid"
    EVALUATOR_OVER_BUDGET = "evaluator_over_budget"
    EVALUATOR_REJECTED = "evaluator_rejected"
    ACCEPTED = "accepted"


_EVALUATOR_FINDING_KINDS = frozenset(
    {
        KnowledgeMaintenanceEvaluationFindingKind.UNSUPPORTED_CLAIM,
        KnowledgeMaintenanceEvaluationFindingKind.INFORMATION_LOSS,
        KnowledgeMaintenanceEvaluationFindingKind.CONTRADICTION_MISHANDLED,
        KnowledgeMaintenanceEvaluationFindingKind.RETENTION_VIOLATION,
        KnowledgeMaintenanceEvaluationFindingKind.POLICY_VIOLATION,
        KnowledgeMaintenanceEvaluationFindingKind.PROMPT_INJECTION,
    }
)

_FINDING_CODES_BY_KIND = {
    kind: frozenset({KnowledgeMaintenanceEvaluationFindingCode(kind.value)})
    for kind in KnowledgeMaintenanceEvaluationFindingKind
}
_FINDING_CODES_BY_KIND[KnowledgeMaintenanceEvaluationFindingKind.EVIDENCE_COVERAGE_INCOMPLETE] = (
    frozenset(
        {
            KnowledgeMaintenanceEvaluationFindingCode.EVIDENCE_COVERAGE_INCOMPLETE,
            KnowledgeMaintenanceEvaluationFindingCode.RELATION_EVIDENCE_MAPPING_INVALID,
            KnowledgeMaintenanceEvaluationFindingCode.UNREFERENCED_EVIDENCE_MAPPING,
        }
    )
)

_InvocationT = TypeVar("_InvocationT")


class KnowledgeMaintenancePlanningLimitExceeded(ValueError):
    """A planning request exceeds a configured pre-invocation ceiling."""

    def __init__(self, limit: str) -> None:
        self.limit = _safe_code(limit, "limit")
        super().__init__("Knowledge maintenance planning exceeds a configured work limit.")


class KnowledgeMaintenanceStageBudget(_PlanningModel):
    """One explicit provider-neutral invocation budget."""

    max_input_bytes: StrictInt
    max_output_bytes: StrictInt
    max_model_calls: StrictInt
    max_cost_micro_usd: StrictInt
    timeout_seconds: StrictFloat
    allowed_model_ids: tuple[StrictStr, ...]

    @field_validator("max_input_bytes", "max_output_bytes")
    @classmethod
    def validate_positive_int(cls, value: int, info) -> int:
        if value <= 0:
            raise ValueError(f"`{info.field_name}` must be greater than 0.")
        if value > MAX_KNOWLEDGE_MAINTENANCE_PLANNING_BYTES:
            raise ValueError(
                f"`{info.field_name}` must be at most {MAX_KNOWLEDGE_MAINTENANCE_PLANNING_BYTES}."
            )
        return value

    @field_validator("max_model_calls")
    @classmethod
    def validate_model_calls(cls, value: int) -> int:
        if value < 0 or value > MAX_KNOWLEDGE_MAINTENANCE_MODEL_CALLS:
            raise ValueError(
                f"`max_model_calls` must be between 0 and {MAX_KNOWLEDGE_MAINTENANCE_MODEL_CALLS}."
            )
        return value

    @field_validator("max_cost_micro_usd")
    @classmethod
    def validate_cost(cls, value: int) -> int:
        if value < 0:
            raise ValueError("`max_cost_micro_usd` must be non-negative.")
        if value > MAX_KNOWLEDGE_MAINTENANCE_COST_MICRO_USD:
            raise ValueError(
                f"`max_cost_micro_usd` must be at most {MAX_KNOWLEDGE_MAINTENANCE_COST_MICRO_USD}."
            )
        return value

    @field_validator("timeout_seconds")
    @classmethod
    def validate_timeout(cls, value: float) -> float:
        value = require_finite(value, "timeout_seconds")
        if value <= 0.0 or value > MAX_KNOWLEDGE_MAINTENANCE_PLANNING_TIMEOUT_SECONDS:
            raise ValueError(
                "`timeout_seconds` must be greater than 0 and at most "
                f"{MAX_KNOWLEDGE_MAINTENANCE_PLANNING_TIMEOUT_SECONDS}."
            )
        return value

    @field_validator("allowed_model_ids", mode="before")
    @classmethod
    def copy_allowed_model_ids(cls, value: object) -> tuple[str, ...]:
        return _copy_model_identities(value, "allowed_model_ids")

    @model_validator(mode="after")
    def validate_model_authority(self) -> KnowledgeMaintenanceStageBudget:
        if self.max_model_calls == 0 and self.allowed_model_ids:
            raise ValueError("Zero model calls cannot authorize a model identity.")
        if self.max_model_calls > 0 and not self.allowed_model_ids:
            raise ValueError("A positive model-call budget requires an allowed model identity.")
        return self


class KnowledgeMaintenancePlannerBudget(KnowledgeMaintenanceStageBudget):
    """Planner invocation budget including every structured-output ceiling."""

    max_evidence_mappings: StrictInt
    max_replacement_text_bytes: StrictInt
    max_claim_bytes: StrictInt

    @field_validator("max_evidence_mappings")
    @classmethod
    def validate_evidence_mapping_count(cls, value: int) -> int:
        if value <= 0 or value > MAX_KNOWLEDGE_MAINTENANCE_PLAN_CLAIMS:
            raise ValueError(
                "`max_evidence_mappings` must be greater than 0 and at most "
                f"{MAX_KNOWLEDGE_MAINTENANCE_PLAN_CLAIMS}."
            )
        return value

    @field_validator("max_replacement_text_bytes")
    @classmethod
    def validate_replacement_text_bytes(cls, value: int) -> int:
        if value <= 0 or value > MAX_KNOWLEDGE_MAINTENANCE_BYTES:
            raise ValueError(
                "`max_replacement_text_bytes` must be greater than 0 and at most "
                f"{MAX_KNOWLEDGE_MAINTENANCE_BYTES}."
            )
        return value

    @field_validator("max_claim_bytes")
    @classmethod
    def validate_claim_bytes(cls, value: int) -> int:
        if value <= 0 or value > MAX_KNOWLEDGE_MAINTENANCE_TEXT_BYTES:
            raise ValueError(
                "`max_claim_bytes` must be greater than 0 and at most "
                f"{MAX_KNOWLEDGE_MAINTENANCE_TEXT_BYTES}."
            )
        return value


class KnowledgeMaintenanceInferenceUsage(_PlanningModel):
    """Measured provider use reported by an injected planning component."""

    model_calls: StrictInt = 0
    input_tokens: StrictInt = 0
    output_tokens: StrictInt = 0
    cost_micro_usd: StrictInt = 0
    model_id: StrictStr | None = None

    @field_validator("model_calls", "input_tokens", "output_tokens", "cost_micro_usd")
    @classmethod
    def validate_nonnegative(cls, value: int, info) -> int:
        if value < 0:
            raise ValueError(f"`{info.field_name}` must be non-negative.")
        maximum = {
            "model_calls": MAX_KNOWLEDGE_MAINTENANCE_MODEL_CALLS,
            "input_tokens": MAX_KNOWLEDGE_MAINTENANCE_TOKEN_COUNT,
            "output_tokens": MAX_KNOWLEDGE_MAINTENANCE_TOKEN_COUNT,
            "cost_micro_usd": MAX_KNOWLEDGE_MAINTENANCE_COST_MICRO_USD,
        }[info.field_name]
        if value > maximum:
            raise ValueError(f"`{info.field_name}` must be at most {maximum}.")
        return value

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _clean(value, "model_id")

    @model_validator(mode="after")
    def validate_model_identity(self) -> KnowledgeMaintenanceInferenceUsage:
        if self.model_calls == 0:
            if self.input_tokens or self.output_tokens or self.cost_micro_usd:
                raise ValueError("Zero model calls cannot report model token or cost usage.")
            if self.model_id is not None:
                raise ValueError("Zero model calls cannot report a model identity.")
        elif self.model_id is None:
            raise ValueError("Non-zero model calls require `model_id`.")
        return self


class KnowledgeMaintenancePlanningConfig(_PlanningModel):
    """Application-owned identities, semantic policy, and resource ceilings."""

    schema_version: Literal[1] = KNOWLEDGE_MAINTENANCE_PLANNING_SCHEMA_VERSION
    planner_id: StrictStr
    planner_version: StrictStr
    evaluator_id: StrictStr
    evaluator_version: StrictStr
    planner_model_ids: tuple[StrictStr, ...] = ()
    evaluator_model_ids: tuple[StrictStr, ...] = ()
    allowed_replacement_kinds: tuple[StrictStr, ...] = BUILTIN_KNOWLEDGE_KINDS
    max_planner_input_bytes: StrictInt = 512 * 1024
    max_plan_bytes: StrictInt = MAX_KNOWLEDGE_MAINTENANCE_BYTES
    max_evaluator_input_bytes: StrictInt = 768 * 1024
    max_evaluator_output_bytes: StrictInt = 128 * 1024
    max_revalidation_bytes: StrictInt = 512 * 1024
    max_revalidation_concurrency: StrictInt = 8
    max_evidence_mappings: StrictInt = 100
    max_replacement_text_bytes: StrictInt = 64 * 1024
    max_claim_bytes: StrictInt = MAX_KNOWLEDGE_MAINTENANCE_TEXT_BYTES
    max_planner_model_calls: StrictInt = 1
    max_evaluator_model_calls: StrictInt = 1
    max_planner_cost_micro_usd: StrictInt = 500_000
    max_evaluator_cost_micro_usd: StrictInt = 500_000
    max_total_cost_micro_usd: StrictInt = 1_000_000
    source_revalidation_timeout_seconds: StrictFloat = 10.0
    planner_timeout_seconds: StrictFloat = 30.0
    evaluator_timeout_seconds: StrictFloat = 30.0

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> object:
        if type(value) is not int or value != KNOWLEDGE_MAINTENANCE_PLANNING_SCHEMA_VERSION:
            raise ValueError("`schema_version` must be the integer 1.")
        return value

    @field_validator("planner_id", "planner_version", "evaluator_id", "evaluator_version")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _clean(value, info.field_name)

    @field_validator("planner_model_ids", "evaluator_model_ids", mode="before")
    @classmethod
    def copy_model_identities(cls, value: object, info) -> tuple[str, ...]:
        return _copy_model_identities(value, info.field_name)

    @field_validator("allowed_replacement_kinds", mode="before")
    @classmethod
    def copy_allowed_kinds(cls, value: object) -> tuple[str, ...]:
        if not isinstance(value, list | tuple) or not value:
            raise ValueError("`allowed_replacement_kinds` must be a non-empty ordered array.")
        if len(value) > _MAX_REPLACEMENT_KINDS:
            raise ValueError(
                f"`allowed_replacement_kinds` must contain at most {_MAX_REPLACEMENT_KINDS} values."
            )
        copied = tuple(sorted(_clean(item, "allowed_replacement_kinds") for item in value))
        if len(copied) != len(set(copied)):
            raise ValueError("`allowed_replacement_kinds` cannot contain duplicates.")
        return copied

    @field_validator(
        "max_planner_input_bytes",
        "max_plan_bytes",
        "max_evaluator_input_bytes",
        "max_evaluator_output_bytes",
        "max_revalidation_bytes",
        "max_revalidation_concurrency",
        "max_evidence_mappings",
        "max_replacement_text_bytes",
        "max_claim_bytes",
    )
    @classmethod
    def validate_positive_int(cls, value: int, info) -> int:
        if value <= 0:
            raise ValueError(f"`{info.field_name}` must be greater than 0.")
        return value

    @field_validator("max_planner_model_calls", "max_evaluator_model_calls")
    @classmethod
    def validate_model_call_limit(cls, value: int, info) -> int:
        if value < 0 or value > MAX_KNOWLEDGE_MAINTENANCE_MODEL_CALLS:
            raise ValueError(
                f"`{info.field_name}` must be between 0 and "
                f"{MAX_KNOWLEDGE_MAINTENANCE_MODEL_CALLS}."
            )
        return value

    @field_validator(
        "max_planner_cost_micro_usd",
        "max_evaluator_cost_micro_usd",
        "max_total_cost_micro_usd",
    )
    @classmethod
    def validate_nonnegative_int(cls, value: int, info) -> int:
        if value < 0:
            raise ValueError(f"`{info.field_name}` must be non-negative.")
        return value

    @field_validator(
        "source_revalidation_timeout_seconds",
        "planner_timeout_seconds",
        "evaluator_timeout_seconds",
    )
    @classmethod
    def validate_timeout(cls, value: float, info) -> float:
        value = require_finite(value, info.field_name)
        if value <= 0.0 or value > MAX_KNOWLEDGE_MAINTENANCE_PLANNING_TIMEOUT_SECONDS:
            raise ValueError(
                f"`{info.field_name}` must be greater than 0 and at most "
                f"{MAX_KNOWLEDGE_MAINTENANCE_PLANNING_TIMEOUT_SECONDS}."
            )
        return value

    @model_validator(mode="after")
    def validate_bounds(self) -> KnowledgeMaintenancePlanningConfig:
        if (self.planner_id, self.planner_version) == (
            self.evaluator_id,
            self.evaluator_version,
        ):
            raise ValueError("Planner and evaluator identities must be distinct.")
        byte_fields = (
            "max_planner_input_bytes",
            "max_plan_bytes",
            "max_evaluator_input_bytes",
            "max_evaluator_output_bytes",
            "max_revalidation_bytes",
        )
        for field_name in byte_fields:
            if getattr(self, field_name) > MAX_KNOWLEDGE_MAINTENANCE_PLANNING_BYTES:
                raise ValueError(
                    f"`{field_name}` must be at most {MAX_KNOWLEDGE_MAINTENANCE_PLANNING_BYTES}."
                )
        if self.max_plan_bytes > MAX_KNOWLEDGE_MAINTENANCE_BYTES:
            raise ValueError(f"`max_plan_bytes` must be at most {MAX_KNOWLEDGE_MAINTENANCE_BYTES}.")
        if self.max_evidence_mappings > MAX_KNOWLEDGE_MAINTENANCE_PLAN_CLAIMS:
            raise ValueError(
                f"`max_evidence_mappings` must be at most {MAX_KNOWLEDGE_MAINTENANCE_PLAN_CLAIMS}."
            )
        if self.max_replacement_text_bytes > MAX_KNOWLEDGE_MAINTENANCE_BYTES:
            raise ValueError("`max_replacement_text_bytes` exceeds the plan byte ceiling.")
        if self.max_claim_bytes > MAX_KNOWLEDGE_MAINTENANCE_TEXT_BYTES:
            raise ValueError(
                f"`max_claim_bytes` must be at most {MAX_KNOWLEDGE_MAINTENANCE_TEXT_BYTES}."
            )
        if self.max_revalidation_concurrency > MAX_KNOWLEDGE_MAINTENANCE_SOURCES:
            raise ValueError(
                "`max_revalidation_concurrency` cannot exceed the maintenance source limit."
            )
        for field_name in (
            "max_planner_cost_micro_usd",
            "max_evaluator_cost_micro_usd",
            "max_total_cost_micro_usd",
        ):
            if getattr(self, field_name) > MAX_KNOWLEDGE_MAINTENANCE_COST_MICRO_USD:
                raise ValueError(
                    f"`{field_name}` must be at most {MAX_KNOWLEDGE_MAINTENANCE_COST_MICRO_USD}."
                )
        if (
            self.max_planner_cost_micro_usd + self.max_evaluator_cost_micro_usd
            > self.max_total_cost_micro_usd
        ):
            raise ValueError("Stage cost ceilings cannot exceed `max_total_cost_micro_usd`.")
        for stage in ("planner", "evaluator"):
            max_model_calls = getattr(self, f"max_{stage}_model_calls")
            model_ids = getattr(self, f"{stage}_model_ids")
            if max_model_calls == 0 and model_ids:
                raise ValueError(f"Zero {stage} model calls cannot authorize model identities.")
            if max_model_calls > 0 and not model_ids:
                raise ValueError(
                    f"A positive {stage} model-call budget requires authorized model identities."
                )
        return self

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            {
                "contract": "cayu.knowledge-maintenance-planning-config.v1",
                "config": self.model_dump(mode="json"),
            },
            "knowledge maintenance planning configuration",
        )

    @property
    def planner_budget(self) -> KnowledgeMaintenancePlannerBudget:
        return KnowledgeMaintenancePlannerBudget(
            max_input_bytes=self.max_planner_input_bytes,
            max_output_bytes=self.max_plan_bytes,
            max_model_calls=self.max_planner_model_calls,
            max_cost_micro_usd=self.max_planner_cost_micro_usd,
            timeout_seconds=self.planner_timeout_seconds,
            allowed_model_ids=self.planner_model_ids,
            max_evidence_mappings=self.max_evidence_mappings,
            max_replacement_text_bytes=self.max_replacement_text_bytes,
            max_claim_bytes=self.max_claim_bytes,
        )

    @property
    def evaluator_budget(self) -> KnowledgeMaintenanceStageBudget:
        return KnowledgeMaintenanceStageBudget(
            max_input_bytes=self.max_evaluator_input_bytes,
            max_output_bytes=self.max_evaluator_output_bytes,
            max_model_calls=self.max_evaluator_model_calls,
            max_cost_micro_usd=self.max_evaluator_cost_micro_usd,
            timeout_seconds=self.evaluator_timeout_seconds,
            allowed_model_ids=self.evaluator_model_ids,
        )


class KnowledgeMaintenanceReplacementDraft(_PlanningModel):
    """Semantic replacement content; application-owned scope fields are absent."""

    text: StrictStr
    title: StrictStr | None = None
    kind: StrictStr
    aspects: tuple[StrictStr, ...] = ()
    impact_targets: tuple[StrictStr, ...] = ()

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _bounded_text(
            value,
            "text",
            max_bytes=MAX_KNOWLEDGE_MAINTENANCE_BYTES,
        )

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _bounded_text(value, "title", max_bytes=_TITLE_MAX_BYTES)

    @field_validator("kind")
    @classmethod
    def validate_kind(cls, value: str) -> str:
        return _clean(value, "kind")

    @field_validator("aspects", "impact_targets", mode="before")
    @classmethod
    def copy_facets(cls, value: object, info) -> tuple[str, ...]:
        if not isinstance(value, list | tuple):
            raise ValueError(f"`{info.field_name}` must be an ordered array.")
        if len(value) > 100:
            raise ValueError(f"`{info.field_name}` must contain at most 100 values.")
        copied = tuple(sorted(_clean(item, info.field_name) for item in value))
        if len(copied) != len(set(copied)):
            raise ValueError(f"`{info.field_name}` cannot contain duplicates.")
        return copied


class KnowledgeMaintenancePlanEndpoint(_PlanningModel):
    """A replacement placeholder or one exact proposed source endpoint."""

    kind: KnowledgeMaintenancePlanEndpointKind
    reference: KnowledgeRevisionRef | None = None

    @field_validator("reference", mode="before")
    @classmethod
    def copy_reference(cls, value: object) -> KnowledgeRevisionRef | None:
        if value is None:
            return None
        return _copy_revision_ref(value)

    @model_validator(mode="after")
    def validate_endpoint(self) -> KnowledgeMaintenancePlanEndpoint:
        if self.kind is KnowledgeMaintenancePlanEndpointKind.REPLACEMENT:
            if self.reference is not None:
                raise ValueError("A replacement endpoint cannot carry a source reference.")
        elif self.reference is None:
            raise ValueError("A source endpoint requires an exact source reference.")
        return self


class KnowledgeMaintenanceRelationDraft(_PlanningModel):
    """One proposed typed disposition before replacement identity allocation."""

    id: StrictStr
    subject: KnowledgeMaintenancePlanEndpoint
    object: KnowledgeMaintenancePlanEndpoint
    kind: KnowledgeRelationKind
    evidence_mapping_ids: tuple[StrictStr, ...]

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _safe_code(value, "id")

    @field_validator("subject", "object", mode="before")
    @classmethod
    def copy_endpoint(cls, value: object) -> object:
        if isinstance(value, KnowledgeMaintenancePlanEndpoint):
            return value.model_dump(mode="python")
        return value

    @field_validator("evidence_mapping_ids", mode="before")
    @classmethod
    def copy_evidence_mapping_ids(cls, value: object) -> tuple[str, ...]:
        return _copy_safe_codes(value, "evidence_mapping_ids")

    @model_validator(mode="after")
    def validate_endpoint_roles(self) -> KnowledgeMaintenanceRelationDraft:
        roles = {self.subject.kind, self.object.kind}
        if roles != {
            KnowledgeMaintenancePlanEndpointKind.REPLACEMENT,
            KnowledgeMaintenancePlanEndpointKind.SOURCE,
        }:
            raise ValueError("A relation draft requires one replacement and one source endpoint.")
        return self

    @property
    def source_reference(self) -> KnowledgeRevisionRef:
        endpoint = (
            self.subject
            if self.subject.kind is KnowledgeMaintenancePlanEndpointKind.SOURCE
            else self.object
        )
        if endpoint.reference is None:  # pragma: no cover - model invariant
            raise RuntimeError("A source endpoint is missing its reference.")
        return copy_knowledge_revision_ref(endpoint.reference)


class KnowledgeMaintenanceEvidenceMapping(_PlanningModel):
    """One replacement claim mapped to exact routed source revisions."""

    id: StrictStr
    claim: StrictStr
    source_references: tuple[KnowledgeRevisionRef, ...]

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _safe_code(value, "id")

    @field_validator("claim")
    @classmethod
    def validate_claim(cls, value: str) -> str:
        return _bounded_text(
            value,
            "claim",
            max_bytes=MAX_KNOWLEDGE_MAINTENANCE_TEXT_BYTES,
        )

    @field_validator("source_references", mode="before")
    @classmethod
    def copy_source_references(cls, value: object) -> tuple[KnowledgeRevisionRef, ...]:
        return _copy_revision_refs(value, "source_references")


class KnowledgeMaintenancePlanDraft(_PlanningModel):
    """Strict untrusted planner output over one exact routing snapshot."""

    schema_version: Literal[1] = KNOWLEDGE_MAINTENANCE_PLANNING_SCHEMA_VERSION
    id: StrictStr
    routing_request_fingerprint: StrictStr
    routing_result_fingerprint: StrictStr
    configuration_fingerprint: StrictStr
    policy_id: StrictStr
    source_references: tuple[KnowledgeRevisionRef, ...]
    replacement: KnowledgeMaintenanceReplacementDraft
    relations: tuple[KnowledgeMaintenanceRelationDraft, ...]
    evidence_mappings: tuple[KnowledgeMaintenanceEvidenceMapping, ...]
    rationale: StrictStr
    evidence_summary: StrictStr

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> object:
        if type(value) is not int or value != KNOWLEDGE_MAINTENANCE_PLANNING_SCHEMA_VERSION:
            raise ValueError("`schema_version` must be the integer 1.")
        return value

    @field_validator("id", "policy_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _clean(value, info.field_name)

    @field_validator(
        "routing_request_fingerprint",
        "routing_result_fingerprint",
        "configuration_fingerprint",
    )
    @classmethod
    def validate_fingerprints(cls, value: str, info) -> str:
        return _validate_fingerprint(value, info.field_name)

    @field_validator("source_references", mode="before")
    @classmethod
    def copy_source_references(cls, value: object) -> tuple[KnowledgeRevisionRef, ...]:
        return _copy_revision_refs(value, "source_references")

    @field_validator("replacement", mode="before")
    @classmethod
    def copy_replacement(cls, value: object) -> object:
        if isinstance(value, KnowledgeMaintenanceReplacementDraft):
            return value.model_dump(mode="python")
        return value

    @field_validator("relations", mode="before")
    @classmethod
    def copy_relations(cls, value: object) -> tuple[KnowledgeMaintenanceRelationDraft, ...]:
        if not isinstance(value, list | tuple) or not value:
            raise ValueError("`relations` must be a non-empty ordered array.")
        if len(value) > MAX_KNOWLEDGE_MAINTENANCE_SOURCES:
            raise ValueError(
                f"`relations` must contain at most {MAX_KNOWLEDGE_MAINTENANCE_SOURCES} values."
            )
        copied = tuple(
            sorted(
                (
                    KnowledgeMaintenanceRelationDraft.model_validate(
                        item.model_dump(mode="python")
                        if isinstance(item, KnowledgeMaintenanceRelationDraft)
                        else item
                    )
                    for item in value
                ),
                key=lambda item: item.id,
            )
        )
        if len({item.id for item in copied}) != len(copied):
            raise ValueError("`relations` cannot repeat an identity.")
        return copied

    @field_validator("evidence_mappings", mode="before")
    @classmethod
    def copy_evidence_mappings(
        cls,
        value: object,
    ) -> tuple[KnowledgeMaintenanceEvidenceMapping, ...]:
        if not isinstance(value, list | tuple) or not value:
            raise ValueError("`evidence_mappings` must be a non-empty ordered array.")
        if len(value) > MAX_KNOWLEDGE_MAINTENANCE_PLAN_CLAIMS:
            raise ValueError(
                "`evidence_mappings` must contain at most "
                f"{MAX_KNOWLEDGE_MAINTENANCE_PLAN_CLAIMS} values."
            )
        copied = tuple(
            sorted(
                (
                    KnowledgeMaintenanceEvidenceMapping.model_validate(
                        item.model_dump(mode="python")
                        if isinstance(item, KnowledgeMaintenanceEvidenceMapping)
                        else item
                    )
                    for item in value
                ),
                key=lambda item: item.id,
            )
        )
        if len({item.id for item in copied}) != len(copied):
            raise ValueError("`evidence_mappings` cannot repeat an identity.")
        return copied

    @field_validator("rationale", "evidence_summary")
    @classmethod
    def validate_bounded_text(cls, value: str, info) -> str:
        return _bounded_text(
            value,
            info.field_name,
            max_bytes=MAX_KNOWLEDGE_MAINTENANCE_TEXT_BYTES,
        )

    @model_validator(mode="after")
    def validate_hard_byte_bound(self) -> KnowledgeMaintenancePlanDraft:
        if _canonical_bytes(self, "knowledge maintenance plan draft") > (
            MAX_KNOWLEDGE_MAINTENANCE_BYTES
        ):
            raise ValueError("Knowledge maintenance plan draft exceeds its byte ceiling.")
        return self

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            {
                "contract": "cayu.knowledge-maintenance-plan-draft.v1",
                "plan": self.model_dump(mode="json"),
            },
            "knowledge maintenance plan draft",
        )


class KnowledgeMaintenanceEvaluationFinding(_PlanningModel):
    """One bounded diagnostic without source or replacement prose."""

    kind: KnowledgeMaintenanceEvaluationFindingKind
    code: KnowledgeMaintenanceEvaluationFindingCode
    source_references: tuple[KnowledgeRevisionRef, ...] = ()
    evidence_mapping_ids: tuple[StrictStr, ...] = ()

    @field_validator("source_references", mode="before")
    @classmethod
    def copy_source_references(cls, value: object) -> tuple[KnowledgeRevisionRef, ...]:
        return _copy_revision_refs(
            value,
            "source_references",
            allow_empty=True,
            allow_same_entry=True,
        )

    @field_validator("evidence_mapping_ids", mode="before")
    @classmethod
    def copy_evidence_mapping_ids(cls, value: object) -> tuple[str, ...]:
        return _copy_safe_codes(value, "evidence_mapping_ids", allow_empty=True)

    @model_validator(mode="after")
    def validate_code_for_kind(self) -> KnowledgeMaintenanceEvaluationFinding:
        if self.code not in _FINDING_CODES_BY_KIND[self.kind]:
            raise ValueError("`code` is not valid for the selected finding kind.")
        return self

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            self.model_dump(mode="json"),
            "knowledge maintenance evaluation finding",
        )


class KnowledgeMaintenanceEvaluatorDecision(_PlanningModel):
    """Strict output from the injected independent semantic evaluator."""

    plan_fingerprint: StrictStr
    routing_result_fingerprint: StrictStr
    configuration_fingerprint: StrictStr
    verdict: KnowledgeMaintenanceEvaluationVerdict
    findings: tuple[KnowledgeMaintenanceEvaluationFinding, ...] = ()

    @field_validator(
        "plan_fingerprint",
        "routing_result_fingerprint",
        "configuration_fingerprint",
    )
    @classmethod
    def validate_fingerprints(cls, value: str, info) -> str:
        return _validate_fingerprint(value, info.field_name)

    @field_validator("findings", mode="before")
    @classmethod
    def copy_findings(
        cls,
        value: object,
    ) -> tuple[KnowledgeMaintenanceEvaluationFinding, ...]:
        if not isinstance(value, list | tuple):
            raise ValueError("`findings` must be an ordered array.")
        if len(value) > MAX_KNOWLEDGE_MAINTENANCE_EVALUATION_FINDINGS:
            raise ValueError(
                "`findings` must contain at most "
                f"{MAX_KNOWLEDGE_MAINTENANCE_EVALUATION_FINDINGS} values."
            )
        copied = tuple(
            sorted(
                (
                    KnowledgeMaintenanceEvaluationFinding.model_validate(
                        item.model_dump(mode="python")
                        if isinstance(item, KnowledgeMaintenanceEvaluationFinding)
                        else item
                    )
                    for item in value
                ),
                key=lambda item: item.fingerprint,
            )
        )
        fingerprints = tuple(item.fingerprint for item in copied)
        if len(fingerprints) != len(set(fingerprints)):
            raise ValueError("`findings` cannot contain duplicates.")
        return copied

    @model_validator(mode="after")
    def validate_verdict(self) -> KnowledgeMaintenanceEvaluatorDecision:
        if self.verdict is KnowledgeMaintenanceEvaluationVerdict.ACCEPTED and self.findings:
            raise ValueError("An accepted evaluation cannot contain findings.")
        if self.verdict is KnowledgeMaintenanceEvaluationVerdict.REJECTED and not self.findings:
            raise ValueError("A rejected evaluation requires at least one finding.")
        return self


class KnowledgeMaintenancePlanEvaluation(_PlanningModel):
    """Final independent evaluation, including whether the injected evaluator ran."""

    deterministic_evaluator_version: Literal[
        "cayu.knowledge-maintenance-deterministic-evaluator.v1"
    ] = KNOWLEDGE_MAINTENANCE_DETERMINISTIC_EVALUATOR_VERSION
    plan_fingerprint: StrictStr
    routing_result_fingerprint: StrictStr
    configuration_fingerprint: StrictStr
    evaluator_id: StrictStr
    evaluator_version: StrictStr
    evaluator_invoked: StrictBool
    verdict: KnowledgeMaintenanceEvaluationVerdict
    code: StrictStr
    findings: tuple[KnowledgeMaintenanceEvaluationFinding, ...] = ()

    @field_validator(
        "plan_fingerprint",
        "routing_result_fingerprint",
        "configuration_fingerprint",
    )
    @classmethod
    def validate_fingerprints(cls, value: str, info) -> str:
        return _validate_fingerprint(value, info.field_name)

    @field_validator("evaluator_id", "evaluator_version")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _clean(value, info.field_name)

    @field_validator("code")
    @classmethod
    def validate_code(cls, value: str) -> str:
        return _safe_code(value, "code")

    @field_validator("findings", mode="before")
    @classmethod
    def copy_findings(
        cls,
        value: object,
    ) -> tuple[KnowledgeMaintenanceEvaluationFinding, ...]:
        return KnowledgeMaintenanceEvaluatorDecision.copy_findings(value)

    @model_validator(mode="after")
    def validate_verdict(self) -> KnowledgeMaintenancePlanEvaluation:
        if self.verdict is KnowledgeMaintenanceEvaluationVerdict.ACCEPTED and self.findings:
            raise ValueError("An accepted evaluation cannot contain findings.")
        if self.verdict is KnowledgeMaintenanceEvaluationVerdict.REJECTED and not self.findings:
            raise ValueError("A rejected evaluation requires at least one finding.")
        return self

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            {
                "contract": "cayu.knowledge-maintenance-plan-evaluation.v1",
                "evaluation": self.model_dump(mode="json"),
            },
            "knowledge maintenance plan evaluation",
        )


class KnowledgeMaintenancePlanningSnapshot(_PlanningModel):
    """Minimum authorized routing material disclosed to planning components."""

    request_id: StrictStr
    policy_id: StrictStr
    namespace: StrictStr
    labels: dict[str, str] = Field(default_factory=dict)
    routing_request_fingerprint: StrictStr
    routing_result_fingerprint: StrictStr
    routing_configuration_fingerprint: StrictStr
    candidate_payload_bytes: StrictInt
    candidates: tuple[KnowledgeMaintenanceRoutedCandidate, ...] = ()
    routed_signals: tuple[KnowledgeMaintenanceCandidateSignal, ...] = ()

    @field_validator("request_id", "policy_id", "namespace")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _clean(value, info.field_name)

    @field_validator("labels", mode="before")
    @classmethod
    def copy_labels(cls, value: object) -> dict[str, str]:
        return copy_label_map(value, "labels")

    @field_validator(
        "routing_request_fingerprint",
        "routing_result_fingerprint",
        "routing_configuration_fingerprint",
    )
    @classmethod
    def validate_fingerprints(cls, value: str, info) -> str:
        return _validate_fingerprint(value, info.field_name)

    @field_validator("candidate_payload_bytes")
    @classmethod
    def validate_candidate_payload_bytes(cls, value: int) -> int:
        if value < 0:
            raise ValueError("`candidate_payload_bytes` must be non-negative.")
        return value

    @field_validator("candidates", mode="before")
    @classmethod
    def copy_candidates(cls, value: object) -> tuple[KnowledgeMaintenanceRoutedCandidate, ...]:
        if not isinstance(value, list | tuple):
            raise ValueError("`candidates` must be an ordered array.")
        if len(value) > MAX_KNOWLEDGE_MAINTENANCE_SOURCES:
            raise ValueError(
                f"`candidates` must contain at most {MAX_KNOWLEDGE_MAINTENANCE_SOURCES} values."
            )
        return tuple(
            KnowledgeMaintenanceRoutedCandidate.model_validate(
                item.model_dump(mode="python")
                if isinstance(item, KnowledgeMaintenanceRoutedCandidate)
                else item
            )
            for item in value
        )

    @field_validator("routed_signals", mode="before")
    @classmethod
    def copy_routed_signals(
        cls,
        value: object,
    ) -> tuple[KnowledgeMaintenanceCandidateSignal, ...]:
        if not isinstance(value, list | tuple):
            raise ValueError("`routed_signals` must be an ordered array.")
        if len(value) > MAX_KNOWLEDGE_MAINTENANCE_ROUTING_SIGNALS:
            raise ValueError(
                "`routed_signals` must contain at most "
                f"{MAX_KNOWLEDGE_MAINTENANCE_ROUTING_SIGNALS} values."
            )
        return tuple(
            KnowledgeMaintenanceCandidateSignal.model_validate(
                item.model_dump(mode="python")
                if isinstance(item, KnowledgeMaintenanceCandidateSignal)
                else item
            )
            for item in value
        )

    @model_validator(mode="after")
    def validate_authorized_snapshot(self) -> KnowledgeMaintenancePlanningSnapshot:
        references = {
            (candidate.reference.entry_id, candidate.reference.revision)
            for candidate in self.candidates
        }
        if len(references) != len(self.candidates):
            raise ValueError("`candidates` cannot repeat an exact revision.")
        signal_ids = tuple(signal.id for signal in self.routed_signals)
        if len(signal_ids) != len(set(signal_ids)):
            raise ValueError("`routed_signals` cannot repeat a signal identity.")
        signal_references = {
            (reference.entry_id, reference.revision)
            for signal in self.routed_signals
            for reference in signal.references
        }
        if references != signal_references:
            raise ValueError("Candidates must exactly cover routed signal references.")
        for candidate in self.candidates:
            if candidate.entry.namespace != self.namespace:
                raise ValueError("A planning candidate is outside the requested namespace.")
            if any(candidate.entry.labels.get(key) != value for key, value in self.labels.items()):
                raise ValueError("A planning candidate is outside the requested label scope.")
            expected_signal_ids = tuple(
                signal.id
                for signal in self.routed_signals
                if candidate.reference in signal.references
            )
            if candidate.signal_ids != expected_signal_ids:
                raise ValueError("Candidate signal identities do not match routed signals.")
            expected_signal_kinds = {
                signal.kind
                for signal in self.routed_signals
                if candidate.reference in signal.references
            }
            if set(candidate.signal_kinds) != expected_signal_kinds:
                raise ValueError("Candidate signal kinds do not match routed signals.")
        if self.candidate_payload_bytes != _snapshot_payload_bytes(
            self.candidates,
            self.routed_signals,
        ):
            raise ValueError("`candidate_payload_bytes` does not match the routed payload.")
        return self

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            {
                "contract": "cayu.knowledge-maintenance-planning-snapshot.v1",
                "snapshot": self.model_dump(mode="json"),
            },
            "knowledge maintenance planning snapshot",
        )


class KnowledgeMaintenancePlannerInput(_PlanningModel):
    """Minimized routed snapshot and budget passed to the untrusted planner."""

    snapshot: KnowledgeMaintenancePlanningSnapshot
    configuration_fingerprint: StrictStr
    allowed_replacement_kinds: tuple[StrictStr, ...]
    budget: KnowledgeMaintenancePlannerBudget

    @field_validator("snapshot", mode="before")
    @classmethod
    def copy_snapshot(cls, value: object) -> object:
        if isinstance(value, KnowledgeMaintenancePlanningSnapshot):
            return value.model_dump(mode="python")
        return value

    @field_validator("configuration_fingerprint")
    @classmethod
    def validate_configuration_fingerprint(cls, value: str) -> str:
        return _validate_fingerprint(value, "configuration_fingerprint")

    @field_validator("allowed_replacement_kinds", mode="before")
    @classmethod
    def copy_allowed_kinds(cls, value: object) -> tuple[str, ...]:
        return KnowledgeMaintenancePlanningConfig.copy_allowed_kinds(value)

    @field_validator("budget", mode="before")
    @classmethod
    def copy_budget(cls, value: object) -> object:
        if isinstance(value, KnowledgeMaintenancePlannerBudget):
            return value.model_dump(mode="python")
        return value

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            {
                "contract": "cayu.knowledge-maintenance-planner-input.v1",
                "input": self.model_dump(mode="json"),
            },
            "knowledge maintenance planner input",
        )


class KnowledgeMaintenanceEvaluatorInput(_PlanningModel):
    """Structurally valid plan and exact snapshot passed to the evaluator."""

    planner_input: KnowledgeMaintenancePlannerInput
    plan: KnowledgeMaintenancePlanDraft
    budget: KnowledgeMaintenanceStageBudget

    @field_validator("planner_input", mode="before")
    @classmethod
    def copy_planner_input(cls, value: object) -> object:
        if isinstance(value, KnowledgeMaintenancePlannerInput):
            return value.model_dump(mode="python")
        return value

    @field_validator("plan", mode="before")
    @classmethod
    def copy_plan(cls, value: object) -> object:
        if isinstance(value, KnowledgeMaintenancePlanDraft):
            return value.model_dump(mode="python")
        return value

    @field_validator("budget", mode="before")
    @classmethod
    def copy_budget(cls, value: object) -> object:
        if isinstance(value, KnowledgeMaintenanceStageBudget):
            return value.model_dump(mode="python")
        return value


class KnowledgeMaintenancePlannerOutput(_PlanningModel):
    """Strict planner response including provider usage accounting."""

    plan: KnowledgeMaintenancePlanDraft
    usage: KnowledgeMaintenanceInferenceUsage = Field(
        default_factory=KnowledgeMaintenanceInferenceUsage
    )

    @field_validator("plan", mode="before")
    @classmethod
    def copy_plan(cls, value: object) -> object:
        if isinstance(value, KnowledgeMaintenancePlanDraft):
            return value.model_dump(mode="python")
        return value

    @field_validator("usage", mode="before")
    @classmethod
    def copy_usage(cls, value: object) -> object:
        if isinstance(value, KnowledgeMaintenanceInferenceUsage):
            return value.model_dump(mode="python")
        return value


class KnowledgeMaintenanceEvaluatorOutput(_PlanningModel):
    """Strict evaluator response including independent provider usage."""

    decision: KnowledgeMaintenanceEvaluatorDecision
    usage: KnowledgeMaintenanceInferenceUsage = Field(
        default_factory=KnowledgeMaintenanceInferenceUsage
    )

    @field_validator("decision", mode="before")
    @classmethod
    def copy_decision(cls, value: object) -> object:
        if isinstance(value, KnowledgeMaintenanceEvaluatorDecision):
            return value.model_dump(mode="python")
        return value

    @field_validator("usage", mode="before")
    @classmethod
    def copy_usage(cls, value: object) -> object:
        if isinstance(value, KnowledgeMaintenanceInferenceUsage):
            return value.model_dump(mode="python")
        return value


class KnowledgeMaintenancePlanningResult(_PlanningModel):
    """Read-only result; accepted plans still have no storage authority."""

    schema_version: Literal[1] = KNOWLEDGE_MAINTENANCE_PLANNING_SCHEMA_VERSION
    request_id: StrictStr
    request_fingerprint: StrictStr
    routing_result_fingerprint: StrictStr
    configuration_fingerprint: StrictStr
    planner_id: StrictStr
    planner_version: StrictStr
    evaluator_id: StrictStr
    evaluator_version: StrictStr
    outcome: KnowledgeMaintenancePlanningOutcome
    code: StrictStr
    plan: KnowledgeMaintenancePlanDraft | None = None
    evaluation: KnowledgeMaintenancePlanEvaluation | None = None
    planner_usage: KnowledgeMaintenanceInferenceUsage | None = None
    evaluator_usage: KnowledgeMaintenanceInferenceUsage | None = None
    processed_at: datetime

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> object:
        if type(value) is not int or value != KNOWLEDGE_MAINTENANCE_PLANNING_SCHEMA_VERSION:
            raise ValueError("`schema_version` must be the integer 1.")
        return value

    @field_validator("request_id")
    @classmethod
    def validate_request_id(cls, value: str) -> str:
        return _clean(value, "request_id")

    @field_validator(
        "request_fingerprint",
        "routing_result_fingerprint",
        "configuration_fingerprint",
    )
    @classmethod
    def validate_fingerprints(cls, value: str, info) -> str:
        return _validate_fingerprint(value, info.field_name)

    @field_validator("planner_id", "planner_version", "evaluator_id", "evaluator_version")
    @classmethod
    def validate_component_identity(cls, value: str, info) -> str:
        return _clean(value, info.field_name)

    @field_validator("code")
    @classmethod
    def validate_code(cls, value: str) -> str:
        return _safe_code(value, "code")

    @field_validator("plan", mode="before")
    @classmethod
    def copy_plan(cls, value: object) -> object:
        if isinstance(value, KnowledgeMaintenancePlanDraft):
            return value.model_dump(mode="python")
        return value

    @field_validator("evaluation", mode="before")
    @classmethod
    def copy_evaluation(cls, value: object) -> object:
        if isinstance(value, KnowledgeMaintenancePlanEvaluation):
            return value.model_dump(mode="python")
        return value

    @field_validator("planner_usage", "evaluator_usage", mode="before")
    @classmethod
    def copy_usage(cls, value: object) -> object:
        if isinstance(value, KnowledgeMaintenanceInferenceUsage):
            return value.model_dump(mode="python")
        return value

    @field_validator("processed_at")
    @classmethod
    def validate_processed_at(cls, value: datetime) -> datetime:
        return _utc(value, "processed_at")

    @model_validator(mode="after")
    def validate_result_shape(self) -> KnowledgeMaintenancePlanningResult:
        plan_required = {
            KnowledgeMaintenancePlanningOutcome.SOURCE_REVALIDATION_FAILED_AFTER_PLANNING,
            KnowledgeMaintenancePlanningOutcome.SOURCE_REVALIDATION_FAILED_AFTER_EVALUATION,
            KnowledgeMaintenancePlanningOutcome.DETERMINISTIC_REJECTED,
            KnowledgeMaintenancePlanningOutcome.EVALUATOR_TIMED_OUT,
            KnowledgeMaintenancePlanningOutcome.EVALUATOR_FAILED,
            KnowledgeMaintenancePlanningOutcome.EVALUATOR_INVALID,
            KnowledgeMaintenancePlanningOutcome.EVALUATOR_OVER_BUDGET,
            KnowledgeMaintenancePlanningOutcome.EVALUATOR_REJECTED,
            KnowledgeMaintenancePlanningOutcome.ACCEPTED,
        }
        evaluation_required = {
            KnowledgeMaintenancePlanningOutcome.DETERMINISTIC_REJECTED,
            KnowledgeMaintenancePlanningOutcome.EVALUATOR_REJECTED,
            KnowledgeMaintenancePlanningOutcome.ACCEPTED,
        }
        if (self.plan is not None) != (self.outcome in plan_required):
            raise ValueError("`plan` presence does not match the planning outcome.")
        if (self.evaluation is not None) != (self.outcome in evaluation_required):
            raise ValueError("`evaluation` presence does not match the planning outcome.")
        if self.plan is not None and (
            self.plan.routing_request_fingerprint != self.request_fingerprint
            or self.plan.routing_result_fingerprint != self.routing_result_fingerprint
            or self.plan.configuration_fingerprint != self.configuration_fingerprint
        ):
            raise ValueError("The returned plan does not bind the planning result.")
        if self.evaluation is not None and self.plan is not None:
            if self.evaluation.plan_fingerprint != self.plan.fingerprint:
                raise ValueError("The evaluation does not bind the returned plan.")
            if self.evaluation.routing_result_fingerprint != self.routing_result_fingerprint:
                raise ValueError("The evaluation does not bind the routing result.")
            if self.evaluation.configuration_fingerprint != self.configuration_fingerprint:
                raise ValueError("The evaluation does not bind the planning configuration.")
            if (
                self.evaluation.evaluator_id != self.evaluator_id
                or self.evaluation.evaluator_version != self.evaluator_version
            ):
                raise ValueError("The evaluation does not bind the configured evaluator.")
            if self.evaluation.code != self.code:
                raise ValueError("The evaluation code does not match the planning result.")
            if self.evaluation.evaluator_invoked:
                source_keys = {
                    (reference.entry_id, reference.revision)
                    for reference in self.plan.source_references
                }
                mapping_ids = {mapping.id for mapping in self.plan.evidence_mappings}
                if any(
                    (reference.entry_id, reference.revision) not in source_keys
                    for finding in self.evaluation.findings
                    for reference in finding.source_references
                ) or any(
                    mapping_id not in mapping_ids
                    for finding in self.evaluation.findings
                    for mapping_id in finding.evidence_mapping_ids
                ):
                    raise ValueError("Evaluator findings do not bind the returned plan.")
        if self.outcome is KnowledgeMaintenancePlanningOutcome.ACCEPTED and (
            self.evaluation is None
            or not self.evaluation.evaluator_invoked
            or self.evaluation.verdict is not KnowledgeMaintenanceEvaluationVerdict.ACCEPTED
        ):
            raise ValueError("An accepted result requires an invoked, accepted evaluation.")
        if self.outcome in {
            KnowledgeMaintenancePlanningOutcome.DETERMINISTIC_REJECTED,
            KnowledgeMaintenancePlanningOutcome.EVALUATOR_REJECTED,
        } and (
            self.evaluation is None
            or self.evaluation.verdict is not KnowledgeMaintenanceEvaluationVerdict.REJECTED
        ):
            raise ValueError("A rejected result requires a rejected evaluation.")
        if (
            self.outcome is KnowledgeMaintenancePlanningOutcome.EVALUATOR_REJECTED
            and self.evaluation is not None
            and not self.evaluation.evaluator_invoked
        ):
            raise ValueError("An evaluator rejection requires an invoked evaluator.")
        planner_usage_required = self.outcome in (
            plan_required | {KnowledgeMaintenancePlanningOutcome.PLANNER_OVER_BUDGET}
        )
        if (self.planner_usage is not None) != planner_usage_required:
            raise ValueError("Planner usage presence does not match the planning outcome.")
        evaluator_usage_required = (
            self.evaluation is not None and self.evaluation.evaluator_invoked
        ) or self.outcome in {
            KnowledgeMaintenancePlanningOutcome.SOURCE_REVALIDATION_FAILED_AFTER_EVALUATION,
            KnowledgeMaintenancePlanningOutcome.EVALUATOR_OVER_BUDGET,
        }
        evaluator_usage_optional = (
            self.outcome is KnowledgeMaintenancePlanningOutcome.EVALUATOR_INVALID
        )
        if evaluator_usage_required and self.evaluator_usage is None:
            raise ValueError("Evaluator usage is required for the planning outcome.")
        if (
            self.evaluator_usage is not None
            and not evaluator_usage_required
            and not evaluator_usage_optional
        ):
            raise ValueError("Evaluator usage is not valid for the planning outcome.")
        if self.evaluator_usage is not None and self.planner_usage is None:
            raise ValueError("Evaluator usage requires planner usage.")
        return self

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            {
                "contract": "cayu.knowledge-maintenance-planning-result.v1",
                "result": self.model_dump(mode="json"),
            },
            "knowledge maintenance planning result",
        )


class KnowledgeMaintenancePlanner(Protocol):
    """Provider-neutral proposal generation over one copied bounded snapshot."""

    async def propose_maintenance(
        self,
        request: KnowledgeMaintenancePlannerInput,
    ) -> KnowledgeMaintenancePlannerOutput: ...


class KnowledgeMaintenancePlanEvaluator(Protocol):
    """Independent semantic and application-policy evaluation of one plan."""

    async def evaluate_maintenance_plan(
        self,
        request: KnowledgeMaintenanceEvaluatorInput,
    ) -> KnowledgeMaintenanceEvaluatorOutput: ...


class _PlanningStore(Protocol):
    async def get_entry(
        self,
        entry_id: str,
        *,
        revision: int | None = None,
        max_bytes: int | None = None,
        access_scope: KnowledgeAccessScope,
    ) -> KnowledgeEntry | None: ...


class _SourceState(StrEnum):
    CURRENT = "current"
    CHANGED = "changed"
    FAILED = "failed"


class KnowledgeMaintenancePlanningWorkflow:
    """Generate and independently evaluate a bounded plan without writes."""

    def __init__(
        self,
        store: _PlanningStore,
        *,
        planner: KnowledgeMaintenancePlanner,
        evaluator: KnowledgeMaintenancePlanEvaluator,
        config: KnowledgeMaintenancePlanningConfig,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not callable(getattr(store, "get_entry", None)):
            raise TypeError("store must implement `get_entry`.")
        if planner is evaluator:
            raise ValueError("Planner and evaluator must be separate components.")
        propose = getattr(planner, "propose_maintenance", None)
        evaluate = getattr(evaluator, "evaluate_maintenance_plan", None)
        if not callable(propose):
            raise TypeError("planner must implement `propose_maintenance`.")
        if not callable(evaluate):
            raise TypeError("evaluator must implement `evaluate_maintenance_plan`.")
        if type(config) is not KnowledgeMaintenancePlanningConfig:
            raise TypeError("config must be a KnowledgeMaintenancePlanningConfig.")
        self._store = store
        self._propose = propose
        self._evaluate = evaluate
        self._config = KnowledgeMaintenancePlanningConfig.model_validate(
            config.model_dump(mode="python")
        )
        self._clock = utc_clock(clock)
        self._invocation_tasks: set[asyncio.Task[Any]] = set()

    @property
    def config(self) -> KnowledgeMaintenancePlanningConfig:
        return KnowledgeMaintenancePlanningConfig.model_validate(
            self._config.model_dump(mode="python")
        )

    async def plan(
        self,
        routing_request: KnowledgeMaintenanceRoutingRequest,
        routing_result: KnowledgeMaintenanceRoutingResult,
    ) -> KnowledgeMaintenancePlanningResult:
        """Plan and evaluate one exact routed snapshot, failing closed."""

        if type(routing_request) is not KnowledgeMaintenanceRoutingRequest:
            raise TypeError("routing_request must be a KnowledgeMaintenanceRoutingRequest.")
        if type(routing_result) is not KnowledgeMaintenanceRoutingResult:
            raise TypeError("routing_result must be a KnowledgeMaintenanceRoutingResult.")
        if len(routing_result.candidates) > MAX_KNOWLEDGE_MAINTENANCE_SOURCES:
            raise KnowledgeMaintenancePlanningLimitExceeded("max_candidates")
        copied_request = KnowledgeMaintenanceRoutingRequest.model_validate(
            routing_request.model_dump(mode="python")
        )
        copied_result = KnowledgeMaintenanceRoutingResult.model_validate(
            routing_result.model_dump(mode="python")
        )
        request_fingerprint = copied_request.fingerprint
        if (
            copied_result.request_id != copied_request.id
            or copied_result.request_fingerprint != request_fingerprint
        ):
            raise ValueError("The routing result does not bind the supplied routing request.")
        self._validate_routing_dispositions(copied_request, copied_result)
        routing_result_fingerprint = copied_result.fingerprint
        if any(
            candidate.entry.namespace != copied_request.namespace
            or any(
                candidate.entry.labels.get(key) != value
                for key, value in copied_request.labels.items()
            )
            for candidate in copied_result.candidates
        ):
            raise ValueError("The routing result contains a candidate outside request scope.")
        snapshot = KnowledgeMaintenancePlanningSnapshot(
            request_id=copied_request.id,
            policy_id=copied_request.policy_id,
            namespace=copied_request.namespace,
            labels=dict(copied_request.labels),
            routing_request_fingerprint=request_fingerprint,
            routing_result_fingerprint=routing_result_fingerprint,
            routing_configuration_fingerprint=copied_result.configuration_fingerprint,
            candidate_payload_bytes=copied_result.candidate_payload_bytes,
            candidates=copied_result.candidates,
            routed_signals=copied_result.routed_signals,
        )
        # These fields are runtime-owned validated values. Avoid recopying the full
        # snapshot here; each component still receives its own deep defensive copy.
        planner_input = KnowledgeMaintenancePlannerInput.model_construct(
            snapshot=snapshot,
            configuration_fingerprint=self._config.fingerprint,
            allowed_replacement_kinds=tuple(self._config.allowed_replacement_kinds),
            budget=self._config.planner_budget,
        )
        if copied_result.truncated:
            return self._result(
                planner_input,
                outcome=KnowledgeMaintenancePlanningOutcome.ROUTING_INCOMPLETE,
                code="routing_incomplete",
            )
        if not copied_result.candidates:
            return self._result(
                planner_input,
                outcome=KnowledgeMaintenancePlanningOutcome.NO_CANDIDATES,
                code="no_candidates",
            )

        self._validate_pre_invocation_bounds(planner_input)

        source_state = await self._revalidate_sources(
            planner_input,
            access_scope=copied_request.access_scope,
        )
        if source_state is _SourceState.CHANGED:
            return self._result(
                planner_input,
                outcome=KnowledgeMaintenancePlanningOutcome.SOURCE_SET_CHANGED,
                code="source_set_changed",
            )
        if source_state is _SourceState.FAILED:
            return self._result(
                planner_input,
                outcome=KnowledgeMaintenancePlanningOutcome.SOURCE_REVALIDATION_FAILED,
                code="source_revalidation_failed",
            )

        planner_output, planner_failure = await self._invoke_planner(planner_input)
        if planner_failure is not None:
            return self._result(
                planner_input,
                outcome=planner_failure,
                code=planner_failure.value,
                planner_usage=(planner_output.usage if planner_output is not None else None),
            )
        if planner_output is None:  # pragma: no cover - internal invariant
            raise RuntimeError("Planner invocation completed without an output or failure.")
        plan = planner_output.plan
        planner_usage = planner_output.usage

        findings = self._deterministic_findings(planner_input, plan)
        if findings:
            evaluation = self._final_evaluation(
                planner_input,
                plan,
                verdict=KnowledgeMaintenanceEvaluationVerdict.REJECTED,
                code="deterministic_validation_rejected",
                findings=findings,
                evaluator_invoked=False,
            )
            return self._result(
                planner_input,
                outcome=KnowledgeMaintenancePlanningOutcome.DETERMINISTIC_REJECTED,
                code="deterministic_validation_rejected",
                plan=plan,
                evaluation=evaluation,
                planner_usage=planner_usage,
            )

        source_state = await self._revalidate_sources(
            planner_input,
            access_scope=copied_request.access_scope,
        )
        if source_state is _SourceState.FAILED:
            return self._result(
                planner_input,
                outcome=(
                    KnowledgeMaintenancePlanningOutcome.SOURCE_REVALIDATION_FAILED_AFTER_PLANNING
                ),
                code="source_revalidation_failed_after_planning",
                plan=plan,
                planner_usage=planner_usage,
            )
        if source_state is _SourceState.CHANGED:
            code = "source_set_changed_during_planning"
            evaluation = self._final_evaluation(
                planner_input,
                plan,
                verdict=KnowledgeMaintenanceEvaluationVerdict.REJECTED,
                code=code,
                findings=(
                    KnowledgeMaintenanceEvaluationFinding(
                        kind=KnowledgeMaintenanceEvaluationFindingKind.STALE_SOURCE,
                        code=KnowledgeMaintenanceEvaluationFindingCode.STALE_SOURCE,
                    ),
                ),
                evaluator_invoked=False,
            )
            return self._result(
                planner_input,
                outcome=KnowledgeMaintenancePlanningOutcome.DETERMINISTIC_REJECTED,
                code=code,
                plan=plan,
                evaluation=evaluation,
                planner_usage=planner_usage,
            )

        evaluator_input = KnowledgeMaintenanceEvaluatorInput.model_construct(
            planner_input=planner_input.model_copy(deep=True),
            plan=plan.model_copy(deep=True),
            budget=self._config.evaluator_budget.model_copy(deep=True),
        )
        evaluator_input_bytes = _canonical_bytes(
            evaluator_input,
            "knowledge maintenance evaluator input",
        )
        if evaluator_input_bytes > self._config.max_evaluator_input_bytes:
            raise KnowledgeMaintenancePlanningLimitExceeded("max_evaluator_input_bytes")

        evaluator_output, evaluator_failure = await self._invoke_evaluator(evaluator_input)
        if evaluator_failure is not None:
            return self._result(
                planner_input,
                outcome=evaluator_failure,
                code=evaluator_failure.value,
                plan=plan,
                planner_usage=planner_usage,
                evaluator_usage=(evaluator_output.usage if evaluator_output is not None else None),
            )
        if evaluator_output is None:  # pragma: no cover - internal invariant
            raise RuntimeError("Evaluator invocation completed without an output or failure.")
        decision = evaluator_output.decision
        evaluator_usage = evaluator_output.usage
        if not self._valid_evaluator_decision(planner_input, plan, decision):
            return self._result(
                planner_input,
                outcome=KnowledgeMaintenancePlanningOutcome.EVALUATOR_INVALID,
                code="evaluator_invalid",
                plan=plan,
                planner_usage=planner_usage,
                evaluator_usage=evaluator_usage,
            )

        source_state = await self._revalidate_sources(
            planner_input,
            access_scope=copied_request.access_scope,
        )
        if source_state is _SourceState.FAILED:
            return self._result(
                planner_input,
                outcome=(
                    KnowledgeMaintenancePlanningOutcome.SOURCE_REVALIDATION_FAILED_AFTER_EVALUATION
                ),
                code="source_revalidation_failed_after_evaluation",
                plan=plan,
                planner_usage=planner_usage,
                evaluator_usage=evaluator_usage,
            )
        if source_state is _SourceState.CHANGED:
            code = "source_set_changed_during_evaluation"
            evaluation = self._final_evaluation(
                planner_input,
                plan,
                verdict=KnowledgeMaintenanceEvaluationVerdict.REJECTED,
                code=code,
                findings=(
                    KnowledgeMaintenanceEvaluationFinding(
                        kind=KnowledgeMaintenanceEvaluationFindingKind.STALE_SOURCE,
                        code=KnowledgeMaintenanceEvaluationFindingCode.STALE_SOURCE,
                    ),
                ),
                evaluator_invoked=True,
            )
            return self._result(
                planner_input,
                outcome=KnowledgeMaintenancePlanningOutcome.DETERMINISTIC_REJECTED,
                code=code,
                plan=plan,
                evaluation=evaluation,
                planner_usage=planner_usage,
                evaluator_usage=evaluator_usage,
            )

        decision_code = (
            "evaluator_accepted"
            if decision.verdict is KnowledgeMaintenanceEvaluationVerdict.ACCEPTED
            else "evaluator_rejected"
        )
        evaluation = self._final_evaluation(
            planner_input,
            plan,
            verdict=decision.verdict,
            code=decision_code,
            findings=decision.findings,
            evaluator_invoked=True,
        )
        outcome = (
            KnowledgeMaintenancePlanningOutcome.ACCEPTED
            if decision.verdict is KnowledgeMaintenanceEvaluationVerdict.ACCEPTED
            else KnowledgeMaintenancePlanningOutcome.EVALUATOR_REJECTED
        )
        return self._result(
            planner_input,
            outcome=outcome,
            code=decision_code,
            plan=plan,
            evaluation=evaluation,
            planner_usage=planner_usage,
            evaluator_usage=evaluator_usage,
        )

    @staticmethod
    def _validate_routing_dispositions(
        request: KnowledgeMaintenanceRoutingRequest,
        result: KnowledgeMaintenanceRoutingResult,
    ) -> None:
        request_signals = {signal.id: signal for signal in request.signals}
        disposition_ids = {
            *(signal.id for signal in result.routed_signals),
            *(omission.signal_id for omission in result.omissions),
        }
        valid = (
            result.signal_count == len(request_signals)
            and disposition_ids == set(request_signals)
            and all(request_signals.get(signal.id) == signal for signal in result.routed_signals)
            and all(
                (expected := request_signals.get(omission.signal_id)) is not None
                and omission.signal_kind is expected.kind
                for omission in result.omissions
            )
        )
        if not valid:
            raise ValueError("The routing result does not bind the supplied routing request.")

    def _validate_pre_invocation_bounds(
        self,
        planner_input: KnowledgeMaintenancePlannerInput,
    ) -> None:
        input_bytes = _canonical_bytes(planner_input, "knowledge maintenance planner input")
        if input_bytes > self._config.max_planner_input_bytes:
            raise KnowledgeMaintenancePlanningLimitExceeded("max_planner_input_bytes")
        if planner_input.snapshot.candidate_payload_bytes > self._config.max_revalidation_bytes:
            raise KnowledgeMaintenancePlanningLimitExceeded("max_revalidation_bytes")

    async def _revalidate_sources(
        self,
        planner_input: KnowledgeMaintenancePlannerInput,
        *,
        access_scope: KnowledgeAccessScope,
    ) -> _SourceState:
        semaphore = asyncio.Semaphore(self._config.max_revalidation_concurrency)
        copied_access_scope = copy_knowledge_access_scope(access_scope)

        async def revalidate(candidate) -> _SourceState:
            expected = candidate.entry
            expected_bytes = knowledge_entry_payload_bytes(expected)
            try:
                async with semaphore:
                    current = await self._store.get_entry(
                        expected.id,
                        max_bytes=expected_bytes,
                        access_scope=copied_access_scope,
                    )
            except KnowledgeEntryReadLimitExceeded as exc:
                if exc.entry_id == expected.id and exc.revision != expected.revision:
                    return _SourceState.CHANGED
                return _SourceState.FAILED
            except asyncio.CancelledError:
                raise
            except Exception:
                return _SourceState.FAILED
            if current is None:
                return _SourceState.CHANGED
            if type(current) is not KnowledgeEntry or current.id != expected.id:
                return _SourceState.FAILED
            if current.revision != expected.revision:
                return _SourceState.CHANGED
            if current != expected:
                return _SourceState.FAILED
            return _SourceState.CURRENT

        try:
            async with asyncio.timeout(self._config.source_revalidation_timeout_seconds):
                states = await asyncio.gather(
                    *(revalidate(candidate) for candidate in planner_input.snapshot.candidates)
                )
        except TimeoutError:
            return _SourceState.FAILED
        if any(state is _SourceState.FAILED for state in states):
            return _SourceState.FAILED
        if any(state is _SourceState.CHANGED for state in states):
            return _SourceState.CHANGED
        return _SourceState.CURRENT

    async def _invoke_component(
        self,
        operation_factory: Callable[[], Awaitable[_InvocationT]],
        *,
        timeout_seconds: float,
        task_name: str,
    ) -> tuple[_InvocationT | None, bool]:
        """Await an injected component without yielding cancellation ownership."""

        async def invoke() -> _InvocationT:
            return await operation_factory()

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        task = asyncio.create_task(invoke(), name=task_name)
        self._retain_invocation_task(task)
        try:
            await asyncio.wait({task}, timeout=timeout_seconds)
        except asyncio.CancelledError:
            task.cancel("Knowledge maintenance planning was cancelled by its caller.")
            raise

        if not task.done() or loop.time() >= deadline:
            if not task.done():
                task.cancel("Knowledge maintenance component exceeded its deadline.")
            return None, True

        # Completion and caller cancellation may become ready in the same event-loop
        # turn. This checkpoint keeps the caller's signal authoritative even when the
        # independently owned component has already produced a result.
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            task.cancel("Knowledge maintenance planning was cancelled by its caller.")
            raise
        if loop.time() >= deadline:
            return None, True
        return task.result(), False

    def _retain_invocation_task(self, task: asyncio.Task[Any]) -> None:
        self._invocation_tasks.add(task)

        def settled(completed: asyncio.Task[Any]) -> None:
            self._invocation_tasks.discard(completed)
            if not completed.cancelled():
                with suppress(BaseException):
                    completed.exception()

        task.add_done_callback(settled)

    async def _invoke_planner(
        self,
        planner_input: KnowledgeMaintenancePlannerInput,
    ) -> tuple[
        KnowledgeMaintenancePlannerOutput | None,
        KnowledgeMaintenancePlanningOutcome | None,
    ]:
        try:
            raw_output, timed_out = await self._invoke_component(
                lambda: self._propose(planner_input.model_copy(deep=True)),
                timeout_seconds=self._config.planner_timeout_seconds,
                task_name="cayu-knowledge-maintenance-planner",
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            return None, KnowledgeMaintenancePlanningOutcome.PLANNER_FAILED
        if timed_out:
            return None, KnowledgeMaintenancePlanningOutcome.PLANNER_TIMED_OUT
        if type(raw_output) is not KnowledgeMaintenancePlannerOutput:
            return None, KnowledgeMaintenancePlanningOutcome.PLANNER_INVALID
        try:
            output = KnowledgeMaintenancePlannerOutput.model_validate(
                raw_output.model_dump(mode="python")
            )
        except (TypeError, ValueError):
            return None, KnowledgeMaintenancePlanningOutcome.PLANNER_INVALID
        if not _usage_model_identity_authorized(output.usage, self._config.planner_budget):
            return None, KnowledgeMaintenancePlanningOutcome.PLANNER_INVALID
        if _canonical_bytes(output.plan, "knowledge maintenance plan") > (
            self._config.max_plan_bytes
        ):
            return output, KnowledgeMaintenancePlanningOutcome.PLANNER_OVER_BUDGET
        if len(output.plan.evidence_mappings) > self._config.max_evidence_mappings:
            return output, KnowledgeMaintenancePlanningOutcome.PLANNER_OVER_BUDGET
        if len(output.plan.replacement.text.encode("utf-8")) > (
            self._config.max_replacement_text_bytes
        ):
            return output, KnowledgeMaintenancePlanningOutcome.PLANNER_OVER_BUDGET
        if any(
            len(mapping.claim.encode("utf-8")) > self._config.max_claim_bytes
            for mapping in output.plan.evidence_mappings
        ):
            return output, KnowledgeMaintenancePlanningOutcome.PLANNER_OVER_BUDGET
        if not _usage_within_budget(output.usage, self._config.planner_budget):
            return output, KnowledgeMaintenancePlanningOutcome.PLANNER_OVER_BUDGET
        return output, None

    async def _invoke_evaluator(
        self,
        evaluator_input: KnowledgeMaintenanceEvaluatorInput,
    ) -> tuple[
        KnowledgeMaintenanceEvaluatorOutput | None,
        KnowledgeMaintenancePlanningOutcome | None,
    ]:
        try:
            raw_output, timed_out = await self._invoke_component(
                lambda: self._evaluate(evaluator_input.model_copy(deep=True)),
                timeout_seconds=self._config.evaluator_timeout_seconds,
                task_name="cayu-knowledge-maintenance-evaluator",
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            return None, KnowledgeMaintenancePlanningOutcome.EVALUATOR_FAILED
        if timed_out:
            return None, KnowledgeMaintenancePlanningOutcome.EVALUATOR_TIMED_OUT
        if type(raw_output) is not KnowledgeMaintenanceEvaluatorOutput:
            return None, KnowledgeMaintenancePlanningOutcome.EVALUATOR_INVALID
        try:
            output = KnowledgeMaintenanceEvaluatorOutput.model_validate(
                raw_output.model_dump(mode="python")
            )
        except (TypeError, ValueError):
            return None, KnowledgeMaintenancePlanningOutcome.EVALUATOR_INVALID
        if not _usage_model_identity_authorized(output.usage, self._config.evaluator_budget):
            return None, KnowledgeMaintenancePlanningOutcome.EVALUATOR_INVALID
        if _canonical_bytes(output, "knowledge maintenance evaluator output") > (
            self._config.max_evaluator_output_bytes
        ):
            return output, KnowledgeMaintenancePlanningOutcome.EVALUATOR_OVER_BUDGET
        if not _usage_within_budget(output.usage, self._config.evaluator_budget):
            return output, KnowledgeMaintenancePlanningOutcome.EVALUATOR_OVER_BUDGET
        return output, None

    def _deterministic_findings(
        self,
        planner_input: KnowledgeMaintenancePlannerInput,
        plan: KnowledgeMaintenancePlanDraft,
    ) -> tuple[KnowledgeMaintenanceEvaluationFinding, ...]:
        findings: list[KnowledgeMaintenanceEvaluationFinding] = []

        if (
            plan.routing_request_fingerprint != planner_input.snapshot.routing_request_fingerprint
            or plan.routing_result_fingerprint != planner_input.snapshot.routing_result_fingerprint
            or plan.configuration_fingerprint != self._config.fingerprint
            or plan.policy_id != planner_input.snapshot.policy_id
        ):
            findings.append(
                KnowledgeMaintenanceEvaluationFinding(
                    kind=KnowledgeMaintenanceEvaluationFindingKind.ROUTING_BINDING_INVALID,
                    code=KnowledgeMaintenanceEvaluationFindingCode.ROUTING_BINDING_INVALID,
                )
            )

        routed_sources = {
            (candidate.reference.entry_id, candidate.reference.revision)
            for candidate in planner_input.snapshot.candidates
        }
        planned_sources = {
            (reference.entry_id, reference.revision) for reference in plan.source_references
        }
        outside = planned_sources - routed_sources
        missing = routed_sources - planned_sources
        if outside:
            findings.append(
                KnowledgeMaintenanceEvaluationFinding(
                    kind=KnowledgeMaintenanceEvaluationFindingKind.SOURCE_OUTSIDE_ROUTE,
                    code=KnowledgeMaintenanceEvaluationFindingCode.SOURCE_OUTSIDE_ROUTE,
                    source_references=_refs_from_keys(outside),
                )
            )
        if missing:
            findings.append(
                KnowledgeMaintenanceEvaluationFinding(
                    kind=KnowledgeMaintenanceEvaluationFindingKind.SOURCE_COVERAGE_INCOMPLETE,
                    code=KnowledgeMaintenanceEvaluationFindingCode.SOURCE_COVERAGE_INCOMPLETE,
                    source_references=_refs_from_keys(missing),
                )
            )

        if plan.replacement.kind not in self._config.allowed_replacement_kinds:
            findings.append(
                KnowledgeMaintenanceEvaluationFinding(
                    kind=KnowledgeMaintenanceEvaluationFindingKind.REPLACEMENT_KIND_DISALLOWED,
                    code=(KnowledgeMaintenanceEvaluationFindingCode.REPLACEMENT_KIND_DISALLOWED),
                )
            )

        mapping_by_id = {mapping.id: mapping for mapping in plan.evidence_mappings}
        evidence_sources: set[tuple[str, int]] = set()
        invalid_evidence_sources: set[tuple[str, int]] = set()
        for mapping in plan.evidence_mappings:
            for reference in mapping.source_references:
                key = (reference.entry_id, reference.revision)
                if key in routed_sources:
                    evidence_sources.add(key)
                else:
                    invalid_evidence_sources.add(key)
        if invalid_evidence_sources:
            findings.append(
                KnowledgeMaintenanceEvaluationFinding(
                    kind=KnowledgeMaintenanceEvaluationFindingKind.EVIDENCE_SOURCE_INVALID,
                    code=KnowledgeMaintenanceEvaluationFindingCode.EVIDENCE_SOURCE_INVALID,
                    source_references=_refs_from_keys(invalid_evidence_sources),
                )
            )
        uncovered_evidence = routed_sources - evidence_sources
        if uncovered_evidence:
            findings.append(
                KnowledgeMaintenanceEvaluationFinding(
                    kind=(KnowledgeMaintenanceEvaluationFindingKind.EVIDENCE_COVERAGE_INCOMPLETE),
                    code=KnowledgeMaintenanceEvaluationFindingCode.EVIDENCE_COVERAGE_INCOMPLETE,
                    source_references=_refs_from_keys(uncovered_evidence),
                )
            )

        relation_sources: list[tuple[str, int]] = []
        relation_coverage_invalid: set[tuple[str, int]] = set()
        orientation_invalid: set[tuple[str, int]] = set()
        exhausted_sources: set[tuple[str, int]] = set()
        invalid_mapping_ids: set[str] = set()
        referenced_mapping_ids: set[str] = set()
        for relation in plan.relations:
            source = relation.source_reference
            source_key = (source.entry_id, source.revision)
            relation_sources.append(source_key)
            if source_key not in routed_sources:
                relation_coverage_invalid.add(source_key)
            if (
                relation.kind
                in {KnowledgeRelationKind.SUPERSEDES, KnowledgeRelationKind.DERIVED_FROM}
                and relation.subject.kind is not KnowledgeMaintenancePlanEndpointKind.REPLACEMENT
            ):
                orientation_invalid.add(source_key)
            if (
                relation.kind is KnowledgeRelationKind.SUPERSEDES
                and source.revision >= MAX_KNOWLEDGE_REVISION
            ):
                exhausted_sources.add(source_key)
            for mapping_id in relation.evidence_mapping_ids:
                referenced_mapping_ids.add(mapping_id)
                mapping = mapping_by_id.get(mapping_id)
                if mapping is None:
                    invalid_mapping_ids.add(mapping_id)
                    continue
                mapping_sources = {
                    (reference.entry_id, reference.revision)
                    for reference in mapping.source_references
                }
                if source_key not in mapping_sources:
                    invalid_mapping_ids.add(mapping_id)
        relation_source_set = set(relation_sources)
        if (
            relation_coverage_invalid
            or relation_source_set != routed_sources
            or len(relation_sources) != len(relation_source_set)
        ):
            implicated = relation_coverage_invalid | (routed_sources - relation_source_set)
            findings.append(
                KnowledgeMaintenanceEvaluationFinding(
                    kind=KnowledgeMaintenanceEvaluationFindingKind.RELATION_COVERAGE_INVALID,
                    code=KnowledgeMaintenanceEvaluationFindingCode.RELATION_COVERAGE_INVALID,
                    source_references=_refs_from_keys(implicated),
                )
            )
        if orientation_invalid:
            findings.append(
                KnowledgeMaintenanceEvaluationFinding(
                    kind=KnowledgeMaintenanceEvaluationFindingKind.RELATION_ORIENTATION_INVALID,
                    code=KnowledgeMaintenanceEvaluationFindingCode.RELATION_ORIENTATION_INVALID,
                    source_references=_refs_from_keys(orientation_invalid),
                )
            )
        if exhausted_sources:
            findings.append(
                KnowledgeMaintenanceEvaluationFinding(
                    kind=KnowledgeMaintenanceEvaluationFindingKind.SOURCE_REVISION_EXHAUSTED,
                    code=KnowledgeMaintenanceEvaluationFindingCode.SOURCE_REVISION_EXHAUSTED,
                    source_references=_refs_from_keys(exhausted_sources),
                )
            )
        if invalid_mapping_ids:
            findings.append(
                KnowledgeMaintenanceEvaluationFinding(
                    kind=KnowledgeMaintenanceEvaluationFindingKind.EVIDENCE_COVERAGE_INCOMPLETE,
                    code=(
                        KnowledgeMaintenanceEvaluationFindingCode.RELATION_EVIDENCE_MAPPING_INVALID
                    ),
                    evidence_mapping_ids=tuple(sorted(invalid_mapping_ids)),
                )
            )
        unreferenced_mapping_ids = set(mapping_by_id) - referenced_mapping_ids
        if unreferenced_mapping_ids:
            findings.append(
                KnowledgeMaintenanceEvaluationFinding(
                    kind=KnowledgeMaintenanceEvaluationFindingKind.EVIDENCE_COVERAGE_INCOMPLETE,
                    code=(KnowledgeMaintenanceEvaluationFindingCode.UNREFERENCED_EVIDENCE_MAPPING),
                    evidence_mapping_ids=tuple(sorted(unreferenced_mapping_ids)),
                )
            )
        return _dedupe_findings(findings)

    def _valid_evaluator_decision(
        self,
        planner_input: KnowledgeMaintenancePlannerInput,
        plan: KnowledgeMaintenancePlanDraft,
        decision: KnowledgeMaintenanceEvaluatorDecision,
    ) -> bool:
        if (
            decision.plan_fingerprint != plan.fingerprint
            or decision.routing_result_fingerprint
            != planner_input.snapshot.routing_result_fingerprint
            or decision.configuration_fingerprint != self._config.fingerprint
        ):
            return False
        source_keys = {
            (reference.entry_id, reference.revision) for reference in plan.source_references
        }
        mapping_ids = {mapping.id for mapping in plan.evidence_mappings}
        for finding in decision.findings:
            if finding.kind not in _EVALUATOR_FINDING_KINDS:
                return False
            if any(
                (reference.entry_id, reference.revision) not in source_keys
                for reference in finding.source_references
            ):
                return False
            if not set(finding.evidence_mapping_ids).issubset(mapping_ids):
                return False
        return True

    def _final_evaluation(
        self,
        planner_input: KnowledgeMaintenancePlannerInput,
        plan: KnowledgeMaintenancePlanDraft,
        *,
        verdict: KnowledgeMaintenanceEvaluationVerdict,
        code: str,
        findings: Iterable[KnowledgeMaintenanceEvaluationFinding],
        evaluator_invoked: bool,
    ) -> KnowledgeMaintenancePlanEvaluation:
        return KnowledgeMaintenancePlanEvaluation(
            plan_fingerprint=plan.fingerprint,
            routing_result_fingerprint=planner_input.snapshot.routing_result_fingerprint,
            configuration_fingerprint=self._config.fingerprint,
            evaluator_id=self._config.evaluator_id,
            evaluator_version=self._config.evaluator_version,
            evaluator_invoked=evaluator_invoked,
            verdict=verdict,
            code=code,
            findings=tuple(findings),
        )

    def _result(
        self,
        planner_input: KnowledgeMaintenancePlannerInput,
        *,
        outcome: KnowledgeMaintenancePlanningOutcome,
        code: str,
        plan: KnowledgeMaintenancePlanDraft | None = None,
        evaluation: KnowledgeMaintenancePlanEvaluation | None = None,
        planner_usage: KnowledgeMaintenanceInferenceUsage | None = None,
        evaluator_usage: KnowledgeMaintenanceInferenceUsage | None = None,
    ) -> KnowledgeMaintenancePlanningResult:
        return KnowledgeMaintenancePlanningResult(
            request_id=planner_input.snapshot.request_id,
            request_fingerprint=planner_input.snapshot.routing_request_fingerprint,
            routing_result_fingerprint=planner_input.snapshot.routing_result_fingerprint,
            configuration_fingerprint=self._config.fingerprint,
            planner_id=self._config.planner_id,
            planner_version=self._config.planner_version,
            evaluator_id=self._config.evaluator_id,
            evaluator_version=self._config.evaluator_version,
            outcome=outcome,
            code=code,
            plan=plan,
            evaluation=evaluation,
            planner_usage=planner_usage,
            evaluator_usage=evaluator_usage,
            processed_at=self._clock(),
        )


def _canonical_bytes(value: BaseModel, field_name: str) -> int:
    return len(canonical_durable_json_bytes(value.model_dump(mode="json"), field_name))


def _usage_within_budget(
    usage: KnowledgeMaintenanceInferenceUsage,
    budget: KnowledgeMaintenanceStageBudget,
) -> bool:
    return (
        usage.model_calls <= budget.max_model_calls
        and usage.cost_micro_usd <= budget.max_cost_micro_usd
    )


def _usage_model_identity_authorized(
    usage: KnowledgeMaintenanceInferenceUsage,
    budget: KnowledgeMaintenanceStageBudget,
) -> bool:
    return usage.model_calls == 0 or usage.model_id in budget.allowed_model_ids


def _refs_from_keys(keys: Iterable[tuple[str, int]]) -> tuple[KnowledgeRevisionRef, ...]:
    return tuple(
        KnowledgeRevisionRef(entry_id=entry_id, revision=revision)
        for entry_id, revision in sorted(keys)
    )


def _dedupe_findings(
    findings: Iterable[KnowledgeMaintenanceEvaluationFinding],
) -> tuple[KnowledgeMaintenanceEvaluationFinding, ...]:
    by_fingerprint = {finding.fingerprint: finding for finding in findings}
    return tuple(by_fingerprint[key] for key in sorted(by_fingerprint))


__all__ = [
    "KNOWLEDGE_MAINTENANCE_DETERMINISTIC_EVALUATOR_VERSION",
    "KNOWLEDGE_MAINTENANCE_PLANNING_SCHEMA_VERSION",
    "MAX_KNOWLEDGE_MAINTENANCE_COST_MICRO_USD",
    "MAX_KNOWLEDGE_MAINTENANCE_EVALUATION_FINDINGS",
    "MAX_KNOWLEDGE_MAINTENANCE_MODEL_CALLS",
    "MAX_KNOWLEDGE_MAINTENANCE_PLANNING_BYTES",
    "MAX_KNOWLEDGE_MAINTENANCE_PLANNING_TIMEOUT_SECONDS",
    "MAX_KNOWLEDGE_MAINTENANCE_PLAN_CLAIMS",
    "MAX_KNOWLEDGE_MAINTENANCE_TOKEN_COUNT",
    "KnowledgeMaintenanceEvaluationFinding",
    "KnowledgeMaintenanceEvaluationFindingCode",
    "KnowledgeMaintenanceEvaluationFindingKind",
    "KnowledgeMaintenanceEvaluationVerdict",
    "KnowledgeMaintenanceEvaluatorDecision",
    "KnowledgeMaintenanceEvaluatorInput",
    "KnowledgeMaintenanceEvaluatorOutput",
    "KnowledgeMaintenanceEvidenceMapping",
    "KnowledgeMaintenanceInferenceUsage",
    "KnowledgeMaintenancePlanDraft",
    "KnowledgeMaintenancePlanEndpoint",
    "KnowledgeMaintenancePlanEndpointKind",
    "KnowledgeMaintenancePlanEvaluation",
    "KnowledgeMaintenancePlanEvaluator",
    "KnowledgeMaintenancePlanner",
    "KnowledgeMaintenancePlannerBudget",
    "KnowledgeMaintenancePlannerInput",
    "KnowledgeMaintenancePlannerOutput",
    "KnowledgeMaintenancePlanningConfig",
    "KnowledgeMaintenancePlanningLimitExceeded",
    "KnowledgeMaintenancePlanningOutcome",
    "KnowledgeMaintenancePlanningResult",
    "KnowledgeMaintenancePlanningSnapshot",
    "KnowledgeMaintenancePlanningWorkflow",
    "KnowledgeMaintenanceRelationDraft",
    "KnowledgeMaintenanceReplacementDraft",
    "KnowledgeMaintenanceStageBudget",
]
