"""Provider-free end-to-end evaluation for reviewed knowledge maintenance."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from math import ceil
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_json_object,
    require_durable_clean_nonblank,
    require_durable_nonblank,
    require_finite,
)
from cayu.knowledge_maintenance import (
    KnowledgeMaintenanceCandidateSignal,
    KnowledgeMaintenanceRouter,
    KnowledgeMaintenanceRouterConfig,
    KnowledgeMaintenanceRoutingRequest,
    KnowledgeMaintenanceSignalKind,
)
from cayu.knowledge_maintenance_persistence import (
    KnowledgeMaintenanceProposalPublication,
    KnowledgeMaintenanceProposalPublicationOutcome,
    KnowledgeMaintenanceProposalPublisher,
    KnowledgeMaintenanceProposalPublisherConfig,
    copy_knowledge_maintenance_proposal_publication,
    copy_knowledge_maintenance_proposal_publication_receipt,
)
from cayu.knowledge_maintenance_planning import (
    KnowledgeMaintenanceEvaluationFinding,
    KnowledgeMaintenanceEvaluationFindingCode,
    KnowledgeMaintenanceEvaluationFindingKind,
    KnowledgeMaintenanceEvaluationVerdict,
    KnowledgeMaintenanceEvaluatorDecision,
    KnowledgeMaintenanceEvaluatorOutput,
    KnowledgeMaintenanceEvidenceMapping,
    KnowledgeMaintenanceInferenceUsage,
    KnowledgeMaintenancePlanDraft,
    KnowledgeMaintenancePlanEndpoint,
    KnowledgeMaintenancePlanEndpointKind,
    KnowledgeMaintenancePlannerOutput,
    KnowledgeMaintenancePlanningConfig,
    KnowledgeMaintenancePlanningOutcome,
    KnowledgeMaintenancePlanningWorkflow,
    KnowledgeMaintenanceRelationDraft,
    KnowledgeMaintenanceReplacementDraft,
)
from cayu.recall import (
    RECALL_MAX_QUERY_BYTES,
    KnowledgeRecallSource,
    RecallRecord,
    RecallSituation,
)
from cayu.storage import (
    DEFAULT_KNOWLEDGE_INDEX_MAX_CHUNKS,
    MAX_KNOWLEDGE_ENTRY_ID_BYTES,
    MAX_KNOWLEDGE_MAINTENANCE_SOURCES,
    KnowledgeAccessScope,
    KnowledgeActorType,
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeEvidence,
    KnowledgeEvidenceDisposition,
    KnowledgeEvidenceResult,
    KnowledgeEvidenceRole,
    KnowledgeLineageCurrentness,
    KnowledgeLineageLink,
    KnowledgeLineageQuery,
    KnowledgeLineageResult,
    KnowledgeLineageRole,
    KnowledgeListQuery,
    KnowledgeListResult,
    KnowledgeMaintenanceDecision,
    KnowledgeMaintenanceDecisionKind,
    KnowledgeMaintenanceDecisionReceipt,
    KnowledgeMaintenanceOutcome,
    KnowledgeMaintenanceProposal,
    KnowledgeMaintenanceStale,
    KnowledgeRelation,
    KnowledgeRelationKind,
    KnowledgeReviewWorkflow,
    KnowledgeRevisionRef,
    KnowledgeStatus,
    KnowledgeStore,
    KnowledgeVisibility,
    prepare_knowledge_maintenance_decision,
)

KNOWLEDGE_MAINTENANCE_EVALUATION_CORPUS_SCHEMA_VERSION = (
    "cayu.knowledge_maintenance_evaluation_corpus.v1"
)
KNOWLEDGE_MAINTENANCE_EVALUATION_RESULT_SCHEMA_VERSION = (
    "cayu.knowledge_maintenance_evaluation_result.v1"
)
_MAX_CORPUS_BYTES = 4 * 1024 * 1024
_MAX_RESULT_BYTES = 16 * 1024 * 1024
_MAX_CONFIGURATION_BYTES = 64 * 1024
_MAX_CASES = 10_000
_MAX_ENTRIES_PER_CASE = 100
_MAX_CLAIMS_PER_CASE = 256
_MAX_CASE_ID_BYTES = 200
_MAX_CLAIM_ID_BYTES = 128
_MAX_LINEAGE_BYTES = 64_000
_MAX_RECALL_RECORD_BYTES = 8_000
_REFERENCE_TIME = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
_SAFE_CODE_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._:-")
_EVALUATION_PLANNING_CONFIG = KnowledgeMaintenancePlanningConfig(
    planner_id="knowledge-maintenance-evaluation-planner",
    planner_version="1",
    evaluator_id="knowledge-maintenance-evaluation-evaluator",
    evaluator_version="1",
    max_planner_model_calls=0,
    max_evaluator_model_calls=0,
    max_planner_cost_micro_usd=0,
    max_evaluator_cost_micro_usd=0,
    max_total_cost_micro_usd=0,
)


@dataclass(frozen=True)
class _CaseEvidenceExpectation:
    publication_replacement: KnowledgeRevisionRef
    replacement: KnowledgeRevisionRef
    expected_sources: tuple[KnowledgeEntry, ...]
    accepted_plan_fingerprint: str
    created_at: datetime
    access_scope: KnowledgeAccessScope


@dataclass(frozen=True)
class _CaseMaintenanceExpectation:
    publication: KnowledgeMaintenanceProposalPublication
    decision_operation_id: str
    decision: KnowledgeMaintenanceDecision | None
    decision_receipt: KnowledgeMaintenanceDecisionReceipt | None
    access_scope: KnowledgeAccessScope


@dataclass(frozen=True)
class _CaseRevisionExpectation:
    entry: KnowledgeEntry
    chunks: tuple[KnowledgeChunk, ...]


@dataclass(frozen=True)
class _CaseRecallExpectation:
    case: KnowledgeMaintenanceEvaluationCase
    storage_outcome: Literal["not_published", "applied", "rejected", "stale"]
    active_scope: KnowledgeAccessScope
    namespace: str
    expected_entries: tuple[KnowledgeEntry, ...]
    expected_relations: tuple[KnowledgeRelation, ...]
    pending_replacement: KnowledgeEntry | None = None
    proposal: KnowledgeMaintenanceProposal | None = None
    replacement: KnowledgeRevisionRef | None = None


@dataclass(frozen=True)
class _CaseTerminalExpectation:
    entries: tuple[KnowledgeEntry, ...]
    revisions: tuple[_CaseRevisionExpectation, ...]
    relations: tuple[KnowledgeRelation, ...]
    evidence: _CaseEvidenceExpectation | None
    maintenance: _CaseMaintenanceExpectation | None
    recall: _CaseRecallExpectation | None


class KnowledgeMaintenanceEvaluationScenario(StrEnum):
    """Reference outcomes required by the reviewed-maintenance architecture."""

    DUPLICATE_MERGE = "duplicate_merge"
    AUTHORITATIVE_SUPERSESSION = "authoritative_supersession"
    UNRESOLVED_CONTRADICTION = "unresolved_contradiction"
    STALE_PROPOSAL = "stale_proposal"
    REVIEWER_REJECTION = "reviewer_rejection"
    HISTORICAL_LINEAGE = "historical_lineage"


class KnowledgeMaintenanceEvaluationEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    id: str
    text: str

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        value = require_durable_clean_nonblank(value, "id")
        if len(value.encode("utf-8")) > MAX_KNOWLEDGE_ENTRY_ID_BYTES:
            raise ValueError(f"id cannot exceed {MAX_KNOWLEDGE_ENTRY_ID_BYTES} UTF-8 bytes.")
        return value

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return require_durable_nonblank(value, "text")


class KnowledgeMaintenanceEvaluationClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    id: str
    text: str
    source_entry_ids: tuple[str, ...]

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        value = require_durable_clean_nonblank(value, "id")
        if (
            len(value.encode("utf-8")) > _MAX_CLAIM_ID_BYTES
            or not value[0].isalnum()
            or any(character not in _SAFE_CODE_CHARS for character in value)
        ):
            raise ValueError(
                "id must be a safe machine-readable code of at most "
                f"{_MAX_CLAIM_ID_BYTES} UTF-8 bytes."
            )
        return value

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return require_durable_nonblank(value, "text")

    @model_validator(mode="after")
    def validate_sources(self) -> KnowledgeMaintenanceEvaluationClaim:
        if not self.source_entry_ids:
            raise ValueError("source_entry_ids must not be empty.")
        if len(self.source_entry_ids) != len(set(self.source_entry_ids)):
            raise ValueError("source_entry_ids must be unique.")
        return self


class KnowledgeMaintenanceEvaluationDisposition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    source_entry_id: str
    relation_kind: KnowledgeRelationKind

    @field_validator("source_entry_id")
    @classmethod
    def validate_source_entry_id(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "source_entry_id")


class KnowledgeMaintenanceEvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    id: str
    scenario: KnowledgeMaintenanceEvaluationScenario
    entries: tuple[KnowledgeMaintenanceEvaluationEntry, ...]
    source_entry_ids: tuple[str, ...]
    signal_kind: KnowledgeMaintenanceSignalKind
    expected_routed_entry_ids: tuple[str, ...]
    replacement_title: str
    replacement_text: str
    claims: tuple[KnowledgeMaintenanceEvaluationClaim, ...]
    dispositions: tuple[KnowledgeMaintenanceEvaluationDisposition, ...]
    evaluator_verdict: KnowledgeMaintenanceEvaluationVerdict
    review_decision: KnowledgeMaintenanceDecisionKind | None = None
    advance_source_before_review: str | None = None
    recall_query: str

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        value = require_durable_clean_nonblank(value, "id")
        if len(value.encode("utf-8")) > _MAX_CASE_ID_BYTES:
            raise ValueError(f"id cannot exceed {_MAX_CASE_ID_BYTES} UTF-8 bytes.")
        return value

    @field_validator("replacement_title", "replacement_text")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return require_durable_nonblank(value, info.field_name)

    @field_validator("recall_query")
    @classmethod
    def validate_recall_query(cls, value: str) -> str:
        value = require_durable_nonblank(value, "recall_query")
        if len(value.encode("utf-8")) > RECALL_MAX_QUERY_BYTES:
            raise ValueError(f"recall_query cannot exceed {RECALL_MAX_QUERY_BYTES} UTF-8 bytes.")
        return value

    @model_validator(mode="after")
    def validate_references(self) -> KnowledgeMaintenanceEvaluationCase:
        entry_ids = [entry.id for entry in self.entries]
        if (
            not entry_ids
            or len(entry_ids) > _MAX_ENTRIES_PER_CASE
            or len(entry_ids) != len(set(entry_ids))
        ):
            raise ValueError("entries must be non-empty with unique ids.")
        known = set(entry_ids)
        sources = set(self.source_entry_ids)
        if not sources or len(sources) != len(self.source_entry_ids) or not sources <= known:
            raise ValueError("source_entry_ids must uniquely reference corpus entries.")
        if len(self.source_entry_ids) > MAX_KNOWLEDGE_MAINTENANCE_SOURCES:
            raise ValueError(
                "source_entry_ids cannot exceed the executable maintenance source bound "
                f"of {MAX_KNOWLEDGE_MAINTENANCE_SOURCES}."
            )
        if not set(self.expected_routed_entry_ids) <= known:
            raise ValueError("expected_routed_entry_ids reference unknown entries.")
        if len(self.expected_routed_entry_ids) != len(set(self.expected_routed_entry_ids)):
            raise ValueError("expected_routed_entry_ids must be unique.")
        disposition_sources = [item.source_entry_id for item in self.dispositions]
        if set(disposition_sources) != sources or len(disposition_sources) != len(
            set(disposition_sources)
        ):
            raise ValueError("dispositions must cover every source exactly.")
        if any(not set(claim.source_entry_ids) <= sources for claim in self.claims):
            raise ValueError("claims may reference only maintenance sources.")
        if (
            not self.claims
            or len(self.claims) > _MAX_CLAIMS_PER_CASE
            or len({claim.id for claim in self.claims}) != len(self.claims)
        ):
            raise ValueError("claims must be non-empty with unique ids.")
        claimed_sources = {entry_id for claim in self.claims for entry_id in claim.source_entry_ids}
        if claimed_sources != sources:
            raise ValueError("claims must cover every maintenance source.")
        if (
            self.signal_kind
            in {
                KnowledgeMaintenanceSignalKind.DUPLICATE_HINT,
                KnowledgeMaintenanceSignalKind.CONTRADICTION,
            }
            and len(self.source_entry_ids) != 2
        ):
            raise ValueError("Paired signals require exactly two source entries.")
        if self.evaluator_verdict is KnowledgeMaintenanceEvaluationVerdict.REJECTED:
            if self.review_decision is not None or self.advance_source_before_review is not None:
                raise ValueError("Rejected plans cannot reach external review.")
        elif self.review_decision is None:
            raise ValueError("Accepted plans require an external review decision.")
        if (
            self.advance_source_before_review is not None
            and self.advance_source_before_review not in sources
        ):
            raise ValueError("advance_source_before_review must identify a source.")
        self._validate_scenario_contract()
        return self

    def _validate_scenario_contract(self) -> None:
        expected_signal = {
            KnowledgeMaintenanceEvaluationScenario.DUPLICATE_MERGE: (
                KnowledgeMaintenanceSignalKind.DUPLICATE_HINT
            ),
            KnowledgeMaintenanceEvaluationScenario.AUTHORITATIVE_SUPERSESSION: (
                KnowledgeMaintenanceSignalKind.EXACT_REFERENCE
            ),
            KnowledgeMaintenanceEvaluationScenario.UNRESOLVED_CONTRADICTION: (
                KnowledgeMaintenanceSignalKind.CONTRADICTION
            ),
            KnowledgeMaintenanceEvaluationScenario.STALE_PROPOSAL: (
                KnowledgeMaintenanceSignalKind.DUPLICATE_HINT
            ),
            KnowledgeMaintenanceEvaluationScenario.REVIEWER_REJECTION: (
                KnowledgeMaintenanceSignalKind.DUPLICATE_HINT
            ),
            KnowledgeMaintenanceEvaluationScenario.HISTORICAL_LINEAGE: (
                KnowledgeMaintenanceSignalKind.EXACT_REFERENCE
            ),
        }[self.scenario]
        expected_evaluator = (
            KnowledgeMaintenanceEvaluationVerdict.REJECTED
            if self.scenario is KnowledgeMaintenanceEvaluationScenario.UNRESOLVED_CONTRADICTION
            else KnowledgeMaintenanceEvaluationVerdict.ACCEPTED
        )
        expected_review = {
            KnowledgeMaintenanceEvaluationScenario.UNRESOLVED_CONTRADICTION: None,
            KnowledgeMaintenanceEvaluationScenario.REVIEWER_REJECTION: (
                KnowledgeMaintenanceDecisionKind.REJECT
            ),
        }.get(self.scenario, KnowledgeMaintenanceDecisionKind.APPROVE)
        expected_relation = (
            KnowledgeRelationKind.CONTRADICTS
            if self.scenario is KnowledgeMaintenanceEvaluationScenario.UNRESOLVED_CONTRADICTION
            else KnowledgeRelationKind.SUPERSEDES
        )
        if self.signal_kind is not expected_signal:
            raise ValueError("signal_kind conflicts with the selected scenario.")
        if self.evaluator_verdict is not expected_evaluator:
            raise ValueError("evaluator_verdict conflicts with the selected scenario.")
        if self.review_decision is not expected_review:
            raise ValueError("review_decision conflicts with the selected scenario.")
        if any(item.relation_kind is not expected_relation for item in self.dispositions):
            raise ValueError("A disposition relation conflicts with the selected scenario.")
        expects_advance = self.scenario is KnowledgeMaintenanceEvaluationScenario.STALE_PROPOSAL
        if (self.advance_source_before_review is not None) is not expects_advance:
            raise ValueError("advance_source_before_review conflicts with the selected scenario.")
        _validate_fixture_plan_is_executable(self)

    @property
    def relation_kinds(self) -> dict[str, KnowledgeRelationKind]:
        return {item.source_entry_id: item.relation_kind for item in self.dispositions}


class KnowledgeMaintenanceEvaluationCorpus(BaseModel):
    """Portable public or private maintenance-quality corpus."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal["cayu.knowledge_maintenance_evaluation_corpus.v1"]
    corpus_revision: str
    origin: Literal["hermetic_public", "external_private"]
    cases: tuple[KnowledgeMaintenanceEvaluationCase, ...]

    @field_validator("corpus_revision")
    @classmethod
    def validate_revision(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "corpus_revision")

    @model_validator(mode="after")
    def validate_cases(self) -> KnowledgeMaintenanceEvaluationCorpus:
        if not self.cases or len(self.cases) > _MAX_CASES:
            raise ValueError(f"cases must contain 1..{_MAX_CASES} items.")
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("Corpus case ids must be unique.")
        entry_ids = [entry.id for case in self.cases for entry in case.entries]
        if len(entry_ids) != len(set(entry_ids)):
            raise ValueError("Corpus entry ids must be globally unique.")
        if (
            len(
                canonical_durable_json_bytes(
                    self.model_dump(mode="json"),
                    "knowledge maintenance evaluation corpus",
                )
            )
            > _MAX_CORPUS_BYTES
        ):
            raise ValueError(f"Knowledge maintenance corpus exceeds {_MAX_CORPUS_BYTES} bytes.")
        return self


class KnowledgeMaintenanceEvaluationCaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    case_id: str
    scenario: KnowledgeMaintenanceEvaluationScenario
    routed_entry_ids: tuple[str, ...]
    routing_precision: float
    routing_recall: float
    planning_outcome: KnowledgeMaintenancePlanningOutcome
    storage_outcome: Literal["not_published", "pending", "applied", "rejected", "stale"]
    information_retention: float
    evidence_retention: float | None
    unsafe_acceptance_count: int
    lifecycle_correct: bool
    lineage_correct: bool
    recalled_entry_ids: tuple[str, ...]
    model_call_count: int
    latency_ms: float

    @field_validator(
        "routing_precision",
        "routing_recall",
        "information_retention",
        "evidence_retention",
    )
    @classmethod
    def validate_rate(cls, value: float | None, info) -> float | None:
        if value is None:
            return None
        value = require_finite(value, info.field_name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{info.field_name} must be between 0 and 1.")
        return value

    @field_validator("unsafe_acceptance_count", mode="before")
    @classmethod
    def validate_unsafe_acceptance_count(cls, value) -> int:
        if type(value) is not int or value not in {0, 1}:
            raise ValueError("unsafe_acceptance_count must be 0 or 1.")
        return value

    @field_validator("model_call_count", mode="before")
    @classmethod
    def validate_model_call_count(cls, value) -> int:
        if type(value) is not int or value < 0:
            raise ValueError("model_call_count must be a non-negative integer.")
        return value

    @field_validator("latency_ms")
    @classmethod
    def validate_latency(cls, value: float) -> float:
        value = require_finite(value, "latency_ms")
        if value < 0.0:
            raise ValueError("latency_ms must be non-negative.")
        return value

    @model_validator(mode="after")
    def validate_selected_ids(self) -> KnowledgeMaintenanceEvaluationCaseResult:
        if len(self.routed_entry_ids) != len(set(self.routed_entry_ids)):
            raise ValueError("routed_entry_ids must be unique.")
        if len(self.recalled_entry_ids) != len(set(self.recalled_entry_ids)):
            raise ValueError("recalled_entry_ids must be unique.")
        return self


class KnowledgeMaintenanceEvaluationMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    case_count: int
    routing_precision: float
    routing_recall: float
    information_retention: float
    evidence_retention: float
    unsafe_acceptance_rate: float
    lifecycle_correctness: float
    lineage_correctness: float
    model_call_count: int
    latency_p50_ms: float
    latency_p95_ms: float

    @field_validator("case_count", mode="before")
    @classmethod
    def validate_case_count(cls, value) -> int:
        if type(value) is not int or value < 1:
            raise ValueError("case_count must be a positive integer.")
        return value

    @field_validator(
        "routing_precision",
        "routing_recall",
        "information_retention",
        "evidence_retention",
        "unsafe_acceptance_rate",
        "lifecycle_correctness",
        "lineage_correctness",
    )
    @classmethod
    def validate_rate(cls, value: float, info) -> float:
        value = require_finite(value, info.field_name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{info.field_name} must be between 0 and 1.")
        return value

    @field_validator("model_call_count", mode="before")
    @classmethod
    def validate_model_call_count(cls, value) -> int:
        if type(value) is not int or value < 0:
            raise ValueError("model_call_count must be a non-negative integer.")
        return value

    @field_validator("latency_p50_ms", "latency_p95_ms")
    @classmethod
    def validate_latency(cls, value: float, info) -> float:
        value = require_finite(value, info.field_name)
        if value < 0.0:
            raise ValueError(f"{info.field_name} must be non-negative.")
        return value

    @model_validator(mode="after")
    def validate_percentiles(self) -> KnowledgeMaintenanceEvaluationMetrics:
        if self.latency_p95_ms < self.latency_p50_ms:
            raise ValueError("latency_p95_ms cannot be less than latency_p50_ms.")
        return self


def _aggregate_metrics(
    cases: tuple[KnowledgeMaintenanceEvaluationCaseResult, ...],
) -> KnowledgeMaintenanceEvaluationMetrics:
    if not cases:
        raise ValueError("Knowledge maintenance results require at least one case.")
    unsafe_cases = sum(
        case.scenario is KnowledgeMaintenanceEvaluationScenario.UNRESOLVED_CONTRADICTION
        for case in cases
    )
    latencies = [case.latency_ms for case in cases]
    return KnowledgeMaintenanceEvaluationMetrics(
        case_count=len(cases),
        routing_precision=_mean(case.routing_precision for case in cases),
        routing_recall=_mean(case.routing_recall for case in cases),
        information_retention=_mean(case.information_retention for case in cases),
        evidence_retention=_mean(
            case.evidence_retention for case in cases if case.evidence_retention is not None
        ),
        unsafe_acceptance_rate=(
            sum(case.unsafe_acceptance_count for case in cases) / unsafe_cases
            if unsafe_cases
            else 0.0
        ),
        lifecycle_correctness=_mean(float(case.lifecycle_correct) for case in cases),
        lineage_correctness=_mean(float(case.lineage_correct) for case in cases),
        model_call_count=sum(case.model_call_count for case in cases),
        latency_p50_ms=_nearest_rank_percentile(latencies, 0.50),
        latency_p95_ms=_nearest_rank_percentile(latencies, 0.95),
    )


class KnowledgeMaintenanceEvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal["cayu.knowledge_maintenance_evaluation_result.v1"]
    corpus_revision: str
    corpus_origin: Literal["hermetic_public", "external_private"]
    backend: str
    execution_mode: Literal["deterministic_fixture"] = "deterministic_fixture"
    planner_identity: str
    evaluator_identity: str
    configuration: dict[str, Any]
    metrics: KnowledgeMaintenanceEvaluationMetrics
    cases: tuple[KnowledgeMaintenanceEvaluationCaseResult, ...]

    @field_validator("backend", "planner_identity", "evaluator_identity")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("configuration", mode="before")
    @classmethod
    def copy_configuration(cls, value) -> dict[str, Any]:
        return copy_durable_json_object(value, "configuration")

    @model_validator(mode="after")
    def validate_case_aggregate(self) -> KnowledgeMaintenanceEvaluationResult:
        if len(self.cases) != self.metrics.case_count:
            raise ValueError("metrics.case_count must equal the number of case results.")
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Result case ids must be unique.")
        if self.metrics != _aggregate_metrics(self.cases):
            raise ValueError("metrics must exactly match the aggregate of the case results.")
        if (
            len(
                canonical_durable_json_bytes(
                    self.model_dump(mode="json"),
                    "knowledge maintenance evaluation result",
                )
            )
            > _MAX_RESULT_BYTES
        ):
            raise ValueError(
                f"Knowledge maintenance evaluation result exceeds {_MAX_RESULT_BYTES} bytes."
            )
        return self


def load_knowledge_maintenance_evaluation_corpus(
    path: str | Path,
) -> KnowledgeMaintenanceEvaluationCorpus:
    corpus_path = Path(path)
    try:
        with corpus_path.open("rb") as stream:
            raw = stream.read(_MAX_CORPUS_BYTES + 1)
        if len(raw) > _MAX_CORPUS_BYTES:
            raise ValueError(f"Knowledge maintenance corpus exceeds {_MAX_CORPUS_BYTES} bytes.")
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to load knowledge maintenance corpus: {exc}") from exc
    return KnowledgeMaintenanceEvaluationCorpus.model_validate(payload)


def _fixture_plan(
    case: KnowledgeMaintenanceEvaluationCase,
    references: dict[str, KnowledgeRevisionRef],
    *,
    routing_request_fingerprint: str,
    routing_result_fingerprint: str,
    configuration_fingerprint: str,
    policy_id: str,
) -> KnowledgeMaintenancePlanDraft:
    source_references = tuple(references[entry_id] for entry_id in case.source_entry_ids)
    mappings = tuple(
        KnowledgeMaintenanceEvidenceMapping(
            id=claim.id,
            claim=claim.text,
            source_references=tuple(references[entry_id] for entry_id in claim.source_entry_ids),
        )
        for claim in case.claims
    )
    mapping_ids_by_source = {
        entry_id: tuple(claim.id for claim in case.claims if entry_id in claim.source_entry_ids)
        for entry_id in case.source_entry_ids
    }
    relations = tuple(
        KnowledgeMaintenanceRelationDraft(
            id=f"relation:{index:02d}",
            subject=KnowledgeMaintenancePlanEndpoint(
                kind=KnowledgeMaintenancePlanEndpointKind.REPLACEMENT
            ),
            object=KnowledgeMaintenancePlanEndpoint(
                kind=KnowledgeMaintenancePlanEndpointKind.SOURCE,
                reference=references[entry_id],
            ),
            kind=case.relation_kinds[entry_id],
            evidence_mapping_ids=mapping_ids_by_source[entry_id],
        )
        for index, entry_id in enumerate(case.source_entry_ids)
    )
    return KnowledgeMaintenancePlanDraft(
        id=f"fixture-plan:{case.id}",
        routing_request_fingerprint=routing_request_fingerprint,
        routing_result_fingerprint=routing_result_fingerprint,
        configuration_fingerprint=configuration_fingerprint,
        policy_id=policy_id,
        source_references=source_references,
        replacement=KnowledgeMaintenanceReplacementDraft(
            text=case.replacement_text,
            title=case.replacement_title,
            kind="fact",
            aspects=("knowledge-maintenance-evaluation",),
        ),
        relations=relations,
        evidence_mappings=mappings,
        rationale="Deterministic corpus fixture over exact routed revisions.",
        evidence_summary="Every fixture claim and disposition names exact sources.",
    )


def _validate_fixture_plan_is_executable(case: KnowledgeMaintenanceEvaluationCase) -> None:
    references = {
        entry_id: KnowledgeRevisionRef(entry_id=entry_id, revision=1)
        for entry_id in case.source_entry_ids
    }
    try:
        plan = _fixture_plan(
            case,
            references,
            routing_request_fingerprint="0" * 64,
            routing_result_fingerprint="0" * 64,
            configuration_fingerprint=_EVALUATION_PLANNING_CONFIG.fingerprint,
            policy_id="reviewed-consolidation-evaluation-v1",
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Case fixture fields exceed downstream executable planning bounds."
        ) from exc
    if len(plan.evidence_mappings) > _EVALUATION_PLANNING_CONFIG.max_evidence_mappings:
        raise ValueError("Case claims exceed the executable evidence-mapping bound.")
    if len(plan.replacement.text.encode("utf-8")) > (
        _EVALUATION_PLANNING_CONFIG.max_replacement_text_bytes
    ):
        raise ValueError("replacement_text exceeds the executable planner byte bound.")
    if (
        len(
            canonical_durable_json_bytes(
                plan.model_dump(mode="json"),
                "knowledge maintenance evaluation fixture plan",
            )
        )
        > _EVALUATION_PLANNING_CONFIG.max_plan_bytes
    ):
        raise ValueError("Case fixture plan exceeds the executable planner byte bound.")


class _FixturePlanner:
    def __init__(self, case: KnowledgeMaintenanceEvaluationCase) -> None:
        self._case = case

    async def propose_maintenance(self, request) -> KnowledgeMaintenancePlannerOutput:
        references = {
            candidate.reference.entry_id: candidate.reference
            for candidate in request.snapshot.candidates
        }
        return KnowledgeMaintenancePlannerOutput(
            plan=_fixture_plan(
                self._case,
                references,
                routing_request_fingerprint=request.snapshot.routing_request_fingerprint,
                routing_result_fingerprint=request.snapshot.routing_result_fingerprint,
                configuration_fingerprint=request.configuration_fingerprint,
                policy_id=request.snapshot.policy_id,
            ),
            usage=KnowledgeMaintenanceInferenceUsage(),
        )


class _FixtureEvaluator:
    def __init__(self, case: KnowledgeMaintenanceEvaluationCase) -> None:
        self._case = case

    async def evaluate_maintenance_plan(self, request) -> KnowledgeMaintenanceEvaluatorOutput:
        findings: tuple[KnowledgeMaintenanceEvaluationFinding, ...] = ()
        if self._case.evaluator_verdict is KnowledgeMaintenanceEvaluationVerdict.REJECTED:
            findings = (
                KnowledgeMaintenanceEvaluationFinding(
                    kind=KnowledgeMaintenanceEvaluationFindingKind.CONTRADICTION_MISHANDLED,
                    code=KnowledgeMaintenanceEvaluationFindingCode.CONTRADICTION_MISHANDLED,
                    source_references=request.plan.source_references,
                    evidence_mapping_ids=tuple(
                        mapping.id for mapping in request.plan.evidence_mappings
                    ),
                ),
            )
        return KnowledgeMaintenanceEvaluatorOutput(
            decision=KnowledgeMaintenanceEvaluatorDecision(
                plan_fingerprint=request.plan.fingerprint,
                routing_result_fingerprint=(
                    request.planner_input.snapshot.routing_result_fingerprint
                ),
                configuration_fingerprint=request.planner_input.configuration_fingerprint,
                verdict=self._case.evaluator_verdict,
                findings=findings,
            ),
            usage=KnowledgeMaintenanceInferenceUsage(),
        )


async def run_knowledge_maintenance_evaluation(
    corpus: KnowledgeMaintenanceEvaluationCorpus,
    store: KnowledgeStore,
    *,
    backend: str,
    configuration: dict[str, Any] | None = None,
) -> KnowledgeMaintenanceEvaluationResult:
    """Run exact maintenance workflows against an empty knowledge store.

    The injected components are deliberately deterministic and have zero model-call
    authority. This evaluates Cayu's orchestration and safety contracts separately
    from provider-dependent planner/evaluator quality.
    """

    if type(corpus) is not KnowledgeMaintenanceEvaluationCorpus:
        raise TypeError("corpus must be a KnowledgeMaintenanceEvaluationCorpus.")
    corpus = KnowledgeMaintenanceEvaluationCorpus.model_validate(corpus.model_dump(mode="json"))
    if not isinstance(store, KnowledgeStore):
        raise TypeError("store must implement KnowledgeStore.")
    backend = require_durable_clean_nonblank(backend, "backend")
    recorded_configuration = copy_durable_json_object(
        {} if configuration is None else configuration,
        "configuration",
    )
    runner_configuration = {
        "fixture_clock": _REFERENCE_TIME.isoformat(),
        "lineage_limit": MAX_KNOWLEDGE_MAINTENANCE_SOURCES,
        "lineage_max_bytes": _MAX_LINEAGE_BYTES,
        "model_call_budget": 0,
        "provider_calls": False,
    }
    for key, value in runner_configuration.items():
        if key in recorded_configuration and recorded_configuration[key] != value:
            raise ValueError(f"configuration cannot override runner-owned field {key!r}.")
        recorded_configuration[key] = value
    if (
        len(
            canonical_durable_json_bytes(
                recorded_configuration,
                "knowledge maintenance evaluation configuration",
            )
        )
        > _MAX_CONFIGURATION_BYTES
    ):
        raise ValueError(
            "Knowledge maintenance evaluation configuration exceeds "
            f"{_MAX_CONFIGURATION_BYTES} bytes."
        )

    privileged = KnowledgeAccessScope.privileged()
    existing = await store.list_entries(
        KnowledgeListQuery(statuses=list(KnowledgeStatus), include_expired=True, limit=1),
        access_scope=privileged,
    )
    if existing.entries:
        raise ValueError("Knowledge maintenance evaluation requires an empty store.")

    results: list[KnowledgeMaintenanceEvaluationCaseResult] = []
    terminal_expectations: list[_CaseTerminalExpectation] = []
    for case in corpus.cases:
        result, expectation = await _run_case(case, store, privileged=privileged)
        results.append(result)
        terminal_expectations.append(expectation)

    global_audit_started_ns = perf_counter_ns()
    all_expected_entries = {
        entry.id: entry for expectation in terminal_expectations for entry in expectation.entries
    }
    global_lifecycle_exact = len(all_expected_entries) == sum(
        len(expectation.entries) for expectation in terminal_expectations
    ) and await _all_entries_match_snapshots(
        store,
        all_expected_entries,
        privileged=privileged,
    )
    global_audit_latency_ms = (
        (perf_counter_ns() - global_audit_started_ns) / 1_000_000 / len(results)
    )
    for index, expectation in enumerate(terminal_expectations):
        audit_started_ns = perf_counter_ns()
        expected_entries = {entry.id: entry for entry in expectation.entries}
        lifecycle_exact = await _entries_match_snapshots(
            store,
            expected_entries,
            privileged=privileged,
        )
        lineage_exact = await _relations_match_snapshots(
            store,
            expected_entries=expected_entries,
            expected_relations=expectation.relations,
            scope=privileged,
        )
        maintenance_exact = expectation.maintenance is None or await _maintenance_artifacts_match(
            store,
            expectation.maintenance,
        )
        revisions_exact = await _revision_expectations_match(
            store,
            expectation.revisions,
            privileged=privileged,
        )
        recalled_entry_ids = results[index].recalled_entry_ids
        recall_lifecycle_exact = True
        recall_lineage_exact = True
        if expectation.recall is not None:
            (
                recalled_entry_ids,
                recall_lifecycle_exact,
                recall_lineage_exact,
            ) = await _rerun_terminal_recall(
                expectation.recall,
                store,
                privileged=privileged,
            )
        evidence_retention = results[index].evidence_retention
        if expectation.evidence is not None:
            terminal_evidence_retention = await _exact_evidence_retention(
                store,
                publication_replacement=expectation.evidence.publication_replacement,
                replacement=expectation.evidence.replacement,
                expected_sources=expectation.evidence.expected_sources,
                accepted_plan_fingerprint=expectation.evidence.accepted_plan_fingerprint,
                created_at=expectation.evidence.created_at,
                access_scope=expectation.evidence.access_scope,
            )
            evidence_retention = (
                terminal_evidence_retention
                if evidence_retention is None
                else min(evidence_retention, terminal_evidence_retention)
            )
        result = results[index]
        results[index] = result.model_copy(
            update={
                "lifecycle_correct": (
                    result.lifecycle_correct
                    and lifecycle_exact
                    and global_lifecycle_exact
                    and maintenance_exact
                    and revisions_exact
                    and recall_lifecycle_exact
                ),
                "lineage_correct": (
                    result.lineage_correct and lineage_exact and recall_lineage_exact
                ),
                "evidence_retention": evidence_retention,
                "recalled_entry_ids": recalled_entry_ids,
                "latency_ms": (
                    result.latency_ms
                    + global_audit_latency_ms
                    + (perf_counter_ns() - audit_started_ns) / 1_000_000
                ),
            }
        )

    metrics = _aggregate_metrics(tuple(results))
    return KnowledgeMaintenanceEvaluationResult(
        schema_version=KNOWLEDGE_MAINTENANCE_EVALUATION_RESULT_SCHEMA_VERSION,
        corpus_revision=corpus.corpus_revision,
        corpus_origin=corpus.origin,
        backend=backend,
        planner_identity="cayu.knowledge-maintenance-evaluation-fixture-planner.v1",
        evaluator_identity="cayu.knowledge-maintenance-evaluation-fixture-evaluator.v1",
        configuration=recorded_configuration,
        metrics=metrics,
        cases=tuple(results),
    )


async def _run_case(
    case: KnowledgeMaintenanceEvaluationCase,
    store: KnowledgeStore,
    *,
    privileged: KnowledgeAccessScope,
) -> tuple[KnowledgeMaintenanceEvaluationCaseResult, _CaseTerminalExpectation]:
    started_ns = perf_counter_ns()
    namespace = f"evaluation:knowledge-maintenance:{case.id}"
    labels = {"evaluation": "knowledge-maintenance-v1", "case": case.id}
    active_scope = KnowledgeAccessScope.for_namespace(
        namespace,
        required_labels=labels,
        allowed_visibilities=[KnowledgeVisibility.PROJECT],
        allowed_statuses=[KnowledgeStatus.ACTIVE],
        include_expired=True,
    )
    review_scope = KnowledgeAccessScope.for_namespace(
        namespace,
        required_labels=labels,
        allowed_visibilities=[KnowledgeVisibility.PROJECT],
        allowed_statuses=[KnowledgeStatus.ACTIVE, KnowledgeStatus.PENDING],
        include_expired=True,
    )
    entries = {
        fixture.id: KnowledgeEntry(
            id=fixture.id,
            text=fixture.text,
            namespace=namespace,
            labels=labels,
            visibility=KnowledgeVisibility.PROJECT,
            status=KnowledgeStatus.ACTIVE,
            created_by_type=KnowledgeActorType.APP,
            created_by="knowledge-maintenance-evaluation",
            created_at=_REFERENCE_TIME - timedelta(days=30),
            updated_at=_REFERENCE_TIME - timedelta(days=30),
            source_type="evaluation_fixture",
            source_id=f"fixture:{fixture.id}",
            source_hash=f"fixture-sha256:{fixture.id}:1",
        )
        for fixture in case.entries
    }
    for entry_id, entry in tuple(entries.items()):
        entries[entry_id] = await store.create_entry(
            entry,
            chunks=[
                KnowledgeChunk(
                    id=f"{entry.id}:r1:0",
                    entry_id=entry.id,
                    entry_revision=1,
                    chunk_index=0,
                    text=entry.text,
                )
            ],
            access_scope=privileged,
        )
    seeded_entries = dict(entries)
    expected_revision_entries = {
        (entry.id, entry.revision): entry for entry in seeded_entries.values()
    }
    terminal_entries = dict(seeded_entries)
    terminal_relations: list[KnowledgeRelation] = []

    contradiction_relation_id: str | None = None
    if case.signal_kind is KnowledgeMaintenanceSignalKind.CONTRADICTION:
        contradiction_relation_id = f"fixture-contradiction:{case.id}"
        left, right = (
            KnowledgeRevisionRef(entry_id=entry_id, revision=1)
            for entry_id in case.source_entry_ids
        )
        contradiction_relation = KnowledgeRelation(
            id=contradiction_relation_id,
            subject=left,
            object=right,
            kind=KnowledgeRelationKind.CONTRADICTS,
            created_by="knowledge-maintenance-evaluation",
            policy_id="reviewed-consolidation-evaluation-v1",
            created_at=_REFERENCE_TIME - timedelta(minutes=1),
        )
        await store.publish_relations(
            [contradiction_relation],
            operation_id=f"fixture-contradiction-publication:{case.id}",
            access_scope=privileged,
        )
        terminal_relations.append(contradiction_relation)

    signals = _signals(case, relation_id=contradiction_relation_id)
    request = KnowledgeMaintenanceRoutingRequest(
        id=f"fixture-routing:{case.id}",
        policy_id="reviewed-consolidation-evaluation-v1",
        namespace=namespace,
        labels=labels,
        access_scope=active_scope,
        signals=signals,
        created_at=_REFERENCE_TIME,
    )
    source_count = len(case.source_entry_ids)
    routed = await KnowledgeMaintenanceRouter(
        store,
        config=KnowledgeMaintenanceRouterConfig(
            max_signals=max(len(signals), 1),
            max_candidate_reads=max(source_count, 1),
            max_candidates=max(source_count, 1),
            max_candidate_bytes=1_048_576,
            max_concurrency=max(source_count, 1),
        ),
    ).route(request)
    routed_ids = tuple(candidate.reference.entry_id for candidate in routed.candidates)
    expected = set(case.expected_routed_entry_ids)
    actual = set(routed_ids)
    routing_precision = len(expected & actual) / len(actual) if actual else float(not expected)
    routing_recall = len(expected & actual) / len(expected) if expected else float(not actual)

    planning = await KnowledgeMaintenancePlanningWorkflow(
        store,
        planner=_FixturePlanner(case),
        evaluator=_FixtureEvaluator(case),
        config=_EVALUATION_PLANNING_CONFIG,
        clock=lambda: _REFERENCE_TIME,
    ).plan(request, routed)
    model_call_count = sum(
        usage.model_calls
        for usage in (planning.planner_usage, planning.evaluator_usage)
        if usage is not None
    )
    information_retention = _information_retention(case, planning)
    evidence_retention: float | None = None
    evidence_expectation: _CaseEvidenceExpectation | None = None
    maintenance_expectation: _CaseMaintenanceExpectation | None = None
    recall_expectation: _CaseRecallExpectation | None = None
    storage_outcome: Literal["not_published", "pending", "applied", "rejected", "stale"] = (
        "not_published"
    )
    lifecycle_correct = False
    lineage_correct = False
    recalled_ids: tuple[str, ...] = ()

    if planning.outcome is KnowledgeMaintenancePlanningOutcome.EVALUATOR_REJECTED:
        storage_outcome = "not_published"
        recalled_ids, lineage_correct = await _recall_unresolved_contradiction(
            case,
            store,
            expected_entries=seeded_entries,
            expected_relations=terminal_relations,
            active_scope=active_scope,
            namespace=namespace,
            privileged=privileged,
        )
        lifecycle_correct = await _entries_match_snapshots(
            store,
            seeded_entries,
            privileged=privileged,
        )
        recall_expectation = _CaseRecallExpectation(
            case=case,
            storage_outcome="not_published",
            active_scope=active_scope,
            namespace=namespace,
            expected_entries=tuple(seeded_entries.values()),
            expected_relations=tuple(terminal_relations),
        )
    elif planning.outcome is KnowledgeMaintenancePlanningOutcome.ACCEPTED:
        publisher = KnowledgeMaintenanceProposalPublisher(
            store,
            access_scope=review_scope,
            config=KnowledgeMaintenanceProposalPublisherConfig(
                publisher_id="knowledge-maintenance-evaluation-publisher",
                publisher_version="1",
            ),
        )
        publication = await publisher.publish(request, routed, planning)
        await _require_exact_pending_publication(publisher, publication)
        publication_replacement = KnowledgeRevisionRef(
            entry_id=publication.replacement.id,
            revision=publication.replacement.revision,
        )
        evidence_replacement = publication_replacement
        storage_outcome = "pending"
        expected_entries = dict(seeded_entries)
        terminal_entries = {
            **expected_entries,
            publication.replacement.id: publication.replacement,
        }
        expected_revision_entries[
            (publication.replacement.id, publication.replacement.revision)
        ] = publication.replacement

        if case.advance_source_before_review is not None:
            current = await store.get_entry(
                case.advance_source_before_review,
                access_scope=privileged,
            )
            if current is None:  # pragma: no cover - seeded above
                raise RuntimeError("Fixture source disappeared before its stale advance.")
            advanced = current.model_copy(
                update={
                    "revision": current.revision + 1,
                    "text": current.text + " A newer exact revision arrived before review.",
                    "updated_at": _REFERENCE_TIME + timedelta(minutes=1),
                }
            )
            expected_entries[case.advance_source_before_review] = await store.append_entry_revision(
                advanced,
                expected_revision=current.revision,
                access_scope=privileged,
            )
            advanced = expected_entries[case.advance_source_before_review]
            expected_revision_entries[(advanced.id, advanced.revision)] = advanced
            terminal_entries[case.advance_source_before_review] = expected_entries[
                case.advance_source_before_review
            ]

        decision_kind = case.review_decision
        if decision_kind is None:  # pragma: no cover - corpus invariant
            raise RuntimeError("Accepted fixture omitted its review decision.")
        decision = KnowledgeMaintenanceDecision(
            operation_id=f"fixture-decision:{case.id}",
            proposal_id=publication.proposal.id,
            proposal_fingerprint=publication.proposal.fingerprint,
            kind=decision_kind,
            reviewer_type=KnowledgeActorType.USER,
            reviewer="knowledge-maintenance-evaluation-reviewer",
            reason="The deterministic reference proposal was explicitly reviewed.",
            decided_at=_REFERENCE_TIME + timedelta(minutes=2),
        )
        receipt: KnowledgeMaintenanceDecisionReceipt | None = None
        final_publication_outcome = KnowledgeMaintenanceProposalPublicationOutcome.EXISTING_PENDING
        try:
            receipt = _validated_decision_receipt(
                await KnowledgeReviewWorkflow(
                    store,
                    access_scope=review_scope,
                    namespace=namespace,
                    labels=labels,
                ).decide_maintenance(publication.proposal, decision)
            )
        except KnowledgeMaintenanceStale as exc:
            if exc.reason != "source_revision":
                raise
            storage_outcome = "stale"
            lifecycle_correct = await _stale_lifecycle_is_safe(
                expected_entries,
                publication.replacement,
                decision.operation_id,
                store,
                privileged=privileged,
                review_scope=review_scope,
            )
            recalled_ids, recalled = await _recall_entry_ids(
                case,
                store,
                active_scope=active_scope,
                namespace=namespace,
            )
            lifecycle_correct = lifecycle_correct and await _recall_contains_exact_entries(
                recalled,
                (expected_entries[entry_id] for entry_id in case.source_entry_ids),
                store=store,
                scope=privileged,
            )
            lineage_correct = await _relations_are_absent(
                store,
                expected_entries=terminal_entries,
                scope=review_scope,
            )
        else:
            final_publication_outcome = (
                KnowledgeMaintenanceProposalPublicationOutcome.EXISTING_DECIDED
            )
            if decision_kind is KnowledgeMaintenanceDecisionKind.REJECT:
                storage_outcome = "rejected"
                rejected_entries = {
                    **expected_entries,
                    publication.replacement.id: publication.replacement,
                }
                terminal_entries = rejected_entries
                retained_replacement = await store.get_entry(
                    publication.replacement.id,
                    access_scope=review_scope,
                )
                recalled_ids, recalled = await _recall_entry_ids(
                    case,
                    store,
                    active_scope=active_scope,
                    namespace=namespace,
                )
                lifecycle_correct = (
                    receipt is not None
                    and _decision_receipt_is_exact(
                        publication.proposal,
                        decision,
                        receipt,
                    )
                    and await _entries_match_snapshots(
                        store,
                        rejected_entries,
                        privileged=privileged,
                    )
                    and retained_replacement == publication.replacement
                    and await _recall_contains_exact_entries(
                        recalled,
                        (expected_entries[entry_id] for entry_id in case.source_entry_ids),
                        store=store,
                        scope=privileged,
                    )
                    and publication.replacement.id not in recalled_ids
                    and _validated_decision_receipt(
                        await store.load_maintenance_decision_receipt(
                            decision.operation_id,
                            access_scope=review_scope,
                        )
                    )
                    == receipt
                )
                lineage_correct = await _relations_are_absent(
                    store,
                    expected_entries=rejected_entries,
                    scope=review_scope,
                )
            else:
                storage_outcome = "applied"
                expected_replacement = KnowledgeRevisionRef(
                    entry_id=publication.replacement.id,
                    revision=publication.replacement.revision + 1,
                )
                evidence_replacement = expected_replacement
                if receipt is not None:
                    approved_entries = _approved_entry_snapshots(
                        publication.proposal,
                        publication.replacement,
                        receipt,
                        expected_entries,
                    )
                    if approved_entries is not None:
                        terminal_entries = approved_entries
                terminal_relations.extend(publication.proposal.relations)
                lifecycle_correct = await _approved_lifecycle_is_correct(
                    publication.proposal,
                    publication.replacement,
                    decision,
                    receipt,
                    expected_entries,
                    store,
                    privileged=privileged,
                    review_scope=review_scope,
                )
                recalled_ids, lineage_correct = await _approved_recall_and_lineage(
                    case,
                    expected_replacement,
                    publication.replacement,
                    publication.proposal,
                    terminal_entries,
                    store,
                    active_scope=active_scope,
                    namespace=namespace,
                    privileged=privileged,
                )
        evidence_retention = await _exact_evidence_retention(
            store,
            publication_replacement=publication_replacement,
            replacement=evidence_replacement,
            expected_sources=[candidate.entry for candidate in routed.candidates],
            accepted_plan_fingerprint=publication.accepted_plan.fingerprint,
            created_at=publication.proposal.created_at,
            access_scope=review_scope,
        )
        evidence_expectation = _CaseEvidenceExpectation(
            publication_replacement=publication_replacement,
            replacement=evidence_replacement,
            expected_sources=tuple(candidate.entry for candidate in routed.candidates),
            accepted_plan_fingerprint=publication.accepted_plan.fingerprint,
            created_at=publication.proposal.created_at,
            access_scope=review_scope,
        )
        maintenance_expectation = _CaseMaintenanceExpectation(
            publication=KnowledgeMaintenanceProposalPublication(
                proposal=publication.proposal,
                accepted_plan=publication.accepted_plan,
                replacement=publication.replacement,
                receipt=copy_knowledge_maintenance_proposal_publication_receipt(
                    publication.receipt,
                    replayed=True,
                ),
                outcome=final_publication_outcome,
            ),
            decision_operation_id=decision.operation_id,
            decision=(
                decision
                if final_publication_outcome
                is KnowledgeMaintenanceProposalPublicationOutcome.EXISTING_DECIDED
                else None
            ),
            decision_receipt=receipt,
            access_scope=review_scope,
        )
        if storage_outcome not in {"applied", "rejected", "stale"}:  # pragma: no cover
            raise RuntimeError("Accepted fixture did not reach a terminal storage outcome.")
        recall_expectation = _CaseRecallExpectation(
            case=case,
            storage_outcome=storage_outcome,
            active_scope=active_scope,
            namespace=namespace,
            expected_entries=tuple(terminal_entries.values()),
            expected_relations=tuple(terminal_relations),
            pending_replacement=publication.replacement,
            proposal=publication.proposal,
            replacement=evidence_replacement if storage_outcome == "applied" else None,
        )

    for entry in terminal_entries.values():
        expected_revision_entries[(entry.id, entry.revision)] = entry
    revision_expectations, revisions_exact = await _capture_revision_expectations(
        store,
        expected_revision_entries.values(),
        privileged=privileged,
    )
    lifecycle_correct = lifecycle_correct and revisions_exact

    unsafe_acceptance = int(
        case.evaluator_verdict is KnowledgeMaintenanceEvaluationVerdict.REJECTED
        and planning.outcome is KnowledgeMaintenancePlanningOutcome.ACCEPTED
    )
    result = KnowledgeMaintenanceEvaluationCaseResult(
        case_id=case.id,
        scenario=case.scenario,
        routed_entry_ids=routed_ids,
        routing_precision=routing_precision,
        routing_recall=routing_recall,
        planning_outcome=planning.outcome,
        storage_outcome=storage_outcome,
        information_retention=information_retention,
        evidence_retention=evidence_retention,
        unsafe_acceptance_count=unsafe_acceptance,
        lifecycle_correct=lifecycle_correct,
        lineage_correct=lineage_correct,
        recalled_entry_ids=recalled_ids,
        model_call_count=model_call_count,
        latency_ms=(perf_counter_ns() - started_ns) / 1_000_000,
    )
    expectation = _CaseTerminalExpectation(
        entries=tuple(sorted(terminal_entries.values(), key=lambda entry: entry.id)),
        revisions=revision_expectations,
        relations=tuple(sorted(terminal_relations, key=lambda relation: relation.id)),
        evidence=evidence_expectation,
        maintenance=maintenance_expectation,
        recall=recall_expectation,
    )
    return result, expectation


async def _require_exact_pending_publication(
    publisher: KnowledgeMaintenanceProposalPublisher,
    publication: KnowledgeMaintenanceProposalPublication,
) -> None:
    loaded = await publisher.load(publication.proposal.id)
    if (
        loaded is None
        or loaded.proposal != publication.proposal
        or loaded.accepted_plan != publication.accepted_plan
        or loaded.replacement != publication.replacement
        or loaded.receipt
        != copy_knowledge_maintenance_proposal_publication_receipt(
            publication.receipt,
            replayed=True,
        )
        or loaded.outcome is not KnowledgeMaintenanceProposalPublicationOutcome.EXISTING_PENDING
    ):
        raise RuntimeError(
            "Maintenance proposal publication does not match its exact durable artifact."
        )


async def _maintenance_artifacts_match(
    store: KnowledgeStore,
    expected: _CaseMaintenanceExpectation,
) -> bool:
    raw_publication = await store.load_maintenance_proposal_publication(
        expected.publication.proposal.id,
        access_scope=expected.access_scope,
    )
    try:
        publication = (
            None
            if raw_publication is None
            else copy_knowledge_maintenance_proposal_publication(raw_publication)
        )
    except (TypeError, ValueError):
        publication = None
    decision_receipt = _validated_decision_receipt(
        await store.load_maintenance_decision_receipt(
            expected.decision_operation_id,
            access_scope=expected.access_scope,
        )
    )
    decision = await store.load_maintenance_decision(
        expected.decision_operation_id,
        access_scope=expected.access_scope,
    )
    if type(decision) is not KnowledgeMaintenanceDecision:
        decision = None
    return (
        publication == expected.publication
        and decision == expected.decision
        and decision_receipt == expected.decision_receipt
    )


def _signals(
    case: KnowledgeMaintenanceEvaluationCase,
    *,
    relation_id: str | None,
) -> tuple[KnowledgeMaintenanceCandidateSignal, ...]:
    references = tuple(
        KnowledgeRevisionRef(entry_id=entry_id, revision=1) for entry_id in case.source_entry_ids
    )
    if case.signal_kind in {
        KnowledgeMaintenanceSignalKind.DUPLICATE_HINT,
        KnowledgeMaintenanceSignalKind.CONTRADICTION,
    }:
        return (
            KnowledgeMaintenanceCandidateSignal(
                id=f"fixture-signal:{case.id}",
                kind=case.signal_kind,
                references=references,
                producer_id="knowledge-maintenance-evaluation",
                producer_version="1",
                reason_code=case.signal_kind.value,
                observed_at=_REFERENCE_TIME,
                relation_id=relation_id,
                raw_score=(
                    1.0
                    if case.signal_kind is KnowledgeMaintenanceSignalKind.DUPLICATE_HINT
                    else None
                ),
                score_kind=(
                    "fixture_exact_match"
                    if case.signal_kind is KnowledgeMaintenanceSignalKind.DUPLICATE_HINT
                    else None
                ),
            ),
        )
    return tuple(
        KnowledgeMaintenanceCandidateSignal(
            id=f"fixture-signal:{case.id}:{index:02d}",
            kind=case.signal_kind,
            references=(reference,),
            producer_id="knowledge-maintenance-evaluation",
            producer_version="1",
            reason_code=case.signal_kind.value,
            observed_at=_REFERENCE_TIME,
        )
        for index, reference in enumerate(references)
    )


def _information_retention(
    case: KnowledgeMaintenanceEvaluationCase,
    planning,
) -> float:
    if planning.plan is None:
        return 0.0
    mappings = {mapping.id: mapping for mapping in planning.plan.evidence_mappings}
    retained = 0
    for claim in case.claims:
        mapping = mappings.get(claim.id)
        if mapping is None:
            continue
        sources = {reference.entry_id for reference in mapping.source_references}
        if claim.text in planning.plan.replacement.text and sources == set(claim.source_entry_ids):
            retained += 1
    return retained / len(case.claims)


async def _exact_evidence_retention(
    store: KnowledgeStore,
    *,
    publication_replacement: KnowledgeRevisionRef,
    replacement: KnowledgeRevisionRef,
    expected_sources: Iterable[KnowledgeEntry],
    accepted_plan_fingerprint: str,
    created_at: datetime,
    access_scope: KnowledgeAccessScope,
) -> float:
    expected_sources = tuple(expected_sources)
    raw_evidence = await store.read_evidence(
        replacement.entry_id,
        revision=replacement.revision,
        access_scope=access_scope,
        max_records=100,
        max_bytes=1_048_576,
    )
    evidence = _validated_evidence_result(raw_evidence)
    if (
        evidence is None
        or evidence.entry_id != replacement.entry_id
        or evidence.entry_revision != replacement.revision
        or evidence.limit != 100
        or evidence.max_bytes != 1_048_576
        or evidence.truncated
        or evidence.total_evidence_known != len(expected_sources)
    ):
        return 0.0
    publication_evidence = [
        _expected_source_evidence(
            publication_replacement,
            source,
            source_ordinal=ordinal,
            accepted_plan_fingerprint=accepted_plan_fingerprint,
            created_at=created_at,
        )
        for ordinal, source in enumerate(expected_sources)
    ]
    expected = [
        _expected_successor_evidence(item, replacement=replacement) for item in publication_evidence
    ]
    observed_by_id = {item.id: item for item in evidence.evidence}
    if len(observed_by_id) != len(evidence.evidence):
        return 0.0
    retained = sum(observed_by_id.get(item.id) == item for item in expected)
    return retained / max(len(expected), len(evidence.evidence), 1)


def _expected_successor_evidence(
    evidence: KnowledgeEvidence,
    *,
    replacement: KnowledgeRevisionRef,
) -> KnowledgeEvidence:
    if evidence.entry_id != replacement.entry_id:
        raise ValueError("Evidence replacement identity changed across review.")
    if evidence.entry_revision == replacement.revision:
        return evidence
    if evidence.entry_revision + 1 != replacement.revision:
        raise ValueError("Evidence replacement revision is not the reviewed successor.")
    evidence_id = (
        "ke_"
        + sha256(
            canonical_durable_json_bytes(
                {
                    "contract": "cayu-knowledge-evidence-successor-v1",
                    "source_evidence_id": evidence.id,
                    "entry_id": replacement.entry_id,
                    "entry_revision": replacement.revision,
                },
                "knowledge evidence successor identity",
            )
        ).hexdigest()
    )
    return KnowledgeEvidence.model_validate(
        {
            **evidence.model_dump(mode="python"),
            "id": evidence_id,
            "entry_revision": replacement.revision,
        }
    )


def _expected_source_evidence(
    replacement: KnowledgeRevisionRef,
    source: KnowledgeEntry,
    *,
    source_ordinal: int,
    accepted_plan_fingerprint: str,
    created_at: datetime,
) -> KnowledgeEvidence:
    source_reference = KnowledgeRevisionRef(entry_id=source.id, revision=source.revision)
    source_hash = sha256(
        canonical_durable_json_bytes(
            source.model_dump(mode="json"),
            "maintenance source revision",
        )
    ).hexdigest()
    evidence_id = (
        "maintenance-evidence-"
        + sha256(
            canonical_durable_json_bytes(
                {
                    "accepted_plan_fingerprint": accepted_plan_fingerprint,
                    "replacement_entry_id": replacement.entry_id,
                    "source": source_reference.model_dump(mode="json"),
                },
                "knowledge maintenance source evidence identity",
            )
        ).hexdigest()
    )
    return KnowledgeEvidence(
        id=evidence_id,
        entry_id=replacement.entry_id,
        entry_revision=replacement.revision,
        role=KnowledgeEvidenceRole.ORIGIN,
        source_type="knowledge_revision",
        source_id=source.id,
        source_revision=str(source.revision),
        source_hash=source_hash,
        locator={"entry_id": source.id, "revision": source.revision},
        disposition=KnowledgeEvidenceDisposition.LIVE,
        created_at=created_at,
        metadata={
            "source_ordinal": source_ordinal,
            "accepted_plan_fingerprint": accepted_plan_fingerprint,
        },
    )


def _validated_decision_receipt(value: Any) -> KnowledgeMaintenanceDecisionReceipt | None:
    if type(value) is not KnowledgeMaintenanceDecisionReceipt:
        return None
    try:
        return KnowledgeMaintenanceDecisionReceipt(
            operation_id=value.operation_id,
            proposal_id=value.proposal_id,
            proposal_fingerprint=value.proposal_fingerprint,
            request_sha256=value.request_sha256,
            outcome=value.outcome,
            replacement=value.replacement,
            archived_revisions=list(value.archived_revisions),
            relation_ids=list(value.relation_ids),
            committed_at=value.committed_at,
            replayed=value.replayed,
        )
    except (TypeError, ValueError):
        return None


def _validated_evidence_result(value: Any) -> KnowledgeEvidenceResult | None:
    if type(value) is not KnowledgeEvidenceResult:
        return None
    try:
        return KnowledgeEvidenceResult(
            entry_id=value.entry_id,
            entry_revision=value.entry_revision,
            evidence=list(value.evidence),
            truncated=value.truncated,
            limit=value.limit,
            max_bytes=value.max_bytes,
            total_evidence_known=value.total_evidence_known,
        )
    except (TypeError, ValueError):
        return None


async def _entries_match_snapshots(
    store: KnowledgeStore,
    expected: dict[str, KnowledgeEntry],
    *,
    privileged: KnowledgeAccessScope,
) -> bool:
    namespaces = {entry.namespace for entry in expected.values()}
    if len(namespaces) != 1:
        return False
    query = KnowledgeListQuery(
        namespace=next(iter(namespaces)),
        statuses=list(KnowledgeStatus),
        include_expired=True,
        limit=_MAX_ENTRIES_PER_CASE + 2,
        max_bytes=_MAX_RESULT_BYTES,
    )
    return await _entries_match_query(
        store,
        expected,
        query=query,
        privileged=privileged,
    )


async def _all_entries_match_snapshots(
    store: KnowledgeStore,
    expected: dict[str, KnowledgeEntry],
    *,
    privileged: KnowledgeAccessScope,
) -> bool:
    query = KnowledgeListQuery(
        statuses=list(KnowledgeStatus),
        include_expired=True,
        limit=len(expected) + 1,
        max_bytes=_MAX_RESULT_BYTES,
    )
    return await _entries_match_query(
        store,
        expected,
        query=query,
        privileged=privileged,
    )


async def _capture_revision_expectations(
    store: KnowledgeStore,
    expected_entries: Iterable[KnowledgeEntry],
    *,
    privileged: KnowledgeAccessScope,
) -> tuple[tuple[_CaseRevisionExpectation, ...], bool]:
    expectations: list[_CaseRevisionExpectation] = []
    exact = True
    for expected_entry in sorted(
        expected_entries,
        key=lambda entry: (entry.id, entry.revision),
    ):
        observed_entry = await store.get_entry(
            expected_entry.id,
            revision=expected_entry.revision,
            max_bytes=_MAX_RESULT_BYTES,
            access_scope=privileged,
        )
        chunks = _validated_revision_chunks(
            await store.read_chunks(
                expected_entry.id,
                revision=expected_entry.revision,
                access_scope=privileged,
                max_chunks=DEFAULT_KNOWLEDGE_INDEX_MAX_CHUNKS,
                max_bytes=_MAX_RESULT_BYTES,
            ),
            expected_entry,
        )
        if observed_entry != expected_entry or chunks is None:
            exact = False
        expectations.append(
            _CaseRevisionExpectation(
                entry=expected_entry,
                chunks=() if chunks is None else chunks,
            )
        )
    return tuple(expectations), exact


async def _revision_expectations_match(
    store: KnowledgeStore,
    expectations: Iterable[_CaseRevisionExpectation],
    *,
    privileged: KnowledgeAccessScope,
) -> bool:
    for expectation in expectations:
        entry = expectation.entry
        observed_entry = await store.get_entry(
            entry.id,
            revision=entry.revision,
            max_bytes=_MAX_RESULT_BYTES,
            access_scope=privileged,
        )
        chunks = _validated_revision_chunks(
            await store.read_chunks(
                entry.id,
                revision=entry.revision,
                access_scope=privileged,
                max_chunks=DEFAULT_KNOWLEDGE_INDEX_MAX_CHUNKS,
                max_bytes=_MAX_RESULT_BYTES,
            ),
            entry,
        )
        if observed_entry != entry or chunks != expectation.chunks:
            return False
    return True


def _validated_revision_chunks(
    value: Any,
    entry: KnowledgeEntry,
) -> tuple[KnowledgeChunk, ...] | None:
    if type(value) is not list or not value or len(value) >= DEFAULT_KNOWLEDGE_INDEX_MAX_CHUNKS:
        return None
    try:
        chunks = tuple(
            KnowledgeChunk.model_validate(chunk.model_dump(mode="python"))
            for chunk in value
            if type(chunk) is KnowledgeChunk
        )
    except (AttributeError, TypeError, ValueError):
        return None
    if len(chunks) != len(value):
        return None
    chunk_ids = [chunk.id for chunk in chunks]
    chunk_indexes = [chunk.chunk_index for chunk in chunks]
    if (
        chunks != tuple(sorted(chunks, key=lambda chunk: chunk.chunk_index))
        or len(set(chunk_ids)) != len(chunks)
        or len(set(chunk_indexes)) != len(chunks)
        or any(
            chunk.entry_id != entry.id or chunk.entry_revision != entry.revision for chunk in chunks
        )
    ):
        return None
    return chunks


async def _entries_match_query(
    store: KnowledgeStore,
    expected: dict[str, KnowledgeEntry],
    *,
    query: KnowledgeListQuery,
    privileged: KnowledgeAccessScope,
) -> bool:
    listed = _validated_list_result(await store.list_entries(query, access_scope=privileged))
    if (
        listed is None
        or listed.query != query
        or listed.truncated
        or (listed.total_entries_known is not None and listed.total_entries_known != len(expected))
        or len(listed.entries) != len(expected)
    ):
        return False
    listed_entries = {item.entry.id: item.entry for item in listed.entries}
    if len(listed_entries) != len(listed.entries) or listed_entries != expected:
        return False
    for entry_id, snapshot in expected.items():
        current = await store.get_entry(entry_id, access_scope=privileged)
        exact = await store.get_entry(
            entry_id,
            revision=snapshot.revision,
            access_scope=privileged,
        )
        if current != snapshot or exact != snapshot:
            return False
    return True


def _validated_list_result(value: Any) -> KnowledgeListResult | None:
    if type(value) is not KnowledgeListResult:
        return None
    try:
        return KnowledgeListResult.model_validate(value.model_dump(mode="python"))
    except (TypeError, ValueError):
        return None


def _validated_lineage_result(value: Any) -> KnowledgeLineageResult | None:
    if type(value) is not KnowledgeLineageResult:
        return None
    try:
        return KnowledgeLineageResult(
            query=value.query,
            reference_current=value.reference_current,
            reference_status=value.reference_status,
            links=list(value.links),
            truncated=value.truncated,
            next_cursor=value.next_cursor,
        )
    except (TypeError, ValueError):
        return None


async def _inspect_exact_lineage(
    reference: KnowledgeRevisionRef,
    store: KnowledgeStore,
    *,
    scope: KnowledgeAccessScope,
) -> KnowledgeLineageResult | None:
    query = KnowledgeLineageQuery(
        reference=reference,
        limit=MAX_KNOWLEDGE_MAINTENANCE_SOURCES,
        max_bytes=_MAX_LINEAGE_BYTES,
    )
    lineage = _validated_lineage_result(await store.inspect_lineage(query, access_scope=scope))
    if (
        lineage is None
        or lineage.query != query
        or lineage.truncated
        or lineage.next_cursor is not None
    ):
        return None
    return lineage


async def _relations_match_snapshots(
    store: KnowledgeStore,
    *,
    expected_entries: dict[str, KnowledgeEntry],
    expected_relations: Iterable[KnowledgeRelation],
    scope: KnowledgeAccessScope,
) -> bool:
    relations = tuple(expected_relations)
    for entry in expected_entries.values():
        for revision in range(1, entry.revision + 1):
            reference = KnowledgeRevisionRef(entry_id=entry.id, revision=revision)
            lineage = await _inspect_exact_lineage(
                reference,
                store,
                scope=scope,
            )
            if lineage is None or not _lineage_result_matches_expected(
                lineage,
                reference,
                relations,
                expected_entries=expected_entries,
            ):
                return False
    return True


def _lineage_result_matches_expected(
    lineage: KnowledgeLineageResult,
    reference: KnowledgeRevisionRef,
    relations: Iterable[KnowledgeRelation],
    *,
    expected_entries: dict[str, KnowledgeEntry],
) -> bool:
    current = expected_entries.get(reference.entry_id)
    expected_links = _expected_lineage_link_models(
        reference,
        relations,
        expected_entries=expected_entries,
    )
    return (
        current is not None
        and expected_links is not None
        and lineage.query.reference == reference
        and lineage.query.limit == MAX_KNOWLEDGE_MAINTENANCE_SOURCES
        and lineage.query.max_bytes == _MAX_LINEAGE_BYTES
        and not lineage.truncated
        and lineage.next_cursor is None
        and lineage.reference_current
        == KnowledgeRevisionRef(entry_id=current.id, revision=current.revision)
        and lineage.reference_status is current.status
        and lineage.links == expected_links
    )


async def _relations_are_absent(
    store: KnowledgeStore,
    *,
    expected_entries: dict[str, KnowledgeEntry],
    scope: KnowledgeAccessScope,
) -> bool:
    return await _relations_match_snapshots(
        store,
        expected_entries=expected_entries,
        expected_relations=(),
        scope=scope,
    )


def _approved_entry_snapshots(
    proposal: KnowledgeMaintenanceProposal,
    pending_replacement: KnowledgeEntry,
    receipt: KnowledgeMaintenanceDecisionReceipt,
    seeded_entries: dict[str, KnowledgeEntry],
) -> dict[str, KnowledgeEntry] | None:
    expected_entries = dict(seeded_entries)
    superseded = {
        relation.object
        for relation in proposal.relations
        if relation.kind is KnowledgeRelationKind.SUPERSEDES
    }
    for source in proposal.sources:
        original = expected_entries.get(source.entry_id)
        if (
            original is None
            or KnowledgeRevisionRef(entry_id=original.id, revision=original.revision) != source
        ):
            return None
        if source in superseded:
            expected_entries[source.entry_id] = original.model_copy(
                update={
                    "revision": original.revision + 1,
                    "status": KnowledgeStatus.ARCHIVED,
                    "updated_at": max(
                        receipt.committed_at,
                        original.created_at,
                        original.updated_at,
                    ),
                }
            )

    expected_replacement = pending_replacement.model_copy(
        update={
            "revision": pending_replacement.revision + 1,
            "status": KnowledgeStatus.ACTIVE,
            "updated_at": max(
                receipt.committed_at,
                pending_replacement.created_at,
                pending_replacement.updated_at,
            ),
        }
    )
    expected_entries[expected_replacement.id] = expected_replacement
    return expected_entries


async def _stale_lifecycle_is_safe(
    expected_sources: dict[str, KnowledgeEntry],
    expected_replacement: KnowledgeEntry,
    decision_operation_id: str,
    store: KnowledgeStore,
    *,
    privileged: KnowledgeAccessScope,
    review_scope: KnowledgeAccessScope,
) -> bool:
    replacement = await store.get_entry(expected_replacement.id, access_scope=review_scope)
    expected_entries = {
        **expected_sources,
        expected_replacement.id: expected_replacement,
    }
    return (
        replacement == expected_replacement
        and await _entries_match_snapshots(
            store,
            expected_entries,
            privileged=privileged,
        )
        and await store.load_maintenance_decision_receipt(
            decision_operation_id,
            access_scope=review_scope,
        )
        is None
    )


async def _approved_lifecycle_is_correct(
    proposal: KnowledgeMaintenanceProposal,
    pending_replacement: KnowledgeEntry,
    decision: KnowledgeMaintenanceDecision,
    receipt: KnowledgeMaintenanceDecisionReceipt | None,
    seeded_entries: dict[str, KnowledgeEntry],
    store: KnowledgeStore,
    *,
    privileged: KnowledgeAccessScope,
    review_scope: KnowledgeAccessScope,
) -> bool:
    if receipt is None or not _decision_receipt_is_exact(proposal, decision, receipt):
        return False
    persisted_receipt = _validated_decision_receipt(
        await store.load_maintenance_decision_receipt(
            decision.operation_id,
            access_scope=review_scope,
        )
    )
    if persisted_receipt != receipt:
        return False

    expected_entries = _approved_entry_snapshots(
        proposal,
        pending_replacement,
        receipt,
        seeded_entries,
    )
    if expected_entries is None:
        return False
    return await _entries_match_snapshots(
        store,
        expected_entries,
        privileged=privileged,
    )


def _decision_receipt_is_exact(
    proposal: KnowledgeMaintenanceProposal,
    decision: KnowledgeMaintenanceDecision,
    receipt: KnowledgeMaintenanceDecisionReceipt,
) -> bool:
    _, _, request_sha256 = prepare_knowledge_maintenance_decision(proposal, decision)
    common_matches = (
        receipt.operation_id == decision.operation_id
        and receipt.proposal_id == proposal.id
        and receipt.proposal_fingerprint == proposal.fingerprint
        and receipt.request_sha256 == request_sha256
        and receipt.committed_at >= proposal.created_at
        and receipt.committed_at >= decision.decided_at
        and not receipt.replayed
    )
    if decision.kind is KnowledgeMaintenanceDecisionKind.REJECT:
        return (
            common_matches
            and receipt.outcome is KnowledgeMaintenanceOutcome.REJECTED
            and receipt.replacement is None
            and not receipt.archived_revisions
            and not receipt.relation_ids
        )

    expected_replacement = KnowledgeRevisionRef(
        entry_id=proposal.replacement.entry_id,
        revision=proposal.replacement.revision + 1,
    )
    superseded = {
        relation.object
        for relation in proposal.relations
        if relation.kind is KnowledgeRelationKind.SUPERSEDES
    }
    expected_archived = sorted(
        (
            KnowledgeRevisionRef(
                entry_id=source.entry_id,
                revision=source.revision + 1,
            )
            for source in proposal.sources
            if source in superseded
        ),
        key=lambda reference: (reference.entry_id, reference.revision),
    )
    return (
        common_matches
        and receipt.outcome is KnowledgeMaintenanceOutcome.APPLIED
        and receipt.replacement == expected_replacement
        and receipt.archived_revisions == expected_archived
        and receipt.relation_ids == sorted(relation.id for relation in proposal.relations)
    )


def _recall_reference(record: Any) -> KnowledgeRevisionRef | None:
    entry_id = record.locator.get("entry_id")
    entry_revision = record.locator.get("entry_revision")
    if type(entry_id) is not str or type(entry_revision) is not int:
        return None
    try:
        return KnowledgeRevisionRef(entry_id=entry_id, revision=entry_revision)
    except ValueError:
        return None


async def _recall_contains_exact_entries(
    recalled: Any,
    expected_entries: Iterable[KnowledgeEntry],
    *,
    store: KnowledgeStore,
    scope: KnowledgeAccessScope,
) -> bool:
    expected_entries = tuple(expected_entries)
    observed = {
        reference
        for record in recalled.records
        if (reference := _recall_reference(record)) is not None
    }
    expected = {
        KnowledgeRevisionRef(entry_id=entry.id, revision=entry.revision)
        for entry in expected_entries
    }
    expected_entry_ids = {reference.entry_id for reference in expected}
    observed_expected_entries = {
        reference for reference in observed if reference.entry_id in expected_entry_ids
    }
    if observed_expected_entries != expected:
        return False
    for entry in expected_entries:
        reference = KnowledgeRevisionRef(entry_id=entry.id, revision=entry.revision)
        records = [record for record in recalled.records if _recall_reference(record) == reference]
        if not records or not await _recall_records_match_exact_material(
            records,
            reference=reference,
            expected_text=entry.text,
            store=store,
            scope=scope,
        ):
            return False
    return True


async def _recall_entry_ids(
    case: KnowledgeMaintenanceEvaluationCase,
    store: KnowledgeStore,
    *,
    active_scope: KnowledgeAccessScope,
    namespace: str,
) -> tuple[tuple[str, ...], Any]:
    recalled = await KnowledgeRecallSource(
        store,
        max_record_bytes=_MAX_RECALL_RECORD_BYTES,
        lineage_limit=MAX_KNOWLEDGE_MAINTENANCE_SOURCES,
        lineage_max_bytes=_MAX_LINEAGE_BYTES,
    ).retrieve(
        RecallSituation(
            query=case.recall_query,
            knowledge_access_scope=active_scope,
            knowledge_namespace=namespace,
            current_time=_REFERENCE_TIME + timedelta(minutes=3),
        )
    )
    entry_ids = tuple(
        sorted(
            {
                entry_id
                for record in recalled.records
                if isinstance((entry_id := record.locator.get("entry_id")), str)
            }
        )
    )
    return entry_ids, recalled


async def _rerun_terminal_recall(
    expectation: _CaseRecallExpectation,
    store: KnowledgeStore,
    *,
    privileged: KnowledgeAccessScope,
) -> tuple[tuple[str, ...], bool, bool]:
    expected_entries = {entry.id: entry for entry in expectation.expected_entries}
    if expectation.storage_outcome == "not_published":
        entry_ids, correct = await _recall_unresolved_contradiction(
            expectation.case,
            store,
            expected_entries=expected_entries,
            expected_relations=expectation.expected_relations,
            active_scope=expectation.active_scope,
            namespace=expectation.namespace,
            privileged=privileged,
        )
        return entry_ids, True, correct

    if expectation.storage_outcome == "applied":
        if (
            expectation.pending_replacement is None
            or expectation.proposal is None
            or expectation.replacement is None
        ):  # pragma: no cover - internal expectation invariant
            raise RuntimeError("Applied terminal recall expectation is incomplete.")
        entry_ids, correct = await _approved_recall_and_lineage(
            expectation.case,
            expectation.replacement,
            expectation.pending_replacement,
            expectation.proposal,
            expected_entries,
            store,
            active_scope=expectation.active_scope,
            namespace=expectation.namespace,
            privileged=privileged,
        )
        return entry_ids, True, correct

    if expectation.pending_replacement is None:  # pragma: no cover
        raise RuntimeError("Reviewed terminal recall expectation is incomplete.")
    entry_ids, recalled = await _recall_entry_ids(
        expectation.case,
        store,
        active_scope=expectation.active_scope,
        namespace=expectation.namespace,
    )
    correct = (
        expectation.pending_replacement.id not in entry_ids
        and await _recall_contains_exact_entries(
            recalled,
            (expected_entries[entry_id] for entry_id in expectation.case.source_entry_ids),
            store=store,
            scope=privileged,
        )
    )
    return entry_ids, correct, True


async def _recall_unresolved_contradiction(
    case: KnowledgeMaintenanceEvaluationCase,
    store: KnowledgeStore,
    *,
    expected_entries: dict[str, KnowledgeEntry],
    expected_relations: Iterable[KnowledgeRelation],
    active_scope: KnowledgeAccessScope,
    namespace: str,
    privileged: KnowledgeAccessScope,
) -> tuple[tuple[str, ...], bool]:
    entry_ids, recalled = await _recall_entry_ids(
        case,
        store,
        active_scope=active_scope,
        namespace=namespace,
    )
    expected_references = {
        KnowledgeRevisionRef(entry_id=entry_id, revision=1) for entry_id in case.source_entry_ids
    }
    records = [
        (record, reference)
        for record in recalled.records
        if (reference := _recall_reference(record)) is not None
        and reference.entry_id in set(case.source_entry_ids)
    ]
    observed_references = {reference for _, reference in records}
    correct = observed_references == expected_references and all(
        record.lineage is not None
        and _lineage_result_matches_expected(
            record.lineage,
            reference,
            expected_relations,
            expected_entries=expected_entries,
        )
        for record, reference in records
    )
    if correct:
        correct = await _recall_contains_exact_entries(
            recalled,
            (expected_entries[entry_id] for entry_id in case.source_entry_ids),
            store=store,
            scope=privileged,
        )
    return entry_ids, correct


async def _recall_records_match_exact_material(
    records: list[RecallRecord],
    *,
    reference: KnowledgeRevisionRef,
    expected_text: str,
    store: KnowledgeStore,
    scope: KnowledgeAccessScope,
) -> bool:
    exact_entry = await store.get_entry(
        reference.entry_id,
        revision=reference.revision,
        access_scope=scope,
    )
    if exact_entry is None or exact_entry.text != expected_text:
        return False
    chunks = await store.read_chunks(
        reference.entry_id,
        revision=reference.revision,
        access_scope=scope,
        max_chunks=DEFAULT_KNOWLEDGE_INDEX_MAX_CHUNKS,
        max_bytes=_MAX_RESULT_BYTES,
    )
    chunks_by_id = {chunk.id: chunk for chunk in chunks}
    if len(chunks_by_id) != len(chunks):
        return False

    for record in records:
        if record.identity.revision != str(reference.revision):
            return False
        if record.representation == "entry_text":
            if (
                record.identity.record_type != "knowledge_entry"
                or record.identity.record_id != reference.entry_id
                or dict(record.locator)
                != {
                    "entry_id": reference.entry_id,
                    "entry_revision": reference.revision,
                }
            ):
                return False
            canonical_text = expected_text
        elif record.representation == "chunk_text":
            chunk_id = record.locator.get("chunk_id")
            chunk_index = record.locator.get("chunk_index")
            chunk = chunks_by_id.get(chunk_id) if type(chunk_id) is str else None
            if (
                chunk is None
                or type(chunk_index) is not int
                or record.identity.record_type != "knowledge_chunk"
                or record.identity.record_id != chunk.id
                or chunk.entry_id != reference.entry_id
                or chunk.entry_revision != reference.revision
                or chunk.chunk_index != chunk_index
                or dict(record.locator)
                != {
                    "entry_id": reference.entry_id,
                    "entry_revision": reference.revision,
                    "chunk_id": chunk.id,
                    "chunk_index": chunk.chunk_index,
                }
            ):
                return False
            canonical_text = chunk.text
        else:
            return False

        encoded = canonical_text.encode("utf-8")
        expected_preview = encoded[:_MAX_RECALL_RECORD_BYTES].decode(
            "utf-8",
            errors="ignore",
        )
        if (
            record.text != expected_preview
            or record.text_complete != (len(expected_preview.encode("utf-8")) == len(encoded))
            or record.content_hash != sha256(encoded).hexdigest()
        ):
            return False
    return True


async def _approved_recall_and_lineage(
    case: KnowledgeMaintenanceEvaluationCase,
    replacement_ref: KnowledgeRevisionRef | None,
    pending_replacement: KnowledgeEntry,
    proposal: KnowledgeMaintenanceProposal,
    expected_entries: dict[str, KnowledgeEntry],
    store: KnowledgeStore,
    *,
    active_scope: KnowledgeAccessScope,
    namespace: str,
    privileged: KnowledgeAccessScope,
) -> tuple[tuple[str, ...], bool]:
    expected_replacement = (
        expected_entries.get(replacement_ref.entry_id) if replacement_ref is not None else None
    )
    if (
        replacement_ref is None
        or expected_replacement is None
        or KnowledgeRevisionRef(
            entry_id=expected_replacement.id,
            revision=expected_replacement.revision,
        )
        != replacement_ref
    ):
        return (), False
    entry_ids, recalled = await _recall_entry_ids(
        case,
        store,
        active_scope=active_scope,
        namespace=namespace,
    )
    replacement_records = [
        record
        for record in recalled.records
        if (record_reference := _recall_reference(record)) is not None
        and record_reference.entry_id == replacement_ref.entry_id
    ]
    if (
        not replacement_records
        or any(_recall_reference(record) != replacement_ref for record in replacement_records)
        or set(case.source_entry_ids) & set(entry_ids)
        or not await _recall_records_match_exact_material(
            replacement_records,
            reference=replacement_ref,
            expected_text=pending_replacement.text,
            store=store,
            scope=privileged,
        )
    ):
        return entry_ids, False
    correct = all(
        record.lineage is not None
        and _lineage_result_matches_expected(
            record.lineage,
            replacement_ref,
            proposal.relations,
            expected_entries=expected_entries,
        )
        for record in replacement_records
    )
    if case.scenario is KnowledgeMaintenanceEvaluationScenario.HISTORICAL_LINEAGE:
        if not correct:
            return entry_ids, False
        fixtures = {entry.id: entry for entry in case.entries}
        for entry_id in case.source_entry_ids:
            source_ref = KnowledgeRevisionRef(entry_id=entry_id, revision=1)
            historical_source = await store.get_entry(
                source_ref.entry_id,
                revision=source_ref.revision,
                access_scope=privileged,
            )
            history = await _inspect_exact_lineage(
                source_ref,
                store,
                scope=privileged,
            )
            if not (
                historical_source is not None
                and historical_source.id == source_ref.entry_id
                and historical_source.revision == source_ref.revision
                and historical_source.text == fixtures[source_ref.entry_id].text
                and historical_source.status is KnowledgeStatus.ACTIVE
                and history is not None
                and _lineage_result_matches_expected(
                    history,
                    source_ref,
                    proposal.relations,
                    expected_entries=expected_entries,
                )
            ):
                return entry_ids, False
    return entry_ids, correct


def _expected_lineage_link_models(
    reference: KnowledgeRevisionRef,
    relations: Iterable[KnowledgeRelation],
    *,
    expected_entries: dict[str, KnowledgeEntry],
) -> list[KnowledgeLineageLink] | None:
    forward_roles = {
        KnowledgeRelationKind.SUPERSEDES: KnowledgeLineageRole.SUPERSEDES,
        KnowledgeRelationKind.DERIVED_FROM: KnowledgeLineageRole.DERIVED_FROM,
        KnowledgeRelationKind.CONTRADICTS: KnowledgeLineageRole.CONTRADICTS,
    }
    reverse_roles = {
        KnowledgeRelationKind.SUPERSEDES: KnowledgeLineageRole.SUPERSEDED_BY,
        KnowledgeRelationKind.DERIVED_FROM: KnowledgeLineageRole.DERIVATION_SOURCE_FOR,
        KnowledgeRelationKind.CONTRADICTS: KnowledgeLineageRole.CONTRADICTS,
    }
    reference_current_entry = expected_entries.get(reference.entry_id)
    if reference_current_entry is None:
        return None
    reference_current = KnowledgeRevisionRef(
        entry_id=reference_current_entry.id,
        revision=reference_current_entry.revision,
    )
    expected: list[KnowledgeLineageLink] = []
    for relation in relations:
        if relation.subject == reference:
            counterpart = relation.object
            role = forward_roles[relation.kind]
        elif relation.object == reference:
            counterpart = relation.subject
            role = reverse_roles[relation.kind]
        else:
            continue
        counterpart_current_entry = expected_entries.get(counterpart.entry_id)
        if counterpart_current_entry is None:
            return None
        counterpart_current = KnowledgeRevisionRef(
            entry_id=counterpart_current_entry.id,
            revision=counterpart_current_entry.revision,
        )
        currentness = (
            KnowledgeLineageCurrentness.CURRENT
            if reference == reference_current and counterpart == counterpart_current
            else KnowledgeLineageCurrentness.STALE
        )
        unresolved_contradiction = (
            relation.kind is KnowledgeRelationKind.CONTRADICTS
            and currentness is KnowledgeLineageCurrentness.CURRENT
            and reference_current_entry.status is KnowledgeStatus.ACTIVE
            and counterpart_current_entry.status is KnowledgeStatus.ACTIVE
        )
        expected.append(
            KnowledgeLineageLink(
                relation_id=relation.id,
                kind=relation.kind,
                role=role,
                counterpart=counterpart,
                counterpart_current=counterpart_current,
                counterpart_status=counterpart_current_entry.status,
                currentness=currentness,
                unresolved_contradiction=unresolved_contradiction,
                created_at=relation.created_at,
            )
        )
    return sorted(expected, key=lambda link: (link.created_at, link.relation_id))


def _mean(values) -> float:
    copied = list(values)
    return sum(copied) / len(copied) if copied else 0.0


def _nearest_rank_percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(ceil(percentile * len(ordered)) - 1, 0)]


__all__ = [
    "KNOWLEDGE_MAINTENANCE_EVALUATION_CORPUS_SCHEMA_VERSION",
    "KNOWLEDGE_MAINTENANCE_EVALUATION_RESULT_SCHEMA_VERSION",
    "KnowledgeMaintenanceEvaluationCase",
    "KnowledgeMaintenanceEvaluationCaseResult",
    "KnowledgeMaintenanceEvaluationClaim",
    "KnowledgeMaintenanceEvaluationCorpus",
    "KnowledgeMaintenanceEvaluationDisposition",
    "KnowledgeMaintenanceEvaluationEntry",
    "KnowledgeMaintenanceEvaluationMetrics",
    "KnowledgeMaintenanceEvaluationResult",
    "KnowledgeMaintenanceEvaluationScenario",
    "load_knowledge_maintenance_evaluation_corpus",
    "run_knowledge_maintenance_evaluation",
]
