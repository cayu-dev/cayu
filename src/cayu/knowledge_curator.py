"""Governed, explicitly invoked knowledge curation.

The curator turns bounded application evidence into reviewable knowledge.  It
does not schedule itself, choose a model, activate knowledge, or replace the
knowledge store and review workflow.  Candidate generation, independent
evaluation, application policy, and atomic revision publication remain distinct
boundaries.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from itertools import islice
from typing import Any, Literal, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from cayu._clock import utc_clock
from cayu._knowledge_publication_owner import (
    KnowledgePublicationCapacityExhausted,
    KnowledgePublicationOperationConflict,
    KnowledgePublicationOwnerClosed,
    RetainedKnowledgePublicationOwner,
)
from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_json_object,
    copy_label_map,
    require_durable_clean_nonblank,
    require_durable_nonblank,
    require_finite,
    revalidate_model_input,
    revalidate_model_inputs,
)
from cayu.storage.knowledge_indexer import (
    DEFAULT_KNOWLEDGE_CHUNK_OVERLAP_BYTES,
    DEFAULT_KNOWLEDGE_CHUNK_TARGET_BYTES,
    MIN_KNOWLEDGE_TEXT_BYTES,
    KnowledgeIndexer,
    KnowledgeIndexRequest,
)
from cayu.storage.memory import (
    BUILTIN_KNOWLEDGE_KINDS,
    DEFAULT_KNOWLEDGE_NAMESPACE,
    MAX_KNOWLEDGE_EVIDENCE_JSON_BYTES,
    KnowledgeAccessScope,
    KnowledgeActorType,
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeEvidence,
    KnowledgeEvidenceDisposition,
    KnowledgeEvidenceRole,
    KnowledgePublicationConflict,
    KnowledgePublicationReceipt,
    KnowledgeRevisionConflict,
    KnowledgeStatus,
    KnowledgeVisibility,
    copy_knowledge_access_scope,
    prepare_knowledge_publication,
)

LEARNING_SCHEMA_VERSION = 1
MAX_LEARNING_SIGNALS = 1_000
MAX_LEARNING_CANDIDATES = 250
MAX_LEARNING_SOURCE_REFERENCES = 1_000
MAX_LEARNING_TEXT_BYTES = 512 * 1024
MAX_LEARNING_BATCH_BYTES = 4 * 1024 * 1024
MAX_LEARNING_METADATA_BYTES = 64 * 1024
MAX_LEARNING_NOTES_BYTES = 64 * 1024

_IDENTITY_MAX_BYTES = 256
_SOURCE_URI_MAX_BYTES = 2_048
_CURATOR_METADATA_KEY = "cayu_curator"


class _CuratorModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        validate_default=True,
    )


def _clean(value: str, field_name: str, *, max_bytes: int = _IDENTITY_MAX_BYTES) -> str:
    value = require_durable_clean_nonblank(value, field_name)
    if len(value.encode("utf-8")) > max_bytes:
        raise ValueError(f"`{field_name}` must be at most {max_bytes} UTF-8 bytes.")
    return value


def _bounded_text(value: str, field_name: str, *, max_bytes: int) -> str:
    value = require_durable_nonblank(value, field_name)
    if len(value.encode("utf-8")) > max_bytes:
        raise ValueError(f"`{field_name}` must be at most {max_bytes} UTF-8 bytes.")
    return value


def _bounded_json_object(value: Any, field_name: str, *, max_bytes: int) -> dict[str, Any]:
    copied = copy_durable_json_object(value, field_name)
    if len(canonical_durable_json_bytes(copied, field_name)) > max_bytes:
        raise ValueError(f"`{field_name}` must be at most {max_bytes} canonical UTF-8 bytes.")
    return copied


def _utc(value: datetime, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"`{field_name}` must be a datetime.")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"`{field_name}` must be timezone-aware.")
    return value.astimezone(UTC)


def _sha256(value: object, field_name: str) -> str:
    return sha256(canonical_durable_json_bytes(value, field_name)).hexdigest()


def _ordered_unique_text(
    value: object,
    field_name: str,
    *,
    maximum: int,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        raise ValueError(f"`{field_name}` must be an ordered array.")
    if len(value) > maximum:
        raise ValueError(f"`{field_name}` must contain at most {maximum} values.")
    copied_items: list[str] = []
    for index, item in enumerate(value):
        if type(item) is not str:
            raise ValueError(f"`{field_name}[{index}]` must be a string.")
        copied_items.append(_clean(item, f"{field_name}[{index}]"))
    copied = tuple(copied_items)
    if not allow_empty and not copied:
        raise ValueError(f"`{field_name}` cannot be empty.")
    if len(copied) != len(set(copied)):
        raise ValueError(f"`{field_name}` cannot contain duplicates.")
    return copied


class LearningSourceReference(_CuratorModel):
    """One bounded exact source reference carried into ``KnowledgeEvidence``."""

    schema_version: Literal[1] = LEARNING_SCHEMA_VERSION
    source_type: StrictStr
    source_id: StrictStr | None = None
    source_uri: StrictStr | None = None
    source_revision: StrictStr | None = None
    source_hash: StrictStr | None = None
    locator: dict[str, Any] = Field(default_factory=dict)
    role: KnowledgeEvidenceRole = KnowledgeEvidenceRole.ORIGIN
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_type")
    @classmethod
    def validate_source_type(cls, value: str) -> str:
        return _clean(value, "source_type")

    @field_validator("source_id", "source_revision", "source_hash")
    @classmethod
    def validate_optional_identity(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _clean(value, info.field_name)

    @field_validator("source_uri")
    @classmethod
    def validate_source_uri(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _clean(value, "source_uri", max_bytes=_SOURCE_URI_MAX_BYTES)

    @field_validator("locator", "metadata", mode="before")
    @classmethod
    def copy_json_fields(cls, value: Any, info) -> dict[str, Any]:
        return _bounded_json_object(
            value,
            info.field_name,
            max_bytes=MAX_KNOWLEDGE_EVIDENCE_JSON_BYTES,
        )

    @model_validator(mode="after")
    def validate_exact_source(self) -> LearningSourceReference:
        if self.source_id is None and self.source_uri is None:
            raise ValueError("A learning source requires `source_id` or `source_uri`.")
        if self.source_revision is None and self.source_hash is None:
            raise ValueError("A learning source requires `source_revision` or `source_hash`.")
        return self

    def identity_material(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @property
    def fingerprint(self) -> str:
        return _sha256(self.identity_material(), "learning source reference")


class LearningSignal(_CuratorModel):
    """One deterministic application observation; it is not knowledge."""

    schema_version: Literal[1] = LEARNING_SCHEMA_VERSION
    id: StrictStr
    deduplication_key: StrictStr
    kind: StrictStr
    scope: StrictStr
    summary: StrictStr
    source_references: tuple[LearningSourceReference, ...]
    occurred_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "deduplication_key", "kind", "scope")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _clean(value, info.field_name)

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        return _bounded_text(value, "summary", max_bytes=MAX_LEARNING_TEXT_BYTES)

    @field_validator("source_references", mode="before")
    @classmethod
    def copy_source_references(cls, value: object) -> object:
        return revalidate_model_inputs(
            value,
            LearningSourceReference,
            maximum=MAX_LEARNING_SOURCE_REFERENCES,
            field_name="source_references",
        )

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: Any) -> dict[str, Any]:
        return _bounded_json_object(
            value,
            "metadata",
            max_bytes=MAX_LEARNING_METADATA_BYTES,
        )

    @field_validator("occurred_at")
    @classmethod
    def validate_occurred_at(cls, value: datetime) -> datetime:
        return _utc(value, "occurred_at")

    @model_validator(mode="after")
    def validate_sources(self) -> LearningSignal:
        if not self.source_references:
            raise ValueError("A learning signal requires at least one source reference.")
        fingerprints = tuple(reference.fingerprint for reference in self.source_references)
        if len(fingerprints) != len(set(fingerprints)):
            raise ValueError("A learning signal cannot repeat a source reference.")
        return self

    @property
    def fingerprint(self) -> str:
        return _sha256(self.model_dump(mode="json"), "learning signal")


class LearningBatch(_CuratorModel):
    """One already-grouped, explicitly submitted curation batch."""

    schema_version: Literal[1] = LEARNING_SCHEMA_VERSION
    id: StrictStr
    signals: tuple[LearningSignal, ...]

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _clean(value, "id")

    @field_validator("signals", mode="before")
    @classmethod
    def copy_signals(cls, value: object) -> object:
        return revalidate_model_inputs(
            value,
            LearningSignal,
            maximum=MAX_LEARNING_SIGNALS,
            field_name="signals",
        )

    @model_validator(mode="after")
    def validate_signal_identity(self) -> LearningBatch:
        if not self.signals:
            raise ValueError("A learning batch requires at least one signal.")
        scopes = {signal.scope for signal in self.signals}
        if len(scopes) != 1:
            raise ValueError("A learning batch cannot cross signal scopes.")
        ids = tuple(signal.id for signal in self.signals)
        if len(ids) != len(set(ids)):
            raise ValueError("A learning batch cannot repeat a signal id.")
        keys = tuple(signal.deduplication_key for signal in self.signals)
        if len(keys) != len(set(keys)):
            raise ValueError("A learning batch cannot repeat a signal deduplication key.")
        return self

    @property
    def scope(self) -> str:
        return self.signals[0].scope

    @property
    def fingerprint(self) -> str:
        return _sha256(self.model_dump(mode="json"), "learning batch")


def group_learning_signals(
    signals: Iterable[LearningSignal],
    *,
    max_signals_per_batch: int = 50,
) -> list[LearningBatch]:
    """Group copied signals deterministically by scope and bounded batch size."""

    if isinstance(max_signals_per_batch, bool) or type(max_signals_per_batch) is not int:
        raise TypeError("`max_signals_per_batch` must be an integer.")
    if max_signals_per_batch <= 0 or max_signals_per_batch > MAX_LEARNING_SIGNALS:
        raise ValueError(f"`max_signals_per_batch` must be between 1 and {MAX_LEARNING_SIGNALS}.")
    if isinstance(signals, str | bytes | dict | BaseModel):
        raise TypeError("`signals` must be an iterable of LearningSignal values.")
    copied_input = tuple(islice(signals, MAX_LEARNING_SIGNALS + 1))
    if len(copied_input) > MAX_LEARNING_SIGNALS:
        raise ValueError(f"`signals` must contain at most {MAX_LEARNING_SIGNALS} values.")
    copied: list[LearningSignal] = []
    for value in copied_input:
        if type(value) is not LearningSignal:
            raise TypeError("`signals` must contain exact LearningSignal values.")
        copied.append(LearningSignal.model_validate(value.model_dump(mode="python")))
    scoped_ids = tuple((signal.scope, signal.id) for signal in copied)
    if len(scoped_ids) != len(set(scoped_ids)):
        raise ValueError("`signals` cannot repeat an id within one scope.")
    scoped_keys = tuple((signal.scope, signal.deduplication_key) for signal in copied)
    if len(scoped_keys) != len(set(scoped_keys)):
        raise ValueError("`signals` cannot repeat a deduplication key within one scope.")
    copied.sort(key=lambda signal: (signal.scope, signal.occurred_at, signal.id))
    grouped: dict[str, list[LearningSignal]] = defaultdict(list)
    for signal in copied:
        grouped[signal.scope].append(signal)
    batches: list[LearningBatch] = []
    for scope in sorted(grouped):
        scope_signals = grouped[scope]
        for start in range(0, len(scope_signals), max_signals_per_batch):
            selected = scope_signals[start : start + max_signals_per_batch]
            batch_material = {
                "contract": "cayu.learning-batch-identity.v1",
                "scope": scope,
                "signals": [
                    {"id": signal.id, "deduplication_key": signal.deduplication_key}
                    for signal in selected
                ],
            }
            batches.append(
                LearningBatch(
                    id=f"learning-batch-{_sha256(batch_material, 'learning batch identity')}",
                    signals=tuple(selected),
                )
            )
    return batches


class LearningCandidate(_CuratorModel):
    """One generator-proposed reusable knowledge item."""

    schema_version: Literal[1] = LEARNING_SCHEMA_VERSION
    proposal_key: StrictStr
    text: StrictStr
    signal_ids: tuple[StrictStr, ...]
    title: StrictStr | None = None
    kind: StrictStr = "fact"
    aspects: tuple[StrictStr, ...] = ()
    impact_targets: tuple[StrictStr, ...] = ()
    confidence_hint: StrictFloat | None = None
    source_references: tuple[LearningSourceReference, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("proposal_key", "kind")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _clean(value, info.field_name)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _bounded_text(value, "text", max_bytes=MAX_LEARNING_TEXT_BYTES)

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _bounded_text(value, "title", max_bytes=MAX_LEARNING_TEXT_BYTES)

    @field_validator("signal_ids", mode="before")
    @classmethod
    def copy_signal_ids(cls, value: object) -> tuple[str, ...]:
        return _ordered_unique_text(
            value,
            "signal_ids",
            maximum=MAX_LEARNING_SIGNALS,
            allow_empty=False,
        )

    @field_validator("aspects", "impact_targets", mode="before")
    @classmethod
    def copy_taxonomy(cls, value: object, info) -> tuple[str, ...]:
        return _ordered_unique_text(
            value,
            info.field_name,
            maximum=MAX_LEARNING_SOURCE_REFERENCES,
        )

    @field_validator("source_references", mode="before")
    @classmethod
    def copy_source_references(cls, value: object) -> object:
        return revalidate_model_inputs(
            value,
            LearningSourceReference,
            maximum=MAX_LEARNING_SOURCE_REFERENCES,
            field_name="source_references",
        )

    @field_validator("confidence_hint")
    @classmethod
    def validate_confidence_hint(cls, value: float | None) -> float | None:
        if value is None:
            return None
        value = require_finite(value, "confidence_hint")
        if value < 0.0 or value > 1.0:
            raise ValueError("`confidence_hint` must be between 0.0 and 1.0.")
        return value

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: Any) -> dict[str, Any]:
        return _bounded_json_object(
            value,
            "metadata",
            max_bytes=MAX_LEARNING_METADATA_BYTES,
        )

    @model_validator(mode="after")
    def validate_sources(self) -> LearningCandidate:
        fingerprints = tuple(reference.fingerprint for reference in self.source_references)
        if len(fingerprints) != len(set(fingerprints)):
            raise ValueError("A learning candidate cannot repeat a source reference.")
        return self

    @property
    def fingerprint(self) -> str:
        return _sha256(self.model_dump(mode="json"), "learning candidate")


class LearningVerdict(StrEnum):
    """Independent evaluator verdict for one exact candidate."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"


class LearningDecision(_CuratorModel):
    """One independent evaluator decision for one exact candidate."""

    schema_version: Literal[1] = LEARNING_SCHEMA_VERSION
    verdict: LearningVerdict
    code: StrictStr
    notes: StrictStr | None = None
    confidence: StrictFloat | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("code")
    @classmethod
    def validate_code(cls, value: str) -> str:
        return _clean(value, "code")

    @field_validator("notes")
    @classmethod
    def validate_notes(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _bounded_text(value, "notes", max_bytes=MAX_LEARNING_NOTES_BYTES)

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float | None) -> float | None:
        if value is None:
            return None
        value = require_finite(value, "confidence")
        if value < 0.0 or value > 1.0:
            raise ValueError("`confidence` must be between 0.0 and 1.0.")
        return value

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: Any) -> dict[str, Any]:
        return _bounded_json_object(
            value,
            "metadata",
            max_bytes=MAX_LEARNING_METADATA_BYTES,
        )


class CandidatePolicyDisposition(StrEnum):
    """Application-policy disposition before independent evaluation."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"


class KnowledgeCandidatePolicyDecision(_CuratorModel):
    """Application-owned rejection or bounded transformation result."""

    schema_version: Literal[1] = LEARNING_SCHEMA_VERSION
    disposition: CandidatePolicyDisposition
    code: StrictStr
    notes: StrictStr | None = None
    candidate: LearningCandidate | None = None

    @field_validator("code")
    @classmethod
    def validate_code(cls, value: str) -> str:
        return _clean(value, "code")

    @field_validator("notes")
    @classmethod
    def validate_notes(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _bounded_text(value, "notes", max_bytes=MAX_LEARNING_NOTES_BYTES)

    @field_validator("candidate", mode="before")
    @classmethod
    def copy_candidate(cls, value: object) -> object:
        return revalidate_model_input(value, LearningCandidate)

    @model_validator(mode="after")
    def validate_disposition(self) -> KnowledgeCandidatePolicyDecision:
        if self.disposition is CandidatePolicyDisposition.ACCEPTED and self.candidate is None:
            raise ValueError("An accepted candidate-policy decision requires `candidate`.")
        if self.disposition is CandidatePolicyDisposition.REJECTED and self.candidate is not None:
            raise ValueError("A rejected candidate-policy decision cannot carry `candidate`.")
        return self


class KnowledgeCuratorConfig(_CuratorModel):
    """Application-owned policy and resource bounds for reviewed curation."""

    schema_version: Literal[1] = LEARNING_SCHEMA_VERSION
    candidate_generator_identity: StrictStr
    evaluator_identity: StrictStr
    policy_identity: StrictStr | None = None
    pipeline_version: StrictStr = "cayu.knowledge-curator.v1"
    namespace: StrictStr = DEFAULT_KNOWLEDGE_NAMESPACE
    labels: dict[str, str] = Field(default_factory=dict)
    visibility: KnowledgeVisibility = KnowledgeVisibility.GLOBAL
    allowed_kinds: tuple[StrictStr, ...] = BUILTIN_KNOWLEDGE_KINDS
    created_by: StrictStr = "knowledge_curator"
    max_signals: StrictInt = 50
    max_signal_bytes: StrictInt = 16 * 1024
    max_batch_bytes: StrictInt = 256 * 1024
    max_candidates: StrictInt = 20
    max_candidate_bytes: StrictInt = 64 * 1024
    max_candidate_text_bytes: StrictInt = 32 * 1024
    max_candidate_title_bytes: StrictInt = 4 * 1024
    max_candidate_batch_bytes: StrictInt = 512 * 1024
    max_source_references_per_signal: StrictInt = 20
    max_source_references_per_candidate: StrictInt = 100
    max_source_reference_bytes: StrictInt = 16 * 1024
    max_metadata_bytes: StrictInt = 16 * 1024
    max_evaluator_notes_bytes: StrictInt = 16 * 1024
    max_evaluator_concurrency: StrictInt = 4
    max_in_flight_publications: StrictInt = 100
    candidate_generator_timeout_seconds: StrictFloat = 120.0
    evaluator_timeout_seconds: StrictFloat = 120.0
    candidate_policy_timeout_seconds: StrictFloat = 30.0
    chunk_target_bytes: StrictInt = DEFAULT_KNOWLEDGE_CHUNK_TARGET_BYTES
    chunk_overlap_bytes: StrictInt = DEFAULT_KNOWLEDGE_CHUNK_OVERLAP_BYTES
    max_chunks: StrictInt = 100

    @field_validator(
        "candidate_generator_identity",
        "evaluator_identity",
        "pipeline_version",
        "namespace",
        "created_by",
    )
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _clean(value, info.field_name)

    @field_validator("policy_identity")
    @classmethod
    def validate_policy_identity(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _clean(value, "policy_identity")

    @field_validator("labels", mode="before")
    @classmethod
    def copy_labels(cls, value: Any) -> dict[str, str]:
        return copy_label_map(value, "labels")

    @field_validator("allowed_kinds", mode="before")
    @classmethod
    def copy_allowed_kinds(cls, value: object) -> tuple[str, ...]:
        copied = _ordered_unique_text(
            value,
            "allowed_kinds",
            maximum=len(BUILTIN_KNOWLEDGE_KINDS) * 10,
            allow_empty=False,
        )
        return tuple(sorted(copied))

    @field_validator(
        "max_signals",
        "max_signal_bytes",
        "max_batch_bytes",
        "max_candidates",
        "max_candidate_bytes",
        "max_candidate_text_bytes",
        "max_candidate_title_bytes",
        "max_candidate_batch_bytes",
        "max_source_references_per_signal",
        "max_source_references_per_candidate",
        "max_source_reference_bytes",
        "max_metadata_bytes",
        "max_evaluator_notes_bytes",
        "max_evaluator_concurrency",
        "max_in_flight_publications",
        "max_chunks",
    )
    @classmethod
    def validate_positive_int(cls, value: int, info) -> int:
        if value <= 0:
            raise ValueError(f"`{info.field_name}` must be greater than 0.")
        return value

    @field_validator("max_candidate_text_bytes", "chunk_target_bytes")
    @classmethod
    def validate_text_capacity(cls, value: int, info) -> int:
        if value < MIN_KNOWLEDGE_TEXT_BYTES:
            raise ValueError(f"`{info.field_name}` must be at least {MIN_KNOWLEDGE_TEXT_BYTES}.")
        return value

    @field_validator("chunk_overlap_bytes")
    @classmethod
    def validate_nonnegative_overlap(cls, value: int) -> int:
        if value < 0:
            raise ValueError("`chunk_overlap_bytes` must be greater than or equal to 0.")
        return value

    @field_validator(
        "candidate_generator_timeout_seconds",
        "evaluator_timeout_seconds",
        "candidate_policy_timeout_seconds",
    )
    @classmethod
    def validate_timeout(cls, value: float, info) -> float:
        value = require_finite(value, info.field_name)
        if value <= 0.0 or value > 3_600.0:
            raise ValueError(f"`{info.field_name}` must be between 0 and 3600 seconds.")
        return value

    @model_validator(mode="after")
    def validate_bounds(self) -> KnowledgeCuratorConfig:
        ceilings = {
            "max_signals": MAX_LEARNING_SIGNALS,
            "max_signal_bytes": MAX_LEARNING_TEXT_BYTES,
            "max_batch_bytes": MAX_LEARNING_BATCH_BYTES,
            "max_candidates": MAX_LEARNING_CANDIDATES,
            "max_candidate_bytes": MAX_LEARNING_TEXT_BYTES,
            "max_candidate_text_bytes": MAX_LEARNING_TEXT_BYTES,
            "max_candidate_title_bytes": MAX_LEARNING_TEXT_BYTES,
            "max_candidate_batch_bytes": MAX_LEARNING_BATCH_BYTES,
            "max_source_references_per_signal": MAX_LEARNING_SOURCE_REFERENCES,
            "max_source_references_per_candidate": MAX_LEARNING_SOURCE_REFERENCES,
            "max_source_reference_bytes": MAX_LEARNING_TEXT_BYTES,
            "max_metadata_bytes": MAX_LEARNING_METADATA_BYTES,
            "max_evaluator_notes_bytes": MAX_LEARNING_NOTES_BYTES,
            "max_evaluator_concurrency": MAX_LEARNING_CANDIDATES,
            "max_in_flight_publications": MAX_LEARNING_SIGNALS,
            "chunk_target_bytes": MAX_LEARNING_TEXT_BYTES,
            "chunk_overlap_bytes": MAX_LEARNING_TEXT_BYTES,
            "max_chunks": MAX_LEARNING_SOURCE_REFERENCES,
        }
        for field_name, maximum in ceilings.items():
            if getattr(self, field_name) > maximum:
                raise ValueError(f"`{field_name}` must be at most {maximum}.")
        if self.max_candidate_text_bytes > self.max_candidate_bytes:
            raise ValueError("`max_candidate_text_bytes` cannot exceed `max_candidate_bytes`.")
        if self.max_candidate_title_bytes > self.max_candidate_bytes:
            raise ValueError("`max_candidate_title_bytes` cannot exceed `max_candidate_bytes`.")
        if self.chunk_overlap_bytes >= self.chunk_target_bytes:
            raise ValueError("`chunk_overlap_bytes` must be less than `chunk_target_bytes`.")
        if self.chunk_overlap_bytes > self.chunk_target_bytes // 2:
            raise ValueError("`chunk_overlap_bytes` must be at most half `chunk_target_bytes`.")
        return self

    @property
    def fingerprint(self) -> str:
        return _sha256(self.model_dump(mode="json"), "knowledge curator configuration")


class LearningCandidateOutcome(StrEnum):
    """Durable or fail-closed outcome for one generated candidate."""

    PENDING_PERSISTED = "pending_persisted"
    EXISTING_PENDING = "existing_pending"
    EXISTING_ACTIVE = "existing_active"
    EXISTING_ARCHIVED = "existing_archived"
    EXISTING_DELETED = "existing_deleted"
    EVALUATOR_REJECTED = "evaluator_rejected"
    POLICY_REJECTED = "policy_rejected"
    INVALID = "invalid"
    FAILED = "failed"
    CONFLICT = "conflict"


class LearningBatchOutcome(StrEnum):
    """Outcome of the batch-wide candidate-generation boundary."""

    COMPLETED = "completed"
    GENERATOR_FAILED = "generator_failed"
    GENERATOR_TIMED_OUT = "generator_timed_out"
    GENERATOR_INVALID = "generator_invalid"


class LearningSignalOutcome(StrEnum):
    """Participation outcome for one submitted signal."""

    CANDIDATE_GENERATED = "candidate_generated"
    NO_CANDIDATE_GENERATED = "no_candidate_generated"
    BATCH_FAILED = "batch_failed"


_ENTRY_STATUS_BY_CANDIDATE_OUTCOME = {
    LearningCandidateOutcome.PENDING_PERSISTED: KnowledgeStatus.PENDING,
    LearningCandidateOutcome.EXISTING_PENDING: KnowledgeStatus.PENDING,
    LearningCandidateOutcome.EXISTING_ACTIVE: KnowledgeStatus.ACTIVE,
    LearningCandidateOutcome.EXISTING_ARCHIVED: KnowledgeStatus.ARCHIVED,
    LearningCandidateOutcome.EXISTING_DELETED: KnowledgeStatus.DELETED,
}


class LearningSignalResult(_CuratorModel):
    """How one submitted signal participated in an explicit curation call."""

    schema_version: Literal[1] = LEARNING_SCHEMA_VERSION
    signal_id: StrictStr
    signal_fingerprint: StrictStr
    outcome: LearningSignalOutcome
    code: StrictStr
    candidate_proposal_keys: tuple[StrictStr, ...] = ()

    @field_validator("signal_id", "code")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _clean(value, info.field_name)

    @field_validator("signal_fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("`signal_fingerprint` must be a lowercase SHA-256 digest.")
        return value

    @field_validator("candidate_proposal_keys", mode="before")
    @classmethod
    def copy_candidate_proposal_keys(cls, value: object) -> tuple[str, ...]:
        return _ordered_unique_text(
            value,
            "candidate_proposal_keys",
            maximum=MAX_LEARNING_CANDIDATES,
        )

    @model_validator(mode="after")
    def validate_outcome(self) -> LearningSignalResult:
        has_candidates = bool(self.candidate_proposal_keys)
        if self.outcome is LearningSignalOutcome.CANDIDATE_GENERATED and not has_candidates:
            raise ValueError("A generated signal result requires a candidate proposal key.")
        if self.outcome is not LearningSignalOutcome.CANDIDATE_GENERATED and has_candidates:
            raise ValueError("Only a generated signal result can carry candidate proposal keys.")
        return self


class LearningCandidateResult(_CuratorModel):
    """Bounded public result for one generated candidate."""

    schema_version: Literal[1] = LEARNING_SCHEMA_VERSION
    proposal_key: StrictStr
    candidate_fingerprint: StrictStr
    outcome: LearningCandidateOutcome
    code: StrictStr
    decision: LearningDecision | None = None
    entry_id: StrictStr | None = None
    entry_revision: StrictInt | None = None
    entry_status: KnowledgeStatus | None = None
    warning_code: StrictStr | None = None

    @field_validator("proposal_key", "code")
    @classmethod
    def validate_required_identity(cls, value: str, info) -> str:
        return _clean(value, info.field_name)

    @field_validator("candidate_fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("`candidate_fingerprint` must be a lowercase SHA-256 digest.")
        return value

    @field_validator("decision", mode="before")
    @classmethod
    def copy_decision(cls, value: object) -> object:
        return revalidate_model_input(value, LearningDecision)

    @field_validator("entry_id", "warning_code")
    @classmethod
    def validate_optional_identity(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _clean(value, info.field_name)

    @field_validator("entry_revision")
    @classmethod
    def validate_revision(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("`entry_revision` must be greater than 0.")
        return value

    @model_validator(mode="after")
    def validate_outcome_contract(self) -> LearningCandidateResult:
        entry_values = (self.entry_id, self.entry_revision, self.entry_status)
        has_any_entry_value = any(value is not None for value in entry_values)
        has_complete_entry = all(value is not None for value in entry_values)
        if has_any_entry_value and not has_complete_entry:
            raise ValueError("Entry result identity, revision, and status must appear together.")
        expected_status = _ENTRY_STATUS_BY_CANDIDATE_OUTCOME.get(self.outcome)
        if expected_status is None:
            if has_complete_entry:
                raise ValueError("Only durable-entry outcomes can carry an entry projection.")
        else:
            if not has_complete_entry:
                raise ValueError("A durable-entry outcome requires an entry projection.")
            if self.entry_status is not expected_status:
                raise ValueError("Candidate outcome does not match the projected entry status.")

        if self.outcome is LearningCandidateOutcome.EVALUATOR_REJECTED:
            if self.decision is None or self.decision.verdict is not LearningVerdict.REJECTED:
                raise ValueError("An evaluator-rejected outcome requires a rejected decision.")
        elif self.outcome is LearningCandidateOutcome.POLICY_REJECTED:
            if self.decision is not None:
                raise ValueError("A policy-rejected outcome cannot carry an evaluator decision.")
        elif self.decision is not None and self.decision.verdict is not LearningVerdict.ACCEPTED:
            raise ValueError("Only an evaluator-rejected outcome can carry a rejected decision.")

        if self.outcome is LearningCandidateOutcome.PENDING_PERSISTED and self.decision is None:
            raise ValueError("A persisted pending outcome requires an accepted decision.")
        if (
            self.warning_code is not None
            and self.outcome is not LearningCandidateOutcome.PENDING_PERSISTED
        ):
            raise ValueError("Only a persisted pending outcome can carry a warning code.")
        return self


class LearningBatchResult(_CuratorModel):
    """Bounded public result for one explicit curation invocation."""

    schema_version: Literal[1] = LEARNING_SCHEMA_VERSION
    batch_id: StrictStr
    batch_fingerprint: StrictStr
    configuration_fingerprint: StrictStr
    scope: StrictStr
    outcome: LearningBatchOutcome
    code: StrictStr
    signal_count: StrictInt
    candidate_count: StrictInt
    signals: tuple[LearningSignalResult, ...]
    candidates: tuple[LearningCandidateResult, ...] = ()
    processed_at: datetime

    @field_validator("batch_id", "scope", "code")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _clean(value, info.field_name)

    @field_validator("batch_fingerprint", "configuration_fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str, info) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"`{info.field_name}` must be a lowercase SHA-256 digest.")
        return value

    @field_validator("candidates", mode="before")
    @classmethod
    def copy_candidates(cls, value: object) -> object:
        return revalidate_model_inputs(
            value,
            LearningCandidateResult,
            maximum=MAX_LEARNING_CANDIDATES,
            field_name="candidates",
        )

    @field_validator("signals", mode="before")
    @classmethod
    def copy_signals(cls, value: object) -> object:
        return revalidate_model_inputs(
            value,
            LearningSignalResult,
            maximum=MAX_LEARNING_SIGNALS,
            field_name="signals",
        )

    @field_validator("processed_at")
    @classmethod
    def validate_processed_at(cls, value: datetime) -> datetime:
        return _utc(value, "processed_at")

    @field_validator("signal_count", "candidate_count")
    @classmethod
    def validate_count(cls, value: int, info) -> int:
        if value < 0:
            raise ValueError(f"`{info.field_name}` must be greater than or equal to 0.")
        return value

    @model_validator(mode="after")
    def validate_result_contract(self) -> LearningBatchResult:
        if self.signal_count != len(self.signals):
            raise ValueError("`signal_count` must match `signals` length.")
        if self.candidate_count != len(self.candidates):
            raise ValueError("`candidate_count` must match `candidates` length.")
        if not self.signals:
            raise ValueError("A learning batch result requires at least one signal result.")
        signal_ids = tuple(signal.signal_id for signal in self.signals)
        if len(signal_ids) != len(set(signal_ids)):
            raise ValueError("A learning batch result cannot repeat a signal id.")
        proposal_keys = tuple(candidate.proposal_key for candidate in self.candidates)
        if len(proposal_keys) != len(set(proposal_keys)):
            raise ValueError("A learning batch result cannot repeat a candidate proposal key.")
        if self.outcome is not LearningBatchOutcome.COMPLETED and self.candidates:
            raise ValueError("A failed generator batch cannot contain candidate results.")
        if self.outcome is LearningBatchOutcome.COMPLETED and any(
            signal.outcome is LearningSignalOutcome.BATCH_FAILED for signal in self.signals
        ):
            raise ValueError("A completed batch cannot contain failed signal results.")
        if self.outcome is not LearningBatchOutcome.COMPLETED and any(
            signal.outcome is not LearningSignalOutcome.BATCH_FAILED for signal in self.signals
        ):
            raise ValueError("A failed batch requires failed signal results.")
        if self.outcome is LearningBatchOutcome.COMPLETED:
            referenced_proposal_keys = {
                proposal_key
                for signal in self.signals
                for proposal_key in signal.candidate_proposal_keys
            }
            if referenced_proposal_keys != set(proposal_keys):
                raise ValueError(
                    "Completed signal results must reference exactly the returned candidates."
                )
        return self


class KnowledgeCandidateGenerator(Protocol):
    """Provider-neutral candidate generation over one copied bounded batch."""

    async def generate_candidates(self, batch: LearningBatch) -> list[LearningCandidate]: ...


class LearningEvaluator(Protocol):
    """Independent evaluation of one candidate against its copied signals."""

    async def evaluate_candidate(
        self,
        candidate: LearningCandidate,
        signals: tuple[LearningSignal, ...],
    ) -> LearningDecision: ...


class KnowledgeCandidatePolicy(Protocol):
    """Optional application-owned content rejection or transformation hook."""

    async def apply_candidate_policy(
        self,
        candidate: LearningCandidate,
        signals: tuple[LearningSignal, ...],
    ) -> KnowledgeCandidatePolicyDecision: ...


class _CuratorStore(Protocol):
    def bound_access_scope(self) -> KnowledgeAccessScope | None: ...

    async def get_entry(
        self,
        entry_id: str,
        *,
        revision: int | None = None,
        max_bytes: int | None = None,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeEntry | None: ...

    async def publish_entry_revision(
        self,
        entry: KnowledgeEntry,
        chunks: list[KnowledgeChunk],
        *,
        evidence: list[KnowledgeEvidence] | None = None,
        access_scope: KnowledgeAccessScope | None = None,
        operation_id: str,
        expected_revision: int | None = None,
    ) -> KnowledgePublicationReceipt: ...

    async def load_entry_publication_receipt(
        self,
        operation_id: str,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgePublicationReceipt | None: ...


class _EvaluationState(_CuratorModel):
    candidate: LearningCandidate
    result: LearningCandidateResult | None = None
    decision: LearningDecision | None = None


@dataclass(frozen=True)
class _PublicationExpectation:
    operation_id: str
    entry_id: str
    entry_revision: int
    expected_revision: int | None
    request_sha256: str
    entry_created_at: datetime
    entry_updated_at: datetime


class _PublicationCapacityExhausted(RuntimeError):
    pass


class KnowledgeCurator:
    """Explicit reviewed-mode knowledge curation over Cayu's revision store."""

    def __init__(
        self,
        store: _CuratorStore,
        *,
        candidate_generator: KnowledgeCandidateGenerator,
        evaluator: LearningEvaluator,
        config: KnowledgeCuratorConfig,
        access_scope: KnowledgeAccessScope | None = None,
        candidate_policy: KnowledgeCandidatePolicy | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        _validate_curator_store(store)
        if candidate_generator is evaluator:
            raise ValueError("Candidate generator and evaluator must be separate components.")
        generate = getattr(candidate_generator, "generate_candidates", None)
        evaluate = getattr(evaluator, "evaluate_candidate", None)
        if not callable(generate):
            raise TypeError("candidate_generator must implement `generate_candidates`.")
        if not callable(evaluate):
            raise TypeError("evaluator must implement `evaluate_candidate`.")
        if type(config) is not KnowledgeCuratorConfig:
            raise TypeError("config must be a KnowledgeCuratorConfig.")
        copied_config = KnowledgeCuratorConfig.model_validate(config.model_dump(mode="python"))
        apply_policy = None
        if candidate_policy is not None:
            apply_policy = getattr(candidate_policy, "apply_candidate_policy", None)
            if not callable(apply_policy):
                raise TypeError("candidate_policy must implement `apply_candidate_policy`.")
            if copied_config.policy_identity is None:
                raise ValueError("A candidate policy requires an explicit `policy_identity`.")
        elif copied_config.policy_identity is not None:
            raise ValueError("`policy_identity` requires a candidate policy.")
        if access_scope is None:
            access_scope = store.bound_access_scope()
        if access_scope is None:
            raise ValueError("KnowledgeCurator requires an explicit knowledge access scope.")
        self._store = store
        self._generate = generate
        self._evaluate = evaluate
        self._apply_policy = apply_policy
        self._config = copied_config
        self._access_scope = copy_knowledge_access_scope(access_scope)
        _validate_curator_access_scope(self._config, self._access_scope)
        self._clock = utc_clock(clock)
        self._evaluator_semaphore = asyncio.Semaphore(self._config.max_evaluator_concurrency)
        self._publication_owner = RetainedKnowledgePublicationOwner[KnowledgePublicationReceipt](
            max_publications=self._config.max_in_flight_publications
        )

    @property
    def config(self) -> KnowledgeCuratorConfig:
        return KnowledgeCuratorConfig.model_validate(self._config.model_dump(mode="python"))

    async def aclose(self, *, timeout_s: float = 10.0) -> bool:
        """Seal and drain publications retained beyond their curation callers."""

        return await self._publication_owner.aclose(timeout_s=timeout_s)

    async def __aenter__(self) -> KnowledgeCurator:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()

    async def curate(self, batch: LearningBatch) -> LearningBatchResult:
        """Generate, independently evaluate, and persist pending knowledge."""

        if type(batch) is not LearningBatch:
            raise TypeError("batch must be a LearningBatch.")
        copied_batch = LearningBatch.model_validate(batch.model_dump(mode="python"))
        processed_at = self._clock()
        self._validate_batch_bounds(copied_batch)
        try:
            async with asyncio.timeout(self._config.candidate_generator_timeout_seconds):
                raw_candidates = await self._generate(
                    LearningBatch.model_validate(copied_batch.model_dump(mode="python"))
                )
        except TimeoutError:
            return self._batch_failure(
                copied_batch,
                processed_at=processed_at,
                outcome=LearningBatchOutcome.GENERATOR_TIMED_OUT,
                code="candidate_generator_timed_out",
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            return self._batch_failure(
                copied_batch,
                processed_at=processed_at,
                outcome=LearningBatchOutcome.GENERATOR_FAILED,
                code="candidate_generator_failed",
            )
        try:
            candidates = self._validate_generated_candidates(copied_batch, raw_candidates)
        except (TypeError, ValueError):
            return self._batch_failure(
                copied_batch,
                processed_at=processed_at,
                outcome=LearningBatchOutcome.GENERATOR_INVALID,
                code="candidate_generator_output_invalid",
            )
        evaluated = await self._evaluate_candidates(copied_batch, candidates)
        results: list[LearningCandidateResult] = []
        for state in evaluated:
            if state.result is not None:
                results.append(state.result)
                continue
            if state.decision is None:  # pragma: no cover - internal invariant
                raise RuntimeError("Accepted curation candidate is missing its decision.")
            results.append(
                await self._persist_candidate(
                    copied_batch,
                    state.candidate,
                    state.decision,
                    processed_at=processed_at,
                )
            )
        return LearningBatchResult(
            batch_id=copied_batch.id,
            batch_fingerprint=copied_batch.fingerprint,
            configuration_fingerprint=self._config.fingerprint,
            scope=copied_batch.scope,
            outcome=LearningBatchOutcome.COMPLETED,
            code="completed",
            signal_count=len(copied_batch.signals),
            candidate_count=len(results),
            signals=_completed_signal_results(copied_batch, candidates),
            candidates=tuple(results),
            processed_at=processed_at,
        )

    def _validate_batch_bounds(self, batch: LearningBatch) -> None:
        if len(batch.signals) > self._config.max_signals:
            raise ValueError(f"Learning batch exceeds max_signals={self._config.max_signals}.")
        batch_bytes = len(canonical_durable_json_bytes(batch.model_dump(mode="json"), "batch"))
        if batch_bytes > self._config.max_batch_bytes:
            raise ValueError("Learning batch exceeds its configured byte limit.")
        for signal in batch.signals:
            signal_bytes = len(
                canonical_durable_json_bytes(signal.model_dump(mode="json"), "signal")
            )
            if signal_bytes > self._config.max_signal_bytes:
                raise ValueError("Learning signal exceeds its configured byte limit.")
            if len(signal.source_references) > self._config.max_source_references_per_signal:
                raise ValueError("Learning signal exceeds its source-reference limit.")
            for reference in signal.source_references:
                self._validate_source_reference_bounds(reference)
            if (
                len(canonical_durable_json_bytes(signal.metadata, "signal metadata"))
                > self._config.max_metadata_bytes
            ):
                raise ValueError("Learning signal metadata exceeds its configured byte limit.")

    def _validate_generated_candidates(
        self,
        batch: LearningBatch,
        raw_candidates: object,
    ) -> tuple[LearningCandidate, ...]:
        if type(raw_candidates) is not list:
            raise TypeError("Candidate generator must return a list.")
        if len(raw_candidates) > self._config.max_candidates:
            raise ValueError("Candidate generator exceeded its candidate limit.")
        candidates: list[LearningCandidate] = []
        batch_signal_ids = {signal.id for signal in batch.signals}
        signal_map = {signal.id: signal for signal in batch.signals}
        total_bytes = 0
        for raw_candidate in raw_candidates:
            if type(raw_candidate) is not LearningCandidate:
                raise TypeError("Candidate generator returned a non-candidate value.")
            candidate = LearningCandidate.model_validate(raw_candidate.model_dump(mode="python"))
            if not set(candidate.signal_ids).issubset(batch_signal_ids):
                raise ValueError("Candidate references a signal outside the submitted batch.")
            self._validate_candidate_bounds(candidate)
            self._validate_complete_source_bounds(
                candidate,
                tuple(signal_map[signal_id] for signal_id in candidate.signal_ids),
            )
            primary_source = signal_map[candidate.signal_ids[0]].source_references[0]
            self._validate_primary_source_access(primary_source)
            candidate_bytes = len(
                canonical_durable_json_bytes(candidate.model_dump(mode="json"), "candidate")
            )
            total_bytes += candidate_bytes
            if total_bytes > self._config.max_candidate_batch_bytes:
                raise ValueError("Candidate generator exceeded its total byte limit.")
            candidates.append(candidate)
        proposal_keys = tuple(candidate.proposal_key for candidate in candidates)
        if len(proposal_keys) != len(set(proposal_keys)):
            raise ValueError("Candidate generator repeated a proposal key.")
        return tuple(candidates)

    def _validate_candidate_bounds(self, candidate: LearningCandidate) -> None:
        candidate_bytes = len(
            canonical_durable_json_bytes(candidate.model_dump(mode="json"), "candidate")
        )
        if candidate_bytes > self._config.max_candidate_bytes:
            raise ValueError("Learning candidate exceeds its configured byte limit.")
        if len(candidate.text.encode("utf-8")) > self._config.max_candidate_text_bytes:
            raise ValueError("Learning candidate text exceeds its configured byte limit.")
        if (
            candidate.title is not None
            and len(candidate.title.encode("utf-8")) > self._config.max_candidate_title_bytes
        ):
            raise ValueError("Learning candidate title exceeds its configured byte limit.")
        if len(candidate.source_references) > self._config.max_source_references_per_candidate:
            raise ValueError("Learning candidate exceeds its source-reference limit.")
        for reference in candidate.source_references:
            self._validate_source_reference_bounds(reference)
        if candidate.kind not in self._config.allowed_kinds:
            raise ValueError("Learning candidate kind is not allowed by application policy.")
        if (
            len(canonical_durable_json_bytes(candidate.metadata, "candidate metadata"))
            > self._config.max_metadata_bytes
        ):
            raise ValueError("Learning candidate metadata exceeds its configured byte limit.")

    def _validate_source_reference_bounds(self, reference: LearningSourceReference) -> None:
        reference_bytes = len(
            canonical_durable_json_bytes(reference.model_dump(mode="json"), "source reference")
        )
        if reference_bytes > self._config.max_source_reference_bytes:
            raise ValueError("Learning source reference exceeds its configured byte limit.")
        for field_name, value in (("locator", reference.locator), ("metadata", reference.metadata)):
            if (
                len(canonical_durable_json_bytes(value, f"source reference {field_name}"))
                > self._config.max_metadata_bytes
            ):
                raise ValueError(
                    f"Learning source reference {field_name} exceeds its configured byte limit."
                )

    def _validate_primary_source_access(self, reference: LearningSourceReference) -> None:
        allowed_source_types = self._access_scope.allowed_source_types
        if allowed_source_types is not None and reference.source_type not in allowed_source_types:
            raise ValueError("Learning candidate primary source type is outside access scope.")
        allowed_source_ids = self._access_scope.allowed_source_ids
        if allowed_source_ids is not None and reference.source_id not in allowed_source_ids:
            raise ValueError("Learning candidate primary source id is outside access scope.")

    def _validate_complete_source_bounds(
        self,
        candidate: LearningCandidate,
        signals: tuple[LearningSignal, ...],
    ) -> None:
        sources = _candidate_sources(candidate, signals)
        if len(sources) > self._config.max_source_references_per_candidate:
            raise ValueError("Learning candidate exceeds its complete source-reference limit.")
        for ordinal, (signal_id, deduplication_key, reference) in enumerate(sources):
            _curation_evidence_metadata(
                signal_id,
                deduplication_key,
                reference,
                ordinal=ordinal,
            )

    async def _evaluate_candidates(
        self,
        batch: LearningBatch,
        candidates: tuple[LearningCandidate, ...],
    ) -> tuple[_EvaluationState, ...]:
        async def evaluate_one(candidate: LearningCandidate) -> _EvaluationState:
            async with self._evaluator_semaphore:
                return await self._evaluate_candidate(batch, candidate)

        return tuple(await asyncio.gather(*(evaluate_one(candidate) for candidate in candidates)))

    async def _evaluate_candidate(
        self,
        batch: LearningBatch,
        candidate: LearningCandidate,
    ) -> _EvaluationState:
        signal_map = {signal.id: signal for signal in batch.signals}
        signals = tuple(signal_map[signal_id] for signal_id in candidate.signal_ids)
        current = candidate
        if self._apply_policy is not None:
            try:
                async with asyncio.timeout(self._config.candidate_policy_timeout_seconds):
                    raw_policy = await self._apply_policy(
                        LearningCandidate.model_validate(current.model_dump(mode="python")),
                        tuple(
                            LearningSignal.model_validate(signal.model_dump(mode="python"))
                            for signal in signals
                        ),
                    )
                if type(raw_policy) is not KnowledgeCandidatePolicyDecision:
                    raise TypeError("Candidate policy returned an invalid decision.")
                policy = KnowledgeCandidatePolicyDecision.model_validate(
                    raw_policy.model_dump(mode="python")
                )
            except TimeoutError:
                return _failed_evaluation(current, "candidate_policy_timed_out")
            except asyncio.CancelledError:
                raise
            except Exception:
                return _failed_evaluation(current, "candidate_policy_failed")
            if policy.disposition is CandidatePolicyDisposition.REJECTED:
                return _EvaluationState(
                    candidate=current,
                    result=_candidate_result(
                        current,
                        outcome=LearningCandidateOutcome.POLICY_REJECTED,
                        code=policy.code,
                    ),
                )
            transformed = policy.candidate
            if transformed is None:  # pragma: no cover - model invariant
                return _failed_evaluation(current, "candidate_policy_invalid")
            if (
                transformed.proposal_key != current.proposal_key
                or transformed.signal_ids != current.signal_ids
            ):
                return _failed_evaluation(current, "candidate_policy_identity_changed")
            try:
                self._validate_candidate_bounds(transformed)
                self._validate_complete_source_bounds(transformed, signals)
            except (TypeError, ValueError):
                return _failed_evaluation(current, "candidate_policy_output_invalid")
            current = transformed
        existing = await self._existing_before_evaluation(current, signals)
        if existing is not None:
            return _EvaluationState(candidate=current, result=existing)
        try:
            async with asyncio.timeout(self._config.evaluator_timeout_seconds):
                raw_decision = await self._evaluate(
                    LearningCandidate.model_validate(current.model_dump(mode="python")),
                    tuple(
                        LearningSignal.model_validate(signal.model_dump(mode="python"))
                        for signal in signals
                    ),
                )
            if type(raw_decision) is not LearningDecision:
                raise TypeError("Evaluator returned an invalid decision.")
            decision = LearningDecision.model_validate(raw_decision.model_dump(mode="python"))
            if (
                decision.notes is not None
                and len(decision.notes.encode("utf-8")) > self._config.max_evaluator_notes_bytes
            ):
                raise ValueError("Evaluator notes exceed their configured byte limit.")
            if (
                len(canonical_durable_json_bytes(decision.metadata, "decision metadata"))
                > self._config.max_metadata_bytes
            ):
                raise ValueError("Evaluator metadata exceeds its configured byte limit.")
        except TimeoutError:
            return _failed_evaluation(current, "evaluator_timed_out")
        except asyncio.CancelledError:
            raise
        except Exception:
            return _failed_evaluation(current, "evaluator_failed")
        if decision.verdict is LearningVerdict.REJECTED:
            return _EvaluationState(
                candidate=current,
                decision=decision,
                result=_candidate_result(
                    current,
                    outcome=LearningCandidateOutcome.EVALUATOR_REJECTED,
                    code=decision.code,
                    decision=decision,
                ),
            )
        return _EvaluationState(candidate=current, decision=decision)

    async def _existing_before_evaluation(
        self,
        candidate: LearningCandidate,
        signals: tuple[LearningSignal, ...],
    ) -> LearningCandidateResult | None:
        operation_id = _curation_operation_id(candidate.proposal_key, self._config)
        entry_id = _curated_entry_id(candidate.proposal_key, self._config)
        input_fingerprint = _candidate_input_fingerprint(
            candidate,
            signals=signals,
            config=self._config,
        )
        try:
            receipt = await self._store.load_entry_publication_receipt(
                operation_id,
                access_scope=self._access_scope,
            )
        except asyncio.CancelledError:
            raise
        except NotImplementedError:
            return _candidate_result(
                candidate,
                outcome=LearningCandidateOutcome.FAILED,
                code="owned_publication_unsupported",
            )
        except Exception:
            return _candidate_result(
                candidate,
                outcome=LearningCandidateOutcome.FAILED,
                code="publication_receipt_read_failed",
            )
        if receipt is None:
            return None
        try:
            receipt = _copy_publication_receipt(receipt)
        except (TypeError, ValueError):
            return _candidate_result(
                candidate,
                outcome=LearningCandidateOutcome.FAILED,
                code="publication_receipt_invalid",
            )
        if not _publication_receipt_has_creation_identity(
            receipt,
            operation_id=operation_id,
            entry_id=entry_id,
        ):
            return _candidate_result(
                candidate,
                outcome=LearningCandidateOutcome.CONFLICT,
                code="publication_receipt_identity_conflict",
            )
        try:
            historical = await self._store.get_entry(
                receipt.entry_id,
                revision=receipt.entry_revision,
                access_scope=self._access_scope,
            )
            current = await self._store.get_entry(
                receipt.entry_id,
                access_scope=self._access_scope,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            return _candidate_result(
                candidate,
                outcome=LearningCandidateOutcome.FAILED,
                code="published_entry_read_failed",
            )
        if historical is None or current is None:
            return _candidate_result(
                candidate,
                outcome=LearningCandidateOutcome.FAILED,
                code="published_entry_unavailable",
            )
        audit = historical.metadata.get(_CURATOR_METADATA_KEY)
        if (
            type(audit) is not dict
            or audit.get("candidate_input_fingerprint") != input_fingerprint
            or historical.namespace != self._config.namespace
            or historical.labels != self._config.labels
            or historical.visibility is not self._config.visibility
            or historical.status is not KnowledgeStatus.PENDING
        ):
            return _candidate_result(
                candidate,
                outcome=LearningCandidateOutcome.CONFLICT,
                code="published_proposal_conflict",
            )
        outcome = _existing_candidate_outcome(current.status)
        return _candidate_result(
            candidate,
            outcome=outcome,
            code=outcome.value,
            entry=current,
        )

    async def _persist_candidate(
        self,
        batch: LearningBatch,
        candidate: LearningCandidate,
        decision: LearningDecision,
        *,
        processed_at: datetime,
    ) -> LearningCandidateResult:
        signal_map = {signal.id: signal for signal in batch.signals}
        signals = tuple(signal_map[signal_id] for signal_id in candidate.signal_ids)
        proposal_fingerprint = _proposal_fingerprint(
            candidate,
            decision,
            signals=signals,
            config=self._config,
        )
        entry_id = _curated_entry_id(candidate.proposal_key, self._config)
        operation_id = _curation_operation_id(candidate.proposal_key, self._config)
        try:
            prior_receipt = await self._store.load_entry_publication_receipt(
                operation_id,
                access_scope=self._access_scope,
            )
        except asyncio.CancelledError:
            raise
        except NotImplementedError:
            return _candidate_result(
                candidate,
                outcome=LearningCandidateOutcome.FAILED,
                code="owned_publication_unsupported",
                decision=decision,
            )
        except Exception:
            return _candidate_result(
                candidate,
                outcome=LearningCandidateOutcome.FAILED,
                code="publication_receipt_read_failed",
                decision=decision,
            )
        if prior_receipt is not None:
            try:
                prior_receipt = _copy_publication_receipt(prior_receipt)
            except (TypeError, ValueError):
                return _candidate_result(
                    candidate,
                    outcome=LearningCandidateOutcome.FAILED,
                    code="publication_receipt_invalid",
                    decision=decision,
                )
            reconciled = await self._reconcile_existing_candidate(
                candidate,
                receipt=prior_receipt,
                expected_operation_id=operation_id,
                expected_entry_id=entry_id,
                proposal_fingerprint=proposal_fingerprint,
            )
            return reconciled.model_copy(update={"decision": decision})
        try:
            existing = await self._store.get_entry(entry_id, access_scope=self._access_scope)
        except asyncio.CancelledError:
            raise
        except Exception:
            return _candidate_result(
                candidate,
                outcome=LearningCandidateOutcome.FAILED,
                code="existing_entry_read_failed",
                decision=decision,
            )
        if existing is not None:
            return _candidate_result(
                candidate,
                outcome=LearningCandidateOutcome.CONFLICT,
                code="entry_identity_occupied_without_receipt",
                decision=decision,
            )
        try:
            indexed, evidence = _build_pending_publication(
                batch,
                candidate,
                decision,
                signals=signals,
                config=self._config,
                entry_id=entry_id,
                proposal_fingerprint=proposal_fingerprint,
                processed_at=processed_at,
            )
        except (TypeError, ValueError):
            return _candidate_result(
                candidate,
                outcome=LearningCandidateOutcome.INVALID,
                code="candidate_publication_invalid",
                decision=decision,
            )
        if indexed.truncated:
            return _candidate_result(
                candidate,
                outcome=LearningCandidateOutcome.INVALID,
                code="candidate_chunk_capacity_exceeded",
                decision=decision,
            )
        try:
            (
                operation_id,
                publication_entry,
                publication_chunks,
                publication_evidence,
                request_sha256,
            ) = prepare_knowledge_publication(
                indexed.entry,
                indexed.chunks,
                evidence=evidence,
                operation_id=operation_id,
                expected_revision=None,
            )
        except (TypeError, ValueError):
            return _candidate_result(
                candidate,
                outcome=LearningCandidateOutcome.INVALID,
                code="candidate_publication_invalid",
                decision=decision,
            )
        expectation = _PublicationExpectation(
            operation_id=operation_id,
            entry_id=publication_entry.id,
            entry_revision=publication_entry.revision,
            expected_revision=None,
            request_sha256=request_sha256,
            entry_created_at=publication_entry.created_at,
            entry_updated_at=publication_entry.updated_at,
        )
        try:
            receipt = await self._publish_owned(
                operation_id=operation_id,
                proposal_fingerprint=proposal_fingerprint,
                entry=publication_entry,
                chunks=publication_chunks,
                evidence=publication_evidence,
            )
        except asyncio.CancelledError:
            raise
        except NotImplementedError:
            return _candidate_result(
                candidate,
                outcome=LearningCandidateOutcome.FAILED,
                code="owned_publication_unsupported",
                decision=decision,
            )
        except _PublicationCapacityExhausted:
            return _candidate_result(
                candidate,
                outcome=LearningCandidateOutcome.FAILED,
                code="publication_capacity_exhausted",
                decision=decision,
            )
        except KnowledgePublicationOwnerClosed:
            return _candidate_result(
                candidate,
                outcome=LearningCandidateOutcome.FAILED,
                code="publication_owner_closed",
                decision=decision,
            )
        except (KnowledgePublicationConflict, KnowledgeRevisionConflict):
            return await self._reconcile_after_publication_error(
                candidate,
                decision=decision,
                operation_id=operation_id,
                entry_id=entry_id,
                proposal_fingerprint=proposal_fingerprint,
                expectation=expectation,
                conflict=True,
            )
        except Exception:
            return await self._reconcile_after_publication_error(
                candidate,
                decision=decision,
                operation_id=operation_id,
                entry_id=entry_id,
                proposal_fingerprint=proposal_fingerprint,
                expectation=expectation,
                conflict=False,
            )
        if not _publication_receipt_matches_expectation(receipt, expectation):
            return await self._reconcile_after_publication_error(
                candidate,
                decision=decision,
                operation_id=operation_id,
                entry_id=entry_id,
                proposal_fingerprint=proposal_fingerprint,
                expectation=expectation,
                conflict=False,
            )
        outcome = (
            LearningCandidateOutcome.EXISTING_PENDING
            if receipt.replayed
            else LearningCandidateOutcome.PENDING_PERSISTED
        )
        return _candidate_result(
            candidate,
            outcome=outcome,
            code=outcome.value,
            decision=decision,
            entry=publication_entry,
        )

    async def _publish_owned(
        self,
        *,
        operation_id: str,
        proposal_fingerprint: str,
        entry: KnowledgeEntry,
        chunks: list[KnowledgeChunk],
        evidence: list[KnowledgeEvidence],
    ) -> KnowledgePublicationReceipt:
        try:
            publication = await self._publication_owner.run(
                operation_id,
                proposal_fingerprint,
                lambda: self._store.publish_entry_revision(
                    entry,
                    chunks,
                    evidence=evidence,
                    access_scope=self._access_scope,
                    operation_id=operation_id,
                    expected_revision=None,
                ),
            )
        except KnowledgePublicationCapacityExhausted:
            raise _PublicationCapacityExhausted from None
        except KnowledgePublicationOperationConflict:
            raise KnowledgePublicationConflict("in_flight_operation_mismatch") from None
        copied = _copy_publication_receipt(publication.value)
        if publication.joined and not copied.replayed:
            copied = copied.model_copy(update={"replayed": True})
        return copied

    async def _reconcile_after_publication_error(
        self,
        candidate: LearningCandidate,
        *,
        decision: LearningDecision,
        operation_id: str,
        entry_id: str,
        proposal_fingerprint: str,
        expectation: _PublicationExpectation,
        conflict: bool,
    ) -> LearningCandidateResult:
        try:
            receipt = await self._store.load_entry_publication_receipt(
                operation_id,
                access_scope=self._access_scope,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            return _candidate_result(
                candidate,
                outcome=(
                    LearningCandidateOutcome.CONFLICT
                    if conflict
                    else LearningCandidateOutcome.FAILED
                ),
                code=("publication_conflict" if conflict else "publication_outcome_ambiguous"),
                decision=decision,
            )
        if receipt is None:
            return _candidate_result(
                candidate,
                outcome=(
                    LearningCandidateOutcome.CONFLICT
                    if conflict
                    else LearningCandidateOutcome.FAILED
                ),
                code=("publication_conflict" if conflict else "publication_outcome_ambiguous"),
                decision=decision,
            )
        try:
            receipt = _copy_publication_receipt(receipt)
        except (TypeError, ValueError):
            return _candidate_result(
                candidate,
                outcome=LearningCandidateOutcome.FAILED,
                code="publication_receipt_invalid",
                decision=decision,
            )
        reconciled = await self._reconcile_existing_candidate(
            candidate,
            receipt=receipt,
            expected_operation_id=operation_id,
            expected_entry_id=entry_id,
            proposal_fingerprint=proposal_fingerprint,
        )
        if (
            not conflict
            and reconciled.outcome is LearningCandidateOutcome.EXISTING_PENDING
            and _publication_receipt_matches_expectation(receipt, expectation)
        ):
            return reconciled.model_copy(
                update={
                    "outcome": LearningCandidateOutcome.PENDING_PERSISTED,
                    "code": "pending_persisted",
                    "decision": decision,
                    "warning_code": "publication_acknowledgement_lost",
                }
            )
        return reconciled.model_copy(update={"decision": decision})

    async def _reconcile_existing_candidate(
        self,
        candidate: LearningCandidate,
        *,
        receipt: KnowledgePublicationReceipt,
        expected_operation_id: str,
        expected_entry_id: str,
        proposal_fingerprint: str,
    ) -> LearningCandidateResult:
        if not _publication_receipt_has_creation_identity(
            receipt,
            operation_id=expected_operation_id,
            entry_id=expected_entry_id,
        ):
            return _candidate_result(
                candidate,
                outcome=LearningCandidateOutcome.CONFLICT,
                code="publication_receipt_identity_conflict",
            )
        try:
            historical = await self._store.get_entry(
                receipt.entry_id,
                revision=receipt.entry_revision,
                access_scope=self._access_scope,
            )
            current = await self._store.get_entry(
                receipt.entry_id,
                access_scope=self._access_scope,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            return _candidate_result(
                candidate,
                outcome=LearningCandidateOutcome.FAILED,
                code="published_entry_read_failed",
            )
        if historical is None or current is None:
            return _candidate_result(
                candidate,
                outcome=LearningCandidateOutcome.FAILED,
                code="published_entry_unavailable",
            )
        audit = historical.metadata.get(_CURATOR_METADATA_KEY)
        if (
            type(audit) is not dict
            or audit.get("proposal_fingerprint") != proposal_fingerprint
            or historical.namespace != self._config.namespace
            or historical.labels != self._config.labels
            or historical.visibility is not self._config.visibility
            or historical.status is not KnowledgeStatus.PENDING
        ):
            return _candidate_result(
                candidate,
                outcome=LearningCandidateOutcome.CONFLICT,
                code="published_proposal_conflict",
            )
        outcome = _existing_candidate_outcome(current.status)
        return _candidate_result(
            candidate,
            outcome=outcome,
            code=outcome.value,
            entry=current,
        )

    def _batch_failure(
        self,
        batch: LearningBatch,
        *,
        processed_at: datetime,
        outcome: LearningBatchOutcome,
        code: str,
    ) -> LearningBatchResult:
        return LearningBatchResult(
            batch_id=batch.id,
            batch_fingerprint=batch.fingerprint,
            configuration_fingerprint=self._config.fingerprint,
            scope=batch.scope,
            outcome=outcome,
            code=code,
            signal_count=len(batch.signals),
            candidate_count=0,
            signals=tuple(
                LearningSignalResult(
                    signal_id=signal.id,
                    signal_fingerprint=signal.fingerprint,
                    outcome=LearningSignalOutcome.BATCH_FAILED,
                    code=code,
                )
                for signal in batch.signals
            ),
            processed_at=processed_at,
        )


def _copy_publication_receipt(value: object) -> KnowledgePublicationReceipt:
    if type(value) is not KnowledgePublicationReceipt:
        raise TypeError("knowledge store returned an invalid publication receipt")
    return KnowledgePublicationReceipt.model_validate(value.model_dump(mode="python"))


def _publication_receipt_has_creation_identity(
    receipt: KnowledgePublicationReceipt,
    *,
    operation_id: str,
    entry_id: str,
) -> bool:
    return (
        receipt.operation_id == operation_id
        and receipt.entry_id == entry_id
        and receipt.expected_revision is None
        and receipt.entry_revision == 1
    )


def _publication_receipt_matches_expectation(
    receipt: KnowledgePublicationReceipt,
    expectation: _PublicationExpectation,
) -> bool:
    return (
        receipt.operation_id == expectation.operation_id
        and receipt.entry_id == expectation.entry_id
        and receipt.entry_revision == expectation.entry_revision
        and receipt.expected_revision == expectation.expected_revision
        and receipt.request_sha256 == expectation.request_sha256
        and receipt.entry_created_at == expectation.entry_created_at
        and receipt.entry_updated_at == expectation.entry_updated_at
    )


def _build_pending_publication(
    batch: LearningBatch,
    candidate: LearningCandidate,
    decision: LearningDecision,
    *,
    signals: tuple[LearningSignal, ...],
    config: KnowledgeCuratorConfig,
    entry_id: str,
    proposal_fingerprint: str,
    processed_at: datetime,
):
    audit = {
        "schema_version": LEARNING_SCHEMA_VERSION,
        "batch_id": batch.id,
        "signal_ids": list(candidate.signal_ids),
        "signal_deduplication_keys": [signal.deduplication_key for signal in signals],
        "proposal_key": candidate.proposal_key,
        "candidate_input_fingerprint": _candidate_input_fingerprint(
            candidate,
            signals=signals,
            config=config,
        ),
        "proposal_fingerprint": proposal_fingerprint,
        "candidate_generator_identity": config.candidate_generator_identity,
        "evaluator_identity": config.evaluator_identity,
        "evaluator_verdict": decision.verdict.value,
        "evaluator_code": decision.code,
        "evaluator_notes": decision.notes,
        "evaluator_confidence": decision.confidence,
        "evaluator_metadata": decision.metadata,
        "candidate_confidence_hint": candidate.confidence_hint,
        "candidate_metadata": candidate.metadata,
        "policy_identity": config.policy_identity,
        "pipeline_version": config.pipeline_version,
        "configuration_fingerprint": config.fingerprint,
        "processed_at": processed_at.isoformat(),
    }
    _bounded_json_object(audit, "curation audit metadata", max_bytes=MAX_LEARNING_METADATA_BYTES)
    sources = _candidate_sources(candidate, signals)
    primary_source = sources[0][2]
    indexer = KnowledgeIndexer()
    indexed = indexer.build(
        KnowledgeIndexRequest(
            text=candidate.text,
            entry_id=entry_id,
            namespace=config.namespace,
            labels=config.labels,
            kind=candidate.kind,
            visibility=config.visibility,
            status=KnowledgeStatus.PENDING,
            created_by_type=KnowledgeActorType.APP,
            created_by=config.created_by,
            source_type=primary_source.source_type,
            source_uri=primary_source.source_uri,
            source_id=primary_source.source_id,
            aspects=list(candidate.aspects),
            impact_targets=list(candidate.impact_targets),
            confidence=decision.confidence,
            title=candidate.title,
            metadata={_CURATOR_METADATA_KEY: audit},
            chunk_metadata={
                "curation_proposal_fingerprint": proposal_fingerprint,
            },
            entry_text_max_bytes=config.max_candidate_text_bytes,
            chunk_target_bytes=config.chunk_target_bytes,
            chunk_overlap_bytes=config.chunk_overlap_bytes,
            max_chunks=config.max_chunks,
            skip_unchanged=False,
        )
    )
    entry = indexed.entry.model_copy(
        update={"created_at": processed_at, "updated_at": processed_at}
    )
    indexed = indexed.model_copy(update={"entry": entry})
    evidence: list[KnowledgeEvidence] = []
    for ordinal, (signal_id, deduplication_key, reference) in enumerate(sources):
        evidence_material = {
            "entry_id": entry_id,
            "proposal_fingerprint": proposal_fingerprint,
            "ordinal": ordinal,
            "source": reference.identity_material(),
        }
        source_metadata = _curation_evidence_metadata(
            signal_id,
            deduplication_key,
            reference,
            ordinal=ordinal,
        )
        evidence.append(
            KnowledgeEvidence(
                id=f"curated-evidence-{_sha256(evidence_material, 'curation evidence identity')}",
                entry_id=entry_id,
                entry_revision=1,
                role=reference.role,
                source_type=reference.source_type,
                source_id=reference.source_id,
                source_uri=reference.source_uri,
                source_revision=reference.source_revision,
                source_hash=reference.source_hash,
                locator=reference.locator,
                disposition=KnowledgeEvidenceDisposition.LIVE,
                created_at=processed_at,
                metadata=source_metadata,
            )
        )
    return indexed, evidence


def _candidate_sources(
    candidate: LearningCandidate,
    signals: tuple[LearningSignal, ...],
) -> list[tuple[str | None, str | None, LearningSourceReference]]:
    sources: list[tuple[str | None, str | None, LearningSourceReference]] = []
    for signal in signals:
        for reference in signal.source_references:
            sources.append((signal.id, signal.deduplication_key, reference))
    for reference in candidate.source_references:
        sources.append((None, None, reference))
    return sources


def _curation_evidence_metadata(
    signal_id: str | None,
    deduplication_key: str | None,
    reference: LearningSourceReference,
    *,
    ordinal: int,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        **reference.metadata,
        "learning_source_origin": "candidate" if signal_id is None else "signal",
        "source_ordinal": ordinal,
    }
    if signal_id is not None and deduplication_key is not None:
        metadata.update(
            {
                "learning_signal_id": signal_id,
                "learning_signal_deduplication_key": deduplication_key,
            }
        )
    return _bounded_json_object(
        metadata,
        "curation evidence metadata",
        max_bytes=MAX_KNOWLEDGE_EVIDENCE_JSON_BYTES,
    )


def _proposal_fingerprint(
    candidate: LearningCandidate,
    decision: LearningDecision,
    *,
    signals: tuple[LearningSignal, ...],
    config: KnowledgeCuratorConfig,
) -> str:
    return _sha256(
        {
            "contract": "cayu.reviewed-knowledge-proposal.v1",
            "candidate_input_fingerprint": _candidate_input_fingerprint(
                candidate,
                signals=signals,
                config=config,
            ),
            "decision": decision.model_dump(mode="json"),
        },
        "reviewed knowledge proposal",
    )


def _candidate_input_fingerprint(
    candidate: LearningCandidate,
    *,
    signals: tuple[LearningSignal, ...],
    config: KnowledgeCuratorConfig,
) -> str:
    return _sha256(
        {
            "contract": "cayu.reviewed-knowledge-candidate-input.v1",
            "scope": {
                "namespace": config.namespace,
                "labels": config.labels,
                "visibility": config.visibility.value,
            },
            "candidate": candidate.model_dump(mode="json"),
            "signals": [signal.model_dump(mode="json") for signal in signals],
            "candidate_generator_identity": config.candidate_generator_identity,
            "evaluator_identity": config.evaluator_identity,
            "policy_identity": config.policy_identity,
            "pipeline_version": config.pipeline_version,
        },
        "reviewed knowledge candidate input",
    )


def _curated_entry_id(proposal_key: str, config: KnowledgeCuratorConfig) -> str:
    material = {
        "contract": "cayu.curated-entry-identity.v1",
        "namespace": config.namespace,
        "labels": config.labels,
        "visibility": config.visibility.value,
        "proposal_key": proposal_key,
    }
    return f"curated-{_sha256(material, 'curated entry identity')}"


def _curation_operation_id(proposal_key: str, config: KnowledgeCuratorConfig) -> str:
    material = {
        "contract": "cayu.curated-publication-operation.v1",
        "namespace": config.namespace,
        "labels": config.labels,
        "visibility": config.visibility.value,
        "proposal_key": proposal_key,
    }
    return f"curate-{_sha256(material, 'curation operation identity')}"


def _candidate_result(
    candidate: LearningCandidate,
    *,
    outcome: LearningCandidateOutcome,
    code: str,
    decision: LearningDecision | None = None,
    entry: KnowledgeEntry | None = None,
) -> LearningCandidateResult:
    return LearningCandidateResult(
        proposal_key=candidate.proposal_key,
        candidate_fingerprint=candidate.fingerprint,
        outcome=outcome,
        code=code,
        decision=decision,
        entry_id=None if entry is None else entry.id,
        entry_revision=None if entry is None else entry.revision,
        entry_status=None if entry is None else entry.status,
    )


def _completed_signal_results(
    batch: LearningBatch,
    candidates: tuple[LearningCandidate, ...],
) -> tuple[LearningSignalResult, ...]:
    proposal_keys_by_signal: dict[str, list[str]] = {signal.id: [] for signal in batch.signals}
    for candidate in candidates:
        for signal_id in candidate.signal_ids:
            proposal_keys_by_signal[signal_id].append(candidate.proposal_key)
    results: list[LearningSignalResult] = []
    for signal in batch.signals:
        proposal_keys = tuple(proposal_keys_by_signal[signal.id])
        outcome = (
            LearningSignalOutcome.CANDIDATE_GENERATED
            if proposal_keys
            else LearningSignalOutcome.NO_CANDIDATE_GENERATED
        )
        results.append(
            LearningSignalResult(
                signal_id=signal.id,
                signal_fingerprint=signal.fingerprint,
                outcome=outcome,
                code=outcome.value,
                candidate_proposal_keys=proposal_keys,
            )
        )
    return tuple(results)


def _existing_candidate_outcome(status: KnowledgeStatus) -> LearningCandidateOutcome:
    return {
        KnowledgeStatus.PENDING: LearningCandidateOutcome.EXISTING_PENDING,
        KnowledgeStatus.ACTIVE: LearningCandidateOutcome.EXISTING_ACTIVE,
        KnowledgeStatus.ARCHIVED: LearningCandidateOutcome.EXISTING_ARCHIVED,
        KnowledgeStatus.DELETED: LearningCandidateOutcome.EXISTING_DELETED,
    }[status]


def _failed_evaluation(candidate: LearningCandidate, code: str) -> _EvaluationState:
    return _EvaluationState(
        candidate=candidate,
        result=_candidate_result(
            candidate,
            outcome=LearningCandidateOutcome.FAILED,
            code=code,
        ),
    )


def _validate_curator_store(store: object) -> None:
    for method_name in (
        "bound_access_scope",
        "get_entry",
        "publish_entry_revision",
        "load_entry_publication_receipt",
    ):
        if not callable(getattr(store, method_name, None)):
            raise TypeError("store must implement the knowledge curator store methods.")


def _validate_curator_access_scope(
    config: KnowledgeCuratorConfig,
    access_scope: KnowledgeAccessScope,
) -> None:
    if (
        not access_scope.allow_all_namespaces
        and config.namespace not in access_scope.allowed_namespaces
    ):
        raise ValueError("Knowledge curator namespace is outside its access scope.")
    for key, value in access_scope.required_labels.items():
        if config.labels.get(key) != value:
            raise ValueError("Knowledge curator labels do not satisfy its access scope.")
    if config.visibility not in access_scope.allowed_visibilities:
        raise ValueError("Knowledge curator visibility is outside its access scope.")
    if KnowledgeStatus.PENDING not in access_scope.allowed_statuses:
        raise ValueError("Knowledge curator access scope must allow pending knowledge.")


__all__ = [
    "CandidatePolicyDisposition",
    "KnowledgeCandidateGenerator",
    "KnowledgeCandidatePolicy",
    "KnowledgeCandidatePolicyDecision",
    "KnowledgeCurator",
    "KnowledgeCuratorConfig",
    "LearningBatch",
    "LearningBatchOutcome",
    "LearningBatchResult",
    "LearningCandidate",
    "LearningCandidateOutcome",
    "LearningCandidateResult",
    "LearningDecision",
    "LearningEvaluator",
    "LearningSignal",
    "LearningSignalOutcome",
    "LearningSignalResult",
    "LearningSourceReference",
    "LearningVerdict",
    "group_learning_signals",
]
