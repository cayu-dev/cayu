"""Deterministic candidate routing for reviewed knowledge maintenance.

This module owns the bounded, read-only boundary between application-produced
maintenance signals and a future injected consolidation planner.  It does not
discover signals, call a model, infer semantic truth, persist proposals, or
change knowledge lifecycle state.
"""

from __future__ import annotations

import asyncio
import re
from collections import deque
from collections.abc import Awaitable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from typing import Any, Literal, Protocol, TypeVar

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

from cayu._validation import (
    canonical_durable_json_bytes,
    copy_label_map,
    require_durable_clean_nonblank,
    require_finite,
)
from cayu.storage.memory import (
    MAX_KNOWLEDGE_MAINTENANCE_SOURCES,
    MAX_KNOWLEDGE_RELATION_BYTES,
    MAX_KNOWLEDGE_RELATION_LIMIT,
    KnowledgeAccessScope,
    KnowledgeEntry,
    KnowledgeEntryReadLimitExceeded,
    KnowledgeRelation,
    KnowledgeRelationKind,
    KnowledgeRelationQuery,
    KnowledgeRelationResult,
    KnowledgeRevisionRef,
    KnowledgeStatus,
    copy_knowledge_access_scope,
    copy_knowledge_entry,
    copy_knowledge_revision_ref,
    knowledge_entry_payload_bytes,
)

KNOWLEDGE_MAINTENANCE_ROUTING_SCHEMA_VERSION = 1
MAX_KNOWLEDGE_MAINTENANCE_ROUTING_SIGNALS = 256
MAX_KNOWLEDGE_MAINTENANCE_ROUTING_CANDIDATE_READS = 512
MAX_KNOWLEDGE_MAINTENANCE_ROUTING_BYTES = 4 * 1024 * 1024
MAX_KNOWLEDGE_MAINTENANCE_ROUTING_TIMEOUT_SECONDS = 300.0

_IDENTITY_MAX_BYTES = 256
_REASON_CODE_MAX_BYTES = 128
_SAFE_CODE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*\Z")
_T = TypeVar("_T")


class _RoutingModel(BaseModel):
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


def _utc(value: datetime, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"`{field_name}` must be a datetime.")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"`{field_name}` must be timezone-aware.")
    return value.astimezone(UTC)


def _fingerprint(value: object, field_name: str) -> str:
    return sha256(canonical_durable_json_bytes(value, field_name)).hexdigest()


def _copy_revision_ref(value: object) -> KnowledgeRevisionRef:
    if isinstance(value, KnowledgeRevisionRef):
        return copy_knowledge_revision_ref(value)
    return KnowledgeRevisionRef.model_validate(value)


class KnowledgeMaintenanceSignalKind(StrEnum):
    """A deterministic hint for why exact knowledge revisions merit review."""

    EXACT_REFERENCE = "exact_reference"
    CONTRADICTION = "contradiction"
    DUPLICATE_HINT = "duplicate_hint"
    EXPIRY = "expiry"
    LOW_USAGE = "low_usage"


class KnowledgeMaintenanceRoutingOmissionReason(StrEnum):
    """Privacy-safe reason why one submitted signal did not reach the planner."""

    UNAVAILABLE = "unavailable"
    STALE_REVISION = "stale_revision"
    SCOPE_MISMATCH = "scope_mismatch"
    LIFECYCLE_MISMATCH = "lifecycle_mismatch"
    CONDITION_NOT_MET = "condition_not_met"
    RELATION_COVERAGE_INCOMPLETE = "relation_coverage_incomplete"
    CANDIDATE_LIMIT = "candidate_limit"
    CANDIDATE_BYTES = "candidate_bytes"


class KnowledgeMaintenanceRoutingLimitExceeded(ValueError):
    """A request exceeds a configured pre-read work ceiling."""

    def __init__(self, limit: str) -> None:
        self.limit = _clean(limit, "limit")
        super().__init__("Knowledge maintenance routing exceeds a configured work limit.")


class KnowledgeMaintenanceRoutingTimeout(TimeoutError):
    """Candidate routing did not complete within its configured deadline."""

    def __init__(self) -> None:
        super().__init__("Knowledge maintenance routing timed out.")


class KnowledgeMaintenanceCandidateSignal(_RoutingModel):
    """One exact, application-produced candidate hint.

    Similarity scores are retained only as diagnostics.  The router never uses
    them as evidence that entries are equivalent, contradictory, or true.
    """

    schema_version: Literal[1] = KNOWLEDGE_MAINTENANCE_ROUTING_SCHEMA_VERSION
    id: StrictStr
    kind: KnowledgeMaintenanceSignalKind
    references: tuple[KnowledgeRevisionRef, ...]
    producer_id: StrictStr
    producer_version: StrictStr
    reason_code: StrictStr
    observed_at: datetime
    threshold_at: datetime | None = None
    relation_id: StrictStr | None = None
    raw_score: StrictFloat | None = None
    score_kind: StrictStr | None = None

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> object:
        if type(value) is not int or value != KNOWLEDGE_MAINTENANCE_ROUTING_SCHEMA_VERSION:
            raise ValueError("`schema_version` must be the integer 1.")
        return value

    @field_validator("id", "producer_id", "producer_version")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _clean(value, info.field_name)

    @field_validator("reason_code")
    @classmethod
    def validate_reason_code(cls, value: str) -> str:
        value = _clean(value, "reason_code", max_bytes=_REASON_CODE_MAX_BYTES)
        if _SAFE_CODE_RE.fullmatch(value) is None:
            raise ValueError("`reason_code` must be a safe machine-readable code.")
        return value

    @field_validator("relation_id")
    @classmethod
    def validate_relation_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _clean(value, "relation_id")

    @field_validator("score_kind")
    @classmethod
    def validate_score_kind(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _clean(value, "score_kind")

    @field_validator("references", mode="before")
    @classmethod
    def copy_references(cls, value: object) -> tuple[KnowledgeRevisionRef, ...]:
        if not isinstance(value, list | tuple):
            raise ValueError("`references` must be an ordered array.")
        if not value or len(value) > 2:
            raise ValueError("`references` must contain one or two exact revisions.")
        copied = tuple(
            sorted(
                (_copy_revision_ref(item) for item in value),
                key=lambda item: (item.entry_id, item.revision),
            )
        )
        if len({item.entry_id for item in copied}) != len(copied):
            raise ValueError("A signal cannot repeat one logical knowledge entry.")
        return copied

    @field_validator("observed_at", "threshold_at")
    @classmethod
    def validate_timestamps(cls, value: datetime | None, info) -> datetime | None:
        if value is None:
            return None
        return _utc(value, info.field_name)

    @field_validator("raw_score")
    @classmethod
    def validate_raw_score(cls, value: float | None) -> float | None:
        if value is None:
            return None
        return require_finite(value, "raw_score")

    @model_validator(mode="after")
    def validate_signal_shape(self) -> KnowledgeMaintenanceCandidateSignal:
        paired = {
            KnowledgeMaintenanceSignalKind.CONTRADICTION,
            KnowledgeMaintenanceSignalKind.DUPLICATE_HINT,
        }
        if self.kind in paired and len(self.references) != 2:
            raise ValueError(f"A {self.kind.value!r} signal requires two exact revisions.")
        if self.kind not in paired and len(self.references) != 1:
            raise ValueError(f"A {self.kind.value!r} signal requires one exact revision.")
        thresholded = {
            KnowledgeMaintenanceSignalKind.EXPIRY,
            KnowledgeMaintenanceSignalKind.LOW_USAGE,
        }
        if self.kind in thresholded and self.threshold_at is None:
            raise ValueError(f"A {self.kind.value!r} signal requires `threshold_at`.")
        if self.kind not in thresholded and self.threshold_at is not None:
            raise ValueError(f"A {self.kind.value!r} signal cannot set `threshold_at`.")
        if self.kind is KnowledgeMaintenanceSignalKind.CONTRADICTION and self.relation_id is None:
            raise ValueError("A contradiction signal requires `relation_id`.")
        if (
            self.kind is not KnowledgeMaintenanceSignalKind.CONTRADICTION
            and self.relation_id is not None
        ):
            raise ValueError("Only a contradiction signal may set `relation_id`.")
        if self.threshold_at is not None and self.threshold_at > self.observed_at:
            raise ValueError("`threshold_at` cannot be later than `observed_at`.")
        if (self.raw_score is None) != (self.score_kind is None):
            raise ValueError("`raw_score` and `score_kind` must be set together.")
        if (
            self.raw_score is not None
            and self.kind is not KnowledgeMaintenanceSignalKind.DUPLICATE_HINT
        ):
            raise ValueError("Only duplicate hints may carry a raw diagnostic score.")
        return self

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            {
                "contract": "cayu.knowledge-maintenance-candidate-signal.v1",
                "signal": self.model_dump(mode="json"),
            },
            "knowledge maintenance candidate signal",
        )


class KnowledgeMaintenanceRoutingRequest(_RoutingModel):
    """One explicit bounded routing request within an application-owned scope."""

    schema_version: Literal[1] = KNOWLEDGE_MAINTENANCE_ROUTING_SCHEMA_VERSION
    id: StrictStr
    policy_id: StrictStr
    namespace: StrictStr
    labels: dict[str, str] = Field(default_factory=dict)
    access_scope: KnowledgeAccessScope
    signals: tuple[KnowledgeMaintenanceCandidateSignal, ...] = ()
    created_at: datetime

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> object:
        if type(value) is not int or value != KNOWLEDGE_MAINTENANCE_ROUTING_SCHEMA_VERSION:
            raise ValueError("`schema_version` must be the integer 1.")
        return value

    @field_validator("id", "policy_id", "namespace")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _clean(value, info.field_name)

    @field_validator("labels", mode="before")
    @classmethod
    def copy_labels(cls, value: Any) -> dict[str, str]:
        return copy_label_map(value, "labels")

    @field_validator("access_scope", mode="before")
    @classmethod
    def copy_access_scope(cls, value: object) -> object:
        if isinstance(value, KnowledgeAccessScope):
            return copy_knowledge_access_scope(value)
        return value

    @field_validator("signals", mode="before")
    @classmethod
    def copy_signals(cls, value: object) -> tuple[KnowledgeMaintenanceCandidateSignal, ...]:
        if not isinstance(value, list | tuple):
            raise ValueError("`signals` must be an ordered array.")
        if len(value) > MAX_KNOWLEDGE_MAINTENANCE_ROUTING_SIGNALS:
            raise ValueError(
                "`signals` must contain at most "
                f"{MAX_KNOWLEDGE_MAINTENANCE_ROUTING_SIGNALS} values."
            )
        copied = [
            KnowledgeMaintenanceCandidateSignal.model_validate(
                item.model_dump(mode="python")
                if isinstance(item, KnowledgeMaintenanceCandidateSignal)
                else item
            )
            for item in value
        ]
        copied.sort(key=lambda signal: (signal.observed_at, signal.id))
        return tuple(copied)

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        return _utc(value, "created_at")

    @model_validator(mode="after")
    def validate_signals(self) -> KnowledgeMaintenanceRoutingRequest:
        ids = tuple(signal.id for signal in self.signals)
        if len(ids) != len(set(ids)):
            raise ValueError("`signals` cannot repeat an identity.")
        semantic_fingerprints = tuple(
            _signal_semantic_fingerprint(signal) for signal in self.signals
        )
        if len(semantic_fingerprints) != len(set(semantic_fingerprints)):
            raise ValueError("`signals` cannot repeat one semantic observation.")
        revisions_by_entry: dict[str, set[int]] = {}
        for signal in self.signals:
            if signal.observed_at > self.created_at:
                raise ValueError("A signal cannot be observed after its routing request.")
            for reference in signal.references:
                revisions_by_entry.setdefault(reference.entry_id, set()).add(reference.revision)
        if any(len(revisions) != 1 for revisions in revisions_by_entry.values()):
            raise ValueError("One request cannot name conflicting revisions of one entry.")
        return self

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            {
                "contract": "cayu.knowledge-maintenance-routing-request.v1",
                "request": self.model_dump(mode="json"),
            },
            "knowledge maintenance routing request",
        )


class KnowledgeMaintenanceRouterConfig(_RoutingModel):
    """Application-owned deterministic priority and resource ceilings."""

    schema_version: Literal[1] = KNOWLEDGE_MAINTENANCE_ROUTING_SCHEMA_VERSION
    router_version: StrictStr = "cayu.knowledge-maintenance-router.v1"
    signal_priority: tuple[KnowledgeMaintenanceSignalKind, ...] = tuple(
        KnowledgeMaintenanceSignalKind
    )
    max_signals: StrictInt = 100
    max_candidate_reads: StrictInt = 100
    max_candidates: StrictInt = 20
    max_candidate_bytes: StrictInt = 256 * 1024
    max_candidate_load_bytes: StrictInt = MAX_KNOWLEDGE_MAINTENANCE_ROUTING_BYTES
    max_concurrency: StrictInt = 8
    timeout_seconds: StrictFloat = 10.0
    relation_page_limit: StrictInt = 50
    max_relation_records_per_signal: StrictInt = 200
    relation_page_max_bytes: StrictInt = 256 * 1024
    max_relation_load_bytes: StrictInt = MAX_KNOWLEDGE_MAINTENANCE_ROUTING_BYTES

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> object:
        if type(value) is not int or value != KNOWLEDGE_MAINTENANCE_ROUTING_SCHEMA_VERSION:
            raise ValueError("`schema_version` must be the integer 1.")
        return value

    @field_validator("router_version")
    @classmethod
    def validate_router_version(cls, value: str) -> str:
        return _clean(value, "router_version")

    @field_validator("signal_priority", mode="before")
    @classmethod
    def copy_signal_priority(cls, value: object) -> tuple[KnowledgeMaintenanceSignalKind, ...]:
        if not isinstance(value, list | tuple):
            raise ValueError("`signal_priority` must be an ordered array.")
        copied = tuple(
            item
            if isinstance(item, KnowledgeMaintenanceSignalKind)
            else KnowledgeMaintenanceSignalKind(item)
            for item in value
        )
        if set(copied) != set(KnowledgeMaintenanceSignalKind) or len(copied) != len(
            KnowledgeMaintenanceSignalKind
        ):
            raise ValueError("`signal_priority` must contain every signal kind exactly once.")
        return copied

    @field_validator(
        "max_signals",
        "max_candidate_reads",
        "max_candidates",
        "max_candidate_bytes",
        "max_candidate_load_bytes",
        "max_concurrency",
        "relation_page_limit",
        "max_relation_records_per_signal",
        "relation_page_max_bytes",
        "max_relation_load_bytes",
    )
    @classmethod
    def validate_positive_int(cls, value: int, info) -> int:
        if value <= 0:
            raise ValueError(f"`{info.field_name}` must be greater than 0.")
        return value

    @field_validator("timeout_seconds")
    @classmethod
    def validate_timeout(cls, value: float) -> float:
        value = require_finite(value, "timeout_seconds")
        if value <= 0.0 or value > MAX_KNOWLEDGE_MAINTENANCE_ROUTING_TIMEOUT_SECONDS:
            raise ValueError(
                "`timeout_seconds` must be greater than 0 and at most "
                f"{MAX_KNOWLEDGE_MAINTENANCE_ROUTING_TIMEOUT_SECONDS}."
            )
        return value

    @model_validator(mode="after")
    def validate_bounds(self) -> KnowledgeMaintenanceRouterConfig:
        if self.max_signals > MAX_KNOWLEDGE_MAINTENANCE_ROUTING_SIGNALS:
            raise ValueError(
                f"`max_signals` must be at most {MAX_KNOWLEDGE_MAINTENANCE_ROUTING_SIGNALS}."
            )
        if self.max_candidate_reads > MAX_KNOWLEDGE_MAINTENANCE_ROUTING_CANDIDATE_READS:
            raise ValueError(
                "`max_candidate_reads` must be at most "
                f"{MAX_KNOWLEDGE_MAINTENANCE_ROUTING_CANDIDATE_READS}."
            )
        if self.max_candidates > MAX_KNOWLEDGE_MAINTENANCE_SOURCES:
            raise ValueError(
                f"`max_candidates` must be at most {MAX_KNOWLEDGE_MAINTENANCE_SOURCES}."
            )
        if self.max_candidates > self.max_candidate_reads:
            raise ValueError("`max_candidates` cannot exceed `max_candidate_reads`.")
        if self.max_candidate_bytes > MAX_KNOWLEDGE_MAINTENANCE_ROUTING_BYTES:
            raise ValueError(
                f"`max_candidate_bytes` must be at most {MAX_KNOWLEDGE_MAINTENANCE_ROUTING_BYTES}."
            )
        if self.max_candidate_bytes < 256:
            raise ValueError("`max_candidate_bytes` must be at least 256.")
        if self.max_candidate_load_bytes > MAX_KNOWLEDGE_MAINTENANCE_ROUTING_BYTES:
            raise ValueError(
                "`max_candidate_load_bytes` must be at most "
                f"{MAX_KNOWLEDGE_MAINTENANCE_ROUTING_BYTES}."
            )
        if self.max_concurrency > MAX_KNOWLEDGE_MAINTENANCE_ROUTING_CANDIDATE_READS:
            raise ValueError(
                "`max_concurrency` must be at most "
                f"{MAX_KNOWLEDGE_MAINTENANCE_ROUTING_CANDIDATE_READS}."
            )
        if self.relation_page_limit > MAX_KNOWLEDGE_RELATION_LIMIT:
            raise ValueError(
                f"`relation_page_limit` must be at most {MAX_KNOWLEDGE_RELATION_LIMIT}."
            )
        if self.max_relation_records_per_signal > MAX_KNOWLEDGE_RELATION_LIMIT:
            raise ValueError(
                f"`max_relation_records_per_signal` must be at most {MAX_KNOWLEDGE_RELATION_LIMIT}."
            )
        if self.relation_page_max_bytes < MAX_KNOWLEDGE_RELATION_BYTES:
            raise ValueError(
                "`relation_page_max_bytes` must be large enough for one valid relation."
            )
        if self.relation_page_max_bytes > MAX_KNOWLEDGE_MAINTENANCE_ROUTING_BYTES:
            raise ValueError("`relation_page_max_bytes` exceeds the routing byte ceiling.")
        if self.max_relation_load_bytes < MAX_KNOWLEDGE_RELATION_BYTES:
            raise ValueError(
                "`max_relation_load_bytes` must be large enough for one valid relation."
            )
        if self.max_relation_load_bytes > MAX_KNOWLEDGE_MAINTENANCE_ROUTING_BYTES:
            raise ValueError("`max_relation_load_bytes` exceeds the routing byte ceiling.")
        return self

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            {
                "contract": "cayu.knowledge-maintenance-router-config.v1",
                "config": self.model_dump(mode="json"),
            },
            "knowledge maintenance router configuration",
        )


class KnowledgeMaintenanceRoutedCandidate(_RoutingModel):
    """One authorized current candidate and its deterministic signal support."""

    reference: KnowledgeRevisionRef
    entry: KnowledgeEntry
    signal_ids: tuple[StrictStr, ...]
    signal_kinds: tuple[KnowledgeMaintenanceSignalKind, ...]

    @field_validator("reference", mode="before")
    @classmethod
    def copy_reference(cls, value: object) -> KnowledgeRevisionRef:
        return _copy_revision_ref(value)

    @field_validator("entry", mode="before")
    @classmethod
    def copy_entry(cls, value: object) -> object:
        if type(value) is KnowledgeEntry:
            return copy_knowledge_entry(value)
        return value

    @field_validator("signal_ids", mode="before")
    @classmethod
    def copy_signal_ids(cls, value: object) -> tuple[str, ...]:
        return _ordered_unique_identities(value, "signal_ids")

    @field_validator("signal_kinds", mode="before")
    @classmethod
    def copy_signal_kinds(cls, value: object) -> tuple[KnowledgeMaintenanceSignalKind, ...]:
        if not isinstance(value, list | tuple) or not value:
            raise ValueError("`signal_kinds` must be a non-empty ordered array.")
        copied = tuple(
            item
            if isinstance(item, KnowledgeMaintenanceSignalKind)
            else KnowledgeMaintenanceSignalKind(item)
            for item in value
        )
        if len(copied) != len(set(copied)):
            raise ValueError("`signal_kinds` cannot contain duplicates.")
        return copied

    @model_validator(mode="after")
    def validate_entry_reference(self) -> KnowledgeMaintenanceRoutedCandidate:
        if (self.entry.id, self.entry.revision) != (
            self.reference.entry_id,
            self.reference.revision,
        ):
            raise ValueError("The routed entry must match its exact revision reference.")
        if self.entry.status is not KnowledgeStatus.ACTIVE:
            raise ValueError("A routed maintenance candidate must be active knowledge.")
        return self


class KnowledgeMaintenanceRoutingOmission(_RoutingModel):
    """One exhaustive, content-free signal disposition."""

    signal_id: StrictStr
    signal_kind: KnowledgeMaintenanceSignalKind
    reason: KnowledgeMaintenanceRoutingOmissionReason

    @field_validator("signal_id")
    @classmethod
    def validate_signal_id(cls, value: str) -> str:
        return _clean(value, "signal_id")


class KnowledgeMaintenanceRoutingResult(_RoutingModel):
    """Bounded deterministic routing output ready for an injected planner."""

    schema_version: Literal[1] = KNOWLEDGE_MAINTENANCE_ROUTING_SCHEMA_VERSION
    request_id: StrictStr
    request_fingerprint: StrictStr
    configuration_fingerprint: StrictStr
    candidates: tuple[KnowledgeMaintenanceRoutedCandidate, ...] = ()
    routed_signals: tuple[KnowledgeMaintenanceCandidateSignal, ...] = ()
    omissions: tuple[KnowledgeMaintenanceRoutingOmission, ...] = ()
    signal_count: StrictInt
    loaded_reference_count: StrictInt
    candidate_payload_bytes: StrictInt
    relation_payload_bytes: StrictInt
    max_candidates: StrictInt
    max_candidate_bytes: StrictInt
    max_relation_load_bytes: StrictInt
    truncated: bool = False

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> object:
        if type(value) is not int or value != KNOWLEDGE_MAINTENANCE_ROUTING_SCHEMA_VERSION:
            raise ValueError("`schema_version` must be the integer 1.")
        return value

    @field_validator("request_id")
    @classmethod
    def validate_request_id(cls, value: str) -> str:
        return _clean(value, "request_id")

    @field_validator("request_fingerprint", "configuration_fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str, info) -> str:
        if type(value) is not str or len(value) != 64:
            raise ValueError(f"`{info.field_name}` must be lowercase SHA-256 hex.")
        try:
            int(value, 16)
        except ValueError as exc:
            raise ValueError(f"`{info.field_name}` must be lowercase SHA-256 hex.") from exc
        if value != value.lower():
            raise ValueError(f"`{info.field_name}` must be lowercase SHA-256 hex.")
        return value

    @field_validator("candidates", mode="before")
    @classmethod
    def copy_candidates(cls, value: object) -> tuple[KnowledgeMaintenanceRoutedCandidate, ...]:
        return _copy_models(value, KnowledgeMaintenanceRoutedCandidate, "candidates")

    @field_validator("routed_signals", mode="before")
    @classmethod
    def copy_routed_signals(cls, value: object) -> tuple[KnowledgeMaintenanceCandidateSignal, ...]:
        return _copy_models(value, KnowledgeMaintenanceCandidateSignal, "routed_signals")

    @field_validator("omissions", mode="before")
    @classmethod
    def copy_omissions(cls, value: object) -> tuple[KnowledgeMaintenanceRoutingOmission, ...]:
        return _copy_models(value, KnowledgeMaintenanceRoutingOmission, "omissions")

    @field_validator(
        "signal_count",
        "loaded_reference_count",
        "candidate_payload_bytes",
        "relation_payload_bytes",
    )
    @classmethod
    def validate_nonnegative_int(cls, value: int, info) -> int:
        if value < 0:
            raise ValueError(f"`{info.field_name}` must be non-negative.")
        return value

    @field_validator("max_candidates", "max_candidate_bytes", "max_relation_load_bytes")
    @classmethod
    def validate_positive_int(cls, value: int, info) -> int:
        if value <= 0:
            raise ValueError(f"`{info.field_name}` must be greater than 0.")
        return value

    @field_validator("truncated", mode="before")
    @classmethod
    def validate_truncated(cls, value: object) -> bool:
        if type(value) is not bool:
            raise ValueError("`truncated` must be a boolean.")
        return value

    @model_validator(mode="after")
    def validate_result(self) -> KnowledgeMaintenanceRoutingResult:
        if self.signal_count > MAX_KNOWLEDGE_MAINTENANCE_ROUTING_SIGNALS:
            raise ValueError(
                f"`signal_count` cannot exceed {MAX_KNOWLEDGE_MAINTENANCE_ROUTING_SIGNALS}."
            )
        if self.loaded_reference_count > MAX_KNOWLEDGE_MAINTENANCE_ROUTING_CANDIDATE_READS:
            raise ValueError(
                "`loaded_reference_count` cannot exceed "
                f"{MAX_KNOWLEDGE_MAINTENANCE_ROUTING_CANDIDATE_READS}."
            )
        if self.max_candidates > MAX_KNOWLEDGE_MAINTENANCE_SOURCES:
            raise ValueError(f"`max_candidates` cannot exceed {MAX_KNOWLEDGE_MAINTENANCE_SOURCES}.")
        if self.max_candidate_bytes > MAX_KNOWLEDGE_MAINTENANCE_ROUTING_BYTES:
            raise ValueError(
                f"`max_candidate_bytes` cannot exceed {MAX_KNOWLEDGE_MAINTENANCE_ROUTING_BYTES}."
            )
        if self.max_relation_load_bytes > MAX_KNOWLEDGE_MAINTENANCE_ROUTING_BYTES:
            raise ValueError(
                "`max_relation_load_bytes` cannot exceed "
                f"{MAX_KNOWLEDGE_MAINTENANCE_ROUTING_BYTES}."
            )
        if len(self.candidates) > self.max_candidates:
            raise ValueError("`candidates` cannot exceed `max_candidates`.")
        if self.candidate_payload_bytes > self.max_candidate_bytes:
            raise ValueError("`candidate_payload_bytes` cannot exceed `max_candidate_bytes`.")
        if self.relation_payload_bytes > self.max_relation_load_bytes:
            raise ValueError("`relation_payload_bytes` cannot exceed `max_relation_load_bytes`.")
        routed_ids = tuple(signal.id for signal in self.routed_signals)
        omitted_ids = tuple(omission.signal_id for omission in self.omissions)
        if len(routed_ids) != len(set(routed_ids)):
            raise ValueError("`routed_signals` cannot repeat a signal identity.")
        if len(omitted_ids) != len(set(omitted_ids)):
            raise ValueError("`omissions` cannot repeat a signal identity.")
        if set(routed_ids) & set(omitted_ids):
            raise ValueError("A signal cannot be both routed and omitted.")
        if len(routed_ids) + len(omitted_ids) != self.signal_count:
            raise ValueError("Every submitted signal requires one exhaustive disposition.")
        candidate_references = {
            (candidate.reference.entry_id, candidate.reference.revision)
            for candidate in self.candidates
        }
        if len(candidate_references) != len(self.candidates):
            raise ValueError("`candidates` cannot repeat an exact revision.")
        if self.loaded_reference_count < len(self.candidates):
            raise ValueError("`loaded_reference_count` cannot be less than candidate count.")
        if self.loaded_reference_count > self.signal_count * 2:
            raise ValueError("`loaded_reference_count` exceeds the submitted signal bounds.")
        routed_references = {
            (reference.entry_id, reference.revision)
            for signal in self.routed_signals
            for reference in signal.references
        }
        if candidate_references != routed_references:
            raise ValueError("Candidates must exactly cover every routed signal reference.")
        for candidate in self.candidates:
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
        expected_bytes = _candidate_payload_bytes(self.candidates, self.routed_signals)
        if self.candidate_payload_bytes != expected_bytes:
            raise ValueError("`candidate_payload_bytes` does not match the routed payload.")
        incomplete_reasons = {
            KnowledgeMaintenanceRoutingOmissionReason.RELATION_COVERAGE_INCOMPLETE,
            KnowledgeMaintenanceRoutingOmissionReason.CANDIDATE_LIMIT,
            KnowledgeMaintenanceRoutingOmissionReason.CANDIDATE_BYTES,
        }
        expected_truncated = any(
            omission.reason in incomplete_reasons for omission in self.omissions
        )
        if self.truncated != expected_truncated:
            raise ValueError("`truncated` must reflect budget or coverage omissions.")
        return self

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            {
                "contract": "cayu.knowledge-maintenance-routing-result.v1",
                "result": self.model_dump(mode="json"),
            },
            "knowledge maintenance routing result",
        )


class _KnowledgeMaintenanceRoutingStore(Protocol):
    async def get_entry(
        self,
        entry_id: str,
        *,
        revision: int | None = None,
        max_bytes: int | None = None,
        access_scope: KnowledgeAccessScope,
    ) -> KnowledgeEntry | None: ...

    async def read_relations(
        self,
        query: KnowledgeRelationQuery,
        *,
        access_scope: KnowledgeAccessScope,
    ) -> KnowledgeRelationResult | None: ...


@dataclass(frozen=True, slots=True)
class _LoadedKnowledgeEntries:
    entries: dict[str, KnowledgeEntry | None]
    payload_bytes: dict[str, int]
    budget_limited: frozenset[str]


@dataclass(frozen=True, slots=True)
class _ContradictionScanState:
    signal: KnowledgeMaintenanceCandidateSignal
    cursor: str | None = None
    records_read: int = 0


@dataclass(frozen=True, slots=True)
class _ContradictionPageOutcome:
    next_state: _ContradictionScanState | None
    omission_reason: KnowledgeMaintenanceRoutingOmissionReason | None
    eligible: bool
    payload_bytes: int


class KnowledgeMaintenanceRouter:
    """Authorize, verify, and bound exact maintenance candidates without writes."""

    def __init__(
        self,
        store: _KnowledgeMaintenanceRoutingStore,
        *,
        config: KnowledgeMaintenanceRouterConfig | None = None,
    ) -> None:
        for method_name in ("get_entry", "read_relations"):
            if not callable(getattr(store, method_name, None)):
                raise TypeError("store must implement the knowledge maintenance routing surface.")
        self._store = store
        selected_config = config or KnowledgeMaintenanceRouterConfig()
        if type(selected_config) is not KnowledgeMaintenanceRouterConfig:
            raise TypeError("config must be a KnowledgeMaintenanceRouterConfig.")
        self._config = KnowledgeMaintenanceRouterConfig.model_validate(
            selected_config.model_dump(mode="python")
        )
        self._priority = {kind: index for index, kind in enumerate(self._config.signal_priority)}

    @property
    def config(self) -> KnowledgeMaintenanceRouterConfig:
        return KnowledgeMaintenanceRouterConfig.model_validate(
            self._config.model_dump(mode="python")
        )

    async def route(
        self,
        request: KnowledgeMaintenanceRoutingRequest,
    ) -> KnowledgeMaintenanceRoutingResult:
        """Route one explicit request or fail without returning a partial payload."""

        if type(request) is not KnowledgeMaintenanceRoutingRequest:
            raise TypeError("request must be a KnowledgeMaintenanceRoutingRequest.")
        copied = KnowledgeMaintenanceRoutingRequest.model_validate(
            request.model_dump(mode="python")
        )
        self._validate_request_limits(copied)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._config.timeout_seconds
        try:
            async with asyncio.timeout_at(deadline):
                if not copied.signals:
                    result = await asyncio.to_thread(
                        self._result,
                        copied,
                        {},
                        (),
                        (),
                        loaded_reference_count=0,
                        candidate_payload_bytes=_EMPTY_CANDIDATE_PAYLOAD_BYTES,
                        relation_payload_bytes=0,
                    )
                else:
                    loaded = await self._load_current_entries(copied)
                    eligible, omissions, relation_payload_bytes = await self._evaluate_signals(
                        copied,
                        loaded.entries,
                        loaded.budget_limited,
                    )
                    routed, budget_omissions, payload_bytes = await asyncio.to_thread(
                        self._apply_payload_budgets,
                        eligible,
                        loaded.payload_bytes,
                    )
                    if loop.time() >= deadline:
                        raise TimeoutError
                    all_omissions = tuple([*omissions, *budget_omissions])
                    result = await asyncio.to_thread(
                        self._result,
                        copied,
                        loaded.entries,
                        routed,
                        all_omissions,
                        loaded_reference_count=sum(
                            entry is not None for entry in loaded.entries.values()
                        ),
                        candidate_payload_bytes=payload_bytes,
                        relation_payload_bytes=relation_payload_bytes,
                    )
                if loop.time() >= deadline:
                    raise TimeoutError
        except TimeoutError as exc:
            raise KnowledgeMaintenanceRoutingTimeout() from exc
        return result

    def _validate_request_limits(self, request: KnowledgeMaintenanceRoutingRequest) -> None:
        if len(request.signals) > self._config.max_signals:
            raise KnowledgeMaintenanceRoutingLimitExceeded("max_signals")
        logical_entries = {
            reference.entry_id for signal in request.signals for reference in signal.references
        }
        if len(logical_entries) > self._config.max_candidate_reads:
            raise KnowledgeMaintenanceRoutingLimitExceeded("max_candidate_reads")

    async def _load_current_entries(
        self,
        request: KnowledgeMaintenanceRoutingRequest,
    ) -> _LoadedKnowledgeEntries:
        entry_ids: list[str] = []
        seen: set[str] = set()
        for signal in sorted(request.signals, key=self._signal_sort_key):
            for reference in signal.references:
                if reference.entry_id not in seen:
                    seen.add(reference.entry_id)
                    entry_ids.append(reference.entry_id)

        entries: dict[str, KnowledgeEntry | None] = {}
        payload_bytes: dict[str, int] = {}
        budget_limited: set[str] = set()
        available_bytes = self._config.max_candidate_load_bytes

        async def load(
            entry_id: str,
            reserved_bytes: int,
        ) -> tuple[str, KnowledgeEntry | None, int | None, bool]:
            try:
                entry = await self._store.get_entry(
                    entry_id,
                    max_bytes=reserved_bytes,
                    access_scope=request.access_scope,
                )
            except KnowledgeEntryReadLimitExceeded as exc:
                if exc.entry_id != entry_id or exc.max_bytes != reserved_bytes:
                    raise TypeError(
                        "store.get_entry() raised a read-limit error for a different request."
                    ) from exc
                return entry_id, None, None, True
            if entry is None:
                return entry_id, None, None, False
            if type(entry) is not KnowledgeEntry:
                raise TypeError("store.get_entry() must return an exact KnowledgeEntry or None.")
            if entry.id != entry_id:
                raise TypeError("store.get_entry() must return the exact requested logical entry.")
            copied = copy_knowledge_entry(entry)
            entry_bytes = await asyncio.to_thread(knowledge_entry_payload_bytes, copied)
            if entry_bytes > reserved_bytes:
                raise TypeError("store.get_entry() returned an entry beyond its max_bytes limit.")
            return entry_id, copied, entry_bytes, False

        next_entry_index = 0
        while next_entry_index < len(entry_ids) and available_bytes > 0:
            wave: list[tuple[str, int]] = []
            while (
                next_entry_index < len(entry_ids)
                and len(wave) < self._config.max_concurrency
                and available_bytes >= self._config.max_candidate_bytes
            ):
                reserved_bytes = self._config.max_candidate_bytes
                entry_id = entry_ids[next_entry_index]
                next_entry_index += 1
                available_bytes -= reserved_bytes
                wave.append((entry_id, reserved_bytes))
            if not wave:
                reserved_bytes = available_bytes
                entry_id = entry_ids[next_entry_index]
                next_entry_index += 1
                available_bytes = 0
                wave.append((entry_id, reserved_bytes))
            loaded_wave = await _gather_cancel_on_error(
                load(entry_id, reserved_bytes) for entry_id, reserved_bytes in wave
            )
            for (entry_id, reserved_bytes), (
                loaded_entry_id,
                entry,
                entry_bytes,
                limited,
            ) in zip(wave, loaded_wave, strict=True):
                if loaded_entry_id != entry_id:
                    raise RuntimeError("Candidate read accounting lost its request order.")
                entries[entry_id] = entry
                if limited:
                    budget_limited.add(entry_id)
                    available_bytes += reserved_bytes
                elif entry_bytes is None:
                    available_bytes += reserved_bytes
                else:
                    payload_bytes[entry_id] = entry_bytes
                    available_bytes += reserved_bytes - entry_bytes
        for entry_id in entry_ids[next_entry_index:]:
            entries[entry_id] = None
            budget_limited.add(entry_id)
        return _LoadedKnowledgeEntries(
            entries=entries,
            payload_bytes=payload_bytes,
            budget_limited=frozenset(budget_limited),
        )

    async def _evaluate_signals(
        self,
        request: KnowledgeMaintenanceRoutingRequest,
        entries: dict[str, KnowledgeEntry | None],
        budget_limited: frozenset[str],
    ) -> tuple[
        tuple[KnowledgeMaintenanceCandidateSignal, ...],
        tuple[KnowledgeMaintenanceRoutingOmission, ...],
        int,
    ]:
        eligible: list[KnowledgeMaintenanceCandidateSignal] = []
        omissions: list[KnowledgeMaintenanceRoutingOmission] = []
        contradiction_signals: list[KnowledgeMaintenanceCandidateSignal] = []
        for signal in sorted(request.signals, key=self._signal_sort_key):
            reason = _entry_omission_reason(
                signal,
                request,
                entries,
                budget_limited=budget_limited,
            )
            if reason is None:
                reason = _condition_omission_reason(signal, entries)
            if reason is not None:
                omissions.append(_omission(signal, reason))
            elif signal.kind is KnowledgeMaintenanceSignalKind.CONTRADICTION:
                contradiction_signals.append(signal)
            else:
                eligible.append(signal)

        pending = deque(_ContradictionScanState(signal=signal) for signal in contradiction_signals)
        available_bytes = self._config.max_relation_load_bytes
        relation_payload_bytes = 0
        while pending and available_bytes >= MAX_KNOWLEDGE_RELATION_BYTES:
            wave: list[tuple[_ContradictionScanState, int]] = []
            while (
                pending
                and len(wave) < self._config.max_concurrency
                and available_bytes >= MAX_KNOWLEDGE_RELATION_BYTES
            ):
                reserved_bytes = min(
                    self._config.relation_page_max_bytes,
                    available_bytes,
                )
                available_bytes -= reserved_bytes
                wave.append((pending.popleft(), reserved_bytes))
            outcomes = await _gather_cancel_on_error(
                self._read_contradiction_page(
                    request,
                    state,
                    max_bytes=reserved_bytes,
                )
                for state, reserved_bytes in wave
            )
            next_states: list[_ContradictionScanState] = []
            for (state, reserved_bytes), outcome in zip(wave, outcomes, strict=True):
                if outcome.payload_bytes > reserved_bytes:
                    raise RuntimeError("Relation read accounting exceeded its reservation.")
                relation_payload_bytes += outcome.payload_bytes
                available_bytes += reserved_bytes - outcome.payload_bytes
                if outcome.eligible:
                    eligible.append(state.signal)
                elif outcome.omission_reason is not None:
                    omissions.append(_omission(state.signal, outcome.omission_reason))
                elif outcome.next_state is not None:
                    next_states.append(outcome.next_state)
                else:
                    raise RuntimeError("Contradiction scan produced no disposition.")
            pending.extend(next_states)
        omissions.extend(
            _omission(
                state.signal,
                KnowledgeMaintenanceRoutingOmissionReason.RELATION_COVERAGE_INCOMPLETE,
            )
            for state in pending
        )
        eligible.sort(key=self._signal_sort_key)
        omissions.sort(key=lambda item: self._omission_sort_key(item, request.signals))
        return tuple(eligible), tuple(omissions), relation_payload_bytes

    async def _read_contradiction_page(
        self,
        request: KnowledgeMaintenanceRoutingRequest,
        state: _ContradictionScanState,
        *,
        max_bytes: int,
    ) -> _ContradictionPageOutcome:
        remaining = self._config.max_relation_records_per_signal - state.records_read
        if remaining <= 0:
            return _ContradictionPageOutcome(
                next_state=None,
                omission_reason=(
                    KnowledgeMaintenanceRoutingOmissionReason.RELATION_COVERAGE_INCOMPLETE
                ),
                eligible=False,
                payload_bytes=0,
            )
        query = KnowledgeRelationQuery(
            reference=state.signal.references[0],
            kinds=[KnowledgeRelationKind.CONTRADICTS],
            limit=min(self._config.relation_page_limit, remaining),
            max_bytes=max_bytes,
            cursor=state.cursor,
        )
        result = await self._store.read_relations(
            query,
            access_scope=request.access_scope,
        )
        if result is None:
            return _ContradictionPageOutcome(
                next_state=None,
                omission_reason=(
                    KnowledgeMaintenanceRoutingOmissionReason.RELATION_COVERAGE_INCOMPLETE
                ),
                eligible=False,
                payload_bytes=0,
            )
        return await asyncio.to_thread(
            self._evaluate_contradiction_page,
            state,
            query,
            result,
        )

    def _evaluate_contradiction_page(
        self,
        state: _ContradictionScanState,
        query: KnowledgeRelationQuery,
        result: KnowledgeRelationResult,
    ) -> _ContradictionPageOutcome:
        if type(result) is not KnowledgeRelationResult:
            raise TypeError(
                "store.read_relations() must return an exact KnowledgeRelationResult or None."
            )
        copied = KnowledgeRelationResult.model_validate(result.model_dump(mode="python"))
        if copied.query != query:
            raise TypeError(
                "store.read_relations() must bind its result to the exact submitted query."
            )
        payload_bytes = _relation_page_payload_bytes(copied.relations)
        signal = state.signal
        expected = {(reference.entry_id, reference.revision) for reference in signal.references}
        for relation in copied.relations:
            if relation.id != signal.relation_id:
                continue
            endpoints = {
                (relation.subject.entry_id, relation.subject.revision),
                (relation.object.entry_id, relation.object.revision),
            }
            reason = (
                None
                if endpoints == expected and relation.created_at <= signal.observed_at
                else KnowledgeMaintenanceRoutingOmissionReason.CONDITION_NOT_MET
            )
            return _ContradictionPageOutcome(
                next_state=None,
                omission_reason=reason,
                eligible=reason is None,
                payload_bytes=payload_bytes,
            )
        records_read = state.records_read + len(copied.relations)
        if not copied.truncated:
            reason = KnowledgeMaintenanceRoutingOmissionReason.CONDITION_NOT_MET
            next_state = None
        elif (
            copied.next_cursor is None
            or not copied.relations
            or records_read >= self._config.max_relation_records_per_signal
        ):
            reason = KnowledgeMaintenanceRoutingOmissionReason.RELATION_COVERAGE_INCOMPLETE
            next_state = None
        else:
            reason = None
            next_state = _ContradictionScanState(
                signal=signal,
                cursor=copied.next_cursor,
                records_read=records_read,
            )
        return _ContradictionPageOutcome(
            next_state=next_state,
            omission_reason=reason,
            eligible=False,
            payload_bytes=payload_bytes,
        )

    def _apply_payload_budgets(
        self,
        eligible: tuple[KnowledgeMaintenanceCandidateSignal, ...],
        entry_payload_bytes: dict[str, int],
    ) -> tuple[
        tuple[KnowledgeMaintenanceCandidateSignal, ...],
        tuple[KnowledgeMaintenanceRoutingOmission, ...],
        int,
    ]:
        routed: list[KnowledgeMaintenanceCandidateSignal] = []
        omissions: list[KnowledgeMaintenanceRoutingOmission] = []
        references: dict[tuple[str, int], KnowledgeRevisionRef] = {}
        candidate_sizes: dict[tuple[str, int], int] = {}
        candidate_support_counts: dict[tuple[str, int], int] = {}
        candidate_signal_kinds: dict[tuple[str, int], set[KnowledgeMaintenanceSignalKind]] = {}
        candidate_body_bytes = 0
        signal_body_bytes = 0
        for signal in eligible:
            signal_references = {
                (reference.entry_id, reference.revision): reference
                for reference in signal.references
            }
            new_reference_count = sum(
                reference_key not in references for reference_key in signal_references
            )
            if len(references) + new_reference_count > self._config.max_candidates:
                omissions.append(
                    _omission(
                        signal,
                        KnowledgeMaintenanceRoutingOmissionReason.CANDIDATE_LIMIT,
                    )
                )
                continue
            trial_candidate_body_bytes = candidate_body_bytes
            trial_sizes: dict[tuple[str, int], int] = {}
            signal_id_bytes = _json_scalar_payload_bytes(signal.id)
            signal_kind_bytes = _json_scalar_payload_bytes(signal.kind.value)
            for reference_key, reference in signal_references.items():
                entry_bytes = entry_payload_bytes.get(reference.entry_id)
                if entry_bytes is None:
                    raise RuntimeError("Eligible routing state lost entry payload accounting.")
                support_count = candidate_support_counts.get(reference_key, 0)
                kinds = candidate_signal_kinds.get(reference_key, set())
                current_size = candidate_sizes.get(
                    reference_key,
                    _routed_candidate_base_payload_bytes(
                        reference,
                        entry_payload_bytes=entry_bytes,
                    ),
                )
                trial_size = current_size + signal_id_bytes + (1 if support_count else 0)
                if signal.kind not in kinds:
                    trial_size += signal_kind_bytes + (1 if kinds else 0)
                trial_candidate_body_bytes += trial_size - candidate_sizes.get(reference_key, 0)
                trial_sizes[reference_key] = trial_size
            trial_signal_body_bytes = signal_body_bytes + _routed_signal_payload_bytes(signal)
            trial_reference_count = len(references) + new_reference_count
            trial_signal_count = len(routed) + 1
            trial_payload_bytes = (
                _EMPTY_CANDIDATE_PAYLOAD_BYTES
                + trial_candidate_body_bytes
                + max(0, trial_reference_count - 1)
                + trial_signal_body_bytes
                + max(0, trial_signal_count - 1)
            )
            if trial_payload_bytes > self._config.max_candidate_bytes:
                omissions.append(
                    _omission(
                        signal,
                        KnowledgeMaintenanceRoutingOmissionReason.CANDIDATE_BYTES,
                    )
                )
                continue
            routed.append(signal)
            candidate_body_bytes = trial_candidate_body_bytes
            signal_body_bytes = trial_signal_body_bytes
            for reference_key, reference in signal_references.items():
                references.setdefault(reference_key, reference)
                candidate_support_counts[reference_key] = (
                    candidate_support_counts.get(reference_key, 0) + 1
                )
                candidate_signal_kinds.setdefault(reference_key, set()).add(signal.kind)
                candidate_sizes[reference_key] = trial_sizes[reference_key]
        payload_bytes = (
            _EMPTY_CANDIDATE_PAYLOAD_BYTES
            + candidate_body_bytes
            + max(0, len(references) - 1)
            + signal_body_bytes
            + max(0, len(routed) - 1)
        )
        return tuple(routed), tuple(omissions), payload_bytes

    def _build_candidates(
        self,
        signals: Iterable[KnowledgeMaintenanceCandidateSignal],
        entries: dict[str, KnowledgeEntry | None],
    ) -> tuple[KnowledgeMaintenanceRoutedCandidate, ...]:
        references: dict[tuple[str, int], KnowledgeRevisionRef] = {}
        supporting_signals: dict[tuple[str, int], list[KnowledgeMaintenanceCandidateSignal]] = {}
        for signal in signals:
            for reference in signal.references:
                reference_key = reference.entry_id, reference.revision
                if reference_key not in references:
                    references[reference_key] = reference
                    supporting_signals[reference_key] = []
                supporting_signals[reference_key].append(signal)
        candidates: list[KnowledgeMaintenanceRoutedCandidate] = []
        for reference_key, reference in references.items():
            entry = entries[reference.entry_id]
            if entry is None:
                raise RuntimeError("Eligible routing state lost an authorized entry.")
            supporting = tuple(supporting_signals[reference_key])
            kinds = tuple(
                sorted(
                    {signal.kind for signal in supporting},
                    key=lambda kind: self._priority[kind],
                )
            )
            candidates.append(
                KnowledgeMaintenanceRoutedCandidate(
                    reference=reference,
                    entry=entry,
                    signal_ids=tuple(signal.id for signal in supporting),
                    signal_kinds=kinds,
                )
            )
        candidates.sort(
            key=lambda candidate: (
                min(self._priority[kind] for kind in candidate.signal_kinds),
                candidate.reference.entry_id,
                candidate.reference.revision,
            )
        )
        return tuple(candidates)

    def _result(
        self,
        request: KnowledgeMaintenanceRoutingRequest,
        entries: dict[str, KnowledgeEntry | None],
        routed: tuple[KnowledgeMaintenanceCandidateSignal, ...],
        omissions: tuple[KnowledgeMaintenanceRoutingOmission, ...],
        *,
        loaded_reference_count: int,
        candidate_payload_bytes: int,
        relation_payload_bytes: int,
    ) -> KnowledgeMaintenanceRoutingResult:
        candidates = self._build_candidates(routed, entries) if routed else ()
        ordered_omissions = tuple(
            sorted(
                omissions,
                key=lambda item: self._omission_sort_key(item, request.signals),
            )
        )
        incomplete_reasons = {
            KnowledgeMaintenanceRoutingOmissionReason.RELATION_COVERAGE_INCOMPLETE,
            KnowledgeMaintenanceRoutingOmissionReason.CANDIDATE_LIMIT,
            KnowledgeMaintenanceRoutingOmissionReason.CANDIDATE_BYTES,
        }
        return KnowledgeMaintenanceRoutingResult(
            request_id=request.id,
            request_fingerprint=request.fingerprint,
            configuration_fingerprint=self._config.fingerprint,
            candidates=candidates,
            routed_signals=routed,
            omissions=ordered_omissions,
            signal_count=len(request.signals),
            loaded_reference_count=loaded_reference_count,
            candidate_payload_bytes=candidate_payload_bytes,
            relation_payload_bytes=relation_payload_bytes,
            max_candidates=self._config.max_candidates,
            max_candidate_bytes=self._config.max_candidate_bytes,
            max_relation_load_bytes=self._config.max_relation_load_bytes,
            truncated=any(omission.reason in incomplete_reasons for omission in ordered_omissions),
        )

    def _signal_sort_key(
        self,
        signal: KnowledgeMaintenanceCandidateSignal,
    ) -> tuple[int, datetime, str]:
        return self._priority[signal.kind], signal.observed_at, signal.id

    def _omission_sort_key(
        self,
        omission: KnowledgeMaintenanceRoutingOmission,
        signals: Iterable[KnowledgeMaintenanceCandidateSignal],
    ) -> tuple[int, datetime, str]:
        signal_by_id = {signal.id: signal for signal in signals}
        signal = signal_by_id[omission.signal_id]
        return self._signal_sort_key(signal)


def _entry_omission_reason(
    signal: KnowledgeMaintenanceCandidateSignal,
    request: KnowledgeMaintenanceRoutingRequest,
    entries: dict[str, KnowledgeEntry | None],
    *,
    budget_limited: frozenset[str],
) -> KnowledgeMaintenanceRoutingOmissionReason | None:
    selected: list[KnowledgeEntry] = []
    for reference in signal.references:
        if reference.entry_id in budget_limited:
            return KnowledgeMaintenanceRoutingOmissionReason.CANDIDATE_BYTES
        entry = entries[reference.entry_id]
        if entry is None:
            return KnowledgeMaintenanceRoutingOmissionReason.UNAVAILABLE
        if entry.revision != reference.revision:
            return KnowledgeMaintenanceRoutingOmissionReason.STALE_REVISION
        selected.append(entry)
    for entry in selected:
        if entry.namespace != request.namespace or any(
            entry.labels.get(key) != value for key, value in request.labels.items()
        ):
            return KnowledgeMaintenanceRoutingOmissionReason.SCOPE_MISMATCH
        if entry.status is not KnowledgeStatus.ACTIVE:
            return KnowledgeMaintenanceRoutingOmissionReason.LIFECYCLE_MISMATCH
    return None


def _condition_omission_reason(
    signal: KnowledgeMaintenanceCandidateSignal,
    entries: dict[str, KnowledgeEntry | None],
) -> KnowledgeMaintenanceRoutingOmissionReason | None:
    entry = entries[signal.references[0].entry_id]
    if entry is None:
        raise RuntimeError("Condition evaluation lost an authorized entry.")
    if signal.kind is KnowledgeMaintenanceSignalKind.EXPIRY:
        threshold = signal.threshold_at
        if threshold is None:
            raise RuntimeError("Validated expiry signal lost its threshold.")
        if entry.expires_at is None or entry.expires_at > threshold:
            return KnowledgeMaintenanceRoutingOmissionReason.CONDITION_NOT_MET
    elif signal.kind is KnowledgeMaintenanceSignalKind.LOW_USAGE:
        threshold = signal.threshold_at
        if threshold is None:
            raise RuntimeError("Validated low-usage signal lost its threshold.")
        latest_activity = max(
            timestamp
            for timestamp in (entry.created_at, entry.updated_at, entry.last_used_at)
            if timestamp is not None
        )
        if latest_activity > threshold:
            return KnowledgeMaintenanceRoutingOmissionReason.CONDITION_NOT_MET
    return None


def _omission(
    signal: KnowledgeMaintenanceCandidateSignal,
    reason: KnowledgeMaintenanceRoutingOmissionReason,
) -> KnowledgeMaintenanceRoutingOmission:
    return KnowledgeMaintenanceRoutingOmission(
        signal_id=signal.id,
        signal_kind=signal.kind,
        reason=reason,
    )


def _ordered_unique_identities(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list | tuple) or not value:
        raise ValueError(f"`{field_name}` must be a non-empty ordered array.")
    copied: list[str] = []
    for item in value:
        if type(item) is not str:
            raise ValueError(f"`{field_name}` must contain only strings.")
        copied.append(_clean(item, field_name))
    if len(copied) != len(set(copied)):
        raise ValueError(f"`{field_name}` cannot contain duplicates.")
    return tuple(copied)


def _copy_models(value: object, model_type: type[_RoutingModel], field_name: str) -> tuple:
    if not isinstance(value, list | tuple):
        raise ValueError(f"`{field_name}` must be an ordered array.")
    return tuple(
        model_type.model_validate(
            item.model_dump(mode="python") if isinstance(item, model_type) else item
        )
        for item in value
    )


def _candidate_payload_bytes(
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


def _routed_signal_payload_bytes(signal: KnowledgeMaintenanceCandidateSignal) -> int:
    return len(
        canonical_durable_json_bytes(
            signal.model_dump(mode="json"),
            "knowledge maintenance routed signal",
        )
    )


def _json_scalar_payload_bytes(value: str) -> int:
    return len(canonical_durable_json_bytes(value, "knowledge maintenance routed scalar"))


def _routed_candidate_base_payload_bytes(
    reference: KnowledgeRevisionRef,
    *,
    entry_payload_bytes: int,
) -> int:
    skeleton = canonical_durable_json_bytes(
        {
            "reference": reference.model_dump(mode="json"),
            "entry": None,
            "signal_ids": [],
            "signal_kinds": [],
        },
        "knowledge maintenance routed candidate",
    )
    return len(skeleton) - len(b"null") + entry_payload_bytes


def _relation_page_payload_bytes(relations: Iterable[KnowledgeRelation]) -> int:
    return sum(
        len(
            canonical_durable_json_bytes(
                relation.model_dump(mode="json"),
                "knowledge relation",
            )
        )
        for relation in relations
    )


_EMPTY_CANDIDATE_PAYLOAD_BYTES = len(
    canonical_durable_json_bytes(
        {
            "contract": "cayu.knowledge-maintenance-routed-payload.v1",
            "candidates": [],
            "signals": [],
        },
        "empty knowledge maintenance routed payload",
    )
)


def _signal_semantic_fingerprint(signal: KnowledgeMaintenanceCandidateSignal) -> str:
    material = signal.model_dump(mode="json")
    material.pop("id")
    material.pop("raw_score")
    return _fingerprint(
        {
            "contract": "cayu.knowledge-maintenance-signal-observation.v1",
            "signal": material,
        },
        "knowledge maintenance signal observation",
    )


async def _gather_cancel_on_error(awaitables: Iterable[Awaitable[_T]]) -> tuple[_T, ...]:
    tasks = tuple(asyncio.ensure_future(awaitable) for awaitable in awaitables)
    try:
        return tuple(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
