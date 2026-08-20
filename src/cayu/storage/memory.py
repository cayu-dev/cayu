from __future__ import annotations

import base64
import binascii
import json
import re
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from math import sqrt
from typing import Any, Literal, TypedDict
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cayu._clock import utc_clock
from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_json_object,
    copy_durable_json_value,
    copy_json_value,
    copy_label_map,
    require_finite,
)
from cayu._validation import (
    require_durable_clean_nonblank as require_clean_nonblank,
)
from cayu._validation import (
    require_durable_nonblank as require_nonblank,
)
from cayu.embeddings import (
    TextEmbeddingProvider,
    TextEmbeddingRequest,
    copy_text_embedding_result,
)

DEFAULT_KNOWLEDGE_NAMESPACE = "default"
DEFAULT_KNOWLEDGE_KIND = "fact"
DEFAULT_KNOWLEDGE_LIMIT = 10
DEFAULT_KNOWLEDGE_MAX_BYTES = 20_000
DEFAULT_KNOWLEDGE_EMBEDDING_WORK_RECORD_LIMIT = 500
MAX_KNOWLEDGE_CHANGE_LIMIT = 1_000
MAX_KNOWLEDGE_CHANGE_SEQUENCE = 2**63 - 1
MAX_KNOWLEDGE_CHUNK_ID_BYTES = 512
MAX_KNOWLEDGE_CHUNK_INDEX = 2**31 - 1
MAX_KNOWLEDGE_ENTRY_ID_BYTES = 256
MAX_KNOWLEDGE_REVISION = 2**31 - 1
MAX_KNOWLEDGE_EVIDENCE_BYTES = DEFAULT_KNOWLEDGE_MAX_BYTES
MAX_KNOWLEDGE_EVIDENCE_JSON_BYTES = 16_384
MAX_KNOWLEDGE_EMBEDDING_DIMENSIONS = 2**31 - 1
MAX_KNOWLEDGE_INDEX_READINESS_LIMIT = 1_000
MAX_KNOWLEDGE_EMBEDDING_WORK_RECORD_LIMIT = 10_000
_MAX_KNOWLEDGE_EMBEDDING_BACKFILL_CURSOR_BYTES = 2_048
_SEARCH_TOKEN_RE = re.compile(r"\w+")
_SHA256_HEX_RE = re.compile(r"[0-9a-f]{64}\Z")
_KNOWLEDGE_EMBEDDING_BACKFILL_CURSOR_VERSION = 1
KNOWLEDGE_CHUNK_TEXT_PROJECTION = "knowledge_chunk_text"
KNOWLEDGE_CHUNK_TEXT_PREPROCESSING_VERSION = "cayu:knowledge-chunk-text:v1"
KNOWLEDGE_CHUNK_TEXT_GENERATOR = "cayu:canonical-knowledge-chunk"
KNOWLEDGE_CHUNK_TEXT_GENERATOR_VERSION = "1"
KNOWLEDGE_VECTOR_INDEX_REPRESENTATION_VERSION = "float32-cosine-v1"

BUILTIN_KNOWLEDGE_KINDS = (
    "fact",
    "preference",
    "procedure",
    "instruction",
    "skill",
    "document",
    "example",
    "warning",
    "decision",
    "event",
    "summary",
)


class _SearchTerms(TypedDict):
    any: list[str]
    all: list[list[str]]
    none: list[str]
    phrases: list[list[str]]


class _StoredChunkEmbedding(TypedDict):
    identity: KnowledgeEmbeddingIdentity
    vector: list[float]
    vector_sha256: str
    readiness_sequence: int
    attempt_id: str


class KnowledgeStatus(StrEnum):
    ACTIVE = "active"
    PENDING = "pending"
    ARCHIVED = "archived"
    DELETED = "deleted"


_KNOWLEDGE_RETIREMENT_STATUSES = frozenset({KnowledgeStatus.ARCHIVED, KnowledgeStatus.DELETED})


class KnowledgeVisibility(StrEnum):
    GLOBAL = "global"
    ORGANIZATION = "organization"
    PROJECT = "project"
    WORKSPACE = "workspace"
    USER = "user"
    SESSION = "session"
    TASK = "task"


class KnowledgeActorType(StrEnum):
    APP = "app"
    USER = "user"
    MODEL = "model"
    SYSTEM = "system"


class KnowledgeSearchMode(StrEnum):
    AUTO = "auto"
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    EXTERNAL = "external"


class KnowledgeEvidenceRole(StrEnum):
    ORIGIN = "origin"
    SUPPORTING = "supporting"


class KnowledgeEvidenceDisposition(StrEnum):
    LIVE = "live"
    DETACHED = "detached"
    RETAINED = "retained"


class KnowledgeChangeKind(StrEnum):
    CREATED = "created"
    REVISION_APPENDED = "revision_appended"
    STATUS_TRANSITIONED = "status_transitioned"
    TOMBSTONED = "tombstoned"
    HARD_DELETED = "hard_deleted"
    EXPIRED = "expired"


class KnowledgeIndexState(StrEnum):
    """Publication state for one exact derived-index identity."""

    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"


class KnowledgeListGroup(StrEnum):
    KIND = "kind"
    LABEL = "label"
    ASPECT = "aspect"
    IMPACT_TARGET = "impact_target"
    VISIBILITY = "visibility"
    SOURCE_TYPE = "source_type"
    NAMESPACE = "namespace"


class KnowledgeAccessDenied(PermissionError):
    """Raised when a knowledge mutation falls outside its explicit access scope."""

    def __init__(self, operation: str) -> None:
        self.operation = require_clean_nonblank(operation, "operation")
        super().__init__(f"Knowledge access denied for {self.operation}.")


class KnowledgeChunkConflict(RuntimeError):
    """A knowledge write conflicts with an occupied global chunk identity."""

    def __init__(self, operation: str) -> None:
        self.operation = require_clean_nonblank(operation, "operation")
        super().__init__("Knowledge chunk identity conflicts with durable state.")


class KnowledgeEvidenceConflict(RuntimeError):
    """A knowledge write conflicts with an occupied global evidence identity."""

    def __init__(self, operation: str) -> None:
        self.operation = require_clean_nonblank(operation, "operation")
        super().__init__("Knowledge evidence identity conflicts with durable state.")


class KnowledgeRevisionConflict(RuntimeError):
    """A canonical write lost a compare-and-swap race."""

    def __init__(
        self,
        entry_id: str,
        *,
        expected_revision: int | None,
        actual_revision: int | None,
    ) -> None:
        self.entry_id = _knowledge_entry_id(entry_id)
        if expected_revision is not None:
            _validate_knowledge_revision(expected_revision, "expected_revision")
        if actual_revision is not None:
            _validate_knowledge_revision(actual_revision, "actual_revision")
        self.expected_revision = expected_revision
        self.actual_revision = actual_revision
        super().__init__(
            f"Knowledge entry {self.entry_id!r} revision conflict: expected "
            f"{self.expected_revision!r}, found {self.actual_revision!r}."
        )


class KnowledgeChangeConsumerConflict(RuntimeError):
    """A knowledge-change consumer or lease fence conflicts with durable state."""

    def __init__(self, reason: str) -> None:
        self.reason = require_clean_nonblank(reason, "reason")
        super().__init__("Knowledge change consumer conflicts with durable state.")


class KnowledgeIndexReadinessConflict(RuntimeError):
    """A readiness publication conflicts with its identity or sequence fence."""

    def __init__(self, reason: str) -> None:
        self.reason = require_clean_nonblank(reason, "reason")
        super().__init__("Knowledge index readiness conflicts with durable state.")


class KnowledgeEmbeddingProjectionConflict(RuntimeError):
    """A projection attempt was reused with a different immutable vector payload."""

    def __init__(self, reason: str) -> None:
        self.reason = require_clean_nonblank(reason, "reason")
        super().__init__("Knowledge embedding projection conflicts with durable state.")


class KnowledgeAccessScope(BaseModel):
    """Principal-derived constraints enforced inside every knowledge operation.

    Cayu deliberately does not model tenants, organizations, users, or RBAC. The
    hosting application maps those concepts into namespaces, labels, visibility,
    source identity, lifecycle state, and expiration eligibility. Namespace-wide
    access must be explicit; constructing a scope with no namespace is invalid.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    allowed_namespaces: list[str] = Field(default_factory=list)
    allow_all_namespaces: bool = False
    required_labels: dict[str, str] = Field(default_factory=dict)
    allowed_visibilities: list[KnowledgeVisibility] = Field(
        default_factory=lambda: [KnowledgeVisibility.GLOBAL]
    )
    allowed_source_types: list[str] | None = None
    allowed_source_ids: list[str] | None = None
    allowed_statuses: list[KnowledgeStatus] = Field(
        default_factory=lambda: [KnowledgeStatus.ACTIVE]
    )
    include_expired: bool = False

    @field_validator(
        "allowed_namespaces", "allowed_source_types", "allowed_source_ids", mode="before"
    )
    @classmethod
    def copy_string_lists(cls, value, info) -> list[str] | None:
        if value is None and info.field_name in {"allowed_source_types", "allowed_source_ids"}:
            return None
        if value is None:
            return []
        copied = copy_json_value(value, info.field_name)
        if type(copied) is not list:
            raise ValueError(f"`{info.field_name}` must be a list.")
        result: list[str] = []
        for index, item in enumerate(copied):
            if type(item) is not str:
                raise ValueError(f"`{info.field_name}[{index}]` must be a string.")
            result.append(require_clean_nonblank(item, f"{info.field_name}[{index}]"))
        return sorted(_dedupe_strings(result))

    @field_validator("required_labels", mode="before")
    @classmethod
    def copy_required_labels(cls, value) -> dict[str, str]:
        return copy_label_map(value, "required_labels")

    @field_validator("allowed_visibilities", "allowed_statuses")
    @classmethod
    def validate_nonempty_enum_lists(cls, value: list[Any], info) -> list[Any]:
        if not value:
            raise ValueError(f"`{info.field_name}` cannot be empty.")
        return sorted(dict.fromkeys(value), key=str)

    @field_validator("allow_all_namespaces", "include_expired", mode="before")
    @classmethod
    def validate_boolean_fields(cls, value, info) -> bool:
        if type(value) is not bool:
            raise ValueError(f"`{info.field_name}` must be a boolean.")
        return value

    @model_validator(mode="after")
    def validate_namespace_boundary(self) -> KnowledgeAccessScope:
        if self.allow_all_namespaces and self.allowed_namespaces:
            raise ValueError(
                "`allowed_namespaces` must be empty when `allow_all_namespaces` is true."
            )
        if not self.allow_all_namespaces and not self.allowed_namespaces:
            raise ValueError(
                "Knowledge access requires `allowed_namespaces` or explicit "
                "`allow_all_namespaces=True`."
            )
        return self

    @classmethod
    def for_namespace(
        cls,
        namespace: str,
        *,
        required_labels: dict[str, str] | None = None,
        allowed_visibilities: list[KnowledgeVisibility] | None = None,
        allowed_source_types: list[str] | None = None,
        allowed_source_ids: list[str] | None = None,
        allowed_statuses: list[KnowledgeStatus] | None = None,
        include_expired: bool = False,
    ) -> KnowledgeAccessScope:
        """Create an explicit single-namespace application scope."""

        values: dict[str, Any] = {
            "allowed_namespaces": [namespace],
            "required_labels": required_labels or {},
            "allowed_source_types": allowed_source_types,
            "allowed_source_ids": allowed_source_ids,
            "include_expired": include_expired,
        }
        if allowed_visibilities is not None:
            values["allowed_visibilities"] = allowed_visibilities
        if allowed_statuses is not None:
            values["allowed_statuses"] = allowed_statuses
        return cls(**values)

    @classmethod
    def privileged(cls) -> KnowledgeAccessScope:
        """Create an explicit all-knowledge scope for trusted host maintenance."""

        return cls(
            allow_all_namespaces=True,
            allowed_visibilities=list(KnowledgeVisibility),
            allowed_statuses=list(KnowledgeStatus),
            include_expired=True,
        )


class KnowledgeEntry(BaseModel):
    """Immutable snapshot of one exact logical knowledge revision."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    id: str
    revision: int = 1
    text: str
    namespace: str = DEFAULT_KNOWLEDGE_NAMESPACE
    labels: dict[str, str] = Field(default_factory=dict)
    kind: str = DEFAULT_KNOWLEDGE_KIND
    visibility: KnowledgeVisibility = KnowledgeVisibility.GLOBAL
    status: KnowledgeStatus = KnowledgeStatus.ACTIVE
    created_by_type: KnowledgeActorType = KnowledgeActorType.APP
    created_by: str = "app"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_type: str | None = None
    source_uri: str | None = None
    source_id: str | None = None
    source_hash: str | None = None
    aspects: list[str] = Field(default_factory=list)
    impact_targets: list[str] = Field(default_factory=list)
    importance: float | None = None
    importance_source: str | None = None
    confidence: float | None = None
    last_used_at: datetime | None = None
    expires_at: datetime | None = None
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_durable_json_object(value, "metadata")

    @field_validator("labels", mode="before")
    @classmethod
    def copy_labels(cls, value) -> dict[str, str]:
        return copy_label_map(value, "labels")

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _knowledge_entry_id(value, "id")

    @field_validator("namespace", "kind", "created_by")
    @classmethod
    def validate_clean_nonblank_fields(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("revision")
    @classmethod
    def validate_revision(cls, value: int) -> int:
        _validate_knowledge_revision(value, "revision")
        return value

    @field_validator("text")
    @classmethod
    def validate_nonblank_text(cls, value: str, info) -> str:
        return require_nonblank(value, info.field_name)

    @field_validator(
        "source_type",
        "source_uri",
        "source_id",
        "source_hash",
        "importance_source",
        "title",
    )
    @classmethod
    def validate_optional_clean_nonblank_fields(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("aspects", "impact_targets", mode="before")
    @classmethod
    def copy_string_list(cls, value, info) -> list[str]:
        if value is None:
            return []
        copied = copy_durable_json_value(value, info.field_name)
        if type(copied) is not list:
            raise ValueError(f"`{info.field_name}` must be a list.")
        result: list[str] = []
        for index, item in enumerate(copied):
            if type(item) is not str:
                raise ValueError(f"`{info.field_name}[{index}]` must be a string.")
            result.append(require_clean_nonblank(item, info.field_name))
        return _dedupe_strings(result)

    @field_validator("importance", "confidence", mode="before")
    @classmethod
    def validate_optional_unit_interval(cls, value, info) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"`{info.field_name}` must be a number.")
        value = require_finite(float(value), info.field_name)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"`{info.field_name}` must be between 0.0 and 1.0.")
        return value

    @field_validator("created_at", "updated_at", "last_used_at", "expires_at")
    @classmethod
    def validate_timezone_aware_datetime(cls, value: datetime | None, info) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"`{info.field_name}` must be timezone-aware.")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_timestamp_order(self) -> KnowledgeEntry:
        if self.updated_at < self.created_at:
            raise ValueError("`updated_at` must be greater than or equal to `created_at`.")
        return self


class _KnowledgeAccessSnapshot(BaseModel):
    """Immutable authorization projection retained beside publication receipts."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    namespace: str
    labels: dict[str, str]
    visibility: KnowledgeVisibility
    source_type: str | None
    source_id: str | None
    status: KnowledgeStatus
    expires_at: datetime | None

    @field_validator("namespace")
    @classmethod
    def validate_namespace(cls, value: str) -> str:
        return require_clean_nonblank(value, "namespace")

    @field_validator("source_type", "source_id")
    @classmethod
    def validate_optional_identity(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("labels", mode="before")
    @classmethod
    def copy_labels(cls, value) -> dict[str, str]:
        return copy_label_map(value, "labels")


class _KnowledgeChangeAudience(BaseModel):
    """One immutable before/after authorization audience for a change."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    kind: Literal["before", "after"]
    snapshot: _KnowledgeAccessSnapshot
    requires_include_expired: bool = False

    @field_validator("snapshot", mode="before")
    @classmethod
    def copy_snapshot(cls, value: _KnowledgeAccessSnapshot) -> _KnowledgeAccessSnapshot:
        if type(value) is not _KnowledgeAccessSnapshot:
            raise TypeError("Knowledge change audiences require an access snapshot.")
        return value.model_copy(deep=True)

    @field_validator("requires_include_expired", mode="before")
    @classmethod
    def validate_requires_include_expired(cls, value) -> bool:
        if type(value) is not bool:
            raise ValueError("`requires_include_expired` must be a boolean.")
        return value


class KnowledgeChunk(BaseModel):
    """Immutable chunk belonging to one exact knowledge revision."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    id: str
    entry_id: str
    entry_revision: int = 1
    text: str
    chunk_index: int
    content_hash: str | None = None
    source_uri: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_durable_json_object(value, "metadata")

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _knowledge_chunk_id(value, "id")

    @field_validator("entry_id")
    @classmethod
    def validate_clean_nonblank_fields(cls, value: str, info) -> str:
        return _knowledge_entry_id(value, info.field_name)

    @field_validator("text")
    @classmethod
    def validate_nonblank_text(cls, value: str, info) -> str:
        return require_nonblank(value, info.field_name)

    @field_validator("content_hash", "source_uri")
    @classmethod
    def validate_optional_clean_nonblank_fields(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("chunk_index")
    @classmethod
    def validate_chunk_index(cls, value: int, info) -> int:
        if isinstance(value, bool) or type(value) is not int:
            raise ValueError(f"`{info.field_name}` must be an integer.")
        if value < 0:
            raise ValueError(f"`{info.field_name}` must be greater than or equal to 0.")
        if value > MAX_KNOWLEDGE_CHUNK_INDEX:
            raise ValueError(
                f"`{info.field_name}` must be less than or equal to {MAX_KNOWLEDGE_CHUNK_INDEX}."
            )
        return value

    @field_validator("entry_revision")
    @classmethod
    def validate_entry_revision(cls, value: int) -> int:
        _validate_knowledge_revision(value, "entry_revision")
        return value


class KnowledgeEvidence(BaseModel):
    """Immutable exact source evidence for one knowledge revision."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    id: str
    entry_id: str
    entry_revision: int = 1
    chunk_id: str | None = None
    role: KnowledgeEvidenceRole = KnowledgeEvidenceRole.ORIGIN
    source_type: str
    source_id: str | None = None
    source_uri: str | None = None
    source_revision: str | None = None
    source_hash: str | None = None
    locator: dict[str, Any] = Field(default_factory=dict)
    disposition: KnowledgeEvidenceDisposition = KnowledgeEvidenceDisposition.LIVE
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "source_type")
    @classmethod
    def validate_required_identity(cls, value: str, info) -> str:
        value = require_clean_nonblank(value, info.field_name)
        if len(value.encode("utf-8")) > 256:
            raise ValueError(f"`{info.field_name}` must be at most 256 UTF-8 bytes.")
        return value

    @field_validator("entry_id")
    @classmethod
    def validate_entry_id(cls, value: str) -> str:
        return _knowledge_entry_id(value)

    @field_validator(
        "source_id",
        "source_uri",
        "source_revision",
        "source_hash",
    )
    @classmethod
    def validate_optional_identity(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        value = require_clean_nonblank(value, info.field_name)
        limit = 256 if info.field_name == "source_id" else 2048
        if len(value.encode("utf-8")) > limit:
            raise ValueError(f"`{info.field_name}` must be at most {limit} UTF-8 bytes.")
        return value

    @field_validator("chunk_id")
    @classmethod
    def validate_chunk_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _knowledge_chunk_id(value)

    @field_validator("entry_revision")
    @classmethod
    def validate_entry_revision(cls, value: int) -> int:
        _validate_knowledge_revision(value, "entry_revision")
        return value

    @field_validator("locator", "metadata", mode="before")
    @classmethod
    def copy_json_objects(cls, value: dict[str, Any], info) -> dict[str, Any]:
        copied = copy_durable_json_object(value, info.field_name)
        if len(canonical_durable_json_bytes(copied, info.field_name)) > (
            MAX_KNOWLEDGE_EVIDENCE_JSON_BYTES
        ):
            raise ValueError(
                f"`{info.field_name}` must be at most "
                f"{MAX_KNOWLEDGE_EVIDENCE_JSON_BYTES} canonical UTF-8 bytes."
            )
        return copied

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("`created_at` must be timezone-aware.")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_stable_source_identity(self) -> KnowledgeEvidence:
        if self.source_id is None and self.source_uri is None:
            raise ValueError("Knowledge evidence requires `source_id` or `source_uri`.")
        if self.source_revision is None and self.source_hash is None:
            raise ValueError("Knowledge evidence requires `source_revision` or `source_hash`.")
        if (
            len(
                canonical_durable_json_bytes(
                    self.model_dump(mode="json"),
                    "knowledge evidence",
                )
            )
            > MAX_KNOWLEDGE_EVIDENCE_BYTES
        ):
            raise ValueError(
                f"Knowledge evidence must be at most {MAX_KNOWLEDGE_EVIDENCE_BYTES} "
                "canonical UTF-8 bytes."
            )
        return self


class KnowledgeEvidenceResult(BaseModel):
    """Bounded evidence for one exact authorized knowledge revision."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    entry_id: str
    entry_revision: int
    evidence: list[KnowledgeEvidence] = Field(default_factory=list)
    truncated: bool = False
    limit: int
    max_bytes: int
    total_evidence_known: int

    @field_validator("entry_id")
    @classmethod
    def validate_entry_id(cls, value: str) -> str:
        return _knowledge_entry_id(value)

    @field_validator("entry_revision")
    @classmethod
    def validate_entry_revision(cls, value: int) -> int:
        _validate_knowledge_revision(value, "entry_revision")
        return value

    @field_validator("evidence", mode="before")
    @classmethod
    def copy_evidence(cls, value) -> list[KnowledgeEvidence]:
        return [copy_knowledge_evidence(item) for item in value]

    @field_validator("limit", "max_bytes")
    @classmethod
    def validate_limits(cls, value: int, info) -> int:
        _validate_positive_int(value, info.field_name)
        return value

    @field_validator("total_evidence_known")
    @classmethod
    def validate_total(cls, value: int) -> int:
        _validate_nonnegative_int(value, "total_evidence_known")
        return value

    @field_validator("truncated", mode="before")
    @classmethod
    def validate_truncated(cls, value) -> bool:
        if type(value) is not bool:
            raise ValueError("`truncated` must be a boolean.")
        return value

    @model_validator(mode="after")
    def validate_result(self) -> KnowledgeEvidenceResult:
        if len(self.evidence) > self.limit:
            raise ValueError("`evidence` cannot contain more records than `limit`.")
        if self.total_evidence_known < len(self.evidence):
            raise ValueError("`total_evidence_known` cannot be less than returned evidence.")
        if self.truncated != (len(self.evidence) < self.total_evidence_known):
            raise ValueError("`truncated` must reflect omitted evidence.")
        for item in self.evidence:
            if item.entry_id != self.entry_id or item.entry_revision != self.entry_revision:
                raise ValueError("Evidence result contains another entry revision.")
        return self


class KnowledgeEmbeddingIdentity(BaseModel):
    """Complete identity of one durable knowledge embedding projection.

    Content hashes alone are not sufficient reuse keys. Comparable vectors must
    agree on the canonical revision, projected content, embedding space, the
    projection generator, preprocessing, and the stored index representation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    entry_id: str
    entry_revision: int
    chunk_id: str | None = None
    projection_type: str
    projection_content_hash: str
    embedding_model: str
    dimensions: int
    preprocessing_version: str
    generator: str
    generator_version: str
    index_representation_version: str

    @field_validator("entry_id")
    @classmethod
    def validate_entry_id(cls, value: str) -> str:
        return _knowledge_entry_id(value)

    @field_validator("entry_revision")
    @classmethod
    def validate_entry_revision(cls, value: int) -> int:
        _validate_knowledge_revision(value, "entry_revision")
        return value

    @field_validator("chunk_id")
    @classmethod
    def validate_chunk_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _knowledge_chunk_id(value)

    @field_validator(
        "projection_type",
        "projection_content_hash",
        "embedding_model",
        "preprocessing_version",
        "generator",
        "generator_version",
        "index_representation_version",
    )
    @classmethod
    def validate_identity_component(cls, value: str, info) -> str:
        value = require_clean_nonblank(value, info.field_name)
        if len(value.encode("utf-8")) > 512:
            raise ValueError(f"`{info.field_name}` must be at most 512 UTF-8 bytes.")
        return value

    @field_validator("dimensions")
    @classmethod
    def validate_dimensions(cls, value: int) -> int:
        _validate_positive_int(value, "dimensions")
        if value > MAX_KNOWLEDGE_EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"`dimensions` must be less than or equal to {MAX_KNOWLEDGE_EMBEDDING_DIMENSIONS}."
            )
        return value


class KnowledgeEmbeddingProjection(BaseModel):
    """One externally computed vector fenced to an exact pending projection attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    identity: KnowledgeEmbeddingIdentity
    readiness_sequence: int
    attempt_id: str
    vector: list[float]

    @field_validator("identity", mode="before")
    @classmethod
    def copy_identity(cls, value: KnowledgeEmbeddingIdentity) -> KnowledgeEmbeddingIdentity:
        return copy_knowledge_embedding_identity(value)

    @field_validator("readiness_sequence")
    @classmethod
    def validate_readiness_sequence(cls, value: int) -> int:
        _validate_knowledge_index_sequence(
            value,
            "readiness_sequence",
            allow_zero=False,
        )
        return value

    @field_validator("attempt_id")
    @classmethod
    def validate_attempt_id(cls, value: str) -> str:
        return _bounded_knowledge_index_identity(value, "attempt_id")

    @field_validator("vector", mode="before")
    @classmethod
    def copy_vector(cls, value) -> list[float]:
        if type(value) is not list:
            raise ValueError("`vector` must be a list.")
        result: list[float] = []
        for index, component in enumerate(value):
            if isinstance(component, bool) or not isinstance(component, int | float):
                raise ValueError(f"`vector[{index}]` must be a number.")
            result.append(require_finite(float(component), f"vector[{index}]"))
        return result

    @model_validator(mode="after")
    def validate_vector_dimensions(self) -> KnowledgeEmbeddingProjection:
        if len(self.vector) != self.identity.dimensions:
            raise ValueError("`vector` length must equal `identity.dimensions`.")
        return self


class KnowledgeEmbeddingProjectionWriteResult(BaseModel):
    """Accepted identities from one bounded projection persistence request."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    submitted_records: int
    stored_identities: list[KnowledgeEmbeddingIdentity] = Field(default_factory=list)

    @field_validator("submitted_records")
    @classmethod
    def validate_submitted_records(cls, value: int) -> int:
        _validate_nonnegative_int(value, "submitted_records")
        if value > MAX_KNOWLEDGE_EMBEDDING_WORK_RECORD_LIMIT:
            raise ValueError(
                "`submitted_records` must be less than or equal to "
                f"{MAX_KNOWLEDGE_EMBEDDING_WORK_RECORD_LIMIT}."
            )
        return value

    @field_validator("stored_identities", mode="before")
    @classmethod
    def copy_stored_identities(
        cls,
        value,
    ) -> list[KnowledgeEmbeddingIdentity]:
        if type(value) is not list:
            raise ValueError("`stored_identities` must be a list.")
        return [copy_knowledge_embedding_identity(identity) for identity in value]

    @model_validator(mode="after")
    def validate_stored_partition(self) -> KnowledgeEmbeddingProjectionWriteResult:
        if len(self.stored_identities) > self.submitted_records:
            raise ValueError("Stored projection identities cannot exceed submitted records.")
        identity_sha256s = {
            _knowledge_embedding_identity_sha256(identity) for identity in self.stored_identities
        }
        if len(identity_sha256s) != len(self.stored_identities):
            raise ValueError("`stored_identities` cannot contain duplicates.")
        return self


class KnowledgeEmbeddingBackfillResult(BaseModel):
    """Portable outcome from one bounded embedding repair/backfill pass."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    scanned_records: int
    indexed_records: int
    failed_records: int
    skipped_records: int
    limit: int
    refresh_existing: bool
    next_cursor: str | None = None

    @field_validator(
        "scanned_records",
        "indexed_records",
        "failed_records",
        "skipped_records",
    )
    @classmethod
    def validate_count(cls, value: int, info) -> int:
        _validate_nonnegative_int(value, info.field_name)
        return value

    @field_validator("limit")
    @classmethod
    def validate_limit(cls, value: int) -> int:
        _validate_knowledge_embedding_work_record_limit(value, field_name="limit")
        return value

    @field_validator("refresh_existing", mode="before")
    @classmethod
    def validate_refresh_existing(cls, value) -> bool:
        if type(value) is not bool:
            raise ValueError("`refresh_existing` must be a boolean.")
        return value

    @field_validator("next_cursor")
    @classmethod
    def validate_next_cursor(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _bounded_knowledge_embedding_backfill_cursor(value, "next_cursor")

    @model_validator(mode="after")
    def validate_partition(self) -> KnowledgeEmbeddingBackfillResult:
        if self.scanned_records > self.limit:
            raise ValueError("`scanned_records` cannot exceed `limit`.")
        if self.indexed_records + self.failed_records + self.skipped_records != (
            self.scanned_records
        ):
            raise ValueError("Backfill outcomes must partition all scanned records.")
        return self


class _KnowledgeEmbeddingBackfillCursor(BaseModel):
    """Validated keyset state carried inside an opaque backfill cursor."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    version: int
    fingerprint: str
    importance: float
    updated_at: datetime
    entry_id: str
    chunk_index: int
    chunk_id: str

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: int) -> int:
        if type(value) is not int or value != _KNOWLEDGE_EMBEDDING_BACKFILL_CURSOR_VERSION:
            raise ValueError("Unsupported knowledge embedding backfill cursor version.")
        return value

    @field_validator("fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str) -> str:
        value = require_clean_nonblank(value, "fingerprint")
        if _SHA256_HEX_RE.fullmatch(value) is None:
            raise ValueError("Backfill cursor fingerprint must be a SHA-256 digest.")
        return value

    @field_validator("importance", mode="before")
    @classmethod
    def validate_importance(cls, value) -> float:
        return _validate_unit_float(value, "importance")

    @field_validator("updated_at")
    @classmethod
    def validate_updated_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("`updated_at` must be timezone-aware.")
        return value.astimezone(UTC)

    @field_validator("entry_id")
    @classmethod
    def validate_entry_id(cls, value: str) -> str:
        return _knowledge_entry_id(value, "entry_id")

    @field_validator("chunk_index")
    @classmethod
    def validate_chunk_index(cls, value: int) -> int:
        _validate_nonnegative_int(value, "chunk_index")
        if value > MAX_KNOWLEDGE_CHUNK_INDEX:
            raise ValueError(
                f"`chunk_index` must be less than or equal to {MAX_KNOWLEDGE_CHUNK_INDEX}."
            )
        return value

    @field_validator("chunk_id")
    @classmethod
    def validate_chunk_id(cls, value: str) -> str:
        return _knowledge_chunk_id(value, "chunk_id")


class KnowledgeIndexReadinessUpdate(BaseModel):
    """One requested state transition for an exact embedding identity."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    identity: KnowledgeEmbeddingIdentity
    state: KnowledgeIndexState
    attempt_id: str
    failure_code: str | None = None

    @field_validator("identity", mode="before")
    @classmethod
    def copy_identity(cls, value: KnowledgeEmbeddingIdentity) -> KnowledgeEmbeddingIdentity:
        if type(value) is not KnowledgeEmbeddingIdentity:
            raise TypeError("Index readiness requires a KnowledgeEmbeddingIdentity.")
        return value.model_copy(deep=True)

    @field_validator("attempt_id")
    @classmethod
    def validate_attempt_id(cls, value: str) -> str:
        return _bounded_knowledge_index_identity(value, "attempt_id")

    @field_validator("failure_code")
    @classmethod
    def validate_failure_code(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _bounded_knowledge_index_identity(value, "failure_code")

    @model_validator(mode="after")
    def validate_failure_state(self) -> KnowledgeIndexReadinessUpdate:
        if self.state is KnowledgeIndexState.FAILED and self.failure_code is None:
            raise ValueError("Failed index readiness requires `failure_code`.")
        if self.state is not KnowledgeIndexState.FAILED and self.failure_code is not None:
            raise ValueError("`failure_code` is valid only for failed index readiness.")
        return self


class KnowledgeIndexReadiness(BaseModel):
    """Immutable sequenced evidence of one derived-index state transition."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    sequence: int
    identity: KnowledgeEmbeddingIdentity
    state: KnowledgeIndexState
    attempt_id: str
    failure_code: str | None = None
    operation_id: str
    published_at: datetime

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, value: int) -> int:
        _validate_knowledge_index_sequence(value, "sequence", allow_zero=False)
        return value

    @field_validator("identity", mode="before")
    @classmethod
    def copy_identity(cls, value: KnowledgeEmbeddingIdentity) -> KnowledgeEmbeddingIdentity:
        if type(value) is not KnowledgeEmbeddingIdentity:
            raise TypeError("Index readiness requires a KnowledgeEmbeddingIdentity.")
        return value.model_copy(deep=True)

    @field_validator("attempt_id", "operation_id")
    @classmethod
    def validate_required_identity(cls, value: str, info) -> str:
        return _bounded_knowledge_index_identity(value, info.field_name)

    @field_validator("failure_code")
    @classmethod
    def validate_failure_code(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _bounded_knowledge_index_identity(value, "failure_code")

    @field_validator("published_at")
    @classmethod
    def validate_published_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("`published_at` must be timezone-aware.")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_failure_state(self) -> KnowledgeIndexReadiness:
        KnowledgeIndexReadinessUpdate(
            identity=self.identity,
            state=self.state,
            attempt_id=self.attempt_id,
            failure_code=self.failure_code,
        )
        return self


class KnowledgeIndexReadinessBatch(BaseModel):
    """Bounded ordered readiness events through one captured high-water mark."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    readiness: list[KnowledgeIndexReadiness] = Field(default_factory=list)
    after_sequence: int = 0
    next_after_sequence: int = 0
    high_water_sequence: int = 0
    truncated: bool = False
    limit: int

    @field_validator("readiness", mode="before")
    @classmethod
    def copy_readiness(cls, value: list[KnowledgeIndexReadiness]) -> list[KnowledgeIndexReadiness]:
        return [copy_knowledge_index_readiness(item) for item in value]

    @field_validator("after_sequence", "next_after_sequence", "high_water_sequence")
    @classmethod
    def validate_sequences(cls, value: int, info) -> int:
        _validate_knowledge_index_sequence(value, info.field_name)
        return value

    @field_validator("limit")
    @classmethod
    def validate_limit(cls, value: int) -> int:
        _validate_knowledge_index_readiness_limit(value)
        return value

    @field_validator("truncated", mode="before")
    @classmethod
    def validate_truncated(cls, value) -> bool:
        if type(value) is not bool:
            raise ValueError("`truncated` must be a boolean.")
        return value

    @model_validator(mode="after")
    def validate_page(self) -> KnowledgeIndexReadinessBatch:
        if self.next_after_sequence < self.after_sequence:
            raise ValueError("`next_after_sequence` cannot precede `after_sequence`.")
        if len(self.readiness) > self.limit:
            raise ValueError("`readiness` cannot contain more records than `limit`.")
        sequences = [item.sequence for item in self.readiness]
        if sequences != sorted(sequences) or len(sequences) != len(set(sequences)):
            raise ValueError("Index readiness records must have unique ascending sequences.")
        if any(sequence <= self.after_sequence for sequence in sequences):
            raise ValueError("Index readiness records fall outside the page frontier.")
        if sequences and sequences[-1] > self.high_water_sequence:
            raise ValueError("Index readiness records cannot exceed `high_water_sequence`.")
        if self.truncated and not self.readiness:
            raise ValueError("A truncated readiness page must contain a continuation record.")
        expected_next = (
            self.readiness[-1].sequence
            if self.truncated
            else max(self.after_sequence, self.high_water_sequence)
        )
        if self.next_after_sequence != expected_next:
            raise ValueError("`next_after_sequence` does not match readiness page semantics.")
        return self


class KnowledgeIndexCoverage(BaseModel):
    """Machine-readable semantic-index coverage for one search projection."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    projection_type: str
    embedding_model: str
    dimensions: int
    preprocessing_version: str
    generator: str
    generator_version: str
    index_representation_version: str
    eligible_records: int
    ready_records: int
    pending_records: int
    failed_records: int
    high_water_sequence: int
    complete: bool

    @field_validator(
        "projection_type",
        "embedding_model",
        "preprocessing_version",
        "generator",
        "generator_version",
        "index_representation_version",
    )
    @classmethod
    def validate_space_identity(cls, value: str, info) -> str:
        value = require_clean_nonblank(value, info.field_name)
        if len(value.encode("utf-8")) > 512:
            raise ValueError(f"`{info.field_name}` must be at most 512 UTF-8 bytes.")
        return value

    @field_validator("dimensions")
    @classmethod
    def validate_dimensions(cls, value: int) -> int:
        _validate_positive_int(value, "dimensions")
        if value > MAX_KNOWLEDGE_EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"`dimensions` must be less than or equal to {MAX_KNOWLEDGE_EMBEDDING_DIMENSIONS}."
            )
        return value

    @field_validator(
        "eligible_records",
        "ready_records",
        "pending_records",
        "failed_records",
    )
    @classmethod
    def validate_counts(cls, value: int, info) -> int:
        _validate_nonnegative_int(value, info.field_name)
        return value

    @field_validator("high_water_sequence")
    @classmethod
    def validate_high_water_sequence(cls, value: int) -> int:
        _validate_knowledge_index_sequence(value, "high_water_sequence")
        return value

    @field_validator("complete", mode="before")
    @classmethod
    def validate_complete(cls, value) -> bool:
        if type(value) is not bool:
            raise ValueError("`complete` must be a boolean.")
        return value

    @model_validator(mode="after")
    def validate_complete_partition(self) -> KnowledgeIndexCoverage:
        if self.ready_records + self.pending_records + self.failed_records != (
            self.eligible_records
        ):
            raise ValueError("Index coverage states must partition all eligible records.")
        if self.complete != (
            self.ready_records == self.eligible_records
            and self.pending_records == 0
            and self.failed_records == 0
        ):
            raise ValueError("`complete` must reflect complete ready index coverage.")
        return self


class KnowledgeEmbeddingWorkerResult(BaseModel):
    """Bounded outcome from consuming canonical changes into one embedding index."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    consumer_id: str
    worker_id: str
    claimed_changes: int
    acknowledged_changes: int
    indexed_records: int
    failed_records: int
    removed_records: int
    limit: int
    processed_records: int
    record_limit: int

    @field_validator("consumer_id", "worker_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _knowledge_change_identity(value, info.field_name)

    @field_validator(
        "claimed_changes",
        "acknowledged_changes",
        "indexed_records",
        "failed_records",
        "removed_records",
        "processed_records",
    )
    @classmethod
    def validate_count(cls, value: int, info) -> int:
        _validate_nonnegative_int(value, info.field_name)
        return value

    @field_validator("limit")
    @classmethod
    def validate_limit(cls, value: int) -> int:
        _validate_knowledge_change_limit(value)
        return value

    @field_validator("record_limit")
    @classmethod
    def validate_record_limit(cls, value: int) -> int:
        _validate_knowledge_embedding_work_record_limit(value)
        return value

    @model_validator(mode="after")
    def validate_claims(self) -> KnowledgeEmbeddingWorkerResult:
        if self.claimed_changes > self.limit:
            raise ValueError("`claimed_changes` cannot exceed `limit`.")
        if self.acknowledged_changes > self.claimed_changes:
            raise ValueError("`acknowledged_changes` cannot exceed `claimed_changes`.")
        if self.processed_records > self.record_limit:
            raise ValueError("`processed_records` cannot exceed `record_limit`.")
        if self.indexed_records + self.failed_records + self.removed_records > (
            self.processed_records
        ):
            raise ValueError("Embedding outcomes cannot exceed processed records.")
        return self


class KnowledgeQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    text: str | None = None
    any_terms: list[str] = Field(default_factory=list)
    all_terms: list[str] = Field(default_factory=list)
    none_terms: list[str] = Field(default_factory=list)
    phrases: list[str] = Field(default_factory=list)
    namespace: str = DEFAULT_KNOWLEDGE_NAMESPACE
    labels: dict[str, str] = Field(default_factory=dict)
    kinds: list[str] | None = None
    statuses: list[KnowledgeStatus] = Field(default_factory=lambda: [KnowledgeStatus.ACTIVE])
    visibilities: list[KnowledgeVisibility] | None = None
    aspects: list[str] = Field(default_factory=list)
    impact_targets: list[str] = Field(default_factory=list)
    source_type: str | None = None
    source_id: str | None = None
    mode: KnowledgeSearchMode = KnowledgeSearchMode.AUTO
    min_score: float | None = None
    include_expired: bool = False
    limit: int = DEFAULT_KNOWLEDGE_LIMIT
    max_bytes: int = DEFAULT_KNOWLEDGE_MAX_BYTES

    @field_validator("labels", mode="before")
    @classmethod
    def copy_labels(cls, value) -> dict[str, str]:
        return copy_label_map(value, "labels")

    @field_validator("text")
    @classmethod
    def validate_optional_nonblank_text(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_nonblank(value, info.field_name)

    @field_validator("min_score", mode="before")
    @classmethod
    def validate_optional_min_score(cls, value, info) -> float | None:
        if value is None:
            return None
        return _validate_unit_float(value, info.field_name)

    @field_validator("namespace")
    @classmethod
    def validate_clean_namespace(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("source_type", "source_id")
    @classmethod
    def validate_optional_clean_nonblank_fields(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator(
        "any_terms",
        "all_terms",
        "none_terms",
        "phrases",
        "kinds",
        "aspects",
        "impact_targets",
        mode="before",
    )
    @classmethod
    def copy_optional_string_list(cls, value, info) -> list[str] | None:
        if value is None and info.field_name == "kinds":
            return None
        if value is None:
            return []
        copied = copy_json_value(value, info.field_name)
        if type(copied) is not list:
            raise ValueError(f"`{info.field_name}` must be a list.")
        result: list[str] = []
        for index, item in enumerate(copied):
            if type(item) is not str:
                raise ValueError(f"`{info.field_name}[{index}]` must be a string.")
            result.append(require_clean_nonblank(item, f"{info.field_name}[{index}]"))
        return _dedupe_strings(result)

    @field_validator("limit", "max_bytes")
    @classmethod
    def validate_positive_int(cls, value: int, info) -> int:
        if isinstance(value, bool) or type(value) is not int:
            raise ValueError(f"`{info.field_name}` must be an integer.")
        if value <= 0:
            raise ValueError(f"`{info.field_name}` must be greater than 0.")
        return value

    @field_validator("statuses")
    @classmethod
    def validate_statuses(cls, value: list[KnowledgeStatus], info) -> list[KnowledgeStatus]:
        if not value:
            raise ValueError(f"`{info.field_name}` cannot be empty.")
        return list(dict.fromkeys(value))

    @field_validator("visibilities")
    @classmethod
    def validate_visibilities(
        cls,
        value: list[KnowledgeVisibility] | None,
        info,
    ) -> list[KnowledgeVisibility] | None:
        if value is None:
            return None
        if not value:
            raise ValueError(f"`{info.field_name}` cannot be empty.")
        return list(dict.fromkeys(value))

    @model_validator(mode="after")
    def validate_has_positive_search_terms(self) -> KnowledgeQuery:
        terms = _knowledge_query_terms(self)
        if _query_terms_have_positive_terms(terms):
            return self
        raise ValueError("Knowledge query requires `text`, `any_terms`, `all_terms`, or `phrases`.")


class KnowledgeListQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    namespace: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    kinds: list[str] | None = None
    statuses: list[KnowledgeStatus] = Field(default_factory=lambda: [KnowledgeStatus.ACTIVE])
    visibilities: list[KnowledgeVisibility] | None = None
    aspects: list[str] = Field(default_factory=list)
    impact_targets: list[str] = Field(default_factory=list)
    source_type: str | None = None
    source_id: str | None = None
    include_expired: bool = False
    group_by: KnowledgeListGroup | None = None
    limit: int = DEFAULT_KNOWLEDGE_LIMIT
    max_bytes: int = DEFAULT_KNOWLEDGE_MAX_BYTES

    @field_validator("labels", mode="before")
    @classmethod
    def copy_labels(cls, value) -> dict[str, str]:
        return copy_label_map(value, "labels")

    @field_validator("namespace", "source_type", "source_id")
    @classmethod
    def validate_optional_clean_nonblank_fields(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("kinds", "aspects", "impact_targets", mode="before")
    @classmethod
    def copy_optional_string_list(cls, value, info) -> list[str] | None:
        if value is None and info.field_name == "kinds":
            return None
        if value is None:
            return []
        copied = copy_json_value(value, info.field_name)
        if type(copied) is not list:
            raise ValueError(f"`{info.field_name}` must be a list.")
        result: list[str] = []
        for index, item in enumerate(copied):
            if type(item) is not str:
                raise ValueError(f"`{info.field_name}[{index}]` must be a string.")
            result.append(require_clean_nonblank(item, f"{info.field_name}[{index}]"))
        return _dedupe_strings(result)

    @field_validator("limit", "max_bytes")
    @classmethod
    def validate_positive_int(cls, value: int, info) -> int:
        _validate_positive_int(value, info.field_name)
        return value

    @field_validator("statuses")
    @classmethod
    def validate_statuses(cls, value: list[KnowledgeStatus], info) -> list[KnowledgeStatus]:
        if not value:
            raise ValueError(f"`{info.field_name}` cannot be empty.")
        return list(dict.fromkeys(value))

    @field_validator("visibilities")
    @classmethod
    def validate_visibilities(
        cls,
        value: list[KnowledgeVisibility] | None,
        info,
    ) -> list[KnowledgeVisibility] | None:
        if value is None:
            return None
        if not value:
            raise ValueError(f"`{info.field_name}` cannot be empty.")
        return list(dict.fromkeys(value))


class KnowledgeHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entry: KnowledgeEntry
    chunk: KnowledgeChunk | None = None
    score: float | None = None
    reason: str | None = None
    rank: int | None = None
    score_kind: str | None = None
    score_normalized: float | None = None
    text_preview: str | None = None
    text_preview_complete: bool = Field(default=False, exclude=True, repr=False)

    @field_validator("entry")
    @classmethod
    def copy_entry(cls, value):
        return copy_knowledge_entry(value)

    @field_validator("chunk")
    @classmethod
    def copy_chunk(cls, value):
        if value is None:
            return None
        return copy_knowledge_chunk(value)

    @field_validator("score", "score_normalized", mode="before")
    @classmethod
    def validate_score(cls, value, info):
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"`{info.field_name}` must be a number.")
        value = require_finite(float(value), info.field_name)
        if info.field_name == "score_normalized" and (value < 0.0 or value > 1.0):
            raise ValueError("`score_normalized` must be between 0.0 and 1.0.")
        return value

    @field_validator("rank")
    @classmethod
    def validate_rank(cls, value: int | None, info) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or type(value) is not int:
            raise ValueError(f"`{info.field_name}` must be an integer.")
        if value <= 0:
            raise ValueError(f"`{info.field_name}` must be greater than 0.")
        return value

    @field_validator("reason", "score_kind", "text_preview")
    @classmethod
    def validate_optional_nonblank_fields(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_nonblank(value, info.field_name)

    @field_validator("text_preview_complete", mode="before")
    @classmethod
    def validate_text_preview_complete(cls, value, info) -> bool:
        if type(value) is not bool:
            raise ValueError(f"`{info.field_name}` must be a boolean.")
        return value

    @model_validator(mode="after")
    def validate_chunk_belongs_to_entry(self) -> KnowledgeHit:
        if self.chunk is not None and self.chunk.entry_id != self.entry.id:
            raise ValueError("`chunk.entry_id` must match `entry.id`.")
        if self.chunk is not None and self.chunk.entry_revision != self.entry.revision:
            raise ValueError("`chunk.entry_revision` must match `entry.revision`.")
        if self.text_preview is None and self.text_preview_complete:
            raise ValueError("`text_preview_complete` requires `text_preview`.")
        return self


class KnowledgeSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: KnowledgeQuery
    hits: list[KnowledgeHit] = Field(default_factory=list)
    truncated: bool = False
    limit: int
    max_bytes: int
    total_hits_known: int | None = None
    index_coverage: list[KnowledgeIndexCoverage] = Field(default_factory=list)

    @field_validator("query")
    @classmethod
    def copy_query(cls, value):
        return copy_knowledge_query(value)

    @field_validator("hits")
    @classmethod
    def copy_hits(cls, value):
        return [copy_knowledge_hit(hit) for hit in value]

    @field_validator("index_coverage", mode="before")
    @classmethod
    def copy_index_coverage(
        cls, value: list[KnowledgeIndexCoverage]
    ) -> list[KnowledgeIndexCoverage]:
        return [copy_knowledge_index_coverage(item) for item in value]

    @field_validator("limit", "max_bytes")
    @classmethod
    def validate_positive_int(cls, value: int, info) -> int:
        _validate_positive_int(value, info.field_name)
        return value

    @field_validator("total_hits_known")
    @classmethod
    def validate_total_hits_known(cls, value: int | None, info) -> int | None:
        if value is None:
            return None
        _validate_nonnegative_int(value, info.field_name)
        return value

    @model_validator(mode="after")
    def validate_total_hits_known_covers_hits(self) -> KnowledgeSearchResult:
        if self.total_hits_known is not None and self.total_hits_known < len(self.hits):
            raise ValueError("`total_hits_known` cannot be less than the number of hits.")
        return self

    @model_validator(mode="after")
    def validate_limits_match_query(self) -> KnowledgeSearchResult:
        if self.limit != self.query.limit:
            raise ValueError("`limit` must match `query.limit`.")
        if self.max_bytes != self.query.max_bytes:
            raise ValueError("`max_bytes` must match `query.max_bytes`.")
        return self

    @model_validator(mode="after")
    def validate_hit_count_and_ranks(self) -> KnowledgeSearchResult:
        if len(self.hits) > self.limit:
            raise ValueError("`hits` cannot contain more entries than `limit`.")
        ranks = [hit.rank for hit in self.hits if hit.rank is not None]
        if len(ranks) != len(set(ranks)):
            raise ValueError("Knowledge hit ranks must be unique when present.")
        projection_spaces = [
            (
                item.projection_type,
                item.embedding_model,
                item.dimensions,
                item.preprocessing_version,
                item.generator,
                item.generator_version,
                item.index_representation_version,
            )
            for item in self.index_coverage
        ]
        if len(projection_spaces) != len(set(projection_spaces)):
            raise ValueError("Index coverage projection spaces must be unique.")
        return self


class KnowledgeListItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entry: KnowledgeEntry
    chunk_count: int = 0
    text_preview: str | None = None
    text_preview_complete: bool = Field(default=False, exclude=True, repr=False)

    @field_validator("entry")
    @classmethod
    def copy_entry(cls, value):
        return copy_knowledge_entry(value)

    @field_validator("chunk_count")
    @classmethod
    def validate_chunk_count(cls, value: int, info) -> int:
        _validate_nonnegative_int(value, info.field_name)
        return value

    @field_validator("text_preview")
    @classmethod
    def validate_optional_nonblank_text(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_nonblank(value, info.field_name)

    @field_validator("text_preview_complete", mode="before")
    @classmethod
    def validate_text_preview_complete(cls, value, info) -> bool:
        if type(value) is not bool:
            raise ValueError(f"`{info.field_name}` must be a boolean.")
        return value

    @model_validator(mode="after")
    def validate_text_preview_provenance(self) -> KnowledgeListItem:
        if self.text_preview is None and self.text_preview_complete:
            raise ValueError("`text_preview_complete` requires `text_preview`.")
        return self


class KnowledgeFacet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: KnowledgeListGroup
    value: str
    count: int
    key: str | None = None

    @field_validator("value", "key")
    @classmethod
    def validate_optional_clean_nonblank(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("count")
    @classmethod
    def validate_count(cls, value: int, info) -> int:
        _validate_nonnegative_int(value, info.field_name)
        return value


class KnowledgeListResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: KnowledgeListQuery
    entries: list[KnowledgeListItem] = Field(default_factory=list)
    facets: list[KnowledgeFacet] = Field(default_factory=list)
    facets_truncated: bool = False
    truncated: bool = False
    limit: int
    max_bytes: int
    total_entries_known: int | None = None

    @field_validator("query")
    @classmethod
    def copy_query(cls, value):
        return copy_knowledge_list_query(value)

    @field_validator("entries")
    @classmethod
    def copy_entries(cls, value):
        return [copy_knowledge_list_item(item) for item in value]

    @field_validator("facets")
    @classmethod
    def copy_facets(cls, value):
        return [copy_knowledge_facet(facet) for facet in value]

    @field_validator("limit", "max_bytes")
    @classmethod
    def validate_positive_int(cls, value: int, info) -> int:
        _validate_positive_int(value, info.field_name)
        return value

    @field_validator("total_entries_known")
    @classmethod
    def validate_total_entries_known(cls, value: int | None, info) -> int | None:
        if value is None:
            return None
        _validate_nonnegative_int(value, info.field_name)
        return value

    @model_validator(mode="after")
    def validate_total_entries_known_covers_entries(self) -> KnowledgeListResult:
        if self.total_entries_known is not None and self.total_entries_known < len(self.entries):
            raise ValueError("`total_entries_known` cannot be less than the number of entries.")
        return self

    @model_validator(mode="after")
    def validate_limits_match_query(self) -> KnowledgeListResult:
        if self.limit != self.query.limit:
            raise ValueError("`limit` must match `query.limit`.")
        if self.max_bytes != self.query.max_bytes:
            raise ValueError("`max_bytes` must match `query.max_bytes`.")
        return self

    @model_validator(mode="after")
    def validate_entry_and_facet_count(self) -> KnowledgeListResult:
        if len(self.entries) > self.limit:
            raise ValueError("`entries` cannot contain more entries than `limit`.")
        if len(self.facets) > self.limit:
            raise ValueError("`facets` cannot contain more buckets than `limit`.")
        return self

    @model_validator(mode="after")
    def validate_facet_group(self) -> KnowledgeListResult:
        if self.query.group_by is None and self.facets:
            raise ValueError("`facets` require `query.group_by`.")
        if self.query.group_by is not None:
            for facet in self.facets:
                if facet.field != self.query.group_by:
                    raise ValueError("Knowledge facets must match `query.group_by`.")
        return self


class KnowledgePublicationConflict(RuntimeError):
    """An idempotent knowledge publication conflicts with durable state."""

    def __init__(self, reason: str) -> None:
        self.reason = require_clean_nonblank(reason, "reason")
        super().__init__("Knowledge publication conflicts with durable state.")


class KnowledgePublicationReceipt(BaseModel):
    """Bounded immutable evidence for one atomic entry-and-chunks publication."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    operation_id: str = Field(max_length=256)
    entry_id: str
    entry_revision: int
    expected_revision: int | None
    request_sha256: str
    entry_created_at: datetime
    entry_updated_at: datetime
    committed_at: datetime
    replayed: bool = False

    @field_validator("operation_id")
    @classmethod
    def validate_clean_ids(cls, value: str, info) -> str:
        value = require_clean_nonblank(value, info.field_name)
        if len(value.encode("utf-8")) > 256:
            raise ValueError(f"`{info.field_name}` must be at most 256 UTF-8 bytes.")
        return value

    @field_validator("entry_id")
    @classmethod
    def validate_entry_id(cls, value: str) -> str:
        return _knowledge_entry_id(value)

    @field_validator("request_sha256")
    @classmethod
    def validate_request_sha256(cls, value: str) -> str:
        if type(value) is not str or _SHA256_HEX_RE.fullmatch(value) is None:
            raise ValueError("`request_sha256` must be a lowercase SHA-256 digest.")
        return value

    @field_validator("entry_revision")
    @classmethod
    def validate_entry_revision(cls, value: int) -> int:
        _validate_knowledge_revision(value, "entry_revision")
        return value

    @field_validator("expected_revision")
    @classmethod
    def validate_expected_revision(cls, value: int | None) -> int | None:
        if value is not None:
            _validate_knowledge_revision(value, "expected_revision")
        return value

    @model_validator(mode="after")
    def validate_revision_transition(self) -> KnowledgePublicationReceipt:
        expected_entry_revision = (
            1 if self.expected_revision is None else self.expected_revision + 1
        )
        if self.entry_revision != expected_entry_revision:
            raise ValueError("`entry_revision` must follow `expected_revision`.")
        return self

    @field_validator("entry_created_at", "entry_updated_at", "committed_at")
    @classmethod
    def validate_receipt_datetime(cls, value: datetime, info) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"`{info.field_name}` must be timezone-aware.")
        return value.astimezone(UTC)

    @field_validator("replayed")
    @classmethod
    def validate_replayed(cls, value: bool) -> bool:
        if type(value) is not bool:
            raise ValueError("`replayed` must be a boolean.")
        return value

    @model_validator(mode="after")
    def validate_entry_timestamp_order(self) -> KnowledgePublicationReceipt:
        if self.entry_updated_at < self.entry_created_at:
            raise ValueError("`entry_updated_at` cannot precede `entry_created_at`.")
        return self


class KnowledgeChange(BaseModel):
    """Metadata-only canonical knowledge mutation published in commit order."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    id: str
    sequence: int
    kind: KnowledgeChangeKind
    entry_id: str
    entry_revision: int
    committed_at: datetime
    operation_id: str | None = None

    @field_validator("id", "operation_id")
    @classmethod
    def validate_identity(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        value = require_clean_nonblank(value, info.field_name)
        if len(value.encode("utf-8")) > 256:
            raise ValueError(f"`{info.field_name}` must be at most 256 UTF-8 bytes.")
        return value

    @field_validator("entry_id")
    @classmethod
    def validate_entry_id(cls, value: str) -> str:
        return _knowledge_entry_id(value)

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, value: int) -> int:
        _validate_knowledge_change_sequence(value, "sequence")
        if value == 0:
            raise ValueError("`sequence` must be greater than 0.")
        return value

    @field_validator("entry_revision")
    @classmethod
    def validate_entry_revision(cls, value: int) -> int:
        _validate_knowledge_revision(value, "entry_revision")
        return value

    @field_validator("committed_at")
    @classmethod
    def validate_committed_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("`committed_at` must be timezone-aware.")
        return value.astimezone(UTC)


class KnowledgeChangeBatch(BaseModel):
    """One bounded ordered change page and its captured accessible high-water mark."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    changes: list[KnowledgeChange] = Field(default_factory=list)
    after_sequence: int = 0
    next_after_sequence: int = 0
    high_water_sequence: int = 0
    truncated: bool = False
    limit: int

    @field_validator("changes", mode="before")
    @classmethod
    def copy_changes(cls, value) -> list[KnowledgeChange]:
        return [copy_knowledge_change(change) for change in value]

    @field_validator("after_sequence", "next_after_sequence", "high_water_sequence")
    @classmethod
    def validate_sequences(cls, value: int, info) -> int:
        _validate_knowledge_change_sequence(value, info.field_name)
        return value

    @field_validator("limit")
    @classmethod
    def validate_limit(cls, value: int) -> int:
        _validate_knowledge_change_limit(value)
        return value

    @field_validator("truncated", mode="before")
    @classmethod
    def validate_truncated(cls, value) -> bool:
        if type(value) is not bool:
            raise ValueError("`truncated` must be a boolean.")
        return value

    @model_validator(mode="after")
    def validate_page(self) -> KnowledgeChangeBatch:
        if len(self.changes) > self.limit:
            raise ValueError("`changes` cannot contain more records than `limit`.")
        sequences = [change.sequence for change in self.changes]
        if sequences != sorted(set(sequences)):
            raise ValueError("Knowledge changes must have unique increasing sequences.")
        if any(sequence <= self.after_sequence for sequence in sequences):
            raise ValueError("Knowledge changes must follow `after_sequence`.")
        if sequences and sequences[-1] > self.high_water_sequence:
            raise ValueError("Knowledge changes cannot exceed `high_water_sequence`.")
        expected_next = (
            sequences[-1]
            if self.truncated and sequences
            else max(self.after_sequence, self.high_water_sequence)
        )
        if self.next_after_sequence != expected_next:
            raise ValueError("`next_after_sequence` does not match the bounded page.")
        if self.truncated and not sequences:
            raise ValueError("A truncated knowledge-change page cannot be empty.")
        return self


class KnowledgeChangeClaim(BaseModel):
    """One fenced at-least-once lease over an ordered knowledge change."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    consumer_id: str
    worker_id: str
    claim_id: str
    change: KnowledgeChange
    attempt: int
    claimed_at: datetime
    lease_expires_at: datetime

    @field_validator("consumer_id", "worker_id", "claim_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        value = require_clean_nonblank(value, info.field_name)
        if len(value.encode("utf-8")) > 256:
            raise ValueError(f"`{info.field_name}` must be at most 256 UTF-8 bytes.")
        return value

    @field_validator("change")
    @classmethod
    def copy_change(cls, value: KnowledgeChange) -> KnowledgeChange:
        return copy_knowledge_change(value)

    @field_validator("attempt")
    @classmethod
    def validate_attempt(cls, value: int) -> int:
        _validate_positive_int(value, "attempt")
        return value

    @field_validator("claimed_at", "lease_expires_at")
    @classmethod
    def validate_datetime(cls, value: datetime, info) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"`{info.field_name}` must be timezone-aware.")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_lease_window(self) -> KnowledgeChangeClaim:
        if self.lease_expires_at <= self.claimed_at:
            raise ValueError("`lease_expires_at` must follow `claimed_at`.")
        return self


class KnowledgeChangeConsumerState(BaseModel):
    """Durable cursor and active lease state for one scope-bound consumer."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    consumer_id: str
    access_scope_sha256: str
    cursor_sequence: int = 0
    pending_change_sequence: int | None = None
    pending_claim_id: str | None = None
    pending_worker_id: str | None = None
    pending_attempt: int = 0
    claimed_at: datetime | None = None
    lease_expires_at: datetime | None = None
    last_acknowledged_claim_id: str | None = None
    updated_at: datetime

    @field_validator(
        "consumer_id",
        "pending_claim_id",
        "pending_worker_id",
        "last_acknowledged_claim_id",
    )
    @classmethod
    def validate_identity(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        value = require_clean_nonblank(value, info.field_name)
        if len(value.encode("utf-8")) > 256:
            raise ValueError(f"`{info.field_name}` must be at most 256 UTF-8 bytes.")
        return value

    @field_validator("access_scope_sha256")
    @classmethod
    def validate_scope_digest(cls, value: str) -> str:
        if type(value) is not str or _SHA256_HEX_RE.fullmatch(value) is None:
            raise ValueError("`access_scope_sha256` must be a lowercase SHA-256 digest.")
        return value

    @field_validator("cursor_sequence")
    @classmethod
    def validate_cursor(cls, value: int) -> int:
        _validate_knowledge_change_sequence(value, "cursor_sequence")
        return value

    @field_validator("pending_change_sequence")
    @classmethod
    def validate_pending_sequence(cls, value: int | None) -> int | None:
        if value is not None:
            _validate_knowledge_change_sequence(value, "pending_change_sequence")
            if value == 0:
                raise ValueError("`pending_change_sequence` must be greater than 0.")
        return value

    @field_validator("pending_attempt")
    @classmethod
    def validate_pending_attempt(cls, value: int) -> int:
        _validate_nonnegative_int(value, "pending_attempt")
        return value

    @field_validator("claimed_at", "lease_expires_at", "updated_at")
    @classmethod
    def validate_datetime(cls, value: datetime | None, info) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"`{info.field_name}` must be timezone-aware.")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_pending_lease(self) -> KnowledgeChangeConsumerState:
        pending = (
            self.pending_change_sequence,
            self.pending_claim_id,
            self.pending_worker_id,
            self.claimed_at,
            self.lease_expires_at,
        )
        if any(value is not None for value in pending) and not all(
            value is not None for value in pending
        ):
            raise ValueError("Knowledge change pending-lease fields must be set together.")
        if self.pending_change_sequence is not None:
            if self.pending_change_sequence <= self.cursor_sequence:
                raise ValueError("A pending change must follow the consumer cursor.")
            if self.pending_attempt <= 0:
                raise ValueError("An active knowledge change lease requires a positive attempt.")
            assert self.claimed_at is not None
            assert self.lease_expires_at is not None
            if self.lease_expires_at <= self.claimed_at:
                raise ValueError("An active knowledge change lease must expire after claim time.")
        return self


class KnowledgeStore(ABC):
    """Searchable knowledge contract."""

    _default_access_scope: KnowledgeAccessScope | None = None

    def bound_access_scope(self) -> KnowledgeAccessScope | None:
        """Return the explicitly bound single-principal scope, if configured."""

        if self._default_access_scope is None:
            return None
        return copy_knowledge_access_scope(self._default_access_scope)

    def _operation_access_scope(
        self,
        access_scope: KnowledgeAccessScope | None,
    ) -> KnowledgeAccessScope:
        default_scope = self._default_access_scope
        if access_scope is None:
            if default_scope is None:
                raise TypeError("knowledge operation requires `access_scope`.")
            return copy_knowledge_access_scope(default_scope)
        explicit_scope = copy_knowledge_access_scope(access_scope)
        if default_scope is not None and explicit_scope != default_scope:
            raise KnowledgeAccessDenied("access_scope_override")
        return explicit_scope

    def supported_search_modes(self) -> tuple[KnowledgeSearchMode, ...]:
        """Return search modes this store can execute directly."""

        return (KnowledgeSearchMode.AUTO, KnowledgeSearchMode.KEYWORD)

    @abstractmethod
    async def create_entry(
        self,
        entry: KnowledgeEntry,
        chunks: list[KnowledgeChunk] | None = None,
        *,
        evidence: list[KnowledgeEvidence] | None = None,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeEntry:
        """Create revision 1 of a previously unoccupied logical entry id."""

    @abstractmethod
    async def append_entry_revision(
        self,
        entry: KnowledgeEntry,
        chunks: list[KnowledgeChunk] | None = None,
        *,
        expected_revision: int,
        evidence: list[KnowledgeEvidence] | None = None,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeEntry:
        """Append exactly one revision using compare-and-swap."""

    @abstractmethod
    async def get_entry(
        self,
        entry_id: str,
        *,
        revision: int | None = None,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeEntry | None:
        """Load the current revision, or one exact historical revision."""

    @abstractmethod
    async def transition_entry_status(
        self,
        entry_id: str,
        *,
        expected_revision: int,
        access_scope: KnowledgeAccessScope | None = None,
        from_status: KnowledgeStatus,
        to_status: KnowledgeStatus,
        expected_namespace: str | None = None,
        expected_labels: dict[str, str] | None = None,
    ) -> KnowledgeEntry:
        """Append one lifecycle-only revision using compare-and-swap."""

    @abstractmethod
    async def delete_entry(
        self,
        entry_id: str,
        *,
        expected_revision: int,
        access_scope: KnowledgeAccessScope | None = None,
        hard: bool = False,
    ) -> KnowledgeEntry | None:
        """Append a tombstone by default, or physically erase after a CAS check."""

    async def publish_entry_revision(
        self,
        entry: KnowledgeEntry,
        chunks: list[KnowledgeChunk],
        *,
        evidence: list[KnowledgeEvidence] | None = None,
        access_scope: KnowledgeAccessScope | None = None,
        operation_id: str,
        expected_revision: int | None = None,
    ) -> KnowledgePublicationReceipt:
        """Publish one create/append exactly once with immutable replay evidence.

        Implementations commit the revision, chunks, evidence, current pointer,
        metadata-only change, and receipt atomically. ``expected_revision=None``
        creates revision 1; a positive value appends exactly its successor.
        Use :func:`prepare_knowledge_publication` to copy and bind the canonical
        authority tuple before entering the store transaction.
        """

        raise NotImplementedError(
            "This KnowledgeStore does not support owned revision publication."
        )

    async def load_entry_publication_receipt(
        self,
        operation_id: str,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgePublicationReceipt | None:
        """Load immutable publication evidence for one operation id."""

        raise NotImplementedError(
            "This KnowledgeStore does not support knowledge publication receipts."
        )

    @abstractmethod
    async def read_evidence(
        self,
        entry_id: str,
        *,
        revision: int | None = None,
        access_scope: KnowledgeAccessScope | None = None,
        max_records: int = DEFAULT_KNOWLEDGE_LIMIT,
        max_bytes: int = DEFAULT_KNOWLEDGE_MAX_BYTES,
    ) -> KnowledgeEvidenceResult | None:
        """Read evidence for the current or one exact authorized revision."""

    @abstractmethod
    async def read_chunks(
        self,
        entry_id: str,
        *,
        revision: int | None = None,
        access_scope: KnowledgeAccessScope | None = None,
        chunk_index: int | None = None,
        around: int = 0,
        max_chunks: int = DEFAULT_KNOWLEDGE_LIMIT,
        max_bytes: int = DEFAULT_KNOWLEDGE_MAX_BYTES,
    ) -> list[KnowledgeChunk]:
        """Read bounded chunks for the current or one exact historical revision."""

    @abstractmethod
    async def search(
        self,
        query: KnowledgeQuery,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeSearchResult:
        """Search knowledge and return a bounded result envelope."""

    @abstractmethod
    async def list_entries(
        self,
        query: KnowledgeListQuery,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeListResult:
        """List entries/facets for discovery without requiring a lexical search term."""

    @abstractmethod
    async def read_changes(
        self,
        *,
        after_sequence: int = 0,
        limit: int = DEFAULT_KNOWLEDGE_LIMIT,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeChangeBatch:
        """Read one bounded ordered page of accessible canonical changes."""

    @abstractmethod
    async def initialize_change_consumer(
        self,
        consumer_id: str,
        *,
        baseline_sequence: int,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeChangeConsumerState:
        """Bind a consumer and its cursor to a captured full-scan high-water mark."""

    @abstractmethod
    async def claim_change(
        self,
        consumer_id: str,
        worker_id: str,
        *,
        lease_seconds: float = 300.0,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeChangeClaim | None:
        """Lease the consumer's next accessible change with at-least-once semantics."""

    @abstractmethod
    async def acknowledge_change(
        self,
        claim: KnowledgeChangeClaim,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeChangeConsumerState:
        """Fenced acknowledgement that advances one consumer cursor."""

    @abstractmethod
    async def release_change(
        self,
        claim: KnowledgeChangeClaim,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeChangeConsumerState:
        """Release a live claim without advancing its consumer cursor."""

    @abstractmethod
    async def load_change_consumer_state(
        self,
        consumer_id: str,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeChangeConsumerState | None:
        """Load one scope-bound consumer cursor and lease state."""

    async def publish_index_readiness(
        self,
        update: KnowledgeIndexReadinessUpdate,
        *,
        expected_sequence: int | None,
        operation_id: str,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeIndexReadiness:
        """Publish one fenced derived-index state transition.

        This hook is intentionally optional so lexical-only custom stores do not
        have to pretend they own derived indexes. Implementations that advertise
        semantic retrieval must provide equivalent durable semantics.
        """

        raise NotImplementedError(
            "This KnowledgeStore does not support index readiness publication."
        )

    async def load_index_readiness(
        self,
        identity: KnowledgeEmbeddingIdentity,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeIndexReadiness | None:
        """Load the latest accessible readiness for one exact identity."""

        raise NotImplementedError("This KnowledgeStore does not support index readiness reads.")

    async def store_embedding_projections(
        self,
        projections: list[KnowledgeEmbeddingProjection],
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeEmbeddingProjectionWriteResult:
        """Persist already-computed vectors for exact current pending attempts.

        Stale, superseded, and unauthorized projections are omitted from the
        returned accepted identities. Readiness remains a separate fenced
        publication step so a vector cannot become searchable prematurely.
        """

        raise NotImplementedError(
            "This KnowledgeStore does not support embedding projection persistence."
        )

    async def backfill_embeddings(
        self,
        query: KnowledgeListQuery | None = None,
        *,
        access_scope: KnowledgeAccessScope | None = None,
        limit: int = DEFAULT_KNOWLEDGE_EMBEDDING_WORK_RECORD_LIMIT,
        refresh_existing: bool = False,
        cursor: str | None = None,
    ) -> KnowledgeEmbeddingBackfillResult:
        """Repair one bounded page of current embedding projections.

        Pass the previous result's ``next_cursor`` to continue the exact same
        query, scope, projection configuration, and refresh mode.
        """

        raise NotImplementedError("This KnowledgeStore does not support embedding backfill.")

    async def read_index_readiness(
        self,
        *,
        after_sequence: int = 0,
        limit: int = DEFAULT_KNOWLEDGE_LIMIT,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeIndexReadinessBatch:
        """Read accessible readiness events through a captured high-water mark."""

        raise NotImplementedError("This KnowledgeStore does not support index readiness reads.")

    async def prune_expired(
        self,
        *,
        access_scope: KnowledgeAccessScope | None = None,
        now: datetime | None = None,
    ) -> int:
        """Hard-delete entries whose ``expires_at`` is at or before ``now`` (default: current UTC).

        Returns the count removed. The read-time filter (:func:`_entry_is_expired`) only *hides*
        expired entries; this reclaims their storage. Hosts call it on a schedule or opportunistically.
        ``now`` is injectable for deterministic tests.

        Default raises ``NotImplementedError`` so out-of-tree stores keep working.
        """
        raise NotImplementedError("This KnowledgeStore does not support prune_expired.")


class InMemoryKnowledgeStore(KnowledgeStore):
    """In-memory knowledge store for tests, demos, and single-process apps."""

    def __init__(
        self,
        entries: list[KnowledgeEntry] | None = None,
        *,
        access_scope: KnowledgeAccessScope | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._default_access_scope = (
            None if access_scope is None else copy_knowledge_access_scope(access_scope)
        )
        self._clock = utc_clock(clock)
        self._entries: dict[str, dict[int, KnowledgeEntry]] = {}
        self._current_revisions: dict[str, int] = {}
        self._chunks: dict[tuple[str, int], list[KnowledgeChunk]] = {}
        self._evidence: dict[tuple[str, int], list[KnowledgeEvidence]] = {}
        self._publication_receipts: dict[str, KnowledgePublicationReceipt] = {}
        self._publication_access: dict[str, _KnowledgeAccessSnapshot] = {}
        self._changes: list[KnowledgeChange] = []
        self._changes_by_sequence: dict[int, KnowledgeChange] = {}
        self._change_access: dict[int, tuple[_KnowledgeChangeAudience, ...]] = {}
        self._revision_change_expiration_access: dict[tuple[str, int], bool] = {}
        self._next_change_sequence = 1
        self._change_consumers: dict[str, KnowledgeChangeConsumerState] = {}
        self._acknowledged_change_claims: dict[tuple[str, str], tuple[str, int]] = {}
        self._index_readiness: list[KnowledgeIndexReadiness] = []
        self._index_readiness_by_identity: dict[str, KnowledgeIndexReadiness] = {}
        self._index_readiness_operations: dict[
            str,
            tuple[str, KnowledgeIndexReadiness],
        ] = {}
        self._next_index_readiness_sequence = 1
        if entries:
            for entry in entries:
                copied = copy_knowledge_entry(entry)
                if copied.revision != 1:
                    raise ValueError("Initial knowledge entries must be revision 1.")
                if copied.id in self._entries:
                    raise ValueError(f"Duplicate knowledge entry id {copied.id!r}.")
                self._entries[copied.id] = {1: copied}
                self._current_revisions[copied.id] = 1
                self._chunks[(copied.id, 1)] = [_default_chunk_for_entry(copied)]
                self._evidence[(copied.id, 1)] = []
                change = self._prepare_change(copied, kind=KnowledgeChangeKind.CREATED)
                self._record_change(change, before_entry=None, after_entry=copied)

    async def create_entry(
        self,
        entry: KnowledgeEntry,
        chunks: list[KnowledgeChunk] | None = None,
        *,
        evidence: list[KnowledgeEvidence] | None = None,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeEntry:
        scope = self._operation_access_scope(access_scope)
        entry = copy_knowledge_entry(entry)
        _validate_revision_append(entry, expected_revision=None)
        _require_knowledge_entry_access(scope, entry, operation="create_entry")
        existing = self._current_entry(entry.id)
        if existing is not None:
            _require_knowledge_entry_access(scope, existing, operation="create_entry")
            raise KnowledgeRevisionConflict(
                entry.id,
                expected_revision=None,
                actual_revision=existing.revision,
            )
        copied_chunks = self._revision_chunks(entry, chunks)
        copied_evidence = _copy_entry_evidence(
            entry.id,
            entry.revision,
            evidence or [],
            chunks=copied_chunks,
        )
        self._require_chunk_ids_available(
            copied_chunks,
            access_scope=scope,
            operation="create_entry",
        )
        self._require_evidence_ids_available(
            copied_evidence,
            access_scope=scope,
            operation="create_entry",
        )
        change = self._prepare_change(entry, kind=KnowledgeChangeKind.CREATED)
        self._entries[entry.id] = {1: entry}
        self._current_revisions[entry.id] = 1
        self._chunks[(entry.id, 1)] = copied_chunks
        self._evidence[(entry.id, 1)] = copied_evidence
        self._record_change(change, before_entry=None, after_entry=entry)
        return copy_knowledge_entry(entry)

    async def append_entry_revision(
        self,
        entry: KnowledgeEntry,
        chunks: list[KnowledgeChunk] | None = None,
        *,
        expected_revision: int,
        evidence: list[KnowledgeEvidence] | None = None,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeEntry:
        scope = self._operation_access_scope(access_scope)
        entry = copy_knowledge_entry(entry)
        return self._append_revision(
            entry,
            chunks=chunks,
            evidence=evidence,
            expected_revision=expected_revision,
            access_scope=scope,
            operation="append_entry_revision",
            change_kind=KnowledgeChangeKind.REVISION_APPENDED,
            inherit_evidence=False,
        )

    async def get_entry(
        self,
        entry_id: str,
        *,
        revision: int | None = None,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeEntry | None:
        scope = self._operation_access_scope(access_scope)
        clean_id = _knowledge_entry_id(entry_id)
        if revision is not None:
            _validate_knowledge_revision(revision, "revision")
        if revision is not None:
            current = self._current_entry(clean_id)
            if current is None or not _knowledge_scope_allows_entry(scope, current):
                return None
        entry = self._entry_revision(clean_id, revision)
        if entry is None or not _knowledge_scope_allows_entry(scope, entry):
            return None
        return copy_knowledge_entry(entry)

    async def transition_entry_status(
        self,
        entry_id: str,
        *,
        expected_revision: int,
        access_scope: KnowledgeAccessScope | None = None,
        from_status: KnowledgeStatus,
        to_status: KnowledgeStatus,
        expected_namespace: str | None = None,
        expected_labels: dict[str, str] | None = None,
    ) -> KnowledgeEntry:
        scope = self._operation_access_scope(access_scope)
        clean_id = _knowledge_entry_id(entry_id)
        _validate_knowledge_revision(expected_revision, "expected_revision")
        if not isinstance(from_status, KnowledgeStatus):
            raise ValueError("from_status must be a KnowledgeStatus.")
        if not isinstance(to_status, KnowledgeStatus):
            raise ValueError("to_status must be a KnowledgeStatus.")
        expected_namespace = (
            require_clean_nonblank(expected_namespace, "expected_namespace")
            if expected_namespace is not None
            else None
        )
        expected_labels = copy_label_map(expected_labels or {}, "expected_labels")
        entry = self._current_entry(clean_id)
        if entry is None:
            raise KeyError(f"Knowledge entry {clean_id!r} does not exist.")
        _require_knowledge_entry_access(scope, entry, operation="transition_entry_status")
        if entry.revision != expected_revision:
            raise KnowledgeRevisionConflict(
                clean_id,
                expected_revision=expected_revision,
                actual_revision=entry.revision,
            )
        if entry.status is not from_status:
            raise ValueError(
                f"Knowledge entry {clean_id!r} is {entry.status.value!r}, "
                f"not {from_status.value!r}."
            )
        if expected_namespace is not None and entry.namespace != expected_namespace:
            raise ValueError(f"Knowledge entry {clean_id!r} does not match expected namespace.")
        for key, value in expected_labels.items():
            if entry.labels.get(key) != value:
                raise ValueError(f"Knowledge entry {clean_id!r} does not match expected labels.")
        updated = entry.model_copy(
            update={
                "revision": _next_knowledge_revision(expected_revision),
                "status": to_status,
                "updated_at": _next_updated_at(entry),
            }
        )
        return self._append_revision(
            updated,
            chunks=None,
            evidence=None,
            expected_revision=expected_revision,
            access_scope=scope,
            operation="transition_entry_status",
            change_kind=(
                KnowledgeChangeKind.TOMBSTONED
                if to_status is KnowledgeStatus.DELETED
                else KnowledgeChangeKind.STATUS_TRANSITIONED
            ),
            inherit_evidence=True,
        )

    async def delete_entry(
        self,
        entry_id: str,
        *,
        expected_revision: int,
        access_scope: KnowledgeAccessScope | None = None,
        hard: bool = False,
    ) -> KnowledgeEntry | None:
        scope = self._operation_access_scope(access_scope)
        clean_id = _knowledge_entry_id(entry_id)
        _validate_knowledge_revision(expected_revision, "expected_revision")
        if type(hard) is not bool:
            raise ValueError("`hard` must be a boolean.")
        entry = self._current_entry(clean_id)
        if entry is None:
            return None
        _require_knowledge_entry_access(scope, entry, operation="delete_entry")
        if entry.revision != expected_revision:
            raise KnowledgeRevisionConflict(
                clean_id,
                expected_revision=expected_revision,
                actual_revision=entry.revision,
            )
        if hard:
            change = self._prepare_change(entry, kind=KnowledgeChangeKind.HARD_DELETED)
            self._entries.pop(clean_id, None)
            self._current_revisions.pop(clean_id, None)
            for key in [key for key in self._chunks if key[0] == clean_id]:
                self._chunks.pop(key, None)
            for key in [key for key in self._evidence if key[0] == clean_id]:
                self._evidence.pop(key, None)
            self._record_change(change, before_entry=entry, after_entry=None)
            return copy_knowledge_entry(entry)
        return await self.transition_entry_status(
            clean_id,
            expected_revision=expected_revision,
            access_scope=scope,
            from_status=entry.status,
            to_status=KnowledgeStatus.DELETED,
        )

    async def prune_expired(
        self,
        *,
        access_scope: KnowledgeAccessScope | None = None,
        now: datetime | None = None,
    ) -> int:
        scope = self._operation_access_scope(access_scope)
        cutoff = _knowledge_change_now(now)
        expired_entries = sorted(
            (
                entry
                for entry_id in self._entries
                if (entry := self._current_entry(entry_id)) is not None
                if entry.expires_at is not None
                and entry.expires_at <= cutoff
                and _knowledge_scope_allows_entry(scope, entry, now=cutoff)
            ),
            key=lambda entry: entry.id,
        )
        prepared_changes = [
            (entry, self._prepare_change(entry, kind=KnowledgeChangeKind.EXPIRED))
            for entry in expired_entries
        ]
        for entry, change in prepared_changes:
            entry_id = entry.id
            self._entries.pop(entry_id, None)
            self._current_revisions.pop(entry_id, None)
            for key in [key for key in self._chunks if key[0] == entry_id]:
                self._chunks.pop(key, None)
            for key in [key for key in self._evidence if key[0] == entry_id]:
                self._evidence.pop(key, None)
            self._record_change(change, before_entry=entry, after_entry=None)
        return len(expired_entries)

    async def publish_entry_revision(
        self,
        entry: KnowledgeEntry,
        chunks: list[KnowledgeChunk],
        *,
        evidence: list[KnowledgeEvidence] | None = None,
        access_scope: KnowledgeAccessScope | None = None,
        operation_id: str,
        expected_revision: int | None = None,
    ) -> KnowledgePublicationReceipt:
        scope = self._operation_access_scope(access_scope)
        (
            operation_id,
            copied_entry,
            copied_chunks,
            copied_evidence,
            request_sha256,
        ) = prepare_knowledge_publication(
            entry,
            chunks,
            evidence=evidence,
            operation_id=operation_id,
            expected_revision=expected_revision,
        )
        _require_knowledge_entry_access(scope, copied_entry, operation="publish_entry_revision")
        existing_receipt = self._publication_receipts.get(operation_id)
        if existing_receipt is not None:
            receipt_access = self._publication_access.get(operation_id)
            if receipt_access is None:
                raise KnowledgePublicationConflict("malformed_receipt")
            if not _knowledge_scope_allows_snapshot(scope, receipt_access):
                raise KnowledgeAccessDenied("publish_entry_revision")
            _validate_knowledge_publication_replay(
                existing_receipt,
                entry=copied_entry,
                chunks=copied_chunks,
                evidence=copied_evidence,
                expected_revision=expected_revision,
                request_sha256=request_sha256,
            )
            return copy_knowledge_publication_receipt(existing_receipt, replayed=True)
        existing_entry = self._current_entry(copied_entry.id)
        actual_revision = None if existing_entry is None else existing_entry.revision
        if existing_entry is not None:
            _require_knowledge_entry_access(
                scope, existing_entry, operation="publish_entry_revision"
            )
        if actual_revision != expected_revision:
            raise KnowledgeRevisionConflict(
                copied_entry.id,
                expected_revision=expected_revision,
                actual_revision=actual_revision,
            )
        if existing_entry is not None:
            _validate_revision_successor(existing_entry, copied_entry)
        self._require_chunk_ids_available(
            copied_chunks,
            access_scope=scope,
            operation="publish_entry_revision",
        )
        self._require_evidence_ids_available(
            copied_evidence,
            access_scope=scope,
            operation="publish_entry_revision",
        )
        receipt = KnowledgePublicationReceipt(
            operation_id=operation_id,
            entry_id=copied_entry.id,
            entry_revision=copied_entry.revision,
            expected_revision=expected_revision,
            request_sha256=request_sha256,
            entry_created_at=copied_entry.created_at,
            entry_updated_at=copied_entry.updated_at,
            committed_at=datetime.now(UTC),
        )
        change = self._prepare_change(
            copied_entry,
            kind=(
                KnowledgeChangeKind.CREATED
                if existing_entry is None
                else KnowledgeChangeKind.REVISION_APPENDED
            ),
            operation_id=operation_id,
            committed_at=receipt.committed_at,
        )
        self._entries.setdefault(copied_entry.id, {})[copied_entry.revision] = copied_entry
        self._chunks[(copied_entry.id, copied_entry.revision)] = copied_chunks
        self._evidence[(copied_entry.id, copied_entry.revision)] = copied_evidence
        self._current_revisions[copied_entry.id] = copied_entry.revision
        self._publication_receipts[operation_id] = receipt
        self._publication_access[operation_id] = _knowledge_access_snapshot(copied_entry)
        self._record_change(
            change,
            before_entry=existing_entry,
            after_entry=copied_entry,
        )
        return copy_knowledge_publication_receipt(receipt)

    def _append_revision(
        self,
        entry: KnowledgeEntry,
        *,
        chunks: list[KnowledgeChunk] | None,
        evidence: list[KnowledgeEvidence] | None,
        expected_revision: int,
        access_scope: KnowledgeAccessScope,
        operation: str,
        change_kind: KnowledgeChangeKind,
        inherit_evidence: bool,
    ) -> KnowledgeEntry:
        _validate_revision_append(entry, expected_revision=expected_revision)
        current = self._current_entry(entry.id)
        if current is None:
            raise KnowledgeRevisionConflict(
                entry.id,
                expected_revision=expected_revision,
                actual_revision=None,
            )
        _require_knowledge_entry_access(access_scope, current, operation=operation)
        if current.revision != expected_revision:
            raise KnowledgeRevisionConflict(
                entry.id,
                expected_revision=expected_revision,
                actual_revision=current.revision,
            )
        _validate_revision_successor(current, entry)
        _require_knowledge_successor_access(access_scope, entry, operation=operation)
        previous_chunks = self._chunks.get((current.id, current.revision), [])
        copied_chunks = self._revision_chunks(entry, chunks, previous=current)
        if inherit_evidence:
            if evidence is not None:
                raise ValueError("Lifecycle evidence inheritance cannot accept evidence.")
            copied_evidence = _copy_evidence_for_revision(
                self._evidence.get((current.id, current.revision), []),
                entry=entry,
                previous_chunks=previous_chunks,
                chunks=copied_chunks,
            )
        else:
            copied_evidence = _copy_entry_evidence(
                entry.id,
                entry.revision,
                evidence or [],
                chunks=copied_chunks,
            )
        self._require_chunk_ids_available(
            copied_chunks,
            access_scope=access_scope,
            operation=operation,
        )
        self._require_evidence_ids_available(
            copied_evidence,
            access_scope=access_scope,
            operation=operation,
        )
        change = self._prepare_change(entry, kind=change_kind)
        self._entries[entry.id][entry.revision] = entry
        self._chunks[(entry.id, entry.revision)] = copied_chunks
        self._evidence[(entry.id, entry.revision)] = copied_evidence
        self._current_revisions[entry.id] = entry.revision
        self._record_change(change, before_entry=current, after_entry=entry)
        return copy_knowledge_entry(entry)

    def _require_evidence_ids_available(
        self,
        evidence: list[KnowledgeEvidence],
        *,
        access_scope: KnowledgeAccessScope,
        operation: str,
    ) -> None:
        proposed_ids = {item.id for item in evidence}
        occupied_entry_ids = {
            entry_id
            for (entry_id, _), stored in self._evidence.items()
            if any(item.id in proposed_ids for item in stored)
        }
        for occupied_entry_id in sorted(occupied_entry_ids):
            owner = self._current_entry(occupied_entry_id)
            if owner is None:
                raise KnowledgeEvidenceConflict(operation)
            _require_knowledge_entry_access(
                access_scope,
                owner,
                operation=operation,
            )
        if occupied_entry_ids:
            raise KnowledgeEvidenceConflict(operation)

    def _prepare_change(
        self,
        entry: KnowledgeEntry,
        *,
        kind: KnowledgeChangeKind,
        operation_id: str | None = None,
        committed_at: datetime | None = None,
    ) -> KnowledgeChange:
        sequence = self._next_change_sequence
        if sequence > MAX_KNOWLEDGE_CHANGE_SEQUENCE:
            raise RuntimeError("Knowledge change sequence is exhausted.")
        self._next_change_sequence += 1
        return KnowledgeChange(
            id=f"kchg_{uuid4().hex}",
            sequence=sequence,
            kind=kind,
            entry_id=entry.id,
            entry_revision=entry.revision,
            committed_at=(datetime.now(UTC) if committed_at is None else committed_at),
            operation_id=operation_id,
        )

    def _record_change(
        self,
        change: KnowledgeChange,
        *,
        before_entry: KnowledgeEntry | None,
        after_entry: KnowledgeEntry | None,
    ) -> None:
        copied = copy_knowledge_change(change)
        self._changes.append(copied)
        self._changes_by_sequence[copied.sequence] = copied
        self._change_access[change.sequence] = _knowledge_change_audiences(
            copied,
            before_entry=before_entry,
            after_entry=after_entry,
            before_requires_include_expired=(
                None
                if before_entry is None
                else self._revision_change_expiration_access.get(
                    (before_entry.id, before_entry.revision)
                )
            ),
        )
        if after_entry is not None:
            after_audience = next(
                audience
                for audience in self._change_access[change.sequence]
                if audience.kind == "after"
            )
            self._revision_change_expiration_access[(after_entry.id, after_entry.revision)] = (
                after_audience.requires_include_expired
            )

    def _change_by_sequence(self, sequence: int) -> KnowledgeChange | None:
        return self._changes_by_sequence.get(sequence)

    def _accessible_changes(
        self,
        scope: KnowledgeAccessScope,
        *,
        after_sequence: int,
        limit: int | None = None,
    ) -> list[KnowledgeChange]:
        result: list[KnowledgeChange] = []
        for change in self._changes:
            audiences = self._change_access.get(change.sequence, ())
            if change.sequence <= after_sequence or not any(
                _knowledge_scope_allows_change_audience(scope, audience) for audience in audiences
            ):
                continue
            result.append(change)
            if limit is not None and len(result) >= limit:
                break
        return result

    def _require_chunk_ids_available(
        self,
        chunks: list[KnowledgeChunk],
        *,
        access_scope: KnowledgeAccessScope,
        operation: str,
    ) -> None:
        proposed_ids = {chunk.id for chunk in chunks}
        occupied_entry_ids: set[str] = set()
        for (existing_entry_id, _), existing_chunks in self._chunks.items():
            if any(chunk.id in proposed_ids for chunk in existing_chunks):
                occupied_entry_ids.add(existing_entry_id)
        for occupied_entry_id in sorted(occupied_entry_ids):
            owner = self._current_entry(occupied_entry_id)
            if owner is None:
                raise KnowledgeChunkConflict(operation)
            _require_knowledge_entry_access(
                access_scope,
                owner,
                operation=operation,
            )
        if occupied_entry_ids:
            raise KnowledgeChunkConflict(operation)

    def _current_entry(self, entry_id: str) -> KnowledgeEntry | None:
        revision = self._current_revisions.get(entry_id)
        if revision is None:
            return None
        return self._entries[entry_id][revision]

    def _entry_revision(
        self,
        entry_id: str,
        revision: int | None,
    ) -> KnowledgeEntry | None:
        selected_revision = self._current_revisions.get(entry_id) if revision is None else revision
        if selected_revision is None:
            return None
        return self._entries.get(entry_id, {}).get(selected_revision)

    def _revision_chunks(
        self,
        entry: KnowledgeEntry,
        chunks: list[KnowledgeChunk] | None,
        *,
        previous: KnowledgeEntry | None = None,
    ) -> list[KnowledgeChunk]:
        if chunks is not None:
            return _copy_entry_chunks(entry.id, entry.revision, chunks)
        if previous is None:
            return [_default_chunk_for_entry(entry)]
        previous_chunks = self._chunks.get((previous.id, previous.revision), [])
        if _has_only_default_chunk(previous, previous_chunks):
            return [_default_chunk_for_entry(entry)]
        return _copy_chunks_for_revision(previous_chunks, entry)

    async def load_entry_publication_receipt(
        self,
        operation_id: str,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgePublicationReceipt | None:
        scope = self._operation_access_scope(access_scope)
        operation_id = _knowledge_publication_operation_id(operation_id)
        receipt = self._publication_receipts.get(operation_id)
        if receipt is None:
            return None
        snapshot = self._publication_access.get(operation_id)
        if snapshot is None or not _knowledge_scope_allows_snapshot(scope, snapshot):
            return None
        return copy_knowledge_publication_receipt(receipt)

    async def read_evidence(
        self,
        entry_id: str,
        *,
        revision: int | None = None,
        access_scope: KnowledgeAccessScope | None = None,
        max_records: int = DEFAULT_KNOWLEDGE_LIMIT,
        max_bytes: int = DEFAULT_KNOWLEDGE_MAX_BYTES,
    ) -> KnowledgeEvidenceResult | None:
        scope = self._operation_access_scope(access_scope)
        clean_id = _knowledge_entry_id(entry_id)
        if revision is not None:
            _validate_knowledge_revision(revision, "revision")
        _validate_positive_int(max_records, "max_records")
        _validate_positive_int(max_bytes, "max_bytes")
        access_now = datetime.now(UTC)
        if revision is not None:
            current = self._current_entry(clean_id)
            if current is None or not _knowledge_scope_allows_entry(
                scope,
                current,
                now=access_now,
            ):
                return None
        entry = self._entry_revision(clean_id, revision)
        if entry is None or not _knowledge_scope_allows_entry(
            scope,
            entry,
            now=access_now,
        ):
            return None
        stored = self._evidence.get((clean_id, entry.revision), [])
        selected = _bounded_knowledge_evidence(
            stored,
            max_records=max_records,
            max_bytes=max_bytes,
        )
        return KnowledgeEvidenceResult(
            entry_id=entry.id,
            entry_revision=entry.revision,
            evidence=selected,
            truncated=len(selected) < len(stored),
            limit=max_records,
            max_bytes=max_bytes,
            total_evidence_known=len(stored),
        )

    async def read_changes(
        self,
        *,
        after_sequence: int = 0,
        limit: int = DEFAULT_KNOWLEDGE_LIMIT,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeChangeBatch:
        scope = self._operation_access_scope(access_scope)
        _validate_knowledge_change_sequence(after_sequence, "after_sequence")
        _validate_knowledge_change_limit(limit)
        current_sequence = self._next_change_sequence - 1
        if after_sequence > current_sequence:
            raise ValueError(
                "`after_sequence` cannot exceed the current knowledge change sequence."
            )
        selected: list[KnowledgeChange] = []
        high_water = 0
        for change in self._changes:
            audiences = self._change_access.get(change.sequence, ())
            if not any(
                _knowledge_scope_allows_change_audience(scope, audience) for audience in audiences
            ):
                continue
            high_water = max(high_water, change.sequence)
            if change.sequence > after_sequence and len(selected) <= limit:
                selected.append(change)
        truncated = len(selected) > limit
        selected = selected[:limit]
        next_after = selected[-1].sequence if truncated else max(after_sequence, high_water)
        return KnowledgeChangeBatch(
            changes=selected,
            after_sequence=after_sequence,
            next_after_sequence=next_after,
            high_water_sequence=high_water,
            truncated=truncated,
            limit=limit,
        )

    async def claim_change(
        self,
        consumer_id: str,
        worker_id: str,
        *,
        lease_seconds: float = 300.0,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeChangeClaim | None:
        scope = self._operation_access_scope(access_scope)
        consumer_id = _knowledge_change_identity(consumer_id, "consumer_id")
        worker_id = _knowledge_change_identity(worker_id, "worker_id")
        lease_seconds = _knowledge_change_lease_seconds(lease_seconds)
        current_time = self._clock()
        scope_sha256 = _knowledge_access_scope_sha256(scope)
        state = self._change_consumers.get(consumer_id)
        if state is None:
            state = KnowledgeChangeConsumerState(
                consumer_id=consumer_id,
                access_scope_sha256=scope_sha256,
                updated_at=current_time,
            )
        elif state.access_scope_sha256 != scope_sha256:
            raise KnowledgeChangeConsumerConflict("access_scope_mismatch")

        if state.pending_change_sequence is not None:
            stored_change = self._change_by_sequence(state.pending_change_sequence)
            audiences = self._change_access.get(state.pending_change_sequence, ())
            still_allowed = stored_change is not None and any(
                _knowledge_scope_allows_change_audience(scope, audience) for audience in audiences
            )
            assert state.lease_expires_at is not None
            if still_allowed and state.lease_expires_at > current_time:
                if state.pending_worker_id != worker_id:
                    self._change_consumers[consumer_id] = state
                    return None
                assert state.pending_claim_id is not None
                assert state.claimed_at is not None
                claim = KnowledgeChangeClaim(
                    consumer_id=consumer_id,
                    worker_id=worker_id,
                    claim_id=state.pending_claim_id,
                    change=stored_change,
                    attempt=state.pending_attempt,
                    claimed_at=state.claimed_at,
                    lease_expires_at=state.lease_expires_at,
                )
                self._change_consumers[consumer_id] = state
                return claim
            state = state.model_copy(
                update={
                    "pending_change_sequence": None,
                    "pending_claim_id": None,
                    "pending_worker_id": None,
                    "claimed_at": None,
                    "lease_expires_at": None,
                    "pending_attempt": (state.pending_attempt if still_allowed else 0),
                    "updated_at": current_time,
                }
            )

        candidates = self._accessible_changes(
            scope,
            after_sequence=state.cursor_sequence,
            limit=1,
        )
        if not candidates:
            self._change_consumers[consumer_id] = state
            return None
        change = candidates[0]
        claim_id = f"kclaim_{uuid4().hex}"
        claimed_at = current_time
        lease_expires_at = claimed_at + timedelta(seconds=lease_seconds)
        attempt = state.pending_attempt + 1
        state = state.model_copy(
            update={
                "pending_change_sequence": change.sequence,
                "pending_claim_id": claim_id,
                "pending_worker_id": worker_id,
                "pending_attempt": attempt,
                "claimed_at": claimed_at,
                "lease_expires_at": lease_expires_at,
                "updated_at": current_time,
            }
        )
        self._change_consumers[consumer_id] = state
        return KnowledgeChangeClaim(
            consumer_id=consumer_id,
            worker_id=worker_id,
            claim_id=claim_id,
            change=change,
            attempt=attempt,
            claimed_at=claimed_at,
            lease_expires_at=lease_expires_at,
        )

    async def initialize_change_consumer(
        self,
        consumer_id: str,
        *,
        baseline_sequence: int,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeChangeConsumerState:
        scope = self._operation_access_scope(access_scope)
        consumer_id = _knowledge_change_identity(consumer_id, "consumer_id")
        _validate_knowledge_change_sequence(baseline_sequence, "baseline_sequence")
        current_time = self._clock()
        current_sequence = self._next_change_sequence - 1
        if baseline_sequence > current_sequence:
            raise ValueError(
                "`baseline_sequence` cannot exceed the current knowledge change sequence."
            )
        state = _initialize_knowledge_change_consumer_state(
            self._change_consumers.get(consumer_id),
            consumer_id=consumer_id,
            access_scope_sha256=_knowledge_access_scope_sha256(scope),
            baseline_sequence=baseline_sequence,
            now=current_time,
        )
        self._change_consumers[consumer_id] = state
        return copy_knowledge_change_consumer_state(state)

    async def acknowledge_change(
        self,
        claim: KnowledgeChangeClaim,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeChangeConsumerState:
        scope = self._operation_access_scope(access_scope)
        claim = copy_knowledge_change_claim(claim)
        current_time = self._clock()
        state = self._change_consumers.get(claim.consumer_id)
        if state is None or state.access_scope_sha256 != _knowledge_access_scope_sha256(scope):
            raise KnowledgeChangeConsumerConflict("unknown_consumer")
        claim_sha256 = _knowledge_change_claim_sha256(claim)
        acknowledged = self._acknowledged_change_claims.get((claim.consumer_id, claim.claim_id))
        if acknowledged is not None:
            if acknowledged != (claim_sha256, claim.change.sequence):
                raise KnowledgeChangeConsumerConflict("stale_claim")
            if state.cursor_sequence < claim.change.sequence:
                raise RuntimeError("Knowledge change acknowledgement is ahead of its consumer.")
            return copy_knowledge_change_consumer_state(state)
        self._require_live_change_claim(state, claim, now=current_time)
        state = state.model_copy(
            update={
                "cursor_sequence": claim.change.sequence,
                "pending_change_sequence": None,
                "pending_claim_id": None,
                "pending_worker_id": None,
                "pending_attempt": 0,
                "claimed_at": None,
                "lease_expires_at": None,
                "last_acknowledged_claim_id": claim.claim_id,
                "updated_at": current_time,
            }
        )
        self._change_consumers[claim.consumer_id] = state
        self._acknowledged_change_claims[(claim.consumer_id, claim.claim_id)] = (
            claim_sha256,
            claim.change.sequence,
        )
        return copy_knowledge_change_consumer_state(state)

    async def release_change(
        self,
        claim: KnowledgeChangeClaim,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeChangeConsumerState:
        scope = self._operation_access_scope(access_scope)
        claim = copy_knowledge_change_claim(claim)
        current_time = self._clock()
        state = self._change_consumers.get(claim.consumer_id)
        if state is None or state.access_scope_sha256 != _knowledge_access_scope_sha256(scope):
            raise KnowledgeChangeConsumerConflict("unknown_consumer")
        self._require_live_change_claim(state, claim, now=current_time)
        state = state.model_copy(
            update={
                "pending_change_sequence": None,
                "pending_claim_id": None,
                "pending_worker_id": None,
                "claimed_at": None,
                "lease_expires_at": None,
                "updated_at": current_time,
            }
        )
        self._change_consumers[claim.consumer_id] = state
        return copy_knowledge_change_consumer_state(state)

    async def load_change_consumer_state(
        self,
        consumer_id: str,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeChangeConsumerState | None:
        scope = self._operation_access_scope(access_scope)
        consumer_id = _knowledge_change_identity(consumer_id, "consumer_id")
        state = self._change_consumers.get(consumer_id)
        if state is None:
            return None
        if state.access_scope_sha256 != _knowledge_access_scope_sha256(scope):
            return None
        return copy_knowledge_change_consumer_state(state)

    async def publish_index_readiness(
        self,
        update: KnowledgeIndexReadinessUpdate,
        *,
        expected_sequence: int | None,
        operation_id: str,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeIndexReadiness:
        scope = self._operation_access_scope(access_scope)
        update = copy_knowledge_index_readiness_update(update)
        operation_id = _bounded_knowledge_index_identity(operation_id, "operation_id")
        if expected_sequence is not None:
            _validate_knowledge_index_sequence(
                expected_sequence,
                "expected_sequence",
                allow_zero=False,
            )
        update_sha256 = _knowledge_index_readiness_update_sha256(update)
        replay = self._index_readiness_operations.get(operation_id)
        if replay is not None:
            stored_sha256, readiness = replay
            if stored_sha256 != update_sha256:
                raise KnowledgeIndexReadinessConflict("operation_reuse")
            if not self._index_identity_is_accessible(scope, update.identity):
                raise KnowledgeAccessDenied("publish_index_readiness")
            return copy_knowledge_index_readiness(readiness)
        if not self._index_identity_is_accessible(
            scope,
            update.identity,
            require_current=True,
        ):
            raise KnowledgeIndexReadinessConflict("stale_identity")
        identity_sha256 = _knowledge_embedding_identity_sha256(update.identity)
        current = self._index_readiness_by_identity.get(identity_sha256)
        _validate_knowledge_index_readiness_transition(
            current,
            update,
            expected_sequence=expected_sequence,
        )
        sequence = self._next_index_readiness_sequence
        if sequence > MAX_KNOWLEDGE_CHANGE_SEQUENCE:
            raise OverflowError("Knowledge index readiness sequence is exhausted.")
        readiness = KnowledgeIndexReadiness(
            sequence=sequence,
            identity=update.identity,
            state=update.state,
            attempt_id=update.attempt_id,
            failure_code=update.failure_code,
            operation_id=operation_id,
            published_at=self._clock(),
        )
        self._next_index_readiness_sequence += 1
        self._index_readiness.append(readiness)
        self._index_readiness_by_identity[identity_sha256] = readiness
        self._index_readiness_operations[operation_id] = (update_sha256, readiness)
        return copy_knowledge_index_readiness(readiness)

    async def load_index_readiness(
        self,
        identity: KnowledgeEmbeddingIdentity,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeIndexReadiness | None:
        scope = self._operation_access_scope(access_scope)
        identity = copy_knowledge_embedding_identity(identity)
        if not self._index_identity_is_accessible(scope, identity):
            return None
        readiness = self._index_readiness_by_identity.get(
            _knowledge_embedding_identity_sha256(identity)
        )
        if readiness is None:
            return None
        return copy_knowledge_index_readiness(readiness)

    async def read_index_readiness(
        self,
        *,
        after_sequence: int = 0,
        limit: int = DEFAULT_KNOWLEDGE_LIMIT,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeIndexReadinessBatch:
        scope = self._operation_access_scope(access_scope)
        _validate_knowledge_index_sequence(after_sequence, "after_sequence")
        _validate_knowledge_index_readiness_limit(limit)
        current_sequence = self._next_index_readiness_sequence - 1
        if after_sequence > current_sequence:
            raise ValueError(
                "`after_sequence` cannot exceed the current knowledge index readiness sequence."
            )
        accessible = [
            item
            for item in self._index_readiness
            if self._index_identity_is_accessible(scope, item.identity)
        ]
        high_water = max((item.sequence for item in accessible), default=0)
        selected = [item for item in accessible if item.sequence > after_sequence]
        truncated = len(selected) > limit
        selected = selected[:limit]
        next_after = selected[-1].sequence if truncated else max(after_sequence, high_water)
        return KnowledgeIndexReadinessBatch(
            readiness=selected,
            after_sequence=after_sequence,
            next_after_sequence=next_after,
            high_water_sequence=high_water,
            truncated=truncated,
            limit=limit,
        )

    def _index_identity_is_accessible(
        self,
        scope: KnowledgeAccessScope,
        identity: KnowledgeEmbeddingIdentity,
        *,
        require_current: bool = False,
    ) -> bool:
        current = self._current_entry(identity.entry_id)
        if current is None or not _knowledge_scope_allows_entry(scope, current):
            return False
        if require_current and current.revision != identity.entry_revision:
            return False
        revision = self._entry_revision(identity.entry_id, identity.entry_revision)
        if revision is None or not _knowledge_scope_allows_entry(scope, revision):
            return False
        if identity.chunk_id is None:
            return True
        chunk = next(
            (
                candidate
                for candidate in self._chunks.get(
                    (identity.entry_id, identity.entry_revision),
                    [],
                )
                if candidate.id == identity.chunk_id
            ),
            None,
        )
        if chunk is None:
            return False
        if identity.projection_type == KNOWLEDGE_CHUNK_TEXT_PROJECTION:
            return identity.projection_content_hash == _knowledge_chunk_content_hash(chunk)
        return True

    def _require_matching_change_claim(
        self,
        state: KnowledgeChangeConsumerState,
        claim: KnowledgeChangeClaim,
    ) -> None:
        stored_change = self._change_by_sequence(claim.change.sequence)
        if (
            state.pending_change_sequence != claim.change.sequence
            or state.pending_claim_id != claim.claim_id
            or state.pending_worker_id != claim.worker_id
            or state.pending_attempt != claim.attempt
            or stored_change != claim.change
        ):
            raise KnowledgeChangeConsumerConflict("stale_claim")

    def _require_live_change_claim(
        self,
        state: KnowledgeChangeConsumerState,
        claim: KnowledgeChangeClaim,
        *,
        now: datetime,
    ) -> None:
        self._require_matching_change_claim(state, claim)
        if state.lease_expires_at is None or state.lease_expires_at <= now:
            raise KnowledgeChangeConsumerConflict("expired_claim")

    async def read_chunks(
        self,
        entry_id: str,
        *,
        revision: int | None = None,
        access_scope: KnowledgeAccessScope | None = None,
        chunk_index: int | None = None,
        around: int = 0,
        max_chunks: int = DEFAULT_KNOWLEDGE_LIMIT,
        max_bytes: int = DEFAULT_KNOWLEDGE_MAX_BYTES,
    ) -> list[KnowledgeChunk]:
        scope = self._operation_access_scope(access_scope)
        clean_id = _knowledge_entry_id(entry_id)
        if revision is not None:
            _validate_knowledge_revision(revision, "revision")
        if revision is not None:
            current = self._current_entry(clean_id)
            if current is None or not _knowledge_scope_allows_entry(scope, current):
                return []
        entry = self._entry_revision(clean_id, revision)
        if entry is None or not _knowledge_scope_allows_entry(scope, entry):
            return []
        if chunk_index is not None:
            _validate_nonnegative_int(chunk_index, "chunk_index")
        _validate_nonnegative_int(around, "around")
        if chunk_index is None and around != 0:
            raise ValueError("`around` requires `chunk_index`.")
        _validate_positive_int(max_chunks, "max_chunks")
        _validate_positive_int(max_bytes, "max_bytes")
        start_index = 0 if chunk_index is None else max(0, chunk_index - around)
        end_index = None if chunk_index is None else chunk_index + around
        chunks = self._chunks.get((clean_id, entry.revision), [])
        if chunk_index is not None:
            chunks = _center_chunk_window(chunks, chunk_index=chunk_index, max_chunks=max_chunks)
        return _bounded_chunks(
            chunks,
            start_index=start_index,
            end_index=end_index,
            max_chunks=max_chunks,
            max_bytes=max_bytes,
        )

    async def search(
        self,
        query: KnowledgeQuery,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeSearchResult:
        scope = self._operation_access_scope(access_scope)
        knowledge_query = copy_knowledge_query(query)
        if knowledge_query.mode not in {KnowledgeSearchMode.AUTO, KnowledgeSearchMode.KEYWORD}:
            raise ValueError("InMemoryKnowledgeStore supports only auto and keyword search modes.")
        terms = _knowledge_query_terms(knowledge_query)
        scored: list[tuple[float, KnowledgeEntry, KnowledgeChunk | None, str, str]] = []
        for entry_id in self._entries:
            entry = self._current_entry(entry_id)
            if entry is None:  # pragma: no cover - internal invariant
                continue
            if not _knowledge_scope_allows_entry(scope, entry):
                continue
            if not _entry_matches_query(entry, knowledge_query):
                continue
            chunks = self._chunks.get((entry.id, entry.revision), [])
            if _entry_matches_none_terms(entry, chunks, terms):
                continue
            score, chunk, reason, preview_text = _score_entry(entry, chunks, knowledge_query)
            if score <= 0:
                continue
            scored.append((score, entry, chunk, reason, preview_text))
        scored.sort(
            key=lambda item: (
                -item[0],
                -(item[1].importance or 0.0),
                -item[1].updated_at.timestamp(),
                item[1].id,
            )
        )
        hits: list[KnowledgeHit] = []
        remaining = knowledge_query.max_bytes
        truncated = False
        for rank, (score, entry, chunk, reason, preview_text) in enumerate(
            scored[: knowledge_query.limit], start=1
        ):
            if remaining <= 0:
                truncated = True
                break
            source_bytes = len(preview_text.encode("utf-8"))
            preview = _truncate_text_to_bytes(preview_text, remaining)
            if not preview:
                truncated = True
                break
            preview_complete = len(preview.encode("utf-8")) == source_bytes
            if not preview_complete:
                truncated = True
            remaining -= len(preview.encode("utf-8"))
            hits.append(
                KnowledgeHit(
                    entry=entry,
                    chunk=chunk,
                    score=score,
                    score_kind="inmemory_keyword",
                    rank=rank,
                    reason=reason,
                    text_preview=preview,
                    text_preview_complete=preview_complete,
                )
            )
        return KnowledgeSearchResult(
            query=knowledge_query,
            hits=hits,
            truncated=truncated or len(hits) < len(scored),
            limit=knowledge_query.limit,
            max_bytes=knowledge_query.max_bytes,
            total_hits_known=len(scored),
        )

    async def list_entries(
        self,
        query: KnowledgeListQuery,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeListResult:
        scope = self._operation_access_scope(access_scope)
        knowledge_query = copy_knowledge_list_query(query)
        entries = [
            entry
            for entry_id in self._entries
            if (entry := self._current_entry(entry_id)) is not None
            if _knowledge_scope_allows_entry(scope, entry)
            if _entry_matches_list_query(entry, knowledge_query)
        ]
        entries.sort(
            key=lambda entry: (
                -(entry.importance or 0.0),
                -entry.updated_at.timestamp(),
                entry.id,
            )
        )
        facets, facets_truncated = _knowledge_facets(
            entries,
            knowledge_query.group_by,
            limit=knowledge_query.limit,
        )
        items: list[KnowledgeListItem] = []
        remaining = knowledge_query.max_bytes
        truncated = False
        for entry in entries[: knowledge_query.limit]:
            if remaining <= 0:
                truncated = True
                break
            preview_source = entry.title or entry.text
            preview = _truncate_text_to_bytes(preview_source, remaining)
            if not preview:
                truncated = True
                break
            preview_complete = len(preview.encode("utf-8")) == len(preview_source.encode("utf-8"))
            if not preview_complete:
                truncated = True
            remaining -= len(preview.encode("utf-8"))
            items.append(
                KnowledgeListItem(
                    entry=entry,
                    chunk_count=len(self._chunks.get((entry.id, entry.revision), [])),
                    text_preview=preview,
                    text_preview_complete=preview_complete,
                )
            )
        return KnowledgeListResult(
            query=knowledge_query,
            entries=items,
            facets=facets,
            facets_truncated=facets_truncated,
            truncated=truncated or len(items) < len(entries) or facets_truncated,
            limit=knowledge_query.limit,
            max_bytes=knowledge_query.max_bytes,
            total_entries_known=len(entries),
        )


class InMemoryEmbeddingKnowledgeStore(InMemoryKnowledgeStore):
    """In-memory knowledge store with opt-in embedding search.

    This backend is intended for tests, demos, and small single-process apps. It
    keeps vectors in memory and does not persist them. Durable production vector
    search should use a store with a real vector index.
    """

    def __init__(
        self,
        *,
        embedding_provider: TextEmbeddingProvider,
        embedding_model: str,
        embedding_dimensions: int,
        entries: list[KnowledgeEntry] | None = None,
        access_scope: KnowledgeAccessScope | None = None,
        clock: Callable[[], datetime] | None = None,
        hybrid_keyword_weight: float = 0.35,
        semantic_min_score: float = 0.55,
    ) -> None:
        if not isinstance(embedding_provider, TextEmbeddingProvider):
            raise TypeError("embedding_provider must implement TextEmbeddingProvider.")
        self.embedding_provider = embedding_provider
        self.embedding_model = require_clean_nonblank(embedding_model, "embedding_model")
        _validate_positive_int(embedding_dimensions, "embedding_dimensions")
        self.embedding_dimensions = embedding_dimensions
        self.hybrid_keyword_weight = _validate_nonnegative_float(
            hybrid_keyword_weight,
            "hybrid_keyword_weight",
        )
        self.semantic_min_score = _validate_unit_float(
            semantic_min_score,
            "semantic_min_score",
        )
        self._chunk_embeddings: dict[str, _StoredChunkEmbedding] = {}
        super().__init__(entries, access_scope=access_scope, clock=clock)

    def supported_search_modes(self) -> tuple[KnowledgeSearchMode, ...]:
        return (
            KnowledgeSearchMode.AUTO,
            KnowledgeSearchMode.KEYWORD,
            KnowledgeSearchMode.SEMANTIC,
            KnowledgeSearchMode.HYBRID,
        )

    async def process_embedding_changes(
        self,
        consumer_id: str,
        worker_id: str,
        *,
        limit: int = DEFAULT_KNOWLEDGE_LIMIT,
        record_limit: int = DEFAULT_KNOWLEDGE_EMBEDDING_WORK_RECORD_LIMIT,
        lease_seconds: float = 300.0,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeEmbeddingWorkerResult:
        """Consume a bounded page of canonical changes into the embedding index."""

        _validate_knowledge_change_limit(limit)
        _validate_knowledge_embedding_work_record_limit(record_limit)
        scope = self._operation_access_scope(access_scope)
        claimed_changes = 0
        acknowledged_changes = 0
        indexed_records = 0
        failed_records = 0
        removed_records = 0
        processed_records = 0
        for _ in range(limit):
            claim = await self.claim_change(
                consumer_id,
                worker_id,
                lease_seconds=lease_seconds,
                access_scope=scope,
            )
            if claim is None:
                break
            claimed_changes += 1
            try:
                current = self._current_entry(claim.change.entry_id)
                remaining = record_limit - processed_records
                if current is None or current.status is KnowledgeStatus.DELETED:
                    removed, cleanup_truncated = self._drop_entry_embeddings(
                        claim.change.entry_id,
                        limit=remaining,
                    )
                    removed_records += removed
                    processed_records += removed
                elif (
                    current.revision != claim.change.entry_revision
                    or not _knowledge_scope_allows_entry(scope, current)
                ):
                    removed, cleanup_truncated = self._drop_stale_entry_embeddings(
                        current.id,
                        limit=remaining,
                    )
                    removed_records += removed
                    processed_records += removed
                else:
                    chunks = [
                        copy_knowledge_chunk(chunk)
                        for chunk in self._chunks.get(
                            (current.id, current.revision),
                            [],
                        )
                    ]
                    chunks, truncated = self._embedding_work_candidates(
                        chunks,
                        limit=record_limit - processed_records,
                    )
                    indexed, failed = await self._index_chunks_with_readiness(
                        chunks,
                        attempt_id=claim.claim_id,
                        operation_prefix=f"kidx:{claim.claim_id}",
                        access_scope=scope,
                    )
                    indexed_records += indexed
                    failed_records += failed
                    processed_records += len(chunks)
                    removed, cleanup_truncated = self._drop_stale_entry_embeddings(
                        current.id,
                        limit=record_limit - processed_records,
                    )
                    removed_records += removed
                    processed_records += removed
                    cleanup_truncated = truncated or cleanup_truncated
                if cleanup_truncated:
                    await self.release_change(claim, access_scope=scope)
                    break
                await self.acknowledge_change(claim, access_scope=scope)
                acknowledged_changes += 1
                if processed_records >= record_limit:
                    break
            except Exception:
                with suppress(KnowledgeChangeConsumerConflict):
                    await self.release_change(claim, access_scope=scope)
                raise
        return KnowledgeEmbeddingWorkerResult(
            consumer_id=consumer_id,
            worker_id=worker_id,
            claimed_changes=claimed_changes,
            acknowledged_changes=acknowledged_changes,
            indexed_records=indexed_records,
            failed_records=failed_records,
            removed_records=removed_records,
            limit=limit,
            processed_records=processed_records,
            record_limit=record_limit,
        )

    def _embedding_work_candidates(
        self,
        chunks: list[KnowledgeChunk],
        *,
        limit: int,
    ) -> tuple[list[KnowledgeChunk], bool]:
        candidates: list[KnowledgeChunk] = []
        for chunk in chunks:
            identity = knowledge_chunk_embedding_identity(
                chunk,
                embedding_model=self.embedding_model,
                dimensions=self.embedding_dimensions,
            )
            identity_sha256 = _knowledge_embedding_identity_sha256(identity)
            readiness = self._index_readiness_by_identity.get(identity_sha256)
            stored = self._chunk_embeddings.get(identity_sha256)
            if (
                readiness is not None
                and readiness.state is KnowledgeIndexState.READY
                and stored is not None
                and stored["identity"] == identity
            ):
                continue
            candidates.append(copy_knowledge_chunk(chunk))
            if len(candidates) > limit:
                break
        return candidates[:limit], len(candidates) > limit

    async def backfill_embeddings(
        self,
        query: KnowledgeListQuery | None = None,
        *,
        access_scope: KnowledgeAccessScope | None = None,
        limit: int = DEFAULT_KNOWLEDGE_EMBEDDING_WORK_RECORD_LIMIT,
        refresh_existing: bool = False,
        cursor: str | None = None,
    ) -> KnowledgeEmbeddingBackfillResult:
        """Repair or refresh one bounded deterministic page of current chunk vectors."""

        _validate_knowledge_embedding_work_record_limit(limit, field_name="limit")
        if type(refresh_existing) is not bool:
            raise ValueError("`refresh_existing` must be a boolean.")
        scope = self._operation_access_scope(access_scope)
        knowledge_query = copy_knowledge_list_query(query or KnowledgeListQuery())
        fingerprint = _knowledge_embedding_backfill_fingerprint(
            knowledge_query,
            scope,
            refresh_existing=refresh_existing,
            embedding_model=self.embedding_model,
            embedding_dimensions=self.embedding_dimensions,
        )
        after = _decode_knowledge_embedding_backfill_cursor(
            cursor,
            fingerprint=fingerprint,
        )
        after_key = (
            None
            if after is None
            else _knowledge_embedding_backfill_sort_key(
                importance=after.importance,
                updated_at=after.updated_at,
                entry_id=after.entry_id,
                chunk_index=after.chunk_index,
                chunk_id=after.chunk_id,
            )
        )
        entries = [
            entry
            for entry_id in self._entries
            if (entry := self._current_entry(entry_id)) is not None
            if _knowledge_scope_allows_entry(scope, entry)
            if _entry_matches_list_query(entry, knowledge_query)
        ]
        entries.sort(
            key=lambda entry: _knowledge_embedding_backfill_sort_key(
                importance=entry.importance or 0.0,
                updated_at=entry.updated_at,
                entry_id=entry.id,
                chunk_index=0,
                chunk_id="",
            )
        )
        candidates: list[tuple[KnowledgeEntry, KnowledgeChunk]] = []
        for entry in entries:
            for chunk in sorted(
                self._chunks.get((entry.id, entry.revision), []),
                key=lambda item: (item.chunk_index, item.id),
            ):
                candidate_key = _knowledge_embedding_backfill_sort_key(
                    importance=entry.importance or 0.0,
                    updated_at=entry.updated_at,
                    entry_id=entry.id,
                    chunk_index=chunk.chunk_index,
                    chunk_id=chunk.id,
                )
                if after_key is not None and candidate_key <= after_key:
                    continue
                identity = knowledge_chunk_embedding_identity(
                    chunk,
                    embedding_model=self.embedding_model,
                    dimensions=self.embedding_dimensions,
                )
                identity_sha256 = _knowledge_embedding_identity_sha256(identity)
                readiness = self._index_readiness_by_identity.get(identity_sha256)
                stored = self._chunk_embeddings.get(identity_sha256)
                if (
                    not refresh_existing
                    and readiness is not None
                    and readiness.state is KnowledgeIndexState.READY
                    and stored is not None
                    and stored["identity"] == identity
                ):
                    continue
                candidates.append((entry, copy_knowledge_chunk(chunk)))
                if len(candidates) > limit:
                    break
            if len(candidates) > limit:
                break
        page = candidates[:limit]
        chunks = [chunk for _, chunk in page]
        next_cursor = (
            _encode_knowledge_embedding_backfill_cursor(
                fingerprint=fingerprint,
                importance=page[-1][0].importance or 0.0,
                updated_at=page[-1][0].updated_at,
                chunk=page[-1][1],
            )
            if len(candidates) > limit and page
            else None
        )
        attempt_id = f"kbackfill_{uuid4().hex}"
        indexed, failed = await self._index_chunks_with_readiness(
            chunks,
            attempt_id=attempt_id,
            operation_prefix=f"kidx:{attempt_id}",
            access_scope=scope,
            refresh_existing=refresh_existing,
        )
        return KnowledgeEmbeddingBackfillResult(
            scanned_records=len(chunks),
            indexed_records=indexed,
            failed_records=failed,
            skipped_records=len(chunks) - indexed - failed,
            limit=limit,
            refresh_existing=refresh_existing,
            next_cursor=next_cursor,
        )

    async def store_embedding_projections(
        self,
        projections: list[KnowledgeEmbeddingProjection],
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeEmbeddingProjectionWriteResult:
        """Persist vectors only while their exact authorized attempt is pending."""

        scope = self._operation_access_scope(access_scope)
        copied = _copy_knowledge_embedding_projections(projections)
        accepted: list[tuple[str, KnowledgeEmbeddingProjection, str]] = []
        writes: list[tuple[str, KnowledgeEmbeddingProjection, str]] = []
        for projection in copied:
            identity = projection.identity
            if not self._embedding_identity_matches_configuration(identity):
                raise ValueError("Embedding projection identity does not match this store.")
            identity_sha256 = _knowledge_embedding_identity_sha256(identity)
            readiness = self._index_readiness_by_identity.get(identity_sha256)
            if (
                readiness is None
                or readiness.sequence != projection.readiness_sequence
                or readiness.state is not KnowledgeIndexState.PENDING
                or readiness.attempt_id != projection.attempt_id
                or not self._index_identity_is_accessible(scope, identity)
                or not self._embedding_identity_is_current(identity)
            ):
                continue
            vector_sha256 = _knowledge_embedding_vector_sha256(projection.vector)
            stored = self._chunk_embeddings.get(identity_sha256)
            if (
                stored is not None
                and stored["readiness_sequence"] == projection.readiness_sequence
                and stored["attempt_id"] == projection.attempt_id
            ):
                if stored["vector_sha256"] != vector_sha256:
                    raise KnowledgeEmbeddingProjectionConflict("attempt_vector_conflict")
            else:
                writes.append((identity_sha256, projection, vector_sha256))
            accepted.append((identity_sha256, projection, vector_sha256))
        for identity_sha256, projection, vector_sha256 in writes:
            self._chunk_embeddings[identity_sha256] = {
                "identity": copy_knowledge_embedding_identity(projection.identity),
                "vector": list(projection.vector),
                "vector_sha256": vector_sha256,
                "readiness_sequence": projection.readiness_sequence,
                "attempt_id": projection.attempt_id,
            }
        return KnowledgeEmbeddingProjectionWriteResult(
            submitted_records=len(copied),
            stored_identities=[projection.identity for _, projection, _ in accepted],
        )

    async def search(
        self,
        query: KnowledgeQuery,
        *,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeSearchResult:
        scope = self._operation_access_scope(access_scope)
        knowledge_query = copy_knowledge_query(query)
        if knowledge_query.mode is KnowledgeSearchMode.KEYWORD:
            return await super().search(knowledge_query, access_scope=scope)
        if knowledge_query.mode not in {
            KnowledgeSearchMode.AUTO,
            KnowledgeSearchMode.SEMANTIC,
            KnowledgeSearchMode.HYBRID,
        }:
            raise ValueError(
                "InMemoryEmbeddingKnowledgeStore supports auto, keyword, semantic, and "
                "hybrid search modes."
            )
        terms = _knowledge_query_terms(knowledge_query)
        candidates: list[tuple[KnowledgeEntry, list[KnowledgeChunk]]] = []
        for entry_id in self._entries:
            entry = self._current_entry(entry_id)
            if entry is None:  # pragma: no cover - internal invariant
                continue
            chunks = self._chunks.get((entry.id, entry.revision), [])
            if not _knowledge_scope_allows_entry(scope, entry):
                continue
            if not _entry_matches_query(entry, knowledge_query):
                continue
            if _entry_matches_none_terms(entry, chunks, terms):
                continue
            candidates.append(
                (
                    copy_knowledge_entry(entry),
                    [copy_knowledge_chunk(chunk) for chunk in chunks],
                )
            )
        candidate_chunks = [chunk for _, chunks in candidates for chunk in chunks]
        candidate_embeddings, coverage = self._ready_embeddings_and_coverage(
            candidate_chunks,
            access_scope=scope,
        )
        if not candidates:
            return KnowledgeSearchResult(
                query=knowledge_query,
                hits=[],
                truncated=False,
                limit=knowledge_query.limit,
                max_bytes=knowledge_query.max_bytes,
                total_hits_known=0,
                index_coverage=[coverage],
            )
        semantic_query_text = _semantic_query_text(knowledge_query)
        query_vector = (
            await self._embed_query(knowledge_query, semantic_query_text)
            if candidate_embeddings
            else None
        )
        semantic_min_score = (
            self.semantic_min_score
            if knowledge_query.min_score is None
            else knowledge_query.min_score
        )
        scored: list[
            tuple[float, KnowledgeEntry, KnowledgeChunk | None, str, str, float | None, bool]
        ] = []
        for entry, chunks in candidates:
            semantic_score, chunk = (
                (None, None)
                if query_vector is None
                else self._best_semantic_score(
                    chunks,
                    candidate_embeddings,
                    query_vector,
                )
            )
            semantic_matched = False
            score = 0.0
            semantic_reason = "semantic projection not ready"
            reason = semantic_reason
            preview_text = entry.text
            score_normalized: float | None = None
            if semantic_score is not None:
                normalized_semantic = _normalize_cosine_similarity(semantic_score)
                semantic_matched = normalized_semantic >= semantic_min_score
                score = normalized_semantic if semantic_matched else 0.0
                semantic_reason = (
                    "semantic chunk match" if chunk is not None else "semantic entry match"
                )
                reason = semantic_reason
                preview_text = chunk.text if chunk is not None else entry.text
                score_normalized = normalized_semantic if semantic_matched else None
            if knowledge_query.mode in {KnowledgeSearchMode.AUTO, KnowledgeSearchMode.HYBRID}:
                keyword_score, keyword_chunk, keyword_reason, keyword_preview = _score_entry(
                    entry,
                    chunks,
                    knowledge_query,
                )
                if keyword_score > 0:
                    keyword_boost = min(keyword_score, 10.0) / 10.0
                    score += self.hybrid_keyword_weight * keyword_boost
                    if keyword_chunk is not None:
                        chunk = keyword_chunk
                    reason = (
                        f"hybrid {semantic_reason}; {keyword_reason}"
                        if semantic_matched
                        else f"hybrid keyword match; {keyword_reason}"
                    )
                    preview_text = keyword_preview
            elif not semantic_matched:
                continue
            if score <= 0:
                continue
            scored.append((score, entry, chunk, reason, preview_text, score_normalized, True))
        scored.sort(
            key=lambda item: (
                -item[0],
                -(item[1].importance or 0.0),
                -item[1].updated_at.timestamp(),
                item[1].id,
            )
        )
        score_kind = (
            "inmemory_semantic"
            if knowledge_query.mode is KnowledgeSearchMode.SEMANTIC
            else "inmemory_hybrid"
        )
        return _search_result_from_scored_embeddings(
            scored,
            knowledge_query,
            score_kind=score_kind,
            index_coverage=[coverage],
        )

    async def _index_chunks_with_readiness(
        self,
        chunks: list[KnowledgeChunk],
        *,
        attempt_id: str,
        operation_prefix: str,
        access_scope: KnowledgeAccessScope,
        refresh_existing: bool = False,
    ) -> tuple[int, int]:
        """Index exact current identities and fence every visible vector with readiness."""

        pending: list[
            tuple[KnowledgeChunk, KnowledgeEmbeddingIdentity, KnowledgeIndexReadiness]
        ] = []
        indexed = 0
        for chunk in chunks:
            identity = knowledge_chunk_embedding_identity(
                chunk,
                embedding_model=self.embedding_model,
                dimensions=self.embedding_dimensions,
            )
            identity_sha256 = _knowledge_embedding_identity_sha256(identity)
            current = await self.load_index_readiness(
                identity,
                access_scope=access_scope,
            )
            stored = self._chunk_embeddings.get(identity_sha256)
            if (
                not refresh_existing
                and current is not None
                and current.state is KnowledgeIndexState.READY
                and stored is not None
                and stored["identity"] == identity
            ):
                continue
            if (
                current is not None
                and current.state is KnowledgeIndexState.PENDING
                and current.attempt_id == attempt_id
            ):
                readiness = current
            else:
                readiness = await self.publish_index_readiness(
                    KnowledgeIndexReadinessUpdate(
                        identity=identity,
                        state=KnowledgeIndexState.PENDING,
                        attempt_id=attempt_id,
                    ),
                    expected_sequence=None if current is None else current.sequence,
                    operation_id=f"{operation_prefix}:{identity_sha256}:pending",
                    access_scope=access_scope,
                )
            if not refresh_existing and stored is not None and stored["identity"] == identity:
                if not self._chunk_is_current(chunk):
                    continue
                await self.publish_index_readiness(
                    KnowledgeIndexReadinessUpdate(
                        identity=identity,
                        state=KnowledgeIndexState.READY,
                        attempt_id=readiness.attempt_id,
                    ),
                    expected_sequence=readiness.sequence,
                    operation_id=f"{operation_prefix}:{identity_sha256}:ready",
                    access_scope=access_scope,
                )
                indexed += 1
                continue
            pending.append((chunk, identity, readiness))
        if not pending:
            return indexed, 0

        try:
            result = copy_text_embedding_result(
                await self.embedding_provider.embed_texts(
                    TextEmbeddingRequest(
                        model=self.embedding_model,
                        texts=[chunk.text for chunk, _, _ in pending],
                        dimensions=self.embedding_dimensions,
                    )
                )
            )
            if result.model != self.embedding_model:
                raise ValueError("Embedding provider returned an unexpected model identity.")
            if len(result.embeddings) != len(pending):
                raise ValueError("Embedding provider returned a different number of embeddings.")
            by_index = {embedding.index: embedding for embedding in result.embeddings}
            if len(by_index) != len(result.embeddings):
                raise ValueError("Embedding provider returned duplicate indexes.")
            for index in range(len(pending)):
                embedding = by_index.get(index)
                if embedding is None:
                    raise ValueError("Embedding provider did not return every requested index.")
                self._validate_embedding_dimension(embedding.vector)
        except Exception:
            failed = 0
            for chunk, identity, readiness in pending:
                if not self._chunk_is_current(chunk):
                    continue
                identity_sha256 = _knowledge_embedding_identity_sha256(identity)
                try:
                    await self.publish_index_readiness(
                        KnowledgeIndexReadinessUpdate(
                            identity=identity,
                            state=KnowledgeIndexState.FAILED,
                            attempt_id=readiness.attempt_id,
                            failure_code="embedding_provider_error",
                        ),
                        expected_sequence=readiness.sequence,
                        operation_id=f"{operation_prefix}:{identity_sha256}:failed",
                        access_scope=access_scope,
                    )
                except KnowledgeIndexReadinessConflict as conflict:
                    if conflict.reason not in {"stale_identity", "stale_sequence"}:
                        raise
                else:
                    failed += 1
            return indexed, failed

        projection_result = await self.store_embedding_projections(
            [
                KnowledgeEmbeddingProjection(
                    identity=identity,
                    readiness_sequence=readiness.sequence,
                    attempt_id=readiness.attempt_id,
                    vector=by_index[index].vector,
                )
                for index, (_, identity, readiness) in enumerate(pending)
            ],
            access_scope=access_scope,
        )
        stored_identity_sha256s = {
            _knowledge_embedding_identity_sha256(identity)
            for identity in projection_result.stored_identities
        }
        for _, identity, readiness in pending:
            identity_sha256 = _knowledge_embedding_identity_sha256(identity)
            if identity_sha256 not in stored_identity_sha256s:
                continue
            try:
                await self.publish_index_readiness(
                    KnowledgeIndexReadinessUpdate(
                        identity=identity,
                        state=KnowledgeIndexState.READY,
                        attempt_id=readiness.attempt_id,
                    ),
                    expected_sequence=readiness.sequence,
                    operation_id=f"{operation_prefix}:{identity_sha256}:ready",
                    access_scope=access_scope,
                )
            except KnowledgeIndexReadinessConflict as conflict:
                if conflict.reason not in {"stale_identity", "stale_sequence"}:
                    raise
                continue
            indexed += 1
        return indexed, 0

    def _ready_embeddings_and_coverage(
        self,
        chunks: list[KnowledgeChunk],
        *,
        access_scope: KnowledgeAccessScope,
    ) -> tuple[dict[str, list[float]], KnowledgeIndexCoverage]:
        embeddings: dict[str, list[float]] = {}
        ready = 0
        failed = 0
        eligible_identity_sha256s: set[str] = set()
        for chunk in chunks:
            identity = knowledge_chunk_embedding_identity(
                chunk,
                embedding_model=self.embedding_model,
                dimensions=self.embedding_dimensions,
            )
            identity_sha256 = _knowledge_embedding_identity_sha256(identity)
            eligible_identity_sha256s.add(identity_sha256)
            readiness = self._index_readiness_by_identity.get(identity_sha256)
            stored = self._chunk_embeddings.get(identity_sha256)
            if (
                readiness is not None
                and readiness.state is KnowledgeIndexState.READY
                and stored is not None
                and stored["identity"] == identity
            ):
                embeddings[chunk.id] = list(stored["vector"])
                ready += 1
            elif readiness is not None and readiness.state is KnowledgeIndexState.FAILED:
                failed += 1
        eligible = len(chunks)
        high_water = max(
            (
                item.sequence
                for item in self._index_readiness
                if _knowledge_embedding_identity_sha256(item.identity) in eligible_identity_sha256s
                and self._index_identity_is_accessible(access_scope, item.identity)
            ),
            default=0,
        )
        pending = eligible - ready - failed
        return embeddings, KnowledgeIndexCoverage(
            projection_type=KNOWLEDGE_CHUNK_TEXT_PROJECTION,
            embedding_model=self.embedding_model,
            dimensions=self.embedding_dimensions,
            preprocessing_version=KNOWLEDGE_CHUNK_TEXT_PREPROCESSING_VERSION,
            generator=KNOWLEDGE_CHUNK_TEXT_GENERATOR,
            generator_version=KNOWLEDGE_CHUNK_TEXT_GENERATOR_VERSION,
            index_representation_version=KNOWLEDGE_VECTOR_INDEX_REPRESENTATION_VERSION,
            eligible_records=eligible,
            ready_records=ready,
            pending_records=pending,
            failed_records=failed,
            high_water_sequence=high_water,
            complete=ready == eligible and pending == 0 and failed == 0,
        )

    async def _embed_query(self, query: KnowledgeQuery, text: str) -> list[float]:
        result = copy_text_embedding_result(
            await self.embedding_provider.embed_texts(
                TextEmbeddingRequest(
                    model=self.embedding_model,
                    texts=[text],
                    dimensions=self.embedding_dimensions,
                )
            )
        )
        if result.model != self.embedding_model:
            raise ValueError("Embedding provider returned an unexpected model identity.")
        if len(result.embeddings) != 1:
            raise ValueError("Embedding provider returned an unexpected query result count.")
        embedding = next((item for item in result.embeddings if item.index == 0), None)
        if embedding is None:
            raise ValueError("Embedding provider did not return query embedding index 0.")
        self._validate_embedding_dimension(embedding.vector)
        return list(embedding.vector)

    def _validate_embedding_dimension(self, vector: list[float]) -> None:
        if len(vector) != self.embedding_dimensions:
            raise ValueError("Embedding provider returned a vector with unexpected dimension.")

    def _best_semantic_score(
        self,
        chunks: list[KnowledgeChunk],
        embeddings: dict[str, list[float]],
        query_vector: list[float],
    ) -> tuple[float | None, KnowledgeChunk | None]:
        best_score: float | None = None
        best_chunk: KnowledgeChunk | None = None
        for chunk in chunks:
            vector = embeddings.get(chunk.id)
            if vector is None:
                continue
            score = _cosine_similarity(query_vector, vector)
            if best_score is None or score > best_score:
                best_score = score
                best_chunk = chunk
        return best_score, best_chunk

    def _drop_entry_embeddings(self, entry_id: str, *, limit: int) -> tuple[int, bool]:
        stale_ids = sorted(
            identity_sha256
            for identity_sha256, embedding in self._chunk_embeddings.items()
            if embedding["identity"].entry_id == entry_id
        )
        selected = stale_ids[:limit]
        for identity_sha256 in selected:
            self._chunk_embeddings.pop(identity_sha256, None)
        return len(selected), len(stale_ids) > limit

    def _drop_stale_entry_embeddings(
        self,
        entry_id: str,
        *,
        limit: int,
    ) -> tuple[int, bool]:
        current = self._current_entry(entry_id)
        current_identity_sha256 = (
            set()
            if current is None
            else {
                _knowledge_embedding_identity_sha256(
                    knowledge_chunk_embedding_identity(
                        chunk,
                        embedding_model=self.embedding_model,
                        dimensions=self.embedding_dimensions,
                    )
                )
                for chunk in self._chunks.get((entry_id, current.revision), [])
            }
        )
        stale_ids = sorted(
            identity_sha256
            for identity_sha256, embedding in self._chunk_embeddings.items()
            if embedding["identity"].entry_id == entry_id
            and identity_sha256 not in current_identity_sha256
        )
        selected = stale_ids[:limit]
        for identity_sha256 in selected:
            self._chunk_embeddings.pop(identity_sha256, None)
        return len(selected), len(stale_ids) > limit

    def _embedding_identity_matches_configuration(
        self,
        identity: KnowledgeEmbeddingIdentity,
    ) -> bool:
        return (
            identity.chunk_id is not None
            and identity.projection_type == KNOWLEDGE_CHUNK_TEXT_PROJECTION
            and identity.embedding_model == self.embedding_model
            and identity.dimensions == self.embedding_dimensions
            and identity.preprocessing_version == KNOWLEDGE_CHUNK_TEXT_PREPROCESSING_VERSION
            and identity.generator == KNOWLEDGE_CHUNK_TEXT_GENERATOR
            and identity.generator_version == KNOWLEDGE_CHUNK_TEXT_GENERATOR_VERSION
            and identity.index_representation_version
            == KNOWLEDGE_VECTOR_INDEX_REPRESENTATION_VERSION
        )

    def _embedding_identity_is_current(self, identity: KnowledgeEmbeddingIdentity) -> bool:
        current = self._current_entry(identity.entry_id)
        if current is None or current.revision != identity.entry_revision:
            return False
        chunk = next(
            (
                item
                for item in self._chunks.get((identity.entry_id, identity.entry_revision), [])
                if item.id == identity.chunk_id
            ),
            None,
        )
        if chunk is None:
            return False
        return (
            knowledge_chunk_embedding_identity(
                chunk,
                embedding_model=self.embedding_model,
                dimensions=self.embedding_dimensions,
            )
            == identity
        )

    def _chunk_is_current(self, chunk: KnowledgeChunk) -> bool:
        current = self._current_entry(chunk.entry_id)
        if current is None or current.revision != chunk.entry_revision:
            return False
        return any(
            stored == chunk
            for stored in self._chunks.get((chunk.entry_id, chunk.entry_revision), [])
        )


def copy_knowledge_access_scope(scope: KnowledgeAccessScope) -> KnowledgeAccessScope:
    if type(scope) is not KnowledgeAccessScope:
        raise TypeError("access_scope must be a KnowledgeAccessScope.")
    return KnowledgeAccessScope(
        allowed_namespaces=list(scope.allowed_namespaces),
        allow_all_namespaces=scope.allow_all_namespaces,
        required_labels=copy_label_map(scope.required_labels, "required_labels"),
        allowed_visibilities=list(scope.allowed_visibilities),
        allowed_source_types=(
            None if scope.allowed_source_types is None else list(scope.allowed_source_types)
        ),
        allowed_source_ids=(
            None if scope.allowed_source_ids is None else list(scope.allowed_source_ids)
        ),
        allowed_statuses=list(scope.allowed_statuses),
        include_expired=scope.include_expired,
    )


def _knowledge_access_snapshot(entry: KnowledgeEntry) -> _KnowledgeAccessSnapshot:
    return _KnowledgeAccessSnapshot(
        namespace=entry.namespace,
        labels=entry.labels,
        visibility=entry.visibility,
        source_type=entry.source_type,
        source_id=entry.source_id,
        status=entry.status,
        expires_at=entry.expires_at,
    )


def _knowledge_access_snapshot_json(snapshot: _KnowledgeAccessSnapshot) -> str:
    if type(snapshot) is not _KnowledgeAccessSnapshot:
        raise TypeError("snapshot must be a _KnowledgeAccessSnapshot.")
    return canonical_durable_json_bytes(
        snapshot.model_dump(mode="json"),
        "knowledge access snapshot",
    ).decode("utf-8")


def _knowledge_access_scope_sha256(scope: KnowledgeAccessScope) -> str:
    scope = copy_knowledge_access_scope(scope)
    return sha256(
        canonical_durable_json_bytes(
            scope.model_dump(mode="json"),
            "knowledge change access scope",
        )
    ).hexdigest()


def _knowledge_change_claim_sha256(claim: KnowledgeChangeClaim) -> str:
    claim = copy_knowledge_change_claim(claim)
    return sha256(
        canonical_durable_json_bytes(
            claim.model_dump(mode="json"),
            "knowledge change claim",
        )
    ).hexdigest()


def _parse_knowledge_access_snapshot_json(value: str) -> _KnowledgeAccessSnapshot:
    if type(value) is not str:
        raise TypeError("Knowledge access snapshot must be JSON text.")
    return _KnowledgeAccessSnapshot.model_validate_json(value)


def _knowledge_scope_allows_snapshot(
    scope: KnowledgeAccessScope,
    snapshot: _KnowledgeAccessSnapshot,
    *,
    now: datetime | None = None,
) -> bool:
    if not _knowledge_scope_allows_snapshot_dimensions(scope, snapshot):
        return False
    cutoff = datetime.now(UTC) if now is None else now
    return scope.include_expired or snapshot.expires_at is None or snapshot.expires_at > cutoff


def _knowledge_scope_allows_snapshot_dimensions(
    scope: KnowledgeAccessScope,
    snapshot: _KnowledgeAccessSnapshot,
) -> bool:
    if not scope.allow_all_namespaces and snapshot.namespace not in scope.allowed_namespaces:
        return False
    for key, value in scope.required_labels.items():
        if snapshot.labels.get(key) != value:
            return False
    if snapshot.visibility not in scope.allowed_visibilities:
        return False
    if (
        scope.allowed_source_types is not None
        and snapshot.source_type not in scope.allowed_source_types
    ):
        return False
    if scope.allowed_source_ids is not None and snapshot.source_id not in scope.allowed_source_ids:
        return False
    return snapshot.status in scope.allowed_statuses


def _knowledge_scope_allows_change_audience(
    scope: KnowledgeAccessScope,
    audience: _KnowledgeChangeAudience,
) -> bool:
    return (
        scope.include_expired or not audience.requires_include_expired
    ) and _knowledge_scope_allows_snapshot_dimensions(scope, audience.snapshot)


def _knowledge_change_audiences(
    change: KnowledgeChange,
    *,
    before_entry: KnowledgeEntry | None,
    after_entry: KnowledgeEntry | None,
    before_requires_include_expired: bool | None = None,
) -> tuple[_KnowledgeChangeAudience, ...]:
    if before_entry is None and after_entry is None:
        raise ValueError("A knowledge change requires a before or after entry.")
    if before_entry is not None and before_entry.id != change.entry_id:
        raise ValueError("Knowledge change before-entry identity does not match the change.")
    if after_entry is not None and (
        after_entry.id != change.entry_id or after_entry.revision != change.entry_revision
    ):
        raise ValueError("Knowledge change after-entry identity does not match the change.")
    if after_entry is None and (
        before_entry is None or before_entry.revision != change.entry_revision
    ):
        raise ValueError("Knowledge removal change revision does not match its before entry.")

    audiences: list[_KnowledgeChangeAudience] = []
    for kind, entry in (("after", after_entry), ("before", before_entry)):
        if entry is None:
            continue
        snapshot = _knowledge_access_snapshot(entry)
        requires_include_expired = (
            snapshot.expires_at is not None and snapshot.expires_at <= change.committed_at
        )
        if kind == "before" and before_requires_include_expired is not None:
            # Preserve the expiration audience captured when this exact revision
            # was published. A slow consumer can then receive its removal signal,
            # while an entry already expired at publication never widens access.
            requires_include_expired = before_requires_include_expired
        if any(
            existing.snapshot == snapshot
            and existing.requires_include_expired == requires_include_expired
            for existing in audiences
        ):
            continue
        audiences.append(
            _KnowledgeChangeAudience(
                kind=kind,
                snapshot=snapshot,
                requires_include_expired=requires_include_expired,
            )
        )
    return tuple(audiences)


def _knowledge_scope_allows_entry(
    scope: KnowledgeAccessScope,
    entry: KnowledgeEntry,
    *,
    now: datetime | None = None,
) -> bool:
    return _knowledge_scope_allows_snapshot(
        scope,
        _knowledge_access_snapshot(entry),
        now=now,
    )


def _require_knowledge_entry_access(
    scope: KnowledgeAccessScope,
    entry: KnowledgeEntry,
    *,
    operation: str,
) -> None:
    if not _knowledge_scope_allows_entry(scope, entry):
        raise KnowledgeAccessDenied(operation)


def _require_knowledge_successor_access(
    scope: KnowledgeAccessScope,
    entry: KnowledgeEntry,
    *,
    operation: str,
) -> None:
    """Authorize a successor without coupling retirement to audit visibility.

    A principal that can mutate the current revision may retire it without also
    receiving read access to archived or deleted material. Every other scope
    dimension remains enforced, and promotion/reactivation still requires the
    destination status to be present in the supplied scope.
    """

    if (
        entry.status in _KNOWLEDGE_RETIREMENT_STATUSES
        and entry.status not in scope.allowed_statuses
    ):
        retirement_scope = scope.model_copy(
            update={
                "allowed_statuses": sorted(
                    {*scope.allowed_statuses, entry.status},
                    key=str,
                )
            }
        )
        _require_knowledge_entry_access(retirement_scope, entry, operation=operation)
        return
    _require_knowledge_entry_access(scope, entry, operation=operation)


def copy_knowledge_entry(entry: KnowledgeEntry) -> KnowledgeEntry:
    if type(entry) is not KnowledgeEntry:
        raise TypeError("KnowledgeEntry instances must not be subclasses.")
    return KnowledgeEntry(
        id=entry.id,
        revision=entry.revision,
        text=entry.text,
        namespace=entry.namespace,
        labels=copy_label_map(entry.labels, "labels"),
        kind=entry.kind,
        visibility=entry.visibility,
        status=entry.status,
        created_by_type=entry.created_by_type,
        created_by=entry.created_by,
        created_at=entry.created_at,
        updated_at=entry.updated_at,
        source_type=entry.source_type,
        source_uri=entry.source_uri,
        source_id=entry.source_id,
        source_hash=entry.source_hash,
        aspects=list(entry.aspects),
        impact_targets=list(entry.impact_targets),
        importance=entry.importance,
        importance_source=entry.importance_source,
        confidence=entry.confidence,
        last_used_at=entry.last_used_at,
        expires_at=entry.expires_at,
        title=entry.title,
        metadata=copy_durable_json_object(entry.metadata, "metadata"),
    )


def copy_knowledge_chunk(chunk: KnowledgeChunk) -> KnowledgeChunk:
    if type(chunk) is not KnowledgeChunk:
        raise TypeError("KnowledgeChunk instances must not be subclasses.")
    return KnowledgeChunk(
        id=chunk.id,
        entry_id=chunk.entry_id,
        entry_revision=chunk.entry_revision,
        text=chunk.text,
        chunk_index=chunk.chunk_index,
        content_hash=chunk.content_hash,
        source_uri=chunk.source_uri,
        metadata=copy_durable_json_object(chunk.metadata, "metadata"),
    )


def copy_knowledge_evidence(evidence: KnowledgeEvidence) -> KnowledgeEvidence:
    if type(evidence) is not KnowledgeEvidence:
        raise TypeError("KnowledgeEvidence instances must not be subclasses.")
    return KnowledgeEvidence(
        id=evidence.id,
        entry_id=evidence.entry_id,
        entry_revision=evidence.entry_revision,
        chunk_id=evidence.chunk_id,
        role=evidence.role,
        source_type=evidence.source_type,
        source_id=evidence.source_id,
        source_uri=evidence.source_uri,
        source_revision=evidence.source_revision,
        source_hash=evidence.source_hash,
        locator=copy_durable_json_object(evidence.locator, "locator"),
        disposition=evidence.disposition,
        created_at=evidence.created_at,
        metadata=copy_durable_json_object(evidence.metadata, "metadata"),
    )


def copy_knowledge_embedding_identity(
    identity: KnowledgeEmbeddingIdentity,
) -> KnowledgeEmbeddingIdentity:
    if type(identity) is not KnowledgeEmbeddingIdentity:
        raise TypeError("KnowledgeEmbeddingIdentity instances must not be subclasses.")
    return KnowledgeEmbeddingIdentity(**identity.model_dump())


def copy_knowledge_embedding_projection(
    projection: KnowledgeEmbeddingProjection,
) -> KnowledgeEmbeddingProjection:
    if type(projection) is not KnowledgeEmbeddingProjection:
        raise TypeError("KnowledgeEmbeddingProjection instances must not be subclasses.")
    return KnowledgeEmbeddingProjection(
        identity=projection.identity,
        readiness_sequence=projection.readiness_sequence,
        attempt_id=projection.attempt_id,
        vector=list(projection.vector),
    )


def _copy_knowledge_embedding_projections(
    projections: list[KnowledgeEmbeddingProjection],
) -> list[KnowledgeEmbeddingProjection]:
    if type(projections) is not list:
        raise TypeError("`projections` must be a list.")
    if len(projections) > MAX_KNOWLEDGE_EMBEDDING_WORK_RECORD_LIMIT:
        raise ValueError(
            "`projections` must contain at most "
            f"{MAX_KNOWLEDGE_EMBEDDING_WORK_RECORD_LIMIT} records."
        )
    copied = [copy_knowledge_embedding_projection(projection) for projection in projections]
    identity_sha256s = {
        _knowledge_embedding_identity_sha256(projection.identity) for projection in copied
    }
    if len(identity_sha256s) != len(copied):
        raise ValueError("`projections` cannot contain duplicate identities.")
    return copied


def copy_knowledge_index_readiness_update(
    update: KnowledgeIndexReadinessUpdate,
) -> KnowledgeIndexReadinessUpdate:
    if type(update) is not KnowledgeIndexReadinessUpdate:
        raise TypeError("KnowledgeIndexReadinessUpdate instances must not be subclasses.")
    return KnowledgeIndexReadinessUpdate(
        identity=copy_knowledge_embedding_identity(update.identity),
        state=update.state,
        attempt_id=update.attempt_id,
        failure_code=update.failure_code,
    )


def copy_knowledge_index_readiness(
    readiness: KnowledgeIndexReadiness,
) -> KnowledgeIndexReadiness:
    if type(readiness) is not KnowledgeIndexReadiness:
        raise TypeError("KnowledgeIndexReadiness instances must not be subclasses.")
    return KnowledgeIndexReadiness(
        sequence=readiness.sequence,
        identity=copy_knowledge_embedding_identity(readiness.identity),
        state=readiness.state,
        attempt_id=readiness.attempt_id,
        failure_code=readiness.failure_code,
        operation_id=readiness.operation_id,
        published_at=readiness.published_at,
    )


def copy_knowledge_index_coverage(
    coverage: KnowledgeIndexCoverage,
) -> KnowledgeIndexCoverage:
    if type(coverage) is not KnowledgeIndexCoverage:
        raise TypeError("KnowledgeIndexCoverage instances must not be subclasses.")
    return KnowledgeIndexCoverage(**coverage.model_dump())


def copy_knowledge_change(change: KnowledgeChange) -> KnowledgeChange:
    if type(change) is not KnowledgeChange:
        raise TypeError("KnowledgeChange instances must not be subclasses.")
    return KnowledgeChange(
        id=change.id,
        sequence=change.sequence,
        kind=change.kind,
        entry_id=change.entry_id,
        entry_revision=change.entry_revision,
        committed_at=change.committed_at,
        operation_id=change.operation_id,
    )


def copy_knowledge_change_claim(claim: KnowledgeChangeClaim) -> KnowledgeChangeClaim:
    if type(claim) is not KnowledgeChangeClaim:
        raise TypeError("KnowledgeChangeClaim instances must not be subclasses.")
    return KnowledgeChangeClaim(
        consumer_id=claim.consumer_id,
        worker_id=claim.worker_id,
        claim_id=claim.claim_id,
        change=copy_knowledge_change(claim.change),
        attempt=claim.attempt,
        claimed_at=claim.claimed_at,
        lease_expires_at=claim.lease_expires_at,
    )


def copy_knowledge_change_consumer_state(
    state: KnowledgeChangeConsumerState,
) -> KnowledgeChangeConsumerState:
    if type(state) is not KnowledgeChangeConsumerState:
        raise TypeError("KnowledgeChangeConsumerState instances must not be subclasses.")
    return KnowledgeChangeConsumerState(**state.model_dump())


def copy_knowledge_publication_receipt(
    receipt: KnowledgePublicationReceipt,
    *,
    replayed: bool | None = None,
) -> KnowledgePublicationReceipt:
    if type(receipt) is not KnowledgePublicationReceipt:
        raise TypeError("KnowledgePublicationReceipt instances must not be subclasses.")
    return KnowledgePublicationReceipt(
        operation_id=receipt.operation_id,
        entry_id=receipt.entry_id,
        entry_revision=receipt.entry_revision,
        expected_revision=receipt.expected_revision,
        request_sha256=receipt.request_sha256,
        entry_created_at=receipt.entry_created_at,
        entry_updated_at=receipt.entry_updated_at,
        committed_at=receipt.committed_at,
        replayed=receipt.replayed if replayed is None else replayed,
    )


def prepare_knowledge_publication(
    entry: KnowledgeEntry,
    chunks: list[KnowledgeChunk],
    *,
    evidence: list[KnowledgeEvidence] | None = None,
    operation_id: str,
    expected_revision: int | None = None,
) -> tuple[
    str,
    KnowledgeEntry,
    list[KnowledgeChunk],
    list[KnowledgeEvidence],
    str,
]:
    """Copy and bind one complete revision-publication authority tuple."""

    clean_operation_id = _knowledge_publication_operation_id(operation_id)
    copied_entry = copy_knowledge_entry(entry)
    _validate_revision_append(copied_entry, expected_revision=expected_revision)
    copied_chunks = _copy_entry_chunks(
        copied_entry.id,
        copied_entry.revision,
        chunks,
    )
    copied_evidence = _copy_entry_evidence(
        copied_entry.id,
        copied_entry.revision,
        evidence or [],
        chunks=copied_chunks,
    )
    request_sha256 = _knowledge_publication_request_sha256(
        copied_entry,
        copied_chunks,
        copied_evidence,
        expected_revision=expected_revision,
    )
    return (
        clean_operation_id,
        copied_entry,
        copied_chunks,
        copied_evidence,
        request_sha256,
    )


def _knowledge_publication_operation_id(operation_id: str) -> str:
    clean = require_clean_nonblank(operation_id, "operation_id")
    if len(clean.encode("utf-8")) > 256:
        raise ValueError("`operation_id` must be at most 256 UTF-8 bytes.")
    return clean


def _validate_knowledge_publication_replay(
    receipt: KnowledgePublicationReceipt,
    *,
    entry: KnowledgeEntry,
    chunks: list[KnowledgeChunk],
    evidence: list[KnowledgeEvidence],
    expected_revision: int | None,
    request_sha256: str,
) -> None:
    receipt = copy_knowledge_publication_receipt(receipt)
    accepted_request_sha256s = {request_sha256}
    if not evidence:
        # Revision 42 receipts bind the same entry-and-chunks authority tuple
        # under the v1 digest contract. Revision 43 preserves those receipts,
        # so an exact empty-evidence retry must remain idempotent after migration.
        # Never permit the weaker digest when the new request carries evidence.
        accepted_request_sha256s.add(
            _knowledge_publication_v1_request_sha256(
                entry,
                chunks,
                expected_revision=expected_revision,
            )
        )
    if (
        receipt.entry_id != entry.id
        or receipt.entry_revision != entry.revision
        or receipt.request_sha256 not in accepted_request_sha256s
        or receipt.entry_created_at != entry.created_at
        or receipt.entry_updated_at != entry.updated_at
    ):
        raise KnowledgePublicationConflict("operation_mismatch")


def _knowledge_publication_request_sha256(
    entry: KnowledgeEntry,
    chunks: list[KnowledgeChunk],
    evidence: list[KnowledgeEvidence],
    *,
    expected_revision: int | None,
) -> str:
    return sha256(
        canonical_durable_json_bytes(
            {
                "contract": "cayu-knowledge-revision-publication-v2",
                "expected_revision": expected_revision,
                "entry": entry.model_dump(mode="json"),
                "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
                "evidence": [item.model_dump(mode="json") for item in evidence],
            },
            "knowledge publication",
        )
    ).hexdigest()


def _knowledge_publication_v1_request_sha256(
    entry: KnowledgeEntry,
    chunks: list[KnowledgeChunk],
    *,
    expected_revision: int | None,
) -> str:
    """Reproduce the revision-42 receipt digest for migration-safe replay."""

    return sha256(
        canonical_durable_json_bytes(
            {
                "contract": "cayu-knowledge-revision-publication-v1",
                "expected_revision": expected_revision,
                "entry": entry.model_dump(mode="json"),
                "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
            },
            "knowledge publication",
        )
    ).hexdigest()


def _validate_revision_append(
    entry: KnowledgeEntry,
    *,
    expected_revision: int | None,
) -> None:
    target_revision = (
        1 if expected_revision is None else _next_knowledge_revision(expected_revision)
    )
    _validate_knowledge_revision(entry.revision, "entry.revision")
    if entry.revision != target_revision:
        raise ValueError(
            f"Knowledge revision must be {target_revision} for expected_revision "
            f"{expected_revision!r}."
        )


def _next_knowledge_revision(expected_revision: int) -> int:
    _validate_knowledge_revision(expected_revision, "expected_revision")
    if expected_revision == MAX_KNOWLEDGE_REVISION:
        raise ValueError(f"Knowledge revision cannot advance beyond {MAX_KNOWLEDGE_REVISION}.")
    return expected_revision + 1


def _validate_revision_successor(
    current: KnowledgeEntry,
    successor: KnowledgeEntry,
) -> None:
    if successor.id != current.id:
        raise ValueError("Knowledge revision must preserve the logical entry id.")
    if successor.namespace != current.namespace:
        raise ValueError("Knowledge revision must preserve the logical namespace.")
    if successor.created_at != current.created_at:
        raise ValueError("Knowledge revision must preserve the logical creation time.")
    if successor.updated_at < current.updated_at:
        raise ValueError("Knowledge revision `updated_at` cannot move backwards.")


def copy_knowledge_query(query: KnowledgeQuery) -> KnowledgeQuery:
    if type(query) is not KnowledgeQuery:
        raise TypeError("KnowledgeQuery instances must not be subclasses.")
    return KnowledgeQuery(
        text=query.text,
        any_terms=list(query.any_terms),
        all_terms=list(query.all_terms),
        none_terms=list(query.none_terms),
        phrases=list(query.phrases),
        namespace=query.namespace,
        labels=copy_label_map(query.labels, "labels"),
        kinds=list(query.kinds) if query.kinds is not None else None,
        statuses=list(query.statuses),
        visibilities=list(query.visibilities) if query.visibilities is not None else None,
        aspects=list(query.aspects),
        impact_targets=list(query.impact_targets),
        source_type=query.source_type,
        source_id=query.source_id,
        mode=query.mode,
        min_score=query.min_score,
        include_expired=query.include_expired,
        limit=query.limit,
        max_bytes=query.max_bytes,
    )


def copy_knowledge_list_query(query: KnowledgeListQuery) -> KnowledgeListQuery:
    if type(query) is not KnowledgeListQuery:
        raise TypeError("KnowledgeListQuery instances must not be subclasses.")
    return KnowledgeListQuery(
        namespace=query.namespace,
        labels=copy_label_map(query.labels, "labels"),
        kinds=list(query.kinds) if query.kinds is not None else None,
        statuses=list(query.statuses),
        visibilities=list(query.visibilities) if query.visibilities is not None else None,
        aspects=list(query.aspects),
        impact_targets=list(query.impact_targets),
        source_type=query.source_type,
        source_id=query.source_id,
        include_expired=query.include_expired,
        group_by=query.group_by,
        limit=query.limit,
        max_bytes=query.max_bytes,
    )


def copy_knowledge_hit(hit: KnowledgeHit) -> KnowledgeHit:
    if type(hit) is not KnowledgeHit:
        raise TypeError("KnowledgeHit instances must not be subclasses.")
    return KnowledgeHit(
        entry=copy_knowledge_entry(hit.entry),
        chunk=copy_knowledge_chunk(hit.chunk) if hit.chunk is not None else None,
        score=hit.score,
        reason=hit.reason,
        rank=hit.rank,
        score_kind=hit.score_kind,
        score_normalized=hit.score_normalized,
        text_preview=hit.text_preview,
        text_preview_complete=hit.text_preview_complete,
    )


def copy_knowledge_list_item(item: KnowledgeListItem) -> KnowledgeListItem:
    if type(item) is not KnowledgeListItem:
        raise TypeError("KnowledgeListItem instances must not be subclasses.")
    return KnowledgeListItem(
        entry=copy_knowledge_entry(item.entry),
        chunk_count=item.chunk_count,
        text_preview=item.text_preview,
        text_preview_complete=item.text_preview_complete,
    )


def copy_knowledge_facet(facet: KnowledgeFacet) -> KnowledgeFacet:
    if type(facet) is not KnowledgeFacet:
        raise TypeError("KnowledgeFacet instances must not be subclasses.")
    return KnowledgeFacet(
        field=facet.field,
        key=facet.key,
        value=facet.value,
        count=facet.count,
    )


def _copy_entry_chunks(
    entry_id: str,
    entry_revision: int,
    chunks: list[KnowledgeChunk],
) -> list[KnowledgeChunk]:
    if type(chunks) is not list:
        raise ValueError("`chunks` must be a list.")
    if not chunks:
        raise ValueError("`chunks` cannot be empty.")
    copied_chunks = [copy_knowledge_chunk(chunk) for chunk in chunks]
    seen_ids: set[str] = set()
    seen_indexes: set[int] = set()
    for chunk in copied_chunks:
        if chunk.entry_id != entry_id:
            raise ValueError("Knowledge chunks must belong to the entry.")
        if chunk.entry_revision != entry_revision:
            raise ValueError("Knowledge chunks must belong to the exact entry revision.")
        if chunk.id in seen_ids:
            raise ValueError("Knowledge chunk ids must be unique within an entry.")
        if chunk.chunk_index in seen_indexes:
            raise ValueError("Knowledge chunk indexes must be unique within an entry.")
        seen_ids.add(chunk.id)
        seen_indexes.add(chunk.chunk_index)
    return sorted(copied_chunks, key=lambda chunk: chunk.chunk_index)


def _copy_entry_evidence(
    entry_id: str,
    entry_revision: int,
    evidence: list[KnowledgeEvidence],
    *,
    chunks: list[KnowledgeChunk],
) -> list[KnowledgeEvidence]:
    if type(evidence) is not list:
        raise ValueError("`evidence` must be a list.")
    copied = [copy_knowledge_evidence(item) for item in evidence]
    chunk_ids = {chunk.id for chunk in chunks}
    seen_ids: set[str] = set()
    for item in copied:
        if item.entry_id != entry_id:
            raise ValueError("Knowledge evidence must belong to the entry.")
        if item.entry_revision != entry_revision:
            raise ValueError("Knowledge evidence must belong to the exact entry revision.")
        if item.chunk_id is not None and item.chunk_id not in chunk_ids:
            raise ValueError("Knowledge evidence chunk must belong to the exact entry revision.")
        if item.id in seen_ids:
            raise ValueError("Knowledge evidence ids must be unique within a revision.")
        seen_ids.add(item.id)
    return sorted(copied, key=lambda item: item.id)


def _copy_evidence_for_revision(
    evidence: list[KnowledgeEvidence],
    *,
    entry: KnowledgeEntry,
    previous_chunks: list[KnowledgeChunk],
    chunks: list[KnowledgeChunk],
) -> list[KnowledgeEvidence]:
    previous_indexes = {chunk.id: chunk.chunk_index for chunk in previous_chunks}
    next_chunks = {chunk.chunk_index: chunk.id for chunk in chunks}
    copied: list[KnowledgeEvidence] = []
    for item in evidence:
        chunk_id: str | None = None
        if item.chunk_id is not None:
            chunk_index = previous_indexes.get(item.chunk_id)
            if chunk_index is None or chunk_index not in next_chunks:
                raise RuntimeError(
                    "Stored knowledge evidence references an unavailable source chunk."
                )
            chunk_id = next_chunks[chunk_index]
        evidence_id = (
            "ke_"
            + sha256(
                canonical_durable_json_bytes(
                    {
                        "contract": "cayu-knowledge-evidence-successor-v1",
                        "source_evidence_id": item.id,
                        "entry_id": entry.id,
                        "entry_revision": entry.revision,
                    },
                    "knowledge evidence successor identity",
                )
            ).hexdigest()
        )
        copied.append(
            KnowledgeEvidence(
                id=evidence_id,
                entry_id=entry.id,
                entry_revision=entry.revision,
                chunk_id=chunk_id,
                role=item.role,
                source_type=item.source_type,
                source_id=item.source_id,
                source_uri=item.source_uri,
                source_revision=item.source_revision,
                source_hash=item.source_hash,
                locator=item.locator,
                disposition=item.disposition,
                created_at=item.created_at,
                metadata=item.metadata,
            )
        )
    return _copy_entry_evidence(
        entry.id,
        entry.revision,
        copied,
        chunks=chunks,
    )


def _copy_chunks_for_revision(
    chunks: list[KnowledgeChunk],
    entry: KnowledgeEntry,
) -> list[KnowledgeChunk]:
    if not chunks:
        return [_default_chunk_for_entry(entry)]
    return [
        KnowledgeChunk(
            id=f"{entry.id}:r{entry.revision}:{chunk.chunk_index}",
            entry_id=entry.id,
            entry_revision=entry.revision,
            text=chunk.text,
            chunk_index=chunk.chunk_index,
            content_hash=chunk.content_hash,
            source_uri=chunk.source_uri,
            metadata=chunk.metadata,
        )
        for chunk in chunks
    ]


def _center_chunk_window(
    chunks: list[KnowledgeChunk],
    *,
    chunk_index: int,
    max_chunks: int,
) -> list[KnowledgeChunk]:
    if len(chunks) <= max_chunks:
        return chunks
    closest = sorted(
        chunks, key=lambda chunk: (abs(chunk.chunk_index - chunk_index), chunk.chunk_index)
    )
    return sorted(closest[:max_chunks], key=lambda chunk: chunk.chunk_index)


def _bounded_chunks(
    chunks: list[KnowledgeChunk],
    *,
    start_index: int,
    end_index: int | None,
    max_chunks: int,
    max_bytes: int,
) -> list[KnowledgeChunk]:
    _validate_positive_int(max_chunks, "max_chunks")
    _validate_positive_int(max_bytes, "max_bytes")
    selected: list[KnowledgeChunk] = []
    remaining = max_bytes
    for chunk in chunks:
        if chunk.chunk_index < start_index:
            continue
        if end_index is not None and chunk.chunk_index > end_index:
            continue
        if len(selected) >= max_chunks or remaining <= 0:
            break
        copied = copy_knowledge_chunk(chunk)
        chunk_bytes = len(copied.text.encode("utf-8"))
        if chunk_bytes > remaining:
            truncated_text = _truncate_text_to_bytes(copied.text, remaining)
            if not truncated_text:
                break
            selected.append(
                KnowledgeChunk(
                    id=copied.id,
                    entry_id=copied.entry_id,
                    entry_revision=copied.entry_revision,
                    text=truncated_text,
                    chunk_index=copied.chunk_index,
                    content_hash=None,
                    source_uri=copied.source_uri,
                    metadata=copied.metadata,
                )
            )
            break
        selected.append(copied)
        remaining -= chunk_bytes
    return selected


def _bounded_knowledge_evidence(
    evidence: list[KnowledgeEvidence],
    *,
    max_records: int,
    max_bytes: int,
) -> list[KnowledgeEvidence]:
    _validate_positive_int(max_records, "max_records")
    _validate_positive_int(max_bytes, "max_bytes")
    selected: list[KnowledgeEvidence] = []
    consumed = 0
    for item in sorted(evidence, key=lambda value: value.id):
        item_size = len(
            canonical_durable_json_bytes(
                item.model_dump(mode="json"),
                "knowledge evidence",
            )
        )
        if len(selected) >= max_records or consumed + item_size > max_bytes:
            break
        selected.append(copy_knowledge_evidence(item))
        consumed += item_size
    return selected


def _entry_matches_query(entry: KnowledgeEntry, query: KnowledgeQuery) -> bool:
    return _entry_matches_metadata(
        entry,
        namespace=query.namespace,
        labels=query.labels,
        kinds=query.kinds,
        statuses=query.statuses,
        visibilities=query.visibilities,
        aspects=query.aspects,
        impact_targets=query.impact_targets,
        source_type=query.source_type,
        source_id=query.source_id,
        include_expired=query.include_expired,
    )


def _entry_matches_list_query(entry: KnowledgeEntry, query: KnowledgeListQuery) -> bool:
    return _entry_matches_metadata(
        entry,
        namespace=query.namespace,
        labels=query.labels,
        kinds=query.kinds,
        statuses=query.statuses,
        visibilities=query.visibilities,
        aspects=query.aspects,
        impact_targets=query.impact_targets,
        source_type=query.source_type,
        source_id=query.source_id,
        include_expired=query.include_expired,
    )


def _entry_matches_metadata(
    entry: KnowledgeEntry,
    *,
    namespace: str | None,
    labels: dict[str, str],
    kinds: list[str] | None,
    statuses: list[KnowledgeStatus],
    visibilities: list[KnowledgeVisibility] | None,
    aspects: list[str],
    impact_targets: list[str],
    source_type: str | None,
    source_id: str | None,
    include_expired: bool,
) -> bool:
    if namespace is not None and entry.namespace != namespace:
        return False
    for key, value in labels.items():
        if entry.labels.get(key) != value:
            return False
    if kinds is not None and entry.kind not in set(kinds):
        return False
    if entry.status not in set(statuses):
        return False
    if visibilities is not None and entry.visibility not in set(visibilities):
        return False
    if source_type is not None and entry.source_type != source_type:
        return False
    if source_id is not None and entry.source_id != source_id:
        return False
    if aspects and not set(aspects).intersection(entry.aspects):
        return False
    if impact_targets and not set(impact_targets).intersection(entry.impact_targets):
        return False
    return not _entry_is_expired(entry, include_expired=include_expired)


def _entry_is_expired(entry: KnowledgeEntry, *, include_expired: bool) -> bool:
    return (
        not include_expired
        and entry.expires_at is not None
        and entry.expires_at <= datetime.now(UTC)
    )


def _score_entry(
    entry: KnowledgeEntry,
    chunks: list[KnowledgeChunk],
    query: KnowledgeQuery,
) -> tuple[float, KnowledgeChunk | None, str, str]:
    terms = _knowledge_query_terms(query)
    if not _query_terms_have_positive_terms(terms):
        return 0.0, None, "empty query", entry.text
    best_score = _score_candidate(entry.text, terms)
    best_chunk: KnowledgeChunk | None = None
    best_reason = "entry text match"
    best_preview_text = entry.text
    if entry.title is not None:
        title_score = _score_candidate(entry.title, terms) * 1.2
        if title_score > best_score:
            best_score = title_score
            best_reason = "title match"
            best_preview_text = entry.title
    for chunk in chunks:
        chunk_search_fields = _entry_chunk_searchable_fields(entry, chunk)
        chunk_score = _score_candidate(
            "\n".join(chunk_search_fields),
            terms,
            phrase_fields=chunk_search_fields,
        )
        if chunk_score > best_score:
            best_score = chunk_score
            best_chunk = chunk
            best_reason = "chunk text match"
            best_preview_text = chunk.text
    return best_score, best_chunk, best_reason, best_preview_text


def _score_candidate(
    text: str,
    terms: _SearchTerms,
    *,
    phrase_fields: list[str] | None = None,
) -> float:
    tokens = _tokenize_search_text(text)
    phrase_token_fields = (
        [tokens]
        if phrase_fields is None
        else [_tokenize_search_text(field) for field in phrase_fields]
    )
    if not _tokens_match_structured_terms(tokens, terms, phrase_token_fields):
        return 0.0
    token_counts = Counter(tokens)
    score = float(sum(token_counts[term] for term in terms["any"]))
    score += float(sum(max(token_counts[term] for term in group) for group in terms["all"]))
    score += float(
        sum(
            2
            for phrase in terms["phrases"]
            if any(_tokens_contain_phrase(field, phrase) for field in phrase_token_fields)
        )
    )
    return score


def _tokens_match_structured_terms(
    tokens: list[str],
    terms: _SearchTerms,
    phrase_token_fields: list[list[str]],
) -> bool:
    token_set = set(tokens)
    if any(term in token_set for term in terms["none"]):
        return False
    if not all(any(term in token_set for term in group) for group in terms["all"]):
        return False
    positives = terms["any"] or terms["phrases"]
    return not positives or (
        any(term in token_set for term in terms["any"])
        or any(
            _tokens_contain_phrase(field, phrase)
            for phrase in terms["phrases"]
            for field in phrase_token_fields
        )
    )


def _tokens_contain_phrase(tokens: list[str], phrase: list[str]) -> bool:
    phrase_length = len(phrase)
    return any(
        tokens[index : index + phrase_length] == phrase
        for index in range(len(tokens) - phrase_length + 1)
    )


def _entry_chunk_searchable_fields(entry: KnowledgeEntry, chunk: KnowledgeChunk) -> list[str]:
    parts: list[str] = []
    if entry.title is not None:
        parts.append(entry.title)
    parts.append(entry.text)
    if chunk.text == entry.text:
        return parts
    parts.append(chunk.text)
    return parts


def _entry_matches_none_terms(
    entry: KnowledgeEntry,
    chunks: list[KnowledgeChunk],
    terms: _SearchTerms,
) -> bool:
    if not terms["none"]:
        return False
    texts = [entry.text]
    if entry.title is not None:
        texts.append(entry.title)
    texts.extend(chunk.text for chunk in chunks)
    tokens = {token for text in texts for token in _tokenize_search_text(text)}
    return any(term in tokens for term in terms["none"])


def _search_result_from_scored_embeddings(
    scored: list[tuple[float, KnowledgeEntry, KnowledgeChunk | None, str, str, float | None, bool]],
    query: KnowledgeQuery,
    *,
    score_kind: str,
    index_coverage: list[KnowledgeIndexCoverage] | None = None,
) -> KnowledgeSearchResult:
    hits: list[KnowledgeHit] = []
    remaining = query.max_bytes
    truncated = False
    for rank, (
        score,
        entry,
        chunk,
        reason,
        preview_text,
        normalized_score,
        source_complete,
    ) in enumerate(
        scored[: query.limit],
        start=1,
    ):
        if remaining <= 0:
            truncated = True
            break
        source_bytes = len(preview_text.encode("utf-8"))
        preview = _truncate_text_to_bytes(preview_text, remaining)
        if not preview:
            truncated = True
            break
        preview_complete = source_complete and len(preview.encode("utf-8")) == source_bytes
        if not preview_complete:
            truncated = True
        remaining -= len(preview.encode("utf-8"))
        hits.append(
            KnowledgeHit(
                entry=entry,
                chunk=chunk,
                score=score,
                score_kind=score_kind,
                score_normalized=normalized_score,
                rank=rank,
                reason=reason,
                text_preview=preview,
                text_preview_complete=preview_complete,
            )
        )
    return KnowledgeSearchResult(
        query=query,
        hits=hits,
        truncated=truncated or len(hits) < len(scored),
        limit=query.limit,
        max_bytes=query.max_bytes,
        total_hits_known=len(scored),
        index_coverage=(
            []
            if index_coverage is None
            else [copy_knowledge_index_coverage(item) for item in index_coverage]
        ),
    )


def _semantic_query_text(query: KnowledgeQuery) -> str:
    parts: list[str] = []
    if query.text is not None:
        parts.append(query.text)
    parts.extend(query.any_terms)
    parts.extend(query.all_terms)
    parts.extend(query.phrases)
    return require_nonblank(" ".join(parts), "semantic query text")


def _knowledge_chunk_content_hash(chunk: KnowledgeChunk) -> str:
    return f"sha256:{sha256(chunk.text.encode('utf-8')).hexdigest()}"


def knowledge_chunk_embedding_identity(
    chunk: KnowledgeChunk,
    *,
    embedding_model: str,
    dimensions: int,
) -> KnowledgeEmbeddingIdentity:
    """Build the built-in canonical chunk-text embedding identity."""

    chunk = copy_knowledge_chunk(chunk)
    return KnowledgeEmbeddingIdentity(
        entry_id=chunk.entry_id,
        entry_revision=chunk.entry_revision,
        chunk_id=chunk.id,
        projection_type=KNOWLEDGE_CHUNK_TEXT_PROJECTION,
        projection_content_hash=_knowledge_chunk_content_hash(chunk),
        embedding_model=embedding_model,
        dimensions=dimensions,
        preprocessing_version=KNOWLEDGE_CHUNK_TEXT_PREPROCESSING_VERSION,
        generator=KNOWLEDGE_CHUNK_TEXT_GENERATOR,
        generator_version=KNOWLEDGE_CHUNK_TEXT_GENERATOR_VERSION,
        index_representation_version=KNOWLEDGE_VECTOR_INDEX_REPRESENTATION_VERSION,
    )


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding vectors must have the same dimension.")
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    dot_product = sum(
        left_item * right_item for left_item, right_item in zip(left, right, strict=True)
    )
    return dot_product / (left_norm * right_norm)


def _normalize_cosine_similarity(value: float) -> float:
    return max(0.0, min(1.0, (value + 1.0) / 2.0))


def _knowledge_query_terms(query: KnowledgeQuery) -> _SearchTerms:
    text_terms = _expand_search_tokens(_tokenize_search_text(query.text or ""))
    return {
        "any": _dedupe_strings(
            [
                *text_terms,
                *(
                    token
                    for term in query.any_terms
                    for group in _normalize_search_term_groups(term)
                    for token in group
                ),
            ]
        ),
        "all": _dedupe_search_term_groups(
            [group for value in query.all_terms for group in _normalize_search_term_groups(value)]
        ),
        "none": _dedupe_strings(
            [
                token
                for value in query.none_terms
                for group in _normalize_search_term_groups(value)
                for token in group
            ]
        ),
        "phrases": _dedupe_search_term_groups(
            [_normalize_search_phrase(phrase) for phrase in query.phrases]
        ),
    }


def _query_terms_have_positive_terms(terms: _SearchTerms) -> bool:
    return bool(terms["any"] or terms["all"] or terms["phrases"])


def _normalize_search_term_groups(value: str) -> list[list[str]]:
    terms = _tokenize_search_text(value)
    if not terms:
        raise ValueError("Structured knowledge search terms must contain at least one token.")
    return [_search_token_variants(term) for term in terms]


def _dedupe_search_term_groups(groups: list[list[str]]) -> list[list[str]]:
    result: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for group in groups:
        key = tuple(group)
        if key not in seen:
            result.append(group)
            seen.add(key)
    return result


def _normalize_search_phrase(value: str) -> list[str]:
    tokens = _tokenize_search_text(require_nonblank(value, "phrase"))
    if not tokens:
        raise ValueError("Structured knowledge search phrases must contain at least one token.")
    return tokens


def _knowledge_facets(
    entries: list[KnowledgeEntry],
    group_by: KnowledgeListGroup | None,
    *,
    limit: int,
) -> tuple[list[KnowledgeFacet], bool]:
    if group_by is None:
        return [], False
    counter: Counter[tuple[str | None, str]] = Counter()
    for entry in entries:
        if group_by is KnowledgeListGroup.KIND:
            counter[(None, entry.kind)] += 1
        elif group_by is KnowledgeListGroup.LABEL:
            for key, value in entry.labels.items():
                counter[(key, value)] += 1
        elif group_by is KnowledgeListGroup.ASPECT:
            for aspect in entry.aspects:
                counter[(None, aspect)] += 1
        elif group_by is KnowledgeListGroup.IMPACT_TARGET:
            for target in entry.impact_targets:
                counter[(None, target)] += 1
        elif group_by is KnowledgeListGroup.VISIBILITY:
            counter[(None, entry.visibility.value)] += 1
        elif group_by is KnowledgeListGroup.SOURCE_TYPE and entry.source_type is not None:
            counter[(None, entry.source_type)] += 1
        elif group_by is KnowledgeListGroup.NAMESPACE:
            counter[(None, entry.namespace)] += 1
    facets = [
        KnowledgeFacet(field=group_by, key=key, value=value, count=count)
        for (key, value), count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]
    return facets[:limit], len(facets) > limit


def _tokenize_search_text(text: str) -> list[str]:
    return _SEARCH_TOKEN_RE.findall(text.casefold())


def _expand_search_tokens(tokens: list[str]) -> list[str]:
    return [variant for token in tokens for variant in _search_token_variants(token)]


def _search_token_variants(token: str) -> list[str]:
    variants = [token]
    if len(token) < 3 or not token.isalpha():
        return variants
    if token.endswith("ies") and len(token) > 4:
        variants.append(token[:-3] + "y")
    elif token.endswith("s") and not token.endswith(("ss", "us", "is")):
        variants.append(token[:-1])
    else:
        variants.append(_plural_search_token(token))
    return _dedupe_strings(variants)


def _plural_search_token(token: str) -> str:
    if token.endswith("y") and len(token) > 1 and token[-2] not in "aeiou":
        return token[:-1] + "ies"
    return token + "s"


def _default_chunk_for_entry(entry: KnowledgeEntry) -> KnowledgeChunk:
    return KnowledgeChunk(
        id=f"{entry.id}:r{entry.revision}:0",
        entry_id=entry.id,
        entry_revision=entry.revision,
        text=entry.text,
        chunk_index=0,
        content_hash=sha256(entry.text.encode("utf-8")).hexdigest(),
        source_uri=entry.source_uri,
    )


def _next_updated_at(entry: KnowledgeEntry) -> datetime:
    return max(datetime.now(UTC), entry.created_at, entry.updated_at)


def _has_only_default_chunk(entry: KnowledgeEntry, chunks: list[KnowledgeChunk]) -> bool:
    if len(chunks) != 1:
        return False
    default_chunk = _default_chunk_for_entry(entry)
    chunk = chunks[0]
    return (
        chunk.id == default_chunk.id
        and chunk.entry_id == default_chunk.entry_id
        and chunk.entry_revision == default_chunk.entry_revision
        and chunk.text == default_chunk.text
        and chunk.chunk_index == default_chunk.chunk_index
        and chunk.content_hash == default_chunk.content_hash
        and chunk.source_uri == default_chunk.source_uri
        and chunk.metadata == default_chunk.metadata
    )


def _truncate_text_to_bytes(text: str, max_bytes: int) -> str:
    if max_bytes <= 0:
        return ""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def _validate_positive_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or type(value) is not int:
        raise ValueError(f"`{field_name}` must be an integer.")
    if value <= 0:
        raise ValueError(f"`{field_name}` must be greater than 0.")


def _knowledge_entry_id(value: str, field_name: str = "entry_id") -> str:
    return _bounded_knowledge_identity(
        value,
        field_name,
        max_bytes=MAX_KNOWLEDGE_ENTRY_ID_BYTES,
    )


def _knowledge_chunk_id(value: str, field_name: str = "chunk_id") -> str:
    return _bounded_knowledge_identity(
        value,
        field_name,
        max_bytes=MAX_KNOWLEDGE_CHUNK_ID_BYTES,
    )


def _bounded_knowledge_identity(value: str, field_name: str, *, max_bytes: int) -> str:
    clean = require_clean_nonblank(value, field_name)
    if len(clean.encode("utf-8")) > max_bytes:
        raise ValueError(f"`{field_name}` must be at most {max_bytes} UTF-8 bytes.")
    return clean


def _validate_knowledge_change_limit(value: int) -> None:
    _validate_positive_int(value, "limit")
    if value > MAX_KNOWLEDGE_CHANGE_LIMIT:
        raise ValueError(f"`limit` must be less than or equal to {MAX_KNOWLEDGE_CHANGE_LIMIT}.")


def _validate_knowledge_revision(value: int, field_name: str) -> None:
    _validate_positive_int(value, field_name)
    if value > MAX_KNOWLEDGE_REVISION:
        raise ValueError(f"`{field_name}` must be at most {MAX_KNOWLEDGE_REVISION}.")


def _validate_knowledge_change_sequence(value: int, field_name: str) -> None:
    _validate_nonnegative_int(value, field_name)
    if value > MAX_KNOWLEDGE_CHANGE_SEQUENCE:
        raise ValueError(f"`{field_name}` must be at most {MAX_KNOWLEDGE_CHANGE_SEQUENCE}.")


def _validate_knowledge_index_sequence(
    value: int,
    field_name: str,
    *,
    allow_zero: bool = True,
) -> None:
    if allow_zero:
        _validate_nonnegative_int(value, field_name)
    else:
        _validate_positive_int(value, field_name)
    if value > MAX_KNOWLEDGE_CHANGE_SEQUENCE:
        raise ValueError(f"`{field_name}` must be at most {MAX_KNOWLEDGE_CHANGE_SEQUENCE}.")


def _validate_knowledge_index_readiness_limit(value: int) -> None:
    _validate_positive_int(value, "limit")
    if value > MAX_KNOWLEDGE_INDEX_READINESS_LIMIT:
        raise ValueError(
            f"`limit` must be less than or equal to {MAX_KNOWLEDGE_INDEX_READINESS_LIMIT}."
        )


def _validate_knowledge_embedding_work_record_limit(
    value: int,
    *,
    field_name: str = "record_limit",
) -> None:
    _validate_positive_int(value, field_name)
    if value > MAX_KNOWLEDGE_EMBEDDING_WORK_RECORD_LIMIT:
        raise ValueError(
            f"`{field_name}` must be less than or equal to "
            f"{MAX_KNOWLEDGE_EMBEDDING_WORK_RECORD_LIMIT}."
        )


def _knowledge_change_now(value: datetime | None) -> datetime:
    result = datetime.now(UTC) if value is None else value
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("`now` must be timezone-aware.")
    return result.astimezone(UTC)


def _knowledge_change_lease_seconds(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("`lease_seconds` must be a number.")
    result = require_finite(float(value), "lease_seconds")
    if result <= 0.0 or result > 86_400.0:
        raise ValueError("`lease_seconds` must be greater than 0 and at most 86400.")
    return result


def _knowledge_change_identity(value: str, field_name: str) -> str:
    clean = require_clean_nonblank(value, field_name)
    if len(clean.encode("utf-8")) > 256:
        raise ValueError(f"`{field_name}` must be at most 256 UTF-8 bytes.")
    return clean


def _bounded_knowledge_index_identity(value: str, field_name: str) -> str:
    return _bounded_knowledge_identity(value, field_name, max_bytes=256)


def _knowledge_embedding_identity_sha256(identity: KnowledgeEmbeddingIdentity) -> str:
    identity = copy_knowledge_embedding_identity(identity)
    return sha256(
        canonical_durable_json_bytes(
            identity.model_dump(mode="json"),
            "knowledge embedding identity",
        )
    ).hexdigest()


def _knowledge_embedding_vector_sha256(vector: list[float]) -> str:
    """Fingerprint the validated vector payload before backend representation casts."""

    return sha256(
        canonical_durable_json_bytes(
            [0.0 if component == 0.0 else component for component in vector],
            "knowledge embedding vector",
        )
    ).hexdigest()


def _bounded_knowledge_embedding_backfill_cursor(value: str, field_name: str) -> str:
    value = require_clean_nonblank(value, field_name)
    if len(value.encode("utf-8")) > _MAX_KNOWLEDGE_EMBEDDING_BACKFILL_CURSOR_BYTES:
        raise ValueError(
            f"`{field_name}` must be at most "
            f"{_MAX_KNOWLEDGE_EMBEDDING_BACKFILL_CURSOR_BYTES} UTF-8 bytes."
        )
    return value


def _knowledge_embedding_backfill_fingerprint(
    query: KnowledgeListQuery,
    access_scope: KnowledgeAccessScope,
    *,
    refresh_existing: bool,
    embedding_model: str,
    embedding_dimensions: int,
) -> str:
    query = copy_knowledge_list_query(query)
    access_scope = copy_knowledge_access_scope(access_scope)
    material = {
        "query": query.model_dump(mode="json"),
        "access_scope_sha256": _knowledge_access_scope_sha256(access_scope),
        "refresh_existing": refresh_existing,
        "projection_type": KNOWLEDGE_CHUNK_TEXT_PROJECTION,
        "embedding_model": require_clean_nonblank(embedding_model, "embedding_model"),
        "dimensions": embedding_dimensions,
        "preprocessing_version": KNOWLEDGE_CHUNK_TEXT_PREPROCESSING_VERSION,
        "generator": KNOWLEDGE_CHUNK_TEXT_GENERATOR,
        "generator_version": KNOWLEDGE_CHUNK_TEXT_GENERATOR_VERSION,
        "index_representation_version": KNOWLEDGE_VECTOR_INDEX_REPRESENTATION_VERSION,
    }
    return sha256(
        canonical_durable_json_bytes(material, "knowledge embedding backfill query")
    ).hexdigest()


def _encode_knowledge_embedding_backfill_cursor(
    *,
    fingerprint: str,
    importance: float,
    updated_at: datetime,
    chunk: KnowledgeChunk,
) -> str:
    cursor = _KnowledgeEmbeddingBackfillCursor(
        version=_KNOWLEDGE_EMBEDDING_BACKFILL_CURSOR_VERSION,
        fingerprint=fingerprint,
        importance=importance,
        updated_at=updated_at,
        entry_id=chunk.entry_id,
        chunk_index=chunk.chunk_index,
        chunk_id=chunk.id,
    )
    raw = canonical_durable_json_bytes(
        cursor.model_dump(mode="json"),
        "knowledge embedding backfill cursor",
    )
    encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return _bounded_knowledge_embedding_backfill_cursor(encoded, "next_cursor")


def _decode_knowledge_embedding_backfill_cursor(
    cursor: str | None,
    *,
    fingerprint: str,
) -> _KnowledgeEmbeddingBackfillCursor | None:
    if cursor is None:
        return None
    cursor = _bounded_knowledge_embedding_backfill_cursor(cursor, "cursor")
    try:
        encoded = cursor.encode("ascii")
        padding = b"=" * (-len(encoded) % 4)
        raw = base64.b64decode(encoded + padding, altchars=b"-_", validate=True)
        if base64.urlsafe_b64encode(raw).rstrip(b"=") != encoded:
            raise ValueError("Non-canonical backfill cursor encoding.")
        decoded = json.loads(raw.decode("utf-8"))
        parsed = _KnowledgeEmbeddingBackfillCursor.model_validate(decoded)
    except (
        binascii.Error,
        json.JSONDecodeError,
        TypeError,
        UnicodeError,
        ValueError,
    ) as exc:
        raise ValueError("Invalid knowledge embedding backfill cursor.") from exc
    if parsed.fingerprint != fingerprint:
        raise ValueError(
            "Knowledge embedding backfill cursor does not match this query, scope, "
            "projection configuration, and refresh mode."
        )
    return parsed


def _knowledge_embedding_backfill_sort_key(
    *,
    importance: float,
    updated_at: datetime,
    entry_id: str,
    chunk_index: int,
    chunk_id: str,
) -> tuple[float, int, str, int, str]:
    updated_at = updated_at.astimezone(UTC)
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    delta = updated_at - epoch
    updated_at_microseconds = (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds
    return (
        -importance,
        -updated_at_microseconds,
        entry_id,
        chunk_index,
        chunk_id,
    )


def _knowledge_index_readiness_update_sha256(
    update: KnowledgeIndexReadinessUpdate,
) -> str:
    update = copy_knowledge_index_readiness_update(update)
    return sha256(
        canonical_durable_json_bytes(
            update.model_dump(mode="json"),
            "knowledge index readiness update",
        )
    ).hexdigest()


def _validate_knowledge_index_readiness_transition(
    current: KnowledgeIndexReadiness | None,
    update: KnowledgeIndexReadinessUpdate,
    *,
    expected_sequence: int | None,
) -> None:
    if current is None:
        if expected_sequence is not None:
            raise KnowledgeIndexReadinessConflict("unknown_expected_sequence")
        if update.state is not KnowledgeIndexState.PENDING:
            raise KnowledgeIndexReadinessConflict("initial_state_must_be_pending")
        return
    if expected_sequence != current.sequence:
        raise KnowledgeIndexReadinessConflict("stale_sequence")
    if update.state is KnowledgeIndexState.PENDING:
        if update.attempt_id == current.attempt_id:
            raise KnowledgeIndexReadinessConflict("attempt_reuse")
        return
    if current.state is not KnowledgeIndexState.PENDING:
        raise KnowledgeIndexReadinessConflict("terminal_state_requires_new_attempt")
    if update.attempt_id != current.attempt_id:
        raise KnowledgeIndexReadinessConflict("stale_attempt")


def _initialize_knowledge_change_consumer_state(
    state: KnowledgeChangeConsumerState | None,
    *,
    consumer_id: str,
    access_scope_sha256: str,
    baseline_sequence: int,
    now: datetime,
) -> KnowledgeChangeConsumerState:
    if state is None:
        return KnowledgeChangeConsumerState(
            consumer_id=consumer_id,
            access_scope_sha256=access_scope_sha256,
            cursor_sequence=baseline_sequence,
            updated_at=now,
        )
    if state.access_scope_sha256 != access_scope_sha256:
        raise KnowledgeChangeConsumerConflict("access_scope_mismatch")
    if state.pending_change_sequence is not None:
        raise KnowledgeChangeConsumerConflict("consumer_has_active_claim")
    if state.cursor_sequence >= baseline_sequence:
        return copy_knowledge_change_consumer_state(state)
    if (
        state.cursor_sequence != 0
        or state.pending_attempt != 0
        or state.last_acknowledged_claim_id is not None
    ):
        raise KnowledgeChangeConsumerConflict("consumer_already_started")
    return state.model_copy(
        update={
            "cursor_sequence": baseline_sequence,
            "updated_at": now,
        }
    )


def _validate_nonnegative_float(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"`{field_name}` must be a number.")
    value = require_finite(float(value), field_name)
    if value < 0.0:
        raise ValueError(f"`{field_name}` must be greater than or equal to 0.")
    return value


def _validate_unit_float(value: float, field_name: str) -> float:
    value = _validate_nonnegative_float(value, field_name)
    if value > 1.0:
        raise ValueError(f"`{field_name}` must be between 0.0 and 1.0.")
    return value


def _validate_nonnegative_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or type(value) is not int:
        raise ValueError(f"`{field_name}` must be an integer.")
    if value < 0:
        raise ValueError(f"`{field_name}` must be greater than or equal to 0.")


def _dedupe_strings(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
