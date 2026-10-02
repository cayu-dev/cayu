"""Backend-independent knowledge records, identities, and detached copies."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_json_object,
    copy_durable_json_value,
    copy_durable_metadata,
    copy_label_map,
    require_finite,
)
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu._validation import require_durable_nonblank as require_nonblank

DEFAULT_KNOWLEDGE_NAMESPACE = "default"
DEFAULT_KNOWLEDGE_KIND = "fact"
DEFAULT_KNOWLEDGE_LIMIT = 10
DEFAULT_KNOWLEDGE_MAX_BYTES = 20_000
MAX_KNOWLEDGE_CHUNK_ID_BYTES = 512
MAX_KNOWLEDGE_CHUNK_INDEX = 2**31 - 1
MAX_KNOWLEDGE_ENTRY_ID_BYTES = 256
MAX_KNOWLEDGE_ENTRY_PAYLOAD_BYTES = 2**31 - 1
MAX_KNOWLEDGE_REVISION = 2**31 - 1
MAX_KNOWLEDGE_REVISION_SEARCH_REFS = 250
MAX_KNOWLEDGE_EVIDENCE_BYTES = DEFAULT_KNOWLEDGE_MAX_BYTES
MAX_KNOWLEDGE_EVIDENCE_JSON_BYTES = 16_384
MAX_KNOWLEDGE_ACTIVATION_IDENTITY_BYTES = 256
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


class KnowledgeStatus(StrEnum):
    ACTIVE = "active"
    PENDING = "pending"
    ARCHIVED = "archived"
    DELETED = "deleted"


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


def _knowledge_activation_identity(value: object, field_name: str) -> str:
    if type(value) is not str:
        raise ValueError(f"`{field_name}` must be a string.")
    value = require_clean_nonblank(value, field_name)
    if len(value.encode("utf-8")) > MAX_KNOWLEDGE_ACTIVATION_IDENTITY_BYTES:
        raise ValueError(
            f"`{field_name}` must be at most {MAX_KNOWLEDGE_ACTIVATION_IDENTITY_BYTES} UTF-8 bytes."
        )
    return value


class KnowledgeEvidenceRole(StrEnum):
    ORIGIN = "origin"
    SUPPORTING = "supporting"


class KnowledgeEvidenceDisposition(StrEnum):
    LIVE = "live"
    DETACHED = "detached"
    RETAINED = "retained"


class KnowledgeEntryReadLimitExceeded(ValueError):
    """An authorized entry exceeds a caller-owned read byte ceiling."""

    def __init__(
        self,
        entry_id: str,
        *,
        revision: int,
        payload_bytes: int,
        max_bytes: int,
    ) -> None:
        self.entry_id = _knowledge_entry_id(entry_id)
        _validate_knowledge_revision(revision, "revision")
        _validate_positive_int(payload_bytes, "payload_bytes")
        _validate_positive_int(max_bytes, "max_bytes")
        if payload_bytes <= max_bytes:
            raise ValueError("payload_bytes must exceed max_bytes.")
        self.revision = revision
        self.payload_bytes = payload_bytes
        self.max_bytes = max_bytes
        super().__init__("Knowledge entry exceeds the configured read byte limit.")


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
        return copy_durable_metadata(value, "metadata")

    @field_validator("labels", mode="before")
    @classmethod
    def copy_labels(cls, value) -> dict[str, str]:
        return copy_label_map(value, "labels")

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _knowledge_entry_id(value, "id")

    @field_validator("namespace", "kind")
    @classmethod
    def validate_clean_nonblank_fields(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("created_by", mode="before")
    @classmethod
    def validate_created_by(cls, value: object) -> str:
        return _knowledge_activation_identity(value, "created_by")

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
        return copy_durable_metadata(value, "metadata")

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


class KnowledgeRevisionRef(BaseModel):
    """Stable reference to one immutable revision of one logical entry."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    entry_id: str
    revision: int

    @field_validator("entry_id")
    @classmethod
    def validate_entry_id(cls, value: str) -> str:
        return _knowledge_entry_id(value)

    @field_validator("revision")
    @classmethod
    def validate_revision(cls, value: int) -> int:
        _validate_knowledge_revision(value, "revision")
        return value


def copy_knowledge_revision_refs(
    value: Sequence[KnowledgeRevisionRef],
) -> tuple[KnowledgeRevisionRef, ...]:
    """Copy, bound, deduplicate, and canonically order exact revision references."""

    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise TypeError("revision_refs must be a sequence of KnowledgeRevisionRef instances.")
    if len(value) > MAX_KNOWLEDGE_REVISION_SEARCH_REFS:
        raise ValueError(
            "revision_refs cannot contain more than "
            f"{MAX_KNOWLEDGE_REVISION_SEARCH_REFS} references."
        )
    copied: dict[tuple[str, int], KnowledgeRevisionRef] = {}
    for item in value:
        if type(item) is not KnowledgeRevisionRef:
            raise TypeError("revision_refs must contain KnowledgeRevisionRef instances.")
        reference = item.model_copy(deep=True)
        copied[(reference.entry_id, reference.revision)] = reference
    return tuple(copied[key] for key in sorted(copied))


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
        metadata=copy_durable_metadata(entry.metadata, "metadata"),
    )


def knowledge_entry_payload_bytes(entry: KnowledgeEntry) -> int:
    """Return the backend-stable canonical byte size of one entry payload."""

    if type(entry) is not KnowledgeEntry:
        raise TypeError("KnowledgeEntry instances must not be subclasses.")
    payload_bytes = len(
        canonical_durable_json_bytes(
            entry.model_dump(mode="json"),
            "knowledge entry payload",
        )
    )
    if payload_bytes > MAX_KNOWLEDGE_ENTRY_PAYLOAD_BYTES:
        raise ValueError(
            "Knowledge entry payload must be at most "
            f"{MAX_KNOWLEDGE_ENTRY_PAYLOAD_BYTES} canonical UTF-8 bytes."
        )
    return payload_bytes


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
        metadata=copy_durable_metadata(chunk.metadata, "metadata"),
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
        metadata=copy_durable_metadata(evidence.metadata, "metadata"),
    )


def copy_knowledge_revision_ref(reference: KnowledgeRevisionRef) -> KnowledgeRevisionRef:
    if type(reference) is not KnowledgeRevisionRef:
        raise TypeError("KnowledgeRevisionRef instances must not be subclasses.")
    return KnowledgeRevisionRef(entry_id=reference.entry_id, revision=reference.revision)


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


def _validate_knowledge_revision(value: int, field_name: str) -> None:
    _validate_positive_int(value, field_name)
    if value > MAX_KNOWLEDGE_REVISION:
        raise ValueError(f"`{field_name}` must be at most {MAX_KNOWLEDGE_REVISION}.")


def _validate_nonnegative_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or type(value) is not int:
        raise ValueError(f"`{field_name}` must be an integer.")
    if value < 0:
        raise ValueError(f"`{field_name}` must be greater than or equal to 0.")


def _dedupe_strings(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _knowledge_publication_operation_id(operation_id: str) -> str:
    clean = require_clean_nonblank(operation_id, "operation_id")
    if len(clean.encode("utf-8")) > 256:
        raise ValueError("`operation_id` must be at most 256 UTF-8 bytes.")
    return clean


def _next_knowledge_revision(expected_revision: int) -> int:
    _validate_knowledge_revision(expected_revision, "expected_revision")
    if expected_revision == MAX_KNOWLEDGE_REVISION:
        raise ValueError(f"Knowledge revision cannot advance beyond {MAX_KNOWLEDGE_REVISION}.")
    return expected_revision + 1


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
